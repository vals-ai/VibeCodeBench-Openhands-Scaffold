import asyncio
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass

import psutil

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action, IPythonRunCellAction
from openhands.events.observation import IPythonRunCellObservation
from openhands.runtime.plugins.jupyter.execute_server import JupyterKernel
from openhands.runtime.plugins.requirement import Plugin, PluginRequirement
from openhands.runtime.utils import find_available_tcp_port
from openhands.utils.shutdown_listener import should_continue

SU_TO_USER = os.getenv('SU_TO_USER', 'true').lower() in (
    '1',
    'true',
    't',
    'yes',
    'y',
    'on',
)


@dataclass
class JupyterRequirement(PluginRequirement):
    name: str = 'jupyter'


class JupyterPlugin(Plugin):
    name: str = 'jupyter'
    kernel_gateway_port: int
    kernel_id: str
    gateway_process: asyncio.subprocess.Process | subprocess.Popen
    python_interpreter_path: str
    kernel_gateway_ready_timeout_seconds: int = 60
    initial_ipython_handshake_timeout_seconds: int = 60

    async def _cleanup_gateway_process(self) -> None:
        kernel = getattr(self, 'kernel', None)
        if kernel is not None:
            del self.kernel
            try:
                await asyncio.wait_for(kernel.shutdown_async(), timeout=5)
            except Exception as exc:
                logger.warning(f'Failed to shut down jupyter kernel: {exc}')

        process = getattr(self, 'gateway_process', None)
        if process is None:
            return
        del self.gateway_process

        try:
            if isinstance(process, asyncio.subprocess.Process):
                descendants: list[psutil.Process] = []
                with suppress(psutil.Error):
                    descendants = psutil.Process(process.pid).children(recursive=True)
                process_group = process.pid
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process_group, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        pass
                with suppress(ProcessLookupError):
                    os.killpg(process_group, signal.SIGKILL)
                for descendant in descendants:
                    with suppress(psutil.Error):
                        descendant.kill()
                if process.returncode is None:
                    await process.wait()
                if descendants:
                    psutil.wait_procs(descendants, timeout=1)
            else:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
        except Exception as exc:
            logger.warning(f'Failed to clean up jupyter kernel gateway process: {exc}')

    async def initialize(
        self, username: str, kernel_id: str = 'openhands-default'
    ) -> None:
        self.kernel_gateway_port = find_available_tcp_port(40000, 49999)
        self.kernel_id = kernel_id
        is_local_runtime = os.environ.get('LOCAL_RUNTIME_MODE') == '1'
        is_windows = sys.platform == 'win32'

        if not is_local_runtime:
            # Non-LocalRuntime
            prefix = f'su - {username} -s ' if SU_TO_USER else ''
            # cd to code repo, setup all env vars and run micromamba
            poetry_prefix = (
                'cd /openhands/code\n'
                'export POETRY_VIRTUALENVS_PATH=/openhands/poetry;\n'
                'export PYTHONPATH=/openhands/code:$PYTHONPATH;\n'
                'export MAMBA_ROOT_PREFIX=/openhands/micromamba;\n'
                '/openhands/micromamba/bin/micromamba run -n openhands '
            )
        else:
            # LocalRuntime
            prefix = ''
            code_repo_path = os.environ.get('OPENHANDS_REPO_PATH')
            if not code_repo_path:
                raise ValueError(
                    'OPENHANDS_REPO_PATH environment variable is not set. '
                    'This is required for the jupyter plugin to work with LocalRuntime.'
                )
            # The correct environment is ensured by the PATH in LocalRuntime.
            poetry_prefix = f'cd {code_repo_path}\n'

        if is_windows:
            # Windows-specific command format
            jupyter_launch_command = (
                f'cd /d "{code_repo_path}" && '
                f'"{sys.executable}" -m jupyter kernelgateway '
                '--KernelGatewayApp.ip=0.0.0.0 '
                f'--KernelGatewayApp.port={self.kernel_gateway_port}'
            )
            logger.debug(f'Jupyter launch command (Windows): {jupyter_launch_command}')

            # Using synchronous subprocess.Popen for Windows as asyncio.create_subprocess_shell
            # has limitations on Windows platforms
            self.gateway_process = subprocess.Popen(  # type: ignore[ASYNC101]
                jupyter_launch_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                text=True,
            )

            # Windows-specific stdout handling with synchronous time.sleep
            # as asyncio has limitations on Windows for subprocess operations
            output = ''
            ready = False
            while should_continue():
                if self.gateway_process.stdout is None:
                    time.sleep(1)  # type: ignore[ASYNC101]
                    continue

                line = self.gateway_process.stdout.readline()
                if not line:
                    time.sleep(1)  # type: ignore[ASYNC101]
                    continue

                output += line
                if 'Jupyter Kernel Gateway' in line and 'http' in line:
                    ready = True
                    break

                time.sleep(1)  # type: ignore[ASYNC101]
                logger.debug('Waiting for jupyter kernel gateway to start...')

            if not ready:
                await self._cleanup_gateway_process()
                raise RuntimeError(
                    'Shutdown requested before jupyter kernel gateway became ready. '
                    f'Output so far: {output}'
                )

            logger.debug(
                f'Jupyter kernel gateway started at port {self.kernel_gateway_port}. Output: {output}'
            )
        else:
            # Unix systems (Linux/macOS)
            jupyter_launch_command = (
                f"{prefix}/bin/bash << 'EOF'\n"
                f'{poetry_prefix}'
                f'"{sys.executable}" -m jupyter kernelgateway '
                '--KernelGatewayApp.ip=0.0.0.0 '
                f'--KernelGatewayApp.port={self.kernel_gateway_port}\n'
                'EOF'
            )
            logger.debug(f'Jupyter launch command: {jupyter_launch_command}')

            # Using asyncio.create_subprocess_shell instead of subprocess.Popen
            # to avoid ASYNC101 linting error
            self.gateway_process = await asyncio.create_subprocess_shell(
                jupyter_launch_command,
                stderr=asyncio.subprocess.STDOUT,
                stdout=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            # read stdout until the kernel gateway is ready
            output = ''
            ready = False
            start_time = time.monotonic()
            while should_continue() and self.gateway_process.stdout is not None:
                elapsed = time.monotonic() - start_time
                remaining = self.kernel_gateway_ready_timeout_seconds - elapsed
                if remaining <= 0:
                    await self._cleanup_gateway_process()
                    raise TimeoutError(
                        'Timed out waiting for jupyter kernel gateway to become ready '
                        f'after {self.kernel_gateway_ready_timeout_seconds}s. '
                        f'Output so far: {output}'
                    )

                try:
                    line_bytes = await asyncio.wait_for(
                        self.gateway_process.stdout.readline(),
                        timeout=min(1.0, remaining),
                    )
                except asyncio.TimeoutError:
                    if self.gateway_process.returncode is not None:
                        await self._cleanup_gateway_process()
                        raise RuntimeError(
                            'Jupyter kernel gateway exited before becoming ready. '
                            f'Output so far: {output}'
                        )
                    logger.debug('Waiting for jupyter kernel gateway to start...')
                    continue

                if not line_bytes:
                    if self.gateway_process.returncode is not None:
                        await self._cleanup_gateway_process()
                        raise RuntimeError(
                            'Jupyter kernel gateway exited before becoming ready. '
                            f'Output so far: {output}'
                        )
                    await asyncio.sleep(0.1)
                    continue

                line = line_bytes.decode('utf-8')
                output += line
                if 'Jupyter Kernel Gateway' in line and 'http' in line:
                    ready = True
                    break
                logger.debug('Waiting for jupyter kernel gateway to start...')

            if not ready:
                await self._cleanup_gateway_process()
                raise RuntimeError(
                    'Shutdown requested before jupyter kernel gateway became ready. '
                    f'Output so far: {output}'
                )

            logger.debug(
                f'Jupyter kernel gateway started at port {self.kernel_gateway_port}. Output: {output}'
            )

        try:
            _obs = await asyncio.wait_for(
                self.run(IPythonRunCellAction(code='import sys; print(sys.executable)')),
                timeout=self.initial_ipython_handshake_timeout_seconds,
            )
        except Exception as exc:
            await self._cleanup_gateway_process()
            raise RuntimeError(
                'Initial IPython handshake failed after kernel gateway startup'
            ) from exc

        self.python_interpreter_path = _obs.content.strip()

    async def _run(self, action: Action) -> IPythonRunCellObservation:
        """Internal method to run a code cell in the jupyter kernel."""
        if not isinstance(action, IPythonRunCellAction):
            raise ValueError(
                f'Jupyter plugin only supports IPythonRunCellAction, but got {action}'
            )

        if not hasattr(self, 'kernel'):
            self.kernel = JupyterKernel(
                f'localhost:{self.kernel_gateway_port}', self.kernel_id
            )

        if not self.kernel.initialized:
            await self.kernel.initialize()

        # Execute the code and get structured output
        output = await self.kernel.execute(action.code, timeout=action.timeout)

        # Extract text content and image URLs from the structured output
        text_content = output.get('text', '')
        image_urls = output.get('images', [])

        return IPythonRunCellObservation(
            content=text_content,
            code=action.code,
            image_urls=image_urls if image_urls else None,
        )

    async def run(self, action: Action) -> IPythonRunCellObservation:
        obs = await self._run(action)
        return obs
