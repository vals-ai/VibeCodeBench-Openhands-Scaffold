import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openhands.runtime.plugins.jupyter import JupyterPlugin
from openhands.runtime.plugins.jupyter.execute_server import JupyterKernel


@pytest.mark.asyncio
async def test_gateway_starts_in_own_process_group(monkeypatch, tmp_path) -> None:
    class Output:
        lines = [
            b'python-dotenv could not parse statement starting at line 1\n',
            b'Jupyter Kernel Gateway at http://127.0.0.1\n',
        ]

        async def readline(self) -> bytes:
            return self.lines.pop(0)

    process = SimpleNamespace(stdout=Output())
    start_new_session = None

    async def create_subprocess_shell(*_args, **kwargs):
        nonlocal start_new_session
        start_new_session = kwargs['start_new_session']
        return process

    monkeypatch.setenv('LOCAL_RUNTIME_MODE', '1')
    monkeypatch.setenv('OPENHANDS_REPO_PATH', str(tmp_path))
    monkeypatch.setattr(
        'openhands.runtime.plugins.jupyter.find_available_tcp_port',
        lambda *_args, **_kwargs: 41000,
    )
    monkeypatch.setattr(asyncio, 'create_subprocess_shell', create_subprocess_shell)

    plugin = JupyterPlugin()
    plugin.run = AsyncMock(return_value=SimpleNamespace(content='/usr/bin/python'))
    await plugin.initialize('openhands')

    assert start_new_session is True
    assert not process.stdout.lines


@pytest.mark.asyncio
async def test_gateway_cleanup_kills_process_group() -> None:
    process = await asyncio.create_subprocess_shell(
        "bash -c 'sleep 60 & wait'",
        start_new_session=True,
    )
    plugin = JupyterPlugin()
    plugin.gateway_process = process
    kernel = AsyncMock()
    plugin.kernel = kernel

    await asyncio.wait_for(plugin._cleanup_gateway_process(), timeout=10)
    await plugin._cleanup_gateway_process()

    kernel.shutdown_async.assert_awaited_once()
    assert process.returncode is not None


@pytest.mark.asyncio
async def test_kernel_cleanup_does_not_require_gateway_process() -> None:
    plugin = JupyterPlugin()
    kernel = AsyncMock()
    plugin.kernel = kernel

    await plugin._cleanup_gateway_process()

    kernel.shutdown_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_gateway_cleanup_kills_detached_kernel_after_shutdown_failure() -> None:
    process = await asyncio.create_subprocess_shell(
        "bash -c 'setsid sleep 60 & echo $!; wait'",
        stdout=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    kernel_pid = int(await process.stdout.readline())
    plugin = JupyterPlugin()
    plugin.gateway_process = process
    kernel = AsyncMock()
    kernel.shutdown_async.side_effect = TimeoutError()
    plugin.kernel = kernel

    await asyncio.wait_for(plugin._cleanup_gateway_process(), timeout=10)

    with pytest.raises(ProcessLookupError):
        os.kill(kernel_pid, 0)


@pytest.mark.asyncio
async def test_closed_kernel_websocket_fails_immediately() -> None:
    class Stream:
        def closed(self) -> bool:
            return False

    class WebSocket:
        stream = Stream()

        async def write_message(self, _message: str) -> None:
            pass

        async def read_message(self) -> None:
            return None

    kernel = JupyterKernel('localhost:40000', 'test')
    kernel.ws = WebSocket()

    with pytest.raises(ConnectionError, match='websocket closed'):
        await kernel.execute('1 + 1')
