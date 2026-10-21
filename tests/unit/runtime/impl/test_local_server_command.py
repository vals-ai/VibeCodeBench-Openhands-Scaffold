from unittest.mock import MagicMock, patch

import pytest

from openhands.core.config import OpenHandsConfig
from openhands.runtime.impl.local import local_runtime
from openhands.runtime.utils.command import get_action_execution_server_startup_command


@pytest.mark.parametrize('platform', ['win32', 'linux'])
def test_local_server_launch_preserves_target_platform(platform, tmp_path):
    config = OpenHandsConfig()
    config.workspace_mount_path_in_sandbox = str(tmp_path / 'workspace with spaces')
    executable = str(tmp_path / 'Python Tools' / 'python.exe')
    process = MagicMock()
    with (
        patch.object(local_runtime.sys, 'platform', platform),
        patch.object(local_runtime.sys, 'executable', executable),
        patch.object(local_runtime.tempfile, 'mkdtemp', return_value=str(tmp_path)),
        patch.object(local_runtime, 'find_available_tcp_port', return_value=12345),
        patch.object(local_runtime, 'get_user_info', return_value=(1000, 'openhands')),
        patch.object(local_runtime.threading, 'Thread'),
        patch.object(local_runtime.subprocess, 'Popen', return_value=process) as popen,
    ):
        info, _ = local_runtime._create_server(config, [], 'test')
    command = popen.call_args.args[0]
    assert info.process is process
    if platform == 'win32':
        assert command[0] == executable
        assert (
            command[command.index('--working-dir') + 1]
            == config.workspace_mount_path_in_sandbox
        )
        assert command[1:4] == ['-u', '-m', 'openhands.runtime.action_execution_server']
    else:
        assert command[:2] == ['bash', '-c']
        assert 'docker info' in command[2]


def test_container_target_keeps_linux_bootstrap_on_windows_host():
    with patch('sys.platform', 'win32'):
        command = get_action_execution_server_startup_command(
            server_port=8000, plugins=[], app_config=OpenHandsConfig()
        )
    assert command[:2] == ['bash', '-c']
    assert 'dockerd' in command[2]
    assert '/openhands/micromamba/bin/micromamba' in command[2]
