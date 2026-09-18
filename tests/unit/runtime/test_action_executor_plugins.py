from types import SimpleNamespace

import pytest

from openhands.runtime.action_execution_server import ActionExecutor


@pytest.mark.asyncio
async def test_failed_plugin_remains_registered_for_cleanup() -> None:
    class FailingPlugin:
        name = 'failing'

        async def initialize(self, _username: str) -> None:
            raise RuntimeError('initialization failed')

    executor = object.__new__(ActionExecutor)
    executor.bash_session = SimpleNamespace(cwd='/workspace')
    executor.plugins = {}
    executor.username = 'root'
    plugin = FailingPlugin()

    with pytest.raises(RuntimeError, match='initialization failed'):
        await executor._init_plugin(plugin)

    assert executor.plugins == {'failing': plugin}
