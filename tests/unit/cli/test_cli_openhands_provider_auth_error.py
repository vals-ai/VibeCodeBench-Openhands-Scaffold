import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.cli import main as cli
from openhands.controller.state.state import State
from openhands.core.config import LLMConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'model,expect_key_link', [('openhands/o3', True), ('openai/o3', False)]
)
async def test_restored_authentication_error_displays_provider_guidance(
    model, expect_key_link
):
    config = MagicMock()
    config.runtime = 'local'
    config.get_llm_config.return_value = LLMConfig(model=model)
    agent = MagicMock()
    agent.config.enable_mcp = False
    runtime = MagicMock()
    runtime.connect = AsyncMock()
    restored = State(last_error='ERROR_LLM_AUTHENTICATION')

    with ExitStack() as stack:
        for name in (
            'display_runtime_initialization_message',
            'display_initialization_animation',
            'display_banner',
            'clear',
            'display_initial_user_prompt',
        ):
            stack.enter_context(patch.object(cli, name))
        stack.enter_context(
            patch.object(
                cli,
                'create_registry_and_conversation_stats',
                return_value=(MagicMock(), MagicMock(), config),
            )
        )
        stack.enter_context(patch.object(cli, 'create_agent', return_value=agent))
        stack.enter_context(patch.object(cli, 'create_runtime', return_value=runtime))
        stack.enter_context(
            patch.object(cli, 'create_controller', return_value=(MagicMock(), restored))
        )
        stack.enter_context(
            patch.object(cli, 'create_memory', return_value=MagicMock())
        )
        stack.enter_context(
            patch.object(
                cli, 'initialize_repository_for_runtime', return_value='/test/dir'
            )
        )
        run_agent = stack.enter_context(
            patch.object(cli, 'run_agent_until_done', new_callable=AsyncMock)
        )
        cleanup = stack.enter_context(
            patch.object(cli, 'cleanup_session', new_callable=AsyncMock)
        )
        welcome = stack.enter_context(patch.object(cli, 'display_welcome_message'))

        result = await cli.run_session(
            asyncio.get_running_loop(),
            config,
            AsyncMock(),
            '/test/dir',
            task_content='Try again',
        )

    message = welcome.call_args.args[0]
    assert (
        'Authentication error with the LLM provider. Please check your API key.'
        in message
    )
    assert ('https://app.all-hands.dev/settings/api-keys' in message) is expect_key_link
    run_agent.assert_awaited_once()
    cleanup.assert_awaited_once()
    assert result is False
