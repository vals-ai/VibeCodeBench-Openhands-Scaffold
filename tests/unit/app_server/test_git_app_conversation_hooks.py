from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openhands.app_server.app_conversation.git_app_conversation_service import (
    PRE_COMMIT_HOOK,
    GitAppConversationService,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'existing_hook,should_preserve',
    [
        (None, False),
        ('#!/bin/sh\necho custom hook', True),
        ('# This hook was installed by OpenHands', False),
    ],
)
async def test_git_hook_setup_awaits_download_and_preserves_custom_hook(
    existing_hook, should_preserve
):
    workspace = MagicMock()
    workspace.working_dir = '/workspace'
    workspace.execute_command = AsyncMock(
        return_value=SimpleNamespace(exit_code=0, stderr='')
    )
    workspace.file_upload = AsyncMock()
    downloaded_paths = []

    async def download(source, destination):
        assert source == PRE_COMMIT_HOOK
        target = Path(destination)
        assert target.is_file()
        downloaded_paths.append(target)
        if existing_hook is not None:
            target.write_text(existing_hook)
        return SimpleNamespace(success=existing_hook is not None)

    workspace.file_download = AsyncMock(side_effect=download)
    await GitAppConversationService.maybe_setup_git_hooks(MagicMock(), workspace)

    workspace.file_download.assert_awaited_once()
    commands = [call.args[0] for call in workspace.execute_command.await_args_list]
    assert (
        any(command.startswith('mv .git/hooks/pre-commit ') for command in commands)
        is should_preserve
    )
    workspace.file_upload.assert_awaited_once()
    assert (
        workspace.file_upload.await_args.kwargs['destination_path'] == PRE_COMMIT_HOOK
    )
    assert commands[-1] == 'chmod +x .git/hooks/pre-commit'
    assert not downloaded_paths[0].exists()
