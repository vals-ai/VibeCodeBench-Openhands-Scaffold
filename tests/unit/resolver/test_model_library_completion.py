from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from model_library.base import QueryResult, TextInput

from openhands.core.config import LLMConfig
from openhands.events.action import MessageAction
from openhands.llm import LLM
from openhands.resolver.interfaces.github import GithubIssueHandler, GithubPRHandler
from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.interfaces.issue_definitions import (
    ServiceContextIssue,
    ServiceContextPR,
)

REAL_LLM_INIT = LLM.__init__


@pytest.mark.parametrize('kind', ['issue', 'pr'])
def test_success_check_reaches_model_query_with_typed_prompt(monkeypatch, kind):
    query = AsyncMock(
        return_value=QueryResult(
            output_text='--- success\ntrue\n--- explanation\nVerified fix'
        )
    )
    monkeypatch.setattr(LLM, '__init__', REAL_LLM_INIT)
    monkeypatch.setattr(
        'openhands.llm.llm.fetch_registry_model',
        lambda config: SimpleNamespace(query=query, reasoning=False),
    )
    config = LLMConfig(model='test', timeout=30)
    if kind == 'pr':
        context = ServiceContextPR(GithubPRHandler('owner', 'repo', 'token'), config)
        success, explanation = context._check_feedback_with_llm(
            'Review this concrete fix'
        )
        expected_prompt = 'Review this concrete fix'
    else:
        context = ServiceContextIssue(
            GithubIssueHandler('owner', 'repo', 'token'), config
        )
        issue = Issue(
            owner='owner',
            repo='repo',
            number=1,
            title='Fix parser',
            body='Preserve multiline input',
        )
        success, _, explanation = context.guess_success(
            issue, [MessageAction(content='Parser fixed')]
        )
        expected_prompt = 'Preserve multiline input'
    assert success is True
    assert explanation.strip() == 'Verified fix'
    query.assert_awaited_once()
    prompt = query.call_args.kwargs['input'][0]
    assert isinstance(prompt, TextInput)
    assert expected_prompt in prompt.text
