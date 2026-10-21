from unittest.mock import MagicMock, patch

import httpx
import pytest

from openhands.core.config import LLMConfig
from openhands.resolver.interfaces.github import GithubPRHandler
from openhands.resolver.interfaces.issue_definitions import (
    ServiceContextPR,
)


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    # suppress logging of completion data to file
    mock_logger = MagicMock()
    monkeypatch.setattr('openhands.llm.debug_mixin.llm_prompt_logger', mock_logger)
    monkeypatch.setattr('openhands.llm.debug_mixin.llm_response_logger', mock_logger)
    return mock_logger


def test_handle_nonexistent_issue_reference():
    llm_config = LLMConfig(model='test', api_key='test')
    handler = ServiceContextPR(
        GithubPRHandler('test-owner', 'test-repo', 'test-token'), llm_config
    )

    # Mock the requests.get to simulate a 404 error
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPError(
        '404 Client Error: Not Found'
    )

    with patch('httpx.get', return_value=mock_response):
        # Call the method with a non-existent issue reference
        result = handler._strategy.get_context_from_external_issues_references(
            closing_issues=[],
            closing_issue_numbers=[],
            issue_body='This references #999999',  # Non-existent issue
            review_comments=[],
            review_threads=[],
            thread_comments=None,
        )

        # The method should return an empty list since the referenced issue couldn't be fetched
        assert result == []


def test_handle_rate_limit_error():
    llm_config = LLMConfig(model='test', api_key='test')
    handler = ServiceContextPR(
        GithubPRHandler('test-owner', 'test-repo', 'test-token'), llm_config
    )

    # Mock the requests.get to simulate a rate limit error
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPError(
        '403 Client Error: Rate Limit Exceeded'
    )

    with patch('httpx.get', return_value=mock_response):
        # Call the method with an issue reference
        result = handler._strategy.get_context_from_external_issues_references(
            closing_issues=[],
            closing_issue_numbers=[],
            issue_body='This references #123',
            review_comments=[],
            review_threads=[],
            thread_comments=None,
        )

        # The method should return an empty list since the request was rate limited
        assert result == []


def test_handle_network_error():
    llm_config = LLMConfig(model='test', api_key='test')
    handler = ServiceContextPR(
        GithubPRHandler('test-owner', 'test-repo', 'test-token'), llm_config
    )

    # Mock the requests.get to simulate a network error
    with patch('httpx.get', side_effect=httpx.NetworkError('Network Error')):
        # Call the method with an issue reference
        result = handler._strategy.get_context_from_external_issues_references(
            closing_issues=[],
            closing_issue_numbers=[],
            issue_body='This references #123',
            review_comments=[],
            review_threads=[],
            thread_comments=None,
        )

        # The method should return an empty list since the network request failed
        assert result == []


def test_successful_issue_reference():
    llm_config = LLMConfig(model='test', api_key='test')
    handler = ServiceContextPR(
        GithubPRHandler('test-owner', 'test-repo', 'test-token'), llm_config
    )

    # Mock a successful response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {'body': 'This is the referenced issue body'}

    with patch('httpx.get', return_value=mock_response):
        # Call the method with an issue reference
        result = handler._strategy.get_context_from_external_issues_references(
            closing_issues=[],
            closing_issue_numbers=[],
            issue_body='This references #123',
            review_comments=[],
            review_threads=[],
            thread_comments=None,
        )

        # The method should return a list with the referenced issue body
        assert result == ['This is the referenced issue body']
