import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from model_library.base import QueryResult, TextInput

from openhands.core.config import LLMConfig
from openhands.llm.llm import LLM


@pytest.fixture
def model_and_llm(monkeypatch):
    model = MagicMock()
    model.reasoning = False
    model.query = AsyncMock(
        return_value=QueryResult(
            output_text='Done', extras={'response_id': 'response-123'}
        )
    )
    monkeypatch.setattr('openhands.llm.llm.fetch_registry_model', lambda _: model)
    runner = MagicMock()
    runner.run.side_effect = lambda coroutine, timeout: asyncio.run(coroutine)
    monkeypatch.setattr('openhands.llm.llm._persistent_loop', runner)
    return model, LLM(LLMConfig(model='openai/gpt-4o', timeout=30), 'test-service')


def test_completion_forwards_inputs_and_returns_query_result(model_and_llm):
    model, llm = model_and_llm
    task = TextInput(text='Inspect the files')
    previous = TextInput(text='Previous task')
    response = llm.completion(input=[task], history=[previous])
    model.query.assert_awaited_once_with(input=[task], history=[previous], tools=[])
    assert response.output_text == 'Done'
    assert len(llm.metrics.token_usages) == 1
    assert llm.metrics.token_usages[0].response_id == 'response-123'


def test_model_errors_propagate_without_recording_success(model_and_llm):
    model, llm = model_and_llm
    error = RuntimeError('provider unavailable')
    model.query.side_effect = error
    with pytest.raises(RuntimeError) as raised:
        llm.completion(input=[TextInput(text='Task')])
    assert raised.value is error
    model.query.assert_awaited_once()
    assert llm.metrics.token_usages == []


@pytest.mark.parametrize(
    'supports_images,disabled,expected',
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_vision_uses_model_capability_and_configuration(
    model_and_llm, supports_images, disabled, expected
):
    model, llm = model_and_llm
    model.supports_images = supports_images
    llm.config.disable_vision = disabled
    assert llm.vision_is_active() is expected
