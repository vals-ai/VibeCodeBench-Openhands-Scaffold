from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from integrations.solvability.models.featurizer import Feature, Featurizer
from integrations.solvability.models.importance_strategy import ImportanceStrategy
from integrations.solvability.models.report import SolvabilityReport
from integrations.solvability.models.summary import SolvabilitySummary
from model_library.base import QueryResult, SystemInput, TextInput, ToolCall
from model_library.base.output.result import QueryResultMetadata

from openhands.core.config import LLMConfig
from openhands.llm import LLM


@pytest.fixture
def model_query(monkeypatch):
    query = AsyncMock()
    model = SimpleNamespace(query=query, reasoning=False)
    monkeypatch.setattr('openhands.llm.llm.fetch_registry_model', lambda config: model)
    return query


def test_featurizer_passes_typed_inputs_and_forced_tool_to_model(model_query):
    model_query.return_value = QueryResult(
        tool_calls=[
            ToolCall(id='features', name='call_featurizer', args={'has_code': True})
        ],
        metadata=QueryResultMetadata(in_tokens=11, out_tokens=7),
    )
    featurizer = Featurizer(
        system_prompt='Extract issue features.',
        message_prefix='Issue: ',
        features=[Feature(identifier='has_code', description='Contains example code.')],
    )
    result = featurizer.embed(
        'print(1)', LLMConfig(model='test', timeout=30), temperature=0.4, samples=2
    )

    assert result.samples == [{'has_code': True}, {'has_code': True}]
    assert (result.prompt_tokens, result.completion_tokens) == (22, 14)
    assert model_query.await_count == 2
    request = model_query.call_args.kwargs
    assert request['input'] == [
        SystemInput(text='Extract issue features.'),
        TextInput(text='Issue: print(1)'),
    ]
    tool = request['tools'][0]
    assert tool.name == 'call_featurizer'
    assert tool.body.properties == {
        'has_code': {'type': 'boolean', 'description': 'Contains example code.'}
    }
    assert request['tool_choice']['function']['name'] == tool.name
    assert request['temperature'] == 0.4


def test_summary_reads_model_library_tool_args_and_usage(model_query):
    model_query.return_value = QueryResult(
        tool_calls=[
            ToolCall(
                id='summary',
                name='solvability_summary',
                args='{"summary":"Clear issue.","actionable_feedback":"Add a test.","positive_feedback":"Includes code."}',
            )
        ],
        metadata=QueryResultMetadata(in_tokens=13, out_tokens=9),
    )
    report = SolvabilityReport(
        identifier='test',
        issue='Fix this code',
        score=0.7,
        prompt_tokens=1,
        completion_tokens=1,
        response_latency=0.1,
        features={'has_code': 1.0},
        samples=1,
        importance_strategy=ImportanceStrategy.IMPURITY,
        feature_importances={'has_code': 1.0},
    )
    result = SolvabilitySummary.from_report(
        report, LLM(LLMConfig(model='test', timeout=30), service_id='test')
    )

    assert result.score == report.score
    assert result.summary == 'Clear issue.'
    assert result.actionable_feedback == 'Add a test.'
    assert result.positive_feedback == 'Includes code.'
    assert (result.prompt_tokens, result.completion_tokens) == (13, 9)
    request = model_query.call_args.kwargs
    assert isinstance(request['input'][0], SystemInput)
    assert isinstance(request['input'][1], TextInput)
    assert 'Fix this code' in request['input'][1].text
    assert request['tool_choice']['function']['name'] == request['tools'][0].name
