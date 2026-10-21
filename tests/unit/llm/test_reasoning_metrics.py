import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openhands.core.config import LLMConfig
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics, TokenUsage, UsageAccounting


def test_token_usage_tracks_visible_reasoning_and_total_output():
    usage = TokenUsage(completion_tokens=5, reasoning_tokens=3)

    assert usage.completion_tokens == 5
    assert usage.reasoning_tokens == 3
    assert usage.total_output_tokens == 8


def test_token_usage_preserves_unreported_and_reported_zero_reasoning():
    unreported = TokenUsage(completion_tokens=4)
    reported_zero = TokenUsage(completion_tokens=6, reasoning_tokens=0)

    assert unreported.reasoning_tokens is None
    assert unreported.total_output_tokens is None
    assert reported_zero.total_output_tokens == 6

    combined = unreported + reported_zero
    assert combined.reasoning_tokens is None
    assert combined.total_output_tokens is None


def test_token_usage_unpickles_legacy_metrics_without_reasoning():
    legacy = TokenUsage(completion_tokens=4)
    del legacy.__dict__["reasoning_tokens"]

    restored = pickle.loads(pickle.dumps(legacy))

    assert restored.reasoning_tokens is None
    assert (restored + TokenUsage()).reasoning_tokens is None


def test_metrics_unpickles_without_usage_accounting():
    legacy = Metrics()
    del legacy._usage_accounting

    restored = pickle.loads(pickle.dumps(legacy))

    assert 'usage_accounting' not in restored.get()


def test_metrics_merge_preserves_optional_reasoning():
    reported = Metrics()
    reported.add_token_usage(1, 4, 0, 0, 0, "", reasoning_tokens=3)
    unreported = Metrics()
    unreported.add_token_usage(2, 5, 0, 0, 0, "", reasoning_tokens=None)

    assert reported.accumulated_token_usage.reasoning_tokens == 3
    reported.merge(Metrics())
    assert reported.accumulated_token_usage.reasoning_tokens == 3

    reported.merge(unreported)

    usage = reported.accumulated_token_usage
    assert usage.completion_tokens == 9
    assert usage.reasoning_tokens is None
    assert usage.total_output_tokens is None


def test_zero_token_unreported_usage_remains_unknown():
    unreported = Metrics()
    unreported.add_token_usage(0, 0, 0, 0, 0, "", reasoning_tokens=None)
    assert unreported.get()['usage_accounting']['response_count'] == 1

    sequential = unreported.copy()
    sequential.add_token_usage(1, 1, 0, 0, 0, "", reasoning_tokens=5)
    assert sequential.accumulated_token_usage.reasoning_tokens is None

    reported = Metrics()
    reported.add_token_usage(1, 1, 0, 0, 0, "", reasoning_tokens=5)

    unreported.merge(reported)

    assert unreported.accumulated_token_usage.reasoning_tokens is None


def test_metrics_merge_preserves_accumulated_only_reasoning():
    accumulated_only = Metrics()
    accumulated_only._accumulated_token_usage = TokenUsage(
        completion_tokens=4,
        reasoning_tokens=3,
    )

    accumulated_only.merge(Metrics())
    assert accumulated_only.accumulated_token_usage.reasoning_tokens == 3

    other = Metrics()
    other._accumulated_token_usage = TokenUsage(
        completion_tokens=5,
        reasoning_tokens=2,
    )
    accumulated_only.merge(other)

    usage = accumulated_only.accumulated_token_usage
    assert usage.reasoning_tokens == 5
    assert usage.total_output_tokens == 14


def test_metrics_diff_preserves_reasoning_reporting():
    metrics = Metrics()
    metrics.add_token_usage(1, 2, 0, 0, 0, "", reasoning_tokens=3)
    baseline = metrics.copy()

    metrics.add_token_usage(1, 2, 0, 0, 0, "", reasoning_tokens=None)
    assert metrics.diff(baseline).accumulated_token_usage.reasoning_tokens is None

    baseline = metrics.copy()
    metrics.add_token_usage(1, 2, 0, 0, 0, "", reasoning_tokens=0)
    assert metrics.diff(baseline).accumulated_token_usage.reasoning_tokens == 0

    baseline = metrics.copy()
    metrics.add_token_usage(1, 2, 0, 0, 0, "", reasoning_tokens=3)
    diff = metrics.diff(baseline).accumulated_token_usage
    assert diff.reasoning_tokens == 3
    assert diff.total_output_tokens == 5


def test_usage_accounting_tracks_complete_and_incomplete_responses():
    metrics = Metrics()
    metrics.add_cost(0)
    metrics.add_token_usage(10, 4, 20, 0, 0, '', reasoning_tokens=3)

    assert metrics.get()['usage_accounting'] == {
        'schema_version': 2,
        'response_count': 1,
        'known_reasoning_tokens': 3,
        'missing_reasoning_responses': 0,
        'missing_cost_responses': 0,
    }
    assert [cost.cost for cost in metrics.costs] == [0]

    metrics.add_cost(None)
    metrics.add_token_usage(2, 5, 30, 0, 0, '', reasoning_tokens=None)

    assert metrics.get()['usage_accounting'] == {
        'schema_version': 2,
        'response_count': 2,
        'known_reasoning_tokens': 3,
        'missing_reasoning_responses': 1,
        'missing_cost_responses': 1,
    }
    assert metrics.accumulated_cost == 0
    assert [cost.cost for cost in metrics.costs] == [0]


def test_usage_accounting_merge_and_diff_preserve_missing_counts():
    metrics = Metrics()
    metrics.add_cost(0)
    metrics.add_token_usage(10, 4, 20, 0, 0, '', reasoning_tokens=3)
    baseline = metrics.copy()

    unreported = Metrics()
    unreported.add_cost(None)
    unreported.add_token_usage(2, 5, 30, 0, 0, '', reasoning_tokens=None)
    metrics.merge(unreported)

    assert metrics.get()['usage_accounting'] == {
        'schema_version': 2,
        'response_count': 2,
        'known_reasoning_tokens': 3,
        'missing_reasoning_responses': 1,
        'missing_cost_responses': 1,
    }
    assert metrics.diff(baseline).get()['usage_accounting'] == {
        'schema_version': 2,
        'response_count': 1,
        'known_reasoning_tokens': 0,
        'missing_reasoning_responses': 1,
        'missing_cost_responses': 1,
    }


def test_merging_v1_accounting_does_not_invent_response_count():
    legacy = Metrics()
    legacy._usage_accounting = UsageAccounting(
        schema_version=1,
        known_reasoning_tokens=2,
        missing_reasoning_responses=0,
        missing_cost_responses=0,
    )
    current = Metrics()
    current.add_token_usage(0, 0, 0, 0, 0, '', reasoning_tokens=3)

    legacy.merge(current)

    assert legacy.get()['usage_accounting'] == {
        'schema_version': 1,
        'known_reasoning_tokens': 5,
        'missing_reasoning_responses': 0,
        'missing_cost_responses': 0,
    }


@pytest.mark.parametrize(
    (
        'reasoning_model',
        'reported_reasoning',
        'expected_reasoning',
        'missing_reasoning_responses',
    ),
    [
        (False, None, 0, 0),
        (True, None, None, 1),
        (True, 6, 6, 0),
    ],
)
def test_llm_records_usage_once_per_query(
    monkeypatch,
    reasoning_model,
    reported_reasoning,
    expected_reasoning,
    missing_reasoning_responses,
):
    metadata = SimpleNamespace(
        duration_seconds=None,
        cost=None,
        in_tokens=10,
        out_tokens=4,
        reasoning_tokens=reported_reasoning,
        cache_read_tokens=None,
        cache_write_tokens=None,
    )
    model = MagicMock()
    model.reasoning = reasoning_model
    model.query = AsyncMock(return_value=SimpleNamespace(metadata=metadata))
    monkeypatch.setattr('openhands.llm.llm.fetch_registry_model', lambda _: model)
    runner = MagicMock()
    runner.run.side_effect = lambda coroutine, timeout: asyncio.run(coroutine)
    monkeypatch.setattr('openhands.llm.llm._persistent_loop', runner)

    llm = LLM(LLMConfig(model='openai/gpt-4o', timeout=30), 'test-service')
    llm.pretty_print = MagicMock()
    llm.completion(input=[])

    model.query.assert_awaited_once_with(input=[], history=[], tools=[])
    assert len(llm.metrics.token_usages) == 1
    usage = llm.metrics.token_usages[0]
    assert usage.completion_tokens == 4
    assert usage.reasoning_tokens == expected_reasoning
    assert usage.per_turn_token == 14
    assert usage.cache_read_tokens == 0
    assert usage.cache_write_tokens == 0
    assert llm.metrics.get()['usage_accounting'] == {
        'schema_version': 2,
        'response_count': 1,
        'known_reasoning_tokens': reported_reasoning or 0,
        'missing_reasoning_responses': missing_reasoning_responses,
        'missing_cost_responses': 1,
    }
    assert llm.metrics.costs == []
    assert (
        llm.pretty_print.call_args_list[0].args[0].startswith('[Turn Cost] unknown\n')
    )
