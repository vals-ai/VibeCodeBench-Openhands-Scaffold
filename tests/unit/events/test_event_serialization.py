import pytest
from model_library.base import QueryResult, QueryResultCost, QueryResultMetadata

from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.action.action import ActionSecurityRisk
from openhands.events.observation import CmdOutputMetadata, CmdOutputObservation
from openhands.events.serialization import event_from_dict, event_to_dict
from openhands.llm.metrics import Cost, Metrics, ResponseLatency, TokenUsage


def test_command_output_success_serialization():
    # Test successful command
    obs = CmdOutputObservation(
        command='ls',
        content='file1.txt\nfile2.txt',
        metadata=CmdOutputMetadata(exit_code=0),
    )
    serialized = event_to_dict(obs)
    assert serialized['success'] is True

    # Test failed command
    obs = CmdOutputObservation(
        command='ls',
        content='No such file or directory',
        metadata=CmdOutputMetadata(exit_code=1),
    )
    serialized = event_to_dict(obs)
    assert serialized['success'] is False


def test_message_model_response_serialization():
    response = QueryResult(
        output_text='Done',
        reasoning='Checked the result',
        metadata=QueryResultMetadata(
            cost=QueryResultCost(
                input=0.01,
                output=0.02,
                reasoning=0.03,
                cache_read=0.004,
                cache_write=0.005,
            ),
            duration_seconds=1.25,
            in_tokens=10,
            out_tokens=4,
            reasoning_tokens=3,
            cache_read_tokens=20,
            cache_write_tokens=2,
        ),
    )
    action = MessageAction(content='Done')
    action.model_response = response
    action.response_id = 'provider-response-id'

    serialized = event_to_dict(action)

    assert serialized['response_id'] == 'provider-response-id'
    assert serialized['model_response']['metadata']['in_tokens'] == 10
    assert serialized['model_response']['metadata']['out_tokens'] == 4
    assert serialized['model_response']['metadata']['reasoning_tokens'] == 3
    assert serialized['model_response']['metadata']['cache_read_tokens'] == 20
    assert serialized['model_response']['metadata']['cache_write_tokens'] == 2
    assert serialized['model_response']['metadata']['cost']['total'] == 0.069
    assert 'history' not in serialized['model_response']
    assert 'raw' not in serialized['model_response']

    deserialized = event_from_dict(serialized)

    assert deserialized.response_id == 'provider-response-id'
    assert deserialized.model_response == response
    assert event_to_dict(deserialized) == serialized


def test_message_model_response_rejects_unknown_type():
    serialized = event_to_dict(MessageAction(content='Done'))
    serialized['model_response'] = 'not-a-query-result'

    with pytest.raises(TypeError, match='model_response must be a dictionary'):
        event_from_dict(serialized)


def test_metrics_basic_serialization():
    # Create a basic action with accumulated usage
    action = MessageAction(content='Hello, world!')
    metrics = Metrics()
    metrics.add_cost(0.03)
    metrics.add_token_usage(30, 12, 70, 0, 0, '', reasoning_tokens=8)
    action._llm_metrics = metrics

    # Test serialization
    serialized = event_to_dict(action)
    assert 'llm_metrics' in serialized
    assert serialized['llm_metrics']['accumulated_cost'] == 0.03
    assert serialized['llm_metrics']['costs'][0]['cost'] == 0.03
    assert serialized['llm_metrics']['response_latencies'] == []
    assert len(serialized['llm_metrics']['token_usages']) == 1
    assert serialized['llm_metrics']['usage_accounting'] == {
        'schema_version': 2,
        'response_count': 1,
        'known_reasoning_tokens': 8,
        'missing_reasoning_responses': 0,
        'missing_cost_responses': 0,
    }
    accumulated = serialized['llm_metrics']['accumulated_token_usage']
    assert accumulated['reasoning_tokens'] == 8
    assert accumulated['total_output_tokens'] == 20

    # Test deserialization
    deserialized = event_from_dict(serialized)
    assert deserialized.llm_metrics is not None
    assert deserialized.llm_metrics.accumulated_cost == 0.03
    assert len(deserialized.llm_metrics.costs) == 1
    assert len(deserialized.llm_metrics.response_latencies) == 0
    assert len(deserialized.llm_metrics.token_usages) == 1
    assert event_to_dict(deserialized) == serialized


def test_legacy_message_remains_schema_less_after_round_trip():
    action = MessageAction(content='Done')
    action._llm_metrics = Metrics()
    serialized = event_to_dict(action)
    del serialized['llm_metrics']['usage_accounting']

    round_tripped = event_to_dict(event_from_dict(serialized))

    assert 'usage_accounting' not in round_tripped['llm_metrics']


def test_v1_usage_accounting_remains_readable():
    action = MessageAction(content='Done')
    action._llm_metrics = Metrics()
    serialized = event_to_dict(action)
    serialized['llm_metrics']['usage_accounting'] = {
        'schema_version': 1,
        'known_reasoning_tokens': 3,
        'missing_reasoning_responses': 1,
        'missing_cost_responses': 1,
    }

    round_tripped = event_to_dict(event_from_dict(serialized))

    assert round_tripped['llm_metrics']['usage_accounting'] == serialized[
        'llm_metrics'
    ]['usage_accounting']


def test_unknown_usage_accounting_version_fails():
    action = MessageAction(content='Done')
    action._llm_metrics = Metrics()
    serialized = event_to_dict(action)
    serialized['llm_metrics']['usage_accounting']['schema_version'] = 3

    with pytest.raises(ValueError, match='union_tag_invalid'):
        event_from_dict(serialized)


def test_metrics_full_serialization():
    # Create an observation with all metrics fields
    obs = CmdOutputObservation(
        command='ls',
        content='test.txt',
        metadata=CmdOutputMetadata(exit_code=0),
    )
    metrics = Metrics(model_name='test-model')
    metrics.accumulated_cost = 0.03

    # Add a cost
    cost = Cost(model='test-model', cost=0.02)
    metrics._costs.append(cost)

    # Add a response latency
    latency = ResponseLatency(model='test-model', latency=0.5, response_id='test-id')
    metrics.response_latencies = [latency]

    # Add token usage
    usage = TokenUsage(
        model='test-model',
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_tokens=7,
        cache_read_tokens=0,
        cache_write_tokens=0,
        response_id='test-id',
    )
    metrics.token_usages = [usage]

    obs._llm_metrics = metrics

    # Test serialization
    serialized = event_to_dict(obs)
    assert 'llm_metrics' in serialized
    metrics_dict = serialized['llm_metrics']
    assert metrics_dict['accumulated_cost'] == 0.03
    assert len(metrics_dict['costs']) == 1
    assert metrics_dict['costs'][0]['cost'] == 0.02
    assert len(metrics_dict['response_latencies']) == 1
    assert metrics_dict['response_latencies'][0]['latency'] == 0.5
    assert len(metrics_dict['token_usages']) == 1
    assert metrics_dict['token_usages'][0]['prompt_tokens'] == 10
    assert metrics_dict['token_usages'][0]['completion_tokens'] == 20
    assert metrics_dict['token_usages'][0]['reasoning_tokens'] == 7
    assert metrics_dict['token_usages'][0]['total_output_tokens'] == 27

    # Test deserialization
    deserialized = event_from_dict(serialized)
    assert deserialized.llm_metrics is not None
    assert deserialized.llm_metrics.accumulated_cost == 0.03
    assert len(deserialized.llm_metrics.costs) == 1
    assert deserialized.llm_metrics.costs[0].cost == 0.02
    assert len(deserialized.llm_metrics.response_latencies) == 1
    assert deserialized.llm_metrics.response_latencies[0].latency == 0.5
    assert len(deserialized.llm_metrics.token_usages) == 1
    assert deserialized.llm_metrics.token_usages[0].prompt_tokens == 10
    assert deserialized.llm_metrics.token_usages[0].completion_tokens == 20
    assert deserialized.llm_metrics.token_usages[0].reasoning_tokens == 7
    assert deserialized.llm_metrics.token_usages[0].total_output_tokens == 27


def test_metrics_none_serialization():
    # Test when metrics is None
    obs = CmdOutputObservation(
        command='ls',
        content='test.txt',
        metadata=CmdOutputMetadata(exit_code=0),
    )
    obs._llm_metrics = None

    # Test serialization
    serialized = event_to_dict(obs)
    assert 'llm_metrics' not in serialized

    # Test deserialization
    deserialized = event_from_dict(serialized)
    assert deserialized.llm_metrics is None


def test_action_risk_serialization():
    # Test action with security risk
    action = CmdRunAction(command='rm -rf /tmp/test')
    action.security_risk = ActionSecurityRisk.HIGH

    # Test serialization
    serialized = event_to_dict(action)
    assert 'security_risk' in serialized['args']
    assert serialized['args']['security_risk'] == ActionSecurityRisk.HIGH.value

    # Test deserialization
    deserialized = event_from_dict(serialized)
    assert deserialized.security_risk == ActionSecurityRisk.HIGH

    # Test action with no security risk
    action = CmdRunAction(command='ls')
    # Don't set action_risk

    # Test serialization
    serialized = event_to_dict(action)
    assert 'security_risk' in serialized['args']
    assert serialized['args']['security_risk'] == ActionSecurityRisk.UNKNOWN.value

    # Test deserialization
    deserialized = event_from_dict(serialized)
    assert deserialized.security_risk == ActionSecurityRisk.UNKNOWN
