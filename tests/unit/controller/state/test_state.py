import pytest

from openhands.controller.state.state import State
from openhands.core.schema import AgentState
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.action.agent import CondensationAction
from openhands.events.event import EventSource
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.events.stream import EventStream
from openhands.storage.memory import InMemoryFileStore


@pytest.fixture
def stream():
    event_stream = EventStream('state-test', InMemoryFileStore())
    yield event_stream
    event_stream.close()


def test_state_view_reads_new_events_and_respects_condensation(stream):
    state = State()
    state.bind_event_stream(stream)
    first = MessageAction(content='Original task')
    stream.add_event(first, EventSource.USER)
    assert [event.content for event in state.view] == ['Original task']

    stream.add_event(MessageAction(content='New detail'), EventSource.USER)
    stream.add_event(
        CondensationAction(
            forgotten_event_ids=[first.id], summary='Task summary', summary_offset=0
        ),
        EventSource.AGENT,
    )
    view = state.view
    assert isinstance(view[0], AgentCondensationObservation)
    assert [event.content for event in view] == ['Task summary', 'New detail']
    assert state.history[0].content == 'Original task'


def test_state_history_respects_delegate_event_bounds(stream):
    for content in ['Parent task', 'Delegate task', 'Later parent event']:
        stream.add_event(MessageAction(content=content), EventSource.USER)
    state = State(start_id=1, end_id=1)
    state.bind_event_stream(stream)
    assert [event.content for event in state.history] == ['Delegate task']


def test_current_intent_uses_messages_after_last_finish(stream):
    state = State()
    state.bind_event_stream(stream)
    stream.add_event(MessageAction(content='First task'), EventSource.USER)
    stream.add_event(MessageAction(content='First detail'), EventSource.USER)
    assert state.get_current_user_intent()[0] == 'First task'
    stream.add_event(AgentFinishAction(), EventSource.AGENT)
    stream.add_event(MessageAction(content='Second task'), EventSource.USER)
    assert state.get_current_user_intent() == ('Second task', None)


def test_session_restore_rebinds_stream_without_serializing_it(stream):
    state = State(agent_state=AgentState.RUNNING)
    state.bind_event_stream(stream)
    stream.add_event(MessageAction(content='Saved task'), EventSource.USER)
    store = InMemoryFileStore()
    state.save_to_session('saved', store, None)
    restored = State.restore_from_session('saved', store, None)
    assert '_event_stream' not in restored.__dict__
    restored.bind_event_stream(stream)
    assert restored.agent_state == AgentState.LOADING
    assert restored.resume_state == AgentState.RUNNING
    assert restored.get_current_user_intent()[0] == 'Saved task'
    stream.add_event(MessageAction(content='New after restore'), EventSource.USER)
    assert restored.history[-1].content == 'New after restore'
