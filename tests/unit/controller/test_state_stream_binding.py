from unittest.mock import MagicMock

import pytest

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.controller.state.state import State
from openhands.events import EventSource, EventStream
from openhands.events.action import MessageAction
from openhands.storage.memory import InMemoryFileStore


@pytest.fixture
def stream_store():
    store = InMemoryFileStore()
    stream = EventStream('state-binding', store)
    yield stream, store
    stream.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('restored', [False, True])
async def test_controller_binds_new_and_restored_state_to_live_history(
    restored, stream_store
):
    stream, store = stream_store
    stream.add_event(MessageAction(content='first task'), EventSource.USER)
    initial_state = None
    if restored:
        saved = State(start_id=0)
        saved.save_to_session('state-binding', store, None)
        initial_state = State.restore_from_session('state-binding', store, None)

    agent = MagicMock(spec=Agent)
    agent.get_system_message.return_value = None
    controller = AgentController(
        agent=agent,
        event_stream=stream,
        conversation_stats=MagicMock(),
        iteration_delta=10,
        initial_state=initial_state,
        is_delegate=True,
    )
    try:
        assert [event.content for event in controller.state.history] == ['first task']
        stream.add_event(MessageAction(content='second task'), EventSource.USER)
        assert [event.content for event in controller.state.history] == [
            'first task',
            'second task',
        ]
    finally:
        await controller.close(set_stop_state=False)
