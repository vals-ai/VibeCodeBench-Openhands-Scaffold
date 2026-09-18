from model_library.base import TextInput

from openhands.controller.state.control_flags import IterationControlFlag
from openhands.controller.state.state import State
from openhands.controller.state.state_tracker import StateTracker
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.memory import InMemoryFileStore


def test_restore_reopens_event_history():
    state = State(end_id=42)
    store = InMemoryFileStore()
    state.save_to_session("test_sid", store, None)

    restored_state = State.restore_from_session("test_sid", store, None)

    assert restored_state.end_id == -1


def test_restore_preserves_agent_history():
    state = State(agent_history=[TextInput(text="remember me")])
    store = InMemoryFileStore()
    state.save_to_session("test_sid", store, None)

    restored_state = State.restore_from_session("test_sid", store, None)

    assert restored_state.agent_history == [TextInput(text="remember me")]


def test_vcb_turn_gets_a_fresh_iteration_budget(monkeypatch):
    monkeypatch.setenv("VCB100_GENERATION_TURN", "1")
    state = State(
        agent_history=[TextInput(text="remember me")],
        iteration_flag=IterationControlFlag(
            limit_increase_amount=6,
            current_value=6,
            max_value=6,
        ),
    )
    tracker = StateTracker(None, None, None)

    tracker.set_initial_state(
        "test_sid",
        state,
        ConversationStats(None, "test_sid", None),
        max_iterations=30,
        max_budget_per_task=None,
    )

    assert tracker.state.iteration_flag.current_value == 0
    assert tracker.state.iteration_flag.max_value == 30
    assert tracker.state.agent_history == [TextInput(text="remember me")]
