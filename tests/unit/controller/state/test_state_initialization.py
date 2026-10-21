from openhands.controller.state.control_flags import IterationControlFlag
from openhands.controller.state.state import State
from openhands.controller.state.state_tracker import StateTracker
from openhands.server.services.conversation_stats import ConversationStats


def test_initial_state_preserves_shared_iteration_budget():
    counter = IterationControlFlag(
        limit_increase_amount=30, current_value=20, max_value=30
    )
    state = State(iteration_flag=counter, parent_iteration=20)
    tracker = StateTracker(None, None, None)

    tracker.set_initial_state(
        "test",
        state,
        ConversationStats(None, "test", None),
        max_iterations=30,
        max_budget_per_task=None,
    )

    assert tracker.state.iteration_flag is counter
    assert counter.current_value == 20
    assert tracker.state.get_local_step() == 0
    tracker.state.iteration_flag.current_value += 2
    assert counter.current_value == 22
    assert tracker.state.get_local_step() == 2
