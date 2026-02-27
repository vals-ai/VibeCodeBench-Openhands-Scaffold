from __future__ import annotations

import base64
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, override

from model_library.base import InputItem, ToolCall

import openhands
from openhands.controller.state.control_flags import (
    BudgetControlFlag,
    IterationControlFlag,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema import AgentState
from openhands.events.action.agent import ChangeAgentStateAction
from openhands.events.action.empty import NullAction
from openhands.events.event import Event
from openhands.events.event_filter import EventFilter
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.events.observation.empty import NullObservation
from openhands.llm.metrics import Metrics
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.files import FileStore
from openhands.storage.locations import get_conversation_agent_state_filename

RESUMABLE_STATES = [
    AgentState.RUNNING,
    AgentState.PAUSED,
    AgentState.AWAITING_USER_INPUT,
    AgentState.FINISHED,
]


@dataclass
class State:
    """Represents the running state of an agent in the OpenHands system, saving data of its operation and memory.

    - Multi-agent/delegate state:
      - store the task (conversation between the agent and the user)
      - the subtask (conversation between an agent and the user or another agent)
      - global and local iterations
      - delegate levels for multi-agent interactions
      - almost stuck state

    - Running state of an agent:
      - current agent state (e.g., LOADING, RUNNING, PAUSED)
      - traffic control state for rate limiting
      - confirmation mode
      - the last error encountered

    - Data for saving and restoring the agent:
      - save to and restore from a session
      - serialize with pickle and base64

    - Save / restore data about message history
      - start and end IDs for events in agent's history
      - summaries and delegate summaries

    - Metrics:
      - global metrics for the current task
      - local metrics for the current subtask

    - Extra data:
      - additional task-specific data
    """

    session_id: str = ''
    user_id: str | None = None
    iteration_flag: IterationControlFlag = field(
        default_factory=lambda: IterationControlFlag(
            limit_increase_amount=100, current_value=0, max_value=100
        )
    )
    conversation_stats: ConversationStats | None = None
    budget_flag: BudgetControlFlag | None = None
    confirmation_mode: bool = False
    inputs: list[InputItem] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)

    pending_tool_calls: dict[str, ToolCall] = field(default_factory=dict)
    agent_state: AgentState = AgentState.LOADING
    resume_state: AgentState | None = None

    # root agent has level 0, and every delegate increases the level by one
    delegate_level: int = 0
    # start_id and end_id track the range of events in history
    start_id: int = -1
    end_id: int = -1

    parent_metrics_snapshot: Metrics | None = None
    parent_iteration: int = 100

    # NOTE: this is used by the controller to track parent's metrics snapshot before delegation
    # evaluation tasks to store extra data needed to track the progress/state of the task.
    extra_data: dict[str, Any] = field(default_factory=dict)
    last_error: str = ''

    metrics: Metrics = field(default_factory=Metrics)

    def save_to_session(
        self, sid: str, file_store: FileStore, user_id: str | None
    ) -> None:
        conversation_stats = self.conversation_stats
        self.conversation_stats = None  # Don't save conversation stats, handles itself

        pickled = pickle.dumps(self)
        logger.debug(f'Saving state to session {sid}:{self.agent_state}')
        encoded = base64.b64encode(pickled).decode('utf-8')
        try:
            file_store.write(
                get_conversation_agent_state_filename(sid, user_id), encoded
            )

            # see if state is in the old directory on saas/remote use cases and delete it.
            if user_id:
                filename = get_conversation_agent_state_filename(sid)
                try:
                    file_store.delete(filename)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f'Failed to save state to session: {e}')
            raise e

        self.conversation_stats = conversation_stats  # restore reference

    @staticmethod
    def restore_from_session(
        sid: str, file_store: FileStore, user_id: str | None = None
    ) -> State:
        """Restores the state from the previously saved session."""
        state: State
        try:
            encoded = file_store.read(
                get_conversation_agent_state_filename(sid, user_id)
            )
            pickled = base64.b64decode(encoded)
            state = pickle.loads(pickled)
        except FileNotFoundError:
            # if user_id is provided, we are in a saas/remote use case
            # and we need to check if the state is in the old directory.
            if user_id:
                filename = get_conversation_agent_state_filename(sid)
                encoded = file_store.read(filename)
                pickled = base64.b64decode(encoded)
                state = pickle.loads(pickled)
            else:
                raise FileNotFoundError(
                    f'Could not restore state from session file for sid: {sid}'
                ) from None
        except Exception as e:
            logger.debug(f'Could not restore state from session: {e}')
            raise e

        # update state
        if state.agent_state in RESUMABLE_STATES:
            state.resume_state = state.agent_state
        else:
            state.resume_state = None

        # first state after restore
        state.agent_state = AgentState.LOADING

        # We don't need to clean up deprecated fields here
        # They will be handled by __getstate__ when the state is saved again

        return state

    @override
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Update the state
        self.__dict__.update(state)

        # Ensure we have default values for new fields if they're missing
        if not hasattr(self, 'iteration_flag'):
            self.iteration_flag = IterationControlFlag(
                limit_increase_amount=100, current_value=0, max_value=100
            )

        if not hasattr(self, 'budget_flag'):
            self.budget_flag = None

    def to_llm_metadata(self, model_name: str, agent_name: str) -> dict[str, Any]:
        metadata = {
            'session_id': self.session_id,
            'trace_version': openhands.__version__,
            'trace_user_id': self.user_id,
            'tags': ','.join(
                [
                    f'model:{model_name}',
                    f'agent:{agent_name}',
                    f'web_host:{os.environ.get("WEB_HOST", "unspecified")}',
                    f'openhands_version:{openhands.__version__}',
                ]
            ),
        }

        return metadata

    def get_local_step(self):
        if not self.parent_iteration:
            return self.iteration_flag.current_value

        return self.iteration_flag.current_value - self.parent_iteration

    def get_local_metrics(self):
        if not self.parent_metrics_snapshot:
            return self.metrics
        return self.metrics.diff(self.parent_metrics_snapshot)

    def get_history_from_stream(self, event_stream: Any) -> list[Event]:
        """Fetch history from event_stream for compatibility with legacy code.

        Args:
            event_stream: The EventStream to query

        Returns:
            List of filtered events
        """

        agent_history_filter = EventFilter(
            exclude_types=(
                NullAction,
                NullObservation,
                ChangeAgentStateAction,
                AgentStateChangedObservation,
            ),
            exclude_hidden=True,
        )

        start_id = self.start_id if self.start_id >= 0 else 0
        end_id = self.end_id if self.end_id >= 0 else event_stream.get_latest_event_id()

        return list(
            event_stream.search_events(
                start_id=start_id,
                end_id=end_id,
                reverse=False,
                filter=agent_history_filter,
            )
        )

    def flush(self) -> None:
        """Clear staged model inputs after a completion call."""
        self.inputs.clear()
