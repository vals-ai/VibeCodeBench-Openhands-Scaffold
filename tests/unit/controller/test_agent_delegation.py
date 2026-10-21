import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.controller.state.control_flags import (
    BudgetControlFlag,
    IterationControlFlag,
)
from openhands.controller.state.state import State
from openhands.core.config import OpenHandsConfig
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.schema import AgentState
from openhands.events import EventSource, EventStream
from openhands.events.action import (
    AgentDelegateAction,
    AgentFinishAction,
    MessageAction,
)
from openhands.events.action.message import SystemMessageAction
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.llm.metrics import Metrics
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.memory import InMemoryFileStore


@pytest.fixture
def llm_registry(monkeypatch):
    def create_llm(service_id, config, retry_listener=None):
        llm = MagicMock(spec=LLM)
        llm.service_id = service_id
        llm.config = config
        llm.metrics = Metrics(model_name=config.model)
        return llm

    monkeypatch.setattr('openhands.llm.llm_registry.LLM', create_llm)
    config = OpenHandsConfig()
    return LLMRegistry(config=config)


@pytest.fixture
def conversation_stats():
    import uuid

    file_store = InMemoryFileStore({})
    # Use a unique conversation ID for each test to avoid conflicts
    conversation_id = f'test-conversation-{uuid.uuid4()}'
    return ConversationStats(
        file_store=file_store, conversation_id=conversation_id, user_id='test-user'
    )


@pytest.fixture
def connected_registry_and_stats(llm_registry, conversation_stats):
    """Connect the LLMRegistry and ConversationStats properly"""
    # Subscribe to LLM registry events to track metrics
    llm_registry.subscribe(conversation_stats.register_llm)
    return llm_registry, conversation_stats


@pytest.fixture
def mock_event_stream():
    """Creates an event stream in memory."""
    sid = f'test-{uuid4()}'
    file_store = InMemoryFileStore({})
    return EventStream(sid=sid, file_store=file_store)


@pytest.fixture
def mock_parent_agent(llm_registry):
    """Creates a mock parent agent for testing delegation."""
    agent = MagicMock(spec=Agent)
    agent.name = 'ParentAgent'
    agent.pending_tool_call_errors = []
    agent.llm = MagicMock(spec=LLM)
    agent.llm.service_id = 'main_agent'
    agent.llm.metrics = Metrics()
    agent.llm.config = LLMConfig()
    agent.llm.retry_listener = None  # Add retry_listener attribute
    agent.config = AgentConfig()
    agent.llm_registry = llm_registry  # Add the missing llm_registry attribute

    # Add a proper system message mock
    system_message = SystemMessageAction(content='Test system message')
    system_message._source = EventSource.AGENT
    system_message._id = -1  # Set invalid ID to avoid the ID check
    agent.get_system_message.return_value = system_message
    return agent


@pytest.fixture
def mock_child_agent(llm_registry):
    """Creates a mock child agent for testing delegation."""
    agent = MagicMock(spec=Agent)
    agent.name = 'ChildAgent'
    agent.pending_tool_call_errors = []
    agent.llm = MagicMock(spec=LLM)
    agent.llm.service_id = 'main_agent'
    agent.llm.metrics = Metrics()
    agent.llm.config = LLMConfig()
    agent.llm.retry_listener = None  # Add retry_listener attribute
    agent.config = AgentConfig()
    agent.llm_registry = llm_registry  # Add the missing llm_registry attribute

    system_message = SystemMessageAction(content='Test system message')
    system_message._source = EventSource.AGENT
    system_message._id = -1  # Set invalid ID to avoid the ID check
    agent.get_system_message.return_value = system_message
    return agent


def create_mock_agent_factory(mock_child_agent, llm_registry):
    """Helper function to create a mock agent factory with proper LLM registration."""

    def create_mock_agent(config, llm_registry=None):
        # Register the mock agent's LLM in the registry so get_combined_metrics() can find it
        if llm_registry:
            mock_child_agent.llm = llm_registry.get_llm('agent_llm', LLMConfig())
            mock_child_agent.llm_registry = (
                llm_registry  # Set the llm_registry attribute
            )
        return mock_child_agent

    return create_mock_agent


@pytest.mark.asyncio
async def test_delegation_flow(
    mock_parent_agent,
    mock_child_agent,
    mock_event_stream,
    connected_registry_and_stats,
    monkeypatch,
):
    llm_registry, conversation_stats = connected_registry_and_stats
    # Drive each controller event explicitly without concurrent subscriber callbacks.
    monkeypatch.setattr(mock_event_stream, 'subscribe', Mock())
    monkeypatch.setattr(
        Agent,
        'get_cls',
        Mock(return_value=create_mock_agent_factory(mock_child_agent, llm_registry)),
    )
    parent_llm = llm_registry.service_to_llm['agent']
    parent_llm.metrics.accumulated_cost = 2
    mock_parent_agent.llm = parent_llm
    parent_state = State(
        inputs=[],
        budget_flag=BudgetControlFlag(
            current_value=2, limit_increase_amount=10, max_value=10
        ),
        iteration_flag=IterationControlFlag(
            current_value=1, limit_increase_amount=10, max_value=10
        ),
    )
    parent = AgentController(
        agent=mock_parent_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=1,
        sid='parent',
        headless_mode=True,
        initial_state=parent_state,
    )
    try:
        delegate_action = AgentDelegateAction(agent='ChildAgent', inputs={'test': True})
        mock_parent_agent.step.return_value = delegate_action
        await parent.set_agent_state_to(AgentState.RUNNING)
        await parent._step()
        await parent._on_event(delegate_action)
        child = parent.delegate
        assert child is not None
        assert parent.state.iteration_flag.current_value == 2
        assert child.state.iteration_flag is parent.state.iteration_flag
        assert child.state.budget_flag is parent.state.budget_flag

        def child_step(state):
            mock_child_agent.llm.metrics.add_cost(1.0)
            return MessageAction(content='Working on delegated task')

        mock_child_agent.step.side_effect = child_step
        await child.set_agent_state_to(AgentState.RUNNING)
        for _ in range(4):
            await child._step()
        assert child.state.get_local_step() == 4
        assert parent.state.iteration_flag.current_value == 6
        assert conversation_stats.get_combined_metrics().accumulated_cost == 6

        finish = AgentFinishAction(outputs={'delegate_result': 'done'})
        await child._on_event(finish)

        def finish_delegate():
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                parent.on_event(finish)
            finally:
                loop.close()

        await asyncio.to_thread(finish_delegate)
        assert parent.delegate is None
        assert child.state.agent_state == AgentState.STOPPED
        result = next(
            event
            for event in reversed(list(mock_event_stream.get_events()))
            if isinstance(event, AgentDelegateObservation)
        )
        assert result.outputs == {'delegate_result': 'done'}
        mock_parent_agent.step.return_value = MessageAction(content='Task complete')
        await parent._on_event(result)
        assert parent.state.iteration_flag.current_value == 7
        assert parent.state.budget_flag.current_value == 6
    finally:
        if parent.delegate is not None:
            await parent.delegate.close()
        await parent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'delegate_state',
    [
        AgentState.RUNNING,
        AgentState.FINISHED,
        AgentState.ERROR,
        AgentState.REJECTED,
    ],
)
async def test_delegate_step_different_states(
    mock_parent_agent, delegate_state, connected_registry_and_stats
):
    """Ensure that delegate is closed or remains open based on the delegate's state."""
    mock_event_stream = MagicMock(spec=EventStream)
    mock_event_stream.get_latest_event_id.return_value = 0
    llm_registry, conversation_stats = connected_registry_and_stats

    # Create a state with iteration_flag.max_value set to 10
    state = State(inputs=[])
    state.iteration_flag.max_value = 10
    controller = AgentController(
        agent=mock_parent_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=1,  # Add the required iteration_delta parameter
        sid='test',
        confirmation_mode=False,
        headless_mode=True,
        initial_state=state,
    )

    mock_delegate = AsyncMock()
    controller.delegate = mock_delegate

    mock_delegate.state.iteration_flag = MagicMock()
    mock_delegate.state.iteration_flag.current_value = 5
    mock_delegate.state.outputs = {'result': 'test'}
    mock_delegate.state.last_error = ''
    mock_delegate.agent.name = 'TestDelegate'

    mock_delegate.get_agent_state = Mock(return_value=delegate_state)
    mock_delegate._step = AsyncMock()
    mock_delegate.close = AsyncMock()

    def call_on_event_with_new_loop():
        """In this thread, create and set a fresh event loop, so that the run_until_complete()
        calls inside controller.on_event(...) find a valid loop.
        """
        loop_in_thread = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop_in_thread)
            msg_action = MessageAction(content='Test message')
            msg_action._source = EventSource.USER
            controller.on_event(msg_action)
        finally:
            loop_in_thread.close()

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        future = loop.run_in_executor(executor, call_on_event_with_new_loop)
        await future

    # Give time for the event loop to process events
    await asyncio.sleep(0.5)

    if delegate_state == AgentState.RUNNING:
        assert controller.delegate is not None
        assert controller.state.iteration_flag.current_value == 0
        mock_delegate.close.assert_not_called()
    else:
        assert controller.delegate is None
        assert controller.state.iteration_flag.current_value == 5
        # The close method is called once in end_delegate
        assert mock_delegate.close.call_count == 1

    await controller.close()


@pytest.mark.asyncio
async def test_delegate_hits_global_limits(
    mock_child_agent,
    mock_event_stream,
    mock_parent_agent,
    connected_registry_and_stats,
    monkeypatch,
):
    llm_registry, conversation_stats = connected_registry_and_stats
    # Drive each controller event explicitly without concurrent subscriber callbacks.
    monkeypatch.setattr(mock_event_stream, 'subscribe', Mock())
    monkeypatch.setattr(
        Agent,
        'get_cls',
        Mock(return_value=create_mock_agent_factory(mock_child_agent, llm_registry)),
    )
    parent_llm = llm_registry.service_to_llm['agent']
    parent_llm.metrics.accumulated_cost = 2
    mock_parent_agent.llm = parent_llm
    parent = AgentController(
        agent=mock_parent_agent,
        event_stream=mock_event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=1,
        sid='parent',
        headless_mode=False,
        initial_state=State(
            inputs=[],
            budget_flag=BudgetControlFlag(
                current_value=2, limit_increase_amount=10, max_value=10
            ),
            iteration_flag=IterationControlFlag(
                current_value=2, limit_increase_amount=3, max_value=3
            ),
        ),
    )
    try:
        delegate_action = AgentDelegateAction(agent='ChildAgent', inputs={'test': True})
        mock_parent_agent.step.return_value = delegate_action
        await parent.set_agent_state_to(AgentState.RUNNING)
        await parent._step()
        await parent._on_event(delegate_action)
        child = parent.delegate
        assert child is not None
        assert child.state.iteration_flag is parent.state.iteration_flag
        assert child.state.budget_flag is parent.state.budget_flag
        await child.set_agent_state_to(AgentState.RUNNING)
        await child._step()
        assert child.state.agent_state == AgentState.ERROR
        assert child.state.last_error == (
            'RuntimeError: Agent reached maximum iteration. Current iteration: 3, max iteration: 3'
        )
        mock_child_agent.step.assert_not_called()

        await child.set_agent_state_to(AgentState.RUNNING)
        assert parent.state.iteration_flag.max_value == 6
        mock_child_agent.step.return_value = MessageAction(content='Continuing task')
        await child._step()
        assert parent.state.iteration_flag.current_value == 4
        mock_child_agent.step.assert_called_once()

        mock_child_agent.llm.metrics.add_cost(8)
        await child._step()
        assert child.state.agent_state == AgentState.ERROR
        assert 'maximum budget' in child.state.last_error
        assert parent.state.budget_flag.current_value == 10
        assert parent.state.iteration_flag.current_value == 5
        mock_child_agent.step.assert_called_once()
    finally:
        if parent.delegate is not None:
            await parent.delegate.close()
        await parent.close()
