from collections import deque
from typing import Union
from unittest.mock import Mock

import pytest
from model_library.base import (
    QueryResult,
    RawResponse,
    TextInput,
    ToolCall,
    ToolDefinition,
)

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.codeact_agent.function_calling import (
    response_to_actions as codeact_response_to_actions,
)
from openhands.agenthub.codeact_agent.tools import (
    BrowserTool,
    IPythonTool,
    LLMBasedFileEditTool,
    ThinkTool,
    create_cmd_run_tool,
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.browser import (
    _BROWSER_DESCRIPTION,
    _BROWSER_TOOL_DESCRIPTION,
)
from openhands.agenthub.readonly_agent.function_calling import (
    response_to_actions as readonly_response_to_actions,
)
from openhands.agenthub.readonly_agent.readonly_agent import ReadOnlyAgent
from openhands.agenthub.readonly_agent.tools import (
    GlobTool,
    GrepTool,
)
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.events.action import (
    CmdRunAction,
    MessageAction,
    NullAction,
)
from openhands.events.action.message import SystemMessageAction
from openhands.events.event import EventSource
from openhands.events.tool import ToolCallMetadata
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry


@pytest.fixture
def create_llm_registry():
    def _get_registry(llm_config):
        config = OpenHandsConfig()
        config.set_llm_config(llm_config)
        registry = Mock(spec=LLMRegistry)
        llm = Mock(spec=LLM)
        llm.config = llm_config
        registry.get_llm_from_agent_config.return_value = llm
        registry.get_router.return_value = llm
        return registry

    return _get_registry


@pytest.fixture(params=['CodeActAgent', 'ReadOnlyAgent'])
def agent_class(request):
    if request.param == 'CodeActAgent':
        return CodeActAgent
    else:
        from openhands.agenthub.readonly_agent.readonly_agent import ReadOnlyAgent

        return ReadOnlyAgent


@pytest.fixture
def agent(agent_class, create_llm_registry) -> Union[CodeActAgent, ReadOnlyAgent]:
    llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
    config = AgentConfig()
    agent = agent_class(config=config, llm_registry=create_llm_registry(llm_config))
    agent.llm = Mock()
    agent.llm.config = Mock()
    agent.llm.config.max_message_chars = 1000
    return agent


def test_agent_with_default_config_has_default_tools(create_llm_registry):
    llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
    config = AgentConfig()
    codeact_agent = CodeActAgent(
        config=config, llm_registry=create_llm_registry(llm_config)
    )
    assert len(codeact_agent.tools) > 0
    default_tool_names = [tool.name for tool in codeact_agent.tools]
    assert {
        'browser',
        'execute_bash',
        'execute_ipython_cell',
        'finish',
        'str_replace_editor',
        'think',
    }.issubset(default_tool_names)


@pytest.fixture
def mock_state() -> State:
    state = State()

    return state


def test_reset(agent):
    # Add some state
    action = MessageAction(content='test')
    action._source = EventSource.AGENT
    agent.pending_actions.append(action)
    agent.pending_tool_call_errors.append(FunctionCallValidationError('test'))

    # Create a mock state with initial user message
    mock_state = Mock(spec=State)
    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    mock_state.history = [initial_user_message]

    # Reset
    agent.reset()

    # Verify state is cleared
    assert len(agent.pending_actions) == 0
    assert not agent.pending_tool_call_errors


def test_step_with_pending_actions(agent):
    # Add a pending action
    pending_action = MessageAction(content='test')
    pending_action._source = EventSource.AGENT
    agent.pending_actions.append(pending_action)

    # Create a mock state with initial user message
    mock_state = Mock(spec=State)
    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    mock_state.history = [initial_user_message]

    # Step should return the pending action
    result = agent.step(mock_state)
    assert result == pending_action
    assert len(agent.pending_actions) == 0


@pytest.mark.parametrize(
    ('tool_call', 'error_type'),
    [
        (
            ToolCall(id='unknown', name='unknown_tool', args='{}'),
            FunctionCallNotExistsError,
        ),
        (
            ToolCall(id='malformed', name='execute_bash', args='{}'),
            FunctionCallValidationError,
        ),
    ],
)
def test_step_preserves_response_on_action_conversion_error(tool_call, error_type):
    response = QueryResult(tool_calls=[tool_call])
    agent = object.__new__(CodeActAgent)
    agent.pending_actions = deque()
    agent.pending_tool_call_errors = []
    agent.llm = Mock()
    agent.history = []
    agent.mcp_tools = {}
    agent._query = Mock(return_value=response)
    state = Mock(spec=State)
    state.inputs = []
    state.pending_tool_calls = {}

    action = agent.step(state)

    assert isinstance(action, NullAction)
    [error] = agent.pending_tool_call_errors
    assert isinstance(error, error_type)
    assert error.model_response is response


def test_step_handles_parallel_tool_calls_individually():
    invalid = ToolCall(id='invalid', name='str_replace_editor', args='null')
    valid = ToolCall(id='valid', name='execute_bash', args='{"command": "ls"}')
    unknown = ToolCall(id='unknown', name='unknown_tool', args='{}')
    response = QueryResult(
        output_text='Inspect the workspace.',
        reasoning='Find the relevant files first.',
        tool_calls=[invalid, valid, unknown],
    )
    agent = object.__new__(CodeActAgent)
    agent.pending_actions = deque()
    agent.pending_tool_call_errors = []
    agent.llm = Mock()
    agent.history = []
    agent.mcp_tools = {}
    agent._query = Mock(return_value=response)
    state = Mock(spec=State)
    state.inputs = []
    state.pending_tool_calls = {}

    action = agent.step(state)

    agent._query.assert_called_once()
    assert list(state.pending_tool_calls) == ['invalid', 'valid', 'unknown']
    assert not agent.pending_actions
    [invalid_error, unknown_error] = agent.pending_tool_call_errors
    assert invalid_error.tool_call is invalid
    assert invalid_error.model_response is response
    assert unknown_error.tool_call is unknown
    assert unknown_error.model_response is response
    assert isinstance(action, CmdRunAction)
    assert action.thought == (
        'Output: Inspect the workspace.\nReasoning: Find the relevant files first.'
    )
    assert action.tool_call_metadata is not None
    assert action.tool_call_metadata.tool_call_id == 'valid'
    assert action.tool_call_metadata.model_response is response
    assert action.tool_call_metadata.total_calls_in_response == 3


def test_response_batch_uses_subclass_parser():
    tool_call = ToolCall(id='valid', name='execute_bash', args='{"command": "ls"}')
    response = QueryResult(tool_calls=[tool_call])
    action = CmdRunAction(command='ls')
    action.tool_call_metadata = ToolCallMetadata(
        tool_call_id=tool_call.id,
        function_name=tool_call.name,
        model_response=response,
        total_calls_in_response=1,
    )
    agent = object.__new__(ReadOnlyAgent)
    agent.response_to_actions = Mock(return_value=[action])

    actions, errors = agent._response_to_actions_and_errors(response)

    assert actions == [action]
    assert not errors
    [single_response] = agent.response_to_actions.call_args.args
    assert single_response.tool_calls == [tool_call]
    assert action.tool_call_metadata.model_response is response


def test_cmd_run_tool():
    CmdRunTool = create_cmd_run_tool()
    assert CmdRunTool.name == 'execute_bash'
    assert 'command' in CmdRunTool.body.properties
    assert 'security_risk' in CmdRunTool.body.properties
    assert CmdRunTool.body.required == [
        'command',
        'security_risk',
    ]


def test_ipython_tool():
    assert IPythonTool.name == 'execute_ipython_cell'
    assert 'code' in IPythonTool.body.properties
    assert 'security_risk' in IPythonTool.body.properties
    assert IPythonTool.body.required == [
        'code',
        'security_risk',
    ]


def test_llm_based_file_edit_tool():
    assert LLMBasedFileEditTool.name == 'edit_file'

    properties = LLMBasedFileEditTool.body.properties
    assert 'path' in properties
    assert 'content' in properties
    assert 'start' in properties
    assert 'end' in properties
    assert 'security_risk' in properties

    assert LLMBasedFileEditTool.body.required == [
        'path',
        'content',
        'security_risk',
    ]


def test_str_replace_editor_tool():
    StrReplaceEditorTool = create_str_replace_editor_tool()
    assert StrReplaceEditorTool.name == 'str_replace_editor'

    properties = StrReplaceEditorTool.body.properties
    assert 'command' in properties
    assert 'path' in properties
    assert 'file_text' in properties
    assert 'old_str' in properties
    assert 'new_str' in properties
    assert 'insert_line' in properties
    assert 'security_risk' in properties

    assert StrReplaceEditorTool.body.required == [
        'command',
        'path',
        'security_risk',
    ]


def test_browser_tool():
    assert BrowserTool.name == 'browser'
    assert 'code' in BrowserTool.body.properties
    assert 'security_risk' in BrowserTool.body.properties
    assert BrowserTool.body.required == [
        'code',
        'security_risk',
    ]
    # Check that the description includes all the functions
    description = _BROWSER_TOOL_DESCRIPTION
    assert 'goto(' in description
    assert 'go_back()' in description
    assert 'go_forward()' in description
    assert 'noop(' in description
    assert 'scroll(' in description
    assert 'fill(' in description
    assert 'select_option(' in description
    assert 'click(' in description
    assert 'dblclick(' in description
    assert 'hover(' in description
    assert 'press(' in description
    assert 'focus(' in description
    assert 'clear(' in description
    assert 'drag_and_drop(' in description
    assert 'upload_file(' in description

    # Test BrowserTool definition
    assert BrowserTool.name == 'browser'
    assert BrowserTool.body.description == _BROWSER_DESCRIPTION
    assert 'code' in BrowserTool.body.properties
    assert BrowserTool.body.required == [
        'code',
        'security_risk',
    ]
    assert BrowserTool.body.properties['code']['type'] == 'string'
    assert 'description' in BrowserTool.body.properties['code']


def test_response_to_actions_invalid_tool():
    response = QueryResult(
        tool_calls=[ToolCall(id='bad', name='invalid_tool', args={})]
    )
    with pytest.raises(FunctionCallNotExistsError):
        codeact_response_to_actions(response)
    with pytest.raises(FunctionCallNotExistsError):
        readonly_response_to_actions(response)


def test_step_with_no_pending_actions(mock_state: State, create_llm_registry):
    agent = CodeActAgent(
        config=AgentConfig(), llm_registry=create_llm_registry(LLMConfig(model='test'))
    )
    mock_state.inputs = [TextInput(text='Initial user message')]
    agent.llm.completion.return_value = QueryResult(output_text='Task completed')
    action = agent.step(mock_state)
    assert isinstance(action, MessageAction)
    assert action.content == 'Task completed'
    assert mock_state.inputs == []


@pytest.mark.parametrize('agent_type', ['CodeActAgent', 'ReadOnlyAgent'])
def test_correct_tool_description_loaded_based_on_model_name(
    agent_type, create_llm_registry
):
    agent_class = CodeActAgent if agent_type == 'CodeActAgent' else ReadOnlyAgent
    agents = [
        agent_class(
            config=AgentConfig(),
            llm_registry=create_llm_registry(LLMConfig(model=model)),
        )
        for model in ('mock_o3_model', 'mock_sonnet_model')
    ]
    assert agents[0].tools == agents[1].tools
    assert all(isinstance(tool, ToolDefinition) for tool in agents[0].tools)
    assert all(tool.body.description for tool in agents[0].tools)


def test_step_passes_inputs_and_retains_provider_history(agent, mock_state: State):
    inputs = [
        TextInput(text='First user message'),
        TextInput(text='Second user message'),
    ]
    history = [RawResponse(response={'id': 'previous'})]
    next_history = [*history, *inputs, RawResponse(response={'id': 'next'})]
    mock_state.inputs = inputs.copy()
    agent.history = history

    def complete(**kwargs):
        assert kwargs == {'input': inputs, 'history': history, 'tools': agent.tools}
        return QueryResult(output_text='Done', history=next_history)

    agent.llm.completion.side_effect = complete
    action = agent.step(mock_state)
    assert action.content == 'Done'
    assert agent.history == next_history
    assert mock_state.inputs == []


def test_grep_tool():
    assert GrepTool['type'] == 'function'
    assert GrepTool['function']['name'] == 'grep'

    properties = GrepTool['function']['parameters']['properties']
    assert 'pattern' in properties
    assert 'path' in properties
    assert 'include' in properties

    assert GrepTool['function']['parameters']['required'] == ['pattern']


def test_glob_tool():
    assert GlobTool['type'] == 'function'
    assert GlobTool['function']['name'] == 'glob'

    properties = GlobTool['function']['parameters']['properties']
    assert 'pattern' in properties
    assert 'path' in properties

    assert GlobTool['function']['parameters']['required'] == ['pattern']


def test_think_tool():
    assert ThinkTool.name == 'think'

    properties = ThinkTool.body.properties
    assert 'thought' in properties

    assert ThinkTool.body.required == ['thought']


def test_readonly_and_loc_step_use_their_registered_tools(create_llm_registry):
    from openhands.agenthub.loc_agent.loc_agent import LocAgent
    from openhands.events.action import FileReadAction, IPythonRunCellAction

    for agent_class, name, args, expected in (
        (ReadOnlyAgent, 'view', {'path': '/tmp/example'}, FileReadAction),
        (
            LocAgent,
            'get_entity_contents',
            {'entity_names': ['example']},
            IPythonRunCellAction,
        ),
    ):
        agent = agent_class(
            config=AgentConfig(),
            llm_registry=create_llm_registry(LLMConfig(model='test')),
        )
        assert name in {tool.name for tool in agent.tools}
        response = QueryResult(tool_calls=[ToolCall(id='call', name=name, args=args)])
        agent.llm.completion.return_value = response
        state = State(inputs=[TextInput(text='Inspect example')])
        action = agent.step(state)
        assert isinstance(action, expected)
        assert action.tool_call_metadata.model_response is response
        assert state.pending_tool_calls['call'].name == name


def test_get_system_message(create_llm_registry):
    """Test that the Agent.get_system_message method returns a SystemMessageAction."""
    # Create a mock agent
    config = AgentConfig()
    llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
    agent = CodeActAgent(config=config, llm_registry=create_llm_registry(llm_config))

    result = agent.get_system_message()

    # Check that the system message was created correctly
    assert isinstance(result, SystemMessageAction)
    assert 'You are OpenHands agent' in result.content
    assert len(result.tools) > 0
    assert any(tool.name == 'execute_bash' for tool in result.tools)
    assert result._source == EventSource.AGENT


def test_step_accepts_empty_inputs_for_continuation(agent, mock_state: State):
    agent.history = [TextInput(text='Prior request')]
    agent.llm.completion.return_value = QueryResult(output_text='Continuation')
    action = agent.step(mock_state)
    assert action.content == 'Continuation'
    assert agent.llm.completion.call_args.kwargs['input'] == []


@pytest.mark.parametrize('agent_kind', ['readonly', 'loc'])
def test_specialized_parser_rejects_malformed_tool_arguments(agent_kind):
    from openhands.agenthub.loc_agent.function_calling import (
        response_to_actions as loc_response,
    )

    parse = readonly_response_to_actions if agent_kind == 'readonly' else loc_response
    response = QueryResult(
        tool_calls=[ToolCall(id='bad-json', name='finish', args='not json')]
    )
    with pytest.raises(FunctionCallValidationError):
        parse(response)


@pytest.mark.parametrize('agent_kind', ['readonly', 'loc'])
def test_specialized_parser_preserves_finish_and_response_metadata(agent_kind):
    from openhands.agenthub.loc_agent.function_calling import (
        response_to_actions as loc_response,
    )
    from openhands.events.action import AgentFinishAction

    parse = readonly_response_to_actions if agent_kind == 'readonly' else loc_response
    response = QueryResult(
        tool_calls=[ToolCall(id='done', name='finish', args='{"message":"Finished"}')]
    )
    response.extras.response_id = 'provider-response'
    action = parse(response)[0]
    assert isinstance(action, AgentFinishAction)
    assert action.final_thought == 'Finished'
    assert action.response_id == 'provider-response'
    assert action.tool_call_metadata.tool_call_id == 'done'
