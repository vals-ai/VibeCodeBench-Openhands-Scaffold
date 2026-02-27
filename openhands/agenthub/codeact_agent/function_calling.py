"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json
from typing import Any, cast

from model_library.base import QueryResult

from openhands.agenthub.codeact_agent.tools import (
    BrowserTool,
    CondensationRequestTool,
    FinishTool,
    IPythonTool,
    LLMBasedFileEditTool,
    ThinkTool,
    create_cmd_run_tool,
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.security_utils import RISK_LEVELS
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    ActionSecurityRisk,
    AgentDelegateAction,
    AgentFinishAction,
    AgentThinkAction,
    BrowseInteractiveAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    IPythonRunCellAction,
    MessageAction,
    TaskTrackingAction,
)
from openhands.events.action.agent import CondensationRequestAction
from openhands.events.action.mcp import MCPAction
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.tool import ToolCallMetadata
from openhands.llm.tool_names import TASK_TRACKER_TOOL_NAME


def combine_thought(action: Action, thought: str) -> Action:
    if not hasattr(action, 'thought'):
        return action

    action_thought = getattr(action, 'thought', None)
    if thought and action_thought:
        action.thought = f'{thought}\n{action_thought}'  # type: ignore[attr-defined]
    elif thought:
        action.thought = thought  # type: ignore[attr-defined]
    return action


def set_security_risk(action: Action, arguments: dict[str, Any]) -> None:
    """Set the security risk level for the action."""

    # Set security_risk attribute if provided
    if 'security_risk' in arguments:
        if arguments['security_risk'] in RISK_LEVELS:
            if hasattr(action, 'security_risk'):
                action.security_risk = getattr(
                    ActionSecurityRisk, arguments['security_risk']
                )
        else:
            logger.warning(f'Invalid security_risk value: {arguments["security_risk"]}')


def response_to_actions(
    response: QueryResult, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    actions: list[Action] = []
    if response.tool_calls:
        # Check if there's output_text. If so, add it to the thought
        thought = ''
        if response.output_text:
            thought = f'Output: {response.output_text}'

        if response.reasoning:
            if thought:
                thought += '\n'

            thought += f'Reasoning: {response.reasoning}'

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(response.tool_calls):
            logger.debug(f'Tool call in function_calling.py: {tool_call}')
            try:
                if isinstance(tool_call.args, str):
                    arguments = json.loads(tool_call.args)
                else:
                    arguments = tool_call.args
            except json.decoder.JSONDecodeError as e:
                raise FunctionCallValidationError(
                    f'Failed to parse tool call arguments: {tool_call.args}',
                    tool_call=tool_call,
                ) from e

            # ================================================
            # CmdRunTool (Bash)
            # ================================================

            if tool_call.name == create_cmd_run_tool().name:
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.name}',
                        tool_call=tool_call,
                    )
                # Validate command is a string
                if not isinstance(arguments['command'], str):
                    raise FunctionCallValidationError(
                        f"Expected 'command' to be a string, got {type(arguments['command']).__name__}",
                        tool_call=tool_call,
                    )
                # Validate command is a string
                if not isinstance(arguments['command'], str):
                    raise FunctionCallValidationError(
                        f"Expected 'command' to be a string, got {type(arguments['command']).__name__}",
                        tool_call=tool_call,
                    )
                # convert is_input to boolean
                is_input_arg = arguments.get('is_input', False)
                if isinstance(is_input_arg, bool):
                    is_input = is_input_arg
                elif isinstance(is_input_arg, str):
                    is_input = is_input_arg.lower() == 'true'
                else:
                    is_input = False
                action = CmdRunAction(command=arguments['command'], is_input=is_input)

                # Set hard timeout if provided
                if 'timeout' in arguments:
                    # Validate timeout is a number
                    if not isinstance(arguments['timeout'], (int, float)):
                        raise FunctionCallValidationError(
                            f"Expected 'timeout' to be a number, got {type(arguments['timeout']).__name__}",
                            tool_call=tool_call,
                        )
                    if arguments['timeout'] > 300:
                        raise FunctionCallValidationError(
                            'Timeout must be less than or equal to 300 seconds.',
                            tool_call=tool_call,
                        )
                    try:
                        action.set_hard_timeout(float(arguments['timeout']))
                    except ValueError as e:
                        raise FunctionCallValidationError(
                            f"Invalid float passed to 'timeout' argument: {arguments['timeout']}",
                            tool_call=tool_call,
                        ) from e
                set_security_risk(action, arguments)

            # ================================================
            # IPythonTool (Jupyter)
            # ================================================
            elif tool_call.name == IPythonTool.name:
                if 'code' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "code" in tool call {tool_call.name}',
                        tool_call=tool_call,
                    )
                # Validate code is a string
                if not isinstance(arguments['code'], str):
                    raise FunctionCallValidationError(
                        f"Expected 'code' to be a string, got {type(arguments['code']).__name__}",
                        tool_call=tool_call,
                    )
                # Validate code is a string
                if not isinstance(arguments['code'], str):
                    raise FunctionCallValidationError(
                        f"Expected 'code' to be a string, got {type(arguments['code']).__name__}",
                        tool_call=tool_call,
                    )
                action = IPythonRunCellAction(code=arguments['code'])
                set_security_risk(action, arguments)

            # ================================================
            # AgentDelegateAction (Delegation to another agent)
            # ================================================
            elif tool_call.name == 'delegate_to_browsing_agent':
                action = AgentDelegateAction(
                    agent='BrowsingAgent',
                    inputs=arguments,
                )

            # ================================================
            # AgentFinishAction
            # ================================================
            elif tool_call.name == FinishTool.name:
                message = arguments.get('message', '')
                # Validate message is a string if provided
                if message and not isinstance(message, str):
                    raise FunctionCallValidationError(
                        f"Expected 'message' to be a string, got {type(message).__name__}",
                        tool_call=tool_call,
                    )
                action = AgentFinishAction(
                    final_thought=message,
                )

            # ================================================
            # LLMBasedFileEditTool (LLM-based file editor, deprecated)
            # ================================================
            elif tool_call.name == LLMBasedFileEditTool.name:
                if 'path' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "path" in tool call {tool_call.name}',
                        tool_call=tool_call,
                    )
                if 'content' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "content" in tool call {tool_call.name}',
                        tool_call=tool_call,
                    )
                # Validate path is a string
                if not isinstance(arguments['path'], str):
                    raise FunctionCallValidationError(
                        f"Expected 'path' to be a string, got {type(arguments['path']).__name__}",
                        tool_call=tool_call,
                    )
                # Validate content is a string
                if not isinstance(arguments['content'], str):
                    raise FunctionCallValidationError(
                        f"Expected 'content' to be a string, got {type(arguments['content']).__name__}",
                        tool_call=tool_call,
                    )
                # Validate start is an integer if provided
                start = arguments.get('start', 1)
                if not isinstance(start, int):
                    raise FunctionCallValidationError(
                        f"Expected 'start' to be an integer, got {type(start).__name__}",
                        tool_call=tool_call,
                    )
                # Validate end is an integer if provided
                end = arguments.get('end', -1)
                if not isinstance(end, int):
                    raise FunctionCallValidationError(
                        f"Expected 'end' to be an integer, got {type(end).__name__}",
                        tool_call=tool_call,
                    )
                # Validate path is a string
                if not isinstance(arguments['path'], str):
                    raise FunctionCallValidationError(
                        f"Expected 'path' to be a string, got {type(arguments['path']).__name__}",
                        tool_call=tool_call,
                    )
                # Validate content is a string
                if not isinstance(arguments['content'], str):
                    raise FunctionCallValidationError(
                        f"Expected 'content' to be a string, got {type(arguments['content']).__name__}",
                        tool_call=tool_call,
                    )
                # Validate start is an integer if provided
                start = arguments.get('start', 1)
                if not isinstance(start, int):
                    raise FunctionCallValidationError(
                        f"Expected 'start' to be an integer, got {type(start).__name__}",
                        tool_call=tool_call,
                    )
                # Validate end is an integer if provided
                end = arguments.get('end', -1)
                if not isinstance(end, int):
                    raise FunctionCallValidationError(
                        f"Expected 'end' to be an integer, got {type(end).__name__}",
                        tool_call=tool_call,
                    )
                action = FileEditAction(
                    path=arguments['path'],
                    content=arguments['content'],
                    start=start,
                    end=end,
                    impl_source=arguments.get(
                        'impl_source', FileEditSource.LLM_BASED_EDIT
                    ),
                )
            elif tool_call.name == create_str_replace_editor_tool().name:
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.name}',
                        tool_call=tool_call,
                    )
                if 'path' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "path" in tool call {tool_call.name}',
                        tool_call=tool_call,
                    )

                # Validate command is a string and is a valid enum value
                command = arguments['command']
                if not isinstance(command, str):
                    raise FunctionCallValidationError(
                        f"Expected 'command' to be a string, got {type(command).__name__}",
                        tool_call=tool_call,
                    )
                VALID_COMMANDS = [
                    'view',
                    'create',
                    'str_replace',
                    'insert',
                    'undo_edit',
                ]
                if command not in VALID_COMMANDS:
                    raise FunctionCallValidationError(
                        f"Invalid command '{command}'. Must be one of {VALID_COMMANDS}",
                        tool_call=tool_call,
                    )

                # Validate path is a string
                path = arguments['path']
                if not isinstance(path, str):
                    raise FunctionCallValidationError(
                        f"Expected 'path' to be a string, got {type(path).__name__}",
                        tool_call=tool_call,
                    )

                other_kwargs = {
                    k: v for k, v in arguments.items() if k not in ['command', 'path']
                }

                if command == 'view':
                    # Validate view_range if provided
                    view_range = other_kwargs.get('view_range', None)
                    if view_range is not None:
                        if not isinstance(view_range, list):
                            raise FunctionCallValidationError(
                                f"Expected 'view_range' to be a list, got {type(view_range).__name__}",
                                tool_call=tool_call,
                            )
                        if len(view_range) != 2:
                            raise FunctionCallValidationError(
                                f"Expected 'view_range' to have exactly 2 elements, got {len(view_range)}",
                                tool_call=tool_call,
                            )
                        if not all(isinstance(x, int) for x in view_range):
                            raise FunctionCallValidationError(
                                f"Expected 'view_range' elements to be integers, got {[type(x).__name__ for x in view_range]}",
                                tool_call=tool_call,
                            )
                    action = FileReadAction(
                        path=path,
                        impl_source=FileReadSource.OH_ACI,
                        view_range=view_range,
                    )
                else:
                    if 'view_range' in other_kwargs:
                        # Remove view_range from other_kwargs since it is not needed for FileEditAction
                        other_kwargs.pop('view_range')

                    # Validate type-specific parameters based on command
                    if 'file_text' in other_kwargs:
                        if not isinstance(other_kwargs['file_text'], str):
                            raise FunctionCallValidationError(
                                f"Expected 'file_text' to be a string, got {type(other_kwargs['file_text']).__name__}",
                                tool_call=tool_call,
                            )

                    if 'old_str' in other_kwargs:
                        if not isinstance(other_kwargs['old_str'], str):
                            raise FunctionCallValidationError(
                                f"Expected 'old_str' to be a string, got {type(other_kwargs['old_str']).__name__}",
                                tool_call=tool_call,
                            )

                    if 'new_str' in other_kwargs:
                        if not isinstance(other_kwargs['new_str'], str):
                            raise FunctionCallValidationError(
                                f"Expected 'new_str' to be a string, got {type(other_kwargs['new_str']).__name__}",
                                tool_call=tool_call,
                            )

                    if 'insert_line' in other_kwargs:
                        if not isinstance(other_kwargs['insert_line'], int):
                            raise FunctionCallValidationError(
                                f"Expected 'insert_line' to be an integer, got {type(other_kwargs['insert_line']).__name__}",
                                tool_call=tool_call,
                            )

                    # Filter out unexpected arguments
                    valid_kwargs_for_editor = {}
                    # Get valid parameters from the str_replace_editor tool definition
                    str_replace_editor_tool = create_str_replace_editor_tool()
                    valid_params = set(str_replace_editor_tool.body.properties.keys())

                    for key, value in other_kwargs.items():
                        if key in valid_params:
                            # security_risk is valid but should NOT be part of editor kwargs
                            if key != 'security_risk':
                                valid_kwargs_for_editor[key] = value
                        else:
                            raise FunctionCallValidationError(
                                f'Unexpected argument {key} in tool call {tool_call.name}. Allowed arguments are: {valid_params}',
                                tool_call=tool_call,
                            )

                    action = FileEditAction(
                        path=path,
                        command=command,
                        impl_source=FileEditSource.OH_ACI,
                        **valid_kwargs_for_editor,
                    )

                set_security_risk(action, arguments)
            # ================================================
            # AgentThinkAction
            # ================================================
            elif tool_call.name == ThinkTool.name:
                thought = arguments.get('thought', '')
                # Validate thought is a string if provided
                if thought and not isinstance(thought, str):
                    raise FunctionCallValidationError(
                        f"Expected 'thought' to be a string, got {type(thought).__name__}",
                        tool_call=tool_call,
                    )
                action = AgentThinkAction(thought=thought)

            # ================================================
            # CondensationRequestAction
            # ================================================
            elif tool_call.name == CondensationRequestTool.name:
                action = CondensationRequestAction()

            # ================================================
            # BrowserTool
            # ================================================
            elif tool_call.name == BrowserTool.name:
                if 'code' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "code" in tool call {tool_call.name}',
                        tool_call=tool_call,
                    )
                # Validate code is a string
                if not isinstance(arguments['code'], str):
                    raise FunctionCallValidationError(
                        f"Expected 'code' to be a string, got {type(arguments['code']).__name__}",
                        tool_call=tool_call,
                    )
                # Validate code is a string
                if not isinstance(arguments['code'], str):
                    raise FunctionCallValidationError(
                        f"Expected 'code' to be a string, got {type(arguments['code']).__name__}",
                        tool_call=tool_call,
                    )
                action = BrowseInteractiveAction(browser_actions=arguments['code'])
                set_security_risk(action, arguments)

            # ================================================
            # TaskTrackingAction
            # ================================================
            elif tool_call.name == TASK_TRACKER_TOOL_NAME:
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.name}',
                        tool_call=tool_call,
                    )
                if arguments['command'] == 'plan' and 'task_list' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "task_list" for "plan" command in tool call {tool_call.name}',
                        tool_call=tool_call,
                    )

                raw_task_list = arguments.get('task_list', [])
                if not isinstance(raw_task_list, list):
                    raise FunctionCallValidationError(
                        f'Invalid format for "task_list". Expected a list but got {type(raw_task_list)}.',
                        tool_call=tool_call,
                    )

                # Normalize task_list to ensure it's always a list of dictionaries
                normalized_task_list: list[dict[str, Any]] = []
                for i, task in enumerate(raw_task_list):
                    if isinstance(task, dict):
                        task = cast(dict[str, Any], task)
                        # Task is already in correct format, ensure required fields exist
                        normalized_task = {
                            'id': task.get('id', f'task-{i + 1}'),
                            'title': task.get('title', 'Untitled task'),
                            'status': task.get('status', 'todo'),
                            'notes': task.get('notes', ''),
                        }
                    else:
                        # Unexpected format, raise validation error
                        logger.warning(
                            f'Unexpected task format in task_list: {type(task)} - {task}'
                        )
                        raise FunctionCallValidationError(
                            f'Unexpected task format in task_list: {type(task)}. Each task shoud be a dictionary.',
                            tool_call=tool_call,
                        )
                    normalized_task_list.append(normalized_task)

                action = TaskTrackingAction(
                    command=arguments['command'],
                    task_list=normalized_task_list,
                )

            # ================================================
            # MCPAction (MCP)
            # ================================================
            elif mcp_tool_names and tool_call.name in mcp_tool_names:
                action = MCPAction(
                    name=tool_call.name,
                    arguments=arguments,
                )
            else:
                raise FunctionCallNotExistsError(
                    f'Tool {tool_call.name} is not registered. (arguments: {arguments}). Please check the tool name and retry with an existing tool.',
                    tool_call=tool_call,
                )

            # We only add thought to the first action
            if i == 0:
                action = combine_thought(action, thought)
            # Add metadata for tool calling
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=tool_call.name,
                model_response=response,
                total_calls_in_response=len(response.tool_calls),
            )
            actions.append(action)
    else:
        actions.append(
            MessageAction(
                content=str(response.output_text) if response.output_text else '',
                wait_for_response=True,
            )
        )

    # Add response id to actions
    # This will ensure we can match both actions without tool calls (e.g. MessageAction)
    # and actions with tool calls (e.g. CmdRunAction, IPythonRunCellAction, etc.)
    # with the token usage data
    for action in actions:
        response_id = response.raw.get('id') if response.raw else None
        action.response_id = response_id

    assert len(actions) >= 1
    return actions
