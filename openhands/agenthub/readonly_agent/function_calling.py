"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import shlex

from model_library.base import QueryResult, ToolBody, ToolDefinition

from openhands.agenthub.codeact_agent.function_calling import (
    combine_thought,
)
from openhands.agenthub.codeact_agent.tools import (
    FinishTool,
    ThinkTool,
)
from openhands.agenthub.readonly_agent.tools import (
    GlobTool,
    GrepTool,
    ViewTool,
)
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentFinishAction,
    AgentThinkAction,
    CmdRunAction,
    FileReadAction,
    MCPAction,
    MessageAction,
)
from openhands.events.event import FileReadSource
from openhands.events.tool import ToolCallMetadata


def grep_to_cmdrun(
    pattern: str, path: str | None = None, include: str | None = None
) -> str:
    # NOTE: This function currently relies on `rg` (ripgrep).
    # `rg` may not be installed when using CLIRuntime or LocalRuntime.
    # TODO: Implement a fallback to `grep` if `rg` is not available.
    """Convert grep tool arguments to a shell command string.

    Args:
        pattern: The regex pattern to search for in file contents
        path: The directory to search in (optional)
        include: Optional file pattern to filter which files to search (e.g., "*.js")

    Returns:
        A properly escaped shell command string for ripgrep
    """
    # Use shlex.quote to properly escape all shell special characters
    quoted_pattern = shlex.quote(pattern)
    path_arg = shlex.quote(path) if path else '.'

    # Build ripgrep command
    rg_cmd = f'rg -li {quoted_pattern} --sortr=modified'

    if include:
        quoted_include = shlex.quote(include)
        rg_cmd += f' --glob {quoted_include}'

    # Build the complete command
    complete_cmd = f'{rg_cmd} {path_arg} | head -n 100'

    # Add a header to the output
    echo_cmd = f'echo "Below are the execution results of the search command: {complete_cmd}\n"; '
    return echo_cmd + complete_cmd


def glob_to_cmdrun(pattern: str, path: str = '.') -> str:
    # NOTE: This function currently relies on `rg` (ripgrep).
    # `rg` may not be installed when using CLIRuntime or LocalRuntime
    # TODO: Implement a fallback to `find` if `rg` is not available.
    """Convert glob tool arguments to a shell command string.

    Args:
        pattern: The glob pattern to match files (e.g., "**/*.js")
        path: The directory to search in (defaults to current directory)

    Returns:
        A properly escaped shell command string for ripgrep implementing glob
    """
    # Use shlex.quote to properly escape all shell special characters
    quoted_path = shlex.quote(path)
    quoted_pattern = shlex.quote(pattern)

    # Use ripgrep in a glob-only mode with -g flag and --files to list files
    # This most closely matches the behavior of the NodeJS glob implementation
    rg_cmd = f'rg --files {quoted_path} -g {quoted_pattern} --sortr=modified'

    # Sort results and limit to 100 entries (matching the Node.js implementation)
    sort_and_limit_cmd = ' | head -n 100'

    complete_cmd = f'{rg_cmd}{sort_and_limit_cmd}'

    # Add a header to the output
    echo_cmd = f'echo "Below are the execution results of the glob command: {complete_cmd}\n"; '
    return echo_cmd + complete_cmd


def response_to_actions(
    response: QueryResult, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    actions: list[Action] = []
    if response.tool_calls:
        thought = response.output_text or response.reasoning or ''

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(response.tool_calls):
            action: Action
            logger.debug(f'Tool call in function_calling.py: {tool_call}')
            arguments = tool_call.parsed_args
            if arguments is None:
                raise FunctionCallValidationError(
                    f'Failed to parse tool call arguments: {tool_call.args}',
                    tool_call=tool_call,
                )

            # ================================================
            # AgentFinishAction
            # ================================================
            if tool_call.name == FinishTool.name:
                action = AgentFinishAction(
                    final_thought=arguments.get('message', ''),
                )

            # ================================================
            # ViewTool (ACI-based file viewer, READ-ONLY)
            # ================================================
            elif tool_call.name == ViewTool['function']['name']:
                if 'path' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "path" in tool call {tool_call.name}'
                    )
                action = FileReadAction(
                    path=arguments['path'],
                    impl_source=FileReadSource.OH_ACI,
                    view_range=arguments.get('view_range', None),
                )

            # ================================================
            # AgentThinkAction
            # ================================================
            elif tool_call.name == ThinkTool.name:
                action = AgentThinkAction(thought=arguments.get('thought', ''))

            # ================================================
            # GrepTool (file content search)
            # ================================================
            elif tool_call.name == GrepTool['function']['name']:
                if 'pattern' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "pattern" in tool call {tool_call.name}'
                    )

                pattern = arguments['pattern']
                path = arguments.get('path')
                include = arguments.get('include')

                grep_cmd = grep_to_cmdrun(pattern, path, include)
                action = CmdRunAction(command=grep_cmd, is_input=False)

            # ================================================
            # GlobTool (file pattern matching)
            # ================================================
            elif tool_call.name == GlobTool['function']['name']:
                if 'pattern' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "pattern" in tool call {tool_call.name}'
                    )

                pattern = arguments['pattern']
                path = arguments.get('path', '.')

                glob_cmd = glob_to_cmdrun(pattern, path)
                action = CmdRunAction(command=glob_cmd, is_input=False)

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
                    f'Tool {tool_call.name} is not registered. (arguments: {arguments}). Please check the tool name and retry with an existing tool.'
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
                content=response.output_text or '',
                wait_for_response=True,
            )
        )

    # Add response id to actions
    # This will ensure we can match both actions without tool calls (e.g. MessageAction)
    # and actions with tool calls (e.g. CmdRunAction, IPythonRunCellAction, etc.)
    # with the token usage data
    for action in actions:
        if response.extras.response_id is not None:
            action.response_id = response.extras.response_id

    assert len(actions) >= 1
    return actions


def get_tools() -> list[ToolDefinition]:
    tools = [ThinkTool, FinishTool]
    legacy_tools = [GrepTool, GlobTool, ViewTool]
    for tool in legacy_tools:
        function = tool['function']
        tools.append(
            ToolDefinition(
                name=function['name'],
                body=ToolBody(
                    name=function['name'],
                    description=function['description'],
                    properties=function['parameters']['properties'],
                    required=function['parameters']['required'],
                ),
            )
        )
    return tools
