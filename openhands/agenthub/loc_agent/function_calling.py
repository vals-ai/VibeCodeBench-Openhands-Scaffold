"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from model_library.base import QueryResult, ToolBody, ToolDefinition

from openhands.agenthub.codeact_agent.function_calling import combine_thought
from openhands.agenthub.codeact_agent.tools import FinishTool
from openhands.agenthub.loc_agent.tools import (
    SearchEntityTool,
    SearchRepoTool,
    create_explore_tree_structure_tool,
)
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentFinishAction,
    IPythonRunCellAction,
    MessageAction,
)
from openhands.events.tool import ToolCallMetadata


def response_to_actions(
    response: QueryResult,
    mcp_tool_names: list[str] | None = None,
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
            # LocAgent's Tools
            # ================================================
            ALL_FUNCTIONS = [
                'explore_tree_structure',
                'search_code_snippets',
                'get_entity_contents',
            ]
            if tool_call.name in ALL_FUNCTIONS:
                # We implement this in agent_skills, which can be used via Jupyter
                func_name = tool_call.name
                code = f'print({func_name}(**{arguments}))'
                logger.debug(f'TOOL CALL: {func_name} with code: {code}')
                action = IPythonRunCellAction(code=code)

            # ================================================
            # AgentFinishAction
            # ================================================
            elif tool_call.name == FinishTool.name:
                action = AgentFinishAction(
                    final_thought=arguments.get('message', ''),
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
    tools = [FinishTool]
    legacy_tools = [
        SearchRepoTool,
        SearchEntityTool,
        create_explore_tree_structure_tool(use_simplified_description=True),
    ]
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
