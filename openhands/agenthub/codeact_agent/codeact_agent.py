import os
from collections import deque
from typing import override

from model_library.base import InputItem, QueryResult, ToolDefinition
from model_library.exceptions import MaxContextWindowExceededError

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.agenthub.codeact_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.codeact_agent.tools.browser import BrowserTool
from openhands.agenthub.codeact_agent.tools.condensation_request import (
    CondensationRequestTool,
)
from openhands.agenthub.codeact_agent.tools.finish import FinishTool
from openhands.agenthub.codeact_agent.tools.ipython import IPythonTool
from openhands.agenthub.codeact_agent.tools.llm_based_edit import LLMBasedFileEditTool
from openhands.agenthub.codeact_agent.tools.str_replace_editor import (
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.task_tracker import (
    create_task_tracker_tool,
)
from openhands.agenthub.codeact_agent.tools.think import ThinkTool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.events.action import Action
from openhands.llm.condenser import HistoryWindowCondenser
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager


class CodeActAgent(Agent):
    VERSION: str = '2.2'
    """
    The Code Act Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents' **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/All-Hands-AI/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - config (AgentConfig): The configuration for this agent
        """
        super().__init__(config, llm_registry)
        self.pending_actions: deque['Action'] = deque()
        self.reset()
        self.tools: list[ToolDefinition] = self._get_tools()

        self.condenser: HistoryWindowCondenser = HistoryWindowCondenser()

        # Override with router if needed
        self.llm: LLM = self.llm_registry.get_router(self.config)
        self.history: list[InputItem] = []

    @property
    @override
    def prompt_manager(self) -> 'PromptManager':
        if self._prompt_manager:
            return self._prompt_manager

        self._prompt_manager: PromptManager | None = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
            system_prompt_filename=self.config.resolved_system_prompt_filename,
        )

        return self._prompt_manager

    def _get_tools(self) -> list[ToolDefinition]:
        use_short_tool_desc = False

        tools: list[ToolDefinition] = []
        if self.config.enable_cmd:
            tools.append(create_cmd_run_tool(use_short_description=False))
        if self.config.enable_think:
            tools.append(ThinkTool)
        if self.config.enable_finish:
            tools.append(FinishTool)
        if self.config.enable_condensation_request:
            tools.append(CondensationRequestTool)
        if self.config.enable_browsing:
            tools.append(BrowserTool)
        if self.config.enable_jupyter:
            tools.append(IPythonTool)
        if self.config.enable_plan_mode:
            # In plan mode, we use the task_tracker tool for task management
            tools.append(create_task_tracker_tool(use_short_tool_desc))
        if self.config.enable_llm_editor:
            tools.append(LLMBasedFileEditTool)
        elif self.config.enable_editor:
            tools.append(
                create_str_replace_editor_tool(
                    use_short_description=use_short_tool_desc,
                    runtime_type=self.config.runtime,
                )
            )
        return tools

    @override
    def reset(self) -> None:
        """Resets the CodeAct Agent's internal state."""
        super().reset()
        # Only clear pending actions, not LLM metrics
        self.pending_actions.clear()

    def _query(self, inputs: list[InputItem], history: list[InputItem]) -> QueryResult:
        try:
            return self.llm.completion(
                input=inputs,
                history=history,
                tools=self.tools,
            )
        except MaxContextWindowExceededError:
            condensed_history = self.condenser.condense_history(history)
            self.llm.pretty_print(
                f'Condensed history from {len(history)} to {len(condensed_history)} items. Removed {len(history) - len(condensed_history)} items.'
            )
            return self.llm.completion(
                input=inputs,
                history=condensed_history,
                tools=self.tools,
            )
        except Exception as e:
            raise e

    @override
    def step(self, state: State) -> 'Action':
        """Performs one step using the CodeAct Agent.

        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - CmdRunAction(command) - bash command to run
        - IPythonRunCellAction(code) - IPython code to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        - CondensationAction(...) - condense conversation history by forgetting specified events and optionally providing a summary
        - FileReadAction(path, ...) - read file content from specified path
        - FileEditAction(path, ...) - edit file using LLM-based (deprecated) or ACI-based editing
        - AgentThinkAction(thought) - log agent's thought/reasoning process
        - CondensationRequestAction() - request condensation of conversation history
        - BrowseInteractiveAction(browser_actions) - interact with browser using specified actions
        - MCPAction(name, arguments) - interact with MCP server tools
        """
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        self.llm.pretty_print('[Input Items]')
        for input_item in state.inputs:
            self.llm.pretty_print(input_item)

        response = self._query(
            state.inputs,
            self.history,
        )

        response_text = response.output_text or response.reasoning
        if response_text:
            self.llm.pretty_print('[Output Response]')
            self.llm.pretty_print(response_text)

        if response.tool_calls:
            self.llm.pretty_print('[Tool Calls]')
            for tool_call in response.tool_calls:
                self.llm.pretty_print(tool_call)

        self.history = response.history

        state.flush()

        if response.tool_calls:
            for tool_call in response.tool_calls:
                state.pending_tool_calls[tool_call.id] = tool_call

        actions = self.response_to_actions(response)

        actions_with_tool_calls = [
            action for action in actions if action.tool_call_metadata
        ]

        assert len(actions_with_tool_calls) == len(response.tool_calls), (
            'Tool call count does not match action count'
        )

        if not actions_with_tool_calls:
            return actions[0]

        for action in actions_with_tool_calls:
            self.pending_actions.append(action)

        return self.pending_actions.popleft()

    def response_to_actions(self, response: 'QueryResult') -> list['Action']:
        return codeact_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )
