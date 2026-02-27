from dataclasses import dataclass

from model_library.base import ToolCall

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class ErrorObservation(Observation):
    """This data class represents an error encountered by the agent.

    This is the type of error that LLM can recover from.
    E.g., Linter error after editing a file.
    """

    observation: str = ObservationType.ERROR
    error_id: str = ''
    tool_call: ToolCall | None = None

    @property
    def message(self) -> str:
        return self.content

    @property
    def tool_call_id(self) -> str | None:
        if self.tool_call is None:
            return None

        if isinstance(self.tool_call, ToolCall):
            return self.tool_call.id

        return self.tool_call['id']

    def __str__(self) -> str:
        return f'**ErrorObservation**\n{self.content}'
