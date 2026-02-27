from typing import Any

from model_library.base import QueryResult
from pydantic import BaseModel, field_serializer


class ToolCallMetadata(BaseModel):
    # See https://docs.litellm.ai/docs/completion/function_call#step-3---second-litellmcompletion-call
    function_name: str  # Name of the function that was called
    tool_call_id: str  # ID of the tool call

    model_response: QueryResult
    total_calls_in_response: int

    @field_serializer("model_response")
    def serialize_model_response(
        self, model_response: QueryResult, _info: Any
    ) -> dict[str, Any]:
        response_dict = model_response.model_dump(exclude={"history", "raw"})

        return response_dict
