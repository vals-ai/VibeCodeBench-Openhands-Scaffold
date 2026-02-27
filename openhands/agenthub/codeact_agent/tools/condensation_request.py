from model_library.base import ToolBody, ToolDefinition

_CONDENSATION_REQUEST_DESCRIPTION = 'Request a condensation of the conversation history when the context becomes too long or when you need to focus on the most relevant information.'

CondensationRequestTool = ToolDefinition(
    name='request_condensation',
    body=ToolBody(
        name='request_condensation',
        description=_CONDENSATION_REQUEST_DESCRIPTION,
        properties={},
        required=[],
    ),
)
