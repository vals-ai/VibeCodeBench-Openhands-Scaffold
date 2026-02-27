from model_library.base import ToolBody, ToolDefinition

from openhands.llm.tool_names import FINISH_TOOL_NAME

_FINISH_DESCRIPTION = """You MUST call this tool to complete the task.

When you have successfully completed the user's requested task, you MUST call this tool with your final message. This is the ONLY way to properly finish the interaction.

DO NOT just send a text message saying you're done. You MUST call this tool.

The message parameter should include:
- A clear summary of actions taken and their results
- Explanation if you're unable to complete the task
- Any errors you encountered while completing the task
- Any parts of the task that were not completed

After using this tool, you will no longer be able to change the code or run commands. The interaction will end.

You should not call this tool if there are still errors testing or running the application, since this will mean that there will be errors in the final product and the application will be not be functional.
"""

FinishTool = ToolDefinition(
    name=FINISH_TOOL_NAME,
    body=ToolBody(
        name=FINISH_TOOL_NAME,
        description=_FINISH_DESCRIPTION,
        properties={
            'message': {
                'type': 'string',
                'description': 'Final message to send to the user',
            },
        },
        required=['message'],
    ),
)
