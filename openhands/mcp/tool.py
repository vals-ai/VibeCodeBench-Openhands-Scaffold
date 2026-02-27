from typing import ClassVar

from mcp.types import Tool
from model_library.base import ToolBody, ToolDefinition
from pydantic import ConfigDict


class MCPClientTool(Tool):
    """Represents a tool proxy that can be called on the MCP server from the client side.

    This version doesn't store a session reference, as sessions are created on-demand
    by the MCPClient for each operation.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def to_tool_definition(self) -> ToolDefinition:
        """Convert tool to ToolDefinition format."""
        # Extract properties and required fields from inputSchema
        properties = self.inputSchema.get('properties', {})
        required = self.inputSchema.get('required', [])

        return ToolDefinition(
            name=self.name,
            body=ToolBody(
                name=self.name,
                description=self.description or '',
                properties=properties,
                required=required,
                kwargs={},
            ),
        )
