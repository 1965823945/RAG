"""Tool registry with metadata and export support."""

from typing import Any

from rag_minimal.schemas import ToolMetadata
from rag_minimal.tools.base import Tool


class ToolRegistry:
    """Registry for standardized tools.

    Features:
    - Register and retrieve tools by name
    - List all tools with metadata
    - Export tools as OpenAI functions or MCP format
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Tool:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance

        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {self.list_tools()}")
        return self._tools[name]

    def get_optional(self, name: str) -> Tool | None:
        """Get a tool by name, returning None if not found."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_metadata(self) -> list[ToolMetadata]:
        """Get metadata for all registered tools.

        Returns:
            List of ToolMetadata objects
        """
        return [tool.get_metadata() for tool in self._tools.values()]

    def get_metadata(self, name: str) -> ToolMetadata:
        """Get metadata for a specific tool.

        Args:
            name: Tool name

        Returns:
            ToolMetadata for the tool
        """
        return self.get(name).get_metadata()

    def find_by_tag(self, tag: str) -> list[Tool]:
        """Find tools by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of tools with the specified tag
        """
        return [
            tool
            for tool in self._tools.values()
            if hasattr(tool, "tags") and tag in tool.tags
        ]

    def export_openai_functions(self) -> list[dict[str, Any]]:
        """Export all tools as OpenAI function definitions.

        Returns:
            List of dicts compatible with OpenAI function calling
        """
        return [tool.to_openai_function() for tool in self._tools.values()]

    def export_mcp_tools(self) -> list[dict[str, Any]]:
        """Export all tools as MCP tool definitions.

        Returns:
            List of dicts compatible with Model Context Protocol
        """
        return [tool.to_mcp_tool() for tool in self._tools.values()]

    def export_tool_descriptions(self) -> str:
        """Export human-readable tool descriptions.

        Returns:
            Formatted string with all tool descriptions
        """
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool.description}")
            if hasattr(tool, "tags") and tool.tags:
                lines.append(f"  Tags: {', '.join(tool.tags)}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
