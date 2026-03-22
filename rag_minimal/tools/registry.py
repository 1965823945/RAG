"""Tool registry."""

from typing import Dict, List
from rag_minimal.tools.base import Tool


class ToolRegistry:
    """Registry for standardized tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
