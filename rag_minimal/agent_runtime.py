"""Minimal agent runtime using standardized tools."""

from typing import Any, Dict

from rag_minimal.schemas import SearchOutput
from rag_minimal.tools.knowledge_search import KnowledgeSearchTool
from rag_minimal.tools.registry import ToolRegistry


class AgentRuntime:
    """A minimal standardized agent runtime."""

    def __init__(self, docs_dir: str = "docs"):
        self.registry = ToolRegistry()
        self.registry.register(KnowledgeSearchTool(docs_dir=docs_dir))

    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Run the agent on a query."""
        tool = self.registry.get("knowledge_search")
        result = tool.invoke({"query": query, "top_k": top_k})

        if isinstance(result, SearchOutput):
            return result.model_dump()
        return {"success": False, "message": "unexpected result"}
