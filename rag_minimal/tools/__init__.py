"""Tools package - Standardized tool interface for RAG system."""

from rag_minimal.tools.base import Tool
from rag_minimal.tools.examples import (
    CalculatorTool,
    EchoTool,
    ListAggregatorTool,
    TextTransformTool,
)
from rag_minimal.tools.knowledge_search import KnowledgeSearchTool
from rag_minimal.tools.logger import ToolLogger, get_tool_logger, logged_invoke
from rag_minimal.tools.registry import ToolRegistry

__all__ = [
    # Base
    "Tool",
    "ToolRegistry",
    # Core tools
    "KnowledgeSearchTool",
    # Example tools
    "CalculatorTool",
    "EchoTool",
    "TextTransformTool",
    "ListAggregatorTool",
    # Logging
    "ToolLogger",
    "get_tool_logger",
    "logged_invoke",
]
