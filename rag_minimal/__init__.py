"""RAG - Minimal Retrieval-Augmented Generation Demo.

This package provides a pluggable RAG system with:
- Standardized tool interface
- Agent runtime for tool orchestration
- MCP server for external integration
"""

from rag_minimal.schemas import (
    ErrorCode,
    ToolOutput,
    SearchOutput,
    RAGOutput,
    ToolMetadata,
)
from rag_minimal.tools import (
    Tool,
    ToolRegistry,
    KnowledgeSearchTool,
    ToolLogger,
)
from rag_minimal.agent_runtime import AgentRuntime

__all__ = [
    # Schemas
    "ErrorCode",
    "ToolOutput",
    "SearchOutput",
    "RAGOutput",
    "ToolMetadata",
    # Tools
    "Tool",
    "ToolRegistry",
    "KnowledgeSearchTool",
    "ToolLogger",
    # Runtime
    "AgentRuntime",
]

__version__ = "0.1.0"
