"""RAG - Minimal Retrieval-Augmented Generation Demo.

This package provides a pluggable RAG system with:
- Standardized tool interface
- Agent runtime for tool orchestration
- MCP server for external integration
- Autonomous Planning Agent with:
  - Task Decomposition
  - Chain of Thought reasoning
  - Self-Reflection
- Conversational AI with:
  - Multi-turn dialogue support
  - Memory system (short-term, long-term, working memory)
- Unified configuration management
"""

from rag_minimal.agent_runtime import AgentRuntime
from rag_minimal.config import (
    RAGSettings,
    configure,
    get_settings,
    reset_settings,
    set_settings,
)
from rag_minimal.memory import (
    ConversationalAgent,
    ConversationManager,
    LongTermMemory,
    MemorySystem,
    ShortTermMemory,
    WorkingMemory,
)
from rag_minimal.planning import (
    ChainOfThought,
    PlanningAgent,
    SelfReflection,
    TaskDecomposer,
)
from rag_minimal.schemas import (
    ChainOfThoughtResult,
    Conversation,
    ConversationalInput,
    ConversationalOutput,
    ConversationContext,
    ErrorCode,
    MemoryEntry,
    MemorySearchResult,
    MemoryType,
    Message,
    # Memory schemas
    MessageRole,
    PlanningAgentOutput,
    PlanningAgentState,
    RAGOutput,
    ReflectionItem,
    ReflectionType,
    SearchOutput,
    SelfReflectionResult,
    SubTask,
    TaskDecomposition,
    TaskPriority,
    # Planning Agent schemas
    TaskStatus,
    ThoughtStep,
    ToolMetadata,
    ToolOutput,
)
from rag_minimal.tools import (
    KnowledgeSearchTool,
    Tool,
    ToolLogger,
    ToolRegistry,
)

__all__ = [
    # Schemas
    "ErrorCode",
    "ToolOutput",
    "SearchOutput",
    "RAGOutput",
    "ToolMetadata",
    # Planning Schemas
    "TaskStatus",
    "TaskPriority",
    "SubTask",
    "TaskDecomposition",
    "ThoughtStep",
    "ChainOfThoughtResult",
    "ReflectionType",
    "ReflectionItem",
    "SelfReflectionResult",
    "PlanningAgentState",
    "PlanningAgentOutput",
    # Memory Schemas
    "MessageRole",
    "Message",
    "Conversation",
    "MemoryType",
    "MemoryEntry",
    "MemorySearchResult",
    "ConversationContext",
    "ConversationalInput",
    "ConversationalOutput",
    # Tools
    "Tool",
    "ToolRegistry",
    "KnowledgeSearchTool",
    "ToolLogger",
    # Runtime
    "AgentRuntime",
    # Planning Agent
    "TaskDecomposer",
    "ChainOfThought",
    "SelfReflection",
    "PlanningAgent",
    # Memory & Conversation
    "ConversationManager",
    "MemorySystem",
    "ShortTermMemory",
    "LongTermMemory",
    "WorkingMemory",
    "ConversationalAgent",
    # Configuration
    "RAGSettings",
    "get_settings",
    "set_settings",
    "reset_settings",
    "configure",
]

__version__ = "0.4.0"
