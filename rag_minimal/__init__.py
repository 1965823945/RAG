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

from rag_minimal.schemas import (
    ErrorCode,
    ToolOutput,
    SearchOutput,
    RAGOutput,
    ToolMetadata,
    # Planning Agent schemas
    TaskStatus,
    TaskPriority,
    SubTask,
    TaskDecomposition,
    ThoughtStep,
    ChainOfThoughtResult,
    ReflectionType,
    ReflectionItem,
    SelfReflectionResult,
    PlanningAgentState,
    PlanningAgentOutput,
    # Memory schemas
    MessageRole,
    Message,
    Conversation,
    MemoryType,
    MemoryEntry,
    MemorySearchResult,
    ConversationContext,
    ConversationalInput,
    ConversationalOutput,
)
from rag_minimal.tools import (
    Tool,
    ToolRegistry,
    KnowledgeSearchTool,
    ToolLogger,
)
from rag_minimal.agent_runtime import AgentRuntime
from rag_minimal.planning import (
    TaskDecomposer,
    ChainOfThought,
    SelfReflection,
    PlanningAgent,
)
from rag_minimal.memory import (
    ConversationManager,
    MemorySystem,
    ShortTermMemory,
    LongTermMemory,
    WorkingMemory,
    ConversationalAgent,
)
from rag_minimal.config import (
    RAGSettings,
    get_settings,
    set_settings,
    reset_settings,
    configure,
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
