"""Shared schemas for standardized tool interfaces.

This module re-exports all schemas from the schemas package for backward compatibility.
New code should import from rag_minimal.schemas (which resolves to the package).
"""

# Re-export everything from the schemas package
from rag_minimal.schemas.base import ErrorCode, ToolInput, ToolOutput
from rag_minimal.schemas.memory import (
    Conversation,
    ConversationalInput,
    ConversationalOutput,
    ConversationContext,
    MemoryEntry,
    MemorySearchResult,
    MemoryType,
    Message,
    MessageRole,
)
from rag_minimal.schemas.planning import (
    ChainOfThoughtResult,
    PlanningAgentOutput,
    PlanningAgentState,
    ReflectionItem,
    ReflectionType,
    SelfReflectionResult,
    SubTask,
    TaskDecomposition,
    TaskPriority,
    TaskStatus,
    ThoughtStep,
)
from rag_minimal.schemas.tools import (
    MultiToolInput,
    MultiToolOutput,
    RAGInput,
    RAGOutput,
    SearchInput,
    SearchOutput,
    SearchResultItem,
    ToolCallLog,
    ToolCallRequest,
    ToolCallResult,
    ToolMetadata,
)

__all__ = [
    # Base
    "ErrorCode",
    "ToolInput",
    "ToolOutput",
    # Tools
    "SearchInput",
    "SearchOutput",
    "SearchResultItem",
    "RAGInput",
    "RAGOutput",
    "ToolMetadata",
    "ToolCallLog",
    "ToolCallRequest",
    "ToolCallResult",
    "MultiToolInput",
    "MultiToolOutput",
    # Planning
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
    # Memory
    "MessageRole",
    "Message",
    "Conversation",
    "MemoryType",
    "MemoryEntry",
    "MemorySearchResult",
    "ConversationContext",
    "ConversationalInput",
    "ConversationalOutput",
]
