"""
Schemas package for standardized tool interfaces.

This package contains all Pydantic models organized by domain:
- base: ErrorCode, ToolInput, ToolOutput
- tools: Search, RAG, ToolMetadata, MultiTool schemas
- planning: Task decomposition, Chain of thought, Self-reflection schemas
- memory: Conversation, Message, Memory entry schemas
"""

# Base schemas
from .base import ErrorCode, ToolInput, ToolOutput

# Tool schemas
from .tools import (
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

# Planning schemas
from .planning import (
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

# Memory schemas
from .memory import (
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
