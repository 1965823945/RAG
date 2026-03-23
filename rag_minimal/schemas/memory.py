"""Memory and conversation schemas for conversational agents."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import ToolInput, ToolOutput
from .tools import SearchResultItem


# ─────────────────────────────────────────────────────────────
# Message & Conversation Schemas
# ─────────────────────────────────────────────────────────────


class MessageRole(str, Enum):
    """Role of the message sender."""

    USER = "user"  # 用户消息
    ASSISTANT = "assistant"  # 助手回复
    SYSTEM = "system"  # 系统消息
    TOOL = "tool"  # 工具调用结果


class Message(BaseModel):
    """A single message in a conversation."""

    id: str = Field(..., description="Unique message ID")
    role: MessageRole = Field(..., description="Role of the sender")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="ISO format timestamp")

    # Optional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # For tool messages
    tool_name: Optional[str] = Field(
        default=None, description="Tool name if tool message"
    )
    tool_call_id: Optional[str] = Field(default=None, description="Tool call ID")

    # For assistant messages
    sources: List[SearchResultItem] = Field(
        default_factory=list, description="Sources used for this response"
    )


class Conversation(BaseModel):
    """A conversation session containing multiple messages."""

    id: str = Field(..., description="Unique conversation ID")
    title: Optional[str] = Field(default=None, description="Conversation title")
    messages: List[Message] = Field(default_factory=list, description="Message history")

    # Timestamps
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Conversation metadata"
    )
    summary: Optional[str] = Field(default=None, description="Conversation summary")

    # Statistics
    message_count: int = Field(default=0, description="Total message count")
    token_count: int = Field(default=0, description="Estimated token count")


# ─────────────────────────────────────────────────────────────
# Memory Entry Schemas
# ─────────────────────────────────────────────────────────────


class MemoryType(str, Enum):
    """Type of memory entry."""

    FACT = "fact"  # 事实信息
    PREFERENCE = "preference"  # 用户偏好
    CONTEXT = "context"  # 上下文信息
    SUMMARY = "summary"  # 对话摘要
    ENTITY = "entity"  # 实体信息
    EPISODE = "episode"  # 情景记忆


class MemoryEntry(BaseModel):
    """A single memory entry."""

    id: str = Field(..., description="Unique memory ID")
    memory_type: MemoryType = Field(..., description="Type of memory")
    content: str = Field(..., description="Memory content")

    # Source tracking
    source_conversation_id: Optional[str] = Field(
        default=None, description="Source conversation ID"
    )
    source_message_id: Optional[str] = Field(
        default=None, description="Source message ID"
    )

    # Timestamps
    created_at: str = Field(..., description="Creation timestamp")
    last_accessed: str = Field(..., description="Last access timestamp")

    # Importance and relevance
    importance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Importance score"
    )
    access_count: int = Field(default=0, description="Number of times accessed")

    # For entity memories
    entity_name: Optional[str] = Field(default=None, description="Entity name")
    entity_type: Optional[str] = Field(default=None, description="Entity type")

    # Embedding for similarity search
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding"
    )

    # TTL for short-term memories
    expires_at: Optional[str] = Field(default=None, description="Expiration timestamp")


class MemorySearchResult(BaseModel):
    """Result of memory search."""

    entry: MemoryEntry = Field(..., description="The memory entry")
    relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relevance to query"
    )


# ─────────────────────────────────────────────────────────────
# Conversation Context
# ─────────────────────────────────────────────────────────────


class ConversationContext(BaseModel):
    """Context built from conversation history and memories."""

    # Recent messages
    recent_messages: List[Message] = Field(
        default_factory=list, description="Recent conversation messages"
    )

    # Relevant memories
    relevant_memories: List[MemoryEntry] = Field(
        default_factory=list, description="Relevant memory entries"
    )

    # Retrieved documents
    retrieved_docs: List[SearchResultItem] = Field(
        default_factory=list, description="Retrieved documents"
    )

    # Conversation summary
    conversation_summary: Optional[str] = Field(
        default=None, description="Summary of conversation so far"
    )

    # Extracted entities
    entities: Dict[str, str] = Field(
        default_factory=dict, description="Extracted entities"
    )

    # Token budget info
    total_tokens: int = Field(default=0, description="Estimated total tokens")
    max_tokens: int = Field(default=4000, description="Maximum context tokens")


# ─────────────────────────────────────────────────────────────
# Conversational Agent Input/Output
# ─────────────────────────────────────────────────────────────


class ConversationalInput(ToolInput):
    """Input for conversational agent."""

    message: str = Field(..., min_length=1, description="User message")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID to continue"
    )
    include_history: bool = Field(
        default=True, description="Whether to include conversation history"
    )
    include_memories: bool = Field(
        default=True, description="Whether to include relevant memories"
    )
    top_k: int = Field(
        default=3, ge=1, le=20, description="Number of documents to retrieve"
    )


class ConversationalOutput(ToolOutput):
    """Output from conversational agent."""

    conversation_id: str = Field(..., description="Conversation ID")
    message_id: str = Field(..., description="Response message ID")
    response: str = Field(..., description="Assistant response")

    # Context used
    messages_used: int = Field(default=0, description="Number of history messages used")
    memories_used: int = Field(default=0, description="Number of memories used")
    documents_used: int = Field(default=0, description="Number of documents retrieved")

    # Sources
    sources: List[SearchResultItem] = Field(
        default_factory=list, description="Source documents"
    )

    # Extracted information
    new_memories: List[str] = Field(
        default_factory=list, description="New memories extracted from this turn"
    )
    entities_mentioned: Dict[str, str] = Field(
        default_factory=dict, description="Entities mentioned"
    )
