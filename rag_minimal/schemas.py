"""Shared schemas for standardized tool interfaces."""

from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Standard Error Codes
# ─────────────────────────────────────────────────────────────


class ErrorCode(str, Enum):
    """Standardized error codes for tool execution."""

    # Success
    OK = "OK"

    # Input errors (4xx)
    INVALID_INPUT = "INVALID_INPUT"  # 参数校验失败
    MISSING_PARAM = "MISSING_PARAM"  # 缺少必要参数
    INVALID_FORMAT = "INVALID_FORMAT"  # 格式错误

    # Tool errors (5xx)
    TOOL_FAILED = "TOOL_FAILED"  # 工具执行失败
    TOOL_TIMEOUT = "TOOL_TIMEOUT"  # 工具超时
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"  # 工具不存在

    # Resource errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"  # 资源不存在
    NO_RESULTS = "NO_RESULTS"  # 无结果

    # LLM errors
    LLM_ERROR = "LLM_ERROR"  # LLM 调用失败
    LLM_TIMEOUT = "LLM_TIMEOUT"  # LLM 超时

    # System errors
    INTERNAL_ERROR = "INTERNAL_ERROR"  # 内部错误
    UNKNOWN_ERROR = "UNKNOWN_ERROR"  # 未知错误


# ─────────────────────────────────────────────────────────────
# Base Schemas
# ─────────────────────────────────────────────────────────────


class ToolInput(BaseModel):
    """Base tool input schema."""

    class Config:
        extra = "forbid"  # 禁止额外字段


class ToolOutput(BaseModel):
    """Base tool output schema with error handling and tracing."""

    success: bool = Field(default=True, description="Whether the operation succeeded")
    error_code: ErrorCode = Field(
        default=ErrorCode.OK, description="Standardized error code"
    )
    message: str = Field(default="ok", description="Human-readable message")

    # Tracing info
    trace_id: Optional[str] = Field(
        default=None, description="Unique trace ID for this call"
    )
    duration_ms: Optional[float] = Field(
        default=None, description="Execution time in milliseconds"
    )
    timestamp: Optional[str] = Field(default=None, description="ISO format timestamp")

    def with_error(
        self, error_code: ErrorCode, message: str, trace_id: Optional[str] = None
    ) -> "ToolOutput":
        """Create a copy with error info."""
        return self.model_copy(
            update={
                "success": False,
                "error_code": error_code,
                "message": message,
                "trace_id": trace_id,
            }
        )


# ─────────────────────────────────────────────────────────────
# Search Tool Schemas
# ─────────────────────────────────────────────────────────────


class SearchInput(ToolInput):
    """Input for knowledge search tool."""

    query: str = Field(..., min_length=1, description="User question or query")
    top_k: int = Field(
        default=3, ge=1, le=20, description="Number of results to return"
    )


class SearchResultItem(BaseModel):
    """Single search result with tracing info."""

    content: str = Field(..., description="The matched content")
    source: str = Field(default="", description="Source file path")
    score: float = Field(default=0.0, ge=0.0, description="Relevance score")

    # Tracing fields
    doc_id: Optional[str] = Field(default=None, description="Document ID")
    chunk_id: Optional[str] = Field(
        default=None, description="Chunk ID within document"
    )
    chunk_index: Optional[int] = Field(
        default=None, description="Chunk index in source"
    )


class SearchOutput(ToolOutput):
    """Output for knowledge search tool."""

    query: str = Field(default="", description="Original query")
    results: List[SearchResultItem] = Field(
        default_factory=list, description="Search results"
    )
    total_chunks: Optional[int] = Field(
        default=None, description="Total chunks searched"
    )


# ─────────────────────────────────────────────────────────────
# RAG Answer Schemas
# ─────────────────────────────────────────────────────────────


class RAGInput(ToolInput):
    """Input for RAG question-answering."""

    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(
        default=3, ge=1, le=20, description="Number of documents to retrieve"
    )


class RAGOutput(ToolOutput):
    """Output for RAG question-answering."""

    question: str = Field(default="", description="Original question")
    answer: str = Field(default="", description="Generated answer")
    sources: List[SearchResultItem] = Field(
        default_factory=list, description="Source documents"
    )

    # Additional metadata
    context_length: Optional[int] = Field(
        default=None, description="Total context chars used"
    )
    model_name: Optional[str] = Field(default=None, description="LLM model used")


# ─────────────────────────────────────────────────────────────
# Tool Metadata Schema (for registry)
# ─────────────────────────────────────────────────────────────


class ToolMetadata(BaseModel):
    """Metadata about a registered tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    version: str = Field(default="1.0.0", description="Tool version")
    input_schema: Dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for input"
    )
    output_schema: Dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for output"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tool tags for categorization"
    )


# ─────────────────────────────────────────────────────────────
# Tool Call Log Schema
# ─────────────────────────────────────────────────────────────


class ToolCallLog(BaseModel):
    """Log entry for a tool invocation."""

    trace_id: str = Field(..., description="Unique trace ID")
    tool_name: str = Field(..., description="Tool that was called")
    timestamp: str = Field(..., description="ISO format timestamp")
    duration_ms: float = Field(..., description="Execution time in ms")

    # Request/Response
    input_params: Dict[str, Any] = Field(
        default_factory=dict, description="Input parameters"
    )
    output_summary: str = Field(default="", description="Brief summary of output")

    # Status
    success: bool = Field(default=True)
    error_code: Optional[ErrorCode] = Field(default=None)
    error_message: Optional[str] = Field(default=None)


# ─────────────────────────────────────────────────────────────
# Planning Agent Schemas (自主规划 Agent)
# ─────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"  # 待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    BLOCKED = "blocked"  # 被阻塞
    CANCELLED = "cancelled"  # 已取消


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"  # 关键
    HIGH = "high"  # 高
    MEDIUM = "medium"  # 中
    LOW = "low"  # 低


class SubTask(BaseModel):
    """A subtask decomposed from the main task."""

    id: str = Field(..., description="Unique subtask ID")
    description: str = Field(..., description="What this subtask does")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status")
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM, description="Task priority"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="IDs of subtasks this depends on"
    )
    result: Optional[str] = Field(default=None, description="Execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    max_retries: int = Field(default=2, description="Maximum retry attempts")


class TaskDecomposition(BaseModel):
    """Result of task decomposition."""

    original_task: str = Field(..., description="The original user task")
    subtasks: List[SubTask] = Field(
        default_factory=list, description="Decomposed subtasks"
    )
    total_steps: int = Field(default=0, description="Total number of steps")
    complexity: str = Field(
        default="medium", description="Estimated complexity: simple/medium/complex"
    )
    estimated_time: Optional[str] = Field(
        default=None, description="Estimated completion time"
    )


class ThoughtStep(BaseModel):
    """A single step in the chain of thought."""

    step_number: int = Field(..., description="Step number in the chain")
    thought: str = Field(..., description="The reasoning thought")
    action: Optional[str] = Field(default=None, description="Action to take")
    action_input: Optional[Dict[str, Any]] = Field(
        default=None, description="Input for the action"
    )
    observation: Optional[str] = Field(default=None, description="Result of the action")
    timestamp: Optional[str] = Field(
        default=None, description="When this step occurred"
    )


class ChainOfThoughtResult(BaseModel):
    """Complete chain of thought reasoning."""

    question: str = Field(..., description="The question being reasoned about")
    thoughts: List[ThoughtStep] = Field(
        default_factory=list, description="Chain of thought steps"
    )
    final_answer: Optional[str] = Field(default=None, description="Final conclusion")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the answer"
    )
    total_steps: int = Field(default=0, description="Total reasoning steps")


class ReflectionType(str, Enum):
    """Types of self-reflection."""

    QUALITY_CHECK = "quality_check"  # 质量检查
    ERROR_ANALYSIS = "error_analysis"  # 错误分析
    IMPROVEMENT = "improvement"  # 改进建议
    COMPLETENESS = "completeness"  # 完整性检查
    CONSISTENCY = "consistency"  # 一致性检查


class ReflectionItem(BaseModel):
    """A single reflection insight."""

    reflection_type: ReflectionType = Field(..., description="Type of reflection")
    aspect: str = Field(..., description="What aspect is being reflected on")
    assessment: str = Field(..., description="Assessment of this aspect")
    score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Score for this aspect"
    )
    issues: List[str] = Field(default_factory=list, description="Issues identified")
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )


class SelfReflectionResult(BaseModel):
    """Result of self-reflection process."""

    context: str = Field(..., description="What is being reflected upon")
    reflections: List[ReflectionItem] = Field(
        default_factory=list, description="List of reflections"
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall quality score"
    )
    should_retry: bool = Field(default=False, description="Whether to retry the task")
    retry_reason: Optional[str] = Field(
        default=None, description="Reason for retry if applicable"
    )
    improvements: List[str] = Field(
        default_factory=list, description="Suggested improvements"
    )


class PlanningAgentState(BaseModel):
    """State of the planning agent execution."""

    task_id: str = Field(..., description="Unique task execution ID")
    original_query: str = Field(..., description="Original user query")

    # Task decomposition
    decomposition: Optional[TaskDecomposition] = Field(
        default=None, description="Task decomposition result"
    )

    # Chain of thought
    reasoning: Optional[ChainOfThoughtResult] = Field(
        default=None, description="Chain of thought reasoning"
    )

    # Self reflection
    reflection: Optional[SelfReflectionResult] = Field(
        default=None, description="Self reflection result"
    )

    # Execution state
    current_step: int = Field(default=0, description="Current execution step")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Overall status")
    iterations: int = Field(default=0, description="Number of iterations")
    max_iterations: int = Field(default=3, description="Maximum iterations allowed")

    # Results
    final_answer: Optional[str] = Field(default=None, description="Final answer")
    execution_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of all executions"
    )

    # Timing
    start_time: Optional[str] = Field(default=None, description="Execution start time")
    end_time: Optional[str] = Field(default=None, description="Execution end time")


class PlanningAgentOutput(ToolOutput):
    """Output from the planning agent."""

    task_id: str = Field(default="", description="Task execution ID")
    query: str = Field(default="", description="Original query")
    answer: str = Field(default="", description="Final answer")

    # Planning details
    subtasks_completed: int = Field(default=0, description="Subtasks completed")
    subtasks_total: int = Field(default=0, description="Total subtasks")
    reasoning_steps: int = Field(default=0, description="Reasoning steps taken")
    iterations: int = Field(default=0, description="Improvement iterations")

    # Quality metrics
    confidence: float = Field(default=0.0, description="Answer confidence")
    quality_score: float = Field(default=0.0, description="Quality score")

    # Detailed state (optional)
    state: Optional[PlanningAgentState] = Field(
        default=None, description="Full agent state"
    )


# ─────────────────────────────────────────────────────────────
# Conversation & Memory Schemas (对话与记忆)
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


# ─────────────────────────────────────────────────────────────
# Multi-Tool Call Schemas (多工具调用)
# ─────────────────────────────────────────────────────────────


class ToolCallRequest(BaseModel):
    """A single tool call request."""

    tool_name: str = Field(..., description="Name of the tool to invoke")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )
    call_id: Optional[str] = Field(
        default=None,
        description="Unique ID for this call (auto-generated if not provided)",
    )


class ToolCallResult(BaseModel):
    """Result of a single tool call."""

    call_id: str = Field(..., description="Unique ID for this call")
    tool_name: str = Field(..., description="Name of the tool that was called")
    success: bool = Field(default=True, description="Whether the call succeeded")
    error_code: ErrorCode = Field(
        default=ErrorCode.OK, description="Error code if failed"
    )
    message: str = Field(default="ok", description="Human-readable message")

    # Result data
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Tool output as dict"
    )

    # Timing
    duration_ms: Optional[float] = Field(
        default=None, description="Execution time in milliseconds"
    )
    timestamp: Optional[str] = Field(default=None, description="ISO format timestamp")


class MultiToolInput(ToolInput):
    """Input for multi-tool invocation."""

    calls: List[ToolCallRequest] = Field(
        ..., min_length=1, description="List of tool calls to execute"
    )
    parallel: bool = Field(
        default=True,
        description="Execute calls in parallel (True) or sequential (False)",
    )
    stop_on_error: bool = Field(
        default=False, description="Stop execution on first error (only for sequential)"
    )


class MultiToolOutput(ToolOutput):
    """Output from multi-tool invocation."""

    results: List[ToolCallResult] = Field(
        default_factory=list, description="Results from each tool call"
    )
    total_calls: int = Field(default=0, description="Total number of calls made")
    successful_calls: int = Field(default=0, description="Number of successful calls")
    failed_calls: int = Field(default=0, description="Number of failed calls")
    total_duration_ms: float = Field(
        default=0.0, description="Total execution time in milliseconds"
    )

    def get_result(self, call_id: str) -> Optional[ToolCallResult]:
        """Get result by call ID."""
        for r in self.results:
            if r.call_id == call_id:
                return r
        return None

    def get_result_by_tool(self, tool_name: str) -> List[ToolCallResult]:
        """Get all results for a specific tool."""
        return [r for r in self.results if r.tool_name == tool_name]
