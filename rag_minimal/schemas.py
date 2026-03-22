"""Shared schemas for standardized tool interfaces."""

from enum import Enum
from typing import List, Optional, Any, Dict
from datetime import datetime
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
