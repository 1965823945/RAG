"""Tool-related schemas for search, RAG, metadata, and multi-tool calls."""

from typing import Any

from pydantic import BaseModel, Field

from .base import ErrorCode, ToolInput, ToolOutput

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
    doc_id: str | None = Field(default=None, description="Document ID")
    chunk_id: str | None = Field(
        default=None, description="Chunk ID within document"
    )
    chunk_index: int | None = Field(
        default=None, description="Chunk index in source"
    )


class SearchOutput(ToolOutput):
    """Output for knowledge search tool."""

    query: str = Field(default="", description="Original query")
    results: list[SearchResultItem] = Field(
        default_factory=list, description="Search results"
    )
    total_chunks: int | None = Field(
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
    sources: list[SearchResultItem] = Field(
        default_factory=list, description="Source documents"
    )

    # Additional metadata
    context_length: int | None = Field(
        default=None, description="Total context chars used"
    )
    model_name: str | None = Field(default=None, description="LLM model used")


# ─────────────────────────────────────────────────────────────
# Tool Metadata Schema (for registry)
# ─────────────────────────────────────────────────────────────


class ToolMetadata(BaseModel):
    """Metadata about a registered tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    version: str = Field(default="1.0.0", description="Tool version")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for input"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for output"
    )
    tags: list[str] = Field(
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
    input_params: dict[str, Any] = Field(
        default_factory=dict, description="Input parameters"
    )
    output_summary: str = Field(default="", description="Brief summary of output")

    # Status
    success: bool = Field(default=True)
    error_code: ErrorCode | None = Field(default=None)
    error_message: str | None = Field(default=None)


# ─────────────────────────────────────────────────────────────
# Multi-Tool Call Schemas (多工具调用)
# ─────────────────────────────────────────────────────────────


class ToolCallRequest(BaseModel):
    """A single tool call request."""

    tool_name: str = Field(..., description="Name of the tool to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )
    call_id: str | None = Field(
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
    result: dict[str, Any] | None = Field(
        default=None, description="Tool output as dict"
    )

    # Timing
    duration_ms: float | None = Field(
        default=None, description="Execution time in milliseconds"
    )
    timestamp: str | None = Field(default=None, description="ISO format timestamp")


class MultiToolInput(ToolInput):
    """Input for multi-tool invocation."""

    calls: list[ToolCallRequest] = Field(
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

    results: list[ToolCallResult] = Field(
        default_factory=list, description="Results from each tool call"
    )
    total_calls: int = Field(default=0, description="Total number of calls made")
    successful_calls: int = Field(default=0, description="Number of successful calls")
    failed_calls: int = Field(default=0, description="Number of failed calls")
    total_duration_ms: float = Field(
        default=0.0, description="Total execution time in milliseconds"
    )

    def get_result(self, call_id: str) -> ToolCallResult | None:
        """Get result by call ID."""
        for r in self.results:
            if r.call_id == call_id:
                return r
        return None

    def get_result_by_tool(self, tool_name: str) -> list[ToolCallResult]:
        """Get all results for a specific tool."""
        return [r for r in self.results if r.tool_name == tool_name]
