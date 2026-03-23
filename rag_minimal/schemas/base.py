"""Base schemas for standardized tool interfaces."""

from enum import Enum
from typing import Optional

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
