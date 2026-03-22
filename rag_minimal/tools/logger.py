"""Tool call logging and tracing."""

import uuid
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

from rag_minimal.schemas import ToolCallLog, ErrorCode


# Configure logger
logger = logging.getLogger("rag_minimal.tools")


class ToolLogger:
    """Logger for tool invocations with tracing support."""

    def __init__(self, max_history: int = 100):
        """Initialize the tool logger.

        Args:
            max_history: Maximum number of log entries to keep in memory
        """
        self._history: List[ToolCallLog] = []
        self._max_history = max_history

    @staticmethod
    def generate_trace_id() -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4())[:8]

    @staticmethod
    def now_iso() -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"

    def log(
        self,
        tool_name: str,
        trace_id: str,
        duration_ms: float,
        input_params: Dict[str, Any],
        success: bool,
        output_summary: str = "",
        error_code: Optional[ErrorCode] = None,
        error_message: Optional[str] = None,
    ) -> ToolCallLog:
        """Log a tool invocation.

        Args:
            tool_name: Name of the tool
            trace_id: Unique trace ID
            duration_ms: Execution time in milliseconds
            input_params: Input parameters (will be sanitized)
            success: Whether the call succeeded
            output_summary: Brief summary of the output
            error_code: Error code if failed
            error_message: Error message if failed

        Returns:
            The created log entry
        """
        # Sanitize input params (remove sensitive data)
        safe_params = self._sanitize_params(input_params)

        entry = ToolCallLog(
            trace_id=trace_id,
            tool_name=tool_name,
            timestamp=self.now_iso(),
            duration_ms=duration_ms,
            input_params=safe_params,
            output_summary=output_summary[:200] if output_summary else "",
            success=success,
            error_code=error_code,
            error_message=error_message,
        )

        # Add to history
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        # Also log to standard logger
        level = logging.INFO if success else logging.ERROR
        logger.log(
            level,
            f"[{trace_id}] {tool_name} | {duration_ms:.1f}ms | "
            f"{'OK' if success else error_code}",
        )

        return entry

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive parameters."""
        sensitive_keys = {"password", "api_key", "token", "secret"}
        result = {}
        for key, value in params.items():
            if key.lower() in sensitive_keys:
                result[key] = "***"
            elif isinstance(value, str) and len(value) > 500:
                result[key] = value[:500] + "...(truncated)"
            else:
                result[key] = value
        return result

    def get_history(self, limit: int = 10) -> List[ToolCallLog]:
        """Get recent log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent log entries
        """
        return self._history[-limit:]

    def get_by_trace_id(self, trace_id: str) -> Optional[ToolCallLog]:
        """Get a log entry by trace ID."""
        for entry in reversed(self._history):
            if entry.trace_id == trace_id:
                return entry
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about tool calls.

        Returns:
            Dictionary with stats like total calls, success rate, avg duration
        """
        if not self._history:
            return {
                "total_calls": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "by_tool": {},
            }

        total = len(self._history)
        successes = sum(1 for e in self._history if e.success)
        avg_duration = sum(e.duration_ms for e in self._history) / total

        # Group by tool
        by_tool: Dict[str, Dict[str, Any]] = {}
        for entry in self._history:
            if entry.tool_name not in by_tool:
                by_tool[entry.tool_name] = {"calls": 0, "successes": 0, "total_ms": 0.0}
            by_tool[entry.tool_name]["calls"] += 1
            if entry.success:
                by_tool[entry.tool_name]["successes"] += 1
            by_tool[entry.tool_name]["total_ms"] += entry.duration_ms

        return {
            "total_calls": total,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_duration_ms": avg_duration,
            "by_tool": by_tool,
        }

    def clear(self):
        """Clear all log entries."""
        self._history.clear()


# Global logger instance
_global_logger: Optional[ToolLogger] = None


def get_tool_logger() -> ToolLogger:
    """Get the global tool logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = ToolLogger()
    return _global_logger


def logged_invoke(func: Callable) -> Callable:
    """Decorator to automatically log tool invocations.

    Usage:
        class MyTool(Tool):
            @logged_invoke
            def invoke(self, payload):
                ...
    """

    @wraps(func)
    def wrapper(self, payload: Dict[str, Any], *args, **kwargs):
        tool_logger = get_tool_logger()
        trace_id = tool_logger.generate_trace_id()

        start_time = time.perf_counter()
        try:
            result = func(self, payload, *args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Extract success info from result
            success = getattr(result, "success", True)
            error_code = getattr(result, "error_code", None)
            message = getattr(result, "message", "")

            # Create output summary
            if hasattr(result, "results"):
                output_summary = f"{len(result.results)} results"
            elif hasattr(result, "answer"):
                output_summary = result.answer[:100] if result.answer else ""
            else:
                output_summary = str(result)[:100]

            tool_logger.log(
                tool_name=getattr(self, "name", "unknown"),
                trace_id=trace_id,
                duration_ms=duration_ms,
                input_params=payload,
                success=success,
                output_summary=output_summary,
                error_code=error_code if not success else None,
                error_message=message if not success else None,
            )

            # Attach trace info to result if possible
            if hasattr(result, "trace_id"):
                result.trace_id = trace_id
            if hasattr(result, "duration_ms"):
                result.duration_ms = duration_ms
            if hasattr(result, "timestamp"):
                result.timestamp = tool_logger.now_iso()

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            tool_logger.log(
                tool_name=getattr(self, "name", "unknown"),
                trace_id=trace_id,
                duration_ms=duration_ms,
                input_params=payload,
                success=False,
                error_code=ErrorCode.TOOL_FAILED,
                error_message=str(e),
            )
            raise

    return wrapper
