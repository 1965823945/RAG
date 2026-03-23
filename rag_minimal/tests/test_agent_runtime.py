"""Tests for AgentRuntime - Multi-tool invocation system."""

import time
from typing import Any, Dict
from pydantic import BaseModel, Field

from rag_minimal.agent_runtime import AgentRuntime
from rag_minimal.tools.base import Tool
from rag_minimal.tools.examples import CalculatorTool, EchoTool, TextTransformTool
from rag_minimal.schemas import (
    ToolCallRequest,
    ToolOutput,
    ErrorCode,
)


# ─────────────────────────────────────────────────────────────
# Test Fixtures - Mock Tools
# ─────────────────────────────────────────────────────────────


class SlowToolInput(BaseModel):
    """Input for slow tool."""

    delay: float = Field(default=0.1, description="Delay in seconds")


class SlowToolOutput(ToolOutput):
    """Output from slow tool."""

    delayed: float = Field(default=0.0, description="Actual delay")


class SlowTool(Tool):
    """A tool that sleeps for testing parallel execution."""

    name = "slow_tool"
    description = "A tool that sleeps for a specified duration"
    input_schema = SlowToolInput
    output_schema = SlowToolOutput

    def invoke(self, payload: Dict[str, Any]) -> SlowToolOutput:
        validated = self.validate_input(payload)
        time.sleep(validated.delay)
        return SlowToolOutput(success=True, delayed=validated.delay)


class FailingToolInput(BaseModel):
    """Input for failing tool."""

    should_fail: bool = Field(default=True)


class FailingToolOutput(ToolOutput):
    """Output from failing tool."""

    pass


class FailingTool(Tool):
    """A tool that always fails for testing error handling."""

    name = "failing_tool"
    description = "A tool that always raises an exception"
    input_schema = FailingToolInput
    output_schema = FailingToolOutput

    def invoke(self, payload: Dict[str, Any]) -> FailingToolOutput:
        validated = self.validate_input(payload)
        if validated.should_fail:
            raise ValueError("This tool always fails!")
        return FailingToolOutput(success=True)


# ─────────────────────────────────────────────────────────────
# Test Classes
# ─────────────────────────────────────────────────────────────


class TestAgentRuntimeBasic:
    """Basic AgentRuntime tests."""

    def test_init_default(self):
        """Test default initialization."""
        runtime = AgentRuntime(docs_dir="docs")
        assert runtime is not None
        assert "knowledge_search" in runtime.list_tools()

    def test_register_tool(self):
        """Test tool registration."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(CalculatorTool())

        assert "calculator" in runtime.list_tools()
        assert len(runtime.list_tools()) == 2  # knowledge_search + calculator

    def test_register_duplicate_tool_fails(self):
        """Test that registering duplicate tool raises error."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(CalculatorTool())

        try:
            runtime.register_tool(CalculatorTool())
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "already registered" in str(e)

    def test_unregister_tool(self):
        """Test tool unregistration."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(CalculatorTool())

        assert runtime.unregister_tool("calculator") is True
        assert "calculator" not in runtime.list_tools()

    def test_unregister_nonexistent_tool(self):
        """Test unregistering non-existent tool returns False."""
        runtime = AgentRuntime(docs_dir="docs")
        assert runtime.unregister_tool("nonexistent") is False


class TestInvokeTool:
    """Tests for single tool invocation."""

    def test_invoke_calculator(self):
        """Test invoking calculator tool."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(CalculatorTool())

        result = runtime.invoke_tool("calculator", {"expression": "2 + 3 * 4"})

        assert result.success is True
        assert result.tool_name == "calculator"
        assert result.error_code == ErrorCode.OK
        assert result.result["result"] == 14.0
        assert result.duration_ms is not None
        assert result.duration_ms > 0

    def test_invoke_echo(self):
        """Test invoking echo tool."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(EchoTool())

        result = runtime.invoke_tool(
            "echo", {"message": "hello", "uppercase": True, "repeat": 2}
        )

        assert result.success is True
        assert result.result["echoed"] == "HELLO HELLO"

    def test_invoke_nonexistent_tool(self):
        """Test invoking non-existent tool returns error."""
        runtime = AgentRuntime(docs_dir="docs")

        result = runtime.invoke_tool("nonexistent", {})

        assert result.success is False
        assert result.error_code == ErrorCode.TOOL_NOT_FOUND
        assert "not found" in result.message

    def test_invoke_with_custom_call_id(self):
        """Test invoking with custom call ID."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(EchoTool())

        result = runtime.invoke_tool("echo", {"message": "test"}, call_id="custom-123")

        assert result.call_id == "custom-123"

    def test_invoke_failing_tool(self):
        """Test invoking a tool that raises exception."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(FailingTool())

        result = runtime.invoke_tool("failing_tool", {"should_fail": True})

        assert result.success is False
        assert result.error_code == ErrorCode.TOOL_FAILED
        assert "always fails" in result.message


class TestInvokeToolsParallel:
    """Tests for parallel multi-tool invocation."""

    def test_invoke_multiple_tools_parallel(self):
        """Test invoking multiple tools in parallel."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(CalculatorTool())
        runtime.register_tool(EchoTool())
        runtime.register_tool(TextTransformTool())

        calls = [
            ToolCallRequest(
                tool_name="calculator", arguments={"expression": "10 * 10"}
            ),
            ToolCallRequest(tool_name="echo", arguments={"message": "hello"}),
            ToolCallRequest(
                tool_name="text_transform",
                arguments={"text": "abc", "operation": "upper"},
            ),
        ]

        result = runtime.invoke_tools(calls, parallel=True)

        assert result.total_calls == 3
        assert result.successful_calls == 3
        assert result.failed_calls == 0
        assert result.success is True
        assert len(result.results) == 3

    def test_parallel_is_faster_than_sequential(self):
        """Test that parallel execution is faster."""
        runtime = AgentRuntime(docs_dir="docs", max_workers=4)
        runtime.register_tool(SlowTool())

        # Create 4 calls that each sleep 0.1s
        calls = [
            ToolCallRequest(tool_name="slow_tool", arguments={"delay": 0.1})
            for _ in range(4)
        ]

        # Parallel should take ~0.1s (all run at once)
        start = time.time()
        result_parallel = runtime.invoke_tools(calls, parallel=True)
        parallel_time = time.time() - start

        # Sequential would take ~0.4s (0.1 * 4)
        start = time.time()
        result_sequential = runtime.invoke_tools(calls, parallel=False)
        sequential_time = time.time() - start

        assert result_parallel.successful_calls == 4
        assert result_sequential.successful_calls == 4
        # Parallel should be at least 2x faster
        assert parallel_time < sequential_time * 0.7

    def test_parallel_with_one_failure(self):
        """Test parallel execution with one failing tool."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(EchoTool())
        runtime.register_tool(FailingTool())

        calls = [
            ToolCallRequest(tool_name="echo", arguments={"message": "ok"}),
            ToolCallRequest(tool_name="failing_tool", arguments={"should_fail": True}),
            ToolCallRequest(tool_name="echo", arguments={"message": "also ok"}),
        ]

        result = runtime.invoke_tools(calls, parallel=True)

        assert result.total_calls == 3
        assert result.successful_calls == 2
        assert result.failed_calls == 1
        assert result.success is False  # Overall failed because one failed


class TestInvokeToolsSequential:
    """Tests for sequential multi-tool invocation."""

    def test_invoke_sequential(self):
        """Test sequential tool invocation."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(CalculatorTool())
        runtime.register_tool(EchoTool())

        calls = [
            ToolCallRequest(tool_name="calculator", arguments={"expression": "1 + 1"}),
            ToolCallRequest(tool_name="echo", arguments={"message": "two"}),
        ]

        result = runtime.invoke_tools(calls, parallel=False)

        assert result.total_calls == 2
        assert result.successful_calls == 2
        # Results should be in order for sequential
        assert result.results[0].tool_name == "calculator"
        assert result.results[1].tool_name == "echo"

    def test_sequential_stop_on_error(self):
        """Test sequential execution stops on error when configured."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(EchoTool())
        runtime.register_tool(FailingTool())

        calls = [
            ToolCallRequest(tool_name="echo", arguments={"message": "first"}),
            ToolCallRequest(tool_name="failing_tool", arguments={"should_fail": True}),
            ToolCallRequest(tool_name="echo", arguments={"message": "third"}),
        ]

        result = runtime.invoke_tools(calls, parallel=False, stop_on_error=True)

        # Should stop after failing_tool, so only 2 calls made
        assert result.total_calls == 2
        assert result.successful_calls == 1
        assert result.failed_calls == 1

    def test_sequential_continue_on_error(self):
        """Test sequential execution continues on error by default."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(EchoTool())
        runtime.register_tool(FailingTool())

        calls = [
            ToolCallRequest(tool_name="echo", arguments={"message": "first"}),
            ToolCallRequest(tool_name="failing_tool", arguments={"should_fail": True}),
            ToolCallRequest(tool_name="echo", arguments={"message": "third"}),
        ]

        result = runtime.invoke_tools(calls, parallel=False, stop_on_error=False)

        # Should continue after failure
        assert result.total_calls == 3
        assert result.successful_calls == 2
        assert result.failed_calls == 1


class TestInvokeMany:
    """Tests for invoke_many - same tool with multiple inputs."""

    def test_invoke_many_calculator(self):
        """Test invoking calculator with multiple expressions."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(CalculatorTool())

        args_list = [
            {"expression": "1 + 1"},
            {"expression": "2 * 2"},
            {"expression": "3 ** 3"},
        ]

        result = runtime.invoke_many("calculator", args_list, parallel=True)

        assert result.total_calls == 3
        assert result.successful_calls == 3

        # Check results (order may vary in parallel)
        results_map = {
            r.result["expression"]: r.result["result"] for r in result.results
        }
        assert results_map["1 + 1"] == 2.0
        assert results_map["2 * 2"] == 4.0
        assert results_map["3 ** 3"] == 27.0

    def test_invoke_many_sequential(self):
        """Test invoke_many in sequential mode."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(EchoTool())

        args_list = [
            {"message": "a"},
            {"message": "b"},
            {"message": "c"},
        ]

        result = runtime.invoke_many("echo", args_list, parallel=False)

        assert result.total_calls == 3
        # Sequential should maintain order
        assert result.results[0].result["echoed"] == "a"
        assert result.results[1].result["echoed"] == "b"
        assert result.results[2].result["echoed"] == "c"


class TestInvokeChain:
    """Tests for invoke_chain - sequential with result passing."""

    def test_chain_stops_on_failure(self):
        """Test that chain stops on first failure."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(FailingTool())
        runtime.register_tool(EchoTool())

        calls = [
            ToolCallRequest(tool_name="failing_tool", arguments={"should_fail": True}),
            ToolCallRequest(tool_name="echo", arguments={"message": "never reached"}),
        ]

        result = runtime.invoke_chain(calls)

        assert result.total_calls == 1
        assert result.failed_calls == 1
        assert result.success is False

    def test_chain_single_tool(self):
        """Test chain with single tool."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(EchoTool())

        calls = [
            ToolCallRequest(tool_name="echo", arguments={"message": "single"}),
        ]

        result = runtime.invoke_chain(calls)

        assert result.total_calls == 1
        assert result.successful_calls == 1
        assert result.success is True
        """Test that chain stops on first failure."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(FailingTool())
        runtime.register_tool(EchoTool())

        calls = [
            ToolCallRequest(tool_name="failing_tool", arguments={"should_fail": True}),
            ToolCallRequest(tool_name="echo", arguments={"message": "never reached"}),
        ]

        result = runtime.invoke_chain(calls)

        assert result.total_calls == 1
        assert result.failed_calls == 1
        assert result.success is False


class TestMultiToolOutput:
    """Tests for MultiToolOutput helper methods."""

    def test_get_result_by_call_id(self):
        """Test getting result by call ID."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(EchoTool())

        calls = [
            ToolCallRequest(
                tool_name="echo", arguments={"message": "a"}, call_id="id-1"
            ),
            ToolCallRequest(
                tool_name="echo", arguments={"message": "b"}, call_id="id-2"
            ),
        ]

        result = runtime.invoke_tools(calls)

        r1 = result.get_result("id-1")
        r2 = result.get_result("id-2")
        r_none = result.get_result("nonexistent")

        assert r1 is not None
        assert r1.result["echoed"] == "a"
        assert r2 is not None
        assert r2.result["echoed"] == "b"
        assert r_none is None

    def test_get_result_by_tool(self):
        """Test getting results by tool name."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(EchoTool())
        runtime.register_tool(CalculatorTool())

        calls = [
            ToolCallRequest(tool_name="echo", arguments={"message": "a"}),
            ToolCallRequest(tool_name="calculator", arguments={"expression": "1+1"}),
            ToolCallRequest(tool_name="echo", arguments={"message": "b"}),
        ]

        result = runtime.invoke_tools(calls)

        echo_results = result.get_result_by_tool("echo")
        calc_results = result.get_result_by_tool("calculator")

        assert len(echo_results) == 2
        assert len(calc_results) == 1


class TestRAGPipeline:
    """Tests for RAG pipeline methods (search, ask)."""

    def test_search(self):
        """Test knowledge search."""
        runtime = AgentRuntime(docs_dir="docs")

        result = runtime.search("RAG", top_k=2)

        # Should return SearchOutput
        assert hasattr(result, "success")
        assert hasattr(result, "results")
        assert hasattr(result, "query")
        assert result.query == "RAG"

    def test_ask(self):
        """Test full RAG pipeline."""
        runtime = AgentRuntime(docs_dir="docs")

        result = runtime.ask("什么是 RAG？", top_k=2)

        assert hasattr(result, "success")
        assert hasattr(result, "answer")
        assert hasattr(result, "sources")
        assert result.question == "什么是 RAG？"

    def test_run_backward_compatible(self):
        """Test run() returns dict for backward compatibility."""
        runtime = AgentRuntime(docs_dir="docs")

        result = runtime.run("test query")

        assert isinstance(result, dict)
        assert "success" in result
        assert "answer" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_calls_list(self):
        """Test invoking with empty calls list."""
        runtime = AgentRuntime(docs_dir="docs")

        result = runtime.invoke_tools([])

        assert result.success is False
        assert result.error_code == ErrorCode.INVALID_INPUT

    def test_invalid_tool_arguments(self):
        """Test invoking with invalid arguments."""
        runtime = AgentRuntime(docs_dir="docs")
        runtime.register_tool(CalculatorTool())

        # expression is required but not provided
        result = runtime.invoke_tool("calculator", {})

        assert result.success is False
        assert result.error_code == ErrorCode.TOOL_FAILED

    def test_max_workers_respected(self):
        """Test that max_workers configuration is respected."""
        runtime = AgentRuntime(docs_dir="docs", max_workers=2)
        assert runtime.max_workers == 2
