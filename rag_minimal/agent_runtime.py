"""Minimal agent runtime using standardized tools."""

import asyncio
import time
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models import BaseLLM

from rag_minimal.constants import DEFAULT_RAG_PROMPT
from rag_minimal.schemas import (
    ErrorCode,
    SearchOutput,
    RAGOutput,
    ToolCallRequest,
    ToolCallResult,
    MultiToolInput,
    MultiToolOutput,
)
from rag_minimal.tools.knowledge_search import KnowledgeSearchTool
from rag_minimal.tools.registry import ToolRegistry
from rag_minimal.tools.base import Tool
from rag_minimal.llm import SimpleLLM

logger = logging.getLogger(__name__)


class AgentRuntime:
    """A minimal standardized agent runtime.

    This runtime provides:
    1. Tool registration and invocation
    2. Multi-tool parallel/sequential execution
    3. RAG pipeline: search -> format context -> LLM generate
    """

    def __init__(
        self,
        docs_dir: str = "docs",
        llm: Optional[BaseLLM] = None,
        prompt_template: str = DEFAULT_RAG_PROMPT,
        max_workers: int = 4,
    ):
        self.registry = ToolRegistry()
        self.registry.register(KnowledgeSearchTool(docs_dir=docs_dir))
        self.llm = llm or SimpleLLM()
        self.prompt_template = prompt_template
        self.max_workers = max_workers

    def register_tool(self, tool: Tool) -> None:
        """Register a tool with the runtime.

        Args:
            tool: Tool instance to register
        """
        self.registry.register(tool)

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed
        """
        return self.registry.unregister(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return self.registry.list_tools()

    def set_llm(self, llm: BaseLLM) -> None:
        """Set or replace the LLM."""
        self.llm = llm

    # ─────────────────────────────────────────────────────────────
    # Single Tool Invocation
    # ─────────────────────────────────────────────────────────────

    def invoke_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None,
    ) -> ToolCallResult:
        """Invoke a single tool by name.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Arguments to pass to the tool
            call_id: Optional call ID (auto-generated if not provided)

        Returns:
            ToolCallResult with the result or error
        """
        call_id = call_id or str(uuid.uuid4())[:8]
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        # Check if tool exists
        if not self.registry.has(tool_name):
            return ToolCallResult(
                call_id=call_id,
                tool_name=tool_name,
                success=False,
                error_code=ErrorCode.TOOL_NOT_FOUND,
                message=f"Tool '{tool_name}' not found. Available: {self.list_tools()}",
                timestamp=timestamp,
            )

        try:
            tool = self.registry.get(tool_name)
            result = tool.invoke(arguments)

            # Convert result to dict
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = {"value": str(result)}

            duration_ms = (time.time() - start_time) * 1000

            return ToolCallResult(
                call_id=call_id,
                tool_name=tool_name,
                success=True,
                error_code=ErrorCode.OK,
                message="ok",
                result=result_dict,
                duration_ms=duration_ms,
                timestamp=timestamp,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Tool '{tool_name}' failed: {e}")

            return ToolCallResult(
                call_id=call_id,
                tool_name=tool_name,
                success=False,
                error_code=ErrorCode.TOOL_FAILED,
                message=str(e),
                duration_ms=duration_ms,
                timestamp=timestamp,
            )

    # ─────────────────────────────────────────────────────────────
    # Multi-Tool Invocation
    # ─────────────────────────────────────────────────────────────

    def invoke_tools(
        self,
        calls: List[ToolCallRequest],
        parallel: bool = True,
        stop_on_error: bool = False,
    ) -> MultiToolOutput:
        """Invoke multiple tools.

        Args:
            calls: List of tool call requests
            parallel: Execute in parallel (True) or sequential (False)
            stop_on_error: Stop on first error (only for sequential mode)

        Returns:
            MultiToolOutput with all results
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        if not calls:
            return MultiToolOutput(
                success=False,
                error_code=ErrorCode.INVALID_INPUT,
                message="No tool calls provided",
                timestamp=timestamp,
            )

        if parallel:
            results = self._invoke_tools_parallel(calls)
        else:
            results = self._invoke_tools_sequential(calls, stop_on_error)

        total_duration_ms = (time.time() - start_time) * 1000
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        return MultiToolOutput(
            success=failed == 0,
            error_code=ErrorCode.OK if failed == 0 else ErrorCode.TOOL_FAILED,
            message="ok" if failed == 0 else f"{failed} tool(s) failed",
            results=results,
            total_calls=len(results),
            successful_calls=successful,
            failed_calls=failed,
            total_duration_ms=total_duration_ms,
            timestamp=timestamp,
        )

    def _invoke_tools_parallel(
        self, calls: List[ToolCallRequest]
    ) -> List[ToolCallResult]:
        """Execute tool calls in parallel using ThreadPoolExecutor."""
        results: List[ToolCallResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_call = {
                executor.submit(
                    self.invoke_tool,
                    call.tool_name,
                    call.arguments,
                    call.call_id or str(uuid.uuid4())[:8],
                ): call
                for call in calls
            }

            # Collect results as they complete
            for future in as_completed(future_to_call):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    call = future_to_call[future]
                    results.append(
                        ToolCallResult(
                            call_id=call.call_id or "unknown",
                            tool_name=call.tool_name,
                            success=False,
                            error_code=ErrorCode.INTERNAL_ERROR,
                            message=f"Execution error: {e}",
                            timestamp=datetime.now().isoformat(),
                        )
                    )

        return results

    def _invoke_tools_sequential(
        self, calls: List[ToolCallRequest], stop_on_error: bool = False
    ) -> List[ToolCallResult]:
        """Execute tool calls sequentially."""
        results: List[ToolCallResult] = []

        for call in calls:
            result = self.invoke_tool(
                call.tool_name,
                call.arguments,
                call.call_id or str(uuid.uuid4())[:8],
            )
            results.append(result)

            if not result.success and stop_on_error:
                logger.warning(f"Stopping execution due to error in '{call.tool_name}'")
                break

        return results

    def invoke_tools_from_input(self, input_data: MultiToolInput) -> MultiToolOutput:
        """Invoke tools from a MultiToolInput object.

        Args:
            input_data: MultiToolInput with calls configuration

        Returns:
            MultiToolOutput with results
        """
        return self.invoke_tools(
            calls=input_data.calls,
            parallel=input_data.parallel,
            stop_on_error=input_data.stop_on_error,
        )

    # ─────────────────────────────────────────────────────────────
    # Convenience Methods for Common Patterns
    # ─────────────────────────────────────────────────────────────

    def invoke_many(
        self,
        tool_name: str,
        arguments_list: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> MultiToolOutput:
        """Invoke the same tool multiple times with different arguments.

        Args:
            tool_name: Name of the tool to invoke
            arguments_list: List of argument dicts
            parallel: Execute in parallel (True) or sequential (False)

        Returns:
            MultiToolOutput with all results
        """
        calls = [
            ToolCallRequest(tool_name=tool_name, arguments=args)
            for args in arguments_list
        ]
        return self.invoke_tools(calls, parallel=parallel)

    def invoke_chain(
        self,
        calls: List[ToolCallRequest],
        result_key: str = "result",
    ) -> MultiToolOutput:
        """Invoke tools in a chain, passing results to next tool.

        Each tool receives the previous tool's result under the specified key.

        Args:
            calls: List of tool calls (executed sequentially)
            result_key: Key to use for passing previous result

        Returns:
            MultiToolOutput with all results
        """
        results: List[ToolCallResult] = []
        start_time = time.time()
        previous_result: Optional[Dict[str, Any]] = None

        for call in calls:
            # Merge previous result into arguments
            arguments = dict(call.arguments)
            if previous_result is not None:
                arguments[result_key] = previous_result

            result = self.invoke_tool(
                call.tool_name,
                arguments,
                call.call_id or str(uuid.uuid4())[:8],
            )
            results.append(result)

            if not result.success:
                break

            previous_result = result.result

        total_duration_ms = (time.time() - start_time) * 1000
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        return MultiToolOutput(
            success=failed == 0,
            error_code=ErrorCode.OK if failed == 0 else ErrorCode.TOOL_FAILED,
            message="ok" if failed == 0 else f"Chain failed at step {len(results)}",
            results=results,
            total_calls=len(results),
            successful_calls=successful,
            failed_calls=failed,
            total_duration_ms=total_duration_ms,
            timestamp=datetime.now().isoformat(),
        )

    # ─────────────────────────────────────────────────────────────
    # Async Tool Invocation
    # ─────────────────────────────────────────────────────────────

    async def ainvoke_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None,
    ) -> ToolCallResult:
        """Async version of invoke_tool.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Arguments to pass to the tool
            call_id: Optional call ID (auto-generated if not provided)

        Returns:
            ToolCallResult with the result or error
        """
        # Run sync tool invocation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.invoke_tool(tool_name, arguments, call_id)
        )

    async def ainvoke_tools(
        self,
        calls: List[ToolCallRequest],
        parallel: bool = True,
        stop_on_error: bool = False,
    ) -> MultiToolOutput:
        """Async version of invoke_tools.

        Args:
            calls: List of tool call requests
            parallel: Execute in parallel (True) or sequential (False)
            stop_on_error: Stop on first error (only for sequential mode)

        Returns:
            MultiToolOutput with all results
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        if not calls:
            return MultiToolOutput(
                success=False,
                error_code=ErrorCode.INVALID_INPUT,
                message="No tool calls provided",
                timestamp=timestamp,
            )

        if parallel:
            results = await self._ainvoke_tools_parallel(calls)
        else:
            results = await self._ainvoke_tools_sequential(calls, stop_on_error)

        total_duration_ms = (time.time() - start_time) * 1000
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        return MultiToolOutput(
            success=failed == 0,
            error_code=ErrorCode.OK if failed == 0 else ErrorCode.TOOL_FAILED,
            message="ok" if failed == 0 else f"{failed} tool(s) failed",
            results=results,
            total_calls=len(results),
            successful_calls=successful,
            failed_calls=failed,
            total_duration_ms=total_duration_ms,
            timestamp=timestamp,
        )

    async def _ainvoke_tools_parallel(
        self, calls: List[ToolCallRequest]
    ) -> List[ToolCallResult]:
        """Execute tool calls in parallel using asyncio."""
        tasks = [
            self.ainvoke_tool(
                call.tool_name,
                call.arguments,
                call.call_id or str(uuid.uuid4())[:8],
            )
            for call in calls
        ]
        return await asyncio.gather(*tasks)

    async def _ainvoke_tools_sequential(
        self, calls: List[ToolCallRequest], stop_on_error: bool = False
    ) -> List[ToolCallResult]:
        """Execute tool calls sequentially (async)."""
        results: List[ToolCallResult] = []

        for call in calls:
            result = await self.ainvoke_tool(
                call.tool_name,
                call.arguments,
                call.call_id or str(uuid.uuid4())[:8],
            )
            results.append(result)

            if not result.success and stop_on_error:
                logger.warning(f"Stopping execution due to error in '{call.tool_name}'")
                break

        return results

    async def ainvoke_many(
        self,
        tool_name: str,
        arguments_list: List[Dict[str, Any]],
        parallel: bool = True,
    ) -> MultiToolOutput:
        """Async version of invoke_many.

        Args:
            tool_name: Name of the tool to invoke
            arguments_list: List of argument dicts
            parallel: Execute in parallel (True) or sequential (False)

        Returns:
            MultiToolOutput with all results
        """
        calls = [
            ToolCallRequest(tool_name=tool_name, arguments=args)
            for args in arguments_list
        ]
        return await self.ainvoke_tools(calls, parallel=parallel)

    async def ainvoke_chain(
        self,
        calls: List[ToolCallRequest],
        result_key: str = "result",
    ) -> MultiToolOutput:
        """Async version of invoke_chain.

        Each tool receives the previous tool's result under the specified key.

        Args:
            calls: List of tool calls (executed sequentially)
            result_key: Key to use for passing previous result

        Returns:
            MultiToolOutput with all results
        """
        results: List[ToolCallResult] = []
        start_time = time.time()
        previous_result: Optional[Dict[str, Any]] = None

        for call in calls:
            # Merge previous result into arguments
            arguments = dict(call.arguments)
            if previous_result is not None:
                arguments[result_key] = previous_result

            result = await self.ainvoke_tool(
                call.tool_name,
                arguments,
                call.call_id or str(uuid.uuid4())[:8],
            )
            results.append(result)

            if not result.success:
                break

            previous_result = result.result

        total_duration_ms = (time.time() - start_time) * 1000
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        return MultiToolOutput(
            success=failed == 0,
            error_code=ErrorCode.OK if failed == 0 else ErrorCode.TOOL_FAILED,
            message="ok" if failed == 0 else f"Chain failed at step {len(results)}",
            results=results,
            total_calls=len(results),
            successful_calls=successful,
            failed_calls=failed,
            total_duration_ms=total_duration_ms,
            timestamp=datetime.now().isoformat(),
        )

    # ─────────────────────────────────────────────────────────────
    # RAG Pipeline (backward compatible)
    # ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> SearchOutput:
        """Run knowledge search tool."""
        tool = self.registry.get("knowledge_search")
        result = tool.invoke({"query": query, "top_k": top_k})
        if isinstance(result, SearchOutput):
            return result
        return SearchOutput(success=False, message="unexpected result", query=query)

    def ask(self, question: str, top_k: int = 3) -> RAGOutput:
        """Full RAG pipeline: search + generate answer.

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            RAGOutput with answer and sources
        """
        # Step 1: Search
        search_result = self.search(question, top_k=top_k)

        if not search_result.success:
            return RAGOutput(
                success=False,
                message=search_result.message,
                question=question,
                answer="检索失败，无法回答问题。",
                sources=[],
            )

        # Step 2: Format context
        context_parts = []
        for item in search_result.results:
            context_parts.append(item.content)
        context = "\n\n".join(context_parts)

        # Step 3: Generate answer with LLM
        prompt = self.prompt_template.format(context=context, question=question)

        try:
            llm_result = self.llm.invoke(prompt)
            if hasattr(llm_result, "content"):
                answer = llm_result.content
            else:
                answer = str(llm_result)
        except Exception as e:
            return RAGOutput(
                success=False,
                message=f"LLM error: {str(e)}",
                question=question,
                answer="生成回答时出错。",
                sources=search_result.results,
            )

        return RAGOutput(
            success=True,
            message="ok",
            question=question,
            answer=answer,
            sources=search_result.results,
        )

    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Run the agent on a query (backward compatible).

        Returns dict format for compatibility.
        """
        result = self.ask(query, top_k=top_k)
        return result.model_dump()

    # ─────────────────────────────────────────────────────────────
    # Async RAG Pipeline
    # ─────────────────────────────────────────────────────────────

    async def asearch(self, query: str, top_k: int = 3) -> SearchOutput:
        """Async version of search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.search(query, top_k))

    async def aask(self, question: str, top_k: int = 3) -> RAGOutput:
        """Async full RAG pipeline: search + generate answer.

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            RAGOutput with answer and sources
        """
        # Step 1: Async Search
        search_result = await self.asearch(question, top_k=top_k)

        if not search_result.success:
            return RAGOutput(
                success=False,
                message=search_result.message,
                question=question,
                answer="检索失败，无法回答问题。",
                sources=[],
            )

        # Step 2: Format context
        context_parts = []
        for item in search_result.results:
            context_parts.append(item.content)
        context = "\n\n".join(context_parts)

        # Step 3: Generate answer with LLM (async if LLM supports it)
        prompt = self.prompt_template.format(context=context, question=question)

        try:
            # Try async invoke if available
            if hasattr(self.llm, "ainvoke"):
                llm_result = await self.llm.ainvoke(prompt)
            else:
                # Fall back to sync invoke in thread pool
                loop = asyncio.get_event_loop()
                llm_result = await loop.run_in_executor(
                    None, lambda: self.llm.invoke(prompt)
                )

            if hasattr(llm_result, "content"):
                answer = llm_result.content
            else:
                answer = str(llm_result)
        except Exception as e:
            return RAGOutput(
                success=False,
                message=f"LLM error: {str(e)}",
                question=question,
                answer="生成回答时出错。",
                sources=search_result.results,
            )

        return RAGOutput(
            success=True,
            message="ok",
            question=question,
            answer=answer,
            sources=search_result.results,
        )

    async def arun(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Async version of run.

        Returns dict format for compatibility.
        """
        result = await self.aask(query, top_k=top_k)
        return result.model_dump()
