"""Planning Agent - Autonomous task planning and execution.

Integrates Task Decomposition, Chain of Thought, and Self-Reflection
to create an autonomous agent capable of complex reasoning.
"""

import uuid
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from langchain_core.language_models import BaseLLM

from rag_minimal.schemas import (
    TaskDecomposition,
    TaskStatus,
    ChainOfThoughtResult,
    SelfReflectionResult,
    PlanningAgentState,
    PlanningAgentOutput,
    SearchOutput,
)
from rag_minimal.planning.task_decomposer import TaskDecomposer
from rag_minimal.planning.chain_of_thought import ChainOfThought
from rag_minimal.planning.self_reflection import SelfReflection


class PlanningAgent:
    """Autonomous Planning Agent.

    This agent integrates three key capabilities:
    1. Task Decomposition - Breaking complex tasks into subtasks
    2. Chain of Thought - Step-by-step reasoning
    3. Self-Reflection - Quality assessment and improvement

    The agent follows this workflow:
    1. Decompose the task into subtasks
    2. Execute each subtask with chain-of-thought reasoning
    3. Reflect on the results and improve if needed
    4. Repeat until quality threshold is met or max iterations reached
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        search_func: Optional[Callable[[str, int], SearchOutput]] = None,
        quality_threshold: float = 0.6,
        max_iterations: int = 3,
        verbose: bool = False,
    ):
        """Initialize the Planning Agent.

        Args:
            llm: Language model for reasoning
            search_func: Function to search knowledge base
            quality_threshold: Minimum quality score to accept
            max_iterations: Maximum improvement iterations
            verbose: Whether to print detailed progress
        """
        self.llm = llm
        self.search_func = search_func
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Initialize sub-modules
        self.decomposer = TaskDecomposer(llm=llm)
        self.cot = ChainOfThought(
            llm=llm,
            tool_executor=self._execute_tool if search_func else None,
        )
        self.reflection = SelfReflection(
            llm=llm,
            quality_threshold=quality_threshold,
        )

        # Available tools for ReAct reasoning
        self.available_tools = [
            {
                "name": "knowledge_search",
                "description": '搜索知识库获取相关信息。输入: {"query": "搜索词", "top_k": 3}',
            }
        ]

    def run(
        self,
        query: str,
        top_k: int = 3,
        use_decomposition: bool = True,
        use_cot: bool = True,
        use_reflection: bool = True,
    ) -> PlanningAgentOutput:
        """Run the planning agent on a query.

        Args:
            query: User's question or task
            top_k: Number of documents to retrieve
            use_decomposition: Whether to use task decomposition
            use_cot: Whether to use chain of thought reasoning
            use_reflection: Whether to use self-reflection

        Returns:
            PlanningAgentOutput with answer and execution details
        """
        # Initialize state
        task_id = str(uuid.uuid4())[:8]
        state = PlanningAgentState(
            task_id=task_id,
            original_query=query,
            start_time=datetime.now().isoformat(),
        )

        self._log(f"开始处理任务 [{task_id}]: {query[:50]}...")

        try:
            # Phase 1: Task Decomposition
            if use_decomposition:
                state.decomposition = self._decompose_task(query)
                self._log(f"任务分解完成: {state.decomposition.total_steps} 个子任务")

            # Phase 2: Execute with Chain of Thought
            context = self._gather_context(query, top_k)

            if use_cot:
                state.reasoning = self._reason_with_cot(
                    query, context, use_react=self.search_func is not None
                )
                answer = state.reasoning.final_answer or ""
                self._log(f"推理完成: {state.reasoning.total_steps} 步")
            else:
                answer = self._simple_answer(query, context)

            # Phase 3: Self-Reflection Loop
            iteration = 0
            while use_reflection and iteration < self.max_iterations:
                iteration += 1
                state.iterations = iteration

                self._log(f"开始第 {iteration} 轮反思...")
                state.reflection = self._reflect_on_answer(
                    query, answer, state.reasoning
                )

                self._log(f"反思评分: {state.reflection.overall_score:.0%}")

                if not state.reflection.should_retry:
                    self._log("质量达标，无需重试")
                    break

                if iteration < self.max_iterations:
                    self._log(f"重试原因: {state.reflection.retry_reason}")
                    # Improve the answer based on reflection
                    answer = self._improve_answer(
                        query, answer, context, state.reflection
                    )
                    if use_cot:
                        state.reasoning = self._reason_with_cot(
                            query,
                            context,
                            previous_answer=answer,
                            improvements=state.reflection.improvements,
                        )
                        answer = state.reasoning.final_answer or answer

            # Finalize
            state.status = TaskStatus.COMPLETED
            state.final_answer = answer
            state.end_time = datetime.now().isoformat()

            # Record execution history
            state.execution_history.append(
                {
                    "phase": "completed",
                    "timestamp": state.end_time,
                    "iterations": state.iterations,
                }
            )

            self._log(f"任务完成: 经过 {state.iterations} 轮迭代")

            return self._create_output(state)

        except Exception as e:
            state.status = TaskStatus.FAILED
            state.end_time = datetime.now().isoformat()

            return PlanningAgentOutput(
                success=False,
                message=f"执行失败: {str(e)}",
                task_id=task_id,
                query=query,
                answer=f"处理过程中发生错误: {str(e)}",
                state=state,
            )

    def _decompose_task(self, query: str) -> TaskDecomposition:
        """Decompose the task into subtasks."""
        return self.decomposer.decompose(query)

    def _gather_context(self, query: str, top_k: int) -> str:
        """Gather context from knowledge base."""
        if not self.search_func:
            return ""

        try:
            result = self.search_func(query, top_k)
            if result.success and result.results:
                context_parts = [item.content for item in result.results]
                return "\n\n".join(context_parts)
        except Exception as e:
            self._log(f"上下文检索失败: {e}")

        return ""

    def _reason_with_cot(
        self,
        query: str,
        context: str,
        use_react: bool = False,
        previous_answer: Optional[str] = None,
        improvements: Optional[List[str]] = None,
    ) -> ChainOfThoughtResult:
        """Perform chain of thought reasoning."""
        # Enhance context with previous answer and improvements if available
        enhanced_context = context
        if previous_answer:
            enhanced_context += f"\n\n之前的回答：{previous_answer}"
        if improvements:
            enhanced_context += "\n\n需要改进的方面：" + "；".join(improvements)

        return self.cot.reason(
            question=query,
            context=enhanced_context,
            use_react=use_react,
            available_tools=self.available_tools if use_react else None,
        )

    def _reflect_on_answer(
        self,
        query: str,
        answer: str,
        reasoning: Optional[ChainOfThoughtResult],
    ) -> SelfReflectionResult:
        """Reflect on the generated answer."""
        return self.reflection.reflect(
            question=query,
            answer=answer,
            reasoning=reasoning,
        )

    def _improve_answer(
        self,
        query: str,
        previous_answer: str,
        context: str,
        reflection: SelfReflectionResult,
    ) -> str:
        """Improve the answer based on reflection feedback."""
        if not self.llm:
            return previous_answer

        improvement_prompt = f"""基于以下反馈改进回答：

原问题：{query}

之前的回答：{previous_answer}

反馈意见：
- 总体评分：{reflection.overall_score:.0%}
- 发现的问题：{"; ".join(sum([r.issues for r in reflection.reflections], []))}
- 改进建议：{"; ".join(reflection.improvements)}

参考上下文：{context[:500] if context else "无"}

请提供改进后的回答："""

        try:
            result = self.llm.invoke(improvement_prompt)
            if hasattr(result, "content"):
                return result.content
            return str(result)
        except Exception:
            return previous_answer

    def _simple_answer(self, query: str, context: str) -> str:
        """Generate a simple answer without CoT."""
        if not self.llm:
            if context:
                return f"根据检索到的信息：{context[:500]}..."
            return "需要更多信息来回答这个问题。"

        prompt = f"""请回答以下问题。

问题：{query}

参考信息：{context if context else "无额外参考信息"}

请直接回答："""

        try:
            result = self.llm.invoke(prompt)
            if hasattr(result, "content"):
                return result.content
            return str(result)
        except Exception as e:
            return f"生成回答时出错: {e}"

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool (for ReAct reasoning)."""
        if tool_name == "knowledge_search" and self.search_func:
            query = tool_input.get("query", tool_input.get("input", ""))
            top_k = tool_input.get("top_k", 3)

            result = self.search_func(query, top_k)
            if result.success and result.results:
                return "\n".join(
                    [
                        f"- {item.content[:200]}... (来源: {item.source})"
                        for item in result.results
                    ]
                )
            return "未找到相关信息"

        return f"未知工具: {tool_name}"

    def _create_output(self, state: PlanningAgentState) -> PlanningAgentOutput:
        """Create the final output from state."""
        subtasks_completed = 0
        subtasks_total = 0

        if state.decomposition:
            subtasks_total = state.decomposition.total_steps
            subtasks_completed = sum(
                1
                for t in state.decomposition.subtasks
                if t.status == TaskStatus.COMPLETED
            )

        reasoning_steps = 0
        confidence = 0.0
        if state.reasoning:
            reasoning_steps = state.reasoning.total_steps
            confidence = state.reasoning.confidence

        quality_score = 0.0
        if state.reflection:
            quality_score = state.reflection.overall_score

        return PlanningAgentOutput(
            success=True,
            message="ok",
            task_id=state.task_id,
            query=state.original_query,
            answer=state.final_answer or "",
            subtasks_completed=subtasks_completed,
            subtasks_total=subtasks_total,
            reasoning_steps=reasoning_steps,
            iterations=state.iterations,
            confidence=confidence,
            quality_score=quality_score,
            state=state,
        )

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def get_execution_summary(self, output: PlanningAgentOutput) -> str:
        """Get a human-readable summary of the execution."""
        lines = [
            f"任务ID: {output.task_id}",
            f"问题: {output.query[:50]}...",
            "=" * 50,
            f"状态: {'成功' if output.success else '失败'}",
            f"答案: {output.answer[:200]}..."
            if len(output.answer) > 200
            else f"答案: {output.answer}",
            "",
            "执行统计:",
            f"  - 子任务: {output.subtasks_completed}/{output.subtasks_total}",
            f"  - 推理步骤: {output.reasoning_steps}",
            f"  - 迭代次数: {output.iterations}",
            f"  - 置信度: {output.confidence:.0%}",
            f"  - 质量评分: {output.quality_score:.0%}",
        ]

        if output.state and output.state.decomposition:
            lines.extend(
                [
                    "",
                    "任务分解:",
                    f"  复杂度: {output.state.decomposition.complexity}",
                    f"  预计时间: {output.state.decomposition.estimated_time}",
                ]
            )
            for task in output.state.decomposition.subtasks[:5]:
                status_icon = "✓" if task.status == TaskStatus.COMPLETED else "○"
                lines.append(f"  {status_icon} {task.description}")

        if output.state and output.state.reasoning:
            lines.extend(
                [
                    "",
                    "推理链:",
                ]
            )
            for thought in output.state.reasoning.thoughts[:5]:
                lines.append(f"  {thought.step_number}. {thought.thought[:60]}...")

        if output.state and output.state.reflection:
            lines.extend(
                [
                    "",
                    "反思结果:",
                    f"  总体评分: {output.state.reflection.overall_score:.0%}",
                ]
            )
            for ref in output.state.reflection.reflections[:3]:
                lines.append(f"  - {ref.aspect}: {ref.score:.0%}")

        return "\n".join(lines)

    # Convenience methods for individual components

    def decompose_only(self, query: str) -> TaskDecomposition:
        """Only perform task decomposition."""
        return self.decomposer.decompose(query)

    def reason_only(self, query: str, context: str = "") -> ChainOfThoughtResult:
        """Only perform chain of thought reasoning."""
        return self.cot.reason(query, context)

    def reflect_only(
        self,
        query: str,
        answer: str,
        reasoning: Optional[ChainOfThoughtResult] = None,
    ) -> SelfReflectionResult:
        """Only perform self-reflection."""
        return self.reflection.reflect(query, answer, reasoning)
