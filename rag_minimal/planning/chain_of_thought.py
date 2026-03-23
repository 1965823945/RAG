"""Chain of Thought Module.

Implements step-by-step reasoning with explicit thought tracking.
"""

import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseLLM

from rag_minimal.schemas import (
    ChainOfThoughtResult,
    ThoughtStep,
)

# Chain of Thought prompt template
COT_PROMPT = """你是一个善于逐步推理的助手。请使用思维链方法来回答问题。

## 问题
{question}

## 上下文信息
{context}

## 要求
请逐步思考，每一步都要清晰地展示你的推理过程：
1. 首先理解问题的核心
2. 分析相关信息
3. 逐步推导
4. 得出结论

## 输出格式
请按以下格式输出你的思考过程：

思考步骤1: [你的第一步思考]
思考步骤2: [你的第二步思考]
思考步骤3: [继续推理...]
...
最终答案: [你的结论]
置信度: [0-100的数字，表示你对答案的确信程度]

请开始逐步思考："""


# ReAct style prompt for action-based reasoning
REACT_PROMPT = """你是一个能够使用工具的智能助手。请使用 ReAct (Reasoning and Acting) 方法来解决问题。

## 问题
{question}

## 可用工具
{tools}

## 格式
请按以下格式交替进行思考和行动：

思考: [分析当前情况，决定下一步]
行动: [选择要执行的工具]
行动输入: [工具的输入参数，JSON格式]
观察: [工具执行的结果 - 这部分由系统填充]

...（可以重复多次）

思考: [基于所有观察得出最终结论]
最终答案: [给出完整答案]

请开始："""


class ChainOfThought:
    """Implements Chain of Thought reasoning.

    This module provides:
    1. Step-by-step reasoning with thought tracking
    2. ReAct-style reasoning with actions
    3. Confidence estimation
    4. Thought chain visualization
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        max_steps: int = 10,
        tool_executor: Callable[[str, dict[str, Any]], str] | None = None,
    ):
        """Initialize the Chain of Thought reasoner.

        Args:
            llm: Language model for reasoning
            max_steps: Maximum reasoning steps
            tool_executor: Optional function to execute tools (for ReAct)
        """
        self.llm = llm
        self.max_steps = max_steps
        self.tool_executor = tool_executor

    def reason(
        self,
        question: str,
        context: str = "",
        use_react: bool = False,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> ChainOfThoughtResult:
        """Perform chain of thought reasoning.

        Args:
            question: The question to reason about
            context: Optional context information
            use_react: Whether to use ReAct-style reasoning
            available_tools: List of available tools for ReAct

        Returns:
            ChainOfThoughtResult with reasoning steps and final answer
        """
        if use_react and self.tool_executor:
            return self._react_reason(question, context, available_tools or [])
        elif self.llm:
            return self._llm_reason(question, context)
        else:
            return self._rule_based_reason(question, context)

    def _llm_reason(self, question: str, context: str) -> ChainOfThoughtResult:
        """Use LLM for chain of thought reasoning."""
        prompt = COT_PROMPT.format(
            question=question, context=context if context else "无额外上下文"
        )

        try:
            result = self.llm.invoke(prompt)
            if hasattr(result, "content"):
                response_text = result.content
            else:
                response_text = str(result)

            return self._parse_cot_response(question, response_text)

        except Exception as e:
            # Return error result
            return ChainOfThoughtResult(
                question=question,
                thoughts=[
                    ThoughtStep(
                        step_number=1,
                        thought=f"推理过程出错: {str(e)}",
                        timestamp=datetime.now().isoformat(),
                    )
                ],
                final_answer="推理过程中发生错误",
                confidence=0.0,
                total_steps=1,
            )

    def _react_reason(
        self,
        question: str,
        context: str,
        available_tools: list[dict[str, Any]],
    ) -> ChainOfThoughtResult:
        """Use ReAct-style reasoning with tool execution."""
        tools_desc = self._format_tools(available_tools)
        thoughts: list[ThoughtStep] = []
        step_number = 0

        # Initial prompt
        messages = [
            REACT_PROMPT.format(
                question=question,
                tools=tools_desc,
            )
        ]

        if context:
            messages[0] += f"\n\n## 已有上下文\n{context}"

        accumulated_response = ""

        while step_number < self.max_steps:
            step_number += 1

            try:
                # Get next reasoning step from LLM
                if self.llm:
                    result = self.llm.invoke("\n".join(messages))
                    if hasattr(result, "content"):
                        response = result.content
                    else:
                        response = str(result)
                else:
                    break

                accumulated_response += response

                # Parse the response
                thought_match = re.search(
                    r"思考:\s*(.+?)(?=行动:|最终答案:|$)", response, re.DOTALL
                )
                action_match = re.search(
                    r"行动:\s*(.+?)(?=行动输入:|$)", response, re.DOTALL
                )
                action_input_match = re.search(
                    r"行动输入:\s*(.+?)(?=观察:|思考:|$)", response, re.DOTALL
                )
                final_answer_match = re.search(
                    r"最终答案:\s*(.+?)$", response, re.DOTALL
                )

                thought_text = thought_match.group(1).strip() if thought_match else ""

                # Check if we have a final answer
                if final_answer_match:
                    thoughts.append(
                        ThoughtStep(
                            step_number=step_number,
                            thought=thought_text,
                            timestamp=datetime.now().isoformat(),
                        )
                    )

                    return ChainOfThoughtResult(
                        question=question,
                        thoughts=thoughts,
                        final_answer=final_answer_match.group(1).strip(),
                        confidence=self._estimate_confidence(thoughts),
                        total_steps=step_number,
                    )

                # Execute action if present
                if action_match and action_input_match and self.tool_executor:
                    action = action_match.group(1).strip()
                    try:
                        action_input = json.loads(action_input_match.group(1).strip())
                    except json.JSONDecodeError:
                        action_input = {"input": action_input_match.group(1).strip()}

                    # Execute the tool
                    observation = self.tool_executor(action, action_input)

                    thoughts.append(
                        ThoughtStep(
                            step_number=step_number,
                            thought=thought_text,
                            action=action,
                            action_input=action_input,
                            observation=observation,
                            timestamp=datetime.now().isoformat(),
                        )
                    )

                    # Add observation to messages for next iteration
                    messages.append(response)
                    messages.append(f"观察: {observation}")
                else:
                    # No action, just thought
                    thoughts.append(
                        ThoughtStep(
                            step_number=step_number,
                            thought=thought_text,
                            timestamp=datetime.now().isoformat(),
                        )
                    )
                    break

            except Exception as e:
                thoughts.append(
                    ThoughtStep(
                        step_number=step_number,
                        thought=f"执行出错: {str(e)}",
                        timestamp=datetime.now().isoformat(),
                    )
                )
                break

        # Extract final answer from accumulated response if not found
        final_answer = self._extract_conclusion(accumulated_response, thoughts)

        return ChainOfThoughtResult(
            question=question,
            thoughts=thoughts,
            final_answer=final_answer,
            confidence=self._estimate_confidence(thoughts),
            total_steps=len(thoughts),
        )

    def _rule_based_reason(self, question: str, context: str) -> ChainOfThoughtResult:
        """Rule-based reasoning without LLM.

        Performs structured reasoning based on question type.
        """
        thoughts: list[ThoughtStep] = []
        timestamp = datetime.now().isoformat()

        # Step 1: Understand the question
        thoughts.append(
            ThoughtStep(
                step_number=1,
                thought=f"分析问题：「{question}」",
                timestamp=timestamp,
            )
        )

        # Step 2: Identify question type
        question_type = self._identify_question_type(question)
        thoughts.append(
            ThoughtStep(
                step_number=2,
                thought=f"问题类型：{question_type}",
                timestamp=timestamp,
            )
        )

        # Step 3: Analyze context
        if context:
            key_points = self._extract_key_points(context)
            thoughts.append(
                ThoughtStep(
                    step_number=3,
                    thought=f"从上下文中提取关键信息：{key_points}",
                    timestamp=timestamp,
                )
            )
        else:
            thoughts.append(
                ThoughtStep(
                    step_number=3,
                    thought="没有额外上下文信息可供参考",
                    timestamp=timestamp,
                )
            )

        # Step 4: Formulate answer approach
        approach = self._determine_approach(question_type)
        thoughts.append(
            ThoughtStep(
                step_number=4,
                thought=f"回答策略：{approach}",
                timestamp=timestamp,
            )
        )

        # Generate answer based on context
        if context:
            final_answer = self._generate_answer_from_context(question, context)
            confidence = 0.7
        else:
            final_answer = "需要更多上下文信息才能回答这个问题。"
            confidence = 0.3

        return ChainOfThoughtResult(
            question=question,
            thoughts=thoughts,
            final_answer=final_answer,
            confidence=confidence,
            total_steps=len(thoughts),
        )

    def _parse_cot_response(self, question: str, response: str) -> ChainOfThoughtResult:
        """Parse chain of thought response from LLM."""
        thoughts: list[ThoughtStep] = []

        # Extract thinking steps
        step_pattern = r"思考步骤(\d+):\s*(.+?)(?=思考步骤\d+:|最终答案:|$)"
        matches = re.findall(step_pattern, response, re.DOTALL)

        for step_num, thought_text in matches:
            thoughts.append(
                ThoughtStep(
                    step_number=int(step_num),
                    thought=thought_text.strip(),
                    timestamp=datetime.now().isoformat(),
                )
            )

        # If no structured steps found, try to split by newlines
        if not thoughts:
            lines = response.split("\n")
            step_num = 0
            for line in lines:
                line = line.strip()
                if (
                    line
                    and not line.startswith("最终答案")
                    and not line.startswith("置信度")
                ):
                    step_num += 1
                    thoughts.append(
                        ThoughtStep(
                            step_number=step_num,
                            thought=line,
                            timestamp=datetime.now().isoformat(),
                        )
                    )

        # Extract final answer
        final_match = re.search(r"最终答案:\s*(.+?)(?=置信度:|$)", response, re.DOTALL)
        final_answer = (
            final_match.group(1).strip() if final_match else response.split("\n")[-1]
        )

        # Extract confidence
        conf_match = re.search(r"置信度:\s*(\d+)", response)
        confidence = int(conf_match.group(1)) / 100 if conf_match else 0.5

        return ChainOfThoughtResult(
            question=question,
            thoughts=thoughts,
            final_answer=final_answer,
            confidence=min(1.0, max(0.0, confidence)),
            total_steps=len(thoughts),
        )

    def _identify_question_type(self, question: str) -> str:
        """Identify the type of question."""
        if any(w in question for w in ["什么", "是什么"]):
            return "定义/解释类问题"
        elif any(w in question for w in ["为什么", "原因"]):
            return "因果/原因类问题"
        elif any(w in question for w in ["如何", "怎么", "怎样"]):
            return "方法/过程类问题"
        elif any(w in question for w in ["比较", "区别", "不同"]):
            return "比较类问题"
        elif any(w in question for w in ["是否", "能否", "可以"]):
            return "是非判断类问题"
        else:
            return "综合类问题"

    def _extract_key_points(self, context: str) -> str:
        """Extract key points from context."""
        # Simple extraction - get first few sentences
        sentences = re.split(r"[。！？\n]", context)
        key_sentences = [s.strip() for s in sentences[:3] if s.strip()]
        return "；".join(key_sentences) if key_sentences else "无明显关键点"

    def _determine_approach(self, question_type: str) -> str:
        """Determine the approach based on question type."""
        approaches = {
            "定义/解释类问题": "提供清晰的定义和解释",
            "因果/原因类问题": "分析原因和结果的关系",
            "方法/过程类问题": "给出步骤化的方法说明",
            "比较类问题": "列出异同点进行对比",
            "是非判断类问题": "给出明确的判断和理由",
            "综合类问题": "综合分析多个方面",
        }
        return approaches.get(question_type, "综合分析回答")

    def _generate_answer_from_context(self, question: str, context: str) -> str:
        """Generate answer based on context (simple extraction)."""
        # Simple context-based answer generation
        # In real usage, this would use the LLM
        return f"根据提供的信息，关于「{question[:20]}...」的回答：{context[:200]}..."

    def _format_tools(self, tools: list[dict[str, Any]]) -> str:
        """Format tools description for prompt."""
        if not tools:
            return "无可用工具"

        lines = []
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "无描述")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def _estimate_confidence(self, thoughts: list[ThoughtStep]) -> float:
        """Estimate confidence based on reasoning chain."""
        if not thoughts:
            return 0.0

        # More steps generally means more thorough reasoning
        step_score = min(1.0, len(thoughts) / 5)

        # Check if observations are present (indicates tool use)
        observation_count = sum(1 for t in thoughts if t.observation)
        observation_score = (
            min(1.0, observation_count / 3) if observation_count > 0 else 0.5
        )

        return (step_score + observation_score) / 2

    def _extract_conclusion(
        self,
        response: str,
        thoughts: list[ThoughtStep],
    ) -> str:
        """Extract conclusion from response or thoughts."""
        # Try to find explicit final answer
        final_match = re.search(r"最终答案:\s*(.+?)$", response, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()

        # Use last observation if available
        for thought in reversed(thoughts):
            if thought.observation:
                return thought.observation

        # Use last thought
        if thoughts:
            return thoughts[-1].thought

        return "无法得出结论"

    def visualize_chain(self, result: ChainOfThoughtResult) -> str:
        """Create a visual representation of the thought chain."""
        lines = [
            f"问题: {result.question}",
            "=" * 50,
            "思维链:",
        ]

        for thought in result.thoughts:
            lines.append(f"\n步骤 {thought.step_number}:")
            lines.append(f"  思考: {thought.thought}")
            if thought.action:
                lines.append(f"  行动: {thought.action}")
            if thought.action_input:
                lines.append(
                    f"  输入: {json.dumps(thought.action_input, ensure_ascii=False)}"
                )
            if thought.observation:
                lines.append(f"  观察: {thought.observation}")

        lines.extend(
            [
                "",
                "=" * 50,
                f"最终答案: {result.final_answer}",
                f"置信度: {result.confidence:.0%}",
                f"总步骤: {result.total_steps}",
            ]
        )

        return "\n".join(lines)
