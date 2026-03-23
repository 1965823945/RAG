"""Self-Reflection Module.

Implements self-assessment and improvement mechanisms for the agent.
"""

import json
import re

from langchain_core.language_models import BaseLLM

from rag_minimal.schemas import (
    ChainOfThoughtResult,
    ReflectionItem,
    ReflectionType,
    SelfReflectionResult,
    TaskDecomposition,
)

# Self-reflection prompt template
REFLECTION_PROMPT = """你是一个严格的质量评估专家。请对以下内容进行全面的自我反思和评估。

## 原始问题
{question}

## 生成的答案
{answer}

## 推理过程
{reasoning}

## 评估要求
请从以下维度进行评估：

1. **质量检查** (quality_check)
   - 答案是否准确
   - 答案是否完整
   - 逻辑是否清晰

2. **错误分析** (error_analysis)
   - 是否存在事实错误
   - 是否存在逻辑谬误
   - 是否有遗漏重要信息

3. **完整性检查** (completeness)
   - 是否回答了问题的所有方面
   - 是否需要补充更多信息

4. **一致性检查** (consistency)
   - 答案内部是否一致
   - 是否与已知事实一致

## 输出格式（JSON）
请严格按以下JSON格式输出评估结果：
```json
{{
    "overall_score": 0.8,
    "should_retry": false,
    "retry_reason": null,
    "reflections": [
        {{
            "reflection_type": "quality_check",
            "aspect": "准确性",
            "assessment": "评估说明",
            "score": 0.9,
            "issues": ["问题1", "问题2"],
            "suggestions": ["建议1", "建议2"]
        }}
    ],
    "improvements": ["总体改进建议1", "总体改进建议2"]
}}
```

请开始评估："""


class SelfReflection:
    """Implements self-reflection and quality assessment.

    This module provides:
    1. Quality assessment of generated answers
    2. Error detection and analysis
    3. Improvement suggestions
    4. Retry decision making
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        quality_threshold: float = 0.6,
        max_retries: int = 2,
    ):
        """Initialize the self-reflection module.

        Args:
            llm: Language model for reflection
            quality_threshold: Minimum quality score to accept (0-1)
            max_retries: Maximum number of retry attempts
        """
        self.llm = llm
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries

    def reflect(
        self,
        question: str,
        answer: str,
        reasoning: ChainOfThoughtResult | None = None,
        task_decomposition: TaskDecomposition | None = None,
        context: str | None = None,
    ) -> SelfReflectionResult:
        """Perform self-reflection on the generated answer.

        Args:
            question: Original question
            answer: Generated answer to evaluate
            reasoning: Optional chain of thought result
            task_decomposition: Optional task decomposition
            context: Optional additional context

        Returns:
            SelfReflectionResult with assessment and suggestions
        """
        if self.llm:
            return self._llm_reflect(question, answer, reasoning)
        else:
            return self._rule_based_reflect(question, answer, reasoning, context)

    def _llm_reflect(
        self,
        question: str,
        answer: str,
        reasoning: ChainOfThoughtResult | None,
    ) -> SelfReflectionResult:
        """Use LLM for self-reflection."""
        reasoning_text = ""
        if reasoning:
            reasoning_text = "\n".join(
                [f"步骤{t.step_number}: {t.thought}" for t in reasoning.thoughts]
            )

        prompt = REFLECTION_PROMPT.format(
            question=question,
            answer=answer,
            reasoning=reasoning_text if reasoning_text else "无详细推理过程",
        )

        try:
            result = self.llm.invoke(prompt)
            if hasattr(result, "content"):
                response_text = result.content
            else:
                response_text = str(result)

            return self._parse_reflection_response(question, response_text)

        except Exception as e:
            # Return error reflection
            return SelfReflectionResult(
                context=question,
                reflections=[
                    ReflectionItem(
                        reflection_type=ReflectionType.ERROR_ANALYSIS,
                        aspect="系统错误",
                        assessment=f"反思过程出错: {str(e)}",
                        score=0.0,
                        issues=["反思模块执行失败"],
                        suggestions=["检查LLM配置"],
                    )
                ],
                overall_score=0.0,
                should_retry=True,
                retry_reason="反思过程失败",
                improvements=["重新执行反思"],
            )

    def _rule_based_reflect(
        self,
        question: str,
        answer: str,
        reasoning: ChainOfThoughtResult | None,
        context: str | None,
    ) -> SelfReflectionResult:
        """Rule-based self-reflection without LLM.

        Performs structured quality assessment based on heuristics.
        """
        reflections: list[ReflectionItem] = []
        issues_found: list[str] = []
        improvements: list[str] = []

        # 1. Quality Check
        quality_reflection = self._check_quality(question, answer)
        reflections.append(quality_reflection)
        issues_found.extend(quality_reflection.issues)

        # 2. Completeness Check
        completeness_reflection = self._check_completeness(question, answer, context)
        reflections.append(completeness_reflection)
        issues_found.extend(completeness_reflection.issues)

        # 3. Consistency Check
        consistency_reflection = self._check_consistency(answer, reasoning)
        reflections.append(consistency_reflection)
        issues_found.extend(consistency_reflection.issues)

        # 4. Error Analysis
        error_reflection = self._analyze_errors(answer)
        reflections.append(error_reflection)
        issues_found.extend(error_reflection.issues)

        # Calculate overall score
        overall_score = sum(r.score for r in reflections) / len(reflections)

        # Determine if retry is needed
        should_retry = overall_score < self.quality_threshold
        retry_reason = None
        if should_retry:
            if issues_found:
                retry_reason = f"发现以下问题: {'; '.join(issues_found[:3])}"
            else:
                retry_reason = f"质量分数 ({overall_score:.0%}) 低于阈值 ({self.quality_threshold:.0%})"

        # Compile improvements
        for reflection in reflections:
            improvements.extend(reflection.suggestions)

        return SelfReflectionResult(
            context=question,
            reflections=reflections,
            overall_score=overall_score,
            should_retry=should_retry,
            retry_reason=retry_reason,
            improvements=list(set(improvements))[:5],  # Dedupe and limit
        )

    def _check_quality(self, question: str, answer: str) -> ReflectionItem:
        """Check the quality of the answer."""
        issues = []
        suggestions = []
        score = 1.0

        # Check answer length
        if len(answer) < 10:
            issues.append("答案过短")
            suggestions.append("提供更详细的回答")
            score -= 0.3
        elif len(answer) > 2000:
            issues.append("答案可能过长")
            suggestions.append("考虑精简回答")
            score -= 0.1

        # Check if answer is relevant to question
        question_keywords = set(self._extract_keywords(question))
        answer_keywords = set(self._extract_keywords(answer))
        relevance = len(question_keywords & answer_keywords) / max(
            len(question_keywords), 1
        )
        if relevance < 0.2:
            issues.append("答案可能与问题不相关")
            suggestions.append("确保答案直接回应问题")
            score -= 0.3

        # Check for uncertainty indicators
        uncertainty_words = ["可能", "也许", "不确定", "不清楚", "不知道"]
        uncertainty_count = sum(1 for w in uncertainty_words if w in answer)
        if uncertainty_count > 2:
            issues.append("答案中包含过多不确定表述")
            suggestions.append("尝试提供更明确的信息")
            score -= 0.2

        return ReflectionItem(
            reflection_type=ReflectionType.QUALITY_CHECK,
            aspect="答案质量",
            assessment=f"质量评分: {max(0, score):.0%}",
            score=max(0, score),
            issues=issues,
            suggestions=suggestions,
        )

    def _check_completeness(
        self,
        question: str,
        answer: str,
        context: str | None,
    ) -> ReflectionItem:
        """Check the completeness of the answer."""
        issues = []
        suggestions = []
        score = 1.0

        # Check if question contains multiple parts
        question_parts = re.split(r"[，,；;、]|和|以及|还有", question)
        question_parts = [p.strip() for p in question_parts if len(p.strip()) > 2]

        if len(question_parts) > 1:
            # Check if answer addresses multiple parts
            addressed_parts = sum(
                1
                for part in question_parts
                if any(kw in answer for kw in self._extract_keywords(part))
            )
            coverage = addressed_parts / len(question_parts)
            if coverage < 0.5:
                issues.append(f"问题包含{len(question_parts)}个部分，但只回答了部分")
                suggestions.append("确保回答问题的所有方面")
                score -= 0.3

        # Check if context was used
        if context and len(context) > 50:
            context_keywords = set(self._extract_keywords(context))
            answer_keywords = set(self._extract_keywords(answer))
            context_usage = len(context_keywords & answer_keywords) / max(
                len(context_keywords), 1
            )
            if context_usage < 0.1:
                issues.append("似乎没有充分利用提供的上下文信息")
                suggestions.append("参考上下文中的相关信息")
                score -= 0.2

        return ReflectionItem(
            reflection_type=ReflectionType.COMPLETENESS,
            aspect="完整性",
            assessment=f"完整性评分: {max(0, score):.0%}",
            score=max(0, score),
            issues=issues,
            suggestions=suggestions,
        )

    def _check_consistency(
        self,
        answer: str,
        reasoning: ChainOfThoughtResult | None,
    ) -> ReflectionItem:
        """Check the consistency of the answer."""
        issues = []
        suggestions = []
        score = 1.0

        # Check for contradictions in answer
        contradiction_pairs = [
            ("是", "不是"),
            ("可以", "不可以"),
            ("正确", "错误"),
            ("增加", "减少"),
            ("支持", "反对"),
        ]

        for pos, neg in contradiction_pairs:
            if pos in answer and neg in answer:
                # Could be legitimate comparison, but flag for review
                issues.append(f"答案中同时出现「{pos}」和「{neg}」，请确认是否矛盾")
                score -= 0.1

        # Check consistency with reasoning
        if reasoning and reasoning.final_answer:
            # Compare reasoning conclusion with answer
            reasoning_keywords = set(self._extract_keywords(reasoning.final_answer))
            answer_keywords = set(self._extract_keywords(answer))
            consistency = len(reasoning_keywords & answer_keywords) / max(
                len(reasoning_keywords), 1
            )
            if consistency < 0.3:
                issues.append("最终答案与推理结论可能不一致")
                suggestions.append("确保答案与推理过程一致")
                score -= 0.2

        if not issues:
            suggestions.append("保持良好的一致性")

        return ReflectionItem(
            reflection_type=ReflectionType.CONSISTENCY,
            aspect="一致性",
            assessment=f"一致性评分: {max(0, score):.0%}",
            score=max(0, score),
            issues=issues,
            suggestions=suggestions,
        )

    def _analyze_errors(self, answer: str) -> ReflectionItem:
        """Analyze potential errors in the answer."""
        issues = []
        suggestions = []
        score = 1.0

        # Check for common error patterns
        error_patterns = [
            (r"根据.*(?:无法|不能).*(?:找到|获取)", "可能存在信息检索问题"),
            (r"(?:错误|失败|异常)", "答案中提及错误或失败"),
            (r"(?:不存在|找不到|没有).*(?:信息|数据|文档)", "可能缺少必要信息"),
        ]

        for pattern, description in error_patterns:
            if re.search(pattern, answer):
                issues.append(description)
                score -= 0.15

        # Check for placeholder or incomplete content
        placeholder_patterns = [
            r"\[.*?\]",  # [placeholder]
            r"\{.*?\}",  # {placeholder}
            r"TODO",
            r"待补充",
            r"待完善",
        ]

        for pattern in placeholder_patterns:
            if re.search(pattern, answer):
                issues.append("答案中可能包含未完成的占位符")
                suggestions.append("完成所有占位符内容")
                score -= 0.2
                break

        if not issues:
            suggestions.append("未发现明显错误，继续保持")

        return ReflectionItem(
            reflection_type=ReflectionType.ERROR_ANALYSIS,
            aspect="错误分析",
            assessment=f"错误分析评分: {max(0, score):.0%}",
            score=max(0, score),
            issues=issues,
            suggestions=suggestions,
        )

    def _parse_reflection_response(
        self,
        question: str,
        response: str,
    ) -> SelfReflectionResult:
        """Parse reflection response from LLM."""
        # Try to extract JSON
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        json_str = json_match.group(1) if json_match else response

        try:
            # Find JSON object
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(json_str[start:end])

                reflections = []
                for r in parsed.get("reflections", []):
                    reflections.append(
                        ReflectionItem(
                            reflection_type=self._parse_reflection_type(
                                r.get("reflection_type", "quality_check")
                            ),
                            aspect=r.get("aspect", ""),
                            assessment=r.get("assessment", ""),
                            score=float(r.get("score", 0.5)),
                            issues=r.get("issues", []),
                            suggestions=r.get("suggestions", []),
                        )
                    )

                return SelfReflectionResult(
                    context=question,
                    reflections=reflections,
                    overall_score=float(parsed.get("overall_score", 0.5)),
                    should_retry=parsed.get("should_retry", False),
                    retry_reason=parsed.get("retry_reason"),
                    improvements=parsed.get("improvements", []),
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: create basic reflection from text
        return SelfReflectionResult(
            context=question,
            reflections=[
                ReflectionItem(
                    reflection_type=ReflectionType.QUALITY_CHECK,
                    aspect="总体评估",
                    assessment=response[:200] if response else "无法解析评估结果",
                    score=0.5,
                    issues=["无法解析详细评估"],
                    suggestions=["重新进行评估"],
                )
            ],
            overall_score=0.5,
            should_retry=False,
            retry_reason=None,
            improvements=[],
        )

    def _parse_reflection_type(self, type_str: str) -> ReflectionType:
        """Parse reflection type string to enum."""
        type_map = {
            "quality_check": ReflectionType.QUALITY_CHECK,
            "error_analysis": ReflectionType.ERROR_ANALYSIS,
            "improvement": ReflectionType.IMPROVEMENT,
            "completeness": ReflectionType.COMPLETENESS,
            "consistency": ReflectionType.CONSISTENCY,
        }
        return type_map.get(type_str.lower(), ReflectionType.QUALITY_CHECK)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        # Simple keyword extraction - remove common words and short words
        stop_words = {
            "的",
            "是",
            "在",
            "有",
            "和",
            "与",
            "了",
            "就",
            "都",
            "而",
            "及",
            "着",
            "或",
            "一",
            "个",
            "这",
            "那",
            "你",
            "我",
            "他",
            "它",
            "她",
            "们",
            "把",
            "被",
        }

        # Extract Chinese words (2+ chars) and English words
        words = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", text)

        return [w for w in words if w.lower() not in stop_words]

    def visualize_reflection(self, result: SelfReflectionResult) -> str:
        """Create a visual representation of the reflection."""
        lines = [
            f"反思对象: {result.context[:50]}...",
            "=" * 50,
            f"总体评分: {result.overall_score:.0%}",
            f"是否需要重试: {'是' if result.should_retry else '否'}",
        ]

        if result.retry_reason:
            lines.append(f"重试原因: {result.retry_reason}")

        lines.append("\n详细评估:")
        for i, ref in enumerate(result.reflections, 1):
            lines.extend(
                [
                    f"\n{i}. {ref.aspect} ({ref.reflection_type.value})",
                    f"   评分: {ref.score:.0%}",
                    f"   评估: {ref.assessment}",
                ]
            )
            if ref.issues:
                lines.append(f"   问题: {', '.join(ref.issues)}")
            if ref.suggestions:
                lines.append(f"   建议: {', '.join(ref.suggestions)}")

        if result.improvements:
            lines.extend(
                [
                    "\n" + "=" * 50,
                    "改进建议:",
                ]
            )
            for imp in result.improvements:
                lines.append(f"  - {imp}")

        return "\n".join(lines)
