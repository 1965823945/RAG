"""Task Decomposition Module.

Breaks down complex tasks into manageable subtasks with dependencies.
"""

import json
import re
from typing import Optional, List

from langchain_core.language_models import BaseLLM

from rag_minimal.schemas import (
    TaskDecomposition,
    SubTask,
    TaskStatus,
    TaskPriority,
)


# Prompt template for task decomposition
DECOMPOSITION_PROMPT = """你是一个任务分解专家。请将用户的复杂任务分解为可执行的子任务。

## 任务
{task}

## 要求
1. 将任务分解为3-7个清晰、可执行的子任务
2. 每个子任务应该是独立且具体的
3. 标注子任务之间的依赖关系（哪些任务需要先完成）
4. 评估任务优先级（critical/high/medium/low）
5. 评估整体复杂度（simple/medium/complex）

## 输出格式（JSON）
请严格按以下JSON格式输出，不要添加其他内容：
```json
{{
    "complexity": "medium",
    "estimated_time": "预计完成时间",
    "subtasks": [
        {{
            "id": "task_1",
            "description": "子任务描述",
            "priority": "high",
            "dependencies": []
        }},
        {{
            "id": "task_2", 
            "description": "另一个子任务",
            "priority": "medium",
            "dependencies": ["task_1"]
        }}
    ]
}}
```

请分解任务："""


class TaskDecomposer:
    """Decomposes complex tasks into subtasks.

    This module analyzes a task and breaks it down into:
    1. Manageable subtasks
    2. Dependencies between subtasks
    3. Priority ordering
    4. Complexity assessment
    """

    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize the task decomposer.

        Args:
            llm: Language model for decomposition. If None, uses rule-based decomposition.
        """
        self.llm = llm

    def decompose(self, task: str) -> TaskDecomposition:
        """Decompose a task into subtasks.

        Args:
            task: The task description to decompose

        Returns:
            TaskDecomposition with subtasks, dependencies, and metadata
        """
        if self.llm:
            return self._llm_decompose(task)
        return self._rule_based_decompose(task)

    def _llm_decompose(self, task: str) -> TaskDecomposition:
        """Use LLM to decompose the task."""
        prompt = DECOMPOSITION_PROMPT.format(task=task)

        try:
            result = self.llm.invoke(prompt)
            if hasattr(result, "content"):
                response_text = result.content
            else:
                response_text = str(result)

            # Parse JSON from response
            parsed = self._parse_json_response(response_text)

            if parsed:
                subtasks = []
                for i, st in enumerate(parsed.get("subtasks", [])):
                    subtasks.append(
                        SubTask(
                            id=st.get("id", f"task_{i + 1}"),
                            description=st.get("description", ""),
                            priority=self._parse_priority(st.get("priority", "medium")),
                            dependencies=st.get("dependencies", []),
                            status=TaskStatus.PENDING,
                        )
                    )

                return TaskDecomposition(
                    original_task=task,
                    subtasks=subtasks,
                    total_steps=len(subtasks),
                    complexity=parsed.get("complexity", "medium"),
                    estimated_time=parsed.get("estimated_time"),
                )

        except Exception as e:
            # Fallback to rule-based if LLM fails
            print(f"LLM decomposition failed: {e}, using rule-based fallback")

        return self._rule_based_decompose(task)

    def _rule_based_decompose(self, task: str) -> TaskDecomposition:
        """Rule-based task decomposition without LLM.

        Uses heuristics to break down tasks:
        1. Identify action keywords
        2. Detect question types
        3. Create logical subtasks
        """
        subtasks = []
        # task_lower reserved for future pattern matching enhancements

        # Detect task type and create appropriate subtasks
        if self._is_question(task):
            subtasks = self._decompose_question(task)
        elif self._is_comparison(task):
            subtasks = self._decompose_comparison(task)
        elif self._is_analysis(task):
            subtasks = self._decompose_analysis(task)
        elif self._is_creation(task):
            subtasks = self._decompose_creation(task)
        else:
            subtasks = self._decompose_general(task)

        # Determine complexity based on subtask count
        complexity = (
            "simple"
            if len(subtasks) <= 2
            else ("medium" if len(subtasks) <= 4 else "complex")
        )

        return TaskDecomposition(
            original_task=task,
            subtasks=subtasks,
            total_steps=len(subtasks),
            complexity=complexity,
            estimated_time=self._estimate_time(len(subtasks)),
        )

    def _is_question(self, task: str) -> bool:
        """Check if the task is a question."""
        question_indicators = [
            "什么",
            "为什么",
            "如何",
            "怎么",
            "哪",
            "是否",
            "能否",
            "?",
            "？",
        ]
        return any(ind in task for ind in question_indicators)

    def _is_comparison(self, task: str) -> bool:
        """Check if the task involves comparison."""
        comparison_words = [
            "比较",
            "对比",
            "区别",
            "差异",
            "相同",
            "不同",
            "优劣",
            "vs",
        ]
        return any(word in task.lower() for word in comparison_words)

    def _is_analysis(self, task: str) -> bool:
        """Check if the task requires analysis."""
        analysis_words = ["分析", "评估", "研究", "调查", "探讨", "深入"]
        return any(word in task for word in analysis_words)

    def _is_creation(self, task: str) -> bool:
        """Check if the task involves creation."""
        creation_words = ["创建", "生成", "编写", "设计", "制作", "开发", "实现"]
        return any(word in task for word in creation_words)

    def _decompose_question(self, task: str) -> List[SubTask]:
        """Decompose a question-type task."""
        return [
            SubTask(
                id="task_1",
                description="理解问题：分析问题的核心要点",
                priority=TaskPriority.HIGH,
                dependencies=[],
            ),
            SubTask(
                id="task_2",
                description=f"检索相关信息：搜索与「{task[:30]}...」相关的知识",
                priority=TaskPriority.HIGH,
                dependencies=["task_1"],
            ),
            SubTask(
                id="task_3",
                description="整合信息：综合检索到的信息形成答案",
                priority=TaskPriority.MEDIUM,
                dependencies=["task_2"],
            ),
            SubTask(
                id="task_4",
                description="验证答案：确保答案准确且完整",
                priority=TaskPriority.MEDIUM,
                dependencies=["task_3"],
            ),
        ]

    def _decompose_comparison(self, task: str) -> List[SubTask]:
        """Decompose a comparison-type task."""
        return [
            SubTask(
                id="task_1",
                description="识别比较对象：明确需要比较的两个或多个对象",
                priority=TaskPriority.HIGH,
                dependencies=[],
            ),
            SubTask(
                id="task_2",
                description="收集第一个对象的信息",
                priority=TaskPriority.HIGH,
                dependencies=["task_1"],
            ),
            SubTask(
                id="task_3",
                description="收集第二个对象的信息",
                priority=TaskPriority.HIGH,
                dependencies=["task_1"],
            ),
            SubTask(
                id="task_4",
                description="确定比较维度：选择合适的比较标准",
                priority=TaskPriority.MEDIUM,
                dependencies=["task_2", "task_3"],
            ),
            SubTask(
                id="task_5",
                description="进行对比分析：在各维度上比较异同",
                priority=TaskPriority.MEDIUM,
                dependencies=["task_4"],
            ),
            SubTask(
                id="task_6",
                description="总结结论：给出比较结果",
                priority=TaskPriority.MEDIUM,
                dependencies=["task_5"],
            ),
        ]

    def _decompose_analysis(self, task: str) -> List[SubTask]:
        """Decompose an analysis-type task."""
        return [
            SubTask(
                id="task_1",
                description="明确分析目标：确定分析的范围和重点",
                priority=TaskPriority.HIGH,
                dependencies=[],
            ),
            SubTask(
                id="task_2",
                description="收集相关数据和信息",
                priority=TaskPriority.HIGH,
                dependencies=["task_1"],
            ),
            SubTask(
                id="task_3",
                description="整理和分类信息",
                priority=TaskPriority.MEDIUM,
                dependencies=["task_2"],
            ),
            SubTask(
                id="task_4",
                description="深入分析：识别模式、趋势和关键发现",
                priority=TaskPriority.HIGH,
                dependencies=["task_3"],
            ),
            SubTask(
                id="task_5",
                description="得出结论并提出建议",
                priority=TaskPriority.MEDIUM,
                dependencies=["task_4"],
            ),
        ]

    def _decompose_creation(self, task: str) -> List[SubTask]:
        """Decompose a creation-type task."""
        return [
            SubTask(
                id="task_1",
                description="需求分析：明确要创建的内容和要求",
                priority=TaskPriority.HIGH,
                dependencies=[],
            ),
            SubTask(
                id="task_2",
                description="收集参考资料和灵感",
                priority=TaskPriority.MEDIUM,
                dependencies=["task_1"],
            ),
            SubTask(
                id="task_3",
                description="制定创建计划和大纲",
                priority=TaskPriority.HIGH,
                dependencies=["task_2"],
            ),
            SubTask(
                id="task_4",
                description="执行创建：生成初步内容",
                priority=TaskPriority.HIGH,
                dependencies=["task_3"],
            ),
            SubTask(
                id="task_5",
                description="审查和修改：完善创建的内容",
                priority=TaskPriority.MEDIUM,
                dependencies=["task_4"],
            ),
        ]

    def _decompose_general(self, task: str) -> List[SubTask]:
        """Decompose a general task."""
        return [
            SubTask(
                id="task_1",
                description="理解任务：分析任务要求",
                priority=TaskPriority.HIGH,
                dependencies=[],
            ),
            SubTask(
                id="task_2",
                description="收集所需信息",
                priority=TaskPriority.HIGH,
                dependencies=["task_1"],
            ),
            SubTask(
                id="task_3",
                description="执行任务：完成主要工作",
                priority=TaskPriority.HIGH,
                dependencies=["task_2"],
            ),
            SubTask(
                id="task_4",
                description="验证结果：确保任务完成质量",
                priority=TaskPriority.MEDIUM,
                dependencies=["task_3"],
            ),
        ]

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Parse JSON from LLM response."""
        # Try to find JSON block
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try direct JSON parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON-like content
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        return None

    def _parse_priority(self, priority_str: str) -> TaskPriority:
        """Parse priority string to enum."""
        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW,
        }
        return priority_map.get(priority_str.lower(), TaskPriority.MEDIUM)

    def _estimate_time(self, num_subtasks: int) -> str:
        """Estimate completion time based on subtask count."""
        if num_subtasks <= 2:
            return "1-2分钟"
        elif num_subtasks <= 4:
            return "2-5分钟"
        else:
            return "5-10分钟"

    def get_execution_order(self, decomposition: TaskDecomposition) -> List[List[str]]:
        """Get the execution order respecting dependencies.

        Returns a list of lists, where each inner list contains
        tasks that can be executed in parallel.
        """
        if not decomposition.subtasks:
            return []

        # Build dependency graph
        tasks = {t.id: t for t in decomposition.subtasks}
        remaining = set(tasks.keys())
        completed = set()
        execution_order = []

        while remaining:
            # Find tasks with all dependencies satisfied
            ready = []
            for task_id in remaining:
                task = tasks[task_id]
                if all(dep in completed for dep in task.dependencies):
                    ready.append(task_id)

            if not ready:
                # Circular dependency or missing dependency - add remaining
                ready = list(remaining)

            execution_order.append(ready)
            completed.update(ready)
            remaining -= set(ready)

        return execution_order
