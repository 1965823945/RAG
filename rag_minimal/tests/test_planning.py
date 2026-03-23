"""Tests for Planning Agent module."""

from rag_minimal.planning import (
    ChainOfThought,
    PlanningAgent,
    SelfReflection,
    TaskDecomposer,
)
from rag_minimal.schemas import (
    ChainOfThoughtResult,
    ReflectionType,
    SearchOutput,
    SearchResultItem,
    SelfReflectionResult,
    TaskDecomposition,
    TaskPriority,
    TaskStatus,
)


class TestTaskDecomposer:
    """Tests for TaskDecomposer."""

    def test_decompose_question(self):
        """Test decomposing a question-type task."""
        decomposer = TaskDecomposer()
        result = decomposer.decompose("什么是机器学习？")

        assert isinstance(result, TaskDecomposition)
        assert result.original_task == "什么是机器学习？"
        assert len(result.subtasks) > 0
        assert result.total_steps == len(result.subtasks)
        assert result.complexity in ["simple", "medium", "complex"]

    def test_decompose_comparison(self):
        """Test decomposing a comparison-type task."""
        decomposer = TaskDecomposer()
        result = decomposer.decompose("比较Python和Java的区别")

        assert isinstance(result, TaskDecomposition)
        assert len(result.subtasks) >= 3  # Comparison tasks have more steps

        # Check that subtasks have proper structure
        for task in result.subtasks:
            assert task.id is not None
            assert task.description is not None
            assert task.status == TaskStatus.PENDING
            assert task.priority in [
                TaskPriority.CRITICAL,
                TaskPriority.HIGH,
                TaskPriority.MEDIUM,
                TaskPriority.LOW,
            ]

    def test_decompose_analysis(self):
        """Test decomposing an analysis-type task."""
        decomposer = TaskDecomposer()
        result = decomposer.decompose("分析当前市场趋势")

        assert isinstance(result, TaskDecomposition)
        assert len(result.subtasks) > 0

    def test_decompose_creation(self):
        """Test decomposing a creation-type task."""
        decomposer = TaskDecomposer()
        result = decomposer.decompose("创建一个项目计划")

        assert isinstance(result, TaskDecomposition)
        assert len(result.subtasks) > 0

    def test_execution_order(self):
        """Test getting execution order respecting dependencies."""
        decomposer = TaskDecomposer()
        result = decomposer.decompose("如何学习编程？")

        order = decomposer.get_execution_order(result)

        assert isinstance(order, list)
        assert len(order) > 0
        # First batch should have no dependencies
        if result.subtasks:
            first_batch_ids = order[0]
            for task_id in first_batch_ids:
                task = next(t for t in result.subtasks if t.id == task_id)
                assert len(task.dependencies) == 0 or all(
                    dep not in [t.id for t in result.subtasks]
                    for dep in task.dependencies
                )


class TestChainOfThought:
    """Tests for ChainOfThought."""

    def test_reason_without_llm(self):
        """Test rule-based reasoning without LLM."""
        cot = ChainOfThought()
        result = cot.reason(
            "什么是人工智能？", context="人工智能是模拟人类智能的技术。"
        )

        assert isinstance(result, ChainOfThoughtResult)
        assert result.question == "什么是人工智能？"
        assert len(result.thoughts) > 0
        assert result.total_steps == len(result.thoughts)
        assert 0.0 <= result.confidence <= 1.0

    def test_reason_with_context(self):
        """Test reasoning with provided context."""
        cot = ChainOfThought()
        context = "Python是一种解释型编程语言，具有简洁的语法。"
        result = cot.reason("Python是什么？", context=context)

        assert isinstance(result, ChainOfThoughtResult)
        assert result.final_answer is not None

    def test_thought_steps_structure(self):
        """Test that thought steps have proper structure."""
        cot = ChainOfThought()
        result = cot.reason("为什么天是蓝的？")

        for thought in result.thoughts:
            assert thought.step_number > 0
            assert thought.thought is not None
            assert thought.timestamp is not None

    def test_visualize_chain(self):
        """Test chain visualization."""
        cot = ChainOfThought()
        result = cot.reason("测试问题")

        visualization = cot.visualize_chain(result)

        assert isinstance(visualization, str)
        assert "问题:" in visualization
        assert "思维链:" in visualization


class TestSelfReflection:
    """Tests for SelfReflection."""

    def test_reflect_without_llm(self):
        """Test rule-based reflection without LLM."""
        reflection = SelfReflection()
        result = reflection.reflect(
            question="什么是机器学习？",
            answer="机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
        )

        assert isinstance(result, SelfReflectionResult)
        assert result.context == "什么是机器学习？"
        assert len(result.reflections) > 0
        assert 0.0 <= result.overall_score <= 1.0

    def test_reflect_short_answer(self):
        """Test reflection on a short answer."""
        reflection = SelfReflection()
        result = reflection.reflect(
            question="什么是AI？",
            answer="AI",  # Too short
        )

        assert isinstance(result, SelfReflectionResult)

        # Should identify issues with short answer
        issues_found = False
        for ref in result.reflections:
            if ref.issues:
                issues_found = True
                break
        assert issues_found

        # Quality check reflection should have lower score
        quality_reflection = next(
            (
                r
                for r in result.reflections
                if r.reflection_type == ReflectionType.QUALITY_CHECK
            ),
            None,
        )
        assert quality_reflection is not None
        assert quality_reflection.score < 0.8

    def test_reflect_with_reasoning(self):
        """Test reflection with chain of thought result."""
        cot = ChainOfThought()
        reasoning = cot.reason("测试问题", "测试上下文")

        reflection = SelfReflection()
        result = reflection.reflect(
            question="测试问题",
            answer="这是一个测试答案，包含了足够的信息来验证反思功能。",
            reasoning=reasoning,
        )

        assert isinstance(result, SelfReflectionResult)

    def test_reflection_types(self):
        """Test that all reflection types are covered."""
        reflection = SelfReflection()
        result = reflection.reflect(
            question="测试问题", answer="测试答案包含足够的信息。"
        )

        reflection_types = {ref.reflection_type for ref in result.reflections}

        # Should have multiple types of reflection
        assert len(reflection_types) >= 2

    def test_visualize_reflection(self):
        """Test reflection visualization."""
        reflection = SelfReflection()
        result = reflection.reflect(question="测试问题", answer="测试答案")

        visualization = reflection.visualize_reflection(result)

        assert isinstance(visualization, str)
        assert "反思对象:" in visualization
        assert "总体评分:" in visualization


class TestPlanningAgent:
    """Tests for PlanningAgent."""

    def test_init(self):
        """Test PlanningAgent initialization."""
        agent = PlanningAgent()

        assert agent.decomposer is not None
        assert agent.cot is not None
        assert agent.reflection is not None

    def test_run_without_llm(self):
        """Test running agent without LLM (rule-based)."""
        agent = PlanningAgent(
            quality_threshold=0.5,
            max_iterations=1,
        )

        result = agent.run(
            query="什么是人工智能？",
            use_decomposition=True,
            use_cot=True,
            use_reflection=True,
        )

        assert result is not None
        assert result.task_id is not None
        assert result.query == "什么是人工智能？"
        assert result.answer is not None

    def test_run_decomposition_only(self):
        """Test running with only decomposition."""
        agent = PlanningAgent()

        result = agent.run(
            query="比较A和B",
            use_decomposition=True,
            use_cot=False,
            use_reflection=False,
        )

        assert result is not None
        if result.state and result.state.decomposition:
            assert result.state.decomposition.total_steps > 0

    def test_decompose_only(self):
        """Test decompose_only method."""
        agent = PlanningAgent()

        result = agent.decompose_only("分析数据趋势")

        assert isinstance(result, TaskDecomposition)
        assert len(result.subtasks) > 0

    def test_reason_only(self):
        """Test reason_only method."""
        agent = PlanningAgent()

        result = agent.reason_only("什么是测试？", "测试是验证功能的过程。")

        assert isinstance(result, ChainOfThoughtResult)

    def test_reflect_only(self):
        """Test reflect_only method."""
        agent = PlanningAgent()

        result = agent.reflect_only(query="测试问题", answer="测试答案")

        assert isinstance(result, SelfReflectionResult)

    def test_get_execution_summary(self):
        """Test execution summary generation."""
        agent = PlanningAgent()

        result = agent.run("测试任务", use_reflection=False)
        summary = agent.get_execution_summary(result)

        assert isinstance(summary, str)
        assert "任务ID:" in summary
        assert "状态:" in summary

    def test_with_search_function(self):
        """Test agent with a mock search function."""

        def mock_search(query: str, top_k: int) -> SearchOutput:
            return SearchOutput(
                success=True,
                message="ok",
                query=query,
                results=[
                    SearchResultItem(
                        content="这是搜索到的相关内容。",
                        source="test.txt",
                        score=0.9,
                    )
                ],
            )

        agent = PlanningAgent(
            search_func=mock_search,
            max_iterations=1,
        )

        result = agent.run(
            query="搜索测试",
            use_decomposition=False,
            use_cot=True,
            use_reflection=False,
        )

        assert result is not None
        assert result.answer is not None

    def test_iteration_limit(self):
        """Test that iteration limit is respected."""
        agent = PlanningAgent(
            quality_threshold=0.99,  # Very high threshold
            max_iterations=2,
        )

        result = agent.run(
            query="测试迭代限制",
            use_decomposition=False,
            use_cot=False,
            use_reflection=True,
        )

        assert result.iterations <= 2


class TestIntegration:
    """Integration tests for the planning module."""

    def test_full_pipeline(self):
        """Test the full planning pipeline."""
        agent = PlanningAgent(
            quality_threshold=0.5,
            max_iterations=2,
        )

        result = agent.run(
            query="请分析并解释什么是深度学习，它有哪些应用场景？",
            use_decomposition=True,
            use_cot=True,
            use_reflection=True,
        )

        assert result.success
        assert result.answer is not None
        assert len(result.answer) > 0

        # Check state is populated
        if result.state:
            assert result.state.task_id is not None
            assert result.state.original_query is not None

    def test_error_handling(self):
        """Test error handling in the agent."""
        agent = PlanningAgent()

        # Empty query should still work
        result = agent.run(
            query="",
            use_decomposition=True,
            use_cot=True,
            use_reflection=True,
        )

        # Should handle gracefully
        assert result is not None
