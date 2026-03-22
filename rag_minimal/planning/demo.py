"""Demo script for Planning Agent.

This script demonstrates the three key capabilities:
1. Task Decomposition
2. Chain of Thought Reasoning
3. Self-Reflection

Run: python -m rag_minimal.planning.demo
"""

from rag_minimal.planning import (
    TaskDecomposer,
    ChainOfThought,
    SelfReflection,
    PlanningAgent,
)
from rag_minimal.schemas import SearchOutput, SearchResultItem


def demo_task_decomposition():
    """Demonstrate task decomposition."""
    print("\n" + "=" * 60)
    print("1. 任务分解 (Task Decomposition) 演示")
    print("=" * 60)

    decomposer = TaskDecomposer()

    tasks = [
        "什么是机器学习？",
        "比较Python和Java的区别",
        "分析当前AI发展趋势并给出投资建议",
    ]

    for task in tasks:
        print(f"\n原始任务: {task}")
        print("-" * 40)

        result = decomposer.decompose(task)

        print(f"复杂度: {result.complexity}")
        print(f"预计时间: {result.estimated_time}")
        print(f"子任务数: {result.total_steps}")
        print("子任务列表:")
        for i, subtask in enumerate(result.subtasks, 1):
            deps = (
                f" (依赖: {', '.join(subtask.dependencies)})"
                if subtask.dependencies
                else ""
            )
            print(f"  {i}. [{subtask.priority.value}] {subtask.description}{deps}")

        # Show execution order
        order = decomposer.get_execution_order(result)
        print("执行顺序 (可并行的任务在同一组):")
        for i, batch in enumerate(order, 1):
            print(f"  批次 {i}: {', '.join(batch)}")


def demo_chain_of_thought():
    """Demonstrate chain of thought reasoning."""
    print("\n" + "=" * 60)
    print("2. 思维链 (Chain of Thought) 演示")
    print("=" * 60)

    cot = ChainOfThought()

    question = "为什么天空是蓝色的？"
    context = "光线穿过大气层时会发生散射。蓝光波长较短，更容易被散射。"

    print(f"\n问题: {question}")
    print(f"上下文: {context}")
    print("-" * 40)

    result = cot.reason(question, context)

    print("\n思维链过程:")
    for thought in result.thoughts:
        print(f"  步骤 {thought.step_number}: {thought.thought}")

    print(f"\n最终答案: {result.final_answer}")
    print(f"置信度: {result.confidence:.0%}")

    # Visualize
    print("\n完整可视化:")
    print(cot.visualize_chain(result))


def demo_self_reflection():
    """Demonstrate self-reflection."""
    print("\n" + "=" * 60)
    print("3. 自我反思 (Self-Reflection) 演示")
    print("=" * 60)

    reflection = SelfReflection()

    # Test with a good answer
    print("\n测试 1: 良好的回答")
    print("-" * 40)

    result1 = reflection.reflect(
        question="什么是人工智能？",
        answer="人工智能(AI)是计算机科学的一个分支，它致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、问题解决、感知和语言理解等能力。AI的主要方法包括机器学习、深度学习和自然语言处理等。",
    )

    print(f"总体评分: {result1.overall_score:.0%}")
    print(f"需要重试: {'是' if result1.should_retry else '否'}")
    for ref in result1.reflections:
        print(f"  - {ref.aspect}: {ref.score:.0%}")

    # Test with a poor answer
    print("\n测试 2: 简短的回答")
    print("-" * 40)

    result2 = reflection.reflect(question="什么是量子计算？", answer="量子计算很复杂。")

    print(f"总体评分: {result2.overall_score:.0%}")
    print(f"需要重试: {'是' if result2.should_retry else '否'}")
    if result2.retry_reason:
        print(f"重试原因: {result2.retry_reason}")
    for ref in result2.reflections:
        print(f"  - {ref.aspect}: {ref.score:.0%}")
        if ref.issues:
            print(f"    问题: {', '.join(ref.issues)}")


def demo_planning_agent():
    """Demonstrate the full planning agent."""
    print("\n" + "=" * 60)
    print("4. 完整自主规划 Agent 演示")
    print("=" * 60)

    # Create a mock search function
    def mock_search(query: str, top_k: int) -> SearchOutput:
        return SearchOutput(
            success=True,
            message="ok",
            query=query,
            results=[
                SearchResultItem(
                    content="机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式。",
                    source="ai_intro.txt",
                    score=0.95,
                ),
                SearchResultItem(
                    content="深度学习使用多层神经网络来处理复杂的模式识别任务。",
                    source="dl_basics.txt",
                    score=0.88,
                ),
            ],
        )

    agent = PlanningAgent(
        search_func=mock_search,
        quality_threshold=0.5,
        max_iterations=2,
        verbose=True,  # Enable logging
    )

    query = "请解释机器学习和深度学习的关系，以及它们的主要应用场景。"

    print(f"\n用户问题: {query}")
    print("-" * 40)

    result = agent.run(
        query=query,
        use_decomposition=True,
        use_cot=True,
        use_reflection=True,
    )

    print("\n" + "=" * 40)
    print("执行结果:")
    print("=" * 40)
    print(agent.get_execution_summary(result))


def main():
    """Run all demos."""
    print("=" * 60)
    print("自主规划 Agent 功能演示")
    print("=" * 60)
    print("""
本演示展示了自主规划Agent的三个核心功能:
1. 任务分解 (Task Decomposition) - 将复杂任务分解为子任务
2. 思维链 (Chain of Thought) - 逐步推理展示思考过程
3. 自我反思 (Self-Reflection) - 评估回答质量并改进
    """)

    demo_task_decomposition()
    demo_chain_of_thought()
    demo_self_reflection()
    demo_planning_agent()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
