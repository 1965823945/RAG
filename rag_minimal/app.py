"""Streamlit app for RAG demo - using AgentRuntime, PlanningAgent, and ConversationalAgent."""

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st  # noqa: E402

from rag_minimal.agent_runtime import AgentRuntime  # noqa: E402
from rag_minimal.llm_config import LLM_PROVIDERS, create_llm  # noqa: E402
from rag_minimal.memory import ConversationalAgent  # noqa: E402
from rag_minimal.planning import PlanningAgent  # noqa: E402
from rag_minimal.schemas import MessageRole, TaskStatus  # noqa: E402


@st.cache_resource
def get_agent_runtime(docs_dir: str, llm_config: dict) -> AgentRuntime:
    """Create and cache the AgentRuntime."""
    llm = create_llm(
        provider=llm_config.get("provider", "simple"),
        model_name=llm_config.get("model", "gpt-3.5-turbo"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        temperature=llm_config.get("temperature", 0.7),
    )
    return AgentRuntime(docs_dir=docs_dir, llm=llm)


@st.cache_resource
def get_planning_agent(docs_dir: str, llm_config: dict) -> PlanningAgent:
    """Create and cache the PlanningAgent."""
    llm = create_llm(
        provider=llm_config.get("provider", "simple"),
        model_name=llm_config.get("model", "gpt-3.5-turbo"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        temperature=llm_config.get("temperature", 0.7),
    )

    # Create base agent for search function
    base_agent = AgentRuntime(docs_dir=docs_dir, llm=llm)

    return PlanningAgent(
        llm=llm,
        search_func=base_agent.search,
        quality_threshold=0.6,
        max_iterations=3,
        verbose=False,
    )


def get_conversational_agent(docs_dir: str, llm_config: dict) -> ConversationalAgent:
    """Create the ConversationalAgent (not cached to maintain state per session)."""
    # Check if we already have an agent for this session
    agent_key = "conversational_agent"
    config_key = "conv_agent_config"

    # Create new config string
    current_config = (
        f"{docs_dir}_{llm_config.get('provider')}_{llm_config.get('model')}"
    )

    # Check if we need to create a new agent
    if (
        agent_key not in st.session_state
        or st.session_state.get(config_key) != current_config
    ):
        llm = create_llm(
            provider=llm_config.get("provider", "simple"),
            model_name=llm_config.get("model", "gpt-3.5-turbo"),
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url"),
            temperature=llm_config.get("temperature", 0.7),
        )

        # Create base agent for search function
        base_agent = AgentRuntime(docs_dir=docs_dir, llm=llm)

        st.session_state[agent_key] = ConversationalAgent(
            llm=llm,
            search_func=base_agent.search,
            max_history_messages=20,
            max_context_tokens=4000,
            auto_extract_memories=True,
        )
        st.session_state[config_key] = current_config

    return st.session_state[agent_key]


def render_task_decomposition(decomposition):
    """Render task decomposition results."""
    st.markdown(f"**复杂度:** {decomposition.complexity}")
    st.markdown(f"**预计时间:** {decomposition.estimated_time or '未知'}")
    st.markdown(f"**子任务数:** {decomposition.total_steps}")

    st.markdown("**子任务列表:**")
    for task in decomposition.subtasks:
        status_icon = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.BLOCKED: "🚫",
            TaskStatus.CANCELLED: "🚫",
        }.get(task.status, "⏳")

        priority_badge = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
        }.get(task.priority.value, "🟡")

        deps = f" (依赖: {', '.join(task.dependencies)})" if task.dependencies else ""
        st.markdown(f"  {status_icon} {priority_badge} {task.description}{deps}")


def render_chain_of_thought(reasoning):
    """Render chain of thought results."""
    st.markdown(f"**推理步骤数:** {reasoning.total_steps}")
    st.markdown(f"**置信度:** {reasoning.confidence:.0%}")

    st.markdown("**思维链:**")
    for thought in reasoning.thoughts:
        with st.container():
            st.markdown(f"**步骤 {thought.step_number}:** {thought.thought}")
            if thought.action:
                st.markdown(f"  *行动:* {thought.action}")
            if thought.observation:
                st.markdown(f"  *观察:* {thought.observation[:200]}...")


def render_self_reflection(reflection):
    """Render self-reflection results."""
    # Overall score with color
    score = reflection.overall_score
    if score >= 0.8:
        score_color = "green"
    elif score >= 0.6:
        score_color = "orange"
    else:
        score_color = "red"

    st.markdown(f"**总体评分:** :{score_color}[{score:.0%}]")
    st.markdown(f"**需要重试:** {'是' if reflection.should_retry else '否'}")

    if reflection.retry_reason:
        st.markdown(f"**重试原因:** {reflection.retry_reason}")

    st.markdown("**详细评估:**")
    for ref in reflection.reflections:
        with st.container():
            ref_score = ref.score
            ref_color = (
                "green"
                if ref_score >= 0.8
                else ("orange" if ref_score >= 0.6 else "red")
            )
            st.markdown(
                f"- **{ref.aspect}** ({ref.reflection_type.value}): :{ref_color}[{ref_score:.0%}]"
            )
            if ref.issues:
                st.markdown(f"  - 问题: {', '.join(ref.issues)}")
            if ref.suggestions:
                st.markdown(f"  - 建议: {', '.join(ref.suggestions)}")

    if reflection.improvements:
        st.markdown("**改进建议:**")
        for imp in reflection.improvements:
            st.markdown(f"  - {imp}")


def render_conversation_message(msg, show_metadata: bool = False):
    """Render a single conversation message."""
    role = msg.role
    is_user = role == MessageRole.USER

    with st.chat_message("user" if is_user else "assistant"):
        st.markdown(msg.content)

        if show_metadata and not is_user:
            # Show sources for assistant messages
            if hasattr(msg, "metadata") and msg.metadata:
                sources = msg.metadata.get("sources", [])
                if sources:
                    with st.expander("📚 参考来源", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.caption(f"**来源 {i}:** {src.get('source', '未知')}")


def render_memory_summary(memory_summary: dict):
    """Render memory system summary."""
    st.markdown("#### 记忆系统状态")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("短期记忆", memory_summary.get("short_term_count", 0))
    with col2:
        st.metric("长期记忆", memory_summary.get("long_term_count", 0))
    with col3:
        st.metric("工作记忆", memory_summary.get("working_memory_count", 0))


def main():
    st.set_page_config(
        page_title="RAG 智能问答系统",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 RAG 检索增强生成系统")
    st.write("基于您的文档进行智能问答，支持自主规划 Agent 和多轮对话")

    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.llm_config = {
            "provider": "simple",
            "model": "simple",
            "api_key": "",
            "base_url": "",
            "temperature": 0.7,
        }
        st.session_state.planning_history = []
        # Conversation state
        st.session_state.current_conversation_id = None
        st.session_state.conversation_messages = []

    with st.sidebar:
        st.header("⚙️ 配置")

        # Document directory
        st.subheader("📁 文档目录")
        docs_dir = st.text_input("文档目录", value="docs")
        k = st.slider("检索文档数量", 1, 10, 3, help="每次检索返回的文档数量")

        # LLM settings
        st.subheader("🤖 模型选择")

        provider = st.selectbox(
            "选择模型提供商",
            options=list(LLM_PROVIDERS.keys()),
            format_func=lambda x: LLM_PROVIDERS[x]["name"],
            index=list(LLM_PROVIDERS.keys()).index(
                st.session_state.llm_config.get("provider", "simple")
            ),
        )

        provider_info = LLM_PROVIDERS[provider]

        # Model selection
        if provider != "simple":
            model = st.selectbox("选择模型", options=provider_info["models"], index=0)

            # API Key
            if provider_info["requires_api_key"]:
                api_key = st.text_input(
                    "API Key", type="password", help="输入您的 API 密钥"
                )
            else:
                api_key = None

            # Base URL (for custom endpoints)
            if provider == "openai":
                base_url = st.text_input(
                    "Base URL (可选)",
                    placeholder="https://api.openai.com/v1",
                    help="使用代理时填写",
                )
            else:
                base_url = None

            temperature = st.slider(
                "Temperature", 0.0, 1.0, 0.7, help="控制输出的随机性"
            )
        else:
            model = "simple"
            api_key = None
            base_url = None
            temperature = 0.7

        # Save config
        st.session_state.llm_config = {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
        }

        # Show current model
        st.divider()
        st.caption(f"当前使用: {provider_info['name']} - {model}")

    # Create tabs for different modes
    tab1, tab2, tab3 = st.tabs(["💬 普通问答", "🧠 自主规划 Agent", "🗣️ 多轮对话"])

    # Tab 1: Normal Q&A
    with tab1:
        st.subheader("普通 RAG 问答")
        question = st.text_input("请输入您的问题:", key="normal_question")

        if st.button("提问", type="primary", key="normal_submit") and question:
            with st.spinner("正在检索并生成回答..."):
                try:
                    agent = get_agent_runtime(docs_dir, st.session_state.llm_config)
                    result = agent.ask(question, top_k=k)

                    if not result.success:
                        st.error(f"错误: {result.message}")
                    else:
                        st.session_state.history.append((question, result.answer))

                        st.success("回答生成完成!")
                        st.markdown(f"**回答:** {result.answer}")

                        if result.sources:
                            with st.expander("📚 参考来源"):
                                for i, src in enumerate(result.sources, 1):
                                    st.markdown(f"**来源 {i}:** {src.source or '未知'}")
                                    st.text(
                                        src.content[:200] + "..."
                                        if len(src.content) > 200
                                        else src.content
                                    )

                except Exception as e:
                    st.error(f"错误: {str(e)}")

        # Show history
        if st.session_state.history:
            st.divider()
            st.subheader("📝 对话历史")
            for q, a in reversed(st.session_state.history[-5:]):
                with st.container():
                    st.markdown(f"**🙋 问题:** {q}")
                    st.markdown(f"**🤖 回答:** {a}")
                    st.divider()

    # Tab 2: Planning Agent
    with tab2:
        st.subheader("🧠 自主规划 Agent")
        st.info("""
        自主规划 Agent 提供以下能力：
        - **任务分解**: 将复杂任务分解为可执行的子任务
        - **思维链推理**: 逐步推理，展示完整的思考过程
        - **自我反思**: 评估回答质量，自动改进
        """)

        # Planning options
        col1, col2, col3 = st.columns(3)
        with col1:
            use_decomposition = st.checkbox(
                "启用任务分解", value=True, key="use_decomp"
            )
        with col2:
            use_cot = st.checkbox("启用思维链", value=True, key="use_cot")
        with col3:
            use_reflection = st.checkbox("启用自我反思", value=True, key="use_reflect")

        planning_question = st.text_area(
            "请输入您的复杂问题或任务:",
            height=100,
            key="planning_question",
            placeholder="例如：请分析并比较不同的机器学习算法的优缺点，并给出选择建议。",
        )

        if (
            st.button("开始规划执行", type="primary", key="planning_submit")
            and planning_question
        ):
            with st.spinner("自主规划 Agent 正在工作..."):
                try:
                    planning_agent = get_planning_agent(
                        docs_dir, st.session_state.llm_config
                    )

                    # Create progress placeholder (reserved for future use)
                    _ = st.empty()

                    result = planning_agent.run(
                        query=planning_question,
                        top_k=k,
                        use_decomposition=use_decomposition,
                        use_cot=use_cot,
                        use_reflection=use_reflection,
                    )

                    # Save to history
                    st.session_state.planning_history.append(
                        {
                            "question": planning_question,
                            "result": result,
                        }
                    )

                    if not result.success:
                        st.error(f"执行失败: {result.message}")
                    else:
                        st.success(f"任务完成! (任务ID: {result.task_id})")

                        # Main answer
                        st.markdown("### 📝 最终答案")
                        st.markdown(result.answer)

                        # Statistics
                        st.markdown("### 📊 执行统计")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("推理步骤", result.reasoning_steps)
                        with col2:
                            st.metric("迭代次数", result.iterations)
                        with col3:
                            st.metric("置信度", f"{result.confidence:.0%}")
                        with col4:
                            st.metric("质量评分", f"{result.quality_score:.0%}")

                        # Detailed results in expanders
                        if result.state:
                            # Task Decomposition
                            if result.state.decomposition and use_decomposition:
                                with st.expander("📋 任务分解详情", expanded=False):
                                    render_task_decomposition(
                                        result.state.decomposition
                                    )

                            # Chain of Thought
                            if result.state.reasoning and use_cot:
                                with st.expander("🔗 思维链详情", expanded=False):
                                    render_chain_of_thought(result.state.reasoning)

                            # Self Reflection
                            if result.state.reflection and use_reflection:
                                with st.expander("🔍 自我反思详情", expanded=False):
                                    render_self_reflection(result.state.reflection)

                except Exception as e:
                    st.error(f"错误: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

        # Show planning history
        if st.session_state.planning_history:
            st.divider()
            st.subheader("📜 规划历史")
            for _i, item in enumerate(reversed(st.session_state.planning_history[-3:])):
                with st.expander(f"问题: {item['question'][:50]}...", expanded=False):
                    r = item["result"]
                    st.markdown(f"**答案:** {r.answer[:300]}...")
                    st.markdown(
                        f"**统计:** 推理{r.reasoning_steps}步, 迭代{r.iterations}次, 质量{r.quality_score:.0%}"
                    )

    # Tab 3: Conversational Agent with Memory
    with tab3:
        st.subheader("🗣️ 多轮对话 (带记忆)")
        st.info("""
        多轮对话 Agent 提供以下能力：
        - **对话历史**: 记住之前的对话内容，理解上下文
        - **记忆系统**: 短期记忆、长期记忆、工作记忆
        - **智能检索**: 结合对话上下文增强检索
        """)

        # Get conversational agent
        conv_agent = get_conversational_agent(docs_dir, st.session_state.llm_config)

        # Conversation management sidebar
        with st.expander("📂 对话管理", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🆕 新建对话", key="new_conv"):
                    st.session_state.current_conversation_id = None
                    st.session_state.conversation_messages = []
                    st.rerun()

            with col2:
                if st.button("🗑️ 清空当前对话", key="clear_conv"):
                    if st.session_state.current_conversation_id:
                        conv_agent.clear_conversation(
                            st.session_state.current_conversation_id
                        )
                    st.session_state.conversation_messages = []
                    st.rerun()

            # List recent conversations
            st.markdown("**最近对话:**")
            conversations = conv_agent.list_conversations(limit=5)
            for conv in conversations:
                title = conv.title or f"对话 {conv.id[:8]}..."
                if st.button(
                    f"📝 {title[:30]}",
                    key=f"conv_{conv.id}",
                    help=f"消息数: {conv.message_count}",
                ):
                    st.session_state.current_conversation_id = conv.id
                    # Load conversation messages
                    loaded_conv = conv_agent.get_conversation(conv.id)
                    if loaded_conv:
                        st.session_state.conversation_messages = [
                            {"role": msg.role.value, "content": msg.content}
                            for msg in loaded_conv.messages
                        ]
                    st.rerun()

        # Memory status
        with st.expander("🧠 记忆状态", expanded=False):
            memory_summary = conv_agent.get_memory_summary()
            render_memory_summary(memory_summary)

            # Manual memory input
            st.markdown("**手动添加记忆:**")
            new_memory = st.text_input(
                "记忆内容",
                key="new_memory_input",
                placeholder="例如: 用户喜欢简洁的回答",
            )
            if st.button("添加记忆", key="add_memory") and new_memory:
                conv_agent.add_memory(new_memory, importance=0.8)
                st.success("记忆已添加!")

        # Conversation options
        col1, col2 = st.columns(2)
        with col1:
            include_history = st.checkbox(
                "包含对话历史", value=True, key="conv_history"
            )
        with col2:
            include_memories = st.checkbox(
                "包含相关记忆", value=True, key="conv_memories"
            )

        # Display conversation messages
        st.markdown("---")

        # Chat container
        chat_container = st.container()

        with chat_container:
            # Display existing messages
            for msg in st.session_state.conversation_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("请输入您的消息...", key="conv_input"):
            # Add user message to display
            st.session_state.conversation_messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Generate response
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        try:
                            result = conv_agent.chat(
                                message=prompt,
                                conversation_id=st.session_state.current_conversation_id,
                                top_k=k,
                                include_history=include_history,
                                include_memories=include_memories,
                            )

                            # Update conversation ID
                            st.session_state.current_conversation_id = (
                                result.conversation_id
                            )

                            if result.success:
                                st.markdown(result.response)

                                # Add assistant message to session
                                st.session_state.conversation_messages.append(
                                    {
                                        "role": "assistant",
                                        "content": result.response,
                                    }
                                )

                                # Show metadata in expander
                                with st.expander("📊 响应详情", expanded=False):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("使用消息数", result.messages_used)
                                    with col2:
                                        st.metric("使用记忆数", result.memories_used)
                                    with col3:
                                        st.metric("检索文档数", result.documents_used)

                                    if result.new_memories:
                                        st.markdown("**新提取的记忆:**")
                                        for mem in result.new_memories:
                                            st.caption(f"- {mem}")

                                    if result.sources:
                                        st.markdown("**参考来源:**")
                                        for i, src in enumerate(result.sources, 1):
                                            st.caption(
                                                f"{i}. {src.source or '未知'}: "
                                                f"{src.content[:100]}..."
                                            )
                            else:
                                st.error(f"错误: {result.message}")

                        except Exception as e:
                            st.error(f"错误: {str(e)}")
                            import traceback

                            st.code(traceback.format_exc())

        # Export conversation
        if st.session_state.current_conversation_id:
            with st.expander("📤 导出对话", expanded=False):
                export_format = st.selectbox(
                    "导出格式",
                    options=["text", "markdown", "json"],
                    key="export_format",
                )
                if st.button("导出", key="export_conv"):
                    exported = conv_agent.export_conversation(
                        st.session_state.current_conversation_id,
                        format=export_format,
                    )
                    if exported:
                        st.download_button(
                            label="下载文件",
                            data=exported,
                            file_name=f"conversation.{export_format if export_format != 'text' else 'txt'}",
                            mime="text/plain",
                        )


if __name__ == "__main__":
    main()
