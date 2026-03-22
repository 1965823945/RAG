"""Streamlit app for RAG demo - using AgentRuntime."""

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st  # noqa: E402
from rag_minimal.agent_runtime import AgentRuntime  # noqa: E402
from rag_minimal.llm_config import create_llm, LLM_PROVIDERS  # noqa: E402


@st.cache_resource
def get_agent_runtime(docs_dir: str, llm_config: dict) -> AgentRuntime:
    """Create and cache the AgentRuntime."""
    # Create LLM based on config
    llm = create_llm(
        provider=llm_config.get("provider", "simple"),
        model_name=llm_config.get("model", "gpt-3.5-turbo"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        temperature=llm_config.get("temperature", 0.7),
    )

    return AgentRuntime(docs_dir=docs_dir, llm=llm)


def main():
    st.title("RAG 检索增强生成系统")
    st.write("基于您的文档进行智能问答")

    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.llm_config = {
            "provider": "simple",
            "model": "simple",
            "api_key": "",
            "base_url": "",
            "temperature": 0.7,
        }

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

    # Main content
    question = st.text_input("请输入您的问题:", key="question_input")

    if st.button("提问", type="primary") and question:
        with st.spinner("正在检索并生成回答..."):
            try:
                # Get agent runtime
                agent = get_agent_runtime(docs_dir, st.session_state.llm_config)

                # Run RAG pipeline
                result = agent.ask(question, top_k=k)

                if not result.success:
                    st.error(f"错误: {result.message}")
                    return

                st.session_state.history.append((question, result.answer))

                # Show sources (optional)
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
        for q, a in reversed(st.session_state.history):
            with st.container():
                st.markdown(f"**🙋 问题:** {q}")
                st.markdown(f"**🤖 回答:** {a}")
                st.divider()


if __name__ == "__main__":
    main()
