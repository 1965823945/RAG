"""Streamlit app for RAG demo - Chinese UI with model selection."""

import streamlit as st
from rag_minimal.chain import create_rag_chain, invoke_rag_chain
from rag_minimal.vectorstore import load_vector_store
from rag_minimal.retriever import SimpleRetriever
from rag_minimal.llm_config import create_llm, LLM_PROVIDERS


@st.cache_resource
def get_rag_chain(persist_dir: str, k: int, llm_config: dict):
    """Create and cache the RAG chain."""
    vector_store = load_vector_store(persist_dir)
    if vector_store is None:
        return None

    # Get documents from vector store
    try:
        docs_data = vector_store.get()
        documents = []
        for i, doc_text in enumerate(docs_data.get("documents", [])):
            from langchain_core.documents import Document

            documents.append(
                Document(
                    page_content=doc_text,
                    metadata=docs_data.get("metadatas", [{}])[i]
                    if i < len(docs_data.get("metadatas", []))
                    else {},
                )
            )

        retriever = SimpleRetriever(documents=documents, k=k)
    except Exception:
        retriever = SimpleRetriever(vector_store=vector_store, k=k)

    # Create LLM based on config
    llm = create_llm(
        provider=llm_config.get("provider", "simple"),
        model_name=llm_config.get("model", "gpt-3.5-turbo"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        temperature=llm_config.get("temperature", 0.7),
    )

    return create_rag_chain(llm=llm, retriever=retriever)


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

        # Vector store settings
        st.subheader("向量数据库")
        persist_dir = st.text_input("向量数据库目录", value="chroma_db")
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
                chain = get_rag_chain(persist_dir, k, st.session_state.llm_config)
                if chain is None:
                    st.error(
                        "未找到向量数据库，请先运行 main.py 或选择「重新加载文档库」"
                    )
                    return

                answer = invoke_rag_chain(chain, question)

                st.session_state.history.append((question, answer))

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
