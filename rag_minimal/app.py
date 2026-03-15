"""Streamlit app for RAG demo - Chinese UI."""
import streamlit as st
from rag_minimal.chain import create_rag_chain, invoke_rag_chain
from rag_minimal.vectorstore import load_vector_store
from rag_minimal.retriever import SimpleRetriever
from rag_minimal.llm import SimpleLLM


@st.cache_resource
def get_rag_chain(persist_dir: str, k: int):
    """Create and cache the RAG chain."""
    vector_store = load_vector_store(persist_dir)
    if vector_store is None:
        return None
    retriever = SimpleRetriever(vector_store=vector_store, k=k)
    llm = SimpleLLM()
    return create_rag_chain(llm=llm, retriever=retriever)


def main():
    st.title("RAG 检索增强生成系统")
    st.write("基于您的文档进行智能问答")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    with st.sidebar:
        st.header("配置")
        persist_dir = st.text_input("向量数据库目录", value="chroma_db")
        k = st.slider("检索文档数量", 1, 10, 3)
    
    question = st.text_input("请输入您的问题:", key="question_input")
    
    if st.button("提问") and question:
        with st.spinner("正在检索并生成回答..."):
            try:
                chain = get_rag_chain(persist_dir, k)
                if chain is None:
                    st.error("未找到向量数据库，请先运行 main.py")
                    return
                
                answer = invoke_rag_chain(chain, question)
                
                st.session_state.history.append((question, answer))
                
            except Exception as e:
                st.error(f"错误: {str(e)}")
    
    if st.session_state.history:
        st.divider()
        st.subheader("对话历史")
        for q, a in reversed(st.session_state.history):
            st.markdown(f"**问题:** {q}")
            st.markdown(f"**回答:** {a}")
            st.divider()


if __name__ == "__main__":
    main()
