"""Streamlit app for RAG demo."""
import streamlit as st
from rag_minimal.chain import create_rag_chain, invoke_rag_chain
from rag_minimal.vectorstore import load_vector_store
from rag_minimal.retriever import SimpleRetriever
from rag_minimal.llm import SimpleLLM


def main():
    st.title("Minimal RAG Demo")
    st.write("Ask questions about your documents using Retrieval-Augmented Generation.")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    with st.sidebar:
        st.header("Configuration")
        persist_dir = st.text_input("Vector Store Directory", value="chroma_db")
        k = st.slider("Number of documents to retrieve", 1, 10, 4)
    
    question = st.text_input("Your question:", key="question_input")
    
    if st.button("Ask") and question:
        with st.spinner("Retrieving and generating answer..."):
            try:
                vector_store = load_vector_store(persist_dir)
                if vector_store is None:
                    st.error("No vector store found. Please run the setup first.")
                    return
                
                retriever = SimpleRetriever(vector_store=vector_store, k=k)
                llm = SimpleLLM()
                chain = create_rag_chain(llm=llm, retriever=retriever)
                
                answer = invoke_rag_chain(chain, question)
                
                st.session_state.history.append((question, answer))
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.session_state.history:
        st.divider()
        st.subheader("Conversation History")
        for q, a in reversed(st.session_state.history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.divider()


if __name__ == "__main__":
    main()
