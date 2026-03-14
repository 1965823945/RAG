import streamlit as st
from agent import QAAgent

def main():
    st.title("Private Domain RAG Demo (Local)")
    st.write("Domain: Python, PyTorch, HF, RAG (single-domain, local).")

    if "agent" not in st.session_state:
        pdf_path = "docs/domain_private_rag.pdf"
        st.session_state["agent"] = QAAgent(pdf_path)

    agent = st.session_state["agent"]
    user_input = st.text_input("Ask a question:")
    if st.button("Ask") and user_input:
        with st.spinner("Thinking..."):
            answer = agent.answer(user_input)
        st.session_state.setdefault("history", []).append((user_input, answer))

    for q, a in st.session_state.get("history", []):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
