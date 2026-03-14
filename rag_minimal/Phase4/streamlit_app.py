"""Phase 4: Simple Streamlit UI to demo the RAG flow (question -> answer).
This app relies on Phase3's answer_question function for retrieval-based answers.
"""
import streamlit as st
from rag_minimal.Phase3.day3 import answer_question


def main():
    st.title("Minimal RAG Demo (Streamlit)")
    st.write("Ask a question. The app will retrieve relevant chunks and generate an answer.")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_query = st.text_input("Your question:")
    if st.button("Ask") and user_query:
        with st.spinner("Thinking..."):
            answer = answer_question(user_query)
        st.session_state["history"].append((user_query, answer))

    st.subheader("Conversation");
    for q, a in st.session_state["history"]:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")


if __name__ == "__main__":
    main()


def simulate_ui(user_input: str) -> str:
    """Simulate a UI interaction by invoking the backend logic directly.
    This provides a testable entry point for UI interaction without launching a browser.
    """
    return answer_question(user_input)
