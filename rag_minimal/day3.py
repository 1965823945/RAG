"""Day 3: Retrieve from ChromaDB and generate answer to a question.
Uses a simple local LLM (SimpleLLM) for demonstration to avoid API keys.
"""
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from rag_minimal.utils import SimpleLLM, FakeEmbeddings
from rag_minimal.day2 import CHROMA_DIR  # type: ignore


def answer_question(question: str, persist_dir: str = CHROMA_DIR) -> str:
    # Load vector store
    vectordb = Chroma(persist_directory=persist_dir, embedding=FakeEmbeddings(size=1536))
    # If embedding model is missing, fall back to a trivial retriever
    try:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])
    except Exception:
        context = "No relevant docs found."

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question. Context: {context} Question: {question} Answer:",
    )
    llm = SimpleLLM()
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"context": context, "question": question})


def _cli_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="What is RAG?", help="Question to answer")
    args = parser.parse_args()
    print(answer_question(args.question))


if __name__ == "__main__":
    _cli_main()
