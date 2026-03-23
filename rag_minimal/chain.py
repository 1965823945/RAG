"""RAG Chain module."""


from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from rag_minimal.constants import DEFAULT_RAG_PROMPT
from rag_minimal.llm import SimpleLLM
from rag_minimal.retriever import SimpleRetriever


def create_rag_chain(
    llm: BaseLLM | None = None,
    retriever: SimpleRetriever | None = None,
    prompt_template: str = DEFAULT_RAG_PROMPT,
) -> Runnable:
    """Create a RAG (Retrieval-Augmented Generation) chain.

    Args:
        llm: Language model to use (defaults to SimpleLLM)
        retriever: Retriever to use for context retrieval
        prompt_template: Prompt template to use

    Returns:
        A Runnable representing the RAG chain
    """
    if llm is None:
        llm = SimpleLLM()

    if retriever is None:
        raise ValueError("Retriever must be provided")

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    def format_docs(docs: list) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": lambda x: x} | prompt | llm
    ).with_config({"recursion_limit": 50})

    return chain


def invoke_rag_chain(chain: Runnable, question: str) -> str:
    """Invoke the RAG chain with a question.

    Args:
        chain: The RAG chain to invoke
        question: The question to ask

    Returns:
        The generated answer
    """
    result = chain.invoke(question)
    if hasattr(result, "content"):
        return result.content
    return str(result)
