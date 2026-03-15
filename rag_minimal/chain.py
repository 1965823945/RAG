"""RAG Chain module."""
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM
from rag_minimal.llm import SimpleLLM
from rag_minimal.retriever import SimpleRetriever


DEFAULT_TEMPLATE = """你是一个有用的助手。请根据以下参考文档来回答用户的问题。

参考文档：
{context}

用户问题：{question}

请根据参考文档给出回答："""


def create_rag_chain(
    llm: BaseLLM = None,
    retriever: SimpleRetriever = None,
    prompt_template: str = DEFAULT_TEMPLATE,
) -> RunnableSequence:
    """Create a RAG (Retrieval-Augmented Generation) chain.
    
    Args:
        llm: Language model to use (defaults to SimpleLLM)
        retriever: Retriever to use for context retrieval
        prompt_template: Prompt template to use
        
    Returns:
        A RunnableSequence representing the RAG chain
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
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
    ).with_config({"recursion_limit": 50})
    
    return chain


def invoke_rag_chain(chain: RunnableSequence, question: str) -> str:
    """Invoke the RAG chain with a question.
    
    Args:
        chain: The RAG chain to invoke
        question: The question to ask
        
    Returns:
        The generated answer
    """
    result = chain.invoke(question)
    if hasattr(result, 'content'):
        return result.content
    return str(result)
