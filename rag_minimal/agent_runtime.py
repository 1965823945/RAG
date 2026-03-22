"""Minimal agent runtime using standardized tools."""

from typing import Any, Dict, Optional

from langchain_core.language_models import BaseLLM

from rag_minimal.constants import DEFAULT_RAG_PROMPT
from rag_minimal.schemas import (
    SearchOutput,
    RAGOutput,
)
from rag_minimal.tools.knowledge_search import KnowledgeSearchTool
from rag_minimal.tools.registry import ToolRegistry
from rag_minimal.llm import SimpleLLM


class AgentRuntime:
    """A minimal standardized agent runtime.

    This runtime provides:
    1. Tool registration and invocation
    2. RAG pipeline: search -> format context -> LLM generate
    """

    def __init__(
        self,
        docs_dir: str = "docs",
        llm: Optional[BaseLLM] = None,
        prompt_template: str = DEFAULT_RAG_PROMPT,
    ):
        self.registry = ToolRegistry()
        self.registry.register(KnowledgeSearchTool(docs_dir=docs_dir))
        self.llm = llm or SimpleLLM()
        self.prompt_template = prompt_template

    def set_llm(self, llm: BaseLLM) -> None:
        """Set or replace the LLM."""
        self.llm = llm

    def search(self, query: str, top_k: int = 3) -> SearchOutput:
        """Run knowledge search tool."""
        tool = self.registry.get("knowledge_search")
        result = tool.invoke({"query": query, "top_k": top_k})
        if isinstance(result, SearchOutput):
            return result
        return SearchOutput(success=False, message="unexpected result", query=query)

    def ask(self, question: str, top_k: int = 3) -> RAGOutput:
        """Full RAG pipeline: search + generate answer.

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            RAGOutput with answer and sources
        """
        # Step 1: Search
        search_result = self.search(question, top_k=top_k)

        if not search_result.success:
            return RAGOutput(
                success=False,
                message=search_result.message,
                question=question,
                answer="检索失败，无法回答问题。",
                sources=[],
            )

        # Step 2: Format context
        context_parts = []
        for item in search_result.results:
            context_parts.append(item.content)
        context = "\n\n".join(context_parts)

        # Step 3: Generate answer with LLM
        prompt = self.prompt_template.format(context=context, question=question)

        try:
            llm_result = self.llm.invoke(prompt)
            if hasattr(llm_result, "content"):
                answer = llm_result.content
            else:
                answer = str(llm_result)
        except Exception as e:
            return RAGOutput(
                success=False,
                message=f"LLM error: {str(e)}",
                question=question,
                answer="生成回答时出错。",
                sources=search_result.results,
            )

        return RAGOutput(
            success=True,
            message="ok",
            question=question,
            answer=answer,
            sources=search_result.results,
        )

    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Run the agent on a query (backward compatible).

        Returns dict format for compatibility.
        """
        result = self.ask(query, top_k=top_k)
        return result.model_dump()
