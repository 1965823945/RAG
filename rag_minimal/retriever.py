"""Keyword-based retriever for better demo results."""

import re

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Common question patterns to ignore
IGNORE_PATTERNS = [
    r"是什么",
    r"是什么？",
    r"什么是",
    r"什么意思",
    r"是什么东西",
    r"干嘛用的",
    r"用来做什么",
    r"的原理",
    r"怎么工作",
    r"如何工作",
    r"is what",
    r"what is",
    r"how does",
    r"what does",
    r"请问",
    r"能不能",
    r"如何",
    r"怎样",
    r"怎么",
]

# Synonym mapping
SYNONYMS = {
    "llm": ["大语言模型", "语言模型", "LLM"],
    "大语言模型": ["llm", "语言模型", "LLM"],
    "langchain": ["langchain", "LangChain"],
    "rag": ["rag", "RAG", "检索增强生成", "检索增强"],
    "检索增强生成": ["rag", "RAG"],
    "向量数据库": ["向量库", "向量存储", "vector database"],
    "embedding": ["嵌入", "向量表示", "文本嵌入"],
    "文档分块": ["chunking", "分块", "文本分块"],
    "语义搜索": ["语义检索", "semantic search"],
    "微调": ["fine-tuning", "微调", "finetune"],
    "提示词": ["prompt", "提示词工程"],
}


class SimpleRetriever(BaseRetriever):
    """Keyword-based retriever for demo - more reliable than vector search with fake embeddings."""

    documents: list[Document]
    k: int = 3

    def __init__(self, documents: list[Document] = None, k: int = 3, **kwargs):
        # Note: vector_store parameter kept for backward compatibility but not used
        # SimpleRetriever uses keyword matching, not vector similarity
        kwargs.pop("vector_store", None)
        super().__init__(documents=documents or [], k=k, **kwargs)

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords from query."""
        # Lowercase
        query = query.lower()

        # Remove question patterns
        for pattern in IGNORE_PATTERNS:
            query = re.sub(pattern, "", query)

        # Extract words
        words = re.findall(r"[\w\u4e00-\u9fff]+", query)

        # Filter short words
        words = [w for w in words if len(w) >= 2]

        # Add synonyms
        expanded = set(words)
        for word in words:
            if word in SYNONYMS:
                expanded.update(SYNONYMS[word])

        return list(expanded)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> list[Document]:
        """Retrieve relevant documents using keyword matching."""
        if not self.documents:
            return []

        # Extract keywords
        keywords = self._extract_keywords(query)

        if not keywords:
            # Fallback: try original query
            keywords = [query.lower()]

        # Score each document
        scored = []
        for i, doc in enumerate(self.documents):
            doc_text = doc.page_content.lower()
            doc_title = ""
            if doc.metadata:
                doc_title = doc.metadata.get("source", "")
                if doc_title:
                    doc_title = doc_title.lower()

            score = 0

            # Check main keywords in document
            for keyword in keywords:
                # Exact substring match in full text
                if keyword in doc_text:
                    score += 30

                # Exact match in title
                if doc_title and keyword in doc_title:
                    score += 50

                # Word appears in first 300 chars
                if keyword in doc_text[:300]:
                    score += 10

                # Count occurrences
                count = doc_text.count(keyword)
                if count > 0:
                    score += count * 2

            # Boost for shorter documents (more focused)
            if len(doc.page_content) < 500:
                score *= 1.3

            # Boost if multiple keywords match
            matched_keywords = sum(1 for kw in keywords if kw in doc_text)
            if matched_keywords >= 2:
                score *= 1.5

            if score > 0:
                scored.append((score, i, doc))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        # Return top k results
        return [doc for score, i, doc in scored[: self.k]]
