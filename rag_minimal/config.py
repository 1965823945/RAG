"""Unified configuration management for RAG system.

This module provides centralized configuration with:
- Pydantic models for type-safe configuration
- Environment variable support
- Default values with overrides
- Configuration validation
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ─────────────────────────────────────────────────────────────
# RAG Configuration
# ─────────────────────────────────────────────────────────────


class DocumentConfig(BaseModel):
    """Configuration for document processing."""

    docs_dir: str = Field(default="docs", description="Directory containing documents")
    chunk_size: int = Field(
        default=400, ge=100, le=4000, description="Chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=50, ge=0, le=500, description="Overlap between chunks"
    )
    file_extensions: list[str] = Field(
        default=[".txt", ".md", ".pdf", ".docx"],
        description="Supported file extensions",
    )


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    persist_dir: str = Field(default="chroma_db", description="Vector store directory")
    collection_name: str = Field(
        default="rag_collection", description="Collection name"
    )
    embedding_dim: int = Field(default=384, description="Embedding dimension")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""

    top_k: int = Field(
        default=3, ge=1, le=20, description="Number of documents to retrieve"
    )
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum relevance score"
    )


class LLMConfig(BaseModel):
    """Configuration for LLM."""

    provider: str = Field(
        default="simple", description="LLM provider: simple, openai, anthropic, ollama"
    )
    model_name: str = Field(default="gpt-3.5-turbo", description="Model name")
    api_key: str | None = Field(default=None, description="API key")
    base_url: str | None = Field(default=None, description="Custom API base URL")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum tokens to generate"
    )


class PlanningConfig(BaseModel):
    """Configuration for planning agent."""

    max_iterations: int = Field(
        default=3, ge=1, le=10, description="Maximum improvement iterations"
    )
    max_subtasks: int = Field(default=7, ge=1, le=20, description="Maximum subtasks")
    max_cot_steps: int = Field(
        default=10, ge=1, le=50, description="Maximum chain of thought steps"
    )
    quality_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Quality threshold for retry"
    )


class MemoryConfig(BaseModel):
    """Configuration for memory system."""

    max_history_messages: int = Field(
        default=20, ge=1, le=100, description="Maximum history messages"
    )
    max_context_tokens: int = Field(
        default=4000, ge=1000, le=128000, description="Maximum context tokens"
    )
    short_term_ttl_seconds: int = Field(
        default=3600, ge=60, description="Short-term memory TTL"
    )


class AgentConfig(BaseModel):
    """Configuration for agent runtime."""

    max_workers: int = Field(
        default=4, ge=1, le=32, description="Maximum parallel workers"
    )
    timeout_seconds: float = Field(
        default=30.0, ge=1.0, description="Tool execution timeout"
    )


# ─────────────────────────────────────────────────────────────
# Main Settings (with environment variable support)
# ─────────────────────────────────────────────────────────────


class RAGSettings(BaseSettings):
    """Main RAG system settings with environment variable support.

    Environment variables are prefixed with RAG_, e.g.:
    - RAG_LLM_PROVIDER -> llm.provider
    - RAG_LLM_API_KEY -> llm.api_key
    - RAG_DOCS_DIR -> document.docs_dir
    """

    # Sub-configurations
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    # RAG prompt template
    rag_prompt: str = Field(
        default="""你是一个有用的助手。请根据以下参考文档来回答用户的问题。

参考文档：
{context}

用户问题：{question}

请根据参考文档给出回答：""",
        description="RAG prompt template",
    )

    # Debug mode
    debug: bool = Field(default=False, description="Enable debug mode")

    model_config = {
        "env_prefix": "RAG_",
        "env_nested_delimiter": "__",
        "extra": "ignore",
    }

    @classmethod
    def from_env(cls) -> "RAGSettings":
        """Create settings from environment variables."""
        return cls()

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RAGSettings":
        """Create settings from a dictionary."""
        return cls(**config)


# ─────────────────────────────────────────────────────────────
# Global Configuration Instance
# ─────────────────────────────────────────────────────────────


_settings: RAGSettings | None = None


def get_settings() -> RAGSettings:
    """Get the global settings instance.

    Creates a new instance from environment variables if not already set.
    """
    global _settings
    if _settings is None:
        _settings = RAGSettings.from_env()
    return _settings


def set_settings(settings: RAGSettings) -> None:
    """Set the global settings instance."""
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Reset the global settings instance to None."""
    global _settings
    _settings = None


# ─────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────


def configure(
    *,
    docs_dir: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    top_k: int | None = None,
    debug: bool | None = None,
    **kwargs,
) -> RAGSettings:
    """Configure RAG settings with common options.

    Args:
        docs_dir: Document directory
        llm_provider: LLM provider (simple, openai, anthropic, ollama)
        llm_model: Model name
        llm_api_key: API key
        top_k: Number of documents to retrieve
        debug: Enable debug mode
        **kwargs: Additional settings

    Returns:
        Updated RAGSettings instance
    """
    settings = get_settings()

    # Update document config
    if docs_dir is not None:
        settings.document.docs_dir = docs_dir

    # Update LLM config
    if llm_provider is not None:
        settings.llm.provider = llm_provider
    if llm_model is not None:
        settings.llm.model_name = llm_model
    if llm_api_key is not None:
        settings.llm.api_key = llm_api_key

    # Update retrieval config
    if top_k is not None:
        settings.retrieval.top_k = top_k

    # Update debug
    if debug is not None:
        settings.debug = debug

    return settings


# ─────────────────────────────────────────────────────────────
# Export defaults (for backward compatibility)
# ─────────────────────────────────────────────────────────────


def get_default_prompt() -> str:
    """Get the default RAG prompt template."""
    return get_settings().rag_prompt


def get_default_top_k() -> int:
    """Get the default top_k value."""
    return get_settings().retrieval.top_k


def get_default_chunk_size() -> int:
    """Get the default chunk size."""
    return get_settings().document.chunk_size


def get_default_docs_dir() -> str:
    """Get the default documents directory."""
    return get_settings().document.docs_dir
