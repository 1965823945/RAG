"""Tests for unified configuration management."""

from rag_minimal.config import (
    DocumentConfig,
    LLMConfig,
    RAGSettings,
    RetrievalConfig,
    configure,
    get_default_prompt,
    get_default_top_k,
    get_settings,
    reset_settings,
    set_settings,
)


class TestConfigModels:
    """Tests for configuration Pydantic models."""

    def test_document_config_defaults(self):
        """Test DocumentConfig has correct defaults."""
        config = DocumentConfig()
        assert config.docs_dir == "docs"
        assert config.chunk_size == 400
        assert config.chunk_overlap == 50

    def test_llm_config_defaults(self):
        """Test LLMConfig has correct defaults."""
        config = LLMConfig()
        assert config.provider == "simple"
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.7

    def test_retrieval_config_defaults(self):
        """Test RetrievalConfig has correct defaults."""
        config = RetrievalConfig()
        assert config.top_k == 3
        assert config.score_threshold == 0.0

    def test_document_config_custom(self):
        """Test DocumentConfig with custom values."""
        config = DocumentConfig(docs_dir="custom", chunk_size=500)
        assert config.docs_dir == "custom"
        assert config.chunk_size == 500

    def test_llm_config_with_api_key(self):
        """Test LLMConfig with API key."""
        config = LLMConfig(provider="openai", api_key="test-key")
        assert config.provider == "openai"
        assert config.api_key == "test-key"


class TestRAGSettings:
    """Tests for RAGSettings."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def test_settings_defaults(self):
        """Test RAGSettings has correct default sub-configs."""
        settings = RAGSettings()
        assert settings.document.docs_dir == "docs"
        assert settings.retrieval.top_k == 3
        assert settings.llm.provider == "simple"
        assert settings.debug is False

    def test_settings_custom_document(self):
        """Test RAGSettings with custom document config."""
        settings = RAGSettings(
            document=DocumentConfig(docs_dir="my_docs", chunk_size=600)
        )
        assert settings.document.docs_dir == "my_docs"
        assert settings.document.chunk_size == 600

    def test_settings_custom_llm(self):
        """Test RAGSettings with custom LLM config."""
        settings = RAGSettings(llm=LLMConfig(provider="openai", model_name="gpt-4o"))
        assert settings.llm.provider == "openai"
        assert settings.llm.model_name == "gpt-4o"

    def test_from_dict(self):
        """Test creating settings from dictionary."""
        config_dict = {
            "debug": True,
            "document": {"docs_dir": "test_docs"},
            "llm": {"provider": "anthropic"},
        }
        settings = RAGSettings.from_dict(config_dict)
        assert settings.debug is True
        assert settings.document.docs_dir == "test_docs"
        assert settings.llm.provider == "anthropic"


class TestGlobalSettings:
    """Tests for global settings management."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def test_get_settings_creates_default(self):
        """Test get_settings creates default settings."""
        settings = get_settings()
        assert settings is not None
        assert settings.document.docs_dir == "docs"

    def test_get_settings_returns_same_instance(self):
        """Test get_settings returns same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_set_settings(self):
        """Test set_settings replaces global settings."""
        custom_settings = RAGSettings(debug=True)
        set_settings(custom_settings)
        assert get_settings().debug is True

    def test_reset_settings(self):
        """Test reset_settings clears global settings."""
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()
        assert settings1 is not settings2


class TestConfigure:
    """Tests for configure convenience function."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def test_configure_docs_dir(self):
        """Test configure updates docs_dir."""
        configure(docs_dir="custom_docs")
        assert get_settings().document.docs_dir == "custom_docs"

    def test_configure_llm_provider(self):
        """Test configure updates LLM provider."""
        configure(llm_provider="openai")
        assert get_settings().llm.provider == "openai"

    def test_configure_top_k(self):
        """Test configure updates top_k."""
        configure(top_k=5)
        assert get_settings().retrieval.top_k == 5

    def test_configure_debug(self):
        """Test configure updates debug mode."""
        configure(debug=True)
        assert get_settings().debug is True

    def test_configure_multiple(self):
        """Test configure with multiple options."""
        configure(
            docs_dir="my_docs",
            llm_provider="anthropic",
            llm_model="claude-3-opus",
            top_k=10,
        )
        settings = get_settings()
        assert settings.document.docs_dir == "my_docs"
        assert settings.llm.provider == "anthropic"
        assert settings.llm.model_name == "claude-3-opus"
        assert settings.retrieval.top_k == 10


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def test_get_default_prompt(self):
        """Test get_default_prompt returns prompt."""
        prompt = get_default_prompt()
        assert "{context}" in prompt
        assert "{question}" in prompt

    def test_get_default_top_k(self):
        """Test get_default_top_k returns correct value."""
        assert get_default_top_k() == 3
        configure(top_k=7)
        assert get_default_top_k() == 7
