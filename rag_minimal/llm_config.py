"""LLM configuration and factory."""

from typing import Optional, Dict, Any


def create_llm(
    provider: str = "simple",
    model_name: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs,
):
    """Create an LLM based on provider.

    Args:
        provider: LLM provider - "simple", "openai", "anthropic", "ollama"
        model_name: Model name
        api_key: API key (optional for some providers)
        base_url: Custom API base URL
        temperature: Sampling temperature

    Returns:
        LLM instance
    """
    if provider == "simple":
        from rag_minimal.llm import SimpleLLM

        return SimpleLLM()

    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                **kwargs,
            )
        except ImportError:
            print("请安装 langchain-openai: pip install langchain-openai")
            from rag_minimal.llm import SimpleLLM

            return SimpleLLM()

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model_name, api_key=api_key, temperature=temperature, **kwargs
            )
        except ImportError:
            print("请安装 langchain-anthropic: pip install langchain-anthropic")
            from rag_minimal.llm import SimpleLLM

            return SimpleLLM()

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(model=model_name, temperature=temperature, **kwargs)
        except ImportError:
            print("请安装 langchain-ollama: pip install langchain-ollama")
            from rag_minimal.llm import SimpleLLM

            return SimpleLLM()

    else:
        from rag_minimal.llm import SimpleLLM

        return SimpleLLM()


# Provider options for UI
LLM_PROVIDERS = {
    "simple": {
        "name": "演示模式 (SimpleLLM)",
        "description": "无需 API 密钥，适合演示",
        "models": ["simple"],
        "requires_api_key": False,
    },
    "openai": {
        "name": "OpenAI",
        "description": "GPT-4, GPT-3.5-Turbo 等",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo"],
        "requires_api_key": True,
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "description": "Claude 3.5 Sonnet 等",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ],
        "requires_api_key": True,
    },
    "ollama": {
        "name": "Ollama (本地)",
        "description": "本地运行的 LLM",
        "models": ["llama3", "mistral", "codellama", "qwen"],
        "requires_api_key": False,
    },
}


def get_provider_info(provider: str) -> Dict[str, Any]:
    """Get information about a provider."""
    return LLM_PROVIDERS.get(provider, LLM_PROVIDERS["simple"])
