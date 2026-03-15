"""Simple LLM wrapper for demo purposes."""
from typing import Any, List, Optional
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun


class SimpleLLM(BaseLLM):
    """A simple deterministic LLM for demonstration purposes.
    
    This is NOT suitable for production use.
    """

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for the given prompts."""
        generations = []
        for prompt in prompts:
            generations.append([Generation(text=f"[SimpleLLM] Response to: {prompt[:50]}...")])
        return LLMResult(generations=generations)

    def _llm_type(self) -> str:
        return "simple_llm"
