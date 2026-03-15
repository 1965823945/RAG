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
            # Extract context from prompt to generate relevant response
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)
            
            # Try to extract and use context
            if "Context:" in prompt_text and "Question:" in prompt_text:
                # Extract context portion
                ctx_start = prompt_text.find("Context:") + len("Context:")
                ctx_end = prompt_text.find("Question:")
                context = prompt_text[ctx_start:ctx_end].strip()
                
                # Extract question
                q_start = prompt_text.find("Question:") + len("Question:")
                q_end = prompt_text.find("Answer:") if "Answer:" in prompt_text else len(prompt_text)
                question = prompt_text[q_start:q_end].strip()
                
                # Generate a response based on the context
                context_snippet = context[:300] if len(context) > 300 else context
                response = f"Based on the retrieved documents: {context_snippet}\n\nThis information directly addresses your question about '{question}'. The key points from the context explain this concept in detail."
            else:
                response = f"[SimpleLLM] Response to: {prompt[:50]}..."
            
            generations.append([Generation(text=response)])
        
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "simple_llm"
