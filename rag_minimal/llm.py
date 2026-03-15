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
            
            # Support both English and Chinese prompt templates
            # English: "Context:", "Question:", "Answer:"
            # Chinese: "参考文档：", "用户问题：", "请根据"
            
            # Try Chinese first
            if "参考文档：" in prompt_text and "用户问题：" in prompt_text:
                ctx_start = prompt_text.find("参考文档：") + len("参考文档：")
                ctx_end = prompt_text.find("用户问题：")
                context = prompt_text[ctx_start:ctx_end].strip()
                
                q_start = prompt_text.find("用户问题：") + len("用户问题：")
                q_end = prompt_text.find("请根据") if "请根据" in prompt_text else len(prompt_text)
                question = prompt_text[q_start:q_end].strip()
                
                context_snippet = context[:500] if len(context) > 500 else context
                response = f"根据检索到的文档内容：\n\n{context_snippet}\n\n针对您的问题「{question}」，以上文档提供了相关信息。"
            
            # Try English
            elif "Context:" in prompt_text and "Question:" in prompt_text:
                ctx_start = prompt_text.find("Context:") + len("Context:")
                ctx_end = prompt_text.find("Question:")
                context = prompt_text[ctx_start:ctx_end].strip()
                
                q_start = prompt_text.find("Question:") + len("Question:")
                q_end = prompt_text.find("Answer:") if "Answer:" in prompt_text else len(prompt_text)
                question = prompt_text[q_start:q_end].strip()
                
                context_snippet = context[:500] if len(context) > 500 else context
                response = f"Based on the retrieved documents: {context_snippet}\n\nThis information directly addresses your question about '{question}'."
            else:
                response = f"[SimpleLLM] Response to: {prompt[:50]}..."
            
            generations.append([Generation(text=response)])
        
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "simple_llm"
