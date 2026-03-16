"""Simple LLM wrapper - returns retrieved content directly."""

import re
from typing import Any, List, Optional
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun


class SimpleLLM(BaseLLM):
    """Simple LLM - returns retrieved content directly for demo."""

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses - just return the retrieved context."""
        generations = []
        for prompt in prompts:
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)

            # Extract context from prompt
            context = ""
            for marker in ["参考文档：", "Context:"]:
                if marker in prompt_text:
                    start = prompt_text.find(marker) + len(marker)
                    end = prompt_text.find("用户问题：", start)
                    if end == -1:
                        end = prompt_text.find("Question:", start)
                    if end == -1:
                        end = len(prompt_text)
                    context = prompt_text[start:end].strip()
                    break

            # Clean up context - remove extra whitespace
            if context:
                # Remove duplicate spaces and newlines
                context = re.sub(r"\s+", " ", context)
                # Clean up bullet points
                context = context.replace("。 ", "。").replace("。", "。\n")

                # Truncate if too long
                if len(context) > 600:
                    context = context[:600] + "..."

                response = context
            else:
                response = "未找到相关内容"

            generations.append([Generation(text=response)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "simple_llm"
