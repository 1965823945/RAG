"""Simple LLM wrapper for demo purposes."""

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import Generation, LLMResult


class SimpleLLM(BaseLLM):
    """A simple deterministic LLM for demonstration purposes.

    This is NOT suitable for production.
    """

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for the given prompts."""
        generations = []
        for prompt in prompts:
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)

            # Extract question - look for actual question content
            question = ""
            for marker in ["用户问题：", "Question:", "问题:"]:
                if marker in prompt_text:
                    start = prompt_text.find(marker) + len(marker)
                    end = prompt_text.find("请根据", start)
                    if end == -1:
                        end = prompt_text.find("Answer:", start)
                    if end == -1:
                        end = len(prompt_text)
                    question = prompt_text[start:end].strip()
                    break

            # If no question found, try to extract from the text
            if not question or len(question) < 2:
                # Look for the actual question in the prompt
                lines = prompt_text.split("\n")
                for line in lines:
                    if "?" in line or "？" in line:
                        question = line.strip()
                        break

            # Extract context
            context = ""
            for marker in ["参考文档：", "Context:"]:
                if marker in prompt_text:
                    start = prompt_text.find(marker) + len(marker)
                    end = prompt_text.find("用户问题：", start)
                    if end == -1:
                        end = prompt_text.find("Question:", start)
                    if end == -1:
                        end = prompt_text.find("Answer:", start)
                    if end == -1:
                        end = len(prompt_text)
                    context = prompt_text[start:end].strip()
                    break

            # Clean context
            context_lines = [
                line.strip() for line in context.split("\n") if line.strip()
            ]
            clean_context = " ".join(context_lines)

            # Truncate
            if len(clean_context) > 500:
                clean_context = clean_context[:500] + "..."

            # Generate response
            if clean_context:
                response = f"{clean_context}"
            else:
                response = "抱歉，未能找到相关信息。"

            generations.append([Generation(text=response)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "simple_llm"
