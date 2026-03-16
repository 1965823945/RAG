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
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)

            # Chinese prompt template
            if "参考文档：" in prompt_text and "用户问题：" in prompt_text:
                ctx_start = prompt_text.find("参考文档：") + len("参考文档：")
                ctx_end = prompt_text.find("用户问题：")
                context = prompt_text[ctx_start:ctx_end].strip()

                q_start = prompt_text.find("用户问题：") + len("用户问题：")
                q_end = (
                    prompt_text.find("请根据")
                    if "请根据" in prompt_text
                    else len(prompt_text)
                )
                question = prompt_text[q_start:q_end].strip()

                # Clean up the context - remove extra whitespace and truncate properly
                context_lines = [
                    line.strip() for line in context.split("\n") if line.strip()
                ]
                clean_context = " ".join(context_lines)

                # Truncate to reasonable length
                if len(clean_context) > 600:
                    clean_context = clean_context[:600] + "..."

                # Generate a natural Chinese response
                response = f"{clean_context}\n\n根据以上文档内容，关于「{question}」的回答如下：\n\n文档中提到了相关内容，您可以参考上述内容了解更多详情。"

            # English prompt template
            elif "Context:" in prompt_text and "Question:" in prompt_text:
                ctx_start = prompt_text.find("Context:") + len("Context:")
                ctx_end = prompt_text.find("Question:")
                context = prompt_text[ctx_start:ctx_end].strip()

                q_start = prompt_text.find("Question:") + len("Question:")
                q_end = (
                    prompt_text.find("Answer:")
                    if "Answer:" in prompt_text
                    else len(prompt_text)
                )
                question = prompt_text[q_start:q_end].strip()

                context_lines = [
                    line.strip() for line in context.split("\n") if line.strip()
                ]
                clean_context = " ".join(context_lines)

                if len(clean_context) > 600:
                    clean_context = clean_context[:600] + "..."

                response = f"{clean_context}\n\nBased on the retrieved documents, regarding your question '{question}':\n\nThe documents contain relevant information that addresses your query."
            else:
                response = f"[SimpleLLM] Response to: {prompt[:50]}..."

            generations.append([Generation(text=response)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "simple_llm"
