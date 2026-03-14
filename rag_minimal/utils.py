"""Utility components: a simple LLM and a fake embedding provider for demo."""
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM


class SimpleLLM(LLM):
    @property
    def _llm_type(self):
        return "simple_llm"

    def _call(self, prompt, stop=None):
        # Basic deterministic response for demonstration
        return f"[SimpleLLM] Prompt: {prompt}"

    async def _acall(self, prompt, stop=None):
        return self._call(prompt, stop)


class FakeEmbeddings(Embeddings):
    """A very small, fake embedding provider for demo purposes.
    This is NOT suitable for production use.
    """

    @property
    def embedding_dimension(self) -> int:
        return 128

    def embed_documents(self, texts):
        # Return deterministic fake embeddings (not real vectors)
        return [[0.01] * self.embedding_dimension for _ in texts]

    def embed_query(self, text: str):
        return [0.01] * self.embedding_dimension
