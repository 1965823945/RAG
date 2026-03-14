"""Day 1: LangChain Hello World (no external API required)."""
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class SimpleLLM(LLM):
    @property
    def _llm_type(self):
        return "simple_llm"

    def _call(self, prompt, stop=None):
        # Minimal deterministic response for demonstration
        return f"Hello from SimpleLLM! You asked: {prompt}"

    async def _acall(self, prompt, stop=None):
        return self._call(prompt, stop)


def day1_hello_world():
    llm = SimpleLLM()
    prompt = PromptTemplate(template="Say hi and describe what LangChain can do.", input_variables=[])
    chain = LLMChain(llm=llm, prompt=prompt)
    # Use the newer invoke API for compatibility with LangChain 0.1.x
    result = chain.invoke({})
    print(result)
    return result


if __name__ == "__main__":
    day1_hello_world()
