from fastapi import FastAPI
from pydantic import BaseModel
from agent import QAAgent

app = FastAPI(title="Private Demo RAG API")

class Query(BaseModel):
    question: str

agent = QAAgent("docs/domain_private_rag.pdf")

@app.post("/query")
def query(req: Query):
    ans = agent.answer(req.question)
    return {"question": req.question, "answer": ans}
