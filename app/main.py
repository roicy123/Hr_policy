from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import ask

app = FastAPI(title="HR Policy Bot")
sessions: dict[str, list[tuple[str, str]]] = {}   # session_id -> [(role, text), ...]

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):          # plain `def`: FastAPI runs it in a threadpool,
    history = sessions.setdefault(req.session_id, [])   # so blocking LLM calls don't freeze the server
    result = ask(req.question, history)
    history += [("User", req.question), ("Assistant", result["answer"])]
    return result