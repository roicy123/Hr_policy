from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import ask

app = FastAPI(title="HR Policy Bot")
sessions: dict[str, list[tuple[str, str]]] = {}

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    history = sessions.setdefault(req.session_id, [])
    try:
        result = ask(req.question, history)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM service error: {type(e).__name__}")
    history += [("User", req.question), ("Assistant", result["answer"])]
    return result