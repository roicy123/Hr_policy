import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from app.ingest import get_embeddings
import re

load_dotenv()
NO_ANSWER = "I don't have enough information in the policy documents to answer that."
TOP_K = 4
MIN_COSINE = 0.38        # tune this using your eval set

db = FAISS.load_local("vectorstore", get_embeddings(),
                      allow_dangerous_deserialization=True)  # safe: it's your own file
llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0)
# Ollama alternative:
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.2", temperature=0)

def rewrite(question, history):
    if not history:
        return question
    convo = "\n".join(f"{role}: {text}" for role, text in history[-6:])
    msgs = [
        SystemMessage(content="Rewrite the user's latest question as a standalone question using "
                      "the chat history. If it is already standalone, return it unchanged. "
                      "Return only the question."),
        HumanMessage(content=f"History:\n{convo}\n\nLatest question: {question}"),
    ]
    return llm.invoke(msgs).content.strip()

def retrieve(query):
    results = db.similarity_search_with_score(query, k=TOP_K)
    # FAISS returns squared L2 distance; on unit vectors d² = 2 - 2cos, so cos = 1 - d²/2
    scored = [(doc, 1 - dist / 2) for doc, dist in results]
    return [(d, s) for d, s in scored if s >= MIN_COSINE]

def label(doc):
    name = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page")
    return f"{name} p.{page + 1}" if page is not None else name

SYSTEM = (
    "You are an HR policy assistant. Answer ONLY using the provided context. "
    f"If the context does not contain the answer, reply exactly: {NO_ANSWER} "
    "Do not include citations, file names, or page numbers in your answer. Be concise."
)

def ask(question, history):
    standalone = rewrite(question, history)
    hits = retrieve(standalone)
    if not hits:                                   # hard gate: skip the LLM entirely
        return {"answer": NO_ANSWER, "sources": [], "rewritten": standalone, "grounded": False}

    context = "\n\n".join(f"[{label(d)}]\n{d.page_content}" for d, _ in hits)
    msgs = [SystemMessage(content=SYSTEM),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {standalone}")]
    answer = llm.invoke(msgs).content.strip()
    if NO_ANSWER.lower() in answer.lower():     # model declined, even if it added citations
        return {"answer": NO_ANSWER, "sources": [], "rewritten": standalone, "grounded": False}

    # strip the model's own citations, e.g. [leave_policy.txt] or 【leave_policy.txt】
    answer = re.sub(r"\s*[\[【][^\]】]*\.txt[\]】]", "", answer).strip()

    return {
        "answer": answer,
        "sources": [{"file": label(d), "score": round(float(s), 3)} for d, s in hits],
        "rewritten": standalone,
        "grounded": True,
    }