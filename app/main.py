import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .rag import RAGService


app = FastAPI(title="Python AI RAG Chat API")
BASE_DIR = Path(__file__).resolve().parent.parent
rag_service = RAGService(
    csv_path=str(BASE_DIR / "data" / "movies500Trimmed.csv"),
    db_path=str(BASE_DIR / "storage" / "rag_store.sqlite3"),
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing OPENAI_API_KEY environment variable.",
        )
    try:
        answer = rag_service.answer_question(request.question, api_key=api_key)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(answer=answer)
