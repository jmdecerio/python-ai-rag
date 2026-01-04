from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import settings
from .rag import RAGService
from .services import AIService


app = FastAPI(title="Python AI RAG Chat API")

ai_service = AIService(
    api_key=settings.openai_api_key,
    embedding_model=settings.embedding_model,
    chat_model=settings.chat_model,
)

rag_service = RAGService(
    csv_path=settings.csv_path,
    db_path=settings.db_path,
    ai_service=ai_service,
    top_k=settings.top_k,
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = rag_service.answer_question(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(answer=answer)
