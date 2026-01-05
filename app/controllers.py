from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .rag import RAGService
from .services import AIService
from .config import settings


ai_service = AIService(
    api_key=settings.openai_api_key,
    embedding_model=settings.embedding_model,
    chat_model=settings.chat_model,
)

rag_service = RAGService(
    csv_path=settings.csv_path,
    chroma_path=settings.chroma_path,
    ai_service=ai_service,
    top_k=settings.top_k,
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


chat_router = APIRouter()


@chat_router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = rag_service.answer_question(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(answer=answer)
