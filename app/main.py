from fastapi import FastAPI

from .config import settings
from .controllers import chat_router


app = FastAPI(title="Python AI RAG Chat API")

app.include_router(chat_router)
