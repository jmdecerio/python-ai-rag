from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    openai_api_key: str
    csv_path: str = str(BASE_DIR / "data" / "movies500Trimmed.csv")
    chroma_path: str = str(BASE_DIR / "storage" / "chroma_db")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
