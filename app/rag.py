import csv
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from .services import AIService
from .database import DatabaseService
from .models import MovieChunk


class RAGService:
    def __init__(
        self,
        csv_path: str,
        db_path: str,
        ai_service: AIService,
        top_k: int = 5,
    ) -> None:
        self.csv_path = csv_path
        self.db_path = db_path
        self.ai_service = ai_service
        self.top_k = top_k
        self.database_service = DatabaseService(db_path)

    def answer_question(self, question: str) -> str:
        self._ensure_index()
        query_embedding = self.ai_service.embed_texts([question])[0]
        matches = self._search(query_embedding)
        context = self._format_context(matches)
        return self.ai_service.generate_answer(question, context)

    def _ensure_index(self) -> None:
        self.database_service._ensure_index()




    def _read_csv_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(self.csv_path, newline="", encoding="latin-1") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                text = self._build_text(item)
                rows.append(
                    {
                        "movie_id": item.get("id", "").strip(),
                        "title": item.get("title", "").strip(),
                        "overview": item.get("overview", "").strip(),
                        "genres": item.get("genres", "").strip(),
                        "release_date": item.get("release_date", "").strip(),
                        "runtime": item.get("runtime", "").strip(),
                        "credits": item.get("credits", "").strip(),
                        "text": text,
                    }
                )
        return rows

    def _build_text(self, item: Dict[str, Any]) -> str:
        payload = {
            "title": item.get("title", ""),
            "genres": item.get("genres", ""),
            "overview": item.get("overview", ""),
            "release_date": item.get("release_date", ""),
            "runtime": item.get("runtime", ""),
            "credits": item.get("credits", ""),
        }
        return json.dumps(payload, ensure_ascii=True)



    def _search(self, query_embedding: np.ndarray) -> List[MovieChunk]:
        chunks = self.database_service._chunks
        if not chunks:
            return []
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scores: List[float] = []
        for chunk in chunks:
            chunk_norm = chunk.embedding / (np.linalg.norm(chunk.embedding) + 1e-8)
            scores.append(float(np.dot(query_norm, chunk_norm)))
        top_indices = np.argsort(scores)[-self.top_k :][::-1]
        return [chunks[idx] for idx in top_indices]

    def _format_context(self, chunks: List[MovieChunk]) -> str:
        parts = []
        for chunk in chunks:
            parts.append(
                "\n".join(
                    [
                        f"Title: {chunk.title}",
                        f"Genres: {chunk.genres}",
                        f"Release date: {chunk.release_date}",
                        f"Runtime: {chunk.runtime}",
                        f"Overview: {chunk.overview}",
                    ]
                )
            )
        return "\n\n".join(parts)

