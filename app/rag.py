import csv
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from .services import AIService


@dataclass
class MovieChunk:
    movie_id: str
    title: str
    overview: str
    genres: str
    release_date: str
    runtime: str
    credits: str
    text: str
    embedding: np.ndarray


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
        self._lock = threading.Lock()
        self._chunks: List[MovieChunk] = []
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def answer_question(self, question: str) -> str:
        self._ensure_index()
        query_embedding = self.ai_service.embed_texts([question])[0]
        matches = self._search(query_embedding)
        context = self._format_context(matches)
        return self.ai_service.generate_answer(question, context)

    def _ensure_index(self) -> None:
        with self._lock:
            if self._chunks:
                return
            self._init_db()
            if not self._db_has_embeddings():
                self._build_index()
            self._chunks = self._load_chunks_from_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS movies (
                    movie_id TEXT PRIMARY KEY,
                    title TEXT,
                    overview TEXT,
                    genres TEXT,
                    release_date TEXT,
                    runtime TEXT,
                    credits TEXT,
                    text TEXT,
                    embedding BLOB
                )
                """
            )

    def _db_has_embeddings(self) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(1) FROM movies").fetchone()
            return bool(row and row[0])

    def _build_index(self) -> None:
        rows = self._read_csv_rows()
        texts = [row["text"] for row in rows]
        embeddings = self.ai_service.embed_texts(texts)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM movies")
            for row, embedding in zip(rows, embeddings):
                embedding_blob = embedding.astype(np.float32).tobytes()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO movies (
                        movie_id,
                        title,
                        overview,
                        genres,
                        release_date,
                        runtime,
                        credits,
                        text,
                        embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["movie_id"],
                        row["title"],
                        row["overview"],
                        row["genres"],
                        row["release_date"],
                        row["runtime"],
                        row["credits"],
                        row["text"],
                        embedding_blob,
                    ),
                )

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


    def _load_chunks_from_db(self) -> List[MovieChunk]:
        chunks: List[MovieChunk] = []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT movie_id, title, overview, genres, release_date, runtime, credits,
                       text, embedding
                FROM movies
                """
            ).fetchall()
        for row in rows:
            embedding = np.frombuffer(row[8], dtype=np.float32)
            chunks.append(
                MovieChunk(
                    movie_id=row[0],
                    title=row[1],
                    overview=row[2],
                    genres=row[3],
                    release_date=row[4],
                    runtime=row[5],
                    credits=row[6],
                    text=row[7],
                    embedding=embedding,
                )
            )
        return chunks

    def _search(self, query_embedding: np.ndarray) -> List[MovieChunk]:
        if not self._chunks:
            return []
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scores: List[float] = []
        for chunk in self._chunks:
            chunk_norm = chunk.embedding / (np.linalg.norm(chunk.embedding) + 1e-8)
            scores.append(float(np.dot(query_norm, chunk_norm)))
        top_indices = np.argsort(scores)[-self.top_k :][::-1]
        return [self._chunks[idx] for idx in top_indices]

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

