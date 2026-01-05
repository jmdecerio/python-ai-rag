import os
import sqlite3
import threading
from typing import Any, Dict, List

import numpy as np

from .models import MovieChunk


class DatabaseService:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._chunks: List[MovieChunk] = []
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _ensure_index(self) -> None:
        with self._lock:
            if self._chunks:
                return
            self._init_db()
            if not self._db_has_embeddings():
                raise ValueError("Database does not have embeddings. Please build the index first.")
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

    def _build_index(self, rows: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> None:
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
