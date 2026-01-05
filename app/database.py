import os
from typing import Any, Dict, List, Optional
import chromadb
from .models import MovieChunk
import numpy as np

class DatabaseService:
    def __init__(self, chroma_path: str) -> None:
        self.chroma_path = chroma_path
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection_name = "movies"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def _ensure_index(self, rag_service: Any) -> None:
        """
        Verifica si la colección tiene documentos. Si está vacía, construye el índice.
        """
        if self.collection.count() == 0:
            rows = rag_service._read_csv_rows()
            texts = [row["text"] for row in rows]
            embeddings = rag_service.ai_service.embed_texts(texts)
            self._build_index(rows, embeddings)

    def _build_index(self, rows: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> None:
        ids = []
        embeddings_list = [emb.tolist() for emb in embeddings]
        metadatas = []
        documents = []
        
        for i, row in enumerate(rows):
            # Usar movie_id y el índice para garantizar unicidad
            ids.append(f"{row['movie_id']}_{i}")
            metadatas.append({
                "movie_id": row["movie_id"],
                "title": row["title"],
                "overview": row["overview"],
                "genres": row["genres"],
                "release_date": row["release_date"],
                "runtime": row["runtime"],
                "credits": row["credits"],
            })
            documents.append(row["text"])

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[MovieChunk]:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        chunks = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]
                chunks.append(
                    MovieChunk(
                        movie_id=metadata.get("movie_id", results["ids"][0][i]),
                        title=metadata.get("title", ""),
                        overview=metadata.get("overview", ""),
                        genres=metadata.get("genres", ""),
                        release_date=metadata.get("release_date", ""),
                        runtime=metadata.get("runtime", ""),
                        credits=metadata.get("credits", ""),
                        text=results["documents"][0][i],
                        embedding=np.array(results["embeddings"][0][i]) if results["embeddings"] else np.array([])
                    )
                )
        return chunks
