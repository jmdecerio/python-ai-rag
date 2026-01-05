from .models import MovieChunk
from dataclasses import dataclass
import numpy as np

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
