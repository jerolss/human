"""An in-memory vector store with cosine-similarity search."""

from __future__ import annotations

from collections.abc import Sequence

from .types import Chunk, RetrievedChunk


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    """Holds chunk vectors and returns the nearest neighbours for a query.

    Vectors are assumed to be L2-normalised, so the dot product equals cosine
    similarity. For larger corpora, swap this for a real ANN index (FAISS,
    pgvector, etc.) behind the same ``add`` / ``search`` interface.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, query: Sequence[float], k: int) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk=chunk, score=_dot(vector, query))
            for chunk, vector in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]
