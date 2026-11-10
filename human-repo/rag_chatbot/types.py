"""Core data types and protocols for the RAG pipeline.

The design keeps every component swappable behind a tiny interface: an
:class:`Embedder` turns text into vectors, a :class:`VectorStore` retrieves the
nearest chunks, and a :class:`ChatModel` turns retrieved context into an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol


@dataclass(frozen=True)
class Document:
    """A source document to be indexed."""

    id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A slice of a document, the unit that gets embedded and retrieved."""

    id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with its similarity score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RagAnswer:
    """The final answer plus the chunks it was grounded in."""

    answer: str
    sources: list[RetrievedChunk]


class Embedder(Protocol):
    """Turns text into fixed-size, comparable vectors."""

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    """Produces an answer to a question given retrieved context."""

    def answer(self, question: str, context: Sequence[RetrievedChunk]) -> str: ...
