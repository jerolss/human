"""A minimal, dependency-free RAG chatbot.

Local embeddings + cosine retrieval + grounded answers with citations, with
optional OpenAI adapters for production use.
"""

from .chat import ExtractiveChatModel, OpenAIChatModel
from .chunking import chunk_text
from .embeddings import LocalEmbedder, OpenAIEmbedder
from .pipeline import RagChatbot
from .store import VectorStore
from .types import Chunk, ChatModel, Document, Embedder, RagAnswer, RetrievedChunk

__version__ = "0.1.0"

__all__ = [
    "RagChatbot",
    "VectorStore",
    "chunk_text",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "ExtractiveChatModel",
    "OpenAIChatModel",
    "Document",
    "Chunk",
    "RetrievedChunk",
    "RagAnswer",
    "Embedder",
    "ChatModel",
]
