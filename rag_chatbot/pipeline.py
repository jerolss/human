"""The end-to-end RAG pipeline: ingest documents, then answer questions."""

from __future__ import annotations

from collections.abc import Sequence

from .chunking import chunk_text
from .store import VectorStore
from .types import ChatModel, Chunk, Document, Embedder, RagAnswer


class RagChatbot:
    """Wires an embedder, a vector store and a chat model into a RAG loop.

    >>> bot = RagChatbot(LocalEmbedder(), ExtractiveChatModel())
    >>> bot.ingest([Document(id="d1", text="The sky is blue.")])
    >>> bot.ask("What colour is the sky?").answer
    'The sky is blue.'
    """

    def __init__(self, embedder: Embedder, chat: ChatModel, top_k: int = 3) -> None:
        self._embedder = embedder
        self._chat = chat
        self._top_k = top_k
        self._store = VectorStore()

    def ingest(self, documents: Sequence[Document]) -> None:
        chunks: list[Chunk] = []
        for document in documents:
            for index, text in enumerate(chunk_text(document.text)):
                chunks.append(Chunk(id=f"{document.id}#{index}", doc_id=document.id, text=text))
        if not chunks:
            return
        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        self._store.add(chunks, vectors)

    def ask(self, question: str) -> RagAnswer:
        query_vector = self._embedder.embed([question])[0]
        sources = self._store.search(query_vector, self._top_k)
        answer = self._chat.answer(question, sources)
        return RagAnswer(answer=answer, sources=sources)
