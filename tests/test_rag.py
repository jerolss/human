"""Tests for the from-scratch RAG pipeline (run with `python -m unittest`)."""

from __future__ import annotations

import unittest

from rag_chatbot import (
    Document,
    ExtractiveChatModel,
    LocalEmbedder,
    RagChatbot,
)
from rag_chatbot.chunking import chunk_text


class ChunkingTest(unittest.TestCase):
    def test_short_paragraph_is_one_chunk(self) -> None:
        self.assertEqual(chunk_text("a b c"), ["a b c"])

    def test_long_paragraph_is_windowed(self) -> None:
        text = " ".join(str(i) for i in range(150))
        chunks = chunk_text(text, max_words=60)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c.split()) <= 60 for c in chunks))


class EmbedderTest(unittest.TestCase):
    def test_vectors_are_normalised(self) -> None:
        [vector] = LocalEmbedder(dimensions=64).embed(["hello world hello"])
        length = sum(value * value for value in vector) ** 0.5
        self.assertAlmostEqual(length, 1.0, places=6)

    def test_deterministic(self) -> None:
        embedder = LocalEmbedder()
        self.assertEqual(embedder.embed(["same text"]), embedder.embed(["same text"]))


class RagTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bot = RagChatbot(LocalEmbedder(), ExtractiveChatModel())
        self.bot.ingest(
            [
                Document(id="sky", text="The sky is blue during the day."),
                Document(id="grass", text="Grass is green because of chlorophyll."),
            ]
        )

    def test_retrieves_and_answers_from_the_right_document(self) -> None:
        result = self.bot.ask("What colour is the sky?")
        self.assertIn("blue", result.answer.lower())
        self.assertEqual(result.sources[0].chunk.doc_id, "sky")

    def test_empty_store_is_graceful(self) -> None:
        empty = RagChatbot(LocalEmbedder(), ExtractiveChatModel())
        answer = empty.ask("anything?")
        self.assertEqual(answer.sources, [])


if __name__ == "__main__":
    unittest.main()
