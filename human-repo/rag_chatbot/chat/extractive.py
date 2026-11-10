"""An offline, extractive answerer.

It returns the sentence from the retrieved context that overlaps the question
most. No model or API key required, which keeps the demo and tests fully
offline while still exercising the real retrieval path.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..types import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


class ExtractiveChatModel:
    def answer(self, question: str, context: Sequence[RetrievedChunk]) -> str:
        if not context:
            return "I don't have any information about that."

        question_tokens = _tokens(question)
        best_sentence = ""
        best_score = -1

        for retrieved in context:
            for sentence in _sentences(retrieved.chunk.text):
                overlap = len(_tokens(sentence) & question_tokens)
                if overlap > best_score:
                    best_sentence, best_score = sentence, overlap

        return best_sentence if best_score > 0 else context[0].chunk.text
