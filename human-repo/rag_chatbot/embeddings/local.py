"""A deterministic, dependency-free embedder.

Tokens are hashed into a fixed-size vector (the "hashing trick"), then the
vector is L2-normalised so cosine similarity reduces to a dot product. It needs
no model, no network and no API key, which makes the whole pipeline runnable and
testable offline. Swap in :class:`OpenAIEmbedder` for production-quality vectors.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def hash_token(token: str) -> int:
    """32-bit FNV-1a hash of a token."""
    h = 2166136261
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


class LocalEmbedder:
    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _TOKEN.findall(text.lower()):
            vector[hash_token(token) % self._dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
