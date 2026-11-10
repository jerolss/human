"""Split documents into retrieval-sized chunks."""

from __future__ import annotations

import re

_PARAGRAPH = re.compile(r"\n\s*\n")


def chunk_text(text: str, max_words: int = 60) -> list[str]:
    """Split ``text`` into chunks of at most ``max_words`` words.

    Paragraphs are kept whole when they fit; longer ones are windowed. This
    keeps semantically coherent pieces together, which improves retrieval.
    """
    chunks: list[str] = []
    for paragraph in (p.strip() for p in _PARAGRAPH.split(text)):
        if not paragraph:
            continue
        words = paragraph.split()
        if len(words) <= max_words:
            chunks.append(paragraph)
            continue
        for start in range(0, len(words), max_words):
            chunks.append(" ".join(words[start : start + max_words]))
    return chunks
