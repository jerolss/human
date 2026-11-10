"""OpenAI Chat Completions adapter that produces grounded, cited answers.

Uses only the standard library (urllib) so the package stays dependency-free.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Sequence

from ..types import RetrievedChunk

_ENDPOINT = "https://api.openai.com/v1/chat/completions"

_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. Cite sources like [1]."
)


class OpenAIChatModel:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    def answer(self, question: str, context: Sequence[RetrievedChunk]) -> str:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        context_block = "\n\n".join(
            f"[{i + 1}] {retrieved.chunk.text}" for i, retrieved in enumerate(context)
        )
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

        payload = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            _ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())
        return data["choices"][0]["message"]["content"].strip()
