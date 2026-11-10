"""OpenAI embeddings adapter using only the standard library (urllib)."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Sequence

_ENDPOINT = "https://api.openai.com/v1/embeddings"


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int = 1536,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model or os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload = json.dumps({"model": self._model, "input": list(texts)}).encode("utf-8")
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
        return [item["embedding"] for item in data["data"]]
