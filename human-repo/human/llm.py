"""OpenRouter chat client (stdlib only), adapted from rag_chatbot/chat/openai.py."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterChat:
    def __init__(
        self, api_key: str, model: str = "openrouter/auto", site_url: str = "", timeout: int = 60
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example)")
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.timeout = timeout

    def complete(self, messages: Sequence[dict], temperature: float = 0.8) -> str:
        payload = json.dumps(
            {"model": self.model, "messages": list(messages), "temperature": temperature}
        ).encode("utf-8")
        request = urllib.request.Request(
            ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url or "https://github.com/",
                "X-Title": "human",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                data = json.loads(response.read())
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"OpenRouter error {err.code}: {detail}") from err
        return data["choices"][0]["message"]["content"].strip()
