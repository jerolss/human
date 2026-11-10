"""Ties memory, persona and the LLM together into one reply function."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from rag_chatbot.chat import ExtractiveChatModel

from .config import Settings
from .llm import OpenRouterChat
from .memory import Memory, format_recall
from .persona import NO_MEMORY, SYSTEM_PROMPT


@dataclass
class Reply:
    text: str
    sources: list[dict]
    offline: bool


class Human:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.memory = Memory(self.settings.memory_path, dims=self.settings.embed_dims)
        self.llm = (
            OpenRouterChat(self.settings.openrouter_api_key, self.settings.model, self.settings.site_url)
            if self.settings.openrouter_api_key
            else None
        )
        self._fallback = ExtractiveChatModel()

    @property
    def offline(self) -> bool:
        return self.llm is None

    def reply(self, message: str, history: list[dict] | None = None) -> Reply:
        history = (history or [])[-self.settings.max_history :]
        recalls = self.memory.recall(message, k=self.settings.top_k)
        sources: list[dict] = []
        seen: set[str] = set()
        for i, r in enumerate(recalls, 1):
            doc_id = r.chunk.chunk.doc_id
            if doc_id in seen:
                continue  # several chunks of one article -> list the article once
            seen.add(doc_id)
            sources.append(
                {
                    "n": i,
                    "title": r.item.get("title", ""),
                    "url": r.item.get("url", ""),
                    "kind": r.item.get("kind", ""),
                    "source": r.item.get("source", ""),
                }
            )
        if self.llm is None:
            # No key: answer with the upstream extractive model so the loop still works.
            text = self._fallback.answer(message, [r.chunk for r in recalls])
            return Reply(f"(offline mode, no OPENROUTER_API_KEY) {text}", sources, True)

        system = SYSTEM_PROMPT.format(
            today=dt.date.today().isoformat(),
            memory=format_recall(recalls) or NO_MEMORY,
        )
        messages = [{"role": "system", "content": system}]
        messages += [m for m in history if m.get("role") in {"user", "assistant"} and m.get("content")]
        messages.append({"role": "user", "content": message})
        return Reply(self.llm.complete(messages), sources, False)
