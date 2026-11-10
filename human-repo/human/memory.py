"""A persistent memory: the upstream VectorStore plus a JSON file on disk.

Everything human "knows" lives here. Chat never searches the web; it only
reads from this file, which is why answers come back fast.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag_chatbot.chunking import chunk_text
from rag_chatbot.embeddings import LocalEmbedder
from rag_chatbot.embeddings.local import hash_token
from rag_chatbot.store import VectorStore
from rag_chatbot.types import Chunk, RetrievedChunk

FORMAT_VERSION = 1

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be but by can do does did for from has have he her his how i if in is it its "
    "me my no not of on or our she so that the their them there they this to too up was we were what "
    "when where which who why will with you your just about into than then out over also been being "
    "would could should get got am im dont its".split()
)


class HumanEmbedder(LocalEmbedder):
    """The upstream hashing embedder, minus stopwords and with crude suffix folding.

    Still offline and deterministic, just less distracted by words like "how" and "the".
    """

    @staticmethod
    def terms(text: str) -> list[str]:
        out = []
        for tok in _TOKEN.findall(text.lower()):
            if tok in STOPWORDS or len(tok) < 2:
                continue
            for suf in ("ing", "ies", "ed", "es", "s", "ly"):
                if tok.endswith(suf) and len(tok) - len(suf) >= 4:
                    tok = tok[: -len(suf)]
                    break
            out.append(tok)
        return out

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for tok in self.terms(text):
            vector[hash_token(tok) % self.dimensions] += 1.0
        vector = [math.sqrt(v) for v in vector]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


@dataclass
class Item:
    """One thing human has read: an article, an episode transcript, a video transcript."""

    id: str
    kind: str  # "news" | "podcast" | "video" | "note"
    source: str
    title: str
    url: str
    text: str
    published: str = ""
    added_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Recall:
    item: dict
    chunk: RetrievedChunk


class Memory:
    def __init__(self, path: Path, dims: int = 512, chunk_words: int = 120) -> None:
        self.path = Path(path)
        self.embedder = HumanEmbedder(dimensions=dims)
        self.chunk_words = chunk_words
        self.items: dict[str, dict] = {}
        self.store = VectorStore()
        if self.path.exists():
            self._load()

    # persistence ---------------------------------------------------------
    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("embed_dims") != self.embedder.dimensions:
            raise ValueError(
                f"{self.path} was built with embed_dims={data.get('embed_dims')}; "
                f"set HUMAN_EMBED_DIMS to match or re-run ingest with --reset"
            )
        self.items = data.get("items", {})
        chunks = [Chunk(id=c["id"], doc_id=c["doc_id"], text=c["text"]) for c in data["chunks"]]
        vectors = [c["v"] for c in data["chunks"]]
        if chunks:
            self.store.add(chunks, vectors)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        chunks = [
            {"id": c.id, "doc_id": c.doc_id, "text": c.text, "v": [round(x, 5) for x in v]}
            for c, v in zip(self.store._chunks, self.store._vectors, strict=True)
        ]
        payload = {
            "format": FORMAT_VERSION,
            "embed_dims": self.embedder.dimensions,
            "items": self.items,
            "chunks": chunks,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # writing -------------------------------------------------------------
    def has(self, item_id: str) -> bool:
        return item_id in self.items

    def add(self, items: Iterable[Item]) -> int:
        added = 0
        for item in items:
            if self.has(item.id) or not item.text.strip():
                continue
            pieces = chunk_text(f"{item.title}\n\n{item.text}", max_words=self.chunk_words)
            chunks = [Chunk(id=f"{item.id}#{i}", doc_id=item.id, text=text) for i, text in enumerate(pieces)]
            if not chunks:
                continue
            self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
            record = asdict(item)
            record.pop("text")
            self.items[item.id] = record
            added += 1
        return added

    # reading -------------------------------------------------------------
    def recall(self, query: str, k: int = 6) -> list[Recall]:
        if not len(self.store):
            return []
        [vector] = self.embedder.embed([query])
        hits = self.store.search(vector, k)
        return [Recall(item=self.items.get(h.chunk.doc_id, {}), chunk=h) for h in hits if h.score > 0]

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for item in self.items.values():
            kinds[item.get("kind", "?")] = kinds.get(item.get("kind", "?"), 0) + 1
        return {"items": len(self.items), "chunks": len(self.store), "by_kind": kinds}


def format_recall(recalls: Sequence[Recall]) -> str:
    """Render recalled memory as a numbered context block for the prompt."""
    lines = []
    for i, r in enumerate(recalls, 1):
        meta = r.item
        head = f"[{i}] ({meta.get('kind', '?')}) {meta.get('title', '')} | {meta.get('source', '')}"
        if meta.get("published"):
            head += f" | {meta['published']}"
        lines.append(f"{head}\n{r.chunk.chunk.text}")
    return "\n\n".join(lines)
