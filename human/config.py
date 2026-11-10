"""Settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader: KEY=value lines, no overrides of real env vars."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    model: str
    memory_path: Path
    sources_path: Path
    top_k: int
    embed_dims: int
    max_history: int
    host: str
    port: int
    site_url: str

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        env = os.environ.get
        return cls(
            openrouter_api_key=env("OPENROUTER_API_KEY", ""),
            model=env("HUMAN_MODEL", "openrouter/auto"),
            memory_path=Path(env("HUMAN_MEMORY_PATH", str(ROOT / "memory" / "memory.json"))),
            sources_path=Path(env("HUMAN_SOURCES", str(ROOT / "config" / "sources.json"))),
            top_k=int(env("HUMAN_TOP_K", "6")),
            embed_dims=int(env("HUMAN_EMBED_DIMS", "512")),
            max_history=int(env("HUMAN_MAX_HISTORY", "12")),
            host=env("HUMAN_HOST", "127.0.0.1"),
            port=int(env("PORT", env("HUMAN_PORT", "8787"))),
            site_url=env("HUMAN_SITE_URL", "https://github.com/"),
        )
