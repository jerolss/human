"""Minimal web chat: stdlib HTTP server + one HTML page.

python -m human.server            # http://127.0.0.1:8787
HUMAN_HOST=0.0.0.0 PORT=8080 python -m human.server
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .brain import Human

WEB = Path(__file__).resolve().parent / "web"
MAX_BODY = 64_000
BOT: Human | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "human/0.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: dict) -> None:
        self._send(code, json.dumps(data).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        assert BOT is not None
        if self.path in {"/", "/index.html"}:
            self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._json(
                200,
                {
                    "ok": True,
                    "offline": BOT.offline,
                    "memory": BOT.memory.stats(),
                    "model": None if BOT.offline else BOT.settings.model,
                },
            )
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        assert BOT is not None
        if self.path != "/api/chat":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY:
            self._json(413, {"error": "message too large"})
            return
        try:
            data = json.loads(self.rfile.read(length))
            message = str(data.get("message", "")).strip()[:4000]
            history = [m for m in data.get("history", []) if isinstance(m, dict)][-24:]
        except (ValueError, AttributeError):
            self._json(400, {"error": "bad json"})
            return
        if not message:
            self._json(400, {"error": "empty message"})
            return
        try:
            reply = BOT.reply(message, history)
        except RuntimeError as err:
            self._json(502, {"error": str(err)})
            return
        self._json(200, {"reply": reply.text, "sources": reply.sources, "offline": reply.offline})

    def log_message(self, fmt: str, *args) -> None:  # noqa: ANN002
        print(f"{self.address_string()} {fmt % args}")


def main() -> int:
    global BOT  # noqa: PLW0603
    BOT = Human()
    host, port = BOT.settings.host, BOT.settings.port
    print(f"human is listening on http://{host}:{port}  ({BOT.memory.stats()['items']} memories)")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
