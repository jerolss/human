"""Talk to human in your terminal.

python -m human.chat
python -m human.chat --once "what's been going on this week?"
"""

from __future__ import annotations

import argparse

from .brain import Human


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chat with human.")
    ap.add_argument("--once", help="ask one question and exit")
    ap.add_argument("--sources", action="store_true", help="print which memories were used")
    args = ap.parse_args(argv)

    bot = Human()
    stats = bot.memory.stats()
    mode = "offline (no OPENROUTER_API_KEY)" if bot.offline else bot.settings.model
    print(f"human · {stats['items']} things in memory · model: {mode}")

    def turn(msg: str, history: list[dict]) -> None:
        reply = bot.reply(msg, history)
        print(f"\nhuman › {reply.text}\n")
        if args.sources:
            for s in reply.sources:
                print(f"   [{s['n']}] {s['kind']}: {s['title']} {s['url']}")
        history += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply.text}]

    history: list[dict] = []
    if args.once:
        try:
            turn(args.once, history)
        except RuntimeError as err:
            print(f"! {err}")
            return 1
        return 0
    print("type, enter to send. /exit to leave, /reset to forget this conversation.\n")
    while True:
        try:
            msg = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if msg in {"/exit", "/quit"}:
            return 0
        if msg == "/reset":
            history.clear()
            print("(conversation cleared, memory kept)\n")
            continue
        if msg:
            try:
                turn(msg, history)
            except RuntimeError as err:
                print(f"! {err}\n")


if __name__ == "__main__":
    raise SystemExit(main())
