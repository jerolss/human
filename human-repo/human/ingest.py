"""Fill human's memory from the internet: RSS news, podcast transcripts, video transcripts.

    python -m human.ingest                 # read everything in config/sources.json
    python -m human.ingest --dry-run       # fetch + parse, print, write nothing
    python -m human.ingest --sample        # load the bundled offline sample (no network)
    python -m human.ingest --note notes.txt --title "my notes"

Standard library only. YouTube transcripts need the optional
`youtube-transcript-api` package (see requirements.txt); without it, videos
fall back to their title + description.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path

from .config import ROOT, Settings
from .memory import Item, Memory

UA = "human-ingest/0.1 (+https://github.com/)"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "media": "http://search.yahoo.com/mrss/",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "podcast": "https://podcastindex.org/namespace/1.0",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
}
SAMPLE = ROOT / "config" / "sample_memory.json"


# --- helpers -------------------------------------------------------------
def fetch(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


class _Text(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        if tag in self.SKIP:
            self._skip += 1
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):  # noqa: ANN001
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "p":
            self.parts.append("\n")

    def handle_data(self, data):  # noqa: ANN001
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    parser = _Text()
    parser.feed(raw)
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def strip_captions(raw: str) -> str:
    """Turn VTT/SRT/JSON transcript bodies into plain text."""
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            return " ".join(seg.get("body", "") for seg in data.get("segments", []))
        except json.JSONDecodeError:
            pass
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if (
            not line
            or line == "WEBVTT"
            or line.isdigit()
            or "-->" in line
            or line.startswith(("NOTE", "Kind:", "Language:"))
        ):
            continue
        lines.append(re.sub(r"<[^>]+>", "", line))
    text = " ".join(lines)
    return html_to_text(text) if "<" in text else text


def make_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _t(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def parse_feed(xml_bytes: bytes) -> list[dict]:
    """Parse RSS 2.0 or Atom into a list of entry dicts (keeps the raw element)."""
    root = ET.fromstring(xml_bytes)
    entries = []
    if root.tag.endswith("rss") or root.find("channel") is not None:
        channel = root.find("channel")
        feed_title = _t(channel.find("title")) if channel is not None else ""
        for it in root.iter("item"):
            body = _t(it.find("content:encoded", NS)) or _t(it.find("description"))
            entries.append(
                {
                    "feed": feed_title,
                    "title": _t(it.find("title")),
                    "url": _t(it.find("link")),
                    "published": _t(it.find("pubDate")),
                    "summary": html_to_text(body),
                    "el": it,
                }
            )
    else:
        feed_title = _t(root.find("atom:title", NS))
        for it in root.findall("atom:entry", NS):
            link = it.find("atom:link[@rel='alternate']", NS)
            if link is None:
                link = it.find("atom:link", NS)
            summary = (
                _t(it.find("atom:summary", NS))
                or _t(it.find("atom:content", NS))
                or _t(it.find("media:group/media:description", NS))
            )
            entries.append(
                {
                    "feed": feed_title,
                    "title": _t(it.find("atom:title", NS)),
                    "url": link.get("href", "") if link is not None else "",
                    "published": _t(it.find("atom:published", NS)) or _t(it.find("atom:updated", NS)),
                    "summary": html_to_text(summary),
                    "el": it,
                    "video_id": _t(it.find("yt:videoId", NS)),
                }
            )
    return entries


# --- source readers ------------------------------------------------------
def read_news(src: dict, limit: int, full_text: bool) -> Iterator[Item]:
    for e in parse_feed(fetch(src["url"]))[:limit]:
        text = e["summary"]
        if full_text and e["url"]:
            try:
                text = html_to_text(fetch(e["url"]).decode("utf-8", "replace")) or text
            except Exception as err:  # noqa: BLE001
                print(f"    ! full text failed for {e['url']}: {err}", file=sys.stderr)
        yield Item(
            id=make_id(e["url"] or e["title"]),
            kind="news",
            source=src.get("name") or e["feed"],
            title=e["title"],
            url=e["url"],
            text=text,
            published=e["published"],
        )


def read_podcast(src: dict, limit: int, _full_text: bool) -> Iterator[Item]:
    """Podcasts: use <podcast:transcript> (Podcasting 2.0) when present, else show notes."""
    for e in parse_feed(fetch(src["url"]))[:limit]:
        text, kind_note = e["summary"], "show notes"
        tr = e["el"].find("podcast:transcript", NS)
        if tr is not None and tr.get("url"):
            try:
                text = strip_captions(fetch(tr.get("url")).decode("utf-8", "replace"))
                kind_note = "transcript"
            except Exception as err:  # noqa: BLE001
                print(f"    ! transcript failed: {err}", file=sys.stderr)
        enclosure = e["el"].find("enclosure")
        url = e["url"] or (enclosure.get("url", "") if enclosure is not None else "")
        yield Item(
            id=make_id(url or e["title"]),
            kind="podcast",
            source=src.get("name") or e["feed"],
            title=f"{e['title']} ({kind_note})",
            url=url,
            text=text,
            published=e["published"],
        )


def youtube_transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import-not-found]
    except ImportError:
        return ""
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):  # < 1.0
            return " ".join(s["text"] for s in YouTubeTranscriptApi.get_transcript(video_id))
        fetched = YouTubeTranscriptApi().fetch(video_id)  # >= 1.0
        return " ".join(s.text for s in fetched)
    except Exception:  # noqa: BLE001  (no captions, region lock, rate limit...)
        return ""


def read_youtube(src: dict, limit: int, _full_text: bool) -> Iterator[Item]:
    url = src.get("url") or f"https://www.youtube.com/feeds/videos.xml?channel_id={src['channel_id']}"
    for e in parse_feed(fetch(url))[:limit]:
        text = youtube_transcript(e.get("video_id", "")) if e.get("video_id") else ""
        note = "transcript" if text else "description"
        yield Item(
            id=make_id(e["url"] or e["title"]),
            kind="video",
            source=src.get("name") or e["feed"],
            title=f"{e['title']} ({note})",
            url=e["url"],
            text=text or e["summary"],
            published=e["published"],
        )


READERS = {"news": read_news, "podcast": read_podcast, "youtube": read_youtube}


def load_sample() -> list[Item]:
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return [Item(id=make_id(d["url"] or d["title"]), **d) for d in data]


# --- main ----------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fill human's memory.")
    ap.add_argument("--dry-run", action="store_true", help="fetch and parse but write nothing")
    ap.add_argument("--sample", action="store_true", help="load the bundled offline sample only")
    ap.add_argument("--limit", type=int, default=15, help="max entries per source (default 15)")
    ap.add_argument("--only", choices=sorted(READERS), action="append", help="only these source types")
    ap.add_argument("--full-text", action="store_true", help="fetch full article pages for news")
    ap.add_argument("--reset", action="store_true", help="start from an empty memory")
    ap.add_argument("--note", type=Path, help="add a local text/markdown file to memory")
    ap.add_argument("--title", default="", help="title for --note")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    if args.reset and settings.memory_path.exists() and not args.dry_run:
        settings.memory_path.unlink()
    memory = Memory(settings.memory_path, dims=settings.embed_dims)

    items: list[Item] = []
    if args.sample:
        items = load_sample()
    elif args.note:
        text = args.note.read_text(encoding="utf-8")
        items = [
            Item(
                id=make_id(str(args.note.resolve()), text[:200]),
                kind="note",
                source="local",
                title=args.title or args.note.stem,
                url="",
                text=text,
            )
        ]
    else:
        sources = json.loads(settings.sources_path.read_text(encoding="utf-8"))["sources"]
        for src in sources:
            if src.get("enabled", True) is False or (args.only and src["type"] not in args.only):
                continue
            print(f"- {src['type']:8} {src.get('name', src.get('url'))}")
            try:
                got = list(READERS[src["type"]](src, args.limit, args.full_text))
            except Exception as err:  # noqa: BLE001
                print(f"    ! skipped: {err}", file=sys.stderr)
                continue
            new = [i for i in got if not memory.has(i.id)]
            print(f"    {len(got)} entries, {len(new)} new")
            items.extend(new)

    if args.dry_run:
        for i in items[:40]:
            print(f"  [{i.kind}] {i.title[:80]}  ({len(i.text.split())} words)")
        print(f"dry run: {len(items)} items parsed, nothing written")
        return 0

    added = memory.add(items)
    memory.save()
    print(f"added {added} items -> {settings.memory_path}  {memory.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
