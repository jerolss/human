"""Offline tests for human (run with `python -m unittest`)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from human.ingest import load_sample, parse_feed, strip_captions
from human.memory import Item, Memory

RSS = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0"><channel><title>Feed</title>
<item><title>Hello</title><link>https://example.com/a</link><description>&lt;p&gt;World news&lt;/p&gt;</description>
<podcast:transcript url="https://example.com/a.vtt" type="text/vtt"/></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015">
<title>Chan</title><entry><title>Vid</title><yt:videoId>abc123</yt:videoId>
<link rel="alternate" href="https://www.youtube.com/watch?v=abc123"/><published>2026-01-01</published></entry></feed>"""


class ParseTest(unittest.TestCase):
    def test_rss(self) -> None:
        [e] = parse_feed(RSS)
        self.assertEqual(
            (e["title"], e["url"], e["summary"]), ("Hello", "https://example.com/a", "World news")
        )

    def test_atom_youtube(self) -> None:
        [e] = parse_feed(ATOM)
        self.assertEqual(e["video_id"], "abc123")
        self.assertEqual(e["url"], "https://www.youtube.com/watch?v=abc123")

    def test_vtt(self) -> None:
        vtt = "WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\n<v Ann>Hi there.\n\n2\n00:00:02.000 --> 00:00:04.000\nHow are you?"
        self.assertEqual(strip_captions(vtt), "Hi there. How are you?")


class MemoryTest(unittest.TestCase):
    def test_roundtrip_and_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            mem = Memory(path, dims=128)
            self.assertEqual(mem.add(load_sample()), 5)
            mem.add([Item(id="x", kind="news", source="s", title="t", url="", text="")])  # empty skipped
            mem.save()
            again = Memory(path, dims=128)
            self.assertEqual(again.stats()["items"], 5)
            top = again.recall("how do I apologize to a friend", k=2)
            self.assertEqual(top[0].item["title"], "Saying sorry well")
            self.assertEqual(again.add(load_sample()), 0)  # dedupe

    def test_dims_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            Memory(path, dims=64).save()
            with self.assertRaises(ValueError):
                Memory(path, dims=128)


if __name__ == "__main__":
    unittest.main()
