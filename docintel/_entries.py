"""The result format shared by every extractor:

    {"text": "all the text of the document",
     "pages": [{"page": 1, "content": {"text": "all the text of the page",
                                       "text_with_coordinates": [{"coordinates": [x0, y0, x1, y1], "text": "..."}]}}]}

Extractors that can locate text produce `Entry` objects (a piece of text + where it sits on its
page); each page's `text` is rebuilt FROM its entries by `render_page`, so `text` and
`text_with_coordinates` always agree. Methods that cannot locate text leave the list empty.
Coordinates are normalized to 0-1 over the page, origin at the top-left.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from itertools import pairwise

GAP_FACTOR = 0.8  # a vertical gap larger than this many (median) line heights starts a new paragraph


@dataclass
class Entry:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @classmethod
    def from_abs(cls, x0: float, y0: float, x1: float, y1: float, width: float, height: float, text: str) -> Entry:
        unit = lambda v: min(max(v, 0.0), 1.0)
        return cls(unit(x0 / width), unit(y0 / height), unit(x1 / width), unit(y1 / height), text)

    def as_dict(self) -> dict:
        return {"coordinates": [round(v, 4) for v in (self.x0, self.y0, self.x1, self.y1)], "text": self.text}


def _structural(text: str) -> bool:
    """Markdown headings and tables are always separated by a blank line."""
    return text.startswith(("#", "|"))


def _reading_order(entries: list[Entry]) -> list[Entry]:
    return sorted(entries, key=lambda e: (round(e.y0, 4), e.x0))


def render_page(entries: list[Entry]) -> str:
    """Rebuild a page's text: entries in reading order, one per line, with a blank line between
    paragraphs (around a structural entry, or after a larger-than-usual vertical gap)."""
    entries = _reading_order(entries)
    if not entries:
        return ""
    typical = statistics.median([e.y1 - e.y0 for e in entries if not _structural(e.text)]
                                or [e.y1 - e.y0 for e in entries])
    parts = [entries[0].text]
    for prev, e in pairwise(entries):
        new_paragraph = _structural(e.text) or _structural(prev.text) or e.y0 - prev.y1 > GAP_FACTOR * typical
        parts += ["\n\n" if new_paragraph else "\n", e.text]
    return "".join(parts).strip()


def _result(pages: list[tuple[str, list[Entry]]]) -> dict:
    return {
        "text": "\n\n".join(text for text, _ in pages if text.strip()).strip(),
        "pages": [{"page": number, "content": {"text": text, "text_with_coordinates": [e.as_dict() for e in entries]}}
                  for number, (text, entries) in enumerate(pages, start=1)],
    }


def build_result(pages: list[list[Entry]]) -> dict:
    """From the entries of each page (an empty list for a page without text)."""
    return _result([(render_page(entries), _reading_order(entries)) for entries in pages])


def text_result(texts: list[str]) -> dict:
    """For methods that cannot locate text (Word, plain text, VLM): one text per page."""
    return _result([(text, []) for text in texts])
