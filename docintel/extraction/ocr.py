"""extract_ocr: text from scanned PDFs and image files with RapidOCR (Apache-2.0).

RapidOCR returns text boxes; their geometry is used to rebuild reading order and rows.
Rows whose columns are far apart come out as "col1 | col2 | col3" (not strict Markdown
tables: OCR has no cell structure; extract_vlm is the better option for complex tables).

Returns {"text": ..., "pages": [...]} (see _entries.py) with one entry per text row.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from ._entries import Entry, build_result
from ._images import iter_page_images

_COLUMN_GAP = 1.5     # horizontal gap, in text heights, that separates two columns
_ROW_TOLERANCE = 0.6  # a box joins a row if its centre is within this many heights
_BORDER_GLYPHS = set("|¦_=~—–")  # table borders / rules the engine may read as text


@dataclass
class _Box:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 1.0)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def _clean(text: str) -> str:
    """Safe fixes for systematic OCR artefacts (e.g. the single glyph for degrees Fahrenheit)."""
    return text.replace("°℉", "°F").replace("°℃", "°C").replace("℉", "°F").replace("℃", "°C").strip()


@lru_cache(maxsize=1)
def _engine():
    """RapidOCR loads its models once per process."""
    try:
        from rapidocr import RapidOCR

        return RapidOCR()
    except ImportError as exc:
        raise RuntimeError(
            f"RapidOCR could not start ({exc}). It needs an inference backend: run `pip install onnxruntime`."
        ) from exc


def _recognize(image) -> list[_Box]:
    import numpy as np

    # use_cls=False: RapidOCR's text-angle classifier flips short boxes on upright documents
    # ("98%" was read as "%86" with 0.99 confidence).
    result = _engine()(np.asarray(image), use_cls=False)
    if result is None or result.txts is None:
        return []
    boxes = []
    for quad, text in zip(result.boxes, result.txts, strict=True):
        text = _clean(text or "")
        if not text or set(text) <= _BORDER_GLYPHS:
            continue
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        boxes.append(_Box(text, min(xs), min(ys), max(xs), max(ys)))
    return boxes


def ocr_rows(image) -> list[tuple[float, float, float, float, str]]:
    """OCR an image and return text rows as (x0, y0, x1, y1, text) in PIXELS of that image.

    Boxes are grouped into rows by vertical position, ordered left to right, and distant
    ones are joined with ' | '."""
    boxes = _recognize(image)
    rows: list[list[_Box]] = []
    for box in sorted(boxes, key=lambda b: b.cy):
        if rows:
            row = rows[-1]
            row_cy = statistics.mean(b.cy for b in row)
            row_h = statistics.median(b.h for b in row)
            if abs(box.cy - row_cy) <= _ROW_TOLERANCE * min(box.h, row_h):
                row.append(box)
                continue
        rows.append([box])

    result = []
    for row in rows:
        row.sort(key=lambda b: b.x0)
        row_h = statistics.median(b.h for b in row)
        parts = [row[0].text]
        for prev, cur in pairwise(row):
            parts.append(" | " if cur.x0 - prev.x1 > _COLUMN_GAP * row_h else " ")
            parts.append(cur.text)
        result.append((min(b.x0 for b in row), min(b.y0 for b in row),
                       max(b.x1 for b in row), max(b.y1 for b in row), "".join(parts)))
    return result


def extract_ocr(path: str | Path, dpi: int = 300) -> dict:
    """OCR a scanned PDF or an image file (png, jpg, tiff, bmp, webp); every page is read as a whole."""
    return build_result([
        [Entry.from_abs(x0, y0, x1, y1, image.width, image.height, text) for x0, y0, x1, y1, text in ocr_rows(image)]
        for image in iter_page_images(path, dpi)
    ])
