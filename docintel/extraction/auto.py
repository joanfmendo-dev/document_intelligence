"""extract_auto: decides what to read with OCR from the structure of the file, never from OCR confidence.

PDF, page by page:
  1. Read the text layer (as extract_digital does).
  2. Find the embedded images that are big enough to hold text and do not already have a text layer
     on top. OCR only those regions.
  3. Convert the OCR coordinates (pixels inside the image) to page coordinates and merge everything.
This one path covers a pure digital PDF (no images, no OCR), a full-page scan (one image covering the
page) and a hybrid page (text plus images with text).

Other files: DOCX / TXT -> extract_digital, image files -> extract_ocr.
"""
from __future__ import annotations

import math
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

from ._entries import Entry, build_result
from ._images import IMAGE_SUFFIXES, render_pdf_page
from .digital import extract_digital, pdf_page_entries
from .ocr import extract_ocr, ocr_rows

# Smaller images (logos, icons, rules) are not worth an OCR pass. Orientation does not matter,
# so a page rotated by 90 degrees gets the same treatment.
MIN_SHORT_SIDE_PT = 30
MIN_LONG_SIDE_PT = 100
# An image is a "searchable scan" (its text is already in the text layer) only if the text layer has
# at least one character per this many square points over it (min 10). A fixed count would let a short
# stamp on a scanned page hide the whole page: 20 characters on 600 x 800 pt is a stamp, not a text layer.
PT2_PER_CHAR = 5000


def extract_auto(path: str | Path, dpi: int = 300) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _auto_pdf(path, dpi)
    if suffix in {".docx", ".txt"}:
        return extract_digital(path)
    if suffix in IMAGE_SUFFIXES:
        return extract_ocr(path, dpi)
    raise ValueError(f"extract_auto does not support '{suffix}'")


def _image_regions(page) -> list[tuple[float, float, float, float]]:
    """Embedded images of a page (x0, top, x1, bottom in points, top-left origin) that need OCR."""
    centers = [((c["x0"] + c["x1"]) / 2, (c["top"] + c["bottom"]) / 2) for c in page.chars if c["text"].strip()]
    regions: list[tuple[float, float, float, float]] = []
    for im in page.images:
        x0, x1 = max(0.0, im["x0"]), min(float(page.width), im["x1"])
        top, bottom = max(0.0, im["top"]), min(float(page.height), im["bottom"])
        width, height = x1 - x0, bottom - top
        if min(width, height) < MIN_SHORT_SIDE_PT or max(width, height) < MIN_LONG_SIDE_PT:
            continue
        over = sum(1 for cx, cy in centers if x0 <= cx <= x1 and top <= cy <= bottom)
        if over < max(10, width * height / PT2_PER_CHAR) and (x0, top, x1, bottom) not in regions:
            regions.append((x0, top, x1, bottom))
    return regions


def _ocr_regions(rendered, regions: list[tuple[float, float, float, float]], scale: float) -> list[Entry]:
    """OCR crops of a rendered page; coordinates come back relative to the whole page."""
    W, H = rendered.size
    entries = []
    for x0, top, x1, bottom in regions:
        ox, oy = int(x0 * scale), int(top * scale)  # crop origin in render pixels, added back below
        crop = rendered.crop((ox, oy, min(W, math.ceil(x1 * scale)), min(H, math.ceil(bottom * scale))))
        entries += [Entry.from_abs(ox + rx0, oy + ry0, ox + rx1, oy + ry1, W, H, text)
                    for rx0, ry0, rx1, ry1, text in ocr_rows(crop)]
    return entries


def _auto_pdf(path: Path, dpi: int) -> dict:
    pages: list[list[Entry]] = []
    renderer = pdfium.PdfDocument(str(path))
    try:
        with pdfplumber.open(path) as pdf:
            for index, page in enumerate(pdf.pages):
                entries = pdf_page_entries(page)
                regions = _image_regions(page)
                if regions:
                    entries += _ocr_regions(render_pdf_page(renderer, index, dpi), regions, dpi / 72)
                pages.append(entries)
    finally:
        renderer.close()
    return build_result(pages)
