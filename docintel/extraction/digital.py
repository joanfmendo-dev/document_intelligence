"""extract_digital: text from documents that have readable text, without OCR.

Supports PDF (text layer), DOCX and TXT. Output is Markdown: headings are detected from
font weight/size (PDF) or styles (DOCX), and tables become Markdown tables.

A PDF page that is a scanned image has no text layer: it contributes nothing to the result
(use extract_ocr, or extract_auto to handle such pages automatically).

Returns {"text": ..., "pages": [...]} (see _entries.py). Coordinates exist only for PDFs (one entry
per line, heading or table); Word and plain text have no page layout, so their lists are empty.
Word pages follow its page breaks, when it has any.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import pdfplumber
from pdfplumber.utils import cluster_objects, extract_text as _plumber_extract_text

from ._entries import Entry, build_result, text_result

_BOLD_KEYWORDS = ("bold", "black", "heavy")
_COLUMN_GAP_EM = 1.2       # horizontal gap (in font sizes) that splits a line into segments
_HEADING_BIG_RATIO = 1.35  # bold line this much bigger than body text -> "#", otherwise "##"


def extract_digital(path: str | Path) -> dict:
    """Read a PDF, DOCX or TXT file that has readable text (no OCR)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pdfplumber.open(path) as pdf:
            return build_result([pdf_page_entries(page) for page in pdf.pages])
    if suffix == ".docx":
        return text_result(_docx(path))
    if suffix == ".txt":
        return text_result(_txt(path))
    raise ValueError(f"extract_digital does not support '{suffix}'. Supported: .pdf, .docx, .txt")


# ----------------------------------------------------------------------------- PDF
def _is_bold(char: dict[str, Any]) -> bool:
    return any(k in (char.get("fontname") or "").lower() for k in _BOLD_KEYWORDS)


def _split_segments(line_chars: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split one visual line into segments wherever there is a column-sized gap.

    Space glyphs stay inside the segment (needed to rebuild the text); gaps are measured
    only between visible characters.
    """
    chars = sorted(line_chars, key=lambda c: c["x0"])
    visible = [c for c in chars if c["text"].strip()]
    if not visible:
        return []
    size = statistics.median(c["size"] for c in visible) or 10.0
    segments: list[list[dict[str, Any]]] = [[]]
    last_visible = None
    for c in chars:
        if c["text"].strip():
            if last_visible is not None and c["x0"] - last_visible["x1"] > _COLUMN_GAP_EM * size:
                segments.append([])
            last_visible = c
        segments[-1].append(c)
    return [seg for seg in segments if any(ch["text"].strip() for ch in seg)]


def _rows_to_markdown(rows: list[list[Any]]) -> str:
    def clean(cell: Any) -> str:
        return " ".join(str(cell or "").split()).replace("|", "\\|")

    grid = [[clean(c) for c in row] for row in rows]
    grid = [r for r in grid if any(r)]
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    lines = ["| " + " | ".join(grid[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(lines)


def pdf_page_entries(page) -> list[Entry]:
    """Text of one PDF page as entries (lines, headings, tables) with normalized coordinates."""
    visible = [c for c in page.chars if c["text"].strip()]
    if not visible:
        return []  # no text layer (scanned page)
    W, H = float(page.width), float(page.height)
    body_size = statistics.median(c["size"] for c in visible) or 10.0

    entries: list[Entry] = []

    def add(x0: float, top: float, x1: float, bottom: float, text: str) -> None:
        entries.append(Entry.from_abs(x0, top, x1, bottom, W, H, text))

    # 1) grid-ruled tables. Single-cell "tables" are decorated boxes/banners: ignore them.
    table_boxes = []
    for table in page.find_tables():
        rows = table.extract()
        if len(rows) <= 1 and all(len(r) <= 1 for r in rows):
            continue
        x0, top, x1, bottom = table.bbox
        table_boxes.append(table.bbox)
        # A banner row (only the first cell filled) is a section title drawn on top of the
        # grid: emit it as a heading and keep the remaining rows as the table.
        first = [str(c or "").strip() for c in rows[0]]
        if len(rows) > 1 and len(first) > 1 and first[0] and not any(first[1:]):
            r0 = table.rows[0].bbox
            add(r0[0], r0[1], r0[2], r0[3], "## " + " ".join(first[0].split()))
            rows = rows[1:]
            top = r0[3]
        md = _rows_to_markdown(rows)
        if md:
            add(x0, top, x1, bottom, md)

    # 2) everything else, line by line, outside the tables
    def outside_tables(obj: dict[str, Any]) -> bool:
        if obj.get("object_type") != "char":
            return True
        cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
        return not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in table_boxes)

    scope = page.filter(outside_tables) if table_boxes else page
    # Lines are grouped from the raw chars: page.extract_text_lines() drops space glyphs.
    for line_chars in cluster_objects(scope.chars, lambda c: c["top"], 3):
        for seg in _split_segments(line_chars):
            text = " ".join((_plumber_extract_text(seg) or "").split())
            if not text:
                continue
            size = statistics.median(c["size"] for c in seg)
            if all(_is_bold(c) for c in seg) and len(text) <= 120 and any(ch.isalpha() for ch in text):
                text = ("# " if size >= _HEADING_BIG_RATIO * body_size else "## ") + text
            add(min(c["x0"] for c in seg), min(c["top"] for c in seg),
                max(c["x1"] for c in seg), max(c["bottom"] for c in seg), text)
    return entries


# ----------------------------------------------------------------------------- DOCX
def _docx(path: Path) -> list[str]:
    """One text per page. Pages follow explicit page breaks, "page break before" paragraphs and the
    breaks Word recorded the last time it laid the document out; with none of those it is one page."""
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    pages: list[list[str]] = [[]]

    def new_page() -> None:
        if pages[-1]:  # never create empty pages
            pages.append([])

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            if child.xpath(".//w:lastRenderedPageBreak | .//w:pageBreakBefore"):
                new_page()
            line = _docx_paragraph(Paragraph(child, doc))
            if line:
                pages[-1].append(line)
            if child.xpath('.//w:br[@w:type="page"]'):
                new_page()
        elif child.tag == qn("w:tbl"):
            pages[-1].append(_docx_table(Table(child, doc)))
    return ["\n\n".join(c for c in chunks if c) for chunks in pages if any(chunks)] or [""]


def _docx_paragraph(para) -> str:
    text = para.text.strip()
    style = ((para.style.name if para.style is not None else "") or "").lower()
    if not text:
        return ""
    if style == "title":
        return f"# {text}"
    if style.startswith("heading"):
        level = "".join(ch for ch in style if ch.isdigit()) or "1"
        return f"{'#' * min(int(level), 6)} {text}"
    if style.startswith("list number"):
        return f"1. {text}"
    if style.startswith("list") or (para._p.pPr is not None and para._p.pPr.numPr is not None):
        return f"- {text}"
    return text


def _docx_table(table) -> str:
    rows = []
    for row in table.rows:
        cells, seen = [], None
        for cell in row.cells:
            if seen is not None and cell._tc is seen:  # horizontally merged cell repeats
                continue
            seen = cell._tc
            cells.append(cell.text)
        rows.append(cells)
    return _rows_to_markdown(rows)


# ----------------------------------------------------------------------------- TXT
def _txt(path: Path) -> list[str]:
    """UTF-8 (with or without BOM), UTF-16 with BOM, otherwise cp1252. A form feed starts a new page.

    Statistical encoding detectors guess badly on short files, so none is used."""
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1252", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pages = [chunk.strip("\n") for chunk in text.split("\f") if chunk.strip()]
    return pages or [""]
