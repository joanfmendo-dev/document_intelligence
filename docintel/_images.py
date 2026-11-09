"""Page images for OCR and VLM: PDFs are rendered with pypdfium2 (it bundles PDFium, so no
Poppler is needed on Windows); image files give one image per frame."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageSequence

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def flatten(img: Image.Image) -> Image.Image:
    """RGB copy; transparency becomes white (black would ruin OCR)."""
    if img.mode in ("RGBA", "LA", "P"):  # "P" may carry transparency
        img = Image.alpha_composite(Image.new("RGBA", img.size, "white"), img.convert("RGBA"))
    return img.convert("RGB")


def render_pdf_page(pdf: pdfium.PdfDocument, index: int, dpi: int) -> Image.Image:
    page = pdf[index]
    try:
        return flatten(page.render(scale=dpi / 72).to_pil())
    finally:
        page.close()


def iter_page_images(path: str | Path, dpi: int) -> Iterator[Image.Image]:
    """One RGB image per page of a PDF (rendered at `dpi`) or per frame of an image file (EXIF-rotated)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pdf = pdfium.PdfDocument(str(path))
        try:
            for index in range(len(pdf)):
                yield render_pdf_page(pdf, index, dpi)
        finally:
            pdf.close()
    elif suffix in IMAGE_SUFFIXES:
        with Image.open(path) as im:
            for frame in ImageSequence.Iterator(im):
                yield flatten(ImageOps.exif_transpose(frame.copy()))
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: .pdf and {sorted(IMAGE_SUFFIXES)}")
