import base64
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pytest
from docx import Document
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import docintel.auto as auto_mod
from docintel import extract_auto, extract_digital, extract_ocr, extract_vlm
from docintel._entries import Entry, render_page
from docintel._images import flatten

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
HYBRID = SAMPLES / "SYNUS-2026-0571.pdf"      # text layer + 3 embedded images with tables
SCAN = SAMPLES / "SYN-2026-0020.pdf"          # one full-page image, no text layer
DIGITAL = SAMPLES / "ENC-2026-88314.pdf"      # text layer only, no images


def entries_of(page: dict) -> list[Entry]:
    return [Entry(*e["coordinates"], e["text"]) for e in page["content"]["text_with_coordinates"]]


def assert_valid(result: dict):
    """{"text": str, "pages": [{"page": n, "content": {"text": str, "text_with_coordinates": [...]}}]}"""
    assert list(result) == ["text", "pages"] and result["pages"]
    for number, page in enumerate(result["pages"], start=1):
        assert list(page) == ["page", "content"] and page["page"] == number
        assert list(page["content"]) == ["text", "text_with_coordinates"]
        for e in page["content"]["text_with_coordinates"]:
            assert list(e) == ["coordinates", "text"] and e["text"].strip()
            x0, y0, x1, y1 = e["coordinates"]
            assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    # the document text is the pages' texts joined, and each page's text is rebuilt from its coordinates
    assert result["text"] == "\n\n".join(p["content"]["text"] for p in result["pages"] if p["content"]["text"].strip())
    for page in result["pages"]:
        if page["content"]["text_with_coordinates"]:
            assert render_page(entries_of(page)) == page["content"]["text"]


def make_pdf(path, *, image=None, invisible_text=None):
    """A one-page PDF with some visible text; `image` = (PIL image, x, y, width, height) in points;
    `invisible_text` is a text layer drawn invisibly (as in a searchable scan)."""
    c = canvas.Canvas(str(path), pagesize=(612, 792))
    c.setFont("Helvetica", 12)
    c.drawString(72, 20, "Plain page text")
    if image:
        img, x, y, w, h = image
        c.drawImage(ImageReader(img), x, y, width=w, height=h)
    if invisible_text:
        t = c.beginText(80, 600)
        t.setTextRenderMode(3)
        t.setFont("Helvetica", 12)
        t.textOut(invisible_text)
        c.drawText(t)
    c.save()
    return path


# ============================================================ extract_digital
def test_digital_pdf_markdown_headings_tables_and_pages():
    r = extract_digital(DIGITAL)
    assert_valid(r)
    assert "## CHIEF COMPLAINT" in r["text"] and "| Troponin I | 0.107 ng/mL | High |" in r["text"]
    assert "Chesttightness" not in r["text"]                   # words are not glued together
    two = extract_digital(SAMPLES / "PRV-2026-1002.pdf")
    assert_valid(two)
    assert [p["page"] for p in two["pages"]] == [1, 2]
    assert "Acute ischemic stroke" in two["pages"][1]["content"]["text"] and "Acute ischemic stroke" not in two["pages"][0]["content"]["text"]
    assert "## VITAL SIGNS\n\n| Blood Pressure |" in two["pages"][0]["content"]["text"]   # banner row becomes a heading


def test_digital_scanned_pdf_gives_an_empty_page():
    assert extract_digital(SCAN) == {"text": "", "pages": [{"page": 1, "content": {"text": "", "text_with_coordinates": []}}]}


def test_digital_silently_misses_text_inside_images():
    """Documents the limitation that extract_auto exists to solve."""
    assert "Samantha Nelson" not in extract_digital(HYBRID)["text"]


def test_digital_docx_pages_and_no_coordinates(tmp_path):
    d = Document()
    d.add_heading("Report", level=0)
    d.add_heading("Section", level=1)
    d.add_paragraph("Body text.")
    d.add_paragraph("item one", style="List Bullet")
    t = d.add_table(rows=2, cols=2)
    for (r, c), v in {(0, 0): "a", (0, 1): "b", (1, 0): "1", (1, 1): "2"}.items():
        t.cell(r, c).text = v
    d.add_page_break()
    d.add_paragraph("Second page.")
    p = d.add_paragraph("Third page, by 'page break before'.")
    p.paragraph_format.page_break_before = True
    d.save(tmp_path / "x.docx")
    r = extract_digital(tmp_path / "x.docx")
    assert_valid(r)
    assert [p["content"]["text"] for p in r["pages"]] == [
        "# Report\n\n# Section\n\nBody text.\n\n- item one\n\n| a | b |\n| --- | --- |\n| 1 | 2 |",
        "Second page.", "Third page, by 'page break before'."]
    assert all(p["content"]["text_with_coordinates"] == [] for p in r["pages"])

    d2 = Document()
    d2.add_paragraph("No page breaks at all.")
    d2.save(tmp_path / "y.docx")
    assert len(extract_digital(tmp_path / "y.docx")["pages"]) == 1


def test_digital_txt_encodings_and_pages(tmp_path):
    (tmp_path / "a.txt").write_bytes("Café ñandú\r\nline two".encode())
    (tmp_path / "b.txt").write_bytes("Café mañana".encode("cp1252"))
    (tmp_path / "c.txt").write_bytes("Año nuevo".encode("utf-16"))
    (tmp_path / "d.txt").write_text("page one\fpage two\f", encoding="utf-8")
    (tmp_path / "e.txt").write_text("", encoding="utf-8")
    assert extract_digital(tmp_path / "a.txt")["text"] == "Café ñandú\nline two"
    assert extract_digital(tmp_path / "b.txt")["text"] == "Café mañana"
    assert extract_digital(tmp_path / "c.txt")["text"] == "Año nuevo"
    d = extract_digital(tmp_path / "d.txt")
    assert [p["content"]["text"] for p in d["pages"]] == ["page one", "page two"] and d["text"] == "page one\n\npage two"
    assert extract_digital(tmp_path / "e.txt") == {"text": "", "pages": [{"page": 1, "content": {"text": "", "text_with_coordinates": []}}]}


@pytest.mark.parametrize("function", [extract_digital, extract_ocr, extract_auto, extract_vlm])
def test_unsupported_extension_is_rejected(function, tmp_path):
    (tmp_path / "x.xyz").write_text("hi")
    with pytest.raises(ValueError):
        function(tmp_path / "x.xyz")


# ============================================================ extract_ocr
def test_ocr_scanned_pdf_text_and_structure():
    r = extract_ocr(SAMPLES / "ADM-2026-77129.pdf")
    assert_valid(r)
    assert len(r["pages"]) == 2
    for expected in ("Robert Mitchell", "September 22, 1956", "88%", "907 pg/mL", "ALLERGIES", "VITAL SIGNS"):
        assert expected in r["text"], f"lost {expected!r}"
    assert "Sepsis secondary to community-acquired pneumonia" in r["pages"][1]["content"]["text"]


def test_ocr_does_not_flip_short_boxes():
    text = extract_ocr(SCAN)["text"]                          # regression: "98%" was read as "%86"
    assert "98%" in text and "%86" not in text and "98.6 °F" in text


def test_ocr_and_digital_coordinates_agree():
    """The same line, found by two unrelated methods, lands in the same place (0-1 space)."""
    line = "Telemetry admission, serial troponins, and cardiology consultation."
    find = lambda r: next(e for e in r["pages"][0]["content"]["text_with_coordinates"] if e["text"] == line)
    d, o = find(extract_digital(DIGITAL)), find(extract_ocr(DIGITAL))
    assert all(abs(a - b) < 0.01 for a, b in zip(d["coordinates"], o["coordinates"], strict=True)), (d, o)


def test_ocr_reads_every_frame_of_a_multipage_tiff(tmp_path):
    frames = [Image.new("RGB", (300, 200), "white") for _ in range(3)]
    frames[0].save(tmp_path / "m.tif", save_all=True, append_images=frames[1:])
    assert [p["page"] for p in extract_ocr(tmp_path / "m.tif")["pages"]] == [1, 2, 3]


def test_flatten_turns_transparency_white():
    assert flatten(Image.new("RGBA", (4, 4), (0, 0, 0, 0))).getpixel((1, 1)) == (255, 255, 255)
    assert flatten(Image.new("RGBA", (4, 4), (10, 20, 30, 255))).getpixel((1, 1)) == (10, 20, 30)
    assert flatten(Image.new("L", (4, 4), 7)).mode == "RGB"


def test_ocr_missing_backend_gives_actionable_error(monkeypatch):
    import rapidocr

    import docintel.ocr as ocr_mod

    def boom(*args, **kwargs):
        raise ImportError("onnxruntime is not installed.")

    monkeypatch.setattr(rapidocr, "RapidOCR", boom)
    monkeypatch.setattr(ocr_mod, "_engine", ocr_mod._engine.__wrapped__)   # bypass the cache
    with pytest.raises(RuntimeError, match="pip install onnxruntime"):
        extract_ocr(SCAN)


# ============================================================ extract_auto
@pytest.fixture
def ocr_images(monkeypatch):
    """Replace the OCR with a fake that records every image it is asked to read."""
    seen = []

    def fake(image):
        seen.append(image)
        return [(10, 20, 110, 40, "FAKE OCR ROW")]

    monkeypatch.setattr(auto_mod, "ocr_rows", fake)
    return seen


def test_auto_pdf_without_images_never_calls_ocr(ocr_images):
    assert extract_auto(DIGITAL) == extract_digital(DIGITAL) and ocr_images == []


def test_auto_converts_image_coordinates_to_page_coordinates(ocr_images):
    import pdfplumber

    r = extract_auto(HYBRID)
    assert_valid(r)
    assert len(ocr_images) == 3                                     # one OCR pass per image
    with pdfplumber.open(HYBRID) as pdf:
        page = pdf.pages[0]
        W, H = float(page.width), float(page.height)
        images = sorted(page.images, key=lambda im: im["top"])
    fakes = [e for e in r["pages"][0]["content"]["text_with_coordinates"] if e["text"] == "FAKE OCR ROW"]
    scale = 300 / 72
    assert len(fakes) == 3
    for fake, im in zip(sorted(fakes, key=lambda e: e["coordinates"][1]), images, strict=True):
        x0, y0, x1, _ = fake["coordinates"]
        # the fake row sits at pixel (10, 20)-(110, 40) INSIDE the image: expect image corner + that offset
        assert x0 == pytest.approx((im["x0"] * scale + 10) / (W * scale), abs=0.002)
        assert y0 == pytest.approx((im["top"] * scale + 20) / (H * scale), abs=0.002)
        assert x1 - x0 == pytest.approx(100 / (W * scale), abs=0.002)


def test_auto_real_ocr_recovers_the_tables_of_a_hybrid_page():
    r = extract_auto(HYBRID)
    assert_valid(r)
    for expected in ("Samantha Nelson", "September 24, 1962", "113/84 mmHg", "100.9 °F", "93%",
                     "WBC Count", "12.6 K/uL", "No growth to date", "## CHIEF COMPLAINT"):
        assert expected in r["text"], f"lost {expected!r}"
    assert "PROGRESS NOTES" in r["pages"][1]["content"]["text"]
    wbc = next(e for e in r["pages"][0]["content"]["text_with_coordinates"] if "WBC Count" in e["text"])
    assert wbc["coordinates"][0] == pytest.approx(56.2 / 612, abs=0.01)   # reference: OCR of the whole page
    assert wbc["coordinates"][1] == pytest.approx(562.6 / 792, abs=0.01)


def test_auto_full_page_scan_matches_plain_ocr():
    assert extract_auto(SCAN)["text"] == extract_ocr(SCAN)["text"]


def test_auto_skips_small_images_like_logos(tmp_path, ocr_images):
    r = extract_auto(make_pdf(tmp_path / "logo.pdf", image=(Image.new("RGB", (60, 20)), 72, 500, 60, 20)))
    assert ocr_images == [] and "Plain page text" in r["text"]


def test_auto_does_not_ocr_a_searchable_scan(tmp_path, ocr_images):
    """An image with a real text layer over it already has its text: OCR would only duplicate it."""
    small = make_pdf(tmp_path / "small.pdf", image=(Image.new("RGB", (400, 200)), 72, 500, 300, 150),
                     invisible_text="Already searchable text here")
    full = make_pdf(tmp_path / "full.pdf", image=(Image.new("RGB", (800, 1000)), 0, 0, 612, 792),
                    invisible_text="searchable page text " * 8)             # ~160 characters on a full page
    for pdf in (small, full):
        assert "searchable" in extract_auto(pdf)["text"]
    assert ocr_images == []


def test_auto_scan_with_a_short_stamp_is_still_ocred(tmp_path, ocr_images):
    """Regression: 22 characters of text layer on a full-page scan (a stamp, a page number) used to
    count as 'searchable scan' and the whole page was skipped."""
    pdf = make_pdf(tmp_path / "stamp.pdf", image=(Image.new("RGB", (800, 1000)), 0, 0, 612, 792))  # visible text: "Plain page text"
    r = extract_auto(pdf)
    assert len(ocr_images) == 1 and "FAKE OCR ROW" in r["text"] and "Plain page text" in r["text"]


def test_auto_large_image_without_text_layer_is_ocred(tmp_path, ocr_images):
    r = extract_auto(make_pdf(tmp_path / "big.pdf", image=(Image.new("RGB", (400, 200)), 72, 500, 300, 150)))
    assert len(ocr_images) == 1 and "FAKE OCR ROW" in r["text"] and "Plain page text" in r["text"]


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_auto_finds_images_on_rotated_pages(tmp_path, ocr_images, angle):
    """Regression: on a page rotated 90 degrees a 300 x 60 pt image became 60 x 300 and failed the
    minimum width, so it was never read. The crop must also be the right part of the page."""
    path = make_pdf(tmp_path / "r.pdf", image=(Image.new("RGB", (400, 80), (255, 0, 0)), 72, 500, 300, 60))
    pdf = pdfium.PdfDocument(str(path))
    pdf[0].set_rotation(angle)
    pdf.save(str(tmp_path / "rot.pdf"))
    pdf.close()
    extract_auto(tmp_path / "rot.pdf")
    assert len(ocr_images) == 1
    crop = ocr_images[0]
    px = np.asarray(crop.convert("RGB"))
    assert ((px[..., 0] > 200) & (px[..., 1] < 60) & (px[..., 2] < 60)).mean() > 0.9   # the crop is the red image
    assert sorted(crop.size) == pytest.approx([60 * 300 / 72, 300 * 300 / 72], abs=3)


def test_auto_other_formats(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    assert extract_auto(tmp_path / "a.txt")["text"] == "hello"
    Image.new("RGB", (300, 200), "white").save(tmp_path / "scan.png")
    assert extract_auto(tmp_path / "scan.png") == {
        "text": "", "pages": [{"page": 1, "content": {"text": "", "text_with_coordinates": []}}]}


# ============================================================ extract_vlm (fake Ollama server, real litellm)
@pytest.fixture
def page_image(tmp_path):
    Image.new("RGB", (300, 200), "white").save(tmp_path / "page.png")
    return tmp_path / "page.png"


def test_vlm_sends_the_image_and_returns_cleaned_pages(ollama, page_image):
    ollama.reply("```markdown\n## ALLERGIES\n\nNKDA.\n```")
    r = extract_vlm(page_image, think=False)
    assert r == {"text": "## ALLERGIES\n\nNKDA.",
                 "pages": [{"page": 1, "content": {"text": "## ALLERGIES\n\nNKDA.", "text_with_coordinates": []}}]}
    body = ollama.requests[0]
    assert body["model"] == "qwen3-vl:8b" and body["think"] is False
    assert body["options"]["temperature"] == 0 and body["options"]["num_predict"] == 4096
    user = body["messages"][-1]
    assert "Transcribe this document page" in user["content"] and len(user["images"]) == 1
    assert base64.b64decode(user["images"][0]).startswith(b"\x89PNG")


def test_vlm_makes_one_call_per_page_and_strips_reasoning(ollama):
    ollama.reply("<think>hmm</think>First page")
    ollama.reply("Second page")
    r = extract_vlm(SAMPLES / "PRV-2026-1002.pdf", model="ollama_chat/other-vlm:7b")
    assert [p["content"]["text"] for p in r["pages"]] == ["First page", "Second page"]
    assert r["text"] == "First page\n\nSecond page"
    assert len(ollama.requests) == 2 and ollama.requests[0]["model"] == "other-vlm:7b"
    assert "think" not in ollama.requests[0]                    # only sent when asked for


def test_vlm_caller_can_override_the_default_options(ollama, page_image):
    """Regression: passing temperature or max_tokens twice raised a TypeError."""
    ollama.reply("ok")
    extract_vlm(page_image, max_tokens=123, temperature=0.5)
    assert ollama.requests[0]["options"]["temperature"] == 0.5 and ollama.requests[0]["options"]["num_predict"] == 123


def test_vlm_warns_when_truncated(ollama, page_image):
    ollama.reply("partial", done_reason="length")
    with pytest.warns(UserWarning, match="output limit"):
        assert extract_vlm(page_image)["text"] == "partial"


def test_vlm_errors_are_actionable(ollama, page_image, monkeypatch):
    from docintel.llm import LLMError
    from docintel.settings import s

    ollama.reply("boom", status=500)
    with pytest.raises(LLMError, match="qwen3-vl"):
        extract_vlm(page_image)
    monkeypatch.setattr(s, "VLM_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(LLMError, match="Is the server running"):
        extract_vlm(page_image, timeout=5)


def test_every_method_returns_the_same_structure(ollama):
    ollama.reply("x")
    for result in (extract_digital(DIGITAL), extract_ocr(DIGITAL), extract_auto(DIGITAL), extract_vlm(DIGITAL)):
        assert_valid(result)
