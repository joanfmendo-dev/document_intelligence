"""Document text extraction (digital / OCR / VLM / auto) and LLM field extraction (get_fields)."""
from .auto import extract_auto
from .digital import extract_digital
from .fields import get_fields
from .ocr import extract_ocr
from .vlm import extract_vlm

__all__ = ["extract_auto", "extract_digital", "extract_ocr", "extract_vlm", "get_fields"]
