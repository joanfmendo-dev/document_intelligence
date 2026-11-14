"""docintel: document extraction, the record database, the DocBase web app, the copilot and its evaluation.

Components (one folder each):
  shared      settings, LLM and embedding clients
  extraction  text extraction (digital / OCR / VLM / auto) and get_fields
  database    RecordDB (DuckDB), ICD-10 coding, embeddings
  web         the DocBase web app
  copilot     the read-only assistant that searches the records
  evaluation  execution accuracy and LLM-as-a-judge for the copilot
"""
from .database import RecordDB, RecordExistsError, RecordNotFoundError
from .extraction import extract_auto, extract_digital, extract_ocr, extract_vlm, get_fields
from .shared import LLM, TokenCount

__all__ = ["LLM", "RecordDB", "RecordExistsError", "RecordNotFoundError", "TokenCount", "extract_auto",
           "extract_digital", "extract_ocr", "extract_vlm", "get_fields"]
