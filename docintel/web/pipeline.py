"""What happens to an uploaded file: read its text, then ask the model for the fields."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..auto import extract_auto
from ..fields import find_ungrounded, get_fields
from ..settings import s


@dataclass
class Extraction:
    fields: dict
    ungrounded: list[str] = field(default_factory=list)  # paths of values that are not in the text (maybe invented)
    pages: int | None = None


def run(path: Path, on_step: Callable[[str], None]) -> Extraction:
    """extract_auto -> get_fields. `on_step` is told 'reading' and then 'extracting'."""
    on_step("reading")
    document = extract_auto(path)
    if not document["text"].strip():
        raise ValueError("No text was found in the document. Is it blank, or is the image unreadable?")
    on_step("extracting")
    fields = get_fields(document["text"], s.DB_JSON_TEMPLATE)
    return Extraction(fields, find_ungrounded(fields, document["text"]), len(document["pages"]))
