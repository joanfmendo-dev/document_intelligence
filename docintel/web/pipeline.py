"""What happens to an uploaded file: read its text, ask the model for the fields, then code the diagnosis (ICD-10)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..database.data_transformation import ICD10
from ..extraction.auto import extract_auto
from ..extraction.fields import find_ungrounded, get_fields
from ..database.records import record_template
from ..shared.settings import s


# The page header repeats words the fields also use ("Record No. X", "Department of Medical Records"); the values are in the patient table.
INSTRUCTIONS = (
    "- record_id: the identifier alone, without its label. A header may say 'Record No. X': the value is X.\n"
    "- department: the clinical department or unit of the encounter (e.g. Cardiology, Intensive Care Unit), taken from the "
    "DEPARTMENT field of the patient details. The header line 'Department of Medical Records' is the hospital's own office, "
    "never the department of the encounter."
)


@dataclass
class Extraction:
    fields: dict
    ungrounded: list[str] = field(default_factory=list)  # paths of values that are not in the text (maybe invented)
    pages: int | None = None
    notice: str = ""  # something the user should know about the result (the ICD-10 coding failed, say)


def run(path: Path, on_step: Callable[[str], None]) -> Extraction:
    """extract_auto -> get_fields -> ICD10.code_fields. `on_step` is told 'reading', 'extracting' and 'coding'.
    A failure of the coding step does not lose the extraction: the ICD-10 fields stay blank, with a notice."""
    on_step("reading")
    if s.MAX_PAGES and path.suffix.lower() == ".pdf":
        import pypdfium2
        pages = len(pypdfium2.PdfDocument(path))
        if pages > s.MAX_PAGES:
            raise ValueError(f"The PDF has {pages} pages; the limit is {s.MAX_PAGES}.")
    document = extract_auto(path)
    if not document["text"].strip():
        raise ValueError("No text was found in the document. Is it blank, or is the image unreadable?")
    on_step("extracting")
    fields = get_fields(document["text"], record_template(), INSTRUCTIONS)
    ungrounded = find_ungrounded(fields, document["text"])  # before coding: the ICD-10 fields are derived, not read from the document
    on_step("coding")
    notice = ""
    try:
        fields = ICD10().code_fields(fields)
        if not fields["icd10_code"]:
            notice = "No ICD-10 code could be assigned automatically. You can type one in the ICD-10 section."
    except Exception as exc:
        fields = {**fields, **dict.fromkeys(ICD10.FIELDS)}
        notice = f"The ICD-10 coding failed ({type(exc).__name__}: {str(exc)[:200]}). You can type the code in the ICD-10 section."
    return Extraction(fields, ungrounded, len(document["pages"]), notice)
