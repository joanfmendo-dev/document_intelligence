"""The record form: its fields, how a record becomes form values, and how posted values become a record again.

`values` is the form as plain strings (what the user sees and types):
    {"patient_name": "Ana", "age": "63", "dob": "1962-09-24", ..., "vital_signs.heart_rate": "90 bpm",
     "labs": [{"test_name": "WBC Count", "result": "12.6 K/uL", "flag": "High"}]}
`fields` is the dictionary get_fields returns and RecordDB stores. `parse` turns the first into the second
and reports what the user got wrong, field by field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..db import COLUMNS


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    kind: str = "text"  # text | textarea | number | datetime | select
    rows: int = 2
    full: bool = False


PATIENT = [
    Field("record_id", "Record ID"), Field("patient_name", "Patient"),
    Field("dob", "Date of birth", "datetime"), Field("age", "Age", "number"),
    Field("sex", "Sex", "select"), Field("insurance", "Insurance"), Field("city", "City"),
    Field("department", "Department"), Field("encounter_date", "Encounter date", "datetime"),
]
HISTORY = [
    Field("chief_complaint", "Chief complaint", "textarea"), Field("allergies", "Allergies", "textarea"),
    Field("history_of_present_illness", "History of present illness", "textarea", 3, True),
    Field("past_medical_history", "Past medical history", "textarea"), Field("medications", "Medications", "textarea"),
]
EXAM = [Field("physical_exam", "Physical exam", "textarea", 3, True)]
FINAL = [
    Field("imaging_results", "Imaging results", "textarea"), Field("final_diagnosis", "Final diagnosis", "textarea"),
    Field("plan", "Plan", "textarea"), Field("progress_notes", "Progress notes", "textarea"),
]
SCALARS = PATIENT + HISTORY + EXAM + FINAL
VITALS = {"blood_pressure": "Blood pressure", "heart_rate": "Heart rate",
          "respiration_rate": "Respiration rate", "temperature": "Temperature", "sp02": "SpO2"}
LAB_COLUMNS = {"test_name": "Test", "result": "Result", "flag": "Flag"}
SEX_CHOICES = ("Female", "Male")
# The 40 tests found in medical_records.xlsx and the flags they use. They are suggestions to standardize records:
# the Test and Flag boxes accept any other text too.
LAB_TESTS = (
    "ALT", "AST", "Absolute Eosinophil Count", "Anion Gap",
    "BNP", "BUN", "Bicarbonate", "Blood Culture",
    "C-Reactive Protein", "Creatinine", "D-Dimer", "Eosinophils",
    "Glucose", "Glucose (point-of-care)", "Hemoglobin", "Lactate",
    "Lipase", "Neutrophils", "Oxygen Saturation (room air)", "Platelets",
    "Potassium", "Procalcitonin", "Rapid Strep Antigen", "Respiratory Viral Panel",
    "Serum Ketones", "Sodium", "Stool PCR (Salmonella)", "TSH",
    "Triglycerides", "Troponin I", "Troponin I (initial)", "Urine Bacteria",
    "Urine Blood", "Urine Nitrite", "Urine Protein", "Urine RBC",
    "Urine WBC", "WBC Count", "pCO2 (arterial)", "pH (arterial)",
)
LAB_FLAGS = ("Normal", "High", "Low", "Borderline")

RECORD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")  # also a URL segment and a folder name: keep it plain
DATE_HELP = "Use YYYY-MM-DD or YYYY-MM-DD HH:MM."


def _show_datetime(iso: str | None) -> str:
    """ISO text -> what the user edits: the date alone at midnight, seconds only when there are any."""
    if not iso:
        return ""
    dt = datetime.fromisoformat(iso)
    if dt.hour == dt.minute == dt.second == 0:
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d %H:%M" if dt.second == 0 else "%Y-%m-%d %H:%M:%S")


def _parse_datetime(text: str) -> str:
    dt = datetime.fromisoformat(text)  # ValueError for anything else; accepts 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM[:SS]', 'T'
    if dt.tzinfo is not None:
        raise ValueError("time zone")
    return dt.replace(microsecond=0).isoformat()


def age_on(dob_iso: str, encounter_iso: str) -> int:
    """Whole years from the date of birth to the encounter date (a birthday counts on the day itself)."""
    dob, enc = datetime.fromisoformat(dob_iso), datetime.fromisoformat(encounter_iso)
    return enc.year - dob.year - ((enc.month, enc.day) < (dob.month, dob.day))


def to_values(fields: dict) -> dict:
    """A record (or get_fields result) as form values."""
    values: dict[str, Any] = {}
    for f in SCALARS:
        v = fields.get(f.name)
        values[f.name] = "" if v is None else _show_datetime(v) if f.kind == "datetime" else str(v)
    vitals = fields.get("vital_signs") or {}
    for key in VITALS:
        v = vitals.get(key)
        values[f"vital_signs.{key}"] = "" if v is None else str(v)
    values["labs"] = [{c: "" if (row or {}).get(c) is None else str(row[c]) for c in LAB_COLUMNS}
                      for row in fields.get("lab_results") or []]
    return values


def from_form(form) -> dict:
    """Posted form data (anything with .get and .getlist) as form values, untouched except for stripping."""
    values: dict[str, Any] = {f.name: (form.get(f.name) or "").strip() for f in SCALARS}
    for key in VITALS:
        values[f"vital_signs.{key}"] = (form.get(f"vital_signs.{key}") or "").strip()
    columns = [form.getlist(f"lab_{c}") for c in LAB_COLUMNS]
    values["labs"] = [dict(zip(LAB_COLUMNS, (v.strip() for v in cells))) for cells in zip(*columns)]
    return values


def flat(values: dict) -> dict[str, str]:
    """Every value under the same names find_ungrounded uses: 'age', 'vital_signs.heart_rate', 'lab_results[0].result'."""
    out = {k: v for k, v in values.items() if k != "labs"}
    for i, row in enumerate(values.get("labs", [])):
        out.update({f"lab_results[{i}].{c}": v for c, v in row.items()})
    return out


def parse(values: dict) -> tuple[dict, dict[str, str]]:
    """Form values -> (fields, errors). `fields` has every column of the table; `errors` maps a field name
    ('age', 'vital_signs.heart_rate'...) to a message. Empty text is None, empty lab rows are dropped."""
    fields: dict[str, Any] = {c: None for c in COLUMNS}
    errors: dict[str, str] = {}
    for f in SCALARS:
        raw = (values.get(f.name) or "").strip()
        if not raw:
            continue
        if f.kind == "number":
            if raw.isascii() and raw.isdigit():
                fields[f.name] = int(raw)
            else:
                errors[f.name] = "Enter a whole number, for example 63."
        elif f.kind == "datetime":
            try:
                fields[f.name] = _parse_datetime(raw)
            except ValueError:
                errors[f.name] = f"Not a valid date. {DATE_HELP}"
        else:
            fields[f.name] = raw
    if not fields["patient_name"] and "patient_name" not in errors:
        errors["patient_name"] = "The patient name is required."
    if fields["dob"] and fields["encounter_date"]:  # the age is derived from the two dates: what was typed is replaced
        if fields["encounter_date"] < fields["dob"]:
            errors["encounter_date"] = "The encounter date cannot be earlier than the date of birth."
        else:
            fields["age"] = age_on(fields["dob"], fields["encounter_date"])
            errors.pop("age", None)
    if not fields["record_id"]:
        errors.setdefault("record_id", "The record ID is required.")
    elif not RECORD_ID.match(fields["record_id"]):
        errors["record_id"] = "Use only letters, digits, hyphen, dot or underscore (64 characters at most)."
    fields["vital_signs"] = {key: (values.get(f"vital_signs.{key}") or "").strip() or None for key in VITALS}
    labs = [{c: (row.get(c) or "").strip() or None for c in LAB_COLUMNS} for row in values.get("labs", [])]
    fields["lab_results"] = [row for row in labs if any(row.values())]
    return fields, errors


def label_of(path: str) -> str:
    """A readable name for a find_ungrounded path."""
    if path.startswith("vital_signs."):
        return VITALS.get(path.split(".", 1)[1], path)
    if path.startswith("lab_results["):
        index, column = re.match(r"lab_results\[(\d+)\]\.(\w+)", path).groups()
        return f"Lab result {int(index) + 1} ({LAB_COLUMNS.get(column, column).lower()})"
    return next((f.label for f in SCALARS if f.name == path), path)
