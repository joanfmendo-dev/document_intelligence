"""The semantic catalog: what the copilot is told about the database.

A small YAML file lists, for every column the copilot can use, a role, a one-line description and (for categorical columns)
the values that exist with their counts. Everything else the code needs (which operators a column accepts, which can be
grouped, how a vital sign is parsed) follows from the role, so it is not repeated in the file.

    columns:
      department:
        role: categorical
        description: Clinical department that handled the encounter.
        values: {Cardiology: 193, Pulmonology: 190}      # generated
      vital_signs.temperature:
        role: measurement
        description: Body temperature.
        unit: °F

The catalog is built by a profiler (`Catalog.build`): code computes the facts (values, counts, ranges); descriptions come from the
built-in text below, from the previous file (a person's edits are kept), or, for a column nobody described, from a model (optional).
`refresh` recomputes only the facts, so a description is never overwritten. The web app calls it after every write, which keeps
`values` current; the copilot never writes the catalog.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..database.records import TABLE
from ..shared.settings import s

ROLES = ("identifier", "name", "categorical", "date", "integer", "free_text", "measurement", "list_of_objects", "icd10_hierarchy")
OPS_BY_ROLE = {
    "identifier": ("eq", "in"),
    "name": ("eq", "contains"),
    "categorical": ("eq", "in"),
    "date": ("between",),
    "integer": ("eq", "gt", "gte", "lt", "lte", "between"),
    "free_text": ("contains",),
    "measurement": ("gt", "gte", "lt", "lte", "between"),
    "list_of_objects": ("has",),
    "icd10_hierarchy": ("eq", "in"),
}
MENTION_STOP = {"patients", "patient", "records", "record", "tested", "which", "where", "there", "these", "those", "about", "their", "since",
                "between", "during", "number", "total", "average", "other", "show", "found", "visit", "visits", "chart"}
GROUPABLE = {"categorical", "icd10_hierarchy"}
NUMERIC = {"integer", "measurement"}
ALL_OPS = ("eq", "in", "contains", "gt", "gte", "lt", "lte", "between", "has")

LAB = "lab_results"                       # the filter field for lab results; the catalog has lab_results.test_name / .flag
LAB_NAME, LAB_FLAG = "lab_results.test_name", "lab_results.flag"

# How each vital sign is read out of the JSON text ("158/95 mmHg", "99.4 °F", "98%"): key, regex for the number, unit, plausible range.
VITALS = {
    "vital_signs.blood_pressure_systolic": ("blood_pressure", r"^\s*(\d+(?:\.\d+)?)\s*/", "mmHg", (40, 300)),
    "vital_signs.blood_pressure_diastolic": ("blood_pressure", r"/\s*(\d+(?:\.\d+)?)", "mmHg", (20, 200)),
    "vital_signs.heart_rate": ("heart_rate", r"(\d+(?:\.\d+)?)", "bpm", (20, 300)),
    "vital_signs.respiration_rate": ("respiration_rate", r"(\d+(?:\.\d+)?)", "/min", (4, 80)),
    "vital_signs.temperature": ("temperature", r"(\d+(?:\.\d+)?)", "°F", (80, 115)),
    "vital_signs.sp02": ("sp02", r"(\d+(?:\.\d+)?)", "%", (50, 100)),
}

DESCRIPTIONS = {
    "record_id": ("identifier", "Unique record identifier, for example ABC-2030-0042."),
    "department": ("categorical", "Clinical department that handled the encounter."),
    "patient_name": ("name", "Full name of the patient."),
    "dob": ("date", "Patient's date of birth."),
    "age": ("integer", "Patient's age in whole years at the encounter."),
    "sex": ("categorical", "Patient's sex."),
    "insurance": ("categorical", "Patient's health insurance."),
    "city": ("categorical", "City and state where the patient lives, written 'City, State'."),
    "encounter_date": ("date", "Date of the medical encounter."),
    "chief_complaint": ("free_text", "The reason the patient came, in the patient's terms."),
    "history_of_present_illness": ("free_text", "Narrative of the current illness."),
    "past_medical_history": ("free_text", "Earlier conditions of the patient."),
    "medications": ("free_text", "Medications the patient takes."),
    "allergies": ("free_text", "Known allergies."),
    "physical_exam": ("free_text", "Findings of the physical examination."),
    "imaging_results": ("free_text", "Imaging and ECG findings."),
    "final_diagnosis": ("free_text", "The diagnosis the clinician concluded."),
    "plan": ("free_text", "Treatment and follow-up plan."),
    "progress_notes": ("free_text", "Notes on the evolution of the patient."),
    "icd10_code": ("icd10_hierarchy", "ICD-10 code of the diagnosis (three characters, for example I10 or J18)."),
    "icd10_chapter": ("icd10_hierarchy", "ICD-10 chapter of the diagnosis."),
    "icd10_category": ("icd10_hierarchy", "ICD-10 category (block) of the diagnosis."),
    LAB: ("list_of_objects", "Laboratory results: a list of tests, each with a name, a result and a flag. Filter with op 'has'."),
    LAB_NAME: ("categorical", "Name of a laboratory test. Some tests have variants, for example 'Troponin I' and 'Troponin I (initial)'."),
    LAB_FLAG: ("categorical", "How a laboratory result was flagged against its reference range."),
    "vital_signs.blood_pressure_systolic": ("measurement", "Systolic blood pressure."),
    "vital_signs.blood_pressure_diastolic": ("measurement", "Diastolic blood pressure."),
    "vital_signs.heart_rate": ("measurement", "Heart rate."),
    "vital_signs.respiration_rate": ("measurement", "Respiration rate."),
    "vital_signs.temperature": ("measurement", "Body temperature, stored in degrees Fahrenheit (convert Celsius)."),
    "vital_signs.sp02": ("measurement", "Oxygen saturation (SpO2)."),
}
FLAG_MEANINGS = {"Normal": "within the reference range", "High": "above the reference range",
                 "Low": "below the reference range", "Borderline": "close to a limit of the reference range"}


@dataclass
class Column:
    role: str
    description: str = ""
    values: dict[str, int] | None = None      # categorical: value -> number of records (None = too many to list)
    distinct: int | None = None
    unit: str | None = None
    range: tuple[Any, Any] | None = None      # observed (min, max) of a date / integer / measurement
    example: str | None = None

    def to_yaml(self) -> dict:
        out: dict[str, Any] = {"role": self.role, "description": self.description}
        for key in ("unit", "example"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.range:
            out["range"] = [str(v) if isinstance(v, (date,)) else v for v in self.range]
        if self.values is not None:
            out["values"] = dict(self.values)
        elif self.distinct:
            out["distinct_values"] = self.distinct
        return out


@dataclass
class Catalog:
    columns: dict[str, Column] = field(default_factory=dict)
    total_records: int = 0
    generated: str = ""

    # ------------------------------------------------------------------ access
    def __contains__(self, name: str) -> bool:
        return name in self.columns

    def role(self, name: str) -> str:
        return self.columns[name].role

    def ops(self, name: str) -> tuple[str, ...]:
        return OPS_BY_ROLE[self.columns[name].role]

    def values(self, name: str) -> dict[str, int] | None:
        return self.columns[name].values

    def filter_fields(self) -> list[str]:
        """The names a filter may use: the columns, the vital signs, and lab_results (but not its parts)."""
        return [n for n in self.columns if n not in (LAB_NAME, LAB_FLAG)]

    def date_span(self, name: str = "encounter_date") -> tuple[date, date] | None:
        col = self.columns.get(name)
        if col and col.range:
            return date.fromisoformat(str(col.range[0])[:10]), date.fromisoformat(str(col.range[1])[:10])
        return None

    def canonical(self, name: str, text: str) -> str | None:
        """The catalog's spelling of `text` for column `name` (ignoring case and surrounding spaces), or None."""
        values = self.values(name)
        if values is None:
            return None
        wanted = str(text).strip().casefold()
        return next((v for v in values if v.casefold() == wanted), None)

    def suggest(self, name: str, text: str, n: int = 3) -> list[str]:
        """Values of the column that look like `text`: similar spelling, or containing it as a word."""
        values = list(self.values(name) or [])
        wanted = str(text).strip().casefold()
        close = difflib.get_close_matches(wanted, [v.casefold() for v in values], n=n, cutoff=0.6)
        found = [next(v for v in values if v.casefold() == c) for c in close]
        if wanted:
            found += [v for v in values if v not in found and re.search(rf"\b{re.escape(wanted)}", v.casefold())]
        return found[:n]

    def expand_test_names(self, text: str) -> list[str]:
        """Lab test names that `text` stands for: the exact name and its variants ('Glucose' -> 'Glucose (point-of-care)'), or every name containing it ('Troponin' -> both variants)."""
        exact = self.canonical(LAB_NAME, text)
        names = self.values(LAB_NAME) or {}
        if exact:  # 'Glucose' also stands for 'Glucose (point-of-care)': names that continue it after a separator
            base = exact.casefold()
            variants = [v for v in names if v != exact and v.casefold().startswith(base) and not v[len(base):len(base) + 1].isalnum()]
            return [exact, *variants]
        wanted = str(text).strip().casefold()
        return [v for v in names if wanted and wanted in v.casefold()]

    def mentions(self, question: str) -> dict[str, list[tuple[str, str]]]:
        """Words of the question that name a value in the catalog (a lab test, a department, a city...): {word: [(field, value), ...]}.
        Words of 5+ letters only, compared with the words of each value ('creatinine' names 'Creatinine'), so a short or
        common word does not match by accident. Used to check that a plan did not drop something the user said."""
        words = {w for w in re.findall(r"[a-z]{5,}", question.lower())} - MENTION_STOP
        found: dict[str, list[tuple[str, str]]] = {}
        for name, col in self.columns.items():
            for value in col.values or {}:
                tokens = re.findall(r"[a-z]{5,}", str(value).lower())
                for w in words:
                    if any(tok == w or tok.startswith(w) or w.startswith(tok) for tok in tokens):
                        found.setdefault(w, []).append((name, str(value)))
        return found

    # ------------------------------------------------------------------ for the model
    def prompt_text(self, question: str = "", max_values: int = 50) -> str:
        """The catalog as the planner reads it. A column with more than `max_values` values lists the most frequent ones and
        those most like the question, so the prompt stays small."""
        words = set(re.findall(r"[a-z0-9]+", question.lower()))
        lines = []
        for name, col in self.columns.items():
            if name in (LAB_NAME, LAB_FLAG):
                continue
            head = f"- {name} [{col.role}; ops: {', '.join(OPS_BY_ROLE[col.role])}]: {col.description}"
            if col.unit:
                head += f" Unit: {col.unit}."
            if col.range and col.role in NUMERIC | {"date"}:
                head += f" Data range: {col.range[0]} to {col.range[1]}."
            lines.append(head)
            if name == LAB:
                for sub, label in ((LAB_NAME, "test_name"), (LAB_FLAG, "flag")):
                    lines.append(f"    {label} values: {self._values_text(sub, words, max_values)}")
                lines.append("    flag meanings: " + "; ".join(f"{k} = {v}" for k, v in FLAG_MEANINGS.items()))
            elif col.values is not None:
                lines.append(f"    values: {self._values_text(name, words, max_values)}")
        return "\n".join(lines)

    def _values_text(self, name: str, words: set[str], limit: int) -> str:
        values = self.columns[name].values or {}
        items = list(values.items())
        if len(items) > limit:
            like = [kv for kv in items if words & set(re.findall(r"[a-z0-9]+", kv[0].lower()))]
            items = (like + [kv for kv in items if kv not in like])[:limit]
        return "; ".join(f"{v} ({n})" for v, n in items)

    # ------------------------------------------------------------------ YAML
    def to_yaml(self, path: str | Path | None = None) -> str:
        data = {"table": TABLE, "generated": self.generated, "total_records": self.total_records,
                "columns": {n: c.to_yaml() for n, c in self.columns.items()}}
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)
        if path:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, source: str | Path) -> "Catalog":
        """Load a catalog from a file path or from YAML text."""
        text = Path(source).read_text(encoding="utf-8") if "\n" not in str(source) and Path(str(source)).exists() else str(source)
        data = yaml.safe_load(text) or {}
        cols = {}
        for name, c in (data.get("columns") or {}).items():
            if c.get("role") not in ROLES:
                raise ValueError(f"catalog column {name!r}: unknown role {c.get('role')!r} (use one of {', '.join(ROLES)})")
            rng = tuple(c["range"]) if c.get("range") else None
            cols[name] = Column(c["role"], c.get("description", ""), {str(k): int(v) for k, v in c["values"].items()} if c.get("values") is not None else None,
                                c.get("distinct_values"), c.get("unit"), rng, c.get("example"))
        return cls(cols, int(data.get("total_records") or 0), str(data.get("generated") or ""))

    # ------------------------------------------------------------------ the profiler
    @classmethod
    def build(cls, con: Any, *, previous: "Catalog | None" = None, llm: Any = None, today: date | None = None) -> "Catalog":
        """Profile the database. `previous` supplies the descriptions already written (a person's edits survive); `llm`, if given,
        drafts a description for a column nobody described (review it before using it)."""
        total = con.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
        cat = cls({}, total, str(today or date.today()))
        limit = s.CATALOG_MAX_VALUES

        def described(name: str, role: str, fallback: str = "") -> tuple[str, str]:
            if previous and name in previous.columns and previous.columns[name].description:
                return previous.columns[name].role, previous.columns[name].description
            if name in DESCRIPTIONS:
                return DESCRIPTIONS[name]
            return role, fallback

        def counted(sql: str) -> tuple[dict[str, int] | None, int]:
            rows = con.execute(sql).fetchall()
            return (None, len(rows)) if len(rows) > limit else ({str(k): int(n) for k, n in rows}, len(rows))

        info = con.execute(f"PRAGMA table_info('{TABLE}')").fetchall()  # what the file really has (an old one lacks the ICD-10 columns)
        present = {r[1] for r in info}
        for _, name, sql_type, *_rest in info:
            if name in ("vital_signs", "lab_results"):
                continue
            guess = "categorical" if sql_type.startswith("VARCHAR") else "date" if sql_type.startswith("TIMESTAMP") else "integer"
            if sql_type.startswith("VARCHAR") and name not in DESCRIPTIONS:
                guess = "categorical" if con.execute(f"SELECT count(DISTINCT {name}) FROM {TABLE}").fetchone()[0] <= limit else "free_text"
            role, desc = described(name, guess)
            col = Column(role, desc)
            if role in ("categorical", "icd10_hierarchy"):
                col.values, col.distinct = counted(f"SELECT {name}, count(*) n FROM {TABLE} WHERE {name} IS NOT NULL AND {name} <> '' "
                                                   f"GROUP BY 1 ORDER BY n DESC, 1")
            elif role == "date":
                lo, hi = con.execute(f"SELECT min({name}), max({name}) FROM {TABLE}").fetchone()
                col.range = (lo.date().isoformat(), hi.date().isoformat()) if lo else None
            elif role == "integer":
                col.range = con.execute(f"SELECT min({name}), max({name}) FROM {TABLE}").fetchone()
                col.range = tuple(col.range) if col.range[0] is not None else None
            elif role == "identifier":
                row = con.execute(f"SELECT {name} FROM {TABLE} ORDER BY 1 LIMIT 1").fetchone()
                col.example = row[0] if row else None
            cat.columns[name] = col

        for name, (key, regex, unit, _) in (VITALS.items() if "vital_signs" in present else ()):
            role, desc = described(name, "measurement")
            col = Column(role, desc, unit=unit)
            lo, hi = con.execute(
                f"SELECT min(v), max(v) FROM (SELECT TRY_CAST(NULLIF(regexp_extract(json_extract_string(vital_signs, '$.{key}'), '{regex}', 1), '') AS DOUBLE) v FROM {TABLE})").fetchone()
            col.range = (lo, hi) if lo is not None else None
            cat.columns[name] = col

        if "lab_results" not in present:
            return cat._describe_missing(llm)
        role, desc = described(LAB, "list_of_objects")
        cat.columns[LAB] = Column(role, desc)
        unnest = f"(SELECT unnest(from_json(lab_results, '[\"JSON\"]')) AS x FROM {TABLE} WHERE lab_results IS NOT NULL)"
        for name, key in ((LAB_NAME, "test_name"), (LAB_FLAG, "flag")):
            role, desc = described(name, "categorical")
            col = Column(role, desc)
            col.values, col.distinct = counted(f"SELECT json_extract_string(x, '$.{key}') k, count(*) n FROM {unnest} GROUP BY 1 HAVING k IS NOT NULL ORDER BY n DESC, 1")
            cat.columns[name] = col

        return cat._describe_missing(llm)

    def _describe_missing(self, llm: Any) -> "Catalog":
        if llm is not None:
            for name, col in self.columns.items():
                if not col.description:
                    col.description = _draft_description(llm, name, col)
        return self

    def refresh(self, con: Any, *, today: date | None = None) -> "Catalog":
        """The same catalog with its facts (values, counts, ranges) recomputed from the database; descriptions are kept."""
        return Catalog.build(con, previous=self, today=today)


def _draft_description(llm: Any, name: str, col: Column) -> str:
    sample = ", ".join(list((col.values or {}))[:8])
    reply = llm.run([{"role": "user", "content": f"Write one short English sentence describing the database column '{name}' "
                      f"(role: {col.role}{'; example values: ' + sample if sample else ''}). Reply with the sentence only."}],
                    temperature=0)  # no max_tokens: a model that reasons first would have none left to answer
    return reply.strip().splitlines()[0][:200] if reply.strip() else name
