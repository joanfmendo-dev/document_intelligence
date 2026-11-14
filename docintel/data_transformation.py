"""Derived variables: values computed from what a record already says. For now only the ICD-10 coding of the diagnosis.

    from docintel.data_transformation import ICD10

    ICD10.resolve("I21.9")                         # -> Coding("I21.9", "Diseases of the circulatory system", "Acute myocardial infarction")
    icd = ICD10()                                  # uses the model of docintel.settings; pass llm=... to use another
    icd.classify("Community-acquired pneumonia", "Cough and fever")   # -> Coding("J18", ...)
    icd.code_fields(get_fields(text, json_format)) # -> the same dict + icd10_code / icd10_chapter / icd10_category

    # Update an Excel: fill only the records that lack the coding, or recode every record; write it where you want (even over the source)
    icd.update_icd10("medical_records.xlsx", "medical_records_icd.xlsx", mode="missing")     # or mode="all"
    # Build the DuckDB from an Excel and update the ICD-10 on the way:  RecordDB(path).import_excel(xlsx, update_icd=True, icd_mode="all")
    # Update the records of an existing DuckDB:                         icd.update_database(db, mode="missing")

Another derived variable would be another class in this file, next to ICD10.

How the coding works: the model only proposes a code. Whether it is a real ICD-10 code, and its chapter and category, come from
the official ICD-10 ontology (simple-icd-10, bundled, no network), so the three fields never contradict each other and a made-up
code is caught. When the model cannot give a valid code the result is None (blank fields), never a placeholder such as "R69":
a blank is visible and is coded again on the next run; a wrong code is not.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import simple_icd_10 as icd

from .db import RecordDB
from .llm import LLM


class ICD10:
    FIELDS = ("icd10_code", "icd10_chapter", "icd10_category")
    MODES = ("missing", "all")  # which records update_icd10 / update_database work on: only the incomplete ones, or every one

    SYSTEM_PROMPT = """\
You are a medical coding expert. Assign the most appropriate ICD-10 code to the record.

Rules:
1. Use the 3-character format only: one uppercase letter followed by exactly two digits (e.g. J44, I21, G43).
2. Base your decision primarily on final_diagnosis, then on chief_complaint.
3. Answer ONLY with JSON: {"code": "J44"}
"""
    _CODE = re.compile(r"[A-Z]\d{2}(\.\d{1,4})?")  # a category (J44) or a subcategory (J44.9), nothing else: no blocks, no chapters
    _CHAPTER = re.compile(r"[IVXLCDM]+")
    _SCHEMA = {"type": "json_schema", "json_schema": {"name": "icd10", "strict": True, "schema": {
        "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}}}

    class Coding(NamedTuple):
        code: str
        chapter: str
        category: str

    @dataclass
    class Report:
        records: int = 0       # looked at
        pending: int = 0       # that needed work in this mode
        derived: int = 0       # of those, finished from a valid code they already had (chapter and category looked up; no model)
        coded: int = 0         # of those, coded by the model
        unresolved: int = 0    # of those, left as they were: the model gave no valid code
        model_calls: int = 0   # distinct (diagnosis, chief complaint) pairs sent to the model

        def __str__(self) -> str:
            return (f"{self.records} records | {self.pending} to update | {self.derived} completed from their code | "
                    f"{self.coded} coded by the model ({self.model_calls} model calls) | {self.unresolved} left unresolved")

    def __init__(self, llm: LLM | None = None):
        self._llm = llm

    @property
    def llm(self) -> LLM:
        """Created on first use, so that counting what is pending (dry runs) never needs a model."""
        if self._llm is None:
            self._llm = LLM(self.SYSTEM_PROMPT)
        return self._llm

    # ------------------------------------------------------------ the ontology (no model)
    @staticmethod
    def normalize(code: object) -> str:
        """What a person may type ('i21.9', ' I219 ') as ICD-10 text with the dot: 'I21.9'. Not validated."""
        text = re.sub(r"\s+", "", str(code or "")).upper()
        return f"{text[:3]}.{text[3:]}" if len(text) > 3 and "." not in text else text

    @staticmethod
    def resolve(code: object, *, prefix_fallback: bool = False) -> "ICD10.Coding | None":
        """The code with its chapter and category descriptions, or None if it is not an ICD-10 code.
        With prefix_fallback, an unknown subcategory (J44.99) falls back to its category (J44)."""
        code = ICD10.normalize(code)
        if not ICD10._CODE.fullmatch(code) or not icd.is_valid_item(code):
            if not (prefix_fallback and ICD10._CODE.fullmatch(code) and icd.is_valid_item(code[:3])):
                return None
            code = code[:3]
        category = icd.get_description(code[:3])
        chapter = next((icd.get_description(a) for a in icd.get_ancestors(code) if ICD10._CHAPTER.fullmatch(a)), "")
        return ICD10.Coding(code, chapter, category)

    @staticmethod
    def is_complete(row: dict) -> bool:
        """The three fields are filled, the code is a real one and the chapter and category are the ontology's."""
        coding = ICD10.resolve(row.get("icd10_code"))
        return bool(coding and row.get("icd10_code") == coding.code and row.get("icd10_chapter") == coding.chapter
                    and row.get("icd10_category") == coding.category)

    # ------------------------------------------------------------ the model
    def classify(self, final_diagnosis: str | None, chief_complaint: str | None = "") -> "ICD10.Coding | None":
        """Ask the model for the ICD-10 code of a diagnosis (and chief complaint) and resolve it.
        None if there is nothing to code or the model does not return a valid code, even after one correction.
        An unreachable model raises docintel.llm.LLMError."""
        diagnosis, complaint = (final_diagnosis or "").strip(), (chief_complaint or "").strip()
        if not diagnosis and not complaint:
            return None
        messages = [{"role": "user", "content": f"final_diagnosis: {diagnosis}\nchief_complaint: {complaint}"}]
        for _ in range(2):
            answer = self.llm.run(messages, temperature=0, max_tokens=60, response_format=self._SCHEMA)
            proposed = self._proposed(answer)
            if coding := self.resolve(proposed, prefix_fallback=True):
                return coding
            messages += [{"role": "assistant", "content": answer},
                         {"role": "user", "content": f"{proposed or answer.strip()[:40]!r} is not a valid ICD-10 code. "
                                                     'Answer again with a valid 3-character code: {"code": "J44"}'}]
        return None

    @staticmethod
    def _proposed(answer: str) -> str:
        """The code in a model answer: the 'code' of a JSON object, or the first thing shaped like a code."""
        try:
            data = json.loads(answer)
            if isinstance(data, dict) and isinstance(data.get("code"), str):
                return data["code"]
        except json.JSONDecodeError:
            pass
        match = re.search(r"\b[A-Za-z]\d{2}(?:\.\d{1,4})?\b", answer)
        return match.group(0) if match else ""

    def code_fields(self, fields: dict) -> dict:
        """A copy of a get_fields result with the three ICD-10 keys filled from its final_diagnosis and chief_complaint
        (None when it cannot be coded)."""
        coding = self.classify(fields.get("final_diagnosis"), fields.get("chief_complaint"))
        return {**fields, **dict(zip(self.FIELDS, coding or (None, None, None)))}

    # ------------------------------------------------------------ updating many records
    def update_rows(self, rows: list[dict], mode: str = "missing", *, dry_run: bool = False,
                    on_update: Callable[[dict], None] | None = None, progress: Callable[[str], None] | None = None) -> "ICD10.Report":
        """Update `rows` in place: dicts with final_diagnosis, chief_complaint and the three ICD-10 keys (what the other
        methods are built on). `on_update(row)` is called as soon as a row changes, so a caller can save as it goes.

        mode="missing": a row is pending when its coding is not complete (see is_complete). If its code is a real one, chapter and
        category are simply looked up from it; otherwise (no code, or one that is not in ICD-10) the model is asked.
        mode="all": every row is sent to the model and its coding replaced. A row the model cannot code is left as it was.
        Rows with the same diagnosis and chief complaint share one model call. With dry_run nothing changes and no model is used."""
        self._check_mode(mode)
        report = ICD10.Report(records=len(rows))
        todo = rows if mode == "all" else [r for r in rows if not self.is_complete(r)]
        report.pending = len(todo)
        if dry_run:
            report.derived = sum(1 for r in todo if mode == "missing" and self.resolve(r.get("icd10_code")))
            return report
        answers: dict[tuple[str, str], ICD10.Coding | None] = {}
        for done, row in enumerate(todo, 1):
            known = self.resolve(row.get("icd10_code")) if mode == "missing" else None
            if known:
                coding, report.derived = known, report.derived + 1
            else:
                key = ((row.get("final_diagnosis") or "").strip(), (row.get("chief_complaint") or "").strip())
                if key not in answers:
                    answers[key] = self.classify(*key)
                    report.model_calls += bool(key[0] or key[1])
                coding = answers[key]
            if coding:
                row.update(zip(self.FIELDS, coding))
                report.coded += not known
                if on_update:
                    on_update(row)
            else:
                report.unresolved += 1
            if progress and (done % 25 == 0 or done == len(todo)):
                progress(f"  {done}/{len(todo)} records processed")
        return report

    def _check_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, not {mode!r}")

    # ------------------------------------------------------------ an Excel
    def update_icd10(self, source: str | Path, output: str | Path, *, mode: str = "missing", sheet: str | None = None,
                     dry_run: bool = False, progress: Callable[[str], None] | None = None) -> "ICD10.Report":
        """Fill in the ICD-10 columns of an Excel sheet (default: the first) and save the result as `output`, which may be
        the same file as `source`. The three columns are added at the end if the Excel does not have them; nothing else
        in the file changes. See update_rows for what `mode` ("missing" or "all") does.

        The Excel needs a header row with final_diagnosis (and, optionally, chief_complaint). If the model fails halfway (LLMError),
        what was already coded is saved before the error is raised, so a run with mode="missing" continues where it stopped.
        With dry_run nothing is written and no model is used: the report says how many records would be updated."""
        self._check_mode(mode)
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise ImportError("update_icd10 needs openpyxl: pip install openpyxl") from exc
        book = load_workbook(source)
        try:
            sheet_ = book[sheet] if sheet else book.worksheets[0]
            header = [c.value for c in sheet_[1]]
            if "final_diagnosis" not in header and "chief_complaint" not in header:
                raise ValueError("the Excel needs a header row with a final_diagnosis column (and optionally chief_complaint)")
            at = {name: header.index(name) + 1 for name in ("final_diagnosis", "chief_complaint", *self.FIELDS) if name in header}
            for name in self.FIELDS:  # a column the Excel does not have yet goes at the end
                if name not in at:
                    at[name] = len(header) + 1
                    header.append(name)
                    if not dry_run:
                        sheet_.cell(row=1, column=at[name], value=name)
            rows, lines = [], []
            for number, cells in enumerate(sheet_.iter_rows(min_row=2, values_only=True), start=2):
                if any(v not in (None, "") for v in cells):
                    get = lambda name: cells[at[name] - 1] if name in at and at[name] <= len(cells) else None
                    rows.append({"final_diagnosis": get("final_diagnosis"), "chief_complaint": get("chief_complaint"),
                                 # a blank, or the 0 some tools write for 'no value', is a missing value
                                 **{name: (get(name).strip() or None) if isinstance(get(name), str) else None for name in self.FIELDS}})
                    lines.append(number)
            before = [dict(r) for r in rows]
            try:
                report = self.update_rows(rows, mode, dry_run=dry_run, progress=progress)
            finally:  # also when the model fails halfway: keep what was done
                if not dry_run:
                    for number, row, old in zip(lines, rows, before):
                        if row != old:
                            for name in self.FIELDS:
                                sheet_.cell(row=number, column=at[name], value=row[name])
                    self._save(book, Path(output))
            return report
        finally:
            book.close()

    @staticmethod
    def _save(book, output: Path) -> None:
        """Save through a temporary file: a failure never leaves a half-written Excel, even when it replaces the source."""
        temporary = output.with_name(output.name + ".tmp")
        try:
            book.save(temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)

    # ------------------------------------------------------------ a DuckDB
    def update_database(self, db: RecordDB, *, mode: str = "missing", dry_run: bool = False,
                        progress: Callable[[str], None] | None = None) -> "ICD10.Report":
        """The same for the records already in a RecordDB. Each record is saved as soon as it is coded, so an interrupted run keeps
        its work. Only the three ICD-10 fields are written."""
        def save(row: dict) -> None:
            db.update_record({"record_id": row["record_id"], **{name: row[name] for name in self.FIELDS}}, keep_existing=False)
        return self.update_rows(db.icd10_rows(), mode, dry_run=dry_run, on_update=save, progress=progress)

