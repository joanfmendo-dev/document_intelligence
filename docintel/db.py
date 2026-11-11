"""RecordDB: keep the dictionaries that get_fields returns in a local DuckDB file.

    db = RecordDB("medical_records.duckdb")
    db.create_record(get_fields(text, json_format))
    db.read_record("SYNUS-2026-0571")      # -> a dict shaped exactly like get_fields' result

The table has the same 21 columns, in the same order, as medical_records.xlsx. Dates are TIMESTAMP,
`age` is INTEGER, `vital_signs` and `lab_results` are JSON (the Excel stores them as JSON text too),
everything else is VARCHAR. Only `record_id` is required (it is the primary key): the other columns
accept NULL, because a document may not give every field.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from .settings import s

TABLE = "medical_records"
COLUMNS = {  # name -> DuckDB type; the order is the Excel's
    "record_id": "VARCHAR PRIMARY KEY", "department": "VARCHAR", "patient_name": "VARCHAR",
    "dob": "TIMESTAMP", "age": "INTEGER", "sex": "VARCHAR", "insurance": "VARCHAR", "city": "VARCHAR",
    "encounter_date": "TIMESTAMP", "chief_complaint": "VARCHAR", "history_of_present_illness": "VARCHAR",
    "past_medical_history": "VARCHAR", "medications": "VARCHAR", "allergies": "VARCHAR",
    "vital_signs": "JSON", "physical_exam": "VARCHAR", "lab_results": "JSON", "imaging_results": "VARCHAR",
    "final_diagnosis": "VARCHAR", "plan": "VARCHAR", "progress_notes": "VARCHAR",
}
_DATES = {"dob", "encounter_date"}
_JSON = {"vital_signs", "lab_results"}


class RecordExistsError(ValueError):
    """create_record: a record with that record_id is already stored."""


class RecordNotFoundError(KeyError):
    """update_record: there is no record with that record_id."""


def _placeholder(column: str) -> str:
    return "CAST(? AS JSON)" if column in _JSON else "?"


def _to_db(column: str, value: Any) -> Any:
    """A get_fields value as DuckDB wants it: datetimes parsed from ISO text, JSON columns dumped to text."""
    if value is None:
        return None
    if column in _DATES:
        return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if column in _JSON:
        return json.dumps(value, ensure_ascii=False)
    return value


def _from_db(column: str, value: Any) -> Any:
    """A stored value in get_fields' format: ISO 8601 text for dates, parsed objects for JSON columns."""
    if value is None:
        return None
    if column in _DATES:
        return value.isoformat()
    if column in _JSON:
        return json.loads(value)
    return value


def _is_absent(value: Any) -> bool:
    """What get_fields returns for 'the document does not say': None, [], or an object whose values are all None."""
    return value is None or value == [] or (isinstance(value, dict) and all(v is None for v in value.values()))


def _check(fields: Any) -> None:
    if not isinstance(fields, dict):
        raise TypeError("fields must be a dict, the result of get_fields")
    unknown = sorted(set(fields) - set(COLUMNS))
    if unknown:
        raise ValueError(f"these keys are not columns of {TABLE}: {unknown}")
    if not fields.get("record_id"):
        raise ValueError("record_id is required: it identifies the record")


class RecordDB:
    def __init__(self, path: str | Path | None = None):
        """Open (and create, if needed) the DuckDB file at `path` (default: DOCINTEL_DB_PATH).
        Use ":memory:" for a throwaway database. DuckDB lets one process open the file for writing at a time."""
        self.path = str(path or s.DB_PATH)
        self.con = duckdb.connect(self.path)
        self.con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ({', '.join(f'{c} {t}' for c, t in COLUMNS.items())})")

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "RecordDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ create
    def _insert_sql(self) -> str:
        return f"INSERT INTO {TABLE} VALUES ({', '.join(_placeholder(c) for c in COLUMNS)})"

    def _insert_params(self, fields: dict) -> list:
        return [_to_db(c, fields.get(c)) for c in COLUMNS]

    def create_record(self, fields: dict) -> dict:
        """Store the result of get_fields as a new record and return it as stored.
        Keys missing from `fields` are stored as NULL. Raises RecordExistsError if the record_id is taken."""
        _check(fields)
        try:
            self.con.execute(self._insert_sql(), self._insert_params(fields))
        except duckdb.ConstraintException as exc:
            raise RecordExistsError(f"record {fields['record_id']!r} already exists; use update_record") from exc
        return self.read_record(fields["record_id"])

    # ------------------------------------------------------------ read
    def read_record(self, record_id: str) -> dict | None:
        """The record in the same format get_fields returns, or None if there is no such record_id."""
        row = self.con.execute(f"SELECT {', '.join(COLUMNS)} FROM {TABLE} WHERE record_id = ?", [record_id]).fetchone()
        return None if row is None else {c: _from_db(c, v) for c, v in zip(COLUMNS, row)}

    def list_records(self, *, search: str = "", department: str = "", limit: int = 25, offset: int = 0) -> tuple[list[dict], int]:
        """One page of summaries for a listing, and how many records match in total.

        `search` matches the record_id or the patient name (any part, any case); `department` must match exactly.
        Newest encounters first. Each summary has record_id, patient_name, department, encounter_date (ISO text,
        like read_record) and final_diagnosis; use read_record for everything else."""
        where, params = [], []
        if search.strip():
            like = "%" + search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            where.append("(record_id ILIKE ? ESCAPE '\\' OR patient_name ILIKE ? ESCAPE '\\')")
            params += [like, like]
        if department:
            where.append("department = ?")
            params.append(department)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        total = self.con.execute(f"SELECT count(*) FROM {TABLE}{clause}", params).fetchone()[0]
        cols = ("record_id", "patient_name", "department", "encounter_date", "final_diagnosis")
        rows = self.con.execute(
            f"SELECT {', '.join(cols)} FROM {TABLE}{clause} ORDER BY encounter_date DESC NULLS LAST, record_id LIMIT ? OFFSET ?",
            params + [max(limit, 0), max(offset, 0)]).fetchall()
        return [{c: _from_db(c, v) for c, v in zip(cols, row)} for row in rows], total

    def departments(self) -> list[str]:
        """The distinct departments stored, sorted: the choices for a filter."""
        return [r[0] for r in self.con.execute(
            f"SELECT DISTINCT department FROM {TABLE} WHERE department IS NOT NULL AND department <> '' ORDER BY 1").fetchall()]

    # ------------------------------------------------------------ update
    def update_record(self, fields: dict, *, keep_existing: bool = True) -> dict:
        """Update the record `fields["record_id"]` with the result of get_fields; return it as stored.

        Only the keys present in `fields` are touched. With keep_existing=True (default), a value that
        get_fields reports as missing (None, [], or an object with all values None) does NOT overwrite what is
        stored: a new document that does not mention a field must not erase it. With keep_existing=False,
        every key in `fields` is written exactly as given, None included, which is how you clear a value.
        A JSON column (vital_signs, lab_results) is replaced as a whole, not merged key by key.
        Raises RecordNotFoundError if the record_id does not exist."""
        _check(fields)
        record_id = fields["record_id"]
        changes = {c: v for c, v in fields.items()
                   if c != "record_id" and not (keep_existing and _is_absent(v))}
        if changes:
            sets = ", ".join(f"{c} = {_placeholder(c)}" for c in changes)
            done = self.con.execute(f"UPDATE {TABLE} SET {sets} WHERE record_id = ? RETURNING record_id",
                                    [_to_db(c, v) for c, v in changes.items()] + [record_id]).fetchall()
            found = bool(done)
        else:
            found = self.read_record(record_id) is not None
        if not found:
            raise RecordNotFoundError(f"record {record_id!r} does not exist; use create_record")
        return self.read_record(record_id)

    # ------------------------------------------------------------ delete
    def delete_record(self, record_id: str) -> bool:
        """Delete the record. Returns True if it existed, False if there was nothing to delete."""
        return bool(self.con.execute(f"DELETE FROM {TABLE} WHERE record_id = ? RETURNING record_id",
                                     [record_id]).fetchall())

    # ------------------------------------------------------------ load the Excel
    def import_excel(self, path: str | Path, sheet: str | None = None) -> int:
        """Load every row of an Excel sheet (default: the first one) into the table; returns how many.
        The header must have exactly the table's columns. All rows go in one transaction: if one fails
        (e.g. a record_id that is already stored), none is added. Needs openpyxl."""
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise ImportError("import_excel needs openpyxl: pip install openpyxl") from exc

        book = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = (book[sheet] if sheet else book.worksheets[0]).iter_rows(values_only=True)
            header = list(next(rows))
            if set(header) != set(COLUMNS):
                raise ValueError(f"the header does not match the table. Missing: {sorted(set(COLUMNS) - set(header))}; "
                                 f"extra: {sorted(set(header) - set(COLUMNS))}")
            params = []
            for row in rows:
                record = {c: (None if v == "" else v) for c, v in zip(header, row)}
                for c in _JSON & set(record):  # the Excel keeps these as JSON text
                    record[c] = json.loads(record[c]) if isinstance(record[c], str) else record[c]
                if record["record_id"]:
                    params.append(self._insert_params(record))
        finally:
            book.close()

        self.con.begin()
        try:
            self.con.executemany(self._insert_sql(), params)
        except Exception:
            self.con.rollback()
            raise
        self.con.commit()
        return len(params)
