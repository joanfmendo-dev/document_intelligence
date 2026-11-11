import json

import pytest
from openpyxl import Workbook

from docintel import RecordDB, RecordExistsError, RecordNotFoundError, get_fields
from docintel.db import COLUMNS

RECORD = {  # what get_fields returns for a full document
    "record_id": "SYNUS-2026-0001", "department": "Cardiology", "patient_name": "Jennifer Parker",
    "dob": "1961-01-28T00:00:00", "age": 63, "sex": "Female", "insurance": "Aetna PPO",
    "city": "Miami, Florida", "encounter_date": "2025-01-01T09:19:07",
    "chief_complaint": "Chest pressure and exertional dyspnea.",
    "history_of_present_illness": "5 days of substernal pressure.", "past_medical_history": "Hypertension.",
    "medications": "Atorvastatin.", "allergies": "NKDA.",
    "vital_signs": {"blood_pressure": "158/95 mmHg", "heart_rate": "101 bpm", "respiration_rate": "23/min",
                    "temperature": "99.4 °F", "sp02": "98%"},
    "physical_exam": "Mild distress.",
    "lab_results": [{"test_name": "Troponin I", "result": "0.042 ng/mL", "flag": "Borderline"},
                    {"test_name": "WBC Count", "result": "5.6 K/uL", "flag": "Normal"}],
    "imaging_results": "ECG with nonspecific ST-T changes.", "final_diagnosis": "Possible NSTEMI.",
    "plan": "Telemetry admission.", "progress_notes": "Ongoing surveillance.",
}
EMPTY = {c: None for c in COLUMNS} | {"vital_signs": {k: None for k in RECORD["vital_signs"]}, "lab_results": []}


@pytest.fixture
def db():
    with RecordDB(":memory:") as d:
        yield d


def test_table_has_the_excel_columns_in_order(db):
    assert [r[0] for r in db.con.execute("DESCRIBE medical_records").fetchall()] == list(COLUMNS) == list(RECORD)


def test_create_then_read_returns_the_same_dict(db):
    assert db.create_record(RECORD) == RECORD
    assert db.read_record("SYNUS-2026-0001") == RECORD
    assert list(db.read_record("SYNUS-2026-0001")) == list(RECORD)      # same key order too


def test_get_fields_result_can_be_stored_and_read_back(db, ollama):
    template = {**RECORD, "dob": "1961-01-28T00:00:00"}
    ollama.reply(json.dumps({**RECORD, "medications": None, "vital_signs": {"heart_rate": "101 bpm"}, "lab_results": []}))
    fields = get_fields(json.dumps(RECORD), template)                   # the text contains the values: no "invented" warning
    assert fields["vital_signs"]["blood_pressure"] is None and fields["medications"] is None
    db.create_record(fields)
    assert db.read_record(fields["record_id"]) == fields


def test_a_document_with_missing_values_is_stored_with_nulls(db):
    db.create_record({**EMPTY, "record_id": "X-1", "patient_name": "Ana"})
    assert db.read_record("X-1") == {**EMPTY, "record_id": "X-1", "patient_name": "Ana"}
    db.create_record({"record_id": "X-2"})                              # keys left out are NULL
    assert db.read_record("X-2")["vital_signs"] is None and db.read_record("X-2")["dob"] is None


def test_create_rejects_bad_input_and_duplicates(db):
    db.create_record(RECORD)
    with pytest.raises(RecordExistsError):
        db.create_record(RECORD)
    with pytest.raises(ValueError, match="record_id"):
        db.create_record({**RECORD, "record_id": None})
    with pytest.raises(ValueError, match="facility"):                  # a template with another key must not be silently dropped
        db.create_record({**RECORD, "facility": "Cardiology"})
    with pytest.raises(TypeError):
        db.create_record(["not", "a", "dict"])
    assert db.con.execute("SELECT count(*) FROM medical_records").fetchone()[0] == 1


def test_read_missing_record_is_none(db):
    assert db.read_record("NOPE") is None


def test_update_changes_only_what_the_new_document_gives(db):
    db.create_record(RECORD)
    new = {**EMPTY, "record_id": RECORD["record_id"], "encounter_date": "2025-01-01T10:00:00",
           "age": 64, "vital_signs": {**EMPTY["vital_signs"], "heart_rate": "88 bpm"}}
    got = db.update_record(new)
    assert got == {**RECORD, "encounter_date": "2025-01-01T10:00:00", "age": 64,
                   "vital_signs": {**EMPTY["vital_signs"], "heart_rate": "88 bpm"}}   # whole JSON object replaced
    assert got["patient_name"] == "Jennifer Parker" and got["lab_results"] == RECORD["lab_results"]   # not erased
    assert db.read_record(RECORD["record_id"]) == got


def test_update_with_keep_existing_false_clears_values(db):
    db.create_record(RECORD)
    got = db.update_record({**EMPTY, "record_id": RECORD["record_id"], "city": "Austin, Texas"}, keep_existing=False)
    assert got == {**EMPTY, "record_id": RECORD["record_id"], "city": "Austin, Texas"}


def test_update_only_touches_the_keys_it_receives(db):
    db.create_record(RECORD)
    got = db.update_record({"record_id": RECORD["record_id"], "plan": None}, keep_existing=False)
    assert got == {**RECORD, "plan": None}


def test_update_missing_record_or_nothing_to_change(db):
    with pytest.raises(RecordNotFoundError):
        db.update_record({"record_id": "NOPE", "age": 1})
    with pytest.raises(RecordNotFoundError):                            # nothing to write still reports a missing record
        db.update_record({**EMPTY, "record_id": "NOPE"})
    db.create_record(RECORD)
    assert db.update_record({**EMPTY, "record_id": RECORD["record_id"]}) == RECORD


def test_delete(db):
    db.create_record(RECORD)
    assert db.delete_record(RECORD["record_id"]) is True
    assert db.read_record(RECORD["record_id"]) is None
    assert db.delete_record(RECORD["record_id"]) is False
    db.create_record(RECORD)                                            # the id can be used again
    assert db.read_record(RECORD["record_id"]) == RECORD


def test_data_survives_closing_and_reopening(tmp_path):
    path = tmp_path / "records.duckdb"
    with RecordDB(path) as d:
        d.create_record(RECORD)
    with RecordDB(path) as d:
        assert d.read_record(RECORD["record_id"]) == RECORD


# ---------------------------------------------------------------- Excel import
def make_xlsx(path, rows, header=None, sheet="data"):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(header or list(COLUMNS))
    for r in rows:
        ws.append(r)
    wb.save(path)


def excel_row(record):
    """A record as the Excel stores it: datetimes as cells, the two JSON columns as text."""
    from datetime import datetime
    return [datetime.fromisoformat(v) if c in ("dob", "encounter_date") else
            json.dumps(v) if c in ("vital_signs", "lab_results") else v for c, v in record.items()]


def test_import_excel_matches_the_rows(db, tmp_path):
    other = {**RECORD, "record_id": "SYNUS-2026-0002", "age": 76, "lab_results": []}
    make_xlsx(tmp_path / "m.xlsx", [excel_row(RECORD), excel_row(other)])
    assert db.import_excel(tmp_path / "m.xlsx") == 2
    assert db.read_record("SYNUS-2026-0001") == RECORD
    assert db.read_record("SYNUS-2026-0002") == other


def test_import_excel_is_all_or_nothing(db, tmp_path):
    make_xlsx(tmp_path / "dup.xlsx", [excel_row(RECORD), excel_row({**RECORD, "age": 1})])
    with pytest.raises(Exception):
        db.import_excel(tmp_path / "dup.xlsx")
    assert db.con.execute("SELECT count(*) FROM medical_records").fetchone()[0] == 0
    bad_header = [c if c != "department" else "facility" for c in COLUMNS]
    make_xlsx(tmp_path / "hdr.xlsx", [], header=bad_header)
    with pytest.raises(ValueError, match="department"):
        db.import_excel(tmp_path / "hdr.xlsx")


# ---------------------------------------------------------------- listing
def seed(db, n=30):
    for i in range(n):
        db.create_record({**RECORD, "record_id": f"SYNUS-2026-{i:04d}", "patient_name": f"Patient {i:02d}",
                          "department": "Cardiology" if i % 3 else "Oncology",
                          "encounter_date": f"2026-01-{i + 1:02d}T09:00:00", "final_diagnosis": f"Dx {i}"})


def test_list_records_pages_newest_first(db):
    seed(db)
    rows, total = db.list_records(limit=10)
    assert total == 30 and [r["record_id"] for r in rows][:2] == ["SYNUS-2026-0029", "SYNUS-2026-0028"]
    assert list(rows[0]) == ["record_id", "patient_name", "department", "encounter_date", "final_diagnosis"]
    assert rows[0]["encounter_date"] == "2026-01-30T09:00:00"
    last, _ = db.list_records(limit=10, offset=20)
    assert [r["record_id"] for r in last][-1] == "SYNUS-2026-0000" and len(last) == 10
    assert db.list_records(limit=10, offset=30) == ([], 30)


def test_list_records_filters(db):
    seed(db)
    assert db.list_records(search="patient 07")[1] == 1
    assert db.list_records(search="2026-0012")[0][0]["record_id"] == "SYNUS-2026-0012"      # by id, any case
    assert db.list_records(search="synus-2026-001")[1] == 10
    assert db.list_records(department="Oncology")[1] == 10
    assert db.list_records(search="patient", department="Oncology")[1] == 10
    assert db.list_records(search="patient 07", department="Oncology")[1] == 0               # 7 % 3 != 0
    assert db.list_records(search="nobody") == ([], 0)


def test_list_records_search_is_literal(db):
    seed(db, 3)
    db.create_record({"record_id": "X-1", "patient_name": "100% sure_1"})
    assert [r["record_id"] for r in db.list_records(search="100%")[0]] == ["X-1"]
    assert [r["record_id"] for r in db.list_records(search="_1")[0]] == ["X-1"]              # '_' is not a wildcard
    assert db.list_records(search="%")[1] == 1 and db.list_records(search="\\")[1] == 0


def test_list_records_nulls_and_empty(db):
    assert db.list_records() == ([], 0)
    db.create_record({"record_id": "A-1"})
    db.create_record({"record_id": "A-2", "encounter_date": "2026-01-01T00:00:00"})
    assert [r["record_id"] for r in db.list_records()[0]] == ["A-2", "A-1"]                  # no date goes last
    assert db.list_records()[0][1]["patient_name"] is None


def test_departments(db):
    assert db.departments() == []
    seed(db, 4)
    db.create_record({"record_id": "NO-DEPT"})
    db.create_record({"record_id": "BLANK", "department": ""})
    assert db.departments() == ["Cardiology", "Oncology"]
