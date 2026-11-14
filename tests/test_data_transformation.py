import json

import pytest

from datetime import datetime

from docintel import RecordDB
from docintel.data_transformation import ICD10
from docintel.llm import LLMError
from openpyxl import Workbook, load_workbook


class FakeLLM:
    """Answers with the next canned reply and records what it was asked."""
    model = "fake"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def run(self, messages, **params):
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else self.replies_default
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    replies_default = '{"code": "R69"}'


def answers_by_diagnosis(table):
    """A fake model that codes by the diagnosis in the prompt."""
    class ByDiagnosis(FakeLLM):
        def run(self, messages, **params):
            self.calls.append(messages)
            text = messages[0]["content"]
            return json.dumps({"code": next((code for dx, code in table.items() if dx in text), "R69")})
    return ByDiagnosis()


# ---------------------------------------------------------------- the ontology
def test_resolve_gives_chapter_and_category():
    assert ICD10.resolve("I21.9") == ("I21.9", "Diseases of the circulatory system", "Acute myocardial infarction")
    assert ICD10.resolve("J44") == ("J44", "Diseases of the respiratory system", "Other chronic obstructive pulmonary disease")


@pytest.mark.parametrize("typed,code", [("i21.9", "I21.9"), (" I219 ", "I21.9"), ("j 44", "J44"), ("J44.9", "J44.9")])
def test_resolve_accepts_what_people_type(typed, code):
    assert ICD10.resolve(typed).code == code


@pytest.mark.parametrize("bad", ["", None, "XYZ", "J4", "J444", "I20-I25", "IX", "N67.1", "L48", "I92.9", "A47.86", "0", 0])
def test_resolve_refuses_what_is_not_a_code(bad):               # N67.1, L48, I92.9... are codes found in a real Excel that are not ICD-10
    assert ICD10.resolve(bad) is None


def test_prefix_fallback_only_for_code_shaped_text():
    assert ICD10.resolve("J44.99", prefix_fallback=True).code == "J44"
    assert ICD10.resolve("I20-I25", prefix_fallback=True) is None and ICD10.resolve("hello", prefix_fallback=True) is None


# ---------------------------------------------------------------- the classifier
def test_classify_asks_with_diagnosis_and_complaint_and_resolves_the_answer():
    llm = FakeLLM({"code": "J18"})
    coding = ICD10(llm).classify("Community-acquired pneumonia", "Cough and fever")
    assert coding == ("J18", "Diseases of the respiratory system", "Pneumonia, organism unspecified")
    assert llm.calls[0][0]["content"] == "final_diagnosis: Community-acquired pneumonia\nchief_complaint: Cough and fever"


def test_classify_accepts_a_subcategory_and_plain_text_answers():
    assert ICD10(FakeLLM({"code": "i21.9"})).classify("x", "").code == "I21.9"
    assert ICD10(FakeLLM("The code is J44.")).classify("x", "").code == "J44"


def test_an_invalid_code_is_corrected_once_then_given_up():
    llm = FakeLLM({"code": "A47.86"}, {"code": "I80"})
    assert ICD10(llm).classify("DVT", "").code == "I80"
    assert "not a valid ICD-10 code" in llm.calls[1][-1]["content"] and len(llm.calls) == 2
    stubborn = FakeLLM({"code": "ZZZ"}, {"code": "K39.9"})
    assert ICD10(stubborn).classify("DVT", "") is None                 # blank, not a made-up placeholder
    assert len(stubborn.calls) == 2


def test_nothing_to_code_means_no_model_call():
    llm = FakeLLM()
    assert ICD10(llm).classify(None, "  ") is None and ICD10(llm).classify("", None) is None and llm.calls == []


def test_an_unreachable_model_raises():
    with pytest.raises(LLMError):
        ICD10(FakeLLM(LLMError("down"))).classify("x", "")


def test_code_fields_adds_the_three_keys_without_touching_the_rest():
    fields = {"final_diagnosis": "COPD", "chief_complaint": None, "age": 5}
    out = ICD10(FakeLLM({"code": "J44"})).code_fields(fields)
    assert out["icd10_code"] == "J44" and out["icd10_category"].startswith("Other chronic") and out["age"] == 5 and "icd10_code" not in fields
    blank = ICD10(FakeLLM()).code_fields({"final_diagnosis": None})
    assert [blank[k] for k in ICD10.FIELDS] == [None, None, None]


def test_the_default_model_path_uses_the_ollama_request_format(ollama):
    ollama.reply(json.dumps({"code": "J18"}))
    assert ICD10().classify("Community-acquired pneumonia", "Cough").code == "J18"
    request = ollama.requests[0]
    assert request["messages"][0]["role"] == "system" and "ICD-10" in request["messages"][0]["content"]
    assert request["format"]["required"] == ["code"] and request["options"]["temperature"] == 0


# ---------------------------------------------------------------- update_rows: the logic shared by Excel and DuckDB
GOOD = {"icd10_code": "J44", "icd10_chapter": "Diseases of the respiratory system", "icd10_category": "Other chronic obstructive pulmonary disease"}
NONE = dict.fromkeys(ICD10.FIELDS)


def row(dx, cc="cc", **icd):
    return {"final_diagnosis": dx, "chief_complaint": cc, **NONE, **icd}


def test_is_complete_needs_all_three_and_a_real_code():
    assert ICD10.is_complete(GOOD)
    assert not ICD10.is_complete({**GOOD, "icd10_chapter": None}) and not ICD10.is_complete({**GOOD, "icd10_category": ""}) and not ICD10.is_complete({})
    assert not ICD10.is_complete({"icd10_code": "I92.9", "icd10_chapter": "0", "icd10_category": "0"})
    assert not ICD10.is_complete({**GOOD, "icd10_category": "something else"})            # disagrees with the ontology


def test_missing_mode_codes_what_is_incomplete_and_shares_calls_between_equal_diagnoses():
    rows = [row("COPD"), row("COPD"), row("Pneumonia"), row("COPD"), row("Asthma", **GOOD)]
    llm = answers_by_diagnosis({"COPD": "J44", "Pneumonia": "J18"})
    report = ICD10(llm).update_rows(rows, "missing")
    assert (report.records, report.pending, report.coded, report.derived, report.unresolved, report.model_calls) == (5, 4, 4, 0, 0, 2)
    assert len(llm.calls) == 2 and rows[2]["icd10_code"] == "J18" and rows[0]["icd10_chapter"] == "Diseases of the respiratory system"
    assert rows[4] == row("Asthma", **GOOD)                                               # already complete: untouched
    assert ICD10(FakeLLM()).update_rows(rows, "missing").pending == 0                     # a second run has nothing to do


def test_a_valid_code_with_missing_chapter_is_completed_without_the_model():
    rows = [row("COPD", icd10_code="i21.9"), row("COPD", icd10_code="J44", icd10_chapter="wrong text", icd10_category="wrong text")]
    llm = FakeLLM()
    report = ICD10(llm).update_rows(rows, "missing")
    assert (report.pending, report.derived, report.coded, report.model_calls) == (2, 2, 0, 0) and llm.calls == []
    assert rows[0]["icd10_code"] == "I21.9" and rows[0]["icd10_category"] == "Acute myocardial infarction" and rows[1] == row("COPD", **GOOD)


def test_a_code_that_is_not_icd10_is_recoded_by_the_model():
    rows = [row("COPD", icd10_code="J48", icd10_chapter="0", icd10_category="0")]                # as found in a real Excel
    assert ICD10(answers_by_diagnosis({"COPD": "J44"})).update_rows(rows).coded == 1 and rows[0]["icd10_code"] == "J44"


def test_all_mode_recodes_every_row_even_the_complete_ones():
    rows = [row("COPD", **GOOD), row("COPD"), row("", cc="")]
    llm = FakeLLM({"code": "J18"})
    report = ICD10(llm).update_rows(rows, "all")
    assert (report.pending, report.coded, report.unresolved, report.model_calls) == (3, 2, 1, 1) and len(llm.calls) == 1
    assert rows[0]["icd10_code"] == rows[1]["icd10_code"] == "J18" and rows[2] == row("", cc="")           # nothing to code: left as it was


def test_a_row_the_model_cannot_code_keeps_what_it_had():
    rows = [row("odd", icd10_code="J48", icd10_chapter="0", icd10_category="0")]
    report = ICD10(FakeLLM("no idea", "no idea")).update_rows(rows, "all")
    assert report.unresolved == 1 and rows[0]["icd10_code"] == "J48"


def test_dry_run_counts_without_changing_or_calling_the_model():
    rows = [row("COPD"), row("COPD", icd10_code="J44"), row("x", **GOOD)]
    report = ICD10(FakeLLM(LLMError("must not be called"))).update_rows(rows, "missing", dry_run=True)
    assert (report.pending, report.derived, report.coded) == (2, 1, 0) and rows[0]["icd10_code"] is None
    assert ICD10().update_rows(rows, "all", dry_run=True).pending == 3                              # no model is created either


def test_on_update_is_called_for_each_changed_row_as_it_happens():
    seen = []
    rows = [row("A"), row("B", **GOOD), row("C")]
    ICD10(FakeLLM({"code": "J44"}, {"code": "J18"})).update_rows(rows, "missing", on_update=lambda r: seen.append(r["final_diagnosis"]))
    assert seen == ["A", "C"]


def test_a_mode_must_be_missing_or_all():
    for call in (lambda: ICD10().update_rows([], "everything"), lambda: ICD10().update_icd10("x.xlsx", "y.xlsx", mode="nope")):
        with pytest.raises(ValueError, match="mode must be"):
            call()


# ---------------------------------------------------------------- update_icd10: an Excel in, an Excel out
HEADER = ["record_id", "patient_name", "final_diagnosis", "chief_complaint", "encounter_date", "icd10_code", "icd10_chapter", "icd10_category"]
WHEN = datetime(2025, 3, 1, 9, 30)


def make_excel(path, rows, header=HEADER):
    wb = Workbook()
    ws = wb.active
    ws.title = "data"
    ws.append(header)
    for r in rows:
        ws.append(r)
    ws.column_dimensions["C"].width = 41
    wb.save(path)


def read_excel(path):
    ws = load_workbook(path).worksheets[0]
    header = [c.value for c in ws[1]]
    return [dict(zip(header, r)) for r in ws.iter_rows(min_row=2, values_only=True)]


@pytest.fixture
def xlsx(tmp_path):
    path = tmp_path / "in.xlsx"
    make_excel(path, [
        ["R-1", "Ana", "COPD", "cough", WHEN, "J44", GOOD["icd10_chapter"], GOOD["icd10_category"]],      # complete
        ["R-2", "Ben", "Pneumonia", "fever", WHEN, None, None, None],                                       # missing
        ["R-3", "Cy", "COPD", "cough", WHEN, "J48", 0, 0],                                                  # a code that is not ICD-10, zeros
        ["R-4", "Di", "Asthma", "wheeze", WHEN, "I21.9", None, None],                                       # a real code, no chapter / category
    ])
    return path


def test_the_excel_is_updated_in_a_new_file_and_the_source_is_untouched(xlsx, tmp_path):
    before = xlsx.read_bytes()
    report = ICD10(answers_by_diagnosis({"COPD": "J44", "Pneumonia": "J18"})).update_icd10(xlsx, tmp_path / "out.xlsx")
    assert (report.records, report.pending, report.derived, report.coded, report.model_calls) == (4, 3, 1, 2, 2)
    assert xlsx.read_bytes() == before
    rows = read_excel(tmp_path / "out.xlsx")
    assert [r["icd10_code"] for r in rows] == ["J44", "J18", "J44", "I21.9"]
    assert rows[1]["icd10_chapter"] == "Diseases of the respiratory system" and rows[3]["icd10_category"] == "Acute myocardial infarction"
    assert [r["patient_name"] for r in rows] == ["Ana", "Ben", "Cy", "Di"] and rows[0]["encounter_date"] == WHEN       # the rest is intact
    assert load_workbook(tmp_path / "out.xlsx").worksheets[0].column_dimensions["C"].width == 41                       # and its formatting


def test_the_excel_can_be_updated_in_place(xlsx):
    ICD10(answers_by_diagnosis({"COPD": "J44", "Pneumonia": "J18"})).update_icd10(xlsx, xlsx)
    assert [r["icd10_code"] for r in read_excel(xlsx)] == ["J44", "J18", "J44", "I21.9"]
    assert not list(xlsx.parent.glob("*.tmp"))                                                                          # no temporary file left


def test_missing_mode_leaves_complete_rows_alone_and_all_mode_recodes_them(xlsx, tmp_path):
    always_j18 = lambda: answers_by_diagnosis({"COPD": "J18", "Pneumonia": "J18", "Asthma": "J18"})
    ICD10(always_j18()).update_icd10(xlsx, tmp_path / "all.xlsx", mode="all")
    assert [r["icd10_code"] for r in read_excel(tmp_path / "all.xlsx")] == ["J18"] * 4                                  # every row, R-1 too
    ICD10(always_j18()).update_icd10(xlsx, tmp_path / "missing.xlsx", mode="missing")
    assert [r["icd10_code"] for r in read_excel(tmp_path / "missing.xlsx")] == ["J44", "J18", "J18", "I21.9"]          # R-1 kept its own


def test_the_icd10_columns_are_added_when_the_excel_has_none(tmp_path):
    path = tmp_path / "old.xlsx"
    make_excel(path, [["R-1", "Ana", "COPD", "cough", WHEN]], header=HEADER[:5])
    ICD10(FakeLLM({"code": "J44"})).update_icd10(path, tmp_path / "new.xlsx")
    row_ = read_excel(tmp_path / "new.xlsx")[0]
    assert list(row_)[-3:] == list(ICD10.FIELDS) and row_["icd10_code"] == "J44" and row_["icd10_category"] == GOOD["icd10_category"]


def test_a_sheet_can_be_chosen_and_a_header_is_required(tmp_path):
    path = tmp_path / "x.xlsx"
    make_excel(path, [])
    wb = load_workbook(path)
    wb.create_sheet("other").append(["a", "b"])
    wb.save(path)
    with pytest.raises(ValueError, match="final_diagnosis"):
        ICD10(FakeLLM()).update_icd10(path, tmp_path / "o.xlsx", sheet="other")


def test_dry_run_on_an_excel_writes_nothing(xlsx, tmp_path):
    report = ICD10(FakeLLM(LLMError("no model"))).update_icd10(xlsx, tmp_path / "out.xlsx", dry_run=True)
    assert report.pending == 3 and not (tmp_path / "out.xlsx").exists()


def test_when_the_model_fails_halfway_the_coded_rows_are_saved_and_the_run_can_continue(xlsx):
    with pytest.raises(LLMError):
        ICD10(FakeLLM({"code": "J44"}, LLMError("down"))).update_icd10(xlsx, xlsx)             # R-2 is coded, then R-3 fails
    done = [r["icd10_code"] for r in read_excel(xlsx)]
    assert done[1] == "J44" and done[0] == "J44"                                              # R-2 was coded before the failure and saved
    report = ICD10(FakeLLM({"code": "J18"})).update_icd10(xlsx, xlsx)
    assert report.pending == 2 and [r["icd10_code"] for r in read_excel(xlsx)][1] == "J44"      # it continues: R-2 is not asked again


# ---------------------------------------------------------------- RecordDB.import_excel(update_icd=...)
def db_excel(tmp_path):
    from docintel.db import COLUMNS
    from tests.test_db import RECORD, excel_row, make_xlsx
    old = [c for c in COLUMNS if not c.startswith("icd10")]
    rows = [excel_row({**RECORD, "record_id": f"R-{i}", **icd})[:len(old)] + [icd.get(c) for c in ICD10.FIELDS]
            for i, icd in enumerate([GOOD, NONE, {"icd10_code": "J48", "icd10_chapter": 0, "icd10_category": 0}])]
    make_xlsx(tmp_path / "m.xlsx", rows)
    return tmp_path / "m.xlsx"


def test_import_excel_does_not_update_the_icd10_by_default(tmp_path):
    with RecordDB(":memory:") as db:
        assert db.import_excel(db_excel(tmp_path)) == 3
        assert [db.read_record(f"R-{i}")["icd10_code"] for i in range(3)] == ["J44", None, "J48"]        # stored as the Excel has them


@pytest.mark.parametrize("mode,expected", [("missing", ["J44", "J18", "J18"]), ("all", ["J18", "J18", "J18"])])
def test_import_excel_can_update_the_icd10_first(tmp_path, monkeypatch, mode, expected):
    path = db_excel(tmp_path)
    monkeypatch.setattr("docintel.data_transformation.LLM", lambda *a, **k: FakeLLM(*[{"code": "J18"}] * 3))
    before = path.read_bytes()
    with RecordDB(":memory:") as db:
        assert db.import_excel(path, update_icd=True, icd_mode=mode) == 3
        assert [db.read_record(f"R-{i}")["icd10_code"] for i in range(3)] == expected
        assert db.read_record("R-2")["icd10_chapter"] == "Diseases of the respiratory system"
    assert path.read_bytes() == before                                                                   # without icd_output the Excel is untouched


def test_import_excel_can_also_save_the_updated_excel(tmp_path, monkeypatch):
    path = db_excel(tmp_path)
    monkeypatch.setattr("docintel.data_transformation.LLM", lambda *a, **k: FakeLLM(*[{"code": "J18"}] * 3))
    with RecordDB(":memory:") as db:
        db.import_excel(path, update_icd=True, icd_output=tmp_path / "updated.xlsx")
    assert [r["icd10_code"] for r in read_excel(tmp_path / "updated.xlsx")] == ["J44", "J18", "J18"]


def test_import_excel_arguments_are_checked_and_a_failed_coding_imports_nothing(tmp_path, monkeypatch):
    path = db_excel(tmp_path)
    with RecordDB(":memory:") as db:
        with pytest.raises(ValueError, match="icd_output is only used"):
            db.import_excel(path, icd_output=tmp_path / "x.xlsx")
        with pytest.raises(ValueError, match="mode must be"):
            db.import_excel(path, update_icd=True, icd_mode="some")
        monkeypatch.setattr("docintel.data_transformation.LLM", lambda *a, **k: FakeLLM(LLMError("down")))
        with pytest.raises(LLMError):
            db.import_excel(path, update_icd=True)
        assert db.con.execute("SELECT count(*) FROM medical_records").fetchone()[0] == 0


# ---------------------------------------------------------------- update_database: records already in a DuckDB
@pytest.fixture
def db():
    with RecordDB(":memory:") as d:
        yield d


def record(i, dx, **icd):
    return {"record_id": f"R-{i}", "patient_name": f"P{i}", "final_diagnosis": dx, "chief_complaint": "cc", **icd}


def test_update_database_codes_the_pending_records_and_changes_nothing_else(db):
    for i, dx in enumerate(["COPD", "Pneumonia"]):
        db.create_record(record(i, dx))
    db.create_record(record(9, "Asthma", **GOOD))
    report = ICD10(answers_by_diagnosis({"COPD": "J44", "Pneumonia": "J18"})).update_database(db)
    assert (report.records, report.pending, report.coded) == (3, 2, 2)
    assert db.read_record("R-1")["icd10_code"] == "J18" and db.read_record("R-1")["patient_name"] == "P1" and db.read_record("R-9")["icd10_code"] == "J44"
    assert ICD10(FakeLLM()).update_database(db).pending == 0


def test_update_database_all_mode_and_dry_run(db):
    db.create_record(record(1, "COPD", **GOOD))
    assert ICD10(FakeLLM(LLMError("no"))).update_database(db, mode="all", dry_run=True).pending == 1
    assert db.read_record("R-1")["icd10_code"] == "J44"
    assert ICD10(FakeLLM({"code": "J18"})).update_database(db, mode="all").coded == 1 and db.read_record("R-1")["icd10_code"] == "J18"


def test_an_interrupted_database_update_keeps_what_it_did(db):
    for i, dx in enumerate(["A", "B", "C"]):
        db.create_record(record(i, dx))
    with pytest.raises(LLMError):
        ICD10(FakeLLM({"code": "J44"}, {"code": "J18"}, LLMError("down"))).update_database(db)
    assert [db.read_record(f"R-{i}")["icd10_code"] for i in range(3)] == ["J44", "J18", None]
