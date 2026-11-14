import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from docintel import RecordDB
from docintel.web import create_app, forms, pipeline
from docintel.web.__main__ import main
from docintel.web.pipeline import Extraction

FIELDS = {  # what the (fake) model found in the uploaded document
    "record_id": "NEW-0001", "department": "Infectious Diseases", "patient_name": "Samantha Nelson", "dob": "1962-09-24T00:00:00",
    "age": 63, "sex": "Female", "insurance": "Blue Cross", "city": "Houston, Texas", "encounter_date": "2026-06-09T09:25:00",
    "chief_complaint": "Red, swollen, painful leg.", "history_of_present_illness": "Two days of erythema.", "past_medical_history": None,
    "medications": "Metformin.", "allergies": "NKDA.", "vital_signs": {"blood_pressure": "113/84 mmHg", "heart_rate": "90 bpm",
    "respiration_rate": None, "temperature": "100.9 °F", "sp02": "93%"}, "physical_exam": "Erythema.",
    "lab_results": [{"test_name": "WBC Count", "result": "12.6 K/uL", "flag": "High"}, {"test_name": "Blood Culture", "result": "No growth", "flag": "Normal"}],
    "imaging_results": None, "final_diagnosis": "Non-purulent cellulitis.", "plan": "Antibiotics.", "progress_notes": None,
    "icd10_code": "L03.1", "icd10_chapter": "Diseases of the skin and subcutaneous tissue", "icd10_category": "Cellulitis",
}
PDF = ("scan.pdf", b"%PDF-1.4 fake pdf", "application/pdf")


def form_data(fields: dict, **changes) -> dict:
    """What the browser posts for a record: the form values, lab rows as repeated fields."""
    values = {**forms.to_values(fields), **changes}
    labs = values.pop("labs")
    return {**values, **{f"lab_{c}": [row[c] for row in labs] for c in forms.LAB_COLUMNS}}


@pytest.fixture
def env(tmp_path):
    """A client for the app with a fake pipeline: `env.result` / `env.error` decide what a job does."""
    e = SimpleNamespace(db=tmp_path / "db.duckdb", up=tmp_path / "uploads", tmp=tmp_path, result=Extraction(FIELDS, ["patient_name"], 2), error=None)

    def fake(path, on_step):
        on_step("extracting")
        if e.error:
            raise RuntimeError(e.error)
        return e.result

    e.app = create_app(db_path=e.db, uploads_dir=e.up, pipeline=fake, allowed_hosts=["testserver"], max_upload_mb=1)
    with TestClient(e.app) as client:
        e.client = client
        yield e


def upload(env, file=PDF) -> str:
    """Upload a file and wait for its job; returns the job page URL."""
    r = env.client.post("/upload", files={"file": file}, follow_redirects=False)
    assert r.status_code == 303, r.text
    url = r.headers["location"]
    for _ in range(100):
        page = env.client.get(url).text
        if "Save to the database" in page or "could not be processed" in page:
            return url
        time.sleep(0.05)
    raise AssertionError("the job never finished")


def seed(env, n=3, **extra):
    with RecordDB(env.db) as db:
        for i in range(n):
            db.create_record({**FIELDS, "record_id": f"REC-{i:03d}", "patient_name": f"Patient {i:02d}",
                              "department": "Cardiology" if i % 2 else "Oncology", "encounter_date": (datetime(2025, 1, 1, 10) + timedelta(days=i)).isoformat(), **extra})


def stored(env, record_id="NEW-0001"):
    with RecordDB(env.db) as db:
        return db.read_record(record_id)


# ---------------------------------------------------------------- load a document
def test_home_redirects_to_upload_and_upload_page_works(env):
    assert env.client.get("/", follow_redirects=False).headers["location"] == "/upload"
    page = env.client.get("/upload").text
    assert "Drag a file here" in page and 'name="file"' in page and "1 MB maximum" in page


def test_upload_review_and_save(env):
    job = upload(env)
    page = env.client.get(job).text
    assert 'value="Samantha Nelson"' in page and 'value="2026-06-09 09:25"' in page and 'value="12.6 K/uL"' in page
    assert "2 pages" in page and "scan.pdf" in page
    assert f'<iframe src="{job}/document#view=FitH"' in page                       # the original, only while reviewing
    assert "1 value was not found in the document: Patient" in page               # find_ungrounded -> flagged
    saved = env.client.post(f"{job}/save", data=form_data(FIELDS), follow_redirects=False)
    assert saved.status_code == 303 and saved.headers["location"] == "/records?ok=saved&id=NEW-0001"
    assert stored(env) == FIELDS
    assert not any(env.up.iterdir())                                                    # the file is not kept: only the data is
    assert env.client.get(job).status_code == 404                                       # the job is over
    listing = env.client.get(saved.headers["location"]).text
    assert "Record NEW-0001 saved." in listing and "Samantha Nelson" in listing
    assert "<iframe" not in listing and "document" not in listing.lower().replace("upload document", "")   # no original afterwards
    assert env.client.get("/records/NEW-0001/document").status_code == 404                              # and no way to ask for it
    assert not any(env.up.iterdir())


def test_the_user_can_correct_what_the_model_found(env):
    job = upload(env)
    data = form_data(FIELDS, patient_name="Samantha R. Nelson", age="99", sex="", allergies="")
    data["lab_test_name"], data["lab_result"], data["lab_flag"] = ["WBC Count", ""], ["13.0 K/uL", ""], ["High", ""]   # 2nd row left blank
    env.client.post(f"{job}/save", data=data)
    record = stored(env)
    assert record["patient_name"] == "Samantha R. Nelson" and record["age"] == 63 and record["sex"] is None and record["allergies"] is None
    assert record["lab_results"] == [{"test_name": "WBC Count", "result": "13.0 K/uL", "flag": "High"}]


def test_invalid_values_keep_the_form_and_the_file(env):
    job = upload(env)
    r = env.client.post(f"{job}/save", data=form_data(FIELDS, age="63 years", encounter_date="2026-06-09T9:25:03", patient_name="Samantha R. Nelson"))
    assert r.status_code == 422
    assert "Check 2 fields before saving" in r.text and "Enter a whole number" in r.text and "Use YYYY-MM-DD" in r.text
    assert 'value="63 years"' in r.text and 'aria-invalid="true"' in r.text                 # what the user typed is still there
    assert "Not found in the document" not in r.text                                   # the flagged name was edited: reviewed
    assert stored(env) is None and any(env.up.iterdir())
    again = env.client.post(f"{job}/save", data=form_data(FIELDS, dob="x"))
    assert "Not found in the document" in again.text                                   # untouched: still flagged
    assert env.client.post(f"{job}/save", data=form_data(FIELDS)).status_code == 200      # fixed: saved (redirect followed)


def test_a_record_id_that_is_not_plain_is_refused(env):
    job = upload(env)
    for bad in ("", "a/b", "../x", "x y", "é", "_x", "-x", "a" * 65):
        r = env.client.post(f"{job}/save", data=form_data(FIELDS, record_id=bad))
        assert r.status_code == 422 and "Record ID" in r.text, bad
    assert stored(env, "NEW-0001") is None


def test_saving_an_existing_id_asks_before_replacing(env):
    env.client.post(f"{upload(env)}/save", data=form_data(FIELDS))
    env.result = Extraction({**FIELDS, "patient_name": "Someone Else", "plan": "New plan."}, [], 1)
    job = upload(env, ("second.pdf", b"%PDF second", "application/pdf"))
    clash = env.client.post(f"{job}/save", data=form_data(env.result.fields))
    assert clash.status_code == 422 and "A record with this ID already exists" in clash.text and 'name="replace"' in clash.text
    assert stored(env)["patient_name"] == "Samantha Nelson"                                  # nothing was overwritten
    done = env.client.post(f"{job}/save", data={**form_data(env.result.fields), "replace": "on"}, follow_redirects=False)
    assert done.status_code == 303
    assert stored(env)["patient_name"] == "Someone Else" and stored(env)["plan"] == "New plan."
    assert not any(env.up.iterdir())                                                         # nothing is kept


def test_cancel_discards_the_upload(env):
    job = upload(env)
    assert any(env.up.iterdir())
    assert env.client.post(f"{job}/cancel", follow_redirects=False).headers["location"] == "/upload"
    assert env.client.get(job).status_code == 404 and not any(env.up.iterdir())
    assert env.client.post("/upload/nope/cancel", follow_redirects=False).status_code == 303   # already gone: no error


def test_a_failed_job_says_why_and_can_be_retried(env):
    env.error = "Ollama is not running"
    job = upload(env)
    page = env.client.get(job).text
    assert "The document could not be processed" in page and "RuntimeError: Ollama is not running" in page and "failed" in page
    assert "Save to the database" not in page
    env.error = None
    assert env.client.post(f"{job}/retry", follow_redirects=False).headers["location"] == job
    for _ in range(100):
        if "Save to the database" in env.client.get(job).text:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("retry did not finish")


def test_uploads_are_checked(env):
    for file, code in [(("a.exe", b"MZ", "application/octet-stream"), "type"), (("a.svg", b"<svg/>", "image/svg+xml"), "type"),
                       (("a.pdf", b"", "application/pdf"), "empty"), (("a.txt", b"x" * (1024 * 1024 + 1), "text/plain"), "size")]:
        r = env.client.post("/upload", files={"file": file}, follow_redirects=False)
        assert r.headers["location"] == f"/upload?error={code}", file[0]
    assert "Unsupported file type" in env.client.get("/upload?error=type").text
    assert "larger than the 1 MB limit" in env.client.get("/upload?error=size").text
    assert env.client.post("/upload", follow_redirects=False).headers["location"] == "/upload?error=empty"
    assert not any(env.up.iterdir())                                        # refused files leave nothing


def test_the_file_name_cannot_escape_the_uploads_folder(env):
    job = upload(env, ("../../evil.pdf", b"%PDF", "application/pdf"))
    assert (env.up / job.rsplit("/", 1)[1] / "evil.pdf").exists() and not (env.tmp / "evil.pdf").exists()
    assert "evil.pdf" in env.client.get(job).text and "../" not in env.client.get(job).text.split("<h1>")[1].split("</section>")[0]


def test_a_word_file_is_offered_as_a_download_not_previewed(env):
    job = upload(env, ("note.docx", b"PK fake", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
    assert "cannot be shown here" in env.client.get(job).text
    assert "attachment" in env.client.get(f"{job}/document").headers["content-disposition"]


# ---------------------------------------------------------------- the database
def test_listing_search_filter_and_pages(env):
    seed(env, 30)
    page1 = env.client.get("/records").text
    assert page1.count(">Edit</a>") == 25 and "Showing 1–25 of 30" in page1 and "30 records" in page1
    assert 'href="/records?page=2"' in page1 and ">Previous</a>" not in page1
    page2 = env.client.get("/records?page=2").text
    assert page2.count(">Edit</a>") == 5 and "Showing 26–30 of 30" in page2 and ">Next</a>" not in page2
    assert env.client.get("/records?page=99").text.count(">Edit</a>") == 5               # a stale page number lands on the last one
    found = env.client.get("/records?q=patient 07").text
    assert found.count(">Edit</a>") == 1 and "1 record found" in found and "REC-007" in found
    assert env.client.get("/records?d=Oncology").text.count(">Edit</a>") == 15
    assert '<option selected>Oncology</option>' in env.client.get("/records?d=Oncology").text
    assert "No records match your search." in env.client.get("/records?q=zzz").text


def test_an_empty_database_says_so(env):
    page = env.client.get("/records").text
    assert "No records." in page and "0 records" in page and ">Next</a>" not in page


def test_text_in_records_is_escaped(env):
    seed(env, 1, final_diagnosis="<script>alert(1)</script>")
    page = env.client.get("/records").text
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_edit_shows_only_the_record_data(env):
    seed(env, 1)
    page = env.client.get("/records/REC-000/edit").text
    assert 'value="Patient 00"' in page and "<iframe" not in page and 'id="show-doc"' not in page and "riginal document" not in page
    assert 'readonly' in page.split('name="record_id"')[1].split(">")[0] and "The ID cannot be changed." in page


def test_edit_saves_every_field_and_clears_what_was_emptied(env):
    seed(env, 1)
    data = form_data(FIELDS, record_id="HACKED", patient_name="Ana", age="70", **{"vital_signs.heart_rate": "55 bpm", "encounter_date": "2026-03-01 08:30:15"})
    data["lab_test_name"], data["lab_result"], data["lab_flag"] = ["Glucose", "", "Sodium"], ["97 mg/dL", "", "138"], ["Normal", "", ""]
    r = env.client.post("/records/REC-000/edit", data=data, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/records?ok=updated&id=REC-000"
    record = stored(env, "REC-000")
    assert record["patient_name"] == "Ana" and record["age"] == 63 and record["vital_signs"]["heart_rate"] == "55 bpm"
    assert record["encounter_date"] == "2026-03-01T08:30:15" and record["record_id"] == "REC-000"        # the ID in the body is ignored
    assert record["lab_results"] == [{"test_name": "Glucose", "result": "97 mg/dL", "flag": "Normal"}, {"test_name": "Sodium", "result": "138", "flag": None}]
    assert stored(env, "HACKED") is None


def test_edit_validation_and_unknown_records(env):
    seed(env, 1)
    r = env.client.post("/records/REC-000/edit", data=form_data(FIELDS, dob="31/12/1990"))
    assert r.status_code == 422 and "Not a valid date" in r.text and 'value="31/12/1990"' in r.text
    assert stored(env, "REC-000")["patient_name"] == "Patient 00"
    assert env.client.get("/records/NOPE/edit").status_code == 404
    assert "The record you are looking for does not exist." in env.client.get("/records/NOPE/edit").text
    assert env.client.post("/records/NOPE/edit", data=form_data(FIELDS)).status_code == 404
    assert env.client.get("/records/REC-000/document").status_code == 404                  # records have no original to show


def test_delete_asks_first_then_removes_the_record(env):
    seed(env, 2)
    page = env.client.get("/records/REC-000/delete?q=Patient").text
    assert 'role="dialog"' in page and "Delete record REC-000?" in page and "The data of Patient 00 will be deleted." in page
    assert "riginal document" not in page.split('role="dialog"')[1]
    assert 'name="q" value="Patient"' in page and "REC-001" in page                         # the list is still there behind it
    assert stored(env, "REC-000") is not None                                                  # asking deletes nothing
    r = env.client.post("/records/REC-000/delete", data={"q": "Patient", "d": "", "page": ""}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/records?q=Patient&ok=deleted&id=REC-000"
    assert stored(env, "REC-000") is None and stored(env, "REC-001") is not None
    assert "Record REC-000 deleted." in env.client.get(r.headers["location"]).text
    again = env.client.post("/records/REC-000/delete", follow_redirects=False)
    assert again.status_code == 303 and "deleted" not in again.headers["location"]            # nothing to delete: no false message
    assert env.client.get("/records/REC-000/delete").status_code == 404


# ---------------------------------------------------------------- safety
def test_other_websites_cannot_post_here(env):
    seed(env, 1)
    r = env.client.post("/records/REC-000/delete", headers={"Origin": "http://evil.example"}, follow_redirects=False)
    assert r.status_code == 403 and stored(env, "REC-000") is not None
    r = env.client.post("/records/REC-000/delete", headers={"Origin": "http://testserver"}, follow_redirects=False)
    assert r.status_code == 303 and stored(env, "REC-000") is None


def test_only_the_expected_host_is_served(env):
    assert TestClient(env.app, base_url="http://evil.example").get("/records").status_code == 400


def test_pages_are_not_cached(env):
    assert env.client.get("/records").headers["cache-control"] == "no-store"


def test_the_app_only_listens_on_this_computer():
    with pytest.raises(SystemExit):
        main(["--host", "0.0.0.0"])


# ---------------------------------------------------------------- the real pipeline, with its two steps replaced
def test_pipeline_reads_then_extracts_and_flags_invented_values(monkeypatch):
    steps = []
    monkeypatch.setattr(pipeline, "extract_auto", lambda path: {"text": "Patient Ana Gomez, age 40", "pages": [{}, {}, {}]})
    monkeypatch.setattr(pipeline, "get_fields", lambda text, template: {"patient_name": "Ana Gomez", "city": "Atlantis", "vital_signs": {"heart_rate": None}})
    monkeypatch.setattr(pipeline.ICD10, "code_fields", lambda self, fields: {**fields, "icd10_code": "J44", "icd10_chapter": "c", "icd10_category": "k"})
    out = pipeline.run(Path("x.pdf"), steps.append)
    assert steps == ["reading", "extracting", "coding"] and out.pages == 3 and out.ungrounded == ["city"]   # the derived ICD-10 values are not 'invented'
    assert out.fields["icd10_code"] == "J44" and out.notice == ""


def test_pipeline_keeps_the_extraction_when_the_coding_fails(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_auto", lambda path: {"text": "Ana Gomez", "pages": [{}]})
    monkeypatch.setattr(pipeline, "get_fields", lambda text, template: {"patient_name": "Ana Gomez", "final_diagnosis": "COPD"})
    def down(fields):
        raise RuntimeError("model down")
    monkeypatch.setattr(pipeline.ICD10, "code_fields", lambda self, fields: down(fields))
    out = pipeline.run(Path("x.pdf"), lambda step: None)
    assert out.fields["patient_name"] == "Ana Gomez" and out.fields["icd10_code"] is None and "model down" in out.notice
    monkeypatch.setattr(pipeline.ICD10, "code_fields", lambda self, fields: {**fields, **dict.fromkeys(pipeline.ICD10.FIELDS)})
    assert "No ICD-10 code could be assigned" in pipeline.run(Path("x.pdf"), lambda step: None).notice


def test_pipeline_refuses_a_document_without_text(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_auto", lambda path: {"text": "  \n", "pages": [{}]})
    monkeypatch.setattr(pipeline, "get_fields", lambda *a: pytest.fail("the model must not see an empty text"))
    with pytest.raises(ValueError, match="No text was found"):
        pipeline.run(Path("blank.pdf"), lambda step: None)


# ---------------------------------------------------------------- form conversion
def test_form_values_round_trip_a_record():
    fields, errors = forms.parse(forms.to_values(FIELDS))
    assert errors == {} and {k: v for k, v in fields.items()} == FIELDS


@pytest.mark.parametrize("text,iso", [("2026-06-09", "2026-06-09T00:00:00"), ("2026-06-09 09:25", "2026-06-09T09:25:00"),
                                      ("2026-06-09 09:25:03", "2026-06-09T09:25:03"), ("2026-06-09T09:25", "2026-06-09T09:25:00")])
def test_dates_are_accepted_in_every_usual_format(text, iso):
    assert forms.parse({"record_id": "A-1", "patient_name": "Ana", "encounter_date": text})[0]["encounter_date"] == iso


@pytest.mark.parametrize("text", ["09/06/2026", "2026-06-09T9:25:03", "2026-13-01", "ayer", "2026-06-09T09:25:00+02:00"])
def test_dates_that_are_not_dates_are_reported(text):
    assert "encounter_date" in forms.parse({"record_id": "A-1", "encounter_date": text})[1]


def test_dates_show_as_the_user_types_them():
    show = lambda iso: forms.to_values({"encounter_date": iso})["encounter_date"]
    assert [show("2026-06-09T00:00:00"), show("2026-06-09T09:25:00"), show("2026-06-09T09:25:03"), show(None)] == ["2026-06-09", "2026-06-09 09:25", "2026-06-09 09:25:03", ""]


def test_empty_text_becomes_none_and_age_must_be_an_integer():
    fields, errors = forms.parse({"record_id": "A-1", "patient_name": "Ana", "city": "   ", "age": ""})
    assert errors == {} and fields["city"] is None and fields["age"] is None and fields["lab_results"] == []
    assert fields["vital_signs"] == {k: None for k in forms.VITALS}
    for bad in ("63.5", "-3", "sixty", "٣"):
        assert "age" in forms.parse({"record_id": "A-1", "patient_name": "Ana", "age": bad})[1], bad


def test_labels_for_the_flagged_values():
    assert [forms.label_of(p) for p in ("age", "vital_signs.sp02", "lab_results[1].result", "nope")] == ["Age", "SpO2", "Lab result 2 (result)", "nope"]


# ---------------------------------------------------------------- patient, age and lab suggestions
def test_the_patient_is_required(env):
    seed(env, 1)
    for blank in ("", "   "):
        r = env.client.post("/records/REC-000/edit", data=form_data(FIELDS, patient_name=blank))
        assert r.status_code == 422 and "The patient name is required." in r.text and 'aria-invalid="true"' in r.text
    assert stored(env, "REC-000")["patient_name"] == "Patient 00"
    job = upload(env)
    r = env.client.post(f"{job}/save", data=form_data(FIELDS, patient_name=""))
    assert r.status_code == 422 and "The patient name is required." in r.text and stored(env) is None


@pytest.mark.parametrize("dob,encounter,age", [
    ("1962-09-24", "2026-06-09", 63), ("1962-09-24", "2026-09-23", 63), ("1962-09-24", "2026-09-24", 64),
    ("1962-09-24", "2026-09-24 08:00", 64), ("2024-02-29", "2025-02-28", 0), ("2024-02-29", "2025-03-01", 1), ("2026-06-09", "2026-06-09", 0)])
def test_age_is_the_whole_years_between_birth_and_encounter(dob, encounter, age):
    fields, errors = forms.parse({"record_id": "A-1", "patient_name": "Ana", "dob": dob, "encounter_date": encounter, "age": "7"})
    assert errors == {} and fields["age"] == age                          # whatever was typed in Age is replaced


def test_age_is_typed_only_when_it_cannot_be_calculated():
    base = {"record_id": "A-1", "patient_name": "Ana"}
    assert forms.parse({**base, "age": "40"})[0]["age"] == 40
    assert forms.parse({**base, "age": "40", "dob": "1980-01-01"})[0]["age"] == 40          # one date is not enough
    assert forms.parse({**base, "age": "40", "encounter_date": "2020-01-01"})[0]["age"] == 40


def test_an_encounter_before_the_birth_is_refused():
    fields, errors = forms.parse({"record_id": "A-1", "patient_name": "Ana", "dob": "2000-01-01", "encounter_date": "1999-12-31"})
    assert "encounter_date" in errors and "earlier than the date of birth" in errors["encounter_date"]


def test_edit_recalculates_the_age_when_a_date_changes(env):
    seed(env, 1)
    env.client.post("/records/REC-000/edit", data=form_data(FIELDS, dob="1990-12-31", encounter_date="2026-06-09 09:00"))
    assert stored(env, "REC-000")["age"] == 35
    env.client.post("/records/REC-000/edit", data=form_data(FIELDS, dob="1990-12-31", encounter_date="2026-12-31"))
    assert stored(env, "REC-000")["age"] == 36


def test_the_age_box_tells_how_it_is_calculated(env):
    seed(env, 1)
    assert "Calculated from the date of birth and the encounter date." in env.client.get("/records/REC-000/edit").text


def test_lab_boxes_suggest_the_standard_tests_and_flags_but_accept_anything(env):
    assert len(forms.LAB_TESTS) == 40 == len(set(forms.LAB_TESTS)) and forms.LAB_FLAGS == ("Normal", "High", "Low", "Borderline")
    seed(env, 1)
    page = env.client.get("/records/REC-000/edit").text
    assert page.count('<datalist id="lab-tests">') == 1 and page.count('<option value="Troponin I (initial)">') == 1 and page.count('<option value="pH (arterial)">') == 1
    assert '<option value="Borderline">' in page
    assert page.count('list="lab-tests"') == len(FIELDS["lab_results"]) + 1 and page.count('list="lab-flags"') == len(FIELDS["lab_results"]) + 1   # each row + the template
    data = form_data(FIELDS)
    data["lab_test_name"], data["lab_result"], data["lab_flag"] = ["Glucose", "My own assay"], ["97", "5"], ["Normal", "Suspicious"]
    assert env.client.post("/records/REC-000/edit", data=data, follow_redirects=False).status_code == 303
    assert [(r["test_name"], r["flag"]) for r in stored(env, "REC-000")["lab_results"]] == [("Glucose", "Normal"), ("My own assay", "Suspicious")]


# ---------------------------------------------------------------- page size and first / last page
def rows_on(page: str) -> int:
    return page.count(">Edit</a>")


def test_first_and_last_page_buttons(env):
    seed(env, 45)                                                                             
    p1 = env.client.get("/records").text
    assert "Page 1 of 2" in p1 and ">Next</a>" in p1 and "Last »" in p1 and "« First" not in p1 and ">Previous</a>" not in p1
    assert 'href="/records?page=2" aria-label="Last page"' in p1
    p2 = env.client.get("/records?page=2").text
    assert "« First" in p2 and 'href="/records" aria-label="First page"' in p2 and ">Previous</a>" in p2 and "Last »" not in p2 and ">Next</a>" not in p2
    single = env.client.get("/records?n=100").text
    assert "Page " not in single and "« First" not in single and "Last »" not in single


def test_records_per_page_can_be_chosen(env):
    seed(env, 120)
    assert rows_on(env.client.get("/records").text) == 25                                        # default
    for n, pages, last in [(20, 6, 20), (25, 5, 20), (50, 3, 20), (100, 2, 20)]:
        first = env.client.get(f"/records?n={n}").text
        assert rows_on(first) == n and f"Page 1 of {pages}" in first, n
        assert rows_on(env.client.get(f"/records?n={n}&page={pages}").text) == last, n
    page = env.client.get("/records?n=50").text
    assert '<option selected>50</option>' in page and [f"<option>{x}</option>" in page for x in (20, 25, 100)] == [True] * 3
    assert 'href="/records?n=50&amp;page=2"' in page and 'href="/records?n=50&amp;page=3" aria-label="Last page"' in page   # the size travels with the links


def test_an_unlisted_page_size_falls_back_to_the_default(env):
    seed(env, 60)
    assert rows_on(env.client.get("/records?n=7").text) == 25 and rows_on(env.client.get("/records?n=100000").text) == 25


def test_the_page_size_survives_the_delete_dialog(env):
    seed(env, 60)
    dialog = env.client.get("/records/REC-000/delete?n=50&page=2").text
    assert 'name="n" value="50"' in dialog and rows_on(dialog) == 10
    r = env.client.post("/records/REC-000/delete", data={"q": "", "d": "", "n": "50", "page": "2"}, follow_redirects=False)
    assert r.headers["location"] == "/records?n=50&page=2&ok=deleted&id=REC-000"


# ---------------------------------------------------------------- ICD-10
def test_the_review_shows_the_three_icd10_fields_and_the_coding_step(env):
    job = upload(env)
    page = env.client.get(job).text
    assert "Assigning the ICD-10 code" in page and "ICD-10 classification" in page
    assert 'name="icd10_code" value="L03.1"' in page
    assert 'value="Diseases of the skin and subcutaneous tissue"' in page and 'value="Cellulitis"' in page
    for name in ("icd10_chapter", "icd10_category"):
        assert "readonly" in page.split(f'name="{name}"')[1].split(">")[0]                    # derived, not typed
    assert "Not found in the document" not in page.split('id="f-icd10_code"')[1].split("</div>")[0]   # a derived value is never flagged


def test_a_notice_from_the_coding_is_shown_in_the_review(env):
    env.result = Extraction({**FIELDS, "icd10_code": None, "icd10_chapter": None, "icd10_category": None}, [], 1, "No ICD-10 code could be assigned automatically.")
    page = env.client.get(upload(env)).text
    assert "No ICD-10 code could be assigned automatically." in page and 'name="icd10_code" value=""' in page


def test_saving_stores_the_icd10_fields(env):
    env.client.post(f"{upload(env)}/save", data=form_data(FIELDS))
    record = stored(env)
    assert [record[c] for c in ("icd10_code", "icd10_chapter", "icd10_category")] == ["L03.1", "Diseases of the skin and subcutaneous tissue", "Cellulitis"]


def test_chapter_and_category_always_follow_the_code(env):
    job = upload(env)
    data = form_data(FIELDS, icd10_code="j44", icd10_chapter="typed by hand", icd10_category="typed by hand")
    env.client.post(f"{job}/save", data=data)
    record = stored(env)
    assert record["icd10_code"] == "J44" and record["icd10_chapter"] == "Diseases of the respiratory system"
    assert record["icd10_category"] == "Other chronic obstructive pulmonary disease"


def test_an_icd10_code_that_does_not_exist_is_refused(env):
    job = upload(env)
    for bad in ("ZZZ", "J4", "I20-I25", "IX", "L48"):
        r = env.client.post(f"{job}/save", data=form_data(FIELDS, icd10_code=bad))
        assert r.status_code == 422 and "Not a valid ICD-10 code" in r.text and f'value="{bad}"' in r.text, bad
    assert stored(env) is None


def test_an_empty_code_clears_chapter_and_category(env):
    env.client.post(f"{upload(env)}/save", data=form_data(FIELDS, icd10_code=""))
    assert [stored(env)[c] for c in ("icd10_code", "icd10_chapter", "icd10_category")] == [None, None, None]


def test_editing_the_code_of_a_stored_record(env):
    seed(env, 1)
    page = env.client.get("/records/REC-000/edit").text
    assert 'name="icd10_code" value="L03.1"' in page and 'value="Cellulitis"' in page
    env.client.post("/records/REC-000/edit", data=form_data({**FIELDS, **dict.fromkeys(("icd10_chapter", "icd10_category"), "")}, icd10_code="I21.9"))
    assert stored(env, "REC-000")["icd10_category"] == "Acute myocardial infarction"
    legacy = {**FIELDS, "record_id": "OLD-1", "icd10_code": "I92.9", "icd10_chapter": "0", "icd10_category": "0"}      # as in the Excel
    with RecordDB(env.db) as db:
        db.create_record(legacy)
    r = env.client.post("/records/OLD-1/edit", data=form_data(legacy))
    assert r.status_code == 422 and "Not a valid ICD-10 code" in r.text                    # the user is asked to fix it


def test_the_code_lookup_for_the_form(env):
    assert env.client.get("/icd10?code=i21.9").json() == {"code": "I21.9", "chapter": "Diseases of the circulatory system", "category": "Acute myocardial infarction"}
    assert env.client.get("/icd10?code=nope").json() == {} and env.client.get("/icd10").json() == {}
