"""The copilot: catalog, dates, plan validation, the read-only executor, answer templates, planner and the whole flow.
No model is needed: a fake LLM returns plans, a fake embedder gives vectors from keywords."""
import json
from datetime import date

import duckdb
import pytest

from docintel.copilot import Catalog, Copilot, Executor, Planner, PlannerError, validate
from docintel.copilot import dates
from docintel.copilot.catalog import LAB, LAB_NAME
from docintel.copilot.plan import PlanFormatError, canonical_plan, measurement_number, parse_plan, plan_schema, plans_equivalent
from docintel.copilot.render import render
from docintel.database import RecordDB
from docintel.shared import EmbeddingError, LLMError
from docintel.shared.settings import s

TODAY = date(2026, 11, 13)
KEYWORDS = ["breath", "chest", "fever", "other"]


class KeywordEmbedder:
    """Vectors by keyword, so similarity is predictable: a text about 'breath' is close to a query about breathing."""
    model, batch_size = "kw", 10

    def __init__(self, down=False):
        self.down = down

    def vec(self, text):
        t = text.lower()
        v = [1.0 if k in t else 0.0 for k in KEYWORDS]
        return v if any(v[:3]) else [0.0, 0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        return [self.vec(t) for t in texts]

    def embed_query(self, text):
        if self.down:
            raise EmbeddingError("down")
        return self.vec(text)


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def run(self, messages, **params):
        self.calls.append((messages, params))
        reply = self.replies.pop(0) if self.replies else "{}"
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def rec(i, dept, sex, age, day, complaint, dx, vitals=None, labs=None, **more):
    return {"record_id": f"R{i:03d}", "department": dept, "patient_name": f"Patient {i}", "sex": sex, "age": age,
            "insurance": "Cigna", "city": "Miami, Florida", "encounter_date": f"{day}T09:00:00", "chief_complaint": complaint,
            "final_diagnosis": dx, "vital_signs": vitals, "lab_results": labs, **more}


def lab(name, flag):
    return {"test_name": name, "result": "1 x", "flag": flag}


def vit(t=None, hr=None, bp=None, sp=None):
    return {"blood_pressure": bp, "heart_rate": hr, "respiration_rate": "18/min", "temperature": t, "sp02": sp}


SEED = [
    rec(1, "Cardiology", "Female", 70, "2026-03-10", "chest pain", "NSTEMI", vit("101.0 °F", "110 bpm", "150/95 mmHg", "97%"),
        [lab("Troponin I", "High"), lab("Glucose", "Normal")], icd10_code="I21", icd10_chapter="Diseases of the circulatory system"),
    rec(2, "Cardiology", "Male", 55, "2026-05-02", "chest pressure", "Stable angina", None, [lab("Troponin I (initial)", "Borderline")]),
    rec(3, "Cardiology", "Female", 62, "2025-11-20", "swollen legs", "Heart failure", None, [lab("BNP", "High"), lab("Troponin I", "High")],
        icd10_code="I50", icd10_chapter="Diseases of the circulatory system"),
    rec(4, "Neurology", "Male", 45, "2025-03-15", "headache", "Migraine", vit("98.6 °F")),
    rec(5, "Neurology", "Female", 80, "2026-01-20", "weakness", "Ischemic stroke", None,
        [lab("Glucose", "High"), lab("Glucose (point-of-care)", "Normal")], icd10_code="I63", icd10_chapter="Diseases of the circulatory system"),
    rec(6, "Pulmonology", "Male", 30, "2025-07-04", "shortness of breath at night", "Asthma", vit(hr="90 bpm", sp="91%")),
    rec(7, "Pulmonology", "Female", 66, "2026-10-30", "cough and fever", "Pneumonia", vit("103.1 °F", sp="89%"),
        icd10_code="J18", icd10_chapter="Diseases of the respiratory system"),
    rec(8, "Urology", "Male", 52, "2026-11-10", "flank pain", "Kidney stone"),
    rec(9, "Urology", "Female", 38, "2026-11-01", "burning urination", "Urinary tract infection", vit("100.4 °F")),
    rec(10, "Cardiology", "Male", 75, "2026-11-12", "palpitations", "Atrial fibrillation", vit(hr="130 bpm")),
    {"record_id": "R011"},
    rec(12, "Cardiology", "Female", 58, "2025-01-05", "palpitations", "SVT"),
]


@pytest.fixture
def db():
    d = RecordDB(":memory:", embedder=KeywordEmbedder())
    for r in SEED:
        d.create_record(r)
    d.embed_pending()
    yield d
    d.close()


@pytest.fixture
def catalog(db):
    return Catalog.build(db.con, today=TODAY)


def cop(db, *replies, embedder=None):
    return Copilot.from_db(db, llm=FakeLLM(*replies), embedder=embedder or KeywordEmbedder(), today=TODAY)


def f(field, op, value):
    return {"field": field, "op": op, "value": value}


def ids_of(answer):
    return sorted(answer.result.ids)


# ================================================================ dates
@pytest.mark.parametrize("period,start,end", [
    ({"year": 2025}, date(2025, 1, 1), date(2026, 1, 1)),
    ({"month": "2025-03"}, date(2025, 3, 1), date(2025, 4, 1)),
    ({"month": "2025-12"}, date(2025, 12, 1), date(2026, 1, 1)),
    ({"day": "2025-03-04"}, date(2025, 3, 4), date(2025, 3, 5)),
    ({"from": "2025-01-01", "to": "2025-03-31"}, date(2025, 1, 1), date(2025, 4, 1)),
    ({"token": "this_year"}, date(2026, 1, 1), date(2027, 1, 1)),
    ({"token": "last_year"}, date(2025, 1, 1), date(2026, 1, 1)),
    ({"token": "this_month"}, date(2026, 11, 1), date(2026, 12, 1)),
    ({"token": "last_month"}, date(2026, 10, 1), date(2026, 11, 1)),
    ({"token": "today"}, date(2026, 11, 13), date(2026, 11, 14)),
    ({"token": "yesterday"}, date(2026, 11, 12), date(2026, 11, 13)),
    ({"token": "last_n_days", "n": 30}, date(2026, 10, 14), date(2026, 11, 14)),
])
def test_periods_expand_to_half_open_ranges(period, start, end):
    assert dates.expand(period, TODAY) == (start, end)


def test_last_month_in_january_is_december_of_the_year_before():
    assert dates.expand({"token": "last_month"}, date(2026, 1, 15)) == (date(2025, 12, 1), date(2026, 1, 1))


@pytest.mark.parametrize("bad", [{"month": "2025-13"}, {"month": "March"}, {"day": "2025-02-30"}, {"from": "2025-05-01", "to": "2025-01-01"},
                                 {"from": "2025-01-01"}, {"token": "next_week"}, {"token": "last_n_days", "n": 0}, {"token": "last_n_days"},
                                 {"year": 1500}, {"year": "abc"}, "2025", {}])
def test_bad_periods_are_rejected(bad):
    with pytest.raises(dates.BadPeriod):
        dates.expand(bad, TODAY)


def test_describe():
    assert dates.describe(date(2025, 1, 1), date(2026, 1, 1)) == "2025"
    assert dates.describe(date(2025, 3, 1), date(2025, 4, 1)) == "March 2025"
    assert dates.describe(date(2025, 3, 4), date(2025, 3, 5)) == "2025-03-04"
    assert dates.describe(date(2025, 1, 1), date(2025, 4, 1)) == "2025-01-01 to 2025-03-31"


# ================================================================ catalog
def test_catalog_lists_values_with_counts_and_ranges(catalog):
    assert catalog.values("department") == {"Cardiology": 5, "Neurology": 2, "Pulmonology": 2, "Urology": 2}
    assert catalog.values("sex") == {"Female": 6, "Male": 5}
    assert catalog.values(LAB_NAME)["Troponin I"] == 2 and "Troponin I (initial)" in catalog.values(LAB_NAME)
    assert catalog.columns["age"].range == (30, 80)
    assert catalog.date_span() == (date(2025, 1, 5), date(2026, 11, 12))
    assert catalog.total_records == 12
    assert catalog.columns["vital_signs.temperature"].unit == "°F"
    assert catalog.role("lab_results") == "list_of_objects" and catalog.role("icd10_chapter") == "icd10_hierarchy"


def test_ops_follow_the_role(catalog):
    assert catalog.ops("department") == ("eq", "in") and catalog.ops("age")[-1] == "between"
    assert catalog.ops("final_diagnosis") == ("contains",) and catalog.ops("lab_results") == ("has",)
    assert LAB_NAME not in catalog.filter_fields() and "vital_signs.sp02" in catalog.filter_fields()


def test_yaml_roundtrip(catalog, tmp_path):
    catalog.to_yaml(tmp_path / "c.yaml")
    again = Catalog.from_yaml(tmp_path / "c.yaml")
    assert again.values("department") == catalog.values("department") and again.date_span() == catalog.date_span()
    assert Catalog.from_yaml(catalog.to_yaml()).total_records == 12  # YAML text works too


def test_unknown_role_in_yaml_is_an_error():
    with pytest.raises(ValueError, match="unknown role"):
        Catalog.from_yaml("columns:\n  x:\n    role: magic\n")


def test_refresh_updates_counts_and_keeps_edited_descriptions(db, catalog):
    catalog.columns["department"].description = "Edited by a person."
    db.create_record(rec(13, "Oncology", "Male", 50, "2026-11-13", "lump", "Tumor"))
    fresh = catalog.refresh(db.con, today=TODAY)
    assert fresh.values("department")["Oncology"] == 1 and fresh.total_records == 13
    assert fresh.columns["department"].description == "Edited by a person."
    db.delete_record("R013")
    assert "Oncology" not in catalog.refresh(db.con).values("department")


def test_a_column_with_too_many_values_does_not_list_them(db, monkeypatch):
    monkeypatch.setattr(s, "CATALOG_MAX_VALUES", 3)
    cat = Catalog.build(db.con)
    assert cat.values("department") is None and cat.columns["department"].distinct == 4
    assert cat.values("sex") == {"Female": 6, "Male": 5}
    assert "values" not in cat.columns["department"].to_yaml() and cat.columns["department"].to_yaml()["distinct_values"] == 4


def test_a_database_without_the_icd10_columns_still_profiles():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE medical_records (record_id VARCHAR PRIMARY KEY, department VARCHAR, age INTEGER, encounter_date TIMESTAMP, "
                "vital_signs JSON, lab_results JSON)")
    con.execute("INSERT INTO medical_records VALUES ('A', 'Cardiology', 50, TIMESTAMP '2025-01-01 10:00:00', NULL, NULL)")
    cat = Catalog.build(con)
    assert "icd10_code" not in cat.columns and cat.values("department") == {"Cardiology": 1}


def test_a_model_drafts_descriptions_only_for_columns_nobody_described():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE medical_records (record_id VARCHAR PRIMARY KEY, department VARCHAR)")
    con.execute("ALTER TABLE medical_records ADD COLUMN bed VARCHAR")
    con.execute("INSERT INTO medical_records VALUES ('A', 'Cardiology', 'B12')")
    llm = FakeLLM("The bed number.\nExtra line")
    cat = Catalog.build(con, llm=llm)
    assert cat.columns["bed"].description == "The bed number." and len(llm.calls) == 1
    assert cat.columns["department"].description.startswith("Clinical department")


def test_suggestions_and_test_name_expansion(catalog):
    assert catalog.canonical("department", " cardiology ") == "Cardiology" and catalog.canonical("department", "Cardiologia") is None
    assert catalog.suggest("department", "Cardiologia") == ["Cardiology"]
    assert catalog.suggest("department", "Pulmon") == ["Pulmonology"]
    assert catalog.expand_test_names("Troponin") == ["Troponin I", "Troponin I (initial)"]
    assert catalog.expand_test_names("troponin i") == ["Troponin I", "Troponin I (initial)"]  # an exact name (any case) also brings its variants
    assert catalog.expand_test_names("Potassium") == []


def test_the_prompt_text_has_values_and_trims_long_lists(catalog, monkeypatch):
    text = catalog.prompt_text("cardiology patients")
    assert "Cardiology (5)" in text and "Troponin I (initial)" in text and "flag meanings" in text
    short = catalog.prompt_text("urology", max_values=2)
    assert short.count("Urology (2)") == 1 and "department" in short  # the value most like the question is kept first


# ================================================================ plan helpers
def test_parse_plan_finds_the_json_in_prose_and_fences():
    assert parse_plan('Sure! ```json\n{"intent": "help"}\n``` hope it helps') == {"intent": "help"}
    for bad in ("", "no json here", "{broken", "[1, 2]"):
        with pytest.raises(PlanFormatError):
            parse_plan(bad)


def test_measurement_numbers_and_units():
    assert measurement_number("vital_signs.temperature", {"value": 39, "unit": "C"}) == 102.2
    assert measurement_number("vital_signs.temperature", {"value": 39, "unit": "°C"}) == 102.2
    assert measurement_number("vital_signs.temperature", 102.2) == 102.2
    assert measurement_number("age", "70") == 70.0
    for bad in (None, True, "abc", float("nan"), {"value": None}):
        with pytest.raises(ValueError):
            measurement_number("age", bad)


def test_plan_schema_lists_the_fields_and_ops(catalog):
    schema = plan_schema(catalog)
    assert "department" in schema["properties"]["filters"]["items"]["properties"]["field"]["enum"]
    assert LAB_NAME not in schema["properties"]["filters"]["items"]["properties"]["field"]["enum"]
    assert "out_of_scope" in schema["properties"]["intent"]["enum"] and schema["required"] == ["intent"]


def test_equivalent_plans_compare_equal(catalog):
    a = {"intent": "search", "filters": [f("department", "in", ["Cardiology"]), f("age", "gt", 60)], "text_query": None}
    b = {"intent": "search", "filters": [f("age", "gt", 60.0), f("department", "eq", "cardiology")]}
    assert plans_equivalent(a, b, catalog)
    assert not plans_equivalent(a, {**b, "filters": [f("age", "gt", 61), f("department", "eq", "Cardiology")]}, catalog)
    t1 = {"intent": "search", "filters": [f("vital_signs.temperature", "gt", 102.2)]}
    t2 = {"intent": "search", "filters": [f("vital_signs.temperature", "gt", {"value": 39, "unit": "C"})]}
    assert plans_equivalent(t1, t2, catalog)
    l1 = {"intent": "search", "filters": [f("lab_results", "has", {"test_name": ["B", "A"], "flag": ["High"]})]}
    assert canonical_plan(l1)["filters"][0]["value"] == {"test_name": ["A", "B"], "flag": ["High"]}


# ================================================================ validator
def v(plan, catalog, **kw):
    return validate(plan, catalog, today=TODAY, **kw)


def test_a_good_search_is_validated_and_described(catalog):
    q = v({"intent": "search", "filters": [f("department", "eq", "cardiology"), f("age", "gt", 60),
                                           f("encounter_date", "between", {"token": "this_year"})]}, catalog)
    assert q.intent == "search" and [x.label for x in q.filters] == ["department: Cardiology", "age > 60", "encounter date: 2026"]
    assert q.filters[0].op == "in" and q.filters[0].value == ["Cardiology"]  # eq on a categorical becomes in, spelled as the catalog does
    assert q.filters[2].value == {"start": "2026-01-01", "end": "2027-01-01"}


@pytest.mark.parametrize("plan", [None, [], "search", {}, {"intent": "delete"}, {"intent": 5}])
def test_malformed_plans_ask_instead_of_failing(plan, catalog):
    q = v(plan, catalog)
    assert q.intent == "clarify" and q.message


def test_unknown_fields_and_operators_are_refused(catalog):
    q = v({"intent": "search", "filters": [f("salary", "gt", 5)]}, catalog)
    assert q.intent == "clarify" and "salary" in q.message
    q = v({"intent": "search", "filters": [f("lab_results.flag", "eq", "High")]}, catalog)  # a part of lab_results is not a filter field
    assert q.intent == "clarify"
    q = v({"intent": "search", "filters": [f("department", "gt", 5)]}, catalog)
    assert q.intent == "clarify" and "accepts: eq, in" in q.message
    assert v({"intent": "search", "filters": [f("age", "contains", "5")]}, catalog).intent == "clarify"
    assert v({"intent": "search", "filters": ["age > 5"]}, catalog).intent == "clarify"
    assert v({"intent": "search", "filters": "age"}, catalog).intent == "clarify"


def test_a_misspelled_value_asks_with_a_suggestion(catalog):
    q = v({"intent": "search", "filters": [f("department", "in", ["Cardiologia"])]}, catalog)
    assert q.intent == "clarify" and q.suggestions == ["Cardiology"] and "Did you mean “Cardiology”" in q.message
    q = v({"intent": "search", "filters": [f("city", "eq", "Brooklyn")]}, catalog)
    assert q.intent == "clarify" and not q.suggestions and "Brooklyn" in q.message
    q = v({"intent": "search", "filters": [f("city", "eq", "Miami")]}, catalog)  # a part of a value is a suggestion, not a silent match
    assert q.intent == "clarify" and q.suggestions == ["Miami, Florida"]


def test_a_value_missing_from_the_catalog_but_in_the_database_is_accepted(db, catalog):
    db.create_record(rec(14, "Oncology", "Male", 50, "2026-11-13", "lump", "Tumor"))  # the catalog (built before) does not know it
    assert v({"intent": "search", "filters": [f("department", "eq", "oncology")]}, catalog).intent == "clarify"
    live = Executor(db.con).value_exists
    q = v({"intent": "search", "filters": [f("department", "eq", "oncology")]}, catalog, value_exists=live)
    assert q.intent == "search" and q.filters[0].value == ["Oncology"]


def test_value_lists_must_be_sensible(catalog):
    for bad in ([], [""], [1], "Cardiology", ["Cardiology"] * 21):
        assert v({"intent": "search", "filters": [f("department", "in", bad)]}, catalog).intent == "clarify"
    q = v({"intent": "search", "filters": [f("department", "in", ["Cardiology", "neurology"])]}, catalog)
    assert q.filters[0].value == ["Cardiology", "Neurology"] and q.filters[0].label == "department: Cardiology or Neurology"


def test_dates_and_the_range_outside_the_data(catalog):
    q = v({"intent": "search", "filters": [f("encounter_date", "between", {"month": "2020-01"})]}, catalog)
    assert q.intent == "search" and "outside that range" in q.warnings[0] and "2025-01-05 to 2026-11-12" in q.warnings[0]
    assert not v({"intent": "search", "filters": [f("encounter_date", "between", {"year": 2025})]}, catalog).warnings
    for bad in ({"month": "2025-13"}, {"from": "2025-05-01", "to": "2025-01-01"}, "2025", {"token": "soon"}):
        q = v({"intent": "search", "filters": [f("encounter_date", "between", bad)]}, catalog)
        assert q.intent == "clarify" and "date" in q.message
    assert v({"intent": "search", "filters": [f("encounter_date", "gt", "2025-01-01")]}, catalog).intent == "clarify"


def test_numbers_are_checked_and_units_converted(catalog):
    q = v({"intent": "search", "filters": [f("vital_signs.temperature", "gt", {"value": 39, "unit": "C"})]}, catalog)
    assert q.filters[0].value == 102.2 and q.filters[0].label == "temperature > 102.2 °F"
    for field, value in (("age", 200), ("age", -1), ("vital_signs.heart_rate", 5000), ("vital_signs.temperature", 39), ("age", "old"), ("age", None)):
        assert v({"intent": "search", "filters": [f(field, "gt", value)]}, catalog).intent == "clarify", (field, value)
    q = v({"intent": "search", "filters": [f("age", "between", [60, 70])]}, catalog)
    assert q.filters[0].value == [60, 70] and q.filters[0].label == "age 60 to 70"
    assert v({"intent": "search", "filters": [f("age", "between", [70, 60])]}, catalog).intent == "clarify"
    assert v({"intent": "search", "filters": [f("age", "between", [60])]}, catalog).intent == "clarify"
    assert v({"intent": "search", "filters": [f("vital_signs.sp02", "lt", 92)]}, catalog).filters[0].label == "sp02 < 92 %"


def test_lab_filters(catalog):
    q = v({"intent": "search", "filters": [f("lab_results", "has", {"test_name": ["troponin i"], "flag": ["high"]})]}, catalog)
    assert q.filters[0].value == {"test_name": ["Troponin I", "Troponin I (initial)"], "flag": ["High"]} and "matches" in q.warnings[0]
    q = v({"intent": "search", "filters": [f("lab_results", "has", {"test_name": ["Troponin"]})]}, catalog)
    assert q.filters[0].value["test_name"] == ["Troponin I", "Troponin I (initial)"] and "matches" in q.warnings[0]
    assert q.filters[0].label == "lab: Troponin I / Troponin I (initial)"
    for bad in ({"test_name": ["Troponim"]}, {"test_name": ["Troponin I"], "flag": ["Huge"]}, {}, "High", {"flag": []}):
        assert v({"intent": "search", "filters": [f("lab_results", "has", bad)]}, catalog).intent == "clarify", bad
    q = v({"intent": "search", "filters": [f("lab_results", "has", {"test_name": ["Troponim"]})]}, catalog)
    assert q.suggestions and q.suggestions[0] == "Troponin I" and "Did you mean" in q.message


def test_text_fields_and_the_icd_code(catalog):
    q = v({"intent": "search", "filters": [f("final_diagnosis", "contains", " pneumonia ")]}, catalog)
    assert q.filters[0].value == "pneumonia" and q.filters[0].label == "final diagnosis contains “pneumonia”"
    assert v({"intent": "search", "filters": [f("final_diagnosis", "contains", "")]}, catalog).intent == "clarify"
    assert v({"intent": "search", "filters": [f("final_diagnosis", "contains", "x" * 101)]}, catalog).intent == "clarify"
    assert v({"intent": "search", "filters": [f("icd10_code", "eq", "i21")]}, catalog).filters[0].value == ["I21"]
    assert v({"intent": "search", "filters": [f("icd10_code", "eq", "Z99")]}, catalog).intent == "clarify"


def test_searches_need_a_condition_and_duplicates_are_dropped(catalog):
    assert v({"intent": "search", "filters": []}, catalog).intent == "clarify"
    assert v({"intent": "search"}, catalog).intent == "clarify"
    assert v({"intent": "search", "filters": [], "text_query": "chest pain"}, catalog).text_query == "chest pain"
    q = v({"intent": "search", "filters": [f("age", "gt", 60), f("age", "gt", 60)]}, catalog)
    assert len(q.filters) == 1
    assert v({"intent": "search", "filters": [f("age", "gt", 1)] * 8 + [f("age", "lt", 99)]}, catalog).intent == "clarify"  # 9 distinct-ish > limit
    assert v({"intent": "search", "text_query": "x" * 201}, catalog).intent == "clarify"


def test_aggregate_rules(catalog):
    ok = v({"intent": "aggregate", "filters": [], "aggregate": {"metric": "avg", "field": "age", "group_by": "department", "top_n": 3}}, catalog)
    assert (ok.metric, ok.agg_field, ok.group_by, ok.top_n) == ("avg", "age", "department", 3)
    assert v({"intent": "aggregate", "aggregate": {"metric": "count"}}, catalog).group_by is None
    for bad in (None, {}, {"metric": "sum"}, {"metric": "avg"}, {"metric": "avg", "field": "department"}, {"metric": "avg", "field": "salary"},
                {"metric": "count", "top_n": 0}, {"metric": "count", "top_n": 51}, {"metric": "count", "top_n": "many"}):
        assert v({"intent": "aggregate", "aggregate": bad}, catalog).intent == "clarify", bad


def test_a_group_by_that_cannot_be_used_is_dropped_and_said(catalog):
    """"How many patients were tested for creatinine?" must give the number, not a question about grouping."""
    for bad in ("age", ["sex", "department"], "salary", "lab_results.test_name", "patient"):
        q = v({"intent": "aggregate", "filters": [], "aggregate": {"metric": "count", "group_by": bad}}, catalog)
        assert (q.intent, q.group_by) == ("aggregate", None), bad
        assert q.warnings and "overall figure" in q.warnings[0] and "department" in q.warnings[0]


def test_records_navigation_help_and_refusals(catalog):
    assert v({"intent": "get_record", "record_id": "SYNUS-2026-0001"}, catalog).record_id == "SYNUS-2026-0001"
    for bad in (None, "", "a b", "x;DROP", "../etc", "x" * 65):
        assert v({"intent": "get_record", "record_id": bad}, catalog).intent == "clarify", bad
    assert v({"intent": "navigate", "destination": "upload"}, catalog).destination == "upload"
    assert v({"intent": "navigate", "destination": "delete"}, catalog).intent == "clarify"
    assert v({"intent": "navigate", "destination": "record"}, catalog).intent == "clarify"
    nav = v({"intent": "navigate", "destination": "records", "filters": [f("department", "eq", "Cardiology")]}, catalog)
    assert nav.filters[0].value == ["Cardiology"]
    assert v({"intent": "help", "topic": "upload_formats"}, catalog).topic == "upload_formats"
    assert v({"intent": "help", "topic": "weird"}, catalog).topic == "capabilities"
    assert v({"intent": "out_of_scope", "reason": "write"}, catalog).reason == "write"
    assert v({"intent": "out_of_scope", "reason": "because"}, catalog).reason == "unrelated"
    c = v({"intent": "clarify", "question": "Which department?", "suggestions": ["A", "B"]}, catalog)
    assert (c.intent, c.message, c.suggestions) == ("clarify", "Which department?", ["A", "B"])


# ================================================================ executor
def run(db, plan, catalog=None, page=1, **kw):
    c = Copilot.from_db(db, llm=FakeLLM(), embedder=KeywordEmbedder(), today=TODAY)
    return c.answer_plan(plan, page)


@pytest.mark.parametrize("filters,expected", [
    ([f("department", "in", ["Cardiology"])], {1, 2, 3, 10, 12}),
    ([f("department", "in", ["Cardiology", "Neurology"]), f("sex", "eq", "Female")], {1, 3, 5, 12}),
    ([f("age", "gt", 60)], {1, 3, 5, 7, 10}),
    ([f("age", "gte", 62), f("age", "lt", 75)], {1, 3, 7}),
    ([f("age", "between", [45, 55])], {2, 4, 8}),
    ([f("department", "in", ["Cardiology"]), f("age", "gt", 60), f("lab_results", "has", {"test_name": ["Troponin I", "Troponin I (initial)"], "flag": ["High"]})], {1, 3}),
    ([f("lab_results", "has", {"test_name": ["Troponin"]})], {1, 2, 3}),
    ([f("lab_results", "has", {"flag": ["High"]})], {1, 3, 5}),
    ([f("lab_results", "has", {"test_name": ["Glucose", "Glucose (point-of-care)"], "flag": ["High"]})], {5}),
    ([f("encounter_date", "between", {"token": "this_year"})], {1, 2, 5, 7, 8, 9, 10}),
    ([f("encounter_date", "between", {"month": "2025-03"})], {4}),
    ([f("encounter_date", "between", {"token": "last_n_days", "n": 30})], {7, 8, 9, 10}),
    ([f("vital_signs.temperature", "gt", 102.2)], {7}),
    ([f("vital_signs.temperature", "gt", {"value": 39, "unit": "C"})], {7}),
    ([f("vital_signs.temperature", "gte", 100.4)], {1, 7, 9}),
    ([f("vital_signs.heart_rate", "gt", 100)], {1, 10}),
    ([f("vital_signs.sp02", "lt", 92)], {6, 7}),
    ([f("vital_signs.blood_pressure_systolic", "gt", 140)], {1}),
    ([f("vital_signs.blood_pressure_diastolic", "gte", 95)], {1}),
    ([f("final_diagnosis", "contains", "pneumo")], {7}),
    ([f("final_diagnosis", "contains", "ANGINA")], {2}),
    ([f("icd10_chapter", "eq", "Diseases of the circulatory system")], {1, 3, 5}),
    ([f("icd10_code", "eq", "j18")], {7}),
    ([f("patient_name", "contains", "patient 1")], {1, 10, 12}),
    ([f("record_id", "eq", "R003")], {3}),
])
def test_every_kind_of_filter_returns_the_right_records(db, filters, expected):
    answer = run(db, {"intent": "search", "filters": filters})
    assert answer.kind == "results", answer.text
    assert {int(i[1:]) for i in answer.result.ids} == expected and answer.result.total == len(expected)


def test_results_are_newest_first_with_a_page_of_summaries(db):
    a = run(db, {"intent": "search", "filters": [f("department", "in", ["Cardiology"])]})
    assert a.result.ids == ["R010", "R002", "R001", "R003", "R012"]
    assert a.columns == ["record_id", "patient_name", "department", "encounter_date", "age", "final_diagnosis"]
    assert a.rows[0] == {"record_id": "R010", "patient_name": "Patient 10", "department": "Cardiology", "encounter_date": "2026-11-12T09:00:00",
                         "age": 75, "final_diagnosis": "Atrial fibrillation"}


def test_paging(db, monkeypatch):
    c = Copilot.from_db(db, llm=FakeLLM(), embedder=KeywordEmbedder(), today=TODAY)
    c.executor.page_size = 2
    plan = {"intent": "search", "filters": [f("department", "in", ["Cardiology"])]}
    a1, a2, a3 = c.answer_plan(plan, 1), c.answer_plan(plan, 2), c.answer_plan(plan, 3)
    assert [r["record_id"] for r in a1.rows] == ["R010", "R002"] and a1.has_more and "Showing 1–2" in a1.summary
    assert [r["record_id"] for r in a2.rows] == ["R001", "R003"] and a2.has_more and "Showing 3–4" in a2.summary
    assert [r["record_id"] for r in a3.rows] == ["R012"] and not a3.has_more and "Showing 5–5" in a3.summary
    assert a1.result.total == a2.result.total == a3.result.total == 5


@pytest.mark.parametrize("agg,expected", [
    ({"metric": "count"}, [{"count": 12}]),
    ({"metric": "count", "group_by": "department"}, [{"department": "Cardiology", "count": 5}, {"department": "Neurology", "count": 2},
                                                      {"department": "Pulmonology", "count": 2}, {"department": "Urology", "count": 2},
                                                      {"department": None, "count": 1}]),
    ({"metric": "count", "group_by": "department", "top_n": 2}, [{"department": "Cardiology", "count": 5}, {"department": "Neurology", "count": 2}]),
    ({"metric": "max", "field": "age"}, [{"max_age": 80}]),
    ({"metric": "min", "field": "age"}, [{"min_age": 30}]),
    ({"metric": "avg", "field": "age", "group_by": "sex"}, [{"sex": "Female", "avg_age": 62.3}, {"sex": "Male", "avg_age": 51.4}, {"sex": None, "avg_age": None}]),
    ({"metric": "max", "field": "vital_signs.temperature"}, [{"max_temperature": 103.1}]),
])
def test_aggregates(db, agg, expected):
    a = run(db, {"intent": "aggregate", "filters": [], "aggregate": agg})
    assert a.kind == "aggregate", a.text
    assert a.result.rows == expected


def test_an_aggregate_with_filters(db):
    a = run(db, {"intent": "aggregate", "filters": [f("encounter_date", "between", {"year": 2025})],
                 "aggregate": {"metric": "count", "group_by": "department"}})
    assert {r["department"]: r["count"] for r in a.result.rows} == {"Cardiology": 2, "Neurology": 1, "Pulmonology": 1}
    assert a.interpretation == "count of records by department · encounter date: 2025"


def test_get_record(db):
    a = run(db, {"intent": "get_record", "record_id": "R001"})
    assert a.kind == "record" and a.result.record["final_diagnosis"] == "NSTEMI" and a.result.record["lab_results"][0]["test_name"] == "Troponin I"
    assert a.links[0].url == "/records/R001/edit" and "Patient 1" in a.summary
    missing = run(db, {"intent": "get_record", "record_id": "NOPE"})
    assert "There is no record NOPE" in missing.summary and missing.result.total == 0


def test_zero_results_come_with_hints_computed_by_the_database(db):
    a = run(db, {"intent": "search", "filters": [f("department", "in", ["Neurology"]), f("age", "gt", 85)]})
    assert a.result.total == 0 and a.summary == "No records match."
    assert a.result.hints == [("age > 85", 2)]  # dropping the department leaves 0, so only the age condition is worth naming
    assert a.notes == ["Without “age > 85” there are 2 matches."]
    a = run(db, {"intent": "search", "filters": [f("department", "in", ["Cardiology"]), f("age", "gt", 85)]})
    assert a.result.hints == [("age > 85", 5)] and "Without “age > 85” there are 5 matches." in a.notes
    a = run(db, {"intent": "search", "filters": [f("department", "in", ["Cardiology"]), f("sex", "eq", "Male"), f("age", "gt", 99)]})
    assert a.result.hints == [("age > 99", 2)]
    a = run(db, {"intent": "search", "filters": [f("sex", "eq", "Female"), f("age", "gt", 79), f("department", "in", ["Pulmonology"])]})
    assert a.result.hints == [("age > 79", 1), ("department: Pulmonology", 1)]  # dropping the sex leaves 0, so it is not named; ties keep the plan's order


def test_a_single_failing_filter_has_no_hints(db):
    a = run(db, {"intent": "search", "filters": [f("age", "gt", 99)]})
    assert a.result.total == 0 and a.result.hints == []


# ---------------------------------------------------------------- semantic search
def test_semantic_search_orders_by_similarity_and_applies_filters(db):
    a = run(db, {"intent": "search", "filters": [], "text_query": "trouble with breath"})
    assert a.result.ids == ["R006"] and a.result.scores["R006"] == pytest.approx(1.0) and "by meaning" in a.summary
    assert a.columns[-1] == "score" and a.rows[0]["score"] == 1.0
    a = run(db, {"intent": "search", "filters": [f("department", "in", ["Cardiology"])], "text_query": "chest"})
    assert sorted(a.result.ids) == ["R001", "R002"]
    a = run(db, {"intent": "search", "filters": [f("department", "in", ["Neurology"])], "text_query": "chest"})
    assert a.result.total == 0 and a.summary == "No records match."


def test_semantic_threshold_and_top_k(db, monkeypatch):
    a = run(db, {"intent": "search", "text_query": "moon"})  # no keyword: close only to the records whose text has none
    assert sorted(a.result.ids) == ["R003", "R004", "R005", "R008", "R009", "R010", "R012"]
    monkeypatch.setattr(s, "SEMANTIC_MIN_SIMILARITY", 0.99)
    a = run(db, {"intent": "search", "text_query": "chest and fever"})  # similar to some, identical to none
    assert a.result.total == 0
    monkeypatch.setattr(s, "SEMANTIC_MIN_SIMILARITY", 0.0)
    monkeypatch.setattr(s, "SEMANTIC_TOP_K", 3)
    assert run(db, {"intent": "search", "text_query": "other"}).result.total == 3


def test_semantic_coverage_is_reported_when_some_records_have_no_vector(db):
    db.create_record(rec(20, "Cardiology", "Male", 50, "2026-11-13", "chest pain", "X"))  # no vector yet
    a = run(db, {"intent": "search", "text_query": "chest"})
    assert a.result.coverage == (11, 13)
    assert any("covers" in n and "no embedding yet" in n for n in a.notes) and "R020" not in a.result.ids


def test_semantic_search_when_the_embedding_model_is_down(db):
    c = Copilot.from_db(db, llm=FakeLLM(), embedder=KeywordEmbedder(down=True), today=TODAY)
    a = c.answer_plan({"intent": "search", "text_query": "chest"})
    assert a.kind == "error" and "meaning" in a.text
    ok = c.answer_plan({"intent": "search", "filters": [f("department", "in", ["Cardiology"])]})  # field searches need no embedding
    assert ok.kind == "results"


# ---------------------------------------------------------------- safety
def test_only_a_single_select_can_run(db):
    ex = Executor(db.con)
    for sql in ("DELETE FROM medical_records", "SELECT 1; DELETE FROM medical_records", "DROP TABLE medical_records",
                "INSERT INTO medical_records (record_id) VALUES ('X')", "UPDATE medical_records SET age = 1", "COPY medical_records TO 'x.csv'"):
        with pytest.raises(ValueError, match="single SELECT"):
            ex._select(sql)
    assert ex._select("SELECT count(*) FROM medical_records") == [(12,)]
    assert db.con.execute("SELECT count(*) FROM medical_records").fetchone()[0] == 12


def test_values_are_parameters_never_part_of_the_sql(db):
    for hostile in ("x'; DROP TABLE medical_records; --", "' OR '1'='1", "Cardiology') OR 1=1 --"):
        a = run(db, {"intent": "search", "filters": [f("department", "in", [hostile])]})
        assert a.kind == "clarify"  # not a department: asked, never run
        a = run(db, {"intent": "search", "filters": [f("final_diagnosis", "contains", hostile)]})
        assert a.kind == "results" and a.result.total == 0
        assert run(db, {"intent": "get_record", "record_id": hostile}).kind == "clarify"
    assert db.con.execute("SELECT count(*) FROM medical_records").fetchone()[0] == 12


def test_like_wildcards_in_the_text_are_literal(db):
    assert run(db, {"intent": "search", "filters": [f("final_diagnosis", "contains", "%")]}).result.total == 0
    assert run(db, {"intent": "search", "filters": [f("final_diagnosis", "contains", "_")]}).result.total == 0
    assert run(db, {"intent": "search", "filters": [f("final_diagnosis", "contains", "\\")]}).result.total == 0


def test_a_read_only_connection_cannot_change_data_or_read_files(tmp_path):
    path = tmp_path / "d.duckdb"
    with RecordDB(path, embedder=KeywordEmbedder()) as d:
        d.create_record(rec(1, "Cardiology", "Male", 50, "2026-01-01", "chest", "X"))
    (tmp_path / "secret.txt").write_text("top secret")
    ex = Executor.open(path)
    try:
        for sql in ("DELETE FROM medical_records", "INSERT INTO medical_records (record_id) VALUES ('Z')", f"SELECT content FROM read_text('{tmp_path / 'secret.txt'}')"):
            with pytest.raises(duckdb.Error):
                ex.con.execute(sql)
        assert ex._select("SELECT count(*) FROM medical_records") == [(1,)]
    finally:
        ex.close()
    with RecordDB(path, embedder=KeywordEmbedder()) as d:  # still intact, and the file can be opened for writing again
        assert d.read_record("R001") is not None


# ================================================================ answers
def test_the_search_answer_says_how_it_was_understood_and_how_many(db):
    a = run(db, {"intent": "search", "filters": [f("department", "in", ["Cardiology"]), f("age", "gt", 60),
                                                  f("lab_results", "has", {"test_name": ["Troponin I"], "flag": ["High"]})]})
    assert a.interpretation == "department: Cardiology · age > 60 · lab: Troponin I / Troponin I (initial) flagged High"
    assert a.summary == "2 records found."
    assert a.text.splitlines()[0] == a.interpretation and "R001 | Patient 1 | Cardiology | 2026-03-10 | 70 | NSTEMI" in a.text
    one = run(db, {"intent": "search", "filters": [f("record_id", "eq", "R001")]})
    assert one.summary == "1 record found."


def test_warnings_reach_the_answer(db):
    a = run(db, {"intent": "search", "filters": [f("encounter_date", "between", {"month": "2020-01"})]})
    assert a.result.total == 0 and any("outside that range" in n for n in a.notes) and "outside that range" in a.text
    a = run(db, {"intent": "search", "filters": [f("lab_results", "has", {"test_name": ["Troponin"]})]})
    assert any("I searched all of them" in n for n in a.notes)


def test_navigation_help_and_refusal_messages(db):
    nav = run(db, {"intent": "navigate", "destination": "upload"})
    assert nav.kind == "navigate" and nav.links[0].url == "/upload"
    rec_nav = run(db, {"intent": "navigate", "destination": "record", "record_id": "R 1/2"})
    assert rec_nav.kind == "clarify"
    nav = run(db, {"intent": "navigate", "destination": "records", "filters": [f("department", "in", ["Cardiology"])]})
    assert nav.links[0].url == "/records?d=Cardiology" and not nav.notes
    nav = run(db, {"intent": "navigate", "destination": "records", "filters": [f("department", "in", ["Cardiology"]), f("age", "gt", 60)]})
    assert nav.links[0].url == "/records?d=Cardiology" and "only filter by department" in nav.text
    assert run(db, {"intent": "navigate", "destination": "records"}).links[0].url == "/records"
    assert "PDF" in run(db, {"intent": "help", "topic": "upload_formats"}).text
    for reason, word in (("write", "only read"), ("export", "cannot export"), ("clinical_advice", "medical advice"), ("unrelated", "medical records")):
        a = run(db, {"intent": "out_of_scope", "reason": reason})
        assert a.kind == "out_of_scope" and word in a.text and a.result is None
    assert run(db, {"intent": "out_of_scope", "reason": "export"}).links[0].url == "/records"


def test_clarify_answers_carry_the_suggestions(db):
    a = run(db, {"intent": "search", "filters": [f("department", "in", ["Cardiologia"])]})
    assert a.kind == "clarify" and a.suggestions == ["Cardiology"] and "Did you mean" in a.text and not a.interpretation


def test_no_number_in_an_answer_can_come_from_anywhere_but_the_result(db):
    a = run(db, {"intent": "aggregate", "filters": [], "aggregate": {"metric": "count", "group_by": "sex"}})
    assert "Female | 6" in a.text and "Male | 5" in a.text and "(none) | 1" in a.text
    a = run(db, {"intent": "aggregate", "filters": [], "aggregate": {"metric": "count"}})
    assert a.summary == "12 records match."


# ================================================================ planner
def test_the_planner_sends_the_catalog_and_the_question(catalog):
    llm = FakeLLM({"intent": "help"})
    plan = Planner(catalog, llm).plan("What can you do?")
    assert plan == {"intent": "help"}
    messages, params = llm.calls[0]
    assert messages[0]["role"] == "system" and "Cardiology (5)" in messages[0]["content"] and "never write SQL" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Question: What can you do?"}
    assert params["temperature"] == 0 and params["response_format"]["json_schema"]["strict"] is True
    assert "department" in json.dumps(params["response_format"]["json_schema"]["schema"])


def test_a_follow_up_sends_the_previous_plan(catalog):
    llm = FakeLLM({"intent": "help"})
    previous = {"intent": "search", "filters": [f("department", "in", ["Cardiology"])]}
    Planner(catalog, llm).plan("only women", previous)
    user = llm.calls[0][0][1]["content"]
    assert "Previous plan: " in user and '"Cardiology"' in user and user.endswith("Follow-up question: only women")


def test_an_unusable_answer_is_retried_once_with_the_error(catalog):
    llm = FakeLLM("I think you want cardiology", '```json\n{"intent": "help"}\n```')
    assert Planner(catalog, llm).plan("hi") == {"intent": "help"}
    retry = llm.calls[1][0]
    assert retry[-2] == {"role": "assistant", "content": "I think you want cardiology"} and "not usable" in retry[-1]["content"]


def test_two_unusable_answers_raise(catalog):
    with pytest.raises(PlannerError, match="no JSON object"):
        Planner(catalog, FakeLLM("nope", "still nope")).plan("hi")


# ================================================================ the whole flow
def test_ask_runs_the_whole_flow(db):
    plan = {"intent": "search", "filters": [f("department", "in", ["Cardiology"]), f("age", "gt", 60)]}
    c = cop(db, plan)
    a = c.ask("Cardiology patients over 60")
    assert a.kind == "results" and ids_of(a) == ["R001", "R003", "R010"] and a.plan == plan


def test_a_follow_up_uses_the_previous_plan(db):
    first = {"intent": "search", "filters": [f("department", "in", ["Cardiology"])]}
    second = {"intent": "search", "filters": [f("department", "in", ["Cardiology"]), f("sex", "eq", "Female")]}
    c = cop(db, first, second)
    a1 = c.ask("Cardiology patients")
    a2 = c.ask("only women", previous=a1.plan)
    assert ids_of(a2) == ["R001", "R003", "R012"] and a2.interpretation == "department: Cardiology · sex: Female"
    assert '"Cardiology"' in c.planner.llm.calls[1][0][1]["content"]


def test_a_misspelling_is_asked_about_not_guessed(db):
    a = cop(db, {"intent": "search", "filters": [f("department", "in", ["Cardiologia"])]}).ask("Cardiologia patients")
    assert a.kind == "clarify" and a.suggestions == ["Cardiology"] and a.result is None


def test_when_the_language_model_is_down_the_user_is_told(db):
    a = cop(db, LLMError("refused")).ask("anything")
    assert a.kind == "error" and "not available" in a.text


def test_a_model_that_never_returns_a_plan_gives_a_polite_clarification(db):
    a = cop(db, "blah", "blah").ask("anything")
    assert a.kind == "clarify" and "rephrase" in a.text


def test_empty_questions_are_not_sent_to_the_model(db):
    c = cop(db)
    assert c.ask("   ").kind == "clarify" and c.planner.llm.calls == []


def test_a_write_request_is_refused_and_nothing_changes(db):
    c = cop(db, {"intent": "out_of_scope", "reason": "write"}, {"intent": "search", "filters": [f("final_diagnosis", "contains", "x'; DELETE FROM medical_records; --")]})
    assert c.ask("Delete record R001").kind == "out_of_scope"
    c.ask("find x'; DELETE FROM medical_records; --")
    assert db.read_record("R001") is not None and db.con.execute("SELECT count(*) FROM medical_records").fetchone()[0] == 12


def test_the_copilot_works_on_its_own_read_only_connection(tmp_path):
    path = tmp_path / "d.duckdb"
    with RecordDB(path, embedder=KeywordEmbedder()) as d:
        for r in SEED[:3]:
            d.create_record(r)
    c = Copilot.open(path, llm=FakeLLM({"intent": "search", "filters": [f("department", "in", ["Cardiology"])]}), today=TODAY)
    try:
        assert c.ask("cardiology").result.total == 3
        assert c.catalog.total_records == 3
        with pytest.raises(duckdb.Error):
            c.executor.con.execute("DELETE FROM medical_records")
    finally:
        c.close()


def test_the_catalog_file_is_used_when_it_exists_and_never_written_by_the_copilot(tmp_path):
    path = tmp_path / "d.duckdb"
    with RecordDB(path, embedder=KeywordEmbedder()) as d:
        d.create_record(SEED[0])
        Catalog.build(d.con).to_yaml(tmp_path / "catalog.yaml")
    before = (tmp_path / "catalog.yaml").read_text()
    c = Copilot.open(path, catalog_path=tmp_path / "catalog.yaml", llm=FakeLLM())
    c.close()
    assert (tmp_path / "catalog.yaml").read_text() == before
    c = Copilot.open(path, catalog_path=tmp_path / "missing.yaml", llm=FakeLLM())  # no file: profiled in memory, nothing written
    c.close()
    assert not (tmp_path / "missing.yaml").exists()


def test_the_planner_sees_the_earlier_turns_of_the_chat(catalog):
    llm = FakeLLM({"intent": "clarify", "question": "x"})
    history = [{"question": "How many tests?", "plan": {"intent": "aggregate", "aggregate": {"metric": "count", "group_by": "lab_results"}},
                "reply": "Which one?"}]
    Planner(catalog, llm).plan("no grouping, just the number", None, history)
    messages = llm.calls[0][0]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "How many tests?" in messages[1]["content"] and "group_by" in messages[2]["content"]
    assert "The user was shown: Which one?" in messages[3]["content"] and "no grouping" in messages[3]["content"]
    assert "How many patients were tested for glucose?" in messages[0]["content"]  # the prompt's example for this kind of question


def test_a_count_of_tested_patients_gives_the_number_at_once(db):
    """The question that used to end in 'I can group by one of...': a bad group_by is dropped, the number is given."""
    plan = {"intent": "aggregate", "filters": [f("lab_results", "has", {"test_name": ["Troponin I"]})],
            "aggregate": {"metric": "count", "field": None, "group_by": "lab_results.test_name", "top_n": None}}
    a = run(db, plan)
    assert a.kind == "aggregate" and a.summary.endswith("match.") and "I cannot group by" in a.text
    assert a.rows == [{"count": a.rows[0]["count"]}] and a.rows[0]["count"] > 0


def test_the_chat_history_is_used_and_a_clarification_can_be_answered(db):
    llm = FakeLLM({"intent": "clarify", "question": "Which department?"},
                  {"intent": "search", "filters": [f("department", "in", ["Cardiology"])]})
    c = Copilot.from_db(db, llm=llm, embedder=KeywordEmbedder(), today=TODAY)
    a1 = c.ask("show patients")
    assert a1.kind == "clarify"
    a2 = c.ask("Cardiology", history=[{"question": "show patients", "plan": a1.plan, "reply": a1.summary}])
    assert a2.kind == "results" and "The user was shown: Which department?" in llm.calls[1][0][-1]["content"]


def test_the_answer_keeps_a_trace_of_everything_that_was_done(db):
    plan = {"intent": "aggregate", "filters": [f("lab_results", "has", {"test_name": ["Troponin"]})],
            "aggregate": {"metric": "count", "field": None, "group_by": "lab_results.test_name", "top_n": None}}
    llm = FakeLLM(plan)
    c = Copilot.from_db(db, llm=llm, embedder=KeywordEmbedder(), today=TODAY)
    a = c.ask("How many patients were tested for troponin?")
    steps = [s["step"] for s in a.trace]
    assert steps[:3] == ["llm", "plan", "validation"] and steps[-1] == "sql"
    call = a.trace[0]
    assert call["reply"] == json.dumps(plan) and call["messages"][-1]["content"] == "Question: How many patients were tested for troponin?"
    sql = a.trace[-1]
    assert sql["sql"].startswith("SELECT count(*) AS value FROM medical_records m WHERE EXISTS") and "?" in sql["sql"]
    assert sql["params"] and all(p in ("Troponin I", "Troponin T", "High-Sensitivity Troponin T") or isinstance(p, str) for p in sql["params"])
    assert sql["rows"] == 1
    validation = a.trace[steps.index("validation")]
    assert validation["result"] == "ok" and any("cannot group by" in w for w in validation["warnings"])
    text = a.explain()
    assert "LLM call 1" in text and "Plan:" in text and "SQL (1 row" in text and "params:" in text


def test_the_trace_shows_unusable_replies_and_a_down_model(db):
    c = Copilot.from_db(db, llm=FakeLLM("not json", "still not json"), embedder=KeywordEmbedder(), today=TODAY)
    a = c.ask("anything")
    assert a.kind == "clarify" and [s["step"] for s in a.trace] == ["llm", "llm"] and "unusable" in a.explain()
    c = Copilot.from_db(db, llm=FakeLLM(LLMError("down")), embedder=KeywordEmbedder(), today=TODAY)
    a = c.ask("anything")
    assert a.kind == "error" and "ERROR" in a.explain() and "LLMError: down" in a.explain()


def test_a_semantic_search_traces_the_embedding_and_hides_the_vector(db):
    class Wide(KeywordEmbedder):
        def embed_query(self, text):
            return super().embed_query(text) + [0.0] * 20  # longer than the 8 values the trace prints

        def embed_documents(self, texts):
            return [v + [0.0] * 20 for v in super().embed_documents(texts)]
    wide = RecordDB(":memory:", embedder=Wide())
    for r in SEED:
        wide.create_record(r)
    wide.embed_pending()
    plan = {"intent": "search", "filters": [], "text_query": "trouble breathing"}
    a = Copilot.from_db(wide, llm=FakeLLM(plan), embedder=Wide(), today=TODAY).ask("trouble breathing")
    wide.close()
    kinds = [s["step"] for s in a.trace]
    assert any(isinstance(p, str) and p.startswith("<list of") for s in a.trace if s["step"] == "sql" for p in s["params"])
    assert "embedding" in kinds and kinds.index("embedding") < kinds.index("sql")
    assert not any(isinstance(p, list) and len(p) > 8 for s in a.trace if s["step"] == "sql" for p in s["params"])


def test_show_more_has_a_trace_without_a_model_call(db):
    plan = {"intent": "search", "filters": [f("department", "in", ["Cardiology"])]}
    a = Copilot.from_db(db, llm=FakeLLM(), embedder=KeywordEmbedder(), today=TODAY).answer_plan(plan)
    assert [s["step"] for s in a.trace][:2] == ["plan", "validation"] and "llm" not in [s["step"] for s in a.trace]


# ---- holes found with a real model: a plan that drops what the user said, and fields that do not apply
BAD_COUNT = {"intent": "aggregate", "filters": [], "aggregate": {"metric": "count", "field": "lab_results.test_name", "group_by": None, "top_n": None},
             "topic": "search_examples"}
GOOD_COUNT = {"intent": "aggregate", "filters": [f("lab_results", "has", {"test_name": ["Troponin"]})],
              "aggregate": {"metric": "count", "field": None, "group_by": None, "top_n": None}}


def test_the_validator_says_what_it_ignores(catalog):
    q = v(BAD_COUNT, catalog)
    assert q.intent == "aggregate" and q.agg_field is None
    assert any("ignored “lab_results.test_name”" in w for w in q.warnings)
    assert any("do not apply here: topic" in w for w in q.warnings)
    assert any("No condition was applied" in w for w in q.warnings)
    # a grouped count over everything is a normal question; a plan with empty extras is clean
    clean = v({"intent": "aggregate", "filters": [], "aggregate": {"metric": "count", "group_by": "sex"}, "topic": None, "record_id": ""}, catalog)
    assert clean.warnings == []


def test_the_catalog_finds_the_values_a_question_names(catalog):
    found = catalog.mentions("How many patients were tested for troponin in Cardiology?")
    assert {"troponin", "cardiology"} <= set(found) and "patients" not in found and "tested" not in found
    assert any(field == "lab_results.test_name" for field, _ in found["troponin"])
    assert catalog.mentions("How many records are there?") == {}


def test_a_plan_that_drops_a_test_is_sent_back_once_and_corrected(db):
    llm = FakeLLM(BAD_COUNT, GOOD_COUNT)
    a = Copilot.from_db(db, llm=llm, embedder=KeywordEmbedder(), today=TODAY).ask("How many patients were tested for troponin?")
    assert a.kind == "aggregate" and a.plan == GOOD_COUNT and "No condition" not in a.text
    assert a.rows[0]["count"] < 12 and len(llm.calls) == 2
    retry = llm.calls[1][0]
    assert [m["role"] for m in retry[-2:]] == ["assistant", "user"] and "does not use “troponin”" in retry[-1]["content"]
    assert [s["step"] for s in a.trace][:4] == ["llm", "llm", "check", "plan"] and a.trace[2]["action"] == "asked the model again"


def test_if_the_model_still_drops_it_the_answer_says_so(db):
    a = Copilot.from_db(db, llm=FakeLLM(BAD_COUNT, BAD_COUNT), embedder=KeywordEmbedder(), today=TODAY).ask("How many patients were tested for troponin?")
    assert a.kind == "aggregate" and "12 records match." in a.summary
    assert "mentions “troponin”, but no condition uses it" in a.text


def test_a_failed_retry_keeps_the_first_plan_with_a_note(db):
    llm = FakeLLM(BAD_COUNT, LLMError("down"))
    a = Copilot.from_db(db, llm=llm, embedder=KeywordEmbedder(), today=TODAY).ask("How many patients were tested for troponin?")
    assert a.kind == "aggregate" and "no condition uses it" in a.text and a.trace[2]["action"] == "kept the plan"


def test_a_plan_that_uses_what_the_user_said_is_not_questioned(db):
    llm = FakeLLM(GOOD_COUNT)
    a = Copilot.from_db(db, llm=llm, embedder=KeywordEmbedder(), today=TODAY).ask("How many patients were tested for troponin?")
    assert len(llm.calls) == 1 and "check" not in [s["step"] for s in a.trace]


def test_a_word_covered_by_a_field_the_plan_uses_is_not_missing(db, monkeypatch):
    """"Oxygen saturation below 92%" names a lab test AND the SpO2 vital sign; filtering on the vital sign is a correct plan."""
    c = Copilot.from_db(db, llm=FakeLLM(), embedder=KeywordEmbedder(), today=TODAY)
    monkeypatch.setattr(c.catalog, "mentions", lambda q: {"oxygen": [("lab_results.test_name", "Oxygen Saturation (room air)")]})
    assert c._missing("Oxygen saturation below 92%", {"intent": "search", "filters": [f("vital_signs.sp02", "lt", 92)]}) == []
    assert c._missing("Oxygen saturation below 92%", {"intent": "search", "filters": [f("sex", "eq", "Male")]}) == ["oxygen"]


# ================================================================ corrections after the first runs with real models
def test_a_period_written_as_last_n_days_is_understood():
    assert dates.expand({"last_n_days": 30}, TODAY) == dates.expand({"token": "last_n_days", "n": 30}, TODAY)
    with pytest.raises(dates.BadPeriod):
        dates.expand({"last_n_days": "many"}, TODAY)


def test_a_record_count_copied_from_the_catalog_is_dropped(catalog):
    q = v({"intent": "search", "filters": [f("department", "in", ["Cardiology (5)"])]}, catalog)
    assert q.filters[0].value == ["Cardiology"]
    q = v({"intent": "search", "filters": [f("lab_results", "has", {"test_name": ["Troponin I (4)"]})]}, catalog)
    assert q.filters[0].value["test_name"][0] == "Troponin I"
    q = v({"intent": "search", "filters": [f("lab_results", "has", {"test_name": ["Troponin I (initial)"]})]}, catalog)
    assert q.filters[0].value["test_name"] == ["Troponin I (initial)"]  # a real parenthesis stays
    assert v({"intent": "search", "filters": [f("department", "in", ["Nowhere (3)"])]}, catalog).intent == "clarify"


def test_the_prompt_gives_todays_date_and_no_literal_year_to_copy(catalog):
    text = Planner(catalog, FakeLLM(), today=date(2026, 11, 13)).messages("patients this year")[0]["content"]
    assert "Today is 2026-11-13" in text and '{"year": 2025}' not in text and "{TODAY}" not in text
    assert "SYNUS-2026-0001" not in text and "never the number in parentheses" in text


def test_a_partial_record_id_finds_its_record(db):
    db.con.execute("UPDATE medical_records SET record_id = 'SYNUS-2026-0007' WHERE record_id = 'R007'")
    db.con.execute("UPDATE medical_records SET record_id = 'SYNUS-2025-0007' WHERE record_id = 'R008'")
    db.con.execute("UPDATE medical_records SET record_id = 'SYNUS-2026-0003' WHERE record_id = 'R003'")
    a = run(db, {"intent": "get_record", "record_id": "0003"})
    assert a.kind == "record" and a.result.record["record_id"] == "SYNUS-2026-0003"
    a = run(db, {"intent": "get_record", "record_id": "0007"})  # fits two records: ask which
    assert a.kind == "clarify" and a.suggestions == ["SYNUS-2025-0007", "SYNUS-2026-0007"]
    assert run(db, {"intent": "get_record", "record_id": "9999"}).summary == "There is no record 9999."


def test_semantic_hits_far_below_the_best_one_are_cut(db, monkeypatch):
    monkeypatch.setattr(s, "SEMANTIC_MIN_SIMILARITY", 0.0)
    monkeypatch.setattr(s, "SEMANTIC_RELATIVE_MARGIN", None)
    wide = run(db, {"intent": "search", "text_query": "chest and fever"}).result.total
    monkeypatch.setattr(s, "SEMANTIC_RELATIVE_MARGIN", 0.01)
    narrow = run(db, {"intent": "search", "text_query": "chest and fever"})
    assert 0 < narrow.result.total < wide
    best = max(narrow.result.scores.values())
    assert all(sc >= best - 0.01 for sc in narrow.result.scores.values())
    assert any(st.get("step") == "semantic_filter" and st["kept"] == narrow.result.total for st in narrow.trace)
