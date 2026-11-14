"""The evaluation: level 1 (execution accuracy), level 2 (LLM judge), the golden set and the runner. No real model is used."""
import json
import re
from datetime import date

import pytest
from test_copilot import SEED, FakeLLM, KeywordEmbedder, f  # the same small database as the copilot tests

from docintel.copilot import Copilot
from docintel.copilot.plan import INTENTS
from docintel.database import RecordDB
from docintel.evaluation import Golden, GoldenCase, Judge, calibrate, evaluate, execution_verdict, load_golden, wilson
from docintel.evaluation.execution import compare_aggregate, compare_list, ids_digest, values_equal
from docintel.evaluation.judge import add_claim, corrupt_number, judge_view

TODAY = date(2026, 11, 13)


# ================================================================ level 1
def test_values_are_compared_like_the_reference_project():
    assert values_equal(5, 5) and not values_equal(5, 6) and not values_equal(5, 5.5)
    assert values_equal(62.3, 62.33) and not values_equal(62.3, 62.4)  # at the coarser precision
    assert values_equal(" Cardiology", "cardiology") and values_equal(None, None) and not values_equal(None, 0)
    assert values_equal(3, 3.0) and not values_equal(3, 3.2) and values_equal(51.4, 51.4)


def test_aggregate_levels():
    exp = [["Cardiology", 5], ["Neurology", 2]]
    assert compare_aggregate(exp, [["Cardiology", 5], ["Neurology", 2]]).level == "exact"
    assert compare_aggregate(exp, [["Neurology", 2], ["Cardiology", 5]]).level == "equivalent"
    assert compare_aggregate([["a", 62.3]], [["a", 62.33]]).level == "equivalent"
    assert compare_aggregate([["Cardiology", 5], ["Neurology", 2]], [["Cardiology", 5], ["Urology", 2]]).level == "mismatch"
    assert compare_aggregate([[0]], []).detail == "agent returned no rows; the correct answer is 0"
    assert compare_aggregate([["a", 1]], [["a", 1], ["b", 1]]).level == "mismatch"


def test_top_n_ties_at_the_cut_off_are_equivalent_but_other_differences_are_not():
    exp = [["A", 10], ["B", 5], ["C", 3]]
    assert compare_aggregate(exp, [["A", 10], ["B", 5], ["D", 3]], ordered_limit=True).level == "equivalent_tie"
    assert compare_aggregate(exp, [["A", 10], ["X", 5], ["C", 3]], ordered_limit=True).level == "mismatch"  # B and X are not at the cut-off
    assert compare_aggregate(exp, [["A", 10], ["B", 5], ["D", 3]], ordered_limit=False).level == "mismatch"


def test_list_comparison_uses_the_total_and_the_set_of_ids():
    exp = {"total": 3, "ids_md5": ids_digest(["a", "b", "c"])}
    assert compare_list(exp, {"total": 3, "ids": ["c", "a", "b"]}).level == "exact"  # order does not matter
    assert "total 2 instead of 3" in compare_list(exp, {"total": 2, "ids": ["a", "b"]}).detail
    assert compare_list(exp, {"total": 3, "ids": ["a", "b", "x"]}).detail == "same total but different records"
    assert compare_list(exp, None).detail == "agent ran no query"
    assert compare_list({"total": 0, "ids_md5": ids_digest([])}, {"total": 0, "ids": []}).level == "exact"
    assert compare_list({"total": 0, "ids_md5": ids_digest([])}, {"total": 1, "ids": ["a"]}).level == "mismatch"  # empty is not the same as one


def test_execution_verdict_for_each_kind_of_case():
    exp = {"total": 1, "ids_md5": ids_digest(["a"])}
    assert execution_verdict(None, "search", None).level == "n/a"          # a person must label it
    assert execution_verdict(exp, "clarify", None).level == "n/a"          # no query expected
    assert execution_verdict(exp, "search", None).level == "mismatch"      # a query was expected and none ran
    assert execution_verdict(exp, "search", {"kind": "aggregate", "rows": [], "columns": []}).level == "mismatch"
    assert execution_verdict(exp, "get_record", {"kind": "record", "total": 1, "ids": ["a"]}).level == "exact"
    agg = execution_verdict({"rows": [["F", 6]]}, "aggregate", {"kind": "aggregate", "columns": ["sex", "count"], "rows": [{"sex": "F", "count": 6}]})
    assert agg.level == "exact" and agg.score == 1
    assert execution_verdict({"rows": [["F", 6]]}, "aggregate", {"kind": "search", "total": 1, "ids": ["a"]}).score == 0


def test_the_wilson_interval():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(10, 10)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.7225, abs=1e-3)
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi and lo == pytest.approx(0.2366, abs=1e-3)


# ================================================================ level 2
class ScriptedJudge:
    """A fake judge model. `answers` are replies in order; the last repeats."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def run(self, messages, **params):
        self.calls.append(messages)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def test_the_judge_reads_the_last_score_and_retries_an_unreadable_reply():
    assert Judge(ScriptedJudge("Looks right.\nSCORE: 1")).fidelity("{}", "x") == 1
    assert Judge(ScriptedJudge("SCORE: 1 was my first thought, but no.\nSCORE: 0")).fidelity("{}", "x") == 0
    llm = ScriptedJudge("I cannot decide", "SCORE: 1")
    assert Judge(llm).fidelity("{}", "x") == 1 and len(llm.calls) == 2
    llm = ScriptedJudge("no score at all")
    assert Judge(llm).fidelity("{}", "x") is None and len(llm.calls) == 2  # one retry, then None (not a failure)
    assert Judge(ScriptedJudge("SCORE: 7"), retries=0).fidelity("{}", "x") is None


def test_each_criterion_gets_its_own_prompt_with_only_what_it_needs():
    llm = ScriptedJudge("SCORE: 1")
    j = Judge(llm)
    j.fidelity('{"total": 5}', "5 records found.")
    j.interpretation("Cardiology patients over 60", "department: Cardiology", previous='{"intent": "search"}', today="2026-11-13")
    fid, interp = llm.calls
    assert "execution result is the sole source of truth" in fid[0]["content"].lower() and "EXECUTION RESULT:\n{\"total\": 5}" in fid[1]["content"]
    assert "USER QUESTION" not in fid[1]["content"]
    assert "USER QUESTION:\nCardiology patients over 60" in interp[1]["content"] and "PREVIOUS SEARCH" in interp[1]["content"]
    assert "TODAY: 2026-11-13" in interp[1]["content"] and "EXECUTION RESULT" not in interp[1]["content"]


def test_the_judge_view_is_compact():
    assert json.loads(judge_view({"kind": "search", "total": 30, "ids": ["x"] * 30, "rows": [{"record_id": "a"}]})) == {"total": 30, "showing": [{"record_id": "a"}]}
    agg = json.loads(judge_view({"kind": "aggregate", "rows": [{"sex": "F", "count": 6}], "total": 1}))
    assert agg == {"groups_or_value": [{"sex": "F", "count": 6}]}
    assert json.loads(judge_view({"kind": "search", "total": 0, "rows": [], "hints": [{"without": "age > 85", "matches": 2}]}, ["a note"]))["notes_shown_to_the_user"] == ["a note"]
    assert judge_view(None) == "{}"


class NumberCheckingJudge:
    """A rule-based stand-in: scores 0 when the text has a number the result does not contain."""

    def run(self, messages, **params):
        user = messages[-1]["content"]
        result, text = user.split("TEXT SHOWN TO THE USER:\n")
        known = set(re.findall(r"\d+", result.split("EXECUTION RESULT:\n")[1]))
        ok = set(re.findall(r"\d+", text)) <= known
        return f"checked\nSCORE: {1 if ok else 0}"


def test_corrupting_answers_for_the_controls():
    assert corrupt_number("5 records found.") == "12 records found."
    assert corrupt_number("No records match.") is None
    assert "12%" in add_claim("5 records found.")


def test_calibration_measures_detection_and_false_alarms():
    samples = [('{"total": 5, "showing": []}', "5 records found."), ('{"total": 2, "showing": []}', "2 records found.")]
    report = calibrate(Judge(NumberCheckingJudge()), samples)
    assert report == {"false_alarms": "0/2", "detected": {"wrong_number": "2/2", "unsupported_claim": "2/2"}, "unparsed": 0}
    blind = calibrate(Judge(ScriptedJudge("SCORE: 1")), samples)  # a judge that approves everything catches nothing
    assert blind["detected"] == {"wrong_number": "0/2", "unsupported_claim": "0/2"} and blind["false_alarms"] == "0/2"
    assert calibrate(Judge(ScriptedJudge("???")), samples)["unparsed"] == 6


# ================================================================ the golden set
def test_the_packaged_golden_set_is_well_formed():
    g = load_golden()
    assert g.as_of == date(2026, 11, 13) and len(g.cases) == 58
    assert len({c.id for c in g.cases}) == 58
    for c in g.cases:
        assert c.question.strip() and c.intent in INTENTS, c.id
        assert set(vars(c)) == {"id", "question", "expected_plan", "expected_result", "previous_plan"}  # only the essential fields
        if c.intent in ("clarify", "out_of_scope", "navigate", "help"):
            assert c.expected_result is None, c.id
        if c.previous_plan:
            assert c.id.startswith("F")
    assert sum(1 for c in g.cases if c.previous_plan) == 5
    assert {c.id for c in g.cases if c.intent in ("search", "get_record", "aggregate") and c.expected_result is None} == {"I01", "I02", "I03", "M01", "M02", "M03"}
    assert load_golden().cases[0].expected_result == g.cases[0].expected_result


# ================================================================ the runner
@pytest.fixture
def db():
    d = RecordDB(":memory:", embedder=KeywordEmbedder())
    for r in SEED:
        d.create_record(r)
    d.embed_pending()
    yield d
    d.close()


class PlanByQuestion:
    """A fake planner model: the plan for each question is given."""

    def __init__(self, plans):
        self.plans = plans

    def run(self, messages, **params):
        reply = self.plans[messages[-1]["content"].split("uestion: ")[-1].strip()]
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply)


CARDIO = {"intent": "search", "filters": [f("department", "in", ["Cardiology"])]}
CARDIO_IDS = ["R001", "R002", "R003", "R010", "R012"]


def golden():
    return Golden(TODAY, [
        GoldenCase("A", "Cardiology patients", CARDIO, {"total": 5, "ids_md5": ids_digest(CARDIO_IDS)}),
        GoldenCase("B", "How many per sex?", {"intent": "aggregate", "filters": [], "aggregate": {"metric": "count", "group_by": "sex"}},
                   {"rows": [["Female", 6], ["Male", 5], [None, 1]]}),
        GoldenCase("C", "Neurology patients", {"intent": "search", "filters": [f("department", "in", ["Neurology"])]},
                   {"total": 2, "ids_md5": ids_digest(["R004", "R005"])}),
        GoldenCase("D", "Delete record R001", {"intent": "out_of_scope", "reason": "write"}),
        GoldenCase("E", "Show me recent patients", {"intent": "clarify"}),
        GoldenCase("F", "Only women", {"intent": "search", "filters": [f("department", "in", ["Cardiology"]), f("sex", "eq", "Female")]},
                   {"total": 3, "ids_md5": ids_digest(["R001", "R003", "R012"])}, previous_plan=CARDIO),
        GoldenCase("G", "Trouble breathing", {"intent": "search", "text_query": "breath"}),
        GoldenCase("H", "Take me to upload", {"intent": "navigate", "destination": "upload"}),
    ])


PLANS = {
    "Cardiology patients": CARDIO,
    "How many per sex?": {"intent": "aggregate", "filters": [], "aggregate": {"metric": "count", "group_by": "sex"}},
    "Neurology patients": {"intent": "search", "filters": [f("department", "in", ["Urology"])]},  # wrong department: right shape, wrong data
    "Delete record R001": {"intent": "out_of_scope", "reason": "write"},
    "Show me recent patients": {"intent": "search", "filters": [f("age", "gt", 1)]},             # should have asked
    "Only women": {"intent": "search", "filters": [f("department", "in", ["Cardiology"]), f("sex", "eq", "Female")]},
    "Trouble breathing": {"intent": "search", "text_query": "breath"},
    "Take me to upload": {"intent": "navigate", "destination": "upload"},
}


def copilot(db, plans=None):
    return Copilot.from_db(db, llm=PlanByQuestion(plans or PLANS), embedder=KeywordEmbedder(), today=None)


def test_level_one_scores_each_kind_of_case(db):
    report = evaluate(copilot(db), golden())
    by = {c.id: c for c in report.cases}
    assert (by["A"].level1, by["A"].plan_match, by["A"].passed) == ("exact", True, True)
    assert (by["B"].level1, by["B"].passed) == ("exact", True)
    assert by["C"].level1 == "mismatch" and by["C"].passed is False and by["C"].plan_match is False and "total 2 instead of 2" not in by["C"].level1_detail
    assert by["D"].level1 == "n/a" and by["D"].intent_ok is True and by["D"].passed is True
    assert by["E"].intent_ok is False and by["E"].passed is False   # a search where a question was due
    assert by["F"].level1 == "exact" and by["F"].passed is True     # the follow-up used the previous plan
    assert by["G"].passed is None and by["G"].level1 == "n/a"       # a person must label semantic answers
    assert by["H"].intent_ok is True
    s = report.summary()
    assert s["execution_accuracy"]["correct"] == 3 and s["execution_accuracy"]["of"] == 4
    assert s["intent_accuracy_no_query_cases"]["correct"] == 2 and s["intent_accuracy_no_query_cases"]["of"] == 3
    assert s["end_to_end"]["correct"] == 5 and s["end_to_end"]["of"] == 7 and s["not_scored"] == ["G"]
    assert [x["id"] for x in s["failed"]] == ["C", "E"]


def test_the_golden_date_is_used_for_the_run_and_restored(db):
    c = copilot(db)
    evaluate(c, golden(), only=["A"])
    assert c.today is None
    seen = []
    c.executor.run = lambda q, page=1, _run=c.executor.run: (seen.append(c.today), _run(q, page))[1]
    evaluate(c, golden(), only=["A"])
    assert seen == [TODAY] and c.today is None


def test_only_and_progress(db):
    calls = []
    report = evaluate(copilot(db), golden(), only=["A", "H"], progress=lambda i, n: calls.append((i, n)))
    assert [c.id for c in report.cases] == ["A", "H"] and calls == [(1, 2), (2, 2)]


def test_a_crash_is_a_failed_case_not_a_failed_run(db):
    plans = {**PLANS, "Cardiology patients": RuntimeError("boom")}
    report = evaluate(copilot(db, plans), golden(), only=["A", "B"])
    assert report.cases[0].error == "RuntimeError: boom" and report.cases[0].passed is False and report.cases[1].passed is True


def test_the_judge_adds_level_two(db):
    class Judges:
        """Fidelity fails when the text shows a "(none)" group; interpretation fails for the question about women."""
        def run(self, messages, **params):
            system, user = messages[0]["content"], messages[1]["content"]
            if "faithfully represents" in system:
                return "ok\nSCORE: " + ("0" if "(none)" in user.split("TEXT SHOWN")[1] else "1")
            return "ok\nSCORE: " + ("0" if "Only women" in user else "1")
    report = evaluate(copilot(db), golden(), Judge(Judges()), only=["A", "B", "F", "D"])
    by = {c.id: c for c in report.cases}
    assert (by["A"].judge_ran, by["A"].judge_fidelity, by["A"].judge_interpretation, by["A"].passed) == (True, 1, 1, True)
    assert (by["B"].level1, by["B"].judge_fidelity, by["B"].passed) == ("exact", 0, False)       # data right, text judged unfaithful
    assert (by["F"].level1, by["F"].judge_interpretation, by["F"].passed) == ("exact", 0, False)  # data right, understanding judged wrong
    assert by["D"].judge_ran is False and by["D"].passed is True                                   # a refusal has no data to judge
    s = report.summary()
    assert (s["judge_fidelity"]["correct"], s["judge_fidelity"]["of"]) == (2, 3) and (s["judge_interpretation"]["correct"], s["judge_interpretation"]["of"]) == (2, 3)
    assert s["level1_vs_judge"] == {"level1 pass + judge pass": 1, "level1 pass + judge fail": 2, "level1 fail + judge pass": 0, "level1 fail + judge fail": 0}


def test_unreadable_judge_replies_are_counted_apart_and_never_pass(db):
    report = evaluate(copilot(db), golden(), Judge(ScriptedJudge("no idea")), only=["A"])
    case = report.cases[0]
    assert (case.judge_fidelity, case.judge_interpretation, case.passed) == (None, None, False)
    assert report.summary()["judge_unparsed"] == 2 and report.summary()["judge_fidelity"]["of"] == 0


def test_the_report_formats_and_saves(db, tmp_path):
    report = evaluate(copilot(db), golden(), Judge(ScriptedJudge("SCORE: 1")))
    text = report.format()
    assert "Execution accuracy: 3/4 = 75.0%" in text and "95% Wilson CI" in text and "Correct end to end: 5/7" in text
    assert "Judge, fidelity" in text and "not scored" in text and "intent WRONG" in text and "mismatch" in text
    assert "Judge" not in evaluate(copilot(db), golden(), only=["A"]).format()
    report.save(tmp_path / "r.json")
    saved = json.loads((tmp_path / "r.json").read_text())
    assert saved["summary"]["cases"] == 8 and saved["cases"][0]["id"] == "A" and saved["cases"][0]["level1"] == "exact"
