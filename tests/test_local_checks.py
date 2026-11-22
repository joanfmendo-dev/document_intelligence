"""The local-checks script, run end to end with fake models on a small database (the real models are what the script is for)."""
import json

import duckdb
import pytest

from docintel import RecordDB
from docintel.evaluation import local_checks
from fakes import SEED, FakeLLM, KeywordEmbedder  # noqa: F401


class ClarifyLLM:
    """A text model that always asks to clarify: enough to run the golden set through the machinery."""

    def run(self, messages, **params):
        last = messages[-1]["content"]
        if "Reply with only this JSON" in last:
            return '{"ok": true}'
        return json.dumps({"intent": "clarify", "question": "Which one?"})


class Judge:
    def run(self, messages, **params):
        return "SCORE: 1" if "TEXT SHOWN" in messages[-1]["content"] or "SCORE" in messages[-1]["content"] else "SCORE: 1"


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "m.duckdb"
    with RecordDB(path) as db:
        for r in SEED:
            db.create_record(r)
    return path


def run(source, tmp_path, *extra, **kw):
    return local_checks.run(["--db", str(source), "--out", str(tmp_path / "out"), *extra], llm=kw.pop("llm", ClarifyLLM()),
                            embedder=kw.pop("embedder", KeywordEmbedder()), **kw)


def test_all_steps_run_and_the_report_is_written(source, tmp_path):
    report = run(source, tmp_path, "--no-judge")
    names = {c.name: c.status for c in report.checks}
    assert names["models: embeddings"] in ("PASS", "WARN") and names["models: text model"] == "PASS"
    assert names["embeddings: full run"] == "PASS" and names["embeddings: second run"] == "PASS"
    assert names["embeddings: dimensions"] == "PASS" and names["embeddings: save/delete hooks"] == "PASS"
    assert names["embeddings: interrupted run"] == "PASS"
    assert names["semantic: scores"] == "INFO" and names["planner: golden set"] in ("PASS", "WARN")
    assert names["models: judge"] == "SKIP" and names["judge: calibration"] == "SKIP"
    out = tmp_path / "out"
    assert "Local checks" in (out / "report.md").read_text() and json.loads((out / "eval_report.json").read_text())["summary"]["cases"] > 0
    assert not [c for c in report.checks if c.status == "FAIL"]


def test_runs_repeats_the_golden_set_and_reports_the_stability(source, tmp_path):
    report = run(source, tmp_path, "--no-judge", "--steps", "planner", "--runs", "2")
    detail = next(c for c in report.checks if c.name == "planner: golden set")
    assert any(line.startswith("stability over 2 runs:") for line in detail.lines)


def test_the_original_database_is_not_touched_unless_asked(source, tmp_path):
    run(source, tmp_path, "--no-judge", "--steps", "embeddings")
    con = duckdb.connect(str(source), read_only=True)
    assert con.execute("SELECT count(*) FROM record_embeddings").fetchone()[0] == 0
    con.close()
    run(source, tmp_path, "--no-judge", "--steps", "embeddings", "--in-place")
    con = duckdb.connect(str(source), read_only=True)
    assert con.execute("SELECT count(*) FROM record_embeddings").fetchone()[0] == len(SEED) - 1  # the seed has one record with no text to embed
    con.close()


def test_vectors_that_already_exist_are_freed_in_the_copy_so_the_interrupted_run_is_tested(source, tmp_path):
    run(source, tmp_path, "--no-judge", "--steps", "embeddings", "--in-place")  # the source now has all its vectors
    report = run(source, tmp_path, "--no-judge", "--steps", "embeddings")
    assert {c.name: c.status for c in report.checks}["embeddings: interrupted run"] == "PASS"
    report = run(source, tmp_path, "--no-judge", "--steps", "embeddings", "--rebuild-embeddings")
    assert {c.name: c.status for c in report.checks}["embeddings: full run"] == "PASS"
    con = duckdb.connect(str(source), read_only=True)
    assert con.execute("SELECT count(*) FROM record_embeddings").fetchone()[0] == len(SEED) - 1  # the source was not touched
    con.close()
    report = run(source, tmp_path, "--no-judge", "--in-place", "--rebuild-embeddings")
    assert [c for c in report.checks if c.status == "FAIL" and "copy" in c.detail]


def test_the_judge_is_calibrated_when_there_is_one(source, tmp_path):
    report = run(source, tmp_path, judge_llm=Judge())
    names = {c.name: c.status for c in report.checks}
    assert names["models: judge"] == "PASS" and names["judge: calibration"] in ("PASS", "WARN", "SKIP")


def test_a_dead_embedding_model_is_reported_and_its_steps_skipped(source, tmp_path):
    from docintel.shared.embeddings import EmbeddingError

    class Down(KeywordEmbedder):
        def embed_documents(self, texts):
            raise EmbeddingError("down")

        def embed_query(self, text):
            raise EmbeddingError("down")

    report = run(source, tmp_path, "--no-judge", "--steps", "models", "embeddings", embedder=Down())
    names = {c.name: c.status for c in report.checks}
    assert names["models: embeddings"] == "FAIL" and names["embeddings: run"] == "SKIP"


def test_a_missing_database_is_a_clear_failure(tmp_path):
    report = local_checks.run(["--db", str(tmp_path / "nope.duckdb"), "--out", str(tmp_path / "o")], llm=ClarifyLLM(), embedder=KeywordEmbedder())
    assert [(c.name, c.status) for c in report.checks] == [("database", "FAIL")]


def test_the_calibration_verdict_depends_on_catching_corrupted_answers(tmp_path):
    from docintel.evaluation.judge import Judge as RealJudge

    class Good:   # reads the answer: right only if it says 7 records and claims nothing else
        def run(self, messages, **params):
            text = messages[-1]["content"].split("TEXT SHOWN TO THE USER:")[-1]
            return "SCORE: 1" if "7 records match." in text and "%" not in text else "SCORE: 0"

    class Lenient:  # approves everything
        def run(self, messages, **params):
            return "SCORE: 1"

    samples = [('{"total": 7}', "7 records match.")] * 4
    report = local_checks.Report(tmp_path)
    local_checks.check_judge(report, RealJudge(llm=Good()), samples)
    local_checks.check_judge(report, RealJudge(llm=Lenient()), samples)
    assert [c.status for c in report.checks] == ["PASS", "WARN"] and "do not trust" in report.checks[1].detail


def test_the_console_and_the_markdown_carry_the_colour_code(tmp_path, capsys):
    report = local_checks.Report(tmp_path, color=True)
    for status in ("PASS", "FAIL", "WARN", "SKIP", "INFO"):
        report.add(local_checks.Check(f"models: {status.lower()} case", status, "detail"))
    shown = capsys.readouterr().out
    assert "\033[1;34m▌" in shown                                     # titles: blue
    assert "\033[1;32mPASS" in shown and "\033[1;31mFAIL" in shown    # pass green, fail red
    assert shown.count("\033[1;33m") == 2                             # warn and skip: yellow
    assert "\033[1;34mINFO" in shown
    md = report.markdown()
    assert "🟢 **1 passed** · 🔴 **1 failed** · 🟡 **1 warnings** · 🟡 **1 skipped**" in md
    assert "### 🔵 pass case" in md and "🟢 **PASS**" in md and "🔴 **FAIL**" in md and md.count("🟡 **WARN**") == 2 and md.count("🟡 **SKIP**") == 2  # table row and section
    assert "## 🔵 1 · Models" in md and "| 🔴 | 🔵 1 · Models — fail case | 🔴 **FAIL** | detail |" in md


def test_there_are_no_escape_codes_without_colour(tmp_path, capsys):
    report = local_checks.Report(tmp_path, color=False)
    report.add(local_checks.Check("models: x", "PASS", "ok"))
    assert "\033" not in capsys.readouterr().out
    assert local_checks.use_color("never") is False and local_checks.use_color("always") is True
