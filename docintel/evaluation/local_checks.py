"""Checks to run once on your own machine, with the real models (Ollama, the embedding model and the commercial judge).

    python -m docintel.evaluation.local_checks --db medical_records.duckdb

Runs, in order, and writes `local_checks/report.md` (what to read) and `local_checks/eval_report.json` (every case):

  1. models      the embedding model, the text model (also: does it obey a JSON schema?) and the judge answer
  2. embeddings  vectors for every record, an interrupted run that resumes, a second run that does nothing, save/delete hooks
  3. semantic    similarity scores of a few probe questions, to choose SEMANTIC_MIN_SIMILARITY
  4. planner     the golden set (level 1: execution accuracy) with the real text model, timing and retries
  5. judge       the judge on the same cases and its calibration against deliberately corrupted answers

The database is COPIED first (`local_checks/work.duckdb`); your file is only changed with --in-place. A step that fails does not stop the
others unless they depend on it. Choose steps with --steps (e.g. --steps models embeddings). The judge is used only on the synthetic
golden set and only when a key is configured (DOCINTEL_JUDGE_API_KEY, or OPENAI_API_KEY / ANTHROPIC_API_KEY); --no-judge turns it off.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..database.records import RecordDB
from ..shared.embeddings import Embedder, EmbeddingError
from ..shared.llm import LLM
from ..shared.settings import s
from .golden import load_golden
from .judge import Judge, calibrate, judge_view
from .runner import evaluate

TOKENS = 400  # room for a model that thinks before it answers (50 left some of them with an empty reply)
SAMPLE_PENDING = 200  # vectors deleted in the copy: well over the 2 batches the interrupted run lets through
STEPS = ("models", "embeddings", "semantic", "planner", "judge")
PROBES = ("trouble breathing at night", "chest pain when walking", "stomach ache after eating", "high blood sugar",
          "headache with nausea", "ankle injury after a fall")
BAD_PLAN_SCHEMA = {"type": "object", "properties": {"intent": {"type": "string", "enum": ["search", "aggregate"]}}, "required": ["intent"]}


@dataclass
class Check:
    name: str
    status: str                 # PASS | WARN | FAIL | SKIP | INFO
    detail: str = ""
    lines: list[str] = field(default_factory=list)


# colours: titles blue, PASS green, FAIL red, WARN and SKIP yellow, INFO blue
STYLE = {"PASS": ("32", "🟢"), "FAIL": ("31", "🔴"), "WARN": ("33", "🟡"), "SKIP": ("33", "🟡"), "INFO": ("34", "🔵")}
GROUP_TITLES = {"environment": "0 · Environment", "database": "Database", "models": "1 · Models", "embeddings": "2 · Embeddings",
                "semantic": "3 · Semantic search", "planner": "4 · Planner (golden set)", "judge": "5 · Judge"}


def title_of(check: "Check") -> str:
    group, _, sub = check.name.partition(": ")
    group_title = GROUP_TITLES.get(group, group)
    return f"{group_title} — {sub}" if sub else group_title


def use_color(mode: str, stream: Any = None) -> bool:
    """auto: colours on a terminal (and not when NO_COLOR is set); always / never override that."""
    if mode != "auto":
        return mode == "always"
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)()) and not os.getenv("NO_COLOR") or bool(os.getenv("FORCE_COLOR"))


def paint(text: str, code: str, on: bool, bold: bool = False) -> str:
    return f"\033[{'1;' if bold else ''}{code}m{text}\033[0m" if on else text


class Report:
    def __init__(self, out: Path, color: bool = False):
        self.out, self.color, self.checks = out, color, []
        if color and os.name == "nt":
            os.system("")  # makes the Windows console understand ANSI colours

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        code = STYLE.get(check.status, ("0", ""))[0]
        group = check.name.partition(": ")[0]
        if len(self.checks) == 1 or group != self.checks[-2].name.partition(": ")[0]:
            print("\n" + paint(f"━━ {GROUP_TITLES.get(group, group)} ".ljust(60, "━"), "34", self.color, bold=True), flush=True)
        print(paint(f"▌{title_of(check).partition(' — ')[2] or title_of(check)}", "34", self.color, bold=True), flush=True)
        print(f"   {paint(check.status.ljust(4), code, self.color, bold=True)} {paint(check.detail, code, self.color)}", flush=True)
        for line in check.lines:
            print("      " + line, flush=True)
        return check

    def failed(self, name: str) -> bool:
        return any(c.name == name and c.status == "FAIL" for c in self.checks)

    def counts(self) -> dict[str, int]:
        return {k: sum(c.status == k for c in self.checks) for k in ("PASS", "FAIL", "WARN", "SKIP", "INFO")}

    def summary_line(self) -> str:
        n = self.counts()
        return "  ".join(paint(f"{k} {n[k]}", STYLE[k][0], self.color, bold=True) for k in ("PASS", "WARN", "SKIP", "FAIL") if n[k] or k in ("PASS", "FAIL"))

    def markdown(self) -> str:
        """Markdown has no colour, so each result carries a coloured circle: 🟢 PASS, 🔴 FAIL, 🟡 WARN or SKIP, 🔵 titles and notes."""
        n = self.counts()
        out = ["# 🔵 Local checks", "",
               f"🟢 **{n['PASS']} passed** · 🔴 **{n['FAIL']} failed** · 🟡 **{n['WARN']} warnings** · 🟡 **{n['SKIP']} skipped**", "",
               "| | Check | Result | Detail |", "|---|---|---|---|"]
        out += [f"| {STYLE[c.status][1]} | 🔵 {title_of(c)} | {STYLE[c.status][1]} **{c.status}** | {c.detail.replace('|', '/')} |" for c in self.checks]
        group = None
        for c in self.checks:
            if c.name.partition(": ")[0] != group:
                group = c.name.partition(": ")[0]
                out += ["", f"## 🔵 {GROUP_TITLES.get(group, group)}"]
            out += ["", f"### 🔵 {title_of(c).partition(' — ')[2] or title_of(c)}", "", f"{STYLE[c.status][1]} **{c.status}** — {c.detail}"]
            if c.lines:
                out += ["", "```", *c.lines, "```"]
        return "\n".join(out) + "\n"

    def save(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / "report.md"
        path.write_text(self.markdown(), encoding="utf-8")
        return path


def _timed(fn, *a, **kw):
    started = time.perf_counter()
    value = fn(*a, **kw)
    return value, time.perf_counter() - started


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


class _FailAfter:
    """An embedder that stops answering after `calls` batches: a model that goes down in the middle of a run."""

    def __init__(self, inner: Any, calls: int, batch_size: int):
        self.inner, self.left, self.batch_size, self.model = inner, calls, batch_size, inner.model

    def embed_documents(self, texts):
        if self.left <= 0:
            raise EmbeddingError(f"'{self.model}' stopped answering (simulated by local_checks)")
        self.left -= 1
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)


# ---------------------------------------------------------------------------------------------------- 0: environment
def environment(report: Report) -> None:
    versions = []
    for pkg in ("duckdb", "litellm", "pyyaml", "pdfplumber", "fastapi"):
        try:
            versions.append(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{pkg} (not installed)")
    report.add(Check("environment", "INFO", f"python {sys.version.split()[0]}; " + ", ".join(versions), [
        f"text model       {s.role('copilot')['model']}", f"embedding model  {s.EMBEDDING_MODEL}",
        f"judge model      {s.JUDGE_MODEL}", f"semantic threshold {s.SEMANTIC_MIN_SIMILARITY}", f"ollama num_ctx   {s.OLLAMA_NUM_CTX}"]))


# ---------------------------------------------------------------------------------------------------- 1: models
def check_models(report: Report, embedder: Any, llm: Any, judge_llm: Any | None) -> None:
    try:
        vectors, secs = _timed(embedder.embed_documents, ["chest pain on exertion", "angina pectoris", "ankle sprain after a fall"])
        near, far = _cos(vectors[0], vectors[1]), _cos(vectors[0], vectors[2])
        ok = near > far
        report.add(Check("models: embeddings", "PASS" if ok else "WARN",
                         f"{len(vectors[0])} dimensions, {secs:.1f} s; related pair {near:.2f} vs unrelated pair {far:.2f}"
                         + ("" if ok else " (the related pair should score higher: is this the right model?)")))
    except Exception as exc:
        report.add(Check("models: embeddings", "FAIL", f"{type(exc).__name__}: {exc}"))
    try:
        reply, secs = _timed(llm.run, [{"role": "user", "content": 'Reply with only this JSON: {"ok": true}'}], temperature=0, max_tokens=TOKENS)
        found = re.search(r"\{.*\}", reply, re.S)
        report.add(Check("models: text model", "PASS" if found else "WARN", f"{secs:.1f} s; reply {reply.strip()[:60]!r}"
                         + ("" if found else " (an empty reply from a model that thinks may mean it ran out of tokens)")))
    except Exception as exc:
        report.add(Check("models: text model", "FAIL", f"{type(exc).__name__}: {exc}"))
        return
    try:  # does the model obey a JSON schema? (the planner relies on it)
        fmt = {"type": "json_schema", "json_schema": {"name": "plan", "schema": BAD_PLAN_SCHEMA, "strict": True}}
        replies, bad, total = [], 0, 3
        for q in ("How many records are there?", "Show patients older than 60", "Count records per department"):
            reply = llm.run([{"role": "user", "content": f"Question: {q}\nReply with a JSON plan."}], temperature=0, max_tokens=TOKENS, response_format=fmt)
            replies.append(reply.strip()[:70])
            try:
                bad += json.loads(reply).get("intent") not in ("search", "aggregate")
            except ValueError:
                bad += 1
        report.add(Check("models: JSON schema", "PASS" if not bad else "WARN",
                         f"{total - bad}/{total} replies followed the schema" + ("" if not bad else " (the server may ignore response_format)"),
                         [f"reply: {r}" for r in replies]))
    except Exception as exc:
        report.add(Check("models: JSON schema", "WARN", f"{type(exc).__name__}: {exc}"))
    if judge_llm is None:
        report.add(Check("models: judge", "SKIP", "no judge key configured (or --no-judge)"))
        return
    try:
        reply, secs = _timed(judge_llm.run, [{"role": "user", "content": "Reply with exactly: SCORE: 1"}], temperature=0, max_tokens=20)
        report.add(Check("models: judge", "PASS" if "SCORE" in reply else "WARN", f"{secs:.1f} s; reply {reply.strip()[:40]!r}"))
    except Exception as exc:
        report.add(Check("models: judge", "FAIL", f"{type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------------------------------- 2: embeddings
def _free_vectors(db_path: Path, keep_pending: int | None) -> None:
    """In the working copy: delete the vectors of `keep_pending` records (all of them if None), so the embedding step has work to do."""
    import duckdb

    from ..database.embeddings import TABLE as VECTORS

    con = duckdb.connect(str(db_path))
    try:
        if con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [VECTORS]).fetchone()[0]:
            if keep_pending is None:
                con.execute(f"DELETE FROM {VECTORS}")
            else:
                con.execute(f"DELETE FROM {VECTORS} WHERE record_id IN (SELECT record_id FROM {VECTORS} ORDER BY record_id LIMIT ?)", [keep_pending])
    finally:
        con.close()


def check_embeddings(report: Report, db_path: Path, embedder: Any) -> None:
    try:
        with RecordDB(db_path, embedder=embedder) as db:
            before = db.embedding_status()
            total = before["records"]
            report.add(Check("embeddings: records", "PASS" if total == 1000 else "WARN",
                             f"{total} records" + ("" if total == 1000 else " (the golden set was written for the 1,000-record database)")))
            # a model that goes down midway: what was done is kept, and a second call continues
            if total >= 4 and before["pending"] >= 4:
                size = max(1, min(embedder.batch_size, before["pending"] // 4))  # the failure comes after 2 batches: fewer than are pending
                real, db._embedder = db.embedder, _FailAfter(embedder, 2, size)
                try:
                    db.embed_pending()
                    report.add(Check("embeddings: interrupted run", "WARN", "the simulated failure did not happen"))
                except EmbeddingError as exc:
                    kept = db.embedding_status()["embedded"] - before["embedded"]
                    report.add(Check("embeddings: interrupted run", "PASS" if kept > 0 else "FAIL",
                                     f"{kept} vectors kept after the failure; message: {str(exc)[:90]}"))
                finally:
                    db._embedder = real
            else:
                report.add(Check("embeddings: interrupted run", "SKIP", "too few pending records"))
            shown = {"at": -1}

            def progress(n: int, of: int) -> None:  # about ten lines for the whole run
                if of and (n == of or n * 10 // of > shown["at"]):
                    shown["at"] = n * 10 // of
                    print(f"      embedded {n}/{of}", flush=True)

            started = time.perf_counter()
            result = db.embed_pending(progress=progress)
            secs = time.perf_counter() - started
            after = db.embedding_status()
            report.add(Check("embeddings: full run", "PASS" if after["pending"] == 0 else "FAIL",
                             f"{result.embedded} embedded, {result.unchanged} already up to date, {result.empty} without text; "
                             f"{secs:.0f} s ({secs / max(1, result.embedded):.2f} s per record); pending {after['pending']}"))
            again = db.embed_pending()
            report.add(Check("embeddings: second run", "PASS" if again.embedded == 0 else "FAIL", f"{again.embedded} re-embedded (must be 0)"))
            dims = db.con.execute("SELECT DISTINCT len(embedding) FROM record_embeddings").fetchall()
            report.add(Check("embeddings: dimensions", "PASS" if len(dims) == 1 else "FAIL", f"vector sizes in the table: {[d[0] for d in dims]}"))
            # the save and delete hooks of the web app, with the real model
            first = db.con.execute("SELECT record_id FROM medical_records LIMIT 1").fetchone()
            if first:
                record = dict(db.read_record(first[0]))
                record["record_id"] = "LOCAL-CHECK-0001"
                db.create_record(record)
                made = db.embed_record("LOCAL-CHECK-0001")
                has = db.con.execute("SELECT count(*) FROM record_embeddings WHERE record_id = 'LOCAL-CHECK-0001'").fetchone()[0]
                db.delete_record("LOCAL-CHECK-0001")
                gone = db.con.execute("SELECT count(*) FROM record_embeddings WHERE record_id = 'LOCAL-CHECK-0001'").fetchone()[0]
                ok = made and has == 1 and gone == 0
                report.add(Check("embeddings: save/delete hooks", "PASS" if ok else "FAIL",
                                 f"created {has} vector, after deleting the record {gone} left"))
    except Exception as exc:
        report.add(Check("embeddings: run", "FAIL", f"{type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------------------------------- 3: semantic
def check_semantic(report: Report, db_path: Path, embedder: Any, llm: Any) -> None:
    from ..copilot import Copilot

    saved, margin, cp = s.SEMANTIC_MIN_SIMILARITY, s.SEMANTIC_RELATIVE_MARGIN, None
    try:
        s.SEMANTIC_MIN_SIMILARITY = 0.0  # look at the whole range, to choose where to cut
        s.SEMANTIC_RELATIVE_MARGIN = None
        cp = Copilot.open(db_path, llm=llm, embedder=embedder)
        lines = [f"{'probe':<32} {'>=0.8':>6} {'>=0.7':>6} {'>=0.6':>6} {'>=0.5':>6} {'kept':>5}  top 3 (score, diagnosis)"]
        for q in PROBES:
            answer = cp.answer_plan({"intent": "search", "filters": [], "text_query": q})
            scores = sorted(answer.result.scores.values(), reverse=True) if answer.result is not None else []
            bands = [sum(x >= t for x in scores) for t in (0.8, 0.7, 0.6, 0.5)]
            kept = sum(x >= max(saved, scores[0] - (margin or 0)) for x in scores) if scores else 0  # what the current settings return
            top = "; ".join(f"{r['score']:.2f} {str(r.get('final_diagnosis'))[:30]}" for r in (answer.result.rows[:3] if answer.result else []))
            lines.append(f"{q:<32} {bands[0]:>6} {bands[1]:>6} {bands[2]:>6} {bands[3]:>6} {kept:>5}  {top}")
        lines += ["", "Counts are capped at SEMANTIC_TOP_K per probe. 'kept' is what the current settings return: scores >= the threshold",
                  f"and within {margin} of the best score. Choose the threshold where the top hits stop being relevant,",
                  "then set DOCINTEL_SEMANTIC_MIN_SIMILARITY (and DOCINTEL_SEMANTIC_RELATIVE_MARGIN). Hand-label M01-M03 of the golden set to measure it properly."]
        report.add(Check("semantic: scores", "INFO", f"current threshold {saved}, margin {margin}", lines))
    except Exception as exc:
        report.add(Check("semantic: scores", "FAIL", f"{type(exc).__name__}: {exc}"))
    finally:
        s.SEMANTIC_MIN_SIMILARITY, s.SEMANTIC_RELATIVE_MARGIN = saved, margin
        if cp is not None:
            cp.close()


# ---------------------------------------------------------------------------------------------------- 4 and 5: planner and judge
def check_planner(report: Report, db_path: Path, embedder: Any, llm: Any, judge: Judge | None, out: Path, min_accuracy: float) -> dict:
    from ..copilot import Copilot

    golden = load_golden()
    cp = Copilot.open(db_path, llm=llm, embedder=embedder, today=golden.as_of)
    seen: dict[str, dict] = {}
    original = cp.ask

    def ask(question, previous=None, *a, **kw):  # the same ask, remembering time, trace and what the judge will read
        started = time.perf_counter()
        answer = original(question, previous, *a, **kw)
        seen[question] = {"seconds": time.perf_counter() - started, "trace": answer.trace, "kind": answer.kind,
                          "view": judge_view(answer.result.to_dict(), answer.notes) if answer.result is not None else None, "text": answer.text}
        return answer

    cp.ask = ask
    try:
        has_icd = "icd10_code" in cp.executor.columns
        only = [c.id for c in golden.cases if has_icd or not c.id.startswith("I")]
        skipped = [c.id for c in golden.cases if c.id not in only]
        eval_report = evaluate(cp, golden, judge, only=only,
                               progress=lambda i, n: print(f"      case {i}/{n}", flush=True) if i % 5 == 0 or i == n else None)
    except Exception as exc:
        report.add(Check("planner: golden set", "FAIL", f"{type(exc).__name__}: {exc}"))
        return {}
    finally:
        cp.close()
    out.mkdir(parents=True, exist_ok=True)
    eval_report.save(out / "eval_report.json")
    summary = eval_report.summary()
    acc = summary["execution_accuracy"]
    times = [v["seconds"] for v in seen.values()]
    checks = sum(any(step["step"] == "check" for step in v["trace"]) for v in seen.values())
    retried = sum(sum(step["step"] == "llm" for step in v["trace"]) > 1 for v in seen.values())
    kinds: dict[str, int] = {}
    for v in seen.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    lines = eval_report.format().splitlines() + ["", f"seconds per question: median {statistics.median(times):.1f}, "
                                                  f"max {max(times):.1f}, total {sum(times) / 60:.1f} min" if times else "",
                                                  f"questions where the plan was sent back (a named value was dropped): {checks}",
                                                  f"questions that needed more than one model call: {retried}",
                                                  f"answer kinds: {kinds}"]
    if skipped:
        lines.append(f"not run (the database has no icd10 columns): {', '.join(skipped)}")
    rate = acc["rate"] or 0
    safety = summary["safety_checks"]
    unsafe = [x["id"] for x in summary["failed"] if x.get("checks_failed")]
    lines.append(f"safety checks (injection, writes, leaks, whole-table dumps): {safety['correct']}/{safety['of']} cases with every check held"
                 + (f"; FAILED: {', '.join(unsafe)}" if unsafe else ""))
    report.add(Check("planner: golden set", "PASS" if rate >= min_accuracy and not unsafe else "WARN",
                     f"execution accuracy {acc['correct']}/{acc['of']} = {rate:.0%} (95% CI {acc['ci95'][0]:.0%}-{acc['ci95'][1]:.0%}); "
                     f"end to end {summary['end_to_end']['correct']}/{summary['end_to_end']['of']}; threshold {min_accuracy:.0%}"
                     + (f"; safety checks failed in {', '.join(unsafe)}" if unsafe else ""), lines))
    passed = {c.question for c in eval_report.cases if c.level1_score == 1}
    return {"samples": [(v["view"], v["text"]) for q, v in seen.items() if q in passed and v["view"] and v["kind"] in ("aggregate", "results")]}


def check_judge(report: Report, judge: Judge | None, samples: list[tuple[str, str]]) -> None:
    if judge is None:
        report.add(Check("judge: calibration", "SKIP", "no judge configured"))
        return
    if not samples:
        report.add(Check("judge: calibration", "SKIP", "no correct answers to corrupt (run the planner step first)"))
        return
    try:
        result = calibrate(judge, samples[:12])
    except Exception as exc:
        report.add(Check("judge: calibration", "FAIL", f"{type(exc).__name__}: {exc}"))
        return

    def frac(text: str) -> float:
        k, n = map(int, text.split("/"))
        return k / n if n else 0.0

    good = frac(result["detected"]["wrong_number"]) >= 0.8 and frac(result["detected"]["unsupported_claim"]) >= 0.8 \
        and frac(result["false_alarms"]) <= 0.1 and result["unparsed"] == 0
    report.add(Check("judge: calibration", "PASS" if good else "WARN",
                     f"wrong numbers caught {result['detected']['wrong_number']}, invented claims caught {result['detected']['unsupported_claim']}, "
                     f"false alarms {result['false_alarms']}, unreadable replies {result['unparsed']}"
                     + ("" if good else " -> do not trust the judge's numbers until this passes (need >=80% caught, <=10% false alarms)")))


# ---------------------------------------------------------------------------------------------------- main
def judge_configured() -> bool:
    return bool(s.JUDGE_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def run(argv: list[str] | None = None, *, llm: Any = None, embedder: Any = None, judge_llm: Any = None) -> Report:
    """Run the checks. `llm`, `embedder` and `judge_llm` replace the real models (used by the tests of this module)."""
    ap = argparse.ArgumentParser(description="Checks with the real models; see the module docstring.")
    ap.add_argument("--db", default=s.DB_PATH, help="the DuckDB file (default: DOCINTEL_DB_PATH)")
    ap.add_argument("--out", default="local_checks", help="folder for the report and the working copy")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--in-place", action="store_true", help="create the vectors in --db itself instead of in a copy")
    ap.add_argument("--rebuild-embeddings", action="store_true", help="delete every vector in the working copy first, so the embedding step builds them all "
                    "(by default a sample is deleted so the interrupted-run check has work to interrupt); not with --in-place")
    ap.add_argument("--no-judge", action="store_true", help="never call the commercial judge")
    ap.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="colours in the console (default: only on a terminal)")
    ap.add_argument("--min-accuracy", type=float, default=0.8, help="execution accuracy that counts as PASS (default 0.8)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = Report(out, use_color(args.color))
    source = Path(args.db)
    if not source.is_file():
        report.add(Check("database", "FAIL", f"{source} does not exist (use --db)"))
        report.save()
        return report
    if args.in_place and args.rebuild_embeddings:
        report.add(Check("database", "FAIL", "--rebuild-embeddings deletes vectors; it only works on the copy (drop --in-place)"))
        report.save()
        return report
    if args.in_place:
        work = source
    else:
        work = out / "work.duckdb"
        shutil.copy2(source, work)
        wal = Path(str(source) + ".wal")
        if wal.exists():
            shutil.copy2(wal, Path(str(work) + ".wal"))
        if "embeddings" in set(args.steps):
            _free_vectors(work, None if args.rebuild_embeddings else SAMPLE_PENDING)
    embedder = embedder or Embedder()
    llm = llm or LLM(role="copilot")
    use_judge = not args.no_judge and (judge_llm is not None or judge_configured())
    judge = Judge(llm=judge_llm) if use_judge else None
    judge_llm = judge.llm if judge else None
    steps = set(args.steps)

    environment(report)
    if "models" in steps:
        check_models(report, embedder, llm, judge_llm)
    if "embeddings" in steps:
        if report.failed("models: embeddings"):
            report.add(Check("embeddings: run", "SKIP", "the embedding model did not answer"))
        else:
            check_embeddings(report, work, embedder)
    if "semantic" in steps:
        check_semantic(report, work, embedder, llm)
    samples: list = []
    if "planner" in steps or "judge" in steps:
        if report.failed("models: text model"):
            report.add(Check("planner: golden set", "SKIP", "the text model did not answer"))
        else:
            run_judge_inline = judge if "judge" in steps else None
            samples = check_planner(report, work, embedder, llm, run_judge_inline, out, args.min_accuracy).get("samples", [])
    if "judge" in steps:
        check_judge(report, judge, samples)
    path = report.save()
    print("\n" + paint("━━ Summary ".ljust(60, "━"), "34", report.color, bold=True) + "\n   " + report.summary_line(), flush=True)
    print(f"\nReport: {path}   Cases: {out / 'eval_report.json'}", flush=True)
    return report


def main() -> None:
    report = run()
    sys.exit(1 if any(c.status == "FAIL" for c in report.checks) else 0)


if __name__ == "__main__":
    main()
