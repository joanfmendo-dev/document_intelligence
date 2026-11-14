"""Run the golden set through the copilot and score it on two levels.

    golden = load_golden()
    copilot = Copilot.open("medical_records.duckdb", today=golden.as_of)
    report = evaluate(copilot, golden, Judge())          # Judge() = the commercial model in settings; omit it to score level 1 only
    print(report.format())
    report.save("evaluation_report.json")

Level 1 (execution.py) compares what the copilot got from the database with the golden result; for questions that run no query
(clarify, refusals, navigation) it checks the intent and its reason. Level 2 (judge.py) is the LLM judge on the text shown:
fidelity to the executed result, and whether the interpretation line reflects the question. The plan is also compared with the
expected plan, but only as a diagnostic: a different plan that returns the right data is not a failure.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..copilot.plan import plans_equivalent
from .execution import execution_verdict, wilson
from .golden import Golden, GoldenCase
from .judge import Judge, judge_view

JUDGED_KINDS = ("results", "aggregate", "record")


@dataclass
class CaseResult:
    id: str
    question: str
    agent_kind: str = ""
    agent_plan: dict | None = None
    plan_match: bool | None = None          # diagnostic: same plan as expected, after normalization
    intent_ok: bool | None = None           # cases that run no query: intent (and reason / destination / topic) as expected
    level1: str = "n/a"
    level1_detail: str = ""
    level1_score: int | None = None
    text: str = ""
    interpretation: str = ""
    judge_ran: bool = False
    judge_fidelity: int | None = None
    judge_interpretation: int | None = None
    error: str = ""

    @property
    def base(self) -> bool | None:
        """Did the copilot get the data (or the intent) right? None when this case cannot be scored."""
        if self.level1_score is not None:
            return self.level1_score == 1
        return self.intent_ok

    @property
    def passed(self) -> bool | None:
        """Correct end to end: level 1 and, if a judge ran, both judge criteria."""
        if self.base is None:
            return None
        if not self.judge_ran:
            return self.base
        return self.base and self.judge_fidelity == 1 and self.judge_interpretation == 1


def _intent_ok(case: GoldenCase, answer) -> bool:
    q, plan = answer.query, case.expected_plan
    if q is None or q.intent != plan.get("intent"):
        return False
    if plan["intent"] == "out_of_scope":
        return q.reason == plan.get("reason")
    if plan["intent"] == "navigate":
        return q.destination == plan.get("destination")
    if plan["intent"] == "help":
        return q.topic == plan.get("topic")
    return True


def run_case(copilot: Any, case: GoldenCase, judge: Judge | None, today: str) -> CaseResult:
    out = CaseResult(case.id, case.question)
    try:
        answer = copilot.ask(case.question, previous=case.previous_plan)
    except Exception as exc:  # a crash is a failed case, not a failed evaluation
        out.error, out.level1, out.level1_detail, out.level1_score = f"{type(exc).__name__}: {exc}", "mismatch", "the copilot raised an error", 0
        return out
    out.agent_kind, out.agent_plan, out.text, out.interpretation = answer.kind, answer.plan, answer.text, answer.interpretation
    if answer.plan is not None:
        out.plan_match = plans_equivalent(case.expected_plan, answer.plan, copilot.catalog)
    result = answer.result.to_dict() if answer.result is not None else None
    verdict = execution_verdict(case.expected_result, case.intent, result)
    out.level1, out.level1_detail, out.level1_score = verdict.level, verdict.detail, verdict.score
    if verdict.level == "n/a" and case.intent not in ("search", "get_record", "aggregate"):
        out.intent_ok = _intent_ok(case, answer)
    if judge is not None and answer.kind in JUDGED_KINDS and answer.result is not None:
        out.judge_ran = True
        out.judge_fidelity = judge.fidelity(judge_view(result, answer.notes), answer.text)
        out.judge_interpretation = judge.interpretation(case.question, answer.interpretation,
                                                        json.dumps(case.previous_plan, ensure_ascii=False) if case.previous_plan else "", today)
    return out


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)

    def summary(self) -> dict:
        def rate(k: int, n: int) -> dict:
            lo, hi = wilson(k, n)
            return {"correct": k, "of": n, "rate": round(k / n, 3) if n else None, "ci95": [round(lo, 3), round(hi, 3)]}

        c = self.cases
        l1 = [x for x in c if x.level1_score is not None]
        intents = [x for x in c if x.intent_ok is not None]
        plans = [x for x in c if x.plan_match is not None and x.base is not None]
        judged = [x for x in c if x.judge_ran]
        fid = [x for x in judged if x.judge_fidelity is not None]
        itp = [x for x in judged if x.judge_interpretation is not None]
        both = [x for x in l1 if x.judge_ran and x.judge_fidelity is not None and x.judge_interpretation is not None]
        combo = {"level1 pass + judge pass": 0, "level1 pass + judge fail": 0, "level1 fail + judge pass": 0, "level1 fail + judge fail": 0}
        for x in both:
            ok2 = x.judge_fidelity == 1 and x.judge_interpretation == 1
            combo[f"level1 {'pass' if x.level1_score == 1 else 'fail'} + judge {'pass' if ok2 else 'fail'}"] += 1
        scored = [x for x in c if x.passed is not None]
        return {
            "cases": len(c),
            "execution_accuracy": rate(sum(x.level1_score for x in l1), len(l1)),
            "intent_accuracy_no_query_cases": rate(sum(bool(x.intent_ok) for x in intents), len(intents)),
            "plan_match_diagnostic": rate(sum(bool(x.plan_match) for x in plans), len(plans)),
            "judge_fidelity": rate(sum(x.judge_fidelity for x in fid), len(fid)),
            "judge_interpretation": rate(sum(x.judge_interpretation for x in itp), len(itp)),
            "judge_unparsed": len(judged) - len(fid) + len(judged) - len(itp),
            "level1_vs_judge": combo,
            "end_to_end": rate(sum(bool(x.passed) for x in scored), len(scored)),
            "not_scored": [x.id for x in c if x.passed is None],
            "failed": [{"id": x.id, "question": x.question, "level1": x.level1, "detail": x.level1_detail or x.error,
                        "intent_ok": x.intent_ok, "judge": [x.judge_fidelity, x.judge_interpretation] if x.judge_ran else None}
                       for x in c if x.passed is False],
        }

    def format(self) -> str:
        lines = [f"{'id':<5} {'level 1':<15} {'plan':<5} {'judge F/I':<10} detail"]
        for x in self.cases:
            plan = "" if x.plan_match is None else "same" if x.plan_match else "diff"
            judge = f"{x.judge_fidelity}/{x.judge_interpretation}" if x.judge_ran else ""
            lines.append(f"{x.id:<5} {x.level1 if x.level1 != 'n/a' else ('intent ok' if x.intent_ok else 'intent WRONG' if x.intent_ok is False else 'not scored'):<15} "
                         f"{plan:<5} {judge:<10} {x.level1_detail or x.error}")
        s = self.summary()
        lines += ["", f"Execution accuracy: {_fmt(s['execution_accuracy'])}",
                  f"Intent accuracy (cases that run no query): {_fmt(s['intent_accuracy_no_query_cases'])}",
                  f"Plan identical to expected (diagnostic only): {_fmt(s['plan_match_diagnostic'])}"]
        if any(x.judge_ran for x in self.cases):
            lines += [f"Judge, fidelity: {_fmt(s['judge_fidelity'])}", f"Judge, interpretation: {_fmt(s['judge_interpretation'])}",
                      f"Judge replies that could not be read: {s['judge_unparsed']}",
                      "Level 1 against the judge: " + "; ".join(f"{k} = {v}" for k, v in s["level1_vs_judge"].items())]
        lines += [f"Correct end to end: {_fmt(s['end_to_end'])}", f"Not scored (a person must label them): {', '.join(s['not_scored']) or 'none'}"]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "cases": [asdict(x) for x in self.cases]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def _fmt(r: dict) -> str:
    if not r["of"]:
        return "n/a (no cases)"
    return f"{r['correct']}/{r['of']} = {r['rate']:.1%} (95% Wilson CI {r['ci95'][0]:.0%}-{r['ci95'][1]:.0%})"


def evaluate(copilot: Any, golden: Golden, judge: Judge | None = None, *, only: list[str] | None = None,
             progress: Callable[[int, int], None] | None = None) -> EvalReport:
    """Score `copilot` on the golden set (all cases, or only the ids in `only`). The copilot's date is set to golden.as_of for the run."""
    cases = [c for c in golden.cases if not only or c.id in only]
    previous_today, copilot.today = copilot.today, golden.as_of
    try:
        report = EvalReport()
        for i, case in enumerate(cases, 1):
            report.cases.append(run_case(copilot, case, judge, str(golden.as_of)))
            if progress:
                progress(i, len(cases))
        return report
    finally:
        copilot.today = previous_today
