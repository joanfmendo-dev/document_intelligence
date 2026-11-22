"""Level 2: an LLM judge reads what the copilot SHOWED and scores it 1 or 0 on two separate criteria.

  fidelity        the judge sees the result the copilot executed (the only source of truth) and the text shown; 0 if the text
                  reports a figure, entity, unit or period the result does not support, adds trends or claims that cannot be derived,
                  omits something essential, or says records exist when the result is empty.
  interpretation  the judge sees the user's question and the line saying how it was understood; 0 if that line does not reflect the question.

Two prompts, because a small model judges one thing at a time better. The judge is a commercial model (settings JUDGE_MODEL), used only on
synthetic data in evaluation, never in production. A reply without "SCORE: 0|1" gives None, counted apart from failures.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..shared.llm import LLM

FIDELITY_SYSTEM = """You check whether a text shown to a user faithfully represents the result of a database query.

Score 1 if the text conveys the information in the EXECUTION RESULT, keeping the figures, entities, units, periods, comparisons and conclusions.
Score 0 if the text:
- reports any figure, entity, unit or period that the result does not support;
- adds claims, trends, percentages, comparisons or conclusions that cannot be derived from the result;
- contradicts or misrepresents the result;
- omits information essential to its meaning (for example the total, when the result has one);
- says or implies that records exist when the result is empty or has a total of 0.

Rules: the execution result is the sole source of truth. "total" is the number of matching records; "showing" lists only the rows on the page, so a text that shows fewer rows than the total is fine when it says so. Accept paraphrases, formatting and rounding within the precision of the less precise number. A line that only describes how the question was understood is not a claim about the data.

Reply with one short sentence of reasoning, then a final line exactly like:
SCORE: 1
or
SCORE: 0"""

INTERPRETATION_SYSTEM = """You check whether a system understood a user's question about a medical-records database.

You get the USER QUESTION and the INTERPRETATION the system wrote of it (the conditions it searched for or the calculation it ran).
Score 1 if the interpretation captures what the question asked: every condition of the question is there, with the right values, ranges and periods, and it adds no condition the question did not ask for.
Score 0 if a condition is missing, wrong or invented, a period or number differs, or it calculates something other than what was asked.

If a PREVIOUS SEARCH is given, the question is a follow-up: the interpretation should be the previous search changed as the question says.
The records cover January 2025 to November 2026; the current date is given as TODAY, so "this year" or "last 30 days" are relative to it. Ignore wording and the order of conditions. Only judge the interpretation, not whether the data exists.

Reply with one short sentence of reasoning, then a final line exactly like:
SCORE: 1
or
SCORE: 0"""


def judge_view(result_dict: dict | None, warnings: list[str] | None = None) -> str:
    """The executed result as the judge reads it: the total and the rows on the page, no long id lists."""
    if result_dict is None:
        return "{}"
    view = {"total": result_dict.get("total"), "showing": result_dict.get("rows", [])}
    if result_dict.get("kind") == "aggregate":
        view = {"groups_or_value": result_dict.get("rows", [])}
    if result_dict.get("hints"):
        view["if_a_condition_is_dropped_the_total_would_be"] = result_dict["hints"]
    if warnings:
        view["notes_shown_to_the_user"] = warnings
    return json.dumps(view, ensure_ascii=False, default=str)


class Judge:
    def __init__(self, llm: Any = None, *, retries: int = 1):
        """`llm` is anything with .run(messages, **params) -> str; default: the commercial model in settings (JUDGE_MODEL)."""
        self.llm = llm if llm is not None else LLM(role="judge")
        self.retries = retries

    def _score(self, system: str, user: str) -> int | None:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(self.retries + 1):
            found = re.findall(r"SCORE:\s*([01])\b", self.llm.run(messages, temperature=0, max_tokens=300))
            if found:
                return int(found[-1])
        return None

    def fidelity(self, result_view: str, text: str) -> int | None:
        return self._score(FIDELITY_SYSTEM, f"EXECUTION RESULT:\n{result_view}\n\nTEXT SHOWN TO THE USER:\n{text}")

    def interpretation(self, question: str, interpretation: str, previous: str = "", today: str = "") -> int | None:
        user = (f"TODAY: {today}\n" if today else "") + (f"PREVIOUS SEARCH (JSON):\n{previous}\n" if previous else "") \
            + f"USER QUESTION:\n{question}\n\nINTERPRETATION:\n{interpretation}"
        return self._score(INTERPRETATION_SYSTEM, user)


# ---------------------------------------------------------------- is the judge itself reliable?
def corrupt_number(text: str) -> str | None:
    """The text with its first number changed (a wrong figure the judge must catch); None if it has no number."""
    match = re.search(r"\b\d+\b", text)
    return None if not match else text[:match.start()] + str(int(match.group()) + 7) + text[match.end():]


def add_claim(text: str) -> str:
    """The text with a claim no result can support (the 'up about 12% from last week' of the reference project)."""
    return text + "\nThis is up about 12% from the same period last week."


def calibrate(judge: Judge, samples: list[tuple[str, str]]) -> dict:
    """Controls for the fidelity judge. `samples` = [(result_view, text)] of answers believed correct. For each the judge must say 1 on the
    original and 0 on a corrupted copy. Reports the detection rate (errors caught) and the false-alarm rate (correct answers failed)."""
    caught = {"wrong_number": [0, 0], "unsupported_claim": [0, 0]}
    original = [0, 0]
    unparsed = 0
    for view, text in samples:
        score = judge.fidelity(view, text)
        if score is None:
            unparsed += 1
        else:
            original[1] += 1
            original[0] += 1 if score == 0 else 0  # a false alarm
        for kind, bad in (("wrong_number", corrupt_number(text)), ("unsupported_claim", add_claim(text))):
            if bad is None:
                continue
            score = judge.fidelity(view, bad)
            if score is None:
                unparsed += 1
                continue
            caught[kind][1] += 1
            caught[kind][0] += 1 if score == 0 else 0
    return {"false_alarms": f"{original[0]}/{original[1]}",
            "detected": {k: f"{c}/{n}" for k, (c, n) in caught.items()}, "unparsed": unparsed}
