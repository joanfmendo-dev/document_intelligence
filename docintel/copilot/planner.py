"""The planner: the one place the copilot asks a language model something.

It gets the question, the previous plan (for follow-ups) and the catalog, and returns a JSON plan (see plan.py). It is not trusted:
whatever it returns goes through validator.py before it can touch the database.
"""
from __future__ import annotations

import json
import time
from datetime import date
from typing import Any

from ..shared.llm import LLM, LLMError
from .catalog import Catalog
from .plan import PlanFormatError, parse_plan, plan_schema

INSTRUCTIONS = """You turn a question about a medical-records database into a JSON plan. You never write SQL and you never answer the question yourself: another program runs the plan.
Reply with ONE JSON object and nothing else.

INTENTS (field "intent")
- search: find records. Uses "filters" and, only for a description by meaning, "text_query".
- aggregate: count / avg / min / max. Uses "filters" and "aggregate": {"metric": "count|avg|min|max", "field": <numeric field or null>, "group_by": <one field or null>, "top_n": <integer or null>}.
- get_record: open one record. Uses "record_id".
- navigate: take the user to a page. Uses "destination" ("records", "record" or "upload"), and "record_id" or "filters" when relevant.
- help: "topic" is one of capabilities, upload_formats, fields, search_examples.
- clarify: the question is ambiguous, or names something that does not exist. Uses "question" (what to ask the user), optionally "suggestions".
- out_of_scope: "reason" is write (anything that adds, changes or deletes data), export, clinical_advice, or unrelated.

FILTERS: a list of {"field", "op", "value"}; ALL of them must hold. Use only the fields and operators listed in the catalog. For "A or B" inside one field use op "in" with a list.
- Categorical values: copy the spelling of the catalog values exactly. If the user names a value that is not in the catalog, use clarify.
- Dates: never calculate dates. Today is {TODAY}. Use op "between" with one of: {"token": "this_year"} (also last_year, this_month, last_month, today, yesterday), {"token": "last_n_days", "n": 30}, or, only when the user names the year, month or day, {"year": <year>}, {"month": "YYYY-MM"}, {"day": "YYYY-MM-DD"}, {"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}. "This year", "last year", "this month", "last month", "today", "yesterday" ALWAYS use the token, never a literal year or date.
- Values: copy only the value, never the number in parentheses that follows it in the catalog (that is a record count).
- Numbers: write plain numbers; op "between" takes a list of two numbers, [0, 35] ("between 0 and 35", "up to 35" with a lower limit). Never use {"from", "to"} for numbers: that is only for dates. Temperature is stored in Fahrenheit; if the user gives Celsius write {"value": 39, "unit": "C"}.
- Lab results: field "lab_results", op "has", value {"test_name": [...], "flag": [...]}. Include EVERY catalog test name that matches what the user said (some tests have variants).
- Text fields (diagnosis, complaint, patient name): op "contains" with a short word or phrase. Prefer this to text_query when you know the word.
- text_query is only for describing a case by meaning ("dizzy when standing up"). Combine it with filters when the question also names a department, a date, etc.

AGGREGATES: "how many ..." is a count with "group_by": null. Set group_by ONLY when the user says "by", "per" or "for each" a field; never to a lab test, a patient or a record ID. A question about tests or results ("how many patients were tested for X") is a count with a lab_results filter, not a grouping.

FOLLOW-UPS: when a previous plan is given, return the COMPLETE new plan: keep every condition that still applies, and change, add or remove only what the new question says ("only women" adds a filter; "make it over 70" replaces the age filter; "remove the age filter" removes it; "now by insurance" changes group_by).

NEVER invent a field, a value or a record ID. Requests to add, edit, delete or run commands are out_of_scope with reason "write".

EXAMPLES
Question: Gastroenterology patients under 40 in January 2026
{"intent": "search", "filters": [{"field": "department", "op": "in", "value": ["Gastroenterology"]}, {"field": "age", "op": "lt", "value": 40}, {"field": "encounter_date", "op": "between", "value": {"month": "2026-01"}}]}
Question: average heart rate by sex
{"intent": "aggregate", "filters": [], "aggregate": {"metric": "avg", "field": "vital_signs.heart_rate", "group_by": "sex", "top_n": null}}
Question: How many patients were tested for glucose?
{"intent": "aggregate", "filters": [{"field": "lab_results", "op": "has", "value": {"test_name": ["Glucose"]}}], "aggregate": {"metric": "count", "field": null, "group_by": null, "top_n": null}}
Question: patients who feel dizzy when they stand up
{"intent": "search", "filters": [], "text_query": "dizziness on standing up"}
Question: please rename patient 12 to Ana
{"intent": "out_of_scope", "reason": "write"}
"""


MAX_TURNS = 4  # earlier turns of the chat the model sees


class PlannerError(LLMError):
    """The model did not return a usable plan, even after one retry."""


class Planner:
    today: date | None = None  # the date given to the model; None = the real date

    def __init__(self, catalog: Catalog, llm: Any = None, *, max_attempts: int = 2, today: date | None = None):
        """`llm` is anything with .run(messages, **params) -> str (default: the model in settings: COPILOT_LLM_MODEL or LLM_MODEL)."""
        self.catalog = catalog
        self.today = today
        self.llm = llm if llm is not None else LLM(role="copilot")
        self.max_attempts = max_attempts
        self.last_attempts: list[dict] = []  # the model's raw replies of the last plan() call, for the answer's trace

    def messages(self, question: str, previous: dict | None = None, history: list[dict] | None = None) -> list[dict]:
        """`history`: the earlier turns of the chat, oldest first, each {"question", "plan", "reply"} (the reply is what the user was shown)."""
        system = INSTRUCTIONS.replace("{TODAY}", (self.today or date.today()).isoformat()) + "\nCATALOG (fields, [role; operators], and for categorical fields the values with their record counts)\n" \
            + self.catalog.prompt_text(question)
        user = f"Question: {question}"
        turns = [h for h in (history or []) if isinstance(h, dict) and isinstance(h.get("plan"), dict)][-MAX_TURNS:]
        chat = []
        for h in turns:
            chat += [{"role": "user", "content": f"Question: {str(h.get('question') or '')[:300]}"},
                     {"role": "assistant", "content": json.dumps(h["plan"], ensure_ascii=False)}]
        if turns and turns[-1].get("reply"):
            user = (f"The user was shown: {str(turns[-1]['reply'])[:300]}\nThis is their next message; use the chat so far and return the COMPLETE plan.\n"
                    f"Next message: {question}")
        if previous and not turns:
            user = f"Previous plan: {json.dumps(previous, ensure_ascii=False)}\nFollow-up question: {question}"
        return [{"role": "system", "content": system}, *chat, {"role": "user", "content": user}]

    def plan(self, question: str, previous: dict | None = None, history: list[dict] | None = None,
             correction: tuple[dict, str] | None = None) -> dict:
        """The model's plan for `question`. One retry, telling the model what was wrong with an unusable answer.
        Raises PlannerError if it still fails, LLMError if the model cannot be reached."""
        messages = self.messages(question, previous, history)
        if correction:  # (the plan the model gave, what is wrong with it): ask again with its own answer and the remark in view
            messages = messages + [{"role": "assistant", "content": json.dumps(correction[0], ensure_ascii=False)},
                                   {"role": "user", "content": correction[1]}]
        response_format = {"type": "json_schema", "json_schema": {"name": "plan", "schema": plan_schema(self.catalog), "strict": True}}
        last = ""
        if not correction:
            self.last_attempts = []  # a correction adds to the attempts of the same question
        for _ in range(self.max_attempts):
            attempt = {"messages": messages}
            self.last_attempts.append(attempt)
            started = time.perf_counter()
            try:
                reply = self.llm.run(messages, temperature=0, max_tokens=700, response_format=response_format)
            except Exception as exc:
                attempt.update(error=f"{type(exc).__name__}: {exc}", seconds=round(time.perf_counter() - started, 2))
                raise
            attempt.update(reply=reply, seconds=round(time.perf_counter() - started, 2))
            try:
                return parse_plan(reply)
            except PlanFormatError as exc:
                last = attempt["error"] = str(exc)
                messages = messages + [{"role": "assistant", "content": reply},
                                       {"role": "user", "content": f"That was not usable: {exc}. Reply with only the JSON plan."}]
        raise PlannerError(f"the model did not return a usable plan ({last})")
