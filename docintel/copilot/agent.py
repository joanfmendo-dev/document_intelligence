"""Copilot: question -> plan (model) -> validation (code) -> read-only query (code) -> answer (template).

    copilot = Copilot.open("medical_records.duckdb")            # its own read-only connection
    answer = copilot.ask("Cardiology patients over 60 with high troponin this year")
    print(answer.text)
    answer = copilot.ask("only women", previous=answer.plan)    # a follow-up: pass the plan of the previous answer

Inside the web app, which already holds the database open for writing, use Copilot.from_db(db): it reads through a cursor of that
connection (DuckDB cannot open the same file read-only and read-write in one process). Either way the copilot only ever runs SELECTs
that it builds itself from a validated plan.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from ..shared.embeddings import EmbeddingError
from ..shared.llm import LLMError
from .catalog import LAB, Catalog
from .executor import Executor
from .plan import Query
from .planner import Planner, PlannerError
from .render import Answer, render

log = logging.getLogger("docintel")
from .validator import validate

MODEL_DOWN = "The assistant's language model is not available right now, so I could not read your question. Please try again in a moment."
EMBEDDING_DOWN = "Searching by meaning is not available right now (the embedding model could not be reached). Searches by field still work."
NOT_UNDERSTOOD = "I could not turn that into a search. Could you rephrase it, for example naming a department, a date range or a diagnosis?"


def load_catalog(con: Any, path: str | Path | None = None) -> Catalog:
    """The catalog file if it exists, else one profiled from the database now (in memory: only the web app writes the file)."""
    if path and Path(path).is_file():
        return Catalog.from_yaml(path)
    return Catalog.build(con)


class Copilot:
    def __init__(self, executor: Executor, catalog: Catalog, planner: Planner | None = None, *, llm: Any = None, today: date | None = None):
        self.executor, self.catalog = executor, catalog
        self.planner = planner or Planner(catalog, llm, today=today)
        self.today = today

    @classmethod
    def open(cls, db_path: str | Path, *, catalog_path: str | Path | None = None, llm: Any = None, embedder: Any = None,
             today: date | None = None) -> "Copilot":
        """A copilot on its own read-only connection to the database file."""
        executor = Executor.open(db_path, embedder=embedder)
        return cls(executor, load_catalog(executor.con, catalog_path), llm=llm, today=today)

    @classmethod
    def from_db(cls, db: Any, *, catalog_path: str | Path | None = None, llm: Any = None, embedder: Any = None,
                today: date | None = None, catalog: Catalog | None = None) -> "Copilot":
        """A copilot that reads through a cursor of an open RecordDB (use this inside the web app)."""
        executor = Executor(db.con.cursor(), embedder=embedder or db.embedder)
        return cls(executor, catalog or load_catalog(executor.con, catalog_path), llm=llm, today=today)

    def close(self) -> None:
        self.executor.close()

    def ask(self, question: str, previous: dict | None = None, page: int = 1, history: list[dict] | None = None) -> Answer:
        """Answer a question. `previous` is the `plan` of the previous answer when this is a follow-up; `history` is a chat
        ([{"question", "plan", "reply"}, ...], oldest first) and, when given, takes its place: the model then sees the last turns."""
        question = (question or "").strip()
        if not question:
            return render(Query("clarify", message="What would you like to know about the records?"))
        try:
            plan = self.planner.plan(question, previous, history)
        except PlannerError:
            answer = render(Query("clarify", message=NOT_UNDERSTOOD))
            answer.trace = self._llm_trace()
            return answer
        except LLMError as exc:
            log.warning("The copilot's model could not be called: %s", exc)  # the person is told only that it is down; the log says why
            return Answer("error", text=MODEL_DOWN, summary=MODEL_DOWN, trace=self._llm_trace())
        answer = self.answer_plan(plan, page)
        missing = self._missing(question, plan) if answer.query is not None and answer.query.intent in ("search", "aggregate") else []
        extra: list[dict] = []
        if missing:
            words = ", ".join(f"“{w}”" for w in missing)
            hint = (f"Your plan does not use {words} from the question. If the question refers to it, include it as a filter "
                    f"(a lab test goes in field \"lab_results\", op \"has\", value {{\"test_name\": [...]}}). Return the COMPLETE corrected plan.")
            try:
                second = self.planner.plan(question, previous, history, correction=(plan, hint))
            except (PlannerError, LLMError):
                second = None
            extra.append({"step": "check", "missing": missing, "action": "asked the model again" if second else "kept the plan"})
            if second is not None and second != plan:
                plan, answer = second, self.answer_plan(second, page)
                missing = self._missing(question, plan) if answer.query is not None and answer.query.intent in ("search", "aggregate") else []
            if missing and answer.query is not None and answer.query.intent != "clarify":
                answer = self.answer_plan(plan, page, notes=[f"The question mentions {', '.join(f'“{w}”' for w in missing)}, but no condition uses "
                                                             f"it, so the result may be broader than you asked."])
        answer.trace = self._llm_trace() + extra + answer.trace
        return answer

    def _missing(self, question: str, plan: dict) -> list[str]:
        """Words of the question that name a catalog value (a lab test, a department...) which the plan does not use anywhere."""
        # what the plan uses: filter values and field names ("oxygen_saturation" covers "oxygen saturation"), the text and the aggregate
        parts = _strings([plan.get("filters"), plan.get("text_query"), plan.get("aggregate")]) + [f.get("field", "") for f in plan.get("filters") or []
                                                                                                    if isinstance(f, dict)]
        fields = [f.get("field") for f in plan.get("filters") or [] if isinstance(f, dict)]
        parts += [self.catalog.columns[n].description for n in fields  # "SpO2" = oxygen saturation (not for lab fields: their text lists test names)
                  if isinstance(n, str) and n in self.catalog.columns and not n.startswith(LAB)]
        used = " | ".join(parts).casefold().replace("_", " ").replace(".", " ")
        return sorted(w for w, matches in self.catalog.mentions(question).items()
                      if w not in used and not any(v.casefold() in used for _, v in matches))

    def _llm_trace(self) -> list[dict]:
        """The model's calls for the last question: what it was sent and what it replied (also when the reply was unusable)."""
        return [{"step": "llm", "attempt": i, **a} for i, a in enumerate(self.planner.last_attempts, 1)]

    def answer_plan(self, plan: dict, page: int = 1, notes: list[str] | None = None) -> Answer:
        """Validate, run and render a plan without asking the model (used for 'show more' and by the evaluation).
        `notes` are added to what the answer says (things the agent noticed that the plan alone does not show)."""
        self.executor.log = []
        trace: list[dict] = [{"step": "plan", "plan": plan}]
        query = validate(plan, self.catalog, today=self.today, value_exists=self.executor.value_exists)
        query.warnings += notes or []
        checks, self.executor.log = self.executor.log, []  # statements the validator ran to look values up in the live data
        trace += checks
        trace.append({"step": "validation", "result": "ok" if query.intent != "clarify" else "needs clarification: " + query.message,
                      "query": None if query.intent == "clarify" else _describe(query), "warnings": list(query.warnings)})
        result = None
        if query.intent in ("search", "aggregate", "get_record"):
            try:
                result = self.executor.run(query, page)
            except EmbeddingError:
                trace += self.executor.log
                return Answer("error", text=EMBEDDING_DOWN, summary=EMBEDDING_DOWN, plan=plan, query=query, trace=trace)
            trace += self.executor.log
        answer = render(query, result, self.executor.page_size)
        answer.plan, answer.trace = plan, trace
        return answer


def _strings(value: Any) -> list[str]:
    """Every string inside a plan's filters (values, lists, dicts), for checking what the plan uses."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _describe(query: Query) -> str:
    """The validated query in one line (filters as the executor will apply them)."""
    parts = [query.intent]
    if query.metric:
        parts.append(f"{query.metric}({query.agg_field or '*'})" + (f" group by {query.group_by}" if query.group_by else "") + (f" top {query.top_n}" if query.top_n else ""))
    parts += [f"{f.field} {f.op} {json.dumps(f.value, ensure_ascii=False, default=str)}" for f in query.filters]
    if query.text_query:
        parts.append(f"by meaning: {query.text_query!r}")
    if query.record_id:
        parts.append(f"record {query.record_id}")
    return " | ".join(parts)
