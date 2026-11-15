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

from datetime import date
from pathlib import Path
from typing import Any

from ..shared.embeddings import EmbeddingError
from ..shared.llm import LLMError
from .catalog import Catalog
from .executor import Executor
from .plan import Query
from .planner import Planner, PlannerError
from .render import Answer, render
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
        self.planner = planner or Planner(catalog, llm)
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
            return render(Query("clarify", message=NOT_UNDERSTOOD))
        except LLMError:
            return Answer("error", text=MODEL_DOWN, summary=MODEL_DOWN)
        return self.answer_plan(plan, page)

    def answer_plan(self, plan: dict, page: int = 1) -> Answer:
        """Validate, run and render a plan without asking the model (used for 'show more' and by the evaluation)."""
        query = validate(plan, self.catalog, today=self.today, value_exists=self.executor.value_exists)
        result = None
        if query.intent in ("search", "aggregate", "get_record"):
            try:
                result = self.executor.run(query, page)
            except EmbeddingError:
                return Answer("error", text=EMBEDDING_DOWN, summary=EMBEDDING_DOWN, plan=plan, query=query)
        answer = render(query, result, self.executor.page_size)
        answer.plan = plan
        return answer
