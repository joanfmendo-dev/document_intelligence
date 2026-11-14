"""The golden set: questions with the plan and the result that are known to be right.

Each case has only what the evaluation needs: `question`, `expected_plan`, `expected_result` (and `previous_plan` for follow-ups).
The results were computed with hand-written SQL, independent of the copilot, on the project's 1,000-record medical_records.duckdb;
the relative dates ("this year", "last 30 days") are fixed by `as_of`, so build the Copilot with today=golden.as_of.
`expected_result` is None where no query runs (clarify, refusals, navigation) or where a person must label the answer (semantic search).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DEFAULT = Path(__file__).parent / "data" / "golden.json"


@dataclass
class GoldenCase:
    id: str
    question: str
    expected_plan: dict
    expected_result: dict | None = None
    previous_plan: dict | None = None

    @property
    def intent(self) -> str:
        return self.expected_plan.get("intent", "")


@dataclass
class Golden:
    as_of: date
    cases: list[GoldenCase]


def load_golden(path: str | Path | None = None) -> Golden:
    data = json.loads(Path(path or DEFAULT).read_text(encoding="utf-8"))
    return Golden(date.fromisoformat(data["as_of"]),
                  [GoldenCase(c["id"], c["question"], c["expected_plan"], c.get("expected_result"), c.get("previous_plan")) for c in data["cases"]])
