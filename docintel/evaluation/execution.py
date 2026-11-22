"""Level 1: execution accuracy. Does what the copilot got from the database equal what the golden set says it should?

Like the reference project (agent_eval), a result counts as correct when it holds the same DATA, however it is laid out:
    exact           identical
    equivalent      same data after normalization (row order, rounding, None vs missing)
    equivalent_tie  top-N where the only differences are tied rows at the cut-off
    mismatch        different data (includes "ran no query" and "empty instead of 0")
    n/a             nothing to compare (no query is expected, or a person must label the answer)
A list of records is compared by its total and by the set of record_ids; an aggregate by its rows.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import NormalDist

SCORES = {"exact": 1, "equivalent": 1, "equivalent_tie": 1, "mismatch": 0, "n/a": None}


@dataclass
class Verdict:
    level: str
    detail: str = ""

    @property
    def score(self) -> int | None:
        return SCORES[self.level]


def ids_digest(ids: list[str]) -> str:
    return hashlib.md5("\n".join(sorted(ids)).encode()).hexdigest()


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _decimals(x) -> int:
    if isinstance(x, int):
        return 0
    text = repr(float(x))
    return 8 if "e" in text.lower() else len(text.split(".")[1]) if "." in text else 0


def values_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if _is_num(a) and _is_num(b):
        if isinstance(a, int) or isinstance(b, int):  # an integer is an exact count: 5 is not 5.5
            return a == b
        d = min(_decimals(a), _decimals(b))  # compare at the coarser precision, like the reference project
        return abs(a - b) <= 0.5 * 10 ** (-d) + 1e-9
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    return a == b


def _rows_equal_unordered(a: list[list], b: list[list]) -> bool:
    if len(a) != len(b):
        return False
    used: set[int] = set()
    for row in a:
        for j, other in enumerate(b):
            if j not in used and len(row) == len(other) and all(values_equal(x, y) for x, y in zip(row, other)):
                used.add(j)
                break
        else:
            return False
    return True


def _tie_match(a: list[list], b: list[list]) -> bool:
    """Top-N: numbers match in order; strings may differ only on rows whose number equals the last (cut-off) number."""
    if len(a) != len(b) or not a or len(a[0]) != 2:
        return False
    num = next((i for i in range(2) if all(_is_num(r[i]) for r in a)), None)
    if num is None or not all(values_equal(x[num], y[num]) for x, y in zip(a, b)):
        return False
    other = 1 - num
    last = a[-1][num]
    return all(values_equal(x[other], y[other]) or values_equal(x[num], last) for x, y in zip(a, b))


def compare_aggregate(expected_rows: list[list], agent_rows: list[list], ordered_limit: bool = False) -> Verdict:
    if expected_rows == agent_rows:
        return Verdict("exact")
    if _rows_equal_unordered(expected_rows, agent_rows):
        return Verdict("equivalent", "same rows, different order or rounding")
    if ordered_limit and _tie_match(expected_rows, agent_rows):
        return Verdict("equivalent_tie", "differences only among tied rows at the cut-off")
    if not agent_rows and expected_rows and all(v == 0 for r in expected_rows for v in r if _is_num(v)):
        return Verdict("mismatch", "agent returned no rows; the correct answer is 0")
    return Verdict("mismatch", "values differ")


def compare_list(expected: dict, agent_result: dict | None) -> Verdict:
    """`expected` = {"total", "ids_md5"[, "ids"]}; `agent_result` = Result.to_dict()."""
    if agent_result is None:
        return Verdict("mismatch", "agent ran no query")
    ids = agent_result.get("ids") or []
    if agent_result.get("total") != expected["total"]:
        return Verdict("mismatch", f"total {agent_result.get('total')} instead of {expected['total']}")
    if expected["total"] and ids_digest(ids) != expected["ids_md5"]:
        return Verdict("mismatch", "same total but different records")
    return Verdict("exact")


def execution_verdict(expected_result: dict | None, expected_intent: str, agent_result: dict | None) -> Verdict:
    """Level-1 verdict for one case. `agent_result` is Result.to_dict() (None if the copilot ran no query)."""
    if expected_result is None or expected_intent not in ("search", "get_record", "aggregate"):
        return Verdict("n/a", "no result to compare")
    if agent_result is None:
        return Verdict("mismatch", "agent ran no query")
    if "rows" in expected_result:
        if agent_result.get("kind") != "aggregate":
            return Verdict("mismatch", "agent did not run an aggregate")
        cols = agent_result["columns"]
        rows = [[r.get(c) for c in cols] for r in agent_result["rows"]]
        return compare_aggregate(expected_result["rows"], rows, ordered_limit=True)
    if agent_result.get("kind") not in ("search", "record"):
        return Verdict("mismatch", "agent did not run a search")
    return compare_list(expected_result, agent_result)


def wilson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
