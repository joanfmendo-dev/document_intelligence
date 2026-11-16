"""Runs a validated Query against DuckDB, read only.

The model never writes SQL: this module builds every statement from the Query, with values passed as parameters (never pasted
into the text). Column and key names that appear in the SQL come from constants in this package, not from the user or the model.
As defence in depth, every statement must still be a single SELECT/WITH, and `Executor.open` opens the database read-only with
file access switched off, so even a bug here could not change a record or read a file.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from ..database import embeddings as emb
from ..database.records import COLUMNS, TABLE, _from_db
from ..shared.embeddings import Embedder
from ..shared.settings import s
from .catalog import LAB, LAB_FLAG, LAB_NAME, VITALS
from .plan import Filter, Query

SUMMARY = ("record_id", "patient_name", "department", "encounter_date", "age", "final_diagnosis")
MAX_IDS = 5000  # the most record ids kept for one search (the page shows far fewer)


@dataclass
class Result:
    kind: str                                              # "search" | "aggregate" | "record"
    total: int = 0                                         # records that match (search) / groups (aggregate)
    ids: list[str] = field(default_factory=list)           # every matching record_id, best first (capped at MAX_IDS)
    rows: list[dict] = field(default_factory=list)         # this page (search), the groups (aggregate)
    columns: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)  # semantic search: similarity of each hit
    record: dict | None = None                             # get_record
    page: int = 1
    page_size: int = 25
    capped: bool = False                                   # more matches than MAX_IDS
    hints: list[tuple[str, int]] = field(default_factory=list)  # zero results: (condition to drop, matches without it)
    coverage: tuple[int, int] | None = None                # semantic search: (records with a vector, records)

    def to_dict(self) -> dict:
        """The result in plain data: what the judge and the evaluation read."""
        out = {"kind": self.kind, "total": self.total, "page": self.page, "page_size": self.page_size, "rows": self.rows,
               "columns": self.columns}
        if self.kind in ("search", "record"):
            out["ids"] = self.ids
        if self.scores:
            out["scores"] = {k: round(v, 3) for k, v in self.scores.items()}
        if self.hints:
            out["hints"] = [{"without": w, "matches": n} for w, n in self.hints]
        if self.record is not None:
            out["record_id"] = self.record.get("record_id")
        return out


def _vital(name: str) -> str:
    key, regex, _, _ = VITALS[name]
    return f"TRY_CAST(NULLIF(regexp_extract(json_extract_string(m.vital_signs, '$.{key}'), '{regex}', 1), '') AS DOUBLE)"


def _number_expr(name: str) -> str:
    return "m.age" if name == "age" else _vital(name)


def _escape_like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _where(filters: list[Filter]) -> tuple[list[str], list[Any]]:
    clauses, params = [], []
    for f in filters:
        if f.field == LAB:
            sub, p = [], []
            for key, col in (("test_name", "test_name"), ("flag", "flag")):
                values = f.value.get(key)
                if values:
                    sub.append(f"json_extract_string(x, '$.{col}') IN ({', '.join('?' * len(values))})")
                    p += values
            clauses.append("EXISTS (SELECT 1 FROM (SELECT unnest(from_json(m.lab_results, '[\"JSON\"]')) AS x) WHERE " + " AND ".join(sub) + ")")
            params += p
        elif f.op == "in":
            clauses.append(f"m.{f.field} IN ({', '.join('?' * len(f.value))})")
            params += f.value
        elif f.op == "contains":
            clauses.append(f"m.{f.field} ILIKE ? ESCAPE '\\'")
            params.append(_escape_like(f.value))
        elif f.field in ("dob", "encounter_date"):
            clauses.append(f"(m.{f.field} >= ? AND m.{f.field} < ?)")
            params += [datetime.fromisoformat(f.value["start"]), datetime.fromisoformat(f.value["end"])]
        else:  # numeric
            expr = _number_expr(f.field)
            if f.op == "between":
                clauses.append(f"({expr} BETWEEN ? AND ?)")
                params += f.value
            else:
                clauses.append(f"({expr} {dict(eq='=', gt='>', gte='>=', lt='<', lte='<=')[f.op]} ?)")
                params.append(f.value)
    return clauses, params


class Executor:
    def __init__(self, con: Any, *, embedder: Embedder | None = None, page_size: int | None = None):
        """`con` is a DuckDB connection (or cursor). To run on its own use Executor.open(path), which opens read-only."""
        self.con = con
        self._embedder = embedder
        self.page_size = page_size or s.COPILOT_PAGE_SIZE
        self._columns: list[str] | None = None
        self.log: list[dict] = []  # every statement run and the embedding calls, in order (the agent copies it into the answer's trace)

    @classmethod
    def open(cls, path: str | Path, **kw) -> "Executor":
        """A read-only connection with file access off: nothing run through it can change the database."""
        return cls(duckdb.connect(str(path), read_only=True, config={"enable_external_access": False}), **kw)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def close(self) -> None:
        self.con.close()

    @property
    def columns(self) -> list[str]:
        """The table's columns that exist in this file (a database made before the ICD-10 columns lacks them)."""
        if self._columns is None:
            present = {r[1] for r in self.con.execute(f"PRAGMA table_info('{TABLE}')").fetchall()}
            self._columns = [c for c in COLUMNS if c in present]
        return self._columns

    def _select(self, sql: str, params: list | tuple = ()) -> list[tuple]:
        if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.I) or ";" in sql:
            raise ValueError("the copilot only runs a single SELECT")
        started = time.perf_counter()
        rows = self.con.execute(sql, list(params)).fetchall()
        shown = [p if not isinstance(p, list) or len(p) <= 8 else f"<list of {len(p)} numbers>" for p in params]  # an embedding is not worth printing
        self.log.append({"step": "sql", "sql": " ".join(sql.split()), "params": shown, "rows": len(rows), "seconds": round(time.perf_counter() - started, 3)})
        return rows

    # ------------------------------------------------------------------ the live fallback of the validator
    def value_exists(self, name: str, text: str) -> str | None:
        """The stored spelling of a value the catalog does not list (the catalog may be out of date), or None."""
        if name in (LAB_NAME, LAB_FLAG):
            key = "test_name" if name == LAB_NAME else "flag"
            rows = self._select(f"SELECT DISTINCT json_extract_string(x, '$.{key}') FROM (SELECT unnest(from_json(lab_results, '[\"JSON\"]')) AS x "
                                f"FROM {TABLE} WHERE lab_results IS NOT NULL) WHERE lower(json_extract_string(x, '$.{key}')) = lower(?)", [text.strip()])
        elif name in COLUMNS and COLUMNS[name].startswith("VARCHAR"):
            rows = self._select(f"SELECT DISTINCT {name} FROM {TABLE} WHERE lower({name}) = lower(?)", [text.strip()])
        else:
            return None
        return rows[0][0] if rows else None

    # ------------------------------------------------------------------ run
    def run(self, query: Query, page: int = 1) -> Result:
        if query.intent == "get_record":
            return self._record(query.record_id)
        if query.intent == "aggregate":
            return self._aggregate(query)
        return self._search(query, max(1, int(page)))

    def _record(self, record_id: str) -> Result:
        row = self._select(f"SELECT {', '.join(self.columns)} FROM {TABLE} WHERE record_id = ?", [record_id])
        if not row and record_id.strip():  # a partial id ('0007') names the record whose id ends with it ('SYNUS-2026-0007')
            tail = record_id.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            found = [r[0] for r in self._select(f"SELECT record_id FROM {TABLE} WHERE record_id LIKE ? ESCAPE '\\' ORDER BY record_id LIMIT 6",
                                                ["%-" + tail])]
            if len(found) > 1:  # ambiguous: the caller asks which one
                return Result("record", total=0, ids=found)
            if found:
                record_id = found[0]
                row = self._select(f"SELECT {', '.join(self.columns)} FROM {TABLE} WHERE record_id = ?", [record_id])
        record = {c: _from_db(c, v) for c, v in zip(self.columns, row[0])} if row else None
        return Result("record", total=1 if record else 0, ids=[record_id] if record else [], record=record, rows=[record] if record else [])

    def _search(self, query: Query, page: int) -> Result:
        clauses, params = _where(query.filters)
        extra = (" AND " + " AND ".join(clauses)) if clauses else ""
        scores: dict[str, float] = {}
        coverage = None
        if query.text_query:
            started = time.perf_counter()
            vector = self.embedder.embed_query(query.text_query)  # raises EmbeddingError if the model is down
            self.log.append({"step": "embedding", "model": self.embedder.model, "text": query.text_query, "dimensions": len(vector),
                             "seconds": round(time.perf_counter() - started, 3)})
            rows = self._select(
                f"SELECT m.record_id, h.score FROM {TABLE} m JOIN (SELECT record_id, list_cosine_similarity(embedding, ?::FLOAT[]) AS score "
                f"FROM {emb.TABLE} WHERE model = ?) h ON h.record_id = m.record_id WHERE h.score >= ?{extra} "
                f"ORDER BY h.score DESC, m.record_id LIMIT ?",
                [vector, self.embedder.model, s.SEMANTIC_MIN_SIMILARITY, *params, s.SEMANTIC_TOP_K])
            raw_hits = len(rows)
            if rows and s.SEMANTIC_RELATIVE_MARGIN is not None:  # scores of this embedding model are compressed: keep what is near the best
                cutoff = float(rows[0][1]) - s.SEMANTIC_RELATIVE_MARGIN
                rows = [r for r in rows if float(r[1]) >= cutoff]
            ids = [r[0] for r in rows]
            scores = {r[0]: float(r[1]) for r in rows}
            total, capped = len(ids), False
            self.log.append({"step": "semantic_filter", "above_min_similarity": raw_hits, "kept": len(rows),
                             "min_similarity": s.SEMANTIC_MIN_SIMILARITY, "relative_margin": s.SEMANTIC_RELATIVE_MARGIN})
            done = self._select(f"SELECT count(*) FROM {emb.TABLE} WHERE model = ?", [self.embedder.model])[0][0]
            coverage = (done, self._select(f"SELECT count(*) FROM {TABLE}")[0][0])
        else:
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            total = self._select(f"SELECT count(*) FROM {TABLE} m {where}", params)[0][0]
            ids = [r[0] for r in self._select(f"SELECT m.record_id FROM {TABLE} m {where} ORDER BY m.encounter_date DESC NULLS LAST, m.record_id LIMIT {MAX_IDS}", params)]
            capped = total > len(ids)
        start = (page - 1) * self.page_size
        wanted = ids[start:start + self.page_size]
        rows = []
        if wanted:
            got = {r[0]: dict(zip(SUMMARY, r)) for r in self._select(
                f"SELECT {', '.join(SUMMARY)} FROM {TABLE} WHERE record_id IN ({', '.join('?' * len(wanted))})", wanted)}
            rows = [{**got[i], "encounter_date": _from_db("encounter_date", got[i]["encounter_date"])} for i in wanted if i in got]
            if scores:
                rows = [{**r, "score": round(scores[r["record_id"]], 3)} for r in rows]
        result = Result("search", total, ids, rows, list(SUMMARY) + (["score"] if scores else []), scores, None, page, self.page_size, capped,
                        coverage=coverage)
        if total == 0 and not query.text_query and len(query.filters) >= 2:
            result.hints = self._hints(query.filters)
        return result

    def _hints(self, filters: list[Filter]) -> list[tuple[str, int]]:
        """When nothing matches: how many records match if each condition is dropped, best first (only those that give any)."""
        out = []
        for i, f in enumerate(filters):
            clauses, params = _where(filters[:i] + filters[i + 1:])
            n = self._select(f"SELECT count(*) FROM {TABLE} m WHERE " + " AND ".join(clauses), params)[0][0]
            if n:
                out.append((f.label, n))
        return sorted(out, key=lambda x: -x[1])

    def _aggregate(self, query: Query) -> Result:
        clauses, params = _where(query.filters)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        expr = {"count": "count(*)",
                "avg": f"round(avg({_number_expr(query.agg_field)}), 1)" if query.agg_field else "",
                "min": f"min({_number_expr(query.agg_field)})" if query.agg_field else "",
                "max": f"max({_number_expr(query.agg_field)})" if query.agg_field else ""}[query.metric]
        label = "count" if query.metric == "count" else f"{query.metric}_{query.agg_field.replace('vital_signs.', '')}"
        if query.group_by:
            limit = f" LIMIT {int(query.top_n)}" if query.top_n else ""
            rows = self._select(f"SELECT m.{query.group_by} AS grp, {expr} AS value FROM {TABLE} m {where} GROUP BY grp "
                                f"ORDER BY value DESC NULLS LAST, grp{limit}", params)
            cols = [query.group_by, label]
        else:
            rows = self._select(f"SELECT {expr} AS value FROM {TABLE} m {where}", params)
            cols = [label]
        data = [dict(zip(cols, r)) for r in rows]
        return Result("aggregate", total=len(data), rows=data, columns=cols)
