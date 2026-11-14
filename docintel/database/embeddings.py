"""The embeddings table and the rules for keeping it in step with the records.

One row per record in `record_embeddings`: the vector, the hash of the text it was computed from, and the model.
A record needs a (new) vector when it has none, when its text changed (hash differs) or when the model changed.
Only the web app and the loaders write this table; the copilot only reads it.

These are plain functions on a DuckDB connection; RecordDB wraps them (embed_record, embed_pending, ...).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from ..shared.settings import s

TABLE = "record_embeddings"
CREATE = (f"CREATE TABLE IF NOT EXISTS {TABLE} (record_id VARCHAR PRIMARY KEY, embedding FLOAT[], "
          "text_hash VARCHAR, model VARCHAR, created_at TIMESTAMP DEFAULT current_timestamp)")


@dataclass
class EmbeddingReport:
    model: str
    embedded: int = 0    # vectors created or replaced
    unchanged: int = 0   # already up to date
    empty: int = 0       # records with no text to embed (their vector, if any, is dropped)
    removed: int = 0     # vectors of records that no longer exist


def embedding_text(record: dict, fields: Iterable[str] | None = None) -> str:
    """The text that represents a record: the chosen narrative fields, always in the same order and format
    ("Chief complaint: ...\\nFinal diagnosis: ..."). Empty when the record has none of them."""
    parts = []
    for field in fields or s.EMBEDDING_FIELDS:
        value = record.get(field)
        if value is not None and str(value).strip():
            parts.append(f"{field.replace('_', ' ').capitalize()}: {str(value).strip()}")
    return "\n".join(parts)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect(con: Any, records_table: str, model: str, fields: Iterable[str] | None = None) -> tuple[list[tuple[str, str, str]], EmbeddingReport]:
    """What has to be embedded: [(record_id, text, hash)] for every record without an up-to-date vector, plus a
    report with the counts that need no work. Also drops vectors of records that were deleted or have no text."""
    fields = list(fields or s.EMBEDDING_FIELDS)
    report = EmbeddingReport(model=model)
    report.removed = con.execute(
        f"DELETE FROM {TABLE} WHERE record_id NOT IN (SELECT record_id FROM {records_table}) RETURNING record_id").fetchall().__len__()
    rows = con.execute(
        f"SELECT m.record_id, {', '.join('m.' + f for f in fields)}, e.text_hash, e.model "
        f"FROM {records_table} m LEFT JOIN {TABLE} e ON e.record_id = m.record_id ORDER BY m.record_id").fetchall()
    todo, drop = [], []
    for row in rows:
        record_id, values, stored_hash, stored_model = row[0], row[1:1 + len(fields)], row[-2], row[-1]
        text = embedding_text(dict(zip(fields, values)), fields)
        if not text:
            report.empty += 1
            if stored_hash is not None:
                drop.append(record_id)
        elif stored_hash == text_hash(text) and stored_model == model:
            report.unchanged += 1
        else:
            todo.append((record_id, text, text_hash(text)))
    for record_id in drop:
        con.execute(f"DELETE FROM {TABLE} WHERE record_id = ?", [record_id])
    return todo, report


def store(con: Any, model: str, rows: list[tuple[str, list[float], str]]) -> None:
    """Save [(record_id, vector, hash)], replacing any earlier vector of the same record."""
    for record_id, vector, digest in rows:
        con.execute(f"INSERT OR REPLACE INTO {TABLE} (record_id, embedding, text_hash, model) VALUES (?, ?, ?, ?)",
                    [record_id, vector, digest, model])
