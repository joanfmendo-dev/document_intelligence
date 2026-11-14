"""The plan: what the model returns, and the validated Query the rest of the copilot works with.

The model never writes SQL. It describes WHAT is asked as a small JSON object:

    {"intent": "search", "filters": [{"field": "department", "op": "in", "value": ["Cardiology"]},
                                     {"field": "age", "op": "gt", "value": 60}], "text_query": null}
    {"intent": "aggregate", "filters": [...], "aggregate": {"metric": "count", "group_by": "department", "top_n": null}}

Other intents: get_record (record_id), navigate (destination), help (topic), clarify (question), out_of_scope (reason).
Filters are a flat list joined by AND; OR inside one field is written with op "in". validator.py turns a plan into a Query
or into a clarification; nothing reaches the database without passing it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .catalog import ALL_OPS, Catalog

INTENTS = ("search", "aggregate", "get_record", "navigate", "help", "clarify", "out_of_scope")
DESTINATIONS = ("records", "record", "upload")
HELP_TOPICS = ("capabilities", "upload_formats", "fields", "search_examples")
REASONS = ("write", "export", "clinical_advice", "unrelated")
METRICS = ("count", "avg", "min", "max")
MAX_FILTERS, MAX_TOP_N, MAX_IN_VALUES, MAX_TEXT = 8, 50, 20, 100


class PlanFormatError(ValueError):
    """The model's answer is not a usable JSON plan."""


@dataclass
class Filter:
    field: str
    op: str
    value: Any
    label: str  # the condition in words, for the interpretation line


@dataclass
class Query:
    """A plan that passed validation. For intent 'clarify' it carries the question to ask the user."""
    intent: str
    filters: list[Filter] = field(default_factory=list)
    text_query: str | None = None
    metric: str | None = None
    agg_field: str | None = None
    group_by: str | None = None
    top_n: int | None = None
    record_id: str | None = None
    destination: str | None = None
    topic: str | None = None
    reason: str | None = None
    message: str = ""                                          # clarify: the question
    suggestions: list[str] = field(default_factory=list)       # clarify: values the user may have meant
    warnings: list[str] = field(default_factory=list)          # things the answer must say (a range outside the data...)


def plan_schema(catalog: Catalog) -> dict:
    """A light JSON schema for the model's answer: the enumerations (intent, field, op) and nothing about values,
    which the validator checks against the real data."""
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(INTENTS)},
            "filters": {"type": "array", "items": {"type": "object", "properties": {
                "field": {"type": "string", "enum": catalog.filter_fields()},
                "op": {"type": "string", "enum": list(ALL_OPS)},
                "value": {}}, "required": ["field", "op", "value"]}},
            "text_query": {"type": ["string", "null"]},
            "aggregate": {"type": ["object", "null"], "properties": {
                "metric": {"type": "string", "enum": list(METRICS)},
                "field": {"type": ["string", "null"]},
                "group_by": {"type": ["string", "null"]},
                "top_n": {"type": ["integer", "null"]}}},
            "record_id": {"type": ["string", "null"]},
            "destination": {"type": ["string", "null"], "enum": [*DESTINATIONS, None]},
            "topic": {"type": ["string", "null"], "enum": [*HELP_TOPICS, None]},
            "reason": {"type": ["string", "null"], "enum": [*REASONS, None]},
            "question": {"type": ["string", "null"]},
        },
        "required": ["intent"],
    }


def parse_plan(text: str) -> dict:
    """The first JSON object in the model's text (models sometimes wrap it in prose or a code fence)."""
    text = (text or "").strip()
    start = text.find("{")
    if start < 0:
        raise PlanFormatError("the answer has no JSON object")
    try:
        plan, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise PlanFormatError(f"the answer is not valid JSON: {exc.msg}") from exc
    if not isinstance(plan, dict):
        raise PlanFormatError("the plan must be a JSON object")
    return plan


# ---------------------------------------------------------------- numbers and units
def measurement_number(field: str, value: Any) -> float:
    """A number for a measurement or integer filter. Accepts 39, "39" and {"value": 39, "unit": "C"}; temperature in Celsius
    is converted to Fahrenheit (the unit the data uses). Raises ValueError."""
    unit = None
    if isinstance(value, dict):
        value, unit = value.get("value"), str(value.get("unit") or "").strip().lstrip("°").upper()
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{value!r} is not a number")
    number = float(str(value).replace(",", "")) if isinstance(value, str) else float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{value!r} is not a number")
    if field.endswith("temperature") and unit in ("C", "CELSIUS"):
        number = number * 9 / 5 + 32
    return round(number, 1)


# ---------------------------------------------------------------- comparing plans (evaluation)
def canonical_plan(plan: dict, catalog: Catalog | None = None) -> dict:
    """The plan in a normal form, so two plans that ask the same thing compare equal: no empty keys, filters sorted,
    lists of values sorted, `eq` on a categorical written as `in`, units converted, text lower-cased."""
    def clean_value(f: dict) -> Any:
        v, name, op = f.get("value"), f.get("field", ""), f.get("op")
        if op == "has" and isinstance(v, dict):
            return {k: sorted(map(str, x if isinstance(x, list) else [x])) for k, x in v.items() if x}
        if isinstance(v, dict) and "unit" in v:
            try:
                return measurement_number(name, v)
            except ValueError:
                return v
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return round(float(v), 1)
        if isinstance(v, list):
            return sorted((str(x).casefold() for x in v)) if all(isinstance(x, str) for x in v) else v
        if isinstance(v, str):
            return v.casefold()
        return v

    filters = []
    for f in plan.get("filters") or []:
        op, value = f.get("op"), clean_value(f)
        column = catalog.columns.get(f.get("field")) if catalog else None
        if op == "eq" and isinstance(value, str) and (column is None or column.role in ("categorical", "icd10_hierarchy")):
            op, value = "in", [value]
        if op == "contains" and isinstance(value, str):
            value = value.strip()
        filters.append({"field": f.get("field"), "op": op, "value": value})
    filters.sort(key=lambda f: (f["field"], f["op"], json.dumps(f["value"], sort_keys=True, default=str)))
    out: dict[str, Any] = {"intent": plan.get("intent")}
    if filters:
        out["filters"] = filters
    if plan.get("text_query"):
        out["text_query"] = re.sub(r"\s+", " ", str(plan["text_query"]).strip().casefold())
    agg = plan.get("aggregate")
    if agg:
        out["aggregate"] = {k: v for k, v in agg.items() if v not in (None, "")}
    for key in ("record_id", "destination", "topic", "reason"):
        if plan.get(key):
            out[key] = plan[key]
    return out


def plans_equivalent(expected: dict, actual: dict, catalog: Catalog | None = None) -> bool:
    return canonical_plan(expected, catalog) == canonical_plan(actual, catalog)
