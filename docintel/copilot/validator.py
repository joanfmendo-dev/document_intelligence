"""Validation of a plan, in code, before anything touches the database.

`validate(plan, catalog, today=...)` returns a Query. If something is wrong or unclear it returns a Query with intent
"clarify" and the question to ask the user: it never repairs a plan silently and never raises on bad model output.

The checks (design, section 6):
 1. the plan has the expected shape and a known intent
 2. every field exists in the catalog and its operator is allowed for its role
 3. categorical values exist (catalog first, then the live database); otherwise ask, with suggestions computed here
 4. dates are expanded, from <= to, and a range outside the data is flagged
 5. numbers are plausible and units are converted
 6. limits: number of filters, one group_by, top_n, record_id format
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Callable

from . import dates
from .catalog import GROUPABLE, LAB, LAB_FLAG, LAB_NAME, NUMERIC, OPS_BY_ROLE, VITALS, Catalog
from .plan import (DESTINATIONS, HELP_TOPICS, INTENTS, MAX_FILTERS, MAX_IN_VALUES, MAX_TEXT, MAX_TOP_N, METRICS, REASONS,
                   Filter, Query, measurement_number)

RECORD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
AGE_RANGE = (0, 120)
OP_WORDS = {"gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "eq": "="}


_OWN_KEYS = {"search": {"intent", "filters", "text_query"}, "aggregate": {"intent", "filters", "text_query", "aggregate"}}


class _Ask(Exception):
    """Internal: stop validating and ask the user."""

    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.message, self.suggestions = message, suggestions or []


def validate(plan: Any, catalog: Catalog, *, today: date | None = None,
             value_exists: Callable[[str, str], str | None] | None = None) -> Query:
    """`value_exists(field, text)` is the live-database fallback for step 3: it returns the stored spelling of a value the
    catalog does not list (the catalog may be out of date), or None."""
    today = today or date.today()
    try:
        return _validate(plan, catalog, today, value_exists)
    except _Ask as ask:
        return Query("clarify", message=ask.message, suggestions=ask.suggestions)


def _validate(plan: Any, catalog: Catalog, today: date, value_exists) -> Query:
    if not isinstance(plan, dict):
        raise _Ask("I could not turn that into a search. Could you rephrase it?")
    intent = plan.get("intent")
    if intent not in INTENTS:
        raise _Ask("I could not tell what you want to do. Could you rephrase it?")

    if intent == "clarify":
        return Query("clarify", message=str(plan.get("question") or "Could you give me more detail?")[:300],
                     suggestions=[str(x) for x in (plan.get("suggestions") or [])][:5])
    if intent == "out_of_scope":
        reason = plan.get("reason") if plan.get("reason") in REASONS else "unrelated"
        return Query("out_of_scope", reason=reason)
    if intent == "help":
        return Query("help", topic=plan.get("topic") if plan.get("topic") in HELP_TOPICS else "capabilities")

    query = Query(intent)
    if intent == "get_record":
        query.record_id = _record_id(plan.get("record_id"))
        return query
    if intent == "navigate":
        dest = plan.get("destination")
        if dest not in DESTINATIONS:
            raise _Ask("Where would you like to go: the records list, a record, or the upload page?")
        query.destination = dest
        if dest == "record":
            query.record_id = _record_id(plan.get("record_id"))
        if dest == "records":
            query.filters = _filters(plan.get("filters"), catalog, today, value_exists, query)
        return query

    # search / aggregate
    ignored = sorted(k for k, v in plan.items() if k not in _OWN_KEYS[intent] and v not in (None, "", [], {}))
    if ignored:  # parts that belong to another intent (a "topic" on a count): they change nothing, but the user should know
        query.warnings.append(f"I ignored parts of the plan that do not apply here: {', '.join(ignored)}.")
    query.filters = _filters(plan.get("filters"), catalog, today, value_exists, query)
    text = plan.get("text_query")
    if text is not None and str(text).strip():
        query.text_query = str(text).strip()
        if len(query.text_query) > 200:
            raise _Ask("That description is too long. Could you shorten it?")
    if intent == "search":
        if not query.filters and not query.text_query:
            raise _Ask("Which records do you want? For example a department, a diagnosis, a date range or a lab result.")
        return query

    agg = plan.get("aggregate")
    if not isinstance(agg, dict):
        raise _Ask("What would you like to calculate: a count, an average, a minimum or a maximum?")
    metric = agg.get("metric")
    if metric not in METRICS:
        raise _Ask("What would you like to calculate: a count, an average, a minimum or a maximum?")
    query.metric = metric
    if metric == "count" and agg.get("field") not in (None, ""):
        query.warnings.append(f"A count does not use a field, so I ignored “{agg['field']}”.")
    if metric != "count":
        f = agg.get("field")
        if f not in catalog or catalog.role(f) not in NUMERIC:
            raise _Ask(f"I can calculate the {metric} of a numeric field (such as age or a vital sign). Which field do you mean?",
                       [n for n in catalog.filter_fields() if catalog.role(n) in NUMERIC][:5])
        query.agg_field = f
    group = agg.get("group_by")
    if group not in (None, ""):
        if isinstance(group, str) and group in catalog and catalog.role(group) in GROUPABLE and group not in (LAB_NAME, LAB_FLAG):
            query.group_by = group
        else:  # the number the user asked for is still right without the grouping: give it, and say what was left out
            shown = group if isinstance(group, str) else "that"
            options = ", ".join(n for n in catalog.filter_fields() if catalog.role(n) in GROUPABLE)
            query.warnings.append(f"I cannot group by “{shown}”, so this is the overall figure. I can group by: {options}.")
    if not query.filters and not query.text_query and not query.group_by:
        query.warnings.append("No condition was applied, so this covers all the records.")
    top = agg.get("top_n")
    if top not in (None, ""):
        try:
            top = int(top)
        except (TypeError, ValueError):
            raise _Ask("How many groups should I show?") from None
        if not 1 <= top <= MAX_TOP_N:
            raise _Ask(f"I can show between 1 and {MAX_TOP_N} groups.")
        query.top_n = top
    return query


def _record_id(value: Any) -> str:
    if not isinstance(value, str) or not RECORD_ID.match(value.strip()):
        raise _Ask("Which record? Give me its ID.")
    return value.strip()


# ---------------------------------------------------------------- filters
def _filters(raw: Any, catalog: Catalog, today: date, value_exists, query: Query) -> list[Filter]:
    if raw in (None, ""):
        return []
    if not isinstance(raw, list) or not all(isinstance(f, dict) for f in raw):
        raise _Ask("I could not read the conditions of that search. Could you rephrase it?")
    if len(raw) > MAX_FILTERS:
        raise _Ask(f"That search has too many conditions (the limit is {MAX_FILTERS}). Could you narrow it down?")
    out: list[Filter] = []
    for f in raw:
        name, op, value = f.get("field"), f.get("op"), f.get("value")
        if name not in catalog or name in (LAB_NAME, LAB_FLAG):
            raise _Ask(f"I do not have a field called {name!r}. Which field do you mean?", [n for n in catalog.filter_fields()][:6])
        if op not in OPS_BY_ROLE[catalog.role(name)]:
            raise _Ask(f"{name} cannot be searched with '{op}'. It accepts: {', '.join(OPS_BY_ROLE[catalog.role(name)])}.")
        item = _one(name, op, value, catalog, today, value_exists, query)
        if all((item.field, item.op, repr(item.value)) != (o.field, o.op, repr(o.value)) for o in out):
            out.append(item)
    return out


def _one(name: str, op: str, value: Any, catalog: Catalog, today: date, value_exists, query: Query) -> Filter:
    role = catalog.role(name)
    if role in ("categorical", "icd10_hierarchy", "identifier"):
        values = [value] if op == "eq" else value
        if not isinstance(values, list) or not values or len(values) > MAX_IN_VALUES or not all(isinstance(v, str) and v.strip() for v in values):
            raise _Ask(f"Which value of {name}?")
        found = [_known(name, v, catalog, value_exists) for v in values] if role != "identifier" else [v.strip() for v in values]
        if name == "icd10_code":
            found = [v.upper() for v in found]
        return Filter(name, "in", found, f"{name.replace('_', ' ')}: {' or '.join(found)}")
    if role in ("name", "free_text"):
        if not isinstance(value, str) or not value.strip():
            raise _Ask(f"What text should {name} contain?")
        if len(value) > MAX_TEXT:
            raise _Ask("That text is too long to search for. Could you shorten it?")
        return Filter(name, "contains", value.strip(), f"{name.replace('_', ' ')} contains “{value.strip()}”")
    if role == "date":
        try:
            start, end = dates.expand(value, today)
        except dates.BadPeriod as exc:
            raise _Ask(f"I could not read the date condition: {exc}.") from exc
        span = catalog.date_span(name)
        words = dates.describe(start, end)
        if span and (end <= span[0] or start > span[1]):
            query.warnings.append(f"The records cover {span[0]} to {span[1]}; {words} is outside that range.")
        return Filter(name, "between", {"start": start.isoformat(), "end": end.isoformat()}, f"{name.replace('_', ' ')}: {words}")
    if role in NUMERIC:
        return _numeric(name, op, value, catalog)
    if role == "list_of_objects":
        return _labs(name, value, catalog, value_exists, query)
    raise _Ask(f"I cannot search {name}.")  # pragma: no cover - every role is handled above


def _known(name: str, text: str, catalog: Catalog, value_exists) -> str:
    """The stored spelling of a categorical value, or ask. Case differences are not a different value; anything else is asked."""
    text = _without_count(text, name, catalog)
    found = catalog.canonical(name, text)
    if found:
        return found
    if catalog.values(name) is None:  # too many values to list: only the database can say
        live = value_exists(name, text) if value_exists else None
        if live:
            return live
        raise _Ask(f"I could not find “{text}” in {name.replace('_', ' ')}. Could you check the spelling?")
    live = value_exists(name, text) if value_exists else None  # the catalog may be out of date
    if live:
        return live
    near = catalog.suggest(name, text)
    if near:
        raise _Ask(f"I do not see “{text}” in {name.replace('_', ' ')}. Did you mean {_or(near)}?", near)
    raise _Ask(f"There is no {name.replace('_', ' ')} called “{text}” in the records.")


def _without_count(text: str, name: str, catalog: Catalog) -> str:
    """'Cardiology (54)' -> 'Cardiology': the prompt shows record counts in parentheses and small models copy them. Only when the
    text without the count is a real value, so 'Troponin I (initial)' stays as it is."""
    if catalog.canonical(name, text):
        return text
    bare = re.sub(r"\s*\(\d+\)\s*$", "", text)
    return bare if bare != text and catalog.canonical(name, bare) else text


def _or(items: list[str]) -> str:
    quoted = [f"“{i}”" for i in items]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " or " + quoted[-1]


def _numeric(name: str, op: str, value: Any, catalog: Catalog) -> Filter:
    lo_hi = AGE_RANGE if name == "age" else VITALS[name][3] if name in VITALS else None
    unit = catalog.columns[name].unit or ""

    def number(v: Any) -> float | int:
        try:
            n = measurement_number(name, v)
        except (TypeError, ValueError):
            raise _Ask(f"I could not read “{v}” as a number for {name.replace('_', ' ')}.") from None
        if lo_hi and not lo_hi[0] <= n <= lo_hi[1]:
            raise _Ask(f"{n:g} {unit} is not a plausible value for {name.replace('vital_signs.', '').replace('_', ' ')} "
                       f"(expected {lo_hi[0]} to {lo_hi[1]}). Did you use the right unit?")
        return int(n) if name == "age" and n == int(n) else n

    label_name = name.replace("vital_signs.", "").replace("_", " ")
    if op == "between":
        if not isinstance(value, list) or len(value) != 2:
            raise _Ask(f"For a range of {label_name} give two numbers, for example 60 and 70.")
        a, b = number(value[0]), number(value[1])
        if a > b:
            raise _Ask(f"The range of {label_name} starts ({a:g}) above where it ends ({b:g}).")
        return Filter(name, op, [a, b], f"{label_name} {a:g} to {b:g} {unit}".strip())
    n = number(value)
    return Filter(name, op, n, f"{label_name} {OP_WORDS[op]} {n:g} {unit}".strip())


def _labs(name: str, value: Any, catalog: Catalog, value_exists, query: Query) -> Filter:
    if not isinstance(value, dict) or not (value.get("test_name") or value.get("flag")):
        raise _Ask("Which lab test, and which flag (Normal, High, Low or Borderline)?")
    out: dict[str, list[str]] = {}
    for key, col in (("test_name", LAB_NAME), ("flag", LAB_FLAG)):
        raw = value.get(key)
        if raw in (None, "", []):
            continue
        raw = raw if isinstance(raw, list) else [raw]
        if not all(isinstance(r, str) and r.strip() for r in raw) or len(raw) > MAX_IN_VALUES:
            raise _Ask(f"Which {key.replace('_', ' ')}?")
        found: list[str] = []
        for r in raw:
            r = _without_count(r, col, catalog)
            if key == "test_name":
                names = catalog.expand_test_names(r)
                if not names:
                    live = value_exists(col, r) if value_exists else None
                    if live:
                        names = [live]
                    else:
                        near = catalog.suggest(col, r)
                        raise _Ask(f"I do not see a lab test called “{r}”." + (f" Did you mean {_or(near)}?" if near else ""), near)
                if len(names) > 1:
                    query.warnings.append(f"“{r}” matches {_or(names)}; I searched all of them.")
                found += names
            else:
                found.append(_known(col, r, catalog, value_exists))
        out[key] = list(dict.fromkeys(found))
    label = "lab: " + " / ".join(out.get("test_name", ["any test"])) + (f" flagged {' or '.join(out['flag'])}" if out.get("flag") else "")
    return Filter(LAB, "has", out, label)
