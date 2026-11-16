"""The answer the user reads, built from the plan and the result by templates.

No model writes any of it, so no number can be invented: every figure comes from the Result, and the line that says how the
question was understood ("Cardiology · age > 60") is generated from the validated plan, so the user can see, and correct, what was
asked. Fixed messages cover help, refusals and navigation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from urllib.parse import quote, urlencode

from .executor import Result
from .plan import Query

OUT_OF_SCOPE = {
    "write": "I can only read the records. To add, change or delete a record, use the Upload and Records pages of DocBase.",
    "export": "I cannot export results yet. I can take you to the Records page, where you can filter the list.",
    "clinical_advice": "I help find information in the records. I cannot give medical advice or interpret results for a patient.",
    "unrelated": "I can only help with the medical records in DocBase: searching them, counting them, and taking you to the right page.",
}
HELP = {
    "capabilities": ("I can search the records (by department, sex, insurance, city, age, dates, diagnosis, ICD-10, lab results and vital signs, "
                     "or by describing a case in words), count and average them, open a record by its ID, and take you to the Records or "
                     "Upload pages. I only read: I never change or delete anything."),
    "upload_formats": "You can upload a PDF, a Word document (.docx), an image (PNG, JPG) or a plain text file on the Upload page.",
    "fields": ("Each record has the patient, department, dates, complaint, history, medications, allergies, vital signs, exam, lab results, "
               "imaging, diagnosis (with its ICD-10 code), plan and notes. Ask about any of them."),
    "search_examples": ("Try: “Cardiology patients over 60 with high troponin this year”, “How many encounters per department in 2025?”, "
                        "“Show record SYNUS-2026-0001”, or “patients with trouble breathing at night”."),
}
LINKS = {"record": "/records/{id}/edit", "records": "/records", "upload": "/upload"}


@dataclass
class Link:
    label: str
    url: str


@dataclass
class Answer:
    kind: str                                  # results | aggregate | record | navigate | help | clarify | out_of_scope | error
    text: str = ""                             # the whole answer as plain text (what the judge reads)
    interpretation: str = ""                   # how the question was understood, in words
    summary: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)       # warnings and zero-result hints
    suggestions: list[str] = field(default_factory=list)
    plan: dict | None = None                   # what the model proposed: kept for follow-up questions
    query: Query | None = None
    result: Result | None = None
    page: int = 1
    has_more: bool = False
    trace: list[dict] = field(default_factory=list)      # what was done to get this answer, in order (see explain())

    def explain(self, system_chars: int = 300) -> str:
        """The trace as readable text: the model's raw replies, the plan, the validated query and every SQL statement with its parameters."""
        out = []
        for i, s in enumerate(self.trace, 1):
            kind = s["step"]
            if kind == "llm":
                out.append(f"{i}. LLM call {s['attempt']} ({s.get('seconds', '?')} s)")
                for m in s["messages"]:
                    text = m["content"] if m["role"] != "system" else m["content"][:system_chars] + f"… [system prompt, {len(m['content'])} characters]"
                    out.append(f"   [{m['role']}] {text}")
                out.append(f"   -> {s.get('reply') if 'reply' in s else 'ERROR ' + s.get('error', '')}")
                if s.get("error") and "reply" in s:
                    out.append(f"   unusable: {s['error']}")
            elif kind == "plan":
                out.append(f"{i}. Plan: {json.dumps(s['plan'], ensure_ascii=False)}")
            elif kind == "validation":
                out.append(f"{i}. Validation: {s['result']}")
                out += [f"   query: {s['query']}"] if s.get("query") else []
                out += [f"   note: {w}" for w in s.get("warnings", [])]
            elif kind == "sql":
                out.append(f"{i}. SQL ({s['rows']} row{'' if s['rows'] == 1 else 's'}, {s['seconds']} s): {s['sql']}")
                out.append(f"   params: {s['params']}")
            elif kind == "embedding":
                out.append(f"{i}. Embedding of the question with {s['model']}: {s['text']!r} ({s['dimensions']} dimensions, {s['seconds']} s)")
            else:
                out.append(f"{i}. {kind}: {s}")
        return "\n".join(out)


def interpretation(query: Query) -> str:
    if query.intent == "aggregate":
        what = "count of records" if query.metric == "count" else f"{query.metric} of {query.agg_field.replace('vital_signs.', '').replace('_', ' ')}"
        parts = [what + (f" by {query.group_by.replace('_', ' ')}" if query.group_by else "")]
        if query.top_n:
            parts.append(f"top {query.top_n}")
    elif query.intent == "get_record":
        return f"record {query.record_id}"
    else:
        parts = []
    parts += [f.label for f in query.filters]
    if query.text_query:
        parts.append(f"similar to “{query.text_query}”")
    return " · ".join(parts) if parts else "all records"


def _cell(value) -> str:
    if value is None:
        return "(none)"
    return str(value)[:10] if isinstance(value, str) and len(value) >= 19 and value[4] == "-" and value[10] == "T" else str(value)


def _table_text(columns: list[str], rows: list[dict]) -> list[str]:
    return [" | ".join(_cell(r.get(c)) for c in columns) for r in rows]


def link_for(destination: str, record_id: str | None = None, department: str | None = None) -> Link:
    if destination == "record":
        return Link(f"Open record {record_id}", LINKS["record"].format(id=quote(record_id, safe="")))
    if destination == "records":
        return Link("Open the Records page" + (f" for {department}" if department else ""),
                    LINKS["records"] + (f"?{urlencode({'d': department})}" if department else ""))
    return Link("Open the Upload page", LINKS["upload"])


def render(query: Query, result: Result | None = None, page_size: int = 25) -> Answer:
    """The Answer for a validated Query (and its Result, when it ran one)."""
    ans = Answer("results", query=query, result=result, interpretation=interpretation(query), notes=list(query.warnings))

    if query.intent == "clarify":
        ans.kind, ans.interpretation = "clarify", ""
        ans.summary, ans.suggestions = query.message, query.suggestions
        ans.text = query.message
    elif query.intent == "out_of_scope":
        ans.kind, ans.interpretation = "out_of_scope", ""
        ans.summary = ans.text = OUT_OF_SCOPE[query.reason]
        if query.reason == "export":
            ans.links = [link_for("records")]
    elif query.intent == "help":
        ans.kind, ans.interpretation = "help", ""
        ans.summary = ans.text = HELP[query.topic]
    elif query.intent == "navigate":
        ans.kind, ans.interpretation = "navigate", ""
        departments = next((f.value for f in query.filters if f.field == "department"), [])
        link = link_for(query.destination, query.record_id, departments[0] if len(departments) == 1 else None)
        ans.links = [link]
        ans.summary = "Here is the link: " + link.label + "."
        if query.destination == "records" and any(f.field != "department" for f in query.filters):
            ans.notes.append("The Records page can only filter by department; ask me here to search by anything else.")
        ans.text = ans.summary + "".join("\n" + n for n in ans.notes)
    elif query.intent == "get_record":
        ans.kind = "record"
        if result is not None and result.record is None and result.ids:  # a partial id that fits several records
            ans.kind = "clarify"
            ans.summary = f"“{query.record_id}” fits {len(result.ids)} records. Which one do you mean?"
            ans.suggestions = list(result.ids)
        elif result is None or result.record is None:
            ans.summary = f"There is no record {query.record_id}."
        else:
            r = result.record
            ans.summary = f"Record {r['record_id']}: {r.get('patient_name') or 'unnamed patient'}, {r.get('department') or 'no department'}."
            ans.links = [link_for("record", r["record_id"])]
        ans.text = ans.interpretation + "\n" + ans.summary
    elif query.intent == "aggregate":
        ans.kind = "aggregate"
        ans.columns, ans.rows = result.columns, result.rows
        if query.group_by:
            ans.summary = f"{result.total} group{'s' if result.total != 1 else ''}." if result.total else "No records match."
        else:
            value = result.rows[0][result.columns[0]] if result.rows else None
            ans.summary = (f"{_cell(value)} record{'' if value == 1 else 's'} match." if query.metric == "count"
                           else f"{result.columns[0].replace('_', ' ')}: {_cell(value)}")
        ans.text = "\n".join([ans.interpretation, ans.summary, *_table_text(ans.columns, ans.rows)[:50]] if query.group_by else [ans.interpretation, ans.summary])
    else:  # search
        ans.columns, ans.rows, ans.page = result.columns, result.rows, result.page
        semantic = bool(query.text_query)
        if result.total == 0:
            ans.summary = "No records match."
            ans.notes += [f"Without “{w}” there {'is' if n == 1 else 'are'} {n} match{'' if n == 1 else 'es'}." for w, n in result.hints[:3]]
        else:
            first = (result.page - 1) * page_size + 1
            last = first + len(result.rows) - 1
            noun = ("semantic match" + ("" if result.total == 1 else "es")) if semantic else ("record" + ("" if result.total == 1 else "s"))
            ans.summary = f"{result.total} {noun}"
            ans.summary += " found" + (f". Showing {first}–{last}." if result.total > len(result.rows) else ".")
            if semantic:
                ans.summary += " These are matches by meaning, not exact text; the most similar come first."
            ans.has_more = last < result.total
        if result.coverage and result.coverage[0] < result.coverage[1]:
            ans.notes.append(f"Semantic search covers {result.coverage[0]} of {result.coverage[1]} records; the rest have no embedding yet.")
        ans.text = "\n".join([ans.interpretation, ans.summary, *_table_text(ans.columns, ans.rows), *ans.notes])
        return ans
    if ans.notes and ans.kind not in ("navigate",):
        ans.text += "".join("\n" + n for n in ans.notes)
    return ans
