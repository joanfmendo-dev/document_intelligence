"""get_fields: turn extracted text into a dictionary that follows a JSON template.

    fields = get_fields(extract_auto("record.pdf")["text"], json_format)

`json_format` only defines the SHAPE: its keys, the type of each value (string, integer, list of
objects...) and the date format. Its values are never sent to the model, because small models copy
example values into fields the document does not contain.

The model is whatever `settings.s.LLM_MODEL` says (default: gemma3:4b through Ollama), or the `llm`
you pass in; nothing here is specific to one model.
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from typing import Any

from .llm import LLM
from .settings import s

SYSTEM_PROMPT = (
    "You extract structured data from the text of a document.\n"
    "- Use only information that is in the document. Never invent, guess or complete anything.\n"
    "- Copy text values exactly as written in the document, including punctuation and units.\n"
    "- If the document does not give a value, use null (an empty list for lists).\n"
    "- Dates and datetimes use ISO 8601: YYYY-MM-DDTHH:MM:SS (time 00:00:00 when the document has none).\n"
    "- Numbers are JSON numbers, not text.\n"
    "- Reply with a single JSON object and nothing else."
)

_ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?$")
_HINT = {"boolean": "<true or false, or null>", "integer": "<integer, or null>", "number": "<number, or null>",
         "datetime": "<ISO 8601 datetime YYYY-MM-DDTHH:MM:SS, or null>",
         "string": "<text copied from the document, or null>", "any": "<value, or null>"}
_JSON_TYPE = {"boolean": "boolean", "integer": "integer", "number": "number", "datetime": "string", "string": "string"}


def _kind(template: Any) -> str:
    """What a template value stands for: object, array, boolean, integer, number, datetime, string or any."""
    if isinstance(template, dict):
        return "object"
    if isinstance(template, list):
        return "array"
    if isinstance(template, bool):  # before int: bool is a subclass of int
        return "boolean"
    if isinstance(template, int):
        return "integer"
    if isinstance(template, float):
        return "number"
    if isinstance(template, str):
        return "datetime" if _ISO_DATETIME.match(template) else "string"
    return "any"


# ---------------------------------------------------------------- what the template means
def _hint(template: Any) -> Any:
    """The template with every value replaced by a description of the expected type."""
    kind = _kind(template)
    if kind == "object":
        return {k: _hint(v) for k, v in template.items()}
    if kind == "array":
        return [_hint(template[0])] if template else []
    return _HINT[kind]


def _schema(template: Any) -> dict:
    """JSON Schema of the template (every value may be null). Providers with structured output enforce it."""
    kind = _kind(template)
    if kind == "object":
        return {"type": "object", "properties": {k: _schema(v) for k, v in template.items()},
                "required": list(template), "additionalProperties": False}
    if kind == "array":
        return {"type": "array", "items": _schema(template[0]) if template else {}}
    return {"type": [_JSON_TYPE[kind], "null"]} if kind in _JSON_TYPE else {}


# ---------------------------------------------------------------- reading the answer
def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    text = fence.group(1) if fence else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:  # e.g. chatter before or after the object
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("the reply is JSON but not an object")
    return data


def _conform(value: Any, template: Any) -> Any:
    """Force `value` into the template's shape and types. Missing or wrong-typed values become null
    (an empty list for lists); keys that are not in the template are dropped."""
    kind = _kind(template)
    if kind == "object":
        value = value if isinstance(value, dict) else {}
        return {k: _conform(value.get(k), sub) for k, sub in template.items()}
    if kind == "array":
        item = template[0] if template else None
        return [v if item is None else _conform(v, item) for v in value] if isinstance(value, list) else []
    if value is None or kind == "any":
        return value
    if isinstance(value, (dict, list)):
        return None
    if kind == "boolean":
        return value if isinstance(value, bool) else None
    if kind in ("integer", "number"):
        found = None if isinstance(value, bool) else re.match(r"\s*(-?\d+(?:\.\d+)?)", str(value))  # "47 years" -> 47
        if not found:
            return None
        number = float(found.group(1))
        return number if kind == "number" else (int(number) if number.is_integer() else None)
    text = str(value).strip()
    if not text or text.lower() == "null":  # small models sometimes write the word instead of a JSON null
        return None
    if kind == "datetime":
        try:
            return datetime.fromisoformat(text).isoformat()
        except ValueError:
            return None
    return text


# ---------------------------------------------------------------- checking the answer
def _squash(text: str) -> str:
    return re.sub(r"[\s|\\#*_`]+", "", text).lower()


def find_ungrounded(fields: Any, text: str, path: str = "") -> list[str]:
    """Paths of text values that do not appear in `text` (ignoring case, spaces and Markdown marks).

    A value the model made up, or copied from somewhere else, shows up here. Dates are skipped
    because they are reformatted, not copied."""
    if isinstance(fields, dict):
        return [p for k, v in fields.items() for p in find_ungrounded(v, text, f"{path}.{k}" if path else k)]
    if isinstance(fields, list):
        return [p for i, v in enumerate(fields) for p in find_ungrounded(v, text, f"{path}[{i}]")]
    if isinstance(fields, str) and not _ISO_DATETIME.match(fields) and _squash(fields) not in _squash(text):
        return [path]
    return []


# ---------------------------------------------------------------- public function
def get_fields(
    text: str,
    json_format: dict,
    instructions: str = "",
    llm: LLM | None = None,
    max_tokens: int = 2048,
    max_retries: int = 1,
) -> dict:
    """Extract the fields of `json_format` from `text` with an LLM.

    Returns a dict with exactly the keys and types of `json_format`; what the document does not give
    is None (or []). `instructions` adds guidance for ambiguous fields (e.g. which label in the
    document a key refers to). A UserWarning lists values that are not found in `text`.
    """
    if not isinstance(json_format, dict):
        raise TypeError("json_format must be a dict")
    if not text.strip():
        return _conform(None, json_format)  # nothing to read: do not ask the model to make something up

    llm = llm or LLM(SYSTEM_PROMPT)
    if llm.model.startswith("ollama") and len(text) / 3 > s.OLLAMA_NUM_CTX - 2500:  # rough: ~3 chars per token
        warnings.warn("The text may not fit in the model's context window (DOCINTEL_OLLAMA_NUM_CTX); "
                      "Ollama would cut it silently and fields could be missed.", stacklevel=2)

    prompt = (
        (f"{instructions.strip()}\n\n" if instructions.strip() else "")
        + f"<document>\n{text}\n</document>\n\n"
        "Fill this JSON structure. Each placeholder describes the type of the value; "
        "for a list, return one element per item found in the document.\n"
        + json.dumps(_hint(json_format), indent=2, ensure_ascii=False)
    )
    messages = [{"role": "user", "content": prompt}]
    response_format = {"type": "json_schema",
                       "json_schema": {"name": "fields", "schema": _schema(json_format), "strict": True}}

    for attempt in range(max(max_retries, 0) + 1):
        done = llm.complete(messages, temperature=0, max_tokens=max_tokens, response_format=response_format)
        try:
            data = _parse_json(done.text)
            break
        except ValueError as exc:  # includes json.JSONDecodeError
            if attempt >= max_retries:
                cut = " The output hit max_tokens: raise it." if done.finish_reason == "length" else ""
                raise ValueError(f"The model did not return valid JSON after {attempt + 1} attempt(s) ({exc}).{cut}") from exc
            messages += [{"role": "assistant", "content": done.text},
                         {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."}]

    fields = _conform(data, json_format)
    if issues := find_ungrounded(fields, text):
        warnings.warn(f"{len(issues)} value(s) were not found in the text and may be invented: {issues}", stacklevel=2)
    return fields
