import json
import warnings

import pytest

from docintel import LLM, get_fields
from docintel.fields import _conform, _hint, _parse_json, _schema, find_ungrounded

# The template exactly as specified: only its shape and types matter, its values are never sent to a model.
TEMPLATE = {
    "record_id": "SYNUS-2026-0001", "facility": "Cardiology", "patient_name": "Jennifer Parker",
    "dob": "1961-01-28T00:00:00", "age": 63, "sex": "Female", "insurance": "Aetna PPO",
    "city": "Miami, Florida", "encounter_date": "2025-01-01T09:00:00",
    "chief_complaint": "Chest pressure and exertional dyspnea.",
    "history_of_present_illness": "The patient reports 5 days of substernal pressure.",
    "past_medical_history": "Hypertension, hyperlipidemia, former smoker.",
    "medications": "Atorvastatin, lisinopril, aspirin.", "allergies": "NKDA.",
    "vital_signs": {"blood_pressure": "158/95 mmHg", "heart_rate": "101 bpm", "respiration_rate": "23/min",
                    "temperature": "99.4 °F", "sp02": "98%"},
    "physical_exam": "Mild distress, regular tachycardia.",
    "lab_results": [{"test_name": "Troponin I (initial)", "result": "0.042 ng/mL", "flag": "Borderline"}],
    "imaging_results": "ECG with nonspecific ST-T changes.", "final_diagnosis": "Possible NSTEMI.",
    "plan": "Telemetry admission.", "progress_notes": "Ongoing surveillance.",
}

DOCUMENT = """# Marvin Monroe Memorial Hospital

## Record No. ENC-2026-88314

| PATIENT NAME | Michael Carter | RECORD NO. | ENC-2026-88314 |
| DATE OF BIRTH | March 11, 1978 | AGE / SEX | 47 years / Male |
| CITY | Austin, Texas | INSURANCE | Aetna PPO |
| ENCOUNTER | May 18, 2025 00:51 | DEPARTMENT | Cardiology |

## CHIEF COMPLAINT

Chest tightness and exertional shortness of breath.

## LABORATORY RESULTS

| Test | Result | Flag |
| --- | --- | --- |
| Troponin I | 0.107 ng/mL | High |
"""

GOOD = {
    "record_id": "ENC-2026-88314", "facility": "Cardiology", "patient_name": "Michael Carter",
    "dob": "1978-03-11T00:00:00", "age": 47, "sex": "Male", "insurance": "Aetna PPO", "city": "Austin, Texas",
    "encounter_date": "2025-05-18T00:51:00", "chief_complaint": "Chest tightness and exertional shortness of breath.",
    "vital_signs": {"blood_pressure": None}, "lab_results": [
        {"test_name": "Troponin I", "result": "0.107 ng/mL", "flag": "High"}],
}


def keys_of(value):
    """Structure of a dict: nested keys, and for lists the keys of the first item."""
    if isinstance(value, dict):
        return {k: keys_of(v) for k, v in value.items()}
    if isinstance(value, list):
        return [keys_of(value[0])] if value else []
    return None


def test_result_has_exactly_the_shape_of_the_template(ollama):
    ollama.reply(json.dumps({**GOOD, "unexpected_key": "dropped"}))
    r = get_fields(DOCUMENT, TEMPLATE)
    assert list(r) == list(TEMPLATE)                                   # same keys, same order, extras dropped
    assert keys_of(r["vital_signs"]) == keys_of(TEMPLATE["vital_signs"])
    assert r["age"] == 47 and r["dob"] == "1978-03-11T00:00:00" and r["encounter_date"] == "2025-05-18T00:51:00"
    assert r["medications"] is None and r["allergies"] is None        # not in the model's answer -> None
    assert r["vital_signs"] == {"blood_pressure": None, "heart_rate": None, "respiration_rate": None,
                                "temperature": None, "sp02": None}
    assert r["lab_results"] == [{"test_name": "Troponin I", "result": "0.107 ng/mL", "flag": "High"}]


def test_template_values_are_never_sent_to_the_model(ollama):
    ollama.reply(json.dumps(GOOD))
    get_fields(DOCUMENT, TEMPLATE)
    request = json.dumps(ollama.requests[0], ensure_ascii=False)
    for example_value in ("Jennifer Parker", "Aetna PPO is", "Atorvastatin", "158/95", "Borderline", "SYNUS-2026-0001"):
        assert example_value not in request
    user_message = ollama.requests[0]["messages"][-1]["content"]
    assert "Michael Carter" in user_message and "<integer, or null>" in user_message
    assert "<ISO 8601 datetime YYYY-MM-DDTHH:MM:SS, or null>" in user_message


def test_request_asks_for_deterministic_structured_output(ollama):
    ollama.reply(json.dumps(GOOD))
    get_fields(DOCUMENT, TEMPLATE, instructions="facility is the DEPARTMENT field.")
    body = ollama.requests[0]
    assert body["options"]["temperature"] == 0 and body["model"] == "gemma3:4b"
    schema = body["format"]                                             # Ollama structured output
    assert set(schema["required"]) == set(TEMPLATE) and schema["additionalProperties"] is False
    assert schema["properties"]["age"] == {"type": ["integer", "null"]}
    assert schema["properties"]["lab_results"]["items"]["required"] == ["test_name", "result", "flag"]
    assert body["messages"][-1]["content"].startswith("facility is the DEPARTMENT field.")
    assert "single JSON object" in body["messages"][0]["content"]


def test_messy_output_is_repaired_into_the_right_types(ollama):
    ollama.reply('```json\n' + json.dumps({"age": "47 years", "dob": "1978-03-11", "sex": "  Male  ",
                                          "chief_complaint": "", "lab_results": "none",
                                          "vital_signs": ["x"], "encounter_date": "not a date"}) + '\n```')
    r = get_fields("Male 47 years", TEMPLATE)
    assert r["age"] == 47 and r["dob"] == "1978-03-11T00:00:00" and r["sex"] == "Male"
    assert r["chief_complaint"] is None and r["encounter_date"] is None
    assert r["lab_results"] == [] and r["vital_signs"]["heart_rate"] is None


def test_retries_once_when_the_reply_is_not_json(ollama):
    ollama.reply("Sure! Here you go: not json")
    ollama.reply(json.dumps(GOOD))
    r = get_fields(DOCUMENT, TEMPLATE)
    assert len(ollama.requests) == 2 and r["patient_name"] == "Michael Carter"
    retry = ollama.requests[1]["messages"]
    assert retry[-2] == {"role": "assistant", "content": "Sure! Here you go: not json", "images": []}
    assert "ONLY the JSON" in retry[-1]["content"]


def test_gives_up_with_a_clear_error_and_mentions_truncation(ollama):
    ollama.reply('{"record_id": "ENC', done_reason="length")
    ollama.reply('{"record_id": "ENC', done_reason="length")
    with pytest.raises(ValueError, match=r"valid JSON after 2 attempt.*max_tokens"):
        get_fields(DOCUMENT, TEMPLATE)


def test_warns_about_values_not_found_in_the_text(ollama):
    ollama.reply(json.dumps({**GOOD, "medications": "Atorvastatin, lisinopril, aspirin.", "city": "Austin, Texas"}))
    with pytest.warns(UserWarning, match=r"medications"):
        get_fields(DOCUMENT, TEMPLATE)
    ollama.reply(json.dumps(GOOD))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # a faithful answer must not warn
        get_fields(DOCUMENT, TEMPLATE)


def test_empty_text_does_not_call_the_model(ollama):
    r = get_fields("   ", TEMPLATE)
    assert ollama.requests == [] and r["age"] is None and r["lab_results"] == []
    assert list(r) == list(TEMPLATE)


def test_model_comes_from_settings_or_from_the_llm_you_pass(ollama, monkeypatch):
    from docintel.settings import s

    ollama.reply(json.dumps(GOOD))
    monkeypatch.setattr(s, "LLM_MODEL", "ollama_chat/llama3.2:3b")
    get_fields(DOCUMENT, TEMPLATE)
    assert ollama.requests[-1]["model"] == "llama3.2:3b"
    ollama.reply(json.dumps(GOOD))
    get_fields(DOCUMENT, TEMPLATE, llm=LLM("custom system", model="ollama_chat/mistral:7b", base_url=ollama.url))
    assert ollama.requests[-1]["model"] == "mistral:7b" and ollama.requests[-1]["messages"][0]["content"] == "custom system"


def test_warns_when_text_will_not_fit_the_ollama_context(ollama):
    ollama.reply(json.dumps(GOOD))
    with pytest.warns(UserWarning, match="context window"):
        get_fields("x " * 20000, TEMPLATE)


def test_json_format_must_be_a_dict():
    with pytest.raises(TypeError):
        get_fields("text", ["not", "a", "dict"])


# ---- small pure functions
def test_parse_json_variants():
    assert _parse_json('<think>hmm</think>{"a": 1}') == {"a": 1}
    assert _parse_json('text before {"a": {"b": 2}} text after') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        _parse_json("[1, 2]")


def test_hint_and_schema_follow_the_template_types():
    assert _hint({"n": 1, "f": 1.5, "b": True, "d": "2025-01-01", "t": "x", "l": [{"k": "v"}]}) == {
        "n": "<integer, or null>", "f": "<number, or null>", "b": "<true or false, or null>",
        "d": "<ISO 8601 datetime YYYY-MM-DDTHH:MM:SS, or null>", "t": "<text copied from the document, or null>",
        "l": [{"k": "<text copied from the document, or null>"}]}
    assert _schema({"b": True})["properties"]["b"] == {"type": ["boolean", "null"]}      # bool is not treated as int
    assert _conform("5", 0) == 5 and _conform(5.5, 0) is None and _conform(True, 0) is None


def test_conform_does_not_invent_numbers_or_nulls():
    assert _conform("47 years", 0) == 47 and _conform("98.6 °F", 0.0) == 98.6
    assert _conform("approximately 47", 0) is None            # the number must come first
    assert _conform({"a": 1}, 0) is None and _conform([2], 0.0) is None      # no digits fished out of containers
    assert _conform("null", "text") is None and _conform(" NULL ", "text") is None   # the word, not a JSON null
    assert _conform("None reported.", "text") == "None reported."                    # real text is kept


def test_find_ungrounded_ignores_markdown_case_and_dates():
    text = "| Troponin I | 0.107 ng/mL | High |\nAustin, Texas"
    assert find_ungrounded({"lab": "troponin i0.107 ng/mL", "city": "AUSTIN, TEXAS", "dob": "1978-03-11T00:00:00"},
                           text) == []
    assert find_ungrounded({"a": {"b": ["invented"]}}, text) == ["a.b[0]"]


def test_a_negative_max_retries_still_makes_one_attempt(ollama):
    ollama.reply(json.dumps(GOOD))
    assert get_fields(DOCUMENT, TEMPLATE, max_retries=-1)["patient_name"] == "Michael Carter"
    ollama.reply("not json")
    with pytest.raises(ValueError, match="after 1 attempt"):
        get_fields(DOCUMENT, TEMPLATE, max_retries=-1)
