"""Model settings shared by every LLM / VLM call.

Models are named the way litellm names them, so switching provider is a one-line change:
  "ollama_chat/gemma3:4b"   local, through Ollama (default)
  "openai/gpt-4o-mini"      commercial API (also set the API key)
Override with environment variables, or by assigning attributes: `settings.s.LLM_MODEL = "..."`.
"""
from __future__ import annotations

import os


class Settings:
    # text model: turns extracted text into structured fields (get_fields)
    LLM_MODEL = os.getenv("DOCINTEL_LLM_MODEL", "ollama_chat/gemma3:4b")
    LLM_API_KEY = os.getenv("DOCINTEL_LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("DOCINTEL_LLM_BASE_URL", "")  # "" = provider default (Ollama: localhost:11434)

    # vision model: transcribes page images (extract_vlm)
    VLM_MODEL = os.getenv("DOCINTEL_VLM_MODEL", "ollama_chat/qwen3-vl:8b")
    VLM_API_KEY = os.getenv("DOCINTEL_VLM_API_KEY", "")
    VLM_BASE_URL = os.getenv("DOCINTEL_VLM_BASE_URL", "")

    # Applied only to Ollama models. Ollama silently cuts whatever does not fit in the context window.
    OLLAMA_NUM_CTX = int(os.getenv("DOCINTEL_OLLAMA_NUM_CTX", "8192"))
    OLLAMA_KEEP_ALIVE = os.getenv("DOCINTEL_OLLAMA_KEEP_ALIVE", "10m")

    # local DuckDB file where RecordDB keeps the records that get_fields returns
    DB_PATH = os.getenv("DOCINTEL_DB_PATH", "medical_records.duckdb")

    # web app (docintel.web): where an uploaded file waits while its fields are reviewed (deleted once the record is
    # saved or the upload cancelled), and the largest upload accepted
    UPLOADS_DIR = os.getenv("DOCINTEL_UPLOADS_DIR", "uploads")
    MAX_UPLOAD_MB = int(os.getenv("DOCINTEL_MAX_UPLOAD_MB", "25"))

    #JSON Templates
    DB_JSON_TEMPLATE = {
        "record_id": "SYNUS-2026-0001",
        "department": "Cardiology",
        "patient_name": "Jennifer Parker",
        "dob": "1961-01-28T00:00:00",
        "age": 63,
        "sex": "Female",
        "insurance": "Aetna PPO",
        "city": "Miami, Florida",
        "encounter_date": "2025-01-01T09:00:00",
        "chief_complaint": "Chest pressure and exertional dyspnea.",
        "history_of_present_illness": "The patient reports 5 days of substernal pressure that worsens with exertion and improves with rest. Mild diaphoresis and nausea are present, but there is no syncope.",
        "past_medical_history": "Hypertension, hyperlipidemia, former smoker.",
        "medications": "Atorvastatin, lisinopril, aspirin.",
        "allergies": "NKDA.",
        "vital_signs": {
            "blood_pressure": "158/95 mmHg",
            "heart_rate": "101 bpm",
            "respiration_rate": "23/min",
            "temperature": "99.4 °F",
            "sp02": "98%"
        },
        "physical_exam": "Mild distress, regular tachycardia, and no focal neurologic deficits.",
        "lab_results": [
            {"test_name": "Troponin I (initial)", "result": "0.042 ng/mL", "flag": "Borderline"},
            {"test_name": "WBC Count", "result": "5.6 K/uL", "flag": "Normal"},
            {"test_name": "Hemoglobin", "result": "14.3 g/dL", "flag": "Normal"},
            {"test_name": "Platelets", "result": "181 K/uL", "flag": "Normal"},
            {"test_name": "Sodium", "result": "138 mmol/L", "flag": "Normal"},
            {"test_name": "Potassium", "result": "4.3 mmol/L", "flag": "Normal"},
            {"test_name": "BUN", "result": "12 mg/dL", "flag": "Normal"},
            {"test_name": "Creatinine", "result": "0.87 mg/dL", "flag": "Normal"},
            {"test_name": "Glucose", "result": "97 mg/dL", "flag": "Normal"}
        ],
        "imaging_results": "ECG with nonspecific ST-T changes; chest X-ray without acute infiltrate.",
        "final_diagnosis": "Possible acute coronary syndrome / NSTEMI versus demand ischemia.",
        "plan": "Telemetry admission, serial troponins, cardiology consultation, aspirin, and risk stratification.",
        "progress_notes": "Ongoing chest pain surveillance with biomarker trending."
    }
    DB_JSON_TEMPLATE = os.getenv("DB_JSON_TEMPLATE", DB_JSON_TEMPLATE)

s = Settings()
