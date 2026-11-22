"""Test doubles and the small seed database shared by the test files (a fake model, a fake embedder, 12 records)."""
import json
from datetime import date

from docintel.shared import EmbeddingError

TODAY = date(2026, 11, 13)
KEYWORDS = ["breath", "chest", "fever", "other"]


class KeywordEmbedder:
    """Vectors by keyword, so similarity is predictable: a text about 'breath' is close to a query about breathing."""
    model, batch_size = "kw", 10

    def __init__(self, down=False):
        self.down = down

    def vec(self, text):
        t = text.lower()
        v = [1.0 if k in t else 0.0 for k in KEYWORDS]
        return v if any(v[:3]) else [0.0, 0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        return [self.vec(t) for t in texts]

    def embed_query(self, text):
        if self.down:
            raise EmbeddingError("down")
        return self.vec(text)


class FakeLLM:
    """Answers with the next canned reply (a dict is sent as JSON, an Exception is raised, `default` when none is left)
    and records every call as (messages, params)."""
    model = "fake"

    def __init__(self, *replies, default="{}"):
        self.replies, self.calls, self.default = list(replies), [], default

    def run(self, messages, **params):
        self.calls.append((messages, params))
        reply = self.replies.pop(0) if self.replies else self.default
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def rec(i, dept, sex, age, day, complaint, dx, vitals=None, labs=None, **more):
    return {"record_id": f"R{i:03d}", "department": dept, "patient_name": f"Patient {i}", "sex": sex, "age": age,
            "insurance": "Cigna", "city": "Miami, Florida", "encounter_date": f"{day}T09:00:00", "chief_complaint": complaint,
            "final_diagnosis": dx, "vital_signs": vitals, "lab_results": labs, **more}


def lab(name, flag):
    return {"test_name": name, "result": "1 x", "flag": flag}


def vit(t=None, hr=None, bp=None, sp=None):
    return {"blood_pressure": bp, "heart_rate": hr, "respiration_rate": "18/min", "temperature": t, "sp02": sp}


SEED = [
    rec(1, "Cardiology", "Female", 70, "2026-03-10", "chest pain", "NSTEMI", vit("101.0 °F", "110 bpm", "150/95 mmHg", "97%"),
        [lab("Troponin I", "High"), lab("Glucose", "Normal")], icd10_code="I21", icd10_chapter="Diseases of the circulatory system"),
    rec(2, "Cardiology", "Male", 55, "2026-05-02", "chest pressure", "Stable angina", None, [lab("Troponin I (initial)", "Borderline")]),
    rec(3, "Cardiology", "Female", 62, "2025-11-20", "swollen legs", "Heart failure", None, [lab("BNP", "High"), lab("Troponin I", "High")],
        icd10_code="I50", icd10_chapter="Diseases of the circulatory system"),
    rec(4, "Neurology", "Male", 45, "2025-03-15", "headache", "Migraine", vit("98.6 °F")),
    rec(5, "Neurology", "Female", 80, "2026-01-20", "weakness", "Ischemic stroke", None,
        [lab("Glucose", "High"), lab("Glucose (point-of-care)", "Normal")], icd10_code="I63", icd10_chapter="Diseases of the circulatory system"),
    rec(6, "Pulmonology", "Male", 30, "2025-07-04", "shortness of breath at night", "Asthma", vit(hr="90 bpm", sp="91%")),
    rec(7, "Pulmonology", "Female", 66, "2026-10-30", "cough and fever", "Pneumonia", vit("103.1 °F", sp="89%"),
        icd10_code="J18", icd10_chapter="Diseases of the respiratory system"),
    rec(8, "Urology", "Male", 52, "2026-11-10", "flank pain", "Kidney stone"),
    rec(9, "Urology", "Female", 38, "2026-11-01", "burning urination", "Urinary tract infection", vit("100.4 °F")),
    rec(10, "Cardiology", "Male", 75, "2026-11-12", "palpitations", "Atrial fibrillation", vit(hr="130 bpm")),
    {"record_id": "R011"},
    rec(12, "Cardiology", "Female", 58, "2025-01-05", "palpitations", "SVT"),
]


def f(field, op, value):
    return {"field": field, "op": op, "value": value}
