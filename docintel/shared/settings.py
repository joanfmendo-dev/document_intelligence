"""Model settings shared by every LLM / VLM call.

Models are named the way litellm names them, so switching provider is a one-line change:
  "ollama_chat/gemma4:e2b"   local, through Ollama (default)
  "openai/gpt-4o-mini"      commercial API (also set the API key)
Override with environment variables, or by assigning attributes: `settings.s.LLM_MODEL = "..."`.
"""
from __future__ import annotations

import os


class Settings:
    # text model: turns extracted text into structured fields (get_fields)
    LLM_MODEL = os.getenv("DOCINTEL_LLM_MODEL", "ollama_chat/gemma4:e2b")
    LLM_API_KEY = os.getenv("DOCINTEL_LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv("DOCINTEL_LLM_BASE_URL", "")  # "" = provider default (Ollama: localhost:11434)

    # vision model: transcribes page images (extract_vlm)
    VLM_MODEL = os.getenv("DOCINTEL_VLM_MODEL", "ollama_chat/qwen3-vl:8b")
    VLM_API_KEY = os.getenv("DOCINTEL_VLM_API_KEY", "")
    VLM_BASE_URL = os.getenv("DOCINTEL_VLM_BASE_URL", "")

    # embedding model (the one clinic_assistant used): turns record text into vectors for the copilot's semantic search
    EMBEDDING_MODEL = os.getenv("DOCINTEL_EMBEDDING_MODEL", "ollama/nomic-embed-text")
    EMBEDDING_API_KEY = os.getenv("DOCINTEL_EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL = os.getenv("DOCINTEL_EMBEDDING_BASE_URL", "")
    # the record fields that make up the text to embed (in this order); decided with the recall test, so configurable
    EMBEDDING_FIELDS = ("chief_complaint", "history_of_present_illness", "final_diagnosis")

    # copilot (docintel.copilot): model that writes the plan, and how far semantic search reaches
    COPILOT_LLM_MODEL = os.getenv("DOCINTEL_COPILOT_LLM_MODEL", "")  # "" = LLM_MODEL
    CATALOG_MAX_VALUES = 100              # a column with more distinct values than this does not list them
    COPILOT_PAGE_SIZE = 25                # rows shown per answer
    SEMANTIC_TOP_K = 50                   # most candidates a semantic search returns
    SEMANTIC_RELATIVE_MARGIN = float(os.getenv("DOCINTEL_SEMANTIC_RELATIVE_MARGIN", "0.05"))  # keep hits within this of the best score
    SEMANTIC_MIN_SIMILARITY = float(os.getenv("DOCINTEL_SEMANTIC_MIN_SIMILARITY", "0.6"))  # to calibrate with the golden set

    # evaluation (docintel.evaluation): the judge is a commercial model, used offline and only on synthetic data
    JUDGE_MODEL = os.getenv("DOCINTEL_JUDGE_MODEL", "openai/gpt-4o-mini")
    JUDGE_API_KEY = os.getenv("DOCINTEL_JUDGE_API_KEY", "")
    JUDGE_BASE_URL = os.getenv("DOCINTEL_JUDGE_BASE_URL", "")

    # Applied only to Ollama models. Ollama silently cuts whatever does not fit in the context window.
    OLLAMA_NUM_CTX = int(os.getenv("DOCINTEL_OLLAMA_NUM_CTX", "8192"))
    OLLAMA_KEEP_ALIVE = os.getenv("DOCINTEL_OLLAMA_KEEP_ALIVE", "10m")

    # local DuckDB file where RecordDB keeps the records that get_fields returns
    DB_PATH = os.getenv("DOCINTEL_DB_PATH", "medical_records.duckdb")

    # web app (docintel.web): where an uploaded file waits while its fields are reviewed (deleted once the record is
    # saved or the upload cancelled), and the largest upload accepted
    UPLOADS_DIR = os.getenv("DOCINTEL_UPLOADS_DIR", "uploads")
    MAX_UPLOAD_MB = int(os.getenv("DOCINTEL_MAX_UPLOAD_MB", "25"))

    # sign-in: "name:hash,name2:hash2" (hashes from `python -m docintel.web.passwd`); the app does not start without users.
    # SECRET_KEY signs the session cookie; without it a random key is made at each start, so a restart signs everybody out.
    USERS = dict(pair.strip().split(":", 1) for pair in os.getenv("DOCINTEL_USERS", "").split(",") if ":" in pair)
    SECRET_KEY = os.getenv("DOCINTEL_SECRET_KEY", "")

    def role(self, name: str) -> dict[str, str]:
        """{"model", "api_key", "base_url"} of a role: "text", "vlm", "embedding", "judge", or "copilot" (the text model unless
        COPILOT_LLM_MODEL names another one; the same key and URL as the text model)."""
        prefix = {"text": "LLM", "copilot": "LLM", "vlm": "VLM", "embedding": "EMBEDDING", "judge": "JUDGE"}[name]
        model = getattr(self, f"{prefix}_MODEL")
        if name == "copilot":
            model = self.COPILOT_LLM_MODEL or model
        return {"model": model, "api_key": getattr(self, f"{prefix}_API_KEY"), "base_url": getattr(self, f"{prefix}_BASE_URL")}


s = Settings()
