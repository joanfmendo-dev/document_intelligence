"""Settings shared by every model call and by the web app.

Models are named the way litellm names them, so switching provider is a one-line change:
  "ollama_chat/gemma4:e2b"   local, through Ollama (default)
  "openai/gpt-4o-mini"      commercial API (also set the api_key)
Where a value comes from, first match wins: an environment variable named like the setting (EXTRACTION_LLM_MODEL, DB_PATH...),
then config.json, then the default here. config.json (path in CONFIG_FILE) holds what is sensitive or machine-specific: the users
who may sign in and one model per task, each with its own key and URL:

    {"users": [{"username": "maria", "hash": "<hash>"}],
     "models": {"extraction_llm_model": {"model": "...", "api_key": "", "base_url": ""},
                "vlm_model": {...}, "embedding_model": {...}, "copilot_model": {...}, "judge_model": {...}}}

Or assign attributes: `settings.s.EXTRACTION_LLM_MODEL = "..."`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")
DEFAULT_MODELS = {
    "extraction_llm_model": "ollama_chat/gemma4:e2b",  # turns the text of a document into structured fields
    "vlm_model": "ollama_chat/qwen3-vl:8b",            # transcribes page images
    "embedding_model": "ollama/nomic-embed-text",      # vectors for the copilot's semantic search
    "copilot_model": "ollama_chat/gemma4:e2b",         # writes the copilot's plan
    "judge_model": "openai/gpt-4o-mini",               # grades the evaluation; commercial, offline, synthetic data only
}


def load_config(path: str) -> dict:
    """The JSON file, or {} when there is none."""
    file = Path(path)
    if not file.is_file():
        return {}
    try:
        config = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file} is not valid JSON ({exc}). JSON allows no comments and no trailing commas.") from exc
    if unknown := set(config.get("models", {})) - set(DEFAULT_MODELS):  # e.g. a file in an older format: better an error than a silent default
        raise ValueError(f"{file}: unknown model {sorted(unknown)}; the models are {sorted(DEFAULT_MODELS)}.")
    if any(not (u.get("username") and u.get("hash")) for u in config.get("users", [])):
        raise ValueError(f'{file}: every user needs a "username" and a "hash".')
    return config


CONFIG = load_config(CONFIG_FILE)


def model_setting(key: str, field: str, env: str, default: str = "") -> str:
    """models -> `key` -> `field` of config.json, unless the environment variable `env` (named like the setting) is set; else `default`."""
    return os.getenv(env) or CONFIG.get("models", {}).get(key, {}).get(field) or default


def users_from(config: dict, env: str = "") -> dict[str, str]:
    """{name: hash} from the config's "users" and from `env` ("name:hash,name2:hash2"); the variable wins."""
    users = {u["username"]: u["hash"] for u in config.get("users", [])}
    users.update(pair.strip().split(":", 1) for pair in env.split(",") if ":" in pair)
    return users


class Settings:
    # text model that turns extracted text into structured fields (get_fields)
    EXTRACTION_LLM_MODEL = model_setting("extraction_llm_model", "model", "EXTRACTION_LLM_MODEL", DEFAULT_MODELS["extraction_llm_model"])
    EXTRACTION_LLM_API_KEY = model_setting("extraction_llm_model", "api_key", "EXTRACTION_LLM_API_KEY")
    EXTRACTION_LLM_BASE_URL = model_setting("extraction_llm_model", "base_url", "EXTRACTION_LLM_BASE_URL")  # "" = the provider's own address (Ollama: localhost:11434)

    # vision model: transcribes page images (extract_vlm)
    VLM_MODEL = model_setting("vlm_model", "model", "VLM_MODEL", DEFAULT_MODELS["vlm_model"])
    VLM_API_KEY = model_setting("vlm_model", "api_key", "VLM_API_KEY")
    VLM_BASE_URL = model_setting("vlm_model", "base_url", "VLM_BASE_URL")  # "" = the provider's own address (Ollama: localhost:11434)

    # embedding model: turns record text into vectors for the copilot's semantic search
    EMBEDDING_MODEL = model_setting("embedding_model", "model", "EMBEDDING_MODEL", DEFAULT_MODELS["embedding_model"])
    EMBEDDING_API_KEY = model_setting("embedding_model", "api_key", "EMBEDDING_API_KEY")
    EMBEDDING_BASE_URL = model_setting("embedding_model", "base_url", "EMBEDDING_BASE_URL")  # "" = the provider's own address (Ollama: localhost:11434)
    # the record fields that make up the text to embed (in this order); decided with the recall test, so configurable
    EMBEDDING_FIELDS = ("chief_complaint", "history_of_present_illness", "final_diagnosis")

    # copilot (docintel.copilot): the model that writes the plan
    COPILOT_MODEL = model_setting("copilot_model", "model", "COPILOT_MODEL", DEFAULT_MODELS["copilot_model"])
    COPILOT_API_KEY = model_setting("copilot_model", "api_key", "COPILOT_API_KEY")
    COPILOT_BASE_URL = model_setting("copilot_model", "base_url", "COPILOT_BASE_URL")  # "" = the provider's own address (Ollama: localhost:11434)
    CATALOG_MAX_VALUES = 100              # a column with more distinct values than this does not list them
    COPILOT_PAGE_SIZE = 25                # rows shown per answer
    SEMANTIC_TOP_K = 50                   # most candidates a semantic search returns
    SEMANTIC_RELATIVE_MARGIN = float(os.getenv("SEMANTIC_RELATIVE_MARGIN", "0.05"))  # keep hits within this of the best score
    SEMANTIC_MIN_SIMILARITY = float(os.getenv("SEMANTIC_MIN_SIMILARITY", "0.4"))  # calibrated for text-embedding-3-small (best hits 0.4-0.55); nomic-embed-text needs ~0.6

    # evaluation (docintel.evaluation): the judge is a commercial model, used offline and only on synthetic data
    JUDGE_MODEL = model_setting("judge_model", "model", "JUDGE_MODEL", DEFAULT_MODELS["judge_model"])
    JUDGE_API_KEY = model_setting("judge_model", "api_key", "JUDGE_API_KEY")
    JUDGE_BASE_URL = model_setting("judge_model", "base_url", "JUDGE_BASE_URL")  # "" = the provider's own address (Ollama: localhost:11434)

    # Applied only to Ollama models. Ollama silently cuts whatever does not fit in the context window.
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

    # local DuckDB file where RecordDB keeps the records that get_fields returns
    DB_PATH = os.getenv("DB_PATH", "medical_records.duckdb")

    # web app (docintel.web): where an uploaded file waits while its fields are reviewed (deleted once the record is
    # saved or the upload cancelled), and the largest upload accepted
    UPLOADS_DIR = os.getenv("UPLOADS_DIR", "uploads")
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
    MAX_PAGES = int(os.getenv("MAX_PAGES", "0"))  # PDFs with more pages are refused before the OCR; 0 = no limit

    # on a server: ALLOWED_HOSTS is the comma-separated list of names the app answers to ("*" = any); setting it is what lets the app
    # listen beyond this computer. BLOB_CONNECTION_STRING keeps the master copy of the database in Azure Blob Storage (see web/blob.py).
    ALLOWED_HOSTS = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()]
    BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING", "")
    BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "database")

    # sign-in: the "users" of config.json (`python -m docintel.web.passwd` writes them) plus USERS ("name:hash,name2:hash2", for
    # servers where a secret is easier than a file); the app does not start without users.
    # SECRET_KEY signs the session cookie; without it a random key is made at each start, so a restart signs everybody out.
    USERS = users_from(CONFIG, os.getenv("USERS", ""))
    SECRET_KEY = os.getenv("SECRET_KEY", "")

    def role(self, name: str) -> dict[str, str]:
        """{"model", "api_key", "base_url"} of a task: "extraction", "copilot", "vlm", "embedding" or "judge"."""
        prefix = {"extraction": "EXTRACTION_LLM", "copilot": "COPILOT", "vlm": "VLM", "embedding": "EMBEDDING", "judge": "JUDGE"}[name]
        return {field: getattr(self, f"{prefix}_{field.upper()}") for field in ("model", "api_key", "base_url")}


s = Settings()
