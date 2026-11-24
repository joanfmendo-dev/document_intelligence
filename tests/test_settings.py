"""config.json feeds the settings: models and users. Environment variable > config.json > default."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from docintel.shared import settings as module

ROOT = Path(__file__).resolve().parent.parent
ROLES = ("extraction", "copilot", "vlm", "embedding", "judge")
CONFIG = {
    "users": [{"username": "maria", "hash": "aa$bb"}],
    "models": {
        "extraction_llm_model": {"model": "openai/gpt-4o-mini", "api_key": "sk-extraction", "base_url": ""},
        "embedding_model": {"model": "azure/emb", "api_key": "k-emb", "base_url": "https://x.openai.azure.com"},
        "copilot_model": {"model": "anthropic/claude-x", "api_key": "k-cop", "base_url": ""},
    },
}


def settings_of_a_new_process(config_file, env=None):
    """The Settings of a process that starts with this config file (they are read once, when the module loads)."""
    code = ("import json; from docintel.shared.settings import s; "
            f"print(json.dumps({{'roles': {{r: s.role(r) for r in {ROLES}}}, 'users': s.USERS}}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                         env={"PATH": "", "CONFIG_FILE": str(config_file), **(env or {})})
    return json.loads(out.stdout)


@pytest.fixture
def config_file(tmp_path):
    file = tmp_path / "config.json"
    file.write_text(json.dumps(CONFIG))
    return file


def test_each_task_has_its_own_model_key_and_url(config_file):
    got = settings_of_a_new_process(config_file)
    assert got["roles"]["extraction"] == {"model": "openai/gpt-4o-mini", "api_key": "sk-extraction", "base_url": ""}
    assert got["roles"]["copilot"] == {"model": "anthropic/claude-x", "api_key": "k-cop", "base_url": ""}  # not the extraction model's
    assert got["roles"]["embedding"] == {"model": "azure/emb", "api_key": "k-emb", "base_url": "https://x.openai.azure.com"}
    assert got["roles"]["vlm"] == {"model": "ollama_chat/qwen3-vl:8b", "api_key": "", "base_url": ""}  # not in the file: the default
    assert got["roles"]["judge"]["model"] == "openai/gpt-4o-mini"
    assert got["users"] == {"maria": "aa$bb"}


def test_an_environment_variable_named_like_the_setting_beats_the_file(config_file):
    got = settings_of_a_new_process(config_file, {"EXTRACTION_LLM_MODEL": "ollama_chat/other", "EXTRACTION_LLM_API_KEY": "from-env",
                                                  "USERS": "luis:cc$dd,maria:new$hash"})
    assert got["roles"]["extraction"] == {"model": "ollama_chat/other", "api_key": "from-env", "base_url": ""}
    assert got["roles"]["copilot"]["model"] == "anthropic/claude-x"  # only what the variable names changes
    assert got["users"] == {"maria": "new$hash", "luis": "cc$dd"}


def test_without_a_config_file_every_task_has_its_default(tmp_path):
    got = settings_of_a_new_process(tmp_path / "missing.json")
    assert {r: got["roles"][r]["model"] for r in ROLES} == {
        "extraction": "ollama_chat/gemma4:e2b", "copilot": "ollama_chat/gemma4:e2b", "vlm": "ollama_chat/qwen3-vl:8b",
        "embedding": "ollama/nomic-embed-text", "judge": "openai/gpt-4o-mini"}
    assert got["users"] == {}


def test_the_default_models_are_the_ones_the_config_names():
    assert set(module.DEFAULT_MODELS) == {"extraction_llm_model", "vlm_model", "embedding_model", "copilot_model", "judge_model"}
    assert module.s.role("copilot")["model"] == module.s.COPILOT_MODEL
    with pytest.raises(KeyError):
        module.s.role("nobody")


@pytest.mark.parametrize("text, message", [
    ('{"users": [],}', "not valid JSON"),  # a trailing comma
    ('{"models": {"llm_model": {"model": "x"}}}', "unknown model"),  # the old name of a model
    ('{"users": [{"username": "maria", "password": "aa$bb"}]}', 'needs a "username" and a "hash"'),  # the old name of the hash
])
def test_a_config_in_the_wrong_shape_says_what_is_wrong(tmp_path, text, message):
    bad = tmp_path / "config.json"
    bad.write_text(text)
    with pytest.raises(ValueError, match=message):
        module.load_config(str(bad))
    assert module.load_config(str(tmp_path / "missing.json")) == {}
