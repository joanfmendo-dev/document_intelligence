import pytest

from docintel import LLM, TokenCount
from docintel.llm import LLMError, image_content
from docintel.settings import s


def test_complete_builds_the_ollama_request_and_counts_tokens(ollama):
    ollama.reply("hola")
    TokenCount.reset()
    done = LLM("SYSTEM TEXT").complete([{"role": "user", "content": "hi"}], temperature=0, max_tokens=500)
    body = ollama.requests[0]
    assert body["model"] == "gemma3:4b"                                       # "ollama_chat/" prefix is routing only
    assert [m["role"] for m in body["messages"]] == ["system", "user"] and body["messages"][0]["content"] == "SYSTEM TEXT"
    assert body["options"]["temperature"] == 0 and body["options"]["num_predict"] == 500
    assert body["options"]["num_ctx"] == s.OLLAMA_NUM_CTX and body["keep_alive"] == s.OLLAMA_KEEP_ALIVE
    assert (done.text, done.finish_reason, done.input_tokens, done.output_tokens) == ("hola", "stop", 123, 45)
    assert TokenCount.totals() == {"input": 123, "output": 45}
    LLM().run([{"role": "user", "content": "again"}])
    assert TokenCount.totals() == {"input": 246, "output": 90}
    TokenCount.reset()
    assert TokenCount.totals() == {"input": 0, "output": 0}


def test_run_returns_only_text_and_finish_reason_is_exposed(ollama):
    ollama.reply("cut off", done_reason="length")
    done = LLM().complete([{"role": "user", "content": "x"}])
    assert done.text == "cut off" and done.finish_reason == "length"


def test_ollama_options_are_not_sent_to_other_providers(monkeypatch):
    import litellm

    seen = {}

    class Usage:
        prompt_tokens, completion_tokens = 1, 1

    class Resp:
        usage = Usage()
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})(), "finish_reason": "stop"})()]

    monkeypatch.setattr(litellm, "completion", lambda **kw: seen.update(kw) or Resp())
    assert LLM(model="openai/gpt-4o-mini", api_key="sk-test").run([{"role": "user", "content": "x"}]) == "ok"
    assert "num_ctx" not in seen and "keep_alive" not in seen
    assert seen["api_key"] == "sk-test" and seen["model"] == "openai/gpt-4o-mini" and seen["drop_params"] is True


def test_errors_become_llm_error_with_a_hint():
    with pytest.raises(LLMError, match="Is the server running"):
        LLM(base_url="http://127.0.0.1:1", timeout=5).run([{"role": "user", "content": "x"}])


def test_image_content_is_openai_style_data_url():
    from PIL import Image

    parts = image_content("look", [Image.new("RGB", (10, 10))])
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
