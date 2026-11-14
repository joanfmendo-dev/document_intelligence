"""Shared LLM access for the whole project (text and vision), built on litellm.

Any model litellm supports works unchanged: local (Ollama, llama.cpp servers, vLLM) or a
commercial API. Nothing else in the project talks to a model directly.
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

from PIL import Image

from .settings import s

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # do not download the price list at import

# ── Token accounting (session-wide) ───────────────────────────────────────────
_totals: dict[str, int] = {"input": 0, "output": 0}


class TokenCount:
    @staticmethod
    def reset() -> None:
        """Reset the session-wide token counters to zero."""
        _totals["input"] = _totals["output"] = 0

    @staticmethod
    def accum(usage) -> tuple[int, int]:
        """Add usage from a litellm response to the running totals. Returns (input, output)."""
        inp = getattr(usage, "prompt_tokens", 0) or 0
        out = getattr(usage, "completion_tokens", 0) or 0
        _totals["input"] += inp
        _totals["output"] += out
        return inp, out

    @staticmethod
    def totals() -> dict[str, int]:
        return dict(_totals)

    @staticmethod
    def print(msg_in: int, msg_out: int) -> None:
        print(f"\033[94m[Tokens] This message - input: {msg_in} output: {msg_out}"
              f" | Total - input: {_totals['input']} output: {_totals['output']}\033[0m")


# ── Shared LLM helper ─────────────────────────────────────────────────────────
class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    finish_reason: str | None  # "stop", or "length" when the model ran out of output tokens
    input_tokens: int
    output_tokens: int


def image_content(text: str, images: list[Image.Image]) -> list[dict]:
    """A user message body with text plus images, in the format every litellm provider accepts."""
    parts: list[dict] = [{"type": "text", "text": text}]
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


class LLM:
    def __init__(self, system: str = "", model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, timeout: float = 600.0):
        self.system = system
        self.model = model or s.LLM_MODEL
        self.api_key = s.LLM_API_KEY if api_key is None else api_key
        self.base_url = s.LLM_BASE_URL if base_url is None else base_url
        self.timeout = timeout

    def complete(self, messages: list[dict], **params) -> Completion:
        """Call the model. `params` go straight to litellm (temperature, max_tokens,
        response_format, or provider options such as think=False for Ollama)."""
        import litellm

        if self.model.startswith("ollama"):  # defaults that only make sense for Ollama
            params.setdefault("num_ctx", s.OLLAMA_NUM_CTX)
            params.setdefault("keep_alive", s.OLLAMA_KEEP_ALIVE)
        system = [{"role": "system", "content": self.system}] if self.system else []
        try:
            resp = litellm.completion(
                model=self.model,
                messages=system + messages,
                api_key=self.api_key or None,
                api_base=self.base_url or None,
                timeout=self.timeout,
                drop_params=True,  # silently skip options a provider does not support
                **params,
            )
        except Exception as exc:
            hint = " Is the server running, and is the model name / API key right?" \
                if type(exc).__name__ in {"APIConnectionError", "AuthenticationError", "NotFoundError"} else ""
            raise LLMError(f"LLM call to '{self.model}' failed: {type(exc).__name__}: {exc}.{hint}") from exc
        inp, out = TokenCount.accum(resp.usage) if resp.usage else (0, 0)
        choice = resp.choices[0]
        return Completion(choice.message.content or "", choice.finish_reason, inp, out)

    def run(self, messages: list[dict], **params) -> str:
        return self.complete(messages, **params).text
