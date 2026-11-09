"""extract_vlm: page image -> Markdown with a vision-language model, through the shared LLM layer.

The model can be local (Ollama) or a commercial API: change `settings.s.VLM_MODEL`, or pass
`model=`. Slower than OCR and it can hallucinate or "correct" values silently, so validate its
output downstream. Use it where OCR struggles (complex tables, handwriting, stamps, photos).
For Ollama: `ollama pull qwen3-vl:8b`.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

from ._entries import text_result
from ._images import iter_page_images
from .llm import LLM, image_content
from .settings import s

DEFAULT_PROMPT = (
    "Transcribe this document page to Markdown.\n"
    "- Copy the text exactly as written: names, numbers, units, dates and codes must not change.\n"
    "- Keep the reading order. Use '#' / '##' headings for section titles and Markdown tables for tables.\n"
    "- Do not summarize, translate, correct, complete or add anything. Keep empty fields empty.\n"
    "- If something is illegible write [illegible]. If the page is blank, output nothing.\n"
    "- Output only the transcription, without code fences or comments."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*\n(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _clean_output(text: str) -> str:
    """Remove reasoning tags and a wrapping code fence if the model added them."""
    text = _THINK_RE.sub("", text or "").strip()
    match = _FENCE_RE.match(text)
    return (match.group(1) if match else text).strip()


def extract_vlm(
    path: str | Path,
    model: str | None = None,
    dpi: int = 150,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 4096,
    timeout: float = 600.0,
    **llm_params,
) -> dict:
    """Transcribe a PDF or image, one page per model call. A VLM gives no reliable positions,
    so every page's "text_with_coordinates" is an empty list.

    `llm_params` go to the model call (e.g. think=False, num_ctx=16384 for Ollama)."""
    llm = LLM(model=model or s.VLM_MODEL, api_key=s.VLM_API_KEY, base_url=s.VLM_BASE_URL, timeout=timeout)
    params = {"temperature": 0, "max_tokens": max_tokens, **llm_params}  # the caller may override any of them
    texts = []
    for number, image in enumerate(iter_page_images(path, dpi), start=1):
        done = llm.complete([{"role": "user", "content": image_content(prompt, [image])}], **params)
        if done.finish_reason == "length":
            warnings.warn(
                f"Page {number} hit the output limit: the transcription may be cut off. "
                "Raise max_tokens, or pass think=False if the model spends tokens reasoning.",
                stacklevel=2,
            )
        texts.append(_clean_output(done.text))
    return text_result(texts)
