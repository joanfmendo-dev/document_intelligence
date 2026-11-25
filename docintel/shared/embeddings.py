"""Embedding client, built on litellm like the LLM client (same model as the clinic_assistant project: nomic-embed-text).

    emb = Embedder()
    vectors = emb.embed_documents(["chest pain on exertion", "ankle sprain"])   # texts to be searched
    query = emb.embed_query("trouble breathing at night")                       # a question

Documents and queries are embedded differently on purpose: nomic-embed-text expects the prefixes
"search_document: " and "search_query: ", and the client adds them (other models get no prefix).
"""
from __future__ import annotations

import os
from typing import Sequence

from .llm import LLMError, is_ollama
from .settings import s

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


class EmbeddingError(LLMError):
    """The embedding model could not be reached, or answered with something unusable."""


class Embedder:
    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, batch_size: int = 32):
        config = s.role("embedding")
        self.model = model or config["model"]
        self.api_key = config["api_key"] if api_key is None else api_key
        self.base_url = config["base_url"] if base_url is None else base_url
        self.timeout = timeout
        self.batch_size = max(1, batch_size)

    # nomic-embed-text was trained with task prefixes; leaving them out noticeably hurts retrieval
    @property
    def _prefixes(self) -> tuple[str, str]:
        return ("search_document: ", "search_query: ") if "nomic" in self.model.lower() else ("", "")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import litellm

        litellm.suppress_debug_info = True  # do not print litellm's "Give Feedback / Get Help" banner on every failed call
        try:
            resp = litellm.embedding(model=self.model, input=texts, api_key=self.api_key or None,
                                     api_base=self.base_url or None, timeout=self.timeout)
            vectors = [list(map(float, item["embedding"])) for item in resp.data]
        except Exception as exc:
            hint = ""
            if type(exc).__name__ in {"APIConnectionError", "NotFoundError", "BadRequestError"}:
                hint = (f" Is the server running, and is the model pulled (ollama pull {self.model.split('/', 1)[-1]})?" if is_ollama(self.model) else
                        " Check the model name, base_url and api_key (Azure: the model is azure/<the name of the deployment>, and base_url the resource's address).")
            raise EmbeddingError(f"embedding call to '{self.model}' failed: {type(exc).__name__}: {exc}.{hint}") from exc
        if len(vectors) != len(texts) or any(not v for v in vectors):
            raise EmbeddingError(f"'{self.model}' returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in order. Texts go to the model in batches of `batch_size`."""
        prefix, out = self._prefixes[0], []
        texts = list(texts)
        for i in range(0, len(texts), self.batch_size):
            out += self._embed([prefix + t for t in texts[i:i + self.batch_size]])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._prefixes[1] + text])[0]
