from __future__ import annotations

import os
from typing import Any

import httpx

from seqoria_agent.memory.embeddings.base import EmbedderProtocol


class OllamaEmbedder(EmbedderProtocol):
    """Ollama embedding adapter for memory retrieval via HTTPX `/api/embed`."""

    def __init__(
        self,
        *,
        model: str,
        host: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        **embed_kwargs: Any,
    ) -> None:
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        self.timeout = timeout
        self._embed_kwargs = embed_kwargs

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            **self._embed_kwargs,
        }

        async with httpx.AsyncClient(
            base_url=self.host,
            headers=headers,
            timeout=self.timeout,
            http2=True,
        ) as client:
            response = await client.post("/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()

        embeddings = self._extract_embeddings(data)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Ollama embed endpoint returned an unexpected number of vectors. "
                f"Expected {len(texts)}, got {len(embeddings)}."
            )
        return embeddings

    @staticmethod
    def _extract_embeddings(response: Any) -> list[list[float]]:
        if isinstance(response, dict):
            embeddings = response.get("embeddings")
        else:
            embeddings = None

        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama embed response did not contain an 'embeddings' list.")

        if not embeddings:
            return []

        first_item = embeddings[0]
        if not isinstance(first_item, list):
            raise RuntimeError("Ollama embed response embeddings were not vectors.")

        return embeddings

