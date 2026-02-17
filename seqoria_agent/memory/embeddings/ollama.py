from __future__ import annotations

import os
from typing import Any

from seqoria_agent.memory.embeddings.base import EmbedderProtocol


class OllamaEmbedder(EmbedderProtocol):
    """Ollama embedding adapter for memory retrieval."""

    def __init__(
        self,
        *,
        model: str = "mxbai-embed-large",
        host: str | None = None,
        api_key: str | None = None,
        **embed_kwargs: Any,
    ) -> None:
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        effective_api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        self.headers = (
            {"Authorization": f"Bearer {effective_api_key}"} if effective_api_key else None
        )
        self._embed_kwargs = embed_kwargs

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            from ollama import AsyncClient  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "OllamaEmbedder requires optional dependency 'ollama'. "
                "Install with `pip install seqoria-agent[ollama]`."
            ) from exc

        client: Any = AsyncClient(host=self.host, headers=self.headers)
        embed = getattr(client, "embed", None)

        if callable(embed):
            response = await embed(model=self.model, input=texts, **self._embed_kwargs)
            embeddings = self._extract_embeddings(response)
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    "Ollama embed endpoint returned an unexpected number of vectors. "
                    f"Expected {len(texts)}, got {len(embeddings)}."
                )
            return embeddings

        # Backward compatibility for older Ollama clients with `embeddings`.
        if not hasattr(client, "embeddings"):
            raise RuntimeError(
                "Installed ollama client does not support embeddings APIs. "
                "Upgrade ollama to a newer version."
            )

        vectors: list[list[float]] = []
        for text in texts:
            legacy_response = await client.embeddings(model=self.model, prompt=text)
            vector = self._extract_embedding(legacy_response)
            vectors.append(vector)
        return vectors

    @staticmethod
    def _extract_embeddings(response: Any) -> list[list[float]]:
        if isinstance(response, dict):
            embeddings = response.get("embeddings")
        else:
            embeddings = getattr(response, "embeddings", None)

        if not isinstance(embeddings, list):
            raise RuntimeError("Ollama embed response did not contain an 'embeddings' list.")

        if not embeddings:
            return []

        first_item = embeddings[0]
        if not isinstance(first_item, list):
            raise RuntimeError("Ollama embed response embeddings were not vectors.")

        return embeddings

    @staticmethod
    def _extract_embedding(response: Any) -> list[float]:
        if isinstance(response, dict):
            vector = response.get("embedding")
        else:
            vector = getattr(response, "embedding", None)

        if not isinstance(vector, list):
            raise RuntimeError("Ollama embeddings response did not contain an 'embedding' vector.")
        return vector
