from __future__ import annotations

import os

import httpx

from seqoria_agent.memory.embeddings.base import EmbedderProtocol


class OpenAIEmbedder(EmbedderProtocol):
    """OpenAI embedding adapter.

    Uses direct HTTPX calls to OpenAI's `/v1/embeddings` endpoint.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAIEmbedder requires OPENAI_API_KEY or an explicit api_key.")

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            http2=True,
        ) as client:
            response = await client.post("/embeddings", json=payload)
            response.raise_for_status()
            data = response.json().get("data", [])

        return [item.get("embedding", []) for item in data if isinstance(item, dict)]

