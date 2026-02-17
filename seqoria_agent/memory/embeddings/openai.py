from __future__ import annotations

from typing import Any

from seqoria_agent.memory.embeddings.base import EmbedderProtocol


class OpenAIEmbedder(EmbedderProtocol):
    """OpenAI embedding adapter.

    This class is optional and requires the `openai` package.
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "OpenAIEmbedder requires optional dependency 'openai'. "
                "Install with `pip install seqoria-agent[openai]`."
            ) from exc

        client: Any = AsyncOpenAI(api_key=self.api_key)
        response = await client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

