"""Client-managed provider resources close with their owner."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seqoria_agent import AsyncAIClient
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.asyncio
async def test_provider_lifecycle_releases_owned_http_resources():

    class Provider(EchoProvider):
        def __init__(self):
            self.async_client = SimpleNamespace(aclose=AsyncMock())

    provider = Provider()
    async with provider:
        async with AsyncAIClient(providers=[provider]):
            pass
        provider.async_client.aclose.assert_not_awaited()
    provider.async_client.aclose.assert_awaited_once()
