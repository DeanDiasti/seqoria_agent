"""Removed constructor and settings options fail explicitly."""

import asyncio

import pytest

from seqoria_agent import AIClient, AsyncAIClient
from seqoria_agent.client.settings import ClientSettings
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class UnusedProvider(AsyncBaseProvider):
    async def async_generate(self, **kwargs):
        raise AssertionError("Tools should execute without model calls")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.parametrize("option,value", [("urls", []), ("client_kwargs", {"timeout": 5})])
def test_obsolete_settings_and_constructor_options_fail_fast(option, value):
    with pytest.raises(TypeError):
        ClientSettings(**{option: value})
    for client_type in (AsyncAIClient, AIClient):
        with pytest.raises(TypeError, match="Unknown client options"):
            client_type(providers=[UnusedProvider()], **{option: value})


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.parametrize("option", ["cache", "cache_ttl"])
@pytest.mark.parametrize("constructor", [AsyncAIClient, AIClient])
def test_removed_cache_options_are_rejected(constructor, option):
    with pytest.raises(TypeError, match="Unknown client options"):
        constructor(providers=[EchoProvider()], **{option: None})


@pytest.mark.parametrize("option", ["cache", "cache_ttl"])
def test_removed_cache_settings_are_rejected(option):
    from seqoria_agent.client.settings import ClientSettings

    with pytest.raises(TypeError, match=option):
        ClientSettings(**{option: None})
