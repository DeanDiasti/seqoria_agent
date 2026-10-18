"""Reusable settings copy containers and honor explicit overrides."""

import asyncio
import threading

import pytest

from seqoria_agent import AsyncAIClient
from seqoria_agent.client.settings import ClientSettings
from seqoria_agent.providers import AsyncBaseProvider, BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class UnusedProvider(AsyncBaseProvider):
    async def async_generate(self, **kwargs):
        raise AssertionError("Tools should execute without model calls")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.asyncio
async def test_settings_use_mcp_servers_and_explicit_empty_override():
    from seqoria_agent.client.mcp_transport import MCPServerConfig

    settings = ClientSettings(
        providers=[UnusedProvider()], mcp_servers=[MCPServerConfig("http://unused.test/mcp")]
    )
    configured = AsyncAIClient(settings=settings)
    assert configured.list_mcp_servers() == list(settings.mcp_servers)
    async with AsyncAIClient(settings=settings, mcp_servers=[]) as client:
        assert client.list_mcp_servers() == []


class _RecordingBlockingProvider(BaseProvider):
    def __init__(self):
        self.thread_ids = []
        self.requests = []

    def generate(self, chat_history, tools=None, tool_choice=None, model=None, **kwargs):
        self.thread_ids.append(threading.get_ident())
        self.requests.append((chat_history, tools, tool_choice, model, kwargs))
        return LLMResponse(model_id=model or "offline", text="ok", type="llm_response")

    def stream(self, chat_history, tools=None, tool_choice=None, model=None, **kwargs):
        yield self.generate(chat_history, tools, tool_choice, model, **kwargs)


def test_async_client_rejects_sync_providers_in_settings_before_allocating_resources():
    with pytest.raises(TypeError, match="requires async providers"):
        AsyncAIClient(settings=ClientSettings(providers=[_RecordingBlockingProvider()]))


def test_settings_copy_detaches_configuration_containers():
    from seqoria_agent import LocalToolbox
    from seqoria_agent.client.settings import ClientSettings

    toolbox = LocalToolbox()
    settings = ClientSettings(
        mcp_servers=["https://example.invalid/mcp?secret=credential"],
        local_toolbox=[toolbox],
        retry_policy={"max_attempts": 2},
    )
    copied = settings.copy()
    copied.mcp_servers.clear()
    copied.local_toolbox.clear()
    copied.retry_policy["max_attempts"] = 3
    assert settings.mcp_servers == ["https://example.invalid/mcp?secret=credential"]
    assert settings.local_toolbox == [toolbox]
    assert settings.retry_policy == {"max_attempts": 2}
    assert "credential" not in repr(settings)


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


def test_settings_copy_and_explicit_overrides():
    from seqoria_agent.client.settings import ClientSettings

    settings = ClientSettings(providers=[EchoProvider()], refresh_interval=30)
    copied = settings.with_provider(EchoProvider())
    assert len(settings.providers) == 1
    assert len(copied.providers) == 2
    client = AsyncAIClient(settings=settings, refresh_interval=0)
    assert client.registry._refresh_interval == 0
    assert client.registry._refresher_task is None
