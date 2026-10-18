"""The synchronous facade adapts blocking providers and guards event-loop use."""

import threading

import pytest

from seqoria_agent import AIClient
from seqoria_agent.client.ai_client.sync_client import _SyncProviderAdapter
from seqoria_agent.client.settings import ClientSettings
from seqoria_agent.providers import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


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


@pytest.mark.asyncio
async def test_sync_provider_adapter_accepts_positional_base_contract_in_worker_thread():
    provider = _RecordingBlockingProvider()
    adapter = _SyncProviderAdapter(provider)
    history = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    result = await adapter.async_generate(history, tools, "auto", "selected", temperature=0.2)
    chunks = [
        chunk
        async for chunk in adapter.async_stream(history, tools, "auto", "selected", temperature=0.2)
    ]
    assert result.model_id == "selected"
    assert chunks == [result]
    assert provider.requests == [(history, tools, "auto", "selected", {"temperature": 0.2})] * 2
    assert all((identifier != threading.get_ident() for identifier in provider.thread_ids))


def test_sync_client_adapts_sync_providers_from_settings():
    with AIClient(settings=ClientSettings(providers=[_RecordingBlockingProvider()])) as client:
        assert client.chat_sync("hello", history=[]).content == "ok"


class _BlockingProvider(BaseProvider):
    def generate(self, **kwargs):
        return LLMResponse(model_id="blocking", text="ok", type="llm_response")

    def stream(self, **kwargs):
        yield self.generate(**kwargs)


def test_sync_client_accepts_a_real_sync_provider_contract():
    with AIClient(providers=[_BlockingProvider()]) as client:
        assert client.chat_sync("hello", history=[]).content == "ok"
        assert list(client.stream_chat("hello", history=[]))[-1].content == "ok"


@pytest.mark.asyncio
async def test_sync_facade_rejects_running_event_loop_cleanly():
    client = AIClient(providers=[_BlockingProvider()])
    with pytest.raises(RuntimeError, match="Use AsyncAIClient"):
        client.close()
