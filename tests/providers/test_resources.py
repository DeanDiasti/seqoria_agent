"""Provider construction, HTTP stream ownership and cancellation."""

from unittest.mock import Mock

import httpx
import pytest

from seqoria_agent.providers.anthropic_provider import AnthropicProvider
from seqoria_agent.providers.errors import ModelNotFoundError
from seqoria_agent.providers.ollama_provider import OllamaProvider
from seqoria_agent.providers.openai_compatible_provider import OpenAICompatibleProvider
from seqoria_agent.providers.openai_provider import OpenAIProvider
from seqoria_agent.providers.schemas import LLMResponse


@pytest.mark.parametrize(
    ("provider_class", "kwargs"),
    [
        (OpenAIProvider, {}),
        (OpenAICompatibleProvider, {}),
        (AnthropicProvider, {}),
        (OllamaProvider, {"validate_on_init": True}),
    ],
)
def test_failed_constructor_closes_owned_http_client(monkeypatch, provider_class, kwargs):
    """A rejected model must not leak the HTTP pool created before validation."""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    error = ModelNotFoundError(model_id="missing", provider="test")
    monkeypatch.setattr(provider_class, "_validate_model_sync", Mock(side_effect=error))

    with pytest.raises(ModelNotFoundError) as caught:
        provider_class(model_id="missing", api_key="test", **kwargs)

    assert caught.value is error
    assert client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "anthropic", "compatible", "ollama"])
async def test_cancelling_error_body_read_closes_http_stream(monkeypatch, provider_name):
    """Cancellation between entering a response and checking status releases it."""
    import asyncio

    from seqoria_agent.providers.anthropic_provider import AsyncAnthropicProvider
    from seqoria_agent.providers.ollama_provider import AsyncOllamaProvider
    from seqoria_agent.providers.openai_compatible_provider import AsyncOpenAICompatibleProvider
    from seqoria_agent.providers.openai_provider import AsyncOpenAIProvider

    class BlockedErrorBody(httpx.AsyncByteStream):
        def __init__(self):
            self.started = asyncio.Event()
            self.closed = False

        async def __aiter__(self):
            self.started.set()
            await asyncio.Event().wait()
            yield b"error"

        async def aclose(self):
            self.closed = True

    body = BlockedErrorBody()
    client = httpx.AsyncClient(
        base_url="https://provider.test",
        transport=httpx.MockTransport(lambda request: httpx.Response(400, stream=body)),
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    provider = {
        "openai": AsyncOpenAIProvider,
        "anthropic": AsyncAnthropicProvider,
        "compatible": AsyncOpenAICompatibleProvider,
        "ollama": AsyncOllamaProvider,
    }[provider_name](model_id="test", api_key="test")
    stream = provider.async_stream([{"role": "user", "content": "hello"}])
    task = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(body.started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed
    finally:
        task.cancel()
        await stream.aclose()
        await provider.aclose()


@pytest.mark.asyncio
async def test_sync_adapter_cancellation_waits_for_pending_next_before_close():
    import asyncio
    import threading

    from seqoria_agent.client.ai_client.sync_client import _SyncProviderAdapter
    from seqoria_agent.providers.base_provider import BaseProvider

    class BlockingProvider(BaseProvider):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()
            self.closed = threading.Event()

        def generate(self, **kwargs):
            return LLMResponse(model_id="test", type="llm_response", text="answer")

        def stream(self, **kwargs):
            try:
                self.started.set()
                self.release.wait(timeout=2)
                yield self.generate()
            finally:
                self.closed.set()

    provider = BlockingProvider()
    stream = _SyncProviderAdapter(provider).async_stream([])
    task = asyncio.create_task(anext(stream))
    try:
        assert await asyncio.to_thread(provider.started.wait, 1)
        task.cancel()
        # Let cancellation enter the adapter while next() is still running.
        await asyncio.sleep(0)
        assert not provider.closed.is_set()
        provider.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.closed.is_set()
    finally:
        provider.release.set()
        await stream.aclose()
