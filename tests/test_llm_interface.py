import pytest

from seqoria_agent.client.ai_client.llm_interface import LLMInterface
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.errors import LLMError
from seqoria_agent.providers.schemas import LLMResponse


class _FailProvider(BaseProvider):
    """Provider that always fails to let fallback logic kick in."""

    def __init__(self):
        super().__init__(model_id="fail")

    async def async_generate(self, **kwargs):  # noqa: D401, ANN001
        raise LLMError("upstream error")

    async def async_stream(self, **kwargs):  # noqa: D401, ANN001
        raise LLMError("upstream error")


class _OkProvider(BaseProvider):
    """Simple provider that returns static responses suitable for unit tests."""

    def __init__(self):
        super().__init__(model_id="ok")

    async def async_generate(self, *, chat_history, **kwargs):  # noqa: D401
        return LLMResponse(model_id="ok", text="hello", tool_calls=[])

    async def async_stream(self, *, chat_history, **kwargs):  # noqa: D401
        async def _gen():
            # Simulate token deltas followed by final response
            yield "he"
            yield "llo"
            yield LLMResponse(model_id="ok", text="hello", tool_calls=[])

        return _gen()


@pytest.mark.asyncio
async def test_generate_fallback():
    """First provider fails – second succeeds."""
    llm = LLMInterface(providers=[_FailProvider(), _OkProvider()])
    resp = await llm.generate(history=[])
    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello"


@pytest.mark.asyncio
async def test_generate_basic():
    """Single happy-path provider returns an LLMResponse."""
    llm = LLMInterface(providers=[_OkProvider()])
    resp = await llm.generate(history=[{"role": "user", "content": "Hi"}])
    assert resp.text == "hello"
    assert resp.tool_calls == []


@pytest.mark.asyncio
async def test_stream_yields_deltas_and_final():
    """Ensure stream() yields string deltas followed by a final LLMResponse."""
    llm = LLMInterface(providers=[_OkProvider()])
    chunks = []
    async for ch in llm.stream(history=[]):
        chunks.append(ch)

    # Expect at least two string deltas and one final LLMResponse
    assert chunks[-1].__class__ is LLMResponse
    assert "".join([c for c in chunks if isinstance(c, str)]) == "hello" 