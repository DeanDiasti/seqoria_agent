from __future__ import annotations

import pytest

from seqoria_agent.client.ai_client.base import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage


class _DummyProvider(BaseProvider):
    """Minimal provider that returns deterministic values for testing."""

    def __init__(self):
        # No configuration needed for dummy provider
        pass

    async def async_generate(self, chat_history, tools=None, tool_choice=None, **kwargs):  # type: ignore[override]
        usage = LLMUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12)
        return LLMResponse(model_id="dummy", text="hello", usage=usage, tool_calls=None)

    async def async_stream(self, chat_history, tools=None, tool_choice=None, **kwargs):  # type: ignore[override]
        usage = LLMUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        # emit two deltas and then the final response
        yield "h"
        yield "i"
        yield LLMResponse(model_id="dummy", text="hi", usage=usage, tool_calls=None)


@pytest.mark.asyncio
async def test_llm_generate_once_usage():
    ai = AIClient(providers=[_DummyProvider()], urls=[])
    hist = [{"role": "user", "content": "hi"}]
    res = await ai.llm_generate_once(hist, use_tools=False, include_usage=True)
    assert isinstance(res, dict)
    assert res["usage"].total_tokens == 12  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_llm_stream_once_usage():
    ai = AIClient(providers=[_DummyProvider()], urls=[])
    hist = [{"role": "user", "content": "hi"}]
    chunks = []
    async for c in ai.llm_stream_once(hist, use_tools=False, include_usage=True):
        chunks.append(c)
    assert any(isinstance(ch, dict) and ch["usage"].total_tokens == 7 for ch in chunks) 