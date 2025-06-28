from __future__ import annotations

import pytest
from seqoria_agent.client.ai_client.base import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage, ToolCall
from seqoria_agent.models.chunk import ChatChunk


class _SingleTurnProvider(BaseProvider):
    def __init__(self):
        pass

    async def async_generate(self, chat_history, tools=None, tool_choice=None, **kwargs):  # type: ignore[override]
        usage = LLMUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        return LLMResponse(model_id="dummy", text="done", usage=usage, tool_calls=None)

    async def async_stream(self, chat_history, tools=None, tool_choice=None, **kwargs):  # type: ignore[override]
        usage = LLMUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        yield "d"
        yield "one"
        yield LLMResponse(model_id="dummy", text="done", usage=usage, tool_calls=None)


@pytest.mark.asyncio
async def test_chat_usage_keys():
    ai = AIClient(providers=[_SingleTurnProvider()], urls=[])
    hist: list[dict] = []
    res = await ai.chat("hi", hist, include_usage=True)
    assert res["usage_turn"].total_tokens == 5  # type: ignore[attr-defined]
    assert res["usage_total"].total_tokens == 5  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stream_chat_usage_chunks():
    ai = AIClient(providers=[_SingleTurnProvider()], urls=[])
    hist: list[dict] = []
    token_chunks = []
    async for ch in ai.stream_chat("hi", hist, include_usage=True):
        if ch.type == "token_usage":
            token_chunks.append(ch)
    # Expect two chunks: per-turn and cumulative
    assert len(token_chunks) == 2
    per_turn = [c for c in token_chunks if not c.cumulative][0]
    cumulative = [c for c in token_chunks if c.cumulative][0]
    assert per_turn.usage.total_tokens == 5  # type: ignore[attr-defined]
    assert cumulative.usage.total_tokens == 5  # type: ignore[attr-defined] 