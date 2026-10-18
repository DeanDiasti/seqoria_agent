"""Invalid chat inputs fail before mutating conversation state."""

import pytest

from seqoria_agent import AsyncAIClient
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCall, ToolCallList


@pytest.mark.asyncio
async def test_llm_only_stream_rejects_unsolicited_tool_calls():

    class UnsolicitedProvider(AsyncBaseProvider):
        async def async_generate(self, **kwargs):
            raise AssertionError("stream path expected")

        async def async_stream(self, **kwargs):
            yield ToolCallList(
                model_id="offline",
                tool_calls=[ToolCall(id="one", function_name="unknown", arguments={})],
                usage=None,
            )

    history = []
    async with AsyncAIClient(providers=[UnsolicitedProvider()]) as client:
        with pytest.raises(ValueError, match="Unexpected tool call without tools"):
            _ = [chunk async for chunk in client.stream_chat("hello", history=history)]
    assert history[-1].role == "user"
    assert all((not hasattr(entry, "tool_call_id") for entry in history))


class ReplyProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(model_id="test", text="done", thinking="reason", type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.parametrize("value", [-1, True, float("inf"), float("nan")])
def test_refresh_interval_requires_a_nonnegative_finite_number(value):
    with pytest.raises(ValueError, match="refresh_interval"):
        AsyncAIClient(providers=[ReplyProvider()], refresh_interval=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
async def test_invalid_turn_limit_does_not_modify_history(streaming, limit):
    client = AsyncAIClient(providers=[ReplyProvider()])
    history = []
    chat = client.stream_chat if streaming else client.chat
    with pytest.raises(ValueError, match="max_turns"):
        await anext(chat("go", history, max_turns=limit))
    assert history == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_missing_query_is_rejected_without_changing_history(streaming):
    client = AsyncAIClient(providers=[ReplyProvider()])
    history = []
    chat = client.stream_chat if streaming else client.chat
    with pytest.raises(ValueError, match="user_query"):
        await anext(chat(None, history))
    assert history == []
