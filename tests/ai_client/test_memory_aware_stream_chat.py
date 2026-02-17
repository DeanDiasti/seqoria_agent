import pytest

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.memory import AgenticMemory
from seqoria_agent.models.chunk import AssistantMessage, ChatChunk
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage


class _StreamingStub(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):  # type: ignore[override]
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return LLMResponse(
            model_id="stream-stub",
            type="llm_response",
            text="stream-final",
            usage=usage,
            stop_reason="end_turn",
        )

    async def async_stream(self, *, chat_history, **kwargs):  # type: ignore[override]
        yield LLMResponse(model_id="stream-stub", type="llm_delta", text="stream-")
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        yield LLMResponse(
            model_id="stream-stub",
            type="llm_response",
            text="stream-final",
            usage=usage,
            stop_reason="end_turn",
        )


@pytest.mark.asyncio
async def test_stream_chat_accepts_agentic_memory_history():
    memory = AgenticMemory()
    ai = AsyncAIClient(urls=[], providers=[_StreamingStub()])

    async with ai as client:
        chunks = [c async for c in client.stream_chat("hello", history=memory)]

    assert any(isinstance(item, ChatChunk) for item in chunks)
    assert isinstance(chunks[-1], AssistantMessage)
    payload = memory.to_dict()
    roles = [item["role"] for item in payload["context"]["raw_messages"]]
    assert roles[-1] == "assistant"


@pytest.mark.asyncio
async def test_chat_and_stream_chat_memory_role_parity():
    provider = _StreamingStub()
    ai = AsyncAIClient(urls=[], providers=[provider])
    chat_memory = AgenticMemory()
    stream_memory = AgenticMemory()

    async with ai as client:
        _ = [c async for c in client.chat("hi", history=chat_memory)]
        _ = [c async for c in client.stream_chat("hi", history=stream_memory)]

    chat_roles = [item["role"] for item in chat_memory.to_dict()["context"]["raw_messages"]]
    stream_roles = [item["role"] for item in stream_memory.to_dict()["context"]["raw_messages"]]
    assert chat_roles == stream_roles

