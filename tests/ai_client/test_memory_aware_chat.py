import pytest

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.memory import AgenticMemory
from seqoria_agent.models.chunk import AssistantMessage, ToolCallRequest, ToolCallResult
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList
from seqoria_agent.toolbox import LocalToolbox


class _LLMOnlyStub(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):  # type: ignore[override]
        usage = LLMUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)
        return LLMResponse(
            model_id="stub",
            type="llm_response",
            text="memory-aware reply",
            usage=usage,
            stop_reason="end_turn",
        )

    async def async_stream(self, *, chat_history, **kwargs):  # type: ignore[override]
        async def _gen():
            usage = LLMUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)
            yield LLMResponse(
                model_id="stub",
                type="llm_response",
                text="memory-aware reply",
                usage=usage,
                stop_reason="end_turn",
            )

        return _gen()


class _ToolStub(AsyncBaseProvider):
    def __init__(self):
        self.calls = 0

    async def async_generate(self, *, chat_history, tools=None, **kwargs):  # type: ignore[override]
        self.calls += 1
        if self.calls == 1 and tools:
            usage = LLMUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)
            return ToolCallList(
                model_id="stub",
                tool_calls=[ToolCall(id="tc-1", function_name="echo_tool", arguments={"x": "42"})],
                usage=usage,
            )
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return LLMResponse(
            model_id="stub",
            type="llm_response",
            text="done",
            usage=usage,
            stop_reason="end_turn",
        )

    async def async_stream(self, *, chat_history, tools=None, **kwargs):  # type: ignore[override]
        async def _gen():
            usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            yield LLMResponse(
                model_id="stub",
                type="llm_response",
                text="done",
                usage=usage,
                stop_reason="end_turn",
            )

        return _gen()


@pytest.mark.asyncio
async def test_chat_accepts_agentic_memory_history():
    memory = AgenticMemory()
    ai = AsyncAIClient(urls=[], providers=[_LLMOnlyStub()])

    async with ai as client:
        messages = [m async for m in client.chat("hello", history=memory)]

    assert isinstance(messages[-1], AssistantMessage)
    payload = memory.to_dict()
    roles = [item["role"] for item in payload["context"]["raw_messages"]]
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.asyncio
async def test_chat_tool_cycle_preserved_in_memory():
    toolbox = LocalToolbox()

    @toolbox.tool()
    def echo_tool(x: str) -> str:  # noqa: ANN001
        return x

    memory = AgenticMemory()
    provider = _ToolStub()
    ai = AsyncAIClient(urls=[], providers=[provider], local_toolbox=toolbox)

    async with ai as client:
        chunks = [c async for c in client.chat("use tool", history=memory)]

    assert provider.calls == 2
    assert any(isinstance(item, list) for item in chunks)
    payload = memory.to_dict()
    roles = [item["role"] for item in payload["context"]["raw_messages"]]
    assert "tool_call" in roles
    assert "tool_result" in roles
    assert any(isinstance(item, ToolCallRequest) for item in chunks[0])
    assert any(isinstance(item, ToolCallResult) for item in chunks[0])


@pytest.mark.asyncio
async def test_chat_plain_list_backwards_compatible():
    history: list[dict] = []
    ai = AsyncAIClient(urls=[], providers=[_LLMOnlyStub()])

    async with ai as client:
        chunks = [c async for c in client.chat("hello", history=history)]

    assert isinstance(chunks[-1], AssistantMessage)
    roles = [item["role"] if isinstance(item, dict) else getattr(item, "role", None) for item in history]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles

