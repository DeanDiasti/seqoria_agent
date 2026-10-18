"""Chat turns and tool results stay paired with their originating requests."""

import asyncio

import pytest

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.models.chunk import (
    AssistantMessage,
    ToolCallError,
    ToolCallRequest,
    ToolCallResult,
    ToolMessages,
)
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList


class _LLMOnlyStub(AsyncBaseProvider):
    """Provider that always returns a plain LLMResponse (no tools)."""

    def __init__(self):
        pass

    async def async_generate(self, *, chat_history, **kwargs):
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return LLMResponse(
            model_id="stub", type="llm_response", text="pong", usage=usage, stop_reason="end_turn"
        )

    async def async_stream(self, *, chat_history, **kwargs):
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        yield LLMResponse(
            model_id="stub", type="llm_response", text="pong", usage=usage, stop_reason="end_turn"
        )


@pytest.mark.asyncio
async def test_chat_llm_only_returns_assistant_message():
    """When no tools are available the chat loop should yield a single AssistantMessage."""
    provider = _LLMOnlyStub()
    ai = AsyncAIClient(mcp_servers=[], providers=[provider])
    async with ai as client:
        msgs = [m async for m in client.chat(user_query="ping", history=[])]
    assert len(msgs) >= 1
    res = msgs[-1]
    assert isinstance(res, AssistantMessage)
    assert res.content == "pong"


class _ToolCallingStub(AsyncBaseProvider):
    """Provider that first requests a tool call then responds with text."""

    def __init__(self):
        self.calls = 0

    async def async_generate(self, *, chat_history, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1 and tools:
            tc = ToolCall(id="1", function_name="foo", arguments={})
            usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return ToolCallList(model_id="stub", tool_calls=[tc], usage=usage)
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return LLMResponse(
            model_id="stub", type="llm_response", text="done", usage=usage, stop_reason="end_turn"
        )

    async def async_stream(self, *, chat_history, tools=None, **kwargs):
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        yield LLMResponse(
            model_id="stub", type="llm_response", text="done", usage=usage, stop_reason="end_turn"
        )


@pytest.mark.asyncio
async def test_chat_with_tool_call_executes_and_returns_tool_messages():
    """Chat loop should execute requested tool calls and yield tool result messages."""
    toolbox = LocalToolbox()

    @toolbox.tool()
    def foo():
        return "42"

    provider = _ToolCallingStub()
    ai = AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=toolbox)
    async with ai as client:
        msgs = [m async for m in client.chat(user_query="what is foo?", history=[])]
    assert provider.calls == 2
    assert len(msgs) == 2
    tool_msgs, final_msg = msgs
    assert isinstance(tool_msgs, list)
    assert any((isinstance(m, ToolCallRequest) for m in tool_msgs))
    assert any((isinstance(m, ToolCallResult) for m in tool_msgs))
    assert isinstance(final_msg, AssistantMessage)


class ReplyProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(model_id="test", text="done", thinking="reason", type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_failed_tools_return_to_model_in_chat_history(streaming):
    box = LocalToolbox()

    @box.tool()
    async def broken():
        raise RuntimeError("failed tool")

    class Provider(ReplyProvider):
        async def async_generate(self, *, chat_history, **kwargs):
            if not any((isinstance(item, ToolCallError) for item in chat_history)):
                return ToolCallList(
                    model_id="test",
                    usage=None,
                    tool_calls=[ToolCall(id="call", function_name="broken", arguments={})],
                )
            return await super().async_generate(chat_history=chat_history, **kwargs)

    async with AsyncAIClient(providers=[Provider()], local_toolbox=box) as client:
        history = []
        chat = client.stream_chat if streaming else client.chat
        events = [event async for event in chat("go", history)]
    assert isinstance(events[0], ToolMessages)
    assert isinstance(events[0][1], ToolCallError)
    assert events[0][0].id == events[0][1].tool_call_id
    assert isinstance(events[-1], AssistantMessage)
    assert events[-1].content == "done"
    assert events[-1].thinking == "reason"
    assert history[-1] is events[-1]


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_parallel_chat_results_stay_paired_with_their_requests(streaming):
    box = LocalToolbox()

    @box.tool()
    async def slow():
        await asyncio.sleep(0.02)
        return "slow-result"

    @box.tool()
    async def fast():
        return "fast-result"

    class ToolsProvider(EchoProvider):
        async def async_generate(self, *, chat_history, **kwargs):
            if not isinstance(chat_history[-1], AssistantMessage) and len(chat_history) == 2:
                return ToolCallList(
                    model_id="tools",
                    usage=None,
                    tool_calls=[
                        ToolCall(id="slow-id", function_name="slow", arguments={}),
                        ToolCall(id="fast-id", function_name="fast", arguments={}),
                    ],
                )
            return LLMResponse(model_id="tools", text="done", type="llm_response")

    async with AsyncAIClient(providers=[ToolsProvider()], local_toolbox=box) as client:
        history = []
        chat = client.stream_chat if streaming else client.chat
        events = [event async for event in chat("go", history)]
        tools = next((event for event in events if isinstance(event, ToolMessages)))
        assert [(tools[i].id, tools[i + 1].tool_call_id) for i in (0, 2)] == [
            ("slow-id", "slow-id"),
            ("fast-id", "fast-id"),
        ]
        assert tools[1].result == "slow-result"
        assert tools[3].result == "fast-result"
