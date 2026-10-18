import pytest

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import ToolCall, ToolCallList, LLMResponse, LLMUsage
from seqoria_agent.models.chunk import AssistantMessage, ToolCallRequest, ToolCallResult


class _LLMOnlyStub(AsyncBaseProvider):
    """Provider that always returns a plain LLMResponse (no tools)."""

    def __init__(self):
        pass

    async def async_generate(self, *, chat_history, **kwargs):  # type: ignore[override]
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return LLMResponse(model_id="stub", type="llm_response", text="pong", usage=usage, stop_reason="end_turn")

    async def async_stream(self, *, chat_history, **kwargs):  # type: ignore[override]
        async def _gen():
            usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            yield LLMResponse(model_id="stub", type="llm_response", text="pong", usage=usage, stop_reason="end_turn")
        return _gen()


class _ToolCallingStub(AsyncBaseProvider):
    """Provider that first requests a tool call then responds with text."""

    def __init__(self):
        self.calls = 0

    async def async_generate(self, *, chat_history, tools=None, **kwargs):  # type: ignore[override]
        self.calls += 1
        # First LLM invocation – request the `foo` tool
        if self.calls == 1 and tools:
            tc = ToolCall(id="1", function_name="foo", arguments={})
            usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return ToolCallList(model_id="stub", tool_calls=[tc], usage=usage)
        # Second invocation after tool execution – return normal response
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return LLMResponse(model_id="stub", type="llm_response", text="done", usage=usage, stop_reason="end_turn")

    async def async_stream(self, *, chat_history, tools=None, **kwargs):  # type: ignore[override]
        async def _gen():
            usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            yield LLMResponse(model_id="stub", type="llm_response", text="done", usage=usage, stop_reason="end_turn")
        return _gen()


@pytest.mark.asyncio
async def test_chat_llm_only_returns_assistant_message():
    """When no tools are available the chat loop should yield a single AssistantMessage."""

    provider = _LLMOnlyStub()
    ai = AsyncAIClient(mcp_servers=[], providers=[provider])

    async with ai as client:
        # Collect all yielded messages from the async generator
        msgs = [m async for m in client.chat(user_query="ping", history=[])]

    # At least one message should be yielded and the final one must be an AssistantMessage
    assert len(msgs) >= 1
    res = msgs[-1]
    assert isinstance(res, AssistantMessage)
    assert res.content == "pong"


@pytest.mark.asyncio
async def test_chat_with_tool_call_executes_and_returns_tool_messages():
    """Chat loop should execute requested tool calls and yield tool result messages."""

    toolbox = LocalToolbox()

    @toolbox.tool()
    def foo():  # noqa: D401, ANN001 – simple demo tool
        return "42"

    provider = _ToolCallingStub()
    ai = AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=toolbox)

    async with ai as client:
        msgs = [m async for m in client.chat(user_query="what is foo?", history=[])]

    # The provider should have been invoked twice:
    #   1) to request the tool call
    #   2) to produce the final assistant answer after the tool result
    assert provider.calls == 2

    # Chat should yield two chunks:
    #   – A ToolMessages list (tool_call + tool_result)
    #   – A final AssistantMessage with the answer
    assert len(msgs) == 2

    tool_msgs, final_msg = msgs

    # First chunk – tool messages
    assert isinstance(tool_msgs, list)
    assert any(isinstance(m, ToolCallRequest) for m in tool_msgs)
    assert any(isinstance(m, ToolCallResult) for m in tool_msgs)

    # Second chunk – final assistant reply
    assert isinstance(final_msg, AssistantMessage)


# ---------------------------------------------------------------------------
# Executor tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executor_run_calls_returns_history_entries():
    """Executor.run_calls should return a list of ToolCallResult entries."""

    toolbox = LocalToolbox()

    @toolbox.tool()
    def foo():  # noqa: D401, ANN001
        return "42"

    provider = _LLMOnlyStub()
    ai = AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=toolbox)

    tc = ToolCallRequest(id="1", name="foo", arguments={})

    async with ai as client:
        entries = await client.executor.run_calls([tc])

    assert isinstance(entries, list)
    assert entries and isinstance(entries[0], ToolCallResult)
    assert entries[0].role == "tool_result" 