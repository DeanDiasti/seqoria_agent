"""Tool selection follows registry availability and explicit blocking."""

import pytest

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage, ToolCall, ToolCallList


class StubProvider(AsyncBaseProvider):
    """Provider that does nothing but satisfy the interface for tests."""

    def __init__(self):
        pass

    async def async_generate(self, *, chat_history, **kwargs):
        return None

    async def async_stream(self, *, chat_history, **kwargs):
        yield LLMResponse(model_id="test", type="llm_delta", text="")


@pytest.mark.asyncio
async def test_tools_enabled_toggle():
    provider = StubProvider()
    ai = AsyncAIClient(mcp_servers=[], providers=[provider])
    async with ai as client:
        assert client.tools_enabled() is False
    box = LocalToolbox()

    @box.tool()
    def foo():
        return 1

    ai2 = AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=box)
    async with ai2 as client:
        assert client.tools_enabled() is True


class _ToolPickingStub(AsyncBaseProvider):
    """Provider that requests a tool call based on the tools list it receives."""

    def __init__(self) -> None:
        self.calls = 0

    async def async_generate(self, *, chat_history, tools=None, **kwargs):
        self.calls += 1
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        if self.calls == 1 and tools:
            tool_names = [t["function"]["name"] for t in tools]
            pick = "foo" if "foo" in tool_names else "bar"
            return ToolCallList(
                model_id="stub",
                tool_calls=[ToolCall(id="1", function_name=pick, arguments={})],
                usage=usage,
            )
        return LLMResponse(
            model_id="stub", type="llm_response", text="done", usage=usage, stop_reason="end_turn"
        )

    async def async_stream(self, *args, **kwargs):
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        yield LLMResponse(
            model_id="stub", type="llm_response", text="done", usage=usage, stop_reason="end_turn"
        )


@pytest.mark.asyncio
async def test_blocked_tool_not_listed_and_not_exposed_to_provider():
    tb = LocalToolbox()

    @tb.tool()
    def foo():
        return "foo"

    @tb.tool()
    def bar():
        return "bar"

    provider = _ToolPickingStub()
    async with AsyncAIClient(
        mcp_servers=[], providers=[provider], local_toolbox=tb, blocked_tools=["foo"]
    ) as client:
        names = [t["function"]["name"] for t in client.list_tools()]
        assert "foo" not in names
        assert "bar" in names
        msgs = [m async for m in client.chat(user_query="hi", history=[])]
        assert provider.calls >= 2
        tool_msgs = next((m for m in msgs if isinstance(m, list)))
        assert any((getattr(x, "name", None) == "bar" for x in tool_msgs))


@pytest.mark.asyncio
async def test_block_tools_runtime_then_refresh_hides_tool():
    tb = LocalToolbox()

    @tb.tool()
    def foo():
        return "foo"

    @tb.tool()
    def bar():
        return "bar"

    provider = _ToolPickingStub()
    async with AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=tb) as client:
        names_before = [t["function"]["name"] for t in client.list_tools()]
        assert "foo" in names_before and "bar" in names_before
        await client.block_tools(["foo"], refresh=True)
        names_after = [t["function"]["name"] for t in client.list_tools()]
        assert "foo" not in names_after and "bar" in names_after


class ReplyProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(model_id="test", text="done", thinking="reason", type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_empty_or_fully_blocked_toolbox_uses_llm_only_path(streaming):
    box = LocalToolbox()

    @box.tool()
    async def hidden():
        return "hidden"

    async with AsyncAIClient(
        providers=[ReplyProvider()], local_toolbox=box, blocked_tools=["hidden"]
    ) as client:
        assert client.list_tools() == []
        assert not client.tools_enabled()
        chat = client.stream_chat if streaming else client.chat
        replies = [reply async for reply in chat("go", [])]
    assert replies[-1].content == "done"
    assert replies[-1].thinking == "reason"
