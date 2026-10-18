import pytest

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import ToolCall, ToolCallList, LLMResponse, LLMUsage


class _ToolPickingStub(AsyncBaseProvider):
    """Provider that requests a tool call based on the tools list it receives."""

    def __init__(self) -> None:
        self.calls = 0

    async def async_generate(self, *, chat_history, tools=None, **kwargs):  # type: ignore[override]  # noqa: ANN001
        self.calls += 1
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)

        # First call: request a tool call if tools provided
        if self.calls == 1 and tools:
            tool_names = [t["function"]["name"] for t in tools]
            pick = "foo" if "foo" in tool_names else "bar"
            return ToolCallList(model_id="stub", tool_calls=[ToolCall(id="1", function_name=pick, arguments={})], usage=usage)

        # Second call: normal assistant response
        return LLMResponse(model_id="stub", type="llm_response", text="done", usage=usage, stop_reason="end_turn")

    async def async_stream(self, *args, **kwargs):  # noqa: ANN001
        async def _gen():
            usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            yield LLMResponse(model_id="stub", type="llm_response", text="done", usage=usage, stop_reason="end_turn")
        return _gen()


@pytest.mark.asyncio
async def test_blocked_tool_not_listed_and_not_exposed_to_provider():
    tb = LocalToolbox()

    @tb.tool()
    def foo():  # noqa: D401, ANN001
        return "foo"

    @tb.tool()
    def bar():  # noqa: D401, ANN001
        return "bar"

    provider = _ToolPickingStub()

    async with AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=tb, blocked_tools=["foo"]) as client:
        # Registry tool list should exclude blocked tool
        names = [t["function"]["name"] for t in client.list_tools()]
        assert "foo" not in names
        assert "bar" in names

        # Chat loop should lead the provider to request `bar` (since `foo` isn't visible)
        msgs = [m async for m in client.chat(user_query="hi", history=[])]
        # Provider invoked at least twice (tool call then final response)
        assert provider.calls >= 2
        # One of the yielded messages should be tool messages containing the chosen tool name
        tool_msgs = next(m for m in msgs if isinstance(m, list))
        assert any(getattr(x, "name", None) == "bar" for x in tool_msgs)


@pytest.mark.asyncio
async def test_block_tools_runtime_then_refresh_hides_tool():
    tb = LocalToolbox()

    @tb.tool()
    def foo():  # noqa: D401, ANN001
        return "foo"

    @tb.tool()
    def bar():  # noqa: D401, ANN001
        return "bar"

    provider = _ToolPickingStub()

    async with AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=tb) as client:
        names_before = [t["function"]["name"] for t in client.list_tools()]
        assert "foo" in names_before and "bar" in names_before

        await client.block_tools(["foo"], refresh=True)
        names_after = [t["function"]["name"] for t in client.list_tools()]
        assert "foo" not in names_after and "bar" in names_after


