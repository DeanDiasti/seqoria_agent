"""In-process toolbox calls execute sync and async tools and propagate failures."""

import pytest

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse
from seqoria_agent.toolbox import LocalToolbox


class _ToolboxProvider(AsyncBaseProvider):
    """Bare-bones provider that just echoes a static response."""

    def __init__(self, *args, **kwargs):
        pass

    async def async_generate(self, *, chat_history, tools, tool_choice="auto"):
        return LLMResponse(model_id="dummy", type="llm_response", text="Static answer")

    async def async_stream(self, *args, **kwargs):
        yield LLMResponse(model_id="test", type="llm_delta", text="Static answer")
        yield LLMResponse(model_id="dummy", type="llm_response", text="Static answer")


def build_ai_client(toolbox: LocalToolbox):
    provider = _ToolboxProvider()
    return AsyncAIClient(mcp_servers=[], providers=[provider], local_toolbox=toolbox)


@pytest.mark.asyncio
async def test_local_tool_execution():
    """LocalToolbox executes tools in-process."""
    tb = LocalToolbox()

    @tb.tool()
    def add(a: int, b: int) -> int:
        return int(a) + int(b)

    async with build_ai_client(tb) as client:
        result = await client.call_mcp_tool("add", {"a": 2, "b": 3})
        text = result[0].text
        assert "5" in text


@pytest.mark.asyncio
async def test_local_tool_with_server_params():
    """LocalToolbox handles server-populated parameters."""
    tb = LocalToolbox()

    @tb.tool(server_populated=["user_id"])
    def greet(name: str, user_id: str) -> str:
        return f"Hello {name}, user {user_id}!"

    result = await tb.call_tool(
        "greet", {"name": "Alice"}, server_args={"user_id": "test-user-123"}
    )
    assert "Hello Alice" in result[0].text
    assert "test-user-123" in result[0].text


@pytest.mark.asyncio
async def test_local_tool_async():
    """LocalToolbox supports async tools."""
    tb = LocalToolbox()

    @tb.tool()
    async def async_add(a: int, b: int) -> int:
        return a + b

    result = await tb.call_tool("async_add", {"a": 5, "b": 7})
    assert "12" in result[0].text


@pytest.mark.asyncio
async def test_local_tool_error_propagation():
    """LocalToolbox propagates tool exceptions to executor."""
    tb = LocalToolbox()

    @tb.tool()
    def fail_tool() -> str:
        raise ValueError("Something went wrong!")

    with pytest.raises(ValueError, match="Something went wrong"):
        await tb.call_tool("fail_tool", {})


@pytest.mark.asyncio
async def test_local_tool_not_found():
    """LocalToolbox raises ValueError for unknown tools."""
    tb = LocalToolbox()
    with pytest.raises(ValueError, match="not found"):
        await tb.call_tool("nonexistent", {})
