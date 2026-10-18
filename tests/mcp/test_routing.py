"""MCP aliases route calls to the correct server and original tool name."""

import asyncio
from unittest.mock import MagicMock

import pytest

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool, _ActiveConnection
from seqoria_agent.models.chunk import ToolCallRequest
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class MockCallToolResult:
    """Mock result from session.call_tool()."""

    def __init__(self, text: str = "ok"):
        self.content = [MagicMock(text=text)]


class MockSession:
    """A minimal mock for MCP ClientSession."""

    def __init__(self, responses=None, exc: Exception | None = None):
        self._responses = list(responses or [])
        self._exc = exc
        self.call_count = 0

    async def call_tool(self, name: str, arguments: dict):
        self.call_count += 1
        if self._exc is not None:
            raise self._exc
        if self._responses:
            return self._responses.pop(0)
        return MockCallToolResult()

    async def list_tools(self):
        return MagicMock(tools=[])


@pytest.mark.asyncio
async def test_call_tool_success():
    """Pool forwards calls to session and returns the result."""
    session = MockSession(responses=[MockCallToolResult(text="yay")])
    pool = MCPConnectionPool(servers=[])
    pool._tool_map = {"echo": session}
    pool._session_to_url = {session: "mem://success"}
    pool._connections = [
        _ActiveConnection(session=session, url="mem://success", transport_type="http")
    ]
    result = await pool.call_tool("echo", {"msg": "hi"})
    assert result[0].text == "yay"
    assert session.call_count == 1


@pytest.mark.asyncio
async def test_tool_not_found():
    """Raises ValueError when tool is not in tool_map."""
    pool = MCPConnectionPool(servers=[])
    with pytest.raises(ValueError, match="not found"):
        await pool.call_tool("nonexistent", {})


class ReplyProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(model_id="test", text="done", thinking="reason", type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


def request(name, **arguments):
    return ToolCallRequest(id=name, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_registry_deduplicates_aliased_schema_and_keeps_first_route():
    primary, first_variant, duplicate_variant = (LocalToolbox(), LocalToolbox(), LocalToolbox())

    @primary.tool(name="lookup")
    async def lookup(value: str):
        return value

    async def variant(count: int):
        return str(count)

    first_variant.tool(name="lookup")(variant)
    duplicate_variant.tool(name="lookup")(variant)
    async with AsyncAIClient(
        providers=[ReplyProvider()], local_toolbox=[primary, first_variant, duplicate_variant]
    ) as client:
        tools = client.list_tools()
        assert len(tools) == 2
        alias = next((t["function"]["name"] for t in tools if t["function"]["name"] != "lookup"))
        assert client.pool._tool_map[alias] is first_variant
        assert (await client.run_tool_calls([request(alias, count=3)]))[0].result == "3"


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


def _tool_request(name, call_id=None, **arguments):
    return ToolCallRequest(id=call_id or name, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_schema_alias_routes_to_original_tool_name():
    first, second = (LocalToolbox(), LocalToolbox())

    @first.tool(name="echo")
    async def one(value: str):
        return value

    @second.tool(name="echo")
    async def two(number: int):
        return number * 2

    async with AsyncAIClient(providers=[EchoProvider()], local_toolbox=[first, second]) as client:
        alias = next(
            (t["function"]["name"] for t in client.list_tools() if "__" in t["function"]["name"])
        )
        result = await client.run_tool_calls([_tool_request(alias, number=3)])
        assert result[0].result == "6"
