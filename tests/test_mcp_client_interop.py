"""Interoperability with independent MCP servers after removing the server wrapper."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import Context, FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

import seqoria_agent
from seqoria_agent import AsyncAIClient
from seqoria_agent.client.mcp_transport import MCPTransportFactory
from seqoria_agent.models import ToolCallError, ToolCallRequest
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class UnusedProvider(AsyncBaseProvider):
    async def async_generate(self, **kwargs):
        return LLMResponse(model_id="offline", type="llm_response", text="unused")

    async def async_stream(self, **kwargs):
        yield await self.async_generate(**kwargs)


def test_server_wrapper_is_not_exported():
    import seqoria_agent.models as models

    assert not hasattr(seqoria_agent, "SeqoriaMCP")
    assert "SeqoriaMCP" not in seqoria_agent.__all__
    assert not hasattr(models, "XMCPError")


@pytest.mark.asyncio
async def test_client_discovers_and_calls_official_mcp_server(monkeypatch):
    server = FastMCP("independent")
    request_ids = []
    closed = False

    @server.tool()
    async def greet(name: str, ctx: Context) -> str:
        request_ids.append(ctx.request_context.request_id)
        return f"Hello {name}"

    @server.tool()
    async def denied() -> str:
        raise ValueError("access denied")

    @asynccontextmanager
    async def connect(config, **kwargs):
        nonlocal closed
        try:
            async with create_connected_server_and_client_session(server) as session:
                yield SimpleNamespace(session=session, transport_type="http")
        finally:
            closed = True

    monkeypatch.setattr(MCPTransportFactory, "connect", connect)
    async with AsyncAIClient(
        providers=[UnusedProvider()], mcp_servers=["http://independent.test/mcp"]
    ) as client:
        tools = {item["function"]["name"]: item["function"] for item in client.list_tools()}
        assert set(tools) == {"greet", "denied"}
        assert "ctx" not in tools["greet"]["parameters"]["properties"]
        results = await client.run_tool_calls([
            ToolCallRequest(id="hello", name="greet", arguments={"name": "Alice"}),
            ToolCallRequest(id="error", name="denied", arguments={}),
        ])
        assert results[0].result == "Hello Alice"
        assert results[0].tool_call_id == "hello"
        assert isinstance(results[1], ToolCallError)
        assert "access denied" in results[1].error
        assert results[1].tool_call_id == "error"
        assert request_ids
    assert closed
    assert client.pool.connections == []
