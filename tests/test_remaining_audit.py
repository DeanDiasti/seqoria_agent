"""Regressions for defects found while auditing the retained core."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, ToolAnnotations

from seqoria_agent import AsyncAIClient
from seqoria_agent.client.ai_client import connection_pool as pool_module
from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.mcp_transport import MCPTransportFactory
from seqoria_agent.models import ToolCallRequest
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class OfflineProvider(AsyncBaseProvider):
    async def async_generate(self, **kwargs):
        return LLMResponse(type="llm_response", model_id="offline", text="unused")

    async def async_stream(self, **kwargs):
        yield await self.async_generate(**kwargs)


@pytest.mark.asyncio
async def test_remote_identity_annotations_do_not_inject_client_context(monkeypatch):
    server = FastMCP("independent")

    @server.tool(annotations=ToolAnnotations(server_populated=["user_id"]))
    async def identity() -> str:
        return "server-authenticated-user"

    @asynccontextmanager
    async def connect(*args, **kwargs):
        async with create_connected_server_and_client_session(server) as session:
            yield SimpleNamespace(session=session, transport_type="http")

    monkeypatch.setattr(MCPTransportFactory, "connect", connect)
    async with AsyncAIClient(
        providers=[OfflineProvider()], mcp_servers=["http://independent.test/mcp"]
    ) as client:
        result = await client.run_tool_calls(
            [ToolCallRequest(id="one", name="identity", arguments={})],
            context={"user_id": "client-identity"},
        )
    assert result[0].role == "tool_result"
    assert result[0].result == "server-authenticated-user"


@pytest.mark.asyncio
async def test_structured_only_mcp_result_preserves_its_payload():
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        content=[], structuredContent={"balance": 42, "currency": "USD"},
    )
    pool = MCPConnectionPool(tool_map={"account": session})
    result = await pool.call_tool("account", {})
    assert json.loads(result[0].text) == {"balance": 42, "currency": "USD"}


@pytest.mark.asyncio
async def test_breaker_cooldown_starts_when_failure_occurs(monkeypatch):
    now = 100.0

    async def slow_failure(*args):
        nonlocal now
        now = 200.0
        raise RuntimeError("transport failed")

    session = AsyncMock()
    session.call_tool.side_effect = slow_failure
    pool = MCPConnectionPool(tool_map={"slow": session}, breaker_threshold=1)
    pool._session_to_url = {session: "http://slow.test/mcp"}
    monkeypatch.setattr(pool_module, "time", SimpleNamespace(monotonic=lambda: now))
    with pytest.raises(RuntimeError, match="transport failed"):
        await pool.call_tool("slow", {})
    assert pool._breaker_until["http://slow.test/mcp"] == 260.0


@pytest.mark.parametrize("kwargs, option", [
    ({"retry_policy": {"max_attempts": 0}}, "max_attempts"),
    ({"retry_policy": {"max_attempts": -1}}, "max_attempts"),
    ({"retry_policy": {"max_attempts": 1.5}}, "max_attempts"),
    ({"retry_policy": {"max_attempts": True}}, "max_attempts"),
    ({"retry_policy": {"backoff_base": -1}}, "backoff_base"),
    ({"retry_policy": {"backoff_base": float("nan")}}, "backoff_base"),
    ({"retry_policy": {"backoff_base": float("inf")}}, "backoff_base"),
    ({"retry_policy": {"backoff_base": True}}, "backoff_base"),
    ({"retry_policy": {"max_retries": 3}}, "Unknown retry options"),
    ({"breaker_threshold": 0}, "breaker_threshold"),
    ({"breaker_threshold": 1.5}, "breaker_threshold"),
    ({"breaker_threshold": True}, "breaker_threshold"),
    ({"breaker_reset_after": -1}, "breaker_reset_after"),
    ({"breaker_reset_after": float("nan")}, "breaker_reset_after"),
    ({"breaker_reset_after": True}, "breaker_reset_after"),
])
def test_invalid_resilience_configuration_fails_before_invoking_tools(kwargs, option):
    with pytest.raises(ValueError, match=option):
        MCPConnectionPool(**kwargs)


@pytest.mark.asyncio
async def test_retry_policy_is_a_snapshot_of_configuration():
    policy = {"max_attempts": 1, "backoff_base": 0}
    session = AsyncMock()
    session.call_tool.side_effect = RuntimeError("transport failed")
    pool = MCPConnectionPool(retry_policy=policy, tool_map={"tool": session})
    policy["max_attempts"] = 3
    with pytest.raises(RuntimeError, match="transport failed"):
        await pool.call_tool("tool", {})
    assert session.call_tool.await_count == 1


@pytest.mark.asyncio
async def test_empty_structured_mcp_payload_is_preserved():
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(content=[], structuredContent={})
    pool = MCPConnectionPool(tool_map={"empty": session})
    assert json.loads((await pool.call_tool("empty", {}))[0].text) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_startup", [False, True])
@pytest.mark.parametrize("provided_pool", [False, True])
async def test_low_level_http_connections_close_only_owned_pools(
    monkeypatch, fail_startup, provided_pool
):
    from seqoria_agent.client import mcp_transport as transport

    pool = SimpleNamespace(get_client=AsyncMock(return_value=object()), close=AsyncMock())
    monkeypatch.setattr(transport, "HTTPClientPool", lambda: pool)

    @asynccontextmanager
    async def streams(*args, **kwargs):
        yield object(), object(), None

    @asynccontextmanager
    async def session(*args):
        async def initialize():
            if fail_startup:
                raise RuntimeError("initialization failed")
        yield SimpleNamespace(initialize=initialize)

    monkeypatch.setattr(transport, "streamable_http_client", streams)
    monkeypatch.setattr(transport, "ClientSession", session)

    async def use_connection():
        async with MCPTransportFactory.connect(
            "https://server.test/mcp", client_pool=pool if provided_pool else None
        ):
            pool.close.assert_not_awaited()

    if fail_startup:
        with pytest.raises(RuntimeError, match="initialization failed"):
            await use_connection()
    else:
        await use_connection()
    assert pool.close.await_count == (0 if provided_pool else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["sse://server.test/events", "stdio://server"])
async def test_oauth_is_rejected_on_unsupported_transports(url):
    from seqoria_agent.client.mcp_transport import MCPServerConfig

    with pytest.raises(ValueError, match="only for Streamable HTTP"):
        async with MCPTransportFactory.connect(MCPServerConfig(url=url, oauth=True)):
            pytest.fail("An unsupported OAuth connection must not be opened")


@pytest.mark.parametrize("kind", ["request", "result", "error"])
def test_incomplete_tool_messages_fail_validation(kind):
    from pydantic import ValidationError
    from seqoria_agent.models import ToolCallError, ToolCallResult

    model = {"request": ToolCallRequest, "result": ToolCallResult, "error": ToolCallError}[kind]
    with pytest.raises(ValidationError):
        model()


@pytest.mark.parametrize("kind, values", [
    ("request", {"id": "", "name": "tool"}),
    ("request", {"id": "call", "name": ""}),
    ("result", {"tool_call_id": "", "result": "ok"}),
    ("error", {"tool_call_id": "", "error": "failed"}),
])
def test_empty_tool_identifiers_fail_validation(kind, values):
    from pydantic import ValidationError
    from seqoria_agent.models import ToolCallError, ToolCallResult

    model = {"request": ToolCallRequest, "result": ToolCallResult, "error": ToolCallError}[kind]
    with pytest.raises(ValidationError):
        model(**values)


def test_valid_tool_messages_round_trip_and_argument_defaults_are_independent():
    from seqoria_agent.models import ToolCallError, ToolCallResult

    first = ToolCallRequest(id="one", name="tool")
    second = ToolCallRequest(id="two", name="tool")
    first.arguments["value"] = 1
    assert second.arguments == {}
    for message in [
        first, second,
        ToolCallResult(tool_call_id="one", result="ok"),
        ToolCallError(tool_call_id="two", error="failed"),
    ]:
        assert type(message).model_validate(message.model_dump()) == message
