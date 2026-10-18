"""Regression coverage for MCP lifetimes and request parameter resolution."""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.mcp_transport import (
    HTTPClientPool,
    MCPServerConfig,
    MCPTransportFactory,
    connect_mcp,
)
from seqoria_agent.client.parameters import ParameterRegistry
from seqoria_agent.server import SeqoriaMCP
from seqoria_agent.toolbox import LocalToolbox


@pytest.mark.asyncio
async def test_cancelled_pool_start_closes_already_connected_servers(monkeypatch):
    connected = asyncio.Event()
    pending = asyncio.Event()
    closed = asyncio.Event()
    session = object()

    @asynccontextmanager
    async def connect(config, **kwargs):
        if config.url.endswith("pending"):
            pending.set()
            await asyncio.Event().wait()
        try:
            yield SimpleNamespace(session=session, transport_type="http")
        finally:
            closed.set()

    monkeypatch.setattr(MCPTransportFactory, "connect", connect)
    pool = MCPConnectionPool(["http://server/connected", "http://server/pending"])
    original_connect = pool._connect

    async def track_connected(config):
        result = await original_connect(config)
        connected.set()
        return result

    monkeypatch.setattr(pool, "_connect", track_connected)
    startup = asyncio.create_task(pool.start())
    await asyncio.wait_for(connected.wait(), 1)
    await asyncio.wait_for(pending.wait(), 1)
    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup
    try:
        assert closed.is_set()
        assert pool.connections == []
        assert not pool._started
        assert pool._http_client_pool is None
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_connect_mcp_is_an_async_context_manager(monkeypatch):
    closed = False
    expected = object()

    @asynccontextmanager
    async def connect(config, **kwargs):
        nonlocal closed
        assert config.url == "stdio://example"
        try:
            yield expected
        finally:
            closed = True

    monkeypatch.setattr(MCPTransportFactory, "connect", connect)
    async with connect_mcp("stdio://example") as connection:
        assert connection is expected
        assert not closed
    assert closed


@pytest.mark.asyncio
async def test_http_pool_does_not_create_clients_after_concurrent_close():
    pool = HTTPClientPool()
    await pool._client_lock.acquire()
    closing = asyncio.create_task(pool.close())
    await asyncio.sleep(0)
    acquiring = asyncio.create_task(pool.get_client("https://example.com"))
    await asyncio.sleep(0)
    pool._client_lock.release()
    await closing
    try:
        with pytest.raises(RuntimeError, match="closed"):
            await acquiring
        assert pool._clients == {}
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_http_pool_honors_zero_timeout():
    pool = HTTPClientPool()
    try:
        client = await pool.get_client("https://example.com", timeout=0)
        assert client.timeout.connect == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_initialization", [False, True])
async def test_oauth_loopback_listener_closes_without_a_callback(
    monkeypatch, fail_initialization
):
    import seqoria_agent.client.mcp_transport as transport

    listener = SimpleNamespace(
        sockets=[SimpleNamespace(getsockname=lambda: ("127.0.0.1", 12345))],
        close=lambda: closed.append(True),
        wait_closed=AsyncMock(),
    )
    closed = []
    headers = {"X-Tenant": "example"}
    client_options = {}
    monkeypatch.setattr(transport.asyncio, "start_server", AsyncMock(return_value=listener))

    @asynccontextmanager
    async def http_client(**kwargs):
        client_options.update(kwargs)
        yield object()

    @asynccontextmanager
    async def streams(*args, **kwargs):
        yield object(), object(), None

    @asynccontextmanager
    async def session(*args, **kwargs):
        async def initialize():
            if fail_initialization:
                raise RuntimeError("connection failed")

        yield SimpleNamespace(initialize=initialize)

    monkeypatch.setattr(transport.httpx, "AsyncClient", http_client)
    monkeypatch.setattr(transport, "streamable_http_client", streams)
    monkeypatch.setattr(transport, "ClientSession", session)
    config = MCPServerConfig("https://example.com/mcp", oauth=True, headers=headers)

    async def connect():
        async with MCPTransportFactory._connect_http_oauth(config, config.url):
            assert not closed

    if fail_initialization:
        with pytest.raises(RuntimeError, match="connection failed"):
            await connect()
    else:
        await connect()
    assert closed
    listener.wait_closed.assert_awaited()
    assert client_options["headers"] == headers


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["registry", "local", "remote"])
async def test_sync_parameter_providers_run_off_loop_with_request_context(kind):
    request_identity = ContextVar("identity", default="missing")
    token = request_identity.set("authenticated-tenant")
    event_loop_thread = get_ident()
    calls = []

    def provider(*args):
        calls.append((get_ident(), request_identity.get()))
        return request_identity.get()

    try:
        if kind == "registry":
            registry = ParameterRegistry({"tenant": provider})
            result = await registry.resolve("tenant", context={})
        else:
            tools = LocalToolbox() if kind == "local" else SeqoriaMCP("thread-test")
            tools.register_param_provider(provider)

            @tools.tool(server_populated=["tenant"])
            async def identify(tenant: str) -> str:
                return tenant

            if kind == "local":
                result = (await tools.call_tool("identify", {}))[0].text
            else:
                async with create_connected_server_and_client_session(tools) as session:
                    response = await session.call_tool("identify", {})
                    assert not response.isError
                    result = response.content[0].text
        assert result == "authenticated-tenant"
        assert calls == [(calls[0][0], "authenticated-tenant")]
        assert calls[0][0] != event_loop_thread
    finally:
        request_identity.reset(token)


def test_local_tool_rejects_invalid_registration_before_discovery():
    tools = LocalToolbox()

    def positional_only(value: int, /) -> int:
        return value

    with pytest.raises(TypeError, match="keyword arguments"):
        tools.tool()(positional_only)

    def valid(value: int) -> int:
        return value

    with pytest.raises(ValueError, match="Unknown server-populated"):
        tools.tool(server_populated=["typo"])(valid)
    assert tools.list_tools() == []


@pytest.mark.asyncio
async def test_unannotated_local_arguments_accept_json_values():
    tools = LocalToolbox()

    @tools.tool()
    async def echo(value):
        return value

    assert (await tools.call_tool("echo", {"value": 42}))[0].text == "42"
    assert (await tools.call_tool("echo", {"value": [1, 2]}))[0].text == "[\n  1,\n  2\n]"


def test_remote_tool_rejects_invalid_hidden_parameter_registration():
    server = SeqoriaMCP("registration-test")

    def valid(value: int) -> int:
        return value

    with pytest.raises(ValueError, match="Unknown server-populated"):
        server.tool(server_populated=["typo"])(valid)
    assert server.tool_metadata == {}


def test_principal_resolution_falls_back_past_empty_state():
    server = SeqoriaMCP("principal-test")
    principal = {"id": "trusted-user"}
    context = SimpleNamespace(request_context=SimpleNamespace(state={}, principal=principal))
    assert server._get_principal_from_context(context) is principal
    context.request_context.state = {"principal": principal}
    assert server._get_principal_from_context(context) is principal


@pytest.mark.asyncio
async def test_remote_principal_can_be_resolved_by_a_custom_provider():
    server = SeqoriaMCP("custom-principal")

    @server.register_param_provider
    async def provide_principal(ctx, tool_name, parameter):
        if parameter == "principal":
            return {"id": "trusted-user"}

    @server.tool(server_populated=["principal"])
    async def identity(principal: dict) -> str:
        return principal["id"]

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("identity", {})
        assert not result.isError
        assert result.content[0].text == "trusted-user"
