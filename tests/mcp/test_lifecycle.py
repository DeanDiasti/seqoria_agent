"""MCP transports close only resources they own, including cancelled startup."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.mcp_transport import HTTPClientPool, MCPTransportFactory, connect_mcp


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
        yield (object(), object(), None)

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
