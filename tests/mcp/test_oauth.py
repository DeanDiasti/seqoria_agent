"""OAuth initialization cleans up listeners and rejects unsupported transports."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from seqoria_agent.client.mcp_transport import MCPServerConfig, MCPTransportFactory


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_initialization", [False, True])
async def test_oauth_loopback_listener_closes_without_a_callback(monkeypatch, fail_initialization):
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
        yield (object(), object(), None)

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
@pytest.mark.parametrize("url", ["sse://server.test/events", "stdio://server"])
async def test_oauth_is_rejected_on_unsupported_transports(url):
    from seqoria_agent.client.mcp_transport import MCPServerConfig

    with pytest.raises(ValueError, match="only for Streamable HTTP"):
        async with MCPTransportFactory.connect(MCPServerConfig(url=url, oauth=True)):
            pytest.fail("An unsupported OAuth connection must not be opened")
