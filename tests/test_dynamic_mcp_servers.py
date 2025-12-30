from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool, _ActiveConnection


@pytest.mark.asyncio
async def test_pool_add_server_after_start_connects_once():
    pool = MCPConnectionPool(servers=[])

    # Start with no servers should mark pool as started but have no connections
    await pool.start()
    assert pool.connections == []

    # Patch _connect so we don't touch real transports
    fake_session = object()
    pool._connect = AsyncMock(  # type: ignore[method-assign]
        return_value=_ActiveConnection(session=fake_session, url="http://example.com/mcp", transport_type="http")
    )

    added = await pool.add_server("http://example.com/mcp", connect=True)
    assert added == 1
    assert len(pool.connections) == 1

    # Adding the same server again should be deduped by URL
    added2 = await pool.add_server("http://example.com/mcp", connect=True)
    assert added2 == 0
    assert len(pool.connections) == 1

    # _connect should only have been called once total
    assert pool._connect.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_pool_remove_server_disconnects_and_prunes_tool_map():
    pool = MCPConnectionPool(servers=[])
    await pool.start()

    url = "http://example.com/mcp"
    fake_session = object()

    # Simulate an active connection with a kept-alive context manager
    cm = AsyncMock()
    conn = _ActiveConnection(session=fake_session, url=url, transport_type="http", _context_manager=cm)
    pool._connections = [conn]
    pool._context_managers = [cm]
    pool._session_to_url = {fake_session: url}  # type: ignore[assignment]

    # Simulate tool routing to that session
    pool._tool_map = {"tool_a": fake_session, "tool_keep": object()}  # type: ignore[assignment]

    # Add server config and then remove it
    added = await pool.add_server(url, connect=False)
    assert added == 1
    assert url in pool.list_server_urls()

    removed = await pool.remove_server(url, disconnect=True)
    assert removed == 1
    assert url not in pool.list_server_urls()

    # Connection removed
    assert pool.connections == []
    # Context manager closed (best-effort)
    assert cm.__aexit__.called
    # Tool map entries pointing to removed session are pruned
    assert "tool_a" not in pool._tool_map
    assert "tool_keep" in pool._tool_map


