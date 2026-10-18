"""Remote servers can be added and removed without duplicate connections."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from seqoria_agent import AsyncAIClient
from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool, _ActiveConnection
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


@pytest.mark.asyncio
async def test_pool_add_server_after_start_connects_once():
    pool = MCPConnectionPool(servers=[])
    await pool.start()
    assert pool.connections == []
    fake_session = object()
    pool._connect = AsyncMock(
        return_value=_ActiveConnection(
            session=fake_session, url="http://example.com/mcp", transport_type="http"
        )
    )
    added = await pool.add_server("http://example.com/mcp", connect=True)
    assert added == 1
    assert len(pool.connections) == 1
    added2 = await pool.add_server("http://example.com/mcp", connect=True)
    assert added2 == 0
    assert len(pool.connections) == 1
    assert pool._connect.call_count == 1


@pytest.mark.asyncio
async def test_pool_remove_server_disconnects_and_prunes_tool_map():
    pool = MCPConnectionPool(servers=[])
    await pool.start()
    url = "http://example.com/mcp"
    fake_session = object()
    stopped = asyncio.Event()
    closed = asyncio.Event()

    async def owner():
        await stopped.wait()
        closed.set()

    task = asyncio.create_task(owner())
    conn = _ActiveConnection(
        session=fake_session, url=url, transport_type="http", _owner_task=task, _stop_event=stopped
    )
    pool._connections = [conn]
    pool._session_to_url = {fake_session: url}
    pool._tool_map = {"tool_a": fake_session, "tool_keep": object()}
    added = await pool.add_server(url, connect=False)
    assert added == 1
    assert url in pool.list_server_urls()
    removed = await pool.remove_server(url, disconnect=True)
    assert removed == 1
    assert url not in pool.list_server_urls()
    assert pool.connections == []
    assert closed.is_set()
    assert task.done()
    assert "tool_a" not in pool._tool_map
    assert "tool_keep" in pool._tool_map


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.asyncio
async def test_generator_server_additions_and_removals_are_not_consumed_twice():
    async with AsyncAIClient(providers=[EchoProvider()]) as client:
        added = await client.add_mcp_servers(
            (url for url in ["https://example.invalid/mcp"]), connect=False
        )
        assert added == 1
        assert len(client.list_mcp_servers()) == 1
        removed = await client.remove_mcp_servers((url for url in ["https://example.invalid/mcp"]))
        assert removed == 1
        assert client.list_mcp_servers() == []
