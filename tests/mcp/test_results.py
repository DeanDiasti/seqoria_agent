"""Structured MCP result payloads survive conversion, including empty objects."""

import json
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult

from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool


@pytest.mark.asyncio
async def test_structured_only_mcp_result_preserves_its_payload():
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        content=[], structuredContent={"balance": 42, "currency": "USD"}
    )
    pool = MCPConnectionPool(tool_map={"account": session})
    result = await pool.call_tool("account", {})
    assert json.loads(result[0].text) == {"balance": 42, "currency": "USD"}


@pytest.mark.asyncio
async def test_empty_structured_mcp_payload_is_preserved():
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(content=[], structuredContent={})
    pool = MCPConnectionPool(tool_map={"empty": session})
    assert json.loads((await pool.call_tool("empty", {}))[0].text) == {}
