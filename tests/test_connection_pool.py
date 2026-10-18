"""Tests for the MCP Connection Pool.

These tests verify the resilience features (circuit breaker, retry)
work correctly with the new MCP-based connection pool.
"""

from unittest.mock import MagicMock

import pytest

from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool, _ActiveConnection
from seqoria_agent.models.errors import CircuitOpenError


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

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
async def test_retry_then_success():
    """First attempt fails, second succeeds respecting retry_policy."""
    class FlakySession(MockSession):
        async def call_tool(self, name: str, arguments: dict):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("boom")
            return MockCallToolResult()

    session = FlakySession()
    pool = MCPConnectionPool(
        servers=[],
        retry_policy={"max_attempts": 2, "backoff_base": 0},
    )
    pool._tool_map = {"echo": session}
    pool._session_to_url = {session: "mem://flaky"}
    pool._connections = [
        _ActiveConnection(session=session, url="mem://flaky", transport_type="http")
    ]

    result = await pool.call_tool("echo", {})
    assert result[0].text == "ok"
    assert session.call_count == 2


@pytest.mark.asyncio
async def test_circuit_breaker_trips():
    """Breaker opens after threshold consecutive failures and blocks further calls."""
    class DeadSession(MockSession):
        async def call_tool(self, name: str, arguments: dict):
            self.call_count += 1
            raise RuntimeError("always broken")

    session = DeadSession()
    threshold = 2
    pool = MCPConnectionPool(
        servers=[],
        retry_policy={"max_attempts": 1, "backoff_base": 0},
        breaker_threshold=threshold,
        breaker_reset_after=60,
    )
    pool._tool_map = {"echo": session}
    pool._session_to_url = {session: "mem://dead"}
    pool._connections = [
        _ActiveConnection(session=session, url="mem://dead", transport_type="http")
    ]

    # Exceed failure threshold
    for _ in range(threshold):
        with pytest.raises(RuntimeError):
            await pool.call_tool("echo", {})

    # Next call should raise CircuitOpenError immediately
    with pytest.raises(CircuitOpenError):
        await pool.call_tool("echo", {})


@pytest.mark.asyncio
async def test_tool_not_found():
    """Raises ValueError when tool is not in tool_map."""
    pool = MCPConnectionPool(servers=[])
    
    with pytest.raises(ValueError, match="not found"):
        await pool.call_tool("nonexistent", {})
