"""MCP retries, circuit breakers and timeout configuration preserve call semantics."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, TextContent

from seqoria_agent.client.ai_client import connection_pool as pool_module
from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool, _ActiveConnection
from seqoria_agent.client.mcp_transport import HTTPClientPool
from seqoria_agent.models.errors import CircuitOpenError, ToolRemoteError


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
async def test_retry_then_success():
    """First attempt fails, second succeeds respecting retry_policy."""

    class FlakySession(MockSession):
        async def call_tool(self, name: str, arguments: dict):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("boom")
            return MockCallToolResult()

    session = FlakySession()
    pool = MCPConnectionPool(servers=[], retry_policy={"max_attempts": 2, "backoff_base": 0})
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
    for _ in range(threshold):
        with pytest.raises(RuntimeError):
            await pool.call_tool("echo", {})
    with pytest.raises(CircuitOpenError):
        await pool.call_tool("echo", {})


@pytest.mark.asyncio
async def test_http_pool_honors_zero_timeout():
    pool = HTTPClientPool()
    try:
        client = await pool.get_client("https://example.com", timeout=0)
        assert client.timeout.connect == 0
    finally:
        await pool.close()


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


@pytest.mark.parametrize(
    "kwargs, option",
    [
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
    ],
)
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
async def test_remote_tool_error_is_not_retried():
    result = CallToolResult(isError=True, content=[TextContent(type="text", text="denied")])
    session = AsyncMock()
    session.call_tool.return_value = result
    pool = MCPConnectionPool(retry_policy={"max_attempts": 3})
    pool._tool_map["protected"] = session
    with pytest.raises(ToolRemoteError, match="denied"):
        await pool.call_tool("protected", {})
    assert session.call_tool.await_count == 1
