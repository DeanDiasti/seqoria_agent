import types

import pytest

from seqoria_agent.client.ai_client.connection_pool import ConnectionPool, _Connection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyTransport:
    """A minimal stand-in for SimpleHTTPTransport used in test-scenarios."""

    def __init__(self, url: str, responses=None, exc: Exception | None = None):
        self.url = url
        self._responses = list(responses or [])
        self._exc = exc
        self.closed = False
        # Stats
        self.call_count = 0

    # Async context-manager interface ----------------------------------
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        self.closed = True

    # Minimal MCP subset -----------------------------------------------
    async def call_tool(self, *, name: str, arguments: dict, server_args: dict | None = None):
        self.call_count += 1
        if self._exc is not None:
            raise self._exc
        if self._responses:
            return self._responses.pop(0)
        # Fallback generic response
        return [types.SimpleNamespace(text="ok")]

    # list_tools is used by ToolRegistry in later phases – implement stub
    async def list_tools(self):
        return []


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_tool_success():
    """Pool forwards calls to transport and returns the result as-is."""
    trans = DummyTransport(url="mem://success", responses=[[types.SimpleNamespace(text="yay")]])
    pool = ConnectionPool(urls=[], transport_cls=lambda url, **kw: trans, tool_map={"echo": trans})

    # No need to connect because we inject mapping directly
    pool._connections = [_Connection(client=trans, url="mem://success")]  # type: ignore[arg-type]

    result = await pool.call_tool("echo", {"msg": "hi"})
    assert result[0].text == "yay"
    assert trans.call_count == 1


@pytest.mark.asyncio
async def test_retry_then_success():
    """First attempt fails, second succeeds respecting retry_policy."""
    class FlakyTransport(DummyTransport):
        async def call_tool(self, *, name: str, arguments: dict, server_args: dict | None = None):
            self.call_count += 1
            # Fail first call
            if self.call_count == 1:
                raise RuntimeError("boom")
            return [types.SimpleNamespace(text="ok")]

    trans = FlakyTransport(url="mem://flaky")
    pool = ConnectionPool(
        urls=[],
        transport_cls=lambda url, **kw: trans,
        tool_map={"echo": trans},
        retry_policy={"max_attempts": 2, "backoff_base": 0},
    )
    pool._connections = [_Connection(client=trans, url="mem://flaky")]  # type: ignore[arg-type]

    result = await pool.call_tool("echo", {})
    assert result[0].text == "ok"
    assert trans.call_count == 2  # one failure + one success


@pytest.mark.asyncio
async def test_circuit_breaker_trips():
    """Breaker opens after threshold consecutive failures and blocks further calls."""
    class DeadTransport(DummyTransport):
        async def call_tool(self, *, name: str, arguments: dict, server_args: dict | None = None):
            self.call_count += 1
            raise RuntimeError("always broken")

    trans = DeadTransport(url="mem://dead")
    threshold = 2
    pool = ConnectionPool(
        urls=[],
        transport_cls=lambda url, **kw: trans,
        tool_map={"echo": trans},
        retry_policy={"max_attempts": 1, "backoff_base": 0},
        breaker_threshold=threshold,
        breaker_reset_after=60,
    )
    pool._connections = [_Connection(client=trans, url="mem://dead")]  # type: ignore[arg-type]

    # Exceed failure threshold
    for _ in range(threshold):
        with pytest.raises(RuntimeError):
            await pool.call_tool("echo", {})

    # Next call should raise CircuitOpenError immediately
    from seqoria_agent.models.errors import CircuitOpenError

    with pytest.raises(CircuitOpenError):
        await pool.call_tool("echo", {}) 