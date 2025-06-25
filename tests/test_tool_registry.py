import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.ai_client.connection_pool import ConnectionPool
from seqoria_agent.toolbox import LocalToolbox

# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------

class _DummyTransport:
    """Minimal stub mimicking SimpleHTTPTransport for testing."""

    def __init__(self, tools):
        self._tools = tools

    async def list_tools(self):  # noqa: D401
        return self._tools

    async def close(self):
        return None


class _DummyConnPool(ConnectionPool):
    """Drop-in replacement that skips real networking for tests."""

    def __init__(self, servers):
        # servers: list[ (url, tools) ] where tools is sequence of SimpleNamespace
        self._connections = [SimpleNamespace(url=url, client=_DummyTransport(tools)) for url, tools in servers]

    @property
    def connections(self):  # type: ignore[override]
        return list(self._connections)

    async def start(self):
        return None

    async def close(self):
        return None


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture()
async def registry():
    # Prepare fake remote tool definitions ----------------------------------
    remote_tool = SimpleNamespace(
        name="remote_echo",
        description="Echo input.",
        inputSchema={"type": "object", "properties": {"text": {"type": "string"}}},
        annotations={
            "is_display_tool": False,
            "server_populated": ["tenant_id"],
            "depends_on": [],
        },
    )

    pool = _DummyConnPool(servers=[("https://mock", [remote_tool])])

    # Local toolbox with one inline display tool -----------------------------
    local_box = LocalToolbox()

    @local_box.display_tool()
    def local_show(text: str):  # noqa: D401
        """Display text."""
        return {"display": text}

    reg = ToolRegistry(pool=pool, local_toolboxes=[local_box], tool_map={}, refresh_interval=0)
    await reg.refresh()
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# All tests in this module are asynchronous
pytestmark = pytest.mark.asyncio

async def test_merge_and_metadata(registry):
    # Remote tool present
    assert any(t["function"]["name"] == "remote_echo" for t in registry.llm_tools)
    # Local tool present
    assert any(t["function"]["name"] == "local_show" for t in registry.llm_tools)

    # is_display_tool works
    assert registry.is_display_tool("local_show") is True
    assert registry.is_display_tool("remote_echo") is False

    # server_params
    assert registry.server_params("remote_echo") == ["tenant_id"]
    assert registry.server_params("local_show") == []


async def test_duplicate_detection():
    # Duplicate name between remote & local should raise RuntimeError
    dup_tool = SimpleNamespace(
        name="dup",
        description="",
        inputSchema={},
        annotations={},
    )
    pool = _DummyConnPool(servers=[("https://mock", [dup_tool])])

    box = LocalToolbox()

    @box.tool()
    def dup():  # noqa: D401, F811
        return "dup"

    reg = ToolRegistry(pool=pool, local_toolboxes=[box], tool_map={}, refresh_interval=0)
    with pytest.raises(RuntimeError):
        await reg.refresh()


async def test_depends_on_access(registry):
    # Default empty set when unknown
    assert registry.depends_on("remote_echo") == set()
    assert registry.depends_on("nonexistent") == set() 