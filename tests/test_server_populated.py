"""Tests for hidden parameters in LocalToolbox."""

import pytest

from seqoria_agent.toolbox import LocalToolbox


@pytest.fixture()
def local_toolbox():
    """Create LocalToolbox with server-populated params."""
    tb = LocalToolbox()

    # Tool with single server-populated param
    @tb.tool(server_populated=["user_id"])
    def greet(name: str, user_id: str) -> str:
        return f"Hi {name} (uid={user_id})"

    # Tool with multiple server-populated params
    @tb.tool(server_populated=["user_id", "tenant_id"])
    def info(user_id: str, tenant_id: str) -> str:
        return f"user={user_id}, tenant={tenant_id}"

    return tb


@pytest.mark.asyncio
async def test_local_toolbox_injection(local_toolbox):
    """Verify LocalToolbox injects server-populated params."""
    result = await local_toolbox.call_tool("greet", {"name": "Alice"}, server_args={"user_id": "local_user_456"})
    text = result[0].text
    assert "Hi Alice" in text
    assert "local_user_456" in text


@pytest.mark.asyncio
async def test_local_toolbox_multiple_params(local_toolbox):
    """Verify multiple server-populated params are injected."""
    result = await local_toolbox.call_tool("info", {}, server_args={"user_id": "local_user_456", "tenant_id": "tenant_abc"})
    text = result[0].text
    assert "local_user_456" in text
    assert "tenant_abc" in text


@pytest.mark.asyncio
async def test_local_toolbox_server_args(local_toolbox):
    """Verify server_args supplies trusted values."""
    result = await local_toolbox.call_tool(
        "greet",
        {"name": "Bob"},
        server_args={"user_id": "explicit_user"},
    )
    text = result[0].text
    assert "explicit_user" in text


@pytest.mark.asyncio
async def test_local_toolbox_forbid_override():
    """Verify strict mode prevents LLM from overriding server params."""
    tb = LocalToolbox(strict_server_params=True)

    @tb.tool(server_populated=["secret"])
    def reveal(secret: str) -> str:
        return f"secret={secret}"

    # Attempting to pass server-populated param from "LLM side" should fail
    with pytest.raises(ValueError, match="Cannot override server-populated"):
        await tb.call_tool("reveal", {"secret": "hacked"})


@pytest.mark.asyncio
async def test_local_toolbox_list_shows_annotations():
    """Verify list_tools exposes annotations for server-populated params."""
    tb = LocalToolbox()

    @tb.tool(server_populated=["user_id"], depends_on=["auth"])
    def protected(data: str, user_id: str) -> str:
        return data

    tools = tb.list_tools()
    tool = next(t for t in tools if t.name == "protected")

    assert "user_id" in tool.annotations.get("server_populated", [])
    assert "auth" in tool.annotations.get("depends_on", [])
