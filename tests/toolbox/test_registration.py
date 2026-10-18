"""Tool registration validates signatures and exposes usable discovery metadata."""

import pytest

from seqoria_agent.toolbox import LocalToolbox


@pytest.mark.asyncio
async def test_local_tool_list():
    """LocalToolbox lists registered tools."""
    tb = LocalToolbox()

    @tb.tool(description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    @tb.tool(description="Subtract two numbers", depends_on=["add"])
    def subtract(a: int, b: int) -> int:
        return a - b

    tools = tb.list_tools()
    assert len(tools) == 2
    add_tool = next((t for t in tools if t.name == "add"))
    assert add_tool.description == "Add two numbers"
    sub_tool = next((t for t in tools if t.name == "subtract"))
    assert "add" in sub_tool.annotations.get("depends_on", [])


def test_local_tool_rejects_invalid_registration_before_discovery():
    tools = LocalToolbox()

    def positional_only(value: int, /) -> int:
        return value

    with pytest.raises(TypeError, match="keyword arguments"):
        tools.tool()(positional_only)

    def valid(value: int) -> int:
        return value

    with pytest.raises(ValueError, match="Unknown server-populated"):
        tools.tool(server_populated=["typo"])(valid)
    assert tools.list_tools() == []
