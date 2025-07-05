"""Test Pydantic Field metadata integration with ToolCore schemas."""

import pytest

try:
    from pydantic import Field
except ModuleNotFoundError:
    pytest.skip("pydantic not installed", allow_module_level=True)

from seqoria_agent.toolbox import LocalToolbox


def test_field_schema_expansion():
    tb = LocalToolbox()

    @tb.tool()
    async def greet(
        name: str = Field(..., description="Name of the person to greet"),
        times: int = Field(1, ge=1, le=5, description="Number of repetitions"),
    ):
        return {"message": "hello"}

    # Retrieve schema via list_tools()
    tools = tb.list_tools()
    schema = next(t.inputSchema for t in tools if t.name == "greet")

    props = schema["properties"]

    # The 'name' property should carry the description and be required
    assert props["name"]["description"] == "Name of the person to greet"
    assert "name" in schema["required"]

    # The 'times' property should include numeric constraints and default
    assert props["times"]["minimum"] == 1
    assert props["times"]["maximum"] == 5
    assert props["times"]["default"] == 1