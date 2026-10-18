"""Tool schemas and argument validation support nested models and JSON values."""

from typing import Literal

import pytest
from pydantic import BaseModel

from seqoria_agent import LocalToolbox


@pytest.mark.asyncio
async def test_unannotated_local_arguments_accept_json_values():
    tools = LocalToolbox()

    @tools.tool()
    async def echo(value):
        return value

    assert (await tools.call_tool("echo", {"value": 42}))[0].text == "42"
    assert (await tools.call_tool("echo", {"value": [1, 2]}))[0].text == "[\n  1,\n  2\n]"


def test_local_schema_supports_nested_and_optional_types():

    class Item(BaseModel):
        label: str

    box = LocalToolbox()

    @box.tool()
    def inspect_items(items: list[Item], mode: Literal["fast", "full"], count: int | None = None):
        return items

    schema = box.list_tools()[0].inputSchema
    assert schema["properties"]["items"]["type"] == "array"
    assert schema["properties"]["mode"]["enum"] == ["fast", "full"]
    assert schema["required"] == ["items", "mode"]
    assert "Item" in schema["$defs"]


@pytest.mark.asyncio
async def test_local_arguments_are_validated_and_nested_models_reach_the_tool():
    from pydantic import ValidationError

    class Payload(BaseModel):
        label: str

    box = LocalToolbox()

    @box.tool()
    async def typed(payload: Payload, count: int = 2):
        return payload.label * count

    assert (await box.call_tool("typed", {"payload": {"label": "a"}}))[0].text == "aa"
    with pytest.raises(ValidationError):
        await box.call_tool("typed", {"payload": {"label": "a"}, "count": "invalid"})
    with pytest.raises(ValidationError):
        await box.call_tool("typed", {"payload": {"label": "a"}, "undeclared": True})
