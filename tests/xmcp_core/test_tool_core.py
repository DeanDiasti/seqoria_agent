from __future__ import annotations

import pytest

from seqoria_agent.server.xmcp_core import ToolCore
from seqoria_agent.models.errors import XMCPError


@pytest.fixture()
def core():
    return ToolCore(strict_mode=False, strict_server_params=True, json_wrapper=False)


def test_duplicate_name_detection(core):
    @core.tool
    def foo(a: int):
        return a

    # registering another tool with the same name must raise
    with pytest.raises(XMCPError):
        @core.tool  # noqa: F811 – intentionally shadow
        def foo(a: int):  # type: ignore[function-redefined]
            return a


def test_json_schema_render(core):
    @core.tool
    def add(x: int, y: int = 1):
        return x + y

    schema = core.tools["add"].input_schema  # type: ignore[attr-defined]
    assert schema["type"] == "object"
    assert schema["properties"]["x"]["type"] == "integer"
    assert "required" in schema and "x" in schema["required"]


@pytest.mark.asyncio
async def test_sync_and_async_execution(core):
    @core.tool
    def hello(name: str):
        return {"greeting": f"hi {name}"}

    @core.tool
    async def wave():
        return {"wave": True}

    xml1 = await core.run_tool("hello", {"name": "Bob"})
    xml2 = await core.run_tool("wave")

    assert xml1.strip().startswith("<tool_response")
    assert xml2.strip().startswith("<tool_response") 


# ---------------------------------------------------------------------------
# BaseModel integration tests
# ---------------------------------------------------------------------------

from pydantic import BaseModel


class Address(BaseModel):
    street: str
    city: str
    country: str


def test_basemodel_schema_generation(core):
    @core.tool
    def locate(address: Address):
        return 1

    schema = core.tools["locate"].input_schema  # type: ignore[attr-defined]
    # Ensure the parameter is represented as an *object* with nested properties
    addr_schema = schema["properties"]["address"]
    assert addr_schema["type"] == "object"
    for prop in ("street", "city", "country"):
        assert prop in addr_schema["properties"], f"{prop} not exposed in schema"


@pytest.mark.asyncio
async def test_basemodel_runtime_coercion(core):
    seen: dict[str, bool] = {}

    @core.tool
    def echo(address: Address):
        # Tool should receive a validated Address instance
        seen["is_instance"] = isinstance(address, Address)
        return {"city": address.city}

    payload = {"address": {"street": "1", "city": "Town", "country": "X"}}
    xml = await core.run_tool("echo", payload)

    assert seen.get("is_instance") is True
    # Basic sanity check on output serialization
    assert "Town" in xml 