"""Structured parameter models reach hidden tool arguments intact."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.parameters import ParamContext, ParameterRegistry
from seqoria_agent.models.chunk import ToolCallRequest


class _StructuredPool:
    async def call_tool(self, *_, **__):
        return [SimpleNamespace(text="OK")]

    @property
    def connections(self):
        return []


class _StructuredRegistry:
    def __init__(self, spp_fields):
        self._spp_fields = spp_fields

    def server_params(self, name: str):
        return self._spp_fields.get(name, [])

    def depends_on(self, name: str):
        return set()


class Address(BaseModel):
    street: str
    city: str
    country: str


@pytest.mark.asyncio
async def test_basemodel_param_injection():

    async def provide_address():
        await asyncio.sleep(0)
        return Address(street="1", city="City", country="X")

    tool_name = "demo_address"
    tc = ToolCallRequest(
        role="tool_call", id="1", type="function_call", name=tool_name, arguments={}
    )
    reg = ParameterRegistry({Address: provide_address})
    exec_ = Executor(
        pool=_StructuredPool(),
        registry=_StructuredRegistry({tool_name: ["address"]}),
        param_registry=reg,
        allow_override=False,
    )
    context: ParamContext = {}
    results = await exec_.run_calls([tc], context=context)
    assert results[0].role == "tool_result"
