import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.parameters import ParamContext, ParameterRegistry
from seqoria_agent.models.llm_models import ToolCallRequest


class _DummyPool:  # minimal stub for ConnectionPool
    async def call_tool(self, *_, **__):
        return [SimpleNamespace(text="<xml>OK</xml>")]

    @property
    def connections(self):
        return []


class _DummyRegistry:  # minimal ToolRegistry stub
    def __init__(self, spp_fields):
        self._spp_fields = spp_fields

    def server_params(self, name: str):  # noqa: D401
        return self._spp_fields.get(name, [])

    def depends_on(self, name: str):  # noqa: D401
        return set()


class Address(BaseModel):
    street: str
    city: str
    country: str


@pytest.mark.asyncio
async def test_basemodel_param_injection():
    async def provide_address():  # pragma: no cover – executed in test
        await asyncio.sleep(0)
        return Address(street="1", city="City", country="X")

    tool_name = "demo_address"
    tc = ToolCallRequest(role="tool_call", id="1", type="function_call", name=tool_name, arguments={})
    reg = ParameterRegistry({Address: provide_address})

    exec_ = Executor(pool=_DummyPool(),
                     registry=_DummyRegistry({tool_name: ["address"]}),
                     param_registry=reg,
                     allow_override=False)

    context: ParamContext = {}
    results = await exec_.run_calls([tc], context=context)
    assert results[0].role == "tool_result" 