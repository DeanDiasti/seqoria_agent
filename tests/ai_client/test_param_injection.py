from types import SimpleNamespace
from typing import Mapping

import pytest

from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.parameters import ParamContext, ParameterSource
from seqoria_agent.models.chunk import ToolCallRequest


class _DummyPool:  # minimal stub for ConnectionPool
    async def call_tool(self, *_, **__):
        # Mimic successful XML wrapper result used by Executor
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


@pytest.mark.asyncio
async def test_constant_param_injection():
    tool_name = "demo"
    tc = ToolCallRequest(role="tool_call", id="1", type="function_call", name=tool_name, arguments={})
    reg_map: Mapping[str, ParameterSource] = {"user_id": "u-42"}
    exec_ = Executor(pool=_DummyPool(),
                     registry=_DummyRegistry({tool_name: ["user_id"]}),
                     param_registry=reg_map,
                     allow_override=False)

    context: ParamContext = {"history": []}
    results = await exec_.run_calls([tc], context=context)
    assert results[0].role == "tool_result"


@pytest.mark.asyncio
async def test_callable_param_injection():
    async def user_callable(history):  # pragma: no cover – executed in test
        return "async-user"

    tool_name = "demo2"
    tc = ToolCallRequest(role="tool_call", id="1", type="function_call", name=tool_name, arguments={})
    reg_map = {"user_id": user_callable}
    exec_ = Executor(pool=_DummyPool(),
                     registry=_DummyRegistry({tool_name: ["user_id"]}),
                     param_registry=reg_map,
                     allow_override=False)
    context: ParamContext = {"history": []}
    results = await exec_.run_calls([tc], context=context)
    assert results[0].role == "tool_result"


@pytest.mark.asyncio
async def test_missing_param_raises():
    tool_name = "demo3"
    tc = ToolCallRequest(role="tool_call", id="1", type="function_call", name=tool_name, arguments={})
    exec_ = Executor(pool=_DummyPool(),
                     registry=_DummyRegistry({tool_name: ["tenant_id"]}),
                     param_registry={},
                     allow_override=False)
    context: ParamContext = {"history": []}
    with pytest.raises(RuntimeError):
        await exec_.run_calls([tc], context=context) 