import types
import pytest

from seqoria_agent.models.errors import XMCPError
from seqoria_agent.server.xmcp_core.models import ToolInfo
from seqoria_agent.server.xmcp_core.param_injection import ParamInjector


async def _dummy_provider(req, tool_name: str, param_name: str):  # noqa: D401
    if param_name == "ts":
        return 42
    return None


@pytest.mark.asyncio
async def test_missing_server_param_raises():
    info = ToolInfo(func=lambda: None, server_params=["user_id"])
    injector = ParamInjector(providers=[], strict_mode=True)
    request_stub = types.SimpleNamespace(state=types.SimpleNamespace(principal=None))
    with pytest.raises(XMCPError) as exc:
        await injector.inject(request_stub, tool_name="foo", info=info, arguments={})
    assert exc.value.code == "MISSING_SERVER_PARAM"


@pytest.mark.asyncio
async def test_forbidden_override():
    info = ToolInfo(func=lambda: None, server_params=["principal"])
    injector = ParamInjector(providers=[], strict_mode=True)
    request_stub = types.SimpleNamespace(state=types.SimpleNamespace(principal=None))
    with pytest.raises(XMCPError) as exc:
        await injector.inject(request_stub, tool_name="foo", info=info, arguments={"principal": "malicious"})
    assert exc.value.code == "FORBIDDEN_PARAM_OVERRIDE"


@pytest.mark.asyncio
async def test_successful_injection():
    info = ToolInfo(func=lambda: None, server_params=["user_id", "ts"])
    injector = ParamInjector(providers=[_dummy_provider], strict_mode=True)
    principal_obj = types.SimpleNamespace(id=123)
    request_stub = types.SimpleNamespace(state=types.SimpleNamespace(principal=principal_obj))
    out = await injector.inject(request_stub, tool_name="foo", info=info, arguments={})
    assert out["user_id"] == 123
    assert out["ts"] == 42 