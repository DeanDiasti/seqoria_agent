import pytest
from fastapi.testclient import TestClient

from seqoria_agent.server.fastxmcp import FastXMCP, FastXMCPRouter
from seqoria_agent.models.errors import XMCPError


@pytest.fixture()
def app_with_router():
    # Base app
    app = FastXMCP(title="Router Test Server")

    # Define a router with some tools ---------------------------------
    math_router = FastXMCPRouter(prefix="/math")

    @math_router.tool()
    def add(a: int, b: int) -> int:  # noqa: D401
        return int(a) + int(b)

    # param provider at router-level
    @math_router.register_param_provider  # type: ignore[arg-type]
    def _provider(req, tool_name: str, param_name: str):  # noqa: ANN001
        if param_name == "secret":
            return "router_injected"
        return None

    @math_router.tool(server_populated=["secret"])
    def reveal_secret(secret: str) -> str:  # noqa: D401
        return f"secret={secret}"

    # Mount router
    app.include_router(math_router)
    return app


def _rpc_msg(method: str, params: dict | None = None, _id: int = 1):
    return {"jsonrpc": "2.0", "id": _id, "method": method, "params": params or {}}


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_router_tools_list(app_with_router):
    client = TestClient(app_with_router)

    r = client.post("/mcp/", json=_rpc_msg("tools/list"))
    assert r.status_code == 200
    body = r.json()
    names = {item["name"] for item in body["result"]["items"]}
    assert {"add", "reveal_secret"}.issubset(names)


def test_router_tool_call(app_with_router):
    client = TestClient(app_with_router)

    # Call add tool
    payload = _rpc_msg("tools/call", {"name": "add", "arguments": {"a": 3, "b": 4}})
    res = client.post("/mcp/", json=payload)
    assert res.status_code == 200
    xml = res.json()["result"]["contents"][0]["text"]
    assert "7" in xml  # Expected result embedded in XML

    # Call reveal_secret (uses router param provider)
    payload2 = _rpc_msg("tools/call", {"name": "reveal_secret", "arguments": {}})
    res2 = client.post("/mcp/", json=payload2)
    assert res2.status_code == 200
    xml2 = res2.json()["result"]["contents"][0]["text"]
    assert "router_injected" in xml2


def test_duplicate_tool_error():
    app = FastXMCP()

    @app.tool()
    def foo():
        return "bar"

    router = FastXMCPRouter()

    @router.tool()
    def foo():  # type: ignore[func-override]
        return "router bar"

    with pytest.raises(XMCPError):
        app.include_router(router) 