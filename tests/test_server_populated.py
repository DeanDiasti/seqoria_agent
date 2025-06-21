import json

import pytest
from fastapi.testclient import TestClient

from seqoria_agent.server.fastxmcp import FastXMCP
from fastapi import Request


@pytest.fixture()
def app():
    app = FastXMCP(title="SPP Test Server")

    # Provider that injects user_id from header
    @app.register_param_provider  # type: ignore[arg-type]
    async def _hdr(req: Request, tool_name: str, param_name: str):  # noqa: ANN401
        if param_name == "user_id":
            return req.headers.get("X-User-Id", "injected")
        return None

    # Tool requiring server-populated param
    @app.tool(server_populated=["user_id"])
    def greet(user_id: str, name: str) -> str:  # noqa: D401
        return f"hi {name} (uid={user_id})"

    return app


def _rpc_msg(method: str, params: dict | None = None, _id: int = 1):
    return {
        "jsonrpc": "2.0",
        "id": _id,
        "method": method,
        "params": params or {},
    }


def test_injection_success(app):
    client = TestClient(app)

    # Call tool without user_id argument – provider should inject
    payload = _rpc_msg(
        "tools/call",
        {
            "name": "greet",
            "arguments": {"name": "Dean"},
            "serverArgs": {"user_id": "injected"},
        },
    )
    r = client.post("/mcp/", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "result" in data and data["error"] is None
    xml = data["result"]["contents"][0]["text"]
    assert "uid=injected" in xml


def test_override_forbidden(app):
    client = TestClient(app)

    # Supply forbidden param – should raise structured error
    payload = _rpc_msg(
        "tools/call",
        {
            "name": "greet",
            "arguments": {"name": "Dean", "user_id": "hacker"},
            "serverArgs": {"user_id": "hacker"},
        },
    )
    r = client.post("/mcp/", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["error"]["code"] == "FORBIDDEN_PARAM_OVERRIDE" 