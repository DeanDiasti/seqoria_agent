from fastapi.testclient import TestClient

from seqoria_agent.server.fastxmcp import FastXMCP


def _rpc_msg(method: str, params: dict | None = None):
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}


def test_noauth_default_tool_executes():
    app = FastXMCP(title="NoAuthTestServer")

    @app.tool()
    async def echo(msg: str):
        return msg

    client = TestClient(app)
    payload = _rpc_msg("tools/call", {"name": "echo", "arguments": {"msg": "hello"}})
    res = client.post("/mcp/", json=payload)
    assert res.status_code == 200
    assert "hello" in res.text 