from fastapi.testclient import TestClient

from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.auth.backends import BaseAuthBackend


class DummyBackend(BaseAuthBackend):
    def __init__(self):
        self.calls = 0

    async def authenticate(self, request):  # noqa: D401
        self.calls += 1
        return {"id": "bob"}


def _rpc_msg(method: str, params: dict | None = None):
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}


def test_backend_called_and_principal_injected():
    backend = DummyBackend()
    app = FastXMCP(title="AuthTestServer", auth_backend=backend)

    @app.tool(server_populated=["principal"])
    async def whoami(principal):  # noqa: ANN001
        return principal["id"] if isinstance(principal, dict) else "unknown"

    client = TestClient(app)
    payload = _rpc_msg("tools/call", {"name": "whoami", "arguments": {}})
    res = client.post("/mcp/", json=payload)
    assert res.status_code == 200
    assert "bob" in res.text
    # Backend should have been called exactly once
    assert backend.calls == 1 