from fastapi.testclient import TestClient

from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.auth.backends import BaseMiddlewareAuthBackend


class DummyMiddlewareBackend(BaseMiddlewareAuthBackend):
    """Backend that counts *class-level* authenticate() invocations."""

    calls = 0

    def __init__(self, *args, **kwargs):
        # Optional ASGI app positional arg when mounted via FastAPI
        if args and not isinstance(args[0], str):
            app, *rest = args
            super().__init__(app)  # type: ignore[misc]

    async def authenticate(self, request):  # noqa: D401
        type(self).calls += 1  # increment class counter so all instances share
        return {"id": "bob"}


def _rpc_msg(method: str, params: dict | None = None):
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}


def test_middleware_backend_called_once():
    backend = DummyMiddlewareBackend()
    app = FastXMCP(title="MiddlewareAuthTest", auth_backend=backend)

    @app.tool(server_populated=["principal"])
    async def whoami(principal):  # noqa: ANN001
        return principal["id"] if isinstance(principal, dict) else "unknown"

    client = TestClient(app)
    payload = _rpc_msg("tools/call", {"name": "whoami", "arguments": {}})
    res = client.post("/mcp/", json=payload)
    assert res.status_code == 200
    assert "bob" in res.text
    assert DummyMiddlewareBackend.calls == 1 