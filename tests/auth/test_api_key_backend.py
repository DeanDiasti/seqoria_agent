import pytest, httpx
from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.auth.backends import HeaderApiKeyBackend

API_HEADER = "X-API-Key"
VALID_KEY = "super-secret-key"


def make_app():
    app = FastXMCP(auth_backend=HeaderApiKeyBackend(key_header=API_HEADER, validator=VALID_KEY))

    @app.tool()
    async def echo():
        return "pong"

    return app


@pytest.mark.asyncio
async def test_api_key_backend_matrix():
    app = make_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(base_url="http://test", transport=transport) as cli:
        # Missing header should return 401
        res = await cli.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {}},
            },
        )
        assert res.status_code == 401

        # Invalid key should return 401
        res = await cli.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {}},
            },
            headers={API_HEADER: "wrong-key"},
        )
        assert res.status_code == 401

        # Valid key should return 200
        res = await cli.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {}},
            },
            headers={API_HEADER: VALID_KEY},
        )
        assert res.status_code == 200
        assert "pong" in res.text 