import pytest, httpx, jwt
from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.auth.backends import BearerCookieBackend

SECRET = "test-secret"
ALGO = "HS256"


def make_app():
    app = FastXMCP(auth_backend=BearerCookieBackend(secret=SECRET, algorithm=ALGO))

    @app.tool(server_populated=["user_id"])
    async def echo(user_id: str):
        return {"user": user_id}

    return app


@pytest.mark.asyncio
async def test_jwt_cookie_backend_matrix():
    app = make_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(base_url="http://test", transport=transport) as cli:
        # Unauthenticated should return 401
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

        # Invalid token should return 401
        bad_cookie = {"access_token": "invalid"}
        res = await cli.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {}},
            },
            cookies=bad_cookie,
        )
        assert res.status_code == 401

        # Valid token should return 200 and inject user_id
        token = jwt.encode({"id": "alice"}, SECRET, algorithm=ALGO)
        res = await cli.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {}},
            },
            cookies={"access_token": token},
        )
        assert res.status_code == 200
        assert "alice" in res.text 