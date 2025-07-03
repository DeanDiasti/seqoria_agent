import pytest, httpx, jwt
from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.server.fastxmcp import FastXMCPRouter
from seqoria_agent.auth.backends import BearerCookieBackend

SECRET = "router-secret"
ALGO = "HS256"


@pytest.mark.asyncio
async def test_per_router_auth_override():
    # App-level backend is *no-auth* by default
    app = FastXMCP()

    # Public router (no auth)
    pub_router = FastXMCPRouter()

    @pub_router.tool
    async def ping():
        return "pong"

    # Protected router
    prot_backend = BearerCookieBackend(secret=SECRET, algorithm=ALGO)
    prot_router = FastXMCPRouter(auth_backend=prot_backend)

    @prot_router.tool(server_populated=["user_id"])
    async def secret_tool(user_id: str):
        return {"uid": user_id}

    app.include_router(pub_router)
    app.include_router(prot_router)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
        # Public tool should work without auth
        res = await client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ping", "arguments": {}}},
        )
        assert res.status_code == 200
        assert "pong" in res.text

        # Protected tool fails without cookie
        res = await client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "secret_tool", "arguments": {}}},
        )
        assert res.status_code == 401

        # Protected tool succeeds with valid cookie
        token = jwt.encode({"id": "bob"}, SECRET, algorithm=ALGO)
        client.cookies.set("access_token", token)
        res = await client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "secret_tool", "arguments": {}}},
        )
        assert res.status_code == 200
        assert "bob" in res.text 