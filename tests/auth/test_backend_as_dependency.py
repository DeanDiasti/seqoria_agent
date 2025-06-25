import pytest, httpx, jwt
from fastapi import FastAPI
from seqoria_agent.auth.backends import BearerCookieBackend

SECRET = "dep-secret"
ALGO = "HS256"

backend = BearerCookieBackend(secret=SECRET, algorithm=ALGO)
app = FastAPI()


@app.get("/me")
async def me(user: dict = backend.as_dependency()):  # type: ignore[valid-type]
    return {"id": user.get("id")}


@pytest.mark.asyncio
async def test_backend_as_dependency():
    token = jwt.encode({"id": "charlie"}, SECRET, algorithm=ALGO)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(base_url="http://test", transport=transport) as cli:
        res = await cli.get("/me", cookies={"access_token": token})
        assert res.status_code == 200
        assert res.json()["id"] == "charlie" 