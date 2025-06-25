import pytest

from fastapi import HTTPException, Request
from seqoria_agent.auth.backends import BaseAuthBackend
from seqoria_agent.auth.providers import BaseAuthProvider
from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.client import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse

# ---------------------------------------------------------------------------
# Helpers – minimal stubs for LLM provider & Auth components
# ---------------------------------------------------------------------------

class DummyProvider(BaseProvider):
    def __init__(self):
        pass

    async def async_generate(self, *, chat_history, **kw):
        return LLMResponse(model_id="dummy", text="hi")

    async def async_stream(self, *_, **__):  # pragma: no cover
        async def _g():
            yield "hi"
            yield LLMResponse(model_id="dummy", text="hi")

        return _g()


class RequireHeaderBackend(BaseAuthBackend):
    def __init__(self, header_name: str, token: str):
        self._name = header_name
        self._token = token

    async def authenticate(self, request: Request):  # type: ignore[override]
        # Allow the *tools/list* call unauthenticated so the client discovers
        # available tools.  Enforce header presence for *tools/call* requests.
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        if payload.get("method") == "tools/call":
            if request.headers.get(self._name) != self._token:
                raise HTTPException(status_code=401, detail="missing or invalid token")

        return {"id": "tester"}


class StaticHeaderProvider(BaseAuthProvider):
    def __init__(self, header_name: str, token: str):
        self._h = header_name
        self._t = token

    async def auth_headers(self):  # type: ignore[override]
        return {self._h: self._t}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HEADER_NAME = "X-Test-Auth"
TOKEN_VALUE = "secret-token"


@pytest.fixture()
def app():
    app = FastXMCP(auth_backend=RequireHeaderBackend(HEADER_NAME, TOKEN_VALUE))

    @app.tool()
    def ping() -> str:  # noqa: D401
        return "pong"

    return app


@pytest.mark.asyncio
async def test_auth_headers_injected(app):
    # Base URL for in-process httpx client
    base_url = "http://testserver/mcp/"

    # ------------------------------------------------------------------
    # 1) NoAuthProvider – expect 401 on tool call
    # ------------------------------------------------------------------
    async with AIClient(
        providers=[DummyProvider()],
        urls=[base_url],
        app=app,
    ) as client:
        # Refresh toolbox to establish connection; subsequent call should fail
        import httpx
        with pytest.raises(Exception) as exc:
            await client.call_mcp_tool("ping", {})
        assert "401" in str(exc.value)

    # ------------------------------------------------------------------
    # 2) StaticHeaderProvider – expect success
    # ------------------------------------------------------------------
    async with AIClient(
        providers=[DummyProvider()],
        urls=[base_url],
        app=app,
        auth_provider=StaticHeaderProvider(HEADER_NAME, TOKEN_VALUE),
    ) as client:
        result = await client.call_mcp_tool("ping", {})
        assert "pong" in str(result[0].text) 