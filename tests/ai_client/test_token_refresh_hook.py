import pytest
from seqoria_agent.client.simple_http_transport import SimpleHTTPTransport
from seqoria_agent.auth.providers import BaseAuthProvider
from seqoria_agent.server.fastxmcp import FastXMCP


class _DummyProvider(BaseAuthProvider):
    def __init__(self):
        self.calls = 0

    async def refresh_token_if_needed(self):  # noqa: D401
        self.calls += 1

    async def auth_headers(self):  # noqa: D401
        return {}


@pytest.mark.asyncio
async def test_token_refresh_hook_invoked():
    app = FastXMCP()

    # minimal tool to satisfy list_tools
    @app.tool
    def foo():
        return "bar"

    prov = _DummyProvider()
    async with SimpleHTTPTransport(url="http://test/mcp/", app=app, base_url="http://test", auth_provider=prov) as t:
        await t.list_tools()
        await t.list_tools()

    # refresh should be called exactly twice (once per request)
    assert prov.calls == 2 