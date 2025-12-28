import pytest
from seqoria_agent.server.fastxmcp import FastXMCP
from seqoria_agent.client import AIClient
from seqoria_agent.llm_providers.base_provider import BaseProvider
from seqoria_agent.models.llm_models import ToolCallRequest, ToolCallError, ToolCallResult


class DummyProvider(BaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):  # noqa: ANN001
        raise RuntimeError("LLM should not be called in tool executor tests")

    async def async_stream(self, *args, **kwargs):  # noqa: ANN001
        async def _gen():
            yield "dummy"
        return _gen()


@pytest.fixture()
def remote_app():
    app = FastXMCP(title="DepSkipTest")

    @app.tool()
    def fail_tool():  # noqa: D401, ANN001
        raise RuntimeError("boom")

    @app.tool(depends_on=["fail_tool"])
    def dep_tool():  # noqa: D401
        return "should not run"

    @app.tool()
    def ok_tool():  # noqa: D401
        return "ok"

    return app


@pytest.mark.asyncio
async def test_remote_dep_skip_basic(remote_app):
    async with AIClient(urls=["http://test/mcp/"],
                        providers=[DummyProvider()],
                        client_kwargs={"app": remote_app, "base_url": "http://test"}) as client:
        await client.refresh_tools()  # ensure registry populated

        calls = [
            ToolCallRequest(id="1", name="fail_tool", arguments={}),
            ToolCallRequest(id="2", name="ok_tool", arguments={}),
            ToolCallRequest(id="3", name="dep_tool", arguments={}),
        ]

        results = await client.executor.run_calls(calls)
        by_name = {r.name: r for r in results}

        assert isinstance(by_name["fail_tool"], ToolCallError)
        assert isinstance(by_name["ok_tool"], ToolCallResult)
        assert isinstance(by_name["dep_tool"], ToolCallError)
        assert "Skipped" in by_name["dep_tool"].error 