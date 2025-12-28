import asyncio
import pytest

from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client import AIClient
from seqoria_agent.models.llm_models import ToolCallRequest, ToolCallError, ToolCallResult
from seqoria_agent.llm_providers.base_provider import BaseProvider


class DummyProvider(BaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):  # noqa: ANN001
        raise RuntimeError("LLM call should not be reached in executor tests")

    async def async_stream(self, *args, **kwargs):  # noqa: ANN001
        async def _gen():
            yield "dummy"
        return _gen()


# ---------------------------------------------------------------------------
# Helper – extract first result for a given tool name from a result list
# ---------------------------------------------------------------------------

def _get_result(results, name):  # noqa: D401
    for r in results:
        if r.name == name:
            return r
    raise AssertionError(f"Missing result for {name}")


# ---------------------------------------------------------------------------
# 1) Basic dependency skip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dep_skip_basic():
    tb = LocalToolbox()

    @tb.tool()
    def fail_tool():  # noqa: D401, ANN001 – intentional failure
        raise RuntimeError("boom")

    @tb.tool(depends_on=["fail_tool"])
    def dep_tool():  # noqa: D401
        return "should not run"

    @tb.tool()
    def ok_tool():  # noqa: D401
        return "ok"

    async with AIClient(urls=[], providers=[DummyProvider()], local_toolbox=tb) as client:
        calls = [
            ToolCallRequest(id="1", name="fail_tool", arguments={}),
            ToolCallRequest(id="2", name="ok_tool", arguments={}),
            ToolCallRequest(id="3", name="dep_tool", arguments={}),
        ]

        results = await client.executor.run_calls(calls)
        assert len(results) == 3

        assert isinstance(_get_result(results, "fail_tool"), ToolCallError)
        assert isinstance(_get_result(results, "ok_tool"), ToolCallResult)

        dep_res = _get_result(results, "dep_tool")
        assert isinstance(dep_res, ToolCallError)
        assert "Skipped" in dep_res.error


# ---------------------------------------------------------------------------
# 2) Duplicate calls where at least one succeeds – dependant should run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dep_skip_duplicates():
    tb = LocalToolbox()

    @tb.tool()
    def flaky_tool(val: int):  # noqa: D401, ANN001
        if val == 0:
            raise RuntimeError("first fails")
        return "ok"

    @tb.tool(depends_on=["flaky_tool"])
    def dep_tool():  # noqa: D401
        return "dep ok"

    async with AIClient(urls=[], providers=[DummyProvider()], local_toolbox=tb) as client:
        calls = [
            ToolCallRequest(id="1", name="flaky_tool", arguments={"val": 0}),
            ToolCallRequest(id="2", name="flaky_tool", arguments={"val": 1}),
            ToolCallRequest(id="3", name="dep_tool", arguments={}),
        ]

        results = await client.executor.run_calls(calls)
        assert len(results) == 3

        # Exactly one success & one failure for flaky_tool
        flaky_results = [r for r in results if r.name == "flaky_tool"]
        assert any(isinstance(r, ToolCallError) for r in flaky_results)
        assert any(isinstance(r, ToolCallResult) for r in flaky_results)

        # Dependant should have run successfully
        dep_res = _get_result(results, "dep_tool")
        assert isinstance(dep_res, ToolCallResult)


# ---------------------------------------------------------------------------
# 3) Transitive skip (A fails -> B skipped -> C skipped)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dep_skip_transitive():
    tb = LocalToolbox()

    @tb.tool()
    def fail_a():  # noqa: D401, ANN001
        raise RuntimeError("boom a")

    @tb.tool(depends_on=["fail_a"])
    def dep_b():  # noqa: D401
        return "b"

    @tb.tool(depends_on=["dep_b"])
    def dep_c():  # noqa: D401
        return "c"

    async with AIClient(urls=[], providers=[DummyProvider()], local_toolbox=tb) as client:
        calls = [
            ToolCallRequest(id="1", name="fail_a", arguments={}),
            ToolCallRequest(id="2", name="dep_b", arguments={}),
            ToolCallRequest(id="3", name="dep_c", arguments={}),
        ]

        results = await client.executor.run_calls(calls)
        assert len(results) == 3

        assert isinstance(_get_result(results, "fail_a"), ToolCallError)
        assert isinstance(_get_result(results, "dep_b"), ToolCallError)
        assert isinstance(_get_result(results, "dep_c"), ToolCallError)
        assert "Skipped" in _get_result(results, "dep_b").error
        assert "Skipped" in _get_result(results, "dep_c").error 