"""Dependency cycles and failed prerequisites prevent invalid tool execution."""

import pytest

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.models.chunk import ToolCallError, ToolCallRequest, ToolCallResult
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class _DependencyProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        raise RuntimeError("LLM call should not be reached in executor tests")

    async def async_stream(self, *args, **kwargs):
        yield LLMResponse(model_id="test", type="llm_delta", text="dummy")


def _get_result(results, name):
    for r in results:
        if r.name == name:
            return r
    raise AssertionError(f"Missing result for {name}")


@pytest.mark.asyncio
async def test_dep_skip_basic():
    tb = LocalToolbox()

    @tb.tool()
    def fail_tool():
        raise RuntimeError("boom")

    @tb.tool(depends_on=["fail_tool"])
    def dep_tool():
        return "should not run"

    @tb.tool()
    def ok_tool():
        return "ok"

    async with AsyncAIClient(
        mcp_servers=[], providers=[_DependencyProvider()], local_toolbox=tb
    ) as client:
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


@pytest.mark.asyncio
async def test_dep_skip_duplicates():
    tb = LocalToolbox()

    @tb.tool()
    def flaky_tool(val: int):
        if val == 0:
            raise RuntimeError("first fails")
        return "ok"

    @tb.tool(depends_on=["flaky_tool"])
    def dep_tool():
        return "dep ok"

    async with AsyncAIClient(
        mcp_servers=[], providers=[_DependencyProvider()], local_toolbox=tb
    ) as client:
        calls = [
            ToolCallRequest(id="1", name="flaky_tool", arguments={"val": 0}),
            ToolCallRequest(id="2", name="flaky_tool", arguments={"val": 1}),
            ToolCallRequest(id="3", name="dep_tool", arguments={}),
        ]
        results = await client.executor.run_calls(calls)
        assert len(results) == 3
        flaky_results = [r for r in results if r.name == "flaky_tool"]
        assert any((isinstance(r, ToolCallError) for r in flaky_results))
        assert any((isinstance(r, ToolCallResult) for r in flaky_results))
        dep_res = _get_result(results, "dep_tool")
        assert isinstance(dep_res, ToolCallResult)


@pytest.mark.asyncio
async def test_dep_skip_transitive():
    tb = LocalToolbox()

    @tb.tool()
    def fail_a():
        raise RuntimeError("boom a")

    @tb.tool(depends_on=["fail_a"])
    def dep_b():
        return "b"

    @tb.tool(depends_on=["dep_b"])
    def dep_c():
        return "c"

    async with AsyncAIClient(
        mcp_servers=[], providers=[_DependencyProvider()], local_toolbox=tb
    ) as client:
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


class ReplyProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(model_id="test", text="done", thinking="reason", type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


def request(name, **arguments):
    return ToolCallRequest(id=name, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_single_call_still_checks_dependency_cycles():
    box = LocalToolbox()
    invoked = []

    @box.tool(depends_on=["loop"])
    async def loop():
        invoked.append(True)
        return "wrong"

    async with AsyncAIClient(providers=[ReplyProvider()], local_toolbox=box) as client:
        result = (await client.run_tool_calls([request("loop")]))[0]
    assert isinstance(result, ToolCallError)
    assert "Cyclic dependency" in result.error
    assert invoked == []
