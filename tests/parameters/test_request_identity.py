"""Concurrent requests retain trusted identities and reject forged hidden arguments."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ToolAnnotations

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.client.mcp_transport import MCPTransportFactory
from seqoria_agent.client.parameters import ParameterRegistry
from seqoria_agent.models import ToolCallError, ToolCallRequest, ToolCallResult
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class UnusedProvider(AsyncBaseProvider):
    async def async_generate(self, **kwargs):
        raise AssertionError("Tools should execute without model calls")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["context", "mapping", "registry"])
async def test_local_hidden_parameters_have_one_resolution_path(source):
    box = LocalToolbox()

    @box.tool(server_populated=["user_id"])
    async def identity(user_id: str) -> str:
        await asyncio.sleep(0)
        return user_id

    async def resolve_user(principal):
        await asyncio.sleep(0)
        return principal["user_id"]

    mapping = {"user_id": resolve_user}
    param_registry = (
        ParameterRegistry(mapping)
        if source == "registry"
        else mapping
        if source == "mapping"
        else None
    )
    async with AsyncAIClient(
        providers=[UnusedProvider()], local_toolbox=box, param_registry=param_registry
    ) as client:

        async def invoke(user_id):
            context = (
                {"user_id": user_id} if source == "context" else {"principal": {"user_id": user_id}}
            )
            call = ToolCallRequest(id=user_id, name="identity", arguments={"user_id": "forged"})
            result = (await client.run_tool_calls([call], context=context))[0]
            assert isinstance(result, ToolCallResult)
            assert call.arguments == {"user_id": "forged"}
            direct = await box.call_tool("identity", {}, server_args={"user_id": user_id})
            assert direct[0].text == result.result == user_id
            return result.result

        assert await asyncio.gather(invoke("alice"), invoke("bob")) == ["alice", "bob"]
        schema = client.list_tools()[0]["function"]["parameters"]
        assert "user_id" not in schema.get("properties", {})


@pytest.mark.asyncio
async def test_missing_hidden_identity_never_runs_tool():
    box = LocalToolbox()
    invoked = []

    @box.tool(server_populated=["user_id"])
    async def identity(user_id: str) -> str:
        invoked.append(user_id)
        return user_id

    async with AsyncAIClient(providers=[UnusedProvider()], local_toolbox=box) as client:
        result = (
            await client.run_tool_calls(
                [ToolCallRequest(id="missing", name="identity", arguments={"user_id": "forged"})]
            )
        )[0]
        assert isinstance(result, ToolCallError)
        assert "Missing server-populated" in result.error
    with pytest.raises(RuntimeError, match="Missing server-populated"):
        await box.call_tool("identity", {})
    assert invoked == []


class ReplyProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(model_id="test", text="done", thinking="reason", type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


def request(name, **arguments):
    return ToolCallRequest(id=name, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_unresolved_hidden_identity_is_an_isolated_tool_failure():
    box = LocalToolbox()
    invoked = []

    @box.tool(server_populated=["user_id"])
    async def protected(user_id: str):
        invoked.append(user_id)
        return user_id

    @box.tool(depends_on=["protected"])
    async def dependent():
        invoked.append("dependent")
        return "must not run"

    @box.tool()
    async def independent():
        await asyncio.sleep(0)
        return "ok"

    async with AsyncAIClient(providers=[ReplyProvider()], local_toolbox=box) as client:
        calls = [
            request("protected", user_id="forged-secret"),
            request("independent"),
            request("dependent"),
        ]
        results = await client.run_tool_calls(calls)
    assert isinstance(results[0], ToolCallError)
    assert "Missing server-populated parameters" in results[0].error
    assert "forged-secret" not in results[0].error
    assert results[1].result == "ok"
    assert isinstance(results[2], ToolCallError)
    assert "Skipped" in results[2].error
    assert invoked == []
    assert calls[0].arguments == {"user_id": "forged-secret"}


class OfflineProvider(AsyncBaseProvider):
    async def async_generate(self, **kwargs):
        return LLMResponse(type="llm_response", model_id="offline", text="unused")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remote_identity_annotations_do_not_inject_client_context(monkeypatch):
    server = FastMCP("independent")

    @server.tool(annotations=ToolAnnotations(server_populated=["user_id"]))
    async def identity() -> str:
        return "server-authenticated-user"

    @asynccontextmanager
    async def connect(*args, **kwargs):
        async with create_connected_server_and_client_session(server) as session:
            yield SimpleNamespace(session=session, transport_type="http")

    monkeypatch.setattr(MCPTransportFactory, "connect", connect)
    async with AsyncAIClient(
        providers=[OfflineProvider()], mcp_servers=["http://independent.test/mcp"]
    ) as client:
        result = await client.run_tool_calls(
            [ToolCallRequest(id="one", name="identity", arguments={})],
            context={"user_id": "client-identity"},
        )
    assert result[0].role == "tool_result"
    assert result[0].result == "server-authenticated-user"


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


def _tool_request(name, call_id=None, **arguments):
    return ToolCallRequest(id=call_id or name, name=name, arguments=arguments)


@pytest.mark.asyncio
async def test_injection_does_not_rewrite_the_original_tool_request():
    box = LocalToolbox()

    @box.tool(server_populated=["user_id"])
    async def identity(user_id: str):
        return user_id

    async with AsyncAIClient(providers=[EchoProvider()], local_toolbox=box) as client:
        call = _tool_request("identity", user_id="forged")
        result = await client.run_tool_calls([call], context={"user_id": "trusted"})
        assert result[0].result == "trusted"
        assert call.arguments == {"user_id": "forged"}


@pytest.mark.asyncio
async def test_repeated_tool_calls_execute_with_current_request_identity():
    box = LocalToolbox()
    invoked = []

    @box.tool(server_populated=["user_id"])
    async def counter(user_id: str) -> str:
        invoked.append(user_id)
        return f"{user_id}:{len(invoked)}"

    async with AsyncAIClient(providers=[EchoProvider()], local_toolbox=box) as client:
        call = _tool_request("counter")
        results = []
        for identity in ["alice", "alice", "bob", "alice"]:
            result = await client.run_tool_calls([call], context={"user_id": identity})
            results.append(result[0].result)
    assert results == ["alice:1", "alice:2", "bob:3", "alice:4"]
    assert invoked == ["alice", "alice", "bob", "alice"]
