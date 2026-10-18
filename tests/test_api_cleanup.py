"""Regressions for the simplified provider, parameter, and tool-result APIs."""

import asyncio
import importlib.util
import json

import pytest

import seqoria_agent
from seqoria_agent import AIClient, AsyncAIClient, LocalToolbox, providers
from seqoria_agent.client.parameters import ParameterRegistry
from seqoria_agent.client.settings import ClientSettings
from seqoria_agent.models import ToolCallError, ToolCallRequest, ToolCallResult
from seqoria_agent.providers import AsyncBaseProvider


class UnusedProvider(AsyncBaseProvider):
    async def async_generate(self, **kwargs):
        raise AssertionError("Tools should execute without model calls")

    async def async_stream(self, **kwargs):
        yield await self.async_generate(**kwargs)


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
    param_registry = ParameterRegistry(mapping) if source == "registry" else mapping if source == "mapping" else None
    async with AsyncAIClient(providers=[UnusedProvider()], local_toolbox=box, param_registry=param_registry) as client:
        async def invoke(user_id):
            context = {"user_id": user_id} if source == "context" else {"principal": {"user_id": user_id}}
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
        result = (await client.run_tool_calls([
            ToolCallRequest(id="missing", name="identity", arguments={"user_id": "forged"}),
        ]))[0]
        assert isinstance(result, ToolCallError)
        assert "Missing server-populated" in result.error
    with pytest.raises(RuntimeError, match="Missing server-populated"):
        await box.call_tool("identity", {})
    assert invoked == []


@pytest.mark.asyncio
async def test_request_context_precedes_registry_without_mutating_defaults():
    registry = ParameterRegistry({"user_id": "default"})
    assert await registry.resolve("user_id", context={"user_id": "request"}) == "request"
    assert await registry.resolve("user_id", context={}) == "default"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"ai_message": "visible", "metadata": {"count": 2}, "records": [1, 2]},
    {"ai_message": {"nested": True}, "other": "preserved"},
    {"ai_message": None, "metadata": {}},
    '<result><error code="TEST">ordinary tool text</error></result>',
    '{"ai_message": "unterminated',
])
async def test_tool_output_preserves_ordinary_json_and_xml_text(payload):
    box = LocalToolbox()

    @box.tool()
    async def output():
        return payload

    direct = (await box.call_tool("output", {}))[0].text
    async with AsyncAIClient(providers=[UnusedProvider()], local_toolbox=box) as client:
        result = (await client.run_tool_calls([ToolCallRequest(id="output", name="output")]))[0]
    assert isinstance(result, ToolCallResult)
    assert result.result == direct
    if isinstance(payload, dict):
        assert json.loads(result.result) == payload
    else:
        assert result.result == payload


def test_removed_local_adapters_are_absent_from_package():
    for name in ("mlx_lm", "mlx_vlm"):
        assert importlib.util.find_spec(f"seqoria_agent.providers.{name}_provider") is None
        with pytest.raises(ValueError, match="Unknown provider"):
            providers.get_provider(name)
    assert set(providers.list_available_providers()) == {"openai", "openai_compatible", "anthropic", "ollama"}
    assert not any("MLX" in name for name in seqoria_agent.__all__ + providers.__all__)


@pytest.mark.parametrize("option,value", [("urls", []), ("client_kwargs", {"timeout": 5})])
def test_obsolete_settings_and_constructor_options_fail_fast(option, value):
    with pytest.raises(TypeError):
        ClientSettings(**{option: value})
    for client_type in (AsyncAIClient, AIClient):
        with pytest.raises(TypeError, match="Unknown client options"):
            client_type(providers=[UnusedProvider()], **{option: value})


@pytest.mark.asyncio
async def test_settings_use_mcp_servers_and_explicit_empty_override():
    from seqoria_agent.client.mcp_transport import MCPServerConfig

    settings = ClientSettings(providers=[UnusedProvider()], mcp_servers=[MCPServerConfig("http://unused.test/mcp")])
    configured = AsyncAIClient(settings=settings)
    assert configured.list_mcp_servers() == list(settings.mcp_servers)
    # Empty explicit configuration overrides settings without opening a transport.
    async with AsyncAIClient(settings=settings, mcp_servers=[]) as client:
        assert client.list_mcp_servers() == []
