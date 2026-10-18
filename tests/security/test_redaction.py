"""Credentials stay out of diagnostics, errors and model-visible failure history."""

import logging
import traceback
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from seqoria_agent._redaction import redact_text, redact_url
from seqoria_agent.client.ai_client.callbacks import on_error
from seqoria_agent.client.ai_client.connection_pool import MCPConnectionPool
from seqoria_agent.client.ai_client.executor import Executor
from seqoria_agent.client.ai_client.tool_registry import ToolRegistry
from seqoria_agent.client.mcp_transport import MCPServerConfig, MCPTransportFactory
from seqoria_agent.client.parameters import ParameterRegistry
from seqoria_agent.models import ToolCallRequest
from seqoria_agent.models.errors import CircuitOpenError, ToolRemoteError

URL = "https://diagnostic-user:private-password@mcp.test:8443/mcp?api_key=private-key&custom=private-custom#private-fragment"
SECRETS = (
    "diagnostic-user",
    "private-password",
    "private-key",
    "private-custom",
    "private-fragment",
    "private-bearer",
)


def assert_safe(text):
    for secret in SECRETS:
        assert secret not in text
    assert "mcp.test:8443/mcp" in text


@pytest.mark.parametrize(
    "url", [URL, URL.replace("https://", "sses://"), URL.replace("https://", "sse://")]
)
def test_url_diagnostics_preserve_address_but_hide_credentials(url):
    safe = redact_url(url)
    assert_safe(safe)
    assert "api_key=" in safe and "custom=" in safe
    assert redact_url(safe) == safe


@pytest.mark.parametrize(
    "text",
    [
        f"HTTP 401 requesting '{URL}'",
        f"HTTP 403 {URL} headers={{'Authorization': 'Bearer private-bearer'}}",
        f"Rejected {URL}; x-api-key=private-key, password=private-password",
        f'Rejected {URL}; password="private-password and private-key"',
        f"Rejected {URL}; Authorization: Custom private-bearer",
    ],
)
def test_exception_text_redacts_urls_and_authorization(text):
    safe = redact_text(text)
    assert_safe(safe)
    assert "HTTP" in safe or "Rejected" in safe


def test_malformed_url_does_not_fall_back_to_secret_text():
    assert redact_url("https://[invalid?api_key=private-key") == "[REDACTED URL]"


@pytest.mark.asyncio
async def test_success_and_discovery_failure_logs_hide_credentials(monkeypatch, caplog):
    observed = []

    @asynccontextmanager
    async def connect(config, **kwargs):
        observed.append(config.url)
        yield SimpleNamespace(session=session, transport_type="http")

    session = AsyncMock()
    session.list_tools.side_effect = RuntimeError(f"HTTP 403 at {URL}")
    monkeypatch.setattr(MCPTransportFactory, "connect", connect)
    caplog.set_level(logging.INFO, logger="seqoria_agent")
    async with MCPConnectionPool([URL]) as pool:
        registry = ToolRegistry(pool=pool)
        await registry.refresh()
        assert registry.llm_tools == []
        assert pool.list_server_urls() == [URL]
    assert observed == [URL]
    assert "Connected to MCP server" in caplog.text
    assert "Failed to list tools" in caplog.text
    assert_safe(caplog.text)


@pytest.mark.asyncio
async def test_remote_tool_errors_and_error_callback_are_safe():
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        isError=True,
        content=[
            TextContent(
                type="text", text=f"HTTP 401 at {URL}; Authorization: Bearer private-bearer"
            )
        ],
    )
    pool = MCPConnectionPool(tool_map={"remote": session})
    registry = ToolRegistry(pool=pool)
    callbacks = []
    executor = Executor(pool=pool, registry=registry, on_error=callbacks.append)
    results = await executor.run_calls([ToolCallRequest(id="one", name="remote", arguments={})])
    error = results[0]
    assert error.role == "tool_error"
    assert "HTTP 401" in error.error
    assert_safe(error.error)
    assert_safe(callbacks[0]["message"])
    assert_safe(error.model_dump_json())
    session.call_tool.assert_awaited_once_with("remote", {})


@pytest.mark.asyncio
async def test_transport_retry_and_breaker_diagnostics_are_safe(caplog):
    session = AsyncMock()
    session.call_tool.side_effect = RuntimeError(f"HTTP 503 at {URL}")
    pool = MCPConnectionPool(
        tool_map={"remote": session},
        retry_policy={"max_attempts": 2, "backoff_base": 0},
        breaker_threshold=1,
    )
    pool._session_to_url[session] = URL
    caplog.set_level(logging.DEBUG, logger="seqoria_agent")
    with pytest.raises(ToolRemoteError) as failure:
        await pool.call_tool("remote", {})
    assert_safe("".join(traceback.format_exception(failure.value)))
    with pytest.raises(CircuitOpenError) as opened:
        await pool.call_tool("remote", {})
    assert_safe(str(opened.value))
    assert_safe(caplog.text)
    assert session.call_tool.await_count == 2


@pytest.mark.asyncio
async def test_direct_transport_errors_do_not_chain_original_credentials(monkeypatch):

    @asynccontextmanager
    async def fail(*args, **kwargs):
        raise RuntimeError(f"HTTP 401 at {URL}")
        yield

    monkeypatch.setattr(MCPTransportFactory, "_connect_http", fail)
    with pytest.raises(RuntimeError) as failure:
        async with MCPTransportFactory.connect(MCPServerConfig(url=URL)):
            pytest.fail("connection should not be established")
    assert_safe("".join(traceback.format_exception(failure.value)))


@pytest.mark.asyncio
async def test_application_callback_and_parameter_failures_are_redacted(caplog):

    def fail(*args):
        raise ValueError(f"HTTP 401 at {URL}; Authorization: Bearer private-bearer")

    caplog.set_level(logging.WARNING, logger="seqoria_agent")
    assert await ParameterRegistry({"user_id": fail}).resolve("user_id", context={}) is None
    on_error({"message": "safe"}, fail)
    assert "failed to resolve" in caplog.text
    assert "Callback raised" in caplog.text
    assert_safe(caplog.text)


@pytest.mark.asyncio
async def test_successful_tool_payload_is_not_rewritten():
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text=URL)])
    pool = MCPConnectionPool(tool_map={"remote": session})
    result = await pool.call_tool("remote", {})
    assert result[0].text == URL


@pytest.mark.asyncio
async def test_provider_fallback_logs_and_final_error_are_redacted(caplog):
    from seqoria_agent.client.ai_client.llm_interface import LLMInterface
    from seqoria_agent.providers.errors import LLMError

    provider = SimpleNamespace(async_generate=AsyncMock(side_effect=LLMError(f"HTTP 401 at {URL}")))
    caplog.set_level(logging.WARNING, logger="seqoria_agent")
    with pytest.raises(RuntimeError) as failure:
        await LLMInterface([provider]).generate([])
    assert_safe(str(failure.value))
    assert_safe(caplog.text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "emitted, error_type", [(False, RuntimeError), (False, ValueError), (True, RuntimeError)]
)
async def test_provider_stream_failures_are_redacted(emitted, error_type):
    from seqoria_agent.client.ai_client.llm_interface import LLMInterface
    from seqoria_agent.providers.schemas import LLMResponse

    async def stream(**kwargs):
        if emitted:
            yield LLMResponse(type="llm_response", model_id="offline", text="safe")
        raise error_type(f"HTTP 401 at {URL}")

    provider = SimpleNamespace(async_stream=stream)
    with pytest.raises(RuntimeError) as failure:
        async for _ in LLMInterface([provider]).stream([]):
            pass
    assert_safe("".join(traceback.format_exception(failure.value)))
