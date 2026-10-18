"""Real stdio transport can be opened and closed from separate request tasks."""

import asyncio
import shlex
import sys
from pathlib import Path

import pytest

from seqoria_agent import AsyncAIClient
from seqoria_agent.client.mcp_transport import MCPServerConfig
from seqoria_agent.models.chunk import ToolCallRequest
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


def _tool_request(name, call_id=None, **arguments):
    return ToolCallRequest(id=call_id or name, name=name, arguments=arguments)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_stdio_transport_can_close_from_a_different_request_task(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(
        'from mcp.server.fastmcp import FastMCP\nserver = FastMCP("stdio")\n@server.tool()\nasync def ping() -> str:\n    return "pong"\nserver.run(transport="stdio")\n'
    )
    config = MCPServerConfig(
        url="stdio://" + shlex.join([sys.executable, str(script)]),
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )
    client = AsyncAIClient(providers=[EchoProvider()], mcp_servers=[config])
    await asyncio.wait_for(asyncio.create_task(client.__aenter__()), timeout=10)
    try:
        assert (await client.run_tool_calls([_tool_request("ping")]))[0].result == "pong"
    finally:
        await asyncio.wait_for(asyncio.create_task(client.close()), timeout=10)
    assert not client.pool.connections
    assert client.pool._http_client_pool is None
