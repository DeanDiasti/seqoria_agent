"""Discovery refresh keeps existing routes usable until replacement is ready."""

import asyncio
from types import SimpleNamespace

import pytest

from seqoria_agent import AsyncAIClient, LocalToolbox
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


@pytest.mark.asyncio
async def test_registry_refresh_keeps_old_routes_live_until_discovery_finishes():
    box = LocalToolbox()

    @box.tool()
    async def ping():
        return "pong"

    async with AsyncAIClient(providers=[EchoProvider()], local_toolbox=box) as client:
        started = asyncio.Event()
        finish = asyncio.Event()

        async def slow_list():
            started.set()
            await finish.wait()
            return SimpleNamespace(tools=[])

        session = SimpleNamespace(list_tools=slow_list)
        client.pool._connections.append(
            SimpleNamespace(session=session, url="fake", _stop_event=None, _owner_task=None)
        )
        refresh = asyncio.create_task(client.refresh_tools())
        await asyncio.wait_for(started.wait(), timeout=1)
        try:
            assert (await client.run_tool_calls([_tool_request("ping")]))[0].result == "pong"
        finally:
            finish.set()
            await refresh
            client.pool._connections.clear()
