import asyncio, json, uuid

import pytest

from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.client.client import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCall
from seqoria_agent.client.ai_client.consts_runtime import AI_MESSAGE_KEY


class DummyProvider(BaseProvider):
    def __init__(self):
        pass

    async def async_generate(self, *args, **kwargs):
        return LLMResponse(model_id="dummy", text="done")

    async def async_stream(self, *args, **kwargs):  # pragma: no cover
        yield ""
        yield LLMResponse(model_id="dummy", text="done")


@pytest.mark.asyncio
async def test_json_wrapper_roundtrip():
    box = LocalToolbox(json_wrapper=True)

    @box.tool()
    def add(a: int, b: int) -> int:  # noqa: D401
        return int(a) + int(b)

    async with AIClient(providers=[DummyProvider()], urls=[], local_toolbox=box) as client:
        # Direct call path ---------------------------------------------------
        res = await client.call_mcp_tool(name="add", arguments={"a": 2, "b": 3})
        assert res and hasattr(res[0], "text")
        payload = json.loads(res[0].text)
        assert payload["metadata"]["tool_name"] == "add"
        assert payload[AI_MESSAGE_KEY].startswith("<tool_response")

        # _run_tool parsing path ---------------------------------------------
        tc = ToolCall(id=str(uuid.uuid4()), function={"name": "add", "arguments": json.dumps({"a": 4, "b": 5})})
        _, history_entry, _ = await client._run_tool(tc)
        assert history_entry is not None
        assert history_entry["content"].startswith("<tool_response") 