"""Tests for AIClient.manual hot-reload behaviour."""

import asyncio

import pytest

from seqoria_agent.client.client import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class _DummyProvider(BaseProvider):
    """Minimal provider stub – never actually called in this test."""

    def __init__(self):
        super().__init__(model_id="dummy")
        self.max_retries = 0

    async def async_generate(self, *args, **kwargs):  # noqa: D401, ANN001
        return LLMResponse(model_id="dummy", text="hi")

    async def async_stream(self, *args, **kwargs):  # noqa: D401, ANN001
        yield "hi"
        yield LLMResponse(model_id="dummy", text="hi")


class _TestClient(AIClient):
    """Subclass that overrides network accesses for unit testing."""

    async def _connect_to_server(self, url):  # noqa: D401, ANN001
        # Skip real network; pretend connection failed → None so __aenter__ raises
        return None

    async def _update_llm_tools(self):  # noqa: D401
        # Override: increment counter each time to detect refresh
        self.counter = getattr(self, "counter", 0) + 1
        # Minimal fake tool list
        self._llm_tools = [
            {
                "type": "function",
                "function": {
                    "name": f"dummy{self.counter}",
                    "description": "",
                    "parameters": {},
                },
            }
        ]


@pytest.mark.asyncio
async def test_manual_refresh_tools():
    client = _TestClient(urls=["http://dummy"], providers=[_DummyProvider()])

    # Call refresh_tools twice manually and ensure counter increments
    await client.refresh_tools()
    first_count = client.counter
    await client.refresh_tools()
    assert client.counter == first_count + 1 