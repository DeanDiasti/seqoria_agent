from typing import List, Dict, Any, Optional, AsyncGenerator, Union

import pytest

from seqoria_agent.client.settings import ClientSettings
from seqoria_agent.client.ai_client.base import AIClient
from seqoria_agent.providers.base_provider import BaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class DummyProvider(BaseProvider):
    """Trivial provider that returns static responses synchronously for tests."""

    def __init__(self, model_id: str = "dummy-model", *_, **__):
        self.model_id = model_id

    async def async_generate(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        **__,
    ) -> LLMResponse:
        return LLMResponse(model_id=self.model_id, text="pong")

    async def async_stream(
        self,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        **__,
    ) -> AsyncGenerator[Union[str, LLMResponse], None]:
        # Yield a single delta then the final response
        yield "pong"
        yield LLMResponse(model_id=self.model_id, text="pong")


@pytest.mark.asyncio
async def test_client_settings_defaults_apply():
    """AIClient should fall back to ClientSettings when explicit kwargs are missing."""
    settings = ClientSettings(urls=["http://example.com"])

    async with AIClient(providers=[DummyProvider()], settings=settings) as ai:
        # The pool should have received our URL via the settings object
        assert ai.pool._urls == ["http://example.com"]


@pytest.mark.asyncio
async def test_client_settings_explicit_overrides():
    """Explicit kwargs must override values provided by ClientSettings."""
    settings = ClientSettings(urls=["http://wrong.com"], refresh_interval=999)

    async with AIClient(
        providers=[DummyProvider()],
        settings=settings,
        urls=["http://right.com"],
        refresh_interval=123,
    ) as ai:
        # Explicit value wins over settings
        assert ai.registry._refresh_interval == 123  # type: ignore[attr-defined]
        assert ai.pool._urls == ["http://right.com"] 