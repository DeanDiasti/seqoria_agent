"""Conversation histories and system prompts remain application-owned."""

import asyncio

import pytest

from seqoria_agent import AsyncAIClient
from seqoria_agent.providers import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse


class _HistoryProvider(AsyncBaseProvider):
    """Minimal provider that echoes a static answer."""

    async def async_generate(self, *, chat_history, tools=None, tool_choice="auto", model=None):
        return LLMResponse(model_id="dummy", type="llm_response", text="ok")

    async def async_stream(self, *args, **kwargs):
        yield LLMResponse(model_id="test", type="llm_delta", text="delta")
        yield LLMResponse(model_id="dummy", type="llm_response", text="ok")


@pytest.mark.asyncio
async def test_default_system_prompt_inserted():
    history: list[dict] = []
    async with AsyncAIClient(
        mcp_servers=[], providers=[_HistoryProvider()], local_toolbox=None
    ) as ai:
        await ai.llm_generate_once(history=history, use_tools=False)
        assert history[0]["role"] == "system"
        assert history[0]["content"]


@pytest.mark.asyncio
async def test_custom_system_prompt():
    custom = "You are a unit-test assistant."
    history: list[dict] = []
    async with AsyncAIClient(
        mcp_servers=[], providers=[_HistoryProvider()], local_toolbox=None, system_prompt=custom
    ) as ai:
        await ai.llm_generate_once(history=history, use_tools=False)
        assert history[0]["content"] == custom


@pytest.mark.asyncio
async def test_existing_system_prompt_not_duplicated():
    existing = {"role": "system", "content": "Pre inserted"}
    history = [existing]
    async with AsyncAIClient(
        mcp_servers=[], providers=[_HistoryProvider()], local_toolbox=None
    ) as ai:
        await ai.llm_generate_once(history=history, use_tools=False)
        assert history[0] is existing
        assert len([msg for msg in history if msg["role"] == "system"]) == 1


class EchoProvider(AsyncBaseProvider):
    async def async_generate(self, *, chat_history, **kwargs):
        await asyncio.sleep(0)
        text = chat_history[-1].content
        return LLMResponse(model_id="echo", text=text, type="llm_response")

    async def async_stream(self, **kwargs):
        yield (await self.async_generate(**kwargs))


@pytest.mark.asyncio
async def test_shared_client_keeps_concurrent_histories_separate():
    async with AsyncAIClient(providers=[EchoProvider()]) as client:

        async def chat(text):
            history = []
            replies = [item async for item in client.chat(text, history)]
            return (history, replies[-1].content)

        (left, left_reply), (right, right_reply) = await asyncio.gather(chat("alice"), chat("bob"))
        assert left_reply == "alice"
        assert right_reply == "bob"
        assert left is not right
        assert len(left) == len(right) == 3
