import pytest

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.memory import AgenticMemory
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage


class _CaptureProvider(AsyncBaseProvider):
    def __init__(self):
        self.last_history = None

    async def async_generate(self, *, chat_history, **kwargs):  # type: ignore[override]
        self.last_history = chat_history
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return LLMResponse(
            model_id="capture",
            type="llm_response",
            text="ok",
            usage=usage,
            stop_reason="end_turn",
        )

    async def async_stream(self, *, chat_history, **kwargs):  # type: ignore[override]
        self.last_history = chat_history
        usage = LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        yield LLMResponse(
            model_id="capture",
            type="llm_response",
            text="ok",
            usage=usage,
            stop_reason="end_turn",
        )


@pytest.mark.asyncio
async def test_memory_aware_history_prevents_duplicate_system_prompt_injection():
    provider = _CaptureProvider()
    memory = AgenticMemory()
    memory.add_system_message("Memory provided system prompt")
    memory.begin_turn("hello")

    async with AsyncAIClient(
        urls=[],
        providers=[provider],
        system_prompt="Client provided system prompt",
    ) as client:
        await client.llm_generate_once(history=memory, use_tools=False)

    assert provider.last_history is not None
    system_messages = [m for m in provider.last_history if m.get("role") == "system"]
    assert len(system_messages) == 1
    assert "Memory provided system prompt" in system_messages[0]["content"]
    assert "Client provided system prompt" not in system_messages[0]["content"]


@pytest.mark.asyncio
async def test_plain_list_history_keeps_existing_system_prompt_behavior():
    provider = _CaptureProvider()
    history: list[dict] = []

    async with AsyncAIClient(
        urls=[],
        providers=[provider],
        system_prompt="Client provided system prompt",
    ) as client:
        await client.llm_generate_once(history=history, use_tools=False)

    assert history
    assert history[0]["role"] == "system"
    assert history[0]["content"] == "Client provided system prompt"

