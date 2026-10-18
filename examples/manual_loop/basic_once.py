"""Run a single offline LLM step: python examples/manual_loop/basic_once.py."""

import asyncio

from seqoria_agent import AsyncAIClient
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage


class StubProvider(AsyncBaseProvider):
    """Offline provider used to demonstrate orchestration without API calls."""

    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(
            model_id="offline",
            type="llm_response",
            text="pong",
            usage=LLMUsage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )

    async def async_stream(self, *, chat_history, **kwargs):
        yield LLMResponse(model_id="offline", type="llm_delta", text="pong")
        yield await self.async_generate(chat_history=chat_history)


async def main() -> None:
    async with StubProvider() as provider, AsyncAIClient(providers=[provider]) as client:
        history = [{"role": "user", "content": "ping"}]
        response = await client.llm_generate_once(history=history)
        print("AI replied:", response.content)


if __name__ == "__main__":
    asyncio.run(main())
