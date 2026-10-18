"""Show example token usage without API calls: python examples/token_counting/client.py.

The offline provider returns illustrative counts. Hosted providers return the
usage supplied by their model service.
"""

import asyncio

from seqoria_agent import AsyncAIClient
from seqoria_agent.models.chunk import AssistantMessage, ChatChunk
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
        result = await client.llm_generate_once(history=history)
        print("One-shot answer:", result.content)
        if result.usage:
            print("Token usage:", result.usage.model_dump())
        print("Streaming answer:", end=" ")
        async for chunk in client.llm_stream_once(history=history):
            if isinstance(chunk, ChatChunk):
                print(chunk.content, end="", flush=True)
            elif isinstance(chunk, AssistantMessage) and chunk.usage:
                print("\nToken usage:", chunk.usage.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
