"""Generate and stream with a deterministic provider; no API keys or services."""

import asyncio

from seqoria_agent import AsyncAIClient
from seqoria_agent.models.chunk import AssistantMessage, ChatChunk
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage


class OfflineProvider(AsyncBaseProvider):
    """A test double: its answer and token counts are illustrative."""

    async def async_generate(self, chat_history, **kwargs):
        return LLMResponse(
            model_id="offline",
            type="llm_response",
            text="pong",
            usage=LLMUsage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )

    async def async_stream(self, chat_history, **kwargs):
        yield LLMResponse(model_id="offline", type="llm_delta", text="pong")
        yield await self.async_generate(chat_history=chat_history)


async def main() -> None:
    # The application owns the provider lifetime and the conversation list.
    async with OfflineProvider() as provider, AsyncAIClient(providers=[provider]) as client:
        history = [{"role": "user", "content": "ping"}]
        reply = await client.llm_generate_once(history=history)
        print("One-shot:", reply.content)

        print("Streaming:", end=" ")
        async for chunk in client.llm_stream_once(history=history):
            if isinstance(chunk, ChatChunk):
                print(chunk.content, end="", flush=True)
            elif isinstance(chunk, AssistantMessage):
                print()
                print("Illustrative usage:", chunk.usage.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
