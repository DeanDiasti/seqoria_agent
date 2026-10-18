from __future__ import annotations

"""Example: Using AIClient with token usage reporting.

Run with:
    python examples/token_counting/client.py

The example uses the same _DummyProvider from the test-suite so it works out
of the box without external API keys.  Replace the provider with a real
OpenAIProvider / AnthropicProvider (and provide keys) to see live numbers.
"""

import asyncio
from seqoria_agent import AsyncAIClient
from seqoria_agent.providers import AsyncOpenAIProvider
import os
from dotenv import load_dotenv
from seqoria_agent.models.chunk import AssistantMessage, ToolMessages, ChatChunk

load_dotenv()

async def main() -> None:
    ai = AsyncAIClient(providers=[AsyncOpenAIProvider(api_key=os.getenv(key="OPENAI_API_KEY"))], urls=[])
    hist: list[dict] = [{"role": "user", "content": "Hello, how are you?"}]
    # Single-shot generation -------------------------------------------------
    result = await ai.llm_generate_once(history=hist)
    print("One-shot answer:", result.content)
    print("Token usage: ", result.usage.model_dump())

    # Streaming generation ---------------------------------------------------
    print("\nStreaming answer:")
    async for chunk in ai.llm_stream_once(history=hist):
        if isinstance(chunk, ChatChunk):
            print(chunk.content, end="", flush=True)
        elif isinstance(chunk, AssistantMessage):
            print("\n(Token usage:", chunk.usage.model_dump(), ")")
        elif isinstance(chunk, ToolMessages):
            print("tool call\n" * (len(chunk) // 2), end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main=main())
