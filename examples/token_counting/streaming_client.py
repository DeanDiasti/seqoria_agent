from __future__ import annotations

"""Example showing token counting with AIClient.chat()."""

import asyncio
import os
from dotenv import load_dotenv
import random

from seqoria_agent import AsyncAIClient
from seqoria_agent.providers import AsyncOpenAIProvider
from seqoria_agent import LocalToolbox
from seqoria_agent.models.chunk import ChatChunk, AssistantMessage, ToolMessages

load_dotenv()

toolbox = LocalToolbox()

@toolbox.tool()
def random_number() -> int:
    """Generate a random number between 0 and 100."""
    return random.randint(0, b=100)


async def main() -> None:
    provider = AsyncOpenAIProvider(api_key=os.getenv(key="OPENAI_API_KEY"))

    async with AsyncAIClient(providers=[provider], local_toolbox=toolbox) as ai:
        async for chunk in ai.stream_chat(
            user_query="generate a random number between 0 and 100",
            history=[],
            max_turns=5,
        ):
            if isinstance(chunk, ChatChunk):
                # Stream token deltas inline
                print(chunk.content, end="", flush=True)
            elif isinstance(chunk, AssistantMessage):
                print(f"usage: {chunk.usage.model_dump()}")
            elif isinstance(chunk, ToolMessages):
                print("tool call\n" * (len(chunk) // 2))


if __name__ == "__main__":
    asyncio.run(main=main())
