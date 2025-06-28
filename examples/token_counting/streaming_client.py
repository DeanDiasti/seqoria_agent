from __future__ import annotations

"""Example showing token counting with AIClient.chat()."""

import asyncio
import os
from dotenv import load_dotenv
import random

from seqoria_agent.client.ai_client.base import AIClient
from seqoria_agent.providers import OpenAIProvider
from seqoria_agent import LocalToolbox

load_dotenv()

toolbox = LocalToolbox()

@toolbox.tool()
def random_number() -> int:
    """Generate a random number between 0 and 100."""
    return random.randint(0, b=100)


async def main() -> None:
    provider = OpenAIProvider(api_key=os.getenv(key="OPENAI_API_KEY"))

    async with AIClient(providers=[provider], local_toolbox=toolbox) as ai:
        per_turn_usage = None
        cumulative_usage = None

        async for chunk in ai.stream_chat(
            user_query="generate a random number between 0 and 100",
            history=[],
            max_turns=5,
            include_usage=True,
        ):
            if chunk.type == "llm_delta":
                # Stream token deltas inline
                print(chunk.text, end="", flush=True)
            elif chunk.type == "token_usage":
                if chunk.cumulative:
                    cumulative_usage = chunk.usage
                else:
                    per_turn_usage = chunk.usage

        # newline after streaming text
        print()
        if per_turn_usage:
            print("Prompt/Completion/Total tokens (turn):", per_turn_usage.model_dump())
        if cumulative_usage:
            print("Cumulative tokens so far:", cumulative_usage.model_dump())


if __name__ == "__main__":
    asyncio.run(main=main())
