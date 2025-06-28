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
    return random.randint(0, 100)


async def main() -> None:
    provider = OpenAIProvider(api_key=os.getenv(key="OPENAI_API_KEY"))

    async with AIClient(providers=[provider], local_toolbox=toolbox) as ai:
        history: list[dict] = []
        result = await ai.chat(user_query="generate a random number between 0 and 100", 
                               history=history, 
                               include_usage=True)

        print("Answer:", result["answer"])
        print("Prompt/Completion/Total tokens (turn):", result["usage_turn"].model_dump())
        print("Cumulative tokens so far:", result["usage_total"].model_dump())


if __name__ == "__main__":
    asyncio.run(main=main())
