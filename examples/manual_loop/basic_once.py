"""Basic manual loop example demonstrating Phase-1/2 helpers.

Run with:
    python examples/manual_loop/basic_once.py

The script creates a tiny ``StubProvider`` that echoes "pong" so no network
access is required.  It showcases how to prepare history, execute a single
LLM step and print the result.
"""

import asyncio

from seqoria_agent.client import AIClient  # type: ignore
from seqoria_agent.providers import OpenAIProvider

from dotenv import load_dotenv
import os

load_dotenv()


async def main() -> None:
    async with AIClient(urls=[], providers=[OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))]) as client:
        history: list[dict] = [{"role": "user", "content": "ping"}]
        resp = await client.llm_generate_once(history=history)
        print("AI replied:", resp.content)


if __name__ == "__main__":
    asyncio.run(main()) 