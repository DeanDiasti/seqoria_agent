from __future__ import annotations

"""Example: Using AIClient with token usage reporting.

Run with:
    python examples/token_counting/client.py

The example uses the same _DummyProvider from the test-suite so it works out
of the box without external API keys.  Replace the provider with a real
OpenAIProvider / AnthropicProvider (and provide keys) to see live numbers.
"""

import asyncio
from seqoria_agent.client.ai_client.base import AIClient
from seqoria_agent.providers import OpenAIProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage
import os
from dotenv import load_dotenv

load_dotenv()

async def main() -> None:
    ai = AIClient(providers=[OpenAIProvider(api_key=os.getenv(key="OPENAI_API_KEY"))], urls=[])
    hist: list[dict] = [{"role": "user", "content": "Hello, how are you?"}]
    # Single-shot generation -------------------------------------------------
    result = await ai.llm_generate_once(history=hist, 
                                        use_tools=False, 
                                        include_usage=True)
    print("One-shot answer:", result["response"].text)
    print("Token usage: ", result["usage"].model_dump())

    # Streaming generation ---------------------------------------------------
    print("\nStreaming answer:")
    async for chunk in ai.llm_stream_once(history=hist, 
                                          use_tools=False, 
                                          include_usage=True):
        if isinstance(chunk, str):
            print(chunk, end="", flush=True)
        else:
            print("\n(Token usage:", chunk["usage"].model_dump(), ")")


if __name__ == "__main__":
    asyncio.run(main=main())
