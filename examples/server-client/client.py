"""Interactive command-line client for demonstrating the seqoria_agent AIClient.

This script connects to example MCP servers, initializes an AIClient with
LLM providers, and enters an interactive loop where the user can chat.

Demonstrates:
- Connecting to multiple MCP servers
- Using multiple LLM providers with fallback
- Streaming responses
- Server-populated parameter injection
- Caching and resilience features

To run:
1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment
2. Start the example servers:
   python examples/server-client/server1.py
   python examples/server-client/server2.py
3. Run this script:
   python examples/server-client/client.py
"""

import asyncio
import os

from dotenv import load_dotenv

from seqoria_agent import AsyncAIClient
from seqoria_agent.client.cache import DictCache
from seqoria_agent.client.parameters import ParameterRegistry
from seqoria_agent.providers import AsyncOpenAIProvider
from seqoria_agent.models.chunk import ChatChunk, ToolMessages

load_dotenv()

# ------------------------------------------------------------------
# MCP server endpoints
# Note: The /mcp endpoint is automatically appended by MCPTransportFactory
# ------------------------------------------------------------------

servers = [
    "http://localhost:8000",  # server1.py - Agentic Tools Server
    "http://localhost:8005",  # server2.py - Utility Server
]


async def main() -> None:
    """Interactive CLI that streams AI responses."""

    # Parameter registry for server-populated parameters
    registry = ParameterRegistry(mapping={
        "tenant_id": "demo-tenant",
        "user_id": "cli-user-123",
    })

    # Initialize AsyncAIClient with caching and resilience
    client = AsyncAIClient(
        mcp_servers=servers,
        providers=[
            AsyncOpenAIProvider(
                model_id="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        ],
        # Caching
        cache=DictCache(),
        cache_ttl=600,
        # Concurrency and resilience
        max_concurrency=5,
        tool_timeout=8.0,
        retry_policy={"max_attempts": 3, "backoff_base": 0.5},
        breaker_threshold=5,
        breaker_reset_after=60,
        # Parameter injection
        param_registry=registry,
    )

    try:
        async with client:
            print("=" * 60)
            print("Welcome to the Seqoria-Agent Multi-Tool Client!")
            print("Connected to MCP servers and discovered tools.")
            print()
            print("Try these commands:")
            print("  - 'Roll 5 dice'")
            print("  - 'What is the sum of rolling 3 dice?'")
            print("  - 'Greet me as Alice'")
            print()
            print("Type 'exit' or 'quit' to end the session.")
            print("=" * 60)

            chat_history = []

            while True:
                try:
                    user_input = input("\nYou:\n").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ("exit", "quit"):
                        print("Goodbye!")
                        break

                    print("\nAssistant: ", end="", flush=True)

                    # Stream the response
                    async for chunk in client.stream_chat(
                        user_query=user_input,
                        history=chat_history,
                        max_turns=5,
                    ):
                        if isinstance(chunk, ChatChunk):
                            print(chunk.content, end="", flush=True)
                        elif isinstance(chunk, ToolMessages):
                            tool_count = len(chunk) // 2
                            print(f"\n[Executing {tool_count} tool(s)...]", end="")

                    print()  # Newline after streaming

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
