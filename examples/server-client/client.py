"""Interactive command-line client for demonstrating the seqoria_agent AIClient.

This script connects to example MCP servers, initializes an AIClient with
LLM providers, and enters an interactive loop where the user can chat.

Demonstrates:
- Connecting to multiple MCP servers
- Streaming responses
- Bounded tool concurrency

To run:
1. Start Ollama and pull the model: ollama pull gpt-oss:20b
2. Start the example servers:
   python examples/server-client/server1.py
   python examples/server-client/server2.py
3. Run this script:
   python examples/server-client/client.py
"""

import asyncio


from seqoria_agent import AsyncAIClient
from seqoria_agent.providers import AsyncOllamaProvider
from seqoria_agent.models.chunk import ChatChunk, ToolMessages


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

    # Providers are owned by the application and closed after the client.
    provider = AsyncOllamaProvider(model_id="gpt-oss:20b", keep_alive="5m")
    # Initialize AsyncAIClient
    client = AsyncAIClient(
        mcp_servers=servers,
        providers=[provider],
        # Concurrency and resilience
        max_concurrency=5,
        tool_timeout=8.0,
        breaker_threshold=5,
        breaker_reset_after=60,
    )

    try:
        async with provider, client:
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

                    thinking_open = False
                    thinking_closed = False

                    # Stream the response
                    async for chunk in client.stream_chat(
                        user_query=user_input,
                        history=chat_history,
                        max_turns=5,
                    ):
                        if isinstance(chunk, ChatChunk):
                            if chunk.thinking:
                                if not thinking_open:
                                    print("\n<thinking>\n", end="", flush=True)
                                    thinking_open = True
                                if not thinking_closed:
                                    print(chunk.thinking, end="", flush=True)
                            if chunk.content:
                                if thinking_open and not thinking_closed:
                                    print("\n</thinking>\n\nassistant: ", end="", flush=True)
                                    thinking_closed = True
                                print(chunk.content, end="", flush=True)
                        elif isinstance(chunk, ToolMessages):
                            tool_count = len(chunk) // 2
                            print(f"\n[Executing {tool_count} tool(s)...]", end="")

                    # Close thinking block if the model produced only thinking (no visible content)
                    if thinking_open and not thinking_closed:
                        print("\n</thinking>\n\nassistant:\n", end="", flush=True)

                    print()  # Newline after streaming

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
