"""Interactive command-line client for demonstrating the seqoria_agent AIClient.

This script connects to the example **FastXMCP** servers (`server1.py`, `server2.py`),
initializes an `AIClient` with multiple LLM providers, and enters an
interactive loop where the user can issue commands.

It demonstrates:
- Connecting to multiple servers.
- Using multiple LLM providers with fallback.
- Streaming responses (`stream_chat`).
- Rendering `DisplayEnvelope` objects received from tools.
- Maintaining a stateless conversation history.

To run this demo:
1. Make sure you have an OpenAI or Anthropic API key set as an
   environment variable (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).
2. Run the example servers in separate terminals:
   `uvicorn examples.servers.server1:app --reload --port 8004`
   `uvicorn examples.servers.server2:app --reload --port 8005`
3. Run this script:
   `python examples/clients/demo.py`
"""
import asyncio
import json
import os

import structlog
from dotenv import load_dotenv

from seqoria_agent import AIClient
from seqoria_agent.client.cache import DictCache
from seqoria_agent.client.parameters import ParameterRegistry
from seqoria_agent.providers import OpenAIProvider
from seqoria_agent.models.chunk import ChatChunk, ToolMessages

load_dotenv()

# ------------------------------------------------------------------
# MCP server endpoints discovered by the demo client
# ------------------------------------------------------------------

servers = ["http://localhost:8004/mcp", 
           "http://localhost:8005/mcp"]

# ------------------------------------------------------------------
# Structured logging setup for the demo
# ------------------------------------------------------------------

# logging.basicConfig(level=logging.INFO, format="%(message)s")
# structlog.configure(
#     wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
#     processors=[
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.processors.add_log_level,
#         structlog.processors.StackInfoRenderer(),
#         structlog.processors.dict_tracebacks,
#         structlog.processors.JSONRenderer(),
#     ],
# )

logger = structlog.get_logger("seqoria_agent_demo")


async def main() -> None:
    """Interactive CLI that streams AI responses and display envelopes.

    Demonstrates the *serverArgs* channel for server-populated parameters –
    no HTTP headers are used; the AIClient injects ``user_id`` into the
    JSON-RPC body under the ``serverArgs`` key.
    """
    # ------------------------------------------------------------------
    # Build provider list – primary plus optional *backup* provider
    # ------------------------------------------------------------------

    registry = ParameterRegistry(mapping={"tenant_id": "demo-tenant"})

    # Initialize AIClient **with caching enabled** (in-memory DictCache)
    client = AIClient(
        urls=servers,
        providers=[OpenAIProvider(model_id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))],
        cache=DictCache(),  # Swap with RedisCache(...) in production
        cache_ttl=600,  # Cache tool results for 10 minutes
        # NEW: Adaptive concurrency & reliability settings
        max_concurrency=5,  # Allow up to 5 simultaneous tool calls
        tool_timeout=8.0,  # Cancel a tool if it runs >8s
        retry_policy={"max_attempts": 3, "backoff_base": 0.5},
        breaker_threshold=5,  # Trip circuit after 5 consecutive failures
        breaker_reset_after=60,  # Auto-reset breaker after 60s
        param_registry=registry,
        # logger=logger,
    )

    try:
        async with client:
            print("===================================================")
            print("Welcome to the Seqoria-Agent Multi-Tool Client!")
            print("I have discovered tools from the connected MCP server.")
            print("\n--- Try out the HYBRID tool! ---")
            print("  - Ask me to 'change the color to pink' (Display Mode)")
            print("  - Ask me 'what is the hex code for blue?' (Data Mode)")
            print("\n--- Try out the DISPLAY-ONLY tool! ---")
            print("  - Ask me to 'flash the screen 3 times with green'")
            print("\n--- Dependency-graph demo ---")
            print("  - Ask me to 'roll 5 dice and give me the sum' → rolls first, then sums (depends_on).")
            print("  - Ask me to 'what time is it and also divide 10 by 2 with timestamp' → executes in two layers.")
            print("\nType 'exit' or 'quit' to end the session.")
            print("===================================================")

            chat_history = []  # To maintain conversation context
            while True:
                try:
                    user_input = input("You:\n")
                    if user_input.lower() in ["exit", "quit"]:
                        print("Goodbye!")
                        break

                    # Stream the response incrementally -----------------------
                    async for chunk in client.stream_chat(user_query=user_input, history=chat_history, max_turns=5):
                        if isinstance(chunk, ChatChunk):
                            print(chunk.content, end="", flush=True)
                        elif isinstance(chunk, ToolMessages):
                            print("tool call\n" * (len(chunk) // 2))

                    print("\n")  # Newline after streaming completes

                    # chat_history is mutated in-place during streaming
                    chat_history = chat_history

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break

    except RuntimeError as e:
        print(f"❌ A critical error occurred: {e}")
        logger.error("Critical error in demo", error=str(object=e))


if __name__ == "__main__":
    asyncio.run(main=main()) 