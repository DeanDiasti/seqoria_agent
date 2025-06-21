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
import os
import logging
import structlog
import json

from dotenv import load_dotenv

from seqoria_agent import AIClient, ChatChunk
from seqoria_agent.client.cache import DictCache
from seqoria_agent.providers import OpenAIProvider, AnthropicProvider
from seqoria_agent.client.parameters import ParameterProvider, ParamContext

load_dotenv()

# ------------------------------------------------------------------
# MCP server endpoints discovered by the demo client
# ------------------------------------------------------------------

servers = [
    "http://localhost:8004/mcp",
    "http://localhost:8005/mcp",
]

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

# ------------------------------------------------------------------
# Client-side parameter provider example
# ------------------------------------------------------------------

class DemoUserProvider:  # simple example
    async def resolve(self, param_name: str, tool_name: str, context: ParamContext):  # type: ignore[override]
        if param_name == "user_id":
            return os.getenv("DEMO_USER_ID", "abc123abc123")
        return None

async def main():
    """Interactive CLI that streams AI responses and display envelopes.

    Demonstrates the *serverArgs* channel for server-populated parameters –
    no HTTP headers are used; the AIClient injects ``user_id`` into the
    JSON-RPC body under the ``serverArgs`` key.
    """

    # ------------------------------------------------------------------
    # Build provider list – primary plus optional *backup* provider
    # ------------------------------------------------------------------

    providers = []

    # Primary OpenAI provider (requires OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        providers.append(OpenAIProvider(model_id="gpt-4o"))

    # Backup Anthropic provider (requires ANTHROPIC_API_KEY)
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append(AnthropicProvider())

    if not providers:
        print(
            "❌ Error: No LLM providers could be initialised. Ensure at least OPENAI_API_KEY or ANTHROPIC_API_KEY is set."
        )
        return

    # Initialize AIClient **with caching enabled** (in-memory DictCache)
    client = AIClient(
        urls=servers,
        providers=providers,
        cache=DictCache(),  # Swap with RedisCache(...) in production
        cache_ttl=600,  # Cache tool results for 10 minutes
        # NEW: Adaptive concurrency & reliability settings
        max_concurrency=5,  # Allow up to 5 simultaneous tool calls
        tool_timeout=8.0,  # Cancel a tool if it runs >8s
        retry_policy={"max_attempts": 3, "backoff_base": 0.5},
        breaker_threshold=5,  # Trip circuit after 5 consecutive failures
        breaker_reset_after=60,  # Auto-reset breaker after 60s
        param_providers=[DemoUserProvider()],
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
                    async for chunk in client.stream_chat(
                        user_query=user_input, history=chat_history, max_turns=5
                    ):
                        if chunk.type == "llm_delta":
                            # Stream token deltas inline
                            print(chunk.text, end="", flush=True)
                        elif chunk.type == "tool_display":
                            print("\n[DISPLAY ENVELOPE]")
                            print(json.dumps(chunk.envelope, indent=2))

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