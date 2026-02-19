"""examples/providers/openrouter_agent.py

Interactive CLI agent using the OpenAI-compatible provider with OpenRouter.

This example demonstrates how to connect to OpenRouter's API using the
``AsyncOpenAICompatibleProvider`` and run a full agentic loop with **streaming**
output, tool calling, memory, and thinking-token display — all powered by
MiniMax M2.5 (via OpenRouter).

Two inline tools are registered:
  - ``get_time``      – returns the current ISO-8601 timestamp
  - ``random_number`` – returns a pseudorandom integer in 0-100

If the model emits reasoning / thinking tokens they are printed in real-time
inside a ``[Thinking]`` block before the visible answer.

Prerequisites
-------------
1. Set ``OPENROUTER_API_KEY`` in your environment (or a ``.env`` file).
2. Python >= 3.10 with ``seqoria_agent`` installed (``pip install -e .[dev]``).

Run with::

    python examples/providers/openrouter_agent.py
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import os
import random

from dotenv import load_dotenv

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.models.chunk import (
    AssistantMessage,
    ChatChunk,
    ToolCallRequestList,
)
from seqoria_agent.memory import AgenticMemory, AgenticMemoryConfig
from seqoria_agent.providers import AsyncOpenAICompatibleProvider
from seqoria_agent.toolbox import LocalToolbox

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = "z-ai/glm-5"

GRAY = "\033[90m"
BLUE = "\033[34m"
DIM = "\033[2m"
RESET = "\033[0m"

box = LocalToolbox()


@box.tool(description="Return the current timestamp (ISO-8601 string)")
def get_time() -> str:
    return _dt.datetime.now().isoformat()


@box.tool(description="Return a random integer between 0 and 100 (inclusive)")
def random_number() -> int:
    return random.randint(0, 100)


async def _stream_and_collect(ai: AsyncAIClient, history) -> AssistantMessage | ToolCallRequestList | None:
    """Stream one LLM turn, printing thinking + text tokens live.

    Returns the final ``AssistantMessage`` or ``ToolCallRequestList`` so the
    caller can feed it back into the agentic loop.
    """
    in_thinking = False
    in_text = False
    final: AssistantMessage | ToolCallRequestList | None = None

    async for chunk in ai.llm_stream_once(history=history):
        if isinstance(chunk, ChatChunk):
            if chunk.thinking:
                if not in_thinking:
                    print(f"  {GRAY}{DIM}[Thinking]\n", end="", flush=True)
                    in_thinking = True
                print(chunk.thinking, end="", flush=True)

            if chunk.content:
                if in_thinking:
                    print(f"{RESET}\n", flush=True)
                    in_thinking = False
                if not in_text:
                    print(f"  {BLUE}Assistant:\n", end="", flush=True)
                    in_text = True
                print(chunk.content, end="", flush=True)

        elif isinstance(chunk, AssistantMessage):
            if in_thinking:
                print(f"{RESET}\n", flush=True)
                in_thinking = False

            if chunk.thinking and not in_text:
                print(f"  {GRAY}{DIM}[Thinking] {chunk.thinking}{RESET}")

            if chunk.content:
                if not in_text:
                    print(f"  {BLUE}Assistant: {chunk.content}{RESET}", flush=True)
                else:
                    print(f"{RESET}", flush=True)

            if chunk.usage:
                print(
                    f"  {DIM}[{chunk.model or MODEL_ID}] "
                    f"tokens: {chunk.usage.prompt_tokens} in / "
                    f"{chunk.usage.completion_tokens} out / "
                    f"{chunk.usage.total_tokens} total{RESET}"
                )
            final = chunk

        elif isinstance(chunk, ToolCallRequestList):
            if in_thinking:
                print(f"{RESET}\n", flush=True)
            print(f"  [Tool calls] {', '.join(tc.name for tc in chunk.tool_calls)}")
            final = chunk

    return final


async def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is missing.")
        print("Get one at https://openrouter.ai/keys")
        return

    provider = AsyncOpenAICompatibleProvider(
        model_id=MODEL_ID,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.7,
        max_tokens=2048,
    )

    print(f"Connected to OpenRouter  |  model: {MODEL_ID}")
    print("Type a message to chat. Empty input or Ctrl-C exits.\n")

    async with AsyncAIClient(
        urls=[],
        providers=[provider],
        local_toolbox=box,
    ) as ai:
        memory = AgenticMemory(
            config=AgenticMemoryConfig(
                session_id="openrouter-demo",
                conversation_id="interactive-cli",
                persist_memories=False,
            )
        )

        MAX_ROUNDS = 10
        for turn in range(1, MAX_ROUNDS + 1):
            try:
                user_query = input(f"  You [{turn}/{MAX_ROUNDS}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGood-bye!")
                break

            if not user_query:
                print("Exiting — empty input.")
                break

            llm_history = memory.begin_turn(user_query, retrieval_query=user_query)

            for _ in range(5):
                resp = await _stream_and_collect(ai, llm_history)

                if isinstance(resp, ToolCallRequestList):
                    tool_results = await ai.run_tool_calls(tool_calls=resp.tool_calls)
                    memory.record_tool_cycle(resp.tool_calls, tool_results)
                    llm_history = memory.messages_for_llm(query=user_query)
                    continue

                if isinstance(resp, AssistantMessage):
                    memory.commit_assistant_response(
                        content=resp.content,
                        model=resp.model,
                        usage=resp.usage,
                    )
                    memory.end_turn()
                    break

                break

        print("\nSession finished.")


if __name__ == "__main__":
    asyncio.run(main())
