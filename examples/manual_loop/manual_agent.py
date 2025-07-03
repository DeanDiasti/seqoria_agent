from __future__ import annotations

"""Manual agentic loop demo (runs in your terminal).

This example shows how to use the new low-level *AIClient* helpers to implement
an **interactive CLI** assistant while keeping full control over every LLM/tool
turn.  Two inline tools are registered via *LocalToolbox*:

• ``get_time`` - returns the current ISO-8601 timestamp
• ``random_number`` - returns a pseudorandom integer in the range 0-100

Requirements
============
1.  ``OPENAI_API_KEY`` must be set in your shell for the *OpenAIProvider*.
2.  Python ≥ 3.8 with ``seqoria_agent`` installed (``pip install -e .[dev]``).

Run the script:

```bash
python examples/manual_loop/manual_agent.py
```

Type a question and watch the agent think, call tools, and answer.
"""

import asyncio
import datetime as _dt
import os
import random
from typing import List

from seqoria_agent.client import AIClient
from seqoria_agent import OpenAIProvider
from seqoria_agent.toolbox import LocalToolbox
from seqoria_agent.models.chunk import ToolCallRequest, AssistantMessage, ToolCallRequestList
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Inline tools
# ---------------------------------------------------------------------------

box = LocalToolbox()


@box.tool(description="Return the current timestamp (ISO-8601 string)")
def get_time() -> str:  # noqa: D401 – simple helper
    return _dt.datetime.now().isoformat()


@box.tool(description="Return a random integer between 0 and 100 (inclusive)")
def random_number() -> int:  # noqa: D401 – simple helper
    return random.randint(0, 100)


# ---------------------------------------------------------------------------
# Helper – print assistant output nicely
# ---------------------------------------------------------------------------

def _print_assistant(text: str | None) -> None:  # noqa: D401 – tiny util
    if not text:
        return
    for line in str(object=text).splitlines():
        print(f"Assistant: {line}")


async def main() -> None:  # noqa: D401
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is missing.")
        return

    providers = [OpenAIProvider(model_id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))]

    async with AIClient(urls=[], providers=providers, local_toolbox=box) as ai:
        history: List[dict] = []

        MAX_ROUNDS = 5
        for turn in range(1, MAX_ROUNDS + 1):
            try:
                user_query = input(f"You [{turn}/{MAX_ROUNDS}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGood-bye!")
                break

            if not user_query:
                print("Exiting - empty input detected.")
                break

            history.append({"role": "user", "content": user_query}) 

            # 1) Inner agentic loop – let the model think & use tools until ready
            for _ in range(5):  # hard-limit internal turns per user input
                resp = await ai.llm_generate_once(history=history)

                if isinstance(resp, ToolCallRequestList):
                    tool_results = await ai.run_tool_calls(tool_calls=resp.tool_calls)

                    # Interleave requests and corresponding results for OpenAI compliance
                    tool_messages: list = []
                    for req, res in zip(resp.tool_calls, tool_results):
                        tool_messages.extend([req, res])

                    history.extend(tool_messages)
                    continue

                if isinstance(resp, AssistantMessage):
                    history.append({"role": "assistant", "content": resp.content})
                    _print_assistant(text=resp.content)
                    break  # finished for this user turn 

                break

        # Conversation end --------------------------------------------------
        print("\nSession finished.")


if __name__ == "__main__":  # pragma: no cover – CLI entrypoint
    asyncio.run(main()) 