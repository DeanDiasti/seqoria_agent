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

from seqoria_agent.client import AIClient  # convenience import re-exports helpers
from seqoria_agent.providers import OpenAIProvider
from seqoria_agent.toolbox import LocalToolbox

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

def _print_assistant(text: str | None):  # noqa: D401 – tiny util
    if not text:
        return
    for line in str(text).splitlines():
        print(f"Assistant: {line}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main() -> None:  # noqa: D401
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is missing.")
        return

    providers = [OpenAIProvider(model_id="gpt-4o")]

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
                print("Exiting – empty input detected.")
                break

            # 1) Update history with system + user message
            ai.prepare_history(user_query, history)

            # 2) Inner agentic loop – let the model think & use tools until ready
            for _ in range(5):  # hard-limit internal turns per user input
                resp = await ai.llm_generate_once(history)
                ai.append_assistant_message(resp.text or "", history, tool_calls=resp.tool_calls)

                if resp.tool_calls:
                    early = await ai.run_tool_calls(resp.tool_calls, history)
                    if early and early.get("answer"):
                        _print_assistant(early["answer"])
                        break  # finished for this user turn
                    # else: tool ran but no immediate answer – let LLM continue
                    continue

                # Model responded directly – done for this user turn
                _print_assistant(resp.text)
                break

        # Conversation end --------------------------------------------------
        print("\nSession finished.")


if __name__ == "__main__":  # pragma: no cover – CLI entrypoint
    asyncio.run(main()) 