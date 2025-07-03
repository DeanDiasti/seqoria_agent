"""examples/providers/tool_response.py

Demonstration of using `OpenAIProvider.async_stream` with a tool declaration.

The example registers a single tool, ``roll_dice``, and asks the model to use
it.  It then streams the assistant response so you can observe how plain text
chunks and the final ``LLMResponse`` object (containing ``tool_calls``) are
emitted.

Prerequisites
-------------
1. ``python -m pip install openai``  (plus ``seqoria-agent`` in editable mode)
2. ``export OPENAI_API_KEY=sk-...`` - your OpenAI API key must be present in
   the environment (or passed explicitly when instantiating ``OpenAIProvider``).

Run with::

    python examples/providers/tool_re   sponse.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, AsyncGenerator, Union

from seqoria_agent.providers.openai_provider import OpenAIProvider
from seqoria_agent.providers.anthropic_provider import AnthropicProvider
from seqoria_agent.providers.schemas import LLMResponse, ToolCallList

from dotenv import load_dotenv

load_dotenv()

def _render_stream(stream: AsyncGenerator[Union[str, LLMResponse], None]) -> None:  # noqa: D401
    """Consume *stream* and print chunks to stdout.

    ``str`` chunks are printed as-is (delta tokens).  The final
    :class:`~seqoria_agent.providers.schemas.LLMResponse` object is displayed
    in a readable form so you can inspect ``tool_calls`` and token usage.
    """

    async def _inner() -> None:  # noqa: D401
        async for chunk in stream:
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
            elif isinstance(chunk, LLMResponse):  # final LLMResponse
                print("\n\n--- final response object ---")
                print(f"text         : {repr(chunk.text)}")
                print(f"stop_reason  : {chunk.stop_reason}")
                print(f"token usage  : {chunk.usage}")
            elif isinstance(chunk, ToolCallList):
                print("\n\n--- tool call list ---")
                for tc in chunk.tool_calls:
                    print(f"tool_calls   : {tc.model_dump()}")
                print(f"tool_calls   : {[tc.model_dump() for tc in (chunk.tool_calls or [])]}")
                print(f"token usage  : {chunk.usage}\n\n")
    asyncio.run(main=_inner())


# ---------------------------------------------------------------------------
# Main demo – declare a *roll_dice* tool and ask the model to use it
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.stderr.write("ERROR: set the OPENAI_API_KEY environment variable first.\n")
        sys.exit(1)

    provider = OpenAIProvider(api_key=api_key)

    # Tool declaration following the OpenAI function-calling schema
    dice_tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "Rolls an n-sided dice and returns the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sides": {
                        "type": "integer",
                        "description": "Number of sides of the dice (default: 6)",
                        "default": 6,
                    }
                },
                "required": [],
            },
        },
    }

    chat_history = [
        {
            "role": "system",
            "content": "You are an assistant that rolls physical dice when asked.",
        },
        {
            "role": "user",
            "content": "Please roll three dice for me three times separately and tell me the result.",
        },
    ]

    print("OpenAI streaming response - watch for tool_calls in the final object…\n")

    stream = provider.async_stream(
        chat_history=chat_history,
        tools=[dice_tool],
        tool_choice="auto",  # let the model decide
    )
    _render_stream(stream=stream)

    # ------------------------------------------------------------------
    # Anthropic demo (if API key is available)
    # ------------------------------------------------------------------
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        provider_a = AnthropicProvider(api_key=anthropic_key)

        print("\nAnthropic streaming response - watch for tool_calls in the final object…\n")

        stream_a = provider_a.async_stream(
            chat_history=chat_history,
            tools=[dice_tool],
            tool_choice="auto",
        )
        _render_stream(stream=stream_a)
    else:
        print("\n[info] Skipping AnthropicProvider demo - set ANTHROPIC_API_KEY to enable it.\n")


if __name__ == "__main__":
    main()
