"""examples/toolbox/tool_blocking_demo.py.

Demonstrates blocking tools from being surfaced to the LLM "tools" context.

This script:
- Creates a LocalToolbox with 3 tools
- Blocks one tool by default (constructor arg: blocked_tools)
- Prints the available tools
- Blocks another tool at runtime via client.block_tools(...)
- Prints again
- Unblocks one tool via client.unblock_tools(...)
- Prints again

Run with:
    uv run python examples/toolbox/tool_blocking_demo.py
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage
from seqoria_agent.toolbox import LocalToolbox


class _OfflineStubProvider(AsyncBaseProvider):
    """Minimal provider so AsyncAIClient can be constructed without API keys."""

    async def async_generate(  # type: ignore[override]
        self,
        *,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice=None,  # noqa: ANN001 - provider API varies by implementation
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return LLMResponse(model_id="offline-stub", type="llm_response", text="ok", usage=usage)

    async def async_stream(  # type: ignore[override]
        self,
        *,
        chat_history: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice=None,  # noqa: ANN001 - provider API varies by implementation
        model: str | None = None,
        **kwargs: Any,
    ):
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        yield LLMResponse(model_id="offline-stub", type="llm_response", text="ok", usage=usage)


def _tool_names(tools: list[dict]) -> list[str]:
    return [t["function"]["name"] for t in tools]


async def main() -> None:
    toolbox = LocalToolbox()

    @toolbox.tool()
    def alpha() -> str:  # noqa: D401
        return "alpha"

    @toolbox.tool()
    def beta() -> str:  # noqa: D401
        return "beta"

    @toolbox.tool()
    def gamma() -> str:  # noqa: D401
        return "gamma"

    provider = _OfflineStubProvider()

    async with AsyncAIClient(
        mcp_servers=[],
        providers=[provider],
        local_toolbox=toolbox,
        blocked_tools=["beta"],  # block one by default
    ) as client:
        print("Initially blocked: ['beta']")
        print("Available tools:", _tool_names(client.list_tools()))

        # Block another tool using the new method
        await client.block_tools(["gamma"], refresh=True)
        print("\nAfter client.block_tools(['gamma']):")
        print("Available tools:", _tool_names(client.list_tools()))

        # Unblock one of the blocked tools
        await client.unblock_tools(["beta"], refresh=True)
        print("\nAfter client.unblock_tools(['beta']):")
        print("Available tools:", _tool_names(client.list_tools()))


if __name__ == "__main__":
    asyncio.run(main())
