"""examples/server-client/dynamic_servers_client.py.

Demonstrates dynamically adding/removing MCP servers after client initialization.

To run (recommended):
1) Start the example servers (in separate terminals):
   - python examples/server-client/server1.py
   - python examples/server-client/server2.py
2) Run this script:
   - uv run python examples/server-client/dynamic_servers_client.py

Notes:
- This script uses an offline stub provider, so no API keys are required.
- If the MCP servers are not running, the add/remove APIs still work (config-level),
  but no tools will be discovered.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from seqoria_agent.client import AsyncAIClient
from seqoria_agent.client.mcp_transport import MCPServerConfig
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage


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
        async def _gen():
            usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            yield LLMResponse(model_id="offline-stub", type="llm_response", text="ok", usage=usage)

        return _gen()


def _tool_names(tools: list[dict]) -> list[str]:
    return [t["function"]["name"] for t in tools]


def _server_urls(configs: list[MCPServerConfig]) -> list[str]:
    return [c.url for c in configs]


async def main() -> None:
    # These match examples/server-client/client.py
    server1 = "http://localhost:8000"
    server2 = "http://localhost:8005"

    provider = _OfflineStubProvider()
    client = AsyncAIClient(urls=[], providers=[provider])

    async with client:
        print("Initial MCP servers:", _server_urls(client.list_mcp_servers()))
        print("Initial tools:", _tool_names(client.list_tools()))

        print("\nAdd server1…")
        added = await client.add_mcp_server(server1, connect=True, refresh_tools=True)
        print("Added configs:", added)
        print("MCP servers:", _server_urls(client.list_mcp_servers()))
        print("Tools:", _tool_names(client.list_tools()))

        print("\nAdd server2…")
        added = await client.add_mcp_server(server2, connect=True, refresh_tools=True)
        print("Added configs:", added)
        print("MCP servers:", _server_urls(client.list_mcp_servers()))
        print("Tools:", _tool_names(client.list_tools()))

        print("\nRemove server1…")
        removed = await client.remove_mcp_server(server1, disconnect=True, refresh_tools=True)
        print("Removed configs:", removed)
        print("MCP servers:", _server_urls(client.list_mcp_servers()))
        print("Tools:", _tool_names(client.list_tools()))

        print("\nRemove server2…")
        removed = await client.remove_mcp_server(server2, disconnect=True, refresh_tools=True)
        print("Removed configs:", removed)
        print("MCP servers:", _server_urls(client.list_mcp_servers()))
        print("Tools:", _tool_names(client.list_tools()))


if __name__ == "__main__":
    asyncio.run(main())


