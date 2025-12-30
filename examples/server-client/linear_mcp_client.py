"""Example: Connect to Linear's remote MCP server with OAuth 2.1.

This script demonstrates connecting to an OAuth 2.1 protected MCP server
using seqoria-agent's built-in OAuth support.

How It Works
------------
When you run this script, it will:

1. Attempt to connect to Linear's MCP server
2. The server returns 401 Unauthorized with OAuth metadata
3. seqoria-agent automatically:
   - Discovers the authorization server endpoints
   - Registers as an OAuth client (Dynamic Client Registration)
   - Opens your browser for you to log in to Linear
4. You log in and authorize the app
5. Copy the callback URL from your browser and paste it back
6. seqoria-agent exchanges the code for tokens
7. Connection completes and tools are discovered!

No API keys or tokens needed - just run the script and log in.

Usage
-----
    python examples/server-client/linear_mcp_client.py

References
----------
- MCP Authorization Spec: https://modelcontextprotocol.io/specification/draft/basic/authorization
- MCP OAuth Tutorial: https://modelcontextprotocol.io/docs/tutorials/security/authorization
"""

from __future__ import annotations

import asyncio

from seqoria_agent import AsyncAIClient
from seqoria_agent.client.mcp_transport import MCPServerConfig, OAuthConfig
from seqoria_agent.providers.base_provider import AsyncBaseProvider
from seqoria_agent.providers.schemas import LLMResponse, LLMUsage

# ---------------------------------------------------------------------------
# Stub provider (no LLM calls needed for this demo)
# ---------------------------------------------------------------------------


class _StubProvider(AsyncBaseProvider):
    """Minimal provider that returns a static response (no API key needed)."""

    async def async_generate(self, *, chat_history, **kwargs):
        return LLMResponse(
            model_id="stub",
            type="llm_response",
            text="(stub response)",
            usage=LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def async_stream(self, *, chat_history, **kwargs):
        async def _gen():
            yield LLMResponse(
                model_id="stub",
                type="llm_response",
                text="(stub response)",
                usage=LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

        return _gen()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

LINEAR_MCP_URL = "https://mcp.linear.app/mcp"


async def main() -> None:
    print("=" * 60)
    print("Linear MCP Server - OAuth 2.1 Demo")
    print("=" * 60)
    print()
    print(f"Connecting to: {LINEAR_MCP_URL}")
    print()
    print("This will open your browser for Linear authentication.")
    print("After logging in, copy the callback URL and paste it here.")
    print()

    # Configure OAuth-enabled connection to Linear
    linear_server = MCPServerConfig(
        url=LINEAR_MCP_URL,
        oauth=OAuthConfig(
            client_name="Seqoria Agent - Linear Demo",
        ),
        timeout=60.0,
    )

    try:
        async with AsyncAIClient(
            mcp_servers=[linear_server],
            providers=[_StubProvider()],
        ) as client:
            # List discovered tools
            tools = client.list_tools()

            print()
            print("=" * 60)
            print(f"Successfully connected! Discovered {len(tools)} tool(s):")
            print("=" * 60)
            print()

            for t in tools:
                fn = t.get("function", {})
                name = fn.get("name", "?")
                desc = fn.get("description", "(no description)")
                # Truncate long descriptions
                if len(desc) > 70:
                    desc = desc[:67] + "..."
                print(f"  • {name}")
                print(f"      {desc}")
                print()

            # List MCP servers
            servers = client.list_mcp_servers()
            print("Connected MCP servers:")
            for s in servers:
                print(f"  - {s.url}")

            # -------------------------------------------------------------
            # Optional: Try calling a tool
            # -------------------------------------------------------------
            # Uncomment to test tool execution:
            #
            # print()
            # print("Calling 'list_issues' tool...")
            # result = await client.call_mcp_tool("list_issues", {"first": 5})
            # print(f"Result: {result}")

    except Exception as e:
        print()
        print(f"Connection failed: {e}")
        print()
        print("If OAuth failed, check that:")
        print("  - You have a Linear account")
        print("  - You authorized the app in your browser")
        print("  - You copied the FULL callback URL including the ?code=... part")


if __name__ == "__main__":
    asyncio.run(main())
