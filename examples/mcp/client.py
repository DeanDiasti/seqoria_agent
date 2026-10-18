"""Discover and call tools on the two local HTTP MCP example servers."""

import argparse
import asyncio
from contextlib import AsyncExitStack

from seqoria_agent.client.mcp_transport import connect_mcp


async def main(servers: list[str]) -> None:
    # Direct transport sessions need no model provider. Stack cleanup follows
    # reverse connection order, closing every MCP session and its HTTP client.
    async with AsyncExitStack() as stack:
        names = []
        utility_session = None
        for url in servers:
            connection = await stack.enter_async_context(connect_mcp(url))
            listing = await connection.session.list_tools()
            names.extend(tool.name for tool in listing.tools)
            if any(tool.name == "divide" for tool in listing.tools):
                utility_session = connection.session
        print("Discovered tools:", ", ".join(sorted(names)))
        if utility_session is None:
            raise RuntimeError("Start the utility server (server2.py) to provide divide")
        result = await utility_session.call_tool("divide", {"a": 8, "b": 2})
        if result.isError:
            raise RuntimeError("The utility server rejected divide")
        print("8 / 2 =", "".join(item.text for item in result.content if item.type == "text"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="append", help="MCP URL; repeat to use multiple servers")
    args = parser.parse_args()
    asyncio.run(main(args.server or ["http://127.0.0.1:8000/mcp", "http://127.0.0.1:8005/mcp"]))
