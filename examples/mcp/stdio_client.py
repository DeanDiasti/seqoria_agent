"""Launch the bundled MCP SDK utility server as a local child process."""

import asyncio
import shlex
import sys
from pathlib import Path

from seqoria_agent.client.mcp_transport import MCPServerConfig, MCPTransportFactory


async def main() -> None:
    server = Path(__file__).with_name("server2.py")
    command = shlex.join([sys.executable, str(server), "--transport", "stdio"])
    config = MCPServerConfig(url=f"stdio://{command}")
    # Exiting closes the MCP session and its child process.
    async with MCPTransportFactory.connect(config) as connection:
        result = await connection.session.call_tool("divide", {"a": 8, "b": 2})
        if result.isError:
            raise RuntimeError("The utility server rejected divide")
        print("8 / 2 =", "".join(item.text for item in result.content if item.type == "text"))


if __name__ == "__main__":
    asyncio.run(main())
