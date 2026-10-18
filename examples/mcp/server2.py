"""External MCP SDK server for the client examples, on port 8005.

Run: python examples/mcp/server2.py
This uses the official MCP SDK directly; Seqoria Agent supplies the client.
"""

import argparse
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

app = FastMCP("Utility Tools", host="127.0.0.1", port=8005)


@app.tool()
async def current_time() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@app.tool()
async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("b must not be zero")
    return a / b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport", choices=["streamable-http", "stdio"], default="streamable-http"
    )
    args = parser.parse_args()
    app.run(transport=args.transport)
