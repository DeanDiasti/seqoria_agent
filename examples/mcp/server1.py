"""External MCP SDK server for the client examples, on port 8000.

Run: python examples/mcp/server1.py
This uses the official MCP SDK directly; Seqoria Agent supplies the client.
"""

import argparse
import random

from mcp.server.fastmcp import FastMCP

app = FastMCP("Dice Tools", host="127.0.0.1", port=8000)


@app.tool()
async def roll_dice(n_dice: int) -> list[int]:
    """Roll between 1 and 100 six-sided dice."""
    if not 1 <= n_dice <= 100:
        raise ValueError("n_dice must be between 1 and 100")
    return [random.randint(1, 6) for _ in range(n_dice)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport", choices=["streamable-http", "stdio"], default="streamable-http"
    )
    args = parser.parse_args()
    app.run(transport=args.transport)
