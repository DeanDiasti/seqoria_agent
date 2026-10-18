# Demonstration of Pydantic Field metadata expansion in Seqoria-Agent tools
# Run with: python examples/toolbox/field_schema_demo.py

from pprint import pprint
from typing import Literal
from enum import Enum

try:
    from pydantic import Field
except ModuleNotFoundError:
    raise SystemExit("This demo requires 'pydantic' to be installed.")

from seqoria_agent.toolbox import LocalToolbox


tb = LocalToolbox()


class SearchType(str, Enum):
    TEST = "test"
    DEMO = "demo"


@tb.tool()
async def search(
    query: Literal[SearchType.TEST, SearchType.DEMO] = Field(
        ..., description="Free-text search query"
    ),
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return"),
):
    """Dummy search tool."""
    return {"answer": "N/A"}


if __name__ == "__main__":
    tool_info = tb.list_tools()[0]
    print("JSON-schema for the 'search' tool parameters (what the AI sees):\n")
    pprint(tool_info.inputSchema)
