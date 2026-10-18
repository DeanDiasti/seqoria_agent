"""Validate and execute local tools without a model or remote server."""

import asyncio
from typing import Annotated

from pydantic import Field, ValidationError

from seqoria_agent import LocalToolbox

toolbox = LocalToolbox()


@toolbox.tool()
def repeat(message: str, times: Annotated[int, Field(ge=1, le=3)] = 1) -> str:
    """Repeat a message between one and three times."""
    return " ".join([message] * times)


async def main() -> None:
    schema = toolbox.list_tools()[0].inputSchema
    print(
        "Allowed repetitions:",
        schema["properties"]["times"]["minimum"],
        "to",
        schema["properties"]["times"]["maximum"],
    )
    result = await toolbox.call_tool("repeat", {"message": "hello", "times": 2})
    print("valid:", result[0].text)
    try:
        await toolbox.call_tool("repeat", {"message": "hello", "times": 0})
    except ValidationError:
        # Direct toolbox calls raise; AsyncAIClient execution wraps failures in ToolCallError.
        print("invalid: rejected invalid arguments")


if __name__ == "__main__":
    asyncio.run(main())
