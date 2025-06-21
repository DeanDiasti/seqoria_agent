# Quickstart

This guide will get you up and running with xmcp in just a few minutes.

## Installation

First, install the library from source. It's recommended to do this in a virtual environment.

```bash
pip install -e .
```

This will install `xmcp` along with its core dependencies, including `fastmcp` and `pydantic`.

To use the full feature set including documentation generation and testing, install the optional extras:

```bash
pip install -e '.[dev,docs]'
```

## 1. Create an XMCP Server

An `XMCP` server exposes your Python functions as tools that an AI can call. The key is the `@mcp.display_tool` decorator, which can return either data for the LLM or a rich `DisplayEnvelope` for the user.

Here's a simple server, which you can find in `examples/servers/server1.py`:

```python
from xmcp import XMCP, DisplayEnvelope

mcp = XMCP()

@mcp.display_tool
def greet(name: str) -> DisplayEnvelope:
    """A simple tool that greets the user."""
    return DisplayEnvelope(type="markdown", payload=f"### Hello {name}! 👋")

@mcp.display_tool
def sales_summary(region: str, show_user: bool = False):
    """
    A tool that returns a sales summary. If show_user is True, it
    returns a DisplayEnvelope for rendering a table. Otherwise, it
    returns raw data for the LLM.
    """
    # In a real app, you would fetch this from a database.
    summary = {"region": region, "sales": 42, "prospects": 120}
    
    if show_user:
        # Construct a fancy table for the user
        return DisplayEnvelope(
            type="table",
            payload=[summary],
            title=f"Sales Summary: {region}"
        )
    
    # Return a simpler dictionary for the LLM to process
    return {"region": region, "sales": summary["sales"]}

# To run this server, save the code and run:
# uvicorn examples.servers.server1:mcp --port 3333
```

To run this server, execute the following command from your terminal:

```bash
uvicorn examples.servers.server1:mcp --port 3333
```

## 2. Call Tools with AIClient

The `AIClient` connects to one or more `XMCP` servers, lets an LLM decide which tools to call based on a user query, and returns the final answer.

This example is available in `examples/clients/demo.py`:

```python
import asyncio
from xmcp import AIClient
from xmcp.providers import OpenAIProvider # Or AnthropicProvider, etc.

# A dummy renderer for demonstration purposes
def render(display_data: dict):
    print("--- DISPLAY ---")
    print(display_data)
    print("-----------------")

async def main():
    # Make sure you have OPENAI_API_KEY set in your environment
    providers = [OpenAIProvider(model_id="gpt-4o")]
    
    # AIClient can connect to multiple servers
    async with AIClient(urls=["http://localhost:3333"], providers=providers) as ai:
        
        # The client needs a mutable history list to manage conversation state
        history = []
        
        result = await ai.chat("Show me sales for EMEA using a chart", history=history)
        
        if result.get("display"):
            render(result["display"])
        else:
            print(f"Assistant: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

To run the client demo:

```bash
python examples/clients/demo.py
```

You should see the `AIClient` call the `sales_summary` tool and the `render` function printing the display envelope. 

New in **v0.2**: servers can declare `depends_on` between tools, and the client runs them in parallel layers automatically.  The client is completely **stateless**; you must pass `history` each turn.

### 3. Ask the AI

If the LLM decides to call `current_time` **and** `timestamped_divide` in one turn, `AIClient` runs them in order thanks to the dependency graph and returns the final answer.

---

Need richer UX?  Pass `stream_chat()` instead and update your UI as chunks arrive.

```python
result = await ai.chat(
    "current_time then divide 10 by 2 and show_user=true",
    history=history,
)
``` 