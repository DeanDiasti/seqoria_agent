# seqoria_agent – Seqoria Agentic Assistant Framework

[![CI](https://github.com/deandiasti/seqoria_agent/actions/workflows/docs.yml/badge.svg)](https://github.com/deandiasti/seqoria_agent/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/seqoria_agent)](https://pypi.org/project/seqoria_agent/)

**seqoria_agent** (Seqoria Agentic Assistant Framework) is a next-generation orchestration layer for building robust, **tool-augmented AI** applications. It provides a unified protocol and a set of powerful components to seamlessly manage interactions between Large Language Models (LLMs), your custom tools, and rich user interfaces.

The core idea is to treat *tool calls, LLM reasoning, and UI display components* as first-class messages in a uniform stream, enabling complex, multi-turn conversations that are easy to build, fast to run, and beautiful to present.

---

## Core Concepts

seqoria_agent is built on three pillars:

1.  **`XMCP` Server**: A high-performance server powered by the *official* [MCP Python SDK](https://pypi.org/project/mcp/) (`mcp.server.fastmcp.FastMCP`). It exposes your Python functions as tools for an AI to use and adds critical features like structured error handling, custom output formatting, and a special decorator (`@display_tool`) for functions that generate UI components.

2.  **`AIClient`**: An intelligent, asynchronous client that acts as the brain of your application. It connects to one or more `XMCP` servers, manages the conversation state, invokes an LLM to decide on the next action, and executes the chosen tools—in parallel where possible. It's stateless, scalable, and packed with production-ready features like caching, retries, and streaming.

3.  **`DisplayEnvelope`**: A simple, frontend-agnostic JSON contract for sending rich UI components directly to the end-user. Instead of having an LLM describe a table or a chart in markdown, a tool can return a `DisplayEnvelope` containing the structured data, which your client application can then render as a native UI element.

## Key Features

-   **Unified Protocol**: A single, consistent architecture for LLM reasoning, tool execution, and rich UI generation.
-   **Pluggable LLM Providers**: Swap between LLM backends (e.g., OpenAI, Anthropic) without changing your application logic.
-   **Parallel Tool Execution**: The `AIClient` automatically runs independent tool calls concurrently to minimize latency.
-   **Response Streaming**: Stream LLM responses token-by-token and render UI components instantly as they arrive for a snappy user experience.
-   **Rich, Structured UI**: Go beyond markdown with `DisplayEnvelope` for rendering interactive charts, tables, images, and more.
-   **Production-Ready**: Built-in support for caching, automatic retries with exponential backoff, circuit breakers, and structured logging.
-   **Effortless Tool Definition**: Expose your existing Python functions as tools with simple decorators.

---

## Installation

> **Prerequisite**   `seqoria_agent` depends on the official **MCP SDK**. If you install `seqoria_agent` with `pip`, the correct `mcp[cli]` version is pulled in automatically—no extra steps required.

```bash
# Create & activate a virtual environment (optional, but recommended)

# Install the core library (brings in MCP automatically)
pip install seqoria_agent

# Development & docs extras
pip install 'seqoria_agent[dev,docs]'
```

---

## Quickstart

Here's a 2-minute example of creating a simple tool server and a client to interact with it.

### 1. Create a Server

Save the following as `my_server.py`. It defines a simple tool that can return either data for the LLM or a `DisplayEnvelope` for a UI.

```python
# my_server.py
from seqoria_agent.server.fastxmcp import FastXMCP as XMCP

mcp = XMCP()

@mcp.display_tool
def sales_summary(region: str, show_user: bool = False):
    """
    Returns a sales summary. If show_user is True, it returns a
    DisplayEnvelope for rendering a table. Otherwise, it returns
    raw data for the LLM.
    """
    # In a real app, you would fetch this from a database
    summary = {"region": region, "sales": 42000, "prospects": 120}

    if show_user:
        return DisplayEnvelope(
            type="table",
            payload=[summary],
            title=f"Sales Summary: {region}"
        )

    # For the LLM, just return the core data
    return {"region": region, "sales": summary["sales"]}

# This allows running the server with `uvicorn my_server:mcp`
```

Now, run the server from your terminal:

```bash
uvicorn my_server:mcp --port 8000
```

### 2. Create a Client

Save the following as `my_client.py`. This client will connect to your server and use an LLM to call the `sales_summary` tool.

```python
# my_client.py
import asyncio
import os
from seqoria_agent import AIClient
from seqoria_agent.providers import OpenAIProvider

# Dummy function to simulate rendering a UI component
def render(display_data: dict):
    print("\n--- Received UI Component---")
    import json
    print(json.dumps(display_data, indent=2))
    print("---------------------------\n")

async def main():
    # Requires OPENAI_API_KEY to be set in the environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        return

    providers = [OpenAIProvider(model_id="gpt-4o")]
    history = [] # The client is stateless, so we manage history

    async with AIClient(urls=["http://localhost:8000"], providers=providers) as ai:
        # The LLM will see the `show_user` parameter and set it to True
        result = await ai.chat(
            "Show me the sales summary for EMEA as a table",
            history=history
        )

        if result.get("display"):
            render(result["display"])
        else:
            print(f"Assistant: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Make sure your `OPENAI_API_KEY` is set, then run the client:

```bash
python my_client.py
```

You will see the `render` function print the structured `DisplayEnvelope` that the tool returned.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/deandiasti/seqoria_agent/issues). Please open an issue first to discuss any substantial changes.

## License

This project is licensed under the MIT License. 