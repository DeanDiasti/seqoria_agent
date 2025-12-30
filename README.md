# seqoria_agent – Seqoria Agentic Assistant Framework

[![CI](https://github.com/deandiasti/seqoria_agent/actions/workflows/docs.yml/badge.svg)](https://github.com/deandiasti/seqoria_agent/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/seqoria_agent)](https://pypi.org/project/seqoria_agent/)

**seqoria_agent** (Seqoria Agentic Assistant Framework) is a next-generation orchestration layer for building robust, **tool-augmented AI** applications. It provides a unified protocol and a set of powerful components to seamlessly manage interactions between Large Language Models (LLMs), your custom tools, and rich user interfaces.

The core idea is to treat *tool calls, LLM reasoning, and UI display components* as first-class messages in a uniform stream, enabling complex, multi-turn conversations that are easy to build, fast to run, and beautiful to present.

---

## Core Concepts

seqoria_agent is built on three pillars:

1.  **`SeqoriaMCP` Server**: A high-performance server built on the *official* [MCP Python SDK](https://pypi.org/project/mcp/) (`mcp.server.fastmcp.FastMCP`). It exposes your Python functions as tools for an AI to use and adds critical features like server-populated parameters, tool dependencies, and structured error handling.

2.  **`AIClient`**: An intelligent, asynchronous client that acts as the brain of your application. It connects to one or more MCP servers, manages the conversation state, invokes an LLM to decide on the next action, and executes the chosen tools—in parallel where possible. It's stateless, scalable, and packed with production-ready features like caching, retries, and streaming.

3.  **`LocalToolbox`**: Run tools directly in-process without an HTTP server. Perfect for simple scripts, testing, or when you want to combine local and remote tools in the same client.

## Key Features

-   **Unified Protocol**: A single, consistent architecture for LLM reasoning, tool execution, and rich UI generation.
-   **Pluggable LLM Providers**: Swap between LLM backends (OpenAI, Anthropic, Ollama) without changing your application logic.
-   **Parallel Tool Execution**: The `AIClient` automatically runs independent tool calls concurrently to minimize latency.
-   **Response Streaming**: Stream LLM responses token-by-token for a snappy user experience.
-   **Server-Populated Parameters**: Hide sensitive parameters (user IDs, tenant IDs) from the LLM and inject them at runtime.
-   **Production-Ready**: Built-in support for caching, automatic retries with exponential backoff, circuit breakers, and structured logging.
-   **Effortless Tool Definition**: Expose your existing Python functions as tools with simple decorators.

---

## Installation

```bash
# Create & activate a virtual environment (optional, but recommended)

# Install the core library (brings in MCP automatically)
pip install seqoria-agent

# Install with a specific LLM provider
pip install 'seqoria-agent[openai]'      # OpenAI
pip install 'seqoria-agent[anthropic]'   # Anthropic
pip install 'seqoria-agent[ollama]'      # Ollama
pip install 'seqoria-agent[providers]'   # All providers

# Development & docs extras
pip install 'seqoria-agent[dev,docs]'
```

---

## Quickstart

Here's a 2-minute example of creating a simple tool server and a client to interact with it.

### 1. Create a Server

Save the following as `my_server.py`. It defines a simple tool that the LLM can call.

```python
# my_server.py
from seqoria_agent.server import SeqoriaMCP

mcp = SeqoriaMCP("MyServer")

@mcp.tool()
def sales_summary(region: str) -> dict:
    """Returns a sales summary for the given region."""
    # In a real app, you would fetch this from a database
    return {"region": region, "sales": 42000, "prospects": 120}

if __name__ == "__main__":
    mcp.run(port=8000)
```

Now, run the server from your terminal:

```bash
python my_server.py
```

### 2. Create a Client

Save the following as `my_client.py`. This client will connect to your server and use an LLM to call the `sales_summary` tool.

```python
# my_client.py
import asyncio
import os
from seqoria_agent import AsyncAIClient
from seqoria_agent.providers import OpenAIProvider

async def main():
    # Requires OPENAI_API_KEY to be set in the environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.")
        return

    providers = [OpenAIProvider(model_id="gpt-4o")]
    history = []  # The client is stateless, so we manage history

    async with AsyncAIClient(
        mcp_servers=["http://localhost:8000/mcp/"],
        providers=providers
    ) as ai:
        async for chunk in ai.stream_chat(
            "What are the sales numbers for EMEA?",
            history=history
        ):
            print(chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())
```

Make sure your `OPENAI_API_KEY` is set, then run the client:

```bash
python my_client.py
```

---

## Using LocalToolbox (No Server Required)

For simpler use cases, you can run tools directly in-process:

```python
import asyncio
from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.providers import OllamaProvider

toolbox = LocalToolbox()

@toolbox.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@toolbox.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

async def main():
    providers = [OllamaProvider(model_id="llama3.2")]
    history = []

    async with AsyncAIClient(
        local_toolbox=toolbox,
        providers=providers
    ) as ai:
        async for chunk in ai.stream_chat(
            "What is 5 + 3, then multiply the result by 2?",
            history=history
        ):
            print(chunk, end="", flush=True)
        print()

asyncio.run(main())
```

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/deandiasti/seqoria_agent/issues). Please open an issue first to discuss any substantial changes.

## License

This project is licensed under the MIT License.

> **Tip**  Need full control?  The new *Manual Loop* helpers let you drive each
> LLM turn step-by-step – see the [guide](docs/guides/ai_client.md#manual-loop).
