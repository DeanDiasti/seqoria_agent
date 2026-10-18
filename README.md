# Seqoria Agent

An async Python library for assistants that call Python tools and remote MCP servers. Run model/tool loops, stream responses, and keep conversation state under your application's control.

[PyPI](https://pypi.org/project/seqoria-agent/) · **Alpha API** · **Python 3.12+** · **Apache-2.0**

## What it does

- Streaming and non-streaming agent loops, plus single-step helpers for application-controlled loops.
- Local Python tools with Pydantic validation and hidden arguments from trusted request context.
- MCP client connections over Streamable HTTP, SSE, and stdio.
- Concurrent tool execution with shared limits, timeouts, dependency scheduling, and explicit failure events.
- OpenAI, OpenAI-compatible HTTP, Anthropic, and Ollama providers.
- Explicit conversation histories and provider fallback before streamed output begins.

The API is alpha and may change between releases. Pin a version and read the [migration guide](docs/migration.md) before upgrading.

## Install

```bash
pip install seqoria-agent
```

Import it as `seqoria_agent`. HTTP providers ship in the core package through HTTPX; vendor SDKs, model runtimes, and databases are not required. Optional extras are `performance` for JSON/hash accelerators, `server` for FastAPI integration, and `all` for both. Contributor tools use `dev,docs`.

## A tool-calling assistant

Set `OPENAI_API_KEY` and `OPENAI_MODEL` to credentials and a model ID available to your account. This example makes hosted model calls:

```python
import asyncio
import os

from seqoria_agent import AsyncAIClient, LocalToolbox
from seqoria_agent.models import AssistantMessage
from seqoria_agent.providers import AsyncOpenAIProvider

tools = LocalToolbox()


@tools.tool()
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main():
    history = []
    async with AsyncOpenAIProvider(model_id=os.environ["OPENAI_MODEL"]) as provider:
        async with AsyncAIClient(providers=[provider], local_toolbox=tools) as client:
            async for event in client.chat("What is 2 + 3?", history=history):
                if isinstance(event, AssistantMessage):
                    print(event.content)


if __name__ == "__main__":
    asyncio.run(main())
```

Reuse `history` for another turn in the same conversation. The client appends the user message, paired tool calls/results, and final response. Providers belong to the application; close clients before closing their providers.

For a checkout or extracted source distribution, run `python examples/quickstart/offline.py` after installing the package. It needs no API key or service. The [example catalog](examples/README.md) covers tools, conversation state, MCP, and provider setup.

## Choose a provider

Import from `seqoria_agent.providers`:

| Backend | Async provider | Service required |
| --- | --- | --- |
| OpenAI | `AsyncOpenAIProvider` | OpenAI API and model access |
| OpenAI-compatible | `AsyncOpenAICompatibleProvider` | Compatible HTTP endpoint |
| Anthropic | `AsyncAnthropicProvider` | Anthropic API and model access |
| Ollama | `AsyncOllamaProvider` | Running Ollama service and model |

Use async providers with `AsyncAIClient`. Blocking scripts use `AIClient` as a regular context manager and `chat_sync()` for the final answer; provider classes without the `Async` prefix are synchronous variants. Endpoint capabilities depend on the backend. See the [provider API](docs/reference/providers.md) for constructors and methods.

## Use it in a web application

Create one `AsyncAIClient` per worker and fixed tool/authentication configuration during application startup. Reuse its connections across independent conversations, give every conversation its own history, and serialize simultaneous turns within one conversation. Multiple workers need storage transactions or distributed coordination.

Pass authenticated identity in per-request `context`. Declare hidden local arguments with `@tools.tool(server_populated=["tenant_id", "user_id"])`; use `ParameterRegistry` for trusted derived values. Hidden identity never comes from model output. Remote MCP sessions with different end-user credentials require separate clients/sessions, and remote servers enforce their own authorization.

The default tool concurrency cap is 16 per client. `tool_timeout` covers parameter resolution and tool execution after a slot is acquired; add application limits for total request time and queue depth. Synchronous tools run in worker threads and cannot be stopped once running. Close abandoned async streams explicitly.

Applications own authentication, authorization, history persistence, context limits, retention, rate limits, and side-effect safety. The library executes each requested call without caching results; retries can repeat operations after ambiguous failures. See the [web application guide](docs/guides/web-applications.md) for a lifespan example and request ownership rules.

## Read the guides

- [Quickstart](docs/quickstart.md): first assistant and streaming events.
- [Tool execution](docs/guides/tool-execution.md): manual loops, trusted parameters, failures, and cancellation.
- [MCP connections](docs/guides/mcp.md): transports, authentication, and session ownership.
- [API reference](docs/reference/client.md): clients, parameters, local tools, messages, providers, and transport helpers.
- [Migration](docs/migration.md): upgrade steps for removed memory, caches, server wrappers, and compatibility APIs.

Tool output stays intact as text/JSON, including fields named `ai_message` or `metadata`. Failures produce `ToolCallError` events. Streaming additionally yields `ChatChunk` deltas; the final `AssistantMessage` contains the complete response and usage. See the execution guide before building a stream renderer or retry policy.

Documentation is available from source and with `mkdocs serve` after installing `.[docs]`. Hosted documentation deployment is a separate launch step; a successful build does not imply a live site.

## Contribute and get help

[CONTRIBUTING.md](CONTRIBUTING.md) covers setup, the organized test suite, formatting, type checks, package validation, and pull requests. Tests use deterministic HTTP transports, in-memory MCP sessions, and a local subprocess; the default suite needs no hosted model or model downloads. CI covers Python 3.12/3.13 on Linux, macOS, and Windows.

Use [GitHub issues](https://github.com/DeanDiasti/seqoria_agent/issues) for ordinary reports and follow [SECURITY.md](SECURITY.md) for vulnerabilities. See [SUPPORT.md](SUPPORT.md) and the [code of conduct](CODE_OF_CONDUCT.md). Issues require access while this repository remains private during launch review.

Maintainer evidence is grouped under [release review](docs/maintainers/release-review.md) and [public alpha review](docs/maintainers/public-alpha.md). Review these before making the repository public or publishing the next release. See [RELEASING.md](RELEASING.md) and [CHANGELOG.md](CHANGELOG.md) for release operations and history.

## License

Apache-2.0. See [LICENSE.txt](LICENSE.txt).
