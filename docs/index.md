# Seqoria Agent

Build Python assistants that call local tools and remote MCP servers. Run model/tool loops, stream responses, and keep conversation state under your application's control. Python 3.12+; alpha API.

## Start building

1. Follow the [quickstart](quickstart.md) for a tool-calling assistant.
2. Read [tool execution](guides/tool-execution.md) for streaming, manual loops, trusted parameters, and failures.
3. Use the [web application guide](guides/web-applications.md) for shared client lifespans and independent conversations.
4. Connect independent servers using the [MCP guide](guides/mcp.md).

Runnable examples are grouped by task in the [example catalog](https://github.com/DeanDiasti/seqoria_agent/blob/main/examples/README.md). Provider, client, toolbox, message, and MCP transport details live in the API reference.

## Integration boundaries

The supported providers are OpenAI, OpenAI-compatible HTTP, Anthropic, and Ollama. HTTP adapters ship in the core package. Applications own authentication, authorization, conversation storage, context budgets, retention, rate limits, and any caching. Remote servers are hosted independently; `LocalToolbox` executes in-process tools.

The API may change during alpha. Consult the [migration guide](migration.md) before upgrading. Maintainer release and launch evidence is grouped separately under [release review](maintainers/release-review.md) and [public alpha review](maintainers/public-alpha.md).
