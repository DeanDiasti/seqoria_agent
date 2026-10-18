# Seqoria Agent

Async tool orchestration for Python web applications. Combine MCP servers and local Python tools with HTTP-based LLM providers. Python 3.12 or newer; alpha stability.

Start with the [quickstart](quickstart.md), then read the [web application guide](topics/multi_tenancy.md).

The supported core includes streaming and non-streaming agent loops, explicit conversation history, concurrent tool execution, tool dependencies, per-request local parameter injection, and remote MCP transports.

Version 0.7 removes automatic memory and planning, embeddings, vector/database adapters, and background capture. Applications own their context, authorization, storage, and retention. See the [release review](release-review.md) for scope and limitations.

The library connects to independently hosted MCP servers and executes in-process tools through `LocalToolbox`. It does not ship an MCP server wrapper.
