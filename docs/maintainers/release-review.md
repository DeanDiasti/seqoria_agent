# Release review

Seqoria Agent is an alpha orchestration library for Python web applications. The supported core is explicit conversation history, local tools, MCP client connections, and HTTP model providers. Version 0.9.1 contains the reviewed provider hardening, diagnostic redaction, packaging checks, and documentation cleanup on top of 0.9.0. Publication is verified through the tagged release workflow and PyPI.

## Public alpha hardening

The launch review focuses on real provider wire behavior, isolation across simultaneous users, cancellation and resource ownership, safe diagnostics, reproducible package checks, and accurate public documentation. See the [public alpha review](public-alpha.md) for the launch gate and integration limits.

Regression tests use HTTPX mock transports, deterministic responses, in-memory MCP sessions, and a real stdio subprocess. They exercise supported protocol shapes without requiring paid services or downloading models. Coverage reports identify gaps; a percentage alone is not a correctness guarantee.

## Supported behavior after 0.9.0

| Area | Supported contract |
| --- | --- |
| Conversation state | Application-owned lists, including serialized Pydantic messages; one writer per conversation |
| Providers | Async and sync OpenAI, OpenAI-compatible HTTP, Anthropic, and Ollama adapters through HTTPX |
| Local identity | Trusted per-request `context` and `ParameterRegistry`; direct toolbox calls pass explicit `server_args` |
| Remote identity | Authentication and identity resolution on the remote MCP server; no client-side hidden-identity injection |
| Tool output | Complete text/JSON retained; `ai_message`, `metadata`, and XML-looking strings have no implicit envelope semantics |
| Failure events | Explicit `ToolCallError.error` and `ToolCallResult.result` payloads with valid tool-call IDs |
| Scheduling | Shared per-client concurrency cap, execution timeout after queueing, dependency ordering, cancellation of unfinished async calls |
| Ownership | Application-owned providers; client-owned MCP sessions and default HTTP pools; caller-supplied pools remain caller-owned |
| Retries | Validated, explicit retry limits; circuit-breaker cooldown measured from failure time; application must assess replay safety |

Structured-only MCP results become JSON text, including empty objects. MCP tool output is text-oriented; typed image/audio/resource result handling needs a separate API design. OAuth is supported only with Streamable HTTP. Unknown client options fail early; configure transports with `MCPServerConfig`.

## Historical removals

These are migration history, not features available in the current package:

- **0.7.0:** removed automatic memory/planning, embeddings, background capture, graph/vector/database adapters, and their extras. Use application-owned history and context preparation.
- **0.8.0:** removed `SeqoriaMCP`, server authoring helpers, `XMCPError`, and built-in tool-result cache backends/options/Redis extras. Host servers independently and own any caching in application services.
- **0.9.0:** removed MLX adapters/extras, duplicate toolbox parameter callbacks, XML/envelope machinery, vendor-SDK validation shims, `urls`, rejected settings fields, global HTTP pool helpers, no-op connection close, and error/connection aliases. Use HTTP providers, `ParameterRegistry`, `mcp_servers`, and explicit resource contexts.

The earlier reviews repaired tool-call/result pairing, abandoned stream tasks, blocking synchronous execution, atomic discovery, route aliases, MCP task ownership, provider fallback, history conversion, settings copies, and startup failure cleanup. Historical cache or MLX repairs do not imply those removed components remain supported. See the [migration guide](../migration.md) for upgrade steps and `CHANGELOG.md` in the source distribution for detailed release notes.

## Application integration boundaries

The library does not provide authentication middleware, distributed conversation locking, per-user remote session storage, durable jobs, context summarization, a database, or application rate limiting. Configure those at the application boundary.

Separate remote MCP sessions when end-user authentication differs. Apply authorization inside every operation. Bound request bodies, queue depth, total request time, history size, and underlying I/O. A synchronous tool already running in a worker thread continues after its caller is cancelled; use async I/O for cancellable operations.

Hosted provider credentials, model-specific capabilities, real network failures, and production HTTP/OAuth deployments need environment-specific integration testing. Synthetic concurrency measurements isolate orchestration overhead; they are not a production capacity claim.

Repository visibility, private vulnerability reporting, Pages activation, and release publication are separate operator actions. The maintainer approved the public alpha launch after the 0.9.1 release; the [launch record](public-alpha.md) includes verification evidence and live integration limits.
