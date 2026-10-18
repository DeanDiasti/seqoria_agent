# Changelog

## Unreleased

- Launch the public alpha with hosted documentation, protected main, private vulnerability reporting, refreshed release/security evidence, and an opt-in live provider check for ordinary and streaming tool roundtrips.

## 0.9.1 — 2026-10-18

- Preserve compatible streamed tool arguments and parallel Responses calls; reject malformed arguments, malformed success payloads, and incomplete streams; support multiline compatible SSE frames across sync/async adapters.
- Reject negative, fractional, and boolean provider retry budgets before opening HTTP clients.
- Redact endpoint credentials in library diagnostics, error events, direct provider exceptions, and unsafe exception chains while preserving safe causes and error subclasses.
- Raise the MCP minimum to 1.30.0, which includes the audited security fixes; retain the supported v1 API range.
- Remove obsolete planning artifacts from source archives; enforce formatting, import ordering, package-wide type checks, a 75% branch-coverage floor, archive checks, and installed-wheel regression tests. Ship the PEP 561 marker for consumers.
- Pin CI actions, add dependency and secret scans with dependency-update configuration, and retain documentation previews for review.
- Add a synthetic multi-user benchmark, contributor/support/security policies, issue templates, and concrete provider/parameter/MCP API references.
- Keep documentation deployment and repository visibility as separate reviewed launch steps.
- Organize guides and migration notes around the supported API, replace repetitive examples with a documented offline-first catalog, and group regression tests by behavior while preserving all existing cases. Exercise the self-contained examples in automated smoke tests.

## 0.9.0 — 2026-10-18

- Keep remote identity resolution on the server instead of injecting client context from remote annotations.
- Preserve structured-only MCP tool results as JSON text, including empty objects.
- Validate and snapshot retry/breaker settings; start breaker cooldown at failure time rather than request start.
- Close HTTP pools owned by low-level connection helpers on exit and startup failure; preserve caller-owned pools.
- Reject unsupported OAuth configurations on SSE/stdio instead of silently ignoring them.
- Require valid tool-message IDs, names, and result/error payloads; use independent argument defaults and clean up model documentation.
- Remove MLX text/vision providers, exports, factory entries, extras, and the local inference example.
- Consolidate hidden local parameter resolution on trusted request context and client `ParameterRegistry`; remove toolbox provider callbacks. Direct toolbox calls use explicit `server_args`.
- Preserve complete tool output instead of auto-unwrapping `ai_message`/`metadata`; remove legacy XML/envelope helpers and the executor's metadata callback.
- Remove SDK-style validation shims; test model discovery and Ollama pulls through real HTTPX transport paths.
- Replace the `urls` alias with `mcp_servers` in constructors/settings; remove unused `ClientSettings.client_kwargs`, global HTTP pool helpers, no-op connection close, old ownership bookkeeping, and the `ConnectionPool` alias.
- Remove the `ToolCallError.result` alias; provider converters read explicit error payloads. Document migration for all removed APIs.

## 0.8.0 — 2026-10-18

### Breaking changes

- Remove the tool-result cache layer, `cache`/`cache_ttl` options and settings fields, `CacheProtocol`/`DictCache`/`RedisCache`/`NullCache`, cache-key helpers, and Redis dependencies/extra. The executor no longer reuses cached results; applications own any required caching.
- Remove `SeqoriaMCP`, the `seqoria_agent.server` module, its formatting/parameter-provider APIs, and the server-only `XMCPError` exception. Host remote tools with an independent MCP server implementation.
- Keep MCP client transports, discovery, remote tool execution, and `LocalToolbox`. Migrate example servers and protocol integration tests to the official MCP SDK directly.

## 0.7.2 — 2026-10-18

- Publish the updated README on PyPI, covering supported providers, streaming and tool failures, multi-user integration, caching, MCP servers, and migration from 0.6.
- No runtime code changes from 0.7.1.

## 0.7.1 — 2026-10-18

- Keep failed tool calls in chat history as `ToolCallError` events, isolate missing identity parameters to their own tool call, and check dependency cycles for single calls.
- Preserve all text blocks in tool results and retain provider reasoning in chat messages. Deduplicate repeated schema aliases without changing their routes.
- Reject invalid concurrency limits, timeouts, refresh intervals, and turn limits before starting work.
- Close partial MCP startup, OAuth callback listeners, failed provider initialization, and cancelled HTTP streams. Fix the `connect_mcp()` context manager and HTTP pool shutdown races.
- Run synchronous parameter callbacks off the event loop, preserve OAuth headers, and repair principal fallback and tool registration validation.
- Repair empty-stream fallback and synchronous iterator cleanup during cancellation.
- Preserve failed-tool feedback and call IDs in all provider converters, including serialized histories; keep MLX VLM media conversion from overwriting later tool feedback.
- Accept tool errors and generator inputs in `ToolMessages`; validate list updates before applying them. Give basic messages valid role and usage defaults.
- Copy mutable in-process cache values, reject lossy Redis serialization and negative TTLs, detach copied settings containers, and omit endpoint credentials from settings representations.
- Include documentation and examples in source distributions so contributor instructions work from an extracted archive.
- Repair example streaming stubs, close provider resources, preserve chatbot history, remove misleading remote identity/replay configuration, and provide offline manual-step and token-usage demos.

## 0.7.0 — 2026-10-18

### Breaking changes

- Remove automatic memory/planning, embeddings, vector/database adapters, and memory-specific examples, extras, and `retrieval_query` parameters. Use application-owned history lists.
- Align package metadata and documentation with the existing Apache-2.0 license file.
- Reject unknown client options and sync providers passed directly to `AsyncAIClient`. Remote identity parameters must resolve on the authenticated server; explicit client `server_args` are local-only.
- Namespace cached results per client and full request context to isolate users and permissions.

### Fixes and release preparation

- Preserve tool request/result pairing across parallel completion and dependency scheduling; cancel unfinished async tools when a stream closes.
- Run synchronous tools in worker threads and generate nested/default/optional argument schemas using Pydantic.
- Refresh tool catalogues atomically and route schema aliases to their original names.
- Own MCP transport lifecycles in dedicated tasks and close client-owned HTTP pools at shutdown.
- Treat MCP error results as failures; fail over streaming providers before the first output only.
- Add provider resource cleanup and a working synchronous provider adapter. Fix settings copies and consumed server iterables.
- Bound the in-process cache with LRU eviction; use monotonic TTLs.
- Remove MLX from the core dependencies and remove database dependencies for deleted features.
- Add Python 3.12/3.13 CI, package smoke checks, verified API documentation, and tokenless PyPI Trusted Publishing configuration.

Earlier entries below describe historical releases and may refer to removed APIs.

## [0.1.0] - 2025-06-21
### Added
- Initial public release of xmcp library, containing `AIClient`, `XMCP` server wrapper, `DisplayEnvelope`, dependency graph execution, streaming API, structured errors, hot-reloadable tool registry, multi-provider support, and a comprehensive test suite and documentation.

## [0.4.0] - 2025-07-08
### Breaking Changes
* **ParameterProvider removed** – client code must migrate to the new dict-based `ParameterRegistry` with `ParamContext` support.
* AIClient constructor now accepts `param_registry=`; passing the removed `param_providers=` kwarg raises `TypeError`.

### Added
* `ParameterRegistry`, `ParameterSource`, `ParamContext` utilities.
* Parameter registry migration documentation (retired in 0.7).
* New `examples/param_demo/` demonstrating constant and callable parameter sources.

### Changed
* Executor and AIClient fully integrated with the registry; server-populated parameters are injected via async resolution.
* Version bumped to `0.4.0`.

[Unreleased]: https://github.com/DeanDiasti/seqoria_agent/compare/v0.9.1...HEAD
[0.1.0]: https://github.com/DeanDiasti/seqoria_agent/releases/tag/v0.1.0
