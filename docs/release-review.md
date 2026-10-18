# Release review

The release scope is a small async orchestration library with application-owned history. The API remains alpha; 0.8.0 removes server authoring and built-in tool-result caching while preserving MCP client interoperability and local tools.

## Cleanup in 0.9.0

Version 0.9.0 includes the following repairs and removals. The review uses source inspection, in-memory MCP protocol sessions, real stdio lifetimes, and HTTP transport mocks. Hosted services and production throughput remain application integration checks.

### Confirmed defects repaired in 0.9.0

- Remote `server_populated` annotations no longer populate arguments from client context. Identity resolution stays on the remote server; local tool injection remains supported.
- MCP results containing only `structuredContent` retain their JSON payload, including an empty object, rather than becoming a generic success message.
- Circuit-breaker cooldown starts at failure time, including after slow requests. Retry settings are copied and validated; zero/fractional attempts, non-finite delays, unknown options, and invalid breaker settings fail during construction.
- Low-level HTTP connections close their own pool after normal exit or failed initialization. A caller-supplied pool retains its caller-owned lifetime. This removes the default helper's dependency on a global async singleton.
- OAuth on SSE or stdio is rejected explicitly; previously the setting was silently ignored. The implemented OAuth flow is Streamable HTTP only.
- Tool messages require nonempty call IDs and request names, and results/errors require their payload. Request arguments default to independent dictionaries. Previously omitted fields produced `None` despite non-optional types, breaking serialization/validation round trips. Unused `slots` configuration and misleading message docstrings were cleaned up.

### Approved removals in 0.9.0

| Area | Change | Verification |
| --- | --- | --- |
| MLX adapters | Removed both provider modules, lazy exports, factory entries, extras, example, and adapter-specific tests. The retained provider set is OpenAI, OpenAI-compatible HTTP, Anthropic, and Ollama. | Package import/factory regressions and built-wheel metadata checks. |
| Competing parameter providers | Removed `LocalToolbox.register_param_provider` and its fallback resolver. Clients resolve hidden fields from trusted request `context` and `ParameterRegistry`; direct tools accept explicit `server_args`. | Concurrent identities stay separate with context, mapping sources, and registry objects; forged/missing identity never executes the tool. |
| Legacy XML and envelopes | Removed XML parsing, unused formatting helpers/constants, automatic `ai_message` extraction, and the executor's metadata callback. Tool JSON/text remains intact. | Round trips preserve nested/null `ai_message`, metadata, other JSON fields, malformed JSON, and XML-looking text. |
| Vendor SDK compatibility | Removed SDK-style model-list/pull branches and model-object parsing. Providers validate through HTTPX only. | Sync and async HTTP transport tests cover model presence, missing models, rate limits, Ollama pulls, and pull failures. |
| Configuration and ownership aliases | Replaced `urls` with `mcp_servers` in settings, constructors, examples, and tests. Removed rejected `ClientSettings.client_kwargs`, the `ConnectionPool` alias, global HTTP pool helpers, no-op `MCPConnection.close()`, and old context-manager bookkeeping. Removed ineffective Pydantic slots settings. | Explicit settings precedence/copy isolation, startup failure cleanup, owner-task teardown, dynamic removal, and unknown-option checks. |
| Error payload alias | Removed `ToolCallError.result`; maintained provider converters read `.error` explicitly, including serialized histories. | All retained provider failed-tool history regressions pass. |

The retained orchestration, HTTP providers, local tools, and MCP client remain the supported core. Sync and async HTTP providers still share wire conversion through mixins while maintaining separate I/O methods; no vendor SDK shims remain. Further deduplication can be considered separately from this API cleanup. Tool output remains text/JSON-oriented; a typed multimodal MCP result API would need a separate design. No live hosted-provider or throughput claim follows from the regression suite.

## Scope of 0.8.0

Version 0.8.0 removes the library-owned `SeqoriaMCP` wrapper, `seqoria_agent.server`, server-only formatting and parameter-provider APIs, and `XMCPError`. MCP client transports/discovery/execution and `LocalToolbox` remain supported. Existing server code must migrate to an independent server implementation. The built-in tool-result cache is also removed: there are no cache backends, cache client/settings options, key-building helpers, or Redis dependency/extra. The executor no longer reuses cached tool results. Future caching design belongs to a separate review; applications can own caching within their data services. The review and verification notes below describe the historical 0.7.0/0.7.1 releases.

## Feature decisions in 0.7.0

| Area | Decision | Reason |
| --- | --- | --- |
| Async model/tool loops and streaming | Keep and repair | Core web application workflow |
| LocalToolbox and MCP support | Keep client and local tools; remove the server wrapper in 0.8.0 | In-process execution and remote interoperability without owning a server implementation |
| Explicit conversation lists | Supported history path | The application controls ownership, persistence, context size, and retention |
| Automatic memory, planning, graph retrieval, compression, embeddings, and database/vector adapters | Remove entirely | Unnecessary hidden state, background work, and incomplete behavior for the intended core |
| OpenAI, Anthropic, Ollama, OpenAI-compatible HTTP providers | Keep | Included through HTTPX without separate vendor SDKs |
| MLX local inference | Optional Apple Silicon extras | No platform-specific runtime in the core installation |
| Tool-result caching | Removed in 0.8.0 | Replay safety, ownership, and invalidation need an application-specific design |
| Retries and circuit breakers | Keep with explicit limits | Replay safety depends on the workload |
| Rich UI / DisplayEnvelope, old FastXMCP APIs, speculative roadmap docs | Retire stale documentation | These are not implemented public APIs in this codebase |

## Confirmed defects fixed

- Parallel results were returned in completion order and zipped against input requests, corrupting tool-call history pairing.
- Abandoned tool streams left unfinished async tool tasks running.
- Sync local/server tools executed on the event loop, blocking other users' requests.
- Cache keys could omit user identity and permissions within one tenant; separate client configurations could share entries.
- Tool discovery cleared live routes before network discovery completed; schema aliases did not route back to original tool names.
- MCP transport context managers could be entered and exited in different tasks, violating the SDK's task-group lifetime. Client-owned HTTP resources were not released.
- MCP tool error results were treated as successful text and could be cached as success.
- Stream fallback tried to catch failures while constructing an async generator, before provider I/O actually ran.
- Server parameter injection did not request MCP context for functions without an explicit `ctx` parameter.
- Local schemas lost nested and optional type information. Synchronous provider adaptation assumed an attribute the providers did not implement.
- Settings copies could mutate original provider lists, and generator-based server additions/removals consumed their inputs twice.
- Unbounded MCP dependencies installed incompatible MCP 2 in a fresh environment; constrain MCP to the FastMCP-compatible v1 line.
- Core dependencies included MLX; CI used Python 3.11 despite a 3.12 minimum; documentation referenced deleted APIs and nonexistent provider extras; license metadata said MIT while the existing license file is Apache-2.0.

Regression tests cover concurrent histories, result ordering, cancellation, sync tool responsiveness, cache isolation, atomic registry refresh, schema aliases, provider fallback, a real MCP protocol session, and a real stdio subprocess lifespan.

## Follow-up audit for 0.7.1

A second code audit found additional defects in the retained features. These repairs are recorded under **0.7.1** in the changelog; they are not part of the 0.7.0 artifacts on PyPI.

| Area | Additional repairs |
| --- | --- |
| Tool orchestration | Accept failed tool events in message groups; isolate missing identity parameters; reject single-call dependency cycles; preserve all textual result blocks; deduplicate schema aliases without overwriting routes |
| Runtime limits | Reject fractional concurrency caps, non-finite timeouts/refresh intervals, and invalid turn limits before history changes; include parameter resolution and cache I/O in execution limits after queueing |
| Providers | Preserve reasoning, failed-tool text and call IDs in both model and saved histories; mark Anthropic errors; preserve later tool feedback during MLX VLM media conversion; fall back from empty streams before output; close HTTP resources on initialization failure and cancellation; serialize blocking iterator cleanup |
| MCP and parameters | Close partially established sessions on cancelled startup and unused OAuth listeners; fix the convenience context manager and shutdown race; forward OAuth headers; run synchronous parameter callbacks in threads; repair principal fallback and reject invalid tool registrations |
| Messages and configuration | Preserve generator inputs and tool errors, validate message-list mutations atomically, repair role/usage defaults, detach copied settings containers, and avoid credentials in settings representations |
| Caches | Copy mutable in-process values, reject negative TTLs and lossy Redis string fallback, and expire entries at their deadline |
| Examples and packaging | Repair streaming stubs and provider lifespans, retain chatbot history, simplify trusted local parameter injection, remove misleading random-result caching and remote identity configuration, provide offline manual/token demos, and include documentation/examples in source archives |

Regression coverage uses deterministic providers, mocked HTTP/Redis transports, in-memory MCP protocol sessions, and the existing real stdio subprocess test. Hosted model services, external Redis, and production OAuth remain deployment-specific integration checks. Blocking synchronous I/O still requires a finite underlying timeout; cancellation cannot stop an executing Python thread.

Local verification for this follow-up: 300 tests on Python 3.12, including 97 new regression cases; Ruff and whitespace checks; strict documentation build; wheel/source builds and strict metadata validation. The suite also passes against an installed wheel outside the source checkout with only core and test dependencies. Six examples run offline, and four example streaming stubs were checked without contacting services.

## Release boundaries

The library does not provide authentication middleware, per-user remote session storage, distributed conversation locking, durable job execution, context summarization, a database, or application rate limiting. The application must implement these where needed.

Hosted provider calls, external Redis behavior, Apple Silicon inference, and production HTTP/OAuth deployments require environment-specific integration testing. The local suite does not measure production throughput. The 0.7.0 release passed CI on Python 3.12/3.13 on Linux, macOS, and Windows and was verified by installing from PyPI. Follow-up changes need their own CI run after pushing.

The release workflow requires a PyPI account with verified email, two-factor authentication, project ownership or an available name, and a configured trusted publisher. Local code changes do not register a project or publish a release. See the repository's `RELEASING.md` for the concrete setup values.
