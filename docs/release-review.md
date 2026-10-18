# Release review for 0.7

The release scope is a small async orchestration library with application-owned history. This is an alpha API, with a breaking version change from 0.6.3 to 0.7.0.

## Feature decisions

| Area | Decision | Reason |
| --- | --- | --- |
| Async model/tool loops and streaming | Keep and repair | Core web application workflow |
| LocalToolbox and MCP server support | Keep and repair | In-process tools and standard remote interoperability |
| Explicit conversation lists | Supported history path | The application controls ownership, persistence, context size, and retention |
| Automatic memory, planning, graph retrieval, compression, embeddings, and database/vector adapters | Remove entirely | Unnecessary hidden state, background work, and incomplete behavior for the intended core |
| OpenAI, Anthropic, Ollama, OpenAI-compatible HTTP providers | Keep | Included through HTTPX without separate vendor SDKs |
| MLX local inference | Optional Apple Silicon extras | No platform-specific runtime in the core installation |
| Caching, retries, circuit breakers | Keep with explicit limits | Cache correctness and replay safety depend on the workload |
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
