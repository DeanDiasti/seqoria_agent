# Changelog

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

[Unreleased]: https://github.com/deandiasti/xmcp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/deandiasti/xmcp/releases/tag/v0.1.0 
