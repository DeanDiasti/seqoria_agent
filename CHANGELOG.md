## [Unreleased]
### Added
- HeaderApiKeyBackend and StaticApiKeyProvider for lightweight header-based authentication.
- Helper migration: legacy FastAPI `Depends(get_current_user)` patterns can now be replaced with `server_populated=["user_id"]` on tools and an `AuthBackend` configured once per app/router.
- Planned improvements and bug fixes.
- Deprecation shims for legacy import paths (`seqoria_agent.server.fastxmcp.app`, `seqoria_agent.server.fastxmcp.router`, `seqoria_agent.toolbox.local`) now emit a single `DeprecationWarning` when imported.
- Comprehensive migration guide covering new server runtime paths (`docs/migration/xmcp_server.md`).
- New upgrade checklist and guidance embedded in docs.
- Tests relocated into scoped packages (`tests/xmcp_core/`) with coverage > 90 % for `xmcp_core`.
- Documentation updated with new canonical import paths; MkDocs reference pages rebuilt.
- Logging defaults tuned – diagnostic `.info()` calls in `xmcp_core` demoted to `.debug()`.

### Changed
- Phase 9 polish: lint/style clean-up, constant centralisation, updated docs, full 100 % coverage verified with pytest-cov, and quieter logging in `ai_client` stack.
- **0.2.0** – Server-side refactor complete (Phases 6-8):
  - Thin façade adapters introduced (`fastxmcp_public`, `local_public`).
  - Deprecation shims added; behaviour unchanged for existing imports.
  - API docs & quick-start snippets use the new canonical import paths.
  - Build pipeline now enforces coverage thresholds for core modules.

## [0.1.0] - 2025-06-21
### Added
- Initial public release of xmcp library, containing `AIClient`, `XMCP` server wrapper, `DisplayEnvelope`, dependency graph execution, streaming API, structured errors, hot-reloadable tool registry, multi-provider support, and a comprehensive test suite and documentation.

[Unreleased]: https://github.com/deandiasti/xmcp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/deandiasti/xmcp/releases/tag/v0.1.0 