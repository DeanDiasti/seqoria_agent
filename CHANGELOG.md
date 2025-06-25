## [Unreleased]
### Added
- HeaderApiKeyBackend and StaticApiKeyProvider for lightweight header-based authentication.
- Helper migration: legacy FastAPI `Depends(get_current_user)` patterns can now be replaced with `server_populated=["user_id"]` on tools and an `AuthBackend` configured once per app/router.
- Planned improvements and bug fixes.

### Changed
- Phase 9 polish: lint/style clean-up, constant centralisation, updated docs, full 100 % coverage verified with pytest-cov, and quieter logging in `ai_client` stack.

## [0.1.0] - 2025-06-21
### Added
- Initial public release of xmcp library, containing `AIClient`, `XMCP` server wrapper, `DisplayEnvelope`, dependency graph execution, streaming API, structured errors, hot-reloadable tool registry, multi-provider support, and a comprehensive test suite and documentation.

[Unreleased]: https://github.com/deandiasti/xmcp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/deandiasti/xmcp/releases/tag/v0.1.0 