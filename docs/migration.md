# Migration guide

The current API is alpha. Pin a package version and review these changes before upgrading. Historical persisted data remains application-owned; upgrading the library does not migrate or delete it.

## Migrating from 0.8 to 0.9

Version 0.9.0 removes MLX adapters and extras, toolbox parameter-provider callbacks, XML helpers, automatic `ai_message`/`metadata` envelopes, and the executor's `on_metadata` callback. Use HTTP providers, request `context` and `ParameterRegistry`, and application-owned display interpretation. Direct toolbox calls pass resolved hidden fields in `server_args`.

Replace `urls` with `mcp_servers` in constructors and `ClientSettings`. Remove `ClientSettings.client_kwargs`; use `MCPServerConfig` for transport options. Read `ToolCallError.error` instead of its removed `result` alias. Use `MCPConnectionPool` instead of `ConnectionPool`. Enter transport/pool context managers and close explicitly owned HTTP pools; `MCPConnection.close()` and `HTTPClientPool.get_shared()`/`close_shared()` are removed. Provider internals accept HTTPX clients only, with no vendor-SDK validation shims.

## Migrating from 0.7 to 0.8

`SeqoriaMCP`, `seqoria_agent.server`, and the server-only `XMCPError` exception have been removed in 0.8.0. Migrate server construction, authentication, parameter injection, and formatting to an independent MCP server implementation. Client MCP configuration and local tools remain supported.

The built-in tool-result caching layer is also removed. Drop `cache` and `cache_ttl` arguments, `ClientSettings` cache fields, imports from `seqoria_agent.client.cache`, and the `redis` extra. `DictCache`, `RedisCache`, `NullCache`, `CacheProtocol`, and `build_cache_key` are no longer provided. Implement any required caching in application-owned tools/services with explicit authorization, invalidation, and side-effect rules. Existing Redis data is application-owned and is not deleted by this removal.

## Migrating from 0.6

`AgenticMemory`, configuration models, automatic planning, background capture, embeddings, vector/database adapters, and memory-specific examples/extras have been removed. Replace memory objects with a per-conversation list. Remove `retrieval_query` arguments. Load and save conversation history in your own authorized application storage; serialize model instances with `model_dump(mode="json")` before storage.

Automatic summaries and retrieval are no longer performed. If needed, build context explicitly in the application before calling the client. Existing persisted memory files are application data; export needed conversation messages before migrating. The library does not delete or migrate those files.

For remote MCP servers, hidden parameters resolve on the server from its authenticated context. Never trust arbitrary identity arguments sent by remote callers. Local tools receive trusted per-request values through the executor's `context`.
