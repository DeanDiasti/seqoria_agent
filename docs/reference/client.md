# Client and parameter API

Start with [tool execution](../guides/tool-execution.md) for orchestration behavior and [web applications](../guides/web-applications.md) for resource ownership and per-request context. Constructor configuration is shared for the client lifespan; request identity belongs in trusted `context`.

## Async and blocking clients

::: seqoria_agent.client.ai_client.async_client.AsyncAIClient

::: seqoria_agent.client.ai_client.sync_client.AIClient

## Settings

::: seqoria_agent.client.settings.ClientSettings

## Hidden local parameters

`ParameterRegistry` is the single registry for hidden local arguments. Request context takes precedence. Sources can be literals or sync/async callables receiving matching context keys. A missing value or source failure prevents the affected tool from running. Direct `LocalToolbox.call_tool()` callers supply trusted `server_args` explicitly.

::: seqoria_agent.client.parameters.ParameterRegistry
