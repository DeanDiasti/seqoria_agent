# Tool execution and conversation history

`AsyncAIClient` accepts one or more asynchronous providers, `local_toolbox`, and `mcp_servers`. Enter the async context manager once at application startup; close it at shutdown. The application owns provider resources and closes them after all clients using them have stopped.

## Agent loops

`chat(query, history=history, context=context)` runs up to `max_turns` model/tool rounds. `stream_chat()` streams text deltas as well. Both append user messages, paired tool calls/results, and final responses to application-owned history. A turn-limit message is returned if the model keeps requesting tools. `max_turns` must be a positive integer.

History is a list containing message dictionaries or the exported Pydantic message models. Save models with `message.model_dump(mode="json")`; dictionaries can be passed back directly on the next turn. The library does not retain histories, compact context, infer plans, or write conversation files. Apply context limits and persistence in your application.

## Manual loops

```python
from seqoria_agent.models import AssistantMessage, ToolCallRequestList

response = await client.llm_generate_once(history, append_user_message="Hello")
if isinstance(response, ToolCallRequestList):
    results = await client.run_tool_calls(response.tool_calls, context=trusted_context)
    for request, result in zip(response.tool_calls, results, strict=True):
        history.extend([request, result])
elif isinstance(response, AssistantMessage):
    history.append(response)
```

The single-step helpers append an explicitly supplied user message but leave response recording to the caller. Repeat the step until an assistant response is returned, with your own round limit. `llm_stream_once()` is the streaming equivalent.

Tool failures, including missing server-populated identity parameters, return `ToolCallError` alongside successful `ToolCallResult` events. Inspect the result type to distinguish them. Missing identity fails only that call, lets independent tools finish, and skips its declared dependents. `ToolMessages` accepts both successes and failures, and both can be serialized into history.

## Hidden local parameters

Declare hidden local arguments with `@tools.tool(server_populated=["user_id"])`. The model does not see those fields. Pass authenticated values in `context` on each tool-executing request, or configure a `ParameterRegistry` once:

```python
from seqoria_agent.client.parameters import ParameterRegistry

async def user_id(principal):
    return principal.user_id

parameters = ParameterRegistry({"user_id": user_id})
# Set param_registry=parameters when constructing the client.
results = await client.run_tool_calls(calls, context={"principal": principal})
```

Request context takes precedence over registry sources. Registry values can be literals or sync/async callables; a callable receives matching context keys as keyword arguments. Sync callbacks run in worker threads. Missing values and callback failures fail that tool call without running it. Model-supplied hidden values are discarded by default; do not enable `allow_override` for identity or authorization fields.

There is one resolver in the client; toolbox callback registration is removed. A direct toolbox call supplies resolved, trusted values explicitly:

```python
result = await tools.call_tool("account_summary", {}, server_args={"user_id": principal.user_id})
```

`call_mcp_tool()` also accepts explicit local `server_args` and bypasses registry resolution. Use `run_tool_calls()` when you need request context and registry sources. Remote MCP servers resolve their own authenticated identity; local `server_args` are never transmitted remotely.

## Tool output

String results pass through as text. Local non-string results are JSON serialized; all MCP text blocks are joined in order, and structured-only MCP results become JSON text. JSON fields named `ai_message` or `metadata` and XML-looking strings have no special meaning. Interpret any application-specific envelope yourself after receiving the complete result.

Failures are explicit `ToolCallError` events with an `error` payload. Successes are `ToolCallResult` events with a `result` payload. MCP image/audio/resource content is not exposed as a typed multimodal result in this text-oriented tool loop.

## Concurrency and cancellation

Independent tools run concurrently, with a default limit of 16 executions per client. Configure `max_concurrency` and `tool_timeout` for your workload. The shared concurrency limit and execution timeout cover injected parameter resolution and tool invocation. The timeout begins after acquiring a slot, so queue time is excluded. Dependencies declared with `depends_on` order requested tools and skip dependents after failures. They do not automatically call unrequested prerequisites or implement authorization.

`run_tool_calls()` preserves input order. `run_call_stream()` yields `(request, result)` pairs in completion order. Closing a tool stream cancels unfinished async calls; explicitly close streams that a consumer abandons early, for example with `contextlib.aclosing`.

Synchronous tools run in Python worker threads and should perform bounded work. A timeout or cancellation stops waiting but cannot kill an already-running thread. Use async I/O for cancellable operations and external workers for CPU-intensive work.

## Providers, retries, and configuration

Provider fallback occurs before output is emitted. A failed stream after its first output propagates the error to avoid replaying partial text or tool calls. Transport retries default to one attempt; enabling retries can replay side effects after ambiguous network failures. Use idempotent tools before enabling retries. MCP tool error responses are failures and are not retried as successful operations.

Prefer explicit `MCPServerConfig` values for transport settings. Unknown client keyword options are rejected. Tool blocklists and dynamic server changes are client-wide configuration, so do not use them as per-request permissions on a shared client. Schema aliases route to the original tool name; use unique tool names when integrating several servers to avoid ambiguity.

Tool-result caching is not part of the current API. Every requested call executes, including repeated calls with identical arguments. Remove old `cache` and `cache_ttl` options; caching policies belong in application tools or services.

Manual `ToolCallRequest` instances require a nonempty `id` and `name`; omitted arguments default to `{}`. Results and errors require a nonempty `tool_call_id` and an explicit result/error payload. This prevents incomplete messages from entering provider histories.

Remote MCP identity annotations are not used for client-side parameter injection. A structured-only MCP result is converted to JSON text for the model. OAuth configuration is supported only for Streamable HTTP connections. Retry options are validated when the client is constructed, and circuit-breaker cooldown begins when the failure occurs.
