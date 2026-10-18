# Tool execution and conversation history

`AsyncAIClient` accepts one or more asynchronous providers, `local_toolbox`, and `mcp_servers`. Enter the async context manager once at application startup; close it at shutdown. The application owns provider resources and closes them after all clients using them have stopped.

## Agent loops

`chat(query, history=history, context=context)` runs up to `max_turns` model/tool rounds. `stream_chat()` streams text deltas as well. Both append user messages, paired tool calls/results, and final responses to application-owned history. A turn-limit message is returned if the model keeps requesting tools. `max_turns` must be positive.

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

## Concurrency and cancellation

Independent tools run concurrently, with a default limit of 16 executions per client. Configure `max_concurrency` and `tool_timeout` for your workload. Dependencies declared with `depends_on` order requested tools and skip dependents after failures. They do not automatically call unrequested prerequisites or implement authorization.

`run_tool_calls()` preserves input order. `run_call_stream()` yields `(request, result)` pairs in completion order. Closing a tool stream cancels unfinished async calls; explicitly close streams that a consumer abandons early, for example with `contextlib.aclosing`.

Synchronous tools run in Python worker threads and should perform bounded work. A timeout or cancellation stops waiting but cannot kill an already-running thread. Use async I/O for cancellable operations and external workers for CPU-intensive work.

## Providers, retries, and configuration

Provider fallback occurs before output is emitted. A failed stream after its first output propagates the error to avoid replaying partial text or tool calls. Transport retries default to one attempt; enabling retries can replay side effects after ambiguous network failures. Use idempotent tools before enabling retries. MCP tool error responses are failures and are not retried or cached as success.

Prefer explicit `MCPServerConfig` values for transport settings. Unknown client keyword options are rejected. Tool blocklists and dynamic server changes are client-wide configuration, so do not use them as per-request permissions on a shared client. Schema aliases route to the original tool name; use unique tool names when integrating several servers to avoid ambiguity.
