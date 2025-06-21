# Seqoria Agent

## Overview

**XMCP** is a next-generation orchestration layer that builds on
[FastMCP](https://github.com/illacloud/fastmcp) to make **tool-augmented AI**
applications easier to author, faster to run, and richer to present. The core
idea is to treat *tool calls, LLM reasoning, and user-facing displays* as first-class
messages in a uniform protocol that any client (CLI, web, mobile) can speak.

The project consists of three pillars:

1. **Server runtime (`XMCP`) —** a thin wrapper around FastMCP that adds
   specialised decorator (`@display_tool`) and automatically
   formats every response as an XML string + optional `DisplayEnvelope` for
   rich UI rendering.
2. **Python client (`AIClient`) —** an async helper that connects to one or more
   XMCP/FastMCP servers, lets an LLM decide which tools to call, and returns
   both the final textual answer *and* any structured display payload.
3. **Display protocol (`DisplayEnvelope`) —** a minimal, self-describing JSON
   contract that front-ends can map to charts, tables, markdown, images, etc.

On top of these building blocks we will add features such as stateless
conversation loops, parallel tool execution, and server-generated parameters
for sensitive inputs—all detailed further below.

> **Why another layer?** While FastMCP excels at defining individual tools, it
> doesn't prescribe how to stitch multiple calls together or how to send rich
> UI artefacts straight to the user. XMCP fills that gap without locking you
> into any front-end framework.

---

## ✨ Key features

1. **`@display_tool`** – executes an action and sends a rich result straight to
the UI without a second LLM round-trip.
2. **Front-end agnostic `DisplayEnvelope`** – a lightweight JSON contract any
   renderer (CLI, web, mobile) can interpret.
3. **`AIClient`** – automatically merges tools from multiple MCP servers and
   adds native support for the new display channel.

---

## Installation

```bash
pip install -e .  # Or poetry install, etc.
```


---

## Quick-start

### 1. Create an XMCP server

```python
from xmcp import XMCP, DisplayEnvelope

mcp = XMCP()

@mcp.display_tool
def greet(name: str) -> DisplayEnvelope:
    return DisplayEnvelope(type="markdown", payload=f"### Hello {name}! 👋")

@mcp.display_tool
def sales_summary(region: str, show_user: bool = False):
    # Heavy lifting …
    summary = {"region": region, "sales": 42}
    if show_user:
        # Construct something fancy for the user
        return DisplayEnvelope(type="table", payload=[summary])
    return summary  # Normal JSON for the LLM

mcp.serve()
```

### 2. Call the tools from an AI client

```python
import asyncio
from xmcp import AIClient

async def main():
    async with AIClient(["http://localhost:3333"], openai_api_key="sk-…") as ai:
        result = await ai.chat("Show me sales for EMEA using a chart")
        if result["display"]:
            render(result["display"])  # Your UI-layer code
        else:
            print(result["answer"])

asyncio.run(main())
```

---

## DisplayEnvelope specification

| Field   | Type                               | Description                               |
|---------|------------------------------------|-------------------------------------------|
| type    | `"chart" \| "table" \| …`          | Semantic category (drives renderer).      |
| payload | `dict \| list \| str`             | The actual data to render.                |
| title   | `str` *(optional)*                | Human-readable heading.                   |
| meta    | `dict` *(optional)*               | Extra info (dimensions, theme, …).        |

Renderers should branch on `type` and ignore unknown keys gracefully.

---

## Project status / roadmap

| Milestone                                | Status |
|------------------------------------------|--------|
| Baseline FastMCP wrappers              | ✅ Done |
| `@display_tool` decorator              | ✅ Done |
| DisplayEnvelope schema                 | ✅ Done |
| AIClient display integration           | ✅ Done |
| End-to-end example                     | ⚠️ *In progress* |
| Front-end reference renderer           | ⏳ Planned |
| Extensive unit tests                   | 🧪 Prototype |
| XML formatter parity (XMCP)             | 🔬 Design |
| Streaming response API                 | ✅ Done |
| Structured logging & observability     | ✅ Done |
| Envelope strict-mode validation (XMCP) | ✅ Done |
| Auto `show_user` injection             | 🔬 Design |
| Structured error channel (XMCP & AIClient) | ✅ Done |
| Dependency graph execution (AIClient)    | ✅ Done |
| Multi-LLM provider layer (AIClient)      | ✅ Done |
| Internal meta-tools (`__think__`, `__summarize__`, `__clarify__`) | ✅ Done |
| Server-populated parameter middleware    | ✅ Done |

*Last updated: 2025-06-20*

---

## Contributing

PRs and bug reports are welcome! Please open an issue first if your change is
substantial so we can discuss the design.

---

## Advanced Concepts & Upcoming Enhancements

Below you'll find **in-depth design notes** for the next wave of XMCP features. Each block follows the same template—*Intuition ➜ Implementation ➜ Benefit*—and includes illustrative code.

### 1. Stateless Conversation History (AI Client) *(updated)*

**Intuition**  
Share a single `AIClient` across many parallel chat sessions (web tabs, API tenants) without risk of state bleed-over.

**Implementation**
```python
history: list[dict] = []   # caller-managed
while True:
    user_msg = input("You › ")
    history.append({"role": "user", "content": user_msg})
    result = await ai.chat(user_msg, history)  # history is *required*
    history = result["history"]               # caller keeps the state
    print(result["answer"] or "[display payload sent]")
```
Key points:
* `AIClient.chat()` will soon **require** `history`; omitting it raises `ValueError`.
* Returned history contains *all* tool and LLM messages—persist or trim as you see fit.

**Benefit**  
Predictable memory usage, easy sharding, and clearer separation between transport (client) and dialogue state (caller).

*Status: Implemented in v0.2 – AIClient no longer retains any state after each `chat()` / `stream_chat()` call. Calls are re-entrant-safe and will raise `RuntimeError` if invoked concurrently with the same instance.*

### 2. Iterative Multi-Tool Feedback Loops (AI Client) *(updated)*

**Intuition**  
Allow the LLM to chain tool invocations until it resolves the user request—e.g. "What's the weather in the capital of Canada?" ➜ lookup capital ➜ lookup weather.

**Implementation Sketch**
```python
MAX_TURNS = 5
for turn in range(MAX_TURNS):
    llm_msg = await _call_llm(history)
    if not llm_msg.tool_calls:
        return llm_msg.content           # done ✅

    # Serial execution in declared order
    for call in llm_msg.tool_calls:
        tool_result = await _run_tool(call)
        history.append(_wrap_tool_msg(call, tool_result))
```
The loop exits when the LLM returns a plain message *or* when `MAX_TURNS` is hit (safety guard).

**Benefit**  
Enables richer tasks without exposing orchestration complexity to end-users.

---

### 3. Parallel Tool Execution (AI Client) *(updated)*

**Intuition**  
If the LLM asks for multiple independent tools in the *same* turn, run them concurrently to slash latency.

**Implementation Snippet**
```python
if llm_msg.tool_calls:
    coros = [_run_tool(tc) for tc in llm_msg.tool_calls]
    results = await asyncio.gather(*coros, return_exceptions=True)
    for call, res in zip(llm_msg.tool_calls, results):
        history.append(_wrap_tool_msg(call, res))
```
Order is preserved to keep `tool_call_id` alignment.

**Benefit**  
Better UX for dashboards & queries that fan-out to multiple data sources.

*Implementation status:*

> The `AIClient` now runs tool calls concurrently via `asyncio.gather`,
> preserving the original order when writing back to conversation history.
> Future improvements: expose a *concurrency limit* and refine merged error
> handling for partially failing tool batches.

---

### 4. XML-Formatted AI Responses (XMCP)

**Intuition**  
XML is streaming-friendly and easy to pattern-match in LLM prompts. `agentic/response_formatter.py` already emits:
```xml
<tool_response tool_name="get_weather">
  …
</tool_response>
```

**Implementation**  
XMCP exposes a single transport contract: **every response from any `/mcp` endpoint** is wrapped in an XML envelope *before* it is forwarded to the LLM *and* to the client SDK.  

```xml
<tool_response tool_name="get_weather">
  <llm_output>It is 18 °C and sunny in Ottawa.</llm_output>
  <display>
    {"type": "markdown", "payload": "**18 °C** and sunny ☀️"}
  </display>
</tool_response>
```

Key points:
* **Dual-channel** – The `<llm_output>` node contains the natural-language answer for the user/LM, while the optional `<display>` node carries the `DisplayEnvelope` JSON for rich UI rendering.
* **String-only to the LLM** – The entire XML blob is delivered as a *single string* in the chat `content` field, allowing prompt-side XPath-style parsing while avoiding JSON escaping issues.
* **Client unwrapping** – `AIClient` parses the XML, extracts `llm_output` and `display`, and routes them to the `answer` and `display` keys of its return dict.

**Benefit**  
• Deterministic, regex-friendly parsing for the LLM.  
• One response object covers both narrative and visual channels.  
• Transport remains text-only, enabling easy streaming.

#### 4.1 Custom XML Formatters

**Why?**  
While the default wrapper will happily turn any dict / list into XML, a hand-picked subset of fields keeps the LLM prompt short and highly relevant.

**How to use**

1. **Define a formatter** by subclassing `DeclarativeXMLFormatter` (or write any callable that accepts `(data, indent_level)` and returns an XML string):

```python
from xmcp.xml_formatters import DeclarativeXMLFormatter

class WeatherXMLFormatter(DeclarativeXMLFormatter):
    # Only these keys will appear in the XML that the LLM sees
    top_level_fields = ["city", "temperature", "condition"]
```

2. **Bind the formatter** to a tool via the decorator:

```python
@mcp.tool(xml_formatter=WeatherXMLFormatter)
# or equivalently
# @mcp.tool(annotations={"xml_formatter": "WeatherXMLFormatter"})

def get_weather(city: str) -> dict:
    return {
        "city": city,
        "temperature": 14,
        "condition": "Cloudy",
        "humidity": 57,          # Will be hidden from the LLM
    }
```

3. **Result delivered to the LLM** (wrapped by XMCP):

```xml
<tool_response tool_name="get_weather">
  <llm_output>
    <![CDATA[
      <result>
        <city>London</city>
        <temperature>14</temperature>
        <condition>Cloudy</condition>
      </result>
    ]]>
  </llm_output>
</tool_response>
```

**Rules & Fallbacks**

* If no `xml_formatter` is provided, XMCP invokes a recursive *default formatter* that serialises every key/value it finds.
* `@display_tool` functions still behave the same: the `<display>` node carries the `DisplayEnvelope`, while `<llm_output>` holds the XML generated by the formatter.
* Advanced builders can override `results_list_field`, `result_item_fields`, etc., or provide a fully custom function for exotic schemas.

#### 4.2 Structured Error Channel

**Intuition**  
Tool failures today show up as opaque text. By wrapping them in a typed `<error>` element we give both the LLM *and* the surrounding app enough structure to react intelligently – retry, fallback, or surface a clean message to the user.

**Implementation (server – `XMCP`)**
1. Wrap every `@tool` invocation in `try/except`.  
2. On exception, build and return an envelope instead of propagating the stack-trace:

```xml
<tool_response tool_name="run_query">
  <error code="DB_TIMEOUT" retryable="true">
    <![CDATA[database timeout after 5 s]]>
  </error>
</tool_response>
```

Attribute semantics  
• `code` – short, upper-snake identifier ("VALIDATION_FAIL", "DB_TIMEOUT", "RATE_LIMIT", …).  
• `retryable` – "true" / "false" hint for the client.  
• CDATA body – human-readable context or stack trace.  
• Optional `<meta>` child containing JSON for advanced tooling.

**Implementation (client – `AIClient`)**
1. Extend XML parser to detect `<error …>` nodes.  
2. If `retryable="true"`, retry the tool (respecting the existing *retry_policy* and *breaker* limits).  
3. Otherwise append a `{"role": "tool", "name": tool_name, "content": "<error …/>"}` message so the LLM can decide how to proceed.  
4. Emit `on_error(envelope, history)` callback for host applications (logging / Sentry etc.).

**Benefit**  
• LLM gets structured signals and can gracefully handle predictable failures.  
• DevOps gain machine-countable error metrics.  
• Users see friendly, localized messages instead of raw traces.

**Example**
```python
@mcp.tool
def run_query(sql: str):
    try:
        return _execute(sql)
    except DBTimeout as exc:
        raise XMCPError(code="DB_TIMEOUT", retryable=True, detail=str(exc))
```

---

#### 4.3 Dependency-Graph Execution

**Intuition**  
Complex tasks often require a chain: *generate SQL* ➜ *run SQL* ➜ *render chart*. Declaring these relationships lets the client execute workflows in the minimal number of rounds while preserving parallelism where possible.

**Implementation**
*Server side* – annotate dependencies once:
```python
@mcp.tool()
def generate_sql(question: str) -> str: ...

@mcp.tool(annotations={"depends_on": ["generate_sql"]})
def run_query(sql: str): ...

@mcp.display_tool(annotations={"depends_on": ["run_query"]})
def render_chart(rows: list[dict], show_user: bool = False): ...
```

*Client side (`AIClient`)*
1. Build a dependency graph from the `depends_on` annotation during `_update_llm_tools()`.  
2. When the LLM asks for multiple tools in one turn, topologically sort them into **layers** that can run concurrently.  
3. Use existing `asyncio.gather()` per layer; move to the next layer only after all predecessors succeed.  
4. Inject predecessor outputs into successors when their parameter names match (e.g. `sql` or `rows`).  
5. Detect cycles at start-up and raise `DependencyError`.

**Benefit**  
• Drastically fewer LLM ↔ tool round-trips.  
• Authors can write small, composable tools without manual orchestration glue.  
• Errors in an early layer cancel the rest, saving wasted work.

**Example call-graph (auto-derived)**
```
Layer 0 : generate_sql
Layer 1 : run_query
Layer 2 : render_chart, summarize_results
```

---

#### 4.4 Multi-LLM Provider Layer

**Intuition**  
Relying solely on OpenAI limits cost-optimisation and on-prem use-cases. A thin provider interface lets us swap back-ends (Anthropic, Mistral, Llama 3 local) without changing application code.

**Implementation**
1. Define a protocol:
```python
class LLMProvider(Protocol):
    async def chat_complete(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> OpenAIStyleResponse: ...
```
2. Ship adapters: `OpenAIProvider` (default), `AnthropicProvider`, `MistralProvider`, `LocalProvider` (Ollama / vLLM).  
3. Accept `llm_provider` param in `AIClient` *or* infer from legacy kwargs:
```python
ai = AIClient(urls, llm_provider=AnthropicProvider(api_key="…"), fallback_models=["claude-3-haiku"])
```
4. Fallback list becomes list[tuple[provider, model]], preserving current retry semantics.  
5. Optional capability introspection: `provider.supports_tools`, `provider.max_context_tokens`, so `AIClient` can auto-downgrade behaviour or chunk prompts.

**Benefit**  
• One orchestration layer works across vendor APIs and on-prem clusters.  
• Teams can route sensitive data to private models, bulk queries to cheaper models, and premium requests to GPT-4-x.  
• Future provider quirks are isolated behind their adapter.

**Example**
```python
from xmcp.providers import AnthropicProvider

aiclient = AIClient(
    urls=[...],
    llm_provider=AnthropicProvider(api_key="sk-claude-…"),
    fallback_models=[("OpenAIProvider", "gpt-3.5-turbo")],
)
```

---

### 5. Standardised Display Outputs

**Intuition**  
Front-ends crave structural guarantees. `DisplayEnvelope` is our single source of truth.

**Implementation Requirements**
* Every `@display_tool` **must** return a `DisplayEnvelope` (object or dict).
* Display tools **must** obey the caller's `show_user` boolean flag.
* XMCP will validate the envelope and raise `ToolError` on mismatch.

Example:
```python
@mcp.display_tool
def open_link(url: str) -> DisplayEnvelope:
    return DisplayEnvelope(type="notification", payload="Opened in your browser")
```

**Benefit**  
Uniform rendering across CLI, web, and mobile clients.

---

### 6. Display Tool Toggle

**Intuition**  
Sometimes the user wants a chart *and* a narrative; other times narrative only.

**Implementation**
The decorator auto-injects a field:
```python
class Params(BaseModel):
    query: str
    show_user: bool = False
```
Tool authors simply use the param:
```python
@mcp.display_tool
def sales(query: str, show_user: bool = False):
    data = _calc(query)
    if show_user:
        return DisplayEnvelope(type="table", payload=data)
    return data
```

**Benefit**  
Fine-grained UX control without duplicating logic.

---

### 7. Server-Populated Parameters

**Intuition**  
Shield the LLM from sensitive or context-heavy arguments (e.g. authenticated user ID, geolocation, org-level API keys).

**Implementation Pipeline**
1. Mark schema fields with `json_schema_extra={"server_populated": True}` (see `agentic/schemas/server_schemas.py`).
2. XMCP pre-execution hook injects real values:
   ```python
   for name, field in params.model_fields.items():
       if field.json_schema_extra.get("server_populated"):
           kwargs[name] = _lookup(name)
   ```
3. Any conflicting value supplied by the LLM is **overridden** (or rejected).
4. The final value is echoed in the XML so the LLM can reason with it downstream.

**Benefit**  
Security & correctness: sensitive data never traverses the LLM boundary, yet the model still "knows" the value post-injection.

---

### 8. Internal Meta-Tools for Reasoning & Flow Control *(updated)*

**Intuition**  
Equip the LLM with lightweight, *client-side only* helpers that improve reasoning flow without exposing chain-of-thought or polluting the user interface.

| Tool | Purpose |
|------|---------|
| `__think__` | Record an internal thought between tool calls. **Not** shown to the user. |
| `__summarize__` | Compress the evolving conversation into a concise recap to save tokens. |
| `__clarify__` | Ask the user a single follow-up question when instructions are ambiguous. |

**Implementation Sketch**
```python
THINK_TOOL_NAME = "__think__"
SUMMARIZE_TOOL_NAME = "__summarize__"
CLARIFY_TOOL_NAME = "__clarify__"

# 1. Expose the meta-tools to the LLM (no server-side counterpart)
llm_tools += [
    {
        "type": "function",
        "function": {
            "name": THINK_TOOL_NAME,
            "description": "Internal reasoning step",
            "parameters": {
                "type": "object",
                "properties": {"thought": {"type": "string"}},
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": SUMMARIZE_TOOL_NAME,
            "description": "Compress the conversation so far",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": CLARIFY_TOOL_NAME,
            "description": "Ask the user a clarifying question",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
]

# 2. Intercept them inside AIClient._run_tool()
if tool_name == THINK_TOOL_NAME:
    history.append({"role": "assistant", "content": arguments["thought"]})
elif tool_name == SUMMARIZE_TOOL_NAME:
    history = _compress(history)
elif tool_name == CLARIFY_TOOL_NAME:
    return (None, None, {
        "answer": arguments["question"],
        "display": None,
        "history": None,
    })
```

**Benefit**  
• Keeps chain-of-thought internal yet available for debugging.  
• Prevents context bloat via on-demand summarisation.  
• Lets the model gather missing details instead of guessing, improving accuracy.

*Note:* Early termination now happens implicitly when the model returns a normal assistant message with **no** `tool_calls`, so the dedicated `__stop__` tool is no longer required.

---

### 9. Structured Logging & Observability *(updated)*

**Intuition**  
Expose a pluggable logger so host applications decide where log events go – terminal, JSON lines, APM, etc.

**API Sketch**
```python
ai = AIClient(
    urls=[...],
    logger=structlog.get_logger("xmcp").bind(session_id=session_id),
)
```
If omitted, `AIClient` falls back to `logging.getLogger("xmcp")`.  All current
`print()` calls become structured `logger.debug/info/warning` events and carry
context fields (turn, tool_name, latency).

**Benefit**  
Production-grade diagnostics with zero coupling to a particular backend.

---

### 10. Adaptive Concurrency & Time-outs

**Intuition**  
Allow callers to cap the number of simultaneous tool invocations and define a per-tool deadline to avoid hung requests.

**API Sketch**
```python
ai = AIClient(
    urls=[...],
    max_concurrency=10,     # semaphore limit
    tool_timeout=8.0,       # seconds
)
```

**Implementation Notes**  
Wrap each `call_mcp_tool` in `asyncio.wait_for()` and guard the task set with an `asyncio.Semaphore(max_concurrency)`.

---

### 11. Retry Policy & Circuit-Breaker

**Intuition**  
Transient network hiccups shouldn't bubble up to users; repeated failures should temporarily quarantine a bad server.

**API Sketch**
```python
ai = AIClient(
    urls=[...],
    retry_policy=dict(max_attempts=3, backoff_base=0.5),
    breaker_threshold=5,
    breaker_reset_after=60,
)
```

**Implementation Notes**  
Exponential back-off for retries; maintain a failure counter per server and raise `CircuitOpenError` while tripped.

---

### 12. Model Fallback & Custom Base URL *(updated)*

**Intuition**  
Gracefully degrade to cheaper / less-busy models or custom Azure/OpenAI endpoints.

**API Sketch**
```python
ai = AIClient(
    urls=[...],
    llm_model="gpt-4o",
    fallback_models=["gpt-4o-mini", "gpt-3.5-turbo"],
    openai_base_url="https://my-proxy/v1",
)
```

**Implementation Notes**  
Loop through `fallback_models` inside `_call_llm` when recoverable exceptions occur.

---

### 13. Streaming Response API *(updated)*

**Intuition**  
Emit tokens and display payloads incrementally for snappier UX.

**API Sketch**
```python
async for chunk in ai.stream_chat(query, history):
    if chunk.type == "llm_delta":
        ui.append_text(chunk.text)
    elif chunk.type == "tool_display":
        ui.render(chunk.envelope)
```

**Implementation Notes**  
Leverage `stream=True` in the OpenAI SDK, interleave tool execution results, and yield structured `ChatChunk` objects.

---

### 14. Display Callback Hook

**Intuition**  
Give host applications an immediate, synchronous way to render or handle a `DisplayEnvelope` before `chat()` finishes.

**API Sketch**
```python
ai = AIClient(
    urls=[...],
    on_display=lambda env, hist: ui.render(env),
)
```

**Implementation Notes**  
Invoke the callback inside `_run_tool` right before returning an early-exit payload.

---

### 15. Token Accounting with Persistent Storage  *(updated)*

Enhanced from the original idea: usage stats can be automatically persisted via a user-supplied handler.

**API Sketch**
```python
ai = AIClient(
    urls=[...],
    usage_sink=lambda usage: db.insert("llm_usage", usage),
)
```

`usage` dict contains `input_tokens`, `output_tokens`, `cost_usd`, `timestamp`, `session_id`.

---

### 16. Pluggable Cache Object  *(updated)*

**Intuition**  
Avoid redundant calls to deterministic tools by injecting any cache backend implementing a simple interface.

**API Sketch**
```python
ai = AIClient(
    urls=[...],
    cache=RedisCache(url="redis://localhost:6379", ttl=300),
)

class CacheProtocol(Protocol):
    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any, ttl: int): ...
```

If `cache` is `None`, no caching is performed.

**Benefit**  
Transparent performance boost without coupling to a particular datastore.

---

### 17. Hot-Reload of Tool Registry

**Intuition**  
In long-lived processes new tools may appear on the MCP servers after the
client has started.  Rather than forcing a restart, let `AIClient` refresh its
tool catalogue on a schedule or via an explicit call.

**API Sketch**
```python
ai = AIClient(
    urls=[...],
    refresh_interval=300,          # seconds; 0 disables auto-refresh
)

# Or imperatively when you know the server changed
await ai.refresh_tools()
```

**Implementation Notes**
1.  Keep the existing `_update_llm_tools()` logic but expose it as a public
    coroutine `refresh_tools()`.
2.  If `refresh_interval` > 0, spawn an `asyncio.Task` in `__aenter__` that
    sleeps and calls `refresh_tools()` periodically.  Cancel the task in
    `close()`.
3.  The refresher merges new tools into `_llm_tools` while preserving existing
    ones so in-flight conversations don't lose references.

**Benefit**  
Zero-downtime deployments of new tools and faster developer iteration.

---

### 18. Multi-Tenant Optimisation

**Intuition**  
Run a single long-lived `AIClient` instance that safely serves many users (tenants) while keeping strict isolation, quotas and cost attribution.

**Key Concepts**
1. **Session-scoped history** – callers are responsible for passing an isolated `history` list per chat.  The client never stores cross-session state.
2. **Tenant context injection** – every public call accepts a `tenant_id` (str/UUID).  Middlewares, cache keys and usage accounting include this value.
3. **Per-tenant quotas** – middleware can throttle token usage, parallel tool calls or monthly spend based on `tenant_id`.
4. **Namespaced cache** – the pluggable cache object must prepend `tenant_id` to its keys to avoid answer leakage.
5. **Usage accounting sink** – the `usage_sink` receives `tenant_id`, enabling roll-ups by customer.

**API Sketch**
```python
ai = AIClient(urls=[...], cache=RedisCache(...), usage_sink=usage_db.log)

# web-request handler
async def handle(request):
    tenant_id = request.headers["X-Tenant"]
    history   = request.session.get("history", [])

    result = await ai.chat(
        user_query=request.json["msg"],
        history=history,
        tenant_id=tenant_id,            # new kwarg propagated through hooks
    )
    request.session["history"] = result["history"]
    return JSONResponse(result)
```

**Implementation Notes**
• Add optional `tenant_id` param to `chat()` and propagate into all middleware/event-bus hooks.  
• Default quota middleware example:
```python
class TenantQuota(Middleware):
    async def before_llm(self, state):
        if quota_db.exceeded(state.tenant_id):
            raise QuotaExceeded()
```

**Benefit**  
Zero duplication of heavy resources (tool list, HTTP sessions) while maintaining strict logical isolation and usage governance per customer.

---

## Updated Roadmap

| Milestone                                | Status |
|------------------------------------------|--------|
| Stateless history (AIClient)             | ✅ Done |
| Serial multi-tool loop (AIClient)        | ✅ Done |
| Parallel tool execution (AIClient)       | ✅ Done |
| XML formatter parity (XMCP)              | ✅ Done |
| Envelope strict-mode validation (XMCP)   | ✅ Done |
| Auto `show_user` injection               | 🔬 Design |
| Structured logging & observability       | ✅ Done |
| Adaptive concurrency & time-outs         | ✅ Done |
| Retry policy & circuit-breaker           | ✅ Done |
| Model fallback & custom base URL         | ✅ Done |
| Streaming response API                   | ✅ Done |
| Display callback hook                    | ✅ Done |
| Token accounting + persistence           | 🧪 Prototype |
| Pluggable cache object                   | ✅ Done |
| Hot-reload tool registry                 | ✅ Done |
| Multi-tenant optimisation                | 🧪 Prototype |
| Structured error channel (XMCP & AIClient)| ✅ Done |
| Dependency graph execution (AIClient)    | ✅ Done |
| Multi-LLM provider layer (AIClient)      | ✅ Done |
| Internal meta-tools (`__think__`, `__summarize__`, `__clarify__`) | ✅ Done |
| Server-populated parameter middleware    | ✅ Done |
| Front-end reference renderer             | ⏳ Planned |
| Extensive unit tests                     | 🧪 Prototype |

*Last updated: 2025-06-20*
