# Building a Manual Agent with Seqoria Agent

This guide walks you through **creating an interactive, fully-featured CLI assistant** that uses the low-level `AIClient` helpers instead of the high-level `chat()` convenience wrapper.  By rolling your own "manual loop" you gain **fine-grained control** over every LLM/tool turn, which is ideal for:

* Terminal-based REPLs and chat-bots.
* Tight integration into existing TUI / GUI applications.
* Educational demos that expose the internal reasoning and tool-usage flow.

The reference implementation is [`examples/manual_loop/manual_agent.py`](../../examples/manual_loop/manual_agent.py).  This guide elaborates on each part and shows how to extend it.

---

## 1. Prerequisites

1. **Python ≥ 3.8**
2. `seqoria_agent` (install from the project root):

   ```bash
   pip install -e .[dev]
   ```
3. An **LLM provider key**.  The example uses OpenAI, so export:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

4. *(Optional)* one or more **FastMCP servers** with tools you want to call.  You may run the public-router demo (see `examples/routers`), or omit servers entirely for LLM-only/manual-toolbox mode.

---

## 2. Minimal skeleton

```python
import asyncio, os, random, datetime as _dt
from typing import List

from seqoria_agent.client import AIClient
from seqoria_agent.providers import OpenAIProvider
from seqoria_agent.toolbox import LocalToolbox

# ---------------------------------------------------------------------
# 1) Define your inline tools
# ---------------------------------------------------------------------
box = LocalToolbox()

@box.tool(description="Return the current timestamp (ISO-8601 string)")
def get_time() -> str:
    return _dt.datetime.now().isoformat()

@box.tool(description="Return a random integer between 0 and 100")
def random_number() -> int:
    return random.randint(0, 100)

# ---------------------------------------------------------------------
# 2) Main REPL loop
# ---------------------------------------------------------------------
async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY – aborting."); return

    providers = [OpenAIProvider(model_id="gpt-4o")]
    async with AIClient(urls=[], providers=providers, local_toolbox=box) as ai:
        history: List[dict] = []     # shared mutable conversation state
        MAX_ROUNDS = 5
        for turn in range(1, MAX_ROUNDS + 1):
            try:
                user_query = input(f"You [{turn}/{MAX_ROUNDS}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGood-bye!"); break
            if not user_query:
                print("Exiting – empty input detected."); break

            # 1️⃣ system + user message
            ai.prepare_history(user_query, history)

            # 2️⃣ inner agent loop (max. 5 LLM/tool turns per user input)
            for _ in range(5):
                resp = await ai.llm_generate_once(history)
                ai.append_assistant_message(resp.text or "", history, tool_calls=resp.tool_calls)

                if resp.tool_calls:
                    early = await ai.run_tool_calls(resp.tool_calls, history)
                    if early and early.get("answer"):
                        print("Assistant:", early["answer"]); break
                    continue  # tool executed, let LLM continue

                # direct answer – finished this round
                print("Assistant:", resp.text); break

if __name__ == "__main__":
    asyncio.run(main())
```

### What just happened?

| Step | API call | Purpose |
|------|----------|---------|
| 1️⃣  | `prepare_history()` | Ensures the system prompt is present and appends the user's question. |
| 2️⃣  | `llm_generate_once()` | Performs **one** LLM call (optionally with tool pick). |
|      | `append_assistant_message()` | Adds the assistant delta to history (and serialises `tool_calls`). |
|      | `run_tool_calls()` | Executes pending tool calls **with dependency-resolution, caching, retries, meta-tool handling, callbacks, etc.** |

Repeat 2️⃣ until the model either returns a direct answer *or* a tool produced an early-exit payload.

---

## 3. Anatomy of the manual loop

### LocalToolbox

* Decorate Python callables with `@box.tool`.
* Optional keywords:
  * `server_params=["tenant_id"]` – server-populated arguments.
  * `depends_on=["other_tool"]` – declare explicit dependencies.

### AIClient construction

| Parameter | Description |
|-----------|-------------|
| `urls` | List of FastMCP server URLs.  Pass `[]` for offline/local-only mode. |
| `providers` | One or more `BaseProvider` subclasses (e.g. `OpenAIProvider`). |
| `local_toolbox` | A `LocalToolbox` instance or list thereof. |
| `refresh_interval` | Auto-refresh remote tool catalogue (seconds). |
| `max_concurrency` | Semaphore limit for concurrent tool execution. |
| `tool_timeout` | Per-tool hard timeout (seconds). |
| `param_registry` | `ParameterRegistry` mapping for server-populated params. |
| `enable_meta_tools` | `True` / `False` / list – controls `__think__`, `__clarify__`, `__summarize__`. |

### Meta-tools in a manual loop

The executor automatically intercepts three synthetic tools:

* `__think__` – captures hidden reasoning.
* `__clarify__` – yields an immediate question for the user (**early exit**).
* `__summarize__` – compresses history so context remains short.

You don't need extra code – they behave like regular tools.

### Callbacks

Pass callables to `AIClient`:

```python
ai = AIClient(
    ..., on_error=my_error_hook,
               on_metadata=my_meta_hook)
```

`Executor` guarantees callbacks never raise into your loop; errors are logged.

---

## 4. Adding remote tools

```python
urls = ["http://localhost:8000", "https://tools.mycorp.ai"]
ai = AIClient(urls=urls, providers=providers, local_toolbox=box)
```

The **ToolRegistry** fetches `/tools/list` from each server, merges them with
local definitions, and provides the resulting OpenAI tool schema back to the
LLM.  Dependencies, display annotations, etc. are parsed automatically.

> If a remote tool shares its name with a local tool, `AIClient` raises a
> `RuntimeError` to avoid ambiguity.

---

## 5. Streaming variant

For progressive rendering:

```python
async for chunk in ai.stream_chat("plot me a graph of …", history=[]):
    if chunk.type == "llm_delta":
        print(chunk.text, end="", flush=True)
    # handle other chunk types here
```

`ChatChunk` types:

* `llm_delta` – incremental text.

The streaming helper automatically handles meta-tools and tool execution in the
background.

---

## 6. Caching & parameter injection

* **Cache** – supply a `CacheProtocol` implementation (e.g. Redis).  The helper
  `build_cache_key()` is used internally.
* **ParameterRegistry** – populate sensitive fields (like `user_id`) without
  exposing them to the model.  Each provider gets `(param_name, tool_name, context)`.

---

## 7. Error handling & retries

* Timeouts, retries, and an exponential back-off circuit breaker are applied by
  `ConnectionPool.call_tool()`.
* Structured XML errors (`<error code="…" retryable="…">…`) are parsed and
  forwarded to `on_error`.

---

## 8. Testing your agent

```bash
pytest -q tests/ai_client
```

The test-suite exercises all low-level helpers, including duplicate-tool
scenarios like the one fixed in **v0.3.5**.

---

## 9. Next steps

* Add domain-specific tools (`@box.tool`) or deploy a custom FastMCP backend.
* Integrate the loop into `prompt_toolkit` or similar for rich terminal UIs.
* Experiment with `enable_meta_tools=False` and handle summarisation yourself.
