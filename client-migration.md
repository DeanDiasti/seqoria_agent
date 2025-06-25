# AI Client Refactor & Migration Guide

This document tracks the **step-by-step migration** of the monolithic `seqoria_agent/client/client.py` (≈1 350 LoC) into a set of focused modules under `seqoria_agent/client/ai_client/`.

The goals are:

* Apply the *Single Responsibility Principle* – every component handles one concern only.
* Reduce cognitive load: smaller files and shorter methods.
* Expose a stable public façade (`seqoria_agent.client.AIClient`) while hiding internal complexity.
* Unlock finer-grained tests and faster iteration for new features (rate limiting, observability, …).

The migration is deliberately incremental; **after each phase the full unit-test suite must be green** so regressions surface early.

---

## Current state (post-PR 1 – constants & utils extracted)

| Module | Responsibilities |
|--------|------------------|
| `client.py` | Everything except two helpers already moved (`constants.py`, `utils.py`). |
| `ai_client/constants.py` | Synthetic meta-tool names. |
| `ai_client/utils.py` | `null_async_cm`, `extract_error_from_xml` (will soon get more helpers). |

All tests pass. API surface unchanged.

---

## Migration phases

### Phase 0 – Ground rules  (✅ already agreed)
* Keep **public import path** `seqoria_agent.client.AIClient` stable.
* After every phase run `pytest -q`; commit only when green.
* No functional changes while cutting; refactor first, polish later.

---

### Phase 1 – ConnectionPool extraction

🔑 *Scope*  Move all connection / retry / circuit-breaker code from `client.py` into `ai_client/connection_pool.py`.

**What to move**
1. `_connect_to_server` coroutine.
2. Breaker state dicts: `_fail_log`, `_breaker_until` and config knobs (`_breaker_threshold`, `_breaker_reset_after`).
3. Retry & exponential-backoff loop inside `call_mcp_tool`.
4. The `call_mcp_tool` public coroutine itself.

**Steps**
1. Create the file & scaffold `class ConnectionPool` with a minimal API:
   ```python
   async def start(self): ...
   async def close(self): ...
   async def call_tool(self, name, arguments, server_args=None): ...
   @property def has_servers(self): ...
   ```
2. Hoist the logic listed above; inject `SimpleHTTPTransport` via constructor.
3. Update `client.py` so `_run_tool` uses `self.pool.call_tool(...)` instead of the now-removed internal method/state.
4. Add `tests/test_connection_pool.py` covering success, retry, and breaker trip.

**Acceptance criteria**
* `pytest -q` passes.
* `ConnectionPool` <= 250 LoC, no logging or caching unrelated code.

---

### Phase 2 – ToolRegistry extraction

🔑 *Scope*  Everything related to **discovering & refreshing tools**.

**What to move**
* `_update_llm_tools` coroutine.
* Periodic refresher task logic.
* Dicts currently stored on `AIClient`: `_server_params`, `_display_tools`, `_tool_deps`, `_llm_tools`.

**Public API**
```python
class ToolRegistry:
    async def refresh(self): ...
    @property def llm_tools(self): ...  # list for OpenAI
    def is_display_tool(self, name): ...
    def server_params(self, name): ...
    def depends_on(self, name): ...
```

**Steps**
1. Create `ai_client/tool_registry.py` and port logic.
2. Inject local toolboxes + ConnectionPool into ctor so registry can call `list_tools()`.
3. Replace direct dict access in `client.py` with registry method calls.
4. Unit tests: merging behaviour, duplicate handling, annotation parsing.

---

### Phase 3 – Executor extraction

🔑 *Scope*  Responsible for **running tool calls** once the LLM has decided.

**Includes**
* `_run_tool` coroutine (bulk of the work).
* Semaphore (`_tool_semaphore`), timeout handling, cache look-ups.
* Meta-tool handling (`__think__`, `__summarize__`, `__clarify__`).
* Early-exit logic for display tools.
* `_compute_execution_layers` (dependency graph).

**Public API**
```python
class Executor:
    async def run_calls(self, tool_calls, history) -> dict|None  # Early-exit payload
```

**Steps**
1. Port code to `ai_client/executor.py`; constructor receives ToolRegistry, ConnectionPool, Cache, callbacks, settings.
2. Delete duplicated helpers from `client.py` – leave thin wrapper.
3. Add focused tests: cache hit/miss, early-exit, dependency ordering, meta-tools.

---

### Phase 4 – LLMInterface extraction

🔑 *Scope*  All **provider interaction** (sync + streaming).

**What to move**
* `_call_llm` and `stream_chat` provider loop.

**Public API**
```python
class LLMInterface:
    async def generate(self, history, tools) -> LLMResponse
    async def stream(self, history, tools) -> AsyncGenerator[ChatChunk]
```

**Steps**
1. Implement in `ai_client/llm_interface.py`.
2. Constructor takes list[BaseProvider] plus logger.
3. Update `client.py` to delegate.
4. New tests for provider fall-back & streaming deltas.

---

### Phase 5 – Helpers & callbacks

* `ai_client/history.py` – system-prompt helper, summariser stub.
* `ai_client/callbacks.py` – thin wrappers so Executor can invoke `on_display` / `on_error` uniformly.
* Move `build_cache_key` into `ai_client/utils.py`.

Mostly mechanical; small or no extra tests (behaviour already covered).

---

### Phase 6 – `ai_client/base.py` orchestrator

Create a **new implementation** of `AIClient` that composes the extracted sub-modules:
```python
class AIClient:
    def __init__(...):
        self.pool      = ConnectionPool(...)
        self.registry  = ToolRegistry(pool=self.pool, ...)
        self.executor  = Executor(pool=self.pool, registry=self.registry, ...)
        self.llm       = LLMInterface(providers, ...)
```

`chat()` becomes ~40 LoC:
1. Ensure system prompt & append user msg.
2. Loop up to `max_turns`:
   * `resp = await self.llm.generate(history, self.registry.llm_tools)`
   * If no tool calls → return.
   * `early = await self.executor.run_calls(resp.tool_calls, history)`
   * If `early` → return.
3. Safety guard → return "turn limit reached".

Mirror for `stream_chat()`.

Run full tests; refine until green.

---

### Phase 7 – Legacy shim & deprecation

Turn the *original* `seqoria_agent/client/client.py` into:
```python
import warnings
from seqoria_agent.client.ai_client.base import AIClient  # re-export
warnings.warn(
    "seqoria_agent.client.client.AIClient is deprecated; import from seqoria_agent.client instead",
    DeprecationWarning,
    stacklevel=2,
)
```
*Other internal symbols stay until v0.4 removal.*


Ensure all internal imports (tests, examples) updated to avoid warning spam.

---

### Phase 8 – Test-suite overhaul & coverage bump

* Move new component-level tests into `tests/ai_client/` sub-folder.
* Tag slow vs. fast tests via `pytest -m slow` if needed.
* Aim for ≥ 90 % coverage in `ai_client/` package.

**Status (2025-06-25): ✅ DONE – Achieved 100 % statement coverage for `ai_client/` with pytest-cov.**

---

### Phase 9 – Documentation & minor clean-ups

* Update `docs/` references, quick-start snippets, and API reference.
* Introduce `@dataclass ClientSettings` to tidy ctor signature (non-breaking).
* Replace magic strings with constants (`AI_MESSAGE_KEY`, `XML_ERROR_TAG`, …).
* Audit logging: promote noisy `.info()` calls to `.debug()`.

---

## Timeline suggestion

| Phase | Effort | Owner | Target PR |
|-------|--------|-------|-----------|
| 1 | ½ day | dev-A | PR-2 |
| 2 | ½ day | dev-A | PR-3 |
| 3 | 1 day | dev-A | PR-4 |
| 4 | ½ day | dev-B | PR-5 |
| 5 | ¼ day | dev-B | PR-6 |
| 6 | 1 day | dev-A+dev-B | PR-7 |
| 7 | ¼ day | dev-A | PR-8 |
| 8 | ½ day | dev-B | PR-9 |
| 9 | ½ day | dev-A | PR-10 |

*(Effort is engineering time, not elapsed time. Parallel work possible on Phases 4 & 5 while 3 is in review.)*

---

## Done-definition for the migration

* `client.py` ≤ 150 LoC and tagged `# DEPRECATED`.
* New `ai_client/` modules documented and unit-tested.
* Public API unchanged for end-users.
* All tests & type-checks (`mypy`, `ruff`) pass.
* Docs updated; CHANGELOG entry added.
