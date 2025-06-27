# Manual Agentic Loop – Implementation Roadmap

> **Goal**: Expose low-level building blocks inside `AIClient` so application authors can drive their **own** agentic loops while preserving full backward-compatibility with `AIClient.chat()` / `AIClient.stream_chat()`.
>
> At the end of this refactor, developers should be able to write:
>
> ```python
> async with AIClient(...) as client:
>     hist = []
>     user_query = "Plot my weekend trip"
>
>     hist = client.prepare_history(user_query, hist)
>     resp = await client.llm_generate_once(hist)  # <= single LLM step
>     if resp.tool_calls:
>         await client.run_tool_calls(resp.tool_calls, hist)
> # …
> ```
>
> without touching any private helpers or internal state.

---

## Current Situation (April 2025)

* `AIClient.chat()` and `AIClient.stream_chat()` embed the entire control flow: history prep → LLM call → tool execution → loop/turn-limit.
* There is **no** public API to run exactly *one* LLM step or to delegate tool execution separately.
* Key code regions today:

```171:202:seqoria_agent/client/ai_client/base.py
# inner agentic loop inside `chat()` – will be lifted out
```
```203:251:seqoria_agent/client/ai_client/base.py
# equivalent loop inside `stream_chat()`
```

---

## Phased Plan (two phases delivered per milestone)

We split the work into **six small phases** grouped into three delivery milestones so we can merge value incrementally while keeping the test-suite green.

| Milestone | Phase | Scope | Acceptance tests & examples |
|-----------|-------|-------|-----------------------------|
| **M1 – Foundations** | **P1** | *History helpers*  
`prepare_history(user_query, history)`  
`tools_enabled()` → returns bool | • `tests/ai_client/test_prepare_history.py`  
• Example snippet in docs |
|           | **P2** | *LLM single-shot*  
`llm_generate_once(history, use_tools=True, tool_choice="auto")`  
`llm_stream_once(...)` (streaming generator) | • `tests/ai_client/test_llm_once.py`  
• `examples/manual_loop/basic_once.py` |
| **M2 – Integration** | **P3** | *Tool execution wrapper*  
`run_tool_calls(tool_calls, history)` – thin delegate to `Executor.run_calls` | • `tests/ai_client/test_run_tool_calls.py` |
|           | **P4** | *Chat refactor*  
Rewrite `chat()` and `stream_chat()` to call the new primitives; behaviour and public surface **must not change** (golden test). | • Regression tests assert byte-for-byte parity with current `chat()` answers on a stub provider. |
| **M3 – Ergonomics & Docs** | **P5** | *Auxiliary helpers*  
Expose low-level `call_tool(...)` as first-class alias; document `append_assistant_message()` if needed. | • `examples/manual_loop/full_loop.py` showing GUI hooks, error callback usage. |
|           | **P6** | *Documentation & polishing*  
Update `docs/guides/ai_client.md`, add this roadmap, and cross-link new APIs. | • Spell-check passes; Dead-link checker clean. |

> **Implementation cadence**: we always merge *two* phases at a time (P1+P2, P3+P4, P5+P6) so each PR remains < ~400 LOC and keeps the CI fast.

---

## Phase Details

### Phase 1 – History helpers

1. Add `prepare_history()` and `tools_enabled()` to `AIClient`.
2. Move the existing system-prompt logic from `chat()` into the helper.
3. Unit test matrix:
   * no prior history ⇒ system prompt added.
   * existing system prompt preserved.
   * history list instance identity maintained.

### Phase 2 – Single LLM step helpers

1. Implement `llm_generate_once()` & `llm_stream_once()` as **thin wrappers** around `LLMInterface.generate/stream`.
2. Respect `use_tools` flag to skip tool metadata for LLM-only mode.
3. Tests use a stub `BaseProvider` that returns deterministic text / tool_calls.
4. Example script `examples/manual_loop/basic_once.py` demonstrates printing deltas in real time.

### Phase 3 – Tool execution wrapper

1. Public coroutine `run_tool_calls()` delegates to `self.executor.run_calls()`.
2. Pass through structured return (early-exit payload).
3. Cover timeout / cache path via mocks.

### Phase 4 – Chat refactor (no behaviour change)

1. Rewrite existing `chat()` and `stream_chat()` to orchestrate via new helpers.
2. Keep **identical** turn-limit logic and early-exit behaviour.
3. Golden regression test hashes previous JSON transcripts.

### Phase 5 – Auxiliary helpers & public surface polish

1. Promote existing private `_tools_enabled()` to `tools_enabled()` (public).
2. Optional `append_assistant_message()` if external callers want history side-effects.
3. Re-export new APIs from `seqoria_agent.client` convenience module.

### Phase 6 – Docs & DX

1. Expand `docs/guides/ai_client.md` with a "Manual Loop" section including:
   * Minimal CLI demo.
   * Streaming UI integration.
   * Debugging tips.
2. Link from README + quickstart.

---

## Continuous Testing & Examples Strategy

* **Every phase introduces its own focused unit tests** under `tests/ai_client/` so CI never regresses.
* Example scripts live in `examples/manual_loop/` and evolve alongside the phases.  Users can run them end-to-end after each milestone.
* We leverage the existing dummy `LocalToolbox` in tests to avoid network I/O.

---

## References

* `Executor.run_calls()` for execution semantics.
* `LLMInterface.generate()` for single-shot calls.
* `seqoria_agent/client/ai_client/base.py` – primary refactor target.
* Prior phased refactor documents (`agent-functionality-update.md`, `feature-implementation.md`) follow the same pattern; we replicate their style.

---

*Maintained by*: @seqoria-core ‑ please update this roadmap after each merged milestone.
