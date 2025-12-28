# XMCP Display Tools – Implementation Plan

## High-Level Overview
Display Tools extend Seqoria Agent’s existing `@tool` framework to let an LLM decide **whether to show a rich, non-textual UI element to the human user** in addition to returning data for the AI’s internal reasoning.  
A Display Tool therefore yields **two coordinated payloads**:

1. **`ai_payload`** – Structured data embedded in the usual `<tool_response>` XML so the LLM can consume it.
2. **`display_payload`** – A serialisable object (table, markdown, custom Vue/React component, …) that the host application can render directly in the user interface **without exposing it back to the LLM**.

This document outlines a phased roadmap to introduce Display Tools while preserving full backward-compatibility with existing tools, servers, and clients.

## Goals
* Maintain the _existing_ developer ergonomics of `@tool` (type-hints, auto JSON-schema, strict mode, etc.).
* Give the LLM explicit control (via a `show_display: bool` flag) over whether a visual representation should be produced.
* Provide a small **standard library** of display objects (Table, Markdown, CustomComponent) and a clear extensibility point for custom widgets.
* Keep the execution engine **framework-agnostic** – all heavy lifting lives in `seqoria_agent/server/xmcp_core` so FastXMCP, the APIRouter variant, and `LocalToolbox` can reuse it.
* Allow incremental rollout: each phase is additive; no phase invalidates earlier public APIs.

---

## Phased Roadmap

### Phase 0 — Groundwork & Vocabulary (⚙ non-code)
* Finalise naming: `display_payload`, `DisplayToolResult`, `@display_tool`, etc.
* Decide on serialisation format for `display_payload` (Pydantic → JSON).

### Phase 1 — Data Models (✅ backward-compatible)
* **New file `seqoria_agent/models/display.py`** containing:
  ```python
  from typing import Any, List, Dict, Union, Literal
  from pydantic import BaseModel, Field
  
  class BaseDisplayObject(BaseModel):
      display_type: str
  
  class TableDisplay(BaseDisplayObject):
      display_type: Literal["table"] = "table"
      title: str
      headers: List[str]
      rows: List[List[Any]]
  
  class MarkdownDisplay(BaseDisplayObject):
      display_type: Literal["markdown"] = "markdown"
      markdown_text: str
  
  class CustomDisplay(BaseDisplayObject):
      display_type: Literal["custom"] = "custom"
      component_name: str
      component_props: Dict[str, Any] = Field(default_factory=dict)
  
  DisplayObject = Union[TableDisplay, MarkdownDisplay, CustomDisplay]
  
  class DisplayToolResult(BaseModel):
      ai_payload: Any  # whatever the LLM should see
      display_payload: Union[DisplayObject, None] = None
  ```
* **No runtime references yet** – purely a typed contract developers can import.

### Phase 2 — `@display_tool` Decorator
* Add `ToolCore._make_display_reg()` which:
  * Validates that the decorated function accepts a `show_display: bool = False` kw-arg.
  * Automatically augments the description with instructions, e.g.  
    _"Set `show_display=true` to present a table of search results to the user."_
  * Internally delegates to the existing `@tool` registration so JSON-schema generation, dependency tracking, etc. work out-of-the-box.
* Expose it via `self.display_tool` on `ToolCore`, `FastXMCP`, and `LocalToolbox`.

Code sketch (inside `ToolCore.__init__`):
```python
self.display_tool = self._make_display_reg()
```

### Phase 3 — Dual-Payload Execution Path
* Update `_wrap_tool_with_error_handling` to recognise `DisplayToolResult`:
  ```python
  if isinstance(result, DisplayToolResult):
      ai_part = result.ai_payload
      disp_part = result.display_payload
  else:
      ai_part = result
      disp_part = None
  ```
* Serialize `ai_part` **exactly as today**, but return **both** values from `run_tool`:
  ```python
  return ai_xml, disp_part
  ```
* Touchpoints to update:
  * `LocalToolbox.call_tool` – now returns `SimpleNamespace(text=xml, display=disp)`.
  * `FastXMCP` & `FastXMCPRouter` – include `display_payload` in HTTP JSON alongside `ai_message`.
  * **`Executor` & `AIClient`** – parse the tuple from `ConnectionPool.call_tool` so that:
    * `ToolCallResult.result` holds the *ai_payload* XML (unchanged for LLM).
    * The *display_payload* is forwarded to user-facing callbacks (`on_display` or similar) or included in the client’s response envelope so front-ends can render it.

### Phase 4 — Standard Display Formatters & Renderer Contract
* Ship default renderers (Table, Markdown) for the demo web UI (outside core package).
* Allow custom display objects by matching on `display_type`.

### Phase 5 — Strict-Mode Validation & Error Handling
* Extend `ToolCore._validate_payload` to also accept `DisplayToolResult`.
* Emit `INVALID_DISPLAY_PAYLOAD` when `display_payload` cannot be serialised.

### Phase 6 — Documentation & Examples
* Update `docs/guides/manual_agent.md` with a full example.
* Add `examples/display/` folder showing both server and client code.

---

## Implementation Checklist (by Feature)
- [ ] Terminology & serialization decision finalised
- [ ] Core data models (`DisplayToolResult`, `BaseDisplayObject`, standard objects) added
- [ ] `@display_tool` decorator implemented and exported
- [ ] ToolCore updated for dual-payload execution (`DisplayToolResult` recognition)
- [ ] LocalToolbox updated to forward display payload to callers
- [ ] FastXMCP / FastXMCPRouter HTTP responses include `display_payload`
- [ ] Standard front-end renderers (Table, Markdown) shipped for demo UI
- [ ] Strict-mode validation & `INVALID_DISPLAY_PAYLOAD` error handling
- [ ] Documentation & runnable examples added
- [ ] Client-side handling in `Executor` & `AIClient` to surface `display_payload` to UI/consumer

> _Tick items as you merge each feature to keep progress transparent._
