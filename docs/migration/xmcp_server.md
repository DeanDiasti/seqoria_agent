# XMCP Server Runtime – Upgrade Guide (v0.2.0)

Seqoria-Agent's server runtime was fully refactored in v0.2.0.  The public API remains **backwards-compatible**, but import paths have changed and formerly monolithic helpers were split into focused sub-modules.

## What changed?

1. **Canonical imports** are now:
   * `from seqoria_agent.server.fastxmcp import FastXMCP, FastXMCPRouter`
   * `from seqoria_agent.toolbox import LocalToolbox`
2. The concrete implementations live under `seqoria_agent.server.fastxmcp_public` and `seqoria_agent.toolbox.local_public`.  Those modules are **implementation details** – import from the package roots shown above.
3. **Deprecation shims** were added for the legacy paths:
   * `seqoria_agent.server.fastxmcp.app.FastXMCP`
   * `seqoria_agent.server.fastxmcp.router.FastXMCPRouter`
   * `seqoria_agent.toolbox.local.LocalToolbox`
   They forward to the new locations and emit a single `DeprecationWarning`.
4. Internal helpers (formatter cache, parameter injection, auth bridge) now live in `seqoria_agent.server.xmcp_core`.

## Migration checklist

✔ Replace `seqoria_agent.server.fastxmcp.app` / `.router` imports with the package-root import.  Example:

```python
- from seqoria_agent.server.fastxmcp.router import FastXMCPRouter
+ from seqoria_agent.server.fastxmcp import FastXMCPRouter
```

✔ Replace `seqoria_agent.toolbox.local` with `seqoria_agent.toolbox`:

```python
- from seqoria_agent.toolbox.local import LocalToolbox
+ from seqoria_agent.toolbox import LocalToolbox
```

✔ (Optional) silence the *single* `DeprecationWarning` in your test suite:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="seqoria_agent")
```

✔ Update any docs / comments referencing the old paths.

## How long will the shims stay?

The shims will be kept for **at least one minor version** (until v0.3) and will be removed no earlier than **2026-01-01**.  They raise only a runtime warning – there is **no impact** on performance or behaviour.

## Troubleshooting

• Still seeing import errors?  Run `pip uninstall xmcp` – a conflicting package of the same name on PyPI may be shadowing `seqoria_agent`.

• Need the old server runtime while you migrate?  Pin `seqoria_agent==0.1.*` in your requirements.

---

Happy building – and thank you for upgrading!  If you hit any issues please open an issue on GitHub. 