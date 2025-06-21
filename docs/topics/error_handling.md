# Structured Error Handling

XMCP surfaces failures as **typed XML** so both the LLM and your application can react intelligently.

```xml
<tool_response tool_name="run_query">
  <error code="DB_TIMEOUT" retryable="true">
    <![CDATA[database timeout after 5 s]]>
    <meta>{"sql": "SELECT * FROM …"}</meta>
  </error>
</tool_response>
```

| Attribute | Meaning |
|-----------|---------|
| `code` | Short, upper-snake identifier (e.g. `VALIDATION_FAIL`) |
| `retryable` | "true" / "false" hint for AIClient retry logic |
| CDATA body | Human-readable detail |
| `<meta>` | Optional JSON for machines |

---

## On the Server

```python
from xmcp.models.errors import XMCPError

@mcp.tool
def run_query(sql: str):
    try:
        return db.execute(sql)
    except DBTimeout as exc:
        raise XMCPError(code="DB_TIMEOUT", detail=str(exc), retryable=True, meta={"sql": sql})
```

## On the Client

```python
def on_error(err, history):
    if err["retryable"] and err["code"] == "DB_TIMEOUT":
        logger.info("retrying query…")
```

`AIClient` parses the XML via `_extract_error_from_xml()` and triggers the optional `on_error` callback **without** stopping the conversation unless you choose to. 