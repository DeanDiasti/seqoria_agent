# The DisplayEnvelope

The `DisplayEnvelope` is a simple, frontend-agnostic JSON contract that allows tools to send rich UI components directly to the user. It's the core of xmcp's display capabilities.

## Schema

A `DisplayEnvelope` is a Pydantic model with the following fields:

| Field   | Type                               | Description                               |
|---------|------------------------------------|-------------------------------------------|
| `type`    | `string`                           | Semantic category (e.g., `chart`, `table`). Drives the renderer. |
| `payload` | `dict \| list \| str`              | The actual data to render. Its structure depends on the `type`. |
| `title`   | `string` (optional)                | A human-readable heading for the UI component. |
| `meta`    | `dict` (optional)                  | Extra, unstructured info for the renderer (e.g., dimensions, theme). |

### Supported Types

The `type` field can be one of the following:

-   `chart`: The payload should be a JSON specification for a chart (e.g., Plotly, ECharts).
-   `table`: The payload should be a list of dictionaries (rows) or a dictionary of lists (columns).
-   `image`: The payload should be a URL or a base64-encoded image string.
-   `markdown`: The payload is a string of Markdown text.
-   `html`: The payload is a raw HTML string.
-   `notification`: The payload is a simple string for a toast or alert.
-   `text`: The payload is a plain text string.

## Example

Here's an example of a `DisplayEnvelope` that represents a table:

```json
{
  "type": "table",
  "payload": [
    {"region": "EMEA", "sales": 42000, "prospects": 120},
    {"region": "NA", "sales": 85000, "prospects": 250}
  ],
  "title": "Sales Summary",
  "meta": {
    "currency": "USD"
  }
}
```

## Rendering

It is the responsibility of the front-end application (web, mobile, or CLI) to interpret the `DisplayEnvelope` and render the appropriate UI component. The `AIClient` simply delivers the envelope to your application.

A web renderer might implement a function like this:

```javascript
function renderComponent(envelope) {
  const container = document.getElementById('display-area');
  switch (envelope.type) {
    case 'table':
      renderTable(container, envelope.payload, envelope.title);
      break;
    case 'chart':
      renderChart(container, envelope.payload, envelope.title);
      break;
    // ... other cases
  }
}
```

## Strict-Mode Validation

When an `XMCP` server runs with `strict_mode=True`, **every** `@display_tool` return value is validated:

1. Must coerce into `DisplayEnvelope`.
2. Payload must be JSON-serialisable.
3. Invalid values raise `XMCPError(code="INVALID_PAYLOAD")` which the client surfaces.

## Toggling UI vs LLM output

Every display tool receives a `show_user: bool` param.  If `True`, return a `DisplayEnvelope`; if `False`, return raw data for further LLM reasoning.

```python
@mcp.display_tool
def sales(region: str, show_user: bool = False):
    data = _query(region)
    if show_user:
        return DisplayEnvelope(type="table", payload=data, title=f"Sales {region}")
    return data  # plain dict for LLM
```

## Streaming

`AIClient.stream_chat()` yields `ChatChunk(type="tool_display", envelope=…)` **immediately** when a display envelope arrives, enabling real-time UI updates. 