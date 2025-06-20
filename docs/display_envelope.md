# The DisplayEnvelope

The `DisplayEnvelope` is a simple, frontend-agnostic JSON contract that allows tools to send rich UI components directly to the user. It's the core of MCPx's display capabilities.

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