from __future__ import annotations
import html
from typing import Any, Sequence, Callable
"""XML formatter plug-in system for seqoria_agent tools.

This implements a minimal but extensible mechanism to convert arbitrary Python
values returned by **regular** `@tool` functions into an XML snippet that is
safe to embed in an LLM prompt.  Display tools continue to return
`DisplayEnvelope` objects directly and are handled separately.

Usage (server side)
-------------------

```python
from seqoria_agent.server.runtime import function_server
from seqoria_agent.server.formatters import DeclarativeXMLFormatter

mcp = function_server()

class WeatherFormatter(DeclarativeXMLFormatter):
    top_level_fields = ["city", "temperature", "condition"]

@mcp.tool(xml_formatter=WeatherFormatter)
async def get_weather(city: str):
    return {"city": city, "temperature": 18, "condition": "Sunny", "humidity": 40}
```

The formatter will emit only the declared fields in the `<llm_output>` section
so the LLM sees a concise, deterministic structure.
"""

__all__: list[str] = [
    "BaseXMLFormatter",
    "DeclarativeXMLFormatter",
]


class BaseXMLFormatter:
    """Base class for converting arbitrary Python data into an XML string.

    Sub-classes must implement the `format` method. The class is callable,
    forwarding calls to the `format` method.

    Attributes
    ----------
    indent : str
        The string used for indentation, by default two spaces.
    """

    indent: str = "  "

    def __call__(self, data: Any, *, indent_level: int = 0) -> str:  # noqa: D401 – callable shortcut
        return self.format(data=data, indent_level=indent_level)

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def format(self, data: Any, *, indent_level: int = 0) -> str:
        """Convert the given data into an XML string.

        This method must be implemented by sub-classes.

        Parameters
        ----------
        data : Any
            The Python data to serialize (e.g., dict, list, scalar).
        indent_level : int, optional
            The current indentation level, used for pretty-printing.
            Defaults to 0.

        Returns
        -------
        str
            An XML string representation of the data.

        Raises
        ------
        NotImplementedError
            If the method is not overridden by a sub-class.
        """
        raise NotImplementedError

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------

    def _indent(self, level: int) -> str:  # noqa: D401
        return self.indent * level

    def _escape(self, text: str) -> str:  # noqa: D401 – xml escape
        return html.escape(text, quote=False)


class DeclarativeXMLFormatter(BaseXMLFormatter):
    """Declarative helper that turns *structured* dictionaries into concise XML.

    Sub-classes customise only a handful of **class attributes** – no code –
    and inherit a ready-made :py:meth:`format` implementation.

    Supported attributes
    --------------------
    top_level_fields:
        Scalar keys to emit directly under ``<tool_response>``.
    results_list_field:
        Name of the key whose *value is a list* that should be rendered via
        ``<result_item_name>`` tags.  Leave empty to skip list handling.
    result_item_name:
        XML tag name for each item in that list (default "item").
    result_item_fields:
        Sub-keys from each list item dict to include as children.
    result_item_attributes:
        Mapping ``attr_name -> callable(item_dict)`` whose return value is
        inserted as an *attribute* on the ``<result_item_name>`` element.
    """

    # Developers override these ---------------------------------------------
    top_level_fields: Sequence[str] | None = None
    results_list_field: str = ""
    result_item_name: str = "item"
    result_item_fields: Sequence[str] = ()
    result_item_attributes: dict[str, Callable[[dict], Any]] = {}

    # ------------------------------------------------------------------
    # Public formatter API
    # ------------------------------------------------------------------

    def format(self, data: Any, *, indent_level: int = 0) -> str:  # noqa: ANN001
        if not isinstance(data, dict):
            # Fallback – scalar or list
            return self._format_fallback(data=data, indent_level=indent_level)

        lines: list[str] = []
        indent = self._indent(indent_level)

        # -------- Top-level scalar fields -----------------------------
        scalar_keys = data.keys()
        if self.top_level_fields:
            scalar_keys = [k for k in scalar_keys if k in self.top_level_fields]

        for key in scalar_keys:
            if key == self.results_list_field:
                continue  # handled separately
            val = data.get(key)
            if val is None:
                continue
            lines.append(f"{indent}<{key}>{self._escape(str(val))}</{key}>")

        # -------- List-of-results block -------------------------------
        if self.results_list_field and self.results_list_field in data:
            results = data[self.results_list_field]
            if isinstance(results, list):
                lines.append(f"{indent}<{self.results_list_field}>")
                for item in results:
                    lines.extend(self._format_result_item(item, indent_level + 1))
                lines.append(f"{indent}</{self.results_list_field}>")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_result_item(self, item: Any, level: int) -> list[str]:  # noqa: ANN001
        """Return list[str] lines for a single *result* element."""
        indent = self._indent(level)
        if not isinstance(item, dict):
            return [f"{indent}<{self.result_item_name}>{self._escape(str(item))}</{self.result_item_name}>"]

        # Build attribute string
        attrs: list[str] = []
        for name, fn in self.result_item_attributes.items():
            try:
                val = fn(item)
            except Exception:
                val = None
            if val is not None:
                attrs.append(f'{name}="{self._escape(str(val))}"')
        attr_str = f" {' '.join(attrs)}" if attrs else ""

        lines: list[str] = [f"{indent}<{self.result_item_name}{attr_str}>"]

        for fld in self.result_item_fields:
            if fld in item and item[fld] is not None:
                val = self._escape(str(item[fld]))
                lines.append(f"{indent}  <{fld}>{val}</{fld}>")

        lines.append(f"{indent}</{self.result_item_name}>")
        return lines

    def _format_fallback(self, data: Any, indent_level: int) -> str:  # noqa: ANN001
        # Re-use old behaviour for arbitrary types
        if isinstance(data, list):
            return "\n".join(
                f"{self._indent(level=indent_level)}<item>{self._escape(text=str(object=v))}</item>" for v in data
            )
        return f"{self._indent(level=indent_level)}{self._escape(text=str(object=data))}"


# ---------------------------------------------------------------------------
# Helper – dynamic factory for inline use
# ---------------------------------------------------------------------------


def make_formatter(
    *,
    top_level_fields: Sequence[str] | None = None,
    results_list_field: str = "",
    result_item_name: str = "item",
    result_item_fields: Sequence[str] | None = None,
    result_item_attributes: dict[str, Callable[[dict], Any]] | None = None,
) -> DeclarativeXMLFormatter:
    """Return a *configured instance* of DeclarativeXMLFormatter.

    Examples
    --------
    >>> fmt = make_formatter(top_level_fields=["answer"], results_list_field="results")
    >>> @app.tool(response_format=fmt)
    ... def search(q: str): ...
    """

    cls_dict = {
        "top_level_fields": top_level_fields,
        "results_list_field": results_list_field,
        "result_item_name": result_item_name,
        "result_item_fields": result_item_fields or (),
        "result_item_attributes": result_item_attributes or {},
    }
    Derived = type("InlineFormatter", (DeclarativeXMLFormatter), cls_dict)  # noqa: N801 – dynamic name
    return Derived()


# ---------------------------------------------------------------------------
# Internal – derive formatter automatically from sample data
# ---------------------------------------------------------------------------


def derive_formatter_from_data(sample: Any) -> DeclarativeXMLFormatter | None:  # noqa: ANN001
    """Attempt to build a formatter that mirrors the structure of *sample*.

    Currently supports a **dict** with:
    • scalar fields → top_level_fields
    • first list-of-dict → results_list_field & result_item_fields
    Returns *None* when heuristic fails.
    """
    if not isinstance(sample, dict):
        return None

    scalar_keys = [k for k, v in sample.items() if not isinstance(v, list)]
    list_key = next((k for k, v in sample.items() if isinstance(v, list) and v and isinstance(v[0], dict)), None)

    if list_key:
        item_fields = list(sample[list_key][0].keys())  # type: ignore[index]
    else:
        item_fields = []

    return make_formatter(
        top_level_fields=scalar_keys,
        results_list_field=list_key or "",
        result_item_fields=item_fields,
    ) 