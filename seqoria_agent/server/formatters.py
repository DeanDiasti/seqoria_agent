from __future__ import annotations

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

from typing import Any, List, Sequence
import html

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
        return self.format(data, indent_level=indent_level)

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
    """A formatter that serializes dicts and lists recursively.

    This implementation provides options to filter keys at different levels,
    allowing for concise XML representations for the LLM.

    Attributes
    ----------
    top_level_fields : Sequence[str] | None
        A list of keys to include from the root object. If `None` or empty,
        all keys are included. Defaults to `None`.
    result_item_fields : dict[str, Sequence[str]]
        A mapping from a key (whose value is a list of dicts) to a list of
        keys that should be included for the nested dicts.
        Example: `{"records": ["id", "name"]}`
    """

    # Developers override this in sub-classes
    top_level_fields: Sequence[str] | None = None

    # Keys whose value is a *list* of dict items can specify **nested**
    # white-list via the mapping below.  Example::
    #
    #     result_item_fields = {
    #         "records": ["id", "name"],
    #     }
    #
    result_item_fields: dict[str, Sequence[str]] = {}

    def format(self, data: Any, *, indent_level: int = 0) -> str:
        """Recursively convert the data into an XML string.

        This method handles dicts, lists, and scalar values, delegating to
        more specific helpers.

        Parameters
        ----------
        data : Any
            The Python data to serialize.
        indent_level : int, optional
            The current indentation level. Defaults to 0.

        Returns
        -------
        str
            The resulting XML string.
        """
        if isinstance(data, dict):
            return self._format_dict(data, indent_level)
        if isinstance(data, list):
            return self._format_list(data, indent_level, "item")
        return f"{self._indent(indent_level)}{self._escape(str(data))}"

    # ----------------------------------------------------------
    def _format_dict(self, data: dict[str, Any], level: int) -> str:  # noqa: ANN001
        keys = list(data.keys())
        if self.top_level_fields:
            keys = [k for k in keys if k in self.top_level_fields]
        lines: list[str] = []
        for key in keys:
            val = data[key]
            tag = self._escape(str(key))
            if isinstance(val, dict):
                inner = self._format_dict(val, level + 1)
                lines.append(f"{self._indent(level)}<{tag}>\n{inner}\n{self._indent(level)}</{tag}>")
            elif isinstance(val, list):
                allowed = self.result_item_fields.get(key)
                inner = self._format_list(val, level + 1, "item", allowed)
                lines.append(f"{self._indent(level)}<{tag}>\n{inner}\n{self._indent(level)}</{tag}>")
            else:
                lines.append(f"{self._indent(level)}<{tag}>{self._escape(str(val))}</{tag}>")
        return "\n".join(lines)

    def _format_list(
        self,
        items: list[Any],
        level: int,
        item_tag: str,
        allowed_fields: Sequence[str] | None = None,
    ) -> str:  # noqa: ANN001
        lines: list[str] = []
        for itm in items:
            if isinstance(itm, dict):
                lines.append(f"{self._indent(level)}<{item_tag}>")
                nested_keys = list(itm.keys())
                if allowed_fields:
                    nested_keys = [k for k in nested_keys if k in allowed_fields]
                for k in nested_keys:
                    val = itm[k]
                    tag = self._escape(str(k))
                    lines.append(f"{self._indent(level + 1)}<{tag}>{self._escape(str(val))}</{tag}>")
                lines.append(f"{self._indent(level)}</{item_tag}>")
            else:
                lines.append(f"{self._indent(level)}<{item_tag}>{self._escape(str(itm))}</{item_tag}>")
        return "\n".join(lines) 