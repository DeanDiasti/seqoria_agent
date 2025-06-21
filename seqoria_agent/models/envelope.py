from __future__ import annotations


from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Public constants
# -----------------------------------------------------------------------------
ValidDisplayType = Literal[
    "chart",
    "table",
    "image",
    "markdown",
    "html",
    "notification",
    "text",
]


class DisplayEnvelope(BaseModel):
    """A minimal JSON structure that front-ends can render generically.

    Attributes
    ----------
    type:
        A short string describing the semantic kind of ``payload``. Front-end
        renderers should branch on this value.
    payload:
        The actual data to display. Structure depends on ``type``:
            * ``chart`` - JSON spec from Plotly, ECharts, …
            * ``table`` - list/record structure (rows or columns)
            * ``image`` - base64 string or URL
            * ``markdown`` - markdown formatted string
            * ``html`` - raw HTML string
            * ``notification`` - simple message
            * ``text`` - raw text string (catch-all)
    title:
        Optional human-readable title.
    meta:
        Arbitrary metadata (dimensions, theme, …).
    """

    type: ValidDisplayType = Field(..., description="Semantic kind of payload")
    payload: Union[Dict[str, Any], List[Any], str] = Field(..., description="Content to render")
    title: str | None = Field(default=None, description="Optional title for the display element")
    meta: Dict[str, Any] | None = Field(default=None, description="Free-form metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "chart",
                    "payload": {"data": []},
                    "title": "Quarterly Sales",
                    "meta": {"library": "plotly"},
                },
                {"type": "markdown", "payload": "# Hello World\nThis is **markdown**."},
            ]
        }
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def dict_for_transport(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (alias for ``model_dump``)."""

        return self.model_dump() 