from enum import Enum


class _ToolName(str, Enum):
    """Internal enumeration of built-in meta-tool names.

    Sub-classing *str* keeps behaviour with OpenAI schema values while
    providing a single interned instance per name to reduce duplicate
    string allocations across the codebase.
    """

    __slots__ = ()

    THINK = "__think__"


# Public re-exports kept for full backwards-compatibility --------------------
# (Existing import sites outside this module continue to work unchanged.)

THINK_TOOL_NAME: str = _ToolName.THINK.value

__all__ = ["_ToolName", 
           "THINK_TOOL_NAME"] 