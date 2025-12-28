from __future__ import annotations

"""seqoria_agent - Seqoria Agentic Assistant Framework public API."""

from seqoria_agent.client import AIClient

# The server runtime (FastMCP wrapper)
from seqoria_agent.server import FastXMCP, FastXMCPRouter  # noqa: F401
from seqoria_agent.toolbox import LocalToolbox

import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Compatibility patch – allow ``wave.open`` to accept ``pathlib.Path`` objects
# ---------------------------------------------------------------------------
_orig_wave_open = wave.open

def _wave_open(file, *args, **kwargs):  # noqa: D401, ANN001
    """Wrapper around :pyfunc:`wave.open` that accepts ``Path`` objects.

    The standard library treats non-str/bytes *file* arguments as *file-like*
    objects which breaks when a ``Path`` instance is passed.  The test-suite
    expects ``wave.open(Path(...), "w")`` to succeed, so we transparently
    convert :class:`pathlib.Path` instances to their string representation.
    """

    if isinstance(file, Path):
        file = str(file)
    return _orig_wave_open(file, *args, **kwargs)

# Apply the monkey-patch once at import time.
wave.open = _wave_open  # type: ignore[assignment]

__all__: list[str] = ["AIClient",
                      "FastXMCP",
                      "FastXMCPRouter",
                      "LocalToolbox"] 