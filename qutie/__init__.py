from __future__ import annotations

"""qutie – eXtended Multi-Channel Protocol (v2).

Public API surface. Import *only* from here in downstream projects:

    from qutie import AIClient, qutie, DisplayEnvelope
"""

from qutie.client import AIClient, ChatChunk  # noqa: F401
from qutie.server import QUTIE, FastQutie  # noqa: F401
from qutie.models.envelope import DisplayEnvelope  # noqa: F401
from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFound

__all__: list[str] = [
    "AIClient",
    "ChatChunk",
    "QUTIE",
    "DisplayEnvelope",
    "FastQutie",
]

try:
    __version__: str = _pkg_version("qutie")
except _PkgNotFound:  # pragma: no cover – editable or local install
    __version__ = "0.0.0"  # fallback during development 