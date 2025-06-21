from __future__ import annotations

"""fastqutie package – public re-exports and convenience imports.

This sub-package replaces the historical ``fastqutie.py`` module while
keeping the external import path **unchanged**::

    from qutie.server.fastqutie import FastQutie

Down-stream projects therefore do **not** need to update their code.
"""

from .app import FastQutie  # noqa: F401

__all__: list[str] = [
    "FastQutie",
] 