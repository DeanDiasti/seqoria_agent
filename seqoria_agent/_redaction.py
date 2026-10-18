"""Credential-safe text for diagnostics; never use redacted values for requests.

All URL query values and fragments are hidden because deployments use arbitrary
credential names. Hosts, paths and query names remain useful for troubleshooting.
This is best-effort text redaction, not a replacement for avoiding secret payloads
in application errors or configuring third-party HTTP/SDK loggers appropriately.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_REDACTED = "[REDACTED]"
_URL = re.compile(r"(?:https?|sses?)://[^\s\"'<>]+", re.IGNORECASE)
_AUTHORIZATION = re.compile(
    r"(\b(?:proxy[-_])?authorization[\"']?\s*[:=]\s*)(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\r\n,;}]+)",
    re.IGNORECASE,
)
_CREDENTIAL = re.compile(
    r"(\b(?:api[-_]?key|access[-_]?token|refresh[-_]?token|id[-_]?token|"
    r"client[-_]?secret|password|passwd|secret|token|x[-_]api[-_]key)"
    r"[\"']?\s*[:=]\s*)(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s&,;\"'}]+)",
    re.IGNORECASE,
)


def redact_url(url: str) -> str:
    """Hide URL credentials while retaining the server's address and path."""
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            netloc = f"redacted@{netloc.rsplit('@', 1)[1]}"
        query = urlencode(
            [(key, _REDACTED) for key, _ in parse_qsl(parts.query, keep_blank_values=True)]
        )
        return urlunsplit(
            (parts.scheme, netloc, parts.path, query, _REDACTED if parts.fragment else "")
        )
    except ValueError:
        # Malformed URLs must not fall back to printing the untrusted string.
        return "[REDACTED URL]"


def redact_text(value: object) -> str:
    """Redact credential-bearing URLs and conventional auth fields in text."""
    text = _URL.sub(lambda match: redact_url(match.group()), str(value))
    text = _AUTHORIZATION.sub(lambda match: match.group(1) + _REDACTED, text)
    return _CREDENTIAL.sub(lambda match: match.group(1) + _REDACTED, text)


def safe_exception_cause(
    error: BaseException | None, *, raised: BaseException | None = None
) -> BaseException | None:
    """Keep diagnostic chains only when their visible messages contain no credentials.

    ``raised`` avoids chaining a canonical error to itself when a mapper returns
    its input unchanged. Existing safe causes remain available to callers.
    """
    if error is raised and error is not None:
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    current = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if redact_text(current) != str(current):
            return None
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return error
