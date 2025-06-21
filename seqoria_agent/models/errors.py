"""Error types shared between seqoria_agent client and server layers."""


class ToolRemoteError(Exception):
    """Wraps exceptions raised during remote tool execution."""


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open for a given server and tool calls are blocked."""


class XMCPError(Exception):
    """Structured error raised by xmcp tools.

    When a tool raises :class:`XMCPError`, the server intercepts it and returns
    a *structured* error envelope instead of propagating the raw stack-trace.
    The attributes mirror the fields that will be encoded in the resulting
    `<error>` XML node defined in the *Structured Error Channel* spec.
    """

    __slots__ = ("code", "detail", "retryable", "meta")

    def __init__(
        self,
        code: str,
        detail: str | None = None,
        *,
        retryable: bool = False,
        meta: dict | None = None,
    ) -> None:
        """Initialise the structured error.

        Parameters
        ----------
        code : str
            A short, upper-snake-case identifier for the error type
            (e.g., "VALIDATION_FAIL", "DB_TIMEOUT").
        detail : str | None, optional
            A human-readable description of the error. Defaults to None.
        retryable : bool, optional
            A hint for the client indicating whether the operation might
            succeed on a subsequent attempt. Defaults to False.
        meta : dict | None, optional
            A dictionary for additional, machine-readable context about
            the error. Defaults to None.
        """
        if not code or not isinstance(code, str):
            raise ValueError("'code' must be a non-empty string.")
        self.code: str = code.upper()
        self.detail: str = detail or ""
        self.retryable: bool = bool(retryable)
        self.meta: dict | None = meta
        super().__init__(self.detail or self.code)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __str__(self) -> str:  # noqa: D401 – simple repr
        return f"{self.code} - {self.detail}" if self.detail else self.code

    def __repr__(self) -> str:  # noqa: D401 – debug-friendly
        return (
            f"XMCPError(code={self.code!r}, detail={self.detail!r}, "
            f"retryable={self.retryable}, meta={self.meta!r})"
        )


class DependencyError(Exception):
    """Raised when a circular or otherwise invalid dependency graph is detected.

    The :class:`AIClient` raises this error as soon as it notices that the
    dependency requirements expressed via the ``depends_on`` annotation cannot
    be satisfied (e.g. because of a cycle like *A → B → A*).
    """
    pass 