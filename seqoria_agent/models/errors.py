"""Error types for tool execution and client orchestration."""


class ToolRemoteError(Exception):
    """Wraps exceptions raised during remote tool execution."""


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open for a given server and tool calls are blocked."""


class DependencyError(Exception):
    """Raised when a circular or otherwise invalid dependency graph is detected.

    The :class:`AIClient` raises this error as soon as it notices that the
    dependency requirements expressed via the ``depends_on`` annotation cannot
    be satisfied (e.g. because of a cycle like *A → B → A*).
    """

    pass
