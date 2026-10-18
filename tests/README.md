# Test suite

Tests are organized by behavior so a failure points to the part of the public contract it protects. Historical release and audit names are replaced by these groups:

| Directory | What belongs here |
| --- | --- |
| `client/` | Chat loops, history ownership, input validation, sync adapters, tool filtering and provider fallback. |
| `execution/` | Tool results and errors, dependency ordering, streaming, shared concurrency and timeouts. |
| `parameters/` | Registry resolution, trusted request identity, hidden argument injection and structured values. |
| `mcp/` | Discovery, routing, dynamic servers, retries, structured results, transport ownership, OAuth and SDK interoperability. |
| `toolbox/` | Tool registration, schemas, direct execution and explicit hidden parameters. |
| `providers/` | HTTP wire contracts, SSE framing, model validation, reasoning and usage, tool conversion, resources and redaction. |
| `models/` | Message round trips, validation and container behavior. |
| `configuration/` | Settings isolation, explicit overrides and rejected options. |
| `security/` | Credential redaction in logs, surfaced exceptions, callbacks and conversation history. |
| `package/` | Supported public exports, removed modules and optional dependency loading. |
| `examples/` | Executable examples checked against their documented output. |

Install the development dependencies, then run the suite from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

The default suite needs no API credentials, external services or downloaded models. Provider tests use deterministic HTTPX transports. Tests marked `integration` exercise local MCP SDK interoperability or stdio subprocesses; they still run without external services.

```bash
python -m pytest -m 'not integration and not slow'
python -m pytest -m integration
python -m pytest tests/parameters tests/execution
python -m pytest --cov=seqoria_agent --cov-branch --cov-report=term-missing
```

Keep fixtures and scripted providers close to the behavior that uses them. Share a fixture only when it represents the same protocol or lifecycle across multiple tests. Check complete request-to-result behavior, preserve parameterized edge cases, and use the real MCP SDK for interoperability checks rather than expanding local mocks to mimic it.

When moving tests, compare collection IDs without the file prefix to ensure that class names, test names and parameter cases remain present. Use `python -m pytest --collect-only -q -o addopts=''` before and after the move. The configured import mode is `importlib`, so tests with the same filename can coexist across these packages and the suite can also run against an installed wheel.
