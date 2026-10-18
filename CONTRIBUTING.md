# Contributing

Seqoria Agent is an alpha library for explicit conversation history, local Python tools, MCP clients, and HTTP model providers. Keep changes focused on that core. Discuss new public APIs, dependencies, and substantial behavior changes in an issue before implementing them.

Participation follows the [code of conduct](CODE_OF_CONDUCT.md). Use issues for ordinary reports and the [security policy](SECURITY.md) for vulnerabilities.

## Set up a checkout

Use Python 3.12 or 3.13. Create a virtual environment, then install editable development and documentation dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,docs]'
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. No API keys, model downloads, Redis, or hosted MCP service are needed for the default tests.

## Make and verify a change

Add a regression test that fails for the reported behavior and passes with the fix. Prefer HTTPX mock transports, deterministic provider responses, in-memory MCP sessions, and bounded async tests. Keep network-dependent experiments separate from the default suite and never commit credentials, real user histories, or sensitive tool results.

```bash
ruff format .
ruff check .
python -m mypy
python -m pytest --cov=seqoria_agent --cov-config=pyproject.toml --cov-report=term-missing --cov-fail-under=75
mkdocs build --strict
python -m build --outdir /tmp/seqoria-contribution-build
python -m twine check --strict /tmp/seqoria-contribution-build/*
python scripts/check_distribution.py /tmp/seqoria-contribution-build
```

Use a fresh build directory. CI checks formatting, lint, type checks across the package, tests with a 75% branch-coverage floor, strict documentation, and the full suite against an installed core wheel. Use `python -m pytest --cov=seqoria_agent --cov-config=pyproject.toml --cov-report=term-missing` to inspect untested paths relevant to your change. Coverage is evidence of exercised lines, not proof of correct behavior.

For concurrency changes, test separate user contexts, cancellation, stream closure, and cleanup after startup failure. For provider changes, test both sync and async paths, fragmented tool arguments, malformed responses, and failure before and after streamed output. For security changes, follow [SECURITY.md](SECURITY.md) before opening an issue with details.

## Find the right tests

Tests are grouped by behavior rather than the release in which a defect was found:

| Directory | Scope |
| --- | --- |
| `tests/client/` | Chat, streaming, manual steps, history, and provider fallback |
| `tests/execution/` | Tool scheduling, dependency ordering, timeouts, and cancellation |
| `tests/parameters/` | Registry resolution, hidden arguments, and trusted request identity |
| `tests/providers/` | Provider HTTP contracts, model validation, and wire conversions |
| `tests/mcp/` | Transports, discovery, connections, OAuth, and SDK interoperability |
| `tests/toolbox/` | Local registration, schemas, and trusted parameter injection |
| `tests/models/` | Message validation and serialization |
| `tests/configuration/` | Settings isolation, explicit overrides, and rejected options |
| `tests/security/` | Credential redaction in diagnostics, exceptions, and logs |
| `tests/package/` | Imports, exports, removed APIs, and package boundaries |
| `tests/examples/` | Runnable offline and local MCP example smoke tests |

For a focused provider change, run `python -m pytest tests/providers`. For a transport change, run `python -m pytest tests/mcp`. See the [test suite guide](tests/README.md) for local integration selectors and fixture conventions. Then run the full suite before submitting. Some regressions cross subsystem boundaries, so a focused directory does not replace full validation.

Examples are grouped under `examples/quickstart`, `examples/tools`, `examples/conversation`, `examples/mcp`, and `examples/providers`. Start with the [example catalog](examples/README.md); offline examples should remain runnable without credentials or services.

## Open a pull request

Explain the concrete problem, resulting behavior, compatibility impact, and validation. Update documentation and the Unreleased changelog when behavior or the public API changes. Include a minimal reproduction for bug fixes. Avoid unrelated formatting changes beyond the enforced formatter.

Public APIs may change during alpha, but removals still need migration instructions. Configuration shared by a client must not become implicit per-user state. Keep identity and authorization in trusted application context, preserve tool call/result pairing, and make ownership of async resources explicit.

Contributions are provided under the repository's [Apache-2.0 license](LICENSE.txt). Do not include code or data you are not entitled to contribute. Maintainers handle release tags, trusted publishing, and repository visibility separately from merging contributions.
