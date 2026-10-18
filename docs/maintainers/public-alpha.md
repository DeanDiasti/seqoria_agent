# Public alpha review

The repository stays private until the maintainer approves a separate public launch. Version 0.9.1 contains the reviewed hardening and cleanup on top of the published 0.9.0 baseline. The maintainer authorized committing, pushing, tagging, and publishing 0.9.1 on 2026-10-18 while keeping GitHub private. Package publication does not deploy documentation or change repository visibility.

## Review the code and package

Before approving launch, review provider regressions covering complete model-to-tool arguments, sync/async streaming, fragmented and malformed responses, errors before and after output, and cancellation. Check endpoint diagnostic redaction, independent user contexts, history pairing, bounded concurrency, and resource cleanup after success, failure, or stream abandonment.

Run formatting, lint, tests with coverage, strict documentation, wheel/source builds, metadata checks, and installation tests outside the checkout. Inspect source and wheel contents for obsolete experiments, private data, or unexpected files. Record the exact commit, environment, commands, and results in the launch review; do not infer readiness from a test count alone.

A deterministic benchmark is included under `benchmarks/`. From an installed checkout, run:

```bash
python benchmarks/multi_user.py --users 100 --rounds 5 --tool-latency-ms 5 --max-concurrency 16
```

Review the [benchmark guide](https://github.com/DeanDiasti/seqoria_agent/blob/main/benchmarks/README.md) and JSON output. It uses synthetic model responses and local tools. Compare results on the same machine and configuration, and measure the real application's hosted-provider/network/database workload separately before sizing a deployment.

## Review repository exposure

Review tracked files, full Git history, release artifacts, and existing Actions logs before changing visibility. A history scanner can find credential patterns; it cannot prove that business data, endpoint names, or all secret formats are absent. Rotate any discovered credential before rewriting or publishing history, and assess existing downloadable artifacts and logs separately.

Use GitHub issues for ordinary support and conduct reports, minimizing personal or sensitive details because issues become public after launch. Sensitive security reports should use GitHub private vulnerability reporting when available; otherwise reporters can open a minimal issue requesting a private advisory or reporting route, without exploit details. The [security policy](https://github.com/DeanDiasti/seqoria_agent/blob/main/SECURITY.md) does not claim that private reporting is currently enabled. No email contact or separate confidential conduct-reporting channel is required for this alpha.

Review issue and pull-request templates, repository description/topics, license, PyPI metadata, and provider reference pages. Keep examples free of real tokens, user histories, and private endpoints.

## Documentation and publication

The ordinary documentation workflow validates a build. A manual deployment workflow, if configured, still requires an explicit deployment and GitHub Pages configuration. Review its generated artifact first. Do not advertise the hosted Pages URL as live until deployment and access have been verified.

After the hardening is approved, choose a new version, update release notes, and publish through the existing trusted-publisher workflow. PyPI release files are immutable. Verify a fresh install of the actual release and its rendered description before announcing it. Repository visibility and documentation deployment each require their own reviewed action.

## Alpha expectations

Supported Python versions are 3.12 and 3.13. Public APIs may change during alpha; release notes should explain migrations. There is no long-term support or response-time guarantee. Use fixed versions for reproducible applications.

The deterministic suite does not certify every provider model, OpenAI-compatible implementation, MCP host, OAuth deployment, or operating environment. Production integrations need workload-specific tests and monitoring. Applications own authentication, authorization, history retention, rate limits, durable side effects, and cross-worker conversation coordination.

## Local evidence — 2026-10-18

This section records local pre-release checks of the hardening work based on commit `97d20044de4dc728fb11c9d0855b41b6fd0cfa04`, the 0.9.0 baseline. The evidence was collected before the 0.9.1 release commit and publication. No Pages deployment, visibility change, or reporting-setting change was performed during this review.

The local environment was Python 3.12.11 on macOS 15.7.3/arm64, HTTPX 0.28.1, MCP 1.30.0, Pydantic 2.14.1, Ruff 0.17.1, and mypy 2.4.0.

| Check | Result |
| --- | --- |
| Regression suite | 623 passed, including 319 additional cases since the published baseline; all 618 pre-organization cases preserved |
| Full package statement + branch coverage | 76.96%; CI minimum is 75%; no modules omitted |
| Installed core wheel | 623 tests passed outside the source checkout with only core and test dependencies |
| Optional runtime extras and dependency floors | 623 tests passed with all runtime extras, and 623 passed with minimum direct versions: HTTPX 0.28.1, MCP 1.30.0, Pydantic 2.12.5 |
| Formatting, import ordering, lint | Passed across all 128 formatted files |
| Type checking | Passed across all 31 package modules; PEP 561 marker included in the wheel |
| Documentation | Strict MkDocs build passed; local links across 22 Markdown documents resolve; provider, parameter, and MCP APIs rendered |
| Workflows | All workflow files passed actionlint 1.7.12; actions pinned to verified commits |
| Package | Wheel/source builds, strict Twine checks, metadata/README/license checks, and archive hygiene checks passed |
| Runtime dependency audit | No known vulnerabilities in a clean core installation or a clean installation including all runtime extras |
| Secret scan | Gitleaks 8.30.1 found no leaks in the publishable candidate files, scanned history (55 commits), 17 downloadable Actions log archives, or all five retained Actions artifacts, including nested distributions |

Fourteen older completed Actions runs (December 2025–April 2026) had no downloadable logs. They were not scanned. The ignored local `.env` is excluded from Git history and release archives. Secret detection is pattern-based and does not certify that all private data or unknown credential formats are absent.

Documentation is grouped into quickstart, practical guides, API references, migration, and maintainer evidence. Nine example programs have explicit prerequisites and ownership rules; five smoke cases execute the four offline demonstrations and the local MCP stdio roundtrip. Tests are grouped by behavior in eleven directories. Collection comparison preserved every original class, function, and parameter case; the five example cases are the only additions during reorganization.

The dependency audit covers the resolved supported runtime environment, not unrelated legacy packages remaining in a developer's virtual environment. The minimum direct MCP dependency was raised to 1.30.0 to include the audited security fixes. CI audits all runtime extras and schedules fresh dependency/secret checks; push and pull-request secret checks scan event commits, while scheduled/manual runs inspect history. A new hosted CI run is still required after the reviewed changes are pushed.

### Synthetic benchmark evidence

Each scenario used 100 simultaneous conversations, five sequential turns per conversation (500 requests), 5 ms artificial async tool latency, and two in-process HTTP model calls per turn. These single-run measurements came from the installed core wheel on the machine described above.

| Tool concurrency cap | Requests/s | p50 latency, ms | p95 latency, ms | Peak active tools |
| --- | --- | --- | --- | --- |
| 1 | 173.64 | 571.850 | 581.586 | 1 |
| 16 | 2385.57 | 38.668 | 45.482 | 16 |
| 64 | 3829.32 | 25.567 | 31.423 | 64 |

All three scenarios passed identity isolation, concurrency-cap enforcement, cancellation cleanup, recovery after cancellation, HTTP closure, and background-task cleanup checks. These timings measure local orchestration and queueing; they do not measure hosted models, remote MCP servers, database capacity, or production throughput. Repeat the benchmark on the intended deployment before using it for sizing.

### Review and launch decisions

The maintainer authorized the 0.9.1 release after review. Push the fixes and require the fresh CI matrix to pass before tagging; the tag workflow repeats validation before publication. Keep the repository private. A separate approved launch step can change repository visibility, enable GitHub private vulnerability reporting, configure branch protection/required checks as available, and deploy the reviewed Pages site. No email address or separate private conduct channel is required by the maintainer's reporting preference.
