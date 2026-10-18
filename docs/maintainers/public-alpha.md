# Public alpha review

The maintainer approved making the repository public on 2026-10-18 after the 0.9.1 release review. Version 0.9.1 contains the reviewed hardening and cleanup on top of the published 0.9.0 baseline. Package publication, repository visibility, security settings, and documentation deployment are verified separately.

## Verified public launch — 2026-10-18

The repository is public, and anonymous access to its API metadata and source README was verified. The [documentation site](https://deandiasti.github.io/seqoria_agent/) deployed successfully; the homepage, quickstart, web application guide, client/provider references, launch record, and search index returned HTTP 200 without authentication.

GitHub private vulnerability reporting, secret scanning, push protection, dependency vulnerability alerts, and Dependabot security-fix pull requests are enabled. At verification there were no open secret-scanning alerts; this supplements the scans below without certifying the absence of every possible secret.

`main` requires a pull request, up-to-date branches, eleven successful GitHub Actions checks, resolved conversations, and linear history. The checks cover quality, package installation, dependency and secret scans, documentation, and Python 3.12/3.13 on Linux/macOS/Windows. Protection applies to administrators; force pushes and branch deletion are blocked. The solo-maintainer workflow requires zero additional review approvals, so it does not claim independent reviewer sign-off.

The launch commit `ec152ab002e5d1cdf3f273906ab3db1cdee8ce67` passed hosted CI and documentation checks before visibility changed. A local rerun passed all 623 tests, and a fresh archive scan plus history scan through the 0.9.1 release (56 commits) found no leaks. The live-check limitations below remain part of the alpha's published evidence.

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

Use GitHub issues for ordinary support and conduct reports, minimizing personal or sensitive details because issues are public. Sensitive security reports should use GitHub private vulnerability reporting; if unavailable, reporters can open a minimal issue requesting a private advisory or reporting route, without exploit details. Follow the [security policy](https://github.com/DeanDiasti/seqoria_agent/blob/main/SECURITY.md). No email contact or separate confidential conduct-reporting channel is required for this alpha.

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

The dependency audit covers the resolved supported runtime environment, not unrelated legacy packages remaining in a developer's virtual environment. The minimum direct MCP dependency was raised to 1.30.0 to include the audited security fixes. CI audits all runtime extras and schedules fresh dependency/secret checks; push and pull-request secret checks scan event commits, while scheduled/manual runs inspect history. Hosted CI, documentation, and the tagged publishing workflow all passed for release commit `3cb1751088c4eb1397b2f899181b28fc1888f35f`. A fresh installation of 0.9.1 from PyPI passed all 623 tests outside the checkout; its README and both release archive digests were verified.

### Synthetic benchmark evidence

Each scenario used 100 simultaneous conversations, five sequential turns per conversation (500 requests), 5 ms artificial async tool latency, and two in-process HTTP model calls per turn. These single-run measurements came from the installed core wheel on the machine described above.

| Tool concurrency cap | Requests/s | p50 latency, ms | p95 latency, ms | Peak active tools |
| --- | --- | --- | --- | --- |
| 1 | 173.64 | 571.850 | 581.586 | 1 |
| 16 | 2385.57 | 38.668 | 45.482 | 16 |
| 64 | 3829.32 | 25.567 | 31.423 | 64 |

All three scenarios passed identity isolation, concurrency-cap enforcement, cancellation cleanup, recovery after cancellation, HTTP closure, and background-task cleanup checks. These timings measure local orchestration and queueing; they do not measure hosted models, remote MCP servers, database capacity, or production throughput. Repeat the benchmark on the intended deployment before using it for sizing.

### Review and launch decisions

The maintainer authorized the 0.9.1 release after review, then approved the public alpha launch. Launch setup includes a repository description and topics, GitHub private vulnerability reporting, required checks on `main`, and the reviewed Pages documentation. No email address or separate private conduct channel is required by the maintainer's reporting preference.

### Live provider checks — 2026-10-18

`scripts/check_live_provider.py` performs one ordinary and one streaming model/tool/result/assistant roundtrip. Each mode checks trusted identity injection, exactly one local tool invocation, paired call IDs, the returned proof string, and the final response; streaming also checks text deltas. It uses synthetic data, disables retries, bounds output and time, closes owned resources, and never pulls models.

| Adapter and service | Model | Result |
| --- | --- | --- |
| Anthropic Messages API | `claude-haiku-4-5-20251001` | Ordinary and streaming roundtrips passed |
| Ollama native API, installed local model | `qwen3.8-27b-abliterated:latest` | Ordinary and streaming roundtrips passed |
| Compatible Chat Completions API, local Ollama endpoint | `qwen3.8-27b-abliterated:latest` | Ordinary and streaming roundtrips passed with `reasoning_effort=none` |
| OpenAI Responses API | `gpt-4.1-nano` | Generation blocked by account HTTP 429, `insufficient_quota` / `credit_balance_exhausted` |
| Compatible Chat Completions API, OpenRouter | `openai/gpt-4.1-nano` | Configured credential rejected; no successful generation |

Account quota and credential failures are not successful live validations. OpenAI's Responses adapter remains covered by deterministic HTTP contract tests; it needs a live rerun once the account has credit. OpenRouter needs a valid credential before it can be certified for the chosen model. The compatible adapter's successful local check does not certify other compatible services. These limits are recorded for the public alpha rather than presented as universal provider validation.

To repeat a check with an accessible model and credentials supplied through the environment:

```bash
python scripts/check_live_provider.py --provider openai --model YOUR_MODEL_ID
python scripts/check_live_provider.py --provider anthropic --model YOUR_MODEL_ID
python scripts/check_live_provider.py --provider compatible --model YOUR_MODEL_ID
python scripts/check_live_provider.py --provider ollama --model YOUR_INSTALLED_MODEL
```

For `compatible`, set `SEQORIA_BASE_URL` and `SEQORIA_API_KEY`; `--reasoning-effort` is optional and endpoint-specific. OpenAI and Anthropic use their standard API-key environment variables. The script does not load local dotenv files or run in default CI. It makes up to four bounded generation requests and may incur provider charges.
