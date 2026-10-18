# Support

Seqoria Agent is alpha software. API changes and integration defects are possible; there is no support SLA or production certification.

Start with the [README](README.md), [quickstart](docs/quickstart.md), and [web application guide](docs/guides/web-applications.md). The documentation source is available in this repository. A hosted documentation site is not yet part of the reviewed launch; build it locally with `mkdocs serve` after installing `.[docs]`.

For bugs and small usage questions, use [GitHub issues](https://github.com/DeanDiasti/seqoria_agent/issues). Issues are public. Include the package version, Python and operating-system versions, provider/transport type, a minimal reproduction, expected behavior, and a sanitized traceback. Do not include API keys, credential-bearing URLs, personal data, or real conversation histories.

For feature proposals, describe the application problem and why the existing supported core cannot meet it. New memory systems, server wrappers, caches, and model runtimes require an explicit design discussion; they are outside the current core.

For vulnerabilities, follow [the security policy](SECURITY.md). For participation concerns, follow [the code of conduct](CODE_OF_CONDUCT.md). Service outages, model access, provider billing, and independently hosted MCP servers belong with their respective operator.
