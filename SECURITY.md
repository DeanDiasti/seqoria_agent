# Security policy

## Supported versions

Security fixes target the latest published alpha release. Older alpha releases do not receive a guaranteed backport. The current published version is 0.9.0; fixes on the default branch are unreleased until a new PyPI version is published. Review [CHANGELOG.md](CHANGELOG.md) before upgrading and pin versions for reproducible deployments.

## Report a vulnerability privately

Do not open a public issue or pull request containing credentials, exploit details, private endpoints, or user data.

Use GitHub for reporting. If **Security → Report a vulnerability** is available on the repository, submit sensitive details there. GitHub private vulnerability reporting has not been confirmed enabled; this policy does not claim that the feature is active or change any repository setting.

If that option is unavailable, open a minimal [GitHub issue](https://github.com/DeanDiasti/seqoria_agent/issues) requesting a private advisory or private vulnerability-reporting route. Include only the affected package version and a general problem category. Wait for a maintainer to establish a private GitHub route before sharing a reproduction or exploit details. Issues become public after launch, so do not include secrets, sensitive data, or names of affected people. No email contact is provided, and there is no promised response time or formal bug bounty.

A useful private report includes the affected version, a minimal reproduction, impact, prerequisites, and proposed mitigation. Replace secrets and personal data with synthetic values. Coordinate disclosure with the maintainer before publishing a reproduction.

## Application security boundaries

Seqoria Agent orchestrates model calls and tools. Applications remain responsible for authentication, authorization, conversation ownership, secret storage, tenant isolation in external services, request limits, and retention. A tool schema is not an authorization policy, and model output is untrusted input.

Hidden local parameters should come from authenticated request context or trusted `ParameterRegistry` sources. Do not enable model overrides for identity fields. Configure remote MCP sessions separately when different end users need different credentials, and authorize every operation on the remote server.

Prefer authorization headers over credentials embedded in URLs. Redaction recognizes URL userinfo, query values, fragments, and common credential assignments; it cannot reliably identify arbitrary secrets in paths, query names, bare query tokens, relative URLs, or free prose. The library redacts endpoint credentials in its own diagnostic messages, but this does not control logging by your application, HTTPX, the MCP SDK, proxies, or tracing integrations. Review those separately before enabling verbose logs. Avoid logging full prompts, tool arguments, results, or OAuth callbacks.

Transport retries and interrupted turns can leave side effects with an uncertain outcome. Use idempotency keys and bounded underlying I/O where needed; cancelling a synchronous tool cannot stop its already-running Python thread.
