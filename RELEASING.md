# Publishing Seqoria Agent

The package name is `seqoria-agent`; consumers import `seqoria_agent`.
The 0.7 series is alpha; 0.7.0 introduced breaking changes from 0.6. Version 0.7.1 repairs retained features. Nothing is published by editing the repository or building artifacts locally.

## Existing PyPI project setup

1. Sign in to the existing PyPI account `deandiasti`. The `seqoria-agent` project already exists under this account. A public metadata lookup can return 404 for a project with no releases, so it does not establish that a project name is available. Keep passwords, recovery codes, and tokens out of this repository and chat.
2. Use the existing project’s [**Manage → Publishing** page](https://pypi.org/manage/project/seqoria-agent/settings/publishing/). Do not create a pending publisher for this existing project.
3. Register the following GitHub publisher, matching the repository actually used for release:

   | Field | Value |
   | --- | --- |
   | PyPI project | `seqoria-agent` |
   | Repository owner | `DeanDiasti` |
   | Repository | `seqoria_agent` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi` |

4. Create the `pypi` GitHub environment in the repository settings. Restrict who can release and which tags may deploy. The workflow uses short-lived trusted publishing credentials; `PYPI_API_TOKEN` is not used.

See the official PyPI guides for [new projects](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) and [existing projects](https://docs.pypi.org/trusted-publishers/adding-a-publisher/). If another account owns the package name, ownership or a different available name is required before release.

## Validate and release

1. Review `docs/release-review.md` and the breaking changes in `CHANGELOG.md`.
2. Confirm version and project URLs in `pyproject.toml`. The metadata matches the existing Apache-2.0 license file.
3. From a clean checkout, run:

   ```bash
   python -m pip install -e '.[dev,docs]'
   ruff check .
   python -m pytest
   python -m build --outdir /tmp/seqoria-release
   python -m twine check --strict /tmp/seqoria-release/*
   mkdocs build --strict
   ```

   Use a fresh output directory for each build to avoid uploading old releases. Install the built wheel in a clean environment outside the checkout and import the core and HTTP providers. CI also checks this on Linux.

4. Commit and push the reviewed code. Tag that commit with the matching version (`v0.7.2` for the prepared metadata), then push the tag when ready to publish.
5. The release workflow runs the test matrix and package checks, verifies the tag/version match, builds distributions, and publishes through the `pypi` environment. PyPI releases are immutable; fix a failed release by correcting the problem and choosing a new version if files have already been uploaded.
6. Verify installation in a new environment with `pip install seqoria-agent==0.7.2`.

## Migrating from 0.6

`AgenticMemory`, configuration models, automatic planning, background capture, embeddings, vector/database adapters, and memory-specific examples/extras have been removed. Replace memory objects with a per-conversation list. Remove `retrieval_query` arguments. Load and save conversation history in your own authorized application storage; serialize model instances with `model_dump(mode="json")` before storage.

Automatic summaries and retrieval are no longer performed. If needed, build context explicitly in the application before calling the client. Existing persisted memory files are application data; export needed conversation messages before migrating. The library does not delete or migrate those files.

For remote MCP servers, hidden parameters resolve on the server from its authenticated context. Never trust arbitrary identity arguments sent by remote callers. Local tools receive trusted per-request values through the executor's `context`.
