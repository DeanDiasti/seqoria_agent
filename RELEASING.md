# Publishing Seqoria Agent

The package name is `seqoria-agent`; consumers import `seqoria_agent`.
Version 0.9.1 is published on PyPI. Nothing is published by editing the repository or building artifacts locally; the tagged release workflow must succeed. The maintainer approved the public alpha launch separately from package publication; see the [public alpha review](docs/maintainers/public-alpha.md).

## Existing PyPI project setup

1. The PyPI project already exists under `deandiasti`. Sign in with the owning account when checking project settings; keep passwords, recovery codes, and tokens out of the repository and chat.
2. Use the existing project’s [**Manage → Publishing** page](https://pypi.org/manage/project/seqoria-agent/settings/publishing/). Do not create a pending publisher for this existing project.
3. Verify the existing trusted publisher matches the release repository:

   | Field | Value |
   | --- | --- |
   | PyPI project | `seqoria-agent` |
   | Repository owner | `DeanDiasti` |
   | Repository | `seqoria_agent` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi` |

4. Verify the existing `pypi` GitHub environment restrictions for release tags. The workflow uses short-lived trusted publishing credentials; `PYPI_API_TOKEN` is not used.

See the official PyPI guide for [existing trusted publishers](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) when changing the release setup.

## Validate and release

1. Review [the release review](docs/maintainers/release-review.md) and the breaking changes in `CHANGELOG.md`.
2. Confirm version and project URLs in `pyproject.toml`. The metadata matches the existing Apache-2.0 license file.
3. From a clean checkout, run:

   ```bash
   python -m pip install -e '.[dev,docs]'
   ruff format --check .
   ruff check .
   python -m mypy
   python -m pytest --cov=seqoria_agent --cov-config=pyproject.toml --cov-fail-under=75
   python -m build --outdir /tmp/seqoria-release
   python -m twine check --strict /tmp/seqoria-release/*
   python scripts/check_distribution.py /tmp/seqoria-release
   mkdocs build --strict
   ```

   Use a fresh output directory for each build to avoid uploading old releases. Install the built wheel in a clean environment outside the checkout and import the core and HTTP providers. CI also checks this on Linux.

4. Commit and push the reviewed code. Tag that commit with the matching version (`vX.Y.Z` for version `X.Y.Z`), then push the tag when ready to publish.
5. The release workflow runs the test matrix and package checks, verifies the tag/version match, builds distributions, and publishes through the `pypi` environment. PyPI releases are immutable; fix a failed release by correcting the problem and choosing a new version if files have already been uploaded.
6. Verify installation in a new environment with `pip install seqoria-agent==X.Y.Z`, using the newly published version.
7. Review the documentation artifact before a manual Pages deployment. Confirm the hosted URL works before advertising it. Repository visibility, vulnerability reporting, and documentation deployment are separate launch actions; publishing a tag does not approve any of them.

For API removals and upgrade steps, read the [migration guide](docs/migration.md).
