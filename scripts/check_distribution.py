"""Check built archives for required metadata and retired/internal files."""

from __future__ import annotations

import argparse
import email
import tarfile
import tomllib
import zipfile
from pathlib import Path

RETIRED = (
    ".plan_mode_artifacts/",
    "seqoria_agent/memory/",
    "seqoria_agent/server/",
    "seqoria_agent/client/cache/",
    "mlx_lm_provider",
    "mlx_vlm_provider",
    "consts_runtime.py",
    "__pycache__/",
    ".env",
    ".coverage",
)


def check(directory: Path) -> None:
    expected = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, "Use a clean build directory with one wheel and sdist"
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        assert "seqoria_agent/py.typed" in names, "Missing PEP 561 marker"
        meta = next(name for name in names if name.endswith(".dist-info/METADATA"))
        info = email.message_from_bytes(wheel.read(meta))
        assert info["Version"] == expected
        assert info["License-Expression"] == "Apache-2.0"
        assert info["Requires-Python"] == ">=3.12"
        assert info.get_payload(decode=True).decode("utf-8") == Path("README.md").read_text(), (
            "README differs from package metadata"
        )
        assert not any("mlx" in item.lower() for item in info.get_all("Requires-Dist", []))
        assert set(info.get_all("Provides-Extra", [])) == {
            "all",
            "dev",
            "docs",
            "performance",
            "server",
        }
        assert any(name.endswith("LICENSE.txt") for name in names)
        assert all(
            not name.startswith(("tests/", "docs/", "examples/", "benchmarks/")) for name in names
        )
    with tarfile.open(sources[0]) as source:
        source_names = [name.split("/", 1)[-1] for name in source.getnames()]
        for required in (
            "README.md",
            "LICENSE.txt",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "SUPPORT.md",
            "CODE_OF_CONDUCT.md",
            "RELEASING.md",
            "tests/README.md",
            "tests/package/test_exports.py",
            "examples/README.md",
            "examples/quickstart/offline.py",
            "benchmarks/multi_user.py",
            "scripts/check_distribution.py",
            "seqoria_agent/py.typed",
        ):
            assert required in source_names, f"Missing from sdist: {required}"
        names += source_names
    for name in names:
        assert not any(retired in name for retired in RETIRED), f"Unwanted archive member: {name}"
    print(
        f"Verified {expected} wheel/sdist metadata, typed marker, contributor files, and archive hygiene."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
