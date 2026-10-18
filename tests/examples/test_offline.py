"""Keep the documented, service-free examples executable from an installed package."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("example", "expected"),
    [
        ("quickstart/offline.py", ("One-shot: pong", "Streaming: pong", "Illustrative usage:")),
        (
            "tools/local_tools.py",
            (
                "Allowed repetitions: 1 to 3",
                "valid: hello hello",
                "invalid: rejected invalid arguments",
            ),
        ),
        ("tools/manual_loop.py", ("Tool: add -> 5", "Assistant: 2 + 3 = 5")),
        (
            "conversation/request_context.py",
            (
                "tenant-a/alice (5 messages in its own history)",
                "tenant-b/bob (5 messages in its own history)",
            ),
        ),
        pytest.param(
            "mcp/stdio_client.py",
            ("8 / 2 = 4",),
            marks=pytest.mark.integration,
            id="mcp-stdio",
        ),
    ],
)
def test_offline_example(example, expected, tmp_path):
    """Execute each finite example without credentials or a checkout working directory."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(part in key.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    }
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(REPOSITORY / "examples" / example)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for output in expected:
        assert output in result.stdout
