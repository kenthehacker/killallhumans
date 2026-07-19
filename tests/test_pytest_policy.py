from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _run_nested_pytest(*args: str, timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "pytest_timeout",
            "-o",
            "required_plugins=",
            "-c",
            str(_PYPROJECT),
            *args,
        ],
        cwd=_PYPROJECT.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )


def test_pytest_timeout_kills_a_deliberately_stuck_test(tmp_path):
    stuck_test = tmp_path / "test_stuck.py"
    stuck_test.write_text(
        "import time\n\ndef test_stuck():\n    time.sleep(60)\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    result = _run_nested_pytest("--timeout=0.20", str(stuck_test))
    elapsed = time.monotonic() - started
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert elapsed < 4.0
    assert "timeout" in output.lower()


def test_unknown_pytest_marker_fails_collection(tmp_path):
    unknown_marker_test = tmp_path / "test_unknown_marker.py"
    unknown_marker_test.write_text(
        "import pytest\n\n@pytest.mark.not_registered_here\ndef test_noop():\n    pass\n",
        encoding="utf-8",
    )
    result = _run_nested_pytest("--collect-only", str(unknown_marker_test))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "not found in `markers` configuration option" in output


def test_default_selection_does_not_execute_slow_benchmark_or_live(tmp_path):
    selection_test = tmp_path / "test_selection.py"
    selection_test.write_text(
        """import pytest

def test_default_runs():
    pass

@pytest.mark.slow
def test_slow_does_not_run():
    raise AssertionError('slow tier leaked into default selection')

@pytest.mark.benchmark
def test_benchmark_does_not_run():
    raise AssertionError('benchmark tier leaked into default selection')

@pytest.mark.live
def test_live_does_not_run():
    raise AssertionError('live tier leaked into default selection')
""",
        encoding="utf-8",
    )
    result = _run_nested_pytest(str(selection_test))
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 passed" in output
    assert "3 deselected" in output
