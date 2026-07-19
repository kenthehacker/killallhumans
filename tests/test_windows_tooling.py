from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DEV = _ROOT / "scripts" / "dev.ps1"
_DEV_CMD = _ROOT / "scripts" / "dev.cmd"
_LAUNCHER = _ROOT / "scripts" / "launch_sim.ps1"
_P2_ENTRYPOINTS = tuple(
    _ROOT / "scripts" / name
    for name in (
        "aigp_campaign.py",
        "aigp_nonlive.py",
        "aigp_promote.py",
        "aigp_replay.py",
        "aigp_trials.py",
    )
)


def test_windows_command_surface_contains_all_passive_development_tasks():
    source = _DEV.read_text(encoding="utf-8")
    for task in (
        "test-target",
        "test-fast",
        "test-unit",
        "test-vq2",
        "test-slow",
        "test-benchmark",
        "test-full-non-live",
        "preflight",
        "launch-sim",
        "sbom",
    ):
        assert f"'{task}'" in source
    assert "--stage', 'preflight'" in source
    assert "--stage', 'gate0'" not in source
    assert "'unit and not live'" in source
    assert "'slow and not live'" in source
    assert "'benchmark and not live'" in source


def test_windows_python_tasks_use_external_process_scoped_bytecode_cache():
    source = _DEV.read_text(encoding="utf-8")
    assert "PYTHONPYCACHEPREFIX" in source
    assert "[IO.Path]::GetTempPath()" in source
    assert "[Guid]::NewGuid()" in source
    assert "bytecode cache prefix must be outside the repository" in source
    assert source.index("$env:PYTHONPYCACHEPREFIX = $PycachePrefix") < source.index(
        "& $Python @PythonArgs"
    )


def test_windows_cmd_bootstrap_bypasses_machine_execution_policy():
    source = _DEV_CMD.read_text(encoding="utf-8")
    assert "-ExecutionPolicy Bypass" in source
    assert '"%~dp0dev.ps1" %*' in source
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(_DEV_CMD), "help"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Available tasks" in result.stdout


def test_target_executes_explicit_benchmark_but_still_excludes_live(tmp_path):
    probe = tmp_path / "test_dev_target_markers.py"
    probe.write_text(
        """\
import pytest

@pytest.mark.benchmark
def test_benchmark_probe():
    assert True

@pytest.mark.live
def test_live_probe():
    raise AssertionError("test-target must never execute live tests")
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            str(_DEV_CMD),
            "test-target",
            f"{probe}::test_benchmark_probe",
            f"{probe}::test_live_probe",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "1 deselected" in result.stdout


def test_launcher_is_parameterized_and_has_no_stale_host_or_session_id():
    source = _LAUNCHER.read_text(encoding="utf-8")
    assert "[string]$SimulatorPath" in source
    assert "AIGP_FLIGHTSIM_PATH" in source
    assert "query.exe" in source
    assert "$InteractiveSessionId" in source
    assert "SessionId -contains $InteractiveSessionId" in source
    assert "refusing to relaunch" in source
    assert "already exists; refusing to overwrite" in source
    assert "/RL HIGHEST /F" not in source
    assert "Could not delete temporary launcher task" in source
    assert "[Guid]::NewGuid()" in source
    assert source.index("Get-Process") < source.index("query.exe")
    assert "Kenichi" not in source
    assert "3364" not in source
    assert "SessionId -contains 1" not in source


@pytest.mark.parametrize("script", _P2_ENTRYPOINTS, ids=lambda path: path.name)
def test_p2_python_entrypoints_support_direct_help_without_package_install(script):
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell syntax check is Windows-only")
@pytest.mark.parametrize("script", [_DEV, _LAUNCHER], ids=lambda path: path.name)
def test_powershell_scripts_parse_without_execution(script):
    command = (
        "$tokens=$null; $errors=$null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        "$env:AIGP_PS_PARSE_PATH,[ref]$tokens,[ref]$errors); "
        "if ($errors.Count) { $errors | Out-String | Write-Error; exit 1 }"
    )
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "AIGP_PS_PARSE_PATH": str(script)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
