from __future__ import annotations

import json
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


def _run_launcher_session_selector(
    session_lines: list[str], query_exit_code: int
) -> subprocess.CompletedProcess[str]:
    command = r"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    $env:AIGP_PS_PARSE_PATH,
    [ref]$tokens,
    [ref]$errors
)
if ($errors.Count) { throw 'launch_sim.ps1 did not parse' }
$functions = @(
    $ast.FindAll(
        {
            param($node)
            $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
                $node.Name -eq 'Select-AigpInteractiveSession'
        },
        $true
    )
)
if ($functions.Count -ne 1) { throw 'session selector is not unique' }
Invoke-Expression $functions[0].Extent.Text
$lines = [string[]]($env:AIGP_SESSION_LINES_JSON | ConvertFrom-Json)
try {
    $selected = Select-AigpInteractiveSession `
        -SessionLines $lines `
        -QueryExitCode ([int]$env:AIGP_QUERY_EXIT_CODE)
    $selected | ConvertTo-Json -Compress
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 7
}
"""
    return subprocess.run(
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
        timeout=10,
        env={
            **os.environ,
            "AIGP_PS_PARSE_PATH": str(_LAUNCHER),
            "AIGP_SESSION_LINES_JSON": json.dumps(session_lines),
            "AIGP_QUERY_EXIT_CODE": str(query_exit_code),
        },
    )


def test_windows_command_surface_separates_tests_from_dedicated_powered_cycle():
    source = _DEV.read_text(encoding="utf-8")
    for task in (
        "test-target",
        "test-fast",
        "test-unit",
        "test-vq2",
        "test-slow",
        "test-benchmark",
        "test-promotion",
        "test-full-non-live",
        "preflight",
        "flight-cycle",
        "launch-sim",
        "sbom",
    ):
        assert f"'{task}'" in source
    assert "--stage', 'preflight'" in source
    assert "scripts.aigp_vq2_fast_cycle" in source
    assert "'-E', '-s', '-B'" in source
    assert "dedicated powered command, not a test task" in source
    assert "interactive confirmation" in source
    assert "'--stage', 'preflight', '--record'" not in source
    assert "'unit and not live'" in source
    assert "'slow and not live'" in source
    assert "'benchmark and not live'" in source
    assert "scripts.aigp_promotion_runner" in source
    assert "10-13 minute promotion boundary" in source
    assert "test-full-non-live is a compatibility alias" in source
    assert source.count("Invoke-PromotionTests $TaskArgs") == 2
    assert "accepts only the optional --fresh recovery flag" in source
    assert "'-m', 'pytest', '-q', '-m', 'not live', '--timeout=300'" not in source


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
    assert "test-promotion" in result.stdout
    assert "test-full-non-live" in result.stdout
    assert "flight-cycle" in result.stdout
    assert "10-13 minute promotion boundary" in result.stdout


@pytest.mark.parametrize("task", ["test-promotion", "test-full-non-live"])
def test_promotion_task_names_reject_unreviewed_arguments(task):
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(_DEV_CMD), task, "--unreviewed"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "accepts only the optional --fresh recovery flag" in (
        result.stdout + result.stderr
    )


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
    assert "$taskQueryErrorAction = $ErrorActionPreference" in source
    assert "$ErrorActionPreference = 'Continue'" in source
    assert "$ErrorActionPreference = $taskQueryErrorAction" in source
    assert "$CurrentSessionId -eq $InteractiveSessionId" in source
    assert "$RunAsUser -ieq $CurrentIdentity" in source
    assert "Start-Process -FilePath $SimulatorPath" in source
    assert "[Guid]::NewGuid()" in source
    assert source.index("Get-Process") < source.index("$querySession = Join-Path")
    assert "Kenichi" not in source
    assert "3364" not in source
    assert "SessionId -contains 1" not in source


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell test is Windows-only")
@pytest.mark.parametrize("query_exit_code", [0, 1])
def test_launcher_accepts_bounded_active_session_table_for_observed_exit_codes(
    query_exit_code,
):
    result = _run_launcher_session_selector(
        [
            " SESSIONNAME       USERNAME                 ID  STATE   TYPE        DEVICE ",
            " services                                    0  Disc                        ",
            ">console           John                      1  Active                      ",
        ],
        query_exit_code,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {
        "Current": True,
        "Session": "console",
        "User": "John",
        "Id": 1,
    }


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell test is Windows-only")
@pytest.mark.parametrize(
    "active_rows, expected",
    [
        (
            [
                " console           John                      1  Active",
                ">rdp-tcp#4         John                      9  Active",
            ],
            {"Current": True, "Session": "rdp-tcp#4", "User": "John", "Id": 9},
        ),
        (
            [
                " rdp-tcp#2         Other                     2  Active",
                " console           John                      8  Active",
            ],
            {"Current": False, "Session": "console", "User": "John", "Id": 8},
        ),
        (
            [
                " rdp-tcp#7         Other                     7  Active",
                " rdp-tcp#3         John                      3  Active",
            ],
            {"Current": False, "Session": "rdp-tcp#3", "User": "John", "Id": 3},
        ),
    ],
)
def test_launcher_preserves_current_console_then_lowest_id_priority(
    active_rows, expected
):
    result = _run_launcher_session_selector(
        [
            " SESSIONNAME       USERNAME                 ID  STATE   TYPE        DEVICE ",
            *active_rows,
        ],
        1,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == expected


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell test is Windows-only")
def test_launcher_rejects_unexpected_query_status_even_with_active_row():
    result = _run_launcher_session_selector(
        [
            " SESSIONNAME       USERNAME                 ID  STATE   TYPE        DEVICE ",
            ">console           John                      1  Active                      ",
        ],
        2,
    )

    assert result.returncode == 7
    assert "Unable to query Windows sessions" in result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell test is Windows-only")
@pytest.mark.parametrize(
    "session_lines, expected_error",
    [
        (["query failed"], "exactly one expected header"),
        (
            [
                " SESSIONNAME       USERNAME                 ID  STATE   TYPE        DEVICE ",
                " services                                    0  Disc                        ",
            ],
            "No active interactive Windows session",
        ),
    ],
)
def test_launcher_requires_header_and_active_session_proof(
    session_lines, expected_error
):
    result = _run_launcher_session_selector(session_lines, 1)

    assert result.returncode == 7
    assert expected_error in result.stderr


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
