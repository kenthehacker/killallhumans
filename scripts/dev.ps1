# Canonical Windows development command surface for the build-3385 VQ2 stack.
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$Task,
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$TaskArgs = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$TaskArgs = @($TaskArgs | Where-Object { $_ -ne '' })
$Python = if ($env:AIGP_PYTHON) {
    $env:AIGP_PYTHON
} else {
    Join-Path $RepoRoot '.venv\Scripts\python.exe'
}

function Assert-Python {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python not found at '$Python'. Create .venv or set AIGP_PYTHON."
    }
}

function Invoke-Python {
    param([string[]]$PythonArgs)
    Assert-Python
    # Keep canonical development commands from creating executable bytecode
    # beside trusted source. Each invocation gets an external, process-scoped
    # cache prefix; the OS temp lifecycle owns the resulting disposable files.
    $TempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    $ResolvedRepoRoot = [IO.Path]::GetFullPath($RepoRoot).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    if (
        $TempRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) -eq $ResolvedRepoRoot -or
        $TempRoot.StartsWith(
            $ResolvedRepoRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw 'The Python bytecode cache prefix must be outside the repository.'
    }
    $PycachePrefix = Join-Path $TempRoot (
        'aigp-dev-pycache-' + [Guid]::NewGuid().ToString('N')
    )
    $PreviousPycachePrefix = $env:PYTHONPYCACHEPREFIX
    try {
        $env:PYTHONPYCACHEPREFIX = $PycachePrefix
        & $Python @PythonArgs
        $PythonExitCode = $LASTEXITCODE
    } finally {
        if ($null -eq $PreviousPycachePrefix) {
            Remove-Item Env:PYTHONPYCACHEPREFIX -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPYCACHEPREFIX = $PreviousPycachePrefix
        }
    }
    if ($PythonExitCode -ne 0) {
        exit $PythonExitCode
    }
}

function Invoke-PromotionTests {
    param([string[]]$PromotionArgs)
    if (
        $PromotionArgs.Count -gt 1 -or
        ($PromotionArgs.Count -eq 1 -and $PromotionArgs[0] -ne '--fresh')
    ) {
        throw 'test-promotion accepts only the optional --fresh recovery flag.'
    }
    Invoke-Python (
        @('-u', '-m', 'scripts.aigp_promotion_runner') + $PromotionArgs
    )
}

Push-Location $RepoRoot
try {
    switch ($Task.ToLowerInvariant()) {
        'help' {
            if ($TaskArgs.Count -ne 0) { throw 'help takes no arguments.' }
            Write-Output @"
Available tasks: test-target, test-fast, test-unit, test-vq2, test-slow,
test-benchmark, test-promotion, test-full-non-live, preflight, flight-cycle,
launch-sim, sbom.
Use scripts\dev.cmd on Windows so the repository command works even when the
machine's script execution policy blocks direct .ps1 invocation.

test-promotion is the durable 10-13 minute promotion boundary. It includes
slow and benchmark tests, reports live per-test progress, and reuses or
attaches to the exact same commit-keyed run instead of launching a duplicate.
The supervisor enforces a hard 15-minute aggregate ceiling.
test-full-non-live remains a compatibility alias.
Both names require an exact pristine commit checkout (including no ignored
files) and may reuse its durable result.

flight-cycle is a dedicated powered command, not a test task. It requires an
existing scoped user authorization and never prompts or reconfirms it. It
performs no separate passive preflight, screenshot, console challenge, or
interactive confirmation.
"@
        }
        'test-target' {
            if ($TaskArgs.Count -eq 0) {
                throw 'test-target requires one or more pytest paths/node IDs.'
            }
            # Override the repository's default marker expression so an
            # explicitly targeted slow/benchmark test actually executes.
            # Keep live excluded even if a caller supplies another -m option.
            Invoke-Python (@('-m', 'pytest', '-q') + $TaskArgs + @('-m', 'not live'))
        }
        'test-fast' {
            if ($TaskArgs.Count -ne 0) { throw 'test-fast takes no arguments.' }
            Invoke-Python @('-m', 'pytest', '-q')
        }
        'test-unit' {
            if ($TaskArgs.Count -ne 0) { throw 'test-unit takes no arguments.' }
            Invoke-Python @('-m', 'pytest', '-q', '-m', 'unit and not live')
        }
        'test-vq2' {
            if ($TaskArgs.Count -ne 0) { throw 'test-vq2 takes no arguments.' }
            Invoke-Python @(
                '-m', 'pytest', '-q',
                'competition/tests',
                'estimation/tests',
                'gate_detection/tests',
                'planning/tests/test_vq2_guidance.py',
                'tests/test_aigp_live_lease.py',
                'tests/test_aigp_vq2_build_reference.py',
                'tests/test_aigp_vq2_calibration_target.py',
                'tests/test_aigp_vq2_fast_cycle.py',
                'tests/test_aigp_vq2_passive_probe.py',
                'tests/test_aigp_vq2_passive_timing_script.py',
                'tests/test_aigp_vq2_powered_attempt.py',
                'tests/test_aigp_vq2_powered_runtime.py',
                'tests/test_aigp_vq2_powered_calibration_probe.py',
                'tests/test_aigp_vq2_powered_calibration_analysis.py',
                'tests/test_aigp_vq2_powered_cleanup.py',
                'tests/test_aigp_vq2_runner.py',
                'tests/test_vision_udp.py',
                'tests/test_vision_udp_listener.py'
            )
        }
        'test-slow' {
            if ($TaskArgs.Count -ne 0) { throw 'test-slow takes no arguments.' }
            Invoke-Python @('-m', 'pytest', '-q', '-m', 'slow and not live', '--timeout=60')
        }
        'test-benchmark' {
            if ($TaskArgs.Count -ne 0) { throw 'test-benchmark takes no arguments.' }
            Invoke-Python @('-m', 'pytest', '-q', '-m', 'benchmark and not live', '--timeout=300')
        }
        'test-promotion' {
            Invoke-PromotionTests $TaskArgs
        }
        'test-full-non-live' {
            Write-Warning (
                'test-full-non-live is a compatibility alias; ' +
                'use test-promotion for the explicit promotion boundary.'
            )
            Invoke-PromotionTests $TaskArgs
        }
        'preflight' {
            if ($TaskArgs.Count -ne 0) { throw 'preflight takes no arguments.' }
            Write-Output 'Running passive VQ2 preflight (no arm or flight targets).'
            Invoke-Python @(
                '-m', 'scripts.aigp_vq2_run',
                '--stage', 'preflight',
                '--preflight-timeout-s', '3'
            )
        }
        'flight-cycle' {
            if ($TaskArgs.Count -gt 1) {
                throw 'flight-cycle accepts at most one powered stage.'
            }
            $stage = if ($TaskArgs.Count -eq 1) {
                $TaskArgs[0]
            } else {
                'calibration-excite'
            }
            $allowed = @(
                'sign-id', 'hover', 'gate0', 'gate0-observe',
                'calibration-excite'
            )
            if ($stage -notin $allowed) {
                throw "Unsupported flight-cycle stage '$stage'."
            }
            Write-Output (
                "Running one POWERED '$stage' cycle with compact evidence " +
                'and no interactive preflight ceremony.'
            )
            Invoke-Python @(
                '-E', '-s', '-B',
                '-m', 'scripts.aigp_vq2_fast_cycle', $stage
            )
        }
        'launch-sim' {
            if ($TaskArgs.Count -gt 1) { throw 'launch-sim accepts at most one FlightSim.exe path.' }
            $launcher = Join-Path $PSScriptRoot 'launch_sim.ps1'
            if ($TaskArgs.Count -eq 1) {
                & $launcher -SimulatorPath $TaskArgs[0]
            } else {
                & $launcher
            }
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        'sbom' {
            if ($TaskArgs.Count -gt 1) { throw 'sbom accepts at most one output path.' }
            $output = if ($TaskArgs.Count -eq 1) {
                $TaskArgs[0]
            } else {
                '.artifacts/dependency-inventory.cdx.json'
            }
            Invoke-Python @('-m', 'scripts.dependency_inventory', '--output', $output)
        }
        default {
            throw @"
Unknown task '$Task'. Available tasks:
  test-target <paths>  Directly affected tests (slow/benchmark allowed; live excluded)
  test-fast            Default non-slow, non-benchmark, non-live suite
  test-unit            Explicit isolated unit tier (always excludes live)
  test-vq2             Canonical build-3385 VQ2 candidate suite
  test-slow            Explicit bounded slow-test tier (always excludes live)
  test-benchmark       Explicit deterministic benchmark tier (always excludes live)
  test-promotion       Durable promotion-only full non-live suite (typically 10-13m)
  test-full-non-live   Compatibility alias for test-promotion
  preflight            Fast passive stream/target health check; no capture
  flight-cycle [stage] Dedicated powered cycle (default: calibration-excite)
  launch-sim [path]    Launch FlightSim in the active interactive session
  sbom [path]          Generate a local CycloneDX dependency inventory
"@
        }
    }
} finally {
    Pop-Location
}
