# Launch FlightSim in the active Windows desktop when invoked locally or over
# SSH. Training-mode selection may still require an interactive desktop action.
param(
    [string]$SimulatorPath,
    [string]$RunAsUser,
    [string]$TaskName,
    [ValidateRange(5, 300)]
    [int]$StartupTimeoutSeconds = 60
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Serialize the process check and launch lifecycle across shells/sessions. A
# plain Get-Process check has a TOCTOU window in which two concurrent launchers
# can both observe no simulator and create different randomly named tasks.
$LaunchMutexName = 'Global\AIGP-FlightSim-Launch'
$LaunchMutex = [System.Threading.Mutex]::new($false, $LaunchMutexName)
$LaunchMutexOwned = $false
try {
    try {
        $LaunchMutexOwned = $LaunchMutex.WaitOne(
            [TimeSpan]::FromSeconds($StartupTimeoutSeconds)
        )
    } catch [System.Threading.AbandonedMutexException] {
        # The previous launcher died while holding the mutex. Windows has
        # transferred ownership to this process, so the guarded proof can be
        # repeated from the beginning.
        $LaunchMutexOwned = $true
    }
    if (-not $LaunchMutexOwned) {
        throw "Timed out waiting for the FlightSim launch guard '$LaunchMutexName'."
    }

    # Never double-launch: an existing process may own a live MAVLink/vision run.
    $existing = Get-Process -Name 'DCGame-Win64-Shipping', 'FlightSim' -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Output 'Sim already running - refusing to relaunch.'
        exit 0
    }

if (-not $SimulatorPath) {
    $SimulatorPath = if ($env:AIGP_FLIGHTSIM_PATH) {
        $env:AIGP_FLIGHTSIM_PATH
    } else {
        Join-Path $env:USERPROFILE 'AIGP\AIGP_3385\FlightSim.exe'
    }
}
if (-not (Test-Path -LiteralPath $SimulatorPath -PathType Leaf)) {
    throw "Launcher not found: $SimulatorPath"
}
$SimulatorPath = (Resolve-Path -LiteralPath $SimulatorPath).Path
$SimulatorDirectory = Split-Path -Parent $SimulatorPath

# `query session` reports the real interactive session IDs. Prefer the session
# attached to this shell, then console, then another active RDP desktop.
$querySession = Join-Path $env:SystemRoot 'System32\query.exe'
$sessionLines = & $querySession session 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to query Windows sessions; cannot safely target a GUI desktop.'
}
$activeSessions = @(
    foreach ($line in $sessionLines) {
        if ($line -match '^\s*(?<current>>)?\s*(?<session>\S+)\s+(?<user>\S+)\s+(?<id>\d+)\s+Active\b') {
            [pscustomobject]@{
                Current = [bool]$Matches.current
                Session = $Matches.session
                User = $Matches.user
                Id = [int]$Matches.id
            }
        }
    }
)
if ($activeSessions.Count -eq 0) {
    throw 'No active interactive Windows session; log in before launching FlightSim.'
}
$interactive = $activeSessions |
    Sort-Object -Property @(
        @{ Expression = 'Current'; Descending = $true },
        @{ Expression = { $_.Session -eq 'console' }; Descending = $true },
        @{ Expression = 'Id'; Descending = $false }
    ) |
    Select-Object -First 1
$InteractiveSessionId = $interactive.Id

if (-not $RunAsUser) {
    $RunAsUser = if ($interactive.User -match '[\\@]') {
        $interactive.User
    } else {
        "$env:COMPUTERNAME\$($interactive.User)"
    }
}
if (-not $TaskName) {
    $TaskName = "LaunchAIGP-$PID-$([Guid]::NewGuid().ToString('N'))"
}

# Never overwrite an unrelated scheduled task. `/Create` intentionally omits
# `/F` as a second race-safe guard if a task appears after this query.
& schtasks.exe /Query /TN $TaskName 2>$null | Out-Null
$taskQueryExit = $LASTEXITCODE
if ($taskQueryExit -eq 0) {
    throw "Scheduled task '$TaskName' already exists; refusing to overwrite it."
}
if ($taskQueryExit -ne 1) {
    throw "Could not prove scheduled task name '$TaskName' is unused."
}

# The UE thin launcher resolves payloads relative to its working directory.
$action = 'cmd.exe /c cd /d "{0}" && start "" "{1}"' -f $SimulatorDirectory, $SimulatorPath
$taskCreated = $false
$ok = $false
$primaryError = $null
$cleanupError = $null
try {
    & schtasks.exe /Create /TN $TaskName /TR $action /SC ONCE /ST 00:00 `
        /RU $RunAsUser /IT /RL HIGHEST | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not create launcher task '$TaskName'." }
    $taskCreated = $true

    & schtasks.exe /Run /TN $TaskName | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not run launcher task '$TaskName'." }

    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    do {
        Start-Sleep -Milliseconds 500
        $processes = Get-Process -Name 'DCGame-Win64-Shipping', 'FlightSim' -ErrorAction SilentlyContinue
        if ($processes -and ($processes.SessionId -contains $InteractiveSessionId)) {
            $ok = $true
            break
        }
    } while ([DateTime]::UtcNow -lt $deadline)
} catch {
    $primaryError = $_
} finally {
    if ($taskCreated) {
        & schtasks.exe /Delete /TN $TaskName /F | Out-Null
        if ($LASTEXITCODE -ne 0) {
            $cleanupError = "Could not delete temporary launcher task '$TaskName'."
        }
    }
}

if ($primaryError -and $cleanupError) {
    throw "$($primaryError.Exception.Message) Cleanup also failed: $cleanupError"
}
if ($primaryError) { throw $primaryError }
if ($cleanupError) { throw $cleanupError }

if (-not $ok) {
    throw "FlightSim was not observed in interactive session $InteractiveSessionId within $StartupTimeoutSeconds seconds."
}
Write-Output (
    "Sim launched in interactive session {0} as {1}: {2}" -f `
        $InteractiveSessionId, $RunAsUser, $SimulatorPath
)
} finally {
    if ($LaunchMutexOwned) {
        $LaunchMutex.ReleaseMutex()
    }
    $LaunchMutex.Dispose()
}
