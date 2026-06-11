# launch_sim.ps1 — launch the AI-GP sim (FlightSim.exe) on DESKTOP-M5VJ10H so it
# renders on the GPU desktop, even when triggered over SSH.
#
# WHY THIS EXISTS: an SSH command runs in Windows session 0 (services). The GPU
# desktop is session 1 (`query session` -> "console Kenichi 1 Active"). A plain
# `ssh ... FlightSim.exe` launches in session 0 and NEVER renders / gets the GPU.
# `schtasks /IT` routes the launch into the logged-on user's interactive session 1,
# and with /RU = the logged-on user it needs NO password (/RP is only for
# non-interactive run-as). We create a one-shot task, run it, verify, delete it.
#
# Run from the Mac:
#   ssh -i ~/.ssh/id_ed25519_winpc -o IdentitiesOnly=yes Kenichi@100.122.0.79 \
#     'powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Kenichi\killallhumans\scripts\launch_sim.ps1"'
#
# LOGIN: the game persists the logged-in account to
#   %LOCALAPPDATA%\FlightSim\Saved\SaveGames\DCLSave-LocalPlayer.sav
# and has an AutoLogin path, so a relaunch SHOULD silently re-login IF the PGOS
# session token is still valid. If it lands on the email/password screen, do a
# ONE-TIME login (Parsec/console) — ideally with "remember me" — or store the
# credential once via `cmdkey /generic:AIGP_PGOS /user:<email> /pass` (DPAPI,
# per-user) + a small autotype shim. Never put the password in this repo.
#
# Source: Phase 0 launch-automation investigation (2026-06-10). See
# docs/aigp/2026-06-10-first-contact-findings.md.

$ErrorActionPreference = 'Stop'
$SimDir   = 'C:\Users\Kenichi\Downloads\AI-GP Simulator v1.0.3364\AIGP_3364'
$SimExe   = Join-Path $SimDir 'FlightSim.exe'
$TaskName = 'LaunchAIGP'

# 0) Guard: never double-launch (a live MAVLink/vision capture may be in progress).
if (Get-Process -Name 'DCGame-Win64-Shipping','FlightSim' -ErrorAction SilentlyContinue) {
    Write-Output 'Sim already running - refusing to relaunch.'; exit 0
}

# 1) Require an interactive desktop session to render into (the /IT target).
$console = (query session 2>$null | Select-String -Pattern '^\s*console\s+(\S+)\s+(\d+)\s+Active')
if (-not $console) { throw 'No active console session - nobody is logged on; cannot launch a GUI app.' }
$consoleUser = $console.Matches[0].Groups[1].Value      # e.g. Kenichi
$runAs       = "$env:COMPUTERNAME\$consoleUser"

if (-not (Test-Path $SimExe)) { throw "Launcher not found: $SimExe" }

# 2) Action: cd into the sim dir first (UE thin launcher resolves relative paths from CWD).
$action = 'cmd.exe /c cd /d "{0}" && start "" "{1}"' -f $SimDir, $SimExe

# 3) (Re)create a one-shot, interactive, elevated task that runs as the logged-on user.
schtasks /Create /TN $TaskName /TR $action /SC ONCE /ST 00:00 `
         /RU $runAs /IT /RL HIGHEST /F | Out-Null

# 4) Fire it now (the only trigger this task will ever have).
schtasks /Run /TN $TaskName | Out-Null

# 5) Confirm the game came up in the interactive session, then clean up the task.
$ok = $false
foreach ($i in 1..30) {
    Start-Sleep -Seconds 2
    $p = Get-Process -Name 'DCGame-Win64-Shipping' -ErrorAction SilentlyContinue
    if ($p -and ($p.SessionId -contains 1)) { $ok = $true; break }
}
schtasks /Delete /TN $TaskName /F | Out-Null    # leave no persistent artifact
if ($ok) { Write-Output 'Sim launched in interactive session.' }
else     { Write-Output 'WARNING: sim process not observed in session 1 within 60s.' }
