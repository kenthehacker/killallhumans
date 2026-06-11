# setup_win_remote.ps1 — One-shot remote-control setup for the AIGP sim host.
# Run in an ELEVATED (Administrator) PowerShell on the Windows PC:
#   irm https://raw.githubusercontent.com/kenthehacker/killallhumans/main/scripts/setup_win_remote.ps1 | iex
#
# What it does:
#   1. OpenSSH Server (auto-start + firewall)
#   2. PowerShell as the default SSH shell
#   3. Authorizes the Mac's public key (pasted when prompted) for both the
#      admin path (ProgramData) and the per-user path
#   4. Tailscale via winget, brought up in --unattended mode (stays connected
#      with nobody logged in) — log in with your PERSONAL account, not corp
#   5. Disables sleep/hibernate so the box stays reachable
#   6. git + Python 3.14 (best-effort; needed for the AIGP kit anyway)
#   7. Prints the facts the Mac side needs: tailnet IPv4, hostname, username

$ErrorActionPreference = "Stop"

Write-Host "`n=== [1/7] OpenSSH Server ===" -ForegroundColor Cyan
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd -ErrorAction SilentlyContinue
if (-not (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
}

Write-Host "=== [2/7] PowerShell as default SSH shell ===" -ForegroundColor Cyan
New-Item -Path "HKLM:\SOFTWARE\OpenSSH" -Force | Out-Null
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
    -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force | Out-Null

Write-Host "=== [3/7] Authorize the Mac's SSH key ===" -ForegroundColor Cyan
$pub = Read-Host "Paste the Mac's PUBLIC key (contents of id_ed25519_winpc.pub)"
if ($pub -notmatch '^(ssh-ed25519|ssh-rsa|ecdsa-) ') { throw "That does not look like an SSH public key." }
# Admin accounts read this file, NOT ~/.ssh/authorized_keys — classic gotcha.
$adminKeys = "C:\ProgramData\ssh\administrators_authorized_keys"
Add-Content -Path $adminKeys -Value $pub
icacls $adminKeys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F" | Out-Null
# Per-user file too, harmless redundancy (covers non-admin accounts).
$userSsh = Join-Path $env:USERPROFILE ".ssh"
New-Item -ItemType Directory -Path $userSsh -Force | Out-Null
Add-Content -Path (Join-Path $userSsh "authorized_keys") -Value $pub

Write-Host "=== [4/7] Tailscale (log in with your PERSONAL account) ===" -ForegroundColor Cyan
winget install --accept-package-agreements --accept-source-agreements --silent Tailscale.Tailscale
$ts = "$env:ProgramFiles\Tailscale\tailscale.exe"
# --unattended keeps the tunnel up with no user logged in (headless home PC).
& $ts up --unattended

Write-Host "=== [5/7] Never sleep ===" -ForegroundColor Cyan
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 10

Write-Host "=== [6/7] git + Python 3.14 (best effort) ===" -ForegroundColor Cyan
try { winget install --accept-package-agreements --accept-source-agreements --silent Git.Git } catch { Write-Warning "git install: $_" }
try { winget install --accept-package-agreements --accept-source-agreements --silent Python.Python.3.14 } catch { Write-Warning "python install: $_" }
# Optional, for GUI access from anywhere (FlightSim login screen etc.):
#   winget install Parsec.Parsec

Write-Host "`n=== [7/7] REPORT — paste these three lines back to Claude ===" -ForegroundColor Green
Write-Host ("TAILSCALE_IP : " + (& $ts ip -4))
Write-Host ("HOSTNAME     : " + $env:COMPUTERNAME)
Write-Host ("USERNAME     : " + $env:USERNAME)
