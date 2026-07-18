# AIGP Sim Host Setup (Task 0.1)

> **Current VQ2 host (2026-07-18):** build 3385 is running locally at
> `C:\Users\John\AIGP\AIGP_3385\FlightSim.exe`, with the repository at
> `C:\Users\John\killallhumans`. See
> [`2026-07-18-vq2-handoff.md`](2026-07-18-vq2-handoff.md) for the active setup.
> The Ken/3364 setup below is retained as VQ1 history and should not be used as
> build-3385 interface documentation.

The plan's primary host was ShadowPC; we used the §3.1 sanctioned alternative — **Ken's own
RTX PC** — since it was available (best latency, free, unlimited hours).

## Topology
- **Mac** = brain: Claude Code + MCP stack + offline dev. Controls the PC over SSH.
- **Windows PC `DESKTOP-M5VJ10H`** = sim + Python pilot, co-located (localhost UDP 14550/5600).
  RTX 3060 12 GB (driver 560.94), Python 3.14.6, git 2.54.
- Link: **Tailscale personal tailnet** (`tailscale switch kenichimatsuo1775@gmail.com` on the
  Mac; `tailscale switch doordash.com` to return to corp). PC runs Tailscale `--unattended`.

## Control the PC
```
ssh -i ~/.ssh/id_ed25519_winpc -o IdentitiesOnly=yes Kenichi@100.122.0.79 "<powershell>"
```
- `IdentitiesOnly=yes` is REQUIRED (else Windows MaxAuthTries → "Too many authentication failures").
- Remote default shell = PowerShell. Repo cloned at `C:\Users\Kenichi\killallhumans`
  (workflow: Mac pushes → PC pulls).

## Installed for AIGP
- `pymavlink`, `opencv-python`, `numpy` (recorder/pilot), `matplotlib`, `keyboard` (stock template).
- **Parsec** (GUI access from anywhere — for the one-time sim login + watching runs).

## Launching the sim (session 0 → session 1)
An SSH command runs in **session 0**; the GPU desktop is **session 1**. A GUI/GPU app launched
directly over SSH won't render. Use `scripts/launch_sim.ps1` (a `schtasks /IT` bridge into the
logged-on session). The sim's online (PGOS) login persists to
`%LOCALAPPDATA%\FlightSim\Saved\SaveGames\DCLSave-LocalPlayer.sav` with an AutoLogin path; a
one-time manual login (Parsec) seeds it. See `2026-06-10-first-contact-findings.md`.

## Dev kit location (on the PC)
`C:\Users\Kenichi\Downloads\AI-GP Simulator v1.0.3364\` → `AIGP_3364\FlightSim.exe` (sim) +
`PyAIPilotExample\` (official stock pilot). Sim must be in the **Virtual Qualifier** (not ACRO
free-flight) to serve the MAVLink/vision interface.
