---
name: aigp-sim-connect
description: >-
  Connect to / verify / launch the official AI Grand Prix simulator (DCL
  "DCGame" / FlightSim.exe) before any live run. Use whenever you need to fly
  the real sim, run scripts/aigp_vq1_run.py, debug why MAVLink won't connect,
  or confirm the sim is in Virtual Qualifier mode. Read this FIRST instead of
  re-deriving the connection procedure.
---

# Connecting to the official AI Grand Prix simulator

This repo flies a **real** simulator, not PyBullet. The sim is the DCL
"DCGame" Unreal build (`FlightSim.exe`, process `DCGame-Win64-Shipping`). It
speaks **MAVLink2 over loopback UDP 14550** + a **JPEG vision stream on UDP
5600**. Do NOT validate flight in PyBullet or the kinematic bench — those are
unit-test fixtures, not the competition target.

## TL;DR — one command to check readiness

```
python -m scripts.sim_connect_check
```
- exit **0 = READY** (heartbeat + track-data; in VQ mode; safe to fly)
- exit **2 = PARTIAL** (MAVLink up but no track → not in Virtual Qualifier mode)
- exit **3 = NO_LINK** (no heartbeat → sim not serving / not launched in VQ)

It is **non-flying** (no arm, no setpoints): it connects, waits for a heartbeat
+ the track-data transfer, then disconnects. Always run it before a live run.

## The two things that must be true to fly

1. **The sim process is running.** `DCGame-Win64-Shipping`. If not:
   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_sim.ps1
   ```
   That uses a `schtasks /IT` bridge to launch into the **interactive desktop
   (session 1)** — a GUI/GPU app launched directly over SSH (session 0) never
   renders. The script refuses to double-launch if the sim is already up.
2. **The sim is in VIRTUAL QUALIFIER mode** (not ACRO free-flight, not a menu).
   Only VQ mode serves the MAVLink/vision interface and sends track data.
   Entering VQ is a **GUI action** — it cannot be done over MAVLink or a shell;
   use **Parsec** (or the physical console) to navigate the sim UI. A SIM_RESET
   (MAVLink cmd 31000) only resets a race *within* VQ; it does not enter VQ.

Login: the game persists the PGOS account to
`%LOCALAPPDATA%\FlightSim\Saved\SaveGames\DCLSave-LocalPlayer.sav` with an
AutoLogin path, so a relaunch usually re-logins silently. If it lands on the
email/password screen, do a one-time Parsec/console login ("remember me").

## Host facts (this machine)

- This PC (`DESKTOP-M5VJ10H`, RTX 3060) is **both** the sim host and the Python
  pilot — everything is **localhost** UDP 14550/5600.
- Sim dev kit: `C:\Users\Kenichi\Downloads\AI-GP Simulator v1.0.3364\AIGP_3364\FlightSim.exe`.
- `pymavlink` is installed (the pilot is headless Python — it runs in any
  session as long as the sim, in session 1, is serving localhost UDP).

## Gotchas that waste time if you don't know them

- **An idle UDP 14550 does NOT mean the sim is down.** `udpin:127.0.0.1:14550`
  means *our pilot* binds 14550 and the sim *sends* to it — so when no pilot is
  running the port is unbound even though the sim is happily in VQ mode. Use the
  readiness check (which actually talks MAVLink), not a port scan.
- **Sim degrades after ~25 runs/session.** Symptoms: start over-climb grows
  (healthy first-3s peak climb Z≈−1.7 m; degraded ≤ −2.4 m) and/or the gate-map
  transfer returns garbage (sign-flipped X, Z≈−350). A degraded *process* needs
  a full `.exe` restart into VQ mode — a per-run SIM_RESET will NOT fix it.
  The runner has guards: `competition/sim_health.py` (over-climb/collision probe,
  `--abort-on-degraded`) and `competition/gate_map_integrity.py` (map-corruption
  monitor). If either fires, restart the sim (relaunch + re-select VQ).
- **Start discipline (DSQ).** Never command flight before the GO countdown. The
  runner's `_reset_and_settle` waits for the authoritative GO crossing
  (`sim_boot_time_ms >= race_start_boot_time_ms + 120 ms`); never gate on a timer
  or on `RaceStatus.race_started` (that flips at "GO scheduled", ~0.6 s).
- **No DSQ signal over MAVLink** — the verdict is sim-GUI-side only.
- **Frames:** FRD/NED. The sim mishandles SET_ATTITUDE_TARGET *attitude* mode
  (it spins); the adapter sends **body-rate** mode (mask 128). See
  `competition/aigp_mavlink.py`.

## Run a flight (once READY)

```
# reliable / max-margin (recommended shakedown):
python -m scripts.aigp_vq1_run --minimal --spline --record captures/run.jsonl.gz --max-seconds 75
# fastest lap (gate-by-gate champion): see docs/aigp/2026-06-16-speed-and-spline-handoff.md
# analyse a capture:
python -m scripts.iter36_compare captures/run.jsonl.gz
# INDI crux experiment + read-out:
python -m scripts.aigp_vq1_run --minimal --indi --record captures/indi.jsonl.gz --max-seconds 75
python -m scripts.analyze_indi_run captures/indi.jsonl.gz
```

Key code: `competition/aigp_mavlink.py` (transport), `scripts/aigp_vq1_run.py`
(runner + CLI), `competition/session.py`. Background: `docs/aigp/host_setup.md`,
`docs/aigp/2026-06-10-first-contact-findings.md`.
