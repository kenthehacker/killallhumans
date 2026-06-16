#!/usr/bin/env python3
"""Readiness check for the official AI Grand Prix simulator (DCL "DCGame").

Run this FIRST, before any live run. It does NOT fly the drone — it checks the
sim process, then opens a MAVLink connection and waits for a heartbeat + the
track-data transfer (track data only arrives once the sim is in VIRTUAL
QUALIFIER mode). Prints a clear verdict and the exact next action.

Usage:
    python -m scripts.sim_connect_check

Exit codes:
    0  READY      heartbeat + track-data (sim is in VQ mode, safe to fly)
    2  PARTIAL    MAVLink heartbeat but NO track-data (likely not in VQ mode)
    3  NO_LINK    no heartbeat (sim not serving / not launched in VQ)
    1  ERROR      unexpected error

See the `aigp-sim-connect` skill for the full procedure (launch, login, gotchas).
"""
from __future__ import annotations

import asyncio
import subprocess
import sys

SIM_PROCESS = "DCGame-Win64-Shipping"  # the FlightSim.exe shipping process
LAUNCH_HINT = (
    "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_sim.ps1"
)


def _sim_process_up() -> bool | None:
    """True/False if the sim process is running, None if the check failed."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {SIM_PROCESS}.exe"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return SIM_PROCESS in out
    except Exception:
        return None


async def _probe(timeout_s: float = 35.0):
    """Non-flying MAVLink probe: connect, wait for heartbeat + track, disconnect."""
    from competition.aigp_mavlink import AIGPMavlinkAdapter

    a = AIGPMavlinkAdapter(enable_vision=False, require_track=False, track_retries=2)
    try:
        await asyncio.wait_for(a.connect(), timeout=timeout_s)
    except asyncio.TimeoutError:
        return ("NO_LINK", None, False)
    except Exception as exc:  # noqa: BLE001 - report any connect failure verbatim
        return ("ERROR", repr(exc), False)
    telem = a.latest_telemetry
    has_track = a.track_data is not None
    try:
        await asyncio.wait_for(a.disconnect(), timeout=6.0)
    except Exception:
        pass
    return ("OK", telem, has_track)


def main() -> int:
    up = _sim_process_up()
    state = "running" if up else ("NOT running" if up is False else "unknown")
    print(f"[1/2] sim process {SIM_PROCESS}: {state}")
    if up is False:
        print(f"      -> launch it:  {LAUNCH_HINT}")
        print("         then, if it lands on the login screen, do the one-time")
        print("         Parsec/console login, and select VIRTUAL QUALIFIER mode.")

    status, info, has_track = asyncio.run(_probe())

    if status == "ERROR":
        print(f"[2/2] MAVLink: ERROR: {info}")
        return 1
    if status == "NO_LINK":
        print("[2/2] MAVLink: no heartbeat on udp 14550.")
        print("      VERDICT: NOT READY — the sim is not serving MAVLink.")
        print("      Make sure it is in VIRTUAL QUALIFIER mode (not ACRO / not a menu);")
        print("      entering VQ is a GUI action (Parsec/console). See aigp-sim-connect skill.")
        return 3

    pos = "" if info is None else f"  pos={tuple(round(x, 2) for x in info.position_ned)}"
    print(f"[2/2] MAVLink: heartbeat OK, telemetry flowing.{pos}")
    if not has_track:
        print("      VERDICT: PARTIAL - MAVLink up but NO track data.")
        print("      The sim is likely NOT in Virtual Qualifier mode. Switch it to VQ, then re-check.")
        return 2
    print("      VERDICT: READY - sim is in VQ mode and serving track + telemetry.")
    print("      Safe to fly, e.g.:  python -m scripts.aigp_vq1_run --minimal --spline "
          "--record captures/run.jsonl.gz --max-seconds 75")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
