"""Flight-behaviour anomaly detector for AIGP telemetry captures.

Reads a ``captures/telemetry_*.jsonl(.gz)`` log written by
``scripts/aigp_vq1_run.py`` and flags behaviours an aggregate metric
(gates-passed, tracking error) silently hides -- most importantly the
"drone spins in circles" / "drone never moves" failure modes that a human
sees instantly on the spectator view but a benchmark number does not.

Usage::

    python scripts/analyze_telemetry.py captures/telemetry_1781170497.jsonl.gz
    python scripts/analyze_telemetry.py captures/            # newest in dir
    python scripts/analyze_telemetry.py --all captures/      # every file

Exit code is non-zero if any ANOMALY (not just WARN) is detected, so the
iteration loop can gate on it.

Detected anomalies
------------------
* FROZEN STATE      -- position never changes (dead telemetry feed / dry-run)
* SPINNING          -- large cumulative yaw travel with little net translation
* CIRCLING          -- path length >> net displacement (orbiting, not racing)
* CMD SATURATION    -- attitude/yaw command pinned at the limit most of the run
* CONSTANT COMMAND  -- identical command every tick (open-loop / stalled control)
* NO REFERENCE      -- tracker never produced a reference point (ref_pos all None)
* GATE STALL        -- zero gate progress despite a full-length run
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
from typing import List, Optional


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def load(path: str) -> List[dict]:
    rows = []
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _unwrap(yaws: List[float]) -> List[float]:
    if not yaws:
        return []
    uw = [yaws[0]]
    for y in yaws[1:]:
        d = y - uw[-1]
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        uw.append(uw[-1] + d)
    return uw


def analyze(path: str) -> dict:
    rows = load(path)
    result = {"path": path, "anomalies": [], "warnings": [], "stats": {}}
    if len(rows) < 2:
        result["anomalies"].append(f"EMPTY/short capture ({len(rows)} rows)")
        return result

    def t_s(r):
        return (r.get("t_us") or 0) / 1e6

    dur = t_s(rows[-1]) - t_s(rows[0])
    xs = [r["pos"][0] for r in rows]
    ys = [r["pos"][1] for r in rows]
    zs = [r["pos"][2] for r in rows]
    yaws = [r.get("yaw", 0.0) for r in rows]
    uw = _unwrap(yaws)

    # Path length / net displacement
    pathlen = 0.0
    for i in range(1, len(rows)):
        pathlen += math.dist(
            (xs[i], ys[i], zs[i]), (xs[i - 1], ys[i - 1], zs[i - 1])
        )
    net = math.dist((xs[-1], ys[-1], zs[-1]), (xs[0], ys[0], zs[0]))

    # Yaw travel / max rate
    yaw_travel = uw[-1] - uw[0]
    revolutions = yaw_travel / (2 * math.pi)
    max_yaw_rate = 0.0
    for i in range(1, len(rows)):
        dt = t_s(rows[i]) - t_s(rows[i - 1])
        if dt > 1e-6:
            max_yaw_rate = max(max_yaw_rate, abs((uw[i] - uw[i - 1]) / dt))

    gates = rows[-1].get("gates_passed", 0) or 0
    pos_span = max(
        max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)
    )

    # Command analysis
    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]

    cmd_yaw = col("cmd_yaw")
    cmd_roll = col("cmd_roll")
    cmd_pitch = col("cmd_pitch")
    has_ref = sum(1 for r in rows if r.get("ref_pos") is not None)

    def constant(vals):
        return len(vals) > 1 and (max(vals) - min(vals)) < 1e-6

    # NOTE: cmd_yaw is a yaw *angle* setpoint, not a rate. A course that runs
    # toward -X legitimately holds yaw ~ +/-pi (|cmd_yaw| ~ 3.14), so a large
    # |cmd_yaw| is NOT saturation. The genuine "stuck yaw command" failure is
    # already covered by CONSTANT COMMAND (identical cmd every tick) and
    # FROZEN STATE; reported here only as an informational stat.
    yaw_extreme_frac = (
        sum(1 for c in cmd_yaw if abs(c) > 3.0) / len(cmd_yaw)
        if cmd_yaw else 0.0
    )
    tilt_sat = 0.0
    if cmd_roll and cmd_pitch:
        tilt_sat = sum(
            1 for r, p in zip(cmd_roll, cmd_pitch)
            if abs(r) > 0.69 or abs(p) > 0.69  # ~40 deg
        ) / len(cmd_roll)

    # ---- MEASURED attitude / body-rate analysis (erratic-flight detection) --
    # Needs the roll/pitch/gyro fields added 2026-06-13. Without them the
    # bounce/tumble that a human sees is invisible to this analyzer (that gap
    # is exactly why "bouncing left/right" had to be reported by eye).
    def _reversals(vals):
        """Direction reversals (local extrema) — the oscillation/jitter count."""
        if len(vals) < 3:
            return 0
        n = 0
        for i in range(1, len(vals) - 1):
            d0 = vals[i] - vals[i - 1]
            d1 = vals[i + 1] - vals[i]
            if d0 * d1 < 0:
                n += 1
        return n

    m_roll = col("roll")
    m_pitch = col("pitch")
    gyros = [r["gyro"] for r in rows if r.get("gyro") is not None]
    have_attitude = bool(m_roll) and bool(m_pitch)
    have_gyro = bool(gyros)

    roll_rev_hz = pitch_rev_hz = 0.0
    flip_frac = 0.0
    gyro_p95 = 0.0
    if have_attitude and dur > 1:
        roll_rev_hz = _reversals(m_roll) / (2 * dur)
        pitch_rev_hz = _reversals(m_pitch) / (2 * dur)
        # "Flip": tilted past ~75 deg (1.3 rad) — drone on its side / inverted.
        flips = sum(1 for r, p in zip(m_roll, m_pitch)
                    if abs(r) > 1.3 or abs(p) > 1.3)
        flip_frac = flips / max(len(m_roll), 1)
    if have_gyro:
        mags = sorted(
            math.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) for g in gyros
        )
        gyro_p95 = mags[int(0.95 * (len(mags) - 1))]

    # Closest approach to each gate the drone TARGETED (sequencer's current
    # gate). "How close did we get to the gate we were trying to fly through"
    # — far more informative than distance to gate 0. Needs the
    # target_gate/dist_target_gate fields added 2026-06-13.
    gate_closest = {}  # gate index -> min dist while it was the target
    for r in rows:
        gi = r.get("target_gate")
        d = r.get("dist_target_gate")
        if gi is None or d is None:
            continue
        if gi not in gate_closest or d < gate_closest[gi]:
            gate_closest[gi] = d
    # final target + how far the run ended from it
    last_target = rows[-1].get("target_gate")
    last_dist = rows[-1].get("dist_target_gate")

    result["stats"] = {
        "duration_s": round(dur, 1),
        "records": len(rows),
        "gates_passed": gates,
        "pos_span_m": round(pos_span, 2),
        "path_len_m": round(pathlen, 1),
        "net_disp_m": round(net, 2),
        "circling_ratio": round(pathlen / max(net, 0.01), 1),
        "yaw_revolutions": round(revolutions, 2),
        "max_yaw_rate_radps": round(max_yaw_rate, 1),
        "cmd_yaw_extreme_frac": round(yaw_extreme_frac, 3),
        "tilt_sat_frac": round(tilt_sat, 3),
        "ref_pos_frac": round(has_ref / len(rows), 3),
        "roll_osc_hz": round(roll_rev_hz, 1),
        "pitch_osc_hz": round(pitch_rev_hz, 1),
        "flip_frac": round(flip_frac, 3),
        "gyro_p95_radps": round(gyro_p95, 2),
        "z_min": round(min(zs), 1),
        "z_max": round(max(zs), 1),
        "gate_closest_m": {int(k): round(v, 1) for k, v in sorted(gate_closest.items())},
        "final_target_gate": last_target,
        "final_dist_to_target_m": round(last_dist, 1) if last_dist is not None else None,
    }

    # ---- Anomaly rules -------------------------------------------------
    A = result["anomalies"].append
    W = result["warnings"].append

    if pos_span < 0.05:
        A(f"FROZEN STATE: position never moves (span {pos_span:.3f} m over "
          f"{dur:.0f} s) -- dead telemetry feed or dry-run capture")

    # Spinning: many revolutions but little translation.
    if abs(revolutions) >= 1.5 and net < 5.0:
        A(f"SPINNING: {revolutions:.1f} yaw revolutions with only {net:.1f} m "
          f"net translation")
    elif max_yaw_rate > 6.0:
        W(f"high yaw rate spike {max_yaw_rate:.1f} rad/s (>6) -- possible "
          f"yaw instability")

    # Circling: covers distance but goes nowhere.
    if pos_span > 0.05 and net > 0.5 and (pathlen / max(net, 0.01)) > 8.0:
        A(f"CIRCLING: path {pathlen:.0f} m vs net {net:.1f} m "
          f"(ratio {pathlen/max(net,0.01):.0f}) -- orbiting, not progressing")

    if cmd_yaw and constant(cmd_yaw) and constant(cmd_roll or [0]) \
            and constant(cmd_pitch or [0]):
        A("CONSTANT COMMAND: identical attitude command every tick -- "
          "control loop is open / stalled (not responding to state)")

    if tilt_sat > 0.3:
        W(f"CMD SATURATION (tilt): roll/pitch>40deg for {tilt_sat:.0%} of ticks")

    if has_ref == 0:
        A("NO REFERENCE: tracker never produced a reference point "
          "(ref_pos None for all ticks) -- trajectory not delivered / recorder bug")

    if gates == 0 and dur > 30:
        A(f"GATE STALL: 0 gates passed in {dur:.0f} s")

    # ---- Erratic-flight rules (measured attitude / body rates) ----------
    if have_attitude:
        # Sustained roll/pitch oscillation = the "bouncing / jittery" failure.
        if roll_rev_hz > 2.0 or pitch_rev_hz > 2.0:
            A(f"ATTITUDE OSCILLATION: roll {roll_rev_hz:.1f} Hz / pitch "
              f"{pitch_rev_hz:.1f} Hz direction reversals -- jittery/bouncing "
              f"attitude (control oscillation, e.g. undamped or wrong-sign loop)")
        if flip_frac > 0.05:
            A(f"ATTITUDE FLIP/TUMBLE: tilted past 75 deg for {flip_frac:.0%} of "
              f"ticks -- drone going on its side / inverting")
    else:
        W("no measured roll/pitch in capture -- attitude oscillation/flip "
          "detection disabled (re-record with the upgraded recorder)")
    if have_gyro and gyro_p95 > 4.0:
        A(f"HIGH BODY RATES: gyro p95 = {gyro_p95:.1f} rad/s -- tumbling / "
          f"violent attitude motion (a stable flight stays well under ~3)")
    # Vertical divergence: the VQ1 course spans z in [-2, 27] m. Leaving that
    # envelope by a wide margin is a climb/dive runaway (altitude control bug).
    if max(zs) > 80 or min(zs) < -80:
        A(f"VERTICAL DIVERGENCE: z range [{min(zs):.0f}, {max(zs):.0f}] m far "
          f"outside the ~[-2,27] m course envelope -- altitude runaway")

    return result


def print_report(res: dict) -> None:
    print(f"\n=== {os.path.basename(res['path'])} ===")
    st = res["stats"]
    if st:
        print(
            f"  dur={st['duration_s']}s records={st['records']} "
            f"gates={st['gates_passed']}"
        )
        print(
            f"  motion: span={st['pos_span_m']}m path={st['path_len_m']}m "
            f"net={st['net_disp_m']}m circle_ratio={st['circling_ratio']}"
        )
        print(
            f"  yaw: revs={st['yaw_revolutions']} "
            f"max_rate={st['max_yaw_rate_radps']}rad/s "
            f"yaw_extreme={st['cmd_yaw_extreme_frac']} ref_pos_frac={st['ref_pos_frac']}"
        )
        if "roll_osc_hz" in st:
            print(
                f"  attitude: roll_osc={st['roll_osc_hz']}Hz pitch_osc={st['pitch_osc_hz']}Hz "
                f"flip_frac={st['flip_frac']} gyro_p95={st['gyro_p95_radps']}rad/s "
                f"z=[{st['z_min']},{st['z_max']}]m"
            )
        if st.get("gate_closest_m"):
            print(
                f"  next-gate: closest approach per targeted gate = {st['gate_closest_m']} m"
            )
            print(
                f"  ended targeting gate {st['final_target_gate']} at "
                f"{st['final_dist_to_target_m']} m from it"
            )
    for a in res["anomalies"]:
        print(f"  [ANOMALY] {a}")
    for w in res["warnings"]:
        print(f"  [warn]    {w}")
    if not res["anomalies"] and not res["warnings"]:
        print("  OK -- no anomalies detected")


def resolve_paths(target: str, all_files: bool) -> List[str]:
    if os.path.isdir(target):
        files = sorted(
            glob.glob(os.path.join(target, "telemetry_*.jsonl*")),
            key=os.path.getmtime,
        )
        if not files:
            return []
        return files if all_files else [files[-1]]
    return [target]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", help="capture file or directory")
    ap.add_argument("--all", action="store_true",
                    help="analyze every capture in a directory (default: newest)")
    args = ap.parse_args(argv)

    paths = resolve_paths(args.target, args.all)
    if not paths:
        print(f"No telemetry captures found at {args.target}", file=sys.stderr)
        return 2

    any_anomaly = False
    for p in paths:
        res = analyze(p)
        print_report(res)
        any_anomaly = any_anomaly or bool(res["anomalies"])

    print()
    return 1 if any_anomaly else 0


if __name__ == "__main__":
    raise SystemExit(main())
