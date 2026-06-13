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

    yaw_sat = (
        sum(1 for c in cmd_yaw if abs(c) > 3.0) / len(cmd_yaw)
        if cmd_yaw else 0.0
    )
    tilt_sat = 0.0
    if cmd_roll and cmd_pitch:
        tilt_sat = sum(
            1 for r, p in zip(cmd_roll, cmd_pitch)
            if abs(r) > 0.69 or abs(p) > 0.69  # ~40 deg
        ) / len(cmd_roll)

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
        "cmd_yaw_sat_frac": round(yaw_sat, 3),
        "tilt_sat_frac": round(tilt_sat, 3),
        "ref_pos_frac": round(has_ref / len(rows), 3),
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

    if yaw_sat > 0.5:
        A(f"CMD SATURATION (yaw): |cmd_yaw|>3 rad for {yaw_sat:.0%} of ticks")
    if tilt_sat > 0.3:
        W(f"CMD SATURATION (tilt): roll/pitch>40deg for {tilt_sat:.0%} of ticks")

    if has_ref == 0:
        A("NO REFERENCE: tracker never produced a reference point "
          "(ref_pos None for all ticks) -- trajectory not delivered / recorder bug")

    if gates == 0 and dur > 30:
        A(f"GATE STALL: 0 gates passed in {dur:.0f} s")

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
            f"yaw_sat={st['cmd_yaw_sat_frac']} ref_pos_frac={st['ref_pos_frac']}"
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
