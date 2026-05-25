#!/usr/bin/env python3
"""
Multi-track regression matrix (iter-003 A17).

Runs the synthetic benchmark against every track config in
`sim_pybullet/configs/` and produces a JSON summary. The goal is to
catch the failure mode the user explicitly called out: "if we optimize
for just a single drone racing course then it might be overfit."

A change that improves race_01 but regresses figure8 (or any other
track) by >25% on `gate_pass_rate`, or introduces a new `crashed=True`
or `disqualified=True` on a track that previously passed, is a
regression and should not ship.

Usage:
    python -m scripts.benchmark_matrix              # JSON to stdout
    python -m scripts.benchmark_matrix --human      # ASCII table to stderr
    python -m scripts.benchmark_matrix --configs race_01,aigp_default
    python -m scripts.benchmark_matrix --duration 15  # per-track sim time

Exit code:
    0 — all tracks PASS (no crashes, no DQ, gate_pass_rate >= 50%)
    1 — at least one track regressed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _list_configs() -> List[Path]:
    """All JSON track configs in sim_pybullet/configs/, sorted by name."""
    cfg_dir = _REPO / "sim_pybullet" / "configs"
    return sorted(cfg_dir.glob("*.json"))


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def run_matrix(
    configs: List[Path], duration: float = 30.0, dt: float = 0.01,
) -> Dict[str, Any]:
    """Run the synthetic benchmark across every config; return JSON-shaped dict."""
    from scripts.benchmark import run_synthetic_benchmark

    matrix: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": duration,
        "dt": dt,
        "tracks": {},
        "all_passed": True,
        "regressions": [],
    }

    for cfg_path in configs:
        name = cfg_path.stem
        try:
            data = _load_config(cfg_path)
        except Exception as e:
            matrix["tracks"][name] = {"error": f"load failed: {e}"}
            matrix["all_passed"] = False
            continue

        # Inject ILC overrides only for race_01; other tracks fall through
        # to the curvature-derived defaults (so we test the generalisation
        # promise, not race_01's hand-tuned schedule).
        try:
            result = run_synthetic_benchmark(
                duration=duration, dt=dt, config=data,
            )
        except Exception as e:
            matrix["tracks"][name] = {"error": f"bench raised: {type(e).__name__}: {e}"}
            matrix["all_passed"] = False
            matrix["regressions"].append(f"{name}: exception {type(e).__name__}")
            continue

        track_summary = {
            "gates_passed": result.get("gates_passed", 0),
            "total_gates": result.get("total_gates", 0),
            "gate_pass_rate": result.get("gate_pass_rate", 0.0),
            "complete": result.get("complete", False),
            "crashed": result.get("crashed", False),
            "disqualified": result.get("disqualified", False),
            "termination_reason": result.get("termination_reason", "unknown"),
            "sim_time_s": result.get("sim_time_s", 0.0),
            "avg_tracking_error_m": result.get("avg_tracking_error_m", 0.0),
            "max_tracking_error_m": result.get("max_tracking_error_m", 0.0),
            "sim_passed": result.get("sim_passed", False),
            "plan_validation": result.get("plan_validation"),  # iter-004 Phase 1
            "is_placeholder": data.get("placeholder", False),
        }
        matrix["tracks"][name] = track_summary

        # Acceptance criteria. Placeholder tracks (e.g. aigp_default) only
        # need to reach 50% completion since they're untested geometry;
        # production tracks must hit `sim_passed` AND have a legal plan.
        if track_summary["is_placeholder"]:
            if track_summary["gate_pass_rate"] < 0.50:
                matrix["all_passed"] = False
                matrix["regressions"].append(
                    f"{name}: placeholder gate_pass_rate {track_summary['gate_pass_rate']:.0%} < 50%"
                )
        else:
            if track_summary["crashed"]:
                matrix["all_passed"] = False
                matrix["regressions"].append(
                    f"{name}: crashed ({track_summary['termination_reason']})"
                )
            if track_summary["disqualified"]:
                matrix["all_passed"] = False
                matrix["regressions"].append(
                    f"{name}: disqualified ({track_summary['termination_reason']})"
                )
            if track_summary["gate_pass_rate"] < 0.75:
                matrix["all_passed"] = False
                matrix["regressions"].append(
                    f"{name}: gate_pass_rate {track_summary['gate_pass_rate']:.0%} < 75%"
                )
            # Iter-006 F11 (Opus) — patched in iter-008: the original
            # `pv.get("ok") is False` had 3 holes:
            #   1) missing plan_validation field (None) treated as OK
            #   2) non-bool ok values (e.g. "yes") accepted as truthy
            #   3) any non-False value (None, 0, "") slipped through
            # Now we require an EXPLICIT True bool; anything else is a
            # regression on production tracks.
            pv = track_summary.get("plan_validation")
            if pv is None:
                matrix["all_passed"] = False
                matrix["regressions"].append(
                    f"{name}: plan_validation MISSING — bench didn't emit it"
                )
            elif pv.get("ok") is not True:
                # Catches False, None, 0, "", truthy-non-bool, etc.
                matrix["all_passed"] = False
                matrix["regressions"].append(
                    f"{name}: plan_validation.ok={pv.get('ok')!r} — {pv.get('reason', 'unknown')}"
                )

    return matrix


def _print_human(matrix: Dict[str, Any], file=sys.stderr) -> None:
    p = lambda *a, **kw: print(*a, **kw, file=file)
    p(f"\n{'=' * 72}")
    p("AI Grand Prix — Multi-Track Regression Matrix")
    p(f"{'=' * 72}")
    p(f"Duration per track: {matrix['duration_s']}s    dt: {matrix['dt']}s\n")
    fmt = "{:<22} {:>6} {:>5} {:>8} {:>7} {:>8} {:>10} {:<24}"
    p(fmt.format("track", "gates", "%", "complete",
                  "crashed", "DQ", "term_reason", "tag"))
    p("-" * 72)
    for name, t in matrix["tracks"].items():
        if "error" in t:
            p(fmt.format(name, "-", "-", "-", "-", "-", "-", f"ERR: {t['error'][:24]}"))
            continue
        tag = "placeholder" if t["is_placeholder"] else "production"
        p(fmt.format(
            name,
            f"{t['gates_passed']}/{t['total_gates']}",
            f"{int(t['gate_pass_rate']*100)}",
            "Y" if t["complete"] else "N",
            "Y" if t["crashed"] else "N",
            "Y" if t["disqualified"] else "N",
            t["termination_reason"][:18],
            tag,
        ))
    p("\n" + "=" * 72)
    if matrix["all_passed"]:
        p("Overall: PASS")
    else:
        p(f"Overall: FAIL ({len(matrix['regressions'])} regression(s))")
        for r in matrix["regressions"]:
            p(f"  - {r}")
    p("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic bench across all track configs",
    )
    parser.add_argument(
        "--configs", type=str, default=None,
        help="Comma-separated config names (without .json); default: all in sim_pybullet/configs/",
    )
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Per-track sim duration in seconds (default 30)")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--human", action="store_true",
                        help="Also print ASCII table to stderr")
    parser.add_argument("--json-only", action="store_true",
                        help="Suppress the ASCII table (overrides --human)")
    args = parser.parse_args()

    if args.configs:
        wanted = set(args.configs.split(","))
        all_paths = _list_configs()
        paths = [p for p in all_paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            print(f"ERROR: unknown configs: {sorted(missing)}", file=sys.stderr)
            return 2
    else:
        paths = _list_configs()

    if not paths:
        print("ERROR: no track configs found", file=sys.stderr)
        return 2

    matrix = run_matrix(paths, duration=args.duration, dt=args.dt)
    print(json.dumps(matrix, indent=2))
    if args.human and not args.json_only:
        _print_human(matrix)

    return 0 if matrix["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
