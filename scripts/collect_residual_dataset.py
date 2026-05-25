"""
Iter-031: collect a multi-track tracker-residual training dataset (v2).

Drives `run_synthetic_benchmark` on every non-figure8 matrix track with
`trace_features=True`, concatenates the resulting v2 traces, tags each
sample with its track index, and writes to a single .npz via
`control.learned_residual.save_feature_trace` + an extra `track_id`
field appended after.

Changes from iter-025:
  - **Includes `aigp_default`**: the placeholder goal track was excluded
    in iter-025 because its baseline tracking error inflated the iter-027
    FEL targets. The iter-031 yaw-corrected FEL target is bounded by
    construction (clamped ±0.05 rad) and per-track weighting keeps
    aigp_default's higher error from dominating.
  - **Skips only figure8**: matrix-known unsolvable; trace would be
    dominated by the crash-at-gate-5 failure mode, biasing the dataset.
  - **Tags `track_id`**: written as an extra field in the .npz next to
    the v2 trace fields so the trainer can compute per-track loss
    weights without re-parsing track names. The trainer raises if this
    field is missing (no silent fallback to one-track collapse).
  - v2 trace fields include `pos`, `vel`, `yaw_des`, `ref_*`,
    `accel_des_baseline` — the trainer reads them straight out for the
    yaw-corrected FEL target construction.

Usage:
    python scripts/collect_residual_dataset.py [--out PATH] [--duration SECONDS]

Default PATH: control/residual_dataset.npz (gitignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Tracks excluded from the training dataset (see module docstring).
# Iter-031 narrowed this from {figure8, aigp_default} → {figure8} so
# aigp_default contributes to training.
_SKIP = {"figure8"}


def collect(out_path: Path, duration: float = 25.0) -> dict:
    """Run the bench on every non-skipped track config, concatenate
    traces, save to .npz with track_id tags. Returns a summary dict."""
    from control.learned_residual import save_feature_trace
    from scripts.benchmark import run_synthetic_benchmark
    from scripts.benchmark_matrix import _list_configs

    combined: list = []
    per_track: list = []
    track_names: list = []
    track_ids: list = []
    for cfg_path in _list_configs():
        name = cfg_path.stem
        if name in _SKIP:
            per_track.append({"track": name, "skipped": True})
            continue
        with open(cfg_path) as f:
            data = json.load(f)
        result = run_synthetic_benchmark(
            duration=duration, config=data,
            tracker_config_overrides={"trace_features": True},
        )
        if result.get("skipped"):
            per_track.append({
                "track": name, "skipped": True,
                "skip_reason": result.get("skip_reason"),
            })
            continue
        trace = result.get("tracker_feature_trace") or []
        per_track.append({
            "track": name,
            "samples": len(trace),
            "sim_passed": result.get("sim_passed"),
            "sim_time_s": result.get("sim_time_s"),
            "avg_tracking_error_m": result.get("avg_tracking_error_m"),
        })
        if trace:
            track_id = len(track_names)
            track_names.append(name)
            track_ids.extend([track_id] * len(trace))
            combined.extend(trace)

    if not combined:
        raise RuntimeError(
            "no traces collected — every track returned an empty trace"
        )

    save_feature_trace(combined, out_path)
    # Append track_id + track_names to the same .npz so the trainer can
    # group samples without re-running collection. We rewrite the file
    # by loading, merging, and saving — small file, cheap operation.
    with np.load(out_path) as data:
        payload = {k: data[k] for k in data.files}
    payload["track_id"] = np.array(track_ids, dtype=np.int64)
    payload["track_names"] = np.array(track_names, dtype=object)
    np.savez(out_path, **payload)

    summary = {
        "total_samples": len(combined),
        "tracks": per_track,
        "track_names": track_names,
        "duration_per_track_s": duration,
        "out_path": str(out_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Collect tracker-residual training data from matrix tracks",
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "control" / "residual_dataset.npz"),
        help="Output .npz path",
    )
    parser.add_argument(
        "--duration", type=float, default=25.0,
        help="Per-track sim duration in seconds (default: 25.0)",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = collect(out_path, duration=args.duration)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
