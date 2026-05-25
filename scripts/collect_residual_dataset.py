"""
Iter-025: collect a multi-track tracker-residual training dataset.

Drives `run_synthetic_benchmark` on every non-figure8 matrix track with
`trace_features=True`, concatenates the resulting `(features_10d,
roll_nom, pitch_nom, thrust_nom, pos_err, vel_err)` traces, and writes
the lot to a single .npz file via
`control.learned_residual.save_feature_trace`.

The output is consumed by a future training script (iter-026+) that
fits the `TrackerResidualMLP` weights via feedback-error-learning or
least-squares.

Design choices:
  - Skip figure8: matrix-known unsolvable; trace would be dominated by
    the crash-at-gate-5 failure mode, biasing the dataset.
  - Skip placeholder tracks (`aigp_default`): geometry is untuned, has
    much higher tracking error (~0.205m); inflates the residual target
    signal. Could be relaxed once aigp_default's racing line is tuned.
  - Concatenate raw — don't normalise features here. Normalisation is a
    training-time concern (training script can fit a StandardScaler).
  - Each entry's `pos_err` and `vel_err` are world-frame xyz tuples; the
    training script can decide whether to project to gate-frame or use
    raw.

Usage:
    python scripts/collect_residual_dataset.py [--out PATH] [--duration SECONDS]

Default PATH: control/residual_dataset.npz (gitignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Tracks excluded from the training dataset (see module docstring).
_SKIP = {"figure8", "aigp_default"}


def collect(out_path: Path, duration: float = 25.0) -> dict:
    """Run the bench on every non-skipped track config, concatenate
    traces, save to .npz. Returns a small summary dict."""
    from control.learned_residual import save_feature_trace
    from scripts.benchmark import run_synthetic_benchmark
    from scripts.benchmark_matrix import _list_configs

    combined: list = []
    per_track: list = []
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
        combined.extend(trace)

    if not combined:
        raise RuntimeError(
            "no traces collected — every track returned an empty trace"
        )

    save_feature_trace(combined, out_path)
    summary = {
        "total_samples": len(combined),
        "tracks": per_track,
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
