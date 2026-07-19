"""
Curvature-derived ILC section partition (iter-001 A9).

Replaces the wall-clock magic numbers that used to live in
`scripts/benchmark.py` (`inflection_start = int(2.0/dt)`, etc. — tuned to
race_01's helix at gate-3 / gate-6 / gate-7). A generic algorithm now
identifies high-curvature segments from the trajectory itself and
produces a list of (start, end, alpha, max_corr, cutoff, vel_scale)
tuples for `compute_ilc_offset_table`.

Per-section hyperparameters and partition tuning come from
`config/ilc_defaults.json`. Track configs may override the whole
partition by providing `ilc_section_overrides` in their JSON.

Research backing:
- Bristow & Alleyne 2007 (ACC): segment-wise ILC with per-section Q-filter
  bandwidth. The repo's existing ILC implementation references this.
- Zhang, Meng & Cai 2024: segment-wise ILC prevents cross-contamination
  between dissimilar dynamics.
- van Haren et al. 2024 (ECC): frequency-domain ILC with class-specific
  cutoffs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "ilc_defaults.json"


def load_ilc_config(path: Optional[Path] = None) -> dict:
    """Load `config/ilc_defaults.json` (or a caller-provided override path)."""
    p = path or DEFAULT_CONFIG_PATH

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in ILC config: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-standard JSON numeric constant in ILC config: {value}")

    with open(p, encoding="utf-8") as handle:
        config = json.load(
            handle,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    if not isinstance(config, dict):
        raise TypeError("ILC config root must be a JSON object")
    return config


def derive_section_boundaries(
    trajectory,
    dt: float,
    config: Optional[dict] = None,
    n_total_steps: Optional[int] = None,
) -> List[Tuple]:
    """Return ILC section tuples derived from the trajectory's curvature.

    Returned tuple format:
        ``(start_step, end_step, alpha, max_correction_m, filter_cutoff_hz, vel_scale)``
    — the same six-element format `compute_ilc_offset_table` accepts.

    Behaviour:
      - Empty / zero-acceleration trajectory → one global section using
        the "low" class.
      - Otherwise: per-point acceleration magnitude is computed, a quantile
        threshold partitions points into "low" and "high" curvature classes,
        runs of identical class are merged, and any below the minimum size
        get merged into their neighbour.
      - Section count is capped at `partition.n_sections_max` (default 4)
        by repeatedly merging the smallest-and-its-neighbour.

    The trajectory is duck-typed: any object with `.points` (a sequence of
    objects with `.time` and `.acceleration`) and `.total_time` works,
    which makes unit testing easy without a full `RaceTrajectory`.

    Args:
        trajectory: object with `.points` (each having `.time: float` and
            `.acceleration: Tuple[float, float, float]`) and `.total_time`.
        dt: simulation timestep used by the benchmark.
        config: pre-loaded ILC defaults dict; if None, loads from disk.
        n_total_steps: total step budget for the run; the returned sections
            cover [0, n_total_steps]. If None, derived from
            `trajectory.total_time / dt` + 50 (matches existing bench behaviour).
    """
    cfg = config or load_ilc_config()
    sections_cfg = cfg["sections"]
    partition_cfg = cfg.get("partition", {})
    quantile = float(partition_cfg.get("curvature_quantile_threshold", 0.6))
    n_max = int(partition_cfg.get("n_sections_max", 4))
    min_steps = int(partition_cfg.get("min_section_steps", 50))

    low_args = _class_to_tuple_args(sections_cfg["low"])
    high_args = _class_to_tuple_args(sections_cfg["high"])

    points = list(getattr(trajectory, "points", []) or [])
    total_time = float(getattr(trajectory, "total_time", 0.0))
    if n_total_steps is None:
        n_total_steps = int(total_time / dt) + 50 if total_time > 0 else 0

    if n_total_steps <= 0 or not points:
        return [(0, max(n_total_steps, 1)) + low_args]

    # Per-point acceleration magnitude. Use numpy for the quantile.
    accels = np.array(
        [
            math.sqrt(sum(float(a) * float(a) for a in p.acceleration))
            for p in points
        ],
        dtype=np.float64,
    )
    if accels.size == 0 or float(accels.max()) <= 1e-6:
        return [(0, n_total_steps) + low_args]

    # Iter-001 review Opus F5: use a RELATIVE threshold (5% of max-accel)
    # instead of an absolute 1e-6 m/s² floor. A track dominated by
    # zero-accel straights with one sharp turn has quantile=0; an absolute
    # 1e-6 floor then collapses everything to "all low," hiding the real
    # high-curvature minority. The relative floor catches a 20 m/s² peak
    # (relative threshold = 1.0 m/s²) even when surrounded by zeros.
    raw_threshold = float(np.quantile(accels, quantile))
    relative_floor = float(accels.max()) * 0.05
    threshold = max(raw_threshold, relative_floor)
    if threshold <= 1e-6:
        return [(0, n_total_steps) + low_args]

    classes = ["high" if a > threshold else "low" for a in accels.tolist()]

    # Walk and segment into runs of same class (in point-index space).
    point_runs: List[Tuple[int, int, str]] = []
    start_pt = 0
    cur_class = classes[0]
    for i in range(1, len(classes)):
        if classes[i] != cur_class:
            point_runs.append((start_pt, i, cur_class))
            start_pt = i
            cur_class = classes[i]
    point_runs.append((start_pt, len(classes), cur_class))

    # Convert point indices to step indices via point.time / dt.
    # Iter-001 review Opus F6: use round-to-nearest instead of floor.
    # With floor (the previous behaviour), point times that quantise to
    # the same step index produced zero-length runs that got silently
    # dropped at the `if e_step <= s_step: continue` filter below. With
    # rounding the boundary lands on the closest step, preserving the
    # partition's coverage when trajectory sample spacing is finer than
    # the simulation step.
    def _pt_to_step(i: int) -> int:
        if i >= len(points):
            return n_total_steps
        return int(round(points[i].time / dt))

    step_runs: List[Tuple[int, int, str]] = []
    for (s_pt, e_pt, cls) in point_runs:
        s_step = _pt_to_step(s_pt)
        e_step = _pt_to_step(e_pt) if e_pt < len(points) else n_total_steps
        if e_step <= s_step:
            continue
        step_runs.append((s_step, e_step, cls))

    if not step_runs:
        return [(0, n_total_steps) + low_args]

    # Cover [0, n_total_steps]: extend first/last to the edges.
    first_start, first_end, first_cls = step_runs[0]
    if first_start > 0:
        step_runs[0] = (0, first_end, first_cls)
    last_start, last_end, last_cls = step_runs[-1]
    if last_end < n_total_steps:
        step_runs[-1] = (last_start, n_total_steps, last_cls)

    # Merge adjacent same-class runs.
    step_runs = _merge_adjacent_same_class(step_runs)

    # Merge below-min sections into their neighbour.
    step_runs = _merge_below_min(step_runs, min_steps)

    # Cap section count by greedy smallest-with-neighbour merging.
    step_runs = _cap_section_count(step_runs, n_max)

    # Materialise to the six-element tuples consumed by
    # compute_ilc_offset_table.
    out: List[Tuple] = []
    for (s, e, cls) in step_runs:
        args = high_args if cls == "high" else low_args
        out.append((s, e) + args)
    return out


def _class_to_tuple_args(class_cfg: dict) -> Tuple[float, float, float, float]:
    """Pack a `sections.<class>` config into the (alpha, max_corr, cutoff, vel_scale) tuple."""
    return (
        float(class_cfg["alpha"]),
        float(class_cfg["max_correction_m"]),
        float(class_cfg["filter_cutoff_hz"]),
        float(class_cfg["vel_scale"]),
    )


def _merge_adjacent_same_class(
    runs: List[Tuple[int, int, str]],
) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    for r in runs:
        if out and out[-1][2] == r[2]:
            out[-1] = (out[-1][0], r[1], out[-1][2])
        else:
            out.append(r)
    return out


def _merge_below_min(
    runs: List[Tuple[int, int, str]], min_steps: int,
) -> List[Tuple[int, int, str]]:
    if not runs:
        return runs
    out: List[Tuple[int, int, str]] = []
    for r in runs:
        size = r[1] - r[0]
        if size < min_steps and out:
            out[-1] = (out[-1][0], r[1], out[-1][2])
        else:
            out.append(r)
    return out


def _cap_section_count(
    runs: List[Tuple[int, int, str]], n_max: int,
) -> List[Tuple[int, int, str]]:
    runs = list(runs)
    while len(runs) > n_max:
        # Identify the smallest section.
        sizes = [r[1] - r[0] for r in runs]
        i = int(np.argmin(sizes))
        if len(runs) == 1:
            break
        # Pick the smaller of left/right neighbour to merge with.
        left = runs[i - 1] if i > 0 else None
        right = runs[i + 1] if i < len(runs) - 1 else None
        if left is None:
            assert right is not None
            merged = (runs[i][0], right[1], right[2])
            runs = [merged] + runs[i + 2:]
        elif right is None:
            merged = (left[0], runs[i][1], left[2])
            runs = runs[:i - 1] + [merged]
        else:
            ls = left[1] - left[0]
            rs = right[1] - right[0]
            if ls <= rs:
                merged = (left[0], runs[i][1], left[2])
                runs = runs[:i - 1] + [merged] + runs[i + 1:]
            else:
                merged = (runs[i][0], right[1], right[2])
                runs = runs[:i] + [merged] + runs[i + 2:]
    return runs
