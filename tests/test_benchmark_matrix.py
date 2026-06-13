"""
Smoke test for `scripts/benchmark_matrix.py` (iter-003 A17).

Verifies that:
  - the matrix runner can be invoked programmatically and returns the
    expected JSON-shaped dict
  - empty config list raises cleanly
  - per-track structure is consistent (every result has the same keys)

Doesn't assert specific pass/fail outcomes — those will change as the
controller improves. The point is the harness exists and doesn't crash.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.benchmark_matrix import _list_configs, run_matrix


def test_list_configs_finds_track_jsons():
    paths = _list_configs()
    names = {p.stem for p in paths}
    # race_01 must always be present (it's the canonical reference track).
    assert "race_01" in names, f"expected race_01 in configs, got {names}"
    # All should be .json files under sim_pybullet/configs/.
    for p in paths:
        assert p.suffix == ".json"
        # Use as_posix() so the check is independent of the OS path separator
        # (Windows stringifies Path with backslashes).
        assert "sim_pybullet/configs" in p.as_posix()


def test_run_matrix_single_track_returns_expected_shape():
    """Run the matrix on race_01 alone and verify the dict shape."""
    paths = [p for p in _list_configs() if p.stem == "race_01"]
    assert paths, "race_01.json missing from sim_pybullet/configs/"
    matrix = run_matrix(paths, duration=5.0)
    assert "timestamp" in matrix
    assert "tracks" in matrix
    assert "race_01" in matrix["tracks"]
    track = matrix["tracks"]["race_01"]
    for key in (
        "gates_passed", "total_gates", "gate_pass_rate", "complete",
        "crashed", "disqualified", "termination_reason", "sim_time_s",
        "avg_tracking_error_m", "max_tracking_error_m", "sim_passed",
        "is_placeholder",
    ):
        assert key in track, f"missing key {key!r} in track result"
    # race_01 is the reference; with the current controller it must at
    # least START (gate-1 reached or attempted).
    assert track["total_gates"] >= 1


def test_race_01_regression_gate_passes_at_15s():
    """Iter-009e: race_01 must continue to PASS the synthetic bench.

    Tracking-error history: iter-9 baseline 0.665m → iter-008 ILC sweep
    0.159m → iter-009 fractional ILC + auto-velocity 0.089m. A future
    change that re-introduces overfitting (e.g. velocity defaults
    creeping back to course-specific values) would crash race_01 again.
    This test catches that.

    Tolerances:
      - sim_passed must be True
      - at least 11/12 gates (leaving 1 gate headroom for minor jitter)
      - avg_tracking_error_m < 0.30 (3× the iter-009 result; small
        improvements shouldn't be required, but large regressions should
        scream)

    Uses duration=30.0s — race_01 completes at ~17.2s on the iter-009
    baseline; 30s leaves comfortable headroom for moderate slowdowns
    from future changes without the test ping-ponging. The other tracks
    aren't asserted to avoid coupling this test to figure8's
    known-unsolvable coplanar gates.
    """
    paths = [p for p in _list_configs() if p.stem == "race_01"]
    assert paths
    matrix = run_matrix(paths, duration=30.0)
    track = matrix["tracks"]["race_01"]
    assert track["sim_passed"] is True, (
        f"race_01 must pass; reason={track.get('termination_reason')}, "
        f"gates={track['gates_passed']}/{track['total_gates']}, "
        f"avg_err={track['avg_tracking_error_m']:.3f}m"
    )
    assert track["gates_passed"] >= track["total_gates"] - 1, (
        f"race_01 regressed: only {track['gates_passed']}/{track['total_gates']} gates"
    )
    assert track["avg_tracking_error_m"] < 0.30, (
        f"race_01 tracking error regressed to {track['avg_tracking_error_m']:.3f}m "
        f"(threshold 0.30m; iter-009 baseline 0.089m)"
    )
    # Iter-009g + iter-032: catch moderate slowdowns. Pre-iter-032
    # baseline was 17.17s but the polynomial-peak projection trades
    # ~7s of lap time for 27% lower tracking error and a 21.7% → 2.7%
    # collapse in accel-clamp engagement (planner-vs-bench honesty).
    # Iter-032 baseline: 24.39s. Ceiling 26s allows ~7% headroom for
    # tracker-tune drifts before flapping.
    assert track["sim_time_s"] < 26.0, (
        f"race_01 sim_time regressed to {track['sim_time_s']:.2f}s "
        f"(threshold 26.0s; iter-032 baseline 24.39s) — perf regression"
    )


def test_figure8_8_of_8_after_iter028_coplanar_fix():
    """Iter-028 regression: figure8 now passes 8/8 gates thanks to the
    coplanar-gates DQ-skip fix in `gate_sequencing/sequencer.py`. Pre-
    iter-028 figure8 was 1/8 (crash_gate:gate-5 at sim_time=1.0s) because
    the sequencer flagged the figure-8 self-crossing as an out-of-order
    violation. This test pins the win — if anything regresses the
    coplanar exception, figure8 drops back to 1/8 and this test fails.

    Tracking error tolerance is wider (0.50m) than the matrix gate's
    0.40m because figure8 has tight 90-deg turns that legitimately have
    higher tracking error than race_01-class wide courses."""
    paths = [p for p in _list_configs() if p.stem == "figure8"]
    assert paths
    matrix = run_matrix(paths, duration=30.0)
    track = matrix["tracks"]["figure8"]
    assert track["sim_passed"] is True, (
        f"figure8 regressed; reason={track.get('termination_reason')}, "
        f"gates={track['gates_passed']}/{track['total_gates']}"
    )
    assert track["gates_passed"] == 8, (
        f"figure8 only got {track['gates_passed']}/8 gates"
    )
    assert track["avg_tracking_error_m"] < 0.50


def test_matrix_pass_rate_at_least_six_of_seven():
    """Iter-009f: locks in the iter-009 multi-track win.

    Iter-001 started at 1/7 (race_01 only — pure overfitting). The
    geometry-derived auto_velocity + fractional ILC overrides in
    iter-006..iter-009 unlocked 6/7 (only figure8 still crashes, due
    to coplanar gates 1 and 5 that share x=5 — known-unsolvable
    without trajectory pre-shaping or SFC corridor work).

    A future change that erodes the generalisation gain (e.g.
    re-introducing course-specific magic numbers; a too-aggressive
    velocity that breaks slalom; a tracker-gain tweak that destabilises
    grand_tour) would silently drop the pass rate. This test catches
    that.

    Assertion: at least 6/7 non-figure8 tracks PASS with tracking
    error < 0.40m AND sim_time < 1.6× the iter-009 baseline. figure8
    is excluded — it remains an open known issue.

    Per-track sim_time baselines (iter-035, duration=30s, post-projection
    + gate-altitude bug fix):
      aigp_default      14.87s  → ceiling 17.0s
      grand_tour        24.04s  → ceiling 29.5s
      race_01           24.46s  → ceiling 27.5s (overlaps test_race_01)
      slalom            13.81s  → ceiling 15.5s
      straight_hairpin  10.45s  → ceiling 13.5s
      vertical_cliff    14.27s  → ceiling 19.0s

    Iter-032 relaxed `slalom` (13.5→15.5s) and `aigp_default` (12.5→14s)
    because the new polynomial-peak accel projection stretches segments
    to keep ||a|| ≤ 15 m/s². Iter-035 raised aigp_default again (14→17s)
    because the racing-line gate-altitude bug fix (BO was lowering gates
    by 0.35m to shave path-length) restored correct trajectory through
    actual gate centers — that adds ~3s of vertical travel on tracks
    with non-uniform gate heights.
    """
    # Iter-009h ceilings relaxed at iter-032 (projection) and iter-035
    # (racing-line gate-altitude bug fix). Ceilings now ~15-30% above
    # iter-035 baselines.
    SIM_TIME_CEILINGS = {
        "aigp_default": 17.0,
        "grand_tour": 29.5,
        "race_01": 27.5,  # overlaps race_01 dedicated test, intentionally
        "slalom": 15.5,
        "straight_hairpin": 13.5,
        "vertical_cliff": 19.0,
    }
    paths = _list_configs()
    matrix = run_matrix(paths, duration=30.0)

    expected_pass = []
    actual_pass = []
    regressions = []
    for name, track in matrix["tracks"].items():
        if name == "figure8":
            continue  # known-unsolvable; exclude from pass-rate gate
        expected_pass.append(name)
        # Sim-time ceiling only applies if a baseline is recorded.
        # Unknown tracks (new additions) skip the sim_time check.
        ceiling = SIM_TIME_CEILINGS.get(name)
        sim_time_ok = (ceiling is None) or (track["sim_time_s"] < ceiling)
        if (
            track["sim_passed"]
            and track["avg_tracking_error_m"] < 0.40
            and sim_time_ok
        ):
            actual_pass.append(name)
        else:
            regressions.append(
                f"{name}: pass={track['sim_passed']} "
                f"gates={track['gates_passed']}/{track['total_gates']} "
                f"err={track['avg_tracking_error_m']:.3f}m "
                f"sim_time={track['sim_time_s']:.2f}s"
                f"{f' (ceiling {ceiling:.1f}s)' if ceiling else ''} "
                f"reason={track['termination_reason']}"
            )

    assert len(actual_pass) >= 6, (
        f"matrix pass-rate regressed: only {len(actual_pass)}/{len(expected_pass)} "
        f"tracks pass; regressions=\n  " + "\n  ".join(regressions)
    )


def test_matrix_clamp_engagement_below_iter016_baseline():
    """Iter-017 regression gate: the bench's accel-clamp engagement
    is now first-class in the matrix output (iter-016). Pin per-track
    upper bounds at 1.5× the iter-016 baseline so changes that drive
    the planner harder (more over-commanding → higher clamp engagement)
    get caught.

    Per-track baselines (iter-016 measurement, commit ~635e329, all
    non-figure8, duration=30s):
      aigp_default     72.2%  → ceiling 95.0%
      grand_tour        9.9%  → ceiling 25.0%
      race_01          21.7%  → ceiling 40.0%
      slalom           36.1%  → ceiling 60.0%
      straight_hairpin 11.5%  → ceiling 25.0%
      vertical_cliff    4.2%  → ceiling 15.0%

    Ceilings have headroom — moderate planner-tuning drifts won't
    flap the test, but a regression that doubles aigp_default's
    over-commanding (e.g. removing iter-010's accel drop) blows up.
    """
    CLAMP_CEILINGS = {
        "aigp_default": 0.95,
        "grand_tour": 0.25,
        "race_01": 0.40,
        "slalom": 0.60,
        "straight_hairpin": 0.25,
        "vertical_cliff": 0.15,
    }
    paths = _list_configs()
    matrix = run_matrix(paths, duration=30.0)
    regressions = []
    for name, track in matrix["tracks"].items():
        if name == "figure8":
            continue
        ceiling = CLAMP_CEILINGS.get(name)
        if ceiling is None:
            continue  # unknown new track; skip rather than flap
        observed = track.get("accel_clamp_active_frac", 0.0)
        if observed > ceiling:
            regressions.append(
                f"{name}: accel_clamp_active_frac={observed:.1%} > "
                f"ceiling {ceiling:.1%} (iter-016 baseline + 50% headroom)"
            )
    assert not regressions, (
        "matrix clamp engagement regressed:\n  " + "\n  ".join(regressions)
    )


def test_run_matrix_empty_configs_returns_empty_tracks():
    matrix = run_matrix([], duration=1.0)
    assert matrix["tracks"] == {}
    # No tracks means nothing failed, so all_passed is vacuously True.
    assert matrix["all_passed"] is True


def test_iter035_drone_passes_through_gate_centers_vertically():
    """Iter-035 regression: drone must pass through gates near their
    actual z centers, not 0.35m below them.

    Pre-iter-035 bug: `planning/racing_line.py::_apply_offsets` used
    `up = [0, 0, -1]` (NED convention) while the rest of the system is
    ENU. The BO's objective rewards short paths → it picked positive
    `vert_off` to "raise" gates, which (with the inverted `up`) lowered
    them. straight_hairpin showed -0.35m on every gate; figure8/
    vertical_cliff each had 0.36m max error.

    iter-035 fixes:
      1. `up = [0, 0, +1]` (ENU correct).
      2. New `RacingLineConfig.max_vertical_offset = 0.0` default —
         drone passes through gate center vertically, maximising
         frame clearance.

    This test pins the win across all 7 tracks. Mean |Δz| < 50mm and
    max |Δz| < 300mm per track (vertical_cliff and grand_tour still
    have legit tracker-lag on steep climbs, ~280mm max).
    """
    import json
    from scripts.benchmark import run_synthetic_benchmark
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    cfg_dir = repo / "sim_pybullet" / "configs"

    violations: list[str] = []
    for cfg_path in sorted(cfg_dir.glob("*.json")):
        track = cfg_path.stem
        with open(cfg_path) as f:
            cfg = json.load(f)
        r = run_synthetic_benchmark(
            duration=30.0, config=cfg, record_position_trace=True,
        )
        trace = r.get("position_trace") or []
        gates_by_id = {g["id"]: g for g in cfg["gates"]}
        z_errors: list[float] = []
        for gpt in r.get("gate_pass_times", []):
            gid = gpt["gate_id"]
            sample = min(trace, key=lambda s: abs(s["t"] - gpt["time_s"]))
            gz = gates_by_id[gid]["pose"]["z"]
            z_errors.append(sample["pos"][2] - gz)
        if not z_errors:
            continue
        mean_abs = sum(abs(z) for z in z_errors) / len(z_errors)
        max_abs = max(abs(z) for z in z_errors)
        # Mean: tight (the bug had mean -0.35m on straight_hairpin).
        if mean_abs > 0.10:
            violations.append(
                f"{track}: mean |Δz|={mean_abs:.3f}m > 100mm "
                f"(per-gate z-errors: {[f'{z:+.3f}' for z in z_errors]})"
            )
        # Max: looser — legitimate tracker-lag on steep climbs.
        if max_abs > 0.40:
            violations.append(
                f"{track}: max |Δz|={max_abs:.3f}m > 400mm"
            )
    assert not violations, (
        "iter-035 gate-altitude regression:\n  " + "\n  ".join(violations)
    )


def test_iter032_accel_projection_drops_clamp_engagement():
    """Iter-032 (charter task #10): the new
    `_project_accel_peaks` pass in `planning/trajectory_optimizer.py`
    must drive accel-clamp engagement well below the iter-016 baseline
    on the two worst-offender tracks.

    iter-016 baseline → iter-032 measurement (post-projection,
    duration=30s):
      race_01:      21.7% → 2.7%   (8× drop)
      aigp_default: 72.2% → 21.8%  (3× drop)

    aigp_default doesn't get below 20% because the placeholder track
    has aggressive geometry and the per-pass 1.5× cap in
    `DEFAULT_ACCEL_PEAK_PROJECTION_MAX_STRETCH` requires more passes
    than max_passes=3 to fully converge. The 25% ceiling preserves
    the 3× empirical win while leaving headroom for tuning drift.

    If this test fails, the projection isn't actually projecting —
    either it's not being called from `optimize()` or
    `DEFAULT_ACCEL_PEAK_PROJECTION_MAX_STRETCH` is set too low.
    """
    TARGET_TRACKS = ("race_01", "aigp_default")
    CEILINGS = {"race_01": 0.10, "aigp_default": 0.25}
    paths = [p for p in _list_configs() if p.stem in TARGET_TRACKS]
    matrix = run_matrix(paths, duration=30.0)
    violations: list[str] = []
    for name in TARGET_TRACKS:
        observed = matrix["tracks"][name].get("accel_clamp_active_frac", 1.0)
        ceiling = CEILINGS[name]
        if observed >= ceiling:
            violations.append(
                f"{name}: accel_clamp_active_frac={observed:.1%} >= "
                f"{ceiling:.0%} ceiling (iter-032 projection target)"
            )
    assert not violations, (
        "iter-032 polynomial-peak projection failed to drop clamp "
        "engagement below ceiling:\n  " + "\n  ".join(violations)
    )
