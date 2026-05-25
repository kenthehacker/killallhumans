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
        assert "sim_pybullet/configs" in str(p)


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
    # Iter-009g: also catch moderate slowdowns. The current test already
    # catches catastrophic slowdowns (sim_time > duration → time_limit
    # termination → sim_passed=False), but a 17.2s → 25s drift would
    # silently pass. Competition lap-time matters: gate the sim_time at
    # 1.3× the iter-009 baseline (≈22.4s). Anything slower means the
    # tracker or planner has regressed in a meaningful way.
    assert track["sim_time_s"] < 22.5, (
        f"race_01 sim_time regressed to {track['sim_time_s']:.2f}s "
        f"(threshold 22.5s; iter-009 baseline 17.17s) — perf regression"
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

    Per-track sim_time baselines (iter-009f, duration=30s):
      aigp_default       7.8s   → ceiling 12.5s
      grand_tour        18.3s   → ceiling 29.5s
      race_01           17.2s   → ceiling 27.5s (overlaps test_race_01)
      slalom             8.2s   → ceiling 13.5s
      straight_hairpin   8.3s   → ceiling 13.5s
      vertical_cliff    11.5s   → ceiling 19.0s
    """
    # Iter-009h: per-track sim_time ceilings. Baselines measured 2026-05-24
    # at iter-009g's `aa5aea1`. Ceilings set at ~1.6× to allow moderate
    # changes without flapping but catch >60% slowdowns.
    SIM_TIME_CEILINGS = {
        "aigp_default": 12.5,
        "grand_tour": 29.5,
        "race_01": 27.5,  # overlaps race_01 dedicated test, intentionally
        "slalom": 13.5,
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
