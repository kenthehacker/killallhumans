"""
Smoke test for `scripts/benchmark_matrix.py` (iter-003 A17).

Verifies that:
  - the matrix runner can be invoked programmatically and returns the
    expected JSON-shaped dict
  - empty config list raises cleanly
  - per-track structure is consistent (every result has the same keys)

Pins the current seven-course clean-completion baseline, safety/validity
envelope, and explicit per-course time/error ceilings.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.benchmark import COMPARISON_SERIES, EVALUATOR_VERSION, prepare_course
from scripts.benchmark_matrix import (
    _list_configs,
    _load_config,
    run_matrix,
    worker_numeric_environment,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.timeout(300)]


def test_preparation_uses_worker_identity_and_forced_rollout_hits_warm_layers(tmp_path):
    race_path = next(path for path in _list_configs() if path.stem == "race_01")
    cache_root = tmp_path / "artifacts"
    with worker_numeric_environment() as preparation_fingerprint:
        prepared = prepare_course(
            _load_config(race_path), dt=0.01, cache_root=cache_root
        )
    matrix = run_matrix(
        [race_path],
        duration=0.01,
        dt=0.01,
        max_workers=1,
        cache_root=cache_root,
        include_results=True,
        use_result_cache=False,
    )
    result = matrix["results"]["race_01"]
    assert prepared.dependency_fingerprint == preparation_fingerprint
    assert matrix["dependency_fingerprint"] == preparation_fingerprint
    assert result["cache"]["benchmark_result"] == "miss"
    assert result["rollout_executed"] is True
    assert result["result_cache_enabled"] is False
    assert {
        layer: result["cache"][layer]
        for layer in ("racing_line", "trajectory", "plan_validation", "ilc")
    } == {
        "racing_line": "hit",
        "trajectory": "hit",
        "plan_validation": "hit",
        "ilc": "hit",
    }


@pytest.fixture(scope="session")
def prepared_matrix_result(tmp_path_factory):
    """One isolated seven-track execution shared by all matrix assertions."""

    cache_root = tmp_path_factory.mktemp("benchmark-artifacts")
    return run_matrix(
        _list_configs(),
        duration=30.0,
        max_workers=4,
        cache_root=cache_root,
        record_position_trace=True,
        include_results=True,
    )


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


def test_run_matrix_single_track_returns_expected_shape(prepared_matrix_result):
    """Run the matrix on race_01 alone and verify the dict shape."""
    paths = [p for p in _list_configs() if p.stem == "race_01"]
    assert paths, "race_01.json missing from sim_pybullet/configs/"
    matrix = prepared_matrix_result
    assert matrix["evaluator_version"] == EVALUATOR_VERSION
    assert matrix["comparison_series"] == COMPARISON_SERIES
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


def test_race_01_evaluator_v4_exact_state_time_regression(prepared_matrix_result):
    """Pin the current clean v4 completion under exact plane scoring.

    The gate-normal throat and conservative modeled-drag compensation now
    complete 12/12 safely in roughly 21.5 simulated seconds. A future narrow
    pre-plane miss is a regression, not a result proximity credit may revive.
    """
    paths = [p for p in _list_configs() if p.stem == "race_01"]
    assert paths
    matrix = prepared_matrix_result
    track = matrix["tracks"]["race_01"]
    assert track["evaluator_version"] == EVALUATOR_VERSION
    assert track["gates_passed"] == track["total_gates"] == 12
    assert track["complete"] is True
    assert track["sim_passed"] is True
    assert track["crashed"] is False
    assert track["disqualified"] is False
    assert track["safety_passed"] is True
    assert track["validity_passed"] is True
    assert track["termination_reason"] == "race_complete"
    assert track["threshold_failures"] == []
    # The current evaluator scores the scheduled reference (including ILC),
    # so it no longer inherits the old globally-nearest-path 0.089 m baseline.
    assert track["avg_tracking_error_m"] < 0.65, (
        f"race_01 tracking error regressed to {track['avg_tracking_error_m']:.3f}m "
        "(v4 exact-state-time scheduled-reference ceiling 0.65m)"
    )
    assert track["avg_nearest_path_error_m"] < 0.15
    # Iter-009g + iter-032: catch moderate slowdowns. Pre-iter-032
    # baseline was 17.17s but the polynomial-peak projection trades
    # ~7s of lap time for 27% lower tracking error and a 21.7% → 2.7%
    # collapse in accel-clamp engagement (planner-vs-bench honesty).
    # Iter-032 baseline: 24.39s. Ceiling 26s allows ~7% headroom for
    # tracker-tune drifts before flapping.
    assert track["sim_time_s"] < 26.0


def test_figure8_v4_clean_completion_regression(prepared_matrix_result):
    """Figure8 now completes cleanly; pin that stronger plane-crossing result."""
    paths = [p for p in _list_configs() if p.stem == "figure8"]
    assert paths
    matrix = prepared_matrix_result
    track = matrix["tracks"]["figure8"]
    assert track["sim_passed"] is True
    assert track["complete"] is True
    assert track["gates_passed"] == track["total_gates"] == 8
    assert track["crashed"] is False
    assert track["disqualified"] is False
    assert track["safety_passed"] is True
    assert track["validity_passed"] is True
    assert track["plan_validation"]["ok"] is True
    assert track["termination_reason"] == "race_complete"
    assert track["threshold_failures"] == []
    assert track["avg_tracking_error_m"] < 0.40
    assert track["sim_time_s"] < 18.0


def test_evaluator_v4_matrix_pins_honest_completed_tracks(prepared_matrix_result):
    """Pin the new comparison series without reviving proximity credit.

    Current prepared evidence completes every track safely. Per-track time and
    scheduled-error ceilings below still prevent that stronger completion
    result from hiding a tracking or timing regression.
    """
    # These ceilings belong only to the current scheduled-reference evaluator.
    SIM_TIME_CEILINGS = {
        "aigp_default": 15.0,
        "figure8": 18.0,
        "grand_tour": 24.0,
        "race_01": 26.0,
        "slalom": 14.0,
        "straight_hairpin": 12.0,
        "vertical_cliff": 16.0,
    }
    ERROR_CEILINGS = {
        "aigp_default": 0.30,
        "figure8": 0.40,
        "grand_tour": 0.75,
        "race_01": 0.65,
        "slalom": 0.25,
        "straight_hairpin": 0.75,
        "vertical_cliff": 0.25,
    }
    paths = _list_configs()
    matrix = prepared_matrix_result

    expected_tracks = {path.stem for path in paths}
    assert set(SIM_TIME_CEILINGS) == set(ERROR_CEILINGS) == expected_tracks
    assert set(matrix["tracks"]) == expected_tracks
    assert set(matrix["results"]) == expected_tracks
    assert matrix["all_passed"] is True
    assert matrix["safety_passed"] is True
    assert matrix["validity_passed"] is True
    assert matrix["completion"]["complete"] is True
    assert matrix["completion"]["requested_tracks"] == len(expected_tracks) == 7
    assert matrix["completion"]["evaluated_tracks"] == len(expected_tracks)
    assert matrix["completion"]["evidence_complete"] is True
    assert set(matrix["completion"]["track_result_references"]) == {
        f"tracks.{name}.completion" for name in expected_tracks
    }
    assert matrix["regressions"] == []
    assert matrix["failure_summary"]["exception"] is None
    assert matrix["failure_summary"]["threshold_failures"] == []
    assert matrix["worker_environment_verified"] is True
    assert matrix["code_provenance_verified"] is True
    assert matrix["resolved_configuration"]["code_provenance"] == matrix[
        "code_provenance"
    ]
    worker_observations = matrix["dependency_fingerprints"]["workers_observed"]
    assert {item["track"] for item in worker_observations} == expected_tracks
    assert all(
        item["fingerprint"] == matrix["dependency_fingerprint"]
        for item in worker_observations
    )

    mirrored_fields = (
        "gates_passed",
        "total_gates",
        "gate_pass_rate",
        "complete",
        "crashed",
        "disqualified",
        "termination_reason",
        "sim_time_s",
        "avg_tracking_error_m",
        "max_tracking_error_m",
        "p95_tracking_error_m",
        "avg_nearest_path_error_m",
        "max_nearest_path_error_m",
        "sim_passed",
        "safety_passed",
        "validity_passed",
        "completion",
        "plan_validation",
        "evaluator_version",
        "schema_version",
        "config_hash",
        "artifact_hashes",
        "cache_hit_or_miss",
        "rollout_executed",
        "result_cache_enabled",
        "threshold_failures",
    )
    for name in expected_tracks:
        summary = matrix["tracks"][name]
        full = matrix["results"][name]
        assert full["available"] is True
        assert full["skipped"] is False
        assert full["failure_summary"]["exception"] is None
        assert full["threshold_failures"] == []
        assert all(summary[field] == full[field] for field in mirrored_fields)
        assert summary["prepared_cache_states"] == {
            layer: full["cache"][layer]
            for layer in ("racing_line", "trajectory", "plan_validation", "ilc")
        }
        assert summary["is_placeholder"] is bool(
            full["resolved_configuration"]["track"].get("placeholder", False)
        )
        assert full["dependency_fingerprint"] == matrix["dependency_fingerprint"]
        assert full["code_provenance"] == matrix["code_provenance"]

    # The placeholder label is diagnostic metadata only. It is deliberately
    # included in every hard-gate assertion above and cannot relax completion,
    # safety, validity, or threshold evidence.
    assert {
        name for name, track in matrix["tracks"].items() if track["is_placeholder"]
    } == {"aigp_default"}
    assert matrix["completion"]["gates_passed"] == sum(
        result["gates_passed"] for result in matrix["results"].values()
    )
    assert matrix["completion"]["total_gates"] == sum(
        result["total_gates"] for result in matrix["results"].values()
    )

    regressions = []
    for name, track in matrix["tracks"].items():
        ceiling = SIM_TIME_CEILINGS[name]
        error_ceiling = ERROR_CEILINGS[name]
        if (
            track["sim_passed"]
            and track["safety_passed"]
            and track["validity_passed"]
            and track["complete"]
            and track["gates_passed"] == track["total_gates"]
            and not track["crashed"]
            and not track["disqualified"]
            and track["termination_reason"] == "race_complete"
            and track["avg_tracking_error_m"] < error_ceiling
            and track["sim_time_s"] < ceiling
        ):
            continue
        else:
            regressions.append(
                f"{name}: pass={track['sim_passed']} "
                f"gates={track['gates_passed']}/{track['total_gates']} "
                f"err={track['avg_tracking_error_m']:.3f}m "
                f"sim_time={track['sim_time_s']:.2f}s"
                f" (time ceiling {ceiling:.1f}s, "
                f"error ceiling {error_ceiling:.2f}m) "
                f"reason={track['termination_reason']}"
            )

    assert not regressions, "matrix completion regression:\n  " + "\n  ".join(regressions)


def test_matrix_clamp_engagement_below_iter016_baseline(prepared_matrix_result):
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
    matrix = prepared_matrix_result
    regressions = []
    for name, track in matrix["tracks"].items():
        if name == "figure8":
            continue
        ceiling = CLAMP_CEILINGS.get(name)
        if ceiling is None:
            continue  # unknown new track; skip rather than flap
        observed = track.get("accel_clamp_active_frac")
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not 0.0 <= float(observed) <= 1.0
        ):
            regressions.append(
                f"{name}: missing or invalid accel_clamp_active_frac={observed!r}"
            )
            continue
        if observed > ceiling:
            regressions.append(
                f"{name}: accel_clamp_active_frac={observed:.1%} > "
                f"ceiling {ceiling:.1%} (iter-016 baseline + 50% headroom)"
            )
    assert not regressions, (
        "matrix clamp engagement regressed:\n  " + "\n  ".join(regressions)
    )


def test_run_matrix_empty_configs_fails_closed():
    matrix = run_matrix([], duration=1.0)
    assert matrix["tracks"] == {}
    assert matrix["all_passed"] is False
    assert any("evidence is missing" in item for item in matrix["regressions"])
    assert matrix["failure_summary"]["threshold_failures"] == matrix["regressions"]


def test_run_matrix_rejects_duplicate_config_identity(tmp_path):
    config = tmp_path / "track.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="uniquely attributable"):
        run_matrix([config, config], duration=1.0)

    first = tmp_path / "one" / "duplicate.json"
    second = tmp_path / "two" / "duplicate.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate stems"):
        run_matrix([first, second], duration=1.0)


def test_iter035_drone_passes_through_gate_centers_vertically(prepared_matrix_result):
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

    This test pins the win across all 7 tracks. The mean remains tight;
    the v4 fully-wired planner allows up to 550mm instantaneous lag on the
    steep grand_tour climb while still catching the old systematic offset.
    """
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    cfg_dir = repo / "sim_pybullet" / "configs"

    violations: list[str] = []
    for cfg_path in sorted(cfg_dir.glob("*.json")):
        track = cfg_path.stem
        with open(cfg_path) as f:
            cfg = json.load(f)
        r = prepared_matrix_result["results"][track]
        trace = r.get("position_trace")
        gates_by_id = {g["id"]: g for g in cfg["gates"]}
        expected_gate_ids = list(gates_by_id)
        gate_pass_times = r.get("gate_pass_times")
        if not isinstance(trace, list) or not trace:
            violations.append(f"{track}: position trace evidence is missing")
            continue
        if not isinstance(gate_pass_times, list):
            violations.append(f"{track}: gate-pass timing evidence is missing")
            continue
        observed_gate_ids = [
            item.get("gate_id") if isinstance(item, dict) else None
            for item in gate_pass_times
        ]
        if observed_gate_ids != expected_gate_ids:
            violations.append(
                f"{track}: gate-pass evidence IDs {observed_gate_ids!r} do not "
                f"match configured gates {expected_gate_ids!r}"
            )
            continue
        z_errors: list[float] = []
        for gpt in gate_pass_times:
            gid = gpt["gate_id"]
            pass_time = gpt.get("time_s")
            if (
                isinstance(pass_time, bool)
                or not isinstance(pass_time, (int, float))
                or not math.isfinite(float(pass_time))
            ):
                violations.append(
                    f"{track}: invalid gate-pass time for {gid}: {pass_time!r}"
                )
                z_errors = []
                break
            sample = min(trace, key=lambda s: abs(s["t"] - pass_time))
            gz = gates_by_id[gid]["pose"]["z"]
            z_errors.append(sample["pos"][2] - gz)
        if not z_errors:
            continue
        mean_abs = sum(abs(z) for z in z_errors) / len(z_errors)
        max_abs = max(abs(z) for z in z_errors)
        # Mean: tight (the bug had mean -0.35m on straight_hairpin).
        if mean_abs > 0.20:
            violations.append(
                f"{track}: mean |Δz|={mean_abs:.3f}m > 200mm "
                f"(per-gate z-errors: {[f'{z:+.3f}' for z in z_errors]})"
            )
        # Max: looser — legitimate tracker-lag on steep climbs.
        if max_abs > 0.55:
            violations.append(
                f"{track}: max |Δz|={max_abs:.3f}m > 550mm"
            )
    assert not violations, (
        "iter-035 gate-altitude regression:\n  " + "\n  ".join(violations)
    )


def test_iter032_accel_projection_drops_clamp_engagement(prepared_matrix_result):
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
    matrix = prepared_matrix_result
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
