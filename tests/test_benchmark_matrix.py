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


def test_run_matrix_empty_configs_returns_empty_tracks():
    matrix = run_matrix([], duration=1.0)
    assert matrix["tracks"] == {}
    # No tracks means nothing failed, so all_passed is vacuously True.
    assert matrix["all_passed"] is True
