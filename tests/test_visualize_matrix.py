"""
Iter-033: smoke tests for the matrix visualizer.

The visualizer is a thin wrapper around `run_synthetic_benchmark(...,
record_position_trace=True)` + matplotlib animation. These tests pin
the data contract (position_trace shape, fields, monotonic time) so a
future bench refactor doesn't silently break the visualizer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark import run_synthetic_benchmark

_REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_benchmark_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGP_CACHE_ROOT", str(tmp_path / "artifacts"))


@pytest.mark.benchmark
def test_position_trace_off_by_default():
    """Default benchmark run must NOT populate `position_trace` — kept
    None so the result dict stays small for matrix tests."""
    cfg_path = _REPO / "sim_pybullet" / "configs" / "race_01.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    r = run_synthetic_benchmark(duration=1.0, config=cfg)
    assert r.get("position_trace") is None


@pytest.mark.benchmark
def test_position_trace_populated_when_enabled():
    """With `record_position_trace=True`, every sim step appends a
    dict with t/pos/vel/yaw/tracking_err_m. Monotonic time."""
    cfg_path = _REPO / "sim_pybullet" / "configs" / "race_01.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    r = run_synthetic_benchmark(
        duration=2.0, config=cfg, record_position_trace=True,
    )
    trace = r["position_trace"]
    assert isinstance(trace, list)
    assert len(trace) > 50  # 2s at 100Hz minus startup
    assert all({"t", "pos", "vel", "yaw", "tracking_err_m"} <= set(p) for p in trace)
    assert all(len(p["pos"]) == 3 for p in trace)
    assert all(len(p["vel"]) == 3 for p in trace)
    # Monotonic time.
    times = [p["t"] for p in trace]
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))


def test_visualizer_resolves_track_names():
    """The CLI helper must list real tracks and reject typos."""
    import scripts.visualize_matrix as vm
    available = [p.stem for p in vm._list_configs()]
    assert "race_01" in available
    # Real track resolves.
    cfg = vm._resolve_config("race_01")
    assert cfg.exists()
    # Typo raises SystemExit (not silent KeyError).
    with pytest.raises(SystemExit):
        vm._resolve_config("not_a_real_track_xyz")
