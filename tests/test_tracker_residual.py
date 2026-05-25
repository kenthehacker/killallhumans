"""
Tests for the learned tracker residual (iter-001 A14).

These pin the safety contract:
  - The MLP forward pass produces the right shape for any 10-dim input.
  - Even a model whose raw output is huge (10, 10, 10) gets clamped at the
    consumer to ±residual_clamp_rad / ±residual_thrust_clamp.
  - With `use_residual=False`, the tracker output is byte-identical to the
    pre-A15 baseline.
  - With `use_residual=True` and zero-init weights, the residual is zero
    so behaviour matches baseline within float-epsilon (an extra branch
    runs but it adds zero).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from control.learned_residual import (
    DEFAULT_N_HIDDEN,
    DEFAULT_N_INPUTS,
    DEFAULT_N_OUTPUTS,
    TrackerResidualMLP,
    build_input_features,
)
from control.mpc_tracker import GeometricTracker, TrackerConfig
from planning.trajectory_optimizer import TrajectoryPoint


def _hover_ref() -> TrajectoryPoint:
    return TrajectoryPoint(
        time=0.0,
        position=(0.0, 0.0, -2.0),
        velocity=(0.0, 0.0, 0.0),
        acceleration=(0.0, 0.0, 0.0),
        jerk=(0.0, 0.0, 0.0),
        yaw=0.0,
        yaw_rate=0.0,
    )


# ---------------------------------------------------------------------------
# Iter-014 — feature-trace hook (ML training infrastructure)
# ---------------------------------------------------------------------------

def test_feature_trace_empty_by_default():
    """TrackerConfig.trace_features=False (the default) must produce
    an empty trace, even after many track() calls. No-overhead contract
    for production callers."""
    tracker = GeometricTracker(TrackerConfig())
    ref = _hover_ref()
    for _ in range(50):
        tracker.track(
            current_position=(0.1, 0.0, -1.9),
            current_velocity=(0.0, 0.0, 0.0),
            current_yaw=0.0,
            reference=ref,
        )
    assert tracker.feature_trace == []


def test_feature_trace_captures_when_enabled():
    """Iter-014: with trace_features=True, every track() call appends
    (features_10d, roll_nom, pitch_nom, thrust_nom, pos_err, vel_err)."""
    tracker = GeometricTracker(TrackerConfig(trace_features=True))
    ref = _hover_ref()
    for _ in range(7):
        tracker.track(
            current_position=(0.2, 0.1, -1.8),
            current_velocity=(0.0, 0.0, 0.0),
            current_yaw=0.0,
            reference=ref,
        )
    assert len(tracker.feature_trace) == 7
    for entry in tracker.feature_trace:
        features, roll, pitch, thrust, pos_err, vel_err = entry
        assert features.shape == (DEFAULT_N_INPUTS,)
        assert np.all(np.isfinite(features))
        assert -math.pi <= roll <= math.pi
        assert -math.pi <= pitch <= math.pi
        assert 0.0 <= thrust <= 1.5  # normalized; near ~1.0 at hover
        assert len(pos_err) == 3
        assert len(vel_err) == 3


def test_feature_trace_save_load_roundtrip(tmp_path):
    """Iter-015: save_feature_trace + load_feature_trace must round-trip
    the trace exactly. Crucial for future ML training pipelines: the
    collected dataset must be persistable across processes."""
    from control.learned_residual import (
        save_feature_trace,
        load_feature_trace,
    )
    tracker = GeometricTracker(TrackerConfig(trace_features=True))
    ref = _hover_ref()
    for i in range(5):
        tracker.track(
            current_position=(0.1 * i, 0.0, -1.8),
            current_velocity=(0.0, 0.0, 0.0),
            current_yaw=0.0,
            reference=ref,
        )
    path = tmp_path / "trace.npz"
    save_feature_trace(tracker.feature_trace, path)
    loaded = load_feature_trace(path)
    assert loaded["features"].shape == (5, DEFAULT_N_INPUTS)
    assert loaded["roll_nom"].shape == (5,)
    assert loaded["pitch_nom"].shape == (5,)
    assert loaded["thrust_nom"].shape == (5,)
    assert loaded["pos_err"].shape == (5, 3)
    assert loaded["vel_err"].shape == (5, 3)
    # Round-trip integrity: same values
    for i, (feats, r, p, t, pe, ve) in enumerate(tracker.feature_trace):
        np.testing.assert_array_equal(loaded["features"][i], feats)
        assert loaded["roll_nom"][i] == r
        assert loaded["pitch_nom"][i] == p
        assert loaded["thrust_nom"][i] == t
        np.testing.assert_array_equal(loaded["pos_err"][i], pe)
        np.testing.assert_array_equal(loaded["vel_err"][i], ve)


def test_save_feature_trace_rejects_empty(tmp_path):
    """Empty trace should raise — saving zero rows wastes disk and
    breaks the load contract."""
    from control.learned_residual import save_feature_trace
    import pytest as _pytest
    with _pytest.raises(ValueError, match="empty"):
        save_feature_trace([], tmp_path / "x.npz")


def test_bench_exposes_tracker_feature_trace_when_enabled():
    """Iter-024: the synthetic matrix bench exposes the GeometricTracker's
    feature_trace in the result dict so external scripts can collect
    real ML training data. With trace_features=True, the trace has
    samples in the canonical 10-dim format; with the flag off the list
    is empty (no overhead for production callers)."""
    import json
    from pathlib import Path
    from scripts.benchmark import run_synthetic_benchmark

    cfg_path = Path(__file__).resolve().parent.parent / "sim_pybullet" / "configs" / "race_01.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Default path: empty trace.
    r_off = run_synthetic_benchmark(duration=1.0, config=cfg)
    assert r_off.get("tracker_feature_trace") == []

    # Enabled path: nonempty trace with canonical 10-dim features.
    r_on = run_synthetic_benchmark(
        duration=1.0, config=cfg,
        tracker_config_overrides={"trace_features": True},
    )
    trace = r_on.get("tracker_feature_trace")
    assert isinstance(trace, list)
    assert len(trace) > 50  # 1s at 100Hz minus startup; should be ~100
    features, roll, pitch, thrust, pos_err, vel_err = trace[0]
    assert features.shape == (DEFAULT_N_INPUTS,)
    assert np.all(np.isfinite(features))
    assert -math.pi <= roll <= math.pi
    assert -math.pi <= pitch <= math.pi
    assert 0.0 <= thrust <= 1.5
    assert len(pos_err) == 3
    assert len(vel_err) == 3


def test_init_residual_weights_script_writes_loadable_zero_npz(tmp_path):
    """Iter-022 connect-the-dots: the init_residual_weights.py script
    must produce a .npz file that `TrackerResidualMLP.from_npz` can
    load AND whose forward pass is zero. This is the round-trip
    contract that lets the pipeline test
    `use_residual=True + this_file` ≡ baseline.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    out = tmp_path / "weights.npz"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "init_residual_weights.py"),
            "--out", str(out),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"script failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert out.exists()
    # Load + verify zero behavior.
    mlp = TrackerResidualMLP.from_npz(out)
    for sample in (
        np.zeros(DEFAULT_N_INPUTS),
        np.ones(DEFAULT_N_INPUTS),
        np.linspace(-3.0, 3.0, DEFAULT_N_INPUTS),
    ):
        y = mlp.forward(sample)
        assert y.shape == (DEFAULT_N_OUTPUTS,)
        np.testing.assert_array_equal(y, np.zeros(DEFAULT_N_OUTPUTS))


# ---------------------------------------------------------------------------
# MLP shape + sanity
# ---------------------------------------------------------------------------

def test_zero_init_forward_is_zero():
    mlp = TrackerResidualMLP.zero_init()
    x = np.linspace(-1.0, 1.0, DEFAULT_N_INPUTS)
    y = mlp.forward(x)
    assert y.shape == (DEFAULT_N_OUTPUTS,)
    assert np.allclose(y, 0.0)


def test_random_init_forward_shape():
    mlp = TrackerResidualMLP.random_init(seed=7)
    y = mlp.forward(np.zeros(DEFAULT_N_INPUTS))
    assert y.shape == (DEFAULT_N_OUTPUTS,)


def test_input_dim_mismatch_raises():
    mlp = TrackerResidualMLP.zero_init()
    with pytest.raises(ValueError):
        mlp.forward(np.zeros(DEFAULT_N_INPUTS - 1))


def test_build_input_features_is_10_dim_and_ordered():
    feats = build_input_features(
        pos_err=np.array([1.0, 2.0, 3.0]),
        vel_err=np.array([4.0, 5.0, 6.0]),
        ref_accel=np.array([7.0, 8.0, 9.0]),
        thrust_normalized=0.5,
    )
    assert feats.shape == (DEFAULT_N_INPUTS,)
    assert list(feats) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.5]


def test_npz_roundtrip(tmp_path: Path):
    mlp = TrackerResidualMLP.random_init(seed=42)
    path = tmp_path / "rt.npz"
    mlp.to_npz(path)
    reloaded = TrackerResidualMLP.from_npz(path)
    assert np.allclose(mlp.W1, reloaded.W1)
    assert np.allclose(mlp.b1, reloaded.b1)
    assert np.allclose(mlp.W2, reloaded.W2)
    assert np.allclose(mlp.b2, reloaded.b2)


def test_no_torch_at_import_time():
    """The control loop must not pay a torch import tax at runtime."""
    import sys
    import importlib

    # Pretend torch is unavailable; the residual module must still import.
    saved = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        # Reload to force re-evaluation of any conditional imports.
        if "control.learned_residual" in sys.modules:
            importlib.reload(sys.modules["control.learned_residual"])
        else:
            import control.learned_residual  # noqa: F401
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
        else:
            sys.modules.pop("torch", None)


# ---------------------------------------------------------------------------
# Hard clamp at the consumer
# ---------------------------------------------------------------------------

def test_residual_output_clamped_at_consumer(tmp_path: Path):
    """Even if the model outputs 10.0 on every channel, the consumer must
    clip the delta to ±residual_clamp_rad / ±residual_thrust_clamp."""
    # Build a model whose forward pass returns a constant huge value.
    # Easiest construction: zero hidden weights but a large b2.
    mlp = TrackerResidualMLP.zero_init()
    mlp.b2 = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    npz_path = tmp_path / "huge.npz"
    mlp.to_npz(npz_path)

    cfg_residual = TrackerConfig(
        use_residual=True,
        residual_weights_path=str(npz_path),
        residual_clamp_rad=0.05,
        residual_thrust_clamp=0.05,
    )
    cfg_baseline = TrackerConfig(use_residual=False)
    tracker_res = GeometricTracker(cfg_residual)
    tracker_base = GeometricTracker(cfg_baseline)

    pos = (0.0, 0.0, -2.0)
    vel = (0.0, 0.0, 0.0)
    yaw = 0.0
    ref = _hover_ref()

    cmd_res = tracker_res.track(pos, vel, yaw, ref)
    cmd_base = tracker_base.track(pos, vel, yaw, ref)

    # Delta must not exceed the clamp magnitude.
    assert abs(cmd_res.roll_rad - cmd_base.roll_rad) <= 0.05 + 1e-9
    assert abs(cmd_res.pitch_rad - cmd_base.pitch_rad) <= 0.05 + 1e-9
    assert abs(cmd_res.thrust - cmd_base.thrust) <= 0.05 + 1e-9
    # And tilt limits remain enforced.
    assert abs(cmd_res.roll_rad) <= cfg_residual.max_tilt_rad
    assert abs(cmd_res.pitch_rad) <= cfg_residual.max_tilt_rad
    assert cfg_residual.min_thrust_normalized <= cmd_res.thrust <= cfg_residual.max_thrust_normalized


# ---------------------------------------------------------------------------
# Off-by-default safety
# ---------------------------------------------------------------------------

def test_residual_off_is_byte_identical_to_baseline():
    """`use_residual=False` (the default) must give the same command as
    a tracker built with no config tweaks at all."""
    cfg_default = TrackerConfig()
    cfg_explicit_off = TrackerConfig(use_residual=False)
    t1 = GeometricTracker(cfg_default)
    t2 = GeometricTracker(cfg_explicit_off)

    pos = (1.0, 0.5, -2.5)
    vel = (0.5, -0.2, 0.0)
    yaw = 0.3
    ref = TrajectoryPoint(
        time=0.0,
        position=(2.0, 0.0, -2.0),
        velocity=(1.0, 0.0, 0.0),
        acceleration=(0.5, 0.0, 0.0),
        jerk=(0.0, 0.0, 0.0),
        yaw=0.0,
        yaw_rate=0.0,
    )

    c1 = t1.track(pos, vel, yaw, ref)
    c2 = t2.track(pos, vel, yaw, ref)
    assert c1.roll_rad == c2.roll_rad
    assert c1.pitch_rad == c2.pitch_rad
    assert c1.yaw_rad == c2.yaw_rad
    assert c1.thrust == c2.thrust


def test_residual_on_with_zero_init_matches_baseline_within_float_epsilon():
    """Turn the feature on with no trained weights -> the MLP outputs zero,
    so the tracker output matches baseline to within numerical noise."""
    cfg_off = TrackerConfig(use_residual=False)
    cfg_on_zero = TrackerConfig(use_residual=True, residual_weights_path=None)
    t_off = GeometricTracker(cfg_off)
    t_on = GeometricTracker(cfg_on_zero)

    pos = (1.0, 0.5, -2.5)
    vel = (0.5, -0.2, 0.0)
    yaw = 0.3
    ref = TrajectoryPoint(
        time=0.0,
        position=(2.0, 0.0, -2.0),
        velocity=(1.0, 0.0, 0.0),
        acceleration=(0.5, 0.0, 0.0),
        jerk=(0.0, 0.0, 0.0),
        yaw=0.0,
        yaw_rate=0.0,
    )

    c_off = t_off.track(pos, vel, yaw, ref)
    c_on = t_on.track(pos, vel, yaw, ref)
    # Should agree to ~1e-12 — the residual adds exactly 0.0, so any
    # difference comes from numpy float reassociation in the branch.
    assert abs(c_on.roll_rad - c_off.roll_rad) < 1e-9
    assert abs(c_on.pitch_rad - c_off.pitch_rad) < 1e-9
    assert abs(c_on.thrust - c_off.thrust) < 1e-9
