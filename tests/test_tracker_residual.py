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
