"""
Drone-calibration identifier tests (iter-001 A13).

Verifies that `DroneCalibrator.identify_thrust_drag_ratios` recovers the
seeded thrust-per-mass and drag-per-mass ratios from synthetic samples
within ±10%. Real DCL telemetry validation is deferred to iter ≥ 002 —
no DCL binary on this worktree to drive it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from competition.calibration import (
    CalibrationSample,
    DroneCalibrator,
)


def _synth_samples(
    k_t: float, k_d: float, gravity: float = 9.81,
    n: int = 200, seed: int = 7,
):
    """Generate `n` samples that obey `a_z + g = -k_t·u - k_d·v_z`.

    `u` and `v_z` are sampled iid uniform; `a_z` is computed exactly per
    the model. With zero noise the fit should be near-perfect; we add
    small Gaussian noise to make the test exercise the least-squares
    residual surface.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.3, 0.9, size=n)
    v = rng.uniform(-3.0, 3.0, size=n)
    a = -k_t * u - k_d * v - gravity
    # Small accelerometer noise (~0.05 m/s²).
    a += rng.normal(0.0, 0.05, size=n)
    return [
        CalibrationSample(
            thrust_normalized=float(u_i),
            velocity_z_world=float(v_i),
            accel_z_world=float(a_i),
        )
        for u_i, v_i, a_i in zip(u, v, a)
    ]


def test_recovers_seeded_ratios_within_10_percent():
    k_t_true, k_d_true = 22.0, 0.40    # representative AIGP-class values
    cal = DroneCalibrator().identify_thrust_drag_ratios(
        _synth_samples(k_t_true, k_d_true)
    )
    assert abs(cal.thrust_per_mass - k_t_true) / k_t_true < 0.10
    assert abs(cal.drag_per_mass - k_d_true) / k_d_true < 0.10


def test_assumed_mass_yields_physical_values():
    k_t_true, k_d_true = 22.0, 0.40
    assumed_mass = 1.5
    cal = DroneCalibrator().identify_thrust_drag_ratios(
        _synth_samples(k_t_true, k_d_true), assumed_mass_kg=assumed_mass,
    )
    assert cal.mass_kg == pytest.approx(assumed_mass)
    # max_thrust_n = k_t * mass; should be ~33 N for these inputs.
    assert cal.max_thrust_n is not None
    assert abs(cal.max_thrust_n - k_t_true * assumed_mass) / (k_t_true * assumed_mass) < 0.10
    assert cal.drag_coefficient is not None


def test_insufficient_samples_raises():
    with pytest.raises(ValueError):
        DroneCalibrator().identify_thrust_drag_ratios([])
    with pytest.raises(ValueError):
        DroneCalibrator().identify_thrust_drag_ratios([
            CalibrationSample(0.5, 0.0, -9.81)
        ])


def test_json_roundtrip(tmp_path: Path):
    samples = _synth_samples(22.0, 0.40)
    cal = DroneCalibrator().identify_thrust_drag_ratios(
        samples, assumed_mass_kg=1.5,
    )
    path = tmp_path / "drone_calibration.json"
    DroneCalibrator.write_calibration_json(cal, path)
    assert path.exists()
    reloaded = DroneCalibrator.read_calibration_json(path)
    assert reloaded.thrust_per_mass == pytest.approx(cal.thrust_per_mass)
    assert reloaded.drag_per_mass == pytest.approx(cal.drag_per_mass)
    assert reloaded.mass_kg == pytest.approx(cal.mass_kg)
    assert reloaded.max_thrust_n == pytest.approx(cal.max_thrust_n)
