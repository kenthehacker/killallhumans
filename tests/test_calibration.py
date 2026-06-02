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
    """Generate `n` physically-correct samples obeying `a_z = g − k_t·u − k_d·v_z`.

    NED convention: z points DOWN, so gravity = +g along +z. Thrust opposes
    gravity (pushes UP, i.e., negative z), so positive normalised thrust
    `u` produces negative z-acceleration. At hover (u ≈ g/k_t, v_z = 0),
    a_z ≈ 0 — the IMU reads ~0 because thrust balances gravity.

    Iter-001 review (Opus F1) caught that the original synth fed the
    `code's` wrong equation back into the regression, masking a sign bug.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.3, 0.9, size=n)
    v = rng.uniform(-3.0, 3.0, size=n)
    a = gravity - k_t * u - k_d * v
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
    # Iter-001 review (Opus F1): positivity guard catches sign-flip bugs.
    assert cal.thrust_per_mass > 0, "thrust_per_mass must be positive (upward thrust)"
    assert abs(cal.thrust_per_mass - k_t_true) / k_t_true < 0.10
    assert abs(cal.drag_per_mass - k_d_true) / k_d_true < 0.10


def test_sign_flipped_synth_now_rejected_via_rmse_guard():
    """Iter-002 review M4: a sign-flipped feeder used to escape the
    positivity guard with a large positive k_t (bias absorbed). Now the
    RMSE/|y| ratio catches the gross misfit.
    """
    k_t_true, k_d_true = 22.0, 0.40
    rng = np.random.default_rng(7)
    u = rng.uniform(0.3, 0.9, size=200)
    v = rng.uniform(-3.0, 3.0, size=200)
    # Sign-flipped physics (matches the OLD buggy synth formula).
    a = -k_t_true * u - k_d_true * v - 9.81
    samples = [
        CalibrationSample(
            thrust_normalized=float(u_i),
            velocity_z_world=float(v_i),
            accel_z_world=float(a_i),
        )
        for u_i, v_i, a_i in zip(u, v, a)
    ]
    # Should raise (either positivity guard or the new RMSE guard).
    with pytest.raises(ValueError):
        DroneCalibrator().identify_thrust_drag_ratios(samples)


def test_hover_only_samples_recover_g_over_u():
    """At a steady hover, thrust must balance gravity: k_t·u = g, so k_t = g/u.

    This is a physics-anchored test that would have caught the Opus-F1 bug
    (the old code's regression returned k_t ≈ −22 at hover instead of +21.8).
    """
    g = 9.81
    u_hover = 0.45  # 45% throttle, representative
    # 50 samples all at hover ± small noise.
    rng = np.random.default_rng(11)
    samples = [
        CalibrationSample(
            thrust_normalized=float(u_hover + rng.normal(0.0, 0.01)),
            velocity_z_world=float(rng.normal(0.0, 0.02)),
            accel_z_world=float(rng.normal(0.0, 0.05)),
        )
        for _ in range(50)
    ]
    cal = DroneCalibrator(gravity=g).identify_thrust_drag_ratios(samples)
    # Expected k_t ≈ g / u_hover ≈ 21.8 with NEW (correct) physics.
    # Under the OLD buggy code, k_t would be ≈ -21.8 (sign-flipped).
    assert cal.thrust_per_mass > 0, (
        f"hover thrust_per_mass must be positive; got {cal.thrust_per_mass:.2f}"
    )
    expected = g / u_hover
    assert abs(cal.thrust_per_mass - expected) / expected < 0.15, (
        f"hover fit deviates >15% from g/u: expected ≈ {expected:.2f}, "
        f"got {cal.thrust_per_mass:.2f}"
    )


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


# ---------------------------------------------------------------------------
# Iter-003 (4/7 reviews MINOR): JSON schema validation
# ---------------------------------------------------------------------------

def test_read_calibration_json_rejects_missing_required_key(tmp_path: Path):
    import json as _json
    path = tmp_path / "bad.json"
    path.write_text(_json.dumps({"drag_per_mass_1_per_s": 0.4}))  # missing k_t
    with pytest.raises(ValueError, match="missing"):
        DroneCalibrator.read_calibration_json(path)


def test_read_calibration_json_rejects_negative_thrust(tmp_path: Path):
    import json as _json
    path = tmp_path / "neg.json"
    path.write_text(_json.dumps({
        "thrust_per_mass_1_per_s2": -22.0,  # impossible
        "drag_per_mass_1_per_s": 0.4,
        "n_samples": 200,
        "rmse_mps2": 0.05,
    }))
    with pytest.raises(ValueError, match="non-positive thrust"):
        DroneCalibrator.read_calibration_json(path)


def test_read_calibration_json_rejects_nan(tmp_path: Path):
    import json as _json
    path = tmp_path / "nan.json"
    # JSON doesn't have NaN per RFC8259, but Python's json module accepts it
    # when serialising; we test by writing the literal text.
    path.write_text(
        '{"thrust_per_mass_1_per_s2": NaN, "drag_per_mass_1_per_s": 0.4, '
        '"n_samples": 200, "rmse_mps2": 0.05}'
    )
    with pytest.raises(ValueError, match="non-finite"):
        DroneCalibrator.read_calibration_json(path)


def test_read_calibration_json_rejects_zero_samples(tmp_path: Path):
    import json as _json
    path = tmp_path / "zero.json"
    path.write_text(_json.dumps({
        "thrust_per_mass_1_per_s2": 22.0,
        "drag_per_mass_1_per_s": 0.4,
        "n_samples": 0,
        "rmse_mps2": 0.05,
    }))
    with pytest.raises(ValueError, match="n_samples"):
        DroneCalibrator.read_calibration_json(path)
