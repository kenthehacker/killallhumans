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
    OnlineDroneCalibrator,
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


# ===========================================================================
# Online recursive estimator (roadmap #1) — OnlineDroneCalibrator
#
# Same NED physics as the batch path (g − a_z = k_t·u + k_d·v_z); these tests
# verify the streaming RLS recovers seeded params, CONVERGES under streaming
# data, TRACKS a drift via the forgetting factor, agrees with the batch
# lstsq, and keeps the batch path's positivity guard. Real DCL telemetry
# validation stays deferred to iter ≥ 002 (no DCL binary on this worktree).
# ===========================================================================


def test_online_recovers_seeded_ratios_within_10_percent():
    """Streaming the same synthetic ticks recovers the seeded ratios."""
    k_t_true, k_d_true = 22.0, 0.40
    cal = OnlineDroneCalibrator()
    res = None
    for s in _synth_samples(k_t_true, k_d_true, n=400):
        res = cal.update(s)
    assert res.thrust_per_mass > 0, "thrust_per_mass must be positive (upward thrust)"
    assert abs(res.thrust_per_mass - k_t_true) / k_t_true < 0.10
    assert abs(res.drag_per_mass - k_d_true) / k_d_true < 0.10
    assert res.n_samples == 400


def test_online_converges_under_streaming_data():
    """Error must DECREASE monotonically-in-trend as ticks accumulate.

    Compare the parameter error after the first 25 ticks vs after all 400:
    a working recursive estimator is far closer once it has seen the stream.
    """
    k_t_true, k_d_true = 22.0, 0.40
    samples = _synth_samples(k_t_true, k_d_true, n=400, seed=3)
    cal = OnlineDroneCalibrator(covariance_init=1.0e3)

    truth = np.array([k_t_true, k_d_true])
    err_early = None
    for i, s in enumerate(samples):
        cal.update(s)
        if i == 24:  # after 25 ticks
            err_early = np.linalg.norm(
                np.array([cal.thrust_per_mass, cal.drag_per_mass]) - truth
            )
    err_final = np.linalg.norm(
        np.array([cal.thrust_per_mass, cal.drag_per_mass]) - truth
    )
    assert err_early is not None
    # Converged estimate is markedly better than the early transient, and the
    # final error is small in absolute terms.
    assert err_final < 0.5 * err_early, (
        f"online estimate did not converge: early err {err_early:.3f}, "
        f"final err {err_final:.3f}"
    )
    assert err_final < 0.5, f"final parameter error too large: {err_final:.3f}"
    # The forgetting-weighted a-priori RMSE should settle near the injected
    # accelerometer noise floor (~0.05 m/s²), not blow up.
    assert cal.rmse < 0.5, f"online rmse too high: {cal.rmse:.3f}"


def test_online_matches_batch_lstsq():
    """Fed identical ticks, the streaming RLS and the batch lstsq agree.

    With NO forgetting (λ=1) RLS is exactly recursive least squares, so on
    the same data it converges to the batch normal-equation solution. Tiny
    tolerance confirms they solve the SAME regression (not merely similar).
    """
    k_t_true, k_d_true = 18.5, 0.62
    samples = _synth_samples(k_t_true, k_d_true, n=300, seed=5)
    batch = DroneCalibrator().identify_thrust_drag_ratios(samples)
    cal = OnlineDroneCalibrator(
        forgetting_factor=1.0, covariance_init=1.0e6, max_covariance=1.0e9,
    )
    res = None
    for s in samples:
        res = cal.update(s)
    assert res.thrust_per_mass == pytest.approx(batch.thrust_per_mass, rel=1e-3, abs=1e-3)
    assert res.drag_per_mass == pytest.approx(batch.drag_per_mass, rel=1e-3, abs=1e-3)


def test_online_forgetting_factor_tracks_drift():
    """A low forgetting factor must TRACK a step change in k_t.

    Stream 300 ticks at k_t=22, then 300 at k_t=30. A forgetting estimator
    (λ<1) re-converges to the new value; a growing-memory estimator (λ=1)
    stays biased toward the stale average. We assert the forgetting estimator
    lands near 30 AND is closer to it than the no-forgetting one — i.e. the
    forgetting factor is doing the drift-tracking, not just luck.
    """
    k_d_true = 0.40
    seg1 = _synth_samples(22.0, k_d_true, n=300, seed=1)
    seg2 = _synth_samples(30.0, k_d_true, n=300, seed=2)

    forget = OnlineDroneCalibrator(forgetting_factor=0.97, covariance_init=1.0e3)
    noforget = OnlineDroneCalibrator(forgetting_factor=1.0, covariance_init=1.0e3)
    for s in seg1 + seg2:
        forget.update(s)
        noforget.update(s)

    # The forgetting estimator tracks the post-step value closely.
    assert abs(forget.thrust_per_mass - 30.0) / 30.0 < 0.05, (
        f"forgetting estimator failed to track drift: k_t={forget.thrust_per_mass:.2f}"
    )
    # And it is strictly closer to the new value than the stale-averaging one.
    assert abs(forget.thrust_per_mass - 30.0) < abs(noforget.thrust_per_mass - 30.0)


def test_online_positivity_guard_on_reported_estimate():
    """The reported thrust_per_mass is floored positive (batch-path guard).

    Before any data, the zero-initialised estimate would report k_t=0; the
    positivity floor keeps the surfaced value > 0 so downstream consumers
    (e.g. thrust normalisation) never divide by / configure a non-physical
    zero/negative thrust. drag_per_mass is NOT floored (drag may be ~0).
    """
    cal = OnlineDroneCalibrator(k_t_init=0.0, k_d_init=0.0, min_k_t=1e-6)
    res = cal.result()
    assert res.thrust_per_mass >= 1e-6
    assert res.drag_per_mass == 0.0
    assert res.n_samples == 0


def test_online_update_from_telemetry_hook():
    """The live-MAVLink hook feeds the same fit without a CalibrationSample."""
    k_t_true, k_d_true = 22.0, 0.40
    cal = OnlineDroneCalibrator()
    res = None
    for s in _synth_samples(k_t_true, k_d_true, n=300):
        res = cal.update_from_telemetry(
            thrust_normalized=s.thrust_normalized,
            velocity_z_world=s.velocity_z_world,
            accel_z_world=s.accel_z_world,
        )
    assert abs(res.thrust_per_mass - k_t_true) / k_t_true < 0.10
    assert abs(res.drag_per_mass - k_d_true) / k_d_true < 0.10


def test_online_stable_under_low_excitation():
    """Covariance-windup guard: a near-constant (under-excited) stream with
    forgetting must NOT blow the estimator up to NaN/inf.

    This is the report's flagged risk — a smooth racing line under-excites a
    parameter, and with λ<1 the covariance otherwise inflates without bound.
    Stream thousands of near-identical ticks and assert the estimate stays
    finite and bounded.
    """
    cal = OnlineDroneCalibrator(forgetting_factor=0.99, max_covariance=1.0e6)
    s = CalibrationSample(thrust_normalized=0.45, velocity_z_world=0.0,
                          accel_z_world=9.81 - 22.0 * 0.45)
    res = None
    for _ in range(20000):
        res = cal.update(s)
    assert np.isfinite(res.thrust_per_mass)
    assert np.isfinite(res.drag_per_mass)
    assert np.isfinite(res.rmse)
    # tr(P) is held at/under the cap rather than overflowing.
    assert float(cal._P[0, 0] + cal._P[1, 1]) <= cal.max_covariance * (1 + 1e-9)
    assert res.thrust_per_mass > 0


def test_online_rejects_bad_forgetting_factor():
    with pytest.raises(ValueError):
        OnlineDroneCalibrator(forgetting_factor=0.0)
    with pytest.raises(ValueError):
        OnlineDroneCalibrator(forgetting_factor=1.5)
    with pytest.raises(ValueError):
        OnlineDroneCalibrator(covariance_init=0.0)
    with pytest.raises(ValueError):
        OnlineDroneCalibrator(covariance_init=1.0e3, max_covariance=1.0e3)
