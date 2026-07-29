"""Unit tests for the VQ2 HIGHRES_IMU-only attitude estimator."""

import math

import pytest

from competition.adapter import IMUData, Quaternion
from estimation.imu_attitude import ImuAttitudeConfig, ImuAttitudeEstimator


G = 9.80665


def _fast_config(**overrides) -> ImuAttitudeConfig:
    values = {
        "calibration_min_samples": 5,
        "calibration_min_duration_s": 0.04,
    }
    values.update(overrides)
    return ImuAttitudeConfig(**values)


def _bootstrap(
    estimator: ImuAttitudeEstimator,
    *,
    accel=(0.0, 0.0, -G),
    gyro=(0.0, 0.0, 0.0),
    start_us=0,
):
    estimate = None
    for i in range(5):
        estimate = estimator.update(start_us + i * 10_000, accel, gyro)
    assert estimator.is_ready
    assert estimate is not None
    return estimate


def _stationary_accel(roll: float, pitch: float):
    """FRD specific force for a stationary body at the given tilt."""

    return (
        G * math.sin(pitch),
        -G * math.sin(roll) * math.cos(pitch),
        -G * math.cos(roll) * math.cos(pitch),
    )


def test_bootstrap_uses_observed_tilt_and_calibrates_gyro_bias():
    # Reproduces the observed v3385 spawn vector: it is about -18 degrees
    # pitch, not level, and must not be flattened to identity.
    accel = (-2.9992, -0.0023, -9.3403)
    bias = (0.011, -0.007, 0.004)
    estimator = ImuAttitudeEstimator(_fast_config(), initial_yaw_rad=0.37)

    estimate = _bootstrap(estimator, accel=accel, gyro=bias)

    expected_roll = math.atan2(-accel[1], -accel[2])
    expected_pitch = math.atan2(accel[0], math.hypot(accel[1], accel[2]))
    assert estimate.roll == pytest.approx(expected_roll, abs=1e-8)
    assert estimate.pitch == pytest.approx(expected_pitch, abs=1e-8)
    assert estimate.yaw == pytest.approx(0.37, abs=1e-8)
    assert estimate.gyro_bias == pytest.approx(bias, abs=1e-12)
    assert estimate.body_rates == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert not estimate.yaw_observable
    assert estimator.calibration_progress == 1.0


def test_update_imu_accepts_repository_imu_data_contract():
    estimator = ImuAttitudeEstimator(_fast_config())
    estimate = None
    for i in range(5):
        estimate = estimator.update_imu(
            IMUData(
                timestamp_us=i * 10_000,
                accel=(0.0, 0.0, -G),
                gyro=(0.0, 0.0, 0.0),
            )
        )
    assert estimate is not None and estimate.healthy


def test_bootstrap_waits_for_stationary_window_and_rejects_wrong_accel_sign():
    estimator = ImuAttitudeEstimator(_fast_config())

    # Correct magnitude but inverted/upside-down is unsafe at pad bootstrap.
    for i in range(10):
        assert estimator.update(i * 10_000, (0.0, 0.0, G), (0.0, 0.0, 0.0)) is None
    assert not estimator.is_ready
    assert estimator.calibration_progress == 0.0

    # A moving sample breaks the otherwise valid stationary window.
    for i in range(4):
        estimator.update(100_000 + i * 10_000, (0.0, 0.0, -G), (0.0, 0.0, 0.0))
    estimator.update(140_000, (0.0, 0.0, -G), (0.3, 0.0, 0.0))
    assert estimator.calibration_progress == 0.0

    estimate = _bootstrap(estimator, start_us=150_000)
    assert estimate.healthy


def test_bootstrap_rejects_stationary_but_excessively_tilted_vehicle():
    estimator = ImuAttitudeEstimator(_fast_config())
    accel = _stationary_accel(0.0, math.radians(60.0))

    for i in range(20):
        assert estimator.update(i * 10_000, accel, (0.0, 0.0, 0.0)) is None

    assert not estimator.is_ready
    assert estimator.calibration_progress == 0.0


def test_exact_gyro_integration_tracks_relative_yaw():
    estimator = ImuAttitudeEstimator(
        _fast_config(
            gravity_correction_kp=0.0,
            gyro_bias_ki=0.0,
        )
    )
    _bootstrap(estimator)

    estimate = None
    # 1 rad/s for exactly 1 second after the calibration endpoint.
    for i in range(1, 101):
        estimate = estimator.update(40_000 + i * 10_000, (0.0, 0.0, -G), (0.0, 0.0, 1.0))

    assert estimate is not None
    assert estimate.yaw == pytest.approx(1.0, abs=1e-10)
    assert estimate.roll == pytest.approx(0.0, abs=1e-10)
    assert estimate.pitch == pytest.approx(0.0, abs=1e-10)
    assert not estimate.yaw_observable


def test_consistent_roll_motion_tracks_gyro_without_fighting_accelerometer():
    estimator = ImuAttitudeEstimator(_fast_config(gyro_bias_ki=0.0))
    _bootstrap(estimator)

    estimate = None
    roll_rate = 0.5
    for i in range(1, 101):
        roll = roll_rate * i * 0.01
        estimate = estimator.update(
            40_000 + i * 10_000,
            _stationary_accel(roll, 0.0),
            (roll_rate, 0.0, 0.0),
        )

    assert estimate is not None
    assert estimate.roll == pytest.approx(0.5, abs=2e-4)
    assert estimate.accel_trust == pytest.approx(1.0, abs=1e-10)


def test_large_linear_acceleration_disables_false_tilt_correction():
    estimator = ImuAttitudeEstimator(_fast_config())
    _bootstrap(estimator)

    estimate = estimator.update(50_000, (8.0, 0.0, -G), (0.0, 0.0, 0.0))

    assert estimate is not None and estimate.healthy
    assert estimate.accel_trust == 0.0
    assert estimate.roll == pytest.approx(0.0, abs=1e-12)
    assert estimate.pitch == pytest.approx(0.0, abs=1e-12)


def test_maneuver_horizontal_specific_force_zeroes_trust():
    # F13 (trace 20260729T134958Z-visual-course-82d72cb5): sustained
    # specific force 25-40 degrees off gravity with |f|-g inside the old
    # 0.20-1.50 ramp band kept the gravity correction partially trusted
    # (0.1-0.9) and converged the tilt estimate 0.3-0.6 rad toward a false
    # gravity direction.  Here a steady 15-degree off-gravity sample keeps
    # FULL magnitude and partial (0.625) innovation trust under the old
    # rules; the horizontal-specific-force gate must zero the correction.
    magnitude = 10.0
    fh = 2.6  # ~15 degrees off gravity, above the 2.50 zero threshold
    accel = (fh, 0.0, -math.sqrt(magnitude * magnitude - fh * fh))
    estimator = ImuAttitudeEstimator(_fast_config(gyro_bias_ki=0.0))
    _bootstrap(estimator)

    estimate = None
    for i in range(1, 101):
        estimate = estimator.update(40_000 + i * 10_000, accel, (0.0, 0.0, 0.0))

    assert estimate is not None and estimate.healthy
    # Bounded slow-leak tilt over the 1 s window shifts the q-rotated
    # horizontal component slightly; the maneuver-band value is unchanged.
    assert estimate.horizontal_specific_force_mps2 == pytest.approx(fh, abs=0.1)
    assert estimate.accel_magnitude_deviation_mps2 == pytest.approx(
        magnitude - G, abs=1e-9
    )
    assert estimate.accel_trust == 0.0
    # The fast correction must not drag the level estimate toward the false
    # gravity direction no matter how long the maneuver lasts.  The F23 slow
    # secular correction adds a bounded leak in this band (~0.3 deg/s at
    # 15 degrees of innovation, ramping to zero at 10) — orders below the
    # 17-34 degree convergence the F13 fast-correction bug produced, and the
    # price of breaking the F23 seal-in; it is asserted explicitly here.
    assert abs(estimate.roll) < math.radians(0.5)
    assert abs(estimate.pitch) < math.radians(0.5)


def test_vertical_acceleration_beyond_tightened_band_zeroes_trust():
    # The |f|-g zero threshold tightened 1.50 -> 0.50 (F13): a sustained
    # ~1 m/s^2 climb acceleration sits at deviation 1.0, which the old band
    # still trusted at ~0.38.  It must now zero the correction.
    estimator = ImuAttitudeEstimator(_fast_config(gyro_bias_ki=0.0))
    _bootstrap(estimator)

    estimate = estimator.update(50_000, (0.0, 0.0, -(G + 1.0)), (0.0, 0.0, 0.0))

    assert estimate is not None and estimate.healthy
    assert estimate.accel_magnitude_deviation_mps2 == pytest.approx(1.0, abs=1e-9)
    assert estimate.accel_trust == 0.0
    assert estimate.roll == pytest.approx(0.0, abs=1e-12)
    assert estimate.pitch == pytest.approx(0.0, abs=1e-12)


def test_gravity_feedback_recovers_from_spurious_gyro_rotation():
    estimator = ImuAttitudeEstimator(_fast_config(gyro_bias_ki=0.0))
    _bootstrap(estimator)

    # A brief false rate moves attitude away from a level gravity reading.
    for i in range(1, 11):
        estimator.update(40_000 + i * 10_000, (0.0, 0.0, -G), (0.5, 0.0, 0.0))
    disturbed = estimator.last_estimate
    assert disturbed is not None and disturbed.roll > 0.04

    estimate = None
    for i in range(11, 411):
        estimate = estimator.update(40_000 + i * 10_000, (0.0, 0.0, -G), (0.0, 0.0, 0.0))
    assert estimate is not None
    assert abs(estimate.roll) < 0.001


def test_duplicate_invalid_and_gap_samples_fail_safely():
    estimator = ImuAttitudeEstimator(_fast_config(max_dt_s=0.02))
    initial = _bootstrap(estimator)

    assert estimator.update(40_000, (0.0, 0.0, -G), (10.0, 0.0, 0.0)) is None
    assert estimator.update(39_999, (0.0, 0.0, -G), (10.0, 0.0, 0.0)) is None
    assert estimator.update(50_000, (math.nan, 0.0, -G), (0.0, 0.0, 0.0)) is None
    assert estimator.rejected_samples == 3

    gap = estimator.update(100_000, (0.0, 0.0, -G), (10.0, 0.0, 0.0))
    assert gap is not None
    assert not gap.healthy
    assert not gap.propagated
    assert gap.reason == "timestamp_gap"
    assert gap.orientation.to_euler() == pytest.approx(initial.orientation.to_euler(), abs=1e-12)

    recovered = estimator.update(110_000, (0.0, 0.0, -G), (0.0, 0.0, 0.0))
    assert recovered is not None and recovered.healthy and recovered.propagated


def test_large_backwards_clock_jump_restarts_calibration():
    estimator = ImuAttitudeEstimator(_fast_config(timestamp_reset_threshold_us=50_000))
    _bootstrap(estimator, start_us=1_000_000)

    # Sim reset makes time_usec jump back; old attitude must not cross races.
    assert estimator.update(0, (0.0, 0.0, -G), (0.0, 0.0, 0.0)) is None
    assert not estimator.is_ready
    assert estimator.clock_resets == 1
    assert estimator.calibration_progress == 0.0

    estimate = None
    for i in range(1, 5):
        estimate = estimator.update(i * 10_000, (0.0, 0.0, -G), (0.0, 0.0, 0.0))
    assert estimate is not None and estimator.is_ready


def test_quaternion_stays_normalized_over_long_rotation():
    estimator = ImuAttitudeEstimator(
        _fast_config(gravity_correction_kp=0.0, gyro_bias_ki=0.0)
    )
    _bootstrap(estimator)

    estimate = None
    for i in range(1, 10_001):
        estimate = estimator.update(
            40_000 + i * 1_000,
            (0.0, 0.0, -G),
            (0.7, -0.4, 1.1),
        )
    assert estimate is not None
    q = estimate.orientation
    assert q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "config",
    [
        ImuAttitudeConfig(gravity_mps2=0.0),
        ImuAttitudeConfig(calibration_min_samples=0),
        ImuAttitudeConfig(max_dt_s=0.0),
        ImuAttitudeConfig(
            accel_trust_full_deviation_mps2=1.0,
            accel_trust_zero_deviation_mps2=0.5,
        ),
    ],
)
def test_invalid_config_fails_at_construction(config):
    with pytest.raises(ValueError):
        ImuAttitudeEstimator(config)


def test_slow_secular_correction_breaks_innovation_veto_seal():
    # F23 (trace 20260729T155209Z-visual-course-fed5026d): once gyro drift
    # carried the estimate past the 30-degree innovation veto, every trust
    # gate closed permanently and the error random-walked 25->47 degrees in
    # flight.  The slow secular correction must relax a sealed error back
    # inside the veto envelope on steady accel alone, while a build with
    # the correction disabled stays sealed.
    corrupt = Quaternion.from_euler(math.radians(40.0), 0.0, 0.0)
    corrupt_q = (corrupt.w, corrupt.x, corrupt.y, corrupt.z)

    def _sealed_estimator(**overrides):
        estimator = ImuAttitudeEstimator(_fast_config(gyro_bias_ki=0.0, **overrides))
        _bootstrap(estimator)
        estimator._q = corrupt_q  # white-box: sealed 40-degree roll error
        return estimator

    def _run(estimator, seconds):
        estimate = None
        for i in range(1, int(seconds * 100) + 1):
            estimate = estimator.update(
                40_000 + i * 10_000, (0.0, 0.0, -G), (0.0, 0.0, 0.0)
            )
        assert estimate is not None
        return estimate

    # Control: with the slow correction disabled the error never relaxes —
    # accel_trust is exactly zero for the whole 40 s (innovation and
    # horizontal-specific-force gates both closed).
    sealed = _run(_sealed_estimator(gravity_correction_kp_slow=0.0), 40.0)
    assert sealed.accel_trust == 0.0
    assert abs(sealed.roll) == pytest.approx(math.radians(40.0), abs=1e-9)

    # The slow correction relaxes the seal; once the error is back inside
    # the fast trust envelope the full correction re-engages and finishes.
    recovered = _run(_sealed_estimator(), 40.0)
    assert abs(recovered.roll) < math.radians(5.0)
    assert recovered.accel_trust > 0.9


def test_slow_correction_ignores_impact_magnitude_spikes():
    # The secular pull must stay gated by the wide magnitude band: a hard
    # impact (|f| far above gravity) carries no gravity direction.
    corrupt = Quaternion.from_euler(math.radians(40.0), 0.0, 0.0)
    estimator = ImuAttitudeEstimator(_fast_config(gyro_bias_ki=0.0))
    _bootstrap(estimator)
    estimator._q = (corrupt.w, corrupt.x, corrupt.y, corrupt.z)

    estimate = None
    for i in range(1, 101):
        estimate = estimator.update(
            40_000 + i * 10_000, (0.0, 0.0, -3.0 * G), (0.0, 0.0, 0.0)
        )
    assert estimate is not None
    assert abs(estimate.roll) == pytest.approx(math.radians(40.0), abs=1e-9)
