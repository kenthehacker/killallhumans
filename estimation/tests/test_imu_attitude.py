"""Unit tests for the VQ2 HIGHRES_IMU-only attitude estimator."""

import math

import pytest

from competition.adapter import IMUData
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
