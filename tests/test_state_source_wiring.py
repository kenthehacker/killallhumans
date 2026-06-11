"""Tests for the AIGP primary state-source wiring in RacePipeline."""

import pytest

from competition.adapter import IMUData, Quaternion, TelemetryState
from estimation.ekf import DroneEKF
from race_pipeline import PipelineConfig, RacePipeline, _gate_normal


def _pipeline_stub():
    pipe = RacePipeline.__new__(RacePipeline)
    pipe.config = PipelineConfig(use_ekf=True)
    pipe.ekf = DroneEKF()
    pipe._ekf_live_initialized = False
    pipe._last_lpn_stamp_ms = None
    pipe._last_odom_reset_counter = None
    return pipe


def _telem(
    *,
    timestamp_us=1_000_000,
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    yaw=0.0,
    lpn_stamp=1000,
    odom_quality=100,
    reset_counter=0,
    imu_time_us=1_000_000,
):
    return TelemetryState(
        timestamp_us=timestamp_us,
        position_ned=position,
        velocity_ned=velocity,
        orientation=Quaternion.from_euler(0.0, 0.0, yaw),
        angular_velocity=(0.0, 0.0, 0.0),
        imu=IMUData(
            timestamp_us=imu_time_us,
            accel=(0.0, 0.0, -9.81),
            gyro=(0.0, 0.0, 0.0),
        ),
        lpn_time_boot_ms=lpn_stamp,
        odom_time_usec=timestamp_us,
        odom_quality=odom_quality,
        odom_reset_counter=reset_counter,
    )


def test_state_estimate_initializes_from_lpn_and_returns_telemetry_yaw():
    pipe = _pipeline_stub()
    pos, vel, yaw = pipe._update_state_estimate(_telem(
        position=(1.0, 2.0, 3.0),
        velocity=(0.1, 0.2, 0.3),
        yaw=0.75,
    ))

    assert pipe.ekf.position == pytest.approx((1.0, 2.0, 3.0))
    assert pipe.ekf.velocity == pytest.approx((0.1, 0.2, 0.3))
    assert pipe.ekf.orientation[2] == pytest.approx(0.75)
    assert pos == pytest.approx((1.0, 2.0, 3.0))
    assert vel == pytest.approx((0.1, 0.2, 0.3))
    assert yaw == pytest.approx(0.75)


def test_duplicate_lpn_stamp_does_not_apply_second_odometry_update():
    pipe = _pipeline_stub()
    pipe._update_state_estimate(_telem(position=(0.0, 0.0, 0.0), lpn_stamp=1000))

    pos, _, _ = pipe._update_state_estimate(_telem(
        timestamp_us=1_010_000,
        imu_time_us=1_010_000,
        position=(10.0, 0.0, 0.0),
        lpn_stamp=1000,
    ))

    assert pos[0] < 1.0


def test_low_quality_odometry_gates_lpn_update():
    pipe = _pipeline_stub()
    pipe._update_state_estimate(_telem(position=(0.0, 0.0, 0.0), lpn_stamp=1000))

    pos, _, _ = pipe._update_state_estimate(_telem(
        timestamp_us=1_010_000,
        imu_time_us=1_010_000,
        position=(10.0, 0.0, 0.0),
        lpn_stamp=1010,
        odom_quality=10,
    ))

    assert pos[0] < 1.0


def test_odom_reset_counter_reinitializes_without_kalman_lag():
    pipe = _pipeline_stub()
    pipe._update_state_estimate(_telem(position=(0.0, 0.0, 0.0), reset_counter=0))

    pos, _, yaw = pipe._update_state_estimate(_telem(
        timestamp_us=2_000_000,
        imu_time_us=2_000_000,
        position=(5.0, 0.0, 0.0),
        yaw=1.2,
        lpn_stamp=2000,
        reset_counter=1,
    ))

    assert pos == pytest.approx((5.0, 0.0, 0.0))
    assert pipe.ekf.position == pytest.approx((5.0, 0.0, 0.0))
    assert pipe.ekf.orientation[2] == pytest.approx(1.2)
    assert yaw == pytest.approx(1.2)


def test_pipeline_gate_normal_positive_pitch_points_up_in_ned():
    assert _gate_normal(0.0, 0.25) == pytest.approx((
        0.9689124217,
        0.0,
        -0.2474039593,
    ))
