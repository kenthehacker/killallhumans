"""Tests for the Extended Kalman Filter (estimation/ekf.py)."""

import math

import numpy as np
import pytest

from estimation.ekf import (
    DroneEKF,
    EKFConfig,
    STATE_DIM,
    POS_IDX,
    VEL_IDX,
    ORI_IDX,
    ABIAS_IDX,
    GBIAS_IDX,
    _rotation_body_to_ned,
    _wrap_angle,
)


# ── Initialization and state access ──────────────────────────────────────


class TestEKFInitialization:
    def test_default_construction(self):
        ekf = DroneEKF()
        assert ekf.x.shape == (STATE_DIM,)
        assert ekf.P.shape == (STATE_DIM, STATE_DIM)
        assert not ekf.is_initialized

    def test_custom_config(self):
        cfg = EKFConfig(accel_noise_std=1.0, gyro_noise_std=0.05)
        ekf = DroneEKF(config=cfg)
        assert ekf.config.accel_noise_std == 1.0
        assert ekf.config.gyro_noise_std == 0.05

    def test_initialize_sets_state(self):
        ekf = DroneEKF()
        ekf.initialize(position=(1, 2, 3), velocity=(0.1, 0.2, 0.3),
                        orientation=(0.01, 0.02, 0.03), timestamp_s=0.0)
        assert ekf.is_initialized
        assert ekf.position == pytest.approx((1, 2, 3))
        assert ekf.velocity == pytest.approx((0.1, 0.2, 0.3))
        assert ekf.orientation == pytest.approx((0.01, 0.02, 0.03))

    def test_initial_covariance_diagonal(self):
        cfg = EKFConfig(initial_position_std=0.5)
        ekf = DroneEKF(config=cfg)
        ekf.initialize((0, 0, 0))
        # Position covariance = 0.5^2
        assert ekf.P[0, 0] == pytest.approx(0.25)
        assert ekf.P[1, 1] == pytest.approx(0.25)
        assert ekf.P[2, 2] == pytest.approx(0.25)
        # Off-diagonals zero for initial covariance
        assert ekf.P[0, 1] == pytest.approx(0.0)

    def test_biases_initialized_to_zero(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0))
        assert ekf.x[ABIAS_IDX] == pytest.approx([0, 0, 0])
        assert ekf.x[GBIAS_IDX] == pytest.approx([0, 0, 0])

    def test_accessors_return_tuples(self):
        ekf = DroneEKF()
        ekf.initialize((1, 2, 3))
        assert isinstance(ekf.position, tuple)
        assert isinstance(ekf.velocity, tuple)
        assert isinstance(ekf.orientation, tuple)
        assert len(ekf.position) == 3

    def test_position_uncertainty(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0))
        unc = ekf.position_uncertainty
        assert unc > 0
        assert isinstance(unc, float)

    def test_velocity_uncertainty(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0))
        unc = ekf.velocity_uncertainty
        assert unc > 0

    def test_set_orientation_hard_sets_wrapped_euler_angles(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), orientation=(0.0, 0.0, 0.0), timestamp_s=0.0)
        ekf.set_orientation(0.1, -0.2, 3.5)
        roll, pitch, yaw = ekf.orientation
        assert roll == pytest.approx(0.1)
        assert pitch == pytest.approx(-0.2)
        assert yaw == pytest.approx(_wrap_angle(3.5))


# ── Prediction step ──────────────────────────────────────────────────────


class TestEKFPrediction:
    def test_predict_before_initialize_is_noop(self):
        ekf = DroneEKF()
        ekf.predict((0, 0, -9.81), (0, 0, 0), 0.1)
        # State should remain all zeros
        assert ekf.x == pytest.approx(np.zeros(STATE_DIM))

    def test_predict_advances_position_with_velocity(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), velocity=(1, 0, 0), timestamp_s=0.0)
        dt = 0.01
        # Stationary IMU (only gravity in z-down body frame)
        ekf.predict((0, 0, -9.81), (0, 0, 0), dt)
        # Position should have moved in x by approximately v*dt
        assert ekf.position[0] == pytest.approx(1.0 * dt, abs=0.01)

    def test_predict_accelerates_from_imu(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), velocity=(0, 0, 0), timestamp_s=0.0)
        # Apply forward acceleration of 1 m/s^2 in body x (which is NED x when level)
        # plus gravity compensation: body z reads -9.81 when hovering
        accel_body = (1.0, 0.0, -9.81)
        dt = 0.1
        ekf.predict(accel_body, (0, 0, 0), dt)
        # Velocity should have increased in x
        assert ekf.velocity[0] > 0

    def test_predict_orientation_evolves_with_gyro(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), orientation=(0, 0, 0), timestamp_s=0.0)
        # Apply yaw rate
        gyro = (0.0, 0.0, 1.0)  # 1 rad/s yaw rate
        dt = 0.01
        ekf.predict((0, 0, -9.81), gyro, dt)
        # Yaw should have increased
        assert abs(ekf.orientation[2]) > 0

    def test_predict_covariance_grows(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), timestamp_s=0.0)
        P_before = ekf.P.copy()
        ekf.predict((0, 0, -9.81), (0, 0, 0), 0.01)
        # Diagonal should generally increase (process noise added)
        assert np.trace(ekf.P) >= np.trace(P_before)

    def test_predict_skips_negative_dt(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), velocity=(1, 0, 0), timestamp_s=1.0)
        pos_before = ekf.position
        ekf.predict((0, 0, -9.81), (0, 0, 0), 0.5)  # backward in time
        assert ekf.position == pos_before  # should skip

    def test_predict_skips_large_dt(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), velocity=(1, 0, 0), timestamp_s=0.0)
        pos_before = ekf.position
        ekf.predict((0, 0, -9.81), (0, 0, 0), 2.0)  # dt > 1.0
        assert ekf.position == pos_before

    def test_predict_wraps_angles(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), orientation=(0, 0, 3.0), timestamp_s=0.0)
        # Large yaw rate to push past pi
        ekf.predict((0, 0, -9.81), (0, 0, 5.0), 0.01)
        yaw = ekf.orientation[2]
        assert -math.pi <= yaw <= math.pi


# ── Odometry update ──────────────────────────────────────────────────────


class TestOdometryUpdate:
    def test_update_before_initialize_is_noop(self):
        ekf = DroneEKF()
        ekf.update_odometry((1, 2, 3), (0, 0, 0))
        assert ekf.x == pytest.approx(np.zeros(STATE_DIM))

    def test_update_moves_state_toward_measurement(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), timestamp_s=0.0)
        ekf.update_odometry((1, 0, 0), (0, 0, 0))
        # Position should move toward (1, 0, 0)
        assert ekf.position[0] > 0

    def test_update_reduces_position_uncertainty(self):
        ekf = DroneEKF()
        cfg = EKFConfig(initial_position_std=1.0, position_noise_std=0.05)
        ekf = DroneEKF(config=cfg)
        ekf.initialize((0, 0, 0), timestamp_s=0.0)
        unc_before = ekf.position_uncertainty
        ekf.update_odometry((0, 0, 0), (0, 0, 0))
        unc_after = ekf.position_uncertainty
        assert unc_after < unc_before

    def test_multiple_updates_converge(self):
        ekf = DroneEKF()
        ekf.initialize((5, 5, 5), timestamp_s=0.0)
        target = (0, 0, 0)
        for _ in range(50):
            ekf.update_odometry(target, (0, 0, 0))
        assert ekf.position[0] == pytest.approx(0.0, abs=0.1)
        assert ekf.position[1] == pytest.approx(0.0, abs=0.1)
        assert ekf.position[2] == pytest.approx(0.0, abs=0.1)

    def test_velocity_update(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), velocity=(0, 0, 0), timestamp_s=0.0)
        ekf.update_odometry((0, 0, 0), (5, 0, 0))
        assert ekf.velocity[0] > 0


# ── PnP position update ─────────────────────────────────────────────────


class TestPnPUpdate:
    def test_pnp_update_corrects_position(self):
        ekf = DroneEKF()
        ekf.initialize((1, 0, 0), timestamp_s=0.0)
        ekf.update_pnp_position((0, 0, 0))
        # Should move toward (0,0,0)
        assert abs(ekf.position[0]) < 1.0

    def test_pnp_noisier_than_odometry(self):
        """PnP correction should be weaker than odometry (higher noise)."""
        cfg = EKFConfig(position_noise_std=0.05, pnp_position_noise_std=0.3)
        ekf1 = DroneEKF(config=cfg)
        ekf1.initialize((1, 0, 0), timestamp_s=0.0)
        ekf1.update_odometry((0, 0, 0), (0, 0, 0))
        shift_odom = abs(ekf1.position[0] - 1.0)

        ekf2 = DroneEKF(config=cfg)
        ekf2.initialize((1, 0, 0), timestamp_s=0.0)
        ekf2.update_pnp_position((0, 0, 0))
        shift_pnp = abs(ekf2.position[0] - 1.0)

        # Odometry update should cause a bigger shift (lower noise → more trust)
        assert shift_odom > shift_pnp

    def test_pnp_before_initialize_is_noop(self):
        ekf = DroneEKF()
        ekf.update_pnp_position((1, 2, 3))
        assert ekf.x == pytest.approx(np.zeros(STATE_DIM))


# ── Convergence test ─────────────────────────────────────────────────────


class TestConvergence:
    def test_predict_update_loop_converges(self):
        """Initialize with error, run predict+update loop, verify convergence."""
        ekf = DroneEKF()
        true_pos = (5.0, 3.0, -2.0)
        # Initialize with large error
        ekf.initialize((0, 0, 0), velocity=(0, 0, 0), timestamp_s=0.0)

        dt = 1.0 / 120  # 120 Hz
        for i in range(500):
            t = (i + 1) * dt
            ekf.predict((0, 0, -9.81), (0, 0, 0), t)
            if i % 5 == 0:  # odometry at 24 Hz
                ekf.update_odometry(true_pos, (0, 0, 0))

        assert ekf.position[0] == pytest.approx(true_pos[0], abs=0.5)
        assert ekf.position[1] == pytest.approx(true_pos[1], abs=0.5)
        assert ekf.position[2] == pytest.approx(true_pos[2], abs=0.5)

    def test_convergence_with_pnp_only(self):
        ekf = DroneEKF()
        ekf.initialize((10, 10, 10), timestamp_s=0.0)
        true_pos = (0, 0, 0)
        dt = 1.0 / 120
        for i in range(300):
            t = (i + 1) * dt
            ekf.predict((0, 0, -9.81), (0, 0, 0), t)
            if i % 10 == 0:
                ekf.update_pnp_position(true_pos)

        # Should converge, though slower than odometry
        assert abs(ekf.position[0]) < 3.0
        assert abs(ekf.position[1]) < 3.0


# ── Numerical stability ─────────────────────────────────────────────────


class TestNumericalStability:
    def test_large_initial_covariance(self):
        cfg = EKFConfig(initial_position_std=100.0, initial_velocity_std=50.0)
        ekf = DroneEKF(config=cfg)
        ekf.initialize((0, 0, 0), timestamp_s=0.0)
        # Should not crash
        ekf.predict((0, 0, -9.81), (0, 0, 0), 0.01)
        ekf.update_odometry((1, 1, 1), (0, 0, 0))
        assert not np.any(np.isnan(ekf.x))
        assert not np.any(np.isinf(ekf.x))

    def test_covariance_stays_symmetric(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), timestamp_s=0.0)
        for i in range(100):
            t = (i + 1) * 0.01
            ekf.predict((0.1, 0.2, -9.81), (0.01, 0.02, 0.03), t)
            if i % 10 == 0:
                ekf.update_odometry((0, 0, 0), (0, 0, 0))
        assert np.allclose(ekf.P, ekf.P.T, atol=1e-10)

    def test_covariance_positive_diagonal(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), timestamp_s=0.0)
        for i in range(50):
            ekf.predict((1, 2, -8), (0.1, 0.2, 0.3), (i + 1) * 0.01)
        diag = np.diag(ekf.P)
        assert np.all(diag >= 0)

    def test_nan_covariance_resets(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), timestamp_s=0.0)
        # Force NaN into covariance
        ekf.P[0, 0] = float('nan')
        ekf.update_odometry((1, 1, 1), (0, 0, 0))
        # Should have reset covariance
        assert not np.any(np.isnan(ekf.P))

    def test_singular_innovation_covariance_handled(self):
        ekf = DroneEKF()
        ekf.initialize((0, 0, 0), timestamp_s=0.0)
        # Zero covariance → potential singularity in Kalman update
        ekf.P = np.zeros((STATE_DIM, STATE_DIM))
        # Should not crash
        ekf.update_odometry((1, 1, 1), (0, 0, 0))


# ── Helper functions ─────────────────────────────────────────────────────


class TestHelpers:
    def test_rotation_identity_at_zero(self):
        R = _rotation_body_to_ned(0, 0, 0)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_rotation_orthogonal(self):
        R = _rotation_body_to_ned(0.3, 0.2, 0.5)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_wrap_angle(self):
        assert _wrap_angle(0) == pytest.approx(0)
        assert _wrap_angle(math.pi) == pytest.approx(math.pi, abs=1e-10)
        assert _wrap_angle(2 * math.pi) == pytest.approx(0, abs=1e-10)
        # -pi and +pi are both valid; atan2(sin(-pi), cos(-pi)) = pi or -pi
        assert abs(_wrap_angle(-math.pi)) == pytest.approx(math.pi, abs=1e-10)
        assert abs(_wrap_angle(3 * math.pi)) == pytest.approx(math.pi, abs=1e-10)
