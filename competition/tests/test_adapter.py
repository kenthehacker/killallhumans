"""Tests for competition adapter types (competition/adapter.py)."""

import math

import numpy as np
import pytest

from competition.adapter import (
    AttitudeCommand,
    AttitudeRateCommand,
    CameraFrame,
    IMUData,
    PositionCommand,
    Quaternion,
    TelemetryState,
)


# ── Quaternion ↔ Euler conversion roundtrip ──────────────────────────────


class TestQuaternionEuler:
    @pytest.mark.parametrize("roll,pitch,yaw", [
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.0, 0.2, 0.0),
        (0.0, 0.0, 0.3),
        (0.1, 0.2, 0.3),
        (-0.5, 0.3, -1.0),
        (0.7, -0.5, 2.5),
    ])
    def test_euler_roundtrip(self, roll, pitch, yaw):
        """from_euler → to_euler should recover the original angles."""
        q = Quaternion.from_euler(roll, pitch, yaw)
        r2, p2, y2 = q.to_euler()
        assert r2 == pytest.approx(roll, abs=1e-10)
        assert p2 == pytest.approx(pitch, abs=1e-10)
        assert y2 == pytest.approx(yaw, abs=1e-10)

    def test_identity_quaternion(self):
        q = Quaternion()  # default: w=1, x=y=z=0
        r, p, y = q.to_euler()
        assert r == pytest.approx(0.0)
        assert p == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_from_euler_identity(self):
        q = Quaternion.from_euler(0.0, 0.0, 0.0)
        assert q.w == pytest.approx(1.0)
        assert q.x == pytest.approx(0.0)
        assert q.y == pytest.approx(0.0)
        assert q.z == pytest.approx(0.0)

    def test_unit_norm(self):
        """Quaternion from Euler should have unit norm."""
        q = Quaternion.from_euler(0.3, -0.5, 1.2)
        norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_90_degree_yaw(self):
        q = Quaternion.from_euler(0.0, 0.0, math.pi / 2)
        r, p, y = q.to_euler()
        assert r == pytest.approx(0.0, abs=1e-10)
        assert p == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(math.pi / 2, abs=1e-10)

    def test_pitch_near_singularity(self):
        """Pitch near ±90° (gimbal lock region) should still work."""
        q = Quaternion.from_euler(0.0, math.pi / 2 - 0.01, 0.0)
        r, p, y = q.to_euler()
        assert p == pytest.approx(math.pi / 2 - 0.01, abs=1e-4)


# ── TelemetryState properties ────────────────────────────────────────────


class TestTelemetryState:
    def _make_state(self, roll=0.0, pitch=0.0, yaw=0.0,
                     velocity=(0, 0, 0)) -> TelemetryState:
        return TelemetryState(
            timestamp_us=0,
            position_ned=(0, 0, 0),
            velocity_ned=velocity,
            orientation=Quaternion.from_euler(roll, pitch, yaw),
            angular_velocity=(0, 0, 0),
        )

    def test_roll_property(self):
        state = self._make_state(roll=0.5)
        assert state.roll == pytest.approx(0.5, abs=1e-6)

    def test_pitch_property(self):
        state = self._make_state(pitch=-0.3)
        assert state.pitch == pytest.approx(-0.3, abs=1e-6)

    def test_yaw_property(self):
        state = self._make_state(yaw=1.0)
        assert state.yaw == pytest.approx(1.0, abs=1e-6)

    def test_speed_stationary(self):
        state = self._make_state(velocity=(0, 0, 0))
        assert state.speed == pytest.approx(0.0)

    def test_speed_forward(self):
        state = self._make_state(velocity=(3, 4, 0))
        assert state.speed == pytest.approx(5.0)

    def test_speed_3d(self):
        state = self._make_state(velocity=(1, 2, 2))
        assert state.speed == pytest.approx(3.0)

    def test_imu_optional(self):
        state = self._make_state()
        assert state.imu is None


# ── AttitudeCommand construction ─────────────────────────────────────────


class TestAttitudeCommand:
    def test_construction(self):
        cmd = AttitudeCommand(
            roll_rad=0.1,
            pitch_rad=-0.2,
            yaw_rad=1.5,
            thrust=0.6,
        )
        assert cmd.roll_rad == 0.1
        assert cmd.pitch_rad == -0.2
        assert cmd.yaw_rad == 1.5
        assert cmd.thrust == 0.6

    def test_zero_command(self):
        cmd = AttitudeCommand(roll_rad=0, pitch_rad=0, yaw_rad=0, thrust=0)
        assert cmd.thrust == 0.0
        assert cmd.roll_rad == 0.0


class TestAttitudeRateCommand:
    def test_construction(self):
        cmd = AttitudeRateCommand(
            roll_rate=1.0, pitch_rate=-0.5,
            yaw_rate=0.3, thrust=0.7,
        )
        assert cmd.roll_rate == 1.0
        assert cmd.pitch_rate == -0.5
        assert cmd.yaw_rate == 0.3
        assert cmd.thrust == 0.7


class TestPositionCommand:
    def test_defaults(self):
        cmd = PositionCommand(position_ned=(1, 2, 3))
        assert cmd.velocity_ned == (0.0, 0.0, 0.0)
        assert cmd.yaw_rad == 0.0


class TestIMUData:
    def test_construction(self):
        imu = IMUData(
            timestamp_us=1000,
            accel=(0, 0, -9.81),
            gyro=(0, 0, 0),
        )
        assert imu.timestamp_us == 1000
        assert imu.accel == (0, 0, -9.81)
        assert imu.mag is None


class TestCameraFrame:
    def test_construction(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = CameraFrame(timestamp_us=5000, image=img)
        assert frame.width == 640
        assert frame.height == 480
        assert frame.image.shape == (480, 640, 3)
