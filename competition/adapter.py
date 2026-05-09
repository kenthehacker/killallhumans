"""
Abstract competition interface and concrete adapters.

Provides a clean abstraction over the communication layer so the autonomy
stack can run against:
  - The DCL competition simulator (MAVLink over UDP)
  - The local PyBullet sim (direct function calls)
  - Future physical hardware

The competition uses MAVLink v2 via MAVSDK-compatible interfaces:
  - Transport: UDP
  - Telemetry: ATTITUDE, HIGHRES_IMU, ODOMETRY, HEARTBEAT, TIMESYNC
  - Control: SET_ATTITUDE_TARGET, SET_POSITION_TARGET_LOCAL_NED
  - Offboard mode for autonomous control
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Quaternion:
    """Unit quaternion (w, x, y, z) for orientation."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_euler(self) -> Tuple[float, float, float]:
        """Convert to (roll, pitch, yaw) in radians."""
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float) -> "Quaternion":
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        return Quaternion(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
        )


@dataclass
class IMUData:
    """High-rate IMU data from HIGHRES_IMU message."""
    timestamp_us: int
    accel: Tuple[float, float, float]  # m/s^2, body frame (x-forward, y-right, z-down in NED)
    gyro: Tuple[float, float, float]   # rad/s, body frame
    mag: Optional[Tuple[float, float, float]] = None


@dataclass
class TelemetryState:
    """Full telemetry snapshot from the simulator."""
    timestamp_us: int
    position_ned: Tuple[float, float, float]     # meters, NED frame
    velocity_ned: Tuple[float, float, float]      # m/s, NED frame
    orientation: Quaternion
    angular_velocity: Tuple[float, float, float]  # rad/s, body frame
    imu: Optional[IMUData] = None

    @property
    def roll(self) -> float:
        return self.orientation.to_euler()[0]

    @property
    def pitch(self) -> float:
        return self.orientation.to_euler()[1]

    @property
    def yaw(self) -> float:
        return self.orientation.to_euler()[2]

    @property
    def speed(self) -> float:
        vn, ve, vd = self.velocity_ned
        return math.sqrt(vn * vn + ve * ve + vd * vd)


@dataclass
class AttitudeCommand:
    """
    Attitude-level command sent to the simulator.

    Maps to MAVLink SET_ATTITUDE_TARGET / MAVSDK offboard.set_attitude().
    """
    roll_rad: float      # desired roll angle (radians)
    pitch_rad: float     # desired pitch angle (radians)
    yaw_rad: float       # desired yaw angle (radians)
    thrust: float        # normalized thrust [0, 1]


@dataclass
class AttitudeRateCommand:
    """
    Body-rate command sent to the simulator.

    Maps to MAVSDK offboard.set_attitude_rate().
    """
    roll_rate: float     # rad/s
    pitch_rate: float    # rad/s
    yaw_rate: float      # rad/s
    thrust: float        # normalized [0, 1]


@dataclass
class PositionCommand:
    """
    Position + velocity command.

    Maps to MAVLink SET_POSITION_TARGET_LOCAL_NED.
    """
    position_ned: Tuple[float, float, float]
    velocity_ned: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_rad: float = 0.0


@dataclass
class CameraFrame:
    """A single frame from the forward-facing camera."""
    timestamp_us: int
    image: np.ndarray  # BGR uint8, shape (H, W, 3)
    width: int = 640
    height: int = 480


class CompetitionInterface(ABC):
    """
    Abstract interface to the competition simulator.

    Implementations handle the transport layer (MAVLink UDP, PyBullet direct
    calls, etc.) while the autonomy stack sees a clean Python API.
    """

    @abstractmethod
    async def connect(self, address: str = "udp://:14540") -> None:
        """Establish connection to the simulator."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean shutdown."""

    @abstractmethod
    async def arm(self) -> None:
        """Arm the vehicle."""

    @abstractmethod
    async def start_offboard(self) -> None:
        """Enter offboard control mode."""

    @abstractmethod
    async def stop_offboard(self) -> None:
        """Exit offboard mode."""

    @abstractmethod
    async def get_telemetry(self) -> TelemetryState:
        """Get the latest telemetry snapshot."""

    @abstractmethod
    async def get_camera_frame(self) -> Optional[CameraFrame]:
        """Get the latest camera frame (None if not yet available)."""

    @abstractmethod
    async def send_attitude(self, cmd: AttitudeCommand) -> None:
        """Send an attitude setpoint."""

    @abstractmethod
    async def send_attitude_rate(self, cmd: AttitudeRateCommand) -> None:
        """Send a body-rate setpoint."""

    @abstractmethod
    async def send_position(self, cmd: PositionCommand) -> None:
        """Send a position/velocity setpoint."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether we have an active connection."""

    @property
    @abstractmethod
    def is_armed(self) -> bool:
        """Whether the vehicle is armed."""
