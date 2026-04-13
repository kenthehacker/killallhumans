"""
PyBullet adapter — implements CompetitionInterface for local development.

Wraps the existing sim_pybullet stack so the autonomy pipeline can run
unchanged against either the local sim or the competition MAVLink bridge.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np

from .adapter import (
    AttitudeCommand,
    AttitudeRateCommand,
    CameraFrame,
    CompetitionInterface,
    IMUData,
    PositionCommand,
    Quaternion,
    TelemetryState,
)


class PyBulletAdapter(CompetitionInterface):
    """
    Adapter that wraps a PyBullet drone for local testing.

    This allows the full autonomy stack to run against the PyBullet
    simulation using the same interface as the competition MAVLink bridge.
    """

    def __init__(self, drone=None, env=None):
        """
        Args:
            drone: A QuadrotorDrone or GPDDrone instance
            env: A DroneRaceEnv instance
        """
        self._drone = drone
        self._env = env
        self._connected = False
        self._armed = False
        self._offboard = False
        self._camera_enabled = True

    async def connect(self, address: str = "pybullet://local") -> None:
        if self._drone is None or self._env is None:
            raise RuntimeError("PyBullet drone and env must be set before connecting")
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def arm(self) -> None:
        self._armed = True

    async def start_offboard(self) -> None:
        self._offboard = True

    async def stop_offboard(self) -> None:
        self._offboard = False

    async def get_telemetry(self) -> TelemetryState:
        if self._drone is None:
            raise RuntimeError("No drone configured")

        pos = self._drone.get_position()
        vel = self._drone.get_velocity()
        rpy = self._drone.get_rpy()

        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
        q = Quaternion.from_euler(roll, pitch, yaw)

        # Convert ENU (PyBullet) to NED for consistency with competition
        # ENU: x-east, y-north, z-up
        # NED: x-north, y-east, z-down
        pos_ned = (pos[1], pos[0], -pos[2])  # swap x/y, negate z
        vel_ned = (vel[1], vel[0], -vel[2])

        return TelemetryState(
            timestamp_us=int(time.time() * 1e6),
            position_ned=pos_ned,
            velocity_ned=vel_ned,
            orientation=q,
            angular_velocity=(0.0, 0.0, 0.0),  # not directly available from basic drone
        )

    async def get_camera_frame(self) -> Optional[CameraFrame]:
        if not self._camera_enabled or self._drone is None:
            return None

        if hasattr(self._drone, "get_fpv_image"):
            img = self._drone.get_fpv_image()
            if img is not None:
                h, w = img.shape[:2]
                return CameraFrame(
                    timestamp_us=int(time.time() * 1e6),
                    image=img,
                    width=w,
                    height=h,
                )
        return None

    async def send_attitude(self, cmd: AttitudeCommand) -> None:
        if self._drone is None:
            return
        # Convert attitude command to TRPY for PyBullet drone
        max_angle = 0.6  # ~35 degrees
        throttle = cmd.thrust
        roll_norm = cmd.roll_rad / max_angle
        pitch_norm = cmd.pitch_rad / max_angle
        yaw_norm = cmd.yaw_rad / math.pi  # normalize yaw

        roll_norm = max(-1.0, min(1.0, roll_norm))
        pitch_norm = max(-1.0, min(1.0, pitch_norm))
        yaw_norm = max(-1.0, min(1.0, yaw_norm))

        self._drone.apply_command(throttle, roll_norm, pitch_norm, yaw_norm)

    async def send_attitude_rate(self, cmd: AttitudeRateCommand) -> None:
        # Approximate: convert rates to angles for the inner PD loop
        dt = 1.0 / 120.0  # physics timestep
        await self.send_attitude(AttitudeCommand(
            roll_rad=cmd.roll_rate * dt,
            pitch_rad=cmd.pitch_rate * dt,
            yaw_rad=cmd.yaw_rate * dt,
            thrust=cmd.thrust,
        ))

    async def send_position(self, cmd: PositionCommand) -> None:
        # Position control not directly supported by low-level drone
        # Would need the full flight controller in the loop
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_armed(self) -> bool:
        return self._armed
