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


def _wrap_angle(a: float) -> float:
    """Wrap a radian angle to (-π, π]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a

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

    Supports two drone backends:

    * ``GPDDrone`` (current sim stack) — exposes ``get_state()`` and
      ``get_camera_image()``, and is driven by position/velocity targets
      via ``step(target_pos, target_vel, target_yaw)``. Attitude commands
      from this adapter are reinterpreted as a low-level position
      perturbation on top of the current hover pose; position commands
      are honored natively.

    * ``QuadrotorDrone`` (legacy sim) — exposes ``get_position/velocity/
      rpy/get_fpv_image/apply_command`` where the fourth argument to
      ``apply_command`` is a *yaw-rate* in normalized units, not an
      absolute heading. The previous adapter treated it as an absolute
      yaw which would spin the drone whenever the tracker asked for a
      non-zero yaw setpoint.
    """

    def __init__(self, drone=None, env=None):
        """
        Args:
            drone: A ``GPDDrone`` (preferred) or ``QuadrotorDrone`` instance.
            env: A ``DroneRaceEnv`` instance.
        """
        self._drone = drone
        self._env = env
        self._connected = False
        self._armed = False
        self._offboard = False
        self._camera_enabled = True
        # ``get_state()`` is the stable modern API on GPDDrone; fall back
        # to the QuadrotorDrone shape if it's missing. Capturing this
        # once at construction avoids per-call hasattr checks.
        self._uses_get_state = drone is not None and hasattr(drone, "get_state")

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

        if self._uses_get_state:
            state = self._drone.get_state()
            pos = state["position"]
            vel = state["velocity"]
            rpy = state["orientation_euler"]
            ang_vel = state.get("angular_velocity", (0.0, 0.0, 0.0))
        else:
            pos = self._drone.get_position()
            vel = self._drone.get_velocity()
            rpy = self._drone.get_rpy()
            ang_vel = (0.0, 0.0, 0.0)

        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
        q = Quaternion.from_euler(roll, pitch, yaw)

        # Convert ENU (PyBullet) to NED for consistency with competition
        # ENU: x-east, y-north, z-up
        # NED: x-north, y-east, z-down
        pos_ned = (pos[1], pos[0], -pos[2])  # swap x/y, negate z
        vel_ned = (vel[1], vel[0], -vel[2])
        ang_vel_ned = (ang_vel[1], ang_vel[0], -ang_vel[2])

        return TelemetryState(
            timestamp_us=int(time.time() * 1e6),
            position_ned=pos_ned,
            velocity_ned=vel_ned,
            orientation=q,
            angular_velocity=ang_vel_ned,
        )

    async def get_camera_frame(self) -> Optional[CameraFrame]:
        if not self._camera_enabled or self._drone is None:
            return None

        # GPDDrone exposes ``get_camera_image``; the legacy QuadrotorDrone
        # exposes ``get_fpv_image``. Try both.
        img = None
        if hasattr(self._drone, "get_camera_image"):
            img = self._drone.get_camera_image()
        elif hasattr(self._drone, "get_fpv_image"):
            img = self._drone.get_fpv_image()

        if img is None:
            return None

        h, w = img.shape[:2]
        return CameraFrame(
            timestamp_us=int(time.time() * 1e6),
            image=img,
            width=w,
            height=h,
        )

    async def send_attitude(self, cmd: AttitudeCommand) -> None:
        if self._drone is None:
            return

        if self._uses_get_state:
            # GPDDrone is position-driven; reinterpret the attitude
            # command as a small hover-target offset. The legacy adapter
            # would fail silently here because GPDDrone has no
            # ``apply_command``. Competition-mode users should prefer
            # ``send_position`` when the pipeline is in geometric-tracker
            # mode — this path exists for tests that only plumb attitude.
            state = self._drone.get_state()
            pos = state["position"]
            # tan(tilt) ≈ lateral accel / g → approximate a position
            # offset of 0.05 m per radian of tilt for gentle steering.
            hop = 0.05
            target_pos = (
                pos[0] + hop * math.sin(cmd.pitch_rad),
                pos[1] + hop * -math.sin(cmd.roll_rad),
                pos[2] + (cmd.thrust - 0.5) * 0.1,
            )
            self._drone.step(
                target_pos=target_pos,
                target_yaw=cmd.yaw_rad,
            )
            return

        # QuadrotorDrone path: apply_command expects a normalized
        # *yaw-rate*, not an absolute yaw. Convert by diff-ing against
        # the current yaw so the semantic matches.
        max_angle = 0.6  # ~35 degrees
        throttle = cmd.thrust
        roll_norm = cmd.roll_rad / max_angle
        pitch_norm = cmd.pitch_rad / max_angle
        current_yaw = self._drone.get_rpy()[2]
        yaw_err = _wrap_angle(cmd.yaw_rad - current_yaw)
        # Map yaw error into a normalized rate command. π rad/s at ±1 is
        # a reasonable default for PyBullet's Crazyflie inner loop.
        yaw_rate_norm = max(-1.0, min(1.0, yaw_err / math.pi))

        roll_norm = max(-1.0, min(1.0, roll_norm))
        pitch_norm = max(-1.0, min(1.0, pitch_norm))

        self._drone.apply_command(throttle, roll_norm, pitch_norm, yaw_rate_norm)

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
        if self._drone is None:
            return
        if not self._uses_get_state:
            # Legacy QuadrotorDrone has no native position control.
            return
        # Convert NED back into ENU (GPDDrone expects world ENU).
        n, e, d = cmd.position_ned
        target_pos = (e, n, -d)
        self._drone.step(
            target_pos=target_pos,
            target_yaw=cmd.yaw_rad,
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_armed(self) -> bool:
        return self._armed
