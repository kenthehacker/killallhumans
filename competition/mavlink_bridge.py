"""
MAVLink/MAVSDK adapter for the DCL competition simulator.

Communicates over MAVLink v2 / UDP using MAVSDK-Python's async API.
Handles:
  - Connection lifecycle and heartbeat maintenance
  - Telemetry subscription (ATTITUDE, HIGHRES_IMU, ODOMETRY)
  - Offboard control (SET_ATTITUDE_TARGET via offboard.set_attitude)
  - Vision stream integration (when specification is released)

Tech spec requirements:
  - Physics sim rate: 120 Hz
  - Recommended command rate: 50-120 Hz
  - Minimum heartbeat rate: 2 Hz (MAVSDK handles this internally)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np

try:
    from mavsdk import System
    from mavsdk.offboard import (
        Attitude,
        AttitudeRate,
        OffboardError,
        PositionNedYaw,
        VelocityNedYaw,
    )

    MAVSDK_AVAILABLE = True
except ImportError:
    MAVSDK_AVAILABLE = False

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

logger = logging.getLogger(__name__)


class MAVLinkBridge(CompetitionInterface):
    """
    MAVSDK-based adapter for the competition simulator.

    Usage:
        bridge = MAVLinkBridge()
        await bridge.connect("udp://:14540")
        await bridge.arm()
        await bridge.start_offboard()

        while racing:
            telem = await bridge.get_telemetry()
            frame = await bridge.get_camera_frame()
            # ... perception, planning, control ...
            await bridge.send_attitude(cmd)

        await bridge.stop_offboard()
        await bridge.disconnect()
    """

    def __init__(self):
        if not MAVSDK_AVAILABLE:
            raise RuntimeError(
                "mavsdk package not installed. Install with: pip install mavsdk"
            )
        self._system = System()
        self._connected = False
        self._armed = False
        self._offboard_active = False

        # Latest telemetry (updated by subscription tasks)
        self._latest_telem: Optional[TelemetryState] = None
        self._latest_imu: Optional[IMUData] = None
        self._subscription_tasks: list[asyncio.Task] = []

    async def connect(self, address: str = "udp://:14540") -> None:
        logger.info("Connecting to simulator at %s", address)
        await self._system.connect(system_address=address)

        # Wait for connection
        logger.info("Waiting for drone connection...")
        async for state in self._system.core.connection_state():
            if state.is_connected:
                logger.info("Connected to drone")
                break

        self._connected = True

        # Start telemetry subscriptions
        self._subscription_tasks = [
            asyncio.create_task(self._subscribe_position_velocity()),
            asyncio.create_task(self._subscribe_attitude()),
            asyncio.create_task(self._subscribe_imu()),
        ]

    async def disconnect(self) -> None:
        for task in self._subscription_tasks:
            task.cancel()
        self._subscription_tasks.clear()
        self._connected = False
        logger.info("Disconnected")

    async def arm(self) -> None:
        logger.info("Arming...")
        await self._system.action.arm()
        self._armed = True
        logger.info("Armed")

    async def start_offboard(self) -> None:
        """
        Enter offboard mode.

        Must set an initial setpoint before starting offboard,
        otherwise PX4/simulator will reject the mode switch.
        """
        logger.info("Starting offboard mode")
        # Set initial setpoint (hover attitude)
        await self._system.offboard.set_attitude(
            Attitude(0.0, 0.0, 0.0, 0.5)  # level, 50% thrust
        )
        try:
            await self._system.offboard.start()
            self._offboard_active = True
            logger.info("Offboard mode active")
        except OffboardError as e:
            logger.error("Failed to start offboard: %s", e)
            raise

    async def stop_offboard(self) -> None:
        if self._offboard_active:
            try:
                await self._system.offboard.stop()
            except OffboardError:
                pass
            self._offboard_active = False
            logger.info("Offboard mode stopped")

    async def get_telemetry(self) -> TelemetryState:
        if self._latest_telem is None:
            raise RuntimeError("No telemetry received yet")
        return self._latest_telem

    async def get_camera_frame(self) -> Optional[CameraFrame]:
        # Vision stream spec not yet released by competition
        # This will be implemented when the vision stream specification
        # is provided. For now, return None.
        return None

    async def send_attitude(self, cmd: AttitudeCommand) -> None:
        if not self._offboard_active:
            raise RuntimeError("Offboard mode not active")
        await self._system.offboard.set_attitude(
            Attitude(
                roll_deg=_rad_to_deg(cmd.roll_rad),
                pitch_deg=_rad_to_deg(cmd.pitch_rad),
                yaw_deg=_rad_to_deg(cmd.yaw_rad),
                thrust_value=cmd.thrust,
            )
        )

    async def send_attitude_rate(self, cmd: AttitudeRateCommand) -> None:
        if not self._offboard_active:
            raise RuntimeError("Offboard mode not active")
        await self._system.offboard.set_attitude_rate(
            AttitudeRate(
                roll_deg_s=_rad_to_deg(cmd.roll_rate),
                pitch_deg_s=_rad_to_deg(cmd.pitch_rate),
                yaw_deg_s=_rad_to_deg(cmd.yaw_rate),
                thrust_value=cmd.thrust,
            )
        )

    async def send_position(self, cmd: PositionCommand) -> None:
        if not self._offboard_active:
            raise RuntimeError("Offboard mode not active")
        n, e, d = cmd.position_ned
        await self._system.offboard.set_position_ned(
            PositionNedYaw(n, e, d, _rad_to_deg(cmd.yaw_rad))
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_armed(self) -> bool:
        return self._armed

    # ── Telemetry subscriptions ──────────────────────────────────

    async def _subscribe_position_velocity(self) -> None:
        """Subscribe to position/velocity via ODOMETRY."""
        async for odom in self._system.telemetry.odometry():
            pos = odom.position_body
            vel = odom.velocity_body
            q = odom.q
            ts = int(time.time() * 1e6)

            self._latest_telem = TelemetryState(
                timestamp_us=ts,
                position_ned=(pos.x_m, pos.y_m, pos.z_m),
                velocity_ned=(vel.x_m_s, vel.y_m_s, vel.z_m_s),
                orientation=Quaternion(w=q.w, x=q.x, y=q.y, z=q.z),
                angular_velocity=(
                    odom.angular_velocity_body.roll_rad_s,
                    odom.angular_velocity_body.pitch_rad_s,
                    odom.angular_velocity_body.yaw_rad_s,
                ),
                imu=self._latest_imu,
            )

    async def _subscribe_attitude(self) -> None:
        """Subscribe to attitude quaternion."""
        async for attitude in self._system.telemetry.attitude_quaternion():
            # Attitude updates are merged into the main telemetry state
            # by the position/velocity subscription. This is a backup
            # in case odometry doesn't include attitude.
            if self._latest_telem is not None:
                self._latest_telem.orientation = Quaternion(
                    w=attitude.w, x=attitude.x, y=attitude.y, z=attitude.z
                )

    async def _subscribe_imu(self) -> None:
        """Subscribe to high-rate IMU data."""
        async for imu in self._system.telemetry.imu():
            self._latest_imu = IMUData(
                timestamp_us=int(time.time() * 1e6),
                accel=(
                    imu.acceleration_frd.forward_m_s2,
                    imu.acceleration_frd.right_m_s2,
                    imu.acceleration_frd.down_m_s2,
                ),
                gyro=(
                    imu.angular_velocity_frd.forward_rad_s,
                    imu.angular_velocity_frd.right_rad_s,
                    imu.angular_velocity_frd.down_rad_s,
                ),
            )
            if self._latest_telem is not None:
                self._latest_telem.imu = self._latest_imu


def _rad_to_deg(rad: float) -> float:
    import math
    return rad * 180.0 / math.pi
