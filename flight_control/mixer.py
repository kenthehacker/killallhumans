"""
TRPY Mixer: converts world-frame ControlCommand to competition-format
(Throttle, Roll, Pitch, Yaw) commands.

The competition API expects normalized TRPY inputs:
  - throttle: 0.0 (no thrust) to 1.0 (max thrust)
  - roll:    -1.0 (max left) to 1.0 (max right)
  - pitch:   -1.0 (max nose-down) to 1.0 (max nose-up)
  - yaw:     -1.0 (max CCW) to 1.0 (max CW)

Our MPC/PID stack produces world-frame accelerations (ax, ay, az, yaw_rate).
This mixer converts between the two representations using the current
drone orientation.
"""

import math
from dataclasses import dataclass
from typing import Tuple

from .types import ControlCommand


@dataclass
class TRPYCommand:
    """Normalized competition-format control command."""
    throttle: float  # 0.0 to 1.0
    roll: float      # -1.0 to 1.0
    pitch: float     # -1.0 to 1.0
    yaw: float       # -1.0 to 1.0


@dataclass
class MixerConfig:
    gravity: float = 9.81
    drone_mass_kg: float = 1.0
    max_thrust_n: float = 20.0
    max_roll_angle: float = 0.35   # radians — match DroneConfig
    max_pitch_angle: float = 0.35
    max_yaw_rate: float = 2.0     # rad/s


class TRPYMixer:
    """
    Converts ControlCommand (world-frame accelerations) to TRPYCommand.

    Standard quadrotor decomposition:
      - Desired az → throttle (compensating for gravity and current tilt)
      - Desired ax (forward) → pitch angle
      - Desired ay (lateral) → roll angle
      - Desired yaw_rate → yaw
    """

    def __init__(self, config: MixerConfig = None):
        self.config = config or MixerConfig()

    def mix(
        self,
        command: ControlCommand,
        current_roll: float = 0.0,
        current_pitch: float = 0.0,
        current_yaw: float = 0.0,
    ) -> TRPYCommand:
        """
        Convert a ControlCommand to TRPYCommand.

        Args:
            command: world-frame acceleration command from the flight controller
            current_roll/pitch/yaw: current drone orientation in radians
        """
        ax_world = command.ax
        ay_world = command.ay
        az_world = command.az
        yaw_rate = command.yaw_rate

        # Rotate world-frame accelerations into body frame
        cy, sy = math.cos(current_yaw), math.sin(current_yaw)
        ax_body = cy * ax_world + sy * ay_world   # forward in body
        ay_body = -sy * ax_world + cy * ay_world   # leftward in body (ENU: +y = left)

        # PyBullet Ry(pitch): body_z = (sin(pitch),0,cos(pitch)). Negative pitch → -x (backward).
        # So for forward (ax>0) we need POSITIVE desired_pitch.
        desired_pitch = math.atan2(
            ax_body, self.config.gravity
        )
        desired_pitch = _clamp(
            desired_pitch,
            -self.config.max_pitch_angle,
            self.config.max_pitch_angle,
        )

        # In ENU/PyBullet: positive roll (right-hand about +x) → tilt RIGHT → go RIGHT (-y)
        # To go LEFT (ay_body > 0 means +y = left), we need negative roll
        desired_roll = -math.atan2(
            ay_body, self.config.gravity
        )
        desired_roll = _clamp(
            desired_roll,
            -self.config.max_roll_angle,
            self.config.max_roll_angle,
        )

        # Throttle: total thrust needed to achieve desired vertical accel
        # thrust / mass = (g + az_world) / cos(roll) / cos(pitch)
        cos_correction = max(
            math.cos(current_roll) * math.cos(current_pitch),
            0.3,  # prevent division explosion at extreme angles
        )
        thrust_needed = (
            self.config.drone_mass_kg
            * (self.config.gravity + az_world)
            / cos_correction
        )
        throttle = thrust_needed / self.config.max_thrust_n
        throttle = _clamp(throttle, 0.0, 1.0)

        # Normalize to [-1, 1] range
        roll_norm = desired_roll / self.config.max_roll_angle
        pitch_norm = desired_pitch / self.config.max_pitch_angle
        yaw_norm = _clamp(
            yaw_rate / self.config.max_yaw_rate, -1.0, 1.0
        )

        return TRPYCommand(
            throttle=throttle,
            roll=_clamp(roll_norm, -1.0, 1.0),
            pitch=_clamp(pitch_norm, -1.0, 1.0),
            yaw=_clamp(yaw_norm, -1.0, 1.0),
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
