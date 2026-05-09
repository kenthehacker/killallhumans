"""
State predictor for latency compensation.

From "On Your Own" (Romero 2025): incorporating a state predictor
compensating for command latency (MPC compute + communication + motor
actuation) was critical for aggressive MPC tuning. Without it,
aggressive control caused instability.

The predictor forward-integrates the current state estimate by the
total system latency so the controller plans from the predicted
future state rather than the (stale) measured state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class LatencyConfig:
    """System latency budget (seconds)."""
    detection_latency: float = 0.020    # gate detection processing time
    estimation_latency: float = 0.002   # EKF update time
    planning_latency: float = 0.005     # MPC solve time
    communication_latency: float = 0.005  # UDP round trip
    actuation_latency: float = 0.010    # motor response time

    @property
    def total_latency(self) -> float:
        return (
            self.detection_latency
            + self.estimation_latency
            + self.planning_latency
            + self.communication_latency
            + self.actuation_latency
        )


class StatePredictor:
    """
    Forward-predicts drone state to compensate for system latency.

    Uses IMU integration for short-horizon prediction (typically 20-50ms).
    This ensures the controller sees the state the drone WILL be in when
    the command actually takes effect, not the state it was in when the
    sensor data was captured.
    """

    def __init__(self, config: LatencyConfig = None):
        self.config = config or LatencyConfig()

    def predict(
        self,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
        orientation: Tuple[float, float, float],
        angular_velocity: Tuple[float, float, float],
        accel_body: Optional[Tuple[float, float, float]] = None,
        dt_override: Optional[float] = None,
    ) -> Tuple[
        Tuple[float, float, float],  # predicted position
        Tuple[float, float, float],  # predicted velocity
        Tuple[float, float, float],  # predicted orientation
    ]:
        """
        Predict state forward by the total system latency.

        Args:
            position: current position (NED, meters)
            velocity: current velocity (NED, m/s)
            orientation: current (roll, pitch, yaw) in radians
            angular_velocity: current body rates (rad/s)
            accel_body: latest accelerometer reading (body frame, optional)
            dt_override: override the prediction horizon (seconds)

        Returns:
            (predicted_position, predicted_velocity, predicted_orientation)
        """
        dt = dt_override if dt_override is not None else self.config.total_latency

        if dt <= 0:
            return position, velocity, orientation

        pos = np.array(position)
        vel = np.array(velocity)
        roll, pitch, yaw = orientation
        p, q, r = angular_velocity

        if accel_body is not None:
            # Full prediction with IMU
            R = _rotation_body_to_ned(roll, pitch, yaw)
            accel_ned = R @ np.array(accel_body)
            accel_ned[2] += 9.81  # remove gravity (NED convention)

            pred_pos = pos + vel * dt + 0.5 * accel_ned * dt * dt
            pred_vel = vel + accel_ned * dt
        else:
            # Constant velocity prediction
            pred_pos = pos + vel * dt
            pred_vel = vel  # assume constant

        # Orientation prediction from angular velocity
        pred_roll = _wrap(roll + p * dt)
        pred_pitch = _wrap(pitch + q * dt)
        pred_yaw = _wrap(yaw + r * dt)

        return (
            tuple(pred_pos),
            tuple(pred_vel),
            (pred_roll, pred_pitch, pred_yaw),
        )


def _rotation_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
