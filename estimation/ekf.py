"""
Extended Kalman Filter for drone state estimation.

Based on the dual-stage approach from "On Your Own" (Romero et al., 2025):
the smooth state estimate produced by dual-stage filtering enabled the
aggressive MPC tuning required for lap-time optimization.

State vector (15-dimensional):
  [0:3]   position (NED, meters)
  [3:6]   velocity (NED, m/s)
  [6:9]   orientation (Euler: roll, pitch, yaw in radians)
  [9:12]  accelerometer bias (m/s^2)
  [12:15] gyroscope bias (rad/s)

Prediction: IMU-driven at sensor rate (120 Hz)
Update: position/velocity from ODOMETRY or PnP corrections
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class EKFConfig:
    """EKF tuning parameters."""
    # Process noise (how much we trust the dynamics model)
    accel_noise_std: float = 0.5        # m/s^2
    gyro_noise_std: float = 0.02        # rad/s
    accel_bias_walk: float = 0.001      # m/s^2 per sqrt(s)
    gyro_bias_walk: float = 0.0001      # rad/s per sqrt(s)

    # Measurement noise
    position_noise_std: float = 0.05    # meters (from odometry)
    velocity_noise_std: float = 0.1     # m/s (from odometry)
    pnp_position_noise_std: float = 0.3 # meters (from PnP — noisier)

    # Initial uncertainties
    initial_position_std: float = 0.1
    initial_velocity_std: float = 0.5
    initial_orientation_std: float = 0.1
    initial_accel_bias_std: float = 0.1
    initial_gyro_bias_std: float = 0.01


STATE_DIM = 15
POS_IDX = slice(0, 3)
VEL_IDX = slice(3, 6)
ORI_IDX = slice(6, 9)
ABIAS_IDX = slice(9, 12)
GBIAS_IDX = slice(12, 15)


class DroneEKF:
    """
    Extended Kalman Filter for quadrotor state estimation.

    Fuses:
      - IMU data (accelerometer + gyroscope) for prediction
      - Odometry measurements (position, velocity) for correction
      - PnP gate pose measurements for drift correction
    """

    def __init__(self, config: EKFConfig = None):
        self.config = config or EKFConfig()
        self.x = np.zeros(STATE_DIM)         # state vector
        self.P = np.eye(STATE_DIM)           # covariance matrix
        self._initialized = False
        self._last_predict_time: Optional[float] = None

        # Set initial covariance
        self._set_initial_covariance()

    def _set_initial_covariance(self) -> None:
        c = self.config
        P0 = np.zeros(STATE_DIM)
        P0[POS_IDX] = c.initial_position_std ** 2
        P0[VEL_IDX] = c.initial_velocity_std ** 2
        P0[ORI_IDX] = c.initial_orientation_std ** 2
        P0[ABIAS_IDX] = c.initial_accel_bias_std ** 2
        P0[GBIAS_IDX] = c.initial_gyro_bias_std ** 2
        self.P = np.diag(P0)

    def initialize(
        self,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float] = (0, 0, 0),
        timestamp_s: float = 0.0,
    ) -> None:
        """Initialize the filter with a known state."""
        self.x[POS_IDX] = position
        self.x[VEL_IDX] = velocity
        self.x[ORI_IDX] = orientation
        self.x[ABIAS_IDX] = 0.0
        self.x[GBIAS_IDX] = 0.0
        self._set_initial_covariance()
        self._initialized = True
        self._last_predict_time = timestamp_s

    def predict(
        self,
        accel_body: Tuple[float, float, float],
        gyro_body: Tuple[float, float, float],
        timestamp_s: float,
    ) -> None:
        """
        IMU-driven prediction step.

        Args:
            accel_body: accelerometer reading (m/s^2, body frame FRD)
            gyro_body: gyroscope reading (rad/s, body frame FRD)
            timestamp_s: sensor timestamp in seconds
        """
        if not self._initialized:
            return

        if self._last_predict_time is None:
            self._last_predict_time = timestamp_s
            return

        dt = timestamp_s - self._last_predict_time
        if dt <= 0 or dt > 1.0:  # sanity check
            self._last_predict_time = timestamp_s
            return
        self._last_predict_time = timestamp_s

        # Extract current state
        roll, pitch, yaw = self.x[ORI_IDX]
        accel_bias = self.x[ABIAS_IDX]
        gyro_bias = self.x[GBIAS_IDX]

        # Correct IMU readings with estimated biases
        accel_corrected = np.array(accel_body) - accel_bias
        gyro_corrected = np.array(gyro_body) - gyro_bias

        # Rotate acceleration from body frame to NED
        R = _rotation_body_to_ned(roll, pitch, yaw)
        accel_ned = R @ accel_corrected
        # Remove gravity (NED: gravity is +9.81 in z-down)
        accel_ned[2] += 9.81

        # State prediction (simple Euler integration)
        self.x[POS_IDX] += self.x[VEL_IDX] * dt + 0.5 * accel_ned * dt * dt
        self.x[VEL_IDX] += accel_ned * dt

        # Orientation prediction from gyro
        # Convert body rates to Euler rates (small-angle approximation for simplicity)
        p, q, r = gyro_corrected
        # Full kinematic equation for Euler rate
        cr, sr = math.cos(roll), math.sin(roll)
        cp = math.cos(pitch)
        tp = math.tan(pitch) if abs(pitch) < 1.4 else math.copysign(5.0, pitch)

        roll_dot = p + sr * tp * q + cr * tp * r
        pitch_dot = cr * q - sr * r
        yaw_dot = (sr / max(cp, 0.1)) * q + (cr / max(cp, 0.1)) * r

        self.x[6] += roll_dot * dt
        self.x[7] += pitch_dot * dt
        self.x[8] += yaw_dot * dt

        # Wrap angles
        self.x[6] = _wrap_angle(self.x[6])
        self.x[7] = _wrap_angle(self.x[7])
        self.x[8] = _wrap_angle(self.x[8])

        # Biases evolve as random walks (no change in prediction)

        # Build process noise Q
        c = self.config
        Q = np.zeros((STATE_DIM, STATE_DIM))
        # Position noise from velocity uncertainty
        Q[POS_IDX, POS_IDX] = np.eye(3) * (c.accel_noise_std * dt * dt) ** 2
        # Velocity noise from acceleration uncertainty
        Q[VEL_IDX, VEL_IDX] = np.eye(3) * (c.accel_noise_std * dt) ** 2
        # Orientation noise from gyro uncertainty
        Q[ORI_IDX, ORI_IDX] = np.eye(3) * (c.gyro_noise_std * dt) ** 2
        # Bias random walk
        Q[ABIAS_IDX, ABIAS_IDX] = np.eye(3) * (c.accel_bias_walk * math.sqrt(dt)) ** 2
        Q[GBIAS_IDX, GBIAS_IDX] = np.eye(3) * (c.gyro_bias_walk * math.sqrt(dt)) ** 2

        # Jacobian F (linearized state transition)
        # Simplified: identity + partial derivatives
        F = np.eye(STATE_DIM)
        F[POS_IDX, VEL_IDX] = np.eye(3) * dt
        # Velocity depends on orientation (through rotation matrix)
        # Simplified: ignore cross-coupling for computational efficiency
        F[VEL_IDX, ABIAS_IDX] = -R * dt

        # Covariance prediction
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            self.P = F @ self.P @ F.T + Q

        # Enforce symmetry and positive definiteness
        self.P = (self.P + self.P.T) / 2
        # Clamp diagonal to prevent divergence
        max_cov = 1e6
        np.clip(self.P, -max_cov, max_cov, out=self.P)
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            self._set_initial_covariance()  # reset on divergence

    def update_odometry(
        self,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
    ) -> None:
        """
        Correction step using odometry measurement.

        Typically from ODOMETRY MAVLink message.
        """
        if not self._initialized:
            return

        # Measurement: [position(3), velocity(3)]
        z = np.array([*position, *velocity])
        H = np.zeros((6, STATE_DIM))
        H[0:3, POS_IDX] = np.eye(3)
        H[3:6, VEL_IDX] = np.eye(3)

        c = self.config
        R = np.diag([
            c.position_noise_std ** 2,
            c.position_noise_std ** 2,
            c.position_noise_std ** 2,
            c.velocity_noise_std ** 2,
            c.velocity_noise_std ** 2,
            c.velocity_noise_std ** 2,
        ])

        self._kalman_update(z, H, R)

    def update_pnp_position(
        self,
        position: Tuple[float, float, float],
    ) -> None:
        """
        Correction step using PnP-derived position.

        When a gate is detected and PnP is solved, we get an absolute
        position measurement. This corrects accumulated drift.
        """
        if not self._initialized:
            return

        z = np.array(position)
        H = np.zeros((3, STATE_DIM))
        H[0:3, POS_IDX] = np.eye(3)

        c = self.config
        R = np.eye(3) * c.pnp_position_noise_std ** 2

        self._kalman_update(z, H, R)

    def _kalman_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Standard Kalman update step."""
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            self._set_initial_covariance()

        y = z - H @ self.x  # innovation
        S = H @ self.P @ H.T + R  # innovation covariance
        try:
            K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        except np.linalg.LinAlgError:
            return  # singular — skip update

        if np.any(np.isnan(K)):
            return

        self.x = self.x + K @ y
        I_KH = np.eye(STATE_DIM) - K @ H
        # Joseph form for numerical stability
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        # Recover from NaN/Inf (can happen in early iterations)
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            self._set_initial_covariance()
            return
        self.P = (self.P + self.P.T) / 2
        np.clip(self.P, -1e6, 1e6, out=self.P)

        # Wrap orientation angles
        self.x[6] = _wrap_angle(self.x[6])
        self.x[7] = _wrap_angle(self.x[7])
        self.x[8] = _wrap_angle(self.x[8])

    # ── Accessors ──────────────────────────────────────────────

    @property
    def position(self) -> Tuple[float, float, float]:
        return tuple(self.x[POS_IDX])

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return tuple(self.x[VEL_IDX])

    @property
    def orientation(self) -> Tuple[float, float, float]:
        """(roll, pitch, yaw) in radians."""
        return tuple(self.x[ORI_IDX])

    @property
    def position_uncertainty(self) -> float:
        """RMS position uncertainty (meters)."""
        return float(np.sqrt(np.trace(self.P[POS_IDX, POS_IDX])))

    @property
    def velocity_uncertainty(self) -> float:
        return float(np.sqrt(np.trace(self.P[VEL_IDX, VEL_IDX])))

    @property
    def is_initialized(self) -> bool:
        return self._initialized


def _rotation_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from body (FRD) to NED frame."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
