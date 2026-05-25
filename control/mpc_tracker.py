"""
MPC trajectory tracker — replaces the brute-force grid-search MPC.

Tracks a pre-computed reference trajectory at 50-120 Hz using either:
  A) Nonlinear MPC (via scipy for now, can upgrade to acados)
  B) Geometric tracking controller (Lee et al. "Geometric Tracking
     Control of a Quadrotor UAV on SE(3)") — simpler, faster, adequate
     for trajectory tracking

The geometric controller is the default: it computes desired thrust
and attitude directly from position/velocity tracking error without
an optimization loop. This runs in <1ms per call, well within the
per-frame budget.

Based on "On Your Own" (Romero 2025):
  - MPC outputs collective thrust + body rates
  - State predictor compensates for command latency
  - Without delay compensation, aggressive tuning caused instability
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


def _spec():
    """Iter-013: lazy import of competition.drone_spec to avoid a
    hard `control → competition` import-time dependency.  Used by
    `TrackerConfig` field defaults via `dataclasses.field(default_factory=...)`.
    """
    import competition.drone_spec as ds
    return ds

import numpy as np

from competition.adapter import AttitudeCommand, AttitudeRateCommand
from planning.trajectory_optimizer import TrajectoryPoint


@dataclass
class TrackerConfig:
    """Gains and limits for the trajectory tracker."""
    # Position gains (proportional + derivative)
    # Tuned via systematic sweep in iteration 38 (40+ configs).
    # Research: NGTC (Pries 2025) uses kp=25, kd=11 (mass-normalized).
    # Damping ratio: ζ=(kd+drag)/(2√kp) = (5.5+0.5)/(2√7) ≈ 1.13.
    kp_xy: float = 7.0
    kd_xy: float = 5.5
    kp_z: float = 8.0
    kd_z: float = 5.0

    # Attitude gains
    kr: float = 8.0       # attitude proportional
    kw: float = 2.5       # angular velocity damping

    # Feed-forward weight (0 = pure feedback, 1 = full feedforward)
    # "Leveling the Playing Field" (Kunapuli 2025): feedforward is the most
    # important single fix. Literature uses ff=1.0 but our kinematic sim
    # with drag=0.5 causes basin switching at ff>0.52. Sweep found ff=0.50
    # optimal: balances feedforward anticipation vs. stability margin.
    # Higher ff (0.6-1.0) triggers catastrophic trajectory tracking failure
    # in the kinematic sim (race time 14s→27s). See iteration 38 analysis.
    feedforward_accel: float = 0.50

    # Drag compensation coefficient (Faessler et al., IEEE RA-L 2018).
    # Set to 0.0 — drag compensation regresses tracking because the sim's
    # drag provides beneficial velocity damping. See iteration 9 analysis.
    drag_coefficient: float = 0.0

    # Reference-velocity drag feedforward (iteration 11).
    # Adds vff * ref_vel to accel_des to cancel drag-induced steady-state error.
    # Unlike drag_coefficient (which uses CURRENT vel and kills damping),
    # this uses REFERENCE vel, preserving drag's velocity-error damping
    # while eliminating drag-induced forcing. See iteration 11 synthesis.
    # Backed by: Tal & Karaman 2018, L1Quad 2025, DATT 2023.
    velocity_feedforward: float = 0.0

    # Physical limits — iter-013: sourced from competition.drone_spec
    # for cross-module consistency. Pre-iter-013 these were inline
    # literals and could silently drift from `DroneConstraints` /
    # bench. `max_thrust_normalized` and `min_thrust_normalized` are
    # tracker-control fractions, not drone-envelope properties, so
    # they stay inline.
    max_tilt_rad: float = field(
        default_factory=lambda: _spec().DEFAULT_MAX_TILT_RAD
    )
    max_thrust_normalized: float = 0.95
    min_thrust_normalized: float = 0.05
    max_body_rate: float = field(
        default_factory=lambda: _spec().DEFAULT_MAX_BODY_RATE_RAD_S
    )

    # Drone parameters — iter-013: drone_spec authority.
    mass: float = field(default_factory=lambda: _spec().DEFAULT_MASS_KG)
    gravity: float = field(
        default_factory=lambda: _spec().DEFAULT_GRAVITY_MPS2
    )
    max_thrust_n: float = field(
        default_factory=lambda: _spec().DEFAULT_MAX_THRUST_N
    )

    # iter-001 A15: learned tracker residual (lightweight ML).
    # Off-by-default — only enable on a track where a trained model has
    # been validated to beat baseline on the holdout set. The MLP is
    # bypassed entirely when use_residual=False, so default-config
    # callers get byte-identical behaviour to the pre-A15 tracker.
    use_residual: bool = False
    residual_weights_path: Optional[str] = None      # .npz path (see control/learned_residual.py)
    residual_clamp_rad: float = 0.05                 # hard clamp on roll/pitch deltas (~2.9°)
    residual_thrust_clamp: float = 0.05              # hard clamp on thrust delta (~5% of full scale)

    # Iter-014: ML training data collection hook. When True, the tracker
    # appends (features_10d, nominal_roll, nominal_pitch, nominal_thrust)
    # to `self.feature_trace` each step. Off by default — zero overhead
    # for production-config callers. A future training script consumes
    # the trace to fit `TrackerResidualMLP` weights via feedback-error-
    # learning (Romero 2025 "On Your Own"; Pries 2025 NGTC).
    trace_features: bool = False


class GeometricTracker:
    """
    Geometric tracking controller for quadrotor trajectory following.

    Computes desired thrust and attitude from:
      - Position/velocity error relative to reference trajectory
      - Feed-forward acceleration from trajectory
      - Yaw reference from trajectory

    Output: AttitudeCommand (roll, pitch, yaw, thrust) for MAVSDK.

    Reference: Lee et al., "Geometric Tracking Control of a Quadrotor
    UAV on SE(3)", CDC 2010.
    """

    def __init__(self, config: TrackerConfig = None):
        self.config = config or TrackerConfig()
        # iter-001 A15: optional learned residual. Loaded lazily so the
        # import / npz-load cost is only paid when the feature is on.
        self._residual = None
        if self.config.use_residual:
            from control.learned_residual import TrackerResidualMLP
            if self.config.residual_weights_path:
                self._residual = TrackerResidualMLP.from_npz(
                    self.config.residual_weights_path
                )
            else:
                # Safety baseline: zero-init weights produce zero residual,
                # so enabling the feature without a trained model is still
                # byte-identical to baseline (modulo float math from the
                # extra branch — see test_residual_off_is_baseline).
                self._residual = TrackerResidualMLP.zero_init()
        # Iter-014: ML training data collection. Empty list unless
        # `config.trace_features=True`; each step appends a tuple of
        # (features_10d, roll_nominal_rad, pitch_nominal_rad, thrust_nominal,
        #  pos_err_world_xyz, vel_err_world_xyz). A future training
        # script consumes this trace to fit the residual MLP.
        self.feature_trace: list = []

    def track(
        self,
        current_position: Tuple[float, float, float],
        current_velocity: Tuple[float, float, float],
        current_yaw: float,
        reference: TrajectoryPoint,
    ) -> AttitudeCommand:
        """
        Compute attitude command to track a reference trajectory point.

        Args:
            current_position: drone position (NED, meters)
            current_velocity: drone velocity (NED, m/s)
            current_yaw: drone heading (radians)
            reference: target trajectory point

        Returns:
            AttitudeCommand for MAVSDK offboard.set_attitude()
        """
        c = self.config
        pos = np.array(current_position)
        vel = np.array(current_velocity)
        ref_pos = np.array(reference.position)
        ref_vel = np.array(reference.velocity)
        ref_acc = np.array(reference.acceleration)

        # Position and velocity errors
        ep = ref_pos - pos
        ev = ref_vel - vel

        # Desired acceleration (PD + feedforward + optional drag compensation)
        # velocity_feedforward: adds vff * ref_vel to cancel drag-induced
        # steady-state error while preserving drag's velocity-error damping.
        accel_des = np.array([
            c.kp_xy * ep[0] + c.kd_xy * ev[0] + c.feedforward_accel * ref_acc[0] + c.drag_coefficient * vel[0] + c.velocity_feedforward * ref_vel[0],
            c.kp_xy * ep[1] + c.kd_xy * ev[1] + c.feedforward_accel * ref_acc[1] + c.drag_coefficient * vel[1] + c.velocity_feedforward * ref_vel[1],
            c.kp_z * ep[2] + c.kd_z * ev[2] + c.feedforward_accel * ref_acc[2] + c.drag_coefficient * vel[2] + c.velocity_feedforward * ref_vel[2],
        ])

        # In NED: gravity is (0, 0, g) pointing down
        # Total thrust = mass * (accel_des - gravity)
        # NED convention: z-down, so gravity adds to z
        thrust_vec = c.mass * (accel_des - np.array([0, 0, c.gravity]))

        # Thrust magnitude
        thrust_mag = float(np.linalg.norm(thrust_vec))
        thrust_normalized = thrust_mag / c.max_thrust_n
        thrust_normalized = max(c.min_thrust_normalized,
                              min(c.max_thrust_normalized, thrust_normalized))

        # Desired body z-axis (thrust direction, normalized)
        if thrust_mag > 0.01:
            z_b_des = thrust_vec / thrust_mag
        else:
            z_b_des = np.array([0, 0, -1])  # NED: hover = thrust upward = -z

        # Extract desired roll and pitch from thrust direction
        # Using desired yaw to resolve the heading ambiguity
        yaw_des = reference.yaw

        # Construct desired rotation matrix
        # x_c = [cos(yaw), sin(yaw), 0] — desired heading projection
        x_c = np.array([math.cos(yaw_des), math.sin(yaw_des), 0])

        # y_b = z_b x x_c (perpendicular to thrust and heading)
        y_b = np.cross(z_b_des, x_c)
        y_b_norm = np.linalg.norm(y_b)
        if y_b_norm > 0.01:
            y_b = y_b / y_b_norm
        else:
            y_b = np.array([0, 1, 0])

        # x_b = y_b x z_b
        x_b = np.cross(y_b, z_b_des)

        # Desired rotation matrix columns
        # R_des = [x_b | y_b | z_b]

        # Extract Euler angles from desired thrust direction.
        # z_b_des points along thrust (upward in hover = [0,0,-1] in NED).
        # Roll and pitch are measured relative to the upward direction,
        # so we negate z_b_des[2] to use -z (up) as the reference axis.
        #
        # pitch = -asin(x-component): positive x thrust -> negative pitch (nose down -> forward)
        # roll = atan2(y, -z): positive y thrust -> positive roll (right side down -> eastward)
        desired_pitch = -math.asin(np.clip(z_b_des[0], -1, 1))
        desired_roll = math.atan2(z_b_des[1], -z_b_des[2])

        # Clamp to limits
        desired_roll = np.clip(desired_roll, -c.max_tilt_rad, c.max_tilt_rad)
        desired_pitch = np.clip(desired_pitch, -c.max_tilt_rad, c.max_tilt_rad)

        # Iter-014: ML training data capture. The trace records the
        # tracker's NOMINAL outputs (before residual + clamp) plus the
        # 10-dim feature vector, so a future training script can fit
        # `TrackerResidualMLP` to whatever target signal it picks
        # (e.g. negative tracking error projected onto the
        # roll/pitch/thrust influence axes). Off by default; zero
        # overhead when `config.trace_features=False`.
        if c.trace_features:
            from control.learned_residual import build_input_features
            features = build_input_features(
                pos_err=ep, vel_err=ev, ref_accel=ref_acc,
                thrust_normalized=thrust_normalized,
            )
            self.feature_trace.append((
                features,
                float(desired_roll),
                float(desired_pitch),
                float(thrust_normalized),
                tuple(float(v) for v in ep),
                tuple(float(v) for v in ev),
            ))

        # iter-001 A15: learned residual on (roll, pitch, thrust). Hard-
        # clamped at the consumer so a corrupted weights file or out-of-
        # distribution input cannot push commands beyond the safety
        # envelope. Re-applies max_tilt_rad / thrust limits after the
        # residual is added — clamp composition: residual-clamp first,
        # then physical clamp. Off-switch: `use_residual=False`.
        if self._residual is not None:
            from control.learned_residual import build_input_features
            x = build_input_features(
                pos_err=ep, vel_err=ev, ref_accel=ref_acc,
                thrust_normalized=thrust_normalized,
            )
            r = self._residual.forward(x)
            d_roll = float(np.clip(r[0], -c.residual_clamp_rad, c.residual_clamp_rad))
            d_pitch = float(np.clip(r[1], -c.residual_clamp_rad, c.residual_clamp_rad))
            d_thrust = float(
                np.clip(r[2], -c.residual_thrust_clamp, c.residual_thrust_clamp)
            )
            desired_roll = float(np.clip(
                desired_roll + d_roll, -c.max_tilt_rad, c.max_tilt_rad,
            ))
            desired_pitch = float(np.clip(
                desired_pitch + d_pitch, -c.max_tilt_rad, c.max_tilt_rad,
            ))
            thrust_normalized = float(np.clip(
                thrust_normalized + d_thrust,
                c.min_thrust_normalized, c.max_thrust_normalized,
            ))

        cmd = AttitudeCommand(
            roll_rad=float(desired_roll),
            pitch_rad=float(desired_pitch),
            yaw_rad=float(yaw_des),
            thrust=float(thrust_normalized),
        )
        # Store desired acceleration for kinematic sim access
        self._last_accel_des = accel_des
        return cmd

    @property
    def last_desired_acceleration(self) -> Optional[np.ndarray]:
        """World-frame desired acceleration from the last track() call.

        Useful for kinematic simulations that need the acceleration
        directly, bypassing attitude-to-acceleration conversion which
        is frame-sensitive.
        """
        return getattr(self, '_last_accel_des', None)


class SimplePositionTracker:
    """
    Simplified position tracker for initial testing.

    Uses PD control on position error, converting to attitude commands
    via the standard quadrotor decomposition. Simpler than the geometric
    controller but adequate for moderate speeds.
    """

    def __init__(self, config: TrackerConfig = None):
        self.config = config or TrackerConfig()
        self._integral_z = 0.0

    def track(
        self,
        current_position: Tuple[float, float, float],
        current_velocity: Tuple[float, float, float],
        current_yaw: float,
        target_position: Tuple[float, float, float],
        target_velocity: Tuple[float, float, float] = (0, 0, 0),
        target_yaw: float = 0.0,
        dt: float = 0.01,
    ) -> AttitudeCommand:
        c = self.config
        pos = np.array(current_position)
        vel = np.array(current_velocity)
        tgt_pos = np.array(target_position)
        tgt_vel = np.array(target_velocity)

        ep = tgt_pos - pos
        ev = tgt_vel - vel

        # Desired acceleration in NED
        ax = c.kp_xy * ep[0] + c.kd_xy * ev[0]
        ay = c.kp_xy * ep[1] + c.kd_xy * ev[1]

        # Z with integral for hover stability
        self._integral_z += ep[2] * dt
        self._integral_z = np.clip(self._integral_z, -2.0, 2.0)
        az = c.kp_z * ep[2] + c.kd_z * ev[2] + 0.5 * self._integral_z

        # Convert to body frame
        cy, sy = math.cos(current_yaw), math.sin(current_yaw)
        ax_body = cy * ax + sy * ay
        ay_body = -sy * ax + cy * ay

        # Desired angles
        desired_pitch = math.atan2(ax_body, c.gravity)
        desired_roll = -math.atan2(ay_body, c.gravity)

        desired_pitch = np.clip(desired_pitch, -c.max_tilt_rad, c.max_tilt_rad)
        desired_roll = np.clip(desired_roll, -c.max_tilt_rad, c.max_tilt_rad)

        # Throttle: compensate for gravity + desired vertical accel
        cos_tilt = max(math.cos(desired_roll) * math.cos(desired_pitch), 0.3)
        thrust = c.mass * (c.gravity - az) / cos_tilt / c.max_thrust_n
        thrust = np.clip(thrust, c.min_thrust_normalized, c.max_thrust_normalized)

        return AttitudeCommand(
            roll_rad=float(desired_roll),
            pitch_rad=float(desired_pitch),
            yaw_rad=float(target_yaw),
            thrust=float(thrust),
        )

    def reset(self):
        self._integral_z = 0.0
