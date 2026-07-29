"""Conservative attitude estimation from AIGP ``HIGHRES_IMU`` only.

VQ2 build 3385 currently exposes gyroscope and accelerometer samples but no
``ATTITUDE``/``ODOMETRY`` attitude.  This module provides the small, transport-
free estimator needed to bridge that gap.  It deliberately estimates only
what a six-axis IMU can observe:

* roll and pitch are gyro-propagated and slowly corrected toward gravity;
* yaw is gyro-propagated relative to ``initial_yaw_rad`` and is *not* absolute;
* gyro bias and initial tilt are learned while the vehicle is stationary on
  the starting pad.

Frame convention
----------------
The repository uses MAVLink FRD body axes and NED world axes.  A stationary,
upright accelerometer therefore measures specific force approximately
``(0, 0, -g)``.  The vehicle need not start level: build 3385 has been observed
near ``(-3.00, 0.00, -9.34) m/s^2``, corresponding to about -18 degrees pitch.
Bootstrap uses that measured gravity direction instead of assuming zero tilt.

The accelerometer is not a tilt sensor while the vehicle is accelerating.
Consequently, gravity correction is smoothly suppressed when either the
acceleration magnitude or its innovation is implausible, and additionally
when the world-frame horizontal specific force says the vehicle is
maneuvering (flight F13, trace 20260729T134958Z-visual-course-82d72cb5:
sustained 25-40 degree off-gravity specific force at |f|-g in the old ramp
band kept the correction partially trusted and converged the tilt estimate
0.3-0.6 rad toward a false gravity direction).  A timestamp gap is
also surfaced as an unhealthy estimate rather than integrated across.  These
guards favor a short conservative stabilization flight over aggressive but
potentially false attitude corrections.

This file performs no I/O and sends no simulator traffic.  Feed it the raw
``timestamp_us``, ``accel`` and ``gyro`` fields from :class:`IMUData`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from competition.adapter import Quaternion


Vector3 = Tuple[float, float, float]
_QuaternionTuple = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ImuAttitudeConfig:
    """Tuning and safety limits for :class:`ImuAttitudeEstimator`.

    Defaults target the simulator's roughly 100--120 Hz ``HIGHRES_IMU``
    stream.  Calibration takes about half a second, comfortably inside the
    race countdown when the estimator is started before reset/GO.
    """

    gravity_mps2: float = 9.80665

    # Fail-closed stationary-pad bootstrap.
    calibration_min_samples: int = 50
    calibration_min_duration_s: float = 0.40
    stationary_accel_tolerance_mps2: float = 0.75
    stationary_gyro_max_rad_s: float = 0.15
    calibration_accel_std_max_mps2: float = 0.20
    calibration_gyro_std_max_rad_s: float = 0.02
    require_upright_bootstrap: bool = True
    bootstrap_max_tilt_rad: float = math.radians(35.0)

    # Mahony-style gravity correction.  ``kp`` is an inverse time constant;
    # bias learning is intentionally much slower and only runs near rest.
    gravity_correction_kp: float = 1.2
    gyro_bias_ki: float = 0.02
    gyro_bias_limit_rad_s: float = 0.25
    bias_learning_gyro_max_rad_s: float = 0.20
    bias_learning_min_accel_trust: float = 0.80

    # Smooth confidence ramps.  Correction has full confidence below the
    # ``full`` value, zero confidence at/above the ``zero`` value.
    accel_trust_full_deviation_mps2: float = 0.20
    # Zero-deviation tightened 1.50 -> 0.50 (F13): |f|-g sat in the old ramp
    # band (~0.9-1.4) throughout maneuvering, keeping the correction
    # partially trusted while the vehicle was 25-40 degrees off gravity.
    accel_trust_zero_deviation_mps2: float = 0.50
    # Horizontal-specific-force maneuver gate (F13): the world-frame
    # horizontal component of measured specific force is ~zero only in
    # near-steady flight; sustained tilt or horizontal acceleration means
    # the accelerometer is not measuring gravity and the correction must be
    # OFF.  Full trust at/below ``full``, zero at/above ``zero``.
    accel_trust_fh_full_mps2: float = 1.00
    accel_trust_fh_zero_mps2: float = 2.50
    accel_innovation_full_rad: float = math.radians(6.0)
    accel_innovation_zero_rad: float = math.radians(30.0)

    # Timing guards.  A large backwards jump is a simulator clock reset and
    # restarts calibration; a small backwards jump is an out-of-order packet.
    max_dt_s: float = 0.050
    timestamp_reset_threshold_us: int = 100_000


@dataclass(frozen=True)
class AttitudeEstimate:
    """One accepted, calibrated attitude estimate.

    ``orientation`` rotates FRD body vectors into NED.  ``body_rates`` are
    bias-corrected measured rates (the accelerometer feedback term is not a
    physical rate and is intentionally excluded).  ``healthy=False`` means a
    timestamp discontinuity was detected and the prior attitude was held.

    ``yaw_observable`` is always false for this estimator.  Callers needing an
    absolute course heading must add a camera/gate or other heading update.
    """

    timestamp_us: int
    orientation: Quaternion
    body_rates: Vector3
    gyro_bias: Vector3
    accel_trust: float
    healthy: bool
    propagated: bool
    yaw_observable: bool = False
    reason: Optional[str] = None
    # Trust-ramp inputs, exported per sample for flight-trace diagnosis
    # (F13 had to be inferred indirectly without them).
    accel_magnitude_deviation_mps2: float = 0.0
    horizontal_specific_force_mps2: float = 0.0

    @property
    def roll(self) -> float:
        return self.orientation.to_euler()[0]

    @property
    def pitch(self) -> float:
        return self.orientation.to_euler()[1]

    @property
    def yaw(self) -> float:
        return self.orientation.to_euler()[2]


class ImuAttitudeEstimator:
    """Quaternion complementary filter driven only by gyro + accelerometer.

    ``update`` returns ``None`` until stationary bootstrap is complete.  This
    makes it difficult for a live controller to accidentally fly on an
    uncalibrated identity attitude.  Check :attr:`is_ready` before arming and
    require ``estimate.healthy`` on each control tick.
    """

    def __init__(
        self,
        config: Optional[ImuAttitudeConfig] = None,
        *,
        initial_yaw_rad: float = 0.0,
    ) -> None:
        self.config = config or ImuAttitudeConfig()
        _validate_config(self.config)
        if not math.isfinite(initial_yaw_rad):
            raise ValueError("initial_yaw_rad must be finite")
        self.initial_yaw_rad = _wrap_angle(float(initial_yaw_rad))
        self.reset()

    def reset(self) -> None:
        """Discard attitude/bias state and require a fresh pad calibration."""

        self._q: _QuaternionTuple = (1.0, 0.0, 0.0, 0.0)
        self._gyro_bias = [0.0, 0.0, 0.0]
        self._ready = False
        self._last_timestamp_us: Optional[int] = None
        self._last_estimate: Optional[AttitudeEstimate] = None

        self._calibration_start_us: Optional[int] = None
        self._calibration_count = 0
        self._calibration_accel_sum = [0.0, 0.0, 0.0]
        self._calibration_accel_sq_sum = [0.0, 0.0, 0.0]
        self._calibration_gyro_sum = [0.0, 0.0, 0.0]
        self._calibration_gyro_sq_sum = [0.0, 0.0, 0.0]

        self.rejected_samples = 0
        self.timestamp_discontinuities = 0
        self.clock_resets = 0
        self.last_rejection_reason: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def calibration_progress(self) -> float:
        """Bootstrap progress in ``[0, 1]`` (both count and time must pass)."""

        if self._ready:
            return 1.0
        cfg = self.config
        count_progress = self._calibration_count / cfg.calibration_min_samples
        if (
            self._calibration_start_us is None
            or self._last_timestamp_us is None
            or cfg.calibration_min_duration_s == 0.0
        ):
            duration_progress = 1.0 if cfg.calibration_min_duration_s == 0.0 else 0.0
        else:
            duration_progress = (
                (self._last_timestamp_us - self._calibration_start_us) * 1e-6
                / cfg.calibration_min_duration_s
            )
        return max(0.0, min(1.0, count_progress, duration_progress))

    @property
    def gyro_bias(self) -> Vector3:
        return tuple(self._gyro_bias)  # type: ignore[return-value]

    @property
    def orientation(self) -> Optional[Quaternion]:
        """Current orientation, or ``None`` while bootstrap is incomplete."""

        return _quaternion_object(self._q) if self._ready else None

    @property
    def last_estimate(self) -> Optional[AttitudeEstimate]:
        return self._last_estimate

    def update_imu(self, imu) -> Optional[AttitudeEstimate]:
        """Convenience adapter for an ``IMUData``-like object.

        Kept structurally typed so this pure module does not depend on a live
        transport or pymavlink.
        """

        return self.update(imu.timestamp_us, imu.accel, imu.gyro)

    def update(
        self,
        timestamp_us: int,
        accel: Sequence[float],
        gyro: Sequence[float],
    ) -> Optional[AttitudeEstimate]:
        """Consume one ``HIGHRES_IMU`` sample.

        Invalid, duplicate and slightly out-of-order samples are rejected and
        return ``None`` without changing the attitude.  A simulator-clock
        reset automatically discards the old estimate and begins a new pad
        calibration.  A forward gap larger than :attr:`max_dt_s` holds the
        previous attitude and returns an explicit unhealthy estimate.
        """

        try:
            stamp = int(timestamp_us)
            accel_v = _finite_vector3(accel, "accel")
            gyro_v = _finite_vector3(gyro, "gyro")
        except (TypeError, ValueError, OverflowError) as exc:
            self._reject(f"invalid_sample: {exc}")
            return None

        if self._last_timestamp_us is not None:
            delta_us = stamp - self._last_timestamp_us
            if delta_us < -self.config.timestamp_reset_threshold_us:
                previous_resets = self.clock_resets + 1
                self.reset()
                self.clock_resets = previous_resets
                self.last_rejection_reason = "clock_reset_recalibrating"
            elif delta_us <= 0:
                self._reject("duplicate_or_out_of_order_timestamp")
                return None

        if not self._ready:
            return self._update_calibration(stamp, accel_v, gyro_v)

        assert self._last_timestamp_us is not None
        dt = (stamp - self._last_timestamp_us) * 1e-6
        self._last_timestamp_us = stamp
        if dt > self.config.max_dt_s:
            self.timestamp_discontinuities += 1
            self.last_rejection_reason = "timestamp_gap"
            estimate = self._make_estimate(
                timestamp_us=stamp,
                body_rates=_subtract(gyro_v, self._gyro_bias),
                accel_trust=0.0,
                healthy=False,
                propagated=False,
                reason="timestamp_gap",
            )
            self._last_estimate = estimate
            return estimate

        cfg = self.config
        body_rates = _subtract(gyro_v, self._gyro_bias)

        # First propagate with the measured angular increment.  Applying the
        # gravity innovation at the sample endpoint avoids adding phase lag to
        # fast, physically consistent rotations.
        q_pred = _integrate_body_rate(self._q, body_rates, dt)
        accel_norm = _norm(accel_v)
        measured_up = _scale(accel_v, 1.0 / accel_norm) if accel_norm > 1e-9 else (0.0, 0.0, 0.0)
        predicted_up = _predicted_specific_force_direction(q_pred)

        magnitude_deviation = abs(accel_norm - cfg.gravity_mps2)
        magnitude_trust = _descending_ramp(
            magnitude_deviation,
            cfg.accel_trust_full_deviation_mps2,
            cfg.accel_trust_zero_deviation_mps2,
        )
        innovation_angle = _angle_between(measured_up, predicted_up)
        innovation_trust = _descending_ramp(
            innovation_angle,
            cfg.accel_innovation_full_rad,
            cfg.accel_innovation_zero_rad,
        )
        # Maneuver gate (F13): world-frame horizontal specific force.  The
        # innovation ramp alone could not catch sustained off-gravity flight
        # because the correction converged predicted_up toward the false
        # measurement; fh stays large for as long as the maneuver does.
        specific_force_world = _rotate_body_to_world(q_pred, accel_v)
        horizontal_specific_force = math.hypot(
            specific_force_world[0], specific_force_world[1]
        )
        fh_trust = _descending_ramp(
            horizontal_specific_force,
            cfg.accel_trust_fh_full_mps2,
            cfg.accel_trust_fh_zero_mps2,
        )
        accel_trust = magnitude_trust * innovation_trust * fh_trust

        # For q(body->NED), measured_up x predicted_up has the body-rate sign
        # that rotates the predicted gravity direction toward the measurement.
        error = _cross(measured_up, predicted_up)
        feedback_rate = _scale(error, cfg.gravity_correction_kp * accel_trust)
        self._q = _integrate_body_rate(q_pred, feedback_rate, dt)

        # Learn bias only in a near-stationary regime.  This keeps sustained
        # racing acceleration from being absorbed as a fictitious gyro bias.
        if (
            cfg.gyro_bias_ki > 0.0
            and accel_trust >= cfg.bias_learning_min_accel_trust
            and _norm(gyro_v) <= cfg.bias_learning_gyro_max_rad_s
        ):
            limit = cfg.gyro_bias_limit_rad_s
            for i in range(3):
                self._gyro_bias[i] = _clamp(
                    self._gyro_bias[i] - cfg.gyro_bias_ki * accel_trust * error[i] * dt,
                    -limit,
                    limit,
                )
            body_rates = _subtract(gyro_v, self._gyro_bias)

        self.last_rejection_reason = None
        estimate = self._make_estimate(
            timestamp_us=stamp,
            body_rates=body_rates,
            accel_trust=accel_trust,
            healthy=True,
            propagated=True,
            accel_magnitude_deviation_mps2=magnitude_deviation,
            horizontal_specific_force_mps2=horizontal_specific_force,
        )
        self._last_estimate = estimate
        return estimate

    def _update_calibration(
        self,
        stamp: int,
        accel: Vector3,
        gyro: Vector3,
    ) -> Optional[AttitudeEstimate]:
        cfg = self.config
        accel_norm = _norm(accel)
        stationary = (
            abs(accel_norm - cfg.gravity_mps2) <= cfg.stationary_accel_tolerance_mps2
            and _norm(gyro) <= cfg.stationary_gyro_max_rad_s
        )
        if cfg.require_upright_bootstrap:
            if accel_norm <= 1e-9 or accel[2] >= 0.0:
                stationary = False
            else:
                tilt = math.acos(_clamp(-accel[2] / accel_norm, -1.0, 1.0))
                if tilt > cfg.bootstrap_max_tilt_rad:
                    stationary = False

        if not stationary:
            self._clear_calibration_window()
            self._last_timestamp_us = stamp
            self.last_rejection_reason = "bootstrap_not_stationary_or_upright"
            return None

        if self._calibration_start_us is None:
            self._calibration_start_us = stamp
        self._last_timestamp_us = stamp
        self._calibration_count += 1
        for i in range(3):
            self._calibration_accel_sum[i] += accel[i]
            self._calibration_accel_sq_sum[i] += accel[i] * accel[i]
            self._calibration_gyro_sum[i] += gyro[i]
            self._calibration_gyro_sq_sum[i] += gyro[i] * gyro[i]

        duration_s = (stamp - self._calibration_start_us) * 1e-6
        if (
            self._calibration_count < cfg.calibration_min_samples
            or duration_s < cfg.calibration_min_duration_s
        ):
            self.last_rejection_reason = "calibrating"
            return None

        accel_mean, accel_std = _mean_and_std(
            self._calibration_accel_sum,
            self._calibration_accel_sq_sum,
            self._calibration_count,
        )
        gyro_mean, gyro_std = _mean_and_std(
            self._calibration_gyro_sum,
            self._calibration_gyro_sq_sum,
            self._calibration_count,
        )
        if (
            max(accel_std) > cfg.calibration_accel_std_max_mps2
            or max(gyro_std) > cfg.calibration_gyro_std_max_rad_s
        ):
            # Start a fresh window at this sample.  A stale noisy prefix should
            # not prevent eventual calibration after the vehicle settles.
            self._clear_calibration_window()
            self._calibration_start_us = stamp
            self._last_timestamp_us = stamp
            self._calibration_count = 1
            for i in range(3):
                self._calibration_accel_sum[i] = accel[i]
                self._calibration_accel_sq_sum[i] = accel[i] * accel[i]
                self._calibration_gyro_sum[i] = gyro[i]
                self._calibration_gyro_sq_sum[i] = gyro[i] * gyro[i]
            self.last_rejection_reason = "bootstrap_variance_too_high"
            return None

        roll, pitch = _tilt_from_specific_force(accel_mean)
        q0 = Quaternion.from_euler(roll, pitch, self.initial_yaw_rad)
        self._q = _normalize_quaternion((q0.w, q0.x, q0.y, q0.z))
        self._gyro_bias = [gyro_mean[0], gyro_mean[1], gyro_mean[2]]
        self._ready = True
        self.last_rejection_reason = None

        estimate = self._make_estimate(
            timestamp_us=stamp,
            body_rates=_subtract(gyro, self._gyro_bias),
            accel_trust=1.0,
            healthy=True,
            propagated=False,
        )
        self._last_estimate = estimate
        return estimate

    def _make_estimate(
        self,
        *,
        timestamp_us: int,
        body_rates: Vector3,
        accel_trust: float,
        healthy: bool,
        propagated: bool,
        reason: Optional[str] = None,
        accel_magnitude_deviation_mps2: float = 0.0,
        horizontal_specific_force_mps2: float = 0.0,
    ) -> AttitudeEstimate:
        return AttitudeEstimate(
            timestamp_us=timestamp_us,
            orientation=_quaternion_object(self._q),
            body_rates=body_rates,
            gyro_bias=tuple(self._gyro_bias),  # type: ignore[arg-type]
            accel_trust=float(accel_trust),
            healthy=healthy,
            propagated=propagated,
            reason=reason,
            accel_magnitude_deviation_mps2=float(accel_magnitude_deviation_mps2),
            horizontal_specific_force_mps2=float(horizontal_specific_force_mps2),
        )

    def _clear_calibration_window(self) -> None:
        self._calibration_start_us = None
        self._calibration_count = 0
        for values in (
            self._calibration_accel_sum,
            self._calibration_accel_sq_sum,
            self._calibration_gyro_sum,
            self._calibration_gyro_sq_sum,
        ):
            values[:] = [0.0, 0.0, 0.0]

    def _reject(self, reason: str) -> None:
        self.rejected_samples += 1
        self.last_rejection_reason = reason


def _validate_config(cfg: ImuAttitudeConfig) -> None:
    finite_nonnegative = {
        "calibration_min_duration_s": cfg.calibration_min_duration_s,
        "stationary_accel_tolerance_mps2": cfg.stationary_accel_tolerance_mps2,
        "stationary_gyro_max_rad_s": cfg.stationary_gyro_max_rad_s,
        "calibration_accel_std_max_mps2": cfg.calibration_accel_std_max_mps2,
        "calibration_gyro_std_max_rad_s": cfg.calibration_gyro_std_max_rad_s,
        "bootstrap_max_tilt_rad": cfg.bootstrap_max_tilt_rad,
        "gravity_correction_kp": cfg.gravity_correction_kp,
        "gyro_bias_ki": cfg.gyro_bias_ki,
        "gyro_bias_limit_rad_s": cfg.gyro_bias_limit_rad_s,
        "bias_learning_gyro_max_rad_s": cfg.bias_learning_gyro_max_rad_s,
        "accel_trust_full_deviation_mps2": cfg.accel_trust_full_deviation_mps2,
        "accel_trust_fh_full_mps2": cfg.accel_trust_fh_full_mps2,
        "accel_innovation_full_rad": cfg.accel_innovation_full_rad,
    }
    if not math.isfinite(cfg.gravity_mps2) or cfg.gravity_mps2 <= 0.0:
        raise ValueError("gravity_mps2 must be finite and > 0")
    if cfg.calibration_min_samples < 1:
        raise ValueError("calibration_min_samples must be >= 1")
    for name, value in finite_nonnegative.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    if not 0.0 <= cfg.bias_learning_min_accel_trust <= 1.0:
        raise ValueError("bias_learning_min_accel_trust must be in [0, 1]")
    if cfg.bootstrap_max_tilt_rad <= 0.0 or cfg.bootstrap_max_tilt_rad > math.pi / 2:
        raise ValueError("bootstrap_max_tilt_rad must be in (0, pi/2]")
    if (
        not math.isfinite(cfg.accel_trust_zero_deviation_mps2)
        or cfg.accel_trust_zero_deviation_mps2
        <= cfg.accel_trust_full_deviation_mps2
    ):
        raise ValueError("accel trust zero threshold must exceed full threshold")
    if (
        not math.isfinite(cfg.accel_trust_fh_zero_mps2)
        or cfg.accel_trust_fh_zero_mps2 <= cfg.accel_trust_fh_full_mps2
    ):
        raise ValueError("accel trust fh zero threshold must exceed full threshold")
    if (
        not math.isfinite(cfg.accel_innovation_zero_rad)
        or cfg.accel_innovation_zero_rad <= cfg.accel_innovation_full_rad
    ):
        raise ValueError("accel innovation zero threshold must exceed full threshold")
    if not math.isfinite(cfg.max_dt_s) or cfg.max_dt_s <= 0.0:
        raise ValueError("max_dt_s must be finite and > 0")
    if cfg.timestamp_reset_threshold_us < 1:
        raise ValueError("timestamp_reset_threshold_us must be >= 1")


def _finite_vector3(values: Sequence[float], name: str) -> Vector3:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = (float(values[0]), float(values[1]), float(values[2]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _tilt_from_specific_force(accel: Vector3) -> Tuple[float, float]:
    """Return FRD/NED roll and pitch from a stationary specific-force vector."""

    ax, ay, az = accel
    roll = math.atan2(-ay, -az)
    pitch = math.atan2(ax, math.hypot(ay, az))
    return roll, pitch


def _predicted_specific_force_direction(q: _QuaternionTuple) -> Vector3:
    """Stationary accelerometer direction predicted by q(body->NED)."""

    w, x, y, z = q
    # Negative third row of R_body_to_ned: R^T * [0, 0, -1].
    return (
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    )


def _rotate_body_to_world(q: _QuaternionTuple, v: Vector3) -> Vector3:
    """Rotate a body-frame vector into the world frame by q(body->world)."""

    w, x, y, z = q
    # v' = v + 2*w*(q_vec x v) + 2*(q_vec x (q_vec x v)).
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def _integrate_body_rate(
    q: _QuaternionTuple,
    body_rate: Vector3,
    dt: float,
) -> _QuaternionTuple:
    """Right-multiply q(body->world) by an exact body-frame increment."""

    rate_norm = _norm(body_rate)
    half_angle = 0.5 * rate_norm * dt
    if rate_norm < 1e-12:
        dq = (1.0, 0.0, 0.0, 0.0)
    else:
        scale = math.sin(half_angle) / rate_norm
        dq = (
            math.cos(half_angle),
            body_rate[0] * scale,
            body_rate[1] * scale,
            body_rate[2] * scale,
        )
    return _normalize_quaternion(_quat_multiply(q, dq))


def _quat_multiply(a: _QuaternionTuple, b: _QuaternionTuple) -> _QuaternionTuple:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _normalize_quaternion(q: _QuaternionTuple) -> _QuaternionTuple:
    magnitude = math.sqrt(sum(value * value for value in q))
    if not math.isfinite(magnitude) or magnitude < 1e-12:
        raise ValueError("attitude quaternion became invalid")
    return tuple(value / magnitude for value in q)  # type: ignore[return-value]


def _quaternion_object(q: _QuaternionTuple) -> Quaternion:
    return Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])


def _mean_and_std(
    sums: Sequence[float],
    square_sums: Sequence[float],
    count: int,
) -> Tuple[Vector3, Vector3]:
    mean = tuple(value / count for value in sums)
    variance = tuple(
        max(0.0, square_sums[i] / count - mean[i] * mean[i])
        for i in range(3)
    )
    std = tuple(math.sqrt(value) for value in variance)
    return mean, std  # type: ignore[return-value]


def _descending_ramp(value: float, full: float, zero: float) -> float:
    if value <= full:
        return 1.0
    if value >= zero:
        return 0.0
    return (zero - value) / (zero - full)


def _angle_between(a: Vector3, b: Vector3) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na < 1e-12 or nb < 1e-12:
        return math.pi
    cosine = _clamp(_dot(a, b) / (na * nb), -1.0, 1.0)
    return math.acos(cosine)


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in v))


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _scale(v: Sequence[float], scalar: float) -> Vector3:
    return (v[0] * scalar, v[1] * scalar, v[2] * scalar)


def _subtract(a: Sequence[float], b: Sequence[float]) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
