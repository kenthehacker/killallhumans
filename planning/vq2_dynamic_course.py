"""Pure dynamic image-space estimation and rolling two-gate guidance.

This module is intentionally separated from the live runner.  It contains no
transport, race-authority, arming, reset, or cleanup behavior.  Callers provide
timestamped camera, IMU, and accepted-command samples; authoritative gate
promotion is an explicit input.

The estimator keeps a stable reference for every gate track.  Camera centers
are converted to rays, rotated through the capture-time attitude, and expressed
in that per-gate reference before a bounded alpha-beta update.  Raw image
motion is also split into the component predicted by attitude rotation and a
residual that can be attributed to relative translation.
"""

from __future__ import annotations

import bisect
import math
import re
import statistics
from dataclasses import dataclass

from competition.vq2_contracts import FrameEdge


Quaternion = tuple[float, float, float, float]
Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]

MAX_TARGET_ROLL_RAD = 0.16
MIN_TARGET_PITCH_RAD = -0.35
MAX_TARGET_PITCH_RAD = 0.15
MAX_YAW_RATE_RAD_S = 0.15
MIN_THRUST = 0.21
MAX_THRUST = 0.32
SUPPORT_THRUST = 0.275

_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_NS_PER_SECOND = 1_000_000_000
_EPSILON = 1e-12


class DynamicCourseError(ValueError):
    """Raised when dynamic-course inputs cannot be used safely."""


def _finite(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicCourseError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise DynamicCourseError(f"{label} must be positive")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise DynamicCourseError(f"{label} must be nonnegative")
    return result


def _exact_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0:
        raise DynamicCourseError(f"{label} must be nonnegative")
    return value


def _token(value: object, label: str) -> str:
    if type(value) is not str or _TOKEN_RE.fullmatch(value) is None:
        raise DynamicCourseError(f"{label} must be a bounded token")
    return value


def _tuple2(
    value: object,
    label: str,
    *,
    positive: bool = False,
    bound: float | None = None,
) -> Vector2:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{label} must be an exact 2-tuple")
    result = (_finite(value[0], f"{label}[0]"), _finite(value[1], f"{label}[1]"))
    if positive and any(item <= 0.0 for item in result):
        raise DynamicCourseError(f"{label} entries must be positive")
    if bound is not None and any(abs(item) > bound for item in result):
        raise DynamicCourseError(f"{label} entries must remain within +/-{bound}")
    return result


def _tuple3(value: object, label: str, *, bound: float | None = None) -> Vector3:
    if type(value) is not tuple or len(value) != 3:
        raise TypeError(f"{label} must be an exact 3-tuple")
    result = tuple(
        _finite(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    if bound is not None and any(abs(item) > bound for item in result):
        raise DynamicCourseError(f"{label} entries must remain within +/-{bound}")
    return result  # type: ignore[return-value]


def _unit_quaternion(value: object, label: str) -> Quaternion:
    if type(value) is not tuple or len(value) != 4:
        raise TypeError(f"{label} must be an exact 4-tuple")
    result = tuple(
        _finite(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    norm = math.sqrt(sum(item * item for item in result))
    if norm <= _EPSILON:
        raise DynamicCourseError(f"{label} must have nonzero length")
    if abs(norm - 1.0) > 1e-6:
        raise DynamicCourseError(f"{label} must be unit length")
    return tuple(item / norm for item in result)  # type: ignore[return-value]


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _quat_conjugate(value: Quaternion) -> Quaternion:
    return (value[0], -value[1], -value[2], -value[3])


def _quat_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_rotate(quaternion: Quaternion, vector: Vector3) -> Vector3:
    pure: Quaternion = (0.0, vector[0], vector[1], vector[2])
    rotated = _quat_multiply(
        _quat_multiply(quaternion, pure),
        _quat_conjugate(quaternion),
    )
    return (rotated[1], rotated[2], rotated[3])


def _quat_slerp(left: Quaternion, right: Quaternion, fraction: float) -> Quaternion:
    dot = sum(a * b for a, b in zip(left, right))
    adjusted = right
    if dot < 0.0:
        dot = -dot
        adjusted = tuple(-item for item in right)  # type: ignore[assignment]
    dot = _clamp(dot, -1.0, 1.0)
    if dot > 0.9995:
        blended = tuple(
            left[index] + fraction * (adjusted[index] - left[index])
            for index in range(4)
        )
        norm = math.sqrt(sum(item * item for item in blended))
        return tuple(item / norm for item in blended)  # type: ignore[return-value]
    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    left_weight = math.sin((1.0 - fraction) * angle) / sin_angle
    right_weight = math.sin(fraction * angle) / sin_angle
    return tuple(
        left_weight * left[index] + right_weight * adjusted[index]
        for index in range(4)
    )  # type: ignore[return-value]


def _normalise_vector(vector: Vector3) -> Vector3:
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= _EPSILON:
        raise DynamicCourseError("camera ray has zero length")
    return tuple(item / norm for item in vector)  # type: ignore[return-value]


def _camera_ray(center_norm: Vector2, horizontal_scale: float, vertical_scale: float) -> Vector3:
    return _normalise_vector(
        (
            1.0,
            center_norm[0] * horizontal_scale,
            center_norm[1] * vertical_scale,
        )
    )


def _ray_bearing(ray: Vector3) -> Vector2:
    horizontal = math.atan2(ray[1], ray[0])
    # The calibrated camera model is gnomonic in each image axis.  Keeping the
    # inverse exactly paired with ``_camera_ray`` matters for coupled rotations.
    vertical = math.atan2(ray[2], ray[0])
    return (horizontal, vertical)


def _bearing_ray(bearing: Vector2) -> Vector3:
    return _normalise_vector(
        (1.0, math.tan(bearing[0]), math.tan(bearing[1]))
    )


@dataclass(frozen=True, slots=True)
class ImuAttitudeSample:
    """Capture-clock attitude and angular velocity.

    ``body_to_reference_wxyz`` is an active rotation from body FRD into the
    stable NED-like reference.  Camera rays first pass through the explicit
    camera-to-body calibration in :class:`DynamicCourseConfig`.  Host
    monotonic time is the only time domain used by this module.
    """

    monotonic_ns: int
    body_to_reference_wxyz: Quaternion
    body_rates_rad_s: Vector3
    attitude_uncertainty_rad: float = 0.01
    source_timestamp_us: int | None = None
    host_clock_id: str = "host-monotonic"

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.monotonic_ns, "monotonic_ns")
        object.__setattr__(
            self,
            "body_to_reference_wxyz",
            _unit_quaternion(
                self.body_to_reference_wxyz,
                "body_to_reference_wxyz",
            ),
        )
        object.__setattr__(
            self,
            "body_rates_rad_s",
            _tuple3(self.body_rates_rad_s, "body_rates_rad_s", bound=20.0),
        )
        object.__setattr__(
            self,
            "attitude_uncertainty_rad",
            _nonnegative(
                self.attitude_uncertainty_rad,
                "attitude_uncertainty_rad",
            ),
        )
        if self.source_timestamp_us is not None:
            _exact_nonnegative_int(self.source_timestamp_us, "source_timestamp_us")
        _token(self.host_clock_id, "host_clock_id")


@dataclass(frozen=True, slots=True)
class AppliedCommandSample:
    """A command accepted by the live transport at a host-monotonic time."""

    monotonic_ns: int
    target_roll_rad: float
    target_pitch_rad: float
    yaw_rate_rad_s: float
    thrust: float
    roll_rate_rad_s: float | None = None
    pitch_rate_rad_s: float | None = None
    host_clock_id: str = "host-monotonic"

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.monotonic_ns, "monotonic_ns")
        roll = _finite(self.target_roll_rad, "target_roll_rad")
        pitch = _finite(self.target_pitch_rad, "target_pitch_rad")
        yaw = _finite(self.yaw_rate_rad_s, "yaw_rate_rad_s")
        thrust = _finite(self.thrust, "thrust")
        if abs(roll) > MAX_TARGET_ROLL_RAD + _EPSILON:
            raise DynamicCourseError("target_roll_rad exceeds the retained envelope")
        if not MIN_TARGET_PITCH_RAD - _EPSILON <= pitch <= MAX_TARGET_PITCH_RAD + _EPSILON:
            raise DynamicCourseError("target_pitch_rad exceeds the retained envelope")
        if abs(yaw) > MAX_YAW_RATE_RAD_S + _EPSILON:
            raise DynamicCourseError("yaw_rate_rad_s exceeds measured +/-0.15 authority")
        if not MIN_THRUST - _EPSILON <= thrust <= MAX_THRUST + _EPSILON:
            raise DynamicCourseError("thrust exceeds the retained envelope")
        for name in ("roll_rate_rad_s", "pitch_rate_rad_s"):
            value = getattr(self, name)
            if value is not None:
                rate = _finite(value, name)
                if abs(rate) > 0.25 + _EPSILON:
                    raise DynamicCourseError(f"{name} exceeds the retained +/-0.25 envelope")
                object.__setattr__(self, name, rate)
        _token(self.host_clock_id, "host_clock_id")
        object.__setattr__(self, "target_roll_rad", roll)
        object.__setattr__(self, "target_pitch_rad", pitch)
        object.__setattr__(self, "yaw_rate_rad_s", yaw)
        object.__setattr__(self, "thrust", thrust)


@dataclass(frozen=True, slots=True)
class GateObservation:
    """One track observation in raw normalized image coordinates."""

    track_id: str
    frame_sequence: int
    observation_monotonic_ns: int
    center_norm: Vector2 | None
    log_scale: float | None
    aperture_half_size_norm: Vector2 | None
    clipping: FrameEdge = FrameEdge.NONE
    center_censored: bool = False
    visible: bool = True
    ambiguous: bool = False
    confidence: float = 1.0
    measurement_std: tuple[float, float, float] = (0.02, 0.02, 0.05)
    capture_monotonic_ns: int | None = None
    timing_basis: str = "final-packet-minus-configured-delay"
    timing_uncertainty_s: float = 0.020
    stream_generation: int = 0
    host_clock_id: str = "host-monotonic"

    def __post_init__(self) -> None:
        _token(self.track_id, "track_id")
        _exact_nonnegative_int(self.frame_sequence, "frame_sequence")
        _exact_nonnegative_int(
            self.observation_monotonic_ns,
            "observation_monotonic_ns",
        )
        if self.capture_monotonic_ns is not None:
            _exact_nonnegative_int(
                self.capture_monotonic_ns,
                "capture_monotonic_ns",
            )
            if self.capture_monotonic_ns > self.observation_monotonic_ns:
                raise DynamicCourseError(
                    "capture_monotonic_ns cannot follow packet observation time"
                )
        _token(self.timing_basis, "timing_basis")
        _token(self.host_clock_id, "host_clock_id")
        _exact_nonnegative_int(self.stream_generation, "stream_generation")
        object.__setattr__(
            self,
            "timing_uncertainty_s",
            _nonnegative(self.timing_uncertainty_s, "timing_uncertainty_s"),
        )
        if type(self.clipping) is not FrameEdge:
            try:
                clipping = FrameEdge(self.clipping)
            except (TypeError, ValueError) as error:
                raise DynamicCourseError("clipping must be a FrameEdge") from error
            object.__setattr__(self, "clipping", clipping)
        if type(self.center_censored) is not bool:
            raise TypeError("center_censored must be bool")
        if type(self.visible) is not bool:
            raise TypeError("visible must be bool")
        if type(self.ambiguous) is not bool:
            raise TypeError("ambiguous must be bool")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise DynamicCourseError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if type(self.measurement_std) is not tuple or len(self.measurement_std) != 3:
            raise TypeError("measurement_std must be an exact 3-tuple")
        measurement_std = tuple(
            _positive(item, f"measurement_std[{index}]")
            for index, item in enumerate(self.measurement_std)
        )
        object.__setattr__(self, "measurement_std", measurement_std)
        if self.visible:
            if self.center_norm is None or self.log_scale is None:
                raise DynamicCourseError(
                    "visible observations require center_norm and log_scale"
                )
            center = _tuple2(self.center_norm, "center_norm", bound=2.5)
            scale = _finite(self.log_scale, "log_scale")
            if abs(scale) > 12.0:
                raise DynamicCourseError("log_scale must remain within +/-12")
            object.__setattr__(self, "center_norm", center)
            object.__setattr__(self, "log_scale", scale)
            if self.aperture_half_size_norm is not None:
                aperture = _tuple2(
                    self.aperture_half_size_norm,
                    "aperture_half_size_norm",
                    positive=True,
                    bound=2.0,
                )
                object.__setattr__(self, "aperture_half_size_norm", aperture)
        elif any(
            value is not None
            for value in (
                self.center_norm,
                self.log_scale,
                self.aperture_half_size_norm,
            )
        ):
            raise DynamicCourseError(
                "invisible observations must not fabricate image measurements"
            )

    @property
    def censored_axes(self) -> tuple[bool, bool]:
        horizontal = self.center_censored or bool(
            self.clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)
        )
        vertical = self.center_censored or bool(
            self.clipping & (FrameEdge.TOP | FrameEdge.BOTTOM)
        )
        return (horizontal, vertical)


@dataclass(frozen=True, slots=True)
class DynamicCourseConfig:
    """Small identified-model and guidance tuning surface."""

    camera_delay_s: float = 0.045
    # Cohort yaw/image response: about 0.63 normalized-x per radian.
    horizontal_angle_scale_rad: float = 1.59
    vertical_angle_scale_rad: float = 0.55
    camera_to_body_wxyz: Quaternion = (1.0, 0.0, 0.0, 0.0)
    max_imu_alignment_gap_s: float = 0.080
    max_capture_timing_uncertainty_s: float = 0.050
    bearing_alpha: float = 0.58
    bearing_beta: float = 0.20
    scale_alpha: float = 0.55
    scale_beta: float = 0.20
    residual_alpha: float = 0.35
    # A raw two-frame image difference is not a qualified translation rate.
    # Require a short odd window so a one-frame contour-completion jump cannot
    # seed collective or crossing guidance before temporal consistency exists.
    residual_rate_median_window: int = 3
    max_bearing_innovation_rad: float = 0.30
    max_log_scale_innovation: float = 0.45
    max_abs_bearing_rad: float = 1.40
    max_abs_bearing_rate_rad_s: float = 3.0
    max_abs_expansion_rate_s: float = 5.0
    process_noise_bearing_rad_s: float = 0.06
    process_noise_scale_s: float = 0.12
    clipping_uncertainty_multiplier: float = 1.8
    yaw_command_delay_s: float = 0.040
    roll_command_delay_s: float = 0.110
    pitch_command_delay_s: float = 0.100
    thrust_command_delay_s: float = 0.080
    # The cohort roll/image fit is confounded.  Authority remains disabled
    # until an isolated characterization injects a signed gain.
    roll_to_lateral_bearing_accel: float = 0.0
    roll_guidance_sign: float = 0.0
    pitch_to_expansion_accel: float = 2.5
    thrust_to_vertical_bearing_accel: float = 1.5
    minimum_ttc_s: float = 0.15
    maximum_ttc_s: float = 8.0
    passage_margin_norm: float = 0.09
    # The detector bbox is outer contour support, not a measured inner
    # aperture.  These scale-free occupancy ceilings are an empirical safety
    # envelope after the 8c31d1f Gate-0 top-frame contact; they deliberately
    # make no claim about unverified vehicle or gate dimensions.
    crossing_max_occupancy_q: Vector2 = (0.50, 0.45)
    vertical_settled_rate_norm_s: float = 0.30
    passage_arm_min_log_scale: float = -0.80
    # After the requested current-aperture collective can reach the wire,
    # retain one bounded contact window before predicted plane crossing.
    terminal_min_post_governor_contact_budget_s: float = 0.12
    passage_successor_bias: float = 0.55
    successor_passage_far_authority: float = 0.25
    successor_passage_full_confidence: float = 0.50
    # Successor guidance cannot accrue while the current aperture is unsafe.
    # Once both centered crossing clearances remain positive, require a short
    # fresh-observation dwell and then release authority continuously.
    successor_clearance_dwell_s: float = 0.12
    successor_clearance_ramp_s: float = 0.20
    crossing_prediction_max_horizon_s: float = 1.20
    successor_prediction_max_horizon_s: float = 0.40
    successor_prediction_max_extrapolation_rad: float = 0.18
    successor_maximum_weight: float = 0.45
    successor_max_yaw_contribution_rad: float = 0.10
    successor_full_weight_ttc_s: float = 0.55
    successor_lookahead_ttc_s: float = 2.2
    yaw_gain: float = 1.25
    roll_gain: float = 0.22
    lateral_rate_gain: float = 0.06
    advance_pitch_rad: float = -0.12
    brake_pitch_rad: float = 0.12
    off_axis_brake_rad: float = 0.18
    rapid_expansion_rate_s: float = 0.45
    dropout_hold_s: float = 0.120
    # Geometry/yaw prediction expires quickly, but an already reviewed exact
    # successor identity survives the longer, expected near-plane occlusion.
    successor_lineage_hold_s: float = 0.350
    max_history_samples: int = 256
    def __post_init__(self) -> None:
        nonnegative = (
            "camera_delay_s",
            "yaw_command_delay_s",
            "roll_command_delay_s",
            "pitch_command_delay_s",
            "thrust_command_delay_s",
        )
        for name in nonnegative:
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        positive = (
            "horizontal_angle_scale_rad",
            "vertical_angle_scale_rad",
            "max_imu_alignment_gap_s",
            "max_capture_timing_uncertainty_s",
            "max_bearing_innovation_rad",
            "max_log_scale_innovation",
            "max_abs_bearing_rad",
            "max_abs_bearing_rate_rad_s",
            "max_abs_expansion_rate_s",
            "process_noise_bearing_rad_s",
            "process_noise_scale_s",
            "clipping_uncertainty_multiplier",
            "minimum_ttc_s",
            "maximum_ttc_s",
            "passage_margin_norm",
            "vertical_settled_rate_norm_s",
            "terminal_min_post_governor_contact_budget_s",
            "successor_clearance_dwell_s",
            "successor_clearance_ramp_s",
            "crossing_prediction_max_horizon_s",
            "successor_prediction_max_horizon_s",
            "successor_prediction_max_extrapolation_rad",
            "successor_max_yaw_contribution_rad",
            "successor_full_weight_ttc_s",
            "successor_lookahead_ttc_s",
            "yaw_gain",
            "roll_gain",
            "lateral_rate_gain",
            "brake_pitch_rad",
            "off_axis_brake_rad",
            "rapid_expansion_rate_s",
            "dropout_hold_s",
            "successor_lineage_hold_s",
        )
        for name in positive:
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        crossing_max_occupancy_q = _tuple2(
            self.crossing_max_occupancy_q,
            "crossing_max_occupancy_q",
            positive=True,
            bound=1.0,
        )
        object.__setattr__(
            self,
            "crossing_max_occupancy_q",
            crossing_max_occupancy_q,
        )
        for name in (
            "bearing_alpha",
            "bearing_beta",
            "scale_alpha",
            "scale_beta",
            "residual_alpha",
            "passage_successor_bias",
            "successor_passage_far_authority",
            "successor_passage_full_confidence",
            "successor_maximum_weight",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise DynamicCourseError(f"{name} must be in (0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "roll_to_lateral_bearing_accel",
            "roll_guidance_sign",
            "pitch_to_expansion_accel",
            "thrust_to_vertical_bearing_accel",
            "passage_arm_min_log_scale",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        advance = _finite(self.advance_pitch_rad, "advance_pitch_rad")
        if not MIN_TARGET_PITCH_RAD <= advance <= 0.0:
            raise DynamicCourseError("advance_pitch_rad must be a bounded forward target")
        if self.brake_pitch_rad > MAX_TARGET_PITCH_RAD:
            raise DynamicCourseError("brake_pitch_rad exceeds pitch authority")
        if self.minimum_ttc_s >= self.maximum_ttc_s:
            raise DynamicCourseError("minimum_ttc_s must be below maximum_ttc_s")
        if self.crossing_prediction_max_horizon_s > 1.20:
            raise DynamicCourseError(
                "crossing prediction horizon exceeds its bounded model"
            )
        if self.successor_full_weight_ttc_s >= self.successor_lookahead_ttc_s:
            raise DynamicCourseError(
                "successor full-weight TTC must precede lookahead TTC"
            )
        if not (
            self.dropout_hold_s
            <= self.successor_lineage_hold_s
            <= 0.50
        ):
            raise DynamicCourseError(
                "successor lineage hold must cover prediction dropout "
                "without exceeding 0.50 seconds"
            )
        if type(self.max_history_samples) is not int:
            raise TypeError("max_history_samples must be an exact integer")
        if self.max_history_samples < 8:
            raise DynamicCourseError("max_history_samples must be at least 8")
        if type(self.residual_rate_median_window) is not int:
            raise TypeError(
                "residual_rate_median_window must be an exact integer"
            )
        if (
            self.residual_rate_median_window < 3
            or self.residual_rate_median_window > 7
            or self.residual_rate_median_window % 2 == 0
        ):
            raise DynamicCourseError(
                "residual_rate_median_window must be an odd integer in [3, 7]"
            )
        object.__setattr__(
            self,
            "camera_to_body_wxyz",
            _unit_quaternion(
                self.camera_to_body_wxyz,
                "camera_to_body_wxyz",
            ),
        )


@dataclass(frozen=True, slots=True)
class DelayedCommandView:
    """Per-channel commands effective at an image capture time."""

    capture_monotonic_ns: int
    target_roll_rad: float
    target_pitch_rad: float
    yaw_rate_rad_s: float
    thrust: float
    roll_source_monotonic_ns: int | None
    pitch_source_monotonic_ns: int | None
    yaw_source_monotonic_ns: int | None
    thrust_source_monotonic_ns: int | None


@dataclass(frozen=True, slots=True)
class TrackDynamicState:
    track_id: str
    stream_generation: int
    state_monotonic_ns: int
    last_measurement_monotonic_ns: int
    frame_sequence: int
    capture_timing_basis: str
    capture_timing_uncertainty_s: float
    raw_center_norm: Vector2 | None
    raw_log_scale: float | None
    aperture_half_size_norm: Vector2 | None
    aperture_seed_monotonic_ns: int | None
    aperture_prediction_deadline_monotonic_ns: int | None
    aperture_propagated: bool
    aperture_dynamics_qualified: bool
    bearing_rad: Vector2
    bearing_rate_rad_s: Vector2
    bearing_rate_qualified: tuple[bool, bool]
    log_scale: float
    expansion_rate_s: float
    scale_rate_qualified: bool
    predicted_rotational_rate_rad_s: Vector2
    residual_translational_rate_rad_s: Vector2
    time_to_contact_s: float | None
    reference_camera_to_world_wxyz: Quaternion
    body_to_reference_wxyz: Quaternion
    body_rates_rad_s: Vector3
    delayed_command: DelayedCommandView
    bearing_std_rad: Vector2
    rate_std_rad_s: Vector2
    log_scale_std: float
    expansion_rate_std_s: float
    clipping: FrameEdge
    censored_axes: tuple[bool, bool]
    visible: bool
    ambiguous: bool
    confidence: float
    sample_count: int
    missed_count: int


@dataclass(frozen=True, slots=True)
class TrackSteeringPrediction:
    """Bounded image-space track prediction at one control instant.

    This is a local camera/attitude projection, not a pose or world-position
    estimate.  Accepted commands propagate the stable bearing while current
    IMU attitude and body rates project its center and rate into the camera.
    """

    track_id: str
    stream_generation: int
    monotonic_ns: int
    source_state_monotonic_ns: int
    last_measurement_monotonic_ns: int
    measurement_age_s: float
    stable_bearing_rad: Vector2
    stable_bearing_rate_rad_s: Vector2
    camera_center_norm: Vector2
    camera_center_rate_norm_s: Vector2
    bearing_std_rad: Vector2
    body_rates_rad_s: Vector3


@dataclass(frozen=True, slots=True)
class DynamicCourseCommand:
    target_roll_rad: float
    target_pitch_rad: float
    yaw_rate_rad_s: float
    thrust: float

    def __post_init__(self) -> None:
        sample = AppliedCommandSample(
            monotonic_ns=0,
            target_roll_rad=self.target_roll_rad,
            target_pitch_rad=self.target_pitch_rad,
            yaw_rate_rad_s=self.yaw_rate_rad_s,
            thrust=self.thrust,
        )
        object.__setattr__(self, "target_roll_rad", sample.target_roll_rad)
        object.__setattr__(self, "target_pitch_rad", sample.target_pitch_rad)
        object.__setattr__(self, "yaw_rate_rad_s", sample.yaw_rate_rad_s)
        object.__setattr__(self, "thrust", sample.thrust)


@dataclass(frozen=True, slots=True)
class GuidanceDecision:
    monotonic_ns: int
    current_gate_index: int
    current_track_id: str
    successor_track_id: str | None
    current_center_norm: Vector2
    camera_current_center_norm: Vector2
    current_aperture_half_size_norm: Vector2 | None
    current_aperture_propagated: bool
    current_aperture_dynamics_qualified: bool
    current_aperture_prediction_age_s: float
    current_aperture_prediction_horizon_remaining_s: float
    passage_point_norm: Vector2
    successor_passage_authority: float
    centered_crossing_clearance_norm: Vector2
    successor_clearance_dwell_s: float
    successor_clearance_authority: float
    passage_error_norm: Vector2
    aperture_margin_norm: Vector2
    crossing_prediction_horizon_s: float
    current_crossing_error_q: Vector2
    crossing_rate_q_s: Vector2
    predicted_crossing_error_norm: Vector2
    predicted_crossing_std_norm: Vector2
    crossing_allowance_norm: Vector2
    crossing_swept_occupancy_norm: Vector2
    predicted_crossing_clearance_norm: Vector2
    terminal_crossing_occupancy_norm: Vector2
    terminal_crossing_clearance_norm: Vector2
    current_bearing_std_rad: Vector2
    successor_bearing_std_rad: Vector2 | None
    successor_weight: float
    predicted_successor_bearing_rad: Vector2 | None
    measured_successor_bearing_rad: Vector2 | None
    successor_rate_rad_s: Vector2 | None
    successor_prediction_horizon_s: float
    successor_prediction_confidence: float
    current_yaw_release: float
    passage_yaw_authority: float
    successor_yaw_contribution_rad: float
    successor_transition_held: bool
    current_time_to_contact_s: float | None
    braking: bool
    brake_reason: str | None
    dropout_held: bool
    proposed_command: DynamicCourseCommand
    command: DynamicCourseCommand


@dataclass(frozen=True, slots=True)
class CrossingQuotientPrediction:
    """Scale-free current-aperture crossing envelope.

    ``q`` is passage-relative center offset divided by the co-timed outer
    aperture half extent.  The robust swept occupancy also reserves the
    successor-biased passage fraction and uncertainty, so its clearance is a
    conservative vehicle/camera envelope rather than point-camera clearance.
    """

    current_error_q: Vector2
    rate_q_s: Vector2
    predicted_error_q: Vector2
    current_std_q: Vector2
    predicted_std_q: Vector2
    swept_occupancy_q: Vector2
    terminal_occupancy_q: Vector2
    allowance_q: Vector2
    clearance_q: Vector2
    terminal_clearance_q: Vector2


def predict_aperture_relative_crossing(
    *,
    center_offset_norm: Vector2,
    passage_offset_norm: Vector2,
    aperture_half_extent_norm: Vector2,
    center_rate_norm_s: Vector2,
    aperture_expansion_rate_s: Vector2,
    center_std_norm: Vector2,
    aperture_log_scale_std: float,
    capture_timing_uncertainty_s: float,
    horizon_s: float,
    allowance_q: Vector2,
) -> CrossingQuotientPrediction:
    """Predict approach-sweep and terminal crossing envelopes in q space.

    The full sweep deliberately includes the current state and remains the
    conservative ownership/braking test.  ``terminal_occupancy_q`` is the
    robust 2-sigma support at the crossing endpoint.  Its prediction standard
    deviation already includes capture/contact timing uncertainty, so it is a
    bounded terminal crossing window rather than a point-time clearance.
    """

    center = _tuple2(center_offset_norm, "center_offset_norm")
    passage = _tuple2(passage_offset_norm, "passage_offset_norm")
    aperture = _tuple2(
        aperture_half_extent_norm,
        "aperture_half_extent_norm",
        positive=True,
    )
    center_rate = _tuple2(center_rate_norm_s, "center_rate_norm_s")
    aperture_expansion = _tuple2(
        aperture_expansion_rate_s,
        "aperture_expansion_rate_s",
    )
    center_std = _tuple2(center_std_norm, "center_std_norm")
    if any(value < 0.0 for value in center_std):
        raise DynamicCourseError("center_std_norm must be nonnegative")
    scale_std = _nonnegative(
        aperture_log_scale_std,
        "aperture_log_scale_std",
    )
    timing_std = _nonnegative(
        capture_timing_uncertainty_s,
        "capture_timing_uncertainty_s",
    )
    horizon = _nonnegative(horizon_s, "horizon_s")
    allowance = _tuple2(allowance_q, "allowance_q", positive=True, bound=1.0)

    center_q = tuple(center[axis] / aperture[axis] for axis in range(2))
    passage_q = tuple(passage[axis] / aperture[axis] for axis in range(2))
    error_q = tuple(
        center_q[axis] + passage_q[axis] for axis in range(2)
    )
    # The passage fraction is fixed in the gate aperture.  Expansion acts on
    # the measured center quotient, not on the deliberately selected passage
    # offset; applying it to both would manufacture successor-guidance motion.
    rate_q = tuple(
        center_rate[axis] / aperture[axis]
        - aperture_expansion[axis] * center_q[axis]
        for axis in range(2)
    )
    predicted_q = tuple(
        error_q[axis] + rate_q[axis] * horizon for axis in range(2)
    )
    current_std_q = tuple(
        center_std[axis] / aperture[axis]
        + (
            abs(center_q[axis]) + abs(passage_q[axis])
        )
        * scale_std
        for axis in range(2)
    )
    predicted_std_q = tuple(
        current_std_q[axis] + abs(rate_q[axis]) * timing_std
        for axis in range(2)
    )
    swept_occupancy = tuple(
        abs(passage_q[axis])
        + max(
            abs(error_q[axis]) + 2.0 * current_std_q[axis],
            abs(predicted_q[axis]) + 2.0 * predicted_std_q[axis],
        )
        for axis in range(2)
    )
    terminal_occupancy = tuple(
        abs(predicted_q[axis]) + 2.0 * predicted_std_q[axis]
        for axis in range(2)
    )
    clearance = tuple(
        allowance[axis] - swept_occupancy[axis] for axis in range(2)
    )
    terminal_clearance = tuple(
        allowance[axis] - terminal_occupancy[axis] for axis in range(2)
    )
    return CrossingQuotientPrediction(
        current_error_q=error_q,  # type: ignore[arg-type]
        rate_q_s=rate_q,  # type: ignore[arg-type]
        predicted_error_q=predicted_q,  # type: ignore[arg-type]
        current_std_q=current_std_q,  # type: ignore[arg-type]
        predicted_std_q=predicted_std_q,  # type: ignore[arg-type]
        swept_occupancy_q=swept_occupancy,  # type: ignore[arg-type]
        terminal_occupancy_q=terminal_occupancy,  # type: ignore[arg-type]
        allowance_q=allowance,
        clearance_q=clearance,  # type: ignore[arg-type]
        terminal_clearance_q=terminal_clearance,  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class CourseDynamicState:
    current_gate_index: int
    current_track_id: str
    successor_track_id: str | None
    current: TrackDynamicState
    successor: TrackDynamicState | None
    recent_commands: tuple[AppliedCommandSample, ...]
    last_applied_command: DynamicCourseCommand | None
    promotion_count: int


@dataclass(slots=True)
class _TrackEstimate:
    state: TrackDynamicState
    reference_camera_to_world: Quaternion
    last_measured_raw_angle: Vector2
    last_measured_stable_ray: Vector3
    last_raw_measurement_ns: int
    last_measurement_camera_to_world: Quaternion
    last_measured_center_norm: Vector2
    last_measured_aperture_half_size_norm: Vector2 | None
    measured_aperture_history: tuple[list[float], list[float]]
    measured_residual_rate_history: tuple[list[float], list[float]]
    residual_rate_reanchor_required: list[bool]
    measured_log_scale_rate_history: list[float]
    scale_rate_reanchor_required: bool
    measured_bearing_history: list[tuple[int, Vector2]]


@dataclass(frozen=True, slots=True)
class _SuccessorPrediction:
    bearing_rad: Vector2
    measured_bearing_rad: Vector2
    robust_rate_rad_s: Vector2
    horizon_s: float
    confidence: float


class DynamicCourseCore:
    """Per-track estimator plus an authority-neutral rolling course lifecycle."""

    def __init__(self, config: DynamicCourseConfig | None = None) -> None:
        self.config = config or DynamicCourseConfig()
        self._imu: list[ImuAttitudeSample] = []
        self._commands: list[AppliedCommandSample] = []
        self._tracks: dict[str, _TrackEstimate] = {}
        self._current_gate_index: int | None = None
        self._current_track_id: str | None = None
        self._successor_track_id: str | None = None
        self._promotion_count = 0
        self._last_promotion_ns: int | None = None
        self._successor_passage_track_id: str | None = None
        self._successor_passage_authority = 0.0
        self._successor_passage_authority_ns: int | None = None
        self._successor_clearance_key: (
            tuple[int, str, str | None] | None
        ) = None
        self._successor_clearance_positive_since_ns: int | None = None
        self._last_applied_command: DynamicCourseCommand | None = None

    @property
    def track_states(self) -> tuple[TrackDynamicState, ...]:
        return tuple(
            item.state
            for _, item in sorted(self._tracks.items(), key=lambda pair: pair[0])
        )

    def record_imu(self, sample: ImuAttitudeSample) -> None:
        if self._imu and sample.monotonic_ns <= self._imu[-1].monotonic_ns:
            raise DynamicCourseError("IMU samples must advance monotonically")
        self._imu.append(sample)
        self._trim(self._imu)

    def record_applied_command(
        self,
        sample: AppliedCommandSample,
    ) -> None:
        if self._commands and sample.monotonic_ns <= self._commands[-1].monotonic_ns:
            raise DynamicCourseError("command samples must advance monotonically")
        self._commands.append(sample)
        self._trim(self._commands)
        self._last_applied_command = DynamicCourseCommand(
            target_roll_rad=sample.target_roll_rad,
            target_pitch_rad=sample.target_pitch_rad,
            yaw_rate_rad_s=sample.yaw_rate_rad_s,
            thrust=sample.thrust,
        )

    def _trim(self, values: list[object]) -> None:
        excess = len(values) - self.config.max_history_samples
        if excess > 0:
            del values[:excess]

    def delayed_command_view(self, capture_monotonic_ns: int) -> DelayedCommandView:
        _exact_nonnegative_int(capture_monotonic_ns, "capture_monotonic_ns")

        def effective(channel: str, delay_s: float, neutral: float) -> tuple[float, int | None]:
            target_ns = capture_monotonic_ns - round(delay_s * _NS_PER_SECOND)
            times = [sample.monotonic_ns for sample in self._commands]
            index = bisect.bisect_right(times, target_ns) - 1
            if index < 0:
                return (neutral, None)
            sample = self._commands[index]
            return (float(getattr(sample, channel)), sample.monotonic_ns)

        roll, roll_ns = effective("target_roll_rad", self.config.roll_command_delay_s, 0.0)
        pitch, pitch_ns = effective(
            "target_pitch_rad",
            self.config.pitch_command_delay_s,
            0.0,
        )
        yaw, yaw_ns = effective("yaw_rate_rad_s", self.config.yaw_command_delay_s, 0.0)
        thrust, thrust_ns = effective(
            "thrust",
            self.config.thrust_command_delay_s,
            SUPPORT_THRUST,
        )
        return DelayedCommandView(
            capture_monotonic_ns=capture_monotonic_ns,
            target_roll_rad=roll,
            target_pitch_rad=pitch,
            yaw_rate_rad_s=yaw,
            thrust=thrust,
            roll_source_monotonic_ns=roll_ns,
            pitch_source_monotonic_ns=pitch_ns,
            yaw_source_monotonic_ns=yaw_ns,
            thrust_source_monotonic_ns=thrust_ns,
        )

    def _aligned_imu(self, monotonic_ns: int) -> ImuAttitudeSample:
        if not self._imu:
            raise DynamicCourseError("an IMU attitude history is required")
        times = [sample.monotonic_ns for sample in self._imu]
        index = bisect.bisect_left(times, monotonic_ns)
        max_gap_ns = round(self.config.max_imu_alignment_gap_s * _NS_PER_SECOND)
        if index < len(self._imu) and self._imu[index].monotonic_ns == monotonic_ns:
            return self._imu[index]
        if 0 < index < len(self._imu):
            left = self._imu[index - 1]
            right = self._imu[index]
            if (
                monotonic_ns - left.monotonic_ns > max_gap_ns
                or right.monotonic_ns - monotonic_ns > max_gap_ns
            ):
                raise DynamicCourseError("camera capture is too far from IMU support")
            fraction = (monotonic_ns - left.monotonic_ns) / (
                right.monotonic_ns - left.monotonic_ns
            )
            rates = tuple(
                left.body_rates_rad_s[axis]
                + fraction
                * (right.body_rates_rad_s[axis] - left.body_rates_rad_s[axis])
                for axis in range(3)
            )
            return ImuAttitudeSample(
                monotonic_ns=monotonic_ns,
                body_to_reference_wxyz=_quat_slerp(
                    left.body_to_reference_wxyz,
                    right.body_to_reference_wxyz,
                    fraction,
                ),
                body_rates_rad_s=rates,  # type: ignore[arg-type]
                attitude_uncertainty_rad=max(
                    left.attitude_uncertainty_rad,
                    right.attitude_uncertainty_rad,
                ),
            )
        nearest = self._imu[0] if index == 0 else self._imu[-1]
        if abs(nearest.monotonic_ns - monotonic_ns) > max_gap_ns:
            raise DynamicCourseError("camera capture is outside bounded IMU support")
        return ImuAttitudeSample(
            monotonic_ns=monotonic_ns,
            body_to_reference_wxyz=nearest.body_to_reference_wxyz,
            body_rates_rad_s=nearest.body_rates_rad_s,
            attitude_uncertainty_rad=nearest.attitude_uncertainty_rad,
        )

    def observe_track(self, observation: GateObservation) -> TrackDynamicState:
        if (
            observation.timing_uncertainty_s
            > self.config.max_capture_timing_uncertainty_s
        ):
            raise DynamicCourseError(
                "camera timing uncertainty exceeds the bounded derotation envelope"
            )
        capture_ns = (
            observation.capture_monotonic_ns
            if observation.capture_monotonic_ns is not None
            else observation.observation_monotonic_ns
            - round(self.config.camera_delay_s * _NS_PER_SECOND)
        )
        if capture_ns < 0:
            raise DynamicCourseError("camera delay precedes the host clock epoch")
        aligned = self._aligned_imu(capture_ns)
        existing = self._tracks.get(observation.track_id)
        if existing is not None:
            if observation.stream_generation != existing.state.stream_generation:
                raise DynamicCourseError(
                    "track identifiers cannot cross stream/reset generations"
                )
            if observation.frame_sequence <= existing.state.frame_sequence:
                raise DynamicCourseError("track frame sequence must advance")
            if capture_ns <= existing.state.state_monotonic_ns:
                raise DynamicCourseError("track capture time must advance")
        delayed = self.delayed_command_view(capture_ns)
        camera_to_world = _quat_multiply(
            aligned.body_to_reference_wxyz,
            self.config.camera_to_body_wxyz,
        )
        if not observation.visible:
            if existing is None:
                raise DynamicCourseError("an invisible observation cannot initialise a track")
            for history in existing.measured_aperture_history:
                history.clear()
            for history in existing.measured_residual_rate_history:
                history.clear()
            existing.residual_rate_reanchor_required[:] = [True, True]
            existing.measured_log_scale_rate_history.clear()
            existing.scale_rate_reanchor_required = True
            return self._coast_track(
                existing,
                observation,
                capture_ns,
                aligned,
                delayed,
            )
        assert observation.center_norm is not None
        assert observation.log_scale is not None
        ray_camera = _camera_ray(
            observation.center_norm,
            self.config.horizontal_angle_scale_rad,
            self.config.vertical_angle_scale_rad,
        )
        ray_world = _quat_rotate(camera_to_world, ray_camera)
        reference = (
            camera_to_world
            if existing is None
            else existing.reference_camera_to_world
        )
        stable_ray = _normalise_vector(
            _quat_rotate(_quat_conjugate(reference), ray_world)
        )
        measured_bearing = _ray_bearing(stable_ray)
        raw_angle = _ray_bearing(ray_camera)
        if existing is None:
            state = self._initial_state(
                observation,
                capture_ns,
                aligned,
                delayed,
                measured_bearing,
                reference,
            )
            self._tracks[observation.track_id] = _TrackEstimate(
                state=state,
                reference_camera_to_world=reference,
                last_measured_raw_angle=raw_angle,
                last_measured_stable_ray=stable_ray,
                last_raw_measurement_ns=capture_ns,
                last_measurement_camera_to_world=camera_to_world,
                last_measured_center_norm=observation.center_norm,
                last_measured_aperture_half_size_norm=(
                    observation.aperture_half_size_norm
                    if (
                        observation.aperture_half_size_norm is not None
                        and not observation.ambiguous
                        and not any(observation.censored_axes)
                    )
                    else None
                ),
                measured_aperture_history=(
                    (
                        []
                        if (
                            observation.aperture_half_size_norm is None
                            or observation.ambiguous
                            or observation.censored_axes[0]
                        )
                        else [observation.aperture_half_size_norm[0]]
                    ),
                    (
                        []
                        if (
                            observation.aperture_half_size_norm is None
                            or observation.ambiguous
                            or observation.censored_axes[1]
                        )
                        else [observation.aperture_half_size_norm[1]]
                    ),
                ),
                measured_residual_rate_history=([], []),
                residual_rate_reanchor_required=[
                    bool(
                        observation.ambiguous
                        or observation.censored_axes[axis]
                    )
                    for axis in range(2)
                ],
                measured_log_scale_rate_history=[],
                scale_rate_reanchor_required=bool(
                    observation.ambiguous
                    or observation.clipping != FrameEdge.NONE
                ),
                measured_bearing_history=(
                    [(capture_ns, measured_bearing)]
                    if (
                        not observation.ambiguous
                        and not observation.censored_axes[0]
                    )
                    else []
                ),
            )
            return state
        rotational_rate, residual_rate = self._split_image_rate(
            existing,
            camera_to_world,
            raw_angle,
            capture_ns,
        )
        dt = (
            capture_ns - existing.last_raw_measurement_ns
        ) / _NS_PER_SECOND
        rate_gap = dt > self.config.dropout_hold_s
        robust_residual_rate: list[float | None] = []
        for axis in range(2):
            history = existing.measured_residual_rate_history[axis]
            if (
                rate_gap
                or observation.ambiguous
                or observation.censored_axes[axis]
            ):
                history.clear()
                existing.residual_rate_reanchor_required[axis] = True
                robust_residual_rate.append(None)
                continue
            if existing.residual_rate_reanchor_required[axis]:
                existing.residual_rate_reanchor_required[axis] = False
                robust_residual_rate.append(None)
                continue
            history.append(residual_rate[axis])
            del history[
                : max(
                    0,
                    len(history)
                    - self.config.residual_rate_median_window,
                )
            ]
            robust_residual_rate.append(
                None
                if len(history)
                < self.config.residual_rate_median_window
                else float(statistics.median(history))
            )
        robust_expansion_rate: float | None = None
        aperture_measurement_usable = bool(
            observation.aperture_half_size_norm is not None
            and not observation.ambiguous
            and not any(observation.censored_axes)
            and observation.clipping == FrameEdge.NONE
        )
        if (
            rate_gap
            or not aperture_measurement_usable
        ):
            existing.measured_log_scale_rate_history.clear()
            existing.scale_rate_reanchor_required = True
        elif existing.scale_rate_reanchor_required:
            existing.scale_rate_reanchor_required = False
        elif existing.state.raw_log_scale is not None:
            scale_history = existing.measured_log_scale_rate_history
            scale_history.append(
                (observation.log_scale - existing.state.raw_log_scale)
                / dt
            )
            del scale_history[
                : max(
                    0,
                    len(scale_history)
                    - self.config.residual_rate_median_window,
                )
            ]
            if (
                len(scale_history)
                >= self.config.residual_rate_median_window
            ):
                robust_expansion_rate = float(
                    statistics.median(scale_history)
                )
        stabilized_aperture = (
            existing.last_measured_aperture_half_size_norm
            if observation.aperture_half_size_norm is not None
            else None
        )
        if rate_gap or observation.ambiguous:
            for history in existing.measured_aperture_history:
                history.clear()
        if (
            observation.aperture_half_size_norm is not None
            and not observation.ambiguous
            and not any(observation.censored_axes)
        ):
            aperture_values = list(
                observation.aperture_half_size_norm
                if stabilized_aperture is None
                else stabilized_aperture
            )
            for axis in range(2):
                if observation.ambiguous or observation.censored_axes[axis]:
                    continue
                aperture_history = existing.measured_aperture_history[axis]
                aperture_history.append(
                    observation.aperture_half_size_norm[axis]
                )
                del aperture_history[
                    : max(
                        0,
                        len(aperture_history)
                        - self.config.residual_rate_median_window,
                    )
                ]
                aperture_values[axis] = float(
                    statistics.median(aperture_history)
                )
            stabilized_aperture = (
                aperture_values[0],
                aperture_values[1],
            )
        state = self._update_track(
            existing.state,
            observation,
            capture_ns,
            aligned,
            delayed,
            measured_bearing,
            rotational_rate,
            (robust_residual_rate[0], robust_residual_rate[1]),
            robust_expansion_rate,
            stabilized_aperture,
        )
        existing.state = state
        existing.last_measured_raw_angle = raw_angle
        existing.last_measured_stable_ray = stable_ray
        existing.last_raw_measurement_ns = capture_ns
        if (
            observation.aperture_half_size_norm is not None
            and not observation.ambiguous
            and not any(observation.censored_axes)
        ):
            # These fields are one coherent aperture anchor.  Outer-support
            # fallback publications may update bearing/rates, but cannot move
            # the aperture seed independently of its measured corners.
            existing.last_measurement_camera_to_world = camera_to_world
            existing.last_measured_center_norm = observation.center_norm
            existing.last_measured_aperture_half_size_norm = (
                stabilized_aperture
            )
        if observation.ambiguous or observation.censored_axes[0]:
            existing.measured_bearing_history.clear()
        else:
            existing.measured_bearing_history.append(
                (capture_ns, measured_bearing)
            )
            history_start_ns = capture_ns - 500_000_000
            existing.measured_bearing_history[:] = [
                sample
                for sample in existing.measured_bearing_history[-12:]
                if sample[0] >= history_start_ns
            ]
        return state

    def _initial_state(
        self,
        observation: GateObservation,
        capture_ns: int,
        aligned: ImuAttitudeSample,
        delayed: DelayedCommandView,
        bearing: Vector2,
        reference: Quaternion,
    ) -> TrackDynamicState:
        assert observation.center_norm is not None
        assert observation.log_scale is not None
        clean_aperture = bool(
            observation.aperture_half_size_norm is not None
            and not observation.ambiguous
            and not any(observation.censored_axes)
        )
        std_x = observation.measurement_std[0] * self.config.horizontal_angle_scale_rad
        std_y = observation.measurement_std[1] * self.config.vertical_angle_scale_rad
        return TrackDynamicState(
            track_id=observation.track_id,
            stream_generation=observation.stream_generation,
            state_monotonic_ns=capture_ns,
            last_measurement_monotonic_ns=capture_ns,
            frame_sequence=observation.frame_sequence,
            capture_timing_basis=observation.timing_basis,
            capture_timing_uncertainty_s=observation.timing_uncertainty_s,
            raw_center_norm=observation.center_norm,
            raw_log_scale=(
                observation.log_scale if clean_aperture else None
            ),
            aperture_half_size_norm=(
                observation.aperture_half_size_norm
                if clean_aperture
                else None
            ),
            aperture_seed_monotonic_ns=(
                capture_ns
                if clean_aperture
                else None
            ),
            aperture_prediction_deadline_monotonic_ns=(
                capture_ns
                + round(
                    self.config.crossing_prediction_max_horizon_s
                    * _NS_PER_SECOND
                )
                if clean_aperture
                else None
            ),
            aperture_propagated=False,
            aperture_dynamics_qualified=False,
            bearing_rad=bearing,
            bearing_rate_rad_s=(0.0, 0.0),
            bearing_rate_qualified=(False, False),
            log_scale=observation.log_scale,
            expansion_rate_s=0.0,
            scale_rate_qualified=False,
            predicted_rotational_rate_rad_s=(0.0, 0.0),
            residual_translational_rate_rad_s=(0.0, 0.0),
            time_to_contact_s=None,
            reference_camera_to_world_wxyz=reference,
            body_to_reference_wxyz=aligned.body_to_reference_wxyz,
            body_rates_rad_s=aligned.body_rates_rad_s,
            delayed_command=delayed,
            bearing_std_rad=(std_x, std_y),
            rate_std_rad_s=(
                self.config.max_abs_bearing_rate_rad_s / 2.0,
                self.config.max_abs_bearing_rate_rad_s / 2.0,
            ),
            log_scale_std=observation.measurement_std[2],
            expansion_rate_std_s=self.config.max_abs_expansion_rate_s / 2.0,
            clipping=observation.clipping,
            censored_axes=observation.censored_axes,
            visible=True,
            ambiguous=observation.ambiguous,
            confidence=observation.confidence,
            sample_count=1,
            missed_count=0,
        )

    def _split_image_rate(
        self,
        existing: _TrackEstimate,
        camera_to_world: Quaternion,
        raw_angle: Vector2,
        capture_ns: int,
    ) -> tuple[Vector2, Vector2]:
        dt = (capture_ns - existing.last_raw_measurement_ns) / _NS_PER_SECOND
        stable_to_world = existing.reference_camera_to_world
        previous_world_ray = _quat_rotate(
            stable_to_world,
            existing.last_measured_stable_ray,
        )
        predicted_camera_ray = _quat_rotate(
            _quat_conjugate(camera_to_world),
            previous_world_ray,
        )
        predicted_raw_angle = _ray_bearing(predicted_camera_ray)
        rotational = tuple(
            (predicted_raw_angle[axis] - existing.last_measured_raw_angle[axis]) / dt
            for axis in range(2)
        )
        measured = tuple(
            (raw_angle[axis] - existing.last_measured_raw_angle[axis]) / dt
            for axis in range(2)
        )
        residual = tuple(
            _clamp(
                measured[axis] - rotational[axis],
                -self.config.max_abs_bearing_rate_rad_s,
                self.config.max_abs_bearing_rate_rad_s,
            )
            for axis in range(2)
        )
        return rotational, residual  # type: ignore[return-value]

    def _predict_components(
        self,
        previous: TrackDynamicState,
        delayed: DelayedCommandView,
        dt: float,
    ) -> tuple[Vector2, Vector2, float, float]:
        acceleration = (
            self.config.roll_to_lateral_bearing_accel * delayed.target_roll_rad,
            self.config.thrust_to_vertical_bearing_accel
            * (delayed.thrust - SUPPORT_THRUST),
        )
        rate = tuple(
            _clamp(
                previous.bearing_rate_rad_s[axis] + acceleration[axis] * dt,
                -self.config.max_abs_bearing_rate_rad_s,
                self.config.max_abs_bearing_rate_rad_s,
            )
            for axis in range(2)
        )
        bearing = tuple(
            _clamp(
                previous.bearing_rad[axis]
                + previous.bearing_rate_rad_s[axis] * dt
                + 0.5 * acceleration[axis] * dt * dt,
                -self.config.max_abs_bearing_rad,
                self.config.max_abs_bearing_rad,
            )
            for axis in range(2)
        )
        expansion_acceleration = (
            -self.config.pitch_to_expansion_accel * delayed.target_pitch_rad
        )
        expansion = _clamp(
            previous.expansion_rate_s + expansion_acceleration * dt,
            -self.config.max_abs_expansion_rate_s,
            self.config.max_abs_expansion_rate_s,
        )
        log_scale = previous.log_scale + previous.expansion_rate_s * dt + (
            0.5 * expansion_acceleration * dt * dt
        )
        return (
            bearing,  # type: ignore[arg-type]
            rate,  # type: ignore[arg-type]
            log_scale,
            expansion,
        )

    def _robust_update(
        self,
        predicted_value: float,
        predicted_rate: float,
        measured_value: float,
        dt: float,
        alpha: float,
        beta: float,
        innovation_bound: float,
        rate_bound: float,
    ) -> tuple[float, float]:
        innovation = _clamp(
            measured_value - predicted_value,
            -innovation_bound,
            innovation_bound,
        )
        value = predicted_value + alpha * innovation
        rate = _clamp(
            predicted_rate + beta * innovation / dt,
            -rate_bound,
            rate_bound,
        )
        return value, rate

    def _update_track(
        self,
        previous: TrackDynamicState,
        observation: GateObservation,
        capture_ns: int,
        aligned: ImuAttitudeSample,
        delayed: DelayedCommandView,
        measured_bearing: Vector2,
        rotational_rate: Vector2,
        measured_residual_rate: tuple[
            float | None,
            float | None,
        ],
        measured_expansion_rate: float | None,
        stabilized_aperture: Vector2 | None,
    ) -> TrackDynamicState:
        assert observation.center_norm is not None
        assert observation.log_scale is not None
        dt = (capture_ns - previous.state_monotonic_ns) / _NS_PER_SECOND
        predicted_bearing, predicted_rate, predicted_scale, predicted_expansion = (
            self._predict_components(previous, delayed, dt)
        )
        quality = max(0.10, observation.confidence)
        if observation.ambiguous:
            quality *= 0.55
        alpha = self.config.bearing_alpha * quality
        censored = observation.censored_axes
        bearing_values: list[float] = []
        bearing_rates: list[float] = []
        residual_rates: list[float] = []
        bearing_rate_qualified: list[bool] = []
        bearing_std: list[float] = []
        rate_std: list[float] = []
        measurement_std_rad = (
            observation.measurement_std[0] * self.config.horizontal_angle_scale_rad,
            observation.measurement_std[1] * self.config.vertical_angle_scale_rad,
        )
        for axis in range(2):
            qualified_residual_rate = measured_residual_rate[axis]
            rate_qualified = qualified_residual_rate is not None
            if censored[axis]:
                value = predicted_bearing[axis]
                rate = predicted_rate[axis]
                residual = previous.residual_translational_rate_rad_s[axis]
                uncertainty_growth = (
                    self.config.clipping_uncertainty_multiplier
                )
            else:
                value, rate = self._robust_update(
                    predicted_bearing[axis],
                    predicted_rate[axis],
                    measured_bearing[axis],
                    dt,
                    alpha,
                    (
                        self.config.bearing_beta * quality
                        if rate_qualified
                        else 0.0
                    ),
                    self.config.max_bearing_innovation_rad,
                    self.config.max_abs_bearing_rate_rad_s,
                )
                if rate_qualified:
                    assert qualified_residual_rate is not None
                    residual = (
                        previous.residual_translational_rate_rad_s[axis]
                        + self.config.residual_alpha
                        * (
                            qualified_residual_rate
                            - previous.residual_translational_rate_rad_s[axis]
                        )
                    )
                else:
                    residual = (
                        previous.residual_translational_rate_rad_s[axis]
                    )
                residual = _clamp(
                    residual,
                    -self.config.max_abs_bearing_rate_rad_s,
                    self.config.max_abs_bearing_rate_rad_s,
                )
                uncertainty_growth = 1.0
            bearing_values.append(value)
            bearing_rates.append(rate)
            residual_rates.append(residual)
            bearing_rate_qualified.append(rate_qualified and not censored[axis])
            predicted_std = previous.bearing_std_rad[axis] + (
                self.config.process_noise_bearing_rad_s * dt
            )
            bearing_std.append(
                min(
                    self.config.max_abs_bearing_rad,
                    (
                        # Censorship inflates process noise; it must not
                        # multiply the entire posterior on every camera frame.
                        previous.bearing_std_rad[axis]
                        + uncertainty_growth
                        * self.config.process_noise_bearing_rad_s
                        * dt
                    )
                    if censored[axis]
                    else math.sqrt(
                        max(
                            1e-10,
                            (1.0 - alpha) * predicted_std * predicted_std
                            + alpha * measurement_std_rad[axis] ** 2,
                        )
                    )
                )
            )
            rate_std.append(
                min(
                    self.config.max_abs_bearing_rate_rad_s,
                    previous.rate_std_rad_s[axis]
                    + uncertainty_growth
                    * self.config.process_noise_bearing_rad_s,
                )
            )
        scale_measurement_usable = bool(
            observation.aperture_half_size_norm is not None
            and observation.clipping == FrameEdge.NONE
            and not observation.ambiguous
            and not any(censored)
        )
        scale_rate_qualified = bool(
            scale_measurement_usable
            and measured_expansion_rate is not None
        )
        if scale_measurement_usable:
            scale_innovation = _clamp(
                observation.log_scale - predicted_scale,
                -self.config.max_log_scale_innovation,
                self.config.max_log_scale_innovation,
            )
            log_scale = (
                predicted_scale
                + self.config.scale_alpha * quality * scale_innovation
            )
            if scale_rate_qualified:
                assert measured_expansion_rate is not None
                expansion = _clamp(
                    predicted_expansion
                    + self.config.scale_beta
                    * quality
                    * (
                        _clamp(
                            measured_expansion_rate,
                            -self.config.max_abs_expansion_rate_s,
                            self.config.max_abs_expansion_rate_s,
                        )
                        - predicted_expansion
                    ),
                    -self.config.max_abs_expansion_rate_s,
                    self.config.max_abs_expansion_rate_s,
                )
            else:
                expansion = predicted_expansion
            scale_multiplier = 1.0
        else:
            log_scale = predicted_scale
            expansion = predicted_expansion
            scale_multiplier = self.config.clipping_uncertainty_multiplier
        filtered_log_scale_std = (
            previous.log_scale_std
            + scale_multiplier
            * self.config.process_noise_scale_s
            * dt
            if not scale_measurement_usable
            else math.sqrt(
                max(
                    1e-10,
                    (1.0 - self.config.scale_alpha)
                    * (
                        previous.log_scale_std
                        + self.config.process_noise_scale_s * dt
                    )
                    ** 2
                    + self.config.scale_alpha
                    * observation.measurement_std[2]
                    ** 2,
                )
            )
        )
        if (
            scale_measurement_usable and not scale_rate_qualified
        ):
            filtered_log_scale_std = max(
                previous.log_scale_std,
                filtered_log_scale_std,
        )
        ttc = self._time_to_contact(expansion)
        measured_aperture = (
            stabilized_aperture
            if (
                observation.aperture_half_size_norm is not None
                and not observation.ambiguous
                and not any(censored)
            )
            else None
        )
        if measured_aperture is not None:
            # The rolling local state has a fixed short model horizon.
            # Closure/TTC qualification controls passage commitment, not
            # whether a clean aperture can support uncertainty-growing
            # steering through a detector/FOV gap.
            prediction_horizon_s = (
                self.config.crossing_prediction_max_horizon_s
            )
            aperture = measured_aperture
            aperture_seed_ns = capture_ns
            candidate_deadline_ns = capture_ns + round(
                prediction_horizon_s * _NS_PER_SECOND
            )
            # A later clean aperture corrects geometry and may extend the
            # bounded lease, but a one-frame loss of qualified closure/TTC
            # must not erase an already earned longer absolute horizon.
            # Retaining the prior deadline never grants time beyond the
            # configured maximum from that earlier clean seed.
            aperture_deadline_ns = max(
                candidate_deadline_ns,
                (
                    previous.aperture_prediction_deadline_monotonic_ns
                    if (
                        previous.aperture_half_size_norm is not None
                        and previous
                        .aperture_prediction_deadline_monotonic_ns
                        is not None
                        and capture_ns
                        <= previous
                        .aperture_prediction_deadline_monotonic_ns
                    )
                    else candidate_deadline_ns
                ),
            )
            aperture_propagated = False
            aperture_dynamics_qualified = bool(
                (
                    all(bearing_rate_qualified)
                    and scale_rate_qualified
                )
                or (
                    previous.aperture_dynamics_qualified
                    and previous
                    .aperture_prediction_deadline_monotonic_ns
                    is not None
                    and capture_ns
                    <= previous
                    .aperture_prediction_deadline_monotonic_ns
                )
            )
        elif (
            previous.aperture_half_size_norm is not None
            and previous.aperture_seed_monotonic_ns is not None
            and previous.aperture_prediction_deadline_monotonic_ns is not None
            and capture_ns
            <= previous.aperture_prediction_deadline_monotonic_ns
            and not observation.ambiguous
        ):
            scale_factor = math.exp(
                _clamp(log_scale - previous.log_scale, -1.0, 1.0)
            )
            aperture = tuple(
                min(2.0, max(1e-6, value * scale_factor))
                for value in previous.aperture_half_size_norm
            )
            aperture_seed_ns = previous.aperture_seed_monotonic_ns
            aperture_deadline_ns = (
                previous.aperture_prediction_deadline_monotonic_ns
            )
            aperture_propagated = True
            aperture_dynamics_qualified = (
                previous.aperture_dynamics_qualified
            )
        else:
            aperture = None
            aperture_seed_ns = None
            aperture_deadline_ns = None
            aperture_propagated = False
            aperture_dynamics_qualified = False
        return TrackDynamicState(
            track_id=observation.track_id,
            stream_generation=observation.stream_generation,
            state_monotonic_ns=capture_ns,
            last_measurement_monotonic_ns=capture_ns,
            frame_sequence=observation.frame_sequence,
            capture_timing_basis=observation.timing_basis,
            capture_timing_uncertainty_s=observation.timing_uncertainty_s,
            raw_center_norm=observation.center_norm,
            raw_log_scale=(
                observation.log_scale
                if scale_measurement_usable
                else None
            ),
            # One clean passage-usable inner aperture seeds a bounded local
            # gate-relative state.  Subsequent censored publications may
            # propagate it with the identified scale/command model, but they
            # cannot renew its deadline or create race/passage credit.
            aperture_half_size_norm=aperture,
            aperture_seed_monotonic_ns=aperture_seed_ns,
            aperture_prediction_deadline_monotonic_ns=(
                aperture_deadline_ns
            ),
            aperture_propagated=aperture_propagated,
            aperture_dynamics_qualified=aperture_dynamics_qualified,
            bearing_rad=(bearing_values[0], bearing_values[1]),
            bearing_rate_rad_s=(bearing_rates[0], bearing_rates[1]),
            bearing_rate_qualified=(
                bearing_rate_qualified[0],
                bearing_rate_qualified[1],
            ),
            log_scale=log_scale,
            expansion_rate_s=expansion,
            scale_rate_qualified=scale_rate_qualified,
            predicted_rotational_rate_rad_s=rotational_rate,
            residual_translational_rate_rad_s=(
                residual_rates[0],
                residual_rates[1],
            ),
            time_to_contact_s=ttc,
            reference_camera_to_world_wxyz=previous.reference_camera_to_world_wxyz,
            body_to_reference_wxyz=aligned.body_to_reference_wxyz,
            body_rates_rad_s=aligned.body_rates_rad_s,
            delayed_command=delayed,
            bearing_std_rad=(bearing_std[0], bearing_std[1]),
            rate_std_rad_s=(rate_std[0], rate_std[1]),
            log_scale_std=filtered_log_scale_std,
            expansion_rate_std_s=min(
                self.config.max_abs_expansion_rate_s,
                previous.expansion_rate_std_s
                + scale_multiplier
                * self.config.process_noise_scale_s,
            ),
            clipping=observation.clipping,
            censored_axes=censored,
            visible=True,
            ambiguous=observation.ambiguous,
            confidence=(
                previous.confidence
                if aperture_propagated
                else observation.confidence
            ),
            sample_count=previous.sample_count + 1,
            missed_count=0,
        )

    def _coast_track(
        self,
        existing: _TrackEstimate,
        observation: GateObservation,
        capture_ns: int,
        aligned: ImuAttitudeSample,
        delayed: DelayedCommandView,
    ) -> TrackDynamicState:
        previous = existing.state
        dt = (capture_ns - previous.state_monotonic_ns) / _NS_PER_SECOND
        bearing, rate, log_scale, expansion = self._predict_components(
            previous,
            delayed,
            dt,
        )
        aperture_prediction_valid = bool(
            previous.aperture_half_size_norm is not None
            and previous.aperture_seed_monotonic_ns is not None
            and previous.aperture_prediction_deadline_monotonic_ns is not None
            and capture_ns
            <= previous.aperture_prediction_deadline_monotonic_ns
            and not observation.ambiguous
        )
        aperture = (
            tuple(
                min(
                    2.0,
                    max(
                        1e-6,
                        value
                        * math.exp(
                            _clamp(
                                log_scale - previous.log_scale,
                                -1.0,
                                1.0,
                            )
                        ),
                    ),
                )
                for value in previous.aperture_half_size_norm
            )
            if aperture_prediction_valid
            and previous.aperture_half_size_norm is not None
            else None
        )
        state = TrackDynamicState(
            track_id=previous.track_id,
            stream_generation=previous.stream_generation,
            state_monotonic_ns=capture_ns,
            last_measurement_monotonic_ns=previous.last_measurement_monotonic_ns,
            frame_sequence=observation.frame_sequence,
            capture_timing_basis=observation.timing_basis,
            capture_timing_uncertainty_s=observation.timing_uncertainty_s,
            raw_center_norm=None,
            raw_log_scale=None,
            aperture_half_size_norm=aperture,
            aperture_seed_monotonic_ns=(
                previous.aperture_seed_monotonic_ns
                if aperture_prediction_valid
                else None
            ),
            aperture_prediction_deadline_monotonic_ns=(
                previous.aperture_prediction_deadline_monotonic_ns
                if aperture_prediction_valid
                else None
            ),
            aperture_propagated=aperture_prediction_valid,
            aperture_dynamics_qualified=bool(
                aperture_prediction_valid
                and previous.aperture_dynamics_qualified
            ),
            bearing_rad=bearing,
            bearing_rate_rad_s=rate,
            bearing_rate_qualified=(False, False),
            log_scale=log_scale,
            expansion_rate_s=expansion,
            scale_rate_qualified=False,
            predicted_rotational_rate_rad_s=previous.predicted_rotational_rate_rad_s,
            residual_translational_rate_rad_s=(
                previous.residual_translational_rate_rad_s
            ),
            time_to_contact_s=self._time_to_contact(expansion),
            reference_camera_to_world_wxyz=previous.reference_camera_to_world_wxyz,
            body_to_reference_wxyz=aligned.body_to_reference_wxyz,
            body_rates_rad_s=aligned.body_rates_rad_s,
            delayed_command=delayed,
            bearing_std_rad=tuple(
                value + self.config.process_noise_bearing_rad_s * dt
                for value in previous.bearing_std_rad
            ),  # type: ignore[arg-type]
            rate_std_rad_s=tuple(
                min(
                    self.config.max_abs_bearing_rate_rad_s,
                    value + self.config.process_noise_bearing_rad_s,
                )
                for value in previous.rate_std_rad_s
            ),  # type: ignore[arg-type]
            log_scale_std=(
                previous.log_scale_std + self.config.process_noise_scale_s * dt
            ),
            expansion_rate_std_s=min(
                self.config.max_abs_expansion_rate_s,
                previous.expansion_rate_std_s + self.config.process_noise_scale_s,
            ),
            clipping=FrameEdge.NONE,
            censored_axes=(True, True),
            visible=False,
            ambiguous=observation.ambiguous,
            confidence=0.0,
            sample_count=previous.sample_count,
            missed_count=previous.missed_count + 1,
        )
        existing.state = state
        return state

    def _time_to_contact(self, expansion_rate_s: float) -> float | None:
        if expansion_rate_s <= 1.0 / self.config.maximum_ttc_s:
            return None
        return _clamp(
            1.0 / expansion_rate_s,
            self.config.minimum_ttc_s,
            self.config.maximum_ttc_s,
        )

    def bind(
        self,
        *,
        current_gate_index: int,
        current_track_id: str,
        successor_track_id: str | None,
    ) -> CourseDynamicState:
        _exact_nonnegative_int(current_gate_index, "current_gate_index")
        _token(current_track_id, "current_track_id")
        if current_track_id not in self._tracks:
            raise DynamicCourseError("current track has no dynamic state")
        if successor_track_id is not None:
            _token(successor_track_id, "successor_track_id")
            if successor_track_id == current_track_id:
                raise DynamicCourseError("current and successor tracks must differ")
            if successor_track_id not in self._tracks:
                raise DynamicCourseError("successor track has no dynamic state")
        ownership_key = (
            current_gate_index,
            current_track_id,
            successor_track_id,
        )
        if ownership_key != self._successor_clearance_key:
            self._successor_clearance_key = None
            self._successor_clearance_positive_since_ns = None
        self._current_track_id = current_track_id
        self._current_gate_index = current_gate_index
        self._successor_track_id = successor_track_id
        return self.course_state()

    def course_state(self) -> CourseDynamicState:
        if self._current_gate_index is None or self._current_track_id is None:
            raise DynamicCourseError("course roles have not been bound")
        current = self._tracks[self._current_track_id].state
        successor = (
            None
            if self._successor_track_id is None
            else self._tracks[self._successor_track_id].state
        )
        return CourseDynamicState(
            current_gate_index=self._current_gate_index,
            current_track_id=self._current_track_id,
            successor_track_id=self._successor_track_id,
            current=current,
            successor=successor,
            recent_commands=tuple(self._commands),
            last_applied_command=self._last_applied_command,
            promotion_count=self._promotion_count,
        )

    def predict_track_steering(
        self,
        track_id: str,
        monotonic_ns: int,
    ) -> TrackSteeringPrediction:
        """Predict one known track without advancing estimator ownership.

        The stable bearing is integrated piecewise across the accepted-command
        delay boundaries.  It is then projected into the current camera using
        the aligned IMU attitude.  The returned image rate also includes the
        instantaneous rotational flow implied by measured body rates.
        """

        _token(track_id, "track_id")
        _exact_nonnegative_int(monotonic_ns, "monotonic_ns")
        try:
            estimate = self._tracks[track_id]
        except KeyError as exc:
            raise DynamicCourseError(
                "steering prediction track has no dynamic state"
            ) from exc
        source = estimate.state
        if monotonic_ns < source.state_monotonic_ns:
            raise DynamicCourseError(
                "steering prediction precedes the track state"
            )

        bearing = source.bearing_rad
        rate = source.bearing_rate_rad_s
        cursor_ns = source.state_monotonic_ns
        boundaries = {monotonic_ns}
        for command in self._commands:
            for delay_s in (
                self.config.roll_command_delay_s,
                self.config.thrust_command_delay_s,
            ):
                boundary_ns = command.monotonic_ns + round(
                    delay_s * _NS_PER_SECOND
                )
                if cursor_ns < boundary_ns < monotonic_ns:
                    boundaries.add(boundary_ns)
        for boundary_ns in sorted(boundaries):
            dt = (boundary_ns - cursor_ns) / _NS_PER_SECOND
            if dt <= 0.0:
                cursor_ns = boundary_ns
                continue
            delayed = self.delayed_command_view(cursor_ns)
            acceleration = (
                self.config.roll_to_lateral_bearing_accel
                * delayed.target_roll_rad,
                self.config.thrust_to_vertical_bearing_accel
                * (delayed.thrust - SUPPORT_THRUST),
            )
            bearing = tuple(
                _clamp(
                    bearing[axis]
                    + rate[axis] * dt
                    + 0.5 * acceleration[axis] * dt * dt,
                    -self.config.max_abs_bearing_rad,
                    self.config.max_abs_bearing_rad,
                )
                for axis in range(2)
            )
            rate = tuple(
                _clamp(
                    rate[axis] + acceleration[axis] * dt,
                    -self.config.max_abs_bearing_rate_rad_s,
                    self.config.max_abs_bearing_rate_rad_s,
                )
                for axis in range(2)
            )
            cursor_ns = boundary_ns

        aligned = self._aligned_imu(monotonic_ns)
        decision_camera_to_world = _quat_multiply(
            aligned.body_to_reference_wxyz,
            self.config.camera_to_body_wxyz,
        )
        stable_ray = (
            1.0,
            math.tan(bearing[0]),
            math.tan(bearing[1]),
        )
        stable_ray_rate = (
            0.0,
            rate[0] / max(math.cos(bearing[0]) ** 2, _EPSILON),
            rate[1] / max(math.cos(bearing[1]) ** 2, _EPSILON),
        )
        world_ray = _quat_rotate(
            estimate.reference_camera_to_world,
            stable_ray,
        )
        world_ray_rate = _quat_rotate(
            estimate.reference_camera_to_world,
            stable_ray_rate,
        )
        world_to_camera = _quat_conjugate(
            decision_camera_to_world
        )
        camera_ray = _quat_rotate(world_to_camera, world_ray)
        translational_camera_ray_rate = _quat_rotate(
            world_to_camera,
            world_ray_rate,
        )
        camera_body_rates = _quat_rotate(
            _quat_conjugate(self.config.camera_to_body_wxyz),
            aligned.body_rates_rad_s,
        )
        rotational_cross = (
            camera_body_rates[1] * camera_ray[2]
            - camera_body_rates[2] * camera_ray[1],
            camera_body_rates[2] * camera_ray[0]
            - camera_body_rates[0] * camera_ray[2],
            camera_body_rates[0] * camera_ray[1]
            - camera_body_rates[1] * camera_ray[0],
        )
        camera_ray_rate = tuple(
            translational_camera_ray_rate[axis]
            - rotational_cross[axis]
            for axis in range(3)
        )
        forward = camera_ray[0]
        if forward <= 1e-6:
            raise DynamicCourseError(
                "steering prediction gate ray is behind the camera"
            )
        horizontal_ratio = camera_ray[1] / forward
        vertical_ratio = camera_ray[2] / forward
        camera_center = (
            horizontal_ratio
            / self.config.horizontal_angle_scale_rad,
            vertical_ratio / self.config.vertical_angle_scale_rad,
        )
        forward_squared = forward * forward
        camera_rate = (
            (
                camera_ray_rate[1] * forward
                - camera_ray[1] * camera_ray_rate[0]
            )
            / forward_squared
            / self.config.horizontal_angle_scale_rad,
            (
                camera_ray_rate[2] * forward
                - camera_ray[2] * camera_ray_rate[0]
            )
            / forward_squared
            / self.config.vertical_angle_scale_rad,
        )
        maximum_camera_rate = (
            self.config.max_abs_bearing_rate_rad_s
            / self.config.horizontal_angle_scale_rad,
            self.config.max_abs_bearing_rate_rad_s
            / self.config.vertical_angle_scale_rad,
        )
        camera_rate = tuple(
            _clamp(
                camera_rate[axis],
                -maximum_camera_rate[axis],
                maximum_camera_rate[axis],
            )
            for axis in range(2)
        )
        elapsed_s = (
            monotonic_ns - source.state_monotonic_ns
        ) / _NS_PER_SECOND
        measurement_age_s = (
            monotonic_ns - source.last_measurement_monotonic_ns
        ) / _NS_PER_SECOND
        bearing_std = tuple(
            min(
                self.config.max_abs_bearing_rad,
                source.bearing_std_rad[axis]
                + self.config.process_noise_bearing_rad_s * elapsed_s,
            )
            for axis in range(2)
        )
        values = (
            *bearing,
            *rate,
            *camera_center,
            *camera_rate,
            *bearing_std,
            *aligned.body_rates_rad_s,
            measurement_age_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise DynamicCourseError(
                "steering prediction produced non-finite state"
            )
        if measurement_age_s < 0.0:
            raise DynamicCourseError(
                "steering prediction precedes the last measurement"
            )
        return TrackSteeringPrediction(
            track_id=track_id,
            stream_generation=source.stream_generation,
            monotonic_ns=monotonic_ns,
            source_state_monotonic_ns=source.state_monotonic_ns,
            last_measurement_monotonic_ns=(
                source.last_measurement_monotonic_ns
            ),
            measurement_age_s=measurement_age_s,
            stable_bearing_rad=bearing,  # type: ignore[arg-type]
            stable_bearing_rate_rad_s=rate,  # type: ignore[arg-type]
            camera_center_norm=camera_center,
            camera_center_rate_norm_s=camera_rate,  # type: ignore[arg-type]
            bearing_std_rad=bearing_std,  # type: ignore[arg-type]
            body_rates_rad_s=aligned.body_rates_rad_s,
        )

    def retains_successor_lineage(
        self,
        successor_track_id: str,
        monotonic_ns: int,
    ) -> bool:
        """Retain only an already bound successor identity through occlusion.

        This authority never supplies geometry or yaw.  It exists solely so a
        near-plane passage can seal the exact successor that was reviewed
        before expected aperture occlusion.
        """

        _token(successor_track_id, "successor_track_id")
        _exact_nonnegative_int(monotonic_ns, "monotonic_ns")
        state = self.course_state()
        successor = state.successor
        if (
            state.successor_track_id != successor_track_id
            or successor is None
            or successor.track_id != successor_track_id
            or successor.stream_generation
            != state.current.stream_generation
            or successor.sample_count < 4
            or successor.ambiguous
        ):
            return False
        age_s = (
            monotonic_ns - successor.last_measurement_monotonic_ns
        ) / _NS_PER_SECOND
        return bool(
            0.0 <= age_s <= self.config.successor_lineage_hold_s
        )

    def promote_authoritative(
        self,
        *,
        from_gate_index: int,
        to_gate_index: int,
        promoted_track_id: str,
        next_successor_track_id: str | None,
        monotonic_ns: int,
    ) -> CourseDynamicState:
        if self._current_gate_index is None or self._current_track_id is None:
            raise DynamicCourseError("course roles have not been bound")
        _exact_nonnegative_int(from_gate_index, "from_gate_index")
        _exact_nonnegative_int(to_gate_index, "to_gate_index")
        _exact_nonnegative_int(monotonic_ns, "monotonic_ns")
        if from_gate_index != self._current_gate_index or to_gate_index != (
            from_gate_index + 1
        ):
            raise DynamicCourseError("authoritative promotions must be sequential")
        if promoted_track_id != self._successor_track_id:
            raise DynamicCourseError(
                "authoritative promotion must preserve the tracked successor"
            )
        if self._last_promotion_ns is not None and monotonic_ns <= self._last_promotion_ns:
            raise DynamicCourseError("authoritative promotion time must advance")
        if next_successor_track_id is not None:
            _token(next_successor_track_id, "next_successor_track_id")
            if next_successor_track_id == promoted_track_id:
                raise DynamicCourseError("promoted and next successor tracks must differ")
            if next_successor_track_id not in self._tracks:
                raise DynamicCourseError("next successor track has no dynamic state")
        self._current_gate_index = to_gate_index
        self._current_track_id = promoted_track_id
        self._successor_track_id = next_successor_track_id
        self._successor_clearance_key = None
        self._successor_clearance_positive_since_ns = None
        self._promotion_count += 1
        self._last_promotion_ns = monotonic_ns
        return self.course_state()

    def guide(self, monotonic_ns: int) -> GuidanceDecision:
        _exact_nonnegative_int(monotonic_ns, "monotonic_ns")
        if self._last_applied_command is None:
            raise DynamicCourseError(
                "guidance requires a confirmed applied command"
            )
        state = self.course_state()
        current = state.current
        successor = state.successor
        if monotonic_ns < current.state_monotonic_ns:
            raise DynamicCourseError("guidance time cannot precede the current state")
        camera_current_center, _ = self._decision_geometry(
            current.track_id,
            monotonic_ns,
        )
        successor_prediction = self._predicted_successor(
            current,
            successor,
            monotonic_ns,
        )
        successor_transition_held = self._successor_transition_hold(
            successor,
            monotonic_ns,
        )
        successor_passage_ready = (
            successor is not None
            and successor.sample_count >= 4
            and (
                successor_transition_held
                or (
                    successor.visible
                    and not successor.ambiguous
                    and not any(successor.censored_axes)
                )
            )
        )
        passage_successor_track_id = (
            None
            if not successor_passage_ready
            else successor.track_id
        )
        (
            current_center,
            current_aperture,
            successor_center,
        ) = self._stable_passage_geometry(
            current.track_id,
            passage_successor_track_id,
        )
        if (
            current.aperture_prediction_deadline_monotonic_ns is None
            or monotonic_ns
            > current.aperture_prediction_deadline_monotonic_ns
        ):
            current_aperture = None
        age_s = (
            monotonic_ns - current.last_measurement_monotonic_ns
        ) / _NS_PER_SECOND
        held = not current.visible and age_s <= self.config.dropout_hold_s
        crossing_prediction_horizon_s = (
            0.0
            if current.time_to_contact_s is None
            else min(
                self.config.crossing_prediction_max_horizon_s,
                max(0.0, current.time_to_contact_s),
            )
        )
        residual_rate_norm = (
            current.residual_translational_rate_rad_s[0]
            / self.config.horizontal_angle_scale_rad,
            current.residual_translational_rate_rad_s[1]
            / self.config.vertical_angle_scale_rad,
        )
        current_std_norm = (
            current.bearing_std_rad[0]
            / self.config.horizontal_angle_scale_rad,
            current.bearing_std_rad[1]
            / self.config.vertical_angle_scale_rad,
        )

        def crossing_for(
            passage_offset_norm: Vector2,
        ) -> CrossingQuotientPrediction | None:
            return (
                None
                if current_aperture is None
                else predict_aperture_relative_crossing(
                    center_offset_norm=current_center,
                    passage_offset_norm=passage_offset_norm,
                    aperture_half_extent_norm=current_aperture,
                    center_rate_norm_s=residual_rate_norm,
                    # The robust log-scale filter is the currently identified
                    # aperture expansion model.  The quotient helper accepts
                    # per-axis rates so a later clean axis fit can replace this
                    # shared rate without changing guidance semantics.
                    aperture_expansion_rate_s=(
                        current.expansion_rate_s,
                        current.expansion_rate_s,
                    ),
                    center_std_norm=current_std_norm,
                    aperture_log_scale_std=current.log_scale_std,
                    capture_timing_uncertainty_s=(
                        current.capture_timing_uncertainty_s
                    ),
                    horizon_s=crossing_prediction_horizon_s,
                    allowance_q=self.config.crossing_max_occupancy_q,
                )
            )

        # Current-gate ownership is evaluated without successor bias.  This
        # breaks the former circularity in which an unsafe successor-selected
        # passage point was itself used to decide whether successor guidance
        # was admissible.
        centered_crossing_prediction = crossing_for((0.0, 0.0))
        centered_crossing_clearance = (
            (0.0, 0.0)
            if centered_crossing_prediction is None
            else centered_crossing_prediction.clearance_q
        )
        (
            successor_clearance_authority,
            successor_clearance_dwell_s,
        ) = self._successor_clearance_authority(
            current_gate_index=state.current_gate_index,
            current=current,
            successor=successor,
            prediction=successor_prediction,
            centered_prediction=centered_crossing_prediction,
        )
        candidate_passage_authority = (
            self._successor_passage_bias_authority(
                current,
                successor,
                successor_prediction,
                monotonic_ns,
            )
            * successor_clearance_authority
        )
        if successor_center is None:
            candidate_passage_authority = 0.0

        def passage_for(
            authority: float,
        ) -> tuple[
            Vector2,
            Vector2,
            CrossingQuotientPrediction | None,
        ]:
            point, remaining = self._passage_point(
                current_aperture,
                current_center,
                successor_center,
                authority,
            )
            return point, remaining, crossing_for(point)

        successor_passage_authority = candidate_passage_authority
        passage, margins, crossing_prediction = passage_for(
            successor_passage_authority
        )
        if (
            successor_passage_authority > 0.0
            and crossing_prediction is not None
            and any(
                clearance <= 0.0
                for clearance in crossing_prediction.clearance_q
            )
        ):
            # The centered envelope is the ownership proof; the successor
            # offset may consume only its remaining reserve.  Find the largest
            # continuously scaled bias that preserves strict clearance on both
            # axes rather than admitting an unsafe all-or-nothing target.
            lower = 0.0
            upper = successor_passage_authority
            passage, margins, crossing_prediction = passage_for(lower)
            for _ in range(18):
                midpoint = 0.5 * (lower + upper)
                (
                    candidate_passage,
                    candidate_margins,
                    candidate_crossing,
                ) = passage_for(midpoint)
                if (
                    candidate_crossing is not None
                    and all(
                        clearance > 0.0
                        for clearance in candidate_crossing.clearance_q
                    )
                ):
                    lower = midpoint
                    passage = candidate_passage
                    margins = candidate_margins
                    crossing_prediction = candidate_crossing
                else:
                    upper = midpoint
            successor_passage_authority = lower
        passage_error = (
            current_center[0] + passage[0],
            current_center[1] + passage[1],
        )

        if (
            centered_crossing_prediction is not None
            and current_aperture is not None
            and current.scale_rate_qualified
        ):
            aperture_relative_rate_norm = tuple(
                (
                    centered_crossing_prediction.rate_q_s[axis]
                    * current_aperture[axis]
                    if current.bearing_rate_qualified[axis]
                    else residual_rate_norm[axis]
                )
                for axis in range(2)
            )
        else:
            aperture_relative_rate_norm = residual_rate_norm

        crossing_prediction = (
            None
            if current_aperture is None
            else crossing_prediction
        )
        if crossing_prediction is None:
            current_crossing_error_q = (0.0, 0.0)
            crossing_rate_q_s = (0.0, 0.0)
            predicted_crossing_error = (0.0, 0.0)
            predicted_crossing_std = (0.0, 0.0)
            crossing_allowance = (0.0, 0.0)
            crossing_swept_occupancy = (0.0, 0.0)
            predicted_crossing_clearance = (0.0, 0.0)
            terminal_crossing_occupancy = (0.0, 0.0)
            terminal_crossing_clearance = (0.0, 0.0)
        else:
            current_crossing_error_q = (
                crossing_prediction.current_error_q
            )
            crossing_rate_q_s = crossing_prediction.rate_q_s
            predicted_crossing_error = (
                crossing_prediction.predicted_error_q
            )
            predicted_crossing_std = (
                crossing_prediction.predicted_std_q
            )
            crossing_allowance = crossing_prediction.allowance_q
            crossing_swept_occupancy = (
                crossing_prediction.swept_occupancy_q
            )
            predicted_crossing_clearance = (
                crossing_prediction.clearance_q
            )
            terminal_crossing_occupancy = (
                crossing_prediction.terminal_occupancy_q
            )
            terminal_crossing_clearance = (
                crossing_prediction.terminal_clearance_q
            )
        current_yaw_release = (
            self._current_yaw_release(
                current,
                successor_passage_ready,
            )
            * successor_clearance_authority
        )
        geometric_passage_yaw_authority = self._passage_yaw_authority(
            current,
            passage_error,
            margins,
            successor_prediction,
        )
        passage_yaw_authority = (
            geometric_passage_yaw_authority
            * successor_clearance_authority
        )
        successor_weight = self._successor_weight(
            current,
            successor,
            successor_prediction,
            geometric_passage_yaw_authority,
            current_yaw_release,
        )
        horizontal_alignment_unsettled = bool(
            not current.bearing_rate_qualified[0]
            or not current.scale_rate_qualified
            or centered_crossing_prediction is None
            or centered_crossing_clearance[0] <= 0.0
            or abs(aperture_relative_rate_norm[0])
            > self.config.vertical_settled_rate_norm_s
        )
        vertical_alignment_unsettled = bool(
            not current.bearing_rate_qualified[1]
            or not current.scale_rate_qualified
            or centered_crossing_prediction is None
            or centered_crossing_clearance[1] <= 0.0
            or abs(aperture_relative_rate_norm[1])
            > self.config.vertical_settled_rate_norm_s
        )
        current_alignment_unsettled = bool(
            horizontal_alignment_unsettled
            or vertical_alignment_unsettled
        )
        proposal, braking, reason, yaw_contribution = self._propose_command(
            current,
            successor,
            camera_current_center,
            passage_error,
            aperture_relative_rate_norm,
            current_yaw_release,
            successor_weight,
            successor_prediction,
            horizontal_alignment_unsettled=(
                horizontal_alignment_unsettled
            ),
            vertical_alignment_unsettled=vertical_alignment_unsettled,
        )
        return GuidanceDecision(
            monotonic_ns=monotonic_ns,
            current_gate_index=state.current_gate_index,
            current_track_id=current.track_id,
            successor_track_id=None if successor is None else successor.track_id,
            current_center_norm=current_center,
            camera_current_center_norm=camera_current_center,
            current_aperture_half_size_norm=current_aperture,
            current_aperture_propagated=current.aperture_propagated,
            current_aperture_dynamics_qualified=(
                current.aperture_dynamics_qualified
            ),
            current_aperture_prediction_age_s=(
                0.0
                if current.aperture_seed_monotonic_ns is None
                else max(
                    0.0,
                    (
                        monotonic_ns
                        - current.aperture_seed_monotonic_ns
                    )
                    / _NS_PER_SECOND,
                )
            ),
            current_aperture_prediction_horizon_remaining_s=(
                0.0
                if (
                    current.aperture_prediction_deadline_monotonic_ns
                    is None
                )
                else max(
                    0.0,
                    (
                        current.aperture_prediction_deadline_monotonic_ns
                        - monotonic_ns
                    )
                    / _NS_PER_SECOND,
                )
            ),
            passage_point_norm=passage,
            successor_passage_authority=(
                successor_passage_authority
            ),
            centered_crossing_clearance_norm=(
                centered_crossing_clearance
            ),
            successor_clearance_dwell_s=(
                successor_clearance_dwell_s
            ),
            successor_clearance_authority=(
                successor_clearance_authority
            ),
            passage_error_norm=passage_error,
            aperture_margin_norm=margins,
            crossing_prediction_horizon_s=(
                crossing_prediction_horizon_s
            ),
            current_crossing_error_q=current_crossing_error_q,
            crossing_rate_q_s=crossing_rate_q_s,
            predicted_crossing_error_norm=(
                predicted_crossing_error
            ),
            predicted_crossing_std_norm=predicted_crossing_std,
            crossing_allowance_norm=crossing_allowance,
            crossing_swept_occupancy_norm=(
                crossing_swept_occupancy
            ),
            predicted_crossing_clearance_norm=(
                predicted_crossing_clearance
            ),
            terminal_crossing_occupancy_norm=(
                terminal_crossing_occupancy
            ),
            terminal_crossing_clearance_norm=(
                terminal_crossing_clearance
            ),
            current_bearing_std_rad=current.bearing_std_rad,
            successor_bearing_std_rad=(
                None if successor is None else successor.bearing_std_rad
            ),
            successor_weight=successor_weight,
            predicted_successor_bearing_rad=(
                None
                if successor_prediction is None
                else successor_prediction.bearing_rad
            ),
            measured_successor_bearing_rad=(
                None
                if successor_prediction is None
                else successor_prediction.measured_bearing_rad
            ),
            successor_rate_rad_s=(
                None
                if successor_prediction is None
                else successor_prediction.robust_rate_rad_s
            ),
            successor_prediction_horizon_s=(
                0.0
                if successor_prediction is None
                else successor_prediction.horizon_s
            ),
            successor_prediction_confidence=(
                0.0
                if successor_prediction is None
                else successor_prediction.confidence
            ),
            current_yaw_release=current_yaw_release,
            passage_yaw_authority=passage_yaw_authority,
            successor_yaw_contribution_rad=yaw_contribution,
            successor_transition_held=successor_transition_held,
            current_time_to_contact_s=current.time_to_contact_s,
            braking=braking,
            brake_reason=reason,
            dropout_held=held,
            proposed_command=proposal,
            command=proposal,
        )

    def _geometry_in_orientation(
        self,
        track_id: str,
        target_camera_to_world: Quaternion,
    ) -> tuple[Vector2, Vector2 | None]:
        estimate = self._tracks[track_id]

        def project_stable_ray(stable_ray: Vector3) -> Vector2:
            world_ray = _quat_rotate(
                estimate.reference_camera_to_world,
                stable_ray,
            )
            target_ray = _quat_rotate(
                _quat_conjugate(target_camera_to_world),
                world_ray,
            )
            if target_ray[0] <= 1e-6:
                raise DynamicCourseError("gate ray is behind the target orientation")
            bearing = _ray_bearing(target_ray)
            return (
                math.tan(bearing[0]) / self.config.horizontal_angle_scale_rad,
                math.tan(bearing[1]) / self.config.vertical_angle_scale_rad,
            )

        center = project_stable_ray(_bearing_ray(estimate.state.bearing_rad))
        aperture = estimate.state.aperture_half_size_norm
        measured_aperture = estimate.last_measured_aperture_half_size_norm
        if aperture is None or measured_aperture is None:
            return center, None

        def reproject_last(raw_center: Vector2) -> Vector2:
            ray = _camera_ray(
                raw_center,
                self.config.horizontal_angle_scale_rad,
                self.config.vertical_angle_scale_rad,
            )
            world_ray = _quat_rotate(
                estimate.last_measurement_camera_to_world,
                ray,
            )
            target_ray = _quat_rotate(
                _quat_conjugate(target_camera_to_world),
                world_ray,
            )
            if target_ray[0] <= 1e-6:
                raise DynamicCourseError(
                    "gate aperture is behind the target orientation"
                )
            bearing = _ray_bearing(target_ray)
            return (
                math.tan(bearing[0]) / self.config.horizontal_angle_scale_rad,
                math.tan(bearing[1]) / self.config.vertical_angle_scale_rad,
            )

        measured_center = estimate.last_measured_center_norm
        left = reproject_last(
            (
                measured_center[0] - measured_aperture[0],
                measured_center[1],
            )
        )
        right = reproject_last(
            (
                measured_center[0] + measured_aperture[0],
                measured_center[1],
            )
        )
        top = reproject_last(
            (
                measured_center[0],
                measured_center[1] - measured_aperture[1],
            )
        )
        bottom = reproject_last(
            (
                measured_center[0],
                measured_center[1] + measured_aperture[1],
            )
        )
        measured_projected_aperture = (
            0.5 * abs(right[0] - left[0]),
            0.5 * abs(bottom[1] - top[1]),
        )
        projected_aperture = tuple(
            measured_projected_aperture[axis]
            * aperture[axis]
            / measured_aperture[axis]
            for axis in range(2)
        )
        return center, projected_aperture

    def _decision_geometry(
        self,
        track_id: str,
        monotonic_ns: int,
    ) -> tuple[Vector2, Vector2 | None]:
        aligned = self._aligned_imu(monotonic_ns)
        decision_camera_to_world = _quat_multiply(
            aligned.body_to_reference_wxyz,
            self.config.camera_to_body_wxyz,
        )
        return self._geometry_in_orientation(
            track_id,
            decision_camera_to_world,
        )

    def _stable_passage_geometry(
        self,
        current_track_id: str,
        successor_track_id: str | None,
    ) -> tuple[Vector2, Vector2 | None, Vector2 | None]:
        current_estimate = self._tracks[current_track_id]
        stable_orientation = current_estimate.reference_camera_to_world
        current_center, current_aperture = self._geometry_in_orientation(
            current_track_id,
            stable_orientation,
        )
        successor_center = (
            None
            if successor_track_id is None
            else self._geometry_in_orientation(
                successor_track_id,
                stable_orientation,
            )[0]
        )
        return current_center, current_aperture, successor_center

    def _passage_point(
        self,
        aperture_half_size_norm: Vector2 | None,
        current_center_norm: Vector2,
        successor_center_norm: Vector2 | None,
        successor_authority: float,
    ) -> tuple[Vector2, Vector2]:
        aperture = aperture_half_size_norm or (0.42, 0.32)
        available = (
            max(0.0, aperture[0] - self.config.passage_margin_norm),
            max(0.0, aperture[1] - self.config.passage_margin_norm),
        )
        if (
            successor_center_norm is None
        ):
            return (0.0, 0.0), available
        successor_authority = _clamp(
            successor_authority,
            0.0,
            1.0,
        )
        offset = (
            successor_center_norm[0] - current_center_norm[0],
            successor_center_norm[1] - current_center_norm[1],
        )
        passage = tuple(
            _clamp(
                self.config.passage_successor_bias
                * successor_authority
                * offset[axis],
                -available[axis],
                available[axis],
            )
            for axis in range(2)
        )
        remaining = tuple(
            max(0.0, available[axis] - abs(passage[axis])) for axis in range(2)
        )
        return passage, remaining  # type: ignore[return-value]

    def _successor_clearance_authority(
        self,
        *,
        current_gate_index: int,
        current: TrackDynamicState,
        successor: TrackDynamicState | None,
        prediction: _SuccessorPrediction | None,
        centered_prediction: CrossingQuotientPrediction | None,
    ) -> tuple[float, float]:
        """Release successor control only after current-centered safety.

        Dwell advances in exact current-observation time, so repeated guide
        calls cannot manufacture temporal consistency.  The release is then a
        continuous product of elapsed ramp and the smaller reserved clearance.
        """

        key = (
            current_gate_index,
            current.track_id,
            None if successor is None else successor.track_id,
        )
        if self._successor_clearance_key != key:
            self._successor_clearance_key = key
            self._successor_clearance_positive_since_ns = None
        qualified = bool(
            successor is not None
            and prediction is not None
            and prediction.confidence > 0.0
            and current.visible
            and not current.ambiguous
            and not any(current.censored_axes)
            and all(current.bearing_rate_qualified)
            and current.scale_rate_qualified
            and current.time_to_contact_s is not None
            and successor.visible
            and not successor.ambiguous
            and not any(successor.censored_axes)
            and successor.sample_count >= 4
            and centered_prediction is not None
            and all(
                clearance > 0.0
                for clearance in centered_prediction.clearance_q
            )
        )
        if not qualified:
            self._successor_clearance_positive_since_ns = None
            return 0.0, 0.0
        measurement_ns = current.last_measurement_monotonic_ns
        if self._successor_clearance_positive_since_ns is None:
            self._successor_clearance_positive_since_ns = measurement_ns
        elapsed_s = max(
            0.0,
            (
                measurement_ns
                - self._successor_clearance_positive_since_ns
            )
            / _NS_PER_SECOND,
        )
        ramp = _clamp(
            (
                elapsed_s - self.config.successor_clearance_dwell_s
            )
            / self.config.successor_clearance_ramp_s,
            0.0,
            1.0,
        )
        reserve = min(
            _clamp(
                centered_prediction.clearance_q[axis]
                / centered_prediction.allowance_q[axis],
                0.0,
                1.0,
            )
            for axis in range(2)
        )
        return reserve * ramp, elapsed_s

    def _successor_passage_bias_authority(
        self,
        current: TrackDynamicState,
        successor: TrackDynamicState | None,
        prediction: _SuccessorPrediction | None,
        monotonic_ns: int,
    ) -> float:
        """Admit successor passage bias continuously, never at sample four."""

        if successor is None:
            self._successor_passage_track_id = None
            self._successor_passage_authority = 0.0
            self._successor_passage_authority_ns = None
            return 0.0
        if self._successor_passage_track_id != successor.track_id:
            self._successor_passage_track_id = successor.track_id
            self._successor_passage_authority = 0.0
            self._successor_passage_authority_ns = None
        clean_prediction = bool(
            prediction is not None
            and prediction.confidence > 0.0
            and current.visible
            and not current.ambiguous
            and not any(current.censored_axes)
            and successor.visible
            and not successor.ambiguous
            and not successor.censored_axes[0]
        )
        if clean_prediction:
            ttc = current.time_to_contact_s
            ttc_progress = (
                0.0
                if ttc is None
                else _clamp(
                    (
                        self.config.successor_lookahead_ttc_s - ttc
                    )
                    / (
                        self.config.successor_lookahead_ttc_s
                        - self.config.successor_full_weight_ttc_s
                    ),
                    0.0,
                    1.0,
                )
            )
            scale_lower_bound = (
                current.log_scale - 2.0 * current.log_scale_std
            )
            scale_start = self.config.passage_arm_min_log_scale - 0.25
            scale_progress = _clamp(
                (
                    scale_lower_bound - scale_start
                )
                / (self.config.passage_arm_min_log_scale - scale_start),
                0.0,
                1.0,
            )
            closure_progress = max(ttc_progress, scale_progress)
            closure_authority = (
                self.config.successor_passage_far_authority
                + (
                    1.0
                    - self.config.successor_passage_far_authority
                )
                * closure_progress
            )
            self._successor_passage_authority = _clamp(
                (
                    prediction.confidence
                    / self.config.successor_passage_full_confidence
                )
                * closure_authority,
                0.0,
                1.0,
            )
            self._successor_passage_authority_ns = monotonic_ns
            return self._successor_passage_authority
        if self._successor_passage_authority_ns is not None:
            age_s = (
                monotonic_ns - self._successor_passage_authority_ns
            ) / _NS_PER_SECOND
            if 0.0 <= age_s <= self.config.dropout_hold_s:
                return self._successor_passage_authority
        self._successor_passage_authority = 0.0
        self._successor_passage_authority_ns = None
        return 0.0

    def _current_yaw_release(
        self,
        current: TrackDynamicState,
        successor_passage_ready: bool,
    ) -> float:
        if (
            not successor_passage_ready
            or not current.visible
            or current.ambiguous
            or any(current.censored_axes)
        ):
            return 0.0
        scale_lower_bound = current.log_scale - 2.0 * current.log_scale_std
        scale_start = self.config.passage_arm_min_log_scale - 0.25
        return _clamp(
            (
                scale_lower_bound - scale_start
            )
            / (self.config.passage_arm_min_log_scale - scale_start),
            0.0,
            1.0,
        )

    def _successor_transition_hold(
        self,
        successor: TrackDynamicState | None,
        monotonic_ns: int,
    ) -> bool:
        if (
            successor is None
            or successor.visible
            or successor.ambiguous
            or successor.sample_count < 4
        ):
            return False
        age_s = (
            monotonic_ns - successor.last_measurement_monotonic_ns
        ) / _NS_PER_SECOND
        return 0.0 <= age_s <= self.config.dropout_hold_s

    def _passage_yaw_authority(
        self,
        current: TrackDynamicState,
        passage_error_norm: Vector2,
        aperture_remaining_norm: Vector2,
        prediction: _SuccessorPrediction | None,
    ) -> float:
        if (
            prediction is None
            or not current.visible
            or current.ambiguous
            or any(current.censored_axes)
        ):
            return 0.0
        current_std_norm = (
            current.bearing_std_rad[0]
            / self.config.horizontal_angle_scale_rad,
            current.bearing_std_rad[1]
            / self.config.vertical_angle_scale_rad,
        )
        passage_alignment = min(
            _clamp(
                (
                    self.config.passage_margin_norm
                    + aperture_remaining_norm[axis]
                    - (
                        abs(passage_error_norm[axis])
                        + 2.0 * current_std_norm[axis]
                    )
                )
                / (0.5 * self.config.passage_margin_norm),
                0.0,
                1.0,
            )
            for axis in range(2)
        )
        return passage_alignment * prediction.confidence

    def _successor_weight(
        self,
        current: TrackDynamicState,
        successor: TrackDynamicState | None,
        prediction: _SuccessorPrediction | None,
        passage_yaw_authority: float,
        current_yaw_release: float,
    ) -> float:
        if successor is None or prediction is None:
            return 0.0
        if (
            not current.visible
            or current.ambiguous
            or any(current.censored_axes)
            or not successor.visible
            or successor.ambiguous
            or successor.censored_axes[0]
        ):
            return 0.0
        ttc = current.time_to_contact_s
        ttc_progress = (
            1.0
            if ttc is None
            else _clamp(
                (
                    self.config.successor_lookahead_ttc_s - ttc
                )
                / (
                    self.config.successor_lookahead_ttc_s
                    - self.config.successor_full_weight_ttc_s
                ),
                0.0,
                1.0,
            )
        )
        return (
            self.config.successor_maximum_weight
            * passage_yaw_authority
            * current_yaw_release
            * ttc_progress
        )

    @staticmethod
    def _robust_history_rate(
        history: list[tuple[int, Vector2]],
        axis: int,
    ) -> tuple[float, float]:
        if len(history) < 4:
            return 0.0, 0.0
        pairwise_rates: list[float] = []
        for left_index, left in enumerate(history):
            for right in history[left_index + 1 :]:
                elapsed_s = (right[0] - left[0]) / _NS_PER_SECOND
                if elapsed_s >= 0.020:
                    pairwise_rates.append(
                        (right[1][axis] - left[1][axis]) / elapsed_s
                    )
        if not pairwise_rates:
            return 0.0, 0.0
        rate = float(statistics.median(pairwise_rates))
        latest_time_ns, latest_bearing = history[-1]
        residuals = [
            abs(
                sample_bearing[axis]
                - (
                    latest_bearing[axis]
                    + rate
                    * (
                        sample_time_ns - latest_time_ns
                    )
                    / _NS_PER_SECOND
                )
            )
            for sample_time_ns, sample_bearing in history
        ]
        residual_median = float(statistics.median(residuals))
        position_consistency = _clamp(
            1.0 - residual_median / 0.060,
            0.0,
            1.0,
        )
        local_rates = [
            (
                newer[1][axis] - older[1][axis]
            )
            / ((newer[0] - older[0]) / _NS_PER_SECOND)
            for older, newer in zip(history, history[1:])
            if newer[0] - older[0] >= 20_000_000
        ]
        directional_rates = [
            value for value in local_rates if abs(value) >= 0.08
        ]
        if abs(rate) < 0.08 or not directional_rates:
            direction_consistency = 1.0
        else:
            agreeing = sum(value * rate > 0.0 for value in directional_rates)
            direction_consistency = _clamp(
                (agreeing / len(directional_rates) - 0.5) * 2.0,
                0.0,
                1.0,
            )
        sample_confidence = _clamp(
            (len(history) - 3) / 4.0,
            0.0,
            1.0,
        )
        return (
            rate,
            sample_confidence
            * position_consistency
            * direction_consistency,
        )

    def _predicted_successor(
        self,
        current: TrackDynamicState,
        successor: TrackDynamicState | None,
        monotonic_ns: int,
    ) -> _SuccessorPrediction | None:
        if successor is None:
            return None
        estimate = self._tracks[successor.track_id]
        history = estimate.measured_bearing_history
        rates_and_confidence = tuple(
            self._robust_history_rate(history, axis)
            for axis in range(2)
        )
        observation_confidence = (
            float(successor.confidence)
            * _clamp(
                1.0
                - successor.bearing_std_rad[0]
                / self.config.max_bearing_innovation_rad,
                0.0,
                1.0,
            )
        )
        temporal_confidence = rates_and_confidence[0][1]
        confidence = (
            0.0
            if (
                not successor.visible
                or successor.ambiguous
                or successor.censored_axes[0]
            )
            else observation_confidence * temporal_confidence
        )
        base_horizon = min(
            current.time_to_contact_s or 0.0,
            self.config.successor_prediction_max_horizon_s,
        )
        horizon = base_horizon * confidence
        future_bearing = tuple(
            _clamp(
                successor.bearing_rad[axis]
                + rates_and_confidence[axis][0] * horizon,
                -self.config.max_abs_bearing_rad,
                self.config.max_abs_bearing_rad,
            )
            for axis in range(2)
        )
        aligned = self._aligned_imu(monotonic_ns)
        decision_camera_to_world = _quat_multiply(
            aligned.body_to_reference_wxyz,
            self.config.camera_to_body_wxyz,
        )
        def project(stable_bearing: Vector2) -> Vector2 | None:
            world_ray = _quat_rotate(
                estimate.reference_camera_to_world,
                _bearing_ray(stable_bearing),
            )
            decision_ray = _quat_rotate(
                _quat_conjugate(decision_camera_to_world),
                world_ray,
            )
            if decision_ray[0] <= 1e-6:
                return None
            return _ray_bearing(decision_ray)

        measured = project(successor.bearing_rad)
        predicted = project(future_bearing)
        if measured is None or predicted is None:
            return None
        bounded_prediction = tuple(
            measured[axis]
            + _clamp(
                predicted[axis] - measured[axis],
                -self.config.successor_prediction_max_extrapolation_rad,
                self.config.successor_prediction_max_extrapolation_rad,
            )
            for axis in range(2)
        )
        # A low-confidence rate estimate may not extrapolate through the
        # optical axis.  The measured successor can still be used once the
        # near-plane/visibility gates admit it, but noisy rate sign changes
        # cannot manufacture an opposite-side heading target.
        if (
            confidence < 0.75
            and measured[0] * bounded_prediction[0] < 0.0
        ):
            bounded_prediction = (
                measured[0],
                bounded_prediction[1],
            )
        return _SuccessorPrediction(
            bearing_rad=bounded_prediction,
            measured_bearing_rad=measured,
            robust_rate_rad_s=(
                rates_and_confidence[0][0],
                rates_and_confidence[1][0],
            ),
            horizon_s=horizon,
            confidence=confidence,
        )

    def _propose_command(
        self,
        current: TrackDynamicState,
        successor: TrackDynamicState | None,
        camera_current_center_norm: Vector2,
        stable_passage_error_norm: Vector2,
        stable_passage_rate_norm_s: Vector2,
        current_yaw_release: float,
        successor_weight: float,
        successor_prediction: _SuccessorPrediction | None,
        *,
        horizontal_alignment_unsettled: bool,
        vertical_alignment_unsettled: bool,
    ) -> tuple[DynamicCourseCommand, bool, str | None, float]:
        camera_current_bearing = (
            math.atan(
                camera_current_center_norm[0]
                * self.config.horizontal_angle_scale_rad
            ),
            math.atan(
                camera_current_center_norm[1]
                * self.config.vertical_angle_scale_rad
            ),
        )
        stable_passage_bearing = (
            math.atan(
                stable_passage_error_norm[0]
                * self.config.horizontal_angle_scale_rad
            ),
            math.atan(
                stable_passage_error_norm[1]
                * self.config.vertical_angle_scale_rad
            ),
        )
        successor_bearing = (
            camera_current_bearing
            if successor_prediction is None
            else successor_prediction.bearing_rad
        )
        # Early in the approach the current gate owns camera heading.  Once a
        # near-plane passage is both geometrically safe and temporally
        # supported, progressively release that recentering term so successor
        # lookahead can turn the camera without masquerading as passage error.
        current_gate_heading = (
            (1.0 - current_yaw_release)
            * camera_current_bearing[0]
        )
        progressive_successor_limit = (
            self.config.successor_max_yaw_contribution_rad
            * current_yaw_release
        )
        successor_contribution = _clamp(
            successor_weight
            * (successor_bearing[0] - current_gate_heading),
            -progressive_successor_limit,
            progressive_successor_limit,
        )
        prediction_confidence = (
            0.0
            if successor_prediction is None
            else successor_prediction.confidence
        )
        if (
            current_yaw_release <= _EPSILON
            and abs(current_gate_heading) <= _EPSILON
        ):
            successor_contribution = 0.0
        if (
            prediction_confidence < 0.75
            and successor_contribution * current_gate_heading < 0.0
        ):
            successor_contribution = math.copysign(
                min(
                    abs(successor_contribution),
                    0.5 * abs(current_gate_heading),
                ),
                successor_contribution,
            )
        if (
            current_yaw_release < 1.0
            and successor_contribution * current_gate_heading < 0.0
        ):
            successor_contribution = math.copysign(
                min(
                    abs(successor_contribution),
                    (
                        0.5 + 0.5 * current_yaw_release
                    )
                    * abs(current_gate_heading),
                ),
                successor_contribution,
            )
        heading_limit = MAX_YAW_RATE_RAD_S / self.config.yaw_gain
        contribution_heading_limit = (
            0.90 * heading_limit
            if prediction_confidence < 0.75
            else heading_limit
        )
        if successor_contribution * current_gate_heading > 0.0:
            successor_contribution = math.copysign(
                min(
                    abs(successor_contribution),
                    max(
                        0.0,
                        contribution_heading_limit
                        - abs(current_gate_heading),
                    ),
                ),
                successor_contribution,
            )
        heading_error = current_gate_heading + successor_contribution
        # Passage and residual translation are expressed in the current
        # gate's fixed reference.  Intentional body yaw therefore cannot
        # manufacture a lateral intercept error or reverse roll.
        lateral_error = stable_passage_bearing[0]
        roll = self.config.roll_guidance_sign * (
            self.config.roll_gain * lateral_error
            + self.config.lateral_rate_gain
            * stable_passage_rate_norm_s[0]
            * self.config.horizontal_angle_scale_rad
        )
        yaw = -self.config.yaw_gain * heading_error
        # Successor geometry may only slow the current-gate approach with the
        # same progressively admitted authority that governs successor yaw.
        # In particular, a visible but temporally unproved successor must not
        # force full braking while its yaw contribution is still exactly zero.
        successor_brake_authority = _clamp(
            successor_weight / self.config.successor_maximum_weight,
            0.0,
            1.0,
        )
        successor_off_axis = (
            successor is not None
            and successor_brake_authority * abs(successor_bearing[0])
            >= self.config.off_axis_brake_rad
        )
        current_off_axis = (
            abs(stable_passage_bearing[0])
            >= self.config.off_axis_brake_rad
        )
        rapid_closure = (
            current.expansion_rate_s >= self.config.rapid_expansion_rate_s
            or (
                current.time_to_contact_s is not None
                and current.time_to_contact_s
                <= self.config.successor_lookahead_ttc_s
            )
        )
        uncertain = (
            current.censored_axes[0]
            or current.ambiguous
            or current.bearing_std_rad[0] > 0.16
        )
        off_axis_braking = (
            successor is not None
            and (current_off_axis or successor_off_axis)
        )
        uncertain_braking = uncertain and rapid_closure
        # Unsafe current-aperture geometry owns closure immediately in either
        # axis.  Waiting for a second expansion/TTC trigger allowed launch
        # momentum to grow before the q-space crossing envelope had settled.
        current_aperture_braking = (
            horizontal_alignment_unsettled
            or vertical_alignment_unsettled
        )
        braking = (
            off_axis_braking
            or uncertain_braking
            or current_aperture_braking
        )
        reason: str | None
        if vertical_alignment_unsettled:
            reason = "vertical_alignment_unsettled"
        elif horizontal_alignment_unsettled:
            reason = "horizontal_alignment_unsettled"
        elif uncertain_braking:
            reason = "uncertain_rapid_closure"
        elif off_axis_braking:
            reason = (
                "off_axis_rapid_closure"
                if rapid_closure
                else "off_axis_successor_intercept"
            )
        else:
            reason = None
        pitch = self.config.brake_pitch_rad if braking else self.config.advance_pitch_rad
        if not braking and abs(heading_error) > self.config.off_axis_brake_rad:
            pitch = max(pitch, -0.03)
        return (
            DynamicCourseCommand(
                target_roll_rad=_clamp(
                    roll,
                    -MAX_TARGET_ROLL_RAD,
                    MAX_TARGET_ROLL_RAD,
                ),
                target_pitch_rad=_clamp(
                    pitch,
                    MIN_TARGET_PITCH_RAD,
                    MAX_TARGET_PITCH_RAD,
                ),
                yaw_rate_rad_s=_clamp(
                    yaw,
                    -MAX_YAW_RATE_RAD_S,
                    MAX_YAW_RATE_RAD_S,
                ),
                thrust=SUPPORT_THRUST,
            ),
            braking,
            reason,
            successor_contribution,
        )


__all__ = [
    "AppliedCommandSample",
    "CourseDynamicState",
    "CrossingQuotientPrediction",
    "DelayedCommandView",
    "DynamicCourseCommand",
    "DynamicCourseConfig",
    "DynamicCourseCore",
    "DynamicCourseError",
    "GateObservation",
    "GuidanceDecision",
    "ImuAttitudeSample",
    "MAX_TARGET_PITCH_RAD",
    "MAX_TARGET_ROLL_RAD",
    "MAX_THRUST",
    "MAX_YAW_RATE_RAD_S",
    "MIN_TARGET_PITCH_RAD",
    "MIN_THRUST",
    "SUPPORT_THRUST",
    "TrackDynamicState",
    "TrackSteeringPrediction",
    "predict_aperture_relative_crossing",
]
