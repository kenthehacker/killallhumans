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
from dataclasses import dataclass, field

from competition.vq2_contracts import FrameEdge


Quaternion = tuple[float, float, float, float]
Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]

MAX_TARGET_ROLL_RAD = 0.16
MIN_TARGET_PITCH_RAD = -0.30
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


def _move_toward(value: float, target: float, maximum_delta: float) -> float:
    return value + _clamp(target - value, -maximum_delta, maximum_delta)


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

    ``body_to_reference_wxyz`` rotates body-FRD vectors into a stable reference
    frame.  Host monotonic time is the only time domain used by this module.
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
class CommandGovernorConfig:
    max_roll_slew_rad_s: float = 0.45
    max_pitch_slew_rad_s: float = 0.45
    max_yaw_slew_rad_s2: float = 0.45
    max_thrust_slew_s: float = 0.10
    max_roll_accel_rad_s2: float = 2.0
    max_pitch_accel_rad_s2: float = 2.0
    max_yaw_accel_rad_s3: float = 2.0
    max_thrust_accel_s2: float = 0.50
    max_step_s: float = 0.100

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _positive(getattr(self, name), name))


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
    prediction_horizon_s: float = 2.0
    passage_margin_norm: float = 0.09
    passage_successor_bias: float = 0.55
    successor_minimum_weight: float = 0.18
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
    max_history_samples: int = 256
    governor: CommandGovernorConfig = field(default_factory=CommandGovernorConfig)

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
            "prediction_horizon_s",
            "passage_margin_norm",
            "successor_full_weight_ttc_s",
            "successor_lookahead_ttc_s",
            "yaw_gain",
            "roll_gain",
            "lateral_rate_gain",
            "brake_pitch_rad",
            "off_axis_brake_rad",
            "rapid_expansion_rate_s",
            "dropout_hold_s",
        )
        for name in positive:
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in (
            "bearing_alpha",
            "bearing_beta",
            "scale_alpha",
            "scale_beta",
            "residual_alpha",
            "passage_successor_bias",
            "successor_minimum_weight",
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
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        advance = _finite(self.advance_pitch_rad, "advance_pitch_rad")
        if not MIN_TARGET_PITCH_RAD <= advance <= 0.0:
            raise DynamicCourseError("advance_pitch_rad must be a bounded forward target")
        if self.brake_pitch_rad > MAX_TARGET_PITCH_RAD:
            raise DynamicCourseError("brake_pitch_rad exceeds pitch authority")
        if self.minimum_ttc_s >= self.maximum_ttc_s:
            raise DynamicCourseError("minimum_ttc_s must be below maximum_ttc_s")
        if self.successor_full_weight_ttc_s >= self.successor_lookahead_ttc_s:
            raise DynamicCourseError(
                "successor full-weight TTC must precede lookahead TTC"
            )
        if type(self.max_history_samples) is not int:
            raise TypeError("max_history_samples must be an exact integer")
        if self.max_history_samples < 8:
            raise DynamicCourseError("max_history_samples must be at least 8")
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
    bearing_rad: Vector2
    bearing_rate_rad_s: Vector2
    log_scale: float
    expansion_rate_s: float
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
    current_aperture_half_size_norm: Vector2 | None
    passage_point_norm: Vector2
    passage_error_norm: Vector2
    aperture_margin_norm: Vector2
    current_bearing_std_rad: Vector2
    successor_bearing_std_rad: Vector2 | None
    successor_weight: float
    predicted_successor_bearing_rad: Vector2 | None
    current_time_to_contact_s: float | None
    braking: bool
    brake_reason: str | None
    dropout_held: bool
    proposed_command: DynamicCourseCommand
    command: DynamicCourseCommand


@dataclass(frozen=True, slots=True)
class CourseDynamicState:
    current_gate_index: int
    current_track_id: str
    successor_track_id: str | None
    current: TrackDynamicState
    successor: TrackDynamicState | None
    recent_commands: tuple[AppliedCommandSample, ...]
    last_governed_command: DynamicCourseCommand | None
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


class CommandGovernor:
    """Slew- and acceleration-bounded command continuity.

    ``preview`` is side-effect free.  Only ``commit`` consumes continuity
    budget, so a dropped or superseded proposal cannot alter later commands.
    """

    def __init__(self, config: CommandGovernorConfig | None = None) -> None:
        self.config = config or CommandGovernorConfig()
        self._last_ns: int | None = None
        self._last: DynamicCourseCommand | None = None
        self._rates = (0.0, 0.0, 0.0, 0.0)

    @property
    def last_command(self) -> DynamicCourseCommand | None:
        return self._last

    def preview(
        self,
        proposal: DynamicCourseCommand,
        monotonic_ns: int,
        *,
        hold: bool = False,
    ) -> DynamicCourseCommand:
        _exact_nonnegative_int(monotonic_ns, "monotonic_ns")
        if self._last_ns is not None and monotonic_ns <= self._last_ns:
            raise DynamicCourseError("governor time must advance")
        if self._last is None:
            return proposal
        assert self._last_ns is not None and self._last is not None
        if hold:
            return self._last
        dt = min(
            (monotonic_ns - self._last_ns) / _NS_PER_SECOND,
            self.config.max_step_s,
        )
        previous = self._last
        targets = (
            proposal.target_roll_rad,
            proposal.target_pitch_rad,
            proposal.yaw_rate_rad_s,
            proposal.thrust,
        )
        values = (
            previous.target_roll_rad,
            previous.target_pitch_rad,
            previous.yaw_rate_rad_s,
            previous.thrust,
        )
        slews = (
            self.config.max_roll_slew_rad_s,
            self.config.max_pitch_slew_rad_s,
            self.config.max_yaw_slew_rad_s2,
            self.config.max_thrust_slew_s,
        )
        accelerations = (
            self.config.max_roll_accel_rad_s2,
            self.config.max_pitch_accel_rad_s2,
            self.config.max_yaw_accel_rad_s3,
            self.config.max_thrust_accel_s2,
        )
        advanced: list[float] = []
        rates: list[float] = []
        for index, (value, target, slew, acceleration) in enumerate(
            zip(values, targets, slews, accelerations)
        ):
            desired_rate = _clamp((target - value) / dt, -slew, slew)
            rate = _move_toward(
                self._rates[index],
                desired_rate,
                acceleration * dt,
            )
            next_value = value + rate * dt
            if (target - value) * (target - next_value) <= 0.0:
                next_value = target
                rate = (next_value - value) / dt
            if index == 0 and value * next_value < 0.0:
                next_value = 0.0
                rate = -value / dt
            advanced.append(next_value)
            rates.append(rate)
        command = DynamicCourseCommand(
            target_roll_rad=advanced[0],
            target_pitch_rad=advanced[1],
            yaw_rate_rad_s=advanced[2],
            thrust=advanced[3],
        )
        return command

    def commit(
        self,
        command: DynamicCourseCommand,
        monotonic_ns: int,
        *,
        discontinuity: bool = False,
    ) -> None:
        """Synchronise to the command actually accepted by the wire.

        ``discontinuity`` is reserved for an outer safety/cleanup bypass.  It
        synchronises the governor without pretending the emergency jump obeyed
        normal slew limits.
        """

        _exact_nonnegative_int(monotonic_ns, "monotonic_ns")
        if self._last_ns is not None and monotonic_ns <= self._last_ns:
            raise DynamicCourseError("governor commit time must advance")
        if self._last is None or discontinuity:
            rates = (0.0, 0.0, 0.0, 0.0)
        else:
            assert self._last_ns is not None
            dt = (monotonic_ns - self._last_ns) / _NS_PER_SECOND
            previous = (
                self._last.target_roll_rad,
                self._last.target_pitch_rad,
                self._last.yaw_rate_rad_s,
                self._last.thrust,
            )
            current = (
                command.target_roll_rad,
                command.target_pitch_rad,
                command.yaw_rate_rad_s,
                command.thrust,
            )
            rates = tuple(
                (current[index] - previous[index]) / dt for index in range(4)
            )
        self._last = command
        self._last_ns = monotonic_ns
        self._rates = rates

    def apply(
        self,
        proposal: DynamicCourseCommand,
        monotonic_ns: int,
        *,
        hold: bool = False,
    ) -> DynamicCourseCommand:
        """Preview and commit convenience for isolated tests."""

        command = self.preview(proposal, monotonic_ns, hold=hold)
        self.commit(command, monotonic_ns)
        return command


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
        self._governor = CommandGovernor(self.config.governor)

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

    def record_applied_command(self, sample: AppliedCommandSample) -> None:
        if self._commands and sample.monotonic_ns <= self._commands[-1].monotonic_ns:
            raise DynamicCourseError("command samples must advance monotonically")
        self._commands.append(sample)
        self._trim(self._commands)
        self._governor.commit(
            DynamicCourseCommand(
                target_roll_rad=sample.target_roll_rad,
                target_pitch_rad=sample.target_pitch_rad,
                yaw_rate_rad_s=sample.yaw_rate_rad_s,
                thrust=sample.thrust,
            ),
            sample.monotonic_ns,
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
                ),
            )
            return state
        rotational_rate, residual_rate = self._split_image_rate(
            existing,
            camera_to_world,
            raw_angle,
            capture_ns,
        )
        state = self._update_track(
            existing.state,
            observation,
            capture_ns,
            aligned,
            delayed,
            measured_bearing,
            rotational_rate,
            residual_rate,
        )
        existing.state = state
        existing.last_measured_raw_angle = raw_angle
        existing.last_measured_stable_ray = stable_ray
        existing.last_raw_measurement_ns = capture_ns
        existing.last_measurement_camera_to_world = camera_to_world
        existing.last_measured_center_norm = observation.center_norm
        if observation.aperture_half_size_norm is not None:
            existing.last_measured_aperture_half_size_norm = (
                observation.aperture_half_size_norm
            )
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
            raw_log_scale=observation.log_scale,
            aperture_half_size_norm=observation.aperture_half_size_norm,
            bearing_rad=bearing,
            bearing_rate_rad_s=(0.0, 0.0),
            log_scale=observation.log_scale,
            expansion_rate_s=0.0,
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
        measured_residual_rate: Vector2,
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
        beta = self.config.bearing_beta * quality
        censored = observation.censored_axes
        bearing_values: list[float] = []
        bearing_rates: list[float] = []
        residual_rates: list[float] = []
        bearing_std: list[float] = []
        rate_std: list[float] = []
        measurement_std_rad = (
            observation.measurement_std[0] * self.config.horizontal_angle_scale_rad,
            observation.measurement_std[1] * self.config.vertical_angle_scale_rad,
        )
        for axis in range(2):
            if censored[axis]:
                value = predicted_bearing[axis]
                rate = predicted_rate[axis]
                residual = previous.residual_translational_rate_rad_s[axis]
                multiplier = self.config.clipping_uncertainty_multiplier
            else:
                value, rate = self._robust_update(
                    predicted_bearing[axis],
                    predicted_rate[axis],
                    measured_bearing[axis],
                    dt,
                    alpha,
                    beta,
                    self.config.max_bearing_innovation_rad,
                    self.config.max_abs_bearing_rate_rad_s,
                )
                residual = previous.residual_translational_rate_rad_s[axis] + (
                    self.config.residual_alpha
                    * (
                        measured_residual_rate[axis]
                        - previous.residual_translational_rate_rad_s[axis]
                    )
                )
                residual = _clamp(
                    residual,
                    -self.config.max_abs_bearing_rate_rad_s,
                    self.config.max_abs_bearing_rate_rad_s,
                )
                multiplier = 1.0
            bearing_values.append(value)
            bearing_rates.append(rate)
            residual_rates.append(residual)
            predicted_std = previous.bearing_std_rad[axis] + (
                self.config.process_noise_bearing_rad_s * dt
            )
            bearing_std.append(
                multiplier
                * math.sqrt(
                    max(
                        1e-10,
                        (1.0 - alpha) * predicted_std * predicted_std
                        + alpha * measurement_std_rad[axis] ** 2,
                    )
                )
            )
            rate_std.append(
                multiplier
                * min(
                    self.config.max_abs_bearing_rate_rad_s,
                    previous.rate_std_rad_s[axis]
                    + self.config.process_noise_bearing_rad_s,
                )
            )
        if observation.clipping == FrameEdge.NONE:
            log_scale, expansion = self._robust_update(
                predicted_scale,
                predicted_expansion,
                observation.log_scale,
                dt,
                self.config.scale_alpha * quality,
                self.config.scale_beta * quality,
                self.config.max_log_scale_innovation,
                self.config.max_abs_expansion_rate_s,
            )
            scale_multiplier = 1.0
        else:
            log_scale = predicted_scale
            expansion = predicted_expansion
            scale_multiplier = self.config.clipping_uncertainty_multiplier
        ttc = self._time_to_contact(expansion)
        return TrackDynamicState(
            track_id=observation.track_id,
            stream_generation=observation.stream_generation,
            state_monotonic_ns=capture_ns,
            last_measurement_monotonic_ns=capture_ns,
            frame_sequence=observation.frame_sequence,
            capture_timing_basis=observation.timing_basis,
            capture_timing_uncertainty_s=observation.timing_uncertainty_s,
            raw_center_norm=observation.center_norm,
            raw_log_scale=observation.log_scale,
            aperture_half_size_norm=(
                observation.aperture_half_size_norm
                if observation.aperture_half_size_norm is not None
                else previous.aperture_half_size_norm
            ),
            bearing_rad=(bearing_values[0], bearing_values[1]),
            bearing_rate_rad_s=(bearing_rates[0], bearing_rates[1]),
            log_scale=log_scale,
            expansion_rate_s=expansion,
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
            log_scale_std=scale_multiplier
            * math.sqrt(
                max(
                    1e-10,
                    (1.0 - self.config.scale_alpha)
                    * (
                        previous.log_scale_std
                        + self.config.process_noise_scale_s * dt
                    )
                    ** 2
                    + self.config.scale_alpha * observation.measurement_std[2] ** 2,
                )
            ),
            expansion_rate_std_s=scale_multiplier
            * min(
                self.config.max_abs_expansion_rate_s,
                previous.expansion_rate_std_s
                + self.config.process_noise_scale_s,
            ),
            clipping=observation.clipping,
            censored_axes=censored,
            visible=True,
            ambiguous=observation.ambiguous,
            confidence=observation.confidence,
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
            aperture_half_size_norm=previous.aperture_half_size_norm,
            bearing_rad=bearing,
            bearing_rate_rad_s=rate,
            log_scale=log_scale,
            expansion_rate_s=expansion,
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
        self._current_gate_index = current_gate_index
        self._current_track_id = current_track_id
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
            last_governed_command=self._governor.last_command,
            promotion_count=self._promotion_count,
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
        self._promotion_count += 1
        self._last_promotion_ns = monotonic_ns
        return self.course_state()

    def guide(self, monotonic_ns: int) -> GuidanceDecision:
        _exact_nonnegative_int(monotonic_ns, "monotonic_ns")
        if self._governor.last_command is None:
            raise DynamicCourseError(
                "guidance requires a confirmed applied command to seed continuity"
            )
        state = self.course_state()
        current = state.current
        successor = state.successor
        if monotonic_ns < current.state_monotonic_ns:
            raise DynamicCourseError("guidance time cannot precede the current state")
        current_center, current_aperture = self._decision_geometry(
            current.track_id,
            monotonic_ns,
        )
        successor_center: Vector2 | None = None
        if successor is not None:
            successor_center, _ = self._decision_geometry(
                successor.track_id,
                monotonic_ns,
            )
        held = False
        if self._governor.last_command is not None:
            age_s = (
                monotonic_ns - current.last_measurement_monotonic_ns
            ) / _NS_PER_SECOND
            held = not current.visible and age_s <= self.config.dropout_hold_s
        passage, margins = self._passage_point(
            current_aperture,
            current_center,
            successor_center,
        )
        successor_weight = self._successor_weight(current, successor)
        predicted_successor = self._predicted_successor(
            current,
            successor,
            monotonic_ns,
        )
        proposal, braking, reason = self._propose_command(
            current,
            successor,
            current_center,
            passage,
            successor_weight,
            predicted_successor,
        )
        command = self._governor.preview(proposal, monotonic_ns, hold=held)
        return GuidanceDecision(
            monotonic_ns=monotonic_ns,
            current_gate_index=state.current_gate_index,
            current_track_id=current.track_id,
            successor_track_id=None if successor is None else successor.track_id,
            current_center_norm=current_center,
            current_aperture_half_size_norm=current_aperture,
            passage_point_norm=passage,
            passage_error_norm=(
                current_center[0] + passage[0],
                current_center[1] + passage[1],
            ),
            aperture_margin_norm=margins,
            current_bearing_std_rad=current.bearing_std_rad,
            successor_bearing_std_rad=(
                None if successor is None else successor.bearing_std_rad
            ),
            successor_weight=successor_weight,
            predicted_successor_bearing_rad=predicted_successor,
            current_time_to_contact_s=current.time_to_contact_s,
            braking=braking,
            brake_reason=reason,
            dropout_held=held,
            proposed_command=proposal,
            command=command,
        )

    def _decision_geometry(
        self,
        track_id: str,
        monotonic_ns: int,
    ) -> tuple[Vector2, Vector2 | None]:
        estimate = self._tracks[track_id]
        aligned = self._aligned_imu(monotonic_ns)
        decision_camera_to_world = _quat_multiply(
            aligned.body_to_reference_wxyz,
            self.config.camera_to_body_wxyz,
        )

        def project_stable_ray(stable_ray: Vector3) -> Vector2:
            world_ray = _quat_rotate(
                estimate.reference_camera_to_world,
                stable_ray,
            )
            decision_ray = _quat_rotate(
                _quat_conjugate(decision_camera_to_world),
                world_ray,
            )
            if decision_ray[0] <= 1e-6:
                raise DynamicCourseError("gate ray is behind the decision camera")
            bearing = _ray_bearing(decision_ray)
            return (
                math.tan(bearing[0]) / self.config.horizontal_angle_scale_rad,
                math.tan(bearing[1]) / self.config.vertical_angle_scale_rad,
            )

        center = project_stable_ray(_bearing_ray(estimate.state.bearing_rad))
        aperture = estimate.last_measured_aperture_half_size_norm
        if aperture is None:
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
            decision_ray = _quat_rotate(
                _quat_conjugate(decision_camera_to_world),
                world_ray,
            )
            if decision_ray[0] <= 1e-6:
                raise DynamicCourseError("gate aperture is behind the decision camera")
            bearing = _ray_bearing(decision_ray)
            return (
                math.tan(bearing[0]) / self.config.horizontal_angle_scale_rad,
                math.tan(bearing[1]) / self.config.vertical_angle_scale_rad,
            )

        measured_center = estimate.last_measured_center_norm
        left = reproject_last(
            (measured_center[0] - aperture[0], measured_center[1])
        )
        right = reproject_last(
            (measured_center[0] + aperture[0], measured_center[1])
        )
        top = reproject_last(
            (measured_center[0], measured_center[1] - aperture[1])
        )
        bottom = reproject_last(
            (measured_center[0], measured_center[1] + aperture[1])
        )
        projected_aperture = (
            0.5 * abs(right[0] - left[0]),
            0.5 * abs(bottom[1] - top[1]),
        )
        return center, projected_aperture

    def _passage_point(
        self,
        aperture_half_size_norm: Vector2 | None,
        current_center_norm: Vector2,
        successor_center_norm: Vector2 | None,
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
        offset = (
            successor_center_norm[0] - current_center_norm[0],
            successor_center_norm[1] - current_center_norm[1],
        )
        passage = tuple(
            _clamp(
                self.config.passage_successor_bias * offset[axis],
                -available[axis],
                available[axis],
            )
            for axis in range(2)
        )
        remaining = tuple(
            max(0.0, available[axis] - abs(passage[axis])) for axis in range(2)
        )
        return passage, remaining  # type: ignore[return-value]

    def _successor_weight(
        self,
        current: TrackDynamicState,
        successor: TrackDynamicState | None,
    ) -> float:
        if successor is None:
            return 0.0
        ttc = current.time_to_contact_s
        if ttc is None:
            return self.config.successor_minimum_weight
        fraction = (
            self.config.successor_lookahead_ttc_s - ttc
        ) / (
            self.config.successor_lookahead_ttc_s
            - self.config.successor_full_weight_ttc_s
        )
        return max(
            self.config.successor_minimum_weight,
            _clamp(fraction, 0.0, 1.0),
        )

    def _predicted_successor(
        self,
        current: TrackDynamicState,
        successor: TrackDynamicState | None,
        monotonic_ns: int,
    ) -> Vector2 | None:
        if successor is None:
            return None
        horizon = min(
            current.time_to_contact_s or self.config.prediction_horizon_s,
            self.config.prediction_horizon_s,
        )
        future_bearing = (
            successor.bearing_rad[0] + successor.bearing_rate_rad_s[0] * horizon,
            successor.bearing_rad[1] + successor.bearing_rate_rad_s[1] * horizon,
        )
        estimate = self._tracks[successor.track_id]
        aligned = self._aligned_imu(monotonic_ns)
        decision_camera_to_world = _quat_multiply(
            aligned.body_to_reference_wxyz,
            self.config.camera_to_body_wxyz,
        )
        world_ray = _quat_rotate(
            estimate.reference_camera_to_world,
            _bearing_ray(future_bearing),
        )
        decision_ray = _quat_rotate(
            _quat_conjugate(decision_camera_to_world),
            world_ray,
        )
        if decision_ray[0] <= 1e-6:
            return None
        bearing = _ray_bearing(decision_ray)
        return tuple(
            _clamp(
                bearing[axis],
                -self.config.max_abs_bearing_rad,
                self.config.max_abs_bearing_rad,
            )
            for axis in range(2)
        )  # type: ignore[return-value]

    def _propose_command(
        self,
        current: TrackDynamicState,
        successor: TrackDynamicState | None,
        current_center_norm: Vector2,
        passage: Vector2,
        successor_weight: float,
        predicted_successor: Vector2 | None,
    ) -> tuple[DynamicCourseCommand, bool, str | None]:
        current_bearing = (
            math.atan(
                current_center_norm[0] * self.config.horizontal_angle_scale_rad
            ),
            math.atan(
                current_center_norm[1] * self.config.vertical_angle_scale_rad
            ),
        )
        passage_bearing = (
            math.atan(passage[0] * self.config.horizontal_angle_scale_rad),
            math.atan(passage[1] * self.config.vertical_angle_scale_rad),
        )
        successor_bearing = predicted_successor or current_bearing
        heading_error = (
            (1.0 - successor_weight) * current_bearing[0]
            + successor_weight * successor_bearing[0]
        )
        lateral_error = (
            current_bearing[0]
            + passage_bearing[0]
            + successor_weight * successor_bearing[0]
        )
        roll = self.config.roll_guidance_sign * (
            self.config.roll_gain * lateral_error
            + self.config.lateral_rate_gain
            * current.residual_translational_rate_rad_s[0]
        )
        yaw = -self.config.yaw_gain * heading_error
        off_axis = max(
            abs(current_bearing[0] + passage_bearing[0]),
            abs(successor_bearing[0]) if successor is not None else 0.0,
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
        braking = (
            off_axis >= self.config.off_axis_brake_rad
            and successor is not None
        ) or (uncertain and rapid_closure)
        reason: str | None
        if braking and uncertain:
            reason = "uncertain_rapid_closure"
        elif braking:
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
        )


__all__ = [
    "AppliedCommandSample",
    "CommandGovernor",
    "CommandGovernorConfig",
    "CourseDynamicState",
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
]
