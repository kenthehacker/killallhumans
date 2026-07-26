"""Pure image-based visual servo for build-3385 VQ2 gate navigation.

The controller deliberately stays in normalized image space.  It does not
claim calibrated camera intrinsics, camera-to-body extrinsics, metric range,
or a world-frame gate map.  The only signed heading authority used here is the
live-identified build-3385 relationship: a positive controller yaw command
moves the tracked gate toward positive image x.  Consequently, positive image
x error requires a negative yaw command.

Runtime safety remains owned by :mod:`scripts.aigp_vq2_run`.  The immutable
ceilings in this module are an additional, narrower controller envelope; they
are not caller-configurable and do not replace command, attitude, freshness,
collision, stage-duration, reset, or cleanup watchdogs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    VisualTrack,
    VisualTrackRole,
    VisualTrackSample,
)


# These are code-owned controller authority ceilings.  Repeated saturated
# 0.12-rad/s successor corrections continued outward, so production consumes
# the collision-free paired-polarity 0.15-rad/s build-3385 capability tier.
# The excursion is a separate per-segment course-turn envelope, not a
# calibration limit.
MAX_VISUAL_YAW_RATE_RAD_S = 0.15
MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD = 0.65
VISUAL_SEGMENT_YAW_SOFT_STOP_RAD = 0.60
MAX_VISUAL_SEGMENT_DURATION_S = 8.0
MAX_VISUAL_TARGET_ROLL_RAD = 0.16
MIN_VISUAL_TARGET_PITCH_RAD = -0.35
MAX_VISUAL_TARGET_PITCH_RAD = 0.15
MIN_VISUAL_THRUST = 0.21
MAX_VISUAL_THRUST = 0.32
MAX_VISUAL_OBSERVATION_AGE_S = 0.10
MAX_NEXT_GATE_BLEND = 0.35
# A pre-pass orientation may continue inside a broader current-aperture
# corridor only after the next identity has first completed the ordinary
# tight-corridor dwell.  Passage may retain this bounded preview while
# requesting advance, but the current aperture remains primary and any failed
# predicate withdraws preview authority.  These immutable continuation bounds
# are grounded in the first exact live handoff trace.  All 54 jointly fresh
# publications stayed below |x|=0.022, |y|=0.234, |vx|=0.265, |vy|=0.439 and
# current log-scale rate=1.591/s.  The short projection rejects high image
# momentum before an instantaneous center can consume the remaining margin.
PREPASS_CURRENT_MAX_ABS_X_NORM = 0.20
PREPASS_CURRENT_MAX_ABS_Y_NORM = 0.28
PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S = 0.60
PREPASS_CURRENT_MIN_LOG_SCALE_RATE_S = -1.50
PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S = 2.00
PREPASS_CURRENT_MAX_APPARENT_SCALE = 0.55
PREPASS_CURRENT_PROJECTION_HORIZON_S = 0.10
# The latest exact replay's four projection-only misses exceeded the vertical
# bound by at most 0.00537 normalized image units.  A transient lease is
# permitted only inside this additional immutable margin; a larger prediction
# miss is a hard withdrawal even when every instantaneous predicate passes.
MAX_TRANSIENT_PROJECTED_VERTICAL_EXCESS_NORM = 0.01
PREPASS_NEXT_MAX_ABS_CENTER_RATE_NORM_S = 0.60
PREPASS_NEXT_MAX_ABS_LOG_SCALE_RATE_S = 1.10
VISUAL_CLOSE_SCALE_BRAKE_LOG = -0.18
VISUAL_RAPID_RETREAT_LOG_SCALE_RATE_S = -1.50


class VisualServoRefusal(ValueError):
    """The observation or requested tuning cannot safely produce authority."""


class PassageSafetyViolation(str, Enum):
    """Structured reasons a latched blend lost passage authority."""

    CURRENT_GEOMETRY_CENSORED = "current_geometry_censored"
    CURRENT_HORIZONTAL_POSITION = "current_horizontal_position"
    CURRENT_VERTICAL_POSITION = "current_vertical_position"
    CURRENT_HORIZONTAL_RATE = "current_horizontal_rate"
    CURRENT_VERTICAL_RATE = "current_vertical_rate"
    CURRENT_LOG_SCALE_RATE = "current_log_scale_rate"
    CURRENT_PROJECTED_HORIZONTAL = "current_projected_horizontal"
    CURRENT_PROJECTED_VERTICAL = "current_projected_vertical"
    CURRENT_APPARENT_SCALE = "current_apparent_scale"
    CURRENT_HORIZONTAL_CORRECTION_REVERSAL = (
        "current_horizontal_correction_reversal"
    )
    CURRENT_VERTICAL_CORRECTION_REVERSAL = (
        "current_vertical_correction_reversal"
    )


@dataclass(frozen=True)
class PassageSafetyViolationDetail:
    """One failed immutable predicate and its scalar margin."""

    violation: PassageSafetyViolation
    observed: float
    limit: float
    excess: float

    def __post_init__(self) -> None:
        if type(self.violation) is not PassageSafetyViolation:
            raise ValueError("passage violation detail requires an exact code")
        for name, value in (
            ("observed", self.observed),
            ("limit", self.limit),
            ("excess", self.excess),
        ):
            if (
                type(value) not in {int, float}
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"passage violation {name} must be finite")
        if float(self.excess) <= 0.0:
            raise ValueError("passage violation excess must be positive")


class VisualServoPassageSafetyUnavailable(VisualServoRefusal):
    """A latched pre-pass blend left its immutable passage corridor."""

    def __init__(
        self,
        message: str,
        *,
        details: Tuple[PassageSafetyViolationDetail, ...],
    ) -> None:
        if (
            type(details) is not tuple
            or not details
            or any(
                type(detail) is not PassageSafetyViolationDetail
                for detail in details
            )
            or len(details)
            != len({detail.violation for detail in details})
        ):
            raise ValueError(
                "passage-safety refusal requires unique structured violations"
            )
        self.details = details
        self.violations = tuple(detail.violation for detail in details)
        # The latest exact build-3385 replay proves only a short projected
        # vertical excursion that re-enters every unchanged predicate.  Do not
        # generalize the suspension lease to raw geometry, rate, scale, or
        # correction-reversal failures.
        self.transient_projection_only = bool(
            self.violations
            == (PassageSafetyViolation.CURRENT_PROJECTED_VERTICAL,)
            and details[0].observed
            < -PREPASS_CURRENT_MAX_ABS_Y_NORM
            and details[0].limit
            == -PREPASS_CURRENT_MAX_ABS_Y_NORM
            and math.isclose(
                details[0].limit - details[0].observed,
                details[0].excess,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and details[0].excess
            <= MAX_TRANSIENT_PROJECTED_VERTICAL_EXCESS_NORM
        )
        super().__init__(message)


@dataclass(frozen=True)
class VisualServoTuning:
    """Bounded controller choices below immutable runtime authority ceilings."""

    horizontal_corridor: float = 0.16
    vertical_corridor: float = 0.18
    edge_brake_x: float = 0.72
    edge_brake_y: float = 0.76
    stable_rate_norm_s: float = 0.30
    stable_scale_rate_s: float = 1.10
    brake_scale_rate_s: float = 2.00
    # The first two exact Gate-0 -> Gate-1 handoffs showed that a 0.15 gain
    # behind a 0.25 bearing blend produced only about 0.012 rad/s of preview
    # yaw and left the promoted track moving outward at 0.304-0.349 norm/s.
    # Use the reviewed tuning ceiling so the generic servo can exploit the
    # separately immutable yaw-rate and course-turn heading envelopes.
    yaw_error_gain: float = 0.30
    yaw_rate_gain: float = 0.035
    # Gate-1 live recovery saturated yaw while horizontal error grew from
    # 0.625 to 0.750.  Use materially stronger coordinated bank inside the
    # separately enforced 0.18-rad measured stage corridor.
    roll_error_gain: float = 0.20
    roll_rate_gain: float = 0.05
    # Retained in the serialized v1 configuration for manifest compatibility.
    # Vertical image feedback is no longer applied to pitch; collective is its
    # single control owner.
    vertical_error_gain: float = 0.16
    vertical_rate_gain: float = 0.035
    collective_error_gain: float = 0.060
    collective_rate_gain: float = 0.080
    advance_pitch_rad: float = -0.105
    brake_pitch_rad: float = 0.035
    # Repeated credited Gate-0 runs establish 0.275 as the generic
    # flight-support collective basis.  Forward closure is allocated through
    # pitch and the small continuous interpolation toward advance collective;
    # cutting airborne alignment to the 0.21 envelope minimum caused measured
    # vertical image divergence and top censorship.
    align_thrust: float = 0.275
    advance_thrust: float = 0.295
    brake_thrust: float = 0.275
    required_corridor_frames: int = 3

    def __post_init__(self) -> None:
        numeric = {
            name: float(value)
            for name, value in vars(self).items()
            if name != "required_corridor_frames"
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            raise VisualServoRefusal("visual-servo tuning must be finite")
        if not 0.08 <= self.horizontal_corridor <= 0.25:
            raise VisualServoRefusal("horizontal corridor is outside bounds")
        if not 0.08 <= self.vertical_corridor <= 0.28:
            raise VisualServoRefusal("vertical corridor is outside bounds")
        if not 0.55 <= self.edge_brake_x <= 0.85:
            raise VisualServoRefusal("horizontal edge brake is outside bounds")
        if not 0.55 <= self.edge_brake_y <= 0.85:
            raise VisualServoRefusal("vertical edge brake is outside bounds")
        if not 0.10 <= self.stable_rate_norm_s <= 0.60:
            raise VisualServoRefusal("stable image-rate bound is outside bounds")
        if not 0.50 <= self.stable_scale_rate_s <= 1.50:
            raise VisualServoRefusal("stable scale-rate bound is outside bounds")
        if not self.stable_scale_rate_s < self.brake_scale_rate_s <= 3.0:
            raise VisualServoRefusal("scale-rate braking bounds are invalid")
        if not 0.05 <= self.yaw_error_gain <= 0.30:
            raise VisualServoRefusal("yaw error gain is outside bounds")
        if not 0.0 <= self.yaw_rate_gain <= 0.08:
            raise VisualServoRefusal("yaw rate gain is outside bounds")
        if not 0.0 <= self.roll_error_gain <= 0.20:
            raise VisualServoRefusal("roll error gain is outside bounds")
        if not 0.0 <= self.roll_rate_gain <= 0.05:
            raise VisualServoRefusal("roll rate gain is outside bounds")
        if not 0.0 <= self.collective_error_gain <= 0.08:
            raise VisualServoRefusal("collective error gain is outside bounds")
        if not 0.0 <= self.collective_rate_gain <= 0.13:
            raise VisualServoRefusal("collective rate gain is outside bounds")
        if not -0.16 <= self.advance_pitch_rad <= -0.06:
            raise VisualServoRefusal("advance pitch is outside bounds")
        if not 0.0 <= self.brake_pitch_rad <= 0.08:
            raise VisualServoRefusal("brake pitch is outside bounds")
        if not MIN_VISUAL_THRUST <= self.align_thrust <= 0.29:
            raise VisualServoRefusal("alignment thrust is outside bounds")
        if not 0.27 <= self.advance_thrust <= MAX_VISUAL_THRUST:
            raise VisualServoRefusal("advance thrust is outside bounds")
        if not MIN_VISUAL_THRUST <= self.brake_thrust <= 0.29:
            raise VisualServoRefusal("brake thrust is outside bounds")
        if type(self.required_corridor_frames) is not int or not (
            3 <= self.required_corridor_frames <= 8
        ):
            raise VisualServoRefusal("required corridor frames are outside bounds")


@dataclass(frozen=True, order=True)
class ServoFrameToken:
    """Exact receiver generation and camera publication identity."""

    stream_id: str
    generation: int
    frame_id: int
    publication_sequence: int

    def __post_init__(self) -> None:
        if (
            type(self.stream_id) is not str
            or not self.stream_id
            or len(self.stream_id) > 128
        ):
            raise VisualServoRefusal("frame stream id must be a bounded string")
        if type(self.generation) is not int or self.generation < 0:
            raise VisualServoRefusal("frame generation must be a nonnegative int")
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise VisualServoRefusal("frame id must be a nonnegative int")
        if (
            type(self.publication_sequence) is not int
            or self.publication_sequence <= 0
        ):
            raise VisualServoRefusal(
                "frame publication sequence must be a positive int"
            )


@dataclass(frozen=True)
class VisualTarget:
    """Image-only target state accepted from the multi-target gate graph."""

    track_id: str
    frame_token: ServoFrameToken
    received_monotonic_s: float
    normalized_x: float
    normalized_y_down: float
    normalized_x_rate_s: float
    normalized_y_rate_down_s: float
    log_scale: float
    log_scale_rate_s: float
    confidence: float
    association_confidence: float
    consecutive_frames: int
    clipped: bool = False
    center_censored: bool = False
    horizontal_censored: bool = False
    vertical_censored: bool = False
    ambiguous: bool = False

    def __post_init__(self) -> None:
        values = (
            self.received_monotonic_s,
            self.normalized_x,
            self.normalized_y_down,
            self.normalized_x_rate_s,
            self.normalized_y_rate_down_s,
            self.log_scale,
            self.log_scale_rate_s,
            self.confidence,
            self.association_confidence,
        )
        if (
            type(self.track_id) is not str
            or not self.track_id
            or len(self.track_id) > 128
        ):
            raise VisualServoRefusal("track id must be a bounded string")
        if not all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in values
        ):
            raise VisualServoRefusal("visual target fields must be finite")
        if abs(float(self.normalized_x)) > 1.25:
            raise VisualServoRefusal("horizontal target coordinate is implausible")
        if abs(float(self.normalized_y_down)) > 1.25:
            raise VisualServoRefusal(
                "image-down vertical target coordinate is implausible"
            )
        if abs(float(self.normalized_x_rate_s)) > 8.0:
            raise VisualServoRefusal("horizontal target rate is implausible")
        if abs(float(self.normalized_y_rate_down_s)) > 8.0:
            raise VisualServoRefusal(
                "image-down vertical target rate is implausible"
            )
        if abs(float(self.log_scale_rate_s)) > 12.0:
            raise VisualServoRefusal("target scale rate is implausible")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise VisualServoRefusal("target confidence is outside [0, 1]")
        if not 0.0 <= float(self.association_confidence) <= 1.0:
            raise VisualServoRefusal(
                "association confidence is outside [0, 1]"
            )
        if type(self.consecutive_frames) is not int or self.consecutive_frames < 1:
            raise VisualServoRefusal("target consecutive-frame count is invalid")
        if (
            type(self.clipped) is not bool
            or type(self.center_censored) is not bool
            or type(self.horizontal_censored) is not bool
            or type(self.vertical_censored) is not bool
            or type(self.ambiguous) is not bool
        ):
            raise VisualServoRefusal("target flags must be exact booleans")

    @property
    def horizontal_geometry_censored(self) -> bool:
        """Whether image x geometry is unsafe for steering.

        The two explicit axis flags are the production contract.  Treat an
        older undifferentiated ``clipped``/``center_censored`` value as
        censoring both axes so compatibility cannot accidentally create
        authority.
        """

        if self.horizontal_censored:
            return True
        return bool(
            not self.horizontal_censored
            and not self.vertical_censored
            and (self.clipped or self.center_censored)
        )

    @property
    def vertical_geometry_censored(self) -> bool:
        """Whether image y geometry is unsafe for steering."""

        if self.vertical_censored:
            return True
        return bool(
            not self.horizontal_censored
            and not self.vertical_censored
            and (self.clipped or self.center_censored)
        )

    @classmethod
    def from_visual_track(
        cls,
        track: VisualTrack,
        *,
        require_current_authority: bool = True,
        expected_gate_index: Optional[int] = None,
    ) -> "VisualTarget":
        """Adapt one exact live tracker snapshot without inventing provenance.

        ``VisualTrack.center_norm[1]`` and its velocity are image-down
        positive.  The gate graph also exposes an elevation-up view, but this
        adapter intentionally consumes the raw image convention so the sign
        cannot be silently inverted at the runner boundary.

        Current-gate authority additionally requires the race-labelled
        ``CURRENT`` role.  Callers adapting a graph-vetted next-gate candidate
        must explicitly set ``require_current_authority=False``; visibility,
        exact live receiver provenance, and coherent latest history remain
        mandatory.
        """

        if type(track) is not VisualTrack:
            raise VisualServoRefusal(
                "servo adaptation requires an exact VisualTrack"
            )
        if type(require_current_authority) is not bool:
            raise VisualServoRefusal(
                "require_current_authority must be an exact bool"
            )
        if expected_gate_index is not None and (
            type(expected_gate_index) is not int or expected_gate_index < 0
        ):
            raise VisualServoRefusal(
                "expected gate index must be a nonnegative exact int or None"
            )
        if expected_gate_index is not None and not require_current_authority:
            raise VisualServoRefusal(
                "an expected gate index requires current-gate authority"
            )

        history = track.history
        if type(history) is not tuple or not history:
            raise VisualServoRefusal("visual track lacks exact sample history")
        if any(type(item) is not VisualTrackSample for item in history):
            raise VisualServoRefusal(
                "visual track history must contain exact tracker samples"
            )
        sample = history[-1]
        token = track.latest_token
        if type(token) is not CameraFrameToken:
            raise VisualServoRefusal("visual track latest token is invalid")
        if type(track.first_token) is not CameraFrameToken:
            raise VisualServoRefusal("visual track first token is invalid")
        if history[0].token != track.first_token:
            raise VisualServoRefusal(
                "visual track first token disagrees with sample history"
            )
        if sample.token != token:
            raise VisualServoRefusal(
                "visual track latest token disagrees with sample history"
            )
        if (
            type(token.stream_id) is not str
            or not token.stream_id
            or type(token.publication_sequence) is not int
            or token.publication_sequence <= 0
        ):
            raise VisualServoRefusal(
                "live servo authority requires receiver publication provenance"
            )
        if sample.provenance_basis is not FrameProvenanceBasis.RECEIVER_TIMING_V1:
            raise VisualServoRefusal(
                "live servo authority requires receiver timing provenance"
            )
        observation_ns = sample.observation_monotonic_ns
        publication_ns = sample.publication_monotonic_ns
        if (
            type(observation_ns) is not int
            or observation_ns < 0
            or type(publication_ns) is not int
            or publication_ns < observation_ns
        ):
            raise VisualServoRefusal(
                "live servo authority requires coherent receiver times"
            )
        center = track.center_norm
        velocity = track.center_velocity_norm_s
        if type(center) is not tuple or len(center) != 2:
            raise VisualServoRefusal("visual track center convention is invalid")
        if type(velocity) is not tuple or len(velocity) != 2:
            raise VisualServoRefusal("visual track velocity convention is invalid")
        apparent_scale = track.apparent_scale
        if (
            type(apparent_scale) not in {int, float}
            or not math.isfinite(float(apparent_scale))
            or float(apparent_scale) <= 0.0
        ):
            raise VisualServoRefusal("visual track apparent scale is invalid")
        if type(track.clipping) is not FrameEdge:
            raise VisualServoRefusal("visual track clipping state is invalid")
        if type(track.center_censored) is not bool:
            raise VisualServoRefusal("visual track censoring state is invalid")
        if type(track.visible) is not bool or not track.visible:
            raise VisualServoRefusal(
                "servo adaptation requires a currently visible visual track"
            )
        if type(track.ambiguous) is not bool:
            raise VisualServoRefusal("visual track ambiguity state is invalid")
        if type(track.role) is not VisualTrackRole:
            raise VisualServoRefusal("visual track role is invalid")
        if track.ambiguous != (track.role is VisualTrackRole.AMBIGUOUS):
            raise VisualServoRefusal(
                "visual track ambiguity disagrees with its lifecycle role"
            )
        if (
            sample.center_norm != track.center_norm
            or sample.bbox_norm != track.bbox_norm
            or sample.apparent_scale != track.apparent_scale
            or sample.association_confidence != track.association_confidence
            or sample.clipping != track.clipping
            or sample.center_censored != track.center_censored
        ):
            raise VisualServoRefusal(
                "visual track latest fields disagree with sample history"
            )
        for label, confidence in (
            ("latest sample confidence", sample.confidence),
            ("smoothed track confidence", track.confidence),
        ):
            if (
                type(confidence) not in {int, float}
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise VisualServoRefusal(f"{label} is outside [0, 1]")
        if require_current_authority:
            if track.role is not VisualTrackRole.CURRENT:
                raise VisualServoRefusal(
                    "current servo authority requires a CURRENT visual track"
                )
            if (
                type(track.authoritative_gate_index) is not int
                or track.authoritative_gate_index < 0
            ):
                raise VisualServoRefusal(
                    "current servo authority requires an authoritative gate index"
                )
            if (
                expected_gate_index is not None
                and track.authoritative_gate_index != expected_gate_index
            ):
                raise VisualServoRefusal(
                    "visual track authoritative gate index does not match"
                )

        horizontal_censored = bool(
            track.clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)
        )
        vertical_censored = bool(
            track.clipping & (FrameEdge.TOP | FrameEdge.BOTTOM)
        )
        # Exact tracker snapshots currently mark every edge fragment as
        # center-censored.  If a future detector censors a center without
        # identifying an edge, suppress both axes rather than guessing.
        if track.center_censored and not (
            horizontal_censored or vertical_censored
        ):
            horizontal_censored = True
            vertical_censored = True

        return cls(
            track_id=track.track_id,
            frame_token=ServoFrameToken(
                stream_id=token.stream_id,
                generation=token.generation,
                frame_id=token.frame_id,
                publication_sequence=token.publication_sequence,
            ),
            received_monotonic_s=observation_ns / 1_000_000_000.0,
            normalized_x=float(center[0]),
            normalized_y_down=float(center[1]),
            normalized_x_rate_s=float(velocity[0]),
            normalized_y_rate_down_s=float(velocity[1]),
            log_scale=math.log(float(apparent_scale)),
            log_scale_rate_s=float(track.log_scale_rate_s),
            confidence=float(track.confidence),
            association_confidence=float(track.association_confidence),
            consecutive_frames=track.consecutive_frame_count,
            clipped=track.clipping != FrameEdge.NONE,
            center_censored=track.center_censored,
            horizontal_censored=horizontal_censored,
            vertical_censored=vertical_censored,
            ambiguous=track.ambiguous,
        )


@dataclass(frozen=True)
class VisualServoOutput:
    """One bounded image-servo proposal for the runner attitude loop."""

    target_roll_rad: float
    target_pitch_rad: float
    yaw_rate_rad_s: float
    thrust: float
    corridor_frames: int
    advance_enabled: bool
    next_gate_blend: float
    horizontal_error: float
    vertical_error_image_down: float
    effective_horizontal_error: float
    effective_vertical_error_image_down: float
    effective_horizontal_rate_s: float
    effective_vertical_rate_down_s: float
    next_horizontal_error: Optional[float]
    next_vertical_error_image_down: Optional[float]
    horizontal_abs_error_delta: Optional[float]
    vertical_abs_error_delta: Optional[float]
    brake_reason: Optional[str]
    yaw_envelope_limited: bool
    reviewed_next_track_id: Optional[str] = None
    passage_preview_retired: bool = False
    passage_preview_retirement_violations: Tuple[
        PassageSafetyViolationDetail,
        ...,
    ] = ()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def visual_bearing_yaw_rate(
    horizontal: float,
    horizontal_rate: float,
    tuning: VisualServoTuning,
) -> float:
    """Allocate bounded yaw continuously from image bearing and rate."""

    if type(tuning) is not VisualServoTuning:
        raise TypeError("visual yaw allocation requires exact tuning")
    if not all(
        type(value) in {int, float} and math.isfinite(float(value))
        for value in (horizontal, horizontal_rate)
    ):
        raise VisualServoRefusal("visual yaw inputs must be finite")
    horizontal = float(horizontal)
    horizontal_rate = float(horizontal_rate)
    maneuver_authority = _clamp(
        abs(horizontal) / tuning.horizontal_corridor,
        0.0,
        1.0,
    )
    corridor_rate = (
        -MAX_VISUAL_YAW_RATE_RAD_S
        * _clamp(
            horizontal / tuning.horizontal_corridor,
            -1.0,
            1.0,
        )
    )
    bearing_rate = (
        (1.0 - maneuver_authority)
        * (-tuning.yaw_error_gain * horizontal)
        + maneuver_authority * corridor_rate
    )
    return _clamp(
        bearing_rate - tuning.yaw_rate_gain * horizontal_rate,
        -MAX_VISUAL_YAW_RATE_RAD_S,
        MAX_VISUAL_YAW_RATE_RAD_S,
    )


def _passage_violation_detail(
    violation: PassageSafetyViolation,
    *,
    observed: float,
    limit: float,
    excess: Optional[float] = None,
) -> PassageSafetyViolationDetail:
    observed_value = float(observed)
    limit_value = float(limit)
    excess_value = (
        observed_value - limit_value
        if excess is None
        else float(excess)
    )
    return PassageSafetyViolationDetail(
        violation=violation,
        observed=observed_value,
        limit=limit_value,
        excess=excess_value,
    )


class ImageVisualServo:
    """Stateful fresh-frame visual servo with align-before-advance gating."""

    def __init__(self, tuning: Optional[VisualServoTuning] = None) -> None:
        self.tuning = tuning or VisualServoTuning()
        self.reset_segment()

    def reset_segment(self) -> None:
        self._last_token: Optional[ServoFrameToken] = None
        self._segment_track_id: Optional[str] = None
        self._latched_next_blend_track_id: Optional[str] = None
        self._advance_passage_preview_retired = False
        self._last_abs_error: Optional[
            Tuple[Optional[float], Optional[float]]
        ] = None
        self._last_vertical_observable_thrust: Optional[float] = None
        self._corridor_frames = 0

    @property
    def corridor_frames(self) -> int:
        return self._corridor_frames

    @property
    def latched_next_track_id(self) -> Optional[str]:
        return self._latched_next_blend_track_id

    def retire_advance_passage_preview(self) -> None:
        """Permanently remove optional next-preview authority for this segment."""

        if self._latched_next_blend_track_id is None:
            raise VisualServoRefusal(
                "cannot retire passage preview without an established latch"
            )
        self._advance_passage_preview_retired = True

    def step(
        self,
        current: VisualTarget,
        *,
        now_monotonic_s: float,
        segment_elapsed_s: float,
        segment_yaw_excursion_rad: float,
        next_target: Optional[VisualTarget] = None,
        requested_next_blend: float = 0.0,
        allow_advance: bool = True,
        allow_passage_safe_next_blend: bool = False,
    ) -> VisualServoOutput:
        """Produce a bounded proposal from one distinct, authoritative frame.

        A caller may pre-orient toward a stable next-gate track.  Initial
        authority still requires the tight advance corridor; after that exact
        next identity is latched, a caller may continue within the immutable
        passage corridor while independently requesting advance.  The current
        gate always retains passage authority.
        """

        scalars = (
            now_monotonic_s,
            segment_elapsed_s,
            segment_yaw_excursion_rad,
            requested_next_blend,
        )
        passage_preview_retirement_violations: Tuple[
            PassageSafetyViolationDetail,
            ...,
        ] = ()
        reviewed_next_track_id: Optional[str] = None
        next_preview_withheld_for_current_envelope = False
        latched_next_track_this_frame = False
        if not all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in scalars
        ):
            raise VisualServoRefusal("visual-servo step inputs must be finite")
        if float(segment_elapsed_s) < 0.0:
            raise VisualServoRefusal("segment elapsed time cannot be negative")
        if float(segment_elapsed_s) > MAX_VISUAL_SEGMENT_DURATION_S:
            raise VisualServoRefusal("visual-servo segment duration exhausted")
        if abs(float(segment_yaw_excursion_rad)) > (
            MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD
        ):
            raise VisualServoRefusal("visual-servo segment yaw envelope exceeded")
        if not 0.0 <= float(requested_next_blend) <= MAX_NEXT_GATE_BLEND:
            raise VisualServoRefusal("next-gate blend is outside bounds")
        if type(allow_advance) is not bool:
            raise VisualServoRefusal("allow_advance must be an exact bool")
        if type(allow_passage_safe_next_blend) is not bool:
            raise VisualServoRefusal(
                "allow_passage_safe_next_blend must be an exact bool"
            )
        if (
            allow_passage_safe_next_blend
            and allow_advance
            and self._latched_next_blend_track_id is None
        ):
            raise VisualServoRefusal(
                "advance passage preview requires an established next-track "
                "latch"
            )
        if allow_passage_safe_next_blend and not (
            self.tuning.horizontal_corridor
            < PREPASS_CURRENT_MAX_ABS_X_NORM
            and self.tuning.vertical_corridor
            < PREPASS_CURRENT_MAX_ABS_Y_NORM
        ):
            raise VisualServoRefusal(
                "configured start corridor must stay inside passage bounds"
            )
        age_s = float(now_monotonic_s) - float(current.received_monotonic_s)
        if age_s < -1e-6 or age_s > MAX_VISUAL_OBSERVATION_AGE_S:
            raise VisualServoRefusal("current visual target is stale or future-dated")
        if (
            current.ambiguous
            or current.confidence < 0.10
            or current.association_confidence < 0.10
        ):
            raise VisualServoRefusal("current visual target lacks authority")
        if self._segment_track_id is None:
            self._segment_track_id = current.track_id
        elif current.track_id != self._segment_track_id:
            raise VisualServoRefusal(
                "current track identity changed without segment promotion"
            )
        if self._last_token is not None:
            if (
                current.frame_token.stream_id != self._last_token.stream_id
                or current.frame_token.generation != self._last_token.generation
                or current.frame_token.publication_sequence
                <= self._last_token.publication_sequence
            ):
                raise VisualServoRefusal(
                    "visual-servo exact publication token did not advance"
                )

        raw_horizontal = float(current.normalized_x)
        raw_vertical = float(current.normalized_y_down)
        raw_horizontal_rate = float(current.normalized_x_rate_s)
        raw_vertical_rate = float(current.normalized_y_rate_down_s)
        horizontal_censored = current.horizontal_geometry_censored
        vertical_censored = current.vertical_geometry_censored
        horizontal = 0.0 if horizontal_censored else raw_horizontal
        vertical = 0.0 if vertical_censored else raw_vertical
        horizontal_rate = (
            0.0 if horizontal_censored else raw_horizontal_rate
        )
        vertical_rate = 0.0 if vertical_censored else raw_vertical_rate
        current_inside_position = (
            not horizontal_censored
            and not vertical_censored
            and abs(raw_horizontal) <= self.tuning.horizontal_corridor
            and abs(raw_vertical) <= self.tuning.vertical_corridor
        )
        stable_rates = (
            not horizontal_censored
            and not vertical_censored
            and abs(raw_horizontal_rate) <= self.tuning.stable_rate_norm_s
            and abs(raw_vertical_rate) <= self.tuning.stable_rate_norm_s
            and abs(float(current.log_scale_rate_s))
            <= self.tuning.stable_scale_rate_s
        )
        previous_abs_error = self._last_abs_error
        horizontal_delta = (
            None
            if (
                horizontal_censored
                or previous_abs_error is None
                or previous_abs_error[0] is None
            )
            else abs(raw_horizontal) - previous_abs_error[0]
        )
        vertical_delta = (
            None
            if (
                vertical_censored
                or previous_abs_error is None
                or previous_abs_error[1] is None
            )
            else abs(raw_vertical) - previous_abs_error[1]
        )
        worsening = bool(
            (horizontal_delta is not None and horizontal_delta > 0.015)
            or (vertical_delta is not None and vertical_delta > 0.015)
        )
        inside_corridor = (
            current_inside_position and stable_rates and not worsening
        )
        passage_continuation = bool(
            allow_passage_safe_next_blend
            and self._latched_next_blend_track_id is not None
            and not self._advance_passage_preview_retired
        )
        passage_violations = []
        if passage_continuation:
            if horizontal_censored:
                passage_violations.append(
                    _passage_violation_detail(
                        PassageSafetyViolation.CURRENT_GEOMETRY_CENSORED,
                        observed=1.0,
                        limit=0.0,
                    )
                )
            if abs(raw_horizontal) > PREPASS_CURRENT_MAX_ABS_X_NORM:
                passage_violations.append(
                    _passage_violation_detail(
                        PassageSafetyViolation.CURRENT_HORIZONTAL_POSITION,
                        observed=abs(raw_horizontal),
                        limit=PREPASS_CURRENT_MAX_ABS_X_NORM,
                    )
                )
            if abs(raw_vertical) > PREPASS_CURRENT_MAX_ABS_Y_NORM:
                passage_violations.append(
                    _passage_violation_detail(
                        PassageSafetyViolation.CURRENT_VERTICAL_POSITION,
                        observed=abs(raw_vertical),
                        limit=PREPASS_CURRENT_MAX_ABS_Y_NORM,
                    )
                )
            if (
                abs(raw_horizontal_rate)
                > PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S
            ):
                passage_violations.append(
                    _passage_violation_detail(
                        PassageSafetyViolation.CURRENT_HORIZONTAL_RATE,
                        observed=abs(raw_horizontal_rate),
                        limit=(
                            PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S
                        ),
                    )
                )
            if (
                abs(raw_vertical_rate)
                > PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S
            ):
                passage_violations.append(
                    _passage_violation_detail(
                        PassageSafetyViolation.CURRENT_VERTICAL_RATE,
                        observed=abs(raw_vertical_rate),
                        limit=(
                            PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S
                        ),
                    )
                )
            if not (
                PREPASS_CURRENT_MIN_LOG_SCALE_RATE_S
                <= float(current.log_scale_rate_s)
                <= PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S
            ):
                passage_violations.append(
                    _passage_violation_detail(
                        PassageSafetyViolation.CURRENT_LOG_SCALE_RATE,
                        observed=float(current.log_scale_rate_s),
                        limit=(
                            PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S
                            if float(current.log_scale_rate_s)
                            > PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S
                            else PREPASS_CURRENT_MIN_LOG_SCALE_RATE_S
                        ),
                        excess=(
                            float(current.log_scale_rate_s)
                            - PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S
                            if float(current.log_scale_rate_s)
                            > PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S
                            else PREPASS_CURRENT_MIN_LOG_SCALE_RATE_S
                            - float(current.log_scale_rate_s)
                        ),
                    )
                )
            if abs(
                raw_horizontal
                + raw_horizontal_rate
                * PREPASS_CURRENT_PROJECTION_HORIZON_S
            ) > PREPASS_CURRENT_MAX_ABS_X_NORM:
                passage_violations.append(
                    _passage_violation_detail(
                        PassageSafetyViolation.CURRENT_PROJECTED_HORIZONTAL,
                        observed=abs(
                            raw_horizontal
                            + raw_horizontal_rate
                            * PREPASS_CURRENT_PROJECTION_HORIZON_S
                        ),
                        limit=PREPASS_CURRENT_MAX_ABS_X_NORM,
                    )
                )
            projected_vertical = (
                raw_vertical
                + raw_vertical_rate
                * PREPASS_CURRENT_PROJECTION_HORIZON_S
            )
            if (
                abs(projected_vertical)
                > PREPASS_CURRENT_MAX_ABS_Y_NORM
            ):
                passage_violations.append(
                    _passage_violation_detail(
                        PassageSafetyViolation.CURRENT_PROJECTED_VERTICAL,
                        observed=projected_vertical,
                        limit=math.copysign(
                            PREPASS_CURRENT_MAX_ABS_Y_NORM,
                            projected_vertical,
                        ),
                        excess=(
                            abs(projected_vertical)
                            - PREPASS_CURRENT_MAX_ABS_Y_NORM
                        ),
                    )
                )
            if (
                math.exp(float(current.log_scale))
                > PREPASS_CURRENT_MAX_APPARENT_SCALE
            ):
                passage_violations.append(
                    _passage_violation_detail(
                        PassageSafetyViolation.CURRENT_APPARENT_SCALE,
                        observed=math.exp(float(current.log_scale)),
                        limit=PREPASS_CURRENT_MAX_APPARENT_SCALE,
                    )
                )
        lateral_passage_violations = tuple(
            detail
            for detail in passage_violations
            if detail.violation
            in {
                PassageSafetyViolation.CURRENT_GEOMETRY_CENSORED,
                PassageSafetyViolation.CURRENT_HORIZONTAL_POSITION,
                PassageSafetyViolation.CURRENT_HORIZONTAL_RATE,
                PassageSafetyViolation.CURRENT_PROJECTED_HORIZONTAL,
            }
        )
        vertical_preview_degraded = bool(
            vertical_censored
            or any(
                detail.violation
                in {
                    PassageSafetyViolation.CURRENT_VERTICAL_POSITION,
                    PassageSafetyViolation.CURRENT_VERTICAL_RATE,
                    PassageSafetyViolation.CURRENT_PROJECTED_VERTICAL,
                }
                for detail in passage_violations
            )
        )
        passage_safe_current = bool(
            passage_continuation and not lateral_passage_violations
        )
        if passage_violations:
            if not allow_advance:
                # These predicates bound optional look-ahead authority, not
                # the independently validated current-aperture controller.
                # Approach therefore withholds preview for this publication
                # and lets the ordinary current-only safety logic below decide
                # whether a command remains available.
                next_preview_withheld_for_current_envelope = True
            elif lateral_passage_violations:
                # Only loss of the current aperture's observable horizontal
                # corridor permanently retires lateral successor authority.
                # Vertical and scale degradation instead remove forward and
                # vertical preview authority while bounded yaw/bank continue.
                self._advance_passage_preview_retired = True
                passage_preview_retirement_violations = (
                    lateral_passage_violations
                )
        passage_safe_start = bool(
            allow_passage_safe_next_blend
            and not self._advance_passage_preview_retired
            and inside_corridor
            and not current.clipped
            and not current.center_censored
            and abs(
                raw_horizontal
                + raw_horizontal_rate
                * PREPASS_CURRENT_PROJECTION_HORIZON_S
            )
            <= PREPASS_CURRENT_MAX_ABS_X_NORM
            and abs(
                raw_vertical
                + raw_vertical_rate
                * PREPASS_CURRENT_PROJECTION_HORIZON_S
            )
            <= PREPASS_CURRENT_MAX_ABS_Y_NORM
            and math.exp(float(current.log_scale))
            <= PREPASS_CURRENT_MAX_APPARENT_SCALE
        )

        blend = 0.0
        next_horizontal: Optional[float] = None
        next_vertical: Optional[float] = None
        next_edge_risk = False
        next_ambiguity_risk = False
        if next_target is not None:
            next_age_s = (
                float(now_monotonic_s)
                - float(next_target.received_monotonic_s)
            )
            same_frame = next_target.frame_token == current.frame_token
            next_horizontal_censored = (
                next_target.horizontal_geometry_censored
            )
            next_vertical_censored = next_target.vertical_geometry_censored
            next_edge_risk = bool(
                next_target.clipped
                or next_target.center_censored
                or next_horizontal_censored
                or next_vertical_censored
            )
            next_ambiguity_risk = next_target.ambiguous
            if (
                allow_passage_safe_next_blend
                and self._latched_next_blend_track_id is not None
                and next_target.track_id
                != self._latched_next_blend_track_id
            ):
                raise VisualServoRefusal(
                    "latched next-gate identity changed without promotion"
                )
            next_center_rate_limit = (
                PREPASS_NEXT_MAX_ABS_CENTER_RATE_NORM_S
                if passage_continuation
                else self.tuning.stable_rate_norm_s
            )
            next_scale_rate_limit = (
                PREPASS_NEXT_MAX_ABS_LOG_SCALE_RATE_S
                if passage_continuation
                else self.tuning.stable_scale_rate_s
            )
            next_usable = (
                not next_ambiguity_risk
                and next_target.confidence >= 0.10
                and next_target.association_confidence >= 0.10
                and next_target.consecutive_frames >= 3
                and (
                    not allow_passage_safe_next_blend
                    or not next_edge_risk
                )
                and not (
                    next_horizontal_censored and next_vertical_censored
                )
                and (
                    next_horizontal_censored
                    or abs(float(next_target.normalized_x_rate_s))
                    <= next_center_rate_limit
                )
                and (
                    next_vertical_censored
                    or abs(float(next_target.normalized_y_rate_down_s))
                    <= next_center_rate_limit
                )
                and (
                    next_edge_risk
                    or abs(float(next_target.log_scale_rate_s))
                    <= next_scale_rate_limit
                )
                and -1e-6 <= next_age_s <= MAX_VISUAL_OBSERVATION_AGE_S
                and same_frame
                and next_target.track_id != current.track_id
            )
            identity_latch_ready = bool(
                next_usable
                and not next_preview_withheld_for_current_envelope
                and passage_safe_start
                and self._corridor_frames + 1
                >= self.tuning.required_corridor_frames
            )
            if (
                identity_latch_ready
                and self._latched_next_blend_track_id is None
            ):
                self._latched_next_blend_track_id = next_target.track_id
                latched_next_track_this_frame = True
            if identity_latch_ready:
                reviewed_next_track_id = next_target.track_id
            if (
                float(requested_next_blend) > 0.0
                and next_usable
                and not next_preview_withheld_for_current_envelope
                and (
                    passage_safe_current
                    or (
                        inside_corridor
                        and self._corridor_frames
                        >= self.tuning.required_corridor_frames
                        and (
                            not allow_passage_safe_next_blend
                            or passage_safe_start
                        )
                    )
                )
            ):
                blend = float(requested_next_blend)
                if not next_horizontal_censored:
                    next_horizontal = float(next_target.normalized_x)
                    horizontal = (
                        (1.0 - blend) * horizontal
                        + blend * next_horizontal
                    )
                    horizontal_rate = (
                        (1.0 - blend) * horizontal_rate
                        + blend * float(next_target.normalized_x_rate_s)
                    )
                # Preserve more vertical authority for the current aperture;
                # early heading blend is useful, but a clipped next gate must
                # not pull the vehicle out of the current vertical corridor.
                if (
                    not next_vertical_censored
                    and not (
                        passage_continuation
                        and vertical_preview_degraded
                    )
                ):
                    next_vertical = float(next_target.normalized_y_down)
                    vertical = (
                        (1.0 - 0.5 * blend) * vertical
                        + 0.5 * blend * next_vertical
                    )
                    vertical_rate = (
                        (1.0 - 0.5 * blend) * vertical_rate
                        + 0.5
                        * blend
                        * float(next_target.normalized_y_rate_down_s)
                    )
                if passage_continuation:
                    horizontal_correction_violation = None
                    if (
                        abs(raw_horizontal)
                        > self.tuning.horizontal_corridor
                        and raw_horizontal * horizontal < 0.0
                    ):
                        horizontal_correction_violation = (
                            _passage_violation_detail(
                                PassageSafetyViolation
                                .CURRENT_HORIZONTAL_CORRECTION_REVERSAL,
                                observed=-(raw_horizontal * horizontal),
                                limit=0.0,
                            )
                        )
                    if (
                        abs(raw_vertical)
                        > self.tuning.vertical_corridor
                        and raw_vertical * vertical < 0.0
                    ):
                        next_vertical = None
                        vertical = (
                            0.0 if vertical_censored else raw_vertical
                        )
                        vertical_rate = (
                            0.0 if vertical_censored else raw_vertical_rate
                        )
                    if horizontal_correction_violation is not None:
                        # A next preview that would reverse current-aperture
                        # correction loses all command authority for this
                        # publication and retires the lateral preview.  A
                        # vertical reversal above merely drops the optional
                        # vertical preview component.
                        blend = 0.0
                        next_horizontal = None
                        next_vertical = None
                        horizontal = (
                            0.0 if horizontal_censored else raw_horizontal
                        )
                        vertical = 0.0 if vertical_censored else raw_vertical
                        horizontal_rate = (
                            0.0
                            if horizontal_censored
                            else raw_horizontal_rate
                        )
                        vertical_rate = (
                            0.0 if vertical_censored else raw_vertical_rate
                        )
                        if not allow_advance:
                            reviewed_next_track_id = None
                            if latched_next_track_this_frame:
                                self._latched_next_blend_track_id = None
                        else:
                            self._advance_passage_preview_retired = True
                            passage_preview_retirement_violations = (
                                horizontal_correction_violation,
                            )

        heading_horizontal = horizontal
        heading_horizontal_rate = horizontal_rate
        bank_horizontal = horizontal
        bank_horizontal_rate = horizontal_rate
        if blend > 0.0 and next_horizontal is not None:
            # Passage admission has already established one exact, stable
            # successor identity.  Yaw consumes the worse of measured and
            # projected current-aperture margin once.  Multiplying those two
            # margins made the controller unwind a physically effective turn
            # while both observations still remained inside the corridor.
            # Current geometry attenuates successor yaw but does not
            # algebraically cancel it with the opposite current-gate bearing.
            # The independently conservative bank product returns
            # continuously to current-only as passage scale approaches the
            # near plane.
            projected_current_horizontal = (
                raw_horizontal
                + raw_horizontal_rate
                * PREPASS_CURRENT_PROJECTION_HORIZON_S
            )
            current_position_authority = _clamp(
                1.0
                - abs(raw_horizontal)
                / self.tuning.horizontal_corridor,
                0.0,
                1.0,
            )
            current_projection_authority = _clamp(
                1.0
                - abs(projected_current_horizontal)
                / self.tuning.horizontal_corridor,
                0.0,
                1.0,
            )
            current_barrier_authority = (
                current_position_authority
                * current_projection_authority
            )
            successor_heading_authority = min(
                current_position_authority,
                current_projection_authority,
            )
            assert next_target is not None
            heading_horizontal = (
                successor_heading_authority * next_horizontal
            )
            heading_horizontal_rate = (
                successor_heading_authority
                * float(next_target.normalized_x_rate_s)
            )
            passage_scale_progress = _clamp(
                blend / MAX_NEXT_GATE_BLEND,
                0.0,
                1.0,
            )
            bank_authority = (
                current_barrier_authority
                * (1.0 - passage_scale_progress)
            )
            bank_horizontal = (
                (1.0 - bank_authority) * raw_horizontal
                + bank_authority * next_horizontal
            )
            bank_horizontal_rate = (
                (1.0 - bank_authority) * raw_horizontal_rate
                + bank_authority
                * float(next_target.normalized_x_rate_s)
            )

        self._last_abs_error = (
            None if horizontal_censored else abs(raw_horizontal),
            None if vertical_censored else abs(raw_vertical),
        )
        self._last_token = current.frame_token

        if inside_corridor:
            self._corridor_frames += 1
        elif not (
            allow_advance
            and self._corridor_frames
            >= self.tuning.required_corridor_frames
        ):
            # Passage permission is an explicit course-FSM decision.  Once
            # granted, normal perspective-driven scale/rate changes must not
            # erase the clean approach dwell that earned it.  Current
            # geometry still continuously controls closure authority below.
            self._corridor_frames = 0
        current_edge_risk = bool(
            abs(raw_horizontal) >= self.tuning.edge_brake_x
            or abs(raw_vertical) >= self.tuning.edge_brake_y
            or current.clipped
            or current.center_censored
            or horizontal_censored
            or vertical_censored
        )
        optional_advance_preview = bool(
            allow_advance
            and allow_passage_safe_next_blend
            and self._latched_next_blend_track_id is not None
        )
        next_risk_has_command_authority = bool(
            float(requested_next_blend) > 0.0
            and (not optional_advance_preview or blend > 0.0)
        )
        effective_next_edge_risk = bool(
            next_edge_risk and next_risk_has_command_authority
        )
        effective_next_ambiguity_risk = bool(
            next_ambiguity_risk and next_risk_has_command_authority
        )
        edge_risk = current_edge_risk or effective_next_edge_risk
        scale_brake = (
            float(current.log_scale_rate_s) >= self.tuning.brake_scale_rate_s
        )
        scale_retreat = (
            float(current.log_scale_rate_s)
            <= VISUAL_RAPID_RETREAT_LOG_SCALE_RATE_S
        )
        close_scale_brake = (
            float(current.log_scale) >= VISUAL_CLOSE_SCALE_BRAKE_LOG
        )

        confidence_authority = _clamp(
            (
                min(
                    float(current.confidence),
                    float(current.association_confidence),
                )
                - 0.10
            )
            / 0.90,
            0.0,
            1.0,
        )
        projected_horizontal = (
            horizontal
            + horizontal_rate * PREPASS_CURRENT_PROJECTION_HORIZON_S
        )
        projected_heading_horizontal = (
            heading_horizontal
            + heading_horizontal_rate
            * PREPASS_CURRENT_PROJECTION_HORIZON_S
        )
        projected_vertical = (
            vertical
            + vertical_rate * PREPASS_CURRENT_PROJECTION_HORIZON_S
        )
        center_authority = _clamp(
            1.0
            - max(
                abs(projected_horizontal)
                / self.tuning.horizontal_corridor,
                abs(projected_vertical)
                / self.tuning.vertical_corridor,
            ),
            0.0,
            1.0,
        )
        projected_bank_horizontal = (
            bank_horizontal
            + bank_horizontal_rate
            * PREPASS_CURRENT_PROJECTION_HORIZON_S
        )
        maneuver_forward_authority = _clamp(
            1.0
            - max(
                abs(projected_heading_horizontal),
                abs(projected_bank_horizontal),
            )
            / self.tuning.edge_brake_x,
            0.0,
            1.0,
        )
        center_authority = min(
            center_authority,
            maneuver_forward_authority,
        )
        expansion_authority = _clamp(
            (
                self.tuning.brake_scale_rate_s
                - float(current.log_scale_rate_s)
            )
            / (
                self.tuning.brake_scale_rate_s
                - self.tuning.stable_scale_rate_s
            ),
            0.0,
            1.0,
        )
        close_apparent_scale = math.exp(VISUAL_CLOSE_SCALE_BRAKE_LOG)
        apparent_scale = math.exp(
            min(
                float(current.log_scale),
                VISUAL_CLOSE_SCALE_BRAKE_LOG,
            )
        )
        proximity_authority = _clamp(
            (
                close_apparent_scale - apparent_scale
            )
            / (
                close_apparent_scale
                - PREPASS_CURRENT_MAX_APPARENT_SCALE
            ),
            0.0,
            1.0,
        )
        forward_authority = 0.0
        if (
            allow_advance
            and self._corridor_frames >= self.tuning.required_corridor_frames
            and not edge_risk
            and not effective_next_ambiguity_risk
            and not scale_brake
            and not scale_retreat
            and not close_scale_brake
            and not worsening
        ):
            forward_authority = min(
                confidence_authority,
                center_authority,
                expansion_authority,
                proximity_authority,
            )
        advance_enabled = forward_authority > 0.0

        # Taper the adjacent preview's reported bearing/elevation blend as the
        # current aperture expands, while the separately corridor-constrained
        # heading above retains coordinated yaw/bank authority.  This starts
        # lateral interception before the physical handoff without allowing
        # successor steering to spend the current aperture's projected
        # horizontal margin.
        if blend > 0.0:
            assert next_target is not None
            preview_steering_authority = (
                expansion_authority * proximity_authority
            )
            blend *= preview_steering_authority
            if next_horizontal is not None:
                horizontal = (
                    (1.0 - blend) * raw_horizontal
                    + blend * next_horizontal
                )
                horizontal_rate = (
                    (1.0 - blend) * raw_horizontal_rate
                    + blend * float(next_target.normalized_x_rate_s)
                )
            if next_vertical is not None:
                vertical = (
                    (1.0 - 0.5 * blend) * raw_vertical
                    + 0.5 * blend * next_vertical
                )
                vertical_rate = (
                    (1.0 - 0.5 * blend) * raw_vertical_rate
                    + 0.5
                    * blend
                    * float(next_target.normalized_y_rate_down_s)
                )

        brake_reason: Optional[str] = None
        if current_edge_risk:
            brake_reason = "target_edge_or_clipping"
        elif effective_next_ambiguity_risk:
            brake_reason = "next_target_ambiguous"
        elif effective_next_edge_risk:
            brake_reason = "next_target_edge_or_clipping"
        elif scale_brake:
            brake_reason = "scale_rate"
        elif scale_retreat:
            brake_reason = "rapid_scale_retreat"
        elif close_scale_brake:
            brake_reason = "close_scale"
        elif worsening:
            brake_reason = "alignment_error_increasing"
        elif not advance_enabled:
            brake_reason = "aligning"

        unconstrained_yaw_rate = visual_bearing_yaw_rate(
            heading_horizontal,
            heading_horizontal_rate,
            self.tuning,
        )
        yaw_rate = _clamp(
            unconstrained_yaw_rate,
            -MAX_VISUAL_YAW_RATE_RAD_S,
            MAX_VISUAL_YAW_RATE_RAD_S,
        )
        steering_load = _clamp(
            abs(unconstrained_yaw_rate) / MAX_VISUAL_YAW_RATE_RAD_S,
            0.0,
            1.0,
        )
        yaw_envelope_limited = bool(
            abs(float(segment_yaw_excursion_rad))
            >= VISUAL_SEGMENT_YAW_SOFT_STOP_RAD
            and float(segment_yaw_excursion_rad) * yaw_rate > 0.0
        )
        if yaw_envelope_limited:
            yaw_rate = 0.0
            steering_load = 0.0
            forward_authority = 0.0
            advance_enabled = False
            brake_reason = "segment_yaw_outward_soft_stop"

        base_target_roll = _clamp(
            self.tuning.roll_error_gain * bank_horizontal
            + self.tuning.roll_rate_gain * bank_horizontal_rate,
            -MAX_VISUAL_TARGET_ROLL_RAD,
            MAX_VISUAL_TARGET_ROLL_RAD,
        )
        target_roll = base_target_roll
        # A large bearing demand that consumes yaw authority also needs
        # lateral translation.  Continuously transfer that load into the
        # already measured bank envelope only when the current-aperture bank
        # and yaw requests are coordinated.  An opposite or zero bank retains
        # the passage barrier instead of importing successor bank through it.
        if base_target_roll * unconstrained_yaw_rate < 0.0:
            bank_steering_load = steering_load * _clamp(
                abs(projected_bank_horizontal)
                / self.tuning.horizontal_corridor,
                0.0,
                1.0,
            )
            coordinated_roll_floor = (
                bank_steering_load * MAX_VISUAL_TARGET_ROLL_RAD
            )
            if abs(base_target_roll) < coordinated_roll_floor:
                target_roll = math.copysign(
                    coordinated_roll_floor,
                    base_target_roll,
                )
        outward_bearing_authority = (
            steering_load
            * _clamp(
                (
                    heading_horizontal
                    * heading_horizontal_rate
                )
                / (
                    self.tuning.horizontal_corridor
                    * self.tuning.stable_rate_norm_s
                ),
                0.0,
                1.0,
            )
        )
        if outward_bearing_authority > 0.0:
            # Same-sign bank and a later counter-bank both failed to center
            # free-flight successors, while counter-bank consumed most of the
            # measured body-rate envelope.  Smoothly unload bank as saturated
            # yaw and outward image motion grow.  This retains calibrated yaw
            # and closure braking without asserting an unmeasured lateral
            # response sign.
            target_roll *= 1.0 - outward_bearing_authority
        if advance_enabled:
            pitch_basis = (
                self.tuning.brake_pitch_rad
                + forward_authority
                * (
                    self.tuning.advance_pitch_rad
                    - self.tuning.brake_pitch_rad
                )
            )
            thrust_basis = (
                self.tuning.brake_thrust
                + forward_authority
                * (
                    self.tuning.advance_thrust
                    - self.tuning.brake_thrust
                )
            )
        elif (
            edge_risk
            or effective_next_ambiguity_risk
            or scale_brake
            or scale_retreat
            or close_scale_brake
            or worsening
            or yaw_envelope_limited
        ):
            pitch_basis = self.tuning.brake_pitch_rad
            thrust_basis = self.tuning.brake_thrust
        else:
            maneuver_brake_authority = max(
                1.0 - maneuver_forward_authority,
                1.0 - expansion_authority,
                1.0 - proximity_authority,
            )
            pitch_basis = (
                maneuver_brake_authority
                * self.tuning.brake_pitch_rad
            )
            thrust_basis = (
                self.tuning.align_thrust
                + maneuver_brake_authority
                * (
                    self.tuning.brake_thrust
                    - self.tuning.align_thrust
                )
            )
        raw_target_pitch = pitch_basis
        if not advance_enabled:
            # Collective owns vertical image-space alignment.  Pitch owns
            # closure and cannot become a nose-down closure command while
            # alignment is withheld.  Allocate additional braking only when
            # observable horizontal error is moving farther outward under
            # steering load; saturated yaw alone does not manufacture
            # cross-axis pitch demand.
            raw_target_pitch = max(
                0.0,
                outward_bearing_authority
                * MAX_VISUAL_TARGET_PITCH_RAD,
                raw_target_pitch,
            )
        target_pitch = _clamp(
            raw_target_pitch,
            MIN_VISUAL_TARGET_PITCH_RAD,
            MAX_VISUAL_TARGET_PITCH_RAD,
        )
        # This retains the live-proved Gate-0 vertical pixel-space collective
        # law on top of the selected basis: a gate high in the image requests
        # more collective, while image motion toward the desired row damps it.
        # The measured flight-support alignment and brake bases preserve
        # altitude while pitch owns forward braking.  When the vertical axis
        # remains observable they provide the normal basis; a censored
        # vertical axis retains the last bounded observable-axis collective
        # below.
        measured_thrust = _clamp(
            thrust_basis
            - self.tuning.collective_error_gain * vertical
            - self.tuning.collective_rate_gain
            * vertical_rate,
            MIN_VISUAL_THRUST,
            MAX_VISUAL_THRUST,
        )
        if (
            vertical_censored
            and self._last_vertical_observable_thrust is not None
        ):
            # The missing vertical axis cannot authorize a new collective.
            # Retain the most recent bounded collective derived while that
            # axis was observable instead of treating censorship as zero
            # error and cutting to minimum thrust.
            thrust = self._last_vertical_observable_thrust
        else:
            thrust = measured_thrust
            if not vertical_censored:
                self._last_vertical_observable_thrust = thrust

        return VisualServoOutput(
            target_roll_rad=target_roll,
            target_pitch_rad=target_pitch,
            yaw_rate_rad_s=yaw_rate,
            thrust=thrust,
            corridor_frames=self._corridor_frames,
            advance_enabled=advance_enabled,
            next_gate_blend=blend,
            horizontal_error=raw_horizontal,
            vertical_error_image_down=raw_vertical,
            effective_horizontal_error=horizontal,
            effective_vertical_error_image_down=vertical,
            effective_horizontal_rate_s=horizontal_rate,
            effective_vertical_rate_down_s=vertical_rate,
            next_horizontal_error=next_horizontal,
            next_vertical_error_image_down=next_vertical,
            horizontal_abs_error_delta=horizontal_delta,
            vertical_abs_error_delta=vertical_delta,
            brake_reason=brake_reason,
            yaw_envelope_limited=yaw_envelope_limited,
            reviewed_next_track_id=reviewed_next_track_id,
            passage_preview_retired=(
                self._advance_passage_preview_retired
            ),
            passage_preview_retirement_violations=(
                passage_preview_retirement_violations
            ),
        )


__all__ = [
    "ImageVisualServo",
    "MAX_NEXT_GATE_BLEND",
    "MAX_TRANSIENT_PROJECTED_VERTICAL_EXCESS_NORM",
    "MAX_VISUAL_OBSERVATION_AGE_S",
    "MAX_VISUAL_SEGMENT_DURATION_S",
    "MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD",
    "MAX_VISUAL_YAW_RATE_RAD_S",
    "PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S",
    "PREPASS_CURRENT_MAX_ABS_X_NORM",
    "PREPASS_CURRENT_MAX_ABS_Y_NORM",
    "PREPASS_CURRENT_MAX_APPARENT_SCALE",
    "PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S",
    "PREPASS_CURRENT_MIN_LOG_SCALE_RATE_S",
    "PREPASS_CURRENT_PROJECTION_HORIZON_S",
    "PREPASS_NEXT_MAX_ABS_CENTER_RATE_NORM_S",
    "PREPASS_NEXT_MAX_ABS_LOG_SCALE_RATE_S",
    "PassageSafetyViolation",
    "PassageSafetyViolationDetail",
    "ServoFrameToken",
    "VISUAL_SEGMENT_YAW_SOFT_STOP_RAD",
    "VisualServoOutput",
    "VisualServoPassageSafetyUnavailable",
    "VisualServoRefusal",
    "VisualServoTuning",
    "VisualTarget",
    "visual_bearing_yaw_rate",
]
