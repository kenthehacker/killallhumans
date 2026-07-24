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
import math
from typing import Optional, Tuple


# These are code-owned controller authority ceilings.  The yaw-rate value is
# the exact magnitude exercised by the successful paired build-3385 sign-ID
# calibration.  The excursion is a separate per-segment course-turn envelope,
# not the retired 0.05 rad calibration-experiment limit.
MAX_VISUAL_YAW_RATE_RAD_S = 0.08
MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD = 0.65
VISUAL_SEGMENT_YAW_SOFT_STOP_RAD = 0.60
MAX_VISUAL_SEGMENT_DURATION_S = 8.0
MAX_VISUAL_TARGET_ROLL_RAD = 0.12
MIN_VISUAL_TARGET_PITCH_RAD = -0.30
MAX_VISUAL_TARGET_PITCH_RAD = 0.10
MIN_VISUAL_THRUST = 0.21
MAX_VISUAL_THRUST = 0.32
MAX_VISUAL_OBSERVATION_AGE_S = 0.10
MAX_NEXT_GATE_BLEND = 0.35
VISUAL_CLOSE_SCALE_BRAKE_LOG = -0.18
VISUAL_RAPID_RETREAT_LOG_SCALE_RATE_S = -1.50


class VisualServoRefusal(ValueError):
    """The observation or requested tuning cannot safely produce authority."""


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
    yaw_error_gain: float = 0.15
    yaw_rate_gain: float = 0.035
    # Roll authority remains zero in schema /1.  Existing live evidence has a
    # sign conflict between Gate-0 centering and the later Gate-1 recovery
    # lineage; enable it only in a future version after an isolated response
    # calibration resolves that conflict.
    roll_error_gain: float = 0.0
    roll_rate_gain: float = 0.0
    vertical_error_gain: float = 0.16
    vertical_rate_gain: float = 0.035
    collective_error_gain: float = 0.060
    collective_rate_gain: float = 0.080
    advance_pitch_rad: float = -0.105
    brake_pitch_rad: float = 0.035
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
        if not 0.0 <= self.roll_error_gain <= 0.10:
            raise VisualServoRefusal("roll error gain is outside bounds")
        if not 0.0 <= self.roll_rate_gain <= 0.03:
            raise VisualServoRefusal("roll rate gain is outside bounds")
        if self.roll_error_gain != 0.0 or self.roll_rate_gain != 0.0:
            raise VisualServoRefusal(
                "visual-servo config/1 keeps lateral roll authority disabled"
            )
        if not 0.05 <= self.vertical_error_gain <= 0.30:
            raise VisualServoRefusal("vertical error gain is outside bounds")
        if not 0.0 <= self.vertical_rate_gain <= 0.08:
            raise VisualServoRefusal("vertical rate gain is outside bounds")
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
            or type(self.ambiguous) is not bool
        ):
            raise VisualServoRefusal("target flags must be exact booleans")

    @classmethod
    def from_visual_track(
        cls,
        track: object,
        *,
        stream_id: str,
    ) -> "VisualTarget":
        """Adapt the tracker convention explicitly for live servo authority.

        ``VisualTrack.center_norm[1]`` and its velocity are image-down
        positive.  The gate graph also exposes an elevation-up view, but this
        adapter intentionally consumes the raw image convention so the sign
        cannot be silently inverted at the runner boundary.
        """

        history = getattr(track, "history", None)
        if type(history) is not tuple or not history:
            raise VisualServoRefusal("visual track lacks exact sample history")
        sample = history[-1]
        token = getattr(track, "latest_token", None)
        publication_sequence = getattr(token, "publication_sequence", None)
        if type(publication_sequence) is not int or publication_sequence <= 0:
            raise VisualServoRefusal(
                "live servo authority requires receiver publication provenance"
            )
        center = getattr(track, "center_norm", None)
        velocity = getattr(track, "center_velocity_norm_s", None)
        if type(center) is not tuple or len(center) != 2:
            raise VisualServoRefusal("visual track center convention is invalid")
        if type(velocity) is not tuple or len(velocity) != 2:
            raise VisualServoRefusal("visual track velocity convention is invalid")
        apparent_scale = getattr(track, "apparent_scale", None)
        if (
            type(apparent_scale) not in {int, float}
            or not math.isfinite(float(apparent_scale))
            or float(apparent_scale) <= 0.0
        ):
            raise VisualServoRefusal("visual track apparent scale is invalid")
        clipping = getattr(track, "clipping", None)
        try:
            clipped = int(clipping) != 0
        except (TypeError, ValueError) as exc:
            raise VisualServoRefusal("visual track clipping state is invalid") from exc
        observation_ns = getattr(sample, "observation_monotonic_ns", None)
        if type(observation_ns) is not int or observation_ns < 0:
            raise VisualServoRefusal("visual track observation time is invalid")
        return cls(
            track_id=getattr(track, "track_id", None),
            frame_token=ServoFrameToken(
                stream_id=stream_id,
                generation=getattr(token, "generation", None),
                frame_id=getattr(token, "frame_id", None),
                publication_sequence=publication_sequence,
            ),
            received_monotonic_s=observation_ns / 1_000_000_000.0,
            normalized_x=float(center[0]),
            normalized_y_down=float(center[1]),
            normalized_x_rate_s=float(velocity[0]),
            normalized_y_rate_down_s=float(velocity[1]),
            log_scale=math.log(float(apparent_scale)),
            log_scale_rate_s=float(getattr(track, "log_scale_rate_s", math.nan)),
            confidence=float(getattr(track, "confidence", math.nan)),
            association_confidence=float(
                getattr(track, "association_confidence", math.nan)
            ),
            consecutive_frames=getattr(track, "consecutive_frame_count", None),
            clipped=clipped,
            center_censored=getattr(track, "center_censored", None),
            ambiguous=getattr(track, "ambiguous", None),
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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


class ImageVisualServo:
    """Stateful fresh-frame visual servo with align-before-advance gating."""

    def __init__(self, tuning: Optional[VisualServoTuning] = None) -> None:
        self.tuning = tuning or VisualServoTuning()
        self.reset_segment()

    def reset_segment(self) -> None:
        self._last_token: Optional[ServoFrameToken] = None
        self._segment_track_id: Optional[str] = None
        self._last_abs_error: Optional[Tuple[float, float]] = None
        self._corridor_frames = 0

    @property
    def corridor_frames(self) -> int:
        return self._corridor_frames

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
    ) -> VisualServoOutput:
        """Produce a bounded proposal from one distinct, authoritative frame.

        A caller may pre-orient toward a stable next-gate track, but the blend
        is suppressed unless the current aperture is already inside its safe
        corridor.  The current gate therefore retains passage authority.
        """

        scalars = (
            now_monotonic_s,
            segment_elapsed_s,
            segment_yaw_excursion_rad,
            requested_next_blend,
        )
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

        horizontal = float(current.normalized_x)
        vertical = float(current.normalized_y_down)
        horizontal_rate = float(current.normalized_x_rate_s)
        vertical_rate = float(current.normalized_y_rate_down_s)
        current_inside_position = (
            abs(horizontal) <= self.tuning.horizontal_corridor
            and abs(vertical) <= self.tuning.vertical_corridor
        )

        blend = 0.0
        next_horizontal: Optional[float] = None
        next_vertical: Optional[float] = None
        if next_target is not None and float(requested_next_blend) > 0.0:
            next_age_s = (
                float(now_monotonic_s)
                - float(next_target.received_monotonic_s)
            )
            same_frame = next_target.frame_token == current.frame_token
            next_authoritative = (
                not next_target.ambiguous
                and next_target.confidence >= 0.10
                and next_target.association_confidence >= 0.10
                and next_target.consecutive_frames >= 3
                and not next_target.clipped
                and not next_target.center_censored
                and abs(float(next_target.normalized_x_rate_s))
                <= self.tuning.stable_rate_norm_s
                and abs(float(next_target.normalized_y_rate_down_s))
                <= self.tuning.stable_rate_norm_s
                and abs(float(next_target.log_scale_rate_s))
                <= self.tuning.stable_scale_rate_s
                and -1e-6 <= next_age_s <= MAX_VISUAL_OBSERVATION_AGE_S
                and same_frame
                and next_target.track_id != current.track_id
            )
            if (
                next_authoritative
                and current_inside_position
                and self._corridor_frames
                >= self.tuning.required_corridor_frames
            ):
                blend = float(requested_next_blend)
                next_horizontal = float(next_target.normalized_x)
                next_vertical = float(next_target.normalized_y_down)
                horizontal = (
                    (1.0 - blend) * horizontal
                    + blend * next_horizontal
                )
                # Preserve more vertical authority for the current aperture;
                # early heading blend is useful, but a clipped next gate must
                # not pull the vehicle out of the current vertical corridor.
                vertical = (
                    (1.0 - 0.5 * blend) * vertical
                    + 0.5 * blend * next_vertical
                )
                horizontal_rate = (
                    (1.0 - blend) * horizontal_rate
                    + blend * float(next_target.normalized_x_rate_s)
                )
                vertical_rate = (
                    (1.0 - 0.5 * blend) * vertical_rate
                    + 0.5
                    * blend
                    * float(next_target.normalized_y_rate_down_s)
                )

        stable_rates = (
            abs(float(current.normalized_x_rate_s))
            <= self.tuning.stable_rate_norm_s
            and abs(float(current.normalized_y_rate_down_s))
            <= self.tuning.stable_rate_norm_s
            and abs(float(current.log_scale_rate_s))
            <= self.tuning.stable_scale_rate_s
        )

        previous_abs_error = self._last_abs_error
        horizontal_delta = (
            None
            if previous_abs_error is None
            else abs(float(current.normalized_x)) - previous_abs_error[0]
        )
        vertical_delta = (
            None
            if previous_abs_error is None
            else abs(float(current.normalized_y_down)) - previous_abs_error[1]
        )
        self._last_abs_error = (
            abs(float(current.normalized_x)),
            abs(float(current.normalized_y_down)),
        )
        self._last_token = current.frame_token

        worsening = bool(
            (horizontal_delta is not None and horizontal_delta > 0.015)
            or (vertical_delta is not None and vertical_delta > 0.015)
        )
        inside_corridor = current_inside_position and stable_rates and not worsening
        if inside_corridor:
            self._corridor_frames += 1
        else:
            self._corridor_frames = 0
        edge_risk = bool(
            abs(float(current.normalized_x)) >= self.tuning.edge_brake_x
            or abs(float(current.normalized_y_down))
            >= self.tuning.edge_brake_y
            or current.clipped
        )
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

        advance_enabled = bool(
            allow_advance
            and self._corridor_frames >= self.tuning.required_corridor_frames
            and not edge_risk
            and not scale_brake
            and not scale_retreat
            and not close_scale_brake
            and not worsening
        )

        brake_reason: Optional[str] = None
        if edge_risk:
            brake_reason = "target_edge_or_clipping"
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

        yaw_rate = _clamp(
            -self.tuning.yaw_error_gain * horizontal
            - self.tuning.yaw_rate_gain
            * horizontal_rate,
            -MAX_VISUAL_YAW_RATE_RAD_S,
            MAX_VISUAL_YAW_RATE_RAD_S,
        )
        yaw_envelope_limited = bool(
            abs(float(segment_yaw_excursion_rad))
            >= VISUAL_SEGMENT_YAW_SOFT_STOP_RAD
            and float(segment_yaw_excursion_rad) * yaw_rate > 0.0
        )
        if yaw_envelope_limited:
            yaw_rate = 0.0
            advance_enabled = False
            brake_reason = "segment_yaw_outward_soft_stop"

        target_roll = _clamp(
            self.tuning.roll_error_gain * float(current.normalized_x)
            + self.tuning.roll_rate_gain
            * float(current.normalized_x_rate_s),
            -MAX_VISUAL_TARGET_ROLL_RAD,
            MAX_VISUAL_TARGET_ROLL_RAD,
        )
        vertical_correction = (
            -self.tuning.vertical_error_gain * vertical
            - self.tuning.vertical_rate_gain
            * vertical_rate
        )
        if advance_enabled:
            pitch_basis = self.tuning.advance_pitch_rad
            thrust_basis = self.tuning.advance_thrust
        elif (
            edge_risk
            or scale_brake
            or scale_retreat
            or close_scale_brake
            or worsening
            or yaw_envelope_limited
        ):
            pitch_basis = self.tuning.brake_pitch_rad
            thrust_basis = self.tuning.brake_thrust
        else:
            pitch_basis = 0.0
            thrust_basis = self.tuning.align_thrust
        raw_target_pitch = pitch_basis + vertical_correction
        if not advance_enabled:
            # Normalized collective owns vertical alignment until the target is
            # in-corridor.  A low target must never turn alignment/braking into
            # a nose-down forward-closure command.
            raw_target_pitch = max(0.0, raw_target_pitch)
        target_pitch = _clamp(
            raw_target_pitch,
            MIN_VISUAL_TARGET_PITCH_RAD,
            MAX_VISUAL_TARGET_PITCH_RAD,
        )
        # This is the normalized form of the live-proved Gate-0 vertical
        # pixel-space collective law: a gate high in the image requests more
        # collective, while image motion toward the desired row damps it.
        thrust = _clamp(
            thrust_basis
            - self.tuning.collective_error_gain * vertical
            - self.tuning.collective_rate_gain
            * vertical_rate,
            MIN_VISUAL_THRUST,
            MAX_VISUAL_THRUST,
        )

        return VisualServoOutput(
            target_roll_rad=target_roll,
            target_pitch_rad=target_pitch,
            yaw_rate_rad_s=yaw_rate,
            thrust=thrust,
            corridor_frames=self._corridor_frames,
            advance_enabled=advance_enabled,
            next_gate_blend=blend,
            horizontal_error=float(current.normalized_x),
            vertical_error_image_down=float(current.normalized_y_down),
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
        )


__all__ = [
    "ImageVisualServo",
    "MAX_NEXT_GATE_BLEND",
    "MAX_VISUAL_OBSERVATION_AGE_S",
    "MAX_VISUAL_SEGMENT_DURATION_S",
    "MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD",
    "MAX_VISUAL_YAW_RATE_RAD_S",
    "ServoFrameToken",
    "VISUAL_SEGMENT_YAW_SOFT_STOP_RAD",
    "VisualServoOutput",
    "VisualServoRefusal",
    "VisualServoTuning",
    "VisualTarget",
]
