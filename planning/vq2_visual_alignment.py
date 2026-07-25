"""Pure acceptance monitor for the bounded VQ2 visual-alignment milestone.

This module does not produce flight commands.  It consumes the exact,
axis-censored observations already admitted by :mod:`vq2_visual_servo` and
decides whether one restricted alignment segment has demonstrated the required
joint horizontal and vertical improvement.  Censored axes never contribute to
an alignment claim, and a gap cannot be bridged by comparing observations from
opposite sides of that gap.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from planning.vq2_visual_servo import ServoFrameToken, VisualTarget


MAX_ALIGNMENT_MONITOR_FRAMES = 64
MAX_ALIGNMENT_DIVERGENCE_NORM = 0.08
MAX_CONSECUTIVE_WORSENING_STEPS = 2
MAX_CONSECUTIVE_GEOMETRY_CENSORED_RESPONSE_FRAMES = 2

# Code-owned admission bounds for the first powered command after an
# authoritative promotion.  These are safety policy, not controller tuning.
POST_PROMOTION_ENTRY_MAX_ABS_X_NORM = 0.67
POST_PROMOTION_ENTRY_MAX_ABS_Y_NORM = 0.71
POST_PROMOTION_ENTRY_MAX_OUTWARD_RATE_NORM_S = 0.25
POST_PROMOTION_ENTRY_MAX_LOG_SCALE_RATE_S = 0.85
# Two earlier collision-free build-3385 Gate-0 passages reached post-credit
# zero authority at -0.0473 and -0.0448 rad; attempt 19 reached the same clean
# handoff at -0.064325 rad.  The -0.065 floor retains a measured margin while
# still requiring a nonnegative braking target to be reachable inside the
# separate 0.08 rad entry-attitude delta and 0.12 rad/s command-rate envelopes.
POST_PROMOTION_ENTRY_MIN_MEASURED_PITCH_RAD = -0.065

# A promoted track that misses the ordinary entry-rate gate may receive a
# short no-advance capture response only inside this stricter predictive
# envelope.  These bounds cover the latest unclipped build-3385 handoff
# (x=0.553, y=-0.661, vx=0.333/s, vy=-0.383/s, scale-rate=0.928/s)
# without admitting the earlier clipped/out-of-bounds handoffs.  The exact
# attempt-19 continuation remained clean and projected inside every unchanged
# image/scale bound at |vertical rate|=0.4471302591/s, so retain 0.45/s as the
# narrow capture ceiling.  These are code-owned safety policy, not controller
# configuration.
POST_PROMOTION_CAPTURE_PROJECTION_HORIZON_S = 0.10
POST_PROMOTION_CAPTURE_MAX_ABS_X_NORM = 0.62
POST_PROMOTION_CAPTURE_MAX_ABS_Y_NORM = 0.70
POST_PROMOTION_CAPTURE_MAX_ABS_CENTER_RATE_NORM_S = 0.45
POST_PROMOTION_CAPTURE_MAX_ABS_LOG_SCALE_RATE_S = 1.00
POST_PROMOTION_CAPTURE_MAX_APPARENT_SCALE = 0.20
POST_PROMOTION_CAPTURE_MAX_PROJECTED_ABS_X_NORM = 0.67
POST_PROMOTION_CAPTURE_MAX_PROJECTED_ABS_Y_NORM = 0.71
POST_PROMOTION_CAPTURE_MAX_PROJECTED_APPARENT_SCALE = 0.22


class VisualAlignmentRefusal(ValueError):
    """The supplied observation cannot support a bounded alignment claim."""


def _signed_outward_rate(error: float, rate: float) -> float:
    """Return positive motion away from image center on one signed axis."""

    if error > 0.0:
        return rate
    if error < 0.0:
        return -rate
    # At exact center either direction increases absolute error.
    return abs(rate)


@dataclass(frozen=True)
class VisualAlignmentEntryAdmission:
    """Exact bounded facts admitted for first post-promotion authority."""

    track_id: str
    frame_token: ServoFrameToken
    horizontal_error: float
    vertical_error_image_down: float
    horizontal_outward_rate_s: float
    vertical_outward_rate_down_s: float
    log_scale_rate_s: float
    measured_pitch_rad: float


@dataclass(frozen=True)
class VisualAlignmentCaptureAdmission:
    """Predictive facts authorizing only short no-advance capture."""

    track_id: str
    frame_token: ServoFrameToken
    horizontal_error: float
    vertical_error_image_down: float
    horizontal_rate_s: float
    vertical_rate_down_s: float
    log_scale_rate_s: float
    apparent_scale: float
    projected_horizontal_error: float
    projected_vertical_error_image_down: float
    projected_apparent_scale: float
    measured_pitch_rad: float


def _require_entry_observation(
    target: VisualTarget,
    *,
    measured_pitch_rad: float,
) -> tuple[float, float, float]:
    """Validate common promoted-track authority and return signed geometry."""

    if type(target) is not VisualTarget:
        raise VisualAlignmentRefusal(
            "post-promotion entry requires an exact VisualTarget"
        )
    if (
        type(measured_pitch_rad) not in {int, float}
        or not math.isfinite(float(measured_pitch_rad))
    ):
        raise VisualAlignmentRefusal(
            "post-promotion measured pitch must be finite"
        )
    if target.ambiguous:
        raise VisualAlignmentRefusal(
            "post-promotion entry target is ambiguous"
        )
    if (
        target.clipped
        or target.center_censored
        or target.horizontal_geometry_censored
        or target.vertical_geometry_censored
    ):
        raise VisualAlignmentRefusal(
            "post-promotion entry target geometry is clipped or censored"
        )
    return (
        float(target.normalized_x),
        float(target.normalized_y_down),
        float(measured_pitch_rad),
    )


def require_visual_alignment_entry(
    target: VisualTarget,
    *,
    measured_pitch_rad: float,
) -> VisualAlignmentEntryAdmission:
    """Fail closed unless a promoted target is safe for first command authority.

    The check is deliberately image-relative and does not claim metric pose or
    distance.  Its bounds are immutable module policy for the post-promotion
    handoff; callers cannot loosen them through controller configuration.
    """

    horizontal, vertical, measured_pitch = _require_entry_observation(
        target,
        measured_pitch_rad=measured_pitch_rad,
    )
    if abs(horizontal) > POST_PROMOTION_ENTRY_MAX_ABS_X_NORM:
        raise VisualAlignmentRefusal(
            "post-promotion horizontal error exceeds the entry bound"
        )
    if abs(vertical) > POST_PROMOTION_ENTRY_MAX_ABS_Y_NORM:
        raise VisualAlignmentRefusal(
            "post-promotion vertical error exceeds the entry bound"
        )

    horizontal_outward_rate = _signed_outward_rate(
        horizontal,
        float(target.normalized_x_rate_s),
    )
    if (
        horizontal_outward_rate
        > POST_PROMOTION_ENTRY_MAX_OUTWARD_RATE_NORM_S
    ):
        raise VisualAlignmentRefusal(
            "post-promotion horizontal motion is outward above the entry bound"
        )
    vertical_outward_rate = _signed_outward_rate(
        vertical,
        float(target.normalized_y_rate_down_s),
    )
    if (
        vertical_outward_rate
        > POST_PROMOTION_ENTRY_MAX_OUTWARD_RATE_NORM_S
    ):
        raise VisualAlignmentRefusal(
            "post-promotion vertical motion is outward above the entry bound"
        )
    if (
        float(target.log_scale_rate_s)
        > POST_PROMOTION_ENTRY_MAX_LOG_SCALE_RATE_S
    ):
        raise VisualAlignmentRefusal(
            "post-promotion scale closure exceeds the entry bound"
        )
    if (
        measured_pitch
        < POST_PROMOTION_ENTRY_MIN_MEASURED_PITCH_RAD
    ):
        raise VisualAlignmentRefusal(
            "post-promotion measured pitch has not reached the entry brake bound"
        )
    return VisualAlignmentEntryAdmission(
        track_id=target.track_id,
        frame_token=target.frame_token,
        horizontal_error=horizontal,
        vertical_error_image_down=vertical,
        horizontal_outward_rate_s=horizontal_outward_rate,
        vertical_outward_rate_down_s=vertical_outward_rate,
        log_scale_rate_s=float(target.log_scale_rate_s),
        measured_pitch_rad=measured_pitch,
    )


def require_visual_alignment_capture_entry(
    target: VisualTarget,
    *,
    measured_pitch_rad: float,
) -> VisualAlignmentCaptureAdmission:
    """Admit only a short predictive, no-advance post-promotion response.

    This is deliberately separate from :func:`require_visual_alignment_entry`:
    it does not weaken the ordinary entry gate.  A caller must revalidate this
    envelope on every fresh frame and prove the ordinary entry gate before
    claiming alignment.
    """

    horizontal, vertical, measured_pitch = _require_entry_observation(
        target,
        measured_pitch_rad=measured_pitch_rad,
    )
    horizontal_rate = float(target.normalized_x_rate_s)
    vertical_rate = float(target.normalized_y_rate_down_s)
    scale_rate = float(target.log_scale_rate_s)
    log_scale = float(target.log_scale)
    if abs(horizontal) > POST_PROMOTION_CAPTURE_MAX_ABS_X_NORM:
        raise VisualAlignmentRefusal(
            "post-promotion capture horizontal error exceeds its bound"
        )
    if abs(vertical) > POST_PROMOTION_CAPTURE_MAX_ABS_Y_NORM:
        raise VisualAlignmentRefusal(
            "post-promotion capture vertical error exceeds its bound"
        )
    if (
        abs(horizontal_rate)
        > POST_PROMOTION_CAPTURE_MAX_ABS_CENTER_RATE_NORM_S
        or abs(vertical_rate)
        > POST_PROMOTION_CAPTURE_MAX_ABS_CENTER_RATE_NORM_S
    ):
        raise VisualAlignmentRefusal(
            "post-promotion capture center motion exceeds its bound"
        )
    if (
        abs(scale_rate)
        > POST_PROMOTION_CAPTURE_MAX_ABS_LOG_SCALE_RATE_S
    ):
        raise VisualAlignmentRefusal(
            "post-promotion capture scale motion exceeds its bound"
        )
    if measured_pitch < POST_PROMOTION_ENTRY_MIN_MEASURED_PITCH_RAD:
        raise VisualAlignmentRefusal(
            "post-promotion measured pitch has not reached the entry brake bound"
        )

    max_log_scale = math.log(
        POST_PROMOTION_CAPTURE_MAX_APPARENT_SCALE
    )
    if log_scale > max_log_scale:
        raise VisualAlignmentRefusal(
            "post-promotion capture apparent scale exceeds its bound"
        )
    horizon = POST_PROMOTION_CAPTURE_PROJECTION_HORIZON_S
    projected_horizontal = horizontal + horizontal_rate * horizon
    projected_vertical = vertical + vertical_rate * horizon
    projected_log_scale = log_scale + scale_rate * horizon
    if (
        abs(projected_horizontal)
        > POST_PROMOTION_CAPTURE_MAX_PROJECTED_ABS_X_NORM
    ):
        raise VisualAlignmentRefusal(
            "post-promotion capture horizontal projection exceeds its bound"
        )
    if (
        abs(projected_vertical)
        > POST_PROMOTION_CAPTURE_MAX_PROJECTED_ABS_Y_NORM
    ):
        raise VisualAlignmentRefusal(
            "post-promotion capture vertical projection exceeds its bound"
        )
    if projected_log_scale > math.log(
        POST_PROMOTION_CAPTURE_MAX_PROJECTED_APPARENT_SCALE
    ):
        raise VisualAlignmentRefusal(
            "post-promotion capture scale projection exceeds its bound"
        )
    return VisualAlignmentCaptureAdmission(
        track_id=target.track_id,
        frame_token=target.frame_token,
        horizontal_error=horizontal,
        vertical_error_image_down=vertical,
        horizontal_rate_s=horizontal_rate,
        vertical_rate_down_s=vertical_rate,
        log_scale_rate_s=scale_rate,
        apparent_scale=math.exp(log_scale),
        projected_horizontal_error=projected_horizontal,
        projected_vertical_error_image_down=projected_vertical,
        projected_apparent_scale=math.exp(projected_log_scale),
        measured_pitch_rad=measured_pitch,
    )


@dataclass(frozen=True)
class VisualAlignmentTrend:
    track_id: str
    latest_token: ServoFrameToken
    processed_frame_count: int
    eligible_joint_frame_count: int
    improving_joint_frame_streak: int
    horizontal_abs_errors: tuple[float, ...]
    vertical_abs_errors: tuple[float, ...]
    horizontal_deltas: tuple[float, ...]
    vertical_deltas: tuple[float, ...]
    log_scale_rates_s: tuple[float, ...]
    horizontal_trend: str
    vertical_trend: str
    scale_rate_trend: str
    corridor_frames: int
    accepted: bool
    abort_reason: Optional[str]


def _signed_trend(values: tuple[float, ...]) -> str:
    if len(values) < 2:
        return "insufficient"
    deltas = tuple(
        values[index] - values[index - 1]
        for index in range(1, len(values))
    )
    if all(delta < 0.0 for delta in deltas):
        return "negative_uninterrupted"
    if all(delta > 0.0 for delta in deltas):
        return "positive_uninterrupted"
    if all(delta == 0.0 for delta in deltas):
        return "flat"
    return "mixed"


class RestrictedAlignmentMonitor:
    """Require consecutive joint image-error improvement on exact frames."""

    def __init__(
        self,
        *,
        track_id: str,
        required_improving_frames: int,
    ) -> None:
        if type(track_id) is not str or not track_id or len(track_id) > 128:
            raise VisualAlignmentRefusal("track_id must be a bounded string")
        if (
            type(required_improving_frames) is not int
            or not 3 <= required_improving_frames <= 6
        ):
            raise VisualAlignmentRefusal(
                "required improving frames must be an exact int in [3, 6]"
            )
        self._track_id = track_id
        self._required = required_improving_frames
        self._last_token: Optional[ServoFrameToken] = None
        self._processed = 0
        self._response_evaluation_started = False
        self._geometry_censored_response_streak = 0
        self._joint_streak = 0
        self._joint_frame_count = 0
        self._last_joint_errors: Optional[tuple[float, float]] = None
        self._joint_horizontal_errors: list[float] = []
        self._joint_vertical_errors: list[float] = []
        self._horizontal_errors: list[float] = []
        self._vertical_errors: list[float] = []
        self._horizontal_deltas: list[float] = []
        self._vertical_deltas: list[float] = []
        self._scale_rates: list[float] = []
        self._horizontal_baseline: Optional[float] = None
        self._vertical_baseline: Optional[float] = None
        self._last_horizontal_error: Optional[float] = None
        self._last_vertical_error: Optional[float] = None
        self._horizontal_worsening_streak = 0
        self._vertical_worsening_streak = 0
        self._abort_reason: Optional[str] = None

    def _reset_joint_evidence(self) -> None:
        self._joint_streak = 0
        self._joint_frame_count = 0
        self._last_joint_errors = None
        self._joint_horizontal_errors.clear()
        self._joint_vertical_errors.clear()

    def _reset_horizontal_evidence(self) -> None:
        self._horizontal_errors.clear()
        self._horizontal_deltas.clear()
        self._horizontal_baseline = None
        self._last_horizontal_error = None
        self._horizontal_worsening_streak = 0

    def _reset_vertical_evidence(self) -> None:
        self._vertical_errors.clear()
        self._vertical_deltas.clear()
        self._vertical_baseline = None
        self._last_vertical_error = None
        self._vertical_worsening_streak = 0

    def observe(
        self,
        target: VisualTarget,
        *,
        response_evaluation_enabled: bool,
        corridor_frames: int,
    ) -> VisualAlignmentTrend:
        if type(target) is not VisualTarget:
            raise VisualAlignmentRefusal(
                "alignment monitor requires an exact VisualTarget"
            )
        if type(response_evaluation_enabled) is not bool:
            raise VisualAlignmentRefusal(
                "response_evaluation_enabled must be an exact bool"
            )
        if type(corridor_frames) is not int or corridor_frames < 0:
            raise VisualAlignmentRefusal(
                "corridor_frames must be a nonnegative exact int"
            )
        if target.track_id != self._track_id:
            raise VisualAlignmentRefusal(
                "alignment target identity changed without promotion"
            )
        token = target.frame_token
        if self._last_token is not None and (
            token.stream_id != self._last_token.stream_id
            or token.generation != self._last_token.generation
            or token.publication_sequence
            <= self._last_token.publication_sequence
        ):
            raise VisualAlignmentRefusal(
                "alignment frame publication did not advance exactly"
            )
        self._last_token = token
        self._processed += 1
        if self._processed > MAX_ALIGNMENT_MONITOR_FRAMES:
            raise VisualAlignmentRefusal(
                "alignment monitor exceeded its bounded frame history"
            )
        scale_rate = float(target.log_scale_rate_s)
        if not math.isfinite(scale_rate):
            raise VisualAlignmentRefusal("alignment scale rate is non-finite")
        self._scale_rates.append(scale_rate)

        self._response_evaluation_started = bool(
            self._response_evaluation_started
            or response_evaluation_enabled
        )
        if not self._response_evaluation_started:
            self._reset_joint_evidence()
            return self._snapshot(corridor_frames)

        horizontal_censored = target.horizontal_geometry_censored
        vertical_censored = target.vertical_geometry_censored
        geometry_censored = horizontal_censored or vertical_censored
        self._geometry_censored_response_streak = (
            self._geometry_censored_response_streak + 1
            if geometry_censored
            else 0
        )
        if (
            self._geometry_censored_response_streak
            >= MAX_CONSECUTIVE_GEOMETRY_CENSORED_RESPONSE_FRAMES
        ):
            self._abort_reason = (
                self._abort_reason or "geometry_censored_uninterrupted"
            )

        horizontal: Optional[float] = None
        if horizontal_censored:
            self._reset_horizontal_evidence()
        else:
            horizontal = abs(float(target.normalized_x))
            self._horizontal_errors.append(horizontal)
            if self._horizontal_baseline is None:
                self._horizontal_baseline = horizontal
            if (
                horizontal
                > self._horizontal_baseline + MAX_ALIGNMENT_DIVERGENCE_NORM
            ):
                self._abort_reason = (
                    self._abort_reason or "horizontal_error_diverged"
                )
            if self._last_horizontal_error is None:
                self._horizontal_worsening_streak = 0
            else:
                horizontal_delta = (
                    horizontal - self._last_horizontal_error
                )
                self._horizontal_deltas.append(horizontal_delta)
                self._horizontal_worsening_streak = (
                    self._horizontal_worsening_streak + 1
                    if horizontal_delta > 0.0
                    else 0
                )
                if (
                    self._horizontal_worsening_streak
                    >= MAX_CONSECUTIVE_WORSENING_STEPS
                ):
                    self._abort_reason = (
                        self._abort_reason
                        or "horizontal_error_worsening_uninterrupted"
                    )
            self._last_horizontal_error = horizontal

        vertical: Optional[float] = None
        if vertical_censored:
            self._reset_vertical_evidence()
        else:
            vertical = abs(float(target.normalized_y_down))
            self._vertical_errors.append(vertical)
            if self._vertical_baseline is None:
                self._vertical_baseline = vertical
            if vertical > self._vertical_baseline + MAX_ALIGNMENT_DIVERGENCE_NORM:
                self._abort_reason = (
                    self._abort_reason or "vertical_error_diverged"
                )
            if self._last_vertical_error is None:
                self._vertical_worsening_streak = 0
            else:
                vertical_delta = vertical - self._last_vertical_error
                self._vertical_deltas.append(vertical_delta)
                self._vertical_worsening_streak = (
                    self._vertical_worsening_streak + 1
                    if vertical_delta > 0.0
                    else 0
                )
                if (
                    self._vertical_worsening_streak
                    >= MAX_CONSECUTIVE_WORSENING_STEPS
                ):
                    self._abort_reason = (
                        self._abort_reason
                        or "vertical_error_worsening_uninterrupted"
                    )
            self._last_vertical_error = vertical

        if geometry_censored:
            self._reset_joint_evidence()
            return self._snapshot(corridor_frames)
        assert horizontal is not None and vertical is not None

        self._joint_horizontal_errors.append(horizontal)
        self._joint_vertical_errors.append(vertical)
        self._joint_frame_count += 1
        previous = self._last_joint_errors
        if previous is None:
            self._joint_streak = 1
        else:
            horizontal_joint_delta = horizontal - previous[0]
            vertical_joint_delta = vertical - previous[1]
            self._joint_streak = (
                self._joint_streak + 1
                if (
                    horizontal_joint_delta < 0.0
                    and vertical_joint_delta < 0.0
                )
                else 1
            )
        self._last_joint_errors = (horizontal, vertical)
        return self._snapshot(corridor_frames)

    def _snapshot(self, corridor_frames: int) -> VisualAlignmentTrend:
        if self._last_token is None:
            raise RuntimeError("alignment monitor has no exact frame")
        horizontal_values = tuple(self._horizontal_errors)
        vertical_values = tuple(self._vertical_errors)
        joint_horizontal_values = tuple(self._joint_horizontal_errors)
        joint_vertical_values = tuple(self._joint_vertical_errors)
        scale_rates = tuple(self._scale_rates)
        horizontal_trend = _signed_trend(horizontal_values)
        vertical_trend = _signed_trend(vertical_values)
        joint_horizontal_trend = _signed_trend(joint_horizontal_values)
        joint_vertical_trend = _signed_trend(joint_vertical_values)
        return VisualAlignmentTrend(
            track_id=self._track_id,
            latest_token=self._last_token,
            processed_frame_count=self._processed,
            eligible_joint_frame_count=self._joint_frame_count,
            improving_joint_frame_streak=self._joint_streak,
            horizontal_abs_errors=horizontal_values,
            vertical_abs_errors=vertical_values,
            horizontal_deltas=tuple(self._horizontal_deltas),
            vertical_deltas=tuple(self._vertical_deltas),
            log_scale_rates_s=scale_rates,
            horizontal_trend=horizontal_trend,
            vertical_trend=vertical_trend,
            scale_rate_trend=_signed_trend(scale_rates),
            corridor_frames=corridor_frames,
            accepted=bool(
                self._abort_reason is None
                and self._joint_streak >= self._required
                and horizontal_trend == "negative_uninterrupted"
                and vertical_trend == "negative_uninterrupted"
                and joint_horizontal_trend == "negative_uninterrupted"
                and joint_vertical_trend == "negative_uninterrupted"
            ),
            abort_reason=self._abort_reason,
        )


__all__ = [
    "MAX_ALIGNMENT_DIVERGENCE_NORM",
    "MAX_ALIGNMENT_MONITOR_FRAMES",
    "MAX_CONSECUTIVE_GEOMETRY_CENSORED_RESPONSE_FRAMES",
    "MAX_CONSECUTIVE_WORSENING_STEPS",
    "POST_PROMOTION_ENTRY_MAX_ABS_X_NORM",
    "POST_PROMOTION_ENTRY_MAX_ABS_Y_NORM",
    "POST_PROMOTION_ENTRY_MAX_LOG_SCALE_RATE_S",
    "POST_PROMOTION_ENTRY_MAX_OUTWARD_RATE_NORM_S",
    "POST_PROMOTION_ENTRY_MIN_MEASURED_PITCH_RAD",
    "POST_PROMOTION_CAPTURE_MAX_ABS_CENTER_RATE_NORM_S",
    "POST_PROMOTION_CAPTURE_MAX_ABS_LOG_SCALE_RATE_S",
    "POST_PROMOTION_CAPTURE_MAX_ABS_X_NORM",
    "POST_PROMOTION_CAPTURE_MAX_ABS_Y_NORM",
    "POST_PROMOTION_CAPTURE_MAX_APPARENT_SCALE",
    "POST_PROMOTION_CAPTURE_MAX_PROJECTED_ABS_X_NORM",
    "POST_PROMOTION_CAPTURE_MAX_PROJECTED_ABS_Y_NORM",
    "POST_PROMOTION_CAPTURE_MAX_PROJECTED_APPARENT_SCALE",
    "POST_PROMOTION_CAPTURE_PROJECTION_HORIZON_S",
    "RestrictedAlignmentMonitor",
    "VisualAlignmentCaptureAdmission",
    "VisualAlignmentEntryAdmission",
    "VisualAlignmentRefusal",
    "VisualAlignmentTrend",
    "require_visual_alignment_capture_entry",
    "require_visual_alignment_entry",
]
