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


class VisualAlignmentRefusal(ValueError):
    """The supplied observation cannot support a bounded alignment claim."""


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
        self._joint_streak = 0
        self._joint_frame_count = 0
        self._last_joint_errors: Optional[tuple[float, float]] = None
        self._horizontal_errors: list[float] = []
        self._vertical_errors: list[float] = []
        self._horizontal_deltas: list[float] = []
        self._vertical_deltas: list[float] = []
        self._scale_rates: list[float] = []
        self._horizontal_baseline: Optional[float] = None
        self._vertical_baseline: Optional[float] = None
        self._horizontal_worsening_streak = 0
        self._vertical_worsening_streak = 0
        self._abort_reason: Optional[str] = None

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

        joint_usable = bool(
            response_evaluation_enabled
            and not target.horizontal_geometry_censored
            and not target.vertical_geometry_censored
        )
        if not joint_usable:
            self._joint_streak = 0
            self._joint_frame_count = 0
            self._last_joint_errors = None
            self._horizontal_worsening_streak = 0
            self._vertical_worsening_streak = 0
            self._horizontal_errors.clear()
            self._vertical_errors.clear()
            self._horizontal_deltas.clear()
            self._vertical_deltas.clear()
            self._horizontal_baseline = None
            self._vertical_baseline = None
            return self._snapshot(corridor_frames)

        horizontal = abs(float(target.normalized_x))
        vertical = abs(float(target.normalized_y_down))
        if not math.isfinite(horizontal) or not math.isfinite(vertical):
            raise VisualAlignmentRefusal("alignment errors are non-finite")
        self._horizontal_errors.append(horizontal)
        self._vertical_errors.append(vertical)
        self._joint_frame_count += 1
        if self._horizontal_baseline is None:
            self._horizontal_baseline = horizontal
        if self._vertical_baseline is None:
            self._vertical_baseline = vertical

        if horizontal > self._horizontal_baseline + MAX_ALIGNMENT_DIVERGENCE_NORM:
            self._abort_reason = "horizontal_error_diverged"
        if vertical > self._vertical_baseline + MAX_ALIGNMENT_DIVERGENCE_NORM:
            self._abort_reason = (
                self._abort_reason or "vertical_error_diverged"
            )

        previous = self._last_joint_errors
        if previous is None:
            self._joint_streak = 1
            self._horizontal_worsening_streak = 0
            self._vertical_worsening_streak = 0
        else:
            horizontal_delta = horizontal - previous[0]
            vertical_delta = vertical - previous[1]
            self._horizontal_deltas.append(horizontal_delta)
            self._vertical_deltas.append(vertical_delta)
            self._horizontal_worsening_streak = (
                self._horizontal_worsening_streak + 1
                if horizontal_delta > 0.0
                else 0
            )
            self._vertical_worsening_streak = (
                self._vertical_worsening_streak + 1
                if vertical_delta > 0.0
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
            if (
                self._vertical_worsening_streak
                >= MAX_CONSECUTIVE_WORSENING_STEPS
            ):
                self._abort_reason = (
                    self._abort_reason
                    or "vertical_error_worsening_uninterrupted"
                )
            self._joint_streak = (
                self._joint_streak + 1
                if horizontal_delta < 0.0 and vertical_delta < 0.0
                else 1
            )
        self._last_joint_errors = (horizontal, vertical)
        return self._snapshot(corridor_frames)

    def _snapshot(self, corridor_frames: int) -> VisualAlignmentTrend:
        if self._last_token is None:
            raise RuntimeError("alignment monitor has no exact frame")
        horizontal_values = tuple(self._horizontal_errors)
        vertical_values = tuple(self._vertical_errors)
        scale_rates = tuple(self._scale_rates)
        horizontal_trend = _signed_trend(horizontal_values)
        vertical_trend = _signed_trend(vertical_values)
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
            ),
            abort_reason=self._abort_reason,
        )


__all__ = [
    "MAX_ALIGNMENT_DIVERGENCE_NORM",
    "MAX_ALIGNMENT_MONITOR_FRAMES",
    "MAX_CONSECUTIVE_WORSENING_STEPS",
    "RestrictedAlignmentMonitor",
    "VisualAlignmentRefusal",
    "VisualAlignmentTrend",
]
