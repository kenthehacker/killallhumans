from __future__ import annotations

from dataclasses import replace

import pytest

from planning.vq2_visual_alignment import (
    MAX_ALIGNMENT_DIVERGENCE_NORM,
    RestrictedAlignmentMonitor,
    VisualAlignmentRefusal,
)
from planning.vq2_visual_servo import ServoFrameToken, VisualTarget


def _target(
    sequence: int,
    *,
    track_id: str = "vq2-track-000004",
    x: float = 0.6,
    y: float = -0.6,
    horizontal_censored: bool = False,
    vertical_censored: bool = False,
    scale_rate: float = 0.5,
) -> VisualTarget:
    return VisualTarget(
        track_id=track_id,
        frame_token=ServoFrameToken(
            "vq2-camera-udp-5600",
            1,
            1000 + sequence,
            sequence,
        ),
        received_monotonic_s=10.0 + 0.03 * sequence,
        normalized_x=x,
        normalized_y_down=y,
        normalized_x_rate_s=0.0,
        normalized_y_rate_down_s=0.0,
        log_scale=-1.5,
        log_scale_rate_s=scale_rate,
        confidence=0.8,
        association_confidence=0.8,
        consecutive_frames=5,
        clipped=horizontal_censored or vertical_censored,
        center_censored=horizontal_censored or vertical_censored,
        horizontal_censored=horizontal_censored,
        vertical_censored=vertical_censored,
    )


def test_accepts_only_joint_negative_uninterrupted_errors():
    monitor = RestrictedAlignmentMonitor(
        track_id="vq2-track-000004",
        required_improving_frames=3,
    )
    first = monitor.observe(
        _target(1, x=0.60, y=-0.62),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    second = monitor.observe(
        _target(2, x=0.58, y=-0.59),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    third = monitor.observe(
        _target(3, x=0.56, y=-0.55),
        response_evaluation_enabled=True,
        corridor_frames=1,
    )

    assert first.accepted is False
    assert second.accepted is False
    assert third.accepted is True
    assert third.improving_joint_frame_streak == 3
    assert third.horizontal_trend == "negative_uninterrupted"
    assert third.vertical_trend == "negative_uninterrupted"
    assert third.horizontal_deltas == pytest.approx((-0.02, -0.02))
    assert third.vertical_deltas == pytest.approx((-0.03, -0.04))
    assert third.corridor_frames == 1


def test_axis_censoring_cannot_bridge_or_create_an_alignment_claim():
    monitor = RestrictedAlignmentMonitor(
        track_id="vq2-track-000004",
        required_improving_frames=3,
    )
    monitor.observe(
        _target(1, x=0.60, y=-0.60),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    clipped = monitor.observe(
        _target(
            2,
            x=0.58,
            y=-0.90,
            vertical_censored=True,
        ),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    after_gap = monitor.observe(
        _target(3, x=0.56, y=-0.55),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )

    assert clipped.eligible_joint_frame_count == 0
    assert clipped.improving_joint_frame_streak == 0
    assert after_gap.improving_joint_frame_streak == 1
    assert after_gap.accepted is False
    assert after_gap.horizontal_abs_errors == pytest.approx((0.56,))
    assert after_gap.vertical_abs_errors == pytest.approx((0.55,))


def test_censored_gap_resets_uninterrupted_worsening_evidence():
    monitor = RestrictedAlignmentMonitor(
        track_id="vq2-track-000004",
        required_improving_frames=3,
    )
    monitor.observe(
        _target(1, x=0.50, y=-0.50),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    first_worse = monitor.observe(
        _target(2, x=0.51, y=-0.51),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    gap = monitor.observe(
        _target(3, x=0.70, y=-0.90, vertical_censored=True),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    after_gap = monitor.observe(
        _target(4, x=0.52, y=-0.52),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )

    assert first_worse.abort_reason is None
    assert gap.abort_reason is None
    assert after_gap.abort_reason is None
    assert after_gap.improving_joint_frame_streak == 1


def test_response_grace_frames_do_not_enter_trend_evidence():
    monitor = RestrictedAlignmentMonitor(
        track_id="vq2-track-000004",
        required_improving_frames=3,
    )
    grace = monitor.observe(
        _target(1, x=0.65, y=-0.73),
        response_evaluation_enabled=False,
        corridor_frames=0,
    )
    first = monitor.observe(
        _target(2, x=0.66, y=-0.74),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    assert grace.eligible_joint_frame_count == 0
    assert first.horizontal_abs_errors == pytest.approx((0.66,))
    assert first.vertical_abs_errors == pytest.approx((0.74,))


def test_two_uninterrupted_worsening_steps_abort():
    monitor = RestrictedAlignmentMonitor(
        track_id="vq2-track-000004",
        required_improving_frames=3,
    )
    states = [
        monitor.observe(
            _target(index, x=x, y=-y),
            response_evaluation_enabled=True,
            corridor_frames=0,
        )
        for index, (x, y) in enumerate(
            ((0.50, 0.50), (0.51, 0.51), (0.52, 0.52)),
            start=1,
        )
    ]
    assert states[-1].accepted is False
    assert states[-1].abort_reason == (
        "horizontal_error_worsening_uninterrupted"
    )
    assert states[-1].horizontal_trend == "positive_uninterrupted"
    assert states[-1].vertical_trend == "positive_uninterrupted"


def test_mixed_errors_cannot_be_relabelled_as_uninterrupted_success():
    monitor = RestrictedAlignmentMonitor(
        track_id="vq2-track-000004",
        required_improving_frames=3,
    )
    states = [
        monitor.observe(
            _target(index, x=value, y=-value),
            response_evaluation_enabled=True,
            corridor_frames=0,
        )
        for index, value in enumerate(
            (0.60, 0.61, 0.59, 0.58),
            start=1,
        )
    ]

    assert states[-1].improving_joint_frame_streak == 3
    assert states[-1].horizontal_trend == "mixed"
    assert states[-1].vertical_trend == "mixed"
    assert states[-1].accepted is False


def test_divergence_identity_and_token_fail_closed():
    monitor = RestrictedAlignmentMonitor(
        track_id="vq2-track-000004",
        required_improving_frames=3,
    )
    initial = _target(1, x=0.40, y=-0.40)
    monitor.observe(
        initial,
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    diverged = monitor.observe(
        _target(
            2,
            x=0.40 + MAX_ALIGNMENT_DIVERGENCE_NORM + 0.001,
            y=-0.40,
        ),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    assert diverged.abort_reason == "horizontal_error_diverged"
    with pytest.raises(VisualAlignmentRefusal, match="identity changed"):
        monitor.observe(
            _target(3, track_id="vq2-track-999999"),
            response_evaluation_enabled=True,
            corridor_frames=0,
        )
    with pytest.raises(VisualAlignmentRefusal, match="did not advance"):
        monitor.observe(
            replace(initial, normalized_x=0.3),
            response_evaluation_enabled=True,
            corridor_frames=0,
        )
