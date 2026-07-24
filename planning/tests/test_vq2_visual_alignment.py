from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math

import pytest

from planning.vq2_visual_alignment import (
    MAX_ALIGNMENT_DIVERGENCE_NORM,
    POST_PROMOTION_ENTRY_MAX_ABS_X_NORM,
    POST_PROMOTION_ENTRY_MAX_ABS_Y_NORM,
    POST_PROMOTION_ENTRY_MAX_LOG_SCALE_RATE_S,
    POST_PROMOTION_ENTRY_MAX_OUTWARD_RATE_NORM_S,
    POST_PROMOTION_ENTRY_MIN_MEASURED_PITCH_RAD,
    RestrictedAlignmentMonitor,
    VisualAlignmentEntryAdmission,
    VisualAlignmentRefusal,
    require_visual_alignment_entry,
)
from planning.vq2_visual_servo import ServoFrameToken, VisualTarget


def _target(
    sequence: int,
    *,
    track_id: str = "vq2-track-000004",
    x: float = 0.6,
    y: float = -0.6,
    x_rate: float = 0.0,
    y_rate: float = 0.0,
    horizontal_censored: bool = False,
    vertical_censored: bool = False,
    clipped: bool | None = None,
    center_censored: bool | None = None,
    ambiguous: bool = False,
    scale_rate: float = 0.5,
) -> VisualTarget:
    if clipped is None:
        clipped = horizontal_censored or vertical_censored
    if center_censored is None:
        center_censored = horizontal_censored or vertical_censored
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
        normalized_x_rate_s=x_rate,
        normalized_y_rate_down_s=y_rate,
        log_scale=-1.5,
        log_scale_rate_s=scale_rate,
        confidence=0.8,
        association_confidence=0.8,
        consecutive_frames=5,
        clipped=clipped,
        center_censored=center_censored,
        horizontal_censored=horizontal_censored,
        vertical_censored=vertical_censored,
        ambiguous=ambiguous,
    )


def test_post_promotion_entry_accepts_exact_code_owned_bounds():
    target = _target(
        1,
        x=-POST_PROMOTION_ENTRY_MAX_ABS_X_NORM,
        y=POST_PROMOTION_ENTRY_MAX_ABS_Y_NORM,
        x_rate=-POST_PROMOTION_ENTRY_MAX_OUTWARD_RATE_NORM_S,
        y_rate=POST_PROMOTION_ENTRY_MAX_OUTWARD_RATE_NORM_S,
        scale_rate=POST_PROMOTION_ENTRY_MAX_LOG_SCALE_RATE_S,
    )

    admission = require_visual_alignment_entry(
        target,
        measured_pitch_rad=POST_PROMOTION_ENTRY_MIN_MEASURED_PITCH_RAD,
    )

    assert type(admission) is VisualAlignmentEntryAdmission
    assert admission.track_id == target.track_id
    assert admission.frame_token == target.frame_token
    assert admission.horizontal_error == -0.67
    assert admission.vertical_error_image_down == 0.71
    assert admission.horizontal_outward_rate_s == 0.25
    assert admission.vertical_outward_rate_down_s == 0.25
    assert admission.log_scale_rate_s == 0.85
    assert admission.measured_pitch_rad == -0.06
    with pytest.raises(FrozenInstanceError):
        admission.horizontal_error = 0.0


@pytest.mark.parametrize(
    ("target", "pitch", "reason"),
    (
        (_target(1, ambiguous=True), 0.0, "ambiguous"),
        (_target(1, clipped=True), 0.0, "clipped or censored"),
        (_target(1, center_censored=True), 0.0, "clipped or censored"),
        (
            _target(
                1,
                horizontal_censored=True,
                clipped=False,
                center_censored=False,
            ),
            0.0,
            "clipped or censored",
        ),
        (
            _target(
                1,
                vertical_censored=True,
                clipped=False,
                center_censored=False,
            ),
            0.0,
            "clipped or censored",
        ),
        (_target(1, x=0.671), 0.0, "horizontal error"),
        (_target(1, x=-0.671), 0.0, "horizontal error"),
        (_target(1, y=0.711), 0.0, "vertical error"),
        (_target(1, y=-0.711), 0.0, "vertical error"),
        (
            _target(1, x=0.5, x_rate=0.251),
            0.0,
            "horizontal motion",
        ),
        (
            _target(1, x=-0.5, x_rate=-0.251),
            0.0,
            "horizontal motion",
        ),
        (
            _target(1, x=0.0, x_rate=-0.251),
            0.0,
            "horizontal motion",
        ),
        (
            _target(1, y=0.5, y_rate=0.251),
            0.0,
            "vertical motion",
        ),
        (
            _target(1, y=-0.5, y_rate=-0.251),
            0.0,
            "vertical motion",
        ),
        (
            _target(1, y=0.0, y_rate=0.251),
            0.0,
            "vertical motion",
        ),
        (_target(1, scale_rate=0.851), 0.0, "scale closure"),
        (
            _target(1),
            math.nextafter(
                POST_PROMOTION_ENTRY_MIN_MEASURED_PITCH_RAD,
                -math.inf,
            ),
            "measured pitch",
        ),
    ),
)
def test_post_promotion_entry_rejects_each_unsafe_bound(
    target,
    pitch,
    reason,
):
    with pytest.raises(VisualAlignmentRefusal, match=reason):
        require_visual_alignment_entry(
            target,
            measured_pitch_rad=pitch,
        )


@pytest.mark.parametrize("pitch", (True, "0.0", math.nan, math.inf, -math.inf))
def test_post_promotion_entry_requires_exact_finite_inputs(pitch):
    with pytest.raises(VisualAlignmentRefusal, match="pitch must be finite"):
        require_visual_alignment_entry(
            _target(1),
            measured_pitch_rad=pitch,
        )
    with pytest.raises(VisualAlignmentRefusal, match="exact VisualTarget"):
        require_visual_alignment_entry(
            object(),
            measured_pitch_rad=0.0,
        )


def test_failed_live_trace_geometry_is_refused_before_first_authority():
    first_control_frame = _target(
        181,
        x=0.6468750000000001,
        y=-0.7055555555555555,
        x_rate=0.3440772588847738,
        y_rate=-0.5593719110199848,
        scale_rate=1.036721019290431,
    )
    with pytest.raises(
        VisualAlignmentRefusal,
        match="horizontal motion",
    ):
        require_visual_alignment_entry(
            first_control_frame,
            measured_pitch_rad=-0.05185464301379744,
        )

    first_clipped_frame = _target(
        182,
        x=0.65625,
        y=-0.7277777777777777,
        x_rate=0.344,
        y_rate=-0.701,
        vertical_censored=True,
        scale_rate=1.0593890808936899,
    )
    with pytest.raises(
        VisualAlignmentRefusal,
        match="clipped or censored",
    ):
        require_visual_alignment_entry(
            first_clipped_frame,
            measured_pitch_rad=-0.0514,
        )


@pytest.mark.parametrize(
    "measured_pitch_rad",
    (-0.0472571, -0.0447612),
)
def test_proved_gate0_handoff_pitch_is_admitted_when_geometry_is_safe(
    measured_pitch_rad,
):
    admission = require_visual_alignment_entry(
        _target(
            180,
            x=0.60,
            y=-0.60,
            x_rate=-0.10,
            y_rate=0.10,
            scale_rate=0.50,
        ),
        measured_pitch_rad=measured_pitch_rad,
    )

    assert admission.measured_pitch_rad == measured_pitch_rad


def test_latest_live_entry_audits_every_admission_bound():
    target = _target(
        180,
        x=0.64375,
        y=-0.7333333333333333,
        x_rate=0.349106,
        y_rate=-0.293239,
        vertical_censored=True,
        scale_rate=0.542511,
    )
    measured_pitch_rad = -0.044761

    assert abs(target.normalized_x) <= POST_PROMOTION_ENTRY_MAX_ABS_X_NORM
    assert abs(target.normalized_y_down) > POST_PROMOTION_ENTRY_MAX_ABS_Y_NORM
    assert (
        target.normalized_x_rate_s
        > POST_PROMOTION_ENTRY_MAX_OUTWARD_RATE_NORM_S
    )
    assert (
        -target.normalized_y_rate_down_s
        > POST_PROMOTION_ENTRY_MAX_OUTWARD_RATE_NORM_S
    )
    assert (
        target.log_scale_rate_s
        <= POST_PROMOTION_ENTRY_MAX_LOG_SCALE_RATE_S
    )
    assert (
        measured_pitch_rad
        >= POST_PROMOTION_ENTRY_MIN_MEASURED_PITCH_RAD
    )
    with pytest.raises(
        VisualAlignmentRefusal,
        match="clipped or censored",
    ):
        require_visual_alignment_entry(
            target,
            measured_pitch_rad=measured_pitch_rad,
        )


def test_signed_outward_rate_allows_motion_toward_image_center():
    admission = require_visual_alignment_entry(
        _target(
            1,
            x=0.6,
            y=-0.6,
            x_rate=-1.0,
            y_rate=1.0,
            scale_rate=-1.0,
        ),
        measured_pitch_rad=0.0,
    )

    assert admission.horizontal_outward_rate_s == -1.0
    assert admission.vertical_outward_rate_down_s == -1.0


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
    assert after_gap.horizontal_abs_errors == pytest.approx(
        (0.60, 0.58, 0.56)
    )
    assert after_gap.horizontal_deltas == pytest.approx((-0.02, -0.02))
    assert after_gap.vertical_abs_errors == pytest.approx((0.55,))


def test_vertical_censoring_does_not_erase_horizontal_worsening_evidence():
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
        _target(3, x=0.52, y=-0.90, vertical_censored=True),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )

    assert first_worse.abort_reason is None
    assert gap.abort_reason == "horizontal_error_worsening_uninterrupted"
    assert gap.eligible_joint_frame_count == 0
    assert gap.improving_joint_frame_streak == 0
    assert gap.horizontal_abs_errors == pytest.approx((0.50, 0.51, 0.52))
    assert gap.horizontal_deltas == pytest.approx((0.01, 0.01))
    assert gap.vertical_abs_errors == ()


def test_two_geometry_censored_response_frames_latch_abort():
    monitor = RestrictedAlignmentMonitor(
        track_id="vq2-track-000004",
        required_improving_frames=3,
    )
    grace = monitor.observe(
        _target(1, vertical_censored=True),
        response_evaluation_enabled=False,
        corridor_frames=0,
    )
    first = monitor.observe(
        _target(2, vertical_censored=True),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    second = monitor.observe(
        _target(3, vertical_censored=True),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )

    assert grace.abort_reason is None
    assert first.abort_reason is None
    assert second.abort_reason == "geometry_censored_uninterrupted"
    assert second.eligible_joint_frame_count == 0
    assert second.improving_joint_frame_streak == 0


def test_joint_acceptance_restarts_after_single_axis_censoring():
    monitor = RestrictedAlignmentMonitor(
        track_id="vq2-track-000004",
        required_improving_frames=3,
    )
    monitor.observe(
        _target(1, x=0.60, y=-0.60),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    before_gap = monitor.observe(
        _target(2, x=0.58, y=-0.58),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    gap = monitor.observe(
        _target(3, x=0.56, y=-0.90, vertical_censored=True),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    first = monitor.observe(
        _target(4, x=0.54, y=-0.55),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    second = monitor.observe(
        _target(5, x=0.52, y=-0.53),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )
    third = monitor.observe(
        _target(6, x=0.50, y=-0.51),
        response_evaluation_enabled=True,
        corridor_frames=0,
    )

    assert before_gap.improving_joint_frame_streak == 2
    assert gap.improving_joint_frame_streak == 0
    assert first.improving_joint_frame_streak == 1
    assert second.improving_joint_frame_streak == 2
    assert second.accepted is False
    assert third.improving_joint_frame_streak == 3
    assert third.accepted is True


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
