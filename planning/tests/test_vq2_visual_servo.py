from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from planning.vq2_visual_servo import (
    ImageVisualServo,
    MAX_VISUAL_SEGMENT_DURATION_S,
    MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD,
    MAX_VISUAL_YAW_RATE_RAD_S,
    ServoFrameToken,
    VISUAL_SEGMENT_YAW_SOFT_STOP_RAD,
    VisualServoRefusal,
    VisualTarget,
)


def target(
    frame_id: int,
    *,
    track_id: str = "vq2-track-000001",
    received: float | None = None,
    x: float = 0.0,
    y: float = 0.0,
    x_rate: float = 0.0,
    y_rate: float = 0.0,
    log_scale: float = -2.0,
    scale_rate: float = 0.0,
    consecutive: int = 4,
    clipped: bool = False,
    center_censored: bool = False,
    ambiguous: bool = False,
    publication_sequence: int | None = None,
) -> VisualTarget:
    return VisualTarget(
        track_id=track_id,
        frame_token=ServoFrameToken(
            "vq2-camera-udp-5600",
            7,
            frame_id,
            frame_id if publication_sequence is None else publication_sequence,
        ),
        received_monotonic_s=(
            10.0 + frame_id * 0.01 if received is None else received
        ),
        normalized_x=x,
        normalized_y_down=y,
        normalized_x_rate_s=x_rate,
        normalized_y_rate_down_s=y_rate,
        log_scale=log_scale,
        log_scale_rate_s=scale_rate,
        confidence=0.8,
        association_confidence=0.8,
        consecutive_frames=consecutive,
        clipped=clipped,
        center_censored=center_censored,
        ambiguous=ambiguous,
    )


def step(
    servo: ImageVisualServo,
    observation: VisualTarget,
    **kwargs,
):
    return servo.step(
        observation,
        now_monotonic_s=observation.received_monotonic_s + 0.01,
        segment_elapsed_s=kwargs.pop("segment_elapsed_s", 0.5),
        segment_yaw_excursion_rad=kwargs.pop(
            "segment_yaw_excursion_rad", 0.0
        ),
        **kwargs,
    )


def test_aligns_before_enabling_forward_closure():
    servo = ImageVisualServo()
    first = step(servo, target(1, x=0.42, y=-0.36))

    assert first.advance_enabled is False
    assert first.brake_reason == "aligning"
    assert first.thrust > servo.tuning.align_thrust
    assert first.target_pitch_rad > 0.0
    assert first.yaw_rate_rad_s < 0.0

    outputs = [
        step(servo, target(frame, x=0.03, y=-0.04))
        for frame in (2, 3, 4)
    ]
    assert [item.corridor_frames for item in outputs] == [1, 2, 3]
    assert outputs[-1].advance_enabled is True
    assert outputs[-1].target_pitch_rad < 0.0
    assert outputs[-1].thrust > servo.tuning.advance_thrust


def test_horizontal_bearing_uses_calibrated_yaw_image_sign():
    servo = ImageVisualServo()
    right = step(servo, target(1, x=0.7))
    servo.reset_segment()
    left = step(servo, target(2, x=-0.7))

    assert right.yaw_rate_rad_s == pytest.approx(
        -MAX_VISUAL_YAW_RATE_RAD_S
    )
    assert left.yaw_rate_rad_s == pytest.approx(
        MAX_VISUAL_YAW_RATE_RAD_S
    )
    assert right.target_roll_rad == 0.0
    assert left.target_roll_rad == 0.0


def test_vertical_image_error_drives_pitch_in_both_directions():
    servo = ImageVisualServo()
    high = step(servo, target(1, y=-0.5))
    servo.reset_segment()
    low = step(servo, target(2, y=0.5))

    assert high.target_pitch_rad > 0.0
    assert low.target_pitch_rad == 0.0
    assert high.thrust > servo.tuning.align_thrust
    assert low.thrust < servo.tuning.align_thrust


def test_scale_rate_brakes_even_after_corridor_dwell():
    servo = ImageVisualServo()
    step(servo, target(1, x=0.02, y=0.01))
    step(servo, target(2, x=0.02, y=0.01))
    accelerating = step(
        servo,
        target(3, x=0.02, y=0.01, scale_rate=2.2),
    )

    assert accelerating.advance_enabled is False
    assert accelerating.brake_reason == "scale_rate"
    assert accelerating.target_pitch_rad > 0.0
    assert accelerating.thrust < servo.tuning.brake_thrust


def test_target_edge_brakes_instead_of_continuing_fixed_thrust():
    servo = ImageVisualServo()
    at_edge = step(servo, target(1, x=0.78))

    assert at_edge.advance_enabled is False
    assert at_edge.brake_reason == "target_edge_or_clipping"
    assert at_edge.target_pitch_rad > 0.0
    assert at_edge.thrust == pytest.approx(servo.tuning.brake_thrust)


def test_clipped_target_brakes_even_when_visible_center_is_near_center():
    servo = ImageVisualServo()
    output = step(servo, target(1, x=0.02, y=0.01, clipped=True))
    assert output.brake_reason == "target_edge_or_clipping"
    assert output.advance_enabled is False


def test_increasing_alignment_error_brakes_and_resets_corridor():
    servo = ImageVisualServo()
    step(servo, target(1, x=0.08, y=0.07))
    output = step(servo, target(2, x=0.13, y=0.12))

    assert output.horizontal_abs_error_delta == pytest.approx(0.05)
    assert output.vertical_abs_error_delta == pytest.approx(0.05)
    assert output.brake_reason == "alignment_error_increasing"
    assert output.corridor_frames == 0


def test_next_gate_blend_requires_current_corridor_and_same_fresh_frame():
    servo = ImageVisualServo()
    current = target(1, x=0.45, y=0.0)
    next_gate = target(
        1,
        track_id="vq2-track-000002",
        x=-0.8,
        y=-0.4,
    )
    outside = step(
        servo,
        current,
        next_target=next_gate,
        requested_next_blend=0.3,
    )
    assert outside.next_gate_blend == 0.0

    servo.reset_segment()
    # The blend cannot begin until the current aperture itself has completed
    # its fresh-frame corridor dwell.
    step(servo, target(2, x=0.02, y=0.01))
    step(servo, target(3, x=0.02, y=0.01))
    step(servo, target(4, x=0.02, y=0.01))
    current = target(5, x=0.02, y=0.01)
    next_gate = target(
        5,
        track_id="vq2-track-000002",
        x=0.62,
        y=-0.5,
    )
    blended = step(
        servo,
        current,
        next_target=next_gate,
        requested_next_blend=0.3,
    )
    assert blended.next_gate_blend == pytest.approx(0.3)
    assert blended.yaw_rate_rad_s < 0.0

    servo.reset_segment()
    stale_token = replace(
        next_gate,
        frame_token=ServoFrameToken(
            "vq2-camera-udp-5600",
            7,
            1,
            1,
        ),
    )
    mismatched = step(
        servo,
        current,
        next_target=stale_token,
        requested_next_blend=0.3,
    )
    assert mismatched.next_gate_blend == 0.0


def test_ambiguous_current_track_refuses_authority():
    servo = ImageVisualServo()
    with pytest.raises(VisualServoRefusal, match="lacks authority"):
        step(servo, target(1, ambiguous=True))


def test_stale_duplicate_and_backward_publications_are_rejected():
    servo = ImageVisualServo()
    observation = target(1, received=10.0)
    with pytest.raises(VisualServoRefusal, match="stale"):
        servo.step(
            observation,
            now_monotonic_s=10.2,
            segment_elapsed_s=0.2,
            segment_yaw_excursion_rad=0.0,
        )
    step(servo, observation)
    with pytest.raises(VisualServoRefusal, match="did not advance"):
        step(servo, observation)
    backward = target(
        2,
        received=10.02,
        publication_sequence=0 + 1,
    )
    with pytest.raises(VisualServoRefusal, match="did not advance"):
        step(servo, backward)


def test_yaw_soft_stop_does_not_reuse_calibration_excursion():
    assert VISUAL_SEGMENT_YAW_SOFT_STOP_RAD > 0.05
    assert MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD > (
        VISUAL_SEGMENT_YAW_SOFT_STOP_RAD
    )
    servo = ImageVisualServo()
    output = step(
        servo,
        target(1, x=-0.5),
        segment_yaw_excursion_rad=VISUAL_SEGMENT_YAW_SOFT_STOP_RAD,
    )

    assert output.yaw_envelope_limited is True
    assert output.yaw_rate_rad_s == 0.0
    assert output.advance_enabled is False
    assert output.brake_reason == "segment_yaw_outward_soft_stop"


def test_hard_yaw_and_duration_envelopes_refuse_authority():
    servo = ImageVisualServo()
    with pytest.raises(VisualServoRefusal, match="yaw envelope"):
        step(
            servo,
            target(1),
            segment_yaw_excursion_rad=(
                MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD + 1e-4
            ),
        )
    servo.reset_segment()
    with pytest.raises(VisualServoRefusal, match="duration"):
        step(
            servo,
            target(2),
            segment_elapsed_s=MAX_VISUAL_SEGMENT_DURATION_S + 1e-4,
        )


def test_all_outputs_remain_inside_immutable_controller_envelope():
    servo = ImageVisualServo()
    output = step(
        servo,
        target(
            1,
            x=1.0,
            y=-1.0,
            x_rate=3.0,
            y_rate=-3.0,
            scale_rate=2.5,
        ),
    )
    assert abs(output.yaw_rate_rad_s) <= MAX_VISUAL_YAW_RATE_RAD_S
    assert abs(output.target_roll_rad) <= 0.12
    assert -0.30 <= output.target_pitch_rad <= 0.10
    assert 0.21 <= output.thrust <= 0.32


def test_nonadvance_and_every_brake_cause_forbid_nose_down_pitch():
    cases = (
        target(1, x=0.5, y=0.5),
        target(2, x=0.8, y=0.5),
        target(3, x=0.02, y=0.5, scale_rate=2.2),
        target(4, x=0.02, y=0.5, scale_rate=-2.2),
        target(5, x=0.02, y=0.5, log_scale=-0.1),
    )
    for observation in cases:
        servo = ImageVisualServo()
        output = step(servo, observation)
        assert output.advance_enabled is False
        assert output.target_pitch_rad >= 0.0


def test_corridor_dwell_is_bound_to_track_identity():
    servo = ImageVisualServo()
    step(servo, target(1, x=0.01, y=0.01))
    step(servo, target(2, x=0.01, y=0.01))
    with pytest.raises(VisualServoRefusal, match="track identity"):
        step(
            servo,
            target(
                3,
                track_id="vq2-track-000002",
                x=0.01,
                y=0.01,
            ),
        )


@pytest.mark.parametrize(
    ("scale", "scale_rate", "reason"),
    [
        (-2.0, -11.0, "rapid_scale_retreat"),
        (-0.10, 0.0, "close_scale"),
    ],
)
def test_retreating_or_close_target_cannot_enable_advance(
    scale, scale_rate, reason
):
    servo = ImageVisualServo()
    for frame in (1, 2, 3):
        output = step(
            servo,
            target(
                frame,
                x=0.01,
                y=0.01,
                log_scale=scale,
                scale_rate=scale_rate,
            ),
        )
    assert output.advance_enabled is False
    assert output.brake_reason == reason


def test_next_gate_blend_rejects_clipped_or_unstable_candidate():
    servo = ImageVisualServo()
    for frame in (1, 2, 3):
        step(servo, target(frame, x=0.01, y=0.01))
    current = target(4, x=0.01, y=0.01)
    clipped_next = target(
        4,
        track_id="vq2-track-000002",
        x=0.6,
        y=-0.5,
        clipped=True,
        center_censored=True,
    )
    clipped = step(
        servo,
        current,
        next_target=clipped_next,
        requested_next_blend=0.3,
    )
    assert clipped.next_gate_blend == 0.0

    servo.reset_segment()
    for frame in (5, 6, 7):
        step(servo, target(frame, x=0.01, y=0.01))
    current = target(8, x=0.01, y=0.01)
    unstable_next = target(
        8,
        track_id="vq2-track-000002",
        x=0.6,
        y=-0.5,
        x_rate=0.8,
    )
    unstable = step(
        servo,
        current,
        next_target=unstable_next,
        requested_next_blend=0.3,
    )
    assert unstable.next_gate_blend == 0.0


def test_yaw_soft_stop_allows_corrective_return_command():
    servo = ImageVisualServo()
    # Positive excursion with a gate left of center asks for positive yaw,
    # which would increase the excursion and must be blocked.
    outward = step(
        servo,
        target(1, x=-0.5),
        segment_yaw_excursion_rad=VISUAL_SEGMENT_YAW_SOFT_STOP_RAD,
    )
    assert outward.yaw_envelope_limited is True
    assert outward.yaw_rate_rad_s == 0.0

    servo.reset_segment()
    # The opposite image error asks for negative yaw back toward reference.
    inward = step(
        servo,
        target(2, x=0.5),
        segment_yaw_excursion_rad=VISUAL_SEGMENT_YAW_SOFT_STOP_RAD,
    )
    assert inward.yaw_envelope_limited is False
    assert inward.yaw_rate_rad_s < 0.0


def test_output_distinguishes_raw_next_and_effective_blended_errors():
    servo = ImageVisualServo()
    for frame in (1, 2, 3):
        step(servo, target(frame, x=0.02, y=0.01))
    current = target(4, x=0.02, y=0.01)
    next_gate = target(
        4,
        track_id="vq2-track-000002",
        x=0.62,
        y=-0.4,
    )
    output = step(
        servo,
        current,
        next_target=next_gate,
        requested_next_blend=0.25,
    )
    assert output.horizontal_error == pytest.approx(0.02)
    assert output.next_horizontal_error == pytest.approx(0.62)
    assert output.effective_horizontal_error > output.horizontal_error


def test_visual_track_adapter_preserves_image_down_axis_explicitly():
    track = SimpleNamespace(
        track_id="vq2-track-000001",
        latest_token=SimpleNamespace(
            generation=3,
            frame_id=91,
            publication_sequence=17,
        ),
        history=(
            SimpleNamespace(observation_monotonic_ns=12_500_000_000),
        ),
        center_norm=(0.25, -0.40),
        center_velocity_norm_s=(0.10, -0.20),
        apparent_scale=0.2,
        log_scale_rate_s=0.3,
        confidence=0.8,
        association_confidence=0.7,
        consecutive_frame_count=5,
        clipping=0,
        center_censored=False,
        ambiguous=False,
    )
    adapted = VisualTarget.from_visual_track(track, stream_id="camera0")
    assert adapted.normalized_y_down == pytest.approx(-0.40)
    assert adapted.normalized_y_rate_down_s == pytest.approx(-0.20)
    assert adapted.frame_token.publication_sequence == 17
