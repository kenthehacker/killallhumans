from __future__ import annotations

from dataclasses import replace

import pytest

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetVisualTracker,
    VisualDetection,
    VisualDetectionFrame,
    VisualTrack,
    VisualTrackRole,
    VisualTrackSample,
)
from planning.vq2_visual_servo import (
    ImageVisualServo,
    MAX_VISUAL_SEGMENT_DURATION_S,
    MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD,
    MAX_VISUAL_YAW_RATE_RAD_S,
    MIN_VISUAL_THRUST,
    PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S,
    PREPASS_CURRENT_MAX_ABS_X_NORM,
    PREPASS_CURRENT_MAX_ABS_Y_NORM,
    PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S,
    PREPASS_NEXT_MAX_ABS_CENTER_RATE_NORM_S,
    ServoFrameToken,
    VISUAL_SEGMENT_YAW_SOFT_STOP_RAD,
    VisualServoPassageSafetyUnavailable,
    VisualServoRefusal,
    VisualServoTuning,
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
    horizontal_censored: bool = False,
    vertical_censored: bool = False,
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
        horizontal_censored=horizontal_censored,
        vertical_censored=vertical_censored,
        ambiguous=ambiguous,
    )


def tracker_track(
    *,
    token: CameraFrameToken | None = None,
    sample_token: CameraFrameToken | None = None,
    provenance: FrameProvenanceBasis = (
        FrameProvenanceBasis.RECEIVER_TIMING_V1
    ),
    publication_monotonic_ns: int | None = 12_500_100_000,
    center: tuple[float, float] = (0.25, -0.40),
    velocity: tuple[float, float] = (0.10, -0.20),
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
    role: VisualTrackRole = VisualTrackRole.CURRENT,
    authoritative_gate_index: int | None = 1,
    authority_sequence: int | None = 22,
    authority_boot_ms: int | None = 6256,
    ambiguous: bool = False,
    visible: bool = True,
) -> VisualTrack:
    token = token or CameraFrameToken(
        generation=3,
        frame_id=91,
        publication_sequence=17,
        stream_id="camera0",
    )
    sample_token = token if sample_token is None else sample_token
    sample = VisualTrackSample(
        tracker_frame_sequence=5,
        token=sample_token,
        observation_monotonic_ns=12_500_000_000,
        publication_monotonic_ns=publication_monotonic_ns,
        provenance_basis=provenance,
        camera_source_time_ns=123_000_000,
        source_index=0,
        center_norm=center,
        bbox_norm=(0.50, 0.15, 0.70, 0.35),
        apparent_scale=0.2,
        confidence=0.8,
        clipping=clipping,
        center_censored=center_censored,
        association_confidence=0.7,
    )
    return VisualTrack(
        track_id="vq2-track-000001",
        first_token=sample_token,
        latest_token=token,
        center_norm=center,
        bbox_norm=sample.bbox_norm,
        apparent_scale=sample.apparent_scale,
        center_velocity_norm_s=velocity,
        log_scale_rate_s=0.3,
        confidence=sample.confidence,
        association_confidence=sample.association_confidence,
        consecutive_frame_count=5,
        total_observation_count=5,
        missed_frame_count=0,
        clipping=clipping,
        center_censored=center_censored,
        role=role,
        authoritative_gate_index=authoritative_gate_index,
        authority_race_status_sequence=authority_sequence,
        authority_race_status_boot_ms=authority_boot_ms,
        ambiguous=ambiguous,
        visible=visible,
        history=(sample,),
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

    assert servo.tuning.align_thrust == MIN_VISUAL_THRUST
    assert servo.tuning.brake_thrust == MIN_VISUAL_THRUST
    assert servo.tuning.advance_thrust == 0.295
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
    assert low.thrust == pytest.approx(MIN_VISUAL_THRUST)


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
    assert accelerating.thrust == pytest.approx(MIN_VISUAL_THRUST)


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


def _latch_passage_blend(
    servo: ImageVisualServo,
    *,
    requested_blend: float = 0.3,
) -> None:
    for frame_id in range(1, 4):
        output = step(
            servo,
            target(frame_id),
            allow_advance=False,
            allow_passage_safe_next_blend=True,
        )
        assert output.next_gate_blend == 0.0
    output = step(
        servo,
        target(4),
        next_target=target(4, track_id="vq2-track-000002"),
        requested_next_blend=requested_blend,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    assert output.next_gate_blend == pytest.approx(requested_blend)


def test_passage_blend_requires_narrow_start_then_broad_continuation() -> None:
    servo = ImageVisualServo()
    outside_narrow = target(
        1,
        x=0.08,
        y=-0.22,
        x_rate=0.05,
        y_rate=-0.05,
        log_scale=-1.0,
    )
    next_gate = target(
        1,
        track_id="vq2-track-000002",
        x=0.45,
        y=-0.48,
        x_rate=0.05,
        y_rate=-0.05,
        log_scale=-2.0,
    )

    withheld = step(
        servo,
        outside_narrow,
        next_target=next_gate,
        requested_next_blend=0.3,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    assert withheld.next_gate_blend == 0.0

    servo.reset_segment()
    _latch_passage_blend(servo)
    current = target(
        5,
        x=0.08,
        y=-0.22,
        x_rate=0.25,
        y_rate=-0.43,
        log_scale=-1.0,
        scale_rate=1.59,
    )
    next_gate = target(
        5,
        track_id="vq2-track-000002",
        x=0.45,
        y=-0.48,
        x_rate=0.44,
        y_rate=-0.44,
        scale_rate=0.63,
    )
    continued = step(
        servo,
        current,
        next_target=next_gate,
        requested_next_blend=0.3,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )

    assert abs(current.normalized_y_down) > servo.tuning.vertical_corridor
    assert abs(next_gate.normalized_x_rate_s) > (
        servo.tuning.stable_rate_norm_s
    )
    assert continued.next_gate_blend == pytest.approx(0.3)
    assert continued.effective_horizontal_error > current.normalized_x
    assert continued.effective_vertical_error_image_down < (
        current.normalized_y_down
    )
    assert continued.yaw_rate_rad_s < 0.0
    assert continued.target_pitch_rad > 0.0
    assert not continued.advance_enabled


def test_exact_latch_frame_must_itself_be_stable_and_passage_safe() -> None:
    servo = ImageVisualServo()
    for frame_id in range(1, 4):
        step(
            servo,
            target(frame_id),
            allow_advance=False,
            allow_passage_safe_next_blend=True,
        )

    unsafe_latch = step(
        servo,
        target(4, x_rate=8.0, log_scale=-0.30),
        next_target=target(4, track_id="vq2-track-000002"),
        requested_next_blend=0.25,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    assert unsafe_latch.next_gate_blend == 0.0
    assert unsafe_latch.corridor_frames == 0

    # This frame would be admissible only under broad continuation.  Its
    # refusal proves the unsafe publication did not silently create a latch.
    no_hidden_latch = step(
        servo,
        target(5, y=-0.22),
        next_target=target(
            5,
            track_id="vq2-track-000002",
            x_rate=0.44,
        ),
        requested_next_blend=0.25,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    assert no_hidden_latch.next_gate_blend == 0.0


@pytest.mark.parametrize(
    "current",
    (
        target(
            5,
            x=PREPASS_CURRENT_MAX_ABS_X_NORM - 0.01,
            x_rate=0.20,
            log_scale=-1.0,
        ),
        target(
            5,
            y=-(PREPASS_CURRENT_MAX_ABS_Y_NORM - 0.01),
            y_rate=-0.20,
            log_scale=-1.0,
        ),
        target(
            5,
            x_rate=(
                PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S + 1e-4
            ),
        ),
        target(
            5,
            scale_rate=PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S + 1e-4,
        ),
        target(5, log_scale=-0.50),
    ),
    ids=(
        "projected-horizontal-edge",
        "projected-vertical-edge",
        "center-rate",
        "scale-rate",
        "close-scale",
    ),
)
def test_latched_passage_corridor_retires_unsafe_current(
    current: VisualTarget,
) -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    with pytest.raises(
        VisualServoPassageSafetyUnavailable,
        match="left its passage corridor",
    ):
        step(
            servo,
            current,
            next_target=target(
                5,
                track_id="vq2-track-000002",
            ),
            requested_next_blend=0.3,
            allow_advance=False,
            allow_passage_safe_next_blend=True,
        )


def test_latched_passage_never_reuses_stale_next_geometry() -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    missing = step(
        servo,
        target(5),
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    assert missing.next_gate_blend == 0.0

    recovered = step(
        servo,
        target(6),
        next_target=target(
            6,
            track_id="vq2-track-000002",
            x_rate=PREPASS_NEXT_MAX_ABS_CENTER_RATE_NORM_S,
        ),
        requested_next_blend=0.3,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    assert recovered.next_gate_blend == pytest.approx(0.3)

    too_fast = step(
        servo,
        target(7),
        next_target=target(
            7,
            track_id="vq2-track-000002",
            x_rate=PREPASS_NEXT_MAX_ABS_CENTER_RATE_NORM_S + 1e-4,
        ),
        requested_next_blend=0.3,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    assert too_fast.next_gate_blend == 0.0


def test_passage_blend_cannot_reverse_current_aperture_correction() -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    with pytest.raises(
        VisualServoPassageSafetyUnavailable,
        match="reversed current-aperture correction",
    ):
        step(
            servo,
            target(5, x=-0.17),
            next_target=target(
                5,
                track_id="vq2-track-000002",
                x=1.0,
            ),
            requested_next_blend=0.3,
            allow_advance=False,
            allow_passage_safe_next_blend=True,
        )


def test_passage_safe_next_blend_cannot_enable_advance_authority() -> None:
    servo = ImageVisualServo()

    with pytest.raises(VisualServoRefusal, match="cannot coexist"):
        step(
            servo,
            target(1),
            next_target=target(1, track_id="vq2-track-000002"),
            requested_next_blend=0.3,
            allow_advance=True,
            allow_passage_safe_next_blend=True,
        )


@pytest.mark.parametrize(
    "tuning",
    (
        VisualServoTuning(
            horizontal_corridor=PREPASS_CURRENT_MAX_ABS_X_NORM,
        ),
        VisualServoTuning(
            vertical_corridor=PREPASS_CURRENT_MAX_ABS_Y_NORM,
        ),
    ),
)
def test_configured_start_corridor_must_stay_inside_passage_bounds(
    tuning: VisualServoTuning,
) -> None:
    servo = ImageVisualServo(tuning)

    with pytest.raises(VisualServoRefusal, match="inside passage bounds"):
        step(
            servo,
            target(1),
            allow_advance=False,
            allow_passage_safe_next_blend=True,
        )


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


def test_default_gain_uses_more_heading_authority_on_latest_live_prepass_frame():
    """Regress the last safe frame from the 20260724T222751Z handoff."""

    servo = ImageVisualServo()
    for frame, y in ((1, -0.1500), (2, -0.1556), (3, -0.1667)):
        step(
            servo,
            target(
                frame,
                x=-0.003125,
                y=y,
                x_rate=-0.004596953355714091,
                y_rate=-0.2952764132559311,
                log_scale=-1.8050598101266444,
                scale_rate=-0.01868892118010406,
                consecutive=109 + frame,
            ),
            next_target=target(
                frame,
                track_id="vq2-track-000002",
                x=0.328125,
                y=-0.2777777777777778,
                x_rate=0.06166095463922315,
                y_rate=-0.2953488580256311,
                log_scale=-2.5112116249241483,
                scale_rate=0.3596903616559018,
                consecutive=109 + frame,
            ),
            requested_next_blend=0.25,
            allow_advance=False,
            allow_passage_safe_next_blend=True,
        )

    output = step(
        servo,
        target(
            4,
            x=-0.003125,
            y=-0.1777777777777778,
            x_rate=-0.004596953355714091,
            y_rate=-0.2952764132559311,
            log_scale=-1.8050598101266444,
            scale_rate=-0.01868892118010406,
            consecutive=112,
        ),
        next_target=target(
            4,
            track_id="vq2-track-000002",
            x=0.328125,
            y=-0.2777777777777778,
            x_rate=0.06166095463922315,
            y_rate=-0.2953488580256311,
            log_scale=-2.5112116249241483,
            scale_rate=0.3596903616559018,
            consecutive=112,
        ),
        requested_next_blend=0.25,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )

    assert output.next_gate_blend == pytest.approx(0.25)
    assert output.effective_horizontal_error == pytest.approx(0.0796875)
    assert output.yaw_rate_rad_s == pytest.approx(-0.024325113327505703)
    assert output.target_pitch_rad >= 0.0
    assert output.advance_enabled is False


def test_visual_track_adapter_preserves_image_down_axis_explicitly():
    adapted = VisualTarget.from_visual_track(
        tracker_track(),
        expected_gate_index=1,
    )
    assert adapted.normalized_y_down == pytest.approx(-0.40)
    assert adapted.normalized_y_rate_down_s == pytest.approx(-0.20)
    assert adapted.frame_token.stream_id == "camera0"
    assert adapted.frame_token.publication_sequence == 17
    assert adapted.received_monotonic_s == pytest.approx(12.5)


def test_visual_track_adapter_accepts_real_multiframe_smoothed_track():
    tracker = MultiTargetVisualTracker()

    def live_frame(
        sequence: int,
        *,
        center_x: float,
        confidence: float,
    ) -> VisualDetectionFrame:
        center_unit_x = 0.5 * (center_x + 1.0)
        center_unit_y = 0.30
        final_packet_ns = 12_000_000_000 + sequence * 33_000_000
        return VisualDetectionFrame(
            token=CameraFrameToken(
                generation=3,
                frame_id=90 + sequence,
                publication_sequence=sequence,
                stream_id="camera0",
            ),
            provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
            time_basis_id="vq2-host-monotonic",
            image_size_px=(640, 360),
            detections=(
                VisualDetection(
                    source_index=0,
                    center_norm=(center_x, -0.40),
                    bbox_norm=(
                        center_unit_x - 0.10,
                        center_unit_y - 0.10,
                        center_unit_x + 0.10,
                        center_unit_y + 0.10,
                    ),
                    confidence=confidence,
                ),
            ),
            camera_source_time_ns=5_000_000_000 + sequence * 33_000_000,
            final_unique_packet_monotonic_ns=final_packet_ns,
            publish_monotonic_ns=final_packet_ns + 100_000,
        )

    first = tracker.update(live_frame(1, center_x=0.24, confidence=0.20))
    track_id = first.visible_track_ids[0]
    tracker.update(live_frame(2, center_x=0.25, confidence=0.90))
    tracker.assign_role(track_id, VisualTrackRole.CURRENT)
    tracker.confirm_authoritative_gate(
        track_id,
        gate_index=1,
        race_status_sequence=22,
        race_status_boot_ms=6256,
    )
    track = tracker.track(track_id)

    # The public track confidence is intentionally smoothed; the latest
    # sample retains raw detector confidence.  Both are valid, but they are
    # not required to be equal.
    assert track.confidence != track.history[-1].confidence
    adapted = VisualTarget.from_visual_track(
        track,
        expected_gate_index=1,
    )
    assert adapted.confidence == pytest.approx(track.confidence)
    assert adapted.normalized_x == pytest.approx(0.25)
    assert adapted.frame_token.publication_sequence == 2


def test_visual_track_adapter_rejects_duck_typed_and_incoherent_history():
    with pytest.raises(VisualServoRefusal, match="exact VisualTrack"):
        VisualTarget.from_visual_track(object())  # type: ignore[arg-type]

    other_token = CameraFrameToken(
        generation=3,
        frame_id=92,
        publication_sequence=18,
        stream_id="camera0",
    )
    incoherent = tracker_track(sample_token=other_token)
    with pytest.raises(VisualServoRefusal, match="latest token disagrees"):
        VisualTarget.from_visual_track(incoherent)

    stale_fields = replace(tracker_track(), center_norm=(0.1, -0.4))
    with pytest.raises(VisualServoRefusal, match="latest fields disagree"):
        VisualTarget.from_visual_track(stale_fields)


@pytest.mark.parametrize(
    ("track", "message"),
    [
        (
            tracker_track(
                provenance=FrameProvenanceBasis.LEGACY_CAPTURE,
                publication_monotonic_ns=None,
            ),
            "receiver timing provenance",
        ),
        (
            tracker_track(publication_monotonic_ns=None),
            "coherent receiver times",
        ),
        (
            tracker_track(
                token=CameraFrameToken(generation=3, frame_id=91),
            ),
            "publication provenance",
        ),
    ],
)
def test_visual_track_adapter_requires_live_receiver_provenance(
    track, message
):
    with pytest.raises(VisualServoRefusal, match=message):
        VisualTarget.from_visual_track(track)


def test_visual_track_adapter_requires_visible_race_labelled_current_authority():
    with pytest.raises(VisualServoRefusal, match="currently visible"):
        VisualTarget.from_visual_track(
            tracker_track(visible=False),
        )
    with pytest.raises(VisualServoRefusal, match="CURRENT"):
        VisualTarget.from_visual_track(
            tracker_track(
                role=VisualTrackRole.NEXT,
                authoritative_gate_index=None,
                authority_sequence=None,
                authority_boot_ms=None,
            )
        )
    with pytest.raises(VisualServoRefusal, match="gate index"):
        VisualTarget.from_visual_track(
            tracker_track(authoritative_gate_index=None),
        )
    with pytest.raises(VisualServoRefusal, match="does not match"):
        VisualTarget.from_visual_track(
            tracker_track(),
            expected_gate_index=2,
        )

    next_candidate = VisualTarget.from_visual_track(
        tracker_track(
            role=VisualTrackRole.NEXT,
            authoritative_gate_index=None,
            authority_sequence=None,
            authority_boot_ms=None,
        ),
        require_current_authority=False,
    )
    assert next_candidate.track_id == "vq2-track-000001"


def test_failed_top_clipped_geometry_brakes_at_minimum_collective():
    servo = ImageVisualServo()
    output = step(
        servo,
        target(
            1,
            x=0.65625,
            y=-0.7277777777777779,
            x_rate=0.345412,
            y_rate=-0.607664,
            log_scale=-1.530552,
            scale_rate=1.0593890808936899,
            clipped=True,
            center_censored=True,
            vertical_censored=True,
        ),
    )

    assert output.advance_enabled is False
    assert output.brake_reason == "target_edge_or_clipping"
    assert output.yaw_rate_rad_s < 0.0
    assert output.effective_horizontal_error == pytest.approx(0.65625)
    assert output.effective_vertical_error_image_down == 0.0
    assert output.effective_vertical_rate_down_s == 0.0
    assert output.target_pitch_rad >= 0.0
    assert output.target_pitch_rad == pytest.approx(
        servo.tuning.brake_pitch_rad
    )
    assert output.thrust == pytest.approx(MIN_VISUAL_THRUST)


def test_left_clipping_suppresses_only_horizontal_correction_and_brakes():
    servo = ImageVisualServo()
    output = step(
        servo,
        target(
            1,
            x=-0.85,
            x_rate=-0.4,
            y=-0.45,
            clipped=True,
            center_censored=True,
            horizontal_censored=True,
        ),
    )

    assert output.advance_enabled is False
    assert output.brake_reason == "target_edge_or_clipping"
    assert output.yaw_rate_rad_s == 0.0
    assert output.effective_horizontal_error == 0.0
    assert output.effective_horizontal_rate_s == 0.0
    assert output.effective_vertical_error_image_down == pytest.approx(-0.45)
    assert output.target_pitch_rad > servo.tuning.brake_pitch_rad
    assert output.thrust > servo.tuning.brake_thrust


def test_axis_clipping_is_derived_from_exact_tracker_edge_flags():
    top = VisualTarget.from_visual_track(
        tracker_track(
            clipping=FrameEdge.TOP,
            center_censored=True,
        )
    )
    assert top.vertical_censored is True
    assert top.horizontal_censored is False

    left = VisualTarget.from_visual_track(
        tracker_track(
            clipping=FrameEdge.LEFT,
            center_censored=True,
        )
    )
    assert left.horizontal_censored is True
    assert left.vertical_censored is False


def test_top_clipped_next_gate_can_blend_heading_but_brakes_closure():
    servo = ImageVisualServo()
    for frame in (1, 2, 3):
        step(servo, target(frame, x=0.01, y=0.01))
    current = target(4, x=0.01, y=0.01)
    next_gate = target(
        4,
        track_id="vq2-track-000002",
        x=0.62,
        y=-0.92,
        clipped=True,
        center_censored=True,
        vertical_censored=True,
    )
    output = step(
        servo,
        current,
        next_target=next_gate,
        requested_next_blend=0.3,
    )

    assert output.next_gate_blend == pytest.approx(0.3)
    assert output.next_horizontal_error == pytest.approx(0.62)
    assert output.next_vertical_error_image_down is None
    assert output.yaw_rate_rad_s < 0.0
    assert output.advance_enabled is False
    assert output.brake_reason == "next_target_edge_or_clipping"


def test_ambiguous_next_gate_brakes_forward_closure_without_blending():
    servo = ImageVisualServo()
    for frame in (1, 2, 3):
        step(servo, target(frame, x=0.01, y=0.01))
    output = step(
        servo,
        target(4, x=0.01, y=0.01),
        next_target=target(
            4,
            track_id="vq2-track-000002",
            x=0.6,
            ambiguous=True,
        ),
        requested_next_blend=0.3,
    )

    assert output.next_gate_blend == 0.0
    assert output.advance_enabled is False
    assert output.brake_reason == "next_target_ambiguous"
    assert output.target_pitch_rad > 0.0
