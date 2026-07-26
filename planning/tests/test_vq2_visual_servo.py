from __future__ import annotations

from dataclasses import replace
import math

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
    MAX_TRANSIENT_PROJECTED_VERTICAL_EXCESS_NORM,
    MAX_VISUAL_SEGMENT_DURATION_S,
    MAX_VISUAL_SEGMENT_YAW_EXCURSION_RAD,
    MAX_VISUAL_TARGET_PITCH_RAD,
    MAX_VISUAL_TARGET_ROLL_RAD,
    MAX_VISUAL_YAW_RATE_RAD_S,
    MIN_VISUAL_THRUST,
    PREPASS_CURRENT_MAX_APPARENT_SCALE,
    PREPASS_CURRENT_MAX_ABS_CENTER_RATE_NORM_S,
    PREPASS_CURRENT_MAX_ABS_X_NORM,
    PREPASS_CURRENT_MAX_ABS_Y_NORM,
    PREPASS_CURRENT_MAX_LOG_SCALE_RATE_S,
    PREPASS_NEXT_MAX_ABS_CENTER_RATE_NORM_S,
    PassageSafetyViolation,
    ServoFrameToken,
    VISUAL_SEGMENT_YAW_SOFT_STOP_RAD,
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

    assert servo.tuning.align_thrust == 0.275
    assert servo.tuning.brake_thrust == 0.275
    assert servo.tuning.advance_thrust == 0.295
    assert first.advance_enabled is False
    assert first.brake_reason == "aligning"
    assert first.thrust == servo.tuning.brake_thrust
    assert first.target_pitch_rad == MAX_VISUAL_TARGET_PITCH_RAD
    assert first.yaw_rate_rad_s < 0.0

    outputs = [
        step(servo, target(frame, x=0.03, y=-0.04))
        for frame in (2, 3, 4)
    ]
    assert [item.corridor_frames for item in outputs] == [1, 2, 3]
    assert outputs[-1].advance_enabled is True
    assert outputs[-1].target_pitch_rad < 0.0
    assert (
        servo.tuning.brake_thrust
        < outputs[-1].thrust
        < 0.32
    )


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
    assert right.target_roll_rad == pytest.approx(
        MAX_VISUAL_TARGET_ROLL_RAD
    )
    assert left.target_roll_rad == pytest.approx(
        -MAX_VISUAL_TARGET_ROLL_RAD
    )
    assert right.target_pitch_rad == MAX_VISUAL_TARGET_PITCH_RAD
    assert left.target_pitch_rad == MAX_VISUAL_TARGET_PITCH_RAD


def test_passage_continuously_reduces_closure_as_expansion_increases():
    servo = ImageVisualServo()
    for frame in (1, 2, 3):
        step(
            servo,
            target(frame, x=0.01, y=0.01),
            allow_advance=False,
        )

    stable = step(
        servo,
        target(4, x=0.01, y=0.01, scale_rate=0.8),
        allow_advance=True,
    )
    expanding = step(
        servo,
        target(5, x=0.01, y=0.01, scale_rate=1.7),
        allow_advance=True,
    )

    assert stable.advance_enabled is True
    assert expanding.advance_enabled is True
    assert expanding.corridor_frames == stable.corridor_frames
    assert stable.target_pitch_rad < expanding.target_pitch_rad
    assert stable.thrust > expanding.thrust


def test_worsening_error_brakes_closure_but_retains_coordinated_steering():
    servo = ImageVisualServo()
    for frame in (1, 2, 3):
        step(
            servo,
            target(frame, x=0.01, y=0.01),
            allow_advance=False,
        )

    output = step(
        servo,
        target(4, x=0.08, y=0.01, x_rate=0.20),
        allow_advance=True,
    )

    assert output.advance_enabled is False
    assert output.brake_reason == "alignment_error_increasing"
    assert output.target_pitch_rad > 0.0
    assert output.yaw_rate_rad_s < 0.0
    assert output.target_roll_rad > 0.0


def test_airborne_alignment_retains_measured_flight_support_collective():
    servo = ImageVisualServo()

    output = step(
        servo,
        target(1, y=-0.08, y_rate=-0.30),
        allow_advance=False,
    )

    assert output.advance_enabled is False
    assert output.target_pitch_rad > 0.0
    assert 0.29 < output.thrust <= 0.32


def test_vertical_image_error_drives_pitch_in_both_directions():
    servo = ImageVisualServo()
    high = step(servo, target(1, y=-0.5))
    servo.reset_segment()
    low = step(servo, target(2, y=0.5))

    assert high.target_pitch_rad > 0.0
    assert low.target_pitch_rad == 0.0
    assert high.thrust > servo.tuning.align_thrust
    assert MIN_VISUAL_THRUST < low.thrust < servo.tuning.align_thrust


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
    assert accelerating.thrust == pytest.approx(
        servo.tuning.brake_thrust - 0.0006
    )


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
    assert blended.yaw_rate_rad_s == -MAX_VISUAL_YAW_RATE_RAD_S
    assert 0.0 < blended.target_roll_rad <= MAX_VISUAL_TARGET_ROLL_RAD
    assert blended.target_pitch_rad == MAX_VISUAL_TARGET_PITCH_RAD
    assert blended.thrust < servo.tuning.advance_thrust

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


def _assert_approach_preview_withheld_current_only(
    output,
    *,
    current: VisualTarget,
    servo: ImageVisualServo,
) -> None:
    assert output.next_gate_blend == 0.0
    assert output.next_horizontal_error is None
    assert output.next_vertical_error_image_down is None
    assert output.reviewed_next_track_id is None
    assert not output.advance_enabled
    assert not output.passage_preview_retired
    assert output.passage_preview_retirement_violations == ()
    assert output.effective_horizontal_error == pytest.approx(
        current.normalized_x
    )
    assert output.effective_vertical_error_image_down == pytest.approx(
        current.normalized_y_down
    )
    assert output.effective_horizontal_rate_s == pytest.approx(
        current.normalized_x_rate_s
    )
    assert output.effective_vertical_rate_down_s == pytest.approx(
        current.normalized_y_rate_down_s
    )
    assert all(
        math.isfinite(value)
        for value in (
            output.target_roll_rad,
            output.target_pitch_rad,
            output.yaw_rate_rad_s,
            output.thrust,
        )
    )
    assert abs(output.yaw_rate_rad_s) <= MAX_VISUAL_YAW_RATE_RAD_S
    assert output.thrust >= MIN_VISUAL_THRUST
    assert servo.latched_next_track_id == "vq2-track-000002"


def _assert_approach_preview_reenters(
    servo: ImageVisualServo,
    *,
    frame_id: int,
    requested_blend: float = 0.3,
) -> None:
    reentered = step(
        servo,
        target(frame_id),
        next_target=target(frame_id, track_id="vq2-track-000002"),
        requested_next_blend=requested_blend,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    assert reentered.next_gate_blend == pytest.approx(requested_blend)
    assert reentered.next_horizontal_error == 0.0
    assert reentered.next_vertical_error_image_down == 0.0
    assert not reentered.advance_enabled
    assert not reentered.passage_preview_retired
    assert servo.latched_next_track_id == "vq2-track-000002"


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
    assert continued.next_gate_blend == pytest.approx(
        0.3 * (2.0 - 1.59) / (2.0 - 1.10)
    )
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
            y=-(PREPASS_CURRENT_MAX_ABS_Y_NORM - 0.011),
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
def test_approach_passage_envelope_violation_withholds_optional_preview(
    current: VisualTarget,
) -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    withheld = step(
        servo,
        current,
        next_target=target(5, track_id="vq2-track-000002"),
        requested_next_blend=0.3,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    _assert_approach_preview_withheld_current_only(
        withheld,
        current=current,
        servo=servo,
    )
    _assert_approach_preview_reenters(servo, frame_id=6)


def test_large_projected_vertical_excursion_only_withholds_approach_preview() -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)
    vertical_rate = -(
        PREPASS_CURRENT_MAX_ABS_Y_NORM
        + MAX_TRANSIENT_PROJECTED_VERTICAL_EXCESS_NORM
        + 0.0001
        - 0.27
    ) / 0.10

    current = target(5, y=-0.27, y_rate=vertical_rate)
    withheld = step(
        servo,
        current,
        next_target=target(5, track_id="vq2-track-000002"),
        requested_next_blend=0.3,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    _assert_approach_preview_withheld_current_only(
        withheld,
        current=current,
        servo=servo,
    )
    _assert_approach_preview_reenters(servo, frame_id=6)


def test_positive_projected_vertical_excursion_only_withholds_preview() -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    current = target(5, y=0.269, y_rate=0.20)
    withheld = step(
        servo,
        current,
        next_target=target(5, track_id="vq2-track-000002"),
        requested_next_blend=0.3,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    _assert_approach_preview_withheld_current_only(
        withheld,
        current=current,
        servo=servo,
    )
    _assert_approach_preview_reenters(servo, frame_id=6)


def test_latest_live_predictive_excursion_reenters_without_servo_reset() -> None:
    """Replay pubs 116-119 from 20260725T025905Z without weakening bounds."""

    servo = ImageVisualServo()
    _latch_passage_blend(servo, requested_blend=0.35)

    def live_target(
        frame_id: int,
        publication: int,
        *,
        track_id: str,
        x: float,
        y: float,
        x_rate: float,
        y_rate: float,
        scale: float,
        scale_rate: float,
    ) -> VisualTarget:
        return target(
            frame_id,
            publication_sequence=publication,
            track_id=track_id,
            received=20.0 + 0.033 * (publication - 116),
            x=x,
            y=y,
            x_rate=x_rate,
            y_rate=y_rate,
            log_scale=math.log(scale),
            scale_rate=scale_rate,
            consecutive=publication - 2,
        )

    current_116 = live_target(
        2_426_868,
        116,
        track_id="vq2-track-000001",
        x=-0.028125,
        y=-0.24444444444444446,
        x_rate=-0.10731768644053542,
        y_rate=-0.22429271246437915,
        scale=0.18220160122362563,
        scale_rate=0.3104297767001267,
    )
    next_116 = live_target(
        2_426_868,
        116,
        track_id="vq2-track-000002",
        x=0.315625,
        y=-0.33333333333333337,
        x_rate=0.0018898974236294065,
        y_rate=-0.33691502527922756,
        scale=0.08471873715745683,
        scale_rate=0.05329394196930489,
    )
    before = step(
        servo,
        current_116,
        next_target=next_116,
        requested_next_blend=0.35,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    assert before.next_gate_blend == pytest.approx(0.35)

    preview_withheld = (
        (
            live_target(
                2_426_869,
                117,
                track_id="vq2-track-000001",
                x=-0.028125,
                y=-0.25555555555555554,
                x_rate=-0.04829295889824094,
                y_rate=-0.278719842422926,
                scale=0.18437205741531323,
                scale_rate=0.3215299877188218,
            ),
            live_target(
                2_426_869,
                117,
                track_id="vq2-track-000002",
                x=0.3125,
                y=-0.34444444444444444,
                x_rate=-0.04915245541954262,
                y_rate=-0.3293998831896078,
                scale=0.08569568250501299,
                scale_rate=0.19604192639730583,
            ),
        ),
        (
            live_target(
                2_426_870,
                118,
                track_id="vq2-track-000001",
                x=-0.03125,
                y=-0.26111111111111107,
                x_rate=-0.07141314843849249,
                y_rate=-0.21374627030682014,
                scale=0.18663863402790394,
                scale_rate=0.4111171654423782,
            ),
            live_target(
                2_426_870,
                118,
                track_id="vq2-track-000002",
                x=0.3125,
                y=-0.3555555555555555,
                x_rate=-0.02211860493879418,
                y_rate=-0.3250390908922066,
                scale=0.08569568250501299,
                scale_rate=0.08821886687878762,
            ),
        ),
    )
    for current, successor in preview_withheld:
        output = step(
            servo,
            current,
            next_target=successor,
            requested_next_blend=0.35,
            allow_advance=False,
            allow_passage_safe_next_blend=True,
        )
        _assert_approach_preview_withheld_current_only(
            output,
            current=current,
            servo=servo,
        )

    current_119 = live_target(
        2_426_871,
        119,
        track_id="vq2-track-000001",
        x=-0.03125,
        y=-0.26111111111111107,
        x_rate=-0.03213949740953352,
        y_rate=-0.09618582163806906,
        scale=0.18663863402790394,
        scale_rate=0.18495510630205794,
    )
    next_119 = live_target(
        2_426_871,
        119,
        track_id="vq2-track-000002",
        x=0.3125,
        y=-0.3555555555555555,
        x_rate=-0.00995337222245738,
        y_rate=-0.14626759090149297,
        scale=0.08784104611578832,
        scale_rate=0.5188482243935452,
    )
    resumed = step(
        servo,
        current_119,
        next_target=next_119,
        requested_next_blend=0.35,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )

    assert (
        abs(
            current_119.normalized_y_down
            + current_119.normalized_y_rate_down_s * 0.10
        )
        < PREPASS_CURRENT_MAX_ABS_Y_NORM
    )
    assert resumed.next_gate_blend == pytest.approx(0.35)
    assert resumed.next_horizontal_error == pytest.approx(
        next_119.normalized_x
    )
    assert resumed.next_vertical_error_image_down == pytest.approx(
        next_119.normalized_y_down
    )
    assert servo.latched_next_track_id == "vq2-track-000002"
    assert not resumed.passage_preview_retired
    assert resumed.yaw_rate_rad_s < 0.0
    assert resumed.target_pitch_rad > 0.0
    assert not resumed.advance_enabled


def test_recorded_prepass_116_158_withholds_preview_without_aborting() -> None:
    """Replay the selected live trace's scalar track evidence as one path.

    Source: 20260725T025905Z-visual-align-375d9622, trace SHA-256
    e9f28ffbc4a628708fe59602d3de2c0fde38167e79086d41a9596d85c5667fe8.
    Columns are publication, observation time, current x/y/vx/vy/scale/rate,
    next x/y/vx/vy/scale/rate/visible, and current clipping edges.
    """

    recorded_csv = """
116,81197.484,-0.028125,-0.244444,-0.107318,-0.224293,0.182289,0.310430,0.315625,-0.333333,0.001890,-0.336915,0.084779,0.053294,1,0
117,81197.515,-0.028125,-0.255556,-0.048293,-0.278720,0.184372,0.321530,0.312500,-0.344444,-0.049152,-0.329400,0.085696,0.196042,1,0
118,81197.546,-0.031250,-0.261111,-0.071413,-0.213746,0.187488,0.411161,0.312500,-0.355556,-0.022119,-0.324875,0.085696,0.088219,1,0
119,81197.578,-0.031250,-0.261111,-0.032136,-0.096186,0.187488,0.185022,0.312500,-0.355556,-0.009953,-0.146194,0.087797,0.518527,1,0
120,81197.609,-0.034375,-0.261111,-0.063828,-0.043284,0.189583,0.258793,0.312500,-0.361111,-0.004479,-0.153550,0.086878,0.067044,1,0
121,81197.656,-0.031250,-0.255556,0.020789,0.068543,0.196806,0.708859,0.315625,-0.372222,0.047496,-0.245139,0.086878,0.030170,1,0
122,81197.687,-0.031250,-0.266667,0.009355,-0.143922,0.192706,-0.012198,0.318750,-0.377778,0.070526,-0.197696,0.087797,0.179150,1,0
123,81197.718,-0.031250,-0.266667,0.004210,-0.064765,0.197906,0.417269,0.318750,-0.383333,0.031737,-0.177167,0.086878,-0.086513,1,0
124,81197.750,-0.031250,-0.272222,0.001894,-0.131469,0.201039,0.477089,0.321875,-0.388889,0.071839,-0.182050,0.088976,0.400548,1,0
125,81197.781,-0.031250,-0.272222,0.000852,-0.059161,0.204167,0.479831,0.321875,-0.394444,0.032328,-0.177337,0.089195,0.222514,1,0
126,81197.812,-0.034375,-0.266667,-0.048107,0.059583,0.205206,0.294691,0.325000,-0.400000,0.063038,-0.166007,0.090139,0.263474,1,0
127,81197.859,-0.034375,-0.266667,-0.021648,0.026812,0.209372,0.441091,0.328125,-0.405556,0.076323,-0.159959,0.090139,0.118563,1,0
128,81197.890,-0.034375,-0.266667,-0.009742,0.012066,0.212500,0.434158,0.331250,-0.411111,0.084014,-0.160281,0.092233,0.418414,1,0
129,81197.921,-0.034375,-0.266667,-0.004384,0.005430,0.215622,0.487008,0.331250,-0.416667,0.037806,-0.183197,0.091073,-0.064800,1,0
130,81197.953,-0.034375,-0.266667,-0.001973,0.002443,0.218740,0.438002,0.334375,-0.422222,0.064655,-0.167136,0.091073,-0.029160,1,0
131,81197.984,-0.034375,-0.266667,-0.000888,0.001099,0.223956,0.599784,0.337500,-0.427778,0.082495,-0.170145,0.092233,0.203196,1,0
132,81198.015,-0.037500,-0.261111,-0.049993,0.088662,0.229157,0.634264,0.340625,-0.427778,0.086717,-0.076565,0.093169,0.251747,1,0
133,81198.046,-0.037500,-0.255556,-0.022497,0.127028,0.234373,0.638363,0.343750,-0.433333,0.088033,-0.121584,0.093169,0.113286,1,0
134,81198.078,-0.037500,-0.250000,-0.010124,0.168331,0.238539,0.639883,0.343750,-0.438889,0.039615,-0.165881,0.094327,0.298032,1,0
135,81198.109,-0.034375,-0.250000,0.045034,0.075749,0.244789,0.698379,0.346875,-0.444444,0.067417,-0.162807,0.095266,0.291237,1,0
136,81198.156,-0.037500,-0.244444,-0.028395,0.120595,0.251040,0.706858,0.353125,-0.450000,0.127660,-0.159771,0.096420,0.318674,1,0
137,81198.187,-0.037500,-0.238889,-0.012778,0.142442,0.257273,0.707350,0.353125,-0.455556,0.057447,-0.160071,0.097561,0.330133,1,0
138,81198.218,-0.037500,-0.233333,-0.005750,0.154390,0.264575,0.773193,0.356250,-0.455556,0.076640,-0.072032,0.098513,0.306356,1,0
139,81198.250,-0.037500,-0.233333,-0.002588,0.069476,0.270825,0.767140,0.359375,-0.461111,0.090594,-0.132159,0.100778,0.545975,1,0
140,81198.281,-0.037500,-0.222222,-0.001164,0.220024,0.279136,0.858663,0.362500,-0.461111,0.093856,-0.059471,0.100778,0.245689,1,0
141,81198.312,-0.037500,-0.216667,-0.000524,0.187627,0.287470,0.855677,0.365625,-0.466667,0.092082,-0.115378,0.100778,0.110560,1,0
142,81198.359,-0.050000,-0.211111,-0.194161,0.170621,0.303988,1.251839,0.371875,-0.472222,0.138399,-0.138109,0.103833,0.512993,1,0
144,81198.421,-0.056250,-0.200000,-0.143063,0.175785,0.330929,1.319958,0.378125,-0.477778,0.117970,-0.111652,0.106066,0.420461,1,0
145,81198.453,-0.053125,-0.194444,-0.015702,0.165639,0.339353,0.985544,0.381250,-0.483333,0.101763,-0.136779,0.107165,0.349795,1,0
146,81198.484,-0.046875,-0.177778,0.093528,0.342787,0.347761,0.837389,0.387500,-0.494444,0.146387,-0.240383,0.109251,0.467653,1,0
147,81198.515,-0.053125,-0.172222,-0.051680,0.237603,0.367376,1.200052,0.390625,-0.494444,0.112758,-0.108173,0.110240,0.345607,1,0
148,81198.546,-0.050000,-0.161111,0.028381,0.290519,0.381045,1.143655,0.393750,-0.500000,0.102378,-0.140476,0.111337,0.319130,1,0
149,81198.578,-0.043750,-0.155556,0.130189,0.235104,0.394680,1.175162,0.396875,-0.505556,0.104779,-0.167585,0.114508,0.671196,1,0
150,81198.609,-0.046875,-0.144444,0.007080,0.288927,0.417551,1.457253,0.403125,-0.505556,0.150161,-0.075413,0.116592,0.599401,1,0
151,81198.656,-0.050000,-0.133333,-0.045709,0.303866,0.440417,1.489945,0.406250,-0.516667,0.116467,-0.207785,0.118677,0.547012,1,0
152,81198.687,-0.053125,-0.116667,-0.070529,0.403193,0.468523,1.659503,0.412500,-0.522222,0.152330,-0.182321,0.120815,0.531672,1,0
153,81198.718,0.031250,-0.105556,1.341126,0.362226,0.554182,3.478811,0.412500,-0.522222,0.152330,-0.182321,0.120815,0.531672,0,0
154,81198.750,0.021875,-0.094444,0.430007,0.368631,0.579938,2.406175,0.412500,-0.522222,0.152330,-0.182321,0.120815,0.531672,0,0
155,81198.781,0.006250,-0.083333,-0.063941,0.348955,0.614682,2.041446,0.412500,-0.522222,0.152330,-0.182321,0.120815,0.531672,0,0
156,81198.812,-0.009375,-0.066667,-0.277212,0.422031,0.657251,1.983332,0.412500,-0.522222,0.152330,-0.182321,0.120815,0.531672,0,0
157,81198.859,-0.037500,-0.055556,-0.554274,0.359604,0.708976,2.049455,0.412500,-0.522222,0.152330,-0.182321,0.120815,0.531672,0,0
158,81198.890,-0.068750,-0.044444,-0.770430,0.347069,0.769072,2.278740,0.412500,-0.522222,0.152330,-0.182321,0.120815,0.531672,0,2
"""
    rows = tuple(
        tuple(float(value) for value in line.split(","))
        for line in recorded_csv.strip().splitlines()
    )
    publications = tuple(int(row[0]) for row in rows)
    assert publications == (
        tuple(range(116, 143)) + tuple(range(144, 159))
    )
    assert 143 not in publications

    servo = ImageVisualServo()
    _latch_passage_blend(servo, requested_blend=0.35)
    preview_withheld_publications = []
    blended_publications = []
    missing_next_current_only_publications = []

    for row in rows:
        (
            publication_value,
            observation_s,
            current_x,
            current_y,
            current_vx,
            current_vy,
            current_scale,
            current_scale_rate,
            next_x,
            next_y,
            next_vx,
            next_vy,
            next_scale,
            next_scale_rate,
            next_visible_value,
            current_clipping_value,
        ) = row
        publication = int(publication_value)
        frame_id = 2_426_752 + publication
        current = target(
            frame_id,
            publication_sequence=publication,
            received=observation_s,
            x=current_x,
            y=current_y,
            x_rate=current_vx,
            y_rate=current_vy,
            log_scale=math.log(current_scale),
            scale_rate=current_scale_rate,
            consecutive=publication - 2,
            clipped=bool(current_clipping_value),
        )
        next_gate = (
            target(
                frame_id,
                publication_sequence=publication,
                received=observation_s,
                track_id="vq2-track-000002",
                x=next_x,
                y=next_y,
                x_rate=next_vx,
                y_rate=next_vy,
                log_scale=math.log(next_scale),
                scale_rate=next_scale_rate,
                consecutive=publication - 2,
            )
            if bool(next_visible_value)
            else None
        )
        output = step(
            servo,
            current,
            next_target=next_gate,
            requested_next_blend=0.35,
            allow_advance=False,
            allow_passage_safe_next_blend=True,
        )
        assert not output.advance_enabled
        assert not output.passage_preview_retired
        assert servo.latched_next_track_id == "vq2-track-000002"
        if next_gate is None:
            assert output.next_gate_blend == 0.0
            assert output.next_horizontal_error is None
            assert output.next_vertical_error_image_down is None
            assert output.reviewed_next_track_id is None
            assert all(
                math.isfinite(value)
                for value in (
                    output.target_roll_rad,
                    output.target_pitch_rad,
                    output.yaw_rate_rad_s,
                    output.thrust,
                )
            )
            missing_next_current_only_publications.append(publication)
        elif output.next_gate_blend == 0.0:
            _assert_approach_preview_withheld_current_only(
                output,
                current=current,
                servo=servo,
            )
            preview_withheld_publications.append(publication)
        else:
            expected_preview_authority = max(
                0.0,
                min(
                    1.0,
                    (2.0 - current_scale_rate)
                    / (2.0 - 1.10),
                ),
            )
            assert output.next_gate_blend == pytest.approx(
                0.35 * expected_preview_authority
            )
            assert output.next_horizontal_error == pytest.approx(
                next_gate.normalized_x
            )
            assert output.next_vertical_error_image_down == pytest.approx(
                next_gate.normalized_y_down
            )
            blended_publications.append(publication)

    assert preview_withheld_publications == [117, 118, 122, 124]
    assert blended_publications == (
        [116]
        + list(range(119, 122))
        + [123]
        + list(range(125, 143))
        + list(range(144, 153))
    )
    assert len(blended_publications) == 32
    assert missing_next_current_only_publications == [
        153,
        154,
        155,
        156,
        157,
        158,
    ]
    assert int(rows[-1][-1]) == int(FrameEdge.TOP)


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


def test_approach_correction_reversal_withholds_preview_and_reenters() -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    current = target(5, x=-0.17)
    withheld = step(
        servo,
        current,
        next_target=target(
            5,
            track_id="vq2-track-000002",
            x=1.0,
        ),
        requested_next_blend=0.3,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )
    _assert_approach_preview_withheld_current_only(
        withheld,
        current=current,
        servo=servo,
    )
    _assert_approach_preview_reenters(servo, frame_id=6)


def test_passage_correction_reversal_still_retires_preview() -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    current = target(5, x=-0.17)
    retired = step(
        servo,
        current,
        next_target=target(
            5,
            track_id="vq2-track-000002",
            x=1.0,
        ),
        requested_next_blend=0.3,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )
    assert retired.next_gate_blend == 0.0
    assert retired.next_horizontal_error is None
    assert retired.next_vertical_error_image_down is None
    assert retired.reviewed_next_track_id is None
    assert not retired.advance_enabled
    assert retired.passage_preview_retired
    assert [
        detail.violation
        for detail in retired.passage_preview_retirement_violations
    ] == [PassageSafetyViolation.CURRENT_HORIZONTAL_CORRECTION_REVERSAL]
    assert retired.effective_horizontal_error == pytest.approx(
        current.normalized_x
    )
    assert servo.latched_next_track_id == "vq2-track-000002"

    remains_retired = step(
        servo,
        target(6),
        next_target=target(6, track_id="vq2-track-000002"),
        requested_next_blend=0.3,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )
    assert remains_retired.next_gate_blend == 0.0
    assert remains_retired.next_horizontal_error is None
    assert remains_retired.next_vertical_error_image_down is None
    assert remains_retired.passage_preview_retired
    assert remains_retired.passage_preview_retirement_violations == ()


def test_passage_safe_next_blend_can_retain_advance_authority() -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    output = step(
        servo,
        target(5),
        next_target=target(
            5,
            track_id="vq2-track-000002",
            x=0.30,
            y=-0.20,
        ),
        requested_next_blend=0.3,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )

    assert output.next_gate_blend == pytest.approx(0.3)
    assert output.next_horizontal_error == pytest.approx(0.30)
    assert output.next_vertical_error_image_down == pytest.approx(-0.20)
    assert output.advance_enabled


def test_rapid_expansion_retains_coordinated_turn_and_brakes() -> None:
    stable_servo = ImageVisualServo()
    expanding_servo = ImageVisualServo()
    _latch_passage_blend(stable_servo)
    _latch_passage_blend(expanding_servo)
    next_gate = target(
        5,
        track_id="vq2-track-000002",
        x=0.30,
    )

    stable = step(
        stable_servo,
        target(5),
        next_target=next_gate,
        requested_next_blend=0.3,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )
    expanding = step(
        expanding_servo,
        target(5, scale_rate=1.7),
        next_target=next_gate,
        requested_next_blend=0.3,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )

    assert stable.next_gate_blend == pytest.approx(0.3)
    assert expanding.next_gate_blend == pytest.approx(0.1)
    assert stable.effective_horizontal_error == pytest.approx(0.09)
    assert expanding.effective_horizontal_error == pytest.approx(0.03)
    assert expanding.yaw_rate_rad_s == pytest.approx(
        stable.yaw_rate_rad_s
    )
    assert expanding.target_roll_rad == pytest.approx(
        stable.target_roll_rad
    )
    assert expanding.advance_enabled is True
    assert expanding.target_pitch_rad > stable.target_pitch_rad
    assert expanding.thrust < stable.thrust


def test_advance_passage_preview_requires_an_existing_latch() -> None:
    servo = ImageVisualServo()

    with pytest.raises(
        VisualServoRefusal,
        match="requires an established next-track latch",
    ):
        step(
            servo,
            target(1),
            next_target=target(1, track_id="vq2-track-000002"),
            requested_next_blend=0.3,
            allow_advance=True,
            allow_passage_safe_next_blend=True,
        )


def test_zero_blend_can_review_identity_before_passage_growth() -> None:
    servo = ImageVisualServo()
    reviewed = None
    for frame_id in range(1, 4):
        reviewed = step(
            servo,
            target(frame_id),
            next_target=target(
                frame_id,
                track_id="vq2-track-000002",
                x=0.30,
            ),
            requested_next_blend=0.0,
            allow_advance=False,
            allow_passage_safe_next_blend=True,
        )
        assert reviewed.next_gate_blend == 0.0

    assert reviewed is not None
    assert reviewed.reviewed_next_track_id == "vq2-track-000002"
    assert servo.latched_next_track_id == "vq2-track-000002"

    passage = step(
        servo,
        target(4),
        next_target=target(
            4,
            track_id="vq2-track-000002",
            x=0.30,
        ),
        requested_next_blend=0.20,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )
    assert passage.next_gate_blend == pytest.approx(0.20)
    assert passage.advance_enabled


def test_broad_passage_preview_cannot_gain_forward_authority() -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    output = step(
        servo,
        target(
            5,
            y=-0.22,
            y_rate=-0.20,
            scale_rate=1.20,
        ),
        next_target=target(
            5,
            track_id="vq2-track-000002",
            x=0.30,
            y=-0.20,
        ),
        requested_next_blend=0.3,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )

    assert output.next_gate_blend == pytest.approx(
        0.3 * (2.0 - 1.20) / (2.0 - 1.10)
    )
    assert not output.advance_enabled
    assert output.target_pitch_rad >= 0.0


def test_passage_retains_heading_through_vertical_scale_degradation() -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    output = step(
        servo,
        target(
            124,
            x=-0.025,
            y=-0.1111111111111111,
            x_rate=-0.00456222891438857,
            y_rate=-1.03370659731062,
            log_scale=math.log(0.171340709957934),
            scale_rate=2.39406892894344,
        ),
        next_target=target(
            124,
            track_id="vq2-track-000002",
            x=0.31875,
            y=-0.244444444444444,
        ),
        requested_next_blend=0.3,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )

    assert output.next_gate_blend == 0.0
    assert output.next_horizontal_error == pytest.approx(0.31875)
    assert output.next_vertical_error_image_down is None
    assert output.yaw_rate_rad_s < -0.05
    # The current-aperture projected demand is small, so successor yaw cannot
    # import full successor bank through the passage barrier.
    assert abs(output.target_roll_rad) < 0.02
    assert not output.advance_enabled
    assert not output.passage_preview_retired
    assert output.passage_preview_retirement_violations == ()

    recovered = step(
        servo,
        target(
            128,
            x=-0.021875,
            y=-0.205555555555556,
            x_rate=0.00424390377519777,
            y_rate=-0.393759905693026,
            log_scale=math.log(0.194788881441763),
            scale_rate=0.481916483947092,
        ),
        next_target=target(
            128,
            track_id="vq2-track-000002",
            x=0.325,
            y=-0.3,
        ),
        requested_next_blend=0.3,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )
    assert recovered.next_gate_blend > 0.0
    assert recovered.next_horizontal_error == pytest.approx(0.325)
    assert not recovered.passage_preview_retired


def test_live_passage_margin_tapers_successor_yaw_without_reversing() -> None:
    """Regress publication 155 from run 20260726T174536Z."""

    servo = ImageVisualServo()
    _latch_passage_blend(servo)

    output = step(
        servo,
        target(
            155,
            x=-0.096875,
            y=-0.1444444444444445,
            x_rate=-0.1525009244685644,
            y_rate=0.2851080482436885,
            log_scale=math.log(0.42041542781449454),
            scale_rate=1.622599080008312,
        ),
        next_target=target(
            155,
            track_id="vq2-track-000002",
            x=0.328125,
            y=-0.48888888888888893,
            x_rate=0.1450844041912014,
            y_rate=-0.2580609878950182,
            log_scale=math.log(0.11133657679906157),
            scale_rate=0.5904640930065196,
        ),
        requested_next_blend=0.25,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )

    assert output.next_gate_blend > 0.0
    assert (
        -MAX_VISUAL_YAW_RATE_RAD_S
        < output.yaw_rate_rad_s
        < -0.05
    )
    assert not output.passage_preview_retired


def test_unusable_optional_passage_preview_is_current_only() -> None:
    missing_servo = ImageVisualServo()
    clipped_servo = ImageVisualServo()
    _latch_passage_blend(missing_servo)
    _latch_passage_blend(clipped_servo)

    current = target(5)
    missing = step(
        missing_servo,
        current,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )
    clipped = step(
        clipped_servo,
        current,
        next_target=target(
            5,
            track_id="vq2-track-000002",
            x=0.60,
            y=-0.50,
            clipped=True,
            center_censored=True,
        ),
        requested_next_blend=0.3,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )

    assert clipped == missing


def test_attempt14_late_passage_preview_retires_at_first_hard_violation() -> None:
    servo = ImageVisualServo()
    _latch_passage_blend(servo)
    apparent_scale_156 = 0.44464405239847
    requested_blend_156 = 0.35 * (
        (math.log(apparent_scale_156) + 1.80) / 1.30
    )

    publication_156 = step(
        servo,
        target(
            156,
            x=0.009374999999999911,
            y=-0.0444444444444444,
            x_rate=0.047301552867145616,
            y_rate=0.1910371891752105,
            log_scale=math.log(apparent_scale_156),
            scale_rate=1.456276983534964,
        ),
        next_target=target(
            156,
            track_id="vq2-track-000002",
            x=0.453125,
            y=-0.43333333333333335,
            x_rate=0.15383742246049958,
            y_rate=-0.2198960994105984,
            log_scale=math.log(0.11975089885999922),
            scale_rate=0.42128522736511775,
        ),
        requested_next_blend=requested_blend_156,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )

    assert publication_156.next_gate_blend == pytest.approx(
        requested_blend_156
        * (2.0 - 1.456276983534964)
        / (2.0 - 1.10)
    )
    assert not publication_156.advance_enabled

    publication_157 = step(
        servo,
        target(
            157,
            x=0.09687500000000004,
            y=-0.03888888888888886,
            x_rate=1.790118948812271,
            y_rate=0.19827360814611752,
            log_scale=math.log(0.5199116845409634),
            scale_rate=3.816682075186591,
        ),
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )
    assert publication_157.next_gate_blend == 0.0
    assert publication_157.brake_reason == "scale_rate"

    safe_again = step(
        servo,
        target(158),
        next_target=target(
            158,
            track_id="vq2-track-000002",
            x=0.30,
            y=-0.20,
        ),
        requested_next_blend=0.3,
        allow_advance=True,
        allow_passage_safe_next_blend=True,
    )
    assert safe_again.next_gate_blend == 0.0


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
    assert abs(output.target_roll_rad) < MAX_VISUAL_TARGET_ROLL_RAD
    assert output.target_pitch_rad == pytest.approx(
        servo.tuning.brake_pitch_rad
    )


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
    assert abs(output.target_roll_rad) <= 0.16
    assert -0.30 <= output.target_pitch_rad <= 0.15
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
    assert abs(outward.target_roll_rad) < MAX_VISUAL_TARGET_ROLL_RAD
    assert outward.target_pitch_rad == pytest.approx(
        servo.tuning.brake_pitch_rad
    )

    servo.reset_segment()
    # The opposite image error asks for negative yaw back toward reference.
    inward = step(
        servo,
        target(2, x=0.5),
        segment_yaw_excursion_rad=VISUAL_SEGMENT_YAW_SOFT_STOP_RAD,
    )
    assert inward.yaw_envelope_limited is False
    assert inward.yaw_rate_rad_s < 0.0
    assert inward.target_roll_rad == MAX_VISUAL_TARGET_ROLL_RAD
    assert inward.target_pitch_rad == MAX_VISUAL_TARGET_PITCH_RAD


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
            requested_next_blend=0.35,
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
        requested_next_blend=0.35,
        allow_advance=False,
        allow_passage_safe_next_blend=True,
    )

    assert output.next_gate_blend == pytest.approx(0.35)
    assert output.effective_horizontal_error == pytest.approx(0.1128125)
    assert -MAX_VISUAL_YAW_RATE_RAD_S <= output.yaw_rate_rad_s < -0.03
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


@pytest.mark.parametrize(
    (
        "frame_id",
        "publication_sequence",
        "x",
        "y",
        "x_rate",
        "y_rate",
        "scale",
        "scale_rate",
    ),
    (
        (
            1,
            1,
            0.65625,
            -0.7277777777777779,
            0.345412,
            -0.607664,
            math.exp(-1.530552),
            1.0593890808936899,
        ),
        (
            1_519_525,
            183,
            0.621875,
            -0.7222222222222222,
            0.2885910808222251,
            -0.7462772693726882,
            0.21556208824579728,
            1.009549,
        ),
    ),
)
def test_top_clip_uses_observable_horizontal_brake_authority(
    frame_id,
    publication_sequence,
    x,
    y,
    x_rate,
    y_rate,
    scale,
    scale_rate,
):
    servo = ImageVisualServo()
    output = step(
        servo,
        target(
            frame_id,
            publication_sequence=publication_sequence,
            x=x,
            y=y,
            x_rate=x_rate,
            y_rate=y_rate,
            log_scale=math.log(scale),
            scale_rate=scale_rate,
            clipped=True,
            center_censored=True,
            vertical_censored=True,
        ),
    )

    assert output.advance_enabled is False
    assert output.brake_reason == "target_edge_or_clipping"
    assert output.yaw_rate_rad_s < 0.0
    assert output.effective_horizontal_error == pytest.approx(x)
    assert output.effective_vertical_error_image_down == 0.0
    assert output.effective_vertical_rate_down_s == 0.0
    assert output.target_roll_rad == pytest.approx(0.0)
    assert output.target_pitch_rad == MAX_VISUAL_TARGET_PITCH_RAD
    assert output.thrust == pytest.approx(servo.tuning.brake_thrust)


def test_top_clip_retains_last_vertical_observable_collective():
    servo = ImageVisualServo()
    clean = step(
        servo,
        target(
            1,
            x=0.60,
            y=-0.68,
            x_rate=0.32,
            y_rate=-0.60,
        ),
    )
    top = step(
        servo,
        target(
            2,
            x=0.62,
            y=-0.72,
            x_rate=0.24,
            y_rate=-0.89,
            clipped=True,
            center_censored=True,
            vertical_censored=True,
        ),
    )

    assert clean.thrust > MIN_VISUAL_THRUST
    assert top.thrust == pytest.approx(clean.thrust)
    assert top.effective_vertical_error_image_down == 0.0
    assert top.effective_vertical_rate_down_s == 0.0
    assert top.yaw_rate_rad_s < 0.0
    assert top.target_roll_rad == pytest.approx(0.0)
    assert top.target_pitch_rad == MAX_VISUAL_TARGET_PITCH_RAD


def test_run7_precredit_successor_rows_produce_bounded_no_advance_recenter():
    rows = (
        (174, 0.538, -0.556, 0.205, -0.227, 0.163, 0.52),
        (175, 0.547, -0.567, 0.241, -0.278, 0.169, 0.73),
        (176, 0.559, -0.583, 0.306, -0.389, 0.172, 0.62),
        (177, 0.569, -0.594, 0.285, -0.350, 0.177, 0.75),
        (178, 0.578, -0.606, 0.318, -0.382, 0.182, 0.92),
        (179, 0.587, -0.617, 0.289, -0.345, 0.187, 0.85),
    )
    servo = ImageVisualServo()
    outputs = [
        step(
            servo,
            target(
                sequence,
                track_id="vq2-track-000003",
                x=x,
                y=y,
                x_rate=x_rate,
                y_rate=y_rate,
                log_scale=math.log(scale),
                scale_rate=scale_rate,
                consecutive=sequence - 171,
            ),
            allow_advance=False,
        )
        for (
            sequence,
            x,
            y,
            x_rate,
            y_rate,
            scale,
            scale_rate,
        ) in rows
    ]

    assert all(not output.advance_enabled for output in outputs)
    assert all(output.next_gate_blend == 0.0 for output in outputs)
    assert all(output.yaw_rate_rad_s == -0.15 for output in outputs)
    assert all(
        output.target_roll_rad == pytest.approx(0.0)
        for output in outputs
    )
    assert all(
        output.target_pitch_rad == MAX_VISUAL_TARGET_PITCH_RAD
        for output in outputs
    )
    assert all(
        output.thrust == pytest.approx(servo.tuning.brake_thrust)
        for output in outputs
    )


def test_large_outward_bearing_uses_measured_yaw_and_unloads_bank():
    servo = ImageVisualServo()
    output = step(
        servo,
        target(
            174,
            x=0.45625,
            y=-0.60,
            x_rate=0.21851,
            y_rate=-0.21950,
            log_scale=math.log(0.1643),
            scale_rate=0.757,
        ),
        allow_advance=False,
    )

    assert output.yaw_rate_rad_s == -MAX_VISUAL_YAW_RATE_RAD_S
    assert output.target_roll_rad == pytest.approx(0.0)
    assert output.target_pitch_rad == MAX_VISUAL_TARGET_PITCH_RAD


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
