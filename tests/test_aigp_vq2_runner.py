"""Pure safety/geometry tests for the staged VQ2 runner."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import math
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.aigp_vq2_run as vq2_module
from aigp_loop.replay import (
    AsyncReplayRecorder,
    ReplayBundleReader,
    ReplayBundleWriter,
)
from competition.adapter import (
    AttitudeRateCommand,
    CameraFrame,
    IMUData,
    Quaternion,
    RaceActiveBoundaryChangedBeforeWire,
    TelemetryState,
    VisionPublicationLeaseAcquisitionTimeout,
)
from competition.aigp_messages import RaceStatus
from competition.aigp_mavlink import (
    AIGPMavlinkAdapter,
    CalibrationCollisionV1,
    CalibrationResetBoundaryV1,
    MavlinkCollisionStats,
    MavlinkIngressStats,
    PoweredMavlinkTransportState,
)
from competition.vq2_capture import (
    ActuatorOutputStatusPayloadV1,
    AttitudeTargetOutboundV1,
    AttitudeTargetWireV1,
    HeartbeatPayloadV1,
    CommandLongWireV1,
    GCSHeartbeatWireV1,
    MavlinkIngressV1,
    NonAttitudeOutboundV1,
    RaceStatusPayloadV1,
    ReceivedActuatorOutputStatusV1,
    ReceivedHeartbeatV1,
    ReceivedIMUSampleV1,
    ReceivedRaceStatusV1,
    TimesyncWireV1,
)
from competition.vq2_contracts import FrameEdge, FrameIdentityV1, FrameTimingV1
from competition.vq2_passive_timing import CameraFrameTimingObservationV1
from competition.vq2_vision import VQ2VisionThread
from competition.vq2_visual_tracker import (
    MultiTargetVisualTracker,
    VisualDetectionFrame,
    VisualTrackRole,
)
from estimation.imu_attitude import (
    AttitudeEstimate,
    ImuAttitudeConfig,
    ImuAttitudeEstimator,
)
from gate_detection.src.gate_detector import GateDetection
from gate_detection.src.vq2_geometry import (
    ApertureSide,
    VQ2ApertureFit,
)
from planning.vq2_visual_servo import (
    MAX_NEXT_GATE_BLEND,
    ServoFrameToken,
    VisualServoOutput,
    VisualTarget,
)
from scripts.aigp_vq2_run import (
    CalibrationAdmission,
    CalibrationAdmissionServices,
    CalibrationAdapterDispatcher,
    CalibrationArguments,
    CalibrationCheckFailure,
    CalibrationChildLifecycle,
    CalibrationChildRunOutput,
    CalibrationChildServices,
    CalibrationClosedArtifacts,
    CalibrationCommandEvidence,
    CalibrationDispatchResult,
    CalibrationExcitationScheduler,
    CalibrationLineageRecorder,
    CalibrationLeaseProof,
    CalibrationSafetyFacts,
    CalibrationSnapshotCapture,
    JsonlRecorder,
    GateTargetTracker,
    ResetProof,
    SafetyAbort,
    VQ2Runner,
    attitude_rate_command,
    clock_rolled_back,
    clock_within_epoch_envelope,
    gate_vertical_reference_px,
    gate_control_center_y_px,
    gate_vertical_thrust,
    crossing_status_decision,
    gate_detection_summary,
    is_close_gate_crossing_candidate,
    is_benign_pad_contact,
    is_crossing_residue,
    next_control_deadline,
    post_gate_observation_deadline,
    select_primary_gate,
    replay_capture_result,
    evaluate_calibration_safety,
)
from scripts import aigp_vq2_powered_attempt as powered_contract
from scripts import aigp_vq2_powered_runtime as powered_runtime
from scripts import aigp_vq2_controller_config as controller_config_module
from tests import test_aigp_vq2_powered_attempt as powered_fixtures


def _detection(x, y, width, height, confidence=0.8):
    return GateDetection(
        center_x=x + width // 2,
        center_y=y + height // 2,
        bbox=(x, y, width, height),
        corners=np.zeros((4, 2)),
        area=width * height,
        estimated_distance=999.0,
        confidence=confidence,
    )


def _nominal_aperture_fit() -> VQ2ApertureFit:
    corners = (
        (69.5, 49.5),
        (130.5, 49.5),
        (130.5, 110.5),
        (69.5, 110.5),
    )
    return VQ2ApertureFit(
        image_size_px=(200, 160),
        support_bbox_px=(59, 39, 83, 83),
        clipping=ApertureSide.NONE,
        fitted_corners_px=corners,
        visible_edges=(
            ApertureSide.LEFT
            | ApertureSide.TOP
            | ApertureSide.RIGHT
            | ApertureSide.BOTTOM
        ),
        visible_corners=(True, True, True, True),
        visible_segments_px=(
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        ),
        geometry_model_id="vq2-visible-inner-quad-lines-v1",
        covariance_model_id="vq2-visible-aperture-diagonal-v1",
        covariance_diagonal=(0.0001, 0.0002, 0.0009, 0.001, 0.001),
        residual_rms_px=0.0,
        inlier_count=240,
        support_count=240,
        confidence=0.9,
        rejection_reason=None,
    )


def _rejected_aperture_fit() -> VQ2ApertureFit:
    return VQ2ApertureFit(
        image_size_px=(200, 160),
        support_bbox_px=(59, 39, 83, 83),
        clipping=ApertureSide.NONE,
        fitted_corners_px=None,
        visible_edges=ApertureSide.NONE,
        visible_corners=(False, False, False, False),
        visible_segments_px=(None, None, None, None),
        geometry_model_id=None,
        covariance_model_id=None,
        covariance_diagonal=None,
        residual_rms_px=None,
        inlier_count=0,
        support_count=240,
        confidence=0.0,
        rejection_reason="ambiguous_multiple_aperture_gaps",
    )


def test_runner_projects_only_nominal_inner_fit_into_passage_geometry() -> None:
    nominal = vq2_module._visual_inner_aperture_from_fit(
        _nominal_aperture_fit()
    )
    low_confidence = vq2_module._visual_inner_aperture_from_fit(
        replace(_nominal_aperture_fit(), confidence=0.10)
    )
    outer_support_touch = vq2_module._visual_inner_aperture_from_fit(
        replace(
            _nominal_aperture_fit(),
            clipping=ApertureSide.TOP,
        )
    )
    clipped = vq2_module._visual_inner_aperture_from_fit(
        replace(
            _nominal_aperture_fit(),
            clipping=ApertureSide.TOP,
            visible_edges=(
                ApertureSide.LEFT
                | ApertureSide.RIGHT
                | ApertureSide.BOTTOM
            ),
            visible_corners=(False, False, True, True),
        )
    )
    rejected = vq2_module._visual_inner_aperture_from_fit(
        _rejected_aperture_fit()
    )
    temporal = vq2_module._visual_inner_aperture_from_fit(
        replace(
            _nominal_aperture_fit(),
            clipping=ApertureSide.BOTTOM,
            geometry_model_id=(
                "vq2-temporally-associated-inner-quad-lines-v1"
            ),
            covariance_model_id=(
                "vq2-temporally-associated-aperture-diagonal-v1"
            ),
        )
    )

    assert nominal.passage_usable
    assert nominal.center_norm == pytest.approx((0.0, 0.0))
    assert nominal.half_size_norm == pytest.approx((0.305, 0.38125))
    assert nominal.measurement_std == pytest.approx((0.01, math.sqrt(0.0002), 0.03))
    assert low_confidence.fitted
    assert not low_confidence.passage_usable
    assert low_confidence.center_norm == nominal.center_norm
    assert low_confidence.half_size_norm == nominal.half_size_norm
    assert low_confidence.log_scale == nominal.log_scale
    assert low_confidence.measurement_std == pytest.approx(
        tuple(
            value * math.sqrt(0.25 / 0.10)
            for value in nominal.measurement_std
        )
    )
    assert low_confidence.health_reason == "aperture_fit_low_confidence"
    assert outer_support_touch.fitted
    assert outer_support_touch.complete_visibility
    assert outer_support_touch.clipping == FrameEdge.NONE
    assert outer_support_touch.health_reason == (
        "outer_support_clipped_tracking_only"
    )
    assert not outer_support_touch.passage_usable
    assert not clipped.passage_usable
    assert clipped.center_norm is None
    assert clipped.health_reason == "aperture_fit_censored:2"
    assert not rejected.passage_usable
    assert rejected.half_size_norm is None
    assert rejected.health_reason == (
        "aperture_fit_rejected:ambiguous_multiple_aperture_gaps"
    )
    assert temporal.fitted
    assert temporal.complete_visibility
    assert temporal.clipping == FrameEdge.NONE
    assert not temporal.passage_usable
    assert temporal.health_reason == "aperture_fit_tracking_prior_only"


def test_runner_thresholds_once_for_all_detection_aperture_fits(
    monkeypatch,
) -> None:
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    mask = np.zeros((160, 200), dtype=np.uint8)
    calls = {"mask": 0, "fit": 0}

    def build_mask(received_image, *, config):
        assert received_image is image
        assert config is vq2_module._VISUAL_INNER_APERTURE_CONFIG
        calls["mask"] += 1
        return mask

    def fit_mask(
        received_mask,
        support_bbox,
        *,
        detection_confidence,
        config,
        tracking_prior,
    ):
        assert received_mask is mask
        assert support_bbox in {(10, 20, 40, 50), (80, 30, 30, 45)}
        assert detection_confidence == pytest.approx(0.8)
        assert config is vq2_module._VISUAL_INNER_APERTURE_CONFIG
        assert tracking_prior is None
        calls["fit"] += 1
        return _rejected_aperture_fit()

    monkeypatch.setattr(vq2_module, "vq2_gate_mask_from_bgr", build_mask)
    monkeypatch.setattr(vq2_module, "fit_vq2_aperture_mask", fit_mask)

    geometries = vq2_module._fit_visual_inner_apertures(
        image,
        (
            _detection(10, 20, 40, 50),
            _detection(80, 30, 30, 45),
        ),
    )

    assert len(geometries) == 2
    assert all(not geometry.passage_usable for geometry in geometries)
    assert calls == {"mask": 1, "fit": 2}


def _visual_frame(
    sequence: int,
    detection: GateDetection,
    *,
    inner=None,
) -> VisualDetectionFrame:
    return VisualDetectionFrame.from_detector_results(
        (detection,),
        generation=4,
        frame_id=100 + sequence,
        publication_sequence=sequence,
        stream_id="runner-prior-test",
        final_unique_packet_monotonic_ns=(
            1_000_000_000 + 40_000_000 * sequence
        ),
        publish_monotonic_ns=(
            1_001_000_000 + 40_000_000 * sequence
        ),
        time_basis_id="host-perf-counter",
        image_size_px=(200, 160),
        aperture_geometries=(inner,),
        camera_source_time_ns=10_000 + sequence,
    )


@pytest.mark.parametrize(
    ("current_detection", "expected_center_y_px"),
    [
        (_detection(10, 20, 120, 140), 92.0),
        (_detection(10, 0, 120, 160), 80.0),
    ],
)
def test_runner_propagates_current_inner_prior_without_using_clipped_height(
    current_detection,
    expected_center_y_px,
) -> None:
    tracker = MultiTargetVisualTracker()
    inner = vq2_module._visual_inner_aperture_from_fit(
        _nominal_aperture_fit()
    )
    first = tracker.update(
        _visual_frame(
            1,
            _detection(20, 20, 100, 100),
            inner=inner,
        )
    )
    track_id = first.visible_track_ids[0]
    tracker.assign_role(track_id, VisualTrackRole.CURRENT)
    pending = _visual_frame(2, current_detection)
    preview = tracker.preview_associations(pending)

    priors, lineages = vq2_module._visual_aperture_tracking_prior_plan(
        pending,
        preview,
        max_predicted_center_displacement_norm=(
            tracker.config.max_center_residual_norm
        ),
    )

    assert lineages == {0: track_id}
    assert len(priors) == 1
    prior = priors[0]
    assert prior is not None
    assert prior.center_px == pytest.approx((106.0, expected_center_y_px))
    assert prior.half_size_px == pytest.approx((36.6, 36.6))
    assert prior.maximum_boundary_residual_px == pytest.approx(8.784)


def test_runner_withholds_inner_prior_from_noncurrent_lineage() -> None:
    tracker = MultiTargetVisualTracker()
    first = tracker.update(
        _visual_frame(
            1,
            _detection(20, 20, 100, 100),
            inner=vq2_module._visual_inner_aperture_from_fit(
                _nominal_aperture_fit()
            ),
        )
    )
    tracker.assign_role(
        first.visible_track_ids[0],
        VisualTrackRole.NEXT,
    )
    pending = _visual_frame(
        2,
        _detection(10, 20, 120, 140),
    )

    priors, lineages = vq2_module._visual_aperture_tracking_prior_plan(
        pending,
        tracker.preview_associations(pending),
        max_predicted_center_displacement_norm=(
            tracker.config.max_center_residual_norm
        ),
    )

    assert priors == (None,)
    assert lineages == {}


def _estimate(roll=0.0, pitch=-0.31, yaw=0.0):
    return AttitudeEstimate(
        timestamp_us=1,
        orientation=Quaternion.from_euler(roll, pitch, yaw),
        body_rates=(0.0, 0.0, 0.0),
        gyro_bias=(0.0, 0.0, 0.0),
        accel_trust=1.0,
        healthy=True,
        propagated=True,
    )


def _vision_snapshot(
    *,
    frame_id=100,
    sim_time_ns=1_000,
    received_monotonic_s=0.0,
    generation=1,
):
    timing = FrameTimingV1(
        identity=FrameIdentityV1(
            "vq2-camera-udp-5600", generation, frame_id
        ),
        camera_source_time_ns=sim_time_ns,
        host_clock_id="host-perf-counter",
        publication_sequence=frame_id,
        first_unique_packet_monotonic_ns=10,
        final_unique_packet_monotonic_ns=11,
        reassembly_complete_monotonic_ns=12,
        decode_start_monotonic_ns=13,
        decode_end_monotonic_ns=14,
        publish_monotonic_ns=15,
    )
    return SimpleNamespace(
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        received_monotonic_s=received_monotonic_s,
        generation=generation,
        timing=timing,
        camera_frame=SimpleNamespace(
            image=np.zeros((360, 640, 3), dtype=np.uint8)
        ),
        age_s=lambda now=None: max(
            0.0,
            (received_monotonic_s if now is None else now)
            - received_monotonic_s,
        ),
        is_fresh=lambda _max_age, _now=None: True,
    )


def test_reset_clock_requires_authoritative_margin():
    assert clock_rolled_back(10_000, 100, 500)
    assert not clock_rolled_back(10_000, 9_600, 500)
    assert not clock_rolled_back(10_000, 9_500, 500)


def test_epoch_envelope_rejects_delayed_old_packet():
    assert clock_within_epoch_envelope(
        500, 1_500, 0.5, units_per_second=1_000.0, slack=700
    )
    assert not clock_within_epoch_envelope(
        500, 1_000_000, 0.5, units_per_second=1_000.0, slack=700
    )


def test_control_deadline_drops_missed_ticks_instead_of_catching_up():
    assert next_control_deadline(10.0, 10.0) == pytest.approx(10.02)

    overrun_now = 10.10
    deadline = next_control_deadline(10.02, overrun_now)
    assert deadline == pytest.approx(overrun_now + 0.02)
    assert deadline - overrun_now >= 0.02 - 1e-12


@pytest.mark.parametrize(
    ("pass_confirmed", "flight_started", "crossing_started", "expected"),
    [
        (10.0, 6.0, 9.9, 10.2),
        (10.0, 6.0, 9.7, 10.1),
        (10.0, 5.05, 9.9, 10.05),
        (10.0, 6.0, None, 10.2),
    ],
)
def test_post_gate_deadline_uses_earliest_fixed_safety_bound(
    pass_confirmed,
    flight_started,
    crossing_started,
    expected,
):
    assert post_gate_observation_deadline(
        pass_confirmed_s=pass_confirmed,
        flight_started_s=flight_started,
        crossing_started_s=crossing_started,
    ) == pytest.approx(expected)


def test_primary_gate_uses_largest_plausible_pixel_box_only():
    far = _detection(410, 138, 27, 45)
    near = _detection(282, 134, 80, 81)
    line = _detection(10, 10, 200, 20)
    low_confidence = _detection(0, 0, 100, 100, confidence=0.01)

    assert select_primary_gate([far, line, near, low_confidence]) is near


def _cyan_course_line_image(*, lower_center_x, upper_center_x):
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cyan_bgr = (255, 255, 0)
    image[
        120:128,
        upper_center_x - 8 : upper_center_x + 9,
    ] = cyan_bgr
    image[
        155:163,
        lower_center_x - 8 : lower_center_x + 9,
    ] = cyan_bgr
    return image


def test_cyan_course_line_observation_rejects_invalid_dimensions():
    assert vq2_module.cyan_course_line_observation(None) is None
    for shape in (
        (360, 640),
        (359, 640, 3),
        (360, 639, 3),
        (1, 360, 640, 3),
    ):
        assert (
            vq2_module.cyan_course_line_observation(
                np.zeros(shape, dtype=np.uint8)
            )
            is None
        )


def test_cyan_course_line_observation_requires_enough_pixels_in_both_bands():
    assert vq2_module.COURSE_LINE_MIN_ROI_PIXELS == 128
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[120, 100:227] = (255, 255, 0)
    image[155, 100:227] = (255, 255, 0)

    assert 227 - 100 == 127
    assert vq2_module.cyan_course_line_observation(image) is None


@pytest.mark.parametrize(
    ("lower_center_x", "upper_center_x", "expected_score"),
    (
        (300, 400, 0.3125),
        (400, 300, -0.3125),
    ),
    ids=("right-turn", "left-turn"),
)
def test_cyan_course_line_observation_reports_signed_turn_and_band_counts(
    lower_center_x,
    upper_center_x,
    expected_score,
):
    observation = vq2_module.cyan_course_line_observation(
        _cyan_course_line_image(
            lower_center_x=lower_center_x,
            upper_center_x=upper_center_x,
        )
    )

    assert observation is not None
    assert observation.turn_score == pytest.approx(expected_score)
    assert observation.lower_center_x == pytest.approx(lower_center_x)
    assert observation.upper_center_x == pytest.approx(upper_center_x)
    assert observation.lower_pixel_count == 136
    assert observation.upper_pixel_count == 136


@pytest.mark.parametrize("score", (True, False, math.nan, math.inf, -math.inf))
def test_course_line_preturn_roll_rejects_bool_and_nonfinite_scores(score):
    with pytest.raises(ValueError):
        vq2_module.course_line_preturn_roll(score)


@pytest.mark.parametrize(
    "score",
    (
        0.0,
        math.nextafter(0.04, 0.0),
        math.nextafter(-0.04, 0.0),
    ),
)
def test_course_line_preturn_roll_has_exact_symmetric_deadband(score):
    assert vq2_module.COURSE_LINE_PRETURN_MIN_SCORE == 0.04
    assert vq2_module.course_line_preturn_roll(score) == 0.0


@pytest.mark.parametrize(
    ("score", "expected_roll"),
    (
        (0.04, 0.032),
        (-0.04, -0.032),
        (0.10, 0.08),
        (-0.10, -0.08),
        (1.0, 0.13),
        (-1.0, -0.13),
    ),
    ids=(
        "positive-threshold",
        "negative-threshold",
        "positive-gain",
        "negative-gain",
        "positive-clamp",
        "negative-clamp",
    ),
)
def test_course_line_preturn_roll_preserves_physical_turn_sign_and_clamps(
    score,
    expected_roll,
):
    assert vq2_module.COURSE_LINE_PRETURN_GAIN == 0.80
    assert vq2_module.COURSE_LINE_PRETURN_LIMIT_RAD == 0.13
    assert vq2_module.course_line_preturn_roll(score) == pytest.approx(
        expected_roll
    )


@pytest.mark.parametrize("score", (True, False, math.nan, math.inf, -math.inf))
def test_course_line_exit_counterroll_rejects_bool_and_nonfinite_scores(score):
    with pytest.raises(ValueError, match="exit-counterroll score"):
        vq2_module.course_line_exit_counterroll(score)


@pytest.mark.parametrize(
    ("score", "expected_roll"),
    (
        (math.nextafter(0.04, 0.0), 0.0),
        (math.nextafter(-0.04, 0.0), 0.0),
        (0.20, -0.08),
        (-0.20, 0.08),
    ),
)
def test_course_line_exit_counterroll_opposes_proved_turn_inside_bounds(
    score,
    expected_roll,
):
    assert vq2_module.COURSE_LINE_EXIT_COUNTERROLL_RAD == 0.08
    assert vq2_module.course_line_exit_counterroll(score) == expected_roll
    assert abs(expected_roll) <= vq2_module.COURSE_LINE_PRETURN_LIMIT_RAD


def test_gate0_centering_roll_preserves_live_proved_sign_and_clamps():
    assert vq2_module.gate0_centering_roll_target(0.25) > 0.0
    assert vq2_module.gate0_centering_roll_target(-0.25) < 0.0
    assert vq2_module.gate0_centering_roll_target(0.0) == 0.0
    assert vq2_module.gate0_centering_roll_target(10.0) == 0.08
    assert vq2_module.gate0_centering_roll_target(-10.0) == -0.08


@pytest.mark.parametrize("normalized_x", (True, math.nan, math.inf, -math.inf))
def test_gate0_centering_roll_rejects_invalid_input(normalized_x):
    with pytest.raises(ValueError, match="gate-0 centering input"):
        vq2_module.gate0_centering_roll_target(normalized_x)


def test_gate_tracker_requires_temporal_continuity():
    tracker = GateTargetTracker()
    for frame_id in range(1, 4):
        tracker.update(
            [_detection(280 + frame_id, 134, 80, 80)],
            frame_id=frame_id,
            sim_time_ns=frame_id * 10,
            received_monotonic_s=1.0 + frame_id * 0.01,
        )
    assert tracker.consecutive == 3

    previous = tracker.target
    tracker.update(
        [_detection(500, 300, 40, 40)],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
    )
    assert tracker.consecutive == 0
    assert tracker.target is previous


def _confirmed_right_edge_tracker():
    tracker = GateTargetTracker()
    prior = _detection(580, 24, 60, 91, confidence=0.402)
    for frame_id in range(1, 4):
        assert tracker.update(
            [prior],
            frame_id=frame_id,
            sim_time_ns=frame_id * 10,
            received_monotonic_s=1.0 + frame_id * 0.01,
        ) is not None
    assert tracker.consecutive == 3
    return tracker, prior


def test_gate_tracker_default_rejects_live_oblique_edge_fragment():
    tracker, prior = _confirmed_right_edge_tracker()
    clipped = _detection(591, 23, 49, 94, confidence=0.360)

    assert tracker.update(
        [clipped],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
    ) is None
    assert tracker.target is not None
    assert tracker.target.bbox == prior.bbox
    assert tracker.consecutive == 0
    assert tracker.last_selection_mode is None


def test_gate_tracker_course_fallback_follows_only_live_right_edge_sequence():
    tracker, _prior = _confirmed_right_edge_tracker()
    later_center_gate = _detection(280, 170, 68, 68, confidence=0.8)
    later_left_gate = _detection(90, 240, 46, 44, confidence=0.8)
    clipped_sequence = (
        _detection(591, 23, 49, 94, confidence=0.360),
        _detection(601, 20, 39, 98, confidence=0.197),
        _detection(612, 22, 28, 52, confidence=0.363),
    )

    for frame_id, clipped in enumerate(clipped_sequence, start=4):
        accepted = tracker.update(
            [later_left_gate, later_center_gate, clipped],
            frame_id=frame_id,
            sim_time_ns=frame_id * 10,
            received_monotonic_s=1.0 + frame_id * 0.01,
            allow_tracked_edge_continuation=True,
            image_width=640,
            image_height=360,
        )
        assert accepted is not None
        assert accepted.bbox == clipped.bbox
        assert tracker.last_selected_detection is clipped
        assert tracker.last_selection_mode == "tracked_edge_continuation"

    last_edge_target = tracker.target
    assert tracker.update(
        [later_left_gate, later_center_gate],
        frame_id=7,
        sim_time_ns=70,
        received_monotonic_s=1.07,
        allow_tracked_edge_continuation=True,
        image_width=640,
        image_height=360,
    ) is None
    assert tracker.consecutive == 0
    assert tracker.target is last_edge_target

    # Once a frame is missed, even another plausible edge fragment cannot
    # resurrect the stale target or hand tracking to a different course gate.
    assert tracker.update(
        [later_center_gate, clipped_sequence[-1]],
        frame_id=8,
        sim_time_ns=80,
        received_monotonic_s=1.08,
        allow_tracked_edge_continuation=True,
        image_width=640,
        image_height=360,
    ) is None
    assert tracker.consecutive == 0
    assert tracker.target is last_edge_target


def test_gate_tracker_course_fallback_accepts_latest_quarter_area_edge_fragment():
    tracker = GateTargetTracker()
    prior = _detection(585, 22, 55, 92, confidence=0.400)
    for frame_id in range(1, 4):
        assert tracker.update(
            [prior],
            frame_id=frame_id,
            sim_time_ns=frame_id * 10,
            received_monotonic_s=1.0 + frame_id * 0.01,
        ) is not None
    predecessor = _detection(596, 21, 44, 95, confidence=0.353)
    assert tracker.update(
        [predecessor],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
        allow_tracked_edge_continuation=True,
        image_width=640,
        image_height=360,
    ) is not None
    clipped = _detection(616, 20, 24, 42, confidence=0.388)
    later_gate = _detection(294, 178, 43, 59, confidence=0.445)

    accepted = tracker.update(
        [later_gate, clipped],
        frame_id=6,
        sim_time_ns=60,
        received_monotonic_s=1.10,
        allow_tracked_edge_continuation=True,
        image_width=640,
        image_height=360,
    )

    assert clipped.bbox[2] * clipped.bbox[3] / predecessor.area == pytest.approx(
        0.2411483254
    )
    assert accepted is not None
    assert accepted.bbox == clipped.bbox
    assert tracker.last_selection_mode == "tracked_edge_continuation"


def test_gate_tracker_prefers_valid_primary_before_edge_fallback():
    tracker, _prior = _confirmed_right_edge_tracker()
    clipped = _detection(591, 23, 49, 94, confidence=0.360)
    valid_primary = _detection(471, 29, 80, 80, confidence=0.8)

    accepted = tracker.update(
        [clipped, valid_primary],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
        allow_tracked_edge_continuation=True,
        image_width=640,
        image_height=360,
    )

    assert accepted is not None
    assert accepted.bbox == valid_primary.bbox
    assert tracker.last_selected_detection is valid_primary
    assert tracker.last_selection_mode == "primary"


@pytest.mark.parametrize(
    "candidate",
    (
        _detection(550, 23, 49, 94, confidence=0.360),
        _detection(601, 0, 39, 98, confidence=0.197),
        _detection(601, 120, 39, 98, confidence=0.197),
        _detection(605, 20, 35, 100, confidence=0.197),
        _detection(591, 23, 49, 94, confidence=0.01),
    ),
    ids=("not-at-edge", "top-and-right", "large-jump", "too-oblique", "low-confidence"),
)
def test_course_edge_fallback_rejects_unproved_geometry(candidate):
    tracker, _prior = _confirmed_right_edge_tracker()

    assert tracker.update(
        [candidate],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
        allow_tracked_edge_continuation=True,
        image_width=640,
        image_height=360,
    ) is None
    assert tracker.consecutive == 0


def test_gate_tracker_course_fragment_union_preserves_live_gate_geometry():
    prior = _detection(466, 0, 137, 119, confidence=0.74)
    upper = _detection(500, 0, 106, 80, confidence=0.69)
    lower = _detection(466, 69, 43, 50, confidence=0.58)
    later_gate = _detection(323, 132, 22, 32, confidence=0.41)

    default_tracker = GateTargetTracker()
    union_tracker = GateTargetTracker()
    for tracker in (default_tracker, union_tracker):
        for frame_id in range(1, 4):
            assert tracker.update(
                [prior],
                frame_id=frame_id,
                sim_time_ns=frame_id * 10,
                received_monotonic_s=1.0 + frame_id * 0.01,
            ) is not None

    default_target = default_tracker.update(
        [upper, lower, later_gate],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
    )
    assert default_target is not None
    assert default_target.bbox == upper.bbox
    assert not default_target.composite

    fused = union_tracker.update(
        [upper, lower, later_gate],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
        allow_tracked_fragment_union=True,
        image_width=640,
        image_height=360,
    )
    assert fused is not None
    assert fused.bbox == (466, 0, 140, 119)
    assert (fused.center_x, fused.center_y) == (536, 59)
    assert fused.composite
    assert union_tracker.last_selected_detection is None
    assert union_tracker.last_selected_detections == (upper, lower)
    assert union_tracker.last_selection_mode == "tracked_fragment_union"


def test_gate_tracker_fragment_union_preserves_directional_live_transition():
    """The faster roll trace remains one continuous, near-square Gate 1."""

    tracker = GateTargetTracker()
    tracker.target = vq2_module.GateTarget(
        frame_id=3,
        sim_time_ns=30,
        received_monotonic_s=1.03,
        center_x=550,
        center_y=59,
        bbox=(465, 0, 171, 118),
        confidence=0.50,
        composite=True,
    )
    tracker.consecutive = 25
    tracker._last_frame_id = 3
    upper = _detection(514, 0, 126, 61, confidence=0.42)
    lower = _detection(466, 63, 48, 55, confidence=0.56)
    later_gate = _detection(313, 146, 25, 34, confidence=0.40)

    fused = tracker.update(
        [lower, later_gate, upper],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
        allow_tracked_fragment_union=True,
        image_width=640,
        image_height=360,
    )

    assert vq2_module.COURSE_FRAGMENT_UNION_MAX_ASPECT_RATIO == 1.45
    assert (
        vq2_module.COURSE_FRAGMENT_UNION_RIGHT_EDGE_MAX_ASPECT_RATIO
        == 1.48
    )
    assert fused is not None
    assert fused.bbox == (466, 0, 174, 118)
    assert fused.composite
    assert tracker.last_selected_detections == (upper, lower)
    assert tracker.last_selection_mode == "tracked_fragment_union"
    assert tracker.consecutive == 26


def test_gate_tracker_fragment_union_keeps_strict_aspect_away_from_right_edge():
    prior = vq2_module.GateTarget(
        frame_id=3,
        sim_time_ns=30,
        received_monotonic_s=1.03,
        center_x=540,
        center_y=59,
        bbox=(455, 0, 171, 118),
        confidence=0.50,
        composite=True,
    )
    upper = _detection(504, 0, 126, 61, confidence=0.42)
    lower = _detection(456, 63, 48, 55, confidence=0.56)

    assert (
        vq2_module.select_tracked_fragment_union(
            [upper, lower],
            prior_target=prior,
            image_width=640,
            image_height=360,
        )
        is None
    )


def test_gate_tracker_fragment_union_exits_to_live_lower_fragment_without_gap():
    tracker = GateTargetTracker()
    tracker.target = vq2_module.GateTarget(
        frame_id=3,
        sim_time_ns=30,
        received_monotonic_s=1.03,
        center_x=567,
        center_y=66,
        bbox=(494, 0, 146, 133),
        confidence=0.53,
        composite=True,
    )
    tracker.consecutive = 20
    tracker._last_frame_id = 3
    upper = _detection(614, 0, 26, 50, confidence=0.34)
    lower = _detection(498, 68, 56, 66, confidence=0.53)
    later_gate = _detection(308, 173, 32, 41, confidence=0.40)

    fused = tracker.update(
        [lower, later_gate, upper],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
        allow_tracked_fragment_union=True,
        image_width=640,
        image_height=360,
    )
    assert fused is not None
    assert fused.bbox == (498, 0, 142, 134)
    assert fused.composite

    lower_only = _detection(504, 68, 59, 68, confidence=0.51)
    accepted = tracker.update(
        [lower_only, later_gate],
        frame_id=6,
        sim_time_ns=60,
        received_monotonic_s=1.10,
        allow_tracked_fragment_union=True,
        image_width=640,
        image_height=360,
    )
    assert accepted is not None
    assert accepted.bbox == lower_only.bbox
    assert not accepted.composite
    assert tracker.last_selection_mode == "primary"
    assert tracker.consecutive == 22


def test_gate_tracker_fragment_union_rejects_unrelated_top_and_later_gate():
    tracker = GateTargetTracker()
    prior = _detection(466, 0, 137, 119, confidence=0.74)
    for frame_id in range(1, 4):
        tracker.update(
            [prior],
            frame_id=frame_id,
            sim_time_ns=frame_id * 10,
            received_monotonic_s=1.0 + frame_id * 0.01,
        )
    upper = _detection(500, 0, 106, 80, confidence=0.69)
    unrelated = _detection(321, 136, 23, 34, confidence=0.39)

    accepted = tracker.update(
        [upper, unrelated],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
        allow_tracked_fragment_union=True,
        image_width=640,
        image_height=360,
    )

    assert accepted is not None
    assert accepted.bbox == upper.bbox
    assert not accepted.composite
    assert tracker.last_selection_mode == "primary"


def test_course_crossing_never_arms_from_composite_union_geometry():
    composite = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        center_x=320,
        center_y=180,
        bbox=(60, 0, 520, 360),
        confidence=0.8,
        composite=True,
    )

    assert not vq2_module.is_course_gate_crossing_candidate(
        composite,
        acquisition_gate_area=1000,
        control_y=180.0,
    )


def test_pitch_leveling_moves_expected_gate_reference_down():
    reference = gate_vertical_reference_px(174.0, -0.311, -0.10)
    assert reference == pytest.approx(242.5, abs=1.0)


@pytest.mark.parametrize(
    ("elapsed_s", "expected_pitch"),
    ((0.0, -0.31), (0.40, -0.155), (0.80, 0.0), (1.20, 0.0)),
)
def test_offline_gate0_full_lap_pitch_schedule_reaches_bounded_exit_pitch(
    elapsed_s,
    expected_pitch,
):
    assert vq2_module.COURSE_GATE0_EXIT_PITCH_RAD == 0.0
    assert vq2_module.gate0_target_pitch_rad(
        -0.31,
        vq2_module.COURSE_GATE0_EXIT_PITCH_RAD,
        elapsed_s,
    ) == pytest.approx(expected_pitch)


@pytest.mark.parametrize(
    ("spawn_pitch", "exit_pitch", "elapsed_s"),
    ((True, -0.05, 0.0), (-0.31, math.nan, 0.0), (-0.31, -0.11, 0.0), (-0.31, -0.05, -0.01)),
)
def test_gate0_pitch_schedule_rejects_invalid_inputs(
    spawn_pitch,
    exit_pitch,
    elapsed_s,
):
    with pytest.raises(ValueError, match="pitch schedule"):
        vq2_module.gate0_target_pitch_rad(spawn_pitch, exit_pitch, elapsed_s)


def test_clipped_square_gate_center_uses_visible_width():
    target = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        center_x=332,
        center_y=210,
        bbox=(79, 60, 506, 300),
        confidence=0.8,
    )
    assert gate_control_center_y_px(target) == pytest.approx(313.0)

    fully_clipped = vq2_module.GateTarget(
        frame_id=2,
        sim_time_ns=2,
        received_monotonic_s=1.1,
        center_x=334,
        center_y=180,
        bbox=(101, 0, 466, 360),
        confidence=0.8,
    )
    assert gate_control_center_y_px(
        fully_clipped,
        previous_center_y=216.5,
    ) == pytest.approx(216.5)


def test_gate_vertical_thrust_has_position_and_motion_damping():
    assert gate_vertical_thrust(150.0, 0.0) > 0.275
    assert gate_vertical_thrust(210.0, 0.0) < 0.275
    assert gate_vertical_thrust(180.0, 60.0) < 0.275
    assert gate_vertical_thrust(210.0, 300.0) == 0.21
    assert 0.21 <= gate_vertical_thrust(0.0, -999.0) <= 0.32


def test_close_crossing_requires_large_centered_both_edge_clipped_gate():
    target = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        center_x=336,
        center_y=180,
        bbox=(61, 0, 551, 360),
        confidence=0.8,
    )
    assert is_close_gate_crossing_candidate(
        target,
        initial_gate_area=6480,
        control_y=217.0,
    )
    assert not is_close_gate_crossing_candidate(
        target,
        initial_gate_area=6480,
        control_y=270.0,
    )

    insufficient_growth = vq2_module.GateTarget(
        frame_id=2,
        sim_time_ns=2,
        received_monotonic_s=1.1,
        center_x=320,
        center_y=180,
        bbox=(160, 0, 320, 360),
        confidence=0.8,
    )
    assert not is_close_gate_crossing_candidate(
        insufficient_growth,
        initial_gate_area=6480,
        control_y=180.0,
    )

    no_vertical_clip = vq2_module.GateTarget(
        frame_id=3,
        sim_time_ns=3,
        received_monotonic_s=1.2,
        center_x=320,
        center_y=180,
        bbox=(44, 10, 551, 340),
        confidence=0.8,
    )
    assert not is_close_gate_crossing_candidate(
        no_vertical_clip,
        initial_gate_area=6480,
        control_y=180.0,
    )

    off_center = vq2_module.GateTarget(
        frame_id=4,
        sim_time_ns=4,
        received_monotonic_s=1.3,
        # A clipped asymmetric contour can have a centroid substantially away
        # from its image-bounded bbox midpoint.
        center_x=410,
        center_y=180,
        bbox=(120, 0, 520, 360),
        confidence=0.8,
    )
    assert not is_close_gate_crossing_candidate(
        off_center,
        initial_gate_area=6480,
        control_y=180.0,
    )


def test_crossing_wait_requires_authoritative_new_race_status():
    common = {
        "baseline_race_boot_ms": 5998,
        "elapsed_s": 0.10,
    }
    assert crossing_status_decision(
        **common,
        current_race_boot_ms=5998,
        active_gate_index=0,
    ) == "waiting"
    assert crossing_status_decision(
        **common,
        current_race_boot_ms=5998,
        active_gate_index=1,
    ) == "waiting"
    assert crossing_status_decision(
        **common,
        current_race_boot_ms=6249,
        active_gate_index=1,
    ) == "passed"
    assert crossing_status_decision(
        **common,
        current_race_boot_ms=6249,
        active_gate_index=0,
    ) == "not_credited"
    assert crossing_status_decision(
        baseline_race_boot_ms=5998,
        current_race_boot_ms=5998,
        active_gate_index=0,
        elapsed_s=0.40,
    ) == "status_timeout"
    assert crossing_status_decision(
        baseline_race_boot_ms=5998,
        current_race_boot_ms=5998,
        active_gate_index=2,
        elapsed_s=0.01,
    ) == "invalid_gate_index"


def test_full_lap_crossing_requires_fresh_exact_progress_or_finish():
    common = {
        "baseline_race_boot_ms": 6000,
        "expected_gate_index": 3,
        "elapsed_s": 0.10,
    }
    assert vq2_module.full_lap_crossing_status_decision(
        **common,
        current_race_boot_ms=6000,
        active_gate_index=4,
        race_finished=True,
    ) == "waiting"
    assert vq2_module.full_lap_crossing_status_decision(
        **common,
        current_race_boot_ms=6250,
        active_gate_index=4,
        race_finished=False,
    ) == "passed"
    assert vq2_module.full_lap_crossing_status_decision(
        **common,
        current_race_boot_ms=6250,
        active_gate_index=3,
        race_finished=True,
    ) == "finished"
    assert vq2_module.full_lap_crossing_status_decision(
        **common,
        current_race_boot_ms=6250,
        active_gate_index=5,
        race_finished=False,
    ) == "invalid_gate_index"
    assert vq2_module.full_lap_crossing_status_decision(
        **common,
        current_race_boot_ms=6250,
        active_gate_index=3,
        race_finished=False,
    ) == "not_credited"


@pytest.mark.parametrize(
    ("area_px", "expected"),
    (
        (5_999, False),
        (6_000, True),
        (8_000, True),
        (8_001, False),
        (True, False),
    ),
)
def test_full_lap_initial_gate_reference_requires_proved_spawn_area(
    area_px,
    expected,
):
    assert (
        vq2_module.full_lap_initial_gate_reference_is_valid(area_px)
        is expected
    )
    assert vq2_module.FULL_LAP_INITIAL_GATE_MIN_AREA_PX == 6_000
    assert vq2_module.FULL_LAP_INITIAL_GATE_MAX_AREA_PX == 8_000


@pytest.mark.parametrize(
    ("now_ns", "expected_delay_s"),
    ((0, 0.190), (190_000_000, 0.0), (200_000_000, 0.240)),
)
def test_gate0_phase_alignment_targets_packet_lead(now_ns, expected_delay_s):
    delay_s = vq2_module.gate0_phase_alignment_delay_s(
        now_monotonic_ns=now_ns,
        last_race_received_monotonic_ns=1_000_000_000,
    )
    assert delay_s == pytest.approx(expected_delay_s)
    expected_loss_ns = now_ns + round(
        (delay_s + vq2_module.COURSE_GATE0_EXPECTED_TARGET_LOSS_S)
        * 1_000_000_000
    )
    lead_ns = round(vq2_module.COURSE_RACE_PACKET_TARGET_LEAD_S * 1_000_000_000)
    period_ns = round(vq2_module.COURSE_RACE_PACKET_PERIOD_S * 1_000_000_000)
    assert (expected_loss_ns + lead_ns - 1_000_000_000) % period_ns == 0


@pytest.mark.parametrize(
    "overrides",
    (
        {"now_monotonic_ns": True},
        {"last_race_received_monotonic_ns": -1},
        {"expected_target_loss_s": math.nan},
        {"packet_period_s": 0.0},
        {"target_lead_s": 0.250},
    ),
)
def test_gate0_phase_alignment_rejects_invalid_inputs(overrides):
    values = {
        "now_monotonic_ns": 0,
        "last_race_received_monotonic_ns": 1_000_000_000,
        **overrides,
    }
    with pytest.raises(ValueError, match="phase alignment"):
        vq2_module.gate0_phase_alignment_delay_s(**values)


@pytest.mark.parametrize(
    ("last_sent_s", "expected_sleep_s"),
    ((None, None), (0.97, None), (0.99, 0.01)),
    ids=("no-prior-command", "slot-already-open", "wait-remainder"),
)
def test_next_flight_command_slot_waits_from_prior_send_completion(
    monkeypatch,
    last_sent_s,
    expected_sleep_s,
):
    clock = [1.0]
    sleeps = []
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._last_flight_command_sent_s = last_sent_s

    async def advance(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(monotonic=lambda: clock[0]),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=advance),
        )
        ready_s = asyncio.run(runner._wait_for_next_flight_command_slot())

    if expected_sleep_s is None:
        assert sleeps == []
        assert ready_s == 1.0
    else:
        assert sleeps == pytest.approx([expected_sleep_s])
        assert ready_s == pytest.approx(1.0 + expected_sleep_s)
    if last_sent_s is not None:
        assert ready_s - last_sent_s >= vq2_module.CONTROL_PERIOD_S


def test_next_flight_command_slot_retries_a_fractionally_early_wakeup(
    monkeypatch,
):
    clock = [1.0]
    sleeps = []
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._last_flight_command_sent_s = 0.99

    async def advance(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            clock[0] += 0.009
        else:
            clock[0] += seconds

    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(monotonic=lambda: clock[0]),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=advance),
        )
        ready_s = asyncio.run(runner._wait_for_next_flight_command_slot())

    assert sleeps == pytest.approx([0.01, 0.001])
    assert ready_s == pytest.approx(1.01)
    assert ready_s - runner._last_flight_command_sent_s >= (
        vq2_module.CONTROL_PERIOD_S
    )


def test_next_flight_command_slot_waits_for_exact_wire_start_after_coarse_release(
    monkeypatch,
):
    coarse_clock_s = 1.03125
    wire_clock_ns = [116_500_000]
    sleeps = []
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._last_flight_command_sent_s = 1.0
    runner._last_flight_command_started_ns = 100_000_000

    async def advance(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            wire_clock_ns[0] += round(seconds * 1_000_000_000) - 100_000
        else:
            wire_clock_ns[0] += round(seconds * 1_000_000_000)

    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(
                monotonic=lambda: coarse_clock_s,
                perf_counter_ns=lambda: wire_clock_ns[0],
            ),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=advance),
        )
        ready_s = asyncio.run(
            runner._wait_for_next_flight_command_slot()
        )

    assert sleeps == pytest.approx([0.0035, 0.0001])
    assert wire_clock_ns[0] == 120_000_000
    assert ready_s == coarse_clock_s


def test_next_flight_command_slot_rejects_exact_wire_clock_regression(
    monkeypatch,
):
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._last_flight_command_sent_s = 1.0
    runner._last_flight_command_started_ns = 100_000_000
    wire_clock = iter((116_500_000, 116_400_000))

    async def return_early(_seconds):
        return None

    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(
                monotonic=lambda: 1.03125,
                perf_counter_ns=lambda: next(wire_clock),
            ),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=return_early),
        )
        with pytest.raises(
            SafetyAbort,
            match="exact pacing wait returned early",
        ):
            asyncio.run(runner._wait_for_next_flight_command_slot())


@pytest.mark.parametrize("previous_wire_start_ns", (True, -1))
def test_next_flight_command_slot_rejects_invalid_prior_exact_wire_timestamp(
    monkeypatch,
    previous_wire_start_ns,
):
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._last_flight_command_started_ns = previous_wire_start_ns
    monkeypatch.setattr(
        vq2_module,
        "time",
        SimpleNamespace(monotonic=lambda: 1.0),
    )

    with pytest.raises(
        SafetyAbort,
        match="last flight-command wire timestamp is invalid",
    ):
        asyncio.run(runner._wait_for_next_flight_command_slot())


@pytest.mark.parametrize("current_wire_ns", (True, -1, 99_999_999))
def test_next_flight_command_slot_rejects_invalid_current_exact_wire_timestamp(
    monkeypatch,
    current_wire_ns,
):
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._last_flight_command_started_ns = 100_000_000
    monkeypatch.setattr(
        vq2_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: 1.0,
            perf_counter_ns=lambda: current_wire_ns,
        ),
    )

    with pytest.raises(
        SafetyAbort,
        match="current flight-command wire timestamp is invalid",
    ):
        asyncio.run(runner._wait_for_next_flight_command_slot())


@pytest.mark.parametrize(
    "last_sent_s",
    (True, math.nan, math.inf, -math.ulp(1.0), 1.01),
)
def test_next_flight_command_slot_rejects_invalid_prior_timestamp(
    monkeypatch,
    last_sent_s,
):
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._last_flight_command_sent_s = last_sent_s
    monkeypatch.setattr(
        vq2_module,
        "time",
        SimpleNamespace(monotonic=lambda: 1.0),
    )

    with pytest.raises(SafetyAbort, match="last flight-command timestamp"):
        asyncio.run(runner._wait_for_next_flight_command_slot())


@pytest.mark.parametrize("now_s", (True, math.nan, math.inf, -math.ulp(1.0)))
def test_next_flight_command_slot_rejects_invalid_current_timestamp(
    monkeypatch,
    now_s,
):
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    monkeypatch.setattr(
        vq2_module,
        "time",
        SimpleNamespace(monotonic=lambda: now_s),
    )

    with pytest.raises(SafetyAbort, match="current flight-command timestamp"):
        asyncio.run(runner._wait_for_next_flight_command_slot())


def test_next_flight_command_slot_fails_closed_if_wait_returns_early(monkeypatch):
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._last_flight_command_sent_s = 0.99

    async def return_early(_seconds):
        return None

    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(monotonic=lambda: 1.0),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=return_early),
        )
        with pytest.raises(SafetyAbort, match="pacing wait returned early"):
            asyncio.run(runner._wait_for_next_flight_command_slot())


def test_next_flight_command_slot_fails_closed_on_monotonic_regression(
    monkeypatch,
):
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._last_flight_command_sent_s = 0.0
    clock = iter((10.0, 1.0))
    monkeypatch.setattr(
        vq2_module,
        "time",
        SimpleNamespace(monotonic=lambda: next(clock)),
    )

    with pytest.raises(SafetyAbort, match="pacing wait returned early"):
        asyncio.run(runner._wait_for_next_flight_command_slot())


def test_gate0_stage_waits_for_prior_flight_command_slot_before_sampling(
    monkeypatch,
):
    class SampleReached(Exception):
        pass

    clock = [1.0]
    events = []
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    runner._last_flight_command_sent_s = 0.99
    context = vq2_module.StartContext(0.0, -0.31, 320, 180, 6400, 1000)

    async def advance(seconds):
        events.append(("sleep", seconds))
        clock[0] += seconds

    def sample():
        events.append(("sample", clock[0]))
        raise SampleReached

    monkeypatch.setattr(runner, "_sample", sample)
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(monotonic=lambda: clock[0]),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=advance),
        )
        with pytest.raises(SampleReached):
            asyncio.run(runner._run_gate0(context))

    assert events[0][0] == "sleep"
    assert events[0][1] == pytest.approx(0.01)
    assert events[1][0] == "sample"
    assert events[1][1] == pytest.approx(1.01)


def test_offline_course_gate_waits_for_prior_command_slot_before_sampling(
    monkeypatch,
):
    class SampleReached(Exception):
        pass

    clock = [1.0]
    events = []
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1000, 0, -1, 1, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    runner.estimate = _estimate()
    runner._last_flight_command_sent_s = 0.99
    runner.tracker.target = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        center_x=320,
        center_y=180,
        bbox=(280, 140, 80, 80),
        confidence=0.8,
    )
    runner.tracker.consecutive = vq2_module.POST_GATE_REQUIRED_FRAMES

    async def advance(seconds):
        events.append(("sleep", seconds))
        clock[0] += seconds

    def sample():
        events.append(("sample", clock[0]))
        raise SampleReached

    monkeypatch.setattr(runner, "_sample", sample)
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(monotonic=lambda: clock[0]),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=advance),
        )
        with pytest.raises(SampleReached):
            asyncio.run(
                runner._run_course_gate(
                    1,
                    acquisition={"initial_gate_area_px": 6400},
                    lap_started_s=0.0,
                    lap_deadline_s=5.0,
                )
            )

    assert events[0][0] == "sleep"
    assert events[0][1] == pytest.approx(0.01)
    assert events[1][0] == "sample"
    assert events[1][1] == pytest.approx(1.01)


def test_gate0_phase_alignment_holds_bounded_exact_zero(monkeypatch):
    clock = [1.0]
    commands = []
    watchdogs = []
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    adapter.latest_received_race_status = SimpleNamespace(
        validate_integrity=lambda: None,
        ingress=SimpleNamespace(received_monotonic_ns=1_000_000_000, sequence=7),
        race_status=SimpleNamespace(sim_boot_time_ms=1000),
    )
    runner = VQ2Runner(adapter, _FakeVision())

    async def capture(command, **_kwargs):
        commands.append(command)

    async def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(runner, "_sample", lambda: None)
    monkeypatch.setattr(runner, "_watchdog", lambda **kwargs: watchdogs.append(kwargs))
    monkeypatch.setattr(runner, "_send_flight_command", capture)
    monkeypatch.setattr(runner, "_record_tick", lambda *_args: None)
    monkeypatch.setattr(vq2_module, "gate0_phase_alignment_delay_s", lambda **_kwargs: 0.05)
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(
                monotonic=lambda: clock[0],
                perf_counter_ns=lambda: round(clock[0] * 1_000_000_000),
            ),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=advance),
        )
        result = asyncio.run(runner._align_gate0_race_phase())

    assert result["applied"]
    assert result["planned_delay_s"] == pytest.approx(0.05)
    assert result["actual_delay_s"] == pytest.approx(0.05)
    assert result["command_count"] == len(commands) == 3
    assert all(command == AttitudeRateCommand(0.0, 0.0, 0.0, 0.0) for command in commands)
    assert all(item["require_target"] for item in watchdogs)


def test_course_crossing_area_cap_is_attainable_from_observed_gate1_scale():
    target = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        center_x=320,
        center_y=180,
        bbox=(45, 0, 550, 360),
        confidence=0.8,
    )
    assert 25 * (117 * 92) > 640 * 360
    assert vq2_module.is_course_gate_crossing_candidate(
        target,
        acquisition_gate_area=117 * 92,
        control_y=180.0,
    )


def test_untracked_contact_guard_selects_the_live_collision_precursor_only():
    precursor = _detection(229, 90, 183, 151)
    harmless = _detection(232, 262, 66, 95)
    assert vq2_module.COURSE_UNTRACKED_CONTACT_MIN_AREA_PX == 23_040
    assert vq2_module.COURSE_UNTRACKED_CONTACT_MIN_WIDTH_PX == 160
    assert vq2_module.COURSE_UNTRACKED_CONTACT_MIN_HEIGHT_PX == 120
    assert (
        vq2_module.select_untracked_contact_risk(
            [harmless, precursor],
            accepted_target=None,
        )
        is precursor
    )

    accepted = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        center_x=320,
        center_y=180,
        bbox=(229, 90, 183, 151),
        confidence=0.8,
    )
    assert (
        vq2_module.select_untracked_contact_risk(
            [precursor],
            accepted_target=accepted,
        )
        is None
    )
    assert (
        vq2_module.select_untracked_contact_risk(
            [_detection(229, 90, 159, 151)],
            accepted_target=None,
        )
        is None
    )
    masking_non_risk = _detection(200, 80, 159, 180)
    assert masking_non_risk.area > precursor.area
    assert (
        vq2_module.select_untracked_contact_risk(
            [masking_non_risk, precursor],
            accepted_target=None,
        )
        is precursor
    )
    with pytest.raises(ValueError, match="exact 640x360"):
        vq2_module.select_untracked_contact_risk(
            [precursor],
            accepted_target=None,
            image_width=641,
        )


def test_offline_course_gate_aborts_large_raw_detection_rejected_by_tracker(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1_000, 0, -1, 1, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    runner.estimate = _estimate(roll=-0.40, pitch=-0.40)
    runner.tracker.target = vq2_module.GateTarget(
        frame_id=10,
        sim_time_ns=10,
        received_monotonic_s=0.90,
        center_x=629,
        center_y=73,
        bbox=(618, 48, 22, 51),
        confidence=0.8,
    )
    runner.tracker.consecutive = 3
    runner._latest_raw_detections = [_detection(229, 90, 183, 151)]
    runner._latest_accepted_target = None
    runner._latest_detection_frame_id = 11
    runner._latest_detection_frame_sim_ns = 11
    runner._latest_detection_generation = 1
    runner._latest_detection_received_s = 1.0

    monkeypatch.setattr(runner, "_sample", lambda: None)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: 1.0)

    with pytest.raises(SafetyAbort, match="large untracked gate geometry at gate 1"):
        asyncio.run(
            runner._run_course_gate(
                1,
                acquisition={"initial_gate_area_px": 9_000},
                lap_started_s=0.0,
                lap_deadline_s=5.0,
            )
        )
    assert adapter.commands == []


def test_offline_course_gate_aborts_if_vision_generation_changes(monkeypatch):
    clock = [1.0]
    samples = [0]
    commands = []
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1_000, 0, -1, 1, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    runner.estimate = _estimate(roll=-0.10, pitch=-0.20)
    target = vq2_module.GateTarget(
        frame_id=10,
        sim_time_ns=10,
        received_monotonic_s=0.95,
        center_x=320,
        center_y=180,
        bbox=(270, 130, 100, 100),
        confidence=0.8,
    )
    runner.tracker.target = target
    runner.tracker.consecutive = 3
    runner._latest_detection_generation = 1
    runner._latest_detection_frame_id = target.frame_id
    runner._latest_detection_frame_sim_ns = target.sim_time_ns
    runner._latest_detection_received_s = target.received_monotonic_s
    runner._latest_accepted_target = target

    def sample():
        samples[0] += 1
        if samples[0] == 2:
            runner._latest_detection_generation = 2

    async def capture(command, **_kwargs):
        commands.append(command)

    async def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_send_flight_command", capture)
    monkeypatch.setattr(runner, "_record_tick", lambda *_args: None)
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(monotonic=lambda: clock[0]),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=advance),
        )
        with pytest.raises(
            SafetyAbort,
            match="vision generation changed during course gate",
        ):
            asyncio.run(
                runner._run_course_gate(
                    1,
                    acquisition={"initial_gate_area_px": 10_000},
                    lap_started_s=0.0,
                    lap_deadline_s=5.0,
                )
            )
    assert samples[0] == 2
    assert len(commands) == 1


@pytest.mark.parametrize(
    ("normalized_x", "recenter"),
    (
        (True, True),
        (math.nan, True),
        (math.inf, True),
        (0.0, 1),
    ),
)
def test_offline_course_gate_roll_target_rejects_invalid_inputs(
    normalized_x,
    recenter,
):
    with pytest.raises(ValueError, match="course roll target inputs"):
        vq2_module.course_gate_roll_target(
            normalized_x,
            recenter=recenter,
        )


def test_offline_course_gate_roll_target_uses_empirically_convergent_sign():
    assert vq2_module.course_gate_roll_target(0.25, recenter=True) > 0.0
    assert vq2_module.course_gate_roll_target(-0.25, recenter=True) < 0.0
    assert vq2_module.course_gate_roll_target(0.25, recenter=False) > 0.0
    assert vq2_module.course_gate_roll_target(-0.25, recenter=False) < 0.0
    assert vq2_module.course_gate_roll_target(0.0, recenter=True) == 0.0
    assert vq2_module.course_gate_roll_target(0.0, recenter=False) == 0.0


@pytest.mark.parametrize(
    ("recenter", "limit"),
    ((True, 0.05), (False, 0.16)),
    ids=("recenter", "approach"),
)
def test_offline_course_gate_roll_target_uses_phase_specific_clamps(
    recenter,
    limit,
):
    assert vq2_module.COURSE_ROLL_GAIN == 0.25
    assert vq2_module.COURSE_RECENTER_ROLL_GAIN == 0.12
    assert vq2_module.COURSE_RECENTER_ROLL_LIMIT_RAD == 0.05
    assert vq2_module.COURSE_APPROACH_ROLL_LIMIT_RAD == 0.16
    assert (
        vq2_module.course_gate_roll_target(10.0, recenter=recenter)
        == limit
    )
    assert (
        vq2_module.course_gate_roll_target(-10.0, recenter=recenter)
        == -limit
    )


@pytest.mark.parametrize(
    ("entry_pitch", "control_y"),
    ((-0.05, 180.0), (-0.05, 60.0), (-0.05, 300.0)),
)
def test_offline_course_gate_recenter_pitch_is_exact_zero(entry_pitch, control_y):
    assert (
        vq2_module.course_gate_recenter_pitch_target(entry_pitch, control_y)
        == 0.0
    )


@pytest.mark.parametrize(
    ("entry_pitch", "control_y"),
    ((True, 180.0), (math.nan, 180.0), (0.0, True), (0.0, math.inf)),
)
def test_offline_course_gate_recenter_pitch_rejects_invalid_inputs(
    entry_pitch,
    control_y,
):
    with pytest.raises(ValueError, match="recenter pitch inputs"):
        vq2_module.course_gate_recenter_pitch_target(entry_pitch, control_y)


@pytest.mark.parametrize(
    ("elapsed_s", "normalized_x", "control_y"),
    (
        (True, 0.0, 100.0),
        (math.nan, 0.0, 100.0),
        (-math.ulp(1.0), 0.0, 100.0),
        (0.0, True, 100.0),
        (0.0, math.inf, 100.0),
        (0.0, 0.0, True),
        (0.0, 0.0, math.nan),
    ),
)
def test_offline_course_gate_recenter_required_rejects_invalid_inputs(
    elapsed_s,
    normalized_x,
    control_y,
):
    with pytest.raises(ValueError, match="course recenter inputs"):
        vq2_module.course_gate_recenter_required(
            elapsed_s,
            normalized_x,
            control_y,
        )


@pytest.mark.parametrize(
    ("elapsed_s", "expected"),
    (
        (math.nextafter(0.60, 0.0), True),
        (0.60, False),
        (math.nextafter(0.60, math.inf), False),
    ),
    ids=("inside-window", "exact-deadline", "past-deadline"),
)
def test_offline_course_gate_recenter_window_is_hard_and_nonrenewable(
    elapsed_s,
    expected,
):
    assert vq2_module.course_gate_recenter_required(
        elapsed_s,
        0.99,
        -100.0,
    ) is expected


def test_course_recenter_rate_limit_preserves_zero_yaw_and_thrust():
    assert vq2_module.COURSE_RECENTER_MAX_RATE_RAD_S == 0.12
    limited = vq2_module.limit_command_rates(
        AttitudeRateCommand(0.30, -0.30, 0.0, 0.30),
        vq2_module.COURSE_RECENTER_MAX_RATE_RAD_S,
    )
    assert limited == AttitudeRateCommand(
        0.12,
        -0.12,
        0.0,
        0.30,
    )


def test_course_recenter_rate_command_never_amplifies_and_preserves_thrust():
    limited = vq2_module.course_recenter_rate_command(
        AttitudeRateCommand(0.20, -0.11, 0.0, 0.275),
    )
    assert limited == AttitudeRateCommand(0.12, -0.11, 0.0, 0.275)


@pytest.mark.parametrize(
    ("normalized_x", "normalized_x_rate_s", "expected"),
    (
        (0.25, 0.0, -0.06),
        (0.25, 0.40, -0.06),
        (-0.25, -0.40, 0.06),
        (1.0, 0.0, -0.12),
        (-1.0, 0.0, 0.12),
    ),
)
def test_gate1_recenter_roll_target_uses_live_identified_negative_sign(
    normalized_x,
    normalized_x_rate_s,
    expected,
):
    assert vq2_module.GATE1_RECENTER_ROLL_GAIN == -0.24
    assert vq2_module.GATE1_RECENTER_ROLL_RATE_GAIN == 0.0
    assert vq2_module.GATE1_RECENTER_MAX_ROLL_RAD == 0.12
    assert vq2_module.gate1_recenter_roll_target(
        normalized_x,
        normalized_x_rate_s,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("normalized_x", "normalized_x_rate_s"),
    (
        (True, 0.0),
        (0.0, False),
        (math.nan, 0.0),
        (0.0, math.inf),
        (0.0, math.nextafter(4.0, math.inf)),
        (0.0, math.nextafter(-4.0, -math.inf)),
    ),
)
def test_gate1_recenter_roll_target_rejects_unbounded_inputs(
    normalized_x,
    normalized_x_rate_s,
):
    with pytest.raises(ValueError, match="horizontal inputs"):
        vq2_module.gate1_recenter_roll_target(
            normalized_x,
            normalized_x_rate_s,
        )


def test_gate1_recenter_absolute_error_slope_uses_only_strict_fresh_times():
    assert vq2_module.gate1_recenter_absolute_error_slope_px_s(
        [(1.0, 200.0)]
    ) is None
    assert vq2_module.gate1_recenter_absolute_error_slope_px_s(
        [(1.0, 200.0), (1.1, 190.0), (1.2, 180.0)]
    ) == pytest.approx(-100.0)
    with pytest.raises(ValueError, match="increase strictly"):
        vq2_module.gate1_recenter_absolute_error_slope_px_s(
            [(1.0, 200.0), (1.0, 190.0)]
        )


def test_gate1_recenter_candidate_contract_constants_are_exact():
    assert vq2_module.GATE1_RECENTER_DURATION_S == 0.60
    assert vq2_module.GATE1_RECENTER_ROLL_GAIN == -0.24
    assert vq2_module.GATE1_RECENTER_ROLL_RATE_GAIN == 0.0
    assert vq2_module.GATE1_RECENTER_MAX_ROLL_RAD == 0.12
    assert vq2_module.GATE1_RECENTER_MAX_COMMAND_RATE_RAD_S == 0.12
    assert vq2_module.GATE1_RECENTER_THRUST == 0.275
    assert vq2_module.GATE1_RECENTER_TRANSITION_THRUST == 0.275
    assert vq2_module.GATE1_RECENTER_TARGET_PITCH_RAD == 0.10
    assert vq2_module.GATE1_RECENTER_MIN_THRUST == 0.21
    assert vq2_module.GATE1_RECENTER_MAX_THRUST == 0.30
    assert vq2_module.GATE1_RECENTER_CORRIDOR_NORMALIZED_X == 0.35
    assert vq2_module.GATE1_RECENTER_REQUIRED_CORRIDOR_FRAMES == 3
    assert vq2_module.GATE1_RECENTER_DIVERGENCE_PX == 24.0
    assert vq2_module.GATE1_RECENTER_MAX_ABS_X_RATE_NORM_S == 4.0
    assert vq2_module.GATE1_RECENTER_MAX_ABS_ROLL_RAD == 0.15
    assert vq2_module.GATE1_RECENTER_MIN_PITCH_RAD == -0.20
    assert vq2_module.GATE1_RECENTER_MAX_PITCH_RAD == 0.10
    assert vq2_module.GATE1_RECENTER_NO_PASSAGE_MAX_AREA_PX == 23_040
    assert vq2_module.GATE1_RECENTER_NO_PASSAGE_MAX_WIDTH_PX == 160


def test_official_lap_time_uses_finish_ns_minus_start_ms():
    race = RaceStatus(
        sim_boot_time_ms=16_000,
        race_start_boot_time_ms=3_300,
        race_finish_time_ns=15_800_000_000,
        active_gate_index=5,
        last_gate_race_time=12_000_000_000,
    )
    assert VQ2Runner._official_lap_time_s(race) == pytest.approx(12.5)


@pytest.mark.parametrize(
    "bbox",
    [
        (107, 0, 452, 360),
        (75, 0, 518, 360),
        (0, 0, 640, 360),
        (4, 4, 632, 352),
        (75, 20, 518, 340),
    ],
)
def test_post_pass_crossing_residue_catches_large_clipped_remnants(bbox):
    assert is_crossing_residue(_detection(*bbox))


@pytest.mark.parametrize(
    "bbox",
    [
        (127, 16, 410, 344),
        (409, 138, 28, 45),
        (560, 100, 80, 140),
        (280, 0, 80, 360),
        (160, 40, 320, 280),
    ],
)
def test_post_pass_crossing_residue_preserves_plausible_next_gate(bbox):
    assert not is_crossing_residue(_detection(*bbox))


def test_detection_diagnostics_are_strict_json_and_reject_nonfinite_values():
    detection = _detection(409, 138, 28, 45, confidence=math.nan)
    detection.rotation_deg = math.inf
    summary = gate_detection_summary(detection, detector_index=2)

    assert summary["confidence"] is None
    assert summary["rotation_deg"] is None
    assert "nonfinite_confidence" in summary["selector_rejections"]
    json.dumps(summary, allow_nan=False)


def test_attitude_loop_is_finite_clamped_and_never_commands_yaw():
    command = attitude_rate_command(
        _estimate(roll=-0.08, pitch=-0.31),
        target_roll_rad=-0.16,
        target_pitch_rad=-0.31,
        thrust=0.27,
    )
    opposite = attitude_rate_command(
        _estimate(roll=-0.08, pitch=-0.31),
        target_roll_rad=0.08,
        target_pitch_rad=-0.31,
        thrust=0.27,
    )
    assert -0.25 <= command.roll_rate < -0.10
    assert 0.10 < opposite.roll_rate <= 0.25
    assert abs(command.roll_rate) <= 0.25
    assert abs(command.pitch_rate) <= 0.25
    assert command.yaw_rate == 0.0
    assert math.isfinite(command.thrust)


def test_attitude_loop_reaches_intercept_braking_pitch_sooner():
    command = attitude_rate_command(
        _estimate(roll=0.0, pitch=0.0),
        target_roll_rad=0.0,
        target_pitch_rad=vq2_module.MAX_VISUAL_TARGET_PITCH_RAD,
        thrust=0.27,
        intercept_response_authority=1.0,
    )

    assert command.roll_rate == pytest.approx(0.0, abs=1e-12)
    assert 0.14 < command.pitch_rate <= vq2_module.MAX_COMMAND_RATE_RAD_S
    assert command.yaw_rate == 0.0


class _FakeVision:
    def __init__(self):
        self.is_running = False
        self.current_snapshot = None
        self.reset_calls = 0
        self.frames_decoded = 0

    def stop(self):
        self.is_running = False

    def start(self):
        self.is_running = True

    def reset(self):
        self.reset_calls += 1

    def snapshot(self, **_kwargs):
        return self.current_snapshot

    def stats(self):
        return SimpleNamespace(
            frames_decoded=self.frames_decoded,
            duplicate_datagrams=0,
        )


class _FakeAdapter:
    enable_vision = False
    telemetry_mode = "imu"
    fetch_track_on_connect = False
    is_armed = False
    heartbeat_sequence = 1
    heartbeat_age_s = 0.0
    imu_age_s = 0.0
    race_status_age_s = 0.0
    actuator_age_s = 0.0
    latest_telemetry = None
    race_status = None

    def __init__(self):
        self.reset_calls = 0
        self.cleanup_post_reset_disarm_requests = 0
        self.arm_calls = 0
        self.imu_samples = []
        self.collisions = []
        self.commands = []
        self.outbound_receipts = []
        self.collision_generation = 1

    async def reset(self):
        self.reset_calls += 1

    async def reset_calibration_with_boundary(
        self,
        persist_boundary,
        *,
        cleanup_post_reset_disarm=False,
    ):
        assert type(cleanup_post_reset_disarm) is bool
        if cleanup_post_reset_disarm:
            self.cleanup_post_reset_disarm_requests += 1
        collisions = tuple(
            CalibrationCollisionV1(
                id=value["id"],
                threat_level=value["threat_level"],
                impulse=value["impulse"],
            )
            for value in self.collisions
        )
        boundary = SimpleNamespace(
            collisions=collisions,
            collision_stats=self.collision_stats(),
        )
        persist_boundary(boundary)
        self.collisions = []
        self.collision_generation += 1
        self.reset_calls += 1
        return boundary

    async def arm(self):
        self.arm_calls += 1

    async def disarm(self):
        pass

    async def send_attitude_rate(
        self,
        command,
        *,
        call_start_not_before_monotonic_ns=None,
        call_start_deadline_monotonic_ns=None,
    ):
        call_start = vq2_module.time.perf_counter_ns()
        if (
            call_start_not_before_monotonic_ns is not None
            and call_start < call_start_not_before_monotonic_ns
        ):
            raise TimeoutError("fake send began before its pacing window")
        if (
            call_start_deadline_monotonic_ns is not None
            and call_start >= call_start_deadline_monotonic_ns
        ):
            raise TimeoutError("fake send reached its call-start deadline")
        self.commands.append(command)
        self.outbound_receipts.append(
            {
                "schema": "aigp-vq2-attitude-target-outbound/1",
                "host_clock_id": "host-perf-counter",
                "call_start_monotonic_ns": call_start,
                "call_end_monotonic_ns": vq2_module.time.perf_counter_ns(),
                "api": "send_attitude_rate",
                "outcome": "returned",
            }
        )

    def drain_outbound_receipts(self):
        values = self.outbound_receipts
        self.outbound_receipts = []
        return values

    def drain_imu_samples(self):
        samples = self.imu_samples
        self.imu_samples = []
        return samples

    def drain_collisions(self):
        collisions = self.collisions
        self.collisions = []
        return collisions

    def drain_collisions_with_stats(self):
        stats = self.collision_stats()
        collisions = self.drain_collisions()
        return collisions, stats

    def collision_stats(self):
        return MavlinkCollisionStats(
            generation=self.collision_generation,
            handled=len(self.collisions),
            dropped=0,
            high_watermark=len(self.collisions),
            capacity=128,
            buffered=len(self.collisions),
        )


def test_visual_wire_authority_pins_receiver_until_transport_returns():
    vision = VQ2VisionThread()
    old_snapshot = _vision_snapshot(
        frame_id=172,
        sim_time_ns=172_000,
        received_monotonic_s=vq2_module.time.monotonic(),
        generation=7,
    )
    new_snapshot = _vision_snapshot(
        frame_id=173,
        sim_time_ns=173_000,
        received_monotonic_s=vq2_module.time.monotonic(),
        generation=7,
    )
    with vision._data_lock:
        vision._latest_snapshot = old_snapshot

    publication_attempted = threading.Event()
    publication_completed = threading.Event()

    def publish_new_snapshot():
        publication_attempted.set()
        with vision._data_lock:
            vision._latest_snapshot = new_snapshot
        publication_completed.set()

    publisher = threading.Thread(target=publish_new_snapshot)

    class InterleavingAdapter(_FakeAdapter):
        async def send_attitude_rate(
            self,
            command,
            *,
            call_start_not_before_monotonic_ns=None,
            call_start_deadline_monotonic_ns=None,
        ):
            call_start = vq2_module.time.perf_counter_ns()
            assert (
                call_start_not_before_monotonic_ns is None
                or call_start >= call_start_not_before_monotonic_ns
            )
            assert (
                call_start_deadline_monotonic_ns is not None
                and call_start < call_start_deadline_monotonic_ns
            )
            publisher.start()
            assert publication_attempted.wait(1.0)
            publisher.join(0.02)
            assert publisher.is_alive()
            assert vision._latest_snapshot is old_snapshot
            call_end = vq2_module.time.perf_counter_ns()
            self.commands.append(command)
            self.outbound_receipts.append(
                {
                    "schema": "aigp-vq2-attitude-target-outbound/1",
                    "host_clock_id": "host-perf-counter",
                    "call_start_monotonic_ns": call_start,
                    "call_end_monotonic_ns": call_end,
                    "api": "send_attitude_rate",
                    "outcome": "returned",
                }
            )

    adapter = InterleavingAdapter()
    runner = VQ2Runner(adapter, vision)
    # The receiver and tracker establish this clock identity during their
    # normal first exact-frame update; isolate only the wire lease here.
    runner.visual_tracker._time_basis_id = "host-perf-counter"
    expected = vq2_module.VisualCameraFrameToken.from_vision_snapshot(
        old_snapshot
    )
    deadline_ns = vq2_module.time.perf_counter_ns() + 1_000_000_000

    receipt = asyncio.run(
        runner._send_flight_command(
            AttitudeRateCommand(0.0, 0.0, -0.05, 0.275),
            require_wire_receipt=True,
            wire_start_deadline_ns=deadline_ns,
            wire_visual_token=expected,
        )
    )
    publisher.join(1.0)

    assert not publisher.is_alive()
    assert publication_completed.is_set()
    assert vision._latest_snapshot is new_snapshot
    assert receipt is not None
    authority = receipt["visual_receiver_authority"]
    assert authority["frame_token"] == {
        "generation": 7,
        "frame_id": 172,
        "publication_sequence": 172,
        "stream_id": "vq2-camera-udp-5600",
    }
    assert authority[
        "publication_pinned_through_transport_return"
    ] is True
    assert (
        authority["publication_lock_acquired_monotonic_ns"]
        <= authority["call_start_monotonic_ns"]
        <= authority["call_end_monotonic_ns"]
        <= authority["transport_return_monotonic_ns"]
    )


def test_visual_wire_maps_only_typed_publication_lease_acquisition_timeout(
    monkeypatch,
):
    vision = VQ2VisionThread()
    snapshot = _vision_snapshot(
        frame_id=174,
        sim_time_ns=174_000,
        received_monotonic_s=vq2_module.time.monotonic(),
        generation=7,
    )
    with vision._data_lock:
        vision._latest_snapshot = snapshot

    def acquisition_timeout(**_kwargs):
        raise VisionPublicationLeaseAcquisitionTimeout(
            "vision publication lease acquisition deadline was reached"
        )

    monkeypatch.setattr(
        vision,
        "snapshot_publication_lease",
        acquisition_timeout,
    )
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, vision)
    runner.visual_tracker._time_basis_id = "host-perf-counter"
    expected = vq2_module.VisualCameraFrameToken.from_vision_snapshot(
        snapshot
    )

    with pytest.raises(
        SafetyAbort,
        match="publication lease acquisition expired",
    ) as raised:
        asyncio.run(
            runner._send_flight_command(
                AttitudeRateCommand(0.0, 0.0, -0.05, 0.275),
                require_wire_receipt=True,
                wire_start_deadline_ns=(
                    vq2_module.time.perf_counter_ns()
                    + 1_000_000_000
                ),
                wire_visual_token=expected,
            )
        )

    assert isinstance(
        raised.value.__cause__,
        VisionPublicationLeaseAcquisitionTimeout,
    )
    assert adapter.commands == []
    assert adapter.drain_outbound_receipts() == []
    assert runner._last_flight_command_started_ns is None
    assert runner._last_flight_command_sent_s is None


def test_visual_wire_preserves_adapter_pacing_timeout_after_lease_acquisition():
    vision = VQ2VisionThread()
    snapshot = _vision_snapshot(
        frame_id=175,
        sim_time_ns=175_000,
        received_monotonic_s=vq2_module.time.monotonic(),
        generation=7,
    )
    with vision._data_lock:
        vision._latest_snapshot = snapshot

    class EarlyAdapter(_FakeAdapter):
        async def send_attitude_rate(self, _command, **_kwargs):
            raise TimeoutError(
                "attitude-target call began before its pacing window"
            )

    adapter = EarlyAdapter()
    runner = VQ2Runner(adapter, vision)
    runner.visual_tracker._time_basis_id = "host-perf-counter"
    expected = vq2_module.VisualCameraFrameToken.from_vision_snapshot(
        snapshot
    )

    with pytest.raises(
        TimeoutError,
        match="attitude-target call began before its pacing window",
    ):
        asyncio.run(
            runner._send_flight_command(
                AttitudeRateCommand(0.0, 0.0, -0.05, 0.275),
                require_wire_receipt=True,
                wire_start_deadline_ns=(
                    vq2_module.time.perf_counter_ns()
                    + 1_000_000_000
                ),
                wire_visual_token=expected,
            )
        )

    assert adapter.commands == []
    assert adapter.drain_outbound_receipts() == []
    assert runner._last_flight_command_started_ns is None
    assert runner._last_flight_command_sent_s is None
    with vision.snapshot_publication_lease() as latest:
        assert latest is snapshot


def test_visual_wire_never_maps_transport_lease_type_after_acquisition():
    vision = VQ2VisionThread()
    snapshot = _vision_snapshot(
        frame_id=176,
        sim_time_ns=176_000,
        received_monotonic_s=vq2_module.time.monotonic(),
        generation=7,
    )
    with vision._data_lock:
        vision._latest_snapshot = snapshot

    class TransportLeaseTypeAdapter(_FakeAdapter):
        async def send_attitude_rate(self, _command, **_kwargs):
            call_ns = vq2_module.time.perf_counter_ns()
            self.outbound_receipts.append(
                {
                    "schema": "aigp-vq2-attitude-target-outbound/1",
                    "host_clock_id": "host-perf-counter",
                    "call_start_monotonic_ns": call_ns,
                    "call_end_monotonic_ns": call_ns,
                    "api": "send_attitude_rate",
                    "outcome": "raised",
                    "error_type": (
                        "VisionPublicationLeaseAcquisitionTimeout"
                    ),
                }
            )
            raise VisionPublicationLeaseAcquisitionTimeout(
                "transport raised after admission"
            )

    adapter = TransportLeaseTypeAdapter()
    runner = VQ2Runner(adapter, vision)
    runner.visual_tracker._time_basis_id = "host-perf-counter"
    expected = vq2_module.VisualCameraFrameToken.from_vision_snapshot(
        snapshot
    )

    with pytest.raises(
        VisionPublicationLeaseAcquisitionTimeout,
        match="transport raised after admission",
    ):
        asyncio.run(
            runner._send_flight_command(
                AttitudeRateCommand(0.0, 0.0, -0.05, 0.275),
                require_wire_receipt=True,
                wire_start_deadline_ns=(
                    vq2_module.time.perf_counter_ns()
                    + 1_000_000_000
                ),
                wire_visual_token=expected,
            )
        )

    receipts = adapter.drain_outbound_receipts()
    assert len(receipts) == 1
    assert receipts[0]["outcome"] == "raised"
    assert runner._last_flight_command_started_ns is None
    assert runner._last_flight_command_sent_s is None
    with vision.snapshot_publication_lease() as latest:
        assert latest is snapshot


def _received_race_status(
    *,
    gate_index=0,
    race_finish_time_ns=-1,
    received_monotonic_ns=100,
):
    payload = RaceStatusPayloadV1(
        sim_boot_time_ms=1_000,
        race_start_boot_time_ms=0,
        race_finish_time_ns=race_finish_time_ns,
        active_gate_index=gate_index,
        last_gate_race_time=-1,
    )
    return ReceivedRaceStatusV1(
        ingress=MavlinkIngressV1(
            stream_id="vq2-mavlink-udp-14550",
            generation=1,
            sequence=4,
            message_type="RACE_STATUS",
            host_clock_id="host-perf-counter",
            received_monotonic_ns=received_monotonic_ns,
            source_time_value=payload.sim_boot_time_ms,
            source_time_unit="ms",
        ),
        race_status=payload,
    )


def _atomic_race_send_adapter(received):
    clock = iter((200, 201))
    calls = []

    class Mav:
        def set_attitude_target_send(self, *args):
            calls.append(args)

    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
        monotonic_ns=lambda: next(clock),
    )
    adapter._conn = SimpleNamespace(mav=Mav())
    adapter._latest_received_race_status = received
    return adapter, calls


def test_adapter_atomic_race_active_send_starts_once_under_same_gate():
    received = _received_race_status(received_monotonic_ns=100)
    adapter, calls = _atomic_race_send_adapter(received)

    authority = asyncio.run(
        adapter.send_attitude_rate_if_race_active(
            AttitudeRateCommand(0.0, 0.0, -0.04, 0.25),
            expected_active_gate_index=0,
            call_start_deadline_monotonic_ns=300,
        )
    )
    receipts = adapter.drain_outbound_receipts()

    assert len(calls) == 1
    assert len(receipts) == 1
    assert receipts[0].call_start_monotonic_ns == 200
    assert authority == {
        "schema": "aigp-vq2-race-active-send-authority/1",
        "expected_active_gate_index": 0,
        "received_race_status": received.to_primitive(),
    }
    assert (
        authority["received_race_status"]["ingress"][
            "received_monotonic_ns"
        ]
        < receipts[0].call_start_monotonic_ns
    )


@pytest.mark.parametrize(
    "received",
    (
        _received_race_status(gate_index=1),
        _received_race_status(race_finish_time_ns=5_000),
    ),
)
def test_adapter_atomic_race_active_send_refuses_transition_or_finish(
    received,
):
    adapter, calls = _atomic_race_send_adapter(received)

    with pytest.raises(
        RaceActiveBoundaryChangedBeforeWire,
        match="boundary changed before wire",
    ):
        asyncio.run(
            adapter.send_attitude_rate_if_race_active(
                AttitudeRateCommand(0.0, 0.0, -0.04, 0.25),
                expected_active_gate_index=0,
                call_start_deadline_monotonic_ns=300,
            )
        )

    assert calls == []
    assert adapter.drain_outbound_receipts() == []
    assert adapter.outbound_audit().attitude_target == 0


def test_runner_preserves_typed_no_wire_race_boundary_outcome():
    vision = VQ2VisionThread()
    snapshot = _vision_snapshot(
        frame_id=179,
        sim_time_ns=179_000,
        received_monotonic_s=vq2_module.time.monotonic(),
        generation=7,
    )
    with vision._data_lock:
        vision._latest_snapshot = snapshot

    class BoundaryAdapter(_FakeAdapter):
        async def send_attitude_rate_if_race_active(
            self,
            _command,
            *,
            expected_active_gate_index,
            **_kwargs,
        ):
            assert expected_active_gate_index == 0
            raise RaceActiveBoundaryChangedBeforeWire(
                "race-active send boundary changed before wire"
            )

    adapter = BoundaryAdapter()
    runner = VQ2Runner(adapter, vision)
    runner.visual_tracker._time_basis_id = "host-perf-counter"
    expected = vq2_module.VisualCameraFrameToken.from_vision_snapshot(
        snapshot
    )

    with pytest.raises(
        RaceActiveBoundaryChangedBeforeWire,
        match="boundary changed before wire",
    ):
        asyncio.run(
            runner._send_flight_command(
                AttitudeRateCommand(0.0, 0.0, -0.04, 0.25),
                require_wire_receipt=True,
                wire_start_deadline_ns=(
                    vq2_module.time.perf_counter_ns()
                    + 1_000_000_000
                ),
                wire_visual_token=expected,
                wire_race_gate_index=0,
            )
        )

    assert adapter.commands == []
    assert adapter.drain_outbound_receipts() == []


def test_runner_records_atomic_race_authority_with_visual_wire_receipt():
    vision = VQ2VisionThread()
    snapshot = _vision_snapshot(
        frame_id=180,
        sim_time_ns=180_000,
        received_monotonic_s=vq2_module.time.monotonic(),
        generation=7,
    )
    with vision._data_lock:
        vision._latest_snapshot = snapshot

    class GuardedAdapter(_FakeAdapter):
        async def send_attitude_rate_if_race_active(
            self,
            command,
            *,
            expected_active_gate_index,
            **kwargs,
        ):
            await self.send_attitude_rate(command, **kwargs)
            return {
                "schema": "aigp-vq2-race-active-send-authority/1",
                "expected_active_gate_index": (
                    expected_active_gate_index
                ),
                "received_race_status": (
                    _received_race_status(
                        gate_index=expected_active_gate_index,
                        received_monotonic_ns=1,
                    ).to_primitive()
                ),
            }

    adapter = GuardedAdapter()
    runner = VQ2Runner(adapter, vision)
    runner.visual_tracker._time_basis_id = "host-perf-counter"
    expected = vq2_module.VisualCameraFrameToken.from_vision_snapshot(
        snapshot
    )

    receipt = asyncio.run(
        runner._send_flight_command(
            AttitudeRateCommand(0.0, 0.0, -0.04, 0.25),
            require_wire_receipt=True,
            wire_start_deadline_ns=(
                vq2_module.time.perf_counter_ns() + 1_000_000_000
            ),
            wire_visual_token=expected,
            wire_race_gate_index=0,
        )
    )

    assert receipt is not None
    race_authority = receipt["race_active_send_authority"]
    assert race_authority["expected_active_gate_index"] == 0
    assert (
        race_authority["received_race_status"]["ingress"][
            "received_monotonic_ns"
        ]
        <= receipt["call_start_monotonic_ns"]
    )


def _configure_gate1_recenter_candidate(monkeypatch, *, entry_center_x=530):
    clock = [0.0]
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1250, 0, -1, 1, 123)
    runner = VQ2Runner(adapter, _FakeVision())
    runner.estimate = _estimate(roll=0.0, pitch=-0.05)
    runner._gate0_transition_proof = vq2_module.GateTransitionProof(
        pre_gate_race_boot_ms=1000,
        post_gate_race_boot_ms=1250,
        flight_started_monotonic_s=-2.0,
        crossing_started_monotonic_s=-0.4,
        pass_confirmed_monotonic_s=-0.1,
        next_control_deadline_s=0.0,
        vision_generation=1,
        vision_frame_id=100,
        vision_sim_time_ns=1000,
        vision_received_monotonic_s=-0.3,
        pass_rpy_rad=(0.0, -0.05, 0.0),
    )
    entry = vq2_module.GateTarget(
        frame_id=103,
        sim_time_ns=1030,
        received_monotonic_s=0.0,
        center_x=entry_center_x,
        center_y=50,
        bbox=(entry_center_x - 50, 0, 100, 100),
        confidence=0.8,
    )
    runner.tracker.target = entry
    runner.tracker.consecutive = 3
    runner.tracker.last_selection_mode = "primary"
    runner._latest_accepted_target = entry
    runner._latest_detection_generation = 1
    runner._latest_detection_frame_id = entry.frame_id
    runner._latest_detection_frame_sim_ns = entry.sim_time_ns
    runner._latest_detection_received_s = entry.received_monotonic_s
    runner._latest_raw_detections = []
    observation = {
        "gate1_observed": True,
        "frame_count": 3,
        "frames": [
            {
                "frame_id": 101,
                "sim_time_ns": 1010,
                "received_monotonic_s": -0.2,
                "center_px": [524, 52],
                "bbox_xywh_px": [474, 2, 100, 100],
            },
            {
                "frame_id": 102,
                "sim_time_ns": 1020,
                "received_monotonic_s": -0.1,
                "center_px": [527, 51],
                "bbox_xywh_px": [477, 1, 100, 100],
            },
            {
                "frame_id": 103,
                "sim_time_ns": 1030,
                "received_monotonic_s": 0.0,
                "center_px": [entry_center_x, 50],
                "bbox_xywh_px": [entry_center_x - 50, 0, 100, 100],
            },
        ],
        "gate_index": 1,
    }

    async def advance(seconds):
        clock[0] += max(0.0, float(seconds))

    monkeypatch.setattr(
        vq2_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            perf_counter_ns=lambda: round(clock[0] * 1_000_000_000),
        ),
    )
    monkeypatch.setattr(
        vq2_module,
        "asyncio",
        SimpleNamespace(sleep=advance, CancelledError=asyncio.CancelledError),
    )
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    return runner, adapter, observation, clock


def _install_gate1_frame_sequence(
    monkeypatch,
    runner,
    clock,
    centers,
    *,
    generation=1,
    sample_step_s=0.10,
):
    queued = deque(int(value) for value in centers)

    def sample():
        if not queued:
            return
        clock[0] += float(sample_step_s)
        center_x = queued.popleft()
        previous = runner.tracker.target
        assert previous is not None
        target = vq2_module.GateTarget(
            frame_id=previous.frame_id + 1,
            sim_time_ns=previous.sim_time_ns + 10,
            received_monotonic_s=clock[0],
            center_x=center_x,
            center_y=50,
            bbox=(center_x - 50, 0, 100, 100),
            confidence=0.8,
        )
        runner.tracker.target = target
        runner.tracker.consecutive += 1
        runner.tracker.last_selection_mode = "primary"
        runner._latest_accepted_target = target
        runner._latest_detection_generation = generation
        runner._latest_detection_frame_id = target.frame_id
        runner._latest_detection_frame_sim_ns = target.sim_time_ns
        runner._latest_detection_received_s = target.received_monotonic_s

    monkeypatch.setattr(runner, "_sample", sample)


@pytest.mark.parametrize(
    "bbox",
    (
        (370, 0, 160, 100),
        (370, 0, 144, 160),
    ),
    ids=("width-equality", "area-equality"),
)
def test_bounded_gate1_recenter_rejects_no_passage_entry_geometry(
    monkeypatch,
    bbox,
):
    runner, _adapter, observation, _clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    entry = runner.tracker.target
    assert entry is not None
    bounded = replace(entry, bbox=bbox)
    runner.tracker.target = bounded
    runner._latest_accepted_target = bounded
    observation["frames"][-1]["bbox_xywh_px"] = list(bbox)

    with pytest.raises(SafetyAbort, match="no-passage geometry bound"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    summary = runner._gate1_recenter_summary
    assert summary is not None
    assert summary["max_target_area_px"] == bbox[2] * bbox[3]
    assert summary["max_target_width_px"] == bbox[2]
    assert summary["no_passage_max_area_px"] == 23_040
    assert summary["no_passage_max_width_px"] == 160


def test_bounded_gate1_recenter_rejects_hidden_large_raw_geometry(monkeypatch):
    runner, _adapter, observation, _clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    runner._latest_raw_detections = [_detection(470, 0, 160, 144)]

    with pytest.raises(SafetyAbort, match="raw no-passage geometry bound"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))


def test_bounded_gate1_recenter_allows_geometry_just_below_bounds(monkeypatch):
    runner, _adapter, observation, _clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    entry = runner.tracker.target
    assert entry is not None
    below = replace(entry, bbox=(370, 0, 159, 144))
    runner.tracker.target = below
    runner._latest_accepted_target = below
    observation["frames"][-1]["bbox_xywh_px"] = list(below.bbox)

    def change_generation():
        runner._latest_detection_generation = 2

    monkeypatch.setattr(runner, "_sample", change_generation)

    with pytest.raises(SafetyAbort, match="vision generation changed"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))


def test_offline_gate1_recenter_requires_error_decrease_and_three_frame_hold(
    monkeypatch,
):
    runner, adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [430, 420, 410],
    )

    result = asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert result["success"] is False
    assert result["recenter_criteria_met"] is True
    assert result["cleanup_confirmed"] is False
    assert result["outcome"] == "corridor_hold"
    assert result["fresh_control_frame_count"] == 3
    assert result["corridor_hold_frame_count"] == 3
    assert result["entry_abs_horizontal_error_px"] == 210.0
    assert result["final_abs_horizontal_error_px"] == 90.0
    assert result["fresh_abs_horizontal_error_slope_px_s"] < 0.0
    assert result["authoritative_max_gate_index"] == 1
    assert result["contact_safety_outcome"] == "clean"
    assert adapter.commands
    assert all(
        abs(command.roll_rate)
        <= vq2_module.GATE1_RECENTER_MAX_COMMAND_RATE_RAD_S
        and abs(command.pitch_rate)
        <= vq2_module.GATE1_RECENTER_MAX_COMMAND_RATE_RAD_S
        and command.yaw_rate == 0.0
        and command.thrust == vq2_module.GATE1_RECENTER_THRUST
        for command in adapter.commands
    )


def test_configured_gate1_yaw_uses_calibrated_negative_image_feedback(
    monkeypatch,
):
    runner, adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    document = controller_config_module.default_controller_config_mapping()
    document["yaw_control"]["gate1_error_gain"] = -0.12
    document["yaw_control"]["command_rate_cap_rad_s"] = 0.08
    runner.controller_config = (
        controller_config_module.validate_controller_config(document)
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [500, 460, 430, 420, 410],
        sample_step_s=0.07,
    )

    result = asyncio.run(runner._run_bounded_gate1_recenter(observation))

    nonzero_yaw = [
        command.yaw_rate
        for command in adapter.commands
        if command.yaw_rate != 0.0
    ]
    assert result["recenter_criteria_met"] is True
    assert nonzero_yaw
    assert all(-0.08 <= value < 0.0 for value in nonzero_yaw)
    assert result["min_command_yaw_rate_rad_s"] < 0.0
    assert result["max_abs_yaw_excursion_rad"] == 0.0


def test_gate1_yaw_envelope_soft_stops_then_aborts_strictly_over_bound():
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._gate1_yaw_reference_rad = 0.0
    runner.estimate = _estimate(
        yaw=0.046
    )

    excursion, soft_stopped = runner._gate1_yaw_envelope_state(
        phase="test",
    )

    assert excursion == pytest.approx(0.046)
    assert soft_stopped is True

    runner.estimate = _estimate(
        yaw=0.051
    )
    with pytest.raises(SafetyAbort, match="yaw excursion"):
        runner._gate1_yaw_envelope_state(phase="test")


def test_offline_gate1_recenter_wires_bounded_braking_pitch_and_fixed_thrust(
    monkeypatch,
):
    runner, _adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [520],
    )
    observed = []
    original = vq2_module.attitude_rate_command

    def capture_objective(estimate, **kwargs):
        observed.append(dict(kwargs))
        return original(estimate, **kwargs)

    monkeypatch.setattr(
        vq2_module,
        "attitude_rate_command",
        capture_objective,
    )

    with pytest.raises(SafetyAbort):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert observed
    assert all(
        objective["target_pitch_rad"] == 0.10
        and objective["thrust"] == 0.275
        for objective in observed
    )


def test_offline_gate1_recenter_duplicate_frames_do_not_count(monkeypatch):
    runner, _adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    calls = [0]

    def sample():
        calls[0] += 1
        if calls[0] in {2, 4, 6}:
            clock[0] += 0.10
            center_x = {2: 430, 4: 420, 6: 410}[calls[0]]
            previous = runner.tracker.target
            assert previous is not None
            target = replace(
                previous,
                frame_id=previous.frame_id + 1,
                sim_time_ns=previous.sim_time_ns + 10,
                received_monotonic_s=clock[0],
                center_x=center_x,
                bbox=(center_x - 50, 0, 100, 100),
            )
            runner.tracker.target = target
            runner.tracker.consecutive += 1
            runner.tracker.last_selection_mode = "primary"
            runner._latest_accepted_target = target
            runner._latest_detection_frame_id = target.frame_id
            runner._latest_detection_frame_sim_ns = target.sim_time_ns
            runner._latest_detection_received_s = target.received_monotonic_s

    monkeypatch.setattr(runner, "_sample", sample)

    result = asyncio.run(runner._run_bounded_gate1_recenter(observation))

    # One commandless sample rechecks authority after the paced cleanup slot.
    assert calls[0] == 7
    assert result["fresh_control_frame_count"] == 3
    assert result["corridor_hold_frame_count"] == 3


def test_offline_gate1_recenter_rejects_transient_improvement_with_bad_finish(
    monkeypatch,
):
    runner, _adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch,
        entry_center_x=430,
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [400, 425, 432],
    )

    with pytest.raises(SafetyAbort):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    summary = runner._gate1_recenter_summary
    assert summary is not None
    assert summary["recenter_criteria_met"] is False
    assert summary["final_abs_horizontal_error_px"] == 112.0
    assert summary["entry_abs_horizontal_error_px"] == 110.0
    assert summary["fresh_abs_horizontal_error_slope_px_s"] > 0.0


@pytest.mark.parametrize(
    ("third_center_x", "expected_reason"),
    (
        (445, "diverged by more than 24px"),
        (444, "target lost primary fresh-frame authority"),
    ),
    ids=("strictly-over-bound", "exact-bound-allowed"),
)
def test_offline_gate1_recenter_divergence_boundary_is_strict(
    monkeypatch,
    third_center_x,
    expected_reason,
):
    runner, _adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch,
        entry_center_x=420,
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [430, 440, third_center_x],
    )

    with pytest.raises(SafetyAbort, match=expected_reason):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert runner._gate1_recenter_summary is not None
    assert runner._gate1_recenter_summary["success"] is False
    assert runner._gate1_recenter_summary["fresh_control_frame_count"] == 3


def test_offline_gate1_recenter_never_sends_at_or_after_hard_deadline(
    monkeypatch,
):
    runner, _adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )

    def refresh_same_error():
        clock[0] += 0.002
        previous = runner.tracker.target
        assert previous is not None
        target = replace(
            previous,
            frame_id=previous.frame_id + 1,
            sim_time_ns=previous.sim_time_ns + 10,
            received_monotonic_s=clock[0],
        )
        runner.tracker.target = target
        runner.tracker.consecutive += 1
        runner.tracker.last_selection_mode = "primary"
        runner._latest_accepted_target = target
        runner._latest_detection_frame_id = target.frame_id
        runner._latest_detection_frame_sim_ns = target.sim_time_ns
        runner._latest_detection_received_s = target.received_monotonic_s

    monkeypatch.setattr(runner, "_sample", refresh_same_error)
    send_times = []
    original_send = runner._send_flight_command

    async def record_send(command, **kwargs):
        send_times.append(clock[0])
        return await original_send(command, **kwargs)

    monkeypatch.setattr(runner, "_send_flight_command", record_send)

    with pytest.raises(SafetyAbort, match="hard 0.60s window expired"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert send_times
    assert all(
        sent < vq2_module.GATE1_RECENTER_DURATION_S
        for sent in send_times
    )
    assert all(
        sent
        < vq2_module.GATE1_RECENTER_DURATION_S
        - vq2_module.CONTROL_PERIOD_S
        for sent in send_times
    )
    assert clock[0] >= (
        vq2_module.GATE1_RECENTER_DURATION_S
        - vq2_module.CONTROL_PERIOD_S
    )


def test_offline_gate1_recenter_wire_deadline_rejects_delayed_send(
    monkeypatch,
):
    runner, adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    entry = runner.tracker.target
    assert entry is not None

    def sample():
        clock[0] = (
            vq2_module.GATE1_RECENTER_DURATION_S
            - vq2_module.CONTROL_PERIOD_S
            - 0.006
        )
        refreshed = replace(
            entry,
            frame_id=entry.frame_id + 1,
            sim_time_ns=entry.sim_time_ns + 10,
            received_monotonic_s=clock[0],
        )
        runner.tracker.target = refreshed
        runner.tracker.consecutive += 1
        runner.tracker.last_selection_mode = "primary"
        runner._latest_accepted_target = refreshed
        runner._latest_detection_frame_id = refreshed.frame_id
        runner._latest_detection_frame_sim_ns = refreshed.sim_time_ns
        runner._latest_detection_received_s = refreshed.received_monotonic_s

    original_send = runner._send_flight_command

    async def delayed_send(command, **kwargs):
        clock[0] += 0.030
        return await original_send(command, **kwargs)

    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_send_flight_command", delayed_send)

    with pytest.raises(SafetyAbort, match="dispatch failed closed"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert adapter.commands == []


def test_offline_gate1_recenter_wire_deadline_survives_start_clock_preemption(
    monkeypatch,
):
    runner, adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    entry = runner.tracker.target
    assert entry is not None
    first_perf_sample = [True]

    def preempted_perf_counter_ns():
        sampled_ns = round(clock[0] * 1_000_000_000)
        if first_perf_sample[0]:
            first_perf_sample[0] = False
            clock[0] += 0.030
        return sampled_ns

    monkeypatch.setattr(
        vq2_module.time,
        "perf_counter_ns",
        preempted_perf_counter_ns,
    )

    def sample():
        clock[0] = (
            vq2_module.GATE1_RECENTER_DURATION_S
            - vq2_module.CONTROL_PERIOD_S
            - 0.006
        )
        refreshed = replace(
            entry,
            frame_id=entry.frame_id + 1,
            sim_time_ns=entry.sim_time_ns + 10,
            received_monotonic_s=clock[0],
        )
        runner.tracker.target = refreshed
        runner.tracker.consecutive += 1
        runner.tracker.last_selection_mode = "primary"
        runner._latest_accepted_target = refreshed
        runner._latest_detection_frame_id = refreshed.frame_id
        runner._latest_detection_frame_sim_ns = refreshed.sim_time_ns
        runner._latest_detection_received_s = refreshed.received_monotonic_s

    original_send = runner._send_flight_command

    async def delayed_send(command, **kwargs):
        clock[0] += 0.030
        return await original_send(command, **kwargs)

    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_send_flight_command", delayed_send)

    with pytest.raises(SafetyAbort, match="dispatch failed closed"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert clock[0] >= vq2_module.GATE1_RECENTER_DURATION_S
    assert adapter.commands == []


@pytest.mark.parametrize("failure_mode", ("missing_receipt", "send_raised"))
def test_offline_gate1_recenter_reserves_cleanup_slot_after_uncertain_dispatch(
    monkeypatch,
    failure_mode,
):
    runner, adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [500],
    )
    dispatch_times = []
    dispatch_terminal_times = []

    async def uncertain_send(
        command,
        *,
        call_start_not_before_monotonic_ns=None,
        call_start_deadline_monotonic_ns=None,
    ):
        del call_start_not_before_monotonic_ns, call_start_deadline_monotonic_ns
        dispatch_times.append(clock[0])
        clock[0] += 0.015
        dispatch_terminal_times.append(clock[0])
        if failure_mode == "send_raised":
            raise TimeoutError("wire outcome is unknown")
        adapter.commands.append(command)

    monkeypatch.setattr(adapter, "send_attitude_rate", uncertain_send)

    with pytest.raises(SafetyAbort, match="dispatch failed closed"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert dispatch_times
    assert dispatch_terminal_times
    assert clock[0] >= (
        dispatch_terminal_times[-1] + vq2_module.CONTROL_PERIOD_S
    )


def test_offline_gate1_recenter_rechecks_target_freshness_at_send(monkeypatch):
    runner, adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [500],
    )
    watchdog_calls = [0]

    def delayed_watchdog(**_kwargs):
        watchdog_calls[0] += 1
        if watchdog_calls[0] == 2:
            clock[0] += vq2_module.MAX_VISION_AGE_S + 0.001

    monkeypatch.setattr(runner, "_watchdog", delayed_watchdog)

    with pytest.raises(SafetyAbort, match="authority changed before command send"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert adapter.commands == []


def test_offline_gate1_recenter_reserves_cleanup_slot_after_unexpected_error(
    monkeypatch,
):
    runner, adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [500],
    )
    error_times = []

    def fail_after_send(*_args, **_kwargs):
        error_times.append(clock[0])
        raise RuntimeError("post-send recorder failed")

    monkeypatch.setattr(runner, "_record_tick", fail_after_send)

    with pytest.raises(RuntimeError, match="post-send recorder failed"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert adapter.commands
    assert error_times
    assert clock[0] >= error_times[-1] + vq2_module.CONTROL_PERIOD_S
    assert runner._gate1_recenter_summary is not None
    assert runner._gate1_recenter_summary["outcome"] == "unexpected_error"
    assert (
        runner._gate1_recenter_summary["contact_safety_outcome"]
        == "infrastructure_error"
    )


def test_offline_gate1_recenter_does_not_delay_late_error_after_proved_send(
    monkeypatch,
):
    runner, adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [500],
    )
    error_times = []

    def fail_well_after_send(*_args, **_kwargs):
        clock[0] += 0.050
        error_times.append(clock[0])
        raise RuntimeError("late post-send failure")

    monkeypatch.setattr(runner, "_record_tick", fail_well_after_send)

    with pytest.raises(RuntimeError, match="late post-send failure"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert adapter.commands
    assert error_times
    assert clock[0] == pytest.approx(error_times[-1])


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        ("generation", "vision generation changed"),
        ("race", "gate index changed"),
        ("target", "primary gate target lost"),
        ("composite", "primary fresh-frame authority"),
    ),
)
def test_offline_gate1_recenter_aborts_on_authority_loss(
    monkeypatch,
    mutation,
    expected_reason,
):
    runner, _adapter, observation, _clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )

    def sample():
        if mutation == "generation":
            runner._latest_detection_generation = 2
        elif mutation == "race":
            runner.adapter.race_status = RaceStatus(1500, 0, -1, 2, 123)
        elif mutation == "target":
            runner._latest_accepted_target = None
        else:
            assert runner._latest_accepted_target is not None
            composite = replace(
                runner._latest_accepted_target,
                composite=True,
            )
            runner._latest_accepted_target = composite
            runner.tracker.target = composite

    monkeypatch.setattr(runner, "_sample", sample)

    with pytest.raises(SafetyAbort, match=expected_reason):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))


def test_offline_gate1_recenter_rejects_unsafe_entry_attitude(monkeypatch):
    runner, _adapter, observation, _clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    runner.estimate = _estimate(
        roll=math.nextafter(
            vq2_module.GATE1_RECENTER_MAX_ABS_ROLL_RAD,
            math.inf,
        ),
        pitch=-0.05,
    )

    with pytest.raises(SafetyAbort, match="entry attitude"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))


def test_offline_gate1_recenter_requires_same_accepted_entry_object(
    monkeypatch,
):
    runner, _adapter, observation, _clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    assert runner._latest_accepted_target is not None
    runner._latest_accepted_target = replace(runner._latest_accepted_target)

    with pytest.raises(SafetyAbort, match="fresh primary target"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))


def test_offline_gate1_recenter_binds_handoff_to_transition_watermark(
    monkeypatch,
):
    runner, _adapter, observation, _clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    observation["frames"][0]["received_monotonic_s"] = -0.4

    with pytest.raises(SafetyAbort, match="did not begin after the proved transition"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))


@pytest.mark.parametrize("cleanup_change", ("target_lost", "gate_advanced"))
def test_offline_gate1_recenter_rechecks_authority_before_cleanup(
    monkeypatch,
    cleanup_change,
):
    runner, _adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [430, 420, 410],
    )
    advance_frame = runner._sample
    sample_calls = [0]

    def sample_then_change_authority():
        sample_calls[0] += 1
        advance_frame()
        if sample_calls[0] == 4:
            if cleanup_change == "target_lost":
                runner._latest_accepted_target = None
            else:
                runner.adapter.race_status = RaceStatus(1500, 0, -1, 2, 123)

    monkeypatch.setattr(runner, "_sample", sample_then_change_authority)

    with pytest.raises(SafetyAbort, match="authority changed before cleanup"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    summary = runner._gate1_recenter_summary
    assert summary is not None
    assert summary["recenter_criteria_met"] is False
    assert summary["authoritative_max_gate_index"] == (
        2 if cleanup_change == "gate_advanced" else 1
    )


def test_offline_gate1_recenter_rejects_regressed_cleanup_frame(monkeypatch):
    runner, _adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [430, 420, 410],
    )
    advance_frame = runner._sample
    sample_calls = [0]

    def sample_then_regress():
        sample_calls[0] += 1
        advance_frame()
        if sample_calls[0] == 4:
            current = runner._latest_accepted_target
            assert current is not None
            regressed = replace(
                current,
                frame_id=50,
                sim_time_ns=500,
                received_monotonic_s=clock[0],
            )
            runner.tracker.target = regressed
            runner._latest_accepted_target = regressed
            runner._latest_detection_frame_id = regressed.frame_id
            runner._latest_detection_frame_sim_ns = regressed.sim_time_ns
            runner._latest_detection_received_s = regressed.received_monotonic_s

    monkeypatch.setattr(runner, "_sample", sample_then_regress)

    with pytest.raises(SafetyAbort, match="authority changed before cleanup"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert runner._gate1_recenter_summary is not None
    assert runner._gate1_recenter_summary["recenter_criteria_met"] is False


def test_offline_gate1_recenter_reports_and_rechecks_final_cleanup_frame(
    monkeypatch,
):
    runner, _adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [430, 420, 410, 450],
    )
    advance_frame = runner._sample
    sample_calls = [0]

    def sample_with_larger_final_target():
        sample_calls[0] += 1
        advance_frame()
        if sample_calls[0] == 4:
            current = runner._latest_accepted_target
            assert current is not None
            larger = replace(current, bbox=(380, 0, 140, 110))
            runner.tracker.target = larger
            runner._latest_accepted_target = larger

    monkeypatch.setattr(runner, "_sample", sample_with_larger_final_target)

    with pytest.raises(SafetyAbort, match="criteria changed before cleanup"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    summary = runner._gate1_recenter_summary
    assert summary is not None
    assert summary["recenter_criteria_met"] is False
    assert summary["final_abs_horizontal_error_px"] == 130.0
    assert summary["max_target_width_px"] == 140


def test_bounded_gate1_recenter_rechecks_raw_geometry_before_send(monkeypatch):
    runner, adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [500],
    )
    watchdog_calls = [0]

    def expose_large_raw_geometry(**_kwargs):
        watchdog_calls[0] += 1
        if watchdog_calls[0] == 2:
            runner._latest_raw_detections = [
                _detection(470, 0, 160, 144)
            ]

    monkeypatch.setattr(runner, "_watchdog", expose_large_raw_geometry)

    with pytest.raises(SafetyAbort, match="raw no-passage geometry bound"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert adapter.commands == []


def test_bounded_gate1_recenter_rechecks_geometry_before_cleanup(monkeypatch):
    runner, _adapter, observation, clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    _install_gate1_frame_sequence(
        monkeypatch,
        runner,
        clock,
        [430, 420, 410, 405],
    )
    advance_frame = runner._sample
    sample_calls = [0]

    def sample_with_close_cleanup_target():
        sample_calls[0] += 1
        advance_frame()
        if sample_calls[0] == 4:
            current = runner._latest_accepted_target
            assert current is not None
            close = replace(current, bbox=(320, 0, 160, 100))
            runner.tracker.target = close
            runner._latest_accepted_target = close

    monkeypatch.setattr(runner, "_sample", sample_with_close_cleanup_target)

    with pytest.raises(SafetyAbort, match="no-passage geometry bound"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert runner._gate1_recenter_summary is not None
    assert runner._gate1_recenter_summary["recenter_criteria_met"] is False


def test_offline_gate1_recenter_classifies_stream_watchdog_as_safety_abort(
    monkeypatch,
):
    runner, _adapter, observation, _clock = _configure_gate1_recenter_candidate(
        monkeypatch
    )
    monkeypatch.setattr(runner, "_sample", lambda: None)

    def fail_watchdog(**_kwargs):
        raise SafetyAbort("IMU timestamp not advancing")

    monkeypatch.setattr(runner, "_watchdog", fail_watchdog)

    with pytest.raises(SafetyAbort, match="IMU timestamp"):
        asyncio.run(runner._run_bounded_gate1_recenter(observation))

    assert runner._gate1_recenter_summary is not None
    assert (
        runner._gate1_recenter_summary["contact_safety_outcome"]
        == "safety_abort"
    )


def test_offline_gate1_recenter_composition_uses_proved_default_stages(
    monkeypatch,
):
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)
    calls = []
    gate0 = {"gate0_passed": True}
    observation = {"gate1_observed": True}
    recenter = {"success": True}

    async def run_gate0(observed_context, **kwargs):
        calls.append(("gate0", observed_context, kwargs))
        return gate0

    async def observe_gate1(observed_gate0, **kwargs):
        calls.append(("observe", observed_gate0, kwargs))
        return observation

    async def run_recenter(observed, **kwargs):
        calls.append(("recenter", observed, kwargs))
        return recenter

    monkeypatch.setattr(runner, "_run_gate0", run_gate0)
    monkeypatch.setattr(runner, "_observe_gate1", observe_gate1)
    monkeypatch.setattr(runner, "_run_bounded_gate1_recenter", run_recenter)

    result = asyncio.run(runner._run_gate1_recenter_candidate(context))

    assert calls == [
        (
            "gate0",
            context,
            {
                "crossing_hold_thrust": 0.275,
                "course_line_preturn": True,
                "course_line_exit_counterroll_enabled": True,
            },
        ),
        (
            "observe",
            gate0,
            {"hold_thrust": 0.275},
        ),
        ("recenter", observation, {}),
    ]
    assert result == {
        "gate0": gate0,
        "gate1_observation": observation,
        "gate1_recenter": recenter,
    }


@pytest.mark.parametrize(
    "minimum_thrust",
    (True, math.nan, 0.20, math.nextafter(0.32, math.inf)),
)
def test_gate0_minimum_thrust_rejects_invalid_bounds_before_sampling(
    monkeypatch,
    minimum_thrust,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("invalid minimum thrust reached flight sampling"),
    )

    with pytest.raises(ValueError, match="minimum.*thrust"):
        asyncio.run(
            runner._run_gate0(context, minimum_thrust=minimum_thrust)
        )

    assert adapter.commands == []


def test_runner_uses_bounded_real_frame_occlusion_association_horizon():
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())

    assert (
        runner.visual_tracker.config.max_association_gap_ns
        == vq2_module.VISUAL_TRACKER_MAX_ASSOCIATION_GAP_NS
        == 500_000_000
    )
    assert runner.visual_tracker.config.max_missed_frames == 12


@pytest.mark.parametrize(
    "boost_until_s",
    (
        True,
        math.nan,
        math.nextafter(0.45, 0.0),
        math.nextafter(1.0, math.inf),
    ),
)
def test_gate0_boost_duration_rejects_invalid_bounds_before_sampling(
    monkeypatch,
    boost_until_s,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("invalid boost duration reached flight sampling"),
    )

    with pytest.raises(ValueError, match="boost.*duration"):
        asyncio.run(
            runner._run_gate0(context, boost_until_s=boost_until_s)
        )

    assert adapter.commands == []


def test_visual_gate0_launch_duration_must_match_hashed_lifecycle(
    monkeypatch,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(
        0.0,
        -0.31,
        322,
        174,
        6400,
        1000,
    )
    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("launch-config drift reached flight sampling"),
    )

    with pytest.raises(ValueError, match="hashed lifecycle"):
        asyncio.run(
            runner._run_gate0(
                context,
                boost_until_s=0.50,
                visual_next_gate_blend=True,
            )
        )

    assert adapter.commands == []


@pytest.mark.parametrize(
    "crossing_hold_thrust",
    (True, math.nan, 0.20, 0.30),
)
def test_gate0_crossing_hold_thrust_rejects_unadmitted_values_before_sampling(
    monkeypatch,
    crossing_hold_thrust,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("invalid crossing thrust reached flight sampling"),
    )

    with pytest.raises(ValueError, match="crossing hold thrust"):
        asyncio.run(
            runner._run_gate0(
                context,
                crossing_hold_thrust=crossing_hold_thrust,
            )
        )

    assert adapter.commands == []


@pytest.mark.parametrize("observe_course_line", (None, 0, 1, 0.0, "true"))
def test_gate0_course_line_observation_rejects_non_bool_before_sampling(
    monkeypatch,
    observe_course_line,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("invalid observation option reached flight sampling"),
    )

    with pytest.raises(ValueError, match="course.*line.*observation"):
        asyncio.run(
            runner._run_gate0(
                context,
                observe_course_line=observe_course_line,
            )
        )

    assert adapter.commands == []


@pytest.mark.parametrize("course_line_preturn", (None, 0, 1, 0.0, "true"))
def test_gate0_course_line_preturn_rejects_non_bool_before_sampling(
    monkeypatch,
    course_line_preturn,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("invalid preturn option reached flight sampling"),
    )

    with pytest.raises(ValueError, match="preturn"):
        asyncio.run(
            runner._run_gate0(
                context,
                course_line_preturn=course_line_preturn,
            )
        )

    assert adapter.commands == []


@pytest.mark.parametrize(
    "course_line_exit_counterroll_enabled",
    (None, 0, 1, 0.0, "true"),
)
def test_gate0_course_line_exit_counterroll_rejects_non_bool_before_sampling(
    monkeypatch,
    course_line_exit_counterroll_enabled,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("invalid counter-roll option reached flight sampling"),
    )

    with pytest.raises(ValueError, match="exit-counterroll flag"):
        asyncio.run(
            runner._run_gate0(
                context,
                course_line_exit_counterroll_enabled=(
                    course_line_exit_counterroll_enabled
                ),
            )
        )

    assert adapter.commands == []


def test_gate0_course_line_exit_counterroll_requires_preturn_before_sampling(
    monkeypatch,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("invalid counter-roll combination reached sampling"),
    )

    with pytest.raises(ValueError, match="requires preturn"):
        asyncio.run(
            runner._run_gate0(
                context,
                course_line_exit_counterroll_enabled=True,
            )
        )

    assert adapter.commands == []


@pytest.mark.parametrize(
    "visual_next_gate_blend",
    (None, 0, 1, 0.0, "true"),
)
def test_gate0_visual_next_gate_blend_rejects_non_bool_before_sampling(
    monkeypatch,
    visual_next_gate_blend,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("invalid visual blend option reached sampling"),
    )

    with pytest.raises(ValueError, match="visual next-gate blend flag"):
        asyncio.run(
            runner._run_gate0(
                context,
                visual_next_gate_blend=visual_next_gate_blend,
            )
        )

    assert adapter.commands == []


@pytest.mark.parametrize(
    "retired_course_line_option",
    (
        {"observe_course_line": True},
        {"course_line_preturn": True},
        {
            "course_line_preturn": True,
            "course_line_exit_counterroll_enabled": True,
        },
    ),
)
def test_gate0_visual_blend_rejects_retired_course_line_authority_before_sampling(
    monkeypatch,
    retired_course_line_option,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("mixed navigation authority reached sampling"),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        asyncio.run(
            runner._run_gate0(
                context,
                visual_next_gate_blend=True,
                **retired_course_line_option,
            )
        )

    assert adapter.commands == []


def test_gate0_visual_blend_requires_zero_crossing_hold_before_sampling(
    monkeypatch,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    monkeypatch.setattr(
        runner,
        "_sample",
        lambda: pytest.fail("nonzero visual crossing hold reached sampling"),
    )

    with pytest.raises(ValueError, match="requires exact-zero crossing"):
        asyncio.run(
            runner._run_gate0(
                context,
                visual_next_gate_blend=True,
                crossing_hold_thrust=0.275,
            )
        )

    assert adapter.commands == []


@pytest.mark.parametrize(
    (
        "withdrawal_exception",
        "passage_violation_kind",
        "withdrawal_reason",
        "expected_suspended_frames",
    ),
    (
        (
            vq2_module.VisualApproachCurrentGeometryUnavailable,
            None,
            "current_aperture_geometry_unavailable",
            0,
        ),
        (
            vq2_module.VisualApproachPassageSafetyUnavailable,
            "transient_projected_vertical",
            "crossing_candidate_armed",
            1,
        ),
        (
            vq2_module.VisualApproachPassageSafetyUnavailable,
            "hard_apparent_scale",
            "passage_safety_hard_violation",
            0,
        ),
        (
            vq2_module.VisualApproachPassageSafetyUnavailable,
            "transient_wall",
            "passage_safety_suspension_wall_duration_exhausted",
            1,
        ),
        (
            vq2_module.VisualApproachPassageSafetyUnavailable,
            "transient_resume",
            "crossing_candidate_armed",
            1,
        ),
        (
            vq2_module.VisualApproachPassageSafetyUnavailable,
            "transient_total_wall",
            "passage_safety_suspension_wall_duration_exhausted",
            3,
        ),
    ),
)
def test_gate0_visual_blend_hard_withdraws_geometry_but_suspends_passage(
    monkeypatch,
    withdrawal_exception,
    passage_violation_kind,
    withdrawal_reason,
    expected_suspended_frames,
):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    vision = _FakeVision()
    vision.current_snapshot = _vision_snapshot(
        frame_id=500,
        received_monotonic_s=0.20,
    )
    runner = VQ2Runner(adapter, vision)
    runner.estimate = _estimate(pitch=-0.05)
    cumulative_wall_case = (
        passage_violation_kind == "transient_total_wall"
    )
    publication_base = 100 if cumulative_wall_case else 1
    clock = [1.0 if cumulative_wall_case else 0.0]
    flight_clock_origin = clock[0]
    sample_count = [0]
    sent_commands = []
    confirmation_commands = []
    bootstrap_attitude_targets = []
    bootstrap_commands = []
    yaw_checks = []
    current_track_id = "current-track"
    next_track_id = "next-track"

    graph = SimpleNamespace(
        latest_snapshot=SimpleNamespace(
            current_track_id=current_track_id,
            current_gate_index=0,
            authority_usable=True,
            tracker_frame_sequence=0,
            latest_camera_token=vq2_module.VisualCameraFrameToken(
                generation=1,
                frame_id=100,
                publication_sequence=publication_base,
                stream_id="vq2-camera",
            ),
        )
    )
    runner.visual_gate_graph = graph
    runner.tracker.target = vq2_module.GateTarget(
        frame_id=100,
        sim_time_ns=100,
        received_monotonic_s=flight_clock_origin,
        center_x=320,
        center_y=180,
        bbox=(240, 120, 160, 120),
        confidence=0.9,
    )
    runner.tracker.consecutive = 3

    class FakeApproach:
        def __init__(
            self,
            expected_current_track_id,
            gate_index,
            _tuning,
            *,
            next_gate_blend,
        ):
            assert expected_current_track_id == current_track_id
            assert gate_index == 0
            assert next_gate_blend == pytest.approx(0.35)
            self.next_gate_blend = next_gate_blend
            self.calls = 0

        def observe(
            self,
            _snapshot,
            _tracker,
            *,
            now_monotonic_s,
            segment_elapsed_s,
            segment_yaw_excursion_rad,
        ):
            assert segment_elapsed_s >= 0.0
            assert abs(segment_yaw_excursion_rad) <= (
                vq2_module.VISUAL_ALIGN_MAX_YAW_EXCURSION_RAD
            )
            self.calls += 1
            if self.calls == 4:
                if passage_violation_kind is None:
                    raise withdrawal_exception(
                        "optional pre-pass authority is unavailable"
                    )
                if (
                    passage_violation_kind
                    in {
                        "transient_projected_vertical",
                        "transient_resume",
                        "transient_total_wall",
                        "transient_wall",
                    }
                ):
                    violation = (
                        "current_projected_vertical",
                        -0.283,
                        -0.28,
                        0.003,
                    )
                else:
                    assert passage_violation_kind == "hard_apparent_scale"
                    violation = (
                        "current_apparent_scale",
                        0.56,
                        0.55,
                        0.01,
                    )
                raise withdrawal_exception(
                    "optional pre-pass authority is unavailable",
                    violation_codes=(violation[0],),
                    violation_evidence=(violation,),
                    camera_observation_monotonic_s=now_monotonic_s,
                )
            token = ServoFrameToken(
                stream_id="vq2-camera",
                generation=1,
                frame_id=100 + self.calls,
                publication_sequence=publication_base + self.calls,
            )
            current = VisualTarget(
                track_id=current_track_id,
                frame_token=token,
                received_monotonic_s=now_monotonic_s,
                normalized_x=0.02,
                normalized_y_down=-0.04,
                normalized_x_rate_s=0.0,
                normalized_y_rate_down_s=0.0,
                log_scale=-1.0,
                log_scale_rate_s=0.2,
                confidence=0.9,
                association_confidence=0.9,
                consecutive_frames=20,
            )
            next_target = VisualTarget(
                track_id=next_track_id,
                frame_token=token,
                received_monotonic_s=now_monotonic_s,
                normalized_x=0.30,
                normalized_y_down=-0.20,
                normalized_x_rate_s=0.0,
                normalized_y_rate_down_s=0.0,
                log_scale=-2.0,
                log_scale_rate_s=0.1,
                confidence=0.8,
                association_confidence=0.8,
                consecutive_frames=20,
            )
            output = VisualServoOutput(
                target_roll_rad=0.0,
                target_pitch_rad=0.03,
                yaw_rate_rad_s=-0.02,
                thrust=0.21,
                corridor_frames=3,
                advance_enabled=False,
                next_gate_blend=self.next_gate_blend,
                horizontal_error=0.02,
                vertical_error_image_down=-0.04,
                effective_horizontal_error=0.118,
                effective_vertical_error_image_down=-0.096,
                effective_horizontal_rate_s=0.0,
                effective_vertical_rate_down_s=0.0,
                next_horizontal_error=0.30,
                next_vertical_error_image_down=-0.20,
                horizontal_abs_error_delta=-0.01,
                vertical_abs_error_delta=-0.01,
                brake_reason="advance_not_authorized",
                yaw_envelope_limited=False,
            )
            return vq2_module.VisualApproachProposal(
                current_target=current,
                next_target=next_target,
                servo_output=output,
                candidate_track_ids=(next_track_id,),
                provisional_track_ids=(),
                withholding_reason=None,
                relationship_basis=None,
                latched_next_track_id=next_track_id,
            )

    def sample():
        sample_count[0] += 1
        if not (
            passage_violation_kind
            in {"transient_total_wall", "transient_wall"}
            and sample_count[0] == 5
        ):
            graph.latest_snapshot.tracker_frame_sequence = sample_count[0]
            graph.latest_snapshot.latest_camera_token = (
                vq2_module.VisualCameraFrameToken(
                    generation=1,
                    frame_id=100 + sample_count[0],
                    publication_sequence=(
                        publication_base + sample_count[0]
                    ),
                    stream_id="vq2-camera",
                )
            )
        if sample_count[0] in {2, 3}:
            center_y = 170 if sample_count[0] == 2 else 150
            runner.tracker.target = vq2_module.GateTarget(
                frame_id=100 + sample_count[0],
                sim_time_ns=100 + sample_count[0],
                received_monotonic_s=clock[0],
                center_x=320,
                center_y=center_y,
                bbox=(240, center_y - 60, 160, 120),
                confidence=0.9,
            )
            runner.tracker.consecutive += 1
        crossing_sample = (
            6
            if passage_violation_kind
            in {
                "transient_resume",
                "transient_total_wall",
                "transient_wall",
            }
            else 5
        )
        if sample_count[0] == crossing_sample:
            runner.tracker.target = vq2_module.GateTarget(
                frame_id=200,
                sim_time_ns=200,
                received_monotonic_s=clock[0],
                center_x=320,
                center_y=180,
                bbox=(60, 0, 520, 360),
                confidence=0.9,
            )
            runner.tracker.consecutive = 3
        if sample_count[0] == crossing_sample + 2:
            adapter.race_status = RaceStatus(1250, 0, -1, 1, 123)

    async def send(command, **_kwargs):
        sent_commands.append(command)
        runner._last_flight_command_sent_s = clock[0]

    async def sleep(seconds):
        minimum_step = (
            0.45
            if sample_count[0] == 1
            else (
                0.05
                if (
                    passage_violation_kind == "transient_resume"
                    and sample_count[0] in {4, 5}
                )
                else 0.25
            )
        )
        clock[0] += max(float(seconds), minimum_step)

    async def next_slot():
        return clock[0]

    def checked_attitude_rate_command(
        _estimate,
        *,
        target_roll_rad,
        target_pitch_rad,
        thrust,
    ):
        del target_roll_rad
        bootstrap_attitude_targets.append(
            (
                clock[0] - flight_clock_origin,
                target_pitch_rad,
                thrust,
            )
        )
        command = AttitudeRateCommand(
            roll_rate=0.18,
            pitch_rate=-0.20,
            yaw_rate=0.0,
            thrust=thrust,
        )
        bootstrap_commands.append(command)
        return command

    original_yaw_rate = vq2_module.visual_alignment_yaw_rate

    def checked_yaw_rate(**kwargs):
        yaw_checks.append(
            (
                sample_count[0],
                kwargs["requested_rate_rad_s"],
                kwargs["reference_yaw_rad"],
            )
        )
        return original_yaw_rate(**kwargs)

    def record_tick(stage, _elapsed, command):
        if stage == "gate0/confirm":
            confirmation_commands.append(command)

    if cumulative_wall_case:
        base_passage_lease = vq2_module.VisualApproachPassageLease

        class PreloadedPassageLease(base_passage_lease):
            def __init__(self):
                super().__init__()
                for publication, observation_s, safe, blend in (
                    (1, 0.00, True, True),
                    (2, 0.01, False, False),
                    (3, 0.10, True, True),
                    (4, 0.11, False, False),
                    (5, 0.20, True, True),
                ):
                    self.observe(
                        vq2_module.VisualCameraFrameToken(
                            generation=1,
                            frame_id=10 + publication,
                            publication_sequence=publication,
                            stream_id="vq2-camera",
                        ),
                        observation_monotonic_s=observation_s,
                        passage_safe=safe,
                        blend_active=blend,
                    )

        monkeypatch.setattr(
            vq2_module,
            "VisualApproachPassageLease",
            PreloadedPassageLease,
        )
    monkeypatch.setattr(
        vq2_module,
        "RollingVisualApproachServo",
        FakeApproach,
    )
    monkeypatch.setattr(
        vq2_module,
        "visual_alignment_yaw_rate",
        checked_yaw_rate,
    )
    monkeypatch.setattr(
        vq2_module,
        "attitude_rate_command",
        checked_attitude_rate_command,
    )
    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_send_flight_command", send)
    monkeypatch.setattr(runner, "_record_tick", record_tick)
    monkeypatch.setattr(
        runner,
        "_wait_for_next_flight_command_slot",
        next_slot,
    )
    monkeypatch.setattr(
        vq2_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            perf_counter_ns=lambda: round(clock[0] * 1_000_000_000),
        ),
    )
    monkeypatch.setattr(
        vq2_module,
        "asyncio",
        SimpleNamespace(
            sleep=sleep,
            CancelledError=asyncio.CancelledError,
        ),
    )

    result = asyncio.run(
        runner._run_gate0(
            vq2_module.StartContext(
                spawn_roll_rad=0.0,
                spawn_pitch_rad=-0.05,
                initial_gate_x=320,
                initial_gate_y=180,
                initial_gate_area=6400,
                go_boot_ms=1000,
            ),
            visual_next_gate_blend=True,
        )
    )

    summary = result["visual_next_gate_blend"]
    assert result["gate0_passed"] is True
    assert summary["started"] is True
    assert summary["blended_next_track_id"] == next_track_id
    expected_visual_command_count = (
        4 if passage_violation_kind == "transient_resume" else 3
    )
    assert (
        summary["fresh_blend_frame_count"]
        == expected_visual_command_count
    )
    assert summary["command_count"] == expected_visual_command_count
    assert summary["withdrawn_before_confirmation"] is True
    assert summary["withdrawal_reason"] == withdrawal_reason
    assert (
        summary["passage_suspension_fresh_frame_count"]
        == expected_suspended_frames
    )
    assert summary["passage_suspension_epoch_count"] == (
        3
        if cumulative_wall_case
        else (1 if expected_suspended_frames else 0)
    )
    assert summary["passage_suspension_resume_count"] == (
        2
        if cumulative_wall_case
        else (1 if passage_violation_kind == "transient_resume" else 0)
    )
    assert summary["passage_suspension_max_streak"] == (
        1 if expected_suspended_frames else 0
    )
    assert summary["passage_suspension_currently_active"] is False
    assert summary["passage_suspension_lease_exhausted"] is (
        passage_violation_kind
        in {"transient_total_wall", "transient_wall"}
    )
    assert summary["passage_suspension_retirement_reason"] == (
        (
            "total_suspension_wall_duration_exhausted"
            if cumulative_wall_case
            else "suspension_epoch_wall_duration_exhausted"
        )
        if passage_violation_kind == "transient_wall"
        or cumulative_wall_case
        else None
    )
    assert summary["max_passage_suspension_fresh_frames"] == 2
    assert summary["max_total_passage_suspension_fresh_frames"] == 4
    assert summary["max_passage_suspension_epochs"] == 3
    assert summary["max_passage_suspension_epoch_duration_s"] == 0.12
    assert summary["max_total_passage_suspension_duration_s"] == 0.20
    assert (
        summary[
            "passage_suspension_visual_authority_withheld_tick_count"
        ]
        == (
            2
            if passage_violation_kind
            in {"transient_total_wall", "transient_wall"}
            else (1 if expected_suspended_frames else 0)
        )
    )
    if passage_violation_kind == "hard_apparent_scale":
        assert summary["passage_hard_violation_codes"] == [
            "current_apparent_scale"
        ]
    else:
        assert "passage_hard_violation_codes" not in summary
    assert summary["launch_collective_hold_s"] == pytest.approx(0.45)
    assert summary["launch_boost_thrust"] == pytest.approx(0.32)
    assert summary["command_axis_authority"] == {
        "yaw": "visual_next_track_blend",
        "pitch": "bounded_visual_brake",
        "roll_collective": "proved_gate0_bootstrap",
    }
    visual_command_indices = (
        [0, 1, 2, 4]
        if passage_violation_kind == "transient_resume"
        else [0, 1, 2]
    )
    visual_commands = [
        sent_commands[index] for index in visual_command_indices
    ]
    visual_attitude_targets = [
        bootstrap_attitude_targets[index]
        for index in visual_command_indices
    ]
    visual_bootstrap_commands = [
        bootstrap_commands[index]
        for index in visual_command_indices
    ]
    assert visual_commands
    assert len(visual_attitude_targets) == len(visual_commands)
    assert all(
        abs(command.roll_rate)
        <= vq2_module.MAX_COMMAND_RATE_RAD_S
        and abs(command.pitch_rate)
        <= vq2_module.VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S
        and abs(command.yaw_rate)
        <= vq2_module.VISUAL_ALIGN_MAX_YAW_RATE_RAD_S
        for command in visual_commands
    )
    if passage_violation_kind in {
        "transient_projected_vertical",
        "transient_resume",
        "transient_total_wall",
        "transient_wall",
    }:
        suspended_command = sent_commands[3]
        suspended_bootstrap = bootstrap_commands[3]
        assert suspended_command == suspended_bootstrap
        assert suspended_command.roll_rate == pytest.approx(0.18)
        assert suspended_command.pitch_rate == pytest.approx(-0.20)
        assert suspended_command.yaw_rate == 0.0
        assert all(
            command.pitch_rate == pytest.approx(
                -vq2_module.VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S
            )
            for command in visual_commands
        )
    assert all(
        command.roll_rate == bootstrap.roll_rate
        and command.pitch_rate
        == pytest.approx(
            max(
                -vq2_module.VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S,
                min(
                    vq2_module.VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S,
                    bootstrap.pitch_rate,
                ),
            )
        )
        and command.thrust == bootstrap.thrust
        for command, bootstrap in zip(
            visual_commands,
            visual_bootstrap_commands,
            strict=True,
        )
    )
    assert all(
        target_pitch
        == pytest.approx(
            max(
                vq2_module.gate0_target_pitch_rad(
                    -0.05,
                    0.0,
                    elapsed_s,
                    blend_duration_s=0.80,
                ),
                0.03,
            )
        )
        for elapsed_s, target_pitch, thrust in visual_attitude_targets
    )
    first_rate = 0.35 * ((170.0 - 180.0) / 0.45)
    second_rate = (
        0.65 * first_rate
        + 0.35 * ((150.0 - 170.0) / 0.25)
    )
    expected_visual_thrusts = [
        0.26,
        (
            0.32
            if cumulative_wall_case
            else vq2_module.gate_vertical_thrust(170.0, first_rate)
        ),
        vq2_module.gate_vertical_thrust(150.0, second_rate),
    ]
    if passage_violation_kind == "transient_resume":
        expected_visual_thrusts.append(
            vq2_module.gate_vertical_thrust(150.0, second_rate)
        )
    assert [target[2] for target in visual_attitude_targets] == pytest.approx(
        expected_visual_thrusts
    )
    assert [command.thrust for command in visual_commands] == pytest.approx(
        [target[2] for target in visual_attitude_targets]
    )
    boundary_and_later = [
        (targets, command)
        for targets, command in zip(
            visual_attitude_targets,
            visual_commands,
            strict=True,
        )
        if targets[0] >= 0.45 - 1e-9
    ]
    expected_boundary_times = [0.45, 0.70]
    if passage_violation_kind == "transient_resume":
        expected_boundary_times.append(1.00)
    assert [targets[0] for targets, _command in boundary_and_later] == (
        pytest.approx(expected_boundary_times)
    )
    assert all(
        targets[1] == pytest.approx(0.03)
        and targets[2] != pytest.approx(0.21)
        and command.yaw_rate == pytest.approx(-0.02)
        for targets, command in boundary_and_later
    )
    assert any(
        command.pitch_rate != 0.0 and command.yaw_rate != 0.0
        for command in visual_commands
    )
    assert confirmation_commands
    assert all(
        command == AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
        for command in confirmation_commands
    )
    assert any(
        sample >= 4 and requested == 0.0
        for sample, requested, _reference in yaw_checks
    )
    assert any(
        sample >= 6 and requested == 0.0
        for sample, requested, _reference in yaw_checks
    )
    assert {reference for _sample, _requested, reference in yaw_checks} == {
        0.0
    }


@pytest.mark.parametrize(
    (
        "target_bbox",
        "target_center_x",
        "turn_score",
        "prove_before_exit",
        "dip_after_exit",
        "exit_counterroll",
        "expected_rolls",
    ),
    (
        (
            (240, 130, 160, 100),
            320,
            0.20,
            False,
            False,
            False,
            (0.0, 0.0, 0.13, 0.13, 0.13, 0.0),
        ),
        (
            (192, 80, 256, 200),
            320,
            0.20,
            False,
            False,
            False,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        (
            (240, 110, 160, 140),
            320,
            0.20,
            True,
            False,
            True,
            (0.0, 0.0, 0.13, -0.08, -0.08, -0.08),
        ),
        (
            (220, 100, 200, 160),
            320,
            0.20,
            True,
            False,
            False,
            (0.0, 0.0, 0.13, 0.13, 0.13, 0.0),
        ),
        (
            (220, 100, 200, 160),
            320,
            0.20,
            False,
            False,
            True,
            (0.0, 0.0, 0.13, 0.13, 0.13, 0.0),
        ),
        (
            (220, 100, 200, 160),
            320,
            -0.20,
            True,
            False,
            True,
            (0.0, 0.0, -0.13, 0.08, 0.08, 0.08),
        ),
        (
            (192, 80, 256, 200),
            288,
            0.20,
            True,
            False,
            True,
            (-0.015, -0.015, 0.115, -0.015, -0.015, -0.015),
        ),
        (
            (220, 100, 200, 160),
            320,
            0.20,
            True,
            True,
            True,
            (0.0, 0.0, 0.13, -0.08, -0.08, -0.08),
        ),
    ),
    ids=(
        "stable-cue",
        "close-range-taper",
        "exact-three-point-five-x-exit-counterroll",
        "counterroll-disabled",
        "no-pre-onset-proof",
        "mirrored-exit-counterroll",
        "wrong-side-retains-centering",
        "latched-across-area-dip",
    ),
)
def test_gate0_course_line_preturn_requires_fresh_streak_and_tapers_close(
    monkeypatch,
    target_bbox,
    target_center_x,
    turn_score,
    prove_before_exit,
    dip_after_exit,
    exit_counterroll,
    expected_rolls,
):
    class CommandsCaptured(Exception):
        pass

    clock = [0.0]
    sample_count = [0]
    target_rolls = []
    preturn_events = []
    exit_counterroll_events = []
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    runner.estimate = _estimate(roll=0.0)
    runner._latest_detection_image = object()
    context = vq2_module.StartContext(0.0, -0.31, 320, 180, 6400, 1000)

    def sample():
        sample_count[0] += 1
        frame_id = min(sample_count[0], 3)
        active_bbox = target_bbox
        if prove_before_exit and sample_count[0] <= 3:
            active_bbox = (240, 130, 160, 100)
        elif dip_after_exit and sample_count[0] >= 5:
            active_bbox = (240, 130, 160, 100)
        runner.tracker.target = vq2_module.GateTarget(
            frame_id=frame_id,
            sim_time_ns=frame_id,
            received_monotonic_s=clock[0],
            center_x=target_center_x,
            center_y=180,
            bbox=active_bbox,
            confidence=0.8,
        )
        runner.tracker.consecutive = 3

    def capture_target_roll(
        _estimate,
        *,
        target_roll_rad,
        target_pitch_rad,
        thrust,
    ):
        del target_pitch_rad
        target_rolls.append(target_roll_rad)
        return AttitudeRateCommand(0.0, 0.0, 0.0, thrust)

    async def capture_command(_command, **_kwargs):
        if len(target_rolls) == 6:
            raise CommandsCaptured

    async def advance_clock(_seconds):
        clock[0] += 0.10

    def capture_event(event, **payload):
        if event == "course_line_preturn_applied":
            preturn_events.append(payload)
        elif event == "course_line_exit_counterroll_applied":
            exit_counterroll_events.append(payload)

    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_send_flight_command", capture_command)
    monkeypatch.setattr(runner.recorder, "emit", capture_event)
    monkeypatch.setattr(vq2_module, "attitude_rate_command", capture_target_roll)
    monkeypatch.setattr(
        vq2_module,
        "cyan_course_line_observation",
        lambda _image: vq2_module.CourseLineObservation(
            turn_score=turn_score,
            upper_center_x=384.0,
            lower_center_x=320.0,
            upper_pixel_count=136,
            lower_pixel_count=136,
        ),
    )

    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(monotonic=lambda: clock[0]),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=advance_clock),
        )
        with pytest.raises(CommandsCaptured):
            asyncio.run(
                runner._run_gate0(
                    context,
                    course_line_preturn=True,
                    course_line_exit_counterroll_enabled=exit_counterroll,
                )
            )

    assert vq2_module.COURSE_LINE_PRETURN_REQUIRED_FRAMES == 3
    assert vq2_module.COURSE_LINE_PRETURN_MAX_AGE_S == 0.25
    assert vq2_module.COURSE_LINE_PRETURN_MIN_GATE_AREA_SCALE == 1.30
    assert vq2_module.COURSE_LINE_EXIT_COUNTERROLL_ONSET_AREA_SCALE == 3.5
    assert vq2_module.COURSE_LINE_PRETURN_TAPER_AREA_SCALE == 8.0
    assert target_rolls == pytest.approx(expected_rolls)
    assert all(
        {
            "frame_id",
            "elapsed_s",
            "gate_area_px",
            "filtered_turn_score",
            "consistent_frame_count",
            "roll_bias_rad",
            "target_roll_rad",
        }
        == set(event)
        for event in preturn_events
    )
    if target_bbox == (240, 130, 160, 100) and not exit_counterroll:
        assert preturn_events
    assert all(
        {
            "frame_id",
            "elapsed_s",
            "gate_area_px",
            "proved_turn_score",
            "filtered_turn_score",
            "consistent_frame_count",
            "normalized_x",
            "target_roll_rad",
        }
        == set(event)
        for event in exit_counterroll_events
    )
    assert bool(exit_counterroll_events) is any(
        abs(roll) == pytest.approx(0.08)
        for roll in expected_rolls
    )


def test_configured_gate0_early_turn_combines_roll_negative_yaw_and_braking(
    monkeypatch,
):
    class CommandsCaptured(Exception):
        pass

    document = controller_config_module.default_controller_config_mapping()
    document["phase_timing"]["gate0_yaw_brake_duration_s"] = 0.21
    document["turn_cue"]["exit_counterroll_enabled"] = False
    document["yaw_control"]["gate0_turn_score_gain"] = -0.80
    document["yaw_control"]["gate0_command_rate_cap_rad_s"] = 0.08
    document["forward_braking"]["gate0_turn_pitch_rad"] = 0.04
    document["forward_braking"]["gate0_turn_thrust_cap"] = 0.24
    config = controller_config_module.validate_controller_config(document)
    clock = [0.0]
    sample_count = [0]
    objectives = []
    commands = []
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    runner = VQ2Runner(
        adapter,
        _FakeVision(),
        controller_config=config,
    )
    runner.estimate = _estimate(roll=0.0, pitch=0.0, yaw=0.0)
    runner._latest_detection_image = object()
    context = vq2_module.StartContext(
        0.0,
        -0.31,
        320,
        180,
        6400,
        1000,
    )

    def sample():
        sample_count[0] += 1
        runner.tracker.target = vq2_module.GateTarget(
            frame_id=sample_count[0],
            sim_time_ns=sample_count[0],
            received_monotonic_s=clock[0],
            center_x=320,
            center_y=180,
            bbox=(240, 130, 160, 100),
            confidence=0.8,
        )
        runner.tracker.consecutive = 3

    def capture_objective(
        _estimate_value,
        *,
        target_roll_rad,
        target_pitch_rad,
        thrust,
    ):
        objectives.append(
            (target_roll_rad, target_pitch_rad, thrust)
        )
        return AttitudeRateCommand(0.0, 0.0, 0.0, thrust)

    async def capture_command(command, **_kwargs):
        commands.append(command)
        if len(commands) == 8:
            raise CommandsCaptured

    async def advance_clock(_seconds):
        clock[0] += 0.05

    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_send_flight_command", capture_command)
    monkeypatch.setattr(vq2_module, "attitude_rate_command", capture_objective)
    monkeypatch.setattr(
        vq2_module,
        "cyan_course_line_observation",
        lambda _image: vq2_module.CourseLineObservation(
            turn_score=0.20,
            upper_center_x=384.0,
            lower_center_x=320.0,
            upper_pixel_count=136,
            lower_pixel_count=136,
        ),
    )

    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(monotonic=lambda: clock[0]),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=advance_clock),
        )
        with pytest.raises(CommandsCaptured):
            asyncio.run(
                runner._run_gate0(
                    context,
                    course_line_preturn=True,
                    course_line_exit_counterroll_enabled=False,
                )
            )

    active = [
        (objective, command)
        for objective, command in zip(objectives, commands)
        if command.yaw_rate != 0.0
    ]
    assert active
    assert all(command.yaw_rate == -0.08 for _objective, command in active)
    assert all(
        target_roll == pytest.approx(0.13)
        and target_pitch == pytest.approx(0.04)
        and thrust == pytest.approx(0.24)
        for (target_roll, target_pitch, thrust), _command in active
    )
    assert commands[0].yaw_rate == commands[1].yaw_rate == 0.0
    assert runner._gate0_early_turn_summary is not None
    assert runner._gate0_early_turn_summary["command_count"] >= len(active)
    assert runner._gate1_max_abs_yaw_excursion_rad == 0.0


def _capture_first_gate0_thrust(
    monkeypatch,
    *,
    elapsed_s,
    minimum_thrust=0.21,
    boost_until_s=0.45,
    pd_thrust=0.21,
    target_bbox=(282, 134, 80, 80),
    control_y=None,
):
    class CommandCaptured(Exception):
        pass

    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    runner.estimate = _estimate()
    x, y, width, height = target_bbox
    runner.tracker.target = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=float(elapsed_s),
        center_x=x + width // 2,
        center_y=(y + height // 2 if control_y is None else control_y),
        bbox=target_bbox,
        confidence=0.8,
    )
    runner.tracker.consecutive = 3
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)
    clock = iter((0.0, float(elapsed_s)))
    commands = []

    async def capture_command(command, **_kwargs):
        commands.append(command)
        raise CommandCaptured

    monkeypatch.setattr(runner, "_sample", lambda: None)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(
        vq2_module,
        "gate_vertical_thrust",
        lambda *_args: float(pd_thrust),
    )
    monkeypatch.setattr(runner, "_send_flight_command", capture_command)

    async def capture_first_command():
        # Replace only the runner module's binding. Mutating Python's shared
        # time module would also consume this finite clock inside asyncio.
        with monkeypatch.context() as clock_patch:
            clock_patch.setattr(
                vq2_module,
                "time",
                SimpleNamespace(monotonic=lambda: next(clock)),
            )
            await runner._run_gate0(
                context,
                minimum_thrust=minimum_thrust,
                boost_until_s=boost_until_s,
            )

    with pytest.raises(CommandCaptured):
        asyncio.run(capture_first_command())

    assert len(commands) == 1
    return commands[0].thrust


@pytest.mark.parametrize(
    ("minimum_thrust", "expected_thrust"),
    ((0.21, 0.21), (0.25, 0.25)),
    ids=("legacy-floor", "configured-floor"),
)
def test_gate0_minimum_thrust_applies_only_after_boost_schedule(
    monkeypatch,
    minimum_thrust,
    expected_thrust,
):
    assert _capture_first_gate0_thrust(
        monkeypatch,
        elapsed_s=0.5,
        minimum_thrust=minimum_thrust,
    ) == expected_thrust


@pytest.mark.parametrize(
    ("elapsed_s", "expected_thrust"),
    (
        (math.nextafter(0.80, 0.0), 0.32),
        (0.80, 0.23),
    ),
    ids=("before-boundary-boost", "at-boundary-pd"),
)
def test_gate0_extended_boost_switches_to_pixel_pd_at_exact_boundary(
    monkeypatch,
    elapsed_s,
    expected_thrust,
):
    assert vq2_module.COURSE_GATE0_BOOST_UNTIL_S == 0.80
    assert _capture_first_gate0_thrust(
        monkeypatch,
        elapsed_s=elapsed_s,
        boost_until_s=vq2_module.COURSE_GATE0_BOOST_UNTIL_S,
        pd_thrust=0.23,
    ) == expected_thrust


def test_post_pass_sampling_filters_residue_before_tracker_selection():
    adapter = _FakeAdapter()
    vision = _FakeVision()
    vision.current_snapshot = _vision_snapshot(
        frame_id=101,
        sim_time_ns=1_010,
        received_monotonic_s=1.0,
    )
    runner = VQ2Runner(adapter, vision)
    residue = _detection(75, 0, 518, 360)
    gate1 = _detection(409, 138, 28, 45)
    assert select_primary_gate([residue, gate1]) is residue
    runner.detector = SimpleNamespace(detect=lambda _image: [residue, gate1])
    runner._post_gate_reacquisition = True

    runner._sample()

    assert runner._latest_accepted_target is not None
    assert runner._latest_accepted_target.bbox == gate1.bbox
    assert runner.tracker.consecutive == 1


def test_no_replay_or_diagnostics_skips_detection_summary_capture_path(monkeypatch):
    adapter = _FakeAdapter()
    vision = _FakeVision()
    vision.current_snapshot = _vision_snapshot(
        frame_id=101, sim_time_ns=1_010, received_monotonic_s=1.0
    )
    runner = VQ2Runner(adapter, vision)
    runner.detector = SimpleNamespace(detect=lambda _image: [_detection(10, 10, 40, 40)])

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("diagnostic summary added work to production path")

    monkeypatch.setattr(vq2_module, "gate_detection_summary", must_not_run)
    monkeypatch.setattr(vq2_module.time, "perf_counter_ns", must_not_run)
    runner._sample()
    assert runner.tracker.target is not None


def test_capture_loaded_sampling_records_exact_passive_frame_timing(monkeypatch):
    class FakeReplay:
        def __init__(self):
            self.events = []
            self.frames = []

        def record_event(self, event, **fields):
            self.events.append((event, fields))
            return True

        def capture_frame(self, image, **fields):
            self.frames.append((image, fields))
            return True

    adapter = _FakeAdapter()
    vision = _FakeVision()
    vision.current_snapshot = _vision_snapshot(
        frame_id=101,
        sim_time_ns=1_010,
        received_monotonic_s=1.0,
    )
    replay = FakeReplay()
    recorder = vq2_module.JsonlRecorder(
        None, replay=replay, capture_fifo_enabled=True
    )
    runner = VQ2Runner(adapter, vision, recorder=recorder)
    runner.detector = SimpleNamespace(
        detect=lambda _image: [_detection(10, 10, 40, 40)]
    )
    clock = iter((20, 21, 22, 23, 24, 25, 26))
    monkeypatch.setattr(vq2_module.time, "perf_counter_ns", lambda: next(clock))

    runner._sample()

    timing_events = [
        fields["observation"]
        for event, fields in replay.events
        if event == "camera_frame_timing_observation"
    ]
    assert len(timing_events) == 1
    observation = CameraFrameTimingObservationV1.from_primitive(
        timing_events[0]
    )
    assert observation.frame_timing == vision.current_snapshot.timing
    assert observation.consume_monotonic_ns == 20
    assert observation.detection_start_monotonic_ns == 22
    assert observation.detection_end_monotonic_ns == 23
    assert observation.tracking_start_monotonic_ns == 24
    assert observation.tracking_end_monotonic_ns == 25
    assert observation.work_end_monotonic_ns == 26
    assert len(replay.frames) == 1


def test_async_replay_boundary_preserves_exact_vq2_contract_schemas(tmp_path):
    bundle = tmp_path / "exact-async.vq2replay"
    replay = AsyncReplayRecorder(
        ReplayBundleWriter(bundle, require_private=False)
    )
    imu_ingress = MavlinkIngressV1(
        stream_id="vq2-mavlink-udp-14550",
        generation=1,
        sequence=0,
        message_type="HIGHRES_IMU",
        host_clock_id="host-perf-counter",
        received_monotonic_ns=100,
        source_time_value=500,
        source_time_unit="us",
    )
    received_imu = ReceivedIMUSampleV1(
        ingress=imu_ingress,
        imu=IMUData(
            timestamp_us=500,
            accel=(1.0, 2.0, -9.0),
            gyro=(0.1, 0.2, 0.3),
        ),
    )
    heartbeat = MavlinkIngressV1(
        stream_id="vq2-mavlink-udp-14550",
        generation=1,
        sequence=1,
        message_type="HEARTBEAT",
        host_clock_id="host-perf-counter",
        received_monotonic_ns=200,
        source_time_value=None,
        source_time_unit=None,
    )
    snapshot = _vision_snapshot(
        generation=2,
        frame_id=101,
        sim_time_ns=1_010,
        received_monotonic_s=1.0,
    )

    assert replay.record_imu(
        received_imu.imu,
        received_monotonic_s=0.0000001,
        received_sample=received_imu,
    )
    assert replay.record_mavlink_ingress(heartbeat)
    assert replay.capture_decoded_snapshot(snapshot)
    stats = replay.close(expected_decoded_frames=1)

    assert stats.complete is True
    _summary, records = ReplayBundleReader(bundle).verify_and_read(
        verify_frames=True
    )
    events = {
        row["event"]: row
        for row in records
        if row["type"] == "event"
    }
    assert ReceivedIMUSampleV1.from_primitive(
        events["received_imu"]["observation"]
    ) == received_imu
    assert MavlinkIngressV1.from_primitive(
        events["mavlink_ingress"]["observation"]
    ) == heartbeat
    assert FrameTimingV1.from_primitive(
        events["camera_frame_timing"]["observation"]
    ) == snapshot.timing


def test_non_passive_capture_uses_latest_snapshot_not_passive_fifo(monkeypatch):
    class LatestOnlyVision(_FakeVision):
        def pop_capture_snapshot(self):
            raise AssertionError("passive FIFO entered outside passive preflight")

    class FakeReplay:
        def __init__(self):
            self.events = []
            self.frames = []

        def record_event(self, event, **fields):
            self.events.append((event, fields))
            return True

        def capture_frame(self, image, **fields):
            self.frames.append((image, fields))
            return True

    vision = LatestOnlyVision()
    vision.current_snapshot = _vision_snapshot(frame_id=101, sim_time_ns=1_010)
    replay = FakeReplay()
    recorder = vq2_module.JsonlRecorder(
        None,
        replay=replay,
        capture_fifo_enabled=False,
    )
    runner = VQ2Runner(_FakeAdapter(), vision, recorder=recorder)
    runner.detector = SimpleNamespace(detect=lambda _image: [])
    clock = iter(range(20, 27))
    monkeypatch.setattr(vq2_module.time, "perf_counter_ns", lambda: next(clock))

    runner._sample()

    assert len(replay.frames) == 1
    assert replay.frames[0][1]["frame_id"] == 101


def test_stopped_vision_tail_consumes_a_final_capture_publication(monkeypatch):
    class FakeReplay:
        def __init__(self):
            self.events = []
            self.frames = []

        def record_event(self, event, **fields):
            self.events.append((event, fields))
            return True

        def capture_frame(self, image, **fields):
            self.frames.append((image, fields))
            return True

    vision = _FakeVision()
    vision.current_snapshot = _vision_snapshot(frame_id=101, sim_time_ns=1_010)
    replay = FakeReplay()
    recorder = vq2_module.JsonlRecorder(None, replay=replay)
    runner = VQ2Runner(_FakeAdapter(), vision, recorder=recorder)
    runner.detector = SimpleNamespace(
        detect=lambda _image: [_detection(10, 10, 40, 40)]
    )
    clock = iter(range(20, 34))
    monkeypatch.setattr(vq2_module.time, "perf_counter_ns", lambda: next(clock))
    runner._sample()

    # Model a publication completing while stop() joins the receiver thread.
    vision.current_snapshot = _vision_snapshot(frame_id=102, sim_time_ns=1_020)
    vision.is_running = False
    vq2_module._consume_stopped_capture_tail(runner, vision)

    timing_events = [
        fields
        for event, fields in replay.events
        if event == "camera_frame_timing_observation"
    ]
    assert len(timing_events) == 2
    assert len(replay.frames) == 2


def test_stopped_vision_tail_rejects_a_live_receiver():
    vision = _FakeVision()
    vision.is_running = True
    runner = VQ2Runner(_FakeAdapter(), vision)

    with pytest.raises(RuntimeError, match="while vision is running"):
        vq2_module._consume_stopped_capture_tail(runner, vision)


def test_stopped_vision_tail_drains_two_publications_between_polls(monkeypatch):
    class QueuedVision(_FakeVision):
        def __init__(self):
            super().__init__()
            self.pending = [
                _vision_snapshot(frame_id=101, sim_time_ns=1_010),
                _vision_snapshot(frame_id=102, sim_time_ns=1_020),
            ]

        def pop_capture_snapshot(self):
            return self.pending.pop(0) if self.pending else None

        def capture_snapshot_queue_depth(self):
            return len(self.pending)

    class FakeReplay:
        def __init__(self):
            self.events = []
            self.frames = []

        def record_event(self, event, **fields):
            self.events.append((event, fields))
            return True

        def capture_frame(self, image, **fields):
            self.frames.append((image, fields))
            return True

    vision = QueuedVision()
    replay = FakeReplay()
    recorder = vq2_module.JsonlRecorder(
        None, replay=replay, capture_fifo_enabled=True
    )
    runner = VQ2Runner(_FakeAdapter(), vision, recorder=recorder)
    runner.detector = SimpleNamespace(
        detect=lambda _image: [_detection(10, 10, 40, 40)]
    )
    clock = iter(range(20, 34))
    monkeypatch.setattr(vq2_module.time, "perf_counter_ns", lambda: next(clock))

    vq2_module._consume_stopped_capture_tail(runner, vision)

    assert vision.capture_snapshot_queue_depth() == 0
    assert len(replay.frames) == 2
    assert len(
        [
            event
            for event, _fields in replay.events
            if event == "camera_frame_timing_observation"
        ]
    ) == 2


def test_sampling_uses_frame_identity_not_opaque_camera_source_timestamp():
    adapter = _FakeAdapter()
    vision = _FakeVision()
    detections = [0]

    def detect(_image):
        detections[0] += 1
        return [_detection(10, 10, 40, 40)]

    runner = VQ2Runner(adapter, vision)
    runner.detector = SimpleNamespace(detect=detect)
    vision.current_snapshot = _vision_snapshot(
        generation=3,
        frame_id=101,
        sim_time_ns=1_000,
        received_monotonic_s=1.0,
    )
    runner._sample()
    assert detections[0] == 1

    # A changed source token cannot relabel and re-run one camera identity.
    vision.current_snapshot = _vision_snapshot(
        generation=3,
        frame_id=101,
        sim_time_ns=9_999,
        received_monotonic_s=1.1,
    )
    runner._sample()
    assert detections[0] == 1
    assert runner._last_frame_sim_ns == 1_000

    # Receiver generation is part of identity, so a restart may reuse the ID.
    vision.current_snapshot = _vision_snapshot(
        generation=4,
        frame_id=101,
        sim_time_ns=5,
        received_monotonic_s=1.2,
    )
    runner._sample()
    assert detections[0] == 2
    assert runner._last_frame_identity == (4, 101)


@pytest.mark.parametrize(
    ("stage", "expected_success"),
    (("preflight", False), (vq2_module.VISUAL_COURSE_STAGE, True)),
)
def test_incomplete_requested_replay_is_warning_for_powered_navigation(
    stage,
    expected_success,
):
    original = vq2_module.StageResult(
        stage=stage,
        success=True,
        reason="pass",
        duration_s=1.0,
        cleanup_confirmed=True,
        details={},
    )
    stats = SimpleNamespace(complete=False)
    # replay_capture_result uses dataclasses.asdict for the production stats;
    # use that exact frozen dataclass shape through a minimal real instance.
    from aigp_loop.replay import AsyncCaptureStats

    failed = replay_capture_result(
        original,
        capture_requested=True,
        capture_stats=AsyncCaptureStats(
            1, 0, 1, 0, 0, 1, 1, 0, 1, False, None, "queue overflow"
        ),
    )
    assert failed.success is expected_success
    assert failed.cleanup_confirmed
    assert ("replay capture incomplete" in failed.reason) is (
        stage == "preflight"
    )
    assert failed.details["replay_capture"]["complete"] is False


def test_post_pass_diagnostic_reference_stays_on_exact_processed_frame():
    adapter = _FakeAdapter()
    vision = _FakeVision()
    processed = _vision_snapshot(
        frame_id=101,
        sim_time_ns=1_010,
        received_monotonic_s=1.0,
        generation=3,
    )
    processed.camera_frame.image.fill(17)
    vision.current_snapshot = processed
    runner = VQ2Runner(adapter, vision)
    runner.detector = SimpleNamespace(
        detect=lambda _image: [_detection(409, 138, 28, 45)]
    )
    runner._post_gate_reacquisition = True

    runner._sample()
    newer = _vision_snapshot(
        frame_id=102,
        sim_time_ns=1_020,
        received_monotonic_s=1.03,
        generation=3,
    )
    newer.camera_frame.image.fill(99)
    vision.current_snapshot = newer

    token, retained_image = runner._post_gate_last_frame
    assert token == (3, 101, 1_010)
    assert retained_image is processed.camera_frame.image
    assert int(retained_image[0, 0, 0]) == 17


def test_emergency_reset_is_sent_even_with_no_fresh_baseline(monkeypatch):
    adapter = _FakeAdapter()
    vision = _FakeVision()
    runner = VQ2Runner(adapter, vision)

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(vq2_module, "RESET_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", no_delay)

    proof = asyncio.run(runner.emergency_reset())

    assert proof is None
    assert adapter.reset_calls == 1


def test_cleanup_reset_without_clock_baseline_reacts_to_newer_armed_heartbeat(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.heartbeat_sequence = 55
    adapter.is_armed = False
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    clock = [0.0]
    disarm_observations = []

    async def advance(delay):
        clock[0] += delay
        adapter.heartbeat_sequence = 56
        adapter.is_armed = True

    async def disarm():
        disarm_observations.append(
            (
                round(clock[0], 6),
                adapter.heartbeat_sequence,
                adapter.is_armed,
            )
        )
        adapter.is_armed = False

    monkeypatch.setattr(vq2_module, "RESET_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)
    monkeypatch.setattr(adapter, "disarm", disarm)

    proof = asyncio.run(runner.emergency_reset())

    assert proof is None
    assert adapter.reset_calls == 1
    assert disarm_observations[:2] == [
        (0.0, 55, False),
        (0.005, 56, True),
    ]
    cadence_times = [row[0] for row in disarm_observations[2:]]
    assert len(cadence_times) == 24
    assert cadence_times[0] == pytest.approx(0.025)
    assert cadence_times[-1] < 0.5
    assert all(
        vq2_module.CONTROL_PERIOD_S - 1e-12
        <= later - earlier
        <= vq2_module.CONTROL_PERIOD_S + 0.005 + 1e-12
        for earlier, later in zip(cadence_times, cadence_times[1:])
    )
    assert all(
        row[1:] == (56, True) for row in disarm_observations[2:]
    )


def test_cleanup_reset_requests_atomic_disarm_before_proof_wait_without_claim(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.heartbeat_sequence = 55
    adapter.is_armed = False
    adapter.race_status = RaceStatus(10_000, 0, -1, 0, -1)
    adapter.latest_telemetry = TelemetryState(
        timestamp_us=0,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion(),
        angular_velocity=(0.0, 0.0, 0.0),
        imu=IMUData(
            timestamp_us=10_000_000,
            accel=(0.0, 0.0, -9.81),
            gyro=(0.0, 0.0, 0.0),
        ),
    )
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )
    events = []
    reset_with_boundary = adapter.reset_calibration_with_boundary

    async def reset_then_return(
        persist_boundary,
        *,
        cleanup_post_reset_disarm,
    ):
        assert cleanup_post_reset_disarm is True
        boundary = await reset_with_boundary(
            persist_boundary,
            cleanup_post_reset_disarm=cleanup_post_reset_disarm,
        )
        events.append("reset_returned")
        return boundary

    async def disarm():
        events.append("post_reset_disarm")

    async def observe(**kwargs):
        assert kwargs["cleanup_disarm_heartbeat_anchor"] == 55
        events.append("proof_wait")
        return proof

    monkeypatch.setattr(
        adapter,
        "reset_calibration_with_boundary",
        reset_then_return,
    )
    monkeypatch.setattr(adapter, "disarm", disarm)
    monkeypatch.setattr(runner, "_observe_reset_proof", observe)

    assert asyncio.run(runner.emergency_reset()) is proof
    assert adapter.cleanup_post_reset_disarm_requests == 1
    assert events == [
        "reset_returned",
        "proof_wait",
    ]


@pytest.mark.parametrize("disarm_raises", [False, True])
def test_cleanup_reset_proof_cuts_on_rollback_then_newer_armed_heartbeat(
    monkeypatch,
    disarm_raises,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    clock = [0.0]
    sample_index = [0]
    events = []

    def install_sample(index):
        adapter.race_status = RaceStatus(
            100 + 50 * index,
            1_000,
            -1,
            0,
            -1,
        )
        adapter.latest_telemetry = TelemetryState(
            timestamp_us=0,
            position_ned=(0.0, 0.0, 0.0),
            velocity_ned=(0.0, 0.0, 0.0),
            orientation=Quaternion(),
            angular_velocity=(0.0, 0.0, 0.0),
            imu=IMUData(
                timestamp_us=100_000 * (index + 1),
                accel=(0.0, 0.0, -9.81),
                gyro=(0.0, 0.0, 0.0),
            ),
        )
        if index < 2:
            adapter.heartbeat_sequence = 55
            adapter.is_armed = False
        elif index == 2:
            # A strictly newer disarmed heartbeat still grants no send.
            adapter.heartbeat_sequence = 56
            adapter.is_armed = False
        else:
            adapter.heartbeat_sequence = 57
            adapter.is_armed = True

    install_sample(0)

    async def disarm():
        events.append(
            (
                "reactive_disarm",
                adapter.heartbeat_sequence,
                adapter.is_armed,
            )
        )
        if disarm_raises:
            raise RuntimeError("injected reactive disarm failure")

    async def advance(delay):
        clock[0] += delay
        sample_index[0] += 1
        install_sample(sample_index[0])

    monkeypatch.setattr(adapter, "disarm", disarm)
    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)

    proof = asyncio.run(
        runner._observe_reset_proof(
            attempt=1,
            pre_race=10_000,
            pre_imu=10_000_000,
            cleanup_disarm_heartbeat_anchor=55,
        )
    )
    events.append("proof_returned")

    assert proof is not None
    assert proof.post_race_boot_ms == 300
    assert proof.post_imu_us == 500_000
    assert events == [
        ("reactive_disarm", 55, False),
        ("reactive_disarm", 57, True),
        "proof_returned",
    ]


def test_cleanup_reset_disarm_cadence_is_exact_and_drops_missed_ticks(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.heartbeat_sequence = 55
    adapter.is_armed = True
    runner = VQ2Runner(adapter, _FakeVision())
    clock = [0.0]
    disarm_times = []

    async def disarm():
        disarm_times.append(clock[0])

    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(adapter, "disarm", disarm)

    async def exercise():
        next_deadline = 0.0
        observed_deadlines = []
        for now_s in (0.0, 0.019, 0.020, 0.075, 0.075, 0.095):
            clock[0] = now_s
            next_deadline = await (
                runner._advance_cleanup_reset_disarm_cadence(
                    attempt=1,
                    heartbeat_anchor=55,
                    next_deadline_s=next_deadline,
                )
            )
            assert next_deadline is not None
            observed_deadlines.append(next_deadline)
        return observed_deadlines

    deadlines = asyncio.run(exercise())

    assert disarm_times == pytest.approx([0.0, 0.020, 0.075, 0.095])
    assert deadlines == pytest.approx(
        [0.020, 0.020, 0.040, 0.095, 0.095, 0.115]
    )


def test_cleanup_reset_disarm_cadence_continues_through_clock_rollback(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.heartbeat_sequence = 55
    adapter.is_armed = True
    adapter.race_status = RaceStatus(11_000, 0, -1, 0, -1)
    adapter.latest_telemetry = TelemetryState(
        timestamp_us=0,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion(),
        angular_velocity=(0.0, 0.0, 0.0),
        imu=IMUData(
            timestamp_us=11_000_000,
            accel=(0.0, 0.0, -9.81),
            gyro=(0.0, 0.0, 0.0),
        ),
    )
    runner = VQ2Runner(adapter, _FakeVision())
    clock = [0.0]
    disarm_calls = []

    async def advance(delay):
        clock[0] += delay
        if clock[0] >= 0.025:
            adapter.race_status = RaceStatus(100, 1_000, -1, 0, -1)
            adapter.latest_telemetry = TelemetryState(
                timestamp_us=0,
                position_ned=(0.0, 0.0, 0.0),
                velocity_ned=(0.0, 0.0, 0.0),
                orientation=Quaternion(),
                angular_velocity=(0.0, 0.0, 0.0),
                imu=IMUData(
                    timestamp_us=100_000,
                    accel=(0.0, 0.0, -9.81),
                    gyro=(0.0, 0.0, 0.0),
                ),
            )

    async def disarm():
        disarm_calls.append(clock[0])

    monkeypatch.setattr(vq2_module, "RESET_PROOF_TIMEOUT_S", 0.061)
    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)
    monkeypatch.setattr(adapter, "disarm", disarm)

    proof = asyncio.run(
        runner._observe_reset_proof(
            attempt=1,
            pre_race=10_000,
            pre_imu=10_000_000,
            cleanup_disarm_heartbeat_anchor=55,
        )
    )

    assert proof is None
    # 0.025 is the preserved one-shot rollback reaction.  It resets the
    # cadence deadline, which then continues through the bounded proof wait.
    assert disarm_calls == pytest.approx([0.0, 0.020, 0.025, 0.045])


def test_cleanup_reset_disarm_cadence_ignores_newer_disarmed_heartbeat(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.heartbeat_sequence = 56
    adapter.is_armed = False
    runner = VQ2Runner(adapter, _FakeVision())
    disarm_calls = []

    async def disarm():
        disarm_calls.append(True)

    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(adapter, "disarm", disarm)

    next_deadline = asyncio.run(
        runner._advance_cleanup_reset_disarm_cadence(
            attempt=1,
            heartbeat_anchor=55,
            next_deadline_s=0.0,
        )
    )

    assert next_deadline == pytest.approx(vq2_module.CONTROL_PERIOD_S)
    assert disarm_calls == [True]


def test_cleanup_reset_disarm_cadence_survives_attempt20_reset_rearm_sequence(
    monkeypatch,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    clock = [0.0]
    disarm_calls = []

    def install_state():
        now_s = round(clock[0], 6)
        if now_s < 0.040:
            race_boot_ms = 11_000
            imu_us = 11_000_000
        else:
            race_boot_ms = 150 if now_s >= 0.100 else 100
            if now_s >= 0.100:
                imu_us = 500_000
            elif now_s >= 0.090:
                imu_us = 400_000
            elif now_s >= 0.080:
                imu_us = 300_000
            elif now_s >= 0.060:
                imu_us = 200_000
            else:
                imu_us = 100_000
        adapter.race_status = RaceStatus(
            race_boot_ms,
            -1,
            -1,
            0,
            -1,
        )
        adapter.latest_telemetry = TelemetryState(
            timestamp_us=0,
            position_ned=(0.0, 0.0, 0.0),
            velocity_ned=(0.0, 0.0, 0.0),
            orientation=Quaternion(),
            angular_velocity=(0.0, 0.0, 0.0),
            imu=IMUData(
                timestamp_us=imu_us,
                accel=(0.0, 0.0, -9.81),
                gyro=(0.0, 0.0, 0.0),
            ),
        )
        if now_s < 0.010:
            adapter.heartbeat_sequence = 79
            adapter.is_armed = True
        elif now_s < 0.080:
            adapter.heartbeat_sequence = 80
            adapter.is_armed = False
        else:
            adapter.heartbeat_sequence = 91
            adapter.is_armed = True

    install_state()

    async def disarm():
        disarm_calls.append(
            (
                round(clock[0], 6),
                adapter.heartbeat_sequence,
                adapter.is_armed,
            )
        )

    async def advance(delay):
        clock[0] += delay
        install_state()

    monkeypatch.setattr(vq2_module, "RESET_PROOF_TIMEOUT_S", 0.121)
    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)
    monkeypatch.setattr(adapter, "disarm", disarm)

    proof = asyncio.run(
        runner._observe_reset_proof(
            attempt=1,
            pre_race=10_000,
            pre_imu=10_000_000,
            cleanup_disarm_heartbeat_anchor=79,
        )
    )

    assert proof is not None
    assert proof.post_race_boot_ms == 150
    assert proof.post_imu_us == 500_000
    assert disarm_calls == [
        (0.0, 79, True),
        (0.02, 80, False),
        (0.04, 80, False),
        (0.065, 80, False),
        (0.08, 91, True),
        (0.1, 91, True),
    ]
    assert all(
        later[0] - earlier[0]
        <= vq2_module.CONTROL_PERIOD_S + 0.005 + 1e-12
        for earlier, later in zip(disarm_calls, disarm_calls[1:])
    )


def test_cleanup_reset_disarm_send_failure_does_not_suppress_reset_proof(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.heartbeat_sequence = 55
    adapter.is_armed = True
    adapter.race_status = RaceStatus(11_000, 0, -1, 0, -1)
    adapter.latest_telemetry = TelemetryState(
        timestamp_us=0,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion(),
        angular_velocity=(0.0, 0.0, 0.0),
        imu=IMUData(
            timestamp_us=11_000_000,
            accel=(0.0, 0.0, -9.81),
            gyro=(0.0, 0.0, 0.0),
        ),
    )
    runner = VQ2Runner(adapter, _FakeVision())
    clock = [0.0]
    sample_index = [0]
    disarm_calls = []

    async def advance(delay):
        clock[0] += delay
        sample_index[0] += 1
        index = sample_index[0]
        adapter.race_status = RaceStatus(
            100 + 50 * (index - 1),
            1_000,
            -1,
            0,
            -1,
        )
        adapter.latest_telemetry = TelemetryState(
            timestamp_us=0,
            position_ned=(0.0, 0.0, 0.0),
            velocity_ned=(0.0, 0.0, 0.0),
            orientation=Quaternion(),
            angular_velocity=(0.0, 0.0, 0.0),
            imu=IMUData(
                timestamp_us=100_000 * index,
                accel=(0.0, 0.0, -9.81),
                gyro=(0.0, 0.0, 0.0),
            ),
        )

    async def failed_disarm():
        disarm_calls.append(clock[0])
        raise RuntimeError("injected disarm send failure")

    monkeypatch.setattr(vq2_module, "RESET_PROOF_TIMEOUT_S", 0.1)
    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)
    monkeypatch.setattr(adapter, "disarm", failed_disarm)

    proof = asyncio.run(
        runner._observe_reset_proof(
            attempt=1,
            pre_race=10_000,
            pre_imu=10_000_000,
            cleanup_disarm_heartbeat_anchor=55,
        )
    )

    assert proof is not None
    assert proof.post_race_boot_ms == 300
    assert proof.post_imu_us == 500_000
    assert disarm_calls == pytest.approx([0.0, 0.005, 0.025])


def test_safe_cleanup_does_not_claim_pre_reset_disarm_confirmation(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.is_armed = False
    adapter.race_status = RaceStatus(10_000, 0, -1, 0, -1)
    adapter.latest_telemetry = SimpleNamespace(
        imu=SimpleNamespace(timestamp_us=10_000_000)
    )
    runner = VQ2Runner(adapter, _FakeVision())
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )
    events = []

    async def pre_reset_disarm():
        events.append("pre_reset_disarm_returned")

    async def final_disarm_confirmed(*_args, **_kwargs):
        events.append("final_disarm_confirmation")
        return False

    async def reset():
        events.append("reset")

    async def observe_reset(**_kwargs):
        return proof

    monkeypatch.setattr(adapter, "disarm", pre_reset_disarm)
    monkeypatch.setattr(
        runner,
        "_disarm_confirmed",
        final_disarm_confirmed,
    )
    monkeypatch.setattr(adapter, "reset", reset)
    monkeypatch.setattr(runner, "_observe_reset_proof", observe_reset)

    assert asyncio.run(runner.safe_cleanup()) is False
    assert events == [
        "pre_reset_disarm_returned",
        "reset",
        "final_disarm_confirmation",
    ]


def test_disarm_confirmation_is_not_erased_by_recorder_failure(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    runner = VQ2Runner(adapter, _FakeVision())

    async def disarm():
        adapter.heartbeat_sequence += 1
        adapter.is_armed = False

    def recorder_failure(*_args, **_kwargs):
        raise RuntimeError("injected recorder failure")

    monkeypatch.setattr(adapter, "disarm", disarm)
    monkeypatch.setattr(runner.recorder, "emit", recorder_failure)

    assert asyncio.run(runner._disarm_confirmed()) is True


def test_safe_cleanup_orders_zero_disarm_send_reset_then_final_proof(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(10_000, 0, -1, 0, -1)
    adapter.latest_telemetry = SimpleNamespace(
        imu=SimpleNamespace(timestamp_us=10_000_000)
    )
    runner = VQ2Runner(adapter, _FakeVision())
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )
    order = []
    recorded = []

    async def zero_send(_command):
        order.append("zero")

    async def pre_reset_disarm():
        order.append("pre_reset_disarm")

    async def reset():
        order.append("reset")

    async def observe_reset(**_kwargs):
        return proof

    async def final_disarm_confirmed(*_args, **_kwargs):
        order.append("final_disarm_confirmation")
        adapter.is_armed = False
        return True

    monkeypatch.setattr(adapter, "send_attitude_rate", zero_send)
    monkeypatch.setattr(adapter, "disarm", pre_reset_disarm)
    monkeypatch.setattr(adapter, "reset", reset)
    monkeypatch.setattr(runner, "_observe_reset_proof", observe_reset)
    monkeypatch.setattr(
        runner,
        "_disarm_confirmed",
        final_disarm_confirmed,
    )
    monkeypatch.setattr(
        runner.recorder,
        "emit",
        lambda event, **payload: recorded.append((event, payload)),
    )

    assert asyncio.run(runner.safe_cleanup()) is True
    assert runner._cleanup_proved_reset_epoch is True
    assert order == [
        "zero",
        "pre_reset_disarm",
        "reset",
        "final_disarm_confirmation",
    ]
    pre_reset_events = [
        payload
        for event, payload in recorded
        if event == "cleanup_pre_reset_disarm_attempt"
    ]
    assert pre_reset_events == [
        {
            "outcome": "returned",
            "error_type": None,
            "heartbeat_sequence_before": 1,
            "armed_before": True,
        }
    ]


def test_safe_cleanup_reset_is_unconditional_after_pre_reset_disarm_failure(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.is_armed = False
    adapter.race_status = RaceStatus(10_000, 0, -1, 0, -1)
    adapter.latest_telemetry = SimpleNamespace(
        imu=SimpleNamespace(timestamp_us=10_000_000)
    )
    runner = VQ2Runner(adapter, _FakeVision())
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )
    order = []
    recorded = []

    async def failed_pre_reset_disarm():
        order.append("pre_reset_disarm")
        raise RuntimeError("injected pre-reset send failure")

    async def reset():
        order.append("reset")

    async def observe_reset(**_kwargs):
        return proof

    async def final_disarm_confirmed(*_args, **_kwargs):
        order.append("final_disarm_confirmation")
        return True

    monkeypatch.setattr(adapter, "disarm", failed_pre_reset_disarm)
    monkeypatch.setattr(adapter, "reset", reset)
    monkeypatch.setattr(runner, "_observe_reset_proof", observe_reset)
    monkeypatch.setattr(
        runner,
        "_disarm_confirmed",
        final_disarm_confirmed,
    )
    monkeypatch.setattr(
        runner.recorder,
        "emit",
        lambda event, **payload: recorded.append((event, payload)),
    )

    assert asyncio.run(runner.safe_cleanup()) is True
    assert order == [
        "pre_reset_disarm",
        "reset",
        "final_disarm_confirmation",
    ]
    pre_reset_events = [
        payload
        for event, payload in recorded
        if event == "cleanup_pre_reset_disarm_attempt"
    ]
    assert pre_reset_events == [
        {
            "outcome": "raised",
            "error_type": "RuntimeError",
            "heartbeat_sequence_before": 1,
            "armed_before": False,
        }
    ]


def test_cleanup_atomic_reset_boundary_retains_harmful_collision(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(10_000, 0, -1, 3, 500)
    adapter.latest_telemetry = TelemetryState(
        timestamp_us=0,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion(),
        angular_velocity=(0.0, 0.0, 0.0),
        imu=IMUData(
            timestamp_us=10_000_000,
            accel=(0.0, 0.0, -9.81),
            gyro=(0.0, 0.0, 0.0),
        ),
    )
    adapter.collisions = [
        {"id": 9, "threat_level": 2, "impulse": 0.2}
    ]
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    async def observe(**_kwargs):
        return proof

    monkeypatch.setattr(runner, "_observe_reset_proof", observe)

    assert asyncio.run(runner.emergency_reset()) is proof
    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is False
    assert safety["capture_complete"] is True
    assert safety["harmful_collision_count"] == 1
    assert safety["observations"] == [
        {
            "phase": "cleanup-atomic-reset-boundary",
            "disposition": "harmful",
            "collision": {
                "id": 9,
                "threat_level": 2,
                "impulse": 0.2,
            },
        }
    ]
    assert adapter.collisions == []


def test_cleanup_post_reset_exact_pad_contact_is_separate_and_bounded():
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    adapter.collisions = [
        {
            "id": 1002,
            "threat_level": 2,
            "impulse": 1.013804316520691,
        }
    ]
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is True
    assert safety["harmful_collision_count"] == 0
    assert safety["benign_reset_pad_contact_count"] == 1
    assert safety["benign_reset_pad_cumulative_impulse"] == pytest.approx(
        1.013804316520691
    )
    assert safety["observations"][0]["disposition"] == "benign_reset_pad"


def test_cleanup_accepts_reproduced_proved_reset_settling_aggregate():
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    reproduced_max_impulse = 1.013804316520691
    reproduced_cumulative_impulse = 4.59668993670493
    remaining_impulse = (
        reproduced_cumulative_impulse - reproduced_max_impulse
    ) / 48
    adapter.collisions = [
        {
            "id": 1002,
            "threat_level": 2,
            "impulse": reproduced_max_impulse,
        },
        *[
            {
                "id": 1002,
                "threat_level": 1 + index % 2,
                "impulse": remaining_impulse,
            }
            for index in range(48)
        ],
    ]
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is True
    assert safety["harmful_collision_count"] == 0
    assert safety["benign_pre_reset_pad_contact_count"] == 0
    assert safety["benign_reset_pad_contact_count"] == 49
    assert safety["benign_reset_pad_cumulative_impulse"] == pytest.approx(
        reproduced_cumulative_impulse
    )
    assert {row["disposition"] for row in safety["observations"]} == {
        "benign_reset_pad"
    }


def test_cleanup_accepts_exact_visual_course_reset_settling_after_disarm():
    impulses = (
        0.12948261201381683,
        0.21557387709617615,
        0.0692632794380188,
        0.11583398282527924,
        0.021046079695224762,
        0.03382188081741333,
        0.0033150799572467804,
        0.004094791132956743,
        0.00811521615833044,
        0.01246686466038227,
        0.008760632947087288,
        0.013718180358409882,
        0.012038972228765488,
        0.019382087513804436,
        0.016297858208417892,
        0.026634542271494865,
        0.02003510482609272,
        0.03291308879852295,
        0.019091397523880005,
        0.031255077570676804,
        0.016673045232892036,
        0.027192195877432823,
        0.014427227899432182,
        0.02339250035583973,
        0.01316026784479618,
        0.021224338561296463,
        0.012569993734359741,
        0.020187407732009888,
        0.012278366833925247,
        0.019656142219901085,
        0.012167157605290413,
        0.019426578655838966,
        0.011905090883374214,
        0.018958456814289093,
        0.011677568778395653,
        0.018545037135481834,
        0.011409454047679901,
        0.018068790435791016,
        0.011132237501442432,
        0.01757722534239292,
        0.010713047347962856,
        0.016864273697137833,
        0.010473364032804966,
        0.016438674181699753,
        0.010240611620247364,
        0.016030406579375267,
        0.009946745820343494,
        0.015527691692113876,
        0.009669233113527298,
        0.015057357028126717,
        0.00946666020900011,
        0.014707034453749657,
        0.009267580695450306,
        0.014370506629347801,
        0.00901004672050476,
        0.013944591395556927,
        0.00876525416970253,
        0.013544324785470963,
        0.008637683466076851,
        0.013330018147826195,
        0.008403528481721878,
        0.01295925211161375,
        0.00817222148180008,
        0.01259362231940031,
        0.008050368167459965,
        0.012405086308717728,
        0.007834559306502342,
        0.012075582519173622,
        0.007619590498507023,
        0.0117528410628438,
        0.007503012660890818,
        0.011584402993321419,
        0.007300864439457655,
        0.011293460614979267,
        0.007118364796042442,
        0.01103296596556902,
        0.01504996232688427,
        0.09983597695827484,
        0.31848597526550293,
        0.3299800157546997,
        0.07763012498617172,
        0.027805613353848457,
        0.02355489507317543,
        0.025834757834672928,
        0.034088119864463806,
    )
    assert len(impulses) == 85
    assert vq2_module.MAX_PROVED_RESET_PAD_CONTACTS == 96
    adapter = _FakeAdapter()
    adapter.is_armed = True
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": impulse}
        for impulse in impulses
    ]
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=3_804,
        post_race_boot_ms=152,
        pre_imu_us=3_944_703,
        post_imu_us=375_078,
        advancing_race_samples=2,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)

    assert adapter.collisions == []
    pending = runner._cleanup_collision_safety_summary()
    assert pending["safe"] is False
    assert pending["pending_reset_collision_batch_count"] == 1
    assert pending["pending_reset_collision_count"] == 85

    adapter.is_armed = False
    runner._classify_quarantined_cleanup_reset_collisions()
    runner._classify_quarantined_cleanup_reset_collisions()

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is True
    assert safety["capture_complete"] is True
    assert safety["harmful_collision_count"] == 0
    assert safety["benign_reset_pad_contact_count"] == 85
    assert safety["benign_reset_pad_cumulative_impulse"] == pytest.approx(
        2.5007399604655802
    )
    assert safety["pending_reset_collision_batch_count"] == 0
    assert safety["pending_reset_collision_count"] == 0
    assert len(safety["observations"]) == 85
    assert {row["disposition"] for row in safety["observations"]} == {
        "benign_reset_pad"
    }


def test_cleanup_rejects_exact_high_energy_visual_course_reset_contact():
    impulses = (
        2.3850607872009277,
        0.02875417098402977,
        0.0446980856359005,
        0.28195932507514954,
        0.14218781888484955,
        0.026842674240469933,
        0.02054809033870697,
        0.04731355607509613,
        0.06768640875816345,
        0.08198506385087967,
        0.09091061353683472,
    )
    adapter = _FakeAdapter()
    adapter.is_armed = True
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    adapter.collisions = [
        {
            "id": 1002,
            "threat_level": 2 if index == 0 else 1,
            "impulse": impulse,
        }
        for index, impulse in enumerate(impulses)
    ]
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=4_067,
        post_race_boot_ms=216,
        pre_imu_us=4_152_508,
        post_imu_us=388_831,
        advancing_race_samples=2,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)

    pending = runner._cleanup_collision_safety_summary()
    assert pending["safe"] is False
    assert pending["pending_reset_collision_count"] == 11

    adapter.is_armed = False
    runner._classify_quarantined_cleanup_reset_collisions()

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is False
    assert safety["capture_complete"] is True
    assert safety["harmful_collision_count"] == 1
    assert safety["benign_reset_pad_contact_count"] == 10
    assert safety["benign_reset_pad_cumulative_impulse"] == pytest.approx(
        0.8328858073800802
    )
    assert len(safety["observations"]) == 11
    assert safety["observations"][0] == {
        "phase": "cleanup-post-reset-proof",
        "disposition": "harmful",
        "collision": {
            "id": 1002,
            "threat_level": 2,
            "impulse": 2.3850607872009277,
        },
    }
    assert {row["disposition"] for row in safety["observations"][1:]} == {
        "benign_reset_pad"
    }


def test_cleanup_rejects_exact_attempt16_high_energy_reset_batch():
    impulses = (
        0.1466638594865799,
        4.1180338859558105,
        0.19095055758953094,
        0.1753886342048645,
        0.08287937194108963,
        0.07079105079174042,
        0.0691244974732399,
        0.061687011271715164,
        0.051315177232027054,
        0.04146957769989967,
        0.0427708774805069,
        0.046275921165943146,
        0.04794364795088768,
        0.04766120761632919,
        0.04559176415205002,
        0.04379289224743843,
        0.041580114513635635,
        0.0400996096432209,
        0.03893681615591049,
        0.03808487206697464,
        0.03750221058726311,
        0.03688058257102966,
        0.0365213118493557,
        0.03598038852214813,
        0.03544771671295166,
        0.03490550443530083,
        0.03435870632529259,
        0.03385394811630249,
        0.03336295485496521,
        0.03292267397046089,
        0.032543573528528214,
        0.03211642801761627,
        0.031827691942453384,
        0.03149475157260895,
        0.031158536672592163,
        0.03090137429535389,
        0.030699390918016434,
        0.03036053664982319,
        0.03014645352959633,
        0.029964374378323555,
        0.02973053604364395,
        0.02953367494046688,
        0.029382750391960144,
        0.029180854558944702,
        0.029059845954179764,
        0.02888437733054161,
        0.028735937550663948,
        0.028646469116210938,
        0.028477296233177185,
    )
    assert len(impulses) == 49
    adapter = _FakeAdapter()
    adapter.is_armed = True
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    adapter.collisions = [
        {
            "id": 1002,
            "threat_level": 2 if index == 1 else 1,
            "impulse": impulse,
        }
        for index, impulse in enumerate(impulses)
    ]
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=6_245,
        post_race_boot_ms=250,
        pre_imu_us=6_475_585,
        post_imu_us=381_884,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)

    pending = runner._cleanup_collision_safety_summary()
    assert pending["safe"] is False
    assert pending["pending_reset_collision_count"] == 49

    adapter.is_armed = False
    runner._classify_quarantined_cleanup_reset_collisions()

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is False
    assert safety["capture_complete"] is True
    assert safety["harmful_collision_count"] == 1
    assert safety["benign_reset_pad_contact_count"] == 48
    assert safety["benign_reset_pad_cumulative_impulse"] == pytest.approx(
        2.247588312253356
    )
    assert safety["observations"][1] == {
        "phase": "cleanup-post-reset-proof",
        "disposition": "harmful",
        "collision": {
            "id": 1002,
            "threat_level": 2,
            "impulse": 4.1180338859558105,
        },
    }
    assert {
        row["disposition"]
        for index, row in enumerate(safety["observations"])
        if index != 1
    } == {"benign_reset_pad"}


def test_cleanup_post_reset_pad_contact_budget_excess_is_unsafe():
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": 0.01}
        for _ in range(vq2_module.MAX_PROVED_RESET_PAD_CONTACTS + 1)
    ]
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is False
    assert safety["harmful_collision_count"] == 1
    assert safety["observations"][-1]["disposition"] == (
        "reset_pad_budget_exceeded"
    )


@pytest.mark.parametrize(
    "collision",
    [
        {"id": 1001, "threat_level": 2, "impulse": 1.0},
        {"id": 1002, "threat_level": 3, "impulse": 1.0},
        {
            "id": 1002,
            "threat_level": 2,
            "impulse": (
                vq2_module.MAX_PROVED_RESET_PAD_SINGLE_IMPULSE + 1e-6
            ),
        },
        {"id": 1002, "threat_level": 2, "impulse": -0.001},
    ],
)
def test_cleanup_proved_reset_pad_settling_rejects_wrong_identity_or_energy(
    collision,
):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    adapter.collisions = [collision]
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is False
    assert safety["harmful_collision_count"] == 1
    assert safety["observations"][0]["disposition"] == "harmful"


def test_cleanup_proved_reset_pad_settling_waits_for_disarmed_state():
    adapter = _FakeAdapter()
    adapter.is_armed = True
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    adapter.collisions = [
        {"id": 1002, "threat_level": 2, "impulse": 1.0}
    ]
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)

    assert adapter.collisions == []
    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is False
    assert safety["harmful_collision_count"] == 0
    assert safety["benign_reset_pad_contact_count"] == 0
    assert safety["pending_reset_collision_batch_count"] == 1
    assert safety["pending_reset_collision_count"] == 1

    adapter.is_armed = False
    runner._classify_quarantined_cleanup_reset_collisions()

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is True
    assert safety["harmful_collision_count"] == 0
    assert safety["benign_reset_pad_contact_count"] == 1
    assert safety["pending_reset_collision_batch_count"] == 0
    assert safety["pending_reset_collision_count"] == 0
    assert safety["observations"][0]["disposition"] == "benign_reset_pad"


def test_cleanup_quarantined_reset_batch_retains_dropped_accounting(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": 0.05}
    ]
    monkeypatch.setattr(
        adapter,
        "collision_stats",
        lambda: MavlinkCollisionStats(
            generation=2,
            handled=1,
            dropped=1,
            high_watermark=2,
            capacity=128,
            buffered=1,
        ),
    )
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)
    adapter.is_armed = False
    runner._classify_quarantined_cleanup_reset_collisions()

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is False
    assert safety["capture_complete"] is False
    assert safety["harmful_collision_count"] == 1
    assert safety["observations"][0]["disposition"] == (
        "collision_buffer_dropped"
    )
    assert safety["observations"][1]["disposition"] == "benign_reset_pad"


def test_cleanup_proved_reset_pad_cumulative_impulse_excess_is_unsafe():
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    adapter.collisions = [
        {"id": 1002, "threat_level": 2, "impulse": 1.0}
        for _ in range(6)
    ]
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is False
    assert safety["harmful_collision_count"] == 1
    assert safety["observations"][-1]["disposition"] == (
        "reset_pad_budget_exceeded"
    )


def test_cleanup_pre_reset_policy_rejects_proved_reset_settling_energy():
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())

    runner._record_cleanup_collision_batch(
        [{"id": 1002, "threat_level": 2, "impulse": 1.0}],
        phase="cleanup-atomic-reset-boundary",
        allow_benign_reset_pad=True,
    )

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is False
    assert safety["harmful_collision_count"] == 1
    assert safety["observations"][0]["disposition"] == "harmful"


def test_cleanup_proved_reset_flag_alone_cannot_grant_settling_authority():
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    runner._cleanup_in_progress = True

    runner._record_cleanup_collision_batch(
        [{"id": 1002, "threat_level": 2, "impulse": 1.0}],
        phase="cleanup-post-reset-proof",
        allow_benign_reset_pad=False,
        allow_proved_reset_pad_settling=True,
    )

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is False
    assert safety["harmful_collision_count"] == 1
    assert safety["observations"][0]["disposition"] == "harmful"


def test_cleanup_pre_reset_and_proved_reset_budgets_are_independent():
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    runner._cleanup_in_progress = True
    runner._record_cleanup_collision_batch(
        [
            {"id": 1002, "threat_level": 1, "impulse": 0.002}
            for _ in range(vq2_module.MAX_BENIGN_PAD_CONTACTS)
        ],
        phase="cleanup-atomic-reset-boundary",
        allow_benign_reset_pad=True,
        benign_pad_disposition="benign_calibration_pad",
    )
    adapter.collisions = [
        {"id": 1002, "threat_level": 2, "impulse": 0.07}
        for _ in range(49)
    ]
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )

    runner._accept_reset_proof(proof, restart_vision=False)

    safety = runner._cleanup_collision_safety_summary()
    assert safety["safe"] is True
    assert safety["benign_pre_reset_pad_contact_count"] == 12
    assert safety["benign_reset_pad_contact_count"] == 49
    assert safety["benign_pre_reset_pad_cumulative_impulse"] == pytest.approx(
        0.024
    )
    assert safety["benign_reset_pad_cumulative_impulse"] == pytest.approx(
        3.43
    )


def test_powered_stage_cleanup_collision_warning_retains_navigation_success(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(1_000, 0, -1, 0, -1)
    adapter.latest_telemetry = SimpleNamespace(
        imu=SimpleNamespace(timestamp_us=10_000_000)
    )
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(
        0.0,
        -0.31,
        322,
        174,
        6_400,
        1_000,
    )

    async def no_op(*_args, **_kwargs):
        return None

    async def wait_for_go():
        return context

    async def stage_body(*, calibration_context):
        assert calibration_context is context
        adapter.collisions = [
            {"id": 7, "threat_level": 2, "impulse": 0.1}
        ]
        return {"yaw_calibration": {"yaw_identified": True}}

    async def disarm(*_args, **_kwargs):
        adapter.is_armed = False
        return True

    async def observe_reset(**_kwargs):
        return ResetProof(
            attempt=1,
            pre_race_boot_ms=10_000,
            post_race_boot_ms=500,
            pre_imu_us=10_000_000,
            post_imu_us=500_000,
            advancing_race_samples=3,
            advancing_imu_samples=5,
            countdown_observed=True,
        )

    monkeypatch.setattr(runner, "establish_reset_epoch", no_op)
    monkeypatch.setattr(runner, "normalize_disarmed", no_op)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(runner, "arm_confirmed", no_op)
    monkeypatch.setattr(runner, "_run_sign_id", stage_body)
    monkeypatch.setattr(runner, "_disarm_confirmed", disarm)
    monkeypatch.setattr(runner, "_observe_reset_proof", observe_reset)

    result = asyncio.run(runner.run_powered_stage("sign-id"))

    assert result.success is True
    assert result.cleanup_confirmed is True
    assert result.reason == "stage completed"
    safety = result.details["cleanup_collision_safety"]
    assert safety["safe"] is False
    assert safety["harmful_collision_count"] == 1
    assert safety["observations"][0]["phase"] == "cleanup-entry"


def test_visual_course_cleanup_collision_does_not_downgrade_nested_success(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(1_500, 0, 42, 0, -1)
    profile = vq2_module.load_yaw_calibration_profile()
    profile_evidence = (
        vq2_module.build_yaw_calibration_profile_evidence(profile)
    )
    runner = VQ2Runner(
        adapter,
        _FakeVision(),
        yaw_calibration_profile=profile,
        yaw_calibration_profile_evidence=profile_evidence,
    )
    context = vq2_module.StartContext(
        0.0,
        -0.31,
        322,
        174,
        6_400,
        1_000,
    )
    finish_ref = vq2_module.AuthoritativeRaceStatusRef.live(
        session_id="cleanup-summary-test",
        reset_epoch=1,
        race_generation=2,
        race_status_sequence=3,
        race_status_boot_ms=1_500,
        active_gate_index=0,
        received_monotonic_ns=500_000_000,
        host_clock_id="host-perf-counter",
        race_finished=True,
    )
    runner.visual_gate_graph._latest_snapshot = SimpleNamespace(
        confirmed_transitions=(),
        race_finished=True,
        current_gate_index=0,
        latest_race_status=finish_ref,
    )

    async def no_op(*_args, **_kwargs):
        return None

    async def wait_for_go():
        return context

    def bind_initial(_context):
        return None

    async def run_course(_context):
        summary = {
            "stage": vq2_module.VISUAL_COURSE_STAGE,
            "success": True,
            "race_finished": True,
            "outcome": "race_finished",
            "first_causal_blocker": None,
            "initial_gate_index": 0,
            "final_gate_index": 0,
            "maximum_authoritative_gate_index": 0,
            "authoritative_transitions": [],
            "yaw_calibration_profile": profile_evidence,
        }
        runner._visual_course_summary = dict(summary)
        return summary

    async def unsafe_cleanup():
        runner._cleanup_harmful_collision_count = 1
        runner._cleanup_collision_observations.append(
            {
                "phase": "cleanup-entry",
                "disposition": "harmful",
                "collision": {
                    "id": 7,
                    "threat_level": 2,
                    "impulse": 0.1,
                },
            }
        )
        return True

    monkeypatch.setattr(runner, "establish_reset_epoch", no_op)
    monkeypatch.setattr(runner, "normalize_disarmed", no_op)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(
        runner,
        "_bind_initial_visual_gate",
        bind_initial,
    )
    monkeypatch.setattr(runner, "arm_confirmed", no_op)
    monkeypatch.setattr(runner, "_run_visual_course", run_course)
    monkeypatch.setattr(runner, "safe_cleanup", unsafe_cleanup)

    result = asyncio.run(
        runner.run_powered_stage(
            vq2_module.VISUAL_COURSE_STAGE,
            write_diagnostic_pngs=False,
        )
    )

    assert result.success is True
    assert result.cleanup_confirmed is True
    assert result.details["cleanup_collision_safety"]["safe"] is False
    nested = result.details["visual_course"]
    assert nested["success"] is True
    assert nested["outcome"] == "race_finished"
    assert nested["first_causal_blocker"] is None


def test_live_finalizer_preserves_existing_cleanup_cause_for_empty_tail(
    tmp_path,
    monkeypatch,
):
    from competition.aigp_mavlink import MavlinkOutboundAudit

    class LiveAdapter(_FakeAdapter):
        async def connect(self, _address):
            return None

        async def disconnect(self):
            return None

        def drain_received_ingress(self):
            return []

        def ingress_stats(self):
            return MavlinkIngressStats(
                generation=1,
                next_sequence=0,
                highres_imu_received=0,
                heartbeat_received=0,
                race_status_received=0,
                actuator_received=0,
                dropped=0,
                high_watermark=0,
                imu_capacity=1,
                other_capacity=1,
                imu_dropped=0,
                other_dropped=0,
                imu_high_watermark=0,
                other_high_watermark=0,
                buffered_imu=0,
                buffered_other=0,
            )

        def outbound_audit(self):
            return MavlinkOutboundAudit(0, 0, 0, 0, 0, 0, 0, 0)

    adapter = LiveAdapter()
    vision = VQ2VisionThread()
    runner = VQ2Runner(adapter, vision)
    runner._cleanup_in_progress = True
    runner._cleanup_harmful_collision_count = 1
    runner._cleanup_collision_observations.append(
        {
            "phase": "cleanup-entry",
            "disposition": "harmful",
            "collision": {
                "id": 7,
                "threat_level": 2,
                "impulse": 0.1,
            },
        }
    )
    cleanup_reason = "cleanup collision evidence was unsafe or incomplete"
    stage_reason = f"visual course finished; {cleanup_reason}"
    stage_result = vq2_module.StageResult(
        stage=vq2_module.VISUAL_COURSE_STAGE,
        success=False,
        reason=stage_reason,
        duration_s=1.0,
        gate_index_before=0,
        gate_index_after=0,
        cleanup_confirmed=True,
        details={
            "cleanup_collision_safety": (
                runner._cleanup_collision_safety_summary()
            ),
            "visual_course": {
                "success": False,
                "outcome": "abort",
                "reason": cleanup_reason,
                "first_causal_blocker": cleanup_reason,
            },
        },
        controller={},
    )

    async def run_stage(*_args, **_kwargs):
        return stage_result

    monkeypatch.setattr(runner, "run_powered_stage", run_stage)
    monkeypatch.setattr(
        vq2_module,
        "_load_live_transport_dependencies",
        lambda: None,
    )
    monkeypatch.setattr(
        vq2_module,
        "AIGPMavlinkAdapter",
        lambda **_kwargs: adapter,
    )
    monkeypatch.setattr(
        vq2_module,
        "VQ2VisionThread",
        lambda **_kwargs: vision,
    )
    monkeypatch.setattr(
        vq2_module,
        "VQ2Runner",
        lambda *_args, **_kwargs: runner,
    )
    visual_config = vq2_module.default_visual_config()

    result = asyncio.run(
        vq2_module.run_live(
            vq2_module.VISUAL_COURSE_STAGE,
            "udpin:127.0.0.1:14550",
            str(tmp_path / "empty-tail.jsonl.gz"),
            replay_bundle=None,
            recording_approved=False,
            preflight_before_powered_stage=False,
            write_diagnostic_pngs=False,
            run_manifest_sha256="a" * 64,
            candidate_commit="b" * 40,
            expected_controller_config_sha256=(
                visual_config.effective_config_sha256
            ),
            expected_yaw_calibration_profile_sha256=(
                vq2_module.YAW_CALIBRATION_PROFILE_SHA256
            ),
        )
    )

    assert result.success is False
    assert result.reason == stage_reason
    assert "post-disconnect" not in result.reason
    nested = result.details["visual_course"]
    assert nested["reason"] == cleanup_reason
    assert nested["first_causal_blocker"] == cleanup_reason
    safety = result.details["cleanup_collision_safety"]
    assert safety["harmful_collision_count"] == 1
    assert len(safety["observations"]) == 1


def test_invalid_imu_after_bootstrap_latches_estimator_failure():
    adapter = _FakeAdapter()
    vision = _FakeVision()
    runner = VQ2Runner(adapter, vision)
    runner.estimator = ImuAttitudeEstimator(
        ImuAttitudeConfig(
            calibration_min_samples=1,
            calibration_min_duration_s=0.0,
            gravity_correction_kp=0.0,
            gyro_bias_ki=0.0,
        )
    )
    adapter.imu_samples = [
        IMUData(
            timestamp_us=1_000_000,
            accel=(0.0, 0.0, -9.80665),
            gyro=(0.0, 0.0, 0.0),
        )
    ]
    runner._sample()
    assert runner.estimator.is_ready
    assert runner.estimate is not None

    adapter.imu_samples = [
        IMUData(
            timestamp_us=1_010_000,
            accel=(math.nan, 0.0, -9.80665),
            gyro=(0.0, 0.0, 0.0),
        )
    ]
    runner._sample()

    assert runner._last_imu_us == 1_010_000
    assert runner._estimator_unhealthy_latched
    failures = runner._stream_failures(
        require_estimator=True,
        require_target=False,
        require_armed=False,
    )
    assert any("attitude estimator failure latched" in failure for failure in failures)


def test_sampling_threads_exact_receiver_imu_envelope_to_recorder():
    imu = IMUData(
        timestamp_us=1_000_000,
        accel=(0.0, 0.0, -9.80665),
        gyro=(0.0, 0.0, 0.0),
    )
    received = ReceivedIMUSampleV1(
        ingress=MavlinkIngressV1(
            stream_id="vq2-mavlink-udp-14550",
            generation=1,
            sequence=0,
            message_type="HIGHRES_IMU",
            host_clock_id="host-perf-counter",
            received_monotonic_ns=123,
            source_time_value=imu.timestamp_us,
            source_time_unit="us",
        ),
        imu=imu,
    )

    class TimedAdapter(_FakeAdapter):
        def __init__(self):
            super().__init__()
            self.received = [received]

        def drain_received_imu_samples(self):
            values = self.received
            self.received = []
            return values

        def drain_mavlink_arrivals(self):
            return []

    class Recorder:
        capture_enabled = False

        def __init__(self):
            self.received = []

        def record_imu(self, sample, estimator, now_s, *, received_sample):
            self.received.append((sample, estimator, now_s, received_sample))

        def record_mavlink_ingress(self, _arrival):
            raise AssertionError("no non-IMU arrival was supplied")

        def emit(self, *_args, **_kwargs):
            pass

    adapter = TimedAdapter()
    recorder = Recorder()
    runner = VQ2Runner(adapter, _FakeVision(), recorder=recorder)
    runner.estimator = ImuAttitudeEstimator(
        ImuAttitudeConfig(
            calibration_min_samples=1,
            calibration_min_duration_s=0.0,
            gravity_correction_kp=0.0,
            gyro_bias_ki=0.0,
        )
    )

    runner._sample()

    assert len(recorder.received) == 1
    assert recorder.received[0][0] == received.imu
    assert recorder.received[0][3] is received


def test_sampling_records_mixed_receiver_ingress_in_global_sequence_order():
    def ingress(sequence, message_type, source_value=None, source_unit=None):
        return MavlinkIngressV1(
            stream_id="vq2-mavlink-udp-14550",
            generation=1,
            sequence=sequence,
            message_type=message_type,
            host_clock_id="host-perf-counter",
            received_monotonic_ns=100 + sequence,
            source_time_value=source_value,
            source_time_unit=source_unit,
        )

    imu_one = IMUData(
        timestamp_us=1_000_000,
        accel=(0.0, 0.0, -9.80665),
        gyro=(0.0, 0.0, 0.0),
    )
    imu_two = IMUData(
        timestamp_us=1_010_000,
        accel=(0.0, 0.0, -9.80665),
        gyro=(0.0, 0.0, 0.0),
    )
    received = [
        ReceivedIMUSampleV1(
            ingress=ingress(1, "HIGHRES_IMU", imu_one.timestamp_us, "us"),
            imu=imu_one,
        ),
        ReceivedIMUSampleV1(
            ingress=ingress(3, "HIGHRES_IMU", imu_two.timestamp_us, "us"),
            imu=imu_two,
        ),
    ]
    other = [
        ingress(0, "HEARTBEAT"),
        ingress(2, "RACE_STATUS", 10, "ms"),
    ]

    class TimedAdapter(_FakeAdapter):
        def drain_received_imu_samples(self):
            values = list(received)
            received.clear()
            return values

        def drain_mavlink_arrivals(self):
            values = list(other)
            other.clear()
            return values

    class Recorder:
        capture_enabled = False

        def __init__(self):
            self.order = []

        def record_imu(self, _sample, _estimator, _now_s, *, received_sample):
            self.order.append(("imu", received_sample.ingress.sequence))

        def record_mavlink_ingress(self, arrival):
            self.order.append(("other", arrival.sequence))

        def emit(self, *_args, **_kwargs):
            pass

    recorder = Recorder()
    runner = VQ2Runner(TimedAdapter(), _FakeVision(), recorder=recorder)
    runner.estimator = ImuAttitudeEstimator(
        ImuAttitudeConfig(
            calibration_min_samples=1,
            calibration_min_duration_s=0.0,
            gravity_correction_kp=0.0,
            gyro_bias_ki=0.0,
        )
    )

    runner._sample()

    assert recorder.order == [
        ("other", 0),
        ("imu", 1),
        ("other", 2),
        ("imu", 3),
    ]


def _configured_yaw_sign_id(
    monkeypatch,
    *,
    image_command_sign=1.0,
    yaw_response_gain=2.0,
    legacy_clock_offset_s=0.0,
):
    clock = [0.0]
    center_x = [322.0]
    yaw = [0.0]
    frame_id = [10]
    imu_timestamp_us = [10_000]
    next_camera_frame_s = [0.0]
    events = []
    send_options = []
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(1_000, 0, -1, 0, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    start_orientation = Quaternion.from_euler(0.0, -0.31, 0.0)
    runner.estimate = replace(
        _estimate(roll=0.0, pitch=-0.31),
        orientation=start_orientation,
    )
    target = vq2_module.GateTarget(
        frame_id=frame_id[0],
        sim_time_ns=frame_id[0] * 1_000,
        received_monotonic_s=clock[0] + legacy_clock_offset_s,
        center_x=round(center_x[0]),
        center_y=174,
        bbox=(282, 134, 80, 80),
        confidence=0.8,
    )
    runner.tracker.target = target
    runner.tracker.consecutive = 3
    runner.tracker.last_selection_mode = "primary"
    runner._latest_detection_generation = 1
    runner._latest_detection_frame_id = target.frame_id
    runner._latest_detection_frame_sim_ns = target.sim_time_ns
    runner._latest_detection_received_s = target.received_monotonic_s

    def sample():
        previous = (
            adapter.commands[-1]
            if adapter.commands
            else AttitudeRateCommand(0.0, 0.0, 0.0, vq2_module.SIGN_ID_THRUST)
        )
        body_rates = (
            1.1 * float(previous.roll_rate),
            2.0 * float(previous.pitch_rate),
            float(yaw_response_gain) * float(previous.yaw_rate),
        )
        yaw[0] += body_rates[2] * vq2_module.CONTROL_PERIOD_S
        center_x[0] += (
            float(image_command_sign)
            * float(previous.yaw_rate)
            * 400.0
            * vq2_module.CONTROL_PERIOD_S
        )
        imu_timestamp_us[0] += round(
            vq2_module.CONTROL_PERIOD_S * 1_000_000
        )
        half_yaw = yaw[0] / 2.0
        cosine = math.cos(half_yaw)
        sine = math.sin(half_yaw)
        body_z_orientation = Quaternion(
            w=(
                start_orientation.w * cosine
                - start_orientation.z * sine
            ),
            x=(
                start_orientation.x * cosine
                + start_orientation.y * sine
            ),
            y=(
                start_orientation.y * cosine
                - start_orientation.x * sine
            ),
            z=(
                start_orientation.w * sine
                + start_orientation.z * cosine
            ),
        )
        runner.estimate = replace(
            _estimate(roll=0.0, pitch=-0.31),
            timestamp_us=imu_timestamp_us[0],
            orientation=body_z_orientation,
            body_rates=body_rates,
        )
        runner._last_imu_received_monotonic_ns = round(
            clock[0] * 1_000_000_000
        )
        runner._last_imu_receipt_exact = True
        if clock[0] + 1e-9 < next_camera_frame_s[0]:
            return
        while next_camera_frame_s[0] <= clock[0] + 1e-9:
            next_camera_frame_s[0] += 1.0 / 32.0
        frame_id[0] += 1
        current_x = round(center_x[0])
        current = vq2_module.GateTarget(
            frame_id=frame_id[0],
            sim_time_ns=frame_id[0] * 1_000,
            received_monotonic_s=(
                clock[0] + legacy_clock_offset_s
            ),
            center_x=current_x,
            center_y=174,
            bbox=(current_x - 40, 134, 80, 80),
            confidence=0.8,
        )
        runner.tracker.target = current
        runner.tracker.consecutive += 1
        runner.tracker.last_selection_mode = "primary"
        runner._latest_detection_frame_id = current.frame_id
        runner._latest_detection_frame_sim_ns = current.sim_time_ns
        runner._latest_detection_received_s = current.received_monotonic_s
        exact_frame_ns = round(clock[0] * 1_000_000_000)
        runner._latest_detection_timing = FrameTimingV1(
            identity=FrameIdentityV1(
                stream_id="vq2-camera",
                generation=1,
                frame_id=current.frame_id,
            ),
            camera_source_time_ns=current.sim_time_ns,
            host_clock_id="host-perf-counter",
            publication_sequence=current.frame_id,
            first_unique_packet_monotonic_ns=exact_frame_ns,
            final_unique_packet_monotonic_ns=exact_frame_ns,
            reassembly_complete_monotonic_ns=exact_frame_ns,
            decode_start_monotonic_ns=exact_frame_ns,
            decode_end_monotonic_ns=exact_frame_ns,
            publish_monotonic_ns=exact_frame_ns,
        )

    async def send(command, **kwargs):
        send_options.append(dict(kwargs))
        adapter.commands.append(command)
        return {
            "schema": "aigp-vq2-attitude-target-outbound/1",
            "host_clock_id": "host-perf-counter",
            "call_start_monotonic_ns": round(clock[0] * 1_000_000_000),
            "call_end_monotonic_ns": round(clock[0] * 1_000_000_000),
            "api": "send_attitude_rate",
            "outcome": "returned",
            "wire": {
                "type_mask": 128,
                "body_rates_rad_s": [
                    -command.roll_rate,
                    -command.pitch_rate,
                    -command.yaw_rate,
                ],
                "thrust": command.thrust,
            },
        }

    async def advance(seconds):
        clock[0] += max(0.0, float(seconds))

    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        vq2_module.time,
        "perf_counter_ns",
        lambda: round(clock[0] * 1_000_000_000),
    )
    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)
    monkeypatch.setattr(
        runner,
        "_wait_for_calibration_release",
        lambda deadline_ns: clock.__setitem__(
            0,
            max(clock[0], deadline_ns / 1_000_000_000.0),
        ),
    )
    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_send_flight_command", send)
    monkeypatch.setattr(
        runner.recorder,
        "emit",
        lambda event, **payload: events.append((event, payload)),
    )
    return runner, adapter, events, send_options


@pytest.mark.parametrize(
    ("image_command_sign", "expected_image_sign"),
    ((1.0, 1), (-1.0, -1)),
)
def test_sign_id_yaw_calibration_is_paired_isolated_and_measured(
    monkeypatch,
    image_command_sign,
    expected_image_sign,
):
    runner, adapter, events, send_options = _configured_yaw_sign_id(
        monkeypatch,
        image_command_sign=image_command_sign,
        yaw_response_gain=2.0,
    )

    details = asyncio.run(runner._run_sign_id())
    yaw = details["yaw_calibration"]
    from scripts import aigp_vq2_yaw_calibration as yaw_contract

    assert vq2_module.SIGN_ID_RATE_RAD_S == 0.08
    assert vq2_module.SIGN_ID_YAW_PULSE_DURATION_S == 0.22
    assert vq2_module.SIGN_ID_YAW_NEUTRAL_DURATION_S == 0.24
    assert vq2_module.SIGN_ID_YAW_REVERSAL_DURATION_S == 0.12
    assert vq2_module.SIGN_ID_YAW_TERMINAL_DURATION_S == 0.10
    assert vq2_module.SIGN_ID_HARD_EXPIRY_S == 1.00
    assert vq2_module.SIGN_ID_MIN_YAW_GYRO_SAMPLES == 4
    assert vq2_module.SIGN_ID_MIN_FRESH_IMAGE_FRAMES == 4
    assert vq2_module.SIGN_ID_MIN_IMAGE_EFFECT_PX_S == 15.0
    assert vq2_module.SIGN_ID_MAX_POLARITY_GAIN_RATIO == 2.0
    assert vq2_module.SIGN_ID_MAX_ATTITUDE_EXCURSION_RAD == 0.05
    assert vq2_module.SIGN_ID_MAX_MEASURED_YAW_RATE_RAD_S == 0.50
    assert vq2_module.SIGN_ID_MAX_GYRO_RESPONSE_DELAY_S == 0.08
    assert (
        vq2_module.SIGN_ID_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S
        == 0.09
    )
    assert vq2_module.SIGN_ID_CONTROL_HOLD_HORIZON_S == 0.12
    assert all(
        option.get("require_wire_receipt") is True
        and type(option.get("wire_start_not_before_ns")) is int
        and type(option.get("wire_start_deadline_ns")) is int
        and option["wire_start_not_before_ns"]
        < option["wire_start_deadline_ns"]
        for option in send_options
    )
    assert (
        len(adapter.commands)
        == details["ticks_sent"]
        == details["ticks_expected"]
        == yaw_contract.YAW_CALIBRATION_TICK_COUNT
    )
    assert details["plan_id"] == yaw_contract.YAW_CALIBRATION_PLAN_ID
    assert (
        details["plan_sha256"]
        == yaw_contract.YAW_CALIBRATION_PLAN_SHA256
    )
    positive = [
        command for command in adapter.commands
        if command.yaw_rate == vq2_module.SIGN_ID_RATE_RAD_S
    ]
    negative = [
        command for command in adapter.commands
        if command.yaw_rate == -vq2_module.SIGN_ID_RATE_RAD_S
    ]
    assert positive and negative
    assert all(
        command.roll_rate == command.pitch_rate == 0.0
        and command.thrust == 0.235
        for command in adapter.commands
    )
    assert yaw["yaw_identified"] is True
    assert yaw["controller_to_body_sign"] == 1
    assert yaw["controller_to_image_sign"] == expected_image_sign
    assert yaw["gyro_rate_gain"] == pytest.approx(2.0)
    assert (
        yaw["max_gyro_response_delay_upper_bound_s"]
        <= vq2_module.SIGN_ID_MAX_GYRO_RESPONSE_DELAY_S
    )
    assert (
        yaw["max_first_image_observation_delay_s"]
        <= vq2_module.SIGN_ID_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S
    )
    assert (
        yaw["control_hold_horizon_s"]
        == vq2_module.SIGN_ID_CONTROL_HOLD_HORIZON_S
    )
    assert math.copysign(
        1.0,
        yaw["image_rate_gain_px_per_command_rad"],
    ) == expected_image_sign
    assert yaw["positive"]["wire_yaw_rate_rad_s"] == -0.08
    assert yaw["negative"]["wire_yaw_rate_rad_s"] == 0.08
    assert (
        yaw["positive"]["fresh_image_frame_count"]
        >= vq2_module.SIGN_ID_MIN_FRESH_IMAGE_FRAMES
    )
    assert (
        yaw["negative"]["fresh_image_frame_count"]
        >= vq2_module.SIGN_ID_MIN_FRESH_IMAGE_FRAMES
    )
    terminals = [
        payload
        for event, payload in events
        if event == "sign_id_yaw_terminal"
    ]
    assert terminals[-1]["success"] is True
    ticks = [payload for event, payload in events if event == "tick"]
    assert list(dict.fromkeys(payload["stage"] for payload in ticks)) == [
        "sign-id/neutral-pre-yaw",
        "sign-id/yaw-positive",
        "sign-id/neutral-reversal",
        "sign-id/yaw-negative",
        "sign-id/neutral-terminal",
    ]
    assert ticks[-1]["elapsed_s"] < vq2_module.SIGN_ID_HARD_EXPIRY_S


def test_sign_id_yaw_rejects_stationary_image_response(monkeypatch):
    runner, _adapter, events, _send_options = _configured_yaw_sign_id(
        monkeypatch,
        image_command_sign=0.0,
        yaw_response_gain=2.0,
    )

    with pytest.raises(SafetyAbort, match="yaw image response"):
        asyncio.run(runner._run_sign_id())

    terminals = [
        payload
        for event, payload in events
        if event == "sign_id_yaw_terminal"
    ]
    assert terminals[-1]["success"] is False


def test_sign_id_yaw_delay_uses_exact_frame_clock_not_legacy_seconds(
    monkeypatch,
):
    runner, _adapter, _events, _send_options = _configured_yaw_sign_id(
        monkeypatch,
        legacy_clock_offset_s=17.0,
    )

    details = asyncio.run(runner._run_sign_id())

    yaw = details["yaw_calibration"]
    assert yaw["max_first_image_observation_delay_s"] >= 0.04
    assert (
        yaw["max_first_image_observation_delay_s"]
        <= vq2_module.SIGN_ID_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S
    )


def test_sign_id_yaw_reuses_attitude_excursion_guard(monkeypatch):
    runner, _adapter, _events, _send_options = _configured_yaw_sign_id(
        monkeypatch,
        image_command_sign=1.0,
        yaw_response_gain=4.5,
    )

    with pytest.raises(SafetyAbort, match="attitude excursion"):
        asyncio.run(runner._run_sign_id())


def test_sign_id_yaw_requires_exact_imu_receipt_provenance(monkeypatch):
    runner, _adapter, _events, _send_options = _configured_yaw_sign_id(
        monkeypatch,
    )
    sample = runner._sample

    def sample_without_exact_receipt():
        sample()
        runner._last_imu_receipt_exact = False

    monkeypatch.setattr(runner, "_sample", sample_without_exact_receipt)

    with pytest.raises(SafetyAbort, match="exact HIGHRES_IMU receipt"):
        asyncio.run(runner._run_sign_id())


def test_sign_id_yaw_requires_exact_frame_timing(monkeypatch):
    runner, _adapter, _events, _send_options = _configured_yaw_sign_id(
        monkeypatch,
    )
    sample = runner._sample

    def sample_without_exact_frame_timing():
        sample()
        runner._latest_detection_timing = None

    monkeypatch.setattr(
        runner,
        "_sample",
        sample_without_exact_frame_timing,
    )

    with pytest.raises(SafetyAbort, match="exact host timing"):
        asyncio.run(runner._run_sign_id())


def test_sign_id_yaw_rejects_response_outside_delay_envelope(monkeypatch):
    runner, _adapter, _events, _send_options = _configured_yaw_sign_id(
        monkeypatch,
    )
    monkeypatch.setattr(
        vq2_module,
        "SIGN_ID_MAX_GYRO_RESPONSE_DELAY_S",
        0.03,
    )

    with pytest.raises(SafetyAbort, match="gyro response delay"):
        asyncio.run(runner._run_sign_id())


def test_sign_id_yaw_rejects_late_first_image_observation(monkeypatch):
    runner, _adapter, _events, _send_options = _configured_yaw_sign_id(
        monkeypatch,
    )
    monkeypatch.setattr(
        vq2_module,
        "SIGN_ID_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S",
        0.03,
    )

    with pytest.raises(SafetyAbort, match="first image observation"):
        asyncio.run(runner._run_sign_id())


def test_sign_id_yaw_terminal_hold_runs_watchdog(monkeypatch):
    runner, _adapter, events, _send_options = _configured_yaw_sign_id(
        monkeypatch,
    )
    calls = [0]
    from scripts import aigp_vq2_yaw_calibration as yaw_contract

    def watchdog(**_kwargs):
        calls[0] += 1
        if calls[0] == yaw_contract.YAW_CALIBRATION_TICK_COUNT + 1:
            raise SafetyAbort("terminal calibration collision")

    monkeypatch.setattr(runner, "_watchdog", watchdog)

    with pytest.raises(SafetyAbort, match="terminal calibration collision"):
        asyncio.run(runner._run_sign_id())

    assert calls[0] == yaw_contract.YAW_CALIBRATION_TICK_COUNT + 1
    assert not any(
        event == "yaw_calibration_plan_complete"
        for event, _payload in events
    )
    assert [
        payload
        for event, payload in events
        if event == "sign_id_yaw_terminal"
    ][-1]["success"] is False


def test_sign_id_powered_dispatch_excites_only_yaw(monkeypatch):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(1_000, 0, -1, 0, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(
        0.0,
        -0.31,
        322,
        174,
        6_400,
        1_000,
    )
    calls = []

    async def establish(**_kwargs):
        calls.append("reset")

    async def normalize():
        calls.append("normalize")

    async def wait_for_go():
        calls.append("go")
        return context

    async def arm():
        calls.append("arm")

    async def hover(_observed_context):
        raise AssertionError("yaw-only sign-ID must not enter hover")

    async def sign_id(*, calibration_context):
        assert calibration_context is context
        calls.append("yaw")
        return {"yaw_calibration": {"yaw_identified": True}}

    async def cleanup():
        calls.append("cleanup")
        return True

    monkeypatch.setattr(runner, "establish_reset_epoch", establish)
    monkeypatch.setattr(runner, "normalize_disarmed", normalize)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(runner, "arm_confirmed", arm)
    monkeypatch.setattr(runner, "_run_hover", hover)
    monkeypatch.setattr(runner, "_run_sign_id", sign_id)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(runner.run_powered_stage("sign-id"))

    assert result.success
    assert result.cleanup_confirmed
    assert result.details["yaw_calibration"] == {
        "yaw_identified": True
    }
    assert result.details["cleanup_collision_safety"]["safe"] is True
    assert calls == [
        "reset",
        "normalize",
        "go",
        "arm",
        "yaw",
        "cleanup",
    ]


def test_yaw_capability_entry_is_short_braked_and_bounded(monkeypatch):
    runner, _adapter, context = _fast_calibration_runner()
    clock = [0.0]
    commands = []

    async def wait_slot():
        return clock[0]

    async def sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    async def send(command, **kwargs):
        assert kwargs["require_wire_receipt"] is True
        commands.append(command)
        return {
            "wire": {
                "type_mask": 128,
                "body_rates_rad_s": [
                    -command.roll_rate,
                    -command.pitch_rate,
                    -command.yaw_rate,
                ],
                "thrust": command.thrust,
            }
        }

    monkeypatch.setattr(
        runner,
        "_wait_for_next_flight_command_slot",
        wait_slot,
    )
    monkeypatch.setattr(runner, "_sample", lambda: None)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_send_flight_command", send)
    monkeypatch.setattr(runner, "_record_tick", lambda *_args: None)
    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vq2_module.asyncio, "sleep", sleep)

    details = asyncio.run(runner._run_yaw_capability_entry(context))

    assert 39 <= len(commands) <= 41
    assert all(command.yaw_rate == 0.0 for command in commands)
    assert all(
        abs(command.roll_rate) <= vq2_module.MAX_COMMAND_RATE_RAD_S
        and abs(command.pitch_rate) <= vq2_module.MAX_COMMAND_RATE_RAD_S
        for command in commands
    )
    assert commands[0].thrust == 0.26
    assert any(command.thrust == 0.32 for command in commands)
    assert commands[-1].thrust == 0.285
    assert details["duration_s"] == 0.80
    assert details["target_pitch_rad"] == (
        vq2_module.MAX_VISUAL_TARGET_PITCH_RAD
    )
    assert details["command_count"] == len(commands)


def test_calibration_excite_runs_exact_progressive_free_flight_yaw_plan(
    monkeypatch,
):
    from scripts import aigp_vq2_yaw_capability as capability

    runner, adapter, context = _fast_calibration_runner()
    clock_ns = [1_000_000_000]
    commands = []

    async def entry(value):
        assert value == context
        return {"free_flight": True}

    async def wait_slot():
        return clock_ns[0] / 1_000_000_000.0

    def wait_release(deadline_ns):
        clock_ns[0] = max(clock_ns[0], int(deadline_ns))

    def sample():
        runner._last_imu_received_monotonic_ns = clock_ns[0]

    async def send(command, **kwargs):
        clock_ns[0] = max(
            clock_ns[0],
            int(kwargs["wire_start_not_before_ns"]),
        )
        commands.append(command)
        return {
            "call_start_monotonic_ns": clock_ns[0],
            "call_end_monotonic_ns": clock_ns[0],
            "wire": {
                "type_mask": 128,
                "body_rates_rad_s": [
                    -command.roll_rate,
                    -command.pitch_rate,
                    -command.yaw_rate,
                ],
                "thrust": command.thrust,
            },
        }

    monkeypatch.setattr(runner, "_run_yaw_capability_entry", entry)
    monkeypatch.setattr(
        runner,
        "_wait_for_next_flight_command_slot",
        wait_slot,
    )
    monkeypatch.setattr(runner, "_wait_for_calibration_release", wait_release)
    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_send_flight_command", send)
    monkeypatch.setattr(runner, "_record_tick", lambda *_args: None)
    monkeypatch.setattr(
        vq2_module.time,
        "perf_counter_ns",
        lambda: clock_ns[0],
    )

    details = asyncio.run(runner._run_calibration_excite(context))

    assert len(commands) == capability.YAW_CAPABILITY_TICK_COUNT
    assert [command.yaw_rate for command in commands if command.yaw_rate] == [
        value
        for level in capability.YAW_CAPABILITY_LEVELS_RAD_S
        for value in [level] * 11 + [-level] * 11
    ]
    assert details["stage"] == "calibration-excite"
    assert details["calibration_scope"] == (
        "free-flight-progressive-yaw-capability-build3385"
    )
    assert details["ticks_sent"] == details["ticks_expected"] == (
        capability.YAW_CAPABILITY_TICK_COUNT
    )
    assert details["authority_effect"] == (
        "characterization-only-no-visual-course-envelope-change"
    )
    assert details["free_flight_entry"] == {"free_flight": True}
    assert adapter.race_status.active_gate_index == 0


def test_delayed_pre_reset_clocks_cannot_unlock_go():
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )
    runner._accept_reset_proof(proof, restart_vision=False)
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1_000_000,
        race_start_boot_time_ms=100,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    adapter.latest_telemetry = TelemetryState(
        timestamp_us=0,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion(),
        angular_velocity=(0.0, 0.0, 0.0),
        imu=IMUData(
            timestamp_us=1_000_000_000,
            accel=(0.0, 0.0, -9.81),
            gyro=(0.0, 0.0, 0.0),
        ),
    )

    with pytest.raises(SafetyAbort, match="proved reset epoch"):
        asyncio.run(runner.wait_for_go(timeout_s=0.05))

    assert adapter.arm_calls == 0


def test_unproved_reset_path_never_calls_arm(monkeypatch):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())

    async def reset_fails(**_kwargs):
        raise SafetyAbort("ignored reset")

    async def cleanup_succeeds():
        return True

    monkeypatch.setattr(runner, "establish_reset_epoch", reset_fails)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup_succeeds)

    result = asyncio.run(runner.run_powered_stage("sign-id"))

    assert not result.success
    assert "ignored reset" in result.reason
    assert adapter.arm_calls == 0


def _fast_calibration_runner():
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1_000,
        race_start_boot_time_ms=0,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    runner = VQ2Runner(adapter, _FakeVision())
    runner.estimate = _estimate(roll=0.0, pitch=-0.31)
    runner.tracker.target = vq2_module.GateTarget(
        frame_id=10,
        sim_time_ns=10,
        received_monotonic_s=0.0,
        center_x=322,
        center_y=174,
        bbox=(282, 134, 80, 80),
        confidence=0.8,
    )
    runner.tracker.consecutive = 3
    runner._latest_detection_image = SimpleNamespace(shape=(360, 640, 3))
    context = vq2_module.StartContext(
        spawn_roll_rad=0.0,
        spawn_pitch_rad=-0.31,
        initial_gate_x=322,
        initial_gate_y=174,
        initial_gate_area=6_400,
        go_boot_ms=1_000,
    )
    return runner, adapter, context


def _append_fake_attitude_receipt(adapter, call_start_ns, call_end_ns):
    adapter.outbound_receipts.append(
        {
            "schema": "aigp-vq2-attitude-target-outbound/1",
            "host_clock_id": "host-perf-counter",
            "call_start_monotonic_ns": call_start_ns,
            "call_end_monotonic_ns": call_end_ns,
            "api": "send_attitude_rate",
            "outcome": "returned",
        }
    )


def test_fast_calibration_waveform_is_exact_50hz_zero_yaw_and_complete(
    monkeypatch,
):
    runner, adapter, context = _fast_calibration_runner()
    clock = [0.0]
    send_times = []
    watchdog_calls = []

    async def advance(seconds):
        clock[0] += max(0.0, float(seconds))

    def sample():
        clock[0] += 0.001

    async def send(
        command,
        *,
        call_start_not_before_monotonic_ns,
        call_start_deadline_monotonic_ns,
    ):
        call_start_ns = int(round(clock[0] * 1_000_000_000))
        assert call_start_ns >= call_start_not_before_monotonic_ns
        assert call_start_ns < call_start_deadline_monotonic_ns
        adapter.commands.append(command)
        send_times.append(clock[0])
        # Transport completion latency must not be added to every 20 ms slot.
        clock[0] += 0.001
        _append_fake_attitude_receipt(
            adapter,
            call_start_ns,
            int(round(clock[0] * 1_000_000_000)),
        )

    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        vq2_module.time,
        "perf_counter_ns",
        lambda: int(round(clock[0] * 1_000_000_000)),
    )
    monkeypatch.setattr(
        runner,
        "_wait_for_calibration_release",
        lambda deadline_ns: clock.__setitem__(
            0, max(clock[0], deadline_ns / 1_000_000_000.0)
        ),
    )
    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)
    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(
        runner, "_watchdog", lambda **kwargs: watchdog_calls.append(kwargs)
    )
    monkeypatch.setattr(adapter, "send_attitude_rate", send)

    details = asyncio.run(
        runner._run_legacy_roll_pitch_calibration_waveform(context)
    )

    fast_plan = powered_contract.fast_excitation_plan()
    assert (
        details["ticks_sent"]
        == details["ticks_expected"]
        == fast_plan["tick_count"]
    )
    assert details["plan_sha256"] == powered_contract.canonical_object_sha256(
        fast_plan
    )
    assert len(adapter.commands) == fast_plan["tick_count"]
    assert all(command.yaw_rate == 0.0 for command in adapter.commands)
    assert all(abs(command.roll_rate) <= 0.02 for command in adapter.commands)
    assert all(abs(command.pitch_rate) <= 0.02 for command in adapter.commands)
    assert all(command.thrust == 0.235 for command in adapter.commands)
    assert all(
        call == {
            "allow_benign_pad_contact": True,
            "enforce_benign_pad_budget": False,
            "benign_pad_max_impulse": 0.02,
        }
        for call in watchdog_calls
    )
    assert all(
        later - earlier == pytest.approx(vq2_module.CONTROL_PERIOD_S)
        for earlier, later in zip(send_times, send_times[1:])
    )
    assert send_times[-1] - send_times[0] == pytest.approx(
        (fast_plan["tick_count"] - 1) * vq2_module.CONTROL_PERIOD_S
    )
    for index, command in enumerate(adapter.commands):
        expected = powered_contract.fast_excitation_tick(index)["command"]
        assert command.roll_rate == expected["roll_rate_rad_s"]
        assert command.pitch_rate == expected["pitch_rate_rad_s"]
        assert command.yaw_rate == expected["yaw_rate_rad_s"]
        assert command.thrust == expected["thrust"]


def test_fast_calibration_variable_safety_cost_does_not_creep_phase(monkeypatch):
    runner, adapter, context = _fast_calibration_runner()
    clock = [0.0]
    sample_count = [0]
    send_times = []

    async def advance(seconds):
        clock[0] += max(0.0, float(seconds))

    def sample():
        if sample_count[0] == 0:
            clock[0] += 0.005
        sample_count[0] += 1

    def watchdog(**_kwargs):
        clock[0] += 0.0001

    async def send(
        command,
        *,
        call_start_not_before_monotonic_ns,
        call_start_deadline_monotonic_ns,
    ):
        call_start_ns = int(round(clock[0] * 1_000_000_000))
        assert call_start_ns >= call_start_not_before_monotonic_ns
        assert call_start_ns < call_start_deadline_monotonic_ns
        send_times.append(clock[0])
        adapter.commands.append(command)
        clock[0] += 0.001
        _append_fake_attitude_receipt(
            adapter,
            call_start_ns,
            int(round(clock[0] * 1_000_000_000)),
        )

    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        vq2_module.time,
        "perf_counter_ns",
        lambda: int(round(clock[0] * 1_000_000_000)),
    )
    monkeypatch.setattr(
        runner,
        "_wait_for_calibration_release",
        lambda deadline_ns: clock.__setitem__(
            0, max(clock[0], deadline_ns / 1_000_000_000.0)
        ),
    )
    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)
    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_watchdog", watchdog)
    monkeypatch.setattr(adapter, "send_attitude_rate", send)

    details = asyncio.run(
        runner._run_legacy_roll_pitch_calibration_waveform(context)
    )

    fast_plan = powered_contract.fast_excitation_plan()
    assert details["ticks_sent"] == fast_plan["tick_count"]
    assert len(send_times) == fast_plan["tick_count"]
    assert all(
        later - earlier >= vq2_module.CONTROL_PERIOD_S - 1e-9
        for earlier, later in zip(send_times, send_times[1:])
    )
    assert send_times[-1] < 4.9


def test_fast_calibration_transport_deadline_blocks_late_wire_send(monkeypatch):
    runner, adapter, context = _fast_calibration_runner()
    clock = [0.0]

    async def advance(seconds):
        clock[0] += max(0.0, float(seconds))

    async def delayed_send(
        command,
        *,
        call_start_not_before_monotonic_ns,
        call_start_deadline_monotonic_ns,
    ):
        del command, call_start_not_before_monotonic_ns
        # Model waiting on the adapter's private send lock. The concrete
        # adapter checks the deadline only after that lock is acquired.
        clock[0] += 0.020
        call_start_ns = int(round(clock[0] * 1_000_000_000))
        if call_start_ns >= call_start_deadline_monotonic_ns:
            raise TimeoutError("attitude-target call-start deadline was reached")
        raise AssertionError("late command must not reach the wire")

    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        vq2_module.time,
        "perf_counter_ns",
        lambda: int(round(clock[0] * 1_000_000_000)),
    )
    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)
    monkeypatch.setattr(runner, "_sample", lambda: None)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(adapter, "send_attitude_rate", delayed_send)

    with pytest.raises(TimeoutError, match="call-start deadline"):
        asyncio.run(
            runner._run_legacy_roll_pitch_calibration_waveform(
                context
            )
        )

    assert adapter.commands == []


def test_fast_calibration_missed_slot_aborts_without_catchup_send(monkeypatch):
    runner, adapter, context = _fast_calibration_runner()
    clock = [0.0]

    def stalled_sample():
        clock[0] += 0.021

    async def advance(seconds):
        clock[0] += max(0.0, float(seconds))

    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        vq2_module.time,
        "perf_counter_ns",
        lambda: int(round(clock[0] * 1_000_000_000)),
    )
    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)
    monkeypatch.setattr(runner, "_sample", stalled_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)

    with pytest.raises(SafetyAbort, match="expired before send"):
        asyncio.run(
            runner._run_legacy_roll_pitch_calibration_waveform(
                context
            )
        )

    assert adapter.commands == []


def test_fast_calibration_rejects_changed_decoded_dimensions():
    runner, _adapter, context = _fast_calibration_runner()
    runner._latest_detection_image = SimpleNamespace(shape=(180, 320, 3))
    reference = runner.estimate.orientation

    with pytest.raises(SafetyAbort, match="dimensions changed"):
        runner._check_calibration_envelope(
            context,
            reference_orientation=reference,
        )


def test_fast_calibration_excursion_matches_the_complete_waveform_envelope():
    runner, _adapter, context = _fast_calibration_runner()
    reference = runner.estimate.orientation
    runner.estimate = _estimate(roll=0.024, pitch=-0.31)
    roll_excursion, _pitch_excursion, _area = (
        runner._check_calibration_envelope(
            context,
            reference_orientation=reference,
        )
    )
    assert roll_excursion == pytest.approx(0.024)

    runner.estimate = _estimate(roll=0.026, pitch=-0.31)
    with pytest.raises(
        SafetyAbort,
        match="transverse body-axis excursion exceeded 0.025 rad",
    ):
        runner._check_calibration_envelope(
            context,
            reference_orientation=reference,
        )


def test_fast_calibration_allows_euler_roll_from_pure_body_z_yaw():
    runner, _adapter, context = _fast_calibration_runner()
    reference = runner.estimate.orientation
    half_yaw = 0.12
    cosine = math.cos(half_yaw)
    sine = math.sin(half_yaw)
    current = Quaternion(
        w=reference.w * cosine - reference.z * sine,
        x=reference.x * cosine + reference.y * sine,
        y=reference.y * cosine - reference.x * sine,
        z=reference.w * sine + reference.z * cosine,
    )
    runner.estimate = replace(runner.estimate, orientation=current)

    roll_excursion, pitch_excursion, _area = (
        runner._check_calibration_envelope(
            context,
            reference_orientation=reference,
        )
    )

    assert roll_excursion > vq2_module.CALIBRATION_MAX_ATTITUDE_EXCURSION_RAD
    assert pitch_excursion < vq2_module.CALIBRATION_MAX_ATTITUDE_EXCURSION_RAD
    transverse, axial = vq2_module.calibration_body_axis_excursions_rad(
        reference,
        current,
    )
    assert transverse == pytest.approx(0.0, abs=1e-12)
    assert axial == pytest.approx(0.24)


def test_calibration_body_axis_corridor_cannot_stack_at_waveform_entry():
    runner, _adapter, context = _fast_calibration_runner()
    entry = Quaternion.from_euler(0.024, -0.31, 0.0)
    half_roll = 0.012
    cosine = math.cos(half_roll)
    sine = math.sin(half_roll)
    stacked = Quaternion(
        w=entry.w * cosine - entry.x * sine,
        x=entry.w * sine + entry.x * cosine,
        y=entry.y * cosine + entry.z * sine,
        z=-entry.y * sine + entry.z * cosine,
    )
    runner.estimate = replace(runner.estimate, orientation=entry)
    runner._check_calibration_envelope(context)
    reference = Quaternion.from_euler(
        context.spawn_roll_rad,
        context.spawn_pitch_rad,
        context.spawn_yaw_rad,
    )
    runner.estimate = replace(runner.estimate, orientation=stacked)

    with pytest.raises(
        SafetyAbort,
        match="transverse body-axis excursion exceeded 0.025 rad",
    ):
        runner._check_calibration_envelope(
            context,
            reference_orientation=reference,
        )


def test_calibration_body_axis_replay_separates_failed_euler_endpoint():
    reference = Quaternion.from_euler(
        -0.0002940814507083663,
        -0.31003194210690266,
        -0.000040163373932831003,
    )
    failed_endpoint = Quaternion.from_euler(
        -0.02465073641901364,
        -0.3063967512776172,
        0.08040249943217197,
    )

    transverse, axial = vq2_module.calibration_body_axis_excursions_rad(
        reference,
        failed_endpoint,
    )

    assert transverse == pytest.approx(0.0026768477920678007)
    assert axial == pytest.approx(0.07668948152213152)
    assert transverse < vq2_module.CALIBRATION_MAX_ATTITUDE_EXCURSION_RAD


def test_powered_readiness_requires_20fps_and_exact_dimensions():
    runner, _adapter, _context = _fast_calibration_runner()
    runner._epoch_vision_started_s = 0.0
    runner._epoch_vision_initial_frames = 0
    runner.vision.frames_decoded = 31

    failures, facts = runner._powered_vision_readiness(1.0)

    assert failures == []
    assert facts["fps"] == pytest.approx(31.0)
    assert facts["dimensions_px"] == [640, 360]

    runner.vision.frames_decoded = 19
    runner._latest_detection_image = SimpleNamespace(shape=(180, 320, 3))
    failures, _facts = runner._powered_vision_readiness(1.0)
    assert any("below 20fps" in failure for failure in failures)
    assert any("640x360" in failure for failure in failures)


@pytest.mark.parametrize("cleanup_confirmed", (True, False))
def test_fast_calibration_stage_requires_confirmed_cleanup_for_run_success(
    monkeypatch,
    cleanup_confirmed,
):
    runner, adapter, context = _fast_calibration_runner()
    calls = []

    async def reset_epoch(*, restart_vision):
        calls.append(("reset_epoch", restart_vision))

    async def normalize_disarmed():
        calls.append("normalize_disarmed")

    async def wait_for_go():
        calls.append("wait_for_go")
        return context

    async def arm_confirmed():
        calls.append("arm_confirmed")

    async def excite(value):
        from scripts import aigp_vq2_yaw_capability as yaw_contract

        assert value == context
        calls.append("calibration-excite")
        return {"ticks_sent": yaw_contract.YAW_CAPABILITY_TICK_COUNT}

    async def cleanup():
        calls.append("cleanup")
        adapter.is_armed = False
        return cleanup_confirmed

    monkeypatch.setattr(runner, "establish_reset_epoch", reset_epoch)
    monkeypatch.setattr(runner, "normalize_disarmed", normalize_disarmed)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(runner, "arm_confirmed", arm_confirmed)
    monkeypatch.setattr(runner, "_run_calibration_excite", excite)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(runner.run_powered_stage("calibration-excite"))

    assert result.success is cleanup_confirmed
    assert result.cleanup_confirmed is cleanup_confirmed
    assert result.details["ticks_sent"] > 0
    if not cleanup_confirmed:
        assert "cleanup unconfirmed" in result.reason
    assert calls == [
        ("reset_epoch", True),
        "normalize_disarmed",
        "wait_for_go",
        "arm_confirmed",
        "calibration-excite",
        "cleanup",
    ]


def test_only_tiny_spawn_pad_contact_is_classified_benign():
    assert is_benign_pad_contact(
        {"id": 1002, "threat_level": 1, "impulse": 0.0025}
    )
    assert not is_benign_pad_contact(
        {"id": 1001, "threat_level": 1, "impulse": 0.0025}
    )
    assert not is_benign_pad_contact(
        {"id": 1002, "threat_level": 2, "impulse": 0.0025}
    )
    assert not is_benign_pad_contact(
        {"id": 1002, "threat_level": 1, "impulse": 0.02}
    )
    assert is_benign_pad_contact(
        {"id": 1002, "threat_level": 1, "impulse": 0.0159},
        max_impulse=0.02,
    )


def test_repeated_tiny_pad_contacts_exceed_cumulative_launch_budget(monkeypatch):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    monkeypatch.setattr(runner, "_stream_failures", lambda **_kwargs: [])

    adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": 0.004}
        for _ in range(12)
    ]
    runner._watchdog(
        allow_benign_pad_contact=True,
        enforce_benign_pad_budget=True,
    )

    adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": 0.004}
    ]
    with pytest.raises(SafetyAbort, match="repeated pad contacts"):
        runner._watchdog(
            allow_benign_pad_contact=True,
            enforce_benign_pad_budget=True,
        )

    impulse_adapter = _FakeAdapter()
    impulse_runner = VQ2Runner(impulse_adapter, _FakeVision())
    monkeypatch.setattr(impulse_runner, "_stream_failures", lambda **_kwargs: [])
    impulse_adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": 0.009}
        for _ in range(6)
    ]
    with pytest.raises(SafetyAbort, match="repeated pad contacts"):
        impulse_runner._watchdog(
            allow_benign_pad_contact=True,
            enforce_benign_pad_budget=True,
        )

    late_adapter = _FakeAdapter()
    late_runner = VQ2Runner(late_adapter, _FakeVision())
    monkeypatch.setattr(late_runner, "_stream_failures", lambda **_kwargs: [])
    late_adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": 0.001}
    ]
    with pytest.raises(SafetyAbort, match="collision reported"):
        late_runner._watchdog(
            allow_benign_pad_contact=False,
            enforce_benign_pad_budget=True,
        )


@pytest.mark.parametrize(
    (
        "post_cross_gate_index",
        "expected_reason",
        "crossing_hold_thrust",
    ),
    [
        (1, None, 0.0),
        (1, None, 0.275),
        (0, "not credited", 0.0),
    ],
)
def test_gate0_confirmation_uses_configured_hold_then_new_race_packet(
    monkeypatch,
    post_cross_gate_index,
    expected_reason,
    crossing_hold_thrust,
):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1000,
        race_start_boot_time_ms=0,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    vision = _FakeVision()
    runner = VQ2Runner(adapter, vision)
    runner.estimate = _estimate()
    runner.tracker.target = vq2_module.GateTarget(
        frame_id=10,
        sim_time_ns=10,
        received_monotonic_s=0.0,
        center_x=336,
        center_y=180,
        bbox=(61, 0, 551, 360),
        confidence=0.8,
    )
    runner.tracker.consecutive = 3
    vision.current_snapshot = _vision_snapshot(
        frame_id=11,
        sim_time_ns=11,
        received_monotonic_s=0.0,
    )

    clock = [0.0]
    sample_count = [0]
    watchdog_target_requirements = []
    confirmation_commands = []

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(float(seconds), 0.02)

    def fake_sample():
        sample_count[0] += 1
        if sample_count[0] == 7:
            # A fresh contour and a same-timestamp gate-1 value must not leave
            # the latched zero-thrust confirmation phase or prove passage.
            runner.tracker.target = vq2_module.GateTarget(
                frame_id=11,
                sim_time_ns=11,
                received_monotonic_s=clock[0],
                center_x=336,
                center_y=180,
                bbox=(61, 0, 551, 360),
                confidence=0.8,
            )
            adapter.race_status = RaceStatus(
                sim_boot_time_ms=1000,
                race_start_boot_time_ms=0,
                race_finish_time_ns=-1,
                active_gate_index=post_cross_gate_index,
                last_gate_race_time=(123 if post_cross_gate_index == 1 else -1),
            )
        elif sample_count[0] >= 8:
            adapter.race_status = RaceStatus(
                sim_boot_time_ms=1250,
                race_start_boot_time_ms=0,
                race_finish_time_ns=-1,
                active_gate_index=post_cross_gate_index,
                last_gate_race_time=(123 if post_cross_gate_index == 1 else -1),
            )

    def fake_watchdog(**kwargs):
        watchdog_target_requirements.append(kwargs["require_target"])

    def record_tick(stage, _elapsed, command):
        if stage == "gate0/confirm":
            confirmation_commands.append(command)

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", fake_watchdog)
    monkeypatch.setattr(runner, "_record_tick", record_tick)

    context = vq2_module.StartContext(
        spawn_roll_rad=0.0,
        spawn_pitch_rad=-0.31,
        initial_gate_x=322,
        initial_gate_y=174,
        initial_gate_area=6480,
        go_boot_ms=1000,
    )
    if expected_reason is None:
        result = asyncio.run(
            runner._run_gate0(
                context,
                boost_until_s=vq2_module.COURSE_GATE0_BOOST_UNTIL_S,
                crossing_hold_thrust=crossing_hold_thrust,
            )
        )
        assert result["gate0_passed"]
        assert result["crossing_confirmation_used"]
    else:
        with pytest.raises(SafetyAbort, match=expected_reason):
            asyncio.run(
                runner._run_gate0(
                    context,
                    boost_until_s=vq2_module.COURSE_GATE0_BOOST_UNTIL_S,
                    crossing_hold_thrust=crossing_hold_thrust,
                )
            )

    assert confirmation_commands
    assert all(
        command
        == AttitudeRateCommand(
            0.0,
            0.0,
            0.0,
            crossing_hold_thrust,
        )
        for command in confirmation_commands
    )
    assert False in watchdog_target_requirements


def _configure_gate1_observer(*, clock, crossing_started_s=9.70):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1250,
        race_start_boot_time_ms=0,
        race_finish_time_ns=-1,
        active_gate_index=1,
        last_gate_race_time=123,
    )
    vision = _FakeVision()
    vision.current_snapshot = _vision_snapshot(
        frame_id=100,
        sim_time_ns=1_000,
        received_monotonic_s=9.99,
        generation=4,
    )
    runner = VQ2Runner(adapter, vision)
    runner.estimate = _estimate(roll=0.01, pitch=-0.05)
    runner._last_flight_command = vq2_module.AttitudeRateCommand(
        0.0, 0.0, 0.0, 0.0
    )
    runner._last_flight_command_sent_s = 9.98
    runner._gate0_transition_proof = vq2_module.GateTransitionProof(
        pre_gate_race_boot_ms=1000,
        post_gate_race_boot_ms=1250,
        flight_started_monotonic_s=5.20,
        crossing_started_monotonic_s=crossing_started_s,
        pass_confirmed_monotonic_s=9.99,
        next_control_deadline_s=clock[0],
        vision_generation=4,
        vision_frame_id=100,
        vision_sim_time_ns=1_000,
        vision_received_monotonic_s=9.99,
        pass_rpy_rad=runner.estimate.orientation.to_euler(),
    )
    details = {
        "gate0_passed": True,
        "gate_transition_proved": True,
        "pre_gate_race_boot_ms": 1000,
        "race_boot_ms": 1250,
    }
    return runner, adapter, vision, details


def _publish_observation_frame(runner, *, frame_id, clock, detection):
    accepted = runner.tracker.update(
        ([] if detection is None else [detection]),
        frame_id=frame_id,
        sim_time_ns=frame_id * 10,
        received_monotonic_s=clock[0],
    )
    runner._latest_detection_generation = 4
    runner._latest_detection_frame_id = frame_id
    runner._latest_detection_frame_sim_ns = frame_id * 10
    runner._latest_detection_received_s = clock[0]
    runner._latest_accepted_target = accepted


@pytest.mark.parametrize("hold_thrust", (0.0, 0.275))
def test_gate1_observation_requires_three_new_frames_and_paced_zero_rates(
    monkeypatch,
    hold_thrust,
):
    clock = [10.0]
    runner, adapter, vision, details = _configure_gate1_observer(clock=clock)
    sample_count = [0]
    command_times = []
    watchdog_require_target = []

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        sample_count[0] += 1
        frame_id = 100 + sample_count[0]
        _publish_observation_frame(
            runner,
            frame_id=frame_id,
            clock=clock,
            detection=_detection(409 + sample_count[0], 138, 28, 45),
        )

    def fake_watchdog(**kwargs):
        watchdog_require_target.append(kwargs["require_target"])

    async def recorded_send(command):
        adapter.commands.append(command)
        command_times.append(clock[0])

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", fake_watchdog)
    monkeypatch.setattr(adapter, "send_attitude_rate", recorded_send)

    result = asyncio.run(
        runner._observe_gate1(details, hold_thrust=hold_thrust)
    )

    assert result["gate1_observed"]
    assert result["frame_count"] == 3
    assert [frame["frame_id"] for frame in result["frames"]] == [101, 102, 103]
    assert len(adapter.commands) == 2
    assert all(
        command.roll_rate == command.pitch_rate == command.yaw_rate == 0.0
        and command.thrust == hold_thrust
        for command in adapter.commands
    )
    assert all(
        later - earlier >= vq2_module.CONTROL_PERIOD_S - 1e-12
        for earlier, later in zip(command_times, command_times[1:])
    )
    assert watchdog_require_target == [False] * 6
    assert vision.reset_calls == 0
    assert vision.is_running is False


@pytest.mark.parametrize("geometry_source", ("accepted", "raw"))
def test_powered_gate1_observation_enforces_no_passage_geometry_before_send(
    monkeypatch,
    geometry_source,
):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(clock=clock)

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        if geometry_source == "accepted":
            target = vq2_module.GateTarget(
                frame_id=101,
                sim_time_ns=1_010,
                received_monotonic_s=clock[0],
                center_x=520,
                center_y=50,
                bbox=(440, 0, 160, 100),
                confidence=0.8,
            )
            runner.tracker.target = target
            runner.tracker.consecutive = 1
            runner.tracker.last_selection_mode = "primary"
            runner._latest_accepted_target = target
        else:
            runner._latest_raw_detections = [
                _detection(440, 0, 160, 144)
            ]

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)

    with pytest.raises(SafetyAbort, match="no-passage geometry bound"):
        asyncio.run(
            runner._observe_gate1(
                details,
                hold_thrust=vq2_module.GATE1_RECENTER_TRANSITION_THRUST,
            )
        )

    assert adapter.commands == []


def test_gate1_observation_resets_streak_after_missing_frame(monkeypatch):
    clock = [10.0]
    runner, _adapter, _vision, details = _configure_gate1_observer(clock=clock)
    sample_count = [0]

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        sample_count[0] += 1
        detection = (
            None
            if sample_count[0] == 2
            else _detection(410 + sample_count[0], 138, 28, 45)
        )
        _publish_observation_frame(
            runner,
            frame_id=100 + sample_count[0],
            clock=clock,
            detection=detection,
        )

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)

    result = asyncio.run(runner._observe_gate1(details))

    assert [frame["frame_id"] for frame in result["frames"]] == [103, 104, 105]


def test_gate1_observation_uses_fixed_nested_deadline_and_never_catches_up(
    monkeypatch,
):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(
        clock=clock,
        crossing_started_s=9.65,
    )
    command_times = []

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        # Repeated control ticks without a new frame can never build a streak.
        return None

    async def recorded_send(command):
        adapter.commands.append(command)
        command_times.append(clock[0])
        if len(command_times) == 1:
            clock[0] += 0.065  # A send stall must drop, not replay, ticks.

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(adapter, "send_attitude_rate", recorded_send)

    with pytest.raises(SafetyAbort, match="timed out"):
        asyncio.run(runner._observe_gate1(details))

    # crossing start + 0.40 is the fixed 10.05 hard limit; the injected send
    # completion overrun is observed once and never followed by catch-up sends.
    assert command_times == [10.0]
    assert clock[0] == pytest.approx(10.065)
    assert all(command.thrust == 0.0 for command in adapter.commands)


def test_gate1_observation_cannot_accept_third_frame_after_hard_deadline(
    monkeypatch,
):
    clock = [10.0]
    runner, _adapter, _vision, details = _configure_gate1_observer(clock=clock)
    sample_count = [0]

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        sample_count[0] += 1
        if sample_count[0] == 3:
            clock[0] += 0.09  # detector stall crosses the fixed 10.10 deadline
        _publish_observation_frame(
            runner,
            frame_id=100 + sample_count[0],
            clock=clock,
            detection=_detection(409 + sample_count[0], 138, 28, 45),
        )

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)

    with pytest.raises(SafetyAbort, match="timed out"):
        asyncio.run(runner._observe_gate1(details))

    assert sample_count[0] == 3
    assert runner.tracker.consecutive == 3
    assert clock[0] > 10.10


def test_gate1_observation_rechecks_deadline_after_diagnostic_logging(monkeypatch):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(clock=clock)
    sample_count = [0]

    class AdvancingRecorder:
        path = None

        def emit(self, event, **fields):
            if event == "post_gate_candidate_frame" and fields["tracker_streak"] == 3:
                clock[0] += 0.09

        def save_png(self, _label, _image):
            raise AssertionError("no image should be encoded while observing")

    runner.recorder = AdvancingRecorder()

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        sample_count[0] += 1
        _publish_observation_frame(
            runner,
            frame_id=100 + sample_count[0],
            clock=clock,
            detection=_detection(409 + sample_count[0], 138, 28, 45),
        )

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)

    with pytest.raises(SafetyAbort, match="timed out"):
        asyncio.run(runner._observe_gate1(details))

    assert sample_count[0] == 3
    assert len(adapter.commands) == 2
    assert clock[0] > 10.10


def test_gate1_observation_rechecks_watchdog_after_logging_stall(monkeypatch):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(clock=clock)
    watchdog_count = [0]

    class AdvancingRecorder:
        path = None

        def emit(self, event, **_fields):
            if event == "post_gate_candidate_frame":
                clock[0] += 0.06

        def save_png(self, _label, _image):
            raise AssertionError("no image should be encoded while observing")

    runner.recorder = AdvancingRecorder()

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        _publish_observation_frame(
            runner,
            frame_id=101,
            clock=clock,
            detection=_detection(410, 138, 28, 45),
        )

    def fake_watchdog(**_kwargs):
        watchdog_count[0] += 1
        if watchdog_count[0] == 2:
            raise SafetyAbort("IMU timestamp not advancing")

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", fake_watchdog)

    with pytest.raises(SafetyAbort, match="IMU timestamp not advancing"):
        asyncio.run(runner._observe_gate1(details))

    assert watchdog_count[0] == 2
    assert clock[0] == pytest.approx(10.06)
    assert adapter.commands == []


def test_gate1_observation_rechecks_deadline_after_final_watchdog(monkeypatch):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(clock=clock)
    sample_count = [0]
    watchdog_count = [0]

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        sample_count[0] += 1
        _publish_observation_frame(
            runner,
            frame_id=100 + sample_count[0],
            clock=clock,
            detection=_detection(409 + sample_count[0], 138, 28, 45),
        )

    def fake_watchdog(**_kwargs):
        watchdog_count[0] += 1
        if watchdog_count[0] == 6:
            clock[0] += 0.09

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", fake_watchdog)

    with pytest.raises(SafetyAbort, match="timed out"):
        asyncio.run(runner._observe_gate1(details))

    assert watchdog_count[0] == 6
    assert len(adapter.commands) == 2


@pytest.mark.parametrize("gate_index", [0, 2])
def test_gate1_observation_aborts_on_any_gate_index_change_before_send(
    monkeypatch,
    gate_index,
):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(clock=clock)

    def fake_sample():
        adapter.race_status = RaceStatus(
            sim_boot_time_ms=1251,
            race_start_boot_time_ms=0,
            race_finish_time_ns=-1,
            active_gate_index=gate_index,
            last_gate_race_time=123,
        )

    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vq2_module.asyncio, "sleep", lambda _seconds: asyncio.sleep(0))
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)

    with pytest.raises(SafetyAbort, match="gate index changed"):
        asyncio.run(runner._observe_gate1(details))

    assert adapter.commands == []


def test_gate0_early_gate1_status_requires_strictly_newer_packet(monkeypatch):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1000,
        race_start_boot_time_ms=0,
        race_finish_time_ns=-1,
        active_gate_index=1,
        last_gate_race_time=123,
    )
    runner = VQ2Runner(adapter, _FakeVision())
    runner.estimate = _estimate()
    runner.tracker.target = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=0.0,
        center_x=322,
        center_y=174,
        bbox=(282, 134, 80, 80),
        confidence=0.8,
    )
    runner.tracker.consecutive = 3
    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(runner, "_sample", lambda: None)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    with pytest.raises(SafetyAbort, match="not strictly newer"):
        asyncio.run(runner._run_gate0(context))

    assert adapter.commands == []


@pytest.mark.parametrize("write_diagnostic_pngs", [True, False])
def test_gate0_observe_dispatch_preserves_credit_with_optional_pngs(
    monkeypatch,
    write_diagnostic_pngs,
):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1000,
        race_start_boot_time_ms=0,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    runner = VQ2Runner(adapter, _FakeVision())
    calls = []
    gate0_details = {
        "gate0_passed": True,
        "gate_transition_proved": True,
        "pre_gate_race_boot_ms": 1000,
        "race_boot_ms": 1250,
    }

    async def establish(**_kwargs):
        calls.append("reset")

    async def normalize():
        calls.append("normalize")

    async def wait_for_go():
        calls.append("go")
        return vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    async def arm():
        calls.append("arm")

    async def gate0(_context, *, capture_transition=False):
        calls.append(("gate0", capture_transition))
        return gate0_details

    async def observe(details):
        calls.append(("observe", details is gate0_details))
        raise SafetyAbort("diagnostic observation timeout")

    async def cleanup():
        calls.append("cleanup")
        return True

    monkeypatch.setattr(runner, "establish_reset_epoch", establish)
    monkeypatch.setattr(runner, "normalize_disarmed", normalize)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(runner, "arm_confirmed", arm)
    monkeypatch.setattr(runner, "_run_gate0", gate0)
    monkeypatch.setattr(runner, "_observe_gate1", observe)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)
    flush_calls = []
    monkeypatch.setattr(
        runner,
        "_flush_deferred_snapshots",
        lambda: (flush_calls.append(True) or ([], [])),
    )

    result = asyncio.run(
        runner.run_powered_stage(
            "gate0-observe",
            write_diagnostic_pngs=write_diagnostic_pngs,
        )
    )

    assert not result.success
    assert result.cleanup_confirmed
    assert result.details["gate0"] == gate0_details
    assert result.details["gate1_observation"] == {
        "gate1_observed": False,
        "reason": "diagnostic observation timeout",
    }
    assert flush_calls == ([True] if write_diagnostic_pngs else [])
    assert calls == [
        "reset",
        "normalize",
        "go",
        "arm",
        ("gate0", write_diagnostic_pngs),
        ("observe", True),
        "cleanup",
    ]


def test_gate0_stage_does_not_enter_post_pass_observation(monkeypatch):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    runner = VQ2Runner(adapter, _FakeVision())

    async def no_op(*_args, **_kwargs):
        return None

    async def wait_for_go():
        return vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    async def gate0(_context, *, capture_transition=False):
        assert not capture_transition
        return {"gate0_passed": True}

    async def observe(_details):
        raise AssertionError("plain gate0 must not observe gate 1")

    async def cleanup():
        return True

    monkeypatch.setattr(runner, "establish_reset_epoch", no_op)
    monkeypatch.setattr(runner, "normalize_disarmed", no_op)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(runner, "arm_confirmed", no_op)
    monkeypatch.setattr(runner, "_run_gate0", gate0)
    monkeypatch.setattr(runner, "_observe_gate1", observe)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(runner.run_powered_stage("gate0"))

    assert result.success
    assert result.details["gate0_passed"] is True
    assert result.details["cleanup_collision_safety"]["safe"] is True


@pytest.mark.parametrize("stage", ["full-lap"])
def test_unaccepted_course_stages_are_rejected_before_powered_lifecycle(
    monkeypatch,
    stage,
):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    monkeypatch.setattr(
        runner,
        "establish_reset_epoch",
        lambda **_kwargs: pytest.fail(f"{stage} reached reset"),
    )

    with pytest.raises(ValueError, match="unsupported powered stage"):
        asyncio.run(
            runner.run_powered_stage(stage, write_diagnostic_pngs=False)
        )


@pytest.mark.parametrize("stage", ["full-lap"])
def test_unaccepted_course_stages_are_rejected_before_live_import_or_contact(
    monkeypatch,
    stage,
):
    live_imports = []
    monkeypatch.setattr(
        vq2_module,
        "_load_live_transport_dependencies",
        lambda: live_imports.append(stage),
    )

    with pytest.raises(ValueError, match="unsupported live stage"):
        asyncio.run(
            vq2_module.run_live(
                stage,
                "udpin:127.0.0.1:14550",
                None,
            )
        )

    assert live_imports == []


@pytest.mark.parametrize("profile_hash", [None, "f" * 64])
def test_visual_course_refuses_missing_or_wrong_yaw_profile_before_transport(
    monkeypatch,
    profile_hash,
):
    live_imports = []
    monkeypatch.setattr(
        vq2_module,
        "_load_live_transport_dependencies",
        lambda: live_imports.append("visual-course"),
    )

    with pytest.raises(
        PermissionError,
        match="exact reviewed yaw profile hash",
    ):
        asyncio.run(
            vq2_module.run_live(
                vq2_module.VISUAL_COURSE_STAGE,
                "udpin:127.0.0.1:14550",
                None,
                expected_yaw_calibration_profile_sha256=profile_hash,
            )
        )

    assert live_imports == []


def test_visual_course_rejects_replay_before_transport_with_other_bindings_valid(
    tmp_path,
    monkeypatch,
):
    live_imports = []
    monkeypatch.setattr(
        vq2_module,
        "_load_live_transport_dependencies",
        lambda: live_imports.append("visual-course"),
    )
    visual_config = vq2_module.default_visual_config()

    with pytest.raises(
        PermissionError,
        match="compact manifest/JSONL/result/lease evidence",
    ):
        asyncio.run(
            vq2_module.run_live(
                vq2_module.VISUAL_COURSE_STAGE,
                "udpin:127.0.0.1:14550",
                str(tmp_path / "session.jsonl.gz"),
                replay_bundle=str(tmp_path / "forbidden.vq2replay"),
                recording_approved=True,
                preflight_before_powered_stage=False,
                write_diagnostic_pngs=False,
                run_manifest_sha256="a" * 64,
                candidate_commit="b" * 40,
                expected_controller_config_sha256=(
                    visual_config.effective_config_sha256
                ),
                expected_yaw_calibration_profile_sha256=(
                    vq2_module.YAW_CALIBRATION_PROFILE_SHA256
                ),
            )
        )

    assert live_imports == []


def test_visual_course_requires_compact_jsonl_before_transport(monkeypatch):
    live_imports = []
    monkeypatch.setattr(
        vq2_module,
        "_load_live_transport_dependencies",
        lambda: live_imports.append("visual-course"),
    )
    visual_config = vq2_module.default_visual_config()

    with pytest.raises(
        PermissionError,
        match="manifest-bound fast-cycle wrapper",
    ):
        asyncio.run(
            vq2_module.run_live(
                vq2_module.VISUAL_COURSE_STAGE,
                "udpin:127.0.0.1:14550",
                None,
                replay_bundle=None,
                recording_approved=False,
                preflight_before_powered_stage=False,
                write_diagnostic_pngs=False,
                run_manifest_sha256="a" * 64,
                candidate_commit="b" * 40,
                expected_controller_config_sha256=(
                    visual_config.effective_config_sha256
                ),
                expected_yaw_calibration_profile_sha256=(
                    vq2_module.YAW_CALIBRATION_PROFILE_SHA256
                ),
            )
        )

    assert live_imports == []


def test_visual_course_exact_compact_arguments_reach_mocked_transport(
    tmp_path,
    monkeypatch,
):
    from competition.aigp_mavlink import (
        MavlinkIngressStats,
        MavlinkOutboundAudit,
    )

    class ContactObserved(RuntimeError):
        pass

    observed = {
        "loads": 0,
        "adapter_kwargs": None,
        "vision_kwargs": None,
        "address": None,
        "disconnects": 0,
    }

    class CompactAdapter(_FakeAdapter):
        async def connect(self, address):
            observed["address"] = address
            raise ContactObserved("compact visual-course reached transport")

        async def disconnect(self):
            observed["disconnects"] += 1

        def drain_received_ingress(self):
            return []

        def ingress_stats(self):
            return MavlinkIngressStats(
                generation=1,
                next_sequence=0,
                highres_imu_received=0,
                heartbeat_received=0,
                race_status_received=0,
                actuator_received=0,
                dropped=0,
                high_watermark=0,
                imu_capacity=1,
                other_capacity=1,
                imu_dropped=0,
                other_dropped=0,
                imu_high_watermark=0,
                other_high_watermark=0,
                buffered_imu=0,
                buffered_other=0,
            )

        def outbound_audit(self):
            return MavlinkOutboundAudit(0, 0, 0, 0, 0, 0, 0, 0)

    def load_transport():
        observed["loads"] += 1

    def adapter_factory(**kwargs):
        observed["adapter_kwargs"] = dict(kwargs)
        return CompactAdapter()

    def vision_factory(**kwargs):
        observed["vision_kwargs"] = dict(kwargs)
        return _FakeVision()

    monkeypatch.setattr(
        vq2_module,
        "_load_live_transport_dependencies",
        load_transport,
    )
    monkeypatch.setattr(vq2_module, "AIGPMavlinkAdapter", adapter_factory)
    monkeypatch.setattr(vq2_module, "VQ2VisionThread", vision_factory)
    monkeypatch.setattr(
        vq2_module,
        "_replay_capture_dependencies",
        lambda: pytest.fail("compact visual-course constructed replay"),
    )
    visual_config = vq2_module.default_visual_config()
    address = "udpin:127.0.0.1:14550"
    trace = tmp_path / "session.jsonl.gz"

    with pytest.raises(
        ContactObserved,
        match="reached transport",
    ):
        asyncio.run(
            vq2_module.run_live(
                vq2_module.VISUAL_COURSE_STAGE,
                address,
                str(trace),
                replay_bundle=None,
                recording_approved=False,
                preflight_before_powered_stage=False,
                write_diagnostic_pngs=False,
                run_manifest_sha256="a" * 64,
                candidate_commit="b" * 40,
                expected_controller_config_sha256=(
                    visual_config.effective_config_sha256
                ),
                expected_yaw_calibration_profile_sha256=(
                    vq2_module.YAW_CALIBRATION_PROFILE_SHA256
                ),
            )
        )

    assert observed == {
        "loads": 1,
        "adapter_kwargs": {
            "enable_vision": False,
            "require_track": False,
            "telemetry_mode": "imu",
            "fetch_track_on_connect": False,
        },
        "vision_kwargs": {
            "on_snapshot": None,
            "capture_snapshot_queue_enabled": False,
        },
        "address": address,
        "disconnects": 1,
    }
    with vq2_module.gzip.open(trace, "rt", encoding="utf-8") as handle:
        binding = json.loads(handle.readline())
    assert binding["event"] == "fast_cycle_binding"
    assert binding["run_manifest_sha256"] == "a" * 64
    assert binding["yaw_calibration_profile"]["sha256"] == (
        vq2_module.YAW_CALIBRATION_PROFILE_SHA256
    )


def test_runner_binds_exact_yaw_profile_and_rejects_mismatched_evidence():
    profile = vq2_module.load_yaw_calibration_profile()
    evidence = vq2_module.build_yaw_calibration_profile_evidence(profile)

    runner = VQ2Runner(
        _FakeAdapter(),
        _FakeVision(),
        yaw_calibration_profile=profile,
        yaw_calibration_profile_evidence=evidence,
    )

    assert runner.yaw_calibration_profile_evidence == evidence
    altered = dict(evidence)
    altered["sha256"] = "f" * 64
    with pytest.raises(
        ValueError,
        match="evidence does not match",
    ):
        VQ2Runner(
            _FakeAdapter(),
            _FakeVision(),
            yaw_calibration_profile=profile,
            yaw_calibration_profile_evidence=altered,
        )


@pytest.mark.parametrize(
    (
        "cleanup_confirmed",
        "criteria_met",
        "cleanup_entry_gate_index",
        "expected_success",
    ),
    (
        (True, True, 1, True),
        (False, True, 1, True),
        (True, False, 1, False),
        (True, True, 2, False),
    ),
)
def test_gate1_recenter_preserves_navigation_but_requires_cleanup_for_run(
    monkeypatch,
    cleanup_confirmed,
    criteria_met,
    cleanup_entry_gate_index,
    expected_success,
):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)
    calls = []

    async def no_op(*_args, **_kwargs):
        return None

    async def wait_for_go():
        return context

    async def run_gate0(observed_context, **kwargs):
        calls.append(("gate0", observed_context, kwargs))
        return {"gate0_passed": True}

    async def observe_gate1(gate0, **kwargs):
        calls.append(("observe", gate0, kwargs))
        return {"gate1_observed": True}

    async def run_recenter(observation):
        calls.append(("recenter", observation))
        adapter.race_status = RaceStatus(
            sim_boot_time_ms=1100,
            race_start_boot_time_ms=0,
            race_finish_time_ns=-1,
            active_gate_index=cleanup_entry_gate_index,
            last_gate_race_time=-1,
        )
        summary = {
            "success": False,
            "recenter_criteria_met": criteria_met,
            "cleanup_confirmed": False,
            "outcome": "corridor_hold",
        }
        runner._gate1_recenter_summary = dict(summary)
        return dict(summary)

    async def cleanup():
        calls.append(("cleanup",))
        runner._gate1_recenter_summary = None
        return cleanup_confirmed

    monkeypatch.setattr(runner, "establish_reset_epoch", no_op)
    monkeypatch.setattr(runner, "normalize_disarmed", no_op)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(runner, "arm_confirmed", no_op)
    monkeypatch.setattr(runner, "_run_gate0", run_gate0)
    monkeypatch.setattr(runner, "_observe_gate1", observe_gate1)
    monkeypatch.setattr(runner, "_run_bounded_gate1_recenter", run_recenter)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(
        runner.run_powered_stage(
            "gate1-recenter",
            write_diagnostic_pngs=False,
        )
    )

    assert result.success is bool(expected_success and cleanup_confirmed)
    assert result.cleanup_confirmed is cleanup_confirmed
    assert result.details["gate1_recenter"]["cleanup_confirmed"] is (
        cleanup_confirmed
    )
    assert result.details["gate1_recenter"]["success"] is expected_success
    if not cleanup_confirmed:
        assert "cleanup unconfirmed" in result.reason
    if not criteria_met:
        assert "without satisfying its criteria" in result.reason
    if cleanup_entry_gate_index != 1:
        assert "cleanup boundary lost gate 1 authority" in result.reason
        assert (
            result.details["gate1_recenter"]["authoritative_max_gate_index"]
            == cleanup_entry_gate_index
        )
    assert [call[0] for call in calls] == [
        "gate0",
        "observe",
        "recenter",
        "cleanup",
    ]
    assert calls[0][2]["crossing_hold_thrust"] == 0.275
    assert calls[0][2]["course_line_preturn"] is True
    assert (
        calls[0][2]["course_line_exit_counterroll_enabled"]
        is True
    )
    assert calls[1][2]["hold_thrust"] == 0.275


def test_gate1_recenter_powered_lifecycle_persists_abort_summary(monkeypatch):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    runner = VQ2Runner(adapter, _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    async def no_op(*_args, **_kwargs):
        return None

    async def wait_for_go():
        return context

    async def run_gate0(_context, **_kwargs):
        return {"gate0_passed": True}

    async def observe_gate1(_gate0, **_kwargs):
        return {"gate1_observed": True}

    async def abort_recenter(_observation):
        runner._gate1_recenter_summary = {
            "success": False,
            "recenter_criteria_met": False,
            "cleanup_confirmed": False,
            "outcome": "abort",
            "reason": "injected bounded abort",
        }
        raise SafetyAbort("injected bounded abort")

    async def cleanup():
        runner._gate1_recenter_summary = None
        return True

    monkeypatch.setattr(runner, "establish_reset_epoch", no_op)
    monkeypatch.setattr(runner, "normalize_disarmed", no_op)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(runner, "arm_confirmed", no_op)
    monkeypatch.setattr(runner, "_run_gate0", run_gate0)
    monkeypatch.setattr(runner, "_observe_gate1", observe_gate1)
    monkeypatch.setattr(runner, "_run_bounded_gate1_recenter", abort_recenter)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(
        runner.run_powered_stage(
            "gate1-recenter",
            write_diagnostic_pngs=False,
        )
    )

    assert result.success is False
    assert result.cleanup_confirmed is True
    summary = result.details["gate1_recenter"]
    assert summary["success"] is False
    assert summary["recenter_criteria_met"] is False
    assert summary["cleanup_confirmed"] is True
    assert summary["outcome"] == "abort"
    assert summary["reason"] == "injected bounded abort"


def test_offline_full_lap_scaffold_disables_unaccepted_gate0_overrides(
    monkeypatch,
):
    class Gate0Observed(Exception):
        pass

    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)
    observed = []

    async def gate0(
        observed_context,
        *,
        capture_transition=False,
        exit_pitch_rad=0.0,
        minimum_thrust=0.21,
        boost_until_s=0.45,
        observe_course_line=False,
        course_line_preturn=False,
        course_line_exit_counterroll_enabled=False,
    ):
        observed.append(
            (
                observed_context,
                capture_transition,
                exit_pitch_rad,
                minimum_thrust,
                boost_until_s,
                observe_course_line,
                course_line_preturn,
                course_line_exit_counterroll_enabled,
            )
        )
        raise Gate0Observed

    monkeypatch.setattr(runner, "_run_gate0", gate0)

    with pytest.raises(Gate0Observed):
        asyncio.run(runner._run_full_lap(context))

    assert observed == [
        (
            context,
            False,
            0.0,
            0.21,
            0.45,
            False,
            False,
            False,
        )
    ]


def test_passive_preflight_requires_the_requested_continuous_healthy_dwell(
    monkeypatch,
):
    clock = [0.0]

    class DwellVision(_FakeVision):
        is_running = True

        def stats(self):
            return SimpleNamespace(
                frames_decoded=int(clock[0] * 31.0),
                duplicate_datagrams=0,
            )

    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(1_000, -1, -1, 0, -1)
    runner = VQ2Runner(adapter, DwellVision())
    runner.estimate = _estimate()
    runner.tracker.target = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=0.0,
        center_x=322,
        center_y=174,
        bbox=(282, 134, 80, 81),
        confidence=0.8,
    )
    runner.tracker.consecutive = 3
    monkeypatch.setattr(runner, "_clear_epoch_state", lambda: None)
    monkeypatch.setattr(runner, "_sample", lambda: None)
    monkeypatch.setattr(runner, "_stream_failures", lambda **_kwargs: [])
    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])

    async def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr(vq2_module.asyncio, "sleep", advance)

    result = asyncio.run(
        runner.preflight(timeout_s=2.0, healthy_dwell_s=0.05)
    )

    assert result["requested_healthy_dwell_s"] == pytest.approx(0.05)
    assert result["healthy_dwell_s"] >= 0.05
    assert result["observation_duration_s"] >= 1.05
    assert result["vision_frames"] >= 32


def test_passive_preflight_rejects_an_unbounded_dwell_before_receiving():
    runner = VQ2Runner(_FakeAdapter(), _FakeVision())
    with pytest.raises(ValueError, match=r"\[0, 8\]"):
        asyncio.run(runner.preflight(healthy_dwell_s=8.1))


def test_diagnostic_png_encoding_is_deferred_until_explicit_flush(tmp_path):
    recorder = vq2_module.JsonlRecorder(str(tmp_path / "trace.jsonl.gz"))
    vision = _FakeVision()
    vision.current_snapshot = _vision_snapshot()
    runner = VQ2Runner(_FakeAdapter(), vision, recorder=recorder)

    metadata = runner._defer_snapshot("gate1 acquired")

    assert metadata is not None
    assert list(tmp_path.glob("*.png")) == []
    paths, errors = runner._flush_deferred_snapshots()
    recorder.close()

    assert errors == []
    assert len(paths) == 1
    assert (tmp_path / "trace_gate1_acquired.png").is_file()


def test_programmatic_replay_capture_requires_exact_recording_approval(tmp_path):
    with pytest.raises(PermissionError, match="recording_approved=True"):
        asyncio.run(
            vq2_module.run_live(
                "preflight",
                "udp://127.0.0.1:14550",
                None,
                replay_bundle=str(tmp_path / "private.vq2replay"),
            )
        )
    with pytest.raises(TypeError, match="exact bool"):
        asyncio.run(
            vq2_module.run_live(
                "preflight",
                "udp://127.0.0.1:14550",
                None,
                recording_approved="true",
            )
        )


def test_connect_failure_still_disconnects_partially_started_transport(monkeypatch):
    from competition.aigp_mavlink import MavlinkIngressStats, MavlinkOutboundAudit

    class FailingConnectAdapter(_FakeAdapter):
        def __init__(self):
            super().__init__()
            self.disconnect_called = False

        async def connect(self, _address):
            raise ConnectionError("connect failed after transport start")

        async def disconnect(self):
            self.disconnect_called = True

        def drain_received_ingress(self):
            return []

        def ingress_stats(self):
            return MavlinkIngressStats(
                generation=1,
                next_sequence=0,
                highres_imu_received=0,
                heartbeat_received=0,
                race_status_received=0,
                actuator_received=0,
                dropped=0,
                high_watermark=0,
                imu_capacity=1,
                other_capacity=1,
                imu_dropped=0,
                other_dropped=0,
                imu_high_watermark=0,
                other_high_watermark=0,
                buffered_imu=0,
                buffered_other=0,
            )

        def outbound_audit(self):
            return MavlinkOutboundAudit(0, 0, 0, 0, 0, 0, 0, 0)

    adapter = FailingConnectAdapter()
    monkeypatch.setattr(
        vq2_module, "AIGPMavlinkAdapter", lambda **_kwargs: adapter
    )

    with pytest.raises(ConnectionError, match="after transport start"):
        asyncio.run(
            vq2_module.run_live(
                "preflight",
                "udpin:127.0.0.1:14550",
                None,
            )
        )

    assert adapter.disconnect_called is True


def test_fast_cycle_trace_first_row_binds_complete_controller_identity(
    tmp_path,
    monkeypatch,
):
    from competition.aigp_mavlink import MavlinkIngressStats, MavlinkOutboundAudit

    class FailingConnectAdapter(_FakeAdapter):
        def __init__(self):
            super().__init__()
            self.disconnect_called = False

        async def connect(self, _address):
            raise ConnectionError("injected pre-contact failure")

        async def disconnect(self):
            self.disconnect_called = True

        def drain_received_ingress(self):
            return []

        def ingress_stats(self):
            return MavlinkIngressStats(
                generation=1,
                next_sequence=0,
                highres_imu_received=0,
                heartbeat_received=0,
                race_status_received=0,
                actuator_received=0,
                dropped=0,
                high_watermark=0,
                imu_capacity=1,
                other_capacity=1,
                imu_dropped=0,
                other_dropped=0,
                imu_high_watermark=0,
                other_high_watermark=0,
                buffered_imu=0,
                buffered_other=0,
            )

        def outbound_audit(self):
            return MavlinkOutboundAudit(0, 0, 0, 0, 0, 0, 0, 0)

    adapter = FailingConnectAdapter()
    monkeypatch.setattr(
        vq2_module,
        "AIGPMavlinkAdapter",
        lambda **_kwargs: adapter,
    )
    document = controller_config_module.default_controller_config_mapping()
    document["yaw_control"]["gate1_error_gain"] = -0.12
    document["yaw_control"]["command_rate_cap_rad_s"] = 0.08
    config = controller_config_module.validate_controller_config(document)
    trace = tmp_path / "binding.jsonl.gz"

    with pytest.raises(ConnectionError, match="pre-contact"):
        asyncio.run(
            vq2_module.run_live(
                "gate1-recenter",
                "udpin:127.0.0.1:14550",
                str(trace),
                run_manifest_sha256="a" * 64,
                controller_config=config,
                candidate_commit="b" * 40,
                expected_controller_config_sha256=(
                    config.effective_config_sha256
                ),
            )
        )

    with vq2_module.gzip.open(trace, "rt", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    assert first["event"] == "fast_cycle_binding"
    assert first["run_manifest_sha256"] == "a" * 64
    assert first["controller"] == vq2_module.controller_config_evidence(
        config,
        candidate_commit="b" * 40,
    )
    assert first["controller"]["config_sha256"] == (
        config.effective_config_sha256
    )
    assert adapter.disconnect_called is True


def test_replay_writer_is_cleaned_up_when_later_runner_construction_fails(
    tmp_path, monkeypatch
):
    created = []
    real_dependencies = vq2_module._replay_capture_dependencies()
    real_recorder = real_dependencies[0]

    def tracking_recorder(*args, **kwargs):
        recorder = real_recorder(*args, **kwargs)
        created.append(recorder)
        return recorder

    class ConstructorFailure:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("vision constructor failed")

    monkeypatch.setattr(
        vq2_module,
        "_replay_capture_dependencies",
        lambda: (tracking_recorder, *real_dependencies[1:]),
    )
    monkeypatch.setattr(vq2_module, "VQ2VisionThread", ConstructorFailure)
    bundle = tmp_path / "constructor-failure.vq2replay"
    with pytest.raises(RuntimeError, match="vision constructor failed"):
        asyncio.run(
            vq2_module.run_live(
                "preflight",
                "udp://127.0.0.1:14550",
                None,
                replay_bundle=str(bundle),
                recording_approved=True,
            )
        )
    assert len(created) == 1
    assert not created[0]._thread.is_alive()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["metadata"]["seed"] == 42


def test_replay_writer_is_aborted_when_async_recorder_construction_fails(
    tmp_path, monkeypatch
):
    class ConstructorFailure:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("async recorder construction failed")

    real_dependencies = vq2_module._replay_capture_dependencies()
    monkeypatch.setattr(
        vq2_module,
        "_replay_capture_dependencies",
        lambda: (ConstructorFailure, *real_dependencies[1:]),
    )
    bundle = tmp_path / "async-constructor-failure.vq2replay"
    with pytest.raises(RuntimeError, match="async recorder construction failed"):
        asyncio.run(
            vq2_module.run_live(
                "preflight",
                "udp://127.0.0.1:14550",
                None,
                replay_bundle=str(bundle),
                recording_approved=True,
            )
        )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert "async replay recorder construction failed" in manifest["abort_reason"]


def test_visual_course_replay_envelope_matches_generic_lifecycle():
    envelope = vq2_module.replay_controller_envelope(
        vq2_module.VISUAL_COURSE_STAGE
    )
    phases = envelope["phase_envelopes"]

    passage = phases["reviewed_gate_passage"]
    assert passage["allow_advance"] is True
    assert passage["next_gate_preview"] is True
    assert passage["next_gate_preview_requires_reviewed_identity"] is True
    assert passage["next_gate_preview_may_retire_fail_closed"] is True

    recovery = phases["post_credit_recovery"]
    assert recovery["promoted_history_preserved"] is True
    assert recovery["same_promoted_identity_required"] is True
    assert recovery["strictly_post_credit_observation_required"] is True
    assert recovery["strictly_post_credit_publication_required"] is True
    assert recovery["neutral_credit_boundary_command_authority"] is False
    assert "complete_visibility_epoch_admission_required" not in recovery


def test_visual_alignment_replay_metadata_declares_phase_envelopes(
    tmp_path,
    monkeypatch,
):
    class ConstructorFailure:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("vision constructor failed")

    monkeypatch.setattr(
        vq2_module,
        "VQ2VisionThread",
        ConstructorFailure,
    )
    bundle = tmp_path / "visual-alignment-envelope.vq2replay"
    config = vq2_module.default_visual_config()

    with pytest.raises(RuntimeError, match="vision constructor failed"):
        asyncio.run(
            vq2_module.run_live(
                vq2_module.VISUAL_ALIGN_STAGE,
                "udp://127.0.0.1:14550",
                str(tmp_path / "visual-alignment.jsonl.gz"),
                replay_bundle=str(bundle),
                recording_approved=True,
                preflight_before_powered_stage=False,
                write_diagnostic_pngs=False,
                run_manifest_sha256="a" * 64,
                candidate_commit="b" * 40,
                expected_controller_config_sha256=(
                    config.effective_config_sha256
                ),
            )
        )

    manifest = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    envelope = manifest["metadata"]["controller_envelope"]
    assert envelope == vq2_module.replay_controller_envelope(
        vq2_module.VISUAL_ALIGN_STAGE
    )
    assert envelope["max_roll_pitch_command_rate_rad_s"] == 0.25
    assert envelope["max_thrust"] == 0.32
    phases = envelope["phase_envelopes"]
    assert phases["gate0_visual_handoff_bootstrap"][
        "axis_authority"
    ] == {
        "yaw": "visual_next_track_blend",
        "pitch": "bounded_visual_brake",
        "roll_collective": "proved_gate0_bootstrap",
    }
    assert phases["gate0_visual_handoff_bootstrap"][
        "passage_refusal_command_authority"
    ] == "exact_zero_visual"
    assert phases["gate0_visual_handoff_bootstrap"][
        "max_consecutive_passage_suspension_fresh_frames"
    ] == 2
    assert phases["gate0_visual_handoff_bootstrap"][
        "max_total_passage_suspension_fresh_frames"
    ] == 4
    assert phases["gate0_visual_handoff_bootstrap"][
        "max_passage_suspension_epochs"
    ] == 3
    assert phases["gate0_visual_handoff_bootstrap"][
        "max_passage_suspension_epoch_duration_s"
    ] == 0.12
    assert phases["gate0_visual_handoff_bootstrap"][
        "max_total_passage_suspension_duration_s"
    ] == 0.20
    assert phases["gate0_visual_handoff_bootstrap"][
        "transient_passage_violation_codes"
    ] == ["current_projected_vertical"]
    assert phases["gate0_visual_handoff_bootstrap"][
        "transient_projected_vertical_side"
    ] == "negative_image_down"
    assert phases["gate0_visual_handoff_bootstrap"][
        "transient_projected_vertical_limit_norm"
    ] == -0.28
    assert phases["gate0_visual_handoff_bootstrap"][
        "max_transient_projected_vertical_excess_norm"
    ] == 0.01
    assert phases["gate0_visual_handoff_bootstrap"][
        "resume_requires_same_latched_identity"
    ] is True
    assert phases["crossing_confirmation"] == {
        "exact_zero_rate_zero_thrust": True,
        "max_wait_s": 0.40,
    }
    assert phases["post_credit_fresh_frame_wait"] == {
        "exact_zero_rate_zero_thrust": True,
        "conditional_when_recovery_not_required": True,
        "max_wait_s": 0.12,
    }
    assert phases["postpromotion_no_advance_recovery"] == {
        "conditional": True,
        "exact_transition_anchor_required": True,
        "credit_boundary_identity_required": True,
        "promotion_anchor_identity_required": True,
        "promotion_history_length_at_credit_required": True,
        "max_initial_postcredit_promotion_frames": 1,
        "stale_credit_anchor_command_allowed": False,
        "receipt_backed_dispatch": True,
        "fresh_postcredit_observation_required_before_repeat": True,
        "wire_target_must_match_latest_receiver_publication": True,
        "receiver_publication_lock_held_through_transport_return": True,
        "wire_time_projection_revalidation": True,
        "promotion_history_sha256_required": True,
        "promotion_history_prevalidated_before_wire_lease": True,
        "wire_time_history_check": "module_sealed_exact_prefix",
        "max_validation_to_wire_delay_s": 0.005,
        "command_response_horizon_s": 0.045,
        "min_projection_horizon_s": 0.080,
        "stable_tail_sample_count": 4,
        "min_stable_tail_span_s": 0.090,
        "min_stable_tail_detection_confidence": 0.65,
        "min_stable_tail_association_confidence": 0.90,
        "transition_tokens_equal_credit_visibility_epoch": True,
        "sealed_postcredit_promotion_suffix_included": True,
        "post_bridge_epoch_clean_associations_required": True,
        "non_bridge_publication_skips_allowed": False,
        "max_filtered_center_rate_norm_s": 0.40,
        "max_filtered_log_scale_rate_s": 1.00,
        "max_raw_center_rate_norm_s": 0.51,
        "max_raw_log_scale_rate_s": 1.10,
        "max_raw_log_dimension_rate_s": 1.30,
        "complete_current_visibility_epoch": {
            "identity_basis": "complete_current_visibility_epoch_v1",
            "cross_gap_identity_claimed": False,
            "full_promotion_history_seal_required": True,
            "transition_tail_must_equal_credit_epoch_prefix": True,
            "sealed_postcredit_promotion_suffix_included": True,
            "exact_tracker_and_publication_progression": True,
            "min_frame_count": 5,
            "root_min_detection_confidence": 0.70,
            "root_min_association_confidence": 0.65,
            "pre_gap_history_sample_count": 3,
            "min_pre_gap_history_span_s": 0.060,
            "min_pre_gap_detection_confidence": 0.63,
            "min_pre_gap_association_confidence": 0.90,
            "max_epoch_start_missed_frames": 12,
            "max_epoch_start_publication_delta": 13,
            "max_epoch_start_unobserved_publications": 0,
            "max_epoch_start_gap_s": 0.450,
            "stable_tail_sample_count": 4,
            "min_stable_tail_span_s": 0.090,
            "min_stable_tail_detection_confidence": 0.65,
            "min_stable_tail_association_confidence": 0.90,
        },
        "reacquisition_bridge": {
            "exact_association_evidence_required": True,
            "pre_association_ambiguity_allowed": False,
            "current_association_ambiguity_allowed": False,
            "min_pretransition_visibility_sample_count": 5,
            "pre_gap_history_sample_count": 3,
            "min_pre_gap_history_span_s": 0.060,
            "min_pre_gap_detection_confidence": 0.64,
            "min_pre_gap_association_confidence": 0.90,
            "max_missed_frames": 11,
            "max_publication_delta": 12,
            "max_unobserved_publications": 1,
            "max_gap_s": 0.410,
            "min_association_confidence": 0.65,
            "min_detection_confidence": 0.70,
            "max_association_cost": 0.29,
            "tracker_max_assignment_cost": 0.82,
            "min_bbox_iou": 0.55,
            "min_direct_bbox_iou": 0.30,
            "max_center_residual_norm": 0.065,
            "max_abs_log_width_change": 0.32,
            "max_abs_log_height_change": 0.22,
            "max_abs_log_area_residual": 0.15,
            "max_center_rate_norm_s": 0.25,
            "max_log_scale_rate_s": 0.70,
            "required_clipping_continuity": 1.0,
        },
        "max_projection_horizon_s": 0.140,
        "max_actual_abs_center_norm": {
            "horizontal": 0.61,
            "vertical": 0.68,
        },
        "max_projected_abs_vertical_center_norm": 0.715,
        "min_projected_bbox_edge_margin_norm": {
            "horizontal": 6.0 / 640.0,
            "vertical": 6.0 / 360.0,
        },
        "forward_advance_allowed": False,
        "next_gate_blend": 0.0,
        "target_roll_rad": 0.0,
        "minimum_target_pitch_rad": 0.0,
        "max_roll_pitch_command_rate_rad_s": 0.12,
            "yaw_rate_rad_s": 0.15,
        "max_thrust": 0.285,
        "max_start_delay_after_credit_s": 0.06,
        "hard_duration_s": 0.16,
        "max_fresh_frames": 5,
        "max_commands": 8,
        "required_consecutive_strict_entry_frames": 2,
        "ordinary_entry_required_after_recovery": True,
    }
    assert phases["restricted_post_promotion_alignment"] == {
        "max_roll_pitch_command_rate_rad_s": 0.12,
            "yaw_rate_rad_s": 0.15,
        "max_thrust": 0.30,
        "hard_duration_s": 0.90,
    }


def test_jsonl_close_failure_still_invalidates_and_closes_replay():
    calls = []

    class BrokenHandle:
        def close(self):
            calls.append("jsonl-close")
            raise OSError("legacy close failed")

    class FakeReplay:
        def fail(self, reason):
            calls.append(("replay-fail", reason))

        def close(self, *, outcome, expected_decoded_frames):
            calls.append(("replay-close", outcome, expected_decoded_frames))
            return object()

    recorder = vq2_module.JsonlRecorder(None, replay=FakeReplay())
    recorder._handle = BrokenHandle()
    outcome = {"vision_capture_stats": {"frames_decoded": 7}}
    with pytest.raises(OSError, match="legacy close failed"):
        recorder.close(outcome=outcome)

    assert recorder._handle is None
    assert calls[0] == "jsonl-close"
    assert calls[1][0] == "replay-fail"
    assert "legacy close failed" in calls[1][1]
    assert calls[2] == ("replay-close", outcome, 7)


def test_jsonl_create_new_never_overwrites_existing_capture(tmp_path):
    path = tmp_path / "existing.jsonl.gz"
    path.write_bytes(b"forensic-bytes")

    with pytest.raises(FileExistsError):
        vq2_module.JsonlRecorder(str(path), create_new=True)

    assert path.read_bytes() == b"forensic-bytes"


def _calibration_received_set(*, generation=2, received_ns=990_000_000):
    def ingress(sequence, message_type, source_value, source_unit):
        return MavlinkIngressV1(
            stream_id="vq2-mavlink-udp-14550",
            generation=generation,
            sequence=sequence,
            message_type=message_type,
            host_clock_id="host-perf-counter",
            received_monotonic_ns=received_ns + sequence,
            source_time_value=source_value,
            source_time_unit=source_unit,
        )

    heartbeat = ReceivedHeartbeatV1(
        ingress=ingress(0, "HEARTBEAT", None, None),
        heartbeat=HeartbeatPayloadV1(base_mode=0x80, custom_mode=0),
    )
    race = ReceivedRaceStatusV1(
        ingress=ingress(1, "RACE_STATUS", 101, "ms"),
        race_status=RaceStatusPayloadV1(
            sim_boot_time_ms=101,
            race_start_boot_time_ms=50,
            race_finish_time_ns=-1,
            active_gate_index=0,
            last_gate_race_time=-1,
        ),
    )
    imu = ReceivedIMUSampleV1(
        ingress=ingress(2, "HIGHRES_IMU", 1_001, "us"),
        imu=IMUData(
            timestamp_us=1_001,
            accel=(0.0, 0.0, -9.81),
            gyro=(0.0, 0.0, 0.0),
        ),
    )
    actuator = ReceivedActuatorOutputStatusV1(
        ingress=ingress(3, "ACTUATOR_OUTPUT_STATUS", 1_002, "us"),
        actuator_output_status=ActuatorOutputStatusPayloadV1(
            time_usec=1_002,
            active=1,
            actuator=(0.0,) * 32,
        ),
    )
    return heartbeat, race, imu, actuator


def _calibration_safety_facts(*, checked_ns=1_000_000_000):
    heartbeat, race, imu, actuator = _calibration_received_set()
    timing = FrameTimingV1(
        identity=FrameIdentityV1("vq2-camera-udp-5600", 2, 9),
        camera_source_time_ns=500,
        host_clock_id="host-perf-counter",
        publication_sequence=9,
        first_unique_packet_monotonic_ns=989_000_000,
        final_unique_packet_monotonic_ns=990_000_000,
        reassembly_complete_monotonic_ns=990_100_000,
        decode_start_monotonic_ns=990_200_000,
        decode_end_monotonic_ns=990_300_000,
        publish_monotonic_ns=990_400_000,
    )
    return CalibrationSafetyFacts(
        checked_monotonic_ns=checked_ns,
        reset_epoch={
            "ingress_generation": 2,
            "race_anchor_boot_ms": 100,
            "imu_anchor_usec": 1_000,
        },
        frame={
            "stream_id": "vq2-camera-udp-5600",
            "generation": 2,
            "frame_id": 9,
            "sim_time_ns": 500,
            "timing": timing.to_primitive(),
            "width": 640,
            "height": 360,
        },
        imu=imu.to_primitive(),
        race=race.to_primitive(),
        heartbeat=heartbeat.to_primitive(),
        actuator=actuator.to_primitive(),
        imu_advance_monotonic_ns=990_000_002,
        race_advance_monotonic_ns=990_000_001,
        estimator_healthy=True,
        target_consecutive=3,
        target_center_px=(320.0, 180.0),
        target_bbox_px=(280.0, 140.0, 80.0, 80.0),
        initial_target_bbox_area_px=6_400.0,
        start_roll_rad=0.01,
        start_pitch_rad=-0.30,
        current_roll_rad=0.01,
        current_pitch_rad=-0.30,
        collision_count=0,
        capture_healthy=True,
        parent_alive=True,
        lease_valid=True,
    )


def test_calibration_safety_gate_uses_strict_lineage_and_closed_corridor():
    facts = _calibration_safety_facts()
    authorization = evaluate_calibration_safety(facts)

    assert authorization.watchdogs["result"] == "pass"
    assert authorization.watchdogs["gate_index"] == 0
    assert authorization.source["imu"]["ingress"]["generation"] == 2

    # Closed corridor edges are admitted exactly.
    evaluate_calibration_safety(
        replace(facts, target_center_px=(64.0, 324.0))
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"collision_count": 1}, "collision_observed"),
        ({"current_roll_rad": 0.0600001}, "attitude_excursion"),
        ({"target_center_px": (63.999, 180.0)}, "target_out_of_corridor"),
        ({"target_bbox_px": (280.0, 140.0, 161.0, 40.0)}, "target_too_large"),
    ],
)
def test_calibration_safety_gate_aborts_every_frozen_limit(change, reason):
    with pytest.raises(CalibrationCheckFailure) as caught:
        evaluate_calibration_safety(replace(_calibration_safety_facts(), **change))
    assert caught.value.reason_code == reason


def _scheduler_contract():
    return SimpleNamespace(
        ATTEMPT_ID=powered_contract.ATTEMPT_ID,
        SESSION_ID=powered_contract.SESSION_ID,
        HOST_CLOCK_ID=powered_contract.HOST_CLOCK_ID,
        EXCITATION_PLAN_ID=powered_contract.EXCITATION_PLAN_ID,
        EXCITATION_PLAN_SHA256=powered_contract.EXCITATION_PLAN_SHA256,
        validate_attempt=lambda value: dict(value),
        frozen_excitation_plan=powered_contract.frozen_excitation_plan,
        excitation_tick=powered_contract.excitation_tick,
        canonical_object_sha256=powered_contract.canonical_object_sha256,
        validate_calibration_command_generated=(
            powered_contract.validate_calibration_command_generated
        ),
        validate_calibration_command_sent=(
            powered_contract.validate_calibration_command_sent
        ),
        validate_calibration_command_not_sent=(
            powered_contract.validate_calibration_command_not_sent
        ),
        validate_calibration_tick_disposition=(
            powered_contract.validate_calibration_tick_disposition
        ),
        validate_phase_deadline_event=powered_contract.validate_phase_deadline_event,
    )


def _scheduler_attempt():
    return {
        "context": {"candidate_commit": "d" * 40},
        "context_sha256": "a" * 64,
    }


class _IntegerClock:
    def __init__(self, now=1_000_000_000):
        self.now = now

    def __call__(self):
        return self.now

    async def wait_until(self, deadline):
        assert deadline >= self.now
        self.now = deadline


def _returned_dispatch(command, clock, sequence):
    wire = AttitudeTargetWireV1(
        time_boot_ms=1,
        target_system=1,
        target_component=1,
        type_mask=128,
        q_wxyz=(1.0, 0.0, 0.0, 0.0),
        body_rates_rad_s=(
            -command.roll_rate,
            -command.pitch_rate,
            -command.yaw_rate,
        ),
        thrust=command.thrust,
    )
    receipt = AttitudeTargetOutboundV1(
        stream_id="vq2-mavlink-udp-14550",
        reset_generation=2,
        outbound_sequence=sequence,
        host_clock_id="host-perf-counter",
        call_start_monotonic_ns=clock.now,
        call_end_monotonic_ns=clock.now,
        api="send_attitude_rate",
        outcome="returned",
        error_type=None,
        wire=wire,
    )
    return CalibrationDispatchResult(
        audit_count_before=sequence,
        audit_count_after=sequence + 1,
        receipt=receipt.to_primitive(),
        call_started_monotonic_ns=clock.now,
        call_ended_monotonic_ns=clock.now,
    )


def test_calibration_scheduler_executes_exact_245_absolute_ticks_without_catchup():
    contract = _scheduler_contract()
    evidence = CalibrationCommandEvidence(
        attempt=_scheduler_attempt(), contract=contract
    )
    authorization = evaluate_calibration_safety(_calibration_safety_facts())
    clock = _IntegerClock()
    dispatched = []

    async def dispatch(command, deadline):
        assert clock.now < deadline
        sequence = len(dispatched)
        dispatched.append((clock.now, command))
        return _returned_dispatch(command, clock, sequence)

    scheduler = CalibrationExcitationScheduler(
        evidence=evidence,
        safety_check=lambda _tick: authorization,
        dispatch=dispatch,
        monotonic_ns=clock,
        wait_until_ns=clock.wait_until,
        contract=contract,
    )
    result = asyncio.run(scheduler.run())

    assert result.completed is True
    assert result.sent_ticks == tuple(range(245))
    assert result.skipped_before_generation == ()
    assert result.skipped_after_generation == ()
    assert [time for time, _command in dispatched] == [
        result.anchor_monotonic_ns + index * 20_000_000
        for index in range(245)
    ]
    assert dispatched[30][1].roll_rate == 0.08
    assert dispatched[54][1].roll_rate == -0.06
    assert dispatched[86][1].pitch_rate == 0.07
    assert dispatched[244][1].yaw_rate == 0.0
    dispositions = [
        row
        for event, row in evidence.observations
        if event == "calibration_tick_disposition"
    ]
    assert [row["absolute_tick"] for row in dispositions] == list(range(245))
    assert all(row["disposition"] == "sent" for row in dispositions)


def test_calibration_cleanup_zero_uses_exact_rich_generated_sent_chain():
    contract = _scheduler_contract()
    evidence = CalibrationCommandEvidence(
        attempt=_scheduler_attempt(), contract=contract
    )
    generated = evidence.record_cleanup_generated(
        checked_monotonic_ns=1_000_000_000,
        generated_monotonic_ns=1_000_000_001,
    )
    receipt = AttitudeTargetOutboundV1(
        stream_id="vq2-mavlink-udp-14550",
        reset_generation=2,
        outbound_sequence=7,
        host_clock_id="host-perf-counter",
        call_start_monotonic_ns=1_000_000_002,
        call_end_monotonic_ns=1_000_000_003,
        api="send_attitude_rate",
        outcome="returned",
        error_type=None,
        wire=AttitudeTargetWireV1(
            time_boot_ms=1,
            target_system=1,
            target_component=1,
            type_mask=128,
            q_wxyz=(1.0, 0.0, 0.0, 0.0),
            body_rates_rad_s=(0.0, 0.0, 0.0),
            thrust=0.0,
        ),
    ).to_primitive()
    sent = evidence.record_sent(
        generated,
        sent_monotonic_ns=1_000_000_004,
        dispatch=CalibrationDispatchResult(
            audit_count_before=7,
            audit_count_after=8,
            receipt=receipt,
            call_started_monotonic_ns=1_000_000_002,
            call_ended_monotonic_ns=1_000_000_003,
        ),
    )

    assert generated["scope"] == "cleanup_zero"
    assert generated["command"] == {
        "roll_rate_rad_s": 0.0,
        "pitch_rate_rad_s": 0.0,
        "yaw_rate_rad_s": 0.0,
        "thrust": 0.0,
    }
    assert all(value is None for value in generated["source"].values())
    assert sent["transport"]["receipt"] == receipt
    powered_contract.validate_calibration_command_sent(sent, generated=generated)


@pytest.mark.parametrize("failure", [RuntimeError("boom"), asyncio.CancelledError()])
@pytest.mark.parametrize("parent_dead", [False, True])
def test_calibration_lifecycle_enters_cleanup_and_publishes_after_early_exit(
    failure, parent_dead
):
    clock = _IntegerClock(now=1_000_000_000)
    published = []
    factories = []

    class Contract:
        TASK_ID = powered_contract.TASK_ID
        SESSION_ID = powered_contract.SESSION_ID
        ATTEMPT_ID = powered_contract.ATTEMPT_ID
        HOST_CLOCK_ID = powered_contract.HOST_CLOCK_ID
        EXCITATION_PLAN_ID = powered_contract.EXCITATION_PLAN_ID
        EXCITATION_PLAN_SHA256 = powered_contract.EXCITATION_PLAN_SHA256

        @staticmethod
        def validate_attempt(value):
            return dict(value)

        @staticmethod
        def validate_phase_deadline(value, **_kwargs):
            return dict(value)

        @staticmethod
        def validate_phase_deadline_event(value):
            return dict(value)

        @staticmethod
        def validate_cleanup_certificate(value):
            return dict(value)

        @staticmethod
        def validate_process_result(value, **_kwargs):
            return dict(value)

        @staticmethod
        def validate_outbound_audit(value):
            return dict(value)

        canonical_file_sha256 = staticmethod(powered_contract.canonical_file_sha256)

    class Boundary:
        def parent_signaled(self, _handle):
            return parent_dead

    class Lease:
        def prove_live_delegation(self, **_kwargs):
            raise AssertionError("lease proof must follow recorder preparation")

        def take_over_abandoned(self, **_kwargs):
            if not parent_dead:
                raise AssertionError("live parent cannot require takeover")
            clock.now += 1
            return CalibrationLeaseProof(
                owner_role="powered-child-parent-death",
                generation=1,
                record_sha256="9" * 64,
                authority_valid=True,
                takeover_completed_monotonic_ns=clock.now,
            )

        def release_takeover(self, *_args, **_kwargs):
            return True

    class Publisher:
        def publish_create_new(
            self, path, value, *, deadline_monotonic_ns, progress
        ):
            assert path.endswith("child-cleanup-certificate.json")
            assert clock.now < deadline_monotonic_ns
            progress()
            published.append(dict(value))
            return Contract.canonical_file_sha256(value)

    arguments = vq2_module.CalibrationArguments(
        stage="calibration-excite",
        powered_attempt_envelope="C:\\evidence\\attempt.json",
        wrapper_process="10:1010",
        powered_process_authority="C:\\evidence\\child-authority.json",
        attempt_capability_handle="41",
        parent_liveness_handle="42",
        record="C:\\evidence\\session.jsonl.gz",
        replay_bundle="C:\\evidence\\session.vq2replay",
        cleanup_certificate="C:\\evidence\\child-cleanup-certificate.json",
        recording_approved=True,
    )
    anchor = clock.now
    admission = CalibrationAdmission(
        arguments=arguments,
        attempt={
            "context": {
                "candidate_commit": "d" * 40,
                "target_config": {"sha256": "a" * 64},
                "deadline_durations_ns": dict(powered_contract.DEADLINE_DURATIONS_NS),
            },
            "context_sha256": "b" * 64,
        },
        live_freeze={},
        process_authority={
            "lease_record_sha256": "c" * 64,
            "absolute_deadlines": {"anchor": anchor},
        },
        current_process={},
        wrapper_process={},
        process_argv=("python.exe",),
        capability_handle=41,
        parent_handle=42,
        role_secret=bytearray(b"x" * 32),
        admitted_monotonic_ns=anchor,
        total_deadline_monotonic_ns=anchor + 110_000_000_000,
        prepower_deadline_monotonic_ns=anchor + 52_000_000_000,
        powered_deadline_monotonic_ns=anchor + 57_000_000_000,
        cleanup_deadline_monotonic_ns=anchor + 72_000_000_000,
        replay_close_deadline_monotonic_ns=anchor + 107_000_000_000,
        exit_deadline_monotonic_ns=anchor + 110_000_000_000,
        attempt_envelope_sha256="e" * 64,
        process_authority_sha256="f" * 64,
    )

    def fail_recorder(_admission):
        factories.append("recorder")
        raise failure

    services = CalibrationChildServices(
        process_boundary=Boundary(),
        lease_boundary=Lease(),
        recorder_factory=fail_recorder,
        adapter_factory=lambda **_kwargs: factories.append("adapter"),
        vision_factory=lambda **_kwargs: factories.append("vision"),
        camera_socket_factory=lambda *_args: None,
        publisher=Publisher(),
        monotonic_ns=clock,
        wait_until_ns=clock.wait_until,
        contract=Contract,
        runtime=powered_runtime,
    )

    output = asyncio.run(CalibrationChildLifecycle(admission, services).run())

    assert isinstance(output, CalibrationChildRunOutput)
    assert output.exit_code == 1
    assert factories == ["recorder"]
    assert len(published) == 1
    assert published[0]["trigger"] == (
        "parent_death" if parent_dead else "stage_abort"
    )
    expected_phases = ["cleanup"]
    if parent_dead:
        expected_phases.append("parent_death_lease_takeover")
    expected_phases.extend(["replay_close", "finalize"])
    assert [
        row["phase"] for row in output.process_result["phase_deadlines"]
    ] == expected_phases
    assert bytes(admission.role_secret) == b"\x00" * 32


class _LifecycleReplay:
    def __init__(self, trace):
        self.trace = trace
        self.closed = False
        self.frame_count = 0
        self.failures = []
        self.close_kwargs = None

    def record_event(self, *_args, **_kwargs):
        return True

    def record_mavlink_ingress(self, *_args, **_kwargs):
        return True

    def record_imu(self, *_args, **_kwargs):
        return True

    def record_race(self, *_args, **_kwargs):
        return True

    def record_command(self, *_args, **_kwargs):
        return True

    def capture_decoded_snapshot(self, _snapshot):
        self.frame_count += 1
        return True

    def fail(self, reason):
        self.failures.append(reason)

    def stats(self):
        return SimpleNamespace(
            enqueued=self.frame_count,
            written=self.frame_count,
            dropped=0,
            duplicate_frame_tokens=0,
            writer_errors=0,
            queue_high_watermark=1 if self.frame_count else 0,
            decoded_frames_enqueued=self.frame_count,
            decoded_frames_written=self.frame_count,
            decoded_frames_dropped=0,
            complete=self.closed,
            failure_reason=None if not self.failures else self.failures[0],
        )

    def close(self, **_kwargs):
        assert self.closed is False
        self.close_kwargs = dict(_kwargs)
        self.trace.append("replay_close")
        self.closed = True
        return SimpleNamespace(complete=True, dataset_hash="7" * 64)


class _LifecycleBoundary:
    def __init__(self, trace, *, death_point=None):
        self.trace = trace
        self.death_point = death_point
        self.signaled = False
        self.parent_checks = 0

    def parent_signaled(self, _handle):
        self.parent_checks += 1
        return self.signaled


class _OpaqueLifecycleLeaseProof:
    def __init__(
        self,
        *,
        owner_role,
        generation,
        record_sha256,
        authority_valid,
        takeover_completed_monotonic_ns=None,
    ):
        self.owner_role = owner_role
        self.generation = generation
        self.record_sha256 = record_sha256
        self.authority_valid = authority_valid
        self.takeover_completed_monotonic_ns = takeover_completed_monotonic_ns


class _LifecycleLease:
    def __init__(self, clock, boundary, trace):
        self.clock = clock
        self.boundary = boundary
        self.trace = trace
        self.live_proved = False
        self.released = False
        self.heartbeats = 0
        self.latest_proof = None

    def prove_live_delegation(self, **_kwargs):
        assert self.boundary.signaled is False
        self.trace.append("lease_live_proved")
        self.live_proved = True
        return _OpaqueLifecycleLeaseProof(
            owner_role="wrapper",
            generation=1,
            record_sha256="c" * 64,
            authority_valid=True,
        )

    def take_over_abandoned(self, **_kwargs):
        assert self.boundary.signaled is True
        self.trace.append("lease_takeover")
        self.clock.now += 1
        self.latest_proof = _OpaqueLifecycleLeaseProof(
            owner_role="powered-child-parent-death",
            generation=2,
            record_sha256="e" * 64,
            authority_valid=True,
            takeover_completed_monotonic_ns=self.clock.now,
        )
        return self.latest_proof

    def heartbeat_takeover(self, proof, **_kwargs):
        assert proof == self.latest_proof
        self.trace.append("lease_heartbeat")
        self.heartbeats += 1
        self.latest_proof = _OpaqueLifecycleLeaseProof(
            owner_role=proof.owner_role,
            generation=proof.generation + 1,
            record_sha256=f"{self.heartbeats % 16:x}" * 64,
            authority_valid=True,
            takeover_completed_monotonic_ns=proof.takeover_completed_monotonic_ns,
        )
        return self.latest_proof

    def release_takeover(self, proof, **_kwargs):
        assert proof == self.latest_proof
        self.trace.append("lease_release")
        self.released = True
        return True


class _LifecyclePublisher:
    def __init__(self, boundary, trace):
        self.boundary = boundary
        self.trace = trace
        self.values = []

    def publish_create_new(
        self, path, value, *, deadline_monotonic_ns, progress_callback
    ):
        assert path == powered_contract.frozen_paths()["child_cleanup_certificate"]
        progress_callback()
        self.trace.append("certificate_publish")
        if self.boundary.death_point == "during_certificate":
            self.boundary.signaled = True
            self.trace.append("parent_dead_during_certificate")
            progress_callback()
        self.values.append(dict(value))
        if self.boundary.death_point == "after_certificate":
            self.boundary.signaled = True
            self.trace.append("parent_dead_after_certificate")
        return powered_contract.canonical_file_sha256(value)


class _LifecycleDetector:
    def detect(self, _image):
        return [_detection(280, 140, 80, 80)]


class _LifecycleVision:
    def __init__(self, clock, trace, *, on_snapshot, bind, options):
        self.clock = clock
        self.trace = trace
        self.on_snapshot = on_snapshot
        self.bind = dict(bind)
        self.options = dict(options)
        self.generation = 0
        self.frame_id = 0
        self.publication_sequence = 0
        self.frames_decoded = 0
        self.queue = deque()
        self.is_running = False

    def reset(self):
        self.generation += 1
        self.frame_id = 0
        self.queue.clear()
        self.trace.append(f"vision_reset:{self.generation}")

    async def start(self):
        self.is_running = True
        self.trace.append(f"vision_start:{self.generation}")

    async def stop(self, **_kwargs):
        self.is_running = False
        self.trace.append(f"vision_stop:{self.generation}")

    def feed_cycle(self):
        if not self.is_running:
            return
        self.frame_id += 1
        self.publication_sequence += 1
        now = self.clock.now
        sim_time_ns = self.generation * 1_000_000_000 + self.frame_id * 20_000_000
        timing = FrameTimingV1(
            identity=FrameIdentityV1(
                "vq2-camera-udp-5600", self.generation, self.frame_id
            ),
            camera_source_time_ns=sim_time_ns,
            host_clock_id=powered_contract.HOST_CLOCK_ID,
            publication_sequence=self.publication_sequence,
            first_unique_packet_monotonic_ns=now,
            final_unique_packet_monotonic_ns=now,
            reassembly_complete_monotonic_ns=now,
            decode_start_monotonic_ns=now,
            decode_end_monotonic_ns=now,
            publish_monotonic_ns=now,
        )
        frame = CameraFrame(
            timestamp_us=sim_time_ns // 1_000,
            image=np.zeros((360, 640, 3), dtype=np.uint8),
            width=640,
            height=360,
        )
        snapshot = SimpleNamespace(
            frame_id=self.frame_id,
            sim_time_ns=sim_time_ns,
            received_monotonic_s=now / 1_000_000_000.0,
            generation=self.generation,
            timing=timing,
            camera_frame=frame,
        )
        assert self.on_snapshot(snapshot) is True
        self.frames_decoded += 1
        self.queue.append(snapshot)

    def pop_capture_snapshot(self):
        return self.queue.popleft() if self.queue else None

    def stats(self):
        return SimpleNamespace(
            datagrams_received=self.frames_decoded,
            unique_datagrams=self.frames_decoded,
            duplicate_datagrams=0,
            malformed_datagrams=0,
            frames_reassembled=self.frames_decoded,
            frames_decoded=self.frames_decoded,
            decode_failures=0,
            out_of_order_frame_drops=0,
            reset_generation_drops=0,
            processing_errors=0,
            socket_errors=0,
            resets=self.generation,
            remembered_chunk_keys=0,
            timing_ledger_entries=0,
            timing_ledger_high_watermark=1 if self.frames_decoded else 0,
            timing_ledger_capacity=4096,
            receiver_buffered_partial_frames=0,
            receiver_buffer_high_watermark=1 if self.frames_decoded else 0,
            receiver_buffer_capacity=64,
            capture_snapshot_queue_entries=len(self.queue),
            capture_snapshot_queue_high_watermark=1 if self.frames_decoded else 0,
            capture_snapshot_queue_capacity=256,
            capture_snapshot_queue_dropped=0,
            capture_snapshot_queue_enabled=True,
            receiver_dropped_partial_frames=0,
            receiver_duplicate_chunks=0,
            receiver_dropped_late_packets=0,
            snapshot_callback_errors=0,
            timing_overflow_latched=False,
        )

    def source_diagnostics(self):
        return SimpleNamespace(source_rejected_latched=False)


class _LifecycleAdapter:
    _STREAM = "vq2-mavlink-udp-14550"

    def __init__(
        self,
        clock,
        boundary,
        trace,
        *,
        outbound_guards,
        parent_alive,
        lease_valid,
        bind,
        failure_mode=None,
        boundary_collision=False,
        cleanup_boundary_collision=False,
    ):
        self.clock = clock
        self.boundary = boundary
        self.trace = trace
        self.powered_outbound_guards = outbound_guards
        self.parent_alive = parent_alive
        self.lease_valid = lease_valid
        self.bind = dict(bind)
        self.failure_mode = failure_mode
        self.boundary_collision = boundary_collision
        self.cleanup_boundary_collision = cleanup_boundary_collision
        self.enable_vision = False
        self.telemetry_mode = "imu"
        self.fetch_track_on_connect = False
        self.powered_source_promoted = False
        self.powered_source_rejected = False
        self.powered_peer = None
        self.vision = None
        self.connected = False
        self.armed = False
        self.generation = 1
        self.ingress_sequence = 0
        self.ingress_counts = {
            "HIGHRES_IMU": 0,
            "HEARTBEAT": 0,
            "RACE_STATUS": 0,
            "ACTUATOR_OUTPUT_STATUS": 0,
        }
        self.ingress_high_watermark = 0
        self.imu_high_watermark = 0
        self.other_high_watermark = 0
        self.outbound_sequence = 0
        self.race_ms = 700
        self.imu_us = 200_000
        self.drain_since_reset = None
        self.ingress_dropped = 0
        self.collision_handled = 0
        self.collision_dropped = 0
        self.receipts = deque()
        self.sent = []
        self.counts = {
            name: 0
            for name in (
                "timesync",
                "gcs_heartbeat",
                "sim_reset",
                "arm",
                "disarm",
                "attitude_target",
                "position_target",
                "other_command",
            )
        }
        self.returned = 0
        self.raised = 0
        self.guard_enable_count = 0
        self._cancelled_once = False
        self._post_arm_drains = None
        self.reset_persistence_failed = False

    def _authorize(self, category, deadline, *, cleanup, exact_zero=None):
        guard = self.powered_outbound_guards
        if cleanup:
            guard.authorize_cleanup(
                category,
                now_monotonic_ns=self.clock.now,
                deadline_monotonic_ns=deadline,
                parent_alive=self.parent_alive(),
                lease_valid=self.lease_valid(),
                source_promoted=self.powered_source_promoted,
                exact_zero=exact_zero,
            )
        else:
            guard.authorize_production(
                category,
                now_monotonic_ns=self.clock.now,
                deadline_monotonic_ns=deadline,
                role_valid=True,
                parent_alive=self.parent_alive(),
                lease_valid=self.lease_valid(),
                peer_frozen=self.powered_peer is not None,
                source_valid=not self.powered_source_rejected,
                source_promoted=self.powered_source_promoted,
            )

    def _receipt(self, category, wire, *, outcome="returned", error_type=None):
        common = dict(
            stream_id=self._STREAM,
            reset_generation=self.generation,
            outbound_sequence=self.outbound_sequence,
            host_clock_id=powered_contract.HOST_CLOCK_ID,
            call_start_monotonic_ns=self.clock.now,
            call_end_monotonic_ns=self.clock.now,
            outcome=outcome,
            error_type=error_type,
        )
        if category == "attitude_target":
            receipt = AttitudeTargetOutboundV1(
                **common,
                api="send_attitude_rate",
                wire=wire,
            )
        else:
            receipt = NonAttitudeOutboundV1(
                **common,
                category=category,
                api=(
                    "command_long_send"
                    if category in {"arm", "disarm", "sim_reset"}
                    else "timesync_send"
                    if category == "timesync"
                    else "heartbeat_send"
                ),
                wire=wire,
            )
        self.outbound_sequence += 1
        self.counts[category] += 1
        if outcome == "returned":
            self.returned += 1
        else:
            self.raised += 1
        self.receipts.append(receipt)
        self.sent.append(
            {
                "category": category,
                "cleanup": self.powered_outbound_guards.cleanup_state
                in {"enabled_live", "enabled_takeover"},
                "production_latched": self.powered_outbound_guards.production_latched,
                "authority_proved": self.lease_valid(),
                "outcome": outcome,
            }
        )
        return receipt

    async def connect(self, _address, *, deadline_monotonic_ns):
        assert self.powered_outbound_guards.production_latched is False
        self.powered_outbound_guards.enable_production()
        self.guard_enable_count += 1
        self.trace.append("guard_enable_production")
        self.connected = True
        self.powered_peer = ("127.0.0.1", 14551)
        self.powered_source_promoted = True
        self._authorize("timesync", deadline_monotonic_ns, cleanup=False)
        self._receipt("timesync", TimesyncWireV1(tc1=0, ts1=1))
        self._authorize("gcs_heartbeat", deadline_monotonic_ns, cleanup=False)
        self._receipt(
            "gcs_heartbeat",
            GCSHeartbeatWireV1(
                type=6,
                autopilot=8,
                base_mode=0,
                custom_mode=0,
                system_status=4,
            ),
        )
        self.trace.append("adapter_connect")

    def _ingress(self, message_type, source_value, source_unit):
        value = MavlinkIngressV1(
            stream_id=self._STREAM,
            generation=self.generation,
            sequence=self.ingress_sequence,
            message_type=message_type,
            host_clock_id=powered_contract.HOST_CLOCK_ID,
            received_monotonic_ns=self.clock.now,
            source_time_value=source_value,
            source_time_unit=source_unit,
        )
        self.ingress_sequence += 1
        self.ingress_counts[message_type] += 1
        return value

    def drain_received_observations(self):
        if self.failure_mode == "cancel_after_arm" and self._post_arm_drains is not None:
            self._post_arm_drains += 1
            if self._post_arm_drains == 2 and not self._cancelled_once:
                self._cancelled_once = True
                raise asyncio.CancelledError()
        if self.drain_since_reset is not None:
            self.drain_since_reset += 1
        self.race_ms += 50
        imu_samples = 60 if (
            self.drain_since_reset == 4
            or (self.drain_since_reset is None and self.imu_us == 200_000)
        ) else 1
        rows = [
            ReceivedHeartbeatV1(
                ingress=self._ingress("HEARTBEAT", None, None),
                heartbeat=HeartbeatPayloadV1(
                    base_mode=128 if self.armed else 0,
                    custom_mode=0,
                ),
            ),
            ReceivedRaceStatusV1(
                ingress=self._ingress("RACE_STATUS", self.race_ms, "ms"),
                race_status=RaceStatusPayloadV1(
                    sim_boot_time_ms=self.race_ms,
                    race_start_boot_time_ms=300,
                    race_finish_time_ns=0,
                    active_gate_index=0,
                    last_gate_race_time=0,
                ),
            ),
        ]
        for _index in range(imu_samples):
            self.imu_us += 10_000
            rows.append(
                ReceivedIMUSampleV1(
                    ingress=self._ingress("HIGHRES_IMU", self.imu_us, "us"),
                    imu=IMUData(
                        timestamp_us=self.imu_us,
                        accel=(0.0, 0.0, -9.80665),
                        gyro=(0.0, 0.0, 0.0),
                        mag=None,
                    ),
                )
            )
        rows.append(
            ReceivedActuatorOutputStatusV1(
                ingress=self._ingress(
                    "ACTUATOR_OUTPUT_STATUS", self.imu_us, "us"
                ),
                actuator_output_status=ActuatorOutputStatusPayloadV1(
                    time_usec=self.imu_us,
                    active=0,
                    actuator=(0.0,) * 32,
                ),
            )
        )
        self.imu_high_watermark = max(self.imu_high_watermark, imu_samples)
        self.other_high_watermark = max(self.other_high_watermark, 3)
        self.ingress_high_watermark = max(
            self.ingress_high_watermark,
            imu_samples + 3,
        )
        if self.vision is not None and self.vision.generation == self.generation:
            self.vision.feed_cycle()
        return rows

    def drain_collisions(self):
        return []

    def ingress_stats(self):
        return MavlinkIngressStats(
            generation=self.generation,
            next_sequence=self.ingress_sequence,
            highres_imu_received=self.ingress_counts["HIGHRES_IMU"],
            heartbeat_received=self.ingress_counts["HEARTBEAT"],
            race_status_received=self.ingress_counts["RACE_STATUS"],
            actuator_received=self.ingress_counts["ACTUATOR_OUTPUT_STATUS"],
            dropped=self.ingress_dropped,
            high_watermark=self.ingress_high_watermark,
            imu_capacity=4096,
            other_capacity=4096,
            imu_dropped=self.ingress_dropped,
            other_dropped=0,
            imu_high_watermark=self.imu_high_watermark,
            other_high_watermark=self.other_high_watermark,
            buffered_imu=0,
            buffered_other=0,
        )

    def collision_stats(self):
        return MavlinkCollisionStats(
            generation=self.generation,
            handled=self.collision_handled,
            dropped=self.collision_dropped,
            high_watermark=0,
            capacity=128,
            buffered=0,
        )

    def drain_outbound_receipts(self):
        values = list(self.receipts)
        self.receipts.clear()
        return values

    def outbound_audit(self):
        return SimpleNamespace(**self.counts)

    def outbound_receipt_stats(self):
        return SimpleNamespace(
            generation=self.generation,
            next_sequence=self.outbound_sequence,
            returned=self.returned,
            raised=self.raised,
            dropped=0,
            high_watermark=max(self.returned + self.raised, 1),
            capacity=4096,
            buffered=len(self.receipts),
        )

    async def arm(self, *, powered_deadline_monotonic_ns, powered_cleanup):
        assert powered_cleanup is False
        self._authorize("arm", powered_deadline_monotonic_ns, cleanup=False)
        self._receipt(
            "arm",
            CommandLongWireV1(1, 1, 400, 0, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        self.armed = True
        self._post_arm_drains = 0
        self.trace.append("arm_send")

    async def disarm(self, *, powered_deadline_monotonic_ns, powered_cleanup):
        if (
            powered_cleanup
            and self.boundary.death_point == "during_disarm_authorization"
        ):
            self.boundary.signaled = True
            self.boundary.death_point = None
            self.trace.append("parent_dead_during_cleanup_disarm_authorization")
        self._authorize(
            "disarm", powered_deadline_monotonic_ns, cleanup=powered_cleanup
        )
        self._receipt(
            "disarm",
            CommandLongWireV1(1, 1, 400, 0, (0.0,) * 7),
        )
        self.armed = False
        self.trace.append("cleanup_disarm" if powered_cleanup else "normalize_disarm")

    async def reset_calibration_with_boundary(
        self,
        persist_boundary,
        *,
        powered_deadline_monotonic_ns,
        powered_cleanup,
        powered_progress=None,
    ):
        old_generation = self.generation
        collisions = ()
        collision_stats = MavlinkCollisionStats(
            generation=old_generation,
            handled=0,
            dropped=0,
            high_watermark=0,
            capacity=128,
            buffered=0,
        )
        if (
            (self.boundary_collision and not powered_cleanup)
            or (self.cleanup_boundary_collision and powered_cleanup)
        ):
            collisions = (CalibrationCollisionV1(9, 1, 0.1),)
            collision_stats = MavlinkCollisionStats(
                generation=old_generation,
                handled=1,
                dropped=0,
                high_watermark=1,
                capacity=128,
                buffered=1,
            )
            if powered_cleanup:
                self.cleanup_boundary_collision = False
            else:
                self.boundary_collision = False
        boundary = CalibrationResetBoundaryV1(
            old_generation=old_generation,
            new_generation=old_generation + 1,
            boundary_monotonic_ns=self.clock.now,
            observations=(),
            collisions=collisions,
            ingress_stats=MavlinkIngressStats(
                generation=old_generation,
                next_sequence=self.ingress_sequence,
                highres_imu_received=self.ingress_counts["HIGHRES_IMU"],
                heartbeat_received=self.ingress_counts["HEARTBEAT"],
                race_status_received=self.ingress_counts["RACE_STATUS"],
                actuator_received=self.ingress_counts["ACTUATOR_OUTPUT_STATUS"],
                dropped=0,
                high_watermark=self.ingress_high_watermark,
                imu_capacity=4096,
                other_capacity=4096,
                imu_dropped=0,
                other_dropped=0,
                imu_high_watermark=self.imu_high_watermark,
                other_high_watermark=self.other_high_watermark,
                buffered_imu=0,
                buffered_other=0,
            ),
            collision_stats=collision_stats,
        )
        self.generation += 1
        self.ingress_sequence = 0
        self.ingress_counts = {
            "HIGHRES_IMU": 0,
            "HEARTBEAT": 0,
            "RACE_STATUS": 0,
            "ACTUATOR_OUTPUT_STATUS": 0,
        }
        self.ingress_high_watermark = 0
        self.imu_high_watermark = 0
        self.other_high_watermark = 0
        self.race_ms = 0
        self.imu_us = 0
        self.drain_since_reset = 0
        self.collision_handled = 0
        self.collision_dropped = 0
        if powered_cleanup and self.failure_mode == "cleanup_persistence_failed":
            self.reset_persistence_failed = True
        else:
            persist_boundary(boundary)
        if powered_progress is not None:
            if self.boundary.death_point == "during_reset_progress":
                self.boundary.signaled = True
                self.boundary.death_point = None
                self.trace.append("parent_dead_during_cleanup_reset_progress")
            powered_progress()
        self._authorize(
            "sim_reset",
            powered_deadline_monotonic_ns,
            cleanup=powered_cleanup,
        )
        self._receipt(
            "sim_reset",
            CommandLongWireV1(1, 1, 31000, 0, (0.0,) * 7),
        )
        self.trace.append("cleanup_reset" if powered_cleanup else "initial_reset")
        return boundary

    def calibration_reset_persistence_state(self):
        return SimpleNamespace(
            failure_latched=self.reset_persistence_failed,
            failures=(() if not self.reset_persistence_failed else (object(),)),
            dropped=0,
        )

    async def send_attitude_rate(
        self,
        command,
        *,
        powered_deadline_monotonic_ns,
        powered_cleanup,
    ):
        exact_zero = bool(
            command.roll_rate == 0.0
            and command.pitch_rate == 0.0
            and command.yaw_rate == 0.0
            and command.thrust == 0.0
        )
        if (
            powered_cleanup
            and self.boundary.death_point == "during_zero_authorization"
        ):
            self.boundary.signaled = True
            self.boundary.death_point = None
            self.trace.append("parent_dead_during_cleanup_zero_authorization")
        self._authorize(
            "attitude_target",
            powered_deadline_monotonic_ns,
            cleanup=powered_cleanup,
            exact_zero=exact_zero if powered_cleanup else None,
        )
        wire = AttitudeTargetWireV1(
            time_boot_ms=max(0, self.race_ms),
            target_system=1,
            target_component=1,
            type_mask=128,
            q_wxyz=(1.0, 0.0, 0.0, 0.0),
            body_rates_rad_s=(
                -command.roll_rate,
                -command.pitch_rate,
                -command.yaw_rate,
            ),
            thrust=command.thrust,
        )
        if self.failure_mode == "raise_first_attitude" and not powered_cleanup:
            self.failure_mode = None
            self._receipt(
                "attitude_target",
                wire,
                outcome="raised",
                error_type="RuntimeError",
            )
            raise RuntimeError("injected attitude failure")
        self._receipt("attitude_target", wire)
        if powered_cleanup:
            self.trace.append("cleanup_zero")
            if self.boundary.death_point == "during_cleanup":
                self.boundary.signaled = True
                self.trace.append("parent_dead_during_cleanup")

    async def disconnect(self, *, deadline_monotonic_ns):
        assert self.clock.now < deadline_monotonic_ns
        self.connected = False
        self.trace.append("adapter_disconnect")


def _lifecycle_harness(
    *,
    death_point=None,
    failure_mode=None,
    boundary_collision=False,
    cleanup_boundary_collision=False,
):
    trace = []
    clock = _IntegerClock(now=1_000_000_000)
    boundary = _LifecycleBoundary(trace, death_point=death_point)
    lease = _LifecycleLease(clock, boundary, trace)
    publisher = _LifecyclePublisher(boundary, trace)
    replay = _LifecycleReplay(trace)
    freeze = powered_contract.validate_live_freeze(powered_fixtures.live_freeze())
    attempt = powered_contract.validate_attempt(
        powered_fixtures.attempt(), live_freeze=freeze
    )
    paths = powered_contract.frozen_paths()
    arguments = CalibrationArguments(
        stage="calibration-excite",
        powered_attempt_envelope=paths["attempt_envelope"],
        wrapper_process="10:1010",
        powered_process_authority=paths["child_authority"],
        attempt_capability_handle="41",
        parent_liveness_handle="42",
        record=paths["legacy_record"],
        replay_bundle=paths["replay_bundle"],
        cleanup_certificate=paths["child_cleanup_certificate"],
        recording_approved=True,
    )
    anchor = clock.now
    current_process = powered_fixtures.process(20)
    wrapper_process = powered_fixtures.process(10)
    process_authority = {
        "lease_record_sha256": "c" * 64,
        "absolute_deadlines": {
            "anchor": anchor,
            "total": anchor + 110_000_000_000,
            "prepower": anchor + 52_000_000_000,
            "powered": anchor + 57_000_000_000,
            "cleanup": anchor + 72_000_000_000,
            "replay_close": anchor + 107_000_000_000,
            "exit": anchor + 110_000_000_000,
        },
    }
    admission = CalibrationAdmission(
        arguments=arguments,
        attempt=attempt,
        live_freeze=freeze,
        process_authority=process_authority,
        current_process=current_process,
        wrapper_process=wrapper_process,
        process_argv=tuple(attempt["context"]["child_argv"]),
        capability_handle=41,
        parent_handle=42,
        role_secret=bytearray(b"s" * 32),
        admitted_monotonic_ns=anchor,
        total_deadline_monotonic_ns=anchor + 110_000_000_000,
        prepower_deadline_monotonic_ns=anchor + 52_000_000_000,
        powered_deadline_monotonic_ns=anchor + 57_000_000_000,
        cleanup_deadline_monotonic_ns=anchor + 72_000_000_000,
        replay_close_deadline_monotonic_ns=anchor + 107_000_000_000,
        exit_deadline_monotonic_ns=anchor + 110_000_000_000,
        attempt_envelope_sha256=powered_contract.canonical_file_sha256(attempt),
        process_authority_sha256="f" * 64,
    )
    made = SimpleNamespace(adapter=None, vision=None, recorder=None)

    def recorder_factory(_admission):
        trace.append("recorder_prepare")
        made.recorder = JsonlRecorder(
            None,
            replay=replay,
            capture_fifo_enabled=True,
        )
        return made.recorder

    def adapter_factory(**kwargs):
        assert lease.live_proved is True
        assert kwargs["bind"] == freeze["transport"]["mavlink_bind"]
        made.adapter = _LifecycleAdapter(
            clock,
            boundary,
            trace,
            outbound_guards=kwargs["outbound_guards"],
            parent_alive=kwargs["parent_alive"],
            lease_valid=kwargs["lease_valid"],
            bind=kwargs["bind"],
            failure_mode=failure_mode,
            boundary_collision=boundary_collision,
            cleanup_boundary_collision=cleanup_boundary_collision,
        )
        trace.append("adapter_factory")
        return made.adapter

    def vision_factory(**kwargs):
        assert kwargs["bind"] == freeze["transport"]["camera_bind"]
        assert kwargs["powered_exclusive"] is True
        assert kwargs["capture_snapshot_queue_enabled"] is True
        assert callable(kwargs["exclusive_socket_factory"])
        made.vision = _LifecycleVision(
            clock,
            trace,
            on_snapshot=kwargs["on_snapshot"],
            bind=kwargs["bind"],
            options=kwargs,
        )
        made.adapter.vision = made.vision
        trace.append("vision_factory")
        return made.vision

    def endpoint_evidence(adapter, vision, endpoint_admission):
        assert adapter.connected is False
        assert vision.is_running is False
        owner = endpoint_admission.current_process

        def endpoint(role, bind, port):
            return {
                "state": "closed_with_peer",
                "bind": {
                    "role": role,
                    "family": "AF_INET",
                    "requested": {"host": bind["host"], "port": bind["port"]},
                    "actual": {"host": bind["host"], "port": bind["port"]},
                    "socket_policy": bind["socket_policy"],
                    "owner_process": owner,
                },
                "frozen_peer": {"host": "127.0.0.1", "port": port},
                "rejected_source_count": 0,
            }

        return {
            "mavlink": endpoint("mavlink", adapter.bind, 14551),
            "camera": endpoint("camera", vision.bind, 5601),
        }

    def transport_evidence(adapter, vision, guards):
        assert adapter.connected is False
        assert vision.is_running is False
        assert guards.production_latched is True
        assert guards.cleanup_state == "closed"
        return {
            "production_guard_latched": True,
            "cleanup_guard_closed": True,
            "vision_closed": True,
            "mavlink_socket_closed": True,
            "receiver_joined": True,
            "announcer_joined": True,
            "owned_handles_closed": True,
        }

    def artifact_closer(recorder, artifact_admission, outcome, deadline):
        assert recorder is made.recorder
        assert made.adapter.connected is False
        assert made.vision.is_running is False
        assert made.adapter.powered_outbound_guards.cleanup_state == "closed"
        assert clock.now < deadline
        recorder.close(outcome=dict(outcome), timeout_s=1.0)
        if boundary.death_point == "after_replay":
            boundary.signaled = True
            trace.append("parent_dead_after_replay")
        return CalibrationClosedArtifacts(
            legacy_record={
                "path": artifact_admission.arguments.record,
                "state": "closed",
                "sha256": "8" * 64,
            },
            replay_bundle={
                "path": artifact_admission.arguments.replay_bundle,
                "state": "closed",
                "dataset_hash": "7" * 64,
                "manifest_sha256": "9" * 64,
                "records_sha256": "a" * 64,
            },
        )

    services = CalibrationChildServices(
        process_boundary=boundary,
        lease_boundary=lease,
        recorder_factory=recorder_factory,
        adapter_factory=adapter_factory,
        vision_factory=vision_factory,
        camera_socket_factory=lambda _host, _port: object(),
        publisher=publisher,
        monotonic_ns=clock,
        wait_until_ns=clock.wait_until,
        detector_factory=_LifecycleDetector,
        endpoint_evidence=endpoint_evidence,
        transport_evidence=transport_evidence,
        artifact_closer=artifact_closer,
        contract=powered_contract,
        runtime=powered_runtime,
    )
    return SimpleNamespace(
        admission=admission,
        services=services,
        trace=trace,
        boundary=boundary,
        lease=lease,
        publisher=publisher,
        replay=replay,
        made=made,
        clock=clock,
    )


def _temporary_lifecycle_certificate_path(harness, path):
    """Use a disposable certificate target while retaining real schema checks."""

    harness.admission.arguments = replace(
        harness.admission.arguments,
        cleanup_certificate=str(path.resolve()),
    )

    class ContractProxy:
        def __getattr__(self, name):
            return getattr(powered_contract, name)

        def validate_process_result(
            self,
            value,
            *,
            cleanup_certificate=None,
        ):
            checked = json.loads(json.dumps(value, allow_nan=False))
            checked["cleanup_certificate"]["path"] = powered_contract.frozen_paths()[
                "child_cleanup_certificate"
            ]
            powered_contract.validate_process_result(
                checked,
                cleanup_certificate=cleanup_certificate,
            )
            return json.loads(json.dumps(value, allow_nan=False))

    harness.services.contract = ContractProxy()
    return harness.admission.arguments.cleanup_certificate


def test_calibration_lifecycle_full_normal_trace_validates_real_contract():
    harness = _lifecycle_harness()
    lifecycle = CalibrationChildLifecycle(harness.admission, harness.services)

    output = asyncio.run(lifecycle.run())

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    process_result = powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 0
    assert certificate["outcome"] == "proved"
    assert process_result["outcome"] == "completed"
    assert [row["phase"] for row in process_result["phase_deadlines"]] == [
        "connect",
        "preflight",
        "reset_epoch",
        "normalize_disarmed",
        "countdown_go",
        "arm",
        "powered_stage",
        "cleanup",
        "replay_close",
        "finalize",
    ]
    sent_ticks = [
        row
        for event, row in lifecycle.evidence.observations
        if event == "calibration_tick_disposition"
    ]
    assert len(sent_ticks) == 245
    assert all(row["disposition"] == "sent" for row in sent_ticks)
    assert process_result["outbound_audit"]["attitude_target"] == 246
    assert harness.made.adapter.guard_enable_count == 1
    assert harness.trace.index("certificate_publish") < harness.trace.index(
        "replay_close"
    )
    assert len(harness.publisher.values) == 1
    assert harness.replay.closed is True
    assert bytes(harness.admission.role_secret) == b"\x00" * 32
    nonannouncements = [
        row
        for row in harness.made.adapter.sent
        if row["category"] not in {"timesync", "gcs_heartbeat"}
    ]
    assert harness.trace.index("lease_live_proved") < harness.trace.index(
        "adapter_factory"
    )
    assert nonannouncements
    assert all(row["authority_proved"] is True for row in nonannouncements)
    assert all(
        row["cleanup"] or row["production_latched"] is False
        for row in nonannouncements
    )
    first_cleanup = next(
        index
        for index, row in enumerate(harness.made.adapter.sent)
        if row["cleanup"]
    )
    assert all(
        row["cleanup"]
        for row in harness.made.adapter.sent[first_cleanup:]
    )


def test_calibration_partial_vision_construction_retains_adapter_close_path():
    harness = _lifecycle_harness()

    def fail_vision(**_kwargs):
        raise RuntimeError("injected vision construction failure")

    harness.services.vision_factory = fail_vision
    lifecycle = CalibrationChildLifecycle(harness.admission, harness.services)
    lifecycle._prepare_recorder()
    lifecycle._prove_live_delegation()

    with pytest.raises(RuntimeError, match="vision construction"):
        lifecycle._construct_transports()

    assert lifecycle.adapter is harness.made.adapter
    assert lifecycle.dispatcher is not None
    asyncio.run(
        lifecycle._close_transports(
            harness.admission.cleanup_deadline_monotonic_ns
        )
    )
    assert harness.trace.count("adapter_disconnect") == 1


def test_calibration_manifest_seals_exact_post_transport_resource_stats():
    harness = _lifecycle_harness()

    output = asyncio.run(
        CalibrationChildLifecycle(harness.admission, harness.services).run()
    )

    assert output.exit_code == 0
    outcome = harness.replay.close_kwargs["outcome"]
    stats = outcome["powered_capture_resource_stats"]
    assert set(stats) == {
        "schema",
        "recorder",
        "vision",
        "ingress",
        "collision",
        "outbound_receipts",
        "snapshot_capture",
    }
    assert stats["schema"] == "aigp-vq2-powered-capture-resource-stats/1"
    assert stats["recorder"]["constructed"] is True
    assert stats["recorder"]["complete"] is False
    assert stats["recorder"]["dropped"] == 0
    assert stats["recorder"]["decoded_frames_dropped"] == 0
    assert stats["recorder"]["writer_errors"] == 0
    assert stats["vision"]["capture_snapshot_queue_entries"] == 0
    assert stats["vision"]["capture_snapshot_queue_dropped"] == 0
    assert stats["vision"]["receiver_buffered_partial_frames"] == 0
    assert stats["ingress"]["buffered_imu"] == 0
    assert stats["ingress"]["buffered_other"] == 0
    assert stats["collision"]["buffered"] == 0
    assert stats["outbound_receipts"]["buffered"] == 0
    assert stats["snapshot_capture"] == {
        "constructed": True,
        "observed_frames": stats["vision"]["frames_decoded"],
        "dimensions_admitted": True,
        "failure_latched": False,
    }


@pytest.mark.parametrize(
    "failure_mode",
    ["raise_first_attitude", "cancel_after_arm"],
)
def test_calibration_lifecycle_post_authority_failure_still_proves_cleanup(
    failure_mode,
):
    harness = _lifecycle_harness(failure_mode=failure_mode)
    lifecycle = CalibrationChildLifecycle(harness.admission, harness.services)

    output = asyncio.run(lifecycle.run())

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 1
    assert certificate["outcome"] == "proved"
    assert "arm_send" in harness.trace
    assert "cleanup_zero" in harness.trace
    assert "cleanup_disarm" in harness.trace
    assert "cleanup_reset" in harness.trace
    assert harness.trace.index("certificate_publish") < harness.trace.index(
        "replay_close"
    )
    assert sum(row["category"] == "arm" for row in harness.made.adapter.sent) == 1
    cleanup_start = next(
        index
        for index, row in enumerate(harness.made.adapter.sent)
        if row["cleanup"]
    )
    assert all(row["cleanup"] for row in harness.made.adapter.sent[cleanup_start:])
    assert bytes(harness.admission.role_secret) == b"\x00" * 32


def test_calibration_cleanup_reset_send_survives_persistence_callback_failure():
    harness = _lifecycle_harness(failure_mode="cleanup_persistence_failed")
    lifecycle = CalibrationChildLifecycle(harness.admission, harness.services)

    output = asyncio.run(lifecycle.run())

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 1
    assert "capture_incomplete" in output.process_result["reason_codes"]
    assert certificate["reset"]["state"] == "confirmed"
    assert certificate["reset"]["boundary"] is not None
    assert harness.trace.count("cleanup_reset") == 1


def test_calibration_cleanup_persistence_failure_retains_collision_certificate():
    harness = _lifecycle_harness(
        failure_mode="cleanup_persistence_failed",
        cleanup_boundary_collision=True,
    )

    output = asyncio.run(
        CalibrationChildLifecycle(harness.admission, harness.services).run()
    )

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    boundary_rows = certificate["reset"]["boundary"]["collisions"]
    complete_rows = certificate["collisions"]["observations"]
    assert output.exit_code == 1
    assert "capture_incomplete" in output.process_result["reason_codes"]
    assert certificate["reset"]["state"] == "confirmed"
    assert certificate["collection_invalidating_codes"] == ["collision_observed"]
    assert len(boundary_rows) == 1
    assert complete_rows.count(boundary_rows[0]) == 1
    assert len(harness.publisher.values) == 1
    assert harness.trace.count("cleanup_reset") == 1
    assert harness.made.adapter.counts["sim_reset"] == 2


@pytest.mark.parametrize(
    "death_point",
    ["during_cleanup", "after_certificate", "after_replay"],
)
def test_calibration_lifecycle_parent_death_takeover_is_ordered_and_single_cleanup(
    death_point,
):
    harness = _lifecycle_harness(death_point=death_point)
    lifecycle = CalibrationChildLifecycle(harness.admission, harness.services)

    output = asyncio.run(lifecycle.run())

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    process_result = powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 1
    assert process_result["outcome"] == "failed"
    assert "wrapper_death" in process_result["reason_codes"]
    assert harness.trace.count("lease_takeover") == 1
    assert harness.trace.count("lease_release") == 1
    assert harness.trace.count("cleanup_zero") == 1
    assert harness.trace.count("cleanup_disarm") == 1
    assert harness.trace.count("cleanup_reset") == 1
    phases = [row["phase"] for row in process_result["phase_deadlines"]]
    assert phases.count("parent_death_lease_takeover") == 1
    if death_point in {"during_cleanup", "after_certificate"}:
        assert harness.trace.index("certificate_publish") < harness.trace.index(
            "lease_release"
        )
        assert harness.trace.index("lease_release") < harness.trace.index(
            "replay_close"
        )
    else:
        assert harness.trace.index("replay_close") < harness.trace.index(
            "lease_takeover"
        )
        assert harness.trace.index("lease_takeover") < harness.trace.index(
            "lease_release"
        )
    if death_point == "during_cleanup":
        assert certificate["parent_state"]["mode"] == "signaled_takeover"
        assert certificate["trigger"] == "parent_death"
    else:
        assert certificate["parent_state"]["mode"] == "live_delegation"


@pytest.mark.parametrize(
    ("death_window", "death_marker", "next_send"),
    [
        (
            "cleanup_live_proof",
            "parent_dead_after_cleanup_live_proof",
            "cleanup_zero",
        ),
        (
            "disarm_heartbeat_wait",
            "parent_dead_after_cleanup_disarm_wait",
            "cleanup_disarm",
        ),
        (
            "reset_baseline_wait",
            "parent_dead_after_cleanup_reset_baseline",
            "cleanup_reset",
        ),
        (
            "reset_same_call_progress",
            "parent_dead_during_cleanup_reset_progress",
            "cleanup_reset",
        ),
        (
            "zero_same_call_authorization",
            "parent_dead_during_cleanup_zero_authorization",
            "cleanup_zero",
        ),
        (
            "disarm_same_call_authorization",
            "parent_dead_during_cleanup_disarm_authorization",
            "cleanup_disarm",
        ),
    ],
)
def test_calibration_cleanup_death_toctou_takes_over_before_next_mandatory_send(
    death_window,
    death_marker,
    next_send,
):
    harness = _lifecycle_harness(
        death_point={
            "reset_same_call_progress": "during_reset_progress",
            "zero_same_call_authorization": "during_zero_authorization",
            "disarm_same_call_authorization": "during_disarm_authorization",
        }.get(death_window)
    )

    class DeathWindowLifecycle(CalibrationChildLifecycle):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.injected = False

        async def _wait_for_heartbeat(self, deadline_ns, **kwargs):
            result = await super()._wait_for_heartbeat(deadline_ns, **kwargs)
            if (
                death_window == "disarm_heartbeat_wait"
                and kwargs["phase"] == "cleanup"
                and not self.injected
            ):
                self.injected = True
                harness.boundary.signaled = True
                harness.trace.append(death_marker)
            return result

        async def _wait_reset_baseline(self, deadline_ns, **kwargs):
            result = await super()._wait_reset_baseline(deadline_ns, **kwargs)
            if (
                death_window == "reset_baseline_wait"
                and kwargs["phase"] == "cleanup"
                and not self.injected
            ):
                self.injected = True
                harness.boundary.signaled = True
                harness.trace.append(death_marker)
            return result

    if death_window == "cleanup_live_proof":
        original_prove = harness.lease.prove_live_delegation
        proof_calls = 0

        def prove_and_signal(**kwargs):
            nonlocal proof_calls
            proof_calls += 1
            proof = original_prove(**kwargs)
            if proof_calls == 2:
                harness.boundary.signaled = True
                harness.trace.append(death_marker)
            return proof

        harness.lease.prove_live_delegation = prove_and_signal

    lifecycle = DeathWindowLifecycle(harness.admission, harness.services)
    output = asyncio.run(lifecycle.run())

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 1
    assert "wrapper_death" in output.process_result["reason_codes"]
    assert certificate["trigger"] == "parent_death"
    assert harness.trace.count("lease_takeover") == 1
    assert harness.trace.count("lease_release") == 1
    assert harness.trace.index(death_marker) < harness.trace.index("lease_takeover")
    assert harness.trace.index("lease_takeover") < harness.trace.index(next_send)
    assert harness.trace.count("cleanup_zero") == 1
    assert harness.trace.count("cleanup_disarm") == 1
    assert harness.trace.count("cleanup_reset") == 1


def test_calibration_takeover_heartbeats_use_latest_proof_until_release():
    harness = _lifecycle_harness(death_point="during_cleanup")

    class LongCleanupLifecycle(CalibrationChildLifecycle):
        async def _wait_final_state(self, deadline_ns):
            await super()._wait_final_state(deadline_ns)
            for _index in range(22):
                await self._poll_pause(deadline_ns)

    output = asyncio.run(
        LongCleanupLifecycle(harness.admission, harness.services).run()
    )

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 1
    assert harness.lease.heartbeats >= 1
    assert harness.trace.index("lease_takeover") < harness.trace.index(
        "lease_heartbeat"
    )
    assert harness.trace.index("lease_heartbeat") < harness.trace.index(
        "lease_release"
    )
    assert harness.lease.released is True


def test_calibration_replay_close_observes_death_without_holding_takeover_mutex():
    harness = _lifecycle_harness()
    normal_closer = harness.services.artifact_closer
    observed = {}

    def blocking_closer(*args, **kwargs):
        result = normal_closer(*args, **kwargs)
        harness.boundary.signaled = True
        harness.trace.append("parent_dead_inside_replay_closer")
        observed["checks_before"] = harness.boundary.parent_checks
        threading.Event().wait(0.15)
        observed["checks_after"] = harness.boundary.parent_checks
        harness.trace.append("replay_closer_return")
        return result

    harness.services.artifact_closer = blocking_closer
    output = asyncio.run(
        CalibrationChildLifecycle(harness.admission, harness.services).run()
    )

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 1
    assert "wrapper_death" in output.process_result["reason_codes"]
    assert observed["checks_after"] > observed["checks_before"]
    assert harness.trace.index("parent_dead_inside_replay_closer") < harness.trace.index(
        "replay_closer_return"
    )
    assert harness.trace.index("replay_closer_return") < harness.trace.index(
        "lease_takeover"
    )
    assert harness.trace.count("lease_takeover") == 1
    assert harness.trace.count("lease_release") == 1
    assert harness.trace.count("cleanup_zero") == 1
    assert harness.trace.count("cleanup_disarm") == 1
    assert harness.trace.count("cleanup_reset") == 1


def test_calibration_parent_death_during_certificate_publication_fails_closed(
    tmp_path,
):
    harness = _lifecycle_harness()
    target = _temporary_lifecycle_certificate_path(
        harness,
        tmp_path / "cleanup-certificate.json",
    )

    class MidPublicationDeath:
        def publish_create_new(
            self,
            path,
            value,
            *,
            deadline_monotonic_ns,
            progress,
        ):
            assert path == target
            progress()
            harness.trace.append("certificate_publish")
            harness.boundary.signaled = True
            harness.trace.append("parent_dead_during_certificate")
            progress()
            return powered_contract.canonical_file_sha256(value)

    harness.services.publisher = MidPublicationDeath()
    output = asyncio.run(
        CalibrationChildLifecycle(harness.admission, harness.services).run()
    )

    harness.services.contract.validate_process_result(
        output.process_result,
        cleanup_certificate=None,
    )
    assert output.exit_code == 1
    assert output.certificate is None
    assert output.process_result["cleanup_certificate"] == {
        "path": target,
        "state": "absent",
        "sha256": None,
    }
    assert "wrapper_death" in output.process_result["reason_codes"]
    assert "cleanup_unconfirmed" in output.process_result["reason_codes"]
    assert harness.trace.count("lease_takeover") == 1
    assert harness.trace.count("lease_release") == 1
    assert harness.trace.index("certificate_publish") < harness.trace.index(
        "lease_takeover"
    )
    assert harness.trace.index("lease_release") < harness.trace.index(
        "replay_close"
    )


@pytest.mark.parametrize("publication_failure", ["partial", "late"])
def test_calibration_failed_certificate_publication_preserves_invalid_file_hash(
    tmp_path,
    publication_failure,
):
    harness = _lifecycle_harness()
    target = _temporary_lifecycle_certificate_path(
        harness,
        tmp_path / f"{publication_failure}-certificate.json",
    )
    written = {}

    class BrokenPublisher:
        def publish_create_new(
            self,
            path,
            value,
            *,
            deadline_monotonic_ns,
            progress,
        ):
            assert path == target
            progress()
            payload = (
                b'{"partial":true'
                if publication_failure == "partial"
                else powered_contract.canonical_json_file_bytes(value)
            )
            Path(path).write_bytes(payload)
            written["sha256"] = vq2_module.hashlib.sha256(payload).hexdigest()
            if publication_failure == "partial":
                raise OSError("injected partial certificate write")
            harness.clock.now = deadline_monotonic_ns
            return powered_contract.canonical_file_sha256(value)

    harness.services.publisher = BrokenPublisher()
    output = asyncio.run(
        CalibrationChildLifecycle(harness.admission, harness.services).run()
    )

    harness.services.contract.validate_process_result(
        output.process_result,
        cleanup_certificate=None,
    )
    assert output.exit_code == 1
    assert output.certificate is None
    assert output.process_result["cleanup_certificate"] == {
        "path": target,
        "state": "invalid",
        "sha256": written["sha256"],
    }
    assert "cleanup_unconfirmed" in output.process_result["reason_codes"]


def test_calibration_parent_death_after_first_result_build_is_still_invalidating():
    harness = _lifecycle_harness()

    class LateDeathLifecycle(CalibrationChildLifecycle):
        result_builds = 0

        def _build_process_result(self, audit):
            result = super()._build_process_result(audit)
            self.result_builds += 1
            if self.result_builds == 1:
                harness.boundary.signaled = True
                harness.trace.append("parent_dead_after_first_result")
            return result

    output = asyncio.run(
        LateDeathLifecycle(harness.admission, harness.services).run()
    )

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 1
    assert "wrapper_death" in output.process_result["reason_codes"]
    assert harness.trace.index("parent_dead_after_first_result") < harness.trace.index(
        "lease_takeover"
    )
    assert harness.trace.count("lease_takeover") == 1
    assert harness.trace.count("lease_release") == 1


def test_calibration_prepower_poll_observes_parent_death_before_any_arm_send():
    harness = _lifecycle_harness()

    class ParentDiesInPreflight(CalibrationChildLifecycle):
        async def _preflight(self, phase):
            harness.boundary.signaled = True
            harness.trace.append("parent_dead_before_prepower_poll")
            await self._poll_pause(phase["deadline_monotonic_ns"])

    output = asyncio.run(
        ParentDiesInPreflight(harness.admission, harness.services).run()
    )

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 1
    assert "wrapper_death" in output.process_result["reason_codes"]
    assert "arm_send" not in harness.trace
    assert harness.trace.index("parent_dead_before_prepower_poll") < harness.trace.index(
        "lease_takeover"
    )
    assert harness.trace.count("lease_takeover") == 1
    assert harness.trace.count("cleanup_zero") == 1
    assert harness.trace.count("cleanup_disarm") == 1
    assert harness.trace.count("cleanup_reset") == 1


def test_calibration_parent_supervision_hard_caps_injected_poll_interval():
    harness = _lifecycle_harness()

    class SlowRuntimeProxy:
        MAX_POLL_INTERVAL_NS = 500_000_000

        def __getattr__(self, name):
            return getattr(powered_runtime, name)

    harness.services.runtime = SlowRuntimeProxy()
    lifecycle = CalibrationChildLifecycle(harness.admission, harness.services)
    before = harness.clock.now

    asyncio.run(lifecycle._poll_pause(before + 1_000_000_000))

    assert harness.clock.now == before + 50_000_000


def test_calibration_arm_rechecks_queue_health_after_heartbeat_wait_and_never_arms():
    harness = _lifecycle_harness()

    class DegradeDuringArmWait(CalibrationChildLifecycle):
        async def _wait_for_heartbeat(self, deadline_ns, **kwargs):
            result = await super()._wait_for_heartbeat(deadline_ns, **kwargs)
            if kwargs["phase"] == "arm":
                self.adapter.ingress_dropped = 1
            return result

    lifecycle = DegradeDuringArmWait(harness.admission, harness.services)
    output = asyncio.run(lifecycle.run())

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 1
    assert "capture_incomplete" in output.process_result["reason_codes"]
    assert "arm_send" not in harness.trace
    assert not any(row["category"] == "arm" for row in harness.made.adapter.sent)
    assert not any(
        row["category"] == "attitude_target" and not row["cleanup"]
        for row in harness.made.adapter.sent
    )
    assert "cleanup_zero" in harness.trace


def test_calibration_reset_boundary_collision_aborts_before_reset_send_or_arm():
    harness = _lifecycle_harness(boundary_collision=True)
    lifecycle = CalibrationChildLifecycle(harness.admission, harness.services)

    output = asyncio.run(lifecycle.run())

    certificate = powered_contract.validate_cleanup_certificate(output.certificate)
    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=certificate,
    )
    assert output.exit_code == 1
    assert certificate["collisions"]["invalidating_occurrence_count"] == 1
    assert certificate["collection_invalidating_codes"] == ["collision_observed"]
    assert "initial_reset" not in harness.trace
    assert "arm_send" not in harness.trace
    assert harness.made.adapter.counts["sim_reset"] == 1
    assert harness.trace.count("cleanup_reset") == 1
    assert not any(
        row["category"] == "attitude_target" and not row["cleanup"]
        for row in harness.made.adapter.sent
    )


def test_calibration_outbound_audit_preserves_categories_when_announcement_drops():
    harness = _lifecycle_harness()
    lifecycle = CalibrationChildLifecycle(harness.admission, harness.services)
    retained = NonAttitudeOutboundV1(
        stream_id="vq2-mavlink-udp-14550",
        reset_generation=1,
        outbound_sequence=1,
        host_clock_id=powered_contract.HOST_CLOCK_ID,
        call_start_monotonic_ns=harness.clock.now,
        call_end_monotonic_ns=harness.clock.now,
        category="gcs_heartbeat",
        api="heartbeat_send",
        outcome="returned",
        error_type=None,
        wire=GCSHeartbeatWireV1(6, 8, 0, 0, 4),
    ).to_primitive()
    lifecycle.lineage = SimpleNamespace(
        outbound_receipts=[retained],
        collisions=[],
    )
    lifecycle.adapter = SimpleNamespace(
        outbound_audit=lambda: SimpleNamespace(
            timesync=1,
            gcs_heartbeat=1,
            sim_reset=0,
            arm=0,
            disarm=0,
            attitude_target=0,
            position_target=0,
            other_command=0,
        ),
        outbound_receipt_stats=lambda: SimpleNamespace(dropped=1, buffered=0),
    )

    audit = lifecycle._outbound_audit()

    assert audit["timesync"] == 1
    assert audit["gcs_heartbeat"] == 1
    assert audit["other_command"] == 0
    assert audit["receipt_count"] == 1
    assert audit["receipt_dropped"] == 1
    assert "receipt_incomplete" in lifecycle._cleanup_failures
    assert "unexpected_outbound" in lifecycle._collection_codes


def test_calibration_process_result_cannot_complete_at_unchanged_exit_deadline():
    harness = _lifecycle_harness()
    normal_closer = harness.services.artifact_closer

    def late_closer(*args, **kwargs):
        result = normal_closer(*args, **kwargs)
        harness.clock.now = harness.admission.exit_deadline_monotonic_ns
        return result

    harness.services.artifact_closer = late_closer
    lifecycle = CalibrationChildLifecycle(harness.admission, harness.services)

    output = asyncio.run(lifecycle.run())

    powered_contract.validate_process_result(
        output.process_result,
        cleanup_certificate=output.certificate,
    )
    assert output.exit_code == 1
    assert output.process_result["outcome"] == "failed"
    assert "deadline_expired" in output.process_result["reason_codes"]
    assert output.process_result["completed_monotonic_ns"] == (
        harness.admission.exit_deadline_monotonic_ns
    )


def test_calibration_default_cleanup_proof_uses_public_adapter_transport_state():
    harness = _lifecycle_harness()
    lifecycle = CalibrationChildLifecycle(harness.admission, harness.services)
    lifecycle.adapter = SimpleNamespace(
        powered_transport_state=lambda: PoweredMavlinkTransportState(
            requested_host="127.0.0.1",
            requested_port=14550,
            actual_host="127.0.0.1",
            actual_port=14550,
            frozen_peer=("127.0.0.1", 14551),
            rejected_source_count=0,
            endpoint_closed=True,
            receiver_joined=True,
            announcer_joined=True,
            connection_closed=True,
        )
    )
    lifecycle.vision = None
    lifecycle.guards = powered_runtime.PoweredOutboundGuards()
    lifecycle.guards.latch_production("test_terminal")
    lifecycle.guards.close_cleanup()

    endpoints = lifecycle._default_endpoint_evidence()
    transport = lifecycle._default_transport_evidence()

    assert endpoints["mavlink"] == {
        "state": "closed_with_peer",
        "bind": {
            "family": "AF_INET",
            "requested": {"host": "127.0.0.1", "port": 14550},
            "actual": {"host": "127.0.0.1", "port": 14550},
            "socket_policy": "ipv4-exclusive-address-use",
            "role": "mavlink",
            "owner_process": harness.admission.current_process,
        },
        "frozen_peer": {"host": "127.0.0.1", "port": 14551},
        "rejected_source_count": 0,
    }
    assert endpoints["camera"]["state"] == "not_opened"
    assert all(transport.values())

    lifecycle.vision = SimpleNamespace(
        is_running=False,
        source_diagnostics=lambda: SimpleNamespace(state="peer_frozen"),
    )
    unclosed = lifecycle._default_transport_evidence()
    assert unclosed["vision_closed"] is False
    assert unclosed["owned_handles_closed"] is False

    lifecycle.vision = SimpleNamespace(
        is_running=False,
        source_diagnostics=lambda: SimpleNamespace(state="closed_with_peer"),
    )
    closed = lifecycle._default_transport_evidence()
    assert closed["vision_closed"] is True
    assert closed["owned_handles_closed"] is True

    lifecycle._vision_start_attempted = True
    lifecycle.vision = SimpleNamespace(
        is_running=False,
        source_diagnostics=lambda: SimpleNamespace(state="not_opened"),
    )
    attempted_but_unproved = lifecycle._default_transport_evidence()
    assert attempted_but_unproved["vision_closed"] is False
    assert attempted_but_unproved["owned_handles_closed"] is False


def test_calibration_scheduler_records_post_generation_crossing_and_multi_tick_skip():
    contract = _scheduler_contract()
    evidence = CalibrationCommandEvidence(
        attempt=_scheduler_attempt(), contract=contract
    )
    authorization = evaluate_calibration_safety(_calibration_safety_facts())
    clock = _IntegerClock()
    checks = []
    dispatches = []

    def check(tick):
        checks.append(tick)
        if len(checks) == 3:
            # Evidence enqueue/check work crosses tick 0 and all of tick 1.
            clock.now += 45_000_000
        if tick == 2:
            raise CalibrationCheckFailure(
                "collision_observed", "injected post-boundary collision"
            )
        return authorization

    async def dispatch(_command, _deadline):
        dispatches.append(True)
        raise AssertionError("stale generated command was dispatched")

    scheduler = CalibrationExcitationScheduler(
        evidence=evidence,
        safety_check=check,
        dispatch=dispatch,
        monotonic_ns=clock,
        wait_until_ns=clock.wait_until,
        contract=contract,
    )
    result = asyncio.run(scheduler.run())

    assert dispatches == []
    assert result.completed is False
    assert result.abort_reason_code == "collision_observed"
    assert result.skipped_after_generation == (0,)
    assert result.skipped_before_generation == tuple(range(1, 245))
    dispositions = [
        row
        for event, row in evidence.observations
        if event == "calibration_tick_disposition"
    ]
    assert len(dispositions) == 245
    assert dispositions[0]["disposition"] == "skipped_after_generation"
    assert dispositions[1]["reason_code"] == "slot_missed"
    assert dispositions[2]["reason_code"] == "collision_observed"


def test_calibration_scheduler_invalidates_a_returned_call_started_at_slot_end():
    contract = _scheduler_contract()
    evidence = CalibrationCommandEvidence(
        attempt=_scheduler_attempt(), contract=contract
    )
    authorization = evaluate_calibration_safety(_calibration_safety_facts())
    clock = _IntegerClock()

    async def dispatch(command, _deadline):
        result = _returned_dispatch(command, clock, 0)
        receipt = dict(result.receipt)
        receipt["call_start_monotonic_ns"] = clock.now + 20_000_000
        receipt["call_end_monotonic_ns"] = clock.now + 20_000_000
        return replace(
            result,
            receipt=receipt,
            call_started_monotonic_ns=receipt["call_start_monotonic_ns"],
            call_ended_monotonic_ns=receipt["call_end_monotonic_ns"],
        )

    result = asyncio.run(
        CalibrationExcitationScheduler(
            evidence=evidence,
            safety_check=lambda _tick: authorization,
            dispatch=dispatch,
            monotonic_ns=clock,
            wait_until_ns=clock.wait_until,
            contract=contract,
        ).run()
    )

    assert result.completed is False
    assert result.abort_reason_code == "slot_missed"
    # The local-call receipt is truthful: it remains sent, then latches all
    # later ticks off instead of being mislabeled as not-sent.
    assert result.sent_ticks == (0,)
    assert result.skipped_before_generation == tuple(range(1, 245))


def test_calibration_dimension_admission_precedes_frame_capture_and_latches_drift():
    calls = []

    class Replay:
        def record_event(self, event, **fields):
            calls.append(("event", event, fields))
            return True

        def capture_decoded_snapshot(self, snapshot):
            calls.append(("frame", snapshot.frame_id))
            return True

        def fail(self, reason):
            calls.append(("fail", reason))
            return True

    replay = Replay()
    recorder = vq2_module.JsonlRecorder(None, replay=replay)
    capture = CalibrationSnapshotCapture(
        recorder=recorder,
        config_sha256="a" * 64,
        monotonic_ns=lambda: 20,
    )
    timing = FrameTimingV1(
        identity=FrameIdentityV1("vq2-camera-udp-5600", 2, 1),
        camera_source_time_ns=100,
        host_clock_id="host-perf-counter",
        publication_sequence=1,
        first_unique_packet_monotonic_ns=10,
        final_unique_packet_monotonic_ns=11,
        reassembly_complete_monotonic_ns=12,
        decode_start_monotonic_ns=13,
        decode_end_monotonic_ns=14,
        publish_monotonic_ns=15,
    )
    snapshot = SimpleNamespace(
        generation=2,
        frame_id=1,
        sim_time_ns=100,
        timing=timing,
        camera_frame=CameraFrame(
            timestamp_us=0,
            image=np.zeros((360, 640, 3), dtype=np.uint8),
            width=640,
            height=360,
        ),
    )

    assert capture(snapshot) is True
    assert capture.admitted is True
    assert calls[0][0:2] == ("event", "decoded_dimensions_admission")
    assert calls[1] == ("frame", 1)

    drift = SimpleNamespace(
        generation=2,
        frame_id=2,
        sim_time_ns=101,
        timing=replace(
            timing,
            identity=FrameIdentityV1("vq2-camera-udp-5600", 2, 2),
            camera_source_time_ns=101,
            publication_sequence=2,
        ),
        camera_frame=CameraFrame(
            timestamp_us=0,
            image=np.zeros((359, 640, 3), dtype=np.uint8),
            width=640,
            height=359,
        ),
    )
    with pytest.raises(ValueError, match="drifted"):
        capture(drift)
    assert capture.failure is not None
    assert [call for call in calls if call[0] == "frame"] == [("frame", 1)]


def test_calibration_dimension_callback_serializes_admission_before_concurrent_frames():
    entered = threading.Event()
    release = threading.Event()
    calls = []
    failures = []

    class Replay:
        def record_event(self, event, **_fields):
            calls.append(("event", event))
            entered.set()
            assert release.wait(1.0)
            return True

        def capture_decoded_snapshot(self, snapshot):
            calls.append(("frame", snapshot.frame_id))
            return True

        def fail(self, reason):
            calls.append(("fail", reason))
            return True

    capture = CalibrationSnapshotCapture(
        recorder=vq2_module.JsonlRecorder(None, replay=Replay()),
        config_sha256="a" * 64,
        monotonic_ns=lambda: 20,
    )

    def snapshot(frame_id):
        timing = FrameTimingV1(
            identity=FrameIdentityV1("vq2-camera-udp-5600", 2, frame_id),
            camera_source_time_ns=100 + frame_id,
            host_clock_id="host-perf-counter",
            publication_sequence=frame_id,
            first_unique_packet_monotonic_ns=10,
            final_unique_packet_monotonic_ns=11,
            reassembly_complete_monotonic_ns=12,
            decode_start_monotonic_ns=13,
            decode_end_monotonic_ns=14,
            publish_monotonic_ns=15,
        )
        return SimpleNamespace(
            generation=2,
            frame_id=frame_id,
            sim_time_ns=100 + frame_id,
            timing=timing,
            camera_frame=CameraFrame(
                timestamp_us=0,
                image=np.zeros((360, 640, 3), dtype=np.uint8),
                width=640,
                height=360,
            ),
        )

    def invoke(value):
        try:
            capture(value)
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=invoke, args=(snapshot(1),))
    second = threading.Thread(target=invoke, args=(snapshot(2),))
    first.start()
    assert entered.wait(1.0)
    second.start()
    second.join(0.02)
    assert second.is_alive(), "second callback bypassed serialized admission"
    release.set()
    first.join(1.0)
    second.join(1.0)

    assert failures == []
    assert calls == [
        ("event", "decoded_dimensions_admission"),
        ("frame", 1),
        ("frame", 2),
    ]


def test_calibration_adapter_dispatch_uses_absolute_deadline_and_live_cleanup_gate():
    clock = _IntegerClock()
    replay_events = []

    class Replay:
        def record_event(self, event, **fields):
            replay_events.append((event, fields))
            return True

        def fail(self, _reason):
            return True

    class Guards:
        def __init__(self):
            self.cleanup = None

        def enable_cleanup_live(self, **kwargs):
            self.cleanup = kwargs

    class Adapter:
        def __init__(self):
            self.count = 0
            self.receipts = []
            self.send_kwargs = None
            self.powered_source_promoted = True
            self.powered_outbound_guards = Guards()

        def outbound_audit(self):
            return SimpleNamespace(attitude_target=self.count)

        def drain_outbound_receipts(self):
            values, self.receipts = self.receipts, []
            return values

        async def send_attitude_rate(self, command, **kwargs):
            self.send_kwargs = kwargs
            self.receipts.append(
                AttitudeTargetOutboundV1(
                    stream_id="vq2-mavlink-udp-14550",
                    reset_generation=2,
                    outbound_sequence=self.count,
                    host_clock_id="host-perf-counter",
                    call_start_monotonic_ns=clock.now,
                    call_end_monotonic_ns=clock.now,
                    api="send_attitude_rate",
                    outcome="returned",
                    error_type=None,
                    wire=AttitudeTargetWireV1(
                        time_boot_ms=1,
                        target_system=1,
                        target_component=1,
                        type_mask=128,
                        q_wxyz=(1.0, 0.0, 0.0, 0.0),
                        body_rates_rad_s=(-command.roll_rate, 0.0, 0.0),
                        thrust=command.thrust,
                    ),
                )
            )
            self.count += 1

    adapter = Adapter()
    recorder = vq2_module.JsonlRecorder(None, replay=Replay())
    lineage = CalibrationLineageRecorder(recorder)
    dispatcher = CalibrationAdapterDispatcher(
        adapter,
        lineage,
        monotonic_ns=clock,
        parent_alive=lambda: True,
        lease_valid=lambda: True,
    )
    deadline = clock.now + 20_000_000
    result = asyncio.run(
        dispatcher.dispatch(AttitudeRateCommand(0.08, 0.0, 0.0, 0.235), deadline)
    )

    assert result.error is None
    assert result.audit_count_after == result.audit_count_before + 1
    assert adapter.send_kwargs == {
        "powered_deadline_monotonic_ns": deadline,
        "powered_cleanup": False,
    }
    assert replay_events[0][0] == "attitude_target_outbound"

    dispatcher.begin_live_cleanup()
    assert adapter.powered_outbound_guards.cleanup == {
        "parent_alive": True,
        "lease_valid": True,
        "source_promoted": True,
    }


def _calibration_cli_fixture():
    suffix = [
        "--stage",
        "calibration-excite",
        "--powered-attempt-envelope",
        "C:\\evidence\\attempt.json",
        "--wrapper-process",
        "10:1010",
        "--powered-process-authority",
        "C:\\evidence\\child-authority.json",
        "--attempt-capability-handle",
        "41",
        "--parent-liveness-handle",
        "42",
        "--record",
        "C:\\evidence\\session.jsonl.gz",
        "--replay-bundle",
        "C:\\evidence\\session.vq2replay",
        "--cleanup-certificate",
        "C:\\evidence\\child-cleanup-certificate.json",
        "--recording-approved",
    ]
    full = [
        "C:\\python.exe",
        "-E",
        "-s",
        "-B",
        "-m",
        "scripts.aigp_vq2_run",
        *suffix,
    ]
    wrapper = {"pid": 10, "creation_filetime_100ns": 1010}
    child = {"pid": 11, "creation_filetime_100ns": 1011}
    paths = {
        "attempt_envelope": suffix[3],
        "child_authority": suffix[7],
        "legacy_record": suffix[13],
        "replay_bundle": suffix[15],
        "child_cleanup_certificate": suffix[17],
    }
    attempt = {
        "context": {
            "live_freeze": {"path": "C:\\evidence\\freeze.json"},
            "paths": paths,
            "child_argv": full,
            "wrapper_process": wrapper,
        },
        "context_sha256": "a" * 64,
    }
    authority = {
        "role": "powered_child",
        "process": child,
        "parent_handle": {"value": 42},
        "capability_sha256": "b" * 64,
        "absolute_deadlines": {
            "anchor": 100,
            "total": 110_000_000_100,
            "prepower": 52_000_000_100,
            "powered": 57_000_000_100,
            "cleanup": 72_000_000_100,
            "replay_close": 107_000_000_100,
            "exit": 110_000_000_100,
        },
    }
    return suffix, full, attempt, authority, wrapper, child


def test_calibration_cli_is_exact_and_fails_before_live_import(monkeypatch):
    suffix, *_rest = _calibration_cli_fixture()
    abbreviated = list(suffix)
    abbreviated[0] = "--sta"
    with pytest.raises(SystemExit):
        vq2_module.parse_calibration_arguments(abbreviated)
    with pytest.raises(SystemExit):
        vq2_module.parse_calibration_arguments([*suffix, "--verbose"])

    monkeypatch.setattr(
        vq2_module,
        "_load_live_transport_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("live import before admission")),
    )
    stderr = SimpleNamespace(data=b"", write=lambda value: None, flush=lambda: None)
    assert vq2_module.main(suffix, stderr=stderr) == 2


def test_runner_module_import_is_inert_for_live_transport_modules():
    code = (
        "import sys; import scripts.aigp_vq2_run; "
        "assert 'competition.aigp_mavlink' not in sys.modules; "
        "assert 'competition.vq2_vision' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-E", "-s", "-B", "-c", code],
        cwd=Path(vq2_module.__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert completed.returncode == 0, completed.stderr


def test_default_calibration_builder_defers_every_post_admission_factory():
    suffix, *_rest = _calibration_cli_fixture()
    arguments = vq2_module.parse_calibration_arguments(suffix)
    events = []

    class CapabilityOperations:
        def close_handle(self, handle):
            events.append(("capability-close", handle))

    class Qpc:
        def now_ns(self):
            return 100

    process_boundary = SimpleNamespace()
    built_child = []

    def recorder(admission):
        events.append(("recorder", admission))
        return "recorder"

    def adapter(**kwargs):
        events.append(("adapter", dict(kwargs["bind"])))
        return "adapter"

    def vision(**kwargs):
        events.append(("vision", dict(kwargs["bind"])))
        return "vision"

    def camera(host, port):
        events.append(("camera", host, port))
        return "socket"

    services = vq2_module.build_default_calibration_services(
        arguments,
        capability_operations_factory=lambda: events.append("capability")
        or CapabilityOperations(),
        qpc_provider_factory=lambda: events.append("qpc") or Qpc(),
        process_boundary_factory=lambda capability, parent: events.append(
            ("process", capability, parent)
        )
        or process_boundary,
        delegated_lease_factory=lambda admission, boundary, qpc: events.append(
            ("lease", admission, boundary, qpc)
        )
        or "lease",
        recorder_builder=recorder,
        adapter_builder=adapter,
        vision_builder=vision,
        camera_socket_builder=camera,
        publisher_factory=lambda **kwargs: events.append(("publisher", kwargs))
        or "publisher",
        child_runner=lambda admission, child: built_child.append(child) or 0,
    )

    assert events == ["capability", "qpc", ("process", 41, 42)]
    admission = SimpleNamespace(name="admitted")
    assert services.run_admitted(admission) == 0
    assert [item[0] for item in events[3:]] == ["lease", "publisher"]
    child = built_child[0]
    assert child.process_boundary is process_boundary
    assert child.lease_boundary == "lease"
    assert child.publisher == "publisher"
    assert child.recorder_factory(admission) == "recorder"
    bind = {
        "host": "127.0.0.1",
        "port": 15555,
        "socket_policy": "ipv4-exclusive-address-use",
    }
    assert child.adapter_factory(admission=admission, bind=bind) == "adapter"
    assert child.vision_factory(admission=admission, bind=bind) == "vision"
    assert child.camera_socket_factory(bind["host"], bind["port"]) == "socket"
    assert events[-3:] == [
        ("adapter", bind),
        ("vision", bind),
        ("camera", "127.0.0.1", 15555),
    ]
    assert services.close_unconsumed_capability() is True
    assert services.close_unconsumed_capability() is True
    assert events.count(("capability-close", 41)) == 1


def test_default_camera_factory_transfers_only_the_frozen_bound_socket():
    socket_token = object()
    calls = []
    endpoints = []

    class Runtime:
        ExclusiveUdpEndpoint = powered_runtime.ExclusiveUdpEndpoint

        @staticmethod
        def create_exclusive_udp_endpoint(host, port):
            calls.append((host, port))
            endpoint = powered_runtime.ExclusiveUdpEndpoint(
                socket=socket_token,
                requested_host=host,
                requested_port=port,
                actual_host=host,
                actual_port=port,
                exclusive_option=9_999,
            )
            endpoints.append(endpoint)
            return endpoint

    result = vq2_module._create_default_calibration_camera_socket(
        "127.0.0.1",
        15678,
        runtime=Runtime,
    )

    assert result is socket_token
    assert calls == [("127.0.0.1", 15678)]
    assert endpoints[0].socket is None
    assert endpoints[0].socket_transferred is True


def test_default_camera_factory_closes_invalid_partial_endpoint_without_binding():
    class InvalidEndpoint:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    invalid = InvalidEndpoint()

    class Runtime:
        ExclusiveUdpEndpoint = powered_runtime.ExclusiveUdpEndpoint

        @staticmethod
        def create_exclusive_udp_endpoint(host, port):
            assert (host, port) == ("127.0.0.1", 0)
            return invalid

    with pytest.raises(
        vq2_module.CalibrationLifecycleError,
        match="exact ExclusiveUdpEndpoint",
    ):
        vq2_module._create_default_calibration_camera_socket(
            "127.0.0.1",
            0,
            runtime=Runtime,
        )

    assert invalid.closed is True


def test_calibration_main_builds_default_services_before_admission(monkeypatch):
    suffix, *_rest = _calibration_cli_fixture()
    events = []

    class Boundary:
        def parent_signaled(self, _handle):
            return False

        def close_owned_handles(self, *, deadline_monotonic_ns, monotonic_ns):
            events.append(("close", deadline_monotonic_ns, monotonic_ns()))
            return SimpleNamespace(proved=True)

    boundary = Boundary()
    services = CalibrationAdmissionServices(
        process_boundary=boundary,
        capability_operations=object(),
        monotonic_ns=lambda: 90,
        run_admitted=lambda admission: events.append(("run", admission)) or 0,
    )
    admission = SimpleNamespace(
        parent_handle=42,
        exit_deadline_monotonic_ns=120,
        erase_role_secret=lambda: events.append("erase"),
    )
    monkeypatch.setattr(
        vq2_module,
        "build_default_calibration_services",
        lambda arguments: events.append(("build", arguments)) or services,
    )
    monkeypatch.setattr(
        vq2_module,
        "admit_calibration_child",
        lambda arguments, active: events.append(("admit", arguments, active))
        or admission,
    )
    monkeypatch.setattr(
        vq2_module,
        "_load_live_transport_dependencies",
        lambda: events.append("live-import") or (object(), object(), object()),
    )

    assert vq2_module.main(
        suffix,
        stderr=SimpleNamespace(write=lambda _value: None, flush=lambda: None),
    ) == 0
    assert [item if isinstance(item, str) else item[0] for item in events] == [
        "build",
        "admit",
        "live-import",
        "run",
        "close",
        "erase",
    ]


def test_calibration_main_consumes_capability_before_lazy_live_import(monkeypatch):
    suffix, full, attempt, authority, wrapper, child = _calibration_cli_fixture()
    events = []

    class Contract:
        @staticmethod
        def validate_attempt(value, *, live_freeze=None):
            return dict(value)

        @staticmethod
        def validate_live_freeze(value):
            return dict(value)

        @staticmethod
        def validate_process_authority(value, *, attempt, argv):
            assert tuple(argv) == tuple(full)
            return dict(value)

        @staticmethod
        def canonical_file_sha256(_value):
            return "c" * 64

        @staticmethod
        def canonical_json_file_bytes(value):
            return (
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")

    class Runtime:
        @staticmethod
        def parse_decimal_handle(value):
            return int(value)

        @staticmethod
        def parse_process_identity_token(value):
            pid, creation = value.split(":")
            return SimpleNamespace(pid=int(pid), creation_filetime_100ns=int(creation))

        @staticmethod
        def validate_process_identity(value):
            return dict(value)

        @staticmethod
        def read_bound_capability(*_args, **kwargs):
            events.append("capability")
            assert kwargs["domain"] == "aigp-vq2-powered-child/1"
            return b"x" * 32

        @staticmethod
        def read_qpc_ns(_clock):
            return 200

    handle_close_calls = []

    class ProcessBoundary:
        def current_argv(self):
            return full

        def current_process_identity(self):
            return child

        def retained_process_identity(self, handle):
            assert handle == 42
            return wrapper

        def prove_inherited_handle_policy(self, **kwargs):
            assert kwargs["capability_handle"] == 41
            return True

        def parent_signaled(self, _handle):
            return False

        def close_owned_handles(self, *, deadline_monotonic_ns, monotonic_ns):
            handle_close_calls.append((deadline_monotonic_ns, monotonic_ns()))
            return SimpleNamespace(proved=True)

    records = {
        suffix[3]: attempt,
        "C:\\evidence\\freeze.json": {"schema": "freeze"},
        suffix[7]: authority,
    }
    retained = []
    services = CalibrationAdmissionServices(
        process_boundary=ProcessBoundary(),
        capability_operations=object(),
        monotonic_ns=lambda: 200,
        contract=Contract,
        runtime=Runtime,
        load_record=lambda path, _contract: records[path],
        run_admitted=lambda admission: retained.append(admission) or 0,
    )
    monkeypatch.setattr(
        vq2_module,
        "_load_live_transport_dependencies",
        lambda: events.append("live-import") or (object(), object(), object()),
    )
    stderr = SimpleNamespace(write=lambda _value: None, flush=lambda: None)

    assert vq2_module.main(
        suffix,
        calibration_services=services,
        stderr=stderr,
    ) == 0
    assert events == ["capability", "live-import"]
    assert len(retained) == 1
    assert handle_close_calls == [(retained[0].exit_deadline_monotonic_ns, 200)]
    assert bytes(retained[0].role_secret) == b"\x00" * 32

    class FailingCloseBoundary(ProcessBoundary):
        def close_owned_handles(self, *, deadline_monotonic_ns, monotonic_ns):
            handle_close_calls.append((deadline_monotonic_ns, monotonic_ns()))
            raise OSError("injected bootstrap handle close failure")

    events.clear()
    retained.clear()
    handle_close_calls.clear()
    close_stderr = []
    failing_services = replace(
        services,
        process_boundary=FailingCloseBoundary(),
    )
    assert vq2_module.main(
        suffix,
        calibration_services=failing_services,
        stderr=SimpleNamespace(
            write=close_stderr.append,
            flush=lambda: None,
        ),
    ) == 1
    assert events == ["capability", "live-import"]
    assert len(retained) == 1
    assert handle_close_calls == [(retained[0].exit_deadline_monotonic_ns, 200)]
    assert close_stderr == [
        b"powered calibration bootstrap handle closure failed\n"
    ]
    assert bytes(retained[0].role_secret) == b"\x00" * 32

    class DeathAfterFlushBoundary(ProcessBoundary):
        def __init__(self):
            self.signaled = False

        def parent_signaled(self, _handle):
            return self.signaled

    death_boundary = DeathAfterFlushBoundary()
    events.clear()
    retained.clear()
    handle_close_calls.clear()
    published_stdout = []

    class SignalingStdout:
        def write(self, value):
            published_stdout.append(value)

        def flush(self):
            death_boundary.signaled = True

    late_output = CalibrationChildRunOutput(
        certificate=None,
        certificate_sha256=None,
        process_result={"schema": "test-process-result"},
        exit_code=0,
    )
    death_services = replace(
        services,
        process_boundary=death_boundary,
        run_admitted=lambda admission: retained.append(admission) or late_output,
    )
    assert vq2_module.main(
        suffix,
        calibration_services=death_services,
        stdout=SignalingStdout(),
        stderr=SimpleNamespace(write=lambda _value: None, flush=lambda: None),
    ) == 1
    assert published_stdout == [b'{"schema":"test-process-result"}\n']
    assert len(retained) == 1
    assert handle_close_calls == [(retained[0].exit_deadline_monotonic_ns, 200)]
