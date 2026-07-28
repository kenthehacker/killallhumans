from __future__ import annotations

from dataclasses import asdict, replace
import math

import pytest

from competition.adapter import AttitudeRateCommand
from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetVisualTracker,
    VisualDetection,
    VisualDetectionFrame,
    VisualInnerApertureGeometry,
    VisualTrack,
    VisualTrackRole,
    visual_track_history_sha256,
)
from planning.vq2_dynamic_course import (
    DynamicCourseConfig,
    DynamicCourseError,
    ImuAttitudeSample,
    MAX_TARGET_PITCH_RAD,
    MAX_TARGET_ROLL_RAD,
    MAX_THRUST,
    MAX_YAW_RATE_RAD_S,
    MIN_TARGET_PITCH_RAD,
    MIN_THRUST,
    TrackSteeringPrediction,
)
from planning.vq2_dynamic_visual_approach import (
    BUILD_3385_EFFECTIVE_CAMERA_TO_BODY_WXYZ,
    DYNAMIC_CONTROLLER_FAMILY,
    DynamicRollingVisualApproachServo,
    DynamicVisualCourseSession,
    PostCreditSuccessorSteeringUnavailable,
    _predicted_successor_pitch_reference,
    production_dynamic_course_config,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    ConfirmedGateReacquisition,
    CreditedUnboundGateAdvance,
    RollingVisualGateGraph,
)
from planning.vq2_visual_approach import (
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachMode,
    VisualApproachRefusal,
)


_BASE_NS = 10_000_000_000
_PERIOD_NS = 33_000_000
_ALL_FRAME_EDGES = (
    FrameEdge.LEFT
    | FrameEdge.TOP
    | FrameEdge.RIGHT
    | FrameEdge.BOTTOM
)
_AUTO_INNER_APERTURE = object()


def test_production_camera_boundary_uses_live_axis_calibration() -> None:
    config = production_dynamic_course_config()

    assert config.camera_to_body_wxyz == (
        BUILD_3385_EFFECTIVE_CAMERA_TO_BODY_WXYZ
    )
    assert config.camera_to_body_wxyz == (0.0, 1.0, 0.0, 0.0)
    # Rx(pi) maps image-right to body-left, while positive FRD roll accelerates
    # body-right.  The bounded translational demand therefore opposes positive
    # stable/image bearing.
    assert config.roll_guidance_sign == -1.0


def test_successor_pitch_reference_steers_from_predicted_vertical_geometry():
    config = production_dynamic_course_config()

    top_target, top_error, top_rate, top_lead = (
        _predicted_successor_pitch_reference(
            camera_center_y_norm=-0.605,
            camera_center_rate_y_norm_s=-0.309,
            vertical_angle_scale_rad=config.vertical_angle_scale_rad,
            pitch_command_delay_s=config.pitch_command_delay_s,
            maximum_lead_rad=(
                config.successor_prediction_max_extrapolation_rad
            ),
            baseline_pitch_rad=config.brake_pitch_rad,
        )
    )
    worsening_target, *_ = _predicted_successor_pitch_reference(
        camera_center_y_norm=-0.696,
        camera_center_rate_y_norm_s=-0.544,
        vertical_angle_scale_rad=config.vertical_angle_scale_rad,
        pitch_command_delay_s=config.pitch_command_delay_s,
        maximum_lead_rad=(
            config.successor_prediction_max_extrapolation_rad
        ),
        baseline_pitch_rad=config.brake_pitch_rad,
    )
    recovering_target, *_ = _predicted_successor_pitch_reference(
        camera_center_y_norm=-0.605,
        camera_center_rate_y_norm_s=0.500,
        vertical_angle_scale_rad=config.vertical_angle_scale_rad,
        pitch_command_delay_s=config.pitch_command_delay_s,
        maximum_lead_rad=(
            config.successor_prediction_max_extrapolation_rad
        ),
        baseline_pitch_rad=config.brake_pitch_rad,
    )
    recovered_target, recovered_error, recovered_rate, recovered_lead = (
        _predicted_successor_pitch_reference(
            camera_center_y_norm=0.0,
            camera_center_rate_y_norm_s=0.0,
            vertical_angle_scale_rad=config.vertical_angle_scale_rad,
            pitch_command_delay_s=config.pitch_command_delay_s,
            maximum_lead_rad=(
                config.successor_prediction_max_extrapolation_rad
            ),
            baseline_pitch_rad=config.brake_pitch_rad,
        )
    )
    bottom_target, *_ = _predicted_successor_pitch_reference(
        camera_center_y_norm=0.020,
        camera_center_rate_y_norm_s=0.0,
        vertical_angle_scale_rad=config.vertical_angle_scale_rad,
        pitch_command_delay_s=config.pitch_command_delay_s,
        maximum_lead_rad=(
            config.successor_prediction_max_extrapolation_rad
        ),
        baseline_pitch_rad=config.brake_pitch_rad,
    )

    assert all(
        math.isfinite(value)
        for value in (
            top_target,
            top_error,
            top_rate,
            top_lead,
            worsening_target,
            recovering_target,
            recovered_target,
            recovered_error,
            recovered_rate,
            recovered_lead,
            bottom_target,
        )
    )
    assert all(
        MIN_TARGET_PITCH_RAD <= target <= MAX_TARGET_PITCH_RAD
        for target in (
            worsening_target,
            top_target,
            recovering_target,
            recovered_target,
            bottom_target,
        )
    )
    assert worsening_target <= top_target < recovering_target < 0.0
    assert top_error < 0.0
    assert top_rate < 0.0
    assert top_lead < 0.0
    assert recovered_target == pytest.approx(config.brake_pitch_rad)
    assert bottom_target > recovered_target
    assert recovered_error == 0.0
    assert recovered_rate == 0.0
    assert recovered_lead == 0.0


def test_successor_steering_uses_full_bank_only_while_off_axis_and_outward():
    session = _session(config=production_dynamic_course_config())
    prediction = TrackSteeringPrediction(
        track_id="gate-b",
        stream_generation=1,
        monotonic_ns=_BASE_NS,
        source_state_monotonic_ns=_BASE_NS,
        last_measurement_monotonic_ns=_BASE_NS,
        measurement_age_s=0.0,
        stable_bearing_rad=(0.40, 0.0),
        stable_bearing_rate_rad_s=(0.10, 0.0),
        camera_center_norm=(0.30, -0.40),
        camera_center_rate_norm_s=(0.05, 0.0),
        bearing_std_rad=(0.02, 0.02),
        body_rates_rad_s=(0.0, 0.0, 0.0),
    )

    outward = session._successor_steering_targets(prediction)
    recovered = session._successor_steering_targets(
        replace(
            prediction,
            stable_bearing_rate_rad_s=(-0.10, 0.0),
        )
    )
    unidentified = _session(
        config=replace(
            production_dynamic_course_config(),
            roll_guidance_sign=0.0,
        )
    )._successor_steering_targets(prediction)

    assert outward["target_roll_rad"] == pytest.approx(
        -MAX_TARGET_ROLL_RAD
    )
    assert outward["target_pitch_rad"] == pytest.approx(
        session.core.config.brake_pitch_rad
    )
    assert (
        -MAX_TARGET_ROLL_RAD
        < recovered["target_roll_rad"]
        < 0.0
    )
    assert (
        recovered["target_pitch_rad"]
        < session.core.config.brake_pitch_rad
    )
    assert unidentified["target_roll_rad"] == 0.0
    assert (
        unidentified["target_pitch_rad"]
        < session.core.config.brake_pitch_rad
    )


def _inner_aperture(
    center_x: float,
    center_y: float,
    *,
    half_width: float,
    half_height: float,
    confidence: float = 0.93,
    measurement_std: tuple[float, float, float] = (
        0.012,
        0.014,
        0.040,
    ),
    health_reason: str | None = None,
) -> VisualInnerApertureGeometry:
    return VisualInnerApertureGeometry(
        center_norm=(center_x, center_y),
        half_size_norm=(half_width, half_height),
        log_scale=math.log(math.sqrt(half_width * half_height)),
        measurement_std=measurement_std,
        confidence=confidence,
        clipping=FrameEdge.NONE,
        visible_edges=_ALL_FRAME_EDGES,
        geometry_model_id="test-inner-aperture-v1",
        covariance_model_id="test-inner-aperture-covariance-v1",
        health_reason=health_reason,
    )


def _rejected_inner_aperture(
    *,
    clipping: FrameEdge,
    health_reason: str,
) -> VisualInnerApertureGeometry:
    return VisualInnerApertureGeometry(
        center_norm=None,
        half_size_norm=None,
        log_scale=None,
        measurement_std=None,
        confidence=0.0,
        clipping=clipping,
        visible_edges=FrameEdge.NONE,
        geometry_model_id=None,
        covariance_model_id=None,
        health_reason=health_reason,
    )


def _detection(
    source_index: int,
    center_x: float,
    *,
    center_y: float = 0.0,
    width: float,
    height: float,
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
    inner_aperture: VisualInnerApertureGeometry | None | object = (
        _AUTO_INNER_APERTURE
    ),
) -> VisualDetection:
    unit_x = 0.5 * (center_x + 1.0)
    unit_y = 0.5 * (center_y + 1.0)
    if inner_aperture is _AUTO_INNER_APERTURE:
        inner_aperture = (
            _rejected_inner_aperture(
                clipping=clipping,
                health_reason="test-clipped-or-censored-inner-aperture",
            )
            if clipping != FrameEdge.NONE or center_censored
            else _inner_aperture(
                center_x,
                center_y,
                half_width=width,
                half_height=height,
            )
        )
    assert (
        inner_aperture is None
        or isinstance(inner_aperture, VisualInnerApertureGeometry)
    )
    return VisualDetection(
        source_index=source_index,
        center_norm=(center_x, center_y),
        bbox_norm=(
            unit_x - width / 2.0,
            unit_y - height / 2.0,
            unit_x + width / 2.0,
            unit_y + height / 2.0,
        ),
        confidence=0.95,
        clipping=clipping,
        center_censored=center_censored,
        inner_aperture=inner_aperture,
    )


def _frame(
    sequence: int,
    *,
    current_width: float = 0.34,
    current_height: float = 0.36,
    include_successor: bool = True,
    current_center_x: float = 0.0,
    current_center_y: float = 0.0,
    current_clipping: FrameEdge = FrameEdge.NONE,
    current_center_censored: bool = False,
    current_inner_aperture: VisualInnerApertureGeometry | None | object = (
        _AUTO_INNER_APERTURE
    ),
    successor_center_x: float = 0.32,
    successor_center_y: float = 0.0,
    successor_width: float = 0.15,
    successor_height: float = 0.17,
    successor_clipping: FrameEdge = FrameEdge.NONE,
    successor_center_censored: bool = False,
    successor_inner_aperture: VisualInnerApertureGeometry | None | object = (
        _AUTO_INNER_APERTURE
    ),
) -> VisualDetectionFrame:
    observation_ns = _BASE_NS + sequence * _PERIOD_NS
    detections = [
        _detection(
            0,
            current_center_x,
            center_y=current_center_y,
            width=current_width,
            height=current_height,
            clipping=current_clipping,
            center_censored=current_center_censored,
            inner_aperture=current_inner_aperture,
        )
    ]
    if include_successor:
        detections.append(
            _detection(
                1,
                successor_center_x,
                center_y=successor_center_y,
                width=successor_width,
                height=successor_height,
                clipping=successor_clipping,
                center_censored=successor_center_censored,
                inner_aperture=successor_inner_aperture,
            )
        )
    return VisualDetectionFrame(
        token=CameraFrameToken(
            generation=1,
            frame_id=1_000 + sequence,
            publication_sequence=sequence,
            stream_id="dynamic-camera",
        ),
        provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
        time_basis_id="host-perf-counter",
        image_size_px=(640, 360),
        detections=tuple(detections),
        camera_source_time_ns=20_000_000_000 + sequence,
        final_unique_packet_monotonic_ns=observation_ns,
        publish_monotonic_ns=observation_ns + 1_000_000,
    )


def _race(received_ns: int) -> AuthoritativeRaceStatusRef:
    return AuthoritativeRaceStatusRef.live(
        session_id="dynamic-adapter-test",
        reset_epoch=1,
        race_generation=1,
        race_status_sequence=1,
        race_status_boot_ms=4_000,
        active_gate_index=0,
        received_monotonic_ns=received_ns,
        host_clock_id="host-perf-counter",
    )


def _credited_race(received_ns: int) -> AuthoritativeRaceStatusRef:
    return AuthoritativeRaceStatusRef.live(
        session_id="dynamic-adapter-test",
        reset_epoch=1,
        race_generation=1,
        race_status_sequence=2,
        race_status_boot_ms=4_250,
        active_gate_index=1,
        received_monotonic_ns=received_ns,
        host_clock_id="host-perf-counter",
    )


def _graph() -> tuple[
    MultiTargetVisualTracker,
    RollingVisualGateGraph,
    object,
    str,
]:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = ""
    snapshot = None
    for sequence in range(1, 6):
        update = tracker.update(_frame(sequence))
        if sequence == 1:
            current_id = update.visible_track_ids[0]
        if sequence == 3:
            assert update.publish_monotonic_ns is not None
            snapshot = graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=_race(
                    update.publish_monotonic_ns + 1_000_000
                ),
            )
        elif sequence > 3:
            snapshot = graph.observe(tracker)
    assert snapshot is not None
    return tracker, graph, snapshot, current_id


def _single_gate_graph(
    *,
    width: float,
    height: float,
    inner_aperture: VisualInnerApertureGeometry | None | object = (
        _AUTO_INNER_APERTURE
    ),
) -> tuple[
    MultiTargetVisualTracker,
    RollingVisualGateGraph,
    object,
    str,
]:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = ""
    snapshot = None
    for sequence in range(1, 6):
        update = tracker.update(
            _frame(
                sequence,
                current_width=width,
                current_height=height,
                include_successor=False,
                current_inner_aperture=inner_aperture,
            )
        )
        if sequence == 1:
            current_id = update.visible_track_ids[0]
        if sequence == 3:
            assert update.publish_monotonic_ns is not None
            snapshot = graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=_race(
                    update.publish_monotonic_ns + 1_000_000
                ),
            )
        elif sequence > 3:
            snapshot = graph.observe(tracker)
    assert snapshot is not None
    return tracker, graph, snapshot, current_id


def _session(
    *,
    config: DynamicCourseConfig | None = None,
    pitch_rate_rad_s: float = 0.0,
) -> DynamicVisualCourseSession:
    session = DynamicVisualCourseSession(config)
    for monotonic_ns in range(
        _BASE_NS - 200_000_000,
        _BASE_NS + 700_000_000,
        10_000_000,
    ):
        pitch_rad = (
            pitch_rate_rad_s
            * (monotonic_ns - _BASE_NS)
            / 1_000_000_000.0
        )
        session.record_imu(
            ImuAttitudeSample(
                monotonic_ns=monotonic_ns,
                body_to_reference_wxyz=(
                    math.cos(pitch_rad / 2.0),
                    0.0,
                    math.sin(pitch_rad / 2.0),
                    0.0,
                ),
                body_rates_rad_s=(0.0, pitch_rate_rad_s, 0.0),
                source_timestamp_us=monotonic_ns // 1_000,
                host_clock_id="host-perf-counter",
            )
        )
    return session


def _observe(
    planner: DynamicRollingVisualApproachServo,
    snapshot: object,
    tracker: MultiTargetVisualTracker,
):
    update = tracker.latest_update
    assert update is not None
    return planner.observe(
        snapshot,
        tracker,
        now_monotonic_s=(
            update.observation_monotonic_ns + 5_000_000
        )
        / 1_000_000_000.0,
        segment_elapsed_s=0.7,
        segment_yaw_excursion_rad=0.0,
    )


def _accept_proposal(
    session: DynamicVisualCourseSession,
    tracker: MultiTargetVisualTracker,
    proposal: object,
) -> None:
    update = tracker.latest_update
    assert update is not None
    output = proposal.servo_output
    wire_ns = update.observation_monotonic_ns + 8_000_000
    session.record_wire_acceptance(
        target_roll_rad=output.target_roll_rad,
        target_pitch_rad=output.target_pitch_rad,
        yaw_rate_rad_s=output.yaw_rate_rad_s,
        thrust=output.thrust,
        wire_command=AttitudeRateCommand(
            0.0,
            0.0,
            output.yaw_rate_rad_s,
            output.thrust,
        ),
        wire_start_monotonic_ns=wire_ns,
    )


def _propagated_vertical_fov_gap(
    *,
    config: DynamicCourseConfig | None = None,
    clipping: FrameEdge = FrameEdge.BOTTOM,
) -> tuple[
    DynamicVisualCourseSession,
    VisualTrack,
    CameraFrameToken,
    int,
]:
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.34,
        height=0.36,
    )
    session = _session(config=config, pitch_rate_rad_s=0.40)
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)
    update = tracker.update(
        _frame(
            6,
            current_width=0.40,
            current_height=0.44,
            include_successor=False,
            current_clipping=clipping,
            current_center_censored=True,
            current_inner_aperture=None,
        )
    )
    snapshot = graph.observe(tracker)
    _observe(planner, snapshot, tracker)
    decision = session.last_decision
    assert decision is not None
    assert decision.current_aperture_propagated
    return (
        session,
        tracker.track(current_id),
        update.token,
        decision.monotonic_ns,
    )


def _bound_post_credit_successor() -> tuple[
    DynamicVisualCourseSession,
    str,
]:
    tracker, graph, snapshot, current_id = _graph()
    session = _session()
    successor_id = snapshot.next_candidates[0].track_id
    session.stage_snapshot(
        snapshot,
        tracker,
        expected_gate_index=0,
        expected_current_track_id=current_id,
        adjacent_precredit=False,
    )
    for sequence in range(6, 10):
        tracker.update(_frame(sequence))
        snapshot = graph.observe(tracker)
        session.stage_snapshot(
            snapshot,
            tracker,
            expected_gate_index=0,
            expected_current_track_id=current_id,
            adjacent_precredit=False,
        )
    session.core.bind(
        current_gate_index=0,
        current_track_id=current_id,
        successor_track_id=successor_id,
    )
    successor = session.core.course_state().successor
    assert successor is not None
    assert successor.sample_count >= 4
    return session, successor_id


def _activated_committed_roll_handoff(
) -> tuple[DynamicVisualCourseSession, object, str, float]:
    tracker, graph, snapshot, current_id = _graph()
    session = _session()
    successor_id = snapshot.next_candidates[0].track_id
    session.stage_snapshot(
        snapshot,
        tracker,
        expected_gate_index=0,
        expected_current_track_id=current_id,
        adjacent_precredit=False,
    )
    for sequence in range(6, 10):
        tracker.update(_frame(sequence))
        snapshot = graph.observe(tracker)
        session.stage_snapshot(
            snapshot,
            tracker,
            expected_gate_index=0,
            expected_current_track_id=current_id,
            adjacent_precredit=False,
        )
    session.core.bind(
        current_gate_index=0,
        current_track_id=current_id,
        successor_track_id=successor_id,
    )
    successor = session.core.course_state().successor
    assert successor is not None
    seed_wire_ns = successor.state_monotonic_ns + 1_000_000
    session.record_wire_acceptance(
        target_roll_rad=0.0,
        target_pitch_rad=0.0,
        yaw_rate_rad_s=0.0,
        thrust=0.275,
        wire_command=AttitudeRateCommand(0.0, 0.0, 0.0, 0.275),
        wire_start_monotonic_ns=seed_wire_ns,
    )
    session.stage_snapshot(
        snapshot,
        tracker,
        expected_gate_index=1,
        expected_current_track_id=successor_id,
        adjacent_precredit=True,
    )
    authority_ns = seed_wire_ns + 1_000_000
    authority = session.adjacent_precredit_successor_steering_authority(
        track_id=successor_id,
        now_monotonic_ns=authority_ns,
    )
    retained_roll_rad = float(authority["target_roll_rad"])
    accepted_ns = authority_ns + 1_000_000
    session.record_wire_acceptance(
        target_roll_rad=retained_roll_rad,
        target_pitch_rad=float(authority["target_pitch_rad"]),
        yaw_rate_rad_s=float(authority["yaw_rate_rad_s"]),
        thrust=float(authority["thrust"]),
        wire_command=AttitudeRateCommand(
            0.02,
            0.01,
            float(authority["yaw_rate_rad_s"]),
            float(authority["thrust"]),
        ),
        wire_start_monotonic_ns=accepted_ns,
    )
    race_received_ns = accepted_ns + 1_000_000
    activation_ns = race_received_ns + 1_000_000
    session.activate_post_credit_successor_steering(
        _credited_race(race_received_ns),
        from_gate_index=0,
        reviewed_track_id=successor_id,
        activation_monotonic_ns=activation_ns,
    )
    assert session.post_credit_roll_reference_handoff_active
    source = session.core.guide(activation_ns + 1_000_000)
    return session, source, successor_id, retained_roll_rad


def test_outward_post_credit_demand_cannot_unwind_retained_successor_bank():
    session, source, successor_id, retained = (
        _activated_committed_roll_handoff()
    )
    estimate = session.core._tracks[successor_id]  # noqa: SLF001
    estimate.state = replace(
        estimate.state,
        residual_translational_rate_rad_s=(0.25, 0.0),
        bearing_rate_qualified=(True, True),
        bearing_std_rad=(0.02, 0.02),
        ambiguous=False,
    )
    normal_command = replace(
        source.command,
        target_roll_rad=0.60 * retained,
    )
    fresh = replace(
        source,
        current_gate_index=1,
        current_track_id=successor_id,
        successor_track_id=None,
        current_center_norm=(0.30, 0.0),
        proposed_command=normal_command,
        command=normal_command,
    )

    constrained = session._apply_post_credit_roll_reference_handoff(  # noqa: SLF001
        fresh
    )

    assert constrained.proposed_command.target_roll_rad == pytest.approx(
        0.60 * retained
    )
    assert constrained.command.target_roll_rad == pytest.approx(retained)
    assert math.isfinite(constrained.command.target_roll_rad)
    assert (
        abs(constrained.command.target_roll_rad)
        <= MAX_TARGET_ROLL_RAD
    )
    assert session.post_credit_roll_reference_handoff_active


def test_unqualified_post_credit_rate_retains_bank_only_until_expiry():
    session, source, successor_id, retained = (
        _activated_committed_roll_handoff()
    )
    estimate = session.core._tracks[successor_id]  # noqa: SLF001
    estimate.state = replace(
        estimate.state,
        residual_translational_rate_rad_s=(-0.05, 0.0),
        bearing_rate_qualified=(False, True),
        bearing_std_rad=(0.02, 0.02),
        ambiguous=False,
    )
    normal_command = replace(
        source.command,
        target_roll_rad=0.60 * retained,
    )
    unqualified = replace(
        source,
        current_gate_index=1,
        current_track_id=successor_id,
        successor_track_id=None,
        current_center_norm=(0.30, 0.0),
        proposed_command=normal_command,
        command=normal_command,
    )

    constrained = session._apply_post_credit_roll_reference_handoff(  # noqa: SLF001
        unqualified
    )

    assert constrained.command.target_roll_rad == pytest.approx(retained)
    assert math.isfinite(constrained.command.target_roll_rad)
    assert abs(constrained.command.target_roll_rad) <= MAX_TARGET_ROLL_RAD
    assert session.post_credit_roll_reference_handoff_active

    handoff = session._post_credit_roll_reference_handoff  # noqa: SLF001
    assert handoff is not None
    expired = replace(
        unqualified,
        monotonic_ns=handoff.expires_monotonic_ns + 1,
    )
    released = session._apply_post_credit_roll_reference_handoff(  # noqa: SLF001
        expired
    )

    assert released.command == normal_command
    assert not session.post_credit_roll_reference_handoff_active


@pytest.mark.parametrize(
    (
        "normal_roll_rad",
        "residual_rate_rad_s",
        "current_center_x",
    ),
    (
        (0.04, 0.25, 0.30),
        (-0.04, -0.05, 0.05),
    ),
)
def test_opposite_or_near_center_recovery_releases_roll_handoff_immediately(
    normal_roll_rad: float,
    residual_rate_rad_s: float,
    current_center_x: float,
):
    session, source, successor_id, _retained = (
        _activated_committed_roll_handoff()
    )
    estimate = session.core._tracks[successor_id]  # noqa: SLF001
    estimate.state = replace(
        estimate.state,
        residual_translational_rate_rad_s=(
            residual_rate_rad_s,
            0.0,
        ),
        bearing_rate_qualified=(True, True),
        bearing_std_rad=(0.02, 0.02),
        ambiguous=False,
    )
    normal_command = replace(
        source.command,
        target_roll_rad=normal_roll_rad,
    )
    fresh = replace(
        source,
        current_gate_index=1,
        current_track_id=successor_id,
        successor_track_id=None,
        current_center_norm=(current_center_x, 0.0),
        proposed_command=normal_command,
        command=normal_command,
    )

    released = session._apply_post_credit_roll_reference_handoff(  # noqa: SLF001
        fresh
    )

    assert released.command.target_roll_rad == pytest.approx(
        normal_roll_rad
    )
    assert session.post_credit_roll_reference_handoff_active is False


def test_dynamic_graph_adapter_releases_bias_after_safe_current_dwell():
    tracker, graph, snapshot, current_id = _graph()
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )

    seed = _observe(planner, snapshot, tracker)
    assert seed.servo_output.brake_reason == "continuity_seed"
    assert seed.servo_output.target_roll_rad == 0.0
    assert seed.servo_output.yaw_rate_rad_s == 0.0
    update = tracker.latest_update
    assert update is not None
    wire_ns = update.observation_monotonic_ns + 8_000_000
    session.record_wire_acceptance(
        target_roll_rad=0.0,
        target_pitch_rad=0.0,
        yaw_rate_rad_s=0.0,
        thrust=0.275,
        wire_command=AttitudeRateCommand(0.0, 0.0, 0.0, 0.275),
        wire_start_monotonic_ns=wire_ns,
    )

    update = tracker.update(
        _frame(6, current_width=0.355, current_height=0.375)
    )
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)
    assert proposal.servo_output.target_roll_rad == 0.0
    assert proposal.servo_output.yaw_rate_rad_s == 0.0
    assert session.last_decision is not None
    assert session.last_decision.passage_point_norm == (0.0, 0.0)
    assert session.last_decision.successor_prediction_confidence == 0.0
    _accept_proposal(session, tracker, proposal)

    for sequence in range(7, 19):
        growth = 0.015 * (sequence - 5)
        tracker.update(
            _frame(
                sequence,
                current_width=0.34 + growth,
                current_height=0.36 + growth,
            )
        )
        snapshot = graph.observe(tracker)
        proposal = _observe(planner, snapshot, tracker)
        _accept_proposal(session, tracker, proposal)

    assert proposal.servo_output.target_roll_rad < 0.0
    assert proposal.servo_output.target_pitch_rad <= 0.12
    assert proposal.servo_output.reviewed_next_track_id is not None
    assert session.last_decision is not None
    assert session.last_decision.current_gate_index == 0
    assert session.last_decision.successor_track_id is not None
    assert all(
        clearance > 0.0
        for clearance in (
            session.last_decision.centered_crossing_clearance_norm
        )
    )
    assert session.last_decision.successor_clearance_dwell_s > (
        session.core.config.successor_clearance_dwell_s
    )
    assert session.last_decision.successor_clearance_authority > 0.0
    assert session.last_decision.passage_point_norm[0] > 0.0
    assert session.last_decision.successor_weight > 0.0
    assert session.last_decision.passage_yaw_authority > 0.0
    assert session.last_decision.successor_yaw_contribution_rad > 0.0
    assert session.last_decision.successor_passage_authority > 0.0
    assert all(
        clearance > 0.0
        for clearance in (
            session.last_decision.predicted_crossing_clearance_norm
        )
    )
    assert session.evidence_summary()["controller_family"] == (
        DYNAMIC_CONTROLLER_FAMILY
    )


def test_graph_vetted_adjacent_rebinds_local_successor_before_race_promotion():
    tracker, graph, snapshot, current_id = _graph()
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)
    original_successor_id = session.core.course_state().successor_track_id
    assert original_successor_id is not None
    original_successor = session.core.course_state().successor
    assert original_successor is not None
    assert original_successor.aperture_half_size_norm is not None

    # Retire the old image identity during the current-gate crossing while
    # preserving its short clean-aperture state in the dynamic session.
    for sequence in range(6, 15):
        tracker.update(_frame(sequence, include_successor=False))
        snapshot = graph.observe(tracker)
        hold = _observe(planner, snapshot, tracker)
        _accept_proposal(session, tracker, hold)
    committed = session.last_decision
    assert committed is not None
    assert committed.successor_track_id == original_successor_id
    committed_roll = -MAX_TARGET_ROLL_RAD
    session._last_decision = replace(  # noqa: SLF001
        committed,
        passage_committed=True,
        committed_successor_roll_authority=1.0,
        committed_successor_target_roll_rad=committed_roll,
        command=replace(
            committed.command,
            target_roll_rad=committed_roll,
        ),
    )
    latest_update = tracker.latest_update
    assert latest_update is not None
    committed_wire_ns = (
        latest_update.observation_monotonic_ns + 9_000_000
    )
    session.record_wire_acceptance(
        target_roll_rad=committed_roll,
        target_pitch_rad=committed.command.target_pitch_rad,
        yaw_rate_rad_s=committed.command.yaw_rate_rad_s,
        thrust=committed.command.thrust,
        wire_command=AttitudeRateCommand(
            -0.10,
            0.0,
            committed.command.yaw_rate_rad_s,
            committed.command.thrust,
        ),
        wire_start_monotonic_ns=committed_wire_ns,
    )

    # A nearby stable degraded-inner replacement supplies graph-vetted
    # center/scale correction, but no new aperture or passage authority.
    for sequence, successor_center_x in zip(
        range(15, 19),
        (0.38, 0.40, 0.42, 0.44),
    ):
        degraded_inner = _inner_aperture(
            successor_center_x,
            0.0,
            half_width=0.15,
            half_height=0.17,
            confidence=0.18,
            health_reason="aperture_fit_low_confidence",
        )
        update = tracker.update(
            _frame(
                sequence,
                successor_center_x=successor_center_x,
                successor_inner_aperture=degraded_inner,
            )
        )
        snapshot = graph.observe(tracker)
    assert snapshot.next_selection_ambiguous is False
    assert snapshot.provisional_track_ids == ()
    assert len(snapshot.next_candidates) == 1
    replacement = snapshot.next_candidates[0]
    assert replacement.promotable
    assert replacement.track_id != original_successor_id

    adjacent = DynamicRollingVisualApproachServo(
        replacement.track_id,
        1,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    proposal = adjacent.observe_promotable_adjacent(
        snapshot,
        tracker,
        now_monotonic_s=(
            update.observation_monotonic_ns + 5_000_000
        )
        / 1_000_000_000.0,
        segment_elapsed_s=1.0,
        segment_yaw_excursion_rad=0.0,
    )
    rebound = session.core.course_state()

    assert rebound.current_gate_index == 0
    assert rebound.current_track_id == current_id
    assert rebound.successor_track_id == replacement.track_id
    assert rebound.promotion_count == 0
    assert rebound.successor is not None
    assert rebound.successor.aperture_half_size_norm is not None
    assert rebound.successor.aperture_propagated
    assert rebound.successor.aperture_seed_monotonic_ns == (
        original_successor.aperture_seed_monotonic_ns
    )
    assert proposal.mode is VisualApproachMode.ADJACENT_RECENTER
    assert proposal.next_target is None
    assert proposal.passage_admission is None
    assert proposal.servo_output.advance_enabled is False
    assert proposal.servo_output.next_gate_blend == 0.0
    assert proposal.servo_output.yaw_rate_rad_s < 0.0
    prediction = session.core.predict_track_steering(
        replacement.track_id,
        update.observation_monotonic_ns + 5_000_000,
    )
    baseline = session._successor_steering_targets(prediction)
    assert abs(float(baseline["target_roll_rad"])) < abs(
        committed_roll
    )
    assert proposal.servo_output.target_roll_rad == pytest.approx(
        committed_roll
    )
    assert proposal.servo_output.target_pitch_rad == pytest.approx(
        baseline["target_pitch_rad"]
    )
    assert proposal.servo_output.yaw_rate_rad_s == pytest.approx(
        baseline["yaw_rate_rad_s"]
    )
    assert proposal.servo_output.thrust == pytest.approx(
        baseline["thrust"]
    )
    assert all(
        math.isfinite(value)
        for value in (
            proposal.servo_output.target_roll_rad,
            proposal.servo_output.target_pitch_rad,
            proposal.servo_output.yaw_rate_rad_s,
            proposal.servo_output.thrust,
        )
    )
    assert (
        abs(proposal.servo_output.target_roll_rad)
        <= MAX_TARGET_ROLL_RAD
    )
    assert (
        abs(proposal.servo_output.yaw_rate_rad_s)
        <= MAX_YAW_RATE_RAD_S
    )

    _accept_proposal(session, tracker, proposal)
    accepted_reference = (
        session._precredit_successor_roll_reference  # noqa: SLF001
    )
    assert accepted_reference is not None
    assert accepted_reference.target_roll_rad == pytest.approx(
        committed_roll
    )
    assert accepted_reference.accepted_wire_start_monotonic_ns == (
        update.observation_monotonic_ns + 8_000_000
    )
    # A newer proposal can be refused atomically by race credit.  It must not
    # overwrite the last reference that actually reached the wire.
    no_wire = session.adjacent_precredit_successor_steering_authority(
        track_id=replacement.track_id,
        now_monotonic_ns=(
            update.observation_monotonic_ns + 9_000_000
        ),
    )
    assert no_wire["target_roll_rad"] == pytest.approx(committed_roll)
    assert (
        session._precredit_successor_roll_reference  # noqa: SLF001
        == accepted_reference
    )
    race_received_ns = update.observation_monotonic_ns + 10_000_000
    activation_ns = race_received_ns + 1_000_000
    session.activate_post_credit_successor_steering(
        _credited_race(race_received_ns),
        from_gate_index=0,
        reviewed_track_id=replacement.track_id,
        activation_monotonic_ns=activation_ns,
    )
    promoted = session.core.course_state()
    assert promoted.current_gate_index == 1
    assert promoted.current_track_id == replacement.track_id
    assert promoted.promotion_count == 1
    handoff = session._post_credit_roll_reference_handoff  # noqa: SLF001
    assert handoff is not None
    assert handoff.retained_target_roll_rad == pytest.approx(
        committed_roll
    )


def test_dynamic_adapter_preserves_bounded_outward_correction_demand():
    tracker, _graph_state, snapshot, current_id = _graph()
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)

    outputs = []
    for sequence, center_x in zip(
        range(6, 10),
        (0.10, 0.20, 0.30, 0.40),
    ):
        tracker.update(
            _frame(
                sequence,
                current_center_x=center_x,
                successor_center_x=0.52,
            )
        )
        snapshot = _graph_state.observe(tracker)
        proposal = _observe(planner, snapshot, tracker)
        outputs.append(proposal.servo_output)
        _accept_proposal(session, tracker, proposal)

    corrective = tuple(
        output.target_roll_rad
        for output in outputs
        if output.yaw_rate_rad_s < 0.0
    )
    assert corrective
    assert all(
        math.isfinite(target_roll)
        and -MAX_TARGET_ROLL_RAD <= target_roll < 0.0
        for target_roll in corrective
    )


@pytest.mark.parametrize("gate_size", (0.36, 0.90))
def test_dynamic_passage_admission_requires_observed_closure(
    gate_size: float,
):
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=gate_size,
        height=gate_size,
    )
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )

    proposal = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, proposal)
    for sequence in range(6, 11):
        tracker.update(
            _frame(
                sequence,
                current_width=gate_size,
                current_height=gate_size,
                include_successor=False,
            )
        )
        snapshot = graph.observe(tracker)
        proposal = _observe(planner, snapshot, tracker)
        _accept_proposal(session, tracker, proposal)

    assert proposal.passage_admission is None
    assert session.last_decision is not None
    assert session.last_decision.current_time_to_contact_s is None
    assert all(
        clearance > 0.0
        for clearance in (
            session.last_decision.terminal_crossing_clearance_norm
        )
    )
    assert proposal.servo_output.corridor_frames == 0


def test_terminal_positive_clearance_commits_before_fixed_scale() -> None:
    tracker, graph, snapshot, current_id = _graph()
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )

    proposal = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, proposal)
    for sequence in (6, 7, 8):
        tracker.update(
            _frame(
                sequence,
                current_width=0.34,
                current_height=0.36,
                include_successor=True,
            )
        )
        snapshot = graph.observe(tracker)
        proposal = _observe(planner, snapshot, tracker)
        _accept_proposal(session, tracker, proposal)

    assert planner.latched_next_track_id is not None
    frames = (
        (9, 0.38, 0.000),
        (10, 0.43, 0.012),
        (11, 0.49, 0.024),
        (12, 0.52, 0.036),
    )
    pre_scale_commitment = None
    for sequence, gate_size, center_y in frames:
        tracker.update(
            _frame(
                sequence,
                current_width=gate_size,
                current_height=gate_size,
                include_successor=False,
                current_center_y=center_y,
            )
        )
        snapshot = graph.observe(tracker)
        proposal = _observe(planner, snapshot, tracker)
        decision = session.last_decision
        assert decision is not None
        assert all(
            clearance >= 0.0
            for clearance in decision.terminal_crossing_clearance_norm
        )
        state = session.core.course_state().current
        scale_lower_bound = (
            state.log_scale - 2.0 * state.log_scale_std
        )
        if (
            proposal.passage_admission is not None
            and scale_lower_bound
            < session.core.config.passage_arm_min_log_scale
        ):
            pre_scale_commitment = (
                proposal,
                decision,
                scale_lower_bound,
            )
        _accept_proposal(session, tracker, proposal)

    assert pre_scale_commitment is not None
    committed, committed_decision, committed_scale_lower_bound = (
        pre_scale_commitment
    )
    assert committed.passage_admission is not None
    assert committed.servo_output.corridor_frames >= 3
    assert committed_decision.current_time_to_contact_s is not None
    assert all(
        clearance >= 0.0
        for clearance in (
            committed_decision.terminal_crossing_clearance_norm
        )
    )
    assert committed_scale_lower_bound < (
        session.core.config.passage_arm_min_log_scale
    )

    tracker.update(
        _frame(
            13,
            current_width=0.64,
            current_height=0.64,
            include_successor=False,
            current_center_y=0.048,
        )
    )
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)
    assert session.last_decision is not None
    assert all(
        clearance >= 0.0
        for clearance in (
            session.last_decision.terminal_crossing_clearance_norm
        )
    )
    assert proposal.current_target.normalized_y_rate_down_s > 0.0
    assert proposal.servo_output.brake_reason == "aligning"
    assert proposal.passage_admission is not None
    _accept_proposal(session, tracker, proposal)

    tracker.update(
        _frame(
            14,
            current_width=0.68,
            current_height=0.68,
            include_successor=False,
            current_center_y=0.010,
        )
    )
    snapshot = graph.observe(tracker)
    rejected = _observe(planner, snapshot, tracker)
    assert session.last_decision is not None
    assert all(
        clearance >= 0.0
        for clearance in (
            session.last_decision.terminal_crossing_clearance_norm
        )
    )
    assert rejected.current_target.normalized_y_rate_down_s < -0.30
    assert rejected.servo_output.corridor_frames == 0
    assert rejected.servo_output.brake_reason != "aligning"
    assert rejected.passage_admission is None


@pytest.mark.parametrize(
    ("seed_center_y", "admission_expected", "clipping"),
    (
        (
            0.10,
            True,
            FrameEdge.LEFT
            | FrameEdge.TOP
            | FrameEdge.RIGHT
            | FrameEdge.BOTTOM,
        ),
        (0.30, False, FrameEdge.TOP | FrameEdge.BOTTOM),
    ),
)
def test_propagated_aperture_mints_only_safe_passage_admission(
    seed_center_y: float,
    admission_expected: bool,
    clipping: FrameEdge,
) -> None:
    tracker, graph, snapshot, current_id = _graph()
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )

    proposal = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, proposal)
    for sequence in (6, 7, 8):
        tracker.update(
            _frame(
                sequence,
                current_width=0.34,
                current_height=0.36,
                include_successor=True,
            )
        )
        snapshot = graph.observe(tracker)
        proposal = _observe(planner, snapshot, tracker)
        _accept_proposal(session, tracker, proposal)

    for sequence, gate_size, center_y in (
        (9, 0.38, 0.000),
        (10, 0.43, 0.012),
    ):
        tracker.update(
            _frame(
                sequence,
                current_width=gate_size,
                current_height=gate_size,
                include_successor=False,
                current_center_y=center_y,
            )
        )
        snapshot = graph.observe(tracker)
        proposal = _observe(planner, snapshot, tracker)
        _accept_proposal(session, tracker, proposal)

    assert planner.latched_next_track_id is not None
    assert proposal.servo_output.corridor_frames == 0
    assert proposal.passage_admission is None

    tracker.update(
        _frame(
            11,
            current_width=0.49,
            current_height=0.49,
            include_successor=False,
            current_center_y=seed_center_y,
        )
    )
    snapshot = graph.observe(tracker)
    session.stage_snapshot(
        snapshot,
        tracker,
        expected_gate_index=0,
        expected_current_track_id=current_id,
        adjacent_precredit=False,
    )
    clean_seed = session.core.course_state().current
    assert not clean_seed.aperture_propagated
    assert clean_seed.aperture_half_size_norm is not None
    assert clean_seed.aperture_seed_monotonic_ns is not None
    assert clean_seed.aperture_dynamics_qualified
    # Staging the clean predictor seed did not execute the image servo.
    assert proposal.servo_output.corridor_frames == 0

    tracker.update(
        _frame(
            12,
            current_width=0.55,
            current_height=0.55,
            include_successor=False,
            current_center_y=seed_center_y,
            current_clipping=clipping,
            current_center_censored=True,
            current_inner_aperture=None,
        )
    )
    snapshot = graph.observe(tracker)
    propagated = _observe(planner, snapshot, tracker)
    state = session.core.course_state().current
    decision = session.last_decision
    assert decision is not None
    assert state.aperture_propagated
    assert state.aperture_dynamics_qualified
    assert decision.current_aperture_propagated
    assert decision.current_aperture_dynamics_qualified
    assert decision.current_time_to_contact_s is not None
    assert state.expansion_rate_s > 0.0
    assert all(
        allowance > 0.0
        for allowance in decision.crossing_allowance_norm
    )
    within_prediction_horizon = bool(
        decision.current_time_to_contact_s
        <= decision.current_aperture_prediction_horizon_remaining_s
    )
    terminal_safe = all(
        clearance >= 0.0
        for clearance in decision.terminal_crossing_clearance_norm
    )
    if admission_expected:
        assert within_prediction_horizon
        assert terminal_safe
        assert propagated.servo_output.corridor_frames >= 3
        assert propagated.passage_admission is not None
    else:
        assert not within_prediction_horizon or not terminal_safe
        assert propagated.servo_output.corridor_frames == 0
        assert propagated.passage_admission is None


def test_passage_retains_clean_seed_through_vertical_occlusion() -> None:
    tracker, graph, snapshot, current_id = _graph()
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )

    proposal = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, proposal)
    for sequence in (6, 7, 8):
        tracker.update(
            _frame(
                sequence,
                current_width=0.34,
                current_height=0.36,
                include_successor=True,
            )
        )
        snapshot = graph.observe(tracker)
        proposal = _observe(planner, snapshot, tracker)
        _accept_proposal(session, tracker, proposal)
    retained_id = planner.latched_next_track_id
    assert retained_id is not None
    assert session.core.course_state().successor.sample_count == 4

    sizes = {
        9: 0.38,
        10: 0.43,
        11: 0.49,
        12: 0.56,
        13: 0.64,
        14: 0.72,
    }
    first_admission = None
    for sequence, gate_size in sizes.items():
        tracker.update(
            _frame(
                sequence,
                current_width=gate_size,
                current_height=gate_size,
                include_successor=False,
            )
        )
        snapshot = graph.observe(tracker)
        proposal = _observe(planner, snapshot, tracker)
        if (
            first_admission is None
            and proposal.passage_admission is not None
        ):
            first_admission = proposal.passage_admission
            assert session.last_decision is not None
            assert all(
                clearance >= 0.0
                for clearance in (
                    session.last_decision
                    .terminal_crossing_clearance_norm
                )
            )
        _accept_proposal(session, tracker, proposal)

    admission = proposal.passage_admission
    successor = session.core.course_state().successor
    assert successor is not None
    assert successor.visible is False
    assert successor.missed_count == 6
    assert snapshot.next_candidates == ()
    assert first_admission is not None
    assert admission is not None
    assert admission.preview_track_id == retained_id
    assert admission.preview_blend == 0.0
    assert proposal.next_target is None
    assert proposal.servo_output.brake_reason == "aligning"
    assert proposal.servo_output.corridor_frames >= 3
    clean_state = session.core.course_state().current
    assert clean_state.aperture_half_size_norm is not None
    assert clean_state.aperture_seed_monotonic_ns is not None
    assert clean_state.aperture_prediction_deadline_monotonic_ns is not None
    assert not clean_state.aperture_propagated
    committed_corridor_frames = proposal.servo_output.corridor_frames

    tracker.update(
        _frame(
            15,
            current_width=0.80,
            current_height=0.80,
            include_successor=False,
            current_clipping=FrameEdge.TOP,
            current_center_censored=True,
            current_inner_aperture=_inner_aperture(
                0.0,
                0.0,
                half_width=0.38,
                half_height=0.38,
                health_reason="outer_support_clipped_tracking_only",
            ),
        )
    )
    snapshot = graph.observe(tracker)
    update = tracker.latest_update
    assert update is not None
    top_passage = planner.observe(
        snapshot,
        tracker,
        now_monotonic_s=(
            update.observation_monotonic_ns + 5_000_000
        )
        / 1_000_000_000.0,
        segment_elapsed_s=0.7,
        segment_yaw_excursion_rad=0.0,
        mode=VisualApproachMode.PASSAGE,
        passage_admission=admission,
    )
    top_state = session.core.course_state().current
    top_decision = session.last_decision
    assert top_decision is not None
    assert top_decision.passage_committed
    assert top_passage.passage_admission is admission
    assert top_passage.servo_output.corridor_frames == (
        committed_corridor_frames
    )
    assert top_state.aperture_propagated
    assert top_decision.current_aperture_propagated
    assert top_decision.current_aperture_half_size_norm is not None
    assert top_decision.current_aperture_prediction_horizon_remaining_s > 0.0
    assert top_state.aperture_seed_monotonic_ns == (
        clean_state.aperture_seed_monotonic_ns
    )
    assert top_state.aperture_prediction_deadline_monotonic_ns == (
        clean_state.aperture_prediction_deadline_monotonic_ns
    )
    _accept_proposal(session, tracker, top_passage)

    tracker.update(
        _frame(
            16,
            current_width=0.82,
            current_height=0.82,
            include_successor=False,
            current_clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
            current_center_censored=True,
        )
    )
    snapshot = graph.observe(tracker)
    update = tracker.latest_update
    assert update is not None
    vertically_censored_passage = planner.observe(
        snapshot,
        tracker,
        now_monotonic_s=(
            update.observation_monotonic_ns + 5_000_000
        )
        / 1_000_000_000.0,
        segment_elapsed_s=0.7,
        segment_yaw_excursion_rad=0.0,
        mode=VisualApproachMode.PASSAGE,
        passage_admission=admission,
    )
    vertically_censored_state = session.core.course_state().current
    vertically_censored_decision = session.last_decision
    assert vertically_censored_decision is not None
    assert vertically_censored_decision.passage_committed
    assert vertically_censored_passage.passage_admission is admission
    assert vertically_censored_passage.servo_output.corridor_frames == (
        committed_corridor_frames
    )
    assert vertically_censored_state.aperture_propagated
    assert vertically_censored_decision.current_aperture_propagated
    assert (
        vertically_censored_state.aperture_seed_monotonic_ns
        == clean_state.aperture_seed_monotonic_ns
    )
    assert (
        vertically_censored_state.aperture_prediction_deadline_monotonic_ns
        == clean_state.aperture_prediction_deadline_monotonic_ns
    )
    assert all(
        math.isfinite(value)
        for value in (
            vertically_censored_passage.servo_output.target_roll_rad,
            vertically_censored_passage.servo_output.target_pitch_rad,
            vertically_censored_passage.servo_output.yaw_rate_rad_s,
            vertically_censored_passage.servo_output.thrust,
        )
    )
    _accept_proposal(session, tracker, vertically_censored_passage)

    tracker.update(
        _frame(
            17,
            current_width=0.84,
            current_height=0.84,
            include_successor=False,
        )
    )
    snapshot = graph.observe(tracker)
    update = tracker.latest_update
    assert update is not None
    passage = planner.observe(
        snapshot,
        tracker,
        now_monotonic_s=(
            update.observation_monotonic_ns + 5_000_000
        )
        / 1_000_000_000.0,
        segment_elapsed_s=0.7,
        segment_yaw_excursion_rad=0.0,
        mode=VisualApproachMode.PASSAGE,
        passage_admission=admission,
    )

    assert passage.passage_admission == admission
    assert passage.next_target is None
    assert passage.latched_next_track_id == retained_id
    assert not session.core.course_state().current.aperture_propagated


def test_inner_aperture_not_outer_support_drives_controller_geometry() -> None:
    inner = _inner_aperture(
        0.08,
        -0.06,
        half_width=0.21,
        half_height=0.23,
    )
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.50,
        height=0.56,
        inner_aperture=inner,
    )
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )

    seed = _observe(planner, snapshot, tracker)
    state = next(
        state
        for state in session.core.track_states
        if state.track_id == current_id
    )
    assert state.raw_center_norm == pytest.approx((0.08, -0.06))
    assert state.aperture_half_size_norm == pytest.approx((0.21, 0.23))
    assert state.raw_log_scale == pytest.approx(inner.log_scale)
    _accept_proposal(session, tracker, seed)

    tracker.update(
        _frame(
            6,
            current_width=0.70,
            current_height=0.76,
            include_successor=False,
            current_inner_aperture=inner,
        )
    )
    snapshot = graph.observe(tracker)
    _observe(planner, snapshot, tracker)

    assert session.last_decision is not None
    passage_aperture = (
        session.last_decision.current_aperture_half_size_norm
    )
    assert passage_aperture is not None
    # The paired angular chart may reproject an off-axis vertical extent by
    # a small amount, but the inner aperture still owns the geometry and the
    # much larger outer contour cannot leak into passage clearance.
    assert passage_aperture[0] == pytest.approx(0.21, abs=0.003)
    assert passage_aperture[1] == pytest.approx(0.23, abs=0.003)
    assert session.last_decision.aperture_margin_norm == pytest.approx(
        tuple(
            value - session.core.config.passage_margin_norm
            for value in passage_aperture
        )
    )
    assert session.last_decision.camera_current_center_norm == pytest.approx(
        (0.08, -0.06)
    )


def test_degraded_inner_confidence_bounds_steering_from_clean_seed() -> None:
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.34,
        height=0.36,
    )
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)
    clean_state = session.core.course_state().current
    assert clean_state.raw_center_norm is not None
    assert clean_state.aperture_half_size_norm is not None
    assert clean_state.aperture_seed_monotonic_ns is not None
    assert clean_state.aperture_prediction_deadline_monotonic_ns is not None

    degraded = _inner_aperture(
        0.70,
        -0.80,
        half_width=0.27,
        half_height=0.30,
        confidence=0.0015,
        measurement_std=(0.52, 0.93, 8.0),
        health_reason="outer_support_clipped_tracking_only",
    )
    tracker.update(
        _frame(
            6,
            current_width=0.36,
            current_height=0.38,
            include_successor=False,
            current_clipping=FrameEdge.RIGHT,
            current_center_censored=True,
            current_inner_aperture=degraded,
        )
    )
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)

    state = session.core.course_state().current
    assert state.raw_center_norm == pytest.approx(degraded.center_norm)
    assert state.censored_axes == (False, False)
    assert all(
        uncertainty
        < session.core.config.successor_prediction_max_extrapolation_rad
        for uncertainty in state.bearing_std_rad
    )
    assert state.aperture_propagated
    assert state.aperture_half_size_norm is not None
    assert state.aperture_half_size_norm != pytest.approx(
        degraded.half_size_norm
    )
    assert state.aperture_seed_monotonic_ns == (
        clean_state.aperture_seed_monotonic_ns
    )
    assert state.aperture_prediction_deadline_monotonic_ns == (
        clean_state.aperture_prediction_deadline_monotonic_ns
    )
    assert abs(
        proposal.current_target.normalized_x
        - clean_state.raw_center_norm[0]
    ) < abs(degraded.center_norm[0] - clean_state.raw_center_norm[0])
    assert abs(
        proposal.current_target.normalized_y_down
        - clean_state.raw_center_norm[1]
    ) < abs(degraded.center_norm[1] - clean_state.raw_center_norm[1])
    assert proposal.current_target.clipped
    assert proposal.current_target.center_censored
    assert all(
        math.isfinite(value)
        for value in (
            proposal.servo_output.target_roll_rad,
            proposal.servo_output.target_pitch_rad,
            proposal.servo_output.yaw_rate_rad_s,
            proposal.servo_output.thrust,
        )
    )
    assert session.last_decision is not None
    assert session.last_decision.current_aperture_propagated
    assert (
        session.last_decision
        .current_aperture_prediction_horizon_remaining_s
        > 0.0
    )
    assert proposal.servo_output.corridor_frames == 0
    assert proposal.passage_admission is None


def test_propagated_current_fov_gap_authority_is_exact_and_steering_only() -> None:
    session, track, token, now_ns = _propagated_vertical_fov_gap()
    decision = session.last_decision
    assert decision is not None
    current = session.core.course_state().current
    assert current.aperture_half_size_norm is not None
    assert decision.current_aperture_half_size_norm is not None
    # Crossing clearance is deliberately not part of this raw-FOV ownership
    # proof.  A propagated state may steer through clipping, but it cannot
    # turn that steering evidence into passage or race-advance authority.
    session._last_decision = replace(  # noqa: SLF001 - exact contract probe
        decision,
        terminal_crossing_clearance_norm=(-0.04, -0.02),
    )

    authority = session.propagated_current_fov_gap_authority(
        track=track,
        camera_token=token,
        now_monotonic_ns=now_ns,
    )

    assert authority["track_id"] == track.track_id
    assert authority["camera_token"] == {
        "generation": token.generation,
        "frame_id": token.frame_id,
        "publication_sequence": token.publication_sequence,
        "stream_id": token.stream_id,
    }
    assert authority["tracker_frame_sequence"] == (
        track.history[-1].tracker_frame_sequence
    )
    assert authority["clipping"] == int(FrameEdge.BOTTOM)
    assert authority["aperture_prediction_horizon_remaining_s"] > 0.0
    assert all(
        math.isfinite(value) and value > 0.0
        for value in authority["aperture_half_size_norm"]
    )
    assert authority["aperture_half_size_norm"] == pytest.approx(
        decision.current_aperture_half_size_norm
    )
    assert authority["state_aperture_half_size_norm"] == pytest.approx(
        current.aperture_half_size_norm
    )
    assert all(
        math.isfinite(value) and value > 0.0
        for value in authority["camera_aperture_half_size_norm"]
    )
    assert any(
        abs(camera_value - state_value) > 1e-8
        for camera_value, state_value in zip(
            authority["camera_aperture_half_size_norm"],
            authority["state_aperture_half_size_norm"],
        )
    )
    assert all(
        math.isfinite(value) and value >= 0.0
        for value in authority["camera_center_std_norm"]
    )
    assert authority["aperture_log_scale_std"] >= 0.0
    assert authority["vertical_angle_scale_rad"] > 0.0
    assert all(
        math.isfinite(value)
        for value in (
            *authority["camera_center_norm"],
            *authority["body_to_reference_wxyz"],
        )
    )
    assert authority["terminal_crossing_clearance_norm"] == [
        -0.04,
        -0.02,
    ]
    assert authority["steering_only"] is True
    assert authority["passage_authority"] is False
    assert authority["advance_authority"] is False


def test_clipped_local_state_guides_after_aperture_authority_expires() -> None:
    tracker, graph, snapshot, current_id = _graph()
    session = _session(
        config=replace(
            production_dynamic_course_config(),
            crossing_prediction_max_horizon_s=0.10,
            # Exercise a clipped current-state uncertainty seed above the
            # successor extrapolation cap without escaping the current
            # estimator's own bounded bearing envelope.
            process_noise_bearing_rad_s=2.10,
        )
    )
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    proposal = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, proposal)
    tracker.update(
        _frame(
            6,
            include_successor=False,
            current_center_x=0.05,
        )
    )
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, proposal)
    tracker.update(
        _frame(
            7,
            include_successor=False,
            current_center_x=0.10,
            current_clipping=FrameEdge.RIGHT,
            current_center_censored=True,
            current_inner_aperture=None,
        )
    )
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, proposal)
    clipped_seed_std = (
        session.core.course_state().current.bearing_std_rad
    )
    assert max(clipped_seed_std) > (
        session.core.config.successor_prediction_max_extrapolation_rad
    )
    assert max(clipped_seed_std) < (
        session.core.config.max_abs_bearing_rad
    )

    missing = replace(
        _frame(8, include_successor=False),
        detections=(),
    )
    update = tracker.update(missing)
    snapshot = graph.observe(tracker)
    assert snapshot.current_track_id == current_id
    assert snapshot.withholding_reason == "current_track_not_visible"
    assert snapshot.authority_usable is False
    with pytest.raises(
        VisualApproachRefusal,
        match="withheld authoritative current-gate identity",
    ):
        _observe(planner, snapshot, tracker)

    authorities = []
    bearing_seed_ns = None
    bearing_deadline_ns = None
    last_measurement_ns = None
    previous_remaining_horizon_s = None
    for sequence in range(8, 18):
        if sequence != 8:
            update = tracker.update(
                replace(
                    _frame(sequence, include_successor=False),
                    detections=(),
                )
            )
            snapshot = graph.observe(tracker)
            with pytest.raises(
                VisualApproachRefusal,
                match="withheld authoritative current-gate identity",
            ):
                _observe(planner, snapshot, tracker)
        now_ns = update.observation_monotonic_ns + 5_000_000
        authority = session.propagated_current_visibility_gap_authority(
            track=tracker.track(current_id),
            camera_token=update.token,
            now_monotonic_ns=now_ns,
        )
        state = session.core.course_state().current
        authorities.append(authority)

        assert state.frame_sequence == update.tracker_frame_sequence
        assert state.visible is False
        assert state.missed_count == sequence - 7
        assert (
            state.aperture_prediction_deadline_monotonic_ns is None
            or state.aperture_prediction_deadline_monotonic_ns < now_ns
        )
        assert authority["camera_token"] == asdict(update.token)
        assert authority["missed_frame_count"] == sequence - 7
        assert authority["steering_prediction_horizon_remaining_s"] > 0.0
        if previous_remaining_horizon_s is not None:
            assert (
                authority["steering_prediction_horizon_remaining_s"]
                < previous_remaining_horizon_s
            )
        previous_remaining_horizon_s = authority[
            "steering_prediction_horizon_remaining_s"
        ]
        assert authority["current_aperture_half_size_norm"] is None
        assert all(
            math.isfinite(value)
            for value in (
                *authority["current_center_norm"],
                *authority["current_bearing_std_rad"],
                *authority["command"].values(),
            )
        )
        assert max(authority["current_bearing_std_rad"]) > (
            session.core.config
            .successor_prediction_max_extrapolation_rad
        )
        assert max(authority["current_bearing_std_rad"]) <= (
            session.core.config.max_abs_bearing_rad
        )
        assert (
            -MAX_TARGET_ROLL_RAD
            <= authority["command"]["target_roll_rad"]
            < 0.0
        )
        assert authority["steering_only"] is True
        assert authority["passage_authority"] is False
        assert authority["advance_authority"] is False
        if bearing_seed_ns is None:
            bearing_seed_ns = authority[
                "bearing_prediction_seed_monotonic_ns"
            ]
            bearing_deadline_ns = authority[
                "bearing_prediction_deadline_monotonic_ns"
            ]
            last_measurement_ns = authority[
                "last_measurement_monotonic_ns"
            ]
        else:
            assert authority[
                "bearing_prediction_seed_monotonic_ns"
            ] == bearing_seed_ns
            assert authority[
                "bearing_prediction_deadline_monotonic_ns"
            ] == bearing_deadline_ns
            assert authority[
                "last_measurement_monotonic_ns"
            ] == last_measurement_ns

    authority = authorities[-1]
    now_ns = update.observation_monotonic_ns + 5_000_000
    assert authority["missed_frame_count"] == 10
    assert authority["basis"] == (
        "propagated-current-visibility-gap-guidance-v2"
    )
    assert authority["last_visible_camera_token"] == asdict(
        tracker.track(current_id).latest_token
    )
    assert authority["last_visible_clipping"] == int(FrameEdge.RIGHT)
    assert authority["steering_prediction_deadline_basis"] == (
        "propagated-local-bearing-state-v1"
    )
    assert now_ns > authority["fallback_steering_deadline_monotonic_ns"]
    assert authority["bearing_prediction_deadline_monotonic_ns"] == (
        authority["bearing_prediction_seed_monotonic_ns"]
        + round(
            session.core.config
            .post_credit_current_prediction_max_horizon_s
            * 1_000_000_000.0
        )
    )

    last_visible = tracker.track(current_id).history[-1]
    unclipped_track = replace(
        tracker.track(current_id),
        clipping=FrameEdge.NONE,
        history=tracker.track(current_id).history[:-1]
        + (replace(last_visible, clipping=FrameEdge.NONE),),
    )
    with pytest.raises(DynamicCourseError, match="exact clipped miss"):
        session.propagated_current_visibility_gap_authority(
            track=unclipped_track,
            camera_token=update.token,
            now_monotonic_ns=now_ns,
        )

    deadline_ns = authority["steering_prediction_deadline_monotonic_ns"]
    with pytest.raises(DynamicCourseError, match="fresh local steering"):
        session.propagated_current_visibility_gap_authority(
            track=tracker.track(current_id),
            camera_token=update.token,
            now_monotonic_ns=deadline_ns + 1,
        )


def test_live_local_aperture_extends_clipped_steering_beyond_fallback() -> None:
    tracker, graph, snapshot, current_id = _graph()
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    proposal = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, proposal)
    tracker.update(
        _frame(
            6,
            include_successor=False,
            current_center_x=0.05,
        )
    )
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, proposal)
    tracker.update(
        _frame(
            7,
            include_successor=False,
            current_center_x=0.10,
            current_clipping=FrameEdge.RIGHT,
            current_center_censored=True,
            current_inner_aperture=None,
        )
    )
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, proposal)

    missing = replace(
        _frame(8, include_successor=False),
        detections=(),
    )
    update = tracker.update(missing)
    snapshot = graph.observe(tracker)
    with pytest.raises(
        VisualApproachRefusal,
        match="withheld authoritative current-gate identity",
    ):
        _observe(planner, snapshot, tracker)

    track = tracker.track(current_id)
    state = session.core.course_state().current
    last_visible = track.history[-1]
    fallback_deadline_ns = (
        last_visible.observation_monotonic_ns
        + round(session.core.config.dropout_hold_s * 1_000_000_000.0)
    )
    aperture_deadline_ns = (
        state.aperture_prediction_deadline_monotonic_ns
    )
    assert state.aperture_propagated
    assert state.aperture_half_size_norm is not None
    assert aperture_deadline_ns is not None
    assert aperture_deadline_ns > fallback_deadline_ns

    authority = session.propagated_current_visibility_gap_authority(
        track=track,
        camera_token=update.token,
        now_monotonic_ns=fallback_deadline_ns + 1_000_000,
    )

    assert authority["steering_prediction_deadline_basis"] == (
        "propagated-local-aperture-state-v1"
    )
    assert authority["fallback_steering_deadline_monotonic_ns"] == (
        fallback_deadline_ns
    )
    assert authority["aperture_prediction_deadline_monotonic_ns"] == (
        aperture_deadline_ns
    )
    assert authority["steering_prediction_deadline_monotonic_ns"] == (
        aperture_deadline_ns
    )
    assert authority["steering_prediction_horizon_remaining_s"] > 0.0
    assert authority["current_aperture_propagated"] is True
    assert authority["current_aperture_half_size_norm"] is not None
    assert all(
        math.isfinite(value)
        for value in authority["command"].values()
    )
    assert abs(authority["command"]["target_roll_rad"]) <= (
        MAX_TARGET_ROLL_RAD
    )
    assert (
        MIN_TARGET_PITCH_RAD
        <= authority["command"]["target_pitch_rad"]
        <= MAX_TARGET_PITCH_RAD
    )
    assert abs(authority["command"]["yaw_rate_rad_s"]) <= (
        MAX_YAW_RATE_RAD_S
    )
    assert (
        MIN_THRUST
        <= authority["command"]["thrust"]
        <= MAX_THRUST
    )
    assert authority["steering_only"] is True
    assert authority["passage_authority"] is False
    assert authority["advance_authority"] is False

    with pytest.raises(
        DynamicCourseError,
        match="fresh local steering",
    ):
        session.propagated_current_visibility_gap_authority(
            track=track,
            camera_token=update.token,
            now_monotonic_ns=aperture_deadline_ns + 1,
        )


def test_propagated_current_fov_gap_refuses_identity_and_frame_mismatch() -> None:
    session, track, token, now_ns = _propagated_vertical_fov_gap()
    sample = track.history[-1]
    wrong_frame_track = replace(
        track,
        history=track.history[:-1]
        + (
            replace(
                sample,
                tracker_frame_sequence=(
                    sample.tracker_frame_sequence + 1
                ),
            ),
        ),
    )

    for mismatched_track, mismatched_token in (
        (replace(track, track_id=f"{track.track_id}-other"), token),
        (wrong_frame_track, token),
        (
            track,
            replace(
                token,
                publication_sequence=token.publication_sequence + 1,
            ),
        ),
    ):
        with pytest.raises(
            DynamicCourseError,
            match="tracker publication",
        ):
            session.propagated_current_fov_gap_authority(
                track=mismatched_track,
                camera_token=mismatched_token,
                now_monotonic_ns=now_ns,
            )


def test_propagated_current_fov_gap_requires_calibrated_camera_boundary() -> None:
    invalid_config = replace(
        production_dynamic_course_config(),
        camera_to_body_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    session, track, token, now_ns = _propagated_vertical_fov_gap(
        config=invalid_config
    )

    with pytest.raises(DynamicCourseError, match="calibrated camera"):
        session.propagated_current_fov_gap_authority(
            track=track,
            camera_token=token,
            now_monotonic_ns=now_ns,
        )


def test_propagated_fov_gap_accepts_multiedge_and_refuses_ambiguity_expiry(
) -> None:
    session, track, token, now_ns = _propagated_vertical_fov_gap()
    ambiguous_track = replace(track, ambiguous=True)

    with pytest.raises(DynamicCourseError, match="unambiguous"):
        session.propagated_current_fov_gap_authority(
            track=ambiguous_track,
            camera_token=token,
            now_monotonic_ns=now_ns,
        )

    all_edges = (
        FrameEdge.LEFT
        | FrameEdge.TOP
        | FrameEdge.RIGHT
        | FrameEdge.BOTTOM
    )
    full_session, full_track, full_token, full_now_ns = (
        _propagated_vertical_fov_gap(clipping=all_edges)
    )
    authority = full_session.propagated_current_fov_gap_authority(
        track=full_track,
        camera_token=full_token,
        now_monotonic_ns=full_now_ns,
    )
    assert authority["clipping"] == int(all_edges)
    assert authority["steering_only"] is True
    assert authority["passage_authority"] is False
    assert authority["advance_authority"] is False

    deadline_ns = (
        session.core.course_state()
        .current.aperture_prediction_deadline_monotonic_ns
    )
    assert deadline_ns is not None
    with pytest.raises(DynamicCourseError, match="expired"):
        session.propagated_current_fov_gap_authority(
            track=track,
            camera_token=token,
            now_monotonic_ns=deadline_ns + 1,
        )


def test_propagated_current_fov_gap_refuses_clean_or_unseeded_aperture() -> None:
    for inner_aperture in (
        _AUTO_INNER_APERTURE,
        _rejected_inner_aperture(
            clipping=FrameEdge.NONE,
            health_reason="no-clean-inner-aperture-seed",
        ),
    ):
        tracker, graph, snapshot, current_id = _single_gate_graph(
            width=0.34,
            height=0.36,
            inner_aperture=inner_aperture,
        )
        session = _session()
        session.record_wire_acceptance(
            target_roll_rad=0.0,
            target_pitch_rad=0.0,
            yaw_rate_rad_s=0.0,
            thrust=0.275,
            wire_command=AttitudeRateCommand(
                0.0,
                0.0,
                0.0,
                0.275,
            ),
            wire_start_monotonic_ns=_BASE_NS,
        )
        planner = DynamicRollingVisualApproachServo(
            current_id,
            0,
            next_gate_blend=0.35,
            next_gate_blend_start_log_scale=-1.80,
            next_gate_blend_full_log_scale=-0.50,
            session=session,
        )
        _observe(planner, snapshot, tracker)
        update = tracker.latest_update
        assert update is not None
        decision = session.last_decision
        assert decision is not None

        with pytest.raises(
            DynamicCourseError,
            match="clean propagated aperture",
        ):
            session.propagated_current_fov_gap_authority(
                track=tracker.track(current_id),
                camera_token=update.token,
                now_monotonic_ns=decision.monotonic_ns,
            )


def test_clipped_outer_support_without_seed_steers_only_observable_axis() -> None:
    rejected = _rejected_inner_aperture(
        clipping=FrameEdge.NONE,
        health_reason="no-clean-inner-aperture-seed",
    )
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.34,
        height=0.36,
        inner_aperture=rejected,
    )
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)
    unseeded = session.core.course_state().current
    assert unseeded.aperture_half_size_norm is None
    assert unseeded.aperture_seed_monotonic_ns is None
    assert unseeded.aperture_prediction_deadline_monotonic_ns is None

    tracker.update(
        _frame(
            6,
            current_width=0.36,
            current_height=0.38,
            include_successor=False,
            current_center_x=0.12,
            current_center_y=-0.10,
            current_clipping=FrameEdge.TOP,
            current_center_censored=True,
        )
    )
    snapshot = graph.observe(tracker)

    proposal = _observe(planner, snapshot, tracker)
    degraded = session.core.course_state().current
    assert degraded.raw_center_norm == pytest.approx((0.12, -0.10))
    assert degraded.raw_log_scale is None
    assert degraded.censored_axes == (False, True)
    assert degraded.aperture_half_size_norm is None
    assert degraded.aperture_seed_monotonic_ns is None
    assert degraded.aperture_prediction_deadline_monotonic_ns is None
    assert not degraded.aperture_propagated
    decision = session.last_decision
    assert decision is not None
    assert decision.current_aperture_half_size_norm is None
    assert decision.crossing_allowance_norm == (0.0, 0.0)
    assert not proposal.current_target.horizontal_geometry_censored
    assert proposal.current_target.vertical_geometry_censored
    assert proposal.current_target.normalized_x == pytest.approx(0.12)
    assert abs(proposal.servo_output.yaw_rate_rad_s) > 0.0
    assert proposal.servo_output.corridor_frames == 0
    assert proposal.passage_admission is None
    assert all(
        math.isfinite(value)
        for value in (
            proposal.servo_output.target_roll_rad,
            proposal.servo_output.target_pitch_rad,
            proposal.servo_output.yaw_rate_rad_s,
            proposal.servo_output.thrust,
        )
    )


def test_unseeded_clipping_that_censors_both_axes_still_refuses() -> None:
    rejected = _rejected_inner_aperture(
        clipping=FrameEdge.NONE,
        health_reason="no-clean-inner-aperture-seed",
    )
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.34,
        height=0.36,
        inner_aperture=rejected,
    )
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)

    tracker.update(
        _frame(
            6,
            current_width=0.36,
            current_height=0.38,
            include_successor=False,
            current_clipping=FrameEdge.TOP | FrameEdge.RIGHT,
            current_center_censored=True,
        )
    )
    snapshot = graph.observe(tracker)

    with pytest.raises(VisualApproachCurrentGeometryUnavailable):
        _observe(planner, snapshot, tracker)
    refused = session.core.course_state().current
    assert refused.censored_axes == (True, True)
    assert refused.aperture_half_size_norm is None


def test_rejected_merged_inner_steers_without_adding_corridor() -> None:
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.34,
        height=0.36,
    )
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)
    clean_state = session.core.course_state().current
    assert clean_state.aperture_half_size_norm is not None

    rejected = _rejected_inner_aperture(
        clipping=FrameEdge.NONE,
        health_reason="merged-current-successor-contour",
    )
    update = tracker.update(
        _frame(
            6,
            current_width=0.48,
            current_height=0.52,
            include_successor=False,
            current_inner_aperture=rejected,
        )
    )
    snapshot = graph.observe(tracker)

    proposal = _observe(planner, snapshot, tracker)

    retained = tracker.track(current_id)
    assert retained.visible
    assert retained.latest_token == update.token
    assert retained.history[-1].inner_aperture == rejected
    state = session.core.course_state().current
    assert state.aperture_propagated
    assert state.aperture_half_size_norm == pytest.approx(
        clean_state.aperture_half_size_norm
    )
    assert state.aperture_seed_monotonic_ns == (
        clean_state.aperture_seed_monotonic_ns
    )
    assert state.aperture_prediction_deadline_monotonic_ns == (
        clean_state.aperture_prediction_deadline_monotonic_ns
    )
    assert not state.ambiguous
    assert state.censored_axes == (False, False)
    assert state.raw_log_scale is None
    assert state.log_scale == pytest.approx(clean_state.log_scale)
    assert session.last_decision is not None
    assert session.last_decision.current_aperture_propagated
    assert (
        session.last_decision.current_aperture_half_size_norm
        == pytest.approx(clean_state.aperture_half_size_norm)
    )
    assert session.last_decision.successor_passage_authority == 0.0
    assert proposal.servo_output.corridor_frames == 0
    assert proposal.passage_admission is None


def test_degraded_fitted_inner_updates_state_without_crossing_authority() -> None:
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.34,
        height=0.36,
    )
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)
    prior = session.core.course_state().current

    degraded = _inner_aperture(
        0.10,
        -0.08,
        half_width=0.26,
        half_height=0.28,
        confidence=0.20,
        health_reason="low-confidence-inner-fit",
    )
    assert degraded.fitted
    assert degraded.complete_visibility
    assert not degraded.passage_usable
    tracker.update(
        _frame(
            6,
            current_width=0.40,
            current_height=0.42,
            include_successor=False,
            current_inner_aperture=degraded,
        )
    )
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)

    state = session.core.course_state().current
    assert state.raw_center_norm == pytest.approx(degraded.center_norm)
    assert state.raw_log_scale == pytest.approx(degraded.log_scale)
    assert state.bearing_rad[0] > prior.bearing_rad[0]
    assert state.bearing_rad[1] < prior.bearing_rad[1]
    assert state.log_scale != pytest.approx(prior.log_scale)
    assert state.expansion_rate_s == pytest.approx(
        prior.expansion_rate_s
    )
    assert state.censored_axes == (False, False)
    assert state.aperture_propagated
    assert state.aperture_half_size_norm is not None
    assert state.aperture_half_size_norm != pytest.approx(
        degraded.half_size_norm
    )
    assert state.aperture_seed_monotonic_ns == (
        prior.aperture_seed_monotonic_ns
    )
    assert state.aperture_prediction_deadline_monotonic_ns == (
        prior.aperture_prediction_deadline_monotonic_ns
    )
    assert session.last_decision is not None
    assert session.last_decision.current_aperture_propagated
    assert session.last_decision.current_aperture_half_size_norm is not None
    assert session.last_decision.successor_passage_authority == 0.0
    assert proposal.servo_output.corridor_frames == 0
    assert proposal.passage_admission is None


def test_rehydrated_inner_lineage_guides_multiedge_clip_without_passage() -> None:
    config = replace(
        production_dynamic_course_config(),
        camera_delay_s=0.0,
        crossing_prediction_max_horizon_s=0.05,
    )
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.34,
        height=0.36,
    )
    session = _session(config=config)
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)
    clean = session.core.course_state().current
    assert clean.aperture_prediction_deadline_monotonic_ns is not None

    degraded = _inner_aperture(
        0.10,
        -0.08,
        half_width=0.36,
        half_height=0.38,
        confidence=0.20,
        health_reason="low-confidence-inner-fit",
    )
    update = tracker.update(
        _frame(
            8,
            current_width=0.40,
            current_height=0.42,
            include_successor=False,
            current_center_x=0.10,
            current_center_y=-0.08,
            current_inner_aperture=degraded,
        )
    )
    assert (
        update.observation_monotonic_ns
        > clean.aperture_prediction_deadline_monotonic_ns
    )
    snapshot = graph.observe(tracker)
    corrected_proposal = _observe(planner, snapshot, tracker)
    rehydrated = session.core.course_state().current
    assert rehydrated.aperture_half_size_norm is not None
    assert rehydrated.aperture_propagated
    assert not rehydrated.aperture_dynamics_qualified
    assert corrected_proposal.passage_admission is None
    deadline_ns = rehydrated.aperture_prediction_deadline_monotonic_ns
    assert deadline_ns is not None
    current_track = tracker.track(current_id)
    fresh_outer_fallback = planner._target(
        current_track,
        now_monotonic_s=(deadline_ns + 1) / 1_000_000_000.0,
        require_current_authority=True,
    )
    assert fresh_outer_fallback.normalized_x == pytest.approx(
        current_track.center_norm[0]
    )
    assert fresh_outer_fallback.normalized_y_down == pytest.approx(
        current_track.center_norm[1]
    )
    _accept_proposal(session, tracker, corrected_proposal)

    tracker.update(
        _frame(
            9,
            current_width=0.42,
            current_height=0.44,
            include_successor=False,
            current_center_x=0.12,
            current_center_y=-0.10,
            current_clipping=FrameEdge.TOP | FrameEdge.RIGHT,
            current_center_censored=True,
            current_inner_aperture=None,
        )
    )
    snapshot = graph.observe(tracker)
    clipped_proposal = _observe(planner, snapshot, tracker)
    propagated = session.core.course_state().current
    output = clipped_proposal.servo_output

    assert propagated.censored_axes == (True, True)
    assert propagated.aperture_half_size_norm is not None
    assert propagated.aperture_propagated
    assert clipped_proposal.current_target.horizontal_geometry_censored
    assert clipped_proposal.current_target.vertical_geometry_censored
    assert clipped_proposal.passage_admission is None
    assert all(
        math.isfinite(value)
        for value in (
            output.target_roll_rad,
            output.target_pitch_rad,
            output.yaw_rate_rad_s,
            output.thrust,
        )
    )
    assert abs(output.target_roll_rad) <= MAX_TARGET_ROLL_RAD
    assert MIN_TARGET_PITCH_RAD <= output.target_pitch_rad <= MAX_TARGET_PITCH_RAD
    assert abs(output.yaw_rate_rad_s) <= MAX_YAW_RATE_RAD_S
    assert MIN_THRUST <= output.thrust <= MAX_THRUST


@pytest.mark.parametrize(
    (
        "clipping",
        "expected_censored_axes",
        "projected_center",
        "observable_axis",
        "propagated_axis",
    ),
    (
        (
            FrameEdge.RIGHT,
            (True, False),
            (0.80, -1.40),
            1,
            0,
        ),
        (
            FrameEdge.TOP,
            (False, True),
            (1.40, -0.80),
            0,
            1,
        ),
    ),
)
def test_one_axis_clip_keeps_fresh_observable_coordinate(
    monkeypatch: pytest.MonkeyPatch,
    clipping: FrameEdge,
    expected_censored_axes: tuple[bool, bool],
    projected_center: tuple[float, float],
    observable_axis: int,
    propagated_axis: int,
) -> None:
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.34,
        height=0.36,
    )
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)

    tracker.update(
        _frame(
            6,
            current_width=0.36,
            current_height=0.38,
            include_successor=False,
            current_center_x=0.12,
            current_center_y=0.19,
            current_clipping=clipping,
            current_center_censored=True,
        )
    )
    snapshot = graph.observe(tracker)
    decision_geometry = session.core._decision_geometry

    def off_frame_on_observable_axis(
        track_id: str,
        monotonic_ns: int,
    ):
        _, aperture = decision_geometry(track_id, monotonic_ns)
        return projected_center, aperture

    monkeypatch.setattr(
        session.core,
        "_decision_geometry",
        off_frame_on_observable_axis,
    )
    proposal = _observe(planner, snapshot, tracker)

    state = session.core.course_state().current
    raw_center = tracker.track(current_id).center_norm
    target_center = (
        proposal.current_target.normalized_x,
        proposal.current_target.normalized_y_down,
    )
    assert state.censored_axes == expected_censored_axes
    assert abs(projected_center[observable_axis]) > 1.25
    assert target_center[observable_axis] == pytest.approx(
        raw_center[observable_axis]
    )
    assert target_center[propagated_axis] == pytest.approx(
        projected_center[propagated_axis]
    )
    assert all(abs(value) <= 1.25 for value in target_center)
    assert session.last_decision is not None
    assert session.last_decision.camera_current_center_norm == pytest.approx(
        projected_center
    )
    assert proposal.servo_output.corridor_frames == 0
    assert proposal.passage_admission is None
    assert all(
        math.isfinite(value)
        for value in (
            proposal.servo_output.target_roll_rad,
            proposal.servo_output.target_pitch_rad,
            proposal.servo_output.yaw_rate_rad_s,
            proposal.servo_output.thrust,
        )
    )
    assert (
        abs(proposal.servo_output.target_roll_rad)
        <= MAX_TARGET_ROLL_RAD
    )
    assert (
        MIN_TARGET_PITCH_RAD
        <= proposal.servo_output.target_pitch_rad
        <= MAX_TARGET_PITCH_RAD
    )
    assert (
        abs(proposal.servo_output.yaw_rate_rad_s)
        <= MAX_YAW_RATE_RAD_S
    )
    assert MIN_THRUST <= proposal.servo_output.thrust <= MAX_THRUST


def test_clipped_outer_degraded_inner_steers_without_scale_authority() -> None:
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.34,
        height=0.36,
        inner_aperture=None,
    )
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)
    prior = session.core.course_state().current
    assert prior.aperture_half_size_norm is None

    degraded = _inner_aperture(
        0.15,
        -0.10,
        half_width=0.07,
        half_height=0.05,
        confidence=0.08,
        measurement_std=(0.025, 0.044, 0.20),
        health_reason="outer_support_clipped_tracking_only",
    )
    assert not degraded.passage_usable
    tracker.update(
        _frame(
            6,
            current_width=0.28,
            current_height=0.30,
            include_successor=False,
            current_center_x=0.12,
            current_center_y=-0.08,
            current_clipping=FrameEdge.RIGHT,
            current_center_censored=True,
            current_inner_aperture=degraded,
        )
    )
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)

    state = session.core.course_state().current
    assert state.raw_center_norm == pytest.approx(degraded.center_norm)
    assert state.raw_log_scale == pytest.approx(degraded.log_scale)
    assert state.aperture_half_size_norm is None
    assert state.censored_axes == (False, False)
    assert proposal.current_target.normalized_x == pytest.approx(
        degraded.center_norm[0]
    )
    assert proposal.current_target.normalized_y_down == pytest.approx(
        degraded.center_norm[1]
    )
    assert proposal.current_target.log_scale == pytest.approx(
        state.log_scale
    )
    assert proposal.current_target.log_scale != pytest.approx(
        degraded.log_scale
    )
    assert proposal.servo_output.corridor_frames == 0
    assert proposal.passage_admission is None
    assert all(
        math.isfinite(value)
        for value in (
            proposal.servo_output.target_roll_rad,
            proposal.servo_output.target_pitch_rad,
            proposal.servo_output.yaw_rate_rad_s,
            proposal.servo_output.thrust,
        )
    )


def test_rejected_inner_outer_support_corrects_steering_only() -> None:
    tracker, graph, snapshot, current_id = _single_gate_graph(
        width=0.34,
        height=0.36,
    )
    session = _session()
    planner = DynamicRollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
        next_gate_blend_start_log_scale=-1.80,
        next_gate_blend_full_log_scale=-0.50,
        session=session,
    )
    seed = _observe(planner, snapshot, tracker)
    _accept_proposal(session, tracker, seed)
    prior = session.core.course_state().current

    rejected = _rejected_inner_aperture(
        clipping=FrameEdge.NONE,
        health_reason="no-supported-inner-quadrilateral",
    )
    tracker.update(
        _frame(
            6,
            current_width=0.48,
            current_height=0.50,
            include_successor=False,
            current_center_x=0.12,
            current_center_y=-0.10,
            current_inner_aperture=rejected,
        )
    )
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)

    retained = tracker.track(current_id)
    assert retained.visible
    assert retained.center_norm == pytest.approx((0.12, -0.10))
    state = session.core.course_state().current
    assert state.bearing_rad[0] > prior.bearing_rad[0]
    assert state.bearing_rad[1] < prior.bearing_rad[1]
    assert state.log_scale == pytest.approx(prior.log_scale)
    assert state.raw_log_scale is None
    assert state.censored_axes == (False, False)
    assert not state.scale_rate_qualified
    assert state.aperture_propagated
    assert state.aperture_half_size_norm == pytest.approx(
        prior.aperture_half_size_norm
    )
    assert state.aperture_seed_monotonic_ns == (
        prior.aperture_seed_monotonic_ns
    )
    assert state.aperture_prediction_deadline_monotonic_ns == (
        prior.aperture_prediction_deadline_monotonic_ns
    )
    assert session.last_decision is not None
    assert session.last_decision.current_aperture_propagated
    assert (
        session.last_decision.current_aperture_half_size_norm
        == pytest.approx(prior.aperture_half_size_norm)
    )
    assert proposal.current_target.normalized_x > prior.raw_center_norm[0]
    assert (
        proposal.current_target.normalized_y_down
        < prior.raw_center_norm[1]
    )
    assert proposal.servo_output.corridor_frames == 0
    assert proposal.passage_admission is None


def test_opposite_valid_demand_reduces_applied_roll_on_next_wire_tick():
    session = _session()
    first_ns = _BASE_NS
    accepted = AttitudeRateCommand(0.12, 0.0, 0.0, 0.275)
    session.record_wire_acceptance(
        target_roll_rad=0.08,
        target_pitch_rad=0.0,
        yaw_rate_rad_s=0.0,
        thrust=0.275,
        wire_command=accepted,
        wire_start_monotonic_ns=first_ns,
    )
    proposed = session.govern_wire_command(
        AttitudeRateCommand(-0.25, 0.0, 0.0, 0.275),
        proposal_monotonic_ns=first_ns + 20_000_000,
        launch_thrust_override=False,
        yaw_safety_override=False,
    )

    assert proposed.roll_rate < accepted.roll_rate
    assert all(
        math.isfinite(value)
        for value in (
            proposed.roll_rate,
            proposed.pitch_rate,
            proposed.yaw_rate,
            proposed.thrust,
        )
    )
    assert abs(proposed.roll_rate) <= 0.25
    assert abs(proposed.pitch_rate) <= 0.25
    assert abs(proposed.yaw_rate) <= 0.15
    assert 0.21 <= proposed.thrust <= 0.32

    next_ns = first_ns + 20_000_000
    session.record_wire_acceptance(
        target_roll_rad=-0.08,
        target_pitch_rad=0.0,
        yaw_rate_rad_s=0.0,
        thrust=0.275,
        wire_command=proposed,
        wire_start_monotonic_ns=next_ns,
    )
    authority = session.continuity_hold_authority(
        now_monotonic_ns=next_ns,
        maximum_age_s=0.12,
    )
    assert authority["wire_command"] == proposed


def test_top_fov_protected_pitch_reduces_wire_pitch_on_next_tick():
    session = _session()
    first_ns = _BASE_NS
    accepted = AttitudeRateCommand(-0.20, 0.066, -0.03, 0.275)
    session.record_wire_acceptance(
        target_roll_rad=-0.25,
        target_pitch_rad=0.120,
        yaw_rate_rad_s=-0.03,
        thrust=0.275,
        wire_command=accepted,
        wire_start_monotonic_ns=first_ns,
    )

    proposed = session.govern_wire_command(
        AttitudeRateCommand(-0.25, -0.25, -0.15, 0.275),
        proposal_monotonic_ns=first_ns + 20_000_000,
        launch_thrust_override=False,
        yaw_safety_override=False,
    )

    assert proposed.pitch_rate < accepted.pitch_rate
    assert all(
        math.isfinite(value)
        for value in (
            proposed.roll_rate,
            proposed.pitch_rate,
            proposed.yaw_rate,
            proposed.thrust,
        )
    )
    assert abs(proposed.roll_rate) <= 0.25
    assert abs(proposed.pitch_rate) <= 0.25
    assert abs(proposed.yaw_rate) <= 0.15
    assert 0.21 <= proposed.thrust <= 0.32


def test_launch_thrust_override_does_not_seed_an_outward_slew():
    session = _session()
    first_ns = _BASE_NS
    session.record_wire_acceptance(
        target_roll_rad=0.0,
        target_pitch_rad=0.0,
        yaw_rate_rad_s=0.0,
        thrust=0.26,
        wire_command=AttitudeRateCommand(0.0, 0.0, 0.0, 0.26),
        wire_start_monotonic_ns=first_ns,
        thrust_slew_override=True,
    )
    session.record_wire_acceptance(
        target_roll_rad=0.0,
        target_pitch_rad=0.0,
        yaw_rate_rad_s=0.0,
        thrust=0.32,
        wire_command=AttitudeRateCommand(0.0, 0.0, 0.0, 0.32),
        wire_start_monotonic_ns=first_ns + 30_000_000,
        thrust_slew_override=True,
    )

    resumed = session.govern_wire_command(
        AttitudeRateCommand(0.0, 0.0, 0.0, 0.275),
        proposal_monotonic_ns=first_ns + 60_000_000,
        launch_thrust_override=False,
        yaw_safety_override=False,
    )

    assert 0.275 <= resumed.thrust <= 0.32


def test_proved_collective_descends_through_wire_governor_after_boost():
    session = _session()
    first_ns = _BASE_NS
    session.record_wire_acceptance(
        target_roll_rad=0.0,
        target_pitch_rad=0.0,
        yaw_rate_rad_s=0.0,
        thrust=0.32,
        wire_command=AttitudeRateCommand(0.0, 0.0, 0.0, 0.32),
        wire_start_monotonic_ns=first_ns,
        thrust_slew_override=True,
    )

    accepted = AttitudeRateCommand(0.0, 0.0, 0.0, 0.32)
    for index in range(1, 21):
        wire_ns = first_ns + index * 30_000_000
        accepted = session.govern_wire_command(
            AttitudeRateCommand(0.0, 0.0, 0.0, 0.21),
            proposal_monotonic_ns=wire_ns,
            launch_thrust_override=False,
            yaw_safety_override=False,
        )
        session.record_wire_acceptance(
            target_roll_rad=0.0,
            target_pitch_rad=0.0,
            yaw_rate_rad_s=0.0,
            thrust=accepted.thrust,
            wire_command=accepted,
            wire_start_monotonic_ns=wire_ns,
        )

    assert 0.21 <= accepted.thrust < 0.275


def test_ea6335c3_censored_coast_keeps_ramping_to_retained_collective():
    """Replay the lagging wire state that was frozen before the top hit."""

    session = _session()
    anchor_ns = _BASE_NS
    previous_ns = anchor_ns - 31_000_000
    session.record_wire_acceptance(
        target_roll_rad=0.1087756,
        target_pitch_rad=0.12,
        yaw_rate_rad_s=-0.0288209,
        thrust=0.2771589,
        wire_command=AttitudeRateCommand(
            0.0338,
            0.0732,
            -0.02138,
            0.2771589,
        ),
        wire_start_monotonic_ns=previous_ns,
    )
    anchor = AttitudeRateCommand(
        0.02846,
        0.07007,
        -0.0288209,
        0.2725533560536844,
    )
    session.record_wire_acceptance(
        target_roll_rad=0.1087756,
        target_pitch_rad=0.12,
        yaw_rate_rad_s=-0.0288209,
        thrust=anchor.thrust,
        wire_command=anchor,
        wire_start_monotonic_ns=anchor_ns,
    )

    accepted = anchor
    thrusts = []
    for elapsed_ms in (64, 95, 126, 191, 221, 253):
        wire_ns = anchor_ns + elapsed_ms * 1_000_000
        accepted = session.govern_wire_command(
            AttitudeRateCommand(
                anchor.roll_rate,
                anchor.pitch_rate,
                anchor.yaw_rate,
                0.21,
            ),
            proposal_monotonic_ns=wire_ns,
            launch_thrust_override=False,
            yaw_safety_override=False,
        )
        session.record_wire_acceptance(
            target_roll_rad=0.1087756,
            target_pitch_rad=0.12,
            yaw_rate_rad_s=accepted.yaw_rate,
            thrust=accepted.thrust,
            wire_command=accepted,
            wire_start_monotonic_ns=wire_ns,
            thrust_slew_override=False,
        )
        thrusts.append(accepted.thrust)

    assert thrusts == sorted(thrusts, reverse=True)
    assert all(
        later < earlier
        for earlier, later in zip(
            (anchor.thrust, *thrusts[:-1]),
            thrusts,
        )
    )
    assert 0.21 < thrusts[-1] < thrusts[0] < anchor.thrust
    assert accepted.roll_rate == pytest.approx(anchor.roll_rate)
    assert accepted.pitch_rate == pytest.approx(anchor.pitch_rate)
    assert accepted.yaw_rate == pytest.approx(anchor.yaw_rate)


def test_final_wire_governor_projects_yaw_momentum_into_measured_envelope():
    session = _session()
    first_ns = _BASE_NS
    for index, yaw in enumerate((-0.126, -0.138, -0.148)):
        session.record_wire_acceptance(
            target_roll_rad=0.0,
            target_pitch_rad=0.0,
            yaw_rate_rad_s=yaw,
            thrust=0.275,
            wire_command=AttitudeRateCommand(0.0, 0.0, yaw, 0.275),
            wire_start_monotonic_ns=first_ns + index * 30_000_000,
        )

    resumed = session.govern_wire_command(
        AttitudeRateCommand(0.0, 0.0, -0.140, 0.275),
        proposal_monotonic_ns=first_ns + 90_000_000,
        launch_thrust_override=False,
        yaw_safety_override=False,
    )

    assert -0.15 <= resumed.yaw_rate <= 0.0


def test_continuity_hold_is_fresh_bounded_and_uses_last_wire_target():
    session = _session()
    wire_ns = _BASE_NS
    command = AttitudeRateCommand(0.04, -0.02, -0.05, 0.275)
    session.record_wire_acceptance(
        target_roll_rad=0.05,
        target_pitch_rad=-0.03,
        yaw_rate_rad_s=-0.05,
        thrust=0.275,
        wire_command=command,
        wire_start_monotonic_ns=wire_ns,
    )
    authority = session.continuity_hold_authority(
        now_monotonic_ns=wire_ns + 100_000_000,
        maximum_age_s=0.12,
    )

    assert authority["wire_command"] == command
    assert authority["target_roll_rad"] == 0.05
    with pytest.raises(
        DynamicCourseError,
        match="fresh applied command",
    ):
        session.continuity_hold_authority(
            now_monotonic_ns=wire_ns + 121_000_000,
            maximum_age_s=0.12,
        )


def test_post_credit_activation_accepts_causal_wire_after_race_ingress():
    session, successor_id = _bound_post_credit_successor()
    successor = session.core.course_state().successor
    assert successor is not None
    race_received_ns = successor.state_monotonic_ns - 1_000_000
    wire_ns = successor.state_monotonic_ns + 2_000_000
    activation_ns = wire_ns + 2_000_000
    command = AttitudeRateCommand(0.03, 0.02, -0.04, 0.275)
    session.record_wire_acceptance(
        target_roll_rad=0.05,
        target_pitch_rad=0.04,
        yaw_rate_rad_s=-0.04,
        thrust=0.275,
        wire_command=command,
        wire_start_monotonic_ns=wire_ns,
    )

    evidence = session.activate_post_credit_successor_steering(
        _credited_race(race_received_ns),
        from_gate_index=0,
        reviewed_track_id=successor_id,
        activation_monotonic_ns=activation_ns,
    )
    authority = session.post_credit_successor_steering_authority(
        now_monotonic_ns=activation_ns,
    )

    assert wire_ns > race_received_ns
    assert evidence["activation_monotonic_ns"] == activation_ns
    assert session.core.course_state().current_track_id == successor_id
    assert authority["source_wire_start_monotonic_ns"] == wire_ns
    assert authority["wire_command"] == command
    assert authority["steering_only"] is True
    assert authority["passage_authority"] is False
    assert authority["advance_authority"] is False
    assert authority["target_pitch_rad"] == pytest.approx(
        session.core.config.brake_pitch_rad
    )
    assert authority["camera_elevation_error_rad"] == pytest.approx(0.0)
    assert authority["camera_elevation_rate_rad_s"] == pytest.approx(0.0)
    assert authority["pitch_delay_lead_rad"] == pytest.approx(0.0)
    assert session.post_credit_roll_reference_handoff_active is False

    accepted_ns = activation_ns + 1_000_000
    session.record_wire_acceptance(
        target_roll_rad=float(authority["target_roll_rad"]),
        target_pitch_rad=float(authority["target_pitch_rad"]),
        yaw_rate_rad_s=float(authority["yaw_rate_rad_s"]),
        thrust=float(authority["thrust"]),
        wire_command=AttitudeRateCommand(
            0.04,
            0.02,
            float(authority["yaw_rate_rad_s"]),
            float(authority["thrust"]),
        ),
        wire_start_monotonic_ns=accepted_ns,
    )

    assert session.post_credit_roll_reference_handoff_active is True
    handoff = session._post_credit_roll_reference_handoff  # noqa: SLF001
    assert handoff is not None
    assert handoff.retained_target_roll_rad == pytest.approx(
        authority["target_roll_rad"]
    )
    assert handoff.source_wire_start_monotonic_ns == accepted_ns


def test_unaccepted_post_credit_roll_target_cannot_create_handoff():
    session, successor_id = _bound_post_credit_successor()
    successor = session.core.course_state().successor
    assert successor is not None
    race_received_ns = successor.state_monotonic_ns - 1_000_000
    wire_ns = successor.state_monotonic_ns + 2_000_000
    activation_ns = wire_ns + 2_000_000
    session.record_wire_acceptance(
        target_roll_rad=0.05,
        target_pitch_rad=0.04,
        yaw_rate_rad_s=-0.04,
        thrust=0.275,
        wire_command=AttitudeRateCommand(0.03, 0.02, -0.04, 0.275),
        wire_start_monotonic_ns=wire_ns,
    )
    session.activate_post_credit_successor_steering(
        _credited_race(race_received_ns),
        from_gate_index=0,
        reviewed_track_id=successor_id,
        activation_monotonic_ns=activation_ns,
    )
    authority = session.post_credit_successor_steering_authority(
        now_monotonic_ns=activation_ns,
    )
    opposite_roll = -float(authority["target_roll_rad"])
    session.record_wire_acceptance(
        target_roll_rad=opposite_roll,
        target_pitch_rad=float(authority["target_pitch_rad"]),
        yaw_rate_rad_s=float(authority["yaw_rate_rad_s"]),
        thrust=float(authority["thrust"]),
        wire_command=AttitudeRateCommand(
            -0.04,
            0.02,
            float(authority["yaw_rate_rad_s"]),
            float(authority["thrust"]),
        ),
        wire_start_monotonic_ns=activation_ns + 1_000_000,
    )

    assert session.post_credit_roll_reference_handoff_active is False


def test_fresh_rebound_outward_roll_arms_geometry_released_handoff():
    session, successor_id = _bound_post_credit_successor()
    successor = session.core.course_state().successor
    assert successor is not None
    race_received_ns = successor.state_monotonic_ns - 1_000_000
    wire_ns = successor.state_monotonic_ns + 2_000_000
    activation_ns = wire_ns + 2_000_000
    session.record_wire_acceptance(
        target_roll_rad=0.0,
        target_pitch_rad=0.04,
        yaw_rate_rad_s=0.0,
        thrust=0.275,
        wire_command=AttitudeRateCommand(0.0, 0.02, 0.0, 0.275),
        wire_start_monotonic_ns=wire_ns,
    )
    session.activate_post_credit_successor_steering(
        _credited_race(race_received_ns),
        from_gate_index=0,
        reviewed_track_id=successor_id,
        activation_monotonic_ns=activation_ns,
    )
    estimate = session.core._tracks[successor_id]  # noqa: SLF001
    estimate.state = replace(
        estimate.state,
        bearing_rad=(0.40, 0.0),
        bearing_rate_rad_s=(0.10, 0.0),
        residual_translational_rate_rad_s=(0.10, 0.0),
        bearing_rate_qualified=(True, True),
        bearing_std_rad=(0.02, 0.02),
        censored_axes=(False, False),
        visible=True,
        ambiguous=False,
    )
    source = session.core.guide(activation_ns + 1_000_000)
    normal_command = replace(
        source.command,
        target_roll_rad=-0.04,
    )
    normal = replace(
        source,
        current_gate_index=1,
        current_track_id=successor_id,
        successor_track_id=None,
        current_center_norm=(0.30, 0.0),
        passage_committed=False,
        proposed_command=normal_command,
        command=normal_command,
    )

    constrained = (
        session._apply_post_credit_rebound_roll_reference(  # noqa: SLF001
            normal
        )
    )

    assert constrained.proposed_command == normal_command
    assert constrained.command.target_roll_rad == pytest.approx(
        -MAX_TARGET_ROLL_RAD
    )
    assert constrained.passage_committed is False
    assert constrained.current_gate_index == 1
    assert session.post_credit_roll_reference_handoff_active is False
    assert all(
        math.isfinite(value)
        for value in (
            constrained.command.target_roll_rad,
            constrained.command.target_pitch_rad,
            constrained.command.yaw_rate_rad_s,
            constrained.command.thrust,
        )
    )

    accepted_ns = constrained.monotonic_ns + 1_000_000
    session.record_wire_acceptance(
        target_roll_rad=constrained.command.target_roll_rad,
        target_pitch_rad=constrained.command.target_pitch_rad,
        yaw_rate_rad_s=constrained.command.yaw_rate_rad_s,
        thrust=constrained.command.thrust,
        wire_command=AttitudeRateCommand(
            -0.05,
            0.02,
            constrained.command.yaw_rate_rad_s,
            constrained.command.thrust,
        ),
        wire_start_monotonic_ns=accepted_ns,
    )
    assert session.post_credit_roll_reference_handoff_active

    # Recovery completion retires the rebound lease, while the accepted
    # reference remains state-owned until the qualified residual recovers.
    session._post_credit_successor_steering = None  # noqa: SLF001
    outward = replace(
        normal,
        monotonic_ns=accepted_ns + 1_000_000,
    )
    retained = session._apply_post_credit_roll_reference_handoff(  # noqa: SLF001
        outward
    )
    assert retained.command.target_roll_rad == pytest.approx(
        -MAX_TARGET_ROLL_RAD
    )

    estimate.state = replace(
        estimate.state,
        residual_translational_rate_rad_s=(-0.10, 0.0),
    )
    recovering_off_axis = (
        session._apply_post_credit_roll_reference_handoff(  # noqa: SLF001
            replace(
                outward,
                monotonic_ns=accepted_ns + 2_000_000,
            )
        )
    )
    assert recovering_off_axis.command.target_roll_rad == pytest.approx(
        -MAX_TARGET_ROLL_RAD
    )
    assert session.post_credit_roll_reference_handoff_active

    recovered = session._apply_post_credit_roll_reference_handoff(  # noqa: SLF001
        replace(
            outward,
            monotonic_ns=accepted_ns + 3_000_000,
            current_center_norm=(0.05, 0.0),
        )
    )
    assert recovered.command == normal_command
    assert session.post_credit_roll_reference_handoff_active is False


def test_post_credit_steering_uses_predicted_camera_elevation(
    monkeypatch: pytest.MonkeyPatch,
):
    session, successor_id = _bound_post_credit_successor()
    successor = session.core.course_state().successor
    assert successor is not None
    race_received_ns = successor.state_monotonic_ns - 1_000_000
    wire_ns = successor.state_monotonic_ns + 2_000_000
    activation_ns = wire_ns + 2_000_000
    session.record_wire_acceptance(
        target_roll_rad=0.05,
        target_pitch_rad=0.04,
        yaw_rate_rad_s=-0.04,
        thrust=0.275,
        wire_command=AttitudeRateCommand(0.03, 0.02, -0.04, 0.275),
        wire_start_monotonic_ns=wire_ns,
    )
    session.activate_post_credit_successor_steering(
        _credited_race(race_received_ns),
        from_gate_index=0,
        reviewed_track_id=successor_id,
        activation_monotonic_ns=activation_ns,
    )
    predict = session.core.predict_track_steering
    prediction = predict(successor_id, activation_ns)

    def live_vertical_prediction(track_id: str, monotonic_ns: int):
        base = predict(track_id, monotonic_ns)
        return replace(
            base,
            camera_center_norm=(base.camera_center_norm[0], -0.605),
            camera_center_rate_norm_s=(
                base.camera_center_rate_norm_s[0],
                -0.309,
            ),
        )

    monkeypatch.setattr(
        session.core,
        "predict_track_steering",
        live_vertical_prediction,
    )
    authority = session.post_credit_successor_steering_authority(
        now_monotonic_ns=activation_ns,
    )

    assert prediction.camera_center_norm[1] == pytest.approx(0.0)
    assert MIN_TARGET_PITCH_RAD <= authority["target_pitch_rad"] < 0.0
    assert authority["camera_elevation_error_rad"] < 0.0
    assert authority["camera_elevation_rate_rad_s"] < 0.0
    assert authority["pitch_delay_lead_rad"] < 0.0
    assert authority["steering_only"] is True
    assert authority["passage_authority"] is False
    assert authority["advance_authority"] is False


def test_post_credit_local_state_steers_through_dual_edge_censorship(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker, graph, snapshot, current_id = _graph()
    session = _session()
    successor_id = snapshot.next_candidates[0].track_id
    session.stage_snapshot(
        snapshot,
        tracker,
        expected_gate_index=0,
        expected_current_track_id=current_id,
        adjacent_precredit=False,
    )
    for sequence in range(6, 10):
        tracker.update(_frame(sequence))
        snapshot = graph.observe(tracker)
        session.stage_snapshot(
            snapshot,
            tracker,
            expected_gate_index=0,
            expected_current_track_id=current_id,
            adjacent_precredit=False,
        )
    session.core.bind(
        current_gate_index=0,
        current_track_id=current_id,
        successor_track_id=successor_id,
    )
    successor = session.core.course_state().successor
    assert successor is not None
    race_received_ns = successor.state_monotonic_ns - 1_000_000
    wire_ns = successor.state_monotonic_ns + 2_000_000
    activation_ns = wire_ns + 2_000_000
    session.record_wire_acceptance(
        target_roll_rad=0.05,
        target_pitch_rad=0.04,
        yaw_rate_rad_s=-0.04,
        thrust=0.275,
        wire_command=AttitudeRateCommand(
            0.03,
            0.02,
            -0.04,
            0.275,
        ),
        wire_start_monotonic_ns=wire_ns,
    )
    activation = session.activate_post_credit_successor_steering(
        _credited_race(race_received_ns),
        from_gate_index=0,
        reviewed_track_id=successor_id,
        activation_monotonic_ns=activation_ns,
    )
    predict = session.core.predict_track_steering

    predicted_vertical = {
        "center_y": -0.605,
        "rate_y": -0.309,
    }

    def top_of_camera_prediction(track_id: str, monotonic_ns: int):
        base = predict(track_id, monotonic_ns)
        return replace(
            base,
            camera_center_norm=(
                base.camera_center_norm[0],
                predicted_vertical["center_y"],
            ),
            camera_center_rate_norm_s=(
                base.camera_center_rate_norm_s[0],
                predicted_vertical["rate_y"],
            ),
        )

    monkeypatch.setattr(
        session.core,
        "predict_track_steering",
        top_of_camera_prediction,
    )
    seeded = session.post_credit_successor_steering_authority(
        now_monotonic_ns=activation_ns,
    )
    assert seeded["target_pitch_rad"] < 0.0

    predicted_vertical.update(center_y=0.0, rate_y=0.0)
    top_update = tracker.update(
        _frame(
            10,
            successor_clipping=FrameEdge.TOP,
            successor_center_censored=True,
            successor_inner_aperture=_inner_aperture(
                0.32,
                0.0,
                half_width=0.15,
                half_height=0.17,
                health_reason="outer_support_clipped_tracking_only",
            ),
        )
    )
    top_snapshot = graph.observe(tracker)
    assert tracker.track(successor_id).latest_token == top_update.token
    assert tracker.track(successor_id).clipping == FrameEdge.TOP
    session.stage_snapshot(
        top_snapshot,
        tracker,
        expected_gate_index=1,
        expected_current_track_id=successor_id,
        adjacent_precredit=False,
    )
    top_state = session.core.course_state().current
    top_authority = session.post_credit_successor_steering_authority(
        now_monotonic_ns=top_update.publish_monotonic_ns + 2_000_000,
    )
    top_decision = session.guide(
        current_track_id=successor_id,
        successor_track_id=None,
        monotonic_ns=top_update.publish_monotonic_ns + 2_000_000,
    )
    assert top_decision is not None
    with pytest.raises(
        DynamicCourseError,
        match="publication is not clipped",
    ):
        session.propagated_current_fov_gap_authority(
            track=tracker.track(successor_id),
            camera_token=top_update.token,
            now_monotonic_ns=(
                top_update.publish_monotonic_ns + 2_000_000
            ),
        )
    top_fov_authority = session.propagated_current_fov_gap_authority(
        track=tracker.track(successor_id),
        camera_token=top_update.token,
        now_monotonic_ns=top_update.publish_monotonic_ns + 2_000_000,
        allow_tracking_only_inner_raw_clipping=True,
    )
    assert top_fov_authority["clipping"] == int(FrameEdge.TOP)
    assert top_fov_authority["camera_aperture_half_size_norm"][1] > 0.0
    assert top_fov_authority["steering_only"] is True
    assert top_fov_authority["passage_authority"] is False
    assert top_fov_authority["advance_authority"] is False

    dual_update = tracker.update(
        _frame(
            11,
            successor_clipping=FrameEdge.TOP | FrameEdge.RIGHT,
            successor_center_censored=True,
        )
    )
    dual_snapshot = graph.observe(tracker)
    assert tracker.track(successor_id).latest_token == dual_update.token
    session.stage_snapshot(
        dual_snapshot,
        tracker,
        expected_gate_index=1,
        expected_current_track_id=successor_id,
        adjacent_precredit=False,
    )
    dual_state = session.core.course_state().current
    dual_authority = session.post_credit_successor_steering_authority(
        now_monotonic_ns=dual_update.publish_monotonic_ns + 2_000_000,
    )
    for sequence in range(12, 18):
        dual_update = tracker.update(
            _frame(
                sequence,
                successor_clipping=FrameEdge.TOP | FrameEdge.RIGHT,
                successor_center_censored=True,
            )
        )
        dual_snapshot = graph.observe(tracker)
        session.stage_snapshot(
            dual_snapshot,
            tracker,
            expected_gate_index=1,
            expected_current_track_id=successor_id,
            adjacent_precredit=False,
        )
        dual_state = session.core.course_state().current
        dual_authority = (
            session.post_credit_successor_steering_authority(
                now_monotonic_ns=(
                    dual_update.publish_monotonic_ns + 2_000_000
                ),
            )
        )

    # A complete tracking-only inner fit may steer both axes, while the raw
    # outer TOP clip independently retains FOV protection.
    assert top_state.clipping == FrameEdge.NONE
    assert top_state.censored_axes == (False, False)
    assert dual_state.censored_axes == (True, True)
    assert top_state.aperture_half_size_norm is not None
    assert dual_state.aperture_half_size_norm is not None
    assert top_state.aperture_propagated is True
    assert dual_state.aperture_propagated is True
    assert (
        dual_state.aperture_seed_monotonic_ns
        == top_state.aperture_seed_monotonic_ns
    )
    assert (
        dual_state.aperture_prediction_deadline_monotonic_ns
        == top_state.aperture_prediction_deadline_monotonic_ns
    )
    assert math.isfinite(dual_state.log_scale)
    assert math.isfinite(dual_state.expansion_rate_s)
    assert top_authority["last_correction_monotonic_ns"] > (
        activation["last_correction_monotonic_ns"]
    )
    assert top_authority["expires_monotonic_ns"] > (
        activation["expires_monotonic_ns"]
    )
    assert dual_authority["last_correction_monotonic_ns"] == (
        top_authority["last_correction_monotonic_ns"]
    )
    assert dual_authority["expires_monotonic_ns"] == (
        top_authority["expires_monotonic_ns"]
    )
    assert (
        dual_update.publish_monotonic_ns
        - top_update.publish_monotonic_ns
        > 200_000_000
    )
    assert (
        dual_authority["expires_monotonic_ns"]
        > dual_update.publish_monotonic_ns
    )
    assert dual_authority["last_measurement_monotonic_ns"] > (
        top_authority["last_measurement_monotonic_ns"]
    )
    assert dual_authority["target_pitch_rad"] <= (
        top_authority["target_pitch_rad"] + 1e-12
    )
    assert top_authority["current_raw_clipping"] == int(FrameEdge.TOP)
    assert top_authority["vertical_axis_censored"] is True
    assert top_authority["target_pitch_rad"] <= (
        seeded["target_pitch_rad"] + 1e-12
    )
    assert dual_authority["vertical_axis_censored"] is True
    assert dual_authority["steering_only"] is True
    assert dual_authority["passage_authority"] is False
    assert dual_authority["advance_authority"] is False
    assert all(
        math.isfinite(float(dual_authority[name]))
        for name in (
            "target_roll_rad",
            "target_pitch_rad",
            "yaw_rate_rad_s",
            "thrust",
        )
    )
    assert (
        abs(float(dual_authority["target_roll_rad"]))
        <= MAX_TARGET_ROLL_RAD
    )
    assert (
        MIN_TARGET_PITCH_RAD
        <= float(dual_authority["target_pitch_rad"])
        <= MAX_TARGET_PITCH_RAD
    )
    assert (
        abs(float(dual_authority["yaw_rate_rad_s"]))
        <= MAX_YAW_RATE_RAD_S
    )
    assert (
        MIN_THRUST
        <= float(dual_authority["thrust"])
        <= MAX_THRUST
    )
    with pytest.raises(
        DynamicCourseError,
        match="lacks clean current state",
    ):
        session.complete_post_credit_recovery(
            camera_token=dual_update.token,
        )

    clean_update = tracker.update(_frame(18))
    clean_snapshot = graph.observe(tracker)
    session.stage_snapshot(
        clean_snapshot,
        tracker,
        expected_gate_index=1,
        expected_current_track_id=successor_id,
        adjacent_precredit=False,
    )
    recovered_authority = (
        session.post_credit_successor_steering_authority(
            now_monotonic_ns=(
                clean_update.publish_monotonic_ns + 2_000_000
            ),
        )
    )
    assert recovered_authority["current_raw_clipping"] == int(
        FrameEdge.NONE
    )
    assert recovered_authority["vertical_axis_censored"] is False
    assert recovered_authority["target_pitch_rad"] == pytest.approx(
        session.core.config.brake_pitch_rad
    )
    assert recovered_authority["target_pitch_rad"] > (
        top_authority["target_pitch_rad"]
    )
    release = session.complete_post_credit_recovery(
        camera_token=clean_update.token,
    )
    assert release["basis"] == (
        "clean-current-post-credit-recovery-release-v1"
    )
    assert release["passage_authority"] is False
    assert release["advance_authority"] is False
    assert session.post_credit_successor_steering_active is False


def test_fresh_cross_id_rebind_renews_bounded_clipped_recovery() -> None:
    tracker, _graph_state, snapshot, current_id = _graph()
    session = _session()
    for monotonic_ns in range(
        _BASE_NS + 700_000_000,
        _BASE_NS + 1_000_000_000,
        10_000_000,
    ):
        session.record_imu(
            ImuAttitudeSample(
                monotonic_ns=monotonic_ns,
                body_to_reference_wxyz=(1.0, 0.0, 0.0, 0.0),
                body_rates_rad_s=(0.0, 0.0, 0.0),
                source_timestamp_us=monotonic_ns // 1_000,
                host_clock_id="host-perf-counter",
            )
        )
    reviewed_id = snapshot.next_candidates[0].track_id
    session.stage_snapshot(
        snapshot,
        tracker,
        expected_gate_index=0,
        expected_current_track_id=current_id,
        adjacent_precredit=False,
    )
    for sequence in range(6, 10):
        tracker.update(_frame(sequence))
        snapshot = _graph_state.observe(tracker)
        session.stage_snapshot(
            snapshot,
            tracker,
            expected_gate_index=0,
            expected_current_track_id=current_id,
            adjacent_precredit=False,
        )
    session.core.bind(
        current_gate_index=0,
        current_track_id=current_id,
        successor_track_id=reviewed_id,
    )
    reviewed_state = session.core.course_state().successor
    assert reviewed_state is not None
    race_status = _credited_race(
        reviewed_state.state_monotonic_ns - 1_000_000
    )
    successor_horizon_ns = round(
        session.core.config.successor_prediction_max_horizon_s
        * 1_000_000_000.0
    )
    current_horizon_ns = round(
        session.core.config.post_credit_current_prediction_max_horizon_s
        * 1_000_000_000.0
    )
    old_expiry_ns = min(
        race_status.received_monotonic_ns + successor_horizon_ns,
        reviewed_state.last_measurement_monotonic_ns
        + successor_horizon_ns,
    )
    seed_wire_ns = reviewed_state.state_monotonic_ns + 500_000
    session.record_wire_acceptance(
        target_roll_rad=0.0,
        target_pitch_rad=0.0,
        yaw_rate_rad_s=0.0,
        thrust=0.275,
        wire_command=AttitudeRateCommand(0.0, 0.0, 0.0, 0.275),
        wire_start_monotonic_ns=seed_wire_ns,
    )
    reviewed_estimate = session.core._tracks[reviewed_id]  # noqa: SLF001
    reviewed_estimate.state = replace(
        reviewed_estimate.state,
        bearing_rad=(0.40, 0.0),
        bearing_rate_rad_s=(0.10, 0.0),
        residual_translational_rate_rad_s=(0.10, 0.0),
        bearing_rate_qualified=(True, True),
        bearing_std_rad=(0.02, 0.02),
        censored_axes=(False, False),
        visible=True,
        ambiguous=False,
    )
    session.stage_snapshot(
        snapshot,
        tracker,
        expected_gate_index=1,
        expected_current_track_id=reviewed_id,
        adjacent_precredit=True,
    )
    authority_ns = seed_wire_ns + 500_000
    precredit = session.adjacent_precredit_successor_steering_authority(
        track_id=reviewed_id,
        now_monotonic_ns=authority_ns,
    )
    retained_roll = float(precredit["target_roll_rad"])
    assert retained_roll == pytest.approx(-MAX_TARGET_ROLL_RAD)
    wire_ns = authority_ns + 500_000
    session.record_wire_acceptance(
        target_roll_rad=retained_roll,
        target_pitch_rad=float(precredit["target_pitch_rad"]),
        yaw_rate_rad_s=float(precredit["yaw_rate_rad_s"]),
        thrust=float(precredit["thrust"]),
        wire_command=AttitudeRateCommand(
            -0.07553,
            0.02,
            float(precredit["yaw_rate_rad_s"]),
            float(precredit["thrust"]),
        ),
        wire_start_monotonic_ns=wire_ns,
    )
    expired = session.activate_post_credit_successor_steering(
        race_status,
        from_gate_index=0,
        reviewed_track_id=reviewed_id,
        activation_monotonic_ns=old_expiry_ns + 1,
    )
    assert expired["steering_available"] is False
    assert expired["steering_unavailable_reason"] == "expired_prediction"
    dormant = session._post_credit_roll_reference_handoff  # noqa: SLF001
    assert dormant is not None
    assert dormant.retained_target_roll_rad == pytest.approx(retained_roll)
    assert dormant.expires_monotonic_ns == old_expiry_ns
    with pytest.raises(
        PostCreditSuccessorSteeringUnavailable,
        match="handoff has no steering authority",
    ):
        session.post_credit_successor_steering_authority(
            now_monotonic_ns=old_expiry_ns + 1,
        )

    reviewed_track = tracker.track(reviewed_id)
    credited_advance = CreditedUnboundGateAdvance(
        from_gate_index=0,
        to_gate_index=1,
        retired_track_id=current_id,
        reviewed_track_id=reviewed_id,
        race_status=race_status,
        camera_token_at_credit=reviewed_track.latest_token,
        reviewed_first_token=reviewed_track.first_token,
        reviewed_latest_token_before_credit=reviewed_track.latest_token,
        reviewed_history_length_at_credit=len(reviewed_track.history),
        reviewed_history_length_at_advance=len(reviewed_track.history),
        reviewed_history_sha256=visual_track_history_sha256(
            reviewed_track.history
        ),
    )

    def fresh_frame(
        sequence: int,
        *,
        center_y: float = -0.70,
        clipping: FrameEdge = FrameEdge.NONE,
        center_censored: bool = False,
    ) -> VisualDetectionFrame:
        base = _frame(sequence)
        fresh = _detection(
            2,
            0.60,
            center_y=center_y,
            width=0.16,
            height=0.24,
            clipping=clipping,
            center_censored=center_censored,
        )
        return replace(
            base,
            detections=(*base.detections, fresh),
        )

    fresh_id = ""
    for sequence in range(10, 23):
        update = tracker.update(
            fresh_frame(
                sequence,
                center_y=(-0.76 if sequence == 22 else -0.70),
                clipping=(
                    FrameEdge.TOP
                    if sequence == 22
                    else FrameEdge.NONE
                ),
                center_censored=(sequence == 22),
            )
        )
        candidates = tuple(
            track_id
            for track_id in update.created_track_ids
            if track_id not in {current_id, reviewed_id}
        )
        if candidates:
            assert not fresh_id
            assert len(candidates) == 1
            fresh_id = candidates[0]
    assert fresh_id
    fresh_track = tracker.track(fresh_id)
    tracker.assign_role(fresh_id, VisualTrackRole.CURRENT)
    tracker.confirm_authoritative_gate(
        fresh_id,
        gate_index=1,
        race_status_sequence=race_status.race_status_sequence,
        race_status_boot_ms=race_status.race_status_boot_ms,
    )
    fresh_track = tracker.track(fresh_id)
    reacquisition = ConfirmedGateReacquisition(
        credited_advance=credited_advance,
        gate_index=1,
        reacquired_track_id=fresh_id,
        camera_token_at_binding=fresh_track.latest_token,
        reacquired_first_token=fresh_track.first_token,
        stable_frame_tokens=tuple(
            sample.token for sample in fresh_track.history[-3:]
        ),
        history_length_at_binding=len(fresh_track.history),
        history_sha256=visual_track_history_sha256(
            fresh_track.history
        ),
        cross_gap_identity_claimed=False,
    )

    rebound = session.rebind_confirmed_reacquisition(
        reacquisition,
        tracker,
    )
    rebound_state = session.core.course_state().current

    assert rebound["reviewed_track_id"] == reviewed_id
    assert rebound["reacquired_track_id"] == fresh_id
    assert rebound["steering_available"] is True
    assert rebound["steering_only"] is True
    assert rebound["passage_authority"] is False
    assert rebound["advance_authority"] is False
    assert rebound["recovery_steering"]["steering_track_id"] == fresh_id
    assert rebound_state.track_id == fresh_id
    assert rebound_state.visible
    assert rebound_state.censored_axes == (False, True)
    assert rebound_state.aperture_half_size_norm is None
    assert rebound_state.aperture_seed_monotonic_ns is None
    assert rebound_state.aperture_propagated is False
    assert (
        rebound["recovery_steering"]["prediction_horizon_s"]
        == session.core.config
        .post_credit_current_prediction_max_horizon_s
    )
    assert rebound["recovery_steering"]["expires_monotonic_ns"] == (
        rebound_state.state_monotonic_ns + current_horizon_ns
    )
    assert (
        rebound["recovery_steering"]["expires_monotonic_ns"]
        > old_expiry_ns
    )
    rebound_handoff = (
        session._post_credit_roll_reference_handoff  # noqa: SLF001
    )
    assert rebound_handoff is not None
    assert rebound_handoff.track_id == fresh_id
    assert rebound_handoff.retained_target_roll_rad == pytest.approx(
        retained_roll
    )
    assert rebound_handoff.expires_monotonic_ns == (
        rebound["recovery_steering"]["expires_monotonic_ns"]
    )
    # Latest-flight facts: the exact fresh cross-ID reanchor starts with an
    # unqualified horizontal rate and a +0.190625 normalized center.  Its
    # ordinary proportional reference is about -0.053 rad, but it must not
    # unwind the already accepted -0.25-rad successor bank.
    fresh_center_x = 0.190625
    fresh_bearing_rad = math.atan(
        fresh_center_x * session.core.config.horizontal_angle_scale_rad
    )
    fresh_estimate = session.core._tracks[fresh_id]  # noqa: SLF001
    fresh_estimate.state = replace(
        fresh_estimate.state,
        bearing_rad=(fresh_bearing_rad, 0.0),
        bearing_rate_rad_s=(0.0, 0.0),
        residual_translational_rate_rad_s=(0.0, 0.0),
        bearing_rate_qualified=(False, True),
        bearing_std_rad=(0.02, 0.02),
    )
    binding_ns = (
        fresh_track.history[-1].publication_monotonic_ns + 2_000_000
    )
    unconstrained_prediction = session.core.predict_track_steering(
        fresh_id,
        binding_ns,
    )
    unconstrained_targets = session._successor_steering_targets(  # noqa: SLF001
        unconstrained_prediction
    )
    binding_authority = (
        session.post_credit_successor_steering_authority(
            now_monotonic_ns=binding_ns,
        )
    )
    assert abs(
        float(binding_authority["unconstrained_target_roll_rad"])
    ) < abs(retained_roll)
    assert binding_authority["unconstrained_target_roll_rad"] == pytest.approx(
        session.core.config.roll_guidance_sign
        * session.core.config.roll_gain
        * fresh_bearing_rad
    )
    assert binding_authority["target_roll_rad"] == pytest.approx(
        retained_roll
    )
    assert binding_authority["retained_roll_reference_applied"] is True
    assert binding_authority["target_pitch_rad"] == pytest.approx(
        unconstrained_targets["target_pitch_rad"]
    )
    assert binding_authority["yaw_rate_rad_s"] == pytest.approx(
        unconstrained_targets["yaw_rate_rad_s"]
    )
    assert binding_authority["thrust"] == pytest.approx(
        unconstrained_targets["thrust"]
    )
    assert binding_authority["vertical_axis_censored"] is True
    assert binding_authority["steering_only"] is True
    assert binding_authority["passage_authority"] is False
    assert binding_authority["advance_authority"] is False
    assert all(
        math.isfinite(float(binding_authority[name]))
        for name in (
            "target_roll_rad",
            "target_pitch_rad",
            "yaw_rate_rad_s",
            "thrust",
        )
    )
    governed = session.govern_wire_command(
        AttitudeRateCommand(
            -MAX_TARGET_ROLL_RAD,
            0.02,
            float(binding_authority["yaw_rate_rad_s"]),
            float(binding_authority["thrust"]),
        ),
        proposal_monotonic_ns=binding_ns + 1_000_000,
        launch_thrust_override=False,
        yaw_safety_override=False,
    )
    assert -MAX_TARGET_ROLL_RAD <= governed.roll_rate <= 0.0
    assert abs(governed.pitch_rate) <= MAX_TARGET_PITCH_RAD
    assert abs(governed.yaw_rate) <= MAX_YAW_RATE_RAD_S
    assert MIN_THRUST <= governed.thrust <= MAX_THRUST
    assert all(
        math.isfinite(value)
        for value in (
            governed.roll_rate,
            governed.pitch_rate,
            governed.yaw_rate,
            governed.thrust,
        )
    )

    clipped_update = tracker.update(
        fresh_frame(
            23,
            center_y=-0.76,
            clipping=FrameEdge.TOP,
            center_censored=True,
        )
    )
    clipped_snapshot = replace(
        snapshot,
        latest_camera_token=clipped_update.token,
        tracker_frame_sequence=clipped_update.tracker_frame_sequence,
        current_track_id=fresh_id,
        current_gate_index=1,
        current_track=tracker.track(fresh_id),
        next_candidates=(),
    )
    session.stage_snapshot(
        clipped_snapshot,
        tracker,
        expected_gate_index=1,
        expected_current_track_id=fresh_id,
        adjacent_precredit=False,
    )
    clipped_state = session.core.course_state().current
    authority = session.post_credit_successor_steering_authority(
        now_monotonic_ns=(
            clipped_update.publish_monotonic_ns + 2_000_000
        ),
    )

    assert clipped_state.censored_axes == (False, True)
    # This fresh identity has never earned aperture authority; steering may
    # use its observable axis, but clipping cannot fabricate a local aperture.
    assert clipped_state.aperture_propagated is False
    assert clipped_state.aperture_half_size_norm is None
    assert authority["basis"] == (
        "authoritative-post-credit-fresh-reacquisition-steering-v1"
    )
    assert authority["steering_track_id"] == fresh_id
    assert authority["steering_only"] is True
    assert authority["passage_authority"] is False
    assert authority["advance_authority"] is False
    assert all(
        math.isfinite(float(authority[name]))
        for name in (
            "target_roll_rad",
            "target_pitch_rad",
            "yaw_rate_rad_s",
            "thrust",
        )
    )
    assert (
        abs(float(authority["target_roll_rad"]))
        <= MAX_TARGET_ROLL_RAD
    )
    assert (
        MIN_TARGET_PITCH_RAD
        <= float(authority["target_pitch_rad"])
        <= MAX_TARGET_PITCH_RAD
    )
    assert (
        abs(float(authority["yaw_rate_rad_s"]))
        <= MAX_YAW_RATE_RAD_S
    )
    assert (
        MIN_THRUST
        <= float(authority["thrust"])
        <= MAX_THRUST
    )

    clean_update = tracker.update(fresh_frame(24))
    clean_snapshot = replace(
        clipped_snapshot,
        latest_camera_token=clean_update.token,
        tracker_frame_sequence=clean_update.tracker_frame_sequence,
        current_track=tracker.track(fresh_id),
    )
    session.stage_snapshot(
        clean_snapshot,
        tracker,
        expected_gate_index=1,
        expected_current_track_id=fresh_id,
        adjacent_precredit=False,
    )
    release = session.complete_post_credit_recovery(
        camera_token=clean_update.token,
    )
    assert release["passage_authority"] is False
    assert release["advance_authority"] is False
    assert session.post_credit_successor_steering_active is False


def test_post_credit_activation_latency_never_extends_prediction_expiry():
    session, successor_id = _bound_post_credit_successor()
    successor = session.core.course_state().successor
    assert successor is not None
    horizon_ns = round(
        session.core.config.successor_prediction_max_horizon_s
        * 1_000_000_000.0
    )
    race_received_ns = successor.state_monotonic_ns - 1_000_000
    wire_ns = successor.state_monotonic_ns + 2_000_000
    activation_ns = race_received_ns + 175_000_000
    session.record_wire_acceptance(
        target_roll_rad=0.04,
        target_pitch_rad=0.06,
        yaw_rate_rad_s=-0.03,
        thrust=0.275,
        wire_command=AttitudeRateCommand(
            0.02,
            0.03,
            -0.03,
            0.275,
        ),
        wire_start_monotonic_ns=wire_ns,
    )
    expected_expiry_ns = min(
        race_received_ns + horizon_ns,
        successor.last_measurement_monotonic_ns + horizon_ns,
    )

    evidence = session.activate_post_credit_successor_steering(
        _credited_race(race_received_ns),
        from_gate_index=0,
        reviewed_track_id=successor_id,
        activation_monotonic_ns=activation_ns,
    )

    assert evidence["expires_monotonic_ns"] == expected_expiry_ns
    assert expected_expiry_ns < activation_ns + horizon_ns
    session.post_credit_successor_steering_authority(
        now_monotonic_ns=expected_expiry_ns,
    )
    with pytest.raises(
        PostCreditSuccessorSteeringUnavailable,
        match="post-credit successor steering expired",
    ):
        session.post_credit_successor_steering_authority(
            now_monotonic_ns=expected_expiry_ns + 1,
        )
    assert session.post_credit_successor_steering_active is False
    downgraded = session.activate_post_credit_successor_steering(
        _credited_race(race_received_ns),
        from_gate_index=0,
        reviewed_track_id=successor_id,
        activation_monotonic_ns=activation_ns,
    )
    assert downgraded["steering_available"] is False
    assert downgraded["steering_unavailable_reason"] == "expired_prediction"


def test_expired_successor_prediction_retains_handoff_without_steering():
    session, successor_id = _bound_post_credit_successor()
    successor = session.core.course_state().successor
    assert successor is not None
    horizon_ns = round(
        session.core.config.successor_prediction_max_horizon_s
        * 1_000_000_000.0
    )
    race_received_ns = successor.state_monotonic_ns - 1_000_000
    expected_expiry_ns = min(
        race_received_ns + horizon_ns,
        successor.last_measurement_monotonic_ns + horizon_ns,
    )
    activation_ns = expected_expiry_ns + 1
    wire_ns = successor.state_monotonic_ns + 2_000_000
    session.record_wire_acceptance(
        target_roll_rad=0.04,
        target_pitch_rad=0.06,
        yaw_rate_rad_s=-0.03,
        thrust=0.275,
        wire_command=AttitudeRateCommand(
            0.02,
            0.03,
            -0.03,
            0.275,
        ),
        wire_start_monotonic_ns=wire_ns,
    )

    evidence = session.activate_post_credit_successor_steering(
        _credited_race(race_received_ns),
        from_gate_index=0,
        reviewed_track_id=successor_id,
        activation_monotonic_ns=activation_ns,
    )

    assert evidence["steering_available"] is False
    assert evidence["steering_unavailable_reason"] == "expired_prediction"
    assert evidence["steering_only"] is False
    assert evidence["passage_authority"] is False
    assert evidence["advance_authority"] is False
    assert session.post_credit_successor_steering_active is False
    assert session.core.course_state().current_gate_index == 1
    assert session.core.course_state().current_track_id == successor_id
    with pytest.raises(
        PostCreditSuccessorSteeringUnavailable,
        match="handoff has no steering authority",
    ):
        session.post_credit_successor_steering_authority(
            now_monotonic_ns=activation_ns,
        )
