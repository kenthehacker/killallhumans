from __future__ import annotations

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
)
from planning.vq2_dynamic_course import (
    DynamicCourseError,
    ImuAttitudeSample,
)
from planning.vq2_dynamic_visual_approach import (
    DYNAMIC_CONTROLLER_FAMILY,
    DynamicRollingVisualApproachServo,
    DynamicVisualCourseSession,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    RollingVisualGateGraph,
)
from planning.vq2_visual_approach import (
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachMode,
)


_BASE_NS = 10_000_000_000
_PERIOD_NS = 33_000_000


def _detection(
    source_index: int,
    center_x: float,
    *,
    width: float,
    height: float,
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
) -> VisualDetection:
    unit_x = 0.5 * (center_x + 1.0)
    return VisualDetection(
        source_index=source_index,
        center_norm=(center_x, 0.0),
        bbox_norm=(
            unit_x - width / 2.0,
            0.5 - height / 2.0,
            unit_x + width / 2.0,
            0.5 + height / 2.0,
        ),
        confidence=0.95,
        clipping=clipping,
        center_censored=center_censored,
    )


def _frame(
    sequence: int,
    *,
    current_width: float = 0.34,
    current_height: float = 0.36,
    include_successor: bool = True,
    current_clipping: FrameEdge = FrameEdge.NONE,
    current_center_censored: bool = False,
) -> VisualDetectionFrame:
    observation_ns = _BASE_NS + sequence * _PERIOD_NS
    detections = [
        _detection(
            0,
            0.0,
            width=current_width,
            height=current_height,
            clipping=current_clipping,
            center_censored=current_center_censored,
        )
    ]
    if include_successor:
        detections.append(
            _detection(1, 0.32, width=0.15, height=0.17)
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


def _session() -> DynamicVisualCourseSession:
    session = DynamicVisualCourseSession()
    for monotonic_ns in range(
        _BASE_NS - 200_000_000,
        _BASE_NS + 700_000_000,
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


def test_dynamic_graph_adapter_biases_passage_without_unadmitted_successor_yaw():
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

    update = tracker.update(_frame(6))
    snapshot = graph.observe(tracker)
    proposal = _observe(planner, snapshot, tracker)
    assert proposal.servo_output.target_roll_rad == 0.0
    assert proposal.servo_output.yaw_rate_rad_s == 0.0
    assert session.last_decision is not None
    assert session.last_decision.passage_point_norm == (0.0, 0.0)
    assert session.last_decision.successor_prediction_confidence == 0.0
    _accept_proposal(session, tracker, proposal)

    for sequence in (7, 8):
        tracker.update(_frame(sequence))
        snapshot = graph.observe(tracker)
        proposal = _observe(planner, snapshot, tracker)
        if sequence == 7:
            _accept_proposal(session, tracker, proposal)

    assert proposal.servo_output.target_roll_rad > 0.0
    assert proposal.servo_output.yaw_rate_rad_s == 0.0
    assert proposal.servo_output.target_pitch_rad < 0.0
    assert proposal.servo_output.brake_reason == (
        "off_axis_successor_intercept"
    )
    assert proposal.servo_output.reviewed_next_track_id is not None
    assert session.last_decision is not None
    assert session.last_decision.current_gate_index == 0
    assert session.last_decision.successor_track_id is not None
    assert session.last_decision.passage_point_norm[0] > 0.0
    assert session.last_decision.successor_weight == 0.0
    assert session.last_decision.passage_yaw_authority == 0.0
    assert session.last_decision.successor_yaw_contribution_rad == 0.0
    assert session.last_decision.braking
    assert abs(
        math.atan(
            session.last_decision.passage_error_norm[0]
            * session.core.config.horizontal_angle_scale_rad
        )
    ) >= session.core.config.off_axis_brake_rad
    assert session.evidence_summary()["controller_family"] == (
        DYNAMIC_CONTROLLER_FAMILY
    )


@pytest.mark.parametrize(
    ("gate_size", "expect_admission"),
    ((0.36, False), (0.90, True)),
)
def test_dynamic_passage_admission_requires_scale_with_uncertainty(
    gate_size: float,
    expect_admission: bool,
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

    assert (proposal.passage_admission is not None) is expect_admission
    if not expect_admission:
        assert proposal.servo_output.brake_reason == (
            "dynamic_plane_not_ready"
        )


def test_5dffc517_passage_seals_successor_through_expected_occlusion() -> None:
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
        if sequence < 14:
            assert proposal.passage_admission is None
        _accept_proposal(session, tracker, proposal)

    admission = proposal.passage_admission
    successor = session.core.course_state().successor
    assert successor is not None
    assert successor.visible is False
    assert successor.missed_count == 6
    assert snapshot.next_candidates == ()
    assert admission is not None
    assert admission.preview_track_id == retained_id
    assert admission.preview_blend == 0.0
    assert proposal.next_target is None
    assert proposal.servo_output.brake_reason == "aligning"
    assert proposal.servo_output.corridor_frames >= 3

    tracker.update(
        _frame(
            15,
            current_width=0.80,
            current_height=0.80,
            include_successor=False,
            current_clipping=FrameEdge.TOP,
            current_center_censored=True,
        )
    )
    snapshot = graph.observe(tracker)
    update = tracker.latest_update
    assert update is not None
    with pytest.raises(VisualApproachCurrentGeometryUnavailable):
        planner.observe(
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
    with pytest.raises(VisualApproachCurrentGeometryUnavailable):
        planner.observe(
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


def test_unit_bbox_full_size_maps_to_signed_center_half_aperture() -> None:
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
    state = next(
        state
        for state in session.core.track_states
        if state.track_id == current_id
    )
    assert state.aperture_half_size_norm == pytest.approx((0.34, 0.36))
    _accept_proposal(session, tracker, seed)

    tracker.update(
        _frame(
            6,
            current_width=0.34,
            current_height=0.36,
            include_successor=False,
        )
    )
    snapshot = graph.observe(tracker)
    _observe(planner, snapshot, tracker)

    assert session.last_decision is not None
    assert session.last_decision.current_aperture_half_size_norm == (
        pytest.approx((0.34, 0.36))
    )
    assert session.last_decision.aperture_margin_norm == pytest.approx(
        (0.25, 0.27)
    )


def test_final_wire_governor_cannot_reverse_roll_in_one_frame():
    session = _session()
    first_ns = _BASE_NS
    session.record_wire_acceptance(
        target_roll_rad=0.08,
        target_pitch_rad=0.0,
        yaw_rate_rad_s=0.0,
        thrust=0.275,
        wire_command=AttitudeRateCommand(0.12, 0.0, 0.0, 0.275),
        wire_start_monotonic_ns=first_ns,
    )
    proposed = session.govern_wire_command(
        AttitudeRateCommand(-0.25, 0.0, 0.0, 0.275),
        proposal_monotonic_ns=first_ns + 20_000_000,
        launch_thrust_override=False,
        yaw_safety_override=False,
    )

    assert proposed.roll_rate >= 0.0
    assert first_ns + 20_000_000 > first_ns


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
