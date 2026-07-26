from __future__ import annotations

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


_BASE_NS = 10_000_000_000
_PERIOD_NS = 33_000_000


def _detection(
    source_index: int,
    center_x: float,
    *,
    width: float,
    height: float,
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
        clipping=FrameEdge.NONE,
        center_censored=False,
    )


def _frame(
    sequence: int,
    *,
    current_width: float = 0.34,
    current_height: float = 0.36,
    include_successor: bool = True,
) -> VisualDetectionFrame:
    observation_ns = _BASE_NS + sequence * _PERIOD_NS
    detections = [
        _detection(
            0,
            0.0,
            width=current_width,
            height=current_height,
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
        _BASE_NS + 500_000_000,
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


def test_dynamic_graph_adapter_seeds_then_brakes_and_turns_to_successor():
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

    assert proposal.servo_output.target_roll_rad > 0.0
    assert proposal.servo_output.yaw_rate_rad_s < 0.0
    assert proposal.servo_output.target_pitch_rad > 0.0
    assert proposal.servo_output.brake_reason in {
        "off_axis_successor_intercept",
        "off_axis_rapid_closure",
    }
    assert proposal.servo_output.reviewed_next_track_id is not None
    assert session.last_decision is not None
    assert session.last_decision.current_gate_index == 0
    assert session.last_decision.successor_track_id is not None
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
