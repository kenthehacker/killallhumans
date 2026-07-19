from __future__ import annotations

import dataclasses
import json
import math
from types import SimpleNamespace

import pytest

from aigp_loop.replay import _validate_processor_command, _validate_processor_detections
from competition.adapter import AttitudeRateCommand
from competition.vq2_contracts import (
    CommandProposalV1,
    EdgeSetV1,
    EventOutcome,
    FeatureCovarianceV1,
    FitDiagnosticsV1,
    FrameEdge,
    FrameIdentityV1,
    FrameTimingV1,
    GateAuthorityEpochV1,
    GateObservationV1,
    LatencyEventKind,
    LatencyEventV1,
    LineSegmentV1,
    MeasurementTimeBasis,
    ObservationHealth,
    PredictionBasis,
    PredictionTimeV1,
    RelativeGateStateV1,
    RelativeStateHealth,
    SaturationDiagnosticsV1,
    SupervisorApprovedCommandV1,
    TrackRole,
    UncertaintyDiagnosticsV1,
    approved_command_to_attitude_rate_command,
    frame_identity_from_snapshot,
    legacy_attitude_rate_to_proposal,
    proposal_to_replay_command_v1,
    validate_frame_timing_sequence,
    validate_gate_observation_batch,
    validate_approved_command_sequence,
    validate_command_latency_correlation,
    validate_command_proposal_source,
    validate_latency_event_sequence,
    validate_relative_gate_state_source,
    validate_relative_gate_state_sequence,
)
from gate_detection.src.vq2_observation_adapter import (
    gate_detection_to_observation_v1,
    observation_to_legacy_gate_target_fields,
    observation_to_replay_detection_v1,
)


_COMMAND_SOURCE_STATE_NS = 2_000_000_025
_COMMAND_PROPOSAL_NS = 2_000_000_030
_COMMAND_APPROVAL_NS = 2_000_000_035
_COMMAND_SEND_NS = 2_000_000_040
_COMMAND_DEADLINE_NS = 2_000_000_050


def _frame(*, generation: int = 3, frame_id: int = 41) -> FrameIdentityV1:
    return FrameIdentityV1("camera0", generation, frame_id)


def _authority(
    *,
    gate_index: int = 0,
    gate_epoch: int = 0,
    camera_generation: int = 3,
    frame_publication_sequence_not_before: int | None = None,
    frame_publish_monotonic_ns_not_before: int | None = None,
) -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id="training-session-1",
        reset_epoch=7,
        gate_epoch=gate_epoch,
        expected_gate_index=gate_index,
        race_status_sequence=20 + gate_epoch,
        race_status_boot_ms=1_250 + 250 * gate_epoch,
        camera_host_clock_id="host-monotonic-1",
        camera_stream_id="camera0",
        camera_generation=camera_generation,
        frame_publication_sequence_not_before=(
            8 + gate_epoch
            if frame_publication_sequence_not_before is None
            else frame_publication_sequence_not_before
        ),
        frame_publish_monotonic_ns_not_before=(
            2_000_000_011 + gate_epoch
            if frame_publish_monotonic_ns_not_before is None
            else frame_publish_monotonic_ns_not_before
        ),
    )


def _frame_timing(
    *,
    frame: FrameIdentityV1 | None = None,
    publication_sequence: int = 8,
    camera_source_time_ns: int = 1_725_000_000_000_000_000,
    base_monotonic_ns: int = 1_999_999_950,
) -> FrameTimingV1:
    return FrameTimingV1(
        identity=frame or _frame(),
        camera_source_time_ns=camera_source_time_ns,
        host_clock_id="host-monotonic-1",
        publication_sequence=publication_sequence,
        first_unique_packet_monotonic_ns=base_monotonic_ns,
        final_unique_packet_monotonic_ns=base_monotonic_ns + 50,
        reassembly_complete_monotonic_ns=base_monotonic_ns + 50,
        decode_start_monotonic_ns=base_monotonic_ns + 51,
        decode_end_monotonic_ns=base_monotonic_ns + 60,
        publish_monotonic_ns=base_monotonic_ns + 61,
    )


def _prediction(
    *,
    source_frame: FrameIdentityV1 | None = None,
    source_frame_publication_sequence: int = 9,
    source_frame_publish_monotonic_ns: int = 2_000_000_012,
) -> PredictionTimeV1:
    return PredictionTimeV1(
        host_clock_id="host-monotonic-1",
        source_frame=source_frame or _frame(),
        source_frame_publication_sequence=source_frame_publication_sequence,
        source_frame_publish_monotonic_ns=source_frame_publish_monotonic_ns,
        measurement_time_monotonic_ns=2_000_000_000,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        measurement_time_model_id=None,
        measurement_uncertainty_ns=2_000_000,
        decision_time_monotonic_ns=2_000_000_020,
        prediction_time_monotonic_ns=2_000_000_025,
        prediction_basis=PredictionBasis.COMMAND_SEND_ESTIMATE,
        delay_model_id="build3385-conservative-send-v1",
        delay_uncertainty_ns=5_000_000,
    )


def _state_covariance(scale: float = 1.0) -> FeatureCovarianceV1:
    names = (
        "bearing_x_norm",
        "bearing_y_norm",
        "log_scale",
        "bearing_rate_x_norm_s",
        "bearing_rate_y_norm_s",
        "expansion_rate_s",
    )
    return FeatureCovarianceV1(
        model_id="relative-filter-v1",
        feature_order=names,
        matrix=tuple(
            tuple(scale if row == column else 0.0 for column in range(6))
            for row in range(6)
        ),
    )


def _relative_state(**changes) -> RelativeGateStateV1:
    values = {
        "timing": _prediction(),
        "authority": _authority(gate_index=1, gate_epoch=1),
        "tracker_id": "active-gate-1",
        "state_sequence": 12,
        "measurement_update_sequence": 5,
        "source_candidate_id": "gate-1-candidate-0",
        "track_role": TrackRole.ACTIVE,
        "bearing_norm": (0.55, -0.73),
        "bearing_rate_norm_s": (0.12, -0.08),
        "log_scale": -1.2,
        "expansion_rate_s": 0.3,
        "covariance": _state_covariance(),
        "metric_position_body_frd_m": None,
        "metric_velocity_body_frd_m_s": None,
        "metric_gate_orientation_body_frd_xyzw": None,
        "metric_covariance": None,
        "last_clipping": FrameEdge.TOP,
        "outer_visibility": FrameEdge.LEFT | FrameEdge.RIGHT | FrameEdge.BOTTOM,
        "inner_visibility": FrameEdge.NONE,
        "normalized_innovation_squared": None,
        "innovation_gate_threshold": None,
        "innovation_accepted": None,
        "dropout_count": 2,
        "health": RelativeStateHealth.COASTING,
        "health_reason": "bounded_frame_dropout",
    }
    values.update(changes)
    return RelativeGateStateV1(**values)


def _proposal(command: AttitudeRateCommand | None = None) -> CommandProposalV1:
    command = command or AttitudeRateCommand(0.1, -0.2, 0.0, 0.27)
    return legacy_attitude_rate_to_proposal(
        command,
        proposal_id=72,
        control_tick_id=71,
        host_clock_id="host-monotonic-1",
        proposal_monotonic_ns=_COMMAND_PROPOSAL_NS,
        control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
        source_state_decision_monotonic_ns=2_000_000_020,
        source_state_prediction_monotonic_ns=_COMMAND_SOURCE_STATE_NS,
        source_frame=_frame(),
        source_frame_publication_sequence=8,
        source_frame_publish_monotonic_ns=2_000_000_011,
        source_tracker_id="active-gate-0",
        source_track_role=TrackRole.ACTIVE,
        source_state_sequence=11,
        source_measurement_update_sequence=4,
        source_candidate_id="gate-0-candidate-0",
        authority=_authority(),
        phase="approach",
        saturation=SaturationDiagnosticsV1(
            body_rate_axes=(False, False, False), thrust=False
        ),
        uncertainty=UncertaintyDiagnosticsV1(limited=False, reason=None),
    )


def _proposal_source_state(**changes) -> RelativeGateStateV1:
    values = {
        "timing": _prediction(
            source_frame_publication_sequence=8,
            source_frame_publish_monotonic_ns=2_000_000_011,
        ),
        "authority": _authority(),
        "tracker_id": "active-gate-0",
        "state_sequence": 11,
        "measurement_update_sequence": 4,
        "source_candidate_id": "gate-0-candidate-0",
        "track_role": TrackRole.ACTIVE,
        "bearing_norm": (0.0, 0.0),
        "bearing_rate_norm_s": (0.0, 0.0),
        "log_scale": -1.0,
        "expansion_rate_s": 0.1,
        "covariance": _state_covariance(),
        "metric_position_body_frd_m": None,
        "metric_velocity_body_frd_m_s": None,
        "metric_gate_orientation_body_frd_xyzw": None,
        "metric_covariance": None,
        "last_clipping": FrameEdge.NONE,
        "outer_visibility": FrameEdge.LEFT
        | FrameEdge.TOP
        | FrameEdge.RIGHT
        | FrameEdge.BOTTOM,
        "inner_visibility": FrameEdge.NONE,
        "normalized_innovation_squared": None,
        "innovation_gate_threshold": None,
        "innovation_accepted": None,
        "dropout_count": 0,
        "health": RelativeStateHealth.HEALTHY,
        "health_reason": None,
    }
    values.update(changes)
    return RelativeGateStateV1(**values)


def _approved(proposal: CommandProposalV1 | None = None, **changes):
    proposal = proposal or _proposal()
    values = {
        "command_id": 73,
        "approval_monotonic_ns": _COMMAND_APPROVAL_NS,
        "valid_until_monotonic_ns": _COMMAND_DEADLINE_NS,
        "proposal": proposal,
        "approved_body_rates_rad_s": proposal.requested_body_rates_rad_s,
        "approved_thrust": proposal.requested_thrust,
        "safety_policy_id": "vq2-training-envelope-v1",
        "safety_limit_reason": None,
    }
    values.update(changes)
    return SupervisorApprovedCommandV1(**values)


def _detection(*, bbox=(282, 134, 80, 81), center=(322, 174), confidence=0.9):
    # Deliberately poisonous placeholder fields prove that the adapter does not
    # relabel them as inner aperture or metric pose evidence.
    return SimpleNamespace(
        bbox=bbox,
        center_x=center[0],
        center_y=center[1],
        confidence=confidence,
        corners=((999.0, 999.0),) * 4,
        estimated_distance=0.001,
    )


def _observation(detection=None, **changes) -> GateObservationV1:
    values = {
        "detection": detection or _detection(),
        "frame_timing": _frame_timing(),
        "authority": _authority(),
        "candidate_id": "gate-0-candidate-0",
        "measurement_uncertainty_ns": 33_333_334,
        "center_covariance": FeatureCovarianceV1(
            model_id="legacy-bbox-conservative-v1",
            feature_order=("center_x_norm", "center_y_norm"),
            matrix=((0.01, 0.0), (0.0, 0.01)),
        ),
    }
    values.update(changes)
    return gate_detection_to_observation_v1(**values)


def test_frame_identity_is_generation_scoped_and_camera_time_is_not_identity():
    first = FrameIdentityV1("camera0", 1, 9)
    reset = FrameIdentityV1("camera0", 2, 9)
    other_stream = FrameIdentityV1("camera1", 1, 9)
    assert len({first, reset, other_stream}) == 3
    assert set(first.to_primitive()) == {"schema", "stream_id", "generation", "frame_id"}
    assert FrameIdentityV1.from_primitive(first.to_primitive()) == first


@pytest.mark.parametrize(
    "field,value",
    [("generation", True), ("frame_id", -1), ("frame_id", 1 << 32)],
)
def test_frame_identity_rejects_coerced_or_out_of_range_values(field, value):
    fields = {"stream_id": "camera0", "generation": 0, "frame_id": 1}
    fields[field] = value
    with pytest.raises((TypeError, ValueError)):
        FrameIdentityV1(**fields)


def test_frame_identity_snapshot_adapter_preserves_legacy_identity_only():
    snapshot = SimpleNamespace(generation=4, frame_id=55, sim_time_ns=999_999)
    identity = frame_identity_from_snapshot(snapshot)
    assert identity == FrameIdentityV1("camera0", 4, 55)
    assert "sim_time" not in json.dumps(identity.to_primitive())


def test_frame_timing_round_trip_and_ordering():
    timing = FrameTimingV1(
        identity=_frame(),
        camera_source_time_ns=1_725_000_000_000_000_000,
        host_clock_id="host-monotonic-1",
        publication_sequence=8,
        first_unique_packet_monotonic_ns=100,
        final_unique_packet_monotonic_ns=110,
        reassembly_complete_monotonic_ns=110,
        decode_start_monotonic_ns=111,
        decode_end_monotonic_ns=120,
        publish_monotonic_ns=121,
    )
    assert FrameTimingV1.from_primitive(timing.to_primitive()) == timing
    with pytest.raises(ValueError, match="monotonic"):
        dataclasses.replace(timing, decode_end_monotonic_ns=109)
    with pytest.raises(ValueError, match="<="):
        dataclasses.replace(timing, camera_source_time_ns=1 << 64)


def test_prediction_time_requires_order_and_characterized_claims():
    assert PredictionTimeV1.from_primitive(_prediction().to_primitive()) == _prediction()
    with pytest.raises(ValueError, match="measurement <= decision"):
        dataclasses.replace(_prediction(), decision_time_monotonic_ns=999)
    with pytest.raises(ValueError, match="model and nonzero uncertainty"):
        dataclasses.replace(_prediction(), delay_model_id=None)
    with pytest.raises(ValueError, match="own model id"):
        dataclasses.replace(
            _prediction(),
            measurement_time_basis=MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
        )
    with pytest.raises(ValueError, match="uncertainty must be nonzero"):
        dataclasses.replace(
            _prediction(),
            measurement_time_basis=MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
            measurement_time_model_id="camera-clock-fit-v1",
            measurement_uncertainty_ns=0,
        )
    with pytest.raises(ValueError, match="must equal decision"):
        dataclasses.replace(
            _prediction(),
            prediction_basis=PredictionBasis.DECISION_TIME,
            delay_model_id=None,
            delay_uncertainty_ns=0,
        )
    decision = dataclasses.replace(
        _prediction(),
        prediction_time_monotonic_ns=_prediction().decision_time_monotonic_ns,
        prediction_basis=PredictionBasis.DECISION_TIME,
        delay_model_id=None,
        delay_uncertainty_ns=0,
    )
    assert decision.prediction_time_monotonic_ns == decision.decision_time_monotonic_ns
    with pytest.raises(ValueError, match="proxy uncertainty"):
        dataclasses.replace(_prediction(), measurement_uncertainty_ns=0)


def test_latency_event_is_correlated_and_failure_reason_is_exact():
    event = LatencyEventV1(
        event_sequence=1,
        host_clock_id="host-monotonic-1",
        monotonic_ns=100,
        kind=LatencyEventKind.FRAME_PUBLISHED,
        frame=_frame(),
        control_tick_id=None,
        command_id=None,
        sensor_sample_id=None,
        sensor_source_time_ns=None,
        outcome=EventOutcome.OK,
        reason_code=None,
        queue_depth=0,
    )
    assert LatencyEventV1.from_primitive(event.to_primitive()) == event
    with pytest.raises(ValueError, match="require a frame"):
        dataclasses.replace(event, frame=None)
    with pytest.raises(ValueError, match="require a reason"):
        dataclasses.replace(event, outcome=EventOutcome.DROPPED)
    with pytest.raises(ValueError, match="cannot carry"):
        dataclasses.replace(event, reason_code="unexpected")


def test_frame_and_latency_sequences_reject_regression_and_bad_correlation():
    first_timing = _frame_timing(
        frame=_frame(frame_id=41),
        publication_sequence=8,
        camera_source_time_ns=100,
    )
    next_timing = _frame_timing(
        frame=_frame(frame_id=42),
        publication_sequence=9,
        camera_source_time_ns=101,
        base_monotonic_ns=2_000_000_100,
    )
    validate_frame_timing_sequence((first_timing, next_timing))
    with pytest.raises(ValueError, match="publication_sequence"):
        validate_frame_timing_sequence(
            (first_timing, dataclasses.replace(next_timing, publication_sequence=8))
        )
    with pytest.raises(ValueError, match="source time"):
        validate_frame_timing_sequence(
            (first_timing, dataclasses.replace(next_timing, camera_source_time_ns=100))
        )
    publish_regression = _frame_timing(
        frame=_frame(frame_id=42),
        publication_sequence=9,
        camera_source_time_ns=101,
        base_monotonic_ns=1_999_999_949,
    )
    with pytest.raises(ValueError, match="publish time must progress"):
        validate_frame_timing_sequence((first_timing, publish_regression))
    with pytest.raises(ValueError, match="repeats a frame identity"):
        validate_frame_timing_sequence((first_timing, first_timing))
    with pytest.raises(ValueError, match="generation cannot regress"):
        validate_frame_timing_sequence(
            (
                first_timing,
                dataclasses.replace(
                    next_timing,
                    identity=_frame(generation=2, frame_id=42),
                ),
            )
        )
    validate_frame_timing_sequence(
        (
            first_timing,
            dataclasses.replace(
                first_timing,
                host_clock_id="other-host-monotonic",
            ),
        )
    )

    due = LatencyEventV1(
        event_sequence=8,
        host_clock_id="host-monotonic-1",
        monotonic_ns=1_000_000_000,
        kind=LatencyEventKind.CONTROL_TICK_DUE,
        frame=None,
        control_tick_id=71,
        command_id=None,
        sensor_sample_id=None,
        sensor_source_time_ns=None,
        outcome=EventOutcome.OK,
        reason_code=None,
        queue_depth=0,
    )
    tick_start = dataclasses.replace(
        due,
        event_sequence=9,
        monotonic_ns=1_000_000_100,
        kind=LatencyEventKind.CONTROL_TICK_START,
        frame=_frame(),
    )
    send_start = LatencyEventV1(
        event_sequence=10,
        host_clock_id="host-monotonic-1",
        monotonic_ns=1_000_000_200,
        kind=LatencyEventKind.COMMAND_SEND_START,
        frame=_frame(),
        control_tick_id=71,
        command_id=73,
        sensor_sample_id=None,
        sensor_source_time_ns=None,
        outcome=EventOutcome.OK,
        reason_code=None,
        queue_depth=0,
    )
    send_end = dataclasses.replace(
        send_start,
        event_sequence=11,
        monotonic_ns=1_000_000_201,
        kind=LatencyEventKind.COMMAND_SEND_END,
    )
    validate_latency_event_sequence((due, tick_start, send_start, send_end))
    with pytest.raises(ValueError, match="no matching earlier"):
        validate_latency_event_sequence((send_end,))
    with pytest.raises(ValueError, match="no matching earlier"):
        validate_latency_event_sequence(
            (due, tick_start, send_start, dataclasses.replace(send_end, command_id=74))
        )
    with pytest.raises(ValueError, match="event_sequence"):
        validate_latency_event_sequence(
            (
                due,
                tick_start,
                send_start,
                dataclasses.replace(send_end, event_sequence=9),
            )
        )
    with pytest.raises(ValueError, match="control_tick_id"):
        dataclasses.replace(send_start, control_tick_id=None)
    with pytest.raises(ValueError, match="command_id"):
        dataclasses.replace(send_start, command_id=None)


@pytest.mark.parametrize(
    "kind,outcome",
    [
        (LatencyEventKind.FRAME_DROPPED, EventOutcome.DROPPED),
        (LatencyEventKind.CONTROL_TICK_SKIPPED, EventOutcome.SKIPPED),
        (LatencyEventKind.DEADLINE_MISSED, EventOutcome.SKIPPED),
    ],
)
def test_non_success_latency_kinds_require_their_exact_outcome(kind, outcome):
    fields = {
        "event_sequence": 1,
        "host_clock_id": "host-monotonic-1",
        "monotonic_ns": 100,
        "kind": kind,
        "frame": _frame() if kind is LatencyEventKind.FRAME_DROPPED else None,
        "control_tick_id": None if kind is LatencyEventKind.FRAME_DROPPED else 71,
        "command_id": None,
        "sensor_sample_id": None,
        "sensor_source_time_ns": None,
        "outcome": outcome,
        "reason_code": "bounded_drop",
        "queue_depth": 1,
    }
    LatencyEventV1(**fields)
    with pytest.raises(ValueError, match="requires outcome"):
        LatencyEventV1(**{**fields, "outcome": EventOutcome.ERROR})


@pytest.mark.parametrize(
    "kind",
    [LatencyEventKind.GYRO_SAMPLE, LatencyEventKind.ACTUATOR_SAMPLE],
)
def test_sensor_latency_events_use_sensor_identity_without_false_command_binding(kind):
    sample = LatencyEventV1(
        event_sequence=1,
        host_clock_id="host-monotonic-1",
        monotonic_ns=100,
        kind=kind,
        frame=None,
        control_tick_id=None,
        command_id=None,
        sensor_sample_id=19,
        sensor_source_time_ns=123_456,
        outcome=EventOutcome.OK,
        reason_code=None,
        queue_depth=0,
    )
    assert LatencyEventV1.from_primitive(sample.to_primitive()) == sample
    with pytest.raises(ValueError, match="sensor_source_time_ns"):
        dataclasses.replace(sample, sensor_source_time_ns=None)


def test_latency_sequence_enforces_due_skip_and_50hz_command_policy():
    def event(
        sequence,
        monotonic_ns,
        kind,
        tick,
        *,
        command_id=None,
        outcome=EventOutcome.OK,
        reason=None,
    ):
        return LatencyEventV1(
            event_sequence=sequence,
            host_clock_id="host-monotonic-1",
            monotonic_ns=monotonic_ns,
            kind=kind,
            frame=_frame() if kind is not LatencyEventKind.CONTROL_TICK_DUE else None,
            control_tick_id=tick,
            command_id=command_id,
            sensor_sample_id=None,
            sensor_source_time_ns=None,
            outcome=outcome,
            reason_code=reason,
            queue_depth=0,
        )

    due = event(1, 1_000_000_000, LatencyEventKind.CONTROL_TICK_DUE, 1)
    start = event(2, 1_000_000_001, LatencyEventKind.CONTROL_TICK_START, 1)
    skipped = event(
        3,
        1_000_000_002,
        LatencyEventKind.CONTROL_TICK_SKIPPED,
        1,
        outcome=EventOutcome.SKIPPED,
        reason="missed_tick",
    )
    with pytest.raises(ValueError, match="earlier due"):
        validate_latency_event_sequence((start,))
    with pytest.raises(ValueError, match="cannot also be skipped"):
        validate_latency_event_sequence((due, start, skipped))

    missed = event(
        2,
        1_000_000_001,
        LatencyEventKind.DEADLINE_MISSED,
        1,
        outcome=EventOutcome.SKIPPED,
        reason="deadline",
    )
    missed_skip = dataclasses.replace(skipped, event_sequence=3)
    with pytest.raises(ValueError, match="must end in a skipped"):
        validate_latency_event_sequence((due, missed))
    validate_latency_event_sequence((due, missed, missed_skip))

    too_soon_due = event(
        2,
        1_019_999_999,
        LatencyEventKind.CONTROL_TICK_DUE,
        2,
    )
    with pytest.raises(ValueError, match="due faster than 50 Hz"):
        validate_latency_event_sequence((due, too_soon_due))

    due_2 = event(2, 1_020_000_000, LatencyEventKind.CONTROL_TICK_DUE, 2)
    stale_start = event(
        3,
        1_020_000_001,
        LatencyEventKind.CONTROL_TICK_START,
        1,
    )
    with pytest.raises(ValueError, match="must be skipped before a newer due tick"):
        validate_latency_event_sequence((due, due_2, stale_start))

    tick_end = event(
        3,
        1_000_000_002,
        LatencyEventKind.CONTROL_TICK_END,
        1,
    )
    duplicate_start = event(
        4,
        1_000_000_003,
        LatencyEventKind.CONTROL_TICK_START,
        1,
    )
    with pytest.raises(ValueError, match="control tick can start only once"):
        validate_latency_event_sequence((due, start, tick_end, duplicate_start))

    early_skip = dataclasses.replace(skipped, event_sequence=2)
    late_missed = dataclasses.replace(
        missed,
        event_sequence=3,
        monotonic_ns=1_000_000_002,
    )
    with pytest.raises(ValueError, match="deadline miss must precede"):
        validate_latency_event_sequence((due, early_skip, late_missed))

    stale_send = event(
        4,
        1_020_000_001,
        LatencyEventKind.COMMAND_SEND_START,
        1,
        command_id=10,
    )
    later_due_2 = dataclasses.replace(due_2, event_sequence=3)
    with pytest.raises(ValueError, match="superseded control tick cannot send"):
        validate_latency_event_sequence((due, start, later_due_2, stale_send))

    send_start_1 = event(
        3,
        1_000_000_010,
        LatencyEventKind.COMMAND_SEND_START,
        1,
        command_id=10,
    )
    send_end_1 = dataclasses.replace(
        send_start_1,
        event_sequence=4,
        monotonic_ns=1_000_000_011,
        kind=LatencyEventKind.COMMAND_SEND_END,
    )
    due_2 = event(5, 1_020_000_000, LatencyEventKind.CONTROL_TICK_DUE, 2)
    start_2 = event(6, 1_020_000_001, LatencyEventKind.CONTROL_TICK_START, 2)
    send_start_2 = event(
        7,
        1_020_000_009,
        LatencyEventKind.COMMAND_SEND_START,
        2,
        command_id=11,
    )
    with pytest.raises(ValueError, match="sends cannot occur faster than 50 Hz"):
        validate_latency_event_sequence(
            (
                due,
                start,
                send_start_1,
                send_end_1,
                due_2,
                start_2,
                send_start_2,
            )
        )


def test_feature_covariance_is_immutable_symmetric_and_psd():
    covariance = FeatureCovarianceV1(
        "test-center-v1",
        ("center_x_norm", "center_y_norm"),
        ((1.0, 0.2), (0.2, 2.0)),
    )
    assert hash(covariance)
    assert FeatureCovarianceV1.from_primitive(covariance.to_primitive()) == covariance
    encoded = covariance.to_primitive()
    encoded["matrix"][0] = tuple(encoded["matrix"][0])
    with pytest.raises(TypeError, match="must be arrays"):
        FeatureCovarianceV1.from_primitive(encoded)
    with pytest.raises(ValueError, match="symmetric"):
        FeatureCovarianceV1(
            "test-center-v1",
            ("center_x_norm", "center_y_norm"),
            ((1.0, 0.2), (0.1, 1.0)),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        FeatureCovarianceV1(
            "test-center-v1",
            ("center_x_norm", "center_y_norm"),
            ((1.0, 2.0), (2.0, 1.0)),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        FeatureCovarianceV1(
            "test-center-v1",
            ("center_x_norm", "center_y_norm"),
            ((0.0, 0.0), (0.0, 1.0)),
        )


def test_visible_gate0_adapter_round_trip_preserves_legacy_pixel_target():
    timing = _frame_timing(camera_source_time_ns=123_456)
    observation = _observation(frame_timing=timing)
    assert observation.clipping == FrameEdge.NONE
    assert observation.outer_edges == EdgeSetV1()
    assert observation.inner_edges == EdgeSetV1()
    assert observation.inner_corners_norm == (None, None, None, None)
    assert observation.log_scale is None
    assert observation.health.value == "degraded"
    assert GateObservationV1.from_primitive(observation.to_primitive()) == observation
    assert observation_to_legacy_gate_target_fields(
        observation,
        frame_timing=timing,
        expected_authority=_authority(),
    ) == {
        "frame_id": 41,
        "sim_time_ns": 123_456,
        "received_monotonic_s": 2.0,
        "center_x": 322,
        "center_y": 174,
        "bbox": (282, 134, 80, 81),
        "confidence": 0.9,
    }


def test_top_clipped_gate1_is_censored_degraded_and_never_gets_placeholder_pose():
    timing = _frame_timing(
        publication_sequence=9,
        camera_source_time_ns=999,
        base_monotonic_ns=1_999_999_951,
    )
    observation = _observation(
        detection=_detection(bbox=(477, 0, 115, 97), center=(536, 48)),
        frame_timing=timing,
        authority=_authority(gate_index=1, gate_epoch=1),
        candidate_id="gate-1-candidate-0",
    )
    assert observation.clipping == FrameEdge.TOP
    assert observation.outer_edges.top is None
    assert observation.outer_edges == EdgeSetV1()
    assert observation.inner_corners_norm == (None, None, None, None)
    assert observation.log_scale is None
    assert observation.covariance.feature_order == (
        "center_x_norm",
        "center_y_norm",
    )
    encoded = json.dumps(observation.to_primitive(), sort_keys=True)
    assert "estimated_distance" not in encoded
    assert "corners\"" not in encoded
    assert "censored_top" in observation.health_reason
    legacy = observation_to_legacy_gate_target_fields(
        observation,
        frame_timing=timing,
        expected_authority=_authority(gate_index=1, gate_epoch=1),
    )
    assert legacy["bbox"] == (477, 0, 115, 97)
    assert legacy["center_x"] == 536
    assert legacy["center_y"] == 48


def test_observation_replay_projection_keeps_historical_exact_shape():
    replay_detection = observation_to_replay_detection_v1(
        _observation(), selector_eligible=True
    )
    assert set(replay_detection) == {"center_px", "selector_eligible", "confidence"}
    _validate_processor_detections([replay_detection])


def test_observation_batch_has_one_frame_binding_and_unique_candidate_ids():
    first = _observation()
    second = dataclasses.replace(
        first,
        candidate_id="gate-0-candidate-1",
        confidence=0.8,
    )
    validate_gate_observation_batch((first, second))
    with pytest.raises(ValueError, match="candidate_id must be unique"):
        validate_gate_observation_batch(
            (first, dataclasses.replace(first, confidence=0.8))
        )
    with pytest.raises(ValueError, match="timing/authority bindings"):
        validate_gate_observation_batch(
            (
                first,
                dataclasses.replace(
                    second,
                    frame_timing=dataclasses.replace(
                        second.frame_timing,
                        camera_source_time_ns=(
                            second.frame_timing.camera_source_time_ns + 1
                        ),
                    ),
                ),
            )
        )


def test_visual_identity_cannot_cross_a_safety_gate_epoch():
    first = _observation()
    next_authority = _authority(gate_index=1, gate_epoch=1)
    with pytest.raises(ValueError, match="authority .* watermark"):
        dataclasses.replace(
            first,
            authority=next_authority,
            candidate_id="crossing-residue",
        )
    fresh_timing = _frame_timing(
        frame=_frame(frame_id=42),
        publication_sequence=9,
        camera_source_time_ns=first.frame_timing.camera_source_time_ns + 1,
        base_monotonic_ns=1_999_999_951,
    )
    next_gate = _observation(
        frame_timing=fresh_timing,
        authority=next_authority,
        candidate_id="gate-1-fresh",
    )
    assert first.frame != next_gate.frame
    assert first.authority != next_gate.authority


def test_gate_observation_exact_codec_rejects_unknown_fields_and_masks():
    encoded = _observation().to_primitive()
    encoded["unknown"] = 1
    with pytest.raises(ValueError, match="fields must be exact"):
        GateObservationV1.from_primitive(encoded)
    encoded = _observation().to_primitive()
    encoded["clipping"] = 16
    with pytest.raises(ValueError, match="unknown mask bits"):
        GateObservationV1.from_primitive(encoded)


def test_gate_observation_rejects_out_of_frame_geometry_and_visible_clipping():
    with pytest.raises(ValueError, match="start_norm"):
        LineSegmentV1(start_norm=(-1.1, 0.0), end_norm=(0.0, 0.0))
    with pytest.raises(ValueError, match="inner corner"):
        dataclasses.replace(
            _observation(),
            inner_corners_norm=((0.0, -1.1), None, None, None),
        )
    top_edge = LineSegmentV1(start_norm=(-0.5, -1.0), end_norm=(0.5, -1.0))
    with pytest.raises(ValueError, match="clipped outer edge"):
        dataclasses.replace(
            _observation(),
            outer_edges=EdgeSetV1(top=top_edge),
            clipping=FrameEdge.TOP,
        )
    with pytest.raises(ValueError, match="positive width and height"):
        dataclasses.replace(
            _observation(), support_bounds_norm=(0.4, 0.2, 0.4, 0.8)
        )
    with pytest.raises(ValueError, match="matching clipping"):
        dataclasses.replace(_observation(), center_norm=(1.1, 0.0))
    with pytest.raises(ValueError, match="uncertainty must be nonzero"):
        dataclasses.replace(
            _observation(),
            measurement_time_basis=MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
            measurement_time_model_id="camera-clock-fit-v1",
            measurement_uncertainty_ns=0,
        )


def test_fitted_aperture_has_exact_scale_skew_covariance_and_fit_support():
    corners = (
        (-0.4, -0.4),
        (0.4, -0.4),
        (0.4, 0.4),
        (-0.4, 0.4),
    )
    covariance = FeatureCovarianceV1(
        model_id="inner-aperture-fit-v1",
        feature_order=(
            "center_x_norm",
            "center_y_norm",
            "log_scale",
            "skew_x",
            "skew_y",
        ),
        matrix=tuple(
            tuple(0.01 if row == column else 0.0 for column in range(5))
            for row in range(5)
        ),
    )
    fitted = dataclasses.replace(
        _observation(),
        center_norm=(0.0, 0.0),
        inner_corners_norm=corners,
        fitted_inner_aperture_corners_norm=corners,
        geometry_model_id="inner-aperture-fit-v1",
        log_scale=math.log(0.8),
        projective_skew=(0.0, 0.0),
        covariance=covariance,
        fit=FitDiagnosticsV1(residual_rms=0.01, inlier_count=20, support_count=24),
        health=ObservationHealth.NOMINAL,
        health_reason=None,
        provenance="vq2_inner_aperture_fit",
    )
    assert GateObservationV1.from_primitive(fitted.to_primitive()) == fitted
    with pytest.raises(ValueError, match="require a fitted inner aperture"):
        dataclasses.replace(
            _observation(),
            geometry_model_id="fabricated-fit",
            log_scale=0.0,
            projective_skew=(0.0, 0.0),
            covariance=covariance,
        )
    with pytest.raises(ValueError, match="nonzero support"):
        dataclasses.replace(fitted, fit=FitDiagnosticsV1(None, 0, 0))
    with pytest.raises(ValueError, match="does not match fitted aperture area"):
        dataclasses.replace(fitted, log_scale=0.0)
    with pytest.raises(ValueError, match="diagonal intersection"):
        dataclasses.replace(fitted, center_norm=(0.01, 0.0))
    cyclically_relabelled = (corners[1], corners[2], corners[3], corners[0])
    with pytest.raises(ValueError, match="corner labels are ambiguous"):
        dataclasses.replace(
            fitted,
            inner_corners_norm=cyclically_relabelled,
            fitted_inner_aperture_corners_norm=cyclically_relabelled,
        )

    top_clipped_corners = (
        (-0.4, -1.2),
        (0.4, -1.2),
        (0.4, -0.4),
        (-0.4, -0.4),
    )
    top_clipped = dataclasses.replace(
        fitted,
        center_norm=(0.0, -0.8),
        inner_corners_norm=(None, None, (0.4, -0.4), (-0.4, -0.4)),
        fitted_inner_aperture_corners_norm=top_clipped_corners,
        clipping=FrameEdge.TOP,
    )
    assert top_clipped.clipping is FrameEdge.TOP
    with pytest.raises(ValueError, match="requires matching clipping"):
        dataclasses.replace(top_clipped, clipping=FrameEdge.NONE)


def test_legacy_observation_projection_requires_exact_timing_and_authority():
    observation = _observation()
    with pytest.raises(ValueError, match="authority"):
        observation_to_legacy_gate_target_fields(
            observation,
            frame_timing=observation.frame_timing,
            expected_authority=_authority(gate_index=1, gate_epoch=1),
        )
    with pytest.raises(ValueError, match="frame_timing"):
        observation_to_legacy_gate_target_fields(
            observation,
            frame_timing=dataclasses.replace(
                observation.frame_timing, camera_source_time_ns=123
            ),
            expected_authority=observation.authority,
        )
    center_outside_bbox = dataclasses.replace(
        observation,
        center_norm=(0.9, 0.0),
    )
    with pytest.raises(ValueError, match="center must lie inside"):
        observation_to_legacy_gate_target_fields(
            center_outside_bbox,
            frame_timing=center_outside_bbox.frame_timing,
            expected_authority=center_outside_bbox.authority,
        )
    subpixel = dataclasses.replace(
        observation,
        center_norm=(0.123456, 0.0),
    )
    with pytest.raises(ValueError, match="legacy pixel grid"):
        observation_to_legacy_gate_target_fields(
            subpixel,
            frame_timing=subpixel.frame_timing,
            expected_authority=subpixel.authority,
        )


def test_legacy_adapter_rejects_center_outside_bbox_and_replay_offscreen_selection():
    with pytest.raises(ValueError, match="support bbox"):
        _observation(detection=_detection(center=(500, 174)))
    offscreen = dataclasses.replace(
        _observation(),
        center_norm=(1.1, 0.0),
        clipping=FrameEdge.RIGHT,
    )
    with pytest.raises(ValueError, match="selector eligible"):
        observation_to_replay_detection_v1(offscreen, selector_eligible=True)
    assert observation_to_replay_detection_v1(
        offscreen, selector_eligible=False
    )["selector_eligible"] is False
    with pytest.raises(ValueError, match="not representable"):
        observation_to_legacy_gate_target_fields(
            offscreen,
            frame_timing=offscreen.frame_timing,
            expected_authority=offscreen.authority,
        )
    with pytest.raises(ValueError, match="declared image"):
        _observation(
            detection=_detection(
                bbox=(600, 100, 40, 100), center=(640, 150)
            )
        )
    with pytest.raises(ValueError, match="support bbox"):
        _observation(
            detection=_detection(
                bbox=(500, 100, 40, 100), center=(540, 150)
            )
        )


def test_relative_state_round_trip_and_bounded_coasting_health():
    state = _relative_state()
    assert RelativeGateStateV1.from_primitive(state.to_primitive()) == state
    assert state.timing.source_frame == _frame()
    assert state.authority.expected_gate_index == 1
    assert state.dropout_count == 2
    assert state.health is RelativeStateHealth.COASTING


def test_relative_state_source_binds_exact_observation_and_blocks_cutover_relabel():
    observation = _observation(
        frame_timing=_frame_timing(
            publication_sequence=9,
            base_monotonic_ns=1_999_999_951,
        ),
        authority=_authority(gate_index=1, gate_epoch=1),
        candidate_id="gate-1-candidate-0",
    )
    timing = dataclasses.replace(
        _prediction(),
        source_frame=observation.frame,
        source_frame_publication_sequence=(
            observation.frame_timing.publication_sequence
        ),
        source_frame_publish_monotonic_ns=(
            observation.frame_timing.publish_monotonic_ns
        ),
        measurement_time_monotonic_ns=observation.measurement_time_monotonic_ns,
        measurement_time_basis=observation.measurement_time_basis,
        measurement_time_model_id=observation.measurement_time_model_id,
        measurement_uncertainty_ns=observation.measurement_uncertainty_ns,
    )
    state = _relative_state(
        timing=timing,
        last_clipping=observation.clipping,
        outer_visibility=observation.outer_edges.visibility,
        inner_visibility=observation.inner_edges.visibility,
    )
    validate_relative_gate_state_source(state, observation)

    invented_publication = dataclasses.replace(
        state,
        timing=dataclasses.replace(
            state.timing,
            source_frame_publication_sequence=(
                observation.frame_timing.publication_sequence + 1
            ),
            source_frame_publish_monotonic_ns=(
                observation.frame_timing.publish_monotonic_ns + 1
            ),
        ),
    )
    with pytest.raises(ValueError, match="publication sequence"):
        validate_relative_gate_state_source(invented_publication, observation)

    old_gate_observation = _observation()
    with pytest.raises(ValueError, match="authority"):
        validate_relative_gate_state_source(state, old_gate_observation)
    with pytest.raises(ValueError, match="clipping"):
        validate_relative_gate_state_source(
            dataclasses.replace(state, last_clipping=FrameEdge.TOP),
            observation,
        )
    unusable = dataclasses.replace(
        observation,
        health=ObservationHealth.UNUSABLE,
        health_reason="rejected_candidate",
    )
    with pytest.raises(ValueError, match="unusable observation"):
        validate_relative_gate_state_source(state, unusable)


def test_relative_state_rejects_partial_metric_pose_and_wrong_covariance_order():
    with pytest.raises(ValueError, match="all-or-none"):
        _relative_state(metric_position_body_frd_m=(1.0, 2.0, 3.0))
    bad_covariance = FeatureCovarianceV1(
        "wrong-order-v1",
        ("bearing_y_norm", "bearing_x_norm"),
        ((1.0, 0.0), (0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="wrong feature order"):
        _relative_state(covariance=bad_covariance)


def test_relative_state_rejects_inconsistent_innovation_and_dropout_health():
    with pytest.raises(ValueError, match="accepted innovation"):
        _relative_state(
            normalized_innovation_squared=10.0,
            innovation_gate_threshold=9.21,
            innovation_accepted=True,
        )
    with pytest.raises(ValueError, match="all-or-none"):
        _relative_state(
            normalized_innovation_squared=2.0,
            innovation_gate_threshold=9.21,
            innovation_accepted=None,
        )
    with pytest.raises(ValueError, match="healthy states cannot contain dropout"):
        _relative_state(
            health=RelativeStateHealth.HEALTHY,
            health_reason=None,
            dropout_count=1,
        )
    with pytest.raises(ValueError, match="require at least one dropout"):
        _relative_state(dropout_count=0)
    with pytest.raises(ValueError, match="current-frame innovation"):
        _relative_state(
            normalized_innovation_squared=2.0,
            innovation_gate_threshold=9.21,
            innovation_accepted=True,
        )
    with pytest.raises(ValueError, match="rejected innovation"):
        _relative_state(
            health=RelativeStateHealth.HEALTHY,
            health_reason=None,
            dropout_count=0,
            normalized_innovation_squared=10.0,
            innovation_gate_threshold=9.21,
            innovation_accepted=False,
        )


def test_relative_state_sequence_binds_candidate_and_prevents_duplicate_updates():
    first = _relative_state()
    same_observation_prediction = dataclasses.replace(
        first,
        state_sequence=13,
        timing=dataclasses.replace(
            first.timing,
            prediction_time_monotonic_ns=(
                first.timing.prediction_time_monotonic_ns + 1
            ),
        ),
    )
    next_timing = dataclasses.replace(
        first.timing,
        source_frame=_frame(frame_id=42),
        source_frame_publication_sequence=10,
        source_frame_publish_monotonic_ns=2_000_000_013,
        decision_time_monotonic_ns=2_000_000_021,
        prediction_time_monotonic_ns=2_000_000_026,
    )
    next_observation = dataclasses.replace(
        first,
        timing=next_timing,
        state_sequence=14,
        measurement_update_sequence=6,
        source_candidate_id="gate-1-candidate-1",
    )
    validate_relative_gate_state_sequence(
        (first, same_observation_prediction, next_observation)
    )
    with pytest.raises(ValueError, match="cannot be applied more than once"):
        validate_relative_gate_state_sequence(
            (
                first,
                dataclasses.replace(
                    same_observation_prediction,
                    measurement_update_sequence=6,
                ),
            )
        )
    with pytest.raises(ValueError, match="multiple active/shadow tracks"):
        validate_relative_gate_state_sequence(
            (
                first,
                dataclasses.replace(
                    first,
                    tracker_id="shadow-gate-1",
                    track_role=TrackRole.SHADOW,
                ),
            )
        )
    refreshed_authority = dataclasses.replace(
        first.authority,
        race_status_sequence=first.authority.race_status_sequence + 1,
        race_status_boot_ms=first.authority.race_status_boot_ms + 1,
    )
    with pytest.raises(ValueError, match="multiple active/shadow tracks"):
        validate_relative_gate_state_sequence(
            (
                first,
                dataclasses.replace(
                    first,
                    authority=refreshed_authority,
                    tracker_id="shadow-gate-1",
                    track_role=TrackRole.SHADOW,
                ),
            )
        )
    regressed_authority = dataclasses.replace(
        first.authority,
        race_status_sequence=first.authority.race_status_sequence - 1,
        race_status_boot_ms=first.authority.race_status_boot_ms - 1,
    )
    with pytest.raises(ValueError, match="authority snapshot regressed"):
        validate_relative_gate_state_sequence(
            (
                first,
                dataclasses.replace(
                    same_observation_prediction,
                    authority=regressed_authority,
                ),
            )
        )
    with pytest.raises(ValueError, match="expected gate index changed"):
        validate_relative_gate_state_sequence(
            (
                first,
                dataclasses.replace(
                    first,
                    authority=dataclasses.replace(
                        first.authority,
                        expected_gate_index=2,
                    ),
                    tracker_id="shadow-gate-2",
                    track_role=TrackRole.SHADOW,
                ),
            )
        )
    replayed_first_observation = dataclasses.replace(
        first,
        timing=dataclasses.replace(
            first.timing,
            decision_time_monotonic_ns=2_000_000_022,
            prediction_time_monotonic_ns=2_000_000_027,
        ),
        state_sequence=15,
        measurement_update_sequence=7,
    )
    with pytest.raises(ValueError, match="cannot be applied more than once"):
        validate_relative_gate_state_sequence(
            (first, next_observation, replayed_first_observation)
        )


def test_command_proposal_and_supervisor_approval_round_trip_are_lossless():
    proposal = _proposal()
    encoded = proposal.to_primitive()
    assert "source_state_monotonic_ns" not in encoded
    assert CommandProposalV1.from_primitive(encoded) == proposal
    assert (proposal.proposal_id, proposal.control_tick_id) == (72, 71)
    assert proposal.source_tracker_id == "active-gate-0"
    assert proposal.source_track_role is TrackRole.ACTIVE
    validate_command_proposal_source(proposal, _proposal_source_state())
    approved = _approved(proposal)
    assert SupervisorApprovedCommandV1.from_primitive(approved.to_primitive()) == approved
    command = approved_command_to_attitude_rate_command(
        approved,
        host_clock_id="host-monotonic-1",
        send_monotonic_ns=_COMMAND_SEND_NS,
        expected_control_tick_id=71,
        expected_control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
        expected_authority=_authority(),
        expected_safety_policy_id="vq2-training-envelope-v1",
        maximum_approval_age_ns=20,
    )
    assert command == AttitudeRateCommand(0.1, -0.2, 0.0, 0.27)
    replay = proposal_to_replay_command_v1(proposal)
    assert replay == {
        "roll_rate": 0.1,
        "pitch_rate": -0.2,
        "yaw_rate": 0.0,
        "thrust": 0.27,
    }
    _validate_processor_command(replay)


def test_command_source_reference_is_exact_and_allows_a_future_prediction_horizon():
    proposal = _proposal()
    state = _proposal_source_state()
    validate_command_proposal_source(proposal, state)

    future_prediction_ns = _COMMAND_PROPOSAL_NS + 10
    future_state = dataclasses.replace(
        state,
        timing=dataclasses.replace(
            state.timing,
            prediction_time_monotonic_ns=future_prediction_ns,
        ),
    )
    future_proposal = dataclasses.replace(
        proposal,
        source_state_prediction_monotonic_ns=future_prediction_ns,
    )
    assert future_proposal.source_state_prediction_monotonic_ns > (
        future_proposal.proposal_monotonic_ns
    )
    validate_command_proposal_source(future_proposal, future_state)

    with pytest.raises(ValueError, match="tracker id"):
        validate_command_proposal_source(
            dataclasses.replace(proposal, source_tracker_id="different-tracker"),
            state,
        )
    with pytest.raises(ValueError, match="track role"):
        validate_command_proposal_source(
            proposal,
            dataclasses.replace(state, track_role=TrackRole.SHADOW),
        )
    with pytest.raises(ValueError, match="state decision time"):
        validate_command_proposal_source(
            dataclasses.replace(
                proposal,
                source_state_decision_monotonic_ns=(
                    proposal.source_state_decision_monotonic_ns + 1
                ),
            ),
            state,
        )


def test_only_exact_zero_proposal_may_omit_source_state_and_shadow_cannot_drive():
    proposal = _proposal()
    no_source = {
        "source_state_decision_monotonic_ns": None,
        "source_state_prediction_monotonic_ns": None,
        "source_frame": None,
        "source_frame_publication_sequence": None,
        "source_frame_publish_monotonic_ns": None,
        "source_tracker_id": None,
        "source_track_role": None,
        "source_state_sequence": None,
        "source_measurement_update_sequence": None,
        "source_candidate_id": None,
    }
    with pytest.raises(ValueError, match="nonzero command proposal requires"):
        dataclasses.replace(proposal, **no_source)
    source_less_zero = dataclasses.replace(
        proposal,
        requested_body_rates_rad_s=(0.0, 0.0, 0.0),
        requested_thrust=0.0,
        **no_source,
    )
    assert source_less_zero.is_exact_zero
    assert CommandProposalV1.from_primitive(
        source_less_zero.to_primitive()
    ) == source_less_zero
    with pytest.raises(ValueError, match="has no relative state"):
        validate_command_proposal_source(source_less_zero, _proposal_source_state())

    shadow_proposal = dataclasses.replace(
        proposal,
        source_track_role=TrackRole.SHADOW,
    )
    validate_command_proposal_source(
        shadow_proposal,
        dataclasses.replace(_proposal_source_state(), track_role=TrackRole.SHADOW),
    )
    with pytest.raises(ValueError, match="active source track"):
        _approved(shadow_proposal)
    shadow_limited_to_zero = _approved(
        shadow_proposal,
        approved_body_rates_rad_s=(0.0, 0.0, 0.0),
        approved_thrust=0.0,
        safety_limit_reason="shadow_track_no_authority",
    )
    assert shadow_limited_to_zero.approved_body_rates_rad_s == (0.0, 0.0, 0.0)
    assert shadow_limited_to_zero.approved_thrust == 0.0


def test_supervisor_projection_preserves_explicitly_approved_values_without_clamping():
    proposal = _proposal(AttitudeRateCommand(2.0, -3.0, 0.4, 0.9))
    approved = _approved(
        proposal,
        approved_body_rates_rad_s=(0.25, -0.25, 0.0),
        safety_limit_reason="build3385_rate_envelope",
    )
    command = approved_command_to_attitude_rate_command(
        approved,
        host_clock_id="host-monotonic-1",
        send_monotonic_ns=_COMMAND_SEND_NS,
        expected_control_tick_id=71,
        expected_control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
        expected_authority=_authority(),
        expected_safety_policy_id="vq2-training-envelope-v1",
        maximum_approval_age_ns=20,
    )
    assert (command.roll_rate, command.pitch_rate, command.yaw_rate, command.thrust) == (
        0.25,
        -0.25,
        0.0,
        0.9,
    )


@pytest.mark.parametrize(
    "rates",
    [
        (0.250_001, 0.0, 0.0),
        (0.0, -0.250_001, 0.0),
        (0.0, 0.0, 0.001),
    ],
)
def test_supervisor_approval_enforces_authoritative_static_rate_bounds(rates):
    proposal = _proposal(AttitudeRateCommand(*rates, 0.27))
    with pytest.raises(ValueError, match="approved (roll/pitch|yaw rate)"):
        _approved(proposal)


def test_proposal_cannot_bypass_supervisor_and_approval_is_fresh_epoch_scoped():
    proposal = _proposal()
    approved = _approved(proposal)
    with pytest.raises(ValueError, match="authority publication watermark"):
        dataclasses.replace(
            proposal,
            authority=_authority(gate_index=1, gate_epoch=1),
        )
    with pytest.raises(TypeError, match="SupervisorApprovedCommandV1"):
        approved_command_to_attitude_rate_command(
            proposal,
            host_clock_id="host-monotonic-1",
            send_monotonic_ns=_COMMAND_SEND_NS,
            expected_control_tick_id=71,
            expected_control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
            expected_authority=_authority(),
            expected_safety_policy_id="vq2-training-envelope-v1",
            maximum_approval_age_ns=20,
        )
    with pytest.raises(ValueError, match="expired"):
        approved_command_to_attitude_rate_command(
            approved,
            host_clock_id="host-monotonic-1",
            send_monotonic_ns=_COMMAND_DEADLINE_NS + 1,
            expected_control_tick_id=71,
            expected_control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
            expected_authority=_authority(),
            expected_safety_policy_id="vq2-training-envelope-v1",
            maximum_approval_age_ns=20,
        )
    with pytest.raises(ValueError, match="safety epoch"):
        approved_command_to_attitude_rate_command(
            approved,
            host_clock_id="host-monotonic-1",
            send_monotonic_ns=_COMMAND_SEND_NS,
            expected_control_tick_id=71,
            expected_control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
            expected_authority=_authority(gate_index=1, gate_epoch=1),
            expected_safety_policy_id="vq2-training-envelope-v1",
            maximum_approval_age_ns=20,
        )
    with pytest.raises(ValueError, match="transport clock"):
        approved_command_to_attitude_rate_command(
            approved,
            host_clock_id="different-clock",
            send_monotonic_ns=_COMMAND_SEND_NS,
            expected_control_tick_id=71,
            expected_control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
            expected_authority=_authority(),
            expected_safety_policy_id="vq2-training-envelope-v1",
            maximum_approval_age_ns=20,
        )
    with pytest.raises(ValueError, match="current control tick"):
        approved_command_to_attitude_rate_command(
            approved,
            host_clock_id="host-monotonic-1",
            send_monotonic_ns=_COMMAND_SEND_NS,
            expected_control_tick_id=72,
            expected_control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
            expected_authority=_authority(),
            expected_safety_policy_id="vq2-training-envelope-v1",
            maximum_approval_age_ns=20,
        )
    with pytest.raises(ValueError, match="trusted control tick deadline"):
        approved_command_to_attitude_rate_command(
            approved,
            host_clock_id="host-monotonic-1",
            send_monotonic_ns=_COMMAND_SEND_NS,
            expected_control_tick_id=71,
            expected_control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS + 1,
            expected_authority=_authority(),
            expected_safety_policy_id="vq2-training-envelope-v1",
            maximum_approval_age_ns=20,
        )
    with pytest.raises(ValueError, match="trusted policy"):
        approved_command_to_attitude_rate_command(
            approved,
            host_clock_id="host-monotonic-1",
            send_monotonic_ns=_COMMAND_SEND_NS,
            expected_control_tick_id=71,
            expected_control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
            expected_authority=_authority(),
            expected_safety_policy_id="untrusted-policy",
            maximum_approval_age_ns=20,
        )
    with pytest.raises(ValueError, match="trusted maximum age"):
        approved_command_to_attitude_rate_command(
            approved,
            host_clock_id="host-monotonic-1",
            send_monotonic_ns=_COMMAND_SEND_NS,
            expected_control_tick_id=71,
            expected_control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
            expected_authority=_authority(),
            expected_safety_policy_id="vq2-training-envelope-v1",
            maximum_approval_age_ns=4,
        )


def test_supervisor_modification_requires_an_exact_limit_reason():
    proposal = _proposal()
    with pytest.raises(ValueError, match="safety_limit_reason"):
        _approved(proposal, approved_thrust=0.2)
    limited = _approved(
        proposal,
        approved_thrust=0.2,
        safety_limit_reason="thrust_envelope",
    )
    assert limited.approved_thrust == 0.2
    with pytest.raises(ValueError, match="safety_limit_reason"):
        _approved(proposal, safety_limit_reason="not_actually_limited")
    zero = dataclasses.replace(
        proposal,
        requested_body_rates_rad_s=(0.0, 0.0, 0.0),
        requested_thrust=0.0,
    )
    with pytest.raises(ValueError, match="only limit proposal magnitude"):
        _approved(
            zero,
            approved_body_rates_rad_s=(0.1, 0.0, 0.0),
            safety_limit_reason="fabricated_limit",
        )
    with pytest.raises(ValueError, match="may not amplify"):
        _approved(
            zero,
            approved_thrust=0.1,
            safety_limit_reason="fabricated_limit",
        )


def test_supervisor_rejects_opposite_sign_minimum_subnormal_without_underflow():
    minimum_subnormal = math.ulp(0.0)
    proposal = _proposal(
        AttitudeRateCommand(minimum_subnormal, 0.0, 0.0, 0.27)
    )
    with pytest.raises(ValueError, match="only limit proposal magnitude"):
        _approved(
            proposal,
            approved_body_rates_rad_s=(-minimum_subnormal, 0.0, 0.0),
            safety_limit_reason="sign_reversal",
        )


def test_approved_command_sequence_rejects_duplicate_or_reordered_authority():
    first = _approved()
    second_proposal = dataclasses.replace(
        _proposal(),
        proposal_id=74,
        control_tick_id=72,
        proposal_monotonic_ns=_COMMAND_PROPOSAL_NS + 20,
        control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS + 20,
        source_state_decision_monotonic_ns=2_000_000_040,
        source_state_prediction_monotonic_ns=_COMMAND_SOURCE_STATE_NS + 20,
    )
    second = _approved(
        second_proposal,
        command_id=75,
        approval_monotonic_ns=_COMMAND_APPROVAL_NS + 20,
        valid_until_monotonic_ns=_COMMAND_DEADLINE_NS + 20,
    )
    validate_approved_command_sequence((first, second))
    with pytest.raises(ValueError, match="repeats a command_id"):
        validate_approved_command_sequence((first, dataclasses.replace(second, command_id=73)))
    with pytest.raises(ValueError, match="control_tick_id"):
        validate_approved_command_sequence((second, first))

    no_source = {
        "source_state_decision_monotonic_ns": None,
        "source_state_prediction_monotonic_ns": None,
        "source_frame": None,
        "source_frame_publication_sequence": None,
        "source_frame_publish_monotonic_ns": None,
        "source_tracker_id": None,
        "source_track_role": None,
        "source_state_sequence": None,
        "source_measurement_update_sequence": None,
        "source_candidate_id": None,
    }

    def zero_approval(authority, *, offset, proposal_id, tick_id, command_id):
        proposal = dataclasses.replace(
            _proposal(),
            proposal_id=proposal_id,
            control_tick_id=tick_id,
            proposal_monotonic_ns=_COMMAND_PROPOSAL_NS + offset,
            control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS + offset,
            authority=authority,
            requested_body_rates_rad_s=(0.0, 0.0, 0.0),
            requested_thrust=0.0,
            **no_source,
        )
        return _approved(
            proposal,
            command_id=command_id,
            approval_monotonic_ns=_COMMAND_APPROVAL_NS + offset,
            valid_until_monotonic_ns=_COMMAND_DEADLINE_NS + offset,
        )

    switched_same_epoch = zero_approval(
        dataclasses.replace(
            first.proposal.authority,
            race_status_sequence=first.proposal.authority.race_status_sequence + 1,
            race_status_boot_ms=first.proposal.authority.race_status_boot_ms + 1,
        ),
        offset=20,
        proposal_id=80,
        tick_id=72,
        command_id=81,
    )
    with pytest.raises(ValueError, match="changed without a gate/reset"):
        validate_approved_command_sequence((first, switched_same_epoch))

    gate_1 = zero_approval(
        _authority(gate_index=1, gate_epoch=1),
        offset=20,
        proposal_id=82,
        tick_id=72,
        command_id=83,
    )
    validate_approved_command_sequence((first, gate_1))
    gate_regression = zero_approval(
        _authority(),
        offset=40,
        proposal_id=84,
        tick_id=73,
        command_id=85,
    )
    with pytest.raises(ValueError, match="gate epoch regressed"):
        validate_approved_command_sequence((first, gate_1, gate_regression))

    race_regression = zero_approval(
        dataclasses.replace(
            _authority(gate_index=1, gate_epoch=1),
            race_status_sequence=first.proposal.authority.race_status_sequence,
        ),
        offset=20,
        proposal_id=86,
        tick_id=72,
        command_id=87,
    )
    with pytest.raises(ValueError, match="race-status sequence did not advance"):
        validate_approved_command_sequence((first, race_regression))

    reset_authority = dataclasses.replace(
        _authority(),
        reset_epoch=8,
        race_status_sequence=22,
        race_status_boot_ms=100,
        camera_generation=4,
        frame_publication_sequence_not_before=10,
        frame_publish_monotonic_ns_not_before=2_000_000_013,
    )
    after_reset = zero_approval(
        reset_authority,
        offset=40,
        proposal_id=88,
        tick_id=73,
        command_id=89,
    )
    validate_approved_command_sequence((first, gate_1, after_reset))


def test_approval_deadline_and_latency_events_bind_ids_tick_frame_and_time():
    approved = _approved()
    with pytest.raises(ValueError, match="control tick deadline"):
        dataclasses.replace(
            approved, valid_until_monotonic_ns=_COMMAND_DEADLINE_NS + 1
        )
    due = LatencyEventV1(
        event_sequence=18,
        host_clock_id="host-monotonic-1",
        monotonic_ns=_COMMAND_PROPOSAL_NS - 30,
        kind=LatencyEventKind.CONTROL_TICK_DUE,
        frame=None,
        control_tick_id=71,
        command_id=None,
        sensor_sample_id=None,
        sensor_source_time_ns=None,
        outcome=EventOutcome.OK,
        reason_code=None,
        queue_depth=0,
    )
    tick_start = dataclasses.replace(
        due,
        event_sequence=19,
        monotonic_ns=_COMMAND_PROPOSAL_NS - 20,
        kind=LatencyEventKind.CONTROL_TICK_START,
        frame=_frame(),
    )
    send_start = LatencyEventV1(
        event_sequence=20,
        host_clock_id="host-monotonic-1",
        monotonic_ns=_COMMAND_SEND_NS,
        kind=LatencyEventKind.COMMAND_SEND_START,
        frame=_frame(),
        control_tick_id=71,
        command_id=73,
        sensor_sample_id=None,
        sensor_source_time_ns=None,
        outcome=EventOutcome.OK,
        reason_code=None,
        queue_depth=0,
    )
    send_end = dataclasses.replace(
        send_start,
        event_sequence=21,
        monotonic_ns=_COMMAND_SEND_NS + 1,
        kind=LatencyEventKind.COMMAND_SEND_END,
    )
    validate_command_latency_correlation(
        approved, (due, tick_start, send_start, send_end)
    )
    wrong_frame_start = dataclasses.replace(
        send_start, frame=_frame(frame_id=42)
    )
    wrong_frame_end = dataclasses.replace(
        send_end, frame=_frame(frame_id=42)
    )
    with pytest.raises(ValueError, match="source frame"):
        validate_command_latency_correlation(
            approved, (due, tick_start, wrong_frame_start, wrong_frame_end)
        )
    wrong_tick_start = dataclasses.replace(send_start, control_tick_id=72)
    wrong_tick_end = dataclasses.replace(send_end, control_tick_id=72)
    wrong_due = dataclasses.replace(due, control_tick_id=72)
    wrong_tick_event = dataclasses.replace(tick_start, control_tick_id=72)
    with pytest.raises(ValueError, match="control tick"):
        validate_command_latency_correlation(
            approved, (wrong_due, wrong_tick_event, wrong_tick_start, wrong_tick_end)
        )
    early_start = dataclasses.replace(
        send_start, monotonic_ns=_COMMAND_APPROVAL_NS - 1
    )
    early_end = dataclasses.replace(
        send_end, monotonic_ns=_COMMAND_APPROVAL_NS
    )
    with pytest.raises(ValueError, match="before supervisor approval"):
        validate_command_latency_correlation(
            approved, (due, tick_start, early_start, early_end)
        )
    late_due = dataclasses.replace(
        due,
        monotonic_ns=_COMMAND_PROPOSAL_NS + 1,
    )
    late_tick_start = dataclasses.replace(
        tick_start,
        monotonic_ns=_COMMAND_PROPOSAL_NS + 2,
    )
    with pytest.raises(ValueError, match="became due after its proposal"):
        validate_command_latency_correlation(
            approved,
            (late_due, late_tick_start, send_start, send_end),
        )
    failed_tick_start = dataclasses.replace(
        tick_start,
        outcome=EventOutcome.ERROR,
        reason_code="controller_start_failed",
    )
    with pytest.raises(ValueError, match="failed control-tick events"):
        validate_command_latency_correlation(
            approved,
            (due, failed_tick_start, send_start, send_end),
        )
    failed_send_end = dataclasses.replace(
        send_end,
        outcome=EventOutcome.ERROR,
        reason_code="transport_error",
    )
    with pytest.raises(ValueError, match="failed command-send events"):
        validate_command_latency_correlation(
            approved,
            (due, tick_start, send_start, failed_send_end),
        )
    expired_send_end = dataclasses.replace(
        send_end,
        monotonic_ns=_COMMAND_DEADLINE_NS + 1,
    )
    with pytest.raises(ValueError, match="ended after supervisor approval expired"):
        validate_command_latency_correlation(
            approved,
            (due, tick_start, send_start, expired_send_end),
        )


def test_legacy_command_adapter_requires_exact_dto_and_explicit_diagnostics():
    with pytest.raises(TypeError, match="exact AttitudeRateCommand"):
        legacy_attitude_rate_to_proposal(
            SimpleNamespace(
                roll_rate=0.1, pitch_rate=-0.2, yaw_rate=0.0, thrust=0.27
            ),
            proposal_id=72,
            control_tick_id=71,
            host_clock_id="host-monotonic-1",
            proposal_monotonic_ns=_COMMAND_PROPOSAL_NS,
            control_tick_deadline_monotonic_ns=_COMMAND_DEADLINE_NS,
            source_state_decision_monotonic_ns=2_000_000_020,
            source_state_prediction_monotonic_ns=_COMMAND_SOURCE_STATE_NS,
            source_frame=_frame(),
            source_frame_publication_sequence=8,
            source_frame_publish_monotonic_ns=2_000_000_011,
            source_tracker_id="active-gate-0",
            source_track_role=TrackRole.ACTIVE,
            source_state_sequence=11,
            source_measurement_update_sequence=4,
            source_candidate_id="gate-0-candidate-0",
            authority=_authority(),
            phase="approach",
            saturation=SaturationDiagnosticsV1((False, False, False), False),
            uncertainty=UncertaintyDiagnosticsV1(False, None),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"proposal_monotonic_ns": True},
        {"requested_body_rates_rad_s": (math.nan, 0.0, 0.0)},
        {"requested_thrust": 1.01},
        {"source_state_decision_monotonic_ns": _COMMAND_PROPOSAL_NS + 1},
        {
            "source_state_prediction_monotonic_ns": 2_000_000_019,
        },
        {"control_tick_deadline_monotonic_ns": _COMMAND_PROPOSAL_NS - 1},
    ],
)
def test_command_proposal_rejects_coercion_nonfinite_and_time_regression(changes):
    with pytest.raises((TypeError, ValueError)):
        dataclasses.replace(_proposal(), **changes)


def test_command_phase_is_diagnostic_and_contract_has_no_authority_methods():
    proposal = _proposal()
    forbidden = {"arm", "reset", "send", "cleanup", "advance_gate", "declare_pass"}
    assert forbidden.isdisjoint(dir(proposal))
    assert proposal.phase == "approach"
    assert not proposal.is_exact_zero
    zero = dataclasses.replace(
        proposal,
        requested_body_rates_rad_s=(0.0, 0.0, 0.0),
        requested_thrust=0.0,
    )
    assert zero.is_exact_zero


def test_contracts_are_frozen_and_nested_values_are_hashable():
    proposal = _proposal()
    state = _relative_state()
    observation = _observation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        proposal.phase = "commit"
    assert len({proposal, state, observation}) == 3


def test_uncertainty_and_saturation_diagnostics_are_exact():
    with pytest.raises(ValueError, match="exactly when limited"):
        UncertaintyDiagnosticsV1(limited=False, reason="not_allowed")
    with pytest.raises(TypeError, match="exact bool"):
        SaturationDiagnosticsV1((False, 0, False), False)
    canonical_zero = _proposal(
        AttitudeRateCommand(-0.0, 0.0, -0.0, -0.0)
    )
    assert all(
        math.copysign(1.0, value) == 1.0
        for value in (
            *canonical_zero.requested_body_rates_rad_s,
            canonical_zero.requested_thrust,
        )
    )
    assert "-0.0" not in json.dumps(canonical_zero.to_primitive())
