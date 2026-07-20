"""Adversarial generated/offline tests for the Wave 3 runtime composition."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from competition.adapter import CameraFrame
from competition.vq2_contracts import (
    EventOutcome,
    FeatureCovarianceV1,
    FrameIdentityV1,
    FrameTimingV1,
    GateAuthorityEpochV1,
    LatencyEventKind,
    LatencyEventV1,
    validate_command_proposal_source,
    validate_latency_event_sequence,
    validate_relative_gate_state_source,
)
from competition.vq2_runtime import MINIMUM_CONTROL_PERIOD_NS
from competition.vq2_vision import VQ2VisionSnapshot
from competition.vq2_wave3_offline_runtime import (
    VQ2OfflinePerceptionTiming,
    VQ2OfflineTickInput,
    VQ2OfflineTickTiming,
    VQ2Wave3OfflineConfig,
    VQ2Wave3OfflineRuntime,
)
from estimation.imu_attitude import ImuAttitudeConfig
from estimation.vq2_imu_derotation import (
    SUPPORTED_CAMERA_RAY_MODEL_ID,
    VQ2CameraToBodyCalibration,
    VQ2DerotationModel,
)
from estimation.vq2_imu_provenance import (
    VQ2ImuLineageError,
    VQ2ImuSource,
    VQ2TimedImuSample,
)
from estimation.vq2_relative_estimator import RelativeEstimatorConfig
from planning.vq2_guidance import (
    VQ2GuidancePhase,
    VQ2GuidanceRaceState,
    VQ2SafetyGuidanceInput,
)


G = 9.80665
BASE_NS = 30_000_000_000
HOST = "wave3b-host-monotonic"
CAMERA_STREAM = "camera0"
CAMERA_GENERATION = 4
SESSION = "wave3b-offline-session"
RESET_EPOCH = 2
IMU_STREAM = "highres-imu0"
IMU_GENERATION = 7


def _gate_image(*, center_x: int = 350) -> np.ndarray:
    image = np.full((360, 640, 3), 18, dtype=np.uint8)
    hsv = np.uint8([[[165, 100, 250]]])
    gate_color = tuple(
        int(channel) for channel in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    )
    cv2.rectangle(image, (center_x - 100, 70), (center_x + 100, 270), gate_color, -1)
    cv2.rectangle(image, (center_x - 75, 95), (center_x + 75, 245), (18, 18, 18), -1)
    image.flags.writeable = False
    return image


def _frame_timing(
    tick_ns: int,
    *,
    frame_id: int,
    publication_sequence: int,
    stream_id: str = CAMERA_STREAM,
    generation: int = CAMERA_GENERATION,
    host_clock_id: str = HOST,
) -> FrameTimingV1:
    return FrameTimingV1(
        identity=FrameIdentityV1(stream_id, generation, frame_id),
        camera_source_time_ns=900_000_000_000 + frame_id * 33_333_333,
        host_clock_id=host_clock_id,
        publication_sequence=publication_sequence,
        first_unique_packet_monotonic_ns=tick_ns - 8_000_000,
        final_unique_packet_monotonic_ns=tick_ns - 7_000_000,
        reassembly_complete_monotonic_ns=tick_ns - 6_800_000,
        decode_start_monotonic_ns=tick_ns - 6_500_000,
        decode_end_monotonic_ns=tick_ns - 5_000_000,
        publish_monotonic_ns=tick_ns - 4_000_000,
    )


def _snapshot(
    timing: FrameTimingV1,
    *,
    image: np.ndarray | None = None,
) -> VQ2VisionSnapshot:
    selected_image = _gate_image() if image is None else image
    return VQ2VisionSnapshot(
        frame_id=timing.identity.frame_id,
        camera_frame=CameraFrame(
            timestamp_us=timing.camera_source_time_ns // 1_000,
            image=selected_image,
            width=640,
            height=360,
        ),
        sim_time_ns=timing.camera_source_time_ns,
        received_monotonic_s=timing.final_unique_packet_monotonic_ns / 1e9,
        generation=timing.identity.generation,
        timing=timing,
    )


def _authority(
    sequence: int,
    frame_timing: FrameTimingV1,
    *,
    session_id: str = SESSION,
    reset_epoch: int = RESET_EPOCH,
    camera_host_clock_id: str = HOST,
    camera_stream_id: str = CAMERA_STREAM,
    camera_generation: int = CAMERA_GENERATION,
) -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id=session_id,
        reset_epoch=reset_epoch,
        gate_epoch=0,
        expected_gate_index=0,
        race_status_sequence=100 + sequence,
        race_status_boot_ms=1_000 + sequence * 100,
        camera_host_clock_id=camera_host_clock_id,
        camera_stream_id=camera_stream_id,
        camera_generation=camera_generation,
        frame_publication_sequence_not_before=frame_timing.publication_sequence,
        frame_publish_monotonic_ns_not_before=frame_timing.publish_monotonic_ns,
    )


def _safety(
    sequence: int,
    frame_timing: FrameTimingV1,
    tick_ns: int,
    phase: VQ2GuidancePhase,
    race_state: VQ2GuidanceRaceState,
    *,
    phase_start_ns: int | None = None,
    authority: GateAuthorityEpochV1 | None = None,
) -> VQ2SafetyGuidanceInput:
    evaluation_ns = tick_ns + 6_000_000
    selected_authority = authority or _authority(sequence, frame_timing)
    return VQ2SafetyGuidanceInput(
        authority=selected_authority,
        phase=phase,
        race_state=race_state,
        evaluation_host_clock_id=selected_authority.camera_host_clock_id,
        evaluation_monotonic_ns=evaluation_ns,
        phase_started_monotonic_ns=(
            evaluation_ns if phase_start_ns is None else phase_start_ns
        ),
    )


def _tick_timing(
    tick_ns: int,
    *,
    distinct: bool,
    finish_ns: int | None = None,
    prediction_end_ns: int | None = None,
) -> VQ2OfflineTickTiming:
    perception = None
    if distinct:
        perception = VQ2OfflinePerceptionTiming(
            detection_start_monotonic_ns=tick_ns + 1_000_000,
            detection_end_monotonic_ns=tick_ns + 2_000_000,
            tracking_start_monotonic_ns=tick_ns + 3_000_000,
            tracking_end_monotonic_ns=tick_ns + 4_000_000,
            prediction_start_monotonic_ns=tick_ns + 5_000_000,
            prediction_end_monotonic_ns=(
                tick_ns + 6_000_000
                if prediction_end_ns is None
                else prediction_end_ns
            ),
            estimator_start_monotonic_ns=tick_ns + 7_000_000,
            estimator_end_monotonic_ns=tick_ns + 8_000_000,
        )
    return VQ2OfflineTickTiming(
        tick_start_monotonic_ns=tick_ns,
        controller_start_monotonic_ns=tick_ns + 9_000_000,
        controller_end_monotonic_ns=tick_ns + 10_000_000,
        tick_finish_monotonic_ns=(
            tick_ns + 11_000_000 if finish_ns is None else finish_ns
        ),
        perception=perception,
    )


def _input(
    snapshot: VQ2VisionSnapshot,
    safety: VQ2SafetyGuidanceInput,
    tick_ns: int,
    *,
    distinct: bool,
    finish_ns: int | None = None,
    prediction_end_ns: int | None = None,
) -> VQ2OfflineTickInput:
    return VQ2OfflineTickInput(
        snapshot,
        safety,
        _tick_timing(
            tick_ns,
            distinct=distinct,
            finish_ns=finish_ns,
            prediction_end_ns=prediction_end_ns,
        ),
    )


def _imu_source(**overrides) -> VQ2ImuSource:
    values = {
        "session_id": SESSION,
        "reset_epoch": RESET_EPOCH,
        "host_clock_id": HOST,
        "stream_id": IMU_STREAM,
        "generation": IMU_GENERATION,
    }
    values.update(overrides)
    return VQ2ImuSource(**values)


def _config(**overrides) -> VQ2Wave3OfflineConfig:
    values = {
        "host_clock_id": HOST,
        "camera_stream_id": CAMERA_STREAM,
        "camera_generation": CAMERA_GENERATION,
        "imu_source": _imu_source(),
        "camera_calibration": VQ2CameraToBodyCalibration(
            calibration_id="wave3b-generated-camera-body-v1",
            camera_ray_model_id=SUPPORTED_CAMERA_RAY_MODEL_ID,
            camera_to_body_wxyz=(1.0, 0.0, 0.0, 0.0),
            rotation_uncertainty_rad=0.001,
        ),
        "derotation_model": VQ2DerotationModel(
            model_id="wave3b-generated-rotation-only-v1",
            attitude_time_model_id="wave3b-host-receive-v1",
            max_capture_alignment_ns=20_000_000,
            max_target_extrapolation_ns=20_000_000,
            max_total_timing_uncertainty_ns=50_000_000,
            angular_rate_uncertainty_rad_s=0.01,
        ),
        "capture_orientation_uncertainty_rad": 0.001,
        "target_orientation_uncertainty_rad": 0.001,
        "capture_host_time_uncertainty_ns": 100_000,
        "target_host_time_uncertainty_ns": 100_000,
        "fallback_center_covariance": FeatureCovarianceV1(
            model_id="wave3b-generated-center-v1",
            feature_order=("center_x_norm", "center_y_norm"),
            matrix=((1e-4, 0.0), (0.0, 1e-4)),
        ),
        "scheduler_start_monotonic_ns": BASE_NS,
        "imu_attitude_config": ImuAttitudeConfig(
            calibration_min_samples=3,
            calibration_min_duration_s=0.02,
            max_dt_s=0.02,
        ),
        "relative_estimator_config": RelativeEstimatorConfig(
            minimum_accepted_updates_for_healthy=1,
            initial_bearing_rate_std_norm_s=0.01,
            initial_expansion_rate_std_s=0.01,
        ),
    }
    values.update(overrides)
    return VQ2Wave3OfflineConfig(**values)


def _sample(
    source: VQ2ImuSource,
    sequence: int,
    receive_ns: int,
    *,
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> VQ2TimedImuSample:
    return VQ2TimedImuSample(
        source=source,
        sample_sequence=sequence,
        source_time_us=8_000_000_000_000 + sequence * 10_000,
        receive_monotonic_ns=receive_ns,
        accel_mps2=(0.0, 0.0, -G),
        gyro_rad_s=gyro,
    )


def _ingest_tie_break_attitudes(
    runtime: VQ2Wave3OfflineRuntime,
    *,
    measurement_ns: int,
    prediction_ns: int,
) -> tuple[VQ2TimedImuSample, ...]:
    receive_times = (
        measurement_ns - 30_000_000,
        measurement_ns - 20_000_000,
        measurement_ns - 10_000_000,
        measurement_ns + 10_000_000,
        prediction_ns - 1_000_000,
    )
    samples = tuple(
        _sample(
            runtime.config.imu_source,
            sequence,
            receive_ns,
            gyro=(0.0, 0.0, 0.0) if sequence < 3 else (0.0, 2.0, 0.0),
        )
        for sequence, receive_ns in enumerate(receive_times)
    )
    outputs = tuple(runtime.ingest_imu(sample) for sample in samples)
    assert outputs[:2] == (None, None)
    assert all(attitude is not None for attitude in outputs[2:])
    return samples


def _advance_to_gate0_approach():
    runtime = VQ2Wave3OfflineRuntime(_config())
    first_timing = _frame_timing(BASE_NS, frame_id=10, publication_sequence=1)
    first_snapshot = _snapshot(first_timing)

    initial = _safety(
        0,
        first_timing,
        BASE_NS,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    initial_result = runtime.step(
        _input(first_snapshot, initial, BASE_NS, distinct=True)
    )
    assert initial_result is not None

    go_tick = BASE_NS + MINIMUM_CONTROL_PERIOD_NS
    go_cutover = _frame_timing(
        go_tick,
        frame_id=11,
        publication_sequence=2,
    )
    go_authority = _authority(1, go_cutover)
    go = _safety(
        1,
        first_timing,
        go_tick,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=initial.phase_started_monotonic_ns,
        authority=go_authority,
    )
    go_result = runtime.step(_input(first_snapshot, go, go_tick, distinct=False))
    assert go_result is not None

    align_tick = go_tick + MINIMUM_CONTROL_PERIOD_NS
    align_cutover = _frame_timing(
        align_tick,
        frame_id=12,
        publication_sequence=3,
    )
    align_authority = _authority(2, align_cutover)
    align = _safety(
        2,
        first_timing,
        align_tick,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        authority=align_authority,
    )
    align_result = runtime.step(
        _input(first_snapshot, align, align_tick, distinct=False)
    )
    assert align_result is not None

    approach_tick = align_tick + MINIMUM_CONTROL_PERIOD_NS
    approach_timing = _frame_timing(
        approach_tick,
        frame_id=11,
        publication_sequence=4,
    )
    approach_snapshot = _snapshot(approach_timing)
    approach = _safety(
        3,
        approach_timing,
        approach_tick,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    samples = _ingest_tie_break_attitudes(
        runtime,
        measurement_ns=approach_timing.final_unique_packet_monotonic_ns,
        prediction_ns=approach.evaluation_monotonic_ns,
    )
    result = runtime.step(
        _input(approach_snapshot, approach, approach_tick, distinct=True)
    )
    assert result is not None
    return runtime, result, approach_snapshot, approach, samples


def _assert_runtime_unchanged(runtime: VQ2Wave3OfflineRuntime, before: tuple) -> None:
    assert (
        runtime.next_due_monotonic_ns,
        runtime.next_control_tick_id,
        runtime.next_proposal_id,
        runtime.adapter_memory,
        runtime.attitude_history,
        runtime.processed_frame_timing,
        runtime.last_result,
        runtime.trace,
    ) == before


def _runtime_state(runtime: VQ2Wave3OfflineRuntime) -> tuple:
    return (
        runtime.next_due_monotonic_ns,
        runtime.next_control_tick_id,
        runtime.next_proposal_id,
        runtime.adapter_memory,
        runtime.attitude_history,
        runtime.processed_frame_timing,
        runtime.last_result,
        runtime.trace,
    )


def test_generated_image_imu_and_safety_chain_stops_at_sourced_proposal() -> None:
    runtime, result, _snapshot_value, safety, samples = _advance_to_gate0_approach()

    assert not result.skipped
    assert result.perception_ran
    assert result.selection is not None
    assert result.transition is not None
    assert result.transition.outer_withholding_reason is None
    assert not result.proposal.is_exact_zero
    assert result.proposal.authority == safety.authority

    update = result.transition.active_update
    assert update is not None and update.current_observation_accepted
    evidence = update.evidence
    observation = evidence.observation
    assert update.state.bearing_norm == observation.center_norm
    assert evidence.derotated_center_norm != observation.center_norm
    assert not update.derotation_applied_to_state
    validate_relative_gate_state_source(update.state, observation)
    validate_command_proposal_source(result.proposal, update.state)

    # Capture selection is the later of two equally close receive stamps;
    # target selection is the newest deterministic sample not after prediction.
    assert evidence.capture_attitude.attitude.sample_sequence == 3
    assert evidence.target_attitude.attitude.sample_sequence == 4
    assert result.transition.accepted_attitude is evidence.target_attitude.attitude
    assert evidence.capture_attitude.attitude.source == runtime.config.imu_source
    assert evidence.target_attitude.attitude.source == runtime.config.imu_source
    assert observation.authority == safety.authority
    assert observation.frame == FrameIdentityV1(CAMERA_STREAM, CAMERA_GENERATION, 11)
    assert observation.candidate_id == "wave3b-4-11"

    gyro_events = tuple(
        event for event in result.trace if event.kind is LatencyEventKind.GYRO_SAMPLE
    )
    assert [event.sensor_sample_id for event in gyro_events] == list(range(5))
    assert [event.sensor_source_time_ns for event in gyro_events] == [
        sample.source_time_us * 1_000 for sample in samples
    ]


def test_exported_result_cannot_relabel_sourced_work_as_a_repeated_frame() -> None:
    _runtime, result, _snapshot_value, _safety_value, _samples = (
        _advance_to_gate0_approach()
    )

    with pytest.raises(ValueError, match="all-or-none"):
        replace(result, perception_ran=False)
    with pytest.raises(ValueError, match="repeated frame cannot carry"):
        replace(result, selection=None, perception_ran=False)


def test_exported_result_trace_cannot_detach_current_or_source_facts() -> None:
    _runtime, result, _snapshot_value, _safety_value, _samples = (
        _advance_to_gate0_approach()
    )

    with pytest.raises(ValueError, match="trace must be non-empty"):
        replace(result, trace=())

    without_current_tick = tuple(
        event
        for event in result.trace
        if event.control_tick_id != result.control_tick_id
    )
    validate_latency_event_sequence(without_current_tick)
    with pytest.raises(ValueError, match="current control-tick due"):
        replace(result, trace=without_current_tick)

    source_kinds = {
        LatencyEventKind.CAMERA_FIRST_PACKET,
        LatencyEventKind.CAMERA_FINAL_PACKET,
        LatencyEventKind.FRAME_REASSEMBLED,
        LatencyEventKind.DECODE_START,
        LatencyEventKind.DECODE_END,
        LatencyEventKind.FRAME_PUBLISHED,
        LatencyEventKind.DETECTION_START,
        LatencyEventKind.DETECTION_END,
        LatencyEventKind.TRACKING_START,
        LatencyEventKind.TRACKING_END,
        LatencyEventKind.PREDICTION_START,
        LatencyEventKind.PREDICTION_END,
        LatencyEventKind.ESTIMATOR_UPDATE_START,
        LatencyEventKind.ESTIMATOR_UPDATE_END,
        LatencyEventKind.GYRO_SAMPLE,
    }
    without_source_facts = tuple(
        event for event in result.trace if event.kind not in source_kinds
    )
    validate_latency_event_sequence(without_source_facts)
    with pytest.raises(ValueError, match="camera_first_packet"):
        replace(result, trace=without_source_facts)


def test_exported_result_trace_rejects_send_and_actuator_authority() -> None:
    _runtime, result, _snapshot_value, _safety_value, _samples = (
        _advance_to_gate0_approach()
    )
    assert result.lease is not None
    controller_end = next(
        event
        for event in result.trace
        if event.control_tick_id == result.control_tick_id
        and event.kind is LatencyEventKind.CONTROLLER_END
    )
    injected = (
        LatencyEventV1(
            event_sequence=0,
            host_clock_id=controller_end.host_clock_id,
            monotonic_ns=controller_end.monotonic_ns + 100_000,
            kind=LatencyEventKind.COMMAND_SEND_START,
            frame=result.lease.frame,
            control_tick_id=result.control_tick_id,
            command_id=900,
            sensor_sample_id=None,
            sensor_source_time_ns=None,
            outcome=EventOutcome.OK,
            reason_code=None,
            queue_depth=0,
        ),
        LatencyEventV1(
            event_sequence=0,
            host_clock_id=controller_end.host_clock_id,
            monotonic_ns=controller_end.monotonic_ns + 200_000,
            kind=LatencyEventKind.COMMAND_SEND_END,
            frame=result.lease.frame,
            control_tick_id=result.control_tick_id,
            command_id=900,
            sensor_sample_id=None,
            sensor_source_time_ns=None,
            outcome=EventOutcome.OK,
            reason_code=None,
            queue_depth=0,
        ),
        LatencyEventV1(
            event_sequence=0,
            host_clock_id=controller_end.host_clock_id,
            monotonic_ns=controller_end.monotonic_ns + 300_000,
            kind=LatencyEventKind.ACTUATOR_SAMPLE,
            frame=result.lease.frame,
            control_tick_id=result.control_tick_id,
            command_id=900,
            sensor_sample_id=1,
            sensor_source_time_ns=1_000,
            outcome=EventOutcome.OK,
            reason_code=None,
            queue_depth=0,
        ),
    )
    ordered = sorted(
        (*result.trace, *injected),
        key=lambda event: (event.monotonic_ns, event.event_sequence),
    )
    forged_trace = tuple(
        replace(event, event_sequence=index)
        for index, event in enumerate(ordered)
    )
    validate_latency_event_sequence(forged_trace)

    with pytest.raises(ValueError, match="send/actuator authority"):
        replace(result, trace=forged_trace)


def test_exported_sourced_result_binds_outcomes_diagnostics_and_gyro_facts() -> None:
    _runtime, result, _snapshot_value, _safety_value, _samples = (
        _advance_to_gate0_approach()
    )
    assert result.selection is not None
    assert result.transition is not None
    assert result.transition.active_update is not None
    frame = result.selection.timing.identity

    detection_end = next(
        event
        for event in result.trace
        if event.kind is LatencyEventKind.DETECTION_END and event.frame == frame
    )
    forged_detection = tuple(
        replace(
            event,
            outcome=EventOutcome.ERROR,
            reason_code="forged_detection_error",
        )
        if event is detection_end
        else event
        for event in result.trace
    )
    validate_latency_event_sequence(forged_detection)
    with pytest.raises(ValueError, match="correlated detection_end"):
        replace(result, trace=forged_detection)

    forged_prediction_time = tuple(
        replace(event, monotonic_ns=event.monotonic_ns + 100_000)
        if event.frame == frame
        and event.kind
        in {
            LatencyEventKind.PREDICTION_START,
            LatencyEventKind.PREDICTION_END,
        }
        else event
        for event in result.trace
    )
    validate_latency_event_sequence(forged_prediction_time)
    with pytest.raises(ValueError, match="at-decision target"):
        replace(result, trace=forged_prediction_time)

    with pytest.raises(ValueError, match="reason differs from its transition"):
        replace(result, reason="forged_result_reason")

    active_evidence = result.transition.active_update.evidence
    capture_sequence = active_evidence.capture_attitude.attitude.sample_sequence
    selected_gyro = next(
        event
        for event in result.trace
        if event.kind is LatencyEventKind.GYRO_SAMPLE
        and event.sensor_sample_id == capture_sequence
    )
    forged_gyro = tuple(
        replace(event, frame=frame) if event is selected_gyro else event
        for event in result.trace
    )
    validate_latency_event_sequence(forged_gyro)
    with pytest.raises(ValueError, match="occurrence-only"):
        replace(result, trace=forged_gyro)

    forged_queue = tuple(
        replace(event, queue_depth=99) if event is result.trace[0] else event
        for event in result.trace
    )
    validate_latency_event_sequence(forged_queue)
    with pytest.raises(ValueError, match="zero queue depth"):
        replace(result, trace=forged_queue)

    estimator_end_index = next(
        index
        for index, event in enumerate(result.trace)
        if event.kind is LatencyEventKind.ESTIMATOR_UPDATE_END
        and event.frame == frame
    )
    estimator_end = result.trace[estimator_end_index]
    forged_drop = LatencyEventV1(
        event_sequence=0,
        host_clock_id=estimator_end.host_clock_id,
        monotonic_ns=estimator_end.monotonic_ns,
        kind=LatencyEventKind.FRAME_DROPPED,
        frame=frame,
        control_tick_id=None,
        command_id=None,
        sensor_sample_id=None,
        sensor_source_time_ns=None,
        outcome=EventOutcome.DROPPED,
        reason_code="forged_active_drop",
        queue_depth=0,
    )
    with_drop = list(result.trace)
    with_drop.insert(estimator_end_index + 1, forged_drop)
    forged_drop_trace = tuple(
        replace(event, event_sequence=index)
        for index, event in enumerate(with_drop)
    )
    validate_latency_event_sequence(forged_drop_trace)
    with pytest.raises(ValueError, match="cannot carry a dropped frame"):
        replace(result, trace=forged_drop_trace)


def test_exported_failed_perception_binds_drop_and_stage_lifecycle() -> None:
    runtime = VQ2Wave3OfflineRuntime(_config())
    frame_timing = _frame_timing(BASE_NS, frame_id=13, publication_sequence=1)
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image.flags.writeable = False
    snapshot = _snapshot(frame_timing, image=image)
    safety = _safety(
        0,
        frame_timing,
        BASE_NS,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    result = runtime.step(_input(snapshot, safety, BASE_NS, distinct=True))
    assert result is not None
    assert result.reason == "gate_detection_missing"
    assert result.transition is not None
    assert result.transition.active_update is None

    without_drop = tuple(
        event
        for event in result.trace
        if not (
            event.kind is LatencyEventKind.FRAME_DROPPED
            and event.frame == frame_timing.identity
        )
    )
    validate_latency_event_sequence(without_drop)
    with pytest.raises(ValueError, match="one dropped-frame"):
        replace(result, trace=without_drop)
    with pytest.raises(ValueError, match="differs from perception failure"):
        replace(result, reason="gate_detection_ambiguous")


def test_exported_skipped_result_requires_exact_skip_lifecycle() -> None:
    runtime = VQ2Wave3OfflineRuntime(_config())
    frame_timing = _frame_timing(BASE_NS, frame_id=12, publication_sequence=1)
    snapshot = _snapshot(frame_timing)
    safety = _safety(
        0,
        frame_timing,
        BASE_NS,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    result = runtime.step(
        _input(
            snapshot,
            safety,
            BASE_NS,
            distinct=False,
            finish_ns=BASE_NS + MINIMUM_CONTROL_PERIOD_NS + 1,
        )
    )
    assert result is not None and result.skipped

    without_skip = tuple(
        event
        for event in result.trace
        if event.kind is not LatencyEventKind.CONTROL_TICK_SKIPPED
    )
    validate_latency_event_sequence(without_skip)
    with pytest.raises(ValueError, match="current skipped-tick"):
        replace(result, trace=without_skip)
    with pytest.raises(ValueError, match="unknown offline reason"):
        replace(result, reason="forged_skip_reason")

    skip = next(
        event
        for event in result.trace
        if event.kind is LatencyEventKind.CONTROL_TICK_SKIPPED
        and event.control_tick_id == result.control_tick_id
    )
    invented_skip = tuple(
        replace(event, reason_code="invented_skip") if event is skip else event
        for event in result.trace
    )
    validate_latency_event_sequence(invented_skip)
    with pytest.raises(ValueError, match="unknown offline reason"):
        replace(
            result,
            reason="invented_skip",
            trace=invented_skip,
        )

    due = next(
        event
        for event in result.trace
        if event.kind is LatencyEventKind.CONTROL_TICK_DUE
        and event.control_tick_id == result.control_tick_id
    )
    forged_due = tuple(
        replace(
            event,
            outcome=EventOutcome.ERROR,
            reason_code="forged_due_error",
        )
        if event is due
        else event
        for event in result.trace
    )
    validate_latency_event_sequence(forged_due)
    with pytest.raises(ValueError, match="differs from its trace lifecycle"):
        replace(result, trace=forged_due)


def test_repeated_frame_is_source_less_zero_and_ticks_remain_20ms_apart() -> None:
    runtime, first, snapshot, approach, _samples = _advance_to_gate0_approach()
    repeated_tick = BASE_NS + 4 * MINIMUM_CONTROL_PERIOD_NS
    repeated_safety = VQ2SafetyGuidanceInput(
        authority=approach.authority,
        phase=approach.phase,
        race_state=approach.race_state,
        evaluation_host_clock_id=approach.evaluation_host_clock_id,
        evaluation_monotonic_ns=repeated_tick + 6_000_000,
        phase_started_monotonic_ns=approach.phase_started_monotonic_ns,
    )
    detection_starts_before = sum(
        event.kind is LatencyEventKind.DETECTION_START for event in first.trace
    )

    repeated = runtime.step(
        _input(snapshot, repeated_safety, repeated_tick, distinct=False)
    )

    assert repeated is not None and not repeated.skipped
    assert repeated.selection is None
    assert not repeated.perception_ran
    assert repeated.reason == "imu_correlated_update_missing"
    assert repeated.proposal.is_exact_zero
    assert repeated.proposal.source_frame is None
    assert repeated.transition is not None
    assert repeated.transition.accepted_attitude is None
    assert sum(
        event.kind is LatencyEventKind.DETECTION_START for event in repeated.trace
    ) == detection_starts_before
    due_times = [
        event.monotonic_ns
        for event in repeated.trace
        if event.kind is LatencyEventKind.CONTROL_TICK_DUE
    ]
    assert due_times == [
        BASE_NS + index * MINIMUM_CONTROL_PERIOD_NS for index in range(5)
    ]
    assert all(
        later - earlier == MINIMUM_CONTROL_PERIOD_NS
        for earlier, later in zip(due_times, due_times[1:])
    )


def test_trace_is_chronological_interleaved_and_has_complete_lifecycles() -> None:
    _runtime, result, _snapshot_value, _safety_value, _samples = (
        _advance_to_gate0_approach()
    )
    trace = result.trace
    validate_latency_event_sequence(trace)
    assert [event.event_sequence for event in trace] == list(range(len(trace)))
    assert [event.monotonic_ns for event in trace] == sorted(
        event.monotonic_ns for event in trace
    )

    update = result.transition.active_update
    assert update is not None
    measurement_ns = update.evidence.observation.measurement_time_monotonic_ns
    tied_tracking_ns = measurement_ns + 10_000_000
    tied_prediction_ns = (
        update.evidence.prediction_target.prediction_time_monotonic_ns - 1_000_000
    )
    assert [
        event.kind for event in trace if event.monotonic_ns == tied_tracking_ns
    ][:2] == [LatencyEventKind.GYRO_SAMPLE, LatencyEventKind.TRACKING_START]
    assert [
        event.kind for event in trace if event.monotonic_ns == tied_prediction_ns
    ][:2] == [LatencyEventKind.GYRO_SAMPLE, LatencyEventKind.PREDICTION_START]

    for tick_id in range(4):
        kinds = [
            event.kind for event in trace if event.control_tick_id == tick_id
        ]
        assert kinds.count(LatencyEventKind.CONTROL_TICK_DUE) == 1
        assert kinds.count(LatencyEventKind.CONTROL_TICK_START) == 1
        assert kinds.count(LatencyEventKind.CONTROLLER_START) == 1
        assert kinds.count(LatencyEventKind.CONTROLLER_END) == 1
        assert kinds.count(LatencyEventKind.CONTROL_TICK_END) == 1

    frame = FrameIdentityV1(CAMERA_STREAM, CAMERA_GENERATION, 11)
    frame_kinds = [event.kind for event in trace if event.frame == frame]
    for kind in (
        LatencyEventKind.CAMERA_FIRST_PACKET,
        LatencyEventKind.CAMERA_FINAL_PACKET,
        LatencyEventKind.FRAME_REASSEMBLED,
        LatencyEventKind.DECODE_START,
        LatencyEventKind.DECODE_END,
        LatencyEventKind.FRAME_PUBLISHED,
        LatencyEventKind.DETECTION_START,
        LatencyEventKind.DETECTION_END,
        LatencyEventKind.TRACKING_START,
        LatencyEventKind.TRACKING_END,
        LatencyEventKind.PREDICTION_START,
        LatencyEventKind.PREDICTION_END,
        LatencyEventKind.ESTIMATOR_UPDATE_START,
        LatencyEventKind.ESTIMATOR_UPDATE_END,
    ):
        assert frame_kinds.count(kind) == 1


def test_deadline_and_planned_skips_rebase_without_catch_up() -> None:
    frame_timing = _frame_timing(BASE_NS, frame_id=20, publication_sequence=1)
    snapshot = _snapshot(frame_timing)

    late_runtime = VQ2Wave3OfflineRuntime(_config())
    late_ns = BASE_NS + MINIMUM_CONTROL_PERIOD_NS + 1
    late_safety = _safety(
        0,
        frame_timing,
        late_ns,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    late = late_runtime.step(
        _input(snapshot, late_safety, late_ns, distinct=False)
    )
    assert late is not None and late.skipped
    assert late.reason == "tick_deadline_elapsed"
    assert [event.kind for event in late.trace] == [
        LatencyEventKind.CONTROL_TICK_DUE,
        LatencyEventKind.DEADLINE_MISSED,
        LatencyEventKind.CONTROL_TICK_SKIPPED,
    ]
    assert late_runtime.next_due_monotonic_ns == late_ns + MINIMUM_CONTROL_PERIOD_NS
    assert late_runtime.processed_frame_timing is None
    assert late_runtime.adapter_memory is None
    assert late_runtime.next_proposal_id == 0

    premature_ns = late_runtime.next_due_monotonic_ns - 1
    premature_safety = _safety(
        1,
        frame_timing,
        premature_ns,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    assert late_runtime.step(
        _input(snapshot, premature_safety, premature_ns, distinct=False)
    ) is None
    assert late_runtime.next_control_tick_id == 1

    planned_runtime = VQ2Wave3OfflineRuntime(_config())
    planned_safety = _safety(
        0,
        frame_timing,
        BASE_NS,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    planned = planned_runtime.step(
        _input(
            snapshot,
            planned_safety,
            BASE_NS,
            distinct=False,
            finish_ns=BASE_NS + MINIMUM_CONTROL_PERIOD_NS + 1,
        )
    )
    assert planned is not None and planned.skipped
    assert planned.reason == "planned_work_exceeds_deadline"
    assert [event.kind for event in planned.trace] == [
        LatencyEventKind.CONTROL_TICK_DUE,
        LatencyEventKind.CONTROL_TICK_SKIPPED,
    ]
    assert planned.trace[-1].outcome is EventOutcome.SKIPPED
    assert planned_runtime.next_due_monotonic_ns == BASE_NS + MINIMUM_CONTROL_PERIOD_NS
    assert planned_runtime.processed_frame_timing is None
    assert planned_runtime.adapter_memory is None
    assert planned_runtime.next_proposal_id == 0


def test_distinct_frame_timing_failure_is_transactional_and_retryable() -> None:
    runtime = VQ2Wave3OfflineRuntime(_config())
    frame_timing = _frame_timing(BASE_NS, frame_id=30, publication_sequence=1)
    snapshot = _snapshot(frame_timing)
    safety = _safety(
        0,
        frame_timing,
        BASE_NS,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    before = _runtime_state(runtime)

    with pytest.raises(ValueError, match="safety evaluation must equal prediction end"):
        runtime.step(
            _input(
                snapshot,
                safety,
                BASE_NS,
                distinct=True,
                prediction_end_ns=BASE_NS + 5_000_000,
            )
        )

    _assert_runtime_unchanged(runtime, before)
    recovered = runtime.step(_input(snapshot, safety, BASE_NS, distinct=True))
    assert recovered is not None
    assert recovered.selection is not None
    assert recovered.selection.timing.identity == frame_timing.identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", "other-session"),
        ("reset_epoch", RESET_EPOCH + 1),
        ("host_clock_id", "other-host"),
        ("stream_id", "other-imu"),
        ("generation", IMU_GENERATION + 1),
    ],
)
def test_imu_source_binding_rejects_every_changed_epoch_field_transactionally(
    field: str,
    value: object,
) -> None:
    runtime = VQ2Wave3OfflineRuntime(_config())
    bad_source = replace(runtime.config.imu_source, **{field: value})
    before = _runtime_state(runtime)

    with pytest.raises(VQ2ImuLineageError, match="changed bound IMU source fields"):
        runtime.ingest_imu(_sample(bad_source, 0, BASE_NS - 1_000_000))

    _assert_runtime_unchanged(runtime, before)


def test_imu_source_time_conversion_overflow_is_transactional() -> None:
    runtime = VQ2Wave3OfflineRuntime(_config())
    sample = replace(
        _sample(runtime.config.imu_source, 0, BASE_NS - 1_000_000),
        source_time_us=(2**64 // 1_000) + 1,
    )
    before = _runtime_state(runtime)

    with pytest.raises(ValueError, match="sensor_source_time_ns"):
        runtime.ingest_imu(sample)

    _assert_runtime_unchanged(runtime, before)


@pytest.mark.parametrize(
    "authority_overrides",
    [
        {"session_id": "other-session"},
        {"reset_epoch": RESET_EPOCH + 1},
        {"camera_stream_id": "camera1"},
        {"camera_generation": CAMERA_GENERATION + 1},
        {"camera_host_clock_id": "other-host"},
    ],
)
def test_safety_authority_must_match_exact_camera_and_imu_epoch(
    authority_overrides: dict[str, object],
) -> None:
    runtime = VQ2Wave3OfflineRuntime(_config())
    timing = _frame_timing(BASE_NS, frame_id=40, publication_sequence=1)
    snapshot = _snapshot(timing)
    authority = _authority(0, timing, **authority_overrides)
    safety = _safety(
        0,
        timing,
        BASE_NS,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
        authority=authority,
    )
    before = _runtime_state(runtime)

    with pytest.raises(
        ValueError,
        match="configured (?:host clock|camera identity|IMU epoch)",
    ):
        runtime.step(_input(snapshot, safety, BASE_NS, distinct=True))

    _assert_runtime_unchanged(runtime, before)


@pytest.mark.parametrize(
    ("stream_id", "generation", "message"),
    [
        ("camera1", CAMERA_GENERATION, "configured camera stream"),
        (CAMERA_STREAM, CAMERA_GENERATION + 1, "configured camera generation"),
    ],
)
def test_snapshot_camera_identity_cannot_drift(
    stream_id: str,
    generation: int,
    message: str,
) -> None:
    runtime = VQ2Wave3OfflineRuntime(_config())
    timing = _frame_timing(
        BASE_NS,
        frame_id=41,
        publication_sequence=1,
        stream_id=stream_id,
        generation=generation,
    )
    snapshot = _snapshot(timing)
    good_timing = _frame_timing(BASE_NS, frame_id=41, publication_sequence=1)
    safety = _safety(
        0,
        good_timing,
        BASE_NS,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    before = _runtime_state(runtime)

    with pytest.raises(ValueError, match=message):
        runtime.step(_input(snapshot, safety, BASE_NS, distinct=True))

    _assert_runtime_unchanged(runtime, before)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("snapshot_frame_id_type", "snapshot identity fields"),
        ("snapshot_generation_type", "snapshot identity fields"),
        ("snapshot_sim_time_type", "snapshot identity fields"),
        ("camera_frame_type", "exact CameraFrame"),
        ("camera_timestamp_type", "camera frame metadata"),
        ("camera_timestamp_value", "timestamp differs"),
        ("camera_width_type", "camera frame metadata"),
        ("image_type", "exact numpy.ndarray"),
        ("image_dtype", "uint8 BGR"),
        ("image_layout", "C-contiguous"),
        ("legacy_clock_type", "finite nonnegative float"),
        ("legacy_clock_nonfinite", "finite nonnegative float"),
    ],
)
def test_snapshot_payload_redundancy_is_exact_and_transactional(
    fault: str,
    message: str,
) -> None:
    runtime = VQ2Wave3OfflineRuntime(_config())
    timing = _frame_timing(BASE_NS, frame_id=42, publication_sequence=1)
    snapshot = _snapshot(timing)
    camera_frame = snapshot.camera_frame
    if fault == "snapshot_frame_id_type":
        snapshot = replace(snapshot, frame_id=float(snapshot.frame_id))
    elif fault == "snapshot_generation_type":
        snapshot = replace(snapshot, generation=float(snapshot.generation))
    elif fault == "snapshot_sim_time_type":
        snapshot = replace(snapshot, sim_time_ns=float(snapshot.sim_time_ns))
    elif fault == "camera_frame_type":
        snapshot = replace(snapshot, camera_frame=object())
    elif fault == "camera_timestamp_type":
        snapshot = replace(
            snapshot,
            camera_frame=replace(
                camera_frame,
                timestamp_us=float(camera_frame.timestamp_us),
            ),
        )
    elif fault == "camera_timestamp_value":
        snapshot = replace(
            snapshot,
            camera_frame=replace(
                camera_frame,
                timestamp_us=camera_frame.timestamp_us + 1,
            ),
        )
    elif fault == "camera_width_type":
        snapshot = replace(
            snapshot,
            camera_frame=replace(camera_frame, width=float(camera_frame.width)),
        )
    elif fault == "image_type":
        snapshot = replace(
            snapshot,
            camera_frame=replace(camera_frame, image=object()),
        )
    elif fault == "image_dtype":
        image = np.zeros((360, 640, 3), dtype=np.float32)
        image.flags.writeable = False
        snapshot = replace(
            snapshot,
            camera_frame=replace(camera_frame, image=image),
        )
    elif fault == "image_layout":
        image = np.zeros((360, 640, 6), dtype=np.uint8)[:, :, ::2]
        assert image.shape == (360, 640, 3) and not image.flags.c_contiguous
        image.flags.writeable = False
        snapshot = replace(
            snapshot,
            camera_frame=replace(camera_frame, image=image),
        )
    elif fault == "legacy_clock_type":
        snapshot = replace(
            snapshot,
            received_monotonic_s=int(snapshot.received_monotonic_s),
        )
    elif fault == "legacy_clock_nonfinite":
        snapshot = replace(snapshot, received_monotonic_s=float("nan"))
    else:
        raise AssertionError(f"unhandled fault {fault}")
    safety = _safety(
        0,
        timing,
        BASE_NS,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    before = _runtime_state(runtime)

    with pytest.raises((TypeError, ValueError), match=message):
        runtime.step(_input(snapshot, safety, BASE_NS, distinct=True))

    _assert_runtime_unchanged(runtime, before)
    recovered = runtime.step(
        _input(_snapshot(timing), safety, BASE_NS, distinct=True)
    )
    assert recovered is not None
    assert recovered.selection is not None
    assert recovered.selection.timing == timing


def test_attitude_selection_tie_break_is_deterministic_across_runs() -> None:
    selected = []
    for _ in range(2):
        _runtime, result, _snapshot_value, _safety_value, _samples = (
            _advance_to_gate0_approach()
        )
        update = result.transition.active_update
        assert update is not None
        selected.append(
            (
                update.evidence.capture_attitude.attitude.lineage_key,
                update.evidence.target_attitude.attitude.lineage_key,
                update.evidence.derotated_center_norm,
                update.state,
                result.proposal,
            )
        )

    assert selected[0] == selected[1]
    assert selected[0][0][-3] == 3
    assert selected[0][1][-3] == 4


def test_runtime_static_boundary_and_trace_exclude_authority_surfaces() -> None:
    module_path = Path(__file__).parents[1] / "vq2_wave3_offline_runtime.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    imported_symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
            imported_symbols.update(alias.name for alias in node.names)

    forbidden_modules = (
        "runner",
        "vision_udp",
        "transport",
        "mavlink",
        "pymavlink",
        "socket",
        "supervisor",
        "simulator",
        "flightsim",
        "aigp_client",
    )
    assert not {
        imported
        for imported in imports
        if any(token in imported.lower() for token in forbidden_modules)
    }
    assert not imported_symbols.intersection(
        {
            "SupervisorApprovedCommandV1",
            "CompetitionInterface",
            "AttitudeCommand",
            "AttitudeRateCommand",
            "PositionCommand",
        }
    )

    _runtime, result, _snapshot_value, _safety_value, _samples = (
        _advance_to_gate0_approach()
    )
    forbidden_kinds = {
        LatencyEventKind.ACTUATOR_SAMPLE,
        LatencyEventKind.COMMAND_SEND_START,
        LatencyEventKind.COMMAND_SEND_END,
    }
    assert not {event.kind for event in result.trace}.intersection(forbidden_kinds)
    assert all(event.command_id is None for event in result.trace)
