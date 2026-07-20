"""Adversarial offline tests for the Wave 3 IMU provenance adapter."""

from __future__ import annotations

import ast
import copy
import math
from dataclasses import replace
from pathlib import Path

import pytest

from competition.vq2_contracts import (
    EdgeSetV1,
    FeatureCovarianceV1,
    FitDiagnosticsV1,
    FrameEdge,
    FrameIdentityV1,
    FrameTimingV1,
    GateAuthorityEpochV1,
    GateObservationV1,
    LineSegmentV1,
    MeasurementTimeBasis,
    ObservationHealth,
    PredictionBasis,
    RelativeGateStateV1,
    RelativeStateHealth,
    TrackRole,
    validate_command_proposal_source,
    validate_relative_gate_state_source,
)
from competition.vq2_controller import ControllerTickInput
from competition.vq2_wave3_imu_adapter import (
    CONTROLLER_ATTITUDE_PROPAGATION_MODEL_ID,
    HARD_MAX_CONTROLLER_ATTITUDE_EFFECTIVE_AGE_NS,
    HARD_MAX_CONTROLLER_ATTITUDE_EXTRAPOLATION_NS,
    HARD_MAX_CONTROLLER_ATTITUDE_UNCERTAINTY_RAD,
    VQ2Gate0PitchProvenance,
    VQ2PropagatedAttitudeProvenance,
    VQ2Wave3CoastLease,
    VQ2Wave3ImuAdapterMemory,
    VQ2Wave3ImuAdapterTransition,
    consume_vq2_wave3_coast_lease,
    step_vq2_wave3_imu_adapter,
)
from estimation.imu_attitude import ImuAttitudeConfig
from estimation.vq2_imu_derotation import (
    SUPPORTED_CAMERA_RAY_MODEL_ID,
    VQ2AttitudeDerotationInput,
    VQ2CameraToBodyCalibration,
    VQ2DerotationModel,
    derotate_gate_observation,
)
from estimation.vq2_imu_provenance import (
    VQ2ImuProvenanceEstimator,
    VQ2ImuSource,
    VQ2TimedImuSample,
    VQ2TimestampedAttitude,
)
from estimation.vq2_relative_estimator import (
    RelativeEstimatorConfig,
    RelativeEstimatorUpdate,
    RelativePredictionTarget,
    VQ2ImuCorrelatedEstimatorCoast,
    VQ2ImuCorrelatedEstimatorUpdate,
    VQ2RelativeGateEstimator,
)
from planning.vq2_guidance import (
    VQ2GuidanceObjectiveKind,
    VQ2GuidancePhase,
    VQ2GuidanceRaceState,
    VQ2SafetyGuidanceInput,
)


G = 9.80665
_BASE_NS = 20_000_000_000
_HOST = "wave3-host-monotonic"
_CAMERA_STREAM = "camera0"
_IMU_STREAM = "highres-imu0"
_SESSION = "wave3-offline-session"
_SOURCE_FIELDS = (
    "source_state_decision_monotonic_ns",
    "source_state_prediction_monotonic_ns",
    "source_frame",
    "source_frame_publication_sequence",
    "source_frame_publish_monotonic_ns",
    "source_tracker_id",
    "source_track_role",
    "source_state_sequence",
    "source_measurement_update_sequence",
    "source_candidate_id",
)


def _quaternion_from_euler(
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _authority(
    sequence: int,
    *,
    gate_epoch: int = 0,
    gate_index: int = 0,
    reset_epoch: int = 2,
    session_id: str = _SESSION,
    host_clock_id: str = _HOST,
    camera_generation: int = 4,
) -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id=session_id,
        reset_epoch=reset_epoch,
        gate_epoch=gate_epoch,
        expected_gate_index=gate_index,
        race_status_sequence=100 + sequence,
        race_status_boot_ms=1_000 + sequence * 100,
        camera_host_clock_id=host_clock_id,
        camera_stream_id=_CAMERA_STREAM,
        camera_generation=camera_generation,
        frame_publication_sequence_not_before=1_000 + sequence * 10,
        frame_publish_monotonic_ns_not_before=(
            _BASE_NS + sequence * 100_000_000
        ),
    )


def _safety(
    sequence: int,
    phase: VQ2GuidancePhase,
    race_state: VQ2GuidanceRaceState,
    *,
    phase_start_ns: int | None = None,
    gate_epoch: int = 0,
    gate_index: int = 0,
    reset_epoch: int = 2,
    session_id: str = _SESSION,
    host_clock_id: str = _HOST,
    camera_generation: int = 4,
) -> VQ2SafetyGuidanceInput:
    authority = _authority(
        sequence,
        gate_epoch=gate_epoch,
        gate_index=gate_index,
        reset_epoch=reset_epoch,
        session_id=session_id,
        host_clock_id=host_clock_id,
        camera_generation=camera_generation,
    )
    evaluation_ns = authority.frame_publish_monotonic_ns_not_before + 50_000_000
    return VQ2SafetyGuidanceInput(
        authority=authority,
        phase=phase,
        race_state=race_state,
        evaluation_host_clock_id=host_clock_id,
        evaluation_monotonic_ns=evaluation_ns,
        phase_started_monotonic_ns=(
            evaluation_ns if phase_start_ns is None else phase_start_ns
        ),
    )


def _observation(
    safety: VQ2SafetyGuidanceInput,
    *,
    frame_id: int,
    ordinal: int,
    center: tuple[float, float],
    log_scale: float = math.log(0.45),
) -> GateObservationV1:
    publish_ns = safety.evaluation_monotonic_ns - 10_000_000 + ordinal * 500_000
    measurement_ns = publish_ns - 1_000_000
    authority = safety.authority
    timing = FrameTimingV1(
        identity=FrameIdentityV1(
            _CAMERA_STREAM,
            authority.camera_generation,
            frame_id,
        ),
        camera_source_time_ns=900_000_000_000_000 + frame_id,
        host_clock_id=safety.evaluation_host_clock_id,
        publication_sequence=(
            authority.frame_publication_sequence_not_before + 1 + ordinal
        ),
        first_unique_packet_monotonic_ns=measurement_ns - 1_000,
        final_unique_packet_monotonic_ns=measurement_ns,
        reassembly_complete_monotonic_ns=measurement_ns,
        decode_start_monotonic_ns=measurement_ns + 100_000,
        decode_end_monotonic_ns=measurement_ns + 200_000,
        publish_monotonic_ns=publish_ns,
    )
    half = math.exp(log_scale) * 0.5
    cx, cy = center
    top_left = (cx - half, cy - half)
    top_right = (cx + half, cy - half)
    bottom_right = (cx + half, cy + half)
    bottom_left = (cx - half, cy + half)
    corners = (top_left, top_right, bottom_right, bottom_left)
    assert all(-1.0 < value < 1.0 for corner in corners for value in corner)
    inner_edges = EdgeSetV1(
        left=LineSegmentV1(top_left, bottom_left),
        top=LineSegmentV1(top_left, top_right),
        right=LineSegmentV1(top_right, bottom_right),
        bottom=LineSegmentV1(bottom_left, bottom_right),
    )
    left = (top_left[0] + 1.0) * 0.5
    right = (top_right[0] + 1.0) * 0.5
    top = (top_left[1] + 1.0) * 0.5
    bottom = (bottom_left[1] + 1.0) * 0.5
    feature_order = (
        "center_x_norm",
        "center_y_norm",
        "log_scale",
        "skew_x",
        "skew_y",
    )
    covariance = tuple(
        tuple(1e-4 if row == column else 0.0 for column in range(5))
        for row in range(5)
    )
    return GateObservationV1(
        frame_timing=timing,
        measurement_time_monotonic_ns=measurement_ns,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        measurement_time_model_id=None,
        measurement_uncertainty_ns=1_000_000,
        authority=authority,
        candidate_id=f"gate-candidate-{frame_id}",
        image_size_px=(640, 360),
        center_norm=center,
        support_bounds_norm=(left, top, right, bottom),
        outer_edges=EdgeSetV1(),
        inner_edges=inner_edges,
        inner_corners_norm=corners,
        fitted_inner_aperture_corners_norm=corners,
        geometry_model_id="synthetic-aperture-v1",
        log_scale=log_scale,
        projective_skew=(0.0, 0.0),
        clipping=FrameEdge.NONE,
        confidence=1.0,
        covariance=FeatureCovarianceV1(
            "synthetic-aperture-covariance-v1",
            feature_order,
            covariance,
        ),
        fit=FitDiagnosticsV1(0.001, 4, 4),
        health=ObservationHealth.NOMINAL,
        health_reason=None,
        provenance="generated-wave3-adapter-test",
    )


def _imu_source(
    safety: VQ2SafetyGuidanceInput,
    *,
    stream_id: str = _IMU_STREAM,
    generation: int = 1,
) -> VQ2ImuSource:
    return VQ2ImuSource(
        session_id=safety.authority.session_id,
        reset_epoch=safety.authority.reset_epoch,
        host_clock_id=safety.evaluation_host_clock_id,
        stream_id=stream_id,
        generation=generation,
    )


def _timestamped_attitude(
    source: VQ2ImuSource,
    *,
    sequence: int,
    source_time_us: int,
    receive_monotonic_ns: int,
    pitch_rad: float = -0.2,
    body_rates_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> VQ2TimestampedAttitude:
    return VQ2TimestampedAttitude(
        source=source,
        sample_sequence=sequence,
        source_time_us=source_time_us,
        receive_monotonic_ns=receive_monotonic_ns,
        orientation_body_to_ned_wxyz=_quaternion_from_euler(pitch=pitch_rad),
        body_rates_rad_s=body_rates_rad_s,
        gyro_bias_rad_s=(0.0, 0.0, 0.0),
        accel_trust=1.0,
        propagated=True,
    )


def _attitude_input(
    attitude: VQ2TimestampedAttitude,
    *,
    orientation_uncertainty_rad: float = 0.001,
    host_time_uncertainty_ns: int = 100_000,
) -> VQ2AttitudeDerotationInput:
    return VQ2AttitudeDerotationInput(
        attitude=attitude,
        orientation_uncertainty_rad=orientation_uncertainty_rad,
        host_time_uncertainty_ns=host_time_uncertainty_ns,
    )


def _prediction_target(
    observation: GateObservationV1,
    *,
    decision_monotonic_ns: int | None = None,
    prediction_monotonic_ns: int | None = None,
) -> RelativePredictionTarget:
    decision_ns = (
        observation.frame_timing.publish_monotonic_ns
        if decision_monotonic_ns is None
        else decision_monotonic_ns
    )
    prediction_ns = (
        decision_ns
        if prediction_monotonic_ns is None
        else prediction_monotonic_ns
    )
    if prediction_ns == decision_ns:
        return RelativePredictionTarget.at_decision(
            observation.host_clock_id,
            decision_ns,
        )
    return RelativePredictionTarget(
        host_clock_id=observation.host_clock_id,
        decision_time_monotonic_ns=decision_ns,
        prediction_time_monotonic_ns=prediction_ns,
        prediction_basis=PredictionBasis.COMMAND_EFFECT_ESTIMATE,
        delay_model_id="synthetic-command-effect-v1",
        delay_uncertainty_ns=1_000_000,
    )


def _derotation_evidence(
    observation: GateObservationV1,
    *,
    capture_attitude: VQ2TimestampedAttitude,
    target_attitude: VQ2TimestampedAttitude,
    decision_monotonic_ns: int | None = None,
    prediction_monotonic_ns: int | None = None,
    capture_orientation_uncertainty_rad: float = 0.001,
    target_orientation_uncertainty_rad: float = 0.001,
    capture_host_time_uncertainty_ns: int = 100_000,
    target_host_time_uncertainty_ns: int = 100_000,
    angular_rate_uncertainty_rad_s: float = 0.01,
):
    target = _prediction_target(
        observation,
        decision_monotonic_ns=decision_monotonic_ns,
        prediction_monotonic_ns=prediction_monotonic_ns,
    )
    return derotate_gate_observation(
        observation,
        target,
        capture_attitude=_attitude_input(
            capture_attitude,
            orientation_uncertainty_rad=capture_orientation_uncertainty_rad,
            host_time_uncertainty_ns=capture_host_time_uncertainty_ns,
        ),
        target_attitude=_attitude_input(
            target_attitude,
            orientation_uncertainty_rad=target_orientation_uncertainty_rad,
            host_time_uncertainty_ns=target_host_time_uncertainty_ns,
        ),
        calibration=VQ2CameraToBodyCalibration(
            calibration_id="synthetic-camera-body-v1",
            camera_ray_model_id=SUPPORTED_CAMERA_RAY_MODEL_ID,
            camera_to_body_wxyz=(1.0, 0.0, 0.0, 0.0),
            rotation_uncertainty_rad=0.001,
        ),
        model=VQ2DerotationModel(
            model_id="synthetic-rotation-only-v1",
            attitude_time_model_id="synthetic-host-receive-v1",
            max_capture_alignment_ns=20_000_000,
            max_target_extrapolation_ns=20_000_000,
            max_total_timing_uncertainty_ns=50_000_000,
            angular_rate_uncertainty_rad_s=angular_rate_uncertainty_rad_s,
        ),
    )


class _UpdateStream:
    def __init__(
        self,
        tracker_id: str,
        *,
        frame_id: int,
        imu_sequence: int,
        minimum_accepted_updates_for_healthy: int = 1,
        dropout_variance_per_s: float = 0.25,
    ) -> None:
        self.estimator = VQ2RelativeGateEstimator(
            tracker_id,
            config=RelativeEstimatorConfig(
                minimum_accepted_updates_for_healthy=(
                    minimum_accepted_updates_for_healthy
                ),
                initial_bearing_rate_std_norm_s=0.01,
                initial_expansion_rate_std_s=0.01,
                dropout_variance_per_s=dropout_variance_per_s,
            ),
        )
        self.frame_id = frame_id
        self.imu_sequence = imu_sequence
        self.ordinal = 0

    def make(
        self,
        safety: VQ2SafetyGuidanceInput,
        *,
        center: tuple[float, float] = (0.05, -0.04),
        pitch_rad: float = -0.2,
        attitude_source: VQ2ImuSource | None = None,
        exact_attitude: VQ2TimestampedAttitude | None = None,
        decision_monotonic_ns: int | None = None,
        prediction_monotonic_ns: int | None = None,
        body_rates_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
        target_orientation_uncertainty_rad: float = 0.001,
        target_host_time_uncertainty_ns: int = 100_000,
        angular_rate_uncertainty_rad_s: float = 0.01,
    ) -> VQ2ImuCorrelatedEstimatorUpdate:
        observation = _observation(
            safety,
            frame_id=self.frame_id,
            ordinal=self.ordinal,
            center=center,
        )
        source = attitude_source or _imu_source(safety)
        sequence = self.imu_sequence
        source_time_us = 8_000_000_000_000 + sequence * 10_000
        if exact_attitude is None:
            capture = _timestamped_attitude(
                source,
                sequence=sequence,
                source_time_us=source_time_us,
                receive_monotonic_ns=observation.measurement_time_monotonic_ns,
                pitch_rad=pitch_rad,
                body_rates_rad_s=body_rates_rad_s,
            )
            target_time = (
                (
                    observation.frame_timing.publish_monotonic_ns
                    if decision_monotonic_ns is None
                    else decision_monotonic_ns
                )
                if prediction_monotonic_ns is None
                else prediction_monotonic_ns
            )
            target_attitude = _timestamped_attitude(
                source,
                sequence=sequence + 1,
                source_time_us=source_time_us + 10_000,
                receive_monotonic_ns=target_time,
                pitch_rad=pitch_rad,
                body_rates_rad_s=body_rates_rad_s,
            )
        else:
            capture = exact_attitude
            target_attitude = exact_attitude
        evidence = _derotation_evidence(
            observation,
            capture_attitude=capture,
            target_attitude=target_attitude,
            decision_monotonic_ns=decision_monotonic_ns,
            prediction_monotonic_ns=prediction_monotonic_ns,
            target_orientation_uncertainty_rad=(
                target_orientation_uncertainty_rad
            ),
            target_host_time_uncertainty_ns=(
                target_host_time_uncertainty_ns
            ),
            angular_rate_uncertainty_rad_s=angular_rate_uncertainty_rad_s,
        )
        result = self.estimator.update_with_imu_correlation(evidence)
        self.frame_id += 1
        self.imu_sequence += 2
        self.ordinal += 1
        return result


def _tick(
    safety: VQ2SafetyGuidanceInput,
    update: VQ2ImuCorrelatedEstimatorUpdate | None,
    *,
    proposal_monotonic_ns: int | None = None,
    proposal_id: int = 1,
    host_clock_id: str | None = None,
) -> ControllerTickInput:
    state = None if update is None else update.state
    proposal_ns = (
        safety.evaluation_monotonic_ns + 5_000_000
        if proposal_monotonic_ns is None
        else proposal_monotonic_ns
    )
    return ControllerTickInput(
        proposal_id=proposal_id,
        control_tick_id=proposal_id,
        host_clock_id=(
            safety.evaluation_host_clock_id
            if host_clock_id is None
            else host_clock_id
        ),
        proposal_monotonic_ns=proposal_ns,
        control_tick_deadline_monotonic_ns=proposal_ns + 10_000_000,
        minimum_state_decision_monotonic_ns=(
            0 if state is None else state.timing.decision_time_monotonic_ns
        ),
        minimum_state_sequence=0 if state is None else state.state_sequence,
        expected_phase_started_monotonic_ns=safety.phase_started_monotonic_ns,
        minimum_phase_evaluation_monotonic_ns=safety.evaluation_monotonic_ns,
        expected_authority=safety.authority,
    )


def _step(
    memory: VQ2Wave3ImuAdapterMemory | None,
    safety: VQ2SafetyGuidanceInput,
    *,
    update: VQ2ImuCorrelatedEstimatorUpdate | None = None,
    coast: VQ2ImuCorrelatedEstimatorCoast | None = None,
    tick: ControllerTickInput | None = None,
    enable_correlated_coast: bool = False,
):
    return step_vq2_wave3_imu_adapter(
        memory,
        safety,
        active_update=update,
        correlated_coast=coast,
        tick=_tick(safety, update) if tick is None else tick,
        enable_correlated_coast=enable_correlated_coast,
    )


def _successor_safety(
    source_safety: VQ2SafetyGuidanceInput,
) -> VQ2SafetyGuidanceInput:
    return replace(
        source_safety,
        evaluation_monotonic_ns=(
            source_safety.evaluation_monotonic_ns + 20_000_000
        ),
    )


def _make_coast(
    stream: _UpdateStream,
    prior_update: VQ2ImuCorrelatedEstimatorUpdate,
    safety: VQ2SafetyGuidanceInput,
    *,
    target_attitude: VQ2TimestampedAttitude | None = None,
) -> VQ2ImuCorrelatedEstimatorCoast:
    target = RelativePredictionTarget.at_decision(
        safety.evaluation_host_clock_id,
        safety.evaluation_monotonic_ns,
    )
    state = stream.estimator.coast(target)
    prior_evidence = prior_update.evidence
    prior_attitude_input = prior_evidence.target_attitude
    prior_attitude = prior_attitude_input.attitude
    if target_attitude is None:
        target_attitude = _timestamped_attitude(
            prior_attitude.source,
            sequence=prior_attitude.sample_sequence + 1,
            source_time_us=prior_attitude.source_time_us + 10_000,
            receive_monotonic_ns=safety.evaluation_monotonic_ns,
            pitch_rad=prior_attitude.pitch_rad,
            body_rates_rad_s=prior_attitude.body_rates_rad_s,
        )
    evidence = derotate_gate_observation(
        prior_evidence.observation,
        target,
        capture_attitude=prior_evidence.capture_attitude,
        target_attitude=VQ2AttitudeDerotationInput(
            attitude=target_attitude,
            orientation_uncertainty_rad=(
                prior_attitude_input.orientation_uncertainty_rad
            ),
            host_time_uncertainty_ns=(
                prior_attitude_input.host_time_uncertainty_ns
            ),
        ),
        calibration=prior_evidence.calibration,
        model=prior_evidence.model,
    )
    return VQ2ImuCorrelatedEstimatorCoast(prior_update, state, evidence)


def _coast_tick(
    safety: VQ2SafetyGuidanceInput,
    coast: VQ2ImuCorrelatedEstimatorCoast,
    lease: VQ2Wave3CoastLease,
) -> ControllerTickInput:
    return ControllerTickInput(
        proposal_id=lease.eligible_control_tick_id,
        control_tick_id=lease.eligible_control_tick_id,
        host_clock_id=safety.evaluation_host_clock_id,
        proposal_monotonic_ns=safety.evaluation_monotonic_ns + 5_000_000,
        control_tick_deadline_monotonic_ns=(
            lease.eligible_deadline_monotonic_ns
        ),
        minimum_state_decision_monotonic_ns=(
            coast.state.timing.decision_time_monotonic_ns
        ),
        minimum_state_sequence=coast.state.state_sequence,
        expected_phase_started_monotonic_ns=safety.phase_started_monotonic_ns,
        minimum_phase_evaluation_monotonic_ns=safety.evaluation_monotonic_ns,
        expected_authority=safety.authority,
    )


def _controller_provenance(
    update: VQ2ImuCorrelatedEstimatorUpdate,
    tick: ControllerTickInput,
) -> VQ2PropagatedAttitudeProvenance:
    evidence = update.evidence
    attitude_input = evidence.target_attitude
    attitude = attitude_input.attitude
    extrapolation_ns = tick.proposal_monotonic_ns - attitude.receive_monotonic_ns
    host_uncertainty_s = attitude_input.host_time_uncertainty_ns * 1e-9
    extrapolation_s = extrapolation_ns * 1e-9
    rate_norm = math.sqrt(sum(value * value for value in attitude.body_rates_rad_s))
    angular_uncertainty = float(
        attitude_input.orientation_uncertainty_rad
        + rate_norm * host_uncertainty_s
        + evidence.model.angular_rate_uncertainty_rad_s
        * (host_uncertainty_s + extrapolation_s)
    )
    return VQ2PropagatedAttitudeProvenance(
        evidence=evidence,
        target_host_clock_id=tick.host_clock_id,
        target_monotonic_ns=tick.proposal_monotonic_ns,
        propagation_model_id=CONTROLLER_ATTITUDE_PROPAGATION_MODEL_ID,
        extrapolation_ns=extrapolation_ns,
        effective_age_ns=(
            extrapolation_ns + attitude_input.host_time_uncertainty_ns
        ),
        angular_uncertainty_rad=angular_uncertainty,
        orientation_body_to_ned_wxyz=attitude_input.orientation_at_host_time(
            tick.host_clock_id,
            tick.proposal_monotonic_ns,
        ),
        body_rates_rad_s=attitude.body_rates_rad_s,
    )


def _assert_exact_source_less_zero(transition) -> None:
    assert transition.proposal.is_exact_zero
    assert all(getattr(transition.proposal, field) is None for field in _SOURCE_FIELDS)


def _gate0_context():
    initial = _safety(
        0,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    transition = _step(None, initial)
    go = _safety(
        1,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=initial.phase_started_monotonic_ns,
    )
    transition = _step(transition.memory, go)
    align = _safety(
        2,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = _step(transition.memory, align)
    approach = _safety(
        3,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    return transition.memory, approach


def _enter_gate0(
    *,
    pitch_rad: float = -0.2,
    enable_correlated_coast: bool = False,
    dropout_variance_per_s: float = 0.25,
):
    memory, safety = _gate0_context()
    stream = _UpdateStream(
        "active-gate-0",
        frame_id=300,
        imu_sequence=100,
        dropout_variance_per_s=dropout_variance_per_s,
    )
    update = stream.make(
        safety,
        pitch_rad=pitch_rad,
        decision_monotonic_ns=(
            safety.evaluation_monotonic_ns
            if enable_correlated_coast
            else None
        ),
    )
    transition = _step(
        memory,
        safety,
        update=update,
        enable_correlated_coast=enable_correlated_coast,
    )
    return transition, safety, update, stream


def _enter_gate1(*, enable_correlated_coast: bool = False):
    transition, _, _, _ = _enter_gate0()
    commit = _safety(
        4,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = _step(transition.memory, commit)
    confirmation = _safety(
        5,
        VQ2GuidancePhase.CONFIRMATION,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = _step(transition.memory, confirmation)
    credited = _safety(
        6,
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    transition = _step(transition.memory, credited)
    acquire = _safety(
        7,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    transition = _step(transition.memory, acquire)
    align = _safety(
        8,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    stream = _UpdateStream("active-gate-1", frame_id=400, imu_sequence=200)
    update = stream.make(
        align,
        center=(0.40, -0.30),
        pitch_rad=0.05,
        decision_monotonic_ns=(
            align.evaluation_monotonic_ns
            if enable_correlated_coast
            else None
        ),
    )
    transition = _step(
        transition.memory,
        align,
        update=update,
        enable_correlated_coast=enable_correlated_coast,
    )
    return transition, align, update, stream


def _armed_gate0_coast():
    source, source_safety, update, stream = _enter_gate0(
        enable_correlated_coast=True,
        dropout_variance_per_s=0.05,
    )
    lease = source.memory.coast_lease
    assert lease is not None
    safety = _successor_safety(source_safety)
    coast = _make_coast(stream, update, safety)
    tick = _coast_tick(safety, coast, lease)
    return source, safety, coast, tick


def _armed_gate1_coast():
    source, source_safety, update, stream = _enter_gate1(
        enable_correlated_coast=True
    )
    lease = source.memory.coast_lease
    assert lease is not None
    safety = _successor_safety(source_safety)
    coast = _make_coast(stream, update, safety)
    tick = _coast_tick(safety, coast, lease)
    return source, safety, coast, tick


def _unapplied(
    update: VQ2ImuCorrelatedEstimatorUpdate,
) -> VQ2ImuCorrelatedEstimatorUpdate:
    rejected = RelativeEstimatorUpdate(
        state=update.state,
        measurement_accepted=False,
        observed_candidate_id=update.evidence.observation.candidate_id,
        normalized_innovation_squared=100.0,
        innovation_gate_threshold=9.0,
        reason="innovation_rejected",
    )
    return VQ2ImuCorrelatedEstimatorUpdate(rejected, update.evidence)


def _forge_target_attitude(
    update: VQ2ImuCorrelatedEstimatorUpdate,
    attitude,
) -> VQ2ImuCorrelatedEstimatorUpdate:
    forged = copy.deepcopy(update)
    object.__setattr__(forged.evidence.target_attitude, "attitude", attitude)
    return forged


def test_imu_correlation_preserves_the_ordinary_raw_camera_state_bit_for_bit():
    _, safety = _gate0_context()
    observation = _observation(
        safety,
        frame_id=290,
        ordinal=0,
        center=(0.07, -0.03),
    )
    source = _imu_source(safety)
    capture = _timestamped_attitude(
        source,
        sequence=90,
        source_time_us=7_000_000_000_000,
        receive_monotonic_ns=observation.measurement_time_monotonic_ns,
        pitch_rad=-0.20,
    )
    target = _timestamped_attitude(
        source,
        sequence=91,
        source_time_us=7_000_000_010_000,
        receive_monotonic_ns=observation.frame_timing.publish_monotonic_ns,
        pitch_rad=0.12,
    )
    evidence = _derotation_evidence(
        observation,
        capture_attitude=capture,
        target_attitude=target,
    )
    config = RelativeEstimatorConfig(
        minimum_accepted_updates_for_healthy=1,
        initial_bearing_rate_std_norm_s=0.01,
        initial_expansion_rate_std_s=0.01,
    )
    ordinary = VQ2RelativeGateEstimator(
        "raw-reference",
        config=config,
    ).update(observation, evidence.prediction_target)
    correlated = VQ2RelativeGateEstimator(
        "raw-reference",
        config=config,
    ).update_with_imu_correlation(evidence)

    assert evidence.derotated_center_norm != observation.center_norm
    assert correlated.estimator_update == ordinary
    assert correlated.state == ordinary.state
    assert correlated.state.bearing_norm == observation.center_norm
    assert not correlated.derotation_applied_to_state
    assert correlated.current_observation_accepted


def test_gate0_approach_is_sourced_and_pitch_is_derived_from_exact_attitude():
    transition, safety, update, _ = _enter_gate0(pitch_rad=-0.23)
    attitude = update.evidence.target_attitude.attitude

    assert transition.outer_withholding_reason is None
    assert transition.accepted_attitude is attitude
    assert transition.memory.last_attitude is attitude
    assert transition.controller_attitude_provenance is not None
    assert transition.controller_attitude_provenance.attitude is attitude
    assert (
        transition.controller_attitude_provenance.target_monotonic_ns
        == transition.proposal.proposal_monotonic_ns
    )
    assert not transition.proposal.is_exact_zero
    assert transition.proposal.authority == safety.authority
    assert transition.decision.objective_kind is VQ2GuidanceObjectiveKind.APPROACH_ACTIVE_GATE
    validate_relative_gate_state_source(
        update.state,
        update.evidence.observation,
    )
    validate_command_proposal_source(transition.proposal, update.state)

    inner_latch = transition.memory.inner_memory.gate0_pitch_latch
    provenance = transition.memory.gate0_pitch_provenance
    assert inner_latch is not None
    assert provenance is not None
    assert provenance.attitude is attitude
    assert provenance.initial_pitch_rad == pytest.approx(attitude.pitch_rad)
    assert inner_latch.initial_pitch_rad == pytest.approx(attitude.pitch_rad)
    assert provenance.session_id == attitude.session_id == safety.authority.session_id
    assert provenance.reset_epoch == attitude.reset_epoch == safety.authority.reset_epoch
    assert provenance.host_clock_id == attitude.host_clock_id == _HOST


def test_nonzero_body_rate_is_propagated_separately_to_phase_and_proposal_time():
    memory, safety = _gate0_context()
    update = _UpdateStream(
        "active-gate-0-rate-propagation",
        frame_id=301,
        imu_sequence=110,
    ).make(
        safety,
        pitch_rad=-0.20,
        body_rates_rad_s=(0.0, 1.0, 0.0),
    )

    result = _step(memory, safety, update=update)

    controller = result.controller_attitude_provenance
    pitch = result.memory.gate0_pitch_provenance
    assert result.outer_withholding_reason is None
    assert controller is not None
    assert pitch is not None
    phase = pitch.attitude_provenance
    assert controller.evidence is update.evidence
    assert phase.evidence is update.evidence
    assert controller.target_monotonic_ns == result.proposal.proposal_monotonic_ns
    assert phase.target_monotonic_ns == safety.phase_started_monotonic_ns
    assert controller.target_monotonic_ns > phase.target_monotonic_ns
    assert controller.orientation_body_to_ned_wxyz == (
        update.evidence.target_attitude.orientation_at_host_time(
            _HOST,
            result.proposal.proposal_monotonic_ns,
        )
    )
    assert phase.orientation_body_to_ned_wxyz == (
        update.evidence.target_attitude.orientation_at_host_time(
            _HOST,
            safety.phase_started_monotonic_ns,
        )
    )
    assert (
        controller.orientation_body_to_ned_wxyz
        != phase.orientation_body_to_ned_wxyz
    )
    assert controller.body_rates_rad_s == phase.body_rates_rad_s == (0.0, 1.0, 0.0)
    assert controller.orientation_body_to_ned_wxyz[2] > (
        phase.orientation_body_to_ned_wxyz[2]
    )
    assert result.memory.inner_memory.gate0_pitch_latch is not None
    assert result.memory.inner_memory.gate0_pitch_latch.initial_pitch_rad == (
        pitch.initial_pitch_rad
    )


def test_gate1_align_is_sourced_without_retaining_gate0_pitch_latch():
    transition, safety, update, _ = _enter_gate1()

    assert transition.outer_withholding_reason is None
    assert transition.accepted_attitude is update.evidence.target_attitude.attitude
    assert not transition.proposal.is_exact_zero
    assert transition.proposal.authority == safety.authority
    assert transition.decision.objective_kind is VQ2GuidanceObjectiveKind.RECENTER_ACTIVE_GATE
    assert transition.memory.inner_memory.gate0_pitch_latch is None
    assert transition.memory.gate0_pitch_provenance is None
    validate_command_proposal_source(transition.proposal, update.state)


def test_coast_lease_is_default_off_and_arms_only_for_healthy_sourced_update():
    ordinary, _, _, _ = _enter_gate0(enable_correlated_coast=False)
    enabled, safety, update, _ = _enter_gate0(enable_correlated_coast=True)

    assert ordinary.memory.coast_lease is None
    lease = enabled.memory.coast_lease
    assert lease is not None
    assert lease.source_update is update
    assert lease.source_proposal == enabled.proposal
    assert lease.source_safety == safety
    assert lease.source_control_tick_id == enabled.proposal.control_tick_id
    assert lease.eligible_control_tick_id == lease.source_control_tick_id + 1
    assert lease.eligible_due_monotonic_ns == (
        lease.source_control_tick_deadline_monotonic_ns
    )
    assert lease.eligible_deadline_monotonic_ns == (
        lease.eligible_due_monotonic_ns + 20_000_000
    )
    assert update.current_observation_accepted
    assert update.state.health is RelativeStateHealth.HEALTHY
    assert not enabled.proposal.is_exact_zero

    initializing_memory, initializing_safety = _gate0_context()
    initializing_update = _UpdateStream(
        "initializing-lease-source",
        frame_id=420,
        imu_sequence=240,
        minimum_accepted_updates_for_healthy=2,
    ).make(
        initializing_safety,
        decision_monotonic_ns=initializing_safety.evaluation_monotonic_ns,
    )
    withheld = _step(
        initializing_memory,
        initializing_safety,
        update=initializing_update,
        enable_correlated_coast=True,
    )
    assert initializing_update.state.health is RelativeStateHealth.INITIALIZING
    assert withheld.proposal.is_exact_zero
    assert withheld.memory.coast_lease is None

    unsourced_memory, unsourced_safety = _gate0_context()
    healthy_outside_corridor = _UpdateStream(
        "healthy-unsourced-lease-source",
        frame_id=421,
        imu_sequence=250,
    ).make(
        unsourced_safety,
        center=(0.70, 0.0),
        decision_monotonic_ns=unsourced_safety.evaluation_monotonic_ns,
    )
    unsourced = _step(
        unsourced_memory,
        unsourced_safety,
        update=healthy_outside_corridor,
        enable_correlated_coast=True,
    )
    assert healthy_outside_corridor.state.health is RelativeStateHealth.HEALTHY
    assert unsourced.proposal.is_exact_zero
    assert unsourced.memory.coast_lease is None


def test_coast_lease_binds_the_exact_source_safety_phase_mapping():
    source, _safety, _coast, _tick = _armed_gate0_coast()
    lease = source.memory.coast_lease
    assert lease is not None

    with pytest.raises(ValueError, match="source tick or safety binding differs"):
        replace(
            lease,
            source_proposal=replace(
                lease.source_proposal,
                phase="gate1_recenter",
            ),
        )


@pytest.mark.parametrize("gate", [0, 1])
def test_valid_gate0_and_gate1_coast_is_sourced_once_with_exact_limitation(gate):
    source, safety, coast, tick = (
        _armed_gate0_coast() if gate == 0 else _armed_gate1_coast()
    )

    result = _step(
        source.memory,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason is None
    assert result.active_update is None
    assert result.correlated_coast is coast
    assert result.consumed_coast_lease == source.memory.coast_lease
    assert result.accepted_attitude is coast.evidence.target_attitude.attitude
    assert result.memory.last_attitude is result.accepted_attitude
    assert result.memory.coast_lease is None
    assert not result.proposal.is_exact_zero
    assert result.proposal.uncertainty.limited
    assert result.proposal.uncertainty.reason == (
        "first_observation_dropout_coast"
    )
    validate_command_proposal_source(result.proposal, coast.state)
    assert coast.state.health is RelativeStateHealth.COASTING
    assert coast.state.dropout_count == 1
    assert coast.state.health_reason == "observation_dropout"
    assert coast.state.measurement_update_sequence == (
        coast.prior_update.state.measurement_update_sequence
    )
    assert coast.state.state_sequence == (
        coast.prior_update.state.state_sequence + 1
    )
    if gate == 0:
        assert result.memory.gate0_pitch_provenance == (
            source.memory.gate0_pitch_provenance
        )
        assert result.memory.inner_memory.gate0_pitch_latch == (
            source.memory.inner_memory.gate0_pitch_latch
        )
    else:
        assert result.memory.gate0_pitch_provenance is None
        assert result.memory.inner_memory.gate0_pitch_latch is None


def test_successful_coast_consumes_lease_and_second_attempt_is_exact_zero():
    source, safety, coast, tick = _armed_gate0_coast()
    first = _step(
        source.memory,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )
    later_safety = replace(
        safety,
        evaluation_monotonic_ns=safety.evaluation_monotonic_ns + 20_000_000,
    )
    later_tick = replace(
        tick,
        proposal_id=tick.proposal_id + 1,
        control_tick_id=tick.control_tick_id + 1,
        proposal_monotonic_ns=tick.proposal_monotonic_ns + 20_000_000,
        control_tick_deadline_monotonic_ns=(
            tick.control_tick_deadline_monotonic_ns + 20_000_000
        ),
        minimum_phase_evaluation_monotonic_ns=(
            later_safety.evaluation_monotonic_ns
        ),
    )

    second = _step(
        first.memory,
        later_safety,
        coast=coast,
        tick=later_tick,
        enable_correlated_coast=True,
    )

    assert second.outer_withholding_reason == (
        "imu_correlated_coast_lease_missing"
    )
    _assert_exact_source_less_zero(second)
    assert second.memory.coast_lease is None


def test_skipped_tick_consumption_helper_is_idempotent_and_cannot_rearm():
    source, safety, coast, tick = _armed_gate0_coast()

    consumed = consume_vq2_wave3_coast_lease(source.memory)
    assert consumed is not None
    assert consumed.coast_lease is None
    assert consume_vq2_wave3_coast_lease(consumed) is consumed
    assert consume_vq2_wave3_coast_lease(None) is None

    result = _step(
        consumed,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )
    assert result.outer_withholding_reason == (
        "imu_correlated_coast_lease_missing"
    )
    _assert_exact_source_less_zero(result)


def test_coast_is_exact_zero_when_feature_is_not_explicitly_enabled():
    source, safety, coast, tick = _armed_gate0_coast()

    result = _step(
        source.memory,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=False,
    )

    assert result.outer_withholding_reason == "imu_correlated_coast_disabled"
    _assert_exact_source_less_zero(result)
    assert result.accepted_attitude is None
    assert result.controller_attitude_provenance is None
    assert result.memory.coast_lease is None
    assert result.correlated_coast is coast
    assert result.consumed_coast_lease == source.memory.coast_lease

    with pytest.raises(ValueError, match="exact consumed lease"):
        replace(result, consumed_coast_lease=None)
    with pytest.raises(
        ValueError,
        match="supported source-less transition requires its outer failure reason",
    ):
        replace(
            result,
            outer_withholding_reason=None,
            correlated_coast=None,
            consumed_coast_lease=None,
        )


def test_missing_coast_lease_is_exact_zero_even_when_feature_is_enabled():
    source, safety, coast, tick = _armed_gate0_coast()
    memory = consume_vq2_wave3_coast_lease(source.memory)

    result = _step(
        memory,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason == (
        "imu_correlated_coast_lease_missing"
    )
    _assert_exact_source_less_zero(result)


@pytest.mark.parametrize(
    ("tick_corruption", "expected_reason"),
    (
        ("identity", "imu_correlated_coast_not_immediate_successor"),
        ("deadline", "imu_correlated_coast_tick_period_invalid"),
    ),
)
def test_coast_requires_exact_successor_tick_and_deadline(
    tick_corruption,
    expected_reason,
):
    source, safety, coast, tick = _armed_gate0_coast()
    if tick_corruption == "identity":
        bad_tick = replace(
            tick,
            proposal_id=tick.proposal_id + 1,
            control_tick_id=tick.control_tick_id + 1,
        )
    else:
        bad_tick = replace(
            tick,
            control_tick_deadline_monotonic_ns=(
                tick.control_tick_deadline_monotonic_ns + 1
            ),
        )

    result = _step(
        source.memory,
        safety,
        coast=coast,
        tick=bad_tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason == expected_reason
    _assert_exact_source_less_zero(result)
    assert result.memory.coast_lease is None


def test_coast_cannot_run_before_the_successor_due_window():
    source, source_safety, update, stream = _enter_gate0(
        enable_correlated_coast=True,
        dropout_variance_per_s=0.05,
    )
    lease = source.memory.coast_lease
    assert lease is not None
    early_safety = replace(
        source_safety,
        evaluation_monotonic_ns=(
            source_safety.evaluation_monotonic_ns + 1_000_000
        ),
    )
    coast = _make_coast(stream, update, early_safety)
    early_tick = ControllerTickInput(
        proposal_id=lease.eligible_control_tick_id,
        control_tick_id=lease.eligible_control_tick_id,
        host_clock_id=early_safety.evaluation_host_clock_id,
        proposal_monotonic_ns=(
            early_safety.evaluation_monotonic_ns + 500_000
        ),
        control_tick_deadline_monotonic_ns=(
            lease.eligible_deadline_monotonic_ns
        ),
        minimum_state_decision_monotonic_ns=(
            coast.state.timing.decision_time_monotonic_ns
        ),
        minimum_state_sequence=coast.state.state_sequence,
        expected_phase_started_monotonic_ns=(
            early_safety.phase_started_monotonic_ns
        ),
        minimum_phase_evaluation_monotonic_ns=(
            early_safety.evaluation_monotonic_ns
        ),
        expected_authority=early_safety.authority,
    )
    assert early_tick.proposal_monotonic_ns < lease.eligible_due_monotonic_ns

    result = _step(
        source.memory,
        early_safety,
        coast=coast,
        tick=early_tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason == (
        "imu_correlated_coast_outside_successor_window"
    )
    _assert_exact_source_less_zero(result)
    assert result.memory.coast_lease is None


@pytest.mark.parametrize(
    "safety_corruption",
    ("phase", "race", "authority", "phase_start"),
)
def test_coast_rejects_phase_race_authority_and_phase_start_changes(
    safety_corruption,
):
    source, safety, coast, tick = _armed_gate0_coast()
    if safety_corruption == "phase":
        bad_safety = replace(safety, phase=VQ2GuidancePhase.ALIGN)
    elif safety_corruption == "race":
        bad_safety = replace(safety, race_state=VQ2GuidanceRaceState.FINISHED)
    elif safety_corruption == "authority":
        bad_safety = replace(
            safety,
            authority=replace(
                safety.authority,
                race_status_sequence=safety.authority.race_status_sequence + 1,
            ),
        )
    else:
        bad_safety = replace(
            safety,
            phase_started_monotonic_ns=(
                safety.phase_started_monotonic_ns + 1
            ),
        )

    result = _step(
        source.memory,
        bad_safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason == (
        "imu_correlated_coast_safety_transition"
    )
    _assert_exact_source_less_zero(result)
    assert result.memory.coast_lease is None


def test_coast_from_another_accepted_source_cannot_spend_the_lease():
    source, safety, _, tick = _armed_gate0_coast()
    _, _, unrelated_coast, _ = _armed_gate1_coast()

    result = _step(
        source.memory,
        safety,
        coast=unrelated_coast,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason == (
        "imu_correlated_coast_source_mismatch"
    )
    _assert_exact_source_less_zero(result)


@pytest.mark.parametrize(
    "corruption",
    ("dropout_count", "health_reason", "attitude_sequence", "uncertainty"),
)
def test_malformed_coast_envelope_is_revalidated_and_exact_zero(corruption):
    source, safety, coast, tick = _armed_gate0_coast()
    forged = copy.deepcopy(coast)
    if corruption == "dropout_count":
        object.__setattr__(forged.state, "dropout_count", 2)
    elif corruption == "health_reason":
        object.__setattr__(forged.state, "health_reason", "forged_dropout")
    elif corruption == "attitude_sequence":
        object.__setattr__(
            forged.evidence.target_attitude,
            "attitude",
            forged.prior_update.evidence.target_attitude.attitude,
        )
    else:
        object.__setattr__(
            forged.evidence.target_attitude,
            "orientation_uncertainty_rad",
            (
                forged.evidence.target_attitude.orientation_uncertainty_rad
                + 0.001
            ),
        )

    result = _step(
        source.memory,
        safety,
        coast=forged,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason == "imu_correlated_coast_malformed"
    _assert_exact_source_less_zero(result)
    assert result.memory.coast_lease is None


def test_coast_attitude_uncertainty_has_exact_outer_reason_and_zero_output():
    source, safety, coast, tick = _armed_gate0_coast()
    high_rate_attitude = replace(
        coast.evidence.target_attitude.attitude,
        body_rates_rad_s=(1_000.0, 0.0, 0.0),
    )
    high_rate_evidence = derotate_gate_observation(
        coast.evidence.observation,
        coast.evidence.prediction_target,
        capture_attitude=coast.evidence.capture_attitude,
        target_attitude=VQ2AttitudeDerotationInput(
            attitude=high_rate_attitude,
            orientation_uncertainty_rad=(
                coast.evidence.target_attitude.orientation_uncertainty_rad
            ),
            host_time_uncertainty_ns=(
                coast.evidence.target_attitude.host_time_uncertainty_ns
            ),
        ),
        calibration=coast.evidence.calibration,
        model=coast.evidence.model,
    )
    uncertain = VQ2ImuCorrelatedEstimatorCoast(
        coast.prior_update,
        coast.state,
        high_rate_evidence,
    )

    result = _step(
        source.memory,
        safety,
        coast=uncertain,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason == "attitude_uncertainty_exceeded"
    _assert_exact_source_less_zero(result)
    assert result.accepted_attitude is None
    assert result.controller_attitude_provenance is None


@pytest.mark.parametrize("bad_coast", [object(), "not-a-coast-envelope"])
def test_wrong_coast_type_is_rejected_before_composition(bad_coast):
    source, safety, _, tick = _armed_gate0_coast()

    with pytest.raises(TypeError, match="VQ2ImuCorrelatedEstimatorCoast"):
        step_vq2_wave3_imu_adapter(
            source.memory,
            safety,
            active_update=None,
            correlated_coast=bad_coast,
            tick=tick,
            enable_correlated_coast=True,
        )


def test_active_update_and_coast_are_mutually_exclusive():
    source, safety, coast, tick = _armed_gate0_coast()

    with pytest.raises(ValueError, match="mutually exclusive"):
        step_vq2_wave3_imu_adapter(
            source.memory,
            safety,
            active_update=coast.prior_update,
            correlated_coast=coast,
            tick=tick,
            enable_correlated_coast=True,
        )


@pytest.mark.parametrize("bad_enable", [1, "true", None])
def test_coast_enable_switch_requires_an_exact_bool(bad_enable):
    source, safety, coast, tick = _armed_gate0_coast()

    with pytest.raises(TypeError, match="exact bool"):
        step_vq2_wave3_imu_adapter(
            source.memory,
            safety,
            active_update=None,
            correlated_coast=coast,
            tick=tick,
            enable_correlated_coast=bad_enable,
        )


@pytest.mark.parametrize("bad_memory", [object(), "not-wave3-memory"])
def test_lease_consumption_helper_requires_exact_memory_type(bad_memory):
    with pytest.raises(TypeError, match="VQ2Wave3ImuAdapterMemory"):
        consume_vq2_wave3_coast_lease(bad_memory)


@pytest.mark.parametrize(
    "forgery",
    (
        "eligible_tick",
        "eligible_deadline",
        "source_safety",
        "source_update",
    ),
)
def test_forged_coast_lease_identity_is_quarantined(forgery):
    source, safety, coast, tick = _armed_gate0_coast()
    forged_memory = copy.deepcopy(source.memory)
    lease = forged_memory.coast_lease
    assert lease is not None
    attacker_attitude = None
    if forgery == "eligible_tick":
        object.__setattr__(
            lease,
            "eligible_control_tick_id",
            lease.eligible_control_tick_id + 1,
        )
    elif forgery == "eligible_deadline":
        object.__setattr__(
            lease,
            "eligible_deadline_monotonic_ns",
            lease.eligible_deadline_monotonic_ns + 1,
        )
    elif forgery == "source_safety":
        object.__setattr__(
            lease,
            "source_safety",
            replace(
                lease.source_safety,
                evaluation_monotonic_ns=(
                    lease.source_safety.evaluation_monotonic_ns + 1
                ),
            ),
        )
    else:
        _, _, unrelated_coast, _ = _armed_gate1_coast()
        attacker_attitude = (
            unrelated_coast.prior_update.evidence.target_attitude.attitude
        )
        object.__setattr__(
            lease,
            "source_update",
            unrelated_coast.prior_update,
        )

    result = _step(
        forged_memory,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason == "outer_memory_malformed"
    _assert_exact_source_less_zero(result)
    assert result.memory.coast_lease is None
    if attacker_attitude is not None:
        assert result.memory.last_attitude != attacker_attitude
        assert source.memory.gate0_pitch_provenance is not None
        assert result.memory.last_attitude == (
            source.memory.gate0_pitch_provenance.attitude
        )


def test_forged_lease_source_proposal_cannot_authorize_a_coast():
    source, safety, coast, tick = _armed_gate0_coast()
    forged_memory = copy.deepcopy(source.memory)
    lease = forged_memory.coast_lease
    assert lease is not None
    initial = _safety(
        0,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    unrelated_zero = _step(None, initial).proposal
    object.__setattr__(lease, "source_proposal", unrelated_zero)

    result = _step(
        forged_memory,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason == "outer_memory_malformed"
    _assert_exact_source_less_zero(result)
    assert result.memory.coast_lease is None


def test_forged_retained_attitude_cannot_bypass_coast_chronology():
    source, safety, coast, tick = _armed_gate0_coast()
    forged_memory = copy.deepcopy(source.memory)
    object.__setattr__(
        forged_memory,
        "last_attitude",
        coast.evidence.target_attitude.attitude,
    )

    result = _step(
        forged_memory,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.outer_withholding_reason == "outer_memory_malformed"
    _assert_exact_source_less_zero(result)
    assert result.memory.coast_lease is None


def test_gate0_coast_retains_pitch_and_no_valid_memory_can_ask_it_to_open_one():
    source, safety, coast, tick = _armed_gate0_coast()
    result = _step(
        source.memory,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.memory.gate0_pitch_provenance == (
        source.memory.gate0_pitch_provenance
    )
    assert result.memory.inner_memory.gate0_pitch_latch == (
        source.memory.inner_memory.gate0_pitch_latch
    )
    without_latch = replace(
        source.memory.inner_memory,
        gate0_pitch_latch=None,
    )
    with pytest.raises(ValueError, match="must retain its latch"):
        VQ2Wave3ImuAdapterMemory(
            inner_memory=without_latch,
            last_attitude=source.memory.last_attitude,
            gate0_pitch_provenance=None,
            coast_lease=source.memory.coast_lease,
        )


def test_exported_sourced_coast_transition_requires_its_exact_consumed_lease():
    source, safety, coast, tick = _armed_gate0_coast()
    result = _step(
        source.memory,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )

    assert result.consumed_coast_lease == source.memory.coast_lease
    with pytest.raises(ValueError, match="exact consumed lease"):
        replace(result, consumed_coast_lease=None)

    unrelated_source = _enter_gate1(enable_correlated_coast=True)[0]
    assert unrelated_source.memory.coast_lease is not None
    with pytest.raises(ValueError, match="differs from its consumed lease"):
        replace(
            result,
            consumed_coast_lease=unrelated_source.memory.coast_lease,
        )


def test_transition_integrity_reconstructs_nested_wave2_diagnostics():
    source, safety, coast, tick = _armed_gate0_coast()
    result = _step(
        source.memory,
        safety,
        coast=coast,
        tick=tick,
        enable_correlated_coast=True,
    )
    forged = copy.deepcopy(result)
    object.__setattr__(
        forged.inner_transition.proposal.saturation,
        "body_rate_axes",
        (1, False, False),
    )

    with pytest.raises(TypeError, match=r"body_rate_axes\[0\]"):
        forged.validate_integrity()


def test_outer_memory_integrity_reconstructs_guidance_history():
    source, safety, coast, tick = _armed_gate0_coast()
    forged = copy.deepcopy(source.memory)
    history = forged.inner_memory.guidance_memory.track_histories[0]
    object.__setattr__(history, "seen_measurements", ())

    with pytest.raises(TypeError, match="seen_measurements"):
        forged.validate_integrity()
    with pytest.raises(ValueError, match="unrecoverable outer memory corruption"):
        _step(
            forged,
            safety,
            coast=coast,
            tick=tick,
            enable_correlated_coast=True,
        )


def test_unrecoverable_pitch_proof_corruption_is_classified():
    source, safety, coast, tick = _armed_gate0_coast()
    forged = copy.deepcopy(source.memory)
    pitch = forged.gate0_pitch_provenance
    assert pitch is not None
    object.__setattr__(
        pitch.attitude_provenance.evidence.model,
        "model_id",
        "",
    )

    with pytest.raises(ValueError, match="unrecoverable outer memory corruption"):
        _step(
            forged,
            safety,
            coast=coast,
            tick=tick,
            enable_correlated_coast=True,
        )


@pytest.mark.parametrize("bad_active", [object(), "relative-state-not-envelope"])
def test_wrong_active_update_type_is_rejected_before_composition(bad_active):
    memory, safety = _gate0_context()

    with pytest.raises(TypeError, match="VQ2ImuCorrelatedEstimatorUpdate"):
        step_vq2_wave3_imu_adapter(
            memory,
            safety,
            active_update=bad_active,
            tick=_tick(safety, None),
        )


def test_bare_relative_state_cannot_cross_the_derotation_boundary():
    memory, safety = _gate0_context()
    update = _UpdateStream(
        "active-gate-0",
        frame_id=500,
        imu_sequence=300,
    ).make(safety)

    with pytest.raises(TypeError, match="VQ2ImuCorrelatedEstimatorUpdate"):
        step_vq2_wave3_imu_adapter(
            memory,
            safety,
            active_update=update.state,
            tick=_tick(safety, update),
        )


def test_missing_imu_correlated_update_is_outer_withheld_and_exact_zero():
    memory, safety = _gate0_context()

    result = _step(memory, safety)

    assert result.outer_withholding_reason == "imu_correlated_update_missing"
    assert result.accepted_attitude is None
    assert result.memory.last_attitude is None
    _assert_exact_source_less_zero(result)


def test_rejected_current_observation_is_withheld_without_accepting_attitude():
    memory, safety = _gate0_context()
    update = _UpdateStream(
        "active-gate-0",
        frame_id=510,
        imu_sequence=310,
    ).make(safety)

    result = _step(memory, safety, update=_unapplied(update))

    assert result.outer_withholding_reason == "current_observation_not_accepted"
    assert result.accepted_attitude is None
    assert result.memory.last_attitude is None
    _assert_exact_source_less_zero(result)


def test_shadow_role_cannot_cross_the_active_correlation_boundary():
    memory, safety = _gate0_context()
    ordinary = _UpdateStream(
        "shadow-role-source",
        frame_id=515,
        imu_sequence=315,
    ).make(safety)
    shadow = VQ2RelativeGateEstimator(
        "shadow-role-source",
        track_role=TrackRole.SHADOW,
        config=RelativeEstimatorConfig(
            minimum_accepted_updates_for_healthy=1,
        ),
    ).update_with_imu_correlation(ordinary.evidence)

    result = _step(memory, safety, update=shadow)

    assert result.outer_withholding_reason == "active_role_mismatch"
    assert result.accepted_attitude is None
    assert result.controller_attitude_provenance is None
    assert result.memory.last_attitude is None
    _assert_exact_source_less_zero(result)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capture_attitude", None),
        ("calibration", None),
        ("derotated_center_norm", (0.9, 0.9)),
    ],
)
def test_low_level_evidence_corruption_is_revalidated_and_withheld(field, value):
    memory, safety = _gate0_context()
    update = _UpdateStream(
        "corrupted-evidence-source",
        frame_id=516,
        imu_sequence=317,
    ).make(safety)
    forged = copy.deepcopy(update)
    object.__setattr__(forged.evidence, field, value)

    result = _step(memory, safety, update=forged)

    assert result.outer_withholding_reason == "imu_correlation_evidence_malformed"
    assert result.accepted_attitude is None
    assert result.controller_attitude_provenance is None
    assert result.memory.last_attitude is None
    _assert_exact_source_less_zero(result)


@pytest.mark.parametrize(
    "corruption",
    [
        "unhealthy_target",
        "unsupported_camera_ray_model",
        "empty_derotation_model_id",
        "observation_candidate_relabel",
    ],
)
def test_nested_evidence_invariants_are_revalidated_before_control(corruption):
    memory, safety = _gate0_context()
    update = _UpdateStream(
        "nested-corruption-source",
        frame_id=517,
        imu_sequence=319,
    ).make(safety)
    forged = copy.deepcopy(update)
    if corruption == "unhealthy_target":
        object.__setattr__(
            forged.evidence.target_attitude.attitude,
            "healthy",
            False,
        )
    elif corruption == "unsupported_camera_ray_model":
        object.__setattr__(
            forged.evidence.calibration,
            "camera_ray_model_id",
            "unsupported-ray",
        )
    elif corruption == "observation_candidate_relabel":
        object.__setattr__(
            forged.evidence.observation,
            "candidate_id",
            "forged-candidate",
        )
    else:
        object.__setattr__(forged.evidence.model, "model_id", "")

    result = _step(memory, safety, update=forged)

    assert result.outer_withholding_reason == "imu_correlation_evidence_malformed"
    assert result.accepted_attitude is None
    assert result.controller_attitude_provenance is None
    assert result.memory.last_attitude is None
    _assert_exact_source_less_zero(result)


def test_effective_age_one_nanosecond_over_hard_boundary_is_withheld():
    memory, safety = _gate0_context()
    stream = _UpdateStream("active-gate-0", frame_id=530, imu_sequence=330)
    update = stream.make(
        safety,
        target_host_time_uncertainty_ns=35_000_001,
    )

    result = _step(memory, safety, update=update)

    assert (
        HARD_MAX_CONTROLLER_ATTITUDE_EFFECTIVE_AGE_NS
        == 50_000_000
    )
    assert result.outer_withholding_reason == "attitude_effective_age_exceeded"
    assert result.memory.last_attitude is None
    _assert_exact_source_less_zero(result)


def test_host_time_uncertainty_exact_effective_age_boundary_remains_eligible():
    memory, safety = _gate0_context()
    update = _UpdateStream(
        "active-gate-0",
        frame_id=540,
        imu_sequence=340,
    ).make(
        safety,
        target_host_time_uncertainty_ns=35_000_000,
    )

    result = _step(memory, safety, update=update)

    assert result.outer_withholding_reason is None
    assert result.controller_attitude_provenance is not None
    assert (
        result.controller_attitude_provenance.effective_age_ns
        == HARD_MAX_CONTROLLER_ATTITUDE_EFFECTIVE_AGE_NS
    )
    assert not result.proposal.is_exact_zero


def test_angular_uncertainty_exact_hard_cap_remains_eligible():
    memory, safety = _gate0_context()
    host_uncertainty_ns = 100_000
    extrapolation_ns = 15_000_000
    exact_input_uncertainty = (
        HARD_MAX_CONTROLLER_ATTITUDE_UNCERTAINTY_RAD
        - 0.01
        * ((host_uncertainty_ns + extrapolation_ns) * 1e-9)
    )
    update = _UpdateStream(
        "angular-uncertainty-boundary",
        frame_id=545,
        imu_sequence=345,
    ).make(
        safety,
        target_orientation_uncertainty_rad=exact_input_uncertainty,
        target_host_time_uncertainty_ns=host_uncertainty_ns,
    )

    result = _step(memory, safety, update=update)

    assert result.outer_withholding_reason is None
    assert result.controller_attitude_provenance is not None
    assert result.controller_attitude_provenance.extrapolation_ns == extrapolation_ns
    assert result.controller_attitude_provenance.angular_uncertainty_rad == (
        HARD_MAX_CONTROLLER_ATTITUDE_UNCERTAINTY_RAD
    )
    assert result.memory.gate0_pitch_provenance is not None
    assert not result.proposal.is_exact_zero


def test_angular_uncertainty_one_over_cap_is_zero_without_pitch_provenance():
    memory, safety = _gate0_context()
    exact_input_uncertainty = (
        HARD_MAX_CONTROLLER_ATTITUDE_UNCERTAINTY_RAD
        - 0.01 * ((100_000 + 15_000_000) * 1e-9)
    )
    update = _UpdateStream(
        "angular-uncertainty-over",
        frame_id=546,
        imu_sequence=346,
    ).make(
        safety,
        target_orientation_uncertainty_rad=exact_input_uncertainty + 1e-9,
    )

    result = _step(memory, safety, update=update)

    assert result.outer_withholding_reason == "attitude_uncertainty_exceeded"
    assert result.accepted_attitude is None
    assert result.controller_attitude_provenance is None
    assert result.memory.last_attitude is None
    assert result.memory.gate0_pitch_provenance is None
    assert result.memory.inner_memory.gate0_pitch_latch is not None
    assert result.memory.inner_memory.gate0_pitch_latch.initial_pitch_rad is None
    _assert_exact_source_less_zero(result)


@pytest.mark.parametrize(
    ("excess_ns", "expected_reason"),
    [(0, None), (1, "attitude_extrapolation_exceeded")],
)
def test_controller_attitude_extrapolation_hard_boundary(
    excess_ns,
    expected_reason,
):
    memory, safety = _gate0_context()
    update = _UpdateStream(
        "attitude-extrapolation-boundary",
        frame_id=547,
        imu_sequence=347,
    ).make(safety)
    attitude = update.evidence.target_attitude.attitude
    tick = _tick(
        safety,
        update,
        proposal_monotonic_ns=(
            attitude.receive_monotonic_ns
            + HARD_MAX_CONTROLLER_ATTITUDE_EXTRAPOLATION_NS
            + excess_ns
        ),
    )

    result = _step(memory, safety, update=update, tick=tick)

    assert result.outer_withholding_reason == expected_reason
    if expected_reason is None:
        assert result.controller_attitude_provenance is not None
        assert (
            result.controller_attitude_provenance.extrapolation_ns
            == HARD_MAX_CONTROLLER_ATTITUDE_EXTRAPOLATION_NS
        )
        assert not result.proposal.is_exact_zero
    else:
        assert result.controller_attitude_provenance is None
        assert result.memory.gate0_pitch_provenance is None
        _assert_exact_source_less_zero(result)


def test_attitude_after_proposal_time_is_withheld_as_future():
    memory, safety = _gate0_context()
    update = _UpdateStream(
        "active-gate-0",
        frame_id=550,
        imu_sequence=350,
    ).make(safety)
    tick = _tick(safety, update)
    attitude = update.evidence.target_attitude.attitude
    future_attitude = replace(
        attitude,
        receive_monotonic_ns=tick.proposal_monotonic_ns + 1,
    )
    forged = _forge_target_attitude(update, future_attitude)

    result = _step(memory, safety, update=forged, tick=tick)

    assert result.outer_withholding_reason == "imu_correlation_evidence_malformed"
    assert result.memory.last_attitude is None
    _assert_exact_source_less_zero(result)


def test_cross_authority_derotation_is_withheld_before_control():
    memory, safety = _gate0_context()
    other_safety = _safety(
        3,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
        reset_epoch=3,
        camera_generation=5,
    )
    update = _UpdateStream(
        "active-other-authority",
        frame_id=560,
        imu_sequence=360,
    ).make(other_safety)

    result = _step(memory, safety, update=update)

    assert result.outer_withholding_reason == "imu_correlation_authority_mismatch"
    assert result.memory.last_attitude is None
    _assert_exact_source_less_zero(result)


@pytest.mark.parametrize(
    "source_change",
    [
        {"session_id": "other-session"},
        {"reset_epoch": 9},
        {"host_clock_id": "other-host-clock"},
    ],
)
def test_forged_cross_session_reset_or_clock_attitude_is_withheld(
    source_change,
):
    memory, safety = _gate0_context()
    update = _UpdateStream(
        "active-gate-0",
        frame_id=570,
        imu_sequence=370,
    ).make(safety)
    attitude = update.evidence.target_attitude.attitude
    forged_attitude = replace(
        attitude,
        source=replace(attitude.source, **source_change),
    )
    forged_update = _forge_target_attitude(update, forged_attitude)

    result = _step(memory, safety, update=forged_update)

    assert result.outer_withholding_reason == "imu_correlation_evidence_malformed"
    assert result.memory.last_attitude is None
    _assert_exact_source_less_zero(result)


def test_tick_clock_mismatch_cannot_relabel_an_accepted_attitude():
    memory, safety = _gate0_context()
    update = _UpdateStream(
        "active-gate-0",
        frame_id=580,
        imu_sequence=380,
    ).make(safety)
    tick = _tick(safety, update, host_clock_id="different-tick-clock")

    result = _step(memory, safety, update=update, tick=tick)

    assert result.outer_withholding_reason == "attitude_host_clock_mismatch"
    assert result.memory.last_attitude is None
    _assert_exact_source_less_zero(result)


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("stream", "attitude_stream_relabelled"),
        ("generation", "attitude_generation_regressed"),
        ("sequence", "attitude_sequence_regressed_or_relabelled"),
        ("source_time", "attitude_source_time_regressed_or_relabelled"),
        ("receive_time", "attitude_receive_time_regressed_or_relabelled"),
    ],
)
def test_attitude_replay_or_relabel_is_transactional_and_next_sample_recovers(
    kind,
    reason,
):
    previous, safety, first_update, stream = _enter_gate0()
    first_attitude = first_update.evidence.target_attitude.attitude
    source = first_attitude.source
    if kind == "stream":
        source = replace(source, stream_id="relabelled-imu-stream")
        bad_attitude = None
    elif kind == "generation":
        source = replace(source, generation=0)
        bad_attitude = None
    else:
        sequence = first_attitude.sample_sequence + 1
        source_time_us = first_attitude.source_time_us + 10_000
        receive_ns = first_attitude.receive_monotonic_ns + 100_000
        if kind == "sequence":
            sequence = first_attitude.sample_sequence
        elif kind == "source_time":
            source_time_us = first_attitude.source_time_us
        else:
            receive_ns = first_attitude.receive_monotonic_ns
        bad_attitude = _timestamped_attitude(
            source,
            sequence=sequence,
            source_time_us=source_time_us,
            receive_monotonic_ns=receive_ns,
            pitch_rad=-0.1,
        )
    bad_update = stream.make(
        safety,
        pitch_rad=-0.1,
        attitude_source=source,
        exact_attitude=bad_attitude,
    )

    rejected = _step(previous.memory, safety, update=bad_update)

    assert rejected.outer_withholding_reason == reason
    assert rejected.memory.last_attitude is first_attitude
    assert rejected.memory.gate0_pitch_provenance == (
        previous.memory.gate0_pitch_provenance
    )
    _assert_exact_source_less_zero(rejected)

    recovery_update = stream.make(safety, pitch_rad=0.1)
    recovered = _step(rejected.memory, safety, update=recovery_update)
    assert recovered.outer_withholding_reason is None
    assert recovered.memory.last_attitude is (
        recovery_update.evidence.target_attitude.attitude
    )
    assert recovered.memory.gate0_pitch_provenance == (
        previous.memory.gate0_pitch_provenance
    )
    assert not recovered.proposal.is_exact_zero


def test_exact_same_attitude_may_be_reused_for_a_new_camera_update():
    previous, safety, first_update, stream = _enter_gate0()
    attitude = first_update.evidence.target_attitude.attitude
    update = stream.make(safety, exact_attitude=attitude)

    result = _step(previous.memory, safety, update=update)

    assert result.outer_withholding_reason is None
    assert result.accepted_attitude is attitude
    assert result.memory.last_attitude is attitude
    assert not result.proposal.is_exact_zero
    validate_command_proposal_source(result.proposal, update.state)


def test_advancing_generation_can_restart_imu_sequence_without_relabel():
    previous, safety, first_update, stream = _enter_gate0()
    first = first_update.evidence.target_attitude.attitude
    source = replace(first.source, generation=first.generation + 1)
    restarted = _timestamped_attitude(
        source,
        sequence=0,
        source_time_us=0,
        receive_monotonic_ns=first.receive_monotonic_ns + 100_000,
        pitch_rad=-0.1,
    )
    update = stream.make(safety, exact_attitude=restarted)

    result = _step(previous.memory, safety, update=update)

    assert result.outer_withholding_reason is None
    assert result.accepted_attitude is restarted
    assert not result.proposal.is_exact_zero


def test_higher_generation_still_requires_strictly_advancing_receive_time():
    previous, safety, first_update, stream = _enter_gate0()
    first = first_update.evidence.target_attitude.attitude
    higher_generation = _timestamped_attitude(
        replace(first.source, generation=first.generation + 1),
        sequence=0,
        source_time_us=0,
        receive_monotonic_ns=first.receive_monotonic_ns,
        pitch_rad=-0.1,
    )
    update = stream.make(safety, exact_attitude=higher_generation)

    result = _step(previous.memory, safety, update=update)

    assert (
        result.outer_withholding_reason
        == "attitude_receive_time_regressed_or_relabelled"
    )
    assert result.memory.last_attitude is first
    assert result.memory.gate0_pitch_provenance is (
        previous.memory.gate0_pitch_provenance
    )
    _assert_exact_source_less_zero(result)


def test_invalid_stream_b_is_quarantined_before_fresh_stream_a_recovers():
    accepted_a, safety, first_update, stream = _enter_gate0()
    first_a = first_update.evidence.target_attitude.attitude
    bad_b = _timestamped_attitude(
        replace(first_a.source, stream_id="imu-stream-b"),
        sequence=first_a.sample_sequence + 1,
        source_time_us=first_a.source_time_us + 10_000,
        receive_monotonic_ns=first_a.receive_monotonic_ns + 100_000,
        pitch_rad=-0.1,
    )
    update_b = stream.make(safety, exact_attitude=bad_b)

    rejected_b = _step(accepted_a.memory, safety, update=update_b)

    assert rejected_b.outer_withholding_reason == "attitude_stream_relabelled"
    assert rejected_b.memory.last_attitude is first_a
    assert rejected_b.memory.gate0_pitch_provenance is (
        accepted_a.memory.gate0_pitch_provenance
    )
    _assert_exact_source_less_zero(rejected_b)

    fresh_a = _timestamped_attitude(
        first_a.source,
        sequence=first_a.sample_sequence + 2,
        source_time_us=first_a.source_time_us + 20_000,
        receive_monotonic_ns=first_a.receive_monotonic_ns + 200_000,
        pitch_rad=0.1,
    )
    recovery_update = stream.make(safety, exact_attitude=fresh_a)
    recovered_a = _step(
        rejected_b.memory,
        safety,
        update=recovery_update,
    )

    assert recovery_update.current_observation_accepted
    assert recovered_a.outer_withholding_reason is None
    assert recovered_a.accepted_attitude is fresh_a
    assert recovered_a.memory.last_attitude is fresh_a
    assert recovered_a.memory.gate0_pitch_provenance is (
        accepted_a.memory.gate0_pitch_provenance
    )
    assert not recovered_a.proposal.is_exact_zero


def test_gate0_pitch_latches_once_while_latest_attitude_continues_to_advance():
    first, safety, first_update, stream = _enter_gate0(pitch_rad=-0.24)
    first_provenance = first.memory.gate0_pitch_provenance
    update = stream.make(safety, pitch_rad=0.17)

    second = _step(first.memory, safety, update=update)

    assert second.outer_withholding_reason is None
    assert second.memory.last_attitude is update.evidence.target_attitude.attitude
    assert second.memory.gate0_pitch_provenance is first_provenance
    assert second.memory.inner_memory.gate0_pitch_latch is not None
    assert second.memory.inner_memory.gate0_pitch_latch.initial_pitch_rad == (
        first_update.evidence.target_attitude.attitude.pitch_rad
    )
    assert not second.proposal.is_exact_zero


def test_missing_pitch_at_gate0_entry_cannot_be_filled_late():
    memory, safety = _gate0_context()
    entered = _step(memory, safety)
    assert entered.memory.inner_memory.gate0_pitch_latch is not None
    assert entered.memory.inner_memory.gate0_pitch_latch.initial_pitch_rad is None
    assert entered.memory.gate0_pitch_provenance is None

    later_safety = _safety(
        4,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=safety.phase_started_monotonic_ns,
    )
    stream = _UpdateStream("active-gate-0", frame_id=600, imu_sequence=400)
    update = stream.make(later_safety, pitch_rad=-0.2)
    late_attitude = update.evidence.target_attitude.attitude
    assert late_attitude.receive_monotonic_ns > safety.phase_started_monotonic_ns
    rejected = _step(entered.memory, later_safety, update=update)

    assert rejected.outer_withholding_reason is None
    assert rejected.memory.last_attitude is update.evidence.target_attitude.attitude
    assert rejected.memory.inner_memory.gate0_pitch_latch is not None
    assert rejected.memory.inner_memory.gate0_pitch_latch.initial_pitch_rad is None
    assert rejected.memory.gate0_pitch_provenance is None
    assert "pitch_basis" in rejected.proposal.reason
    _assert_exact_source_less_zero(rejected)


def test_phase_exit_gate_credit_and_gate1_clear_gate0_pitch_provenance():
    gate0, _, _, _ = _enter_gate0()
    assert gate0.memory.gate0_pitch_provenance is not None

    commit = _safety(
        4,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    exited = _step(gate0.memory, commit)
    assert exited.memory.inner_memory.gate0_pitch_latch is None
    assert exited.memory.gate0_pitch_provenance is None

    gate1, _, _, _ = _enter_gate1()
    assert gate1.memory.inner_memory.gate0_pitch_latch is None
    assert gate1.memory.gate0_pitch_provenance is None


def test_reset_clears_attitude_and_pitch_and_requires_a_new_bootstrap_basis():
    previous, _, _, _ = _enter_gate0()
    old = previous.memory.inner_memory.guidance_memory.safety
    reset_authority = replace(
        old.authority,
        reset_epoch=old.authority.reset_epoch + 1,
        gate_epoch=0,
        expected_gate_index=0,
        race_status_sequence=old.authority.race_status_sequence + 1,
        race_status_boot_ms=old.authority.race_status_boot_ms + 100,
        camera_generation=old.authority.camera_generation + 1,
        frame_publication_sequence_not_before=(
            old.authority.frame_publication_sequence_not_before + 1
        ),
        frame_publish_monotonic_ns_not_before=(
            old.authority.frame_publish_monotonic_ns_not_before + 100_000_000
        ),
    )
    evaluation_ns = reset_authority.frame_publish_monotonic_ns_not_before + 50_000_000
    reset_safety = VQ2SafetyGuidanceInput(
        authority=reset_authority,
        phase=VQ2GuidancePhase.ACQUIRE,
        race_state=VQ2GuidanceRaceState.NOT_UNDERWAY,
        evaluation_host_clock_id=_HOST,
        evaluation_monotonic_ns=evaluation_ns,
        phase_started_monotonic_ns=evaluation_ns,
    )

    reset = _step(previous.memory, reset_safety)

    assert reset.memory.last_attitude is None
    assert reset.memory.inner_memory.gate0_pitch_latch is None
    assert reset.memory.gate0_pitch_provenance is None


def test_invalid_supplied_update_is_quarantined_during_unsupported_commit():
    previous, _, _, stream = _enter_gate0()
    prior_guidance = previous.memory.inner_memory.guidance_memory
    commit = _safety(
        4,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    rejected_update = _unapplied(stream.make(commit))

    result = _step(previous.memory, commit, update=rejected_update)

    assert result.active_update is rejected_update
    assert result.outer_withholding_reason == "current_observation_not_accepted"
    assert result.accepted_attitude is None
    assert result.controller_attitude_provenance is None
    assert result.memory.last_attitude is previous.memory.last_attitude
    assert (
        result.memory.inner_memory.guidance_memory.track_histories
        == prior_guidance.track_histories
    )
    assert result.memory.inner_memory.guidance_memory.active_source == (
        prior_guidance.active_source
    )
    assert all(
        history.latest_source.source_candidate_id
        != rejected_update.evidence.observation.candidate_id
        for history in result.memory.inner_memory.guidance_memory.track_histories
    )
    _assert_exact_source_less_zero(result)


def test_rejected_requested_safety_preserves_outer_lineage_without_raising():
    previous, safety, _, stream = _enter_gate0()
    requested = replace(
        safety,
        evaluation_monotonic_ns=safety.evaluation_monotonic_ns + 1_000_000,
        phase_started_monotonic_ns=(
            safety.phase_started_monotonic_ns + 1_000_000
        ),
    )
    assert requested.phase_started_monotonic_ns != safety.phase_started_monotonic_ns
    update = stream.make(requested)
    tick = _tick(requested, update)

    result = _step(previous.memory, requested, update=update, tick=tick)

    assert result.active_update is update
    assert result.outer_withholding_reason == "guidance_safety_not_accepted"
    assert result.accepted_attitude is None
    assert result.controller_attitude_provenance is None
    assert result.memory == previous.memory
    assert result.memory.last_attitude is previous.memory.last_attitude
    assert result.memory.gate0_pitch_provenance is (
        previous.memory.gate0_pitch_provenance
    )
    assert result.memory.inner_memory.gate0_pitch_latch == (
        previous.memory.inner_memory.gate0_pitch_latch
    )
    _assert_exact_source_less_zero(result)

    forged_memory = replace(
        result.memory,
        last_attitude=update.evidence.target_attitude.attitude,
    )
    with pytest.raises(ValueError, match="guidance source"):
        VQ2Wave3ImuAdapterTransition(
            memory=forged_memory,
            inner_transition=result.inner_transition,
            active_update=update,
            accepted_attitude=update.evidence.target_attitude.attitude,
            controller_attitude_provenance=_controller_provenance(update, tick),
            outer_withholding_reason=None,
        )


@pytest.mark.parametrize(
    ("gate", "expected_outer_reason"),
    [(0, None), (1, None)],
)
def test_commit_is_always_source_less_zero_even_with_valid_correlation(
    gate,
    expected_outer_reason,
):
    if gate == 0:
        previous, _, _, stream = _enter_gate0()
        commit = _safety(
            4,
            VQ2GuidancePhase.COMMIT,
            VQ2GuidanceRaceState.UNDERWAY,
        )
        update = stream.make(commit)
    else:
        previous, _, _, stream = _enter_gate1()
        approach = _safety(
            9,
            VQ2GuidancePhase.APPROACH,
            VQ2GuidanceRaceState.UNDERWAY,
            gate_epoch=1,
            gate_index=1,
        )
        previous = _step(previous.memory, approach)
        commit = _safety(
            10,
            VQ2GuidancePhase.COMMIT,
            VQ2GuidanceRaceState.UNDERWAY,
            gate_epoch=1,
            gate_index=1,
        )
        update = stream.make(commit, center=(0.40, -0.30), pitch_rad=0.05)

    result = _step(previous.memory, commit, update=update)

    assert result.outer_withholding_reason == expected_outer_reason
    assert update.current_observation_accepted
    assert result.accepted_attitude is None
    assert result.controller_attitude_provenance is None
    _assert_exact_source_less_zero(result)


def test_gate2_align_is_always_source_less_zero_after_legal_credit_lifecycle():
    transition, _, _, _ = _enter_gate1()
    approach = _safety(
        9,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    transition = _step(transition.memory, approach)
    commit = _safety(
        10,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    transition = _step(transition.memory, commit)
    confirmation = _safety(
        11,
        VQ2GuidancePhase.CONFIRMATION,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    transition = _step(transition.memory, confirmation)
    credited = _safety(
        12,
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=2,
        gate_index=2,
    )
    transition = _step(transition.memory, credited)
    acquire = _safety(
        13,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=2,
        gate_index=2,
    )
    transition = _step(transition.memory, acquire)
    align = _safety(
        14,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=2,
        gate_index=2,
    )
    update = _UpdateStream(
        "active-gate-2",
        frame_id=700,
        imu_sequence=500,
    ).make(align, center=(0.4, -0.3), pitch_rad=0.0)

    result = _step(transition.memory, align, update=update)

    assert result.outer_withholding_reason is None
    assert result.memory.inner_memory.guidance_memory.safety == align
    assert result.accepted_attitude is None
    assert result.controller_attitude_provenance is None
    _assert_exact_source_less_zero(result)


def test_forged_pitch_memory_and_inner_transition_mismatch_are_rejected():
    transition, _, _, _ = _enter_gate0()

    with pytest.raises(ValueError, match="requires outer provenance"):
        replace(transition.memory, gate0_pitch_provenance=None)
    assert transition.memory.gate0_pitch_provenance is not None
    with pytest.raises(ValueError, match="derived from phase-entry attitude"):
        replace(
            transition.memory.gate0_pitch_provenance,
            initial_pitch_rad=(
                transition.memory.gate0_pitch_provenance.initial_pitch_rad + 0.01
            ),
        )

    forged_inner_memory = replace(
        transition.inner_transition.memory,
        gate0_pitch_latch=None,
    )
    with pytest.raises(ValueError, match="must retain its latch"):
        VQ2Wave3ImuAdapterMemory(
            inner_memory=forged_inner_memory,
            last_attitude=transition.memory.last_attitude,
            gate0_pitch_provenance=None,
        )

    with pytest.raises(ValueError, match="requires its pitch latch"):
        replace(
            transition.inner_transition,
            memory=forged_inner_memory,
        )


def test_forged_memory_cannot_retain_attitude_from_another_epoch():
    transition, _, _, _ = _enter_gate0()
    attitude = transition.memory.last_attitude
    assert attitude is not None
    forged = replace(
        attitude,
        source=replace(attitude.source, reset_epoch=attitude.reset_epoch + 1),
    )

    with pytest.raises(ValueError, match="attitude"):
        replace(transition.memory, last_attitude=forged)


def test_exported_transition_rejects_unrelated_correlated_update_source():
    transition, safety, _, stream = _enter_gate0()
    unrelated = stream.make(safety, pitch_rad=0.1)
    unrelated_transition = _step(
        transition.memory,
        safety,
        update=unrelated,
        tick=replace(
            _tick(safety, unrelated),
            proposal_id=transition.proposal.proposal_id,
            control_tick_id=transition.proposal.control_tick_id,
            proposal_monotonic_ns=transition.proposal.proposal_monotonic_ns,
            control_tick_deadline_monotonic_ns=(
                transition.proposal.proposal_monotonic_ns + 10_000_000
            ),
        ),
    )
    unrelated_attitude = unrelated.evidence.target_attitude.attitude
    forged_memory = replace(
        transition.memory,
        last_attitude=unrelated_attitude,
    )
    assert unrelated_transition.controller_attitude_provenance is not None

    with pytest.raises(ValueError, match="source"):
        VQ2Wave3ImuAdapterTransition(
            memory=forged_memory,
            inner_transition=transition.inner_transition,
            active_update=unrelated,
            accepted_attitude=unrelated_attitude,
            controller_attitude_provenance=(
                unrelated_transition.controller_attitude_provenance
            ),
            outer_withholding_reason=None,
        )


def test_exported_transition_rejects_changed_accepted_attitude():
    transition, _, update, _ = _enter_gate0()
    attitude = update.evidence.target_attitude.attitude
    changed = replace(
        attitude,
        orientation_body_to_ned_wxyz=_quaternion_from_euler(pitch=0.1),
    )

    with pytest.raises(ValueError, match="does not match IMU correlation"):
        replace(transition, accepted_attitude=changed)


def test_exported_transition_requires_attitude_and_propagation_all_or_none():
    transition, _, _, _ = _enter_gate0()
    assert transition.accepted_attitude is not None
    assert transition.controller_attitude_provenance is not None

    with pytest.raises(ValueError, match="all-or-none"):
        replace(transition, accepted_attitude=None)
    with pytest.raises(ValueError, match="all-or-none"):
        replace(transition, controller_attitude_provenance=None)


def test_full_raw_provenance_derotation_estimator_adapter_chain_is_exact():
    memory, safety = _gate0_context()
    observation = _observation(
        safety,
        frame_id=800,
        ordinal=0,
        center=(0.05, -0.04),
    )
    source = _imu_source(safety)
    provenance = VQ2ImuProvenanceEstimator(
        source,
        config=ImuAttitudeConfig(
            calibration_min_samples=3,
            calibration_min_duration_s=0.02,
            max_dt_s=0.02,
        ),
    )
    pitch = -0.19
    accel = (
        G * math.sin(pitch),
        0.0,
        -G * math.cos(pitch),
    )
    capture = None
    for offset in range(3):
        capture = provenance.update(
            VQ2TimedImuSample(
                source=source,
                sample_sequence=offset,
                source_time_us=9_000_000_000_000 + offset * 10_000,
                receive_monotonic_ns=(
                    observation.measurement_time_monotonic_ns
                    - (2 - offset) * 10_000_000
                ),
                accel_mps2=accel,
                gyro_rad_s=(0.0, 0.0, 0.0),
            )
        )
    assert capture is not None and capture.calibrated
    target = provenance.update(
        VQ2TimedImuSample(
            source=source,
            sample_sequence=3,
            source_time_us=9_000_000_030_000,
            receive_monotonic_ns=observation.frame_timing.publish_monotonic_ns,
            accel_mps2=accel,
            gyro_rad_s=(0.0, 0.0, 0.0),
        )
    )
    assert target is not None and target.propagated
    evidence = _derotation_evidence(
        observation,
        capture_attitude=capture,
        target_attitude=target,
    )
    update = VQ2RelativeGateEstimator(
        "full-chain-active-gate-0",
        config=RelativeEstimatorConfig(
            minimum_accepted_updates_for_healthy=1,
            initial_bearing_rate_std_norm_s=0.01,
            initial_expansion_rate_std_s=0.01,
        ),
    ).update_with_imu_correlation(evidence)

    result = _step(memory, safety, update=update)

    assert result.outer_withholding_reason is None
    assert result.accepted_attitude is target
    assert result.active_update is update
    assert update.evidence is evidence
    assert evidence.target_attitude.attitude is target
    assert update.state.source_candidate_id == observation.candidate_id
    assert result.memory.gate0_pitch_provenance is not None
    assert result.memory.gate0_pitch_provenance.attitude is target
    assert result.memory.gate0_pitch_provenance.initial_pitch_rad == target.pitch_rad
    assert not result.proposal.is_exact_zero
    validate_relative_gate_state_source(update.state, observation)
    validate_command_proposal_source(result.proposal, update.state)


def test_adapter_module_static_imports_exclude_every_authority_surface():
    module_path = Path(__file__).parents[1] / "vq2_wave3_imu_adapter.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    forbidden = (
        "runner",
        "runtime",
        "transport",
        "mavlink",
        "socket",
        "supervisor",
        "simulator",
        "flightsim",
    )

    assert not {
        imported
        for imported in imports
        if any(token in imported.lower() for token in forbidden)
    }


def test_first_dropout_lower_callsite_is_private_and_owned_only_by_wave3():
    competition_dir = Path(__file__).parents[1]
    wave2_tree = ast.parse(
        (competition_dir / "vq2_wave2_adapter.py").read_text(encoding="utf-8")
    )
    functions = {
        node.name: node
        for node in wave2_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    public_step = functions["step_vq2_wave2_adapter"]
    public_names = {
        argument.arg
        for argument in (
            *public_step.args.args,
            *public_step.args.kwonlyargs,
        )
    }
    assert "allow_first_observation_dropout" not in public_names

    private_name = "_step_vq2_wave2_first_observation_dropout"
    private_step = functions[private_name]
    assert private_name.startswith("_")
    assert "capability" in {
        argument.arg for argument in private_step.args.kwonlyargs
    }

    wave3_tree = ast.parse(
        (competition_dir / "vq2_wave3_imu_adapter.py").read_text(
            encoding="utf-8"
        )
    )
    wave2_private_imports = {
        alias.name
        for node in ast.walk(wave3_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "competition.vq2_wave2_adapter"
        for alias in node.names
        if alias.name.startswith("_")
    }
    assert wave2_private_imports == {
        "_VQ2_WAVE3_FIRST_DROPOUT_CAPABILITY",
        private_name,
    }
