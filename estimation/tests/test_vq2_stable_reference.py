from __future__ import annotations

import ast
import dataclasses
import inspect
import math
from pathlib import Path

import numpy as np
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
    TrackRole,
)
from estimation.vq2_imu_derotation import (
    SUPPORTED_CAMERA_RAY_MODEL_ID,
    VQ2AttitudeDerotationInput,
    VQ2CameraToBodyCalibration,
    VQ2DerotationEvidence,
    VQ2DerotationModel,
    derotate_gate_observation,
)
from estimation.vq2_imu_provenance import VQ2ImuSource, VQ2TimestampedAttitude
from estimation.vq2_relative_estimator import RelativePredictionTarget
from estimation.vq2_stable_reference import (
    HARD_MAX_REFERENCE_AGE_NS,
    IDENTITY_CHART_TO_CAMERA_RAY,
    LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID,
    LOCAL_FEATURE_ORDER,
    SUPPORTED_STABLE_CHART_MODEL_ID,
    VQ2CameraFeatureTime,
    VQ2CovarianceScope,
    VQ2LocalDifferentialFeatureState,
    VQ2LocalFeatureBasis,
    VQ2StableFeatureTransformEvidence,
    VQ2StableReference,
    VQ2StableReferenceChronologyError,
    VQ2StableReferenceError,
    VQ2StableReferenceGeometryError,
    VQ2StableReferenceMismatchError,
    VQ2StableReferenceModel,
    VQ2StableTransformDirection,
    camera_to_stable_local_differential,
    establish_stable_reference,
    stable_to_camera_local_differential,
    validate_stable_measurement_sequence,
)


_BASE_NS = 20_000_000_000
_HOST_CLOCK_ID = "stable-host-clock"
_SESSION_ID = "stable-training-session"
_CAMERA_STREAM_ID = "camera0"
_CAMERA_GENERATION = 4
_IMU_STREAM_ID = "highres-imu0"
_IMU_GENERATION = 3
_CALIBRATION_ID = "stable-camera0-to-body-calibration"
_COVARIANCE_MODEL_ID = "stable-local-feature-covariance-v1"
_ALL_EDGES = FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT | FrameEdge.BOTTOM


def _quaternion_from_euler(
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


def _imu_source(
    *,
    session_id: str = _SESSION_ID,
    reset_epoch: int = 7,
    host_clock_id: str = _HOST_CLOCK_ID,
    stream_id: str = _IMU_STREAM_ID,
    generation: int = _IMU_GENERATION,
) -> VQ2ImuSource:
    return VQ2ImuSource(
        session_id=session_id,
        reset_epoch=reset_epoch,
        host_clock_id=host_clock_id,
        stream_id=stream_id,
        generation=generation,
    )


def _attitude(
    *,
    source: VQ2ImuSource | None = None,
    sample_sequence: int,
    source_time_us: int,
    receive_monotonic_ns: int,
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    body_rates: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_uncertainty_rad: float = 0.001,
    host_time_uncertainty_ns: int = 100_000,
) -> VQ2AttitudeDerotationInput:
    return VQ2AttitudeDerotationInput(
        attitude=VQ2TimestampedAttitude(
            source=source or _imu_source(),
            sample_sequence=sample_sequence,
            source_time_us=source_time_us,
            receive_monotonic_ns=receive_monotonic_ns,
            orientation_body_to_ned_wxyz=orientation,
            body_rates_rad_s=body_rates,
            gyro_bias_rad_s=(0.0, 0.0, 0.0),
            accel_trust=1.0,
            propagated=True,
        ),
        orientation_uncertainty_rad=orientation_uncertainty_rad,
        host_time_uncertainty_ns=host_time_uncertainty_ns,
    )


def _authority(
    *,
    race_status_sequence: int = 12,
    race_status_boot_ms: int = 2_000,
    cutover_publication_sequence: int = 1,
    cutover_publish_ns: int = _BASE_NS - 1_000,
    session_id: str = _SESSION_ID,
    reset_epoch: int = 7,
    gate_epoch: int = 2,
    gate_index: int = 0,
    host_clock_id: str = _HOST_CLOCK_ID,
    camera_stream_id: str = _CAMERA_STREAM_ID,
    camera_generation: int = _CAMERA_GENERATION,
) -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id=session_id,
        reset_epoch=reset_epoch,
        gate_epoch=gate_epoch,
        expected_gate_index=gate_index,
        race_status_sequence=race_status_sequence,
        race_status_boot_ms=race_status_boot_ms,
        camera_host_clock_id=host_clock_id,
        camera_stream_id=camera_stream_id,
        camera_generation=camera_generation,
        frame_publication_sequence_not_before=cutover_publication_sequence,
        frame_publish_monotonic_ns_not_before=cutover_publish_ns,
    )


def _observation(
    *,
    frame_id: int = 41,
    measurement_ns: int = _BASE_NS,
    publication_sequence: int | None = None,
    camera_source_time_ns: int | None = None,
    host_clock_id: str = _HOST_CLOCK_ID,
    camera_stream_id: str = _CAMERA_STREAM_ID,
    camera_generation: int = _CAMERA_GENERATION,
    image_size_px: tuple[int, int] = (640, 360),
    measurement_time_basis: MeasurementTimeBasis = (
        MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY
    ),
    measurement_time_model_id: str | None = None,
    center: tuple[float, float] = (0.1, -0.05),
    side: float = 0.4,
    authority: GateAuthorityEpochV1 | None = None,
    candidate_id: str | None = None,
    clipping: FrameEdge = FrameEdge.NONE,
    health: ObservationHealth = ObservationHealth.NOMINAL,
    fitted: bool = True,
) -> GateObservationV1:
    publication_sequence = (
        frame_id + 1 if publication_sequence is None else publication_sequence
    )
    timing = FrameTimingV1(
        identity=FrameIdentityV1(camera_stream_id, camera_generation, frame_id),
        camera_source_time_ns=(
            900_000_000_000 + frame_id
            if camera_source_time_ns is None
            else camera_source_time_ns
        ),
        host_clock_id=host_clock_id,
        publication_sequence=publication_sequence,
        first_unique_packet_monotonic_ns=measurement_ns - 1_000,
        final_unique_packet_monotonic_ns=measurement_ns,
        reassembly_complete_monotonic_ns=measurement_ns,
        decode_start_monotonic_ns=measurement_ns + 100_000,
        decode_end_monotonic_ns=measurement_ns + 200_000,
        publish_monotonic_ns=measurement_ns + 300_000,
    )
    authority = authority or _authority(
        host_clock_id=host_clock_id,
        camera_stream_id=camera_stream_id,
        camera_generation=camera_generation,
    )
    if not fitted:
        return GateObservationV1(
            frame_timing=timing,
            measurement_time_monotonic_ns=measurement_ns,
            measurement_time_basis=measurement_time_basis,
            measurement_time_model_id=measurement_time_model_id,
            measurement_uncertainty_ns=1_000_000,
            authority=authority,
            candidate_id=candidate_id or f"candidate-{frame_id}",
            image_size_px=image_size_px,
            center_norm=center,
            support_bounds_norm=(0.4, 0.4, 0.6, 0.6),
            outer_edges=EdgeSetV1(),
            inner_edges=EdgeSetV1(),
            inner_corners_norm=(None, None, None, None),
            fitted_inner_aperture_corners_norm=None,
            geometry_model_id=None,
            log_scale=None,
            projective_skew=None,
            clipping=FrameEdge.NONE,
            confidence=0.8,
            covariance=FeatureCovarianceV1(
                "center-only-covariance-v1",
                ("center_x_norm", "center_y_norm"),
                ((1e-4, 0.0), (0.0, 1e-4)),
            ),
            fit=FitDiagnosticsV1(None, 0, 0),
            health=ObservationHealth.DEGRADED,
            health_reason="no_fitted_aperture",
            provenance="synthetic-stable-test",
        )
    half = side * 0.5
    cx, cy = center
    corners = (
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    )
    top_left, top_right, bottom_right, bottom_left = corners
    inner_edges = EdgeSetV1(
        left=LineSegmentV1(top_left, bottom_left),
        top=LineSegmentV1(top_left, top_right),
        right=LineSegmentV1(top_right, bottom_right),
        bottom=LineSegmentV1(bottom_left, bottom_right),
    )
    covariance_order = (
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
    health_reason = None if health is ObservationHealth.NOMINAL else "degraded-test"
    return GateObservationV1(
        frame_timing=timing,
        measurement_time_monotonic_ns=measurement_ns,
        measurement_time_basis=measurement_time_basis,
        measurement_time_model_id=measurement_time_model_id,
        measurement_uncertainty_ns=1_000_000,
        authority=authority,
        candidate_id=candidate_id or f"candidate-{frame_id}",
        image_size_px=image_size_px,
        center_norm=center,
        support_bounds_norm=(0.4, 0.4, 0.6, 0.6),
        outer_edges=EdgeSetV1(),
        inner_edges=inner_edges,
        inner_corners_norm=corners,
        fitted_inner_aperture_corners_norm=corners,
        geometry_model_id="synthetic-complete-quad-v1",
        log_scale=math.log(side),
        projective_skew=(0.0, 0.0),
        clipping=clipping,
        confidence=0.9,
        covariance=FeatureCovarianceV1(
            "synthetic-complete-quad-covariance-v1",
            covariance_order,
            covariance,
        ),
        fit=FitDiagnosticsV1(0.001, 4, 4),
        health=health,
        health_reason=health_reason,
        provenance="synthetic-stable-test",
    )


def _derotation_model(**overrides: object) -> VQ2DerotationModel:
    values: dict[str, object] = {
        "model_id": "stable-rotation-only-v1",
        "attitude_time_model_id": "stable-imu-host-time-v1",
        "max_capture_alignment_ns": 20_000_000,
        "max_target_extrapolation_ns": 20_000_000,
        "max_total_timing_uncertainty_ns": 10_000_000,
        "angular_rate_uncertainty_rad_s": 0.02,
    }
    values.update(overrides)
    return VQ2DerotationModel(**values)  # type: ignore[arg-type]


def _calibration(
    *,
    calibration_id: str = _CALIBRATION_ID,
    camera_to_body: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    uncertainty: float = 0.001,
) -> VQ2CameraToBodyCalibration:
    return VQ2CameraToBodyCalibration(
        calibration_id=calibration_id,
        camera_ray_model_id=SUPPORTED_CAMERA_RAY_MODEL_ID,
        camera_to_body_wxyz=camera_to_body,
        rotation_uncertainty_rad=uncertainty,
    )


def _evidence(
    *,
    frame_id: int = 41,
    measurement_ns: int = _BASE_NS,
    publication_sequence: int | None = None,
    observation: GateObservationV1 | None = None,
    authority: GateAuthorityEpochV1 | None = None,
    capture_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    target_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    capture_body_rates: tuple[float, float, float] = (0.0, 0.0, 0.0),
    target_body_rates: tuple[float, float, float] = (0.0, 0.0, 0.0),
    capture_receive_offset_ns: int = -1_000_000,
    target_receive_offset_ns: int = 5_000_000,
    decision_offset_ns: int = 10_000_000,
    prediction_offset_ns: int | None = None,
    capture_sequence: int | None = None,
    target_sequence: int | None = None,
    capture_source_time_us: int | None = None,
    target_source_time_us: int | None = None,
    source: VQ2ImuSource | None = None,
    capture: VQ2AttitudeDerotationInput | None = None,
    target_attitude: VQ2AttitudeDerotationInput | None = None,
    calibration: VQ2CameraToBodyCalibration | None = None,
    model: VQ2DerotationModel | None = None,
) -> VQ2DerotationEvidence:
    observation = observation or _observation(
        frame_id=frame_id,
        measurement_ns=measurement_ns,
        publication_sequence=publication_sequence,
        authority=authority,
    )
    ordinal = observation.frame_timing.publication_sequence
    capture_sequence = 100 + ordinal * 2 if capture_sequence is None else capture_sequence
    target_sequence = capture_sequence + 1 if target_sequence is None else target_sequence
    capture_source_time_us = (
        9_000_000_000_000 + ordinal * 20_000
        if capture_source_time_us is None
        else capture_source_time_us
    )
    target_source_time_us = (
        capture_source_time_us + 10_000
        if target_source_time_us is None
        else target_source_time_us
    )
    source = source or _imu_source()
    capture = capture or _attitude(
        source=source,
        sample_sequence=capture_sequence,
        source_time_us=capture_source_time_us,
        receive_monotonic_ns=measurement_ns + capture_receive_offset_ns,
        orientation=capture_orientation,
        body_rates=capture_body_rates,
    )
    target_attitude = target_attitude or _attitude(
        source=source,
        sample_sequence=target_sequence,
        source_time_us=target_source_time_us,
        receive_monotonic_ns=measurement_ns + target_receive_offset_ns,
        orientation=target_orientation,
        body_rates=target_body_rates,
    )
    decision_ns = measurement_ns + decision_offset_ns
    prediction_ns = (
        decision_ns if prediction_offset_ns is None else measurement_ns + prediction_offset_ns
    )
    target = (
        RelativePredictionTarget.at_decision(observation.host_clock_id, decision_ns)
        if prediction_ns == decision_ns
        else RelativePredictionTarget(
            host_clock_id=observation.host_clock_id,
            decision_time_monotonic_ns=decision_ns,
            prediction_time_monotonic_ns=prediction_ns,
            prediction_basis=PredictionBasis.COMMAND_EFFECT_ESTIMATE,
            delay_model_id="stable-command-effect-v1",
            delay_uncertainty_ns=500_000,
        )
    )
    return derotate_gate_observation(
        observation,
        target,
        capture_attitude=capture,
        target_attitude=target_attitude,
        calibration=calibration or _calibration(),
        model=model or _derotation_model(),
    )


def _dense_covariance(scale: float = 1.0) -> tuple[tuple[float, ...], ...]:
    lower = np.array(
        (
            (0.020, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.004, 0.025, 0.0, 0.0, 0.0, 0.0),
            (-0.003, 0.002, 0.030, 0.0, 0.0, 0.0),
            (0.002, -0.001, 0.003, 0.040, 0.0, 0.0),
            (0.001, 0.002, -0.002, 0.004, 0.045, 0.0),
            (-0.002, 0.001, 0.004, -0.003, 0.002, 0.050),
        ),
        dtype=np.float64,
    )
    covariance = lower @ lower.T * scale
    return tuple(tuple(float(value) for value in row) for row in covariance)


def _diagonal_covariance(value: float) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(value if row == column else 0.0 for column in range(6))
        for row in range(6)
    )


def _stable_model(**overrides: object) -> VQ2StableReferenceModel:
    values: dict[str, object] = {
        "model_id": "stable-reference-local-feature-v1",
        "local_feature_model_id": LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID,
        "chart_model_id": SUPPORTED_STABLE_CHART_MODEL_ID,
        "chart_to_camera_ray": IDENTITY_CHART_TO_CAMERA_RAY,
        "covariance_model_id": _COVARIANCE_MODEL_ID,
        "max_reference_age_ns": 250_000_000,
        "max_total_timing_uncertainty_ns": 20_000_000,
        "max_relative_angular_uncertainty_rad": 0.1,
        "max_angular_rate_rad_s": 4.0,
        "max_angular_rate_uncertainty_rad_s": 0.1,
        "max_angular_acceleration_rad_s2": 20.0,
        "max_feature_acceleration_norm_s2": 20.0,
        "minimum_forward": 1e-3,
        "max_homography_condition": 10.0,
        "max_feature_jacobian_condition": 20.0,
        "max_projective_magnification": 10.0,
        "max_abs_bearing_norm": 3.5,
        "max_abs_bearing_rate_norm_s": 7.0,
        "max_abs_local_log_scale": 10.0,
        "max_abs_local_expansion_rate_s": 7.0,
        "covariance_psd_tolerance": 1e-10,
        "covariance_eigenvalue_floor": 1e-12,
        "joint_nuisance_envelope_covariance": _dense_covariance(0.05),
        "model_floor_covariance": _diagonal_covariance(1e-8),
    }
    values.update(overrides)
    return VQ2StableReferenceModel(**values)  # type: ignore[arg-type]


def _reference(
    seed: VQ2DerotationEvidence | None = None,
    *,
    tracker_id: str = "stable-track-0",
    track_role: TrackRole = TrackRole.ACTIVE,
    model: VQ2StableReferenceModel | None = None,
) -> VQ2StableReference:
    return establish_stable_reference(
        reference_id="stable-reference-0",
        tracker_id=tracker_id,
        track_role=track_role,
        seed_evidence=seed or _evidence(),
        model=model or _stable_model(),
    )


def _state(
    reference: VQ2StableReference,
    evidence: VQ2DerotationEvidence,
    *,
    basis: VQ2LocalFeatureBasis = VQ2LocalFeatureBasis.CAMERA,
    camera_time: VQ2CameraFeatureTime = VQ2CameraFeatureTime.CAPTURE,
    values: tuple[float, float, float, float, float, float] = (
        0.2,
        -0.15,
        -0.7,
        0.4,
        -0.1,
        0.25,
    ),
    covariance: tuple[tuple[float, ...], ...] | None = None,
    scope: VQ2CovarianceScope = VQ2CovarianceScope.CONDITIONAL_INPUT,
    **overrides: object,
) -> VQ2LocalDifferentialFeatureState:
    time_ns = (
        evidence.observation.measurement_time_monotonic_ns
        if camera_time is VQ2CameraFeatureTime.CAPTURE
        else evidence.prediction_target.prediction_time_monotonic_ns
    )
    kwargs: dict[str, object] = {
        "feature_model_id": LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID,
        "chart_model_id": reference.key.stable_model.chart_model_id,
        "basis": basis,
        "basis_id": (
            reference.key.calibration.calibration_id
            if basis is VQ2LocalFeatureBasis.CAMERA
            else reference.basis_id
        ),
        "host_clock_id": _HOST_CLOCK_ID,
        "time_monotonic_ns": time_ns,
        "values": values,
        "covariance_model_id": _COVARIANCE_MODEL_ID,
        "covariance_scope": scope,
        "covariance": covariance or _dense_covariance(),
    }
    kwargs.update(overrides)
    return VQ2LocalDifferentialFeatureState(**kwargs)  # type: ignore[arg-type]


def _later_evidence(
    seed: VQ2DerotationEvidence,
    *,
    frame_delta: int = 1,
    time_delta_ns: int = 20_000_000,
    capture_orientation: tuple[float, float, float, float] = _quaternion_from_euler(
        roll=0.08,
        pitch=-0.12,
        yaw=0.18,
    ),
    target_orientation: tuple[float, float, float, float] = _quaternion_from_euler(
        roll=0.09,
        pitch=-0.10,
        yaw=0.20,
    ),
    capture_body_rates: tuple[float, float, float] = (0.1, -0.2, 0.3),
    target_body_rates: tuple[float, float, float] = (0.12, -0.18, 0.28),
    authority: GateAuthorityEpochV1 | None = None,
    **overrides: object,
) -> VQ2DerotationEvidence:
    seed_observation = seed.observation
    frame_id = seed_observation.frame.frame_id + frame_delta
    publication_sequence = seed_observation.frame_timing.publication_sequence + frame_delta
    measurement_ns = seed_observation.measurement_time_monotonic_ns + time_delta_ns
    authority = authority or dataclasses.replace(
        seed_observation.authority,
        race_status_sequence=seed_observation.authority.race_status_sequence + frame_delta,
        race_status_boot_ms=seed_observation.authority.race_status_boot_ms + 5 * frame_delta,
    )
    return _evidence(
        frame_id=frame_id,
        measurement_ns=measurement_ns,
        publication_sequence=publication_sequence,
        authority=authority,
        capture_orientation=capture_orientation,
        target_orientation=target_orientation,
        capture_body_rates=capture_body_rates,
        target_body_rates=target_body_rates,
        capture_sequence=seed.capture_attitude.attitude.sample_sequence + 2 * frame_delta,
        target_sequence=seed.capture_attitude.attitude.sample_sequence + 2 * frame_delta + 1,
        capture_source_time_us=seed.capture_attitude.attitude.source_time_us
        + 20_000 * frame_delta,
        target_source_time_us=seed.capture_attitude.attitude.source_time_us
        + 20_000 * frame_delta
        + 10_000,
        **overrides,
    )


def _conditional_from_forward(
    transform: VQ2StableFeatureTransformEvidence,
) -> VQ2LocalDifferentialFeatureState:
    output = transform.output_state
    return dataclasses.replace(
        output,
        covariance_scope=VQ2CovarianceScope.CONDITIONAL_INPUT,
        covariance=transform.coordinate_covariance,
    )


def test_contract_freeze_uses_distinct_local_semantic_and_required_model():
    assert LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID == "vq2-local-differential-area-v1"
    assert LOCAL_FEATURE_ORDER == (
        "bearing_x_norm",
        "bearing_y_norm",
        "local_log_scale",
        "bearing_rate_x_norm_s",
        "bearing_rate_y_norm_s",
        "local_expansion_rate_s",
    )
    assert "GateObservationV1" not in inspect.signature(
        camera_to_stable_local_differential
    ).__str__()
    required = inspect.signature(VQ2StableReferenceModel).parameters
    assert all(parameter.default is inspect.Parameter.empty for parameter in required.values())
    assert tuple(inspect.signature(establish_stable_reference).parameters) == (
        "reference_id",
        "tracker_id",
        "track_role",
        "seed_evidence",
        "model",
    )
    assert tuple(
        inspect.signature(camera_to_stable_local_differential).parameters
    ) == ("reference", "evidence", "state", "camera_time")
    assert tuple(
        inspect.signature(stable_to_camera_local_differential).parameters
    ) == ("reference", "evidence", "state", "camera_time")
    assert tuple(inspect.signature(validate_stable_measurement_sequence).parameters) == (
        "reference",
        "transforms",
    )


def test_identity_capture_transform_preserves_local_state_and_adds_named_envelopes():
    seed = _evidence()
    reference = _reference(seed)
    state = _state(reference, seed)

    result = camera_to_stable_local_differential(
        reference,
        seed,
        state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )

    assert result.direction is VQ2StableTransformDirection.CAMERA_TO_STABLE
    assert result.output_state.basis is VQ2LocalFeatureBasis.STABLE_REFERENCE
    assert result.output_state.basis_id == reference.basis_id
    assert result.output_state.values == pytest.approx(state.values, abs=1e-12)
    assert np.asarray(result.state_jacobian) == pytest.approx(np.eye(6), abs=1e-12)
    expected_total = (
        np.asarray(state.covariance)
        + np.asarray(reference.key.stable_model.joint_nuisance_envelope_covariance)
        + np.asarray(reference.key.stable_model.model_floor_covariance)
    )
    assert np.asarray(result.total_covariance) == pytest.approx(expected_total, abs=1e-12)
    assert result.output_state.covariance_scope is VQ2CovarianceScope.TRANSFORM_TOTAL


@pytest.mark.parametrize("camera_time", tuple(VQ2CameraFeatureTime))
def test_nontrivial_forward_inverse_roundtrip_recovers_state_jacobian_and_coordinate_covariance(
    camera_time: VQ2CameraFeatureTime,
):
    seed = _evidence()
    reference = _reference(seed)
    current = _later_evidence(seed)
    state = _state(reference, current, camera_time=camera_time)
    forward = camera_to_stable_local_differential(
        reference,
        current,
        state,
        camera_time=camera_time,
    )
    inverse = stable_to_camera_local_differential(
        reference,
        current,
        _conditional_from_forward(forward),
        camera_time=camera_time,
    )

    assert inverse.output_state.values == pytest.approx(state.values, abs=2e-11)
    composed = np.asarray(inverse.state_jacobian) @ np.asarray(forward.state_jacobian)
    assert composed == pytest.approx(np.eye(6), abs=2e-10)
    assert np.asarray(inverse.coordinate_covariance) == pytest.approx(
        np.asarray(state.covariance),
        abs=2e-11,
    )


def test_direct_total_covariance_label_is_rejected_as_conditional_input():
    seed = _evidence()
    reference = _reference(seed)
    state = _state(reference, seed)
    result = camera_to_stable_local_differential(
        reference,
        seed,
        state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    with pytest.raises(VQ2StableReferenceError, match="exclude previously added"):
        stable_to_camera_local_differential(
            reference,
            seed,
            result.output_state,
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


@pytest.mark.parametrize("camera_time", tuple(VQ2CameraFeatureTime))
def test_analytic_six_by_six_jacobian_matches_independent_central_difference(
    camera_time: VQ2CameraFeatureTime,
):
    seed = _evidence()
    reference = _reference(seed)
    current = _later_evidence(seed)
    state = _state(reference, current, camera_time=camera_time)
    nominal = camera_to_stable_local_differential(
        reference,
        current,
        state,
        camera_time=camera_time,
    )
    numerical = np.zeros((6, 6), dtype=np.float64)
    epsilon = 1e-6
    for column in range(6):
        plus = list(state.values)
        minus = list(state.values)
        plus[column] += epsilon
        minus[column] -= epsilon
        plus_result = camera_to_stable_local_differential(
            reference,
            current,
            dataclasses.replace(state, values=tuple(plus)),
            camera_time=camera_time,
        )
        minus_result = camera_to_stable_local_differential(
            reference,
            current,
            dataclasses.replace(state, values=tuple(minus)),
            camera_time=camera_time,
        )
        numerical[:, column] = (
            np.asarray(plus_result.output_state.values)
            - np.asarray(minus_result.output_state.values)
        ) / (2.0 * epsilon)
    assert np.asarray(nominal.state_jacobian) == pytest.approx(numerical, abs=2e-8)


def test_dense_covariance_congruence_retains_cross_terms_and_is_psd():
    seed = _evidence()
    reference = _reference(seed)
    current = _later_evidence(seed)
    state = _state(reference, current)
    result = camera_to_stable_local_differential(
        reference,
        current,
        state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    jacobian = np.asarray(result.state_jacobian)
    expected = jacobian @ np.asarray(state.covariance) @ jacobian.T
    assert np.asarray(result.coordinate_covariance) == pytest.approx(expected, abs=2e-12)
    assert abs(result.coordinate_covariance[0][5]) > 0.0
    assert np.linalg.eigvalsh(np.asarray(result.coordinate_covariance))[0] > 0.0
    increment = np.asarray(result.total_covariance) - np.asarray(
        result.coordinate_covariance
    )
    assert np.linalg.eigvalsh(increment)[0] > 0.0


def test_positive_body_yaw_maps_current_forward_ray_right_in_frozen_reference():
    seed = _evidence()
    reference = _reference(seed)
    current = _later_evidence(
        seed,
        capture_orientation=_quaternion_from_euler(yaw=0.2),
        target_orientation=_quaternion_from_euler(yaw=0.2),
        capture_body_rates=(0.0, 0.0, 0.0),
        target_body_rates=(0.0, 0.0, 0.0),
    )
    state = _state(
        reference,
        current,
        values=(0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
    )
    result = camera_to_stable_local_differential(
        reference,
        current,
        state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert result.output_state.values[0] == pytest.approx(math.tan(0.2), abs=1e-12)
    assert result.output_state.values[1] == pytest.approx(0.0, abs=1e-12)


def test_nonidentity_yaw_freezes_local_area_scale_law_and_diagnostic():
    seed = _evidence()
    reference = _reference(seed)
    yaw = 0.2
    local_log_scale = -0.5
    current = _later_evidence(
        seed,
        capture_orientation=_quaternion_from_euler(yaw=yaw),
        target_orientation=_quaternion_from_euler(yaw=yaw),
        capture_body_rates=(0.0, 0.0, 0.0),
        target_body_rates=(0.0, 0.0, 0.0),
    )
    result = camera_to_stable_local_differential(
        reference,
        current,
        _state(
            reference,
            current,
            values=(0.0, 0.0, local_log_scale, 0.0, 0.0, 0.0),
        ),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    expected_area_determinant = 1.0 / math.cos(yaw) ** 3
    assert result.output_state.values[2] == pytest.approx(
        local_log_scale - 1.5 * math.log(math.cos(yaw)),
        abs=1e-12,
    )
    assert result.local_area_jacobian_determinant == pytest.approx(
        expected_area_determinant,
        abs=1e-12,
    )
    assert math.exp(2.0 * (result.output_state.values[2] - local_log_scale)) \
        == pytest.approx(expected_area_determinant, abs=1e-12)


def test_positive_body_pitch_maps_current_forward_ray_up_in_frozen_reference():
    seed = _evidence()
    reference = _reference(seed)
    pitch = 0.2
    current = _later_evidence(
        seed,
        capture_orientation=_quaternion_from_euler(pitch=pitch),
        target_orientation=_quaternion_from_euler(pitch=pitch),
        capture_body_rates=(0.0, 0.0, 0.0),
        target_body_rates=(0.0, 0.0, 0.0),
    )
    result = camera_to_stable_local_differential(
        reference,
        current,
        _state(
            reference,
            current,
            values=(0.0, 0.0, -0.5, 0.0, 0.0, 0.0),
        ),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert result.output_state.values[0] == pytest.approx(0.0, abs=1e-12)
    assert result.output_state.values[1] == pytest.approx(-math.tan(pitch), abs=1e-12)


def test_positive_body_roll_rotates_current_right_bearing_down_in_frozen_reference():
    seed = _evidence()
    reference = _reference(seed)
    roll = 0.2
    right = 0.3
    current = _later_evidence(
        seed,
        capture_orientation=_quaternion_from_euler(roll=roll),
        target_orientation=_quaternion_from_euler(roll=roll),
        capture_body_rates=(0.0, 0.0, 0.0),
        target_body_rates=(0.0, 0.0, 0.0),
    )
    result = camera_to_stable_local_differential(
        reference,
        current,
        _state(
            reference,
            current,
            values=(right, 0.0, -0.5, 0.0, 0.0, 0.0),
        ),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert result.output_state.values[0] == pytest.approx(
        math.cos(roll) * right,
        abs=1e-12,
    )
    assert result.output_state.values[1] == pytest.approx(
        math.sin(roll) * right,
        abs=1e-12,
    )


def test_nonidentity_extrinsic_rotates_body_rate_into_camera_chain_rule():
    camera_to_body = _quaternion_from_euler(roll=math.pi / 2.0)
    calibration = _calibration(camera_to_body=camera_to_body)
    seed = _evidence(calibration=calibration)
    reference = _reference(seed)
    yaw_rate = 0.4
    current = _later_evidence(
        seed,
        capture_orientation=(1.0, 0.0, 0.0, 0.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        capture_body_rates=(0.0, 0.0, yaw_rate),
        target_body_rates=(0.0, 0.0, yaw_rate),
        capture_receive_offset_ns=0,
        calibration=calibration,
    )
    # A +body-z rate becomes +camera-y under this camera-to-body rotation.
    # The matching +v camera bearing rate holds a fixed forward stable ray.
    result = camera_to_stable_local_differential(
        reference,
        current,
        _state(
            reference,
            current,
            values=(0.0, 0.0, -0.5, 0.0, yaw_rate, 0.0),
        ),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert result.selected_camera_angular_rate_rad_s == pytest.approx(
        (0.0, yaw_rate, 0.0),
        abs=1e-12,
    )
    assert result.output_state.values[3:6] == pytest.approx(
        (0.0, 0.0, 0.0),
        abs=2e-12,
    )


def test_accepted_near_unit_extrinsic_uses_one_normalized_rotation_for_rate_and_chart():
    scale = math.sqrt(1.0 + 0.9e-9)
    calibration = _calibration(camera_to_body=(scale, 0.0, 0.0, 0.0))
    seed = _evidence(calibration=calibration)
    reference = _reference(seed)
    yaw_rate = 0.4
    current = _later_evidence(
        seed,
        capture_orientation=(1.0, 0.0, 0.0, 0.0),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        capture_body_rates=(0.0, 0.0, yaw_rate),
        target_body_rates=(0.0, 0.0, yaw_rate),
        capture_receive_offset_ns=0,
        calibration=calibration,
    )
    result = camera_to_stable_local_differential(
        reference,
        current,
        _state(
            reference,
            current,
            values=(0.0, 0.0, -0.5, -yaw_rate, 0.0, 0.0),
        ),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert result.selected_camera_angular_rate_rad_s == pytest.approx(
        (0.0, 0.0, yaw_rate),
        abs=1e-15,
    )
    assert result.output_state.values[3:6] == pytest.approx(
        (0.0, 0.0, 0.0),
        abs=2e-12,
    )


def test_rate_and_expansion_chain_rule_removes_camera_rotation_in_stable_chart():
    seed = _evidence()
    reference = _reference(seed)
    yaw = 0.12
    yaw_rate = 0.4
    current = _later_evidence(
        seed,
        capture_orientation=_quaternion_from_euler(yaw=yaw),
        target_orientation=_quaternion_from_euler(yaw=yaw),
        capture_body_rates=(0.0, 0.0, yaw_rate),
        target_body_rates=(0.0, 0.0, yaw_rate),
    )
    # The capture sample is propagated from its receive timestamp to the image
    # measurement timestamp.  A stationary reference-frame forward ray appears
    # at u=-tan(effective_yaw) in that propagated camera, with the exact
    # rotation-induced slope and scale rates.
    effective_yaw = yaw + yaw_rate * current.capture_alignment_ns * 1e-9
    u = -math.tan(effective_yaw)
    u_rate = -yaw_rate * (1.0 + u * u)
    local_expansion = -1.5 * yaw_rate * u
    state = _state(
        reference,
        current,
        values=(u, 0.0, -0.6, u_rate, 0.0, local_expansion),
    )
    result = camera_to_stable_local_differential(
        reference,
        current,
        state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert result.output_state.values[0] == pytest.approx(0.0, abs=2e-12)
    assert result.output_state.values[3] == pytest.approx(0.0, abs=2e-12)
    assert result.output_state.values[5] == pytest.approx(0.0, abs=2e-12)


def test_negative_capture_alignment_matches_existing_public_point_derotation():
    seed = _evidence()
    reference = _reference(seed)
    current = _later_evidence(
        seed,
        capture_orientation=_quaternion_from_euler(yaw=0.15),
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        capture_body_rates=(0.0, 0.0, 0.2),
        target_body_rates=(0.0, 0.0, 0.0),
        capture_receive_offset_ns=1_000_000,
    )
    assert current.capture_alignment_ns == -1_000_000
    state = _state(
        reference,
        current,
        values=(*current.observation.center_norm, -0.7, 0.0, 0.0, 0.0),
    )
    result = camera_to_stable_local_differential(
        reference,
        current,
        state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert result.output_state.values[:2] == pytest.approx(
        current.derotated_center_norm,
        abs=2e-12,
    )


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * abs(
        sum(
            x1 * y2 - y1 * x2
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    )


def _yaw_project(point: tuple[float, float], yaw: float) -> tuple[float, float]:
    x, y = point
    denominator = math.cos(yaw) - math.sin(yaw) * x
    return (
        (math.sin(yaw) + math.cos(yaw) * x) / denominator,
        y / denominator,
    )


def test_finite_quad_counterexample_proves_frozen_v1_scale_is_not_local_scale():
    wide = ((-0.4, -0.1), (0.4, -0.1), (0.4, 0.1), (-0.4, 0.1))
    square = ((-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2))
    assert _polygon_area(wide) == pytest.approx(0.16)
    assert _polygon_area(square) == pytest.approx(0.16)
    yaw = math.radians(20.0)
    wide_area = _polygon_area(tuple(_yaw_project(point, yaw) for point in wide))
    square_area = _polygon_area(tuple(_yaw_project(point, yaw) for point in square))
    assert wide_area == pytest.approx(0.20126625582118124, abs=1e-15)
    assert square_area == pytest.approx(0.1948845456305093, abs=1e-15)
    assert wide_area != pytest.approx(square_area, abs=1e-4)


def test_reference_establishment_requires_complete_visible_unclipped_quad():
    with pytest.raises(VQ2StableReferenceError, match="complete all-visible"):
        _reference(_evidence(observation=_observation(fitted=False)))
    clipped = _observation(clipping=FrameEdge.TOP, health=ObservationHealth.DEGRADED)
    with pytest.raises(VQ2StableReferenceError, match="unclipped"):
        _reference(_evidence(observation=clipped))


def test_later_local_feature_transform_does_not_require_v1_finite_quad_summaries():
    seed = _evidence()
    reference = _reference(seed)
    partial_observation = _observation(
        frame_id=42,
        measurement_ns=_BASE_NS + 20_000_000,
        publication_sequence=43,
        fitted=False,
    )
    current = _later_evidence(seed, observation=partial_observation)
    result = camera_to_stable_local_differential(
        reference,
        current,
        _state(reference, current),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert result.output_state.feature_model_id == LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID


def test_reference_is_deterministic_immutable_and_deeply_revalidated():
    seed = _evidence()
    first = _reference(seed)
    second = _reference(seed)
    assert first == second
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.reference_id = "changed"  # type: ignore[misc]
    object.__setattr__(first, "reference_angular_uncertainty_rad", 0.05)
    with pytest.raises(VQ2StableReferenceError, match="angular uncertainty"):
        first.validate_integrity()

    corrupted_model = _stable_model()
    object.__setattr__(
        corrupted_model,
        "max_reference_age_ns",
        HARD_MAX_REFERENCE_AGE_NS + 1,
    )
    with pytest.raises(VQ2StableReferenceError, match="max_reference_age_ns"):
        establish_stable_reference(
            reference_id="corrupted-model-reference",
            tracker_id="stable-track-0",
            track_role=TrackRole.ACTIVE,
            seed_evidence=seed,
            model=corrupted_model,
        )


def test_same_caller_reference_id_cannot_alias_distinct_seed_charts():
    first_seed = _evidence()
    second_seed = _evidence(
        capture_orientation=_quaternion_from_euler(yaw=0.2),
        target_orientation=_quaternion_from_euler(yaw=0.2),
    )
    first = _reference(first_seed)
    second = _reference(second_seed)
    assert first.reference_id == second.reference_id
    assert first.key == second.key
    assert first.basis_id != second.basis_id
    stable_under_first = _state(
        first,
        first_seed,
        basis=VQ2LocalFeatureBasis.STABLE_REFERENCE,
    )
    with pytest.raises(VQ2StableReferenceMismatchError, match="basis"):
        stable_to_camera_local_differential(
            second,
            second_seed,
            stable_under_first,
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("feature_model_id", "wrong-feature-v1", "wrong semantic"),
        ("chart_model_id", "wrong-chart-v1", "chart model changed"),
        ("basis_id", "wrong-basis", "basis"),
        ("host_clock_id", "wrong-clock", "host clock"),
        ("time_monotonic_ns", _BASE_NS + 1, "state time"),
        ("covariance_model_id", "wrong-covariance", "covariance model"),
        (
            "covariance_scope",
            VQ2CovarianceScope.TRANSFORM_TOTAL,
            "exclude previously added",
        ),
    ),
)
def test_transform_rejects_wrong_state_semantic_basis_time_or_covariance_scope(
    field: str,
    value: object,
    match: str,
):
    seed = _evidence()
    reference = _reference(seed)
    state = _state(reference, seed)
    if field == "feature_model_id":
        with pytest.raises(VQ2StableReferenceError, match="wrong semantic"):
            dataclasses.replace(state, **{field: value})
        return
    changed = dataclasses.replace(state, **{field: value})
    with pytest.raises(VQ2StableReferenceError, match=match):
        camera_to_stable_local_differential(
            reference,
            seed,
            changed,
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_camera_and_stable_basis_directions_are_exact():
    seed = _evidence()
    reference = _reference(seed)
    camera = _state(reference, seed)
    stable = dataclasses.replace(
        camera,
        basis=VQ2LocalFeatureBasis.STABLE_REFERENCE,
        basis_id=reference.basis_id,
    )
    with pytest.raises(VQ2StableReferenceMismatchError, match="basis"):
        camera_to_stable_local_differential(
            reference,
            seed,
            stable,
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )
    with pytest.raises(VQ2StableReferenceMismatchError, match="basis"):
        stable_to_camera_local_differential(
            reference,
            seed,
            camera,
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_capture_and_target_time_bindings_are_not_interchangeable():
    seed = _evidence(prediction_offset_ns=12_000_000)
    reference = _reference(seed)
    capture = _state(reference, seed, camera_time=VQ2CameraFeatureTime.CAPTURE)
    with pytest.raises(VQ2StableReferenceMismatchError, match="state time"):
        camera_to_stable_local_differential(
            reference,
            seed,
            capture,
            camera_time=VQ2CameraFeatureTime.TARGET,
        )


def test_reference_key_rejects_tracker_role_calibration_model_and_imu_epoch_changes():
    seed = _evidence()
    reference = _reference(seed)
    current = _later_evidence(seed)
    camera_state = _state(reference, current)
    assert _reference(seed, tracker_id="other-track").key != reference.key
    assert _reference(seed, track_role=TrackRole.SHADOW).key != reference.key

    changed_calibration = _later_evidence(
        seed,
        calibration=_calibration(camera_to_body=_quaternion_from_euler(roll=0.01)),
    )
    with pytest.raises(VQ2StableReferenceMismatchError, match="reference key"):
        camera_to_stable_local_differential(
            reference,
            changed_calibration,
            _state(reference, changed_calibration),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )
    changed_model = _later_evidence(
        seed,
        model=_derotation_model(angular_rate_uncertainty_rad_s=0.03),
    )
    with pytest.raises(VQ2StableReferenceMismatchError, match="reference key"):
        camera_to_stable_local_differential(
            reference,
            changed_model,
            _state(reference, changed_model),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )
    changed_source = _imu_source(generation=_IMU_GENERATION + 1)
    changed_imu = _later_evidence(seed, source=changed_source)
    with pytest.raises(VQ2StableReferenceMismatchError, match="reference key"):
        camera_to_stable_local_differential(
            reference,
            changed_imu,
            _state(reference, changed_imu),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )
    # Keep one ordinary valid call beside the adversarial variants.
    assert camera_to_stable_local_differential(
        reference,
        current,
        camera_state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    ).reference == reference


@pytest.mark.parametrize(
    "lifecycle_change",
    (
        "session",
        "reset",
        "gate_epoch",
        "gate_index",
        "camera_clock",
        "camera_stream_generation",
        "image_size",
        "measurement_time_model",
    ),
)
def test_reference_key_rejects_every_camera_authority_and_time_lifecycle_change(
    lifecycle_change: str,
):
    seed = _evidence()
    reference = _reference(seed)
    observation_kwargs: dict[str, object] = {}
    source = _imu_source()
    if lifecycle_change == "session":
        observation_kwargs["authority"] = _authority(session_id="other-session")
        source = _imu_source(session_id="other-session")
    elif lifecycle_change == "reset":
        observation_kwargs["authority"] = _authority(reset_epoch=8)
        source = _imu_source(reset_epoch=8)
    elif lifecycle_change == "gate_epoch":
        observation_kwargs["authority"] = _authority(gate_epoch=3)
    elif lifecycle_change == "gate_index":
        observation_kwargs["authority"] = _authority(gate_index=1)
    elif lifecycle_change == "camera_clock":
        observation_kwargs["host_clock_id"] = "other-stable-clock"
        source = _imu_source(host_clock_id="other-stable-clock")
    elif lifecycle_change == "camera_stream_generation":
        observation_kwargs["camera_stream_id"] = "camera1"
        observation_kwargs["camera_generation"] = 5
    elif lifecycle_change == "image_size":
        observation_kwargs["image_size_px"] = (800, 450)
    else:
        observation_kwargs["measurement_time_basis"] = (
            MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED
        )
        observation_kwargs["measurement_time_model_id"] = "camera-time-v2"
    changed = _evidence(
        observation=_observation(**observation_kwargs),  # type: ignore[arg-type]
        source=source,
    )
    assert _reference(changed).key != reference.key
    with pytest.raises(VQ2StableReferenceMismatchError, match="reference key"):
        camera_to_stable_local_differential(
            reference,
            changed,
            _state(reference, changed),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_same_authority_snapshot_sequence_cannot_change_boot_time():
    seed = _evidence()
    reference = _reference(seed)
    authority = dataclasses.replace(
        seed.observation.authority,
        race_status_boot_ms=seed.observation.authority.race_status_boot_ms + 1,
    )
    current = _later_evidence(seed, authority=authority)
    with pytest.raises(VQ2StableReferenceChronologyError, match="multiple boot times"):
        camera_to_stable_local_differential(
            reference,
            current,
            _state(reference, current),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


@pytest.mark.parametrize(
    "regressed_field",
    (
        "race_status_sequence",
        "race_status_boot_ms",
        "frame_publication_sequence_not_before",
        "frame_publish_monotonic_ns_not_before",
    ),
)
def test_each_authority_snapshot_watermark_must_not_regress(regressed_field: str):
    seed = _evidence()
    reference = _reference(seed)
    previous = seed.observation.authority
    updates: dict[str, object] = {
        "race_status_sequence": previous.race_status_sequence + 1,
        "race_status_boot_ms": previous.race_status_boot_ms + 5,
        "frame_publication_sequence_not_before": (
            previous.frame_publication_sequence_not_before + 1
        ),
        "frame_publish_monotonic_ns_not_before": (
            previous.frame_publish_monotonic_ns_not_before + 1
        ),
    }
    updates[regressed_field] = getattr(previous, regressed_field) - 1
    authority = dataclasses.replace(previous, **updates)
    current = _later_evidence(seed, authority=authority)
    with pytest.raises(VQ2StableReferenceChronologyError, match="snapshot regressed"):
        camera_to_stable_local_differential(
            reference,
            current,
            _state(reference, current),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_camera_source_time_must_advance_from_reference_and_within_sequence():
    seed = _evidence()
    reference = _reference(seed)
    seed_source_time = seed.observation.frame_timing.camera_source_time_ns
    regressed_observation = _observation(
        frame_id=42,
        measurement_ns=_BASE_NS + 20_000_000,
        publication_sequence=43,
        camera_source_time_ns=seed_source_time,
    )
    regressed = _later_evidence(seed, observation=regressed_observation)
    with pytest.raises(VQ2StableReferenceChronologyError, match="source/publication"):
        camera_to_stable_local_differential(
            reference,
            regressed,
            _state(reference, regressed),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )

    second_observation = _observation(
        frame_id=42,
        measurement_ns=_BASE_NS + 20_000_000,
        publication_sequence=43,
        camera_source_time_ns=seed_source_time + 20,
    )
    third_observation = _observation(
        frame_id=43,
        measurement_ns=_BASE_NS + 40_000_000,
        publication_sequence=44,
        camera_source_time_ns=seed_source_time + 10,
    )
    second = _later_evidence(seed, observation=second_observation)
    third = _later_evidence(
        seed,
        frame_delta=2,
        time_delta_ns=40_000_000,
        observation=third_observation,
    )
    transforms = tuple(
        camera_to_stable_local_differential(
            reference,
            evidence,
            _state(reference, evidence),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )
        for evidence in (second, third)
    )
    with pytest.raises(VQ2StableReferenceChronologyError, match="source/publication"):
        validate_stable_measurement_sequence(reference, transforms)


def test_monotonic_same_epoch_authority_snapshot_refresh_is_accepted():
    seed = _evidence()
    reference = _reference(seed)
    authority = dataclasses.replace(
        seed.observation.authority,
        race_status_sequence=seed.observation.authority.race_status_sequence + 1,
        race_status_boot_ms=seed.observation.authority.race_status_boot_ms + 5,
        frame_publication_sequence_not_before=(
            seed.observation.authority.frame_publication_sequence_not_before + 1
        ),
        frame_publish_monotonic_ns_not_before=(
            seed.observation.authority.frame_publish_monotonic_ns_not_before + 1
        ),
    )
    current = _later_evidence(seed, authority=authority)
    result = camera_to_stable_local_differential(
        reference,
        current,
        _state(reference, current),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert result.reference.key == reference.key


def test_later_decision_and_prediction_cannot_regress_from_reference():
    seed = _evidence()
    reference = _reference(seed)
    current = _later_evidence(
        seed,
        time_delta_ns=5_000_000,
        target_receive_offset_ns=300_000,
        decision_offset_ns=400_000,
    )
    assert (
        current.prediction_target.decision_time_monotonic_ns
        < seed.prediction_target.decision_time_monotonic_ns
    )
    with pytest.raises(VQ2StableReferenceChronologyError, match="decision or prediction"):
        camera_to_stable_local_differential(
            reference,
            current,
            _state(reference, current),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_reference_age_boundary_and_one_nanosecond_over_are_fail_closed():
    seed = _evidence()
    model = _stable_model(max_reference_age_ns=40_000_000)
    reference = _reference(seed, model=model)
    at_bound = _later_evidence(seed, time_delta_ns=40_000_000)
    camera_to_stable_local_differential(
        reference,
        at_bound,
        _state(reference, at_bound),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    over = _later_evidence(seed, time_delta_ns=40_000_001)
    with pytest.raises(VQ2StableReferenceChronologyError, match="age"):
        camera_to_stable_local_differential(
            reference,
            over,
            _state(reference, over),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


@pytest.mark.parametrize(
    ("field", "match"),
    (
        ("max_total_timing_uncertainty_ns", "timing uncertainty"),
        ("max_relative_angular_uncertainty_rad", "angular uncertainty"),
        ("max_angular_rate_rad_s", "angular rate"),
        ("max_angular_rate_uncertainty_rad_s", "angular-rate uncertainty"),
    ),
)
def test_uncertainty_and_rate_bounds_accept_exact_boundary_and_reject_below(
    field: str,
    match: str,
):
    seed = _evidence()
    current = _later_evidence(seed)
    exact: int | float
    below: int | float
    if field == "max_total_timing_uncertainty_ns":
        exact = (
            seed.combined_timing_uncertainty_ns
            + current.combined_timing_uncertainty_ns
        )
        below = exact - 1
    elif field == "max_relative_angular_uncertainty_rad":
        exact = (
            seed.combined_angular_uncertainty_rad
            + current.combined_angular_uncertainty_rad
        )
        below = math.nextafter(exact, 0.0)
    elif field == "max_angular_rate_rad_s":
        exact = math.sqrt(
            sum(rate * rate for rate in current.capture_attitude.attitude.body_rates_rad_s)
        )
        below = math.nextafter(exact, 0.0)
    else:
        exact = current.model.angular_rate_uncertainty_rad_s
        below = math.nextafter(exact, 0.0)

    at_boundary = _reference(seed, model=_stable_model(**{field: exact}))
    camera_to_stable_local_differential(
        at_boundary,
        current,
        _state(at_boundary, current),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    below_boundary = _reference(seed, model=_stable_model(**{field: below}))
    with pytest.raises(VQ2StableReferenceError, match=match):
        camera_to_stable_local_differential(
            below_boundary,
            current,
            _state(below_boundary, current),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_exact_attitude_reuse_is_allowed_but_same_sample_relabel_is_rejected():
    seed = _evidence()
    reference = _reference(seed)
    # Keep the reused sample exactly on the derotation model's capture-alignment
    # boundary; a later frame would correctly be rejected by that lower layer.
    reused = _later_evidence(
        seed,
        time_delta_ns=19_000_000,
        capture=seed.capture_attitude,
    )
    camera_to_stable_local_differential(
        reference,
        reused,
        _state(reference, reused),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    relabelled_capture = dataclasses.replace(
        seed.capture_attitude,
        orientation_uncertainty_rad=seed.capture_attitude.orientation_uncertainty_rad + 1e-6,
    )
    relabelled = _later_evidence(
        seed,
        time_delta_ns=19_000_000,
        capture=relabelled_capture,
    )
    with pytest.raises(VQ2StableReferenceChronologyError, match="sample identity"):
        camera_to_stable_local_differential(
            reference,
            relabelled,
            _state(reference, relabelled),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_target_attitude_exact_reuse_is_allowed_but_same_sample_relabel_is_rejected():
    seed = _evidence()
    reference = _reference(seed)
    reused = _later_evidence(
        seed,
        time_delta_ns=14_000_000,
        capture=seed.capture_attitude,
        target_attitude=seed.target_attitude,
    )
    camera_to_stable_local_differential(
        reference,
        reused,
        _state(reference, reused, camera_time=VQ2CameraFeatureTime.TARGET),
        camera_time=VQ2CameraFeatureTime.TARGET,
    )
    relabelled_target = dataclasses.replace(
        seed.target_attitude,
        orientation_uncertainty_rad=seed.target_attitude.orientation_uncertainty_rad + 1e-6,
    )
    relabelled = _later_evidence(
        seed,
        time_delta_ns=14_000_000,
        capture=seed.capture_attitude,
        target_attitude=relabelled_target,
    )
    with pytest.raises(VQ2StableReferenceChronologyError, match="sample identity"):
        camera_to_stable_local_differential(
            reference,
            relabelled,
            _state(reference, relabelled, camera_time=VQ2CameraFeatureTime.TARGET),
            camera_time=VQ2CameraFeatureTime.TARGET,
        )


def test_seed_frame_cannot_be_retargeted_with_another_evidence_context():
    seed = _evidence()
    reference = _reference(seed)
    retargeted = _evidence(
        observation=seed.observation,
        capture=seed.capture_attitude,
        target_orientation=_quaternion_from_euler(yaw=0.1),
    )
    with pytest.raises(VQ2StableReferenceChronologyError, match="evidence context"):
        camera_to_stable_local_differential(
            reference,
            retargeted,
            _state(reference, retargeted, camera_time=VQ2CameraFeatureTime.TARGET),
            camera_time=VQ2CameraFeatureTime.TARGET,
        )


@pytest.mark.parametrize("regression", ("sample", "source", "receipt"))
def test_capture_imu_sample_source_and_receipt_must_advance_coherently(
    regression: str,
):
    seed = _evidence()
    reference = _reference(seed)
    previous = seed.capture_attitude.attitude
    time_delta_ns = 19_000_000
    capture = _attitude(
        source=previous.source,
        sample_sequence=(
            previous.sample_sequence
            if regression == "sample"
            else previous.sample_sequence + 2
        ),
        source_time_us=(
            previous.source_time_us
            if regression == "source"
            else previous.source_time_us + 20_000
        ),
        receive_monotonic_ns=(
            previous.receive_monotonic_ns
            if regression == "receipt"
            else _BASE_NS + time_delta_ns - 1_000_000
        ),
        orientation=_quaternion_from_euler(yaw=0.05),
    )
    current = _later_evidence(
        seed,
        time_delta_ns=time_delta_ns,
        capture=capture,
    )
    with pytest.raises(
        VQ2StableReferenceChronologyError,
        match="sample identity|chronology did not advance coherently",
    ):
        camera_to_stable_local_differential(
            reference,
            current,
            _state(reference, current),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_measurement_sequence_accepts_seed_and_new_frames_and_rejects_reuse_or_reorder():
    seed = _evidence()
    reference = _reference(seed)
    second = _later_evidence(seed)
    third = _later_evidence(seed, frame_delta=2, time_delta_ns=40_000_000)
    transforms = tuple(
        camera_to_stable_local_differential(
            reference,
            evidence,
            _state(reference, evidence),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )
        for evidence in (seed, second, third)
    )
    validate_stable_measurement_sequence(reference, transforms)
    with pytest.raises(VQ2StableReferenceChronologyError, match="repeats"):
        validate_stable_measurement_sequence(reference, (transforms[0], transforms[0]))
    with pytest.raises(VQ2StableReferenceChronologyError, match="advance strictly"):
        validate_stable_measurement_sequence(reference, (transforms[2], transforms[1]))


def test_measurement_sequence_rejects_target_or_inverse_transform():
    seed = _evidence()
    reference = _reference(seed)
    target_state = _state(reference, seed, camera_time=VQ2CameraFeatureTime.TARGET)
    target_transform = camera_to_stable_local_differential(
        reference,
        seed,
        target_state,
        camera_time=VQ2CameraFeatureTime.TARGET,
    )
    with pytest.raises(VQ2StableReferenceError, match="capture camera-to-stable"):
        validate_stable_measurement_sequence(reference, (target_transform,))
    inverse_state = dataclasses.replace(
        target_state,
        basis=VQ2LocalFeatureBasis.STABLE_REFERENCE,
        basis_id=reference.basis_id,
    )
    inverse = stable_to_camera_local_differential(
        reference,
        seed,
        inverse_state,
        camera_time=VQ2CameraFeatureTime.TARGET,
    )
    with pytest.raises(VQ2StableReferenceError, match="capture camera-to-stable"):
        validate_stable_measurement_sequence(reference, (inverse,))


def test_nonforward_or_out_of_bound_projective_geometry_is_rejected():
    seed = _evidence()
    reference = _reference(seed)
    near_horizon = _later_evidence(
        seed,
        capture_orientation=_quaternion_from_euler(yaw=math.radians(89.9)),
        target_orientation=_quaternion_from_euler(yaw=math.radians(89.9)),
        capture_body_rates=(0.0, 0.0, 0.0),
        target_body_rates=(0.0, 0.0, 0.0),
    )
    with pytest.raises(VQ2StableReferenceGeometryError, match="forward|bounds"):
        camera_to_stable_local_differential(
            reference,
            near_horizon,
            _state(
                reference,
                near_horizon,
                values=(0.0, 0.0, -0.7, 0.0, 0.0, 0.0),
            ),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_minimum_forward_guard_is_distinct_from_output_bounds():
    seed = _evidence()
    reference = _reference(seed)
    sideways = _later_evidence(
        seed,
        capture_orientation=_quaternion_from_euler(yaw=math.pi / 2.0),
        target_orientation=_quaternion_from_euler(yaw=math.pi / 2.0),
        capture_body_rates=(0.0, 0.0, 0.0),
        target_body_rates=(0.0, 0.0, 0.0),
    )
    with pytest.raises(VQ2StableReferenceGeometryError, match="forward margin"):
        camera_to_stable_local_differential(
            reference,
            sideways,
            _state(
                reference,
                sideways,
                values=(0.0, 0.0, -0.7, 0.0, 0.0, 0.0),
            ),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


@pytest.mark.parametrize(
    ("model_override", "match"),
    (
        ({"max_projective_magnification": 1.1}, "magnification"),
        ({"max_feature_jacobian_condition": 1.05}, "Jacobian condition"),
    ),
)
def test_projective_magnification_and_feature_condition_have_independent_guards(
    model_override: dict[str, object],
    match: str,
):
    seed = _evidence()
    reference = _reference(seed, model=_stable_model(**model_override))
    current = _later_evidence(
        seed,
        capture_orientation=_quaternion_from_euler(yaw=0.2),
        target_orientation=_quaternion_from_euler(yaw=0.2),
        capture_body_rates=(0.0, 0.0, 0.0),
        target_body_rates=(0.0, 0.0, 0.0),
    )
    with pytest.raises(VQ2StableReferenceGeometryError, match=match):
        camera_to_stable_local_differential(
            reference,
            current,
            _state(
                reference,
                current,
                values=(0.5, 0.0, -0.7, 0.0, 0.0, 0.0),
            ),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_stable_model_input_bounds_are_enforced_below_hard_state_bounds():
    seed = _evidence()
    reference = _reference(seed, model=_stable_model(max_abs_bearing_norm=0.25))
    with pytest.raises(VQ2StableReferenceError, match="input feature"):
        camera_to_stable_local_differential(
            reference,
            seed,
            _state(
                reference,
                seed,
                values=(0.3, 0.0, -0.7, 0.0, 0.0, 0.0),
            ),
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("chart_to_camera_ray", ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.0))),
        ("max_reference_age_ns", HARD_MAX_REFERENCE_AGE_NS + 1),
        ("max_angular_acceleration_rad_s2", 0.0),
        ("max_feature_acceleration_norm_s2", 0.0),
        ("covariance_psd_tolerance", 0.0),
        ("max_abs_bearing_norm", 4.1),
    ),
)
def test_stable_model_has_no_relaxable_or_uncalibrated_defaults(field: str, value: object):
    with pytest.raises((TypeError, ValueError)):
        _stable_model(**{field: value})


def test_stable_model_covariance_uses_its_explicit_relative_psd_tolerance():
    almost_psd = np.eye(6)
    almost_psd[0, 1] = almost_psd[1, 0] = 1.0 + 1e-11
    covariance = tuple(
        tuple(float(value) for value in row) for row in almost_psd
    )
    with pytest.raises(VQ2StableReferenceError, match="positive semidefinite"):
        _stable_model(
            covariance_psd_tolerance=1e-12,
            joint_nuisance_envelope_covariance=covariance,
        )


def test_local_state_rejects_bool_nonfinite_wrong_shape_and_indefinite_covariance():
    seed = _evidence()
    reference = _reference(seed)
    with pytest.raises(TypeError, match="numeric and not bool"):
        _state(reference, seed, values=(True, 0.0, 0.0, 0.0, 0.0, 0.0))
    with pytest.raises(VQ2StableReferenceError, match="finite"):
        _state(reference, seed, values=(math.nan, 0.0, 0.0, 0.0, 0.0, 0.0))
    with pytest.raises(TypeError, match="6x6"):
        _state(reference, seed, covariance=((1.0,),))
    indefinite = np.eye(6)
    indefinite[0, 1] = indefinite[1, 0] = 2.0
    with pytest.raises(VQ2StableReferenceError, match="positive semidefinite"):
        _state(
            reference,
            seed,
            covariance=tuple(tuple(float(value) for value in row) for row in indefinite),
        )
    tiny_indefinite = np.eye(6) * 1e-12
    tiny_indefinite[0, 1] = tiny_indefinite[1, 0] = 2e-11
    with pytest.raises(VQ2StableReferenceError, match="positive semidefinite"):
        _state(
            reference,
            seed,
            covariance=tuple(
                tuple(float(value) for value in row) for row in tiny_indefinite
            ),
        )


def test_transform_result_is_deterministic_immutable_and_detects_forged_derived_fields():
    seed = _evidence()
    reference = _reference(seed)
    current = _later_evidence(seed)
    state = _state(reference, current)
    first = camera_to_stable_local_differential(
        reference,
        current,
        state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    second = camera_to_stable_local_differential(
        reference,
        current,
        state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert first == second
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.projective_forward = 2.0  # type: ignore[misc]
    object.__setattr__(first, "projective_forward", first.projective_forward + 0.01)
    with pytest.raises(VQ2StableReferenceError, match="projective_forward"):
        first.validate_integrity()


def test_transform_detects_forged_nested_output_and_covariance():
    seed = _evidence()
    reference = _reference(seed)
    state = _state(reference, seed)
    result = camera_to_stable_local_differential(
        reference,
        seed,
        state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    forged_output = dataclasses.replace(
        result.output_state,
        values=tuple(
            value + (0.01 if index == 0 else 0.0)
            for index, value in enumerate(result.output_state.values)
        ),
    )
    object.__setattr__(result, "output_state", forged_output)
    with pytest.raises(VQ2StableReferenceError, match="output_state"):
        result.validate_integrity()

    for field in (
        "coordinate_covariance",
        "joint_nuisance_envelope_covariance",
        "model_floor_covariance",
        "total_covariance",
    ):
        fresh = camera_to_stable_local_differential(
            reference,
            seed,
            state,
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )
        forged = np.asarray(getattr(fresh, field), dtype=np.float64).copy()
        forged[0, 0] += 1e-5
        object.__setattr__(
            fresh,
            field,
            tuple(tuple(float(value) for value in row) for row in forged),
        )
        with pytest.raises(VQ2StableReferenceError, match=field):
            fresh.validate_integrity()

    fresh = camera_to_stable_local_differential(
        reference,
        seed,
        state,
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    forged_output_covariance = np.asarray(
        fresh.output_state.covariance,
        dtype=np.float64,
    ).copy()
    forged_output_covariance[0, 0] += 1e-5
    object.__setattr__(
        fresh,
        "output_state",
        dataclasses.replace(
            fresh.output_state,
            covariance=tuple(
                tuple(float(value) for value in row)
                for row in forged_output_covariance
            ),
        ),
    )
    with pytest.raises(VQ2StableReferenceError, match="output_state"):
        fresh.validate_integrity()


@pytest.mark.parametrize("nested_target", ("current_timing", "seed_authority"))
def test_transform_integrity_revalidates_nested_public_observation_contracts(
    nested_target: str,
):
    seed = _evidence()
    reference = _reference(seed)
    current = _later_evidence(seed)
    result = camera_to_stable_local_differential(
        reference,
        current,
        _state(reference, current),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    if nested_target == "current_timing":
        object.__setattr__(
            result.evidence.observation.frame_timing,
            "camera_source_time_ns",
            False,
        )
    else:
        object.__setattr__(
            result.reference.seed_evidence.observation.authority,
            "race_status_boot_ms",
            1.5,
        )
    with pytest.raises((TypeError, ValueError)):
        result.validate_integrity()


def test_quaternion_sign_equivalence_produces_identical_chart_transform():
    seed = _evidence()
    reference = _reference(seed)
    orientation = _quaternion_from_euler(roll=0.1, pitch=-0.08, yaw=0.2)
    positive = _later_evidence(
        seed,
        capture_orientation=orientation,
        target_orientation=orientation,
        capture_body_rates=(0.0, 0.0, 0.0),
        target_body_rates=(0.0, 0.0, 0.0),
    )
    negative_orientation = tuple(-value for value in orientation)
    negative = _later_evidence(
        seed,
        capture_orientation=negative_orientation,  # type: ignore[arg-type]
        target_orientation=negative_orientation,  # type: ignore[arg-type]
        capture_body_rates=(0.0, 0.0, 0.0),
        target_body_rates=(0.0, 0.0, 0.0),
    )
    positive_result = camera_to_stable_local_differential(
        reference,
        positive,
        _state(reference, positive),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    negative_result = camera_to_stable_local_differential(
        reference,
        negative,
        _state(reference, negative),
        camera_time=VQ2CameraFeatureTime.CAPTURE,
    )
    assert negative_result.output_state.values == pytest.approx(
        positive_result.output_state.values,
        abs=1e-12,
    )
    assert np.asarray(negative_result.state_jacobian) == pytest.approx(
        np.asarray(positive_result.state_jacobian),
        abs=1e-12,
    )


def test_public_module_has_no_private_derotation_import_or_authority_runtime_dependency():
    module_path = Path(__file__).parents[1] / "vq2_stable_reference.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not any(name.startswith("_") for name in imported_names)
    source = module_path.read_text(encoding="utf-8")
    for forbidden in (
        "RelativeGateStateV1",
        "CommandProposalV1",
        "SupervisorApprovedCommandV1",
        "aigp_mavlink",
        "scripts.aigp_vq2_run",
    ):
        assert forbidden not in source


def test_no_production_module_imports_the_standalone_stable_reference_module():
    repository = Path(__file__).parents[2]
    offenders: list[str] = []
    excluded_parts = {
        ".loop",
        ".research_loop",
        ".pytest_cache",
        "__pycache__",
        "docs",
        "tests",
    }
    for path in repository.rglob("*.py"):
        relative = path.relative_to(repository)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if path.name == "vq2_stable_reference.py":
            continue
        if "vq2_stable_reference" in path.read_text(encoding="utf-8"):
            offenders.append(str(relative))
    assert offenders == []


def test_exact_public_types_reject_v1_observation_as_local_feature_input():
    seed = _evidence()
    reference = _reference(seed)
    with pytest.raises(TypeError, match="state must be exact"):
        camera_to_stable_local_differential(
            reference,
            seed,
            seed.observation,  # type: ignore[arg-type]
            camera_time=VQ2CameraFeatureTime.CAPTURE,
        )


def test_public_enums_sequences_and_integer_bounds_require_exact_types():
    seed = _evidence()
    reference = _reference(seed)
    state = _state(reference, seed)
    with pytest.raises(TypeError, match="camera_time"):
        camera_to_stable_local_differential(
            reference,
            seed,
            state,
            camera_time="capture",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="exact tuple"):
        validate_stable_measurement_sequence(reference, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="basis must be exact"):
        dataclasses.replace(state, basis="camera")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="covariance_scope must be exact"):
        dataclasses.replace(
            state,
            covariance_scope="conditional_input",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="exact integer"):
        _stable_model(max_reference_age_ns=True)
