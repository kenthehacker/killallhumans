from __future__ import annotations

import dataclasses
import math

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
    MeasurementTimeBasis,
    ObservationHealth,
    PredictionBasis,
)
from estimation.vq2_imu_derotation import (
    HARD_MAX_CAPTURE_ALIGNMENT_NS,
    HARD_MAX_DEROTATION_INTERVAL_NS,
    HARD_MAX_TARGET_EXTRAPOLATION_NS,
    SUPPORTED_CAMERA_RAY_MODEL_ID,
    VQ2AttitudeDerotationInput,
    VQ2CameraToBodyCalibration,
    VQ2DerotationEvidence,
    VQ2DerotationModel,
    VQ2ImuDerotationError,
    derotate_gate_observation,
)
from estimation.vq2_imu_provenance import VQ2ImuSource, VQ2TimestampedAttitude
from estimation.vq2_relative_estimator import RelativePredictionTarget


_BASE_NS = 10_000_000_000
_HOST_CLOCK_ID = "host-monotonic-test"
_SESSION_ID = "training-session-test"


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


def _source(
    *,
    session_id: str = _SESSION_ID,
    reset_epoch: int = 7,
    host_clock_id: str = _HOST_CLOCK_ID,
    stream_id: str = "highres-imu0",
    generation: int = 3,
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
) -> VQ2AttitudeDerotationInput:
    attitude = VQ2TimestampedAttitude(
        source=source or _source(),
        sample_sequence=sample_sequence,
        source_time_us=source_time_us,
        receive_monotonic_ns=receive_monotonic_ns,
        orientation_body_to_ned_wxyz=orientation,
        body_rates_rad_s=body_rates,
        gyro_bias_rad_s=(0.0, 0.0, 0.0),
        accel_trust=1.0,
        propagated=True,
    )
    return VQ2AttitudeDerotationInput(
        attitude=attitude,
        orientation_uncertainty_rad=0.001,
        host_time_uncertainty_ns=100_000,
    )


def _observation(
    *,
    center: tuple[float, float] = (0.2, -0.1),
    measurement_ns: int = _BASE_NS,
    host_clock_id: str = _HOST_CLOCK_ID,
    session_id: str = _SESSION_ID,
    reset_epoch: int = 7,
    frame_id: int = 41,
    candidate_id: str = "gate-candidate-41",
    gate_epoch: int = 2,
    gate_index: int = 0,
    health: ObservationHealth = ObservationHealth.DEGRADED,
    covariance: FeatureCovarianceV1 | None = None,
) -> GateObservationV1:
    frame = FrameIdentityV1("camera0", 4, frame_id)
    timing = FrameTimingV1(
        identity=frame,
        camera_source_time_ns=999_999_999_999_999,
        host_clock_id=host_clock_id,
        publication_sequence=frame_id + 1,
        first_unique_packet_monotonic_ns=measurement_ns - 1_000,
        final_unique_packet_monotonic_ns=measurement_ns,
        reassembly_complete_monotonic_ns=measurement_ns,
        decode_start_monotonic_ns=measurement_ns + 100_000,
        decode_end_monotonic_ns=measurement_ns + 200_000,
        publish_monotonic_ns=measurement_ns + 300_000,
    )
    authority = GateAuthorityEpochV1(
        session_id=session_id,
        reset_epoch=reset_epoch,
        gate_epoch=gate_epoch,
        expected_gate_index=gate_index,
        race_status_sequence=12,
        race_status_boot_ms=2_000,
        camera_host_clock_id=host_clock_id,
        camera_stream_id="camera0",
        camera_generation=4,
        frame_publication_sequence_not_before=1,
        frame_publish_monotonic_ns_not_before=measurement_ns - 1_000,
    )
    covariance = covariance or FeatureCovarianceV1(
        "synthetic-center-covariance-v1",
        ("center_x_norm", "center_y_norm"),
        ((4e-4, 1e-4), (1e-4, 9e-4)),
    )
    return GateObservationV1(
        frame_timing=timing,
        measurement_time_monotonic_ns=measurement_ns,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        measurement_time_model_id=None,
        measurement_uncertainty_ns=1_000_000,
        authority=authority,
        candidate_id=candidate_id,
        image_size_px=(640, 360),
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
        covariance=covariance,
        fit=FitDiagnosticsV1(None, 0, 0),
        health=health,
        health_reason="center-only-test-observation",
        provenance="synthetic-test",
    )


def _model(**overrides: object) -> VQ2DerotationModel:
    values: dict[str, object] = {
        "model_id": "vq2-rotation-only-derotation-v1",
        "attitude_time_model_id": "imu-receive-host-proxy-v1",
        "max_capture_alignment_ns": HARD_MAX_CAPTURE_ALIGNMENT_NS,
        "max_target_extrapolation_ns": HARD_MAX_TARGET_EXTRAPOLATION_NS,
        "max_total_timing_uncertainty_ns": 10_000_000,
        "angular_rate_uncertainty_rad_s": 0.02,
    }
    values.update(overrides)
    return VQ2DerotationModel(**values)  # type: ignore[arg-type]


def _calibration(
    *,
    camera_to_body: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> VQ2CameraToBodyCalibration:
    return VQ2CameraToBodyCalibration(
        calibration_id="camera0-to-body-calibration-20260719",
        camera_ray_model_id=SUPPORTED_CAMERA_RAY_MODEL_ID,
        camera_to_body_wxyz=camera_to_body,
        rotation_uncertainty_rad=0.001,
    )


def _target(
    observation: GateObservationV1,
    *,
    decision_ns: int = _BASE_NS + 10_000_000,
    prediction_ns: int | None = None,
) -> RelativePredictionTarget:
    prediction_ns = decision_ns if prediction_ns is None else prediction_ns
    if prediction_ns == decision_ns:
        return RelativePredictionTarget.at_decision(observation.host_clock_id, decision_ns)
    return RelativePredictionTarget(
        host_clock_id=observation.host_clock_id,
        decision_time_monotonic_ns=decision_ns,
        prediction_time_monotonic_ns=prediction_ns,
        prediction_basis=PredictionBasis.COMMAND_EFFECT_ESTIMATE,
        delay_model_id="command-effect-test-v1",
        delay_uncertainty_ns=500_000,
    )


def _inputs(
    *,
    capture_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    target_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    capture_receive_ns: int = _BASE_NS,
    target_receive_ns: int = _BASE_NS + 5_000_000,
    capture_source_time_us: int = 9_000_000_000_000,
    target_source_time_us: int = 9_000_000_010_000,
    capture_source: VQ2ImuSource | None = None,
    target_source: VQ2ImuSource | None = None,
    capture_sequence: int = 100,
    target_sequence: int = 101,
    target_body_rates: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[VQ2AttitudeDerotationInput, VQ2AttitudeDerotationInput]:
    source = capture_source or _source()
    return (
        _attitude(
            source=source,
            sample_sequence=capture_sequence,
            source_time_us=capture_source_time_us,
            receive_monotonic_ns=capture_receive_ns,
            orientation=capture_orientation,
        ),
        _attitude(
            source=target_source or source,
            sample_sequence=target_sequence,
            source_time_us=target_source_time_us,
            receive_monotonic_ns=target_receive_ns,
            orientation=target_orientation,
            body_rates=target_body_rates,
        ),
    )


def _derotate(
    observation: GateObservationV1 | None = None,
    target: RelativePredictionTarget | None = None,
    *,
    capture: VQ2AttitudeDerotationInput | None = None,
    target_attitude: VQ2AttitudeDerotationInput | None = None,
    calibration: VQ2CameraToBodyCalibration | None = None,
    model: VQ2DerotationModel | None = None,
) -> VQ2DerotationEvidence:
    observation = observation or _observation()
    if capture is None or target_attitude is None:
        default_capture, default_target = _inputs()
        capture = capture or default_capture
        target_attitude = target_attitude or default_target
    return derotate_gate_observation(
        observation,
        target or _target(observation),
        capture_attitude=capture,
        target_attitude=target_attitude,
        calibration=calibration or _calibration(),
        model=model or _model(),
    )


def _minimum_eigenvalue(matrix: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return 0.5 * (
        matrix[0][0]
        + matrix[1][1]
        - math.hypot(matrix[0][0] - matrix[1][1], 2.0 * matrix[0][1])
    )


def test_identity_rotation_preserves_center_and_binds_complete_sources():
    observation = _observation(center=(0.25, -0.15), candidate_id="candidate-exact")
    target = _target(observation)
    capture, target_attitude = _inputs()

    evidence = _derotate(observation, target, capture=capture, target_attitude=target_attitude)

    assert type(evidence) is VQ2DerotationEvidence
    assert evidence.observation is observation
    assert evidence.prediction_target is target
    assert evidence.observation.frame == observation.frame
    assert evidence.observation.candidate_id == "candidate-exact"
    assert evidence.observation.authority.gate_epoch == 2
    assert evidence.observation.authority.expected_gate_index == 0
    assert evidence.capture_attitude is capture
    assert evidence.target_attitude is target_attitude
    assert evidence.input_center_norm == (0.25, -0.15)
    assert evidence.derotated_center_norm == pytest.approx((0.25, -0.15), abs=1e-12)
    assert evidence.capture_alignment_ns == 0
    assert evidence.target_extrapolation_ns == 5_000_000
    # The enormous opaque source stamp has no arithmetic relationship to host time.
    assert evidence.capture_attitude.attitude.source_time_us == 9_000_000_000_000


def test_positive_body_yaw_moves_stationary_world_ray_left_in_target_camera():
    yaw = math.radians(10.0)
    observation = _observation(center=(0.0, 0.0))
    capture, target_attitude = _inputs(target_orientation=_quaternion_from_euler(yaw=yaw))

    evidence = _derotate(observation, capture=capture, target_attitude=target_attitude)

    assert evidence.derotated_center_norm[0] == pytest.approx(-math.tan(yaw), abs=1e-12)
    assert evidence.derotated_center_norm[1] == pytest.approx(0.0, abs=1e-12)


def test_positive_nose_up_body_pitch_moves_stationary_world_ray_down_in_camera():
    pitch = math.radians(10.0)
    observation = _observation(center=(0.0, 0.0))
    capture, target_attitude = _inputs(
        target_orientation=_quaternion_from_euler(pitch=pitch)
    )

    evidence = _derotate(
        observation,
        capture=capture,
        target_attitude=target_attitude,
    )

    assert evidence.derotated_center_norm[0] == pytest.approx(0.0, abs=1e-12)
    assert evidence.derotated_center_norm[1] == pytest.approx(math.tan(pitch), abs=1e-12)


def test_explicit_roll_extrinsic_rotates_yaw_correction_into_camera_vertical_axis():
    yaw = math.radians(10.0)
    observation = _observation(center=(0.0, 0.0))
    capture, target_attitude = _inputs(target_orientation=_quaternion_from_euler(yaw=yaw))
    calibration = _calibration(camera_to_body=_quaternion_from_euler(roll=math.pi / 2.0))

    evidence = _derotate(
        observation,
        capture=capture,
        target_attitude=target_attitude,
        calibration=calibration,
    )

    assert evidence.derotated_center_norm[0] == pytest.approx(0.0, abs=1e-12)
    assert evidence.derotated_center_norm[1] == pytest.approx(math.tan(yaw), abs=1e-12)


def test_rotation_roundtrip_recovers_original_normalized_ray():
    first_observation = _observation(center=(0.31, -0.22))
    intermediate_orientation = _quaternion_from_euler(
        roll=math.radians(7.0),
        pitch=math.radians(-6.0),
        yaw=math.radians(8.0),
    )
    first_capture, first_target = _inputs(target_orientation=intermediate_orientation)
    calibration = _calibration(camera_to_body=_quaternion_from_euler(roll=0.2, pitch=-0.1))
    forward = _derotate(
        first_observation,
        capture=first_capture,
        target_attitude=first_target,
        calibration=calibration,
    )

    second_measurement_ns = _BASE_NS + 30_000_000
    second_observation = _observation(
        center=forward.derotated_center_norm,
        measurement_ns=second_measurement_ns,
        frame_id=42,
        candidate_id="gate-candidate-42",
    )
    reverse_capture, reverse_target = _inputs(
        capture_orientation=intermediate_orientation,
        target_orientation=(1.0, 0.0, 0.0, 0.0),
        capture_receive_ns=second_measurement_ns,
        target_receive_ns=second_measurement_ns + 5_000_000,
        capture_source_time_us=9_000_000_020_000,
        target_source_time_us=9_000_000_030_000,
        capture_sequence=102,
        target_sequence=103,
    )
    reverse = _derotate(
        second_observation,
        _target(second_observation, decision_ns=second_measurement_ns + 10_000_000),
        capture=reverse_capture,
        target_attitude=reverse_target,
        calibration=calibration,
    )

    assert reverse.derotated_center_norm == pytest.approx(first_observation.center_norm, abs=1e-12)


def test_target_body_rate_is_extrapolated_only_on_host_time():
    observation = _observation(center=(0.0, 0.0))
    capture, target_attitude = _inputs(target_body_rates=(0.0, 0.0, 1.0))
    target = _target(observation, decision_ns=_BASE_NS + 15_000_000)

    evidence = _derotate(
        observation,
        target,
        capture=capture,
        target_attitude=target_attitude,
    )

    assert evidence.target_extrapolation_ns == 10_000_000
    assert evidence.derotated_center_norm[0] == pytest.approx(-math.tan(0.01), abs=1e-12)
    assert evidence.effective_target_orientation_body_to_ned_wxyz == pytest.approx(
        (math.cos(0.005), 0.0, 0.0, math.sin(0.005)),
        abs=1e-12,
    )


def test_derotation_covariance_is_psd_and_never_tighter_than_input():
    evidence = _derotate()
    original = evidence.input_center_covariance_norm2
    output = evidence.derotated_center_covariance_norm2
    increment = (
        (output[0][0] - original[0][0], output[0][1] - original[0][1]),
        (output[1][0] - original[1][0], output[1][1] - original[1][1]),
    )

    assert _minimum_eigenvalue(output) >= -1e-15
    assert _minimum_eigenvalue(increment) >= -1e-15
    assert output[0][0] > original[0][0]
    assert output[1][1] > original[1][1]


def test_camera_timing_uncertainty_inflates_angular_and_bearing_uncertainty():
    observation = _observation()
    capture, target_attitude = _inputs()
    capture = dataclasses.replace(
        capture,
        attitude=dataclasses.replace(capture.attitude, body_rates_rad_s=(1.0, 0.0, 0.0)),
    )
    baseline = _derotate(
        observation,
        capture=capture,
        target_attitude=target_attitude,
    )
    less_certain_observation = dataclasses.replace(
        observation,
        measurement_uncertainty_ns=2_000_000,
    )
    inflated = _derotate(
        less_certain_observation,
        capture=capture,
        target_attitude=target_attitude,
    )

    assert inflated.combined_angular_uncertainty_rad > baseline.combined_angular_uncertainty_rad
    assert (
        inflated.derotated_center_covariance_norm2[0][0]
        > baseline.derotated_center_covariance_norm2[0][0]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera_ray_model_id", "unknown-camera-model-v1"),
        ("camera_to_body_wxyz", (2.0, 0.0, 0.0, 0.0)),
        ("rotation_uncertainty_rad", 0.0),
    ],
)
def test_calibration_is_explicit_supported_and_uncertain(field: str, value: object):
    values: dict[str, object] = {
        "calibration_id": "camera0-to-body-calibration-20260719",
        "camera_ray_model_id": SUPPORTED_CAMERA_RAY_MODEL_ID,
        "camera_to_body_wxyz": (1.0, 0.0, 0.0, 0.0),
        "rotation_uncertainty_rad": 0.001,
    }
    values[field] = value
    with pytest.raises((TypeError, VQ2ImuDerotationError)):
        VQ2CameraToBodyCalibration(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_capture_alignment_ns", HARD_MAX_CAPTURE_ALIGNMENT_NS + 1),
        ("max_target_extrapolation_ns", HARD_MAX_TARGET_EXTRAPOLATION_NS + 1),
        ("max_total_timing_uncertainty_ns", 50_000_001),
        ("angular_rate_uncertainty_rad_s", 0.0),
    ],
)
def test_model_cannot_relax_hard_safety_bounds(field: str, value: object):
    with pytest.raises((TypeError, VQ2ImuDerotationError)):
        _model(**{field: value})


def test_exact_target_extrapolation_boundary_is_accepted_and_one_ns_more_rejected():
    observation = _observation()
    capture, target_attitude = _inputs()
    at_bound = _target(
        observation,
        decision_ns=target_attitude.attitude.receive_monotonic_ns + HARD_MAX_TARGET_EXTRAPOLATION_NS,
    )
    evidence = _derotate(
        observation,
        at_bound,
        capture=capture,
        target_attitude=target_attitude,
    )
    assert evidence.target_extrapolation_ns == HARD_MAX_TARGET_EXTRAPOLATION_NS

    beyond = _target(
        observation,
        decision_ns=(
            target_attitude.attitude.receive_monotonic_ns
            + HARD_MAX_TARGET_EXTRAPOLATION_NS
            + 1
        ),
    )
    with pytest.raises(VQ2ImuDerotationError, match="target attitude extrapolation"):
        _derotate(observation, beyond, capture=capture, target_attitude=target_attitude)


def test_camera_to_target_interval_has_independent_hard_bound():
    observation = _observation()
    capture, target_attitude = _inputs(
        target_receive_ns=_BASE_NS + HARD_MAX_DEROTATION_INTERVAL_NS,
        target_source_time_us=9_000_000_100_000,
    )
    target = _target(
        observation,
        decision_ns=_BASE_NS + HARD_MAX_DEROTATION_INTERVAL_NS,
    )
    evidence = _derotate(
        observation,
        target,
        capture=capture,
        target_attitude=target_attitude,
    )
    assert (
        evidence.prediction_target.prediction_time_monotonic_ns
        - evidence.observation.measurement_time_monotonic_ns
        == HARD_MAX_DEROTATION_INTERVAL_NS
    )

    later_target = _target(
        observation,
        decision_ns=_BASE_NS + HARD_MAX_DEROTATION_INTERVAL_NS + 1,
    )
    target_attitude = dataclasses.replace(
        target_attitude,
        attitude=dataclasses.replace(
            target_attitude.attitude,
            receive_monotonic_ns=_BASE_NS + HARD_MAX_DEROTATION_INTERVAL_NS + 1,
        ),
    )
    with pytest.raises(VQ2ImuDerotationError, match="derotation interval"):
        _derotate(
            observation,
            later_target,
            capture=capture,
            target_attitude=target_attitude,
        )


def test_exact_capture_alignment_boundary_is_accepted_and_one_ns_more_rejected():
    observation = _observation()
    capture_at_bound, target_attitude = _inputs(
        capture_receive_ns=_BASE_NS - HARD_MAX_CAPTURE_ALIGNMENT_NS,
    )
    evidence = _derotate(
        observation,
        capture=capture_at_bound,
        target_attitude=target_attitude,
    )
    assert evidence.capture_alignment_ns == HARD_MAX_CAPTURE_ALIGNMENT_NS

    capture_beyond, target_attitude = _inputs(
        capture_receive_ns=_BASE_NS - HARD_MAX_CAPTURE_ALIGNMENT_NS - 1,
    )
    with pytest.raises(VQ2ImuDerotationError, match="capture attitude alignment"):
        _derotate(observation, capture=capture_beyond, target_attitude=target_attitude)


@pytest.mark.parametrize(
    "source",
    [
        _source(session_id="another-session"),
        _source(reset_epoch=8),
        _source(host_clock_id="another-host-clock"),
    ],
)
def test_camera_authority_and_imu_lineage_must_share_session_reset_and_host_clock(
    source: VQ2ImuSource,
):
    capture, target_attitude = _inputs(capture_source=source, target_source=source)
    with pytest.raises(VQ2ImuDerotationError):
        _derotate(capture=capture, target_attitude=target_attitude)


@pytest.mark.parametrize(
    "target_source",
    [
        _source(stream_id="highres-imu1"),
        _source(generation=4),
    ],
)
def test_capture_and_target_require_one_imu_stream_generation(target_source: VQ2ImuSource):
    capture, target_attitude = _inputs(target_source=target_source)
    with pytest.raises(VQ2ImuDerotationError, match="IMU sources differ"):
        _derotate(capture=capture, target_attitude=target_attitude)


def test_sample_identity_and_each_time_axis_must_advance_together():
    cases = [
        _inputs(target_sequence=99),
        _inputs(target_source_time_us=9_000_000_000_000),
        _inputs(target_receive_ns=_BASE_NS),
    ]
    for capture, target_attitude in cases:
        with pytest.raises(VQ2ImuDerotationError):
            _derotate(capture=capture, target_attitude=target_attitude)

    capture, _ = _inputs()
    conflicting_same_identity = _attitude(
        sample_sequence=capture.attitude.sample_sequence,
        source_time_us=capture.attitude.source_time_us,
        receive_monotonic_ns=capture.attitude.receive_monotonic_ns,
        orientation=_quaternion_from_euler(yaw=0.1),
    )
    with pytest.raises(VQ2ImuDerotationError, match="conflicting attitudes"):
        _derotate(capture=capture, target_attitude=conflicting_same_identity)


def test_attitude_samples_must_be_causally_available_at_decision():
    observation = _observation()
    decision_ns = _BASE_NS + 10_000_000
    capture, target_attitude = _inputs(target_receive_ns=decision_ns + 1)
    with pytest.raises(VQ2ImuDerotationError, match="unavailable at decision"):
        _derotate(
            observation,
            _target(observation, decision_ns=decision_ns),
            capture=capture,
            target_attitude=target_attitude,
        )


def test_decision_must_follow_bound_frame_publication_and_host_clock_must_match():
    observation = _observation()
    capture, target_attitude = _inputs()
    before_publish = RelativePredictionTarget.at_decision(
        observation.host_clock_id,
        observation.frame_timing.publish_monotonic_ns - 1,
    )
    with pytest.raises(VQ2ImuDerotationError, match="decision predates"):
        _derotate(
            observation,
            before_publish,
            capture=capture,
            target_attitude=target_attitude,
        )

    wrong_clock = RelativePredictionTarget.at_decision(
        "another-host-clock",
        _BASE_NS + 10_000_000,
    )
    with pytest.raises(VQ2ImuDerotationError, match="host clocks differ"):
        _derotate(
            observation,
            wrong_clock,
            capture=capture,
            target_attitude=target_attitude,
        )


def test_explicit_combined_timing_uncertainty_is_bounded():
    capture, target_attitude = _inputs()
    capture = dataclasses.replace(capture, host_time_uncertainty_ns=5_000_000)
    target_attitude = dataclasses.replace(target_attitude, host_time_uncertainty_ns=5_000_000)
    with pytest.raises(VQ2ImuDerotationError, match="combined timing uncertainty"):
        _derotate(capture=capture, target_attitude=target_attitude)


@pytest.mark.parametrize("yaw", [math.radians(80.0), math.pi])
def test_nonforward_or_out_of_bounds_derotated_ray_is_rejected(yaw: float):
    observation = _observation(center=(0.0, 0.0))
    capture, target_attitude = _inputs(target_orientation=_quaternion_from_euler(yaw=yaw))
    with pytest.raises(VQ2ImuDerotationError, match="forward-facing|normalized bounds"):
        _derotate(observation, capture=capture, target_attitude=target_attitude)


def test_unusable_observation_and_contract_invalid_center_covariance_are_rejected():
    unusable = _observation(health=ObservationHealth.UNUSABLE)
    with pytest.raises(VQ2ImuDerotationError, match="unusable observation"):
        _derotate(unusable)

    # The frozen observation contract rejects this before derotation can see
    # it; the derotator independently retains the same fail-closed guard.
    with pytest.raises(ValueError, match="must cover both center features"):
        _observation(
            covariance=FeatureCovarianceV1(
                "unrelated-covariance-v1",
                ("log_scale",),
                ((0.01,),),
            )
        )


def test_attitude_health_and_calibration_are_construction_invariants():
    attitude = _inputs()[0].attitude
    with pytest.raises(ValueError, match="healthy and calibrated"):
        dataclasses.replace(attitude, healthy=False)
    with pytest.raises(ValueError, match="healthy and calibrated"):
        dataclasses.replace(attitude, calibrated=False)


def test_evidence_is_immutable_and_derived_fields_cannot_be_relabelled():
    evidence = _derotate()
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.derotated_center_norm = (0.0, 0.0)  # type: ignore[misc]
    with pytest.raises(VQ2ImuDerotationError, match="does not match its sources"):
        dataclasses.replace(evidence, target_extrapolation_ns=0)
    with pytest.raises(VQ2ImuDerotationError, match="does not match its sources"):
        dataclasses.replace(evidence, derotated_center_norm=(0.0, 0.0))


def test_function_has_no_uncalibrated_default():
    observation = _observation()
    capture, target_attitude = _inputs()
    with pytest.raises(TypeError):
        derotate_gate_observation(
            observation,
            _target(observation),
            capture_attitude=capture,
            target_attitude=target_attitude,
            model=_model(),
        )
