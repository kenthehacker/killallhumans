"""Pure, bounded IMU attitude derotation for VQ2 camera bearings.

This module is a local, offline seam.  It rotates the normalized camera ray
``(forward, right, down) = (1, center_x_norm, center_y_norm)`` from the camera
attitude at observation time into the camera attitude at a requested target
time.  It never estimates translation, range, metric pose, gate passage, or
control authority, and it does not alter any frozen ``/1`` contract.

All camera/IMU correlation uses the explicitly shared host-monotonic clock.
The HIGHRES_IMU source timestamp is retained only as an opaque ordering and
integrity token; it is never subtracted from a host timestamp.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from competition.vq2_contracts import GateObservationV1, ObservationHealth
from estimation.vq2_imu_provenance import VQ2TimestampedAttitude
from estimation.vq2_relative_estimator import RelativePredictionTarget


QuaternionTuple = tuple[float, float, float, float]
Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Matrix2 = tuple[tuple[float, float], tuple[float, float]]

SUPPORTED_CAMERA_RAY_MODEL_ID = "vq2-frd-normalized-pinhole-v1"
HARD_MAX_CAPTURE_ALIGNMENT_NS = 20_000_000
HARD_MAX_TARGET_EXTRAPOLATION_NS = 20_000_000
HARD_MAX_TOTAL_TIMING_UNCERTAINTY_NS = 50_000_000
HARD_MAX_DEROTATION_INTERVAL_NS = 100_000_000
MAX_ABS_NORMALIZED_BEARING = 4.0

_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_QUATERNION_NORM_TOLERANCE = 1e-9
_MIN_FORWARD_RAY = 1e-9
_EQUALITY_TOLERANCE = 1e-12


class VQ2ImuDerotationError(ValueError):
    """Raised when derotation provenance, timing, or geometry fails closed."""


def _bounded_token(value: object, label: str) -> str:
    if type(value) is not str or _TOKEN_RE.fullmatch(value) is None:
        raise VQ2ImuDerotationError(f"{label} must be a bounded token")
    return value


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise VQ2ImuDerotationError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise VQ2ImuDerotationError(f"{label} must be <= {maximum}")
    return value


def _finite_float(
    value: object,
    label: str,
    *,
    positive: bool = False,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise VQ2ImuDerotationError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise VQ2ImuDerotationError(f"{label} must be positive")
    if maximum is not None and result > maximum:
        raise VQ2ImuDerotationError(f"{label} must be <= {maximum}")
    return result


def _vector(
    value: object,
    length: int,
    label: str,
    *,
    bound: float | None = None,
) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise TypeError(f"{label} must be an exact {length}-tuple")
    result = tuple(_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value))
    if bound is not None and any(abs(item) > bound for item in result):
        raise VQ2ImuDerotationError(f"{label} must remain within +/-{bound}")
    return result


def _unit_quaternion(value: object, label: str) -> QuaternionTuple:
    q = _vector(value, 4, label)
    norm_sq = sum(item * item for item in q)
    if not math.isclose(norm_sq, 1.0, rel_tol=0.0, abs_tol=_QUATERNION_NORM_TOLERANCE):
        raise VQ2ImuDerotationError(f"{label} must be unit length")
    return q  # type: ignore[return-value]


def _matrix2(value: object, label: str) -> Matrix2:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{label} must be an exact 2x2 tuple")
    rows = tuple(_vector(row, 2, f"{label}[{index}]") for index, row in enumerate(value))
    matrix: Matrix2 = (rows[0], rows[1])  # type: ignore[assignment]
    scale = max(1.0, *(abs(item) for row in matrix for item in row))
    if abs(matrix[0][1] - matrix[1][0]) > 1e-10 * scale:
        raise VQ2ImuDerotationError(f"{label} must be symmetric")
    if matrix[0][0] <= 0.0 or matrix[1][1] <= 0.0:
        raise VQ2ImuDerotationError(f"{label} variances must be positive")
    if _minimum_eigenvalue_2x2(matrix) < -1e-10 * scale:
        raise VQ2ImuDerotationError(f"{label} must be positive semidefinite")
    return matrix


@dataclass(frozen=True, slots=True)
class VQ2CameraToBodyCalibration:
    """Explicit camera-ray model and camera-FRD to body-FRD rotation.

    ``camera_to_body_wxyz`` rotates vectors from the camera's FRD-like frame
    into the vehicle body FRD frame.  No identity/default calibration is
    supplied by this module; callers must name and bound an actual calibration.
    """

    calibration_id: str
    camera_ray_model_id: str
    camera_to_body_wxyz: QuaternionTuple
    rotation_uncertainty_rad: float

    def __post_init__(self) -> None:
        _bounded_token(self.calibration_id, "calibration_id")
        _bounded_token(self.camera_ray_model_id, "camera_ray_model_id")
        if self.camera_ray_model_id != SUPPORTED_CAMERA_RAY_MODEL_ID:
            raise VQ2ImuDerotationError("unsupported camera ray model")
        object.__setattr__(
            self,
            "camera_to_body_wxyz",
            _unit_quaternion(self.camera_to_body_wxyz, "camera_to_body_wxyz"),
        )
        uncertainty = _finite_float(
            self.rotation_uncertainty_rad,
            "rotation_uncertainty_rad",
            positive=True,
            maximum=math.pi,
        )
        object.__setattr__(self, "rotation_uncertainty_rad", uncertainty)


@dataclass(frozen=True, slots=True)
class VQ2DerotationModel:
    """Named, hard-bounded timing and angular-error model."""

    model_id: str
    attitude_time_model_id: str
    max_capture_alignment_ns: int
    max_target_extrapolation_ns: int
    max_total_timing_uncertainty_ns: int
    angular_rate_uncertainty_rad_s: float

    def __post_init__(self) -> None:
        _bounded_token(self.model_id, "model_id")
        _bounded_token(self.attitude_time_model_id, "attitude_time_model_id")
        _exact_int(
            self.max_capture_alignment_ns,
            "max_capture_alignment_ns",
            minimum=1,
            maximum=HARD_MAX_CAPTURE_ALIGNMENT_NS,
        )
        _exact_int(
            self.max_target_extrapolation_ns,
            "max_target_extrapolation_ns",
            minimum=1,
            maximum=HARD_MAX_TARGET_EXTRAPOLATION_NS,
        )
        _exact_int(
            self.max_total_timing_uncertainty_ns,
            "max_total_timing_uncertainty_ns",
            minimum=1,
            maximum=HARD_MAX_TOTAL_TIMING_UNCERTAINTY_NS,
        )
        rate_uncertainty = _finite_float(
            self.angular_rate_uncertainty_rad_s,
            "angular_rate_uncertainty_rad_s",
            positive=True,
            maximum=math.tau,
        )
        object.__setattr__(self, "angular_rate_uncertainty_rad_s", rate_uncertainty)


@dataclass(frozen=True, slots=True)
class VQ2AttitudeDerotationInput:
    """One calibrated attitude plus explicit orientation/host-time uncertainty."""

    attitude: VQ2TimestampedAttitude
    orientation_uncertainty_rad: float
    host_time_uncertainty_ns: int

    def __post_init__(self) -> None:
        if type(self.attitude) is not VQ2TimestampedAttitude:
            raise TypeError("attitude must be VQ2TimestampedAttitude")
        if type(self.attitude.healthy) is not bool or not self.attitude.healthy:
            raise VQ2ImuDerotationError("attitude must be healthy")
        if type(self.attitude.calibrated) is not bool or not self.attitude.calibrated:
            raise VQ2ImuDerotationError("attitude must be calibrated")
        uncertainty = _finite_float(
            self.orientation_uncertainty_rad,
            "orientation_uncertainty_rad",
            positive=True,
            maximum=math.pi,
        )
        _exact_int(
            self.host_time_uncertainty_ns,
            "host_time_uncertainty_ns",
            minimum=1,
        )
        object.__setattr__(self, "orientation_uncertainty_rad", uncertainty)

    def orientation_at_host_time(
        self,
        host_clock_id: str,
        monotonic_ns: int,
    ) -> QuaternionTuple:
        """Propagate this sample forward on its identified host clock.

        This is a deterministic constant-body-rate operation, not a new IMU
        sample or a wire value.  Callers remain responsible for applying a
        reviewed extrapolation and uncertainty bound for their use case.
        """

        _bounded_token(host_clock_id, "host_clock_id")
        target = _exact_int(monotonic_ns, "monotonic_ns", minimum=0)
        if host_clock_id != self.attitude.host_clock_id:
            raise VQ2ImuDerotationError("attitude propagation host clock differs")
        if target < self.attitude.receive_monotonic_ns:
            raise VQ2ImuDerotationError("attitude propagation cannot predate receipt")
        return _propagate_body_to_ned(
            self.attitude.orientation_body_to_ned_wxyz,
            self.attitude.body_rates_rad_s,
            (target - self.attitude.receive_monotonic_ns) * 1e-9,
        )


@dataclass(frozen=True, slots=True)
class VQ2DerotationEvidence:
    """Exact local evidence for one bounded rotation-only correction.

    The frozen observation remains the measurement source of record.  This
    wrapper does not relabel it as an IMU measurement and is not a wire schema.
    """

    observation: GateObservationV1
    prediction_target: RelativePredictionTarget
    capture_attitude: VQ2AttitudeDerotationInput
    target_attitude: VQ2AttitudeDerotationInput
    calibration: VQ2CameraToBodyCalibration
    model: VQ2DerotationModel
    capture_alignment_ns: int
    target_extrapolation_ns: int
    combined_timing_uncertainty_ns: int
    combined_angular_uncertainty_rad: float
    input_center_norm: Vector2
    derotated_center_norm: Vector2
    input_center_covariance_norm2: Matrix2
    derotated_center_covariance_norm2: Matrix2

    def __post_init__(self) -> None:
        _validate_exact_context_types(
            self.observation,
            self.prediction_target,
            self.capture_attitude,
            self.target_attitude,
            self.calibration,
            self.model,
        )
        _validate_context_bindings(
            self.observation,
            self.prediction_target,
            self.capture_attitude,
            self.target_attitude,
            self.model,
        )
        expected_capture_alignment = (
            self.observation.measurement_time_monotonic_ns
            - self.capture_attitude.attitude.receive_monotonic_ns
        )
        expected_target_extrapolation = (
            self.prediction_target.prediction_time_monotonic_ns
            - self.target_attitude.attitude.receive_monotonic_ns
        )
        expected_timing_uncertainty = _combined_timing_uncertainty_ns(
            self.observation,
            self.prediction_target,
            self.capture_attitude,
            self.target_attitude,
        )
        expected_angular_uncertainty = _combined_angular_uncertainty_rad(
            self.observation,
            self.prediction_target,
            self.capture_attitude,
            self.target_attitude,
            self.calibration,
            self.model,
            expected_capture_alignment,
            expected_target_extrapolation,
        )
        if self.capture_alignment_ns != expected_capture_alignment:
            raise VQ2ImuDerotationError("capture_alignment_ns does not match its sources")
        if self.target_extrapolation_ns != expected_target_extrapolation:
            raise VQ2ImuDerotationError("target_extrapolation_ns does not match its sources")
        if self.combined_timing_uncertainty_ns != expected_timing_uncertainty:
            raise VQ2ImuDerotationError(
                "combined_timing_uncertainty_ns does not match its sources"
            )
        angular_uncertainty = _finite_float(
            self.combined_angular_uncertainty_rad,
            "combined_angular_uncertainty_rad",
            positive=True,
        )
        if not math.isclose(
            angular_uncertainty,
            expected_angular_uncertainty,
            rel_tol=0.0,
            abs_tol=_EQUALITY_TOLERANCE,
        ):
            raise VQ2ImuDerotationError(
                "combined_angular_uncertainty_rad does not match its sources"
            )
        input_center = _vector(
            self.input_center_norm,
            2,
            "input_center_norm",
            bound=MAX_ABS_NORMALIZED_BEARING,
        )
        output_center = _vector(
            self.derotated_center_norm,
            2,
            "derotated_center_norm",
            bound=MAX_ABS_NORMALIZED_BEARING,
        )
        if input_center != self.observation.center_norm:
            raise VQ2ImuDerotationError("input center does not match the observation")
        expected_center, jacobian = _derive_center_and_jacobian(
            input_center,
            self.capture_attitude,
            self.target_attitude,
            self.calibration,
            expected_capture_alignment,
            expected_target_extrapolation,
        )
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=_EQUALITY_TOLERANCE)
            for actual, expected in zip(output_center, expected_center)
        ):
            raise VQ2ImuDerotationError("derotated center does not match its sources")
        input_covariance = _matrix2(
            self.input_center_covariance_norm2,
            "input_center_covariance_norm2",
        )
        expected_input_covariance = _observation_center_covariance(self.observation)
        if not _matrix_close(input_covariance, expected_input_covariance):
            raise VQ2ImuDerotationError("input covariance does not match the observation")
        output_covariance = _matrix2(
            self.derotated_center_covariance_norm2,
            "derotated_center_covariance_norm2",
        )
        expected_output_covariance = _conservative_output_covariance(
            input_covariance,
            jacobian,
            expected_center,
            expected_angular_uncertainty,
        )
        if not _matrix_close(output_covariance, expected_output_covariance):
            raise VQ2ImuDerotationError("derotated covariance does not match its sources")
        covariance_increment = (
            (
                output_covariance[0][0] - input_covariance[0][0],
                output_covariance[0][1] - input_covariance[0][1],
            ),
            (
                output_covariance[1][0] - input_covariance[1][0],
                output_covariance[1][1] - input_covariance[1][1],
            ),
        )
        if _minimum_eigenvalue_2x2(covariance_increment) < -1e-12:
            raise VQ2ImuDerotationError("derotation uncertainty must never tighten")
        object.__setattr__(self, "combined_angular_uncertainty_rad", angular_uncertainty)
        object.__setattr__(self, "input_center_norm", input_center)
        object.__setattr__(self, "derotated_center_norm", output_center)
        object.__setattr__(self, "input_center_covariance_norm2", input_covariance)
        object.__setattr__(self, "derotated_center_covariance_norm2", output_covariance)

    def validate_integrity(self) -> None:
        """Re-run all evidence bindings and derived-value checks transactionally.

        Frozen values normally need validation only at construction.  The
        outer offline trust boundary calls this method as defense in depth
        against deliberately corrupted objects created with low-level Python
        mutation helpers.  Validation uses reconstructed values and never
        canonicalizes or otherwise mutates the supplied evidence.
        """

        observation = replace(self.observation)
        prediction_target = replace(self.prediction_target)

        def validated_attitude_input(
            attitude_input: VQ2AttitudeDerotationInput,
        ) -> VQ2AttitudeDerotationInput:
            source = replace(attitude_input.attitude.source)
            attitude = replace(attitude_input.attitude, source=source)
            return replace(attitude_input, attitude=attitude)

        capture_attitude = validated_attitude_input(self.capture_attitude)
        target_attitude = validated_attitude_input(self.target_attitude)
        calibration = replace(self.calibration)
        model = replace(self.model)
        replace(
            self,
            observation=observation,
            prediction_target=prediction_target,
            capture_attitude=capture_attitude,
            target_attitude=target_attitude,
            calibration=calibration,
            model=model,
        )

    @property
    def effective_target_orientation_body_to_ned_wxyz(self) -> QuaternionTuple:
        """Exact target-time orientation used by the ray transformation."""

        return self.target_attitude.orientation_at_host_time(
            self.prediction_target.host_clock_id,
            self.prediction_target.prediction_time_monotonic_ns,
        )


def derotate_gate_observation(
    observation: GateObservationV1,
    prediction_target: RelativePredictionTarget,
    *,
    capture_attitude: VQ2AttitudeDerotationInput,
    target_attitude: VQ2AttitudeDerotationInput,
    calibration: VQ2CameraToBodyCalibration,
    model: VQ2DerotationModel,
) -> VQ2DerotationEvidence:
    """Rotate one camera bearing between bounded, provenance-bound attitudes."""

    _validate_exact_context_types(
        observation,
        prediction_target,
        capture_attitude,
        target_attitude,
        calibration,
        model,
    )
    capture_alignment_ns, target_extrapolation_ns = _validate_context_bindings(
        observation,
        prediction_target,
        capture_attitude,
        target_attitude,
        model,
    )
    combined_timing_uncertainty_ns = _combined_timing_uncertainty_ns(
        observation,
        prediction_target,
        capture_attitude,
        target_attitude,
    )
    if combined_timing_uncertainty_ns > model.max_total_timing_uncertainty_ns:
        raise VQ2ImuDerotationError("combined timing uncertainty exceeds model bound")
    combined_angular_uncertainty_rad = _combined_angular_uncertainty_rad(
        observation,
        prediction_target,
        capture_attitude,
        target_attitude,
        calibration,
        model,
        capture_alignment_ns,
        target_extrapolation_ns,
    )
    input_center = observation.center_norm
    derotated_center, jacobian = _derive_center_and_jacobian(
        input_center,
        capture_attitude,
        target_attitude,
        calibration,
        capture_alignment_ns,
        target_extrapolation_ns,
    )
    input_covariance = _observation_center_covariance(observation)
    output_covariance = _conservative_output_covariance(
        input_covariance,
        jacobian,
        derotated_center,
        combined_angular_uncertainty_rad,
    )
    return VQ2DerotationEvidence(
        observation=observation,
        prediction_target=prediction_target,
        capture_attitude=capture_attitude,
        target_attitude=target_attitude,
        calibration=calibration,
        model=model,
        capture_alignment_ns=capture_alignment_ns,
        target_extrapolation_ns=target_extrapolation_ns,
        combined_timing_uncertainty_ns=combined_timing_uncertainty_ns,
        combined_angular_uncertainty_rad=combined_angular_uncertainty_rad,
        input_center_norm=input_center,
        derotated_center_norm=derotated_center,
        input_center_covariance_norm2=input_covariance,
        derotated_center_covariance_norm2=output_covariance,
    )


def _validate_exact_context_types(
    observation: GateObservationV1,
    prediction_target: RelativePredictionTarget,
    capture_attitude: VQ2AttitudeDerotationInput,
    target_attitude: VQ2AttitudeDerotationInput,
    calibration: VQ2CameraToBodyCalibration,
    model: VQ2DerotationModel,
) -> None:
    for value, expected, label in (
        (observation, GateObservationV1, "observation"),
        (prediction_target, RelativePredictionTarget, "prediction_target"),
        (capture_attitude, VQ2AttitudeDerotationInput, "capture_attitude"),
        (target_attitude, VQ2AttitudeDerotationInput, "target_attitude"),
        (calibration, VQ2CameraToBodyCalibration, "calibration"),
        (model, VQ2DerotationModel, "model"),
    ):
        if type(value) is not expected:
            raise TypeError(f"{label} must be {expected.__name__}")


def _validate_context_bindings(
    observation: GateObservationV1,
    prediction_target: RelativePredictionTarget,
    capture_input: VQ2AttitudeDerotationInput,
    target_input: VQ2AttitudeDerotationInput,
    model: VQ2DerotationModel,
) -> tuple[int, int]:
    if observation.health is ObservationHealth.UNUSABLE:
        raise VQ2ImuDerotationError("unusable observation cannot be derotated")
    authority = observation.authority
    capture = capture_input.attitude
    target = target_input.attitude
    if prediction_target.host_clock_id != observation.host_clock_id:
        raise VQ2ImuDerotationError("prediction and observation host clocks differ")
    if capture.host_clock_id != observation.host_clock_id or target.host_clock_id != observation.host_clock_id:
        raise VQ2ImuDerotationError("camera and IMU host clocks differ")
    if capture.session_id != authority.session_id or target.session_id != authority.session_id:
        raise VQ2ImuDerotationError("camera authority and IMU sessions differ")
    if capture.reset_epoch != authority.reset_epoch or target.reset_epoch != authority.reset_epoch:
        raise VQ2ImuDerotationError("camera authority and IMU reset epochs differ")
    if capture.stream_id != target.stream_id or capture.generation != target.generation:
        raise VQ2ImuDerotationError("capture and target IMU sources differ")
    if prediction_target.decision_time_monotonic_ns < observation.frame_timing.publish_monotonic_ns:
        raise VQ2ImuDerotationError("decision predates observation publication")
    derotation_interval_ns = (
        prediction_target.prediction_time_monotonic_ns
        - observation.measurement_time_monotonic_ns
    )
    if derotation_interval_ns > HARD_MAX_DEROTATION_INTERVAL_NS:
        raise VQ2ImuDerotationError("camera-to-target derotation interval exceeds hard bound")
    if capture.receive_monotonic_ns > prediction_target.decision_time_monotonic_ns:
        raise VQ2ImuDerotationError("capture attitude was unavailable at decision time")
    if target.receive_monotonic_ns > prediction_target.decision_time_monotonic_ns:
        raise VQ2ImuDerotationError("target attitude was unavailable at decision time")
    if target.receive_monotonic_ns > prediction_target.prediction_time_monotonic_ns:
        raise VQ2ImuDerotationError("target attitude postdates the requested target")
    if target.sample_sequence < capture.sample_sequence:
        raise VQ2ImuDerotationError("target IMU sequence regresses capture identity")
    if target.sample_sequence == capture.sample_sequence:
        if target != capture:
            raise VQ2ImuDerotationError("one IMU sample identity has conflicting attitudes")
    else:
        if target.source_time_us <= capture.source_time_us:
            raise VQ2ImuDerotationError("target IMU source time does not advance")
        if target.receive_monotonic_ns <= capture.receive_monotonic_ns:
            raise VQ2ImuDerotationError("target IMU receive time does not advance")
    capture_alignment_ns = (
        observation.measurement_time_monotonic_ns - capture.receive_monotonic_ns
    )
    if abs(capture_alignment_ns) > model.max_capture_alignment_ns:
        raise VQ2ImuDerotationError("capture attitude alignment exceeds model bound")
    target_extrapolation_ns = (
        prediction_target.prediction_time_monotonic_ns - target.receive_monotonic_ns
    )
    if target_extrapolation_ns > model.max_target_extrapolation_ns:
        raise VQ2ImuDerotationError("target attitude extrapolation exceeds model bound")
    combined_timing_uncertainty = _combined_timing_uncertainty_ns(
        observation,
        prediction_target,
        capture_input,
        target_input,
    )
    if combined_timing_uncertainty > model.max_total_timing_uncertainty_ns:
        raise VQ2ImuDerotationError("combined timing uncertainty exceeds model bound")
    return capture_alignment_ns, target_extrapolation_ns


def _combined_timing_uncertainty_ns(
    observation: GateObservationV1,
    prediction_target: RelativePredictionTarget,
    capture: VQ2AttitudeDerotationInput,
    target: VQ2AttitudeDerotationInput,
) -> int:
    return (
        observation.measurement_uncertainty_ns
        + prediction_target.delay_uncertainty_ns
        + capture.host_time_uncertainty_ns
        + target.host_time_uncertainty_ns
    )


def _combined_angular_uncertainty_rad(
    observation: GateObservationV1,
    prediction_target: RelativePredictionTarget,
    capture: VQ2AttitudeDerotationInput,
    target: VQ2AttitudeDerotationInput,
    calibration: VQ2CameraToBodyCalibration,
    model: VQ2DerotationModel,
    capture_alignment_ns: int,
    target_extrapolation_ns: int,
) -> float:
    capture_rate = _norm3(capture.attitude.body_rates_rad_s)
    target_rate = _norm3(target.attitude.body_rates_rad_s)
    capture_time_uncertainty_s = (
        observation.measurement_uncertainty_ns + capture.host_time_uncertainty_ns
    ) * 1e-9
    target_time_uncertainty_s = (
        prediction_target.delay_uncertainty_ns + target.host_time_uncertainty_ns
    ) * 1e-9
    propagated_interval_s = (abs(capture_alignment_ns) + target_extrapolation_ns) * 1e-9
    return (
        capture.orientation_uncertainty_rad
        + target.orientation_uncertainty_rad
        + 2.0 * calibration.rotation_uncertainty_rad
        + capture_rate * capture_time_uncertainty_s
        + target_rate * target_time_uncertainty_s
        + model.angular_rate_uncertainty_rad_s * propagated_interval_s
    )


def _derive_center_and_jacobian(
    center: tuple[float, ...],
    capture: VQ2AttitudeDerotationInput,
    target: VQ2AttitudeDerotationInput,
    calibration: VQ2CameraToBodyCalibration,
    capture_alignment_ns: int,
    target_extrapolation_ns: int,
) -> tuple[Vector2, Matrix2]:
    capture_q = _propagate_body_to_ned(
        capture.attitude.orientation_body_to_ned_wxyz,
        capture.attitude.body_rates_rad_s,
        capture_alignment_ns * 1e-9,
    )
    target_q = _propagate_body_to_ned(
        target.attitude.orientation_body_to_ned_wxyz,
        target.attitude.body_rates_rad_s,
        target_extrapolation_ns * 1e-9,
    )
    camera_to_body = calibration.camera_to_body_wxyz

    def transform(ray_camera: Vector3) -> Vector3:
        ray_body_at_capture = _quat_rotate(camera_to_body, ray_camera)
        ray_ned = _quat_rotate(capture_q, ray_body_at_capture)
        ray_body_at_target = _quat_rotate(_quat_conjugate(target_q), ray_ned)
        return _quat_rotate(_quat_conjugate(camera_to_body), ray_body_at_target)

    columns = (
        transform((1.0, 0.0, 0.0)),
        transform((0.0, 1.0, 0.0)),
        transform((0.0, 0.0, 1.0)),
    )
    transformed = tuple(
        columns[0][row] + center[0] * columns[1][row] + center[1] * columns[2][row]
        for row in range(3)
    )
    forward, right, down = transformed
    if not math.isfinite(forward) or forward <= _MIN_FORWARD_RAY:
        raise VQ2ImuDerotationError("derotated ray must remain forward-facing")
    output = (right / forward, down / forward)
    if any(not math.isfinite(item) or abs(item) > MAX_ABS_NORMALIZED_BEARING for item in output):
        raise VQ2ImuDerotationError("derotated bearing exceeds normalized bounds")
    inverse_forward_sq = 1.0 / (forward * forward)
    jacobian = (
        (
            (columns[1][1] * forward - right * columns[1][0]) * inverse_forward_sq,
            (columns[2][1] * forward - right * columns[2][0]) * inverse_forward_sq,
        ),
        (
            (columns[1][2] * forward - down * columns[1][0]) * inverse_forward_sq,
            (columns[2][2] * forward - down * columns[2][0]) * inverse_forward_sq,
        ),
    )
    return output, jacobian


def _observation_center_covariance(observation: GateObservationV1) -> Matrix2:
    order = observation.covariance.feature_order
    try:
        x_index = order.index("center_x_norm")
        y_index = order.index("center_y_norm")
    except ValueError as exc:
        raise VQ2ImuDerotationError(
            "observation covariance must include center_x_norm and center_y_norm"
        ) from exc
    matrix = observation.covariance.matrix
    return _matrix2(
        (
            (matrix[x_index][x_index], matrix[x_index][y_index]),
            (matrix[y_index][x_index], matrix[y_index][y_index]),
        ),
        "observation center covariance",
    )


def _conservative_output_covariance(
    input_covariance: Matrix2,
    jacobian: Matrix2,
    output_center: Vector2,
    angular_uncertainty_rad: float,
) -> Matrix2:
    transformed = _sandwich_2x2(jacobian, input_covariance)
    angular_to_slope = 1.0 + output_center[0] ** 2 + output_center[1] ** 2
    added_variance = (angular_to_slope * angular_uncertainty_rad) ** 2
    # An isotropic envelope simultaneously dominates both the original and
    # linearly transformed covariances.  Thus P_out - P_in is PSD: derotation
    # can never make the camera observation look more certain.
    envelope_variance = max(
        _maximum_eigenvalue_2x2(input_covariance),
        _maximum_eigenvalue_2x2(transformed),
    ) + added_variance
    return ((envelope_variance, 0.0), (0.0, envelope_variance))


def _sandwich_2x2(jacobian: Matrix2, covariance: Matrix2) -> Matrix2:
    j00, j01 = jacobian[0]
    j10, j11 = jacobian[1]
    p00, p01 = covariance[0]
    p10, p11 = covariance[1]
    a00 = j00 * p00 + j01 * p10
    a01 = j00 * p01 + j01 * p11
    a10 = j10 * p00 + j11 * p10
    a11 = j10 * p01 + j11 * p11
    return (
        (a00 * j00 + a01 * j01, a00 * j10 + a01 * j11),
        (a10 * j00 + a11 * j01, a10 * j10 + a11 * j11),
    )


def _minimum_eigenvalue_2x2(matrix: Matrix2) -> float:
    trace = matrix[0][0] + matrix[1][1]
    discriminant = math.hypot(matrix[0][0] - matrix[1][1], 2.0 * matrix[0][1])
    return 0.5 * (trace - discriminant)


def _maximum_eigenvalue_2x2(matrix: Matrix2) -> float:
    trace = matrix[0][0] + matrix[1][1]
    discriminant = math.hypot(matrix[0][0] - matrix[1][1], 2.0 * matrix[0][1])
    return 0.5 * (trace + discriminant)


def _matrix_close(first: Matrix2, second: Matrix2) -> bool:
    return all(
        math.isclose(first[row][column], second[row][column], rel_tol=0.0, abs_tol=_EQUALITY_TOLERANCE)
        for row in range(2)
        for column in range(2)
    )


def _norm3(value: Vector3) -> float:
    return math.sqrt(sum(item * item for item in value))


def _propagate_body_to_ned(
    orientation_body_to_ned_wxyz: QuaternionTuple,
    body_rates_rad_s: Vector3,
    dt_s: float,
) -> QuaternionTuple:
    orientation = _unit_quaternion(
        orientation_body_to_ned_wxyz,
        "orientation_body_to_ned_wxyz",
    )
    rates = _vector(body_rates_rad_s, 3, "body_rates_rad_s")
    rotation_vector = tuple(rate * dt_s for rate in rates)
    delta = _quaternion_from_rotation_vector(rotation_vector)  # type: ignore[arg-type]
    return _normalize_quaternion(_quat_multiply(orientation, delta))


def _quaternion_from_rotation_vector(rotation_vector: Vector3) -> QuaternionTuple:
    angle = _norm3(rotation_vector)
    if angle <= 1e-15:
        return _normalize_quaternion(
            (1.0, 0.5 * rotation_vector[0], 0.5 * rotation_vector[1], 0.5 * rotation_vector[2])
        )
    scale = math.sin(0.5 * angle) / angle
    return (
        math.cos(0.5 * angle),
        scale * rotation_vector[0],
        scale * rotation_vector[1],
        scale * rotation_vector[2],
    )


def _quat_multiply(first: QuaternionTuple, second: QuaternionTuple) -> QuaternionTuple:
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_conjugate(value: QuaternionTuple) -> QuaternionTuple:
    return (value[0], -value[1], -value[2], -value[3])


def _quat_rotate(quaternion: QuaternionTuple, vector: Vector3) -> Vector3:
    pure = (0.0, vector[0], vector[1], vector[2])
    rotated = _quat_multiply(_quat_multiply(quaternion, pure), _quat_conjugate(quaternion))
    return (rotated[1], rotated[2], rotated[3])


def _normalize_quaternion(value: QuaternionTuple) -> QuaternionTuple:
    norm = math.sqrt(sum(item * item for item in value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise VQ2ImuDerotationError("quaternion propagation became invalid")
    return tuple(item / norm for item in value)  # type: ignore[return-value]


__all__ = [
    "HARD_MAX_CAPTURE_ALIGNMENT_NS",
    "HARD_MAX_DEROTATION_INTERVAL_NS",
    "HARD_MAX_TARGET_EXTRAPOLATION_NS",
    "HARD_MAX_TOTAL_TIMING_UNCERTAINTY_NS",
    "MAX_ABS_NORMALIZED_BEARING",
    "SUPPORTED_CAMERA_RAY_MODEL_ID",
    "VQ2AttitudeDerotationInput",
    "VQ2CameraToBodyCalibration",
    "VQ2DerotationEvidence",
    "VQ2DerotationModel",
    "VQ2ImuDerotationError",
    "derotate_gate_observation",
]
