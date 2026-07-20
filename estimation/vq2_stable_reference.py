"""Standalone stable-orientation transforms for a local VQ2 feature model.

This module deliberately does *not* transform the frozen ``/1`` finite-quad
``log_scale``.  Its six-dimensional state uses a separately named local area-
density semantic.  Public derotation evidence supplies only immutable attitude,
time, calibration, and lineage inputs.  No estimator, controller, runtime,
supervisor, transport, or powered authority is present here.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

from competition.vq2_contracts import (
    FrameEdge,
    GateObservationV1,
    MeasurementTimeBasis,
    ObservationHealth,
    TrackRole,
)
from estimation.vq2_imu_derotation import (
    VQ2CameraToBodyCalibration,
    VQ2DerotationEvidence,
    VQ2DerotationModel,
)


Vector3 = tuple[float, float, float]
Vector6 = tuple[float, float, float, float, float, float]
QuaternionTuple = tuple[float, float, float, float]
Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Matrix6 = tuple[
    tuple[float, float, float, float, float, float],
    tuple[float, float, float, float, float, float],
    tuple[float, float, float, float, float, float],
    tuple[float, float, float, float, float, float],
    tuple[float, float, float, float, float, float],
    tuple[float, float, float, float, float, float],
]


LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID = "vq2-local-differential-area-v1"
SUPPORTED_STABLE_CHART_MODEL_ID = "vq2-synthetic-frd-normalized-chart-v1"
LOCAL_FEATURE_ORDER = (
    "bearing_x_norm",
    "bearing_y_norm",
    "local_log_scale",
    "bearing_rate_x_norm_s",
    "bearing_rate_y_norm_s",
    "local_expansion_rate_s",
)
IDENTITY_CHART_TO_CAMERA_RAY: Matrix3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

HARD_MAX_REFERENCE_AGE_NS = 1_000_000_000
HARD_MAX_TOTAL_TIMING_UNCERTAINTY_NS = 200_000_000
HARD_MAX_RELATIVE_ANGULAR_UNCERTAINTY_RAD = math.pi / 4.0
HARD_MAX_ANGULAR_RATE_RAD_S = 20.0
HARD_MAX_ANGULAR_RATE_UNCERTAINTY_RAD_S = 5.0
HARD_MAX_ANGULAR_ACCELERATION_RAD_S2 = 200.0
HARD_MAX_FEATURE_ACCELERATION_NORM_S2 = 200.0
HARD_MAX_ABS_BEARING_NORM = 4.0
HARD_MAX_ABS_BEARING_RATE_NORM_S = 8.0
HARD_MAX_ABS_LOCAL_LOG_SCALE = 20.0
HARD_MAX_ABS_LOCAL_EXPANSION_RATE_S = 8.0

_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_QUATERNION_NORM_TOLERANCE = 1e-9
_DERIVED_ABS_TOLERANCE = 1e-11
_MATRIX_SYMMETRY_REL_TOLERANCE = 1e-10
_MATRIX_PSD_REL_TOLERANCE = 1e-10
_ALL_FRAME_EDGES = FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT | FrameEdge.BOTTOM


class VQ2StableReferenceError(ValueError):
    """Raised when stable-reference semantics or evidence fail closed."""


class VQ2StableReferenceMismatchError(VQ2StableReferenceError):
    """Raised when evidence does not belong to one immutable reference."""


class VQ2StableReferenceChronologyError(VQ2StableReferenceError):
    """Raised when reference or measurement chronology is replayed/relabelled."""


class VQ2StableReferenceGeometryError(VQ2StableReferenceError):
    """Raised when a projective chart leaves its reviewed numerical envelope."""


class VQ2LocalFeatureBasis(str, Enum):
    CAMERA = "camera"
    STABLE_REFERENCE = "stable_reference"


class VQ2CameraFeatureTime(str, Enum):
    CAPTURE = "capture"
    TARGET = "target"


class VQ2StableTransformDirection(str, Enum):
    CAMERA_TO_STABLE = "camera_to_stable"
    STABLE_TO_CAMERA = "stable_to_camera"


class VQ2CovarianceScope(str, Enum):
    CONDITIONAL_INPUT = "conditional_input"
    TRANSFORM_TOTAL = "transform_total"


def _bounded_token(value: object, label: str) -> str:
    if type(value) is not str or _TOKEN_RE.fullmatch(value) is None:
        raise VQ2StableReferenceError(f"{label} must be a bounded token")
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
        raise VQ2StableReferenceError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise VQ2StableReferenceError(f"{label} must be <= {maximum}")
    return value


def _finite_float(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise VQ2StableReferenceError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise VQ2StableReferenceError(f"{label} must be positive")
    if nonnegative and result < 0.0:
        raise VQ2StableReferenceError(f"{label} must be nonnegative")
    if minimum is not None and result < minimum:
        raise VQ2StableReferenceError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise VQ2StableReferenceError(f"{label} must be <= {maximum}")
    return result


def _vector(value: object, length: int, label: str) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise TypeError(f"{label} must be an exact {length}-tuple")
    return tuple(
        _finite_float(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )


def _unit_quaternion(value: object, label: str) -> QuaternionTuple:
    quaternion = _vector(value, 4, label)
    norm_sq = sum(component * component for component in quaternion)
    if not math.isclose(
        norm_sq,
        1.0,
        rel_tol=0.0,
        abs_tol=_QUATERNION_NORM_TOLERANCE,
    ):
        raise VQ2StableReferenceError(f"{label} must be unit length")
    return quaternion  # type: ignore[return-value]


def _matrix(
    value: object,
    dimension: int,
    label: str,
    *,
    covariance: bool = False,
    covariance_tolerance: float = _MATRIX_PSD_REL_TOLERANCE,
) -> tuple[tuple[float, ...], ...]:
    if type(value) is not tuple or len(value) != dimension:
        raise TypeError(f"{label} must be an exact {dimension}x{dimension} tuple")
    rows = tuple(
        _vector(row, dimension, f"{label}[{index}]")
        for index, row in enumerate(value)
    )
    if not covariance:
        return rows
    array = np.asarray(rows, dtype=np.float64)
    # A covariance's admissible roundoff is relative to that covariance, not
    # to an implicit unit matrix.  Using a unit lower bound here would accept a
    # materially indefinite matrix merely because all of its entries are
    # small in absolute units.
    scale = float(np.max(np.abs(array)))
    relative_tolerance = min(
        _MATRIX_SYMMETRY_REL_TOLERANCE,
        covariance_tolerance,
    )
    symmetry_tolerance = relative_tolerance * scale
    if not np.allclose(array, array.T, rtol=0.0, atol=symmetry_tolerance):
        raise VQ2StableReferenceError(f"{label} must be symmetric")
    array = (array + array.T) * 0.5
    if np.any(np.diag(array) <= 0.0):
        raise VQ2StableReferenceError(f"{label} variances must be positive")
    minimum_eigenvalue = float(np.linalg.eigvalsh(array)[0])
    if minimum_eigenvalue < -covariance_tolerance * scale:
        raise VQ2StableReferenceError(f"{label} must be positive semidefinite")
    return _matrix_tuple(array)


def _matrix3(value: object, label: str) -> Matrix3:
    matrix = _matrix(value, 3, label)
    return matrix  # type: ignore[return-value]


def _covariance6(
    value: object,
    label: str,
    *,
    tolerance: float = _MATRIX_PSD_REL_TOLERANCE,
) -> Matrix6:
    matrix = _matrix(
        value,
        6,
        label,
        covariance=True,
        covariance_tolerance=tolerance,
    )
    return matrix  # type: ignore[return-value]


def _matrix_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(component) for component in row) for row in value)


def _matrix_close(
    first: tuple[tuple[float, ...], ...],
    second: tuple[tuple[float, ...], ...],
    *,
    tolerance: float = _DERIVED_ABS_TOLERANCE,
) -> bool:
    if len(first) != len(second) or any(
        len(first_row) != len(second_row)
        for first_row, second_row in zip(first, second)
    ):
        return False
    return all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
        for actual_row, expected_row in zip(first, second)
        for actual, expected in zip(actual_row, expected_row)
    )


def _vector_close(
    first: tuple[float, ...],
    second: tuple[float, ...],
    *,
    tolerance: float = _DERIVED_ABS_TOLERANCE,
) -> bool:
    return len(first) == len(second) and all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)
        for actual, expected in zip(first, second)
    )


def _floor_covariance(
    value: Matrix6,
    *,
    tolerance: float,
    eigenvalue_floor: float,
    label: str,
) -> Matrix6:
    """Canonicalize only tolerated PSD roundoff by adding uncertainty."""

    array = np.asarray(value, dtype=np.float64)
    scale = float(np.max(np.abs(array)))
    minimum = float(np.linalg.eigvalsh(array)[0])
    if minimum < -tolerance * scale:
        raise VQ2StableReferenceError(f"{label} must be positive semidefinite")
    if minimum < eigenvalue_floor:
        array += np.eye(6) * (eigenvalue_floor - minimum)
    array = (array + array.T) * 0.5
    return _matrix_tuple(array)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class VQ2StableReferenceModel:
    """Explicit synthetic chart, numerical envelope, and covariance floors."""

    model_id: str
    local_feature_model_id: str
    chart_model_id: str
    chart_to_camera_ray: Matrix3
    covariance_model_id: str
    max_reference_age_ns: int
    max_total_timing_uncertainty_ns: int
    max_relative_angular_uncertainty_rad: float
    max_angular_rate_rad_s: float
    max_angular_rate_uncertainty_rad_s: float
    max_angular_acceleration_rad_s2: float
    max_feature_acceleration_norm_s2: float
    minimum_forward: float
    max_homography_condition: float
    max_feature_jacobian_condition: float
    max_projective_magnification: float
    max_abs_bearing_norm: float
    max_abs_bearing_rate_norm_s: float
    max_abs_local_log_scale: float
    max_abs_local_expansion_rate_s: float
    covariance_psd_tolerance: float
    covariance_eigenvalue_floor: float
    joint_nuisance_envelope_covariance: Matrix6
    model_floor_covariance: Matrix6

    def __post_init__(self) -> None:
        _bounded_token(self.model_id, "model_id")
        _bounded_token(self.local_feature_model_id, "local_feature_model_id")
        if self.local_feature_model_id != LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID:
            raise VQ2StableReferenceError("unsupported local feature semantic")
        _bounded_token(self.chart_model_id, "chart_model_id")
        if self.chart_model_id != SUPPORTED_STABLE_CHART_MODEL_ID:
            raise VQ2StableReferenceError("unsupported stable chart model")
        chart = _matrix3(self.chart_to_camera_ray, "chart_to_camera_ray")
        if not _matrix_close(chart, IDENTITY_CHART_TO_CAMERA_RAY):
            raise VQ2StableReferenceError(
                "the first stable chart model requires explicit synthetic identity rays"
            )
        _bounded_token(self.covariance_model_id, "covariance_model_id")
        _exact_int(
            self.max_reference_age_ns,
            "max_reference_age_ns",
            minimum=1,
            maximum=HARD_MAX_REFERENCE_AGE_NS,
        )
        _exact_int(
            self.max_total_timing_uncertainty_ns,
            "max_total_timing_uncertainty_ns",
            minimum=1,
            maximum=HARD_MAX_TOTAL_TIMING_UNCERTAINTY_NS,
        )
        for name, hard_maximum in (
            (
                "max_relative_angular_uncertainty_rad",
                HARD_MAX_RELATIVE_ANGULAR_UNCERTAINTY_RAD,
            ),
            ("max_angular_rate_rad_s", HARD_MAX_ANGULAR_RATE_RAD_S),
            (
                "max_angular_rate_uncertainty_rad_s",
                HARD_MAX_ANGULAR_RATE_UNCERTAINTY_RAD_S,
            ),
            (
                "max_angular_acceleration_rad_s2",
                HARD_MAX_ANGULAR_ACCELERATION_RAD_S2,
            ),
            (
                "max_feature_acceleration_norm_s2",
                HARD_MAX_FEATURE_ACCELERATION_NORM_S2,
            ),
        ):
            value = _finite_float(
                getattr(self, name),
                name,
                positive=True,
                maximum=hard_maximum,
            )
            object.__setattr__(self, name, value)
        minimum_forward = _finite_float(
            self.minimum_forward,
            "minimum_forward",
            positive=True,
            maximum=0.5,
        )
        object.__setattr__(self, "minimum_forward", minimum_forward)
        for name in (
            "max_homography_condition",
            "max_feature_jacobian_condition",
            "max_projective_magnification",
        ):
            value = _finite_float(
                getattr(self, name),
                name,
                minimum=1.0,
                maximum=1e6,
            )
            object.__setattr__(self, name, value)
        for name, hard_maximum in (
            ("max_abs_bearing_norm", HARD_MAX_ABS_BEARING_NORM),
            (
                "max_abs_bearing_rate_norm_s",
                HARD_MAX_ABS_BEARING_RATE_NORM_S,
            ),
            ("max_abs_local_log_scale", HARD_MAX_ABS_LOCAL_LOG_SCALE),
            (
                "max_abs_local_expansion_rate_s",
                HARD_MAX_ABS_LOCAL_EXPANSION_RATE_S,
            ),
        ):
            value = _finite_float(
                getattr(self, name),
                name,
                positive=True,
                maximum=hard_maximum,
            )
            object.__setattr__(self, name, value)
        tolerance = _finite_float(
            self.covariance_psd_tolerance,
            "covariance_psd_tolerance",
            positive=True,
            maximum=1e-6,
        )
        floor = _finite_float(
            self.covariance_eigenvalue_floor,
            "covariance_eigenvalue_floor",
            positive=True,
            maximum=tolerance,
        )
        nuisance = _covariance6(
            self.joint_nuisance_envelope_covariance,
            "joint_nuisance_envelope_covariance",
            tolerance=tolerance,
        )
        model_floor = _covariance6(
            self.model_floor_covariance,
            "model_floor_covariance",
            tolerance=tolerance,
        )
        nuisance = _floor_covariance(
            nuisance,
            tolerance=tolerance,
            eigenvalue_floor=floor,
            label="joint_nuisance_envelope_covariance",
        )
        model_floor = _floor_covariance(
            model_floor,
            tolerance=tolerance,
            eigenvalue_floor=floor,
            label="model_floor_covariance",
        )
        object.__setattr__(self, "chart_to_camera_ray", chart)
        object.__setattr__(self, "covariance_psd_tolerance", tolerance)
        object.__setattr__(self, "covariance_eigenvalue_floor", floor)
        object.__setattr__(self, "joint_nuisance_envelope_covariance", nuisance)
        object.__setattr__(self, "model_floor_covariance", model_floor)


@dataclass(frozen=True, slots=True)
class VQ2LocalDifferentialFeatureState:
    """One local-only feature state; never a frozen ``/1`` state."""

    feature_model_id: str
    chart_model_id: str
    basis: VQ2LocalFeatureBasis
    basis_id: str
    host_clock_id: str
    time_monotonic_ns: int
    values: Vector6
    covariance_model_id: str
    covariance_scope: VQ2CovarianceScope
    covariance: Matrix6

    def __post_init__(self) -> None:
        _bounded_token(self.feature_model_id, "feature_model_id")
        if self.feature_model_id != LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID:
            raise VQ2StableReferenceError("feature state has the wrong semantic")
        _bounded_token(self.chart_model_id, "chart_model_id")
        if type(self.basis) is not VQ2LocalFeatureBasis:
            raise TypeError("basis must be exact VQ2LocalFeatureBasis")
        _bounded_token(self.basis_id, "basis_id")
        _bounded_token(self.host_clock_id, "host_clock_id")
        _exact_int(self.time_monotonic_ns, "time_monotonic_ns", minimum=0)
        values = _vector(self.values, 6, "values")
        bounds = (
            HARD_MAX_ABS_BEARING_NORM,
            HARD_MAX_ABS_BEARING_NORM,
            HARD_MAX_ABS_LOCAL_LOG_SCALE,
            HARD_MAX_ABS_BEARING_RATE_NORM_S,
            HARD_MAX_ABS_BEARING_RATE_NORM_S,
            HARD_MAX_ABS_LOCAL_EXPANSION_RATE_S,
        )
        if any(abs(component) > bound for component, bound in zip(values, bounds)):
            raise VQ2StableReferenceError("local feature value exceeds its hard bound")
        _bounded_token(self.covariance_model_id, "covariance_model_id")
        if type(self.covariance_scope) is not VQ2CovarianceScope:
            raise TypeError("covariance_scope must be exact VQ2CovarianceScope")
        covariance = _covariance6(self.covariance, "covariance")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True, slots=True)
class VQ2StableReferenceKey:
    """Exact immutable owner/source/model identity for one reference chart."""

    tracker_id: str
    track_role: TrackRole
    session_id: str
    reset_epoch: int
    gate_epoch: int
    expected_gate_index: int
    camera_host_clock_id: str
    camera_stream_id: str
    camera_generation: int
    image_size_px: tuple[int, int]
    imu_epoch_key: tuple[str, int, str, str, int]
    measurement_time_basis: MeasurementTimeBasis
    measurement_time_model_id: str | None
    calibration: VQ2CameraToBodyCalibration
    derotation_model: VQ2DerotationModel
    stable_model: VQ2StableReferenceModel

    def __post_init__(self) -> None:
        _bounded_token(self.tracker_id, "tracker_id")
        if type(self.track_role) is not TrackRole:
            raise TypeError("track_role must be exact TrackRole")
        _bounded_token(self.session_id, "session_id")
        _exact_int(self.reset_epoch, "reset_epoch", minimum=0)
        _exact_int(self.gate_epoch, "gate_epoch", minimum=0)
        _exact_int(self.expected_gate_index, "expected_gate_index", minimum=0)
        _bounded_token(self.camera_host_clock_id, "camera_host_clock_id")
        _bounded_token(self.camera_stream_id, "camera_stream_id")
        _exact_int(self.camera_generation, "camera_generation", minimum=0)
        if (
            type(self.image_size_px) is not tuple
            or len(self.image_size_px) != 2
            or any(type(value) is not int or value <= 0 for value in self.image_size_px)
        ):
            raise TypeError("image_size_px must be an exact positive integer pair")
        if type(self.imu_epoch_key) is not tuple or len(self.imu_epoch_key) != 5:
            raise TypeError("imu_epoch_key must be an exact five-tuple")
        _bounded_token(self.imu_epoch_key[0], "imu_epoch_key.session_id")
        _exact_int(self.imu_epoch_key[1], "imu_epoch_key.reset_epoch", minimum=0)
        _bounded_token(self.imu_epoch_key[2], "imu_epoch_key.host_clock_id")
        _bounded_token(self.imu_epoch_key[3], "imu_epoch_key.stream_id")
        _exact_int(self.imu_epoch_key[4], "imu_epoch_key.generation", minimum=0)
        if type(self.measurement_time_basis) is not MeasurementTimeBasis:
            raise TypeError("measurement_time_basis must be exact MeasurementTimeBasis")
        if self.measurement_time_model_id is not None:
            _bounded_token(self.measurement_time_model_id, "measurement_time_model_id")
        if type(self.calibration) is not VQ2CameraToBodyCalibration:
            raise TypeError("calibration must be exact VQ2CameraToBodyCalibration")
        if type(self.derotation_model) is not VQ2DerotationModel:
            raise TypeError("derotation_model must be exact VQ2DerotationModel")
        if type(self.stable_model) is not VQ2StableReferenceModel:
            raise TypeError("stable_model must be exact VQ2StableReferenceModel")
        # Reconstruct nested frozen values so low-level mutation cannot smuggle
        # an invalid calibration or model into a newly created reference key.
        object.__setattr__(self, "calibration", replace(self.calibration))
        object.__setattr__(self, "derotation_model", replace(self.derotation_model))
        object.__setattr__(self, "stable_model", replace(self.stable_model))


@dataclass(frozen=True, slots=True)
class VQ2StableReference:
    """Immutable capture-time orientation chart and its complete seed lineage."""

    reference_id: str
    key: VQ2StableReferenceKey
    seed_evidence: VQ2DerotationEvidence
    reference_camera_to_ned_wxyz: QuaternionTuple
    reference_angular_uncertainty_rad: float

    @property
    def basis_id(self) -> str:
        """Derived local identity for this exact seed chart, never a wire ID."""

        return _reference_basis_id(
            self.reference_id,
            self.key,
            self.seed_evidence,
            self.reference_camera_to_ned_wxyz,
        )

    def __post_init__(self) -> None:
        _bounded_token(self.reference_id, "reference_id")
        if type(self.key) is not VQ2StableReferenceKey:
            raise TypeError("key must be exact VQ2StableReferenceKey")
        if type(self.seed_evidence) is not VQ2DerotationEvidence:
            raise TypeError("seed_evidence must be exact VQ2DerotationEvidence")
        _validate_derotation_evidence_integrity(self.seed_evidence)
        _validate_reference_observation(self.seed_evidence)
        expected_key = _build_reference_key(
            self.seed_evidence,
            tracker_id=self.key.tracker_id,
            track_role=self.key.track_role,
            stable_model=self.key.stable_model,
        )
        if self.key != expected_key:
            raise VQ2StableReferenceMismatchError("reference key does not match seed evidence")
        orientation = _unit_quaternion(
            self.reference_camera_to_ned_wxyz,
            "reference_camera_to_ned_wxyz",
        )
        expected_orientation, _ = _camera_chart_at(
            self.seed_evidence,
            VQ2CameraFeatureTime.CAPTURE,
        )
        if not _vector_close(orientation, expected_orientation):
            raise VQ2StableReferenceError(
                "reference camera orientation does not match seed capture"
            )
        uncertainty = _finite_float(
            self.reference_angular_uncertainty_rad,
            "reference_angular_uncertainty_rad",
            positive=True,
        )
        expected_uncertainty = self.seed_evidence.combined_angular_uncertainty_rad
        if not math.isclose(
            uncertainty,
            expected_uncertainty,
            rel_tol=0.0,
            abs_tol=_DERIVED_ABS_TOLERANCE,
        ):
            raise VQ2StableReferenceError(
                "reference angular uncertainty does not match seed evidence"
            )
        if uncertainty > self.key.stable_model.max_relative_angular_uncertainty_rad:
            raise VQ2StableReferenceError("seed angular uncertainty exceeds stable model")
        object.__setattr__(self, "reference_camera_to_ned_wxyz", orientation)
        object.__setattr__(self, "reference_angular_uncertainty_rad", uncertainty)

    def validate_integrity(self) -> None:
        """Deeply reconstruct the reference and all nested model/evidence values."""

        _validate_derotation_evidence_integrity(self.seed_evidence)
        key = replace(
            self.key,
            calibration=replace(self.key.calibration),
            derotation_model=replace(self.key.derotation_model),
            stable_model=replace(self.key.stable_model),
        )
        replace(self, key=key)


@dataclass(frozen=True, slots=True)
class VQ2StableFeatureTransformEvidence:
    """Deeply bound result of one pure local-feature chart transform."""

    reference: VQ2StableReference
    evidence: VQ2DerotationEvidence
    direction: VQ2StableTransformDirection
    camera_time: VQ2CameraFeatureTime
    input_state: VQ2LocalDifferentialFeatureState
    homography_input_to_output: Matrix3
    homography_rate: Matrix3
    state_jacobian: Matrix6
    coordinate_covariance: Matrix6
    joint_nuisance_envelope_covariance: Matrix6
    model_floor_covariance: Matrix6
    total_covariance: Matrix6
    selected_camera_angular_rate_rad_s: Vector3
    combined_timing_uncertainty_ns: int
    combined_angular_uncertainty_rad: float
    projective_forward: float
    local_area_jacobian_determinant: float
    homography_condition: float
    feature_jacobian_condition: float
    projective_magnification: float
    output_state: VQ2LocalDifferentialFeatureState

    def __post_init__(self) -> None:
        if type(self.reference) is not VQ2StableReference:
            raise TypeError("reference must be exact VQ2StableReference")
        if type(self.evidence) is not VQ2DerotationEvidence:
            raise TypeError("evidence must be exact VQ2DerotationEvidence")
        if type(self.direction) is not VQ2StableTransformDirection:
            raise TypeError("direction must be exact VQ2StableTransformDirection")
        if type(self.camera_time) is not VQ2CameraFeatureTime:
            raise TypeError("camera_time must be exact VQ2CameraFeatureTime")
        if type(self.input_state) is not VQ2LocalDifferentialFeatureState:
            raise TypeError("input_state must be exact VQ2LocalDifferentialFeatureState")
        if type(self.output_state) is not VQ2LocalDifferentialFeatureState:
            raise TypeError("output_state must be exact VQ2LocalDifferentialFeatureState")
        expected = _derive_transform(
            self.reference,
            self.evidence,
            self.input_state,
            direction=self.direction,
            camera_time=self.camera_time,
        )
        for name in (
            "homography_input_to_output",
            "homography_rate",
            "state_jacobian",
            "coordinate_covariance",
            "joint_nuisance_envelope_covariance",
            "model_floor_covariance",
            "total_covariance",
        ):
            actual_matrix = _matrix(
                getattr(self, name),
                3 if name in {"homography_input_to_output", "homography_rate"} else 6,
                name,
                covariance=name.endswith("covariance"),
            )
            if not _matrix_close(actual_matrix, expected[name]):
                raise VQ2StableReferenceError(f"{name} does not match its sources")
            object.__setattr__(self, name, actual_matrix)
        angular_rate = _vector(
            self.selected_camera_angular_rate_rad_s,
            3,
            "selected_camera_angular_rate_rad_s",
        )
        if not _vector_close(angular_rate, expected["selected_camera_angular_rate_rad_s"]):
            raise VQ2StableReferenceError(
                "selected camera angular rate does not match its sources"
            )
        timing_uncertainty = _exact_int(
            self.combined_timing_uncertainty_ns,
            "combined_timing_uncertainty_ns",
            minimum=1,
        )
        if timing_uncertainty != expected["combined_timing_uncertainty_ns"]:
            raise VQ2StableReferenceError(
                "combined timing uncertainty does not match its sources"
            )
        angular_uncertainty = _finite_float(
            self.combined_angular_uncertainty_rad,
            "combined_angular_uncertainty_rad",
            positive=True,
        )
        if not math.isclose(
            angular_uncertainty,
            expected["combined_angular_uncertainty_rad"],
            rel_tol=0.0,
            abs_tol=_DERIVED_ABS_TOLERANCE,
        ):
            raise VQ2StableReferenceError(
                "combined angular uncertainty does not match its sources"
            )
        for name in (
            "projective_forward",
            "local_area_jacobian_determinant",
            "homography_condition",
            "feature_jacobian_condition",
            "projective_magnification",
        ):
            actual = _finite_float(getattr(self, name), name, positive=True)
            if not math.isclose(
                actual,
                expected[name],
                rel_tol=0.0,
                abs_tol=_DERIVED_ABS_TOLERANCE,
            ):
                raise VQ2StableReferenceError(f"{name} does not match its sources")
            object.__setattr__(self, name, actual)
        if replace(self.output_state) != expected["output_state"]:
            raise VQ2StableReferenceError("output_state does not match its sources")
        object.__setattr__(self, "selected_camera_angular_rate_rad_s", angular_rate)
        object.__setattr__(self, "combined_angular_uncertainty_rad", angular_uncertainty)

    def validate_integrity(self) -> None:
        """Reconstruct all nested values and every derived transform field."""

        self.reference.validate_integrity()
        _validate_derotation_evidence_integrity(self.evidence)
        input_state = replace(self.input_state)
        output_state = replace(self.output_state)
        replace(self, input_state=input_state, output_state=output_state)


def establish_stable_reference(
    *,
    reference_id: str,
    tracker_id: str,
    track_role: TrackRole,
    seed_evidence: VQ2DerotationEvidence,
    model: VQ2StableReferenceModel,
) -> VQ2StableReference:
    """Latch one immutable capture-time orientation/lineage reference."""

    _bounded_token(reference_id, "reference_id")
    _bounded_token(tracker_id, "tracker_id")
    if type(track_role) is not TrackRole:
        raise TypeError("track_role must be exact TrackRole")
    if type(seed_evidence) is not VQ2DerotationEvidence:
        raise TypeError("seed_evidence must be exact VQ2DerotationEvidence")
    if type(model) is not VQ2StableReferenceModel:
        raise TypeError("model must be exact VQ2StableReferenceModel")
    model = replace(model)
    _validate_derotation_evidence_integrity(seed_evidence)
    _validate_reference_observation(seed_evidence)
    key = _build_reference_key(
        seed_evidence,
        tracker_id=tracker_id,
        track_role=track_role,
        stable_model=model,
    )
    orientation, _ = _camera_chart_at(seed_evidence, VQ2CameraFeatureTime.CAPTURE)
    return VQ2StableReference(
        reference_id=reference_id,
        key=key,
        seed_evidence=seed_evidence,
        reference_camera_to_ned_wxyz=orientation,
        reference_angular_uncertainty_rad=(
            seed_evidence.combined_angular_uncertainty_rad
        ),
    )


def camera_to_stable_local_differential(
    reference: VQ2StableReference,
    evidence: VQ2DerotationEvidence,
    state: VQ2LocalDifferentialFeatureState,
    *,
    camera_time: VQ2CameraFeatureTime,
) -> VQ2StableFeatureTransformEvidence:
    """Transform one conditional local feature from camera to stable chart."""

    return _build_transform(
        reference,
        evidence,
        state,
        direction=VQ2StableTransformDirection.CAMERA_TO_STABLE,
        camera_time=camera_time,
    )


def stable_to_camera_local_differential(
    reference: VQ2StableReference,
    evidence: VQ2DerotationEvidence,
    state: VQ2LocalDifferentialFeatureState,
    *,
    camera_time: VQ2CameraFeatureTime,
) -> VQ2StableFeatureTransformEvidence:
    """Transform one conditional local feature from stable to camera chart."""

    return _build_transform(
        reference,
        evidence,
        state,
        direction=VQ2StableTransformDirection.STABLE_TO_CAMERA,
        camera_time=camera_time,
    )


def validate_stable_measurement_sequence(
    reference: VQ2StableReference,
    transforms: tuple[VQ2StableFeatureTransformEvidence, ...],
) -> None:
    """Validate a chronological sequence of capture-camera measurement maps."""

    if type(reference) is not VQ2StableReference:
        raise TypeError("reference must be exact VQ2StableReference")
    if type(transforms) is not tuple:
        raise TypeError("transforms must be an exact tuple")
    reference.validate_integrity()
    previous: VQ2StableFeatureTransformEvidence | None = None
    seen_frames: set[object] = set()
    for transform in transforms:
        if type(transform) is not VQ2StableFeatureTransformEvidence:
            raise TypeError("measurement sequence contains a non-transform value")
        transform.validate_integrity()
        if transform.reference != reference:
            raise VQ2StableReferenceMismatchError(
                "measurement transform belongs to another reference"
            )
        if (
            transform.direction is not VQ2StableTransformDirection.CAMERA_TO_STABLE
            or transform.camera_time is not VQ2CameraFeatureTime.CAPTURE
        ):
            raise VQ2StableReferenceError(
                "measurement sequence accepts only capture camera-to-stable transforms"
            )
        observation = transform.evidence.observation
        if observation.frame in seen_frames:
            raise VQ2StableReferenceChronologyError(
                "measurement sequence repeats a camera frame"
            )
        seen_frames.add(observation.frame)
        if previous is not None:
            _validate_transform_progression(previous, transform)
        previous = transform


def _build_transform(
    reference: VQ2StableReference,
    evidence: VQ2DerotationEvidence,
    state: VQ2LocalDifferentialFeatureState,
    *,
    direction: VQ2StableTransformDirection,
    camera_time: VQ2CameraFeatureTime,
) -> VQ2StableFeatureTransformEvidence:
    if type(reference) is not VQ2StableReference:
        raise TypeError("reference must be exact VQ2StableReference")
    if type(evidence) is not VQ2DerotationEvidence:
        raise TypeError("evidence must be exact VQ2DerotationEvidence")
    if type(state) is not VQ2LocalDifferentialFeatureState:
        raise TypeError("state must be exact VQ2LocalDifferentialFeatureState")
    if type(direction) is not VQ2StableTransformDirection:
        raise TypeError("direction must be exact VQ2StableTransformDirection")
    if type(camera_time) is not VQ2CameraFeatureTime:
        raise TypeError("camera_time must be exact VQ2CameraFeatureTime")
    derived = _derive_transform(
        reference,
        evidence,
        state,
        direction=direction,
        camera_time=camera_time,
    )
    return VQ2StableFeatureTransformEvidence(
        reference=reference,
        evidence=evidence,
        direction=direction,
        camera_time=camera_time,
        input_state=state,
        **derived,
    )


def _derive_transform(
    reference: VQ2StableReference,
    evidence: VQ2DerotationEvidence,
    state: VQ2LocalDifferentialFeatureState,
    *,
    direction: VQ2StableTransformDirection,
    camera_time: VQ2CameraFeatureTime,
) -> dict[str, Any]:
    reference.validate_integrity()
    _validate_derotation_evidence_integrity(evidence)
    state = replace(state)
    _validate_transform_context(reference, evidence, state, direction, camera_time)
    model = reference.key.stable_model
    current_orientation, camera_rate = _camera_chart_at(evidence, camera_time)
    reference_rotation = _quaternion_to_matrix(
        reference.reference_camera_to_ned_wxyz
    )
    current_rotation = _quaternion_to_matrix(current_orientation)
    ray_rotation = reference_rotation.T @ current_rotation
    ray_rotation_rate = ray_rotation @ _skew(camera_rate)
    chart = np.asarray(model.chart_to_camera_ray, dtype=np.float64)
    chart_inverse = np.linalg.inv(chart)
    homography = chart_inverse @ ray_rotation @ chart
    homography_rate = chart_inverse @ ray_rotation_rate @ chart
    if direction is VQ2StableTransformDirection.STABLE_TO_CAMERA:
        inverse = np.linalg.inv(homography)
        homography_rate = -inverse @ homography_rate @ inverse
        homography = inverse
    homography_condition = float(np.linalg.cond(homography))
    if (
        not math.isfinite(homography_condition)
        or homography_condition > model.max_homography_condition
    ):
        raise VQ2StableReferenceGeometryError("homography condition exceeds model")
    determinant = float(np.linalg.det(homography))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise VQ2StableReferenceGeometryError("homography must preserve orientation")
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise VQ2StableReferenceGeometryError(
            "stable homography must retain its determinant-one normalization"
        )
    output_values, state_jacobian, diagnostics = _transform_local_state(
        np.asarray(state.values, dtype=np.float64),
        homography,
        homography_rate,
        model,
    )
    input_covariance = np.asarray(state.covariance, dtype=np.float64)
    coordinate_covariance = _stabilize_psd(
        state_jacobian @ input_covariance @ state_jacobian.T,
        model,
        label="coordinate covariance",
        require_positive_diagonal=True,
    )
    nuisance = np.asarray(model.joint_nuisance_envelope_covariance, dtype=np.float64)
    model_floor = np.asarray(model.model_floor_covariance, dtype=np.float64)
    total_covariance = _stabilize_psd(
        coordinate_covariance + nuisance + model_floor,
        model,
        label="total covariance",
        require_positive_diagonal=True,
    )
    output_basis = (
        VQ2LocalFeatureBasis.STABLE_REFERENCE
        if direction is VQ2StableTransformDirection.CAMERA_TO_STABLE
        else VQ2LocalFeatureBasis.CAMERA
    )
    output_basis_id = (
        reference.basis_id
        if output_basis is VQ2LocalFeatureBasis.STABLE_REFERENCE
        else reference.key.calibration.calibration_id
    )
    output_state = VQ2LocalDifferentialFeatureState(
        feature_model_id=model.local_feature_model_id,
        chart_model_id=model.chart_model_id,
        basis=output_basis,
        basis_id=output_basis_id,
        host_clock_id=reference.key.camera_host_clock_id,
        time_monotonic_ns=state.time_monotonic_ns,
        values=tuple(float(value) for value in output_values),  # type: ignore[arg-type]
        covariance_model_id=model.covariance_model_id,
        covariance_scope=VQ2CovarianceScope.TRANSFORM_TOTAL,
        covariance=_matrix_tuple(total_covariance),  # type: ignore[arg-type]
    )
    combined_timing_uncertainty_ns = (
        reference.seed_evidence.combined_timing_uncertainty_ns
        + evidence.combined_timing_uncertainty_ns
    )
    combined_angular_uncertainty_rad = (
        reference.reference_angular_uncertainty_rad
        + evidence.combined_angular_uncertainty_rad
    )
    return {
        "homography_input_to_output": _matrix_tuple(homography),
        "homography_rate": _matrix_tuple(homography_rate),
        "state_jacobian": _matrix_tuple(state_jacobian),
        "coordinate_covariance": _matrix_tuple(coordinate_covariance),
        "joint_nuisance_envelope_covariance": model.joint_nuisance_envelope_covariance,
        "model_floor_covariance": model.model_floor_covariance,
        "total_covariance": _matrix_tuple(total_covariance),
        "selected_camera_angular_rate_rad_s": camera_rate,
        "combined_timing_uncertainty_ns": combined_timing_uncertainty_ns,
        "combined_angular_uncertainty_rad": combined_angular_uncertainty_rad,
        "homography_condition": homography_condition,
        "output_state": output_state,
        **diagnostics,
    }


def _transform_local_state(
    state: np.ndarray,
    homography: np.ndarray,
    homography_rate: np.ndarray,
    model: VQ2StableReferenceModel,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    u, v, local_scale, u_rate, v_rate, expansion = state
    ray = np.array((1.0, u, v), dtype=np.float64)
    ray_rate = np.array((0.0, u_rate, v_rate), dtype=np.float64)
    mapped = homography @ ray
    mapped_rate = homography_rate @ ray + homography @ ray_rate
    forward, right, down = (float(value) for value in mapped)
    forward_rate, right_rate, down_rate = (float(value) for value in mapped_rate)
    if not math.isfinite(forward) or forward <= model.minimum_forward:
        raise VQ2StableReferenceGeometryError(
            "projective ray does not retain the minimum forward margin"
        )
    output_u = right / forward
    output_v = down / forward
    inverse_forward_sq = 1.0 / (forward * forward)
    output_u_rate = (
        right_rate * forward - right * forward_rate
    ) * inverse_forward_sq
    output_v_rate = (
        down_rate * forward - down * forward_rate
    ) * inverse_forward_sq
    determinant = float(np.linalg.det(homography))
    trace_rate = float(np.trace(np.linalg.solve(homography, homography_rate)))
    alpha = 0.5 * math.log(determinant) - 1.5 * math.log(forward)
    alpha_rate = 0.5 * trace_rate - 1.5 * forward_rate / forward
    output_scale = local_scale + alpha
    output_expansion = expansion + alpha_rate
    output = np.array(
        (
            output_u,
            output_v,
            output_scale,
            output_u_rate,
            output_v_rate,
            output_expansion,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(output)):
        raise VQ2StableReferenceGeometryError("local feature transform became non-finite")
    if (
        abs(output_u) > model.max_abs_bearing_norm
        or abs(output_v) > model.max_abs_bearing_norm
        or abs(output_scale) > model.max_abs_local_log_scale
        or abs(output_u_rate) > model.max_abs_bearing_rate_norm_s
        or abs(output_v_rate) > model.max_abs_bearing_rate_norm_s
        or abs(output_expansion) > model.max_abs_local_expansion_rate_s
    ):
        raise VQ2StableReferenceGeometryError("transformed feature exceeds model bounds")
    columns = homography
    feature_jacobian = np.array(
        (
            (
                (columns[1, 1] * forward - right * columns[0, 1])
                * inverse_forward_sq,
                (columns[1, 2] * forward - right * columns[0, 2])
                * inverse_forward_sq,
            ),
            (
                (columns[2, 1] * forward - down * columns[0, 1])
                * inverse_forward_sq,
                (columns[2, 2] * forward - down * columns[0, 2])
                * inverse_forward_sq,
            ),
        ),
        dtype=np.float64,
    )
    local_area_determinant = float(np.linalg.det(feature_jacobian))
    if not math.isfinite(local_area_determinant) or local_area_determinant <= 0.0:
        raise VQ2StableReferenceGeometryError("local area Jacobian must be positive")
    singular_values = np.linalg.svd(feature_jacobian, compute_uv=False)
    magnification = float(singular_values[0])
    feature_condition = float(np.linalg.cond(feature_jacobian))
    if magnification > model.max_projective_magnification:
        raise VQ2StableReferenceGeometryError("projective magnification exceeds model")
    if (
        not math.isfinite(feature_condition)
        or feature_condition > model.max_feature_jacobian_condition
    ):
        raise VQ2StableReferenceGeometryError("feature Jacobian condition exceeds model")
    state_jacobian = _analytic_state_jacobian(
        state,
        homography,
        homography_rate,
        mapped,
        mapped_rate,
        output,
    )
    return (
        output,
        state_jacobian,
        {
            "projective_forward": forward,
            "local_area_jacobian_determinant": local_area_determinant,
            "feature_jacobian_condition": feature_condition,
            "projective_magnification": magnification,
        },
    )


def _analytic_state_jacobian(
    state: np.ndarray,
    homography: np.ndarray,
    homography_rate: np.ndarray,
    mapped: np.ndarray,
    mapped_rate: np.ndarray,
    output: np.ndarray,
) -> np.ndarray:
    forward, right, down = (float(value) for value in mapped)
    forward_rate, right_rate, down_rate = (float(value) for value in mapped_rate)
    output_u, output_v = float(output[0]), float(output[1])
    inverse_forward_sq = 1.0 / (forward * forward)
    jacobian = np.zeros((6, 6), dtype=np.float64)
    for column in range(6):
        ray_derivative = np.array(
            (
                0.0,
                1.0 if column == 0 else 0.0,
                1.0 if column == 1 else 0.0,
            ),
            dtype=np.float64,
        )
        rate_derivative = np.array(
            (
                0.0,
                1.0 if column == 3 else 0.0,
                1.0 if column == 4 else 0.0,
            ),
            dtype=np.float64,
        )
        mapped_derivative = homography @ ray_derivative
        mapped_rate_derivative = (
            homography_rate @ ray_derivative + homography @ rate_derivative
        )
        d_forward, d_right, d_down = (
            float(value) for value in mapped_derivative
        )
        d_forward_rate, d_right_rate, d_down_rate = (
            float(value) for value in mapped_rate_derivative
        )
        d_u = (
            d_right * forward - right * d_forward
        ) * inverse_forward_sq
        d_v = (
            d_down * forward - down * d_forward
        ) * inverse_forward_sq
        jacobian[0, column] = d_u
        jacobian[1, column] = d_v
        jacobian[2, column] = (
            (1.0 if column == 2 else 0.0) - 1.5 * d_forward / forward
        )
        numerator_u = right_rate - output_u * forward_rate
        numerator_v = down_rate - output_v * forward_rate
        d_numerator_u = (
            d_right_rate - d_u * forward_rate - output_u * d_forward_rate
        )
        d_numerator_v = (
            d_down_rate - d_v * forward_rate - output_v * d_forward_rate
        )
        jacobian[3, column] = (
            d_numerator_u / forward
            - numerator_u * d_forward * inverse_forward_sq
        )
        jacobian[4, column] = (
            d_numerator_v / forward
            - numerator_v * d_forward * inverse_forward_sq
        )
        jacobian[5, column] = (
            (1.0 if column == 5 else 0.0)
            - 1.5
            * (
                d_forward_rate * forward - forward_rate * d_forward
            )
            * inverse_forward_sq
        )
    if not np.all(np.isfinite(jacobian)):
        raise VQ2StableReferenceGeometryError("state Jacobian became non-finite")
    return jacobian


def _validate_transform_context(
    reference: VQ2StableReference,
    evidence: VQ2DerotationEvidence,
    state: VQ2LocalDifferentialFeatureState,
    direction: VQ2StableTransformDirection,
    camera_time: VQ2CameraFeatureTime,
) -> None:
    expected_key = _build_reference_key(
        evidence,
        tracker_id=reference.key.tracker_id,
        track_role=reference.key.track_role,
        stable_model=reference.key.stable_model,
    )
    if expected_key != reference.key:
        raise VQ2StableReferenceMismatchError(
            "transform evidence does not match the stable reference key"
        )
    _validate_authority_snapshot_progression(
        reference.seed_evidence.observation.authority,
        evidence.observation.authority,
    )
    seed_observation = reference.seed_evidence.observation
    observation = evidence.observation
    if observation == seed_observation:
        if evidence != reference.seed_evidence:
            raise VQ2StableReferenceChronologyError(
                "seed observation was relabelled with another evidence context"
            )
    else:
        if observation.frame == seed_observation.frame:
            raise VQ2StableReferenceChronologyError(
                "one reference frame cannot be relabelled as another observation"
            )
        if (
            observation.frame_timing.publication_sequence
            <= seed_observation.frame_timing.publication_sequence
            or observation.frame_timing.camera_source_time_ns
            <= seed_observation.frame_timing.camera_source_time_ns
            or observation.frame_timing.publish_monotonic_ns
            <= seed_observation.frame_timing.publish_monotonic_ns
            or observation.measurement_time_monotonic_ns
            <= seed_observation.measurement_time_monotonic_ns
        ):
            raise VQ2StableReferenceChronologyError(
                "transform observation source/publication/time does not strictly "
                "advance the reference"
            )
        if (
            evidence.prediction_target.decision_time_monotonic_ns
            < reference.seed_evidence.prediction_target.decision_time_monotonic_ns
            or evidence.prediction_target.prediction_time_monotonic_ns
            < reference.seed_evidence.prediction_target.prediction_time_monotonic_ns
        ):
            raise VQ2StableReferenceChronologyError(
                "transform decision or prediction time regressed from the reference"
            )
        _validate_attitude_progression(
            reference.seed_evidence.capture_attitude,
            evidence.capture_attitude,
        )
        _validate_attitude_progression(
            reference.seed_evidence.target_attitude,
            evidence.target_attitude,
        )
    model = reference.key.stable_model
    selected_time = _selected_camera_time(evidence, camera_time)
    reference_time = seed_observation.measurement_time_monotonic_ns
    reference_age = selected_time - reference_time
    if reference_age < 0:
        raise VQ2StableReferenceChronologyError("camera chart predates its reference")
    if reference_age > model.max_reference_age_ns:
        raise VQ2StableReferenceChronologyError("stable reference age exceeds model")
    combined_timing_uncertainty = (
        reference.seed_evidence.combined_timing_uncertainty_ns
        + evidence.combined_timing_uncertainty_ns
    )
    if combined_timing_uncertainty > model.max_total_timing_uncertainty_ns:
        raise VQ2StableReferenceError("combined timing uncertainty exceeds stable model")
    combined_angular_uncertainty = (
        reference.reference_angular_uncertainty_rad
        + evidence.combined_angular_uncertainty_rad
    )
    if combined_angular_uncertainty > model.max_relative_angular_uncertainty_rad:
        raise VQ2StableReferenceError("combined angular uncertainty exceeds stable model")
    _, camera_rate = _camera_chart_at(evidence, camera_time)
    if _norm3(camera_rate) > model.max_angular_rate_rad_s:
        raise VQ2StableReferenceError("camera angular rate exceeds stable model")
    if (
        evidence.model.angular_rate_uncertainty_rad_s
        > model.max_angular_rate_uncertainty_rad_s
    ):
        raise VQ2StableReferenceError(
            "angular-rate uncertainty exceeds stable model"
        )
    if state.feature_model_id != model.local_feature_model_id:
        raise VQ2StableReferenceMismatchError("state feature semantic changed")
    if state.chart_model_id != model.chart_model_id:
        raise VQ2StableReferenceMismatchError("state chart model changed")
    if state.covariance_model_id != model.covariance_model_id:
        raise VQ2StableReferenceMismatchError("state covariance model changed")
    if state.host_clock_id != reference.key.camera_host_clock_id:
        raise VQ2StableReferenceMismatchError("state host clock changed")
    if state.time_monotonic_ns != selected_time:
        raise VQ2StableReferenceMismatchError("state time does not match camera basis")
    if state.covariance_scope is not VQ2CovarianceScope.CONDITIONAL_INPUT:
        raise VQ2StableReferenceError(
            "transform input must exclude previously added nuisance envelopes"
        )
    expected_basis = (
        VQ2LocalFeatureBasis.CAMERA
        if direction is VQ2StableTransformDirection.CAMERA_TO_STABLE
        else VQ2LocalFeatureBasis.STABLE_REFERENCE
    )
    expected_basis_id = (
        reference.key.calibration.calibration_id
        if expected_basis is VQ2LocalFeatureBasis.CAMERA
        else reference.basis_id
    )
    if state.basis is not expected_basis or state.basis_id != expected_basis_id:
        raise VQ2StableReferenceMismatchError("state basis does not match transform")
    value_bounds = (
        model.max_abs_bearing_norm,
        model.max_abs_bearing_norm,
        model.max_abs_local_log_scale,
        model.max_abs_bearing_rate_norm_s,
        model.max_abs_bearing_rate_norm_s,
        model.max_abs_local_expansion_rate_s,
    )
    if any(abs(value) > bound for value, bound in zip(state.values, value_bounds)):
        raise VQ2StableReferenceError("input feature exceeds stable model bounds")


def _validate_derotation_evidence_integrity(
    evidence: VQ2DerotationEvidence,
) -> None:
    """Deeply revalidate public evidence, including nested `/1` contracts."""

    if type(evidence) is not VQ2DerotationEvidence:
        raise TypeError("evidence must be exact VQ2DerotationEvidence")
    evidence.validate_integrity()
    reconstructed = GateObservationV1.from_primitive(
        evidence.observation.to_primitive()
    )
    if reconstructed != evidence.observation:
        raise VQ2StableReferenceError(
            "observation public-schema round trip changed evidence"
        )


def _validate_reference_observation(evidence: VQ2DerotationEvidence) -> None:
    observation = evidence.observation
    if observation.health is ObservationHealth.UNUSABLE:
        raise VQ2StableReferenceError("unusable observation cannot anchor stable math")
    if observation.clipping != FrameEdge.NONE:
        raise VQ2StableReferenceError("stable reference requires an unclipped aperture")
    if (
        observation.fitted_inner_aperture_corners_norm is None
        or observation.log_scale is None
        or observation.projective_skew is None
        or observation.geometry_model_id is None
        or any(corner is None for corner in observation.inner_corners_norm)
        or observation.inner_edges.visibility != _ALL_FRAME_EDGES
    ):
        raise VQ2StableReferenceError(
            "stable reference requires a complete all-visible fitted aperture"
        )


def _build_reference_key(
    evidence: VQ2DerotationEvidence,
    *,
    tracker_id: str,
    track_role: TrackRole,
    stable_model: VQ2StableReferenceModel,
) -> VQ2StableReferenceKey:
    observation = evidence.observation
    authority = observation.authority
    return VQ2StableReferenceKey(
        tracker_id=tracker_id,
        track_role=track_role,
        session_id=authority.session_id,
        reset_epoch=authority.reset_epoch,
        gate_epoch=authority.gate_epoch,
        expected_gate_index=authority.expected_gate_index,
        camera_host_clock_id=authority.camera_host_clock_id,
        camera_stream_id=authority.camera_stream_id,
        camera_generation=authority.camera_generation,
        image_size_px=observation.image_size_px,
        imu_epoch_key=evidence.capture_attitude.attitude.source.epoch_key,
        measurement_time_basis=observation.measurement_time_basis,
        measurement_time_model_id=observation.measurement_time_model_id,
        calibration=evidence.calibration,
        derotation_model=evidence.model,
        stable_model=stable_model,
    )


def _reference_basis_id(
    reference_id: str,
    key: VQ2StableReferenceKey,
    seed_evidence: VQ2DerotationEvidence,
    orientation: QuaternionTuple,
) -> str:
    """Fingerprint the complete local chart identity without creating a wire ID."""

    material = repr((reference_id, key, seed_evidence, orientation)).encode("utf-8")
    return f"vq2-stable:{hashlib.sha256(material).hexdigest()}"


def _validate_authority_snapshot_progression(previous: Any, current: Any) -> None:
    if (
        current.session_id != previous.session_id
        or current.reset_epoch != previous.reset_epoch
        or current.gate_epoch != previous.gate_epoch
        or current.expected_gate_index != previous.expected_gate_index
        or current.camera_host_clock_id != previous.camera_host_clock_id
        or current.camera_stream_id != previous.camera_stream_id
        or current.camera_generation != previous.camera_generation
    ):
        raise VQ2StableReferenceMismatchError("stable authority lifecycle changed")
    if (
        current.race_status_sequence < previous.race_status_sequence
        or current.race_status_boot_ms < previous.race_status_boot_ms
        or current.frame_publication_sequence_not_before
        < previous.frame_publication_sequence_not_before
        or current.frame_publish_monotonic_ns_not_before
        < previous.frame_publish_monotonic_ns_not_before
    ):
        raise VQ2StableReferenceChronologyError("authority snapshot regressed")
    if (
        current.race_status_sequence == previous.race_status_sequence
        and current.race_status_boot_ms != previous.race_status_boot_ms
    ):
        raise VQ2StableReferenceChronologyError(
            "one race-status sequence cannot carry multiple boot times"
        )


def _validate_attitude_progression(previous: Any, current: Any) -> None:
    if current == previous:
        return
    previous_attitude = previous.attitude
    current_attitude = current.attitude
    if current_attitude.source.epoch_key != previous_attitude.source.epoch_key:
        raise VQ2StableReferenceMismatchError("IMU epoch changed")
    if current_attitude.sample_sequence == previous_attitude.sample_sequence:
        raise VQ2StableReferenceChronologyError(
            "one IMU sample identity was relabelled"
        )
    if (
        current_attitude.sample_sequence <= previous_attitude.sample_sequence
        or current_attitude.source_time_us <= previous_attitude.source_time_us
        or current_attitude.receive_monotonic_ns
        <= previous_attitude.receive_monotonic_ns
    ):
        raise VQ2StableReferenceChronologyError(
            "IMU sample/source/receipt chronology did not advance coherently"
        )


def _validate_transform_progression(
    previous: VQ2StableFeatureTransformEvidence,
    current: VQ2StableFeatureTransformEvidence,
) -> None:
    previous_observation = previous.evidence.observation
    current_observation = current.evidence.observation
    if (
        current_observation.frame_timing.publication_sequence
        <= previous_observation.frame_timing.publication_sequence
        or current_observation.frame_timing.camera_source_time_ns
        <= previous_observation.frame_timing.camera_source_time_ns
        or current_observation.frame_timing.publish_monotonic_ns
        <= previous_observation.frame_timing.publish_monotonic_ns
        or current_observation.measurement_time_monotonic_ns
        <= previous_observation.measurement_time_monotonic_ns
    ):
        raise VQ2StableReferenceChronologyError(
            "measurement source/publication/time did not advance strictly"
        )
    if (
        current.evidence.prediction_target.decision_time_monotonic_ns
        < previous.evidence.prediction_target.decision_time_monotonic_ns
        or current.evidence.prediction_target.prediction_time_monotonic_ns
        < previous.evidence.prediction_target.prediction_time_monotonic_ns
    ):
        raise VQ2StableReferenceChronologyError(
            "measurement decision/prediction time regressed"
        )
    _validate_authority_snapshot_progression(
        previous_observation.authority,
        current_observation.authority,
    )
    _validate_attitude_progression(
        previous.evidence.capture_attitude,
        current.evidence.capture_attitude,
    )
    _validate_attitude_progression(
        previous.evidence.target_attitude,
        current.evidence.target_attitude,
    )


def _selected_camera_time(
    evidence: VQ2DerotationEvidence,
    camera_time: VQ2CameraFeatureTime,
) -> int:
    if camera_time is VQ2CameraFeatureTime.CAPTURE:
        return evidence.observation.measurement_time_monotonic_ns
    return evidence.prediction_target.prediction_time_monotonic_ns


def _camera_chart_at(
    evidence: VQ2DerotationEvidence,
    camera_time: VQ2CameraFeatureTime,
) -> tuple[QuaternionTuple, Vector3]:
    if camera_time is VQ2CameraFeatureTime.CAPTURE:
        attitude_input = evidence.capture_attitude
        delta_ns = evidence.capture_alignment_ns
    else:
        attitude_input = evidence.target_attitude
        delta_ns = evidence.target_extrapolation_ns
    body_orientation = _propagate_body_to_ned(
        attitude_input.attitude.orientation_body_to_ned_wxyz,
        attitude_input.attitude.body_rates_rad_s,
        delta_ns * 1e-9,
    )
    # The lower-level calibration accepts unit quaternions within a numerical
    # tolerance.  Normalize once here so both orientation composition and rate
    # rotation use exactly the same proper rotation at that accepted boundary.
    camera_to_body = _normalize_quaternion(
        evidence.calibration.camera_to_body_wxyz
    )
    camera_to_ned = _normalize_quaternion(
        _quat_multiply(body_orientation, camera_to_body)
    )
    camera_rate = _quat_rotate(
        _quat_conjugate(camera_to_body),
        attitude_input.attitude.body_rates_rad_s,
    )
    return camera_to_ned, camera_rate


def _stabilize_psd(
    covariance: np.ndarray,
    model: VQ2StableReferenceModel,
    *,
    label: str,
    require_positive_diagonal: bool,
) -> np.ndarray:
    result = np.asarray(covariance, dtype=np.float64)
    if result.shape != (6, 6) or not np.all(np.isfinite(result)):
        raise VQ2StableReferenceError(f"{label} must be finite 6x6")
    result = (result + result.T) * 0.5
    # Preserve scale-aware rejection for very small covariances.  A zero
    # congruence can still be lifted by the explicit eigenvalue floor below,
    # but a negative eigenvalue is compared with the matrix's own scale.
    scale = float(np.max(np.abs(result)))
    minimum = float(np.linalg.eigvalsh(result)[0])
    tolerance = model.covariance_psd_tolerance * scale
    if minimum < -tolerance:
        raise VQ2StableReferenceError(f"{label} is materially indefinite")
    if minimum < model.covariance_eigenvalue_floor:
        result += np.eye(6) * (model.covariance_eigenvalue_floor - minimum)
    result = (result + result.T) * 0.5
    if require_positive_diagonal and np.any(np.diag(result) <= 0.0):
        raise VQ2StableReferenceError(f"{label} variances must be positive")
    return result


def _skew(value: Vector3) -> np.ndarray:
    x, y, z = value
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


def _quaternion_to_matrix(value: QuaternionTuple) -> np.ndarray:
    quaternion = _unit_quaternion(value, "orientation quaternion")
    return np.column_stack(
        (
            _quat_rotate(quaternion, (1.0, 0.0, 0.0)),
            _quat_rotate(quaternion, (0.0, 1.0, 0.0)),
            _quat_rotate(quaternion, (0.0, 0.0, 1.0)),
        )
    )


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
            (
                1.0,
                0.5 * rotation_vector[0],
                0.5 * rotation_vector[1],
                0.5 * rotation_vector[2],
            )
        )
    scale = math.sin(0.5 * angle) / angle
    return (
        math.cos(0.5 * angle),
        scale * rotation_vector[0],
        scale * rotation_vector[1],
        scale * rotation_vector[2],
    )


def _quat_multiply(
    first: QuaternionTuple,
    second: QuaternionTuple,
) -> QuaternionTuple:
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
    rotated = _quat_multiply(
        _quat_multiply(quaternion, pure),
        _quat_conjugate(quaternion),
    )
    return (rotated[1], rotated[2], rotated[3])


def _normalize_quaternion(value: QuaternionTuple) -> QuaternionTuple:
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise VQ2StableReferenceError("quaternion propagation became invalid")
    return tuple(component / norm for component in value)  # type: ignore[return-value]


def _norm3(value: tuple[float, ...]) -> float:
    return math.sqrt(sum(component * component for component in value))


__all__ = [
    "HARD_MAX_ABS_BEARING_NORM",
    "HARD_MAX_ABS_BEARING_RATE_NORM_S",
    "HARD_MAX_ABS_LOCAL_EXPANSION_RATE_S",
    "HARD_MAX_ABS_LOCAL_LOG_SCALE",
    "HARD_MAX_ANGULAR_ACCELERATION_RAD_S2",
    "HARD_MAX_ANGULAR_RATE_RAD_S",
    "HARD_MAX_ANGULAR_RATE_UNCERTAINTY_RAD_S",
    "HARD_MAX_FEATURE_ACCELERATION_NORM_S2",
    "HARD_MAX_REFERENCE_AGE_NS",
    "HARD_MAX_RELATIVE_ANGULAR_UNCERTAINTY_RAD",
    "HARD_MAX_TOTAL_TIMING_UNCERTAINTY_NS",
    "IDENTITY_CHART_TO_CAMERA_RAY",
    "LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID",
    "LOCAL_FEATURE_ORDER",
    "SUPPORTED_STABLE_CHART_MODEL_ID",
    "VQ2CameraFeatureTime",
    "VQ2CovarianceScope",
    "VQ2LocalDifferentialFeatureState",
    "VQ2LocalFeatureBasis",
    "VQ2StableFeatureTransformEvidence",
    "VQ2StableReference",
    "VQ2StableReferenceChronologyError",
    "VQ2StableReferenceError",
    "VQ2StableReferenceGeometryError",
    "VQ2StableReferenceKey",
    "VQ2StableReferenceMismatchError",
    "VQ2StableReferenceModel",
    "VQ2StableTransformDirection",
    "camera_to_stable_local_differential",
    "establish_stable_reference",
    "stable_to_camera_local_differential",
    "validate_stable_measurement_sequence",
]
