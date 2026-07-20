"""Pure rectified-homography reducer for the VQ2 local-scale semantic.

This proof-only module starts after an external producer has supplied a
center-gauge homography and dense conditional covariance.  It deliberately
does not accept the frozen VQ2 observation, image corners, a feature state, or
any estimator/runtime authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

from competition.vq2_contracts import (
    FrameTimingV1,
    GateAuthorityEpochV1,
    MeasurementTimeBasis,
)


Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Vector8 = tuple[float, float, float, float, float, float, float, float]
Matrix2 = tuple[Vector2, Vector2]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Matrix8 = tuple[
    Vector8,
    Vector8,
    Vector8,
    Vector8,
    Vector8,
    Vector8,
    Vector8,
    Vector8,
]
Matrix3x8 = tuple[Vector8, Vector8, Vector8]
ProjectedQuad = tuple[Vector2, Vector2, Vector2, Vector2]


LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID = "vq2-local-differential-area-v1"
CANONICAL_APERTURE_MODEL_ID = "vq2-canonical-square-aperture-v1"
RECTIFIED_CHART_MODEL_ID = "vq2-rectified-camera-frd-slope-chart-v1"
HOMOGRAPHY_GAUGE_MODEL_ID = "vq2-center-forward-h00-one-v1"
CANONICAL_CORNER_ORDER = (
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
)
CANONICAL_CORNERS: ProjectedQuad = (
    (-1.0, -1.0),
    (1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
)
HOMOGRAPHY_PARAMETER_ORDER = (
    "h01",
    "h02",
    "h10",
    "h11",
    "h12",
    "h20",
    "h21",
    "h22",
)
LOCAL_MEASUREMENT_ORDER = (
    "bearing_x_norm",
    "bearing_y_norm",
    "local_log_scale",
)

HARD_MAX_MEASUREMENT_UNCERTAINTY_NS = 200_000_000
HARD_MIN_FORWARD = 1e-6
HARD_MIN_LOCAL_AREA_DETERMINANT = 1e-12
HARD_MIN_PROJECTED_EDGE_LENGTH = 1e-6
HARD_MIN_PROJECTED_CORNER_CROSS = 1e-12
HARD_MAX_HOMOGRAPHY_CONDITION = 1e6
HARD_MAX_LOCAL_DIFFERENTIAL_CONDITION = 1e6
HARD_MAX_ABS_CENTER_BEARING_NORM = 4.0
HARD_MAX_ABS_PROJECTED_CORNER_BEARING_NORM = 4.0
HARD_MAX_ABS_LOCAL_LOG_SCALE = 20.0
HARD_MAX_INPUT_VARIANCE = 1e6
HARD_MAX_OUTPUT_VARIANCE = 1e6
HARD_MAX_COVARIANCE_PSD_TOLERANCE = 1e-10

_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CAMERA_TIME_BASES = {
    MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
    MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
}

_MODEL_FLOAT_FIELDS = (
    "minimum_forward",
    "minimum_local_area_determinant",
    "minimum_projected_edge_length",
    "minimum_projected_corner_cross",
    "max_homography_condition",
    "max_local_differential_condition",
    "max_abs_center_bearing_norm",
    "max_abs_projected_corner_bearing_norm",
    "max_abs_local_log_scale",
    "max_input_variance",
    "max_output_variance",
    "covariance_psd_tolerance",
)

_EVIDENCE_SCALAR_FLOAT_FIELDS = (
    "local_area_jacobian_determinant",
    "minimum_canonical_forward",
    "homography_condition",
    "local_differential_condition",
    "minimum_projected_edge_length",
    "minimum_projected_corner_cross",
    "maximum_projected_abs_bearing",
)


class VQ2LocalDifferentialMeasurementError(ValueError):
    """Base error for invalid local measurement evidence."""


class VQ2LocalDifferentialProvenanceError(VQ2LocalDifferentialMeasurementError):
    """Raised when declared source/model lineage does not bind exactly."""


class VQ2LocalDifferentialGeometryError(VQ2LocalDifferentialMeasurementError):
    """Raised when the projective geometry leaves the frozen envelope."""


class VQ2LocalDifferentialCovarianceError(VQ2LocalDifferentialMeasurementError):
    """Raised when conditional covariance evidence is malformed."""


class VQ2HomographyCovarianceScope(str, Enum):
    CONDITIONAL_FIT = "conditional_fit"


def _bounded_token(value: object, label: str) -> str:
    if type(value) is not str or _TOKEN_RE.fullmatch(value) is None:
        raise VQ2LocalDifferentialProvenanceError(
            f"{label} must be a bounded token"
        )
    return value


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise VQ2LocalDifferentialProvenanceError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
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
        raise VQ2LocalDifferentialMeasurementError(
            f"{label} must be >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise VQ2LocalDifferentialMeasurementError(
            f"{label} must be <= {maximum}"
        )
    return value


def _finite_float(
    value: object,
    label: str,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise VQ2LocalDifferentialMeasurementError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise VQ2LocalDifferentialMeasurementError(f"{label} must be positive")
    if minimum is not None and result < minimum:
        raise VQ2LocalDifferentialMeasurementError(
            f"{label} must be >= {minimum}"
        )
    if maximum is not None and result > maximum:
        raise VQ2LocalDifferentialMeasurementError(
            f"{label} must be <= {maximum}"
        )
    return result


def _exact_image_size(value: object, label: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{label} must be an exact two-tuple")
    width = _exact_int(value[0], f"{label}[0]", minimum=1)
    height = _exact_int(value[1], f"{label}[1]", minimum=1)
    return width, height


def _exact_string_tuple(
    value: object,
    expected: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) != len(expected):
        raise TypeError(f"{label} must be an exact {len(expected)}-tuple")
    if any(type(component) is not str for component in value):
        raise TypeError(f"{label} entries must be exact strings")
    if value != expected:
        raise VQ2LocalDifferentialProvenanceError(f"unsupported {label}")
    return value


def _vector(value: object, length: int, label: str) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise TypeError(f"{label} must be an exact {length}-tuple")
    return tuple(
        _finite_float(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )


def _matrix(
    value: object,
    rows: int,
    columns: int,
    label: str,
) -> tuple[tuple[float, ...], ...]:
    if type(value) is not tuple or len(value) != rows:
        raise TypeError(f"{label} must be an exact {rows}x{columns} tuple")
    return tuple(
        _vector(row, columns, f"{label}[{index}]")
        for index, row in enumerate(value)
    )


def _matrix_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(component) for component in row) for row in value)


def _assert_float_matrix_storage(
    value: object,
    rows: int,
    columns: int,
    label: str,
) -> None:
    if type(value) is not tuple or len(value) != rows:
        raise VQ2LocalDifferentialProvenanceError(
            f"{label} lost its canonical tuple storage"
        )
    for row_index, row in enumerate(value):
        if type(row) is not tuple or len(row) != columns:
            raise VQ2LocalDifferentialProvenanceError(
                f"{label}[{row_index}] lost its canonical tuple storage"
            )
        if any(type(component) is not float for component in row):
            raise VQ2LocalDifferentialProvenanceError(
                f"{label} entries must remain exact stored floats"
            )


def _assert_model_storage(
    model: "VQ2RectifiedHomographyMeasurementModel",
) -> None:
    for name in _MODEL_FLOAT_FIELDS:
        if type(getattr(model, name)) is not float:
            raise VQ2LocalDifferentialProvenanceError(
                f"model.{name} must remain an exact stored float"
            )


def _assert_source_storage(source: "VQ2RectifiedHomographyInput") -> None:
    _assert_float_matrix_storage(source.homography, 3, 3, "source.homography")
    _assert_float_matrix_storage(
        source.homography_covariance,
        8,
        8,
        "source.homography_covariance",
    )


def _assert_evidence_storage(
    evidence: "VQ2RectifiedHomographyMeasurementEvidence",
) -> None:
    if type(evidence.values) is not tuple or len(evidence.values) != 3 or any(
        type(component) is not float for component in evidence.values
    ):
        raise VQ2LocalDifferentialProvenanceError(
            "evidence.values must remain exact stored floats"
        )
    for name, rows, columns in (
        ("projected_canonical_corners", 4, 2),
        ("local_differential", 2, 2),
        ("measurement_jacobian", 3, 8),
        ("canonical_homography_covariance", 8, 8),
        ("conditional_covariance", 3, 3),
    ):
        _assert_float_matrix_storage(
            getattr(evidence, name),
            rows,
            columns,
            f"evidence.{name}",
        )
    for name in _EVIDENCE_SCALAR_FLOAT_FIELDS:
        if type(getattr(evidence, name)) is not float:
            raise VQ2LocalDifferentialProvenanceError(
                f"evidence.{name} must remain an exact stored float"
            )


def _covariance(
    value: object,
    dimension: int,
    label: str,
    *,
    relative_tolerance: float,
    max_variance: float,
    canonicalize: bool,
) -> tuple[tuple[float, ...], ...]:
    rows = _matrix(value, dimension, dimension, label)
    array = np.asarray(rows, dtype=np.float64)
    scale = float(np.max(np.abs(array)))
    if scale <= 0.0:
        raise VQ2LocalDifferentialCovarianceError(
            f"{label} must have positive marginals"
        )
    tolerance = relative_tolerance * scale
    asymmetry = float(np.max(np.abs(array - array.T)))
    if asymmetry > tolerance:
        raise VQ2LocalDifferentialCovarianceError(
            f"{label} exceeds its symmetry tolerance"
        )
    symmetric = (array + array.T) * 0.5
    diagonal = np.diag(symmetric)
    if np.any(diagonal <= 0.0):
        raise VQ2LocalDifferentialCovarianceError(
            f"{label} marginals must be strictly positive"
        )
    if np.any(diagonal > max_variance):
        raise VQ2LocalDifferentialCovarianceError(
            f"{label} exceeds its marginal variance limit"
        )
    minimum_eigenvalue = float(np.linalg.eigvalsh(symmetric)[0])
    if minimum_eigenvalue < -tolerance:
        raise VQ2LocalDifferentialCovarianceError(
            f"{label} is materially indefinite"
        )
    return _matrix_tuple(symmetric if canonicalize else array)


def _matrix_close(
    actual: tuple[tuple[float, ...], ...],
    expected: tuple[tuple[float, ...], ...],
) -> bool:
    if len(actual) != len(expected) or any(
        len(actual_row) != len(expected_row)
        for actual_row, expected_row in zip(actual, expected)
    ):
        return False
    return all(
        actual_value.hex() == expected_value.hex()
        for actual_row, expected_row in zip(actual, expected)
        for actual_value, expected_value in zip(actual_row, expected_row)
    )


def _vector_close(actual: tuple[float, ...], expected: tuple[float, ...]) -> bool:
    return len(actual) == len(expected) and all(
        actual_value.hex() == expected_value.hex()
        for actual_value, expected_value in zip(actual, expected)
    )


def _camera_time_model(
    basis: object,
    model_id: object,
    *,
    label: str,
) -> tuple[MeasurementTimeBasis, str | None]:
    if type(basis) is not MeasurementTimeBasis:
        raise TypeError(f"{label}.measurement_time_basis must be exact MeasurementTimeBasis")
    if basis not in _CAMERA_TIME_BASES:
        raise VQ2LocalDifferentialProvenanceError(
            f"{label} requires a camera measurement-time basis"
        )
    if model_id is not None:
        _bounded_token(model_id, f"{label}.measurement_time_model_id")
    if basis is MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED:
        if model_id is None:
            raise VQ2LocalDifferentialProvenanceError(
                f"{label} calibrated camera time requires a model id"
            )
    elif model_id is not None:
        raise VQ2LocalDifferentialProvenanceError(
            f"{label} final-packet proxy cannot claim a model id"
        )
    return basis, model_id  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class VQ2RectifiedHomographyMeasurementModel:
    model_id: str
    local_feature_model_id: str
    canonical_aperture_model_id: str
    canonical_corner_order: tuple[str, str, str, str]
    rectified_chart_model_id: str
    homography_gauge_model_id: str
    homography_parameter_order: tuple[str, str, str, str, str, str, str, str]
    image_size_px: tuple[int, int]
    measurement_time_basis: MeasurementTimeBasis
    measurement_time_model_id: str | None
    geometry_producer_model_id: str
    geometry_producer_config_sha256: str
    homography_fit_model_id: str
    rectification_model_id: str
    rectification_calibration_id: str
    rectification_calibration_sha256: str
    homography_covariance_model_id: str
    output_covariance_model_id: str
    max_measurement_uncertainty_ns: int
    minimum_forward: float
    minimum_local_area_determinant: float
    minimum_projected_edge_length: float
    minimum_projected_corner_cross: float
    max_homography_condition: float
    max_local_differential_condition: float
    max_abs_center_bearing_norm: float
    max_abs_projected_corner_bearing_norm: float
    max_abs_local_log_scale: float
    max_input_variance: float
    max_output_variance: float
    covariance_psd_tolerance: float

    def __post_init__(self) -> None:
        _bounded_token(self.model_id, "model_id")
        _bounded_token(self.local_feature_model_id, "local_feature_model_id")
        if self.local_feature_model_id != LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID:
            raise VQ2LocalDifferentialProvenanceError(
                "unsupported local feature semantic"
            )
        _bounded_token(
            self.canonical_aperture_model_id,
            "canonical_aperture_model_id",
        )
        if self.canonical_aperture_model_id != CANONICAL_APERTURE_MODEL_ID:
            raise VQ2LocalDifferentialProvenanceError(
                "unsupported canonical aperture model"
            )
        _exact_string_tuple(
            self.canonical_corner_order,
            CANONICAL_CORNER_ORDER,
            "canonical_corner_order",
        )
        _bounded_token(self.rectified_chart_model_id, "rectified_chart_model_id")
        if self.rectified_chart_model_id != RECTIFIED_CHART_MODEL_ID:
            raise VQ2LocalDifferentialProvenanceError(
                "unsupported rectified chart model"
            )
        _bounded_token(self.homography_gauge_model_id, "homography_gauge_model_id")
        if self.homography_gauge_model_id != HOMOGRAPHY_GAUGE_MODEL_ID:
            raise VQ2LocalDifferentialProvenanceError(
                "unsupported homography gauge model"
            )
        _exact_string_tuple(
            self.homography_parameter_order,
            HOMOGRAPHY_PARAMETER_ORDER,
            "homography_parameter_order",
        )
        image_size = _exact_image_size(self.image_size_px, "image_size_px")
        basis, time_model = _camera_time_model(
            self.measurement_time_basis,
            self.measurement_time_model_id,
            label="model",
        )
        for name in (
            "geometry_producer_model_id",
            "homography_fit_model_id",
            "rectification_model_id",
            "rectification_calibration_id",
            "homography_covariance_model_id",
            "output_covariance_model_id",
        ):
            _bounded_token(getattr(self, name), name)
        _sha256(
            self.geometry_producer_config_sha256,
            "geometry_producer_config_sha256",
        )
        _sha256(
            self.rectification_calibration_sha256,
            "rectification_calibration_sha256",
        )
        maximum_uncertainty = _exact_int(
            self.max_measurement_uncertainty_ns,
            "max_measurement_uncertainty_ns",
            minimum=1,
            maximum=HARD_MAX_MEASUREMENT_UNCERTAINTY_NS,
        )
        minimum_fields = (
            ("minimum_forward", HARD_MIN_FORWARD),
            ("minimum_local_area_determinant", HARD_MIN_LOCAL_AREA_DETERMINANT),
            ("minimum_projected_edge_length", HARD_MIN_PROJECTED_EDGE_LENGTH),
            ("minimum_projected_corner_cross", HARD_MIN_PROJECTED_CORNER_CROSS),
        )
        for name, hard_minimum in minimum_fields:
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=hard_minimum),
            )
        maximum_fields = (
            ("max_homography_condition", 1.0, HARD_MAX_HOMOGRAPHY_CONDITION),
            (
                "max_local_differential_condition",
                1.0,
                HARD_MAX_LOCAL_DIFFERENTIAL_CONDITION,
            ),
            (
                "max_abs_center_bearing_norm",
                0.0,
                HARD_MAX_ABS_CENTER_BEARING_NORM,
            ),
            (
                "max_abs_projected_corner_bearing_norm",
                0.0,
                HARD_MAX_ABS_PROJECTED_CORNER_BEARING_NORM,
            ),
            ("max_abs_local_log_scale", 0.0, HARD_MAX_ABS_LOCAL_LOG_SCALE),
            ("max_input_variance", 0.0, HARD_MAX_INPUT_VARIANCE),
            ("max_output_variance", 0.0, HARD_MAX_OUTPUT_VARIANCE),
        )
        for name, lower, hard_maximum in maximum_fields:
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name,
                    positive=lower == 0.0,
                    minimum=None if lower == 0.0 else lower,
                    maximum=hard_maximum,
                ),
            )
        tolerance = _finite_float(
            self.covariance_psd_tolerance,
            "covariance_psd_tolerance",
            positive=True,
            maximum=HARD_MAX_COVARIANCE_PSD_TOLERANCE,
        )
        object.__setattr__(self, "image_size_px", image_size)
        object.__setattr__(self, "measurement_time_basis", basis)
        object.__setattr__(self, "measurement_time_model_id", time_model)
        object.__setattr__(
            self, "max_measurement_uncertainty_ns", maximum_uncertainty
        )
        object.__setattr__(self, "covariance_psd_tolerance", tolerance)


def _roundtrip_frame(value: object) -> FrameTimingV1:
    if type(value) is not FrameTimingV1:
        raise TypeError("frame_timing must be exact FrameTimingV1")
    try:
        rebuilt = FrameTimingV1.from_primitive(value.to_primitive())
    except Exception as error:
        raise VQ2LocalDifferentialProvenanceError(
            "frame_timing failed public-codec reconstruction"
        ) from error
    if rebuilt != value:
        raise VQ2LocalDifferentialProvenanceError(
            "frame_timing changed during public-codec reconstruction"
        )
    return rebuilt


def _roundtrip_authority(value: object) -> GateAuthorityEpochV1:
    if type(value) is not GateAuthorityEpochV1:
        raise TypeError("authority must be exact GateAuthorityEpochV1")
    try:
        rebuilt = GateAuthorityEpochV1.from_primitive(value.to_primitive())
    except Exception as error:
        raise VQ2LocalDifferentialProvenanceError(
            "authority failed public-codec reconstruction"
        ) from error
    if rebuilt != value:
        raise VQ2LocalDifferentialProvenanceError(
            "authority changed during public-codec reconstruction"
        )
    return rebuilt


def _validate_authority_binding(
    frame: FrameTimingV1,
    authority: GateAuthorityEpochV1,
) -> None:
    if frame.host_clock_id != authority.camera_host_clock_id:
        raise VQ2LocalDifferentialProvenanceError(
            "frame host clock does not match authority"
        )
    if frame.identity.stream_id != authority.camera_stream_id:
        raise VQ2LocalDifferentialProvenanceError(
            "frame stream does not match authority"
        )
    if frame.identity.generation != authority.camera_generation:
        raise VQ2LocalDifferentialProvenanceError(
            "frame generation does not match authority"
        )
    if (
        frame.publication_sequence
        < authority.frame_publication_sequence_not_before
    ):
        raise VQ2LocalDifferentialProvenanceError(
            "frame predates authority publication watermark"
        )
    if (
        frame.publish_monotonic_ns
        < authority.frame_publish_monotonic_ns_not_before
    ):
        raise VQ2LocalDifferentialProvenanceError(
            "frame predates authority host-time watermark"
        )


@dataclass(frozen=True, slots=True)
class VQ2RectifiedHomographyInput:
    frame_timing: FrameTimingV1
    authority: GateAuthorityEpochV1
    measurement_time_monotonic_ns: int
    measurement_time_basis: MeasurementTimeBasis
    measurement_time_model_id: str | None
    measurement_uncertainty_ns: int
    candidate_id: str
    image_size_px: tuple[int, int]
    canonical_aperture_model_id: str
    rectified_chart_model_id: str
    homography_gauge_model_id: str
    geometry_producer_model_id: str
    geometry_producer_config_sha256: str
    homography_fit_model_id: str
    rectification_model_id: str
    rectification_calibration_id: str
    rectification_calibration_sha256: str
    homography_covariance_model_id: str
    homography_covariance_scope: VQ2HomographyCovarianceScope
    homography_parameter_order: tuple[str, str, str, str, str, str, str, str]
    homography: Matrix3
    homography_covariance: Matrix8

    def __post_init__(self) -> None:
        frame = _roundtrip_frame(self.frame_timing)
        authority = _roundtrip_authority(self.authority)
        _validate_authority_binding(frame, authority)
        measurement_time = _exact_int(
            self.measurement_time_monotonic_ns,
            "measurement_time_monotonic_ns",
            minimum=0,
        )
        basis, time_model = _camera_time_model(
            self.measurement_time_basis,
            self.measurement_time_model_id,
            label="source",
        )
        uncertainty = _exact_int(
            self.measurement_uncertainty_ns,
            "measurement_uncertainty_ns",
            minimum=1,
            maximum=HARD_MAX_MEASUREMENT_UNCERTAINTY_NS,
        )
        if basis is MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY and (
            measurement_time != frame.final_unique_packet_monotonic_ns
        ):
            raise VQ2LocalDifferentialProvenanceError(
                "final-packet proxy must equal the final-packet host time"
            )
        if measurement_time > frame.publish_monotonic_ns:
            raise VQ2LocalDifferentialProvenanceError(
                "measurement time cannot postdate frame publication"
            )
        _bounded_token(self.candidate_id, "candidate_id")
        image_size = _exact_image_size(self.image_size_px, "image_size_px")
        _bounded_token(
            self.canonical_aperture_model_id,
            "canonical_aperture_model_id",
        )
        if self.canonical_aperture_model_id != CANONICAL_APERTURE_MODEL_ID:
            raise VQ2LocalDifferentialProvenanceError(
                "source has the wrong canonical aperture model"
            )
        _bounded_token(self.rectified_chart_model_id, "rectified_chart_model_id")
        if self.rectified_chart_model_id != RECTIFIED_CHART_MODEL_ID:
            raise VQ2LocalDifferentialProvenanceError(
                "source has the wrong rectified chart model"
            )
        _bounded_token(self.homography_gauge_model_id, "homography_gauge_model_id")
        if self.homography_gauge_model_id != HOMOGRAPHY_GAUGE_MODEL_ID:
            raise VQ2LocalDifferentialProvenanceError(
                "source has the wrong homography gauge model"
            )
        for name in (
            "geometry_producer_model_id",
            "homography_fit_model_id",
            "rectification_model_id",
            "rectification_calibration_id",
            "homography_covariance_model_id",
        ):
            _bounded_token(getattr(self, name), name)
        _sha256(
            self.geometry_producer_config_sha256,
            "geometry_producer_config_sha256",
        )
        _sha256(
            self.rectification_calibration_sha256,
            "rectification_calibration_sha256",
        )
        if type(self.homography_covariance_scope) is not VQ2HomographyCovarianceScope:
            raise TypeError(
                "homography_covariance_scope must be exact "
                "VQ2HomographyCovarianceScope"
            )
        if (
            self.homography_covariance_scope
            is not VQ2HomographyCovarianceScope.CONDITIONAL_FIT
        ):
            raise VQ2LocalDifferentialCovarianceError(
                "only conditional-fit covariance is accepted"
            )
        _exact_string_tuple(
            self.homography_parameter_order,
            HOMOGRAPHY_PARAMETER_ORDER,
            "homography_parameter_order",
        )
        homography = _matrix(self.homography, 3, 3, "homography")
        if homography[0][0] != 1.0:
            raise VQ2LocalDifferentialGeometryError(
                "homography must use exact center-forward H[0,0]=1 gauge"
            )
        covariance = _covariance(
            self.homography_covariance,
            8,
            "homography_covariance",
            relative_tolerance=HARD_MAX_COVARIANCE_PSD_TOLERANCE,
            max_variance=HARD_MAX_INPUT_VARIANCE,
            canonicalize=False,
        )
        object.__setattr__(self, "frame_timing", frame)
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "measurement_time_monotonic_ns", measurement_time)
        object.__setattr__(self, "measurement_time_basis", basis)
        object.__setattr__(self, "measurement_time_model_id", time_model)
        object.__setattr__(self, "measurement_uncertainty_ns", uncertainty)
        object.__setattr__(self, "image_size_px", image_size)
        object.__setattr__(self, "homography", homography)
        object.__setattr__(self, "homography_covariance", covariance)


def _model_primitive(model: VQ2RectifiedHomographyMeasurementModel) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "local_feature_model_id": model.local_feature_model_id,
        "canonical_aperture_model_id": model.canonical_aperture_model_id,
        "canonical_corner_order": list(model.canonical_corner_order),
        "rectified_chart_model_id": model.rectified_chart_model_id,
        "homography_gauge_model_id": model.homography_gauge_model_id,
        "homography_parameter_order": list(model.homography_parameter_order),
        "image_size_px": list(model.image_size_px),
        "measurement_time_basis": model.measurement_time_basis.value,
        "measurement_time_model_id": model.measurement_time_model_id,
        "geometry_producer_model_id": model.geometry_producer_model_id,
        "geometry_producer_config_sha256": model.geometry_producer_config_sha256,
        "homography_fit_model_id": model.homography_fit_model_id,
        "rectification_model_id": model.rectification_model_id,
        "rectification_calibration_id": model.rectification_calibration_id,
        "rectification_calibration_sha256": model.rectification_calibration_sha256,
        "homography_covariance_model_id": model.homography_covariance_model_id,
        "output_covariance_model_id": model.output_covariance_model_id,
        "max_measurement_uncertainty_ns": model.max_measurement_uncertainty_ns,
        "minimum_forward": model.minimum_forward,
        "minimum_local_area_determinant": model.minimum_local_area_determinant,
        "minimum_projected_edge_length": model.minimum_projected_edge_length,
        "minimum_projected_corner_cross": model.minimum_projected_corner_cross,
        "max_homography_condition": model.max_homography_condition,
        "max_local_differential_condition": model.max_local_differential_condition,
        "max_abs_center_bearing_norm": model.max_abs_center_bearing_norm,
        "max_abs_projected_corner_bearing_norm": (
            model.max_abs_projected_corner_bearing_norm
        ),
        "max_abs_local_log_scale": model.max_abs_local_log_scale,
        "max_input_variance": model.max_input_variance,
        "max_output_variance": model.max_output_variance,
        "covariance_psd_tolerance": model.covariance_psd_tolerance,
    }


def _source_primitive(source: VQ2RectifiedHomographyInput) -> dict[str, Any]:
    return {
        "frame_timing": source.frame_timing.to_primitive(),
        "authority": source.authority.to_primitive(),
        "measurement_time_monotonic_ns": source.measurement_time_monotonic_ns,
        "measurement_time_basis": source.measurement_time_basis.value,
        "measurement_time_model_id": source.measurement_time_model_id,
        "measurement_uncertainty_ns": source.measurement_uncertainty_ns,
        "candidate_id": source.candidate_id,
        "image_size_px": list(source.image_size_px),
        "canonical_aperture_model_id": source.canonical_aperture_model_id,
        "rectified_chart_model_id": source.rectified_chart_model_id,
        "homography_gauge_model_id": source.homography_gauge_model_id,
        "geometry_producer_model_id": source.geometry_producer_model_id,
        "geometry_producer_config_sha256": source.geometry_producer_config_sha256,
        "homography_fit_model_id": source.homography_fit_model_id,
        "rectification_model_id": source.rectification_model_id,
        "rectification_calibration_id": source.rectification_calibration_id,
        "rectification_calibration_sha256": source.rectification_calibration_sha256,
        "homography_covariance_model_id": source.homography_covariance_model_id,
        "homography_covariance_scope": source.homography_covariance_scope.value,
        "homography_parameter_order": list(source.homography_parameter_order),
        "homography": [list(row) for row in source.homography],
        "homography_covariance": [
            list(row) for row in source.homography_covariance
        ],
    }


def _fingerprint(domain: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"domain": domain, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _input_fingerprint(source: VQ2RectifiedHomographyInput) -> str:
    return _fingerprint("vq2-rectified-homography-input-v1", _source_primitive(source))


def _derivation_fingerprint(
    model: VQ2RectifiedHomographyMeasurementModel,
    source: VQ2RectifiedHomographyInput,
) -> str:
    return _fingerprint(
        "vq2-local-differential-derivation-v1",
        {"model": _model_primitive(model), "source": _source_primitive(source)},
    )


def _clone_source(source: object) -> VQ2RectifiedHomographyInput:
    if type(source) is not VQ2RectifiedHomographyInput:
        raise TypeError("source must be exact VQ2RectifiedHomographyInput")
    _assert_source_storage(source)
    frame = _roundtrip_frame(source.frame_timing)
    authority = _roundtrip_authority(source.authority)
    return replace(source, frame_timing=frame, authority=authority)


def _validated_model(model: object) -> VQ2RectifiedHomographyMeasurementModel:
    if type(model) is not VQ2RectifiedHomographyMeasurementModel:
        raise TypeError(
            "model must be exact VQ2RectifiedHomographyMeasurementModel"
        )
    _assert_model_storage(model)
    return replace(model)


def _bind_model_source(
    model: VQ2RectifiedHomographyMeasurementModel,
    source: VQ2RectifiedHomographyInput,
) -> tuple[VQ2RectifiedHomographyInput, Matrix8]:
    matches = (
        ("image_size_px", model.image_size_px, source.image_size_px),
        (
            "measurement_time_basis",
            model.measurement_time_basis,
            source.measurement_time_basis,
        ),
        (
            "measurement_time_model_id",
            model.measurement_time_model_id,
            source.measurement_time_model_id,
        ),
        (
            "canonical_aperture_model_id",
            model.canonical_aperture_model_id,
            source.canonical_aperture_model_id,
        ),
        (
            "rectified_chart_model_id",
            model.rectified_chart_model_id,
            source.rectified_chart_model_id,
        ),
        (
            "homography_gauge_model_id",
            model.homography_gauge_model_id,
            source.homography_gauge_model_id,
        ),
        (
            "homography_parameter_order",
            model.homography_parameter_order,
            source.homography_parameter_order,
        ),
        (
            "geometry_producer_model_id",
            model.geometry_producer_model_id,
            source.geometry_producer_model_id,
        ),
        (
            "geometry_producer_config_sha256",
            model.geometry_producer_config_sha256,
            source.geometry_producer_config_sha256,
        ),
        (
            "homography_fit_model_id",
            model.homography_fit_model_id,
            source.homography_fit_model_id,
        ),
        (
            "rectification_model_id",
            model.rectification_model_id,
            source.rectification_model_id,
        ),
        (
            "rectification_calibration_id",
            model.rectification_calibration_id,
            source.rectification_calibration_id,
        ),
        (
            "rectification_calibration_sha256",
            model.rectification_calibration_sha256,
            source.rectification_calibration_sha256,
        ),
        (
            "homography_covariance_model_id",
            model.homography_covariance_model_id,
            source.homography_covariance_model_id,
        ),
    )
    for label, expected, actual in matches:
        if actual != expected:
            raise VQ2LocalDifferentialProvenanceError(
                f"source {label} does not match the reducer model"
            )
    if source.measurement_uncertainty_ns > model.max_measurement_uncertainty_ns:
        raise VQ2LocalDifferentialProvenanceError(
            "source measurement uncertainty exceeds the reducer model"
        )
    covariance = _covariance(
        source.homography_covariance,
        8,
        "homography_covariance",
        relative_tolerance=model.covariance_psd_tolerance,
        max_variance=model.max_input_variance,
        canonicalize=True,
    )
    return source, covariance  # type: ignore[return-value]


def _projected_geometry(
    homography: np.ndarray,
    *,
    minimum_forward: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    projected: list[tuple[float, float]] = []
    forwards: list[float] = []
    for canonical_x, canonical_y in CANONICAL_CORNERS:
        mapped = homography @ np.asarray(
            (1.0, canonical_x, canonical_y), dtype=np.float64
        )
        forward = float(mapped[0])
        if not math.isfinite(forward) or forward < minimum_forward:
            raise VQ2LocalDifferentialGeometryError(
                "canonical aperture leaves the positive-forward envelope"
            )
        forwards.append(forward)
        projected.append((float(mapped[1] / forward), float(mapped[2] / forward)))
    points = np.asarray(projected, dtype=np.float64)
    edges = np.roll(points, -1, axis=0) - points
    edge_lengths = np.linalg.norm(edges, axis=1)
    crosses = np.asarray(
        [
            edges[index, 0] * edges[(index + 1) % 4, 1]
            - edges[index, 1] * edges[(index + 1) % 4, 0]
            for index in range(4)
        ],
        dtype=np.float64,
    )
    return (
        points,
        np.asarray(forwards, dtype=np.float64),
        float(np.min(edge_lengths)),
        float(np.min(crosses)),
        float(np.max(np.abs(points))),
    )


def _derive_fields(
    model: VQ2RectifiedHomographyMeasurementModel,
    source: VQ2RectifiedHomographyInput,
    canonical_homography_covariance: Matrix8,
) -> dict[str, Any]:
    homography = np.asarray(source.homography, dtype=np.float64)
    h01, h02 = homography[0, 1], homography[0, 2]
    h10, h11, h12 = homography[1]
    h20, h21, h22 = homography[2]
    local = np.asarray(
        (
            (h11 - h10 * h01, h12 - h10 * h02),
            (h21 - h20 * h01, h22 - h20 * h02),
        ),
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(local))
    if not math.isfinite(determinant) or (
        determinant < model.minimum_local_area_determinant
    ):
        raise VQ2LocalDifferentialGeometryError(
            "local area determinant is outside the reducer model"
        )
    local_log_scale = math.log(2.0) + 0.5 * math.log(determinant)
    if abs(float(h10)) > model.max_abs_center_bearing_norm or abs(
        float(h20)
    ) > model.max_abs_center_bearing_norm:
        raise VQ2LocalDifferentialGeometryError(
            "center bearing exceeds the reducer model"
        )
    if abs(local_log_scale) > model.max_abs_local_log_scale:
        raise VQ2LocalDifferentialGeometryError(
            "local log scale exceeds the reducer model"
        )

    points, forwards, edge_length, corner_cross, max_corner = _projected_geometry(
        homography,
        minimum_forward=model.minimum_forward,
    )
    minimum_forward = float(np.min(forwards))
    if minimum_forward < model.minimum_forward:
        raise VQ2LocalDifferentialGeometryError(
            "canonical aperture leaves the positive-forward envelope"
        )
    if edge_length < model.minimum_projected_edge_length:
        raise VQ2LocalDifferentialGeometryError(
            "projected canonical edge collapses"
        )
    if corner_cross < model.minimum_projected_corner_cross:
        raise VQ2LocalDifferentialGeometryError(
            "projected canonical quad is not positively convex and ordered"
        )
    if max_corner > model.max_abs_projected_corner_bearing_norm:
        raise VQ2LocalDifferentialGeometryError(
            "projected canonical bearing exceeds the reducer model"
        )
    homography_condition = float(np.linalg.cond(homography, p=2))
    local_condition = float(np.linalg.cond(local, p=2))
    if not math.isfinite(homography_condition) or (
        homography_condition > model.max_homography_condition
    ):
        raise VQ2LocalDifferentialGeometryError(
            "homography condition exceeds the reducer model"
        )
    if not math.isfinite(local_condition) or (
        local_condition > model.max_local_differential_condition
    ):
        raise VQ2LocalDifferentialGeometryError(
            "local differential condition exceeds the reducer model"
        )

    a, b = float(local[0, 0]), float(local[0, 1])
    c, d = float(local[1, 0]), float(local[1, 1])
    delta_gradient = np.asarray(
        (
            -h10 * d + b * h20,
            -a * h20 + h10 * c,
            -h01 * d + h02 * c,
            d,
            -c,
            -a * h02 + b * h01,
            -b,
            a,
        ),
        dtype=np.float64,
    )
    jacobian = np.zeros((3, 8), dtype=np.float64)
    jacobian[0, 2] = 1.0
    jacobian[1, 5] = 1.0
    jacobian[2] = 0.5 * delta_gradient / determinant

    input_covariance = np.asarray(
        canonical_homography_covariance,
        dtype=np.float64,
    )
    conditional = jacobian @ input_covariance @ jacobian.T
    conditional_tuple = _covariance(
        _matrix_tuple(conditional),
        3,
        "conditional_covariance",
        relative_tolerance=model.covariance_psd_tolerance,
        max_variance=model.max_output_variance,
        canonicalize=True,
    )
    return {
        "feature_model_id": model.local_feature_model_id,
        "rectified_chart_model_id": model.rectified_chart_model_id,
        "host_clock_id": source.frame_timing.host_clock_id,
        "measurement_time_monotonic_ns": source.measurement_time_monotonic_ns,
        "measurement_time_basis": source.measurement_time_basis,
        "measurement_time_model_id": source.measurement_time_model_id,
        "measurement_uncertainty_ns": source.measurement_uncertainty_ns,
        "covariance_model_id": model.output_covariance_model_id,
        "covariance_scope": VQ2HomographyCovarianceScope.CONDITIONAL_FIT,
        "input_fingerprint_sha256": _input_fingerprint(source),
        "derivation_fingerprint_sha256": _derivation_fingerprint(model, source),
        "measurement_order": LOCAL_MEASUREMENT_ORDER,
        "values": (float(h10), float(h20), float(local_log_scale)),
        "projected_canonical_corners": _matrix_tuple(points),
        "local_differential": _matrix_tuple(local),
        "local_area_jacobian_determinant": determinant,
        "measurement_jacobian": _matrix_tuple(jacobian),
        "canonical_homography_covariance": canonical_homography_covariance,
        "conditional_covariance": conditional_tuple,
        "minimum_canonical_forward": minimum_forward,
        "homography_condition": homography_condition,
        "local_differential_condition": local_condition,
        "minimum_projected_edge_length": edge_length,
        "minimum_projected_corner_cross": corner_cross,
        "maximum_projected_abs_bearing": max_corner,
    }


@dataclass(frozen=True, slots=True)
class VQ2RectifiedHomographyMeasurementEvidence:
    model: VQ2RectifiedHomographyMeasurementModel
    source: VQ2RectifiedHomographyInput
    feature_model_id: str
    rectified_chart_model_id: str
    host_clock_id: str
    measurement_time_monotonic_ns: int
    measurement_time_basis: MeasurementTimeBasis
    measurement_time_model_id: str | None
    measurement_uncertainty_ns: int
    covariance_model_id: str
    covariance_scope: VQ2HomographyCovarianceScope
    input_fingerprint_sha256: str
    derivation_fingerprint_sha256: str
    measurement_order: tuple[str, str, str]
    values: Vector3
    projected_canonical_corners: ProjectedQuad
    local_differential: Matrix2
    local_area_jacobian_determinant: float
    measurement_jacobian: Matrix3x8
    canonical_homography_covariance: Matrix8
    conditional_covariance: Matrix3
    minimum_canonical_forward: float
    homography_condition: float
    local_differential_condition: float
    minimum_projected_edge_length: float
    minimum_projected_corner_cross: float
    maximum_projected_abs_bearing: float

    def __post_init__(self) -> None:
        model = _validated_model(self.model)
        source, canonical_covariance = _bind_model_source(
            model,
            _clone_source(self.source),
        )
        expected = _derive_fields(model, source, canonical_covariance)
        for name in (
            "feature_model_id",
            "rectified_chart_model_id",
            "host_clock_id",
            "covariance_model_id",
        ):
            _bounded_token(getattr(self, name), name)
            if getattr(self, name) != expected[name]:
                raise VQ2LocalDifferentialProvenanceError(
                    f"{name} does not match the retained model/source"
                )
        if self.measurement_time_model_id is not None:
            _bounded_token(
                self.measurement_time_model_id,
                "measurement_time_model_id",
            )
        if self.measurement_time_model_id != expected["measurement_time_model_id"]:
            raise VQ2LocalDifferentialProvenanceError(
                "measurement_time_model_id does not match the retained source"
            )
        for name in (
            "input_fingerprint_sha256",
            "derivation_fingerprint_sha256",
        ):
            _sha256(getattr(self, name), name)
            if getattr(self, name) != expected[name]:
                raise VQ2LocalDifferentialProvenanceError(
                    f"{name} does not match the retained model/source"
                )
        if type(self.measurement_time_basis) is not MeasurementTimeBasis or (
            self.measurement_time_basis is not expected["measurement_time_basis"]
        ):
            raise VQ2LocalDifferentialProvenanceError(
                "measurement_time_basis does not match the retained source"
            )
        if type(self.covariance_scope) is not VQ2HomographyCovarianceScope or (
            self.covariance_scope is not expected["covariance_scope"]
        ):
            raise VQ2LocalDifferentialCovarianceError(
                "covariance_scope does not match conditional evidence"
            )
        for name in (
            "measurement_time_monotonic_ns",
            "measurement_uncertainty_ns",
        ):
            _exact_int(getattr(self, name), name, minimum=0)
            if getattr(self, name) != expected[name]:
                raise VQ2LocalDifferentialProvenanceError(
                    f"{name} does not match the retained source"
                )
        _exact_string_tuple(
            self.measurement_order,
            LOCAL_MEASUREMENT_ORDER,
            "measurement_order",
        )
        values = _vector(self.values, 3, "values")
        if not _vector_close(values, expected["values"]):
            raise VQ2LocalDifferentialGeometryError(
                "values do not match the retained homography"
            )
        matrix_fields = (
            ("projected_canonical_corners", 4, 2),
            ("local_differential", 2, 2),
            ("measurement_jacobian", 3, 8),
            ("canonical_homography_covariance", 8, 8),
            ("conditional_covariance", 3, 3),
        )
        for name, rows, columns in matrix_fields:
            matrix = _matrix(getattr(self, name), rows, columns, name)
            if not _matrix_close(matrix, expected[name]):
                error_type = (
                    VQ2LocalDifferentialCovarianceError
                    if name
                    in {
                        "canonical_homography_covariance",
                        "conditional_covariance",
                    }
                    else VQ2LocalDifferentialGeometryError
                )
                raise error_type(f"{name} does not match its retained sources")
        scalar_fields = (
            "local_area_jacobian_determinant",
            "minimum_canonical_forward",
            "homography_condition",
            "local_differential_condition",
            "minimum_projected_edge_length",
            "minimum_projected_corner_cross",
            "maximum_projected_abs_bearing",
        )
        for name in scalar_fields:
            value = _finite_float(getattr(self, name), name)
            if value.hex() != expected[name].hex():
                raise VQ2LocalDifferentialGeometryError(
                    f"{name} does not match its retained sources"
                )
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "source", source)
        for name, value in expected.items():
            object.__setattr__(self, name, value)

    def validate_integrity(self) -> None:
        """Deeply reconstruct public lineage and rederive every output field."""

        _assert_evidence_storage(self)
        model = _validated_model(self.model)
        source = _clone_source(self.source)
        replace(self, model=model, source=source)


def derive_local_differential_measurement(
    source: VQ2RectifiedHomographyInput,
    *,
    model: VQ2RectifiedHomographyMeasurementModel,
) -> VQ2RectifiedHomographyMeasurementEvidence:
    """Reduce one external full homography to indivisible conditional evidence."""

    validated_model = _validated_model(model)
    validated_source, canonical_covariance = _bind_model_source(
        validated_model,
        _clone_source(source),
    )
    derived = _derive_fields(
        validated_model,
        validated_source,
        canonical_covariance,
    )
    return VQ2RectifiedHomographyMeasurementEvidence(
        model=validated_model,
        source=validated_source,
        **derived,
    )
