from __future__ import annotations

import ast
import dataclasses
import inspect
import math
from pathlib import Path
from typing import get_type_hints

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
    MeasurementTimeBasis,
    ObservationHealth,
    RelativeGateStateV1,
)
from estimation import vq2_local_differential_measurement as subject
from estimation.vq2_imu_derotation import VQ2DerotationEvidence
from estimation.vq2_local_differential_measurement import (
    CANONICAL_APERTURE_MODEL_ID,
    CANONICAL_CORNER_ORDER,
    CANONICAL_CORNERS,
    HARD_MAX_ABS_CENTER_BEARING_NORM,
    HARD_MAX_ABS_LOCAL_LOG_SCALE,
    HARD_MAX_ABS_PROJECTED_CORNER_BEARING_NORM,
    HARD_MAX_COVARIANCE_PSD_TOLERANCE,
    HARD_MAX_HOMOGRAPHY_CONDITION,
    HARD_MAX_INPUT_VARIANCE,
    HARD_MAX_LOCAL_DIFFERENTIAL_CONDITION,
    HARD_MAX_MEASUREMENT_UNCERTAINTY_NS,
    HARD_MAX_OUTPUT_VARIANCE,
    HARD_MIN_FORWARD,
    HARD_MIN_LOCAL_AREA_DETERMINANT,
    HARD_MIN_PROJECTED_CORNER_CROSS,
    HARD_MIN_PROJECTED_EDGE_LENGTH,
    HOMOGRAPHY_GAUGE_MODEL_ID,
    HOMOGRAPHY_PARAMETER_ORDER,
    LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID,
    LOCAL_MEASUREMENT_ORDER,
    RECTIFIED_CHART_MODEL_ID,
    VQ2HomographyCovarianceScope,
    VQ2LocalDifferentialCovarianceError,
    VQ2LocalDifferentialGeometryError,
    VQ2LocalDifferentialMeasurementError,
    VQ2LocalDifferentialProvenanceError,
    VQ2RectifiedHomographyInput,
    VQ2RectifiedHomographyMeasurementEvidence,
    VQ2RectifiedHomographyMeasurementModel,
    derive_local_differential_measurement,
)
from estimation.vq2_stable_reference import (
    VQ2CovarianceScope,
    VQ2LocalDifferentialFeatureState,
)


_BASE_NS = 50_000_000_000
_HOST_CLOCK_ID = "wave3e-host-clock"
_STREAM_ID = "camera0"
_GENERATION = 5
_IMAGE_SIZE = (640, 360)
_CONFIG_HASH = "1" * 64
_CALIBRATION_HASH = "2" * 64
_OTHER_HASH = "3" * 64
_BASE_H = (
    (1.0, 0.08, -0.06),
    (0.20, 0.70, 0.10),
    (-0.10, 0.03, 0.65),
)


def _matrix_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _base_covariance(scale: float = 2.0e-4) -> tuple[tuple[float, ...], ...]:
    rng = np.random.default_rng(20260719)
    factor = rng.normal(size=(8, 8))
    matrix = factor @ factor.T
    matrix /= float(np.max(np.diag(matrix)))
    matrix = scale * (matrix + 2.0 * np.eye(8))
    return _matrix_tuple(matrix)


def _frame_timing(
    *,
    host_clock_id: str = _HOST_CLOCK_ID,
    stream_id: str = _STREAM_ID,
    generation: int = _GENERATION,
    frame_id: int = 17,
    publication_sequence: int = 31,
    final_packet_ns: int = _BASE_NS,
    publish_ns: int = _BASE_NS + 4_000_000,
) -> FrameTimingV1:
    return FrameTimingV1(
        identity=FrameIdentityV1(
            stream_id=stream_id,
            generation=generation,
            frame_id=frame_id,
        ),
        camera_source_time_ns=123_456_789,
        host_clock_id=host_clock_id,
        publication_sequence=publication_sequence,
        first_unique_packet_monotonic_ns=final_packet_ns - 3_000_000,
        final_unique_packet_monotonic_ns=final_packet_ns,
        reassembly_complete_monotonic_ns=final_packet_ns + 500_000,
        decode_start_monotonic_ns=final_packet_ns + 1_000_000,
        decode_end_monotonic_ns=final_packet_ns + 2_000_000,
        publish_monotonic_ns=publish_ns,
    )


def _authority(
    *,
    host_clock_id: str = _HOST_CLOCK_ID,
    stream_id: str = _STREAM_ID,
    generation: int = _GENERATION,
    cutover_sequence: int = 30,
    cutover_ns: int = _BASE_NS + 3_000_000,
) -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id="wave3e-training-session",
        reset_epoch=4,
        gate_epoch=2,
        expected_gate_index=1,
        race_status_sequence=91,
        race_status_boot_ms=8_000,
        camera_host_clock_id=host_clock_id,
        camera_stream_id=stream_id,
        camera_generation=generation,
        frame_publication_sequence_not_before=cutover_sequence,
        frame_publish_monotonic_ns_not_before=cutover_ns,
    )


def _model(**overrides: object) -> VQ2RectifiedHomographyMeasurementModel:
    values: dict[str, object] = {
        "model_id": "wave3e-local-reducer-v1",
        "local_feature_model_id": LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID,
        "canonical_aperture_model_id": CANONICAL_APERTURE_MODEL_ID,
        "canonical_corner_order": CANONICAL_CORNER_ORDER,
        "rectified_chart_model_id": RECTIFIED_CHART_MODEL_ID,
        "homography_gauge_model_id": HOMOGRAPHY_GAUGE_MODEL_ID,
        "homography_parameter_order": HOMOGRAPHY_PARAMETER_ORDER,
        "image_size_px": _IMAGE_SIZE,
        "measurement_time_basis": MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        "measurement_time_model_id": None,
        "geometry_producer_model_id": "external-full-homography-producer-v1",
        "geometry_producer_config_sha256": _CONFIG_HASH,
        "homography_fit_model_id": "conditional-homography-fit-v1",
        "rectification_model_id": "rectified-camera-slopes-v1",
        "rectification_calibration_id": "camera0-rectification-calibration-v1",
        "rectification_calibration_sha256": _CALIBRATION_HASH,
        "homography_covariance_model_id": "dense-homography-fit-covariance-v1",
        "output_covariance_model_id": "local-differential-congruence-v1",
        "max_measurement_uncertainty_ns": 50_000_000,
        "minimum_forward": 1.0e-5,
        "minimum_local_area_determinant": 1.0e-10,
        "minimum_projected_edge_length": 1.0e-5,
        "minimum_projected_corner_cross": 1.0e-10,
        "max_homography_condition": 100_000.0,
        "max_local_differential_condition": 100_000.0,
        "max_abs_center_bearing_norm": 3.5,
        "max_abs_projected_corner_bearing_norm": 3.5,
        "max_abs_local_log_scale": 10.0,
        "max_input_variance": 10.0,
        "max_output_variance": 10.0,
        "covariance_psd_tolerance": HARD_MAX_COVARIANCE_PSD_TOLERANCE,
    }
    values.update(overrides)
    return VQ2RectifiedHomographyMeasurementModel(**values)  # type: ignore[arg-type]


def _source(**overrides: object) -> VQ2RectifiedHomographyInput:
    values: dict[str, object] = {
        "frame_timing": _frame_timing(),
        "authority": _authority(),
        "measurement_time_monotonic_ns": _BASE_NS,
        "measurement_time_basis": MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        "measurement_time_model_id": None,
        "measurement_uncertainty_ns": 20_000_000,
        "candidate_id": "gate-candidate-17",
        "image_size_px": _IMAGE_SIZE,
        "canonical_aperture_model_id": CANONICAL_APERTURE_MODEL_ID,
        "rectified_chart_model_id": RECTIFIED_CHART_MODEL_ID,
        "homography_gauge_model_id": HOMOGRAPHY_GAUGE_MODEL_ID,
        "geometry_producer_model_id": "external-full-homography-producer-v1",
        "geometry_producer_config_sha256": _CONFIG_HASH,
        "homography_fit_model_id": "conditional-homography-fit-v1",
        "rectification_model_id": "rectified-camera-slopes-v1",
        "rectification_calibration_id": "camera0-rectification-calibration-v1",
        "rectification_calibration_sha256": _CALIBRATION_HASH,
        "homography_covariance_model_id": "dense-homography-fit-covariance-v1",
        "homography_covariance_scope": VQ2HomographyCovarianceScope.CONDITIONAL_FIT,
        "homography_parameter_order": HOMOGRAPHY_PARAMETER_ORDER,
        "homography": _BASE_H,
        "homography_covariance": _base_covariance(),
    }
    values.update(overrides)
    return VQ2RectifiedHomographyInput(**values)  # type: ignore[arg-type]


def _result(
    *,
    source: VQ2RectifiedHomographyInput | None = None,
    model: VQ2RectifiedHomographyMeasurementModel | None = None,
) -> VQ2RectifiedHomographyMeasurementEvidence:
    return derive_local_differential_measurement(
        source or _source(),
        model=model or _model(),
    )


def _theta(homography: tuple[tuple[float, ...], ...]) -> np.ndarray:
    h = np.asarray(homography, dtype=np.float64)
    return np.array(
        [h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2]],
        dtype=np.float64,
    )


def _homography(theta: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, float(theta[0]), float(theta[1])),
        (float(theta[2]), float(theta[3]), float(theta[4])),
        (float(theta[5]), float(theta[6]), float(theta[7])),
    )


def _project(homography: np.ndarray, a: float, b: float) -> np.ndarray:
    ray = homography @ np.array((1.0, a, b), dtype=np.float64)
    return ray[1:] / ray[0]


def _oracle(homography: tuple[tuple[float, ...], ...]) -> dict[str, object]:
    h = np.asarray(homography, dtype=np.float64)
    center_ray = h[:, 0]
    center = center_ray[1:] / center_ray[0]
    differential_columns = []
    for column in (1, 2):
        direction = h[:, column]
        differential_columns.append(
            (
                direction[1] * center_ray[0] - center_ray[1] * direction[0]
            )
            / center_ray[0] ** 2
        )
        differential_columns.append(
            (
                direction[2] * center_ray[0] - center_ray[2] * direction[0]
            )
            / center_ray[0] ** 2
        )
    local = np.array(
        (
            (differential_columns[0], differential_columns[2]),
            (differential_columns[1], differential_columns[3]),
        ),
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(local))
    corners = np.array([_project(h, a, b) for a, b in CANONICAL_CORNERS])
    edges = np.roll(corners, -1, axis=0) - corners
    successor_edges = np.roll(edges, -1, axis=0)
    crosses = (
        edges[:, 0] * successor_edges[:, 1]
        - edges[:, 1] * successor_edges[:, 0]
    )
    values = np.array(
        (center[0], center[1], math.log(2.0) + 0.5 * math.log(determinant))
    )
    forwards = np.array(
        [(h @ np.array((1.0, a, b), dtype=np.float64))[0] for a, b in CANONICAL_CORNERS]
    )
    return {
        "values": values,
        "corners": corners,
        "local": local,
        "determinant": determinant,
        "minimum_forward": float(np.min(forwards)),
        "minimum_edge": float(np.min(np.linalg.norm(edges, axis=1))),
        "minimum_cross": float(np.min(crosses)),
        "maximum_corner": float(np.max(np.abs(corners))),
        "homography_condition": float(np.linalg.cond(h)),
        "local_condition": float(np.linalg.cond(local)),
    }


def _finite_difference_jacobian(
    homography: tuple[tuple[float, ...], ...],
    *,
    step: float = 1.0e-6,
) -> np.ndarray:
    theta = _theta(homography)
    columns = []
    for index in range(8):
        offset = np.zeros(8, dtype=np.float64)
        offset[index] = step
        plus = np.asarray(_oracle(_homography(theta + offset))["values"])
        minus = np.asarray(_oracle(_homography(theta - offset))["values"])
        columns.append((plus - minus) / (2.0 * step))
    return np.column_stack(columns)


def _polygon_area(corners: np.ndarray) -> float:
    return 0.5 * abs(
        sum(
            corners[index, 0] * corners[(index + 1) % 4, 1]
            - corners[(index + 1) % 4, 0] * corners[index, 1]
            for index in range(4)
        )
    )


def _summary_vector(theta: np.ndarray) -> np.ndarray:
    oracle = _oracle(_homography(theta))
    corners = np.asarray(oracle["corners"])
    lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    return np.array(
        (
            float(np.asarray(oracle["values"])[0]),
            float(np.asarray(oracle["values"])[1]),
            0.5 * math.log(_polygon_area(corners)),
            math.log(lengths[1] / lengths[3]),
            math.log(lengths[2] / lengths[0]),
        )
    )


def _summary_jacobian(theta: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    columns = []
    for index in range(8):
        offset = np.zeros(8)
        offset[index] = step
        columns.append(
            (_summary_vector(theta + offset) - _summary_vector(theta - offset))
            / (2.0 * step)
        )
    return np.column_stack(columns)


def _local_scale_gradient(theta: np.ndarray, step: float = 1.0e-6) -> np.ndarray:
    values = []
    for index in range(8):
        offset = np.zeros(8)
        offset[index] = step
        plus = float(np.asarray(_oracle(_homography(theta + offset))["values"])[2])
        minus = float(np.asarray(_oracle(_homography(theta - offset))["values"])[2])
        values.append((plus - minus) / (2.0 * step))
    return np.asarray(values)


def _replace_matrix_entry(
    matrix: tuple[tuple[float, ...], ...],
    row: int,
    column: int,
    value: object,
) -> tuple[tuple[object, ...], ...]:
    rows = [list(item) for item in matrix]
    rows[row][column] = value
    return tuple(tuple(item) for item in rows)


def _assert_rejected(callable_: object) -> None:
    with pytest.raises((TypeError, ValueError, VQ2LocalDifferentialMeasurementError)):
        callable_()  # type: ignore[operator]


def test_public_surface_and_constants_are_exact() -> None:
    assert LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID == "vq2-local-differential-area-v1"
    assert CANONICAL_APERTURE_MODEL_ID == "vq2-canonical-square-aperture-v1"
    assert RECTIFIED_CHART_MODEL_ID == "vq2-rectified-camera-frd-slope-chart-v1"
    assert HOMOGRAPHY_GAUGE_MODEL_ID == "vq2-center-forward-h00-one-v1"
    assert CANONICAL_CORNER_ORDER == (
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    )
    assert CANONICAL_CORNERS == (
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    )
    assert HOMOGRAPHY_PARAMETER_ORDER == (
        "h01",
        "h02",
        "h10",
        "h11",
        "h12",
        "h20",
        "h21",
        "h22",
    )
    assert LOCAL_MEASUREMENT_ORDER == (
        "bearing_x_norm",
        "bearing_y_norm",
        "local_log_scale",
    )
    assert list(VQ2HomographyCovarianceScope) == [
        VQ2HomographyCovarianceScope.CONDITIONAL_FIT
    ]
    assert VQ2HomographyCovarianceScope.CONDITIONAL_FIT.value == "conditional_fit"
    assert issubclass(VQ2LocalDifferentialProvenanceError, VQ2LocalDifferentialMeasurementError)
    assert issubclass(VQ2LocalDifferentialGeometryError, VQ2LocalDifferentialMeasurementError)
    assert issubclass(VQ2LocalDifferentialCovarianceError, VQ2LocalDifferentialMeasurementError)


def test_frozen_dataclass_fields_and_operation_signature_are_exact() -> None:
    expected = {
        VQ2RectifiedHomographyMeasurementModel: (
            "model_id", "local_feature_model_id", "canonical_aperture_model_id",
            "canonical_corner_order", "rectified_chart_model_id",
            "homography_gauge_model_id", "homography_parameter_order", "image_size_px",
            "measurement_time_basis", "measurement_time_model_id",
            "geometry_producer_model_id", "geometry_producer_config_sha256",
            "homography_fit_model_id", "rectification_model_id",
            "rectification_calibration_id", "rectification_calibration_sha256",
            "homography_covariance_model_id", "output_covariance_model_id",
            "max_measurement_uncertainty_ns", "minimum_forward",
            "minimum_local_area_determinant", "minimum_projected_edge_length",
            "minimum_projected_corner_cross", "max_homography_condition",
            "max_local_differential_condition", "max_abs_center_bearing_norm",
            "max_abs_projected_corner_bearing_norm", "max_abs_local_log_scale",
            "max_input_variance", "max_output_variance", "covariance_psd_tolerance",
        ),
        VQ2RectifiedHomographyInput: (
            "frame_timing", "authority", "measurement_time_monotonic_ns",
            "measurement_time_basis", "measurement_time_model_id",
            "measurement_uncertainty_ns", "candidate_id", "image_size_px",
            "canonical_aperture_model_id", "rectified_chart_model_id",
            "homography_gauge_model_id", "geometry_producer_model_id",
            "geometry_producer_config_sha256", "homography_fit_model_id",
            "rectification_model_id", "rectification_calibration_id",
            "rectification_calibration_sha256", "homography_covariance_model_id",
            "homography_covariance_scope", "homography_parameter_order", "homography",
            "homography_covariance",
        ),
        VQ2RectifiedHomographyMeasurementEvidence: (
            "model", "source", "feature_model_id", "rectified_chart_model_id",
            "host_clock_id", "measurement_time_monotonic_ns", "measurement_time_basis",
            "measurement_time_model_id", "measurement_uncertainty_ns",
            "covariance_model_id", "covariance_scope", "input_fingerprint_sha256",
            "derivation_fingerprint_sha256", "measurement_order", "values",
            "projected_canonical_corners", "local_differential",
            "local_area_jacobian_determinant", "measurement_jacobian",
            "canonical_homography_covariance", "conditional_covariance",
            "minimum_canonical_forward",
            "homography_condition", "local_differential_condition",
            "minimum_projected_edge_length", "minimum_projected_corner_cross",
            "maximum_projected_abs_bearing",
        ),
    }
    for class_, names in expected.items():
        assert dataclasses.is_dataclass(class_)
        assert class_.__dataclass_params__.frozen is True
        assert "__slots__" in class_.__dict__
        assert tuple(field.name for field in dataclasses.fields(class_)) == names
    signature = inspect.signature(derive_local_differential_measurement)
    assert tuple(signature.parameters) == ("source", "model")
    assert signature.parameters["source"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert signature.parameters["model"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["source"].default is inspect.Parameter.empty
    assert signature.parameters["model"].default is inspect.Parameter.empty
    assert get_type_hints(derive_local_differential_measurement) == {
        "source": VQ2RectifiedHomographyInput,
        "model": VQ2RectifiedHomographyMeasurementModel,
        "return": VQ2RectifiedHomographyMeasurementEvidence,
    }


def test_public_tuple_aliases_and_dataclass_annotations_are_exact() -> None:
    assert subject.Vector2 == tuple[float, float]
    assert subject.Vector3 == tuple[float, float, float]
    assert subject.Vector8 == tuple[float, float, float, float, float, float, float, float]
    assert subject.Matrix2 == tuple[subject.Vector2, subject.Vector2]
    assert subject.Matrix3 == tuple[subject.Vector3, subject.Vector3, subject.Vector3]
    assert subject.Matrix8 == tuple[
        subject.Vector8,
        subject.Vector8,
        subject.Vector8,
        subject.Vector8,
        subject.Vector8,
        subject.Vector8,
        subject.Vector8,
        subject.Vector8,
    ]
    assert subject.Matrix3x8 == tuple[
        subject.Vector8,
        subject.Vector8,
        subject.Vector8,
    ]
    assert subject.ProjectedQuad == tuple[
        subject.Vector2,
        subject.Vector2,
        subject.Vector2,
        subject.Vector2,
    ]
    model_hints = get_type_hints(VQ2RectifiedHomographyMeasurementModel)
    assert model_hints == {
        "model_id": str,
        "local_feature_model_id": str,
        "canonical_aperture_model_id": str,
        "canonical_corner_order": tuple[str, str, str, str],
        "rectified_chart_model_id": str,
        "homography_gauge_model_id": str,
        "homography_parameter_order": tuple[str, str, str, str, str, str, str, str],
        "image_size_px": tuple[int, int],
        "measurement_time_basis": MeasurementTimeBasis,
        "measurement_time_model_id": str | None,
        "geometry_producer_model_id": str,
        "geometry_producer_config_sha256": str,
        "homography_fit_model_id": str,
        "rectification_model_id": str,
        "rectification_calibration_id": str,
        "rectification_calibration_sha256": str,
        "homography_covariance_model_id": str,
        "output_covariance_model_id": str,
        "max_measurement_uncertainty_ns": int,
        "minimum_forward": float,
        "minimum_local_area_determinant": float,
        "minimum_projected_edge_length": float,
        "minimum_projected_corner_cross": float,
        "max_homography_condition": float,
        "max_local_differential_condition": float,
        "max_abs_center_bearing_norm": float,
        "max_abs_projected_corner_bearing_norm": float,
        "max_abs_local_log_scale": float,
        "max_input_variance": float,
        "max_output_variance": float,
        "covariance_psd_tolerance": float,
    }
    source_hints = get_type_hints(VQ2RectifiedHomographyInput)
    assert source_hints == {
        "frame_timing": FrameTimingV1,
        "authority": GateAuthorityEpochV1,
        "measurement_time_monotonic_ns": int,
        "measurement_time_basis": MeasurementTimeBasis,
        "measurement_time_model_id": str | None,
        "measurement_uncertainty_ns": int,
        "candidate_id": str,
        "image_size_px": tuple[int, int],
        "canonical_aperture_model_id": str,
        "rectified_chart_model_id": str,
        "homography_gauge_model_id": str,
        "geometry_producer_model_id": str,
        "geometry_producer_config_sha256": str,
        "homography_fit_model_id": str,
        "rectification_model_id": str,
        "rectification_calibration_id": str,
        "rectification_calibration_sha256": str,
        "homography_covariance_model_id": str,
        "homography_covariance_scope": VQ2HomographyCovarianceScope,
        "homography_parameter_order": tuple[str, str, str, str, str, str, str, str],
        "homography": subject.Matrix3,
        "homography_covariance": subject.Matrix8,
    }
    evidence_hints = get_type_hints(VQ2RectifiedHomographyMeasurementEvidence)
    assert evidence_hints == {
        "model": VQ2RectifiedHomographyMeasurementModel,
        "source": VQ2RectifiedHomographyInput,
        "feature_model_id": str,
        "rectified_chart_model_id": str,
        "host_clock_id": str,
        "measurement_time_monotonic_ns": int,
        "measurement_time_basis": MeasurementTimeBasis,
        "measurement_time_model_id": str | None,
        "measurement_uncertainty_ns": int,
        "covariance_model_id": str,
        "covariance_scope": VQ2HomographyCovarianceScope,
        "input_fingerprint_sha256": str,
        "derivation_fingerprint_sha256": str,
        "measurement_order": tuple[str, str, str],
        "values": subject.Vector3,
        "projected_canonical_corners": subject.ProjectedQuad,
        "local_differential": subject.Matrix2,
        "local_area_jacobian_determinant": float,
        "measurement_jacobian": subject.Matrix3x8,
        "canonical_homography_covariance": subject.Matrix8,
        "conditional_covariance": subject.Matrix3,
        "minimum_canonical_forward": float,
        "homography_condition": float,
        "local_differential_condition": float,
        "minimum_projected_edge_length": float,
        "minimum_projected_corner_cross": float,
        "maximum_projected_abs_bearing": float,
    }
    assert get_type_hints(
        VQ2RectifiedHomographyMeasurementEvidence.validate_integrity
    ) == {"return": type(None)}


def test_nominal_result_is_deterministic_indivisible_and_integrity_checked() -> None:
    first = _result()
    second = _result()
    assert first == second
    assert first.validate_integrity() is None
    assert first.feature_model_id == LOCAL_DIFFERENTIAL_FEATURE_MODEL_ID
    assert first.rectified_chart_model_id == RECTIFIED_CHART_MODEL_ID
    assert first.host_clock_id == _HOST_CLOCK_ID
    assert first.measurement_time_monotonic_ns == _BASE_NS
    assert first.measurement_time_basis is MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY
    assert first.measurement_time_model_id is None
    assert first.measurement_uncertainty_ns == 20_000_000
    assert first.covariance_model_id == _model().output_covariance_model_id
    assert first.covariance_scope is VQ2HomographyCovarianceScope.CONDITIONAL_FIT
    assert first.measurement_order == LOCAL_MEASUREMENT_ORDER
    assert len(first.input_fingerprint_sha256) == 64
    assert len(first.derivation_fingerprint_sha256) == 64
    assert first.input_fingerprint_sha256.isascii()
    assert first.input_fingerprint_sha256 == first.input_fingerprint_sha256.lower()
    assert first.derivation_fingerprint_sha256 == first.derivation_fingerprint_sha256.lower()
    assert not hasattr(subject, "VQ2LocalDifferentialMeasurement")


def test_independent_projective_oracle_and_dense_covariance_congruence() -> None:
    source = _source()
    result = _result(source=source)
    oracle = _oracle(source.homography)
    assert np.asarray(result.values) == pytest.approx(oracle["values"], abs=1e-12)
    assert np.asarray(result.projected_canonical_corners) == pytest.approx(
        oracle["corners"], abs=1e-12
    )
    assert np.asarray(result.local_differential) == pytest.approx(oracle["local"], abs=1e-12)
    assert result.local_area_jacobian_determinant == pytest.approx(
        oracle["determinant"], abs=1e-12
    )
    assert result.minimum_canonical_forward == pytest.approx(oracle["minimum_forward"])
    assert result.minimum_projected_edge_length == pytest.approx(oracle["minimum_edge"])
    assert result.minimum_projected_corner_cross == pytest.approx(oracle["minimum_cross"])
    assert result.maximum_projected_abs_bearing == pytest.approx(oracle["maximum_corner"])
    assert result.homography_condition == pytest.approx(oracle["homography_condition"])
    assert result.local_differential_condition == pytest.approx(oracle["local_condition"])
    expected = (
        np.asarray(result.measurement_jacobian)
        @ np.asarray(result.canonical_homography_covariance)
        @ np.asarray(result.measurement_jacobian).T
    )
    assert np.asarray(result.conditional_covariance) == pytest.approx(expected, abs=1e-12)


def test_analytic_jacobian_matches_independent_central_differences() -> None:
    result = _result()
    finite_difference = _finite_difference_jacobian(result.source.homography)
    assert np.asarray(result.measurement_jacobian) == pytest.approx(
        finite_difference, rel=2e-7, abs=2e-8
    )


def test_randomized_projective_oracles_remain_inside_reviewed_envelope() -> None:
    rng = np.random.default_rng(3385)
    accepted = 0
    while accepted < 80:
        theta = np.array(
            (
                rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12),
                rng.uniform(-0.35, 0.35), rng.uniform(0.45, 0.95),
                rng.uniform(-0.16, 0.16), rng.uniform(-0.35, 0.35),
                rng.uniform(-0.16, 0.16), rng.uniform(0.45, 0.95),
            )
        )
        homography = _homography(theta)
        oracle = _oracle(homography)
        if float(oracle["determinant"]) <= 0.02 or float(oracle["minimum_cross"]) <= 0.01:
            continue
        source = _source(homography=homography)
        result = _result(source=source)
        assert np.asarray(result.values) == pytest.approx(oracle["values"], abs=2e-12)
        assert np.asarray(result.projected_canonical_corners) == pytest.approx(
            oracle["corners"], abs=2e-12
        )
        assert np.asarray(result.measurement_jacobian) == pytest.approx(
            _finite_difference_jacobian(homography), rel=3e-7, abs=3e-8
        )
        accepted += 1


def test_affine_scale_equals_finite_polygon_scale_but_perspective_does_not() -> None:
    affine = (
        (1.0, 0.0, 0.0),
        (0.2, 0.72, 0.11),
        (-0.1, -0.05, 0.58),
    )
    affine_result = _result(source=_source(homography=affine))
    affine_area = _polygon_area(np.asarray(affine_result.projected_canonical_corners))
    assert affine_result.values[2] == pytest.approx(0.5 * math.log(affine_area), abs=1e-12)
    perspective_result = _result()
    perspective_area = _polygon_area(
        np.asarray(perspective_result.projected_canonical_corners)
    )
    assert not math.isclose(
        perspective_result.values[2],
        0.5 * math.log(perspective_area),
        rel_tol=0.0,
        abs_tol=1e-4,
    )


def test_documented_five_summary_nullspace_changes_local_scale() -> None:
    theta = np.array((0.12, -0.08, 0.15, 0.72, 0.09, -0.11, -0.06, 0.55))
    summary_jacobian = _summary_jacobian(theta)
    _, singular_values, right_vectors = np.linalg.svd(summary_jacobian)
    assert np.linalg.matrix_rank(summary_jacobian, tol=1e-7) == 5
    assert singular_values == pytest.approx(
        (2.2121, 1.9941, 1.1674, 1.0000, 0.9825), rel=3e-4, abs=3e-4
    )
    nullspace = right_vectors[5:].T
    projected_gradient = nullspace.T @ _local_scale_gradient(theta)
    assert np.linalg.norm(projected_gradient) == pytest.approx(0.09570, rel=5e-4)
    assert np.linalg.norm(projected_gradient) > 0.09


def test_hard_constants_and_exact_boundary_model_are_accepted() -> None:
    assert HARD_MAX_MEASUREMENT_UNCERTAINTY_NS == 200_000_000
    assert HARD_MIN_FORWARD == 1e-6
    assert HARD_MIN_LOCAL_AREA_DETERMINANT == 1e-12
    assert HARD_MIN_PROJECTED_EDGE_LENGTH == 1e-6
    assert HARD_MIN_PROJECTED_CORNER_CROSS == 1e-12
    assert HARD_MAX_HOMOGRAPHY_CONDITION == 1e6
    assert HARD_MAX_LOCAL_DIFFERENTIAL_CONDITION == 1e6
    assert HARD_MAX_ABS_CENTER_BEARING_NORM == 4.0
    assert HARD_MAX_ABS_PROJECTED_CORNER_BEARING_NORM == 4.0
    assert HARD_MAX_ABS_LOCAL_LOG_SCALE == 20.0
    assert HARD_MAX_INPUT_VARIANCE == 1e6
    assert HARD_MAX_OUTPUT_VARIANCE == 1e6
    assert HARD_MAX_COVARIANCE_PSD_TOLERANCE == 1e-10
    assert not hasattr(subject, "DERIVED_ABS_TOLERANCE")
    model = _model(
        max_measurement_uncertainty_ns=HARD_MAX_MEASUREMENT_UNCERTAINTY_NS,
        minimum_forward=HARD_MIN_FORWARD,
        minimum_local_area_determinant=HARD_MIN_LOCAL_AREA_DETERMINANT,
        minimum_projected_edge_length=HARD_MIN_PROJECTED_EDGE_LENGTH,
        minimum_projected_corner_cross=HARD_MIN_PROJECTED_CORNER_CROSS,
        max_homography_condition=HARD_MAX_HOMOGRAPHY_CONDITION,
        max_local_differential_condition=HARD_MAX_LOCAL_DIFFERENTIAL_CONDITION,
        max_abs_center_bearing_norm=HARD_MAX_ABS_CENTER_BEARING_NORM,
        max_abs_projected_corner_bearing_norm=HARD_MAX_ABS_PROJECTED_CORNER_BEARING_NORM,
        max_abs_local_log_scale=HARD_MAX_ABS_LOCAL_LOG_SCALE,
        max_input_variance=HARD_MAX_INPUT_VARIANCE,
        max_output_variance=HARD_MAX_OUTPUT_VARIANCE,
        covariance_psd_tolerance=HARD_MAX_COVARIANCE_PSD_TOLERANCE,
    )
    assert _result(model=model).model == model


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_measurement_uncertainty_ns", HARD_MAX_MEASUREMENT_UNCERTAINTY_NS + 1),
        ("minimum_forward", HARD_MIN_FORWARD * 0.5),
        ("minimum_local_area_determinant", HARD_MIN_LOCAL_AREA_DETERMINANT * 0.5),
        ("minimum_projected_edge_length", HARD_MIN_PROJECTED_EDGE_LENGTH * 0.5),
        ("minimum_projected_corner_cross", HARD_MIN_PROJECTED_CORNER_CROSS * 0.5),
        ("max_homography_condition", HARD_MAX_HOMOGRAPHY_CONDITION * 1.01),
        ("max_local_differential_condition", HARD_MAX_LOCAL_DIFFERENTIAL_CONDITION * 1.01),
        ("max_abs_center_bearing_norm", HARD_MAX_ABS_CENTER_BEARING_NORM + 0.01),
        (
            "max_abs_projected_corner_bearing_norm",
            HARD_MAX_ABS_PROJECTED_CORNER_BEARING_NORM + 0.01,
        ),
        ("max_abs_local_log_scale", HARD_MAX_ABS_LOCAL_LOG_SCALE + 0.01),
        ("max_input_variance", HARD_MAX_INPUT_VARIANCE + 1.0),
        ("max_output_variance", HARD_MAX_OUTPUT_VARIANCE + 1.0),
        (
            "covariance_psd_tolerance",
            HARD_MAX_COVARIANCE_PSD_TOLERANCE * 1.01,
        ),
    ],
)
def test_model_cannot_relax_hard_limits(field: str, value: object) -> None:
    _assert_rejected(lambda: _model(**{field: value}))


@pytest.mark.parametrize(
    "field",
    ["max_homography_condition", "max_local_differential_condition"],
)
def test_condition_number_caps_accept_one_and_reject_below_one(field: str) -> None:
    assert _model(**{field: 1.0})
    _assert_rejected(lambda: _model(**{field: math.nextafter(1.0, 0.0)}))


@pytest.mark.parametrize(
    "field",
    [
        "max_measurement_uncertainty_ns",
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
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_model_numeric_fields_reject_nonpositive_bool_or_nonfinite(
    field: str, value: object
) -> None:
    _assert_rejected(lambda: _model(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("local_feature_model_id", "wrong-local-feature"),
        ("canonical_aperture_model_id", "wrong-canonical-aperture"),
        ("canonical_corner_order", tuple(reversed(CANONICAL_CORNER_ORDER))),
        ("canonical_corner_order", list(CANONICAL_CORNER_ORDER)),
        ("canonical_corner_order", (*CANONICAL_CORNER_ORDER[:-1], 1)),
        ("rectified_chart_model_id", "wrong-chart"),
        ("homography_gauge_model_id", "wrong-gauge"),
        ("homography_parameter_order", tuple(reversed(HOMOGRAPHY_PARAMETER_ORDER))),
        ("homography_parameter_order", list(HOMOGRAPHY_PARAMETER_ORDER)),
        ("image_size_px", [640, 360]),
        ("image_size_px", (True, 360)),
        ("model_id", "contains whitespace"),
        ("geometry_producer_config_sha256", "A" * 64),
        ("geometry_producer_config_sha256", "a" * 63),
        ("rectification_calibration_sha256", "z" * 64),
    ],
)
def test_model_semantics_tokens_hashes_and_orders_fail_closed(
    field: str, value: object
) -> None:
    _assert_rejected(lambda: _model(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_id", "bad candidate"),
        ("canonical_aperture_model_id", "wrong-canonical"),
        ("rectified_chart_model_id", "wrong-chart"),
        ("homography_gauge_model_id", "wrong-gauge"),
        ("geometry_producer_config_sha256", "A" * 64),
        ("rectification_calibration_sha256", "2" * 63),
        ("homography_covariance_scope", "conditional_fit"),
        ("homography_parameter_order", tuple(reversed(HOMOGRAPHY_PARAMETER_ORDER))),
        ("homography_parameter_order", list(HOMOGRAPHY_PARAMETER_ORDER)),
        ("homography_parameter_order", (*HOMOGRAPHY_PARAMETER_ORDER[:-1], 7)),
        ("image_size_px", [640, 360]),
        ("image_size_px", (True, 360)),
        ("image_size_px", (640, 0)),
        ("measurement_uncertainty_ns", True),
        ("measurement_uncertainty_ns", 0),
    ],
)
def test_source_exact_types_semantics_hashes_and_orders_fail_closed(
    field: str, value: object
) -> None:
    _assert_rejected(lambda: _source(**{field: value}))


def test_homography_gauge_dimensions_and_numeric_entries_fail_closed() -> None:
    bad_homographies: list[object] = [
        ((2.0, 0.08, -0.06), _BASE_H[1], _BASE_H[2]),
        list(_BASE_H),
        (_BASE_H[0], _BASE_H[1]),
        (_BASE_H[0], _BASE_H[1], (0.0, 1.0)),
        _replace_matrix_entry(_BASE_H, 1, 1, True),
        _replace_matrix_entry(_BASE_H, 1, 1, float("nan")),
        _replace_matrix_entry(_BASE_H, 1, 1, float("inf")),
    ]
    for value in bad_homographies:
        _assert_rejected(lambda value=value: _source(homography=value))
    covariance = _base_covariance()
    bad_covariances: list[object] = [
        list(covariance),
        covariance[:-1],
        tuple(tuple(0.0 for _ in range(9)) for _ in range(9)),
        _replace_matrix_entry(covariance, 0, 0, True),
        _replace_matrix_entry(covariance, 0, 0, float("nan")),
    ]
    for value in bad_covariances:
        _assert_rejected(lambda value=value: _source(homography_covariance=value))
    _assert_rejected(
        lambda: _source(homography_covariance=_matrix_tuple(np.zeros((8, 8))))
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("frame_timing", object()), ("authority", object())],
)
def test_nested_timing_and_authority_require_exact_public_types(
    field: str, value: object
) -> None:
    _assert_rejected(lambda: _source(**{field: value}))


def test_documented_error_taxonomy_is_used_for_each_failure_family() -> None:
    with pytest.raises(VQ2LocalDifferentialProvenanceError):
        _result(model=_model(geometry_producer_model_id="different-producer"))
    with pytest.raises(VQ2LocalDifferentialGeometryError):
        _source(homography=((2.0, 0.08, -0.06), _BASE_H[1], _BASE_H[2]))
    zero_marginal = np.eye(8)
    zero_marginal[0, 0] = 0.0
    with pytest.raises(VQ2LocalDifferentialCovarianceError):
        _source(homography_covariance=_matrix_tuple(zero_marginal))


def test_camera_proxy_and_calibrated_time_rules_are_exact() -> None:
    _assert_rejected(
        lambda: _source(measurement_time_monotonic_ns=_BASE_NS - 1)
    )
    _assert_rejected(lambda: _source(measurement_time_model_id="proxy-cannot-map"))
    calibrated_source = _source(
        measurement_time_monotonic_ns=_BASE_NS - 2_000_000,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
        measurement_time_model_id="camera-capture-map-v1",
    )
    calibrated_model = _model(
        measurement_time_basis=MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
        measurement_time_model_id="camera-capture-map-v1",
    )
    assert _result(source=calibrated_source, model=calibrated_model).measurement_time_basis \
        is MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED
    _assert_rejected(
        lambda: _source(
            measurement_time_basis=MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
            measurement_time_model_id=None,
        )
    )
    _assert_rejected(
        lambda: _source(
            measurement_time_monotonic_ns=_BASE_NS + 5_000_000,
            measurement_time_basis=MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
            measurement_time_model_id="camera-capture-map-v1",
        )
    )
    _assert_rejected(
        lambda: _source(measurement_time_basis=MeasurementTimeBasis.IMU_SAMPLE)
    )
    _assert_rejected(
        lambda: _source(measurement_time_basis="camera_final_packet_proxy")
    )
    _assert_rejected(
        lambda: _model(measurement_time_basis=MeasurementTimeBasis.IMU_SAMPLE)
    )
    _assert_rejected(
        lambda: _model(measurement_time_basis="camera_final_packet_proxy")
    )
    _assert_rejected(
        lambda: _model(
            measurement_time_basis=MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
            measurement_time_model_id=None,
        )
    )
    _assert_rejected(lambda: _model(measurement_time_model_id="proxy-cannot-map"))


def test_measurement_uncertainty_exact_type_hard_boundary_and_tightening() -> None:
    hard_source = _source(
        measurement_uncertainty_ns=HARD_MAX_MEASUREMENT_UNCERTAINTY_NS
    )
    hard_model = _model(
        max_measurement_uncertainty_ns=HARD_MAX_MEASUREMENT_UNCERTAINTY_NS
    )
    assert _result(source=hard_source, model=hard_model)
    equality_source = _source(measurement_uncertainty_ns=20_000_000)
    assert _result(
        source=equality_source,
        model=_model(max_measurement_uncertainty_ns=20_000_000),
    )
    _assert_rejected(
        lambda: _result(
            source=equality_source,
            model=_model(max_measurement_uncertainty_ns=19_999_999),
        )
    )
    _assert_rejected(
        lambda: _source(
            measurement_uncertainty_ns=HARD_MAX_MEASUREMENT_UNCERTAINTY_NS + 1
        )
    )
    _assert_rejected(lambda: _source(measurement_uncertainty_ns=20_000_000.0))
    _assert_rejected(
        lambda: _model(max_measurement_uncertainty_ns=20_000_000.0)
    )
    _assert_rejected(lambda: _source(measurement_time_monotonic_ns=float(_BASE_NS)))
    _assert_rejected(lambda: _source(measurement_time_monotonic_ns=True))


def test_authority_cutovers_are_inclusive_and_mismatches_reject() -> None:
    timing = _frame_timing()
    inclusive = _authority(
        cutover_sequence=timing.publication_sequence,
        cutover_ns=timing.publish_monotonic_ns,
    )
    assert _result(source=_source(frame_timing=timing, authority=inclusive))
    invalid_authorities = [
        _authority(host_clock_id="different-host-clock"),
        _authority(stream_id="different-camera"),
        _authority(generation=_GENERATION + 1),
        _authority(cutover_sequence=timing.publication_sequence + 1),
        _authority(cutover_ns=timing.publish_monotonic_ns + 1),
    ]
    for authority in invalid_authorities:
        _assert_rejected(lambda authority=authority: _source(authority=authority))


def test_model_source_binding_rejects_every_identity_mismatch() -> None:
    source = _source()
    mismatches: list[tuple[str, object]] = [
        ("image_size_px", (800, 600)),
        ("geometry_producer_model_id", "different-producer"),
        ("geometry_producer_config_sha256", _OTHER_HASH),
        ("homography_fit_model_id", "different-fit"),
        ("rectification_model_id", "different-rectification"),
        ("rectification_calibration_id", "different-calibration"),
        ("rectification_calibration_sha256", _OTHER_HASH),
        ("homography_covariance_model_id", "different-covariance"),
    ]
    for field, value in mismatches:
        try:
            model = _model(**{field: value})
        except (TypeError, ValueError):
            continue
        _assert_rejected(lambda model=model: _result(source=source, model=model))
    calibrated_source = _source(
        measurement_time_monotonic_ns=_BASE_NS - 2_000_000,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
        measurement_time_model_id="camera-capture-map-v1",
    )
    _assert_rejected(
        lambda: _result(source=calibrated_source, model=_model())
    )
    _assert_rejected(
        lambda: _result(
            source=calibrated_source,
            model=_model(
                measurement_time_basis=(
                    MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED
                ),
                measurement_time_model_id="different-camera-capture-map-v1",
            ),
        )
    )
    _assert_rejected(
        lambda: _result(model=_model(max_measurement_uncertainty_ns=10_000_000))
    )


def test_nested_public_contract_corruption_is_revalidated() -> None:
    source = _source()
    object.__setattr__(source.frame_timing, "publication_sequence", True)
    _assert_rejected(lambda: _result(source=source))
    source = _source()
    object.__setattr__(source.authority, "camera_generation", "5")
    _assert_rejected(lambda: _result(source=source))


def test_fingerprints_bind_source_and_reducer_model_values() -> None:
    baseline = _result()
    changed_h = [list(row) for row in _BASE_H]
    changed_h[1][1] += 0.001
    homography_result = _result(
        source=_source(homography=tuple(tuple(row) for row in changed_h))
    )
    covariance = np.asarray(_base_covariance())
    covariance[0, 0] += 1e-5
    covariance_result = _result(
        source=_source(homography_covariance=_matrix_tuple(covariance))
    )
    config_result = _result(
        source=_source(geometry_producer_config_sha256=_OTHER_HASH),
        model=_model(geometry_producer_config_sha256=_OTHER_HASH),
    )
    calibration_result = _result(
        source=_source(rectification_calibration_sha256=_OTHER_HASH),
        model=_model(rectification_calibration_sha256=_OTHER_HASH),
    )
    for changed in (
        homography_result,
        covariance_result,
        config_result,
        calibration_result,
    ):
        assert changed.input_fingerprint_sha256 != baseline.input_fingerprint_sha256
        assert changed.derivation_fingerprint_sha256 != baseline.derivation_fingerprint_sha256
    changed_model = _model(max_homography_condition=90_000.0)
    model_result = _result(model=changed_model)
    assert model_result.input_fingerprint_sha256 == baseline.input_fingerprint_sha256
    assert model_result.derivation_fingerprint_sha256 != baseline.derivation_fingerprint_sha256


@pytest.mark.parametrize(
    ("row", "column"),
    [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
)
def test_input_fingerprint_binds_every_gauge_fixed_homography_parameter(
    row: int, column: int
) -> None:
    baseline = _result()
    changed = [list(values) for values in _BASE_H]
    changed[row][column] += 1.0e-5
    result = _result(
        source=_source(homography=tuple(tuple(values) for values in changed))
    )
    assert result.input_fingerprint_sha256 != baseline.input_fingerprint_sha256
    assert result.derivation_fingerprint_sha256 != (
        baseline.derivation_fingerprint_sha256
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "wave3e-local-reducer-v1-alternate"),
        ("output_covariance_model_id", "local-differential-congruence-v1-alt"),
        ("max_measurement_uncertainty_ns", 60_000_000),
        ("minimum_forward", 5.0e-6),
        ("minimum_local_area_determinant", 5.0e-11),
        ("minimum_projected_edge_length", 5.0e-6),
        ("minimum_projected_corner_cross", 5.0e-11),
        ("max_homography_condition", 110_000.0),
        ("max_local_differential_condition", 110_000.0),
        ("max_abs_center_bearing_norm", 3.6),
        ("max_abs_projected_corner_bearing_norm", 3.6),
        ("max_abs_local_log_scale", 11.0),
        ("max_input_variance", 11.0),
        ("max_output_variance", 11.0),
        ("covariance_psd_tolerance", 5.0e-11),
    ],
)
def test_derivation_fingerprint_binds_each_independently_variable_model_field(
    field: str, value: object
) -> None:
    baseline = _result()
    changed = _result(model=_model(**{field: value}))
    assert changed.input_fingerprint_sha256 == baseline.input_fingerprint_sha256
    assert changed.derivation_fingerprint_sha256 != (
        baseline.derivation_fingerprint_sha256
    )


@pytest.mark.parametrize(
    "field",
    [
        "feature_model_id",
        "rectified_chart_model_id",
        "host_clock_id",
        "measurement_time_monotonic_ns",
        "measurement_time_basis",
        "measurement_time_model_id",
        "measurement_uncertainty_ns",
        "covariance_model_id",
        "covariance_scope",
        "input_fingerprint_sha256",
        "derivation_fingerprint_sha256",
        "measurement_order",
        "values",
        "projected_canonical_corners",
        "local_differential",
        "local_area_jacobian_determinant",
        "measurement_jacobian",
        "canonical_homography_covariance",
        "conditional_covariance",
        "minimum_canonical_forward",
        "homography_condition",
        "local_differential_condition",
        "minimum_projected_edge_length",
        "minimum_projected_corner_cross",
        "maximum_projected_abs_bearing",
    ],
)
def test_integrity_rejects_every_detached_binding_and_rederived_value(
    field: str,
) -> None:
    result = _result()
    matrices = {
        "projected_canonical_corners",
        "local_differential",
        "measurement_jacobian",
        "canonical_homography_covariance",
        "conditional_covariance",
    }
    replacements: dict[str, object] = {
        "feature_model_id": "tampered-feature-model",
        "rectified_chart_model_id": "tampered-rectified-chart",
        "host_clock_id": "tampered-host-clock",
        "measurement_time_monotonic_ns": result.measurement_time_monotonic_ns + 1,
        "measurement_time_basis": MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
        "measurement_time_model_id": "tampered-time-model",
        "measurement_uncertainty_ns": result.measurement_uncertainty_ns + 1,
        "covariance_model_id": "tampered-output-covariance",
        "covariance_scope": "conditional_fit",
        "input_fingerprint_sha256": "f" * 64,
        "derivation_fingerprint_sha256": "e" * 64,
        "measurement_order": tuple(reversed(LOCAL_MEASUREMENT_ORDER)),
        "values": (
            math.nextafter(result.values[0], math.inf),
            *result.values[1:],
        ),
    }
    if field in matrices:
        changed = [list(row) for row in getattr(result, field)]
        changed[0][0] = math.nextafter(float(changed[0][0]), math.inf)
        replacement: object = tuple(tuple(row) for row in changed)
    elif field in replacements:
        replacement = replacements[field]
    else:
        replacement = math.nextafter(float(getattr(result, field)), math.inf)
    object.__setattr__(result, field, replacement)
    _assert_rejected(result.validate_integrity)


@pytest.mark.parametrize(
    ("path", "output_field", "replacement"),
    [
        (("model", "local_feature_model_id"), "feature_model_id", "changed-feature"),
        (("model", "rectified_chart_model_id"), "rectified_chart_model_id", "changed-chart"),
        (
            ("model", "measurement_time_basis"),
            "measurement_time_basis",
            MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
        ),
        (("model", "measurement_time_model_id"), "measurement_time_model_id", "changed-time"),
        (("model", "output_covariance_model_id"), "covariance_model_id", "changed-output-cov"),
        (("source", "rectified_chart_model_id"), "rectified_chart_model_id", "changed-chart"),
        (
            ("source", "frame_timing", "host_clock_id"),
            "host_clock_id",
            "changed-host-clock",
        ),
        (
            ("source", "measurement_time_monotonic_ns"),
            "measurement_time_monotonic_ns",
            _BASE_NS + 1,
        ),
        (
            ("source", "measurement_time_basis"),
            "measurement_time_basis",
            MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
        ),
        (
            ("source", "measurement_time_model_id"),
            "measurement_time_model_id",
            "changed-time",
        ),
        (
            ("source", "measurement_uncertainty_ns"),
            "measurement_uncertainty_ns",
            20_000_001,
        ),
        (
            ("source", "homography_covariance_scope"),
            "covariance_scope",
            "conditional_fit",
        ),
    ],
)
def test_nested_tampering_cannot_relabel_retained_output_duplicates(
    path: tuple[str, ...], output_field: str, replacement: object
) -> None:
    result = _result()
    retained_output = getattr(result, output_field)
    retained_input_fingerprint = result.input_fingerprint_sha256
    retained_derivation_fingerprint = result.derivation_fingerprint_sha256
    target: object = result
    for component in path[:-1]:
        target = getattr(target, component)
    object.__setattr__(target, path[-1], replacement)
    assert getattr(result, output_field) == retained_output
    assert result.input_fingerprint_sha256 == retained_input_fingerprint
    assert result.derivation_fingerprint_sha256 == retained_derivation_fingerprint
    _assert_rejected(result.validate_integrity)


def test_integrity_rejects_replaced_or_mutated_complete_model_and_source() -> None:
    result = _result()
    object.__setattr__(result, "model", _model(max_homography_condition=90_000.0))
    _assert_rejected(result.validate_integrity)
    result = _result()
    object.__setattr__(result, "source", _source(candidate_id="replacement-candidate"))
    _assert_rejected(result.validate_integrity)
    result = _result()
    changed_homography = [list(row) for row in result.source.homography]
    changed_homography[1][1] += 1.0e-4
    object.__setattr__(
        result.source,
        "homography",
        tuple(tuple(row) for row in changed_homography),
    )
    _assert_rejected(result.validate_integrity)
    result = _result()
    changed_covariance = np.asarray(result.source.homography_covariance)
    changed_covariance[0, 0] += 1.0e-5
    object.__setattr__(
        result.source,
        "homography_covariance",
        _matrix_tuple(changed_covariance),
    )
    _assert_rejected(result.validate_integrity)


@pytest.mark.parametrize(
    "field",
    [
        "projected_canonical_corners",
        "local_differential",
        "measurement_jacobian",
        "canonical_homography_covariance",
        "conditional_covariance",
    ],
)
def test_integrity_rejects_malformed_derived_matrix_dimensions(field: str) -> None:
    result = _result()
    object.__setattr__(result, field, getattr(result, field)[:-1])
    _assert_rejected(result.validate_integrity)


def _identity_covariance_result() -> VQ2RectifiedHomographyMeasurementEvidence:
    identity_homography = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    return _result(
        source=_source(
            homography=identity_homography,
            homography_covariance=_matrix_tuple(np.eye(8)),
        )
    )


def test_integrity_rejects_equal_int_mutation_of_retained_model_float() -> None:
    result = _identity_covariance_result()
    original = result.model.max_output_variance
    assert type(original) is float and original.is_integer()
    object.__setattr__(result.model, "max_output_variance", int(original))
    assert result.model.max_output_variance == original
    assert type(result.model.max_output_variance) is int
    _assert_rejected(result.validate_integrity)


@pytest.mark.parametrize("field", ["homography", "homography_covariance"])
def test_integrity_rejects_equal_int_mutation_of_retained_source_matrix(
    field: str,
) -> None:
    result = _identity_covariance_result()
    changed = [list(row) for row in getattr(result.source, field)]
    original = changed[0][0]
    assert type(original) is float and original == 1.0
    changed[0][0] = int(original)
    object.__setattr__(result.source, field, tuple(tuple(row) for row in changed))
    assert getattr(result.source, field)[0][0] == original
    assert type(getattr(result.source, field)[0][0]) is int
    _assert_rejected(result.validate_integrity)


@pytest.mark.parametrize(
    "field",
    [
        "projected_canonical_corners",
        "local_differential",
        "measurement_jacobian",
        "canonical_homography_covariance",
        "conditional_covariance",
    ],
)
def test_integrity_rejects_equal_int_mutation_of_derived_evidence_matrix(
    field: str,
) -> None:
    result = _identity_covariance_result()
    changed = [list(row) for row in getattr(result, field)]
    position = next(
        (row, column)
        for row, values in enumerate(changed)
        for column, value in enumerate(values)
        if type(value) is float and value.is_integer()
    )
    row, column = position
    original = changed[row][column]
    changed[row][column] = int(original)
    object.__setattr__(result, field, tuple(tuple(values) for values in changed))
    assert getattr(result, field)[row][column] == original
    assert type(getattr(result, field)[row][column]) is int
    _assert_rejected(result.validate_integrity)


def test_dense_off_diagonal_covariance_is_retained_without_floor() -> None:
    dense = np.asarray(_base_covariance())
    diagonal = np.diag(np.diag(dense))
    dense_result = _result(source=_source(homography_covariance=_matrix_tuple(dense)))
    diagonal_result = _result(source=_source(homography_covariance=_matrix_tuple(diagonal)))
    assert not np.allclose(
        dense_result.conditional_covariance,
        diagonal_result.conditional_covariance,
        rtol=1e-12,
        atol=1e-15,
    )
    tiny = np.eye(8) * 1e-20
    tiny_result = _result(source=_source(homography_covariance=_matrix_tuple(tiny)))
    expected = (
        np.asarray(tiny_result.measurement_jacobian)
        @ tiny
        @ np.asarray(tiny_result.measurement_jacobian).T
    )
    assert np.asarray(tiny_result.conditional_covariance) == pytest.approx(
        expected, rel=1e-12, abs=1e-30
    )
    assert float(np.max(np.diag(tiny_result.conditional_covariance))) < 1e-18


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e6])
def test_raw_covariance_is_preserved_then_canonicalized_under_active_tolerance(
    scale: float,
) -> None:
    within = np.eye(8) * scale
    within[0, 1] = 0.4 * HARD_MAX_COVARIANCE_PSD_TOLERANCE * scale
    affine = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    source = _source(
        homography=affine,
        homography_covariance=_matrix_tuple(within),
    )
    assert np.asarray(source.homography_covariance) == pytest.approx(
        within, rel=0.0, abs=0.0
    )
    variance_model_fields = {
        "max_input_variance": max(10.0, float(scale)),
        "max_output_variance": HARD_MAX_OUTPUT_VARIANCE,
    }
    result = _result(source=source, model=_model(**variance_model_fields))
    canonical = 0.5 * (within + within.T)
    assert np.asarray(result.canonical_homography_covariance) == pytest.approx(
        canonical, rel=0.0, abs=0.0
    )
    expected_output = (
        np.asarray(result.measurement_jacobian)
        @ canonical
        @ np.asarray(result.measurement_jacobian).T
    )
    assert np.asarray(result.conditional_covariance) == pytest.approx(
        expected_output, rel=1.0e-12, abs=1.0e-24
    )
    _assert_rejected(
        lambda: _result(
            source=source,
            model=_model(
                **variance_model_fields,
                covariance_psd_tolerance=(
                    0.1 * HARD_MAX_COVARIANCE_PSD_TOLERANCE
                )
            ),
        )
    )
    beyond = np.eye(8) * scale
    beyond[0, 1] = 1.1 * HARD_MAX_COVARIANCE_PSD_TOLERANCE * scale
    _assert_rejected(
        lambda: _source(homography_covariance=_matrix_tuple(beyond))
    )


def test_raw_covariance_fingerprint_precedes_canonical_symmetrization() -> None:
    raw = np.eye(8)
    raw[0, 1] = 0.4 * HARD_MAX_COVARIANCE_PSD_TOLERANCE
    canonical = 0.5 * (raw + raw.T)
    raw_result = _result(
        source=_source(homography_covariance=_matrix_tuple(raw))
    )
    canonical_result = _result(
        source=_source(homography_covariance=_matrix_tuple(canonical))
    )
    assert raw_result.source.homography_covariance != (
        canonical_result.source.homography_covariance
    )
    assert raw_result.input_fingerprint_sha256 != (
        canonical_result.input_fingerprint_sha256
    )
    assert raw_result.derivation_fingerprint_sha256 != (
        canonical_result.derivation_fingerprint_sha256
    )
    assert raw_result.canonical_homography_covariance == (
        canonical_result.canonical_homography_covariance
    )
    assert raw_result.conditional_covariance == canonical_result.conditional_covariance


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e6])
def test_tiny_negative_roundoff_is_retained_but_material_indefiniteness_rejects(
    scale: float,
) -> None:
    within = np.eye(8) * scale
    epsilon = 0.4 * HARD_MAX_COVARIANCE_PSD_TOLERANCE * scale
    within[0, 1] = within[1, 0] = scale + epsilon
    source = _source(homography_covariance=_matrix_tuple(within))
    assert float(np.min(np.linalg.eigvalsh(source.homography_covariance))) < 0.0
    assert source.homography_covariance[0][1] > source.homography_covariance[0][0]
    below = np.eye(8) * scale
    below[0, 1] = below[1, 0] = (
        scale + 1.2 * HARD_MAX_COVARIANCE_PSD_TOLERANCE * scale
    )
    _assert_rejected(lambda: _source(homography_covariance=_matrix_tuple(below)))


def test_input_nonpositive_and_derived_zero_marginals_reject() -> None:
    zero = np.eye(8)
    zero[0, 0] = 0.0
    _assert_rejected(lambda: _source(homography_covariance=_matrix_tuple(zero)))
    tiny_negative = np.eye(8)
    tiny_negative[0, 0] = -1.0e-20
    _assert_rejected(
        lambda: _source(homography_covariance=_matrix_tuple(tiny_negative))
    )
    affine = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    covariance = np.eye(8)
    covariance[3, 7] = covariance[7, 3] = -1.0
    _assert_rejected(
        lambda: _result(
            source=_source(
                homography=affine,
                homography_covariance=_matrix_tuple(covariance),
            )
        )
    )


def test_tighten_only_thresholds_accept_equality_and_reject_tighter() -> None:
    baseline = _result()
    source = baseline.source
    observed: list[tuple[str, float]] = [
        ("minimum_forward", baseline.minimum_canonical_forward),
        ("minimum_local_area_determinant", baseline.local_area_jacobian_determinant),
        ("minimum_projected_edge_length", baseline.minimum_projected_edge_length),
        ("minimum_projected_corner_cross", baseline.minimum_projected_corner_cross),
        ("max_homography_condition", baseline.homography_condition),
        ("max_local_differential_condition", baseline.local_differential_condition),
        ("max_abs_center_bearing_norm", max(abs(baseline.values[0]), abs(baseline.values[1]))),
        ("max_abs_projected_corner_bearing_norm", baseline.maximum_projected_abs_bearing),
        ("max_abs_local_log_scale", abs(baseline.values[2])),
        (
            "max_input_variance",
            float(np.max(np.diag(source.homography_covariance))),
        ),
        (
            "max_output_variance",
            float(np.max(np.diag(baseline.conditional_covariance))),
        ),
    ]
    minimum_fields = {name for name, _ in observed if name.startswith("minimum_")}
    for field, value in observed:
        equality_model = _model(**{field: value})
        assert _result(source=source, model=equality_model)
        tightened = value * (1.0 + 1e-8) if field in minimum_fields else value * (1.0 - 1e-8)
        _assert_rejected(
            lambda field=field, tightened=tightened: _result(
                source=source,
                model=_model(**{field: tightened}),
            )
        )


@pytest.mark.parametrize(
    "homography",
    [
        ((1.0, 0.9999995, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        ((1.0, 0.0, 0.0), (0.0, 1e-7, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0), (4.1, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 0.0), (0.0, 4.1, 0.0), (0.0, 0.0, 1.0)),
    ],
)
def test_geometry_denominator_orientation_collapse_condition_and_bearing_guards(
    homography: tuple[tuple[float, ...], ...],
) -> None:
    _assert_rejected(lambda: _result(source=_source(homography=homography)))


def test_exact_source_type_rejects_forbidden_current_and_state_inputs() -> None:
    covariance = FeatureCovarianceV1(
        model_id="center-only-v1",
        feature_order=("center_x_norm", "center_y_norm"),
        matrix=((0.01, 0.0), (0.0, 0.01)),
    )
    forbidden = [
        covariance,
        ((-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)),
        object.__new__(GateObservationV1),
        object.__new__(VQ2DerotationEvidence),
        object.__new__(VQ2LocalDifferentialFeatureState),
        object.__new__(RelativeGateStateV1),
    ]
    for value in forbidden:
        _assert_rejected(
            lambda value=value: derive_local_differential_measurement(
                value, model=_model()  # type: ignore[arg-type]
            )
        )
    _assert_rejected(
        lambda: _source(
            homography_covariance_scope=VQ2CovarianceScope.TRANSFORM_TOTAL
        )
    )


def test_exact_model_and_source_subclasses_are_rejected() -> None:
    class ModelSubclass(VQ2RectifiedHomographyMeasurementModel):
        pass

    class SourceSubclass(VQ2RectifiedHomographyInput):
        pass

    model = ModelSubclass(
        **{
            field.name: getattr(_model(), field.name)
            for field in dataclasses.fields(_model())
        }
    )
    source = SourceSubclass(
        **{
            field.name: getattr(_source(), field.name)
            for field in dataclasses.fields(_source())
        }
    )
    _assert_rejected(lambda: derive_local_differential_measurement(_source(), model=model))
    _assert_rejected(lambda: derive_local_differential_measurement(source, model=_model()))


def test_new_module_has_no_private_contract_or_forbidden_runtime_imports() -> None:
    module_path = Path(subject.__file__).resolve()
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    allowed_direct_imports = {
        "hashlib",
        "json",
        "math",
        "re",
        "numpy",
    }
    allowed_from_imports = {
        "__future__",
        "dataclasses",
        "enum",
        "typing",
        "competition.vq2_contracts",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module in allowed_from_imports
            if node.module == "competition.vq2_contracts":
                assert all(not alias.name.startswith("_") for alias in node.names)
                assert all(alias.name != "_validate_authority_frame_cutover" for alias in node.names)
        elif isinstance(node, ast.Import):
            assert all(alias.name in allowed_direct_imports for alias in node.names)


def test_no_production_python_module_imports_or_calls_wave3e_reducer() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module_name = "estimation.vq2_local_differential_measurement"
    offenders: list[str] = []
    for path in repo_root.rglob("*.py"):
        relative = path.relative_to(repo_root)
        if any(part in {".loop", ".research_loop", ".venv", "__pycache__"} for part in relative.parts):
            continue
        if "tests" in relative.parts or path.resolve() == Path(subject.__file__).resolve():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = any(
            (
                isinstance(node, ast.ImportFrom)
                and (
                    node.module == module_name
                    or (
                        node.module == "estimation"
                        and any(
                            alias.name == "vq2_local_differential_measurement"
                            for alias in node.names
                        )
                    )
                    or (
                        node.level > 0
                        and any(
                            alias.name == "vq2_local_differential_measurement"
                            for alias in node.names
                        )
                    )
                )
            )
            or (
                isinstance(node, ast.Import)
                and any(alias.name == module_name for alias in node.names)
            )
            for node in ast.walk(tree)
        )
        dynamically_named = any(
            isinstance(node, ast.Constant)
            and type(node.value) is str
            and node.value
            in {
                module_name,
                "vq2_local_differential_measurement",
                "derive_local_differential_measurement",
            }
            for node in ast.walk(tree)
        )
        called = any(
            isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "derive_local_differential_measurement"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "derive_local_differential_measurement"
            )
            for node in ast.walk(tree)
        )
        if imported or called or dynamically_named:
            offenders.append(str(relative))
    assert offenders == []


def test_frozen_v1_covariance_and_observation_codecs_round_trip_unchanged() -> None:
    covariance = FeatureCovarianceV1(
        model_id="codec-center-v1",
        feature_order=("center_x_norm", "center_y_norm"),
        matrix=((0.01, 0.002), (0.002, 0.02)),
    )
    covariance_primitive = {
        "model_id": "codec-center-v1",
        "feature_order": ["center_x_norm", "center_y_norm"],
        "matrix": [[0.01, 0.002], [0.002, 0.02]],
    }
    assert covariance.to_primitive() == covariance_primitive
    assert FeatureCovarianceV1.from_primitive(covariance_primitive) == covariance
    timing = _frame_timing()
    observation = GateObservationV1(
        frame_timing=timing,
        measurement_time_monotonic_ns=timing.final_unique_packet_monotonic_ns,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        measurement_time_model_id=None,
        measurement_uncertainty_ns=20_000_000,
        authority=_authority(),
        candidate_id="codec-candidate",
        image_size_px=_IMAGE_SIZE,
        center_norm=(0.0, 0.0),
        support_bounds_norm=(0.25, 0.25, 0.75, 0.75),
        outer_edges=EdgeSetV1(left=None, top=None, right=None, bottom=None),
        inner_edges=EdgeSetV1(left=None, top=None, right=None, bottom=None),
        inner_corners_norm=(None, None, None, None),
        fitted_inner_aperture_corners_norm=None,
        geometry_model_id=None,
        log_scale=None,
        projective_skew=None,
        clipping=FrameEdge.NONE,
        confidence=0.5,
        covariance=covariance,
        fit=FitDiagnosticsV1(residual_rms=None, inlier_count=0, support_count=0),
        health=ObservationHealth.DEGRADED,
        health_reason="codec_fixture_degraded",
        provenance="codec-fixture-v1",
    )
    primitive = observation.to_primitive()
    empty_edges = {"left": None, "top": None, "right": None, "bottom": None}
    assert primitive == {
        "schema": "aigp-vq2-gate-observation/1",
        "frame_timing": timing.to_primitive(),
        "measurement_time_monotonic_ns": timing.final_unique_packet_monotonic_ns,
        "measurement_time_basis": "camera_final_packet_proxy",
        "measurement_time_model_id": None,
        "measurement_uncertainty_ns": 20_000_000,
        "authority": _authority().to_primitive(),
        "candidate_id": "codec-candidate",
        "image_size_px": [640, 360],
        "center_norm": [0.0, 0.0],
        "support_bounds_norm": [0.25, 0.25, 0.75, 0.75],
        "outer_edges": empty_edges,
        "inner_edges": empty_edges,
        "inner_corners_norm": [None, None, None, None],
        "fitted_inner_aperture_corners_norm": None,
        "geometry_model_id": None,
        "log_scale": None,
        "projective_skew": None,
        "clipping": 0,
        "confidence": 0.5,
        "covariance": covariance_primitive,
        "fit": {"residual_rms": None, "inlier_count": 0, "support_count": 0},
        "health": "degraded",
        "health_reason": "codec_fixture_degraded",
        "provenance": "codec-fixture-v1",
    }
    assert GateObservationV1.from_primitive(primitive) == observation
