"""Versioned, authority-neutral data contracts for the build-3385 VQ2 stack.

The contracts in this module are deliberately pure values.  They do not arm,
reset, transmit, advance a gate, declare passage, or perform cleanup.  In
particular, :class:`GateAuthorityEpochV1` is issued by the safety supervisor;
camera generation and visual continuity can never manufacture one.

All wire codecs are exact.  Missing or unknown fields, booleans used as
numbers, non-finite values, unknown enum/mask values, and inconsistent timing
or covariance data fail closed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Any, ClassVar, Mapping, Optional, Sequence, Tuple


_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_VQ2_MAX_ROLL_PITCH_COMMAND_RATE_RAD_S = 0.25
_GEOMETRY_MATCH_REL_TOL = 1e-9
_GEOMETRY_MATCH_ABS_TOL = 1e-9
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_FEATURE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _exact_object(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact object")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{label} fields must be exact; "
            f"missing={sorted(keys - actual)}, unknown={sorted(actual - keys)}"
        )
    return value


def _exact_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{label} must be >= {minimum}{suffix}")
    return value


def _finite_float(
    value: Any,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return 0.0 if result == 0.0 else result


def _optional_finite_float(
    value: Any,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        return None
    return _finite_float(value, label, minimum=minimum, maximum=maximum)


def _exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact bool")
    return value


def _bounded_string(
    value: Any,
    label: str,
    *,
    maximum_length: int = 128,
    token: bool = False,
) -> str:
    if type(value) is not str or not value or len(value) > maximum_length:
        raise ValueError(
            f"{label} must be a non-empty string no longer than {maximum_length}"
        )
    if token and _TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} contains unsupported characters")
    return value


def _optional_string(
    value: Any,
    label: str,
    *,
    maximum_length: int = 128,
    token: bool = False,
) -> Optional[str]:
    if value is None:
        return None
    return _bounded_string(
        value, label, maximum_length=maximum_length, token=token
    )


def _tuple_of_floats(
    value: Any,
    length: int,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise TypeError(f"{label} must be an exact {length}-tuple")
    return tuple(
        _finite_float(
            item,
            f"{label}[{index}]",
            minimum=minimum,
            maximum=maximum,
        )
        for index, item in enumerate(value)
    )


def _tuple_of_ints(
    value: Any,
    length: int,
    label: str,
    *,
    minimum: int = 0,
) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != length:
        raise TypeError(f"{label} must be an exact {length}-tuple")
    return tuple(
        _exact_int(item, f"{label}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _primitive_tuple(
    value: Any,
    length: int,
    label: str,
) -> tuple[Any, ...]:
    if type(value) is not list or len(value) != length:
        raise TypeError(f"{label} must be an exact {length}-element array")
    return tuple(value)


def _enum_value(value: Any, enum_type: type[Enum], label: str) -> Enum:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"{label} has an unknown value") from exc


def _mask_value(value: Any, mask_type: type[IntFlag], label: str) -> IntFlag:
    raw = _exact_int(value, label)
    allowed = 0
    for member in mask_type:
        allowed |= int(member)
    if raw & ~allowed:
        raise ValueError(f"{label} contains unknown mask bits")
    return mask_type(raw)


def _validate_mask_instance(value: Any, mask_type: type[IntFlag], label: str) -> None:
    if type(value) is not mask_type:
        raise TypeError(f"{label} must be {mask_type.__name__}")
    _mask_value(int(value), mask_type, label)


def _validate_covariance(
    matrix: Any,
    dimension: int,
    label: str,
) -> tuple[tuple[float, ...], ...]:
    if type(matrix) is not tuple or len(matrix) != dimension:
        raise TypeError(f"{label} must be an exact {dimension}x{dimension} tuple")
    rows = tuple(
        _tuple_of_floats(row, dimension, f"{label}[{index}]")
        for index, row in enumerate(matrix)
    )
    scale = max(1.0, max((abs(value) for row in rows for value in row), default=1.0))
    tolerance = 1e-10 * scale
    for row in range(dimension):
        if rows[row][row] < -tolerance:
            raise ValueError(f"{label} has a negative diagonal")
        for column in range(row):
            if abs(rows[row][column] - rows[column][row]) > tolerance:
                raise ValueError(f"{label} must be symmetric")

    # Semidefinite Cholesky.  Zero pivots are allowed only when their residual
    # row/column is also zero, which accepts honest singular covariances while
    # rejecting indefinite matrices without adding a NumPy dependency here.
    lower = [[0.0] * dimension for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row + 1):
            residual = rows[row][column] - sum(
                lower[row][k] * lower[column][k] for k in range(column)
            )
            if row == column:
                if residual < -tolerance:
                    raise ValueError(f"{label} must be positive semidefinite")
                lower[row][column] = math.sqrt(max(0.0, residual))
            elif lower[column][column] > tolerance:
                lower[row][column] = residual / lower[column][column]
            elif abs(residual) > tolerance:
                raise ValueError(f"{label} must be positive semidefinite")
    return rows


class MeasurementTimeBasis(str, Enum):
    CAMERA_CAPTURE_CALIBRATED = "camera_capture_calibrated"
    CAMERA_FINAL_PACKET_PROXY = "camera_final_packet_proxy"
    IMU_SAMPLE = "imu_sample"
    IMU_PROPAGATED = "imu_propagated"


class PredictionBasis(str, Enum):
    DECISION_TIME = "decision_time"
    COMMAND_SEND_ESTIMATE = "command_send_estimate"
    COMMAND_EFFECT_ESTIMATE = "command_effect_estimate"


class LatencyEventKind(str, Enum):
    CAMERA_FIRST_PACKET = "camera_first_packet"
    CAMERA_FINAL_PACKET = "camera_final_packet"
    FRAME_REASSEMBLED = "frame_reassembled"
    DECODE_START = "decode_start"
    DECODE_END = "decode_end"
    FRAME_PUBLISHED = "frame_published"
    DETECTION_START = "detection_start"
    DETECTION_END = "detection_end"
    TRACKING_START = "tracking_start"
    TRACKING_END = "tracking_end"
    ESTIMATOR_UPDATE_START = "estimator_update_start"
    ESTIMATOR_UPDATE_END = "estimator_update_end"
    PREDICTION_START = "prediction_start"
    PREDICTION_END = "prediction_end"
    CONTROLLER_START = "controller_start"
    CONTROLLER_END = "controller_end"
    CONTROL_TICK_DUE = "control_tick_due"
    CONTROL_TICK_START = "control_tick_start"
    CONTROL_TICK_END = "control_tick_end"
    CONTROL_TICK_SKIPPED = "control_tick_skipped"
    COMMAND_SEND_START = "command_send_start"
    COMMAND_SEND_END = "command_send_end"
    ACTUATOR_SAMPLE = "actuator_sample"
    GYRO_SAMPLE = "gyro_sample"
    FRAME_DROPPED = "frame_dropped"
    DEADLINE_MISSED = "deadline_missed"


class EventOutcome(str, Enum):
    OK = "ok"
    DROPPED = "dropped"
    SKIPPED = "skipped"
    ERROR = "error"


class FrameEdge(IntFlag):
    NONE = 0
    LEFT = 1
    TOP = 2
    RIGHT = 4
    BOTTOM = 8


class TrackRole(str, Enum):
    ACTIVE = "active"
    SHADOW = "shadow"


class ObservationHealth(str, Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    UNUSABLE = "unusable"


class RelativeStateHealth(str, Enum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    COASTING = "coasting"
    UNHEALTHY = "unhealthy"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class FrameIdentityV1:
    """Camera-frame identity; ``generation`` is not reset proof."""

    SCHEMA: ClassVar[str] = "aigp-vq2-frame-identity/1"

    stream_id: str
    generation: int
    frame_id: int

    def __post_init__(self) -> None:
        _bounded_string(self.stream_id, "stream_id", token=True)
        _exact_int(self.generation, "generation")
        _exact_int(self.frame_id, "frame_id", maximum=_UINT32_MAX)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "stream_id": self.stream_id,
            "generation": self.generation,
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "FrameIdentityV1":
        row = _exact_object(
            value, {"schema", "stream_id", "generation", "frame_id"}, cls.SCHEMA
        )
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported frame identity schema")
        return cls(
            stream_id=row["stream_id"],
            generation=row["generation"],
            frame_id=row["frame_id"],
        )


@dataclass(frozen=True, slots=True)
class FrameTimingV1:
    """Fully instrumented frame timing on one host-monotonic clock.

    ``camera_source_time_ns`` is only an opaque uint64 source ordering and
    integrity token.  It is not frame identity, reset proof, or a calibrated
    capture time and is never subtracted from host-monotonic timestamps.
    """

    SCHEMA: ClassVar[str] = "aigp-vq2-frame-timing/1"

    identity: FrameIdentityV1
    camera_source_time_ns: int
    host_clock_id: str
    publication_sequence: int
    first_unique_packet_monotonic_ns: int
    final_unique_packet_monotonic_ns: int
    reassembly_complete_monotonic_ns: int
    decode_start_monotonic_ns: int
    decode_end_monotonic_ns: int
    publish_monotonic_ns: int

    def __post_init__(self) -> None:
        if type(self.identity) is not FrameIdentityV1:
            raise TypeError("identity must be FrameIdentityV1")
        _exact_int(
            self.camera_source_time_ns,
            "camera_source_time_ns",
            maximum=_UINT64_MAX,
        )
        _bounded_string(self.host_clock_id, "host_clock_id", token=True)
        _exact_int(self.publication_sequence, "publication_sequence")
        ordered = (
            self.first_unique_packet_monotonic_ns,
            self.final_unique_packet_monotonic_ns,
            self.reassembly_complete_monotonic_ns,
            self.decode_start_monotonic_ns,
            self.decode_end_monotonic_ns,
            self.publish_monotonic_ns,
        )
        for index, item in enumerate(ordered):
            _exact_int(item, f"frame timing point {index}")
        if any(later < earlier for earlier, later in zip(ordered, ordered[1:])):
            raise ValueError("frame timing points must be monotonic")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "identity": self.identity.to_primitive(),
            "camera_source_time_ns": self.camera_source_time_ns,
            "host_clock_id": self.host_clock_id,
            "publication_sequence": self.publication_sequence,
            "first_unique_packet_monotonic_ns": self.first_unique_packet_monotonic_ns,
            "final_unique_packet_monotonic_ns": self.final_unique_packet_monotonic_ns,
            "reassembly_complete_monotonic_ns": self.reassembly_complete_monotonic_ns,
            "decode_start_monotonic_ns": self.decode_start_monotonic_ns,
            "decode_end_monotonic_ns": self.decode_end_monotonic_ns,
            "publish_monotonic_ns": self.publish_monotonic_ns,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "FrameTimingV1":
        keys = {
            "schema",
            "identity",
            "camera_source_time_ns",
            "host_clock_id",
            "publication_sequence",
            "first_unique_packet_monotonic_ns",
            "final_unique_packet_monotonic_ns",
            "reassembly_complete_monotonic_ns",
            "decode_start_monotonic_ns",
            "decode_end_monotonic_ns",
            "publish_monotonic_ns",
        }
        row = _exact_object(value, keys, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported frame timing schema")
        return cls(
            identity=FrameIdentityV1.from_primitive(row["identity"]),
            camera_source_time_ns=row["camera_source_time_ns"],
            host_clock_id=row["host_clock_id"],
            publication_sequence=row["publication_sequence"],
            first_unique_packet_monotonic_ns=row[
                "first_unique_packet_monotonic_ns"
            ],
            final_unique_packet_monotonic_ns=row[
                "final_unique_packet_monotonic_ns"
            ],
            reassembly_complete_monotonic_ns=row[
                "reassembly_complete_monotonic_ns"
            ],
            decode_start_monotonic_ns=row["decode_start_monotonic_ns"],
            decode_end_monotonic_ns=row["decode_end_monotonic_ns"],
            publish_monotonic_ns=row["publish_monotonic_ns"],
        )


@dataclass(frozen=True, slots=True)
class PredictionTimeV1:
    """Measurement/decision/prediction time with explicit uncertainty basis."""

    SCHEMA: ClassVar[str] = "aigp-vq2-prediction-time/1"

    host_clock_id: str
    source_frame: Optional[FrameIdentityV1]
    source_frame_publication_sequence: Optional[int]
    source_frame_publish_monotonic_ns: Optional[int]
    measurement_time_monotonic_ns: int
    measurement_time_basis: MeasurementTimeBasis
    measurement_time_model_id: Optional[str]
    measurement_uncertainty_ns: int
    decision_time_monotonic_ns: int
    prediction_time_monotonic_ns: int
    prediction_basis: PredictionBasis
    delay_model_id: Optional[str]
    delay_uncertainty_ns: int

    def __post_init__(self) -> None:
        _bounded_string(self.host_clock_id, "host_clock_id", token=True)
        if self.source_frame is not None and type(self.source_frame) is not FrameIdentityV1:
            raise TypeError("source_frame must be FrameIdentityV1 or None")
        for name, value in (
            (
                "source_frame_publication_sequence",
                self.source_frame_publication_sequence,
            ),
            (
                "source_frame_publish_monotonic_ns",
                self.source_frame_publish_monotonic_ns,
            ),
        ):
            if value is not None:
                _exact_int(value, name)
        source_items = (
            self.source_frame,
            self.source_frame_publication_sequence,
            self.source_frame_publish_monotonic_ns,
        )
        if any(item is None for item in source_items) and not all(
            item is None for item in source_items
        ):
            raise ValueError("source frame identity/publication fields are all-or-none")
        measurement = _exact_int(
            self.measurement_time_monotonic_ns, "measurement_time_monotonic_ns"
        )
        decision = _exact_int(
            self.decision_time_monotonic_ns, "decision_time_monotonic_ns"
        )
        prediction = _exact_int(
            self.prediction_time_monotonic_ns, "prediction_time_monotonic_ns"
        )
        if not measurement <= decision <= prediction:
            raise ValueError("prediction timing must satisfy measurement <= decision <= prediction")
        if (
            self.source_frame_publish_monotonic_ns is not None
            and self.source_frame_publish_monotonic_ns > decision
        ):
            raise ValueError("source frame publication cannot postdate the decision")
        if type(self.measurement_time_basis) is not MeasurementTimeBasis:
            raise TypeError("measurement_time_basis must be MeasurementTimeBasis")
        if type(self.prediction_basis) is not PredictionBasis:
            raise TypeError("prediction_basis must be PredictionBasis")
        measurement_uncertainty = _exact_int(
            self.measurement_uncertainty_ns, "measurement_uncertainty_ns"
        )
        delay_uncertainty = _exact_int(
            self.delay_uncertainty_ns, "delay_uncertainty_ns"
        )
        _optional_string(
            self.measurement_time_model_id,
            "measurement_time_model_id",
            token=True,
        )
        _optional_string(self.delay_model_id, "delay_model_id", token=True)
        camera_basis = self.measurement_time_basis in {
            MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
            MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        }
        if camera_basis and self.source_frame is None:
            raise ValueError("camera measurement timing requires a source_frame")
        if (
            camera_basis
            and self.source_frame_publish_monotonic_ns is not None
            and measurement > self.source_frame_publish_monotonic_ns
        ):
            raise ValueError("camera measurement cannot postdate source publication")
        if self.measurement_time_basis in {
            MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
            MeasurementTimeBasis.IMU_PROPAGATED,
        }:
            if self.measurement_time_model_id is None:
                raise ValueError("calibrated/propagated measurement time requires its own model id")
            if measurement_uncertainty == 0:
                raise ValueError(
                    "calibrated/propagated measurement uncertainty must be nonzero"
                )
        elif self.measurement_time_model_id is not None:
            raise ValueError("raw/proxy measurement time cannot claim a mapping model")
        if (
            self.measurement_time_basis
            is MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY
            and measurement_uncertainty == 0
        ):
            raise ValueError("camera final-packet proxy uncertainty must be nonzero")
        if self.prediction_basis is PredictionBasis.DECISION_TIME:
            if prediction != decision:
                raise ValueError("decision-time prediction must equal decision time")
            if self.delay_model_id is not None or delay_uncertainty != 0:
                raise ValueError("decision-time prediction cannot claim a delay model")
        elif self.delay_model_id is None or delay_uncertainty == 0:
            raise ValueError("estimated send/effect prediction requires a model and nonzero uncertainty")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "host_clock_id": self.host_clock_id,
            "source_frame": (
                None if self.source_frame is None else self.source_frame.to_primitive()
            ),
            "source_frame_publication_sequence": self.source_frame_publication_sequence,
            "source_frame_publish_monotonic_ns": self.source_frame_publish_monotonic_ns,
            "measurement_time_monotonic_ns": self.measurement_time_monotonic_ns,
            "measurement_time_basis": self.measurement_time_basis.value,
            "measurement_time_model_id": self.measurement_time_model_id,
            "measurement_uncertainty_ns": self.measurement_uncertainty_ns,
            "decision_time_monotonic_ns": self.decision_time_monotonic_ns,
            "prediction_time_monotonic_ns": self.prediction_time_monotonic_ns,
            "prediction_basis": self.prediction_basis.value,
            "delay_model_id": self.delay_model_id,
            "delay_uncertainty_ns": self.delay_uncertainty_ns,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "PredictionTimeV1":
        keys = {
            "schema",
            "host_clock_id",
            "source_frame",
            "source_frame_publication_sequence",
            "source_frame_publish_monotonic_ns",
            "measurement_time_monotonic_ns",
            "measurement_time_basis",
            "measurement_time_model_id",
            "measurement_uncertainty_ns",
            "decision_time_monotonic_ns",
            "prediction_time_monotonic_ns",
            "prediction_basis",
            "delay_model_id",
            "delay_uncertainty_ns",
        }
        row = _exact_object(value, keys, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported prediction time schema")
        source = row["source_frame"]
        return cls(
            host_clock_id=row["host_clock_id"],
            source_frame=(
                None if source is None else FrameIdentityV1.from_primitive(source)
            ),
            source_frame_publication_sequence=row[
                "source_frame_publication_sequence"
            ],
            source_frame_publish_monotonic_ns=row[
                "source_frame_publish_monotonic_ns"
            ],
            measurement_time_monotonic_ns=row["measurement_time_monotonic_ns"],
            measurement_time_basis=_enum_value(
                row["measurement_time_basis"],
                MeasurementTimeBasis,
                "measurement_time_basis",
            ),
            measurement_time_model_id=row["measurement_time_model_id"],
            measurement_uncertainty_ns=row["measurement_uncertainty_ns"],
            decision_time_monotonic_ns=row["decision_time_monotonic_ns"],
            prediction_time_monotonic_ns=row["prediction_time_monotonic_ns"],
            prediction_basis=_enum_value(
                row["prediction_basis"], PredictionBasis, "prediction_basis"
            ),
            delay_model_id=row["delay_model_id"],
            delay_uncertainty_ns=row["delay_uncertainty_ns"],
        )


@dataclass(frozen=True, slots=True)
class LatencyEventV1:
    """One host-monotonic pipeline event with explicit correlation keys.

    Sensor source time is an opaque uint64 ordering/integrity token for the
    sensor stream.  It does not establish a calibrated host-clock mapping or
    causal attribution to an optional command ID.
    """

    SCHEMA: ClassVar[str] = "aigp-vq2-latency-event/1"

    event_sequence: int
    host_clock_id: str
    monotonic_ns: int
    kind: LatencyEventKind
    frame: Optional[FrameIdentityV1]
    control_tick_id: Optional[int]
    command_id: Optional[int]
    sensor_sample_id: Optional[int]
    sensor_source_time_ns: Optional[int]
    outcome: EventOutcome
    reason_code: Optional[str]
    queue_depth: Optional[int]

    def __post_init__(self) -> None:
        _exact_int(self.event_sequence, "event_sequence")
        _bounded_string(self.host_clock_id, "host_clock_id", token=True)
        _exact_int(self.monotonic_ns, "monotonic_ns")
        if type(self.kind) is not LatencyEventKind:
            raise TypeError("kind must be LatencyEventKind")
        if self.frame is not None and type(self.frame) is not FrameIdentityV1:
            raise TypeError("frame must be FrameIdentityV1 or None")
        for name, value in (
            ("control_tick_id", self.control_tick_id),
            ("command_id", self.command_id),
            ("sensor_sample_id", self.sensor_sample_id),
            ("queue_depth", self.queue_depth),
        ):
            if value is not None:
                _exact_int(value, name)
        if self.sensor_source_time_ns is not None:
            _exact_int(
                self.sensor_source_time_ns,
                "sensor_source_time_ns",
                maximum=_UINT64_MAX,
            )
        if type(self.outcome) is not EventOutcome:
            raise TypeError("outcome must be EventOutcome")
        _optional_string(self.reason_code, "reason_code", token=True)
        if self.outcome is EventOutcome.OK and self.reason_code is not None:
            raise ValueError("successful latency events cannot carry a failure reason")
        if self.outcome is not EventOutcome.OK and self.reason_code is None:
            raise ValueError("non-success latency events require a reason_code")
        if (
            self.frame is None
            and self.control_tick_id is None
            and self.command_id is None
            and self.sensor_sample_id is None
        ):
            raise ValueError(
                "latency events require a frame, control tick, command, or sensor sample id"
            )
        frame_required = self.kind in {
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
            LatencyEventKind.ESTIMATOR_UPDATE_START,
            LatencyEventKind.ESTIMATOR_UPDATE_END,
            LatencyEventKind.PREDICTION_START,
            LatencyEventKind.PREDICTION_END,
            LatencyEventKind.CONTROLLER_START,
            LatencyEventKind.CONTROLLER_END,
            LatencyEventKind.FRAME_DROPPED,
        }
        tick_required = self.kind in {
            LatencyEventKind.PREDICTION_START,
            LatencyEventKind.PREDICTION_END,
            LatencyEventKind.CONTROLLER_START,
            LatencyEventKind.CONTROLLER_END,
            LatencyEventKind.CONTROL_TICK_DUE,
            LatencyEventKind.CONTROL_TICK_START,
            LatencyEventKind.CONTROL_TICK_END,
            LatencyEventKind.CONTROL_TICK_SKIPPED,
            LatencyEventKind.COMMAND_SEND_START,
            LatencyEventKind.COMMAND_SEND_END,
            LatencyEventKind.DEADLINE_MISSED,
        }
        command_required = self.kind in {
            LatencyEventKind.COMMAND_SEND_START,
            LatencyEventKind.COMMAND_SEND_END,
        }
        sample_required = self.kind in {
            LatencyEventKind.ACTUATOR_SAMPLE,
            LatencyEventKind.GYRO_SAMPLE,
        }
        if frame_required and self.frame is None:
            raise ValueError(f"{self.kind.value} requires a frame identity")
        if tick_required and self.control_tick_id is None:
            raise ValueError(f"{self.kind.value} requires a control_tick_id")
        if command_required and self.command_id is None:
            raise ValueError(f"{self.kind.value} requires a command_id")
        if sample_required and self.sensor_sample_id is None:
            raise ValueError(f"{self.kind.value} requires a sensor_sample_id")
        if sample_required and self.sensor_source_time_ns is None:
            raise ValueError(f"{self.kind.value} requires a sensor_source_time_ns")
        if not sample_required and (
            self.sensor_sample_id is not None or self.sensor_source_time_ns is not None
        ):
            raise ValueError(
                f"{self.kind.value} cannot carry sensor sample correlation"
            )
        expected_non_success = {
            LatencyEventKind.FRAME_DROPPED: EventOutcome.DROPPED,
            LatencyEventKind.CONTROL_TICK_SKIPPED: EventOutcome.SKIPPED,
            LatencyEventKind.DEADLINE_MISSED: EventOutcome.SKIPPED,
        }
        expected_outcome = expected_non_success.get(self.kind)
        if expected_outcome is not None and self.outcome is not expected_outcome:
            raise ValueError(f"{self.kind.value} requires outcome={expected_outcome.value}")
        if expected_outcome is None and self.outcome in {
            EventOutcome.DROPPED,
            EventOutcome.SKIPPED,
        }:
            raise ValueError(f"{self.kind.value} cannot use {self.outcome.value} outcome")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "event_sequence": self.event_sequence,
            "host_clock_id": self.host_clock_id,
            "monotonic_ns": self.monotonic_ns,
            "kind": self.kind.value,
            "frame": None if self.frame is None else self.frame.to_primitive(),
            "control_tick_id": self.control_tick_id,
            "command_id": self.command_id,
            "sensor_sample_id": self.sensor_sample_id,
            "sensor_source_time_ns": self.sensor_source_time_ns,
            "outcome": self.outcome.value,
            "reason_code": self.reason_code,
            "queue_depth": self.queue_depth,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "LatencyEventV1":
        keys = {
            "schema",
            "event_sequence",
            "host_clock_id",
            "monotonic_ns",
            "kind",
            "frame",
            "control_tick_id",
            "command_id",
            "sensor_sample_id",
            "sensor_source_time_ns",
            "outcome",
            "reason_code",
            "queue_depth",
        }
        row = _exact_object(value, keys, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported latency event schema")
        return cls(
            event_sequence=row["event_sequence"],
            host_clock_id=row["host_clock_id"],
            monotonic_ns=row["monotonic_ns"],
            kind=_enum_value(row["kind"], LatencyEventKind, "kind"),
            frame=(
                None
                if row["frame"] is None
                else FrameIdentityV1.from_primitive(row["frame"])
            ),
            control_tick_id=row["control_tick_id"],
            command_id=row["command_id"],
            sensor_sample_id=row["sensor_sample_id"],
            sensor_source_time_ns=row["sensor_source_time_ns"],
            outcome=_enum_value(row["outcome"], EventOutcome, "outcome"),
            reason_code=row["reason_code"],
            queue_depth=row["queue_depth"],
        )


@dataclass(frozen=True, slots=True)
class GateAuthorityEpochV1:
    """Safety-issued gate authority plus an inclusive camera cutover watermark."""

    SCHEMA: ClassVar[str] = "aigp-vq2-gate-authority-epoch/1"

    session_id: str
    reset_epoch: int
    gate_epoch: int
    expected_gate_index: int
    race_status_sequence: int
    race_status_boot_ms: int
    camera_host_clock_id: str
    camera_stream_id: str
    camera_generation: int
    frame_publication_sequence_not_before: int
    frame_publish_monotonic_ns_not_before: int

    def __post_init__(self) -> None:
        _bounded_string(self.session_id, "session_id", token=True)
        _exact_int(self.reset_epoch, "reset_epoch")
        _exact_int(self.gate_epoch, "gate_epoch")
        _exact_int(self.expected_gate_index, "expected_gate_index")
        _exact_int(self.race_status_sequence, "race_status_sequence")
        _exact_int(self.race_status_boot_ms, "race_status_boot_ms")
        _bounded_string(self.camera_host_clock_id, "camera_host_clock_id", token=True)
        _bounded_string(self.camera_stream_id, "camera_stream_id", token=True)
        _exact_int(self.camera_generation, "camera_generation")
        _exact_int(
            self.frame_publication_sequence_not_before,
            "frame_publication_sequence_not_before",
        )
        _exact_int(
            self.frame_publish_monotonic_ns_not_before,
            "frame_publish_monotonic_ns_not_before",
        )

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "session_id": self.session_id,
            "reset_epoch": self.reset_epoch,
            "gate_epoch": self.gate_epoch,
            "expected_gate_index": self.expected_gate_index,
            "race_status_sequence": self.race_status_sequence,
            "race_status_boot_ms": self.race_status_boot_ms,
            "camera_host_clock_id": self.camera_host_clock_id,
            "camera_stream_id": self.camera_stream_id,
            "camera_generation": self.camera_generation,
            "frame_publication_sequence_not_before": self.frame_publication_sequence_not_before,
            "frame_publish_monotonic_ns_not_before": self.frame_publish_monotonic_ns_not_before,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "GateAuthorityEpochV1":
        keys = {
            "schema",
            "session_id",
            "reset_epoch",
            "gate_epoch",
            "expected_gate_index",
            "race_status_sequence",
            "race_status_boot_ms",
            "camera_host_clock_id",
            "camera_stream_id",
            "camera_generation",
            "frame_publication_sequence_not_before",
            "frame_publish_monotonic_ns_not_before",
        }
        row = _exact_object(value, keys, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported gate authority schema")
        return cls(
            session_id=row["session_id"],
            reset_epoch=row["reset_epoch"],
            gate_epoch=row["gate_epoch"],
            expected_gate_index=row["expected_gate_index"],
            race_status_sequence=row["race_status_sequence"],
            race_status_boot_ms=row["race_status_boot_ms"],
            camera_host_clock_id=row["camera_host_clock_id"],
            camera_stream_id=row["camera_stream_id"],
            camera_generation=row["camera_generation"],
            frame_publication_sequence_not_before=row[
                "frame_publication_sequence_not_before"
            ],
            frame_publish_monotonic_ns_not_before=row[
                "frame_publish_monotonic_ns_not_before"
            ],
        )


def _validate_authority_frame_cutover(
    authority: GateAuthorityEpochV1,
    *,
    host_clock_id: str,
    frame: FrameIdentityV1,
    publication_sequence: int,
    publish_monotonic_ns: int,
    label: str,
) -> None:
    if host_clock_id != authority.camera_host_clock_id:
        raise ValueError(f"{label} host clock does not match gate authority")
    if frame.stream_id != authority.camera_stream_id:
        raise ValueError(f"{label} camera stream does not match gate authority")
    if frame.generation != authority.camera_generation:
        raise ValueError(f"{label} camera generation does not match gate authority")
    if publication_sequence < authority.frame_publication_sequence_not_before:
        raise ValueError(f"{label} predates the authority publication watermark")
    if publish_monotonic_ns < authority.frame_publish_monotonic_ns_not_before:
        raise ValueError(f"{label} predates the authority host-time watermark")


@dataclass(frozen=True, slots=True)
class LineSegmentV1:
    """Visible segment in image coordinates: -1 left/top, +1 right/bottom."""

    start_norm: tuple[float, float]
    end_norm: tuple[float, float]

    def __post_init__(self) -> None:
        start = _tuple_of_floats(
            self.start_norm, 2, "start_norm", minimum=-1.0, maximum=1.0
        )
        end = _tuple_of_floats(
            self.end_norm, 2, "end_norm", minimum=-1.0, maximum=1.0
        )
        if start == end:
            raise ValueError("line segment endpoints must differ")
        object.__setattr__(self, "start_norm", start)
        object.__setattr__(self, "end_norm", end)

    def to_primitive(self) -> list[list[float]]:
        return [list(self.start_norm), list(self.end_norm)]

    @classmethod
    def from_primitive(cls, value: Any) -> "LineSegmentV1":
        endpoints = _primitive_tuple(value, 2, "line segment")
        return cls(
            start_norm=_primitive_tuple(endpoints[0], 2, "line start"),
            end_norm=_primitive_tuple(endpoints[1], 2, "line end"),
        )


@dataclass(frozen=True, slots=True)
class EdgeSetV1:
    left: Optional[LineSegmentV1] = None
    top: Optional[LineSegmentV1] = None
    right: Optional[LineSegmentV1] = None
    bottom: Optional[LineSegmentV1] = None

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            value = getattr(self, name)
            if value is not None and type(value) is not LineSegmentV1:
                raise TypeError(f"{name} must be LineSegmentV1 or None")

    @property
    def visibility(self) -> FrameEdge:
        result = FrameEdge.NONE
        for edge, name in (
            (FrameEdge.LEFT, "left"),
            (FrameEdge.TOP, "top"),
            (FrameEdge.RIGHT, "right"),
            (FrameEdge.BOTTOM, "bottom"),
        ):
            if getattr(self, name) is not None:
                result |= edge
        return result

    def to_primitive(self) -> dict[str, Any]:
        return {
            name: None if getattr(self, name) is None else getattr(self, name).to_primitive()
            for name in ("left", "top", "right", "bottom")
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "EdgeSetV1":
        keys = {"left", "top", "right", "bottom"}
        row = _exact_object(value, keys, "edge set")
        return cls(
            **{
                name: (
                    None
                    if row[name] is None
                    else LineSegmentV1.from_primitive(row[name])
                )
                for name in keys
            }
        )


@dataclass(frozen=True, slots=True)
class FeatureCovarianceV1:
    model_id: str
    feature_order: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        _bounded_string(self.model_id, "covariance model_id", token=True)
        if type(self.feature_order) is not tuple or not self.feature_order:
            raise TypeError("feature_order must be a non-empty tuple")
        if len(self.feature_order) > 9:
            raise ValueError("feature covariance dimension exceeds contract maximum")
        for item in self.feature_order:
            if type(item) is not str or _FEATURE_RE.fullmatch(item) is None:
                raise ValueError("feature_order contains an invalid feature name")
        if len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("feature_order must not contain duplicates")
        matrix = _validate_covariance(
            self.matrix, len(self.feature_order), "feature covariance"
        )
        if any(matrix[index][index] <= 0.0 for index in range(len(matrix))):
            raise ValueError("feature covariance variances must be strictly positive")
        object.__setattr__(self, "matrix", matrix)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "feature_order": list(self.feature_order),
            "matrix": [list(row) for row in self.matrix],
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "FeatureCovarianceV1":
        row = _exact_object(
            value, {"model_id", "feature_order", "matrix"}, "feature covariance"
        )
        if (
            type(row["feature_order"]) is not list
            or type(row["matrix"]) is not list
            or any(type(item) is not list for item in row["matrix"])
        ):
            raise TypeError("covariance feature_order and matrix must be arrays")
        return cls(
            model_id=row["model_id"],
            feature_order=tuple(row["feature_order"]),
            matrix=tuple(tuple(item) for item in row["matrix"]),
        )


@dataclass(frozen=True, slots=True)
class FitDiagnosticsV1:
    residual_rms: Optional[float]
    inlier_count: int
    support_count: int

    def __post_init__(self) -> None:
        residual = _optional_finite_float(
            self.residual_rms, "residual_rms", minimum=0.0
        )
        _exact_int(self.inlier_count, "inlier_count")
        _exact_int(self.support_count, "support_count")
        if self.inlier_count > self.support_count:
            raise ValueError("inlier_count cannot exceed support_count")
        if self.support_count == 0 and residual is not None:
            raise ValueError("a residual requires at least one support sample")
        object.__setattr__(self, "residual_rms", residual)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "residual_rms": self.residual_rms,
            "inlier_count": self.inlier_count,
            "support_count": self.support_count,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "FitDiagnosticsV1":
        row = _exact_object(
            value, {"residual_rms", "inlier_count", "support_count"}, "fit diagnostics"
        )
        return cls(
            residual_rms=row["residual_rms"],
            inlier_count=row["inlier_count"],
            support_count=row["support_count"],
        )


_OBSERVATION_FEATURES = frozenset(
    {"center_x_norm", "center_y_norm", "log_scale", "skew_x", "skew_y"}
)
_STATE_FEATURE_ORDER = (
    "bearing_x_norm",
    "bearing_y_norm",
    "log_scale",
    "bearing_rate_x_norm_s",
    "bearing_rate_y_norm_s",
    "expansion_rate_s",
)
_METRIC_STATE_FEATURE_ORDER = (
    "position_x_body_frd_m",
    "position_y_body_frd_m",
    "position_z_body_frd_m",
    "velocity_x_body_frd_m_s",
    "velocity_y_body_frd_m_s",
    "velocity_z_body_frd_m_s",
    "orientation_error_x_rad",
    "orientation_error_y_rad",
    "orientation_error_z_rad",
)


def _fitted_aperture_geometry(
    value: Any,
) -> tuple[
    tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    float,
    tuple[float, float],
    tuple[float, float],
]:
    if type(value) is not tuple or len(value) != 4:
        raise TypeError("fitted inner aperture must be an exact four-tuple")
    points = tuple(
        _tuple_of_floats(
            point,
            2,
            f"fitted inner aperture corner {index}",
            minimum=-4.0,
            maximum=4.0,
        )
        for index, point in enumerate(value)
    )
    crosses = []
    for index in range(4):
        first = points[index]
        second = points[(index + 1) % 4]
        third = points[(index + 2) % 4]
        crosses.append(
            (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0])
        )
    if any(cross <= 1e-12 for cross in crosses):
        raise ValueError(
            "fitted inner aperture must be a strictly convex clockwise "
            "top-left/top-right/bottom-right/bottom-left quad"
        )
    top_midpoint_y = 0.5 * (points[0][1] + points[1][1])
    bottom_midpoint_y = 0.5 * (points[3][1] + points[2][1])
    left_midpoint_x = 0.5 * (points[0][0] + points[3][0])
    right_midpoint_x = 0.5 * (points[1][0] + points[2][0])
    if not top_midpoint_y < bottom_midpoint_y:
        raise ValueError("fitted aperture top/bottom corner labels are ambiguous")
    if not left_midpoint_x < right_midpoint_x:
        raise ValueError("fitted aperture left/right corner labels are ambiguous")
    area = 0.5 * abs(
        sum(
            points[index][0] * points[(index + 1) % 4][1]
            - points[(index + 1) % 4][0] * points[index][1]
            for index in range(4)
        )
    )
    lengths = tuple(
        math.hypot(
            points[(index + 1) % 4][0] - points[index][0],
            points[(index + 1) % 4][1] - points[index][1],
        )
        for index in range(4)
    )
    top, right, bottom, left = lengths
    first_diagonal = (
        points[2][0] - points[0][0],
        points[2][1] - points[0][1],
    )
    second_diagonal = (
        points[3][0] - points[1][0],
        points[3][1] - points[1][1],
    )
    diagonal_offset = (
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
    )
    diagonal_cross = (
        first_diagonal[0] * second_diagonal[1]
        - first_diagonal[1] * second_diagonal[0]
    )
    if abs(diagonal_cross) <= 1e-12:
        raise ValueError("fitted inner aperture diagonals must intersect uniquely")
    first_fraction = (
        diagonal_offset[0] * second_diagonal[1]
        - diagonal_offset[1] * second_diagonal[0]
    ) / diagonal_cross
    fitted_center = (
        points[0][0] + first_fraction * first_diagonal[0],
        points[0][1] + first_fraction * first_diagonal[1],
    )
    return (
        points,
        area,
        (math.log(right / left), math.log(bottom / top)),
        fitted_center,
    )


@dataclass(frozen=True, slots=True)
class GateObservationV1:
    """One unassociated perception candidate in normalized image coordinates.

    ``center_norm``, visible edge endpoints, corners, and skew use ``x`` right
    and ``y`` down with image borders at -1/+1.  A censored inferred center may
    extend to +/-4; genuinely visible edge endpoints/corners must stay in the
    image. ``support_bounds_norm`` alone uses the conventional [0, 1] bbox
    interval. Inner corners are ordered top-left, top-right, bottom-right,
    bottom-left. ``log_scale`` is the natural logarithm of the square root of
    the fitted inner-aperture polygon area in normalized image units.
    ``projective_skew`` is ``(log(right/left edge length),
    log(bottom/top edge length))`` for a fit that supports all four positive
    lengths; zero is fronto-parallel symmetry. For a fitted aperture,
    ``center_norm`` is the projected aperture center at the quadrilateral's
    diagonal intersection and must match it within relative/absolute ``1e-9``.
    Active/shadow association belongs to the tracker, not this observation.
    """

    SCHEMA: ClassVar[str] = "aigp-vq2-gate-observation/1"

    frame_timing: FrameTimingV1
    measurement_time_monotonic_ns: int
    measurement_time_basis: MeasurementTimeBasis
    measurement_time_model_id: Optional[str]
    measurement_uncertainty_ns: int
    authority: GateAuthorityEpochV1
    candidate_id: str
    image_size_px: tuple[int, int]
    center_norm: tuple[float, float]
    support_bounds_norm: tuple[float, float, float, float]
    outer_edges: EdgeSetV1
    inner_edges: EdgeSetV1
    inner_corners_norm: tuple[
        Optional[tuple[float, float]],
        Optional[tuple[float, float]],
        Optional[tuple[float, float]],
        Optional[tuple[float, float]],
    ]
    fitted_inner_aperture_corners_norm: Optional[
        tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ]
    ]
    geometry_model_id: Optional[str]
    log_scale: Optional[float]
    projective_skew: Optional[tuple[float, float]]
    clipping: FrameEdge
    confidence: float
    covariance: FeatureCovarianceV1
    fit: FitDiagnosticsV1
    health: ObservationHealth
    health_reason: Optional[str]
    provenance: str

    def __post_init__(self) -> None:
        if type(self.frame_timing) is not FrameTimingV1:
            raise TypeError("frame_timing must be FrameTimingV1")
        measurement_time = _exact_int(
            self.measurement_time_monotonic_ns, "measurement_time_monotonic_ns"
        )
        if type(self.measurement_time_basis) is not MeasurementTimeBasis:
            raise TypeError("measurement_time_basis must be MeasurementTimeBasis")
        if self.measurement_time_basis not in {
            MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED,
            MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        }:
            raise ValueError("gate observations require a camera measurement time basis")
        _optional_string(
            self.measurement_time_model_id,
            "measurement_time_model_id",
            token=True,
        )
        measurement_uncertainty = _exact_int(
            self.measurement_uncertainty_ns, "measurement_uncertainty_ns"
        )
        if self.measurement_time_basis is MeasurementTimeBasis.CAMERA_CAPTURE_CALIBRATED:
            if self.measurement_time_model_id is None:
                raise ValueError("calibrated camera measurement requires its own model id")
            if measurement_uncertainty == 0:
                raise ValueError(
                    "calibrated camera measurement uncertainty must be nonzero"
                )
        else:
            if self.measurement_time_model_id is not None:
                raise ValueError("camera final-packet proxy cannot claim a mapping model")
            if measurement_uncertainty == 0:
                raise ValueError("camera final-packet proxy uncertainty must be nonzero")
            if measurement_time != self.frame_timing.final_unique_packet_monotonic_ns:
                raise ValueError("final-packet proxy must use the frame final-packet time")
        if measurement_time > self.frame_timing.publish_monotonic_ns:
            raise ValueError("observation measurement cannot postdate frame publication")
        if type(self.authority) is not GateAuthorityEpochV1:
            raise TypeError("authority must be GateAuthorityEpochV1")
        _validate_authority_frame_cutover(
            self.authority,
            host_clock_id=self.frame_timing.host_clock_id,
            frame=self.frame_timing.identity,
            publication_sequence=self.frame_timing.publication_sequence,
            publish_monotonic_ns=self.frame_timing.publish_monotonic_ns,
            label="observation frame",
        )
        _bounded_string(self.candidate_id, "candidate_id", token=True)
        image_size = _tuple_of_ints(self.image_size_px, 2, "image_size_px", minimum=1)
        center = _tuple_of_floats(
            self.center_norm, 2, "center_norm", minimum=-4.0, maximum=4.0
        )
        bounds = _tuple_of_floats(
            self.support_bounds_norm,
            4,
            "support_bounds_norm",
            minimum=0.0,
            maximum=1.0,
        )
        if not bounds[0] < bounds[2] or not bounds[1] < bounds[3]:
            raise ValueError(
                "support bounds must have positive width and height in "
                "left/top/right/bottom order"
            )
        _validate_mask_instance(self.clipping, FrameEdge, "clipping")
        required_clipping = FrameEdge.NONE
        if center[0] < -1.0:
            required_clipping |= FrameEdge.LEFT
        elif center[0] > 1.0:
            required_clipping |= FrameEdge.RIGHT
        if center[1] < -1.0:
            required_clipping |= FrameEdge.TOP
        elif center[1] > 1.0:
            required_clipping |= FrameEdge.BOTTOM
        if required_clipping & ~self.clipping:
            raise ValueError(
                "an inferred center outside the image requires matching clipping"
            )
        if type(self.outer_edges) is not EdgeSetV1 or type(self.inner_edges) is not EdgeSetV1:
            raise TypeError("inner_edges and outer_edges must be EdgeSetV1")
        if self.outer_edges.visibility & self.clipping:
            raise ValueError("a clipped outer edge cannot also be marked visible")
        if type(self.inner_corners_norm) is not tuple or len(self.inner_corners_norm) != 4:
            raise TypeError("inner_corners_norm must be an exact four-tuple")
        corners: list[Optional[tuple[float, float]]] = []
        for index, corner in enumerate(self.inner_corners_norm):
            if corner is None:
                corners.append(None)
            else:
                corners.append(
                    _tuple_of_floats(
                        corner,
                        2,
                        f"inner corner {index}",
                        minimum=-1.0,
                        maximum=1.0,
                    )
                )
        log_scale = _optional_finite_float(self.log_scale, "log_scale")
        skew = None
        if self.projective_skew is not None:
            skew = _tuple_of_floats(self.projective_skew, 2, "projective_skew")
        _optional_string(self.geometry_model_id, "geometry_model_id", token=True)
        fitted = None
        if self.fitted_inner_aperture_corners_norm is not None:
            fitted, fitted_area, fitted_skew, fitted_center = _fitted_aperture_geometry(
                self.fitted_inner_aperture_corners_norm
            )
            if self.geometry_model_id is None or log_scale is None or skew is None:
                raise ValueError(
                    "fitted aperture, geometry model, log_scale, and skew are all-or-none"
                )
            expected_log_scale = math.log(math.sqrt(fitted_area))
            if not math.isclose(
                log_scale, expected_log_scale, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError("log_scale does not match fitted aperture area")
            if any(
                not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9)
                for observed, expected in zip(skew, fitted_skew)
            ):
                raise ValueError("projective_skew does not match fitted aperture edges")
            if any(
                not math.isclose(
                    observed,
                    expected,
                    rel_tol=_GEOMETRY_MATCH_REL_TOL,
                    abs_tol=_GEOMETRY_MATCH_ABS_TOL,
                )
                for observed, expected in zip(center, fitted_center)
            ):
                raise ValueError(
                    "center_norm does not match the fitted aperture diagonal intersection"
                )
            fitted_required_clipping = FrameEdge.NONE
            for fitted_x, fitted_y in fitted:
                if fitted_x < -1.0:
                    fitted_required_clipping |= FrameEdge.LEFT
                elif fitted_x > 1.0:
                    fitted_required_clipping |= FrameEdge.RIGHT
                if fitted_y < -1.0:
                    fitted_required_clipping |= FrameEdge.TOP
                elif fitted_y > 1.0:
                    fitted_required_clipping |= FrameEdge.BOTTOM
            if fitted_required_clipping & ~self.clipping:
                raise ValueError(
                    "an out-of-frame fitted aperture requires matching clipping"
                )
            for index, visible_corner in enumerate(corners):
                if visible_corner is not None and visible_corner != fitted[index]:
                    raise ValueError(
                        "visible inner corner does not match the fitted aperture"
                    )
        elif (
            self.geometry_model_id is not None or log_scale is not None or skew is not None
        ):
            raise ValueError(
                "geometry model, log_scale, and skew require a fitted inner aperture"
            )
        confidence = _finite_float(
            self.confidence, "confidence", minimum=0.0, maximum=1.0
        )
        if type(self.covariance) is not FeatureCovarianceV1:
            raise TypeError("covariance must be FeatureCovarianceV1")
        if not set(self.covariance.feature_order) <= _OBSERVATION_FEATURES:
            raise ValueError("observation covariance contains an unknown feature")
        if "center_x_norm" not in self.covariance.feature_order or "center_y_norm" not in self.covariance.feature_order:
            raise ValueError("observation covariance must cover both center features")
        if (log_scale is None) != ("log_scale" not in self.covariance.feature_order):
            raise ValueError("log_scale covariance presence must match log_scale")
        skew_features = {"skew_x", "skew_y"} & set(self.covariance.feature_order)
        if skew_features not in (set(), {"skew_x", "skew_y"}):
            raise ValueError("skew covariance must include both skew axes")
        has_skew_covariance = bool(skew_features)
        if (skew is not None) != has_skew_covariance:
            raise ValueError("skew covariance presence must match projective_skew")
        if type(self.fit) is not FitDiagnosticsV1:
            raise TypeError("fit must be FitDiagnosticsV1")
        if fitted is not None and (
            self.fit.support_count == 0
            or self.fit.inlier_count == 0
            or self.fit.residual_rms is None
        ):
            raise ValueError("fitted aperture requires nonzero support and a residual")
        if type(self.health) is not ObservationHealth:
            raise TypeError("health must be ObservationHealth")
        _optional_string(self.health_reason, "health_reason", maximum_length=256)
        if self.health is ObservationHealth.NOMINAL and self.health_reason is not None:
            raise ValueError("nominal observations cannot carry a health reason")
        if self.health is ObservationHealth.NOMINAL and fitted is None:
            raise ValueError("nominal observation requires a fitted inner aperture")
        if self.health is not ObservationHealth.NOMINAL and self.health_reason is None:
            raise ValueError("degraded/unusable observations require a health reason")
        _bounded_string(self.provenance, "provenance", token=True)
        object.__setattr__(self, "image_size_px", image_size)
        object.__setattr__(self, "center_norm", center)
        object.__setattr__(self, "support_bounds_norm", bounds)
        object.__setattr__(self, "inner_corners_norm", tuple(corners))
        object.__setattr__(self, "fitted_inner_aperture_corners_norm", fitted)
        object.__setattr__(self, "log_scale", log_scale)
        object.__setattr__(self, "projective_skew", skew)
        object.__setattr__(self, "confidence", confidence)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "frame_timing": self.frame_timing.to_primitive(),
            "measurement_time_monotonic_ns": self.measurement_time_monotonic_ns,
            "measurement_time_basis": self.measurement_time_basis.value,
            "measurement_time_model_id": self.measurement_time_model_id,
            "measurement_uncertainty_ns": self.measurement_uncertainty_ns,
            "authority": self.authority.to_primitive(),
            "candidate_id": self.candidate_id,
            "image_size_px": list(self.image_size_px),
            "center_norm": list(self.center_norm),
            "support_bounds_norm": list(self.support_bounds_norm),
            "outer_edges": self.outer_edges.to_primitive(),
            "inner_edges": self.inner_edges.to_primitive(),
            "inner_corners_norm": [
                None if corner is None else list(corner)
                for corner in self.inner_corners_norm
            ],
            "fitted_inner_aperture_corners_norm": (
                None
                if self.fitted_inner_aperture_corners_norm is None
                else [list(corner) for corner in self.fitted_inner_aperture_corners_norm]
            ),
            "geometry_model_id": self.geometry_model_id,
            "log_scale": self.log_scale,
            "projective_skew": (
                None if self.projective_skew is None else list(self.projective_skew)
            ),
            "clipping": int(self.clipping),
            "confidence": self.confidence,
            "covariance": self.covariance.to_primitive(),
            "fit": self.fit.to_primitive(),
            "health": self.health.value,
            "health_reason": self.health_reason,
            "provenance": self.provenance,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "GateObservationV1":
        keys = {
            "schema",
            "frame_timing",
            "measurement_time_monotonic_ns",
            "measurement_time_basis",
            "measurement_time_model_id",
            "measurement_uncertainty_ns",
            "authority",
            "candidate_id",
            "image_size_px",
            "center_norm",
            "support_bounds_norm",
            "outer_edges",
            "inner_edges",
            "inner_corners_norm",
            "fitted_inner_aperture_corners_norm",
            "geometry_model_id",
            "log_scale",
            "projective_skew",
            "clipping",
            "confidence",
            "covariance",
            "fit",
            "health",
            "health_reason",
            "provenance",
        }
        row = _exact_object(value, keys, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported gate observation schema")
        raw_corners = _primitive_tuple(row["inner_corners_norm"], 4, "inner corners")
        raw_fitted = row["fitted_inner_aperture_corners_norm"]
        raw_skew = row["projective_skew"]
        return cls(
            frame_timing=FrameTimingV1.from_primitive(row["frame_timing"]),
            measurement_time_monotonic_ns=row["measurement_time_monotonic_ns"],
            measurement_time_basis=_enum_value(
                row["measurement_time_basis"],
                MeasurementTimeBasis,
                "measurement_time_basis",
            ),
            measurement_time_model_id=row["measurement_time_model_id"],
            measurement_uncertainty_ns=row["measurement_uncertainty_ns"],
            authority=GateAuthorityEpochV1.from_primitive(row["authority"]),
            candidate_id=row["candidate_id"],
            image_size_px=_primitive_tuple(row["image_size_px"], 2, "image_size_px"),
            center_norm=_primitive_tuple(row["center_norm"], 2, "center_norm"),
            support_bounds_norm=_primitive_tuple(
                row["support_bounds_norm"], 4, "support_bounds_norm"
            ),
            outer_edges=EdgeSetV1.from_primitive(row["outer_edges"]),
            inner_edges=EdgeSetV1.from_primitive(row["inner_edges"]),
            inner_corners_norm=tuple(
                None if item is None else _primitive_tuple(item, 2, "inner corner")
                for item in raw_corners
            ),
            fitted_inner_aperture_corners_norm=(
                None
                if raw_fitted is None
                else tuple(
                    _primitive_tuple(item, 2, "fitted inner aperture corner")
                    for item in _primitive_tuple(
                        raw_fitted, 4, "fitted inner aperture"
                    )
                )
            ),
            geometry_model_id=row["geometry_model_id"],
            log_scale=row["log_scale"],
            projective_skew=(
                None
                if raw_skew is None
                else _primitive_tuple(raw_skew, 2, "projective_skew")
            ),
            clipping=_mask_value(row["clipping"], FrameEdge, "clipping"),
            confidence=row["confidence"],
            covariance=FeatureCovarianceV1.from_primitive(row["covariance"]),
            fit=FitDiagnosticsV1.from_primitive(row["fit"]),
            health=_enum_value(row["health"], ObservationHealth, "health"),
            health_reason=row["health_reason"],
            provenance=row["provenance"],
        )

    @property
    def frame(self) -> FrameIdentityV1:
        return self.frame_timing.identity

    @property
    def host_clock_id(self) -> str:
        return self.frame_timing.host_clock_id


def validate_gate_observation_batch(
    observations: Sequence[GateObservationV1],
) -> None:
    """Require one timing/authority and unique candidate IDs per camera frame."""

    if type(observations) is not tuple:
        raise TypeError("gate observation batch must be an exact tuple")
    frame_bindings: dict[
        tuple[str, FrameIdentityV1],
        tuple[FrameTimingV1, GateAuthorityEpochV1],
    ] = {}
    candidate_ids: set[tuple[str, FrameIdentityV1, str]] = set()
    for observation in observations:
        if type(observation) is not GateObservationV1:
            raise TypeError("gate observation batch contains a non-contract value")
        frame_key = (observation.host_clock_id, observation.frame)
        binding = (observation.frame_timing, observation.authority)
        previous_binding = frame_bindings.get(frame_key)
        if previous_binding is not None and binding != previous_binding:
            raise ValueError(
                "one camera frame cannot carry multiple timing/authority bindings"
            )
        frame_bindings[frame_key] = binding
        candidate_key = (*frame_key, observation.candidate_id)
        if candidate_key in candidate_ids:
            raise ValueError("candidate_id must be unique within one camera frame")
        candidate_ids.add(candidate_key)


@dataclass(frozen=True, slots=True)
class RelativeGateStateV1:
    """Predicted gate-relative feature state.

    Bearing coordinates use x-right/y-down normalized image coordinates.
    ``log_scale`` inherits the observation definition. Optional metric state
    uses the gate-center translation and relative velocity expressed in body
    FRD. The unit ``(x, y, z, w)`` quaternion actively rotates a gate-local
    vector into body FRD; gate-local +x is aperture-right, +y is down, and +z
    is the reviewed travel-direction normal. Producers that cannot disambiguate
    that normal leave the entire metric state absent. Its 9x9 covariance uses
    the fixed position, velocity, body-FRD small-angle-error order above.
    """

    SCHEMA: ClassVar[str] = "aigp-vq2-relative-gate-state/1"

    timing: PredictionTimeV1
    authority: GateAuthorityEpochV1
    tracker_id: str
    state_sequence: int
    measurement_update_sequence: int
    source_candidate_id: str
    track_role: TrackRole
    bearing_norm: tuple[float, float]
    bearing_rate_norm_s: tuple[float, float]
    log_scale: float
    expansion_rate_s: float
    covariance: FeatureCovarianceV1
    metric_position_body_frd_m: Optional[tuple[float, float, float]]
    metric_velocity_body_frd_m_s: Optional[tuple[float, float, float]]
    metric_gate_orientation_body_frd_xyzw: Optional[
        tuple[float, float, float, float]
    ]
    metric_covariance: Optional[FeatureCovarianceV1]
    last_clipping: FrameEdge
    outer_visibility: FrameEdge
    inner_visibility: FrameEdge
    normalized_innovation_squared: Optional[float]
    innovation_gate_threshold: Optional[float]
    innovation_accepted: Optional[bool]
    dropout_count: int
    health: RelativeStateHealth
    health_reason: Optional[str]

    def __post_init__(self) -> None:
        if type(self.timing) is not PredictionTimeV1:
            raise TypeError("timing must be PredictionTimeV1")
        if self.timing.source_frame is None:
            raise ValueError("relative gate state requires a source frame")
        if type(self.authority) is not GateAuthorityEpochV1:
            raise TypeError("authority must be GateAuthorityEpochV1")
        if (
            self.timing.source_frame is None
            or self.timing.source_frame_publication_sequence is None
            or self.timing.source_frame_publish_monotonic_ns is None
        ):
            raise ValueError("relative state source frame publication is incomplete")
        _validate_authority_frame_cutover(
            self.authority,
            host_clock_id=self.timing.host_clock_id,
            frame=self.timing.source_frame,
            publication_sequence=self.timing.source_frame_publication_sequence,
            publish_monotonic_ns=self.timing.source_frame_publish_monotonic_ns,
            label="relative-state source frame",
        )
        _bounded_string(self.tracker_id, "tracker_id", token=True)
        _exact_int(self.state_sequence, "state_sequence")
        _exact_int(self.measurement_update_sequence, "measurement_update_sequence")
        _bounded_string(self.source_candidate_id, "source_candidate_id", token=True)
        if type(self.track_role) is not TrackRole:
            raise TypeError("track_role must be TrackRole")
        bearing = _tuple_of_floats(
            self.bearing_norm, 2, "bearing_norm", minimum=-4.0, maximum=4.0
        )
        bearing_rate = _tuple_of_floats(
            self.bearing_rate_norm_s, 2, "bearing_rate_norm_s"
        )
        log_scale = _finite_float(self.log_scale, "log_scale")
        expansion = _finite_float(self.expansion_rate_s, "expansion_rate_s")
        if type(self.covariance) is not FeatureCovarianceV1:
            raise TypeError("covariance must be FeatureCovarianceV1")
        if self.covariance.feature_order != _STATE_FEATURE_ORDER:
            raise ValueError("relative-state covariance has the wrong feature order")
        metric_items = (
            self.metric_position_body_frd_m,
            self.metric_velocity_body_frd_m_s,
            self.metric_gate_orientation_body_frd_xyzw,
            self.metric_covariance,
        )
        if any(item is None for item in metric_items) and not all(
            item is None for item in metric_items
        ):
            raise ValueError(
                "metric position, velocity, orientation, and covariance are all-or-none"
            )
        position = velocity = orientation = None
        if self.metric_position_body_frd_m is not None:
            position = _tuple_of_floats(
                self.metric_position_body_frd_m, 3, "metric_position_body_frd_m"
            )
            velocity = _tuple_of_floats(
                self.metric_velocity_body_frd_m_s,
                3,
                "metric_velocity_body_frd_m_s",
            )
            orientation = _tuple_of_floats(
                self.metric_gate_orientation_body_frd_xyzw,
                4,
                "metric_gate_orientation_body_frd_xyzw",
            )
            orientation_norm = math.sqrt(sum(value * value for value in orientation))
            if abs(orientation_norm - 1.0) > 1e-6:
                raise ValueError("metric gate orientation quaternion must be unit length")
            if type(self.metric_covariance) is not FeatureCovarianceV1:
                raise TypeError("metric_covariance must be FeatureCovarianceV1")
            if self.metric_covariance.feature_order != _METRIC_STATE_FEATURE_ORDER:
                raise ValueError("metric covariance has the wrong feature order")
        for name, value in (
            ("last_clipping", self.last_clipping),
            ("outer_visibility", self.outer_visibility),
            ("inner_visibility", self.inner_visibility),
        ):
            _validate_mask_instance(value, FrameEdge, name)
        if self.outer_visibility & self.last_clipping:
            raise ValueError("a clipped outer edge cannot also be visible")
        nis = _optional_finite_float(
            self.normalized_innovation_squared,
            "normalized_innovation_squared",
            minimum=0.0,
        )
        threshold = _optional_finite_float(
            self.innovation_gate_threshold,
            "innovation_gate_threshold",
            minimum=0.0,
        )
        if threshold == 0.0:
            raise ValueError("innovation_gate_threshold must be positive")
        if self.innovation_accepted is not None:
            _exact_bool(self.innovation_accepted, "innovation_accepted")
        innovation_items = (nis, threshold, self.innovation_accepted)
        if any(item is None for item in innovation_items) and not all(
            item is None for item in innovation_items
        ):
            raise ValueError("innovation diagnostics are all-or-none")
        if self.innovation_accepted is True and nis is not None and nis > threshold:
            raise ValueError("accepted innovation exceeds its gate threshold")
        if self.innovation_accepted is False and nis is not None and nis <= threshold:
            raise ValueError("rejected innovation does not exceed its gate threshold")
        dropout_count = _exact_int(self.dropout_count, "dropout_count")
        if type(self.health) is not RelativeStateHealth:
            raise TypeError("health must be RelativeStateHealth")
        _optional_string(self.health_reason, "health_reason", maximum_length=256)
        if self.health is RelativeStateHealth.HEALTHY and self.health_reason is not None:
            raise ValueError("healthy states cannot carry a health reason")
        if self.health is RelativeStateHealth.HEALTHY and dropout_count != 0:
            raise ValueError("healthy states cannot contain dropout predictions")
        if (
            self.health is RelativeStateHealth.HEALTHY
            and self.innovation_accepted is False
        ):
            raise ValueError("healthy states cannot report a rejected innovation")
        if self.health in {RelativeStateHealth.COASTING, RelativeStateHealth.LOST} and dropout_count == 0:
            raise ValueError("coasting/lost states require at least one dropout")
        if dropout_count > 0 and self.health not in {
            RelativeStateHealth.COASTING,
            RelativeStateHealth.LOST,
        }:
            raise ValueError("dropout predictions require coasting or lost health")
        if dropout_count > 0 and not all(item is None for item in innovation_items):
            raise ValueError(
                "dropout predictions cannot claim a current-frame innovation result"
            )
        if self.health in {
            RelativeStateHealth.DEGRADED,
            RelativeStateHealth.COASTING,
            RelativeStateHealth.UNHEALTHY,
            RelativeStateHealth.LOST,
        } and self.health_reason is None:
            raise ValueError("non-healthy terminal/degraded states require a reason")
        object.__setattr__(self, "bearing_norm", bearing)
        object.__setattr__(self, "bearing_rate_norm_s", bearing_rate)
        object.__setattr__(self, "log_scale", log_scale)
        object.__setattr__(self, "expansion_rate_s", expansion)
        object.__setattr__(self, "metric_position_body_frd_m", position)
        object.__setattr__(self, "metric_velocity_body_frd_m_s", velocity)
        object.__setattr__(self, "metric_gate_orientation_body_frd_xyzw", orientation)
        object.__setattr__(self, "normalized_innovation_squared", nis)
        object.__setattr__(self, "innovation_gate_threshold", threshold)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "timing": self.timing.to_primitive(),
            "authority": self.authority.to_primitive(),
            "tracker_id": self.tracker_id,
            "state_sequence": self.state_sequence,
            "measurement_update_sequence": self.measurement_update_sequence,
            "source_candidate_id": self.source_candidate_id,
            "track_role": self.track_role.value,
            "bearing_norm": list(self.bearing_norm),
            "bearing_rate_norm_s": list(self.bearing_rate_norm_s),
            "log_scale": self.log_scale,
            "expansion_rate_s": self.expansion_rate_s,
            "covariance": self.covariance.to_primitive(),
            "metric_position_body_frd_m": (
                None
                if self.metric_position_body_frd_m is None
                else list(self.metric_position_body_frd_m)
            ),
            "metric_velocity_body_frd_m_s": (
                None
                if self.metric_velocity_body_frd_m_s is None
                else list(self.metric_velocity_body_frd_m_s)
            ),
            "metric_gate_orientation_body_frd_xyzw": (
                None
                if self.metric_gate_orientation_body_frd_xyzw is None
                else list(self.metric_gate_orientation_body_frd_xyzw)
            ),
            "metric_covariance": (
                None
                if self.metric_covariance is None
                else self.metric_covariance.to_primitive()
            ),
            "last_clipping": int(self.last_clipping),
            "outer_visibility": int(self.outer_visibility),
            "inner_visibility": int(self.inner_visibility),
            "normalized_innovation_squared": self.normalized_innovation_squared,
            "innovation_gate_threshold": self.innovation_gate_threshold,
            "innovation_accepted": self.innovation_accepted,
            "dropout_count": self.dropout_count,
            "health": self.health.value,
            "health_reason": self.health_reason,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "RelativeGateStateV1":
        keys = {
            "schema",
            "timing",
            "authority",
            "tracker_id",
            "state_sequence",
            "measurement_update_sequence",
            "source_candidate_id",
            "track_role",
            "bearing_norm",
            "bearing_rate_norm_s",
            "log_scale",
            "expansion_rate_s",
            "covariance",
            "metric_position_body_frd_m",
            "metric_velocity_body_frd_m_s",
            "metric_gate_orientation_body_frd_xyzw",
            "metric_covariance",
            "last_clipping",
            "outer_visibility",
            "inner_visibility",
            "normalized_innovation_squared",
            "innovation_gate_threshold",
            "innovation_accepted",
            "dropout_count",
            "health",
            "health_reason",
        }
        row = _exact_object(value, keys, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported relative gate state schema")
        position = row["metric_position_body_frd_m"]
        velocity = row["metric_velocity_body_frd_m_s"]
        orientation = row["metric_gate_orientation_body_frd_xyzw"]
        metric_covariance = row["metric_covariance"]
        return cls(
            timing=PredictionTimeV1.from_primitive(row["timing"]),
            authority=GateAuthorityEpochV1.from_primitive(row["authority"]),
            tracker_id=row["tracker_id"],
            state_sequence=row["state_sequence"],
            measurement_update_sequence=row["measurement_update_sequence"],
            source_candidate_id=row["source_candidate_id"],
            track_role=_enum_value(row["track_role"], TrackRole, "track_role"),
            bearing_norm=_primitive_tuple(row["bearing_norm"], 2, "bearing_norm"),
            bearing_rate_norm_s=_primitive_tuple(
                row["bearing_rate_norm_s"], 2, "bearing_rate_norm_s"
            ),
            log_scale=row["log_scale"],
            expansion_rate_s=row["expansion_rate_s"],
            covariance=FeatureCovarianceV1.from_primitive(row["covariance"]),
            metric_position_body_frd_m=(
                None
                if position is None
                else _primitive_tuple(position, 3, "metric_position_body_frd_m")
            ),
            metric_velocity_body_frd_m_s=(
                None
                if velocity is None
                else _primitive_tuple(velocity, 3, "metric_velocity_body_frd_m_s")
            ),
            metric_gate_orientation_body_frd_xyzw=(
                None
                if orientation is None
                else _primitive_tuple(
                    orientation, 4, "metric_gate_orientation_body_frd_xyzw"
                )
            ),
            metric_covariance=(
                None
                if metric_covariance is None
                else FeatureCovarianceV1.from_primitive(metric_covariance)
            ),
            last_clipping=_mask_value(row["last_clipping"], FrameEdge, "last_clipping"),
            outer_visibility=_mask_value(
                row["outer_visibility"], FrameEdge, "outer_visibility"
            ),
            inner_visibility=_mask_value(
                row["inner_visibility"], FrameEdge, "inner_visibility"
            ),
            normalized_innovation_squared=row["normalized_innovation_squared"],
            innovation_gate_threshold=row["innovation_gate_threshold"],
            innovation_accepted=row["innovation_accepted"],
            dropout_count=row["dropout_count"],
            health=_enum_value(row["health"], RelativeStateHealth, "health"),
            health_reason=row["health_reason"],
        )


def validate_relative_gate_state_source(
    state: RelativeGateStateV1,
    observation: GateObservationV1,
) -> None:
    """Bind a relative state to the exact observation measurement it cites."""

    if type(state) is not RelativeGateStateV1:
        raise TypeError("state must be RelativeGateStateV1")
    if type(observation) is not GateObservationV1:
        raise TypeError("observation must be GateObservationV1")
    if observation.health is ObservationHealth.UNUSABLE:
        raise ValueError("an unusable observation cannot source a relative state")
    expected = (
        ("authority", state.authority, observation.authority),
        ("host clock", state.timing.host_clock_id, observation.host_clock_id),
        ("source frame", state.timing.source_frame, observation.frame),
        (
            "publication sequence",
            state.timing.source_frame_publication_sequence,
            observation.frame_timing.publication_sequence,
        ),
        (
            "publication time",
            state.timing.source_frame_publish_monotonic_ns,
            observation.frame_timing.publish_monotonic_ns,
        ),
        ("candidate id", state.source_candidate_id, observation.candidate_id),
        (
            "measurement time",
            state.timing.measurement_time_monotonic_ns,
            observation.measurement_time_monotonic_ns,
        ),
        (
            "measurement basis",
            state.timing.measurement_time_basis,
            observation.measurement_time_basis,
        ),
        (
            "measurement model",
            state.timing.measurement_time_model_id,
            observation.measurement_time_model_id,
        ),
        (
            "measurement uncertainty",
            state.timing.measurement_uncertainty_ns,
            observation.measurement_uncertainty_ns,
        ),
        ("clipping", state.last_clipping, observation.clipping),
        (
            "outer visibility",
            state.outer_visibility,
            observation.outer_edges.visibility,
        ),
        (
            "inner visibility",
            state.inner_visibility,
            observation.inner_edges.visibility,
        ),
    )
    for label, state_value, observation_value in expected:
        if state_value != observation_value:
            raise ValueError(f"relative state {label} does not match source observation")


@dataclass(frozen=True, slots=True)
class SaturationDiagnosticsV1:
    body_rate_axes: tuple[bool, bool, bool]
    thrust: bool

    def __post_init__(self) -> None:
        if type(self.body_rate_axes) is not tuple or len(self.body_rate_axes) != 3:
            raise TypeError("body_rate_axes must be an exact three-tuple")
        for index, value in enumerate(self.body_rate_axes):
            _exact_bool(value, f"body_rate_axes[{index}]")
        _exact_bool(self.thrust, "thrust saturation")

    def to_primitive(self) -> dict[str, Any]:
        return {"body_rate_axes": list(self.body_rate_axes), "thrust": self.thrust}

    @classmethod
    def from_primitive(cls, value: Any) -> "SaturationDiagnosticsV1":
        row = _exact_object(value, {"body_rate_axes", "thrust"}, "saturation")
        axes = _primitive_tuple(row["body_rate_axes"], 3, "body_rate_axes")
        return cls(body_rate_axes=axes, thrust=row["thrust"])


@dataclass(frozen=True, slots=True)
class UncertaintyDiagnosticsV1:
    limited: bool
    reason: Optional[str]

    def __post_init__(self) -> None:
        _exact_bool(self.limited, "uncertainty limited")
        _optional_string(self.reason, "uncertainty reason", maximum_length=256)
        if self.limited != (self.reason is not None):
            raise ValueError("uncertainty reason must be present exactly when limited")

    def to_primitive(self) -> dict[str, Any]:
        return {"limited": self.limited, "reason": self.reason}

    @classmethod
    def from_primitive(cls, value: Any) -> "UncertaintyDiagnosticsV1":
        row = _exact_object(value, {"limited", "reason"}, "uncertainty")
        return cls(limited=row["limited"], reason=row["reason"])


@dataclass(frozen=True, slots=True)
class CommandProposalV1:
    """Pure requested control; the safety supervisor still owns authority.

    The deadline is the current control tick's host-monotonic send deadline,
    not permission to send. Missed ticks are discarded rather than replayed.
    Source decision time cannot postdate proposal creation. Source prediction
    time is the state's modeled horizon and may postdate the proposal. Only an
    exact-zero failsafe proposal may omit the complete source-state identity.
    """

    SCHEMA: ClassVar[str] = "aigp-vq2-command-proposal/1"

    proposal_id: int
    control_tick_id: int
    host_clock_id: str
    proposal_monotonic_ns: int
    control_tick_deadline_monotonic_ns: int
    source_state_decision_monotonic_ns: Optional[int]
    source_state_prediction_monotonic_ns: Optional[int]
    source_frame: Optional[FrameIdentityV1]
    source_frame_publication_sequence: Optional[int]
    source_frame_publish_monotonic_ns: Optional[int]
    source_tracker_id: Optional[str]
    source_track_role: Optional[TrackRole]
    source_state_sequence: Optional[int]
    source_measurement_update_sequence: Optional[int]
    source_candidate_id: Optional[str]
    authority: GateAuthorityEpochV1
    requested_body_rates_rad_s: tuple[float, float, float]
    requested_thrust: float
    phase: str
    reason: str
    saturation: SaturationDiagnosticsV1
    uncertainty: UncertaintyDiagnosticsV1

    def __post_init__(self) -> None:
        _exact_int(self.proposal_id, "proposal_id")
        _exact_int(self.control_tick_id, "control_tick_id")
        _bounded_string(self.host_clock_id, "host_clock_id", token=True)
        proposal_time = _exact_int(self.proposal_monotonic_ns, "proposal_monotonic_ns")
        tick_deadline = _exact_int(
            self.control_tick_deadline_monotonic_ns,
            "control_tick_deadline_monotonic_ns",
        )
        if tick_deadline < proposal_time:
            raise ValueError("control tick deadline cannot predate the proposal")
        if self.source_frame is not None and type(self.source_frame) is not FrameIdentityV1:
            raise TypeError("source_frame must be FrameIdentityV1 or None")
        for name, value in (
            (
                "source_state_decision_monotonic_ns",
                self.source_state_decision_monotonic_ns,
            ),
            (
                "source_state_prediction_monotonic_ns",
                self.source_state_prediction_monotonic_ns,
            ),
            (
                "source_frame_publication_sequence",
                self.source_frame_publication_sequence,
            ),
            (
                "source_frame_publish_monotonic_ns",
                self.source_frame_publish_monotonic_ns,
            ),
            ("source_state_sequence", self.source_state_sequence),
            (
                "source_measurement_update_sequence",
                self.source_measurement_update_sequence,
            ),
        ):
            if value is not None:
                _exact_int(value, name)
        _optional_string(self.source_tracker_id, "source_tracker_id", token=True)
        if self.source_track_role is not None and type(self.source_track_role) is not TrackRole:
            raise TypeError("source_track_role must be TrackRole or None")
        _optional_string(self.source_candidate_id, "source_candidate_id", token=True)
        source_items = (
            self.source_state_decision_monotonic_ns,
            self.source_state_prediction_monotonic_ns,
            self.source_frame,
            self.source_frame_publication_sequence,
            self.source_frame_publish_monotonic_ns,
            self.source_tracker_id,
            self.source_track_role,
            self.source_state_sequence,
            self.source_measurement_update_sequence,
            self.source_candidate_id,
        )
        if any(item is None for item in source_items) and not all(
            item is None for item in source_items
        ):
            raise ValueError("command proposal source-state identity is all-or-none")
        if type(self.authority) is not GateAuthorityEpochV1:
            raise TypeError("authority must be GateAuthorityEpochV1")
        if self.source_frame is not None:
            if (
                self.source_frame_publication_sequence is None
                or self.source_frame_publish_monotonic_ns is None
            ):
                raise ValueError("command source frame publication is incomplete")
            _validate_authority_frame_cutover(
                self.authority,
                host_clock_id=self.host_clock_id,
                frame=self.source_frame,
                publication_sequence=self.source_frame_publication_sequence,
                publish_monotonic_ns=self.source_frame_publish_monotonic_ns,
                label="command source frame",
            )
            if (
                self.source_state_decision_monotonic_ns is None
                or self.source_state_prediction_monotonic_ns is None
            ):
                raise ValueError("command source state timing is incomplete")
            if self.source_state_decision_monotonic_ns > proposal_time:
                raise ValueError("source state decision cannot postdate the proposal")
            if (
                self.source_state_decision_monotonic_ns
                > self.source_state_prediction_monotonic_ns
            ):
                raise ValueError(
                    "source state decision cannot postdate its prediction horizon"
                )
            if (
                self.source_frame_publish_monotonic_ns
                > self.source_state_decision_monotonic_ns
            ):
                raise ValueError(
                    "command source frame cannot postdate the source state decision"
                )
        rates = _tuple_of_floats(
            self.requested_body_rates_rad_s, 3, "requested_body_rates_rad_s"
        )
        thrust = _finite_float(
            self.requested_thrust,
            "requested_thrust",
            minimum=0.0,
            maximum=1.0,
        )
        has_source = self.source_frame is not None
        if not has_source and not (
            rates == (0.0, 0.0, 0.0) and thrust == 0.0
        ):
            raise ValueError(
                "a nonzero command proposal requires a complete source state"
            )
        _bounded_string(self.phase, "phase", token=True)
        _bounded_string(self.reason, "reason", maximum_length=256)
        if type(self.saturation) is not SaturationDiagnosticsV1:
            raise TypeError("saturation must be SaturationDiagnosticsV1")
        if type(self.uncertainty) is not UncertaintyDiagnosticsV1:
            raise TypeError("uncertainty must be UncertaintyDiagnosticsV1")
        object.__setattr__(self, "requested_body_rates_rad_s", rates)
        object.__setattr__(self, "requested_thrust", thrust)

    @property
    def is_exact_zero(self) -> bool:
        return self.requested_body_rates_rad_s == (0.0, 0.0, 0.0) and self.requested_thrust == 0.0

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "proposal_id": self.proposal_id,
            "control_tick_id": self.control_tick_id,
            "host_clock_id": self.host_clock_id,
            "proposal_monotonic_ns": self.proposal_monotonic_ns,
            "control_tick_deadline_monotonic_ns": self.control_tick_deadline_monotonic_ns,
            "source_state_decision_monotonic_ns": self.source_state_decision_monotonic_ns,
            "source_state_prediction_monotonic_ns": self.source_state_prediction_monotonic_ns,
            "source_frame": (
                None if self.source_frame is None else self.source_frame.to_primitive()
            ),
            "source_frame_publication_sequence": self.source_frame_publication_sequence,
            "source_frame_publish_monotonic_ns": self.source_frame_publish_monotonic_ns,
            "source_tracker_id": self.source_tracker_id,
            "source_track_role": (
                None if self.source_track_role is None else self.source_track_role.value
            ),
            "source_state_sequence": self.source_state_sequence,
            "source_measurement_update_sequence": self.source_measurement_update_sequence,
            "source_candidate_id": self.source_candidate_id,
            "authority": self.authority.to_primitive(),
            "requested_body_rates_rad_s": list(self.requested_body_rates_rad_s),
            "requested_thrust": self.requested_thrust,
            "phase": self.phase,
            "reason": self.reason,
            "saturation": self.saturation.to_primitive(),
            "uncertainty": self.uncertainty.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "CommandProposalV1":
        keys = {
            "schema",
            "proposal_id",
            "control_tick_id",
            "host_clock_id",
            "proposal_monotonic_ns",
            "control_tick_deadline_monotonic_ns",
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
            "authority",
            "requested_body_rates_rad_s",
            "requested_thrust",
            "phase",
            "reason",
            "saturation",
            "uncertainty",
        }
        row = _exact_object(value, keys, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported command proposal schema")
        source = row["source_frame"]
        return cls(
            proposal_id=row["proposal_id"],
            control_tick_id=row["control_tick_id"],
            host_clock_id=row["host_clock_id"],
            proposal_monotonic_ns=row["proposal_monotonic_ns"],
            control_tick_deadline_monotonic_ns=row[
                "control_tick_deadline_monotonic_ns"
            ],
            source_state_decision_monotonic_ns=row[
                "source_state_decision_monotonic_ns"
            ],
            source_state_prediction_monotonic_ns=row[
                "source_state_prediction_monotonic_ns"
            ],
            source_frame=(
                None if source is None else FrameIdentityV1.from_primitive(source)
            ),
            source_frame_publication_sequence=row[
                "source_frame_publication_sequence"
            ],
            source_frame_publish_monotonic_ns=row[
                "source_frame_publish_monotonic_ns"
            ],
            source_tracker_id=row["source_tracker_id"],
            source_track_role=(
                None
                if row["source_track_role"] is None
                else _enum_value(
                    row["source_track_role"], TrackRole, "source_track_role"
                )
            ),
            source_state_sequence=row["source_state_sequence"],
            source_measurement_update_sequence=row[
                "source_measurement_update_sequence"
            ],
            source_candidate_id=row["source_candidate_id"],
            authority=GateAuthorityEpochV1.from_primitive(row["authority"]),
            requested_body_rates_rad_s=_primitive_tuple(
                row["requested_body_rates_rad_s"],
                3,
                "requested_body_rates_rad_s",
            ),
            requested_thrust=row["requested_thrust"],
            phase=row["phase"],
            reason=row["reason"],
            saturation=SaturationDiagnosticsV1.from_primitive(row["saturation"]),
            uncertainty=UncertaintyDiagnosticsV1.from_primitive(row["uncertainty"]),
        )


def validate_command_proposal_source(
    proposal: CommandProposalV1,
    state: RelativeGateStateV1,
) -> None:
    """Bind a sourced proposal to the exact relative-state identity it cites."""

    if type(proposal) is not CommandProposalV1:
        raise TypeError("proposal must be CommandProposalV1")
    if type(state) is not RelativeGateStateV1:
        raise TypeError("state must be RelativeGateStateV1")
    if proposal.source_frame is None:
        raise ValueError("a source-less exact-zero proposal has no relative state")
    expected = (
        ("host clock", proposal.host_clock_id, state.timing.host_clock_id),
        ("authority", proposal.authority, state.authority),
        (
            "state decision time",
            proposal.source_state_decision_monotonic_ns,
            state.timing.decision_time_monotonic_ns,
        ),
        (
            "state prediction time",
            proposal.source_state_prediction_monotonic_ns,
            state.timing.prediction_time_monotonic_ns,
        ),
        ("source frame", proposal.source_frame, state.timing.source_frame),
        (
            "source frame publication sequence",
            proposal.source_frame_publication_sequence,
            state.timing.source_frame_publication_sequence,
        ),
        (
            "source frame publication time",
            proposal.source_frame_publish_monotonic_ns,
            state.timing.source_frame_publish_monotonic_ns,
        ),
        ("tracker id", proposal.source_tracker_id, state.tracker_id),
        ("track role", proposal.source_track_role, state.track_role),
        ("state sequence", proposal.source_state_sequence, state.state_sequence),
        (
            "measurement update sequence",
            proposal.source_measurement_update_sequence,
            state.measurement_update_sequence,
        ),
        ("candidate id", proposal.source_candidate_id, state.source_candidate_id),
    )
    for label, observed, source_value in expected:
        if observed != source_value:
            raise ValueError(f"command proposal {label} does not match source state")


@dataclass(frozen=True, slots=True)
class SupervisorApprovedCommandV1:
    """A short-lived transport value issued only by the safety supervisor.

    The nested proposal preserves controller intent and gate/reset provenance.
    The approved values may differ only when ``safety_limit_reason`` explains
    the supervisor's intervention.  Possessing a proposal alone is never
    sufficient to create the legacy transport DTO. Approved roll/pitch rates
    also obey the frozen +/-0.25 rad/s build-3385 envelope and yaw is exact
    zero; stage-specific thrust and tighter limits remain safety-policy owned.
    """

    SCHEMA: ClassVar[str] = "aigp-vq2-supervisor-approved-command/1"

    command_id: int
    approval_monotonic_ns: int
    valid_until_monotonic_ns: int
    proposal: CommandProposalV1
    approved_body_rates_rad_s: tuple[float, float, float]
    approved_thrust: float
    safety_policy_id: str
    safety_limit_reason: Optional[str]

    def __post_init__(self) -> None:
        _exact_int(self.command_id, "command_id")
        approval_time = _exact_int(
            self.approval_monotonic_ns, "approval_monotonic_ns"
        )
        valid_until = _exact_int(
            self.valid_until_monotonic_ns, "valid_until_monotonic_ns"
        )
        if type(self.proposal) is not CommandProposalV1:
            raise TypeError("proposal must be CommandProposalV1")
        if approval_time < self.proposal.proposal_monotonic_ns:
            raise ValueError("supervisor approval cannot predate its proposal")
        if valid_until < approval_time:
            raise ValueError("command validity cannot end before approval")
        if valid_until > self.proposal.control_tick_deadline_monotonic_ns:
            raise ValueError("command validity cannot exceed its control tick deadline")
        rates = _tuple_of_floats(
            self.approved_body_rates_rad_s, 3, "approved_body_rates_rad_s"
        )
        if any(
            abs(value) > _VQ2_MAX_ROLL_PITCH_COMMAND_RATE_RAD_S
            for value in rates[:2]
        ):
            raise ValueError(
                "approved roll/pitch rate exceeds the build-3385 VQ2 safety envelope"
            )
        if rates[2] != 0.0:
            raise ValueError("approved yaw rate must be exact zero in build-3385 VQ2")
        thrust = _finite_float(
            self.approved_thrust,
            "approved_thrust",
            minimum=0.0,
            maximum=1.0,
        )
        _bounded_string(self.safety_policy_id, "safety_policy_id", token=True)
        _optional_string(
            self.safety_limit_reason,
            "safety_limit_reason",
            maximum_length=256,
            token=True,
        )
        values_changed = (
            rates != self.proposal.requested_body_rates_rad_s
            or thrust != self.proposal.requested_thrust
        )
        if (
            rates != (0.0, 0.0, 0.0) or thrust != 0.0
        ) and self.proposal.source_track_role is not TrackRole.ACTIVE:
            raise ValueError(
                "a nonzero supervisor approval requires an active source track"
            )
        for requested, allowed in zip(
            self.proposal.requested_body_rates_rad_s, rates
        ):
            if (
                abs(allowed) > abs(requested)
                or (requested == 0.0 and allowed != 0.0)
                or (requested > 0.0 and allowed < 0.0)
                or (requested < 0.0 and allowed > 0.0)
            ):
                raise ValueError(
                    "supervisor-approved body rates may only limit proposal magnitude"
                )
        if thrust > self.proposal.requested_thrust:
            raise ValueError(
                "supervisor-approved thrust may not amplify the proposal"
            )
        if values_changed != (self.safety_limit_reason is not None):
            raise ValueError(
                "safety_limit_reason must be present exactly when approved values "
                "differ from the proposal"
            )
        object.__setattr__(self, "approved_body_rates_rad_s", rates)
        object.__setattr__(self, "approved_thrust", thrust)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "command_id": self.command_id,
            "approval_monotonic_ns": self.approval_monotonic_ns,
            "valid_until_monotonic_ns": self.valid_until_monotonic_ns,
            "proposal": self.proposal.to_primitive(),
            "approved_body_rates_rad_s": list(self.approved_body_rates_rad_s),
            "approved_thrust": self.approved_thrust,
            "safety_policy_id": self.safety_policy_id,
            "safety_limit_reason": self.safety_limit_reason,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "SupervisorApprovedCommandV1":
        keys = {
            "schema",
            "command_id",
            "approval_monotonic_ns",
            "valid_until_monotonic_ns",
            "proposal",
            "approved_body_rates_rad_s",
            "approved_thrust",
            "safety_policy_id",
            "safety_limit_reason",
        }
        row = _exact_object(value, keys, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported supervisor-approved command schema")
        return cls(
            command_id=row["command_id"],
            approval_monotonic_ns=row["approval_monotonic_ns"],
            valid_until_monotonic_ns=row["valid_until_monotonic_ns"],
            proposal=CommandProposalV1.from_primitive(row["proposal"]),
            approved_body_rates_rad_s=_primitive_tuple(
                row["approved_body_rates_rad_s"],
                3,
                "approved_body_rates_rad_s",
            ),
            approved_thrust=row["approved_thrust"],
            safety_policy_id=row["safety_policy_id"],
            safety_limit_reason=row["safety_limit_reason"],
        )


def frame_identity_from_snapshot(
    snapshot: Any,
    *,
    stream_id: str = "camera0",
) -> FrameIdentityV1:
    """Compatibility projection without treating camera time as identity."""

    return FrameIdentityV1(
        stream_id=stream_id,
        generation=getattr(snapshot, "generation"),
        frame_id=getattr(snapshot, "frame_id"),
    )


def legacy_attitude_rate_to_proposal(
    command: Any,
    *,
    proposal_id: int,
    control_tick_id: int,
    host_clock_id: str,
    proposal_monotonic_ns: int,
    control_tick_deadline_monotonic_ns: int,
    source_state_decision_monotonic_ns: Optional[int],
    source_state_prediction_monotonic_ns: Optional[int],
    source_frame: Optional[FrameIdentityV1],
    source_frame_publication_sequence: Optional[int],
    source_frame_publish_monotonic_ns: Optional[int],
    source_tracker_id: Optional[str],
    source_track_role: Optional[TrackRole],
    source_state_sequence: Optional[int],
    source_measurement_update_sequence: Optional[int],
    source_candidate_id: Optional[str],
    authority: GateAuthorityEpochV1,
    phase: str,
    saturation: SaturationDiagnosticsV1,
    uncertainty: UncertaintyDiagnosticsV1,
    reason: str = "legacy_gate0",
) -> CommandProposalV1:
    """Wrap the exact legacy DTO without inventing controller diagnostics."""

    from competition.adapter import AttitudeRateCommand

    if type(command) is not AttitudeRateCommand:
        raise TypeError("command must be the exact AttitudeRateCommand DTO")
    if type(saturation) is not SaturationDiagnosticsV1:
        raise TypeError("saturation must be SaturationDiagnosticsV1")
    if type(uncertainty) is not UncertaintyDiagnosticsV1:
        raise TypeError("uncertainty must be UncertaintyDiagnosticsV1")

    return CommandProposalV1(
        proposal_id=proposal_id,
        control_tick_id=control_tick_id,
        host_clock_id=host_clock_id,
        proposal_monotonic_ns=proposal_monotonic_ns,
        control_tick_deadline_monotonic_ns=control_tick_deadline_monotonic_ns,
        source_state_decision_monotonic_ns=source_state_decision_monotonic_ns,
        source_state_prediction_monotonic_ns=source_state_prediction_monotonic_ns,
        source_frame=source_frame,
        source_frame_publication_sequence=source_frame_publication_sequence,
        source_frame_publish_monotonic_ns=source_frame_publish_monotonic_ns,
        source_tracker_id=source_tracker_id,
        source_track_role=source_track_role,
        source_state_sequence=source_state_sequence,
        source_measurement_update_sequence=source_measurement_update_sequence,
        source_candidate_id=source_candidate_id,
        authority=authority,
        requested_body_rates_rad_s=(
            command.roll_rate,
            command.pitch_rate,
            command.yaw_rate,
        ),
        requested_thrust=command.thrust,
        phase=phase,
        reason=reason,
        saturation=saturation,
        uncertainty=uncertainty,
    )


def approved_command_to_attitude_rate_command(
    approved: SupervisorApprovedCommandV1,
    *,
    host_clock_id: str,
    send_monotonic_ns: int,
    expected_control_tick_id: int,
    expected_control_tick_deadline_monotonic_ns: int,
    expected_authority: GateAuthorityEpochV1,
    expected_safety_policy_id: str,
    maximum_approval_age_ns: int,
) -> Any:
    """Project a fresh, epoch-matched approval to the legacy transport DTO.

    This function performs no send.  Requiring the current host clock, safety
    authority, and send time prevents a stale approval from being reused at the
    compatibility boundary.
    """

    if type(approved) is not SupervisorApprovedCommandV1:
        raise TypeError("approved must be SupervisorApprovedCommandV1")
    _bounded_string(host_clock_id, "host_clock_id", token=True)
    send_time = _exact_int(send_monotonic_ns, "send_monotonic_ns")
    expected_tick = _exact_int(expected_control_tick_id, "expected_control_tick_id")
    expected_deadline = _exact_int(
        expected_control_tick_deadline_monotonic_ns,
        "expected_control_tick_deadline_monotonic_ns",
    )
    if type(expected_authority) is not GateAuthorityEpochV1:
        raise TypeError("expected_authority must be GateAuthorityEpochV1")
    _bounded_string(
        expected_safety_policy_id, "expected_safety_policy_id", token=True
    )
    maximum_age = _exact_int(
        maximum_approval_age_ns, "maximum_approval_age_ns", minimum=1
    )
    proposal = approved.proposal
    if host_clock_id != proposal.host_clock_id:
        raise ValueError("approval host clock does not match the transport clock")
    if expected_tick != proposal.control_tick_id:
        raise ValueError("approval does not match the current control tick")
    if expected_deadline != proposal.control_tick_deadline_monotonic_ns:
        raise ValueError("approval does not match the trusted control tick deadline")
    if expected_authority != proposal.authority:
        raise ValueError("approval authority does not match the current safety epoch")
    if expected_safety_policy_id != approved.safety_policy_id:
        raise ValueError("approval safety policy does not match the trusted policy")
    if send_time < approved.approval_monotonic_ns:
        raise ValueError("command send cannot predate supervisor approval")
    if send_time > approved.valid_until_monotonic_ns:
        raise ValueError("supervisor approval has expired")
    if send_time - approved.approval_monotonic_ns > maximum_age:
        raise ValueError("supervisor approval exceeds the trusted maximum age")
    from competition.adapter import AttitudeRateCommand

    rates = approved.approved_body_rates_rad_s
    return AttitudeRateCommand(
        roll_rate=rates[0],
        pitch_rate=rates[1],
        yaw_rate=rates[2],
        thrust=approved.approved_thrust,
    )


def proposal_to_replay_command_v1(proposal: CommandProposalV1) -> dict[str, float]:
    """Project intent to the historical open-loop replay `/1` shape; never send."""

    if type(proposal) is not CommandProposalV1:
        raise TypeError("proposal must be CommandProposalV1")
    rates = proposal.requested_body_rates_rad_s
    return {
        "roll_rate": rates[0],
        "pitch_rate": rates[1],
        "yaw_rate": rates[2],
        "thrust": proposal.requested_thrust,
    }


def validate_frame_timing_sequence(timings: Sequence[FrameTimingV1]) -> None:
    """Validate publication order per host/stream without crossing clocks."""

    if type(timings) is not tuple:
        raise TypeError("frame timing sequence must be an exact tuple")
    last_publication: dict[tuple[str, str], int] = {}
    last_source_time: dict[tuple[str, str, int], int] = {}
    last_generation: dict[tuple[str, str], int] = {}
    last_publish_time: dict[tuple[str, str], int] = {}
    seen_frames: set[tuple[str, FrameIdentityV1]] = set()
    for timing in timings:
        if type(timing) is not FrameTimingV1:
            raise TypeError("frame timing sequence contains a non-contract value")
        frame_key = (timing.host_clock_id, timing.identity)
        if frame_key in seen_frames:
            raise ValueError("frame timing sequence repeats a frame identity")
        seen_frames.add(frame_key)
        stream_key = (timing.host_clock_id, timing.identity.stream_id)
        previous_generation = last_generation.get(stream_key)
        if (
            previous_generation is not None
            and timing.identity.generation < previous_generation
        ):
            raise ValueError("frame generation cannot regress within a stream")
        last_generation[stream_key] = timing.identity.generation
        previous_publication = last_publication.get(stream_key)
        if (
            previous_publication is not None
            and timing.publication_sequence <= previous_publication
        ):
            raise ValueError("publication_sequence must progress strictly per stream")
        last_publication[stream_key] = timing.publication_sequence
        previous_publish = last_publish_time.get(stream_key)
        if (
            previous_publish is not None
            and timing.publish_monotonic_ns <= previous_publish
        ):
            raise ValueError("frame publish time must progress strictly per stream")
        last_publish_time[stream_key] = timing.publish_monotonic_ns
        generation_key = (*stream_key, timing.identity.generation)
        previous_source = last_source_time.get(generation_key)
        if previous_source is not None and timing.camera_source_time_ns <= previous_source:
            raise ValueError("camera source time must progress strictly within a generation")
        last_source_time[generation_key] = timing.camera_source_time_ns


def validate_relative_gate_state_sequence(
    states: Sequence[RelativeGateStateV1],
) -> None:
    """Validate one-update-per-observation tracking and active/shadow ownership."""

    if type(states) is not tuple:
        raise TypeError("relative gate state sequence must be an exact tuple")
    last: dict[
        tuple[str, str, int, int, int, str],
        tuple[int, int, int, tuple[FrameIdentityV1, str]],
    ] = {}
    associations: dict[
        tuple[str, int, int, int, str, FrameIdentityV1, str],
        tuple[str, TrackRole],
    ] = {}
    last_authority: dict[tuple[str, str], GateAuthorityEpochV1] = {}
    source_updates: dict[
        tuple[
            tuple[str, str, int, int, int, str],
            tuple[FrameIdentityV1, str],
        ],
        int,
    ] = {}
    for state in states:
        if type(state) is not RelativeGateStateV1:
            raise TypeError("relative state sequence contains a non-contract value")
        key = (
            state.timing.host_clock_id,
            state.authority.session_id,
            state.authority.reset_epoch,
            state.authority.gate_epoch,
            state.authority.expected_gate_index,
            state.tracker_id,
        )
        authority_trace_key = (
            state.timing.host_clock_id,
            state.authority.session_id,
        )
        previous_authority = last_authority.get(authority_trace_key)
        if previous_authority is not None:
            _validate_authority_transition(
                previous_authority,
                state.authority,
                allow_snapshot_refresh=True,
            )
        last_authority[authority_trace_key] = state.authority
        if state.timing.source_frame is None:
            raise ValueError("relative state sequence contains no source frame")
        source = (state.timing.source_frame, state.source_candidate_id)
        source_update_key = (key, source)
        previous_source_update = source_updates.get(source_update_key)
        if (
            previous_source_update is not None
            and state.measurement_update_sequence != previous_source_update
        ):
            raise ValueError(
                "one source observation cannot be applied more than once"
            )
        previous = last.get(key)
        if previous is not None:
            previous_state, previous_update, previous_time, previous_source = previous
            if state.state_sequence <= previous_state:
                raise ValueError("state_sequence must progress strictly per tracker")
            if state.timing.prediction_time_monotonic_ns < previous_time:
                raise ValueError("relative state prediction time regressed")
            if source == previous_source:
                if state.measurement_update_sequence != previous_update:
                    raise ValueError(
                        "one source observation cannot be applied more than once"
                    )
            elif state.measurement_update_sequence <= previous_update:
                raise ValueError(
                    "a new source observation must advance measurement_update_sequence"
                )
        association_key = (
            state.authority.session_id,
            state.authority.reset_epoch,
            state.authority.gate_epoch,
            state.authority.expected_gate_index,
            state.timing.host_clock_id,
            source[0],
            source[1],
        )
        owner = (state.tracker_id, state.track_role)
        previous_owner = associations.get(association_key)
        if previous_owner is not None and previous_owner != owner:
            raise ValueError(
                "one source observation cannot seed multiple active/shadow tracks"
            )
        associations[association_key] = owner
        source_updates[source_update_key] = state.measurement_update_sequence
        last[key] = (
            state.state_sequence,
            state.measurement_update_sequence,
            state.timing.prediction_time_monotonic_ns,
            source,
        )


def validate_latency_event_sequence(events: Sequence[LatencyEventV1]) -> None:
    """Validate a trace's order, command correlation, and stage lifecycles.

    An unfinished start is allowed so an aborted/truncated trace remains
    representable.  An end without an earlier matching start is never valid.
    """

    if type(events) is not tuple:
        raise TypeError("latency event sequence must be an exact tuple")
    last: dict[str, tuple[int, int]] = {}
    command_ticks: dict[tuple[str, int], int] = {}
    due_ticks: dict[tuple[str, int], int] = {}
    last_due: dict[str, tuple[int, int]] = {}
    started_ticks: set[tuple[str, int]] = set()
    ended_ticks: set[tuple[str, int]] = set()
    skipped_ticks: set[tuple[str, int]] = set()
    deadline_missed_ticks: set[tuple[str, int]] = set()
    sent_ticks: set[tuple[str, int]] = set()
    sent_commands: set[tuple[str, int]] = set()
    last_send_start: dict[str, int] = {}
    open_stages: set[tuple[str, LatencyEventKind, tuple[Any, ...]]] = set()
    end_to_start = {
        LatencyEventKind.DECODE_END: LatencyEventKind.DECODE_START,
        LatencyEventKind.DETECTION_END: LatencyEventKind.DETECTION_START,
        LatencyEventKind.TRACKING_END: LatencyEventKind.TRACKING_START,
        LatencyEventKind.ESTIMATOR_UPDATE_END: LatencyEventKind.ESTIMATOR_UPDATE_START,
        LatencyEventKind.PREDICTION_END: LatencyEventKind.PREDICTION_START,
        LatencyEventKind.CONTROLLER_END: LatencyEventKind.CONTROLLER_START,
        LatencyEventKind.CONTROL_TICK_END: LatencyEventKind.CONTROL_TICK_START,
        LatencyEventKind.COMMAND_SEND_END: LatencyEventKind.COMMAND_SEND_START,
    }
    start_kinds = frozenset(end_to_start.values())

    def stage_correlation(event: LatencyEventV1) -> tuple[Any, ...]:
        if event.kind in {
            LatencyEventKind.DECODE_START,
            LatencyEventKind.DECODE_END,
            LatencyEventKind.DETECTION_START,
            LatencyEventKind.DETECTION_END,
            LatencyEventKind.TRACKING_START,
            LatencyEventKind.TRACKING_END,
            LatencyEventKind.ESTIMATOR_UPDATE_START,
            LatencyEventKind.ESTIMATOR_UPDATE_END,
        }:
            return (event.frame,)
        if event.kind in {
            LatencyEventKind.PREDICTION_START,
            LatencyEventKind.PREDICTION_END,
            LatencyEventKind.CONTROLLER_START,
            LatencyEventKind.CONTROLLER_END,
        }:
            return (event.frame, event.control_tick_id)
        if event.kind in {
            LatencyEventKind.CONTROL_TICK_START,
            LatencyEventKind.CONTROL_TICK_END,
        }:
            return (event.control_tick_id,)
        return (event.control_tick_id, event.command_id)

    for event in events:
        if type(event) is not LatencyEventV1:
            raise TypeError("latency event sequence contains a non-contract value")
        previous = last.get(event.host_clock_id)
        if previous is not None:
            previous_sequence, previous_time = previous
            if event.event_sequence <= previous_sequence:
                raise ValueError("event_sequence must progress strictly per host clock")
            if event.monotonic_ns < previous_time:
                raise ValueError("latency event occurrence time regressed")
        last[event.host_clock_id] = (event.event_sequence, event.monotonic_ns)
        tick_key = (
            None
            if event.control_tick_id is None
            else (event.host_clock_id, event.control_tick_id)
        )
        if event.kind in {
            LatencyEventKind.CONTROL_TICK_DUE,
            LatencyEventKind.CONTROL_TICK_START,
            LatencyEventKind.CONTROL_TICK_END,
            LatencyEventKind.CONTROL_TICK_SKIPPED,
            LatencyEventKind.DEADLINE_MISSED,
            LatencyEventKind.COMMAND_SEND_START,
        } and tick_key is None:
            raise ValueError(f"{event.kind.value} is missing its required control tick")
        if event.kind is LatencyEventKind.CONTROL_TICK_DUE:
            if tick_key is None:
                raise ValueError("control tick due event is missing its identity")
            if tick_key in due_ticks:
                raise ValueError("control tick can become due only once")
            previous_due = last_due.get(event.host_clock_id)
            if previous_due is not None:
                previous_tick, previous_due_time = previous_due
                if event.control_tick_id <= previous_tick:
                    raise ValueError("due control_tick_id must progress strictly")
                if event.monotonic_ns - previous_due_time < 20_000_000:
                    raise ValueError("control ticks cannot be due faster than 50 Hz")
                previous_tick_key = (event.host_clock_id, previous_tick)
                if (
                    previous_tick_key not in started_ticks
                    and previous_tick_key not in skipped_ticks
                ):
                    raise ValueError(
                        "an unstarted control tick must be skipped before a newer due tick"
                    )
            due_ticks[tick_key] = event.monotonic_ns
            last_due[event.host_clock_id] = (
                event.control_tick_id,
                event.monotonic_ns,
            )
        elif event.kind is LatencyEventKind.CONTROL_TICK_START:
            if tick_key is None:
                raise ValueError("control tick start is missing its identity")
            if tick_key not in due_ticks:
                raise ValueError("control tick start requires an earlier due event")
            if tick_key in skipped_ticks or tick_key in deadline_missed_ticks:
                raise ValueError("a missed/skipped control tick cannot start")
            if tick_key in started_ticks:
                raise ValueError("a control tick can start only once")
            latest_due_tick = last_due[event.host_clock_id][0]
            if event.control_tick_id != latest_due_tick:
                raise ValueError("a superseded control tick cannot start")
            started_ticks.add(tick_key)
        elif event.kind is LatencyEventKind.CONTROL_TICK_END:
            if tick_key is None:
                raise ValueError("control tick end is missing its identity")
            if tick_key in skipped_ticks:
                raise ValueError("a skipped control tick cannot end")
            ended_ticks.add(tick_key)
        elif event.kind is LatencyEventKind.CONTROL_TICK_SKIPPED:
            if tick_key is None:
                raise ValueError("skipped control tick is missing its identity")
            if tick_key not in due_ticks:
                raise ValueError("skipped control tick requires an earlier due event")
            if tick_key in started_ticks or tick_key in ended_ticks or tick_key in sent_ticks:
                raise ValueError("a started/sent control tick cannot also be skipped")
            if tick_key in skipped_ticks:
                raise ValueError("control tick can be skipped only once")
            skipped_ticks.add(tick_key)
        elif event.kind is LatencyEventKind.DEADLINE_MISSED:
            if tick_key is None:
                raise ValueError("deadline miss is missing its control tick")
            if tick_key not in due_ticks:
                raise ValueError("deadline miss requires an earlier due event")
            if tick_key in started_ticks:
                raise ValueError("a started control tick cannot be declared missed")
            if tick_key in skipped_ticks:
                raise ValueError("a deadline miss must precede its skipped-tick event")
            if tick_key in deadline_missed_ticks:
                raise ValueError("control tick deadline can be missed only once")
            deadline_missed_ticks.add(tick_key)
        elif event.kind is LatencyEventKind.COMMAND_SEND_START:
            if tick_key is None:
                raise ValueError("command send is missing its control tick")
            if tick_key not in started_ticks or tick_key in skipped_ticks:
                raise ValueError("command send requires a started, non-skipped tick")
            latest_due_tick = last_due[event.host_clock_id][0]
            if event.control_tick_id != latest_due_tick:
                raise ValueError("a superseded control tick cannot send a command")
            command_key = (event.host_clock_id, event.command_id)
            if tick_key in sent_ticks or command_key in sent_commands:
                raise ValueError("each control tick and command may be sent only once")
            previous_send = last_send_start.get(event.host_clock_id)
            if (
                previous_send is not None
                and event.monotonic_ns - previous_send < 20_000_000
            ):
                raise ValueError("command sends cannot occur faster than 50 Hz")
            sent_ticks.add(tick_key)
            sent_commands.add(command_key)
            last_send_start[event.host_clock_id] = event.monotonic_ns
        if event.command_id is not None and event.control_tick_id is not None:
            command_key = (event.host_clock_id, event.command_id)
            known_tick = command_ticks.get(command_key)
            if known_tick is not None and known_tick != event.control_tick_id:
                raise ValueError("command_id cannot correlate to multiple control ticks")
            command_ticks[command_key] = event.control_tick_id
        if event.kind in start_kinds:
            key = (event.host_clock_id, event.kind, stage_correlation(event))
            if key in open_stages:
                raise ValueError(f"duplicate open {event.kind.value} stage")
            open_stages.add(key)
        elif event.kind in end_to_start:
            start_kind = end_to_start[event.kind]
            key = (event.host_clock_id, start_kind, stage_correlation(event))
            if key not in open_stages:
                raise ValueError(
                    f"{event.kind.value} has no matching earlier {start_kind.value}"
                )
            open_stages.remove(key)
    if not deadline_missed_ticks <= skipped_ticks:
        raise ValueError("every missed deadline must end in a skipped control tick")


def _validate_authority_transition(
    previous: GateAuthorityEpochV1,
    current: GateAuthorityEpochV1,
    *,
    allow_snapshot_refresh: bool,
) -> None:
    if current == previous:
        return
    if current.session_id != previous.session_id:
        raise ValueError("approved authority cannot switch safety sessions")
    if (
        current.camera_host_clock_id != previous.camera_host_clock_id
        or current.camera_stream_id != previous.camera_stream_id
    ):
        raise ValueError("approved authority cannot switch camera identity")
    if current.reset_epoch < previous.reset_epoch:
        raise ValueError("approved authority reset epoch regressed")
    if current.reset_epoch == previous.reset_epoch:
        if current.gate_epoch < previous.gate_epoch:
            raise ValueError("approved authority gate epoch regressed")
        if current.gate_epoch == previous.gate_epoch:
            if current.expected_gate_index != previous.expected_gate_index:
                raise ValueError(
                    "expected gate index changed without a gate epoch transition"
                )
            if current.camera_generation != previous.camera_generation:
                raise ValueError("camera generation changed within a gate epoch")
            if not allow_snapshot_refresh:
                raise ValueError(
                    "approved authority changed without a gate/reset epoch transition"
                )
            if (
                current.race_status_sequence < previous.race_status_sequence
                or current.race_status_boot_ms < previous.race_status_boot_ms
                or current.frame_publication_sequence_not_before
                < previous.frame_publication_sequence_not_before
                or current.frame_publish_monotonic_ns_not_before
                < previous.frame_publish_monotonic_ns_not_before
            ):
                raise ValueError("gate authority snapshot regressed within its epoch")
            if (
                current.race_status_sequence == previous.race_status_sequence
                and current.race_status_boot_ms != previous.race_status_boot_ms
            ):
                raise ValueError(
                    "one race-status sequence cannot carry multiple boot times"
                )
            return
        if current.race_status_sequence <= previous.race_status_sequence:
            raise ValueError("gate authority race-status sequence did not advance")
        if current.race_status_boot_ms <= previous.race_status_boot_ms:
            raise ValueError("gate authority race-status boot time did not advance")
        if current.camera_generation != previous.camera_generation:
            raise ValueError("camera generation changed during a gate-only transition")
    else:
        if current.gate_epoch != 0 or current.expected_gate_index != 0:
            raise ValueError("a new reset authority must restart at gate epoch/index zero")
        if current.race_status_sequence <= previous.race_status_sequence:
            raise ValueError("reset authority race-status sequence did not advance")
        if current.camera_generation <= previous.camera_generation:
            raise ValueError("reset authority must advance the camera generation")
    if (
        current.frame_publication_sequence_not_before
        <= previous.frame_publication_sequence_not_before
        or current.frame_publish_monotonic_ns_not_before
        <= previous.frame_publish_monotonic_ns_not_before
    ):
        raise ValueError("forward authority transition must advance camera cutovers")


def validate_approved_command_sequence(
    approvals: Sequence[SupervisorApprovedCommandV1],
) -> None:
    """Reject duplicate/reordered approvals before stateful transport use."""

    if type(approvals) is not tuple:
        raise TypeError("approved command sequence must be an exact tuple")
    last: dict[str, tuple[int, int, GateAuthorityEpochV1]] = {}
    command_ids: set[tuple[str, int]] = set()
    proposal_ids: set[tuple[str, int]] = set()
    for approved in approvals:
        if type(approved) is not SupervisorApprovedCommandV1:
            raise TypeError("approved command sequence contains a non-contract value")
        proposal = approved.proposal
        clock = proposal.host_clock_id
        command_key = (clock, approved.command_id)
        proposal_key = (clock, proposal.proposal_id)
        if command_key in command_ids:
            raise ValueError("approved command sequence repeats a command_id")
        if proposal_key in proposal_ids:
            raise ValueError("approved command sequence reuses a proposal_id")
        previous = last.get(clock)
        if previous is not None:
            previous_tick, previous_approval, previous_authority = previous
            if proposal.control_tick_id <= previous_tick:
                raise ValueError("control_tick_id must progress strictly per host clock")
            if approved.approval_monotonic_ns < previous_approval:
                raise ValueError("supervisor approval time regressed")
            _validate_authority_transition(
                previous_authority,
                proposal.authority,
                allow_snapshot_refresh=False,
            )
        command_ids.add(command_key)
        proposal_ids.add(proposal_key)
        last[clock] = (
            proposal.control_tick_id,
            approved.approval_monotonic_ns,
            proposal.authority,
        )


def validate_command_latency_correlation(
    approved: SupervisorApprovedCommandV1,
    events: Sequence[LatencyEventV1],
) -> None:
    """Bind one approval to exactly one matching command-send start/end pair."""

    if type(approved) is not SupervisorApprovedCommandV1:
        raise TypeError("approved must be SupervisorApprovedCommandV1")
    validate_latency_event_sequence(events)
    proposal = approved.proposal
    matching = tuple(
        event
        for event in events
        if event.host_clock_id == proposal.host_clock_id
        and event.command_id == approved.command_id
        and event.kind
        in {LatencyEventKind.COMMAND_SEND_START, LatencyEventKind.COMMAND_SEND_END}
    )
    starts = tuple(
        event
        for event in matching
        if event.kind is LatencyEventKind.COMMAND_SEND_START
    )
    ends = tuple(
        event for event in matching if event.kind is LatencyEventKind.COMMAND_SEND_END
    )
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            "approved command requires exactly one matching send start and end"
        )
    due_events = tuple(
        event
        for event in events
        if event.host_clock_id == proposal.host_clock_id
        and event.control_tick_id == proposal.control_tick_id
        and event.kind is LatencyEventKind.CONTROL_TICK_DUE
    )
    tick_starts = tuple(
        event
        for event in events
        if event.host_clock_id == proposal.host_clock_id
        and event.control_tick_id == proposal.control_tick_id
        and event.kind is LatencyEventKind.CONTROL_TICK_START
    )
    if len(due_events) != 1 or len(tick_starts) != 1:
        raise ValueError(
            "approved command requires exactly one matching control tick due and start"
        )
    if (
        due_events[0].outcome is not EventOutcome.OK
        or tick_starts[0].outcome is not EventOutcome.OK
    ):
        raise ValueError(
            "failed control-tick events cannot satisfy approved correlation"
        )
    if due_events[0].monotonic_ns > proposal.proposal_monotonic_ns:
        raise ValueError("control tick became due after its proposal was created")
    if tick_starts[0].monotonic_ns > proposal.proposal_monotonic_ns:
        raise ValueError("control tick started after its proposal was created")
    for event in matching:
        if event.control_tick_id != proposal.control_tick_id:
            raise ValueError("command latency control tick does not match the proposal")
        if proposal.source_frame is not None and event.frame != proposal.source_frame:
            raise ValueError("command latency source frame does not match the proposal")
        if event.outcome is not EventOutcome.OK:
            raise ValueError(
                "failed command-send events cannot satisfy approved correlation"
            )
    if starts[0].monotonic_ns < approved.approval_monotonic_ns:
        raise ValueError("command send started before supervisor approval")
    if starts[0].monotonic_ns > approved.valid_until_monotonic_ns:
        raise ValueError("command send started after supervisor approval expired")
    if ends[0].monotonic_ns > approved.valid_until_monotonic_ns:
        raise ValueError("command send ended after supervisor approval expired")


__all__ = [
    "CommandProposalV1",
    "EdgeSetV1",
    "EventOutcome",
    "FeatureCovarianceV1",
    "FitDiagnosticsV1",
    "FrameEdge",
    "FrameIdentityV1",
    "FrameTimingV1",
    "GateAuthorityEpochV1",
    "GateObservationV1",
    "LatencyEventKind",
    "LatencyEventV1",
    "LineSegmentV1",
    "MeasurementTimeBasis",
    "ObservationHealth",
    "PredictionBasis",
    "PredictionTimeV1",
    "RelativeGateStateV1",
    "RelativeStateHealth",
    "SaturationDiagnosticsV1",
    "SupervisorApprovedCommandV1",
    "TrackRole",
    "UncertaintyDiagnosticsV1",
    "frame_identity_from_snapshot",
    "legacy_attitude_rate_to_proposal",
    "approved_command_to_attitude_rate_command",
    "proposal_to_replay_command_v1",
    "validate_frame_timing_sequence",
    "validate_gate_observation_batch",
    "validate_latency_event_sequence",
    "validate_relative_gate_state_source",
    "validate_relative_gate_state_sequence",
    "validate_approved_command_sequence",
    "validate_command_latency_correlation",
    "validate_command_proposal_source",
]
