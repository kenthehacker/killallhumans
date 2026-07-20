"""Pure, authority-neutral capture contracts for build-3385 VQ2 ingress.

The values in this module preserve receiver-boundary facts only. They do not
open sockets, convert clocks, infer capture time, reset or arm the simulator,
or send any MAVLink message. Source timestamps remain opaque values paired
with their exact declared unit.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from competition.adapter import IMUData


HOST_PERF_COUNTER_CLOCK_ID = "host-perf-counter"

HIGHRES_IMU_MESSAGE_TYPE = "HIGHRES_IMU"
HEARTBEAT_MESSAGE_TYPE = "HEARTBEAT"
RACE_STATUS_MESSAGE_TYPE = "RACE_STATUS"
ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE = "ACTUATOR_OUTPUT_STATUS"

SUPPORTED_MAVLINK_MESSAGE_TYPES = frozenset(
    {
        HIGHRES_IMU_MESSAGE_TYPE,
        HEARTBEAT_MESSAGE_TYPE,
        RACE_STATUS_MESSAGE_TYPE,
        ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE,
    }
)

_EXPECTED_SOURCE_TIME_UNIT = {
    HIGHRES_IMU_MESSAGE_TYPE: "us",
    HEARTBEAT_MESSAGE_TYPE: None,
    RACE_STATUS_MESSAGE_TYPE: "ms",
    ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE: "us",
}
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_UINT64_MAX = (1 << 64) - 1


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


def _exact_nonnegative_int(
    value: Any,
    label: str,
    *,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < 0 or (maximum is not None and value > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{label} must be >= 0{suffix}")
    return value


def _bounded_token(value: Any, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if _TOKEN_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded ASCII token")
    return value


def _finite_float(value: Any, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _direct_vector3(value: Any, label: str) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise TypeError(f"{label} must be an exact 3-tuple")
    return tuple(
        _finite_float(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _primitive_vector3(value: Any, label: str) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise TypeError(f"{label} must be an exact 3-element array")
    return tuple(
        _finite_float(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _copy_imu_data(value: Any) -> IMUData:
    if type(value) is not IMUData:
        raise TypeError("imu must be exact IMUData")
    timestamp_us = _exact_nonnegative_int(
        value.timestamp_us,
        "imu.timestamp_us",
        maximum=_UINT64_MAX,
    )
    accel = _direct_vector3(value.accel, "imu.accel")
    gyro = _direct_vector3(value.gyro, "imu.gyro")
    if value.mag is None:
        mag = None
    else:
        mag = _direct_vector3(value.mag, "imu.mag")
    return IMUData(
        timestamp_us=timestamp_us,
        accel=accel,
        gyro=gyro,
        mag=mag,
    )


@dataclass(frozen=True, slots=True)
class MavlinkIngressV1:
    """One exact MAVLink receive-boundary observation on the host QPC clock."""

    SCHEMA: ClassVar[str] = "aigp-vq2-mavlink-ingress/1"

    stream_id: str
    generation: int
    sequence: int
    message_type: str
    host_clock_id: str
    received_monotonic_ns: int
    source_time_value: int | None
    source_time_unit: str | None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _bounded_token(self.stream_id, "stream_id")
        _exact_nonnegative_int(self.generation, "generation")
        _exact_nonnegative_int(self.sequence, "sequence")
        if type(self.message_type) is not str:
            raise TypeError("message_type must be an exact string")
        if self.message_type not in SUPPORTED_MAVLINK_MESSAGE_TYPES:
            raise ValueError("unsupported MAVLink ingress message_type")
        _bounded_token(self.host_clock_id, "host_clock_id")
        if self.host_clock_id != HOST_PERF_COUNTER_CLOCK_ID:
            raise ValueError("MAVLink ingress must use host-perf-counter")
        _exact_nonnegative_int(
            self.received_monotonic_ns,
            "received_monotonic_ns",
            maximum=_UINT64_MAX,
        )
        value_present = self.source_time_value is not None
        unit_present = self.source_time_unit is not None
        if value_present != unit_present:
            raise ValueError(
                "source_time_value and source_time_unit must be both present or both absent"
            )
        expected_unit = _EXPECTED_SOURCE_TIME_UNIT[self.message_type]
        if not value_present:
            return
        assert self.source_time_value is not None
        assert self.source_time_unit is not None
        _exact_nonnegative_int(
            self.source_time_value,
            "source_time_value",
            maximum=_UINT64_MAX,
        )
        if type(self.source_time_unit) is not str:
            raise TypeError("source_time_unit must be an exact string")
        if expected_unit is None:
            raise ValueError(f"{self.message_type} has no admitted source timestamp")
        if self.source_time_unit != expected_unit:
            raise ValueError(
                f"{self.message_type} source_time_unit must be {expected_unit!r}"
            )

    def validate_integrity(self) -> None:
        """Revalidate fields after any hostile low-level mutation."""

        self._validate()

    def to_primitive(self) -> dict[str, Any]:
        self.validate_integrity()
        return {
            "schema": self.SCHEMA,
            "stream_id": self.stream_id,
            "generation": self.generation,
            "sequence": self.sequence,
            "message_type": self.message_type,
            "host_clock_id": self.host_clock_id,
            "received_monotonic_ns": self.received_monotonic_ns,
            "source_time_value": self.source_time_value,
            "source_time_unit": self.source_time_unit,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "MavlinkIngressV1":
        keys = {
            "schema",
            "stream_id",
            "generation",
            "sequence",
            "message_type",
            "host_clock_id",
            "received_monotonic_ns",
            "source_time_value",
            "source_time_unit",
        }
        row = _exact_object(value, keys, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported MAVLink ingress schema")
        return cls(
            stream_id=row["stream_id"],
            generation=row["generation"],
            sequence=row["sequence"],
            message_type=row["message_type"],
            host_clock_id=row["host_clock_id"],
            received_monotonic_ns=row["received_monotonic_ns"],
            source_time_value=row["source_time_value"],
            source_time_unit=row["source_time_unit"],
        )


@dataclass(frozen=True, slots=True)
class ReceivedIMUSampleV1:
    """One HIGHRES_IMU ingress paired with a defensive raw-sample copy."""

    SCHEMA: ClassVar[str] = "aigp-vq2-received-imu/1"

    ingress: MavlinkIngressV1
    imu: IMUData

    def __post_init__(self) -> None:
        if type(self.ingress) is not MavlinkIngressV1:
            raise TypeError("ingress must be exact MavlinkIngressV1")
        self.ingress.validate_integrity()
        copied = _copy_imu_data(self.imu)
        self._validate_binding(copied)
        object.__setattr__(self, "imu", copied)

    def _validate_binding(self, imu: IMUData) -> None:
        if self.ingress.message_type != HIGHRES_IMU_MESSAGE_TYPE:
            raise ValueError("received IMU ingress must be HIGHRES_IMU")
        if self.ingress.source_time_unit != "us":
            raise ValueError("received IMU ingress source time must be in us")
        if self.ingress.source_time_value != imu.timestamp_us:
            raise ValueError("received IMU source time does not match timestamp_us")

    def validate_integrity(self) -> None:
        """Revalidate the nested ingress and mutable ``IMUData`` container."""

        if type(self.ingress) is not MavlinkIngressV1:
            raise TypeError("ingress must be exact MavlinkIngressV1")
        self.ingress.validate_integrity()
        copied = _copy_imu_data(self.imu)
        self._validate_binding(copied)

    def to_primitive(self) -> dict[str, Any]:
        self.validate_integrity()
        imu = _copy_imu_data(self.imu)
        return {
            "schema": self.SCHEMA,
            "ingress": self.ingress.to_primitive(),
            "imu": {
                "timestamp_us": imu.timestamp_us,
                "accel": list(imu.accel),
                "gyro": list(imu.gyro),
                "mag": None if imu.mag is None else list(imu.mag),
            },
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "ReceivedIMUSampleV1":
        row = _exact_object(value, {"schema", "ingress", "imu"}, cls.SCHEMA)
        if row["schema"] != cls.SCHEMA:
            raise ValueError("unsupported received IMU schema")
        imu_row = _exact_object(
            row["imu"],
            {"timestamp_us", "accel", "gyro", "mag"},
            "received IMU payload",
        )
        timestamp_us = _exact_nonnegative_int(
            imu_row["timestamp_us"],
            "imu.timestamp_us",
            maximum=_UINT64_MAX,
        )
        accel = _primitive_vector3(imu_row["accel"], "imu.accel")
        gyro = _primitive_vector3(imu_row["gyro"], "imu.gyro")
        mag_value = imu_row["mag"]
        mag = (
            None
            if mag_value is None
            else _primitive_vector3(mag_value, "imu.mag")
        )
        return cls(
            ingress=MavlinkIngressV1.from_primitive(row["ingress"]),
            imu=IMUData(
                timestamp_us=timestamp_us,
                accel=accel,
                gyro=gyro,
                mag=mag,
            ),
        )


__all__ = [
    "ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE",
    "HEARTBEAT_MESSAGE_TYPE",
    "HIGHRES_IMU_MESSAGE_TYPE",
    "HOST_PERF_COUNTER_CLOCK_ID",
    "MavlinkIngressV1",
    "RACE_STATUS_MESSAGE_TYPE",
    "ReceivedIMUSampleV1",
    "SUPPORTED_MAVLINK_MESSAGE_TYPES",
]
