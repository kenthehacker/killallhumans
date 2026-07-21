"""Pure, authority-neutral capture contracts for build-3385 VQ2 I/O facts.

The values in this module preserve receiver-boundary facts and local outbound
call receipts only. They do not open sockets, convert clocks, infer capture
time, reset or arm the simulator, or send any MAVLink message. An outbound
receipt proves only the exact local API call and normal return or raise, never
vehicle receipt or execution. Source timestamps remain opaque values paired
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
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_UINT8_MAX = (1 << 8) - 1
_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1
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


def _exact_bounded_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
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


def _direct_finite_tuple(
    value: Any,
    label: str,
    *,
    length: int,
) -> tuple[float, ...]:
    if type(value) is not tuple or len(value) != length:
        raise TypeError(f"{label} must be an exact {length}-tuple")
    return tuple(
        _finite_float(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _primitive_finite_tuple(
    value: Any,
    label: str,
    *,
    length: int,
) -> tuple[float, ...]:
    if type(value) is not list or len(value) != length:
        raise TypeError(f"{label} must be an exact {length}-element array")
    return tuple(
        _finite_float(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _validate_schema(value: Any, expected: str, label: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{label} schema must be an exact string")
    if value != expected:
        raise ValueError(f"unsupported {label} schema")


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


@dataclass(frozen=True, slots=True)
class HeartbeatPayloadV1:
    """Exact immutable HEARTBEAT fields admitted by the powered contract."""

    base_mode: int
    custom_mode: int

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        _exact_bounded_int(
            self.base_mode,
            "heartbeat.base_mode",
            minimum=0,
            maximum=_UINT8_MAX,
        )
        _exact_bounded_int(
            self.custom_mode,
            "heartbeat.custom_mode",
            minimum=0,
            maximum=_UINT32_MAX,
        )

    def to_primitive(self) -> dict[str, int]:
        self.validate_integrity()
        return {"base_mode": self.base_mode, "custom_mode": self.custom_mode}

    @classmethod
    def from_primitive(cls, value: Any) -> "HeartbeatPayloadV1":
        row = _exact_object(
            value,
            {"base_mode", "custom_mode"},
            "received heartbeat payload",
        )
        return cls(base_mode=row["base_mode"], custom_mode=row["custom_mode"])


@dataclass(frozen=True, slots=True)
class RaceStatusPayloadV1:
    """Exact immutable RACE_STATUS fields admitted by the powered contract."""

    sim_boot_time_ms: int
    race_start_boot_time_ms: int
    race_finish_time_ns: int
    active_gate_index: int
    last_gate_race_time: int

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        _exact_nonnegative_int(
            self.sim_boot_time_ms,
            "race_status.sim_boot_time_ms",
            maximum=_UINT64_MAX,
        )
        _exact_bounded_int(
            self.race_start_boot_time_ms,
            "race_status.race_start_boot_time_ms",
            minimum=_INT64_MIN,
            maximum=_INT64_MAX,
        )
        _exact_bounded_int(
            self.race_finish_time_ns,
            "race_status.race_finish_time_ns",
            minimum=_INT64_MIN,
            maximum=_INT64_MAX,
        )
        _exact_nonnegative_int(
            self.active_gate_index,
            "race_status.active_gate_index",
            maximum=_UINT32_MAX,
        )
        _exact_bounded_int(
            self.last_gate_race_time,
            "race_status.last_gate_race_time",
            minimum=_INT64_MIN,
            maximum=_INT64_MAX,
        )

    def to_primitive(self) -> dict[str, int]:
        self.validate_integrity()
        return {
            "sim_boot_time_ms": self.sim_boot_time_ms,
            "race_start_boot_time_ms": self.race_start_boot_time_ms,
            "race_finish_time_ns": self.race_finish_time_ns,
            "active_gate_index": self.active_gate_index,
            "last_gate_race_time": self.last_gate_race_time,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "RaceStatusPayloadV1":
        row = _exact_object(
            value,
            {
                "sim_boot_time_ms",
                "race_start_boot_time_ms",
                "race_finish_time_ns",
                "active_gate_index",
                "last_gate_race_time",
            },
            "received race-status payload",
        )
        return cls(
            sim_boot_time_ms=row["sim_boot_time_ms"],
            race_start_boot_time_ms=row["race_start_boot_time_ms"],
            race_finish_time_ns=row["race_finish_time_ns"],
            active_gate_index=row["active_gate_index"],
            last_gate_race_time=row["last_gate_race_time"],
        )


@dataclass(frozen=True, slots=True)
class ActuatorOutputStatusPayloadV1:
    """Exact immutable ACTUATOR_OUTPUT_STATUS fields and 32-channel vector."""

    time_usec: int
    active: int
    actuator: tuple[float, ...]

    def __post_init__(self) -> None:
        _exact_nonnegative_int(
            self.time_usec,
            "actuator_output_status.time_usec",
            maximum=_UINT64_MAX,
        )
        _exact_nonnegative_int(
            self.active,
            "actuator_output_status.active",
            maximum=_UINT32_MAX,
        )
        object.__setattr__(
            self,
            "actuator",
            _direct_finite_tuple(
                self.actuator,
                "actuator_output_status.actuator",
                length=32,
            ),
        )

    def validate_integrity(self) -> None:
        _exact_nonnegative_int(
            self.time_usec,
            "actuator_output_status.time_usec",
            maximum=_UINT64_MAX,
        )
        _exact_nonnegative_int(
            self.active,
            "actuator_output_status.active",
            maximum=_UINT32_MAX,
        )
        _direct_finite_tuple(
            self.actuator,
            "actuator_output_status.actuator",
            length=32,
        )

    def to_primitive(self) -> dict[str, Any]:
        self.validate_integrity()
        return {
            "time_usec": self.time_usec,
            "active": self.active,
            "actuator": list(self.actuator),
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "ActuatorOutputStatusPayloadV1":
        row = _exact_object(
            value,
            {"time_usec", "active", "actuator"},
            "received actuator-output-status payload",
        )
        return cls(
            time_usec=row["time_usec"],
            active=row["active"],
            actuator=_primitive_finite_tuple(
                row["actuator"],
                "actuator_output_status.actuator",
                length=32,
            ),
        )


def _copy_heartbeat_payload(value: Any) -> HeartbeatPayloadV1:
    if type(value) is not HeartbeatPayloadV1:
        raise TypeError("heartbeat must be exact HeartbeatPayloadV1")
    value.validate_integrity()
    return HeartbeatPayloadV1(
        base_mode=value.base_mode,
        custom_mode=value.custom_mode,
    )


def _copy_race_status_payload(value: Any) -> RaceStatusPayloadV1:
    if type(value) is not RaceStatusPayloadV1:
        raise TypeError("race_status must be exact RaceStatusPayloadV1")
    value.validate_integrity()
    return RaceStatusPayloadV1(
        sim_boot_time_ms=value.sim_boot_time_ms,
        race_start_boot_time_ms=value.race_start_boot_time_ms,
        race_finish_time_ns=value.race_finish_time_ns,
        active_gate_index=value.active_gate_index,
        last_gate_race_time=value.last_gate_race_time,
    )


def _copy_actuator_output_status_payload(
    value: Any,
) -> ActuatorOutputStatusPayloadV1:
    if type(value) is not ActuatorOutputStatusPayloadV1:
        raise TypeError(
            "actuator_output_status must be exact ActuatorOutputStatusPayloadV1"
        )
    value.validate_integrity()
    return ActuatorOutputStatusPayloadV1(
        time_usec=value.time_usec,
        active=value.active,
        actuator=tuple(value.actuator),
    )


def _validate_received_ingress(
    ingress: Any,
    *,
    message_type: str,
    source_time_unit: str | None,
    source_time_value: int | None,
    label: str,
) -> None:
    if type(ingress) is not MavlinkIngressV1:
        raise TypeError("ingress must be exact MavlinkIngressV1")
    ingress.validate_integrity()
    if ingress.message_type != message_type:
        raise ValueError(f"{label} ingress must be {message_type}")
    if ingress.source_time_unit != source_time_unit:
        if source_time_unit is None:
            raise ValueError(f"{label} ingress source time must be absent")
        raise ValueError(f"{label} ingress source time must be in {source_time_unit}")
    if ingress.source_time_value != source_time_value:
        raise ValueError(f"{label} source time does not match payload")


@dataclass(frozen=True, slots=True)
class ReceivedHeartbeatV1:
    """One exact HEARTBEAT ingress paired with a defensive payload copy."""

    SCHEMA: ClassVar[str] = "aigp-vq2-received-heartbeat/1"

    ingress: MavlinkIngressV1
    heartbeat: HeartbeatPayloadV1

    def __post_init__(self) -> None:
        copied = _copy_heartbeat_payload(self.heartbeat)
        _validate_received_ingress(
            self.ingress,
            message_type=HEARTBEAT_MESSAGE_TYPE,
            source_time_unit=None,
            source_time_value=None,
            label="received heartbeat",
        )
        object.__setattr__(self, "heartbeat", copied)

    def validate_integrity(self) -> None:
        payload = _copy_heartbeat_payload(self.heartbeat)
        _validate_received_ingress(
            self.ingress,
            message_type=HEARTBEAT_MESSAGE_TYPE,
            source_time_unit=None,
            source_time_value=None,
            label="received heartbeat",
        )
        payload.validate_integrity()

    def to_primitive(self) -> dict[str, Any]:
        self.validate_integrity()
        return {
            "schema": self.SCHEMA,
            "ingress": self.ingress.to_primitive(),
            "heartbeat": self.heartbeat.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "ReceivedHeartbeatV1":
        row = _exact_object(
            value,
            {"schema", "ingress", "heartbeat"},
            cls.SCHEMA,
        )
        _validate_schema(row["schema"], cls.SCHEMA, "received heartbeat")
        return cls(
            ingress=MavlinkIngressV1.from_primitive(row["ingress"]),
            heartbeat=HeartbeatPayloadV1.from_primitive(row["heartbeat"]),
        )


@dataclass(frozen=True, slots=True)
class ReceivedRaceStatusV1:
    """One exact RACE_STATUS ingress paired with a defensive payload copy."""

    SCHEMA: ClassVar[str] = "aigp-vq2-received-race-status/1"

    ingress: MavlinkIngressV1
    race_status: RaceStatusPayloadV1

    def __post_init__(self) -> None:
        copied = _copy_race_status_payload(self.race_status)
        _validate_received_ingress(
            self.ingress,
            message_type=RACE_STATUS_MESSAGE_TYPE,
            source_time_unit="ms",
            source_time_value=copied.sim_boot_time_ms,
            label="received race status",
        )
        object.__setattr__(self, "race_status", copied)

    def validate_integrity(self) -> None:
        copied = _copy_race_status_payload(self.race_status)
        _validate_received_ingress(
            self.ingress,
            message_type=RACE_STATUS_MESSAGE_TYPE,
            source_time_unit="ms",
            source_time_value=copied.sim_boot_time_ms,
            label="received race status",
        )

    def to_primitive(self) -> dict[str, Any]:
        self.validate_integrity()
        return {
            "schema": self.SCHEMA,
            "ingress": self.ingress.to_primitive(),
            "race_status": self.race_status.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "ReceivedRaceStatusV1":
        row = _exact_object(
            value,
            {"schema", "ingress", "race_status"},
            cls.SCHEMA,
        )
        _validate_schema(row["schema"], cls.SCHEMA, "received race status")
        return cls(
            ingress=MavlinkIngressV1.from_primitive(row["ingress"]),
            race_status=RaceStatusPayloadV1.from_primitive(row["race_status"]),
        )


@dataclass(frozen=True, slots=True)
class ReceivedActuatorOutputStatusV1:
    """One exact actuator-status ingress and defensive 32-channel copy."""

    SCHEMA: ClassVar[str] = "aigp-vq2-received-actuator-output-status/1"

    ingress: MavlinkIngressV1
    actuator_output_status: ActuatorOutputStatusPayloadV1

    def __post_init__(self) -> None:
        copied = _copy_actuator_output_status_payload(self.actuator_output_status)
        _validate_received_ingress(
            self.ingress,
            message_type=ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE,
            source_time_unit="us",
            source_time_value=copied.time_usec,
            label="received actuator output status",
        )
        object.__setattr__(self, "actuator_output_status", copied)

    def validate_integrity(self) -> None:
        copied = _copy_actuator_output_status_payload(self.actuator_output_status)
        _validate_received_ingress(
            self.ingress,
            message_type=ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE,
            source_time_unit="us",
            source_time_value=copied.time_usec,
            label="received actuator output status",
        )

    def to_primitive(self) -> dict[str, Any]:
        self.validate_integrity()
        return {
            "schema": self.SCHEMA,
            "ingress": self.ingress.to_primitive(),
            "actuator_output_status": self.actuator_output_status.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "ReceivedActuatorOutputStatusV1":
        row = _exact_object(
            value,
            {"schema", "ingress", "actuator_output_status"},
            cls.SCHEMA,
        )
        _validate_schema(
            row["schema"],
            cls.SCHEMA,
            "received actuator output status",
        )
        return cls(
            ingress=MavlinkIngressV1.from_primitive(row["ingress"]),
            actuator_output_status=ActuatorOutputStatusPayloadV1.from_primitive(
                row["actuator_output_status"]
            ),
        )


@dataclass(frozen=True, slots=True)
class AttitudeTargetWireV1:
    """Exact primitive payload passed to MAVLink SET_ATTITUDE_TARGET."""

    time_boot_ms: int
    target_system: int
    target_component: int
    type_mask: int
    q_wxyz: tuple[float, ...]
    body_rates_rad_s: tuple[float, ...]
    thrust: float

    def __post_init__(self) -> None:
        _exact_nonnegative_int(
            self.time_boot_ms,
            "attitude wire.time_boot_ms",
            maximum=_UINT32_MAX,
        )
        for name in ("target_system", "target_component", "type_mask"):
            _exact_nonnegative_int(
                getattr(self, name),
                f"attitude wire.{name}",
                maximum=_UINT8_MAX,
            )
        object.__setattr__(
            self,
            "q_wxyz",
            _direct_finite_tuple(self.q_wxyz, "attitude wire.q_wxyz", length=4),
        )
        object.__setattr__(
            self,
            "body_rates_rad_s",
            _direct_finite_tuple(
                self.body_rates_rad_s,
                "attitude wire.body_rates_rad_s",
                length=3,
            ),
        )
        object.__setattr__(
            self,
            "thrust",
            _finite_float(self.thrust, "attitude wire.thrust"),
        )

    def validate_integrity(self) -> None:
        _exact_nonnegative_int(
            self.time_boot_ms,
            "attitude wire.time_boot_ms",
            maximum=_UINT32_MAX,
        )
        for name in ("target_system", "target_component", "type_mask"):
            _exact_nonnegative_int(
                getattr(self, name),
                f"attitude wire.{name}",
                maximum=_UINT8_MAX,
            )
        _direct_finite_tuple(self.q_wxyz, "attitude wire.q_wxyz", length=4)
        _direct_finite_tuple(
            self.body_rates_rad_s,
            "attitude wire.body_rates_rad_s",
            length=3,
        )
        _finite_float(self.thrust, "attitude wire.thrust")

    def to_primitive(self) -> dict[str, Any]:
        self.validate_integrity()
        return {
            "time_boot_ms": self.time_boot_ms,
            "target_system": self.target_system,
            "target_component": self.target_component,
            "type_mask": self.type_mask,
            "q_wxyz": list(self.q_wxyz),
            "body_rates_rad_s": list(self.body_rates_rad_s),
            "thrust": self.thrust,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "AttitudeTargetWireV1":
        row = _exact_object(
            value,
            {
                "time_boot_ms",
                "target_system",
                "target_component",
                "type_mask",
                "q_wxyz",
                "body_rates_rad_s",
                "thrust",
            },
            "attitude target wire",
        )
        return cls(
            time_boot_ms=row["time_boot_ms"],
            target_system=row["target_system"],
            target_component=row["target_component"],
            type_mask=row["type_mask"],
            q_wxyz=_primitive_finite_tuple(
                row["q_wxyz"],
                "attitude wire.q_wxyz",
                length=4,
            ),
            body_rates_rad_s=_primitive_finite_tuple(
                row["body_rates_rad_s"],
                "attitude wire.body_rates_rad_s",
                length=3,
            ),
            thrust=row["thrust"],
        )


@dataclass(frozen=True, slots=True)
class CommandLongWireV1:
    """Exact primitive COMMAND_LONG payload for arm, disarm, or reset."""

    target_system: int
    target_component: int
    command: int
    confirmation: int
    params: tuple[float, ...]

    def __post_init__(self) -> None:
        _exact_nonnegative_int(
            self.target_system,
            "command-long wire.target_system",
            maximum=_UINT8_MAX,
        )
        _exact_nonnegative_int(
            self.target_component,
            "command-long wire.target_component",
            maximum=_UINT8_MAX,
        )
        _exact_nonnegative_int(
            self.command,
            "command-long wire.command",
            maximum=_UINT16_MAX,
        )
        _exact_nonnegative_int(
            self.confirmation,
            "command-long wire.confirmation",
            maximum=_UINT8_MAX,
        )
        object.__setattr__(
            self,
            "params",
            _direct_finite_tuple(
                self.params,
                "command-long wire.params",
                length=7,
            ),
        )

    def validate_integrity(self) -> None:
        _exact_nonnegative_int(
            self.target_system,
            "command-long wire.target_system",
            maximum=_UINT8_MAX,
        )
        _exact_nonnegative_int(
            self.target_component,
            "command-long wire.target_component",
            maximum=_UINT8_MAX,
        )
        _exact_nonnegative_int(
            self.command,
            "command-long wire.command",
            maximum=_UINT16_MAX,
        )
        _exact_nonnegative_int(
            self.confirmation,
            "command-long wire.confirmation",
            maximum=_UINT8_MAX,
        )
        _direct_finite_tuple(
            self.params,
            "command-long wire.params",
            length=7,
        )

    def to_primitive(self) -> dict[str, Any]:
        self.validate_integrity()
        return {
            "target_system": self.target_system,
            "target_component": self.target_component,
            "command": self.command,
            "confirmation": self.confirmation,
            "params": list(self.params),
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "CommandLongWireV1":
        row = _exact_object(
            value,
            {
                "target_system",
                "target_component",
                "command",
                "confirmation",
                "params",
            },
            "command-long wire",
        )
        return cls(
            target_system=row["target_system"],
            target_component=row["target_component"],
            command=row["command"],
            confirmation=row["confirmation"],
            params=_primitive_finite_tuple(
                row["params"],
                "command-long wire.params",
                length=7,
            ),
        )


@dataclass(frozen=True, slots=True)
class TimesyncWireV1:
    """Exact primitive TIMESYNC payload."""

    tc1: int
    ts1: int

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        _exact_bounded_int(
            self.tc1,
            "timesync wire.tc1",
            minimum=_INT64_MIN,
            maximum=_INT64_MAX,
        )
        _exact_bounded_int(
            self.ts1,
            "timesync wire.ts1",
            minimum=_INT64_MIN,
            maximum=_INT64_MAX,
        )

    def to_primitive(self) -> dict[str, int]:
        self.validate_integrity()
        return {"tc1": self.tc1, "ts1": self.ts1}

    @classmethod
    def from_primitive(cls, value: Any) -> "TimesyncWireV1":
        row = _exact_object(value, {"tc1", "ts1"}, "timesync wire")
        return cls(tc1=row["tc1"], ts1=row["ts1"])


@dataclass(frozen=True, slots=True)
class GCSHeartbeatWireV1:
    """Exact primitive GCS HEARTBEAT payload."""

    type: int
    autopilot: int
    base_mode: int
    custom_mode: int
    system_status: int

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        for name in ("type", "autopilot", "base_mode", "system_status"):
            _exact_nonnegative_int(
                getattr(self, name),
                f"GCS heartbeat wire.{name}",
                maximum=_UINT8_MAX,
            )
        _exact_nonnegative_int(
            self.custom_mode,
            "GCS heartbeat wire.custom_mode",
            maximum=_UINT32_MAX,
        )

    def to_primitive(self) -> dict[str, int]:
        self.validate_integrity()
        return {
            "type": self.type,
            "autopilot": self.autopilot,
            "base_mode": self.base_mode,
            "custom_mode": self.custom_mode,
            "system_status": self.system_status,
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "GCSHeartbeatWireV1":
        row = _exact_object(
            value,
            {"type", "autopilot", "base_mode", "custom_mode", "system_status"},
            "GCS heartbeat wire",
        )
        return cls(
            type=row["type"],
            autopilot=row["autopilot"],
            base_mode=row["base_mode"],
            custom_mode=row["custom_mode"],
            system_status=row["system_status"],
        )


_ATTITUDE_TARGET_APIS = frozenset(
    {
        "send_attitude_rate",
        "send_attitude_rate_from_attitude",
        "send_attitude_quaternion",
    }
)
_COMMAND_LONG_CATEGORIES = frozenset({"arm", "disarm", "sim_reset"})
_NONATTITUDE_CATEGORY_API = {
    "arm": "command_long_send",
    "disarm": "command_long_send",
    "sim_reset": "command_long_send",
    "timesync": "timesync_send",
    "gcs_heartbeat": "heartbeat_send",
}
_OUTBOUND_OUTCOMES = frozenset({"returned", "raised"})


def _copy_attitude_target_wire(value: Any) -> AttitudeTargetWireV1:
    if type(value) is not AttitudeTargetWireV1:
        raise TypeError("wire must be exact AttitudeTargetWireV1")
    value.validate_integrity()
    return AttitudeTargetWireV1(
        time_boot_ms=value.time_boot_ms,
        target_system=value.target_system,
        target_component=value.target_component,
        type_mask=value.type_mask,
        q_wxyz=tuple(value.q_wxyz),
        body_rates_rad_s=tuple(value.body_rates_rad_s),
        thrust=value.thrust,
    )


def _copy_nonattitude_wire(
    value: Any,
) -> CommandLongWireV1 | TimesyncWireV1 | GCSHeartbeatWireV1:
    if type(value) is CommandLongWireV1:
        value.validate_integrity()
        return CommandLongWireV1(
            target_system=value.target_system,
            target_component=value.target_component,
            command=value.command,
            confirmation=value.confirmation,
            params=tuple(value.params),
        )
    if type(value) is TimesyncWireV1:
        value.validate_integrity()
        return TimesyncWireV1(tc1=value.tc1, ts1=value.ts1)
    if type(value) is GCSHeartbeatWireV1:
        value.validate_integrity()
        return GCSHeartbeatWireV1(
            type=value.type,
            autopilot=value.autopilot,
            base_mode=value.base_mode,
            custom_mode=value.custom_mode,
            system_status=value.system_status,
        )
    raise TypeError("wire must be an exact admitted nonattitude wire type")


def _validate_outbound_metadata(
    *,
    stream_id: Any,
    reset_generation: Any,
    outbound_sequence: Any,
    host_clock_id: Any,
    call_start_monotonic_ns: Any,
    call_end_monotonic_ns: Any,
    outcome: Any,
    error_type: Any,
) -> None:
    _bounded_token(stream_id, "stream_id")
    _exact_nonnegative_int(reset_generation, "reset_generation")
    _exact_nonnegative_int(outbound_sequence, "outbound_sequence")
    _bounded_token(host_clock_id, "host_clock_id")
    if host_clock_id != HOST_PERF_COUNTER_CLOCK_ID:
        raise ValueError("outbound receipt must use host-perf-counter")
    start = _exact_nonnegative_int(
        call_start_monotonic_ns,
        "call_start_monotonic_ns",
        maximum=_UINT64_MAX,
    )
    end = _exact_nonnegative_int(
        call_end_monotonic_ns,
        "call_end_monotonic_ns",
        maximum=_UINT64_MAX,
    )
    if end < start:
        raise ValueError("call_end_monotonic_ns must be >= call_start_monotonic_ns")
    if type(outcome) is not str:
        raise TypeError("outcome must be an exact string")
    if outcome not in _OUTBOUND_OUTCOMES:
        raise ValueError("unsupported outbound outcome")
    if outcome == "returned":
        if error_type is not None:
            raise ValueError("returned outbound receipt must have null error_type")
    else:
        _bounded_token(error_type, "error_type")


@dataclass(frozen=True, slots=True)
class AttitudeTargetOutboundV1:
    """Exact local attitude-target call receipt, not a delivery ACK."""

    SCHEMA: ClassVar[str] = "aigp-vq2-attitude-target-outbound/1"

    stream_id: str
    reset_generation: int
    outbound_sequence: int
    host_clock_id: str
    call_start_monotonic_ns: int
    call_end_monotonic_ns: int
    api: str
    outcome: str
    error_type: str | None
    wire: AttitudeTargetWireV1

    def __post_init__(self) -> None:
        copied = _copy_attitude_target_wire(self.wire)
        self._validate_metadata()
        object.__setattr__(self, "wire", copied)

    def _validate_metadata(self) -> None:
        _validate_outbound_metadata(
            stream_id=self.stream_id,
            reset_generation=self.reset_generation,
            outbound_sequence=self.outbound_sequence,
            host_clock_id=self.host_clock_id,
            call_start_monotonic_ns=self.call_start_monotonic_ns,
            call_end_monotonic_ns=self.call_end_monotonic_ns,
            outcome=self.outcome,
            error_type=self.error_type,
        )
        if type(self.api) is not str:
            raise TypeError("api must be an exact string")
        if self.api not in _ATTITUDE_TARGET_APIS:
            raise ValueError("unsupported attitude-target API")

    def validate_integrity(self) -> None:
        self._validate_metadata()
        _copy_attitude_target_wire(self.wire)

    def to_primitive(self) -> dict[str, Any]:
        self.validate_integrity()
        return {
            "schema": self.SCHEMA,
            "stream_id": self.stream_id,
            "reset_generation": self.reset_generation,
            "outbound_sequence": self.outbound_sequence,
            "host_clock_id": self.host_clock_id,
            "call_start_monotonic_ns": self.call_start_monotonic_ns,
            "call_end_monotonic_ns": self.call_end_monotonic_ns,
            "api": self.api,
            "outcome": self.outcome,
            "error_type": self.error_type,
            "wire": self.wire.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "AttitudeTargetOutboundV1":
        row = _exact_object(
            value,
            {
                "schema",
                "stream_id",
                "reset_generation",
                "outbound_sequence",
                "host_clock_id",
                "call_start_monotonic_ns",
                "call_end_monotonic_ns",
                "api",
                "outcome",
                "error_type",
                "wire",
            },
            cls.SCHEMA,
        )
        _validate_schema(row["schema"], cls.SCHEMA, "attitude-target outbound")
        return cls(
            stream_id=row["stream_id"],
            reset_generation=row["reset_generation"],
            outbound_sequence=row["outbound_sequence"],
            host_clock_id=row["host_clock_id"],
            call_start_monotonic_ns=row["call_start_monotonic_ns"],
            call_end_monotonic_ns=row["call_end_monotonic_ns"],
            api=row["api"],
            outcome=row["outcome"],
            error_type=row["error_type"],
            wire=AttitudeTargetWireV1.from_primitive(row["wire"]),
        )


@dataclass(frozen=True, slots=True)
class NonAttitudeOutboundV1:
    """Exact local nonattitude call receipt, not a delivery ACK."""

    SCHEMA: ClassVar[str] = "aigp-vq2-nonattitude-outbound/1"

    stream_id: str
    reset_generation: int
    outbound_sequence: int
    host_clock_id: str
    call_start_monotonic_ns: int
    call_end_monotonic_ns: int
    category: str
    api: str
    outcome: str
    error_type: str | None
    wire: CommandLongWireV1 | TimesyncWireV1 | GCSHeartbeatWireV1

    def __post_init__(self) -> None:
        copied = _copy_nonattitude_wire(self.wire)
        self._validate_metadata(copied)
        object.__setattr__(self, "wire", copied)

    def _validate_metadata(
        self,
        wire: CommandLongWireV1 | TimesyncWireV1 | GCSHeartbeatWireV1,
    ) -> None:
        _validate_outbound_metadata(
            stream_id=self.stream_id,
            reset_generation=self.reset_generation,
            outbound_sequence=self.outbound_sequence,
            host_clock_id=self.host_clock_id,
            call_start_monotonic_ns=self.call_start_monotonic_ns,
            call_end_monotonic_ns=self.call_end_monotonic_ns,
            outcome=self.outcome,
            error_type=self.error_type,
        )
        if type(self.category) is not str:
            raise TypeError("category must be an exact string")
        expected_api = _NONATTITUDE_CATEGORY_API.get(self.category)
        if expected_api is None:
            raise ValueError("unsupported nonattitude category")
        if type(self.api) is not str:
            raise TypeError("api must be an exact string")
        if self.api != expected_api:
            raise ValueError("nonattitude category/API mismatch")
        if self.category in _COMMAND_LONG_CATEGORIES:
            expected_wire = CommandLongWireV1
        elif self.category == "timesync":
            expected_wire = TimesyncWireV1
        else:
            expected_wire = GCSHeartbeatWireV1
        if type(wire) is not expected_wire:
            raise TypeError("nonattitude category/wire mismatch")

    def validate_integrity(self) -> None:
        wire = _copy_nonattitude_wire(self.wire)
        self._validate_metadata(wire)

    def to_primitive(self) -> dict[str, Any]:
        self.validate_integrity()
        return {
            "schema": self.SCHEMA,
            "stream_id": self.stream_id,
            "reset_generation": self.reset_generation,
            "outbound_sequence": self.outbound_sequence,
            "host_clock_id": self.host_clock_id,
            "call_start_monotonic_ns": self.call_start_monotonic_ns,
            "call_end_monotonic_ns": self.call_end_monotonic_ns,
            "category": self.category,
            "api": self.api,
            "outcome": self.outcome,
            "error_type": self.error_type,
            "wire": self.wire.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: Any) -> "NonAttitudeOutboundV1":
        row = _exact_object(
            value,
            {
                "schema",
                "stream_id",
                "reset_generation",
                "outbound_sequence",
                "host_clock_id",
                "call_start_monotonic_ns",
                "call_end_monotonic_ns",
                "category",
                "api",
                "outcome",
                "error_type",
                "wire",
            },
            cls.SCHEMA,
        )
        _validate_schema(row["schema"], cls.SCHEMA, "nonattitude outbound")
        category = row["category"]
        if type(category) is not str:
            raise TypeError("category must be an exact string")
        if category in _COMMAND_LONG_CATEGORIES:
            wire = CommandLongWireV1.from_primitive(row["wire"])
        elif category == "timesync":
            wire = TimesyncWireV1.from_primitive(row["wire"])
        elif category == "gcs_heartbeat":
            wire = GCSHeartbeatWireV1.from_primitive(row["wire"])
        else:
            raise ValueError("unsupported nonattitude category")
        return cls(
            stream_id=row["stream_id"],
            reset_generation=row["reset_generation"],
            outbound_sequence=row["outbound_sequence"],
            host_clock_id=row["host_clock_id"],
            call_start_monotonic_ns=row["call_start_monotonic_ns"],
            call_end_monotonic_ns=row["call_end_monotonic_ns"],
            category=category,
            api=row["api"],
            outcome=row["outcome"],
            error_type=row["error_type"],
            wire=wire,
        )


__all__ = [
    "ACTUATOR_OUTPUT_STATUS_MESSAGE_TYPE",
    "ActuatorOutputStatusPayloadV1",
    "AttitudeTargetOutboundV1",
    "AttitudeTargetWireV1",
    "CommandLongWireV1",
    "GCSHeartbeatWireV1",
    "HEARTBEAT_MESSAGE_TYPE",
    "HIGHRES_IMU_MESSAGE_TYPE",
    "HOST_PERF_COUNTER_CLOCK_ID",
    "HeartbeatPayloadV1",
    "MavlinkIngressV1",
    "NonAttitudeOutboundV1",
    "RACE_STATUS_MESSAGE_TYPE",
    "RaceStatusPayloadV1",
    "ReceivedActuatorOutputStatusV1",
    "ReceivedHeartbeatV1",
    "ReceivedIMUSampleV1",
    "ReceivedRaceStatusV1",
    "SUPPORTED_MAVLINK_MESSAGE_TYPES",
    "TimesyncWireV1",
]
