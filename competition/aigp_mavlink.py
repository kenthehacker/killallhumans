"""Raw pymavlink transport for the official AIGP simulator.

The official sim speaks MAVLink2 on UDP 14550 plus a separate JPEG-over-UDP
vision stream. This adapter deliberately uses pymavlink directly so custom
``ENCAPSULATED_DATA`` race-status and track-info packets are visible.

``pymavlink`` is imported lazily in :meth:`connect`; importing this module and
running unit tests does not require the package or a live socket.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import logging
import math
import re
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Deque, Dict, Optional

from competition.adapter import (
    AttitudeCommand,
    AttitudeRateCommand,
    CameraFrame,
    CompetitionInterface,
    IMUData,
    PositionCommand,
    Quaternion,
    RaceActiveBoundaryChangedBeforeWire,
    TelemetryState,
)
from competition.aigp_geometry import AIGP_CAM_UDP_PORT
from competition.aigp_messages import (
    ENCAPSULATED_RACE_STATUS_MSG_ID,
    ENCAPSULATED_TRACK_INFO_MSG_ID,
    RaceStatus,
    TrackData,
    TrackInfoReassembler,
    parse_race_status,
)
from competition.aigp_recorder import (
    mavlink_msg_to_fields,
    race_status_fields,
    record_for_message,
    track_data_fields,
    write_jsonl,
)
from competition.vq2_runtime import VQ2_HOST_CLOCK_ID
from competition.vision_udp import VisionUdpListener

if TYPE_CHECKING:
    from competition.vq2_capture import (
        AttitudeTargetOutboundV1,
        MavlinkIngressV1,
        NonAttitudeOutboundV1,
        ReceivedActuatorOutputStatusV1,
        ReceivedHeartbeatV1,
        ReceivedIMUSampleV1,
        ReceivedRaceStatusV1,
    )

logger = logging.getLogger(__name__)

DEFAULT_MAVLINK_URL = "udpin:127.0.0.1:14550"
SIM_RESET_COMMAND = 31000
MAV_CMD_COMPONENT_ARM_DISARM = 400
SET_ATTITUDE_TARGET_MASK_ATTITUDE_THRUST = 0b00000111
SET_ATTITUDE_TARGET_MASK_RATES_THRUST = 128
SET_POSITION_TARGET_LOCAL_NED_MASK = 2496
MAV_FRAME_LOCAL_NED = 1

VQ2_MAVLINK_STREAM_ID = "vq2-mavlink-udp-14550"
DEFAULT_INGRESS_BUFFER_CAPACITY = 4096
DEFAULT_OUTBOUND_RECEIPT_CAPACITY = 4096
DEFAULT_COLLISION_BUFFER_CAPACITY = 128
DEFAULT_RESET_PERSISTENCE_FAILURE_CAPACITY = 16
POWERED_MAX_DATAGRAM_BYTES = 65_535
POWERED_WORKER_POLL_NS = 50_000_000
POWERED_OUTBOUND_CALL_NS = 250_000_000
POWERED_RECEIVE_MODE_WORKER = "worker"
POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP = "external_cleanup"
POWERED_RECEIVE_OWNER_WORKER = "adapter_worker"
POWERED_RECEIVE_OWNER_EXTERNAL_CLEANUP = "external_cleanup"
_OUTBOUND_ERROR_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


@dataclass(frozen=True)
class MavlinkIngressStats:
    """Bounded receiver-ingress diagnostics for one adapter connection."""

    generation: int
    next_sequence: int
    highres_imu_received: int
    heartbeat_received: int
    race_status_received: int
    actuator_received: int
    dropped: int
    high_watermark: int
    imu_capacity: int
    other_capacity: int
    imu_dropped: int
    other_dropped: int
    imu_high_watermark: int
    other_high_watermark: int
    buffered_imu: int
    buffered_other: int


@dataclass(frozen=True)
class MavlinkOutboundAudit:
    """Attempted outbound MAVLink messages, separated by passive authority."""

    timesync: int
    gcs_heartbeat: int
    sim_reset: int
    arm: int
    disarm: int
    attitude_target: int
    position_target: int
    other_command: int

    @property
    def disallowed_count(self) -> int:
        return (
            self.sim_reset
            + self.arm
            + self.disarm
            + self.attitude_target
            + self.position_target
            + self.other_command
        )


@dataclass(frozen=True)
class MavlinkOutboundReceiptStats:
    """Bounded attempt-global local-call receipt diagnostics.

    ``generation`` identifies the reset generation used by the next receipt;
    sequence and queue counters never restart at an in-attempt reset.
    """

    generation: int
    next_sequence: int
    returned: int
    raised: int
    dropped: int
    high_watermark: int
    capacity: int
    buffered: int


@dataclass(frozen=True)
class PoweredMavlinkTransportState:
    """Public immutable endpoint/worker state for powered close evidence."""

    requested_host: str
    requested_port: int
    actual_host: str
    actual_port: int
    frozen_peer: Optional[tuple[str, int]]
    rejected_source_count: int
    endpoint_closed: bool
    receiver_joined: bool
    announcer_joined: bool
    connection_closed: bool

    def __post_init__(self) -> None:
        for name in ("requested_host", "actual_host"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty exact string")
        for name in ("requested_port", "actual_port"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 65_535:
                raise ValueError(f"{name} must be an exact UDP port")
        if self.actual_port == 0:
            raise ValueError("actual_port must be a bound UDP port")
        if self.frozen_peer is not None:
            if (
                type(self.frozen_peer) is not tuple
                or len(self.frozen_peer) != 2
                or type(self.frozen_peer[0]) is not str
                or not self.frozen_peer[0]
                or type(self.frozen_peer[1]) is not int
                or not 1 <= self.frozen_peer[1] <= 65_535
            ):
                raise ValueError("frozen_peer must be an exact host/port tuple")
        if (
            type(self.rejected_source_count) is not int
            or self.rejected_source_count < 0
        ):
            raise ValueError("rejected_source_count must be nonnegative")
        for name in (
            "endpoint_closed",
            "receiver_joined",
            "announcer_joined",
            "connection_closed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")

    @property
    def endpoint_state(self) -> str:
        if self.endpoint_closed:
            return (
                "closed_with_peer"
                if self.frozen_peer is not None
                else "closed_without_peer"
            )
        return "peer_frozen" if self.frozen_peer is not None else "bound"

    @property
    def owned_handles_closed(self) -> bool:
        return bool(
            self.endpoint_closed
            and self.receiver_joined
            and self.announcer_joined
            and self.connection_closed
        )

    def bind_proof(self) -> dict[str, Any]:
        return {
            "family": "AF_INET",
            "requested": {
                "host": self.requested_host,
                "port": self.requested_port,
            },
            "actual": {"host": self.actual_host, "port": self.actual_port},
            "socket_policy": "ipv4-exclusive-address-use",
        }


@dataclass(frozen=True, slots=True)
class PoweredDatagramDispatch:
    """One source-gated raw datagram dispatch, without scratch-parser state."""

    source_accepted: bool
    peer_frozen_now: bool
    rejected_source: bool
    malformed: bool
    production_dispatched: bool
    source_promoted: bool
    peer: tuple[str, int] | None
    admitted_message_type: str | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        for name in (
            "source_accepted",
            "peer_frozen_now",
            "rejected_source",
            "malformed",
            "production_dispatched",
            "source_promoted",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        source_outcomes = sum(
            (self.source_accepted, self.rejected_source, self.malformed)
        )
        if source_outcomes not in {0, 1} or (
            source_outcomes == 0 and self.failure_reason is None
        ):
            raise ValueError(
                "powered datagram dispatch must have one source outcome or fail"
            )
        if self.production_dispatched and not self.source_accepted:
            raise ValueError("only an accepted source may reach production parsing")
        if self.peer_frozen_now and not self.source_accepted:
            raise ValueError("only an accepted source may freeze a peer")
        if self.peer is not None and (
            type(self.peer) is not tuple
            or len(self.peer) != 2
            or type(self.peer[0]) is not str
            or type(self.peer[1]) is not int
        ):
            raise ValueError("powered datagram peer is invalid")
        if self.admitted_message_type is not None and (
            type(self.admitted_message_type) is not str
            or not self.admitted_message_type
        ):
            raise ValueError("admitted message type is invalid")
        if self.failure_reason is not None and (
            type(self.failure_reason) is not str or not self.failure_reason
        ):
            raise ValueError("powered datagram failure reason is invalid")


@dataclass(frozen=True)
class MavlinkCollisionStats:
    """Calibration-only collision-buffer diagnostics for one generation."""

    generation: int
    handled: int
    dropped: int
    high_watermark: int
    capacity: int
    buffered: int

    def to_primitive(self) -> dict[str, int]:
        return {
            "generation": self.generation,
            "handled": self.handled,
            "dropped": self.dropped,
            "high_watermark": self.high_watermark,
            "capacity": self.capacity,
            "buffered": self.buffered,
        }


@dataclass(frozen=True, slots=True)
class CalibrationCollisionV1:
    """Immutable copy of one legacy collision item at the reset boundary."""

    id: int
    threat_level: int
    impulse: float

    def __post_init__(self) -> None:
        if type(self.id) is not int or not 0 <= self.id <= 0xFFFFFFFF:
            raise ValueError("collision id must be a uint32 exact integer")
        if (
            type(self.threat_level) is not int
            or not 0 <= self.threat_level <= 0xFF
        ):
            raise ValueError("collision threat_level must be a uint8 exact integer")
        if isinstance(self.impulse, bool) or not isinstance(self.impulse, (int, float)):
            raise TypeError("collision impulse must be a real number")
        if not math.isfinite(float(self.impulse)):
            raise ValueError("collision impulse must be finite")
        object.__setattr__(self, "impulse", float(self.impulse))

    def to_primitive(self) -> dict[str, int | float]:
        return {
            "id": self.id,
            "threat_level": self.threat_level,
            "impulse": self.impulse,
        }


@dataclass(frozen=True, slots=True)
class CalibrationResetBoundaryV1:
    """Atomic old-generation capture returned before a calibration reset send."""

    SCHEMA: ClassVar[str] = "aigp-vq2-calibration-reset-boundary/1"

    old_generation: int
    new_generation: int
    boundary_monotonic_ns: int
    observations: tuple[Any, ...]
    collisions: tuple[CalibrationCollisionV1, ...]
    ingress_stats: MavlinkIngressStats
    collision_stats: MavlinkCollisionStats

    def __post_init__(self) -> None:
        if type(self.old_generation) is not int or self.old_generation < 0:
            raise ValueError("old_generation must be a non-negative exact integer")
        if self.new_generation != self.old_generation + 1:
            raise ValueError("new_generation must equal old_generation + 1")
        if (
            type(self.boundary_monotonic_ns) is not int
            or self.boundary_monotonic_ns < 0
        ):
            raise ValueError(
                "boundary_monotonic_ns must be a non-negative exact integer"
            )
        if self.ingress_stats.generation != self.old_generation:
            raise ValueError("ingress_stats must describe old_generation")
        if self.collision_stats.generation != self.old_generation:
            raise ValueError("collision_stats must describe old_generation")
        prior_sequence = -1
        for observation in self.observations:
            ingress = observation.ingress
            observation.validate_integrity()
            if ingress.generation != self.old_generation:
                raise ValueError("boundary observation generation mismatch")
            if ingress.sequence <= prior_sequence:
                raise ValueError("boundary observations must be in ingress order")
            prior_sequence = ingress.sequence

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "old_generation": self.old_generation,
            "new_generation": self.new_generation,
            "boundary_monotonic_ns": self.boundary_monotonic_ns,
            "observations": [item.to_primitive() for item in self.observations],
            "collisions": [item.to_primitive() for item in self.collisions],
            "ingress_stats": {
                name: getattr(self.ingress_stats, name)
                for name in self.ingress_stats.__dataclass_fields__
            },
            "collision_stats": self.collision_stats.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class CalibrationResetPersistenceFailureV1:
    """Redacted in-memory proof that cleanup boundary persistence failed."""

    SCHEMA: ClassVar[str] = "aigp-vq2-reset-persistence-failure/1"

    old_generation: int
    new_generation: int
    boundary_monotonic_ns: int
    error_type: str

    def __post_init__(self) -> None:
        if type(self.old_generation) is not int or self.old_generation < 0:
            raise ValueError("old_generation must be a non-negative exact integer")
        if self.new_generation != self.old_generation + 1:
            raise ValueError("new_generation must equal old_generation + 1")
        if (
            type(self.boundary_monotonic_ns) is not int
            or self.boundary_monotonic_ns < 0
        ):
            raise ValueError(
                "boundary_monotonic_ns must be a non-negative exact integer"
            )
        if (
            type(self.error_type) is not str
            or _OUTBOUND_ERROR_TOKEN_RE.fullmatch(self.error_type) is None
        ):
            raise ValueError("error_type must be a bounded redacted token")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "old_generation": self.old_generation,
            "new_generation": self.new_generation,
            "boundary_monotonic_ns": self.boundary_monotonic_ns,
            "error_type": self.error_type,
        }


@dataclass(frozen=True, slots=True)
class CalibrationResetPersistenceState:
    """Bounded attempt-global snapshot of cleanup persistence failures."""

    failures: tuple[CalibrationResetPersistenceFailureV1, ...]
    dropped: int

    def __post_init__(self) -> None:
        if type(self.failures) is not tuple:
            raise TypeError("failures must be an exact tuple")
        if any(
            type(item) is not CalibrationResetPersistenceFailureV1
            for item in self.failures
        ):
            raise TypeError("failures contain an invalid persistence proof")
        if type(self.dropped) is not int or self.dropped < 0:
            raise ValueError("dropped must be a non-negative exact integer")

    @property
    def failure_latched(self) -> bool:
        return bool(self.failures or self.dropped)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "failures": [item.to_primitive() for item in self.failures],
            "dropped": self.dropped,
            "failure_latched": self.failure_latched,
        }


def _admitted_outbound_error_type(exc: BaseException) -> str:
    """Return a deterministic capture-token without exposing exception text."""

    name = type(exc).__name__
    if _OUTBOUND_ERROR_TOKEN_RE.fullmatch(name) is not None:
        return name
    digest = hashlib.sha256(
        name.encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:20]
    return f"ExceptionType-{digest}"


class _FrozenPeerDatagramWriter:
    """Pymavlink file seam that can write only to the source-frozen peer."""

    def __init__(self, endpoint, source_gate, authority_error_type) -> None:
        self._endpoint = endpoint
        self._source_gate = source_gate
        self._authority_error_type = authority_error_type
        self._lock = threading.Lock()
        self._authorized_thread: Optional[int] = None
        self._writes_remaining = 0

    def begin_authorized_call(self) -> None:
        with self._lock:
            if self._authorized_thread is not None:
                raise self._authority_error_type(
                    "powered MAVLink writer authorization is already active"
                )
            self._authorized_thread = threading.get_ident()
            self._writes_remaining = 1

    def finish_authorized_call(self, *, require_write: bool) -> None:
        with self._lock:
            if self._authorized_thread != threading.get_ident():
                raise self._authority_error_type(
                    "powered MAVLink writer authorization thread changed"
                )
            remaining = self._writes_remaining
            self._authorized_thread = None
            self._writes_remaining = 0
        if require_write and remaining != 0:
            raise self._authority_error_type(
                "powered MAVLink call returned without one datagram write"
            )

    def cancel_authorized_call(self) -> None:
        with self._lock:
            if self._authorized_thread == threading.get_ident():
                self._authorized_thread = None
                self._writes_remaining = 0

    def write(self, payload) -> int:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("encoded MAVLink payload must be bytes-like")
        raw = bytes(payload)
        if not raw:
            raise ValueError("encoded MAVLink payload must not be empty")
        peer = self._source_gate.peer
        if peer is None:
            raise self._authority_error_type("MAVLink peer is not frozen")
        if self._endpoint.closed:
            raise self._authority_error_type("powered UDP endpoint is closed")
        with self._lock:
            if self._authorized_thread != threading.get_ident():
                raise self._authority_error_type(
                    "powered MAVLink write lacks adapter authorization"
                )
            if self._writes_remaining != 1:
                raise self._authority_error_type(
                    "powered MAVLink call attempted multiple datagram writes"
                )
        sent = self._endpoint.socket.sendto(raw, peer)
        if type(sent) is not int or sent != len(raw):
            raise OSError("powered UDP datagram send was incomplete")
        with self._lock:
            self._writes_remaining = 0
        return sent


class _PoweredConnection:
    """Minimal connection projection consumed by the established adapter."""

    def __init__(self, transport: "PoweredMavlinkTransport") -> None:
        self._transport = transport
        self.mav = transport.mavlink
        self.target_system = 1
        self.target_component = 1

    def close(self) -> None:
        self._transport.close()


class PoweredMavlinkTransport:
    """Opt-in raw UDP transport backed by a caller-bound exclusive endpoint.

    Construction takes ownership of ``endpoint``. Any partial construction
    failure closes it. Scratch validation and the production pymavlink object
    are deliberately distinct; only raw bytes accepted by ``source_gate`` are
    passed to ``mavlink.parse_buffer``.
    """

    def __init__(
        self,
        endpoint,
        *,
        scratch_parser_factory: Callable[[], Any],
        mavlink_factory: Callable[[Any], Any],
        outbound_guards,
        role_valid: Callable[[], bool],
        parent_alive: Callable[[], bool],
        lease_valid: Callable[[], bool],
        external_cleanup_authorize: Optional[Callable[..., int]] = None,
    ) -> None:
        from scripts.aigp_vq2_powered_runtime import (
            ExclusiveUdpEndpoint,
            MavlinkSourceFreeze,
            OutboundAuthorityError,
            PoweredOutboundGuards,
            normalize_ipv4_loopback_peer,
        )

        if type(endpoint) is not ExclusiveUdpEndpoint:
            raise TypeError("endpoint must be exact ExclusiveUdpEndpoint")
        try:
            if endpoint.closed is not False:
                raise ValueError("powered UDP endpoint must be open")
            if getattr(endpoint.socket, "family", None) != socket.AF_INET:
                raise ValueError("powered UDP socket must be IPv4")
            socket_type = getattr(endpoint.socket, "type", None)
            if (
                isinstance(socket_type, bool)
                or not isinstance(socket_type, int)
                or socket_type & socket.SOCK_DGRAM == 0
            ):
                raise ValueError("powered UDP socket must be datagram")
            normalize_ipv4_loopback_peer(
                (endpoint.actual_host, endpoint.actual_port)
            )
            actual = endpoint.socket.getsockname()
            if type(actual) is not tuple or len(actual) < 2:
                raise ValueError("powered UDP socket address is invalid")
            if (actual[0], actual[1]) != (
                endpoint.actual_host,
                endpoint.actual_port,
            ):
                raise ValueError("powered UDP endpoint bind proof changed")
            for name in ("recvfrom", "sendto", "close"):
                if not callable(getattr(endpoint.socket, name, None)):
                    raise TypeError(f"powered UDP socket lacks {name}()")
            if not callable(getattr(endpoint.socket, "getsockopt", None)):
                raise TypeError("powered UDP socket lacks getsockopt()")
            if type(endpoint.exclusive_option) is not int:
                raise ValueError("powered UDP exclusive option is invalid")
            if endpoint.socket.getsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
            ) != 0:
                raise ValueError("powered UDP socket unexpectedly permits reuse")
            if endpoint.socket.getsockopt(
                socket.SOL_SOCKET,
                endpoint.exclusive_option,
            ) != 1:
                raise ValueError("powered UDP socket exclusive-use proof changed")
            if not callable(scratch_parser_factory):
                raise TypeError("scratch_parser_factory must be callable")
            if not callable(mavlink_factory):
                raise TypeError("mavlink_factory must be callable")
            if type(outbound_guards) is not PoweredOutboundGuards:
                raise TypeError(
                    "outbound_guards must be exact PoweredOutboundGuards"
                )
            for callback, name in (
                (role_valid, "role_valid"),
                (parent_alive, "parent_alive"),
                (lease_valid, "lease_valid"),
            ):
                if not callable(callback):
                    raise TypeError(f"{name} must be callable")
            if (
                external_cleanup_authorize is not None
                and not callable(external_cleanup_authorize)
            ):
                raise TypeError(
                    "external_cleanup_authorize must be callable or None"
                )

            source_gate = MavlinkSourceFreeze(scratch_parser_factory)
            writer = _FrozenPeerDatagramWriter(
                endpoint,
                source_gate,
                OutboundAuthorityError,
            )
            mavlink = mavlink_factory(writer)
            if not callable(getattr(mavlink, "parse_buffer", None)):
                raise TypeError("production MAVLink object lacks parse_buffer()")
            for name in (
                "set_attitude_target_send",
                "command_long_send",
                "heartbeat_send",
                "timesync_send",
            ):
                if not callable(getattr(mavlink, name, None)):
                    raise TypeError(f"production MAVLink object lacks {name}()")
            if getattr(mavlink, "file", None) is not writer:
                raise ValueError("production MAVLink sender did not retain frozen writer")
            mavlink.robust_parsing = False
            if mavlink.robust_parsing is not False:
                raise ValueError("production MAVLink robust parsing stayed enabled")
        except BaseException:
            try:
                endpoint.close()
            except BaseException as close_exc:
                raise RuntimeError(
                    "partial powered transport construction could not close endpoint"
                ) from close_exc
            raise

        self.endpoint = endpoint
        self.source_gate = source_gate
        self._writer = writer
        self.mavlink = mavlink
        self.outbound_guards = outbound_guards
        self.role_valid = role_valid
        self.parent_alive = parent_alive
        self.lease_valid = lease_valid
        self.external_cleanup_authorize = external_cleanup_authorize
        self._receive_owner: Optional[str] = None
        self._receive_owner_lock = threading.Lock()
        self._receive_call_lock = threading.Lock()
        self.connection = _PoweredConnection(self)

    @classmethod
    def from_pymavlink(
        cls,
        endpoint,
        *,
        outbound_guards,
        role_valid: Callable[[], bool],
        parent_alive: Callable[[], bool],
        lease_valid: Callable[[], bool],
        external_cleanup_authorize: Optional[Callable[..., int]] = None,
    ) -> "PoweredMavlinkTransport":
        """Build distinct scratch/production pymavlink parsers lazily."""

        try:
            from pymavlink import mavutil
        except ImportError as exc:  # pragma: no cover - env-dependent
            try:
                endpoint.close()
            except BaseException as close_exc:
                raise RuntimeError(
                    "missing pymavlink and powered endpoint close failed"
                ) from close_exc
            raise RuntimeError(
                "pymavlink is required for powered AIGP transport"
            ) from exc

        return cls(
            endpoint,
            scratch_parser_factory=lambda: mavutil.mavlink.MAVLink(None),
            mavlink_factory=lambda writer: mavutil.mavlink.MAVLink(
                writer,
                srcSystem=255,
                srcComponent=190,
            ),
            outbound_guards=outbound_guards,
            role_valid=role_valid,
            parent_alive=parent_alive,
            lease_valid=lease_valid,
            external_cleanup_authorize=external_cleanup_authorize,
        )

    @property
    def peer(self) -> tuple[str, int] | None:
        return self.source_gate.peer

    @property
    def promoted(self) -> bool:
        return self.source_gate.promoted

    @property
    def receive_owner(self) -> Optional[str]:
        with self._receive_owner_lock:
            return self._receive_owner

    def claim_receive_owner(self, owner: str) -> None:
        if owner not in {
            POWERED_RECEIVE_OWNER_WORKER,
            POWERED_RECEIVE_OWNER_EXTERNAL_CLEANUP,
        }:
            raise ValueError("powered receive owner is invalid")
        with self._receive_owner_lock:
            if self._receive_owner is not None:
                raise RuntimeError("powered UDP receiver already has an owner")
            if self.endpoint.closed:
                raise RuntimeError("powered UDP endpoint is closed")
            self._receive_owner = owner

    def recvfrom(
        self,
        *,
        owner: str,
        max_wait_ns: Optional[int] = None,
    ) -> Optional[tuple[bytes, Any]]:
        with self._receive_owner_lock:
            if owner != self._receive_owner:
                raise RuntimeError("powered UDP receive ownership is invalid")
        if max_wait_ns is not None and (
            type(max_wait_ns) is not int
            or not 1 <= max_wait_ns <= POWERED_WORKER_POLL_NS
        ):
            raise ValueError(
                "bounded powered receive wait must be 1..50,000,000 ns"
            )
        if not self._receive_call_lock.acquire(blocking=False):
            raise RuntimeError("powered UDP receive call is already active")
        try:
            if max_wait_ns is None:
                received = self.endpoint.socket.recvfrom(
                    POWERED_MAX_DATAGRAM_BYTES
                )
            else:
                gettimeout = getattr(self.endpoint.socket, "gettimeout", None)
                settimeout = getattr(self.endpoint.socket, "settimeout", None)
                if not callable(gettimeout) or not callable(settimeout):
                    raise TypeError(
                        "bounded powered receive requires socket timeout APIs"
                    )
                previous_timeout = gettimeout()
                settimeout(max_wait_ns / 1_000_000_000.0)
                try:
                    received = self.endpoint.socket.recvfrom(
                        POWERED_MAX_DATAGRAM_BYTES
                    )
                except (socket.timeout, TimeoutError):
                    return None
                finally:
                    if not self.endpoint.closed:
                        settimeout(previous_timeout)
            raw, source = received
            if not isinstance(raw, (bytes, bytearray, memoryview)):
                raise TypeError("powered UDP recvfrom payload must be bytes-like")
            return bytes(raw), source
        finally:
            self._receive_call_lock.release()

    def parse_production(self, raw: bytes) -> Any:
        parsed = self.mavlink.parse_buffer(raw)
        if type(parsed) is not list or len(parsed) != 1:
            raise ValueError(
                "production MAVLink parser did not return exactly one message"
            )
        message = parsed[0]
        if message.get_type() == "BAD_DATA":
            raise ValueError("production MAVLink parser returned BAD_DATA")
        if bytes(message.get_msgbuf()) != raw:
            raise ValueError("production MAVLink parser did not preserve datagram")
        return message

    def begin_authorized_write(self) -> None:
        self._writer.begin_authorized_call()

    def finish_authorized_write(self) -> None:
        self._writer.finish_authorized_call(require_write=True)

    def cancel_authorized_write(self) -> None:
        self._writer.cancel_authorized_call()

    def close(self) -> None:
        self.endpoint.close()


class AIGPMavlinkAdapter(CompetitionInterface):
    """CompetitionInterface implementation for the official AIGP sim."""

    def __init__(
        self,
        *,
        enable_vision: bool = True,
        vision_port: int = AIGP_CAM_UDP_PORT,
        require_track: bool = True,
        track_retries: int = 3,
        telemetry_mode: str = "pose",
        fetch_track_on_connect: bool = True,
        ingress_buffer_capacity: int = DEFAULT_INGRESS_BUFFER_CAPACITY,
        outbound_receipt_capacity: int = DEFAULT_OUTBOUND_RECEIPT_CAPACITY,
        host_clock_id: str = VQ2_HOST_CLOCK_ID,
        monotonic_ns: Optional[Callable[[], int]] = None,
        powered_transport: Optional[PoweredMavlinkTransport] = None,
        powered_receive_mode: str = POWERED_RECEIVE_MODE_WORKER,
    ) -> None:
        if (
            powered_transport is not None
            and type(powered_transport) is not PoweredMavlinkTransport
        ):
            raise TypeError(
                "powered_transport must be exact PoweredMavlinkTransport or None"
            )
        if telemetry_mode not in {"pose", "imu"}:
            raise ValueError("telemetry_mode must be 'pose' or 'imu'")
        if powered_receive_mode not in {
            POWERED_RECEIVE_MODE_WORKER,
            POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP,
        }:
            raise ValueError("powered_receive_mode is invalid")
        if (
            powered_transport is None
            and powered_receive_mode != POWERED_RECEIVE_MODE_WORKER
        ):
            raise ValueError(
                "external cleanup receive mode requires powered transport"
            )
        if require_track and not fetch_track_on_connect:
            raise ValueError(
                "require_track=True is incompatible with "
                "fetch_track_on_connect=False"
            )
        if (
            type(ingress_buffer_capacity) is not int
            or ingress_buffer_capacity < 1
        ):
            raise ValueError("ingress_buffer_capacity must be a positive exact integer")
        if (
            type(outbound_receipt_capacity) is not int
            or outbound_receipt_capacity < 1
        ):
            raise ValueError(
                "outbound_receipt_capacity must be a positive exact integer"
            )
        if host_clock_id != VQ2_HOST_CLOCK_ID:
            raise ValueError(
                f"host_clock_id must be exact {VQ2_HOST_CLOCK_ID!r}"
            )
        if monotonic_ns is not None and not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable or None")
        if powered_transport is not None and (
            telemetry_mode != "imu"
            or require_track
            or fetch_track_on_connect
            or powered_transport.endpoint.closed
        ):
            try:
                powered_transport.close()
            except BaseException as close_exc:
                raise RuntimeError(
                    "invalid powered adapter configuration could not close transport"
                ) from close_exc
            raise ValueError(
                "powered transport requires open endpoint, telemetry_mode='imu', "
                "require_track=False, and fetch_track_on_connect=False"
            )
        self.enable_vision = enable_vision
        self.require_track = require_track
        self.track_retries = int(track_retries)
        self.telemetry_mode = telemetry_mode
        self.fetch_track_on_connect = bool(fetch_track_on_connect)
        self.ingress_buffer_capacity = ingress_buffer_capacity
        self.outbound_receipt_capacity = outbound_receipt_capacity
        self.host_clock_id = host_clock_id
        self._monotonic_ns = monotonic_ns or time.perf_counter_ns
        self._powered_transport = powered_transport
        self.powered_receive_mode = powered_receive_mode
        from competition.vq2_capture import (
            ActuatorOutputStatusPayloadV1,
            AttitudeTargetOutboundV1,
            AttitudeTargetWireV1,
            CommandLongWireV1,
            GCSHeartbeatWireV1,
            HeartbeatPayloadV1,
            MavlinkIngressV1,
            NonAttitudeOutboundV1,
            RaceStatusPayloadV1,
            ReceivedActuatorOutputStatusV1,
            ReceivedHeartbeatV1,
            ReceivedIMUSampleV1,
            ReceivedRaceStatusV1,
            TimesyncWireV1,
        )

        self._actuator_payload_type = ActuatorOutputStatusPayloadV1
        self._attitude_outbound_type = AttitudeTargetOutboundV1
        self._attitude_wire_type = AttitudeTargetWireV1
        self._command_long_wire_type = CommandLongWireV1
        self._gcs_heartbeat_wire_type = GCSHeartbeatWireV1
        self._heartbeat_payload_type = HeartbeatPayloadV1
        self._mavlink_ingress_type = MavlinkIngressV1
        self._nonattitude_outbound_type = NonAttitudeOutboundV1
        self._race_status_payload_type = RaceStatusPayloadV1
        self._received_actuator_type = ReceivedActuatorOutputStatusV1
        self._received_heartbeat_type = ReceivedHeartbeatV1
        self._received_imu_type = ReceivedIMUSampleV1
        self._received_race_status_type = ReceivedRaceStatusV1
        self._timesync_wire_type = TimesyncWireV1

        self._conn = None
        self._target_system = 1
        self._target_component = 1
        self._send_lock = threading.Lock()
        self._ingress_dispatch_lock = threading.Lock()
        self._audit_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_event = threading.Event()
        self._telemetry_ready_event = threading.Event()
        self._track_event = threading.Event()
        self._powered_promotion_event = threading.Event()
        self._powered_failure_event = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._announce_thread: Optional[threading.Thread] = None
        self._powered_connect_deadline_monotonic_ns: Optional[int] = None
        self._powered_failure_reason: Optional[str] = None
        self._vision: Optional[VisionUdpListener] = (
            VisionUdpListener(port=vision_port) if enable_vision else None
        )

        self._latest_telem: Optional[TelemetryState] = None
        self._race_status: Optional[RaceStatus] = None
        self._track_data: Optional[TrackData] = None
        self._imu_samples: Deque[ReceivedIMUSampleV1] = deque(
            maxlen=ingress_buffer_capacity
        )
        self._mavlink_arrivals: Deque[
            ReceivedHeartbeatV1
            | ReceivedRaceStatusV1
            | ReceivedActuatorOutputStatusV1
        ] = deque(
            maxlen=ingress_buffer_capacity
        )
        self._latest_received_heartbeat: Optional[ReceivedHeartbeatV1] = None
        self._latest_received_race_status: Optional[ReceivedRaceStatusV1] = None
        self._latest_received_actuator_output_status: Optional[
            ReceivedActuatorOutputStatusV1
        ] = None
        self._collisions: Deque[Dict] = deque(
            maxlen=DEFAULT_COLLISION_BUFFER_CAPACITY
        )
        self._reset_persistence_failures: Deque[
            CalibrationResetPersistenceFailureV1
        ] = deque(maxlen=DEFAULT_RESET_PERSISTENCE_FAILURE_CAPACITY)
        self._reset_persistence_failures_dropped = 0
        self._collision_handled = 0
        self._collision_dropped = 0
        self._collision_high_watermark = 0
        self._actuator_outputs: Optional[Dict] = None
        self._indi_debug: Optional[Dict] = None
        self._reassembler = TrackInfoReassembler()
        # Diagnostics for the DSQ investigation (iter-39): the sim announces
        # human-readable verdicts (e.g. disqualification) via STATUSTEXT, which
        # we previously dropped silently. Capture them, and log every OTHER
        # message type the first time we see it so a DSQ on any unexpected
        # channel becomes visible instead of invisible.
        self._status_texts: Deque[Dict] = deque(maxlen=256)
        self._seen_msg_types: set = set()

        self._last_heartbeat_monotonic = 0.0
        self._heartbeat_sequence = 0
        self._last_imu_monotonic = 0.0
        self._last_race_status_monotonic = 0.0
        self._last_actuator_monotonic = 0.0
        self._ingress_generation = 0
        self._ingress_next_sequence = 0
        self._ingress_counts = {
            "HIGHRES_IMU": 0,
            "HEARTBEAT": 0,
            "RACE_STATUS": 0,
            "ACTUATOR_OUTPUT_STATUS": 0,
        }
        self._ingress_dropped = 0
        self._ingress_high_watermark = 0
        self._imu_ingress_dropped = 0
        self._other_ingress_dropped = 0
        self._imu_ingress_high_watermark = 0
        self._other_ingress_high_watermark = 0
        self._outbound_counts = {
            "timesync": 0,
            "gcs_heartbeat": 0,
            "sim_reset": 0,
            "arm": 0,
            "disarm": 0,
            "attitude_target": 0,
            "position_target": 0,
            "other_command": 0,
        }
        self._outbound_receipts: Deque[
            AttitudeTargetOutboundV1 | NonAttitudeOutboundV1
        ] = deque(maxlen=outbound_receipt_capacity)
        self._outbound_generation = 0
        self._outbound_next_sequence = 0
        self._outbound_returned = 0
        self._outbound_raised = 0
        self._outbound_dropped = 0
        self._outbound_high_watermark = 0
        self._armed = False
        self._have_attitude = False
        self._have_lpn = False
        self._have_odometry = False
        self._have_imu = False

        # The live AIGP sim MISHANDLES SET_ATTITUDE_TARGET attitude mode
        # (type_mask 0b111): a held level attitude makes the drone spin up to
        # ~9 rad/s the moment it is airborne (bench-confirmed 2026-06-13,
        # scripts/aigp_bench.py). It DOES honor body-rate mode (mask 128).
        # So `send_attitude` converts the commanded attitude into a body-rate
        # setpoint via a quaternion attitude-error P loop and sends rate mode.
        # Off-switch + gains exposed for tuning / fallback.
        self._use_rate_control = True
        # Inner attitude->rate loop gains, RE-tuned 2026-06-13 (iter-022) after
        # an isolation bench overturned the prior tuning. Key findings
        # (scripts/aigp_bench.py): (1) pure zero body-rate is perfectly clean
        # (gyro p95~0); (2) the old (2.0,2.5,1.0) loop LIMIT-CYCLES at ~9 Hz
        # (gyro p95~4.5) and the jitter rectifies thrust into a runaway climb
        # — this was THE flight failure, not the trajectory; (3) per-axis rate
        # ID shows the sim sign-flips all axes (so _rate_sign=-1 is right) but
        # the gain is axis-dependent (~1.0 roll, ~2.1 pitch/yaw), NOT a uniform
        # 2.5x. The old loop gain (2*kp * ~2.1 amp = ~8.4) was far past the
        # delay-limited stability margin -> the limit cycle. Cutting the gain
        # ~4x removes the oscillation AND the climb while still tracking a 0.3
        # rad roll step cleanly (gyro p95<0.6, no flip). kd is the SAME sign as
        # before (genuine damping: the sim's flip applies to the kd term too,
        # so it stays negative feedback) — just much smaller.
        # PER-AXIS body-rate P/D gains (roll, pitch, yaw). The sim amplifies
        # the rate channels asymmetrically (~1.0x roll vs ~2.1x pitch/yaw,
        # bench rate-ID), so a uniform kp=0.5 leaves ROLL at HALF the
        # closed-loop bandwidth of pitch — roll under-tracks (0.46x amplitude,
        # ~0.6s lag, captures min_v28) and the cross-track centering oscillates
        # and clips frames at cruise >=5. iter-32: raise ROLL only to kp=1.0
        # (effective gain ~1.0, matching pitch's proven-safe ~1.05) with kd=0.4
        # to damp the now-faster roll loop; pitch/yaw unchanged (eff ~1.05,
        # already crisp+stable). Watch gyro p95 (<1.0 clean, abort >2.0).
        # iter-44 TESTED LIVE & FALSIFIED — KEEP 1.0/0.4/0.8. To make the fast
        # gate5 turn (achieved roll only 0.53x commanded), raising the ROLL loop
        # to kp 1.3 / kd 0.55 / rate-clamp 1.1 made tracking WORSE, not better:
        # achieved/cmd roll fell 0.53 -> 0.33, gyro p95 rose 1.0 -> 1.62, and it
        # nearly clipped gate0 (0.049 m). The higher kp+clamp under-damped the
        # cascade into oscillation (the 9 Hz limit cycle the kd term suppresses;
        # the adversarial workflow predicted exactly this). The 0.53 roll
        # attenuation is the sim's INHERENT behaviour — fighting it with gains
        # destabilises. The fast-turn cap must be solved by reducing the REQUIRED
        # turn (racing line / variable speed), not by more inner-loop bandwidth.
        self._att_rate_kp = (1.0, 0.5, 0.5)   # (roll, pitch, yaw)
        self._att_rate_kd = (0.4, 0.2, 0.2)
        # iter-51 TESTED & NEUTRAL — keep 0.8. Raising ONLY the roll rate clamp
        # to 1.0 (kp/kd unchanged) is SAFE (gyro p95/max unchanged at 1.16/2.13,
        # NO limit cycle — so the iter-44 limit cycle was the kp, not the clamp),
        # but it does NOT help: the roll rate rarely reaches 0.8 on the slalom
        # turns, so the lateral undershoot (the 50-km/h clearance limiter at
        # gates 2/3) is TILT/attenuation-limited, not rate-limited. Reverted.
        self._att_rate_max = 0.8      # rad/s clamp per axis
        # Per-axis CLOSED-LOOP sign, corrected 2026-06-13 (iter-023). The
        # open-loop gyro probe suggested all three axes were sign-flipped, but
        # that conflated the body-rate actuator sign with the spawn yaw=pi
        # frame rotation and the euler-rate mapping. The CLOSED loop is the
        # ground truth, and it disagrees per axis:
        #   * ROLL  (-1): bench att-hold commanding roll +0.30 drove measured
        #     roll to +0.26 — converges. Correct.
        #   * PITCH (+1): commanding pitch -0.50 drove measured pitch the WRONG
        #     way (-0.31 -> -0.08), and in flight pitch -0.62 diverged to +1.5
        #     (positive feedback) — the cause of "hover-stable, flips the moment
        #     it translates". Un-flip pitch so the loop is negative feedback.
        #   * YAW   (-1): held cleanly at pi in every run (never excited with a
        #     real error); left at -1. Revisit if yaw drifts after this fix.
        # Two independent Opus reviews + the bench/flight captures all point to
        # the pitch axis being the single inverted sign.
        self._rate_sign = (-1.0, 1.0, -1.0)
        if self.telemetry_mode == "imu":
            # VQ2 build 3385 live rate-ID (2026-07-18) differs from the older
            # pose-enabled build on PITCH.  At 0.24 thrust, desired +0.10 with
            # wire +0.10 produced measured q=-0.19 rad/s; wire -0.10 is
            # therefore required.  Roll retained the proven -1 mapping.
            # Yaw remains deliberately unexcited by the VQ2 runner.
            self._rate_sign = (-1.0, -1.0, -1.0)

        # --- OPT-IN measured-accel INDI inner loop (roadmap #2) -------------
        # OFF by default. When _use_indi is True, send_attitude computes the
        # body-rate setpoint via control.indi_inner_loop.IndiInnerLoop (filtered
        # gyro-derivative inversion + online-G) INSTEAD of the PD law in
        # _attitude_error_body_rates. It STILL applies self._rate_sign and sends
        # rates mode exactly as the PD path, so the only difference is how the
        # rate vector is produced. When False, the code path below is unchanged
        # (byte-identical to the validated champion PD path). The INDI object is
        # lazily built on first use so importing this module never requires the
        # control package. See the module docstring for the discriminator
        # read-out ("recovered => mismatch; still clamped => bandwidth limit").
        self._use_indi = False
        self._indi_config = None  # optional control.indi_inner_loop.IndiConfig
        self._indi = None         # lazily-built IndiInnerLoop
        self._indi_last_t_us: Optional[int] = None

    def _ensure_indi(self):
        """Lazily construct the IndiInnerLoop (kept out of __init__ so importing
        this module does not pull in the control package)."""
        if self._indi is None:
            from control.indi_inner_loop import IndiInnerLoop
            # Default the INDI rate clamp to the SAME envelope as the PD path
            # (_att_rate_max) so the opt-in branch never commands outside the
            # validated rate range, unless the caller supplied an explicit cfg.
            cfg = self._indi_config
            if cfg is None:
                from control.indi_inner_loop import IndiConfig
                # Gentler accel-PD than the module default (18/3): real-sim
                # iters 3-4 showed the hot default slams the rate clamp on the
                # large startup attitude error and winds up -> divergence. Plus
                # the physical gyro-derivative clamp (max_ang_accel) that kills
                # the telem-rate spikes. Tuned against the live sim, not offline.
                cfg = IndiConfig(
                    max_rate=self._att_rate_max,
                    kp_att=(6.0, 6.0, 4.0), kd_att=(2.5, 2.5, 2.0),
                    max_ang_accel=30.0,
                )
            self._indi = IndiInnerLoop(cfg)
        return self._indi

    async def connect(
        self,
        address: str = DEFAULT_MAVLINK_URL,
        *,
        deadline_monotonic_ns: Optional[int] = None,
    ) -> None:
        """Open the UDP MAVLink socket, announce as GCS, then fetch track data.

        The RX thread starts before any heartbeat wait so a connect-time
        track transfer cannot be discarded by ``wait_heartbeat()``.

        Idempotent: if already connected, returns immediately without spawning
        duplicate threads or re-fetching track data. ``RaceSession`` calls
        ``connect()`` unconditionally; the runner calls it first with the
        correct address, so this guard is required.
        """
        if self._conn is not None:
            return
        if self._powered_transport is None:
            if deadline_monotonic_ns is not None:
                raise ValueError(
                    "deadline_monotonic_ns is supported only by powered transport"
                )
        else:
            try:
                if address != DEFAULT_MAVLINK_URL:
                    raise ValueError(
                        "powered transport uses its caller-bound endpoint, not address"
                    )
                self._validate_powered_deadline(
                    deadline_monotonic_ns,
                    "connect",
                )
            except BaseException:
                self._powered_transport.outbound_guards.latch_production(
                    "powered_connect_contract_invalid"
                )
                try:
                    self._powered_transport.close()
                except BaseException as close_exc:
                    raise RuntimeError(
                        "invalid powered connect could not close endpoint"
                    ) from close_exc
                raise

        # A new socket is a new simulator epoch.  Never let reconnect reuse
        # readiness events, pose/IMU snapshots, race status, or decoded vision
        # from the prior connection.
        self._heartbeat_event.clear()
        self._telemetry_ready_event.clear()
        self._track_event.clear()
        self._last_heartbeat_monotonic = 0.0
        self._heartbeat_sequence = 0
        self._last_imu_monotonic = 0.0
        self._last_race_status_monotonic = 0.0
        self._last_actuator_monotonic = 0.0
        self._armed = False
        self._have_attitude = False
        self._have_lpn = False
        self._have_odometry = False
        self._have_imu = False
        with self._state_lock:
            self._latest_telem = None
            self._race_status = None
            self._track_data = None
            self._actuator_outputs = None
            self._latest_received_heartbeat = None
            self._latest_received_race_status = None
            self._latest_received_actuator_output_status = None
            self._begin_ingress_generation_locked()
            self._reset_collision_generation_locked()
            self._reset_persistence_failures.clear()
            self._reset_persistence_failures_dropped = 0
            generation = self._ingress_generation
        with self._audit_lock:
            for name in self._outbound_counts:
                self._outbound_counts[name] = 0
            self._begin_outbound_generation_locked(
                generation,
                initialize_attempt=True,
            )
        if self._vision is not None:
            self._vision.reset()

        if self._powered_transport is not None:
            await self._connect_powered(deadline_monotonic_ns)
            return

        try:
            from pymavlink import mavutil
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError("pymavlink is required for live AIGP transport") from exc

        self._conn = mavutil.mavlink_connection(address)
        self._target_system = getattr(self._conn, "target_system", 1) or 1
        self._target_component = getattr(self._conn, "target_component", 1) or 1
        self._stop_event.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="aigp-mavlink-rx", daemon=True)
        self._announce_thread = threading.Thread(
            target=self._announce_loop,
            name="aigp-mavlink-announce",
            daemon=True,
        )
        self._rx_thread.start()
        self._announce_thread.start()

        heartbeat_ok = await asyncio.to_thread(self._heartbeat_event.wait, 10.0)
        if not heartbeat_ok:
            raise ConnectionError("No AIGP heartbeat received. Is the sim in Virtual Qualifier mode?")
        telemetry_ok = await asyncio.to_thread(self._telemetry_ready_event.wait, 10.0)
        if not telemetry_ok:
            raise ConnectionError("AIGP telemetry did not become ready")

        if self._vision is not None:
            try:
                await self._vision.start()
            except OSError:
                logger.exception("Could not start AIGP vision listener")

        if self.fetch_track_on_connect and self._track_data is None:
            for _ in range(max(1, self.track_retries)):
                await self._send_sim_reset(clear_track_event=True)
                if await asyncio.to_thread(self._track_event.wait, 5.0):
                    break
            if self._track_data is None and self.require_track:
                raise ConnectionError("AIGP track data not received after SIM_RESET")
            if self._track_data is None:
                logger.warning("AIGP track data not received after SIM_RESET")

    async def disconnect(
        self,
        *,
        deadline_monotonic_ns: Optional[int] = None,
        powered_progress: Optional[Callable[[], None]] = None,
    ) -> None:
        if self._powered_transport is not None:
            deadline = self._validate_powered_deadline(
                deadline_monotonic_ns,
                "disconnect",
                allow_reached=True,
            )
            if powered_progress is not None and not callable(powered_progress):
                progress_contract_error: Optional[BaseException] = TypeError(
                    "powered_progress must be callable or None"
                )
                self._powered_transport.outbound_guards.latch_production(
                    "powered_progress_invalid"
                )
                powered_progress = None
            else:
                progress_contract_error = None
            failures, progress_error = self._close_powered_transport_and_join(
                deadline,
                powered_progress=powered_progress,
            )
            if progress_error is None:
                progress_error = progress_contract_error
            if self._vision is not None:
                if progress_error is None and powered_progress is not None:
                    try:
                        powered_progress()
                    except BaseException as exc:
                        self._powered_transport.outbound_guards.latch_production(
                            "powered_progress_failed"
                        )
                        progress_error = exc
                try:
                    now = self._read_monotonic_ns()
                    remaining = max(0.0, (deadline - now) / 1_000_000_000.0)
                    if remaining <= 0.0:
                        raise TimeoutError("disconnect deadline reached before vision stop")
                    await asyncio.wait_for(self._vision.stop(), timeout=remaining)
                except BaseException as exc:
                    failures.append(
                        "powered vision termination failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                if progress_error is None and powered_progress is not None:
                    try:
                        powered_progress()
                    except BaseException as exc:
                        self._powered_transport.outbound_guards.latch_production(
                            "powered_progress_failed"
                        )
                        progress_error = exc
            if progress_error is not None:
                if failures:
                    add_note = getattr(progress_error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "powered disconnect also reported: "
                            + "; ".join(failures)
                        )
                raise progress_error
            if failures:
                raise RuntimeError("; ".join(failures))
            return

        if deadline_monotonic_ns is not None or powered_progress is not None:
            raise ValueError(
                "powered disconnect options require powered transport"
            )
        self._stop_event.set()
        threads = tuple(
            thread
            for thread in (self._rx_thread, self._announce_thread)
            if thread is not None
        )
        for thread in threads:
            thread.join(timeout=2.0)
        close_error: Optional[BaseException] = None
        if self._conn is not None:
            close = getattr(self._conn, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as exc:
                    close_error = exc
        # Closing the transport is a second unblock for a receiver that did
        # not leave recv_match during the first bounded join.
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        alive = [thread.name for thread in threads if thread.is_alive()]
        if not alive:
            self._rx_thread = None
            self._announce_thread = None
        if close_error is None:
            self._conn = None
        failures = []
        if alive:
            failures.append(
                "MAVLink worker termination unproved: " + ", ".join(alive)
            )
        if close_error is not None:
            failures.append(
                "MAVLink transport close failed: "
                f"{type(close_error).__name__}: {close_error}"
            )
        if self._vision is not None:
            try:
                await self._vision.stop()
            except BaseException as exc:
                failures.append(
                    "legacy vision termination failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        if failures:
            raise RuntimeError("; ".join(failures))

    def _validate_powered_deadline(
        self,
        value: Optional[int],
        phase: str,
        *,
        allow_reached: bool = False,
    ) -> int:
        from scripts.aigp_vq2_powered_runtime import PoweredDeadlineExpired

        if type(value) is not int or value < 1:
            raise ValueError(
                f"powered {phase} deadline must be a positive exact integer"
            )
        now = self._read_monotonic_ns()
        if now >= value and not allow_reached:
            raise PoweredDeadlineExpired(
                f"powered {phase} absolute deadline was already reached"
            )
        return value

    def _powered_callback_bool(self, callback: Callable[[], bool], label: str) -> bool:
        from scripts.aigp_vq2_powered_runtime import OutboundAuthorityError

        try:
            value = callback()
        except BaseException as exc:
            self._powered_transport.outbound_guards.latch_production(
                f"{label}_callback_raised"
            )
            raise OutboundAuthorityError(f"{label} callback raised") from exc
        if type(value) is not bool:
            self._powered_transport.outbound_guards.latch_production(
                f"{label}_callback_invalid"
            )
            raise OutboundAuthorityError(
                f"{label} callback must return an exact boolean"
            )
        return value

    async def _connect_powered(
        self,
        deadline_monotonic_ns: Optional[int],
    ) -> None:
        from scripts.aigp_vq2_powered_runtime import (
            OutboundAuthorityError,
            PoweredRuntimeError,
        )

        transport = self._powered_transport
        assert deadline_monotonic_ns is not None
        deadline = deadline_monotonic_ns
        self._powered_promotion_event.clear()
        self._powered_failure_event.clear()
        self._powered_failure_reason = None
        self._powered_connect_deadline_monotonic_ns = deadline
        try:
            if self._read_monotonic_ns() >= deadline:
                from scripts.aigp_vq2_powered_runtime import PoweredDeadlineExpired

                raise PoweredDeadlineExpired(
                    "powered connect deadline reached before worker start"
                )
            if transport.endpoint.closed or transport.peer is not None:
                raise PoweredRuntimeError(
                    "powered transport is single-use and unavailable"
                )
            if self.powered_receive_mode == POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP:
                if not transport.outbound_guards.production_latched:
                    raise OutboundAuthorityError(
                        "external cleanup requires production to be latched"
                    )
                if not callable(transport.external_cleanup_authorize):
                    raise OutboundAuthorityError(
                        "external cleanup announcement authority is unavailable"
                    )
                cleanup_parent_alive = self._powered_callback_bool(
                    transport.parent_alive,
                    "parent_alive",
                )
                if (
                    not cleanup_parent_alive
                    and transport.outbound_guards.cleanup_state
                    != "takeover_pending"
                ):
                    transport.outbound_guards.note_parent_death()
                    raise OutboundAuthorityError(
                        "wrapper parent is not live at cleanup connect"
                    )
                if not self._powered_callback_bool(
                    transport.lease_valid,
                    "lease_valid",
                ):
                    raise OutboundAuthorityError(
                        "cleanup lease lineage is invalid"
                    )
                transport.claim_receive_owner(
                    POWERED_RECEIVE_OWNER_EXTERNAL_CLEANUP
                )
                self._conn = transport.connection
                self._target_system = 1
                self._target_component = 1
                self._stop_event.clear()
                return

            transport.outbound_guards.enable_production()
            if not self._powered_callback_bool(transport.role_valid, "role_valid"):
                transport.outbound_guards.latch_production("production_role_invalid")
                raise OutboundAuthorityError("production role is invalid")
            if not self._powered_callback_bool(transport.parent_alive, "parent_alive"):
                transport.outbound_guards.note_parent_death()
                raise OutboundAuthorityError("wrapper parent is not live")
            if not self._powered_callback_bool(transport.lease_valid, "lease_valid"):
                transport.outbound_guards.latch_production("production_lease_invalid")
                raise OutboundAuthorityError("production lease lineage is invalid")

            self._conn = transport.connection
            self._target_system = 1
            self._target_component = 1
            self._stop_event.clear()
            transport.claim_receive_owner(POWERED_RECEIVE_OWNER_WORKER)
            self._rx_thread = threading.Thread(
                target=self._powered_rx_loop,
                name="aigp-powered-mavlink-rx",
                daemon=True,
            )
            self._announce_thread = threading.Thread(
                target=self._announce_loop,
                name="aigp-powered-mavlink-announce",
                daemon=True,
            )
            self._rx_thread.start()
            self._announce_thread.start()
            await asyncio.to_thread(self._wait_for_powered_promotion, deadline)

            announce = self._announce_thread
            if announce is not None:
                now = self._read_monotonic_ns()
                announce.join(timeout=max(0.0, (deadline - now) / 1_000_000_000.0))
                if announce.is_alive():
                    transport.outbound_guards.latch_production(
                        "announcement_worker_unproved"
                    )
                    raise PoweredRuntimeError(
                        "announcement worker did not stop before connect deadline"
                    )
                self._announce_thread = None

            if self._vision is not None:
                now = self._read_monotonic_ns()
                remaining = max(0.0, (deadline - now) / 1_000_000_000.0)
                if remaining <= 0.0:
                    raise PoweredRuntimeError(
                        "connect deadline reached before vision start"
                    )
                await asyncio.wait_for(self._vision.start(), timeout=remaining)
        except BaseException as exc:
            transport.outbound_guards.latch_production("powered_connect_failed")
            failures, _progress_error = self._close_powered_transport_and_join(
                deadline
            )
            if failures:
                raise RuntimeError(
                    f"powered connect failed and cleanup was unproved: {'; '.join(failures)}"
                ) from exc
            raise

    def _wait_for_powered_promotion(self, deadline_monotonic_ns: int) -> None:
        from scripts.aigp_vq2_powered_runtime import (
            OutboundAuthorityError,
            PoweredDeadlineExpired,
            PoweredRuntimeError,
        )

        transport = self._powered_transport
        while True:
            if transport.promoted:
                return
            if self._powered_failure_event.is_set():
                raise PoweredRuntimeError(
                    self._powered_failure_reason or "powered receiver failed"
                )
            if not self._powered_callback_bool(transport.parent_alive, "parent_alive"):
                transport.outbound_guards.note_parent_death()
                raise OutboundAuthorityError("wrapper parent died before promotion")
            now = self._read_monotonic_ns()
            if now >= deadline_monotonic_ns:
                transport.outbound_guards.latch_production(
                    "source_promotion_deadline_reached"
                )
                raise PoweredDeadlineExpired(
                    "MAVLink source promotion deadline was reached"
                )
            wait_ns = min(
                POWERED_WORKER_POLL_NS,
                deadline_monotonic_ns - now,
            )
            self._powered_promotion_event.wait(wait_ns / 1_000_000_000.0)

    def _close_powered_transport_and_join(
        self,
        deadline_monotonic_ns: int,
        *,
        powered_progress: Optional[Callable[[], None]] = None,
    ) -> tuple[list[str], Optional[BaseException]]:
        self._stop_event.set()
        failures: list[str] = []
        progress_error: Optional[BaseException] = None
        try:
            self._powered_transport.close()
        except BaseException as exc:
            failures.append(
                "powered UDP close failed: "
                f"{type(exc).__name__}: {exc}"
            )
        threads = tuple(
            thread
            for thread in (self._rx_thread, self._announce_thread)
            if thread is not None
        )
        accounted_now: Optional[int] = None
        for thread in threads:
            while thread.is_alive():
                observed_now = self._read_monotonic_ns()
                now = (
                    observed_now
                    if accounted_now is None
                    else max(accounted_now, observed_now)
                )
                if now >= deadline_monotonic_ns:
                    break
                if powered_progress is not None and progress_error is None:
                    try:
                        powered_progress()
                    except BaseException as exc:
                        self._powered_transport.outbound_guards.latch_production(
                            "powered_progress_failed"
                        )
                        progress_error = exc
                    observed_now = self._read_monotonic_ns()
                    now = (
                        observed_now
                        if accounted_now is None
                        else max(accounted_now, observed_now)
                    )
                    if now >= deadline_monotonic_ns:
                        break
                wait_ns = min(
                    POWERED_WORKER_POLL_NS,
                    deadline_monotonic_ns - now,
                )
                thread.join(timeout=wait_ns / 1_000_000_000.0)
                after_join = self._read_monotonic_ns()
                elapsed_ns = max(0, after_join - now)
                if thread.is_alive() and elapsed_ns < wait_ns:
                    # ``Thread.join`` is specified to wait until timeout, but
                    # injected/foreign handle seams may return spuriously.
                    # Avoid a CPU spin while preserving the same fixed slice.
                    threading.Event().wait(
                        (wait_ns - elapsed_ns) / 1_000_000_000.0
                    )
                    accounted_now = now + wait_ns
                else:
                    accounted_now = (
                        after_join
                        if accounted_now is None
                        else max(accounted_now, after_join)
                    )
        if powered_progress is not None and progress_error is None:
            try:
                powered_progress()
            except BaseException as exc:
                self._powered_transport.outbound_guards.latch_production(
                    "powered_progress_failed"
                )
                progress_error = exc
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            failures.append(
                "powered MAVLink worker termination unproved: " + ", ".join(alive)
            )
        else:
            self._rx_thread = None
            self._announce_thread = None
            self._conn = None
        return failures, progress_error

    async def arm(
        self,
        *,
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
    ) -> None:
        if self._armed:
            return
        self._require_conn()
        wire = self._command_long_wire_type(
            target_system=self._target_system,
            target_component=self._target_component,
            command=MAV_CMD_COMPONENT_ARM_DISARM,
            confirmation=0,
            params=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        with self._send_lock:
            self._call_nonattitude_locked(
                category="arm",
                api="command_long_send",
                audit_name="arm",
                wire=wire,
                call=lambda: self._conn.mav.command_long_send(
                    wire.target_system,
                    wire.target_component,
                    wire.command,
                    wire.confirmation,
                    *wire.params,
                ),
                powered_deadline_monotonic_ns=powered_deadline_monotonic_ns,
                powered_cleanup=powered_cleanup,
            )
        if not self._armed:
            logger.warning("Arm command sent but vehicle still reports disarmed")

    async def disarm(
        self,
        *,
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
    ) -> None:
        """Disarm the simulated vehicle as an emergency reset fallback."""

        self._require_conn()
        wire = self._command_long_wire_type(
            target_system=self._target_system,
            target_component=self._target_component,
            command=MAV_CMD_COMPONENT_ARM_DISARM,
            confirmation=0,
            params=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        with self._send_lock:
            self._call_nonattitude_locked(
                category="disarm",
                api="command_long_send",
                audit_name="disarm",
                wire=wire,
                call=lambda: self._conn.mav.command_long_send(
                    wire.target_system,
                    wire.target_component,
                    wire.command,
                    wire.confirmation,
                    *wire.params,
                ),
                powered_deadline_monotonic_ns=powered_deadline_monotonic_ns,
                powered_cleanup=powered_cleanup,
            )

    async def start_offboard(self) -> None:
        """No-op: the sim accepts setpoints without a PX4 offboard handshake."""

    async def stop_offboard(self) -> None:
        """No-op: the sim accepts setpoints without a PX4 offboard handshake."""

    async def get_telemetry(self) -> TelemetryState:
        with self._state_lock:
            if self._latest_telem is None:
                raise RuntimeError("No telemetry received yet")
            return self._latest_telem

    async def get_camera_frame(self) -> Optional[CameraFrame]:
        return self._vision.latest_frame() if self._vision is not None else None

    async def send_attitude(
        self,
        cmd: AttitudeCommand,
        *,
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
    ) -> None:
        if self.telemetry_mode == "imu":
            raise RuntimeError(
                "send_attitude() requires pose telemetry; VQ2 IMU mode must "
                "use an external estimator with send_attitude_rate()"
            )
        self._require_conn()
        thrust = _clamp_thrust(cmd.thrust)
        q = Quaternion.from_euler(cmd.roll_rad, cmd.pitch_rad, cmd.yaw_rad)
        # Canonicalize: MAVLink requires w >= 0 (positive scalar convention).
        # Double-cover means q and -q represent the same rotation, but a
        # negative w component can cause sim-side interpolation glitches.
        if q.w < 0:
            q = Quaternion(-q.w, -q.x, -q.y, -q.z)

        # The sim mishandles attitude mode (it spins) — convert the desired
        # attitude into a body-rate command it DOES honor. See __init__.
        if self._use_rate_control:
            with self._state_lock:
                telem = self._latest_telem
            q_cur = telem.orientation if (telem and telem.orientation) else Quaternion()
            omega = (telem.angular_velocity if (telem and telem.angular_velocity)
                     else (0.0, 0.0, 0.0))
            if self._use_indi:
                # OPT-IN: measured-accel INDI inner loop. dt from telemetry
                # timestamps (us). Produces the SAME (roll,pitch,yaw) rate
                # setpoint contract as the PD law; sign + send below are
                # identical. See control/indi_inner_loop.py.
                t_us = telem.timestamp_us if telem else None
                if t_us is not None and self._indi_last_t_us is not None:
                    dt = (t_us - self._indi_last_t_us) * 1e-6
                else:
                    dt = 0.0  # first tick (or no stamp): INDI guard holds command
                self._indi_last_t_us = t_us
                indi = self._ensure_indi()
                rr, pr, yr = indi.compute(q_cur, q, omega=omega, dt=dt)
                with self._state_lock:
                    self._indi_debug = indi.debug_dict()
            else:
                # Desired body rate in our FRD convention (kp on attitude error,
                # kd damping on measured gyro — gyro is FRD-consistent).
                rr, pr, yr = _attitude_error_body_rates(
                    q_cur, q, omega=omega, kp=self._att_rate_kp,
                    kd=self._att_rate_kd, max_rate=self._att_rate_max,
                )
            sx, sy, sz = self._rate_sign  # sim applies rates with opposite sign
            wire = self._attitude_wire_type(
                time_boot_ms=self._time_boot_ms(),
                target_system=self._target_system,
                target_component=self._target_component,
                type_mask=SET_ATTITUDE_TARGET_MASK_RATES_THRUST,
                q_wxyz=(1.0, 0.0, 0.0, 0.0),
                body_rates_rad_s=(sx * rr, sy * pr, sz * yr),
                thrust=thrust,
            )
            with self._send_lock:
                self._call_attitude_target_locked(
                    api="send_attitude_rate_from_attitude",
                    wire=wire,
                    powered_deadline_monotonic_ns=powered_deadline_monotonic_ns,
                    powered_cleanup=powered_cleanup,
                    powered_exact_zero=False if powered_cleanup else None,
                )
            return

        wire = self._attitude_wire_type(
            time_boot_ms=self._time_boot_ms(),
            target_system=self._target_system,
            target_component=self._target_component,
            type_mask=SET_ATTITUDE_TARGET_MASK_ATTITUDE_THRUST,
            q_wxyz=(q.w, q.x, q.y, q.z),
            body_rates_rad_s=(0.0, 0.0, 0.0),
            thrust=thrust,
        )
        with self._send_lock:
            self._call_attitude_target_locked(
                api="send_attitude_quaternion",
                wire=wire,
                powered_deadline_monotonic_ns=powered_deadline_monotonic_ns,
                powered_cleanup=powered_cleanup,
                powered_exact_zero=False if powered_cleanup else None,
            )

    async def send_attitude_rate(
        self,
        cmd: AttitudeRateCommand,
        *,
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
        call_start_not_before_monotonic_ns: Optional[int] = None,
        call_start_deadline_monotonic_ns: Optional[int] = None,
    ) -> None:
        self._require_conn()
        if not all(math.isfinite(value) for value in (
            cmd.roll_rate,
            cmd.pitch_rate,
            cmd.yaw_rate,
        )):
            raise ValueError("body rates must be finite")
        thrust = _clamp_thrust(cmd.thrust)
        sx, sy, sz = self._rate_sign  # sim applies body rates with opposite sign
        wire = self._attitude_wire_type(
            time_boot_ms=self._time_boot_ms(),
            target_system=self._target_system,
            target_component=self._target_component,
            type_mask=SET_ATTITUDE_TARGET_MASK_RATES_THRUST,
            q_wxyz=(1.0, 0.0, 0.0, 0.0),
            body_rates_rad_s=(
                sx * cmd.roll_rate,
                sy * cmd.pitch_rate,
                sz * cmd.yaw_rate,
            ),
            thrust=thrust,
        )
        powered_exact_zero = None
        if self._powered_transport is not None and powered_cleanup:
            from scripts.aigp_vq2_powered_runtime import exact_zero_rate_thrust

            powered_exact_zero = exact_zero_rate_thrust(
                {
                    "roll_rate_rad_s": cmd.roll_rate,
                    "pitch_rate_rad_s": cmd.pitch_rate,
                    "yaw_rate_rad_s": cmd.yaw_rate,
                    "thrust": cmd.thrust,
                }
            )
        with self._send_lock:
            self._call_attitude_target_locked(
                api="send_attitude_rate",
                wire=wire,
                powered_deadline_monotonic_ns=powered_deadline_monotonic_ns,
                powered_cleanup=powered_cleanup,
                powered_exact_zero=powered_exact_zero,
                call_start_not_before_monotonic_ns=(
                    call_start_not_before_monotonic_ns
                ),
                call_start_deadline_monotonic_ns=(
                    call_start_deadline_monotonic_ns
                ),
            )

    async def send_attitude_rate_if_race_active(
        self,
        cmd: AttitudeRateCommand,
        *,
        expected_active_gate_index: int,
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
        call_start_not_before_monotonic_ns: Optional[int] = None,
        call_start_deadline_monotonic_ns: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Atomically exclude race ingress while starting one rate command.

        The lock order matches the existing reset-boundary transaction:
        send -> ingress dispatch -> state.  Consequently, a race status whose
        receive timestamp is admitted before the exact transport call-start
        must be visible to this guard.  A transition or finish refuses the
        command before any wire call; ingress received after the call waits
        until the transport returns.
        """

        if (
            type(expected_active_gate_index) is not int
            or expected_active_gate_index < 0
        ):
            raise ValueError(
                "expected_active_gate_index must be a non-negative exact int"
            )
        self._require_conn()
        if not all(
            math.isfinite(value)
            for value in (cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate)
        ):
            raise ValueError("body rates must be finite")
        thrust = _clamp_thrust(cmd.thrust)
        sx, sy, sz = self._rate_sign
        wire = self._attitude_wire_type(
            time_boot_ms=self._time_boot_ms(),
            target_system=self._target_system,
            target_component=self._target_component,
            type_mask=SET_ATTITUDE_TARGET_MASK_RATES_THRUST,
            q_wxyz=(1.0, 0.0, 0.0, 0.0),
            body_rates_rad_s=(
                sx * cmd.roll_rate,
                sy * cmd.pitch_rate,
                sz * cmd.yaw_rate,
            ),
            thrust=thrust,
        )
        powered_exact_zero = None
        if self._powered_transport is not None and powered_cleanup:
            from scripts.aigp_vq2_powered_runtime import exact_zero_rate_thrust

            powered_exact_zero = exact_zero_rate_thrust(
                {
                    "roll_rate_rad_s": cmd.roll_rate,
                    "pitch_rate_rad_s": cmd.pitch_rate,
                    "yaw_rate_rad_s": cmd.yaw_rate,
                    "thrust": cmd.thrust,
                }
            )

        with self._send_lock:
            with self._ingress_dispatch_lock:
                with self._state_lock:
                    received = self._latest_received_race_status
                    if received is None:
                        raise RuntimeError(
                            "race-active send lacks received race authority"
                        )
                    race_status = received.race_status
                    if (
                        race_status.active_gate_index
                        != expected_active_gate_index
                        or race_status.race_finish_time_ns >= 0
                    ):
                        raise RaceActiveBoundaryChangedBeforeWire(
                            "race-active send boundary changed before wire"
                        )
                    race_authority = {
                        "schema": "aigp-vq2-race-active-send-authority/1",
                        "expected_active_gate_index": (
                            expected_active_gate_index
                        ),
                        "received_race_status": received.to_primitive(),
                    }
                self._call_attitude_target_locked(
                    api="send_attitude_rate",
                    wire=wire,
                    powered_deadline_monotonic_ns=(
                        powered_deadline_monotonic_ns
                    ),
                    powered_cleanup=powered_cleanup,
                    powered_exact_zero=powered_exact_zero,
                    call_start_not_before_monotonic_ns=(
                        call_start_not_before_monotonic_ns
                    ),
                    call_start_deadline_monotonic_ns=(
                        call_start_deadline_monotonic_ns
                    ),
                )
        return race_authority

    async def send_position(
        self,
        cmd: PositionCommand,
        *,
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
    ) -> None:
        """Send SET_POSITION_TARGET_LOCAL_NED for parity only.

        First-contact live testing showed this path does not track velocity
        cleanly and can produce runaway climb. Use attitude targets for VQ1.
        """
        self._require_conn()
        n, e, d = cmd.position_ned
        vn, ve, vd = cmd.velocity_ned
        with self._send_lock:
            self._audit_outbound("position_target")
            call_start = self._read_monotonic_ns()
            self._authorize_powered_outbound_locked(
                "position_target",
                call_start=call_start,
                deadline_monotonic_ns=powered_deadline_monotonic_ns,
                cleanup=powered_cleanup,
                exact_zero=None,
            )
            self._conn.mav.set_position_target_local_ned_send(
                self._time_boot_ms(),
                self._target_system,
                self._target_component,
                MAV_FRAME_LOCAL_NED,
                SET_POSITION_TARGET_LOCAL_NED_MASK,
                n,
                e,
                d,
                vn,
                ve,
                vd,
                0.0,
                0.0,
                0.0,
                cmd.yaw_rad,
                0.0,
            )

    async def reset(
        self,
        *,
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
    ) -> Optional[TrackData]:
        if not self.fetch_track_on_connect:
            # VQ2 has no track-transfer acknowledgement.  Clear the local
            # epoch before sending and require the runner to prove an IMU/race
            # clock rollback before it can command flight.
            with self._state_lock:
                self._telemetry_ready_event.clear()
                self._have_imu = False
                self._last_imu_monotonic = 0.0
                self._last_race_status_monotonic = 0.0
                self._last_actuator_monotonic = 0.0
                self._latest_telem = None
                self._race_status = None
                self._actuator_outputs = None
                self._latest_received_race_status = None
                self._latest_received_actuator_output_status = None
                self._begin_ingress_generation_locked()
                self._reset_collision_generation_locked()
                generation = self._ingress_generation
            with self._audit_lock:
                self._begin_outbound_generation_locked(generation)
        await self._send_sim_reset(
            clear_track_event=True,
            powered_deadline_monotonic_ns=powered_deadline_monotonic_ns,
            powered_cleanup=powered_cleanup,
        )
        if not self.fetch_track_on_connect:
            return None
        await asyncio.to_thread(self._track_event.wait, 5.0)
        return self.track_data

    async def reset_calibration_with_boundary(
        self,
        persist_boundary: Callable[[CalibrationResetBoundaryV1], None],
        *,
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
        powered_progress: Optional[Callable[[], None]] = None,
    ) -> CalibrationResetBoundaryV1:
        """Persist an atomic old-generation capture before sending SIM_RESET.

        This calibration-only API leaves the legacy ``reset()`` contract
        intact.  The callback runs synchronously after the receiver/collision
        queues have moved to exactly one new generation and before any reset
        packet is attempted.  A prepower persistence failure sends nothing.
        Once powered cleanup is mandatory, a persistence failure is instead
        redacted and latched in :meth:`calibration_reset_persistence_state`,
        and cannot suppress the guarded reset attempt.  The callback must not
        re-enter an adapter send API.  ``powered_progress``, when supplied for
        powered cleanup, runs under the same atomic locks after persistence and
        immediately before the final reset authorization/send so parent-death
        takeover can complete without creating a second generation boundary.

        Collision rows here are immutable copies of the adapter's legacy raw
        collision facts.  The powered dispatcher, which knows phase and
        disposition, owns conversion to runner collision observations.
        """

        if not callable(persist_boundary):
            raise TypeError("persist_boundary must be callable")
        if powered_progress is not None:
            if not callable(powered_progress):
                raise TypeError("powered_progress must be callable or None")
            if self._powered_transport is None or powered_cleanup is not True:
                raise ValueError(
                    "powered_progress requires powered cleanup reset authority"
                )
        self._require_conn()

        # Exclude the announce thread and every command API from the boundary
        # transition through persistence and the reset call itself.  Excluding
        # receiver dispatch also prevents a receive timestamp captured before
        # this boundary from being appended later as a new-generation item.
        with self._send_lock:
            with self._ingress_dispatch_lock:
                with self._state_lock:
                    old_generation = self._ingress_generation
                    ingress_stats = self._ingress_stats_locked()
                    collision_stats = self._collision_stats_locked()
                    observations = self._snapshot_received_observations_locked()
                    collisions = tuple(
                        CalibrationCollisionV1(
                            id=item["id"],
                            threat_level=item["threat_level"],
                            impulse=item["impulse"],
                        )
                        for item in self._collisions
                    )
                    boundary_monotonic_ns = self._read_monotonic_ns()

                    self._telemetry_ready_event.clear()
                    self._have_attitude = False
                    self._have_lpn = False
                    self._have_odometry = False
                    self._have_imu = False
                    self._last_imu_monotonic = 0.0
                    self._last_race_status_monotonic = 0.0
                    self._last_actuator_monotonic = 0.0
                    self._latest_telem = None
                    self._race_status = None
                    self._actuator_outputs = None
                    self._indi_debug = None
                    self._indi_last_t_us = None
                    self._latest_received_race_status = None
                    self._latest_received_actuator_output_status = None
                    # Heartbeat state is intentionally retained so reset proof
                    # can demand a strictly newer post-reset heartbeat.
                    self._begin_ingress_generation_locked()
                    self._reset_collision_generation_locked()
                    new_generation = self._ingress_generation
                    boundary = CalibrationResetBoundaryV1(
                        old_generation=old_generation,
                        new_generation=new_generation,
                        boundary_monotonic_ns=boundary_monotonic_ns,
                        observations=observations,
                        collisions=collisions,
                        ingress_stats=ingress_stats,
                        collision_stats=collision_stats,
                    )
                    with self._audit_lock:
                        self._begin_outbound_generation_locked(new_generation)

                try:
                    persist_boundary(boundary)
                except BaseException as exc:
                    if (
                        self._powered_transport is None
                        or powered_cleanup is not True
                    ):
                        raise
                    self._powered_transport.outbound_guards.latch_production(
                        "reset_boundary_persistence_failed"
                    )
                    failure = CalibrationResetPersistenceFailureV1(
                        old_generation=boundary.old_generation,
                        new_generation=boundary.new_generation,
                        boundary_monotonic_ns=boundary.boundary_monotonic_ns,
                        error_type=_admitted_outbound_error_type(exc),
                    )
                    with self._state_lock:
                        if (
                            len(self._reset_persistence_failures)
                            >= DEFAULT_RESET_PERSISTENCE_FAILURE_CAPACITY
                        ):
                            self._reset_persistence_failures_dropped += 1
                        self._reset_persistence_failures.append(failure)
                self._track_event.clear()
                if self._vision is not None:
                    self._vision.reset()
                wire = self._command_long_wire_type(
                    target_system=self._target_system,
                    target_component=self._target_component,
                    command=SIM_RESET_COMMAND,
                    confirmation=0,
                    params=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                )
                if powered_progress is not None:
                    powered_progress()
                self._call_nonattitude_locked(
                    category="sim_reset",
                    api="command_long_send",
                    audit_name="sim_reset",
                    wire=wire,
                    call=lambda: self._conn.mav.command_long_send(
                        wire.target_system,
                        wire.target_component,
                        wire.command,
                        wire.confirmation,
                        *wire.params,
                    ),
                    powered_deadline_monotonic_ns=powered_deadline_monotonic_ns,
                    powered_cleanup=powered_cleanup,
                )
        return boundary

    async def wait_for_track_data(self, timeout_s: float = 10.0) -> Optional[TrackData]:
        await asyncio.to_thread(self._track_event.wait, timeout_s)
        return self.track_data

    @property
    def is_connected(self) -> bool:
        return self._conn is not None and (time.monotonic() - self._last_heartbeat_monotonic) < 3.0

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def heartbeat_age_s(self) -> float:
        """Wall-clock age of the newest simulator heartbeat."""

        return _monotonic_age(self._last_heartbeat_monotonic)

    @property
    def heartbeat_sequence(self) -> int:
        """Monotonic token incremented for every received heartbeat."""

        with self._state_lock:
            return self._heartbeat_sequence

    @property
    def imu_age_s(self) -> float:
        """Wall-clock age of the newest ``HIGHRES_IMU`` sample."""

        return _monotonic_age(self._last_imu_monotonic)

    @property
    def race_status_age_s(self) -> float:
        """Wall-clock age of the newest decoded race-status packet."""

        return _monotonic_age(self._last_race_status_monotonic)

    @property
    def actuator_age_s(self) -> float:
        """Wall-clock age of the newest actuator-output status packet."""

        return _monotonic_age(self._last_actuator_monotonic)

    @property
    def latest_telemetry(self) -> Optional[TelemetryState]:
        with self._state_lock:
            return self._latest_telem

    def drain_imu_samples(self) -> list[IMUData]:
        """Return every buffered IMU sample in receive order and clear it."""

        with self._state_lock:
            samples = [received.imu for received in self._imu_samples]
            self._imu_samples.clear()
            return samples

    def drain_received_imu_samples(self) -> list[ReceivedIMUSampleV1]:
        """Return exact receiver-boundary IMU envelopes and clear the queue."""

        with self._state_lock:
            samples = list(self._imu_samples)
            self._imu_samples.clear()
            return samples

    def drain_mavlink_arrivals(self) -> list[MavlinkIngressV1]:
        """Project non-IMU envelopes to legacy ingress rows, then clear them."""

        with self._state_lock:
            arrivals = [received.ingress for received in self._mavlink_arrivals]
            self._mavlink_arrivals.clear()
            return arrivals

    def drain_received_observations(
        self,
    ) -> list[
        ReceivedHeartbeatV1
        | ReceivedIMUSampleV1
        | ReceivedRaceStatusV1
        | ReceivedActuatorOutputStatusV1
    ]:
        """Atomically drain every exact received envelope in ingress order.

        The legacy drains are projections of these same two bounded queues;
        the adapter never stores a second copy of an occurrence.
        """

        with self._state_lock:
            values = list(self._mavlink_arrivals)
            values.extend(self._imu_samples)
            self._mavlink_arrivals.clear()
            self._imu_samples.clear()
        values.sort(key=lambda item: item.ingress.sequence)
        return values

    def drain_received_ingress(
        self,
    ) -> list[MavlinkIngressV1 | ReceivedIMUSampleV1]:
        """Atomically drain both exact ingress queues in global receive order."""

        with self._state_lock:
            values = [received.ingress for received in self._mavlink_arrivals]
            values.extend(self._imu_samples)
            self._mavlink_arrivals.clear()
            self._imu_samples.clear()
        values.sort(
            key=lambda item: (
                item.ingress.sequence
                if isinstance(item, self._received_imu_type)
                else item.sequence
            )
        )
        return values

    def ingress_stats(self) -> MavlinkIngressStats:
        with self._state_lock:
            return self._ingress_stats_locked()

    def outbound_audit(self) -> MavlinkOutboundAudit:
        with self._audit_lock:
            return MavlinkOutboundAudit(**self._outbound_counts)

    @property
    def powered_peer(self) -> Optional[tuple[str, int]]:
        if self._powered_transport is None:
            return None
        return self._powered_transport.peer

    @property
    def powered_source_promoted(self) -> bool:
        return bool(
            self._powered_transport is not None
            and self._powered_transport.promoted
        )

    @property
    def powered_source_rejected(self) -> bool:
        return bool(
            self._powered_transport is not None
            and self._powered_transport.source_gate.source_rejected_latched
        )

    @property
    def powered_source_authority(self):
        """Return the one transport-owned source freeze used by production."""

        if self._powered_transport is None:
            return None
        return self._powered_transport.source_gate

    @property
    def powered_receive_owner(self) -> Optional[str]:
        if self._powered_transport is None:
            return None
        return self._powered_transport.receive_owner

    def powered_transport_state(self) -> Optional[PoweredMavlinkTransportState]:
        """Snapshot caller-visible powered endpoint and worker closure facts."""

        transport = self._powered_transport
        if transport is None:
            return None
        endpoint = transport.endpoint
        receiver = self._rx_thread
        announcer = self._announce_thread
        peer = transport.peer
        return PoweredMavlinkTransportState(
            requested_host=endpoint.requested_host,
            requested_port=endpoint.requested_port,
            actual_host=endpoint.actual_host,
            actual_port=endpoint.actual_port,
            frozen_peer=None if peer is None else tuple(peer),
            rejected_source_count=transport.source_gate.rejected_source_count,
            endpoint_closed=endpoint.closed is True,
            receiver_joined=receiver is None or not receiver.is_alive(),
            announcer_joined=announcer is None or not announcer.is_alive(),
            connection_closed=self._conn is None,
        )

    def calibration_reset_persistence_state(
        self,
    ) -> CalibrationResetPersistenceState:
        """Return redacted, bounded proof of cleanup persistence failures."""

        with self._state_lock:
            return CalibrationResetPersistenceState(
                failures=tuple(self._reset_persistence_failures),
                dropped=self._reset_persistence_failures_dropped,
            )

    def receive_powered_external(
        self,
        max_wait_ns: int,
    ) -> Optional[PoweredDatagramDispatch]:
        """Boundedly read and dispatch one cleanup-owned powered datagram.

        This API is unavailable in normal worker mode. It is the sole receive
        call used by cleanup integration, so the transport source freeze and
        the production parser see the same datagram under one receive owner.
        """

        if (
            self._powered_transport is None
            or self.powered_receive_mode
            != POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP
        ):
            raise RuntimeError("external powered receive mode is not active")
        if self._conn is None:
            raise RuntimeError("external powered receive is not connected")
        received = self._powered_transport.recvfrom(
            owner=POWERED_RECEIVE_OWNER_EXTERNAL_CLEANUP,
            max_wait_ns=max_wait_ns,
        )
        if received is None:
            return None
        raw, source = received
        return self._handle_powered_datagram(raw, source)

    def announce_powered_external_cleanup(self) -> None:
        """Synchronously send one dispatcher-authorized TIMESYNC/GCS pair."""

        if (
            self._powered_transport is None
            or self.powered_receive_mode
            != POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP
        ):
            raise RuntimeError("external powered cleanup mode is not active")
        self._require_conn()
        timesync_wire = self._timesync_wire_type(tc1=0, ts1=time.time_ns())
        heartbeat_wire = self._gcs_heartbeat_wire_type(
            type=6,
            autopilot=8,
            base_mode=0,
            custom_mode=0,
            system_status=4,
        )
        with self._send_lock:
            self._call_nonattitude_locked(
                category="timesync",
                api="timesync_send",
                audit_name="timesync",
                wire=timesync_wire,
                call=lambda: self._conn.mav.timesync_send(
                    timesync_wire.tc1,
                    timesync_wire.ts1,
                ),
                powered_cleanup=True,
            )
            self._call_nonattitude_locked(
                category="gcs_heartbeat",
                api="heartbeat_send",
                audit_name="gcs_heartbeat",
                wire=heartbeat_wire,
                call=lambda: self._conn.mav.heartbeat_send(
                    heartbeat_wire.type,
                    heartbeat_wire.autopilot,
                    heartbeat_wire.base_mode,
                    heartbeat_wire.custom_mode,
                    heartbeat_wire.system_status,
                ),
                powered_cleanup=True,
            )

    @property
    def powered_outbound_guards(self):
        if self._powered_transport is None:
            return None
        return self._powered_transport.outbound_guards

    def outbound_receipt_stats(self) -> MavlinkOutboundReceiptStats:
        with self._audit_lock:
            return MavlinkOutboundReceiptStats(
                generation=self._outbound_generation,
                next_sequence=self._outbound_next_sequence,
                returned=self._outbound_returned,
                raised=self._outbound_raised,
                dropped=self._outbound_dropped,
                high_watermark=self._outbound_high_watermark,
                capacity=self.outbound_receipt_capacity,
                buffered=len(self._outbound_receipts),
            )

    def drain_outbound_receipts(
        self,
    ) -> list[AttitudeTargetOutboundV1 | NonAttitudeOutboundV1]:
        """Return bounded exact local-call receipts and clear their queue."""

        with self._audit_lock:
            receipts = list(self._outbound_receipts)
            self._outbound_receipts.clear()
            return receipts

    def collision_stats(self) -> MavlinkCollisionStats:
        """Return calibration-only collision-buffer diagnostics."""

        with self._state_lock:
            return self._collision_stats_locked()

    def drain_collisions_with_stats(
        self,
    ) -> tuple[list[dict[str, int | float]], MavlinkCollisionStats]:
        """Atomically snapshot collision accounting and drain its exact batch."""

        with self._state_lock:
            stats = self._collision_stats_locked()
            collisions = [dict(item) for item in self._collisions]
            self._collisions.clear()
            return collisions, stats

    @property
    def race_status(self) -> Optional[RaceStatus]:
        with self._state_lock:
            return self._race_status

    @property
    def latest_received_heartbeat(self) -> Optional[ReceivedHeartbeatV1]:
        with self._state_lock:
            return self._latest_received_heartbeat

    @property
    def latest_received_race_status(self) -> Optional[ReceivedRaceStatusV1]:
        with self._state_lock:
            return self._latest_received_race_status

    @property
    def latest_received_actuator_output_status(
        self,
    ) -> Optional[ReceivedActuatorOutputStatusV1]:
        with self._state_lock:
            return self._latest_received_actuator_output_status

    @property
    def track_data(self) -> Optional[TrackData]:
        with self._state_lock:
            return self._track_data

    @property
    def actuator_outputs(self) -> Optional[Dict]:
        with self._state_lock:
            return self._actuator_outputs

    @property
    def indi_debug(self) -> Optional[Dict]:
        """Latest INDI inner-loop debug snapshot (None unless _use_indi is on).

        Mirrors ``actuator_outputs`` so the recorder/runner can log the INDI
        read-out (alpha_des, alpha_meas, Ghat, saturation flags, u) per tick.
        """
        with self._state_lock:
            return self._indi_debug

    def drain_collisions(self):
        with self._state_lock:
            out = list(self._collisions)
            self._collisions.clear()
            return out

    async def _send_sim_reset(
        self,
        clear_track_event: bool = False,
        *,
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
    ) -> None:
        self._require_conn()
        if clear_track_event:
            self._track_event.clear()
        if self._vision is not None:
            self._vision.reset()
        wire = self._command_long_wire_type(
            target_system=self._target_system,
            target_component=self._target_component,
            command=SIM_RESET_COMMAND,
            confirmation=0,
            params=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        with self._send_lock:
            self._call_nonattitude_locked(
                category="sim_reset",
                api="command_long_send",
                audit_name="sim_reset",
                wire=wire,
                call=lambda: self._conn.mav.command_long_send(
                    wire.target_system,
                    wire.target_component,
                    wire.command,
                    wire.confirmation,
                    *wire.params,
                ),
                powered_deadline_monotonic_ns=powered_deadline_monotonic_ns,
                powered_cleanup=powered_cleanup,
            )

    def _begin_ingress_generation_locked(self) -> None:
        self._ingress_generation += 1
        self._ingress_next_sequence = 0
        for name in self._ingress_counts:
            self._ingress_counts[name] = 0
        self._ingress_dropped = 0
        self._ingress_high_watermark = 0
        self._imu_ingress_dropped = 0
        self._other_ingress_dropped = 0
        self._imu_ingress_high_watermark = 0
        self._other_ingress_high_watermark = 0
        self._imu_samples.clear()
        self._mavlink_arrivals.clear()

    def _snapshot_received_observations_locked(self) -> tuple[Any, ...]:
        values = list(self._mavlink_arrivals)
        values.extend(self._imu_samples)
        values.sort(key=lambda item: item.ingress.sequence)
        return tuple(
            type(item).from_primitive(item.to_primitive()) for item in values
        )

    def _ingress_stats_locked(self) -> MavlinkIngressStats:
        return MavlinkIngressStats(
            generation=self._ingress_generation,
            next_sequence=self._ingress_next_sequence,
            highres_imu_received=self._ingress_counts["HIGHRES_IMU"],
            heartbeat_received=self._ingress_counts["HEARTBEAT"],
            race_status_received=self._ingress_counts["RACE_STATUS"],
            actuator_received=self._ingress_counts["ACTUATOR_OUTPUT_STATUS"],
            dropped=self._ingress_dropped,
            high_watermark=self._ingress_high_watermark,
            imu_capacity=self.ingress_buffer_capacity,
            other_capacity=self.ingress_buffer_capacity,
            imu_dropped=self._imu_ingress_dropped,
            other_dropped=self._other_ingress_dropped,
            imu_high_watermark=self._imu_ingress_high_watermark,
            other_high_watermark=self._other_ingress_high_watermark,
            buffered_imu=len(self._imu_samples),
            buffered_other=len(self._mavlink_arrivals),
        )

    def _collision_stats_locked(self) -> MavlinkCollisionStats:
        return MavlinkCollisionStats(
            generation=self._ingress_generation,
            handled=self._collision_handled,
            dropped=self._collision_dropped,
            high_watermark=self._collision_high_watermark,
            capacity=DEFAULT_COLLISION_BUFFER_CAPACITY,
            buffered=len(self._collisions),
        )

    def _reset_collision_generation_locked(self) -> None:
        self._collisions.clear()
        self._collision_handled = 0
        self._collision_dropped = 0
        self._collision_high_watermark = 0

    def _append_collision_locked(self, collision: dict[str, int | float]) -> None:
        self._collision_handled += 1
        if len(self._collisions) >= DEFAULT_COLLISION_BUFFER_CAPACITY:
            self._collision_dropped += 1
        self._collisions.append(dict(collision))
        self._collision_high_watermark = max(
            self._collision_high_watermark,
            len(self._collisions),
        )

    def _new_ingress_locked(
        self,
        message_type: str,
        received_monotonic_ns: int,
        *,
        source_time_value: Optional[int] = None,
        source_time_unit: Optional[str] = None,
    ) -> MavlinkIngressV1:
        ingress = self._mavlink_ingress_type(
            stream_id=VQ2_MAVLINK_STREAM_ID,
            generation=self._ingress_generation,
            sequence=self._ingress_next_sequence,
            message_type=message_type,
            host_clock_id=self.host_clock_id,
            received_monotonic_ns=received_monotonic_ns,
            source_time_value=source_time_value,
            source_time_unit=source_time_unit,
        )
        return ingress

    def _commit_ingress_locked(self, ingress: MavlinkIngressV1) -> None:
        """Commit one fully constructed receive envelope to exact counters."""

        if (
            ingress.generation != self._ingress_generation
            or ingress.sequence != self._ingress_next_sequence
            or ingress.message_type not in self._ingress_counts
        ):
            raise RuntimeError("MAVLink ingress commit does not match current state")
        self._ingress_next_sequence += 1
        self._ingress_counts[ingress.message_type] += 1

    def _append_ingress_locked(self, queue: Deque, value) -> None:
        is_imu_queue = queue is self._imu_samples
        if len(queue) >= self.ingress_buffer_capacity:
            self._ingress_dropped += 1
            if is_imu_queue:
                self._imu_ingress_dropped += 1
            else:
                self._other_ingress_dropped += 1
        queue.append(value)
        if is_imu_queue:
            self._imu_ingress_high_watermark = max(
                self._imu_ingress_high_watermark, len(queue)
            )
        else:
            self._other_ingress_high_watermark = max(
                self._other_ingress_high_watermark, len(queue)
            )
        self._ingress_high_watermark = max(
            self._ingress_high_watermark,
            len(self._imu_samples) + len(self._mavlink_arrivals),
        )

    def _read_monotonic_ns(self) -> int:
        value = self._monotonic_ns()
        if type(value) is not int or value < 0:
            raise ValueError("monotonic_ns clock must return a non-negative exact int")
        return value

    def _audit_outbound(self, name: str) -> None:
        with self._audit_lock:
            self._outbound_counts[name] += 1

    def _call_attitude_target_locked(
        self,
        *,
        api: str,
        wire,
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
        powered_exact_zero: Optional[bool] = None,
        call_start_not_before_monotonic_ns: Optional[int] = None,
        call_start_deadline_monotonic_ns: Optional[int] = None,
    ) -> None:
        """Call SET_ATTITUDE_TARGET and emit one exact return/raise receipt."""

        call_start = self._read_monotonic_ns()
        for label, value in (
            ("call-start not-before", call_start_not_before_monotonic_ns),
            ("call-start deadline", call_start_deadline_monotonic_ns),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{label} must be a non-negative exact integer")
        if (
            call_start_not_before_monotonic_ns is not None
            and call_start_deadline_monotonic_ns is not None
            and call_start_not_before_monotonic_ns
            >= call_start_deadline_monotonic_ns
        ):
            raise ValueError("call-start pacing window is empty")
        if (
            call_start_not_before_monotonic_ns is not None
            and call_start < call_start_not_before_monotonic_ns
        ):
            raise TimeoutError("attitude-target call began before its pacing window")
        if (
            call_start_deadline_monotonic_ns is not None
            and call_start >= call_start_deadline_monotonic_ns
        ):
            raise TimeoutError("attitude-target call-start deadline was reached")
        self._authorize_powered_outbound_locked(
            "attitude_target",
            call_start=call_start,
            deadline_monotonic_ns=powered_deadline_monotonic_ns,
            cleanup=powered_cleanup,
            exact_zero=powered_exact_zero,
        )
        if self._powered_transport is not None:
            self._powered_transport.begin_authorized_write()
        self._audit_outbound("attitude_target")
        try:
            self._conn.mav.set_attitude_target_send(
                wire.time_boot_ms,
                wire.target_system,
                wire.target_component,
                wire.type_mask,
                list(wire.q_wxyz),
                *wire.body_rates_rad_s,
                wire.thrust,
            )
            if self._powered_transport is not None:
                self._powered_transport.finish_authorized_write()
        except BaseException as exc:
            if self._powered_transport is not None:
                self._powered_transport.cancel_authorized_write()
            call_end = self._read_monotonic_ns()
            self._append_attitude_receipt(
                api=api,
                wire=wire,
                call_start=call_start,
                call_end=call_end,
                outcome="raised",
                error_type=_admitted_outbound_error_type(exc),
            )
            raise
        call_end = self._read_monotonic_ns()
        self._append_attitude_receipt(
            api=api,
            wire=wire,
            call_start=call_start,
            call_end=call_end,
            outcome="returned",
            error_type=None,
        )

    def _call_nonattitude_locked(
        self,
        *,
        category: str,
        api: str,
        audit_name: str,
        wire,
        call: Callable[[], None],
        powered_deadline_monotonic_ns: Optional[int] = None,
        powered_cleanup: bool = False,
    ) -> None:
        """Call an admitted nonattitude API and receipt return or raise."""

        call_start = self._read_monotonic_ns()
        call_deadline = self._authorize_powered_outbound_locked(
            category,
            call_start=call_start,
            deadline_monotonic_ns=powered_deadline_monotonic_ns,
            cleanup=powered_cleanup,
            exact_zero=None,
        )
        if self._powered_transport is not None:
            self._powered_transport.begin_authorized_write()
        self._audit_outbound(audit_name)
        try:
            call()
            if self._powered_transport is not None:
                self._powered_transport.finish_authorized_write()
        except BaseException as exc:
            if self._powered_transport is not None:
                self._powered_transport.cancel_authorized_write()
            call_end = self._read_monotonic_ns()
            self._append_nonattitude_receipt(
                category=category,
                api=api,
                wire=wire,
                call_start=call_start,
                call_end=call_end,
                outcome="raised",
                error_type=_admitted_outbound_error_type(exc),
            )
            if call_deadline is not None and call_end >= call_deadline:
                self._latch_late_powered_nonattitude_call(
                    cleanup=powered_cleanup
                )
            raise
        call_end = self._read_monotonic_ns()
        self._append_nonattitude_receipt(
            category=category,
            api=api,
            wire=wire,
            call_start=call_start,
            call_end=call_end,
            outcome="returned",
            error_type=None,
        )
        if call_deadline is not None and call_end >= call_deadline:
            from scripts.aigp_vq2_powered_runtime import OutboundAuthorityError

            self._latch_late_powered_nonattitude_call(
                cleanup=powered_cleanup
            )
            raise OutboundAuthorityError(
                "powered nonattitude call completed after its clipped deadline"
            )

    def _latch_late_powered_nonattitude_call(self, *, cleanup: bool) -> None:
        guards = self._powered_transport.outbound_guards
        if cleanup is True:
            guards.close_cleanup()
        else:
            guards.latch_production("powered_nonattitude_call_deadline_reached")

    def _authorize_powered_outbound_locked(
        self,
        category: str,
        *,
        call_start: int,
        deadline_monotonic_ns: Optional[int],
        cleanup: bool,
        exact_zero: Optional[bool],
    ) -> Optional[int]:
        if self._powered_transport is None:
            if deadline_monotonic_ns is not None or cleanup:
                raise ValueError(
                    "powered outbound options require powered transport"
                )
            return None

        from scripts.aigp_vq2_powered_runtime import OutboundAuthorityError

        transport = self._powered_transport
        if type(cleanup) is not bool:
            transport.outbound_guards.latch_production(
                "powered_cleanup_flag_invalid"
            )
            raise OutboundAuthorityError(
                "powered_cleanup must be an exact boolean"
            )
        external_cleanup_announcement = bool(
            cleanup
            and self.powered_receive_mode
            == POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP
            and category in {"timesync", "gcs_heartbeat"}
        )
        if external_cleanup_announcement:
            if deadline_monotonic_ns is not None:
                raise OutboundAuthorityError(
                    "external cleanup announcement deadline is dispatcher-owned"
                )
            if transport.outbound_guards.cleanup_state == "closed":
                raise OutboundAuthorityError("cleanup guard is closed")
            if not transport.source_gate.outbound_permitted(category):
                raise OutboundAuthorityError(
                    "cleanup announcement source authority is unavailable"
                )
            authorize = transport.external_cleanup_authorize
            if not callable(authorize):
                raise OutboundAuthorityError(
                    "cleanup announcement dispatcher is unavailable"
                )
            self._powered_callback_bool(
                transport.parent_alive,
                "parent_alive",
            )
            try:
                dispatch_deadline = authorize(category)
            except BaseException as exc:
                raise OutboundAuthorityError(
                    "cleanup announcement dispatcher rejected the call"
                ) from exc
            if type(dispatch_deadline) is not int or dispatch_deadline < 1:
                raise OutboundAuthorityError(
                    "cleanup announcement dispatcher returned an invalid deadline"
                )
            if call_start >= dispatch_deadline:
                raise OutboundAuthorityError(
                    "cleanup announcement deadline was reached"
                )
            if not self._powered_callback_bool(
                transport.lease_valid,
                "lease_valid",
            ):
                raise OutboundAuthorityError(
                    "cleanup announcement lease lineage is invalid"
                )
            parent_alive = self._powered_callback_bool(
                transport.parent_alive,
                "parent_alive",
            )
            cleanup_state = transport.outbound_guards.cleanup_state
            if parent_alive and cleanup_state in {
                "takeover_pending",
                "enabled_takeover",
            }:
                raise OutboundAuthorityError(
                    "cleanup announcement parent state conflicts with takeover"
                )
            if not parent_alive and cleanup_state not in {
                "takeover_pending",
                "enabled_takeover",
            }:
                transport.outbound_guards.note_parent_death()
                raise OutboundAuthorityError(
                    "cleanup announcement requires abandoned takeover"
                )
            return min(
                dispatch_deadline,
                call_start + POWERED_OUTBOUND_CALL_NS,
            )
        if type(deadline_monotonic_ns) is not int or deadline_monotonic_ns < 1:
            transport.outbound_guards.latch_production(
                "powered_call_deadline_missing"
            )
            raise OutboundAuthorityError(
                "powered outbound call requires an exact absolute deadline"
            )
        call_deadline = min(
            deadline_monotonic_ns,
            call_start + POWERED_OUTBOUND_CALL_NS,
        )
        parent_alive = self._powered_callback_bool(
            transport.parent_alive,
            "parent_alive",
        )
        if not cleanup and not parent_alive:
            transport.outbound_guards.note_parent_death()
            raise OutboundAuthorityError("wrapper parent is not live")
        lease_valid = self._powered_callback_bool(
            transport.lease_valid,
            "lease_valid",
        )
        if cleanup:
            transport.outbound_guards.authorize_cleanup(
                category,
                now_monotonic_ns=call_start,
                deadline_monotonic_ns=call_deadline,
                parent_alive=parent_alive,
                lease_valid=lease_valid,
                source_promoted=transport.promoted,
                exact_zero=exact_zero,
            )
            return call_deadline
        role_valid = self._powered_callback_bool(
            transport.role_valid,
            "role_valid",
        )
        transport.outbound_guards.authorize_production(
            category,
            now_monotonic_ns=call_start,
            deadline_monotonic_ns=call_deadline,
            role_valid=role_valid,
            parent_alive=parent_alive,
            lease_valid=lease_valid,
            peer_frozen=transport.peer is not None,
            source_valid=not transport.source_gate.source_rejected_latched,
            source_promoted=transport.promoted,
        )
        return call_deadline

    def _append_attitude_receipt(
        self,
        *,
        api: str,
        wire,
        call_start: int,
        call_end: int,
        outcome: str,
        error_type: Optional[str],
    ) -> None:
        with self._audit_lock:
            receipt = self._attitude_outbound_type(
                stream_id=VQ2_MAVLINK_STREAM_ID,
                reset_generation=self._outbound_generation,
                outbound_sequence=self._outbound_next_sequence,
                host_clock_id=self.host_clock_id,
                call_start_monotonic_ns=call_start,
                call_end_monotonic_ns=call_end,
                api=api,
                outcome=outcome,
                error_type=error_type,
                wire=wire,
            )
            self._append_outbound_receipt_locked(receipt, outcome=outcome)

    def _append_nonattitude_receipt(
        self,
        *,
        category: str,
        api: str,
        wire,
        call_start: int,
        call_end: int,
        outcome: str,
        error_type: Optional[str],
    ) -> None:
        with self._audit_lock:
            receipt = self._nonattitude_outbound_type(
                stream_id=VQ2_MAVLINK_STREAM_ID,
                reset_generation=self._outbound_generation,
                outbound_sequence=self._outbound_next_sequence,
                host_clock_id=self.host_clock_id,
                call_start_monotonic_ns=call_start,
                call_end_monotonic_ns=call_end,
                category=category,
                api=api,
                outcome=outcome,
                error_type=error_type,
                wire=wire,
            )
            self._append_outbound_receipt_locked(receipt, outcome=outcome)

    def _append_outbound_receipt_locked(self, receipt, *, outcome: str) -> None:
        if len(self._outbound_receipts) >= self.outbound_receipt_capacity:
            self._outbound_dropped += 1
        self._outbound_receipts.append(receipt)
        self._outbound_next_sequence += 1
        if outcome == "returned":
            self._outbound_returned += 1
        else:
            self._outbound_raised += 1
        self._outbound_high_watermark = max(
            self._outbound_high_watermark,
            len(self._outbound_receipts),
        )

    def _begin_outbound_generation_locked(
        self,
        generation: int,
        *,
        initialize_attempt: bool = False,
    ) -> None:
        self._outbound_generation = generation
        if initialize_attempt:
            self._outbound_next_sequence = 0
            self._outbound_returned = 0
            self._outbound_raised = 0
            self._outbound_dropped = 0
            self._outbound_high_watermark = 0
            self._outbound_receipts.clear()

    def _handle_message(self, msg, *, received_monotonic_ns: Optional[int] = None) -> None:
        with self._ingress_dispatch_lock:
            self._handle_message_locked(
                msg,
                received_monotonic_ns=received_monotonic_ns,
            )

    def _handle_message_locked(
        self,
        msg,
        *,
        received_monotonic_ns: Optional[int] = None,
        raise_handler_errors: bool = False,
    ) -> None:
        try:
            received_ns = (
                self._read_monotonic_ns()
                if received_monotonic_ns is None
                else received_monotonic_ns
            )
            if type(received_ns) is not int or received_ns < 0:
                raise ValueError(
                    "received_monotonic_ns must be a non-negative exact int"
                )
            msg_type = msg.get_type()
            if msg_type == "BAD_DATA":
                return
            if msg_type == "HEARTBEAT":
                self._handle_heartbeat(msg, received_ns)
            elif msg_type == "LOCAL_POSITION_NED":
                self._handle_local_position(msg)
            elif msg_type == "ODOMETRY":
                self._handle_odometry(msg)
            elif msg_type == "ATTITUDE":
                self._handle_attitude(msg)
            elif msg_type == "HIGHRES_IMU":
                self._handle_highres_imu(msg, received_ns)
            elif msg_type == "ACTUATOR_OUTPUT_STATUS":
                self._handle_actuator(msg, received_ns)
            elif msg_type == "COLLISION":
                self._handle_collision(msg)
            elif msg_type == "DATA_TRANSMISSION_HANDSHAKE":
                self._reassembler.begin_transfer(msg.width, msg.packets)
            elif msg_type == "ENCAPSULATED_DATA":
                self._handle_encapsulated(msg, received_ns)
            elif msg_type == "STATUSTEXT":
                self._handle_statustext(msg)
            else:
                # First-sighting log of any unhandled type. The sim may report a
                # disqualification (DSQ) or other verdict on a channel we don't
                # decode; surface it once rather than dropping it silently.
                if msg_type not in self._seen_msg_types:
                    self._seen_msg_types.add(msg_type)
                    logger.info("AIGP: first %s message seen (unhandled): %s",
                                msg_type, msg.to_dict() if hasattr(msg, "to_dict") else msg)
        except Exception:
            logger.exception("AIGP MAVLink message handler failed")
            if raise_handler_errors:
                raise

    def _handle_statustext(self, msg) -> None:
        """Capture + log STATUSTEXT. The DSQ verdict (if the sim sends one over
        MAVLink) almost certainly arrives here. Always log it at WARNING so it
        is impossible to miss in a run's output."""
        text = getattr(msg, "text", "")
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8", "replace")
        text = str(text).strip("\x00").strip()
        severity = getattr(msg, "severity", None)
        with self._state_lock:
            self._status_texts.append({
                "severity": severity,
                "text": text,
                "monotonic": time.monotonic(),
            })
        logger.warning("AIGP STATUSTEXT (sev=%s): %s", severity, text)

    def _handle_heartbeat(self, msg, received_monotonic_ns: int) -> None:
        payload = self._heartbeat_payload_type(
            base_mode=msg.base_mode,
            custom_mode=msg.custom_mode,
        )
        with self._state_lock:
            ingress = self._new_ingress_locked(
                "HEARTBEAT", received_monotonic_ns
            )
            received = self._received_heartbeat_type(
                ingress=ingress,
                heartbeat=payload,
            )
            self._commit_ingress_locked(ingress)
            self._append_ingress_locked(self._mavlink_arrivals, received)
            self._latest_received_heartbeat = received
            self._last_heartbeat_monotonic = time.monotonic()
            self._heartbeat_sequence += 1
            self._armed = bool(payload.base_mode & 0x80)
            if self._conn is not None:
                self._target_system = getattr(self._conn, "target_system", self._target_system) or self._target_system
                self._target_component = getattr(self._conn, "target_component", self._target_component) or self._target_component
            self._heartbeat_event.set()

    def _handle_local_position(self, msg) -> None:
        with self._state_lock:
            old = self._latest_telem
            self._latest_telem = _telem_with(
                old,
                timestamp_us=int(msg.time_boot_ms) * 1000,
                position_ned=(msg.x, msg.y, msg.z),
                velocity_ned=(msg.vx, msg.vy, msg.vz),
                lpn_time_boot_ms=int(msg.time_boot_ms),
            )
            self._have_lpn = True
            self._maybe_ready()

    def _handle_odometry(self, msg) -> None:
        q = msg.q
        with self._state_lock:
            old = self._latest_telem
            self._latest_telem = _telem_with(
                old,
                timestamp_us=int(msg.time_usec),
                orientation=Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]),
                odom_time_usec=int(msg.time_usec),
                odom_quality=getattr(msg, "quality", None),
                odom_reset_counter=getattr(msg, "reset_counter", None),
            )
            self._have_odometry = True

    def _handle_attitude(self, msg) -> None:
        with self._state_lock:
            orientation = None
            if not self._have_odometry:
                orientation = Quaternion.from_euler(msg.roll, msg.pitch, msg.yaw)
            old = self._latest_telem
            self._latest_telem = _telem_with(
                old,
                timestamp_us=int(msg.time_boot_ms) * 1000,
                orientation=orientation,
                angular_velocity=(msg.rollspeed, msg.pitchspeed, msg.yawspeed),
            )
            self._have_attitude = True
            self._maybe_ready()

    def _handle_highres_imu(self, msg, received_monotonic_ns: int) -> None:
        imu = IMUData(
            timestamp_us=int(msg.time_usec),
            accel=(msg.xacc, msg.yacc, msg.zacc),
            gyro=(msg.xgyro, msg.ygyro, msg.zgyro),
            mag=None,
        )
        with self._state_lock:
            ingress = self._new_ingress_locked(
                "HIGHRES_IMU",
                received_monotonic_ns,
                source_time_value=imu.timestamp_us,
                source_time_unit="us",
            )
            received = self._received_imu_type(ingress=ingress, imu=imu)
            self._commit_ingress_locked(ingress)
            self._last_imu_monotonic = time.monotonic()
            self._append_ingress_locked(self._imu_samples, received)
            self._latest_telem = _telem_with(self._latest_telem, imu=imu)
            self._have_imu = True
            self._maybe_ready()

    def _handle_actuator(self, msg, received_monotonic_ns: int) -> None:
        payload = self._actuator_payload_type(
            time_usec=msg.time_usec,
            active=msg.active,
            actuator=tuple(msg.actuator),
        )
        with self._state_lock:
            ingress = self._new_ingress_locked(
                "ACTUATOR_OUTPUT_STATUS",
                received_monotonic_ns,
                source_time_value=payload.time_usec,
                source_time_unit="us",
            )
            received = self._received_actuator_type(
                ingress=ingress,
                actuator_output_status=payload,
            )
            self._commit_ingress_locked(ingress)
            self._append_ingress_locked(self._mavlink_arrivals, received)
            self._latest_received_actuator_output_status = received
            self._last_actuator_monotonic = time.monotonic()
            self._actuator_outputs = {
                "time_usec": payload.time_usec,
                "active": payload.active,
                "actuator": list(payload.actuator),
            }

    def _handle_collision(self, msg) -> None:
        collision = CalibrationCollisionV1(
            id=msg.id,
            threat_level=msg.threat_level,
            impulse=msg.horizontal_minimum_delta,
        ).to_primitive()
        with self._state_lock:
            self._append_collision_locked(collision)

    def _handle_encapsulated(self, msg, received_monotonic_ns: int) -> None:
        payload = bytes(msg.data)
        if not payload:
            return
        data_type = payload[0]
        if data_type == ENCAPSULATED_RACE_STATUS_MSG_ID:
            race_status = parse_race_status(payload)
            race_payload = self._race_status_payload_type(
                sim_boot_time_ms=race_status.sim_boot_time_ms,
                race_start_boot_time_ms=race_status.race_start_boot_time_ms,
                race_finish_time_ns=race_status.race_finish_time_ns,
                active_gate_index=race_status.active_gate_index,
                last_gate_race_time=race_status.last_gate_race_time,
            )
            with self._state_lock:
                ingress = self._new_ingress_locked(
                    "RACE_STATUS",
                    received_monotonic_ns,
                    source_time_value=race_payload.sim_boot_time_ms,
                    source_time_unit="ms",
                )
                received = self._received_race_status_type(
                    ingress=ingress,
                    race_status=race_payload,
                )
                self._commit_ingress_locked(ingress)
                self._append_ingress_locked(self._mavlink_arrivals, received)
                self._latest_received_race_status = received
                self._last_race_status_monotonic = time.monotonic()
                self._race_status = race_status
            return
        if data_type == ENCAPSULATED_TRACK_INFO_MSG_ID:
            if len(payload) < 3:
                return
            transfer_id = int.from_bytes(payload[1:3], "little")
            track = self._reassembler.feed_chunk(transfer_id, msg.seqnr, payload[3:])
            if track is not None:
                with self._state_lock:
                    self._track_data = track
                    self._track_event.set()

    def _maybe_ready(self) -> None:
        if self.telemetry_mode == "imu":
            ready = self._have_imu
        else:
            ready = self._have_attitude and self._have_lpn
        if ready:
            self._telemetry_ready_event.set()

    def _rx_loop(self) -> None:  # pragma: no cover - live socket loop
        while not self._stop_event.is_set():
            try:
                msg = self._conn.recv_match(blocking=True, timeout=0.5)
            except Exception:
                logger.exception("AIGP MAVLink recv failed")
                continue
            if msg is not None:
                self._handle_message(msg)

    def _powered_rx_loop(self) -> None:  # pragma: no cover - live socket loop
        transport = self._powered_transport
        while not self._stop_event.is_set():
            try:
                received = transport.recvfrom(
                    owner=POWERED_RECEIVE_OWNER_WORKER,
                    max_wait_ns=POWERED_WORKER_POLL_NS,
                )
                if received is None:
                    continue
                raw, source = received
            except BaseException:
                if self._stop_event.is_set() or transport.endpoint.closed:
                    return
                self._latch_powered_receiver_failure("powered_udp_receive_failed")
                return
            self._handle_powered_datagram(raw, source)

    def _handle_powered_datagram(
        self,
        raw: bytes,
        source: Any,
    ) -> PoweredDatagramDispatch:
        """Gate raw source, then forward accepted bytes to production parsing."""

        transport = self._powered_transport
        admitted_type: Optional[str] = None
        admitted_generation: Optional[int] = None
        message = None
        with self._ingress_dispatch_lock:
            received_monotonic_ns = self._read_monotonic_ns()
            decision = transport.source_gate.ingest(raw, source)
            if decision.rejected_source:
                transport.outbound_guards.latch_production(
                    "mavlink_source_rejected"
                )
                if not transport.promoted:
                    self._powered_failure_reason = "MAVLink source was rejected"
                    self._powered_failure_event.set()
                    self._powered_promotion_event.set()
                return PoweredDatagramDispatch(
                    source_accepted=False,
                    peer_frozen_now=False,
                    rejected_source=True,
                    malformed=False,
                    production_dispatched=False,
                    source_promoted=transport.promoted,
                    peer=transport.peer,
                    admitted_message_type=None,
                    failure_reason=None,
                )
            if decision.malformed:
                return PoweredDatagramDispatch(
                    source_accepted=False,
                    peer_frozen_now=False,
                    rejected_source=False,
                    malformed=True,
                    production_dispatched=False,
                    source_promoted=transport.promoted,
                    peer=transport.peer,
                    admitted_message_type=None,
                    failure_reason=None,
                )
            if not decision.accepted:
                self._latch_powered_receiver_failure(
                    "mavlink_source_decision_invalid"
                )
                return PoweredDatagramDispatch(
                    source_accepted=False,
                    peer_frozen_now=False,
                    rejected_source=False,
                    malformed=False,
                    production_dispatched=False,
                    source_promoted=transport.promoted,
                    peer=transport.peer,
                    admitted_message_type=None,
                    failure_reason="mavlink_source_decision_invalid",
                )

            # Deliberately ignore decision.message: it came from the scratch
            # validator. Production state receives only the accepted raw bytes
            # parsed again by the established production pymavlink object.
            try:
                message = transport.parse_production(raw)
            except BaseException:
                self._latch_powered_receiver_failure(
                    "production_mavlink_parse_failed"
                )
                return PoweredDatagramDispatch(
                    source_accepted=True,
                    peer_frozen_now=decision.peer_frozen_now,
                    rejected_source=False,
                    malformed=False,
                    production_dispatched=False,
                    source_promoted=transport.promoted,
                    peer=transport.peer,
                    admitted_message_type=None,
                    failure_reason="production_mavlink_parse_failed",
                )

            with self._state_lock:
                generation_before = self._ingress_generation
                counts_before = dict(self._ingress_counts)
            try:
                self._handle_message_locked(
                    message,
                    received_monotonic_ns=received_monotonic_ns,
                    raise_handler_errors=True,
                )
            except BaseException:
                self._latch_powered_receiver_failure(
                    "production_mavlink_handler_failed"
                )
                return PoweredDatagramDispatch(
                    source_accepted=True,
                    peer_frozen_now=decision.peer_frozen_now,
                    rejected_source=False,
                    malformed=False,
                    production_dispatched=True,
                    source_promoted=transport.promoted,
                    peer=transport.peer,
                    admitted_message_type=None,
                    failure_reason="production_mavlink_handler_failed",
                )
            with self._state_lock:
                generation_after = self._ingress_generation
                changed = [
                    name
                    for name, count in self._ingress_counts.items()
                    if count == counts_before[name] + 1
                ]
                no_other_changes = all(
                    self._ingress_counts[name] == counts_before[name]
                    for name in self._ingress_counts
                    if name not in changed
                )
            if generation_after != generation_before:
                self._latch_powered_receiver_failure(
                    "ingress_generation_changed_during_datagram"
                )
                return PoweredDatagramDispatch(
                    source_accepted=True,
                    peer_frozen_now=decision.peer_frozen_now,
                    rejected_source=False,
                    malformed=False,
                    production_dispatched=True,
                    source_promoted=transport.promoted,
                    peer=transport.peer,
                    admitted_message_type=None,
                    failure_reason="ingress_generation_changed_during_datagram",
                )
            if len(changed) == 1 and no_other_changes:
                admitted_type = changed[0]
                admitted_generation = generation_after

        # Promotion shares the outbound send lock. Thus an announcement is
        # linearly either wholly before promotion or denied after promotion.
        if admitted_type in {"HEARTBEAT", "RACE_STATUS", "HIGHRES_IMU"}:
            with self._send_lock:
                with self._state_lock:
                    generation_still_current = (
                        self._ingress_generation == admitted_generation
                    )
                if (
                    generation_still_current
                    and transport.source_gate.observe_fresh_stream(admitted_type)
                ):
                    self._powered_promotion_event.set()

        source_system = getattr(message, "get_srcSystem", None)
        source_component = getattr(message, "get_srcComponent", None)
        if callable(source_system) and callable(source_component):
            system = source_system()
            component = source_component()
            if type(system) is int and 1 <= system <= 255:
                self._target_system = system
                transport.connection.target_system = system
            if type(component) is int and 1 <= component <= 255:
                self._target_component = component
                transport.connection.target_component = component

        return PoweredDatagramDispatch(
            source_accepted=True,
            peer_frozen_now=decision.peer_frozen_now,
            rejected_source=False,
            malformed=False,
            production_dispatched=True,
            source_promoted=transport.promoted,
            peer=transport.peer,
            admitted_message_type=admitted_type,
            failure_reason=None,
        )

    def _latch_powered_receiver_failure(self, reason: str) -> None:
        transport = self._powered_transport
        transport.outbound_guards.latch_production(reason)
        if self._powered_failure_reason is None:
            self._powered_failure_reason = reason
        self._powered_failure_event.set()
        self._powered_promotion_event.set()

    def _announce_loop(self) -> None:  # pragma: no cover - live socket loop
        if self._powered_transport is not None:
            self._powered_announce_loop()
            return
        while not self._stop_event.is_set():
            try:
                now_ns = time.time_ns()
                timesync_wire = self._timesync_wire_type(tc1=0, ts1=now_ns)
                heartbeat_wire = self._gcs_heartbeat_wire_type(
                    type=6,
                    autopilot=8,
                    base_mode=0,
                    custom_mode=0,
                    system_status=4,
                )
                with self._send_lock:
                    self._call_nonattitude_locked(
                        category="timesync",
                        api="timesync_send",
                        audit_name="timesync",
                        wire=timesync_wire,
                        call=lambda: self._conn.mav.timesync_send(
                            timesync_wire.tc1,
                            timesync_wire.ts1,
                        ),
                    )
                    self._call_nonattitude_locked(
                        category="gcs_heartbeat",
                        api="heartbeat_send",
                        audit_name="gcs_heartbeat",
                        wire=heartbeat_wire,
                        call=lambda: self._conn.mav.heartbeat_send(
                            heartbeat_wire.type,
                            heartbeat_wire.autopilot,
                            heartbeat_wire.base_mode,
                            heartbeat_wire.custom_mode,
                            heartbeat_wire.system_status,
                        ),
                    )
            except Exception:
                logger.exception("AIGP MAVLink announce failed")
            self._stop_event.wait(0.1)

    def _powered_announce_loop(self) -> None:
        transport = self._powered_transport
        while not self._stop_event.is_set():
            if transport.promoted:
                return
            if transport.peer is None:
                self._stop_event.wait(POWERED_WORKER_POLL_NS / 1_000_000_000.0)
                continue
            try:
                now_ns = time.time_ns()
                timesync_wire = self._timesync_wire_type(tc1=0, ts1=now_ns)
                heartbeat_wire = self._gcs_heartbeat_wire_type(
                    type=6,
                    autopilot=8,
                    base_mode=0,
                    custom_mode=0,
                    system_status=4,
                )
                with self._send_lock:
                    # Promotion can race the outer check. Make the final check
                    # under the same lock that brackets authorization and send.
                    if transport.promoted:
                        return
                    deadline = self._powered_connect_deadline_monotonic_ns
                    self._call_nonattitude_locked(
                        category="timesync",
                        api="timesync_send",
                        audit_name="timesync",
                        wire=timesync_wire,
                        call=lambda: self._conn.mav.timesync_send(
                            timesync_wire.tc1,
                            timesync_wire.ts1,
                        ),
                        powered_deadline_monotonic_ns=deadline,
                    )
                    self._call_nonattitude_locked(
                        category="gcs_heartbeat",
                        api="heartbeat_send",
                        audit_name="gcs_heartbeat",
                        wire=heartbeat_wire,
                        call=lambda: self._conn.mav.heartbeat_send(
                            heartbeat_wire.type,
                            heartbeat_wire.autopilot,
                            heartbeat_wire.base_mode,
                            heartbeat_wire.custom_mode,
                            heartbeat_wire.system_status,
                        ),
                        powered_deadline_monotonic_ns=deadline,
                    )
            except BaseException:
                self._latch_powered_receiver_failure(
                    "powered_announcement_failed"
                )
                return
            self._stop_event.wait(POWERED_WORKER_POLL_NS / 1_000_000_000.0)

    def _time_boot_ms(self) -> int:
        return int(time.monotonic() * 1000) & 0xFFFFFFFF

    def _require_conn(self) -> None:
        if self._conn is None:
            raise RuntimeError("AIGP MAVLink adapter is not connected")


def _clamp_thrust(thrust: float) -> float:
    if not math.isfinite(thrust):
        raise ValueError("thrust must be finite")
    return max(0.0, min(1.0, thrust))


def _monotonic_age(received_at: float) -> float:
    """Return a non-negative stream age, or infinity before first receipt."""

    if received_at <= 0.0:
        return math.inf
    return max(0.0, time.monotonic() - received_at)


def _attitude_error_body_rates(q_cur, q_des, omega=(0.0, 0.0, 0.0),
                               kp=5.0, kd=0.0, max_rate=4.0):
    """PD body-rate command (FRD) that drives q_cur toward q_des.

    The error quaternion ``q_err = conj(q_cur) (x) q_des`` is expressed in the
    body frame; its vector part is ``sin(theta/2)*axis``, so ``2*kp*vec`` is a
    proportional, singularity-free body-rate (euler-error control cross-couples
    when tilted). The ``-kd*omega`` term damps the cascade (sim rate loop + our
    P loop limit-cycles ~5 Hz without it). Shortest-path via ``w >= 0``.
    Returns (roll_rate, pitch_rate, yaw_rate) in FRD, clamped to +/- max_rate.
    The CALLER applies the sim's per-axis rate sign — see __init__._rate_sign.

    ``kp``/``kd`` may be scalars or 3-tuples (per-axis roll/pitch/yaw). PER-AXIS
    gains matter because the sim amplifies the rate channels asymmetrically
    (~1.0x roll vs ~2.1x pitch/yaw, bench-measured), so a single kp leaves the
    ROLL loop at half the closed-loop bandwidth of pitch — roll under-tracks
    (0.46x amplitude, ~0.6s lag) and the cross-track centering oscillates at
    speed. Raising ONLY roll's kp equalises the bandwidth.

    Used because the AIGP sim honors body-rate (mask 128) but spins under
    attitude mode (mask 7) — see AIGPMavlinkAdapter.__init__.
    """
    kpx, kpy, kpz = (kp, kp, kp) if isinstance(kp, (int, float)) else kp
    kdx, kdy, kdz = (kd, kd, kd) if isinstance(kd, (int, float)) else kd
    qc = (q_cur.w, q_cur.x, q_cur.y, q_cur.z)
    qd = (q_des.w, q_des.x, q_des.y, q_des.z)
    # conj(qc) (x) qd
    cw, cx, cy, cz = qc[0], -qc[1], -qc[2], -qc[3]
    ew = cw * qd[0] - cx * qd[1] - cy * qd[2] - cz * qd[3]
    ex = cw * qd[1] + cx * qd[0] + cy * qd[3] - cz * qd[2]
    ey = cw * qd[2] - cx * qd[3] + cy * qd[0] + cz * qd[1]
    ez = cw * qd[3] + cx * qd[2] - cy * qd[1] + cz * qd[0]
    if ew < 0:
        ex, ey, ez = -ex, -ey, -ez
    rates = (
        2.0 * kpx * ex - kdx * omega[0],
        2.0 * kpy * ey - kdy * omega[1],
        2.0 * kpz * ez - kdz * omega[2],
    )
    # max_rate may be a scalar or a 3-tuple (per-axis). ROLL gets more rate
    # headroom (iter-44): at a fast slalom/finish turn the roll command saturates
    # the clamp and builds too slowly (~1 s to 0.8 rad), so achieved roll is only
    # ~0.53x commanded -> under-turn -> over-command -> tumble (gate5 @ base 15.5,
    # gyro 33). A higher ROLL clamp lets the bank build in time; pitch/yaw stay
    # at the proven 0.8 (they were never the bottleneck).
    mxx, mxy, mxz = (max_rate, max_rate, max_rate) if isinstance(
        max_rate, (int, float)) else max_rate
    return (
        max(-mxx, min(mxx, rates[0])),
        max(-mxy, min(mxy, rates[1])),
        max(-mxz, min(mxz, rates[2])),
    )


def _default_telem() -> TelemetryState:
    return TelemetryState(
        timestamp_us=0,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion(),
        angular_velocity=(0.0, 0.0, 0.0),
    )


def _telem_with(old: Optional[TelemetryState], **updates) -> TelemetryState:
    base = old or _default_telem()
    values = {
        "timestamp_us": base.timestamp_us,
        "position_ned": base.position_ned,
        "velocity_ned": base.velocity_ned,
        "orientation": base.orientation,
        "angular_velocity": base.angular_velocity,
        "imu": base.imu,
        "lpn_time_boot_ms": base.lpn_time_boot_ms,
        "odom_time_usec": base.odom_time_usec,
        "odom_quality": base.odom_quality,
        "odom_reset_counter": base.odom_reset_counter,
    }
    for key, value in updates.items():
        if value is not None:
            values[key] = value
    return TelemetryState(**values)


def _iter_records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def main(argv=None) -> None:  # pragma: no cover - live CLI
    parser = argparse.ArgumentParser(description="AIGP pymavlink transport utility")
    parser.add_argument("--record", default=None, help="write JSONL capture to this path")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--jpeg-dir", default=None)
    parser.add_argument("--attitude-test", action="store_true")
    args = parser.parse_args(argv)

    async def _run():
        adapter = AIGPMavlinkAdapter(enable_vision=not args.no_vision)
        await adapter.connect()
        if args.attitude_test:
            await adapter.send_attitude(AttitudeCommand(0.0, 0.0, 0.0, 0.5))
        if args.record:
            records = []
            start = time.monotonic()
            while time.monotonic() - start < args.duration:
                await asyncio.sleep(0.02)
                telem = adapter.latest_telemetry
                if telem is not None:
                    records.append(record_for_message("telemetry_snapshot", {
                        "timestamp_us": telem.timestamp_us,
                        "position_ned": list(telem.position_ned),
                        "velocity_ned": list(telem.velocity_ned),
                    }, time.time_ns()))
            with open(args.record, "w") as out:
                write_jsonl(records, out)
        await adapter.disconnect()

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
