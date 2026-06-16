"""Pure parsers for the AIGP simulator's custom MAVLink payloads.

The DCL/AIGP sim carries two competition-specific payloads inside MAVLink
``ENCAPSULATED_DATA`` messages, plus a ``COLLISION`` message. The byte
layouts here are taken verbatim from the official ``PyAIPilotExample``
(``mavlink_rx.py``, 2026-05-28) so our transport layer decodes the real
wire format rather than an assumed one:

  * Race status  — ``"<BQqqIq"`` =
        data_type(=1), sim_boot_time_ms, race_start_boot_time_ms,
        race_finish_time_ns, active_gate_index, last_gate_race_time
  * Track header — ``"<H"`` = num_gates
  * Track gate   — ``"<Hfffffffff"`` =
        gate_id, pos_ned_x, pos_ned_y, pos_ned_z,
        orient_w, orient_x, orient_y, orient_z, width, height

These functions are deliberately transport-free (stdlib ``struct`` only) so
they unit-test without pymavlink, a socket, or the live sim. The live
adapter/recorder strips the per-chunk ``"<BH"`` (data_type, transfer_id)
header and reassembles chunks via :class:`TrackInfoReassembler` before
calling :func:`parse_track_data`.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from competition.adapter import Quaternion

# ENCAPSULATED_DATA payload type IDs (first byte of the payload).
ENCAPSULATED_RACE_STATUS_MSG_ID: int = 1
ENCAPSULATED_TRACK_INFO_MSG_ID: int = 2

# COLLISION.id values (handled live by the transport; codified here for reuse).
COLLISION_ID_GATE: int = 1001
COLLISION_ID_ENVIRONMENT: int = 1002

# Struct layouts (little-endian, no padding) — verbatim from mavlink_rx.py.
RACE_STATUS_STRUCT: str = "<BQqqIq"
RACE_STATUS_SIZE: int = struct.calcsize(RACE_STATUS_STRUCT)  # 37

TRACK_HEADER_STRUCT: str = "<H"
TRACK_HEADER_SIZE: int = struct.calcsize(TRACK_HEADER_STRUCT)  # 2

TRACK_GATE_STRUCT: str = "<Hfffffffff"
TRACK_GATE_SIZE: int = struct.calcsize(TRACK_GATE_STRUCT)  # 38


# ---------------------------------------------------------------------------
# Race status
# ---------------------------------------------------------------------------
@dataclass
class RaceStatus:
    """Decoded race-status payload.

    ``race_start_boot_time_ms`` is < 0 until the race starts;
    ``race_finish_time_ns`` is < 0 while the race is ongoing (per the
    official template's field semantics).
    """

    sim_boot_time_ms: int
    race_start_boot_time_ms: int
    race_finish_time_ns: int
    active_gate_index: int
    last_gate_race_time: int

    @property
    def race_started(self) -> bool:
        """A GO time has been SCHEDULED. NOTE: this is True ~0.6 s after a
        SIM_RESET — it does NOT mean the race has begun. The sim runs a 3 s
        countdown after the reset and only starts the race when the race clock
        reaches the scheduled GO time. Use :pyattr:`race_underway` to know the
        race has actually started; gating flight on ``race_started`` jumps the
        start (commands during the countdown) and is DISQUALIFIED."""
        return self.race_start_boot_time_ms >= 0

    @property
    def race_underway(self) -> bool:
        """The 3 s countdown has elapsed and the race has actually STARTED: the
        sim race clock (``sim_boot_time_ms``, reset to ~0 by SIM_RESET) has
        reached the scheduled GO time (``race_start_boot_time_ms`` ~3300 ms).
        This is the authoritative "safe to fly" signal. CAVEAT: right after a
        reset the status can briefly be STALE (a pre-reset frame with a large
        ``sim_boot_time_ms``), which would read True spuriously — callers must
        confirm they are in the fresh post-reset countdown first (see
        ``_reset_and_settle``)."""
        return (
            self.race_start_boot_time_ms >= 0
            and self.sim_boot_time_ms >= self.race_start_boot_time_ms
        )

    @property
    def race_finished(self) -> bool:
        return self.race_finish_time_ns >= 0


def parse_race_status(payload: bytes) -> RaceStatus:
    """Decode a race-status ENCAPSULATED_DATA payload (incl. its type byte)."""
    if len(payload) < RACE_STATUS_SIZE:
        raise ValueError(
            f"race-status payload too short: got {len(payload)} bytes, "
            f"need >= {RACE_STATUS_SIZE}"
        )
    (data_type, sim_boot, start, finish, gate_idx, last) = struct.unpack_from(
        RACE_STATUS_STRUCT, payload, 0
    )
    if data_type != ENCAPSULATED_RACE_STATUS_MSG_ID:
        raise ValueError(
            f"not a race-status payload: data_type={data_type}, "
            f"expected {ENCAPSULATED_RACE_STATUS_MSG_ID}"
        )
    return RaceStatus(
        sim_boot_time_ms=sim_boot,
        race_start_boot_time_ms=start,
        race_finish_time_ns=finish,
        active_gate_index=gate_idx,
        last_gate_race_time=last,
    )


def encode_race_status(
    sim_boot_time_ms: int,
    race_start_boot_time_ms: int,
    race_finish_time_ns: int,
    active_gate_index: int,
    last_gate_race_time: int,
) -> bytes:
    """Build a race-status payload (with the leading type byte). For tests
    and local mock senders."""
    return struct.pack(
        RACE_STATUS_STRUCT,
        ENCAPSULATED_RACE_STATUS_MSG_ID,
        sim_boot_time_ms,
        race_start_boot_time_ms,
        race_finish_time_ns,
        active_gate_index,
        last_gate_race_time,
    )


# ---------------------------------------------------------------------------
# Track data (the gate map, in NED)
# ---------------------------------------------------------------------------
@dataclass
class TrackGate:
    """One gate as delivered by the sim's track-info packet (NED frame)."""

    gate_id: int
    position_ned: Tuple[float, float, float]
    orientation: Quaternion  # (w, x, y, z)
    width: float
    height: float


@dataclass
class TrackData:
    """The full set of gates for a course, in delivery order."""

    gates: List[TrackGate] = field(default_factory=list)

    @property
    def num_gates(self) -> int:
        return len(self.gates)


def parse_track_data(payload: bytes) -> TrackData:
    """Decode a reassembled track-info payload (starts at ``num_gates``).

    The per-chunk ``"<BH"`` header and chunk reassembly are handled upstream
    (see :class:`TrackInfoReassembler`); this operates on the assembled body.
    """
    if len(payload) < TRACK_HEADER_SIZE:
        raise ValueError(
            f"track payload too short for header: got {len(payload)} bytes, "
            f"need >= {TRACK_HEADER_SIZE}"
        )
    (num_gates,) = struct.unpack_from(TRACK_HEADER_STRUCT, payload, 0)
    required = TRACK_HEADER_SIZE + num_gates * TRACK_GATE_SIZE
    if len(payload) < required:
        raise ValueError(
            f"track payload truncated: header claims {num_gates} gates "
            f"(need {required} bytes), got {len(payload)}"
        )
    gates: List[TrackGate] = []
    offset = TRACK_HEADER_SIZE
    for _ in range(num_gates):
        (gid, x, y, z, qw, qx, qy, qz, w, h) = struct.unpack_from(
            TRACK_GATE_STRUCT, payload, offset
        )
        gates.append(
            TrackGate(
                gate_id=gid,
                position_ned=(x, y, z),
                orientation=Quaternion(w=qw, x=qx, y=qy, z=qz),
                width=w,
                height=h,
            )
        )
        offset += TRACK_GATE_SIZE
    return TrackData(gates=gates)


def encode_track_data(gates: Sequence[TrackGate]) -> bytes:
    """Build a track-info body (``num_gates`` + gate blocks). For tests and
    local mock senders."""
    out = bytearray(struct.pack(TRACK_HEADER_STRUCT, len(gates)))
    for g in gates:
        out += struct.pack(
            TRACK_GATE_STRUCT,
            g.gate_id,
            g.position_ned[0],
            g.position_ned[1],
            g.position_ned[2],
            g.orientation.w,
            g.orientation.x,
            g.orientation.y,
            g.orientation.z,
            g.width,
            g.height,
        )
    return bytes(out)


class TrackInfoReassembler:
    """Reassemble a chunked track-info transfer.

    Mirrors the official template: a ``DATA_TRANSMISSION_HANDSHAKE`` opens a
    transfer (``begin_transfer``) declaring the chunk count, then each
    ``ENCAPSULATED_DATA`` track chunk (after its ``"<BH"`` header is stripped)
    is fed via ``feed_chunk``. When the last chunk arrives the assembled body
    is parsed and returned as :class:`TrackData`.
    """

    def __init__(self) -> None:
        self._chunks: Dict[int, Dict[int, bytes]] = {}
        self._expected: Dict[int, int] = {}

    def begin_transfer(self, transfer_id: int, num_chunks: int) -> None:
        self._chunks[transfer_id] = {}
        self._expected[transfer_id] = num_chunks

    def feed_chunk(
        self, transfer_id: int, seqnr: int, chunk: bytes
    ) -> Optional[TrackData]:
        """Add one chunk. Returns parsed TrackData once complete, else None.

        Chunks for an unknown transfer (no prior ``begin_transfer``) are
        ignored — matching the template's defensive behavior.
        """
        if transfer_id not in self._expected:
            return None
        self._chunks[transfer_id][seqnr] = chunk
        if len(self._chunks[transfer_id]) < self._expected[transfer_id]:
            return None
        body = b"".join(
            self._chunks[transfer_id][i]
            for i in range(self._expected[transfer_id])
        )
        del self._chunks[transfer_id]
        del self._expected[transfer_id]
        return parse_track_data(body)
