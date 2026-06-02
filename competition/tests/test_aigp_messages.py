"""Tests for competition.aigp_messages — pure parsers for the AIGP sim's
custom MAVLink ENCAPSULATED_DATA payloads (race status + track info).

Byte formats are taken verbatim from the official PyAIPilotExample/mavlink_rx.py:
  race status:  struct "<BQqqIq" = data_type, sim_boot_ms, race_start_ms,
                race_finish_ns, active_gate_index, last_gate_race_time
  track header: "<H"  = num_gates
  track gate:   "<Hfffffffff" = gate_id, x, y, z, qw, qx, qy, qz, width, height
"""
import struct

import pytest

from competition.adapter import Quaternion
from competition.aigp_messages import (
    ENCAPSULATED_RACE_STATUS_MSG_ID,
    ENCAPSULATED_TRACK_INFO_MSG_ID,
    RaceStatus,
    TrackData,
    TrackGate,
    TrackInfoReassembler,
    encode_race_status,
    encode_track_data,
    parse_race_status,
    parse_track_data,
)


def _quat(w, x, y, z):
    return Quaternion(w=w, x=x, y=y, z=z)


# --------------------------------------------------------------------------
# Race status
# --------------------------------------------------------------------------
def test_race_status_round_trip_running():
    payload = encode_race_status(
        sim_boot_time_ms=12345,
        race_start_boot_time_ms=1000,
        race_finish_time_ns=-1,
        active_gate_index=3,
        last_gate_race_time=7,
    )
    rs = parse_race_status(payload)
    assert rs.sim_boot_time_ms == 12345
    assert rs.race_start_boot_time_ms == 1000
    assert rs.race_finish_time_ns == -1
    assert rs.active_gate_index == 3
    assert rs.last_gate_race_time == 7
    assert rs.race_started is True
    assert rs.race_finished is False


def test_race_status_first_byte_is_msg_id():
    payload = encode_race_status(
        sim_boot_time_ms=1,
        race_start_boot_time_ms=-1,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    assert payload[0] == ENCAPSULATED_RACE_STATUS_MSG_ID


def test_race_status_not_started_and_ongoing():
    rs = parse_race_status(
        encode_race_status(
            sim_boot_time_ms=500,
            race_start_boot_time_ms=-1,
            race_finish_time_ns=-1,
            active_gate_index=0,
            last_gate_race_time=-1,
        )
    )
    assert rs.race_started is False
    assert rs.race_finished is False


def test_race_status_finished():
    rs = parse_race_status(
        encode_race_status(
            sim_boot_time_ms=99999,
            race_start_boot_time_ms=1000,
            race_finish_time_ns=88888,
            active_gate_index=9,
            last_gate_race_time=42,
        )
    )
    assert rs.race_finished is True


def test_race_status_rejects_wrong_msg_id():
    # data_type byte = 2 (track info), not 1 (race status)
    bad = bytes([ENCAPSULATED_TRACK_INFO_MSG_ID]) + struct.pack(
        "<QqqIq", 1, -1, -1, 0, -1
    )
    with pytest.raises(ValueError):
        parse_race_status(bad)


def test_race_status_rejects_short_payload():
    with pytest.raises(ValueError):
        parse_race_status(b"\x01\x02\x03")


# --------------------------------------------------------------------------
# Track data
# --------------------------------------------------------------------------
def test_track_data_empty():
    td = parse_track_data(encode_track_data([]))
    assert td.num_gates == 0
    assert td.gates == []


def test_track_data_single_gate_round_trip():
    g = TrackGate(
        gate_id=0,
        position_ned=(8.0, 0.0, -1.5),
        orientation=_quat(1.0, 0.0, 0.0, 0.0),
        width=1.5,
        height=1.5,
    )
    td = parse_track_data(encode_track_data([g]))
    assert td.num_gates == 1
    out = td.gates[0]
    assert out.gate_id == 0
    assert out.position_ned == pytest.approx((8.0, 0.0, -1.5))
    assert (
        out.orientation.w,
        out.orientation.x,
        out.orientation.y,
        out.orientation.z,
    ) == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert out.width == pytest.approx(1.5)
    assert out.height == pytest.approx(1.5)


def test_track_data_multi_gate_preserves_order_and_fields():
    gates = [
        TrackGate(
            gate_id=i,
            position_ned=(float(i), 2.0 * i, -1.0),
            orientation=_quat(1, 0, 0, 0),
            width=1.5,
            height=1.5,
        )
        for i in range(5)
    ]
    td = parse_track_data(encode_track_data(gates))
    assert td.num_gates == 5
    assert [g.gate_id for g in td.gates] == [0, 1, 2, 3, 4]
    assert td.gates[3].position_ned == pytest.approx((3.0, 6.0, -1.0))


def test_track_data_rejects_truncated_gate_block():
    good = encode_track_data(
        [TrackGate(0, (0, 0, 0), _quat(1, 0, 0, 0), 1.5, 1.5)]
    )
    # header claims 2 gates but only one gate's bytes follow
    truncated = struct.pack("<H", 2) + good[2:]
    with pytest.raises(ValueError):
        parse_track_data(truncated)


def test_track_data_rejects_empty_payload():
    with pytest.raises(ValueError):
        parse_track_data(b"")


# --------------------------------------------------------------------------
# Track-info chunk reassembly (DATA_TRANSMISSION_HANDSHAKE + chunks)
# --------------------------------------------------------------------------
def test_reassembler_single_chunk():
    full = encode_track_data([TrackGate(0, (8, 0, -1.5), _quat(1, 0, 0, 0), 1.5, 1.5)])
    r = TrackInfoReassembler()
    r.begin_transfer(transfer_id=7, num_chunks=1)
    td = r.feed_chunk(transfer_id=7, seqnr=0, chunk=full)
    assert td is not None
    assert td.num_gates == 1


def test_reassembler_multi_chunk_completes_in_order():
    gates = [TrackGate(i, (float(i), 0, -1.5), _quat(1, 0, 0, 0), 1.5, 1.5) for i in range(3)]
    full = encode_track_data(gates)
    mid = len(full) // 2
    r = TrackInfoReassembler()
    r.begin_transfer(transfer_id=1, num_chunks=2)
    assert r.feed_chunk(1, 0, full[:mid]) is None  # incomplete
    td = r.feed_chunk(1, 1, full[mid:])
    assert td is not None
    assert td.num_gates == 3


def test_reassembler_ignores_chunk_for_unknown_transfer():
    r = TrackInfoReassembler()
    # no begin_transfer first → unknown transfer_id is ignored, returns None
    assert r.feed_chunk(transfer_id=99, seqnr=0, chunk=b"\x00\x00") is None
