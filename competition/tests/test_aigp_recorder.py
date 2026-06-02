"""Tests for competition.aigp_recorder's pure core — the JSONL record schema,
MAVLink message-field normalization, and the JSONL writer.

The live socket/pymavlink loop (`run`) is validated on first contact against
the real sim, not in CI (there's no MAVLink endpoint here). These tests cover
everything that does NOT need a socket.
"""
import io
import json
from types import SimpleNamespace

from competition.aigp_messages import encode_race_status
from competition.aigp_recorder import (
    mavlink_msg_to_fields,
    race_status_fields,
    record_for_message,
    write_jsonl,
)


def test_record_for_message_schema():
    r = record_for_message("ATTITUDE", {"roll": 0.1}, recv_wall_ns=42)
    assert r == {"type": "ATTITUDE", "recv_wall_ns": 42, "roll": 0.1}


def test_race_status_fields_from_payload():
    payload = encode_race_status(
        sim_boot_time_ms=100,
        race_start_boot_time_ms=10,
        race_finish_time_ns=-1,
        active_gate_index=2,
        last_gate_race_time=5,
    )
    f = race_status_fields(payload)
    assert f["sim_boot_time_ms"] == 100
    assert f["active_gate_index"] == 2
    assert f["race_started"] is True
    assert f["race_finished"] is False


def test_mavlink_fields_local_position_ned():
    # This is the message whose population we most need to confirm on first
    # contact (no-GPS sim — does it still publish local position?).
    msg = SimpleNamespace(
        x=1.0, y=2.0, z=-3.0, vx=0.1, vy=0.2, vz=0.3, time_boot_ms=1234,
        get_type=lambda: "LOCAL_POSITION_NED",
    )
    assert mavlink_msg_to_fields(msg) == {
        "x": 1.0, "y": 2.0, "z": -3.0,
        "vx": 0.1, "vy": 0.2, "vz": 0.3, "time_boot_ms": 1234,
    }


def test_mavlink_fields_attitude():
    msg = SimpleNamespace(
        roll=0.1, pitch=0.2, yaw=0.3,
        rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0, time_boot_ms=7,
        get_type=lambda: "ATTITUDE",
    )
    f = mavlink_msg_to_fields(msg)
    assert f["roll"] == 0.1 and f["yaw"] == 0.3 and f["time_boot_ms"] == 7


def test_mavlink_fields_collision():
    msg = SimpleNamespace(
        id=1001, threat_level=2, horizontal_minimum_delta=0.5,
        get_type=lambda: "COLLISION",
    )
    f = mavlink_msg_to_fields(msg)
    assert f["id"] == 1001 and f["threat_level"] == 2 and f["impulse"] == 0.5


def test_mavlink_fields_unknown_type_is_empty():
    # Unknown types are still recorded (by their type label) so the
    # first-contact log shows everything the sim emits, with no fields.
    msg = SimpleNamespace(get_type=lambda: "SYS_STATUS")
    assert mavlink_msg_to_fields(msg) == {}


def test_write_jsonl_one_object_per_line():
    buf = io.StringIO()
    n = write_jsonl(
        [{"type": "A", "recv_wall_ns": 1}, {"type": "B", "recv_wall_ns": 2}], buf
    )
    assert n == 2
    lines = buf.getvalue().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "A"
    assert json.loads(lines[1])["type"] == "B"
