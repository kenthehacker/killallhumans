"""Tests for competition.aigp_recorder's pure core — the JSONL record schema,
MAVLink message-field normalization, track-data + vision-frame records, and
the JSONL writer.

The live socket/pymavlink loop (`run`) and the vision-capture thread are
validated on first contact against the real sim, not in CI (there's no
MAVLink/UDP endpoint here). These tests cover everything that does NOT need a
socket.
"""
import io
import json
from types import SimpleNamespace

from competition.adapter import Quaternion
from competition.aigp_messages import TrackData, TrackGate, encode_race_status
from competition.aigp_recorder import (
    mavlink_msg_to_fields,
    race_status_fields,
    record_for_message,
    track_data_fields,
    vision_frame_record,
    write_jsonl,
)
from competition.vision_udp import ReassembledFrame


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


def test_mavlink_fields_unknown_type_dumps_all_fields():
    # A first-contact recorder must NOT silently drop the fields of a message
    # it doesn't explicitly model — that's the whole point of the log.
    # Unmodeled types fall back to the message's full to_dict() (minus the
    # redundant mavpackettype tag).
    msg = SimpleNamespace(
        get_type=lambda: "SYS_STATUS",
        to_dict=lambda: {
            "mavpackettype": "SYS_STATUS",
            "onboard_control_sensors_present": 1234,
            "voltage_battery": 11100,
        },
    )
    assert mavlink_msg_to_fields(msg) == {
        "onboard_control_sensors_present": 1234,
        "voltage_battery": 11100,
    }


def test_mavlink_fields_modeled_type_missing_attr_falls_back():
    # If the real sim's message lacks a field our curated extractor assumes
    # (a wrong field-name guess), fall back to the generic dump rather than
    # crash mid-capture. This COLLISION mock is missing horizontal_minimum_delta.
    msg = SimpleNamespace(
        id=1001, threat_level=2,
        get_type=lambda: "COLLISION",
        to_dict=lambda: {
            "mavpackettype": "COLLISION",
            "id": 1001, "threat_level": 2, "time_usec": 999,
        },
    )
    assert mavlink_msg_to_fields(msg) == {
        "id": 1001, "threat_level": 2, "time_usec": 999,
    }


def test_track_data_fields_captures_full_gate_map():
    # The gate map is a primary §4 question — log every gate's pose, not just
    # the count.
    track = TrackData(gates=[
        TrackGate(
            gate_id=3,
            position_ned=(1.0, 2.0, -3.0),
            orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
            width=1.5, height=1.5,
        ),
    ])
    f = track_data_fields(track)
    assert f["num_gates"] == 1
    assert f["gates"][0] == {
        "gate_id": 3,
        "position_ned": [1.0, 2.0, -3.0],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "width": 1.5, "height": 1.5,
    }


def test_vision_frame_record_schema():
    frame = ReassembledFrame(frame_id=7, jpeg_bytes=b"\xff\xd8\xff\xd9", sim_time_ns=123)
    r = vision_frame_record(frame, recv_wall_ns=42)
    assert r == {
        "type": "vision_frame", "recv_wall_ns": 42,
        "frame_id": 7, "sim_time_ns": 123, "jpeg_size": 4,
    }


def test_vision_frame_record_with_jpeg_path():
    frame = ReassembledFrame(frame_id=7, jpeg_bytes=b"abc", sim_time_ns=1)
    r = vision_frame_record(frame, recv_wall_ns=9, jpeg_path="frames/frame_00000007.jpg")
    assert r["jpeg_path"] == "frames/frame_00000007.jpg"
    assert r["jpeg_size"] == 3


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


def test_write_jsonl_is_json_safe_for_bytes():
    # to_dict() of some MAVLink messages carries bytes/bytearray array fields;
    # the writer must not crash on them.
    buf = io.StringIO()
    n = write_jsonl([{"type": "X", "recv_wall_ns": 1, "data": b"\x01\x02"}], buf)
    assert n == 1
    rec = json.loads(buf.getvalue().splitlines()[0])
    assert rec["data"] == [1, 2]
