"""Unit tests for the raw pymavlink AIGP transport adapter.

These tests exercise only pure seams: fake MAVLink messages go through
``_handle_message`` and fake ``mav.*_send`` calls capture outgoing command
wires. They do not import pymavlink, bind sockets, or decode vision frames.
"""
import asyncio
import math

import pytest

from competition.adapter import AttitudeCommand, AttitudeRateCommand, PositionCommand
from competition.aigp_messages import (
    ENCAPSULATED_RACE_STATUS_MSG_ID,
    ENCAPSULATED_TRACK_INFO_MSG_ID,
    TrackGate,
    encode_race_status,
    encode_track_data,
)
from competition.adapter import Quaternion
from competition.aigp_mavlink import AIGPMavlinkAdapter, SIM_RESET_COMMAND


class FakeMsg:
    def __init__(self, msg_type, **fields):
        self._msg_type = msg_type
        for key, value in fields.items():
            setattr(self, key, value)

    def get_type(self):
        return self._msg_type


class FakeMav:
    def __init__(self):
        self.calls = []

    def set_attitude_target_send(self, *args):
        self.calls.append(("set_attitude_target_send", args))

    def set_position_target_local_ned_send(self, *args):
        self.calls.append(("set_position_target_local_ned_send", args))

    def command_long_send(self, *args):
        self.calls.append(("command_long_send", args))

    def heartbeat_send(self, *args):
        self.calls.append(("heartbeat_send", args))

    def timesync_send(self, *args):
        self.calls.append(("timesync_send", args))


class FakeConn:
    def __init__(self):
        self.mav = FakeMav()
        self.target_system = 42
        self.target_component = 99
        self.closed = False

    def close(self):
        self.closed = True


def _adapter_with_fake_conn():
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    adapter._conn = FakeConn()
    adapter._target_system = 42
    adapter._target_component = 99
    return adapter


def _track_payload():
    gates = [
        TrackGate(
            gate_id=0,
            position_ned=(-23.3, -0.4, -0.03),
            orientation=Quaternion(w=0.7071, x=0.0, y=0.0, z=0.7071),
            width=2.72,
            height=2.72,
        ),
        TrackGate(
            gate_id=1,
            position_ned=(-46.9, -2.5, 5.07),
            orientation=Quaternion(w=0.7071, x=0.0, y=0.0, z=0.7071),
            width=2.72,
            height=2.72,
        ),
    ]
    return encode_track_data(gates)


def test_track_chunks_reassemble_out_of_order_and_orphans_are_ignored():
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    body = _track_payload()
    transfer_id = 513
    first = bytes([ENCAPSULATED_TRACK_INFO_MSG_ID]) + transfer_id.to_bytes(2, "little") + body[:15]
    second = bytes([ENCAPSULATED_TRACK_INFO_MSG_ID]) + transfer_id.to_bytes(2, "little") + body[15:]

    orphan = bytes([ENCAPSULATED_TRACK_INFO_MSG_ID]) + (999).to_bytes(2, "little") + b"ignored"
    adapter._handle_message(FakeMsg("ENCAPSULATED_DATA", data=orphan, seqnr=0))
    assert adapter.track_data is None

    adapter._handle_message(FakeMsg("DATA_TRANSMISSION_HANDSHAKE", width=transfer_id, packets=2))
    adapter._handle_message(FakeMsg("ENCAPSULATED_DATA", data=second, seqnr=1))
    assert adapter.track_data is None
    adapter._handle_message(FakeMsg("ENCAPSULATED_DATA", data=first, seqnr=0))

    assert adapter.track_data is not None
    assert adapter.track_data.num_gates == 2
    assert adapter.track_data.gates[0].position_ned == pytest.approx((-23.3, -0.4, -0.03), abs=1e-3)


def test_telemetry_population_and_odometry_orientation_wxyz():
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    adapter._handle_message(FakeMsg(
        "ATTITUDE",
        roll=0.0,
        pitch=0.0,
        yaw=0.25,
        rollspeed=0.1,
        pitchspeed=0.2,
        yawspeed=0.3,
        time_boot_ms=123,
    ))
    adapter._handle_message(FakeMsg(
        "LOCAL_POSITION_NED",
        x=1.0,
        y=2.0,
        z=3.0,
        vx=4.0,
        vy=5.0,
        vz=6.0,
        time_boot_ms=124,
    ))
    adapter._handle_message(FakeMsg(
        "HIGHRES_IMU",
        xacc=-1.0,
        yacc=-2.0,
        zacc=-9.81,
        xgyro=0.4,
        ygyro=0.5,
        zgyro=0.6,
        time_usec=125000,
    ))

    telem = adapter.latest_telemetry
    assert telem is not None
    assert telem.timestamp_us == 124000
    assert telem.position_ned == pytest.approx((1.0, 2.0, 3.0))
    assert telem.velocity_ned == pytest.approx((4.0, 5.0, 6.0))
    assert telem.angular_velocity == pytest.approx((0.1, 0.2, 0.3))
    assert telem.imu.timestamp_us == 125000
    assert telem.imu.accel == pytest.approx((-1.0, -2.0, -9.81))
    assert telem.orientation.to_euler()[2] == pytest.approx(0.25)

    adapter._handle_message(FakeMsg(
        "ODOMETRY",
        q=[0.70710678, 0.0, 0.0, 0.70710678],
        x=999.0,
        y=999.0,
        z=999.0,
        vx=999.0,
        vy=999.0,
        vz=999.0,
        quality=100,
        reset_counter=7,
        time_usec=130000,
    ))
    telem = adapter.latest_telemetry
    assert telem.timestamp_us == 130000
    assert telem.position_ned == pytest.approx((1.0, 2.0, 3.0))
    assert telem.velocity_ned == pytest.approx((4.0, 5.0, 6.0))
    assert telem.orientation.to_euler()[2] == pytest.approx(math.pi / 2, abs=1e-6)
    assert telem.odom_quality == 100
    assert telem.odom_reset_counter == 7
    assert telem.odom_time_usec == 130000
    assert telem.lpn_time_boot_ms == 124


def test_side_state_race_status_collision_actuator_and_heartbeat():
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    race_payload = encode_race_status(
        sim_boot_time_ms=1000,
        race_start_boot_time_ms=-1,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    adapter._handle_message(FakeMsg("ENCAPSULATED_DATA", data=race_payload, seqnr=0))
    adapter._handle_message(FakeMsg(
        "COLLISION",
        id=1001,
        threat_level=2,
        horizontal_minimum_delta=3.5,
    ))
    adapter._handle_message(FakeMsg(
        "ACTUATOR_OUTPUT_STATUS",
        time_usec=50,
        active=15,
        actuator=[0.1, 0.2, 0.3, 0.4] + [0.0] * 28,
    ))
    adapter._handle_message(FakeMsg("HEARTBEAT", base_mode=193, custom_mode=0))

    assert adapter.race_status.active_gate_index == 0
    assert adapter.drain_collisions()[0]["impulse"] == pytest.approx(3.5)
    assert adapter.drain_collisions() == []
    assert adapter.actuator_outputs["active"] == 15
    assert adapter.is_armed is True

    adapter._handle_message(FakeMsg("HEARTBEAT", base_mode=65, custom_mode=0))
    assert adapter.is_armed is False


def test_send_attitude_rate_position_reset_and_arm_wires():
    adapter = _adapter_with_fake_conn()

    asyncio.run(adapter.send_attitude(AttitudeCommand(
        roll_rad=0.1,
        pitch_rad=-0.2,
        yaw_rad=0.3,
        thrust=1.5,
    )))
    name, args = adapter._conn.mav.calls[-1]
    assert name == "set_attitude_target_send"
    assert args[3] == 0b00000111
    assert args[4] == pytest.approx(tuple(Quaternion.from_euler(0.1, -0.2, 0.3).__dict__.values()))
    assert args[5:8] == (0.0, 0.0, 0.0)
    assert args[8] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="finite"):
        asyncio.run(adapter.send_attitude(AttitudeCommand(0.0, 0.0, 0.0, float("nan"))))

    asyncio.run(adapter.send_attitude_rate(AttitudeRateCommand(0.4, 0.5, 0.6, -1.0)))
    name, args = adapter._conn.mav.calls[-1]
    assert name == "set_attitude_target_send"
    assert args[3] == 128
    assert args[4] == [1.0, 0.0, 0.0, 0.0]
    assert args[5:8] == (0.4, 0.5, 0.6)
    assert args[8] == pytest.approx(0.0)

    asyncio.run(adapter.send_position(PositionCommand(
        position_ned=(1.0, 2.0, 3.0),
        velocity_ned=(4.0, 5.0, 6.0),
        yaw_rad=0.7,
    )))
    name, args = adapter._conn.mav.calls[-1]
    assert name == "set_position_target_local_ned_send"
    assert args[3] == 1
    assert args[4] == 2496
    assert args[5:11] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert args[14] == pytest.approx(0.7)

    asyncio.run(adapter._send_sim_reset())
    name, args = adapter._conn.mav.calls[-1]
    assert name == "command_long_send"
    assert args[2] == SIM_RESET_COMMAND
    assert args[4:] == (0, 0, 0, 0, 0, 0, 0)

    adapter._armed = True
    asyncio.run(adapter.arm())
    assert adapter._conn.mav.calls[-1][1][2] == SIM_RESET_COMMAND

    adapter._armed = False
    asyncio.run(adapter.arm())
    name, args = adapter._conn.mav.calls[-1]
    assert name == "command_long_send"
    assert args[2] == 400
    assert args[4] == 1


def test_import_does_not_require_pymavlink():
    import competition.aigp_mavlink as aigp_mavlink

    assert hasattr(aigp_mavlink, "AIGPMavlinkAdapter")


def test_connect_is_idempotent():
    """Second call to connect() must be a no-op: no new threads, keep existing _track_data."""
    adapter = AIGPMavlinkAdapter(enable_vision=False, require_track=False)
    fake_conn = FakeConn()
    adapter._conn = fake_conn
    adapter._track_event.set()

    body = _track_payload()
    transfer_id = 7
    adapter._handle_message(FakeMsg("DATA_TRANSMISSION_HANDSHAKE", width=transfer_id, packets=2))
    part1 = bytes([ENCAPSULATED_TRACK_INFO_MSG_ID]) + transfer_id.to_bytes(2, "little") + body[:15]
    part2 = bytes([ENCAPSULATED_TRACK_INFO_MSG_ID]) + transfer_id.to_bytes(2, "little") + body[15:]
    adapter._handle_message(FakeMsg("ENCAPSULATED_DATA", data=part1, seqnr=0))
    adapter._handle_message(FakeMsg("ENCAPSULATED_DATA", data=part2, seqnr=1))
    first_track = adapter.track_data
    assert first_track is not None

    original_conn = adapter._conn

    asyncio.run(adapter.connect("udpin:127.0.0.1:9999"))

    assert adapter._conn is original_conn, "second connect() must not replace existing connection"
    assert adapter.track_data is first_track, "second connect() must not clear existing track data"
    assert adapter._rx_thread is None, "second connect() must not spawn a new rx thread"
