"""Unit tests for the raw pymavlink AIGP transport adapter.

These tests exercise only pure seams: fake MAVLink messages go through
``_handle_message`` and fake ``mav.*_send`` calls capture outgoing command
wires. They do not import pymavlink, bind sockets, or decode vision frames.
"""
import asyncio
import math
import socket
import threading
import time
from dataclasses import FrozenInstanceError

import pytest

from competition.adapter import AttitudeCommand, AttitudeRateCommand, PositionCommand
from competition.aigp_messages import (
    ENCAPSULATED_RACE_STATUS_MSG_ID,
    ENCAPSULATED_TRACK_INFO_MSG_ID,
    TrackGate,
    encode_race_status,
    encode_track_data,
)
from competition.adapter import Quaternion, TelemetryState
from competition.aigp_mavlink import (
    AIGPMavlinkAdapter,
    POWERED_OUTBOUND_CALL_NS,
    POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP,
    POWERED_WORKER_POLL_NS,
    PoweredMavlinkTransport,
    SIM_RESET_COMMAND,
)
from competition.vq2_capture import (
    AttitudeTargetOutboundV1,
    NonAttitudeOutboundV1,
    ReceivedActuatorOutputStatusV1,
    ReceivedHeartbeatV1,
    ReceivedIMUSampleV1,
    ReceivedRaceStatusV1,
)
from scripts.aigp_vq2_powered_runtime import (
    ExclusiveUdpEndpoint,
    OutboundAuthorityError,
    PoweredDeadlineExpired,
    PoweredOutboundGuards,
)


def _default_telem_identity() -> TelemetryState:
    """Telemetry with identity orientation + zero gyro for rate-control tests."""
    return TelemetryState(
        timestamp_us=0,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )


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


def _ephemeral_loopback_port():
    candidate = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        candidate.bind(("127.0.0.1", 0))
        return candidate.getsockname()[1]
    finally:
        candidate.close()


class FakeExclusiveRawSocket:
    family = socket.AF_INET
    type = socket.SOCK_DGRAM

    def __init__(self, port=None):
        self.port = port or _ephemeral_loopback_port()
        self.closed = False
        self.sent = []
        self.events = []
        self._condition = threading.Condition()
        self._incoming = []
        self._timeout = None

    def getsockname(self):
        return ("127.0.0.1", self.port)

    def getsockopt(self, _level, option):
        if option == socket.SO_REUSEADDR:
            return 0
        return 1

    def push(self, raw, peer):
        with self._condition:
            self._incoming.append((bytes(raw), peer))
            self._condition.notify_all()

    def recvfrom(self, _capacity):
        with self._condition:
            deadline = (
                None
                if self._timeout is None
                else time.monotonic() + self._timeout
            )
            while not self._incoming and not self.closed:
                if deadline is None:
                    self._condition.wait(0.05)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise socket.timeout("fake receive timeout")
                    self._condition.wait(remaining)
            if self.closed:
                raise OSError("fake socket closed")
            return self._incoming.pop(0)

    def gettimeout(self):
        return self._timeout

    def settimeout(self, value):
        self._timeout = value

    def sendto(self, raw, peer):
        if self.closed:
            raise OSError("fake socket closed")
        self.sent.append((bytes(raw), peer))
        self.events.append(("sendto", peer))
        return len(raw)

    def close(self):
        with self._condition:
            self.events.append(("close",))
            self.closed = True
            self._condition.notify_all()


def _powered_v1_frame(message_id, payload=b""):
    return (
        bytes((0xFE, len(payload), 1, 1, 1, message_id))
        + bytes(payload)
        + b"\xaa\x55"
    )


def _powered_v2_frame(message_id, payload=b""):
    return (
        bytes(
            (
                0xFD,
                len(payload),
                0,
                0,
                1,
                1,
                1,
                message_id & 0xFF,
                (message_id >> 8) & 0xFF,
                (message_id >> 16) & 0xFF,
            )
        )
        + bytes(payload)
        + b"\xaa\x55"
    )


class ScratchOnlyMessage:
    def __init__(self, raw):
        self.raw = bytes(raw)
        self.base_mode = 193

    def get_type(self):
        return "SCRATCH_ONLY"

    def get_msgbuf(self):
        return self.raw

    def get_msgId(self):
        if self.raw[0] == 0xFD:
            return self.raw[7] | (self.raw[8] << 8) | (self.raw[9] << 16)
        return self.raw[5]


class ScratchOnlyParser:
    def __init__(self, parsed_raw):
        self.robust_parsing = True
        self.parsed_raw = parsed_raw

    def parse_buffer(self, raw):
        if bytes(raw[-2:]) != b"\xaa\x55":
            raise ValueError("bad fake checksum")
        self.parsed_raw.append(bytes(raw))
        return [ScratchOnlyMessage(raw)]


class ProductionRawMessage(FakeMsg):
    def __init__(self, raw, msg_type, **fields):
        super().__init__(msg_type, **fields)
        self._raw = bytes(raw)

    def get_msgbuf(self):
        return self._raw

    def get_srcSystem(self):
        return 1

    def get_srcComponent(self):
        return 1


class FakeProductionMavlink:
    def __init__(self, writer):
        self.file = writer
        self.robust_parsing = True
        self.parsed_raw = []
        self.calls = []

    def parse_buffer(self, raw):
        raw = bytes(raw)
        self.parsed_raw.append(raw)
        message_id = raw[5]
        payload = raw[6:-2]
        if message_id == 0:
            message = ProductionRawMessage(
                raw,
                "HEARTBEAT",
                base_mode=65,
                custom_mode=9,
            )
        elif message_id == 105:
            message = ProductionRawMessage(
                raw,
                "HIGHRES_IMU",
                xacc=1.0,
                yacc=2.0,
                zacc=-9.0,
                xgyro=0.1,
                ygyro=0.2,
                zgyro=0.3,
                time_usec=700,
            )
        elif message_id == 131:
            message = ProductionRawMessage(
                raw,
                "ENCAPSULATED_DATA",
                data=payload,
                seqnr=0,
            )
        else:
            message = ProductionRawMessage(raw, "STATUSTEXT", text="ignored", severity=6)
        return [message]

    def _send(self, name, args):
        self.calls.append((name, args))
        self.file.write((name + ":wire").encode("ascii"))

    def set_attitude_target_send(self, *args):
        self._send("set_attitude_target_send", args)

    def command_long_send(self, *args):
        self._send("command_long_send", args)

    def heartbeat_send(self, *args):
        self._send("heartbeat_send", args)

    def timesync_send(self, *args):
        self._send("timesync_send", args)


def _powered_adapter(
    *,
    authority=None,
    receive_mode="worker",
    external_cleanup_authorize=None,
    monotonic_ns=None,
):
    authority = authority or {
        "role_valid": True,
        "parent_alive": True,
        "lease_valid": True,
    }
    raw_socket = FakeExclusiveRawSocket()
    endpoint = ExclusiveUdpEndpoint(
        socket=raw_socket,
        requested_host="127.0.0.1",
        requested_port=0,
        actual_host="127.0.0.1",
        actual_port=raw_socket.port,
        exclusive_option=1,
    )
    guards = PoweredOutboundGuards()
    scratch_parsed = []
    production = []
    transport = PoweredMavlinkTransport(
        endpoint,
        scratch_parser_factory=lambda: ScratchOnlyParser(scratch_parsed),
        mavlink_factory=lambda writer: production.append(
            FakeProductionMavlink(writer)
        ) or production[-1],
        outbound_guards=guards,
        role_valid=lambda: authority["role_valid"],
        parent_alive=lambda: authority["parent_alive"],
        lease_valid=lambda: authority["lease_valid"],
        external_cleanup_authorize=external_cleanup_authorize,
    )
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
        powered_transport=transport,
        powered_receive_mode=receive_mode,
        monotonic_ns=monotonic_ns,
    )
    peer = ("127.0.0.1", _ephemeral_loopback_port())
    return adapter, raw_socket, guards, production[0], scratch_parsed, peer


def _promotion_frames():
    race_payload = encode_race_status(
        sim_boot_time_ms=800,
        race_start_boot_time_ms=-1,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    return [
        _powered_v1_frame(0),
        _powered_v1_frame(131, race_payload),
        _powered_v1_frame(105),
    ]


def _semantic_failure_message(case, raw):
    if case == "actuator_wrong_length":
        return ProductionRawMessage(
            raw,
            "ACTUATOR_OUTPUT_STATUS",
            time_usec=700,
            active=15,
            actuator=[0.0],
        )
    if case == "actuator_nonfinite":
        return ProductionRawMessage(
            raw,
            "ACTUATOR_OUTPUT_STATUS",
            time_usec=700,
            active=15,
            actuator=[0.0] * 31 + [float("nan")],
        )
    if case == "imu_nonfinite":
        return ProductionRawMessage(
            raw,
            "HIGHRES_IMU",
            xacc=float("nan"),
            yacc=2.0,
            zacc=-9.0,
            xgyro=0.1,
            ygyro=0.2,
            zgyro=0.3,
            time_usec=700,
        )
    if case == "collision_nonfinite":
        return ProductionRawMessage(
            raw,
            "COLLISION",
            id=1,
            threat_level=2,
            horizontal_minimum_delta=float("nan"),
        )
    raise AssertionError(f"unsupported semantic-failure case {case!r}")


def _semantic_failure_frame(case):
    if case.startswith("actuator_"):
        return _powered_v2_frame(375)
    if case == "imu_nonfinite":
        return _powered_v1_frame(105)
    if case == "collision_nonfinite":
        return _powered_v1_frame(247)
    raise AssertionError(f"unsupported semantic-failure case {case!r}")


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


def test_vq2_imu_mode_becomes_ready_without_pose_or_track():
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
    )

    adapter._handle_message(FakeMsg(
        "HIGHRES_IMU",
        xacc=-3.0,
        yacc=0.0,
        zacc=-9.34,
        xgyro=0.0,
        ygyro=0.0,
        zgyro=0.0,
        time_usec=125000,
    ))

    assert adapter._telemetry_ready_event.is_set()
    assert adapter.latest_telemetry.imu.timestamp_us == 125000
    assert [sample.timestamp_us for sample in adapter.drain_imu_samples()] == [125000]
    assert adapter.drain_imu_samples() == []
    assert adapter.track_data is None

    adapter._conn = FakeConn()
    with pytest.raises(RuntimeError, match="send_attitude_rate"):
        asyncio.run(adapter.send_attitude(AttitudeCommand(0.0, 0.0, 0.0, 0.2)))


def test_vq2_receiver_ingress_preserves_qpc_sequence_and_source_binding():
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
    )
    adapter._handle_message(
        FakeMsg("HEARTBEAT", base_mode=65, custom_mode=0),
        received_monotonic_ns=100,
    )
    adapter._handle_message(
        FakeMsg(
            "HIGHRES_IMU",
            xacc=1.0,
            yacc=2.0,
            zacc=-9.0,
            xgyro=0.1,
            ygyro=0.2,
            zgyro=0.3,
            time_usec=500,
        ),
        received_monotonic_ns=110,
    )
    race_payload = encode_race_status(
        sim_boot_time_ms=600,
        race_start_boot_time_ms=-1,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    adapter._handle_message(
        FakeMsg("ENCAPSULATED_DATA", data=race_payload, seqnr=0),
        received_monotonic_ns=120,
    )
    adapter._handle_message(
        FakeMsg(
            "ACTUATOR_OUTPUT_STATUS",
            time_usec=700,
            active=15,
            actuator=[0.0] * 32,
        ),
        received_monotonic_ns=130,
    )

    stats = adapter.ingress_stats()
    assert stats.next_sequence == 4
    assert stats.highres_imu_received == 1
    assert stats.heartbeat_received == 1
    assert stats.race_status_received == 1
    assert stats.actuator_received == 1
    assert stats.dropped == 0
    assert stats.high_watermark == 4
    assert stats.imu_capacity == 4096
    assert stats.other_capacity == 4096
    assert stats.imu_dropped == 0
    assert stats.other_dropped == 0
    assert stats.imu_high_watermark == 1
    assert stats.other_high_watermark == 3

    received = adapter.drain_received_imu_samples()
    assert len(received) == 1
    assert received[0].ingress.sequence == 1
    assert received[0].ingress.received_monotonic_ns == 110
    assert received[0].ingress.source_time_value == 500
    assert received[0].imu.timestamp_us == 500
    arrivals = adapter.drain_mavlink_arrivals()
    assert [item.sequence for item in arrivals] == [0, 2, 3]
    assert [item.received_monotonic_ns for item in arrivals] == [100, 120, 130]
    assert [item.source_time_unit for item in arrivals] == [None, "ms", "us"]


def test_vq2_receiver_ingress_overflow_is_counted_and_sequence_remains_strict():
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        ingress_buffer_capacity=2,
    )
    for sequence in range(3):
        adapter._handle_message(
            FakeMsg("HEARTBEAT", base_mode=65, custom_mode=0),
            received_monotonic_ns=100 + sequence,
        )

    stats = adapter.ingress_stats()
    assert stats.next_sequence == 3
    assert stats.dropped == 1
    assert stats.high_watermark == 2
    assert stats.imu_capacity == 2
    assert stats.other_capacity == 2
    assert stats.imu_dropped == 0
    assert stats.other_dropped == 1
    assert stats.imu_high_watermark == 0
    assert stats.other_high_watermark == 2
    assert [item.sequence for item in adapter.drain_mavlink_arrivals()] == [1, 2]


def test_vq2_receiver_ingress_atomic_drain_preserves_cross_queue_order():
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    adapter._handle_message(
        FakeMsg("HEARTBEAT", base_mode=65, custom_mode=0),
        received_monotonic_ns=100,
    )
    adapter._handle_message(
        FakeMsg(
            "HIGHRES_IMU",
            xacc=1.0,
            yacc=2.0,
            zacc=-9.0,
            xgyro=0.1,
            ygyro=0.2,
            zgyro=0.3,
            time_usec=500,
        ),
        received_monotonic_ns=110,
    )
    adapter._handle_message(
        FakeMsg("HEARTBEAT", base_mode=65, custom_mode=0),
        received_monotonic_ns=120,
    )

    values = adapter.drain_received_ingress()

    assert [
        value.ingress.sequence
        if isinstance(value, ReceivedIMUSampleV1)
        else value.sequence
        for value in values
    ] == [0, 1, 2]
    assert isinstance(values[1], ReceivedIMUSampleV1)
    assert adapter.ingress_stats().buffered_imu == 0
    assert adapter.ingress_stats().buffered_other == 0


def test_disconnect_rejects_an_unterminated_mavlink_worker_after_close_retry():
    class StuckThread:
        name = "stuck-rx"

        def __init__(self):
            self.join_calls = 0

        def join(self, timeout):
            assert timeout == 2.0
            self.join_calls += 1

        def is_alive(self):
            return True

    class Connection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    adapter = AIGPMavlinkAdapter(enable_vision=False)
    thread = StuckThread()
    connection = Connection()
    adapter._rx_thread = thread
    adapter._conn = connection

    with pytest.raises(RuntimeError, match="termination unproved"):
        asyncio.run(adapter.disconnect())

    assert thread.join_calls == 2
    assert connection.closed is True
    assert adapter._rx_thread is thread


def test_vq2_adapter_rejects_a_non_qpc_capture_clock():
    with pytest.raises(ValueError, match="host-perf-counter"):
        AIGPMavlinkAdapter(enable_vision=False, host_clock_id="coarse-clock")


def test_vq2_rate_wire_uses_live_measured_pitch_flip():
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
    )
    adapter._conn = FakeConn()

    asyncio.run(adapter.send_attitude_rate(
        AttitudeRateCommand(0.10, 0.20, 0.0, 0.24)
    ))

    name, args = adapter._conn.mav.calls[-1]
    assert name == "set_attitude_target_send"
    assert args[5:8] == pytest.approx((-0.10, -0.20, 0.0))
    audit = adapter.outbound_audit()
    assert audit.attitude_target == 1
    assert audit.disallowed_count == 1


def test_outbound_audit_separates_allowed_announcements_from_commands():
    adapter = _adapter_with_fake_conn()
    adapter._audit_outbound("timesync")
    adapter._audit_outbound("gcs_heartbeat")
    asyncio.run(adapter.arm())
    asyncio.run(adapter.disarm())
    asyncio.run(
        adapter.send_position(
            PositionCommand(
                position_ned=(0.0, 0.0, 0.0),
                velocity_ned=(0.0, 0.0, 0.0),
                yaw_rad=0.0,
            )
        )
    )

    audit = adapter.outbound_audit()
    assert audit.timesync == 1
    assert audit.gcs_heartbeat == 1
    assert audit.arm == 1
    assert audit.disarm == 1
    assert audit.position_target == 1
    assert audit.disallowed_count == 3


def test_pose_mode_still_requires_attitude_and_local_position():
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    adapter._handle_message(FakeMsg(
        "HIGHRES_IMU",
        xacc=0.0,
        yacc=0.0,
        zacc=-9.81,
        xgyro=0.0,
        ygyro=0.0,
        zgyro=0.0,
        time_usec=1,
    ))

    assert not adapter._telemetry_ready_event.is_set()


def test_vq2_mode_rejects_incompatible_track_requirement():
    with pytest.raises(ValueError, match="require_track"):
        AIGPMavlinkAdapter(
            enable_vision=False,
            require_track=True,
            telemetry_mode="imu",
            fetch_track_on_connect=False,
        )


def test_vq2_reset_clears_stale_epoch_before_returning():
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
    )
    adapter._conn = FakeConn()
    adapter._handle_message(FakeMsg(
        "HIGHRES_IMU",
        xacc=0.0,
        yacc=0.0,
        zacc=-9.81,
        xgyro=0.0,
        ygyro=0.0,
        zgyro=0.0,
        time_usec=100_000,
    ))
    adapter._race_status = object()

    asyncio.run(adapter.reset())

    assert adapter.latest_telemetry is None
    assert adapter.race_status is None
    assert not adapter._telemetry_ready_event.is_set()
    assert not adapter._have_imu
    assert adapter.drain_imu_samples() == []
    assert math.isinf(adapter.imu_age_s)
    assert math.isinf(adapter.race_status_age_s)
    assert math.isinf(adapter.actuator_age_s)
    assert adapter._conn.mav.calls[-1][1][2] == SIM_RESET_COMMAND


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
    assert adapter.heartbeat_age_s < 0.1
    assert adapter.heartbeat_sequence == 1
    assert adapter.race_status_age_s < 0.1
    assert adapter.actuator_age_s < 0.1

    adapter._handle_message(FakeMsg("HEARTBEAT", base_mode=65, custom_mode=0))
    assert adapter.is_armed is False
    assert adapter.heartbeat_sequence == 2


def test_send_attitude_legacy_attitude_mode_wires():
    # Fallback path (use_rate_control=False): raw attitude quaternion, mask 7.
    adapter = _adapter_with_fake_conn()
    adapter._use_rate_control = False

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


def test_send_attitude_rate_control_default_emits_body_rates():
    # Default path: the sim spins under attitude mode, so send_attitude
    # converts the desired attitude to a body-RATE command (mask 128) with the
    # sim's per-axis sign correction applied. From an identity measured
    # attitude, a desired +yaw error -> +yaw rate in FRD -> NEGATED on the wire
    # (sim applies rates with opposite sign); roll/pitch error ~0 -> ~0.
    adapter = _adapter_with_fake_conn()
    assert adapter._use_rate_control is True
    adapter._latest_telem = _default_telem_identity()

    asyncio.run(adapter.send_attitude(AttitudeCommand(0.0, 0.0, 0.5, 0.4)))
    name, args = adapter._conn.mav.calls[-1]
    assert name == "set_attitude_target_send"
    assert args[3] == 128                      # body-rate mode
    assert args[4] == [1.0, 0.0, 0.0, 0.0]     # attitude ignored
    rr, pr, yr = args[5], args[6], args[7]
    assert abs(rr) < 1e-6 and abs(pr) < 1e-6   # no roll/pitch error
    # desired +yaw -> +yaw rate (FRD) -> sign-flipped to negative on the wire
    assert yr < 0.0
    assert args[8] == pytest.approx(0.4)


def test_send_attitude_rate_position_reset_and_arm_wires():
    adapter = _adapter_with_fake_conn()

    # send_attitude_rate applies the PER-AXIS _rate_sign=(-1,+1,-1): roll & yaw
    # are sign-flipped (the sim applies them opposite-signed) but PITCH is NOT —
    # the bench gyro ID + flight validation (iter-22/23) found pitch was positive
    # feedback under (-1,-1,-1) and only stabilised at +1 (see _rate_sign comment
    # in aigp_mavlink.py). So (0.4,0.5,0.6) -> (-0.4,+0.5,-0.6) on the wire.
    asyncio.run(adapter.send_attitude_rate(AttitudeRateCommand(0.4, 0.5, 0.6, -1.0)))
    name, args = adapter._conn.mav.calls[-1]
    assert name == "set_attitude_target_send"
    assert args[3] == 128
    assert args[4] == [1.0, 0.0, 0.0, 0.0]
    assert args[5:8] == pytest.approx((-0.4, 0.5, -0.6))
    assert args[8] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="rates must be finite"):
        asyncio.run(adapter.send_attitude_rate(
            AttitudeRateCommand(float("nan"), 0.0, 0.0, 0.2)
        ))

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

    asyncio.run(adapter.disarm())
    name, args = adapter._conn.mav.calls[-1]
    assert name == "command_long_send"
    assert args[2] == 400
    assert args[4] == 0


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


def test_received_envelopes_are_single_occurrence_exact_and_globally_ordered():
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    actuator = [float(index) / 10.0 for index in range(32)]
    race_payload = encode_race_status(
        sim_boot_time_ms=600,
        race_start_boot_time_ms=-1,
        race_finish_time_ns=-1,
        active_gate_index=3,
        last_gate_race_time=25,
    )
    adapter._handle_message(
        FakeMsg("HEARTBEAT", base_mode=193, custom_mode=7),
        received_monotonic_ns=100,
    )
    adapter._handle_message(
        FakeMsg(
            "HIGHRES_IMU",
            xacc=1.0,
            yacc=2.0,
            zacc=-9.0,
            xgyro=0.1,
            ygyro=0.2,
            zgyro=0.3,
            time_usec=500,
        ),
        received_monotonic_ns=110,
    )
    adapter._handle_message(
        FakeMsg("ENCAPSULATED_DATA", data=race_payload, seqnr=0),
        received_monotonic_ns=120,
    )
    adapter._handle_message(
        FakeMsg(
            "ACTUATOR_OUTPUT_STATUS",
            time_usec=700,
            active=15,
            actuator=actuator,
        ),
        received_monotonic_ns=130,
    )
    actuator[0] = 999.0

    observations = adapter.drain_received_observations()

    assert [type(item) for item in observations] == [
        ReceivedHeartbeatV1,
        ReceivedIMUSampleV1,
        ReceivedRaceStatusV1,
        ReceivedActuatorOutputStatusV1,
    ]
    assert [item.ingress.sequence for item in observations] == [0, 1, 2, 3]
    assert observations[0].heartbeat.to_primitive() == {
        "base_mode": 193,
        "custom_mode": 7,
    }
    assert observations[2].race_status.active_gate_index == 3
    assert observations[3].actuator_output_status.actuator[0] == 0.0
    assert adapter.latest_received_heartbeat is observations[0]
    assert adapter.latest_received_race_status is observations[2]
    assert adapter.latest_received_actuator_output_status is observations[3]
    # Every drain consumes the same underlying occurrence queues.
    assert adapter.drain_received_observations() == []
    assert adapter.drain_received_imu_samples() == []
    assert adapter.drain_mavlink_arrivals() == []


def test_outbound_receipts_cover_every_admitted_send_with_exact_wire_values():
    ticks = iter(range(1_000, 2_000))
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        monotonic_ns=lambda: next(ticks),
    )
    adapter._conn = FakeConn()
    adapter._target_system = 42
    adapter._target_component = 99
    adapter._latest_telem = _default_telem_identity()

    asyncio.run(adapter.send_attitude_rate(AttitudeRateCommand(0.1, 0.2, 0.3, 0.4)))
    asyncio.run(adapter.send_attitude(AttitudeCommand(0.0, 0.0, 0.2, 0.3)))
    adapter._use_rate_control = False
    asyncio.run(adapter.send_attitude(AttitudeCommand(0.1, -0.2, 0.3, 0.5)))
    asyncio.run(adapter.arm())
    asyncio.run(adapter.disarm())
    asyncio.run(adapter._send_sim_reset())

    class OnePassStopEvent:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, timeout):
            assert timeout == 0.1
            self.stopped = True
            return True

    adapter._stop_event = OnePassStopEvent()
    adapter._announce_loop()

    receipts = adapter.drain_outbound_receipts()
    assert [item.outbound_sequence for item in receipts] == list(range(8))
    assert [item.outcome for item in receipts] == ["returned"] * 8
    assert [item.api for item in receipts[:3]] == [
        "send_attitude_rate",
        "send_attitude_rate_from_attitude",
        "send_attitude_quaternion",
    ]
    assert all(isinstance(item, AttitudeTargetOutboundV1) for item in receipts[:3])
    assert [item.category for item in receipts[3:]] == [
        "arm",
        "disarm",
        "sim_reset",
        "timesync",
        "gcs_heartbeat",
    ]
    assert all(isinstance(item, NonAttitudeOutboundV1) for item in receipts[3:])
    attitude_call = adapter._conn.mav.calls[0][1]
    assert receipts[0].wire.q_wxyz == tuple(attitude_call[4])
    assert receipts[0].wire.body_rates_rad_s == attitude_call[5:8]
    assert receipts[0].wire.thrust == attitude_call[8]
    assert receipts[3].wire.params == adapter._conn.mav.calls[3][1][4:]
    stats = adapter.outbound_receipt_stats()
    assert stats.next_sequence == 8
    assert stats.returned == 8
    assert stats.raised == 0
    assert stats.dropped == 0
    assert stats.high_watermark == 8
    assert stats.buffered == 0

    # Position target is deliberately outside the frozen admitted schemas.
    asyncio.run(
        adapter.send_position(
            PositionCommand(
                position_ned=(0.0, 0.0, 0.0),
                velocity_ned=(0.0, 0.0, 0.0),
                yaw_rad=0.0,
            )
        )
    )
    assert adapter.drain_outbound_receipts() == []
    assert adapter.outbound_audit().position_target == 1


def test_outbound_receipts_record_raised_calls_and_bounded_overflow():
    class RaisingMav(FakeMav):
        def set_attitude_target_send(self, *args):
            self.calls.append(("set_attitude_target_send", args))
            raise RuntimeError("attitude failed")

        def command_long_send(self, *args):
            self.calls.append(("command_long_send", args))
            raise OSError("command failed")

    ticks = iter(range(2_000, 2_100))
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        monotonic_ns=lambda: next(ticks),
    )
    adapter._conn = FakeConn()
    adapter._conn.mav = RaisingMav()

    with pytest.raises(RuntimeError, match="attitude failed"):
        asyncio.run(
            adapter.send_attitude_rate(AttitudeRateCommand(0.1, 0.0, 0.0, 0.2))
        )
    with pytest.raises(OSError, match="command failed"):
        asyncio.run(adapter.disarm())

    receipts = adapter.drain_outbound_receipts()
    assert [(item.outcome, item.error_type) for item in receipts] == [
        ("raised", "RuntimeError"),
        ("raised", "OSError"),
    ]
    assert adapter.outbound_audit().attitude_target == 1
    assert adapter.outbound_audit().disarm == 1

    bounded_ticks = iter(range(3_000, 3_100))
    bounded = AIGPMavlinkAdapter(
        enable_vision=False,
        outbound_receipt_capacity=2,
        monotonic_ns=lambda: next(bounded_ticks),
    )
    bounded._conn = FakeConn()
    for _ in range(3):
        asyncio.run(bounded.disarm())
    stats = bounded.outbound_receipt_stats()
    assert stats.next_sequence == 3
    assert stats.returned == 3
    assert stats.dropped == 1
    assert stats.high_watermark == 2
    assert [item.outbound_sequence for item in bounded.drain_outbound_receipts()] == [1, 2]


def test_outbound_receipt_sequence_and_queue_are_attempt_global_across_reset():
    ticks = iter(range(4_000, 4_100))
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
        monotonic_ns=lambda: next(ticks),
    )
    adapter._conn = FakeConn()

    asyncio.run(adapter.disarm())
    asyncio.run(adapter.reset())

    before_drain = adapter.drain_outbound_receipts()
    assert [item.outbound_sequence for item in before_drain] == [0, 1]
    assert [item.reset_generation for item in before_drain] == [0, 1]
    assert [item.category for item in before_drain] == ["disarm", "sim_reset"]

    asyncio.run(adapter.disarm())
    after_drain = adapter.drain_outbound_receipts()
    assert [item.outbound_sequence for item in after_drain] == [2]
    assert [item.reset_generation for item in after_drain] == [1]
    stats = adapter.outbound_receipt_stats()
    assert stats.generation == 1
    assert stats.next_sequence == 3
    assert stats.returned == 3
    assert stats.raised == 0
    assert stats.dropped == 0


def test_calibration_reset_boundary_is_atomic_defensive_and_persisted_before_send():
    ticks = iter((10_000, 10_010, 10_020))
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        require_track=False,
        telemetry_mode="imu",
        fetch_track_on_connect=False,
        monotonic_ns=lambda: next(ticks),
    )
    adapter._conn = FakeConn()
    actuator = [0.25] * 32
    race_payload = encode_race_status(
        sim_boot_time_ms=800,
        race_start_boot_time_ms=-1,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    adapter._handle_message(
        FakeMsg("HEARTBEAT", base_mode=65, custom_mode=3),
        received_monotonic_ns=100,
    )
    retained_heartbeat = adapter.latest_received_heartbeat
    adapter._handle_message(
        FakeMsg(
            "HIGHRES_IMU",
            xacc=0.0,
            yacc=0.0,
            zacc=-9.81,
            xgyro=0.0,
            ygyro=0.0,
            zgyro=0.0,
            time_usec=700,
        ),
        received_monotonic_ns=110,
    )
    adapter._handle_message(
        FakeMsg("ENCAPSULATED_DATA", data=race_payload, seqnr=0),
        received_monotonic_ns=120,
    )
    adapter._handle_message(
        FakeMsg(
            "ACTUATOR_OUTPUT_STATUS",
            time_usec=900,
            active=15,
            actuator=actuator,
        ),
        received_monotonic_ns=130,
    )
    adapter._handle_message(
        FakeMsg(
            "COLLISION",
            id=1001,
            threat_level=2,
            horizontal_minimum_delta=3.5,
        ),
        received_monotonic_ns=140,
    )
    actuator[0] = 99.0
    persisted = []

    def persist(boundary):
        assert adapter._conn.mav.calls == []
        persisted.append(boundary)

    boundary = asyncio.run(adapter.reset_calibration_with_boundary(persist))

    assert persisted == [boundary]
    assert boundary.old_generation == 0
    assert boundary.new_generation == 1
    assert boundary.boundary_monotonic_ns == 10_000
    assert [item.ingress.sequence for item in boundary.observations] == [0, 1, 2, 3]
    assert boundary.observations[3].actuator_output_status.actuator[0] == 0.25
    assert boundary.ingress_stats.next_sequence == 4
    assert boundary.ingress_stats.buffered_imu == 1
    assert boundary.ingress_stats.buffered_other == 3
    assert boundary.collision_stats.handled == 1
    assert boundary.collision_stats.dropped == 0
    assert boundary.collisions[0].to_primitive() == {
        "id": 1001,
        "threat_level": 2,
        "impulse": 3.5,
    }
    with pytest.raises(FrozenInstanceError):
        boundary.collisions[0].impulse = 9.0
    assert adapter.ingress_stats().generation == 1
    assert adapter.ingress_stats().next_sequence == 0
    assert adapter.collision_stats().generation == 1
    assert adapter.collision_stats().handled == 0
    assert adapter.drain_received_observations() == []
    assert adapter.drain_collisions() == []
    assert adapter.latest_telemetry is None
    assert adapter.latest_received_race_status is None
    assert adapter.latest_received_actuator_output_status is None
    assert adapter.latest_received_heartbeat is retained_heartbeat
    assert adapter.heartbeat_sequence == 1
    assert adapter._conn.mav.calls[0][1][2] == SIM_RESET_COMMAND
    reset_receipt = adapter.drain_outbound_receipts()
    assert len(reset_receipt) == 1
    assert reset_receipt[0].reset_generation == 1
    assert reset_receipt[0].outbound_sequence == 0
    assert reset_receipt[0].category == "sim_reset"


def test_calibration_boundary_persistence_failure_advances_once_and_sends_nothing():
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        monotonic_ns=lambda: 50_000,
    )
    adapter._conn = FakeConn()
    adapter._handle_message(
        FakeMsg("HEARTBEAT", base_mode=65, custom_mode=0),
        received_monotonic_ns=100,
    )

    def fail_persistence(_boundary):
        raise RuntimeError("disk failed")

    with pytest.raises(RuntimeError, match="disk failed"):
        asyncio.run(adapter.reset_calibration_with_boundary(fail_persistence))

    assert adapter.ingress_stats().generation == 1
    assert adapter.ingress_stats().next_sequence == 0
    assert adapter._conn.mav.calls == []
    assert adapter.drain_received_observations() == []
    assert adapter.drain_outbound_receipts() == []
    assert not adapter.calibration_reset_persistence_state().failure_latched


def test_powered_cleanup_persistence_failure_is_latched_but_cannot_suppress_reset():
    ticks = iter((100, 200, 300))
    adapter, raw_socket, guards, _production, _scratch, peer = _powered_adapter(
        monotonic_ns=lambda: next(ticks)
    )
    _manually_promote_powered_adapter(adapter, guards, peer)
    guards.enable_cleanup_live(
        parent_alive=True,
        lease_valid=True,
        source_promoted=True,
    )
    retained = []

    class PersistenceBoom(RuntimeError):
        pass

    def fail_persistence(boundary):
        retained.append(boundary)
        raise PersistenceBoom("private persistence detail")

    boundary = asyncio.run(
        adapter.reset_calibration_with_boundary(
            fail_persistence,
            powered_deadline_monotonic_ns=1_000_000_000,
            powered_cleanup=True,
        )
    )

    assert retained == [boundary]
    assert len(raw_socket.sent) == 1
    assert adapter.outbound_audit().sim_reset == 1
    receipt = adapter.drain_outbound_receipts()
    assert len(receipt) == 1
    assert receipt[0].category == "sim_reset"
    assert receipt[0].outcome == "returned"
    state = adapter.calibration_reset_persistence_state()
    assert state.failure_latched
    assert state.dropped == 0
    assert len(state.failures) == 1
    assert state.failures[0].error_type == "PersistenceBoom"
    assert "private" not in repr(state.to_primitive())
    assert guards.production_latched
    assert guards.cleanup_state == "enabled_live"
    adapter._powered_transport.close()


def test_powered_cleanup_reset_progress_can_take_over_after_boundary_before_send():
    authority = {
        "role_valid": True,
        "parent_alive": True,
        "lease_valid": True,
    }
    ticks = iter((100, 200, 300))
    adapter, raw_socket, guards, _production, _scratch, peer = _powered_adapter(
        authority=authority,
        monotonic_ns=lambda: next(ticks),
    )
    _manually_promote_powered_adapter(adapter, guards, peer)
    guards.enable_cleanup_live(
        parent_alive=True,
        lease_valid=True,
        source_promoted=True,
    )
    order = []

    def persist(_boundary):
        assert raw_socket.sent == []
        order.append("persist")

    def take_over():
        assert order == ["persist"]
        assert raw_socket.sent == []
        authority["parent_alive"] = False
        guards.note_parent_death()
        guards.enable_cleanup_takeover(
            parent_signaled=True,
            abandoned_lease_owned=True,
            authority_valid=True,
            source_promoted=True,
        )
        order.append("takeover")

    boundary = asyncio.run(
        adapter.reset_calibration_with_boundary(
            persist,
            powered_deadline_monotonic_ns=1_000_000_000,
            powered_cleanup=True,
            powered_progress=take_over,
        )
    )

    assert boundary.new_generation == 1
    assert order == ["persist", "takeover"]
    assert len(raw_socket.sent) == 1
    assert adapter.outbound_audit().sim_reset == 1
    assert guards.cleanup_state == "enabled_takeover"
    adapter._powered_transport.close()


def test_reset_progress_rejects_noncleanup_use_before_generation_transition():
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        monotonic_ns=lambda: 50_000,
    )
    adapter._conn = FakeConn()

    with pytest.raises(ValueError, match="requires powered cleanup"):
        asyncio.run(
            adapter.reset_calibration_with_boundary(
                lambda _boundary: None,
                powered_progress=lambda: None,
            )
        )

    assert adapter.ingress_stats().generation == 0
    assert adapter._conn.mav.calls == []


def test_powered_prepower_persistence_failure_still_suppresses_reset():
    adapter, raw_socket, guards, _production, _scratch, peer = _powered_adapter()
    _manually_promote_powered_adapter(adapter, guards, peer)

    def fail_persistence(_boundary):
        raise RuntimeError("disk failed")

    with pytest.raises(RuntimeError, match="disk failed"):
        asyncio.run(
            adapter.reset_calibration_with_boundary(
                fail_persistence,
                powered_deadline_monotonic_ns=time.perf_counter_ns()
                + 1_000_000_000,
                powered_cleanup=False,
            )
        )

    assert raw_socket.sent == []
    assert adapter.outbound_audit().sim_reset == 0
    assert adapter.drain_outbound_receipts() == []
    assert not adapter.calibration_reset_persistence_state().failure_latched
    adapter._powered_transport.close()


def test_calibration_boundary_excludes_receiver_dispatch_through_reset_call():
    ticks = iter((60_000, 60_010, 60_020))
    adapter = AIGPMavlinkAdapter(
        enable_vision=False,
        monotonic_ns=lambda: next(ticks),
    )
    adapter._conn = FakeConn()
    receiver_started = threading.Event()
    receiver_finished = threading.Event()
    receiver_thread = None

    def receive_new_heartbeat():
        receiver_started.set()
        adapter._handle_message(
            FakeMsg("HEARTBEAT", base_mode=65, custom_mode=0),
            received_monotonic_ns=60_100,
        )
        receiver_finished.set()

    def persist(_boundary):
        nonlocal receiver_thread
        receiver_thread = threading.Thread(target=receive_new_heartbeat)
        receiver_thread.start()
        assert receiver_started.wait(0.5)
        assert not receiver_finished.wait(0.02)
        assert adapter.ingress_stats().next_sequence == 0

    boundary = asyncio.run(adapter.reset_calibration_with_boundary(persist))
    assert receiver_thread is not None
    receiver_thread.join(timeout=0.5)
    assert not receiver_thread.is_alive()
    assert receiver_finished.is_set()
    observations = adapter.drain_received_observations()
    assert len(observations) == 1
    assert observations[0].ingress.generation == boundary.new_generation
    assert observations[0].ingress.sequence == 0


def test_collision_diagnostics_count_bounded_overflow_and_legacy_drain():
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    for collision_id in range(129):
        adapter._handle_message(
            FakeMsg(
                "COLLISION",
                id=collision_id,
                threat_level=1,
                horizontal_minimum_delta=0.5,
            ),
            received_monotonic_ns=collision_id,
        )

    stats = adapter.collision_stats()
    assert stats.handled == 129
    assert stats.dropped == 1
    assert stats.high_watermark == 128
    assert stats.capacity == 128
    assert stats.buffered == 128
    collisions = adapter.drain_collisions()
    assert [item["id"] for item in collisions] == list(range(1, 129))
    drained_stats = adapter.collision_stats()
    assert drained_stats.handled == 129
    assert drained_stats.dropped == 1
    assert drained_stats.buffered == 0


def test_powered_transport_construction_failure_closes_owned_endpoint():
    raw_socket = FakeExclusiveRawSocket()
    endpoint = ExclusiveUdpEndpoint(
        socket=raw_socket,
        requested_host="127.0.0.1",
        requested_port=0,
        actual_host="127.0.0.1",
        actual_port=raw_socket.port,
        exclusive_option=1,
    )

    def fail_factory(_writer):
        raise RuntimeError("injected production parser failure")

    with pytest.raises(RuntimeError, match="injected production parser failure"):
        PoweredMavlinkTransport(
            endpoint,
            scratch_parser_factory=lambda: ScratchOnlyParser([]),
            mavlink_factory=fail_factory,
            outbound_guards=PoweredOutboundGuards(),
            role_valid=lambda: True,
            parent_alive=lambda: True,
            lease_valid=lambda: True,
        )
    assert endpoint.closed
    assert raw_socket.closed


def test_powered_rx_worker_polls_at_50ms_and_continues_after_timeout():
    adapter, _raw_socket, _guards, _production, _scratch, peer = _powered_adapter()
    transport = adapter._powered_transport
    transport.claim_receive_owner("adapter_worker")
    calls = []
    dispatched = []

    def receive(*, owner, max_wait_ns):
        calls.append((owner, max_wait_ns))
        if len(calls) == 1:
            return None
        adapter._stop_event.set()
        return _powered_v1_frame(0), peer

    transport.recvfrom = receive
    adapter._handle_powered_datagram = (
        lambda raw, source: dispatched.append((raw, source))
    )
    adapter._stop_event.clear()

    adapter._powered_rx_loop()

    assert calls == [
        ("adapter_worker", POWERED_WORKER_POLL_NS),
        ("adapter_worker", POWERED_WORKER_POLL_NS),
    ]
    assert dispatched == [(_powered_v1_frame(0), peer)]
    transport.close()


def test_powered_connect_forwards_raw_to_production_not_scratch_and_freezes_target():
    adapter, raw_socket, _guards, production, scratch_raw, peer = _powered_adapter()
    frames = _promotion_frames()
    for frame in frames:
        raw_socket.push(frame, peer)

    asyncio.run(
        adapter.connect(
            deadline_monotonic_ns=time.perf_counter_ns() + 2_000_000_000
        )
    )

    assert adapter.powered_source_promoted
    assert adapter.powered_peer == peer
    assert scratch_raw == frames
    assert production.parsed_raw == frames
    # Scratch reports base_mode=193. Production reports 65; production must win.
    assert adapter.latest_received_heartbeat.heartbeat.base_mode == 65
    assert adapter.is_armed is False
    assert adapter.latest_received_race_status.race_status.sim_boot_time_ms == 800
    assert adapter.latest_telemetry.imu.timestamp_us == 700
    assert adapter._announce_thread is None
    connected_state = adapter.powered_transport_state()
    assert connected_state.endpoint_state == "peer_frozen"
    assert connected_state.frozen_peer == peer
    assert connected_state.rejected_source_count == 0
    assert connected_state.endpoint_closed is False
    assert connected_state.receiver_joined is False
    assert connected_state.announcer_joined is True
    assert connected_state.connection_closed is False
    assert connected_state.owned_handles_closed is False
    assert connected_state.bind_proof() == {
        "family": "AF_INET",
        "requested": {
            "host": "127.0.0.1",
            "port": 0,
        },
        "actual": {
            "host": "127.0.0.1",
            "port": raw_socket.port,
        },
        "socket_policy": "ipv4-exclusive-address-use",
    }
    sent_after_connect = len(raw_socket.sent)
    time.sleep(0.02)
    assert len(raw_socket.sent) == sent_after_connect

    asyncio.run(
        adapter.send_attitude_rate(
            AttitudeRateCommand(0.1, 0.0, 0.0, 0.2),
            powered_deadline_monotonic_ns=time.perf_counter_ns()
            + 1_000_000_000,
        )
    )
    assert raw_socket.sent[-1][1] == peer
    receipts = adapter.drain_outbound_receipts()
    attitude_receipts = [
        item
        for item in receipts
        if isinstance(item, AttitudeTargetOutboundV1)
    ]
    assert len(attitude_receipts) == 1
    assert attitude_receipts[0].api == "send_attitude_rate"
    assert attitude_receipts[0].outcome == "returned"

    asyncio.run(
        adapter.disconnect(
            deadline_monotonic_ns=time.perf_counter_ns() + 1_000_000_000
        )
    )
    assert raw_socket.closed
    closed_state = adapter.powered_transport_state()
    assert closed_state.endpoint_state == "closed_with_peer"
    assert closed_state.frozen_peer == peer
    assert closed_state.endpoint_closed
    assert closed_state.receiver_joined
    assert closed_state.announcer_joined
    assert closed_state.connection_closed
    assert closed_state.owned_handles_closed


@pytest.mark.parametrize(
    "case",
    [
        "actuator_wrong_length",
        "actuator_nonfinite",
        "imu_nonfinite",
        "collision_nonfinite",
    ],
)
def test_powered_semantic_handler_failure_latches_before_further_authorization(case):
    adapter, raw_socket, guards, production, _scratch_raw, peer = _powered_adapter()
    _manually_promote_powered_adapter(adapter, guards, peer)
    frame = _semantic_failure_frame(case)
    production.parse_buffer = lambda raw: [
        _semantic_failure_message(case, bytes(raw))
    ]
    ingress_before = adapter.ingress_stats()
    collisions_before = adapter.collision_stats()

    try:
        dispatch = adapter._handle_powered_datagram(frame, peer)

        assert dispatch.source_accepted
        assert dispatch.production_dispatched
        assert dispatch.admitted_message_type is None
        assert dispatch.failure_reason == "production_mavlink_handler_failed"
        assert guards.production_latched
        assert guards.production_reason == "production_mavlink_handler_failed"
        assert adapter._powered_failure_reason == "production_mavlink_handler_failed"
        assert adapter._powered_failure_event.is_set()
        assert adapter.ingress_stats() == ingress_before
        assert adapter.collision_stats() == collisions_before
        assert adapter.drain_received_observations() == []
        assert adapter.drain_collisions() == []

        with pytest.raises(OutboundAuthorityError, match="latched off"):
            asyncio.run(
                adapter.send_attitude_rate(
                    AttitudeRateCommand(0.0, 0.0, 0.0, 0.0),
                    powered_deadline_monotonic_ns=time.perf_counter_ns()
                    + 1_000_000_000,
                )
            )
        assert raw_socket.sent == []
    finally:
        adapter._powered_transport.close()


@pytest.mark.parametrize(
    "case",
    [
        "actuator_wrong_length",
        "actuator_nonfinite",
        "imu_nonfinite",
        "collision_nonfinite",
    ],
)
def test_passive_semantic_handler_failure_remains_contained(case):
    adapter = AIGPMavlinkAdapter(enable_vision=False)
    ingress_before = adapter.ingress_stats()
    collisions_before = adapter.collision_stats()

    adapter._handle_message(
        _semantic_failure_message(case, b""),
        received_monotonic_ns=100,
    )

    assert adapter.ingress_stats() == ingress_before
    assert adapter.collision_stats() == collisions_before
    assert adapter.drain_received_observations() == []
    assert adapter.drain_collisions() == []


def test_powered_second_source_is_rejected_before_production_mutation_and_cleanup_keeps_peer():
    adapter, raw_socket, guards, production, _scratch_raw, peer = _powered_adapter()
    for frame in _promotion_frames():
        raw_socket.push(frame, peer)
    asyncio.run(
        adapter.connect(
            deadline_monotonic_ns=time.perf_counter_ns() + 2_000_000_000
        )
    )
    parsed_before = len(production.parsed_raw)
    ingress_before = adapter.ingress_stats().next_sequence
    second_peer = ("127.0.0.1", _ephemeral_loopback_port())
    raw_socket.push(_powered_v1_frame(0), second_peer)
    deadline = time.monotonic() + 0.5
    while not adapter.powered_source_rejected and time.monotonic() < deadline:
        time.sleep(0.005)

    assert adapter.powered_source_rejected
    assert adapter.powered_peer == peer
    assert len(production.parsed_raw) == parsed_before
    assert adapter.ingress_stats().next_sequence == ingress_before
    assert guards.production_latched
    assert guards.production_reason == "mavlink_source_rejected"

    guards.enable_cleanup_live(
        parent_alive=True,
        lease_valid=True,
        source_promoted=True,
    )
    asyncio.run(
        adapter.send_attitude_rate(
            AttitudeRateCommand(0.0, 0.0, 0.0, 0.0),
            powered_deadline_monotonic_ns=time.perf_counter_ns()
            + 1_000_000_000,
            powered_cleanup=True,
        )
    )
    assert raw_socket.sent[-1][1] == peer
    asyncio.run(
        adapter.disarm(
            powered_deadline_monotonic_ns=time.perf_counter_ns()
            + 1_000_000_000,
            powered_cleanup=True,
        )
    )
    assert raw_socket.sent[-1][1] == peer
    cleanup_receipts = adapter.drain_outbound_receipts()
    assert any(
        isinstance(item, AttitudeTargetOutboundV1)
        and item.api == "send_attitude_rate"
        for item in cleanup_receipts
    )
    assert cleanup_receipts[-1].category == "disarm"
    asyncio.run(
        adapter.disconnect(
            deadline_monotonic_ns=time.perf_counter_ns() + 1_000_000_000
        )
    )


def test_external_cleanup_receive_has_one_owner_no_workers_and_one_source_gate():
    authorized = []

    def authorize(category):
        authorized.append(category)
        return time.perf_counter_ns() + 1_000_000_000

    adapter, raw_socket, guards, production, scratch_raw, peer = _powered_adapter(
        receive_mode=POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP,
        external_cleanup_authorize=authorize,
    )
    guards.latch_production("test_cleanup_epoch")
    asyncio.run(
        adapter.connect(
            deadline_monotonic_ns=time.perf_counter_ns() + 1_000_000_000
        )
    )

    assert adapter.powered_receive_owner == "external_cleanup"
    assert adapter._rx_thread is None
    assert adapter._announce_thread is None
    with pytest.raises(RuntimeError, match="already has an owner"):
        adapter._powered_transport.claim_receive_owner("adapter_worker")
    with pytest.raises(RuntimeError, match="ownership"):
        adapter._powered_transport.recvfrom(owner="adapter_worker")

    frames = _promotion_frames()
    for frame in frames:
        raw_socket.push(frame, peer)
    dispatches = [
        adapter.receive_powered_external(50_000_000)
        for _frame in frames
    ]
    assert all(item.source_accepted for item in dispatches)
    assert dispatches[0].peer_frozen_now
    assert dispatches[-1].source_promoted
    assert adapter.powered_source_promoted
    assert adapter.powered_source_authority is adapter._powered_transport.source_gate
    assert scratch_raw == frames
    assert production.parsed_raw == frames
    assert authorized == []

    second_peer = ("127.0.0.1", _ephemeral_loopback_port())
    rejected_frame = _powered_v1_frame(0)
    parsed_before = list(production.parsed_raw)
    raw_socket.push(rejected_frame, second_peer)
    rejected = adapter.receive_powered_external(50_000_000)
    assert rejected.rejected_source
    assert not rejected.production_dispatched
    assert adapter.powered_peer == peer
    assert production.parsed_raw == parsed_before
    assert adapter.powered_source_rejected

    asyncio.run(
        adapter.disconnect(
            deadline_monotonic_ns=time.perf_counter_ns() + 1_000_000_000
        )
    )
    assert raw_socket.closed


def test_external_cleanup_announcements_use_dispatcher_and_stop_after_guard_close():
    authorized = []

    def authorize(category):
        authorized.append(category)
        return time.perf_counter_ns() + 1_000_000_000

    adapter, raw_socket, guards, _production, _scratch, peer = _powered_adapter(
        receive_mode=POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP,
        external_cleanup_authorize=authorize,
    )
    guards.latch_production("test_cleanup_epoch")
    asyncio.run(
        adapter.connect(
            deadline_monotonic_ns=time.perf_counter_ns() + 1_000_000_000
        )
    )
    raw_socket.push(_powered_v1_frame(0), peer)
    assert adapter.receive_powered_external(50_000_000).source_accepted

    adapter.announce_powered_external_cleanup()
    assert authorized == ["timesync", "gcs_heartbeat"]
    assert [item[1] for item in raw_socket.sent] == [peer, peer]
    audit = adapter.outbound_audit()
    assert audit.timesync == audit.gcs_heartbeat == 1
    receipts = adapter.drain_outbound_receipts()
    assert [item.category for item in receipts] == [
        "timesync",
        "gcs_heartbeat",
    ]

    sent_before = list(raw_socket.sent)
    guards.close_cleanup()
    with pytest.raises(OutboundAuthorityError, match="closed"):
        adapter.announce_powered_external_cleanup()
    assert raw_socket.sent == sent_before
    assert adapter.outbound_audit().timesync == 1
    assert adapter.outbound_audit().gcs_heartbeat == 1
    asyncio.run(
        adapter.disconnect(
            deadline_monotonic_ns=time.perf_counter_ns() + 1_000_000_000
        )
    )


def test_external_cleanup_connect_without_dispatcher_fails_closed_before_workers():
    adapter, raw_socket, guards, _production, _scratch, _peer = _powered_adapter(
        receive_mode=POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP,
    )
    guards.latch_production("test_cleanup_epoch")
    with pytest.raises(OutboundAuthorityError, match="authority is unavailable"):
        asyncio.run(
            adapter.connect(
                deadline_monotonic_ns=time.perf_counter_ns()
                + 1_000_000_000
            )
        )
    assert raw_socket.closed
    assert adapter._rx_thread is None
    assert adapter._announce_thread is None
    assert adapter.powered_receive_owner is None


def test_external_cleanup_announcement_completion_deadline_closes_guard():
    ticks = iter((100, 101, 200, 300, 400))
    adapter, raw_socket, guards, _production, _scratch, peer = _powered_adapter(
        receive_mode=POWERED_RECEIVE_MODE_EXTERNAL_CLEANUP,
        external_cleanup_authorize=lambda _category: 250,
        monotonic_ns=lambda: next(ticks),
    )
    guards.latch_production("test_cleanup_epoch")
    asyncio.run(adapter.connect(deadline_monotonic_ns=1_000))
    assert adapter._powered_transport.source_gate.ingest(
        _powered_v1_frame(0),
        peer,
    ).accepted

    with pytest.raises(OutboundAuthorityError, match="clipped deadline"):
        adapter.announce_powered_external_cleanup()
    assert guards.cleanup_state == "closed"
    assert len(raw_socket.sent) == 1
    assert adapter.outbound_audit().timesync == 1
    assert adapter.outbound_audit().gcs_heartbeat == 0
    receipts = adapter.drain_outbound_receipts()
    assert len(receipts) == 1
    assert receipts[0].category == "timesync"
    assert receipts[0].outcome == "returned"
    asyncio.run(adapter.disconnect(deadline_monotonic_ns=1_000))


def test_powered_nonannouncement_is_denied_before_promotion_and_before_send():
    adapter, raw_socket, guards, _production, _scratch_raw, peer = _powered_adapter()
    transport = adapter._powered_transport
    guards.enable_production()
    adapter._conn = transport.connection
    assert transport.source_gate.ingest(_powered_v1_frame(0), peer).accepted
    assert not transport.promoted
    with pytest.raises(OutboundAuthorityError, match="lacks adapter authorization"):
        transport.mavlink.timesync_send(0, 1)

    with pytest.raises(OutboundAuthorityError, match="not promoted"):
        asyncio.run(
            adapter.disarm(
                powered_deadline_monotonic_ns=time.perf_counter_ns()
                + 1_000_000_000
            )
        )

    assert guards.production_latched
    assert raw_socket.sent == []
    assert adapter.drain_outbound_receipts() == []
    transport.close()


def _manually_promote_powered_adapter(adapter, guards, peer):
    transport = adapter._powered_transport
    guards.enable_production()
    adapter._conn = transport.connection
    assert transport.source_gate.ingest(_powered_v1_frame(0), peer).accepted
    assert not transport.source_gate.observe_fresh_stream("HEARTBEAT")
    assert not transport.source_gate.observe_fresh_stream("RACE_STATUS")
    assert transport.source_gate.observe_fresh_stream("HIGHRES_IMU")
    assert transport.promoted


def test_powered_nonattitude_call_is_clipped_to_exact_250ms_and_latched_late():
    call_start = 1_000
    clipped_deadline = call_start + POWERED_OUTBOUND_CALL_NS
    ticks = iter((call_start, clipped_deadline))
    adapter, raw_socket, guards, _production, _scratch, peer = _powered_adapter(
        monotonic_ns=lambda: next(ticks)
    )
    _manually_promote_powered_adapter(adapter, guards, peer)

    with pytest.raises(OutboundAuthorityError, match="clipped deadline"):
        asyncio.run(
            adapter.disarm(
                powered_deadline_monotonic_ns=clipped_deadline
                + 10_000_000_000,
            )
        )

    assert len(raw_socket.sent) == 1
    receipt = adapter.drain_outbound_receipts()
    assert len(receipt) == 1
    assert receipt[0].category == "disarm"
    assert receipt[0].call_start_monotonic_ns == call_start
    assert receipt[0].call_end_monotonic_ns == clipped_deadline
    # The pymavlink call really returned; the adapter then rejected lateness.
    assert receipt[0].outcome == "returned"
    assert guards.production_latched
    assert guards.production_reason == "powered_nonattitude_call_deadline_reached"
    adapter._powered_transport.close()


def test_powered_late_underlying_raise_keeps_truthful_raised_receipt():
    call_start = 2_000
    clipped_deadline = call_start + POWERED_OUTBOUND_CALL_NS
    ticks = iter((call_start, clipped_deadline))
    adapter, raw_socket, guards, production, _scratch, peer = _powered_adapter(
        monotonic_ns=lambda: next(ticks)
    )
    _manually_promote_powered_adapter(adapter, guards, peer)

    class WireFailure(RuntimeError):
        pass

    def raise_after_write(*_args):
        production.file.write(b"raised:wire")
        raise WireFailure("injected wire failure")

    production.command_long_send = raise_after_write
    with pytest.raises(WireFailure, match="injected wire failure"):
        asyncio.run(
            adapter.disarm(
                powered_deadline_monotonic_ns=clipped_deadline
                + 10_000_000_000,
            )
        )

    assert len(raw_socket.sent) == 1
    receipt = adapter.drain_outbound_receipts()
    assert len(receipt) == 1
    assert receipt[0].call_end_monotonic_ns == clipped_deadline
    assert receipt[0].outcome == "raised"
    assert receipt[0].error_type == "WireFailure"
    assert guards.production_latched
    adapter._powered_transport.close()


def test_powered_connect_announcement_uses_same_exact_250ms_clip():
    call_start = 10_000
    clipped_deadline = call_start + POWERED_OUTBOUND_CALL_NS
    ticks = iter((call_start, clipped_deadline))
    adapter, raw_socket, guards, _production, _scratch, peer = _powered_adapter(
        monotonic_ns=lambda: next(ticks)
    )
    transport = adapter._powered_transport
    guards.enable_production()
    adapter._conn = transport.connection
    assert transport.source_gate.ingest(_powered_v1_frame(0), peer).accepted
    adapter._powered_connect_deadline_monotonic_ns = (
        clipped_deadline + 10_000_000_000
    )

    adapter._powered_announce_loop()

    assert len(raw_socket.sent) == 1
    receipt = adapter.drain_outbound_receipts()
    assert len(receipt) == 1
    assert receipt[0].category == "timesync"
    assert receipt[0].call_start_monotonic_ns == call_start
    assert receipt[0].call_end_monotonic_ns == clipped_deadline
    assert receipt[0].outcome == "returned"
    assert guards.production_latched
    assert guards.production_reason == "powered_nonattitude_call_deadline_reached"
    assert adapter._powered_failure_event.is_set()
    transport.close()


def test_powered_parent_deadline_and_position_paths_latch_before_wire_mutation():
    authority = {
        "role_valid": True,
        "parent_alive": True,
        "lease_valid": True,
    }
    parent_adapter, parent_socket, parent_guards, _prod, _scratch, peer = (
        _powered_adapter(authority=authority)
    )
    _manually_promote_powered_adapter(parent_adapter, parent_guards, peer)
    authority["parent_alive"] = False
    with pytest.raises(OutboundAuthorityError, match="parent"):
        asyncio.run(
            parent_adapter.send_attitude_rate(
                AttitudeRateCommand(0.1, 0.0, 0.0, 0.2),
                powered_deadline_monotonic_ns=time.perf_counter_ns()
                + 1_000_000_000,
            )
        )
    assert parent_guards.production_latched
    assert parent_guards.cleanup_state == "takeover_pending"
    assert parent_socket.sent == []
    parent_adapter._powered_transport.close()

    deadline_adapter, deadline_socket, deadline_guards, _prod, _scratch, peer = (
        _powered_adapter()
    )
    _manually_promote_powered_adapter(deadline_adapter, deadline_guards, peer)
    with pytest.raises(OutboundAuthorityError, match="deadline"):
        asyncio.run(
            deadline_adapter.disarm(
                powered_deadline_monotonic_ns=time.perf_counter_ns()
            )
        )
    assert deadline_guards.production_latched
    assert deadline_socket.sent == []
    deadline_adapter._powered_transport.close()

    position_adapter, position_socket, position_guards, _prod, _scratch, peer = (
        _powered_adapter()
    )
    _manually_promote_powered_adapter(position_adapter, position_guards, peer)
    with pytest.raises(OutboundAuthorityError, match="allowlisted"):
        asyncio.run(
            position_adapter.send_position(
                PositionCommand(
                    position_ned=(0.0, 0.0, 0.0),
                    velocity_ned=(0.0, 0.0, 0.0),
                    yaw_rad=0.0,
                ),
                powered_deadline_monotonic_ns=time.perf_counter_ns()
                + 1_000_000_000,
            )
        )
    assert position_guards.production_latched
    assert position_adapter.outbound_audit().position_target == 1
    assert position_socket.sent == []
    position_adapter._powered_transport.close()


def test_powered_connect_deadline_closes_socket_and_unblocks_receiver():
    adapter, raw_socket, guards, _production, _scratch, _peer = _powered_adapter()
    started = time.perf_counter()
    with pytest.raises((PoweredDeadlineExpired, RuntimeError)) as raised:
        asyncio.run(
            adapter.connect(
                deadline_monotonic_ns=time.perf_counter_ns() + 100_000_000
            )
        )
    # Connect failure cleanup is clipped to the original deadline. It may
    # truthfully replace the phase error with unproved-cleanup if worker exit
    # loses the zero-remaining scheduling race, but it gets no fresh grace.
    assert time.perf_counter() - started < 0.4
    if isinstance(raised.value, RuntimeError):
        assert "cleanup was unproved" in str(raised.value)
        assert isinstance(raised.value.__cause__, PoweredDeadlineExpired)
    assert raw_socket.closed
    state = adapter.powered_transport_state()
    assert state.endpoint_closed
    if isinstance(raised.value, PoweredDeadlineExpired):
        assert adapter._rx_thread is None
        assert adapter._announce_thread is None
    else:
        assert not state.owned_handles_closed
    assert guards.production_latched


def test_powered_disconnect_closes_before_bounded_stuck_worker_join():
    adapter, raw_socket, _guards, _production, _scratch, _peer = _powered_adapter()
    adapter._conn = adapter._powered_transport.connection

    class StuckWorker:
        name = "stuck-powered-rx"

        def __init__(self):
            self.join_timeouts = []

        def join(self, timeout):
            assert raw_socket.closed
            assert timeout >= 0.0
            self.join_timeouts.append(timeout)

        def is_alive(self):
            return True

    stuck = StuckWorker()
    adapter._rx_thread = stuck
    with pytest.raises(RuntimeError, match="termination unproved"):
        asyncio.run(
            adapter.disconnect(
                deadline_monotonic_ns=time.perf_counter_ns() + 100_000_000
            )
        )
    assert raw_socket.events[0] == ("close",)
    assert 1 <= len(stuck.join_timeouts) <= 3
    assert all(0.0 <= value <= 0.05 for value in stuck.join_timeouts)


def test_powered_disconnect_services_progress_between_50ms_join_slices():
    clock = [1_000]
    adapter, raw_socket, _guards, _production, _scratch, _peer = _powered_adapter(
        monotonic_ns=lambda: clock[0]
    )
    adapter._conn = adapter._powered_transport.connection

    class TwoSliceWorker:
        name = "two-slice-powered-rx"

        def __init__(self):
            self.alive = True
            self.timeouts = []

        def join(self, timeout):
            assert raw_socket.closed
            self.timeouts.append(timeout)
            clock[0] += int(timeout * 1_000_000_000)
            if len(self.timeouts) == 2:
                self.alive = False

        def is_alive(self):
            return self.alive

    worker = TwoSliceWorker()
    adapter._rx_thread = worker
    progress = []
    asyncio.run(
        adapter.disconnect(
            deadline_monotonic_ns=clock[0] + 200_000_000,
            powered_progress=lambda: progress.append(clock[0]),
        )
    )

    assert worker.timeouts == [0.05, 0.05]
    assert progress == [1_000, 50_001_000, 100_001_000]
    assert adapter.powered_transport_state().owned_handles_closed


def test_powered_disconnect_propagates_progress_failure_after_mandatory_close():
    clock = [10]
    adapter, raw_socket, guards, _production, _scratch, _peer = _powered_adapter(
        monotonic_ns=lambda: clock[0]
    )
    adapter._conn = adapter._powered_transport.connection

    class ClosingWorker:
        name = "closing-powered-rx"

        def __init__(self):
            self.alive = True

        def join(self, timeout):
            assert timeout <= 0.05
            clock[0] += int(timeout * 1_000_000_000)
            self.alive = False

        def is_alive(self):
            return self.alive

    class ProgressFailure(BaseException):
        pass

    adapter._rx_thread = ClosingWorker()

    def fail_progress():
        raise ProgressFailure("heartbeat failed")

    with pytest.raises(ProgressFailure, match="heartbeat failed"):
        asyncio.run(
            adapter.disconnect(
                deadline_monotonic_ns=clock[0] + 100_000_000,
                powered_progress=fail_progress,
            )
        )

    assert raw_socket.closed
    assert adapter.powered_transport_state().owned_handles_closed
    assert guards.production_latched


def test_raised_receipt_sanitizes_arbitrary_exception_class_name_without_masking():
    unusual_error = type("_\u2603" * 100, (Exception,), {})

    class UnusualMav(FakeMav):
        def set_attitude_target_send(self, *args):
            self.calls.append(("set_attitude_target_send", args))
            raise unusual_error("secret exception text")

    adapter = AIGPMavlinkAdapter(enable_vision=False)
    adapter._conn = FakeConn()
    adapter._conn.mav = UnusualMav()
    with pytest.raises(unusual_error, match="secret exception text"):
        asyncio.run(
            adapter.send_attitude_rate(
                AttitudeRateCommand(0.1, 0.0, 0.0, 0.2)
            )
        )
    receipts = adapter.drain_outbound_receipts()
    assert len(receipts) == 1
    assert receipts[0].outcome == "raised"
    assert receipts[0].error_type.startswith("ExceptionType-")
    assert len(receipts[0].error_type) <= 128
    assert "secret" not in receipts[0].error_type
    assert adapter.outbound_audit().attitude_target == 1
