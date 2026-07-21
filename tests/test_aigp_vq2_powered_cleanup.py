from __future__ import annotations

import io
import json
import socket
import threading
import time
from pathlib import Path

import pytest

from competition.aigp_mavlink import PoweredMavlinkTransport
from competition.aigp_messages import encode_race_status
from scripts import aigp_vq2_powered_attempt as real_contract
from scripts import aigp_vq2_powered_cleanup as cleanup
from scripts import aigp_vq2_powered_runtime as runtime


class StepClock:
    def __init__(self, value=1_000_000, step=1_000_000):
        self.value = value
        self.step = step

    def __call__(self):
        current = self.value
        self.value += self.step
        return current


def process_identity(pid, created, argv_hash, image="C:\\Python\\python.exe"):
    return {
        "pid": pid,
        "creation_filetime_100ns": created,
        "windows_session_id": 1,
        "image_path": image,
        "image_sha256": "a" * 64,
        "argv_sha256": argv_hash,
    }


class FakeContract:
    TASK_ID = real_contract.TASK_ID
    SESSION_ID = real_contract.SESSION_ID
    ATTEMPT_ID = real_contract.ATTEMPT_ID

    def __init__(self):
        self.certificate_calls = []
        self.result_calls = []

    @staticmethod
    def canonical_json_file_bytes(value):
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )

    @classmethod
    def canonical_file_sha256(cls, value):
        import hashlib

        return hashlib.sha256(cls.canonical_json_file_bytes(value)).hexdigest()

    @staticmethod
    def validate_attempt(value, *, live_freeze=None):
        assert type(value) is dict
        return json.loads(json.dumps(value))

    @staticmethod
    def validate_live_freeze(value):
        assert type(value) is dict
        return json.loads(json.dumps(value))

    @staticmethod
    def validate_process_authority(value, *, attempt=None, argv=None):
        assert attempt is not None
        assert list(argv) == attempt["context"]["cleanup_argv"]
        return json.loads(json.dumps(value))

    @staticmethod
    def validate_received_heartbeat(value):
        return real_contract.validate_received_heartbeat(value)

    @staticmethod
    def validate_received_race_status(value):
        return real_contract.validate_received_race_status(value)

    @staticmethod
    def validate_received_imu(value):
        return real_contract.validate_received_imu(value)

    @staticmethod
    def validate_nonattitude_outbound(value):
        assert type(value) is dict
        return json.loads(json.dumps(value))

    @staticmethod
    def validate_reset_boundary(value):
        assert type(value) is dict
        assert value["new_generation"] == value["old_generation"] + 1
        return json.loads(json.dumps(value))

    def validate_cleanup_certificate(self, value):
        assert value["producer_role"] == "cleanup_fallback"
        assert value["endpoints"]["camera"] is None
        self.certificate_calls.append(json.loads(json.dumps(value)))
        return json.loads(json.dumps(value))

    def validate_process_result(self, value, *, cleanup_certificate=None):
        assert cleanup_certificate is not None
        assert value["producer_role"] == "cleanup_fallback"
        self.result_calls.append(json.loads(json.dumps(value)))
        return json.loads(json.dumps(value))


class FakePipeOperations:
    def __init__(self, frame, *, parent_signaled=False):
        self.frame = frame
        self.peeks = [runtime.PipePeek(36, False), runtime.PipePeek(0, True)]
        self.parent_is_signaled = parent_signaled
        self.reads = []
        self.closed = []

    def peek_named_pipe(self, handle):
        return self.peeks.pop(0)

    def read_file(self, handle, size):
        self.reads.append((handle, size))
        return self.frame

    def process_signaled(self, handle):
        return self.parent_is_signaled

    def wait_ns(self, duration_ns):
        raise AssertionError("complete fake capability must not wait")

    def close_handle(self, handle):
        self.closed.append(handle)


class FakeProcessBoundary:
    def __init__(self, argv, current, parent, state):
        self.argv = list(argv)
        self.current = current
        self.parent = parent
        self.state = state
        self.policy_calls = []

    def current_argv(self):
        return list(self.argv)

    def current_process_identity(self):
        return dict(self.current)

    def retained_process_identity(self, handle):
        assert handle == 22
        return dict(self.parent)

    def prove_inherited_handle_policy(
        self, *, capability_handle, parent_handle, process_authority
    ):
        self.policy_calls.append((capability_handle, parent_handle))
        return self.state.get("handle_policy", True)

    def parent_signaled(self, handle):
        assert handle == 22
        return self.state.get("parent_dead", False)


class LiveLeaseEvidenceError(RuntimeError):
    pass


class FakeLeaseBoundary:
    def __init__(self, clock, state):
        self.clock = clock
        self.state = state
        self.takeovers = 0
        self.releases = 0
        self.secret_snapshots = []
        self.heartbeats = []
        self.heartbeat_attempts = []
        self.latest_proof = None
        self.last_heartbeat_ns = None

    def prove_live_delegation(self, *, attempt, process_authority):
        self.latest_proof = cleanup.LeaseProof("wrapper", 7, "1" * 64, True)
        return self.latest_proof

    def take_over_abandoned(
        self,
        *,
        role_secret,
        attempt,
        process_authority,
        deadline_monotonic_ns,
    ):
        self.takeovers += 1
        self.secret_snapshots.append(bytes(role_secret))
        if self.state.get("takeover_fail"):
            raise RuntimeError("injected takeover failure")
        completed = self.clock()
        assert completed < deadline_monotonic_ns
        self.latest_proof = cleanup.LeaseProof(
            "cleanup-fallback-parent-death",
            8,
            "2" * 64,
            True,
            takeover_completed_monotonic_ns=completed,
        )
        self.last_heartbeat_ns = completed
        return self.latest_proof

    def heartbeat_takeover(
        self,
        proof,
        *,
        phase,
        deadline_monotonic_ns,
    ):
        assert proof == self.latest_proof
        now = self.clock()
        assert now < deadline_monotonic_ns
        occurrence = (now, phase, threading.get_ident(), proof)
        self.heartbeat_attempts.append(occurrence)
        if (
            self.state.get("enforce_heartbeat_cadence")
            and now < self.last_heartbeat_ns + 1_000_000_000
        ):
            raise LiveLeaseEvidenceError(
                "delegated powered heartbeat preceded its frozen one-second cadence"
            )
        self.heartbeats.append(occurrence)
        if self.state.get("stale_heartbeat"):
            return proof
        self.latest_proof = cleanup.LeaseProof(
            "cleanup-fallback-parent-death",
            proof.generation + 1,
            f"{proof.generation + 1:064x}",
            True,
            takeover_completed_monotonic_ns=(
                proof.takeover_completed_monotonic_ns
            ),
        )
        self.last_heartbeat_ns = now
        return self.latest_proof

    def release_takeover(self, proof, *, deadline_monotonic_ns):
        self.releases += 1
        assert proof == self.latest_proof
        return self.state.get("release_ok", True)


class FakePublisher:
    def __init__(self, contract, state):
        self.contract = contract
        self.state = state
        self.calls = []

    def publish_create_new(
        self,
        path,
        value,
        *,
        deadline_monotonic_ns,
        progress_callback=None,
    ):
        publish_advance = self.state.pop("heartbeat_during_publish", False)
        if publish_advance:
            self.state["_clock"].value += (
                1_000_000_000
                if publish_advance is True
                else publish_advance
            )
        if progress_callback is not None:
            progress_callback()
        self.calls.append((path, json.loads(json.dumps(value)), deadline_monotonic_ns))
        if self.state.get("die_on_publish"):
            self.state["parent_dead"] = True
        if progress_callback is not None:
            progress_callback()
        return self.contract.canonical_file_sha256(value)


class FakeMessage:
    def __init__(self, raw):
        self.raw = raw

    def get_type(self):
        return "HEARTBEAT"

    def get_msgbuf(self):
        return self.raw

    def get_msgId(self):
        return self.raw[5]


class FakeParser:
    def __init__(self):
        self.robust_parsing = True

    def parse_buffer(self, raw):
        return [FakeMessage(raw)]


def mav_frame(message_id):
    return bytes((0xFE, 0, message_id, 1, 1, message_id, 0, 0))


def adapter_mav_frame(message_id, payload=b""):
    return (
        bytes((0xFE, len(payload), 1, 1, 1, message_id))
        + bytes(payload)
        + b"\x00\x00"
    )


class ManualClock:
    def __init__(self, value=1_000_000_000):
        self.value = value

    def __call__(self):
        return self.value


class AdapterRawSocket:
    family = socket.AF_INET
    type = socket.SOCK_DGRAM

    def __init__(self, port=45_001):
        self.port = port
        self.closed = False
        self.timeout = None
        self.incoming = []
        self.sent = []

    def getsockname(self):
        return ("127.0.0.1", self.port)

    def getsockopt(self, _level, option):
        if option == socket.SO_REUSEADDR:
            return 0
        return 1

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeout = value

    def recvfrom(self, _capacity):
        if self.closed:
            raise OSError("adapter test socket is closed")
        if not self.incoming:
            raise socket.timeout("adapter test receive timeout")
        return self.incoming.pop(0)

    def sendto(self, payload, peer):
        if self.closed:
            raise OSError("adapter test socket is closed")
        raw = bytes(payload)
        self.sent.append((raw, peer))
        return len(raw)

    def close(self):
        self.closed = True

    def push(self, payload, peer):
        self.incoming.append((bytes(payload), peer))


class AdapterProductionMessage:
    def __init__(self, raw, message_type, **fields):
        self.raw = bytes(raw)
        self.message_type = message_type
        for name, value in fields.items():
            setattr(self, name, value)

    def get_type(self):
        return self.message_type

    def get_msgbuf(self):
        return self.raw

    def get_srcSystem(self):
        return 1

    def get_srcComponent(self):
        return 1


class AdapterProductionMavlink:
    def __init__(self, writer):
        self.file = writer
        self.robust_parsing = True
        self.parsed = []
        self.calls = []

    def parse_buffer(self, raw):
        raw = bytes(raw)
        self.parsed.append(raw)
        message_id = raw[5]
        payload = raw[6:-2]
        if message_id == 0:
            message = AdapterProductionMessage(
                raw,
                "HEARTBEAT",
                base_mode=0,
                custom_mode=0,
            )
        elif message_id == 105:
            message = AdapterProductionMessage(
                raw,
                "HIGHRES_IMU",
                time_usec=100_000,
                xacc=0.0,
                yacc=0.0,
                zacc=-9.0,
                xgyro=0.0,
                ygyro=0.0,
                zgyro=0.0,
            )
        elif message_id == 131:
            message = AdapterProductionMessage(
                raw,
                "ENCAPSULATED_DATA",
                data=payload,
                seqnr=0,
            )
        elif message_id == 247:
            if len(payload) != 3:
                raise AssertionError("collision test payload must have three bytes")
            message = AdapterProductionMessage(
                raw,
                "COLLISION",
                id=payload[0],
                threat_level=payload[1],
                horizontal_minimum_delta=float(payload[2]),
            )
        else:
            message = AdapterProductionMessage(
                raw,
                "STATUSTEXT",
                text="ignored",
                severity=6,
            )
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


def ingress(message_type, generation, sequence, received, source_value, unit):
    return {
        "schema": "aigp-vq2-mavlink-ingress/1",
        "stream_id": "test-stream",
        "generation": generation,
        "sequence": sequence,
        "message_type": message_type,
        "host_clock_id": "host-perf-counter",
        "received_monotonic_ns": received,
        "source_time_value": source_value,
        "source_time_unit": unit,
    }


def heartbeat(generation, sequence, received, *, armed=False):
    return {
        "schema": "aigp-vq2-received-heartbeat/1",
        "ingress": ingress("HEARTBEAT", generation, sequence, received, None, None),
        "heartbeat": {"base_mode": 128 if armed else 0, "custom_mode": 0},
    }


def race(generation, sequence, received, boot_ms):
    return {
        "schema": "aigp-vq2-received-race-status/1",
        "ingress": ingress(
            "RACE_STATUS", generation, sequence, received, boot_ms, "ms"
        ),
        "race_status": {
            "sim_boot_time_ms": boot_ms,
            "race_start_boot_time_ms": -1,
            "race_finish_time_ns": -1,
            "active_gate_index": 0,
            "last_gate_race_time": -1,
        },
    }


def imu(generation, sequence, received, timestamp_us):
    return {
        "schema": "aigp-vq2-received-imu/1",
        "ingress": ingress(
            "HIGHRES_IMU", generation, sequence, received, timestamp_us, "us"
        ),
        "imu": {
            "timestamp_us": timestamp_us,
            "accel": [0.0, 0.0, 0.0],
            "gyro": [0.0, 0.0, 0.0],
            "mag": None,
        },
    }


def receipt(category, sequence, call_start):
    return {
        "schema": "fake-nonattitude",
        "category": category,
        "outcome": "returned",
        "outbound_sequence": sequence,
        "call_start_monotonic_ns": call_start,
        "call_end_monotonic_ns": call_start + 1,
    }


def real_attitude_receipt(sequence, call_start):
    return {
        "schema": "aigp-vq2-attitude-target-outbound/1",
        "stream_id": "test-stream",
        "reset_generation": 0,
        "outbound_sequence": sequence,
        "host_clock_id": "host-perf-counter",
        "call_start_monotonic_ns": call_start,
        "call_end_monotonic_ns": call_start + 1,
        "api": "send_attitude_rate",
        "outcome": "returned",
        "error_type": None,
        "wire": {
            "time_boot_ms": 0,
            "target_system": 1,
            "target_component": 1,
            "type_mask": 128,
            "q_wxyz": [1.0, 0.0, 0.0, 0.0],
            "body_rates_rad_s": [0.0, 0.0, 0.0],
            "thrust": 0.0,
        },
    }


def real_nonattitude_receipt(category, sequence, call_start, *, generation=0):
    return {
        "schema": "aigp-vq2-nonattitude-outbound/1",
        "stream_id": "test-stream",
        "reset_generation": generation,
        "outbound_sequence": sequence,
        "host_clock_id": "host-perf-counter",
        "call_start_monotonic_ns": call_start,
        "call_end_monotonic_ns": call_start + 1,
        "category": category,
        "api": "command_long_send",
        "outcome": "returned",
        "error_type": None,
        "wire": {
            "target_system": 1,
            "target_component": 1,
            "command": 400 if category == "disarm" else 31_000,
            "confirmation": 0,
            "params": [0.0] * 7,
        },
    }


def real_reset_boundary(boundary_monotonic_ns):
    return {
        "schema": "aigp-vq2-calibration-reset-boundary/1",
        "old_generation": 0,
        "new_generation": 1,
        "boundary_monotonic_ns": boundary_monotonic_ns,
        "observations": [],
        "collisions": [],
        "ingress_stats": {
            "generation": 0,
            "next_sequence": 4,
            "highres_imu_received": 1,
            "heartbeat_received": 2,
            "race_status_received": 1,
            "actuator_received": 0,
            "dropped": 0,
            "high_watermark": 1,
            "imu_capacity": 1,
            "other_capacity": 1,
            "imu_dropped": 0,
            "other_dropped": 0,
            "imu_high_watermark": 1,
            "other_high_watermark": 1,
            "buffered_imu": 0,
            "buffered_other": 0,
        },
        "collision_stats": {
            "generation": 0,
            "handled": 0,
            "dropped": 0,
            "high_watermark": 0,
            "capacity": 1,
            "buffered": 0,
        },
    }


def runner_collision(
    generation,
    sequence,
    observed,
    *,
    collision_id,
    phase,
    disposition,
):
    return {
        "schema": "aigp-vq2-runner-collision-observation/1",
        "reset_generation": generation,
        "observation_sequence": sequence,
        "host_clock_id": real_contract.HOST_CLOCK_ID,
        "observed_monotonic_ns": observed,
        "phase": phase,
        "disposition": disposition,
        "boundary": "runner_drain_not_receiver_receipt",
        "collision": {
            "id": collision_id,
            "threat_level": 2,
            "impulse": 3.0,
        },
    }


def real_zero_command(receipt_value):
    generated_time = receipt_value["call_start_monotonic_ns"]
    watchdogs = {
        "checked_monotonic_ns": generated_time,
        "heartbeat_age_ns": None,
        "imu_age_ns": None,
        "imu_advance_age_ns": None,
        "race_age_ns": None,
        "race_advance_age_ns": None,
        "actuator_age_ns": None,
        "vision_age_ns": None,
        "estimator_healthy": None,
        "target_consecutive": None,
        "target_center_px": None,
        "target_bbox_px": None,
        "target_bbox_area_px": None,
        "initial_target_bbox_area_px": None,
        "roll_excursion_rad": None,
        "pitch_excursion_rad": None,
        "collision_count": None,
        "gate_index": None,
        "result": "cleanup_authorized",
        "failure_codes": [],
    }
    common = {
        "attempt_id": real_contract.ATTEMPT_ID,
        "session_id": real_contract.SESSION_ID,
        "candidate_commit": "0" * 40,
        "attempt_context_sha256": "c" * 64,
        "host_clock_id": "host-perf-counter",
        "reset_epoch": None,
        "plan": None,
        "scope": "cleanup_zero",
        "command_id": "cleanup/zero/0",
        "absolute_tick": None,
        "segment_id": None,
        "slot": None,
        "command": dict(cleanup.ZERO_COMMAND),
        "source": {
            "frame": None,
            "imu": None,
            "race": None,
            "heartbeat": None,
            "actuator": None,
        },
        "watchdogs": watchdogs,
    }
    generated = {
        **common,
        "schema": "aigp-vq2-calibration-command-generated/1",
        "event_sequence": 1,
        "generated_monotonic_ns": generated_time,
    }
    terminal = {
        **common,
        "schema": "aigp-vq2-calibration-command-sent/1",
        "event_sequence": 2,
        "sent_monotonic_ns": receipt_value["call_end_monotonic_ns"],
        "generated_event_sequence": 1,
        "generation_sha256": real_contract.canonical_object_sha256(generated),
        "transport": {
            "receipt": receipt_value,
            "audit_count_before": 0,
            "audit_count_after": 1,
        },
    }
    return {
        "state": "returned",
        "required": True,
        "requested": dict(cleanup.ZERO_COMMAND),
        "generated": generated,
        "terminal": terminal,
        "outbound_receipt": receipt_value,
    }


class FakeBackend:
    def __init__(
        self,
        endpoint,
        state,
        *,
        second_source=False,
        nonloopback_source=False,
        empty=False,
        bad_reset_order=False,
        announce_after_freeze=False,
        boundary_collision=False,
        post_boundary_collision=False,
        authorize_outbound=None,
    ):
        self.endpoint = endpoint
        self.authorize_outbound = authorize_outbound
        self.state = state
        self.forwarded = []
        self.waits = []
        self.pending = []
        self.current = None
        self.stop_calls = 0
        self.join_calls = 0
        self.zero_calls = 0
        self.disarm_calls = 0
        self.reset_calls = 0
        self.receipts = []
        self._source_authority = runtime.MavlinkSourceFreeze(FakeParser)
        self.open_calls = 0
        peer = ("127.0.0.1", 40_001)
        other = ("127.0.0.1", 40_002)
        self.queue = [] if empty else [
            (cleanup.ReceivedDatagram(mav_frame(1), peer), [heartbeat(0, 0, 1_000_000_000)]),
            (cleanup.ReceivedDatagram(mav_frame(2), peer), [race(0, 1, 1_000_000_001, 1_000)]),
            (cleanup.ReceivedDatagram(mav_frame(3), peer), [imu(0, 2, 1_000_000_002, 100_000)]),
        ]
        self.second_source = second_source
        self.nonloopback_source = nonloopback_source
        self.bad_reset_order = bad_reset_order
        self.announce_after_freeze = announce_after_freeze
        self.boundary_collision = boundary_collision
        self.post_boundary_collision = post_boundary_collision
        self._post_boundary_collision_recorded = False
        self.collision_rows = []
        self.announcement_deadlines = []
        self.peer = peer
        self.other = other

    @property
    def source_authority(self):
        return self._source_authority

    def open(self, *, deadline_monotonic_ns):
        assert deadline_monotonic_ns > 0
        self.open_calls += 1

    def receive_and_dispatch_datagram(self, max_wait_ns):
        self.waits.append(max_wait_ns)
        if not self.queue:
            return None
        self.current = self.queue.pop(0)
        datagram = self.current[0]
        decision = self._source_authority.ingest(
            datagram.payload,
            datagram.source,
        )
        if decision.rejected_source:
            return cleanup.CleanupDatagramDispatch(
                False,
                False,
                True,
                False,
                False,
                self._source_authority.promoted,
                self._source_authority.peer,
            )
        if decision.malformed:
            return cleanup.CleanupDatagramDispatch(
                False,
                False,
                False,
                True,
                False,
                self._source_authority.promoted,
                self._source_authority.peer,
            )
        assert decision.accepted
        payload = datagram.payload
        source = datagram.source
        assert type(payload) is bytes
        assert self.current is not None
        assert payload == self.current[0].payload
        assert source == self.current[0].source
        self.forwarded.append(payload)
        if self.announce_after_freeze and len(self.forwarded) == 1:
            self.announcement_deadlines.append(
                self.authorize_outbound("timesync")
            )
        self.pending.extend(self.current[1])
        for observation in self.current[1]:
            message_type = observation["ingress"]["message_type"]
            if message_type in {"HEARTBEAT", "RACE_STATUS", "HIGHRES_IMU"}:
                self._source_authority.observe_fresh_stream(message_type)
        return cleanup.CleanupDatagramDispatch(
            True,
            decision.peer_frozen_now,
            False,
            False,
            True,
            self._source_authority.promoted,
            self._source_authority.peer,
        )

    def drain_received_observations(self):
        values = list(self.pending)
        self.pending.clear()
        return values

    def send_exact_zero(self, command, *, deadline_monotonic_ns):
        self.zero_calls += 1
        assert command == cleanup.ZERO_COMMAND
        zero_receipt = {
            "schema": "fake-attitude",
            "category": "attitude_target",
            "outcome": "returned",
            "outbound_sequence": 0,
        }
        self.receipts.append(zero_receipt)
        if self.state.get("stall_zero"):
            self.state["_clock"].value = deadline_monotonic_ns
        return {
            "state": "returned",
            "required": True,
            "requested": dict(command),
            "generated": {"schema": "fake-generated"},
            "terminal": {"schema": "fake-sent"},
            "outbound_receipt": zero_receipt,
        }

    def send_disarm(self, *, deadline_monotonic_ns):
        self.disarm_calls += 1
        request = 1_050_000_000
        value = receipt("disarm", 1, request)
        self.receipts.append(value)
        if self.second_source:
            self.queue.append(
                (
                    cleanup.ReceivedDatagram(
                        mav_frame(4),
                        (
                            ("192.0.2.1", 40_002)
                            if self.nonloopback_source
                            else self.other
                        ),
                    ),
                    [],
                )
            )
        self.queue.append(
            (
                cleanup.ReceivedDatagram(mav_frame(5), self.peer),
                [heartbeat(0, 3, 1_100_000_000)],
            )
        )
        if self.state.get("die_after_disarm"):
            self.state["parent_dead"] = True
        return cleanup.NonattitudeDispatch(request, "returned", value)

    def send_reset(self, *, baseline, deadline_monotonic_ns):
        self.reset_calls += 1
        request = 1_150_000_000
        value = receipt("sim_reset", 2, request)
        self.receipts.append(value)
        self.queue.extend(
            [
                (cleanup.ReceivedDatagram(mav_frame(6), self.peer), [race(1, 0, 1_200_000_000, 10)]),
                (cleanup.ReceivedDatagram(mav_frame(7), self.peer), [imu(1, 1, 1_200_000_001, 100)]),
                (cleanup.ReceivedDatagram(mav_frame(8), self.peer), [heartbeat(1, 2, 1_200_000_002)]),
                (cleanup.ReceivedDatagram(mav_frame(9), self.peer), [race(1, 3, 1_200_000_003, 11)]),
                (cleanup.ReceivedDatagram(mav_frame(10), self.peer), [imu(1, 4, 1_200_000_004, 101)]),
                (
                    cleanup.ReceivedDatagram(mav_frame(11), self.peer),
                    [
                        race(
                            1,
                            5,
                            1_200_000_005,
                            9 if self.bad_reset_order else 12,
                        )
                    ],
                ),
                (cleanup.ReceivedDatagram(mav_frame(12), self.peer), [imu(1, 6, 1_200_000_006, 102)]),
            ]
        )
        if self.bad_reset_order:
            self.queue.append(
                (
                    cleanup.ReceivedDatagram(mav_frame(13), self.peer),
                    [race(1, 7, 1_200_000_007, 12)],
                )
            )
        boundary = {"old_generation": 0, "new_generation": 1, "collisions": []}
        if self.boundary_collision:
            boundary_row = runner_collision(
                0,
                0,
                1_150_000_001,
                collision_id=41,
                phase="fallback-reset-and-epoch",
                disposition="reset_boundary_discard",
            )
            boundary["collisions"].append(boundary_row)
            self.collision_rows.append(boundary_row)
        return cleanup.ResetDispatch(
            cleanup.NonattitudeDispatch(request, "returned", value),
            boundary,
        )

    def outbound_receipts(self):
        return list(self.receipts)

    def outbound_audit(self):
        return {
            **cleanup._zero_outbound_audit(),
            "sim_reset": self.reset_calls,
            "disarm": self.disarm_calls,
            "attitude_target": self.zero_calls,
            "receipt_count": len(self.receipts),
            "receipt_returned": len(self.receipts),
        }

    def collision_observations(self):
        if self.post_boundary_collision and not self._post_boundary_collision_recorded:
            self._post_boundary_collision_recorded = True
            self.collision_rows.append(
                runner_collision(
                    1,
                    0,
                    1_200_000_008,
                    collision_id=42,
                    phase="fallback-finalize",
                    disposition="cleanup_continue",
                )
            )
        return json.loads(json.dumps(self.collision_rows))

    def request_stop(self):
        self.stop_calls += 1

    def join_workers(
        self,
        *,
        deadline_monotonic_ns,
        progress_callback=None,
    ):
        self.join_calls += 1
        join_advance = self.state.pop("heartbeat_during_join", False)
        increments = (
            [1_000_000_000]
            if join_advance is True
            else list(join_advance)
            if type(join_advance) in {list, tuple}
            else [join_advance]
            if join_advance
            else [0]
        )
        for increment in increments:
            self.state["_clock"].value += increment
            if progress_callback is not None:
                progress_callback()
        return cleanup.WorkerCloseProof(True, True, True)


def build_fixture(*, state=None, backend_options=None, clock=None):
    state = {} if state is None else state
    backend_options = {} if backend_options is None else backend_options
    clock = StepClock() if clock is None else clock
    state["_clock"] = clock
    contract = FakeContract()
    secret = bytes(range(32))
    context_hash = "c" * 64
    expected_capability = runtime.derive_capability_sha256(
        cleanup.CAPABILITY_DOMAIN,
        context_hash,
        secret,
    )
    arguments = cleanup.CleanupArguments(
        powered_attempt_envelope="C:\\evidence\\attempt.json",
        wrapper_process="100:200",
        powered_process_authority="C:\\evidence\\cleanup-authority.json",
        cleanup_capability_handle="11",
        parent_liveness_handle="22",
        cleanup_certificate="C:\\evidence\\fallback-cleanup-certificate.json",
    )
    argv = [
        "C:\\Python\\python.exe",
        "-E",
        "-s",
        "-B",
        "-m",
        "scripts.aigp_vq2_powered_cleanup",
        "--powered-attempt-envelope",
        arguments.powered_attempt_envelope,
        "--wrapper-process",
        arguments.wrapper_process,
        "--powered-process-authority",
        arguments.powered_process_authority,
        "--cleanup-capability-handle",
        arguments.cleanup_capability_handle,
        "--parent-liveness-handle",
        arguments.parent_liveness_handle,
        "--cleanup-certificate",
        arguments.cleanup_certificate,
    ]
    argv_hash = runtime.argv_sha256(argv)
    parent = process_identity(100, 200, "d" * 64, "C:\\wrapper.exe")
    current = process_identity(101, 201, argv_hash)
    live_freeze_path = "C:\\evidence\\live-freeze.json"
    durations = dict(real_contract.DEADLINE_DURATIONS_NS)
    attempt = {
        "context": {
            "live_freeze": {"path": live_freeze_path, "sha256": "e" * 64},
            "paths": {
                "attempt_envelope": arguments.powered_attempt_envelope,
                "cleanup_authority": arguments.powered_process_authority,
                "fallback_cleanup_certificate": arguments.cleanup_certificate,
            },
            "cleanup_argv": argv,
            "wrapper_process": parent,
            "deadline_durations_ns": durations,
        },
        "context_sha256": context_hash,
    }
    live_freeze = {
        "transport": {
            "mavlink_bind": {
                "host": "127.0.0.1",
                "port": 0,
                "socket_policy": "ipv4-exclusive-address-use",
            }
        }
    }
    authority = {
        "role": "cleanup_fallback",
        "process": current,
        "wrapper_process": parent,
        "parent_handle": {"value": 22},
        "capability_sha256": expected_capability,
        "lease_record_sha256": "1" * 64,
        "absolute_deadlines": {
            "anchor": 0,
            "total": 25_000_000_000,
            "exit": 25_000_000_000,
        },
    }
    documents = {
        arguments.powered_attempt_envelope: attempt,
        live_freeze_path: live_freeze,
        arguments.powered_process_authority: authority,
    }

    def load_record(path, _contract):
        return json.loads(json.dumps(documents[path]))

    process = FakeProcessBoundary(argv, current, parent, state)
    lease = FakeLeaseBoundary(clock, state)
    publisher = FakePublisher(contract, state)
    created_backends = []

    def backend_factory(endpoint, authority):
        backend = FakeBackend(
            endpoint,
            state,
            authorize_outbound=authority.authorize_outbound,
            **backend_options,
        )
        created_backends.append(backend)
        return backend

    services = cleanup.CleanupServices(
        process_boundary=process,
        lease_boundary=lease,
        backend_factory=backend_factory,
        publisher=publisher,
        capability_operations=FakePipeOperations(
            runtime.encode_capability_frame(secret)
        ),
        monotonic_ns=clock,
        contract=contract,
        load_record=load_record,
    )
    return {
        "arguments": arguments,
        "services": services,
        "contract": contract,
        "clock": clock,
        "state": state,
        "lease": lease,
        "publisher": publisher,
        "backends": created_backends,
        "secret": secret,
        "authority": authority,
    }


def argument_tail(arguments):
    return [
        "--powered-attempt-envelope",
        arguments.powered_attempt_envelope,
        "--wrapper-process",
        arguments.wrapper_process,
        "--powered-process-authority",
        arguments.powered_process_authority,
        "--cleanup-capability-handle",
        arguments.cleanup_capability_handle,
        "--parent-liveness-handle",
        arguments.parent_liveness_handle,
        "--cleanup-certificate",
        arguments.cleanup_certificate,
    ]


def real_contract_round_trip(output, *, nominal):
    """Validate generated fallback output with production-shaped adapter evidence.

    The test fixture binds port zero.  Replacing that recorded port with the
    frozen production literal is an evidence-only operation and never opens or
    sends through the production endpoint.
    """

    certificate = json.loads(json.dumps(output.certificate))
    paths = real_contract.frozen_paths()
    certificate["authority"]["process_authority"]["path"] = paths[
        "cleanup_authority"
    ]
    endpoint = certificate["endpoints"]["mavlink"]
    if endpoint["bind"] is not None:
        endpoint["bind"]["requested"] = {"host": "127.0.0.1", "port": 14_550}
        endpoint["bind"]["actual"] = {"host": "127.0.0.1", "port": 14_550}

    if nominal:
        zero_start = certificate["started_monotonic_ns"] + 1
        assert zero_start + 1 <= certificate["disarm"]["request_monotonic_ns"]
        zero_receipt = real_attitude_receipt(0, zero_start)
        disarm_receipt = real_nonattitude_receipt(
            "disarm",
            1,
            1_050_000_001,
        )
        reset_receipt = real_nonattitude_receipt(
            "sim_reset",
            2,
            1_150_000_002,
            generation=1,
        )
        certificate["outbound_receipts"] = [
            zero_receipt,
            disarm_receipt,
            reset_receipt,
        ]
        certificate["zero_command"] = real_zero_command(zero_receipt)
        certificate["disarm"]["receipt"] = disarm_receipt
        certificate["reset"]["receipt"] = reset_receipt
        boundary_collisions = certificate["reset"]["boundary"].get(
            "collisions", []
        )
        boundary = real_reset_boundary(1_150_000_001)
        boundary["collisions"] = boundary_collisions
        boundary["collision_stats"].update(
            {
                "handled": len(boundary_collisions),
                "high_watermark": len(boundary_collisions),
                "buffered": len(boundary_collisions),
            }
        )
        certificate["reset"]["boundary"] = boundary
        observed_times = [
            row["ingress"]["received_monotonic_ns"]
            for name in ("advancing_race", "advancing_imu")
            for row in certificate["reset"][name]
        ]
        if certificate["final_state"]["state"] != "unobserved":
            observed_times.extend(
                certificate["final_state"][name]["ingress"][
                    "received_monotonic_ns"
                ]
                for name in ("heartbeat", "last_race", "last_imu")
            )
        observed_times.extend(
            row["observed_monotonic_ns"]
            for row in certificate["collisions"]["observations"]
        )
        certificate["completed_monotonic_ns"] = max(
            certificate["completed_monotonic_ns"],
            reset_receipt["call_end_monotonic_ns"],
            *observed_times,
        ) + 1
        assert (
            certificate["completed_monotonic_ns"]
            < certificate["deadline_monotonic_ns"]
        )

    certificate_bytes = real_contract.canonical_json_file_bytes(certificate)
    certificate_from_wire = json.loads(certificate_bytes)
    checked_certificate = real_contract.validate_cleanup_certificate(
        certificate_from_wire
    )
    assert certificate_bytes == real_contract.canonical_json_file_bytes(
        checked_certificate
    )

    result = json.loads(json.dumps(output.process_result))
    result["completed_monotonic_ns"] = max(
        result["completed_monotonic_ns"],
        checked_certificate["completed_monotonic_ns"],
    )
    result["cleanup_certificate"] = {
        "path": paths["fallback_cleanup_certificate"],
        "state": "published",
        "sha256": real_contract.canonical_file_sha256(checked_certificate),
    }
    result_bytes = real_contract.canonical_json_file_bytes(result)
    result_from_wire = json.loads(result_bytes)
    checked_result = real_contract.validate_process_result(
        result_from_wire,
        cleanup_certificate=checked_certificate,
    )
    assert result_bytes == real_contract.canonical_json_file_bytes(checked_result)
    return checked_certificate, checked_result


def test_cleanup_cli_is_exact_mandatory_and_disallows_abbreviation_or_extras():
    fixture = build_fixture()
    arguments = fixture["arguments"]
    assert cleanup.parse_cleanup_arguments(argument_tail(arguments)) == arguments
    with pytest.raises(SystemExit):
        cleanup.parse_cleanup_arguments(argument_tail(arguments)[:-2])
    abbreviated = argument_tail(arguments)
    abbreviated[0] = "--powered-attempt-envel"
    with pytest.raises(SystemExit):
        cleanup.parse_cleanup_arguments(abbreviated)
    with pytest.raises(SystemExit):
        cleanup.parse_cleanup_arguments(argument_tail(arguments) + ["--stage", "x"])


def test_admission_consumes_exact_capability_after_identity_and_handle_proof():
    fixture = build_fixture()
    admission = cleanup.admit_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    pipe = fixture["services"].capability_operations
    assert bytes(admission.role_secret) == fixture["secret"]
    assert pipe.reads == [(11, 36)]
    assert pipe.closed == [11]
    assert fixture["services"].process_boundary.policy_calls == [(11, 22)]
    assert fixture["backends"] == []
    assert fixture["publisher"].calls == []
    admission.erase_role_secret()
    assert bytes(admission.role_secret) == bytes(32)


def test_failed_capability_or_process_identity_never_constructs_live_backend():
    identity_failure = build_fixture()
    wrong_current = dict(identity_failure["services"].process_boundary.current)
    wrong_current["pid"] = 999
    identity_failure["services"].process_boundary.current = wrong_current
    with pytest.raises(cleanup.CleanupBootstrapError, match="current process"):
        cleanup.admit_cleanup_fallback(
            identity_failure["arguments"], identity_failure["services"]
        )
    assert identity_failure["services"].capability_operations.reads == []
    assert identity_failure["backends"] == []

    capability_failure = build_fixture()
    capability_failure["authority"]["capability_sha256"] = "f" * 64
    with pytest.raises(cleanup.CleanupBootstrapError, match="capability"):
        cleanup.admit_cleanup_fallback(
            capability_failure["arguments"], capability_failure["services"]
        )
    assert capability_failure["services"].capability_operations.closed == [11]
    assert capability_failure["backends"] == []
    assert capability_failure["publisher"].calls == []


def test_post_capability_lease_factory_precedes_live_backend_construction():
    fixture = build_fixture()
    lease = fixture["lease"]
    fixture["services"].lease_boundary = None
    calls = []

    def make_lease(admission):
        assert bytes(admission.role_secret) == fixture["secret"]
        assert fixture["services"].capability_operations.closed == [11]
        assert fixture["backends"] == []
        calls.append("lease")
        return lease

    fixture["services"].lease_boundary_factory = make_lease
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    assert calls == ["lease"]
    assert len(fixture["backends"]) == 1
    assert output.process_result["outcome"] == "completed"


def test_nominal_cleanup_uses_one_ephemeral_exclusive_bind_and_exact_order():
    fixture = build_fixture()
    admission = cleanup.admit_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    output = cleanup.CleanupFallbackMachine(admission, fixture["services"]).run()
    backend = fixture["backends"][0]
    assert output.certificate["outcome"] == "proved"
    assert output.process_result["outcome"] == "completed"
    assert output.exit_code == 0
    assert [item["phase"] for item in output.certificate["phase_deadlines"]] == [
        "connect",
        "disarm",
        "reset_and_epoch",
        "finalize",
    ]
    endpoint = output.certificate["endpoints"]["mavlink"]
    assert endpoint["bind"]["requested"]["port"] == 0
    assert 0 < endpoint["bind"]["actual"]["port"] <= 65_535
    assert endpoint["state"] == "closed_with_peer"
    assert output.certificate["endpoints"]["camera"] is None
    assert backend.zero_calls == backend.disarm_calls == backend.reset_calls == 1
    assert backend.stop_calls == backend.join_calls == 1
    assert backend.endpoint.closed
    assert all(type(item) is bytes for item in backend.forwarded)
    assert all(wait <= runtime.MAX_POLL_INTERVAL_NS for wait in backend.waits)
    assert fixture["publisher"].calls and len(fixture["publisher"].calls) == 1
    assert bytes(admission.role_secret) == bytes(32)
    with pytest.raises(runtime.OutboundAuthorityError):
        backend.authorize_outbound("timesync")


def test_generated_nominal_certificate_and_result_round_trip_real_contract():
    fixture = build_fixture()
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    assert output.certificate["endpoints"]["mavlink"]["bind"]["requested"][
        "port"
    ] == 0
    certificate, result = real_contract_round_trip(output, nominal=True)
    assert certificate["outcome"] == "proved"
    assert certificate["zero_command"]["state"] == "returned"
    assert result["outcome"] == "completed"


def test_fallback_certificate_accumulates_boundary_and_post_boundary_collisions():
    fixture = build_fixture(
        backend_options={
            "boundary_collision": True,
            "post_boundary_collision": True,
        }
    )
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )

    certificate = output.certificate
    rows = certificate["collisions"]["observations"]
    boundary_rows = certificate["reset"]["boundary"]["collisions"]
    assert certificate["outcome"] == "proved"
    assert certificate["collection_invalidating_codes"] == ["collision_observed"]
    assert certificate["collisions"]["invalidating_occurrence_count"] == 2
    assert [item["reset_generation"] for item in rows] == [0, 1]
    assert [item["observation_sequence"] for item in rows] == [0, 0]
    assert [item["collision"]["id"] for item in rows] == [41, 42]
    assert boundary_rows == [rows[0]]
    assert rows.count(boundary_rows[0]) == 1

    checked_certificate, checked_result = real_contract_round_trip(
        output,
        nominal=True,
    )
    assert checked_certificate["collisions"]["observations"] == rows
    assert checked_certificate["reset"]["boundary"]["collisions"] == [rows[0]]
    assert checked_certificate["collection_invalidating_codes"] == [
        "collision_observed"
    ]
    assert checked_result["outcome"] == "failed"
    assert checked_result["reason_codes"] == ["capture_incomplete"]


def test_scratch_message_never_enters_production_parser_state():
    fixture = build_fixture()
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    backend = fixture["backends"][0]
    assert output.certificate["outcome"] == "proved"
    assert backend.forwarded
    assert not any(isinstance(item, FakeMessage) for item in backend.forwarded)


def test_adapter_cleanup_backend_single_owner_parser_cadence_and_wrong_source():
    clock = ManualClock()
    raw_socket = AdapterRawSocket()
    endpoint = runtime.ExclusiveUdpEndpoint(
        socket=raw_socket,
        requested_host="127.0.0.1",
        requested_port=0,
        actual_host="127.0.0.1",
        actual_port=raw_socket.port,
        exclusive_option=1,
    )
    guards = runtime.PoweredOutboundGuards()
    guards.latch_production("test_cleanup_epoch")
    authorized = []
    production = []

    def authorize(category, **_kwargs):
        authorized.append((category, clock.value))
        return clock.value + 500_000_000

    authority = cleanup.CleanupBackendAuthority(
        outbound_guards=guards,
        authorize_outbound=authorize,
        role_valid=lambda: False,
        parent_alive=lambda: True,
        lease_valid=lambda: True,
    )

    def transport_factory(owned_endpoint, **kwargs):
        return PoweredMavlinkTransport(
            owned_endpoint,
            scratch_parser_factory=FakeParser,
            mavlink_factory=lambda writer: production.append(
                AdapterProductionMavlink(writer)
            )
            or production[-1],
            **kwargs,
        )

    zero_factory_calls = []

    def zero_evidence_factory(**values):
        zero_factory_calls.append(values)
        return {
            "state": values["outcome"],
            "required": True,
            "requested": dict(values["command"]),
            "generated": {"test": "generated"},
            "terminal": {"test": "terminal"},
            "outbound_receipt": values["receipt"],
        }

    backend = cleanup.create_aigp_mavlink_cleanup_backend(
        endpoint,
        authority,
        zero_evidence_factory=zero_evidence_factory,
        monotonic_ns=clock,
        transport_factory=transport_factory,
    )
    backend.open(deadline_monotonic_ns=10_000_000_000)
    adapter = backend.adapter
    assert adapter.powered_receive_owner == "external_cleanup"
    assert adapter._rx_thread is None
    assert adapter._announce_thread is None
    assert backend.source_authority is adapter.powered_source_authority
    with pytest.raises(RuntimeError, match="already has an owner"):
        adapter._powered_transport.claim_receive_owner("adapter_worker")

    peer = ("127.0.0.1", 41_001)
    race_payload = encode_race_status(
        sim_boot_time_ms=1_000,
        race_start_boot_time_ms=-1,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    promotion = (
        adapter_mav_frame(0),
        adapter_mav_frame(131, race_payload),
        adapter_mav_frame(105),
    )
    dispatches = []
    for frame in promotion:
        raw_socket.push(frame, peer)
        dispatches.append(backend.receive_and_dispatch_datagram(50_000_000))
    assert all(item.source_accepted for item in dispatches)
    assert dispatches[-1].source_promoted
    assert backend.source_authority.promoted
    assert production[0].parsed == list(promotion)
    observations = backend.drain_received_observations()
    assert [item["ingress"]["message_type"] for item in observations] == [
        "HEARTBEAT",
        "RACE_STATUS",
        "HIGHRES_IMU",
    ]

    # The first frozen peer triggers one synchronous pair. No worker exists.
    assert [item[0] for item in authorized] == ["timesync", "gcs_heartbeat"]
    assert len(raw_socket.sent) == 2
    assert all(item[1] == peer for item in raw_socket.sent)
    first_audit = backend.outbound_audit()
    assert first_audit["timesync"] == first_audit["gcs_heartbeat"] == 1

    # The next pair is not early and is sent exactly at the frozen 100 ms tick.
    clock.value = 1_099_999_999
    raw_socket.push(adapter_mav_frame(200), peer)
    backend.receive_and_dispatch_datagram(50_000_000)
    assert len(raw_socket.sent) == 2
    clock.value = 1_100_000_000
    raw_socket.push(adapter_mav_frame(200), peer)
    backend.receive_and_dispatch_datagram(50_000_000)
    assert len(raw_socket.sent) == 4
    assert [item[0] for item in authorized] == [
        "timesync",
        "gcs_heartbeat",
        "timesync",
        "gcs_heartbeat",
    ]

    # A collision in the old generation is held by the adapter until the
    # atomic reset boundary. The cleanup backend owns normalization because
    # the adapter intentionally retains only the raw collision payload.
    clock.value = 1_125_000_000
    raw_socket.push(adapter_mav_frame(247, bytes((41, 2, 3))), peer)
    boundary_collision_dispatch = backend.receive_and_dispatch_datagram(50_000_000)
    assert boundary_collision_dispatch.source_accepted

    guards.enable_cleanup_live(
        parent_alive=True,
        lease_valid=True,
        source_promoted=True,
    )
    clock.value = 1_150_000_000
    zero = backend.send_exact_zero(
        cleanup.ZERO_COMMAND,
        deadline_monotonic_ns=5_000_000_000,
    )
    assert zero["state"] == "returned"
    assert zero_factory_calls[0]["receipt"]["schema"] == (
        "aigp-vq2-attitude-target-outbound/1"
    )
    disarm = backend.send_disarm(deadline_monotonic_ns=5_000_000_000)
    assert disarm.outcome == "returned"
    assert disarm.receipt["category"] == "disarm"
    reset = backend.send_reset(
        baseline={},
        deadline_monotonic_ns=5_000_000_000,
    )
    assert reset.request.outcome == "returned"
    assert reset.request.receipt["category"] == "sim_reset"
    assert reset.boundary["new_generation"] == reset.boundary["old_generation"] + 1
    old_generation = reset.boundary["old_generation"]
    new_generation = reset.boundary["new_generation"]
    boundary_rows = reset.boundary["collisions"]
    assert len(boundary_rows) == 1
    assert boundary_rows[0] == runner_collision(
        old_generation,
        0,
        reset.boundary["boundary_monotonic_ns"],
        collision_id=41,
        phase="fallback-reset-and-epoch",
        disposition="reset_boundary_discard",
    )
    assert real_contract.validate_reset_boundary(reset.boundary) == reset.boundary

    # A later collision starts a fresh generation-local sequence. The backend
    # returns one accumulated, defensive snapshot containing the boundary row
    # exactly once and the later row exactly once.
    clock.value = 1_175_000_000
    raw_socket.push(adapter_mav_frame(247, bytes((42, 2, 3))), peer)
    post_boundary_dispatch = backend.receive_and_dispatch_datagram(50_000_000)
    assert post_boundary_dispatch.source_accepted
    collision_rows = backend.collision_observations()
    assert collision_rows == [
        boundary_rows[0],
        runner_collision(
            new_generation,
            0,
            1_175_000_000,
            collision_id=42,
            phase="fallback-finalize",
            disposition="cleanup_continue",
        ),
    ]
    assert backend.collision_observations() == collision_rows
    assert collision_rows.count(boundary_rows[0]) == 1

    # Closing cleanup authority suppresses a due announcement before any write.
    guards.close_cleanup()
    sent_before_close = list(raw_socket.sent)
    authorized_before_close = list(authorized)
    clock.value = 1_200_000_000
    raw_socket.push(adapter_mav_frame(200), peer)
    backend.receive_and_dispatch_datagram(50_000_000)
    assert raw_socket.sent == sent_before_close
    assert authorized == authorized_before_close

    # A second source latches collection invalidation and never reaches the
    # established production parser or changes the frozen peer.
    parsed_before_rejection = list(production[0].parsed)
    raw_socket.push(adapter_mav_frame(0), ("127.0.0.1", 41_002))
    rejected = backend.receive_and_dispatch_datagram(50_000_000)
    assert rejected.rejected_source
    assert not rejected.production_dispatched
    assert production[0].parsed == parsed_before_rejection
    assert backend.source_authority.peer == peer
    assert backend.source_authority.source_rejected_latched

    audit = backend.outbound_audit()
    receipts = backend.outbound_receipts()
    assert audit["timesync"] == audit["gcs_heartbeat"] == 2
    assert audit["attitude_target"] == audit["disarm"] == audit["sim_reset"] == 1
    assert audit["receipt_count"] == len(receipts) == 7
    assert audit["receipt_raised"] == audit["receipt_dropped"] == 0

    backend.request_stop()
    endpoint.close()
    proof = backend.join_workers(deadline_monotonic_ns=5_000_000_000)
    assert proof == cleanup.WorkerCloseProof(True, True, True)
    assert raw_socket.closed


def test_backend_announcement_uses_source_lease_and_deadline_dispatcher():
    fixture = build_fixture(backend_options={"announce_after_freeze": True})
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    backend = fixture["backends"][0]
    assert output.certificate["outcome"] == "proved"
    assert len(backend.announcement_deadlines) == 1
    connect_deadline = output.certificate["phase_deadlines"][0][
        "deadline_monotonic_ns"
    ]
    assert backend.announcement_deadlines[0] <= connect_deadline


def test_second_source_invalidates_collection_but_does_not_suppress_cleanup():
    fixture = build_fixture(backend_options={"second_source": True})
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    backend = fixture["backends"][0]
    assert output.certificate["outcome"] == "proved"
    assert output.certificate["collection_invalidating_codes"] == [
        "source_rejected"
    ]
    assert output.process_result["outcome"] == "failed"
    assert output.process_result["reason_codes"] == ["capture_incomplete"]
    assert backend.zero_calls == backend.disarm_calls == backend.reset_calls == 1
    assert mav_frame(4) not in backend.forwarded


def test_nonloopback_source_is_latched_by_source_gate_and_never_forwarded():
    fixture = build_fixture(
        backend_options={"second_source": True, "nonloopback_source": True}
    )
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    backend = fixture["backends"][0]
    assert output.certificate["collection_invalidating_codes"] == [
        "source_rejected"
    ]
    assert output.certificate["outcome"] == "proved"
    assert mav_frame(4) not in backend.forwarded
    assert backend.zero_calls == backend.disarm_calls == backend.reset_calls == 1


def test_parent_death_after_disarm_takes_over_once_without_repeating_phase():
    state = {"die_after_disarm": True}
    fixture = build_fixture(state=state)
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    backend = fixture["backends"][0]
    phases = [item["phase"] for item in output.certificate["phase_deadlines"]]
    assert phases == [
        "connect",
        "disarm",
        "parent_death_lease_takeover",
        "reset_and_epoch",
        "finalize",
    ]
    assert output.certificate["outcome"] == "failed"
    assert "parent_dead" in output.certificate["failure_codes"]
    assert output.process_result["outcome"] == "failed"
    assert output.certificate["parent_state"]["mode"] == "signaled_takeover"
    assert output.process_result["reason_codes"] == [
        "cleanup_unconfirmed",
        "wrapper_death",
    ]
    checked_certificate, checked_result = real_contract_round_trip(
        output,
        nominal=True,
    )
    assert checked_certificate["outcome"] == "failed"
    assert checked_result["outcome"] == "failed"
    assert fixture["lease"].takeovers == fixture["lease"].releases == 1
    assert backend.zero_calls == backend.disarm_calls == backend.reset_calls == 1


def test_takeover_heartbeats_refresh_latest_proof_on_worker_and_certificate_steps():
    state = {
        "die_after_disarm": True,
        "heartbeat_during_join": [950_000_000, 20_000_000],
        "heartbeat_during_publish": True,
        "enforce_heartbeat_cadence": True,
    }
    fixture = build_fixture(state=state)
    owner_thread = threading.get_ident()
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    successful = fixture["lease"].heartbeats
    assert len(successful) == 2
    assert all(item[2] == owner_thread for item in successful)
    gaps = [
        later[0] - earlier[0]
        for earlier, later in zip(successful, successful[1:])
    ]
    assert all(1_000_000_000 <= gap <= 1_500_000_000 for gap in gaps)
    assert successful[0][1] == "fallback-finalize"
    assert successful[1][1] == "fallback-certificate"
    assert fixture["lease"].latest_proof.generation == 10
    assert fixture["lease"].releases == 1
    assert output.certificate["outcome"] == "failed"
    assert output.process_result["outcome"] == "failed"


def test_poll_loop_retries_early_heartbeat_and_never_exceeds_fifty_ms():
    fixture = build_fixture(
        state={"parent_dead": True, "enforce_heartbeat_cadence": True},
        backend_options={"empty": True},
        clock=StepClock(step=10_000_000),
    )
    owner_thread = threading.get_ident()
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    backend = fixture["backends"][0]
    successful = fixture["lease"].heartbeats
    assert successful
    assert all(item[2] == owner_thread for item in successful)
    assert all(
        1_000_000_000 <= later[0] - earlier[0] <= 1_500_000_000
        for earlier, later in zip(successful, successful[1:])
    )
    assert backend.waits
    assert max(backend.waits) <= 50_000_000
    assert output.process_result["outcome"] == "failed"


def test_stale_takeover_heartbeat_is_rejected_and_cannot_release_authority():
    state = {
        "die_after_disarm": True,
        "heartbeat_during_join": True,
        "stale_heartbeat": True,
    }
    fixture = build_fixture(state=state)
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    assert fixture["lease"].heartbeats
    assert fixture["lease"].releases == 0
    assert output.certificate["lease"]["authority_valid"] is False
    assert "lease_invalid" in output.certificate["failure_codes"]
    assert "lease_release_unconfirmed" in output.process_result["reason_codes"]


def test_failed_takeover_is_attempted_once_and_stops_all_later_sends():
    state = {"die_after_disarm": True, "takeover_fail": True}
    fixture = build_fixture(state=state)
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    backend = fixture["backends"][0]
    assert fixture["lease"].takeovers == 1
    assert fixture["lease"].releases == 0
    assert backend.zero_calls == backend.disarm_calls == 1
    assert backend.reset_calls == 0
    assert output.certificate["outcome"] == "failed"
    assert output.certificate["trigger"] == "parent_death"
    assert output.certificate["lease"]["authority_valid"] is False
    assert "lease_invalid" in output.certificate["failure_codes"]
    assert "wrapper_death" in output.process_result["reason_codes"]


def test_outbound_call_overrun_closes_guard_and_forbids_later_sends():
    fixture = build_fixture(state={"stall_zero": True})
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    backend = fixture["backends"][0]
    assert backend.zero_calls == 1
    assert backend.disarm_calls == backend.reset_calls == 0
    assert output.certificate["outcome"] == "failed"
    assert "deadline_expired" in output.certificate["failure_codes"]
    assert output.certificate["transport"]["cleanup_guard_closed"] is True


def test_nonmonotonic_reset_epoch_is_recorded_and_never_claimed_confirmed():
    fixture = build_fixture(
        backend_options={"bad_reset_order": True},
        clock=StepClock(step=100_000_000),
    )
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    reset = output.certificate["reset"]
    assert reset["state"] == "unconfirmed"
    assert [row["race_status"]["sim_boot_time_ms"] for row in reset["advancing_race"]] == [
        11,
        9,
        12,
    ]
    assert reset["rollback_and_advance_confirmed"] is False
    assert output.certificate["outcome"] == "failed"
    checked_certificate, checked_result = real_contract_round_trip(
        output,
        nominal=True,
    )
    assert checked_certificate["reset"]["state"] == "unconfirmed"
    assert checked_result["outcome"] == "failed"


def test_late_parent_death_extends_result_prefix_without_repeating_cleanup():
    state = {"die_on_publish": True}
    fixture = build_fixture(state=state)
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    backend = fixture["backends"][0]
    certificate_phases = output.certificate["phase_deadlines"]
    result_phases = output.process_result["phase_deadlines"]
    assert [item["phase"] for item in certificate_phases] == [
        "connect",
        "disarm",
        "reset_and_epoch",
        "finalize",
    ]
    assert result_phases[: len(certificate_phases)] == certificate_phases
    assert result_phases[-1]["phase"] == "parent_death_lease_takeover"
    assert fixture["lease"].takeovers == fixture["lease"].releases == 1
    assert backend.zero_calls == backend.disarm_calls == backend.reset_calls == 1
    assert len(fixture["publisher"].calls) == 1


def test_connect_deadline_publishes_failed_certificate_and_disconnects_bounded():
    fixture = build_fixture(
        backend_options={"empty": True},
        clock=StepClock(step=100_000_000),
    )
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    backend = fixture["backends"][0]
    assert output.certificate["outcome"] == "failed"
    assert "connect_failed" in output.certificate["failure_codes"]
    assert output.process_result["reason_codes"] == [
        "cleanup_unconfirmed",
        "deadline_expired",
    ]
    assert backend.zero_calls == backend.disarm_calls == backend.reset_calls == 0
    assert backend.endpoint.closed
    assert backend.stop_calls == backend.join_calls == 1
    assert backend.waits and max(backend.waits) <= runtime.MAX_POLL_INTERVAL_NS
    assert len(fixture["publisher"].calls) == 1


def test_generated_failed_certificate_and_result_round_trip_real_contract():
    fixture = build_fixture(
        backend_options={"empty": True},
        clock=StepClock(step=100_000_000),
    )
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    certificate, result = real_contract_round_trip(output, nominal=False)
    assert certificate["outcome"] == "failed"
    assert "connect_failed" in certificate["failure_codes"]
    assert result["outcome"] == "failed"


class RaisingCloseSocket:
    def close(self):
        raise OSError("injected close failure")


def test_endpoint_close_failure_is_unproved_transport_not_suppressed():
    fixture = build_fixture()

    def endpoint_factory(host, port):
        return runtime.ExclusiveUdpEndpoint(
            socket=RaisingCloseSocket(),
            requested_host=host,
            requested_port=port,
            actual_host=host,
            actual_port=40_100,
            exclusive_option=1,
        )

    fixture["services"].endpoint_factory = endpoint_factory
    output = cleanup.run_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    assert output.certificate["outcome"] == "failed"
    assert "transport_unclosed" in output.certificate["failure_codes"]
    assert output.certificate["transport"]["mavlink_socket_closed"] is False
    assert len(fixture["publisher"].calls) == 1


def test_main_emits_no_stdout_before_capability_and_sanitizes_stderr():
    fixture = build_fixture()
    different_secret = b"z" * 32
    fixture["services"].capability_operations.frame = runtime.encode_capability_frame(
        different_secret
    )
    stdout = io.BytesIO()
    stderr = io.BytesIO()
    code = cleanup.main(
        argument_tail(fixture["arguments"]),
        services=fixture["services"],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 2
    assert stdout.getvalue() == b""
    diagnostic = stderr.getvalue()
    assert fixture["secret"].hex().encode() not in diagnostic
    assert different_secret.hex().encode() not in diagnostic
    assert len(diagnostic) < cleanup.STDERR_LIMIT_BYTES
    assert fixture["backends"] == []
    assert fixture["publisher"].calls == []


def test_main_emits_exactly_one_canonical_process_result():
    fixture = build_fixture()
    stdout = io.BytesIO()
    stderr = io.BytesIO()
    code = cleanup.main(
        argument_tail(fixture["arguments"]),
        services=fixture["services"],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    assert stderr.getvalue() == b""
    payload = stdout.getvalue()
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    parsed = json.loads(payload)
    assert parsed["schema"] == "aigp-vq2-powered-process-result/1"
    assert payload == fixture["contract"].canonical_json_file_bytes(parsed)


def test_main_closes_owned_bootstrap_handles_against_one_frozen_deadline():
    fixture = build_fixture()

    class CloseProof:
        proved = True

    class OwnedBoundary:
        def __init__(self):
            self.calls = []

        def close_owned_handles(self, *, deadline_monotonic_ns, monotonic_ns):
            self.calls.append((deadline_monotonic_ns, monotonic_ns()))
            return CloseProof()

    owned = OwnedBoundary()
    fixture["services"].owned_process_boundary = owned
    stdout = io.BytesIO()
    stderr = io.BytesIO()
    code = cleanup.main(
        argument_tail(fixture["arguments"]),
        services=fixture["services"],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    assert len(owned.calls) == 1
    deadline, close_occurrence = owned.calls[0]
    assert 0 < deadline - close_occurrence <= cleanup.OWNED_HANDLE_CLOSE_DURATION_NS
    assert stdout.getvalue()
    assert stderr.getvalue() == b""


def test_unproved_owned_handle_close_suppresses_result_and_fails_stage():
    fixture = build_fixture()

    class CloseProof:
        proved = False

    class OwnedBoundary:
        def close_owned_handles(self, *, deadline_monotonic_ns, monotonic_ns):
            return CloseProof()

    fixture["services"].owned_process_boundary = OwnedBoundary()
    stdout = io.BytesIO()
    stderr = io.BytesIO()
    code = cleanup.main(
        argument_tail(fixture["arguments"]),
        services=fixture["services"],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 1
    assert stdout.getvalue() == b""
    assert stderr.getvalue() == b"powered cleanup failed after admission\n"


def test_default_builder_defers_lease_and_backend_until_post_capability():
    fixture = build_fixture()
    events = []

    class Qpc:
        def now_ns(self):
            return 1_000_000

        def query_performance_frequency_hz(self):
            return 10_000_000

    class Process:
        pass

    class Publisher:
        pass

    process = Process()
    capability = object()
    lease = object()
    backend = object()

    def make_qpc():
        events.append("qpc")
        return Qpc()

    def make_capability():
        events.append("capability_operations")
        return capability

    def make_process(capability_handle, parent_handle):
        events.append(("process", capability_handle, parent_handle))
        return process

    def make_publisher(**kwargs):
        events.append("publisher")
        assert kwargs["contract"] is real_contract
        return Publisher()

    def make_lease(admission, owned_process, qpc):
        events.append("lease")
        assert owned_process is process
        assert isinstance(qpc, Qpc)
        return lease

    backend_calls = []

    def make_backend(endpoint, authority, **kwargs):
        events.append("backend")
        backend_calls.append((endpoint, authority, kwargs))
        return backend

    services = cleanup.build_default_cleanup_services(
        fixture["arguments"],
        qpc_provider_factory=make_qpc,
        process_boundary_factory=make_process,
        capability_operations_factory=make_capability,
        delegated_lease_factory=make_lease,
        backend_builder=make_backend,
        publisher_factory=make_publisher,
    )
    assert events == [
        "qpc",
        "capability_operations",
        ("process", 11, 22),
        "publisher",
    ]
    assert services.lease_boundary is None
    assert services.owned_process_boundary is process

    admission = cleanup.admit_cleanup_fallback(
        fixture["arguments"], fixture["services"]
    )
    admission.attempt["context"].update(
        {
            "task_id": real_contract.TASK_ID,
            "session_id": real_contract.SESSION_ID,
            "attempt_id": real_contract.ATTEMPT_ID,
            "candidate_commit": "0" * 40,
            "host": {"host_clock_id": real_contract.HOST_CLOCK_ID},
        }
    )
    assert services.lease_boundary_factory(admission) is lease
    assert events[-1] == "lease"

    class Endpoint:
        closed = False

        def close(self):
            self.closed = True

    endpoint = Endpoint()
    authority = object()
    assert services.backend_factory(endpoint, authority) is backend
    assert events[-1] == "backend"
    assert callable(backend_calls[0][2]["zero_evidence_factory"])
    assert backend_calls[0][2]["monotonic_ns"]() == 1_000_000


def test_production_zero_factory_builds_real_contract_evidence():
    attempt = {
        "context_sha256": "c" * 64,
        "context": {
            "attempt_id": real_contract.ATTEMPT_ID,
            "session_id": real_contract.SESSION_ID,
            "candidate_commit": "0" * 40,
            "host": {"host_clock_id": real_contract.HOST_CLOCK_ID},
        },
    }
    outbound = real_attitude_receipt(0, 1_000_000)
    factory = cleanup._cleanup_zero_evidence_factory(attempt, real_contract)
    evidence = factory(
        command=dict(cleanup.ZERO_COMMAND),
        request_monotonic_ns=1_000_000,
        completed_monotonic_ns=1_000_001,
        outcome="returned",
        receipt=outbound,
        audit_count_before=0,
        audit_count_after=1,
    )
    checked_generated = real_contract.validate_command_generated(
        evidence["generated"]
    )
    checked_terminal = real_contract.validate_command_sent(
        evidence["terminal"], generated=checked_generated
    )
    assert checked_terminal["transport"]["receipt"] == outbound
    assert evidence["state"] == "returned"


def test_create_new_publisher_flushes_readback_and_never_overwrites(tmp_path):
    contract = FakeContract()
    target = (tmp_path / "certificate.json").resolve()
    publisher = cleanup.CanonicalCreateNewPublisher(
        contract=contract,
        monotonic_ns=lambda: 1,
    )
    value = {"schema": "test", "value": 1}
    digest = publisher.publish_create_new(
        str(target),
        value,
        deadline_monotonic_ns=2,
    )
    assert digest == contract.canonical_file_sha256(value)
    assert target.read_bytes() == contract.canonical_json_file_bytes(value)
    with pytest.raises(cleanup.CleanupEvidenceError, match="create-new"):
        publisher.publish_create_new(
            str(target),
            value,
            deadline_monotonic_ns=2,
        )
