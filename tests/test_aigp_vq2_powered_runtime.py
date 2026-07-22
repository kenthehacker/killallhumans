from __future__ import annotations

import hashlib
import os
import socket
import sys
import time
from pathlib import Path

import pytest

import scripts.aigp_vq2_powered_runtime as runtime


def process_identity(pid, argv, *, created=20):
    return {
        "pid": pid,
        "creation_filetime_100ns": created,
        "windows_session_id": 1,
        "image_path": r"C:\Python\python.exe",
        "image_sha256": "a" * 64,
        "argv_sha256": runtime.argv_sha256(argv),
    }


class FakeWindowsOperations:
    def __init__(self):
        self.current_pid = 10
        self.next_open_handle = 100
        self.identities = {}
        self.argv_by_handle = {}
        self.inheritable = set()
        self.closed = []
        self.close_failures = set()
        self.signaled_sequences = {}
        self.exit_codes = {}
        self.pipe_available = 0
        self.writes = []
        self.calls = []
        self.job_flags = 0
        self.in_job = True
        self.job_pid_sequences = []
        self.process_creation = runtime.ProcessCreation(400, 401, 20, 21)
        self.terminated_processes = []
        self.terminated_jobs = []

    def current_process_id(self):
        return self.current_pid

    def open_process(self, pid, *, inheritable, terminate_access):
        handle = self.next_open_handle
        self.next_open_handle += 1
        argv = ("wrapper",)
        self.identities[handle] = process_identity(pid, argv)
        self.argv_by_handle[handle] = argv
        self.signaled_sequences.setdefault(handle, [False])
        self.exit_codes.setdefault(handle, runtime.STILL_ACTIVE)
        if inheritable:
            self.inheritable.add(handle)
        self.calls.append(("open_process", pid, inheritable, terminate_access))
        return handle

    def query_process_argv(self, process_handle):
        self.calls.append(("argv", process_handle))
        return self.argv_by_handle[process_handle]

    def query_process_identity(self, process_handle, expected_argv):
        self.calls.append(("identity", process_handle))
        assert tuple(expected_argv) == self.argv_by_handle[process_handle]
        return dict(self.identities[process_handle])

    def process_signaled(self, process_handle):
        values = self.signaled_sequences.setdefault(process_handle, [False])
        value = values.pop(0) if len(values) > 1 else values[0]
        self.calls.append(("signaled", process_handle, value))
        return value

    def process_exit_code(self, process_handle):
        return self.exit_codes[process_handle]

    def handle_is_inheritable(self, handle):
        self.calls.append(("inheritable", handle))
        return handle in self.inheritable

    def close_handle(self, handle):
        self.calls.append(("close", handle))
        if handle in self.close_failures:
            raise OSError("injected close failure")
        self.closed.append(handle)

    def create_capability_pipe(self):
        self.inheritable.add(11)
        self.calls.append(("create_pipe",))
        return 11, 12

    def pipe_available_bytes(self, read_handle):
        assert read_handle == 11
        return self.pipe_available

    def write_file(self, handle, payload):
        self.calls.append(("write", handle))
        self.writes.append((handle, bytes(payload)))
        return len(payload)

    def create_non_breakaway_job(self):
        self.calls.append(("create_job",))
        return 300

    def spawn_process_explicit(
        self,
        argv,
        *,
        cwd,
        environment,
        inherited_handles,
        stdin_handle,
        stdout_handle,
        stderr_handle,
    ):
        self.calls.append(("spawn", tuple(inherited_handles)))
        handle = self.process_creation.process_handle
        values = tuple(argv)
        self.argv_by_handle[handle] = values
        self.identities[handle] = process_identity(
            self.process_creation.pid, values, created=30
        )
        self.signaled_sequences.setdefault(handle, [False])
        self.exit_codes.setdefault(handle, runtime.STILL_ACTIVE)
        return self.process_creation

    def assign_process_to_job(self, job_handle, process_handle):
        self.calls.append(("assign", job_handle, process_handle))

    def query_job_limit_flags(self, job_handle):
        self.calls.append(("job_flags", job_handle))
        return self.job_flags

    def process_in_job(self, process_handle, job_handle):
        self.calls.append(("in_job", process_handle, job_handle))
        return self.in_job

    def job_active_pids(self, job_handle):
        if self.job_pid_sequences:
            value = self.job_pid_sequences.pop(0)
        else:
            value = ()
        self.calls.append(("job_pids", job_handle, value))
        return value

    def terminate_process(self, process_handle, exit_code):
        self.calls.append(("terminate_process", process_handle, exit_code))
        self.terminated_processes.append(process_handle)
        self.signaled_sequences[process_handle] = [True]
        self.exit_codes[process_handle] = exit_code

    def terminate_job(self, job_handle, exit_code):
        self.calls.append(("terminate_job", job_handle, exit_code))
        self.terminated_jobs.append(job_handle)
        self.signaled_sequences[self.process_creation.process_handle] = [True, True]
        self.exit_codes[self.process_creation.process_handle] = exit_code
        self.job_pid_sequences = [(), ()]


class FakeUdpOwnerOperations:
    def __init__(self, ipv4, ipv6):
        self.rows = {
            socket.AF_INET: list(ipv4),
            socket.AF_INET6: list(ipv6),
        }

    def udp_owner_rows(self, family):
        values = self.rows[family]
        if values and type(values[0]) is list:
            return tuple(values.pop(0))
        return tuple(values)


def sequence_clock(*values: int):
    remaining = iter(values)
    return lambda: next(remaining)


def os_assigned_udp_ports(count=1):
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(count)]
    try:
        for candidate in sockets:
            candidate.bind(("127.0.0.1", 0))
        return tuple(candidate.getsockname()[1] for candidate in sockets)
    finally:
        for candidate in sockets:
            candidate.close()


class FakeQpcOperations:
    def __init__(self, *, frequencies, counters):
        self.frequencies = list(frequencies)
        self.counters = list(counters)
        self.calls = []

    def query_performance_frequency_hz(self):
        self.calls.append("frequency")
        return self.frequencies.pop(0)

    def query_performance_counter(self):
        self.calls.append("counter")
        return self.counters.pop(0)


def test_windows_qpc_provider_uses_exact_native_frequency_and_counter_values():
    operations = FakeQpcOperations(
        frequencies=[3_000_000, 3_000_000],
        counters=[12_345_678],
    )
    provider = runtime.WindowsQpcProvider(operations)
    assert provider.clock_id == runtime.HOST_CLOCK_ID
    assert provider.query_performance_frequency_hz() == 3_000_000
    assert provider.now_ns() == (12_345_678 * 1_000_000_000) // 3_000_000
    assert operations.calls == ["frequency", "frequency", "counter"]


def test_windows_qpc_provider_latches_and_rejects_native_frequency_drift():
    operations = FakeQpcOperations(
        frequencies=[10_000_000, 3_000_000],
        counters=[12_345_678],
    )
    provider = runtime.WindowsQpcProvider(operations)
    assert provider.query_performance_frequency_hz() == 10_000_000
    with pytest.raises(runtime.PoweredRuntimeError, match="changed"):
        provider.now_ns()
    assert operations.calls == ["frequency", "frequency"]


@pytest.mark.parametrize(
    ("frequency", "counter", "message"),
    [
        (0, 1, "Frequency"),
        (True, 1, "Frequency"),
        (10_000_000, -1, "Counter"),
        (10_000_000, False, "Counter"),
    ],
)
def test_windows_qpc_provider_rejects_inexact_or_invalid_native_results(
    frequency, counter, message
):
    provider = runtime.WindowsQpcProvider(
        FakeQpcOperations(frequencies=[frequency], counters=[counter])
    )
    with pytest.raises(runtime.PoweredRuntimeError, match=message):
        provider.now_ns()


@pytest.mark.skipif(os.name != "nt", reason="native QPC is Windows-only")
def test_real_windows_qpc_provider_reports_frequency_and_advancing_occurrences():
    provider = runtime.WindowsQpcProvider()
    frequency = provider.query_performance_frequency_hz()
    before = provider.now_ns()
    after = provider.now_ns()
    assert type(frequency) is int and frequency > 0
    assert type(before) is int and before >= 0
    assert type(after) is int and after >= before


def test_qpc_deadline_is_one_read_exact_minimum_and_never_refreshes():
    clock = sequence_clock(100)
    deadline = runtime.freeze_phase_deadline(
        "connect", 50, 130, monotonic_ns=clock
    )
    assert deadline.to_primitive() == {
        "phase": "connect",
        "started_monotonic_ns": 100,
        "duration_ns": 50,
        "parent_deadline_monotonic_ns": 130,
        "deadline_monotonic_ns": 130,
    }
    assert runtime.remaining_ns(130, 100) == 30
    assert runtime.remaining_ns(130, 130) == 0
    assert runtime.remaining_ns(130, 140) == 0
    assert runtime.deadline_reached(130, 129) is False
    assert runtime.deadline_reached(130, 130) is True


@pytest.mark.parametrize(
    ("clock_value", "duration", "parent", "message"),
    [
        (10, 0, 20, "duration"),
        (20, 1, 20, "already reached"),
        (True, 1, 20, "exact integer"),
    ],
)
def test_qpc_deadline_rejects_invalid_or_expired_inputs(
    clock_value, duration, parent, message
):
    with pytest.raises((ValueError, runtime.PoweredDeadlineExpired), match=message):
        runtime.freeze_phase_deadline(
            "connect",
            duration,
            parent,
            monotonic_ns=lambda: clock_value,
        )


def test_bounded_poll_caps_waits_checks_parent_and_returns_value():
    waits: list[int] = []
    probes = iter((None, None, "ready"))
    result = runtime.bounded_poll(
        lambda: next(probes),
        deadline_monotonic_ns=170_000_000,
        monotonic_ns=sequence_clock(0, 50_000_000, 100_000_000),
        wait_ns=waits.append,
        parent_alive=lambda: True,
    )
    assert result == "ready"
    assert waits == [50_000_000, 50_000_000]

    with pytest.raises(runtime.PoweredRuntimeError, match="parent"):
        runtime.bounded_poll(
            lambda: None,
            deadline_monotonic_ns=1,
            monotonic_ns=lambda: 0,
            wait_ns=lambda _value: None,
            parent_alive=lambda: False,
        )


def test_bounded_poll_uses_remaining_time_and_now_equal_deadline_fails():
    waits: list[int] = []
    with pytest.raises(runtime.PoweredDeadlineExpired):
        runtime.bounded_poll(
            lambda: None,
            deadline_monotonic_ns=60,
            monotonic_ns=sequence_clock(20, 60),
            wait_ns=waits.append,
        )
    assert waits == [40]

    with pytest.raises(ValueError, match="50 milliseconds"):
        runtime.bounded_poll(
            lambda: None,
            deadline_monotonic_ns=1,
            max_poll_interval_ns=runtime.MAX_POLL_INTERVAL_NS + 1,
        )


def test_capability_hash_and_frame_are_exact_and_context_bound():
    secret = bytes(range(32))
    context = "ab" * 32
    domain = "aigp-vq2-powered-child/1"
    expected = hashlib.sha256(
        domain.encode("utf-8")
        + b"\x00"
        + bytes.fromhex(context)
        + b"\x00"
        + secret
    ).hexdigest()
    assert runtime.derive_capability_sha256(domain, context, secret) == expected
    frame = runtime.encode_capability_frame(secret)
    assert frame[:4] == b"\x20\x00\x00\x00"
    assert len(frame) == 36
    assert runtime.decode_capability_frame(frame) == secret

    with pytest.raises(ValueError, match="allowlisted"):
        runtime.derive_capability_sha256("wrong", context, secret)
    with pytest.raises(ValueError, match="32 bytes"):
        runtime.encode_capability_frame(b"short")
    with pytest.raises(runtime.CapabilityProtocolError, match="prefix"):
        runtime.decode_capability_frame(b"\x1f\x00\x00\x00" + secret)


class FakePipeOperations:
    def __init__(
        self,
        *,
        peeks,
        frame: bytes,
        parent_states=None,
        short_read: bool = False,
        close_error: bool = False,
    ):
        self.peeks = list(peeks)
        self.frame = frame
        self.parent_states = list(parent_states or [])
        self.short_read = short_read
        self.close_error = close_error
        self.waits: list[int] = []
        self.reads: list[tuple[int, int]] = []
        self.closed: list[int] = []

    def peek_named_pipe(self, handle):
        assert handle == 11
        return self.peeks.pop(0)

    def read_file(self, handle, size):
        self.reads.append((handle, size))
        return self.frame[:-1] if self.short_read else self.frame

    def process_signaled(self, handle):
        assert handle == 22
        return self.parent_states.pop(0) if self.parent_states else False

    def wait_ns(self, duration_ns):
        self.waits.append(duration_ns)

    def close_handle(self, handle):
        self.closed.append(handle)
        if self.close_error:
            raise runtime.CapabilityProtocolError("injected close failure")


def _read_capability(operations, *, expected=None, clock=None):
    secret = bytes(range(32))
    context = "cd" * 32
    expected_hash = expected or runtime.derive_capability_sha256(
        "aigp-vq2-powered-child/1", context, secret
    )
    result = runtime.read_bound_capability(
        11,
        22,
        domain="aigp-vq2-powered-child/1",
        context_sha256=context,
        expected_capability_sha256=expected_hash,
        deadline_monotonic_ns=1_000_000_000,
        operations=operations,
        monotonic_ns=clock or sequence_clock(0, 10, 20, 30, 40, 50),
    )
    return result


def test_capability_pipe_polls_reads_once_then_proves_broken_pipe_eof():
    secret = bytes(range(32))
    operations = FakePipeOperations(
        peeks=[
            runtime.PipePeek(0, False),
            runtime.PipePeek(36, False),
            runtime.PipePeek(0, False),
            runtime.PipePeek(0, True),
        ],
        frame=runtime.encode_capability_frame(secret),
    )
    assert _read_capability(operations) == secret
    assert operations.reads == [(11, 36)]
    assert operations.closed == [11]
    assert operations.waits == [50_000_000, 50_000_000]


@pytest.mark.parametrize(
    ("peeks", "short_read", "parent_states", "expected", "message"),
    [
        ([runtime.PipePeek(0, True)], False, None, None, "closed before"),
        ([runtime.PipePeek(37, False)], False, None, None, "extra bytes"),
        ([runtime.PipePeek(36, False)], True, None, None, "short"),
        (
            [runtime.PipePeek(36, False), runtime.PipePeek(1, False)],
            False,
            None,
            None,
            "extra bytes",
        ),
        ([runtime.PipePeek(0, False)], False, [True], None, "parent process"),
        (
            [runtime.PipePeek(36, False), runtime.PipePeek(0, True)],
            False,
            None,
            "e" * 64,
            "digest",
        ),
    ],
)
def test_capability_pipe_failures_are_closed_without_a_second_read(
    peeks, short_read, parent_states, expected, message
):
    operations = FakePipeOperations(
        peeks=peeks,
        frame=runtime.encode_capability_frame(bytes(range(32))),
        parent_states=parent_states,
        short_read=short_read,
    )
    with pytest.raises(runtime.CapabilityProtocolError, match=message) as caught:
        _read_capability(operations, expected=expected)
    assert operations.closed == [11]
    assert len(operations.reads) <= 1
    assert bytes(range(32)).hex() not in str(caught.value)


def test_capability_pipe_deadline_and_close_failure_are_fail_closed():
    operations = FakePipeOperations(
        peeks=[runtime.PipePeek(0, False)],
        frame=runtime.encode_capability_frame(bytes(range(32))),
    )
    with pytest.raises(runtime.CapabilityProtocolError, match="deadline"):
        _read_capability(
            operations,
            clock=sequence_clock(1_000_000_000),
        )
    assert operations.closed == [11]

    close_failure = FakePipeOperations(
        peeks=[runtime.PipePeek(36, False), runtime.PipePeek(0, True)],
        frame=runtime.encode_capability_frame(bytes(range(32))),
        close_error=True,
    )
    with pytest.raises(runtime.CapabilityProtocolError, match="close failure"):
        _read_capability(close_failure)


def test_capability_pipe_closes_on_invalid_preflight_and_rejects_long_poll():
    operations = FakePipeOperations(peeks=[], frame=b"")
    with pytest.raises(ValueError, match="allowlisted"):
        runtime.read_bound_capability(
            11,
            22,
            domain="wrong",
            context_sha256="a" * 64,
            expected_capability_sha256="b" * 64,
            deadline_monotonic_ns=1,
            operations=operations,
        )
    assert operations.closed == [11]
    assert operations.reads == []

    operations = FakePipeOperations(peeks=[], frame=b"")
    with pytest.raises(ValueError, match="50 milliseconds"):
        runtime.read_bound_capability(
            11,
            22,
            domain="aigp-vq2-powered-child/1",
            context_sha256="a" * 64,
            expected_capability_sha256="b" * 64,
            deadline_monotonic_ns=1,
            operations=operations,
            max_poll_interval_ns=runtime.MAX_POLL_INTERVAL_NS + 1,
        )
    assert operations.closed == [11]


def test_win32_pipe_boundary_is_lazy_off_windows(monkeypatch):
    monkeypatch.setattr(runtime.os, "name", "posix")
    with pytest.raises(runtime.CapabilityProtocolError, match="Windows"):
        runtime.Win32CapabilityPipeOperations()


@pytest.mark.parametrize(
    "value",
    ["", "0:1", "1:0", "01:2", "1:02", "1", "1:2:3", "+1:2", " 1:2"],
)
def test_process_identity_token_is_canonical(value):
    with pytest.raises(runtime.ProcessIdentityError):
        runtime.parse_process_identity_token(value)


def test_process_identity_and_handle_token_round_trip():
    token = runtime.parse_process_identity_token("123:456789")
    assert token == runtime.ProcessIdentityToken(123, 456789)
    assert str(token) == "123:456789"
    assert runtime.parse_decimal_handle("987") == 987
    for invalid in ("0", "01", "-1", "1.0", "", 1):
        with pytest.raises(runtime.ProcessIdentityError):
            runtime.parse_decimal_handle(invalid)  # type: ignore[arg-type]


def test_process_identity_primitive_and_argv_hash_are_exact():
    value = {
        "pid": 10,
        "creation_filetime_100ns": 20,
        "windows_session_id": 1,
        "image_path": r"C:\Python\python.exe",
        "image_sha256": "a" * 64,
        "argv_sha256": "b" * 64,
    }
    assert runtime.validate_process_identity(value) == value
    assert runtime.argv_sha256(["python", "-m", "module"]) == hashlib.sha256(
        b'["python","-m","module"]'
    ).hexdigest()
    with pytest.raises(runtime.ProcessIdentityError, match="missing or unknown"):
        runtime.validate_process_identity({**value, "extra": 1})
    with pytest.raises(runtime.ProcessIdentityError, match="argv"):
        runtime.argv_sha256(["python", 1])  # type: ignore[list-item]


def test_win32_process_identity_uses_bounded_large_image_hash(monkeypatch):
    image_path = r"C:\AIGP_3385\DCGame-Win64-Shipping.exe"
    argv = (image_path, "FlightSim")
    stable_calls = []

    class FakeProcessOperations:
        def _process_pid(self, handle):
            assert handle == 123
            return 456

        def _creation_filetime(self, handle):
            assert handle == 123
            return 789

        def _session_id(self, pid):
            assert pid == 456
            return 1

        def _image_path(self, handle):
            assert handle == 123
            return image_path

        def query_process_argv(self, handle):
            assert handle == 123
            return argv

    def stable_image(path, *, max_bytes):
        stable_calls.append((path, max_bytes))
        return runtime.StableFileIdentity(path, 91_968_000, "a" * 64)

    monkeypatch.setattr(runtime, "stable_file_identity", stable_image)
    identity = runtime.Win32ProcessOperations.query_process_identity(
        FakeProcessOperations(), 123, argv
    )

    assert runtime.MAX_PROCESS_IMAGE_BYTES == 128 * 1024 * 1024
    assert 91_968_000 <= runtime.MAX_PROCESS_IMAGE_BYTES
    assert stable_calls == [(image_path, runtime.MAX_PROCESS_IMAGE_BYTES)]
    assert identity["image_path"] == image_path
    assert identity["image_sha256"] == "a" * 64
    assert identity["argv_sha256"] == runtime.argv_sha256(argv)


def test_new_win32_boundaries_are_lazy_off_windows(monkeypatch):
    monkeypatch.setattr(runtime.os, "name", "posix")
    with pytest.raises(runtime.Win32RuntimeUnavailable, match="Windows"):
        runtime.Win32QpcOperations()
    with pytest.raises(runtime.Win32RuntimeUnavailable, match="Windows"):
        runtime.WindowsQpcProvider()
    with pytest.raises(runtime.Win32RuntimeUnavailable, match="Windows"):
        runtime.Win32ProcessOperations()
    with pytest.raises(runtime.Win32RuntimeUnavailable, match="Windows"):
        runtime.Win32UdpOwnerTableOperations()


def test_retained_process_binds_identity_argv_liveness_and_close():
    operations = FakeWindowsOperations()
    retained = runtime.retain_process(
        10,
        ["wrapper"],
        inheritable=True,
        operations=operations,
    )
    assert retained.identity == process_identity(10, ("wrapper",))
    assert retained.token == runtime.ProcessIdentityToken(10, 20)
    assert retained.alive() is True
    assert retained.reprove() == retained.identity
    retained.close()
    assert retained.closed and retained.handle_value in operations.closed

    mismatch = FakeWindowsOperations()
    original_query = mismatch.query_process_identity

    def wrong_identity(handle, argv):
        value = original_query(handle, argv)
        value["creation_filetime_100ns"] += 1
        return value

    mismatch.query_process_identity = wrong_identity
    retained = runtime.retain_process(10, ["wrapper"], operations=mismatch)
    mismatch.query_process_identity = original_query
    with pytest.raises(runtime.ProcessIdentityError, match="changed"):
        retained.reprove()
    retained.close()


@pytest.mark.skipif(os.name != "nt", reason="retained process handles are Windows-only")
def test_real_current_process_identity_is_handle_bound_and_live():
    retained = runtime.retain_current_process()
    try:
        identity = retained.identity
        assert identity["pid"] == os.getpid()
        assert identity["creation_filetime_100ns"] > 0
        assert os.path.isabs(identity["image_path"])
        assert len(identity["image_sha256"]) == 64
        assert retained.alive() is True
        assert retained.signaled() is False
        assert retained.reprove() == identity
    finally:
        retained.close()
    assert retained.closed


def _retained_child_boundary(operations=None):
    operations = operations or FakeWindowsOperations()
    capability_handle = 51
    parent_handle = 52
    parent_argv = ("wrapper-parent", "--retained")
    operations.inheritable.update({capability_handle, parent_handle})
    operations.argv_by_handle[parent_handle] = parent_argv
    operations.identities[parent_handle] = process_identity(
        9,
        parent_argv,
        created=19,
    )
    operations.signaled_sequences[parent_handle] = [False]
    operations.exit_codes[parent_handle] = runtime.STILL_ACTIVE
    boundary = runtime.RetainedChildBootstrapProcessBoundary(
        capability_handle,
        parent_handle,
        operations=operations,
    )
    current = boundary.current_process_identity()
    parent = boundary.retained_process_identity(parent_handle)
    authority = {
        "process": current,
        "argv_sha256": current["argv_sha256"],
        "wrapper_process": parent,
        "parent_handle": {
            "value": parent_handle,
            "process": parent,
            "access": "synchronize_query_limited_information",
            "inherited": True,
        },
    }
    return operations, boundary, authority


def test_retained_child_boundary_rebinds_current_parent_and_handle_policy():
    operations, boundary, authority = _retained_child_boundary()
    current_handle = boundary.current_handle_value
    assert boundary.current_argv() == ("wrapper",)
    assert boundary.current_process_identity() == authority["process"]
    assert boundary.retained_process_identity(52) == authority["wrapper_process"]
    assert boundary.parent_signaled(52) is False
    assert boundary.prove_inherited_handle_policy(
        capability_handle=51,
        parent_handle=52,
        process_authority=authority,
    ) is True
    assert ("argv", current_handle) in operations.calls
    assert ("identity", current_handle) in operations.calls
    assert ("argv", 52) in operations.calls
    assert ("identity", 52) in operations.calls

    wrong_authority = {
        **authority,
        "argv_sha256": "f" * 64,
    }
    assert boundary.prove_inherited_handle_policy(
        capability_handle=51,
        parent_handle=52,
        process_authority=wrong_authority,
    ) is False
    assert boundary.prove_inherited_handle_policy(
        capability_handle=53,
        parent_handle=52,
        process_authority=authority,
    ) is False
    with pytest.raises(runtime.ProcessIdentityError, match="not retained"):
        boundary.parent_signaled(53)

    proof = boundary.close_owned_handles(
        deadline_monotonic_ns=100,
        monotonic_ns=sequence_clock(10, 20),
    )
    assert proof.proved
    assert proof.current_process_closed
    assert proof.parent_process_closed
    assert proof.within_deadline
    assert boundary.closed
    assert 51 not in operations.closed
    assert operations.closed[-2:] == [52, current_handle]
    with pytest.raises(runtime.ProcessIdentityError, match="closed"):
        boundary.current_process_identity()


def test_retained_child_boundary_identity_and_inheritance_drift_fail_closed():
    operations, boundary, authority = _retained_child_boundary()
    operations.inheritable.remove(51)
    assert boundary.prove_inherited_handle_policy(
        capability_handle=51,
        parent_handle=52,
        process_authority=authority,
    ) is False
    operations.inheritable.add(51)
    operations.identities[52] = {
        **operations.identities[52],
        "creation_filetime_100ns": 20,
    }
    with pytest.raises(runtime.ProcessIdentityError, match="changed"):
        boundary.retained_process_identity(52)
    boundary.close_owned_handles(
        deadline_monotonic_ns=100,
        monotonic_ns=sequence_clock(10, 20),
    )


def test_retained_child_boundary_constructor_failure_closes_owned_parent_and_current():
    operations = FakeWindowsOperations()
    operations.inheritable.add(51)
    with pytest.raises(runtime.ProcessIdentityError, match="not inherited"):
        runtime.RetainedChildBootstrapProcessBoundary(
            51,
            52,
            operations=operations,
        )
    assert operations.closed == [52]

    operations = FakeWindowsOperations()
    operations.inheritable.update({51, 52})
    operations.argv_by_handle[52] = ("parent",)
    operations.identities[52] = process_identity(9, ("parent",), created=19)
    original_query = operations.query_process_argv

    def fail_parent_query(handle):
        if handle == 52:
            raise OSError("injected parent argv failure")
        return original_query(handle)

    operations.query_process_argv = fail_parent_query
    with pytest.raises(OSError, match="injected parent"):
        runtime.RetainedChildBootstrapProcessBoundary(
            51,
            52,
            operations=operations,
        )
    assert set(operations.closed) == {52, 100}

    alias = FakeWindowsOperations()
    with pytest.raises(runtime.ProcessIdentityError, match="distinct"):
        runtime.RetainedChildBootstrapProcessBoundary(
            51,
            51,
            operations=alias,
        )
    assert alias.closed == []


def test_retained_child_boundary_bounded_close_attempts_all_and_records_failures():
    operations, boundary, _authority = _retained_child_boundary()
    current_handle = boundary.current_handle_value
    operations.close_failures.add(52)
    with pytest.raises(runtime.HandleCloseError, match="parent_process:OSError"):
        boundary.close_owned_handles(
            deadline_monotonic_ns=100,
            monotonic_ns=sequence_clock(10, 20),
        )
    assert current_handle in operations.closed
    assert 52 not in operations.closed
    assert boundary.last_close_proof.current_process_closed
    assert not boundary.last_close_proof.parent_process_closed

    operations.close_failures.remove(52)
    with pytest.raises(runtime.PoweredDeadlineExpired, match="cannot be refreshed"):
        boundary.close_owned_handles(
            deadline_monotonic_ns=1_000,
            monotonic_ns=sequence_clock(30, 40),
        )
    proof = boundary.close_owned_handles(
        deadline_monotonic_ns=100,
        monotonic_ns=sequence_clock(30, 40),
    )
    assert proof.proved and boundary.closed

    expired_operations, expired, _authority = _retained_child_boundary()
    with pytest.raises(runtime.PoweredDeadlineExpired, match="outside"):
        expired.close_owned_handles(
            deadline_monotonic_ns=100,
            monotonic_ns=sequence_clock(100, 101),
        )
    assert expired.closed
    assert expired.last_close_proof.within_deadline is False
    assert {52, expired.current_handle_value}.issubset(
        set(expired_operations.closed)
    )


@pytest.mark.skipif(os.name != "nt", reason="child bootstrap handles are Windows-only")
def test_real_retained_child_boundary_uses_inherited_safe_handles_and_closes():
    operations = runtime.Win32ProcessOperations()
    qpc = runtime.WindowsQpcProvider()
    capability_read, capability_write = operations.create_capability_pipe()
    parent_handle = operations.open_process(
        os.getpid(),
        inheritable=True,
        terminate_access=False,
    )
    boundary = None
    try:
        boundary = runtime.RetainedChildBootstrapProcessBoundary(
            capability_read,
            parent_handle,
            operations=operations,
        )
        current = boundary.current_process_identity()
        parent = boundary.retained_process_identity(parent_handle)
        assert current["pid"] == parent["pid"] == os.getpid()
        assert current["argv_sha256"] == runtime.argv_sha256(
            boundary.current_argv()
        )
        authority = {
            "process": current,
            "argv_sha256": current["argv_sha256"],
            "wrapper_process": parent,
            "parent_handle": {
                "value": parent_handle,
                "process": parent,
                "access": "synchronize_query_limited_information",
                "inherited": True,
            },
        }
        assert boundary.prove_inherited_handle_policy(
            capability_handle=capability_read,
            parent_handle=parent_handle,
            process_authority=authority,
        ) is True
        assert boundary.parent_signaled(parent_handle) is False
        deadline = qpc.now_ns() + 1_000_000_000
        proof = boundary.close_owned_handles(
            deadline_monotonic_ns=deadline,
            monotonic_ns=qpc.now_ns,
        )
        assert proof.proved
    finally:
        if boundary is not None and not boundary.closed:
            boundary.close_owned_handles(
                deadline_monotonic_ns=qpc.now_ns() + 1_000_000_000,
                monotonic_ns=qpc.now_ns,
            )
        runtime.close_owned_handles(
            {
                "capability_read": capability_read,
                "capability_write": capability_write,
            },
            operations=operations,
        )


def test_safe_handle_close_attempts_every_handle_and_surfaces_failures():
    operations = FakeWindowsOperations()
    operations.close_failures.add(2)
    with pytest.raises(runtime.HandleCloseError, match="second:OSError"):
        runtime.close_owned_handles(
            {"first": 1, "second": 2, "third": 3}, operations=operations
        )
    assert operations.closed == [1, 3]
    with pytest.raises(runtime.HandleCloseError, match="distinct"):
        runtime.close_owned_handles(
            {"first": 4, "second": 4}, operations=operations
        )


def test_capability_pipe_release_is_exact_single_use_and_close_proved():
    operations = FakeWindowsOperations()
    pipe = runtime.create_capability_pipe(operations=operations)
    pipe.close_parent_reader()
    pipe.release(
        bytes(range(32)),
        deadline_monotonic_ns=20,
        monotonic_ns=lambda: 10,
    )
    assert operations.writes == [
        (12, runtime.encode_capability_frame(bytes(range(32))))
    ]
    assert pipe.released and pipe.read_closed and pipe.write_closed
    with pytest.raises(runtime.CapabilityProtocolError, match="single-use"):
        pipe.release(
            bytes(range(32)),
            deadline_monotonic_ns=20,
            monotonic_ns=lambda: 10,
        )

    early = FakeWindowsOperations()
    early.pipe_available = 1
    with pytest.raises(runtime.CapabilityProtocolError, match="before child"):
        runtime.create_capability_pipe(operations=early)
    assert set(early.closed) == {11, 12}


def _fake_blocked_child(tmp_path, *, operations=None):
    operations = operations or FakeWindowsOperations()
    parent = runtime.retain_process(
        10, ["wrapper"], inheritable=True, operations=operations
    )
    pipe = runtime.create_capability_pipe(operations=operations)
    operations.inheritable.update({31, 32, 33})
    argv = [r"C:\Python\python.exe", "-c", "blocked", str(pipe.read_handle)]
    child = runtime.spawn_blocked_child(
        argv,
        cwd=str(tmp_path.resolve()),
        environment={"SAFE": "1"},
        capability_pipe=pipe,
        parent_process=parent,
        stdin_handle=31,
        stdout_handle=32,
        stderr_handle=33,
        operations=operations,
    )
    return operations, parent, child


def test_blocked_child_assigns_exact_nonbreakaway_job_before_release(tmp_path):
    operations, parent, child = _fake_blocked_child(tmp_path)
    assert child.inherited_handles == (11, parent.handle_value, 32, 33, 31)
    assert child.containment.to_primitive() == {
        "handle_value": 300,
        "assigned_before_capability_release": True,
        "breakaway_allowed": False,
        "silent_breakaway_allowed": False,
        "kill_on_close": False,
        "process_in_job": True,
    }
    assign_index = next(
        index for index, call in enumerate(operations.calls) if call[0] == "assign"
    )
    reader_close_index = next(
        index
        for index, call in enumerate(operations.calls)
        if call == ("close", 11)
    )
    assert assign_index < reader_close_index
    child.release_capability(
        bytes(range(32)),
        deadline_monotonic_ns=20,
        monotonic_ns=lambda: 10,
    )
    write_index = next(
        index for index, call in enumerate(operations.calls) if call[0] == "write"
    )
    assert assign_index < write_index
    operations.signaled_sequences[400] = [True, True]
    operations.exit_codes[400] = 0
    operations.job_pid_sequences = [(), ()]
    proof = runtime.wait_job_process_tree_exit(
        child,
        deadline_monotonic_ns=10,
        monotonic_ns=sequence_clock(1, 2),
        wait_ns=lambda _value: None,
    )
    assert proof.state == "exited" and proof.natural_exit_proved
    assert child.close_retained_handles(tree_exit_proof=proof) == (
        "primary_thread",
        "process",
        "job",
    )
    parent.close()


def test_bad_job_policy_fails_closed_terminates_and_closes(tmp_path):
    operations = FakeWindowsOperations()
    operations.job_flags = runtime.JOB_OBJECT_LIMIT_BREAKAWAY_OK
    parent = runtime.retain_process(
        10, ["wrapper"], inheritable=True, operations=operations
    )
    pipe = runtime.create_capability_pipe(operations=operations)
    operations.inheritable.update({31, 32, 33})
    with pytest.raises(runtime.ChildSpawnError, match="blocked child spawn failed"):
        runtime.spawn_blocked_child(
            [r"C:\Python\python.exe", "-c", "blocked", "11"],
            cwd=str(tmp_path.resolve()),
            environment={"SAFE": "1"},
            capability_pipe=pipe,
            parent_process=parent,
            stdin_handle=31,
            stdout_handle=32,
            stderr_handle=33,
            operations=operations,
        )
    assert operations.terminated_jobs == [300]
    assert {11, 12, 300, 400, 401}.issubset(set(operations.closed))
    assert operations.writes == []
    parent.close()


def test_bounded_process_wait_distinguishes_exit_residue_and_forced_exit():
    operations = FakeWindowsOperations()
    retained = runtime.retain_process(
        10,
        ["wrapper"],
        terminate_access=True,
        operations=operations,
    )
    operations.signaled_sequences[retained.handle_value] = [False, True]
    operations.exit_codes[retained.handle_value] = 7
    exited = runtime.wait_retained_process_exit(
        retained,
        deadline_monotonic_ns=10,
        monotonic_ns=sequence_clock(1, 2),
        wait_ns=lambda _value: None,
    )
    assert exited.state == "exited"
    assert exited.exit_code == 7
    assert exited.termination_is_cleanup_proof is False

    operations.signaled_sequences[retained.handle_value] = [False]
    operations.exit_codes[retained.handle_value] = runtime.STILL_ACTIVE
    residue = runtime.wait_retained_process_exit(
        retained,
        deadline_monotonic_ns=10,
        monotonic_ns=lambda: 10,
        wait_ns=lambda _value: None,
    )
    assert residue.state == "residue"
    forced = runtime.terminate_process_residue(
        retained,
        exit_code=9,
        deadline_monotonic_ns=20,
        monotonic_ns=lambda: 11,
        wait_ns=lambda _value: None,
    )
    assert forced.state == "terminated_residue"
    assert forced.exit_code == 9
    assert forced.termination_attempted is True
    assert forced.termination_is_cleanup_proof is False
    retained.close()


def test_job_tree_exit_and_termination_residue_are_distinct(tmp_path):
    operations, parent, child = _fake_blocked_child(tmp_path)
    operations.signaled_sequences[400] = [False]
    operations.job_pid_sequences = [(400,)]
    residue = runtime.wait_job_process_tree_exit(
        child,
        deadline_monotonic_ns=10,
        monotonic_ns=lambda: 10,
        wait_ns=lambda _value: None,
    )
    assert residue.state == "residue"
    assert residue.observations[-1].active_pids == (400,)
    forced = runtime.terminate_job_process_tree_residue(
        child,
        exit_code=9,
        deadline_monotonic_ns=20,
        monotonic_ns=sequence_clock(11, 12, 13),
        wait_ns=lambda _value: None,
    )
    assert forced.state == "terminated_residue"
    assert forced.natural_exit_proved is False
    assert forced.termination_is_cleanup_proof is False
    assert operations.terminated_jobs == [300]
    child.close_retained_handles(tree_exit_proof=forced)
    parent.close()


@pytest.mark.skipif(os.name != "nt", reason="explicit handle-list spawn is Windows-only")
def test_real_blocked_helper_is_job_contained_released_and_exits(tmp_path):
    import msvcrt

    operations = runtime.Win32ProcessOperations()
    parent = runtime.retain_current_process(
        inheritable=True, operations=operations
    )
    pipe = runtime.create_capability_pipe(operations=operations)
    stdin = open(os.devnull, "rb", buffering=0)
    stdout = open(tmp_path / "helper.stdout", "wb", buffering=0)
    stderr = open(tmp_path / "helper.stderr", "wb", buffering=0)
    child = None
    tree_proof = None
    try:
        standard_handles = []
        for stream in (stdin, stdout, stderr):
            handle = msvcrt.get_osfhandle(stream.fileno())
            os.set_handle_inheritable(handle, True)
            standard_handles.append(handle)
        helper = (
            "import msvcrt,os,sys;"
            "f=msvcrt.open_osfhandle(int(sys.argv[1]),os.O_RDONLY);"
            "d=os.read(f,36);"
            "raise SystemExit(0 if len(d)==36 else 7)"
        )
        argv = [
            sys.executable,
            "-E",
            "-s",
            "-B",
            "-c",
            helper,
            str(pipe.read_handle),
        ]
        child = runtime.spawn_blocked_child(
            argv,
            cwd=str(tmp_path.resolve()),
            environment=dict(os.environ),
            capability_pipe=pipe,
            parent_process=parent,
            stdin_handle=standard_handles[0],
            stdout_handle=standard_handles[1],
            stderr_handle=standard_handles[2],
            operations=operations,
        )
        assert child.process.signaled() is False
        assert child.containment.to_primitive() == {
            "handle_value": child.job_handle,
            "assigned_before_capability_release": True,
            "breakaway_allowed": False,
            "silent_breakaway_allowed": False,
            "kill_on_close": False,
            "process_in_job": True,
        }
        child.release_capability(
            bytes(range(32)),
            deadline_monotonic_ns=time.perf_counter_ns() + 3_000_000_000,
        )
        tree_proof = runtime.wait_job_process_tree_exit(
            child,
            deadline_monotonic_ns=time.perf_counter_ns() + 3_000_000_000,
        )
        assert tree_proof.state == "exited"
        assert tree_proof.observations[-1].active_pids == ()
        assert tree_proof.observations[-1].root_signaled is True
        assert child.process.exit_code() == 0
    finally:
        if child is not None:
            if tree_proof is None or tree_proof.state != "exited":
                tree_proof = runtime.wait_job_process_tree_exit(
                    child,
                    deadline_monotonic_ns=time.perf_counter_ns()
                    + 1_000_000_000,
                )
                if tree_proof.state != "exited":
                    tree_proof = runtime.terminate_job_process_tree_residue(
                        child,
                        exit_code=9,
                        deadline_monotonic_ns=time.perf_counter_ns()
                        + 3_000_000_000,
                    )
            assert tree_proof is not None
            child.close_retained_handles(tree_exit_proof=tree_proof)
        else:
            pipe.abort()
        parent.close()
        stdin.close()
        stdout.close()
        stderr.close()


def test_exact_zero_rate_thrust_rejects_bool_negative_zero_and_extra_fields():
    zero = {
        "roll_rate_rad_s": 0.0,
        "pitch_rate_rad_s": 0,
        "yaw_rate_rad_s": 0.0,
        "thrust": 0.0,
    }
    assert runtime.exact_zero_rate_thrust(zero)
    assert not runtime.exact_zero_rate_thrust({**zero, "thrust": -0.0})
    assert not runtime.exact_zero_rate_thrust({**zero, "thrust": True})
    assert not runtime.exact_zero_rate_thrust({**zero, "extra": 0})


def _authorize_production(guard, category="attitude_target", **overrides):
    values = {
        "now_monotonic_ns": 10,
        "deadline_monotonic_ns": 20,
        "role_valid": True,
        "parent_alive": True,
        "lease_valid": True,
        "peer_frozen": True,
        "source_valid": True,
        "source_promoted": True,
    }
    values.update(overrides)
    return guard.authorize_production(category, **values)


def test_production_guard_allows_only_announcements_before_promotion():
    announcements = runtime.PoweredOutboundGuards()
    announcements.enable_production()
    _authorize_production(
        announcements, "timesync", source_promoted=False
    )
    _authorize_production(
        announcements, "gcs_heartbeat", source_promoted=False
    )
    assert announcements.production_latched is False

    flight = runtime.PoweredOutboundGuards()
    flight.enable_production()
    with pytest.raises(runtime.OutboundAuthorityError, match="not promoted"):
        _authorize_production(flight, source_promoted=False)
    assert flight.production_latched is True
    with pytest.raises(runtime.OutboundAuthorityError, match="re-enabled"):
        flight.enable_production()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parent_alive": False}, "parent"),
        ({"lease_valid": False}, "lease"),
        ({"role_valid": False}, "role"),
        ({"peer_frozen": False}, "peer"),
        ({"now_monotonic_ns": 20}, "deadline"),
    ],
)
def test_production_guard_failure_permanently_latches(overrides, message):
    guard = runtime.PoweredOutboundGuards()
    guard.enable_production()
    with pytest.raises(runtime.OutboundAuthorityError, match=message):
        _authorize_production(guard, **overrides)
    assert guard.production_latched


def _authorize_cleanup(guard, category="sim_reset", **overrides):
    values = {
        "now_monotonic_ns": 10,
        "deadline_monotonic_ns": 20,
        "parent_alive": True,
        "lease_valid": True,
        "source_promoted": True,
        "exact_zero": None,
    }
    values.update(overrides)
    return guard.authorize_cleanup(category, **values)


def test_cleanup_guard_is_single_epoch_exact_zero_and_takeover_only():
    guard = runtime.PoweredOutboundGuards()
    guard.enable_production()
    guard.enable_cleanup_live(
        parent_alive=True, lease_valid=True, source_promoted=True
    )
    assert guard.production_latched
    assert guard.cleanup_state == "enabled_live"
    _authorize_cleanup(guard, "attitude_target", exact_zero=True)
    with pytest.raises(runtime.OutboundAuthorityError, match="exact zero"):
        _authorize_cleanup(guard, "attitude_target", exact_zero=False)

    with pytest.raises(runtime.OutboundAuthorityError, match="takeover"):
        _authorize_cleanup(guard, parent_alive=False)
    assert guard.cleanup_state == "takeover_pending"
    guard.enable_cleanup_takeover(
        parent_signaled=True,
        abandoned_lease_owned=True,
        authority_valid=True,
        source_promoted=True,
    )
    _authorize_cleanup(guard, parent_alive=False)
    guard.close_cleanup()
    assert guard.cleanup_state == "closed"
    with pytest.raises(runtime.OutboundAuthorityError, match="single-use"):
        guard.enable_cleanup_live(
            parent_alive=True, lease_valid=True, source_promoted=True
        )


def test_cleanup_guard_denies_arm_but_retains_mandatory_cleanup_authority():
    guard = runtime.PoweredOutboundGuards()
    guard.enable_cleanup_live(
        parent_alive=True, lease_valid=True, source_promoted=True
    )
    with pytest.raises(runtime.OutboundAuthorityError, match="allowlisted"):
        _authorize_cleanup(guard, "arm")
    assert guard.cleanup_state == "enabled_live"
    _authorize_cleanup(guard, "disarm")
    _authorize_cleanup(guard, "sim_reset")


def test_cleanup_takeover_requires_every_proof_and_deadline_closes():
    guard = runtime.PoweredOutboundGuards()
    guard.note_parent_death()
    with pytest.raises(runtime.OutboundAuthorityError, match="invalid"):
        guard.enable_cleanup_takeover(
            parent_signaled=True,
            abandoned_lease_owned=False,
            authority_valid=True,
            source_promoted=True,
        )
    assert guard.cleanup_state == "takeover_pending"

    guard.enable_cleanup_takeover(
        parent_signaled=True,
        abandoned_lease_owned=True,
        authority_valid=True,
        source_promoted=True,
    )
    with pytest.raises(runtime.OutboundAuthorityError, match="deadline"):
        _authorize_cleanup(
            guard,
            parent_alive=False,
            now_monotonic_ns=20,
        )
    assert guard.cleanup_state == "closed"


def test_udp_owner_snapshots_cover_both_families_and_require_stability():
    first_port, second_port = os_assigned_udp_ports(2)
    operations = FakeUdpOwnerOperations(
        ipv4=[
            [(first_port, 7)],
            [(first_port, 8)],
            [(first_port, 8)],
        ],
        ipv6=[
            [(second_port, 9)],
            [(second_port, 9)],
            [(second_port, 9)],
        ],
    )
    snapshots = runtime.capture_stable_udp_owner_snapshots(
        [second_port, first_port],
        deadline_monotonic_ns=10,
        operations=operations,
        monotonic_ns=sequence_clock(0, 1, 2, 3, 4, 5),
        wait_ns=lambda _value: None,
    )
    assert len(snapshots) == 2
    assert snapshots[0].ownership_key() == snapshots[1].ownership_key()
    assert snapshots[-1].owner_pids(socket.AF_INET, first_port) == (8,)
    assert snapshots[-1].owner_pids(socket.AF_INET6, second_port) == (9,)
    assert snapshots[-1].to_contract_observation() == {
        "observed_monotonic_ns": 5,
        f"ipv4_{first_port}": [8],
        f"ipv4_{second_port}": [],
        f"ipv6_{first_port}": [],
        f"ipv6_{second_port}": [9],
    }


@pytest.mark.skipif(os.name != "nt", reason="UDP owner PID tables are Windows-only")
def test_real_udp_owner_tables_map_current_pid_on_ephemeral_ipv4_and_ipv6():
    ipv4 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ipv6 = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    try:
        ipv4.bind(("127.0.0.1", 0))
        ipv6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        ipv6.bind(("::1", 0))
        ipv4_port = ipv4.getsockname()[1]
        ipv6_port = ipv6.getsockname()[1]
        snapshots = runtime.capture_stable_udp_owner_snapshots(
            [ipv4_port, ipv6_port],
            deadline_monotonic_ns=time.perf_counter_ns() + 2_000_000_000,
        )
        assert snapshots[-1].owner_pids(socket.AF_INET, ipv4_port) == (
            os.getpid(),
        )
        assert snapshots[-1].owner_pids(socket.AF_INET6, ipv6_port) == (
            os.getpid(),
        )
    finally:
        ipv4.close()
        ipv6.close()


class FakeSocket:
    def __init__(
        self,
        *,
        fail_bind=False,
        reuse=0,
        exclusive_value=1,
        close_error=False,
    ):
        self.family = socket.AF_INET
        self.type = socket.SOCK_DGRAM
        self.fail_bind = fail_bind
        self.reuse = reuse
        self.exclusive_value = exclusive_value
        self.close_error = close_error
        self.exclusive = 0
        self.bound = None
        self.closed = False
        self.calls = []

    def getsockopt(self, level, option):
        self.calls.append(("get", level, option))
        if option == socket.SO_REUSEADDR:
            return self.reuse
        return self.exclusive

    def setsockopt(self, level, option, value):
        self.calls.append(("set", level, option, value))
        self.exclusive = self.exclusive_value if value else 0

    def bind(self, endpoint):
        self.calls.append(("bind", endpoint))
        if self.fail_bind:
            raise OSError("injected bind conflict")
        self.bound = endpoint

    def getsockname(self):
        host, port = self.bound
        return (host, 40_001 if port == 0 else port)

    def close(self):
        self.calls.append(("close",))
        if self.close_error:
            raise OSError("injected close failure")
        self.closed = True


def test_exclusive_udp_sets_and_proves_option_before_ephemeral_bind():
    fake = FakeSocket()
    endpoint = runtime.create_exclusive_udp_endpoint(
        "127.0.0.1",
        0,
        socket_factory=lambda family, kind: fake,
        exclusive_option=9_999,
    )
    assert endpoint.proof() == {
        "family": "AF_INET",
        "requested": {"host": "127.0.0.1", "port": 0},
        "actual": {"host": "127.0.0.1", "port": 40_001},
        "socket_policy": "ipv4-exclusive-address-use",
    }
    set_index = next(i for i, call in enumerate(fake.calls) if call[0] == "set")
    bind_index = next(i for i, call in enumerate(fake.calls) if call[0] == "bind")
    assert set_index < bind_index
    assert all(
        call[:3] != ("set", socket.SOL_SOCKET, socket.SO_REUSEADDR)
        for call in fake.calls
    )
    endpoint.close()
    assert fake.closed


def test_exclusive_udp_raw_socket_transfer_is_one_shot_and_removes_ownership():
    fake = FakeSocket()
    endpoint = runtime.create_exclusive_udp_endpoint(
        "127.0.0.1",
        0,
        socket_factory=lambda family, kind: fake,
        exclusive_option=9_999,
    )

    transferred = endpoint.transfer_socket()

    assert transferred is fake
    assert endpoint.socket is None
    assert endpoint.socket_transferred is True
    assert endpoint.closed is False
    with pytest.raises(runtime.ExclusiveUdpError, match="already transferred"):
        endpoint.transfer_socket()
    with pytest.raises(runtime.ExclusiveUdpError, match="no longer owns"):
        endpoint.close()
    assert fake.closed is False
    transferred.close()
    assert fake.closed is True


@pytest.mark.parametrize(
    ("fake", "message"),
    [
        (FakeSocket(fail_bind=True), "injected bind conflict"),
        (FakeSocket(reuse=1), "SO_REUSEADDR"),
        (FakeSocket(exclusive_value=0), "verification"),
    ],
)
def test_exclusive_udp_closes_every_partial_failure(fake, message):
    with pytest.raises((OSError, runtime.ExclusiveUdpError), match=message):
        runtime.create_exclusive_udp_endpoint(
            "127.0.0.1",
            0,
            socket_factory=lambda family, kind: fake,
            exclusive_option=9_999,
        )
    assert fake.closed


def test_exclusive_udp_partial_close_failure_is_not_suppressed():
    fake = FakeSocket(fail_bind=True, close_error=True)
    with pytest.raises(runtime.ExclusiveUdpError, match="could not close"):
        runtime.create_exclusive_udp_endpoint(
            "127.0.0.1",
            0,
            socket_factory=lambda family, kind: fake,
            exclusive_option=9_999,
        )
    assert fake.calls[-1] == ("close",)


def test_real_exclusive_udp_uses_only_os_assigned_loopback_port():
    if not hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        pytest.skip("SO_EXCLUSIVEADDRUSE is Windows-specific")
    with runtime.create_exclusive_udp_endpoint("127.0.0.1", 0) as endpoint:
        assert endpoint.requested_port == 0
        assert endpoint.actual_port > 0
        competitor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            with pytest.raises(OSError):
                competitor.bind((endpoint.actual_host, endpoint.actual_port))
        finally:
            competitor.close()
    assert endpoint.closed


def test_exclusive_probe_requires_caller_port_and_surfaces_close_failure():
    (port,) = os_assigned_udp_ports()
    fake = FakeSocket(close_error=True)

    def endpoint_factory(host, target):
        return runtime.ExclusiveUdpEndpoint(
            socket=fake,
            requested_host=host,
            requested_port=target,
            actual_host=host,
            actual_port=target,
            exclusive_option=9_999,
        )

    with pytest.raises(runtime.ExclusiveUdpError, match="close"):
        runtime.probe_exclusive_udp_port(
            "127.0.0.1",
            port,
            deadline_monotonic_ns=3,
            monotonic_ns=sequence_clock(1, 2),
            endpoint_factory=endpoint_factory,
        )
    with pytest.raises(ValueError, match="exact integer"):
        runtime.probe_exclusive_udp_port(
            "127.0.0.1",
            0,
            deadline_monotonic_ns=3,
            monotonic_ns=sequence_clock(1, 2),
        )


@pytest.mark.skipif(
    not hasattr(socket, "SO_EXCLUSIVEADDRUSE"),
    reason="exclusive UDP probes are Windows-only",
)
def test_real_exclusive_probe_binds_and_closes_caller_ephemeral_port():
    (port,) = os_assigned_udp_ports()
    proof = runtime.probe_exclusive_udp_port(
        "127.0.0.1",
        port,
        deadline_monotonic_ns=time.perf_counter_ns() + 2_000_000_000,
    )
    assert proof.to_primitive() == {
        "host": "127.0.0.1",
        "port": port,
        "started_monotonic_ns": proof.started_monotonic_ns,
        "ended_monotonic_ns": proof.ended_monotonic_ns,
        "result": "bound_and_closed",
    }
    rebound = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rebound.bind(("127.0.0.1", port))
    finally:
        rebound.close()


class FakeMessage:
    def __init__(self, raw: bytes, message_id: int, message_type="HEARTBEAT"):
        self.raw = raw
        self.message_id = message_id
        self.message_type = message_type

    def get_type(self):
        return self.message_type

    def get_msgbuf(self):
        return self.raw

    def get_msgId(self):
        return self.message_id


class FakeMavlinkParser:
    def __init__(self, *, count=1, bad_buffer=False, bad_type=False):
        self.robust_parsing = True
        self.count = count
        self.bad_buffer = bad_buffer
        self.bad_type = bad_type

    def parse_buffer(self, raw):
        if bytes(raw[-2:]) != b"\xaa\x55":
            raise ValueError("bad injected checksum")
        if raw[0] == 0xFE:
            message_id = raw[5]
        else:
            message_id = raw[7] | (raw[8] << 8) | (raw[9] << 16)
        buffer = bytes(raw[:-1]) if self.bad_buffer else bytes(raw)
        message_type = "BAD_DATA" if self.bad_type else "HEARTBEAT"
        return [FakeMessage(buffer, message_id, message_type)] * self.count


def v1_frame(payload=b"", message_id=0):
    return bytes((0xFE, len(payload), 1, 1, 1, message_id)) + payload + b"\xaa\x55"


def v2_frame(payload=b"", message_id=0, incompat=0):
    return (
        bytes(
            (
                0xFD,
                len(payload),
                incompat,
                0,
                1,
                1,
                1,
                message_id & 0xFF,
                (message_id >> 8) & 0xFF,
                (message_id >> 16) & 0xFF,
            )
        )
        + payload
        + b"\xaa\x55"
    )


@pytest.mark.parametrize("raw", [v1_frame(b"abc", 7), v2_frame(b"abc", 257)])
def test_scratch_mavlink_accepts_one_full_unsigned_checksum_valid_frame(raw):
    message = runtime.validate_scratch_mavlink_datagram(
        raw, parser_factory=FakeMavlinkParser
    )
    assert bytes(message.get_msgbuf()) == raw


@pytest.mark.parametrize(
    ("raw", "factory", "message"),
    [
        (b"", FakeMavlinkParser, "empty"),
        (b"x" * 12, FakeMavlinkParser, "marker"),
        (v1_frame() + b"x", FakeMavlinkParser, "length"),
        (v2_frame(incompat=1), FakeMavlinkParser, "signed or incompatible"),
        (v1_frame()[:-1] + b"x", FakeMavlinkParser, "parser rejected"),
        (v1_frame(), lambda: FakeMavlinkParser(count=2), "exactly one"),
        (v1_frame(), lambda: FakeMavlinkParser(bad_buffer=True), "does not equal"),
        (v1_frame(), lambda: FakeMavlinkParser(bad_type=True), "BAD_DATA"),
    ],
)
def test_scratch_mavlink_rejects_partial_signed_bad_or_multi_frame(
    raw, factory, message
):
    with pytest.raises(runtime.MavlinkDatagramError, match=message):
        runtime.validate_scratch_mavlink_datagram(raw, parser_factory=factory)


def test_source_freeze_ignores_malformed_then_freezes_and_promotes_same_peer():
    gate = runtime.MavlinkSourceFreeze(FakeMavlinkParser)
    malformed = gate.ingest(v1_frame()[:-1] + b"x", ("127.0.0.1", 40_001))
    assert malformed.malformed and gate.peer is None
    accepted = gate.ingest(v1_frame(), ("127.0.0.1", 40_001))
    assert accepted.accepted and accepted.peer_frozen_now
    assert gate.peer == ("127.0.0.1", 40_001)
    assert gate.outbound_permitted("timesync")
    assert not gate.outbound_permitted("sim_reset")
    assert not gate.observe_fresh_stream("HEARTBEAT")
    assert not gate.observe_fresh_stream("RACE_STATUS")
    assert gate.observe_fresh_stream("HIGHRES_IMU")
    assert gate.promoted
    assert gate.outbound_permitted("sim_reset")


def test_source_freeze_latches_second_or_nonloopback_source_before_parsing():
    parser_calls = []

    def parser_factory():
        parser_calls.append(True)
        return FakeMavlinkParser()

    gate = runtime.MavlinkSourceFreeze(parser_factory)
    assert gate.ingest(v1_frame(), ("127.0.0.1", 40_001)).accepted
    second = gate.ingest(v1_frame(), ("127.0.0.1", 40_002))
    assert second.rejected_source
    assert len(parser_calls) == 1
    assert gate.source_rejected_latched
    assert not gate.outbound_permitted("timesync")

    other = runtime.MavlinkSourceFreeze(parser_factory)
    rejected = other.ingest(v1_frame(), ("192.0.2.1", 40_003))
    assert rejected.rejected_source
    assert other.peer is None


def test_source_rejection_blocks_production_but_preserves_frozen_peer_cleanup():
    gate = runtime.MavlinkSourceFreeze(FakeMavlinkParser)
    assert gate.ingest(v1_frame(), ("127.0.0.1", 40_001)).accepted
    assert not gate.observe_fresh_stream("HEARTBEAT")
    assert gate.ingest(v1_frame(), ("127.0.0.1", 40_002)).rejected_source
    assert not gate.outbound_permitted("timesync")
    assert not gate.observe_fresh_stream("RACE_STATUS")
    assert gate.observe_fresh_stream("HIGHRES_IMU")
    assert gate.promoted

    guard = runtime.PoweredOutboundGuards()
    guard.enable_production()
    with pytest.raises(runtime.OutboundAuthorityError, match="source validity"):
        _authorize_production(guard, "gcs_heartbeat", source_valid=False)
    assert guard.production_latched

    guard.enable_cleanup_live(
        parent_alive=True,
        lease_valid=True,
        source_promoted=gate.promoted,
    )
    _authorize_cleanup(guard, "disarm")


def test_loopback_peer_and_promotion_tokens_are_strict():
    assert runtime.normalize_ipv4_loopback_peer(("127.0.0.1", 1234)) == (
        "127.0.0.1",
        1234,
    )
    for invalid in (
        ("0.0.0.0", 1),
        ("::1", 1),
        ("127.0.0.1", 0),
        ["127.0.0.1", 1],
    ):
        with pytest.raises(runtime.MavlinkDatagramError):
            runtime.normalize_ipv4_loopback_peer(invalid)
    gate = runtime.MavlinkSourceFreeze(FakeMavlinkParser)
    with pytest.raises(runtime.MavlinkDatagramError, match="promotion"):
        gate.observe_fresh_stream("ACTUATOR_OUTPUT_STATUS")


def test_stable_file_identity_hashes_regular_file_under_explicit_root(tmp_path):
    target = tmp_path / "candidate.bin"
    target.write_bytes(b"candidate bytes")
    identity = runtime.stable_file_identity(target.resolve(), root=tmp_path.resolve())
    assert identity.path == str(target.resolve())
    assert identity.size_bytes == len(b"candidate bytes")
    assert identity.sha256 == hashlib.sha256(b"candidate bytes").hexdigest()
    assert identity.to_primitive(name="candidate") == {
        "name": "candidate",
        "path": str(target.resolve()),
        "size_bytes": len(b"candidate bytes"),
        "sha256": hashlib.sha256(b"candidate bytes").hexdigest(),
    }


def test_stable_file_identity_rejects_relative_outside_and_oversize(tmp_path):
    target = tmp_path / "candidate.bin"
    target.write_bytes(b"1234")
    with pytest.raises(runtime.StableFileError, match="absolute"):
        runtime.stable_file_identity(Path("candidate.bin"))
    other_root = tmp_path / "other"
    other_root.mkdir()
    with pytest.raises(runtime.StableFileError, match="outside"):
        runtime.stable_file_identity(target.resolve(), root=other_root.resolve())
    with pytest.raises(runtime.StableFileError, match="bounded"):
        runtime.stable_file_identity(target.resolve(), max_bytes=3)


def test_stable_file_identity_rejects_lexical_alias(tmp_path):
    target = tmp_path / "candidate.bin"
    target.write_bytes(b"1234")
    alias_component = tmp_path / "alias-component"
    alias_component.mkdir()
    aliased = alias_component / ".." / target.name
    with pytest.raises(runtime.StableFileError, match="canonical"):
        runtime.stable_file_identity(aliased, root=tmp_path.resolve())
