import json
from pathlib import Path

import pytest

import scripts.aigp_live_lease as lease_module
from scripts.aigp_live_lease import (
    LIVE_LEASE_EVIDENCE_SCHEMA,
    LIVE_LEASE_MUTEX_NAME,
    WAIT_ABANDONED,
    WAIT_FAILED,
    WAIT_OBJECT_0,
    WAIT_TIMEOUT,
    LiveLeaseAbandonedError,
    LiveLeaseBusyError,
    LiveLeaseCleanupError,
    LiveLeaseEvidenceError,
    LiveLeaseUnavailableError,
    LiveSimulatorLease,
    load_live_lease_evidence,
    validate_live_lease_evidence,
)


class FakeKernel:
    def __init__(
        self,
        *,
        handle=1234,
        wait_result=WAIT_OBJECT_0,
        release_result=True,
        close_result=True,
        last_error=0,
    ):
        self.handle = handle
        self.wait_result = wait_result
        self.release_result = release_result
        self.close_result = close_result
        self.error = last_error
        self.calls = []

    def create_mutex(self, name):
        self.calls.append(("create", name))
        return self.handle

    def wait(self, handle, timeout_ms):
        self.calls.append(("wait", handle, timeout_ms))
        return self.wait_result

    def release_mutex(self, handle):
        self.calls.append(("release", handle))
        return self.release_result

    def close_handle(self, handle):
        self.calls.append(("close", handle))
        return self.close_result

    def last_error(self):
        self.calls.append(("last_error",))
        return self.error


def ticking_clock(*values):
    remaining = iter(values)
    return lambda: next(remaining)


def expected_initial_evidence(*, timestamp=1_000, phase="acquired"):
    return {
        "schema": LIVE_LEASE_EVIDENCE_SCHEMA,
        "mutex_name": LIVE_LEASE_MUTEX_NAME,
        "owner_token": "a" * 64,
        "wrapper_pid": 4321,
        "acquired_wall_time_ns": timestamp,
        "heartbeat_wall_time_ns": timestamp,
        "phase": phase,
        "child_pid": None,
        "released_wall_time_ns": None,
    }


def make_lease(
    tmp_path,
    kernel,
    monkeypatch,
    *,
    clock=None,
    process_guard=None,
    filename="lease.json",
):
    monkeypatch.setattr(lease_module.secrets, "token_hex", lambda _count: "a" * 64)
    return LiveSimulatorLease(
        (tmp_path / filename).resolve(),
        _kernel=kernel,
        _clock_ns=clock or ticking_clock(1_000, 2_000, 3_000),
        _pid=4321,
        _process_guard=process_guard or lease_module.threading.Lock(),
    )


def test_context_acquires_heartbeats_and_cleanly_releases(tmp_path, monkeypatch):
    kernel = FakeKernel()
    lease = make_lease(tmp_path, kernel, monkeypatch)

    with lease as acquired:
        assert acquired is lease
        assert lease.is_active is True
        assert load_live_lease_evidence(lease.evidence_path) == (
            expected_initial_evidence()
        )
        heartbeat = lease.heartbeat(phase="child_running", child_pid=9876)
        assert heartbeat["heartbeat_wall_time_ns"] == 2_000
        assert heartbeat["phase"] == "child_running"
        assert heartbeat["child_pid"] == 9876
        assert kernel.calls == [
            ("create", LIVE_LEASE_MUTEX_NAME),
            ("wait", 1234, 0),
        ]

    final = load_live_lease_evidence(lease.evidence_path)
    assert lease.is_active is False
    assert final["phase"] == "released"
    assert final["heartbeat_wall_time_ns"] == 3_000
    assert final["released_wall_time_ns"] == 3_000
    assert final["child_pid"] == 9876
    assert kernel.calls[-2:] == [("release", 1234), ("close", 1234)]


def test_busy_mutex_fails_nonblocking_and_closes_handle(tmp_path, monkeypatch):
    kernel = FakeKernel(wait_result=WAIT_TIMEOUT)
    lease = make_lease(tmp_path, kernel, monkeypatch)

    with pytest.raises(LiveLeaseBusyError, match="busy"):
        lease.acquire()

    assert kernel.calls == [
        ("create", LIVE_LEASE_MUTEX_NAME),
        ("wait", 1234, 0),
        ("close", 1234),
    ]
    assert not lease.evidence_path.exists()


def test_abandoned_mutex_is_released_and_rejected(tmp_path, monkeypatch):
    kernel = FakeKernel(wait_result=WAIT_ABANDONED)
    lease = make_lease(tmp_path, kernel, monkeypatch)

    with pytest.raises(LiveLeaseAbandonedError, match="abandoned"):
        lease.acquire()

    assert kernel.calls == [
        ("create", LIVE_LEASE_MUTEX_NAME),
        ("wait", 1234, 0),
        ("release", 1234),
        ("close", 1234),
    ]
    assert not lease.evidence_path.exists()


@pytest.mark.parametrize("wait_result", [WAIT_FAILED, 0xDEADBEEF])
def test_failed_or_unverifiable_wait_is_inaccessible(
    tmp_path, monkeypatch, wait_result
):
    kernel = FakeKernel(wait_result=wait_result, last_error=5)
    lease = make_lease(tmp_path, kernel, monkeypatch)

    with pytest.raises(LiveLeaseUnavailableError, match="failed|unverifiable"):
        lease.acquire()

    assert ("close", 1234) in kernel.calls
    assert ("release", 1234) not in kernel.calls
    assert not lease.evidence_path.exists()


def test_inaccessible_create_fails_before_wait_or_evidence(tmp_path, monkeypatch):
    kernel = FakeKernel(handle=0, last_error=5)
    lease = make_lease(tmp_path, kernel, monkeypatch)

    with pytest.raises(LiveLeaseUnavailableError, match="CreateMutexW"):
        lease.acquire()

    assert ("wait", 0, 0) not in kernel.calls
    assert not lease.evidence_path.exists()


def test_body_failure_still_performs_clean_release(tmp_path, monkeypatch):
    kernel = FakeKernel()
    lease = make_lease(
        tmp_path,
        kernel,
        monkeypatch,
        clock=ticking_clock(1_000, 2_000),
    )

    with pytest.raises(ValueError, match="probe failed"):
        with lease:
            raise ValueError("probe failed")

    assert kernel.calls[-2:] == [("release", 1234), ("close", 1234)]
    assert load_live_lease_evidence(lease.evidence_path)[
        "released_wall_time_ns"
    ] == 2_000


def test_release_failure_closes_handle_and_never_claims_clean_release(
    tmp_path, monkeypatch
):
    kernel = FakeKernel(release_result=False, last_error=6)
    lease = make_lease(tmp_path, kernel, monkeypatch)
    lease.acquire()

    with pytest.raises(LiveLeaseCleanupError, match="ReleaseMutex"):
        lease.release()

    assert kernel.calls[-3:] == [
        ("release", 1234),
        ("last_error",),
        ("close", 1234),
    ]
    evidence = load_live_lease_evidence(lease.evidence_path)
    assert evidence["released_wall_time_ns"] is None
    assert evidence["phase"] == "acquired"


def test_initial_evidence_failure_releases_mutex_and_removes_temporary(
    tmp_path, monkeypatch
):
    kernel = FakeKernel()
    lease = make_lease(tmp_path, kernel, monkeypatch)

    def fail_link(_source, _destination):
        raise OSError("publication failed")

    monkeypatch.setattr(lease_module.os, "link", fail_link)
    with pytest.raises(LiveLeaseEvidenceError, match="publication failed"):
        lease.acquire()

    assert kernel.calls[-2:] == [("release", 1234), ("close", 1234)]
    assert not lease.evidence_path.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_invalid_initial_timestamp_releases_mutex_before_failing(
    tmp_path, monkeypatch
):
    kernel = FakeKernel()
    lease = make_lease(
        tmp_path,
        kernel,
        monkeypatch,
        clock=lambda: 0,
    )

    with pytest.raises(LiveLeaseEvidenceError, match="publication failed"):
        lease.acquire()

    assert kernel.calls[-2:] == [("release", 1234), ("close", 1234)]
    assert not lease.evidence_path.exists()


def test_every_evidence_publication_uses_same_directory_atomic_replace(
    tmp_path, monkeypatch
):
    kernel = FakeKernel()
    lease = make_lease(tmp_path, kernel, monkeypatch)
    real_link = lease_module.os.link
    real_replace = lease_module.os.replace
    links = []
    replacements = []

    def observed_link(source, destination):
        source = Path(source)
        destination = Path(destination)
        assert source.exists()
        links.append((source, destination))
        return real_link(source, destination)

    def observed_replace(source, destination):
        source = Path(source)
        destination = Path(destination)
        assert source.exists()
        replacements.append((source, destination))
        return real_replace(source, destination)

    monkeypatch.setattr(lease_module.os, "link", observed_link)
    monkeypatch.setattr(lease_module.os, "replace", observed_replace)
    with lease:
        lease.heartbeat(phase="post_port_check")

    assert len(links) == 3
    assert all(source.parent == tmp_path for source, _target in links)
    assert all(target == lease.evidence_path for _source, target in links)
    assert len(replacements) == 2
    assert all(source == lease.evidence_path for source, _target in replacements)
    assert all(target.parent == tmp_path for _source, target in replacements)
    assert all(".previous." in target.name for _source, target in replacements)
    assert list(tmp_path.glob(".*.tmp")) == []
    raw = lease.evidence_path.read_bytes()
    assert raw.endswith(b"\n")
    assert json.loads(raw)["phase"] == "released"


def test_failed_heartbeat_publication_preserves_previous_complete_envelope(
    tmp_path, monkeypatch
):
    kernel = FakeKernel()
    lease = make_lease(tmp_path, kernel, monkeypatch)
    lease.acquire()
    before = lease.evidence_path.read_bytes()
    real_replace = lease_module.os.replace
    failed_once = False

    def fail_one_replace(source, destination):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("heartbeat publication failed")
        return real_replace(source, destination)

    monkeypatch.setattr(lease_module.os, "replace", fail_one_replace)
    with pytest.raises(LiveLeaseEvidenceError, match="atomically published"):
        lease.heartbeat(phase="child_running", child_pid=9876)

    assert lease.is_active is True
    assert lease.evidence_path.read_bytes() == before
    assert load_live_lease_evidence(lease.evidence_path)["phase"] == "acquired"
    lease.release()
    assert load_live_lease_evidence(lease.evidence_path)["phase"] == "released"


def test_process_guard_rejects_recursive_same_thread_acquisition(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(lease_module.secrets, "token_hex", lambda _count: "a" * 64)
    process_guard = lease_module.threading.Lock()
    first_kernel = FakeKernel(handle=111)
    second_kernel = FakeKernel(handle=222)
    first = LiveSimulatorLease(
        (tmp_path / "first.json").resolve(),
        _kernel=first_kernel,
        _clock_ns=ticking_clock(1_000, 2_000),
        _pid=4321,
        _process_guard=process_guard,
    )
    second = LiveSimulatorLease(
        (tmp_path / "second.json").resolve(),
        _kernel=second_kernel,
        _clock_ns=ticking_clock(1_000),
        _pid=4321,
        _process_guard=process_guard,
    )
    first.acquire()

    with pytest.raises(LiveLeaseBusyError, match="already owned in this process"):
        second.acquire()

    assert second_kernel.calls == []
    first.release()


def test_heartbeat_preserves_bound_child_and_rejects_rebinding(
    tmp_path, monkeypatch
):
    kernel = FakeKernel()
    lease = make_lease(
        tmp_path,
        kernel,
        monkeypatch,
        clock=ticking_clock(1_000, 2_000, 3_000, 4_000),
    )
    lease.acquire()
    lease.heartbeat(phase="child_running", child_pid=9876)

    preserved = lease.heartbeat(phase="postcheck")
    assert preserved["child_pid"] == 9876
    with pytest.raises(LiveLeaseEvidenceError, match="cannot change"):
        lease.heartbeat(phase="postcheck", child_pid=None)

    lease.release()
    assert load_live_lease_evidence(lease.evidence_path)["child_pid"] == 9876


def test_initial_atomic_create_never_overwrites_racing_evidence_path(
    tmp_path, monkeypatch
):
    kernel = FakeKernel()
    lease = make_lease(tmp_path, kernel, monkeypatch)
    sentinel = b"preexisting-private-evidence\n"

    def race_link(_source, destination):
        Path(destination).write_bytes(sentinel)
        raise FileExistsError("racing evidence appeared")

    monkeypatch.setattr(lease_module.os, "link", race_link)
    with pytest.raises(LiveLeaseEvidenceError, match="publication failed"):
        lease.acquire()

    assert lease.evidence_path.read_bytes() == sentinel
    assert kernel.calls[-2:] == [("release", 1234), ("close", 1234)]


def test_heartbeat_refuses_replaced_or_foreign_owner_evidence(
    tmp_path, monkeypatch
):
    kernel = FakeKernel()
    lease = make_lease(tmp_path, kernel, monkeypatch)
    lease.acquire()
    original = lease.evidence
    foreign = {**original, "owner_token": "b" * 64}
    lease.evidence_path.write_text(
        json.dumps(foreign, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(LiveLeaseEvidenceError, match="current owner state"):
        lease.heartbeat(phase="precheck")

    lease.evidence_path.write_text(
        json.dumps(original, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    lease.release()


def test_unverified_release_poisons_process_guard_against_recursive_reentry(
    tmp_path, monkeypatch
):
    process_guard = lease_module.threading.Lock()
    first_kernel = FakeKernel(handle=111, release_result=False, last_error=6)
    first = make_lease(
        tmp_path,
        first_kernel,
        monkeypatch,
        process_guard=process_guard,
        filename="first.json",
    )
    first.acquire()
    with pytest.raises(LiveLeaseCleanupError, match="ReleaseMutex"):
        first.release()

    second_kernel = FakeKernel(handle=222)
    second = make_lease(
        tmp_path,
        second_kernel,
        monkeypatch,
        process_guard=process_guard,
        filename="second.json",
    )
    with pytest.raises(LiveLeaseBusyError, match="already owned in this process"):
        second.acquire()
    assert second_kernel.calls == []


def test_failed_acquire_cleanup_poison_blocks_another_same_process_lease(
    tmp_path, monkeypatch
):
    process_guard = lease_module.threading.Lock()
    first_kernel = FakeKernel(handle=111, release_result=False, last_error=6)
    first = make_lease(
        tmp_path,
        first_kernel,
        monkeypatch,
        clock=lambda: 0,
        process_guard=process_guard,
        filename="first.json",
    )
    with pytest.raises(LiveLeaseEvidenceError, match="publication failed"):
        first.acquire()

    second_kernel = FakeKernel(handle=222)
    second = make_lease(
        tmp_path,
        second_kernel,
        monkeypatch,
        process_guard=process_guard,
        filename="second.json",
    )
    with pytest.raises(LiveLeaseBusyError, match="already owned in this process"):
        second.acquire()
    assert second_kernel.calls == []


def test_update_race_restores_unexpected_predecessor_instead_of_clobbering(
    tmp_path, monkeypatch
):
    kernel = FakeKernel()
    lease = make_lease(tmp_path, kernel, monkeypatch)
    lease.acquire()
    original = lease.evidence
    foreign = {**original, "owner_token": "b" * 64}
    foreign_bytes = (
        json.dumps(foreign, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    real_replace = lease_module.os.replace
    raced = False

    def interposed_replace(source, destination):
        nonlocal raced
        if Path(source) == lease.evidence_path and not raced:
            raced = True
            lease.evidence_path.write_bytes(foreign_bytes)
        return real_replace(source, destination)

    monkeypatch.setattr(lease_module.os, "replace", interposed_replace)
    with pytest.raises(LiveLeaseEvidenceError, match="changed during"):
        lease.heartbeat(phase="precheck")

    assert lease.evidence_path.read_bytes() == foreign_bytes
    lease.evidence_path.write_text(
        json.dumps(original, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    lease.release()


def test_evidence_validation_is_exact_and_rejects_timestamp_or_type_drift():
    evidence = expected_initial_evidence()
    assert validate_live_lease_evidence(evidence) == evidence

    unknown = {**evidence, "unexpected": True}
    with pytest.raises(LiveLeaseEvidenceError, match="missing or unknown"):
        validate_live_lease_evidence(unknown)

    boolean_pid = {**evidence, "wrapper_pid": True}
    with pytest.raises(LiveLeaseEvidenceError, match="wrapper PID"):
        validate_live_lease_evidence(boolean_pid)

    regressed = {**evidence, "heartbeat_wall_time_ns": 999}
    with pytest.raises(LiveLeaseEvidenceError, match="precede acquisition"):
        validate_live_lease_evidence(regressed)

    false_release = {**evidence, "phase": "released"}
    with pytest.raises(LiveLeaseEvidenceError, match="release timestamp"):
        validate_live_lease_evidence(false_release)
