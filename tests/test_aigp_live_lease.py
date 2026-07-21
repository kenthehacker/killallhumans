import hashlib
import json
import threading
from pathlib import Path

import pytest

import scripts.aigp_live_lease as lease_module
from scripts.aigp_live_lease import (
    LIVE_LEASE_EVIDENCE_SCHEMA,
    LIVE_LEASE_MUTEX_NAME,
    POWERED_LIVE_LEASE_EVIDENCE_SCHEMA,
    POWERED_LIVE_LEASE_LEDGER_SCHEMA,
    WAIT_ABANDONED,
    WAIT_FAILED,
    WAIT_OBJECT_0,
    WAIT_TIMEOUT,
    LiveLeaseAbandonedError,
    LiveLeaseBusyError,
    LiveLeaseCleanupError,
    LiveLeaseError,
    LiveLeaseEvidenceError,
    LiveLeaseUnavailableError,
    LiveSimulatorLease,
    DelegatedPoweredLeaseBoundary,
    DelegatedPoweredLeaseProof,
    PoweredLeaseLedgerStore,
    PoweredLiveSimulatorLease,
    derive_powered_takeover_owner_sha256,
    load_live_lease_evidence,
    load_powered_live_lease_index,
    load_powered_live_lease_record,
    validate_live_lease_evidence,
    validate_powered_live_lease_ledger,
    validate_powered_live_lease_index,
    validate_powered_live_lease_record,
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


def powered_process(tmp_path, *, pid=4321, marker="a"):
    return {
        "pid": pid,
        "creation_filetime_100ns": 10_000 + pid,
        "windows_session_id": 1,
        "image_path": str((tmp_path / f"python-{pid}.exe").resolve()),
        "image_sha256": marker * 64,
        "argv_sha256": marker * 64,
    }


def make_powered_store(
    tmp_path, *, clock_values=None, publisher=None, publish_final_index=True
):
    ledger = tmp_path / "lease"
    ledger.mkdir(exist_ok=True)
    values = iter(clock_values or range(100, 1000, 10))
    kwargs = {}
    if publisher is not None:
        kwargs["_no_replace_publish"] = publisher
    wrapper = powered_process(tmp_path)
    return PoweredLeaseLedgerStore(
        ledger.resolve(),
        (tmp_path / "live-lease.json").resolve(),
        task_id="vq2-package2-powered-calibration-pilot",
        session_id="F00",
        attempt_id="F00-A01",
        attempt_envelope_sha256="b" * 64,
        attempt_context_sha256="c" * 64,
        wrapper_process=wrapper,
        qpc_frequency_hz=10_000_000,
        publish_final_index=publish_final_index,
        _clock_ns=lambda: next(values),
        **kwargs,
    )


def append_wrapper_record(store, tmp_path, event, phase, *, release=False):
    wrapper = powered_process(tmp_path)
    return store.append(
        event=event,
        owner_role="wrapper",
        owner_token_sha256="d" * 64,
        owner_process=wrapper,
        phase=phase,
        abandoned=False,
        release_proved=release,
    )


def test_powered_record_validator_is_exact_and_rejects_bool_or_role_drift(
    tmp_path,
):
    store = make_powered_store(tmp_path)
    record, _digest = append_wrapper_record(
        store, tmp_path, "acquired", "lease_acquire"
    )
    assert record["schema"] == POWERED_LIVE_LEASE_EVIDENCE_SCHEMA
    assert validate_powered_live_lease_record(record) == record

    with pytest.raises(LiveLeaseEvidenceError, match="missing or unknown"):
        validate_powered_live_lease_record({**record, "unknown": 1})
    with pytest.raises(LiveLeaseEvidenceError, match="non-negative"):
        validate_powered_live_lease_record({**record, "generation": True})
    with pytest.raises(LiveLeaseEvidenceError, match="wrapper role"):
        validate_powered_live_lease_record(
            {
                **record,
                "owner_process": powered_process(tmp_path, pid=9999, marker="e"),
            }
        )
    with pytest.raises(LiveLeaseEvidenceError, match="generation zero cannot"):
        validate_powered_live_lease_record({**record, "abandoned": True})
    with pytest.raises(LiveLeaseEvidenceError, match="generation zero"):
        validate_powered_live_lease_record({**record, "event": "heartbeat"})


def test_powered_store_appends_canonical_chain_and_seals_released_index(tmp_path):
    store = make_powered_store(tmp_path)
    events = [
        ("acquired", "lease_acquire", False),
        ("heartbeat", "launcher_return", False),
        ("phase", "child_supervision", False),
        ("release_intent", "lease_release_and_verify", False),
        ("released", "lease_release_and_verify", True),
    ]
    digests = []
    for event, phase, release in events:
        record, digest = append_wrapper_record(
            store, tmp_path, event, phase, release=release
        )
        digests.append(digest)
        path = store.ledger_directory / (
            f"generation-{record['generation']:06d}.json"
        )
        assert load_powered_live_lease_record(path) == record
        assert list(store.ledger_directory.glob("pending-*.json")) == []

        # Returned/property values cannot mutate the in-memory chain that will
        # later be sealed and used as authority lineage.
        record["wrapper_process"]["pid"] = 99_999
        copied_records = store.records
        copied_records[-1]["wrapper_process"]["pid"] = 88_888
        assert store.records[-1]["wrapper_process"]["pid"] == 4321

    index, index_sha = store.seal_released_index()
    assert index["schema"] == POWERED_LIVE_LEASE_LEDGER_SCHEMA
    assert index["final_generation"] == 4
    assert index["final_record_sha256"] == digests[-1]
    assert index["release_proved"] is True
    assert load_powered_live_lease_index(store.final_index_path) == index
    assert len(index_sha) == 64
    raw = store.final_index_path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw == (
        json.dumps(
            json.loads(raw),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def test_powered_store_can_defer_final_index_publication_to_wrapper(tmp_path):
    store = make_powered_store(tmp_path, publish_final_index=False)
    events = (
        ("acquired", "lease_acquire", False),
        ("release_intent", "lease_release_and_verify", False),
        ("released", "lease_release_and_verify", True),
    )
    for event, phase, release in events:
        append_wrapper_record(store, tmp_path, event, phase, release=release)

    index, index_sha = store.seal_released_index()

    assert validate_powered_live_lease_index(index) == index
    assert index_sha == hashlib.sha256(
        (
            json.dumps(
                index,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    assert not store.final_index_path.exists()


def test_powered_store_preserves_failed_pending_and_refuses_initial_restart(
    tmp_path,
):
    def fail_publish(_source, _destination):
        raise LiveLeaseEvidenceError("injected no-replace failure")

    store = make_powered_store(tmp_path, publisher=fail_publish)
    with pytest.raises(LiveLeaseEvidenceError, match="injected"):
        append_wrapper_record(store, tmp_path, "acquired", "lease_acquire")

    pending = list(store.ledger_directory.glob("pending-*.json"))
    assert len(pending) == 1
    resumed = make_powered_store(tmp_path)
    assert len(resumed.orphaned_pending_files) == 1
    with pytest.raises(LiveLeaseEvidenceError, match="cannot start over"):
        append_wrapper_record(resumed, tmp_path, "acquired", "lease_acquire")


def test_powered_store_binds_one_orphaned_pending_into_takeover_chain(tmp_path):
    store = make_powered_store(tmp_path)
    append_wrapper_record(store, tmp_path, "acquired", "lease_acquire")
    real_publish = lease_module._default_no_replace_publish

    def fail_publish(_source, _destination):
        raise LiveLeaseEvidenceError("parent died before generation publication")

    store._no_replace_publish = fail_publish
    with pytest.raises(LiveLeaseEvidenceError, match="parent died"):
        append_wrapper_record(store, tmp_path, "heartbeat", "child_supervision")

    resumed = make_powered_store(tmp_path, publisher=real_publish)
    child = powered_process(tmp_path, pid=7777, marker="e")
    takeover, _digest = resumed.append(
        event="takeover",
        owner_role="powered-child-parent-death",
        owner_token_sha256="f" * 64,
        owner_process=child,
        child_process=child,
        phase="child_cleanup",
        abandoned=True,
    )
    assert takeover["generation"] == 1
    assert takeover["orphaned_pending"] == resumed.orphaned_pending_files[0]
    resumed.append(
        event="release_intent",
        owner_role="powered-child-parent-death",
        owner_token_sha256="f" * 64,
        owner_process=child,
        child_process=child,
        phase="lease_release_and_verify",
        abandoned=True,
    )
    resumed.append(
        event="released",
        owner_role="powered-child-parent-death",
        owner_token_sha256="f" * 64,
        owner_process=child,
        child_process=child,
        phase="lease_release_and_verify",
        abandoned=True,
        release_proved=True,
    )
    index, _index_sha = resumed.seal_released_index()
    assert len(index["orphaned_pending_files"]) == 1
    assert index["orphaned_pending_files"][0]["owner_role"] == "wrapper"

    reloaded = PoweredLeaseLedgerStore(
        resumed.ledger_directory,
        (tmp_path / "second-index.json").resolve(),
        task_id=resumed.task_id,
        session_id=resumed.session_id,
        attempt_id=resumed.attempt_id,
        attempt_envelope_sha256=resumed.attempt_envelope_sha256,
        attempt_context_sha256=resumed.attempt_context_sha256,
        wrapper_process=resumed.wrapper_process,
        qpc_frequency_hz=resumed.qpc_frequency_hz,
    )
    assert len(reloaded.records) == 4
    assert reloaded.records[1]["event"] == "takeover"
    assert reloaded.orphaned_pending_files == index["orphaned_pending_files"]


def test_powered_takeover_rejects_fabricated_or_wrong_owner_orphan(tmp_path):
    store = make_powered_store(tmp_path)
    append_wrapper_record(store, tmp_path, "acquired", "lease_acquire")
    child = powered_process(tmp_path, pid=7777, marker="e")
    fabricated = {
        "path": str((store.ledger_directory / "fabricated.json").resolve()),
        "size_bytes": 1,
        "sha256": "a" * 64,
        "owner_role": "wrapper",
    }
    with pytest.raises(LiveLeaseEvidenceError, match="sole preserved"):
        store.append(
            event="takeover",
            owner_role="powered-child-parent-death",
            owner_token_sha256="f" * 64,
            owner_process=child,
            child_process=child,
            phase="child_cleanup",
            abandoned=True,
            orphaned_pending=fabricated,
        )


def test_powered_reload_requires_bound_takeover_orphan_file(tmp_path):
    store = make_powered_store(tmp_path)
    append_wrapper_record(store, tmp_path, "acquired", "lease_acquire")
    real_publish = lease_module._default_no_replace_publish

    def fail_publish(_source, _destination):
        raise LiveLeaseEvidenceError("injected pending publication")

    store._no_replace_publish = fail_publish
    with pytest.raises(LiveLeaseEvidenceError, match="injected pending"):
        append_wrapper_record(store, tmp_path, "heartbeat", "child_supervision")

    resumed = make_powered_store(tmp_path, publisher=real_publish)
    child = powered_process(tmp_path, pid=7777, marker="e")
    takeover, _digest = resumed.append(
        event="takeover",
        owner_role="powered-child-parent-death",
        owner_token_sha256="f" * 64,
        owner_process=child,
        child_process=child,
        phase="child_cleanup",
        abandoned=True,
    )
    Path(takeover["orphaned_pending"]["path"]).unlink()

    with pytest.raises(LiveLeaseEvidenceError, match="must match"):
        PoweredLeaseLedgerStore(
            resumed.ledger_directory,
            (tmp_path / "second-index.json").resolve(),
            task_id=resumed.task_id,
            session_id=resumed.session_id,
            attempt_id=resumed.attempt_id,
            attempt_envelope_sha256=resumed.attempt_envelope_sha256,
            attempt_context_sha256=resumed.attempt_context_sha256,
            wrapper_process=resumed.wrapper_process,
            qpc_frequency_hz=resumed.qpc_frequency_hz,
        )


def test_powered_store_rejects_owner_change_without_takeover(tmp_path):
    store = make_powered_store(tmp_path)
    append_wrapper_record(store, tmp_path, "acquired", "lease_acquire")
    child = powered_process(tmp_path, pid=7777, marker="e")
    with pytest.raises(LiveLeaseEvidenceError, match="only at takeover"):
        store.append(
            event="phase",
            owner_role="powered-child-parent-death",
            owner_token_sha256="f" * 64,
            owner_process=child,
            child_process=child,
            phase="child_cleanup",
            abandoned=True,
        )


def test_powered_index_validator_rejects_noncontiguous_or_unreleased(tmp_path):
    store = make_powered_store(tmp_path)
    append_wrapper_record(store, tmp_path, "acquired", "lease_acquire")
    append_wrapper_record(
        store,
        tmp_path,
        "release_intent",
        "lease_release_and_verify",
    )
    released, _ = append_wrapper_record(
        store,
        tmp_path,
        "released",
        "lease_release_and_verify",
        release=True,
    )
    index, _ = store.seal_released_index()
    assert validate_powered_live_lease_index(index) == index

    bad_generation = {
        **index,
        "records": [dict(item) for item in index["records"]],
    }
    bad_generation["records"][1]["generation"] = 3
    with pytest.raises(LiveLeaseEvidenceError, match="contiguous"):
        validate_powered_live_lease_index(bad_generation)

    with pytest.raises(LiveLeaseEvidenceError, match="final released"):
        validate_powered_live_lease_index(
            {
                **index,
                "records": [
                    *index["records"][:-1],
                    {**index["records"][-1], "event": "phase"},
                ],
            }
        )
    assert released["release_proved"] is True


def test_powered_mutex_initial_owner_records_heartbeats_and_releases(tmp_path):
    store = make_powered_store(tmp_path)
    kernel = FakeKernel()
    wrapper = powered_process(tmp_path)
    lease = PoweredLiveSimulatorLease(
        store,
        owner_role="wrapper",
        owner_token_sha256="d" * 64,
        owner_process=wrapper,
        initial_phase="lease_acquire",
        _kernel=kernel,
        _clock_ns=ticking_clock(100, 200, 300, 400, 500),
        _process_guard=lease_module.threading.Lock(),
    )

    with lease:
        assert lease.is_active is True
        assert lease.heartbeat(phase="launcher_return")["event"] == "heartbeat"
        assert lease.publish_phase("child_supervision")["event"] == "phase"

    assert lease.is_active is False
    assert [row["event"] for row in store.records] == [
        "acquired",
        "heartbeat",
        "phase",
        "release_intent",
        "released",
    ]
    assert load_powered_live_lease_index(store.final_index_path)[
        "release_proved"
    ] is True
    assert kernel.calls == [
        ("create", LIVE_LEASE_MUTEX_NAME),
        ("wait", 1234, 5000),
        ("release", 1234),
        ("close", 1234),
    ]


def test_powered_initial_owner_rejects_abandoned_mutex_without_record(tmp_path):
    store = make_powered_store(tmp_path)
    kernel = FakeKernel(wait_result=WAIT_ABANDONED)
    lease = PoweredLiveSimulatorLease(
        store,
        owner_role="wrapper",
        owner_token_sha256="d" * 64,
        owner_process=powered_process(tmp_path),
        initial_phase="lease_acquire",
        _kernel=kernel,
        _process_guard=lease_module.threading.Lock(),
    )

    with pytest.raises(LiveLeaseAbandonedError, match="abandoned"):
        lease.acquire()

    assert store.records == []
    assert kernel.calls[-2:] == [("release", 1234), ("close", 1234)]


def test_powered_takeover_requires_abandoned_and_revalidates_authority(tmp_path):
    store = make_powered_store(tmp_path)
    append_wrapper_record(store, tmp_path, "acquired", "lease_acquire")
    kernel = FakeKernel(wait_result=WAIT_ABANDONED)
    child = powered_process(tmp_path, pid=7777, marker="e")
    calls = []

    def verify():
        calls.append("verify")
        return True

    lease = PoweredLiveSimulatorLease(
        store,
        owner_role="powered-child-parent-death",
        owner_token_sha256="f" * 64,
        owner_process=child,
        child_process=child,
        initial_phase="child_cleanup",
        takeover=True,
        verify_takeover=verify,
        _kernel=kernel,
        _clock_ns=ticking_clock(200, 300, 400),
        _process_guard=lease_module.threading.Lock(),
    )
    lease.acquire()
    assert calls == ["verify", "verify"]
    assert store.records[-1]["event"] == "takeover"
    assert store.records[-1]["abandoned"] is True
    lease.release()
    assert store.records[-1]["event"] == "released"
    assert all(row["abandoned"] for row in store.records[1:])


def test_powered_takeover_rejects_clean_mutex_transfer_and_sends_nothing(
    tmp_path,
):
    store = make_powered_store(tmp_path)
    append_wrapper_record(store, tmp_path, "acquired", "lease_acquire")
    kernel = FakeKernel(wait_result=WAIT_OBJECT_0)
    child = powered_process(tmp_path, pid=7777, marker="e")
    lease = PoweredLiveSimulatorLease(
        store,
        owner_role="powered-child-parent-death",
        owner_token_sha256="f" * 64,
        owner_process=child,
        child_process=child,
        initial_phase="child_cleanup",
        takeover=True,
        verify_takeover=lambda: True,
        _kernel=kernel,
        _clock_ns=ticking_clock(200),
        _process_guard=lease_module.threading.Lock(),
    )

    with pytest.raises(LiveLeaseUnavailableError, match="requires exact abandoned"):
        lease.acquire()

    assert len(store.records) == 1
    assert kernel.calls[-2:] == [("release", 1234), ("close", 1234)]


def test_powered_heartbeat_gap_latches_but_still_releases_with_proof(tmp_path):
    store = make_powered_store(tmp_path)
    kernel = FakeKernel()
    lease = PoweredLiveSimulatorLease(
        store,
        owner_role="wrapper",
        owner_token_sha256="d" * 64,
        owner_process=powered_process(tmp_path),
        initial_phase="lease_acquire",
        _kernel=kernel,
        _clock_ns=ticking_clock(
            100,
            100 + PoweredLiveSimulatorLease.HEARTBEAT_MAX_GAP_NS + 1,
            100 + PoweredLiveSimulatorLease.HEARTBEAT_MAX_GAP_NS + 2,
            100 + PoweredLiveSimulatorLease.HEARTBEAT_MAX_GAP_NS + 3,
        ),
        _process_guard=lease_module.threading.Lock(),
    )
    lease.acquire()
    with pytest.raises(LiveLeaseEvidenceError, match="maximum gap"):
        lease.heartbeat()
    assert lease.is_latched_invalid is True

    index, _digest = lease.release()
    assert index["release_proved"] is True
    assert [row["event"] for row in store.records] == [
        "acquired",
        "release_intent",
        "released",
    ]


def test_powered_process_bindings_are_single_assignment(tmp_path):
    store = make_powered_store(tmp_path)
    kernel = FakeKernel()
    lease = PoweredLiveSimulatorLease(
        store,
        owner_role="wrapper",
        owner_token_sha256="d" * 64,
        owner_process=powered_process(tmp_path),
        initial_phase="lease_acquire",
        _kernel=kernel,
        _clock_ns=ticking_clock(100, 200, 300),
        _process_guard=lease_module.threading.Lock(),
    )
    lease.acquire()
    child = powered_process(tmp_path, pid=7777, marker="e")
    lease.bind_child_process(child)
    lease.bind_child_process(child)
    with pytest.raises(LiveLeaseEvidenceError, match="cannot be rebound"):
        lease.bind_child_process(powered_process(tmp_path, pid=8888, marker="f"))
    lease.release()
    assert store.records[-1]["child_process"] == child


class FakeDelegatedContract:
    @staticmethod
    def canonical_json_file_bytes(value):
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    @classmethod
    def canonical_file_sha256(cls, value):
        return hashlib.sha256(cls.canonical_json_file_bytes(value)).hexdigest()

    @staticmethod
    def _copy(value):
        return json.loads(json.dumps(value))

    @classmethod
    def validate_attempt(cls, value):
        if type(value) is not dict or set(value) != {
            "schema",
            "context",
            "context_sha256",
            "capabilities",
        }:
            raise ValueError("invalid injected attempt")
        if value["schema"] != "test-powered-attempt/1":
            raise ValueError("invalid injected attempt schema")
        context = value["context"]
        if type(context) is not dict or set(context) != {
            "task_id",
            "session_id",
            "attempt_id",
            "host",
            "wrapper_process",
            "paths",
        }:
            raise ValueError("invalid injected attempt context")
        if type(context["paths"]) is not dict or set(context["paths"]) != {
            "child_authority",
            "cleanup_authority",
        }:
            raise ValueError("invalid injected attempt paths")
        if type(value["capabilities"]) is not dict or set(
            value["capabilities"]
        ) != {"lease_owner_sha256", "child_sha256", "cleanup_sha256"}:
            raise ValueError("invalid injected attempt capabilities")
        return cls._copy(value)

    @classmethod
    def validate_process_authority(cls, value, *, attempt):
        if type(value) is not dict or set(value) != {
            "role",
            "process",
            "wrapper_process",
            "parent_handle",
            "capability_sha256",
            "lease_record_sha256",
            "attempt_envelope_sha256",
            "attempt_context_sha256",
        }:
            raise ValueError("invalid injected process authority")
        role = value["role"]
        if role not in {"powered_child", "cleanup_fallback"}:
            raise ValueError("invalid injected process role")
        capability_key = (
            "child_sha256" if role == "powered_child" else "cleanup_sha256"
        )
        if (
            value["wrapper_process"] != attempt["context"]["wrapper_process"]
            or value["parent_handle"]
            != {
                "value": 52,
                "process": attempt["context"]["wrapper_process"],
                "access": "synchronize_query_limited_information",
                "inherited": True,
            }
            or value["capability_sha256"]
            != attempt["capabilities"][capability_key]
            or value["attempt_context_sha256"] != attempt["context_sha256"]
            or value["attempt_envelope_sha256"]
            != cls.canonical_file_sha256(attempt)
        ):
            raise ValueError("injected process authority is not attempt-bound")
        return cls._copy(value)


class SteppingClock:
    def __init__(self, current, *, step=1):
        self.current = current
        self.step = step
        self.calls = []

    def __call__(self):
        value = self.current
        self.current += self.step
        self.calls.append(value)
        return value


class CallbackKernel(FakeKernel):
    def __init__(self, *, on_wait=None, **kwargs):
        super().__init__(**kwargs)
        self.on_wait = on_wait

    def wait(self, handle, timeout_ms):
        self.calls.append(("wait", handle, timeout_ms))
        if self.on_wait is not None:
            self.on_wait()
        return self.wait_result


def delegated_capability_sha256(role, context_sha256, secret):
    domain = (
        b"aigp-vq2-powered-child/1"
        if role == "powered_child"
        else b"aigp-vq2-powered-cleanup/1"
    )
    return hashlib.sha256(
        domain
        + b"\x00"
        + bytes.fromhex(context_sha256)
        + b"\x00"
        + secret
    ).hexdigest()


def make_delegated_fixture(tmp_path, *, role="cleanup_fallback", kernel=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    wrapper = powered_process(tmp_path, pid=4321, marker="a")
    child = powered_process(tmp_path, pid=7777, marker="e")
    cleanup = powered_process(tmp_path, pid=8888, marker="f")
    process = child if role == "powered_child" else cleanup
    context_sha256 = "c" * 64
    secret = bytes(range(32))
    capability = delegated_capability_sha256(role, context_sha256, secret)
    child_capability = capability if role == "powered_child" else "1" * 64
    cleanup_capability = capability if role == "cleanup_fallback" else "2" * 64
    attempt_path = (tmp_path / "attempt.json").resolve()
    child_authority_path = (tmp_path / "child-authority.json").resolve()
    cleanup_authority_path = (tmp_path / "cleanup-authority.json").resolve()
    attempt = {
        "schema": "test-powered-attempt/1",
        "context": {
            "task_id": "vq2-package2-powered-calibration-pilot",
            "session_id": "F00",
            "attempt_id": "F00-A01",
            "host": {"qpc_frequency_hz": 10_000_000},
            "wrapper_process": wrapper,
            "paths": {
                "child_authority": str(child_authority_path),
                "cleanup_authority": str(cleanup_authority_path),
            },
        },
        "context_sha256": context_sha256,
        "capabilities": {
            "lease_owner_sha256": "d" * 64,
            "child_sha256": child_capability,
            "cleanup_sha256": cleanup_capability,
        },
    }
    attempt_path.write_bytes(FakeDelegatedContract.canonical_json_file_bytes(attempt))
    ledger = tmp_path / "delegated-lease"
    ledger.mkdir()
    store = PoweredLeaseLedgerStore(
        ledger.resolve(),
        (tmp_path / "delegated-live-lease.json").resolve(),
        task_id=attempt["context"]["task_id"],
        session_id=attempt["context"]["session_id"],
        attempt_id=attempt["context"]["attempt_id"],
        attempt_envelope_sha256=FakeDelegatedContract.canonical_file_sha256(
            attempt
        ),
        attempt_context_sha256=context_sha256,
        wrapper_process=wrapper,
        qpc_frequency_hz=10_000_000,
    )
    store.append(
        event="acquired",
        owner_role="wrapper",
        owner_token_sha256="d" * 64,
        owner_process=wrapper,
        phase="lease_acquire",
        abandoned=False,
        observed_monotonic_ns=100,
    )
    _delegation_record, delegation_hash = store.append(
        event="phase",
        owner_role="wrapper",
        owner_token_sha256="d" * 64,
        owner_process=wrapper,
        child_process=child,
        cleanup_process=cleanup if role == "cleanup_fallback" else None,
        phase="fallback_spawn" if role == "cleanup_fallback" else "child_spawn",
        abandoned=False,
        observed_monotonic_ns=200,
    )
    store.append(
        event="heartbeat",
        owner_role="wrapper",
        owner_token_sha256="d" * 64,
        owner_process=wrapper,
        child_process=child,
        cleanup_process=cleanup if role == "cleanup_fallback" else None,
        phase=(
            "fallback_supervision"
            if role == "cleanup_fallback"
            else "child_supervision"
        ),
        abandoned=False,
        observed_monotonic_ns=1_000_000_100,
    )
    authority = {
        "role": role,
        "process": process,
        "wrapper_process": wrapper,
        "parent_handle": {
            "value": 52,
            "process": wrapper,
            "access": "synchronize_query_limited_information",
            "inherited": True,
        },
        "capability_sha256": capability,
        "lease_record_sha256": delegation_hash,
        "attempt_envelope_sha256": FakeDelegatedContract.canonical_file_sha256(
            attempt
        ),
        "attempt_context_sha256": context_sha256,
    }
    authority_path = (
        child_authority_path if role == "powered_child" else cleanup_authority_path
    )
    authority_path.write_bytes(
        FakeDelegatedContract.canonical_json_file_bytes(authority)
    )
    parent_state = {"signaled": False}
    clock = SteppingClock(1_100_000_000)
    injected_kernel = kernel or FakeKernel(wait_result=WAIT_ABANDONED)
    boundary = DelegatedPoweredLeaseBoundary(
        store,
        attempt_path,
        parent_signaled=lambda handle: handle == 52 and parent_state["signaled"],
        _kernel=injected_kernel,
        _clock_ns=clock,
        _process_guard=threading.Lock(),
        _contract=FakeDelegatedContract,
    )
    return {
        "attempt": attempt,
        "authority": authority,
        "authority_path": authority_path,
        "boundary": boundary,
        "child": child,
        "cleanup": cleanup,
        "clock": clock,
        "kernel": injected_kernel,
        "parent_state": parent_state,
        "secret": secret,
        "store": store,
        "wrapper": wrapper,
    }


def test_delegated_live_proof_reloads_files_and_accepts_wrapper_descendants(
    tmp_path,
):
    fixture = make_delegated_fixture(tmp_path)
    proof = fixture["boundary"].prove_live_delegation(
        attempt=fixture["attempt"],
        process_authority=fixture["authority"],
    )
    assert proof == DelegatedPoweredLeaseProof(
        owner_role="wrapper",
        generation=2,
        record_sha256=fixture["store"].record_hashes[-1],
        authority_valid=True,
    )
    assert fixture["authority"]["lease_record_sha256"] == fixture["store"].record_hashes[1]
    assert fixture["kernel"].calls == []

    changed = {
        **fixture["authority"],
        "lease_record_sha256": "9" * 64,
    }
    with pytest.raises(LiveLeaseEvidenceError, match="immutable bytes"):
        fixture["boundary"].prove_live_delegation(
            attempt=fixture["attempt"], process_authority=changed
        )
    assert fixture["kernel"].calls == []

    changed_attempt = {
        **fixture["attempt"],
        "capabilities": {
            **fixture["attempt"]["capabilities"],
            "lease_owner_sha256": "8" * 64,
        },
    }
    fixture["boundary"].attempt_envelope_path.write_bytes(
        FakeDelegatedContract.canonical_json_file_bytes(changed_attempt)
    )
    with pytest.raises(LiveLeaseEvidenceError, match="hash does not match"):
        fixture["boundary"].prove_live_delegation(
            attempt=changed_attempt,
            process_authority=fixture["authority"],
        )
    assert fixture["kernel"].calls == []


def test_delegated_takeover_hash_heartbeat_latest_proof_and_release(tmp_path):
    fixture = make_delegated_fixture(tmp_path)
    fixture["parent_state"]["signaled"] = True
    started = fixture["clock"].current
    proof = fixture["boundary"].take_over_abandoned(
        role_secret=memoryview(fixture["secret"]),
        attempt=fixture["attempt"],
        process_authority=fixture["authority"],
        deadline_monotonic_ns=started + 250_500_000,
    )
    expected_owner = hashlib.sha256(
        b"aigp-vq2-takeover-owner/1"
        + b"\x00"
        + bytes.fromhex(fixture["attempt"]["context_sha256"])
        + b"\x00cleanup-fallback-parent-death\x00"
        + fixture["secret"]
    ).hexdigest()
    assert derive_powered_takeover_owner_sha256(
        fixture["attempt"]["context_sha256"],
        "cleanup-fallback-parent-death",
        fixture["secret"],
    ) == expected_owner
    assert proof.owner_role == "cleanup-fallback-parent-death"
    assert fixture["boundary"].takeover_active
    assert fixture["boundary"].ledger_store.record_hashes[-1] != proof.record_sha256
    takeover_record = fixture["boundary"]._takeover_lease.ledger_store.records[-1]
    assert takeover_record["owner_token_sha256"] == expected_owner
    assert fixture["kernel"].calls[:2] == [
        ("create", LIVE_LEASE_MUTEX_NAME),
        ("wait", 1234, 250),
    ]

    fixture["clock"].current = (
        takeover_record["observed_monotonic_ns"]
        + PoweredLiveSimulatorLease.HEARTBEAT_PERIOD_NS
    )
    heartbeat_proof = fixture["boundary"].heartbeat_takeover(
        proof,
        phase="fallback_cleanup",
        deadline_monotonic_ns=fixture["clock"].current + 100_000_000,
    )
    assert heartbeat_proof.generation == proof.generation + 1
    assert heartbeat_proof.record_sha256 != proof.record_sha256
    with pytest.raises(LiveLeaseEvidenceError, match="stale"):
        fixture["boundary"].release_takeover(
            proof,
            deadline_monotonic_ns=fixture["clock"].current + 100_000_000,
        )

    fixture["clock"].current += PoweredLiveSimulatorLease.HEARTBEAT_PERIOD_NS
    assert fixture["boundary"].release_takeover(
        heartbeat_proof,
        deadline_monotonic_ns=fixture["clock"].current + 100_000_000,
    ) is True
    index = load_powered_live_lease_index(
        fixture["store"].final_index_path
    )
    assert validate_powered_live_lease_ledger(index) == index
    assert fixture["kernel"].calls[-2:] == [
        ("release", 1234),
        ("close", 1234),
    ]
    with pytest.raises(LiveLeaseError, match="single-attempt"):
        fixture["boundary"].take_over_abandoned(
            role_secret=memoryview(fixture["secret"]),
            attempt=fixture["attempt"],
            process_authority=fixture["authority"],
            deadline_monotonic_ns=fixture["clock"].current + 1_000_000,
        )


def test_delegated_takeover_wrong_thread_and_early_heartbeat_fail_closed(tmp_path):
    fixture = make_delegated_fixture(tmp_path, role="powered_child")
    fixture["parent_state"]["signaled"] = True
    proof = fixture["boundary"].take_over_abandoned(
        role_secret=memoryview(fixture["secret"]),
        attempt=fixture["attempt"],
        process_authority=fixture["authority"],
        deadline_monotonic_ns=fixture["clock"].current + 500_000_000,
    )
    errors = []

    def wrong_thread():
        try:
            fixture["boundary"].heartbeat_takeover(
                proof,
                phase="child_cleanup",
                deadline_monotonic_ns=fixture["clock"].current + 100_000_000,
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=wrong_thread)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(errors) == 1 and "owning thread" in str(errors[0])
    with pytest.raises(LiveLeaseEvidenceError, match="one-second cadence"):
        fixture["boundary"].heartbeat_takeover(
            proof,
            phase="child_cleanup",
            deadline_monotonic_ns=fixture["clock"].current + 100_000_000,
        )

    takeover_record = fixture["boundary"]._takeover_lease.ledger_store.records[-1]
    fixture["clock"].current = (
        takeover_record["observed_monotonic_ns"]
        + PoweredLiveSimulatorLease.HEARTBEAT_PERIOD_NS
    )
    refreshed = fixture["boundary"].heartbeat_takeover(
        proof,
        phase="child_cleanup",
        deadline_monotonic_ns=fixture["clock"].current + 100_000_000,
    )
    fixture["clock"].current += PoweredLiveSimulatorLease.HEARTBEAT_PERIOD_NS
    assert fixture["boundary"].release_takeover(
        refreshed,
        deadline_monotonic_ns=fixture["clock"].current + 100_000_000,
    )


def test_delegated_release_after_gap_releases_mutex_but_is_acceptance_invalid(
    tmp_path,
):
    fixture = make_delegated_fixture(tmp_path)
    fixture["parent_state"]["signaled"] = True
    proof = fixture["boundary"].take_over_abandoned(
        role_secret=memoryview(fixture["secret"]),
        attempt=fixture["attempt"],
        process_authority=fixture["authority"],
        deadline_monotonic_ns=fixture["clock"].current + 500_000_000,
    )
    takeover_record = fixture["boundary"]._takeover_lease.ledger_store.records[-1]
    fixture["clock"].current = (
        takeover_record["observed_monotonic_ns"]
        + PoweredLiveSimulatorLease.HEARTBEAT_MAX_GAP_NS
        + 1
    )
    with pytest.raises(LiveLeaseCleanupError, match="acceptance-invalid|cadence"):
        fixture["boundary"].release_takeover(
            proof,
            deadline_monotonic_ns=fixture["clock"].current + 100_000_000,
        )
    assert fixture["kernel"].calls[-2:] == [
        ("release", 1234),
        ("close", 1234),
    ]
    assert not fixture["boundary"].takeover_active
    index = load_powered_live_lease_index(fixture["store"].final_index_path)
    with pytest.raises(LiveLeaseEvidenceError, match="maximum gap"):
        validate_powered_live_lease_ledger(index)


def test_delegated_takeover_rechecks_predecessor_after_abandoned_wait(tmp_path):
    callback = {"value": None}
    kernel = CallbackKernel(
        wait_result=WAIT_ABANDONED,
        on_wait=lambda: callback["value"](),
    )
    fixture = make_delegated_fixture(tmp_path, kernel=kernel)
    fixture["parent_state"]["signaled"] = True

    def mutate_predecessor():
        fixture["store"].append(
            event="heartbeat",
            owner_role="wrapper",
            owner_token_sha256="d" * 64,
            owner_process=fixture["wrapper"],
            child_process=fixture["child"],
            cleanup_process=fixture["cleanup"],
            phase="fallback_supervision",
            abandoned=False,
            observed_monotonic_ns=1_050_000_100,
        )

    callback["value"] = mutate_predecessor
    with pytest.raises(LiveLeaseEvidenceError, match="takeover authority proof"):
        fixture["boundary"].take_over_abandoned(
            role_secret=memoryview(fixture["secret"]),
            attempt=fixture["attempt"],
            process_authority=fixture["authority"],
            deadline_monotonic_ns=fixture["clock"].current + 500_000_000,
        )
    assert kernel.calls[-2:] == [("release", 1234), ("close", 1234)]
    assert all(row["event"] != "takeover" for row in fixture["store"].records)


def test_delegated_wrong_secret_or_live_parent_never_waits_on_mutex(tmp_path):
    wrong_secret = make_delegated_fixture(tmp_path / "wrong-secret")
    wrong_secret["parent_state"]["signaled"] = True
    with pytest.raises(LiveLeaseEvidenceError, match="role secret"):
        wrong_secret["boundary"].take_over_abandoned(
            role_secret=memoryview(b"x" * 32),
            attempt=wrong_secret["attempt"],
            process_authority=wrong_secret["authority"],
            deadline_monotonic_ns=wrong_secret["clock"].current + 500_000_000,
        )
    assert wrong_secret["kernel"].calls == []

    live_parent = make_delegated_fixture(tmp_path / "live-parent")
    with pytest.raises(LiveLeaseEvidenceError, match="signaled"):
        live_parent["boundary"].take_over_abandoned(
            role_secret=memoryview(live_parent["secret"]),
            attempt=live_parent["attempt"],
            process_authority=live_parent["authority"],
            deadline_monotonic_ns=live_parent["clock"].current + 500_000_000,
        )
    assert live_parent["kernel"].calls == []


def test_final_ledger_readback_rejects_shape_valid_synthetic_release(tmp_path):
    ledger = tmp_path / "synthetic-ledger"
    ledger.mkdir()
    path = (ledger / "generation-000000.json").resolve()
    wrapper = powered_process(tmp_path)
    synthetic = {
        "schema": POWERED_LIVE_LEASE_EVIDENCE_SCHEMA,
        "mutex_name": LIVE_LEASE_MUTEX_NAME,
        "attempt_id": "F00-A01",
        "attempt_envelope_sha256": "b" * 64,
        "attempt_context_sha256": "c" * 64,
        "generation": 0,
        "predecessor_sha256": None,
        "event": "released",
        "abandoned": False,
        "owner_role": "wrapper",
        "owner_token_sha256": "d" * 64,
        "wrapper_process": wrapper,
        "owner_process": wrapper,
        "child_process": None,
        "cleanup_process": None,
        "host_clock_id": "host-perf-counter",
        "qpc_frequency_hz": 10_000_000,
        "observed_monotonic_ns": 100,
        "phase": "lease_release_and_verify",
        "orphaned_pending": None,
        "release_proved": True,
    }
    payload = (
        json.dumps(synthetic, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    index = {
        "schema": POWERED_LIVE_LEASE_LEDGER_SCHEMA,
        "task_id": "vq2-package2-powered-calibration-pilot",
        "session_id": "F00",
        "attempt_id": "F00-A01",
        "attempt_envelope_sha256": "b" * 64,
        "records": [
            {
                "generation": 0,
                "path": str(path),
                "sha256": digest,
                "event": "released",
            }
        ],
        "orphaned_pending_files": [],
        "final_generation": 0,
        "final_record_sha256": digest,
        "release_proved": True,
    }
    assert validate_powered_live_lease_index(index) == index
    with pytest.raises(LiveLeaseEvidenceError, match="generation zero"):
        validate_powered_live_lease_ledger(index)


def test_final_ledger_rejects_stale_wrapper_gap_before_safety_takeover(tmp_path):
    store = make_powered_store(tmp_path)
    wrapper = powered_process(tmp_path)
    child = powered_process(tmp_path, pid=7777, marker="e")
    store.append(
        event="acquired",
        owner_role="wrapper",
        owner_token_sha256="d" * 64,
        owner_process=wrapper,
        phase="lease_acquire",
        abandoned=False,
        observed_monotonic_ns=100,
    )
    store.append(
        event="heartbeat",
        owner_role="wrapper",
        owner_token_sha256="d" * 64,
        owner_process=wrapper,
        child_process=child,
        phase="child_supervision",
        abandoned=False,
        observed_monotonic_ns=1_000_000_100,
    )
    store.append(
        event="takeover",
        owner_role="powered-child-parent-death",
        owner_token_sha256="f" * 64,
        owner_process=child,
        child_process=child,
        phase="child_cleanup",
        abandoned=True,
        observed_monotonic_ns=20_000_000_100,
    )
    store.append(
        event="release_intent",
        owner_role="powered-child-parent-death",
        owner_token_sha256="f" * 64,
        owner_process=child,
        child_process=child,
        phase="lease_release_and_verify",
        abandoned=True,
        observed_monotonic_ns=20_100_000_100,
    )
    store.append(
        event="released",
        owner_role="powered-child-parent-death",
        owner_token_sha256="f" * 64,
        owner_process=child,
        child_process=child,
        phase="lease_release_and_verify",
        abandoned=True,
        release_proved=True,
        observed_monotonic_ns=20_200_000_100,
    )
    index, _digest = store.seal_released_index()
    with pytest.raises(LiveLeaseEvidenceError, match="prior owner heartbeat gap"):
        validate_powered_live_lease_ledger(index)


def test_final_ledger_rejects_indexed_generation_from_another_parent(tmp_path):
    store = make_powered_store(tmp_path)
    append_wrapper_record(store, tmp_path, "acquired", "lease_acquire")
    append_wrapper_record(
        store, tmp_path, "release_intent", "lease_release_and_verify"
    )
    append_wrapper_record(
        store,
        tmp_path,
        "released",
        "lease_release_and_verify",
        release=True,
    )
    index, _digest = store.seal_released_index()
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    copied = (alternate / "generation-000001.json").resolve()
    copied.write_bytes(Path(index["records"][1]["path"]).read_bytes())
    changed = {
        **index,
        "records": [dict(item) for item in index["records"]],
    }
    changed["records"][1]["path"] = str(copied)
    assert validate_powered_live_lease_index(changed) == changed
    with pytest.raises(LiveLeaseEvidenceError, match="share one directory"):
        validate_powered_live_lease_ledger(changed)
