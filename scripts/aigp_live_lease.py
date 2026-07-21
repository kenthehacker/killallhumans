"""Fail-closed Windows lease for passive FlightSim probes.

This module only owns a named mutex and a private evidence envelope.  It does
not inspect, launch, or contact FlightSim and it does not bind network ports.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


LIVE_LEASE_MUTEX_NAME = r"Global\AIGP-FlightSim-LiveLease-v1"
LIVE_LEASE_EVIDENCE_SCHEMA = "aigp-vq2-live-lease-evidence/1"
POWERED_LIVE_LEASE_EVIDENCE_SCHEMA = "aigp-vq2-live-lease-evidence/2"
POWERED_LIVE_LEASE_LEDGER_SCHEMA = "aigp-vq2-live-lease-ledger/1"

WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF

_MAX_EVIDENCE_BYTES = 64 * 1024
_OWNER_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PHASE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_ATTEMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROCESS_LEASE_GUARD = threading.Lock()
_UNCHANGED = object()
_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "mutex_name",
        "owner_token",
        "wrapper_pid",
        "acquired_wall_time_ns",
        "heartbeat_wall_time_ns",
        "phase",
        "child_pid",
        "released_wall_time_ns",
    }
)

_POWERED_RECORD_KEYS = frozenset(
    {
        "schema",
        "mutex_name",
        "attempt_id",
        "attempt_envelope_sha256",
        "attempt_context_sha256",
        "generation",
        "predecessor_sha256",
        "event",
        "abandoned",
        "owner_role",
        "owner_token_sha256",
        "wrapper_process",
        "owner_process",
        "child_process",
        "cleanup_process",
        "host_clock_id",
        "qpc_frequency_hz",
        "observed_monotonic_ns",
        "phase",
        "orphaned_pending",
        "release_proved",
    }
)
_POWERED_INDEX_KEYS = frozenset(
    {
        "schema",
        "task_id",
        "session_id",
        "attempt_id",
        "attempt_envelope_sha256",
        "records",
        "orphaned_pending_files",
        "final_generation",
        "final_record_sha256",
        "release_proved",
    }
)
_PROCESS_IDENTITY_KEYS = frozenset(
    {
        "pid",
        "creation_filetime_100ns",
        "windows_session_id",
        "image_path",
        "image_sha256",
        "argv_sha256",
    }
)
_ORPHANED_PENDING_KEYS = frozenset(
    {"path", "size_bytes", "sha256", "owner_role"}
)
_POWERED_OWNER_ROLES = frozenset(
    {
        "wrapper",
        "powered-child-parent-death",
        "cleanup-fallback-parent-death",
    }
)
_POWERED_EVENTS = frozenset(
    {"acquired", "heartbeat", "phase", "takeover", "release_intent", "released"}
)
_POWERED_PHASES = frozenset(
    {
        "lease_acquire",
        "launcher_return",
        "topology_and_training_attestation",
        "prechild_identity_and_ports",
        "child_spawn",
        "child_supervision",
        "child_cleanup",
        "child_exit_proof",
        "fallback_spawn",
        "fallback_supervision",
        "fallback_cleanup",
        "postcheck_identity_process_ports",
        "lease_release_and_verify",
    }
)
_POWERED_RECORD_NAME = re.compile(r"^generation-(?P<generation>[0-9]{6})\.json$")
_POWERED_PENDING_NAME = re.compile(
    r"^pending-generation-(?P<generation>[0-9]{6})-(?P<role>"
    r"wrapper|powered-child-parent-death|cleanup-fallback-parent-death)\.json$"
)
_MAX_POWERED_GENERATION = 4095
_POWERED_HEARTBEAT_PERIOD_NS = 1_000_000_000
_POWERED_HEARTBEAT_MAX_GAP_NS = 1_500_000_000
_TAKEOVER_OWNER_DOMAIN = b"aigp-vq2-takeover-owner/1"
_DELEGATED_ROLE_TO_OWNER = {
    "powered_child": "powered-child-parent-death",
    "cleanup_fallback": "cleanup-fallback-parent-death",
}
_DELEGATED_ROLE_TO_CAPABILITY = {
    "powered_child": ("child_sha256", b"aigp-vq2-powered-child/1"),
    "cleanup_fallback": ("cleanup_sha256", b"aigp-vq2-powered-cleanup/1"),
}


class LiveLeaseError(RuntimeError):
    """Base class for a fail-closed live-lease failure."""


class LiveLeaseBusyError(LiveLeaseError):
    """The named mutex is currently owned by another process or thread."""


class LiveLeaseAbandonedError(LiveLeaseError):
    """A previous owner abandoned the mutex; the current probe must stop."""


class LiveLeaseUnavailableError(LiveLeaseError):
    """The mutex could not be created, inspected, or verified."""


class LiveLeaseEvidenceError(LiveLeaseError):
    """The private lease evidence could not be validated or published."""


class LiveLeaseCleanupError(LiveLeaseError):
    """Mutex release, handle closure, or final evidence publication failed."""


@dataclass(frozen=True)
class DelegatedPoweredLeaseProof:
    """Immutable authority proof returned to a powered child or fallback."""

    owner_role: str
    generation: int
    record_sha256: str
    authority_valid: bool
    takeover_completed_monotonic_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if self.owner_role not in _POWERED_OWNER_ROLES:
            raise ValueError("delegated powered lease owner role is invalid")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("delegated powered lease generation is invalid")
        if (
            type(self.record_sha256) is not str
            or _SHA256.fullmatch(self.record_sha256) is None
        ):
            raise ValueError("delegated powered lease record hash is invalid")
        if type(self.authority_valid) is not bool:
            raise TypeError("delegated powered lease authority state must be boolean")
        completed = self.takeover_completed_monotonic_ns
        if completed is not None and (type(completed) is not int or completed < 0):
            raise ValueError("delegated powered takeover completion is invalid")
        if self.owner_role == "wrapper" and completed is not None:
            raise ValueError("live wrapper delegation cannot have takeover completion")
        if self.owner_role != "wrapper" and self.authority_valid and completed is None:
            raise ValueError("valid takeover authority requires completion evidence")


def _exact_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise LiveLeaseEvidenceError(f"{label} must be an exact positive integer")
    return value


def _validate_phase(value: Any) -> str:
    if type(value) is not str or _PHASE.fullmatch(value) is None:
        raise LiveLeaseEvidenceError(
            "lease phase must match ^[a-z][a-z0-9._-]{0,63}$"
        )
    return value


def validate_live_lease_evidence(value: Any) -> dict[str, Any]:
    """Validate and copy one exact live-lease evidence envelope."""

    if type(value) is not dict or set(value) != _EVIDENCE_KEYS:
        raise LiveLeaseEvidenceError(
            "live-lease evidence has missing or unknown fields"
        )
    if value["schema"] != LIVE_LEASE_EVIDENCE_SCHEMA:
        raise LiveLeaseEvidenceError("live-lease evidence schema is invalid")
    if value["mutex_name"] != LIVE_LEASE_MUTEX_NAME:
        raise LiveLeaseEvidenceError("live-lease evidence mutex name is invalid")
    owner_token = value["owner_token"]
    if type(owner_token) is not str or _OWNER_TOKEN.fullmatch(owner_token) is None:
        raise LiveLeaseEvidenceError(
            "live-lease owner token must be 64 lowercase hexadecimal characters"
        )
    _exact_positive_int(value["wrapper_pid"], "wrapper PID")
    acquired = _exact_positive_int(
        value["acquired_wall_time_ns"], "acquisition timestamp"
    )
    heartbeat = _exact_positive_int(
        value["heartbeat_wall_time_ns"], "heartbeat timestamp"
    )
    if heartbeat < acquired:
        raise LiveLeaseEvidenceError(
            "heartbeat timestamp cannot precede acquisition"
        )
    _validate_phase(value["phase"])
    child_pid = value["child_pid"]
    if child_pid is not None:
        _exact_positive_int(child_pid, "child PID")
    released = value["released_wall_time_ns"]
    if released is not None:
        _exact_positive_int(released, "release timestamp")
        if released < heartbeat:
            raise LiveLeaseEvidenceError(
                "release timestamp cannot precede the last heartbeat"
            )
        if value["phase"] != "released":
            raise LiveLeaseEvidenceError(
                "released evidence must use exact released phase"
            )
    elif value["phase"] == "released":
        raise LiveLeaseEvidenceError(
            "released phase requires a clean release timestamp"
        )
    return dict(value)


def _strict_json_loads(payload: bytes) -> Any:
    if len(payload) > _MAX_EVIDENCE_BYTES:
        raise LiveLeaseEvidenceError("live-lease evidence exceeds size limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LiveLeaseEvidenceError(
                    f"duplicate live-lease evidence key: {key}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                LiveLeaseEvidenceError(
                    f"non-standard live-lease JSON constant: {value}"
                )
            ),
        )
    except UnicodeDecodeError as exc:
        raise LiveLeaseEvidenceError("live-lease evidence must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise LiveLeaseEvidenceError("live-lease evidence is invalid JSON") from exc


def load_live_lease_evidence(path: Path | str) -> dict[str, Any]:
    """Read and validate one bounded regular evidence file."""

    target = Path(path)
    try:
        info = target.lstat()
    except OSError as exc:
        raise LiveLeaseEvidenceError("live-lease evidence is unavailable") from exc
    if target.is_symlink() or not target.is_file() or info.st_size > _MAX_EVIDENCE_BYTES:
        raise LiveLeaseEvidenceError(
            "live-lease evidence must be a bounded regular file"
        )
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise LiveLeaseEvidenceError("live-lease evidence could not be read") from exc
    return validate_live_lease_evidence(_strict_json_loads(payload))


def _exact_nonnegative_int(
    value: Any,
    label: str,
    *,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int or value < 0:
        raise LiveLeaseEvidenceError(
            f"{label} must be an exact non-negative integer"
        )
    if maximum is not None and value > maximum:
        raise LiveLeaseEvidenceError(f"{label} must be <= {maximum}")
    return value


def _exact_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise LiveLeaseEvidenceError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def derive_powered_takeover_owner_sha256(
    attempt_context_sha256: str,
    owner_role: str,
    role_secret: bytes | bytearray | memoryview,
) -> str:
    """Derive the exact context/role-bound owner hash for one takeover."""

    context = bytes.fromhex(
        _exact_sha256(attempt_context_sha256, "attempt_context_sha256")
    )
    if owner_role not in {
        "powered-child-parent-death",
        "cleanup-fallback-parent-death",
    }:
        raise LiveLeaseEvidenceError("takeover owner role is invalid")
    if not isinstance(role_secret, (bytes, bytearray, memoryview)):
        raise TypeError("takeover role secret must be bytes-like")
    secret = bytes(role_secret)
    if len(secret) != 32:
        raise LiveLeaseEvidenceError(
            "takeover role secret must contain exactly 32 bytes"
        )
    return hashlib.sha256(
        _TAKEOVER_OWNER_DOMAIN
        + b"\x00"
        + context
        + b"\x00"
        + owner_role.encode("utf-8")
        + b"\x00"
        + secret
    ).hexdigest()


def _exact_identifier(value: Any, label: str) -> str:
    if type(value) is not str or _ATTEMPT_ID.fullmatch(value) is None:
        raise LiveLeaseEvidenceError(f"{label} must be a bounded identifier")
    return value


def _validate_absolute_path(value: Any, label: str) -> str:
    if type(value) is not str or value == "":
        raise LiveLeaseEvidenceError(f"{label} must be an absolute path")
    if not Path(value).is_absolute():
        raise LiveLeaseEvidenceError(f"{label} must be an absolute path")
    return value


def _validate_process_identity(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _PROCESS_IDENTITY_KEYS:
        raise LiveLeaseEvidenceError(
            f"{label} must have the exact process-identity fields"
        )
    result = dict(value)
    _exact_positive_int(result["pid"], f"{label}.pid")
    _exact_positive_int(
        result["creation_filetime_100ns"],
        f"{label}.creation_filetime_100ns",
    )
    _exact_nonnegative_int(
        result["windows_session_id"], f"{label}.windows_session_id"
    )
    _validate_absolute_path(result["image_path"], f"{label}.image_path")
    _exact_sha256(result["image_sha256"], f"{label}.image_sha256")
    _exact_sha256(result["argv_sha256"], f"{label}.argv_sha256")
    return result


def _validate_optional_process_identity(
    value: Any, label: str
) -> Optional[dict[str, Any]]:
    return None if value is None else _validate_process_identity(value, label)


def _validate_orphaned_pending(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if type(value) is not dict or set(value) != _ORPHANED_PENDING_KEYS:
        raise LiveLeaseEvidenceError(
            "orphaned_pending must have exact pending-artifact fields"
        )
    result = dict(value)
    _validate_absolute_path(result["path"], "orphaned_pending.path")
    _exact_nonnegative_int(result["size_bytes"], "orphaned_pending.size_bytes")
    _exact_sha256(result["sha256"], "orphaned_pending.sha256")
    role = result["owner_role"]
    if role not in _POWERED_OWNER_ROLES:
        raise LiveLeaseEvidenceError("orphaned_pending.owner_role is invalid")
    return result


def validate_powered_live_lease_record(value: Any) -> dict[str, Any]:
    """Validate and defensively copy one append-only powered lease row."""

    if type(value) is not dict or set(value) != _POWERED_RECORD_KEYS:
        raise LiveLeaseEvidenceError(
            "powered live-lease record has missing or unknown fields"
        )
    result = dict(value)
    if result["schema"] != POWERED_LIVE_LEASE_EVIDENCE_SCHEMA:
        raise LiveLeaseEvidenceError("powered live-lease schema is invalid")
    if result["mutex_name"] != LIVE_LEASE_MUTEX_NAME:
        raise LiveLeaseEvidenceError("powered live-lease mutex name is invalid")
    _exact_identifier(result["attempt_id"], "attempt_id")
    _exact_sha256(
        result["attempt_envelope_sha256"], "attempt_envelope_sha256"
    )
    _exact_sha256(result["attempt_context_sha256"], "attempt_context_sha256")
    generation = _exact_nonnegative_int(
        result["generation"],
        "generation",
        maximum=_MAX_POWERED_GENERATION,
    )
    predecessor = result["predecessor_sha256"]
    if generation == 0:
        if predecessor is not None:
            raise LiveLeaseEvidenceError(
                "generation zero predecessor_sha256 must be null"
            )
    else:
        _exact_sha256(predecessor, "predecessor_sha256")
    event = result["event"]
    if event not in _POWERED_EVENTS:
        raise LiveLeaseEvidenceError("powered live-lease event is invalid")
    if type(result["abandoned"]) is not bool:
        raise LiveLeaseEvidenceError("abandoned must be an exact boolean")
    role = result["owner_role"]
    if role not in _POWERED_OWNER_ROLES:
        raise LiveLeaseEvidenceError("powered live-lease owner_role is invalid")
    if generation == 0 and (event != "acquired" or role != "wrapper"):
        raise LiveLeaseEvidenceError(
            "generation zero must be a wrapper acquired record"
        )
    if generation == 0 and result["abandoned"]:
        raise LiveLeaseEvidenceError(
            "generation zero cannot be abandoned"
        )
    if generation > 0 and event == "acquired":
        raise LiveLeaseEvidenceError(
            "acquired is permitted only at generation zero"
        )
    if event == "takeover":
        if not result["abandoned"] or role == "wrapper":
            raise LiveLeaseEvidenceError(
                "takeover requires abandoned non-wrapper ownership"
            )
    elif result["orphaned_pending"] is not None:
        raise LiveLeaseEvidenceError(
            "orphaned_pending is permitted only on a takeover record"
        )
    _exact_sha256(result["owner_token_sha256"], "owner_token_sha256")
    wrapper = _validate_process_identity(result["wrapper_process"], "wrapper_process")
    owner = _validate_process_identity(result["owner_process"], "owner_process")
    child = _validate_optional_process_identity(result["child_process"], "child_process")
    cleanup = _validate_optional_process_identity(
        result["cleanup_process"], "cleanup_process"
    )
    if role == "wrapper" and owner != wrapper:
        raise LiveLeaseEvidenceError("wrapper role must own through wrapper_process")
    if role == "powered-child-parent-death" and owner != child:
        raise LiveLeaseEvidenceError("child takeover owner must equal child_process")
    if role == "cleanup-fallback-parent-death" and owner != cleanup:
        raise LiveLeaseEvidenceError(
            "fallback takeover owner must equal cleanup_process"
        )
    result["wrapper_process"] = wrapper
    result["owner_process"] = owner
    result["child_process"] = child
    result["cleanup_process"] = cleanup
    if result["host_clock_id"] != "host-perf-counter":
        raise LiveLeaseEvidenceError(
            "powered live-lease host_clock_id must be host-perf-counter"
        )
    _exact_positive_int(result["qpc_frequency_hz"], "qpc_frequency_hz")
    _exact_nonnegative_int(
        result["observed_monotonic_ns"], "observed_monotonic_ns"
    )
    if result["phase"] not in _POWERED_PHASES:
        raise LiveLeaseEvidenceError("powered live-lease phase is invalid")
    result["orphaned_pending"] = _validate_orphaned_pending(
        result["orphaned_pending"]
    )
    if type(result["release_proved"]) is not bool:
        raise LiveLeaseEvidenceError("release_proved must be an exact boolean")
    if result["release_proved"] != (event == "released"):
        raise LiveLeaseEvidenceError(
            "release_proved must be true exactly for released records"
        )
    return result


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LiveLeaseEvidenceError(
            "powered lease evidence is not canonical JSON data"
        ) from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_bounded_regular_json(path: Path, *, label: str) -> tuple[bytes, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise LiveLeaseEvidenceError(f"{label} is unavailable") from exc
    if path.is_symlink() or not path.is_file() or info.st_size > _MAX_EVIDENCE_BYTES:
        raise LiveLeaseEvidenceError(f"{label} must be a bounded regular file")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise LiveLeaseEvidenceError(f"{label} could not be read") from exc
    return payload, _strict_json_loads(payload)


def load_powered_live_lease_record(path: Path | str) -> dict[str, Any]:
    payload, value = _load_bounded_regular_json(
        Path(path), label="powered live-lease record"
    )
    result = validate_powered_live_lease_record(value)
    if payload != _canonical_json_line(result):
        raise LiveLeaseEvidenceError(
            "powered live-lease record bytes are not canonical"
        )
    return result


def validate_powered_live_lease_index(value: Any) -> dict[str, Any]:
    """Validate the immutable final index for a released powered lease."""

    if type(value) is not dict or set(value) != _POWERED_INDEX_KEYS:
        raise LiveLeaseEvidenceError(
            "powered live-lease index has missing or unknown fields"
        )
    result = dict(value)
    if result["schema"] != POWERED_LIVE_LEASE_LEDGER_SCHEMA:
        raise LiveLeaseEvidenceError("powered live-lease index schema is invalid")
    _exact_identifier(result["task_id"], "task_id")
    _exact_identifier(result["session_id"], "session_id")
    _exact_identifier(result["attempt_id"], "attempt_id")
    _exact_sha256(
        result["attempt_envelope_sha256"], "attempt_envelope_sha256"
    )
    records = result["records"]
    if type(records) is not list or not records:
        raise LiveLeaseEvidenceError("powered lease index records must be nonempty")
    copied_records: list[dict[str, Any]] = []
    expected_generation = 0
    record_keys = {"generation", "path", "sha256", "event"}
    for entry in records:
        if type(entry) is not dict or set(entry) != record_keys:
            raise LiveLeaseEvidenceError(
                "powered lease index record entry fields are invalid"
            )
        copied = dict(entry)
        generation = _exact_nonnegative_int(
            copied["generation"],
            "records.generation",
            maximum=_MAX_POWERED_GENERATION,
        )
        if generation != expected_generation:
            raise LiveLeaseEvidenceError(
                "powered lease index generations must be contiguous from zero"
            )
        _validate_absolute_path(copied["path"], "records.path")
        _exact_sha256(copied["sha256"], "records.sha256")
        if copied["event"] not in _POWERED_EVENTS:
            raise LiveLeaseEvidenceError("powered lease index event is invalid")
        copied_records.append(copied)
        expected_generation += 1
    orphaned = result["orphaned_pending_files"]
    if type(orphaned) is not list:
        raise LiveLeaseEvidenceError(
            "orphaned_pending_files must be an exact array"
        )
    copied_orphaned = [_validate_orphaned_pending(item) for item in orphaned]
    if any(item is None for item in copied_orphaned):
        raise LiveLeaseEvidenceError("orphaned pending index entries cannot be null")
    orphan_values = [item for item in copied_orphaned if item is not None]
    if orphan_values != sorted(orphan_values, key=lambda item: item["path"]):
        raise LiveLeaseEvidenceError(
            "orphaned pending index entries must be sorted by path"
        )
    paths = [item["path"] for item in orphan_values]
    if len(paths) != len(set(paths)):
        raise LiveLeaseEvidenceError(
            "orphaned pending index paths must be unique"
        )
    final_generation = _exact_nonnegative_int(
        result["final_generation"],
        "final_generation",
        maximum=_MAX_POWERED_GENERATION,
    )
    if final_generation != len(copied_records) - 1:
        raise LiveLeaseEvidenceError(
            "final_generation must match the final indexed record"
        )
    _exact_sha256(result["final_record_sha256"], "final_record_sha256")
    if result["final_record_sha256"] != copied_records[-1]["sha256"]:
        raise LiveLeaseEvidenceError(
            "final_record_sha256 must match the final indexed record"
        )
    if type(result["release_proved"]) is not bool:
        raise LiveLeaseEvidenceError("index release_proved must be a boolean")
    if not result["release_proved"] or copied_records[-1]["event"] != "released":
        raise LiveLeaseEvidenceError(
            "powered lease index requires a final released record"
        )
    result["records"] = copied_records
    result["orphaned_pending_files"] = orphan_values
    return result


def load_powered_live_lease_index(path: Path | str) -> dict[str, Any]:
    payload, value = _load_bounded_regular_json(
        Path(path), label="powered live-lease index"
    )
    result = validate_powered_live_lease_index(value)
    if payload != _canonical_json_line(result):
        raise LiveLeaseEvidenceError(
            "powered live-lease index bytes are not canonical"
        )
    return result


def validate_powered_live_lease_ledger(
    value: Any,
    *,
    expected_ledger_directory: Path | str | None = None,
    expected_attempt_context_sha256: Optional[str] = None,
    expected_wrapper_process: Optional[Mapping[str, Any]] = None,
    expected_initial_owner_token_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Read back and validate every immutable row referenced by a final index.

    The returned value has the same shape as ``validate_powered_live_lease_index``.
    Unlike that shape-only validator, this acceptance boundary reads every
    canonical generation, checks its hash and full transition chain, rejects
    preserved orphan files, and proves the powered heartbeat maximum gap through
    ``release_intent``.
    """

    index = validate_powered_live_lease_index(value)
    expected_directory: Optional[Path] = None
    if expected_ledger_directory is not None:
        expected_directory = Path(expected_ledger_directory)
        if not expected_directory.is_absolute():
            raise LiveLeaseEvidenceError(
                "expected powered lease ledger directory must be absolute"
            )
        try:
            expected_directory = expected_directory.resolve(strict=True)
        except OSError as exc:
            raise LiveLeaseEvidenceError(
                "expected powered lease ledger directory is unavailable"
            ) from exc
        if expected_directory.is_symlink() or not expected_directory.is_dir():
            raise LiveLeaseEvidenceError(
                "expected powered lease ledger directory is invalid"
            )
    expected_context = (
        None
        if expected_attempt_context_sha256 is None
        else _exact_sha256(
            expected_attempt_context_sha256,
            "expected_attempt_context_sha256",
        )
    )
    expected_wrapper = (
        None
        if expected_wrapper_process is None
        else _validate_process_identity(
            dict(expected_wrapper_process), "expected_wrapper_process"
        )
    )
    expected_initial_owner = (
        None
        if expected_initial_owner_token_sha256 is None
        else _exact_sha256(
            expected_initial_owner_token_sha256,
            "expected_initial_owner_token_sha256",
        )
    )
    if index["orphaned_pending_files"]:
        raise LiveLeaseEvidenceError(
            "powered lease final ledger cannot contain orphaned pending files"
        )

    records: list[dict[str, Any]] = []
    record_hashes: list[str] = []
    ledger_parent: Optional[Path] = None
    previous_hash: Optional[str] = None
    first: Optional[dict[str, Any]] = None
    child_process: Optional[dict[str, Any]] = None
    cleanup_process: Optional[dict[str, Any]] = None
    last_heartbeat_ns: Optional[int] = None

    for entry in index["records"]:
        generation = entry["generation"]
        path = Path(entry["path"])
        if path.name != f"generation-{generation:06d}.json":
            raise LiveLeaseEvidenceError(
                "powered lease index path does not match its generation"
            )
        try:
            canonical_path = path.resolve(strict=True)
        except OSError as exc:
            raise LiveLeaseEvidenceError(
                "powered lease indexed generation is unavailable"
            ) from exc
        if os.path.normcase(str(canonical_path)) != os.path.normcase(str(path)):
            raise LiveLeaseEvidenceError(
                "powered lease indexed generation path is not canonical"
            )
        if ledger_parent is None:
            ledger_parent = path.parent
        elif path.parent != ledger_parent:
            raise LiveLeaseEvidenceError(
                "powered lease indexed generations must share one directory"
            )

        payload, raw = _load_bounded_regular_json(
            path, label="powered live-lease indexed generation"
        )
        record = validate_powered_live_lease_record(raw)
        if payload != _canonical_json_line(record):
            raise LiveLeaseEvidenceError(
                "powered lease indexed generation bytes are not canonical"
            )
        digest = _sha256_bytes(payload)
        if digest != entry["sha256"]:
            raise LiveLeaseEvidenceError(
                "powered lease indexed generation hash mismatched"
            )
        if record["generation"] != generation or record["event"] != entry["event"]:
            raise LiveLeaseEvidenceError(
                "powered lease indexed generation identity mismatched"
            )
        if record["attempt_id"] != index["attempt_id"]:
            raise LiveLeaseEvidenceError(
                "powered lease indexed generation attempt_id mismatched"
            )
        if record["attempt_envelope_sha256"] != index["attempt_envelope_sha256"]:
            raise LiveLeaseEvidenceError(
                "powered lease indexed generation attempt envelope mismatched"
            )
        if record["predecessor_sha256"] != previous_hash:
            raise LiveLeaseEvidenceError(
                "powered lease indexed predecessor hash mismatched"
            )
        if record["orphaned_pending"] is not None:
            raise LiveLeaseEvidenceError(
                "powered lease accepted final ledger cannot bind an orphan"
            )

        if first is None:
            first = record
            if expected_context is not None and (
                record["attempt_context_sha256"] != expected_context
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease initial context does not match expected context"
                )
            if expected_wrapper is not None and (
                record["wrapper_process"] != expected_wrapper
                or record["owner_process"] != expected_wrapper
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease initial wrapper owner does not match expected"
                )
            if expected_initial_owner is not None and (
                record["owner_token_sha256"] != expected_initial_owner
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease initial owner hash does not match expected"
                )
        else:
            previous = records[-1]
            for field in (
                "mutex_name",
                "attempt_id",
                "attempt_envelope_sha256",
                "attempt_context_sha256",
                "wrapper_process",
                "host_clock_id",
                "qpc_frequency_hz",
            ):
                if record[field] != first[field]:
                    raise LiveLeaseEvidenceError(
                        f"powered lease indexed generation changed {field}"
                    )
            if record["observed_monotonic_ns"] < previous["observed_monotonic_ns"]:
                raise LiveLeaseEvidenceError(
                    "powered lease indexed occurrence time regressed"
                )
            if previous["event"] == "released":
                raise LiveLeaseEvidenceError(
                    "powered lease indexed chain continued after release"
                )
            if previous["abandoned"] and not record["abandoned"]:
                raise LiveLeaseEvidenceError(
                    "powered lease indexed abandoned state cleared"
                )
            if (
                not previous["abandoned"]
                and record["abandoned"]
                and record["event"] != "takeover"
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease indexed abandoned state lacks takeover"
                )
            if record["event"] == "takeover":
                if previous["owner_role"] != "wrapper" or previous["abandoned"]:
                    raise LiveLeaseEvidenceError(
                        "powered lease indexed takeover predecessor is invalid"
                    )
                if record["owner_token_sha256"] == previous["owner_token_sha256"]:
                    raise LiveLeaseEvidenceError(
                        "powered lease indexed takeover did not change owner hash"
                    )
            elif (
                record["owner_role"] != previous["owner_role"]
                or record["owner_token_sha256"]
                != previous["owner_token_sha256"]
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease indexed owner changed without takeover"
                )
            if record["event"] == "released" and previous["event"] != "release_intent":
                raise LiveLeaseEvidenceError(
                    "powered lease indexed release lacks release_intent"
                )
            if (
                previous["event"] == "release_intent"
                and record["event"] not in {"released", "takeover"}
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease indexed release_intent successor is invalid"
                )

        if child_process is not None and record["child_process"] != child_process:
            raise LiveLeaseEvidenceError(
                "powered lease indexed child process binding changed"
            )
        if cleanup_process is not None and record["cleanup_process"] != cleanup_process:
            raise LiveLeaseEvidenceError(
                "powered lease indexed cleanup process binding changed"
            )
        child_process = record["child_process"] or child_process
        cleanup_process = record["cleanup_process"] or cleanup_process

        event = record["event"]
        observed = record["observed_monotonic_ns"]
        if event == "acquired":
            last_heartbeat_ns = observed
        elif event == "takeover":
            if (
                last_heartbeat_ns is None
                or observed - last_heartbeat_ns
                > _POWERED_HEARTBEAT_MAX_GAP_NS
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease takeover exceeded the prior owner heartbeat gap"
                )
            last_heartbeat_ns = observed
        elif event == "heartbeat":
            if (
                last_heartbeat_ns is None
                or observed - last_heartbeat_ns > _POWERED_HEARTBEAT_MAX_GAP_NS
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease indexed heartbeat maximum gap was exceeded"
                )
            last_heartbeat_ns = observed
        elif event == "release_intent" and (
            last_heartbeat_ns is None
            or observed - last_heartbeat_ns > _POWERED_HEARTBEAT_MAX_GAP_NS
        ):
            raise LiveLeaseEvidenceError(
                "powered lease release_intent exceeded the heartbeat maximum gap"
            )

        records.append(record)
        record_hashes.append(digest)
        previous_hash = digest

    if not records or records[-1]["event"] != "released":
        raise LiveLeaseEvidenceError(
            "powered lease accepted final ledger requires released last"
        )
    if record_hashes[-1] != index["final_record_sha256"]:
        raise LiveLeaseEvidenceError(
            "powered lease accepted final record hash mismatched"
        )
    if ledger_parent is None:
        raise LiveLeaseEvidenceError("powered lease ledger directory is unavailable")
    if expected_directory is not None and ledger_parent != expected_directory:
        raise LiveLeaseEvidenceError(
            "powered lease indexed parent does not match expected ledger directory"
        )
    expected_paths = {Path(entry["path"]) for entry in index["records"]}
    try:
        actual_paths = set(ledger_parent.iterdir())
    except OSError as exc:
        raise LiveLeaseEvidenceError(
            "powered lease ledger directory cannot be read back"
        ) from exc
    if actual_paths != expected_paths:
        raise LiveLeaseEvidenceError(
            "powered lease ledger contains unindexed or missing generation files"
        )
    return validate_powered_live_lease_index(index)


def _write_create_new_readback(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for optional in ("O_BINARY", "O_NOINHERIT"):
        flags |= int(getattr(os, optional, 0))
    fd = -1
    try:
        fd = os.open(path, flags, 0o600)
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise OSError("short powered lease evidence write")
            written += count
        os.fsync(fd)
    except OSError as exc:
        raise LiveLeaseEvidenceError(
            "powered lease evidence create-new publication failed"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
    try:
        if path.read_bytes() != payload:
            raise LiveLeaseEvidenceError(
                "powered lease evidence readback mismatch"
            )
    except OSError as exc:
        raise LiveLeaseEvidenceError(
            "powered lease evidence readback failed"
        ) from exc


def _default_no_replace_publish(source: Path, destination: Path) -> None:
    """Publish a same-volume pending file without replacing a destination."""

    try:
        if os.name == "nt":
            # MOVEFILE_WRITE_THROUGH without MOVEFILE_REPLACE_EXISTING is the
            # required same-volume, no-clobber publication primitive.
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.MoveFileExW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.LPCWSTR,
                wintypes.DWORD,
            ]
            kernel32.MoveFileExW.restype = wintypes.BOOL
            movefile_write_through = 0x00000008
            if not kernel32.MoveFileExW(
                os.fspath(source),
                os.fspath(destination),
                movefile_write_through,
            ):
                raise OSError(
                    int(ctypes.get_last_error()),
                    "MoveFileExW no-replace write-through publication failed",
                )
        else:
            # Tests and tooling may run elsewhere; link+unlink retains the same
            # atomic create-only property without POSIX rename replacement.
            os.link(source, destination)
            source.unlink()
    except OSError as exc:
        raise LiveLeaseEvidenceError(
            "powered lease generation no-replace publication failed"
        ) from exc


class PoweredLeaseLedgerStore:
    """Strict append-only persistence for powered live-lease `/2` evidence.

    This class deliberately owns no mutex and grants no send authority.  The
    powered mutex state machine supplies already-proved ownership and uses this
    store to make each occurrence immutable before proceeding.
    """

    def __init__(
        self,
        ledger_directory: Path | str,
        final_index_path: Path | str,
        *,
        task_id: str,
        session_id: str,
        attempt_id: str,
        attempt_envelope_sha256: str,
        attempt_context_sha256: str,
        wrapper_process: Mapping[str, Any],
        qpc_frequency_hz: int,
        publish_final_index: bool = True,
        _clock_ns: Callable[[], int] = time.perf_counter_ns,
        _no_replace_publish: Callable[[Path, Path], None] = _default_no_replace_publish,
    ) -> None:
        directory = Path(ledger_directory)
        index_path = Path(final_index_path)
        if not directory.is_absolute() or not index_path.is_absolute():
            raise LiveLeaseEvidenceError(
                "powered lease ledger paths must be absolute"
            )
        try:
            directory_info = directory.lstat()
            index_parent_info = index_path.parent.lstat()
        except OSError as exc:
            raise LiveLeaseEvidenceError(
                "powered lease ledger parents must already exist"
            ) from exc
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or index_path.parent.is_symlink()
            or not index_path.parent.is_dir()
            or not directory_info
            or not index_parent_info
        ):
            raise LiveLeaseEvidenceError(
                "powered lease ledger paths must use regular directories"
            )
        if index_path.exists():
            raise LiveLeaseEvidenceError(
                "powered lease final index already exists"
            )
        if not callable(_clock_ns) or not callable(_no_replace_publish):
            raise TypeError("powered lease injected operations must be callable")
        if type(publish_final_index) is not bool:
            raise TypeError("publish_final_index must be an exact boolean")
        self.ledger_directory = directory
        self.final_index_path = index_path
        self.task_id = _exact_identifier(task_id, "task_id")
        self.session_id = _exact_identifier(session_id, "session_id")
        self.attempt_id = _exact_identifier(attempt_id, "attempt_id")
        self.attempt_envelope_sha256 = _exact_sha256(
            attempt_envelope_sha256, "attempt_envelope_sha256"
        )
        self.attempt_context_sha256 = _exact_sha256(
            attempt_context_sha256, "attempt_context_sha256"
        )
        self.wrapper_process = _validate_process_identity(
            dict(wrapper_process), "wrapper_process"
        )
        self.qpc_frequency_hz = _exact_positive_int(
            qpc_frequency_hz, "qpc_frequency_hz"
        )
        self.publish_final_index = publish_final_index
        self._clock_ns = _clock_ns
        self._no_replace_publish = _no_replace_publish
        self._records: list[dict[str, Any]] = []
        self._record_hashes: list[str] = []
        self._orphaned_pending_files: list[dict[str, Any]] = []
        self._sealed = False
        self._load_existing()

    def _load_existing(self) -> None:
        record_paths: dict[int, Path] = {}
        pending_paths: list[tuple[int, str, Path]] = []
        try:
            children = list(self.ledger_directory.iterdir())
        except OSError as exc:
            raise LiveLeaseEvidenceError(
                "powered lease ledger directory cannot be enumerated"
            ) from exc
        for child in children:
            if child.is_symlink() or not child.is_file():
                raise LiveLeaseEvidenceError(
                    "powered lease ledger contains a non-regular entry"
                )
            record_match = _POWERED_RECORD_NAME.fullmatch(child.name)
            if record_match is not None:
                generation = int(record_match.group("generation"))
                if generation in record_paths:
                    raise LiveLeaseEvidenceError(
                        "powered lease ledger has a duplicate generation"
                    )
                record_paths[generation] = child
                continue
            pending_match = _POWERED_PENDING_NAME.fullmatch(child.name)
            if pending_match is not None:
                pending_paths.append(
                    (
                        int(pending_match.group("generation")),
                        pending_match.group("role"),
                        child,
                    )
                )
                continue
            raise LiveLeaseEvidenceError(
                "powered lease ledger contains an unknown entry"
            )
        expected_generations = list(range(len(record_paths)))
        if sorted(record_paths) != expected_generations:
            raise LiveLeaseEvidenceError(
                "powered lease generations must be contiguous from zero"
            )
        previous_hash: Optional[str] = None
        abandoned = False
        owner_role: Optional[str] = None
        owner_token_sha256: Optional[str] = None
        observed_ns = -1
        child_process: Optional[dict[str, Any]] = None
        cleanup_process: Optional[dict[str, Any]] = None
        for generation in expected_generations:
            path = record_paths[generation]
            payload = path.read_bytes()
            record = load_powered_live_lease_record(path)
            if record["generation"] != generation:
                raise LiveLeaseEvidenceError(
                    "powered lease filename generation does not match content"
                )
            if record["attempt_id"] != self.attempt_id:
                raise LiveLeaseEvidenceError("powered lease attempt_id changed")
            if record["attempt_envelope_sha256"] != self.attempt_envelope_sha256:
                raise LiveLeaseEvidenceError(
                    "powered lease attempt envelope identity changed"
                )
            if record["attempt_context_sha256"] != self.attempt_context_sha256:
                raise LiveLeaseEvidenceError(
                    "powered lease attempt context identity changed"
                )
            if record["wrapper_process"] != self.wrapper_process:
                raise LiveLeaseEvidenceError("powered lease wrapper process changed")
            if record["qpc_frequency_hz"] != self.qpc_frequency_hz:
                raise LiveLeaseEvidenceError("powered lease QPC frequency changed")
            if record["predecessor_sha256"] != previous_hash:
                raise LiveLeaseEvidenceError(
                    "powered lease predecessor hash does not match"
                )
            if record["observed_monotonic_ns"] < observed_ns:
                raise LiveLeaseEvidenceError(
                    "powered lease occurrence time regressed"
                )
            if generation > 0:
                if abandoned and not record["abandoned"]:
                    raise LiveLeaseEvidenceError(
                        "powered lease abandoned state cannot clear"
                    )
                if not abandoned and record["abandoned"] and record["event"] != "takeover":
                    raise LiveLeaseEvidenceError(
                        "powered lease abandoned state begins only at takeover"
                    )
                if record["event"] == "takeover":
                    if owner_role != "wrapper" or abandoned:
                        raise LiveLeaseEvidenceError(
                            "powered lease takeover must replace the live wrapper owner"
                        )
                    if record["owner_token_sha256"] == owner_token_sha256:
                        raise LiveLeaseEvidenceError(
                            "powered lease takeover must use a new owner token hash"
                        )
                    orphan = record["orphaned_pending"]
                    if (
                        orphan is not None
                        and orphan["owner_role"] != owner_role
                    ):
                        raise LiveLeaseEvidenceError(
                            "takeover orphan must belong to the predecessor owner"
                        )
                else:
                    if record["owner_role"] != owner_role:
                        raise LiveLeaseEvidenceError(
                            "powered lease owner role changed without takeover"
                        )
                    if record["owner_token_sha256"] != owner_token_sha256:
                        raise LiveLeaseEvidenceError(
                            "powered lease owner token changed without takeover"
                        )
                if record["event"] == "released" and self._records[-1]["event"] != "release_intent":
                    raise LiveLeaseEvidenceError(
                        "powered lease released record requires release_intent predecessor"
                    )
                if (
                    self._records[-1]["event"] == "release_intent"
                    and record["event"] not in {"released", "takeover"}
                ):
                    raise LiveLeaseEvidenceError(
                        "release_intent may be followed only by released or takeover"
                    )
            if child_process is not None and record["child_process"] != child_process:
                raise LiveLeaseEvidenceError("powered lease child process changed")
            if cleanup_process is not None and record["cleanup_process"] != cleanup_process:
                raise LiveLeaseEvidenceError("powered lease cleanup process changed")
            child_process = record["child_process"] or child_process
            cleanup_process = record["cleanup_process"] or cleanup_process
            abandoned = record["abandoned"]
            owner_role = record["owner_role"]
            owner_token_sha256 = record["owner_token_sha256"]
            observed_ns = record["observed_monotonic_ns"]
            digest = _sha256_bytes(payload)
            self._records.append(record)
            self._record_hashes.append(digest)
            previous_hash = digest
        if pending_paths:
            if len(pending_paths) != 1:
                raise LiveLeaseEvidenceError(
                    "powered lease ledger permits at most one orphaned pending file"
                )
            generation, role, path = pending_paths[0]
            payload, value = _load_bounded_regular_json(
                path, label="powered lease orphaned pending file"
            )
            pending_record = validate_powered_live_lease_record(value)
            if pending_record["generation"] != generation:
                raise LiveLeaseEvidenceError(
                    "orphaned pending content generation does not match"
                )
            if pending_record["owner_role"] != role:
                raise LiveLeaseEvidenceError(
                    "orphaned pending owner role does not match its name"
                )
            evidence = {
                "path": str(path),
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "owner_role": role,
            }
            if generation == len(self._records):
                pass
            elif (
                generation < len(self._records)
                and self._records[generation]["event"] == "takeover"
                and self._records[generation]["orphaned_pending"] == evidence
            ):
                pass
            else:
                raise LiveLeaseEvidenceError(
                    "orphaned pending generation is not the next or bound takeover generation"
                )
            self._orphaned_pending_files.append(evidence)
        bound_orphans = [
            record["orphaned_pending"]
            for record in self._records
            if record["event"] == "takeover"
            and record["orphaned_pending"] is not None
        ]
        if bound_orphans and bound_orphans != self._orphaned_pending_files:
            raise LiveLeaseEvidenceError(
                "takeover orphan evidence must match the sole preserved pending file"
            )

    @property
    def records(self) -> list[dict[str, Any]]:
        return [
            validate_powered_live_lease_record(record)
            for record in self._records
        ]

    @property
    def record_hashes(self) -> list[str]:
        return list(self._record_hashes)

    @property
    def orphaned_pending_files(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._orphaned_pending_files]

    def append(
        self,
        *,
        event: str,
        owner_role: str,
        owner_token_sha256: str,
        owner_process: Mapping[str, Any],
        phase: str,
        abandoned: bool,
        child_process: Optional[Mapping[str, Any]] = None,
        cleanup_process: Optional[Mapping[str, Any]] = None,
        orphaned_pending: Optional[Mapping[str, Any]] = None,
        release_proved: bool = False,
        observed_monotonic_ns: Optional[int] = None,
    ) -> tuple[dict[str, Any], str]:
        if self._sealed:
            raise LiveLeaseEvidenceError("powered lease ledger is already sealed")
        generation = len(self._records)
        if generation > _MAX_POWERED_GENERATION:
            raise LiveLeaseEvidenceError("powered lease generation cap exceeded")
        if generation == 0 and self._orphaned_pending_files:
            raise LiveLeaseEvidenceError(
                "an initial powered lease cannot start over orphaned evidence"
            )
        if orphaned_pending is None and event == "takeover":
            orphaned_pending = (
                self._orphaned_pending_files[0]
                if self._orphaned_pending_files
                else None
            )
        if event == "takeover" and orphaned_pending is not None:
            candidate_orphan = _validate_orphaned_pending(orphaned_pending)
            if self._records and (
                candidate_orphan is None
                or candidate_orphan["owner_role"]
                != self._records[-1]["owner_role"]
            ):
                raise LiveLeaseEvidenceError(
                    "takeover orphan must belong to the predecessor owner"
                )
            if [candidate_orphan] != self._orphaned_pending_files:
                raise LiveLeaseEvidenceError(
                    "takeover orphan must be the sole preserved pending file"
                )
            orphaned_pending = candidate_orphan
        observed = _exact_nonnegative_int(
            self._clock_ns()
            if observed_monotonic_ns is None
            else observed_monotonic_ns,
            "observed_monotonic_ns",
        )
        record = validate_powered_live_lease_record(
            {
                "schema": POWERED_LIVE_LEASE_EVIDENCE_SCHEMA,
                "mutex_name": LIVE_LEASE_MUTEX_NAME,
                "attempt_id": self.attempt_id,
                "attempt_envelope_sha256": self.attempt_envelope_sha256,
                "attempt_context_sha256": self.attempt_context_sha256,
                "generation": generation,
                "predecessor_sha256": (
                    None if generation == 0 else self._record_hashes[-1]
                ),
                "event": event,
                "abandoned": abandoned,
                "owner_role": owner_role,
                "owner_token_sha256": owner_token_sha256,
                "wrapper_process": self.wrapper_process,
                "owner_process": dict(owner_process),
                "child_process": (
                    None if child_process is None else dict(child_process)
                ),
                "cleanup_process": (
                    None if cleanup_process is None else dict(cleanup_process)
                ),
                "host_clock_id": "host-perf-counter",
                "qpc_frequency_hz": self.qpc_frequency_hz,
                "observed_monotonic_ns": observed,
                "phase": phase,
                "orphaned_pending": (
                    None if orphaned_pending is None else dict(orphaned_pending)
                ),
                "release_proved": release_proved,
            }
        )
        if self._records:
            previous = self._records[-1]
            if observed < previous["observed_monotonic_ns"]:
                raise LiveLeaseEvidenceError(
                    "powered lease occurrence time cannot regress"
                )
            if previous["event"] == "released":
                raise LiveLeaseEvidenceError(
                    "powered lease cannot append after released"
                )
            if previous["abandoned"] and not record["abandoned"]:
                raise LiveLeaseEvidenceError(
                    "powered lease abandoned state cannot clear"
                )
            if (
                not previous["abandoned"]
                and record["abandoned"]
                and event != "takeover"
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease abandoned state begins only at takeover"
                )
            if event == "takeover":
                if previous["owner_role"] != "wrapper" or previous["abandoned"]:
                    raise LiveLeaseEvidenceError(
                        "powered lease takeover must replace the live wrapper owner"
                    )
                if record["owner_token_sha256"] == previous["owner_token_sha256"]:
                    raise LiveLeaseEvidenceError(
                        "powered lease takeover must use a new owner token hash"
                    )
            else:
                if record["owner_role"] != previous["owner_role"]:
                    raise LiveLeaseEvidenceError(
                        "powered lease owner role change requires takeover"
                    )
                if record["owner_token_sha256"] != previous["owner_token_sha256"]:
                    raise LiveLeaseEvidenceError(
                        "powered lease owner token change requires takeover"
                    )
            if event == "released" and previous["event"] != "release_intent":
                raise LiveLeaseEvidenceError(
                    "powered lease released record requires release_intent predecessor"
                )
            if (
                previous["event"] == "release_intent"
                and event not in {"released", "takeover"}
            ):
                raise LiveLeaseEvidenceError(
                    "release_intent may be followed only by released or takeover"
                )
            if (
                previous["child_process"] is not None
                and record["child_process"] != previous["child_process"]
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease child process cannot disappear or change"
                )
            if (
                previous["cleanup_process"] is not None
                and record["cleanup_process"] != previous["cleanup_process"]
            ):
                raise LiveLeaseEvidenceError(
                    "powered lease cleanup process cannot disappear or change"
                )
        payload = _canonical_json_line(record)
        pending = self.ledger_directory / (
            f"pending-generation-{generation:06d}-{owner_role}.json"
        )
        destination = self.ledger_directory / f"generation-{generation:06d}.json"
        _write_create_new_readback(pending, payload)
        try:
            self._no_replace_publish(pending, destination)
        except Exception:
            # Preserve the complete pending file as failure/takeover evidence.
            raise
        try:
            if destination.read_bytes() != payload:
                raise LiveLeaseEvidenceError(
                    "powered lease generation readback mismatch"
                )
        except OSError as exc:
            raise LiveLeaseEvidenceError(
                "powered lease generation readback failed"
            ) from exc
        digest = _sha256_bytes(payload)
        self._records.append(record)
        self._record_hashes.append(digest)
        return validate_powered_live_lease_record(record), digest

    def seal_released_index(self) -> tuple[dict[str, Any], str]:
        if self._sealed:
            raise LiveLeaseEvidenceError("powered lease index is already sealed")
        if not self._records or self._records[-1]["event"] != "released":
            raise LiveLeaseEvidenceError(
                "powered lease index requires a final released record"
            )
        index = validate_powered_live_lease_index(
            {
                "schema": POWERED_LIVE_LEASE_LEDGER_SCHEMA,
                "task_id": self.task_id,
                "session_id": self.session_id,
                "attempt_id": self.attempt_id,
                "attempt_envelope_sha256": self.attempt_envelope_sha256,
                "records": [
                    {
                        "generation": generation,
                        "path": str(
                            self.ledger_directory
                            / f"generation-{generation:06d}.json"
                        ),
                        "sha256": self._record_hashes[generation],
                        "event": record["event"],
                    }
                    for generation, record in enumerate(self._records)
                ],
                "orphaned_pending_files": sorted(
                    self._orphaned_pending_files,
                    key=lambda item: item["path"],
                ),
                "final_generation": len(self._records) - 1,
                "final_record_sha256": self._record_hashes[-1],
                "release_proved": True,
            }
        )
        payload = _canonical_json_line(index)
        if self.publish_final_index:
            _write_create_new_readback(self.final_index_path, payload)
        self._sealed = True
        return dict(index), _sha256_bytes(payload)


class PoweredLiveSimulatorLease:
    """Thread-affine powered mutex owner backed by an immutable `/2` ledger.

    The class grants no flight or cleanup permission by itself.  Callers must
    separately prove their process/capability authority and install the send
    guards.  Initial ownership accepts only ``WAIT_OBJECT_0``; a parent-death
    cleanup takeover accepts only ``WAIT_ABANDONED`` and re-runs its injected
    authority proof after ownership transfers.
    """

    HEARTBEAT_PERIOD_NS = _POWERED_HEARTBEAT_PERIOD_NS
    HEARTBEAT_MAX_GAP_NS = _POWERED_HEARTBEAT_MAX_GAP_NS

    def __init__(
        self,
        ledger_store: PoweredLeaseLedgerStore,
        *,
        owner_role: str,
        owner_token_sha256: str,
        owner_process: Mapping[str, Any],
        initial_phase: str,
        child_process: Optional[Mapping[str, Any]] = None,
        cleanup_process: Optional[Mapping[str, Any]] = None,
        takeover: bool = False,
        verify_takeover: Optional[Callable[[], bool]] = None,
        wait_timeout_ms: Optional[int] = None,
        _kernel: Optional[Any] = None,
        _clock_ns: Optional[Callable[[], int]] = None,
        _process_guard: Optional[Any] = None,
    ) -> None:
        if type(ledger_store) is not PoweredLeaseLedgerStore:
            raise TypeError("ledger_store must be exact PoweredLeaseLedgerStore")
        if owner_role not in _POWERED_OWNER_ROLES:
            raise LiveLeaseEvidenceError("powered lease owner role is invalid")
        if type(takeover) is not bool:
            raise TypeError("takeover must be an exact boolean")
        if takeover != (owner_role != "wrapper"):
            raise LiveLeaseEvidenceError(
                "only a non-wrapper parent-death role may request takeover"
            )
        if takeover and not callable(verify_takeover):
            raise LiveLeaseEvidenceError(
                "powered takeover requires an authority verifier"
            )
        if not takeover and verify_takeover is not None:
            raise LiveLeaseEvidenceError(
                "initial wrapper ownership cannot name a takeover verifier"
            )
        default_timeout = 1_000 if takeover else 5_000
        timeout = default_timeout if wait_timeout_ms is None else wait_timeout_ms
        if type(timeout) is not int or timeout < 0 or timeout > default_timeout:
            raise LiveLeaseEvidenceError(
                "powered lease wait timeout exceeds its frozen bound"
            )
        self.ledger_store = ledger_store
        self.owner_role = owner_role
        self.owner_token_sha256 = _exact_sha256(
            owner_token_sha256, "owner_token_sha256"
        )
        self.owner_process = _validate_process_identity(
            dict(owner_process), "owner_process"
        )
        self.child_process = _validate_optional_process_identity(
            None if child_process is None else dict(child_process),
            "child_process",
        )
        self.cleanup_process = _validate_optional_process_identity(
            None if cleanup_process is None else dict(cleanup_process),
            "cleanup_process",
        )
        if owner_role == "wrapper" and self.owner_process != ledger_store.wrapper_process:
            raise LiveLeaseEvidenceError(
                "wrapper powered owner must equal wrapper process"
            )
        if owner_role == "powered-child-parent-death" and (
            self.child_process is None or self.owner_process != self.child_process
        ):
            raise LiveLeaseEvidenceError(
                "child takeover owner must equal child process"
            )
        if owner_role == "cleanup-fallback-parent-death" and (
            self.cleanup_process is None
            or self.owner_process != self.cleanup_process
        ):
            raise LiveLeaseEvidenceError(
                "fallback takeover owner must equal cleanup process"
            )
        if initial_phase not in _POWERED_PHASES:
            raise LiveLeaseEvidenceError("powered lease initial phase is invalid")
        self.initial_phase = initial_phase
        self.takeover = takeover
        self._verify_takeover = verify_takeover
        self.wait_timeout_ms = timeout
        self._kernel = _kernel
        self._clock_ns = _clock_ns or ledger_store._clock_ns
        if not callable(self._clock_ns):
            raise TypeError("_clock_ns must be callable")
        self._process_guard = _process_guard or _PROCESS_LEASE_GUARD
        self._process_guard_held = False
        self._process_guard_poisoned = False
        self._handle: Any = None
        self._owner_thread_id: Optional[int] = None
        self._last_heartbeat_ns: Optional[int] = None
        self._current_phase = initial_phase
        self._entered = False
        self._released = False
        self._latched_invalid = False
        self._state_lock = threading.RLock()

    def _now(self) -> int:
        return _exact_nonnegative_int(self._clock_ns(), "observed_monotonic_ns")

    def _last_error(self) -> int:
        try:
            value = self._kernel.last_error()
        except Exception:
            return 0
        return value if type(value) is int and value >= 0 else 0

    def _drop_process_guard(self) -> None:
        if self._process_guard_held and not self._process_guard_poisoned:
            self._process_guard_held = False
            self._process_guard.release()

    def _release_and_close_uncommitted(self, handle, *, owned: bool) -> list[str]:
        failures: list[str] = []
        if owned:
            try:
                if not self._kernel.release_mutex(handle):
                    self._process_guard_poisoned = True
                    failures.append(
                        f"ReleaseMutex failed with Win32 error {self._last_error()}"
                    )
            except Exception as exc:
                self._process_guard_poisoned = True
                failures.append(f"ReleaseMutex raised {type(exc).__name__}: {exc}")
        try:
            if not self._kernel.close_handle(handle):
                self._process_guard_poisoned = True
                failures.append(
                    f"CloseHandle failed with Win32 error {self._last_error()}"
                )
        except Exception as exc:
            self._process_guard_poisoned = True
            failures.append(f"CloseHandle raised {type(exc).__name__}: {exc}")
        return failures

    def _require_owner_thread(self) -> None:
        if self._handle is None or self._released:
            raise LiveLeaseError("powered live lease is not active")
        if threading.get_ident() != self._owner_thread_id:
            raise LiveLeaseError(
                "powered live lease must be used by its owning thread"
            )

    def _verify_takeover_now(self) -> None:
        if self._verify_takeover is None:
            raise LiveLeaseEvidenceError("takeover verifier is unavailable")
        try:
            result = self._verify_takeover()
        except Exception as exc:
            raise LiveLeaseEvidenceError("takeover authority proof failed") from exc
        if result is not True:
            raise LiveLeaseEvidenceError(
                "takeover authority verifier must return exact true"
            )

    def acquire(self) -> "PoweredLiveSimulatorLease":
        with self._state_lock:
            if self._entered or self._released or self._handle is not None:
                raise LiveLeaseError("powered live lease instances are single-use")
            if self.takeover:
                if not self.ledger_store.records:
                    raise LiveLeaseEvidenceError(
                        "takeover requires a complete predecessor record"
                    )
                if any(record["abandoned"] for record in self.ledger_store.records):
                    raise LiveLeaseEvidenceError(
                        "a powered lease permits at most one abandoned takeover"
                    )
                if self.ledger_store.records[-1]["event"] == "released":
                    raise LiveLeaseEvidenceError(
                        "a released powered lease cannot be taken over"
                    )
                self._verify_takeover_now()
            elif self.ledger_store.records or self.ledger_store.orphaned_pending_files:
                raise LiveLeaseEvidenceError(
                    "initial powered ownership requires an empty ledger"
                )
            self._entered = True
            try:
                if not self._process_guard.acquire(blocking=False):
                    raise LiveLeaseBusyError(
                        "FlightSim live lease is already owned in this process"
                    )
                self._process_guard_held = True
                self._kernel = self._kernel or _Win32Kernel()
                try:
                    handle = self._kernel.create_mutex(LIVE_LEASE_MUTEX_NAME)
                except Exception as exc:
                    raise LiveLeaseUnavailableError(
                        "CreateMutexW could not access the powered live lease"
                    ) from exc
                if not handle:
                    raise LiveLeaseUnavailableError(
                        "CreateMutexW could not access the powered live lease; "
                        f"Win32 error {self._last_error()}"
                    )
                try:
                    result = self._kernel.wait(handle, self.wait_timeout_ms)
                except Exception as exc:
                    failures = self._release_and_close_uncommitted(
                        handle, owned=False
                    )
                    suffix = f"; cleanup: {'; '.join(failures)}" if failures else ""
                    raise LiveLeaseUnavailableError(
                        f"powered live-lease wait was inaccessible{suffix}"
                    ) from exc
                expected = WAIT_ABANDONED if self.takeover else WAIT_OBJECT_0
                if type(result) is not int or result != expected:
                    owns = result in {WAIT_OBJECT_0, WAIT_ABANDONED}
                    failures = self._release_and_close_uncommitted(
                        handle, owned=owns
                    )
                    suffix = f"; cleanup: {'; '.join(failures)}" if failures else ""
                    if result == WAIT_TIMEOUT:
                        raise LiveLeaseBusyError(
                            f"FlightSim powered live lease is busy{suffix}"
                        )
                    if result == WAIT_ABANDONED:
                        raise LiveLeaseAbandonedError(
                            "initial powered live lease was abandoned and is rejected"
                            f"{suffix}"
                        )
                    if result == WAIT_OBJECT_0 and self.takeover:
                        raise LiveLeaseUnavailableError(
                            "cleanup takeover requires exact abandoned ownership"
                            f"{suffix}"
                        )
                    if result == WAIT_FAILED:
                        raise LiveLeaseUnavailableError(
                            "powered live-lease wait failed"
                            f" with Win32 error {self._last_error()}{suffix}"
                        )
                    raise LiveLeaseUnavailableError(
                        f"powered live-lease wait status is unverifiable: {result}{suffix}"
                    )
                if self.takeover:
                    try:
                        self._verify_takeover_now()
                    except Exception:
                        failures = self._release_and_close_uncommitted(
                            handle, owned=True
                        )
                        if failures:
                            self._process_guard_poisoned = True
                        raise
                self._handle = handle
                self._owner_thread_id = threading.get_ident()
                now = self._now()
                event = "takeover" if self.takeover else "acquired"
                try:
                    self.ledger_store.append(
                        event=event,
                        owner_role=self.owner_role,
                        owner_token_sha256=self.owner_token_sha256,
                        owner_process=self.owner_process,
                        child_process=self.child_process,
                        cleanup_process=self.cleanup_process,
                        phase=self.initial_phase,
                        abandoned=self.takeover,
                        release_proved=False,
                        observed_monotonic_ns=now,
                    )
                except Exception:
                    failures = self._release_and_close_uncommitted(
                        handle, owned=True
                    )
                    self._handle = None
                    self._owner_thread_id = None
                    if failures:
                        self._process_guard_poisoned = True
                    raise
                self._last_heartbeat_ns = now
                return self
            except Exception:
                self._drop_process_guard()
                raise

    def bind_child_process(self, value: Mapping[str, Any]) -> None:
        with self._state_lock:
            self._require_owner_thread()
            process = _validate_process_identity(dict(value), "child_process")
            if self.child_process is not None and self.child_process != process:
                raise LiveLeaseEvidenceError("child process cannot be rebound")
            self.child_process = process

    def bind_cleanup_process(self, value: Mapping[str, Any]) -> None:
        with self._state_lock:
            self._require_owner_thread()
            process = _validate_process_identity(dict(value), "cleanup_process")
            if self.cleanup_process is not None and self.cleanup_process != process:
                raise LiveLeaseEvidenceError("cleanup process cannot be rebound")
            self.cleanup_process = process

    def _append_active(
        self,
        event: str,
        phase: str,
        *,
        allow_latched: bool = False,
    ) -> dict[str, Any]:
        self._require_owner_thread()
        if self._latched_invalid and not allow_latched:
            raise LiveLeaseEvidenceError("powered lease evidence is latched invalid")
        if phase not in _POWERED_PHASES:
            raise LiveLeaseEvidenceError("powered lease phase is invalid")
        now = self._now()
        if (
            not allow_latched
            and self._last_heartbeat_ns is not None
            and now - self._last_heartbeat_ns > self.HEARTBEAT_MAX_GAP_NS
        ):
            self._latched_invalid = True
            raise LiveLeaseEvidenceError(
                "powered lease heartbeat maximum gap was exceeded"
            )
        try:
            record, _digest = self.ledger_store.append(
                event=event,
                owner_role=self.owner_role,
                owner_token_sha256=self.owner_token_sha256,
                owner_process=self.owner_process,
                child_process=self.child_process,
                cleanup_process=self.cleanup_process,
                phase=phase,
                abandoned=self.takeover,
                release_proved=False,
                observed_monotonic_ns=now,
            )
        except Exception:
            self._latched_invalid = True
            raise
        if event == "heartbeat":
            self._last_heartbeat_ns = now
        self._current_phase = phase
        return record

    def heartbeat(self, *, phase: Optional[str] = None) -> dict[str, Any]:
        with self._state_lock:
            return self._append_active(
                "heartbeat", self._current_phase if phase is None else phase
            )

    def publish_phase(self, phase: str) -> dict[str, Any]:
        with self._state_lock:
            return self._append_active("phase", phase)

    def release(self) -> tuple[dict[str, Any], str]:
        with self._state_lock:
            self._require_owner_thread()
            release_intent_error: Optional[BaseException] = None
            try:
                self._append_active(
                    "release_intent",
                    "lease_release_and_verify",
                    allow_latched=True,
                )
            except BaseException as exc:
                release_intent_error = exc
            handle = self._handle
            failures = self._release_and_close_uncommitted(handle, owned=True)
            self._handle = None
            self._owner_thread_id = None
            if release_intent_error is not None or failures:
                self._process_guard_poisoned = True
                details = "; ".join(failures)
                message = "powered lease release intent or kernel release failed"
                if details:
                    message += f": {details}"
                error = LiveLeaseCleanupError(message)
                if release_intent_error is not None:
                    raise error from release_intent_error
                raise error
            now = self._now()
            try:
                self.ledger_store.append(
                    event="released",
                    owner_role=self.owner_role,
                    owner_token_sha256=self.owner_token_sha256,
                    owner_process=self.owner_process,
                    child_process=self.child_process,
                    cleanup_process=self.cleanup_process,
                    phase="lease_release_and_verify",
                    abandoned=self.takeover,
                    release_proved=True,
                    observed_monotonic_ns=now,
                )
                index, digest = self.ledger_store.seal_released_index()
            except Exception as exc:
                self._process_guard_poisoned = True
                raise LiveLeaseCleanupError(
                    "powered mutex was released but final lease proof failed"
                ) from exc
            self._released = True
            self._drop_process_guard()
            return index, digest

    @property
    def is_active(self) -> bool:
        return self._handle is not None and not self._released

    @property
    def is_latched_invalid(self) -> bool:
        return self._latched_invalid

    def __enter__(self) -> "PoweredLiveSimulatorLease":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.release()
        except Exception as cleanup_exc:
            if exc is not None:
                cleanup_exc.add_note(
                    f"powered lease body also failed: {type(exc).__name__}: {exc}"
                )
            raise
        return False


class _DeadlineClippedKernel:
    """Delegate one mutex wait with a non-refreshing absolute deadline clip."""

    def __init__(self, backend: Any, deadline_ns: int, clock_ns: Callable[[], int]):
        self._backend = backend
        self._deadline_ns = deadline_ns
        self._clock_ns = clock_ns

    def create_mutex(self, name: str) -> Any:
        return self._backend.create_mutex(name)

    def wait(self, handle: Any, timeout_ms: int) -> int:
        now = _exact_nonnegative_int(
            self._clock_ns(), "delegated powered mutex wait occurrence"
        )
        if now >= self._deadline_ns:
            raise LiveLeaseUnavailableError(
                "delegated powered mutex wait deadline expired"
            )
        remaining_ms = (self._deadline_ns - now) // 1_000_000
        return self._backend.wait(handle, min(timeout_ms, remaining_ms))

    def release_mutex(self, handle: Any) -> bool:
        return self._backend.release_mutex(handle)

    def close_handle(self, handle: Any) -> bool:
        return self._backend.close_handle(handle)

    def last_error(self) -> int:
        return self._backend.last_error()


class DelegatedPoweredLeaseBoundary:
    """Generic live-delegation and parent-death takeover boundary.

    Construction is inert: it neither loads evidence nor opens the production
    mutex. Every proof reloads canonical attempt/authority bytes and a fresh
    immutable ledger view. A successful takeover remains owned by the calling
    thread until its latest proof is heartbeated or released.
    """

    def __init__(
        self,
        ledger_store: PoweredLeaseLedgerStore,
        attempt_envelope_path: Path | str,
        *,
        parent_signaled: Callable[[int], bool],
        _kernel: Optional[Any] = None,
        _clock_ns: Optional[Callable[[], int]] = None,
        _process_guard: Optional[Any] = None,
        _contract: Optional[Any] = None,
    ) -> None:
        if type(ledger_store) is not PoweredLeaseLedgerStore:
            raise TypeError("ledger_store must be exact PoweredLeaseLedgerStore")
        attempt_path = Path(attempt_envelope_path)
        if not attempt_path.is_absolute() or attempt_path.suffix.casefold() != ".json":
            raise LiveLeaseEvidenceError(
                "delegated powered attempt path must be an absolute .json path"
            )
        if not callable(parent_signaled):
            raise TypeError("parent_signaled must be callable")
        clock = ledger_store._clock_ns if _clock_ns is None else _clock_ns
        if not callable(clock):
            raise TypeError("delegated powered lease clock must be callable")
        self.ledger_store = ledger_store
        self.attempt_envelope_path = attempt_path
        self._parent_signaled = parent_signaled
        self._kernel = _kernel
        self._clock_ns = clock
        self._process_guard = _process_guard
        self._contract_override = _contract
        self._state_lock = threading.RLock()
        self._active_deadline_ns: Optional[int] = None
        self._active_minimum_ns: Optional[int] = None
        self._takeover_attempted = False
        self._takeover_lease: Optional[PoweredLiveSimulatorLease] = None
        self._takeover_proof: Optional[DelegatedPoweredLeaseProof] = None
        self._owner_thread_id: Optional[int] = None
        self._last_heartbeat_ns: Optional[int] = None
        self._released = False

    def _contract(self) -> Any:
        if self._contract_override is not None:
            return self._contract_override
        # Lazy by design so importing this module stays independent and inert.
        from scripts import aigp_vq2_powered_attempt

        return aigp_vq2_powered_attempt

    def _raw_now(self) -> int:
        return _exact_nonnegative_int(
            self._clock_ns(), "delegated powered lease clock occurrence"
        )

    def _lease_clock(self) -> int:
        now = self._raw_now()
        deadline = self._active_deadline_ns
        if deadline is not None and now >= deadline:
            raise LiveLeaseEvidenceError(
                "delegated powered lease operation deadline expired"
            )
        minimum = self._active_minimum_ns
        if minimum is not None and now < minimum:
            raise LiveLeaseEvidenceError(
                "delegated powered heartbeat preceded its frozen one-second cadence"
            )
        return now

    @staticmethod
    def _validated_deadline(value: int) -> int:
        return _exact_positive_int(value, "delegated powered lease deadline")

    def _canonical_contract_file(
        self,
        path: Path,
        *,
        label: str,
        validator: Callable[[Any], Mapping[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        payload, raw = _load_bounded_regular_json(path, label=label)
        try:
            validated = dict(validator(raw))
            canonical = self._contract().canonical_json_file_bytes(validated)
        except Exception as exc:
            raise LiveLeaseEvidenceError(f"{label} validation failed") from exc
        if payload != canonical:
            raise LiveLeaseEvidenceError(f"{label} bytes are not canonical")
        return validated, _sha256_bytes(payload)

    def _load_attempt(self, supplied: Mapping[str, Any]) -> dict[str, Any]:
        contract = self._contract()
        loaded, digest = self._canonical_contract_file(
            self.attempt_envelope_path,
            label="delegated powered attempt envelope",
            validator=contract.validate_attempt,
        )
        try:
            supplied_validated = dict(contract.validate_attempt(supplied))
        except Exception as exc:
            raise LiveLeaseEvidenceError(
                "supplied delegated powered attempt validation failed"
            ) from exc
        if supplied_validated != loaded:
            raise LiveLeaseEvidenceError(
                "supplied delegated powered attempt changed from immutable bytes"
            )
        if digest != self.ledger_store.attempt_envelope_sha256:
            raise LiveLeaseEvidenceError(
                "delegated powered attempt hash does not match the lease"
            )
        context = loaded["context"]
        if (
            loaded["context_sha256"] != self.ledger_store.attempt_context_sha256
            or context["task_id"] != self.ledger_store.task_id
            or context["session_id"] != self.ledger_store.session_id
            or context["attempt_id"] != self.ledger_store.attempt_id
            or context["wrapper_process"] != self.ledger_store.wrapper_process
            or context["host"]["qpc_frequency_hz"]
            != self.ledger_store.qpc_frequency_hz
        ):
            raise LiveLeaseEvidenceError(
                "delegated powered attempt does not bind the lease store"
            )
        return loaded

    def _load_authority(
        self,
        supplied: Mapping[str, Any],
        *,
        attempt: Mapping[str, Any],
    ) -> dict[str, Any]:
        contract = self._contract()
        try:
            supplied_validated = dict(
                contract.validate_process_authority(supplied, attempt=attempt)
            )
        except Exception as exc:
            raise LiveLeaseEvidenceError(
                "supplied delegated process authority validation failed"
            ) from exc
        role = supplied_validated["role"]
        if role not in _DELEGATED_ROLE_TO_OWNER:
            raise LiveLeaseEvidenceError("delegated process role is invalid")
        path_key = "child_authority" if role == "powered_child" else "cleanup_authority"
        authority_path = Path(attempt["context"]["paths"][path_key])
        if not authority_path.is_absolute():
            raise LiveLeaseEvidenceError(
                "delegated process authority path must be absolute"
            )
        loaded, _digest = self._canonical_contract_file(
            authority_path,
            label="delegated powered process authority",
            validator=lambda value: contract.validate_process_authority(
                value, attempt=attempt
            ),
        )
        if loaded != supplied_validated:
            raise LiveLeaseEvidenceError(
                "supplied process authority changed from immutable bytes"
            )
        return loaded

    def _reload_store(self) -> PoweredLeaseLedgerStore:
        return PoweredLeaseLedgerStore(
            self.ledger_store.ledger_directory,
            self.ledger_store.final_index_path,
            task_id=self.ledger_store.task_id,
            session_id=self.ledger_store.session_id,
            attempt_id=self.ledger_store.attempt_id,
            attempt_envelope_sha256=self.ledger_store.attempt_envelope_sha256,
            attempt_context_sha256=self.ledger_store.attempt_context_sha256,
            wrapper_process=self.ledger_store.wrapper_process,
            qpc_frequency_hz=self.ledger_store.qpc_frequency_hz,
            _clock_ns=self._lease_clock,
            _no_replace_publish=self.ledger_store._no_replace_publish,
        )

    @staticmethod
    def _bound_process_field(role: str) -> str:
        return "child_process" if role == "powered_child" else "cleanup_process"

    def _validate_delegation_state(
        self,
        *,
        attempt: Mapping[str, Any],
        process_authority: Mapping[str, Any],
        require_parent_signaled: bool,
    ) -> tuple[
        PoweredLeaseLedgerStore,
        dict[str, Any],
        dict[str, Any],
        str,
        str,
    ]:
        loaded_attempt = self._load_attempt(attempt)
        authority = self._load_authority(
            process_authority, attempt=loaded_attempt
        )
        role = authority["role"]
        capability_key, _domain = _DELEGATED_ROLE_TO_CAPABILITY[role]
        if (
            authority["capability_sha256"]
            != loaded_attempt["capabilities"][capability_key]
        ):
            raise LiveLeaseEvidenceError(
                "delegated process capability hash does not match its role"
            )
        parent_handle = authority["parent_handle"]["value"]
        try:
            parent_is_signaled = self._parent_signaled(parent_handle)
        except Exception as exc:
            raise LiveLeaseEvidenceError(
                "delegated wrapper liveness proof failed"
            ) from exc
        if type(parent_is_signaled) is not bool:
            raise LiveLeaseEvidenceError(
                "delegated wrapper liveness proof is not an exact boolean"
            )
        if parent_is_signaled is not require_parent_signaled:
            state = "signaled" if require_parent_signaled else "live"
            raise LiveLeaseEvidenceError(
                f"delegated wrapper is not proved {state}"
            )

        store = self._reload_store()
        records = store.records
        hashes = store.record_hashes
        if not records:
            raise LiveLeaseEvidenceError(
                "delegated powered lease has no immutable predecessor"
            )
        authority_hash = authority["lease_record_sha256"]
        matching = [
            index for index, digest in enumerate(hashes) if digest == authority_hash
        ]
        if len(matching) != 1:
            raise LiveLeaseEvidenceError(
                "delegated authority record is not unique in the lease chain"
            )
        delegation_index = matching[0]
        bound_field = self._bound_process_field(role)
        expected_process = authority["process"]
        expected_owner_hash = loaded_attempt["capabilities"]["lease_owner_sha256"]
        for record in records[delegation_index:]:
            if (
                record["owner_role"] != "wrapper"
                or record["abandoned"]
                or record["event"] in {"release_intent", "released"}
                or record["owner_token_sha256"] != expected_owner_hash
                or record[bound_field] != expected_process
            ):
                raise LiveLeaseEvidenceError(
                    "delegated lease descendant no longer grants wrapper authority"
                )
        try:
            parent_rechecked = self._parent_signaled(parent_handle)
        except Exception as exc:
            raise LiveLeaseEvidenceError(
                "delegated wrapper liveness recheck failed"
            ) from exc
        if (
            type(parent_rechecked) is not bool
            or parent_rechecked is not require_parent_signaled
        ):
            raise LiveLeaseEvidenceError(
                "delegated wrapper liveness changed during proof"
            )
        return store, loaded_attempt, authority, role, hashes[-1]

    @staticmethod
    def _prove_live_heartbeat_cadence(
        records: list[dict[str, Any]], now: int
    ) -> None:
        latest = records[-1]["observed_monotonic_ns"]
        if now < latest:
            raise LiveLeaseEvidenceError(
                "delegated live proof clock precedes the current lease record"
            )
        last_heartbeat: Optional[int] = None
        for record in records:
            event = record["event"]
            observed = record["observed_monotonic_ns"]
            if event in {"acquired", "takeover"}:
                last_heartbeat = observed
            elif event == "heartbeat":
                if (
                    last_heartbeat is None
                    or observed - last_heartbeat
                    > _POWERED_HEARTBEAT_MAX_GAP_NS
                ):
                    raise LiveLeaseEvidenceError(
                        "delegated live lease heartbeat maximum gap was exceeded"
                    )
                last_heartbeat = observed
        if (
            last_heartbeat is None
            or now - last_heartbeat > _POWERED_HEARTBEAT_MAX_GAP_NS
        ):
            raise LiveLeaseEvidenceError(
                "delegated live lease heartbeat is stale"
            )

    def prove_live_delegation(
        self,
        *,
        attempt: Mapping[str, Any],
        process_authority: Mapping[str, Any],
    ) -> DelegatedPoweredLeaseProof:
        with self._state_lock:
            if self._takeover_lease is not None or self._released:
                raise LiveLeaseError(
                    "live wrapper delegation is unavailable after takeover"
                )
            store, _loaded, _authority, _role, latest_hash = (
                self._validate_delegation_state(
                    attempt=attempt,
                    process_authority=process_authority,
                    require_parent_signaled=False,
                )
            )
            if store.orphaned_pending_files:
                raise LiveLeaseEvidenceError(
                    "live wrapper delegation cannot continue over pending evidence"
                )
            self._prove_live_heartbeat_cadence(store.records, self._raw_now())
            return DelegatedPoweredLeaseProof(
                owner_role="wrapper",
                generation=len(store.records) - 1,
                record_sha256=latest_hash,
                authority_valid=True,
            )

    @staticmethod
    def _role_capability_sha256(
        role: str, context_sha256: str, secret: bytes
    ) -> str:
        _key, domain = _DELEGATED_ROLE_TO_CAPABILITY[role]
        return hashlib.sha256(
            domain
            + b"\x00"
            + bytes.fromhex(_exact_sha256(context_sha256, "context_sha256"))
            + b"\x00"
            + secret
        ).hexdigest()

    def _release_failed_takeover(self, lease: PoweredLiveSimulatorLease) -> None:
        if not lease.is_active:
            return
        self._active_deadline_ns = None
        self._active_minimum_ns = None
        lease.release()

    def take_over_abandoned(
        self,
        *,
        role_secret: memoryview,
        attempt: Mapping[str, Any],
        process_authority: Mapping[str, Any],
        deadline_monotonic_ns: int,
    ) -> DelegatedPoweredLeaseProof:
        with self._state_lock:
            if self._takeover_attempted:
                raise LiveLeaseError("delegated powered takeover is single-attempt")
            self._takeover_attempted = True
            if type(role_secret) is not memoryview:
                raise TypeError("delegated powered role_secret must be a memoryview")
            secret = role_secret.tobytes()
            if len(secret) != 32:
                raise LiveLeaseEvidenceError(
                    "delegated powered role secret must contain exactly 32 bytes"
                )
            deadline = self._validated_deadline(deadline_monotonic_ns)
            started = self._raw_now()
            if started >= deadline:
                raise LiveLeaseEvidenceError(
                    "delegated powered takeover deadline expired"
                )
            effective_deadline = min(
                deadline, started + _POWERED_HEARTBEAT_PERIOD_NS
            )
            remaining_ns = effective_deadline - started

            store, loaded_attempt, authority, role, predecessor_hash = (
                self._validate_delegation_state(
                    attempt=attempt,
                    process_authority=process_authority,
                    require_parent_signaled=True,
                )
            )
            capability_key, _domain = _DELEGATED_ROLE_TO_CAPABILITY[role]
            if self._role_capability_sha256(
                role, loaded_attempt["context_sha256"], secret
            ) != loaded_attempt["capabilities"][capability_key]:
                raise LiveLeaseEvidenceError(
                    "delegated powered role secret does not match capability"
                )
            owner_role = _DELEGATED_ROLE_TO_OWNER[role]
            owner_hash = derive_powered_takeover_owner_sha256(
                loaded_attempt["context_sha256"], owner_role, secret
            )
            frozen_attempt = dict(loaded_attempt)
            frozen_authority = dict(authority)

            def verify_takeover() -> bool:
                fresh, _attempt, _authority, fresh_role, latest_hash = (
                    self._validate_delegation_state(
                        attempt=frozen_attempt,
                        process_authority=frozen_authority,
                        require_parent_signaled=True,
                    )
                )
                if (
                    fresh_role != role
                    or latest_hash != predecessor_hash
                    or len(fresh.records) != len(store.records)
                ):
                    raise LiveLeaseEvidenceError(
                        "delegated powered takeover predecessor changed"
                    )
                return True

            latest = store.records[-1]
            child = (
                authority["process"]
                if role == "powered_child"
                else latest["child_process"]
            )
            cleanup = (
                authority["process"]
                if role == "cleanup_fallback"
                else latest["cleanup_process"]
            )
            phase = "child_cleanup" if role == "powered_child" else "fallback_cleanup"
            lease = PoweredLiveSimulatorLease(
                store,
                owner_role=owner_role,
                owner_token_sha256=owner_hash,
                owner_process=authority["process"],
                child_process=child,
                cleanup_process=cleanup,
                initial_phase=phase,
                takeover=True,
                verify_takeover=verify_takeover,
                wait_timeout_ms=remaining_ns // 1_000_000,
                _kernel=_DeadlineClippedKernel(
                    (
                        self._kernel
                        if self._kernel is not None
                        else _Win32Kernel()
                    ),
                    effective_deadline,
                    self._raw_now,
                ),
                _clock_ns=self._lease_clock,
                _process_guard=self._process_guard,
            )
            self._active_deadline_ns = effective_deadline
            self._active_minimum_ns = None
            try:
                lease.acquire()
                completed = self._lease_clock()
                takeover_record = store.records[-1]
                takeover_hash = store.record_hashes[-1]
                if (
                    takeover_record["event"] != "takeover"
                    or takeover_record["predecessor_sha256"] != predecessor_hash
                    or completed < takeover_record["observed_monotonic_ns"]
                ):
                    raise LiveLeaseEvidenceError(
                        "delegated powered takeover record is inconsistent"
                    )
                proof = DelegatedPoweredLeaseProof(
                    owner_role=owner_role,
                    generation=takeover_record["generation"],
                    record_sha256=takeover_hash,
                    authority_valid=True,
                    takeover_completed_monotonic_ns=completed,
                )
            except Exception as exc:
                try:
                    self._release_failed_takeover(lease)
                except Exception as cleanup_exc:
                    exc.add_note(
                        "failed takeover also failed to release the mutex: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                raise
            finally:
                self._active_deadline_ns = None
                self._active_minimum_ns = None

            self._takeover_lease = lease
            self._takeover_proof = proof
            self._owner_thread_id = threading.get_ident()
            self._last_heartbeat_ns = takeover_record["observed_monotonic_ns"]
            return proof

    def _require_latest_takeover(
        self, proof: DelegatedPoweredLeaseProof
    ) -> PoweredLiveSimulatorLease:
        if type(proof) is not DelegatedPoweredLeaseProof:
            raise TypeError("takeover proof must be exact DelegatedPoweredLeaseProof")
        if (
            self._takeover_lease is None
            or self._takeover_proof is None
            or self._released
        ):
            raise LiveLeaseError("delegated powered takeover is not active")
        if threading.get_ident() != self._owner_thread_id:
            raise LiveLeaseError(
                "delegated powered takeover must remain on its owning thread"
            )
        if proof != self._takeover_proof or proof.authority_valid is not True:
            raise LiveLeaseEvidenceError(
                "delegated powered takeover proof is stale or invalid"
            )
        return self._takeover_lease

    def heartbeat_takeover(
        self,
        proof: DelegatedPoweredLeaseProof,
        *,
        phase: str,
        deadline_monotonic_ns: int,
    ) -> DelegatedPoweredLeaseProof:
        with self._state_lock:
            lease = self._require_latest_takeover(proof)
            deadline = self._validated_deadline(deadline_monotonic_ns)
            if self._last_heartbeat_ns is None:
                raise LiveLeaseEvidenceError(
                    "delegated powered takeover heartbeat baseline is unavailable"
                )
            self._active_deadline_ns = deadline
            self._active_minimum_ns = (
                self._last_heartbeat_ns + _POWERED_HEARTBEAT_PERIOD_NS
            )
            try:
                record = lease.heartbeat(phase=phase)
                digest = lease.ledger_store.record_hashes[-1]
                refreshed = DelegatedPoweredLeaseProof(
                    owner_role=proof.owner_role,
                    generation=record["generation"],
                    record_sha256=digest,
                    authority_valid=True,
                    takeover_completed_monotonic_ns=(
                        proof.takeover_completed_monotonic_ns
                    ),
                )
            finally:
                self._active_deadline_ns = None
                self._active_minimum_ns = None
            self._last_heartbeat_ns = record["observed_monotonic_ns"]
            self._takeover_proof = refreshed
            return refreshed

    def release_takeover(
        self,
        proof: DelegatedPoweredLeaseProof,
        *,
        deadline_monotonic_ns: int,
    ) -> bool:
        with self._state_lock:
            lease = self._require_latest_takeover(proof)
            deadline = self._validated_deadline(deadline_monotonic_ns)
            release_started = self._raw_now()
            cadence_invalid = bool(
                self._last_heartbeat_ns is None
                or release_started - self._last_heartbeat_ns
                > _POWERED_HEARTBEAT_MAX_GAP_NS
                or lease.is_latched_invalid
            )
            self._active_deadline_ns = deadline
            self._active_minimum_ns = None
            try:
                index, digest = lease.release()
            except Exception:
                if not lease.is_active:
                    self._released = True
                    self._takeover_lease = None
                    self._owner_thread_id = None
                raise
            finally:
                self._active_deadline_ns = None
                self._active_minimum_ns = None

            self._released = True
            self._takeover_lease = None
            self._owner_thread_id = None
            try:
                readback = load_powered_live_lease_index(
                    lease.ledger_store.final_index_path
                )
                validated = validate_powered_live_lease_ledger(readback)
                if validated != index or digest != _sha256_bytes(
                    _canonical_json_line(index)
                ):
                    raise LiveLeaseEvidenceError(
                        "delegated powered final lease index readback mismatched"
                    )
            except Exception as exc:
                raise LiveLeaseCleanupError(
                    "powered mutex was released but final ledger is acceptance-invalid"
                ) from exc
            completed = self._raw_now()
            if completed >= deadline:
                raise LiveLeaseCleanupError(
                    "powered mutex was released outside the frozen release deadline"
                )
            if cadence_invalid:
                raise LiveLeaseCleanupError(
                    "powered mutex was released after invalid heartbeat cadence"
                )
            return True

    @property
    def latest_takeover_proof(self) -> Optional[DelegatedPoweredLeaseProof]:
        return self._takeover_proof

    @property
    def takeover_active(self) -> bool:
        return self._takeover_lease is not None and not self._released


def _evidence_path(value: Path | str) -> Path:
    target = Path(value)
    if not target.is_absolute() or target.suffix.casefold() != ".json":
        raise LiveLeaseEvidenceError(
            "live-lease evidence path must be an absolute .json path"
        )
    try:
        target.parent.lstat()
    except OSError as exc:
        raise LiveLeaseEvidenceError(
            "live-lease evidence parent must already exist"
        ) from exc
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise LiveLeaseEvidenceError(
            "live-lease evidence parent must be a regular directory"
        )
    return target


def _atomic_publish_evidence(
    path: Path,
    value: Mapping[str, Any],
    *,
    replace: bool,
    expected_previous: Optional[Mapping[str, Any]] = None,
) -> None:
    envelope = validate_live_lease_evidence(dict(value))
    encoded = (
        json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_EVIDENCE_BYTES:
        raise LiveLeaseEvidenceError("live-lease evidence exceeds size limit")
    if replace:
        if expected_previous is None:
            raise LiveLeaseEvidenceError(
                "evidence replacement requires the exact prior envelope"
            )
        expected = validate_live_lease_evidence(dict(expected_previous))
        try:
            observed = load_live_lease_evidence(path)
        except LiveLeaseEvidenceError as exc:
            raise LiveLeaseEvidenceError(
                "existing live-lease evidence is missing or invalid"
            ) from exc
        if observed != expected:
            raise LiveLeaseEvidenceError(
                "existing live-lease evidence does not match current owner state"
            )
    elif expected_previous is not None:
        raise LiveLeaseEvidenceError(
            "initial evidence publication cannot name prior evidence"
        )
    fd = -1
    temporary: Optional[Path] = None
    previous: Optional[Path] = None
    previous_contains_evidence = False
    try:
        fd, raw_temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(raw_temporary)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            previous_fd, raw_previous = tempfile.mkstemp(
                prefix=f".{path.name}.previous.", suffix=".json", dir=path.parent
            )
            os.close(previous_fd)
            previous = Path(raw_previous)

            # Atomically move the actual predecessor aside, then verify those
            # exact bytes.  A replacement racing after the earlier read is
            # captured here and restored rather than silently overwritten.
            os.replace(path, previous)
            previous_contains_evidence = True
            actual_previous = load_live_lease_evidence(previous)
            if actual_previous != expected:
                try:
                    os.link(previous, path)
                    previous.unlink()
                    previous = None
                except OSError:
                    # Preserve the unexpected predecessor at ``previous`` if
                    # no-clobber restoration itself cannot be proved.
                    pass
                raise LiveLeaseEvidenceError(
                    "live-lease evidence changed during atomic replacement"
                )
            try:
                os.link(temporary, path)
            except OSError:
                try:
                    os.link(previous, path)
                    previous.unlink()
                    previous = None
                except OSError:
                    # Keep the verified predecessor at ``previous`` for
                    # failure provenance rather than clobber a racing path.
                    pass
                raise
            temporary.unlink()
            temporary = None
            previous.unlink()
            previous = None
        else:
            # A hard-link publication is an atomic create-only operation on
            # the same filesystem.  Unlike os.replace, it cannot clobber a
            # path that appears during publication.
            os.link(temporary, path)
            temporary.unlink()
            temporary = None
    except LiveLeaseEvidenceError:
        raise
    except OSError as exc:
        raise LiveLeaseEvidenceError(
            "live-lease evidence could not be atomically published"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if previous is not None and not previous_contains_evidence:
            try:
                previous.unlink(missing_ok=True)
            except OSError:
                pass


class _Win32Kernel:
    """Small ctypes boundary kept lazy so tests remain cross-platform."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise LiveLeaseUnavailableError(
                "the FlightSim live lease requires Windows"
            )
        from ctypes import wintypes

        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateMutexW.argtypes = [
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        self._kernel32.CreateMutexW.restype = wintypes.HANDLE
        self._kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        self._kernel32.WaitForSingleObject.restype = wintypes.DWORD
        self._kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
        self._kernel32.ReleaseMutex.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    @staticmethod
    def last_error() -> int:
        return int(ctypes.get_last_error())

    def create_mutex(self, name: str):
        return self._kernel32.CreateMutexW(None, False, name)

    def wait(self, handle, timeout_ms: int) -> int:
        return int(self._kernel32.WaitForSingleObject(handle, timeout_ms))

    def release_mutex(self, handle) -> bool:
        return bool(self._kernel32.ReleaseMutex(handle))

    def close_handle(self, handle) -> bool:
        return bool(self._kernel32.CloseHandle(handle))


class LiveSimulatorLease:
    """Nonblocking process-wide FlightSim lease with atomic private evidence.

    The thread that enters the context must also leave it because Win32 mutex
    ownership is thread-affine.  ``heartbeat`` may be called while the handle
    remains owned; it never touches the simulator or any socket.
    """

    def __init__(
        self,
        evidence_path: Path | str,
        *,
        initial_phase: str = "acquired",
        _kernel: Optional[Any] = None,
        _clock_ns: Callable[[], int] = time.time_ns,
        _pid: Optional[int] = None,
        _process_guard: Optional[Any] = None,
    ) -> None:
        self.evidence_path = _evidence_path(evidence_path)
        self.initial_phase = _validate_phase(initial_phase)
        if not callable(_clock_ns):
            raise TypeError("_clock_ns must be callable")
        self._clock_ns = _clock_ns
        self._kernel = _kernel
        self._wrapper_pid = _exact_positive_int(
            os.getpid() if _pid is None else _pid, "wrapper PID"
        )
        self._owner_token = secrets.token_hex(32)
        if _OWNER_TOKEN.fullmatch(self._owner_token) is None:
            raise LiveLeaseEvidenceError("random owner token generation failed")
        self._handle: Any = None
        self._owner_thread_id: Optional[int] = None
        self._evidence: Optional[dict[str, Any]] = None
        self._released = False
        self._entered = False
        self._process_guard = _process_guard or _PROCESS_LEASE_GUARD
        self._process_guard_held = False
        self._process_guard_poisoned = False
        self._state_lock = threading.RLock()

    def _now(self) -> int:
        return _exact_positive_int(self._clock_ns(), "wall timestamp")

    def _last_error(self) -> int:
        try:
            value = self._kernel.last_error()
        except Exception:
            return 0
        return int(value) if type(value) is int and value >= 0 else 0

    def _close_unowned(self, handle) -> Optional[str]:
        try:
            if self._kernel.close_handle(handle):
                return None
        except Exception as exc:
            return f"CloseHandle raised {type(exc).__name__}: {exc}"
        return f"CloseHandle failed with Win32 error {self._last_error()}"

    def _drop_process_guard(self) -> None:
        if self._process_guard_held and not self._process_guard_poisoned:
            self._process_guard_held = False
            self._process_guard.release()

    def _reject_wait_result(self, handle, result: int) -> None:
        # ctypes' use_last_error bookkeeping can be changed by cleanup calls,
        # so snapshot a WAIT_FAILED error before closing the handle.
        wait_error = self._last_error() if result == WAIT_FAILED else None
        cleanup: list[str] = []
        if result == WAIT_ABANDONED:
            try:
                if not self._kernel.release_mutex(handle):
                    self._process_guard_poisoned = True
                    cleanup.append(
                        "ReleaseMutex failed with Win32 error "
                        f"{self._last_error()}"
                    )
            except Exception as exc:
                self._process_guard_poisoned = True
                cleanup.append(
                    f"ReleaseMutex raised {type(exc).__name__}: {exc}"
                )
        close_error = self._close_unowned(handle)
        if close_error is not None:
            cleanup.append(close_error)
        suffix = f"; cleanup: {'; '.join(cleanup)}" if cleanup else ""
        if result == WAIT_TIMEOUT:
            raise LiveLeaseBusyError(
                f"FlightSim live lease is busy{suffix}"
            )
        if result == WAIT_ABANDONED:
            raise LiveLeaseAbandonedError(
                "FlightSim live lease was abandoned; this probe must stop"
                f"{suffix}"
            )
        if result == WAIT_FAILED:
            raise LiveLeaseUnavailableError(
                "FlightSim live lease wait failed with Win32 error "
                f"{wait_error}{suffix}"
            )
        raise LiveLeaseUnavailableError(
            f"FlightSim live lease returned unverifiable wait status {result}{suffix}"
        )

    def acquire(self) -> "LiveSimulatorLease":
        """Acquire the global mutex immediately or fail before live contact."""

        with self._state_lock:
            if self._entered or self._handle is not None or self._released:
                raise LiveLeaseError("live lease instances are single-use")
            if self.evidence_path.exists():
                raise LiveLeaseEvidenceError(
                    "live-lease evidence path already exists; use a unique probe path"
                )
            self._entered = True
            try:
                if not self._process_guard.acquire(blocking=False):
                    raise LiveLeaseBusyError(
                        "FlightSim live lease is already owned in this process"
                    )
                self._process_guard_held = True
                self._kernel = self._kernel or _Win32Kernel()
                try:
                    handle = self._kernel.create_mutex(LIVE_LEASE_MUTEX_NAME)
                except Exception as exc:
                    raise LiveLeaseUnavailableError(
                        "CreateMutexW could not access the FlightSim live lease"
                    ) from exc
                if not handle:
                    raise LiveLeaseUnavailableError(
                        "CreateMutexW could not access the FlightSim live lease; "
                        f"Win32 error {self._last_error()}"
                    )
                try:
                    result = self._kernel.wait(handle, 0)
                except Exception as exc:
                    close_error = self._close_unowned(handle)
                    suffix = (
                        f"; {close_error}" if close_error is not None else ""
                    )
                    raise LiveLeaseUnavailableError(
                        f"FlightSim live lease wait was inaccessible{suffix}"
                    ) from exc
                if type(result) is not int or result != WAIT_OBJECT_0:
                    normalized = result if type(result) is int else -1
                    self._reject_wait_result(handle, normalized)

                self._handle = handle
                self._owner_thread_id = threading.get_ident()
                try:
                    acquired = self._now()
                    evidence = {
                        "schema": LIVE_LEASE_EVIDENCE_SCHEMA,
                        "mutex_name": LIVE_LEASE_MUTEX_NAME,
                        "owner_token": self._owner_token,
                        "wrapper_pid": self._wrapper_pid,
                        "acquired_wall_time_ns": acquired,
                        "heartbeat_wall_time_ns": acquired,
                        "phase": self.initial_phase,
                        "child_pid": None,
                        "released_wall_time_ns": None,
                    }
                    _atomic_publish_evidence(
                        self.evidence_path, evidence, replace=False
                    )
                except Exception as exc:
                    release_verified, cleanup_errors = self._release_kernel_handle()
                    if not release_verified:
                        self._process_guard_poisoned = True
                    suffix = (
                        f"; cleanup: {'; '.join(cleanup_errors)}"
                        if cleanup_errors
                        else ""
                    )
                    raise LiveLeaseEvidenceError(
                        f"initial live-lease evidence publication failed{suffix}"
                    ) from exc
                self._evidence = evidence
                return self
            except Exception:
                self._drop_process_guard()
                raise

    def heartbeat(
        self, *, phase: str, child_pid: Any = _UNCHANGED
    ) -> dict[str, Any]:
        """Publish one exact heartbeat while retaining mutex ownership."""

        with self._state_lock:
            if self._handle is None or self._evidence is None or self._released:
                raise LiveLeaseError("cannot heartbeat an inactive live lease")
            phase = _validate_phase(phase)
            prior_child_pid = self._evidence["child_pid"]
            if child_pid is _UNCHANGED:
                child_pid = prior_child_pid
            elif child_pid is not None:
                child_pid = _exact_positive_int(child_pid, "child PID")
            if prior_child_pid is not None and child_pid != prior_child_pid:
                raise LiveLeaseEvidenceError(
                    "live-lease child PID cannot change after it is bound"
                )
            now = self._now()
            if now < self._evidence["heartbeat_wall_time_ns"]:
                raise LiveLeaseEvidenceError(
                    "wall clock regressed across live-lease heartbeat"
                )
            evidence = {
                **self._evidence,
                "heartbeat_wall_time_ns": now,
                "phase": phase,
                "child_pid": child_pid,
            }
            _atomic_publish_evidence(
                self.evidence_path,
                evidence,
                replace=True,
                expected_previous=self._evidence,
            )
            self._evidence = evidence
            return dict(evidence)

    def _release_kernel_handle(self) -> tuple[bool, list[str]]:
        handle = self._handle
        self._handle = None
        errors: list[str] = []
        if handle is None:
            return False, ["live-lease handle is missing"]
        release_verified = False
        try:
            if self._kernel.release_mutex(handle):
                release_verified = True
            else:
                errors.append(
                    f"ReleaseMutex failed with Win32 error {self._last_error()}"
                )
        except Exception as exc:
            errors.append(f"ReleaseMutex raised {type(exc).__name__}: {exc}")
        close_error = self._close_unowned(handle)
        if close_error is not None:
            errors.append(close_error)
        return release_verified, errors

    def release(self) -> dict[str, Any]:
        """Release and close the mutex, then publish clean-release evidence."""

        with self._state_lock:
            if self._released:
                raise LiveLeaseError("live lease was already released")
            if self._handle is None or self._evidence is None:
                raise LiveLeaseError("cannot release an inactive live lease")
            if threading.get_ident() != self._owner_thread_id:
                raise LiveLeaseCleanupError(
                    "Win32 mutex must be released by its acquiring thread"
                )
            release_verified, cleanup_errors = self._release_kernel_handle()
            if not release_verified:
                self._process_guard_poisoned = True
            if cleanup_errors:
                self._drop_process_guard()
                raise LiveLeaseCleanupError("; ".join(cleanup_errors))
            try:
                released = self._now()
                if released < self._evidence["heartbeat_wall_time_ns"]:
                    raise LiveLeaseEvidenceError(
                        "wall clock regressed before live-lease release evidence"
                    )
            except Exception as exc:
                self._drop_process_guard()
                raise LiveLeaseCleanupError(
                    "mutex was released but release timestamp validation failed"
                ) from exc
            evidence = {
                **self._evidence,
                "heartbeat_wall_time_ns": released,
                "phase": "released",
                "released_wall_time_ns": released,
            }
            try:
                _atomic_publish_evidence(
                    self.evidence_path,
                    evidence,
                    replace=True,
                    expected_previous=self._evidence,
                )
            except Exception as exc:
                self._drop_process_guard()
                raise LiveLeaseCleanupError(
                    "mutex was released but clean-release evidence failed"
                ) from exc
            self._evidence = evidence
            self._released = True
            self._drop_process_guard()
            return dict(evidence)

    @property
    def evidence(self) -> dict[str, Any]:
        if self._evidence is None:
            raise LiveLeaseError("live lease has no acquired evidence")
        return dict(self._evidence)

    @property
    def is_active(self) -> bool:
        return self._handle is not None and not self._released

    def __enter__(self) -> "LiveSimulatorLease":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.release()
        except Exception as cleanup_exc:
            if exc is not None:
                cleanup_exc.add_note(
                    f"lease body also failed: {type(exc).__name__}: {exc}"
                )
            raise
        return False


def live_simulator_lease(
    evidence_path: Path | str,
    *,
    initial_phase: str = "acquired",
) -> LiveSimulatorLease:
    """Construct a production Win32 live-lease context manager."""

    return LiveSimulatorLease(evidence_path, initial_phase=initial_phase)


__all__ = [
    "LIVE_LEASE_EVIDENCE_SCHEMA",
    "LIVE_LEASE_MUTEX_NAME",
    "POWERED_LIVE_LEASE_EVIDENCE_SCHEMA",
    "POWERED_LIVE_LEASE_LEDGER_SCHEMA",
    "LiveLeaseAbandonedError",
    "LiveLeaseBusyError",
    "LiveLeaseCleanupError",
    "LiveLeaseError",
    "LiveLeaseEvidenceError",
    "LiveLeaseUnavailableError",
    "LiveSimulatorLease",
    "DelegatedPoweredLeaseBoundary",
    "DelegatedPoweredLeaseProof",
    "PoweredLeaseLedgerStore",
    "PoweredLiveSimulatorLease",
    "derive_powered_takeover_owner_sha256",
    "live_simulator_lease",
    "load_live_lease_evidence",
    "load_powered_live_lease_index",
    "load_powered_live_lease_record",
    "validate_live_lease_evidence",
    "validate_powered_live_lease_ledger",
    "validate_powered_live_lease_index",
    "validate_powered_live_lease_record",
]
