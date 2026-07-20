"""Fail-closed Windows lease for passive FlightSim probes.

This module only owns a named mutex and a private evidence envelope.  It does
not inspect, launch, or contact FlightSim and it does not bind network ports.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


LIVE_LEASE_MUTEX_NAME = r"Global\AIGP-FlightSim-LiveLease-v1"
LIVE_LEASE_EVIDENCE_SCHEMA = "aigp-vq2-live-lease-evidence/1"

WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF

_MAX_EVIDENCE_BYTES = 64 * 1024
_OWNER_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_PHASE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
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
    "LiveLeaseAbandonedError",
    "LiveLeaseBusyError",
    "LiveLeaseCleanupError",
    "LiveLeaseError",
    "LiveLeaseEvidenceError",
    "LiveLeaseUnavailableError",
    "LiveSimulatorLease",
    "live_simulator_lease",
    "load_live_lease_evidence",
    "validate_live_lease_evidence",
]
