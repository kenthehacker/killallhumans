"""Fail-closed runtime primitives for the VQ2 powered calibration pilot.

This module deliberately does not launch or inspect FlightSim, bind a fixed
production port, acquire the live mutex, or read the private evidence root.
The small operating-system boundaries are either caller-triggered or injected,
which keeps ordinary imports and unit tests non-live and cross-platform.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import socket
import stat
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar


HOST_CLOCK_ID = "host-perf-counter"
MAX_POLL_INTERVAL_NS = 50_000_000
CAPABILITY_SECRET_BYTES = 32
CAPABILITY_FRAME_BYTES = 4 + CAPABILITY_SECRET_BYTES
ERROR_BROKEN_PIPE = 109
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_TERMINATE = 0x0001
SYNCHRONIZE = 0x00100000
HANDLE_FLAG_INHERIT = 0x00000001
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x00001000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
MAX_PROCESS_IMAGE_BYTES = 128 * 1024 * 1024

CAPABILITY_DOMAINS = frozenset(
    {
        "aigp-vq2-lease-owner/1",
        "aigp-vq2-powered-child/1",
        "aigp-vq2-powered-cleanup/1",
    }
)

PRODUCTION_OUTBOUND_CATEGORIES = frozenset(
    {
        "arm",
        "attitude_target",
        "disarm",
        "gcs_heartbeat",
        "sim_reset",
        "timesync",
    }
)
ANNOUNCEMENT_CATEGORIES = frozenset({"gcs_heartbeat", "timesync"})
CLEANUP_OUTBOUND_CATEGORIES = frozenset(
    {"attitude_target", "disarm", "gcs_heartbeat", "sim_reset", "timesync"}
)
PROMOTION_STREAMS = frozenset({"HEARTBEAT", "RACE_STATUS", "HIGHRES_IMU"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_TOKEN_RE = re.compile(r"^(?P<pid>[1-9][0-9]*):(?P<created>[1-9][0-9]*)$")
_DECIMAL_HANDLE_RE = re.compile(r"^[1-9][0-9]*$")
_PHASE_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")


class PoweredRuntimeError(RuntimeError):
    """Base class for fail-closed powered runtime failures."""


class PoweredDeadlineExpired(PoweredRuntimeError):
    """An absolute QPC deadline was reached."""


class CapabilityProtocolError(PoweredRuntimeError):
    """The one-shot capability pipe protocol was not proved."""


class ProcessIdentityError(PoweredRuntimeError):
    """A process or handle identity token is malformed."""


class OutboundAuthorityError(PoweredRuntimeError):
    """An outbound operation lacks current production or cleanup authority."""


class ExclusiveUdpError(PoweredRuntimeError):
    """An exclusive IPv4 UDP endpoint could not be proved."""


class MavlinkDatagramError(PoweredRuntimeError):
    """A datagram is not one exact, checksum-valid MAVLink frame."""


class StableFileError(PoweredRuntimeError):
    """A file/path identity could not be read without ambiguity."""


class Win32RuntimeUnavailable(PoweredRuntimeError):
    """A caller-triggered Windows boundary is unavailable on this host."""


class HandleCloseError(PoweredRuntimeError):
    """One or more owned Win32 handles could not be closed."""


class ChildSpawnError(PoweredRuntimeError):
    """A blocked-bootstrap child could not be contained and proved."""


class ProcessResidueError(PoweredRuntimeError):
    """A process or non-breakaway job tree could not be proved exited."""


class UdpOwnershipError(PoweredRuntimeError):
    """Windows UDP owner-table evidence was unavailable or unstable."""


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an exact integer >= {minimum}")
    return value


def _exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact boolean")
    return value


def _sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _clock_now(monotonic_ns: Callable[[], int]) -> int:
    if not callable(monotonic_ns):
        raise TypeError("monotonic_ns must be callable")
    return _exact_int(monotonic_ns(), "QPC nanosecond clock value")


def read_qpc_ns(monotonic_ns: Callable[[], int] = time.perf_counter_ns) -> int:
    """Read one validated host-performance-counter nanosecond occurrence."""

    return _clock_now(monotonic_ns)


class WindowsQpcOperations(Protocol):
    """Injectable exact QueryPerformanceCounter/Frequency boundary."""

    def query_performance_counter(self) -> int: ...

    def query_performance_frequency_hz(self) -> int: ...


class Win32QpcOperations:
    """Lazy ctypes calls to the native Windows performance-counter APIs."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise Win32RuntimeUnavailable(
                "Win32 QPC primitives require the Windows runtime"
            )
        from ctypes import wintypes

        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.QueryPerformanceCounter.argtypes = [
            ctypes.POINTER(ctypes.c_longlong)
        ]
        self._kernel32.QueryPerformanceCounter.restype = wintypes.BOOL
        self._kernel32.QueryPerformanceFrequency.argtypes = [
            ctypes.POINTER(ctypes.c_longlong)
        ]
        self._kernel32.QueryPerformanceFrequency.restype = wintypes.BOOL

    @staticmethod
    def _failure(label: str) -> PoweredRuntimeError:
        return PoweredRuntimeError(
            f"{label} failed with Win32 error {int(ctypes.get_last_error())}"
        )

    def query_performance_counter(self) -> int:
        value = ctypes.c_longlong(0)
        if not self._kernel32.QueryPerformanceCounter(ctypes.byref(value)):
            raise self._failure("QueryPerformanceCounter")
        return int(value.value)

    def query_performance_frequency_hz(self) -> int:
        value = ctypes.c_longlong(0)
        if not self._kernel32.QueryPerformanceFrequency(ctypes.byref(value)):
            raise self._failure("QueryPerformanceFrequency")
        return int(value.value)


class WindowsQpcProvider:
    """Exact nanosecond projection of native Windows QPC occurrences."""

    clock_id = HOST_CLOCK_ID

    def __init__(self, operations: WindowsQpcOperations | None = None) -> None:
        self.operations = Win32QpcOperations() if operations is None else operations
        if not callable(
            getattr(self.operations, "query_performance_counter", None)
        ) or not callable(
            getattr(self.operations, "query_performance_frequency_hz", None)
        ):
            raise TypeError("QPC operations do not implement the exact API")
        self._frequency_hz: int | None = None
        self._frequency_lock = threading.Lock()

    def query_performance_frequency_hz(self) -> int:
        value = self.operations.query_performance_frequency_hz()
        if type(value) is not int or value <= 0:
            raise PoweredRuntimeError(
                "QueryPerformanceFrequency returned an invalid exact integer"
            )
        with self._frequency_lock:
            if self._frequency_hz is None:
                self._frequency_hz = value
            elif value != self._frequency_hz:
                raise PoweredRuntimeError(
                    "QueryPerformanceFrequency changed after it was latched"
                )
            return self._frequency_hz

    def now_ns(self) -> int:
        frequency = self.query_performance_frequency_hz()
        counter = self.operations.query_performance_counter()
        if type(counter) is not int or counter < 0:
            raise PoweredRuntimeError(
                "QueryPerformanceCounter returned an invalid exact integer"
            )
        return (counter * 1_000_000_000) // frequency


@dataclass(frozen=True)
class PhaseDeadline:
    """One frozen, non-refreshing phase deadline."""

    phase: str
    started_monotonic_ns: int
    duration_ns: int
    parent_deadline_monotonic_ns: int
    deadline_monotonic_ns: int

    def __post_init__(self) -> None:
        if type(self.phase) is not str or _PHASE_RE.fullmatch(self.phase) is None:
            raise ValueError("phase must be a canonical lowercase token")
        start = _exact_int(self.started_monotonic_ns, "phase start")
        duration = _exact_int(self.duration_ns, "phase duration", minimum=1)
        parent = _exact_int(
            self.parent_deadline_monotonic_ns, "parent deadline", minimum=1
        )
        deadline = _exact_int(self.deadline_monotonic_ns, "phase deadline", minimum=1)
        if start >= parent:
            raise ValueError("phase start must precede its parent deadline")
        if deadline != min(start + duration, parent):
            raise ValueError("phase deadline must be the exact frozen minimum")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "started_monotonic_ns": self.started_monotonic_ns,
            "duration_ns": self.duration_ns,
            "parent_deadline_monotonic_ns": self.parent_deadline_monotonic_ns,
            "deadline_monotonic_ns": self.deadline_monotonic_ns,
        }


def freeze_phase_deadline(
    phase: str,
    duration_ns: int,
    parent_deadline_monotonic_ns: int,
    *,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> PhaseDeadline:
    """Freeze ``min(start + duration, parent)`` from exactly one clock read."""

    start = _clock_now(monotonic_ns)
    duration = _exact_int(duration_ns, "phase duration", minimum=1)
    parent = _exact_int(
        parent_deadline_monotonic_ns, "parent deadline", minimum=1
    )
    if start >= parent:
        raise PoweredDeadlineExpired("phase parent deadline was already reached")
    return PhaseDeadline(
        phase=phase,
        started_monotonic_ns=start,
        duration_ns=duration,
        parent_deadline_monotonic_ns=parent,
        deadline_monotonic_ns=min(start + duration, parent),
    )


def remaining_ns(deadline_monotonic_ns: int, now_monotonic_ns: int) -> int:
    """Return nonnegative remaining nanoseconds without refreshing a deadline."""

    deadline = _exact_int(deadline_monotonic_ns, "deadline")
    now = _exact_int(now_monotonic_ns, "current time")
    return max(0, deadline - now)


def deadline_reached(deadline_monotonic_ns: int, now_monotonic_ns: int) -> bool:
    deadline = _exact_int(deadline_monotonic_ns, "deadline")
    now = _exact_int(now_monotonic_ns, "current time")
    return now >= deadline


_T = TypeVar("_T")


def bounded_poll(
    probe: Callable[[], _T | None],
    *,
    deadline_monotonic_ns: int,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    wait_ns: Callable[[int], None] | None = None,
    parent_alive: Callable[[], bool] | None = None,
    max_poll_interval_ns: int = MAX_POLL_INTERVAL_NS,
) -> _T:
    """Poll until ``probe`` returns non-``None`` under one absolute deadline."""

    if not callable(probe):
        raise TypeError("probe must be callable")
    deadline = _exact_int(deadline_monotonic_ns, "deadline", minimum=1)
    interval = _exact_int(max_poll_interval_ns, "poll interval", minimum=1)
    if interval > MAX_POLL_INTERVAL_NS:
        raise ValueError("poll interval must not exceed 50 milliseconds")
    sleeper = wait_ns or (lambda value: time.sleep(value / 1_000_000_000.0))
    if not callable(sleeper):
        raise TypeError("wait_ns must be callable")
    if parent_alive is not None and not callable(parent_alive):
        raise TypeError("parent_alive must be callable or None")

    while True:
        if parent_alive is not None and parent_alive() is not True:
            raise PoweredRuntimeError("parent process is no longer live")
        now = _clock_now(monotonic_ns)
        if now >= deadline:
            raise PoweredDeadlineExpired("absolute poll deadline was reached")
        value = probe()
        if value is not None:
            return value
        delay = min(interval, deadline - now)
        if delay <= 0:
            raise PoweredDeadlineExpired("absolute poll deadline was reached")
        sleeper(delay)


def derive_capability_sha256(
    domain: str,
    context_sha256: str,
    secret: bytes | bytearray | memoryview,
) -> str:
    """Derive the contract's context-bound capability digest."""

    if type(domain) is not str or domain not in CAPABILITY_DOMAINS:
        raise ValueError("capability domain is not allowlisted")
    context = bytes.fromhex(_sha256(context_sha256, "attempt context SHA-256"))
    if not isinstance(secret, (bytes, bytearray, memoryview)):
        raise TypeError("capability secret must be bytes-like")
    secret_bytes = bytes(secret)
    if len(secret_bytes) != CAPABILITY_SECRET_BYTES:
        raise ValueError("capability secret must contain exactly 32 bytes")
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + context + b"\x00" + secret_bytes
    ).hexdigest()


def encode_capability_frame(secret: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(secret, (bytes, bytearray, memoryview)):
        raise TypeError("capability secret must be bytes-like")
    value = bytes(secret)
    if len(value) != CAPABILITY_SECRET_BYTES:
        raise ValueError("capability secret must contain exactly 32 bytes")
    return struct.pack("<I", CAPABILITY_SECRET_BYTES) + value


def decode_capability_frame(frame: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise TypeError("capability frame must be bytes-like")
    value = bytes(frame)
    if len(value) != CAPABILITY_FRAME_BYTES:
        raise CapabilityProtocolError("capability frame length is invalid")
    (declared_size,) = struct.unpack_from("<I", value, 0)
    if declared_size != CAPABILITY_SECRET_BYTES:
        raise CapabilityProtocolError("capability frame prefix is invalid")
    return value[4:]


@dataclass(frozen=True)
class PipePeek:
    available_bytes: int
    writer_closed: bool

    def __post_init__(self) -> None:
        _exact_int(self.available_bytes, "pipe available byte count")
        _exact_bool(self.writer_closed, "pipe writer-closed state")
        if self.writer_closed and self.available_bytes != 0:
            raise ValueError("a proved broken pipe cannot report buffered bytes")


class CapabilityPipeOperations(Protocol):
    """Injected boundary for the bounded anonymous-pipe reader."""

    def peek_named_pipe(self, handle: int) -> PipePeek: ...

    def read_file(self, handle: int, size: int) -> bytes: ...

    def process_signaled(self, handle: int) -> bool: ...

    def wait_ns(self, duration_ns: int) -> None: ...

    def close_handle(self, handle: int) -> None: ...


class Win32CapabilityPipeOperations:
    """Lazy ctypes implementation; construction is Windows-only."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise CapabilityProtocolError(
                "Win32 capability pipes require the Windows runtime"
            )
        from ctypes import wintypes

        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.PeekNamedPipe.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        self._kernel32.PeekNamedPipe.restype = wintypes.BOOL
        self._kernel32.ReadFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        self._kernel32.ReadFile.restype = wintypes.BOOL
        self._kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        self._kernel32.WaitForSingleObject.restype = wintypes.DWORD
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    @staticmethod
    def _handle(value: int) -> int:
        return _exact_int(value, "Win32 handle", minimum=1)

    def peek_named_pipe(self, handle: int) -> PipePeek:
        available = self._wintypes.DWORD(0)
        ok = self._kernel32.PeekNamedPipe(
            self._handle(handle), None, 0, None, ctypes.byref(available), None
        )
        if ok:
            return PipePeek(int(available.value), False)
        error = int(ctypes.get_last_error())
        if error == ERROR_BROKEN_PIPE:
            return PipePeek(0, True)
        raise CapabilityProtocolError(
            f"PeekNamedPipe failed with Win32 error {error}"
        )

    def read_file(self, handle: int, size: int) -> bytes:
        count = _exact_int(size, "pipe read size", minimum=1)
        buffer = ctypes.create_string_buffer(count)
        received = self._wintypes.DWORD(0)
        ok = self._kernel32.ReadFile(
            self._handle(handle),
            buffer,
            count,
            ctypes.byref(received),
            None,
        )
        if not ok:
            error = int(ctypes.get_last_error())
            raise CapabilityProtocolError(
                f"ReadFile failed with Win32 error {error}"
            )
        return bytes(buffer.raw[: int(received.value)])

    def process_signaled(self, handle: int) -> bool:
        result = int(self._kernel32.WaitForSingleObject(self._handle(handle), 0))
        if result == WAIT_OBJECT_0:
            return True
        if result == WAIT_TIMEOUT:
            return False
        raise CapabilityProtocolError(
            f"parent-process wait returned unverifiable status {result}"
        )

    @staticmethod
    def wait_ns(duration_ns: int) -> None:
        duration = _exact_int(duration_ns, "pipe poll duration", minimum=1)
        time.sleep(duration / 1_000_000_000.0)

    def close_handle(self, handle: int) -> None:
        if not self._kernel32.CloseHandle(self._handle(handle)):
            error = int(ctypes.get_last_error())
            raise CapabilityProtocolError(
                f"CloseHandle failed with Win32 error {error}"
            )


def read_bound_capability(
    read_handle: int,
    parent_process_handle: int,
    *,
    domain: str,
    context_sha256: str,
    expected_capability_sha256: str,
    deadline_monotonic_ns: int,
    operations: CapabilityPipeOperations,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    max_poll_interval_ns: int = MAX_POLL_INTERVAL_NS,
) -> bytes:
    """Consume one exact capability frame and prove writer EOF.

    The only synchronous read occurs after ``PeekNamedPipe`` reports all 36
    bytes available. The read handle is consumed and closed on every path.
    """

    read_value = _exact_int(read_handle, "capability read handle", minimum=1)
    if operations is None:
        raise TypeError("capability pipe operations are required")
    try:
        parent_value = _exact_int(
            parent_process_handle, "parent-process handle", minimum=1
        )
        deadline = _exact_int(
            deadline_monotonic_ns, "capability deadline", minimum=1
        )
        interval = _exact_int(max_poll_interval_ns, "poll interval", minimum=1)
        if interval > MAX_POLL_INTERVAL_NS:
            raise ValueError("poll interval must not exceed 50 milliseconds")
        expected = _sha256(
            expected_capability_sha256, "expected capability SHA-256"
        )
        if type(domain) is not str or domain not in CAPABILITY_DOMAINS:
            raise ValueError("capability domain is not allowlisted")
        _sha256(context_sha256, "attempt context SHA-256")
    except BaseException:
        operations.close_handle(read_value)
        raise

    def check_live_and_time() -> int:
        parent_signaled = operations.process_signaled(parent_value)
        _exact_bool(parent_signaled, "parent-process signaled state")
        if parent_signaled:
            raise CapabilityProtocolError(
                "parent process exited before capability admission"
            )
        now = _clock_now(monotonic_ns)
        if now >= deadline:
            raise CapabilityProtocolError(
                "capability release deadline was reached"
            )
        return now

    def bounded_wait(now: int) -> None:
        delay = min(interval, deadline - now)
        if delay <= 0:
            raise CapabilityProtocolError(
                "capability release deadline was reached"
            )
        operations.wait_ns(delay)

    try:
        while True:
            now = check_live_and_time()
            peek = operations.peek_named_pipe(read_value)
            if peek.writer_closed:
                raise CapabilityProtocolError(
                    "capability pipe closed before the complete frame"
                )
            if peek.available_bytes > CAPABILITY_FRAME_BYTES:
                raise CapabilityProtocolError("capability pipe contains extra bytes")
            if peek.available_bytes == CAPABILITY_FRAME_BYTES:
                frame = operations.read_file(read_value, CAPABILITY_FRAME_BYTES)
                if len(frame) != CAPABILITY_FRAME_BYTES:
                    raise CapabilityProtocolError("capability frame read was short")
                break
            bounded_wait(now)

        while True:
            now = check_live_and_time()
            peek = operations.peek_named_pipe(read_value)
            if peek.writer_closed:
                break
            if peek.available_bytes != 0:
                raise CapabilityProtocolError("capability pipe contains extra bytes")
            bounded_wait(now)

        secret = decode_capability_frame(frame)
        actual = derive_capability_sha256(domain, context_sha256, secret)
        if not hmac.compare_digest(actual, expected):
            raise CapabilityProtocolError("capability digest did not match context")
        return secret
    finally:
        operations.close_handle(read_value)


@dataclass(frozen=True)
class ProcessIdentityToken:
    pid: int
    creation_filetime_100ns: int

    def __post_init__(self) -> None:
        _exact_int(self.pid, "process PID", minimum=1)
        _exact_int(
            self.creation_filetime_100ns,
            "process creation FILETIME",
            minimum=1,
        )

    def __str__(self) -> str:
        return f"{self.pid}:{self.creation_filetime_100ns}"


def parse_process_identity_token(value: str) -> ProcessIdentityToken:
    if type(value) is not str:
        raise ProcessIdentityError("process identity token must be an exact string")
    match = _PROCESS_TOKEN_RE.fullmatch(value)
    if match is None:
        raise ProcessIdentityError(
            "process identity token must be canonical PID:CREATION_FILETIME"
        )
    return ProcessIdentityToken(
        pid=int(match.group("pid")),
        creation_filetime_100ns=int(match.group("created")),
    )


def parse_decimal_handle(value: str) -> int:
    if type(value) is not str or _DECIMAL_HANDLE_RE.fullmatch(value) is None:
        raise ProcessIdentityError("handle token must be a canonical positive decimal")
    return int(value)


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


def validate_process_identity(value: Any) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != _PROCESS_IDENTITY_KEYS:
        raise ProcessIdentityError("process identity has missing or unknown fields")
    result = dict(value)
    _exact_int(result["pid"], "process PID", minimum=1)
    _exact_int(
        result["creation_filetime_100ns"],
        "process creation FILETIME",
        minimum=1,
    )
    _exact_int(result["windows_session_id"], "Windows session ID")
    image_path = result["image_path"]
    if type(image_path) is not str or not image_path:
        raise ProcessIdentityError("process image path must be a nonempty string")
    _sha256(result["image_sha256"], "process image SHA-256")
    _sha256(result["argv_sha256"], "process argv SHA-256")
    return result


def argv_sha256(argv: Sequence[str]) -> str:
    if type(argv) not in {list, tuple} or not argv:
        raise ProcessIdentityError("argv must be a nonempty exact list or tuple")
    values: list[str] = []
    for item in argv:
        if type(item) is not str:
            raise ProcessIdentityError("every argv item must be an exact string")
        values.append(item)
    payload = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ProcessCreation:
    process_handle: int
    thread_handle: int
    pid: int
    thread_id: int

    def __post_init__(self) -> None:
        _exact_int(self.process_handle, "created process handle", minimum=1)
        _exact_int(self.thread_handle, "created thread handle", minimum=1)
        _exact_int(self.pid, "created process PID", minimum=1)
        _exact_int(self.thread_id, "created thread ID", minimum=1)


class WindowsProcessOperations(Protocol):
    """Injectable boundary used by retained-process and child/job primitives."""

    def current_process_id(self) -> int: ...

    def open_process(
        self,
        pid: int,
        *,
        inheritable: bool,
        terminate_access: bool,
    ) -> int: ...

    def query_process_argv(self, process_handle: int) -> tuple[str, ...]: ...

    def query_process_identity(
        self,
        process_handle: int,
        expected_argv: Sequence[str],
    ) -> dict[str, Any]: ...

    def process_signaled(self, process_handle: int) -> bool: ...

    def process_exit_code(self, process_handle: int) -> int: ...

    def handle_is_inheritable(self, handle: int) -> bool: ...

    def close_handle(self, handle: int) -> None: ...

    def create_capability_pipe(self) -> tuple[int, int]: ...

    def pipe_available_bytes(self, read_handle: int) -> int: ...

    def write_file(self, handle: int, payload: bytes) -> int: ...

    def create_non_breakaway_job(self) -> int: ...

    def spawn_process_explicit(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        environment: Mapping[str, str],
        inherited_handles: Sequence[int],
        stdin_handle: int,
        stdout_handle: int,
        stderr_handle: int,
    ) -> ProcessCreation: ...

    def assign_process_to_job(self, job_handle: int, process_handle: int) -> None: ...

    def query_job_limit_flags(self, job_handle: int) -> int: ...

    def process_in_job(self, process_handle: int, job_handle: int) -> bool: ...

    def job_active_pids(self, job_handle: int) -> tuple[int, ...]: ...

    def terminate_process(self, process_handle: int, exit_code: int) -> None: ...

    def terminate_job(self, job_handle: int, exit_code: int) -> None: ...


def _process_operations(
    operations: WindowsProcessOperations | None,
) -> WindowsProcessOperations:
    return Win32ProcessOperations() if operations is None else operations


def close_owned_handles(
    handles: Mapping[str, int],
    *,
    operations: WindowsProcessOperations,
) -> tuple[str, ...]:
    """Close every distinct named handle and surface all observed failures."""

    if type(handles) is not dict or not handles:
        raise HandleCloseError("owned handles must be a nonempty exact mapping")
    checked: list[tuple[str, int]] = []
    seen: set[int] = set()
    for name, handle in handles.items():
        if type(name) is not str or not name:
            raise HandleCloseError("owned handle names must be nonempty strings")
        value = _exact_int(handle, f"owned handle {name}", minimum=1)
        if value in seen:
            raise HandleCloseError("owned handle values must be distinct")
        seen.add(value)
        checked.append((name, value))
    closed: list[str] = []
    failures: list[str] = []
    for name, handle in checked:
        try:
            operations.close_handle(handle)
        except BaseException as exc:
            failures.append(f"{name}:{type(exc).__name__}")
        else:
            closed.append(name)
    if failures:
        raise HandleCloseError(
            "owned handle close failures: " + ",".join(failures)
        )
    return tuple(closed)


class RetainedProcessHandle:
    """A real retained process handle bound to one exact identity and argv."""

    def __init__(
        self,
        handle_value: int,
        identity: Mapping[str, Any],
        expected_argv: Sequence[str],
        *,
        operations: WindowsProcessOperations,
        owns_handle: bool = True,
        terminate_access: bool = False,
    ) -> None:
        self.handle_value = _exact_int(
            handle_value, "retained process handle", minimum=1
        )
        self._identity = validate_process_identity(dict(identity))
        self.expected_argv = tuple(expected_argv)
        if argv_sha256(self.expected_argv) != self._identity["argv_sha256"]:
            raise ProcessIdentityError("retained process argv does not match identity")
        self.operations = operations
        self.owns_handle = _exact_bool(owns_handle, "retained handle ownership")
        self.terminate_access = _exact_bool(
            terminate_access, "retained termination access"
        )
        self.closed = False

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    @property
    def token(self) -> ProcessIdentityToken:
        return ProcessIdentityToken(
            self._identity["pid"], self._identity["creation_filetime_100ns"]
        )

    def reprove(self) -> dict[str, Any]:
        if self.closed:
            raise ProcessIdentityError("retained process handle is closed")
        observed = validate_process_identity(
            self.operations.query_process_identity(
                self.handle_value, self.expected_argv
            )
        )
        if observed != self._identity:
            raise ProcessIdentityError("retained process identity changed")
        return dict(observed)

    def signaled(self) -> bool:
        if self.closed:
            raise ProcessIdentityError("retained process handle is closed")
        result = self.operations.process_signaled(self.handle_value)
        return _exact_bool(result, "retained process signaled state")

    def alive(self) -> bool:
        if self.signaled():
            return False
        self.reprove()
        return True

    def exit_code(self) -> int:
        if self.closed:
            raise ProcessIdentityError("retained process handle is closed")
        return _exact_int(
            self.operations.process_exit_code(self.handle_value),
            "process exit code",
        )

    def close(self) -> None:
        if self.closed:
            return
        if self.owns_handle:
            self.operations.close_handle(self.handle_value)
        self.closed = True

    def __enter__(self) -> "RetainedProcessHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False


def retain_process(
    pid: int,
    expected_argv: Sequence[str],
    *,
    inheritable: bool = False,
    terminate_access: bool = False,
    operations: WindowsProcessOperations | None = None,
) -> RetainedProcessHandle:
    """Open, retain, and identity-bind a process; close on every partial failure."""

    process_id = _exact_int(pid, "process PID", minimum=1)
    argv = tuple(expected_argv)
    argv_sha256(argv)
    ops = _process_operations(operations)
    handle = ops.open_process(
        process_id,
        inheritable=_exact_bool(inheritable, "process handle inheritance"),
        terminate_access=_exact_bool(terminate_access, "process termination access"),
    )
    handle = _exact_int(handle, "opened process handle", minimum=1)
    try:
        identity = validate_process_identity(
            ops.query_process_identity(handle, argv)
        )
        if identity["pid"] != process_id:
            raise ProcessIdentityError("opened process handle names a different PID")
        if inheritable and ops.handle_is_inheritable(handle) is not True:
            raise ProcessIdentityError("opened process handle is not inheritable")
        return RetainedProcessHandle(
            handle,
            identity,
            argv,
            operations=ops,
            terminate_access=terminate_access,
        )
    except BaseException:
        try:
            ops.close_handle(handle)
        except BaseException as close_exc:
            raise HandleCloseError(
                "failed process retention also failed to close its handle"
            ) from close_exc
        raise


def retain_current_process(
    expected_argv: Sequence[str] | None = None,
    *,
    inheritable: bool = False,
    terminate_access: bool = False,
    operations: WindowsProcessOperations | None = None,
) -> RetainedProcessHandle:
    ops = _process_operations(operations)
    pid = _exact_int(ops.current_process_id(), "current process PID", minimum=1)
    if expected_argv is not None:
        return retain_process(
            pid,
            expected_argv,
            inheritable=inheritable,
            terminate_access=terminate_access,
            operations=ops,
        )
    provisional = ops.open_process(
        pid,
        inheritable=inheritable,
        terminate_access=terminate_access,
    )
    provisional = _exact_int(provisional, "current process handle", minimum=1)
    try:
        argv = ops.query_process_argv(provisional)
    finally:
        try:
            ops.close_handle(provisional)
        except BaseException as close_exc:
            raise HandleCloseError(
                "current argv observation handle could not be closed"
            ) from close_exc
    return retain_process(
        pid,
        argv,
        inheritable=inheritable,
        terminate_access=terminate_access,
        operations=ops,
    )


def adopt_retained_process_handle(
    handle_value: int,
    expected_identity: Mapping[str, Any],
    expected_argv: Sequence[str],
    *,
    owns_handle: bool = True,
    require_inheritable: bool = False,
    operations: WindowsProcessOperations | None = None,
) -> RetainedProcessHandle:
    """Bind an inherited or freshly-created real handle to expected evidence."""

    ops = _process_operations(operations)
    handle = _exact_int(handle_value, "adopted process handle", minimum=1)
    expected = validate_process_identity(dict(expected_identity))
    argv = tuple(expected_argv)
    if expected["argv_sha256"] != argv_sha256(argv):
        raise ProcessIdentityError("expected identity does not bind expected argv")
    observed = validate_process_identity(ops.query_process_identity(handle, argv))
    if observed != expected:
        raise ProcessIdentityError("adopted process handle identity mismatched")
    if require_inheritable and ops.handle_is_inheritable(handle) is not True:
        raise ProcessIdentityError("adopted process handle is not inheritable")
    return RetainedProcessHandle(
        handle,
        observed,
        argv,
        operations=ops,
        owns_handle=owns_handle,
    )


@dataclass(frozen=True)
class ChildBootstrapHandleCloseProof:
    """Bounded closure evidence for the child bootstrap's owned handles."""

    started_monotonic_ns: int
    deadline_monotonic_ns: int
    completed_monotonic_ns: int
    current_process_closed: bool
    parent_process_closed: bool
    within_deadline: bool

    def __post_init__(self) -> None:
        started = _exact_int(
            self.started_monotonic_ns,
            "bootstrap handle close start",
        )
        deadline = _exact_int(
            self.deadline_monotonic_ns,
            "bootstrap handle close deadline",
            minimum=1,
        )
        completed = _exact_int(
            self.completed_monotonic_ns,
            "bootstrap handle close completion",
        )
        if completed < started:
            raise ValueError("bootstrap handle close completion precedes start")
        for value, label in (
            (self.current_process_closed, "current-process close state"),
            (self.parent_process_closed, "parent-process close state"),
            (self.within_deadline, "bootstrap close deadline state"),
        ):
            _exact_bool(value, label)
        if self.within_deadline != (started < deadline and completed < deadline):
            raise ValueError("bootstrap handle close deadline evidence is inconsistent")

    @property
    def proved(self) -> bool:
        return bool(
            self.current_process_closed
            and self.parent_process_closed
            and self.within_deadline
        )


class RetainedChildBootstrapProcessBoundary:
    """Runner/cleanup process boundary bound to retained Win32 handles.

    Construction takes ownership of ``parent_handle`` and opens one additional
    non-inheritable current-process handle. The capability reader remains owned
    by the one-shot capability boundary and is never closed here.
    """

    def __init__(
        self,
        capability_handle: int,
        parent_handle: int,
        *,
        operations: WindowsProcessOperations | None = None,
    ) -> None:
        capability = _exact_int(
            capability_handle,
            "bootstrap capability handle",
            minimum=1,
        )
        parent = _exact_int(
            parent_handle,
            "bootstrap parent-process handle",
            minimum=1,
        )
        if capability == parent:
            raise ProcessIdentityError(
                "capability and parent-process handles must be distinct"
            )
        ops = _process_operations(operations)
        current_handle: int | None = None
        try:
            if _exact_bool(
                ops.handle_is_inheritable(capability),
                "capability handle inheritance",
            ) is not True:
                raise ProcessIdentityError(
                    "capability handle is not an inherited-safe handle"
                )
            if _exact_bool(
                ops.handle_is_inheritable(parent),
                "parent-process handle inheritance",
            ) is not True:
                raise ProcessIdentityError(
                    "parent-process handle is not inherited"
                )

            current_pid = _exact_int(
                ops.current_process_id(),
                "current process PID",
                minimum=1,
            )
            current_handle = _exact_int(
                ops.open_process(
                    current_pid,
                    inheritable=False,
                    terminate_access=False,
                ),
                "retained current-process handle",
                minimum=1,
            )
            if current_handle in {capability, parent}:
                raise ProcessIdentityError(
                    "retained current-process handle aliases an inherited handle"
                )
            if _exact_bool(
                ops.handle_is_inheritable(current_handle),
                "current-process handle inheritance",
            ) is not False:
                raise ProcessIdentityError(
                    "retained current-process handle is unexpectedly inheritable"
                )

            current_argv = tuple(ops.query_process_argv(current_handle))
            argv_sha256(current_argv)
            current_identity = validate_process_identity(
                ops.query_process_identity(current_handle, current_argv)
            )
            if current_identity["pid"] != current_pid:
                raise ProcessIdentityError(
                    "retained current-process handle names another process"
                )

            parent_argv = tuple(ops.query_process_argv(parent))
            argv_sha256(parent_argv)
            parent_identity = validate_process_identity(
                ops.query_process_identity(parent, parent_argv)
            )

            current_retained = RetainedProcessHandle(
                current_handle,
                current_identity,
                current_argv,
                operations=ops,
                owns_handle=True,
            )
            parent_retained = RetainedProcessHandle(
                parent,
                parent_identity,
                parent_argv,
                operations=ops,
                owns_handle=True,
            )
        except BaseException as exc:
            close_failures: list[str] = []
            closed_values: set[int] = set()
            for name, handle in (
                ("current_process", current_handle),
                ("parent_process", parent),
            ):
                if handle is None or handle in closed_values:
                    continue
                closed_values.add(handle)
                try:
                    ops.close_handle(handle)
                except BaseException as close_exc:
                    close_failures.append(f"{name}:{type(close_exc).__name__}")
            if close_failures:
                raise HandleCloseError(
                    "bootstrap construction close failures: "
                    + ",".join(close_failures)
                ) from exc
            raise

        self.capability_handle = capability
        self.parent_handle = parent
        self.operations = ops
        self._current = current_retained
        self._parent = parent_retained
        self._lock = threading.Lock()
        self._last_close_proof: ChildBootstrapHandleCloseProof | None = None
        self._close_deadline_monotonic_ns: int | None = None

    def _require_open(self) -> None:
        if self._current.closed or self._parent.closed:
            raise ProcessIdentityError("bootstrap process boundary is closed")

    @property
    def current_handle_value(self) -> int:
        return self._current.handle_value

    @property
    def closed(self) -> bool:
        return self._current.closed and self._parent.closed

    @property
    def last_close_proof(self) -> ChildBootstrapHandleCloseProof | None:
        return self._last_close_proof

    def current_argv(self) -> tuple[str, ...]:
        with self._lock:
            self._require_open()
            identity = self._current.reprove()
            if identity["argv_sha256"] != argv_sha256(
                self._current.expected_argv
            ):
                raise ProcessIdentityError("current process argv binding changed")
            return tuple(self._current.expected_argv)

    def current_process_identity(self) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            return self._current.reprove()

    def retained_process_identity(self, handle: int) -> dict[str, Any]:
        if _exact_int(handle, "requested parent handle", minimum=1) != self.parent_handle:
            raise ProcessIdentityError("requested parent handle is not retained")
        with self._lock:
            self._require_open()
            return self._parent.reprove()

    def prove_inherited_handle_policy(
        self,
        *,
        capability_handle: int,
        parent_handle: int,
        process_authority: Mapping[str, Any],
    ) -> bool:
        capability = _exact_int(
            capability_handle,
            "policy capability handle",
            minimum=1,
        )
        parent = _exact_int(
            parent_handle,
            "policy parent handle",
            minimum=1,
        )
        if capability == parent:
            return False
        if capability != self.capability_handle or parent != self.parent_handle:
            return False
        if type(process_authority) is not dict:
            return False
        with self._lock:
            self._require_open()
            current = self._current.reprove()
            retained_parent = self._parent.reprove()
            current_argv_hash = argv_sha256(self._current.expected_argv)
            try:
                parent_evidence = process_authority["parent_handle"]
                if type(parent_evidence) is not dict:
                    return False
                authority_matches = bool(
                    process_authority["process"] == current
                    and process_authority["argv_sha256"] == current_argv_hash
                    and process_authority["wrapper_process"] == retained_parent
                    and frozenset(parent_evidence)
                    == {"value", "process", "access", "inherited"}
                    and parent_evidence["value"] == parent
                    and parent_evidence["process"] == retained_parent
                    and parent_evidence["access"]
                    == "synchronize_query_limited_information"
                    and parent_evidence["inherited"] is True
                )
            except (KeyError, TypeError):
                return False
            inheritance_matches = bool(
                _exact_bool(
                    self.operations.handle_is_inheritable(capability),
                    "capability handle inheritance",
                )
                and _exact_bool(
                    self.operations.handle_is_inheritable(parent),
                    "parent-process handle inheritance",
                )
                and not _exact_bool(
                    self.operations.handle_is_inheritable(
                        self._current.handle_value
                    ),
                    "current-process handle inheritance",
                )
            )
            return bool(authority_matches and inheritance_matches)

    def parent_signaled(self, handle: int) -> bool:
        if _exact_int(handle, "parent liveness handle", minimum=1) != self.parent_handle:
            raise ProcessIdentityError("parent liveness handle is not retained")
        with self._lock:
            self._require_open()
            signaled = self._parent.signaled()
            if not signaled:
                self._parent.reprove()
            return signaled

    def close_owned_handles(
        self,
        *,
        deadline_monotonic_ns: int,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> ChildBootstrapHandleCloseProof:
        deadline = _exact_int(
            deadline_monotonic_ns,
            "bootstrap handle close deadline",
            minimum=1,
        )
        with self._lock:
            if self._close_deadline_monotonic_ns is None:
                self._close_deadline_monotonic_ns = deadline
            elif deadline != self._close_deadline_monotonic_ns:
                raise PoweredDeadlineExpired(
                    "bootstrap owned-handle close deadline cannot be refreshed"
                )
            deadline = self._close_deadline_monotonic_ns
            if self.closed and self._last_close_proof is not None:
                if not self._last_close_proof.within_deadline:
                    raise PoweredDeadlineExpired(
                        "bootstrap owned handles closed outside their deadline"
                    )
                if not self._last_close_proof.proved:
                    raise HandleCloseError(
                        "bootstrap owned-handle closure is unproved"
                    )
                return self._last_close_proof
            started = _clock_now(monotonic_ns)
            failures: list[str] = []
            for name, retained in (
                ("parent_process", self._parent),
                ("current_process", self._current),
            ):
                if retained.closed:
                    continue
                try:
                    retained.close()
                except BaseException as exc:
                    failures.append(f"{name}:{type(exc).__name__}")
            completed = _clock_now(monotonic_ns)
            proof = ChildBootstrapHandleCloseProof(
                started_monotonic_ns=started,
                deadline_monotonic_ns=deadline,
                completed_monotonic_ns=completed,
                current_process_closed=self._current.closed,
                parent_process_closed=self._parent.closed,
                within_deadline=started < deadline and completed < deadline,
            )
            self._last_close_proof = proof
            if failures:
                raise HandleCloseError(
                    "bootstrap owned-handle close failures: "
                    + ",".join(failures)
                )
            if not proof.within_deadline:
                raise PoweredDeadlineExpired(
                    "bootstrap owned handles closed outside their deadline"
                )
            if not proof.proved:
                raise HandleCloseError("bootstrap owned-handle closure is unproved")
            return proof


class CapabilityPipeHandles:
    """Parent-owned ends of one inheritable-reader capability pipe."""

    def __init__(
        self,
        read_handle: int,
        write_handle: int,
        *,
        operations: WindowsProcessOperations,
    ) -> None:
        self.read_handle = _exact_int(
            read_handle, "capability read handle", minimum=1
        )
        self.write_handle = _exact_int(
            write_handle, "capability write handle", minimum=1
        )
        if self.read_handle == self.write_handle:
            raise CapabilityProtocolError("capability pipe handles must be distinct")
        self.operations = operations
        self.read_closed = False
        self.write_closed = False
        self.released = False

    def prove_unreleased(self) -> None:
        if self.read_closed or self.write_closed or self.released:
            raise CapabilityProtocolError("capability pipe is not pending release")
        if self.operations.handle_is_inheritable(self.read_handle) is not True:
            raise CapabilityProtocolError("capability read handle is not inheritable")
        if self.operations.handle_is_inheritable(self.write_handle) is not False:
            raise CapabilityProtocolError("capability write handle is inheritable")
        available = _exact_int(
            self.operations.pipe_available_bytes(self.read_handle),
            "capability pipe available bytes",
        )
        if available != 0:
            raise CapabilityProtocolError(
                "capability pipe was written before child containment"
            )

    def close_parent_reader(self) -> None:
        if not self.read_closed:
            self.operations.close_handle(self.read_handle)
            self.read_closed = True

    def release(
        self,
        secret: bytes | bytearray | memoryview,
        *,
        deadline_monotonic_ns: int,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not self.read_closed:
            raise CapabilityProtocolError(
                "parent capability reader must close before release"
            )
        if self.write_closed or self.released:
            raise CapabilityProtocolError("capability release is single-use")
        deadline = _exact_int(
            deadline_monotonic_ns, "capability release deadline", minimum=1
        )
        if _clock_now(monotonic_ns) >= deadline:
            raise PoweredDeadlineExpired("capability release deadline was reached")
        frame = encode_capability_frame(secret)
        try:
            written = _exact_int(
                self.operations.write_file(self.write_handle, frame),
                "capability write byte count",
            )
            if written != len(frame):
                raise CapabilityProtocolError("capability frame write was short")
        finally:
            try:
                self.operations.close_handle(self.write_handle)
            except BaseException as close_exc:
                raise HandleCloseError(
                    "capability writer could not be closed after release"
                ) from close_exc
            self.write_closed = True
        self.released = True

    def abort(self) -> None:
        handles: dict[str, int] = {}
        if not self.read_closed:
            handles["capability_read"] = self.read_handle
        if not self.write_closed:
            handles["capability_write"] = self.write_handle
        if not handles:
            return
        close_owned_handles(handles, operations=self.operations)
        self.read_closed = True
        self.write_closed = True


def create_capability_pipe(
    *, operations: WindowsProcessOperations | None = None
) -> CapabilityPipeHandles:
    ops = _process_operations(operations)
    read_handle, write_handle = ops.create_capability_pipe()
    pipe = CapabilityPipeHandles(read_handle, write_handle, operations=ops)
    try:
        pipe.prove_unreleased()
    except BaseException:
        pipe.abort()
        raise
    return pipe


@dataclass(frozen=True)
class JobContainmentProof:
    handle_value: int
    assigned_before_capability_release: bool
    breakaway_allowed: bool
    silent_breakaway_allowed: bool
    kill_on_close: bool
    process_in_job: bool

    def __post_init__(self) -> None:
        _exact_int(self.handle_value, "job handle", minimum=1)
        for name in (
            "assigned_before_capability_release",
            "breakaway_allowed",
            "silent_breakaway_allowed",
            "kill_on_close",
            "process_in_job",
        ):
            _exact_bool(getattr(self, name), f"job proof {name}")
        if (
            self.assigned_before_capability_release is not True
            or self.breakaway_allowed
            or self.silent_breakaway_allowed
            or self.kill_on_close
            or self.process_in_job is not True
        ):
            raise ChildSpawnError("job containment policy is not exact")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "handle_value": self.handle_value,
            "assigned_before_capability_release": self.assigned_before_capability_release,
            "breakaway_allowed": self.breakaway_allowed,
            "silent_breakaway_allowed": self.silent_breakaway_allowed,
            "kill_on_close": self.kill_on_close,
            "process_in_job": self.process_in_job,
        }


def prove_job_containment(
    job_handle: int,
    process_handle: int,
    *,
    capability_released: bool,
    operations: WindowsProcessOperations,
) -> JobContainmentProof:
    job = _exact_int(job_handle, "job handle", minimum=1)
    process = _exact_int(process_handle, "job process handle", minimum=1)
    released = _exact_bool(capability_released, "capability released state")
    flags = _exact_int(operations.query_job_limit_flags(job), "job limit flags")
    in_job = operations.process_in_job(process, job)
    _exact_bool(in_job, "process-in-job state")
    return JobContainmentProof(
        handle_value=job,
        assigned_before_capability_release=not released,
        breakaway_allowed=bool(flags & JOB_OBJECT_LIMIT_BREAKAWAY_OK),
        silent_breakaway_allowed=bool(
            flags & JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
        ),
        kill_on_close=bool(flags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE),
        process_in_job=in_job,
    )


class SpawnedBlockedChild:
    """Normally-started child blocked on capability input inside a proved job."""

    def __init__(
        self,
        *,
        process: RetainedProcessHandle,
        thread_handle: int,
        thread_id: int,
        job_handle: int,
        capability_pipe: CapabilityPipeHandles,
        containment: JobContainmentProof,
        inherited_handles: Sequence[int],
        operations: WindowsProcessOperations,
    ) -> None:
        self.process = process
        self.thread_handle = _exact_int(
            thread_handle, "primary thread handle", minimum=1
        )
        self.thread_id = _exact_int(thread_id, "primary thread ID", minimum=1)
        self.job_handle = _exact_int(job_handle, "job handle", minimum=1)
        self.capability_pipe = capability_pipe
        self.containment = containment
        self.inherited_handles = tuple(inherited_handles)
        self.operations = operations
        self.thread_closed = False
        self.job_closed = False

    @property
    def identity(self) -> dict[str, Any]:
        return self.process.identity

    def reprove_containment(self) -> JobContainmentProof:
        proof = prove_job_containment(
            self.job_handle,
            self.process.handle_value,
            capability_released=False,
            operations=self.operations,
        )
        if proof.to_primitive() != self.containment.to_primitive():
            raise ChildSpawnError("job containment proof changed")
        return proof

    def release_capability(
        self,
        secret: bytes | bytearray | memoryview,
        *,
        deadline_monotonic_ns: int,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if self.process.signaled():
            raise ChildSpawnError("blocked child exited before capability release")
        self.process.reprove()
        self.reprove_containment()
        self.capability_pipe.release(
            secret,
            deadline_monotonic_ns=deadline_monotonic_ns,
            monotonic_ns=monotonic_ns,
        )

    def close_retained_handles(
        self, *, tree_exit_proof: JobProcessTreeProof
    ) -> tuple[str, ...]:
        if not isinstance(tree_exit_proof, JobProcessTreeProof):
            raise ProcessResidueError(
                "retained job/process handles require exact tree-exit evidence"
            )
        if (
            tree_exit_proof.root_process != self.identity
            or tree_exit_proof.state not in {"exited", "terminated_residue"}
            or not tree_exit_proof.observations[-1].root_signaled
            or tree_exit_proof.observations[-1].active_pids
        ):
            raise ProcessResidueError(
                "retained job/process handles require an empty signaled tree"
            )
        if not self.capability_pipe.write_closed:
            self.capability_pipe.abort()
        handles: dict[str, int] = {}
        if not self.thread_closed:
            handles["primary_thread"] = self.thread_handle
        if not self.process.closed and self.process.owns_handle:
            handles["process"] = self.process.handle_value
        if not self.job_closed:
            handles["job"] = self.job_handle
        if not handles:
            return ()
        closed = close_owned_handles(handles, operations=self.operations)
        self.thread_closed = "primary_thread" in closed or self.thread_closed
        self.process.closed = "process" in closed or self.process.closed
        self.job_closed = "job" in closed or self.job_closed
        return closed


def _validate_spawn_inputs(
    argv: Sequence[str], cwd: str, environment: Mapping[str, str]
) -> tuple[tuple[str, ...], str, dict[str, str]]:
    values = tuple(argv)
    argv_sha256(values)
    executable = values[0]
    if (
        not os.path.isabs(executable)
        or os.path.normpath(executable) != executable
        or os.path.abspath(executable) != executable
    ):
        raise ChildSpawnError("child executable must be canonical absolute")
    if type(cwd) is not str or not cwd or not os.path.isabs(cwd):
        raise ChildSpawnError("child cwd must be an absolute string")
    if os.path.normpath(cwd) != cwd or os.path.abspath(cwd) != cwd:
        raise ChildSpawnError("child cwd must be lexically canonical")
    if not os.path.isdir(cwd):
        raise ChildSpawnError("child cwd must name an existing directory")
    if type(environment) is not dict:
        raise ChildSpawnError("child environment must be an exact mapping")
    copied: dict[str, str] = {}
    folded: set[str] = set()
    for name, value in environment.items():
        if (
            type(name) is not str
            or not name
            or "=" in name
            or "\x00" in name
            or type(value) is not str
            or "\x00" in value
        ):
            raise ChildSpawnError("child environment contains an invalid entry")
        key = name.casefold()
        if key in folded:
            raise ChildSpawnError("child environment names collide case-insensitively")
        folded.add(key)
        copied[name] = value
    return values, cwd, copied


def spawn_blocked_child(
    argv: Sequence[str],
    *,
    cwd: str,
    environment: Mapping[str, str],
    capability_pipe: CapabilityPipeHandles,
    parent_process: RetainedProcessHandle,
    stdin_handle: int,
    stdout_handle: int,
    stderr_handle: int,
    operations: WindowsProcessOperations | None = None,
) -> SpawnedBlockedChild:
    """Normally start, immediately job-contain, then return an unreleased child."""

    ops = _process_operations(operations)
    if capability_pipe.operations is not ops or parent_process.operations is not ops:
        raise ChildSpawnError("spawn handles must share one process boundary")
    child_argv, child_cwd, child_environment = _validate_spawn_inputs(
        argv, cwd, environment
    )
    capability_pipe.prove_unreleased()
    if parent_process.alive() is not True:
        raise ChildSpawnError("wrapper process is not live")
    inherited = (
        capability_pipe.read_handle,
        parent_process.handle_value,
        _exact_int(stdout_handle, "child stdout handle", minimum=1),
        _exact_int(stderr_handle, "child stderr handle", minimum=1),
        _exact_int(stdin_handle, "child stdin handle", minimum=1),
    )
    if len(set(inherited)) != 5:
        raise ChildSpawnError("explicit inherited handle list must be five distinct handles")
    for handle in inherited:
        if ops.handle_is_inheritable(handle) is not True:
            raise ChildSpawnError("every explicit child handle must be inheritable")

    job_handle: int | None = None
    creation: ProcessCreation | None = None
    assigned = False
    try:
        job_handle = _exact_int(
            ops.create_non_breakaway_job(), "created job handle", minimum=1
        )
        creation = ops.spawn_process_explicit(
            child_argv,
            cwd=child_cwd,
            environment=child_environment,
            inherited_handles=inherited,
            stdin_handle=inherited[4],
            stdout_handle=inherited[2],
            stderr_handle=inherited[3],
        )
        ops.assign_process_to_job(job_handle, creation.process_handle)
        assigned = True
        containment = prove_job_containment(
            job_handle,
            creation.process_handle,
            capability_released=capability_pipe.released,
            operations=ops,
        )
        identity = validate_process_identity(
            ops.query_process_identity(creation.process_handle, child_argv)
        )
        if identity["pid"] != creation.pid:
            raise ChildSpawnError("created process handle PID mismatched CreateProcess")
        process = RetainedProcessHandle(
            creation.process_handle,
            identity,
            child_argv,
            operations=ops,
            terminate_access=True,
        )
        capability_pipe.close_parent_reader()
        return SpawnedBlockedChild(
            process=process,
            thread_handle=creation.thread_handle,
            thread_id=creation.thread_id,
            job_handle=job_handle,
            capability_pipe=capability_pipe,
            containment=containment,
            inherited_handles=inherited,
            operations=ops,
        )
    except BaseException as exc:
        containment_failures: list[str] = []
        if creation is not None:
            try:
                if assigned and job_handle is not None:
                    ops.terminate_job(job_handle, 1)
                else:
                    ops.terminate_process(creation.process_handle, 1)
            except BaseException as terminate_exc:
                containment_failures.append(type(terminate_exc).__name__)
        handles: dict[str, int] = {}
        if creation is not None:
            handles["created_process"] = creation.process_handle
            handles["created_thread"] = creation.thread_handle
        if job_handle is not None:
            handles["created_job"] = job_handle
        try:
            if handles:
                close_owned_handles(handles, operations=ops)
        except BaseException as close_exc:
            containment_failures.append(type(close_exc).__name__)
        try:
            capability_pipe.abort()
        except BaseException as pipe_exc:
            containment_failures.append(type(pipe_exc).__name__)
        detail = (
            " with containment failures " + ",".join(containment_failures)
            if containment_failures
            else ""
        )
        raise ChildSpawnError("blocked child spawn failed" + detail) from exc


@dataclass(frozen=True)
class ProcessExitProof:
    state: str
    process: dict[str, Any]
    observed_monotonic_ns: int
    exit_code: int | None
    termination_attempted: bool
    termination_returned: bool | None
    termination_is_cleanup_proof: bool = False

    def __post_init__(self) -> None:
        if self.state not in {"exited", "residue", "terminated_residue"}:
            raise ValueError("process exit state is invalid")
        validate_process_identity(self.process)
        _exact_int(self.observed_monotonic_ns, "process exit observation")
        _exact_bool(self.termination_attempted, "termination attempted")
        if self.termination_returned is not None:
            _exact_bool(self.termination_returned, "termination return state")
        if self.termination_is_cleanup_proof is not False:
            raise ValueError("termination can never be cleanup proof")
        if self.state == "exited":
            if self.exit_code is None or self.termination_attempted:
                raise ValueError("natural exit proof is inconsistent")
        elif self.state == "residue":
            if self.exit_code is not None or self.termination_attempted:
                raise ValueError("process residue proof is inconsistent")
        elif not self.termination_attempted or self.termination_returned is not True:
            raise ValueError("terminated residue proof is inconsistent")
        if self.exit_code is not None:
            _exact_int(self.exit_code, "process exit code")


def wait_retained_process_exit(
    process: RetainedProcessHandle,
    *,
    deadline_monotonic_ns: int,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    wait_ns: Callable[[int], None] | None = None,
    max_poll_interval_ns: int = MAX_POLL_INTERVAL_NS,
) -> ProcessExitProof:
    deadline = _exact_int(deadline_monotonic_ns, "process wait deadline", minimum=1)
    interval = _exact_int(max_poll_interval_ns, "process poll interval", minimum=1)
    if interval > MAX_POLL_INTERVAL_NS:
        raise ValueError("process poll interval must not exceed 50 milliseconds")
    sleeper = wait_ns or (lambda value: time.sleep(value / 1_000_000_000.0))
    while True:
        now = _clock_now(monotonic_ns)
        if process.signaled():
            code = process.exit_code()
            if code == STILL_ACTIVE:
                raise ProcessResidueError("signaled process reported STILL_ACTIVE")
            return ProcessExitProof(
                "exited", process.identity, now, code, False, None
            )
        if now >= deadline:
            return ProcessExitProof(
                "residue", process.identity, now, None, False, None
            )
        sleeper(min(interval, deadline - now))


def terminate_process_residue(
    process: RetainedProcessHandle,
    *,
    exit_code: int,
    deadline_monotonic_ns: int,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    wait_ns: Callable[[int], None] | None = None,
) -> ProcessExitProof:
    """Force residual process exit while permanently labeling it non-cleanup proof."""

    code = _exact_int(exit_code, "forced process exit code")
    if not process.terminate_access:
        raise ProcessResidueError("retained process lacks terminate access")
    if process.signaled():
        raise ProcessResidueError("process is not live residue")
    process.operations.terminate_process(process.handle_value, code)
    waited = wait_retained_process_exit(
        process,
        deadline_monotonic_ns=deadline_monotonic_ns,
        monotonic_ns=monotonic_ns,
        wait_ns=wait_ns,
    )
    if waited.state != "exited":
        raise ProcessResidueError("terminated process residue remained live")
    return ProcessExitProof(
        "terminated_residue",
        process.identity,
        waited.observed_monotonic_ns,
        waited.exit_code,
        True,
        True,
        False,
    )


@dataclass(frozen=True)
class JobTreeObservation:
    observed_monotonic_ns: int
    root_signaled: bool
    active_pids: tuple[int, ...]

    def __post_init__(self) -> None:
        _exact_int(self.observed_monotonic_ns, "job-tree observation")
        _exact_bool(self.root_signaled, "job-tree root signaled state")
        if type(self.active_pids) is not tuple:
            raise TypeError("job-tree active PIDs must be an exact tuple")
        prior = 0
        for pid in self.active_pids:
            _exact_int(pid, "job-tree active PID", minimum=1)
            if pid <= prior:
                raise ValueError("job-tree active PIDs must be sorted and unique")
            prior = pid

    def to_primitive(self) -> dict[str, Any]:
        return {
            "observed_monotonic_ns": self.observed_monotonic_ns,
            "root_signaled": self.root_signaled,
            "active_pids": list(self.active_pids),
        }


@dataclass(frozen=True)
class JobProcessTreeProof:
    state: str
    root_process: dict[str, Any]
    observations: tuple[JobTreeObservation, ...]
    natural_exit_proved: bool
    termination_attempted: bool
    termination_returned: bool | None
    termination_is_cleanup_proof: bool = False

    def __post_init__(self) -> None:
        if self.state not in {"exited", "residue", "terminated_residue"}:
            raise ValueError("job-tree proof state is invalid")
        validate_process_identity(self.root_process)
        if type(self.observations) is not tuple or not self.observations:
            raise ValueError("job-tree proof requires observations")
        if any(
            current.observed_monotonic_ns <= prior.observed_monotonic_ns
            for prior, current in zip(self.observations, self.observations[1:])
        ):
            raise ValueError("job-tree observations must strictly advance")
        _exact_bool(self.natural_exit_proved, "natural job-tree exit proof")
        _exact_bool(self.termination_attempted, "job-tree termination attempted")
        if self.termination_returned is not None:
            _exact_bool(self.termination_returned, "job termination return state")
        if self.termination_is_cleanup_proof is not False:
            raise ValueError("job termination can never be cleanup proof")
        last = self.observations[-1]
        if self.state == "exited":
            if (
                not self.natural_exit_proved
                or self.termination_attempted
                or not last.root_signaled
                or last.active_pids
            ):
                raise ValueError("natural job-tree exit proof is inconsistent")
        elif self.state == "residue":
            if self.natural_exit_proved or self.termination_attempted:
                raise ValueError("job-tree residue proof is inconsistent")
        elif (
            self.natural_exit_proved
            or not self.termination_attempted
            or self.termination_returned is not True
            or not last.root_signaled
            or last.active_pids
        ):
            raise ValueError("terminated job-tree residue proof is inconsistent")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "root_process": dict(self.root_process),
            "observations": [item.to_primitive() for item in self.observations],
            "natural_exit_proved": self.natural_exit_proved,
            "termination_attempted": self.termination_attempted,
            "termination_returned": self.termination_returned,
            "termination_is_cleanup_proof": self.termination_is_cleanup_proof,
        }


def _observe_job_tree(
    child: SpawnedBlockedChild,
    monotonic_ns: Callable[[], int],
) -> JobTreeObservation:
    pids = child.operations.job_active_pids(child.job_handle)
    if type(pids) is not tuple:
        raise ProcessResidueError("job active-PID query was not an exact tuple")
    checked = tuple(sorted({_exact_int(pid, "job PID", minimum=1) for pid in pids}))
    if checked != pids:
        raise ProcessResidueError("job active-PID query was not sorted and unique")
    return JobTreeObservation(
        _clock_now(monotonic_ns), child.process.signaled(), checked
    )


def wait_job_process_tree_exit(
    child: SpawnedBlockedChild,
    *,
    deadline_monotonic_ns: int,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    wait_ns: Callable[[int], None] | None = None,
    stable_exit_observations: int = 2,
    max_poll_interval_ns: int = MAX_POLL_INTERVAL_NS,
) -> JobProcessTreeProof:
    """Boundedly require a signaled root and stable empty non-breakaway job."""

    deadline = _exact_int(deadline_monotonic_ns, "job-tree deadline", minimum=1)
    required = _exact_int(
        stable_exit_observations, "stable job-tree observations", minimum=2
    )
    interval = _exact_int(max_poll_interval_ns, "job-tree poll interval", minimum=1)
    if interval > MAX_POLL_INTERVAL_NS:
        raise ValueError("job-tree poll interval must not exceed 50 milliseconds")
    sleeper = wait_ns or (lambda value: time.sleep(value / 1_000_000_000.0))
    observations: list[JobTreeObservation] = []
    stable = 0
    while True:
        observation = _observe_job_tree(child, monotonic_ns)
        observations.append(observation)
        if observation.root_signaled and not observation.active_pids:
            stable += 1
            if stable >= required:
                return JobProcessTreeProof(
                    "exited",
                    child.identity,
                    tuple(observations),
                    True,
                    False,
                    None,
                )
        else:
            stable = 0
        now = observation.observed_monotonic_ns
        if now >= deadline:
            return JobProcessTreeProof(
                "residue",
                child.identity,
                tuple(observations),
                False,
                False,
                None,
            )
        sleeper(min(interval, deadline - now))


def terminate_job_process_tree_residue(
    child: SpawnedBlockedChild,
    *,
    exit_code: int,
    deadline_monotonic_ns: int,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    wait_ns: Callable[[int], None] | None = None,
) -> JobProcessTreeProof:
    """Remove a live job tree while retaining the literal non-cleanup label."""

    before = _observe_job_tree(child, monotonic_ns)
    if before.root_signaled and not before.active_pids:
        raise ProcessResidueError("job tree has no observed residue to terminate")
    child.operations.terminate_job(
        child.job_handle, _exact_int(exit_code, "forced job exit code")
    )
    after = wait_job_process_tree_exit(
        child,
        deadline_monotonic_ns=deadline_monotonic_ns,
        monotonic_ns=monotonic_ns,
        wait_ns=wait_ns,
    )
    if after.state != "exited":
        raise ProcessResidueError("terminated job tree retained process residue")
    return JobProcessTreeProof(
        "terminated_residue",
        child.identity,
        (before,) + after.observations,
        False,
        True,
        True,
        False,
    )


def exact_zero_rate_thrust(value: Any) -> bool:
    """Recognize the exact four-field cleanup-zero payload."""

    keys = {"roll_rate_rad_s", "pitch_rate_rad_s", "yaw_rate_rad_s", "thrust"}
    if type(value) is not dict or set(value) != keys:
        return False
    for item in value.values():
        if type(item) not in {int, float} or not math.isfinite(item) or item != 0.0:
            return False
        if type(item) is float and math.copysign(1.0, item) < 0.0:
            return False
    return True


class PoweredOutboundGuards:
    """Thread-safe, one-way production and cleanup authority latches."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._production_enabled = False
        self._production_latched = False
        self._production_reason: str | None = None
        self._cleanup_state = "disabled"

    @property
    def production_latched(self) -> bool:
        with self._lock:
            return self._production_latched

    @property
    def production_reason(self) -> str | None:
        with self._lock:
            return self._production_reason

    @property
    def cleanup_state(self) -> str:
        with self._lock:
            return self._cleanup_state

    def enable_production(self) -> None:
        with self._lock:
            if self._production_latched or self._production_enabled:
                raise OutboundAuthorityError("production authority cannot be re-enabled")
            if self._cleanup_state != "disabled":
                raise OutboundAuthorityError("cleanup state already consumed authority")
            self._production_enabled = True

    def latch_production(self, reason: str) -> None:
        if type(reason) is not str or not reason or len(reason) > 128:
            raise ValueError("production latch reason must be 1..128 characters")
        with self._lock:
            self._production_enabled = False
            self._production_latched = True
            if self._production_reason is None:
                self._production_reason = reason

    def _deny_production(self, reason: str) -> None:
        self.latch_production(reason)
        raise OutboundAuthorityError(reason)

    def authorize_production(
        self,
        category: str,
        *,
        now_monotonic_ns: int,
        deadline_monotonic_ns: int,
        role_valid: bool,
        parent_alive: bool,
        lease_valid: bool,
        peer_frozen: bool,
        source_valid: bool,
        source_promoted: bool,
    ) -> None:
        with self._lock:
            if not self._production_enabled or self._production_latched:
                raise OutboundAuthorityError("production authority is latched off")
            if type(category) is not str or category not in PRODUCTION_OUTBOUND_CATEGORIES:
                self._deny_production("outbound category is not production-allowlisted")
            for value, label in (
                (role_valid, "production role is invalid"),
                (parent_alive, "wrapper parent is not live"),
                (lease_valid, "production lease lineage is invalid"),
                (peer_frozen, "MAVLink peer is not frozen"),
                (source_valid, "MAVLink source validity is latched"),
            ):
                _exact_bool(value, label)
                if not value:
                    self._deny_production(label)
            now = _exact_int(now_monotonic_ns, "production call start")
            deadline = _exact_int(deadline_monotonic_ns, "production deadline")
            if now >= deadline:
                self._deny_production("production deadline was reached")
            _exact_bool(source_promoted, "source-promotion state")
            if category not in ANNOUNCEMENT_CATEGORIES and not source_promoted:
                self._deny_production("MAVLink source is not promoted")

    def enable_cleanup_live(
        self,
        *,
        parent_alive: bool,
        lease_valid: bool,
        source_promoted: bool,
    ) -> None:
        with self._lock:
            if self._cleanup_state != "disabled":
                raise OutboundAuthorityError("cleanup epoch is single-use")
            if not all(
                _exact_bool(value, label)
                for value, label in (
                    (parent_alive, "parent-alive state"),
                    (lease_valid, "cleanup lease state"),
                    (source_promoted, "source-promotion state"),
                )
            ):
                raise OutboundAuthorityError("live cleanup delegation is invalid")
            self.latch_production("cleanup_epoch_started")
            self._cleanup_state = "enabled_live"

    def note_parent_death(self) -> None:
        with self._lock:
            self.latch_production("parent_dead")
            if self._cleanup_state != "closed":
                self._cleanup_state = "takeover_pending"

    def enable_cleanup_takeover(
        self,
        *,
        parent_signaled: bool,
        abandoned_lease_owned: bool,
        authority_valid: bool,
        source_promoted: bool,
    ) -> None:
        with self._lock:
            if self._cleanup_state != "takeover_pending":
                raise OutboundAuthorityError("cleanup takeover is not pending")
            checks = (
                (parent_signaled, "parent-signaled state"),
                (abandoned_lease_owned, "abandoned-lease ownership"),
                (authority_valid, "takeover authority"),
                (source_promoted, "source-promotion state"),
            )
            if not all(_exact_bool(value, label) for value, label in checks):
                raise OutboundAuthorityError("cleanup takeover proof is invalid")
            self._cleanup_state = "enabled_takeover"

    def authorize_cleanup(
        self,
        category: str,
        *,
        now_monotonic_ns: int,
        deadline_monotonic_ns: int,
        parent_alive: bool,
        lease_valid: bool,
        source_promoted: bool,
        exact_zero: bool | None = None,
    ) -> None:
        with self._lock:
            if self._cleanup_state not in {"enabled_live", "enabled_takeover"}:
                raise OutboundAuthorityError("cleanup guard is not enabled")
            if type(category) is not str or category not in CLEANUP_OUTBOUND_CATEGORIES:
                raise OutboundAuthorityError("outbound category is not cleanup-allowlisted")
            now = _exact_int(now_monotonic_ns, "cleanup call start")
            deadline = _exact_int(deadline_monotonic_ns, "cleanup deadline")
            if now >= deadline:
                self._cleanup_state = "closed"
                raise OutboundAuthorityError("cleanup deadline was reached")
            _exact_bool(parent_alive, "parent-alive state")
            _exact_bool(lease_valid, "cleanup lease state")
            _exact_bool(source_promoted, "source-promotion state")
            if not lease_valid or not source_promoted:
                self._cleanup_state = "closed"
                raise OutboundAuthorityError("cleanup lineage is invalid")
            if self._cleanup_state == "enabled_live" and not parent_alive:
                self.note_parent_death()
                raise OutboundAuthorityError("parent death requires abandoned takeover")
            if self._cleanup_state == "enabled_takeover" and parent_alive:
                self._cleanup_state = "closed"
                raise OutboundAuthorityError("takeover cleanup requires a dead parent")
            if category == "attitude_target":
                if exact_zero is not True:
                    raise OutboundAuthorityError(
                        "cleanup attitude target must be exact zero"
                    )
            elif exact_zero is not None:
                raise OutboundAuthorityError(
                    "exact-zero evidence is valid only for attitude target"
                )

    def close_cleanup(self) -> None:
        with self._lock:
            self.latch_production("cleanup_closed")
            self._cleanup_state = "closed"


class UdpOwnerTableOperations(Protocol):
    def udp_owner_rows(self, family: int) -> tuple[tuple[int, int], ...]: ...


@dataclass(frozen=True)
class UdpOwnerSnapshot:
    observed_monotonic_ns: int
    ports: tuple[int, ...]
    ipv4: tuple[tuple[int, tuple[int, ...]], ...]
    ipv6: tuple[tuple[int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        _exact_int(self.observed_monotonic_ns, "UDP owner observation")
        if type(self.ports) is not tuple or not self.ports:
            raise ValueError("UDP owner snapshot ports must be a nonempty tuple")
        prior = -1
        for port in self.ports:
            _exact_int(port, "UDP owner port", minimum=1)
            if port > 65_535 or port <= prior:
                raise ValueError("UDP owner ports must be sorted and unique")
            prior = port
        for family_name, rows in (("IPv4", self.ipv4), ("IPv6", self.ipv6)):
            if type(rows) is not tuple or tuple(item[0] for item in rows) != self.ports:
                raise ValueError(f"{family_name} UDP rows must cover exact ports")
            for port, pids in rows:
                if type(pids) is not tuple:
                    raise TypeError(f"{family_name} UDP PIDs must be a tuple")
                previous_pid = 0
                for pid in pids:
                    _exact_int(pid, f"{family_name} UDP owner PID", minimum=1)
                    if pid <= previous_pid:
                        raise ValueError(
                            f"{family_name} UDP owner PIDs must be sorted and unique"
                        )
                    previous_pid = pid

    def owner_pids(self, family: int, port: int) -> tuple[int, ...]:
        if family not in {socket.AF_INET, socket.AF_INET6}:
            raise UdpOwnershipError("UDP owner family must be IPv4 or IPv6")
        target = _exact_int(port, "UDP owner lookup port", minimum=1)
        rows = self.ipv4 if family == socket.AF_INET else self.ipv6
        for candidate, pids in rows:
            if candidate == target:
                return pids
        raise UdpOwnershipError("UDP owner lookup port was not captured")

    def ownership_key(self) -> tuple[Any, ...]:
        return (self.ports, self.ipv4, self.ipv6)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "observed_monotonic_ns": self.observed_monotonic_ns,
            "ipv4": [
                {"port": port, "pids": list(pids)} for port, pids in self.ipv4
            ],
            "ipv6": [
                {"port": port, "pids": list(pids)} for port, pids in self.ipv6
            ],
        }

    def to_contract_observation(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "observed_monotonic_ns": self.observed_monotonic_ns
        }
        for family_name, rows in (("ipv4", self.ipv4), ("ipv6", self.ipv6)):
            for port, pids in rows:
                value[f"{family_name}_{port}"] = list(pids)
        return value


def _capture_ports(ports: Sequence[int]) -> tuple[int, ...]:
    if type(ports) not in {list, tuple} or not ports:
        raise UdpOwnershipError("UDP snapshot ports must be a nonempty list or tuple")
    values = tuple(sorted(_exact_int(port, "UDP snapshot port", minimum=1) for port in ports))
    if any(port > 65_535 for port in values) or len(set(values)) != len(values):
        raise UdpOwnershipError("UDP snapshot ports must be unique and <= 65535")
    return values


def capture_udp_owner_snapshot(
    ports: Sequence[int],
    *,
    operations: UdpOwnerTableOperations | None = None,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
) -> UdpOwnerSnapshot:
    """Capture one PID-owner table occurrence for both UDP address families."""

    targets = _capture_ports(ports)
    ops = Win32UdpOwnerTableOperations() if operations is None else operations

    def collect(family: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
        rows = ops.udp_owner_rows(family)
        if type(rows) is not tuple:
            raise UdpOwnershipError("UDP owner rows must be an exact tuple")
        owners: dict[int, set[int]] = {port: set() for port in targets}
        prior: tuple[int, int] | None = None
        for row in rows:
            if type(row) is not tuple or len(row) != 2:
                raise UdpOwnershipError("UDP owner row shape is invalid")
            port = _exact_int(row[0], "UDP table port", minimum=1)
            pid_value = _exact_int(row[1], "UDP table owner PID")
            if port > 65_535:
                raise UdpOwnershipError("UDP table port exceeds 65535")
            normalized = (port, pid_value)
            if prior is not None and normalized <= prior:
                raise UdpOwnershipError("UDP owner rows must be sorted and unique")
            prior = normalized
            if port in owners:
                if pid_value == 0:
                    raise UdpOwnershipError("target UDP owner PID is unavailable")
                owners[port].add(pid_value)
        return tuple((port, tuple(sorted(owners[port]))) for port in targets)

    ipv4 = collect(socket.AF_INET)
    ipv6 = collect(socket.AF_INET6)
    return UdpOwnerSnapshot(
        _clock_now(monotonic_ns), targets, ipv4, ipv6
    )


def capture_stable_udp_owner_snapshots(
    ports: Sequence[int],
    *,
    deadline_monotonic_ns: int,
    operations: UdpOwnerTableOperations | None = None,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    wait_ns: Callable[[int], None] | None = None,
    required_identical: int = 2,
    max_poll_interval_ns: int = MAX_POLL_INTERVAL_NS,
) -> tuple[UdpOwnerSnapshot, ...]:
    """Return consecutive identical IPv4+IPv6 owner-table observations."""

    targets = _capture_ports(ports)
    deadline = _exact_int(deadline_monotonic_ns, "UDP snapshot deadline", minimum=1)
    required = _exact_int(required_identical, "stable UDP snapshot count", minimum=2)
    interval = _exact_int(max_poll_interval_ns, "UDP snapshot interval", minimum=1)
    if interval > MAX_POLL_INTERVAL_NS:
        raise ValueError("UDP snapshot interval must not exceed 50 milliseconds")
    ops = Win32UdpOwnerTableOperations() if operations is None else operations
    sleeper = wait_ns or (lambda value: time.sleep(value / 1_000_000_000.0))
    stable: list[UdpOwnerSnapshot] = []
    while True:
        before = _clock_now(monotonic_ns)
        if before >= deadline:
            raise PoweredDeadlineExpired("UDP owner snapshot deadline was reached")
        current = capture_udp_owner_snapshot(
            targets, operations=ops, monotonic_ns=monotonic_ns
        )
        if current.observed_monotonic_ns >= deadline:
            raise PoweredDeadlineExpired("UDP owner snapshot exceeded its deadline")
        if stable and current.ownership_key() != stable[-1].ownership_key():
            stable.clear()
        if stable and (
            current.observed_monotonic_ns
            <= stable[-1].observed_monotonic_ns
        ):
            raise UdpOwnershipError(
                "stable UDP owner observations must strictly advance"
            )
        stable.append(current)
        if len(stable) >= required:
            return tuple(stable[-required:])
        sleeper(min(interval, deadline - current.observed_monotonic_ns))


@dataclass
class ExclusiveUdpEndpoint:
    """Caller-owned exclusive socket plus immutable bind proof."""

    socket: socket.socket | None
    requested_host: str
    requested_port: int
    actual_host: str
    actual_port: int
    exclusive_option: int
    closed: bool = False
    _socket_transferred: bool = field(default=False, init=False, repr=False)

    @property
    def socket_transferred(self) -> bool:
        """Whether the one raw-socket ownership transfer was consumed."""

        return self._socket_transferred

    def transfer_socket(self) -> socket.socket:
        """Move raw-socket ownership to one caller without retaining an alias."""

        if self.closed is not False:
            raise ExclusiveUdpError("closed UDP endpoint cannot transfer its socket")
        if self._socket_transferred or self.socket is None:
            raise ExclusiveUdpError("UDP endpoint socket was already transferred")
        owned = self.socket
        self.socket = None
        self._socket_transferred = True
        return owned

    def proof(self) -> dict[str, Any]:
        return {
            "family": "AF_INET",
            "requested": {
                "host": self.requested_host,
                "port": self.requested_port,
            },
            "actual": {"host": self.actual_host, "port": self.actual_port},
            "socket_policy": "ipv4-exclusive-address-use",
        }

    def close(self) -> None:
        if self._socket_transferred:
            raise ExclusiveUdpError(
                "transferred UDP endpoint no longer owns its socket"
            )
        if not self.closed:
            if self.socket is None:
                raise ExclusiveUdpError("UDP endpoint lost its owned socket")
            self.socket.close()
            self.closed = True

    def __enter__(self) -> "ExclusiveUdpEndpoint":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False


def _ipv4_host(value: Any) -> str:
    if type(value) is not str:
        raise ExclusiveUdpError("UDP bind host must be an exact IPv4 string")
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ExclusiveUdpError("UDP bind host is not a canonical IPv4 address") from exc
    if not isinstance(parsed, ipaddress.IPv4Address) or str(parsed) != value:
        raise ExclusiveUdpError("UDP bind host is not a canonical IPv4 address")
    return value


def create_exclusive_udp_endpoint(
    host: str,
    port: int,
    *,
    socket_factory: Callable[[int, int], socket.socket] = socket.socket,
    exclusive_option: int | None = None,
) -> ExclusiveUdpEndpoint:
    """Create and prove one IPv4 UDP socket with exclusive-use set pre-bind."""

    bind_host = _ipv4_host(host)
    bind_port = _exact_int(port, "UDP bind port")
    if bind_port > 65_535:
        raise ExclusiveUdpError("UDP bind port must be <= 65535")
    if not callable(socket_factory):
        raise TypeError("socket_factory must be callable")
    option = (
        getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive_option is None
        else exclusive_option
    )
    if type(option) is not int:
        raise ExclusiveUdpError("SO_EXCLUSIVEADDRUSE is unavailable")

    candidate: socket.socket | None = None
    try:
        candidate = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        if candidate.family != socket.AF_INET or candidate.type & socket.SOCK_DGRAM == 0:
            raise ExclusiveUdpError("socket factory did not return IPv4 UDP")
        reuse_before = candidate.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
        if reuse_before != 0:
            raise ExclusiveUdpError("SO_REUSEADDR was unexpectedly enabled")
        candidate.setsockopt(socket.SOL_SOCKET, option, 1)
        if candidate.getsockopt(socket.SOL_SOCKET, option) != 1:
            raise ExclusiveUdpError("SO_EXCLUSIVEADDRUSE verification failed")
        if candidate.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) != 0:
            raise ExclusiveUdpError("SO_REUSEADDR changed during exclusive setup")
        candidate.bind((bind_host, bind_port))
        actual = candidate.getsockname()
        if type(actual) is not tuple or len(actual) < 2:
            raise ExclusiveUdpError("exclusive UDP getsockname is invalid")
        actual_host = _ipv4_host(actual[0])
        actual_port = _exact_int(actual[1], "actual UDP port", minimum=1)
        if bind_port != 0 and actual_port != bind_port:
            raise ExclusiveUdpError("exclusive UDP port changed during bind")
        if actual_host != bind_host:
            raise ExclusiveUdpError("exclusive UDP host changed during bind")
        return ExclusiveUdpEndpoint(
            socket=candidate,
            requested_host=bind_host,
            requested_port=bind_port,
            actual_host=actual_host,
            actual_port=actual_port,
            exclusive_option=option,
        )
    except BaseException:
        if candidate is not None:
            try:
                candidate.close()
            except BaseException as close_exc:
                raise ExclusiveUdpError(
                    "partial exclusive UDP construction could not close its socket"
                ) from close_exc
        raise


@dataclass(frozen=True)
class ExclusiveUdpProbeProof:
    host: str
    port: int
    started_monotonic_ns: int
    ended_monotonic_ns: int
    result: str = "bound_and_closed"

    def __post_init__(self) -> None:
        _ipv4_host(self.host)
        _exact_int(self.port, "exclusive probe port", minimum=1)
        if self.port > 65_535:
            raise ValueError("exclusive probe port must be <= 65535")
        start = _exact_int(self.started_monotonic_ns, "exclusive probe start")
        end = _exact_int(self.ended_monotonic_ns, "exclusive probe end")
        if end < start:
            raise ValueError("exclusive probe end must not precede start")
        if self.result != "bound_and_closed":
            raise ValueError("exclusive probe result must be bound_and_closed")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "started_monotonic_ns": self.started_monotonic_ns,
            "ended_monotonic_ns": self.ended_monotonic_ns,
            "result": self.result,
        }


def probe_exclusive_udp_port(
    host: str,
    port: int,
    *,
    deadline_monotonic_ns: int,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    endpoint_factory: Callable[[str, int], ExclusiveUdpEndpoint] = (
        create_exclusive_udp_endpoint
    ),
) -> ExclusiveUdpProbeProof:
    """Exclusively bind and close one caller-selected nonzero IPv4 UDP port."""

    address = _ipv4_host(host)
    target = _exact_int(port, "exclusive probe port", minimum=1)
    if target > 65_535:
        raise ExclusiveUdpError("exclusive probe port must be <= 65535")
    deadline = _exact_int(deadline_monotonic_ns, "exclusive probe deadline", minimum=1)
    start = _clock_now(monotonic_ns)
    if start >= deadline:
        raise PoweredDeadlineExpired("exclusive probe deadline was reached")
    endpoint: ExclusiveUdpEndpoint | None = None
    try:
        endpoint = endpoint_factory(address, target)
        if (
            endpoint.requested_host != address
            or endpoint.requested_port != target
            or endpoint.actual_host != address
            or endpoint.actual_port != target
        ):
            raise ExclusiveUdpError("exclusive probe bound a different endpoint")
    finally:
        if endpoint is not None:
            try:
                endpoint.close()
            except BaseException as close_exc:
                raise ExclusiveUdpError(
                    "exclusive probe socket close could not be proved"
                ) from close_exc
    end = _clock_now(monotonic_ns)
    if end >= deadline:
        raise PoweredDeadlineExpired("exclusive probe exceeded its deadline")
    if endpoint is None or endpoint.closed is not True:
        raise ExclusiveUdpError("exclusive probe socket closure is unproved")
    return ExclusiveUdpProbeProof(address, target, start, end)


def normalize_ipv4_loopback_peer(value: Any) -> tuple[str, int]:
    if type(value) is not tuple or len(value) != 2:
        raise MavlinkDatagramError("UDP source must be one IPv4 host/port pair")
    host = value[0]
    port = value[1]
    if type(host) is not str:
        raise MavlinkDatagramError("UDP source host must be an exact string")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise MavlinkDatagramError("UDP source host is invalid") from exc
    if not isinstance(address, ipaddress.IPv4Address) or not address.is_loopback:
        raise MavlinkDatagramError("UDP source must be IPv4 loopback")
    try:
        source_port = _exact_int(port, "UDP source port", minimum=1)
    except ValueError as exc:
        raise MavlinkDatagramError("UDP source port is invalid") from exc
    if source_port > 65_535:
        raise MavlinkDatagramError("UDP source port must be <= 65535")
    return (str(address), source_port)


def validate_scratch_mavlink_datagram(
    datagram: bytes | bytearray | memoryview,
    *,
    parser_factory: Callable[[], Any],
) -> Any:
    """Return the sole parsed message from one exact unsigned MAVLink frame."""

    if not isinstance(datagram, (bytes, bytearray, memoryview)):
        raise MavlinkDatagramError("MAVLink datagram must be bytes-like")
    raw = bytes(datagram)
    if not raw:
        raise MavlinkDatagramError("MAVLink datagram is empty")
    marker = raw[0]
    if marker == 0xFE:
        if len(raw) < 8:
            raise MavlinkDatagramError("MAVLink v1 frame is truncated")
        payload_length = raw[1]
        expected_length = 8 + payload_length
        expected_message_id = raw[5]
    elif marker == 0xFD:
        if len(raw) < 12:
            raise MavlinkDatagramError("MAVLink v2 frame is truncated")
        payload_length = raw[1]
        incompat_flags = raw[2]
        if incompat_flags != 0:
            raise MavlinkDatagramError(
                "signed or incompatible MAVLink v2 frame is forbidden"
            )
        expected_length = 12 + payload_length
        expected_message_id = raw[7] | (raw[8] << 8) | (raw[9] << 16)
    else:
        raise MavlinkDatagramError("MAVLink frame marker is invalid")
    if len(raw) != expected_length:
        raise MavlinkDatagramError(
            "MAVLink declared frame length does not equal datagram length"
        )
    if not callable(parser_factory):
        raise TypeError("parser_factory must be callable")
    try:
        parser = parser_factory()
        parser.robust_parsing = False
        if parser.robust_parsing is not False:
            raise MavlinkDatagramError("scratch parser robust mode stayed enabled")
        parsed = parser.parse_buffer(raw)
    except MavlinkDatagramError:
        raise
    except BaseException as exc:
        raise MavlinkDatagramError("scratch MAVLink parser rejected datagram") from exc
    if type(parsed) is not list or len(parsed) != 1:
        raise MavlinkDatagramError("scratch parser did not return exactly one message")
    message = parsed[0]
    try:
        if message.get_type() == "BAD_DATA":
            raise MavlinkDatagramError("scratch parser returned BAD_DATA")
        if bytes(message.get_msgbuf()) != raw:
            raise MavlinkDatagramError(
                "parsed MAVLink message buffer does not equal datagram"
            )
        if int(message.get_msgId()) != expected_message_id:
            raise MavlinkDatagramError("parsed MAVLink message ID is inconsistent")
    except MavlinkDatagramError:
        raise
    except BaseException as exc:
        raise MavlinkDatagramError("parsed MAVLink message evidence is unavailable") from exc
    return message


@dataclass(frozen=True)
class SourceDatagramDecision:
    accepted: bool
    peer_frozen_now: bool
    rejected_source: bool
    malformed: bool
    peer: tuple[str, int] | None
    message: Any | None


class MavlinkSourceFreeze:
    """Freeze one loopback sender before production-parser mutation."""

    def __init__(self, parser_factory: Callable[[], Any]) -> None:
        if not callable(parser_factory):
            raise TypeError("parser_factory must be callable")
        self._parser_factory = parser_factory
        self._lock = threading.Lock()
        self._peer: tuple[str, int] | None = None
        self._rejected_source_count = 0
        self._malformed_count = 0
        self._source_rejected_latched = False
        self._promotion_streams: set[str] = set()

    @property
    def peer(self) -> tuple[str, int] | None:
        with self._lock:
            return self._peer

    @property
    def rejected_source_count(self) -> int:
        with self._lock:
            return self._rejected_source_count

    @property
    def malformed_count(self) -> int:
        with self._lock:
            return self._malformed_count

    @property
    def source_rejected_latched(self) -> bool:
        with self._lock:
            return self._source_rejected_latched

    @property
    def promoted(self) -> bool:
        with self._lock:
            return self._peer is not None and self._promotion_streams == PROMOTION_STREAMS

    def ingest(
        self,
        datagram: bytes | bytearray | memoryview,
        source: Any,
    ) -> SourceDatagramDecision:
        try:
            peer = normalize_ipv4_loopback_peer(source)
        except MavlinkDatagramError:
            with self._lock:
                self._rejected_source_count += 1
                self._source_rejected_latched = True
                return SourceDatagramDecision(
                    False, False, True, False, self._peer, None
                )
        with self._lock:
            if self._peer is not None and peer != self._peer:
                self._rejected_source_count += 1
                self._source_rejected_latched = True
                return SourceDatagramDecision(
                    False, False, True, False, self._peer, None
                )
        try:
            message = validate_scratch_mavlink_datagram(
                datagram, parser_factory=self._parser_factory
            )
        except MavlinkDatagramError:
            with self._lock:
                self._malformed_count += 1
                return SourceDatagramDecision(
                    False, False, False, True, self._peer, None
                )
        with self._lock:
            frozen_now = self._peer is None
            if frozen_now:
                self._peer = peer
            elif peer != self._peer:
                # A second check closes the race between parsing and latching.
                self._rejected_source_count += 1
                self._source_rejected_latched = True
                return SourceDatagramDecision(
                    False, False, True, False, self._peer, None
                )
            return SourceDatagramDecision(
                True, frozen_now, False, False, self._peer, message
            )

    def observe_fresh_stream(self, message_type: str) -> bool:
        if type(message_type) is not str or message_type not in PROMOTION_STREAMS:
            raise MavlinkDatagramError("promotion stream type is invalid")
        with self._lock:
            if self._peer is None:
                return False
            self._promotion_streams.add(message_type)
            return self._promotion_streams == PROMOTION_STREAMS

    def outbound_permitted(self, category: str) -> bool:
        if type(category) is not str or category not in PRODUCTION_OUTBOUND_CATEGORIES:
            return False
        with self._lock:
            if self._peer is None or self._source_rejected_latched:
                return False
            if category in ANNOUNCEMENT_CATEGORIES:
                return True
            return self._promotion_streams == PROMOTION_STREAMS


@dataclass(frozen=True)
class StableFileIdentity:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path:
            raise ValueError("stable file path must be nonempty")
        _exact_int(self.size_bytes, "stable file size")
        _sha256(self.sha256, "stable file SHA-256")

    def to_primitive(self, *, name: str | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        if name is not None:
            if type(name) is not str or not name:
                raise ValueError("artifact name must be nonempty")
            value = {"name": name, **value}
        return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _canonical_absolute_path(
    value: str | os.PathLike[str], label: str
) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise StableFileError(f"{label} must be path-like") from exc
    if type(raw) is not str:
        raise StableFileError(f"{label} must be a text path")
    target = Path(raw)
    if not target.is_absolute():
        raise StableFileError(f"{label} must be absolute")
    if os.path.normpath(raw) != raw or os.path.abspath(raw) != raw:
        raise StableFileError(f"{label} must be lexically canonical")
    return target


def _reject_linked_ancestry(target: Path) -> None:
    cursor = target
    while True:
        try:
            info = cursor.lstat()
        except OSError as exc:
            raise StableFileError("path component is unavailable") from exc
        if cursor.is_symlink() or _is_reparse(info):
            raise StableFileError("path traverses a link or reparse point")
        if cursor.parent == cursor:
            return
        cursor = cursor.parent


def stable_file_identity(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
    max_bytes: int = 64 * 1024 * 1024,
) -> StableFileIdentity:
    """Hash a regular file through one handle and reject observed path drift."""

    limit = _exact_int(max_bytes, "maximum file size", minimum=1)
    target = _canonical_absolute_path(path, "file path")
    root_path: Path | None = None
    if root is not None:
        root_path = _canonical_absolute_path(root, "allowed root")
        try:
            target.relative_to(root_path)
        except ValueError as exc:
            raise StableFileError("file path is outside the allowed root") from exc
    _reject_linked_ancestry(target)
    if root_path is not None:
        try:
            if target.resolve(strict=True) != target:
                raise StableFileError("file path is aliased")
            if root_path.resolve(strict=True) != root_path:
                raise StableFileError("allowed root is aliased")
        except OSError as exc:
            raise StableFileError("file path could not be resolved") from exc
    try:
        path_before = target.lstat()
    except OSError as exc:
        raise StableFileError("file is unavailable") from exc
    if target.is_symlink() or _is_reparse(path_before) or not stat.S_ISREG(path_before.st_mode):
        raise StableFileError("file must be regular and non-reparse")
    if path_before.st_size > limit:
        raise StableFileError("file exceeds its bounded size")

    digest = hashlib.sha256()
    try:
        with target.open("rb") as stream:
            opened_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_before.st_mode) or opened_before.st_size > limit:
                raise StableFileError("opened file identity is invalid")
            total = 0
            while True:
                chunk = stream.read(min(1024 * 1024, limit - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise StableFileError("file exceeded its bounded size while reading")
                digest.update(chunk)
            opened_after = os.fstat(stream.fileno())
    except StableFileError:
        raise
    except OSError as exc:
        raise StableFileError("file read failed") from exc
    try:
        path_after = target.lstat()
    except OSError as exc:
        raise StableFileError("file disappeared after reading") from exc
    identities = {
        _stat_identity(path_before),
        _stat_identity(opened_before),
        _stat_identity(opened_after),
        _stat_identity(path_after),
    }
    if len(identities) != 1 or total != opened_after.st_size:
        raise StableFileError("file identity changed while hashing")
    return StableFileIdentity(
        path=str(target),
        size_bytes=total,
        sha256=digest.hexdigest(),
    )


class Win32ProcessOperations:
    """Lazy production Win32 process, pipe, explicit-spawn, and job boundary."""

    _PROCESS_COMMAND_LINE_INFORMATION = 60
    _JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
    _EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    _CREATE_UNICODE_ENVIRONMENT = 0x00000400
    _STARTF_USESTDHANDLES = 0x00000100
    _ERROR_INSUFFICIENT_BUFFER = 122
    _ERROR_MORE_DATA = 234
    _STATUS_INFO_LENGTH_MISMATCH = 0xC0000004

    def __init__(self) -> None:
        if os.name != "nt":
            raise Win32RuntimeUnavailable(
                "Win32 process primitives require the Windows runtime"
            )
        from ctypes import wintypes

        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        self._shell32 = ctypes.WinDLL("shell32", use_last_error=True)

        class FILETIME(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD),
            ]

        class SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", wintypes.LPVOID),
                ("bInheritHandle", wintypes.BOOL),
            ]

        class STARTUPINFOW(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR),
                ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD),
                ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD),
                ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD),
                ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
                ("hStdInput", wintypes.HANDLE),
                ("hStdOutput", wintypes.HANDLE),
                ("hStdError", wintypes.HANDLE),
            ]

        class STARTUPINFOEXW(ctypes.Structure):
            _fields_ = [
                ("StartupInfo", STARTUPINFOW),
                ("lpAttributeList", wintypes.LPVOID),
            ]

        class PROCESS_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("hProcess", wintypes.HANDLE),
                ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD),
                ("dwThreadId", wintypes.DWORD),
            ]

        class UNICODE_STRING(ctypes.Structure):
            _fields_ = [
                ("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", wintypes.LPVOID),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._FILETIME = FILETIME
        self._SECURITY_ATTRIBUTES = SECURITY_ATTRIBUTES
        self._STARTUPINFOEXW = STARTUPINFOEXW
        self._PROCESS_INFORMATION = PROCESS_INFORMATION
        self._UNICODE_STRING = UNICODE_STRING
        self._JOB_EXTENDED = JOBOBJECT_EXTENDED_LIMIT_INFORMATION

        k32 = self._kernel32
        k32.GetCurrentProcessId.argtypes = []
        k32.GetCurrentProcessId.restype = wintypes.DWORD
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetProcessId.argtypes = [wintypes.HANDLE]
        k32.GetProcessId.restype = wintypes.DWORD
        k32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
        ]
        k32.GetProcessTimes.restype = wintypes.BOOL
        k32.ProcessIdToSessionId.argtypes = [
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        k32.ProcessIdToSessionId.restype = wintypes.BOOL
        k32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        k32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.GetHandleInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        k32.GetHandleInformation.restype = wintypes.BOOL
        k32.SetHandleInformation.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        k32.SetHandleInformation.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        k32.CreatePipe.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.POINTER(SECURITY_ATTRIBUTES),
            wintypes.DWORD,
        ]
        k32.CreatePipe.restype = wintypes.BOOL
        k32.PeekNamedPipe.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        k32.PeekNamedPipe.restype = wintypes.BOOL
        k32.WriteFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        k32.WriteFile.restype = wintypes.BOOL
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        k32.QueryInformationJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        k32.IsProcessInJob.restype = wintypes.BOOL
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateProcess.restype = wintypes.BOOL
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateJobObject.restype = wintypes.BOOL
        k32.InitializeProcThreadAttributeList.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        k32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
        k32.UpdateProcThreadAttribute.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.c_size_t,
            wintypes.LPVOID,
            ctypes.c_size_t,
            wintypes.LPVOID,
            wintypes.LPVOID,
        ]
        k32.UpdateProcThreadAttribute.restype = wintypes.BOOL
        k32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
        k32.DeleteProcThreadAttributeList.restype = None
        k32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPCWSTR,
            ctypes.POINTER(STARTUPINFOEXW),
            ctypes.POINTER(PROCESS_INFORMATION),
        ]
        k32.CreateProcessW.restype = wintypes.BOOL

        self._ntdll.NtQueryInformationProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.ULONG,
            ctypes.POINTER(wintypes.ULONG),
        ]
        self._ntdll.NtQueryInformationProcess.restype = ctypes.c_long
        self._shell32.CommandLineToArgvW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_int),
        ]
        self._shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
        k32.LocalFree.argtypes = [wintypes.HLOCAL]
        k32.LocalFree.restype = wintypes.HLOCAL

    @staticmethod
    def _handle(value: int) -> int:
        return _exact_int(value, "Win32 handle", minimum=1)

    @staticmethod
    def _unsigned_status(value: int) -> int:
        return int(ctypes.c_ulong(value).value)

    @staticmethod
    def _filetime(value: Any) -> int:
        return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

    @staticmethod
    def _error(label: str) -> PoweredRuntimeError:
        return PoweredRuntimeError(
            f"{label} failed with Win32 error {int(ctypes.get_last_error())}"
        )

    def current_process_id(self) -> int:
        return int(self._kernel32.GetCurrentProcessId())

    def open_process(
        self,
        pid: int,
        *,
        inheritable: bool,
        terminate_access: bool,
    ) -> int:
        access = PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE
        if terminate_access:
            access |= PROCESS_TERMINATE
        handle = self._kernel32.OpenProcess(
            access,
            bool(inheritable),
            _exact_int(pid, "process PID", minimum=1),
        )
        if not handle:
            raise self._error("OpenProcess")
        return int(handle)

    def _process_pid(self, handle: int) -> int:
        pid = int(self._kernel32.GetProcessId(self._handle(handle)))
        if pid == 0:
            raise self._error("GetProcessId")
        return pid

    def _creation_filetime(self, handle: int) -> int:
        creation = self._FILETIME()
        exit_time = self._FILETIME()
        kernel = self._FILETIME()
        user = self._FILETIME()
        if not self._kernel32.GetProcessTimes(
            self._handle(handle),
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise self._error("GetProcessTimes")
        value = self._filetime(creation)
        if value <= 0:
            raise ProcessIdentityError("process creation FILETIME is invalid")
        return value

    def _session_id(self, pid: int) -> int:
        session = self._wintypes.DWORD(0)
        if not self._kernel32.ProcessIdToSessionId(pid, ctypes.byref(session)):
            raise self._error("ProcessIdToSessionId")
        return int(session.value)

    def _image_path(self, handle: int) -> str:
        capacity = 32_768
        buffer = ctypes.create_unicode_buffer(capacity)
        size = self._wintypes.DWORD(capacity)
        if not self._kernel32.QueryFullProcessImageNameW(
            self._handle(handle), 0, buffer, ctypes.byref(size)
        ):
            raise self._error("QueryFullProcessImageNameW")
        raw = buffer.value
        if len(raw) != int(size.value) or not os.path.isabs(raw):
            raise ProcessIdentityError("process image path is not canonical absolute")
        canonical = os.path.realpath(raw)
        if os.path.normcase(canonical) != os.path.normcase(raw):
            raise ProcessIdentityError("process image path resolves through an alias")
        return canonical

    def _process_command_line(self, handle: int) -> str:
        needed = self._wintypes.ULONG(0)
        status = int(
            self._ntdll.NtQueryInformationProcess(
                self._handle(handle),
                self._PROCESS_COMMAND_LINE_INFORMATION,
                None,
                0,
                ctypes.byref(needed),
            )
        )
        if (
            self._unsigned_status(status) != self._STATUS_INFO_LENGTH_MISMATCH
            or int(needed.value) < ctypes.sizeof(self._UNICODE_STRING)
        ):
            raise ProcessIdentityError(
                "process command line size could not be queried"
            )
        buffer = ctypes.create_string_buffer(int(needed.value))
        status = int(
            self._ntdll.NtQueryInformationProcess(
                self._handle(handle),
                self._PROCESS_COMMAND_LINE_INFORMATION,
                ctypes.byref(buffer),
                int(needed.value),
                ctypes.byref(needed),
            )
        )
        if status < 0:
            raise ProcessIdentityError("process command line query failed")
        value = self._UNICODE_STRING.from_buffer(buffer)
        length = int(value.Length)
        maximum = int(value.MaximumLength)
        pointer = int(value.Buffer or 0)
        base = ctypes.addressof(buffer)
        if (
            length == 0
            or length % 2
            or maximum < length
            or pointer < base
            or pointer + length > base + ctypes.sizeof(buffer)
        ):
            raise ProcessIdentityError("process command line buffer is invalid")
        return ctypes.wstring_at(pointer, length // 2)

    def query_process_argv(self, process_handle: int) -> tuple[str, ...]:
        command_line = self._process_command_line(process_handle)
        count = ctypes.c_int(0)
        pointer = self._shell32.CommandLineToArgvW(
            command_line, ctypes.byref(count)
        )
        if not pointer or count.value <= 0:
            raise ProcessIdentityError("CommandLineToArgvW failed")
        try:
            result = tuple(pointer[index] for index in range(count.value))
        finally:
            if self._kernel32.LocalFree(pointer):
                raise HandleCloseError("CommandLineToArgvW allocation could not be freed")
        argv_sha256(result)
        return result

    def query_process_identity(
        self,
        process_handle: int,
        expected_argv: Sequence[str],
    ) -> dict[str, Any]:
        argv = tuple(expected_argv)
        expected_hash = argv_sha256(argv)

        def facts() -> tuple[int, int, int, str, tuple[str, ...]]:
            pid = self._process_pid(process_handle)
            return (
                pid,
                self._creation_filetime(process_handle),
                self._session_id(pid),
                self._image_path(process_handle),
                self.query_process_argv(process_handle),
            )

        before = facts()
        if before[4] != argv:
            raise ProcessIdentityError("retained process argv mismatched")
        image = stable_file_identity(
            before[3],
            max_bytes=MAX_PROCESS_IMAGE_BYTES,
        )
        after = facts()
        if before != after or image.path != before[3]:
            raise ProcessIdentityError("process identity changed while proving image")
        return validate_process_identity(
            {
                "pid": before[0],
                "creation_filetime_100ns": before[1],
                "windows_session_id": before[2],
                "image_path": before[3],
                "image_sha256": image.sha256,
                "argv_sha256": expected_hash,
            }
        )

    def process_signaled(self, process_handle: int) -> bool:
        result = int(
            self._kernel32.WaitForSingleObject(self._handle(process_handle), 0)
        )
        if result == WAIT_OBJECT_0:
            return True
        if result == WAIT_TIMEOUT:
            return False
        raise self._error("WaitForSingleObject(process)")

    def process_exit_code(self, process_handle: int) -> int:
        code = self._wintypes.DWORD(0)
        if not self._kernel32.GetExitCodeProcess(
            self._handle(process_handle), ctypes.byref(code)
        ):
            raise self._error("GetExitCodeProcess")
        return int(code.value)

    def handle_is_inheritable(self, handle: int) -> bool:
        flags = self._wintypes.DWORD(0)
        if not self._kernel32.GetHandleInformation(
            self._handle(handle), ctypes.byref(flags)
        ):
            raise self._error("GetHandleInformation")
        return bool(flags.value & HANDLE_FLAG_INHERIT)

    def close_handle(self, handle: int) -> None:
        if not self._kernel32.CloseHandle(self._handle(handle)):
            raise self._error("CloseHandle")

    def create_capability_pipe(self) -> tuple[int, int]:
        attributes = self._SECURITY_ATTRIBUTES()
        attributes.nLength = ctypes.sizeof(attributes)
        attributes.lpSecurityDescriptor = None
        attributes.bInheritHandle = True
        reader = self._wintypes.HANDLE()
        writer = self._wintypes.HANDLE()
        if not self._kernel32.CreatePipe(
            ctypes.byref(reader), ctypes.byref(writer), ctypes.byref(attributes), 0
        ):
            raise self._error("CreatePipe")
        read_value = int(reader.value)
        write_value = int(writer.value)
        try:
            if not self._kernel32.SetHandleInformation(
                write_value, HANDLE_FLAG_INHERIT, 0
            ):
                raise self._error("SetHandleInformation(capability writer)")
            return read_value, write_value
        except BaseException:
            close_owned_handles(
                {"capability_read": read_value, "capability_write": write_value},
                operations=self,
            )
            raise

    def pipe_available_bytes(self, read_handle: int) -> int:
        available = self._wintypes.DWORD(0)
        if not self._kernel32.PeekNamedPipe(
            self._handle(read_handle),
            None,
            0,
            None,
            ctypes.byref(available),
            None,
        ):
            raise self._error("PeekNamedPipe")
        return int(available.value)

    def write_file(self, handle: int, payload: bytes) -> int:
        if type(payload) is not bytes or not payload:
            raise CapabilityProtocolError("WriteFile payload must be nonempty bytes")
        buffer = ctypes.create_string_buffer(payload)
        written = self._wintypes.DWORD(0)
        if not self._kernel32.WriteFile(
            self._handle(handle),
            buffer,
            len(payload),
            ctypes.byref(written),
            None,
        ):
            raise self._error("WriteFile")
        return int(written.value)

    def create_non_breakaway_job(self) -> int:
        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise self._error("CreateJobObjectW")
        value = int(handle)
        info = self._JOB_EXTENDED()
        if not self._kernel32.SetInformationJobObject(
            value,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            try:
                self.close_handle(value)
            except BaseException as close_exc:
                raise HandleCloseError(
                    "failed job setup also failed to close its handle"
                ) from close_exc
            raise self._error("SetInformationJobObject")
        return value

    def spawn_process_explicit(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        environment: Mapping[str, str],
        inherited_handles: Sequence[int],
        stdin_handle: int,
        stdout_handle: int,
        stderr_handle: int,
    ) -> ProcessCreation:
        import subprocess

        values = tuple(argv)
        argv_sha256(values)
        handles = tuple(self._handle(item) for item in inherited_handles)
        if len(handles) != 5 or len(set(handles)) != 5:
            raise ChildSpawnError("Win32 explicit handle list must contain five values")
        if any(not self.handle_is_inheritable(handle) for handle in handles):
            raise ChildSpawnError("Win32 explicit handle is not inheritable")

        size = ctypes.c_size_t(0)
        ctypes.set_last_error(0)
        self._kernel32.InitializeProcThreadAttributeList(
            None, 1, 0, ctypes.byref(size)
        )
        if (
            int(ctypes.get_last_error()) != self._ERROR_INSUFFICIENT_BUFFER
            or size.value == 0
        ):
            raise self._error("InitializeProcThreadAttributeList(size)")
        attribute_buffer = ctypes.create_string_buffer(size.value)
        attribute_pointer = ctypes.cast(attribute_buffer, self._wintypes.LPVOID)
        if not self._kernel32.InitializeProcThreadAttributeList(
            attribute_pointer, 1, 0, ctypes.byref(size)
        ):
            raise self._error("InitializeProcThreadAttributeList")
        handle_array_type = self._wintypes.HANDLE * len(handles)
        handle_array = handle_array_type(*handles)
        try:
            if not self._kernel32.UpdateProcThreadAttribute(
                attribute_pointer,
                0,
                self._PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                ctypes.cast(handle_array, self._wintypes.LPVOID),
                ctypes.sizeof(handle_array),
                None,
                None,
            ):
                raise self._error("UpdateProcThreadAttribute(handle list)")

            startup = self._STARTUPINFOEXW()
            startup.StartupInfo.cb = ctypes.sizeof(startup)
            startup.StartupInfo.dwFlags = self._STARTF_USESTDHANDLES
            startup.StartupInfo.hStdInput = self._handle(stdin_handle)
            startup.StartupInfo.hStdOutput = self._handle(stdout_handle)
            startup.StartupInfo.hStdError = self._handle(stderr_handle)
            startup.lpAttributeList = attribute_pointer

            command_line = ctypes.create_unicode_buffer(
                subprocess.list2cmdline(list(values))
            )
            ordered_environment = sorted(
                environment.items(), key=lambda item: item[0].upper()
            )
            environment_block = ctypes.create_unicode_buffer(
                "\x00".join(f"{name}={value}" for name, value in ordered_environment)
                + "\x00\x00"
            )
            process_info = self._PROCESS_INFORMATION()
            flags = (
                self._EXTENDED_STARTUPINFO_PRESENT
                | self._CREATE_UNICODE_ENVIRONMENT
            )
            if not self._kernel32.CreateProcessW(
                values[0],
                command_line,
                None,
                None,
                True,
                flags,
                ctypes.cast(environment_block, self._wintypes.LPVOID),
                cwd,
                ctypes.byref(startup),
                ctypes.byref(process_info),
            ):
                raise self._error("CreateProcessW")
            return ProcessCreation(
                int(process_info.hProcess),
                int(process_info.hThread),
                int(process_info.dwProcessId),
                int(process_info.dwThreadId),
            )
        finally:
            self._kernel32.DeleteProcThreadAttributeList(attribute_pointer)

    def assign_process_to_job(self, job_handle: int, process_handle: int) -> None:
        if not self._kernel32.AssignProcessToJobObject(
            self._handle(job_handle), self._handle(process_handle)
        ):
            raise self._error("AssignProcessToJobObject")

    def query_job_limit_flags(self, job_handle: int) -> int:
        info = self._JOB_EXTENDED()
        returned = self._wintypes.DWORD(0)
        if not self._kernel32.QueryInformationJobObject(
            self._handle(job_handle),
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(returned),
        ):
            raise self._error("QueryInformationJobObject(limits)")
        if int(returned.value) != ctypes.sizeof(info):
            raise ChildSpawnError("job limit query returned a partial structure")
        return int(info.BasicLimitInformation.LimitFlags)

    def process_in_job(self, process_handle: int, job_handle: int) -> bool:
        result = self._wintypes.BOOL(False)
        if not self._kernel32.IsProcessInJob(
            self._handle(process_handle),
            self._handle(job_handle),
            ctypes.byref(result),
        ):
            raise self._error("IsProcessInJob")
        return bool(result.value)

    def job_active_pids(self, job_handle: int) -> tuple[int, ...]:
        capacity = 16
        pointer_size = ctypes.sizeof(ctypes.c_size_t)
        while capacity <= 65_536:
            buffer = ctypes.create_string_buffer(8 + capacity * pointer_size)
            returned = self._wintypes.DWORD(0)
            ctypes.set_last_error(0)
            ok = self._kernel32.QueryInformationJobObject(
                self._handle(job_handle),
                self._JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                buffer,
                ctypes.sizeof(buffer),
                ctypes.byref(returned),
            )
            assigned, included = struct.unpack_from("<II", buffer.raw, 0)
            if ok:
                if assigned != included or included > capacity:
                    raise ProcessResidueError("job PID list was incomplete")
                array_type = ctypes.c_size_t * included
                values = tuple(int(item) for item in array_type.from_buffer(buffer, 8))
                if any(value <= 0 for value in values):
                    raise ProcessResidueError("job PID list contained an invalid PID")
                return tuple(sorted(set(values)))
            error = int(ctypes.get_last_error())
            if error != self._ERROR_MORE_DATA:
                raise self._error("QueryInformationJobObject(process IDs)")
            capacity = max(capacity * 2, int(assigned), int(included) + 1)
        raise ProcessResidueError("job PID list exceeded its bounded capacity")

    def terminate_process(self, process_handle: int, exit_code: int) -> None:
        if not self._kernel32.TerminateProcess(
            self._handle(process_handle),
            _exact_int(exit_code, "process termination code"),
        ):
            raise self._error("TerminateProcess")

    def terminate_job(self, job_handle: int, exit_code: int) -> None:
        if not self._kernel32.TerminateJobObject(
            self._handle(job_handle),
            _exact_int(exit_code, "job termination code"),
        ):
            raise self._error("TerminateJobObject")


class Win32UdpOwnerTableOperations:
    """Lazy GetExtendedUdpTable owner-PID reader for IPv4 and IPv6."""

    _UDP_TABLE_OWNER_PID = 1
    _ERROR_INSUFFICIENT_BUFFER = 122

    def __init__(self) -> None:
        if os.name != "nt":
            raise Win32RuntimeUnavailable(
                "Win32 UDP owner tables require the Windows runtime"
            )
        from ctypes import wintypes

        self._wintypes = wintypes
        self._iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)

        class IPV4_ROW(ctypes.Structure):
            _fields_ = [
                ("dwLocalAddr", wintypes.DWORD),
                ("dwLocalPort", wintypes.DWORD),
                ("dwOwningPid", wintypes.DWORD),
            ]

        class IPV6_ROW(ctypes.Structure):
            _fields_ = [
                ("ucLocalAddr", ctypes.c_ubyte * 16),
                ("dwLocalScopeId", wintypes.DWORD),
                ("dwLocalPort", wintypes.DWORD),
                ("dwOwningPid", wintypes.DWORD),
            ]

        self._rows = {socket.AF_INET: IPV4_ROW, socket.AF_INET6: IPV6_ROW}
        self._iphlpapi.GetExtendedUdpTable.argtypes = [
            wintypes.LPVOID,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.BOOL,
            wintypes.ULONG,
            ctypes.c_int,
            wintypes.ULONG,
        ]
        self._iphlpapi.GetExtendedUdpTable.restype = wintypes.DWORD

    def udp_owner_rows(self, family: int) -> tuple[tuple[int, int], ...]:
        if family not in self._rows:
            raise UdpOwnershipError("UDP owner family must be IPv4 or IPv6")
        size = self._wintypes.DWORD(0)
        result = int(
            self._iphlpapi.GetExtendedUdpTable(
                None,
                ctypes.byref(size),
                True,
                family,
                self._UDP_TABLE_OWNER_PID,
                0,
            )
        )
        if result != self._ERROR_INSUFFICIENT_BUFFER or size.value < 4:
            raise UdpOwnershipError(
                f"GetExtendedUdpTable size failed with Windows status {result}"
            )
        for _attempt in range(4):
            buffer = ctypes.create_string_buffer(int(size.value))
            result = int(
                self._iphlpapi.GetExtendedUdpTable(
                    buffer,
                    ctypes.byref(size),
                    True,
                    family,
                    self._UDP_TABLE_OWNER_PID,
                    0,
                )
            )
            if result == self._ERROR_INSUFFICIENT_BUFFER:
                continue
            if result != 0:
                raise UdpOwnershipError(
                    f"GetExtendedUdpTable failed with Windows status {result}"
                )
            count = struct.unpack_from("<I", buffer.raw, 0)[0]
            row_type = self._rows[family]
            offset = 4
            required = offset + count * ctypes.sizeof(row_type)
            if required > len(buffer):
                raise UdpOwnershipError("UDP owner table buffer is truncated")
            rows: set[tuple[int, int]] = set()
            for index in range(count):
                row = row_type.from_buffer_copy(
                    buffer, offset + index * ctypes.sizeof(row_type)
                )
                port = socket.ntohs(int(row.dwLocalPort) & 0xFFFF)
                pid = int(row.dwOwningPid)
                if port <= 0 or port > 65_535:
                    raise UdpOwnershipError("UDP owner table port is invalid")
                rows.add((port, pid))
            return tuple(sorted(rows))
        raise UdpOwnershipError("UDP owner table changed beyond bounded retries")
