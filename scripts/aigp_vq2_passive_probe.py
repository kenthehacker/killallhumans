"""Leased, passive-only build-3385 preflight evidence collection.

This wrapper has no stage selector.  It can invoke only the VQ2 ``preflight``
stage with the reviewed five-second healthy dwell and approved private replay
capture.  A cross-process live mutex is held from process/port validation
through runner cleanup and post-probe port release.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))

from aigp_loop._util import (  # noqa: E402
    git_provenance,
    private_path_guard,
    strict_json_loads,
)
from aigp_loop.replay import ReplayBundleReader  # noqa: E402
from scripts.aigp_live_lease import (  # noqa: E402
    live_simulator_lease,
    load_live_lease_evidence,
)
from scripts.aigp_vq2_passive_timing import analyze_bundle  # noqa: E402


PROBE_CONTEXT_SCHEMA = "aigp-vq2-passive-probe-context/1"
CAPTURE_INVALID_SCHEMA = "aigp-vq2-passive-capture-invalid/1"
PROBE_POISON_SCHEMA = "aigp-vq2-live-probe-poison/1"
PROBE_POISON_FILENAME = "live-probe-poison.json"
EVIDENCE_ROOT = Path(
    r"C:\Users\John\aigp-evidence\2026-07-20-package3b-m1-passive-timing"
)
LAUNCHER_PATH = Path(r"C:\Users\John\AIGP\AIGP_3385\FlightSim.exe")
PAYLOAD_PATH = Path(
    r"C:\Users\John\AIGP\AIGP_3385\FlightSim\Binaries\Win64"
    r"\DCGame-Win64-Shipping.exe"
)
LAUNCHER_SHA256 = (
    "0d3217fa72e9fee847b2c154432476a687f21b79f0ab6b910728a6254b4dce32"
)
PAYLOAD_SHA256 = (
    "9064dd1547a30afea1e3fb87652cc8194c3f5af556be40629dc491bb4f681362"
)
MAVLINK_RECEIVE_PORT = 14550
CAMERA_RECEIVE_PORT = 5600
HEALTHY_DWELL_S = 5.0
RUNNER_TIMEOUT_S = 30.0


class PassiveProbeError(RuntimeError):
    """A fail-closed passive probe precondition or cleanup failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _permanently_invalidate_capture(
    root: Path, bundle_path: Path, reason: str
) -> None:
    """Make a failed requested session unambiguously unusable as a replay."""

    bounded_reason = str(reason)[:4096]
    session_marker = {
        "schema": CAPTURE_INVALID_SCHEMA,
        "invalidated_at": _utc_now(),
        "bundle_path": str(bundle_path),
        "reason": bounded_reason,
    }
    if bundle_path.exists() and not bundle_path.is_dir():
        raise PassiveProbeError("failed replay bundle path is not a directory")
    bundle_path.mkdir(exist_ok=True)
    _atomic_json(
        bundle_path / "capture-invalid.json",
        {
            "schema": "aigp-vq2-replay-invalid/1",
            "reason": bounded_reason,
            "invalidated_at": _utc_now(),
        },
    )
    _atomic_json(root / "capture-invalid.json", session_marker)


def _mark_failed_session(root: Path, bundle_path: Path, reason: str) -> None:
    """Record a failed session without mutating an already sealed bundle."""

    _atomic_json(
        root / "capture-invalid.json",
        {
            "schema": CAPTURE_INVALID_SCHEMA,
            "invalidated_at": _utc_now(),
            "bundle_path": str(bundle_path),
            "reason": str(reason)[:4096],
        },
    )


def _qpc_frequency_hz() -> int:
    if os.name != "nt":
        raise PassiveProbeError("passive FlightSim probing requires Windows")
    frequency = ctypes.c_longlong()
    if not ctypes.windll.kernel32.QueryPerformanceFrequency(  # type: ignore[attr-defined]
        ctypes.byref(frequency)
    ):
        raise PassiveProbeError("QueryPerformanceFrequency failed")
    if frequency.value <= 0:
        raise PassiveProbeError("QueryPerformanceFrequency was not positive")
    return int(frequency.value)


def _port_is_available(host: str, port: int) -> bool:
    candidate = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            candidate.setsockopt(
                socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
            )
        candidate.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        candidate.close()


def _receive_port_state() -> dict[str, bool]:
    return {
        str(MAVLINK_RECEIVE_PORT): _port_is_available(
            "127.0.0.1", MAVLINK_RECEIVE_PORT
        ),
        str(CAMERA_RECEIVE_PORT): _port_is_available(
            "0.0.0.0", CAMERA_RECEIVE_PORT
        ),
    }


def _require_receive_ports_free() -> dict[str, bool]:
    state = _receive_port_state()
    if not all(state.values()):
        raise PassiveProbeError(f"receive port is already owned: {state}")
    return state


def _powershell_process_rows() -> list[dict[str, Any]]:
    launcher = str(LAUNCHER_PATH).replace("'", "''")
    payload = str(PAYLOAD_PATH).replace("'", "''")
    script = f"""
$paths = @('{launcher}', '{payload}')
$rows = @(
  Get-CimInstance Win32_Process | Where-Object {{ $_.ExecutablePath -in $paths }} | ForEach-Object {{
    $process = Get-Process -Id $_.ProcessId -ErrorAction Stop
    [pscustomobject]@{{
      pid = [int]$_.ProcessId
      parent_pid = [int]$_.ParentProcessId
      path = [string]$_.ExecutablePath
      command_line = [string]$_.CommandLine
      creation_time = $process.StartTime.ToUniversalTime().ToString('o')
      session_id = [int]$_.SessionId
      cpu_total_ns = [int64]($process.TotalProcessorTime.Ticks * 100)
      working_set_bytes = [int64]$process.WorkingSet64
      responding = [bool]$process.Responding
      main_window_handle = [int64]$process.MainWindowHandle
      main_window_title = [string]$process.MainWindowTitle
    }}
  }}
)
ConvertTo-Json -Compress -InputObject $rows
"""
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=15.0,
        shell=False,
    )
    value = strict_json_loads(completed.stdout)
    if type(value) is dict:
        value = [value]
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise PassiveProbeError("simulator process query returned invalid JSON")
    return value


def _window_context(handle: int) -> dict[str, Any]:
    if os.name != "nt":
        raise PassiveProbeError("window inspection requires Windows")
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    foreground = int(user32.GetForegroundWindow() or 0)
    valid = bool(handle and user32.IsWindow(handle))
    return {
        "valid": valid,
        "visible": bool(user32.IsWindowVisible(handle)) if valid else False,
        "minimized": bool(user32.IsIconic(handle)) if valid else False,
        "foreground": bool(valid and foreground == handle),
        "foreground_window_handle": foreground,
    }


_PROCESS_ROW_KEYS = frozenset(
    {
        "pid",
        "parent_pid",
        "path",
        "command_line",
        "creation_time",
        "session_id",
        "cpu_total_ns",
        "working_set_bytes",
        "responding",
        "main_window_handle",
        "main_window_title",
    }
)


def _validated_process_row(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _PROCESS_ROW_KEYS:
        raise PassiveProbeError("simulator process row has missing or unknown fields")
    for key in ("pid",):
        if type(value[key]) is not int or value[key] <= 0:
            raise PassiveProbeError(f"simulator process {key} is invalid")
    for key in (
        "parent_pid",
        "session_id",
        "cpu_total_ns",
        "working_set_bytes",
        "main_window_handle",
    ):
        if type(value[key]) is not int or value[key] < 0:
            raise PassiveProbeError(f"simulator process {key} is invalid")
    for key in ("path", "command_line", "creation_time", "main_window_title"):
        if type(value[key]) is not str:
            raise PassiveProbeError(f"simulator process {key} is invalid")
    if not value["path"] or not value["creation_time"]:
        raise PassiveProbeError("simulator process path/time is empty")
    if type(value["responding"]) is not bool:
        raise PassiveProbeError("simulator process responding state is invalid")
    return dict(value)


def _validated_process_snapshot() -> dict[str, Any]:
    if _sha256_file(LAUNCHER_PATH) != LAUNCHER_SHA256:
        raise PassiveProbeError("FlightSim launcher hash mismatch")
    if _sha256_file(PAYLOAD_PATH) != PAYLOAD_SHA256:
        raise PassiveProbeError("FlightSim payload hash mismatch")
    rows = [_validated_process_row(row) for row in _powershell_process_rows()]
    by_path: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            resolved_path = Path(row["path"]).resolve(strict=True)
        except OSError as exc:
            raise PassiveProbeError("simulator process path is unavailable") from exc
        by_path.setdefault(str(resolved_path).casefold(), []).append(row)
    launcher_rows = by_path.get(
        str(LAUNCHER_PATH.resolve(strict=True)).casefold(), []
    )
    payload_rows = by_path.get(
        str(PAYLOAD_PATH.resolve(strict=True)).casefold(), []
    )
    if len(launcher_rows) != 1 or len(payload_rows) != 1:
        raise PassiveProbeError(
            "expected exactly one build-3385 launcher and payload process"
        )
    launcher = launcher_rows[0]
    payload = payload_rows[0]
    if payload["parent_pid"] != launcher["pid"]:
        raise PassiveProbeError("FlightSim payload parent is not the launcher")
    if payload["session_id"] != launcher["session_id"]:
        raise PassiveProbeError("FlightSim processes are in different sessions")
    if payload.get("responding") is not True:
        raise PassiveProbeError("FlightSim payload is not responding")
    payload = dict(payload)
    payload["window"] = _window_context(int(payload["main_window_handle"]))
    if (
        payload["window"]["valid"] is not True
        or payload["window"]["visible"] is not True
        or payload["window"]["minimized"] is not False
    ):
        raise PassiveProbeError(
            "FlightSim payload window must be valid, visible, and unminimized"
        )
    return {
        "launcher": launcher,
        "payload": payload,
        "launcher_sha256": LAUNCHER_SHA256,
        "payload_sha256": PAYLOAD_SHA256,
    }


def _require_same_process_identity(
    initial: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    for role in ("launcher", "payload"):
        first = initial.get(role)
        later = current.get(role)
        if type(first) is not dict or type(later) is not dict:
            raise PassiveProbeError("simulator process identity evidence is invalid")
        identity_keys = ("pid", "creation_time", "path", "session_id")
        if any(first.get(key) != later.get(key) for key in identity_keys):
            raise PassiveProbeError(
                f"FlightSim {role} process identity changed during passive probe"
            )
    for key in ("launcher_sha256", "payload_sha256"):
        if initial.get(key) != current.get(key):
            raise PassiveProbeError("FlightSim build identity changed during probe")


def _require_clean_expected_commit(expected_commit: str) -> dict[str, str]:
    if (
        type(expected_commit) is not str
        or len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
    ):
        raise PassiveProbeError("expected commit must be an exact SHA-1")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(_REPO),
        check=True,
        capture_output=True,
        timeout=15.0,
        shell=False,
    ).stdout
    if status:
        raise PassiveProbeError("candidate worktree is dirty")
    commit_hash, dirty_diff_hash, code_hash = git_provenance(_REPO)
    if commit_hash != expected_commit:
        raise PassiveProbeError("candidate HEAD differs from --expected-commit")
    return {
        "commit_hash": commit_hash,
        "dirty_diff_hash": dirty_diff_hash,
        "code_hash": code_hash,
    }


def _runner_command(python: Path, record: Path, bundle: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "scripts.aigp_vq2_run",
        "--stage",
        "preflight",
        "--record",
        str(record),
        "--replay-bundle",
        str(bundle),
        "--recording-approved",
        "--preflight-healthy-dwell-s",
        str(HEALTHY_DWELL_S),
    ]


def _parse_stage_result(stdout_path: Path) -> dict[str, Any]:
    value = strict_json_loads(stdout_path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise PassiveProbeError("runner stdout was not one JSON object")
    if (
        value.get("stage") != "preflight"
        or value.get("success") is not True
        or value.get("cleanup_confirmed") is not True
    ):
        raise PassiveProbeError("passive runner result was not successful and clean")
    details = value.get("details")
    if type(details) is not dict:
        raise PassiveProbeError("passive runner details are missing")
    audit = details.get("mavlink_outbound_audit")
    if (
        type(audit) is not dict
        or type(audit.get("disallowed_count")) is not int
        or audit["disallowed_count"] != 0
    ):
        raise PassiveProbeError("passive runner outbound audit was not zero")
    ingress = details.get("mavlink_ingress_stats")
    if (
        type(ingress) is not dict
        or type(ingress.get("dropped")) is not int
        or ingress["dropped"] != 0
    ):
        raise PassiveProbeError("passive runner ingress overflowed or is missing")
    capture = details.get("replay_capture")
    if (
        type(capture) is not dict
        or capture.get("complete") is not True
        or any(
            type(capture.get(key)) is not int or capture[key] != 0
            for key in ("dropped", "decoded_frames_dropped", "writer_errors")
        )
    ):
        raise PassiveProbeError("passive replay capture was not complete")
    requested_dwell = details.get("requested_healthy_dwell_s")
    observed_dwell = details.get("healthy_dwell_s")
    if (
        type(requested_dwell) not in {int, float}
        or not math.isfinite(requested_dwell)
        or float(requested_dwell) != HEALTHY_DWELL_S
        or type(observed_dwell) not in {int, float}
        or not math.isfinite(observed_dwell)
        or float(observed_dwell) < HEALTHY_DWELL_S
    ):
        raise PassiveProbeError("passive runner did not prove the requested dwell")
    return value


def _require_capture_binding(
    *,
    manifest: Mapping[str, Any],
    git_state: Mapping[str, str],
    verification: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = manifest.get("metadata")
    if type(metadata) is not dict:
        raise PassiveProbeError("replay metadata is missing")
    expected = {
        "simulator_build": "3385",
        "simulator_mode": "Training",
        "simulator_mode_basis": "operator-attested-2026-07-20",
        "stage": "preflight",
        "preflight_healthy_dwell_s": HEALTHY_DWELL_S,
        "mavlink_address": "udpin:127.0.0.1:14550",
        "commit_hash": git_state["commit_hash"],
        "dirty_diff_hash": git_state["dirty_diff_hash"],
        "code_hash": git_state["code_hash"],
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise PassiveProbeError(
                f"replay metadata {key} is not bound to probe provenance"
            )
    dataset_hash = verification.get("dataset_hash")
    if (
        type(dataset_hash) is not str
        or analysis.get("dataset_hash") != dataset_hash
        or analysis.get("code_hash") != git_state["code_hash"]
    ):
        raise PassiveProbeError("timing analysis identity differs from capture")
    checks = analysis.get("acceptance_checks")
    if type(checks) is not dict:
        raise PassiveProbeError("timing analysis acceptance checks are missing")
    required_checks = (
        "generic_passive_timing_valid",
        "capture_complete",
        "five_second_healthy_dwell",
        "camera_capture_shortfalls_zero",
        "camera_observations_at_least_140",
        "highres_imu_arrivals_at_least_600",
        "disallowed_outbound_zero",
        "ingress_queue_drops_zero",
        "ingress_queue_capacity_proved",
    )
    failed = [name for name in required_checks if checks.get(name) is not True]
    if failed:
        raise PassiveProbeError(
            "passive timing live acceptance failed: " + ", ".join(failed)
        )
    analyzer_module = sys.modules.get(analyze_bundle.__module__)
    analyzer_path = getattr(analyzer_module, "__file__", None)
    if type(analyzer_path) is not str:
        raise PassiveProbeError("passive timing analyzer source is unavailable")
    return {
        "dataset_hash": dataset_hash,
        "code_hash": git_state["code_hash"],
        "analyzer_source_sha256": _sha256_file(Path(analyzer_path)),
        "required_acceptance_checks": list(required_checks),
    }


def _require_released_lease(
    evidence: Mapping[str, Any], *, child_pid: int
) -> dict[str, Any]:
    if (
        evidence.get("phase") != "released"
        or type(evidence.get("released_wall_time_ns")) is not int
        or evidence["released_wall_time_ns"] <= 0
        or evidence.get("wrapper_pid") != os.getpid()
        or evidence.get("child_pid") != child_pid
    ):
        raise PassiveProbeError(
            "live lease does not prove this wrapper and child released cleanly"
        )
    return dict(evidence)


def _stop_child_and_prove_exit(
    child: subprocess.Popen[str],
) -> tuple[bool, list[BaseException]]:
    errors: list[BaseException] = []
    try:
        running = child.poll() is None
    except BaseException as exc:
        errors.append(exc)
        running = True
    if running:
        try:
            child.terminate()
        except BaseException as exc:
            errors.append(exc)
        try:
            child.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
        except BaseException as exc:
            errors.append(exc)
    try:
        running = child.poll() is None
    except BaseException as exc:
        errors.append(exc)
        running = True
    if running:
        try:
            child.kill()
        except BaseException as exc:
            errors.append(exc)
        try:
            child.wait(timeout=2.0)
        except BaseException as exc:
            errors.append(exc)
    exit_proved = False
    try:
        exit_proved = child.poll() is not None
        if not exit_proved:
            errors.append(
                PassiveProbeError("passive runner process termination is unproved")
            )
    except BaseException as exc:
        errors.append(exc)
    return exit_proved, errors


class _FailClosedLeaseScope:
    """Release a lease only when the owner proves release is safe."""

    def __init__(self, lease: Any, release_permitted) -> None:
        self._lease = lease
        self._release_permitted = release_permitted

    def __enter__(self):
        return self._lease.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self._release_permitted():
            return False
        try:
            self._lease.release()
        except BaseException as cleanup_exc:
            if exc is not None:
                cleanup_exc.add_note(
                    f"lease body also failed: {type(exc).__name__}: {exc}"
                )
            raise
        return False


def run_probe(
    *,
    output_dir: Path,
    python: Path,
    expected_commit: str,
    recording_approved: bool,
    training_mode_attested: bool,
) -> dict[str, Any]:
    if recording_approved is not True:
        raise PassiveProbeError("private frame recording approval is required")
    if training_mode_attested is not True:
        raise PassiveProbeError("Training-mode operator attestation is required")
    raw_output_dir = Path(output_dir)
    if not raw_output_dir.is_absolute():
        raise PassiveProbeError("output directory must be absolute")
    root = Path(os.path.abspath(raw_output_dir))
    approved_root = EVIDENCE_ROOT.resolve(strict=True)
    poison_path = approved_root / PROBE_POISON_FILENAME
    if poison_path.exists():
        raise PassiveProbeError(
            f"live probe poison marker requires human review: {poison_path}"
        )
    try:
        relative_output = root.resolve().relative_to(approved_root)
    except ValueError as exc:
        raise PassiveProbeError(
            f"output directory must be below approved evidence root {approved_root}"
        ) from exc
    if not relative_output.parts:
        raise PassiveProbeError("output directory must be a new session subdirectory")
    private_path_guard(root, _REPO)
    if not python.is_absolute():
        raise PassiveProbeError("Python executable path must be absolute")
    python = python.resolve(strict=True)
    if not python.is_file():
        raise PassiveProbeError("Python executable is not a regular file")
    root.mkdir(parents=True, exist_ok=False)

    context_path = root / "probe-context.json"
    lease_path = root / "live-lease.json"
    stdout_path = root / "runner-stdout.json"
    stderr_path = root / "runner-stderr.txt"
    record_path = root / "preflight.jsonl.gz"
    bundle_path = root / "preflight.vq2replay"
    analysis_path = root / "passive-timing-analysis.json"
    command = _runner_command(python, record_path, bundle_path)
    context: dict[str, Any] = {
        "schema": PROBE_CONTEXT_SCHEMA,
        "started_at": _utc_now(),
        "finished_at": None,
        "success": False,
        "failure": None,
        "simulator_build": "3385",
        "simulator_mode": "Training",
        "simulator_mode_basis": "operator-attested-2026-07-20",
        "capture_load": "decoded-frame-replay-enabled",
        "simulator_wall_ratio": "unmeasured",
        "qpc_frequency_hz": None,
        "git": None,
        "postcheck_git": None,
        "runner_command": command,
        "precheck_ports_free": None,
        "postcheck_ports_free": None,
        "postcheck_port_samples": [],
        "process_samples": [],
        "runner_exit_code": None,
        "stage_result": None,
        "bundle_verification": None,
        "timing_analysis": None,
        "analysis_binding": None,
        "artifacts": None,
        "lease": None,
        "lease_release_permitted": True,
    }
    child: Optional[subprocess.Popen[str]] = None
    initial_process_snapshot: Optional[dict[str, Any]] = None
    release_permitted = [True]
    try:
        lease_owner = live_simulator_lease(
            lease_path, initial_phase="precheck"
        )
        with _FailClosedLeaseScope(
            lease_owner, lambda: release_permitted[0]
        ) as lease:
            body_error: Optional[BaseException] = None
            cleanup_errors: list[BaseException] = []
            try:
                context["qpc_frequency_hz"] = _qpc_frequency_hz()
                context["git"] = _require_clean_expected_commit(expected_commit)
                initial_process_snapshot = _validated_process_snapshot()
                context["process_samples"].append(
                    {"sampled_at": _utc_now(), **initial_process_snapshot}
                )
                context["precheck_ports_free"] = _require_receive_ports_free()
                lease.heartbeat(phase="starting_child")
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                with stdout_path.open("x", encoding="utf-8") as stdout_handle, (
                    stderr_path.open("x", encoding="utf-8")
                ) as stderr_handle:
                    child = subprocess.Popen(
                        command,
                        cwd=str(_REPO),
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                        shell=False,
                        creationflags=creationflags,
                    )
                    lease.heartbeat(phase="child_running", child_pid=child.pid)
                    deadline = time.monotonic() + RUNNER_TIMEOUT_S
                    next_sample = time.monotonic() + 1.0
                    while child.poll() is None:
                        now = time.monotonic()
                        if now >= deadline:
                            raise PassiveProbeError(
                                "passive runner exceeded timeout"
                            )
                        if now >= next_sample:
                            lease.heartbeat(
                                phase="child_running", child_pid=child.pid
                            )
                            current_snapshot = _validated_process_snapshot()
                            _require_same_process_identity(
                                initial_process_snapshot, current_snapshot
                            )
                            context["process_samples"].append(
                                {"sampled_at": _utc_now(), **current_snapshot}
                            )
                            next_sample = now + 1.0
                        time.sleep(0.05)
                context["runner_exit_code"] = child.returncode
                if child.returncode != 0:
                    raise PassiveProbeError(
                        f"passive runner exited {child.returncode}"
                    )
                context["stage_result"] = _parse_stage_result(stdout_path)
                reader = ReplayBundleReader(bundle_path)
                context["bundle_verification"] = reader.verify(
                    verify_frames=True
                )
                context["timing_analysis"] = analyze_bundle(bundle_path)
                context["analysis_binding"] = _require_capture_binding(
                    manifest=reader.manifest,
                    git_state=context["git"],
                    verification=context["bundle_verification"],
                    analysis=context["timing_analysis"],
                )
                _atomic_json(analysis_path, context["timing_analysis"])
            except BaseException as exc:
                body_error = exc
            finally:
                if child is not None:
                    exit_proved, stop_errors = _stop_child_and_prove_exit(child)
                    cleanup_errors.extend(stop_errors)
                    if not exit_proved:
                        release_permitted[0] = False
                        context["lease_release_permitted"] = False
                        try:
                            _atomic_json(
                                poison_path,
                                {
                                    "schema": PROBE_POISON_SCHEMA,
                                    "created_at": _utc_now(),
                                    "session_output": str(root),
                                    "child_pid": child.pid,
                                    "reason": (
                                        "passive runner termination was not proved; "
                                        "do not run another live probe"
                                    ),
                                },
                            )
                        except BaseException as exc:
                            cleanup_errors.append(exc)
                if child is not None:
                    try:
                        context["runner_exit_code"] = child.poll()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                try:
                    lease.heartbeat(phase="postcheck")
                except BaseException as exc:
                    cleanup_errors.append(exc)
                try:
                    current_snapshot = _validated_process_snapshot()
                    if initial_process_snapshot is not None:
                        _require_same_process_identity(
                            initial_process_snapshot, current_snapshot
                        )
                    context["process_samples"].append(
                        {
                            "sampled_at": _utc_now(),
                            **current_snapshot,
                        }
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)
                try:
                    context["postcheck_git"] = _require_clean_expected_commit(
                        expected_commit
                    )
                    if context["postcheck_git"] != context["git"]:
                        raise PassiveProbeError(
                            "candidate provenance changed during passive probe"
                        )
                except BaseException as exc:
                    cleanup_errors.append(exc)
            invalidation_written = False
            if body_error is not None or cleanup_errors:
                reasons = []
                if body_error is not None:
                    reasons.append(f"{type(body_error).__name__}: {body_error}")
                reasons.extend(
                    f"{type(exc).__name__}: {exc}" for exc in cleanup_errors
                )
                try:
                    _permanently_invalidate_capture(
                        root, bundle_path, "; ".join(reasons)
                    )
                    invalidation_written = True
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                port_state = _receive_port_state()
                context["postcheck_port_samples"].append(port_state)
                context["postcheck_ports_free"] = port_state
                if not all(port_state.values()):
                    raise PassiveProbeError(
                        "receive port remained owned after passive runner: "
                        f"{port_state}"
                    )
            except BaseException as exc:
                cleanup_errors.append(exc)
                if not invalidation_written:
                    try:
                        _permanently_invalidate_capture(
                            root,
                            bundle_path,
                            f"{type(exc).__name__}: {exc}",
                        )
                        invalidation_written = True
                    except BaseException as invalidation_exc:
                        cleanup_errors.append(invalidation_exc)
                try:
                    final_port_state = _receive_port_state()
                    context["postcheck_port_samples"].append(final_port_state)
                    context["postcheck_ports_free"] = final_port_state
                    if not all(final_port_state.values()):
                        cleanup_errors.append(
                            PassiveProbeError(
                                "receive port release remains unproved after "
                                f"capture invalidation: {final_port_state}"
                            )
                        )
                except BaseException as final_port_exc:
                    cleanup_errors.append(final_port_exc)
            if body_error is not None:
                for cleanup_error in cleanup_errors:
                    body_error.add_note(
                        "post-probe cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                raise body_error
            if cleanup_errors:
                message = "; ".join(
                    f"{type(exc).__name__}: {exc}" for exc in cleanup_errors
                )
                raise PassiveProbeError(
                    f"post-probe cleanup was not proved: {message}"
                )
        if child is None:
            raise PassiveProbeError("passive runner child identity is unavailable")
        context["lease"] = _require_released_lease(
            load_live_lease_evidence(lease_path), child_pid=child.pid
        )
        context["artifacts"] = {
            "bundle_dataset_hash": context["bundle_verification"][
                "dataset_hash"
            ],
            "bundle_manifest_sha256": _sha256_file(
                bundle_path / "manifest.json"
            ),
            "bundle_records_sha256": _sha256_file(
                bundle_path / "records.jsonl"
            ),
            "runner_stdout_sha256": _sha256_file(stdout_path),
            "runner_stderr_sha256": _sha256_file(stderr_path),
            "legacy_record_sha256": _sha256_file(record_path),
            "lease_evidence_sha256": _sha256_file(lease_path),
            "timing_analysis_sha256": _sha256_file(analysis_path),
        }
        context["success"] = True
        return context
    except BaseException as exc:
        context["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if (
            context["success"] is not True
            and not (root / "capture-invalid.json").exists()
        ):
            try:
                _mark_failed_session(
                    root,
                    bundle_path,
                    context["failure"] or "passive probe did not complete",
                )
            except BaseException as exc:
                invalidation_error = (
                    "capture invalidation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                context["failure"] = (
                    f"{context['failure']}; {invalidation_error}"
                    if context["failure"]
                    else invalidation_error
                )
        context["finished_at"] = _utc_now()
        try:
            if lease_path.exists() and context["lease"] is None:
                context["lease"] = load_live_lease_evidence(lease_path)
        except BaseException as exc:
            context["failure"] = context["failure"] or (
                f"lease evidence load failed: {type(exc).__name__}: {exc}"
            )
        _atomic_json(context_path, context)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--recording-approved", action="store_true")
    parser.add_argument("--training-mode-attested", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_probe(
            output_dir=Path(args.output_dir),
            python=Path(args.python),
            expected_commit=args.expected_commit,
            recording_approved=args.recording_approved,
            training_mode_attested=args.training_mode_attested,
        )
    except BaseException as exc:
        print(f"passive probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
