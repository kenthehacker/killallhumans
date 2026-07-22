"""Pure contracts for the build-3385 powered-calibration recovery.

This module deliberately has no simulator, socket, subprocess, or filesystem
authority.  It turns the frozen I0 document into reusable JSON validators and
derivations.  Runtime code remains responsible for handle-based path,
process, ACL, reparse-point, and volume proofs; the path checks here enforce
the canonical lexical representation and frozen root relationships.
"""

from __future__ import annotations

import hashlib
import json
import math
import ntpath
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Callable


class PoweredAttemptContractError(ValueError):
    """Raised when a powered-attempt value is outside the frozen contract."""


TASK_ID = "vq2-package2-f04-powered-calibration-attempt"
SESSION_ID = "F04"
ATTEMPT_ID = "F04-A01"
HOST_CLOCK_ID = "host-perf-counter"

EXCITATION_PLAN_SCHEMA = "aigp-vq2-calibration-excitation-plan/1"
EXCITATION_PLAN_ID = "vq2-build3385-training-f04-excite-v1"
EXCITATION_PLAN_SHA256 = (
    "fae9d932e269e7de6513589d6f7bfd19862696d7222f1edad6eb3226292de773"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_DRIVE_PATH_RE = re.compile(r"^[A-Z]:\\")
_UINT8_MAX = (1 << 8) - 1
_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _fail(path: str, message: str) -> None:
    raise PoweredAttemptContractError(f"{path}: {message}")


def _json_tree(value: Any, path: str = "$", *, allow_tuple: bool = True) -> Any:
    """Return a plain JSON tree while rejecting non-JSON and nonfinite values."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail(path, "number must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                _fail(path, "object keys must be exact strings")
            if key in result:
                _fail(path, f"duplicate key {key!r}")
            result[key] = _json_tree(item, f"{path}.{key}", allow_tuple=allow_tuple)
        return result
    if type(value) is list or (allow_tuple and type(value) is tuple):
        return [
            _json_tree(item, f"{path}[{index}]", allow_tuple=allow_tuple)
            for index, item in enumerate(value)
        ]
    _fail(path, f"unsupported JSON value of type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the canonical object form (UTF-8, sorted, compact, no LF)."""

    tree = _json_tree(value)
    try:
        text = json.dumps(
            tree,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # defensive backstop
        raise PoweredAttemptContractError(f"$: cannot encode canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_file_bytes(value: Any) -> bytes:
    """Encode canonical JSON file bytes (object bytes followed by one LF)."""

    return canonical_json_bytes(value) + b"\n"


def canonical_object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_file_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_file_bytes(value)).hexdigest()


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        _fail("$", "hash input must be bytes-like")
    return hashlib.sha256(bytes(value)).hexdigest()


def _reject_constant(value: str) -> None:
    raise PoweredAttemptContractError(f"$: nonfinite JSON constant {value!r}")


def _pairs_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PoweredAttemptContractError(f"$: duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(payload: bytes | bytearray | memoryview | str) -> Any:
    """Parse UTF-8 JSON while rejecting BOMs, duplicates, and nonfinite values."""

    if type(payload) is str:
        text = payload
        if text.startswith("\ufeff"):
            _fail("$", "UTF-8 BOM is forbidden")
    elif isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload)
        if raw.startswith(b"\xef\xbb\xbf"):
            _fail("$", "UTF-8 BOM is forbidden")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PoweredAttemptContractError("$: input is not strict UTF-8") from exc
    else:
        _fail("$", "JSON input must be an exact string or bytes-like")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
    except PoweredAttemptContractError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PoweredAttemptContractError(f"$: invalid JSON: {exc}") from exc
    return _json_tree(value, allow_tuple=False)


def parse_canonical_json_bytes(
    payload: bytes | bytearray | memoryview,
    *,
    file_form: bool = False,
) -> Any:
    """Parse and require exact canonical bytes, optionally including one LF."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        _fail("$", "canonical JSON input must be bytes-like")
    raw = bytes(payload)
    if file_form:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or raw.endswith(b"\r\n"):
            _fail("$", "canonical file must end in exactly one LF")
        body = raw[:-1]
    else:
        body = raw
    value = strict_json_loads(body)
    expected = canonical_json_file_bytes(value) if file_form else canonical_json_bytes(value)
    if raw != expected:
        _fail("$", "JSON bytes are not canonical")
    return value


def defensive_copy(value: Any) -> Any:
    """Return a mutable, plain-JSON defensive copy."""

    return strict_json_loads(canonical_json_bytes(value))


def _object(value: Any, keys: Iterable[str], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(path, "must be an exact object")
    expected = frozenset(keys)
    actual = frozenset(value)
    if actual != expected:
        _fail(
            path,
            f"fields must be exact; missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}",
        )
    return value


def _array(value: Any, path: str, *, length: int | None = None) -> list[Any]:
    if type(value) is not list:
        _fail(path, "must be an exact array")
    if length is not None and len(value) != length:
        _fail(path, f"must contain exactly {length} items")
    return value


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        _fail(path, "must be an exact string")
    if nonempty and not value:
        _fail(path, "must not be empty")
    return value


def _literal(value: Any, expected: Any, path: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(path, f"must equal {expected!r}")


def _enum(value: Any, allowed: Iterable[str], path: str) -> str:
    item = _string(value, path)
    choices = frozenset(allowed)
    if item not in choices:
        _fail(path, f"must be one of {sorted(choices)!r}")
    return item


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be an exact boolean")
    return value


def _int(
    value: Any,
    path: str,
    *,
    minimum: int | None = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        _fail(path, "must be an exact integer (boolean is forbidden)")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        _fail(path, f"must be <= {maximum}")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float | int:
    if type(value) not in {int, float}:
        _fail(path, "must be a finite JSON number (boolean is forbidden)")
    if not math.isfinite(value):
        _fail(path, "must be finite")
    if positive and value <= 0:
        _fail(path, "must be positive")
    return value


def _sha256(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    item = _string(value, path)
    if _SHA256_RE.fullmatch(item) is None:
        _fail(path, "must be canonical lowercase SHA-256")
    return item


def _commit(value: Any, path: str) -> str:
    item = _string(value, path)
    if _COMMIT_RE.fullmatch(item) is None:
        _fail(path, "must be a 40-character lowercase Git commit")
    return item


def _utc(value: Any, path: str) -> str:
    item = _string(value, path)
    if _UTC_RE.fullmatch(item) is None:
        _fail(path, "must use YYYY-MM-DDTHH:MM:SS.ffffffZ")
    # Shape is authoritative; this additionally rejects impossible fields.
    from datetime import datetime

    try:
        datetime.strptime(item, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise PoweredAttemptContractError(f"{path}: invalid UTC timestamp") from exc
    return item


def _token(value: Any, path: str) -> str:
    item = _string(value, path)
    if _TOKEN_RE.fullmatch(item) is None:
        _fail(path, "must be a bounded ASCII token")
    return item


def _sanitized_text(value: Any, path: str, *, maximum_utf8_bytes: int) -> str:
    item = _string(value, path, nonempty=False)
    if len(item.encode("utf-8")) > maximum_utf8_bytes:
        _fail(path, f"must be at most {maximum_utf8_bytes} UTF-8 bytes")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in item):
        _fail(path, "must not contain control characters")
    return item


def validate_absolute_windows_path(
    value: Any,
    *,
    path: str = "$",
    root: str | None = None,
) -> str:
    """Validate canonical drive-absolute syntax and optional root containment.

    This intentionally does no filesystem lookup.  The runtime must additionally
    prove handles, volumes, ACLs, and absence of reparse points.
    """

    item = _string(value, path)
    if "\x00" in item or "/" in item:
        _fail(path, "must be a canonical Windows path without NUL or '/' aliases")
    if item.startswith(("\\\\", "\\?\\", "\\.\\")):
        _fail(path, "UNC and device aliases are forbidden")
    if _DRIVE_PATH_RE.match(item) is None:
        _fail(path, "must be drive-absolute with an uppercase drive letter")
    if ":" in item[2:]:
        _fail(path, "alternate data streams are forbidden")
    normalized = ntpath.normpath(item)
    if normalized != item:
        _fail(path, "must be lexically normalized")
    parts = item[3:].split("\\") if len(item) > 3 else []
    if any(part in {"", ".", ".."} for part in parts):
        _fail(path, "empty, dot, and parent segments are forbidden")
    if any(part.endswith((" ", ".")) for part in parts):
        _fail(path, "segments with trailing space/dot aliases are forbidden")
    if root is not None:
        checked_root = validate_absolute_windows_path(root, path=f"{path}.root")
        try:
            common = ntpath.commonpath([checked_root, item])
        except ValueError as exc:
            raise PoweredAttemptContractError(f"{path}: path crosses frozen root") from exc
        if ntpath.normcase(common) != ntpath.normcase(checked_root):
            _fail(path, "path crosses frozen root")
    return item


def _sorted_unique_strings(value: Any, path: str) -> list[str]:
    items = _array(value, path)
    checked = [_string(item, f"{path}[{index}]") for index, item in enumerate(items)]
    expected = sorted(set(checked), key=lambda item: item.encode("utf-8"))
    if checked != expected or len(expected) != len(checked):
        _fail(path, "must be unique and sorted by ordinal UTF-8 bytes")
    return checked


def _sorted_unique_objects(
    value: Any,
    path: str,
    *,
    key: str,
    validator: Callable[[Any, str], Any],
) -> list[Any]:
    items = _array(value, path)
    names: list[str] = []
    for index, item in enumerate(items):
        validator(item, f"{path}[{index}]")
        row = item
        names.append(_string(row[key], f"{path}[{index}].{key}"))
    expected = sorted(set(names), key=lambda item: item.encode("utf-8"))
    if names != expected or len(expected) != len(names):
        _fail(path, f"must be unique and sorted by {key!r} ordinal UTF-8 bytes")
    return items


def validate_identity_ref(value: Any, path: str = "$", *, root: str | None = None) -> dict[str, Any]:
    row = _object(value, {"path", "sha256"}, path)
    validate_absolute_windows_path(row["path"], path=f"{path}.path", root=root)
    _sha256(row["sha256"], f"{path}.sha256")
    return row


def validate_process_identity(value: Any, path: str = "$") -> dict[str, Any]:
    row = _object(
        value,
        {"pid", "creation_filetime_100ns", "windows_session_id", "image_path", "image_sha256", "argv_sha256"},
        path,
    )
    _int(row["pid"], f"{path}.pid", minimum=1, maximum=_UINT32_MAX)
    _int(row["creation_filetime_100ns"], f"{path}.creation_filetime_100ns", minimum=1, maximum=_UINT64_MAX)
    _int(row["windows_session_id"], f"{path}.windows_session_id", maximum=_UINT32_MAX)
    validate_absolute_windows_path(row["image_path"], path=f"{path}.image_path")
    _sha256(row["image_sha256"], f"{path}.image_sha256")
    _sha256(row["argv_sha256"], f"{path}.argv_sha256")
    return row


def validate_artifact_ref(value: Any, path: str = "$", *, root: str | None = None) -> dict[str, Any]:
    row = _object(value, {"name", "path", "size_bytes", "sha256"}, path)
    _token(row["name"], f"{path}.name")
    validate_absolute_windows_path(row["path"], path=f"{path}.path", root=root)
    _int(row["size_bytes"], f"{path}.size_bytes")
    _sha256(row["sha256"], f"{path}.sha256")
    return row


def validate_phase_deadline(
    value: Any,
    path: str = "$",
    *,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    row = _object(
        value,
        {"phase", "started_monotonic_ns", "duration_ns", "parent_deadline_monotonic_ns", "deadline_monotonic_ns"},
        path,
    )
    phase = _token(row["phase"], f"{path}.phase")
    if expected_phase is not None and phase != expected_phase:
        _fail(f"{path}.phase", f"must equal {expected_phase!r}")
    start = _int(row["started_monotonic_ns"], f"{path}.started_monotonic_ns")
    duration = _int(row["duration_ns"], f"{path}.duration_ns", minimum=1)
    parent = _int(row["parent_deadline_monotonic_ns"], f"{path}.parent_deadline_monotonic_ns")
    deadline = _int(row["deadline_monotonic_ns"], f"{path}.deadline_monotonic_ns")
    if start >= parent:
        _fail(path, "phase must start before its parent deadline")
    if deadline != min(start + duration, parent):
        _fail(f"{path}.deadline_monotonic_ns", "must equal min(start + duration, parent deadline)")
    return row


def validate_artifact_timing(value: Any, path: str = "$", *, expected_phase: str | None = None) -> dict[str, Any]:
    row = _object(
        value,
        {"phase", "started_monotonic_ns", "duration_ns", "parent_deadline_monotonic_ns", "deadline_monotonic_ns", "prepared_monotonic_ns"},
        path,
    )
    validate_phase_deadline({key: row[key] for key in row if key != "prepared_monotonic_ns"}, path, expected_phase=expected_phase)
    if expected_phase is not None and expected_phase in DEADLINE_DURATIONS_NS:
        expected_duration = DEADLINE_DURATIONS_NS[expected_phase]
        if row["duration_ns"] != expected_duration:
            _fail(f"{path}.duration_ns", f"must equal frozen phase duration {expected_duration}")
    prepared = _int(row["prepared_monotonic_ns"], f"{path}.prepared_monotonic_ns")
    if not row["started_monotonic_ns"] <= prepared < row["deadline_monotonic_ns"]:
        _fail(f"{path}.prepared_monotonic_ns", "must be at/after start and before deadline")
    return row


def validate_terminal_publication_timing(value: Any, path: str = "$", *, expected_phase: str | None = None) -> dict[str, Any]:
    return validate_artifact_timing(value, path, expected_phase=expected_phase)


DEADLINE_DURATIONS_NS = MappingProxyType(
    {
        "wrapper_total": 390000000000,
        "wrapper_live_contact_absolute_offset": 300000000000,
        "postrelease_total": 90000000000,
        "offline_precheck": 10000000000,
        "attempt_publish": 2000000000,
        "lease_acquire": 5000000000,
        "launcher_return": 60000000000,
        "topology_and_training_attestation": 30000000000,
        "prechild_identity_and_ports": 5000000000,
        "child_spawn": 3000000000,
        "child_total": 110000000000,
        "child_connect": 15000000000,
        "child_preflight": 10000000000,
        "child_reset_epoch": 20000000000,
        "child_normalize_disarmed": 2000000000,
        "child_countdown_go": 8000000000,
        "child_arm": 2000000000,
        "child_prepower_absolute_offset": 52000000000,
        "powered_stage": 5000000000,
        "child_powered_absolute_offset": 57000000000,
        "child_cleanup": 15000000000,
        "child_cleanup_absolute_offset": 72000000000,
        "child_replay_close": 35000000000,
        "child_replay_close_absolute_offset": 107000000000,
        "child_finalize": 3000000000,
        "child_exit_absolute_offset": 110000000000,
        "child_exit_proof": 3000000000,
        "parent_death_lease_takeover": 1000000000,
        "fallback_spawn": 2000000000,
        "fallback_total": 25000000000,
        "fallback_connect": 5000000000,
        "fallback_disarm": 2000000000,
        "fallback_reset_and_epoch": 15000000000,
        "fallback_finalize": 2000000000,
        "postcheck_identity_process_ports": 5000000000,
        "lease_release_and_verify": 2000000000,
        "bundle_verify": 20000000000,
        "capture_seal": 10000000000,
        "analysis": 20000000000,
        "split_publish": 5000000000,
        "terminal_publish": 5000000000,
        "poison_publish": 5000000000,
        "invalid_terminal_publish": 5000000000,
        "outbound_call": 250000000,
        "lease_heartbeat_period": 1000000000,
        "lease_heartbeat_max_gap": 1500000000,
        "poll_interval_max": 50000000,
    }
)


def validate_deadline_durations(value: Any, path: str = "$") -> dict[str, Any]:
    row = _object(value, DEADLINE_DURATIONS_NS.keys(), path)
    for name, expected in DEADLINE_DURATIONS_NS.items():
        _int(row[name], f"{path}.{name}", minimum=1)
        if row[name] != expected:
            _fail(f"{path}.{name}", f"must equal frozen value {expected}")
    return row


_PLAN_LITERAL: dict[str, Any] = {
    "schema": EXCITATION_PLAN_SCHEMA,
    "plan_id": EXCITATION_PLAN_ID,
    "stage": "calibration-excite",
    "control_period_ns": 20000000,
    "tick_count": 245,
    "nominal_end_offset_ns": 4900000000,
    "powered_hard_expiry_offset_ns": 5000000000,
    "command": {"thrust": 0.235, "yaw_rate_rad_s": 0.0},
    "segments": [
        {"segment_id": "dwell-0", "first_tick": 0, "last_tick": 29, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
        {"segment_id": "roll-positive", "first_tick": 30, "last_tick": 44, "roll_rate_rad_s": 0.08, "pitch_rate_rad_s": 0.0},
        {"segment_id": "dwell-1", "first_tick": 45, "last_tick": 53, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
        {"segment_id": "roll-negative", "first_tick": 54, "last_tick": 73, "roll_rate_rad_s": -0.06, "pitch_rate_rad_s": 0.0},
        {"segment_id": "dwell-2", "first_tick": 74, "last_tick": 85, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
        {"segment_id": "pitch-positive", "first_tick": 86, "last_tick": 105, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.07},
        {"segment_id": "dwell-3", "first_tick": 106, "last_tick": 115, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
        {"segment_id": "pitch-negative", "first_tick": 116, "last_tick": 133, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": -0.08},
        {"segment_id": "dwell-4", "first_tick": 134, "last_tick": 149, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
        {"segment_id": "coupled-1", "first_tick": 150, "last_tick": 164, "roll_rate_rad_s": 0.06, "pitch_rate_rad_s": 0.04},
        {"segment_id": "coupled-2", "first_tick": 165, "last_tick": 179, "roll_rate_rad_s": -0.06, "pitch_rate_rad_s": 0.04},
        {"segment_id": "coupled-3", "first_tick": 180, "last_tick": 194, "roll_rate_rad_s": -0.06, "pitch_rate_rad_s": -0.04},
        {"segment_id": "coupled-4", "first_tick": 195, "last_tick": 209, "roll_rate_rad_s": 0.06, "pitch_rate_rad_s": -0.04},
        {"segment_id": "dwell-final", "first_tick": 210, "last_tick": 244, "roll_rate_rad_s": 0.0, "pitch_rate_rad_s": 0.0},
    ],
}


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


FROZEN_EXCITATION_PLAN = _freeze_json(_PLAN_LITERAL)


def frozen_excitation_plan() -> dict[str, Any]:
    return defensive_copy(_PLAN_LITERAL)


def fast_excitation_plan() -> dict[str, Any]:
    """Return the compact-cycle balanced waveform for pad-safe identification."""

    plan = validate_excitation_plan(frozen_excitation_plan())
    plan["plan_id"] = "vq2-build3385-training-fast-excite-v5"
    plan["tick_count"] = 45
    plan["nominal_end_offset_ns"] = 900_000_000
    plan["powered_hard_expiry_offset_ns"] = 1_000_000_000
    segments: list[dict[str, Any]] = []
    cursor = 0

    def add(name: str, count: int, roll: float, pitch: float) -> None:
        nonlocal cursor
        segments.append(
            {
                "segment_id": name,
                "first_tick": cursor,
                "last_tick": cursor + count - 1,
                "roll_rate_rad_s": roll,
                "pitch_rate_rad_s": pitch,
            }
        )
        cursor += count

    add("dwell-0", 10, 0.0, 0.0)
    add("roll-positive", 4, 0.02, 0.0)
    add("roll-negative", 4, -0.02, 0.0)
    add("pitch-positive", 4, 0.0, 0.0175)
    add("pitch-negative", 4, 0.0, -0.0175)
    coupled = (
        (0.015, 0.0125),
        (-0.015, -0.0125),
        (0.015, -0.0125),
        (-0.015, 0.0125),
    )
    for phase, (roll, pitch) in enumerate(coupled):
        add(f"coupled-{phase}", 3, roll, pitch)
    add("dwell-final", 7, 0.0, 0.0)
    if cursor != plan["tick_count"]:
        raise AssertionError("fast excitation segments do not cover 45 ticks")
    plan["segments"] = segments
    return plan


def derive_excitation_plan(value: Any) -> dict[str, Any]:
    row = _object(
        value,
        {"schema", "plan_id", "stage", "control_period_ns", "tick_count", "nominal_end_offset_ns", "powered_hard_expiry_offset_ns", "command", "segments"},
        "$plan",
    )
    _string(row["schema"], "$plan.schema")
    _string(row["plan_id"], "$plan.plan_id")
    _string(row["stage"], "$plan.stage")
    period = _int(row["control_period_ns"], "$plan.control_period_ns", minimum=1)
    tick_count = _int(row["tick_count"], "$plan.tick_count", minimum=1)
    nominal_end = _int(row["nominal_end_offset_ns"], "$plan.nominal_end_offset_ns", minimum=1)
    hard_expiry = _int(row["powered_hard_expiry_offset_ns"], "$plan.powered_hard_expiry_offset_ns", minimum=1)
    command = _object(row["command"], {"thrust", "yaw_rate_rad_s"}, "$plan.command")
    _number(command["thrust"], "$plan.command.thrust")
    _number(command["yaw_rate_rad_s"], "$plan.command.yaw_rate_rad_s")
    segments = _array(row["segments"], "$plan.segments")
    if not segments:
        _fail("$plan.segments", "must not be empty")
    next_tick = 0
    ids: list[str] = []
    derived_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        path = f"$plan.segments[{index}]"
        item = _object(segment, {"segment_id", "first_tick", "last_tick", "roll_rate_rad_s", "pitch_rate_rad_s"}, path)
        segment_id = _token(item["segment_id"], f"{path}.segment_id")
        if segment_id in ids:
            _fail(f"{path}.segment_id", "must be unique")
        ids.append(segment_id)
        first = _int(item["first_tick"], f"{path}.first_tick")
        last = _int(item["last_tick"], f"{path}.last_tick")
        if first != next_tick or last < first:
            _fail(path, "segments must be contiguous, ordered, non-overlapping, and nonempty")
        _number(item["roll_rate_rad_s"], f"{path}.roll_rate_rad_s")
        _number(item["pitch_rate_rad_s"], f"{path}.pitch_rate_rad_s")
        count = last - first + 1
        derived_segments.append(
            {
                "segment_id": segment_id,
                "first_tick": first,
                "last_tick": last,
                "tick_count": count,
                "duration_ns": count * period,
            }
        )
        next_tick = last + 1
    if next_tick != tick_count:
        _fail("$plan.tick_count", "must equal the ticks covered by segments")
    if nominal_end != tick_count * period:
        _fail("$plan.nominal_end_offset_ns", "must equal tick_count * control_period_ns")
    if hard_expiry <= nominal_end:
        _fail("$plan.powered_hard_expiry_offset_ns", "must be later than nominal end")
    return {
        "control_period_ns": period,
        "tick_count": tick_count,
        "nominal_end_offset_ns": nominal_end,
        "powered_hard_expiry_offset_ns": hard_expiry,
        "segments": derived_segments,
    }


def validate_excitation_plan(value: Any, *, expected_sha256: str = EXCITATION_PLAN_SHA256) -> dict[str, Any]:
    derive_excitation_plan(value)
    _sha256(expected_sha256, "$expected_sha256")
    actual_hash = canonical_object_sha256(value)
    if actual_hash != expected_sha256 or expected_sha256 != EXCITATION_PLAN_SHA256:
        _fail("$plan", f"object SHA-256 must equal frozen {EXCITATION_PLAN_SHA256}")
    if value != _PLAN_LITERAL:
        _fail("$plan", "must equal the exact frozen plan literal")
    return defensive_copy(value)


def excitation_command_for_tick(tick: Any) -> dict[str, float]:
    absolute_tick = _int(tick, "$tick", maximum=_PLAN_LITERAL["tick_count"] - 1)
    for segment in _PLAN_LITERAL["segments"]:
        if segment["first_tick"] <= absolute_tick <= segment["last_tick"]:
            return {
                "roll_rate_rad_s": segment["roll_rate_rad_s"],
                "pitch_rate_rad_s": segment["pitch_rate_rad_s"],
                "yaw_rate_rad_s": _PLAN_LITERAL["command"]["yaw_rate_rad_s"],
                "thrust": _PLAN_LITERAL["command"]["thrust"],
            }
    raise AssertionError("frozen plan has an uncovered tick")


def excitation_tick(tick: Any, *, anchor_monotonic_ns: int | None = None) -> dict[str, Any]:
    absolute_tick = _int(tick, "$tick", maximum=_PLAN_LITERAL["tick_count"] - 1)
    anchor = 0 if anchor_monotonic_ns is None else _int(anchor_monotonic_ns, "$anchor_monotonic_ns")
    period = _PLAN_LITERAL["control_period_ns"]
    segment = next(item for item in _PLAN_LITERAL["segments"] if item["first_tick"] <= absolute_tick <= item["last_tick"])
    result = {
        "absolute_tick": absolute_tick,
        "segment_id": segment["segment_id"],
        "release_monotonic_ns": anchor + absolute_tick * period,
        "end_monotonic_ns": anchor + (absolute_tick + 1) * period,
        "powered_expiry_monotonic_ns": anchor + _PLAN_LITERAL["powered_hard_expiry_offset_ns"],
        "command": excitation_command_for_tick(absolute_tick),
    }
    if anchor_monotonic_ns is None:
        result = {
            "absolute_tick": result["absolute_tick"],
            "segment_id": result["segment_id"],
            "release_offset_ns": result["release_monotonic_ns"],
            "end_offset_ns": result["end_monotonic_ns"],
            "powered_expiry_offset_ns": result["powered_expiry_monotonic_ns"],
            "command": result["command"],
        }
    return result


def fast_excitation_tick(
    tick: Any,
    *,
    anchor_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    plan = fast_excitation_plan()
    absolute_tick = _int(tick, "$tick", maximum=plan["tick_count"] - 1)
    anchor = (
        0
        if anchor_monotonic_ns is None
        else _int(anchor_monotonic_ns, "$anchor_monotonic_ns")
    )
    period = plan["control_period_ns"]
    segment = next(
        item
        for item in plan["segments"]
        if item["first_tick"] <= absolute_tick <= item["last_tick"]
    )
    result = {
        "absolute_tick": absolute_tick,
        "segment_id": segment["segment_id"],
        "release_monotonic_ns": anchor + absolute_tick * period,
        "end_monotonic_ns": anchor + (absolute_tick + 1) * period,
        "powered_expiry_monotonic_ns": (
            anchor + plan["powered_hard_expiry_offset_ns"]
        ),
        "command": {
            "roll_rate_rad_s": segment["roll_rate_rad_s"],
            "pitch_rate_rad_s": segment["pitch_rate_rad_s"],
            "yaw_rate_rad_s": plan["command"]["yaw_rate_rad_s"],
            "thrust": plan["command"]["thrust"],
        },
    }
    if anchor_monotonic_ns is None:
        result = {
            "absolute_tick": result["absolute_tick"],
            "segment_id": result["segment_id"],
            "release_offset_ns": result["release_monotonic_ns"],
            "end_offset_ns": result["end_monotonic_ns"],
            "powered_expiry_offset_ns": result["powered_expiry_monotonic_ns"],
            "command": result["command"],
        }
    return result


def iter_excitation_ticks(*, anchor_monotonic_ns: int | None = None) -> Iterable[dict[str, Any]]:
    for tick in range(_PLAN_LITERAL["tick_count"]):
        yield excitation_tick(tick, anchor_monotonic_ns=anchor_monotonic_ns)


if canonical_object_sha256(_PLAN_LITERAL) != EXCITATION_PLAN_SHA256:  # import-time self-test
    raise RuntimeError("frozen excitation plan literal/hash mismatch")


EVIDENCE_ROOT = r"C:\Users\John\aigp-evidence\2026-07-22-package2-f04-powered-calibration-attempt"
_ATTEMPT_ROOT = EVIDENCE_ROOT + r"\F04-A01"
_LIVE_FREEZE_ID = (
    "vq2-package2-f04-powered-calibration-attempt-f04-a01-live-freeze"
)
_FROZEN_PATHS = MappingProxyType(
    {
        "evidence_root": EVIDENCE_ROOT,
        "live_freeze": (
            EVIDENCE_ROOT + r"\live-freeze-F04-A01.json"
        ),
        "attempt_dir": _ATTEMPT_ROOT,
        "attempt_envelope": _ATTEMPT_ROOT + r"\attempt.json",
        "training_attestation": _ATTEMPT_ROOT + r"\training-attestation.json",
        "process_proof": _ATTEMPT_ROOT + r"\process-proof.json",
        "process_final_proof": _ATTEMPT_ROOT + r"\process-final-proof.json",
        "child_authority": _ATTEMPT_ROOT + r"\child-authority.json",
        "cleanup_authority": _ATTEMPT_ROOT + r"\cleanup-authority.json",
        "child_cleanup_certificate": _ATTEMPT_ROOT + r"\child-cleanup-certificate.json",
        "fallback_cleanup_certificate": _ATTEMPT_ROOT + r"\fallback-cleanup-certificate.json",
        "lease_directory": _ATTEMPT_ROOT + r"\lease",
        "lease_final": _ATTEMPT_ROOT + r"\live-lease.json",
        "wrapper_ledger_directory": _ATTEMPT_ROOT + r"\wrapper-ledger",
        "wrapper_lifecycle": _ATTEMPT_ROOT + r"\wrapper-lifecycle.json",
        "runner_stdout": _ATTEMPT_ROOT + r"\runner-stdout.json",
        "runner_stderr": _ATTEMPT_ROOT + r"\runner-stderr.txt",
        "legacy_record": _ATTEMPT_ROOT + r"\session.jsonl.gz",
        "replay_bundle": _ATTEMPT_ROOT + r"\session.vq2replay",
        "bundle_verification": _ATTEMPT_ROOT + r"\bundle-verification.json",
        "capture_seal": _ATTEMPT_ROOT + r"\capture-seal.json",
        "analysis_report": _ATTEMPT_ROOT + r"\analysis.json",
        "split_claim": _ATTEMPT_ROOT + r"\split-claim.json",
        "split_registry": EVIDENCE_ROOT + r"\split-registry\registry-000001.json",
        "attempt_complete": _ATTEMPT_ROOT + r"\attempt-complete.json",
        "attempt_invalid": _ATTEMPT_ROOT + r"\attempt-invalid.json",
        "cleanup_stdout": _ATTEMPT_ROOT + r"\cleanup-stdout.json",
        "cleanup_stderr": _ATTEMPT_ROOT + r"\cleanup-stderr.txt",
        "live_poison": EVIDENCE_ROOT + r"\live-poison.json",
    }
)


def frozen_paths() -> dict[str, str]:
    return dict(_FROZEN_PATHS)


def validate_frozen_paths(value: Any, path: str = "$") -> dict[str, Any]:
    row = _object(value, _FROZEN_PATHS.keys(), path)
    for name, expected in _FROZEN_PATHS.items():
        validate_absolute_windows_path(row[name], path=f"{path}.{name}", root=EVIDENCE_ROOT)
        if row[name] != expected:
            _fail(f"{path}.{name}", f"must equal frozen path {expected!r}")
    return row


def _schema_identity(
    value: Any,
    path: str,
    *,
    expected_schema: str,
    plan: bool = False,
) -> dict[str, Any]:
    keys = {"schema", "path", "sha256"} | ({"plan_id"} if plan else set())
    row = _object(value, keys, path)
    _literal(row["schema"], expected_schema, f"{path}.schema")
    validate_absolute_windows_path(row["path"], path=f"{path}.path")
    _sha256(row["sha256"], f"{path}.sha256")
    if plan:
        _literal(row["plan_id"], EXCITATION_PLAN_ID, f"{path}.plan_id")
        if row["sha256"] != EXCITATION_PLAN_SHA256:
            _fail(f"{path}.sha256", "must bind the frozen excitation plan")
    return row


def _path_hash_identity(value: Any, path: str, *, plan: bool = False) -> dict[str, Any]:
    keys = {"path", "sha256"} | ({"plan_id"} if plan else set())
    row = _object(value, keys, path)
    validate_absolute_windows_path(row["path"], path=f"{path}.path")
    _sha256(row["sha256"], f"{path}.sha256")
    if plan:
        _literal(row["plan_id"], EXCITATION_PLAN_ID, f"{path}.plan_id")
        if row["sha256"] != EXCITATION_PLAN_SHA256:
            _fail(f"{path}.sha256", "must bind the frozen excitation plan")
    return row


def _string_array(value: Any, path: str, *, nonempty: bool = False) -> list[str]:
    items = _array(value, path)
    if nonempty and not items:
        _fail(path, "must not be empty")
    for index, item in enumerate(items):
        _string(item, f"{path}[{index}]")
    return items


def _validate_bind(value: Any, path: str, *, host: str, port: int) -> dict[str, Any]:
    row = _object(value, {"host", "port", "socket_policy"}, path)
    _literal(row["host"], host, f"{path}.host")
    _literal(row["port"], port, f"{path}.port")
    _literal(row["socket_policy"], "ipv4-exclusive-address-use", f"{path}.socket_policy")
    return row


def validate_live_freeze(
    value: Any,
    *,
    implementation_inventory: Any | None = None,
    environment_inventory: Any | None = None,
    import_inventory: Any | None = None,
) -> dict[str, Any]:
    path = "$live_freeze"
    row = _object(
        value,
        {"schema", "task_id", "freeze_id", "candidate", "session", "inputs", "runtime", "simulator", "transport", "execution", "paths", "deadline_durations_ns"},
        path,
    )
    _literal(row["schema"], "aigp-vq2-powered-calibration-live-freeze/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["freeze_id"], _LIVE_FREEZE_ID, f"{path}.freeze_id")

    candidate = _object(
        row["candidate"],
        {"commit", "code_sha256", "live_worktree", "detached_head_required", "clean_tracked_untracked_ignored_required", "implementation_inventory"},
        f"{path}.candidate",
    )
    _commit(candidate["commit"], f"{path}.candidate.commit")
    _sha256(candidate["code_sha256"], f"{path}.candidate.code_sha256")
    validate_absolute_windows_path(candidate["live_worktree"], path=f"{path}.candidate.live_worktree")
    _literal(
        candidate["live_worktree"],
        r"C:\Users\John\aigp-worktrees\wt-package2-f04-powered-calibration-attempt-live",
        f"{path}.candidate.live_worktree",
    )
    _literal(candidate["detached_head_required"], True, f"{path}.candidate.detached_head_required")
    _literal(candidate["clean_tracked_untracked_ignored_required"], True, f"{path}.candidate.clean_tracked_untracked_ignored_required")
    validate_identity_ref(candidate["implementation_inventory"], f"{path}.candidate.implementation_inventory")
    if implementation_inventory is not None:
        inventory = validate_implementation_inventory(implementation_inventory)
        semantic = {name: inventory[name] for name in ("commit", "tree", "entries")}
        if candidate["implementation_inventory"]["sha256"] != canonical_file_sha256(inventory):
            _fail(f"{path}.candidate.implementation_inventory.sha256", "does not bind supplied implementation inventory file")
        if candidate["commit"] != inventory["commit"] or candidate["code_sha256"] != canonical_object_sha256(semantic):
            _fail(f"{path}.candidate", "commit/code hash does not bind supplied implementation inventory")

    session = _object(row["session"], {"session_id", "attempt_id", "attempt_limit", "split"}, f"{path}.session")
    _literal(session["session_id"], SESSION_ID, f"{path}.session.session_id")
    _literal(session["attempt_id"], ATTEMPT_ID, f"{path}.session.attempt_id")
    _literal(session["attempt_limit"], 1, f"{path}.session.attempt_limit")
    _literal(session["split"], "discovery_fit", f"{path}.session.split")

    inputs = _object(row["inputs"], {"target_config", "capture_authorization", "excitation_plan"}, f"{path}.inputs")
    _schema_identity(inputs["target_config"], f"{path}.inputs.target_config", expected_schema="aigp-vq2-sim-calibration-collection-config/1")
    _schema_identity(inputs["capture_authorization"], f"{path}.inputs.capture_authorization", expected_schema="aigp-vq2-simulation-capture-authorization/1")
    _schema_identity(inputs["excitation_plan"], f"{path}.inputs.excitation_plan", expected_schema=EXCITATION_PLAN_SCHEMA, plan=True)

    runtime = _object(row["runtime"], {"python", "powershell", "development_test_lock", "environment_inventory", "import_inventory"}, f"{path}.runtime")
    python = _object(runtime["python"], {"path", "implementation", "version", "sha256"}, f"{path}.runtime.python")
    validate_absolute_windows_path(python["path"], path=f"{path}.runtime.python.path")
    _literal(python["implementation"], "CPython", f"{path}.runtime.python.implementation")
    _literal(python["version"], "3.12.2", f"{path}.runtime.python.version")
    _sha256(python["sha256"], f"{path}.runtime.python.sha256")
    powershell = _object(runtime["powershell"], {"path", "product_version", "sha256"}, f"{path}.runtime.powershell")
    validate_absolute_windows_path(powershell["path"], path=f"{path}.runtime.powershell.path")
    _string(powershell["product_version"], f"{path}.runtime.powershell.product_version")
    _sha256(powershell["sha256"], f"{path}.runtime.powershell.sha256")
    for name in ("development_test_lock", "environment_inventory", "import_inventory"):
        validate_identity_ref(runtime[name], f"{path}.runtime.{name}")

    simulator = _object(row["simulator"], {"build", "mode", "launcher_script", "launcher", "payload", "topology", "mode_evidence"}, f"{path}.simulator")
    _literal(simulator["build"], 3385, f"{path}.simulator.build")
    _literal(simulator["mode"], "Training", f"{path}.simulator.mode")
    for name in ("launcher_script", "launcher", "payload"):
        validate_identity_ref(simulator[name], f"{path}.simulator.{name}")
    expected_launcher_script = candidate["live_worktree"] + r"\scripts\launch_sim.ps1"
    if simulator["launcher_script"]["path"] != expected_launcher_script:
        _fail(f"{path}.simulator.launcher_script.path", "must name launch_sim.ps1 in the frozen live worktree")
    expected_launcher = r"C:\Users\John\AIGP\AIGP_3385\FlightSim.exe"
    if simulator["launcher"]["path"] != expected_launcher:
        _fail(f"{path}.simulator.launcher.path", "must equal the frozen FlightSim launcher path")
    if simulator["payload"]["path"] == simulator["launcher"]["path"]:
        _fail(f"{path}.simulator.payload.path", "payload process image must differ from launcher image")
    _literal(simulator["topology"], "one_launcher_parent_retained_one_payload_child", f"{path}.simulator.topology")
    _literal(simulator["mode_evidence"], "post_topology_local_interactive_attestation", f"{path}.simulator.mode_evidence")

    transport = _object(row["transport"], {"mavlink_bind", "camera_bind", "peer_policy", "allowed_outbound_categories", "unknown_category_policy"}, f"{path}.transport")
    _validate_bind(transport["mavlink_bind"], f"{path}.transport.mavlink_bind", host="127.0.0.1", port=14550)
    _validate_bind(transport["camera_bind"], f"{path}.transport.camera_bind", host="0.0.0.0", port=5600)
    _literal(transport["peer_policy"], "freeze_first_valid_build3385_source", f"{path}.transport.peer_policy")
    categories = _sorted_unique_strings(transport["allowed_outbound_categories"], f"{path}.transport.allowed_outbound_categories")
    if categories != ["arm", "attitude_target", "disarm", "gcs_heartbeat", "sim_reset", "timesync"]:
        _fail(f"{path}.transport.allowed_outbound_categories", "must equal the frozen allowlist")
    _literal(transport["unknown_category_policy"], "invalidate", f"{path}.transport.unknown_category_policy")

    execution = _object(row["execution"], {"wrapper_cwd", "security_environment", "launcher_cwd", "launcher_argv", "launcher_environment_sha256", "child_cwd", "cleanup_cwd"}, f"{path}.execution")
    for name in ("wrapper_cwd", "launcher_cwd", "child_cwd", "cleanup_cwd"):
        validate_absolute_windows_path(execution[name], path=f"{path}.execution.{name}")
        if execution[name] != candidate["live_worktree"]:
            _fail(f"{path}.execution.{name}", "must equal candidate.live_worktree")
    security = _object(execution["security_environment"], {"PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "forbidden_defined"}, f"{path}.execution.security_environment")
    _literal(security["PYTHONNOUSERSITE"], "1", f"{path}.execution.security_environment.PYTHONNOUSERSITE")
    _literal(security["PYTHONDONTWRITEBYTECODE"], "1", f"{path}.execution.security_environment.PYTHONDONTWRITEBYTECODE")
    forbidden = _sorted_unique_strings(security["forbidden_defined"], f"{path}.execution.security_environment.forbidden_defined")
    if forbidden != ["PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"]:
        _fail(f"{path}.execution.security_environment.forbidden_defined", "must equal the frozen forbidden-variable list")
    launcher_argv = _string_array(execution["launcher_argv"], f"{path}.execution.launcher_argv", nonempty=True)
    expected_argv = [
        powershell["path"], "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
        simulator["launcher_script"]["path"], "-SimulatorPath", expected_launcher,
        "-TaskName", "AIGP-P2-F04-A01-Launch", "-StartupTimeoutSeconds", "25",
    ]
    if launcher_argv != expected_argv:
        _fail(f"{path}.execution.launcher_argv", "must equal the sole frozen launcher argv")
    _sha256(execution["launcher_environment_sha256"], f"{path}.execution.launcher_environment_sha256")
    if environment_inventory is not None:
        inventory = validate_environment_inventory(environment_inventory)
        if runtime["environment_inventory"]["sha256"] != canonical_file_sha256(
            inventory
        ):
            _fail(
                f"{path}.runtime.environment_inventory.sha256",
                "does not bind supplied environment inventory file",
            )
        if execution["launcher_environment_sha256"] != environment_variables_sha256(
            inventory
        ):
            _fail(
                f"{path}.execution.launcher_environment_sha256",
                "does not bind supplied environment inventory variables",
            )
    if import_inventory is not None:
        inventory = validate_import_inventory(import_inventory)
        if runtime["import_inventory"]["sha256"] != canonical_file_sha256(
            inventory
        ):
            _fail(
                f"{path}.runtime.import_inventory.sha256",
                "does not bind supplied import inventory file",
            )
        if inventory["python_sha256"] != runtime["python"]["sha256"]:
            _fail(
                f"{path}.runtime.import_inventory.sha256",
                "inventory Python does not bind the frozen runtime",
            )
    validate_frozen_paths(row["paths"], f"{path}.paths")
    validate_deadline_durations(row["deadline_durations_ns"], f"{path}.deadline_durations_ns")
    return defensive_copy(row)


def validate_implementation_inventory(value: Any) -> dict[str, Any]:
    path = "$implementation_inventory"
    row = _object(value, {"schema", "commit", "tree", "entries"}, path)
    _literal(row["schema"], "aigp-vq2-powered-implementation-inventory/1", f"{path}.schema")
    _commit(row["commit"], f"{path}.commit")
    _commit(row["tree"], f"{path}.tree")
    entries = _array(row["entries"], f"{path}.entries")
    paths: list[str] = []
    for index, entry in enumerate(entries):
        item_path = f"{path}.entries[{index}]"
        item = _object(entry, {"path", "size_bytes", "sha256"}, item_path)
        relative = _string(item["path"], f"{item_path}.path")
        if "\\" in relative or relative.startswith("/") or any(part in {"", ".", ".."} for part in relative.split("/")):
            _fail(f"{item_path}.path", "must be a canonical repository-relative Git path")
        paths.append(relative)
        _int(item["size_bytes"], f"{item_path}.size_bytes")
        _sha256(item["sha256"], f"{item_path}.sha256")
    if paths != sorted(set(paths), key=lambda item: item.encode("utf-8")):
        _fail(f"{path}.entries", "must be unique and sorted by repository path")
    return defensive_copy(row)


def validate_environment_inventory(value: Any) -> dict[str, Any]:
    path = "$environment_inventory"
    row = _object(value, {"schema", "created_at_utc", "variables"}, path)
    _literal(
        row["schema"],
        "aigp-vq2-powered-environment-inventory/1",
        f"{path}.schema",
    )
    _utc(row["created_at_utc"], f"{path}.created_at_utc")
    variables = _array(row["variables"], f"{path}.variables")
    names: list[str] = []
    for index, variable in enumerate(variables):
        item_path = f"{path}.variables[{index}]"
        item = _object(
            variable,
            {"name", "defined", "value_sha256"},
            item_path,
        )
        name = _string(item["name"], f"{item_path}.name")
        if name != name.upper():
            _fail(f"{item_path}.name", "must use its uppercase Windows name")
        _literal(item["defined"], True, f"{item_path}.defined")
        _sha256(item["value_sha256"], f"{item_path}.value_sha256")
        names.append(name)
    folded = [name.casefold() for name in names]
    expected = sorted(
        names,
        key=lambda name: (name.casefold(), name.encode("utf-8")),
    )
    if names != expected or len(set(folded)) != len(folded):
        _fail(
            f"{path}.variables",
            "must be case-insensitively unique and sorted by name",
        )
    return defensive_copy(row)


def environment_variables_sha256(value: Any) -> str:
    """Hash the validated, timestamp-independent environment semantics."""

    inventory = validate_environment_inventory(value)
    return canonical_object_sha256({"variables": inventory["variables"]})


IMPORT_INVENTORY_SEEDS = (
    "scripts.aigp_vq2_powered_attempt",
    "scripts.aigp_vq2_powered_calibration_analysis",
    "scripts.aigp_vq2_powered_calibration_probe",
    "scripts.aigp_vq2_powered_cleanup",
    "scripts.aigp_vq2_powered_runtime",
    "scripts.aigp_vq2_run",
)
if IMPORT_INVENTORY_SEEDS != tuple(
    sorted(set(IMPORT_INVENTORY_SEEDS), key=lambda item: item.encode("utf-8"))
):  # pragma: no cover - import-time code-owned invariant
    raise RuntimeError("IMPORT_INVENTORY_SEEDS must be unique and ordinal-sorted")
_MODULE_NAME_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
RUNTIME_IMPORT_MODULES = (
    "cv2.utils.fs",
    "cv2.utils.logging",
    "cv2.utils.nested",
    "typing.io",
    "typing.re",
)
if RUNTIME_IMPORT_MODULES != tuple(
    sorted(set(RUNTIME_IMPORT_MODULES), key=lambda item: item.encode("utf-8"))
):  # pragma: no cover - import-time code-owned invariant
    raise RuntimeError("RUNTIME_IMPORT_MODULES must be unique and ordinal-sorted")


def validate_import_inventory(value: Any) -> dict[str, Any]:
    path = "$import_inventory"
    row = _object(
        value,
        {"schema", "python_sha256", "seeds", "entries"},
        path,
    )
    _literal(
        row["schema"],
        "aigp-vq2-powered-import-inventory/1",
        f"{path}.schema",
    )
    _sha256(row["python_sha256"], f"{path}.python_sha256")
    seeds = _sorted_unique_strings(row["seeds"], f"{path}.seeds")
    if seeds != list(IMPORT_INVENTORY_SEEDS):
        _fail(f"{path}.seeds", "must equal the frozen import seed array")
    entries = _array(row["entries"], f"{path}.entries")
    module_names: list[str] = []
    for index, entry in enumerate(entries):
        item_path = f"{path}.entries[{index}]"
        item = _object(
            entry,
            {
                "module",
                "origin",
                "size_bytes",
                "sha256",
                "root_class",
                "namespace_roots",
            },
            item_path,
        )
        module = _string(item["module"], f"{item_path}.module")
        if _MODULE_NAME_RE.fullmatch(module) is None:
            _fail(f"{item_path}.module", "must be a canonical Python module name")
        root_class = _enum(
            item["root_class"],
            {
                "candidate",
                "venv",
                "stdlib",
                "builtin",
                "frozen",
                "namespace",
                "runtime",
            },
            f"{item_path}.root_class",
        )
        namespace_roots = _array(
            item["namespace_roots"], f"{item_path}.namespace_roots"
        )
        if root_class in {"candidate", "venv", "stdlib"}:
            validate_absolute_windows_path(
                item["origin"], path=f"{item_path}.origin"
            )
            _int(item["size_bytes"], f"{item_path}.size_bytes")
            _sha256(item["sha256"], f"{item_path}.sha256")
            if namespace_roots:
                _fail(
                    f"{item_path}.namespace_roots",
                    "must be empty for a file-backed module",
                )
        elif root_class == "runtime":
            if module not in RUNTIME_IMPORT_MODULES:
                _fail(
                    f"{item_path}.module",
                    "is not an allowed spec-less runtime entry",
                )
            validate_absolute_windows_path(
                item["origin"], path=f"{item_path}.origin"
            )
            _int(item["size_bytes"], f"{item_path}.size_bytes")
            _sha256(item["sha256"], f"{item_path}.sha256")
            if namespace_roots:
                _fail(
                    f"{item_path}.namespace_roots",
                    "must be empty for a spec-less runtime entry",
                )
        else:
            for name in ("origin", "size_bytes", "sha256"):
                if item[name] is not None:
                    _fail(
                        f"{item_path}.{name}",
                        "must be null for a non-file-backed module",
                    )
            if root_class == "namespace":
                if not namespace_roots:
                    _fail(
                        f"{item_path}.namespace_roots",
                        "must be nonempty for a namespace package",
                    )
                checked_roots = [
                    validate_absolute_windows_path(
                        root,
                        path=f"{item_path}.namespace_roots[{root_index}]",
                    )
                    for root_index, root in enumerate(namespace_roots)
                ]
                expected_roots = sorted(
                    set(checked_roots), key=lambda root: root.encode("utf-8")
                )
                if checked_roots != expected_roots:
                    _fail(
                        f"{item_path}.namespace_roots",
                        "must be unique and sorted by ordinal UTF-8 bytes",
                    )
            elif namespace_roots:
                _fail(
                    f"{item_path}.namespace_roots",
                    "must be empty outside namespace packages",
                )
        module_names.append(module)
    expected_modules = sorted(
        set(module_names), key=lambda module: module.encode("utf-8")
    )
    if module_names != expected_modules:
        _fail(f"{path}.entries", "must be unique and sorted by module")
    return defensive_copy(row)


_CAPABILITY_DOMAINS = frozenset(
    {"aigp-vq2-lease-owner/1", "aigp-vq2-powered-child/1", "aigp-vq2-powered-cleanup/1"}
)


def derive_capability_sha256(domain: Any, context_sha256: Any, secret32: Any) -> str:
    domain_value = _enum(domain, _CAPABILITY_DOMAINS, "$domain")
    context = _sha256(context_sha256, "$context_sha256")
    if not isinstance(secret32, (bytes, bytearray, memoryview)) or len(secret32) != 32:
        _fail("$secret32", "must be exactly 32 bytes")
    assert context is not None
    payload = domain_value.encode("utf-8") + b"\x00" + bytes.fromhex(context) + b"\x00" + bytes(secret32)
    return hashlib.sha256(payload).hexdigest()


def encode_capability_frame(secret32: Any) -> bytes:
    if not isinstance(secret32, (bytes, bytearray, memoryview)) or len(secret32) != 32:
        _fail("$secret32", "must be exactly 32 bytes")
    return struct.pack("<I", 32) + bytes(secret32)


def decode_capability_frame(frame: Any) -> bytes:
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        _fail("$frame", "must be bytes-like")
    raw = bytes(frame)
    if len(raw) != 36 or raw[:4] != struct.pack("<I", 32):
        _fail("$frame", "must be uint32-le(32) followed by exactly 32 bytes")
    return raw[4:]


def _validate_wrapper_absolute_deadlines(value: Any, path: str) -> dict[str, Any]:
    row = _object(value, {"started_monotonic_ns", "live_contact_deadline_monotonic_ns", "total_deadline_monotonic_ns"}, path)
    start = _int(row["started_monotonic_ns"], f"{path}.started_monotonic_ns")
    live = _int(row["live_contact_deadline_monotonic_ns"], f"{path}.live_contact_deadline_monotonic_ns")
    total = _int(row["total_deadline_monotonic_ns"], f"{path}.total_deadline_monotonic_ns")
    if live != start + DEADLINE_DURATIONS_NS["wrapper_live_contact_absolute_offset"]:
        _fail(f"{path}.live_contact_deadline_monotonic_ns", "must equal wrapper start + 300 seconds")
    if total != start + DEADLINE_DURATIONS_NS["wrapper_total"]:
        _fail(f"{path}.total_deadline_monotonic_ns", "must equal wrapper start + 390 seconds")
    return row


def _positive_decimal(value: str, path: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        _fail(path, "must be a canonical positive base-10 integer")
    return int(value)


def _validate_attempt_role_argv(
    value: Any,
    path: str,
    *,
    role: str,
    context: dict[str, Any],
    python_path: str | None,
) -> tuple[int, int]:
    argv = _string_array(value, path, nonempty=True)
    wrapper = context["wrapper_process"]
    paths = context["paths"]
    if role == "powered_child":
        expected = [
            python_path, "-E", "-s", "-B", "-m", "scripts.aigp_vq2_run",
            "--stage", "calibration-excite", "--powered-attempt-envelope", paths["attempt_envelope"],
            "--wrapper-process", f"{wrapper['pid']}:{wrapper['creation_filetime_100ns']}",
            "--powered-process-authority", paths["child_authority"],
            "--attempt-capability-handle", None, "--parent-liveness-handle", None,
            "--record", paths["legacy_record"], "--replay-bundle", paths["replay_bundle"],
            "--cleanup-certificate", paths["child_cleanup_certificate"], "--recording-approved",
        ]
        capability_index, parent_index = 15, 17
    else:
        expected = [
            python_path, "-E", "-s", "-B", "-m", "scripts.aigp_vq2_powered_cleanup",
            "--powered-attempt-envelope", paths["attempt_envelope"],
            "--wrapper-process", f"{wrapper['pid']}:{wrapper['creation_filetime_100ns']}",
            "--powered-process-authority", paths["cleanup_authority"],
            "--cleanup-capability-handle", None, "--parent-liveness-handle", None,
            "--cleanup-certificate", paths["fallback_cleanup_certificate"],
        ]
        capability_index, parent_index = 13, 15
    if len(argv) != len(expected):
        _fail(path, f"must contain exactly {len(expected)} frozen arguments")
    for index, expected_item in enumerate(expected):
        if index in {capability_index, parent_index}:
            continue
        if expected_item is None:
            if python_path is not None or index != 0:
                raise AssertionError("unexpected wildcard in frozen argv")
            validate_absolute_windows_path(argv[index], path=f"{path}[{index}]")
        elif argv[index] != expected_item:
            _fail(f"{path}[{index}]", f"must equal {expected_item!r}")
    capability_handle = _positive_decimal(argv[capability_index], f"{path}[{capability_index}]")
    parent_handle = _positive_decimal(argv[parent_index], f"{path}[{parent_index}]")
    if capability_handle == parent_handle:
        _fail(path, "capability and parent-liveness handles must differ")
    return capability_handle, parent_handle


def validate_attempt(value: Any, *, live_freeze: Any | None = None) -> dict[str, Any]:
    path = "$attempt"
    row = _object(value, {"schema", "context", "context_sha256", "capabilities"}, path)
    _literal(row["schema"], "aigp-vq2-powered-calibration-attempt/1", f"{path}.schema")
    context = _object(
        row["context"],
        {"task_id", "session_id", "attempt_id", "created_at_utc", "host", "live_freeze", "candidate_commit", "target_config", "capture_authorization", "excitation_plan", "wrapper_process", "paths", "child_argv", "cleanup_argv", "deadline_durations_ns", "wrapper_absolute_deadlines", "prepublication_timing"},
        f"{path}.context",
    )
    _literal(context["task_id"], TASK_ID, f"{path}.context.task_id")
    _literal(context["session_id"], SESSION_ID, f"{path}.context.session_id")
    _literal(context["attempt_id"], ATTEMPT_ID, f"{path}.context.attempt_id")
    _utc(context["created_at_utc"], f"{path}.context.created_at_utc")
    host = _object(context["host"], {"host_clock_id", "host_boot_id_sha256", "qpc_frequency_hz"}, f"{path}.context.host")
    _literal(host["host_clock_id"], HOST_CLOCK_ID, f"{path}.context.host.host_clock_id")
    _sha256(host["host_boot_id_sha256"], f"{path}.context.host.host_boot_id_sha256")
    _int(host["qpc_frequency_hz"], f"{path}.context.host.qpc_frequency_hz", minimum=1)
    _path_hash_identity(context["live_freeze"], f"{path}.context.live_freeze")
    _commit(context["candidate_commit"], f"{path}.context.candidate_commit")
    _path_hash_identity(context["target_config"], f"{path}.context.target_config")
    _path_hash_identity(context["capture_authorization"], f"{path}.context.capture_authorization")
    _path_hash_identity(context["excitation_plan"], f"{path}.context.excitation_plan", plan=True)
    validate_process_identity(context["wrapper_process"], f"{path}.context.wrapper_process")
    validate_frozen_paths(context["paths"], f"{path}.context.paths")
    _validate_attempt_role_argv(context["child_argv"], f"{path}.context.child_argv", role="powered_child", context=context, python_path=None)
    _validate_attempt_role_argv(context["cleanup_argv"], f"{path}.context.cleanup_argv", role="cleanup_fallback", context=context, python_path=None)
    validate_deadline_durations(context["deadline_durations_ns"], f"{path}.context.deadline_durations_ns")
    absolute = _validate_wrapper_absolute_deadlines(context["wrapper_absolute_deadlines"], f"{path}.context.wrapper_absolute_deadlines")
    pre = _object(context["prepublication_timing"], {"wrapper_started_monotonic_ns", "offline_precheck", "attempt_publish"}, f"{path}.context.prepublication_timing")
    _int(pre["wrapper_started_monotonic_ns"], f"{path}.context.prepublication_timing.wrapper_started_monotonic_ns")
    if pre["wrapper_started_monotonic_ns"] != absolute["started_monotonic_ns"]:
        _fail(f"{path}.context.prepublication_timing.wrapper_started_monotonic_ns", "must equal wrapper absolute start")
    offline = _object(pre["offline_precheck"], {"phase", "started_monotonic_ns", "duration_ns", "parent_deadline_monotonic_ns", "deadline_monotonic_ns", "completed_monotonic_ns", "outcome"}, f"{path}.context.prepublication_timing.offline_precheck")
    validate_phase_deadline({key: offline[key] for key in offline if key not in {"completed_monotonic_ns", "outcome"}}, f"{path}.context.prepublication_timing.offline_precheck", expected_phase="offline_precheck")
    if offline["duration_ns"] != DEADLINE_DURATIONS_NS["offline_precheck"]:
        _fail(f"{path}.context.prepublication_timing.offline_precheck.duration_ns", "must equal frozen duration")
    completed = _int(offline["completed_monotonic_ns"], f"{path}.context.prepublication_timing.offline_precheck.completed_monotonic_ns")
    if not offline["started_monotonic_ns"] <= completed < offline["deadline_monotonic_ns"]:
        _fail(f"{path}.context.prepublication_timing.offline_precheck.completed_monotonic_ns", "must complete within the phase")
    _literal(offline["outcome"], "completed", f"{path}.context.prepublication_timing.offline_precheck.outcome")
    attempt_publish = validate_phase_deadline(pre["attempt_publish"], f"{path}.context.prepublication_timing.attempt_publish", expected_phase="attempt_publish")
    if attempt_publish["duration_ns"] != DEADLINE_DURATIONS_NS["attempt_publish"]:
        _fail(f"{path}.context.prepublication_timing.attempt_publish.duration_ns", "must equal frozen duration")

    context_hash = _sha256(row["context_sha256"], f"{path}.context_sha256")
    if context_hash != canonical_object_sha256(context):
        _fail(f"{path}.context_sha256", "must hash the complete canonical context")
    capabilities = _object(row["capabilities"], {"algorithm", "lease_owner_sha256", "child_sha256", "cleanup_sha256"}, f"{path}.capabilities")
    _literal(capabilities["algorithm"], "sha256-domain-separated-context-v1", f"{path}.capabilities.algorithm")
    for name in ("lease_owner_sha256", "child_sha256", "cleanup_sha256"):
        _sha256(capabilities[name], f"{path}.capabilities.{name}")
    if len({capabilities[name] for name in ("lease_owner_sha256", "child_sha256", "cleanup_sha256")}) != 3:
        _fail(f"{path}.capabilities", "all three capability hashes must be distinct")

    if live_freeze is not None:
        freeze = validate_live_freeze(live_freeze)
        _validate_attempt_role_argv(context["child_argv"], f"{path}.context.child_argv", role="powered_child", context=context, python_path=freeze["runtime"]["python"]["path"])
        _validate_attempt_role_argv(context["cleanup_argv"], f"{path}.context.cleanup_argv", role="cleanup_fallback", context=context, python_path=freeze["runtime"]["python"]["path"])
        if context["live_freeze"]["sha256"] != canonical_file_sha256(freeze):
            _fail(f"{path}.context.live_freeze.sha256", "does not bind supplied live freeze file bytes")
        if context["candidate_commit"] != freeze["candidate"]["commit"]:
            _fail(f"{path}.context.candidate_commit", "does not match live freeze")
        for name in ("target_config", "capture_authorization", "excitation_plan"):
            for field in ("path", "sha256"):
                if context[name][field] != freeze["inputs"][name][field]:
                    _fail(f"{path}.context.{name}.{field}", "does not match live freeze")
        if context["paths"] != freeze["paths"] or context["deadline_durations_ns"] != freeze["deadline_durations_ns"]:
            _fail(f"{path}.context", "paths/deadline durations do not match live freeze")
    return defensive_copy(row)


def attempt_file_sha256(value: Any, *, live_freeze: Any | None = None) -> str:
    validated = validate_attempt(value, live_freeze=live_freeze)
    return canonical_file_sha256(validated)


def validate_process_authority(
    value: Any,
    *,
    attempt: Any | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    path = "$process_authority"
    row = _object(
        value,
        {"schema", "task_id", "session_id", "attempt_id", "role", "created_at_utc", "created_monotonic_ns", "attempt_envelope_sha256", "attempt_context_sha256", "live_freeze_sha256", "wrapper_process", "process", "parent_handle", "capability_sha256", "lease_record_sha256", "training_attestation_sha256", "simulator_process_proof_sha256", "argv_sha256", "job", "absolute_deadlines"},
        path,
    )
    _literal(row["schema"], "aigp-vq2-powered-process-authority/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    role = _enum(row["role"], {"powered_child", "cleanup_fallback"}, f"{path}.role")
    _utc(row["created_at_utc"], f"{path}.created_at_utc")
    created = _int(row["created_monotonic_ns"], f"{path}.created_monotonic_ns")
    for name in ("attempt_envelope_sha256", "attempt_context_sha256", "live_freeze_sha256", "capability_sha256", "lease_record_sha256", "training_attestation_sha256", "simulator_process_proof_sha256", "argv_sha256"):
        _sha256(row[name], f"{path}.{name}")
    validate_process_identity(row["wrapper_process"], f"{path}.wrapper_process")
    validate_process_identity(row["process"], f"{path}.process")
    if row["process"] == row["wrapper_process"]:
        _fail(f"{path}.process", "authority process must differ from wrapper")
    if row["process"]["argv_sha256"] != row["argv_sha256"]:
        _fail(f"{path}.process.argv_sha256", "must equal authority argv_sha256")
    parent = _object(row["parent_handle"], {"value", "process", "access", "inherited"}, f"{path}.parent_handle")
    _int(parent["value"], f"{path}.parent_handle.value", minimum=1)
    validate_process_identity(parent["process"], f"{path}.parent_handle.process")
    if parent["process"] != row["wrapper_process"]:
        _fail(f"{path}.parent_handle.process", "must byte-equal wrapper_process")
    _literal(parent["access"], "synchronize_query_limited_information", f"{path}.parent_handle.access")
    _literal(parent["inherited"], True, f"{path}.parent_handle.inherited")
    job = _object(row["job"], {"handle_value", "assigned_before_capability_release", "breakaway_allowed", "silent_breakaway_allowed", "kill_on_close", "process_in_job"}, f"{path}.job")
    _int(job["handle_value"], f"{path}.job.handle_value", minimum=1)
    _literal(job["assigned_before_capability_release"], True, f"{path}.job.assigned_before_capability_release")
    _literal(job["breakaway_allowed"], False, f"{path}.job.breakaway_allowed")
    _literal(job["silent_breakaway_allowed"], False, f"{path}.job.silent_breakaway_allowed")
    _literal(job["kill_on_close"], False, f"{path}.job.kill_on_close")
    _literal(job["process_in_job"], True, f"{path}.job.process_in_job")

    anchor: int
    if role == "powered_child":
        deadlines = _object(row["absolute_deadlines"], {"anchor", "total", "prepower", "powered", "cleanup", "replay_close", "exit"}, f"{path}.absolute_deadlines")
        anchor = _int(deadlines["anchor"], f"{path}.absolute_deadlines.anchor")
        expected_offsets = {"total": 110, "prepower": 52, "powered": 57, "cleanup": 72, "replay_close": 107, "exit": 110}
    else:
        deadlines = _object(row["absolute_deadlines"], {"anchor", "total", "exit"}, f"{path}.absolute_deadlines")
        anchor = _int(deadlines["anchor"], f"{path}.absolute_deadlines.anchor")
        expected_offsets = {"total": 25, "exit": 25}
    for name, seconds in expected_offsets.items():
        _int(deadlines[name], f"{path}.absolute_deadlines.{name}")
        if deadlines[name] != anchor + seconds * 1_000_000_000:
            _fail(f"{path}.absolute_deadlines.{name}", f"must equal anchor + {seconds} seconds")

    if argv is not None:
        argv_list = list(argv)
        _string_array(argv_list, "$argv", nonempty=True)
        if row["argv_sha256"] != canonical_object_sha256(argv_list):
            _fail(f"{path}.argv_sha256", "must hash the exact argv array")
    if attempt is not None:
        envelope = validate_attempt(attempt)
        if row["attempt_envelope_sha256"] != canonical_file_sha256(envelope):
            _fail(f"{path}.attempt_envelope_sha256", "does not bind supplied attempt file")
        context = envelope["context"]
        if row["attempt_context_sha256"] != envelope["context_sha256"]:
            _fail(f"{path}.attempt_context_sha256", "does not bind supplied attempt context")
        if row["live_freeze_sha256"] != context["live_freeze"]["sha256"]:
            _fail(f"{path}.live_freeze_sha256", "does not match attempt")
        if row["wrapper_process"] != context["wrapper_process"]:
            _fail(f"{path}.wrapper_process", "does not match attempt wrapper")
        expected_capability = "child_sha256" if role == "powered_child" else "cleanup_sha256"
        if row["capability_sha256"] != envelope["capabilities"][expected_capability]:
            _fail(f"{path}.capability_sha256", "does not match role capability")
        expected_argv = context["child_argv"] if role == "powered_child" else context["cleanup_argv"]
        if row["argv_sha256"] != canonical_object_sha256(expected_argv):
            _fail(f"{path}.argv_sha256", "does not bind attempt role argv")
        _, expected_parent_handle = _validate_attempt_role_argv(
            expected_argv,
            "$attempt.context.role_argv",
            role=role,
            context=context,
            python_path=None,
        )
        if parent["value"] != expected_parent_handle:
            _fail(f"{path}.parent_handle.value", "does not match role argv parent-liveness handle")
    return defensive_copy(row)


def _validate_ingress(value: Any, path: str, *, message_type: str) -> dict[str, Any]:
    row = _object(
        value,
        {"schema", "stream_id", "generation", "sequence", "message_type", "host_clock_id", "received_monotonic_ns", "source_time_value", "source_time_unit"},
        path,
    )
    _literal(row["schema"], "aigp-vq2-mavlink-ingress/1", f"{path}.schema")
    _token(row["stream_id"], f"{path}.stream_id")
    _int(row["generation"], f"{path}.generation")
    _int(row["sequence"], f"{path}.sequence")
    _literal(row["message_type"], message_type, f"{path}.message_type")
    _literal(row["host_clock_id"], HOST_CLOCK_ID, f"{path}.host_clock_id")
    _int(row["received_monotonic_ns"], f"{path}.received_monotonic_ns", maximum=_UINT64_MAX)
    expected_unit = {"HEARTBEAT": None, "RACE_STATUS": "ms", "HIGHRES_IMU": "us", "ACTUATOR_OUTPUT_STATUS": "us"}[message_type]
    if expected_unit is None:
        if row["source_time_value"] is not None or row["source_time_unit"] is not None:
            _fail(path, "this ingress type must have null source-time fields")
    else:
        _int(row["source_time_value"], f"{path}.source_time_value", maximum=_UINT64_MAX)
        _literal(row["source_time_unit"], expected_unit, f"{path}.source_time_unit")
    return row


def validate_received_heartbeat(value: Any, path: str = "$received_heartbeat") -> dict[str, Any]:
    row = _object(value, {"schema", "ingress", "heartbeat"}, path)
    _literal(row["schema"], "aigp-vq2-received-heartbeat/1", f"{path}.schema")
    _validate_ingress(row["ingress"], f"{path}.ingress", message_type="HEARTBEAT")
    heartbeat = _object(row["heartbeat"], {"base_mode", "custom_mode"}, f"{path}.heartbeat")
    _int(heartbeat["base_mode"], f"{path}.heartbeat.base_mode", maximum=_UINT8_MAX)
    _int(heartbeat["custom_mode"], f"{path}.heartbeat.custom_mode", maximum=_UINT32_MAX)
    return defensive_copy(row)


def validate_received_race_status(value: Any, path: str = "$received_race_status") -> dict[str, Any]:
    row = _object(value, {"schema", "ingress", "race_status"}, path)
    _literal(row["schema"], "aigp-vq2-received-race-status/1", f"{path}.schema")
    ingress = _validate_ingress(row["ingress"], f"{path}.ingress", message_type="RACE_STATUS")
    status = _object(row["race_status"], {"sim_boot_time_ms", "race_start_boot_time_ms", "race_finish_time_ns", "active_gate_index", "last_gate_race_time"}, f"{path}.race_status")
    _int(status["sim_boot_time_ms"], f"{path}.race_status.sim_boot_time_ms", maximum=_UINT64_MAX)
    for name in ("race_start_boot_time_ms", "race_finish_time_ns", "last_gate_race_time"):
        _int(status[name], f"{path}.race_status.{name}", minimum=_INT64_MIN, maximum=_INT64_MAX)
    _int(status["active_gate_index"], f"{path}.race_status.active_gate_index", maximum=_UINT32_MAX)
    if ingress["source_time_value"] != status["sim_boot_time_ms"]:
        _fail(f"{path}.ingress.source_time_value", "must equal race_status.sim_boot_time_ms")
    return defensive_copy(row)


def validate_received_actuator_output_status(value: Any, path: str = "$received_actuator") -> dict[str, Any]:
    row = _object(value, {"schema", "ingress", "actuator_output_status"}, path)
    _literal(row["schema"], "aigp-vq2-received-actuator-output-status/1", f"{path}.schema")
    ingress = _validate_ingress(row["ingress"], f"{path}.ingress", message_type="ACTUATOR_OUTPUT_STATUS")
    status = _object(row["actuator_output_status"], {"time_usec", "active", "actuator"}, f"{path}.actuator_output_status")
    _int(status["time_usec"], f"{path}.actuator_output_status.time_usec", maximum=_UINT64_MAX)
    _int(status["active"], f"{path}.actuator_output_status.active", maximum=_UINT32_MAX)
    for index, item in enumerate(_array(status["actuator"], f"{path}.actuator_output_status.actuator", length=32)):
        _number(item, f"{path}.actuator_output_status.actuator[{index}]")
    if ingress["source_time_value"] != status["time_usec"]:
        _fail(f"{path}.ingress.source_time_value", "must equal actuator_output_status.time_usec")
    return defensive_copy(row)


def validate_received_imu(value: Any, path: str = "$received_imu") -> dict[str, Any]:
    row = _object(value, {"schema", "ingress", "imu"}, path)
    _literal(row["schema"], "aigp-vq2-received-imu/1", f"{path}.schema")
    ingress = _validate_ingress(row["ingress"], f"{path}.ingress", message_type="HIGHRES_IMU")
    imu = _object(row["imu"], {"timestamp_us", "accel", "gyro", "mag"}, f"{path}.imu")
    _int(imu["timestamp_us"], f"{path}.imu.timestamp_us", maximum=_UINT64_MAX)
    for name in ("accel", "gyro"):
        for index, item in enumerate(_array(imu[name], f"{path}.imu.{name}", length=3)):
            _number(item, f"{path}.imu.{name}[{index}]")
    if imu["mag"] is not None:
        for index, item in enumerate(_array(imu["mag"], f"{path}.imu.mag", length=3)):
            _number(item, f"{path}.imu.mag[{index}]")
    if ingress["source_time_value"] != imu["timestamp_us"]:
        _fail(f"{path}.ingress.source_time_value", "must equal imu.timestamp_us")
    return defensive_copy(row)


_RECEIVED_VALIDATORS: dict[str, Callable[[Any, str], dict[str, Any]]] = {
    "aigp-vq2-received-heartbeat/1": validate_received_heartbeat,
    "aigp-vq2-received-race-status/1": validate_received_race_status,
    "aigp-vq2-received-actuator-output-status/1": validate_received_actuator_output_status,
    "aigp-vq2-received-imu/1": validate_received_imu,
}


def _validate_received(value: Any, path: str) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("schema")) is not str:
        _fail(path, "must be a received-envelope object with a schema")
    validator = _RECEIVED_VALIDATORS.get(value["schema"])
    if validator is None:
        _fail(f"{path}.schema", "unsupported received-envelope schema")
    return validator(value, path)


def _validate_frame_timing(value: Any, path: str) -> dict[str, Any]:
    row = _object(
        value,
        {"schema", "identity", "camera_source_time_ns", "host_clock_id", "publication_sequence", "first_unique_packet_monotonic_ns", "final_unique_packet_monotonic_ns", "reassembly_complete_monotonic_ns", "decode_start_monotonic_ns", "decode_end_monotonic_ns", "publish_monotonic_ns"},
        path,
    )
    _literal(row["schema"], "aigp-vq2-frame-timing/1", f"{path}.schema")
    identity = _object(row["identity"], {"schema", "stream_id", "generation", "frame_id"}, f"{path}.identity")
    _literal(identity["schema"], "aigp-vq2-frame-identity/1", f"{path}.identity.schema")
    _token(identity["stream_id"], f"{path}.identity.stream_id")
    _int(identity["generation"], f"{path}.identity.generation")
    _int(identity["frame_id"], f"{path}.identity.frame_id")
    _int(row["camera_source_time_ns"], f"{path}.camera_source_time_ns", maximum=_UINT64_MAX)
    _literal(row["host_clock_id"], HOST_CLOCK_ID, f"{path}.host_clock_id")
    _int(row["publication_sequence"], f"{path}.publication_sequence")
    names = (
        "first_unique_packet_monotonic_ns", "final_unique_packet_monotonic_ns", "reassembly_complete_monotonic_ns",
        "decode_start_monotonic_ns", "decode_end_monotonic_ns", "publish_monotonic_ns",
    )
    points = [_int(row[name], f"{path}.{name}") for name in names]
    if any(later < earlier for earlier, later in zip(points, points[1:])):
        _fail(path, "frame timing points must be monotonic")
    return row


def validate_decoded_dimensions_admission(value: Any, path: str = "$dimensions") -> dict[str, Any]:
    row = _object(value, {"schema", "config_sha256", "expected", "observed", "first_frame_timing", "admitted_monotonic_ns", "status"}, path)
    _literal(row["schema"], "aigp-vq2-decoded-dimensions-admission/1", f"{path}.schema")
    _sha256(row["config_sha256"], f"{path}.config_sha256")
    for name in ("expected", "observed"):
        dims = _object(row[name], {"width", "height"}, f"{path}.{name}")
        _int(dims["width"], f"{path}.{name}.width", minimum=1)
        _int(dims["height"], f"{path}.{name}.height", minimum=1)
    if row["expected"] != {"width": 640, "height": 360} or row["observed"] != row["expected"]:
        _fail(path, "admission must prove exact stable 640x360 dimensions")
    timing = _validate_frame_timing(row["first_frame_timing"], f"{path}.first_frame_timing")
    admitted = _int(row["admitted_monotonic_ns"], f"{path}.admitted_monotonic_ns")
    if admitted < timing["publish_monotonic_ns"]:
        _fail(f"{path}.admitted_monotonic_ns", "must not precede first frame publication")
    _literal(row["status"], "admitted", f"{path}.status")
    return defensive_copy(row)


def _validate_ingress_stats(value: Any, path: str) -> dict[str, Any]:
    keys = {"generation", "next_sequence", "highres_imu_received", "heartbeat_received", "race_status_received", "actuator_received", "dropped", "high_watermark", "imu_capacity", "other_capacity", "imu_dropped", "other_dropped", "imu_high_watermark", "other_high_watermark", "buffered_imu", "buffered_other"}
    row = _object(value, keys, path)
    for name in keys:
        _int(row[name], f"{path}.{name}")
    for name in ("imu_capacity", "other_capacity"):
        _int(row[name], f"{path}.{name}", minimum=1)
    received_imu = row["highres_imu_received"]
    received_other = (
        row["heartbeat_received"]
        + row["race_status_received"]
        + row["actuator_received"]
    )
    received_total = received_imu + received_other
    if row["next_sequence"] != received_total:
        _fail(f"{path}.next_sequence", "must equal the sum of received message counters")
    if row["dropped"] != row["imu_dropped"] + row["other_dropped"]:
        _fail(f"{path}.dropped", "must equal imu_dropped + other_dropped")
    for prefix, received in (("imu", received_imu), ("other", received_other)):
        capacity = row[f"{prefix}_capacity"]
        dropped = row[f"{prefix}_dropped"]
        high_watermark = row[f"{prefix}_high_watermark"]
        buffered = row[f"buffered_{prefix}"]
        if dropped > received:
            _fail(f"{path}.{prefix}_dropped", "cannot exceed received messages for the queue")
        if buffered > capacity or high_watermark > capacity:
            _fail(path, f"{prefix} buffered/high-water counts cannot exceed capacity")
        if buffered > high_watermark:
            _fail(f"{path}.buffered_{prefix}", f"cannot exceed {prefix}_high_watermark")
        if high_watermark > received:
            _fail(f"{path}.{prefix}_high_watermark", "cannot exceed received messages for the queue")
        if received and high_watermark == 0:
            _fail(f"{path}.{prefix}_high_watermark", "must be positive after a received message")
        if dropped + buffered > received:
            _fail(path, f"{prefix} dropped plus buffered cannot exceed received messages")
        if dropped and high_watermark != capacity:
            _fail(path, f"{prefix} drops require the queue to have reached capacity")
    buffered_total = row["buffered_imu"] + row["buffered_other"]
    if not (
        buffered_total <= row["high_watermark"] <= received_total
        and row["high_watermark"] <= row["imu_capacity"] + row["other_capacity"]
        and max(row["imu_high_watermark"], row["other_high_watermark"])
        <= row["high_watermark"]
        <= row["imu_high_watermark"] + row["other_high_watermark"]
    ):
        _fail(path, "aggregate ingress high-water accounting is inconsistent")
    return row


def _validate_collision_stats(value: Any, path: str) -> dict[str, Any]:
    keys = {"generation", "handled", "dropped", "high_watermark", "capacity", "buffered"}
    row = _object(value, keys, path)
    for name in keys:
        _int(row[name], f"{path}.{name}")
    _int(row["capacity"], f"{path}.capacity", minimum=1)
    if row["dropped"] > row["handled"]:
        _fail(f"{path}.dropped", "cannot exceed handled collision count")
    if row["buffered"] > row["capacity"] or row["high_watermark"] > row["capacity"]:
        _fail(path, "collision buffered/high-water counts cannot exceed capacity")
    if row["buffered"] > row["high_watermark"]:
        _fail(f"{path}.buffered", "cannot exceed collision high_watermark")
    if row["high_watermark"] > row["handled"]:
        _fail(f"{path}.high_watermark", "cannot exceed handled collision count")
    if row["dropped"] + row["buffered"] > row["handled"]:
        _fail(path, "collision dropped plus buffered cannot exceed handled count")
    if row["dropped"] and row["high_watermark"] != row["capacity"]:
        _fail(path, "a collision drop requires the queue to have reached capacity")
    return row


def validate_collision_observation(value: Any, path: str = "$collision") -> dict[str, Any]:
    row = _object(value, {"schema", "reset_generation", "observation_sequence", "host_clock_id", "observed_monotonic_ns", "phase", "disposition", "boundary", "collision"}, path)
    _literal(row["schema"], "aigp-vq2-runner-collision-observation/1", f"{path}.schema")
    _int(row["reset_generation"], f"{path}.reset_generation")
    _int(row["observation_sequence"], f"{path}.observation_sequence")
    _literal(row["host_clock_id"], HOST_CLOCK_ID, f"{path}.host_clock_id")
    _int(row["observed_monotonic_ns"], f"{path}.observed_monotonic_ns")
    _token(row["phase"], f"{path}.phase")
    _token(row["disposition"], f"{path}.disposition")
    _literal(row["boundary"], "runner_drain_not_receiver_receipt", f"{path}.boundary")
    collision = _object(row["collision"], {"id", "threat_level", "impulse"}, f"{path}.collision")
    _int(collision["id"], f"{path}.collision.id", maximum=_UINT32_MAX)
    _int(collision["threat_level"], f"{path}.collision.threat_level", maximum=_UINT8_MAX)
    _number(collision["impulse"], f"{path}.collision.impulse")
    return defensive_copy(row)


def validate_reset_boundary(value: Any, path: str = "$reset_boundary") -> dict[str, Any]:
    row = _object(value, {"schema", "old_generation", "new_generation", "boundary_monotonic_ns", "observations", "collisions", "ingress_stats", "collision_stats"}, path)
    _literal(row["schema"], "aigp-vq2-calibration-reset-boundary/1", f"{path}.schema")
    old = _int(row["old_generation"], f"{path}.old_generation")
    new = _int(row["new_generation"], f"{path}.new_generation")
    if new != old + 1:
        _fail(f"{path}.new_generation", "must equal old_generation + 1")
    boundary_time = _int(
        row["boundary_monotonic_ns"], f"{path}.boundary_monotonic_ns"
    )
    observations = _array(row["observations"], f"{path}.observations")
    checked_observations: list[dict[str, Any]] = []
    prior_sequence = -1
    for index, observation in enumerate(observations):
        checked = _validate_received(observation, f"{path}.observations[{index}]")
        checked_observations.append(checked)
        ingress = checked["ingress"]
        if (
            ingress["generation"] != old
            or ingress["sequence"] <= prior_sequence
            or ingress["received_monotonic_ns"] > boundary_time
        ):
            _fail(f"{path}.observations[{index}].ingress", "must preserve old-generation ingress order")
        prior_sequence = ingress["sequence"]
    collisions = _array(row["collisions"], f"{path}.collisions")
    checked_collisions: list[dict[str, Any]] = []
    prior_collision = -1
    for index, collision in enumerate(collisions):
        checked = validate_collision_observation(collision, f"{path}.collisions[{index}]")
        checked_collisions.append(checked)
        if (
            checked["reset_generation"] != old
            or checked["observation_sequence"] <= prior_collision
            or checked["observed_monotonic_ns"] != boundary_time
        ):
            _fail(f"{path}.collisions[{index}]", "must preserve old-generation collision order")
        prior_collision = checked["observation_sequence"]
    ingress_stats = _validate_ingress_stats(row["ingress_stats"], f"{path}.ingress_stats")
    collision_stats = _validate_collision_stats(row["collision_stats"], f"{path}.collision_stats")
    if ingress_stats["generation"] != old or collision_stats["generation"] != old:
        _fail(path, "diagnostics must describe the old generation")
    buffered_imu = sum(
        item["schema"] == "aigp-vq2-received-imu/1"
        for item in checked_observations
    )
    if (
        ingress_stats["buffered_imu"] != buffered_imu
        or ingress_stats["buffered_other"] != len(checked_observations) - buffered_imu
    ):
        _fail(f"{path}.ingress_stats", "buffered counts must equal preserved boundary observations by queue")
    received_fields = {
        "HIGHRES_IMU": "highres_imu_received",
        "HEARTBEAT": "heartbeat_received",
        "RACE_STATUS": "race_status_received",
        "ACTUATOR_OUTPUT_STATUS": "actuator_received",
    }
    for message_type, field in received_fields.items():
        represented = sum(
            item["ingress"]["message_type"] == message_type
            for item in checked_observations
        )
        if represented > ingress_stats[field]:
            _fail(f"{path}.ingress_stats.{field}", "cannot be smaller than represented observations")
    if any(
        item["ingress"]["sequence"] >= ingress_stats["next_sequence"]
        for item in checked_observations
    ):
        _fail(f"{path}.observations", "every ingress sequence must precede next_sequence")
    if collision_stats["buffered"] != len(checked_collisions):
        _fail(f"{path}.collision_stats.buffered", "must equal preserved boundary collisions")
    if any(
        item["observation_sequence"] >= collision_stats["handled"]
        for item in checked_collisions
    ):
        _fail(f"{path}.collisions", "every collision sequence must precede handled count")
    return defensive_copy(row)


def _validate_outbound_common(value: Any, keys: set[str], path: str, *, schema: str) -> dict[str, Any]:
    row = _object(value, keys, path)
    _literal(row["schema"], schema, f"{path}.schema")
    _token(row["stream_id"], f"{path}.stream_id")
    _int(row["reset_generation"], f"{path}.reset_generation")
    _int(row["outbound_sequence"], f"{path}.outbound_sequence")
    _literal(row["host_clock_id"], HOST_CLOCK_ID, f"{path}.host_clock_id")
    start = _int(row["call_start_monotonic_ns"], f"{path}.call_start_monotonic_ns")
    end = _int(row["call_end_monotonic_ns"], f"{path}.call_end_monotonic_ns")
    if end < start:
        _fail(f"{path}.call_end_monotonic_ns", "must not precede call start")
    outcome = _enum(row["outcome"], {"returned", "raised"}, f"{path}.outcome")
    if outcome == "returned":
        if row["error_type"] is not None:
            _fail(f"{path}.error_type", "must be null for returned calls")
    else:
        _token(row["error_type"], f"{path}.error_type")
    return row


def validate_attitude_target_outbound(value: Any, path: str = "$attitude_outbound") -> dict[str, Any]:
    keys = {"schema", "stream_id", "reset_generation", "outbound_sequence", "host_clock_id", "call_start_monotonic_ns", "call_end_monotonic_ns", "api", "outcome", "error_type", "wire"}
    row = _validate_outbound_common(value, keys, path, schema="aigp-vq2-attitude-target-outbound/1")
    _enum(row["api"], {"send_attitude_rate", "send_attitude_rate_from_attitude", "send_attitude_quaternion"}, f"{path}.api")
    wire = _object(row["wire"], {"time_boot_ms", "target_system", "target_component", "type_mask", "q_wxyz", "body_rates_rad_s", "thrust"}, f"{path}.wire")
    _int(wire["time_boot_ms"], f"{path}.wire.time_boot_ms", maximum=_UINT32_MAX)
    for name in ("target_system", "target_component", "type_mask"):
        _int(wire[name], f"{path}.wire.{name}", maximum=_UINT8_MAX)
    for name, length in (("q_wxyz", 4), ("body_rates_rad_s", 3)):
        for index, item in enumerate(_array(wire[name], f"{path}.wire.{name}", length=length)):
            _number(item, f"{path}.wire.{name}[{index}]")
    _number(wire["thrust"], f"{path}.wire.thrust")
    return defensive_copy(row)


def validate_nonattitude_outbound(value: Any, path: str = "$nonattitude_outbound") -> dict[str, Any]:
    keys = {"schema", "stream_id", "reset_generation", "outbound_sequence", "host_clock_id", "call_start_monotonic_ns", "call_end_monotonic_ns", "category", "api", "outcome", "error_type", "wire"}
    row = _validate_outbound_common(value, keys, path, schema="aigp-vq2-nonattitude-outbound/1")
    category = _enum(row["category"], {"arm", "disarm", "sim_reset", "timesync", "gcs_heartbeat"}, f"{path}.category")
    expected_api = "command_long_send" if category in {"arm", "disarm", "sim_reset"} else ("timesync_send" if category == "timesync" else "heartbeat_send")
    _literal(row["api"], expected_api, f"{path}.api")
    if category in {"arm", "disarm", "sim_reset"}:
        wire = _object(row["wire"], {"target_system", "target_component", "command", "confirmation", "params"}, f"{path}.wire")
        for name in ("target_system", "target_component"):
            _int(wire[name], f"{path}.wire.{name}", maximum=_UINT8_MAX)
        expected_command = 31_000 if category == "sim_reset" else 400
        _literal(wire["command"], expected_command, f"{path}.wire.command")
        _literal(wire["confirmation"], 0, f"{path}.wire.confirmation")
        expected_params = (
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if category == "arm"
            else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        params = _array(wire["params"], f"{path}.wire.params", length=7)
        for index, (item, expected) in enumerate(zip(params, expected_params)):
            item_path = f"{path}.wire.params[{index}]"
            _literal(item, expected, item_path)
            if item == 0.0 and math.copysign(1.0, item) < 0.0:
                _fail(item_path, "must use canonical positive zero")
    elif category == "timesync":
        wire = _object(row["wire"], {"tc1", "ts1"}, f"{path}.wire")
        _literal(wire["tc1"], 0, f"{path}.wire.tc1")
        _int(wire["ts1"], f"{path}.wire.ts1", minimum=1, maximum=_INT64_MAX)
    else:
        wire = _object(row["wire"], {"type", "autopilot", "base_mode", "custom_mode", "system_status"}, f"{path}.wire")
        expected_heartbeat = {
            "type": 6,
            "autopilot": 8,
            "base_mode": 0,
            "custom_mode": 0,
            "system_status": 4,
        }
        for name, expected in expected_heartbeat.items():
            _literal(wire[name], expected, f"{path}.wire.{name}")
    return defensive_copy(row)


def _validate_outbound_receipt(value: Any, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(path, "must be an outbound receipt object")
    schema = value.get("schema")
    if schema == "aigp-vq2-attitude-target-outbound/1":
        return validate_attitude_target_outbound(value, path)
    if schema == "aigp-vq2-nonattitude-outbound/1":
        return validate_nonattitude_outbound(value, path)
    _fail(f"{path}.schema", "unsupported outbound receipt schema")


_COMMAND_COMMON_KEYS = {
    "attempt_id", "session_id", "candidate_commit", "attempt_context_sha256",
    "event_sequence", "host_clock_id", "reset_epoch", "plan", "scope",
    "command_id", "absolute_tick", "segment_id", "slot", "command", "source",
    "watchdogs",
}
_COMMAND_FAILURE_CODES = frozenset(
    {
        "slot_missed", "deadline_expired", "stream_stale", "imu_not_advancing",
        "race_not_advancing", "estimator_unhealthy", "target_missing",
        "target_unstable", "target_out_of_corridor", "target_too_large",
        "attitude_excursion", "collision_observed", "gate_changed",
        "capture_failed", "parent_dead", "lease_invalid", "send_raised",
        "internal_error",
    }
)


def _validate_reset_epoch(value: Any, path: str) -> dict[str, Any]:
    row = _object(value, {"ingress_generation", "race_anchor_boot_ms", "imu_anchor_usec"}, path)
    _int(row["ingress_generation"], f"{path}.ingress_generation")
    _int(row["race_anchor_boot_ms"], f"{path}.race_anchor_boot_ms", maximum=_UINT64_MAX)
    _int(row["imu_anchor_usec"], f"{path}.imu_anchor_usec", maximum=_UINT64_MAX)
    return row


def _validate_plan_identity(value: Any, path: str) -> dict[str, Any]:
    row = _object(value, {"plan_id", "sha256"}, path)
    _literal(row["plan_id"], EXCITATION_PLAN_ID, f"{path}.plan_id")
    _literal(row["sha256"], EXCITATION_PLAN_SHA256, f"{path}.sha256")
    return row


def _validate_command(value: Any, path: str) -> dict[str, Any]:
    row = _object(value, {"roll_rate_rad_s", "pitch_rate_rad_s", "yaw_rate_rad_s", "thrust"}, path)
    for name in row:
        _number(row[name], f"{path}.{name}")
    return row


def _validate_slot(value: Any, path: str) -> dict[str, Any]:
    row = _object(value, {"release_monotonic_ns", "end_monotonic_ns", "powered_expiry_monotonic_ns"}, path)
    release = _int(row["release_monotonic_ns"], f"{path}.release_monotonic_ns")
    end = _int(row["end_monotonic_ns"], f"{path}.end_monotonic_ns")
    expiry = _int(row["powered_expiry_monotonic_ns"], f"{path}.powered_expiry_monotonic_ns")
    if not release < end <= expiry:
        _fail(path, "must satisfy release < end <= powered expiry")
    return row


def _validate_source_frame(value: Any, path: str) -> dict[str, Any]:
    # These names make the prose's complete timing primitive and admitted
    # decoded width/height explicit without importing the live vision stack.
    row = _object(value, {"stream_id", "generation", "frame_id", "sim_time_ns", "timing", "width", "height"}, path)
    _token(row["stream_id"], f"{path}.stream_id")
    _int(row["generation"], f"{path}.generation")
    _int(row["frame_id"], f"{path}.frame_id")
    _int(row["sim_time_ns"], f"{path}.sim_time_ns", maximum=_UINT64_MAX)
    timing = _validate_frame_timing(row["timing"], f"{path}.timing")
    _int(row["width"], f"{path}.width", minimum=1)
    _int(row["height"], f"{path}.height", minimum=1)
    if row["width"] != 640 or row["height"] != 360:
        _fail(path, "source dimensions must equal admitted 640x360")
    identity = timing["identity"]
    if any(row[name] != identity[name] for name in ("stream_id", "generation", "frame_id")):
        _fail(path, "frame token must match timing identity")
    if row["sim_time_ns"] != timing["camera_source_time_ns"]:
        _fail(f"{path}.sim_time_ns", "must equal timing camera source token")
    return row


def _validate_command_source(value: Any, path: str, *, scope: str, reset_epoch: dict[str, Any] | None) -> dict[str, Any]:
    row = _object(value, {"frame", "imu", "race", "heartbeat", "actuator"}, path)
    if scope == "cleanup_zero":
        if any(row[name] is not None for name in row):
            _fail(path, "all cleanup-zero source values must be null")
        return row
    for name in row:
        if row[name] is None:
            _fail(f"{path}.{name}", "excitation source must be nonnull")
    frame = _validate_source_frame(row["frame"], f"{path}.frame")
    imu = validate_received_imu(row["imu"], f"{path}.imu")
    race = validate_received_race_status(row["race"], f"{path}.race")
    heartbeat = validate_received_heartbeat(row["heartbeat"], f"{path}.heartbeat")
    actuator = validate_received_actuator_output_status(row["actuator"], f"{path}.actuator")
    assert reset_epoch is not None
    generation = reset_epoch["ingress_generation"]
    for name, received in (("imu", imu), ("race", race), ("heartbeat", heartbeat), ("actuator", actuator)):
        if received["ingress"]["generation"] != generation:
            _fail(f"{path}.{name}.ingress.generation", "must match reset epoch")
    if frame["generation"] != generation:
        _fail(f"{path}.frame.generation", "must match reset epoch")
    if imu["imu"]["timestamp_us"] <= reset_epoch["imu_anchor_usec"]:
        _fail(f"{path}.imu", "must advance beyond reset IMU anchor")
    if race["race_status"]["sim_boot_time_ms"] <= reset_epoch["race_anchor_boot_ms"]:
        _fail(f"{path}.race", "must advance beyond reset race anchor")
    return row


_WATCHDOG_KEYS = {
    "checked_monotonic_ns", "heartbeat_age_ns", "imu_age_ns", "imu_advance_age_ns",
    "race_age_ns", "race_advance_age_ns", "actuator_age_ns", "vision_age_ns",
    "estimator_healthy", "target_consecutive", "target_center_px", "target_bbox_px",
    "target_bbox_area_px", "initial_target_bbox_area_px", "roll_excursion_rad",
    "pitch_excursion_rad", "collision_count", "gate_index", "result", "failure_codes",
}


def _validate_watchdogs(value: Any, path: str, *, scope: str) -> dict[str, Any]:
    row = _object(value, _WATCHDOG_KEYS, path)
    _int(row["checked_monotonic_ns"], f"{path}.checked_monotonic_ns")
    _array(row["failure_codes"], f"{path}.failure_codes", length=0)
    nullable_names = {
        "heartbeat_age_ns", "imu_age_ns", "imu_advance_age_ns", "race_age_ns",
        "race_advance_age_ns", "actuator_age_ns", "vision_age_ns", "estimator_healthy",
        "target_consecutive", "target_center_px", "target_bbox_px", "target_bbox_area_px",
        "initial_target_bbox_area_px", "roll_excursion_rad", "pitch_excursion_rad",
        "collision_count", "gate_index",
    }
    if scope == "cleanup_zero":
        if any(row[name] is not None for name in nullable_names):
            _fail(path, "cleanup watchdog facts other than checked time must be null")
        _literal(row["result"], "cleanup_authorized", f"{path}.result")
        return row
    for name in ("heartbeat_age_ns", "imu_age_ns", "imu_advance_age_ns", "race_age_ns", "race_advance_age_ns", "actuator_age_ns", "vision_age_ns", "target_consecutive", "collision_count", "gate_index"):
        _int(row[name], f"{path}.{name}")
    _bool(row["estimator_healthy"], f"{path}.estimator_healthy")
    for name, length in (("target_center_px", 2), ("target_bbox_px", 4)):
        for index, item in enumerate(_array(row[name], f"{path}.{name}", length=length)):
            _number(item, f"{path}.{name}[{index}]")
    bbox = row["target_bbox_px"]
    if bbox[2] <= 0 or bbox[3] <= 0:
        _fail(f"{path}.target_bbox_px", "bbox width and height must be positive")
    for name in ("target_bbox_area_px", "initial_target_bbox_area_px"):
        _number(row[name], f"{path}.{name}", positive=True)
    for name in ("roll_excursion_rad", "pitch_excursion_rad"):
        _number(row[name], f"{path}.{name}")
    _literal(row["gate_index"], 0, f"{path}.gate_index")
    _literal(row["estimator_healthy"], True, f"{path}.estimator_healthy")
    _literal(row["result"], "pass", f"{path}.result")
    return row


def _validate_command_common(row: dict[str, Any], path: str, *, time_key: str) -> str:
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _commit(row["candidate_commit"], f"{path}.candidate_commit")
    _sha256(row["attempt_context_sha256"], f"{path}.attempt_context_sha256")
    _int(row["event_sequence"], f"{path}.event_sequence")
    _literal(row["host_clock_id"], HOST_CLOCK_ID, f"{path}.host_clock_id")
    event_time = _int(row[time_key], f"{path}.{time_key}")
    scope = _enum(row["scope"], {"excitation", "cleanup_zero"}, f"{path}.scope")
    if scope == "excitation":
        reset_epoch = _validate_reset_epoch(row["reset_epoch"], f"{path}.reset_epoch")
        _validate_plan_identity(row["plan"], f"{path}.plan")
        tick = _int(row["absolute_tick"], f"{path}.absolute_tick", maximum=244)
        expected_tick = excitation_tick(tick)
        _literal(row["command_id"], f"excitation/{tick:03d}", f"{path}.command_id")
        _literal(row["segment_id"], expected_tick["segment_id"], f"{path}.segment_id")
        slot = _validate_slot(row["slot"], f"{path}.slot")
        period = _PLAN_LITERAL["control_period_ns"]
        anchor = slot["release_monotonic_ns"] - tick * period
        if anchor < 0 or slot["end_monotonic_ns"] != slot["release_monotonic_ns"] + period:
            _fail(f"{path}.slot", "must be the exact paced plan slot")
        if slot["powered_expiry_monotonic_ns"] != anchor + _PLAN_LITERAL["powered_hard_expiry_offset_ns"]:
            _fail(f"{path}.slot.powered_expiry_monotonic_ns", "must equal plan anchor + hard expiry")
        command = _validate_command(row["command"], f"{path}.command")
        if command != expected_tick["command"]:
            _fail(f"{path}.command", "must equal the frozen command for this tick")
        _validate_command_source(row["source"], f"{path}.source", scope=scope, reset_epoch=reset_epoch)
    else:
        for name in ("reset_epoch", "plan", "absolute_tick", "segment_id", "slot"):
            if row[name] is not None:
                _fail(f"{path}.{name}", "must be null for cleanup_zero")
        _literal(row["command_id"], "cleanup/zero/0", f"{path}.command_id")
        command = _validate_command(row["command"], f"{path}.command")
        if any(command[name] != 0 for name in command):
            _fail(f"{path}.command", "cleanup command must be exact zero")
        _validate_command_source(row["source"], f"{path}.source", scope=scope, reset_epoch=None)
    watchdogs = _validate_watchdogs(row["watchdogs"], f"{path}.watchdogs", scope=scope)
    if watchdogs["checked_monotonic_ns"] > event_time:
        _fail(f"{path}.watchdogs.checked_monotonic_ns", "must not follow the event time")
    return scope


def validate_command_generated(value: Any, path: str = "$command_generated") -> dict[str, Any]:
    row = _object(value, _COMMAND_COMMON_KEYS | {"schema", "generated_monotonic_ns"}, path)
    _literal(row["schema"], "aigp-vq2-calibration-command-generated/1", f"{path}.schema")
    _validate_command_common(row, path, time_key="generated_monotonic_ns")
    return defensive_copy(row)


def _require_generated_copy(row: dict[str, Any], generated: dict[str, Any], path: str) -> None:
    if row["generated_event_sequence"] != generated["event_sequence"]:
        _fail(f"{path}.generated_event_sequence", "must name the supplied generated event")
    if row["generation_sha256"] != canonical_object_sha256(generated):
        _fail(f"{path}.generation_sha256", "must hash the complete generated observation")
    copied = _COMMAND_COMMON_KEYS - {"candidate_commit", "attempt_context_sha256", "event_sequence", "host_clock_id"}
    copied |= {"attempt_id", "session_id", "candidate_commit", "attempt_context_sha256", "host_clock_id"}
    for name in copied:
        if row[name] != generated[name]:
            _fail(f"{path}.{name}", "must be byte-equivalent to supplied generated event")


def validate_command_sent(
    value: Any,
    path: str = "$command_sent",
    *,
    generated: Any | None = None,
) -> dict[str, Any]:
    row = _object(value, _COMMAND_COMMON_KEYS | {"schema", "sent_monotonic_ns", "generated_event_sequence", "generation_sha256", "transport"}, path)
    _literal(row["schema"], "aigp-vq2-calibration-command-sent/1", f"{path}.schema")
    _validate_command_common(row, path, time_key="sent_monotonic_ns")
    _int(row["generated_event_sequence"], f"{path}.generated_event_sequence")
    _sha256(row["generation_sha256"], f"{path}.generation_sha256")
    if row["generated_event_sequence"] >= row["event_sequence"]:
        _fail(f"{path}.generated_event_sequence", "must name an earlier event")
    transport = _object(row["transport"], {"receipt", "audit_count_before", "audit_count_after"}, f"{path}.transport")
    receipt = validate_attitude_target_outbound(transport["receipt"], f"{path}.transport.receipt")
    before = _int(transport["audit_count_before"], f"{path}.transport.audit_count_before")
    after = _int(transport["audit_count_after"], f"{path}.transport.audit_count_after")
    if after != before + 1:
        _fail(f"{path}.transport.audit_count_after", "must equal audit_count_before + 1")
    if receipt["outcome"] != "returned":
        _fail(f"{path}.transport.receipt.outcome", "sent command requires a returned receipt")
    if generated is not None:
        generated_row = validate_command_generated(generated)
        _require_generated_copy(row, generated_row, path)
        if row["sent_monotonic_ns"] < generated_row["generated_monotonic_ns"]:
            _fail(f"{path}.sent_monotonic_ns", "must not precede generation")
    return defensive_copy(row)


def _validate_not_sent_outcome(value: Any, path: str) -> dict[str, Any]:
    row = _object(value, {"kind", "reason_code", "detail", "audit_count_before", "audit_count_after", "call_started_monotonic_ns", "call_ended_monotonic_ns"}, path)
    kind = _enum(row["kind"], {"skipped_after_generation", "send_failed_or_uncertain"}, f"{path}.kind")
    reason = _enum(row["reason_code"], _COMMAND_FAILURE_CODES, f"{path}.reason_code")
    _sanitized_text(row["detail"], f"{path}.detail", maximum_utf8_bytes=512)
    before = _int(row["audit_count_before"], f"{path}.audit_count_before")
    after = _int(row["audit_count_after"], f"{path}.audit_count_after")
    if kind == "skipped_after_generation":
        if after != before or row["call_started_monotonic_ns"] is not None or row["call_ended_monotonic_ns"] is not None:
            _fail(path, "skip must have unchanged audit and null call boundaries")
        if reason == "send_raised":
            _fail(f"{path}.reason_code", "send_raised is not valid without a call")
    else:
        if reason not in {"send_raised", "deadline_expired", "parent_dead", "lease_invalid", "internal_error"}:
            _fail(f"{path}.reason_code", "not permitted for failed/uncertain call")
        if row["call_started_monotonic_ns"] is None:
            if row["call_ended_monotonic_ns"] is not None or after != before:
                _fail(path, "no-call failure must have null boundaries and unchanged audit")
        else:
            started = _int(row["call_started_monotonic_ns"], f"{path}.call_started_monotonic_ns")
            if after != before + 1:
                _fail(f"{path}.audit_count_after", "started call must increment audit exactly once")
            if row["call_ended_monotonic_ns"] is not None:
                ended = _int(row["call_ended_monotonic_ns"], f"{path}.call_ended_monotonic_ns")
                if ended < started:
                    _fail(f"{path}.call_ended_monotonic_ns", "must not precede call start")
    return row


def validate_command_not_sent(
    value: Any,
    path: str = "$command_not_sent",
    *,
    generated: Any | None = None,
) -> dict[str, Any]:
    row = _object(value, _COMMAND_COMMON_KEYS | {"schema", "recorded_monotonic_ns", "generated_event_sequence", "generation_sha256", "outcome"}, path)
    _literal(row["schema"], "aigp-vq2-calibration-command-not-sent/1", f"{path}.schema")
    _validate_command_common(row, path, time_key="recorded_monotonic_ns")
    _int(row["generated_event_sequence"], f"{path}.generated_event_sequence")
    _sha256(row["generation_sha256"], f"{path}.generation_sha256")
    if row["generated_event_sequence"] >= row["event_sequence"]:
        _fail(f"{path}.generated_event_sequence", "must name an earlier event")
    _validate_not_sent_outcome(row["outcome"], f"{path}.outcome")
    if generated is not None:
        generated_row = validate_command_generated(generated)
        _require_generated_copy(row, generated_row, path)
        if row["recorded_monotonic_ns"] < generated_row["generated_monotonic_ns"]:
            _fail(f"{path}.recorded_monotonic_ns", "must not precede generation")
    return defensive_copy(row)


def validate_tick_disposition(value: Any, path: str = "$tick_disposition") -> dict[str, Any]:
    row = _object(value, {"schema", "attempt_id", "session_id", "attempt_context_sha256", "plan_id", "plan_sha256", "event_sequence", "host_clock_id", "recorded_monotonic_ns", "absolute_tick", "segment_id", "slot", "disposition", "generated_event_sequence", "terminal_event_sequence", "reason_code"}, path)
    _literal(row["schema"], "aigp-vq2-calibration-tick-disposition/1", f"{path}.schema")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _sha256(row["attempt_context_sha256"], f"{path}.attempt_context_sha256")
    _literal(row["plan_id"], EXCITATION_PLAN_ID, f"{path}.plan_id")
    _literal(row["plan_sha256"], EXCITATION_PLAN_SHA256, f"{path}.plan_sha256")
    _int(row["event_sequence"], f"{path}.event_sequence")
    _literal(row["host_clock_id"], HOST_CLOCK_ID, f"{path}.host_clock_id")
    _int(row["recorded_monotonic_ns"], f"{path}.recorded_monotonic_ns")
    tick = _int(row["absolute_tick"], f"{path}.absolute_tick", maximum=244)
    expected = excitation_tick(tick)
    _literal(row["segment_id"], expected["segment_id"], f"{path}.segment_id")
    slot = _validate_slot(row["slot"], f"{path}.slot")
    period = _PLAN_LITERAL["control_period_ns"]
    anchor = slot["release_monotonic_ns"] - tick * period
    if anchor < 0 or slot["end_monotonic_ns"] != slot["release_monotonic_ns"] + period or slot["powered_expiry_monotonic_ns"] != anchor + 5_000_000_000:
        _fail(f"{path}.slot", "must be the exact frozen plan slot")
    disposition = _enum(row["disposition"], {"sent", "skipped_before_generation", "skipped_after_generation"}, f"{path}.disposition")
    if disposition == "sent":
        _int(row["generated_event_sequence"], f"{path}.generated_event_sequence")
        _int(row["terminal_event_sequence"], f"{path}.terminal_event_sequence")
        if row["reason_code"] is not None:
            _fail(f"{path}.reason_code", "must be null for sent disposition")
    elif disposition == "skipped_before_generation":
        if row["generated_event_sequence"] is not None or row["terminal_event_sequence"] is not None:
            _fail(path, "pre-generation skip must have null event links")
        reason = _enum(row["reason_code"], _COMMAND_FAILURE_CODES, f"{path}.reason_code")
        if reason == "send_raised":
            _fail(f"{path}.reason_code", "pre-generation skip cannot be send_raised")
    else:
        _int(row["generated_event_sequence"], f"{path}.generated_event_sequence")
        _int(row["terminal_event_sequence"], f"{path}.terminal_event_sequence")
        _enum(row["reason_code"], _COMMAND_FAILURE_CODES, f"{path}.reason_code")
    for name in ("generated_event_sequence", "terminal_event_sequence"):
        if row[name] is not None and row[name] >= row["event_sequence"]:
            _fail(f"{path}.{name}", "must name an earlier event")
    return defensive_copy(row)


INVALIDATION_REASON_CODES = frozenset(
    {
        "lease_busy", "lease_abandoned", "lease_unverifiable", "launch_failed",
        "topology_failed", "training_unattested", "build_or_candidate_changed",
        "ports_busy", "child_spawn_failed", "child_failed", "child_timeout",
        "wrapper_death", "stream_stale", "watchdog_failed", "capture_incomplete",
        "unexpected_outbound", "command_reconciliation_failed", "deadline_expired",
        "cleanup_unconfirmed", "process_residue", "port_residue",
        "lease_release_unconfirmed", "artifact_mismatch", "terminal_write_failed",
        "internal_error",
    }
)

WRAPPER_PHASES = (
    "attempt_publish", "lease_acquire", "launcher_return",
    "topology_and_training_attestation", "prechild_identity_and_ports",
    "child_spawn", "child_supervision", "child_exit_proof", "fallback_spawn",
    "fallback_supervision", "postcheck_identity_process_ports",
    "lease_release_and_verify", "bundle_verify", "capture_seal", "analysis",
    "split_publish", "terminal_ready", "poison_publish", "invalid_ready",
)


def _wrapper_phase_duration(phase: str) -> int:
    mapped = {
        "child_supervision": "child_total",
        "fallback_supervision": "fallback_total",
        "terminal_ready": "terminal_publish",
        "invalid_ready": "terminal_publish",
    }.get(phase, phase)
    return DEADLINE_DURATIONS_NS[mapped]


def validate_wrapper_event(
    value: Any,
    path: str = "$wrapper_event",
    *,
    prior_file_sha256: str | None = None,
) -> dict[str, Any]:
    row = _object(
        value,
        {"schema", "task_id", "session_id", "attempt_id", "event_sequence", "predecessor_sha256", "event", "phase", "observed_monotonic_ns", "duration_ns", "parent_deadline_monotonic_ns", "deadline_monotonic_ns", "outcome", "reason_code", "artifacts"},
        path,
    )
    _literal(row["schema"], "aigp-vq2-powered-wrapper-event/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    sequence = _int(row["event_sequence"], f"{path}.event_sequence")
    if sequence == 0:
        if row["predecessor_sha256"] is not None:
            _fail(f"{path}.predecessor_sha256", "must be null at sequence zero")
    else:
        predecessor = _sha256(row["predecessor_sha256"], f"{path}.predecessor_sha256")
        if prior_file_sha256 is not None and predecessor != prior_file_sha256:
            _fail(f"{path}.predecessor_sha256", "does not match prior complete-file hash")
    event = _enum(row["event"], {"phase_start", "phase_end"}, f"{path}.event")
    phase = _enum(row["phase"], WRAPPER_PHASES, f"{path}.phase")
    observed = _int(row["observed_monotonic_ns"], f"{path}.observed_monotonic_ns")
    duration = _int(row["duration_ns"], f"{path}.duration_ns", minimum=1)
    if duration != _wrapper_phase_duration(phase):
        _fail(f"{path}.duration_ns", "must equal the frozen phase duration")
    parent = _int(row["parent_deadline_monotonic_ns"], f"{path}.parent_deadline_monotonic_ns")
    deadline = _int(row["deadline_monotonic_ns"], f"{path}.deadline_monotonic_ns")
    if event == "phase_start":
        if observed >= parent or deadline != min(observed + duration, parent):
            _fail(path, "phase-start deadline must be frozen from observed start")
        if row["outcome"] is not None or row["reason_code"] is not None:
            _fail(path, "phase start must have null outcome and reason")
    else:
        outcome = _enum(row["outcome"], {"completed", "failed"}, f"{path}.outcome")
        if observed >= deadline:
            _fail(f"{path}.observed_monotonic_ns", "phase end must occur before frozen deadline")
        if outcome == "completed":
            if row["reason_code"] is not None:
                _fail(f"{path}.reason_code", "must be null for completed phase")
        else:
            _enum(row["reason_code"], INVALIDATION_REASON_CODES, f"{path}.reason_code")
    artifacts = _sorted_unique_objects(row["artifacts"], f"{path}.artifacts", key="name", validator=validate_artifact_ref)
    if event == "phase_start" or row["outcome"] == "failed":
        if artifacts:
            _fail(f"{path}.artifacts", "starts and failed ends must have no artifacts")
    expected_artifacts = {
        "bundle_verify": ["bundle_verification"],
        "capture_seal": ["capture_seal"],
        "split_publish": ["analysis_report", "split_claim", "split_registry"],
        "lease_release_and_verify": ["lease_final"],
        "poison_publish": ["live_poison"],
    }
    if event == "phase_end" and row["outcome"] == "completed":
        names = [item["name"] for item in artifacts]
        expected = expected_artifacts.get(phase, [])
        if names != sorted(expected, key=lambda item: item.encode("utf-8")):
            _fail(f"{path}.artifacts", "does not match the completed-phase artifact set")
    return defensive_copy(row)


def validate_wrapper_lifecycle(
    value: Any,
    *,
    ledger_events: Sequence[Any] | None = None,
) -> dict[str, Any]:
    path = "$wrapper_lifecycle"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "records", "final_sequence", "final_record_sha256", "live_contact_deadline_monotonic_ns", "total_deadline_monotonic_ns"}, path)
    _literal(row["schema"], "aigp-vq2-powered-wrapper-lifecycle/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    records = _array(row["records"], f"{path}.records")
    if not records:
        _fail(f"{path}.records", "must contain at least one ledger record")
    record_keys = {"event_sequence", "path", "sha256", "event", "phase", "observed_monotonic_ns", "outcome", "reason_code", "artifacts"}
    for index, record in enumerate(records):
        item_path = f"{path}.records[{index}]"
        item = _object(record, record_keys, item_path)
        _literal(item["event_sequence"], index, f"{item_path}.event_sequence")
        expected_path = _FROZEN_PATHS["wrapper_ledger_directory"] + f"\\event-{index:06d}.json"
        validate_absolute_windows_path(item["path"], path=f"{item_path}.path", root=EVIDENCE_ROOT)
        if item["path"] != expected_path:
            _fail(f"{item_path}.path", "must be the contiguous frozen ledger path")
        _sha256(item["sha256"], f"{item_path}.sha256")
        _enum(item["event"], {"phase_start", "phase_end"}, f"{item_path}.event")
        _enum(item["phase"], WRAPPER_PHASES, f"{item_path}.phase")
        _int(item["observed_monotonic_ns"], f"{item_path}.observed_monotonic_ns")
        if item["event"] == "phase_start":
            if item["outcome"] is not None or item["reason_code"] is not None:
                _fail(item_path, "start summary must have null outcome/reason")
        else:
            outcome = _enum(item["outcome"], {"completed", "failed"}, f"{item_path}.outcome")
            if outcome == "completed":
                if item["reason_code"] is not None:
                    _fail(f"{item_path}.reason_code", "must be null for completion")
            else:
                _enum(item["reason_code"], INVALIDATION_REASON_CODES, f"{item_path}.reason_code")
        _sorted_unique_objects(item["artifacts"], f"{item_path}.artifacts", key="name", validator=validate_artifact_ref)
    cursor = 0
    prior_phase_rank = -1
    if records[0]["event"] == "phase_end" and records[0]["phase"] == "attempt_publish":
        if records[0]["outcome"] != "completed":
            _fail(f"{path}.records[0]", "the inherited attempt_publish end must be completed")
        prior_phase_rank = WRAPPER_PHASES.index("attempt_publish")
        cursor = 1
    while cursor < len(records):
        if cursor + 1 >= len(records):
            _fail(f"{path}.records[{cursor}]", "phase start is missing its terminal pair")
        start_record = records[cursor]
        end_record = records[cursor + 1]
        if start_record["event"] != "phase_start" or end_record["event"] != "phase_end" or start_record["phase"] != end_record["phase"]:
            _fail(f"{path}.records[{cursor}]", "records after attempt_publish must be exact start/end pairs")
        rank = WRAPPER_PHASES.index(start_record["phase"])
        if rank <= prior_phase_rank:
            _fail(f"{path}.records[{cursor}].phase", "wrapper phases must be unique and follow frozen order")
        if start_record["observed_monotonic_ns"] > end_record["observed_monotonic_ns"]:
            _fail(f"{path}.records[{cursor + 1}].observed_monotonic_ns", "phase end must not precede phase start")
        prior_phase_rank = rank
        cursor += 2
    _literal(row["final_sequence"], len(records) - 1, f"{path}.final_sequence")
    _sha256(row["final_record_sha256"], f"{path}.final_record_sha256")
    if row["final_record_sha256"] != records[-1]["sha256"]:
        _fail(f"{path}.final_record_sha256", "must equal final record hash")
    live = _int(row["live_contact_deadline_monotonic_ns"], f"{path}.live_contact_deadline_monotonic_ns")
    total = _int(row["total_deadline_monotonic_ns"], f"{path}.total_deadline_monotonic_ns")
    if total - live != DEADLINE_DURATIONS_NS["wrapper_total"] - DEADLINE_DURATIONS_NS["wrapper_live_contact_absolute_offset"]:
        _fail(path, "wrapper absolute deadlines must retain the frozen 90-second difference")
    if ledger_events is not None:
        events = list(ledger_events)
        if len(events) != len(records):
            _fail(f"{path}.records", "does not cover supplied ledger events")
        prior_hash: str | None = None
        for index, event in enumerate(events):
            checked = validate_wrapper_event(event, f"$ledger_events[{index}]", prior_file_sha256=prior_hash)
            file_hash = canonical_file_sha256(checked)
            record = records[index]
            copied = {key: checked[key] for key in record_keys if key not in {"path", "sha256"}}
            observed = {key: record[key] for key in record_keys if key not in {"path", "sha256"}}
            if observed != copied or record["sha256"] != file_hash:
                _fail(f"{path}.records[{index}]", "does not byte/hash bind supplied ledger event")
            prior_hash = file_hash
    return defensive_copy(row)


_CHILD_PHASES = (
    "connect", "preflight", "reset_epoch", "normalize_disarmed", "countdown_go",
    "arm", "powered_stage", "cleanup", "replay_close", "finalize",
)
_FALLBACK_PHASES = ("connect", "disarm", "reset_and_epoch", "finalize")


def _expected_process_phase_duration(role: str, phase: str) -> int:
    if phase == "parent_death_lease_takeover":
        return DEADLINE_DURATIONS_NS[phase]
    if role == "powered_child":
        name = "powered_stage" if phase == "powered_stage" else f"child_{phase}"
    else:
        name = f"fallback_{phase}"
    return DEADLINE_DURATIONS_NS[name]


def _validate_process_phase_deadlines(value: Any, path: str, *, role: str) -> list[Any]:
    rows = _array(value, path)
    base = _CHILD_PHASES if role == "powered_child" else _FALLBACK_PHASES
    last_rank = -1
    takeover_seen = False
    previous_start = -1
    for index, item in enumerate(rows):
        checked = validate_phase_deadline(item, f"{path}[{index}]")
        phase = checked["phase"]
        if phase == "parent_death_lease_takeover":
            if takeover_seen:
                _fail(f"{path}[{index}].phase", "takeover may appear at most once")
            takeover_seen = True
        else:
            if phase not in base:
                _fail(f"{path}[{index}].phase", "phase is not valid for producer role")
            rank = base.index(phase)
            if rank <= last_rank:
                _fail(path, "process phases must be ordered and unique")
            last_rank = rank
        if checked["duration_ns"] != _expected_process_phase_duration(role, phase):
            _fail(f"{path}[{index}].duration_ns", "must equal frozen role phase duration")
        if checked["started_monotonic_ns"] < previous_start:
            _fail(path, "phase starts must be monotonic")
        previous_start = checked["started_monotonic_ns"]
    return rows


def validate_outbound_audit(value: Any, path: str = "$outbound_audit") -> dict[str, Any]:
    keys = {"timesync", "gcs_heartbeat", "sim_reset", "arm", "disarm", "attitude_target", "position_target", "other_command", "receipt_count", "receipt_returned", "receipt_raised", "receipt_dropped", "receipt_buffered"}
    row = _object(value, keys, path)
    for name in keys:
        _int(row[name], f"{path}.{name}")
    if row["receipt_count"] != row["receipt_returned"] + row["receipt_raised"]:
        _fail(path, "receipt_count must equal returned plus raised receipts")
    attempted = sum(row[name] for name in ("timesync", "gcs_heartbeat", "sim_reset", "arm", "disarm", "attitude_target", "position_target", "other_command"))
    if attempted != row["receipt_count"] + row["receipt_dropped"]:
        _fail(path, "outbound category total must equal recorded plus dropped receipts")
    if row["receipt_buffered"] > row["receipt_count"]:
        _fail(path, "buffered receipts cannot exceed recorded receipt count")
    return row


_CLEANUP_FAILURE_CODES = frozenset(
    {"authority_invalid", "deadline_expired", "parent_dead", "lease_invalid", "connect_failed", "zero_failed", "disarm_failed", "reset_failed", "final_state_unproved", "transport_unclosed", "receipt_incomplete", "internal_error"}
)
_COLLECTION_INVALIDATING_CODES = frozenset(
    {"camera_missing", "collision_observed", "source_rejected", "unexpected_outbound"}
)


def _sorted_unique_enum(value: Any, path: str, allowed: Iterable[str], *, nonempty: bool = False) -> list[str]:
    items = _sorted_unique_strings(value, path)
    choices = frozenset(allowed)
    if nonempty and not items:
        _fail(path, "must not be empty")
    unknown = set(items) - choices
    if unknown:
        _fail(path, f"contains unsupported values {sorted(unknown)!r}")
    return items


def _validate_endpoint(value: Any, path: str, *, camera: bool, producer_role: str) -> dict[str, Any]:
    row = _object(value, {"state", "bind", "frozen_peer", "rejected_source_count"}, path)
    state = _enum(row["state"], {"not_opened", "bound", "peer_frozen", "closed_without_peer", "closed_with_peer"}, f"{path}.state")
    _int(row["rejected_source_count"], f"{path}.rejected_source_count")
    if state == "not_opened":
        if row["bind"] is not None or row["frozen_peer"] is not None:
            _fail(path, "not-opened endpoint must have null bind and peer")
        return row
    bind = _object(row["bind"], {"role", "family", "requested", "actual", "socket_policy", "owner_process"}, f"{path}.bind")
    expected_role = "camera" if camera else "mavlink"
    _literal(bind["role"], expected_role, f"{path}.bind.role")
    _literal(bind["family"], "AF_INET", f"{path}.bind.family")
    expected_host, expected_port = (("0.0.0.0", 5600) if camera else ("127.0.0.1", 14550))
    for name in ("requested", "actual"):
        endpoint = _object(bind[name], {"host", "port"}, f"{path}.bind.{name}")
        _literal(endpoint["host"], expected_host, f"{path}.bind.{name}.host")
        _literal(endpoint["port"], expected_port, f"{path}.bind.{name}.port")
    _literal(bind["socket_policy"], "ipv4-exclusive-address-use", f"{path}.bind.socket_policy")
    validate_process_identity(bind["owner_process"], f"{path}.bind.owner_process")
    if state in {"peer_frozen", "closed_with_peer"}:
        peer = _object(row["frozen_peer"], {"host", "port"}, f"{path}.frozen_peer")
        host = _string(peer["host"], f"{path}.frozen_peer.host")
        if host != "127.0.0.1":
            _fail(f"{path}.frozen_peer.host", "must be exact IPv4 loopback")
        _int(peer["port"], f"{path}.frozen_peer.port", minimum=1, maximum=_UINT16_MAX)
    elif row["frozen_peer"] is not None:
        _fail(f"{path}.frozen_peer", "must be null before peer freeze")
    if producer_role == "cleanup_fallback" and camera:
        _fail(path, "fallback camera endpoint must be null, not an object")
    return row


def _receipt_member(receipt: Any, receipts: list[Any], *, outcome: str | None = None) -> bool:
    if receipt is None:
        return False
    return any(item == receipt and (outcome is None or item["outcome"] == outcome) for item in receipts)


def _validate_zero_command(value: Any, path: str, receipts: list[Any]) -> dict[str, Any]:
    row = _object(value, {"state", "required", "requested", "generated", "terminal", "outbound_receipt"}, path)
    state = _enum(row["state"], {"not_required", "not_attempted", "failed", "returned"}, f"{path}.state")
    required = _bool(row["required"], f"{path}.required")
    if state == "not_required":
        if required or any(row[name] is not None for name in ("requested", "generated", "terminal", "outbound_receipt")):
            _fail(path, "not-required zero must have false required and null evidence")
        return row
    if not required:
        _fail(f"{path}.required", "must be true for this zero state")
    requested = _validate_command(row["requested"], f"{path}.requested")
    if any(requested[name] != 0 for name in requested):
        _fail(f"{path}.requested", "must be the exact zero command")
    if state == "not_attempted":
        if any(row[name] is not None for name in ("generated", "terminal", "outbound_receipt")):
            _fail(path, "not-attempted zero must have null generated/terminal/receipt")
        return row
    generated = validate_command_generated(row["generated"], f"{path}.generated")
    if generated["scope"] != "cleanup_zero":
        _fail(f"{path}.generated.scope", "must be cleanup_zero")
    if state == "returned":
        terminal = validate_command_sent(row["terminal"], f"{path}.terminal", generated=generated)
        receipt = validate_attitude_target_outbound(row["outbound_receipt"], f"{path}.outbound_receipt")
        if terminal["transport"]["receipt"] != receipt or not _receipt_member(receipt, receipts, outcome="returned"):
            _fail(path, "returned zero receipt must match terminal and complete receipt array")
    else:
        terminal = validate_command_not_sent(row["terminal"], f"{path}.terminal", generated=generated)
        receipt = row["outbound_receipt"]
        call_started = terminal["outcome"]["call_started_monotonic_ns"] is not None
        call_ended = terminal["outcome"]["call_ended_monotonic_ns"] is not None
        if receipt is None:
            if call_started and call_ended:
                _fail(f"{path}.outbound_receipt", "observed raise must have its raised receipt")
        else:
            checked = validate_attitude_target_outbound(receipt, f"{path}.outbound_receipt")
            if checked["outcome"] != "raised" or not _receipt_member(checked, receipts, outcome="raised"):
                _fail(f"{path}.outbound_receipt", "failed zero receipt must be a complete raised receipt")
    return row


def _validate_disarm(value: Any, path: str, receipts: list[Any]) -> dict[str, Any]:
    row = _object(value, {"state", "request_monotonic_ns", "receipt", "heartbeat_before", "heartbeat_after", "newer_confirmed"}, path)
    state = _enum(row["state"], {"not_required", "not_attempted", "request_failed", "unconfirmed", "confirmed"}, f"{path}.state")
    _bool(row["newer_confirmed"], f"{path}.newer_confirmed")
    evidence = ("request_monotonic_ns", "receipt", "heartbeat_before", "heartbeat_after")
    if state in {"not_required", "not_attempted"}:
        if any(row[name] is not None for name in evidence) or row["newer_confirmed"]:
            _fail(path, "unstarted disarm state must have null evidence and false confirmation")
        return row
    request = _int(row["request_monotonic_ns"], f"{path}.request_monotonic_ns")
    before = validate_received_heartbeat(row["heartbeat_before"], f"{path}.heartbeat_before")
    before_ingress = before["ingress"]
    if before_ingress["received_monotonic_ns"] > request:
        _fail(f"{path}.heartbeat_before.ingress.received_monotonic_ns", "must not follow the disarm request")
    if state == "request_failed":
        if row["heartbeat_after"] is not None or row["newer_confirmed"]:
            _fail(path, "failed request must have null heartbeat-after and false confirmation")
        if row["receipt"] is not None:
            receipt = validate_nonattitude_outbound(row["receipt"], f"{path}.receipt")
            if receipt["category"] != "disarm" or receipt["outcome"] != "raised" or not _receipt_member(receipt, receipts, outcome="raised"):
                _fail(f"{path}.receipt", "must be the complete raised disarm receipt")
            if receipt["reset_generation"] != before_ingress["generation"]:
                _fail(f"{path}.receipt.reset_generation", "must match heartbeat-before generation")
            if receipt["stream_id"] != before_ingress["stream_id"]:
                _fail(f"{path}.receipt.stream_id", "must match heartbeat-before stream")
            if receipt["call_start_monotonic_ns"] < request:
                _fail(f"{path}.receipt", "receipt call cannot precede request")
        return row
    receipt = validate_nonattitude_outbound(row["receipt"], f"{path}.receipt")
    if receipt["category"] != "disarm" or receipt["outcome"] != "returned" or not _receipt_member(receipt, receipts, outcome="returned"):
        _fail(f"{path}.receipt", "must be the complete returned disarm receipt")
    if receipt["reset_generation"] != before_ingress["generation"]:
        _fail(f"{path}.receipt.reset_generation", "must match heartbeat-before generation")
    if receipt["stream_id"] != before_ingress["stream_id"]:
        _fail(f"{path}.receipt.stream_id", "must match heartbeat-before stream")
    if receipt["call_start_monotonic_ns"] < request:
        _fail(f"{path}.receipt", "receipt call cannot precede request")
    after = None if row["heartbeat_after"] is None else validate_received_heartbeat(row["heartbeat_after"], f"{path}.heartbeat_after")
    if after is not None:
        after_ingress = after["ingress"]
        if after_ingress["generation"] != before_ingress["generation"]:
            _fail(f"{path}.heartbeat_after.ingress.generation", "must match heartbeat-before generation")
        if after_ingress["stream_id"] != before_ingress["stream_id"]:
            _fail(f"{path}.heartbeat_after.ingress.stream_id", "must match heartbeat-before stream")
        if (
            after_ingress["sequence"] <= before_ingress["sequence"]
            or after_ingress["received_monotonic_ns"] <= request
            or after_ingress["received_monotonic_ns"] < receipt["call_start_monotonic_ns"]
        ):
            _fail(f"{path}.heartbeat_after", "must be strictly newer and post-request")
    if state == "unconfirmed":
        if row["newer_confirmed"]:
            _fail(f"{path}.newer_confirmed", "must be false for unconfirmed state")
    else:
        if after is None or not row["newer_confirmed"]:
            _fail(path, "confirmed disarm requires heartbeat-after and literal true")
        if after["heartbeat"]["base_mode"] & 128:
            _fail(f"{path}.heartbeat_after", "armed bit must be clear")
    return row


def _validate_reset(value: Any, path: str, receipts: list[Any]) -> dict[str, Any]:
    row = _object(value, {"state", "request_monotonic_ns", "receipt", "boundary", "baseline", "clean_epoch", "advancing_race", "advancing_imu", "rollback_and_advance_confirmed"}, path)
    state = _enum(row["state"], {"not_required", "not_attempted", "request_failed", "unconfirmed", "confirmed"}, f"{path}.state")
    _bool(row["rollback_and_advance_confirmed"], f"{path}.rollback_and_advance_confirmed")
    races = _array(row["advancing_race"], f"{path}.advancing_race")
    imus = _array(row["advancing_imu"], f"{path}.advancing_imu")
    scalar = ("request_monotonic_ns", "receipt", "boundary", "baseline", "clean_epoch")
    if state in {"not_required", "not_attempted"}:
        if any(row[name] is not None for name in scalar) or races or imus or row["rollback_and_advance_confirmed"]:
            _fail(path, "unstarted reset state must have null/empty evidence and false confirmation")
        return row
    request = _int(row["request_monotonic_ns"], f"{path}.request_monotonic_ns")
    boundary = validate_reset_boundary(row["boundary"], f"{path}.boundary")
    boundary_time = boundary["boundary_monotonic_ns"]
    if boundary_time < request:
        _fail(f"{path}.boundary.boundary_monotonic_ns", "must not precede the reset request")
    baseline = _object(row["baseline"], {"race", "imu"}, f"{path}.baseline")
    baseline_race = validate_received_race_status(baseline["race"], f"{path}.baseline.race")
    baseline_imu = validate_received_imu(baseline["imu"], f"{path}.baseline.imu")
    baseline_stream = baseline_race["ingress"]["stream_id"]
    if baseline_race["ingress"]["generation"] != boundary["old_generation"]:
        _fail(f"{path}.baseline.race.ingress.generation", "must equal boundary old generation")
    if baseline_imu["ingress"]["generation"] != boundary["old_generation"]:
        _fail(f"{path}.baseline.imu.ingress.generation", "must equal boundary old generation")
    if baseline_imu["ingress"]["stream_id"] != baseline_stream:
        _fail(f"{path}.baseline.imu.ingress.stream_id", "must match baseline race stream")
    for name, observation in (("race", baseline_race), ("imu", baseline_imu)):
        received = observation["ingress"]["received_monotonic_ns"]
        if received > request or received > boundary_time:
            _fail(f"{path}.baseline.{name}.ingress.received_monotonic_ns", "must not follow the reset request or boundary")
    for index, observation in enumerate(boundary["observations"]):
        if observation["ingress"]["stream_id"] != baseline_stream:
            _fail(f"{path}.boundary.observations[{index}].ingress.stream_id", "must match reset baseline stream")

    def validate_reset_receipt(receipt_value: Any, *, outcome: str) -> dict[str, Any]:
        checked = validate_nonattitude_outbound(receipt_value, f"{path}.receipt")
        if checked["category"] != "sim_reset" or checked["outcome"] != outcome or not _receipt_member(checked, receipts, outcome=outcome):
            _fail(f"{path}.receipt", f"must be the complete {outcome} reset receipt")
        if checked["reset_generation"] != boundary["new_generation"]:
            _fail(f"{path}.receipt.reset_generation", "must equal boundary new generation")
        if checked["stream_id"] != baseline_stream:
            _fail(f"{path}.receipt.stream_id", "must match reset baseline stream")
        if checked["call_start_monotonic_ns"] < request:
            _fail(f"{path}.receipt", "receipt call cannot precede request")
        if checked["call_start_monotonic_ns"] < boundary_time:
            _fail(f"{path}.receipt.call_start_monotonic_ns", "must not precede the reset boundary")
        return checked

    if state == "request_failed":
        if row["clean_epoch"] is not None or races or imus or row["rollback_and_advance_confirmed"]:
            _fail(path, "failed reset request must not contain proof evidence")
        if row["receipt"] is not None:
            validate_reset_receipt(row["receipt"], outcome="raised")
        return row
    reset_receipt = validate_reset_receipt(row["receipt"], outcome="returned")
    reset_proof_floor = reset_receipt["call_start_monotonic_ns"]
    checked_races: list[dict[str, Any]] = []
    for index, item in enumerate(races):
        checked = validate_received_race_status(item, f"{path}.advancing_race[{index}]")
        ingress = checked["ingress"]
        if ingress["generation"] != boundary["new_generation"]:
            _fail(f"{path}.advancing_race[{index}].ingress.generation", "must equal boundary new generation")
        if ingress["stream_id"] != baseline_stream:
            _fail(f"{path}.advancing_race[{index}].ingress.stream_id", "must match reset baseline stream")
        if (
            ingress["received_monotonic_ns"] <= request
            or ingress["received_monotonic_ns"] < boundary_time
            or ingress["received_monotonic_ns"] < reset_proof_floor
        ):
            _fail(f"{path}.advancing_race[{index}].ingress.received_monotonic_ns", "must follow reset request, boundary, and reset call start")
        checked_races.append(checked)
    checked_imus: list[dict[str, Any]] = []
    for index, item in enumerate(imus):
        checked = validate_received_imu(item, f"{path}.advancing_imu[{index}]")
        ingress = checked["ingress"]
        if ingress["generation"] != boundary["new_generation"]:
            _fail(f"{path}.advancing_imu[{index}].ingress.generation", "must equal boundary new generation")
        if ingress["stream_id"] != baseline_stream:
            _fail(f"{path}.advancing_imu[{index}].ingress.stream_id", "must match reset baseline stream")
        if (
            ingress["received_monotonic_ns"] <= request
            or ingress["received_monotonic_ns"] < boundary_time
            or ingress["received_monotonic_ns"] < reset_proof_floor
        ):
            _fail(f"{path}.advancing_imu[{index}].ingress.received_monotonic_ns", "must follow reset request, boundary, and reset call start")
        checked_imus.append(checked)
    for name, items in (
        ("advancing_race", checked_races),
        ("advancing_imu", checked_imus),
    ):
        sequences = [item["ingress"]["sequence"] for item in items]
        if any(later <= earlier for earlier, later in zip(sequences, sequences[1:])):
            _fail(f"{path}.{name}", "must preserve strictly increasing ingress occurrence order")
    ordered_observations = sorted(
        checked_races + checked_imus,
        key=lambda item: item["ingress"]["sequence"],
    )
    ordered_sequences = [item["ingress"]["sequence"] for item in ordered_observations]
    if len(set(ordered_sequences)) != len(ordered_sequences):
        _fail(path, "advancing race/IMU observations must have unique global ingress sequences")
    ordered_receipts = [
        item["ingress"]["received_monotonic_ns"] for item in ordered_observations
    ]
    if any(later < earlier for earlier, later in zip(ordered_receipts, ordered_receipts[1:])):
        _fail(path, "advancing race/IMU ingress order must preserve host receipt order")
    epoch = None if row["clean_epoch"] is None else _validate_reset_epoch(row["clean_epoch"], f"{path}.clean_epoch")
    if epoch is not None:
        if epoch["ingress_generation"] != boundary["new_generation"]:
            _fail(f"{path}.clean_epoch.ingress_generation", "must equal boundary new generation")
        if epoch["race_anchor_boot_ms"] >= baseline_race["race_status"]["sim_boot_time_ms"]:
            _fail(f"{path}.clean_epoch.race_anchor_boot_ms", "must prove rollback below the race baseline")
        if epoch["imu_anchor_usec"] >= baseline_imu["imu"]["timestamp_us"]:
            _fail(f"{path}.clean_epoch.imu_anchor_usec", "must prove rollback below the IMU baseline")
    if state == "unconfirmed":
        if row["rollback_and_advance_confirmed"]:
            _fail(f"{path}.rollback_and_advance_confirmed", "must be false for unconfirmed state")
    else:
        if epoch is None or len(races) < 2 or len(imus) < 2 or not row["rollback_and_advance_confirmed"]:
            _fail(path, "confirmed reset requires epoch, two advancing race/IMU samples, and literal true")
        race_times = [item["race_status"]["sim_boot_time_ms"] for item in checked_races]
        imu_times = [item["imu"]["timestamp_us"] for item in checked_imus]
        if race_times[0] <= epoch["race_anchor_boot_ms"] or imu_times[0] <= epoch["imu_anchor_usec"] or any(b <= a for a, b in zip(race_times, race_times[1:])) or any(b <= a for a, b in zip(imu_times, imu_times[1:])):
            _fail(path, "confirmed reset samples must strictly advance after anchors")
    return row


def validate_cleanup_certificate(value: Any) -> dict[str, Any]:
    path = "$cleanup_certificate"
    row = _object(
        value,
        {"schema", "task_id", "session_id", "attempt_id", "producer_role", "cleanup_epoch", "authority", "trigger", "started_monotonic_ns", "deadline_monotonic_ns", "completed_monotonic_ns", "parent_state", "lease", "phase_deadlines", "endpoints", "outbound_receipts", "zero_command", "disarm", "reset", "collisions", "final_state", "transport", "outcome", "failure_codes", "collection_invalidating_codes"},
        path,
    )
    _literal(row["schema"], "aigp-vq2-powered-cleanup-certificate/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    role = _enum(row["producer_role"], {"powered_child", "cleanup_fallback"}, f"{path}.producer_role")
    _literal(row["cleanup_epoch"], "child-cleanup-0" if role == "powered_child" else "fallback-cleanup-0", f"{path}.cleanup_epoch")
    _enum(row["trigger"], {"normal_completion", "stage_abort", "parent_death", "wrapper_fallback"}, f"{path}.trigger")
    start = _int(row["started_monotonic_ns"], f"{path}.started_monotonic_ns")
    deadline = _int(row["deadline_monotonic_ns"], f"{path}.deadline_monotonic_ns")
    completed = _int(row["completed_monotonic_ns"], f"{path}.completed_monotonic_ns")
    if not start <= completed < deadline:
        _fail(path, "cleanup must complete after start and before deadline")
    authority = _object(row["authority"], {"process_authority", "attempt_context_sha256", "attempt_envelope_sha256", "producer"}, f"{path}.authority")
    validate_identity_ref(authority["process_authority"], f"{path}.authority.process_authority")
    _sha256(authority["attempt_context_sha256"], f"{path}.authority.attempt_context_sha256")
    _sha256(authority["attempt_envelope_sha256"], f"{path}.authority.attempt_envelope_sha256")
    validate_process_identity(authority["producer"], f"{path}.authority.producer")
    parent = _object(row["parent_state"], {"mode", "wrapper_process", "observed_monotonic_ns", "takeover_completed_monotonic_ns", "takeover_lease_record_sha256"}, f"{path}.parent_state")
    mode = _enum(parent["mode"], {"live_delegation", "signaled_takeover"}, f"{path}.parent_state.mode")
    validate_process_identity(parent["wrapper_process"], f"{path}.parent_state.wrapper_process")
    observed_parent = _int(parent["observed_monotonic_ns"], f"{path}.parent_state.observed_monotonic_ns")
    if observed_parent > completed:
        _fail(
            f"{path}.parent_state.observed_monotonic_ns",
            "must not follow certificate completion",
        )
    if mode == "live_delegation":
        if parent["takeover_completed_monotonic_ns"] is not None or parent["takeover_lease_record_sha256"] is not None:
            _fail(f"{path}.parent_state", "live delegation must have null takeover fields")
    else:
        takeover = _int(parent["takeover_completed_monotonic_ns"], f"{path}.parent_state.takeover_completed_monotonic_ns")
        _sha256(parent["takeover_lease_record_sha256"], f"{path}.parent_state.takeover_lease_record_sha256")
        if takeover <= observed_parent:
            _fail(f"{path}.parent_state.takeover_completed_monotonic_ns", "must follow parent-death observation")
        if takeover > completed:
            _fail(
                f"{path}.parent_state.takeover_completed_monotonic_ns",
                "must not follow certificate completion",
            )
    lease = _object(row["lease"], {"owner_role", "generation", "record_sha256", "authority_valid"}, f"{path}.lease")
    owner = _enum(lease["owner_role"], {"wrapper", "powered-child-parent-death", "cleanup-fallback-parent-death"}, f"{path}.lease.owner_role")
    _int(lease["generation"], f"{path}.lease.generation")
    _sha256(lease["record_sha256"], f"{path}.lease.record_sha256")
    _bool(lease["authority_valid"], f"{path}.lease.authority_valid")
    if mode == "live_delegation" and owner != "wrapper":
        _fail(f"{path}.lease.owner_role", "live delegation requires wrapper ownership")
    if mode == "signaled_takeover" and owner != ("powered-child-parent-death" if role == "powered_child" else "cleanup-fallback-parent-death"):
        _fail(f"{path}.lease.owner_role", "takeover owner must match producer role")
    phase_deadlines = _validate_process_phase_deadlines(
        row["phase_deadlines"],
        f"{path}.phase_deadlines",
        role=role,
    )
    if any(item["started_monotonic_ns"] > completed for item in phase_deadlines):
        _fail(
            f"{path}.phase_deadlines",
            "phase start must not follow certificate completion",
        )
    endpoints = _object(row["endpoints"], {"mavlink", "camera"}, f"{path}.endpoints")
    _validate_endpoint(endpoints["mavlink"], f"{path}.endpoints.mavlink", camera=False, producer_role=role)
    if role == "cleanup_fallback":
        if endpoints["camera"] is not None:
            _fail(f"{path}.endpoints.camera", "fallback camera must be null")
    else:
        _validate_endpoint(endpoints["camera"], f"{path}.endpoints.camera", camera=True, producer_role=role)
    receipts = _array(row["outbound_receipts"], f"{path}.outbound_receipts")
    prior_sequence = -1
    for index, receipt in enumerate(receipts):
        checked = _validate_outbound_receipt(receipt, f"{path}.outbound_receipts[{index}]")
        if checked["outbound_sequence"] <= prior_sequence:
            _fail(f"{path}.outbound_receipts", "must preserve strictly increasing outbound sequence")
        if checked["call_end_monotonic_ns"] > completed:
            _fail(
                f"{path}.outbound_receipts[{index}].call_end_monotonic_ns",
                "must not follow certificate completion",
            )
        prior_sequence = checked["outbound_sequence"]
    zero = _validate_zero_command(row["zero_command"], f"{path}.zero_command", receipts)
    zero_required = (
        role == "cleanup_fallback"
        or endpoints["mavlink"]["state"] in {"peer_frozen", "closed_with_peer"}
        or any(
            receipt["schema"] == "aigp-vq2-attitude-target-outbound/1"
            or receipt.get("category") == "arm"
            for receipt in receipts
        )
    )
    if zero["required"] != zero_required:
        _fail(f"{path}.zero_command.required", f"must equal derived value {zero_required}")
    disarm = _validate_disarm(row["disarm"], f"{path}.disarm", receipts)
    reset = _validate_reset(row["reset"], f"{path}.reset", receipts)

    def require_cleanup_window(value: int, value_path: str) -> None:
        if value < start or value > completed:
            _fail(value_path, "must be within the cleanup certificate interval")

    if zero["generated"] is not None:
        require_cleanup_window(
            zero["generated"]["generated_monotonic_ns"],
            f"{path}.zero_command.generated.generated_monotonic_ns",
        )
    if zero["terminal"] is not None:
        terminal = zero["terminal"]
        terminal_time_name = (
            "sent_monotonic_ns"
            if terminal["schema"] == "aigp-vq2-calibration-command-sent/1"
            else "recorded_monotonic_ns"
        )
        require_cleanup_window(
            terminal[terminal_time_name],
            f"{path}.zero_command.terminal.{terminal_time_name}",
        )
        if terminal["schema"] == "aigp-vq2-calibration-command-not-sent/1":
            for name in ("call_started_monotonic_ns", "call_ended_monotonic_ns"):
                if terminal["outcome"][name] is not None:
                    require_cleanup_window(
                        terminal["outcome"][name],
                        f"{path}.zero_command.terminal.outcome.{name}",
                    )
    if zero["outbound_receipt"] is not None:
        require_cleanup_window(
            zero["outbound_receipt"]["call_start_monotonic_ns"],
            f"{path}.zero_command.outbound_receipt.call_start_monotonic_ns",
        )
    if disarm["request_monotonic_ns"] is not None:
        require_cleanup_window(
            disarm["request_monotonic_ns"],
            f"{path}.disarm.request_monotonic_ns",
        )
    if disarm["receipt"] is not None:
        require_cleanup_window(
            disarm["receipt"]["call_start_monotonic_ns"],
            f"{path}.disarm.receipt.call_start_monotonic_ns",
        )
    if reset["request_monotonic_ns"] is not None:
        require_cleanup_window(
            reset["request_monotonic_ns"],
            f"{path}.reset.request_monotonic_ns",
        )
    if reset["boundary"] is not None:
        require_cleanup_window(
            reset["boundary"]["boundary_monotonic_ns"],
            f"{path}.reset.boundary.boundary_monotonic_ns",
        )
    if reset["receipt"] is not None:
        require_cleanup_window(
            reset["receipt"]["call_start_monotonic_ns"],
            f"{path}.reset.receipt.call_start_monotonic_ns",
        )
    if (
        zero["outbound_receipt"] is not None
        and disarm["request_monotonic_ns"] is not None
        and zero["outbound_receipt"]["call_end_monotonic_ns"]
        > disarm["request_monotonic_ns"]
    ):
        _fail(
            f"{path}.zero_command.outbound_receipt.call_end_monotonic_ns",
            "must not follow the disarm request",
        )
    if (
        disarm["receipt"] is not None
        and reset["request_monotonic_ns"] is not None
        and disarm["receipt"]["call_end_monotonic_ns"]
        > reset["request_monotonic_ns"]
    ):
        _fail(
            f"{path}.disarm.receipt.call_end_monotonic_ns",
            "must not follow the reset request",
        )
    if reset["boundary"] is not None and disarm["heartbeat_before"] is not None:
        reset_stream = reset["baseline"]["race"]["ingress"]["stream_id"]
        if disarm["heartbeat_before"]["ingress"]["generation"] != reset["boundary"]["old_generation"]:
            _fail(
                f"{path}.disarm.heartbeat_before.ingress.generation",
                "must equal reset boundary old generation",
            )
        if disarm["heartbeat_before"]["ingress"]["stream_id"] != reset_stream:
            _fail(
                f"{path}.disarm.heartbeat_before.ingress.stream_id",
                "must match the reset baseline stream",
            )
        for index, receipt in enumerate(receipts):
            if receipt["stream_id"] != reset_stream:
                _fail(
                    f"{path}.outbound_receipts[{index}].stream_id",
                    "all cleanup evidence must bind one adapter stream",
                )
        zero_receipt = zero["outbound_receipt"]
        if (
            zero_receipt is not None
            and zero_receipt["reset_generation"]
            != reset["boundary"]["old_generation"]
        ):
            _fail(
                f"{path}.zero_command.outbound_receipt.reset_generation",
                "must equal reset boundary old generation",
            )
        if (
            disarm["heartbeat_after"] is not None
            and disarm["heartbeat_after"]["ingress"]["received_monotonic_ns"]
            > reset["request_monotonic_ns"]
        ):
            _fail(
                f"{path}.disarm.heartbeat_after.ingress.received_monotonic_ns",
                "must not follow the reset request",
            )
    collisions = _object(row["collisions"], {"observations", "invalidating_occurrence_count"}, f"{path}.collisions")
    collision_rows = _array(collisions["observations"], f"{path}.collisions.observations")
    checked_collision_rows: list[dict[str, Any]] = []
    collision_tokens: set[tuple[int, int]] = set()
    collision_order: dict[int, tuple[int, int]] = {}
    for index, item in enumerate(collision_rows):
        checked = validate_collision_observation(item, f"{path}.collisions.observations[{index}]")
        if checked["observed_monotonic_ns"] > completed:
            _fail(
                f"{path}.collisions.observations[{index}].observed_monotonic_ns",
                "must not follow certificate completion",
            )
        token = (checked["reset_generation"], checked["observation_sequence"])
        if token in collision_tokens:
            _fail(f"{path}.collisions.observations[{index}]", "duplicates a collision occurrence token")
        collision_tokens.add(token)
        prior = collision_order.get(checked["reset_generation"])
        current = (checked["observation_sequence"], checked["observed_monotonic_ns"])
        if prior is not None and (current[0] <= prior[0] or current[1] < prior[1]):
            _fail(f"{path}.collisions.observations[{index}]", "must preserve per-generation sequence and host-time order")
        collision_order[checked["reset_generation"]] = current
        checked_collision_rows.append(checked)
    if reset["boundary"] is not None:
        old_generation = reset["boundary"]["old_generation"]
        for index, item in enumerate(reset["boundary"]["collisions"]):
            if sum(candidate == item for candidate in checked_collision_rows) != 1:
                _fail(
                    f"{path}.reset.boundary.collisions[{index}]",
                    "must appear exactly once in the complete cleanup collision array",
                )
        old_rows = sum(
            item["reset_generation"] == old_generation
            for item in checked_collision_rows
        )
        collision_stats = reset["boundary"]["collision_stats"]
        if old_rows + collision_stats["dropped"] != collision_stats["handled"]:
            _fail(
                f"{path}.reset.boundary.collision_stats",
                "handled must equal represented old-generation collisions plus dropped",
            )
    _literal(collisions["invalidating_occurrence_count"], len(collision_rows), f"{path}.collisions.invalidating_occurrence_count")
    final = _object(row["final_state"], {"state", "heartbeat", "disarmed", "reset_epoch", "last_race", "last_imu"}, f"{path}.final_state")
    final_state = _enum(final["state"], {"unobserved", "partial", "confirmed"}, f"{path}.final_state.state")
    facts = ("heartbeat", "disarmed", "reset_epoch", "last_race", "last_imu")
    if final_state == "unobserved":
        if any(final[name] is not None for name in facts):
            _fail(f"{path}.final_state", "unobserved final state must have null facts")
    else:
        if any(final[name] is None for name in facts):
            _fail(f"{path}.final_state", "partial/confirmed final state requires all facts")
        final_heartbeat = validate_received_heartbeat(final["heartbeat"], f"{path}.final_state.heartbeat")
        _bool(final["disarmed"], f"{path}.final_state.disarmed")
        final_epoch = _validate_reset_epoch(final["reset_epoch"], f"{path}.final_state.reset_epoch")
        final_race = validate_received_race_status(final["last_race"], f"{path}.final_state.last_race")
        final_imu = validate_received_imu(final["last_imu"], f"{path}.final_state.last_imu")
        heartbeat_disarmed = final_heartbeat["heartbeat"]["base_mode"] & 128 == 0
        if final["disarmed"] is not heartbeat_disarmed:
            _fail(f"{path}.final_state.disarmed", "must equal the final heartbeat armed-bit derivation")

        reset_confirmed = (
            reset["state"] == "confirmed"
            and reset["boundary"] is not None
            and reset["baseline"] is not None
            and reset["clean_epoch"] is not None
            and bool(reset["advancing_race"])
            and bool(reset["advancing_imu"])
        )
        if reset_confirmed:
            generation = reset["boundary"]["new_generation"]
            stream_id = reset["baseline"]["race"]["ingress"]["stream_id"]
            final_observations = (final_heartbeat, final_race, final_imu)
            generation_consistent = all(
                observation["ingress"]["generation"] == generation
                for observation in final_observations
            )
            stream_consistent = all(
                observation["ingress"]["stream_id"] == stream_id
                for observation in final_observations
            )
            post_request = all(
                observation["ingress"]["received_monotonic_ns"]
                > reset["request_monotonic_ns"]
                and observation["ingress"]["received_monotonic_ns"]
                >= reset["boundary"]["boundary_monotonic_ns"]
                and observation["ingress"]["received_monotonic_ns"]
                >= reset["receipt"]["call_start_monotonic_ns"]
                for observation in final_observations
            )
            race_anchor = final_epoch["race_anchor_boot_ms"]
            imu_anchor = final_epoch["imu_anchor_usec"]
            final_race_value = final_race["race_status"]["sim_boot_time_ms"]
            final_imu_value = final_imu["imu"]["timestamp_us"]
            last_race = reset["advancing_race"][-1]
            last_imu = reset["advancing_imu"][-1]
            race_not_preceding = (
                final_race["ingress"]["sequence"] >= last_race["ingress"]["sequence"]
                and final_race["ingress"]["received_monotonic_ns"]
                >= last_race["ingress"]["received_monotonic_ns"]
                and final_race_value >= last_race["race_status"]["sim_boot_time_ms"]
            )
            imu_not_preceding = (
                final_imu["ingress"]["sequence"] >= last_imu["ingress"]["sequence"]
                and final_imu["ingress"]["received_monotonic_ns"]
                >= last_imu["ingress"]["received_monotonic_ns"]
                and final_imu_value >= last_imu["imu"]["timestamp_us"]
            )
            role_consistent = role != "cleanup_fallback" or (
                final_race == last_race and final_imu == last_imu
            )
            final_consistent = all(
                (
                    final_epoch == reset["clean_epoch"],
                    generation_consistent,
                    stream_consistent,
                    post_request,
                    heartbeat_disarmed,
                    final_race_value > race_anchor,
                    final_imu_value > imu_anchor,
                    race_not_preceding,
                    imu_not_preceding,
                    role_consistent,
                )
            )
        else:
            final_consistent = False
        if final_state == "confirmed" and not final_consistent:
            _fail(f"{path}.final_state", "confirmed state does not bind the complete reset generation and epoch")
        if final_state == "partial" and final_consistent:
            _fail(f"{path}.final_state.state", "must be confirmed when every final consistency check passes")

    occurrence_rows: list[tuple[str, Mapping[str, Any]]] = []

    def add_occurrence(name: str, observation: Any) -> None:
        if observation is not None:
            occurrence_rows.append((name, observation))

    add_occurrence("disarm.heartbeat_before", disarm["heartbeat_before"])
    add_occurrence("disarm.heartbeat_after", disarm["heartbeat_after"])
    if reset["baseline"] is not None:
        add_occurrence("reset.baseline.race", reset["baseline"]["race"])
        add_occurrence("reset.baseline.imu", reset["baseline"]["imu"])
    if reset["boundary"] is not None:
        for index, observation in enumerate(reset["boundary"]["observations"]):
            add_occurrence(f"reset.boundary.observations[{index}]", observation)
    for name in ("advancing_race", "advancing_imu"):
        for index, observation in enumerate(reset[name]):
            add_occurrence(f"reset.{name}[{index}]", observation)
    if final_state != "unobserved":
        add_occurrence("final_state.heartbeat", final["heartbeat"])
        add_occurrence("final_state.last_race", final["last_race"])
        add_occurrence("final_state.last_imu", final["last_imu"])

    occurrence_by_token: dict[tuple[str, int, int], tuple[str, Mapping[str, Any]]] = {}
    for name, observation in occurrence_rows:
        ingress = observation["ingress"]
        if ingress["received_monotonic_ns"] > completed:
            _fail(
                f"{path}.{name}.ingress.received_monotonic_ns",
                "must not follow certificate completion",
            )
        token = (ingress["stream_id"], ingress["generation"], ingress["sequence"])
        prior = occurrence_by_token.get(token)
        if prior is not None and prior[1] != observation:
            _fail(
                f"{path}.{name}",
                f"reuses ingress occurrence token with different evidence from {prior[0]}",
            )
        occurrence_by_token[token] = (name, observation)
    if reset["boundary"] is not None:
        old_generation = reset["boundary"]["old_generation"]
        ingress_stats = reset["boundary"]["ingress_stats"]
        old_occurrences = [
            observation
            for (stream_id, generation, _sequence), (_name, observation)
            in occurrence_by_token.items()
            if generation == old_generation
        ]
        if any(
            observation["ingress"]["sequence"] >= ingress_stats["next_sequence"]
            for observation in old_occurrences
        ):
            _fail(
                f"{path}.reset.boundary.ingress_stats.next_sequence",
                "must follow every represented old-generation ingress sequence",
            )
        counter_fields = {
            "HIGHRES_IMU": "highres_imu_received",
            "HEARTBEAT": "heartbeat_received",
            "RACE_STATUS": "race_status_received",
            "ACTUATOR_OUTPUT_STATUS": "actuator_received",
        }
        for message_type, field in counter_fields.items():
            represented = sum(
                observation["ingress"]["message_type"] == message_type
                for observation in old_occurrences
            )
            if represented > ingress_stats[field]:
                _fail(
                    f"{path}.reset.boundary.ingress_stats.{field}",
                    "cannot be smaller than represented old-generation observations",
                )
    occurrence_groups: dict[tuple[str, int], list[tuple[str, Mapping[str, Any]]]] = {}
    for (stream_id, generation, _sequence), named in occurrence_by_token.items():
        occurrence_groups.setdefault((stream_id, generation), []).append(named)
    for named_rows in occurrence_groups.values():
        ordered = sorted(named_rows, key=lambda item: item[1]["ingress"]["sequence"])
        prior_received = -1
        for name, observation in ordered:
            received = observation["ingress"]["received_monotonic_ns"]
            if received < prior_received:
                _fail(f"{path}.{name}.ingress.received_monotonic_ns", "must preserve host receipt order within its generation")
            prior_received = received
    transport = _object(row["transport"], {"production_guard_latched", "cleanup_guard_closed", "vision_closed", "mavlink_socket_closed", "receiver_joined", "announcer_joined", "owned_handles_closed"}, f"{path}.transport")
    for name in transport:
        _bool(transport[name], f"{path}.transport.{name}")
    outcome = _enum(row["outcome"], {"proved", "failed"}, f"{path}.outcome")
    failures = _sorted_unique_enum(row["failure_codes"], f"{path}.failure_codes", _CLEANUP_FAILURE_CODES, nonempty=outcome == "failed")
    if outcome == "proved" and failures:
        _fail(f"{path}.failure_codes", "must be empty for proved outcome")
    collection_codes = _sorted_unique_enum(row["collection_invalidating_codes"], f"{path}.collection_invalidating_codes", _COLLECTION_INVALIDATING_CODES)
    derived_collection_codes: set[str] = set()
    if collision_rows:
        derived_collection_codes.add("collision_observed")
    if endpoints["mavlink"]["rejected_source_count"] or (
        endpoints["camera"] is not None and endpoints["camera"]["rejected_source_count"]
    ):
        derived_collection_codes.add("source_rejected")
    if role == "powered_child" and endpoints["camera"]["state"] in {"not_opened", "closed_without_peer"}:
        derived_collection_codes.add("camera_missing")
    if not derived_collection_codes.issubset(collection_codes):
        _fail(f"{path}.collection_invalidating_codes", f"missing derived codes {sorted(derived_collection_codes - set(collection_codes))!r}")
    if outcome == "proved":
        if not lease["authority_valid"] or endpoints["mavlink"]["state"] != "closed_with_peer" or row["disarm"]["state"] != "confirmed" or row["reset"]["state"] != "confirmed" or final_state != "confirmed" or row["zero_command"]["state"] not in {"not_required", "returned"} or not all(transport.values()):
            _fail(path, "proved cleanup requires authority, closed peer, confirmed disarm/reset/final state, required zero, and closed transport")
    return defensive_copy(row)


def validate_process_result(value: Any, *, cleanup_certificate: Any | None = None) -> dict[str, Any]:
    path = "$process_result"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "producer_role", "process_authority_sha256", "started_monotonic_ns", "completed_monotonic_ns", "outcome", "reason_codes", "phase_deadlines", "cleanup_certificate", "outbound_audit", "artifacts"}, path)
    _literal(row["schema"], "aigp-vq2-powered-process-result/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    role = _enum(row["producer_role"], {"powered_child", "cleanup_fallback"}, f"{path}.producer_role")
    _sha256(row["process_authority_sha256"], f"{path}.process_authority_sha256")
    start = _int(row["started_monotonic_ns"], f"{path}.started_monotonic_ns")
    completed = _int(row["completed_monotonic_ns"], f"{path}.completed_monotonic_ns")
    if completed < start:
        _fail(f"{path}.completed_monotonic_ns", "must not precede process start")
    outcome = _enum(row["outcome"], {"completed", "failed"}, f"{path}.outcome")
    reasons = _sorted_unique_enum(row["reason_codes"], f"{path}.reason_codes", INVALIDATION_REASON_CODES, nonempty=outcome == "failed")
    if outcome == "completed" and reasons:
        _fail(f"{path}.reason_codes", "must be empty for completed outcome")
    phases = _validate_process_phase_deadlines(row["phase_deadlines"], f"{path}.phase_deadlines", role=role)
    cert_ref = _object(row["cleanup_certificate"], {"path", "state", "sha256"}, f"{path}.cleanup_certificate")
    validate_absolute_windows_path(cert_ref["path"], path=f"{path}.cleanup_certificate.path", root=EVIDENCE_ROOT)
    expected_certificate_path = _FROZEN_PATHS["child_cleanup_certificate"] if role == "powered_child" else _FROZEN_PATHS["fallback_cleanup_certificate"]
    if cert_ref["path"] != expected_certificate_path:
        _fail(f"{path}.cleanup_certificate.path", "must equal the frozen role certificate path")
    state = _enum(cert_ref["state"], {"absent", "published", "invalid"}, f"{path}.cleanup_certificate.state")
    if state == "absent":
        if cert_ref["sha256"] is not None:
            _fail(f"{path}.cleanup_certificate.sha256", "must be null when absent")
    else:
        _sha256(cert_ref["sha256"], f"{path}.cleanup_certificate.sha256")
    validate_outbound_audit(row["outbound_audit"], f"{path}.outbound_audit")
    artifacts = _object(row["artifacts"], {"legacy_record", "replay_bundle"}, f"{path}.artifacts")
    if role == "cleanup_fallback":
        if artifacts["legacy_record"] is not None or artifacts["replay_bundle"] is not None:
            _fail(f"{path}.artifacts", "fallback has no legacy record or replay bundle")
    else:
        legacy = _object(artifacts["legacy_record"], {"path", "state", "sha256"}, f"{path}.artifacts.legacy_record")
        validate_absolute_windows_path(legacy["path"], path=f"{path}.artifacts.legacy_record.path", root=EVIDENCE_ROOT)
        if legacy["path"] != _FROZEN_PATHS["legacy_record"]:
            _fail(f"{path}.artifacts.legacy_record.path", "must equal frozen legacy-record path")
        legacy_state = _enum(legacy["state"], {"absent", "partial", "closed"}, f"{path}.artifacts.legacy_record.state")
        if legacy_state == "closed":
            _sha256(legacy["sha256"], f"{path}.artifacts.legacy_record.sha256")
        elif legacy["sha256"] is not None:
            _fail(f"{path}.artifacts.legacy_record.sha256", "must be null unless closed")
        replay = _object(artifacts["replay_bundle"], {"path", "state", "dataset_hash", "manifest_sha256", "records_sha256"}, f"{path}.artifacts.replay_bundle")
        validate_absolute_windows_path(replay["path"], path=f"{path}.artifacts.replay_bundle.path", root=EVIDENCE_ROOT)
        if replay["path"] != _FROZEN_PATHS["replay_bundle"]:
            _fail(f"{path}.artifacts.replay_bundle.path", "must equal frozen replay-bundle path")
        replay_state = _enum(replay["state"], {"absent", "partial", "closed"}, f"{path}.artifacts.replay_bundle.state")
        hash_names = ("dataset_hash", "manifest_sha256", "records_sha256")
        if replay_state == "closed":
            for name in hash_names:
                _sha256(replay[name], f"{path}.artifacts.replay_bundle.{name}")
        elif any(replay[name] is not None for name in hash_names):
            _fail(f"{path}.artifacts.replay_bundle", "hashes must be null unless closed")
    if cleanup_certificate is not None:
        certificate = validate_cleanup_certificate(cleanup_certificate)
        if certificate["producer_role"] != role:
            _fail(f"{path}.producer_role", "does not match supplied cleanup certificate")
        if state == "absent" or cert_ref["sha256"] != canonical_file_sha256(certificate):
            _fail(f"{path}.cleanup_certificate", "does not bind supplied cleanup certificate")
        certificate_phases = certificate["phase_deadlines"]
        if phases[: len(certificate_phases)] != certificate_phases:
            _fail(f"{path}.phase_deadlines", "cleanup certificate phases must be an exact prefix")
        if outcome == "completed" and certificate["outcome"] != "proved":
            _fail(f"{path}.outcome", "completed process requires a proved cleanup certificate")
    if outcome == "completed":
        if state != "published":
            _fail(f"{path}.cleanup_certificate.state", "completed process requires a published certificate")
        if role == "powered_child" and (
            artifacts["legacy_record"]["state"] != "closed"
            or artifacts["replay_bundle"]["state"] != "closed"
        ):
            _fail(f"{path}.artifacts", "completed child requires closed legacy and replay artifacts")
    return defensive_copy(row)


_CHILD_EXIT_STATES = frozenset({"not_created", "proved", "unproved"})
_FALLBACK_STATES = frozenset({"not_eligible", "not_required", "proved", "failed", "unproved"})
_PORT_STATES = frozenset({"not_opened", "free", "owned", "unproved"})
_LEASE_STATES = frozenset({"not_acquired", "retained", "released", "unproved"})
_PROCESS_STATES = frozenset({"not_created", "exited", "residue", "unproved"})
_TRANSPORT_STATES = frozenset({"not_opened", "closed", "open", "unproved"})
_TASK_STATES = frozenset({"not_created", "absent", "present", "unproved"})
_TOPOLOGY_STATES = frozenset({"not_launched", "unchanged", "changed", "unproved"})
_RESPONSIVE_STATES = frozenset({"not_launched", "yes", "no", "unproved"})


def validate_invalid_cleanup_state(value: Any, path: str = "$cleanup_state") -> dict[str, Any]:
    row = _object(value, {"child_exit", "fallback", "ports", "lease", "processes", "transport", "scheduled_task", "simulator_topology", "simulator_responsive"}, path)
    _enum(row["child_exit"], _CHILD_EXIT_STATES, f"{path}.child_exit")
    _enum(row["fallback"], _FALLBACK_STATES, f"{path}.fallback")
    _enum(row["ports"], _PORT_STATES, f"{path}.ports")
    _enum(row["lease"], _LEASE_STATES, f"{path}.lease")
    _enum(row["processes"], _PROCESS_STATES, f"{path}.processes")
    _enum(row["transport"], _TRANSPORT_STATES, f"{path}.transport")
    _enum(row["scheduled_task"], _TASK_STATES, f"{path}.scheduled_task")
    _enum(row["simulator_topology"], _TOPOLOGY_STATES, f"{path}.simulator_topology")
    _enum(row["simulator_responsive"], _RESPONSIVE_STATES, f"{path}.simulator_responsive")
    return row


def validate_terminal_cleanup(value: Any, path: str = "$cleanup") -> dict[str, Any]:
    row = _object(value, {"child_certificate_sha256", "fallback_used", "fallback_certificate_sha256", "child_exit", "fallback", "processes", "transport", "ports", "lease", "simulator_topology", "simulator_responsive", "scheduled_task"}, path)
    _sha256(row["child_certificate_sha256"], f"{path}.child_certificate_sha256")
    fallback_used = _bool(row["fallback_used"], f"{path}.fallback_used")
    _sha256(row["fallback_certificate_sha256"], f"{path}.fallback_certificate_sha256", nullable=not fallback_used)
    if fallback_used and row["fallback_certificate_sha256"] is None:
        _fail(f"{path}.fallback_certificate_sha256", "must be present when fallback was used")
    if not fallback_used and row["fallback_certificate_sha256"] is not None:
        _fail(f"{path}.fallback_certificate_sha256", "must be null when fallback was not used")
    _enum(row["child_exit"], _CHILD_EXIT_STATES, f"{path}.child_exit")
    fallback = _enum(row["fallback"], _FALLBACK_STATES, f"{path}.fallback")
    _enum(row["processes"], _PROCESS_STATES, f"{path}.processes")
    _enum(row["transport"], _TRANSPORT_STATES, f"{path}.transport")
    _enum(row["ports"], _PORT_STATES, f"{path}.ports")
    _enum(row["lease"], _LEASE_STATES, f"{path}.lease")
    _enum(row["simulator_topology"], _TOPOLOGY_STATES, f"{path}.simulator_topology")
    _enum(row["simulator_responsive"], _RESPONSIVE_STATES, f"{path}.simulator_responsive")
    _enum(row["scheduled_task"], _TASK_STATES, f"{path}.scheduled_task")
    if fallback_used != (fallback in {"proved", "failed", "unproved"}):
        _fail(path, "fallback_used must agree with fallback state")
    return row


def _validate_simulator_launch(value: Any, path: str) -> dict[str, Any]:
    row = _object(value, {"disposition", "observed_before_launch_monotonic_ns", "launcher_return_monotonic_ns", "launcher_exit_code", "prelaunch_launcher_process", "prelaunch_payload_process"}, path)
    disposition = _enum(row["disposition"], {"absent_before_launcher_current_after", "preexisting_exact_topology"}, f"{path}.disposition")
    before = _int(row["observed_before_launch_monotonic_ns"], f"{path}.observed_before_launch_monotonic_ns")
    returned = _int(row["launcher_return_monotonic_ns"], f"{path}.launcher_return_monotonic_ns")
    if before >= returned:
        _fail(path, "before-launch occurrence must precede launcher return")
    _literal(row["launcher_exit_code"], 0, f"{path}.launcher_exit_code")
    if disposition == "absent_before_launcher_current_after":
        if row["prelaunch_launcher_process"] is not None or row["prelaunch_payload_process"] is not None:
            _fail(path, "absent-before launch requires null prelaunch processes")
    else:
        validate_process_identity(row["prelaunch_launcher_process"], f"{path}.prelaunch_launcher_process")
        validate_process_identity(row["prelaunch_payload_process"], f"{path}.prelaunch_payload_process")
    return row


def _validate_scheduled_task(value: Any, path: str, *, phase: str) -> dict[str, Any]:
    row = _object(value, {"name", "observations"}, path)
    _literal(row["name"], "AIGP-P2-F04-A01-Launch", f"{path}.name")
    observations = _array(row["observations"], f"{path}.observations")
    expected = ["before_launch", "after_launcher_return", "before_child"]
    if phase == "postchild":
        expected.append("after_child_or_fallback")
    if len(observations) != len(expected):
        _fail(f"{path}.observations", "must contain the exact phase observation set")
    prior = -1
    for index, expected_phase in enumerate(expected):
        item_path = f"{path}.observations[{index}]"
        item = _object(observations[index], {"phase", "observed_monotonic_ns", "query_exit_code", "absent"}, item_path)
        _literal(item["phase"], expected_phase, f"{item_path}.phase")
        observed = _int(item["observed_monotonic_ns"], f"{item_path}.observed_monotonic_ns")
        if observed <= prior:
            _fail(item_path, "task observations must be strictly ordered")
        prior = observed
        _int(item["query_exit_code"], f"{item_path}.query_exit_code", minimum=_INT64_MIN, maximum=_INT64_MAX)
        _literal(item["absent"], True, f"{item_path}.absent")
    return row


def _sorted_unique_ints(value: Any, path: str) -> list[int]:
    items = _array(value, path)
    checked = [_int(item, f"{path}[{index}]", maximum=_UINT32_MAX) for index, item in enumerate(items)]
    if checked != sorted(set(checked)) or len(checked) != len(set(checked)):
        _fail(path, "must be sorted unique integers")
    return checked


def _validate_process_proof_ports(value: Any, path: str, *, phase: str) -> dict[str, Any]:
    row = _object(value, {"owner_table_observations", "active_owner_observations", "exclusive_probes", "status"}, path)
    _literal(row["status"], "free", f"{path}.status")
    owners = _array(row["owner_table_observations"], f"{path}.owner_table_observations")
    expected_owner_count = 2 if phase == "prechild" else 3
    if len(owners) != expected_owner_count:
        _fail(f"{path}.owner_table_observations", f"must have exactly {expected_owner_count} observations")
    prior = -1
    for index, owner in enumerate(owners):
        item_path = f"{path}.owner_table_observations[{index}]"
        item = _object(owner, {"observed_monotonic_ns", "ipv4_14550", "ipv6_14550", "ipv4_5600", "ipv6_5600"}, item_path)
        observed = _int(item["observed_monotonic_ns"], f"{item_path}.observed_monotonic_ns")
        if observed <= prior:
            _fail(item_path, "owner observations must be strictly ordered")
        prior = observed
        for name in ("ipv4_14550", "ipv6_14550", "ipv4_5600", "ipv6_5600"):
            if _sorted_unique_ints(item[name], f"{item_path}.{name}"):
                _fail(f"{item_path}.{name}", "free-port proof requires empty owner arrays")
    active = _array(row["active_owner_observations"], f"{path}.active_owner_observations")
    if phase == "prechild" and active:
        _fail(f"{path}.active_owner_observations", "must be empty prechild")
    prior_active = -1
    for index, observation in enumerate(active):
        item_path = f"{path}.active_owner_observations[{index}]"
        item = _object(observation, {"observed_monotonic_ns", "port", "role", "pid", "creation_filetime_100ns"}, item_path)
        observed = _int(item["observed_monotonic_ns"], f"{item_path}.observed_monotonic_ns")
        if observed <= prior_active:
            _fail(item_path, "active observations must be strictly ordered")
        prior_active = observed
        _int(item["port"], f"{item_path}.port")
        if item["port"] not in {14550, 5600}:
            _fail(f"{item_path}.port", "must be 14550 or 5600")
        _enum(item["role"], {"powered_child", "cleanup_fallback"}, f"{item_path}.role")
        _int(item["pid"], f"{item_path}.pid", minimum=1)
        _int(item["creation_filetime_100ns"], f"{item_path}.creation_filetime_100ns", minimum=1)
    probes = _array(row["exclusive_probes"], f"{path}.exclusive_probes", length=2)
    expected = [("127.0.0.1", 14550), ("0.0.0.0", 5600)]
    for index, (host, port) in enumerate(expected):
        item_path = f"{path}.exclusive_probes[{index}]"
        item = _object(probes[index], {"host", "port", "started_monotonic_ns", "ended_monotonic_ns", "result"}, item_path)
        _literal(item["host"], host, f"{item_path}.host")
        _literal(item["port"], port, f"{item_path}.port")
        started = _int(item["started_monotonic_ns"], f"{item_path}.started_monotonic_ns")
        ended = _int(item["ended_monotonic_ns"], f"{item_path}.ended_monotonic_ns")
        if ended < started:
            _fail(item_path, "probe end must not precede start")
        _literal(item["result"], "bound_and_closed", f"{item_path}.result")
    return row


def validate_simulator_process_proof(value: Any) -> dict[str, Any]:
    path = "$simulator_process_proof"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "phase", "observed_at_utc", "observed_monotonic_ns", "host_clock_id", "wrapper_process", "launch", "launcher_process", "payload_process", "window", "build", "topology", "scheduled_task", "ports", "responsive"}, path)
    _literal(row["schema"], "aigp-vq2-simulator-process-proof/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    phase = _enum(row["phase"], {"prechild", "postchild"}, f"{path}.phase")
    _utc(row["observed_at_utc"], f"{path}.observed_at_utc")
    observed = _int(row["observed_monotonic_ns"], f"{path}.observed_monotonic_ns")
    _literal(row["host_clock_id"], HOST_CLOCK_ID, f"{path}.host_clock_id")
    validate_process_identity(row["wrapper_process"], f"{path}.wrapper_process")
    launch = _validate_simulator_launch(row["launch"], f"{path}.launch")
    if launch["launcher_return_monotonic_ns"] > observed:
        _fail(f"{path}.launch.launcher_return_monotonic_ns", "must not follow proof observation")
    launcher = validate_process_identity(row["launcher_process"], f"{path}.launcher_process")
    payload = validate_process_identity(row["payload_process"], f"{path}.payload_process")
    if launch["disposition"] == "preexisting_exact_topology" and (launch["prelaunch_launcher_process"] != launcher or launch["prelaunch_payload_process"] != payload):
        _fail(f"{path}.launch", "preexisting identities must match accepted current topology")
    window = _object(row["window"], {"hwnd", "owner_pid", "visible", "unminimized", "responsive"}, f"{path}.window")
    _int(window["hwnd"], f"{path}.window.hwnd", minimum=1)
    _int(window["owner_pid"], f"{path}.window.owner_pid", minimum=1)
    if window["owner_pid"] != payload["pid"]:
        _fail(f"{path}.window.owner_pid", "must equal payload PID")
    for name in ("visible", "unminimized", "responsive"):
        _literal(window[name], True, f"{path}.window.{name}")
    _literal(row["build"], 3385, f"{path}.build")
    _literal(row["topology"], "one_launcher_parent_retained_one_payload_child", f"{path}.topology")
    _validate_scheduled_task(row["scheduled_task"], f"{path}.scheduled_task", phase=phase)
    _validate_process_proof_ports(row["ports"], f"{path}.ports", phase=phase)
    _literal(row["responsive"], True, f"{path}.responsive")
    return defensive_copy(row)


def validate_training_attestation(value: Any, *, process_proof: Any | None = None) -> dict[str, Any]:
    path = "$training_attestation"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "attested_at_utc", "attested_monotonic_ns", "host_clock_id", "mode", "method", "challenge_sha256", "wrapper_process", "simulator_process_proof_sha256"}, path)
    _literal(row["schema"], "aigp-vq2-training-mode-attestation/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _utc(row["attested_at_utc"], f"{path}.attested_at_utc")
    _int(row["attested_monotonic_ns"], f"{path}.attested_monotonic_ns")
    _literal(row["host_clock_id"], HOST_CLOCK_ID, f"{path}.host_clock_id")
    _literal(row["mode"], "Training", f"{path}.mode")
    _literal(row["method"], "post_topology_visual_training_check_challenge", f"{path}.method")
    _sha256(row["challenge_sha256"], f"{path}.challenge_sha256")
    validate_process_identity(row["wrapper_process"], f"{path}.wrapper_process")
    _sha256(row["simulator_process_proof_sha256"], f"{path}.simulator_process_proof_sha256")
    if process_proof is not None:
        proof = validate_simulator_process_proof(process_proof)
        if proof["phase"] != "prechild" or row["simulator_process_proof_sha256"] != canonical_file_sha256(proof) or row["wrapper_process"] != proof["wrapper_process"] or row["attested_monotonic_ns"] < proof["observed_monotonic_ns"]:
            _fail(path, "does not bind the supplied prechild process proof")
    return defensive_copy(row)


_PUBLICATION_STATES = frozenset({"absent", "partial", "valid"})


def _state_hash_pair(row: dict[str, Any], state_key: str, hash_key: str, path: str, *, valid_state: str = "valid") -> None:
    state = row[state_key]
    if state == valid_state:
        _sha256(row[hash_key], f"{path}.{hash_key}")
    elif row[hash_key] is not None:
        _fail(f"{path}.{hash_key}", f"must be null unless {state_key}={valid_state!r}")


def _validate_invalid_artifact_state(value: Any, path: str) -> dict[str, Any]:
    keys = {
        "legacy_record", "legacy_record_sha256", "replay_bundle", "replay_dataset_hash",
        "replay_manifest_sha256", "replay_records_sha256", "bundle_verification",
        "bundle_verification_sha256", "capture_seal", "capture_seal_sha256",
        "split_claim", "split_claim_sha256", "split_registry", "split_registry_sha256",
        "analysis_report", "analysis_report_sha256", "wrapper_lifecycle",
        "wrapper_lifecycle_sha256", "attempt_complete", "attempt_complete_partial_sha256",
        "terminal_publication", "forensic_bytes_preserved",
    }
    row = _object(value, keys, path)
    legacy = _enum(row["legacy_record"], {"absent", "partial", "closed"}, f"{path}.legacy_record")
    if legacy == "closed":
        _sha256(row["legacy_record_sha256"], f"{path}.legacy_record_sha256")
    elif row["legacy_record_sha256"] is not None:
        _fail(f"{path}.legacy_record_sha256", "must be null unless legacy record is closed")
    replay = _enum(row["replay_bundle"], {"absent", "partial", "sealed"}, f"{path}.replay_bundle")
    replay_hashes = ("replay_dataset_hash", "replay_manifest_sha256", "replay_records_sha256")
    if replay == "sealed":
        for name in replay_hashes:
            _sha256(row[name], f"{path}.{name}")
    elif any(row[name] is not None for name in replay_hashes):
        _fail(path, "replay hashes must all be null unless bundle is sealed")
    pairs = (
        ("bundle_verification", "bundle_verification_sha256"),
        ("capture_seal", "capture_seal_sha256"),
        ("split_claim", "split_claim_sha256"),
        ("split_registry", "split_registry_sha256"),
        ("analysis_report", "analysis_report_sha256"),
        ("wrapper_lifecycle", "wrapper_lifecycle_sha256"),
    )
    for state_key, hash_key in pairs:
        _enum(row[state_key], _PUBLICATION_STATES, f"{path}.{state_key}")
        _state_hash_pair(row, state_key, hash_key, path)
    complete = _enum(row["attempt_complete"], {"absent", "partial"}, f"{path}.attempt_complete")
    if complete == "partial":
        _sha256(row["attempt_complete_partial_sha256"], f"{path}.attempt_complete_partial_sha256")
    elif row["attempt_complete_partial_sha256"] is not None:
        _fail(f"{path}.attempt_complete_partial_sha256", "must be null when attempt-complete is absent")
    _literal(row["terminal_publication"], "invalid_record", f"{path}.terminal_publication")
    _literal(row["forensic_bytes_preserved"], True, f"{path}.forensic_bytes_preserved")
    return row


def derive_poison_required(
    cleanup_state: Any,
    artifact_state: Any,
    reason_codes: Sequence[str],
    *,
    attempt_envelope_state: str,
) -> bool:
    cleanup = validate_invalid_cleanup_state(defensive_copy(cleanup_state))
    artifacts = _validate_invalid_artifact_state(defensive_copy(artifact_state), "$artifact_state")
    reasons = list(reason_codes)
    _sorted_unique_enum(reasons, "$reason_codes", INVALIDATION_REASON_CODES, nonempty=True)
    envelope_state = _enum(attempt_envelope_state, {"absent", "partial", "valid"}, "$attempt_envelope_state")
    safe_a = (
        cleanup["child_exit"] == "not_created"
        and cleanup["fallback"] == "not_eligible"
        and cleanup["ports"] in {"not_opened", "free"}
        and cleanup["lease"] in {"not_acquired", "released"}
        and cleanup["simulator_topology"] in {"not_launched", "unchanged"}
        and cleanup["processes"] in {"not_created", "exited"}
        and cleanup["transport"] in {"not_opened", "closed"}
        and cleanup["scheduled_task"] in {"not_created", "absent"}
        and cleanup["simulator_responsive"] in {"not_launched", "yes"}
    )
    safe_b = (
        cleanup == {
            "child_exit": "proved",
            "fallback": cleanup["fallback"],
            "ports": "free",
            "lease": "released",
            "processes": "exited",
            "transport": "closed",
            "scheduled_task": "absent",
            "simulator_topology": "unchanged",
            "simulator_responsive": "yes",
        }
        and cleanup["fallback"] in {"not_required", "proved"}
    )
    publications_safe = (
        artifacts["bundle_verification"] in {"absent", "valid"}
        and artifacts["capture_seal"] in {"absent", "valid"}
        and (
            all(artifacts[name] == "absent" for name in ("split_claim", "split_registry", "analysis_report"))
            or all(artifacts[name] == "valid" for name in ("split_claim", "split_registry", "analysis_report"))
        )
        and artifacts["attempt_complete"] == "absent"
    )
    lifecycle_safe = artifacts["wrapper_lifecycle"] == "valid"
    no_contact_exception = (
        safe_a
        and envelope_state in {"absent", "partial"}
        and artifacts["wrapper_lifecycle"] == "absent"
        and cleanup == {
            "child_exit": "not_created",
            "fallback": "not_eligible",
            "ports": "not_opened",
            "lease": "not_acquired",
            "processes": "not_created",
            "transport": "not_opened",
            "scheduled_task": "not_created",
            "simulator_topology": "not_launched",
            "simulator_responsive": "not_launched",
        }
    )
    safe = (safe_a or safe_b) and publications_safe and (lifecycle_safe or no_contact_exception) and "wrapper_death" not in reasons
    return not safe


def validate_attempt_invalid(value: Any) -> dict[str, Any]:
    path = "$attempt_invalid"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "invalidated_at_utc", "invalidated_monotonic_ns", "publication_timing", "phase", "reason_codes", "reason_detail", "identity", "artifact_state", "cleanup_state", "poison"}, path)
    _literal(row["schema"], "aigp-vq2-powered-calibration-attempt-invalid/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _utc(row["invalidated_at_utc"], f"{path}.invalidated_at_utc")
    invalidated = _int(row["invalidated_monotonic_ns"], f"{path}.invalidated_monotonic_ns")
    timing = validate_terminal_publication_timing(row["publication_timing"], f"{path}.publication_timing", expected_phase="invalid_terminal_publish")
    if timing["prepared_monotonic_ns"] != invalidated:
        _fail(f"{path}.publication_timing.prepared_monotonic_ns", "must equal invalidated_monotonic_ns")
    _token(row["phase"], f"{path}.phase")
    reasons = _sorted_unique_enum(row["reason_codes"], f"{path}.reason_codes", INVALIDATION_REASON_CODES, nonempty=True)
    _sanitized_text(row["reason_detail"], f"{path}.reason_detail", maximum_utf8_bytes=4096)
    identity = _object(row["identity"], {"attempt_envelope_state", "live_freeze_sha256", "attempt_context_sha256", "attempt_envelope_sha256", "candidate_commit", "target_config_sha256", "capture_authorization_sha256", "excitation_plan_sha256"}, f"{path}.identity")
    envelope_state = _enum(identity["attempt_envelope_state"], {"absent", "partial", "valid"}, f"{path}.identity.attempt_envelope_state")
    _sha256(identity["live_freeze_sha256"], f"{path}.identity.live_freeze_sha256")
    _commit(identity["candidate_commit"], f"{path}.identity.candidate_commit")
    for name in ("target_config_sha256", "capture_authorization_sha256", "excitation_plan_sha256"):
        _sha256(identity[name], f"{path}.identity.{name}")
    if identity["excitation_plan_sha256"] != EXCITATION_PLAN_SHA256:
        _fail(f"{path}.identity.excitation_plan_sha256", "must bind frozen excitation plan")
    if envelope_state == "valid":
        _sha256(identity["attempt_context_sha256"], f"{path}.identity.attempt_context_sha256")
        _sha256(identity["attempt_envelope_sha256"], f"{path}.identity.attempt_envelope_sha256")
    else:
        _sha256(identity["attempt_context_sha256"], f"{path}.identity.attempt_context_sha256", nullable=True)
        _sha256(identity["attempt_envelope_sha256"], f"{path}.identity.attempt_envelope_sha256", nullable=True)
    artifacts = _validate_invalid_artifact_state(row["artifact_state"], f"{path}.artifact_state")
    cleanup = validate_invalid_cleanup_state(row["cleanup_state"], f"{path}.cleanup_state")
    poison = _object(row["poison"], {"required", "path", "sha256"}, f"{path}.poison")
    required = _bool(poison["required"], f"{path}.poison.required")
    validate_absolute_windows_path(poison["path"], path=f"{path}.poison.path", root=EVIDENCE_ROOT)
    if poison["path"] != _FROZEN_PATHS["live_poison"]:
        _fail(f"{path}.poison.path", "must equal frozen poison path")
    if required:
        _sha256(poison["sha256"], f"{path}.poison.sha256", nullable=True)
    elif poison["sha256"] is not None:
        _fail(f"{path}.poison.sha256", "must be null when poison is not required")
    derived = derive_poison_required(cleanup, artifacts, reasons, attempt_envelope_state=envelope_state)
    if required != derived:
        _fail(f"{path}.poison.required", f"must equal derived poison predicate {derived}")
    return defensive_copy(row)


def validate_live_poison(value: Any) -> dict[str, Any]:
    path = "$live_poison"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "created_at_utc", "created_monotonic_ns", "publication_timing", "phase", "reason_codes", "attempt_context_sha256", "attempt_envelope_sha256", "wrapper_process", "child_process", "cleanup_process", "lease_state", "port_state", "process_state", "transport_state", "scheduled_task_state", "publication_state", "simulator_state", "required_action"}, path)
    _literal(row["schema"], "aigp-vq2-powered-calibration-live-poison/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _utc(row["created_at_utc"], f"{path}.created_at_utc")
    created = _int(row["created_monotonic_ns"], f"{path}.created_monotonic_ns")
    timing = validate_artifact_timing(row["publication_timing"], f"{path}.publication_timing", expected_phase="poison_publish")
    if timing["prepared_monotonic_ns"] != created:
        _fail(f"{path}.publication_timing.prepared_monotonic_ns", "must equal created_monotonic_ns")
    _token(row["phase"], f"{path}.phase")
    _sorted_unique_enum(row["reason_codes"], f"{path}.reason_codes", INVALIDATION_REASON_CODES, nonempty=True)
    _sha256(row["attempt_context_sha256"], f"{path}.attempt_context_sha256", nullable=True)
    _sha256(row["attempt_envelope_sha256"], f"{path}.attempt_envelope_sha256", nullable=True)
    for name in ("wrapper_process", "child_process", "cleanup_process"):
        if row[name] is not None:
            validate_process_identity(row[name], f"{path}.{name}")
    lease = _object(row["lease_state"], {"phase", "owner_token_sha256", "release_proved"}, f"{path}.lease_state")
    _token(lease["phase"], f"{path}.lease_state.phase")
    _sha256(lease["owner_token_sha256"], f"{path}.lease_state.owner_token_sha256", nullable=True)
    _bool(lease["release_proved"], f"{path}.lease_state.release_proved")
    ports = _object(row["port_state"], {"mavlink_14550", "camera_5600"}, f"{path}.port_state")
    for name in ports:
        _enum(ports[name], _PORT_STATES, f"{path}.port_state.{name}")
    _enum(row["process_state"], _PROCESS_STATES, f"{path}.process_state")
    _enum(row["transport_state"], _TRANSPORT_STATES, f"{path}.transport_state")
    _enum(row["scheduled_task_state"], _TASK_STATES, f"{path}.scheduled_task_state")
    publication = _object(row["publication_state"], {"bundle_verification", "capture_seal", "claim", "registry", "report", "wrapper_lifecycle", "attempt_complete", "terminal"}, f"{path}.publication_state")
    for name in ("bundle_verification", "capture_seal", "claim", "registry", "report", "wrapper_lifecycle"):
        _enum(publication[name], _PUBLICATION_STATES, f"{path}.publication_state.{name}")
    _enum(publication["attempt_complete"], {"absent", "partial"}, f"{path}.publication_state.attempt_complete")
    _enum(publication["terminal"], {"missing", "partial_complete"}, f"{path}.publication_state.terminal")
    simulator = _object(row["simulator_state"], {"topology", "responsive"}, f"{path}.simulator_state")
    _enum(simulator["topology"], _TOPOLOGY_STATES, f"{path}.simulator_state.topology")
    _enum(simulator["responsive"], _RESPONSIVE_STATES, f"{path}.simulator_state.responsive")
    _literal(row["required_action"], "new_reviewed_recovery_task_no_automatic_clear", f"{path}.required_action")
    return defensive_copy(row)


_BUNDLE_CHECKS = (
    "manifest_schema_valid", "records_schema_valid", "dataset_hash_valid",
    "records_complete", "frame_blob_set_exact", "frame_blob_hashes_valid",
    "decoded_frame_shape_valid", "camera_timing_links_exact",
    "observation_schemas_valid", "event_sequences_contiguous",
    "resource_stats_zero", "writer_closed",
)


def validate_bundle_verification(value: Any) -> dict[str, Any]:
    path = "$bundle_verification"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "verified_at_utc", "verified_monotonic_ns", "timing", "identity", "bundle", "checks", "valid"}, path)
    _literal(row["schema"], "aigp-vq2-replay-bundle-verification/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _utc(row["verified_at_utc"], f"{path}.verified_at_utc")
    verified = _int(row["verified_monotonic_ns"], f"{path}.verified_monotonic_ns")
    timing = validate_artifact_timing(row["timing"], f"{path}.timing", expected_phase="bundle_verify")
    if timing["prepared_monotonic_ns"] != verified:
        _fail(f"{path}.timing.prepared_monotonic_ns", "must equal verified_monotonic_ns")
    identity_keys = {"candidate_commit", "live_freeze_sha256", "attempt_context_sha256", "attempt_envelope_sha256", "child_authority_sha256", "child_process_result_sha256", "child_cleanup_certificate_sha256", "lease_final_sha256"}
    identity = _object(row["identity"], identity_keys, f"{path}.identity")
    _commit(identity["candidate_commit"], f"{path}.identity.candidate_commit")
    for name in identity_keys - {"candidate_commit"}:
        _sha256(identity[name], f"{path}.identity.{name}")
    bundle = _object(row["bundle"], {"path", "dataset_hash", "manifest", "records", "frames"}, f"{path}.bundle")
    validate_absolute_windows_path(bundle["path"], path=f"{path}.bundle.path", root=EVIDENCE_ROOT)
    if bundle["path"] != _FROZEN_PATHS["replay_bundle"]:
        _fail(f"{path}.bundle.path", "must equal frozen replay-bundle path")
    _sha256(bundle["dataset_hash"], f"{path}.bundle.dataset_hash")
    manifest = validate_artifact_ref(bundle["manifest"], f"{path}.bundle.manifest", root=EVIDENCE_ROOT)
    records = validate_artifact_ref(bundle["records"], f"{path}.bundle.records", root=EVIDENCE_ROOT)
    _literal(manifest["name"], "replay_manifest", f"{path}.bundle.manifest.name")
    _literal(records["name"], "replay_records", f"{path}.bundle.records.name")
    frames = _sorted_unique_objects(bundle["frames"], f"{path}.bundle.frames", key="name", validator=validate_artifact_ref)
    for index, frame in enumerate(frames):
        name_path = f"{path}.bundle.frames[{index}].name"
        prefix = "replay_frame/"
        if not frame["name"].startswith(prefix):
            _fail(name_path, "must embed decoded SHA-256")
        _sha256(frame["name"][len(prefix):], name_path)
    checks = _object(row["checks"], _BUNDLE_CHECKS, f"{path}.checks")
    for name in _BUNDLE_CHECKS:
        _literal(checks[name], True, f"{path}.checks.{name}")
    _literal(row["valid"], True, f"{path}.valid")
    return defensive_copy(row)


_CAPTURE_STATS = (
    "record_count", "decoded_frames", "frame_blobs", "camera_timing_records",
    "imu_records", "mavlink_ingress_records", "race_records", "heartbeat_records",
    "actuator_records", "collision_records", "generated_commands", "sent_commands",
    "not_sent_commands", "tick_dispositions", "capture_drops", "decoded_frame_drops",
    "writer_queue_drops", "writer_errors", "ingress_drops", "observation_queue_drops",
    "collision_queue_drops", "outbound_trace_drops", "queue_overflows",
)
_CAPTURE_LOSS_STATS = frozenset(
    {"capture_drops", "decoded_frame_drops", "writer_queue_drops", "writer_errors", "ingress_drops", "observation_queue_drops", "collision_queue_drops", "outbound_trace_drops", "queue_overflows"}
)


def validate_capture_seal(value: Any) -> dict[str, Any]:
    path = "$capture_seal"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "sealed_at_utc", "timing", "identity", "artifacts", "capture_stats", "outbound_audit", "cleanup"}, path)
    _literal(row["schema"], "aigp-vq2-powered-calibration-capture-seal/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _utc(row["sealed_at_utc"], f"{path}.sealed_at_utc")
    validate_artifact_timing(row["timing"], f"{path}.timing", expected_phase="capture_seal")
    identity_keys = {"candidate_commit", "code_sha256", "live_freeze_sha256", "attempt_context_sha256", "attempt_envelope_sha256", "target_config_sha256", "capture_authorization_sha256", "excitation_plan_id", "excitation_plan_sha256", "training_attestation_sha256", "simulator_process_proof_sha256", "simulator_final_process_proof_sha256", "child_authority_sha256", "cleanup_authority_sha256", "lease_final_sha256", "bundle_verification_sha256"}
    identity = _object(row["identity"], identity_keys, f"{path}.identity")
    _commit(identity["candidate_commit"], f"{path}.identity.candidate_commit")
    _literal(identity["excitation_plan_id"], EXCITATION_PLAN_ID, f"{path}.identity.excitation_plan_id")
    for name in identity_keys - {"candidate_commit", "excitation_plan_id", "cleanup_authority_sha256"}:
        _sha256(identity[name], f"{path}.identity.{name}")
    _sha256(identity["cleanup_authority_sha256"], f"{path}.identity.cleanup_authority_sha256", nullable=True)
    if identity["excitation_plan_sha256"] != EXCITATION_PLAN_SHA256:
        _fail(f"{path}.identity.excitation_plan_sha256", "must bind frozen plan")
    artifacts = _sorted_unique_objects(row["artifacts"], f"{path}.artifacts", key="name", validator=validate_artifact_ref)
    names = [item["name"] for item in artifacts]
    base_required = {
        "live_freeze", "implementation_inventory", "environment_inventory", "import_inventory",
        "attempt_envelope", "training_attestation", "process_prechild", "process_postchild",
        "child_authority", "child_cleanup_certificate", "lease_final", "bundle_verification",
        "runner_stdout", "runner_stderr", "legacy_record", "replay_manifest", "replay_records",
    }
    if not base_required.issubset(names):
        _fail(f"{path}.artifacts", f"missing required names {sorted(base_required - set(names))!r}")
    for name in names:
        if name not in base_required and name not in {"cleanup_authority", "fallback_cleanup_certificate", "cleanup_stdout", "cleanup_stderr"} and not name.startswith("replay_frame/"):
            _fail(f"{path}.artifacts", f"unsupported artifact name {name!r}")
    stats = _object(row["capture_stats"], _CAPTURE_STATS, f"{path}.capture_stats")
    for name in _CAPTURE_STATS:
        _int(stats[name], f"{path}.capture_stats.{name}")
        if name in _CAPTURE_LOSS_STATS and stats[name] != 0:
            _fail(f"{path}.capture_stats.{name}", "sealed capture requires zero loss/error")
    audit = validate_outbound_audit(row["outbound_audit"], f"{path}.outbound_audit")
    for name in ("receipt_raised", "receipt_dropped", "receipt_buffered", "position_target", "other_command"):
        if audit[name] != 0:
            _fail(f"{path}.outbound_audit.{name}", "sealed capture requires zero disallowed/failed receipts")
    cleanup = validate_terminal_cleanup(row["cleanup"], f"{path}.cleanup")
    conditional = {"cleanup_authority", "fallback_cleanup_certificate", "cleanup_stdout", "cleanup_stderr"}
    if cleanup["fallback_used"]:
        if not conditional.issubset(names) or identity["cleanup_authority_sha256"] is None:
            _fail(path, "fallback seal must include all fallback identities/artifacts")
    elif conditional & set(names) or identity["cleanup_authority_sha256"] is not None:
        _fail(path, "no-fallback seal must exclude all fallback identities/artifacts")
    return defensive_copy(row)


_REPORT_CHECKS = (
    "identity_bound", "build3385_training_attested", "bundle_complete",
    "frame_hashes_valid", "decoded_dimensions_640x360_stable",
    "camera_lineage_complete", "imu_lineage_complete",
    "race_heartbeat_actuator_collision_lineage_complete", "capture_loss_zero",
    "ingress_loss_zero", "outbound_allowlist_exact", "command_pairs_exact",
    "ticks_0_through_244_accounted", "plan_exact", "watchdogs_passed",
    "cleanup_confirmed", "fallback_not_used", "child_process_tree_exited",
    "ports_released", "lease_released", "simulator_topology_unchanged",
    "simulator_responsive", "scheduled_task_absent", "exclusive_binds_and_peers_exact",
    "collection_invalidating_codes_empty", "conditional_on_nominal_gate_config",
    "no_fit_or_rank_inspection",
)
_REPORT_COUNTS = (
    "decoded_frames", "unique_decoded_hashes", "camera_timing_records", "imu_records",
    "mavlink_ingress_records", "race_records", "heartbeat_records", "actuator_records",
    "collision_records", "generated_commands", "sent_commands", "not_sent_commands",
    "ticks_sent", "ticks_skipped_before_generation", "ticks_skipped_after_generation",
    "capture_drops", "decoded_frame_drops", "writer_errors", "ingress_drops",
    "queue_overflows", "send_failed_or_uncertain",
)
_REPORT_IDENTITY_KEYS = {
    "candidate_commit", "live_freeze_sha256", "attempt_context_sha256",
    "attempt_envelope_sha256", "target_config_sha256", "capture_authorization_sha256",
    "excitation_plan_id", "excitation_plan_sha256", "training_attestation_sha256",
    "simulator_process_proof_sha256", "simulator_final_process_proof_sha256",
    "child_authority_sha256", "cleanup_authority_sha256", "lease_final_sha256",
    "bundle_verification_sha256",
}


def validate_acquisition_report(value: Any) -> dict[str, Any]:
    path = "$acquisition_report"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "generated_at_utc", "timing", "collection_valid", "invalid_reasons", "reference_scope", "identity", "input_artifacts", "checks", "counts", "command_accounting", "excitation_accounting", "descriptive_support", "calibration_status", "unmeasured", "split"}, path)
    _literal(row["schema"], "aigp-vq2-powered-calibration-acquisition-report/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _utc(row["generated_at_utc"], f"{path}.generated_at_utc")
    validate_artifact_timing(row["timing"], f"{path}.timing", expected_phase="split_publish")
    _literal(row["collection_valid"], True, f"{path}.collection_valid")
    _array(row["invalid_reasons"], f"{path}.invalid_reasons", length=0)
    scope = _object(row["reference_scope"], {"conditional_on_nominal_gate_config", "geometry_status", "target_config_sha256"}, f"{path}.reference_scope")
    _literal(scope["conditional_on_nominal_gate_config"], True, f"{path}.reference_scope.conditional_on_nominal_gate_config")
    _literal(scope["geometry_status"], "nominal_unverified_for_build_3385_training", f"{path}.reference_scope.geometry_status")
    _sha256(scope["target_config_sha256"], f"{path}.reference_scope.target_config_sha256")
    identity = _object(row["identity"], _REPORT_IDENTITY_KEYS, f"{path}.identity")
    _commit(identity["candidate_commit"], f"{path}.identity.candidate_commit")
    _literal(identity["excitation_plan_id"], EXCITATION_PLAN_ID, f"{path}.identity.excitation_plan_id")
    for name in _REPORT_IDENTITY_KEYS - {"candidate_commit", "excitation_plan_id", "cleanup_authority_sha256"}:
        _sha256(identity[name], f"{path}.identity.{name}")
    _sha256(identity["cleanup_authority_sha256"], f"{path}.identity.cleanup_authority_sha256", nullable=True)
    if identity["target_config_sha256"] != scope["target_config_sha256"] or identity["excitation_plan_sha256"] != EXCITATION_PLAN_SHA256:
        _fail(f"{path}.identity", "must agree with nominal config scope and frozen plan")
    input_keys = {"capture_seal_sha256", "bundle_dataset_hash", "bundle_verification_sha256", "bundle_manifest_sha256", "bundle_records_sha256", "legacy_record_sha256", "lease_final_sha256", "runner_stdout_sha256", "runner_stderr_sha256", "child_cleanup_certificate_sha256", "fallback_cleanup_certificate_sha256"}
    inputs = _object(row["input_artifacts"], input_keys, f"{path}.input_artifacts")
    for name in input_keys - {"fallback_cleanup_certificate_sha256"}:
        _sha256(inputs[name], f"{path}.input_artifacts.{name}")
    _sha256(inputs["fallback_cleanup_certificate_sha256"], f"{path}.input_artifacts.fallback_cleanup_certificate_sha256", nullable=True)
    checks = _object(row["checks"], _REPORT_CHECKS, f"{path}.checks")
    for name in _REPORT_CHECKS:
        _literal(checks[name], True, f"{path}.checks.{name}")
    if identity["cleanup_authority_sha256"] is not None or inputs["fallback_cleanup_certificate_sha256"] is not None:
        _fail(path, "success report's fallback_not_used proof requires null fallback hashes")
    counts = _object(row["counts"], _REPORT_COUNTS, f"{path}.counts")
    for name in _REPORT_COUNTS:
        _int(counts[name], f"{path}.counts.{name}")
    for name in ("capture_drops", "decoded_frame_drops", "writer_errors", "ingress_drops", "queue_overflows", "send_failed_or_uncertain"):
        _literal(counts[name], 0, f"{path}.counts.{name}")
    if counts["ticks_sent"] + counts["ticks_skipped_before_generation"] + counts["ticks_skipped_after_generation"] != 245:
        _fail(f"{path}.counts", "tick disposition counts must total 245")
    accounting_keys = {"attitude_target_audit_delta", "generated_count", "sent_count", "not_sent_count", "unmatched_generation_count", "unmatched_sent_count", "failed_or_uncertain_count", "envelope_violation_count", "payload_mismatch_count", "all_reconciled"}
    accounting = _object(row["command_accounting"], accounting_keys, f"{path}.command_accounting")
    for name in accounting_keys - {"all_reconciled"}:
        _int(accounting[name], f"{path}.command_accounting.{name}")
    _literal(accounting["all_reconciled"], True, f"{path}.command_accounting.all_reconciled")
    for name in ("unmatched_generation_count", "unmatched_sent_count", "failed_or_uncertain_count", "envelope_violation_count", "payload_mismatch_count"):
        _literal(accounting[name], 0, f"{path}.command_accounting.{name}")
    if accounting["generated_count"] != accounting["sent_count"] + accounting["not_sent_count"] or accounting["attitude_target_audit_delta"] != accounting["sent_count"]:
        _fail(f"{path}.command_accounting", "generated/sent/audit arithmetic does not reconcile")
    excitation = _object(row["excitation_accounting"], {"plan_id", "plan_sha256", "tick_count", "segments", "first_release_monotonic_ns", "last_slot_end_monotonic_ns", "powered_expiry_monotonic_ns"}, f"{path}.excitation_accounting")
    _literal(excitation["plan_id"], EXCITATION_PLAN_ID, f"{path}.excitation_accounting.plan_id")
    _literal(excitation["plan_sha256"], EXCITATION_PLAN_SHA256, f"{path}.excitation_accounting.plan_sha256")
    _literal(excitation["tick_count"], 245, f"{path}.excitation_accounting.tick_count")
    segment_rows = _array(excitation["segments"], f"{path}.excitation_accounting.segments", length=len(_PLAN_LITERAL["segments"]))
    for index, planned in enumerate(_PLAN_LITERAL["segments"]):
        segment_path = f"{path}.excitation_accounting.segments[{index}]"
        segment = _object(segment_rows[index], {"segment_id", "planned_ticks", "generated", "sent", "skipped"}, segment_path)
        _literal(segment["segment_id"], planned["segment_id"], f"{segment_path}.segment_id")
        planned_ticks = planned["last_tick"] - planned["first_tick"] + 1
        _literal(segment["planned_ticks"], planned_ticks, f"{segment_path}.planned_ticks")
        for name in ("generated", "sent", "skipped"):
            _int(segment[name], f"{segment_path}.{name}")
        if segment["sent"] + segment["skipped"] != planned_ticks or segment["generated"] < segment["sent"]:
            _fail(segment_path, "segment accounting does not reconcile")
    first = _int(excitation["first_release_monotonic_ns"], f"{path}.excitation_accounting.first_release_monotonic_ns")
    last = _int(excitation["last_slot_end_monotonic_ns"], f"{path}.excitation_accounting.last_slot_end_monotonic_ns")
    expiry = _int(excitation["powered_expiry_monotonic_ns"], f"{path}.excitation_accounting.powered_expiry_monotonic_ns")
    if last != first + 4_900_000_000 or expiry != first + 5_000_000_000:
        _fail(f"{path}.excitation_accounting", "timing must equal the frozen plan envelope")
    support_keys = {"target_observation_count", "target_center_x_px_min", "target_center_x_px_max", "target_center_y_px_min", "target_center_y_px_max", "target_bbox_area_px_min", "target_bbox_area_px_max", "gyro_x_rad_s_min", "gyro_x_rad_s_max", "gyro_y_rad_s_min", "gyro_y_rad_s_max", "gyro_z_rad_s_min", "gyro_z_rad_s_max", "roll_reversal_count", "pitch_reversal_count", "semantics"}
    support = _object(row["descriptive_support"], support_keys, f"{path}.descriptive_support")
    _int(support["target_observation_count"], f"{path}.descriptive_support.target_observation_count", minimum=1)
    for name in ("roll_reversal_count", "pitch_reversal_count"):
        _int(support[name], f"{path}.descriptive_support.{name}")
    for name in support_keys - {"target_observation_count", "roll_reversal_count", "pitch_reversal_count", "semantics"}:
        _number(support[name], f"{path}.descriptive_support.{name}")
    for prefix in ("target_center_x_px", "target_center_y_px", "target_bbox_area_px", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"):
        if support[f"{prefix}_min"] > support[f"{prefix}_max"]:
            _fail(f"{path}.descriptive_support", f"{prefix} minimum exceeds maximum")
    _literal(support["semantics"], "descriptive_only_no_acceptance_threshold", f"{path}.descriptive_support.semantics")
    calibration_keys = {"intrinsics", "distortion", "camera_to_body_rotation", "camera_imu_time_model", "rank", "covariance", "empirical_limits"}
    calibration = _object(row["calibration_status"], calibration_keys, f"{path}.calibration_status")
    for name in calibration_keys:
        _literal(calibration[name], "uncomputed", f"{path}.calibration_status.{name}")
    expected_unmeasured = ["absolute_host_phase", "accepted_calibration_coefficients", "command_to_actuator_response", "empirical_limits", "encode_queue_component_delays", "package2_acceptance", "render_exposure_delay"]
    if _sorted_unique_strings(row["unmeasured"], f"{path}.unmeasured") != expected_unmeasured:
        _fail(f"{path}.unmeasured", "must equal the frozen unmeasured set")
    split = _object(row["split"], {"assigned_split", "claim_path", "claim_sha256", "registry_path", "registry_sha256", "activation"}, f"{path}.split")
    _literal(split["assigned_split"], "discovery_fit", f"{path}.split.assigned_split")
    for name, expected_path in (("claim_path", _FROZEN_PATHS["split_claim"]), ("registry_path", _FROZEN_PATHS["split_registry"])):
        validate_absolute_windows_path(split[name], path=f"{path}.split.{name}", root=EVIDENCE_ROOT)
        if split[name] != expected_path:
            _fail(f"{path}.split.{name}", "must equal frozen path")
    _sha256(split["claim_sha256"], f"{path}.split.claim_sha256")
    _sha256(split["registry_sha256"], f"{path}.split.registry_sha256")
    _literal(split["activation"], "requires_matching_attempt_complete", f"{path}.split.activation")
    return defensive_copy(row)


def _validate_decoded_hash_array(value: Any, path: str) -> list[str]:
    items = _sorted_unique_strings(value, path)
    for index, item in enumerate(items):
        _sha256(item, f"{path}[{index}]")
    return items


def validate_split_claim(value: Any) -> dict[str, Any]:
    path = "$split_claim"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "claimed_at_utc", "claimed_monotonic_ns", "timing", "run_id", "assigned_split", "identity", "reset_epochs", "run_artifacts", "decoded_content_sha256", "derivative_sha256", "collision_policy"}, path)
    _literal(row["schema"], "aigp-vq2-package2-run-split-claim/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _utc(row["claimed_at_utc"], f"{path}.claimed_at_utc")
    claimed = _int(row["claimed_monotonic_ns"], f"{path}.claimed_monotonic_ns")
    timing = validate_artifact_timing(row["timing"], f"{path}.timing", expected_phase="split_publish")
    if timing["prepared_monotonic_ns"] != claimed:
        _fail(f"{path}.timing.prepared_monotonic_ns", "must equal claimed_monotonic_ns")
    _literal(row["run_id"], "F04-A01/reset-epoch-1/excitation-1", f"{path}.run_id")
    _literal(row["assigned_split"], "discovery_fit", f"{path}.assigned_split")
    identity = _object(row["identity"], {"attempt_context_sha256", "attempt_envelope_sha256", "capture_seal_sha256", "excitation_plan_id", "excitation_plan_sha256"}, f"{path}.identity")
    for name in ("attempt_context_sha256", "attempt_envelope_sha256", "capture_seal_sha256"):
        _sha256(identity[name], f"{path}.identity.{name}")
    _literal(identity["excitation_plan_id"], EXCITATION_PLAN_ID, f"{path}.identity.excitation_plan_id")
    _literal(identity["excitation_plan_sha256"], EXCITATION_PLAN_SHA256, f"{path}.identity.excitation_plan_sha256")
    epochs = _array(row["reset_epochs"], f"{path}.reset_epochs", length=1)
    _validate_reset_epoch(epochs[0], f"{path}.reset_epochs[0]")
    artifacts = _sorted_unique_objects(row["run_artifacts"], f"{path}.run_artifacts", key="name", validator=validate_artifact_ref)
    names = [item["name"] for item in artifacts]
    required = {"bundle_verification", "child_cleanup_certificate", "legacy_record", "replay_manifest", "replay_records", "runner_stdout", "runner_stderr"}
    if not required.issubset(names):
        _fail(f"{path}.run_artifacts", f"missing required names {sorted(required - set(names))!r}")
    if any(name not in required and not name.startswith("replay_frame/") for name in names):
        _fail(f"{path}.run_artifacts", "contains an unsupported or fallback artifact")
    decoded = _validate_decoded_hash_array(row["decoded_content_sha256"], f"{path}.decoded_content_sha256")
    if not decoded:
        _fail(f"{path}.decoded_content_sha256", "must not be empty")
    frame_hashes = sorted(
        [name.removeprefix("replay_frame/") for name in names if name.startswith("replay_frame/")],
        key=lambda item: item.encode("utf-8"),
    )
    if frame_hashes != decoded:
        _fail(f"{path}.run_artifacts", "replay-frame names must exactly cover decoded_content_sha256")
    _array(row["derivative_sha256"], f"{path}.derivative_sha256", length=0)
    _literal(row["collision_policy"], "f04_fixed_future_whole_run_discovery_fit_or_global_exclusion", f"{path}.collision_policy")
    return defensive_copy(row)


def validate_split_registry(value: Any, *, split_claim: Any | None = None) -> dict[str, Any]:
    path = "$split_registry"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "published_at_utc", "published_monotonic_ns", "timing", "registry_id", "revision", "previous_registry_sha256", "claims", "content_groups"}, path)
    _literal(row["schema"], "aigp-vq2-package2-split-registry/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _utc(row["published_at_utc"], f"{path}.published_at_utc")
    published = _int(row["published_monotonic_ns"], f"{path}.published_monotonic_ns")
    timing = validate_artifact_timing(row["timing"], f"{path}.timing", expected_phase="split_publish")
    if timing["prepared_monotonic_ns"] != published:
        _fail(f"{path}.timing.prepared_monotonic_ns", "must equal published_monotonic_ns")
    _literal(row["registry_id"], "vq2-package2-calibration", f"{path}.registry_id")
    _literal(row["revision"], 1, f"{path}.revision")
    if row["previous_registry_sha256"] is not None:
        _fail(f"{path}.previous_registry_sha256", "initial registry predecessor must be null")
    claims = _array(row["claims"], f"{path}.claims", length=1)
    claim = _object(claims[0], {"claim_path", "claim_sha256", "session_id", "attempt_id", "run_id", "assigned_split", "activation"}, f"{path}.claims[0]")
    validate_absolute_windows_path(claim["claim_path"], path=f"{path}.claims[0].claim_path", root=EVIDENCE_ROOT)
    _literal(claim["claim_path"], _FROZEN_PATHS["split_claim"], f"{path}.claims[0].claim_path")
    _sha256(claim["claim_sha256"], f"{path}.claims[0].claim_sha256")
    _literal(claim["session_id"], SESSION_ID, f"{path}.claims[0].session_id")
    _literal(claim["attempt_id"], ATTEMPT_ID, f"{path}.claims[0].attempt_id")
    _literal(claim["run_id"], "F04-A01/reset-epoch-1/excitation-1", f"{path}.claims[0].run_id")
    _literal(claim["assigned_split"], "discovery_fit", f"{path}.claims[0].assigned_split")
    _literal(claim["activation"], "requires_matching_attempt_complete", f"{path}.claims[0].activation")
    groups = _array(row["content_groups"], f"{path}.content_groups")
    hashes: list[str] = []
    for index, group in enumerate(groups):
        item_path = f"{path}.content_groups[{index}]"
        item = _object(group, {"decoded_sha256", "run_ids", "assigned_split", "disposition", "activation"}, item_path)
        hashes.append(_sha256(item["decoded_sha256"], f"{item_path}.decoded_sha256"))
        run_ids = _sorted_unique_strings(item["run_ids"], f"{item_path}.run_ids")
        if run_ids != ["F04-A01/reset-epoch-1/excitation-1"]:
            _fail(f"{item_path}.run_ids", "initial registry group must name the sole run")
        _literal(item["assigned_split"], "discovery_fit", f"{item_path}.assigned_split")
        _literal(item["disposition"], "assigned", f"{item_path}.disposition")
        _literal(item["activation"], "requires_matching_attempt_complete", f"{item_path}.activation")
    if hashes != sorted(set(hashes), key=lambda item: item.encode("utf-8")):
        _fail(f"{path}.content_groups", "must be unique and sorted by decoded_sha256")
    if split_claim is not None:
        source_claim = validate_split_claim(split_claim)
        if claim["claim_sha256"] != canonical_file_sha256(source_claim):
            _fail(f"{path}.claims[0].claim_sha256", "does not bind supplied claim file")
        if hashes != source_claim["decoded_content_sha256"]:
            _fail(f"{path}.content_groups", "must exactly cover claim decoded content")
    return defensive_copy(row)


_COMPLETE_IDENTITY_KEYS = {
    "candidate_commit", "code_sha256", "live_freeze_sha256", "attempt_context_sha256",
    "attempt_envelope_sha256", "target_config_sha256", "capture_authorization_sha256",
    "excitation_plan_id", "excitation_plan_sha256", "wrapper_lifecycle_sha256",
}
_COMPLETE_ARTIFACT_HASH_KEYS = {
    "bundle_dataset_hash", "bundle_verification_sha256", "capture_seal_sha256",
    "analysis_report_sha256", "split_claim_sha256", "split_registry_sha256",
    "bundle_manifest_sha256", "bundle_records_sha256", "legacy_record_sha256",
    "runner_stdout_sha256", "runner_stderr_sha256", "lease_final_sha256",
    "training_attestation_sha256", "simulator_process_proof_sha256",
    "simulator_final_process_proof_sha256", "implementation_inventory_sha256",
    "environment_inventory_sha256", "import_inventory_sha256", "child_authority_sha256",
    "cleanup_authority_sha256", "child_cleanup_certificate_sha256",
    "fallback_cleanup_certificate_sha256", "cleanup_stdout_sha256",
    "cleanup_stderr_sha256", "wrapper_lifecycle_sha256",
}


def validate_attempt_complete(value: Any, *, wrapper_lifecycle: Any | None = None) -> dict[str, Any]:
    path = "$attempt_complete"
    row = _object(value, {"schema", "task_id", "session_id", "attempt_id", "completed_at_utc", "completed_monotonic_ns", "deadline_monotonic_ns", "publication_timing", "identity", "artifact_hashes", "cleanup"}, path)
    _literal(row["schema"], "aigp-vq2-powered-calibration-attempt-complete/1", f"{path}.schema")
    _literal(row["task_id"], TASK_ID, f"{path}.task_id")
    _literal(row["session_id"], SESSION_ID, f"{path}.session_id")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    _utc(row["completed_at_utc"], f"{path}.completed_at_utc")
    completed = _int(row["completed_monotonic_ns"], f"{path}.completed_monotonic_ns")
    deadline = _int(row["deadline_monotonic_ns"], f"{path}.deadline_monotonic_ns")
    if completed >= deadline:
        _fail(f"{path}.completed_monotonic_ns", "must be before frozen terminal deadline")
    timing = validate_terminal_publication_timing(row["publication_timing"], f"{path}.publication_timing", expected_phase="terminal_publish")
    if timing["prepared_monotonic_ns"] != completed or timing["deadline_monotonic_ns"] != deadline:
        _fail(f"{path}.publication_timing", "must bind completion time and terminal deadline")
    identity = _object(row["identity"], _COMPLETE_IDENTITY_KEYS, f"{path}.identity")
    _commit(identity["candidate_commit"], f"{path}.identity.candidate_commit")
    _literal(identity["excitation_plan_id"], EXCITATION_PLAN_ID, f"{path}.identity.excitation_plan_id")
    for name in _COMPLETE_IDENTITY_KEYS - {"candidate_commit", "excitation_plan_id"}:
        _sha256(identity[name], f"{path}.identity.{name}")
    _literal(identity["excitation_plan_sha256"], EXCITATION_PLAN_SHA256, f"{path}.identity.excitation_plan_sha256")
    hashes = _object(row["artifact_hashes"], _COMPLETE_ARTIFACT_HASH_KEYS, f"{path}.artifact_hashes")
    nullable = {"cleanup_authority_sha256", "fallback_cleanup_certificate_sha256", "cleanup_stdout_sha256", "cleanup_stderr_sha256"}
    for name in _COMPLETE_ARTIFACT_HASH_KEYS - nullable:
        _sha256(hashes[name], f"{path}.artifact_hashes.{name}")
    for name in nullable:
        if hashes[name] is not None:
            _fail(f"{path}.artifact_hashes.{name}", "must be null on the sole successful no-fallback branch")
    if hashes["wrapper_lifecycle_sha256"] != identity["wrapper_lifecycle_sha256"]:
        _fail(f"{path}.artifact_hashes.wrapper_lifecycle_sha256", "must equal identity lifecycle hash")
    cleanup = validate_terminal_cleanup(row["cleanup"], f"{path}.cleanup")
    expected_cleanup = {
        "fallback_used": False, "child_exit": "proved", "fallback": "not_required",
        "processes": "exited", "transport": "closed", "ports": "free",
        "lease": "released", "simulator_topology": "unchanged",
        "simulator_responsive": "yes", "scheduled_task": "absent",
    }
    for name, expected in expected_cleanup.items():
        if cleanup[name] != expected:
            _fail(f"{path}.cleanup.{name}", f"must equal {expected!r} for completion")
    if cleanup["fallback_certificate_sha256"] is not None:
        _fail(f"{path}.cleanup.fallback_certificate_sha256", "must be null for completion")
    if cleanup["child_certificate_sha256"] != hashes["child_cleanup_certificate_sha256"]:
        _fail(f"{path}.cleanup.child_certificate_sha256", "must equal child cleanup artifact hash")
    if wrapper_lifecycle is not None:
        lifecycle = validate_wrapper_lifecycle(wrapper_lifecycle)
        if identity["wrapper_lifecycle_sha256"] != canonical_file_sha256(lifecycle):
            _fail(f"{path}.identity.wrapper_lifecycle_sha256", "does not bind supplied lifecycle")
    return defensive_copy(row)


def validate_phase_deadline_event(value: Any, path: str = "$phase_deadline_event") -> dict[str, Any]:
    row = _object(value, {"schema", "attempt_id", "producer_role", "phase", "event_sequence", "started_monotonic_ns", "duration_ns", "parent_deadline_monotonic_ns", "deadline_monotonic_ns"}, path)
    _literal(row["schema"], "aigp-vq2-phase-deadline/1", f"{path}.schema")
    _literal(row["attempt_id"], ATTEMPT_ID, f"{path}.attempt_id")
    role = _enum(row["producer_role"], {"powered_child", "cleanup_fallback"}, f"{path}.producer_role")
    _int(row["event_sequence"], f"{path}.event_sequence")
    deadline = {name: row[name] for name in ("phase", "started_monotonic_ns", "duration_ns", "parent_deadline_monotonic_ns", "deadline_monotonic_ns")}
    validate_phase_deadline(deadline, path)
    phase = row["phase"]
    allowed = set(_CHILD_PHASES[:-1]) | {"parent_death_lease_takeover"} if role == "powered_child" else set(_FALLBACK_PHASES) | {"parent_death_lease_takeover"}
    if phase not in allowed:
        _fail(f"{path}.phase", "not admitted for this producer/event channel")
    if row["duration_ns"] != _expected_process_phase_duration(role, phase):
        _fail(f"{path}.duration_ns", "must equal frozen role phase duration")
    return defensive_copy(row)


_SCHEMA_VALIDATORS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "aigp-vq2-calibration-excitation-plan/1": validate_excitation_plan,
    "aigp-vq2-powered-implementation-inventory/1": validate_implementation_inventory,
    "aigp-vq2-powered-environment-inventory/1": validate_environment_inventory,
    "aigp-vq2-powered-import-inventory/1": validate_import_inventory,
    "aigp-vq2-powered-calibration-live-freeze/1": validate_live_freeze,
    "aigp-vq2-powered-calibration-attempt/1": validate_attempt,
    "aigp-vq2-powered-process-authority/1": validate_process_authority,
    "aigp-vq2-received-heartbeat/1": validate_received_heartbeat,
    "aigp-vq2-received-race-status/1": validate_received_race_status,
    "aigp-vq2-received-actuator-output-status/1": validate_received_actuator_output_status,
    "aigp-vq2-received-imu/1": validate_received_imu,
    "aigp-vq2-decoded-dimensions-admission/1": validate_decoded_dimensions_admission,
    "aigp-vq2-runner-collision-observation/1": validate_collision_observation,
    "aigp-vq2-calibration-reset-boundary/1": validate_reset_boundary,
    "aigp-vq2-attitude-target-outbound/1": validate_attitude_target_outbound,
    "aigp-vq2-nonattitude-outbound/1": validate_nonattitude_outbound,
    "aigp-vq2-calibration-command-generated/1": validate_command_generated,
    "aigp-vq2-calibration-command-sent/1": validate_command_sent,
    "aigp-vq2-calibration-command-not-sent/1": validate_command_not_sent,
    "aigp-vq2-calibration-tick-disposition/1": validate_tick_disposition,
    "aigp-vq2-powered-wrapper-event/1": validate_wrapper_event,
    "aigp-vq2-powered-wrapper-lifecycle/1": validate_wrapper_lifecycle,
    "aigp-vq2-powered-cleanup-certificate/1": validate_cleanup_certificate,
    "aigp-vq2-powered-process-result/1": validate_process_result,
    "aigp-vq2-simulator-process-proof/1": validate_simulator_process_proof,
    "aigp-vq2-training-mode-attestation/1": validate_training_attestation,
    "aigp-vq2-powered-calibration-attempt-invalid/1": validate_attempt_invalid,
    "aigp-vq2-powered-calibration-live-poison/1": validate_live_poison,
    "aigp-vq2-replay-bundle-verification/1": validate_bundle_verification,
    "aigp-vq2-powered-calibration-capture-seal/1": validate_capture_seal,
    "aigp-vq2-powered-calibration-acquisition-report/1": validate_acquisition_report,
    "aigp-vq2-package2-run-split-claim/1": validate_split_claim,
    "aigp-vq2-package2-split-registry/1": validate_split_registry,
    "aigp-vq2-powered-calibration-attempt-complete/1": validate_attempt_complete,
    "aigp-vq2-phase-deadline/1": validate_phase_deadline_event,
}


def validate_powered_record(value: Any, *, expected_schema: str | None = None) -> dict[str, Any]:
    """Dispatch a strict top-level record validator by its exact schema."""

    if type(value) is not dict:
        _fail("$record", "must be an exact object")
    schema = _string(value.get("schema"), "$record.schema")
    if expected_schema is not None and schema != expected_schema:
        _fail("$record.schema", f"must equal expected schema {expected_schema!r}")
    validator = _SCHEMA_VALIDATORS.get(schema)
    if validator is None:
        _fail("$record.schema", f"unsupported powered schema {schema!r}")
    return validator(value)


def parse_and_validate_powered_record(
    payload: bytes | bytearray | memoryview,
    *,
    expected_schema: str | None = None,
    file_form: bool = True,
) -> dict[str, Any]:
    value = parse_canonical_json_bytes(payload, file_form=file_form)
    return validate_powered_record(value, expected_schema=expected_schema)


# Descriptive aliases used by the runner, cleanup helper, and offline analyzer.
validate_calibration_command_generated = validate_command_generated
validate_calibration_command_sent = validate_command_sent
validate_calibration_command_not_sent = validate_command_not_sent
validate_calibration_tick_disposition = validate_tick_disposition
validate_powered_cleanup_certificate = validate_cleanup_certificate
validate_powered_process_result = validate_process_result
validate_powered_attempt_complete = validate_attempt_complete
validate_powered_attempt_invalid = validate_attempt_invalid
validate_powered_live_poison = validate_live_poison
validate_replay_bundle_verification = validate_bundle_verification
validate_powered_capture_seal = validate_capture_seal
validate_powered_acquisition_report = validate_acquisition_report


__all__ = [
    "ATTEMPT_ID",
    "DEADLINE_DURATIONS_NS",
    "EVIDENCE_ROOT",
    "EXCITATION_PLAN_ID",
    "EXCITATION_PLAN_SCHEMA",
    "EXCITATION_PLAN_SHA256",
    "FROZEN_EXCITATION_PLAN",
    "HOST_CLOCK_ID",
    "IMPORT_INVENTORY_SEEDS",
    "INVALIDATION_REASON_CODES",
    "PoweredAttemptContractError",
    "RUNTIME_IMPORT_MODULES",
    "SESSION_ID",
    "TASK_ID",
    "WRAPPER_PHASES",
    "attempt_file_sha256",
    "canonical_file_sha256",
    "canonical_json_bytes",
    "canonical_json_file_bytes",
    "canonical_object_sha256",
    "decode_capability_frame",
    "defensive_copy",
    "derive_capability_sha256",
    "derive_excitation_plan",
    "derive_poison_required",
    "encode_capability_frame",
    "environment_variables_sha256",
    "excitation_command_for_tick",
    "excitation_tick",
    "frozen_excitation_plan",
    "frozen_paths",
    "iter_excitation_ticks",
    "parse_and_validate_powered_record",
    "parse_canonical_json_bytes",
    "sha256_bytes",
    "strict_json_loads",
    "validate_absolute_windows_path",
    "validate_acquisition_report",
    "validate_artifact_ref",
    "validate_artifact_timing",
    "validate_attempt",
    "validate_attempt_complete",
    "validate_attempt_invalid",
    "validate_attitude_target_outbound",
    "validate_bundle_verification",
    "validate_calibration_command_generated",
    "validate_calibration_command_not_sent",
    "validate_calibration_command_sent",
    "validate_calibration_tick_disposition",
    "validate_capture_seal",
    "validate_cleanup_certificate",
    "validate_collision_observation",
    "validate_deadline_durations",
    "validate_decoded_dimensions_admission",
    "validate_excitation_plan",
    "validate_environment_inventory",
    "validate_frozen_paths",
    "validate_identity_ref",
    "validate_implementation_inventory",
    "validate_import_inventory",
    "validate_invalid_cleanup_state",
    "validate_live_freeze",
    "validate_live_poison",
    "validate_nonattitude_outbound",
    "validate_phase_deadline",
    "validate_phase_deadline_event",
    "validate_powered_acquisition_report",
    "validate_powered_attempt_complete",
    "validate_powered_attempt_invalid",
    "validate_powered_capture_seal",
    "validate_powered_cleanup_certificate",
    "validate_powered_live_poison",
    "validate_powered_process_result",
    "validate_powered_record",
    "validate_process_authority",
    "validate_process_identity",
    "validate_process_result",
    "validate_received_actuator_output_status",
    "validate_received_heartbeat",
    "validate_received_imu",
    "validate_received_race_status",
    "validate_replay_bundle_verification",
    "validate_reset_boundary",
    "validate_simulator_process_proof",
    "validate_split_claim",
    "validate_split_registry",
    "validate_terminal_cleanup",
    "validate_terminal_publication_timing",
    "validate_tick_disposition",
    "validate_training_attestation",
    "validate_wrapper_event",
    "validate_wrapper_lifecycle",
]
