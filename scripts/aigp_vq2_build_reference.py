"""Strict offline build-reference discovery and structural validation.

The production commands in this module are gated by a separately reviewed
rules-clearance record.  They never launch or contact FlightSim and can never
admit a calibration reference: every report fixes ``admitted`` to ``False``.
R1 tests exercise the parser only with synthetic PAK fixtures.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import stat
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence


RULES_SCHEMA = "aigp-vq2-build-reference-rules-clearance/1"
CANDIDATE_SCHEMA = "aigp-vq2-build-reference-candidate/1"
DISCOVERY_SCHEMA = "aigp-vq2-build-reference-discovery/1"
VALIDATION_SCHEMA = "aigp-vq2-build-reference-validation/1"
OBSERVATION_SCHEMA = "aigp-vq2-target-reference-observation/1"
PARSER_IMPLEMENTATION = "aigp-vq2-stdlib-pak-parser/1"

REFERENCE_ROOT = Path(
    r"C:\Users\John\aigp-evidence"
    r"\2026-07-20-package2-powered-calibration-pilot\reference"
)
LAUNCHER_PATH = Path(r"C:\Users\John\AIGP\AIGP_3385\FlightSim.exe")
PAYLOAD_PATH = Path(
    r"C:\Users\John\AIGP\AIGP_3385\FlightSim\Binaries\Win64"
    r"\DCGame-Win64-Shipping.exe"
)
PAK_PATH = Path(
    r"C:\Users\John\AIGP\AIGP_3385\FlightSim\Content\Paks"
    r"\FlightSim-WindowsNoEditor.pak"
)
EXPECTED_SHA256 = {
    "launcher": "0d3217fa72e9fee847b2c154432476a687f21b79f0ab6b910728a6254b4dce32",
    "payload": "9064dd1547a30afea1e3fb87652cc8194c3f5af556be40629dc491bb4f681362",
    "pak": "dae7ed0f4d51f7755814bf069cc9299b439ff874a2f77912a0c5678afaff299f",
}
INTERPRETER_SHA256 = (
    "9b0bffb7a259cd2722df454fdfff41ee13665820cff1f578b1d97d31f9ef93d5"
)

PAK_MAGIC = 0x5A6F12E1
PAK_VERSION = 11
PAK_V11_FOOTER_BYTES = 221
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_INDEX_BYTES = 64 * 1024 * 1024
MAX_STRING_BYTES = 1024 * 1024
MAX_ENTRIES = 2_000_000
MAX_SERIALIZED_I64 = 2**63 - 1
TRANSFORM_RELATIVE_ROUNDOFF = 1e-6
TRANSFORM_ORTHOGONALITY_ROUNDOFF = 1e-6
ALLOWED_COMPRESSION = frozenset({"Zlib", "Gzip", "Oodle", "Zstd", "LZ4"})

FROZEN_PACKAGE_STEMS = (
    "FlightSim/Content/Anduril-TrackEditor/Gates/BP_gate",
    "FlightSim/Content/Anduril-TrackEditor/Gates/"
    "SM_Gates_Anduril_Square_Combined",
    "FlightSim/Content/levels/MAP_arsenal_track01",
    "FlightSim/Content/levelsMaster/MAP_arsenal_master",
)
MISSING_R3_CLAIMS = (
    "active_render_mesh_and_lod",
    "visible_feature_geometry_and_uncertainty",
    "complete_uniform_similarity_transform_chain",
    "active_training_map_and_udp_view_linkage",
    "visibility_model",
    "render_and_annotation_systematics",
    "independent_extraction_check_1",
    "independent_extraction_check_2",
    "rules_scope_and_competition_use",
    "immutable_observation_annotation_and_shared_nuisance_contract",
)


class ReferenceToolError(RuntimeError):
    """A fail-closed reference-tool input or structural error."""


def _reject_constant(value: str) -> None:
    raise ReferenceToolError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceToolError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _walk_finite(value: Any) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ReferenceToolError("non-finite JSON number is forbidden")
    if type(value) is list:
        for item in value:
            _walk_finite(item)
    elif type(value) is dict:
        for item in value.values():
            _walk_finite(item)


def strict_json_bytes(payload: bytes) -> Any:
    if len(payload) > MAX_JSON_BYTES:
        raise ReferenceToolError("JSON input exceeds the resource limit")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ReferenceToolError("UTF-8 BOM is forbidden")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ReferenceToolError("JSON input is not strict UTF-8") from exc
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (ValueError, RecursionError) as exc:
        raise ReferenceToolError("JSON input is malformed") from exc
    try:
        _walk_finite(value)
    except RecursionError as exc:
        raise ReferenceToolError("JSON input is too deeply nested") from exc
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        _walk_finite(value)
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ReferenceToolError("value cannot be encoded as canonical JSON") from exc
    return text.encode("utf-8") + b"\n"


def envelope_bytes(schema: str, payload: Mapping[str, Any]) -> bytes:
    payload_bytes = canonical_json_bytes(dict(payload))
    envelope = {
        "schema": schema,
        "payload": dict(payload),
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }
    return canonical_json_bytes(envelope)


def parser_config_sha256() -> str:
    config = {
        "allowed_compression": sorted(ALLOWED_COMPRESSION),
        "footer_bytes": PAK_V11_FOOTER_BYTES,
        "frozen_package_stems": list(FROZEN_PACKAGE_STEMS),
        "magic": PAK_MAGIC,
        "maximum_entries": MAX_ENTRIES,
        "maximum_index_bytes": MAX_INDEX_BYTES,
        "maximum_serialized_i64": MAX_SERIALIZED_I64,
        "maximum_string_bytes": MAX_STRING_BYTES,
        "transform_orthogonality_roundoff": TRANSFORM_ORTHOGONALITY_ROUNDOFF,
        "transform_relative_roundoff": TRANSFORM_RELATIVE_ROUNDOFF,
        "version": PAK_VERSION,
    }
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def _signature(info: os.stat_result) -> tuple[Any, ...]:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return tuple(getattr(info, field, None) for field in fields)


def _hash_handle(handle: BinaryIO) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _stable_identity(
    path: Path,
    *,
    label: str,
    inspect: Any = None,
    expected_hash: str | None = None,
) -> tuple[dict[str, Any], Any]:
    if os.name == "nt":
        target = _absolute_lexical(path, label=label)
        locked = _windows_lock_file(target)
    else:
        target = _secure_input(path, label=label)
        locked = []
    flags = os.O_RDONLY
    for optional in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, optional, 0))
    try:
        descriptor = os.open(target, flags)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                opened = os.fstat(descriptor)
                named = os.stat(target, follow_symlinks=False)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or not os.path.samestat(opened, named)
                ):
                    raise ReferenceToolError(f"{label} changed while opening")
                first_hash = _hash_handle(handle)
                if expected_hash is not None and first_hash != expected_hash:
                    raise ReferenceToolError(f"{label} hash mismatch")
                handle.seek(0)
                inspected = (
                    inspect(handle, opened.st_size)
                    if inspect is not None
                    else None
                )
                second_hash = _hash_handle(handle)
                after = os.fstat(descriptor)
                named_after = os.stat(target, follow_symlinks=False)
                if os.name == "nt":
                    _windows_validate_locks(locked)
                if (
                    first_hash != second_hash
                    or _signature(opened) != _signature(after)
                    or not os.path.samestat(opened, named_after)
                    or _signature(named) != _signature(named_after)
                ):
                    raise ReferenceToolError(f"{label} changed while being read")
                return (
                    {
                        "path": str(target),
                        "size_bytes": opened.st_size,
                        "sha256": first_hash,
                    },
                    inspected,
                )
        finally:
            os.close(descriptor)
    finally:
        if os.name == "nt":
            _windows_close_locks(locked)


def _is_reparse(info: os.stat_result) -> bool:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & reparse
    )


def _windows_drive_type(root: str) -> int:
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    kernel.GetDriveTypeW.restype = wintypes.UINT
    return int(kernel.GetDriveTypeW(root))


def _absolute_lexical(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ReferenceToolError(f"{label} path must be absolute")
    lexical = Path(os.path.abspath(path))
    if os.name == "nt":
        raw = str(path)
        if raw.startswith(("\\\\?\\", "\\\\.\\")):
            raise ReferenceToolError(f"{label} uses a Windows device path")
        if raw.startswith("\\\\"):
            raise ReferenceToolError(f"{label} uses a network path")
        if _windows_drive_type(lexical.anchor) != 3:
            raise ReferenceToolError(f"{label} is not on a fixed local drive")
        reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "CLOCK$",
            "CONIN$",
            "CONOUT$",
        }
        reserved.update(f"COM{index}" for index in range(1, 10))
        reserved.update(f"LPT{index}" for index in range(1, 10))
        reserved.update(f"COM{index}" for index in "¹²³")
        reserved.update(f"LPT{index}" for index in "¹²³")
        for component in lexical.parts[1:]:
            stem = component.split(".", 1)[0].rstrip(" .").upper()
            if (
                ":" in component
                or component.endswith((" ", "."))
                or stem in reserved
            ):
                raise ReferenceToolError(f"{label} uses a Windows path alias")
    return lexical


def _secure_existing(path: Path, *, directory: bool, label: str) -> Path:
    lexical = _absolute_lexical(path, label=label)
    probe = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        probe = probe / part
        try:
            info = probe.lstat()
        except OSError as exc:
            raise ReferenceToolError(f"{label} path is missing: {lexical}") from exc
        if _is_reparse(info):
            raise ReferenceToolError(f"{label} traverses a symlink/reparse point")
    info = probe.lstat()
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise ReferenceToolError(f"{label} is not a {kind}")
    return probe.resolve(strict=True)


def _secure_input(path: Path, *, label: str) -> Path:
    return _secure_existing(path, directory=False, label=label)


def _windows_private_acl_subject(
    *, path: Path | None = None, handle: Any = None
) -> bool:
    """Allow only the current user and Windows administrative principals."""

    if os.name != "nt":
        raise ReferenceToolError("Windows ACL check called on a non-Windows host")
    from ctypes import wintypes

    class SidAndAttributes(ctypes.Structure):
        _fields_ = [("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD)]

    class TokenUser(ctypes.Structure):
        _fields_ = [("user", SidAndAttributes)]

    class AclSizeInformation(ctypes.Structure):
        _fields_ = [
            ("ace_count", wintypes.DWORD),
            ("acl_bytes_in_use", wintypes.DWORD),
            ("acl_bytes_free", wintypes.DWORD),
        ]

    class AceHeader(ctypes.Structure):
        _fields_ = [
            ("ace_type", ctypes.c_ubyte),
            ("ace_flags", ctypes.c_ubyte),
            ("ace_size", wintypes.WORD),
        ]

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi.OpenProcessToken.restype = wintypes.BOOL
    advapi.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_uint,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetTokenInformation.restype = wintypes.BOOL
    advapi.CreateWellKnownSid.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.CreateWellKnownSid.restype = wintypes.BOOL
    advapi.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi.GetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi.GetSecurityInfo.restype = wintypes.DWORD
    advapi.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    advapi.EqualSid.restype = wintypes.BOOL
    advapi.GetAclInformation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_int,
    ]
    advapi.GetAclInformation.restype = wintypes.BOOL
    advapi.GetAce.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi.GetAce.restype = wintypes.BOOL
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    token = wintypes.HANDLE()
    token_query = 0x0008
    if not advapi.OpenProcessToken(
        kernel.GetCurrentProcess(), token_query, ctypes.byref(token)
    ):
        raise ReferenceToolError("OpenProcessToken failed")
    security_descriptor = ctypes.c_void_p()
    try:
        required = wintypes.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(required))
        if not required.value:
            raise ReferenceToolError("GetTokenInformation size failed")
        token_buffer = ctypes.create_string_buffer(required.value)
        if not advapi.GetTokenInformation(
            token,
            1,
            token_buffer,
            required,
            ctypes.byref(required),
        ):
            raise ReferenceToolError("GetTokenInformation failed")
        current_sid = ctypes.cast(
            token_buffer, ctypes.POINTER(TokenUser)
        ).contents.user.sid

        def well_known(kind: int) -> ctypes.Array[Any]:
            size = wintypes.DWORD(68)
            buffer = ctypes.create_string_buffer(size.value)
            if not advapi.CreateWellKnownSid(
                kind, None, buffer, ctypes.byref(size)
            ):
                raise ReferenceToolError("CreateWellKnownSid failed")
            return buffer

        system_sid = well_known(22)
        administrators_sid = well_known(26)
        owner = ctypes.c_void_p()
        dacl = ctypes.c_void_p()
        security_information = 0x00000001 | 0x00000004
        if handle is None:
            if path is None:
                raise ReferenceToolError("ACL path or handle is required")
            result = advapi.GetNamedSecurityInfoW(
                str(path),
                1,
                security_information,
                ctypes.byref(owner),
                None,
                ctypes.byref(dacl),
                None,
                ctypes.byref(security_descriptor),
            )
        else:
            result = advapi.GetSecurityInfo(
                handle,
                1,
                security_information,
                ctypes.byref(owner),
                None,
                ctypes.byref(dacl),
                None,
                ctypes.byref(security_descriptor),
            )
        if result != 0 or not owner.value or not dacl.value:
            raise ReferenceToolError("Windows security information query failed")
        if not advapi.EqualSid(owner, current_sid):
            return False
        info = AclSizeInformation()
        if not advapi.GetAclInformation(
            dacl,
            ctypes.byref(info),
            ctypes.sizeof(info),
            2,
        ):
            raise ReferenceToolError("GetAclInformation failed")
        trusted = (
            current_sid,
            ctypes.addressof(system_sid),
            ctypes.addressof(administrators_sid),
        )
        for index in range(info.ace_count):
            ace = ctypes.c_void_p()
            if not advapi.GetAce(dacl, index, ctypes.byref(ace)):
                raise ReferenceToolError("GetAce failed")
            header = ctypes.cast(ace, ctypes.POINTER(AceHeader)).contents
            if header.ace_type in {4, 5, 9, 11}:
                return False
            if header.ace_type != 0:
                continue
            if header.ace_size < 12:
                raise ReferenceToolError("access-control entry is malformed")
            mask = ctypes.c_uint32.from_address(ace.value + 4).value
            if mask == 0:
                continue
            sid = ctypes.c_void_p(ace.value + 8)
            if not any(advapi.EqualSid(sid, allowed) for allowed in trusted):
                return False
        return True
    finally:
        if security_descriptor.value:
            kernel.LocalFree(security_descriptor)
        kernel.CloseHandle(token)


def _windows_private_acl(path: Path) -> bool:
    return _windows_private_acl_subject(path=path)


def _windows_private_acl_handle(handle: Any) -> bool:
    return _windows_private_acl_subject(handle=handle)


class _WindowsFileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class _WindowsFileInformation(ctypes.Structure):
    _fields_ = [
        ("attributes", ctypes.c_uint32),
        ("creation_time", _WindowsFileTime),
        ("access_time", _WindowsFileTime),
        ("write_time", _WindowsFileTime),
        ("volume_serial", ctypes.c_uint32),
        ("size_high", ctypes.c_uint32),
        ("size_low", ctypes.c_uint32),
        ("link_count", ctypes.c_uint32),
        ("file_index_high", ctypes.c_uint32),
        ("file_index_low", ctypes.c_uint32),
    ]


class _WindowsDispositionInformation(ctypes.Structure):
    _fields_ = [("delete_file", ctypes.c_ubyte)]


def _windows_file_api() -> Any:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.GetFileInformationByHandle.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsFileInformation),
    ]
    kernel.GetFileInformationByHandle.restype = ctypes.c_int
    kernel.GetFileType.argtypes = [ctypes.c_void_p]
    kernel.GetFileType.restype = ctypes.c_uint32
    kernel.GetFinalPathNameByHandleW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    kernel.GetFinalPathNameByHandleW.restype = ctypes.c_uint32
    kernel.WriteFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    kernel.WriteFile.restype = ctypes.c_int
    kernel.FlushFileBuffers.argtypes = [ctypes.c_void_p]
    kernel.FlushFileBuffers.restype = ctypes.c_int
    kernel.SetFileInformationByHandle.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel.SetFileInformationByHandle.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    return kernel


def _windows_final_path(handle: Any) -> Path:
    kernel = _windows_file_api()
    required = kernel.GetFinalPathNameByHandleW(handle, None, 0, 0)
    if not required:
        raise ReferenceToolError("GetFinalPathNameByHandleW size failed")
    buffer = ctypes.create_unicode_buffer(required + 1)
    length = kernel.GetFinalPathNameByHandleW(
        handle, buffer, len(buffer), 0
    )
    if not length or length >= len(buffer):
        raise ReferenceToolError("GetFinalPathNameByHandleW failed")
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        raise ReferenceToolError("Windows handle resolved to a network path")
    if value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(os.path.abspath(value))


def _windows_file_identity(handle: Any, *, directory: bool) -> tuple[int, ...]:
    kernel = _windows_file_api()
    if kernel.GetFileType(handle) != 1:
        raise ReferenceToolError("Windows handle is not a disk file")
    info = _WindowsFileInformation()
    if not kernel.GetFileInformationByHandle(handle, ctypes.byref(info)):
        raise ReferenceToolError("GetFileInformationByHandle failed")
    is_directory = bool(info.attributes & 0x10)
    is_reparse = bool(info.attributes & 0x400)
    if is_directory != directory or is_reparse:
        kind = "directory" if directory else "regular non-reparse file"
        raise ReferenceToolError(f"Windows handle is not a {kind}")
    return (
        info.volume_serial,
        info.file_index_high,
        info.file_index_low,
    )


def _windows_open_locked(
    path: Path,
    *,
    directory: bool,
    read_control: bool = True,
    share_write: bool = True,
    share_delete: bool = False,
) -> tuple[Any, tuple[int, ...]]:
    kernel = _windows_file_api()
    read_attributes = 0x00000080
    read_control_access = 0x00020000
    share_read_write = 0x00000001
    if share_write:
        share_read_write |= 0x00000002
    if share_delete:
        share_read_write |= 0x00000004
    open_existing = 3
    backup_semantics = 0x02000000
    open_reparse_point = 0x00200000
    handle = kernel.CreateFileW(
        str(path),
        read_attributes | (read_control_access if read_control else 0),
        share_read_write,
        None,
        open_existing,
        backup_semantics | open_reparse_point,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        error = ctypes.get_last_error()
        raise ReferenceToolError(
            f"CreateFileW locked open failed ({error}): {path}"
        )
    try:
        identity = _windows_file_identity(handle, directory=directory)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    return handle, identity


def _windows_lock_components(
    path: Path,
    *,
    final_share_write: bool = True,
) -> list[tuple[Any, tuple[int, ...], Path, bool]]:
    lexical = _absolute_lexical(path, label="locked directory")
    prefixes = [Path(lexical.anchor)]
    for part in lexical.parts[1:]:
        prefixes.append(prefixes[-1] / part)
    locked: list[tuple[Any, tuple[int, ...], Path, bool]] = []
    try:
        for index, prefix in enumerate(prefixes):
            share_write = final_share_write and index == len(prefixes) - 1
            handle, identity = _windows_open_locked(
                prefix, directory=True, share_write=share_write
            )
            locked.append((handle, identity, prefix, True))
            if _windows_final_path(handle) != Path(os.path.abspath(prefix)):
                raise ReferenceToolError(
                    f"locked directory final path mismatch: {prefix}"
                )
    except BaseException:
        _windows_close_locks(locked)
        raise
    return locked


def _windows_lock_file(
    path: Path,
) -> list[tuple[Any, tuple[int, ...], Path, bool]]:
    locked = _windows_lock_components(path.parent, final_share_write=False)
    try:
        handle, identity = _windows_open_locked(
            path, directory=False, read_control=False
        )
        locked.append((handle, identity, path, False))
        if _windows_final_path(handle) != Path(os.path.abspath(path)):
            raise ReferenceToolError(f"locked file final path mismatch: {path}")
    except BaseException:
        _windows_close_locks(locked)
        raise
    return locked


def _windows_validate_locks(
    locked: Sequence[tuple[Any, tuple[int, ...], Path, bool]],
) -> None:
    for handle, identity, path, directory in locked:
        if _windows_file_identity(handle, directory=directory) != identity:
            raise ReferenceToolError(f"locked directory changed: {path}")
        if _windows_final_path(handle) != Path(os.path.abspath(path)):
            raise ReferenceToolError(f"locked path moved: {path}")


def _windows_close_locks(
    locked: Sequence[tuple[Any, tuple[int, ...], Path, bool]],
) -> None:
    kernel = _windows_file_api()
    for handle, _identity, _path, _directory in reversed(locked):
        kernel.CloseHandle(handle)


def _windows_mark_delete(handle: Any) -> None:
    kernel = _windows_file_api()
    disposition = _WindowsDispositionInformation(1)
    if not kernel.SetFileInformationByHandle(
        handle,
        4,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        error = ctypes.get_last_error()
        raise ReferenceToolError(f"owned output cleanup failed ({error})")


def _windows_write_new(path: Path, encoded: bytes) -> None:
    if len(encoded) > 0xFFFFFFFF:
        raise ReferenceToolError("output exceeds the Windows write limit")
    locked = _windows_lock_components(path.parent)
    kernel = _windows_file_api()
    output_handle: Any = None
    created = False
    try:
        _windows_validate_locks(locked)
        if not _windows_private_acl_handle(locked[-1][0]):
            raise ReferenceToolError("output parent is not current-user-only")
        generic_write = 0x40000000
        delete_access = 0x00010000
        read_attributes = 0x00000080
        read_control = 0x00020000
        create_new = 1
        normal = 0x00000080
        open_reparse_point = 0x00200000
        output_handle = kernel.CreateFileW(
            str(path),
            generic_write | delete_access | read_attributes | read_control,
            0x00000001,
            None,
            create_new,
            normal | open_reparse_point,
            None,
        )
        if output_handle == ctypes.c_void_p(-1).value:
            output_handle = None
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError(str(path))
            raise ReferenceToolError(f"CreateFileW output failed ({error})")
        created = True
        created_identity = _windows_file_identity(output_handle, directory=False)
        if _windows_final_path(output_handle) != Path(os.path.abspath(path)):
            raise ReferenceToolError("created output final path mismatch")
        buffer = ctypes.create_string_buffer(encoded, len(encoded))
        written = ctypes.c_uint32()
        if not kernel.WriteFile(
            output_handle,
            buffer,
            len(encoded),
            ctypes.byref(written),
            None,
        ) or written.value != len(encoded):
            raise ReferenceToolError("WriteFile did not write the complete artifact")
        if not kernel.FlushFileBuffers(output_handle):
            raise ReferenceToolError("FlushFileBuffers failed")
        _windows_validate_locks(locked)
        if _windows_final_path(output_handle) != Path(os.path.abspath(path)):
            raise ReferenceToolError("created output path moved")
        named_handle, named_identity = _windows_open_locked(
            path, directory=False, share_delete=True
        )
        try:
            if named_identity != created_identity:
                raise ReferenceToolError("created output path identity changed")
        finally:
            kernel.CloseHandle(named_handle)
        if not _windows_private_acl_handle(locked[-1][0]):
            raise ReferenceToolError("output parent ACL changed")
        if not _windows_private_acl_handle(output_handle):
            raise ReferenceToolError("created output is not current-user-only")
    except BaseException as exc:
        if created and output_handle is not None:
            try:
                _windows_mark_delete(output_handle)
            except ReferenceToolError as cleanup_exc:
                raise cleanup_exc from exc
        raise
    finally:
        if output_handle is not None:
            kernel.CloseHandle(output_handle)
        _windows_close_locks(locked)


def _require_private_parent(path: Path) -> None:
    """Conservatively require a private output parent.

    R1 does not create the production root.  NTFS allows only the current user,
    LocalSystem, and built-in administrators.  POSIX rejects every group/other
    permission bit.
    """

    parent = _secure_existing(path.parent, directory=True, label="output parent")
    root = _secure_existing(REFERENCE_ROOT, directory=True, label="reference root")
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise ReferenceToolError("output is outside the frozen reference root") from exc
    if os.name == "nt":
        if not _windows_private_acl(parent):
            raise ReferenceToolError("output parent is not current-user-only")
    elif parent.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ReferenceToolError("output parent is not current-user-only")


def _secure_output(path: Path) -> Path:
    lexical = _absolute_lexical(path, label="output")
    if lexical.exists() or lexical.is_symlink():
        raise ReferenceToolError("output path already exists")
    _require_private_parent(lexical)
    return lexical


def _write_envelope(path: Path, schema: str, payload: Mapping[str, Any]) -> str:
    target = _secure_output(path)
    encoded = envelope_bytes(schema, payload)
    if os.name == "nt":
        _windows_write_new(target, encoded)
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(target, flags, 0o600)
        created = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(descriptor)
            named = os.stat(target, follow_symlinks=False)
            if not os.path.samestat(created, named):
                raise ReferenceToolError("created output path identity changed")
        except BaseException as exc:
            try:
                named = os.stat(target, follow_symlinks=False)
                if not os.path.samestat(created, named):
                    raise ReferenceToolError(
                        "owned output cleanup identity is ambiguous"
                    ) from exc
                target.unlink()
            finally:
                os.close(descriptor)
            raise
        os.close(descriptor)
    return hashlib.sha256(encoded).hexdigest()


def _read_input_bytes(path: Path, *, label: str, limit: int) -> tuple[Path, bytes]:
    if os.name == "nt":
        target = _absolute_lexical(path, label=label)
        locked = _windows_lock_file(target)
    else:
        target = _secure_input(path, label=label)
        locked = []
    flags = os.O_RDONLY
    for optional in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, optional, 0))
    try:
        descriptor = os.open(target, flags)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                opened = os.fstat(descriptor)
                named = os.stat(target, follow_symlinks=False)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or not os.path.samestat(opened, named)
                ):
                    raise ReferenceToolError(f"{label} changed while opening")
                if opened.st_size > limit:
                    raise ReferenceToolError(f"{label} exceeds the resource limit")
                data = handle.read(limit + 1)
                handle.seek(0)
                repeated = handle.read(limit + 1)
                after = os.fstat(descriptor)
                named_after = os.stat(target, follow_symlinks=False)
                if os.name == "nt":
                    _windows_validate_locks(locked)
                if len(data) > limit:
                    raise ReferenceToolError(f"{label} exceeds the resource limit")
                if (
                    data != repeated
                    or _signature(opened) != _signature(after)
                    or not os.path.samestat(opened, named_after)
                    or _signature(named) != _signature(named_after)
                ):
                    raise ReferenceToolError(f"{label} changed while being read")
                return target, data
        finally:
            os.close(descriptor)
    finally:
        if os.name == "nt":
            _windows_close_locks(locked)


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ReferenceToolError(f"{label} has missing or unknown fields")
    return value


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ReferenceToolError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReferenceToolError(f"{label} must be an exact integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise ReferenceToolError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ReferenceToolError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ReferenceToolError(f"{label} must be a finite number")
    outside = result <= 0.0 if positive else result < 0.0
    if outside:
        relation = "> 0" if positive else ">= 0"
        raise ReferenceToolError(f"{label} must be {relation}")
    return result


def _hash(value: Any, label: str, length: int = 64) -> str:
    result = _string(value, label)
    if len(result) != length or any(ch not in "0123456789abcdef" for ch in result):
        raise ReferenceToolError(f"{label} must be lowercase hexadecimal")
    return result


def _sorted_unique_strings(
    value: Any, label: str, *, nonempty: bool = True
) -> list[str]:
    if type(value) is not list:
        raise ReferenceToolError(f"{label} must be an array")
    result = [_string(item, f"{label} item") for item in value]
    if nonempty and not result:
        raise ReferenceToolError(f"{label} must not be empty")
    if result != sorted(result) or len(result) != len(set(result)):
        raise ReferenceToolError(f"{label} must be sorted and unique")
    return result


def validate_rules_clearance(value: Any) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "schema",
            "record_id",
            "reviewer",
            "reviewed_at_utc",
            "authority_basis",
            "build_sha256",
            "asset_scope",
            "local_read_only_derivation_permitted",
            "competition_use_permitted",
            "publication_limits",
        },
        "rules clearance",
    )
    if item["schema"] != RULES_SCHEMA:
        raise ReferenceToolError("rules clearance schema is not supported")
    for key in ("record_id", "reviewer", "reviewed_at_utc", "authority_basis"):
        _string(item[key], f"rules clearance {key}")
    hashes = _exact_object(
        item["build_sha256"], {"launcher", "payload", "pak"}, "build hashes"
    )
    for role, expected in EXPECTED_SHA256.items():
        if _hash(hashes[role], f"{role} hash") != expected:
            raise ReferenceToolError(f"rules clearance {role} hash mismatch")
    _sorted_unique_strings(item["asset_scope"], "asset scope")
    if not set(FROZEN_PACKAGE_STEMS) <= set(item["asset_scope"]):
        raise ReferenceToolError("rules clearance asset scope is incomplete")
    for key in (
        "local_read_only_derivation_permitted",
        "competition_use_permitted",
    ):
        if item[key] is not True:
            raise ReferenceToolError(f"rules clearance {key} must be exact true")
    publication_limits = item["publication_limits"]
    if type(publication_limits) is not list or any(
        type(limit) is not str or "\x00" in limit for limit in publication_limits
    ):
        raise ReferenceToolError("publication limits must be a string array")
    return item


def _load_clearance(path: Path) -> tuple[Path, dict[str, Any], bytes]:
    target, data = _read_input_bytes(
        path, label="rules clearance", limit=MAX_JSON_BYTES
    )
    return target, validate_rules_clearance(strict_json_bytes(data)), data


class _Cursor:
    def __init__(self, data: bytes, label: str) -> None:
        self.data = data
        self.label = label
        self.offset = 0

    def read(self, size: int) -> bytes:
        if type(size) is not int or size < 0 or self.offset + size > len(self.data):
            raise ReferenceToolError(f"{self.label} is truncated")
        result = self.data[self.offset : self.offset + size]
        self.offset += size
        return result

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def fstring(self) -> str:
        length = self.i32()
        if length == 0:
            return ""
        if length > 0:
            if length > MAX_STRING_BYTES:
                raise ReferenceToolError(f"{self.label} string exceeds limit")
            raw = self.read(length)
            if not raw.endswith(b"\x00") or b"\x00" in raw[:-1]:
                raise ReferenceToolError(f"{self.label} ANSI string is not canonical")
            try:
                return raw[:-1].decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ReferenceToolError(
                    f"{self.label} ANSI string is invalid"
                ) from exc
        units = -length
        if units > MAX_STRING_BYTES // 2:
            raise ReferenceToolError(f"{self.label} wide string exceeds limit")
        raw = self.read(units * 2)
        if not raw.endswith(b"\x00\x00"):
            raise ReferenceToolError(f"{self.label} wide string is not terminated")
        try:
            result = raw[:-2].decode("utf-16-le", errors="strict")
        except UnicodeDecodeError as exc:
            raise ReferenceToolError(f"{self.label} wide string is invalid") from exc
        if "\x00" in result:
            raise ReferenceToolError(f"{self.label} wide string contains NUL")
        return result

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise ReferenceToolError(f"{self.label} has trailing bytes")


def _bounded_region(
    offset: int,
    size: int,
    *,
    file_size: int,
    footer_offset: int,
    label: str,
) -> tuple[int, int]:
    if size <= 0 or size > MAX_INDEX_BYTES:
        raise ReferenceToolError(f"{label} size is invalid")
    if offset > MAX_SERIALIZED_I64 or size > MAX_SERIALIZED_I64:
        raise ReferenceToolError(f"{label} exceeds signed int64")
    end = offset + size
    if offset < 0 or end < offset or end > footer_offset or end > file_size:
        raise ReferenceToolError(f"{label} bounds escape the PAK")
    return offset, end


def _read_region(handle: BinaryIO, offset: int, size: int, label: str) -> bytes:
    handle.seek(offset)
    data = handle.read(size)
    if len(data) != size:
        raise ReferenceToolError(f"{label} is truncated")
    return data


def _parse_pointer(cursor: _Cursor, label: str) -> dict[str, Any] | None:
    present = cursor.u32()
    if present not in {0, 1}:
        raise ReferenceToolError(f"{label} presence flag is invalid")
    if not present:
        return None
    return {
        "offset": cursor.u64(),
        "size": cursor.u64(),
        "sha1": cursor.read(20).hex(),
    }


def _canonical_mount_suffix(mount: str) -> list[str]:
    if "\\" in mount or "\x00" in mount:
        raise ReferenceToolError("PAK mount path is not canonical")
    if not mount or not mount.endswith("/") or "//" in mount:
        raise ReferenceToolError("PAK mount path is not canonical")
    mount_parts = mount[:-1].split("/")
    parent_count = 0
    while parent_count < len(mount_parts) and mount_parts[parent_count] == "..":
        parent_count += 1
    if any(part in {"", ".", ".."} for part in mount_parts[parent_count:]):
        raise ReferenceToolError("PAK mount traversal is malformed")
    return mount_parts[parent_count:]


def _canonical_package_path(mount: str, directory: str, filename: str) -> str:
    for label, value in (("directory", directory), ("file", filename)):
        if "\\" in value or "\x00" in value:
            raise ReferenceToolError(f"PAK {label} path is not canonical")
    if "/" in filename or filename in {"", ".", ".."}:
        raise ReferenceToolError("PAK filename is invalid")
    if (
        not directory
        or directory.startswith("/")
        or not directory.endswith("/")
        or "//" in directory
    ):
        raise ReferenceToolError("PAK directory must end with slash")
    directory_parts = directory[:-1].split("/")
    if any(part in {".", ".."} for part in directory_parts):
        raise ReferenceToolError("PAK directory traversal is forbidden")
    mounted_parts = _canonical_mount_suffix(mount) + directory_parts + [filename]
    return "/".join(mounted_parts)


def inspect_pak(handle: BinaryIO, file_size: int) -> dict[str, Any]:
    if file_size < PAK_V11_FOOTER_BYTES:
        raise ReferenceToolError("PAK is shorter than a v11 footer")
    footer_offset = file_size - PAK_V11_FOOTER_BYTES
    footer = _read_region(
        handle, footer_offset, PAK_V11_FOOTER_BYTES, "PAK footer"
    )
    cursor = _Cursor(footer, "PAK footer")
    encryption_guid = cursor.read(16).hex()
    encrypted_byte = cursor.read(1)[0]
    if encrypted_byte not in {0, 1}:
        raise ReferenceToolError("PAK encryption flag is invalid")
    magic = cursor.u32()
    version = cursor.u32()
    index_offset = cursor.u64()
    index_size = cursor.u64()
    index_sha1 = cursor.read(20).hex()
    compression_names: list[str] = []
    for _ in range(5):
        raw = cursor.read(32)
        name_bytes, separator, padding = raw.partition(b"\x00")
        if not separator or any(padding):
            raise ReferenceToolError("PAK compression name is not NUL padded")
        try:
            name = name_bytes.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise ReferenceToolError("PAK compression name is not ASCII") from exc
        if name and name not in ALLOWED_COMPRESSION:
            raise ReferenceToolError("PAK compression name is unsupported")
        compression_names.append(name)
    populated_compression = [name for name in compression_names if name]
    if len(populated_compression) != len(set(populated_compression)):
        raise ReferenceToolError("PAK compression names are duplicated")
    cursor.finish()
    if magic != PAK_MAGIC or version != PAK_VERSION:
        raise ReferenceToolError("PAK magic/version mismatch")
    if encrypted_byte:
        raise ReferenceToolError("encrypted PAK index is forbidden")
    index_bounds = _bounded_region(
        index_offset,
        index_size,
        file_size=file_size,
        footer_offset=footer_offset,
        label="primary index",
    )
    primary = _read_region(handle, index_offset, index_size, "primary index")
    if hashlib.sha1(primary).hexdigest() != index_sha1:
        raise ReferenceToolError("primary index SHA-1 mismatch")
    primary_cursor = _Cursor(primary, "primary index")
    mount_point = primary_cursor.fstring()
    entry_count = primary_cursor.u32()
    if entry_count > MAX_ENTRIES:
        raise ReferenceToolError("primary entry count exceeds limit")
    path_hash_seed = primary_cursor.u64()
    path_pointer = _parse_pointer(primary_cursor, "path-hash index")
    directory_pointer = _parse_pointer(primary_cursor, "full-directory index")
    if path_pointer is None or directory_pointer is None:
        raise ReferenceToolError("both secondary indices are required")
    encoded_size = primary_cursor.u32()
    if encoded_size > MAX_INDEX_BYTES or (entry_count and encoded_size == 0):
        raise ReferenceToolError("encoded-entry table exceeds limit")
    primary_cursor.read(encoded_size)
    non_encoded_count = primary_cursor.u32()
    if non_encoded_count != 0:
        raise ReferenceToolError("non-encoded PAK entries are unsupported")
    primary_cursor.finish()

    regions = [index_bounds]
    secondary: dict[str, bytes] = {}
    for label, pointer in (
        ("path-hash index", path_pointer),
        ("full-directory index", directory_pointer),
    ):
        assert pointer is not None
        bounds = _bounded_region(
            pointer["offset"],
            pointer["size"],
            file_size=file_size,
            footer_offset=footer_offset,
            label=label,
        )
        if any(max(bounds[0], old[0]) < min(bounds[1], old[1]) for old in regions):
            raise ReferenceToolError("PAK index regions overlap")
        regions.append(bounds)
        data = _read_region(handle, bounds[0], pointer["size"], label)
        if hashlib.sha1(data).hexdigest() != pointer["sha1"]:
            raise ReferenceToolError(f"{label} SHA-1 mismatch")
        secondary[label] = data

    path_cursor = _Cursor(secondary["path-hash index"], "path-hash index")
    path_count = path_cursor.u32()
    if path_count > MAX_ENTRIES:
        raise ReferenceToolError("path-hash entry count exceeds limit")
    path_hashes: set[int] = set()
    path_locations: set[int] = set()
    for _ in range(path_count):
        value = path_cursor.u64()
        location = path_cursor.i32()
        if value in path_hashes:
            raise ReferenceToolError("duplicate path hash is forbidden")
        if location < 0 or location >= encoded_size:
            raise ReferenceToolError("path-hash entry location is invalid")
        if location in path_locations:
            raise ReferenceToolError("path-hash entry location is duplicated")
        path_hashes.add(value)
        path_locations.add(location)
    path_cursor.finish()

    directory_cursor = _Cursor(
        secondary["full-directory index"], "full-directory index"
    )
    directory_count = directory_cursor.u32()
    if directory_count > MAX_ENTRIES:
        raise ReferenceToolError("directory count exceeds limit")
    paths: list[str] = []
    raw_paths_by_canonical: dict[str, str] = {}
    casefold_paths: set[str] = set()
    directory_locations: set[int] = set()
    for _ in range(directory_count):
        directory = directory_cursor.fstring()
        file_count = directory_cursor.u32()
        if file_count > MAX_ENTRIES or len(paths) + file_count > MAX_ENTRIES:
            raise ReferenceToolError("directory file count exceeds limit")
        for _ in range(file_count):
            filename = directory_cursor.fstring()
            encoded_offset = directory_cursor.i32()
            if encoded_offset < 0 or encoded_offset >= encoded_size:
                raise ReferenceToolError("directory entry location is invalid")
            if encoded_offset in directory_locations:
                raise ReferenceToolError("directory entry location is duplicated")
            directory_locations.add(encoded_offset)
            raw_path = f"{directory}{filename}"
            package_path = _canonical_package_path(mount_point, directory, filename)
            folded = package_path.casefold()
            if folded in casefold_paths:
                raise ReferenceToolError("duplicate/case-ambiguous package path")
            casefold_paths.add(folded)
            paths.append(package_path)
            raw_paths_by_canonical[package_path] = raw_path
    directory_cursor.finish()
    paths.sort()
    if (
        len(paths) != entry_count
        or path_count != entry_count
        or path_locations != directory_locations
    ):
        raise ReferenceToolError("PAK index entry counts do not reconcile")
    matches = sorted(
        path
        for path in paths
        if any(path == f"{stem}.uasset" for stem in FROZEN_PACKAGE_STEMS)
    )
    missing = [
        stem
        for stem in FROZEN_PACKAGE_STEMS
        if f"{stem}.uasset" not in matches
    ]
    if missing:
        raise ReferenceToolError(f"frozen candidate package is missing: {missing}")
    directory_index_paths = sorted(raw_paths_by_canonical[path] for path in matches)
    return {
        "encryption_guid": encryption_guid,
        "encrypted": False,
        "magic": magic,
        "version": version,
        "index_offset": index_offset,
        "index_size": index_size,
        "index_sha1": index_sha1,
        "path_hash_seed": path_hash_seed,
        "path_hash_index_offset": path_pointer["offset"],
        "path_hash_index_size": path_pointer["size"],
        "path_hash_index_sha1": path_pointer["sha1"],
        "full_directory_index_offset": directory_pointer["offset"],
        "full_directory_index_size": directory_pointer["size"],
        "full_directory_index_sha1": directory_pointer["sha1"],
        "compression_names": compression_names,
        "mount_point": mount_point,
        "entry_count": entry_count,
        "directory_index_paths": directory_index_paths,
        "candidate_package_paths": matches,
    }


def _file_identity(value: Any, label: str) -> dict[str, Any]:
    item = _exact_object(value, {"path", "size_bytes", "sha256"}, label)
    path = Path(_string(item["path"], f"{label} path"))
    if not path.is_absolute():
        raise ReferenceToolError(f"{label} path must be absolute")
    _integer(item["size_bytes"], f"{label} size")
    _hash(item["sha256"], f"{label} SHA-256")
    return item


def _evidence_identity(value: Any, label: str) -> dict[str, Any]:
    item = _exact_object(
        value,
        {"id", "producer_id", "method_id", "artifact_sha256"},
        label,
    )
    for key in ("id", "producer_id", "method_id"):
        _string(item[key], f"{label} {key}")
    _hash(item["artifact_sha256"], f"{label} artifact SHA-256")
    return item


def _evidence_array(value: Any, label: str) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise ReferenceToolError(f"{label} must be a nonempty array")
    result = [_evidence_identity(item, f"{label} item") for item in value]
    ids = [item["id"] for item in result]
    if len(ids) != len(set(ids)):
        raise ReferenceToolError(f"{label} IDs must be unique")
    return result


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if type(value) is not list or len(value) != length:
        raise ReferenceToolError(f"{label} must have exactly {length} values")
    result: list[float] = []
    for item in value:
        if type(item) not in {int, float}:
            raise ReferenceToolError(f"{label} values must be finite numbers")
        try:
            converted = float(item)
        except OverflowError as exc:
            raise ReferenceToolError(
                f"{label} values must be finite numbers"
            ) from exc
        if not math.isfinite(converted):
            raise ReferenceToolError(f"{label} values must be finite numbers")
        result.append(converted)
    return result


def _determinant3(matrix: Sequence[float]) -> float:
    a, b, c = matrix[0], matrix[1], matrix[2]
    d, e, f = matrix[4], matrix[5], matrix[6]
    g, h, i = matrix[8], matrix[9], matrix[10]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _relative_close(left: float, right: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    largest = max(abs(left), abs(right))
    return (
        left == right
        if largest == 0.0
        else abs(left - right) <= TRANSFORM_RELATIVE_ROUNDOFF * largest
    )


def _transform_link(value: Any, label: str) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "id",
            "parent_frame",
            "child_frame",
            "matrix_row_major",
            "determinant",
            "singular_values",
            "handedness",
            "scale_model",
            "evidence",
        },
        label,
    )
    for key in ("id", "parent_frame", "child_frame"):
        _string(item[key], f"{label} {key}")
    matrix = _finite_vector(item["matrix_row_major"], 16, f"{label} matrix")
    determinant = _number(item["determinant"], f"{label} determinant", positive=True)
    singular = _finite_vector(item["singular_values"], 3, f"{label} singular values")
    if any(value <= 0.0 for value in singular):
        raise ReferenceToolError(f"{label} singular values must be positive")
    if item["handedness"] != "right" or item["scale_model"] != "uniform":
        raise ReferenceToolError(f"{label} must be right-handed uniform scale")
    if matrix[12:16] != [0.0, 0.0, 0.0, 1.0]:
        raise ReferenceToolError(f"{label} is not an affine row-major matrix")
    computed_det = _determinant3(matrix)
    if (
        not math.isfinite(computed_det)
        or computed_det <= 0.0
        or not _relative_close(computed_det, determinant)
    ):
        raise ReferenceToolError(f"{label} determinant is inconsistent")
    columns = (
        (matrix[0], matrix[4], matrix[8]),
        (matrix[1], matrix[5], matrix[9]),
        (matrix[2], matrix[6], matrix[10]),
    )
    norms = [
        math.sqrt(sum(component * component for component in column))
        for column in columns
    ]
    scale = sum(norms) / 3.0
    if (
        not all(math.isfinite(norm) for norm in norms)
        or not math.isfinite(scale)
        or scale <= 0.0
        or any(not _relative_close(norm / scale, 1.0) for norm in norms)
    ):
        raise ReferenceToolError(f"{label} is not uniform scale")
    for first in range(3):
        for second in range(first + 1, 3):
            dot = sum(
                columns[first][axis] * columns[second][axis]
                for axis in range(3)
            )
            normalized_dot = dot / (norms[first] * norms[second])
            if (
                not math.isfinite(normalized_dot)
                or abs(normalized_dot) > TRANSFORM_ORTHOGONALITY_ROUNDOFF
            ):
                raise ReferenceToolError(f"{label} contains shear")
    if any(not _relative_close(value / scale, 1.0) for value in singular):
        raise ReferenceToolError(f"{label} singular values are inconsistent")
    _evidence_array(item["evidence"], f"{label} evidence")
    return item


def _validate_pak_index(value: Any, *, pak_size: int) -> dict[str, Any]:
    keys = {
        "magic",
        "version",
        "encryption_guid",
        "encrypted",
        "index_offset",
        "index_size",
        "index_sha1",
        "path_hash_seed",
        "path_hash_index_offset",
        "path_hash_index_size",
        "path_hash_index_sha1",
        "full_directory_index_offset",
        "full_directory_index_size",
        "full_directory_index_sha1",
        "compression_names",
        "mount_point",
        "entry_count",
    }
    item = _exact_object(value, keys, "build PAK index")
    if (
        _integer(item["magic"], "build PAK magic") != PAK_MAGIC
        or _integer(item["version"], "build PAK version") != PAK_VERSION
    ):
        raise ReferenceToolError("candidate PAK magic/version mismatch")
    _hash(item["encryption_guid"], "build PAK encryption GUID", length=32)
    if item["encrypted"] is not False:
        raise ReferenceToolError("candidate PAK must have unencrypted index")
    for key in (
        "index_offset",
        "path_hash_index_offset",
        "full_directory_index_offset",
    ):
        number = _integer(item[key], f"build PAK {key}")
        if number > MAX_SERIALIZED_I64:
            raise ReferenceToolError(f"build PAK {key} exceeds signed int64")
    for key in (
        "index_size",
        "path_hash_index_size",
        "full_directory_index_size",
    ):
        size = _integer(item[key], f"build PAK {key}", minimum=1)
        if size > min(MAX_INDEX_BYTES, MAX_SERIALIZED_I64):
            raise ReferenceToolError(f"build PAK {key} exceeds limit")
    path_hash_seed = _integer(item["path_hash_seed"], "build PAK path-hash seed")
    if path_hash_seed > 2**64 - 1:
        raise ReferenceToolError("build PAK path-hash seed exceeds uint64")
    entry_count = _integer(item["entry_count"], "build PAK entry count")
    if entry_count > MAX_ENTRIES:
        raise ReferenceToolError("build PAK entry count exceeds limit")
    for key in (
        "index_sha1",
        "path_hash_index_sha1",
        "full_directory_index_sha1",
    ):
        _hash(item[key], f"build PAK {key}", length=40)
    names = item["compression_names"]
    if (
        type(names) is not list
        or len(names) != 5
        or any(type(name) is not str for name in names)
    ):
        raise ReferenceToolError("compression names must contain five exact strings")
    populated = [name for name in names if name]
    if len(populated) != len(set(populated)) or any(
        name not in ALLOWED_COMPRESSION for name in populated
    ):
        raise ReferenceToolError("compression names are invalid")
    mount = _string(item["mount_point"], "PAK mount point")
    _canonical_mount_suffix(mount)
    if pak_size < PAK_V11_FOOTER_BYTES:
        raise ReferenceToolError("candidate PAK size is too small")
    footer_offset = pak_size - PAK_V11_FOOTER_BYTES
    regions = []
    for prefix in ("index", "path_hash_index", "full_directory_index"):
        bounds = _bounded_region(
            item[f"{prefix}_offset"],
            item[f"{prefix}_size"],
            file_size=pak_size,
            footer_offset=footer_offset,
            label=f"candidate {prefix}",
        )
        if any(
            max(bounds[0], old[0]) < min(bounds[1], old[1])
            for old in regions
        ):
            raise ReferenceToolError("candidate PAK index regions overlap")
        regions.append(bounds)
    return item


def _validate_build(value: Any) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "build",
            "mode",
            "launcher",
            "payload",
            "pak",
            "pak_index",
            "package_paths",
        },
        "candidate build",
    )
    if (
        _integer(item["build"], "candidate build number") != 3385
        or item["mode"] != "Training"
    ):
        raise ReferenceToolError("candidate build/mode mismatch")
    identities = {
        role: _file_identity(item[role], f"candidate {role}")
        for role in ("launcher", "payload", "pak")
    }
    expected_paths = {
        "launcher": LAUNCHER_PATH,
        "payload": PAYLOAD_PATH,
        "pak": PAK_PATH,
    }
    for role, identity in identities.items():
        if Path(identity["path"]) != expected_paths[role]:
            raise ReferenceToolError(f"candidate {role} path mismatch")
        if identity["sha256"] != EXPECTED_SHA256[role]:
            raise ReferenceToolError(f"candidate {role} hash mismatch")
    _validate_pak_index(item["pak_index"], pak_size=identities["pak"]["size_bytes"])
    package_paths = _sorted_unique_strings(item["package_paths"], "package paths")
    expected_package_paths = sorted(
        f"{stem}.uasset" for stem in FROZEN_PACKAGE_STEMS
    )
    if package_paths != expected_package_paths:
        raise ReferenceToolError("candidate package paths are not the frozen set")
    return item


def _validate_parser(
    value: Any,
    *,
    source_identity: Mapping[str, Any] | None = None,
    interpreter_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "implementation_id",
            "source",
            "interpreter_sha256",
            "dependencies",
            "config_sha256",
        },
        "candidate parser",
    )
    if item["implementation_id"] != PARSER_IMPLEMENTATION:
        raise ReferenceToolError("candidate parser implementation mismatch")
    source = _file_identity(item["source"], "candidate parser source")
    actual_source = Path(__file__).resolve()
    if Path(source["path"]) != actual_source:
        raise ReferenceToolError("candidate parser source path mismatch")
    actual_source_identity = (
        _identity_for(actual_source)
        if source_identity is None
        else dict(source_identity)
    )
    if source != actual_source_identity:
        raise ReferenceToolError("candidate parser source identity mismatch")
    interpreter = (
        _identity_for(Path(sys.executable).resolve())
        if interpreter_identity is None
        else dict(interpreter_identity)
    )
    if interpreter["sha256"] != INTERPRETER_SHA256:
        raise ReferenceToolError("running interpreter identity mismatch")
    if _hash(item["interpreter_sha256"], "interpreter hash") != interpreter["sha256"]:
        raise ReferenceToolError("candidate interpreter hash mismatch")
    if type(item["dependencies"]) is not list or item["dependencies"]:
        raise ReferenceToolError("candidate parser dependencies must be empty")
    if _hash(item["config_sha256"], "parser config hash") != parser_config_sha256():
        raise ReferenceToolError("candidate parser config hash mismatch")
    return item


def _validate_geometry(value: Any) -> tuple[dict[str, Any], set[str], set[str]]:
    item = _exact_object(
        value,
        {
            "mesh_package",
            "active_lod",
            "render_not_collision_only",
            "coordinate_convention",
            "units",
            "features",
            "bounds",
        },
        "candidate geometry",
    )
    if (
        _string(item["mesh_package"], "geometry mesh package")
        != FROZEN_PACKAGE_STEMS[1]
    ):
        raise ReferenceToolError("geometry mesh package is not the frozen candidate")
    _integer(item["active_lod"], "geometry active LOD")
    if item["render_not_collision_only"] is not True:
        raise ReferenceToolError("geometry must identify render, not collision, mesh")
    _string(item["coordinate_convention"], "geometry convention")
    _string(item["units"], "geometry units")
    if type(item["features"]) is not list or not item["features"]:
        raise ReferenceToolError("geometry features must be nonempty")
    feature_ids: set[str] = set()
    kinds: set[str] = set()
    pending_refs: list[str] = []
    surface_ids: set[str] = set()
    for raw in item["features"]:
        feature = _exact_object(
            raw,
            {"id", "kind", "coordinates", "references", "evidence"},
            "geometry feature",
        )
        feature_id = _string(feature["id"], "geometry feature ID")
        if feature_id in feature_ids:
            raise ReferenceToolError("geometry feature IDs must be unique")
        feature_ids.add(feature_id)
        if feature["kind"] not in {"vertex", "edge", "surface"}:
            raise ReferenceToolError("geometry feature kind is invalid")
        kinds.add(feature["kind"])
        if feature["kind"] == "surface":
            surface_ids.add(feature_id)
        if type(feature["coordinates"]) is not list or not feature["coordinates"]:
            raise ReferenceToolError("geometry coordinates must be nonempty")
        for point in feature["coordinates"]:
            _finite_vector(point, 3, "geometry coordinate")
        refs = _sorted_unique_strings(
            feature["references"], "geometry feature references", nonempty=False
        )
        pending_refs.extend(refs)
        _evidence_array(feature["evidence"], "geometry feature evidence")
    if kinds != {"vertex", "edge", "surface"}:
        raise ReferenceToolError("geometry must include vertex, edge, and surface")
    if any(reference not in feature_ids for reference in pending_refs):
        raise ReferenceToolError("geometry feature reference does not resolve")
    bounds = _exact_object(
        item["bounds"], {"planarity", "aspect", "thickness", "bevel"}, "geometry bounds"
    )
    for key in bounds:
        _number(bounds[key], f"geometry {key} bound")
    return item, feature_ids, surface_ids


def _validate_transform_chain(value: Any) -> dict[str, Any]:
    item = _exact_object(
        value, {"links", "active_actor_overrides"}, "candidate transform chain"
    )
    all_ids: set[str] = set()
    for key in ("links", "active_actor_overrides"):
        if type(item[key]) is not list or not item[key]:
            raise ReferenceToolError(f"transform {key} has invalid cardinality")
        links = [_transform_link(raw, f"transform {key} link") for raw in item[key]]
        ids = [link["id"] for link in links]
        if len(ids) != len(set(ids)):
            raise ReferenceToolError(f"transform {key} IDs must be unique")
        if all_ids.intersection(ids):
            raise ReferenceToolError("transform IDs must be unique across the chain")
        all_ids.update(ids)
        for prior, current in zip(links, links[1:]):
            if prior["child_frame"] != current["parent_frame"]:
                raise ReferenceToolError(f"transform {key} is not an ordered chain")
    return item


def _validate_training_linkage(
    value: Any, geometry: Mapping[str, Any]
) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "master_map",
            "track_map",
            "gate_blueprint",
            "component",
            "mesh",
            "material",
            "lod",
            "udp_camera",
            "proved",
            "evidence",
        },
        "candidate training linkage",
    )
    for key in (
        "master_map",
        "track_map",
        "gate_blueprint",
        "component",
        "mesh",
        "material",
        "udp_camera",
    ):
        _string(item[key], f"training linkage {key}")
    _integer(item["lod"], "training linkage LOD")
    if item["proved"] is not True:
        raise ReferenceToolError("training linkage proved must be exact true")
    if (
        item["mesh"] != geometry["mesh_package"]
        or item["lod"] != geometry["active_lod"]
    ):
        raise ReferenceToolError("training linkage and geometry disagree")
    _evidence_array(item["evidence"], "training linkage evidence")
    return item


def _validate_visibility(
    value: Any, feature_ids: set[str], surface_ids: set[str]
) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "model_id",
            "surface_ids",
            "feature_ids",
            "front_policy",
            "back_policy",
            "bevel_policy",
            "clipping_policy",
            "occlusion_policy",
            "systematic_bounds",
            "evidence",
        },
        "candidate visibility",
    )
    for key in (
        "model_id",
        "front_policy",
        "back_policy",
        "bevel_policy",
        "clipping_policy",
        "occlusion_policy",
    ):
        _string(item[key], f"visibility {key}")
    selected_surfaces = set(
        _sorted_unique_strings(item["surface_ids"], "visibility surfaces")
    )
    selected_features = set(
        _sorted_unique_strings(item["feature_ids"], "visibility features")
    )
    if not selected_surfaces <= surface_ids or not selected_features <= feature_ids:
        raise ReferenceToolError("visibility feature reference does not resolve")
    bounds = _exact_object(
        item["systematic_bounds"],
        {"front_back_px", "bevel_px", "clipping_px", "occlusion_px"},
        "visibility bounds",
    )
    for key in bounds:
        _number(bounds[key], f"visibility {key}")
    _evidence_array(item["evidence"], "visibility evidence")
    return item


def _validate_uncertainty(value: Any) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "conditional_pixel_model_id",
            "shared_nuisance_ledger_id",
            "pixel_center_convention",
            "bounds",
            "evidence",
        },
        "candidate uncertainty",
    )
    _string(item["conditional_pixel_model_id"], "conditional pixel model ID")
    _string(item["shared_nuisance_ledger_id"], "shared nuisance ledger ID")
    if item["pixel_center_convention"] != "integer-coordinates-are-pixel-centers":
        raise ReferenceToolError("pixel-center convention is not frozen")
    bounds = _exact_object(
        item["bounds"],
        {
            "render_lod_px",
            "material_px",
            "antialias_px",
            "jpeg_px",
            "annotation_px",
            "geometry_units",
            "transform_units",
        },
        "uncertainty bounds",
    )
    for key in bounds:
        _number(bounds[key], f"uncertainty {key}")
    _evidence_array(item["evidence"], "uncertainty evidence")
    return item


def _validate_independent_checks(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) < 2:
        raise ReferenceToolError("at least two independent checks are required")
    result: list[dict[str, Any]] = []
    for raw in value:
        item = _exact_object(
            raw,
            {
                "check_id",
                "implementation_id",
                "producer_id",
                "input_sha256",
                "output_sha256",
                "passed",
            },
            "independent check",
        )
        for key in ("check_id", "implementation_id", "producer_id"):
            _string(item[key], f"independent check {key}")
        _hash(item["input_sha256"], "independent check input hash")
        _hash(item["output_sha256"], "independent check output hash")
        if item["passed"] is not True:
            raise ReferenceToolError("independent check passed must be exact true")
        result.append(item)
    for key in ("check_id", "implementation_id", "producer_id"):
        values = [item[key] for item in result]
        if len(values) != len(set(values)):
            raise ReferenceToolError(f"independent check {key}s must differ")
    return result


def _validate_rules_binding(
    value: Any,
    clearance_path: Path,
    clearance: Mapping[str, Any],
    clearance_bytes: bytes,
) -> dict[str, Any]:
    item = _exact_object(value, {"clearance", "record_id"}, "candidate rules")
    identity = _file_identity(item["clearance"], "candidate clearance")
    if Path(identity["path"]) != clearance_path:
        raise ReferenceToolError("candidate clearance path mismatch")
    if identity["size_bytes"] != len(clearance_bytes):
        raise ReferenceToolError("candidate clearance size mismatch")
    if identity["sha256"] != hashlib.sha256(clearance_bytes).hexdigest():
        raise ReferenceToolError("candidate clearance hash mismatch")
    if item["record_id"] != clearance["record_id"]:
        raise ReferenceToolError("candidate clearance record ID mismatch")
    return item


def _validate_annotation_contract(value: Any) -> dict[str, Any]:
    item = _exact_object(
        value,
        {
            "observation_schema",
            "producer_id",
            "producer_sha256",
            "preprocessing_sha256",
            "correspondence_sha256",
            "rejection_sha256",
            "covariance_sha256",
            "shared_nuisance_ledger_sha256",
            "checker_id",
            "checker_sha256",
        },
        "annotation contract",
    )
    if item["observation_schema"] != OBSERVATION_SCHEMA:
        raise ReferenceToolError("annotation observation schema mismatch")
    producer = _string(item["producer_id"], "annotation producer ID")
    checker = _string(item["checker_id"], "annotation checker ID")
    if producer == checker:
        raise ReferenceToolError("annotation checker must be independent")
    for key in (
        "producer_sha256",
        "preprocessing_sha256",
        "correspondence_sha256",
        "rejection_sha256",
        "covariance_sha256",
        "shared_nuisance_ledger_sha256",
        "checker_sha256",
    ):
        _hash(item[key], f"annotation {key}")
    if item["producer_sha256"] == item["checker_sha256"]:
        raise ReferenceToolError("annotation producer/checker code must differ")
    return item


def validate_candidate(
    value: Any,
    *,
    clearance_path: Path,
    clearance: Mapping[str, Any],
    clearance_bytes: bytes,
    parser_source_identity: Mapping[str, Any] | None = None,
    interpreter_identity: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    item = _exact_object(
        value,
        {
            "schema",
            "build",
            "parser",
            "geometry",
            "transform_chain",
            "training_linkage",
            "visibility",
            "uncertainty",
            "independent_checks",
            "rules",
            "annotation_contract",
        },
        "reference candidate",
    )
    if item["schema"] != CANDIDATE_SCHEMA:
        raise ReferenceToolError("reference candidate schema is not supported")
    build = _validate_build(item["build"])
    _validate_parser(
        item["parser"],
        source_identity=parser_source_identity,
        interpreter_identity=interpreter_identity,
    )
    geometry, feature_ids, surface_ids = _validate_geometry(item["geometry"])
    _validate_transform_chain(item["transform_chain"])
    _validate_training_linkage(item["training_linkage"], geometry)
    _validate_visibility(item["visibility"], feature_ids, surface_ids)
    _validate_uncertainty(item["uncertainty"])
    _validate_independent_checks(item["independent_checks"])
    _validate_rules_binding(
        item["rules"], clearance_path, clearance, clearance_bytes
    )
    _validate_annotation_contract(item["annotation_contract"])
    build_hashes = clearance["build_sha256"]
    for role in ("launcher", "payload", "pak"):
        if build[role]["sha256"] != build_hashes[role]:
            raise ReferenceToolError("candidate build and clearance disagree")
    return {
        "build": True,
        "parser": True,
        "geometry": True,
        "transform_chain": True,
        "training_linkage": True,
        "visibility": True,
        "uncertainty": True,
        "independent_checks_structural_only": True,
        "rules_record_structural_only": True,
        "annotation_contract": True,
    }


def _identity_for(path: Path, *, expected_hash: str | None = None) -> dict[str, Any]:
    identity, _unused = _stable_identity(
        path, label="build source", expected_hash=expected_hash
    )
    return identity


def run_inspect_build(*, rules_clearance: Path, output: Path) -> dict[str, Any]:
    script_before = _identity_for(Path(__file__).resolve())
    interpreter = _identity_for(
        Path(sys.executable).resolve(), expected_hash=INTERPRETER_SHA256
    )
    clearance_path, clearance, clearance_bytes = _load_clearance(rules_clearance)
    launcher = _identity_for(
        LAUNCHER_PATH, expected_hash=EXPECTED_SHA256["launcher"]
    )
    payload = _identity_for(PAYLOAD_PATH, expected_hash=EXPECTED_SHA256["payload"])
    pak_identity, pak_index = _stable_identity(
        PAK_PATH,
        label="PAK source",
        inspect=inspect_pak,
        expected_hash=EXPECTED_SHA256["pak"],
    )
    assert type(pak_index) is dict
    script_identity = _identity_for(
        Path(__file__).resolve(), expected_hash=script_before["sha256"]
    )
    if script_identity != script_before:
        raise ReferenceToolError("parser source identity changed")
    payload_value = {
        "build": 3385,
        "mode": "Training",
        "sources": {
            "launcher": launcher,
            "payload": payload,
            "pak": pak_identity,
        },
        "rules_clearance": {
            "path": str(clearance_path),
            "sha256": hashlib.sha256(clearance_bytes).hexdigest(),
            "record_id": clearance["record_id"],
            "structural_only": True,
        },
        "parser": {
            "implementation_id": PARSER_IMPLEMENTATION,
            "source": script_identity,
            "interpreter": interpreter,
            "dependencies": [],
            "config_sha256": parser_config_sha256(),
        },
        "pak_index": pak_index,
        "admitted": False,
        "missing_r3_claims": list(MISSING_R3_CLAIMS),
        "semantic_claims": "unmeasured",
    }
    script_final = _identity_for(
        Path(__file__).resolve(), expected_hash=script_before["sha256"]
    )
    if script_final != script_before:
        raise ReferenceToolError("parser source identity changed before output")
    artifact_sha256 = _write_envelope(output, DISCOVERY_SCHEMA, payload_value)
    return {"path": str(output), "sha256": artifact_sha256, "admitted": False}


def run_validate_candidate(
    *, rules_clearance: Path, candidate: Path, output: Path
) -> dict[str, Any]:
    script_before = _identity_for(Path(__file__).resolve())
    interpreter = _identity_for(
        Path(sys.executable).resolve(), expected_hash=INTERPRETER_SHA256
    )
    clearance_path, clearance, clearance_bytes = _load_clearance(rules_clearance)
    candidate_path, candidate_bytes = _read_input_bytes(
        candidate, label="reference candidate", limit=MAX_JSON_BYTES
    )
    candidate_value = strict_json_bytes(candidate_bytes)
    checks = validate_candidate(
        candidate_value,
        clearance_path=clearance_path,
        clearance=clearance,
        clearance_bytes=clearance_bytes,
        parser_source_identity=script_before,
        interpreter_identity=interpreter,
    )
    payload_value = {
        "candidate": {
            "path": str(candidate_path),
            "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        },
        "rules_clearance": {
            "path": str(clearance_path),
            "sha256": hashlib.sha256(clearance_bytes).hexdigest(),
            "record_id": clearance["record_id"],
        },
        "checks": checks,
        "structurally_valid": True,
        "admitted": False,
        "independent_review_required": True,
        "semantic_linkage": "unverified",
        "rules_authority": "unverified-by-tool",
        "independence": "structural-claims-only",
    }
    script_final = _identity_for(
        Path(__file__).resolve(), expected_hash=script_before["sha256"]
    )
    if script_final != script_before:
        raise ReferenceToolError("parser source identity changed before output")
    artifact_sha256 = _write_envelope(output, VALIDATION_SCHEMA, payload_value)
    return {"path": str(output), "sha256": artifact_sha256, "admitted": False}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect-build", allow_abbrev=False)
    inspect.add_argument("--rules-clearance", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate-candidate", allow_abbrev=False)
    validate.add_argument("--rules-clearance", type=Path, required=True)
    validate.add_argument("--candidate", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "inspect-build":
            result = run_inspect_build(
                rules_clearance=args.rules_clearance, output=args.output
            )
        else:
            result = run_validate_candidate(
                rules_clearance=args.rules_clearance,
                candidate=args.candidate,
                output=args.output,
            )
    except (OSError, ReferenceToolError) as exc:
        print(f"reference tool failed: {exc}", file=sys.stderr)
        return 2
    print(f"{result['sha256']}  {result['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
