"""Private, immutable VQ2 replay bundles and deterministic scoring.

Each bundle is one flight/session directory.  Decoded BGR frame arrays are
content-addressed and referenced by synchronized JSONL frame records.  The
bundle is private by default: a destination inside a Git checkout must be
ignored by Git or creation fails before any pixel is written.
"""

from __future__ import annotations

import dataclasses
import base64
import io
import importlib
import inspect
import json
import math
import os
import platform
import queue
import random
import re
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from types import MappingProxyType
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from competition.vq2_capture import MavlinkIngressV1, ReceivedIMUSampleV1
from competition.vq2_contracts import FrameTimingV1

from ._util import (
    canonical_json,
    git_provenance,
    json_hash,
    private_path_guard,
    read_secure_regular_file,
    secure_directory,
    secure_relative_regular_file,
    secure_regular_file,
    sha256_bytes,
    sha256_file,
    sha256_text,
    strict_json_load,
    strict_json_loads,
)


BUNDLE_SCHEMA = "aigp-vq2-replay/1"
RECORD_SCHEMA = "aigp-vq2-replay-record/1"
ANNOTATION_SCHEMA = "aigp-vq2-replay-annotation/1"
CORPUS_SCHEMA = "aigp-vq2-replay-corpus/1"
RECORDING_NOTICE = (
    "Private competition-development artifact. Do not publish, commit, or "
    "broadcast without organizer approval."
)
_RECORD_ENVELOPE_KEYS = frozenset(
    {"schema", "session_id", "sequence", "type", "capture_wall_time_ns"}
)
_CORE_RECORD_FIELDS = {
    "imu": frozenset({"received_monotonic_s", "imu", "estimator"}),
    "race_status": frozenset({"received_monotonic_s", "race_status"}),
    "command": frozenset({"kind", "monotonic_s", "frame_token", "command"}),
    "decoded_frame": frozenset(
        {
            "generation",
            "frame_id",
            "sim_time_ns",
            "received_monotonic_s",
            "frame_blob",
            "frame_hash",
            "image_shape",
            "image_dtype",
        }
    ),
    "frame": frozenset(
        {
            "generation",
            "frame_id",
            "sim_time_ns",
            "received_monotonic_s",
            "frame_blob",
            "frame_hash",
            "image_shape",
            "image_dtype",
            "detector_latency_ms",
            "detections",
            "tracker",
            "imu",
            "estimator",
            "race_status",
            "generated_command",
            "sent_command",
            "phase",
        }
    ),
}
_EVENT_FORBIDDEN_FIELDS = frozenset(
    {
        "record_type",
        "record_schema",
        "dataset_hash",
        "integrity",
        "manifest",
        "frame_blob",
        "frame_hash",
        "image_shape",
        "image_dtype",
    }
)
_EVENT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")

# Resource ceilings are part of the replay format's trusted-reader boundary.
# They leave ample headroom for 4K VQ2 capture while preventing tiny hostile
# headers/manifests from driving unbounded allocation or corpus-wide work.
MAX_REPLAY_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_REPLAY_RECORDS_BYTES = 128 * 1024 * 1024
MAX_REPLAY_RECORD_LINE_BYTES = 2 * 1024 * 1024
MAX_REPLAY_RECORD_COUNT = 250_000
MAX_REPLAY_FRAME_BLOB_COUNT = 20_000
MAX_REPLAY_FRAME_WIDTH = 4096
MAX_REPLAY_FRAME_HEIGHT = 2160
MAX_REPLAY_FRAME_PIXELS = MAX_REPLAY_FRAME_WIDTH * MAX_REPLAY_FRAME_HEIGHT
MAX_REPLAY_FRAME_DECODED_BYTES = MAX_REPLAY_FRAME_PIXELS * 3
MAX_REPLAY_FRAME_BLOB_BYTES = 32 * 1024 * 1024
MAX_REPLAY_SESSION_BLOB_BYTES = 16 * 1024 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 64 * 1024
MAX_REPLAY_ANNOTATIONS_BYTES = 64 * 1024 * 1024
MAX_REPLAY_ANNOTATION_LINE_BYTES = 1024 * 1024
MAX_REPLAY_ANNOTATION_COUNT = 100_000
MAX_REPLAY_CORPUS_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_REPLAY_POLICY_BYTES = 2 * 1024 * 1024
MAX_REPLAY_CORPUS_SESSION_COUNT = 1_000
_MANIFEST_FINALIZATION_RESERVE_BYTES = 128 * 1024


def _read_bounded_secure_file(
    path: Path | str, *, maximum_bytes: int, label: str
) -> bytes:
    """Read one stable regular file with a descriptor-enforced byte ceiling."""

    try:
        return read_secure_regular_file(path, maximum_bytes=maximum_bytes)
    except ValueError as exc:
        if "exceeds resource limit" in str(exc):
            raise ValueError(f"{label} exceeds replay resource limit") from exc
        raise


def _validate_npy_payload_header(payload: bytes) -> tuple[int, int, int]:
    """Validate allocation-relevant NPY metadata before ``numpy.load``."""

    if len(payload) > MAX_REPLAY_FRAME_BLOB_BYTES:
        raise ValueError("frame blob exceeds replay resource limit")
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=MAX_NPY_HEADER_BYTES
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream, max_header_size=MAX_NPY_HEADER_BYTES
            )
        else:
            raise ValueError("unsupported NPY frame version")
    except (EOFError, UnicodeError, ValueError) as exc:
        raise ValueError("frame blob has an invalid bounded NPY header") from exc
    if (
        type(shape) is not tuple
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
        or shape[2] != 3
        or fortran_order is not False
        or np.dtype(dtype) != np.dtype(np.uint8)
    ):
        raise ValueError("frame blob must encode one C-order HxWx3 uint8 array")
    height, width, channels = shape
    pixels = height * width
    decoded_bytes = pixels * channels
    if (
        height > MAX_REPLAY_FRAME_HEIGHT
        or width > MAX_REPLAY_FRAME_WIDTH
        or pixels > MAX_REPLAY_FRAME_PIXELS
        or decoded_bytes > MAX_REPLAY_FRAME_DECODED_BYTES
    ):
        raise ValueError("frame blob decoded shape exceeds replay resource limit")
    if stream.tell() + decoded_bytes != len(payload):
        raise ValueError("frame blob payload length contradicts its NPY header")
    return height, width, channels


def _validate_record_shape(row: Mapping[str, Any], *, location: str) -> None:
    if type(row) is not dict:
        raise ValueError(f"record must be an exact object {location}")
    record_type = row.get("type")
    if record_type in _CORE_RECORD_FIELDS:
        expected = _RECORD_ENVELOPE_KEYS | _CORE_RECORD_FIELDS[record_type]
        if set(row) != expected:
            raise ValueError(f"{record_type} record has missing/unknown fields {location}")
    elif record_type == "event":
        if not (_RECORD_ENVELOPE_KEYS | {"event"}) <= set(row):
            raise ValueError(f"event record has missing fields {location}")
        if set(row) & _EVENT_FORBIDDEN_FIELDS:
            raise ValueError(f"event record uses reserved semantic aliases {location}")
        event = row.get("event")
        if type(event) is not str or _EVENT_NAME.fullmatch(event) is None:
            raise ValueError(f"event record name is invalid {location}")
    else:
        raise ValueError(f"unknown replay record type {location}: {record_type!r}")
    if record_type in {"imu", "race_status"}:
        received = row.get("received_monotonic_s")
        if received is not None and (
            type(received) not in {int, float}
            or not math.isfinite(received)
            or received < 0.0
        ):
            raise ValueError(f"sensor receive time is invalid {location}")
        payload_name = "imu" if record_type == "imu" else "race_status"
        if row.get(payload_name) is not None and type(row[payload_name]) is not dict:
            raise ValueError(f"{payload_name} payload is invalid {location}")
    if record_type == "command":
        if row.get("kind") not in {"generated", "sent"}:
            raise ValueError(f"command kind is invalid {location}")
        monotonic = row.get("monotonic_s")
        if monotonic is not None and (
            type(monotonic) not in {int, float}
            or not math.isfinite(monotonic)
            or monotonic < 0.0
        ):
            raise ValueError(f"command monotonic time is invalid {location}")
        token = row.get("frame_token")
        if token is not None and (
            type(token) is not list
            or len(token) != 3
            or any(type(value) is not int or value < 0 for value in token)
        ):
            raise ValueError(f"command frame token is invalid {location}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("JSON evidence mapping keys must be exact strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON evidence is forbidden")
        return value
    raise TypeError(f"unsupported JSON evidence type: {type(value).__name__}")


def _frame_hash(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    descriptor = canonical_json(
        {"shape": list(contiguous.shape), "dtype": contiguous.dtype.str}
    ).encode("utf-8")
    return sha256_bytes(descriptor + b"\0" + contiguous.tobytes(order="C"))


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(_json_safe(value)) + "\n").encode("utf-8")


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    maximum_bytes: Optional[int] = None,
) -> None:
    encoded = _json_bytes(value)
    if maximum_bytes is not None and len(encoded) > maximum_bytes:
        raise ValueError(f"{path.name} exceeds replay resource limit")
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _create_secure_directory_tree(path: Path | str) -> Path:
    """Create a new directory leaf without traversing existing indirection."""

    lexical = Path(path)
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    lexical = Path(os.path.abspath(lexical))
    try:
        lexical.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"replay session is immutable/existing: {lexical}")
    missing: list[str] = []
    ancestor = lexical
    while True:
        try:
            ancestor.lstat()
            break
        except FileNotFoundError:
            missing.append(ancestor.name)
            parent = ancestor.parent
            if parent == ancestor:
                raise ValueError("replay path has no existing secure ancestor")
            ancestor = parent
    secure_directory(ancestor)
    current = ancestor
    for component in reversed(missing):
        current = current / component
        try:
            os.mkdir(current)
        except FileExistsError as exc:
            raise ValueError("replay path changed during directory creation") from exc
        secure_directory(current)
    return secure_directory(lexical)


class ReplayBundleWriter:
    """Append synchronized records, then seal the session exactly once."""

    def __init__(
        self,
        path: Path | str,
        *,
        session_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        repo_root: Optional[Path | str] = None,
        require_private: bool = True,
    ) -> None:
        lexical_path = Path(path)
        if not lexical_path.is_absolute():
            lexical_path = Path.cwd() / lexical_path
        lexical_path = Path(os.path.abspath(lexical_path))
        existing_parent = lexical_path.parent
        while not existing_parent.exists():
            if existing_parent.parent == existing_parent:
                raise ValueError("replay path has no existing secure parent")
            existing_parent = existing_parent.parent
        secure_directory(existing_parent)
        self.path = lexical_path
        if require_private:
            private_path_guard(
                self.path,
                secure_directory(repo_root) if repo_root is not None else None,
            )
        if session_id is not None and (
            type(session_id) is not str or not session_id.strip()
        ):
            raise ValueError("session_id must be an exact non-empty string")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        self.session_id = session_id or uuid.uuid4().hex
        self._metadata = _json_safe({} if metadata is None else metadata)
        self.path = _create_secure_directory_tree(self.path)
        self.frames_path = self.path / "frames"
        self.records_path = self.path / "records.jsonl"
        self.manifest_path = self.path / "manifest.json"
        self._records: Any = None
        try:
            if require_private:
                private_path_guard(
                    self.path,
                    secure_directory(repo_root) if repo_root is not None else None,
                )
            self.frames_path.mkdir()
            self._started_at = _utc_now()
            self._records = self.records_path.open("xb")
            self._lock = threading.RLock()
            self._closed = False
            self._failure_reason: Optional[str] = None
            self._record_count = 0
            self._record_sequence = 0
            self._frame_tokens: set[tuple[int, int, int]] = set()
            self._decoded_frame_tokens: set[tuple[int, int, int]] = set()
            self._frame_label_keys: set[tuple[int, int]] = set()
            self._decoded_frame_label_keys: set[tuple[int, int]] = set()
            self._frame_blobs: set[str] = set()
            self._records_bytes = 0
            self._frame_blob_file_bytes = 0
            self._manifest = {
                "schema": BUNDLE_SCHEMA,
                "record_schema": RECORD_SCHEMA,
                "session_id": self.session_id,
                "started_at": self._started_at,
                "finished_at": None,
                "complete": False,
                "private": True,
                "recording_notice": RECORDING_NOTICE,
                "metadata": self._metadata,
                "record_count": 0,
                "frame_record_count": 0,
                "decoded_frame_record_count": 0,
                "unique_frame_blob_count": 0,
            }
            # Reserve room for completion/abort fields so a valid initial
            # capture can always publish durable terminal state.
            _atomic_json(
                self.manifest_path,
                self._manifest,
                maximum_bytes=(
                    MAX_REPLAY_MANIFEST_BYTES
                    - _MANIFEST_FINALIZATION_RESERVE_BYTES
                ),
            )
        except Exception:
            if self._records is not None:
                try:
                    self._records.close()
                except (OSError, ValueError):
                    pass
            for owned_file in (self.manifest_path, self.records_path):
                try:
                    owned_file.unlink(missing_ok=True)
                except OSError:
                    pass
            for owned_directory in (self.frames_path, self.path):
                try:
                    owned_directory.rmdir()
                except OSError:
                    pass
            raise

    def _latch_failure(self, reason: str) -> None:
        with self._lock:
            if not self._closed and self._failure_reason is None:
                self._failure_reason = reason

    @property
    def closed(self) -> bool:
        return self._closed

    def append(self, record_type: str, **fields: Any) -> int:
        try:
            if type(record_type) is not str or not record_type.strip():
                raise ValueError("record_type must be non-empty")
            reserved = {
                "schema",
                "session_id",
                "sequence",
                "type",
                "capture_wall_time_ns",
            }
            collisions = reserved & set(fields)
            if collisions:
                raise ValueError(
                    "record fields cannot override trusted envelope fields: "
                    + ", ".join(sorted(collisions))
                )
            safe_fields = _json_safe(fields)
        except Exception as exc:
            self._latch_failure(f"{type(exc).__name__}: {exc}")
            raise
        with self._lock:
            if self._closed:
                raise RuntimeError("replay bundle is sealed")
            if self._failure_reason is not None:
                raise RuntimeError("replay bundle has a latched write failure")
            sequence = self._record_sequence
            row = {
                "schema": RECORD_SCHEMA,
                "session_id": self.session_id,
                "sequence": sequence,
                "type": record_type,
                "capture_wall_time_ns": time.time_ns(),
                **safe_fields,
            }
            try:
                _validate_record_shape(row, location="during append")
            except Exception as exc:
                self._failure_reason = f"{type(exc).__name__}: {exc}"
                raise
            try:
                encoded = (canonical_json(row) + "\n").encode("utf-8")
                if self._record_count >= MAX_REPLAY_RECORD_COUNT:
                    raise ValueError("replay record count exceeds format limit")
                if len(encoded) > MAX_REPLAY_RECORD_LINE_BYTES:
                    raise ValueError("replay record line exceeds format limit")
                if self._records_bytes + len(encoded) > MAX_REPLAY_RECORDS_BYTES:
                    raise ValueError("records.jsonl exceeds format limit")
                self._records.write(encoded)
            except Exception as exc:
                self._failure_reason = f"{type(exc).__name__}: {exc}"
                raise
            self._records_bytes += len(encoded)
            self._record_sequence += 1
            self._record_count += 1
            return sequence

    def record_imu(
        self,
        imu: Any,
        *,
        estimator: Optional[Any] = None,
        received_monotonic_s: Optional[float] = None,
        received_sample: Optional[Any] = None,
    ) -> int:
        try:
            if received_sample is not None:
                if type(received_sample) is not ReceivedIMUSampleV1:
                    received_sample = ReceivedIMUSampleV1.from_primitive(
                        received_sample
                    )
                if _json_safe(imu) != received_sample.to_primitive()["imu"]:
                    raise ValueError(
                        "received IMU envelope differs from the core IMU payload"
                    )
        except Exception as exc:
            self._latch_failure(f"{type(exc).__name__}: {exc}")
            raise
        sequence = self.append(
            "imu",
            received_monotonic_s=received_monotonic_s,
            imu=imu,
            estimator=estimator,
        )
        if received_sample is not None:
            self.append(
                "event",
                event="received_imu",
                observation=received_sample.to_primitive(),
                linked_imu_record_sequence=sequence,
            )
        return sequence

    def record_mavlink_ingress(self, ingress: Any) -> int:
        try:
            if type(ingress) is not MavlinkIngressV1:
                ingress = MavlinkIngressV1.from_primitive(ingress)
        except Exception as exc:
            self._latch_failure(f"{type(exc).__name__}: {exc}")
            raise
        return self.append(
            "event",
            event="mavlink_ingress",
            observation=ingress.to_primitive(),
        )

    def record_race(self, race_status: Any, *, received_monotonic_s: Optional[float] = None) -> int:
        return self.append(
            "race_status",
            received_monotonic_s=received_monotonic_s,
            race_status=race_status,
        )

    def record_command(
        self,
        kind: str,
        command: Any,
        *,
        monotonic_s: Optional[float] = None,
        frame_token: Optional[Sequence[int]] = None,
    ) -> int:
        if kind not in {"generated", "sent"}:
            raise ValueError("command kind must be generated or sent")
        return self.append(
            "command",
            kind=kind,
            monotonic_s=monotonic_s,
            frame_token=list(frame_token) if frame_token is not None else None,
            command=command,
        )

    def record_event(self, event: str, **fields: Any) -> int:
        return self.append("event", event=event, **fields)

    def _persist_frame_blob(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("decoded frame must be an HxWx3 uint8 array")
        height, width, _channels = array.shape
        if (
            height > MAX_REPLAY_FRAME_HEIGHT
            or width > MAX_REPLAY_FRAME_WIDTH
            or height * width > MAX_REPLAY_FRAME_PIXELS
            or array.nbytes > MAX_REPLAY_FRAME_DECODED_BYTES
        ):
            raise ValueError("decoded frame exceeds replay resource limits")
        digest = _frame_hash(array)
        blob = self.frames_path / f"{digest}.npy"
        if not blob.exists():
            if len(self._frame_blobs) >= MAX_REPLAY_FRAME_BLOB_COUNT:
                raise ValueError("replay frame blob count exceeds format limit")
            fd, raw_temp = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=self.frames_path
            )
            os.close(fd)
            temp = Path(raw_temp)
            try:
                with temp.open("wb") as handle:
                    np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                blob_size = temp.stat().st_size
                if blob_size > MAX_REPLAY_FRAME_BLOB_BYTES:
                    raise ValueError("replay frame blob exceeds format limit")
                if (
                    self._frame_blob_file_bytes + blob_size
                    > MAX_REPLAY_SESSION_BLOB_BYTES
                ):
                    raise ValueError("replay frame blobs exceed session limit")
                os.replace(temp, blob)
                self._frame_blob_file_bytes += blob_size
            finally:
                if temp.exists():
                    temp.unlink()
        self._frame_blobs.add(digest)
        return array, digest

    def capture_decoded_frame(
        self,
        image: np.ndarray,
        *,
        generation: int,
        frame_id: int,
        sim_time_ns: int,
        received_monotonic_s: float,
        frame_timing: Optional[Any] = None,
    ) -> Optional[int]:
        """Persist every frame published by the duplicate-suppressing receiver."""

        try:
            if (
                type(received_monotonic_s) not in {int, float}
                or not math.isfinite(received_monotonic_s)
                or received_monotonic_s < 0
            ):
                raise ValueError(
                    "received_monotonic_s must be finite and non-negative"
                )
            token = (generation, frame_id, sim_time_ns)
            if any(type(value) is not int or value < 0 for value in token):
                raise ValueError(
                    "frame generation/id/sim time must be non-negative exact integers"
                )
            if frame_timing is not None:
                if type(frame_timing) is not FrameTimingV1:
                    frame_timing = FrameTimingV1.from_primitive(frame_timing)
                if (
                    frame_timing.identity.generation != generation
                    or frame_timing.identity.frame_id != frame_id
                    or frame_timing.camera_source_time_ns != sim_time_ns
                ):
                    raise ValueError(
                        "frame timing identity differs from decoded frame"
                    )
        except Exception as exc:
            self._latch_failure(f"{type(exc).__name__}: {exc}")
            raise
        with self._lock:
            if token in self._decoded_frame_tokens:
                return None
            if self._closed:
                raise RuntimeError("replay bundle is sealed")
            label_key = token[:2]
            if label_key in self._decoded_frame_label_keys:
                error = ValueError(
                    "decoded frame generation/frame_id must be unique for labels"
                )
                self._latch_failure(f"{type(error).__name__}: {error}")
                raise error
            try:
                array, digest = self._persist_frame_blob(image)
                self._decoded_frame_tokens.add(token)
                self._decoded_frame_label_keys.add(label_key)
                sequence = self.append(
                    "decoded_frame",
                    generation=token[0],
                    frame_id=token[1],
                    sim_time_ns=token[2],
                    received_monotonic_s=float(received_monotonic_s),
                    frame_blob=f"frames/{digest}.npy",
                    frame_hash=digest,
                    image_shape=list(array.shape),
                    image_dtype=array.dtype.str,
                )
                if frame_timing is not None:
                    self.append(
                        "event",
                        event="camera_frame_timing",
                        observation=frame_timing.to_primitive(),
                        linked_decoded_frame_record_sequence=sequence,
                    )
                return sequence
            except Exception as exc:
                self._latch_failure(f"{type(exc).__name__}: {exc}")
                raise

    def capture_frame(
        self,
        image: np.ndarray,
        *,
        generation: int,
        frame_id: int,
        sim_time_ns: int,
        received_monotonic_s: float,
        detector_latency_ms: Optional[float],
        detections: Sequence[Any],
        tracker: Optional[Any],
        imu: Optional[Any],
        estimator: Optional[Any],
        race_status: Optional[Any],
        generated_command: Optional[Any],
        sent_command: Optional[Any],
        phase: Optional[str] = None,
    ) -> Optional[int]:
        """Persist one decoded frame and its same-sample autonomy state.

        Repeated ``(generation, frame_id, sim_time_ns)`` deliveries are ignored.
        Identical pixel arrays across distinct frame tokens share one blob.
        """

        if (
            type(received_monotonic_s) not in {int, float}
            or not math.isfinite(received_monotonic_s)
            or received_monotonic_s < 0
        ):
            raise ValueError("received_monotonic_s must be finite and non-negative")
        token = (generation, frame_id, sim_time_ns)
        if any(type(value) is not int or value < 0 for value in token):
            raise ValueError("frame generation/id/sim time must be non-negative exact integers")
        with self._lock:
            if token in self._frame_tokens:
                return None
            if self._closed:
                raise RuntimeError("replay bundle is sealed")
            label_key = token[:2]
            if label_key in self._frame_label_keys:
                error = ValueError(
                    "frame generation/frame_id must be unique for labels"
                )
                self._latch_failure(f"{type(error).__name__}: {error}")
                raise error
            try:
                array, digest = self._persist_frame_blob(image)
                self._frame_tokens.add(token)
                self._frame_label_keys.add(label_key)
                return self.append(
                    "frame",
                    generation=token[0],
                    frame_id=token[1],
                    sim_time_ns=token[2],
                    received_monotonic_s=float(received_monotonic_s),
                    frame_blob=f"frames/{digest}.npy",
                    frame_hash=digest,
                    image_shape=list(array.shape),
                    image_dtype=array.dtype.str,
                    detector_latency_ms=detector_latency_ms,
                    detections=detections,
                    tracker=tracker,
                    imu=imu,
                    estimator=estimator,
                    race_status=race_status,
                    generated_command=generated_command,
                    sent_command=sent_command,
                    phase=phase,
                )
            except Exception as exc:
                self._latch_failure(f"{type(exc).__name__}: {exc}")
                raise

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._records.flush()
                os.fsync(self._records.fileno())

    def close(self, *, outcome: Optional[Mapping[str, Any]] = None) -> str:
        """Seal the bundle and return its dataset hash."""

        try:
            if outcome is not None and not isinstance(outcome, Mapping):
                raise TypeError("outcome must be a mapping")
            safe_outcome = _json_safe({} if outcome is None else outcome)
        except Exception as exc:
            self._latch_failure(f"invalid outcome: {type(exc).__name__}: {exc}")
            try:
                self.abort(self._failure_reason or "invalid outcome")
            except Exception:
                # Preserve the validation error; abort has already exhausted
                # its best-effort close/durable-incomplete path.
                pass
            raise
        with self._lock:
            if self._closed:
                if self._manifest.get("complete") is not True:
                    raise RuntimeError("replay bundle is incomplete")
                return str(self._manifest["dataset_hash"])
            try:
                if self._failure_reason is not None:
                    raise RuntimeError(
                        f"cannot seal replay bundle: {self._failure_reason}"
                    )
                self.flush()
                blob_file_hashes = {
                    digest: sha256_file(self.frames_path / f"{digest}.npy")
                    for digest in sorted(self._frame_blobs)
                }
                records_hash = sha256_file(self.records_path)
                frame_count = len(self._frame_tokens)
                decoded_frame_count = len(self._decoded_frame_tokens)
                integrity = {
                    "records_sha256": records_hash,
                    "frame_blob_file_sha256": blob_file_hashes,
                }
                finished_at = _utc_now()
                dataset_hash = json_hash(
                    {
                        "schema": BUNDLE_SCHEMA,
                        "session_id": self.session_id,
                        "started_at": self._started_at,
                        "finished_at": finished_at,
                        "metadata": self._metadata,
                        "outcome": safe_outcome,
                        "records_sha256": records_hash,
                        "frame_blob_file_sha256": blob_file_hashes,
                    }
                )
                final_manifest = {
                    **self._manifest,
                    **{
                        "finished_at": finished_at,
                        "complete": True,
                        "record_count": self._record_count,
                        "frame_record_count": frame_count,
                        "decoded_frame_record_count": decoded_frame_count,
                        "unique_frame_blob_count": len(self._frame_blobs),
                        "integrity": integrity,
                        "dataset_hash": dataset_hash,
                        "outcome": safe_outcome,
                    },
                }
                if len(_json_bytes(final_manifest)) > MAX_REPLAY_MANIFEST_BYTES:
                    raise ValueError(
                        "final replay manifest exceeds replay resource limit"
                    )
                self._records.close()
                _atomic_json(
                    self.manifest_path,
                    final_manifest,
                    maximum_bytes=MAX_REPLAY_MANIFEST_BYTES,
                )
            except Exception as exc:
                reason = self._failure_reason or (
                    f"finalization failed: {type(exc).__name__}: {exc}"
                )
                self._failure_reason = reason
                try:
                    self._abort_locked(reason)
                except Exception:
                    # `_abort_locked` always closes/marks the writer before
                    # reporting its own persistence failure. Preserve the
                    # original precommit/finalization exception here.
                    pass
                raise
            self._manifest = final_manifest
            self._closed = True
            return dataset_hash

    def _abort_locked(self, reason: str) -> None:
        """Exhaust cleanup and terminal-incomplete publication under the lock."""

        if self._closed:
            return
        failures: list[BaseException] = []
        bounded_reason = str(reason)[:4096] or "unspecified replay abort"
        if not self._records.closed:
            try:
                self._records.flush()
                os.fsync(self._records.fileno())
            except (OSError, ValueError) as exc:
                failures.append(exc)
            try:
                self._records.close()
            except (OSError, ValueError) as exc:
                failures.append(exc)
        self._manifest.update(
            {
                "finished_at": _utc_now(),
                "complete": False,
                "abort_reason": bounded_reason,
                "record_count": self._record_count,
                "frame_record_count": len(self._frame_tokens),
                "decoded_frame_record_count": len(self._decoded_frame_tokens),
                "unique_frame_blob_count": len(self._frame_blobs),
            }
        )
        try:
            _atomic_json(
                self.manifest_path,
                self._manifest,
                maximum_bytes=MAX_REPLAY_MANIFEST_BYTES,
            )
        except Exception as exc:
            failures.append(exc)
        finally:
            self._closed = True
        if failures:
            raise RuntimeError("replay abort cleanup was incomplete") from failures[0]

    def abort(self, reason: str) -> None:
        """Close an incomplete session without claiming it is replayable."""

        with self._lock:
            self._abort_locked(reason)

    def mark_invalid(self, reason: str) -> None:
        """Permanently invalidate a bundle, including a concurrent finalizer."""

        _atomic_json(
            self.path / "capture-invalid.json",
            {
                "schema": "aigp-vq2-replay-invalid/1",
                "reason": str(reason)[:4096],
                "invalidated_at": _utc_now(),
            },
        )
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._closed:
                self._manifest["complete"] = False
                self._manifest["invalidated"] = True
                self._manifest["abort_reason"] = str(reason)[:4096]
                _atomic_json(
                    self.manifest_path,
                    self._manifest,
                    maximum_bytes=MAX_REPLAY_MANIFEST_BYTES,
                )
        finally:
            self._lock.release()

    def __enter__(self) -> "ReplayBundleWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc is None:
            self.close()
        else:
            self.abort(f"{exc_type.__name__}: {exc}")


@dataclasses.dataclass(frozen=True)
class AsyncCaptureStats:
    enqueued: int
    written: int
    dropped: int
    duplicate_frame_tokens: int
    writer_errors: int
    queue_high_watermark: int
    decoded_frames_enqueued: int
    decoded_frames_written: int
    decoded_frames_dropped: int
    complete: bool
    dataset_hash: Optional[str]
    failure_reason: Optional[str]


class AsyncReplayRecorder:
    """Bounded background facade that keeps all bundle I/O off flight threads.

    Vision images are defensively copied and frozen at the call boundary;
    NumPy read-only views are not ownership proof because a writable base can
    still mutate them. Every non-image value is
    strict-JSON-normalized at the call boundary so later mutation of live
    telemetry objects cannot alter synchronized evidence.  The worker performs
    serialization and fsync.  A full queue never blocks control: it latches an
    incomplete capture and drops the record.
    """

    _STOP = object()

    def __init__(
        self,
        writer: ReplayBundleWriter,
        *,
        max_queue_records: int = 256,
    ) -> None:
        if type(max_queue_records) is not int or max_queue_records < 1:
            raise ValueError("max_queue_records must be a positive exact integer")
        self.writer = writer
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max_queue_records)
        self._lock = threading.RLock()
        self._frame_tokens: set[tuple[int, int, int]] = set()
        self._decoded_frame_tokens: set[tuple[int, int, int]] = set()
        self._frame_label_keys: set[tuple[int, int]] = set()
        self._decoded_frame_label_keys: set[tuple[int, int]] = set()
        self._enqueued = 0
        self._written = 0
        self._dropped = 0
        self._duplicate_frame_tokens = 0
        self._writer_errors = 0
        self._high_watermark = 0
        self._decoded_enqueued = 0
        self._decoded_written = 0
        self._decoded_dropped = 0
        self._failure_reason: Optional[str] = None
        self._outcome: Mapping[str, Any] = {}
        self._dataset_hash: Optional[str] = None
        self._closing = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="aigp-replay-writer",
            daemon=True,
        )
        self._thread.start()

    def _enqueue(self, operation: str, *args: Any, **kwargs: Any) -> bool:
        # Avoid copying a large mutable frame or traversing invalid telemetry
        # after the evidence has already been sealed.
        with self._lock:
            if self._closed:
                return False
            if self._closing:
                self._dropped += 1
                self._failure_reason = self._failure_reason or "record queued after close"
                return False
        try:
            def snapshot(value: Any) -> Any:
                if isinstance(value, np.ndarray):
                    frozen = np.array(value, copy=True, order="C")
                    frozen.setflags(write=False)
                    return frozen
                return _json_safe(value)

            safe_args = tuple(snapshot(value) for value in args)
            safe_kwargs = {
                name: snapshot(value) for name, value in kwargs.items()
            }
        except Exception as exc:
            with self._lock:
                # A seal completed while snapshotting is immutable: a late
                # producer cannot retroactively make its stats incomplete.
                if not self._closed:
                    self._dropped += 1
                    self._failure_reason = self._failure_reason or (
                        f"invalid capture snapshot: {type(exc).__name__}: {exc}"
                    )
            return False
        with self._lock:
            if self._closed:
                # Once sealing is complete, a late producer is rejected
                # idempotently.  It cannot retroactively mutate the evidence
                # or invalidate a bundle whose completeness was already fixed.
                return False
            if self._closing:
                self._dropped += 1
                self._failure_reason = self._failure_reason or "record queued after close"
                return False
            # Keep the close-state check and non-blocking put in one critical
            # section.  Otherwise close can set ``_closing`` and the worker
            # can exit between the check and put, orphaning an accepted item.
            try:
                self._queue.put_nowait((operation, safe_args, safe_kwargs))
            except queue.Full:
                self._dropped += 1
                self._failure_reason = self._failure_reason or "bounded capture queue overflow"
                return False
            self._enqueued += 1
            self._high_watermark = max(self._high_watermark, self._queue.qsize())
        return True

    def record_imu(
        self,
        imu: Any,
        *,
        estimator: Optional[Any] = None,
        received_monotonic_s: Optional[float] = None,
        received_sample: Optional[Any] = None,
    ) -> bool:
        return self._enqueue(
            "record_imu",
            imu,
            estimator=estimator,
            received_monotonic_s=received_monotonic_s,
            received_sample=received_sample,
        )

    def record_mavlink_ingress(self, ingress: Any) -> bool:
        return self._enqueue("record_mavlink_ingress", ingress)

    def record_race(self, race_status: Any, *, received_monotonic_s: Optional[float] = None) -> bool:
        return self._enqueue(
            "record_race", race_status, received_monotonic_s=received_monotonic_s
        )

    def record_command(self, kind: str, command: Any, **fields: Any) -> bool:
        return self._enqueue("record_command", kind, command, **fields)

    def record_event(self, event: str, **fields: Any) -> bool:
        return self._enqueue("record_event", event, **fields)

    def fail(self, reason: str) -> bool:
        """Latch an external completeness failure without blocking a caller."""

        with self._lock:
            if self._closed:
                return False
            if type(reason) is not str or not reason:
                reason = "unspecified external capture failure"
            self._failure_reason = self._failure_reason or reason
            return True

    def _reject_capture(self, reason: str, *, decoded: bool = False) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._dropped += 1
            if decoded:
                self._decoded_dropped += 1
            self._failure_reason = self._failure_reason or reason
        return False

    def capture_frame(self, image: np.ndarray, **fields: Any) -> bool:
        try:
            token = tuple(fields[name] for name in ("generation", "frame_id", "sim_time_ns"))
            received = fields["received_monotonic_s"]
        except (KeyError, TypeError):
            return self._reject_capture("invalid frame callback fields")
        if any(type(value) is not int or value < 0 for value in token) or (
            type(received) not in {int, float}
            or not math.isfinite(received)
            or received < 0.0
        ):
            return self._reject_capture("invalid exact frame callback token/time")
        with self._lock:
            if self._closed:
                return False
            if self._closing:
                self._dropped += 1
                self._failure_reason = self._failure_reason or "record queued after close"
                return False
            if token in self._frame_tokens:
                self._duplicate_frame_tokens += 1
                return False
            label_key = (token[0], token[1])
            if label_key in self._frame_label_keys:
                self._dropped += 1
                self._failure_reason = self._failure_reason or (
                    "ambiguous frame generation/frame_id label identity"
                )
                return False
            self._frame_tokens.add(token)
            self._frame_label_keys.add(label_key)
            accepted = self._enqueue("capture_frame", image, **fields)
        if not accepted:
            # Keep the token remembered: retrying later would no longer be
            # synchronized to the same estimator/race/controller sample.
            return False
        return True

    def capture_decoded_snapshot(self, snapshot: Any) -> bool:
        """Non-blocking callback suitable for ``VQ2VisionThread``."""

        try:
            token = (
                snapshot.generation,
                snapshot.frame_id,
                snapshot.sim_time_ns,
            )
            received = snapshot.received_monotonic_s
            image = snapshot.camera_frame.image
            timing = getattr(snapshot, "timing", None)
        except (AttributeError, TypeError):
            return self._reject_capture(
                "invalid decoded-frame callback fields", decoded=True
            )
        if any(type(value) is not int or value < 0 for value in token) or (
            type(received) not in {int, float}
            or not math.isfinite(received)
            or received < 0.0
        ):
            return self._reject_capture(
                "invalid exact decoded-frame callback token/time", decoded=True
            )
        if timing is not None:
            if type(timing) is not FrameTimingV1 or (
                timing.identity.generation != token[0]
                or timing.identity.frame_id != token[1]
                or timing.camera_source_time_ns != token[2]
            ):
                return self._reject_capture(
                    "invalid exact decoded-frame timing identity", decoded=True
                )
        with self._lock:
            if self._closed:
                return False
            if self._closing:
                self._dropped += 1
                self._decoded_dropped += 1
                self._failure_reason = self._failure_reason or "record queued after close"
                return False
            if token in self._decoded_frame_tokens:
                self._duplicate_frame_tokens += 1
                return False
            label_key = (token[0], token[1])
            if label_key in self._decoded_frame_label_keys:
                self._dropped += 1
                self._decoded_dropped += 1
                self._failure_reason = self._failure_reason or (
                    "ambiguous decoded-frame generation/frame_id label identity"
                )
                return False
            self._decoded_frame_tokens.add(token)
            self._decoded_frame_label_keys.add(label_key)
            accepted = self._enqueue(
                "capture_decoded_frame",
                image,
                generation=token[0],
                frame_id=token[1],
                sim_time_ns=token[2],
                received_monotonic_s=received,
                frame_timing=timing,
            )
            if accepted:
                self._decoded_enqueued += 1
            else:
                self._decoded_dropped += 1
        return accepted

    def _worker(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                with self._lock:
                    if self._closing:
                        break
                continue
            try:
                if item is self._STOP:
                    break
                operation, args, kwargs = item
                try:
                    getattr(self.writer, operation)(*args, **kwargs)
                except Exception as exc:
                    with self._lock:
                        self._writer_errors += 1
                        self._failure_reason = self._failure_reason or (
                            f"{type(exc).__name__}: {exc}"
                        )
                else:
                    with self._lock:
                        self._written += 1
                        if operation == "capture_decoded_frame":
                            self._decoded_written += 1
            finally:
                self._queue.task_done()
        with self._lock:
            failure = self._failure_reason
            outcome = self._outcome
        try:
            if failure is None:
                candidate_hash = self.writer.close(outcome=outcome)
                with self._lock:
                    late_failure = self._failure_reason
                    if late_failure is None:
                        self._dataset_hash = candidate_hash
                        self._closed = True
                if late_failure is not None:
                    self.writer.mark_invalid(late_failure)
            else:
                self.writer.abort(failure)
        except Exception as exc:
            with self._lock:
                self._writer_errors += 1
                self._failure_reason = self._failure_reason or (
                    f"finalize {type(exc).__name__}: {exc}"
                )
        finally:
            with self._lock:
                self._closed = True

    def close(
        self,
        *,
        outcome: Optional[Mapping[str, Any]] = None,
        expected_decoded_frames: Optional[int] = None,
        timeout_s: float = 30.0,
    ) -> AsyncCaptureStats:
        if (
            type(timeout_s) not in {int, float}
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")
        if expected_decoded_frames is not None and (
            type(expected_decoded_frames) is not int
            or expected_decoded_frames < 0
        ):
            raise ValueError(
                "expected_decoded_frames must be a non-negative exact integer"
            )
        outcome_failure: Optional[str] = None
        try:
            if outcome is not None and not isinstance(outcome, Mapping):
                raise TypeError("outcome must be a mapping")
            safe_outcome = _json_safe({} if outcome is None else outcome)
        except Exception as exc:
            safe_outcome = {}
            outcome_failure = (
                f"invalid close outcome: {type(exc).__name__}: {exc}"
            )
        with self._lock:
            if self._closed:
                already_closed = True
            else:
                already_closed = False
            if not already_closed and outcome_failure is not None:
                self._failure_reason = self._failure_reason or outcome_failure
            if (
                not already_closed
                and
                expected_decoded_frames is not None
                and self._decoded_enqueued != expected_decoded_frames
            ):
                self._failure_reason = self._failure_reason or (
                    "decoded-frame callback count mismatch: "
                    f"queued={self._decoded_enqueued} expected={expected_decoded_frames}"
                )
            if already_closed:
                first_close = False
            elif not self._closing:
                # Outcome validation happened before publishing the closing
                # state, so the worker cannot race ahead and seal defaults.
                self._outcome = safe_outcome
                self._closing = True
                first_close = True
            else:
                first_close = False
        if already_closed:
            return self.stats()
        if first_close:
            # Wake an idle worker immediately.  The put remains non-blocking:
            # when the queue is full the worker is already runnable and will
            # observe ``_closing`` after it drains accepted work.
            try:
                self._queue.put_nowait(self._STOP)
            except queue.Full:
                pass
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            with self._lock:
                timeout_reason = "capture finalization timeout"
                if self._failure_reason is None:
                    self._failure_reason = timeout_reason
                elif timeout_reason not in self._failure_reason:
                    self._failure_reason = f"{self._failure_reason}; {timeout_reason}"
                failure_reason = self._failure_reason
            # This durable marker is written by the cleanup caller, never a
            # flight/control callback.  The reader rejects it even if a slow
            # concurrent writer later reaches its nominal complete commit.
            self.writer.mark_invalid(failure_reason)
        return self.stats()

    def stats(self) -> AsyncCaptureStats:
        with self._lock:
            complete = bool(
                self._closed
                and self._failure_reason is None
                and self._dropped == 0
                and self._writer_errors == 0
                and self._written == self._enqueued
                and self._decoded_written == self._decoded_enqueued
                and self._queue.empty()
                and self._dataset_hash
            )
            return AsyncCaptureStats(
                enqueued=self._enqueued,
                written=self._written,
                dropped=self._dropped,
                duplicate_frame_tokens=self._duplicate_frame_tokens,
                writer_errors=self._writer_errors,
                queue_high_watermark=self._high_watermark,
                decoded_frames_enqueued=self._decoded_enqueued,
                decoded_frames_written=self._decoded_written,
                decoded_frames_dropped=self._decoded_dropped,
                complete=complete,
                dataset_hash=self._dataset_hash if complete else None,
                failure_reason=self._failure_reason,
            )


class ReplayBundleReader:
    def __init__(self, path: Path | str, *, require_complete: bool = True) -> None:
        self.path = secure_directory(path)
        if (self.path / "capture-invalid.json").exists():
            raise ValueError("replay bundle is permanently invalidated")
        self._manifest_bytes = _read_bounded_secure_file(
            self.path / "manifest.json",
            maximum_bytes=MAX_REPLAY_MANIFEST_BYTES,
            label="replay manifest",
        )
        try:
            manifest_text = self._manifest_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("replay manifest must be UTF-8") from exc
        self.manifest = strict_json_loads(manifest_text)
        self._validate_manifest()
        if require_complete and self.manifest["complete"] is not True:
            raise ValueError("replay bundle is incomplete")

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        base = {
            "schema",
            "record_schema",
            "session_id",
            "started_at",
            "finished_at",
            "complete",
            "private",
            "recording_notice",
            "metadata",
            "record_count",
            "frame_record_count",
            "decoded_frame_record_count",
            "unique_frame_blob_count",
        }
        if type(manifest) is not dict or not base <= set(manifest):
            raise ValueError("replay manifest has missing/invalid base fields")
        if manifest["complete"] is True:
            expected = base | {"integrity", "dataset_hash", "outcome"}
        elif manifest["complete"] is False:
            expected = base | ({"abort_reason"} if "abort_reason" in manifest else set())
        else:
            raise ValueError("replay manifest complete must be an exact bool")
        if set(manifest) != expected:
            raise ValueError("replay manifest has missing or unknown fields")
        if (
            manifest["schema"] != BUNDLE_SCHEMA
            or manifest["record_schema"] != RECORD_SCHEMA
            or type(manifest["session_id"]) is not str
            or not manifest["session_id"].strip()
            or type(manifest["started_at"]) is not str
            or not manifest["started_at"]
            or (
                manifest["finished_at"] is not None
                and type(manifest["finished_at"]) is not str
            )
            or manifest["private"] is not True
            or manifest["recording_notice"] != RECORDING_NOTICE
            or type(manifest["metadata"]) is not dict
        ):
            raise ValueError("replay manifest provenance fields are invalid")
        for name in (
            "record_count",
            "frame_record_count",
            "decoded_frame_record_count",
            "unique_frame_blob_count",
        ):
            if type(manifest[name]) is not int or manifest[name] < 0:
                raise ValueError("replay manifest counts must be exact non-negative integers")
        if (
            manifest["record_count"] > MAX_REPLAY_RECORD_COUNT
            or manifest["unique_frame_blob_count"] > MAX_REPLAY_FRAME_BLOB_COUNT
            or manifest["frame_record_count"] > manifest["record_count"]
            or manifest["decoded_frame_record_count"] > manifest["record_count"]
            or manifest["unique_frame_blob_count"]
            > manifest["frame_record_count"] + manifest["decoded_frame_record_count"]
        ):
            raise ValueError("replay manifest counts exceed format limits")
        if manifest["complete"] is True:
            integrity = manifest["integrity"]
            if (
                type(integrity) is not dict
                or set(integrity) != {
                    "records_sha256",
                    "frame_blob_file_sha256",
                }
                or type(integrity["frame_blob_file_sha256"]) is not dict
                or type(manifest["outcome"]) is not dict
                or type(manifest["finished_at"]) is not str
                or not manifest["finished_at"]
            ):
                raise ValueError("replay manifest integrity fields are invalid")
            blob_hashes = integrity["frame_blob_file_sha256"]
            if not _is_sha256(integrity["records_sha256"]) or not _is_sha256(
                manifest["dataset_hash"]
            ):
                raise ValueError("replay manifest integrity values must be SHA-256")
            if any(
                not _is_sha256(blob_digest) or not _is_sha256(file_digest)
                for blob_digest, file_digest in blob_hashes.items()
            ):
                raise ValueError("replay manifest frame integrity map is invalid")
            if manifest["unique_frame_blob_count"] != len(blob_hashes):
                raise ValueError("replay manifest unique frame count is inconsistent")
        elif "abort_reason" in manifest:
            if (
                type(manifest["abort_reason"]) is not str
                or not manifest["abort_reason"]
                or type(manifest["finished_at"]) is not str
                or not manifest["finished_at"]
            ):
                raise ValueError("aborted replay manifest is invalid")
        elif manifest["finished_at"] is not None:
            raise ValueError("active incomplete replay cannot have a finish time")

    @property
    def dataset_hash(self) -> str:
        return self.manifest["dataset_hash"]

    @property
    def session_id(self) -> str:
        return self.manifest["session_id"]

    def _parse_records(self, payload: bytes) -> list[Dict[str, Any]]:
        if len(payload) > MAX_REPLAY_RECORDS_BYTES:
            raise ValueError("records.jsonl exceeds replay resource limit")
        expected_sequence = 0
        rows: list[Dict[str, Any]] = []
        for line_number, encoded_line in enumerate(io.BytesIO(payload), start=1):
            if len(encoded_line) > MAX_REPLAY_RECORD_LINE_BYTES:
                raise ValueError(
                    f"record line exceeds replay resource limit on line {line_number}"
                )
            if len(rows) >= MAX_REPLAY_RECORD_COUNT:
                raise ValueError("replay record count exceeds format limit")
            try:
                line = encoded_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"records.jsonl must be UTF-8 on line {line_number}"
                ) from exc
            row = strict_json_loads(line)
            envelope = {
                "schema",
                "session_id",
                "sequence",
                "type",
                "capture_wall_time_ns",
            }
            if type(row) is not dict or not envelope <= set(row):
                raise ValueError(f"invalid record envelope on line {line_number}")
            if row["schema"] != RECORD_SCHEMA:
                raise ValueError(f"invalid record schema on line {line_number}")
            if (
                type(row["session_id"]) is not str
                or not row["session_id"].strip()
                or row["session_id"] != self.session_id
            ):
                raise ValueError(f"session mismatch on line {line_number}")
            if type(row["sequence"]) is not int or row["sequence"] != expected_sequence:
                raise ValueError(f"non-contiguous sequence on line {line_number}")
            if type(row["type"]) is not str or not row["type"].strip():
                raise ValueError(f"invalid record type on line {line_number}")
            if (
                type(row["capture_wall_time_ns"]) is not int
                or row["capture_wall_time_ns"] < 0
            ):
                raise ValueError(f"invalid capture timestamp on line {line_number}")
            _validate_record_shape(row, location=f"on line {line_number}")
            expected_sequence += 1
            rows.append(row)
        return rows

    def records(self) -> Iterator[Dict[str, Any]]:
        # One read produces both the parsed rows and their byte identity.  A
        # caller that needs integrity evidence should use ``verify_and_read``.
        return iter(
                self._parse_records(
                _read_bounded_secure_file(
                    self.path / "records.jsonl",
                    maximum_bytes=MAX_REPLAY_RECORDS_BYTES,
                    label="records.jsonl",
                )
            )
        )

    def frame_records(self) -> Iterator[Dict[str, Any]]:
        return (row for row in self.records() if row.get("type") == "frame")

    @staticmethod
    def _validate_frame_record(record: Mapping[str, Any]) -> tuple[int, int, int]:
        if type(record) is not dict:
            raise ValueError("frame record must be an exact object")
        token = tuple(record.get(name) for name in ("generation", "frame_id", "sim_time_ns"))
        if any(type(value) is not int or value < 0 for value in token):
            raise ValueError("frame token must contain exact non-negative integers")
        received = record.get("received_monotonic_s")
        if (
            type(received) not in {int, float}
            or not math.isfinite(received)
            or received < 0.0
        ):
            raise ValueError("frame receive time must be finite and non-negative")
        shape = record.get("image_shape")
        if (
            type(shape) is not list
            or len(shape) != 3
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[2] != 3
            or record.get("image_dtype") != np.dtype(np.uint8).str
            or shape[0] > MAX_REPLAY_FRAME_HEIGHT
            or shape[1] > MAX_REPLAY_FRAME_WIDTH
            or shape[0] * shape[1] > MAX_REPLAY_FRAME_PIXELS
            or shape[0] * shape[1] * shape[2]
            > MAX_REPLAY_FRAME_DECODED_BYTES
        ):
            raise ValueError("frame metadata must declare an HxWx3 uint8 image")
        return token  # type: ignore[return-value]

    def load_frame(self, record: Mapping[str, Any], *, verify: bool = True) -> np.ndarray:
        self._validate_frame_record(record)
        raw_relative = record.get("frame_blob")
        frame_hash = record.get("frame_hash")
        if type(raw_relative) is not str or not _is_sha256(frame_hash):
            raise ValueError("frame path must be an exact string")
        relative = Path(raw_relative)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 2
            or relative.parts[0] != "frames"
            or relative.suffix != ".npy"
            or raw_relative != f"frames/{frame_hash}.npy"
        ):
            raise ValueError("frame path escapes replay bundle")
        target = self.path / relative
        listed_blobs = self.manifest.get("integrity", {}).get(
            "frame_blob_file_sha256", {}
        )
        expected_file_hash = listed_blobs.get(frame_hash)
        if not _is_sha256(expected_file_hash):
            raise ValueError("frame blob is absent from manifest integrity")
        # Recheck at point of use.  Retaining every verified frame would make
        # memory grow with an entire multi-GB corpus; a post-verification
        # replacement instead fails closed here.
        payload = _read_bounded_secure_file(
            target,
            maximum_bytes=MAX_REPLAY_FRAME_BLOB_BYTES,
            label=f"frame blob {frame_hash}",
        )
        if sha256_bytes(payload) != expected_file_hash:
            raise ValueError(f"frame blob file hash mismatch: {frame_hash}")
        header_shape = _validate_npy_payload_header(payload)
        if list(header_shape) != record["image_shape"]:
            raise ValueError("frame blob header contradicts frame metadata")
        image = np.load(io.BytesIO(payload), allow_pickle=False)
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or list(image.shape) != record["image_shape"]
            or image.dtype.str != record["image_dtype"]
        ):
            raise ValueError("decoded frame contradicts HxWx3 uint8 metadata")
        if verify and _frame_hash(image) != record.get("frame_hash"):
            raise ValueError("decoded frame content hash mismatch")
        image.setflags(write=False)
        return image

    def verify_and_read(
        self, *, verify_frames: bool = True
    ) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
        """Verify and return the exact record snapshot used by scoring."""

        integrity = self.manifest.get("integrity", {})
        records_payload = _read_bounded_secure_file(
            self.path / "records.jsonl",
            maximum_bytes=MAX_REPLAY_RECORDS_BYTES,
            label="records.jsonl",
        )
        records_hash = sha256_bytes(records_payload)
        if records_hash != integrity.get("records_sha256"):
            raise ValueError("records.jsonl hash mismatch")
        records = self._parse_records(records_payload)
        frames = [row for row in records if row.get("type") == "frame"]
        decoded_frames = [row for row in records if row.get("type") == "decoded_frame"]
        for group in (frames, decoded_frames):
            tokens = [self._validate_frame_record(row) for row in group]
            if len(tokens) != len(set(tokens)):
                raise ValueError("duplicate frame token in replay stream")
            label_keys = [(token[0], token[1]) for token in tokens]
            if len(label_keys) != len(set(label_keys)):
                raise ValueError(
                    "generation/frame_id must be unique within each replay stream"
                )
        if len(records) != self.manifest["record_count"]:
            raise ValueError("record count mismatch")
        if len(frames) != self.manifest["frame_record_count"]:
            raise ValueError("frame count mismatch")
        if len(decoded_frames) != self.manifest["decoded_frame_record_count"]:
            raise ValueError("decoded frame count mismatch")
        listed_blobs = integrity.get("frame_blob_file_sha256", {})
        if type(listed_blobs) is not dict:
            raise ValueError("invalid frame blob integrity map")
        frames_dir = secure_directory(self.path / "frames")
        actual_blob_names: set[str] = set()
        try:
            with os.scandir(frames_dir) as entries:
                for entry in entries:
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        raise ValueError("frames directory contains a non-regular entry")
                    if not entry.name.endswith(".npy"):
                        raise ValueError("frames directory contains an unknown file")
                    actual_blob_names.add(entry.name[:-4])
                    if len(actual_blob_names) > MAX_REPLAY_FRAME_BLOB_COUNT:
                        raise ValueError("frame blob count exceeds replay resource limit")
        except OSError as exc:
            raise ValueError("frames directory could not be inspected") from exc
        if actual_blob_names != set(listed_blobs):
            raise ValueError("frame blob set mismatch")
        referenced_blobs = {
            row.get("frame_hash") for row in (*frames, *decoded_frames)
        }
        if referenced_blobs != set(listed_blobs):
            raise ValueError("manifest frame blobs are not exactly referenced")
        total_blob_bytes = 0
        for digest, expected_file_hash in listed_blobs.items():
            blob_path = secure_regular_file(
                self.path / "frames" / f"{digest}.npy"
            )
            blob_size = blob_path.stat().st_size
            if blob_size > MAX_REPLAY_FRAME_BLOB_BYTES:
                raise ValueError(f"frame blob exceeds resource limit: {digest}")
            total_blob_bytes += blob_size
            if total_blob_bytes > MAX_REPLAY_SESSION_BLOB_BYTES:
                raise ValueError("frame blobs exceed replay session resource limit")
            payload = _read_bounded_secure_file(
                blob_path,
                maximum_bytes=MAX_REPLAY_FRAME_BLOB_BYTES,
                label=f"frame blob {digest}",
            )
            if sha256_bytes(payload) != expected_file_hash:
                raise ValueError(f"frame blob file hash mismatch: {digest}")
        if verify_frames:
            for row in (*decoded_frames, *frames):
                self.load_frame(row, verify=True)
        recomputed_dataset_hash = json_hash(
            {
                "schema": BUNDLE_SCHEMA,
                "session_id": self.session_id,
                "started_at": self.manifest["started_at"],
                "finished_at": self.manifest["finished_at"],
                "metadata": self.manifest.get("metadata", {}),
                "outcome": self.manifest["outcome"],
                "records_sha256": records_hash,
                "frame_blob_file_sha256": dict(listed_blobs),
            }
        )
        if recomputed_dataset_hash != self.manifest.get("dataset_hash"):
            raise ValueError("dataset hash mismatch")
        summary = {
            "dataset_hash": self.dataset_hash,
            "records": len(records),
            "frames": len(frames),
            "decoded_frames": len(decoded_frames),
            "unique_frame_blobs": self.manifest["unique_frame_blob_count"],
        }
        return summary, records

    def verify(self, *, verify_frames: bool = True) -> Dict[str, Any]:
        summary, _records = self.verify_and_read(verify_frames=verify_frames)
        return summary


_ANNOTATION_REQUIRED_KEYS = {"schema", "session_id", "generation", "frame_id", "gates"}
_ANNOTATION_OPTIONAL_KEYS = {
    "expected_command",
    "estimator_rpy_rad",
    "expected_estimator_healthy",
    "active_gate_index",
}
_ANNOTATION_GATE_KEYS = {"center_px", "corners_px", "gate_index"}
_ANNOTATION_COMMAND_KEYS = {"roll_rate", "pitch_rate", "yaw_rate", "thrust"}


def _exact_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _exact_finite_vector(value: Any, shape: tuple[int, ...]) -> bool:
    """Validate JSON numeric geometry without bool/NumPy/string coercion."""

    if len(shape) == 1:
        return (
            type(value) is list
            and len(value) == shape[0]
            and all(_exact_finite_number(component) for component in value)
        )
    return (
        type(value) is list
        and len(value) == shape[0]
        and all(_exact_finite_vector(component, shape[1:]) for component in value)
    )


def load_annotations_bytes(
    payload: bytes,
) -> Dict[tuple[str, int, int], Dict[str, Any]]:
    """Parse the exact annotation bytes whose digest is bound into evidence."""

    if len(payload) > MAX_REPLAY_ANNOTATIONS_BYTES:
        raise ValueError("annotations exceed replay resource limit")
    result: Dict[tuple[str, int, int], Dict[str, Any]] = {}
    for line_number, encoded_line in enumerate(io.BytesIO(payload), start=1):
        if len(encoded_line) > MAX_REPLAY_ANNOTATION_LINE_BYTES:
            raise ValueError(
                f"annotation line exceeds replay resource limit on line {line_number}"
            )
        if len(result) >= MAX_REPLAY_ANNOTATION_COUNT:
            raise ValueError("annotation count exceeds replay resource limit")
        try:
            line = encoded_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"annotations must be UTF-8 on line {line_number}"
            ) from exc
        if not line.strip():
            continue
        row = strict_json_loads(line)
        if type(row) is not dict:
            raise ValueError(f"annotation row must be an object on line {line_number}")
        keys = set(row)
        missing = _ANNOTATION_REQUIRED_KEYS - keys
        unknown = keys - (_ANNOTATION_REQUIRED_KEYS | _ANNOTATION_OPTIONAL_KEYS)
        if missing or unknown:
            raise ValueError(
                f"invalid annotation keys on line {line_number}: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if row["schema"] != ANNOTATION_SCHEMA:
            raise ValueError(f"invalid annotation schema on line {line_number}")
        session_id = row["session_id"]
        generation = row["generation"]
        frame_id = row["frame_id"]
        if type(session_id) is not str or not session_id.strip():
            raise ValueError(f"invalid annotation session_id on line {line_number}")
        if type(generation) is not int or generation < 0:
            raise ValueError(f"invalid annotation generation on line {line_number}")
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError(f"invalid annotation frame_id on line {line_number}")
        key = (session_id, generation, frame_id)
        if key in result:
            raise ValueError(f"duplicate annotation key on line {line_number}")
        gates = row["gates"]
        if type(gates) is not list:
            raise ValueError(f"annotation gates must be a list on line {line_number}")
        for gate in gates:
            if type(gate) is not dict or set(gate) - _ANNOTATION_GATE_KEYS:
                raise ValueError(f"annotation gate keys are invalid on line {line_number}")
            if "center_px" not in gate:
                raise ValueError(f"annotation gate keys are invalid on line {line_number}")
            if not _exact_finite_vector(gate["center_px"], (2,)):
                raise ValueError(f"annotation gate center is invalid on line {line_number}")
            if "corners_px" in gate and not _exact_finite_vector(gate["corners_px"], (4, 2)):
                raise ValueError(f"annotation corners invalid on line {line_number}")
            if "gate_index" in gate and (
                type(gate["gate_index"]) is not int or gate["gate_index"] < 0
            ):
                raise ValueError(f"annotation gate index invalid on line {line_number}")
        if "expected_command" in row:
            command = row["expected_command"]
            if (
                type(command) is not dict
                or set(command) != _ANNOTATION_COMMAND_KEYS
                or not all(_exact_finite_number(command[name]) for name in _ANNOTATION_COMMAND_KEYS)
            ):
                raise ValueError(f"annotation expected_command invalid on line {line_number}")
        if "estimator_rpy_rad" in row:
            if not _exact_finite_vector(row["estimator_rpy_rad"], (3,)):
                raise ValueError(f"annotation estimator_rpy_rad invalid on line {line_number}")
        if "expected_estimator_healthy" in row and type(
            row["expected_estimator_healthy"]
        ) is not bool:
            raise ValueError(
                f"annotation expected_estimator_healthy invalid on line {line_number}"
            )
        if "active_gate_index" in row and (
            type(row["active_gate_index"]) is not int
            or row["active_gate_index"] < 0
        ):
            raise ValueError(
                f"annotation active_gate_index invalid on line {line_number}"
            )
        result[key] = row
    return result


def load_annotations(path: Optional[Path | str]) -> Dict[tuple[str, int, int], Dict[str, Any]]:
    if path is None:
        return {}
    return load_annotations_bytes(
        _read_bounded_secure_file(
            path,
            maximum_bytes=MAX_REPLAY_ANNOTATIONS_BYTES,
            label="replay annotations",
        )
    )


def grouped_session_split(
    sessions: Iterable[tuple[str, str]],
    *,
    validation_fraction: float = 0.2,
    salt: str = "aigp-vq2-session-split-v1",
) -> Dict[str, str]:
    """Assign whole ``(session_id, dataset_hash)`` groups to train/validation."""

    if type(validation_fraction) not in {int, float} or not math.isfinite(
        validation_fraction
    ) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0,1)")
    if type(salt) is not str or not salt:
        raise ValueError("salt must be a non-empty exact string")
    assignments: Dict[str, str] = {}
    buckets: Dict[str, int] = {}
    threshold = int(validation_fraction * (1 << 64))
    for item in sessions:
        if type(item) not in {tuple, list} or len(item) != 2:
            raise ValueError("each session must be an exact (session_id, dataset_hash) pair")
        session_id, dataset_hash = item
        if type(session_id) is not str or not session_id.strip():
            raise ValueError("session id must be a non-empty exact string")
        if not _is_sha256(dataset_hash):
            raise ValueError("dataset hash must be a lowercase SHA-256")
        if session_id in assignments:
            raise ValueError(f"duplicate session id: {session_id}")
        digest = bytes.fromhex(sha256_text(f"{salt}\0{session_id}\0{dataset_hash}"))
        bucket = int.from_bytes(digest[:8], "big")
        buckets[session_id] = bucket
        assignments[session_id] = "validation" if bucket < threshold else "train"
    if len(assignments) >= 2 and len(set(assignments.values())) == 1:
        # Preserve deterministic hash assignment while guaranteeing that the
        # retained corpus can actually measure held-out behavior.  Move only
        # the session nearest the corresponding edge of the hash interval.
        if next(iter(assignments.values())) == "train":
            assignments[min(buckets, key=lambda key: (buckets[key], key))] = "validation"
        else:
            assignments[max(buckets, key=lambda key: (buckets[key], key))] = "train"
    return assignments


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _point(value: Any) -> Optional[np.ndarray]:
    if not _exact_finite_vector(value, (2,)):
        return None
    return np.asarray(value, dtype=float)


def _center(detection: Mapping[str, Any]) -> Optional[np.ndarray]:
    return _point(detection.get("center_px", detection.get("center")))


def _corner_error(predicted: Any, truth: Any) -> Optional[float]:
    if not _exact_finite_vector(predicted, (4, 2)) or not _exact_finite_vector(
        truth, (4, 2)
    ):
        return None
    p = np.asarray(predicted, dtype=float)
    t = np.asarray(truth, dtype=float)
    candidates = []
    for reverse in (False, True):
        ordering = p[::-1] if reverse else p
        for shift in range(4):
            aligned = np.roll(ordering, shift, axis=0)
            candidates.append(float(np.mean(np.linalg.norm(aligned - t, axis=1))))
    return min(candidates)


def _match_gates(
    detections: Sequence[Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
    *,
    max_center_error_px: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    distances: Dict[tuple[int, int], float] = {}
    for detection_index, detection in enumerate(detections):
        predicted = _center(detection)
        if predicted is None:
            continue
        for truth_index, truth in enumerate(truths):
            actual = _center(truth)
            if actual is None:
                continue
            distance = float(np.linalg.norm(predicted - actual))
            if distance <= max_center_error_px:
                distances[(detection_index, truth_index)] = distance

    # Unit-capacity min-cost max-flow gives maximum recall first, then the
    # minimum total center error among all maximum-cardinality assignments.
    # A greedy shortest-edge matcher can undercount valid gates when the
    # nearest detection is the only alternative for a second truth.
    detection_count = len(detections)
    truth_count = len(truths)
    source = 0
    detection_offset = 1
    truth_offset = detection_offset + detection_count
    sink = truth_offset + truth_count
    graph: list[list[list[Any]]] = [[] for _ in range(sink + 1)]

    def add_edge(origin: int, target: int, capacity: int, cost: float) -> int:
        forward = [target, len(graph[target]), capacity, cost]
        reverse = [origin, len(graph[origin]), 0, -cost]
        graph[origin].append(forward)
        graph[target].append(reverse)
        return len(graph[origin]) - 1

    for detection_index in range(detection_count):
        add_edge(source, detection_offset + detection_index, 1, 0.0)
    for truth_index in range(truth_count):
        add_edge(truth_offset + truth_index, sink, 1, 0.0)
    assignment_edges: Dict[tuple[int, int], tuple[int, int]] = {}
    for (detection_index, truth_index), distance in sorted(distances.items()):
        origin = detection_offset + detection_index
        edge_index = add_edge(
            origin, truth_offset + truth_index, 1, distance
        )
        assignment_edges[(detection_index, truth_index)] = (origin, edge_index)

    node_count = len(graph)
    while True:
        best = [math.inf] * node_count
        predecessor: list[Optional[tuple[int, int]]] = [None] * node_count
        best[source] = 0.0
        # Bellman-Ford is small and handles negative residual reverse edges.
        for _ in range(node_count - 1):
            changed = False
            for origin in range(node_count):
                if not math.isfinite(best[origin]):
                    continue
                for edge_index, edge in enumerate(graph[origin]):
                    target, _reverse, capacity, cost = edge
                    if capacity <= 0:
                        continue
                    candidate = best[origin] + cost
                    if candidate < best[target] - 1e-12:
                        best[target] = candidate
                        predecessor[target] = (origin, edge_index)
                        changed = True
            if not changed:
                break
        if predecessor[sink] is None:
            break
        node = sink
        while node != source:
            origin, edge_index = predecessor[node]  # type: ignore[misc]
            edge = graph[origin][edge_index]
            reverse_index = edge[1]
            edge[2] -= 1
            graph[node][reverse_index][2] += 1
            node = origin

    matches = [
        (detection_index, truth_index, distances[(detection_index, truth_index)])
        for (detection_index, truth_index), (origin, edge_index) in assignment_edges.items()
        if graph[origin][edge_index][2] == 0
    ]
    matches.sort()
    matched_detections = {item[0] for item in matches}
    matched_truths = {item[1] for item in matches}
    return (
        matches,
        [index for index in range(len(detections)) if index not in matched_detections],
        [index for index in range(len(truths)) if index not in matched_truths],
    )


def _command_values(command: Any) -> Optional[np.ndarray]:
    if type(command) is not dict:
        return None
    canonical = ("roll_rate", "pitch_rate", "yaw_rate", "thrust")
    alternate = (
        "roll_rate_rad_s",
        "pitch_rate_rad_s",
        "yaw_rate_rad_s",
        "thrust",
    )
    if set(command) == set(canonical):
        names = canonical
    elif set(command) == set(alternate):
        names = alternate
    else:
        return None
    raw_values = [command[name] for name in names]
    if any(
        type(value) not in {int, float} or not math.isfinite(value)
        for value in raw_values
    ):
        return None
    return np.asarray(raw_values, dtype=float)


def _valid_estimator(estimator: Any) -> bool:
    required = {"healthy", "rpy_rad", "body_rates"}
    allowed = required | {
        "timestamp_us",
        "orientation_wxyz",
        "gyro_bias",
        "reason",
        "propagated",
    }
    if type(estimator) is not dict or not required <= set(estimator) or set(estimator) - allowed:
        return False
    if type(estimator["healthy"]) is not bool:
        return False
    if not _exact_finite_vector(estimator["rpy_rad"], (3,)):
        return False
    if not _exact_finite_vector(estimator["body_rates"], (3,)):
        return False
    if "timestamp_us" in estimator and (
        type(estimator["timestamp_us"]) is not int or estimator["timestamp_us"] < 0
    ):
        return False
    if "orientation_wxyz" in estimator and not _exact_finite_vector(
        estimator["orientation_wxyz"], (4,)
    ):
        return False
    if "gyro_bias" in estimator and not _exact_finite_vector(
        estimator["gyro_bias"], (3,)
    ):
        return False
    if "reason" in estimator and type(estimator["reason"]) is not str:
        return False
    if "propagated" in estimator and type(estimator["propagated"]) is not bool:
        return False
    return True


def score_records(
    records: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
    annotations: Optional[Mapping[tuple[str, int, int], Mapping[str, Any]]] = None,
    max_center_error_px: float = 80.0,
) -> Dict[str, Any]:
    """Score perception, temporal, estimator, and open-loop command behavior."""

    if type(session_id) is not str or not session_id.strip():
        raise ValueError("session_id must be an exact non-empty string")
    if (
        type(max_center_error_px) not in {int, float}
        or not math.isfinite(max_center_error_px)
        or max_center_error_px <= 0
    ):
        raise ValueError("max_center_error_px must be finite and positive")
    labels = annotations or {}
    ordered_records: list[Dict[str, Any]] = []
    observed_sequences: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("score records must be mappings")
        sequence = record.get("sequence")
        if type(sequence) is not int or sequence < 0:
            raise ValueError("score record sequence must be a non-negative exact integer")
        if sequence in observed_sequences:
            raise ValueError("score record sequences must be unique")
        observed_sequences.add(sequence)
        ordered_records.append(dict(record))
    ordered_records.sort(key=lambda row: row["sequence"])
    # Publication sequence is the causal/controller order. Simulator clocks
    # may regress or arrive out of order and are evidence fields, not ordering
    # authorities.
    frames = [row for row in ordered_records if row.get("type") == "frame"]
    frame_tokens = {
        (session_id, int(frame.get("generation", 0)), int(frame["frame_id"]))
        for frame in frames
    }
    label_tokens = set(labels)
    if any(
        type(token) is not tuple
        or len(token) != 3
        or type(token[0]) is not str
        or type(token[1]) is not int
        or type(token[2]) is not int
        for token in label_tokens
    ):
        raise ValueError("annotation mapping contains an invalid exact frame token")
    wrong_session = sorted(
        token for token in label_tokens if token[0] != session_id
    )
    if wrong_session:
        raise ValueError("annotation mapping contains a wrong-session frame token")
    orphaned = sorted(label_tokens - frame_tokens)
    if orphaned:
        raise ValueError("annotation mapping contains an orphan frame token")

    truth_count = matches_count = false_positives = labeled_frames = 0
    center_errors: list[float] = []
    corner_errors: list[float] = []
    miss_streak = longest_miss_streak = 0
    detector_latencies: list[float] = []
    full_stack_latencies: list[float] = []
    tracked_centers: list[tuple[int, Optional[int], int, np.ndarray]] = []
    transition_times: list[float] = []
    transition_count = 0
    pending_transition: Optional[tuple[float, int, int]] = None
    prior_gate: Optional[int] = None
    prior_generation: Optional[int] = None
    miss_epoch: Optional[tuple[int, Optional[int]]] = None
    healthy_estimates = unhealthy_estimates = 0
    estimator_present = estimator_missing = estimator_invalid = 0
    estimator_rpy_squared_errors: list[np.ndarray] = []
    estimator_rpy_frame_rmse: list[float] = []
    estimator_rpy_references: list[np.ndarray] = []
    estimator_health_labeled = estimator_health_compared = 0
    estimator_health_mismatches = 0
    active_gate_labeled = active_gate_mismatches = 0
    expected_command_squared_errors: list[np.ndarray] = []
    expected_command_frame_rmse: list[float] = []
    expected_command_references: list[np.ndarray] = []
    frame_generated_count = 0
    frame_invalid_commands = 0
    frame_command_limit_violations = 0
    frame_zero_commands = 0
    frame_generated_sent_errors: list[float] = []

    for frame in frames:
        latency = frame.get("detector_latency_ms")
        if type(latency) in {int, float} and math.isfinite(latency) and latency >= 0:
            detector_latencies.append(float(latency))
        full_stack_latency = frame.get("full_stack_latency_ms")
        if (
            type(full_stack_latency) in {int, float}
            and math.isfinite(full_stack_latency)
            and full_stack_latency >= 0
        ):
            full_stack_latencies.append(float(full_stack_latency))
        token = (session_id, int(frame.get("generation", 0)), int(frame["frame_id"]))
        generation = token[1]
        sequence = int(frame["sequence"])
        race = frame.get("race_status")
        current_gate: Optional[int] = None
        if isinstance(race, Mapping):
            raw_gate = race.get("active_gate_index", race.get("gate_index"))
            if isinstance(raw_gate, int) and not isinstance(raw_gate, bool):
                current_gate = raw_gate
        if prior_generation is not None and generation != prior_generation:
            prior_gate = None
            pending_transition = None
            miss_streak = 0
        prior_generation = generation
        epoch = (generation, current_gate)
        if epoch != miss_epoch:
            miss_streak = 0
            miss_epoch = epoch
        frame_time = float(frame.get("received_monotonic_s", 0.0))
        if current_gate is not None and prior_gate is not None and current_gate != prior_gate:
            transition_count += 1
            pending_transition = (frame_time, sequence, current_gate)
        if current_gate is not None:
            prior_gate = current_gate
        annotation = labels.get(token)
        detections = [
            item for item in (frame.get("detections") or [])
            if isinstance(item, Mapping) and item.get("selector_eligible", True)
        ]
        frame_matched = False
        if annotation is not None:
            labeled_frames += 1
            truths = [item for item in annotation.get("gates", []) if isinstance(item, Mapping)]
            if "active_gate_index" in annotation:
                labeled_gate = annotation["active_gate_index"]
                if type(labeled_gate) is not int or labeled_gate < 0:
                    raise ValueError("active gate annotation must be a non-negative integer")
                active_gate_labeled += 1
                if current_gate != labeled_gate:
                    active_gate_mismatches += 1
            truth_count += len(truths)
            matched, unmatched_detection, _unmatched_truth = _match_gates(
                detections, truths, max_center_error_px=max_center_error_px
            )
            matches_count += len(matched)
            false_positives += len(unmatched_detection)
            frame_matched = bool(matched)
            for detection_index, truth_index, distance in matched:
                center_errors.append(distance)
                corner = _corner_error(
                    detections[detection_index].get("corners_px", detections[detection_index].get("corners")),
                    truths[truth_index].get("corners_px", truths[truth_index].get("corners")),
                )
                if corner is not None:
                    corner_errors.append(corner)
            if truths and not frame_matched:
                miss_streak += 1
                longest_miss_streak = max(longest_miss_streak, miss_streak)
            else:
                miss_streak = 0

            expected_command = annotation.get("expected_command")
            generated = _command_values(frame.get("generated_command"))
            expected = _command_values(expected_command)
            if expected is not None:
                expected_command_references.append(expected)
            if generated is not None and expected is not None:
                squared = np.square(generated - expected)
                expected_command_squared_errors.append(squared)
                expected_command_frame_rmse.append(float(np.sqrt(np.mean(squared))))

            reference_rpy = annotation.get("estimator_rpy_rad")
            estimate = frame.get("estimator")
            if reference_rpy is not None:
                estimator_rpy_references.append(np.asarray(reference_rpy, dtype=float))
            if isinstance(estimate, Mapping) and reference_rpy is not None:
                predicted_rpy = np.asarray(estimate.get("rpy_rad", estimate.get("rpy", [])), dtype=float)
                actual_rpy = np.asarray(reference_rpy, dtype=float)
                if predicted_rpy.shape == (3,) and actual_rpy.shape == (3,) and np.all(np.isfinite(predicted_rpy)) and np.all(np.isfinite(actual_rpy)):
                    wrapped = np.arctan2(np.sin(predicted_rpy - actual_rpy), np.cos(predicted_rpy - actual_rpy))
                    squared = np.square(wrapped)
                    estimator_rpy_squared_errors.append(squared)
                    estimator_rpy_frame_rmse.append(float(np.sqrt(np.mean(squared))))

            if "expected_estimator_healthy" in annotation:
                expected_health = annotation["expected_estimator_healthy"]
                if type(expected_health) is not bool:
                    raise ValueError("expected estimator health label must be an exact bool")
                estimator_health_labeled += 1
                if _valid_estimator(estimate):
                    estimator_health_compared += 1
                    if estimate["healthy"] is not expected_health:
                        estimator_health_mismatches += 1

            if (
                pending_transition is not None
                and any(
                    truths[truth_index].get("gate_index") == pending_transition[2]
                    for _detection_index, truth_index, _distance in matched
                )
                and annotation.get("active_gate_index") == pending_transition[2]
                and current_gate == pending_transition[2]
                and sequence > pending_transition[1]
            ):
                transition_times.append(
                    max(0.0, frame_time - pending_transition[0]) * 1000.0
                )
                pending_transition = None

        raw_frame_command = frame.get("generated_command")
        if raw_frame_command is not None:
            frame_generated_count += 1
            generated = _command_values(raw_frame_command)
            if generated is None:
                frame_invalid_commands += 1
            else:
                if np.all(generated == 0.0):
                    frame_zero_commands += 1
                if (
                    abs(generated[0]) > 0.25 + 1e-12
                    or abs(generated[1]) > 0.25 + 1e-12
                    or abs(generated[2]) > 1e-12
                    or not 0.0 <= generated[3] <= 0.35 + 1e-12
                ):
                    frame_command_limit_violations += 1
                synchronized_sent = _command_values(frame.get("sent_command"))
                if synchronized_sent is not None:
                    frame_generated_sent_errors.append(
                        float(np.max(np.abs(generated - synchronized_sent)))
                    )

        tracker = frame.get("tracker")
        tracker_center: Optional[np.ndarray] = None
        if isinstance(tracker, Mapping):
            target = tracker.get("target", tracker)
            if isinstance(target, Mapping):
                tracker_center = _point(
                    target.get(
                        "center_px",
                        [target.get("center_x"), target.get("center_y")],
                    )
                )
        primary_center = (
            tracker_center
            if tracker_center is not None
            else (_center(detections[0]) if detections else None)
        )
        if primary_center is not None:
            tracked_centers.append(
                (
                    generation,
                    current_gate,
                    int(frame.get("sim_time_ns", 0)),
                    primary_center,
                )
            )

        estimate = frame.get("estimator")
        if estimate is None:
            estimator_missing += 1
        elif not _valid_estimator(estimate):
            estimator_present += 1
            estimator_invalid += 1
        else:
            estimator_present += 1
            if estimate["healthy"]:
                healthy_estimates += 1
            else:
                unhealthy_estimates += 1

    center_steps: list[float] = []
    for (generation_a, gate_a, _time_a, center_a), (
        generation_b,
        gate_b,
        _time_b,
        center_b,
    ) in zip(tracked_centers, tracked_centers[1:]):
        if generation_a == generation_b and gate_a == gate_b:
            center_steps.append(float(np.linalg.norm(center_b - center_a)))

    command_rows = [
        row for row in ordered_records if row.get("type") == "command"
    ]
    generated_rows = [row for row in command_rows if row.get("kind") == "generated"]
    sent_rows = [row for row in command_rows if row.get("kind") == "sent"]
    invalid_commands = command_limit_violations = zero_commands = 0
    for row in command_rows:
        values = _command_values(row.get("command"))
        if values is None:
            invalid_commands += 1
        else:
            if np.all(values == 0.0):
                zero_commands += 1
            if (
                abs(values[0]) > 0.25 + 1e-12
                or abs(values[1]) > 0.25 + 1e-12
                or abs(values[2]) > 1e-12
                or not 0.0 <= values[3] <= 0.35 + 1e-12
            ):
                command_limit_violations += 1
    generated_sent_errors: list[float] = []
    def command_key(row: Mapping[str, Any]) -> tuple[str, Any]:
        token = row.get("frame_token")
        if isinstance(token, Sequence) and not isinstance(token, (str, bytes)):
            return ("frame", tuple(int(value) for value in token))
        stamp = row.get("monotonic_s")
        if isinstance(stamp, (int, float)) and math.isfinite(float(stamp)):
            return ("time_us", int(round(float(stamp) * 1_000_000.0)))
        return ("sequence", int(row.get("sequence", -1)))
    sent_by_key: Dict[tuple[str, Any], list[Mapping[str, Any]]] = {}
    for sent_row in sent_rows:
        sent_by_key.setdefault(command_key(sent_row), []).append(sent_row)
    unmatched_generated = 0
    for generated_row in generated_rows:
        candidates = sent_by_key.get(command_key(generated_row), [])
        if not candidates:
            unmatched_generated += 1
            continue
        sent_row = candidates.pop(0)
        generated = _command_values(generated_row.get("command"))
        sent = _command_values(sent_row.get("command"))
        if generated is not None and sent is not None:
            generated_sent_errors.append(float(np.max(np.abs(generated - sent))))

    return {
        "schema": "aigp-vq2-replay-score/1",
        "session_id": session_id,
        "frames": len(frames),
        "labeled_frames": labeled_frames,
        "annotation_frames_provided": len(label_tokens),
        "annotation_frame_coverage": (
            labeled_frames / len(frames) if frames else None
        ),
        "active_gate_labeled_frames": active_gate_labeled,
        "active_gate_label_coverage": (
            active_gate_labeled / len(frames) if frames else None
        ),
        "active_gate_label_mismatch_count": active_gate_mismatches,
        "perception": {
            "gate_truth_count": truth_count,
            "gate_matches": matches_count,
            "gate_recall": matches_count / truth_count if truth_count else None,
            "false_positives": false_positives,
            "false_positives_per_frame": false_positives / labeled_frames if labeled_frames else None,
            "center_error_px_mean": statistics.fmean(center_errors) if center_errors else None,
            "center_error_px_p95": _percentile(center_errors, 95.0),
            "corner_error_px_mean": statistics.fmean(corner_errors) if corner_errors else None,
            "corner_error_px_p95": _percentile(corner_errors, 95.0),
            "longest_consecutive_missed_frames": longest_miss_streak,
            "temporal_center_step_px_p50": _percentile(center_steps, 50.0),
            "temporal_center_step_px_p95": _percentile(center_steps, 95.0),
            "post_gate_reacquisition_latency_ms": transition_times,
            "post_gate_reacquisition_latency_ms_p95": _percentile(transition_times, 95.0),
            "transition_count": transition_count,
            "reacquired_count": len(transition_times),
            "unreacquired_count": transition_count - len(transition_times),
            "detector_latency_ms_p50": _percentile(detector_latencies, 50.0),
            "detector_latency_ms_p95": _percentile(detector_latencies, 95.0),
            "full_stack_latency_ms_p50": _percentile(full_stack_latencies, 50.0),
            "full_stack_latency_ms_p95": _percentile(full_stack_latencies, 95.0),
        },
        "estimator": {
            "present_frame_estimates": estimator_present,
            "missing_frame_estimates": estimator_missing,
            "invalid_frame_estimates": estimator_invalid,
            "healthy_frame_estimates": healthy_estimates,
            "unhealthy_frame_estimates": unhealthy_estimates,
            "rpy_labeled_frames": len(estimator_rpy_references),
            "rpy_label_coverage": (
                len(estimator_rpy_references) / len(frames) if frames else None
            ),
            "rpy_compared_frames": len(estimator_rpy_squared_errors),
            "rpy_comparison_coverage": (
                len(estimator_rpy_squared_errors) / len(estimator_rpy_references)
                if estimator_rpy_references
                else None
            ),
            "health_labeled_frames": estimator_health_labeled,
            "health_label_coverage": (
                estimator_health_labeled / len(frames) if frames else None
            ),
            "health_compared_frames": estimator_health_compared,
            "health_comparison_coverage": (
                estimator_health_compared / estimator_health_labeled
                if estimator_health_labeled
                else None
            ),
            "health_mismatch_count": estimator_health_mismatches,
            "rpy_reference_rms_rad": (
                float(
                    np.sqrt(
                        np.mean(
                            np.square(np.concatenate(estimator_rpy_references))
                        )
                    )
                )
                if estimator_rpy_references
                else None
            ),
            "rpy_rmse_rad": (
                float(
                    np.sqrt(
                        np.mean(np.concatenate(estimator_rpy_squared_errors))
                    )
                )
                if estimator_rpy_squared_errors
                else None
            ),
            "rpy_mean_frame_rmse_rad": (
                statistics.fmean(estimator_rpy_frame_rmse)
                if estimator_rpy_frame_rmse
                else None
            ),
        },
        "open_loop_commands": {
            "recorded_stream": {
                "generated_count": len(generated_rows),
                "sent_count": len(sent_rows),
                "invalid_count": invalid_commands,
                "zero_command_count": zero_commands,
                "envelope_violation_count": command_limit_violations,
                "generated_sent_max_abs_error": max(generated_sent_errors) if generated_sent_errors else None,
                "generated_without_matching_send": unmatched_generated,
                "sent_without_matching_generation": sum(len(rows) for rows in sent_by_key.values()),
            },
            "replay_frames": {
                "generated_count": frame_generated_count,
                "invalid_count": frame_invalid_commands,
                "zero_command_count": frame_zero_commands,
                "envelope_violation_count": frame_command_limit_violations,
                "expected_command_labeled_frames": len(
                    expected_command_references
                ),
                "expected_command_label_coverage": (
                    len(expected_command_references) / len(frames)
                    if frames
                    else None
                ),
                "expected_command_compared_frames": len(
                    expected_command_squared_errors
                ),
                "expected_command_comparison_coverage": (
                    len(expected_command_squared_errors)
                    / len(expected_command_references)
                    if expected_command_references
                    else None
                ),
                "expected_command_reference_rms": (
                    float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    np.concatenate(expected_command_references)
                                )
                            )
                        )
                    )
                    if expected_command_references
                    else None
                ),
                "generated_sent_max_abs_error": (
                    max(frame_generated_sent_errors)
                    if frame_generated_sent_errors
                    else None
                ),
                "expected_command_rmse": (
                    float(
                        np.sqrt(
                            np.mean(
                                np.concatenate(expected_command_squared_errors)
                            )
                        )
                    )
                    if expected_command_squared_errors
                    else None
                ),
                "expected_command_mean_frame_rmse": (
                    statistics.fmean(expected_command_frame_rmse)
                    if expected_command_frame_rmse
                    else None
                ),
            },
        },
        "limitations": [
            "Open-loop replay cannot validate future observations changed by commands.",
            "Closed-loop command/controller changes still require promoted simulation and explicitly authorized live evaluation.",
        ],
    }


def _load_processor(spec: str) -> Callable[[np.ndarray, Mapping[str, Any]], Mapping[str, Any]]:
    if ":" not in spec:
        raise ValueError("processor must use module:function syntax")
    module_name, attribute = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), attribute)
    if not callable(function):
        raise TypeError("replay processor is not callable")
    return function


def _candidate_worktree_root(value: Optional[Path | str]) -> Path:
    root = Path(value or Path.cwd()).resolve()
    if not root.is_dir():
        raise ValueError("candidate worktree is missing")
    try:
        top = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(root),
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
                shell=False,
            ).stdout.strip()
        ).resolve()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("candidate worktree must be an exact Git worktree") from exc
    if top != root:
        raise ValueError("candidate worktree must be the exact Git top level")
    return root


def _assert_pristine_candidate_worktree(root: Path) -> None:
    """Require the exact lexical checkout represented by candidate HEAD."""

    root = secure_directory(root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
            shell=False,
        ).stdout.strip()
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "-z", commit],
            cwd=str(root),
            check=True,
            capture_output=True,
            timeout=30.0,
            shell=False,
        ).stdout
        tracked_diff = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "--no-ext-diff",
                "--no-textconv",
                "HEAD",
                "--",
            ],
            cwd=str(root),
            capture_output=True,
            timeout=30.0,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("candidate pristine checkout could not be verified") from exc
    if (
        len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
        or tracked_diff.returncode != 0
    ):
        raise ValueError("candidate worktree is not an exact pristine checkout")

    expected_files: set[str] = set()
    expected_directories: set[str] = set()
    for raw_name in listing.split(b"\0"):
        if not raw_name:
            continue
        relative = Path(os.fsdecode(raw_name))
        if (
            relative.is_absolute()
            or relative.drive
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("candidate Git tree contains an unsafe path")
        expected_files.add(relative.as_posix())
        parent = relative.parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        retained_names = []
        for name in sorted(names):
            target = base / name
            info = target.lstat()
            if stat.S_ISLNK(info.st_mode) or (
                getattr(info, "st_file_attributes", 0) & reparse_flag
            ):
                raise ValueError(
                    "candidate worktree contains a symlink/reparse directory"
                )
            if base == root and name == ".git":
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("candidate worktree contains an unsafe directory")
            retained_names.append(name)
            observed_directories.add(target.relative_to(root).as_posix())
        names[:] = retained_names
        for name in sorted(files):
            target = base / name
            if base == root and name == ".git":
                secure_regular_file(target)
                continue
            secure_regular_file(target)
            observed_files.add(target.relative_to(root).as_posix())
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("candidate worktree is not an exact pristine checkout")


def _processor_source_path(
    spec: str, candidate_worktree: Optional[Path | str] = None
) -> tuple[Path, str, str]:
    if ":" not in spec:
        raise ValueError("processor must use module:function syntax")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("processor must use non-empty module:function names")
    root = _candidate_worktree_root(candidate_worktree)
    module_parts = module_name.split(".")
    if any(not part or not part.isidentifier() for part in module_parts):
        raise ValueError("processor module must be a dotted identifier")
    if not attribute.isidentifier():
        raise ValueError("processor attribute must be an identifier")
    module_relative = Path(*module_parts)
    candidates = (
        module_relative.with_suffix(".py"),
        module_relative / "__init__.py",
    )
    path: Optional[Path] = None
    for candidate in candidates:
        try:
            path = secure_relative_regular_file(root, candidate)
            break
        except ValueError:
            continue
    if path is None or root not in path.parents:
        raise ValueError("processor must resolve to a local worktree source file")
    # Importlib can merge a dotted path with an installed regular package when
    # a local parent is only a PEP 420 namespace.  Promotion evidence permits
    # dotted modules only through explicit, securely resolved local packages.
    for depth in range(1, len(module_parts)):
        parent_init = Path(*module_parts[:depth]) / "__init__.py"
        try:
            secure_relative_regular_file(root, parent_init)
        except ValueError as exc:
            raise ValueError(
                "processor dotted module requires secure local package parents"
            ) from exc
    return path, module_name, attribute


def _processor_code_hash(
    spec: str,
    candidate_worktree: Optional[Path | str] = None,
    *,
    require_pristine: bool = False,
) -> str:
    path, _module_name, _attribute = _processor_source_path(
        spec, candidate_worktree
    )
    root = _candidate_worktree_root(candidate_worktree)
    try:
        if require_pristine:
            _assert_pristine_candidate_worktree(root)
        code_hash = git_provenance(root)[2]
        if require_pristine:
            # Close the inventory/provenance capture window before launch.
            _assert_pristine_candidate_worktree(root)
        return code_hash
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            "processor provenance requires a complete Git worktree identity"
        ) from exc


class IsolatedReplayProcessor:
    """Line protocol to a candidate process launched by a trusted OS wrapper."""

    def __init__(
        self,
        spec: str,
        wrapper: Path | str,
        wrapper_sha256: str,
        *,
        seed: int = 0,
        response_timeout_s: float = 5.0,
        max_response_bytes: int = 1_048_576,
        candidate_worktree: Optional[Path | str] = None,
    ) -> None:
        self.spec = spec
        if type(seed) is not int:
            raise ValueError("replay processor seed must be an exact integer")
        if (
            type(response_timeout_s) not in {int, float}
            or not math.isfinite(response_timeout_s)
            or response_timeout_s <= 0
        ):
            raise ValueError("response_timeout_s must be finite and positive")
        if type(max_response_bytes) is not int or max_response_bytes < 1024:
            raise ValueError("max_response_bytes must be an exact integer >= 1024")
        self.seed = seed
        self.response_timeout_s = float(response_timeout_s)
        self.max_response_bytes = max_response_bytes
        self.candidate_worktree = _candidate_worktree_root(candidate_worktree)
        processor_source, _module_name, _attribute = _processor_source_path(
            spec, self.candidate_worktree
        )
        processor_source_relative = processor_source.relative_to(
            self.candidate_worktree
        ).as_posix()
        wrapper_path = Path(wrapper)
        if (
            type(wrapper_sha256) is not str
            or len(wrapper_sha256) != 64
            or any(character not in "0123456789abcdef" for character in wrapper_sha256)
            or not wrapper_path.is_absolute()
        ):
            raise ValueError("isolation wrapper path/hash is missing or mismatched")
        try:
            self.wrapper = secure_regular_file(wrapper_path)
            if sha256_bytes(read_secure_regular_file(self.wrapper)) != wrapper_sha256:
                raise ValueError("isolation wrapper digest mismatch")
        except ValueError as exc:
            raise ValueError(
                "isolation wrapper path/hash is missing or mismatched"
            ) from exc
        self._wrapper_sha256 = wrapper_sha256
        attestation_run = subprocess.run(
            [str(self.wrapper), "--attest"],
            capture_output=True,
            text=True,
            timeout=5.0,
            shell=False,
        )
        if attestation_run.returncode != 0:
            raise RuntimeError("isolation wrapper attestation failed")
        attestation = strict_json_loads(attestation_run.stdout)
        expected = {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "denied",
            "filesystem": "readonly-worktree-only",
            "non_interactive": True,
            "process_tree_containment": "kill-on-wrapper-exit",
            "host_process_access": "denied",
        }
        if type(attestation) is not dict or attestation != expected:
            raise RuntimeError("isolation wrapper did not attest required controls")
        # Attestation is a separate process invocation.  Reverify the exact
        # executable bytes immediately before the candidate-bearing launch.
        if (
            secure_regular_file(self.wrapper) != self.wrapper
            or sha256_bytes(read_secure_regular_file(self.wrapper))
            != self._wrapper_sha256
        ):
            raise RuntimeError("isolation wrapper changed after attestation")
        self.attestation = {
            **attestation,
            "wrapper_sha256": wrapper_sha256,
        }
        process_group: Dict[str, Any]
        if os.name == "nt":
            process_group = {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
            }
        else:
            process_group = {"start_new_session": True}
        allowed_environment = {
            "PATH",
            "SystemRoot",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "TEMP",
            "TMP",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
            "LANG",
            "LC_ALL",
        }
        allowed_folded = {name.casefold() for name in allowed_environment}
        worker_environment = {
            name: value
            for name, value in os.environ.items()
            if name.casefold() in allowed_folded
        }
        worker_environment.update(
            {
                "AIGP_REPLAY_CANDIDATE": "1",
                "AIGP_REPLAY_SEED": str(seed),
                "AIGP_TRIAL_OFFLINE": "1",
                # Hash randomization is part of the candidate's deterministic
                # execution identity.  Python accepts only the unsigned
                # 32-bit seed domain for this variable.
                "PYTHONHASHSEED": str(seed & 0xFFFFFFFF),
                "PYTHONUNBUFFERED": "1",
            }
        )
        worker = secure_relative_regular_file(
            self.candidate_worktree, "scripts/aigp_replay_worker.py"
        )
        self._process = subprocess.Popen(
            [
                str(self.wrapper),
                "--",
                sys.executable,
                "-I",
                str(worker),
                spec,
                str(seed),
                processor_source_relative,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(self.candidate_worktree),
            shell=False,
            bufsize=1,
            env=worker_environment,
            **process_group,
        )
        self._call_lock = threading.Lock()
        self._stderr_lock = threading.Lock()
        self._stderr_tail = ""
        self._request_id = 0
        self._closed = False
        self._protocol_failed = False
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="aigp-replay-candidate-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    return
                with self._stderr_lock:
                    self._stderr_tail = (self._stderr_tail + chunk)[-32_768:]
        except (OSError, ValueError):
            return

    def _stderr_snapshot(self) -> str:
        with self._stderr_lock:
            return self._stderr_tail

    def _terminate_worker_tree(self) -> None:
        if self._process.poll() is not None:
            return
        containment_error: Optional[BaseException] = None
        try:
            if os.name == "nt":
                killed = subprocess.run(
                    [
                        "taskkill",
                        "/PID",
                        str(self._process.pid),
                        "/T",
                        "/F",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    shell=False,
                )
                if killed.returncode != 0 and self._process.poll() is None:
                    raise RuntimeError(
                        "taskkill could not confirm isolated worker tree termination"
                    )
            else:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            containment_error = exc
            if self._process.poll() is None:
                self._process.kill()
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("isolated worker did not terminate") from exc
        if containment_error is not None:
            raise RuntimeError(
                "isolated worker parent was killed, but process-tree termination "
                "could not be confirmed"
            ) from containment_error

    def _fail_protocol(self, message: str) -> RuntimeError:
        self._protocol_failed = True
        try:
            self._terminate_worker_tree()
        except RuntimeError as exc:
            return RuntimeError(f"{message}; containment failure: {exc}")
        return RuntimeError(message)

    def __call__(self, image: np.ndarray, context: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._call_lock:
            if self._closed:
                raise RuntimeError("isolated candidate processor is closed")
            if self._process.poll() is not None:
                raise RuntimeError("isolated candidate processor exited unexpectedly")
            buffer = io.BytesIO()
            np.save(buffer, image, allow_pickle=False)
            request_id = self._request_id
            self._request_id += 1
            request = {
                "schema": "aigp-replay-worker-request/1",
                "request_id": request_id,
                "image_npy_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "context": _json_safe(context),
            }
            stdin = self._process.stdin
            stdout = self._process.stdout
            assert stdin is not None and stdout is not None
            deadline = time.monotonic() + self.response_timeout_s
            write_queue: queue.Queue[Optional[BaseException]] = queue.Queue(maxsize=1)

            def write_request() -> None:
                try:
                    stdin.write(canonical_json(request) + "\n")
                    stdin.flush()
                    write_queue.put(None)
                except BaseException as exc:  # surfaced on the trusted caller thread
                    write_queue.put(exc)

            writer_thread = threading.Thread(
                target=write_request,
                name=f"aigp-replay-request-{request_id}",
                daemon=True,
            )
            writer_thread.start()
            try:
                write_error = write_queue.get(
                    timeout=max(0.0, deadline - time.monotonic())
                )
            except queue.Empty as exc:
                raise self._fail_protocol(
                    "isolated candidate processor request deadline exceeded"
                ) from exc
            if write_error is not None:
                raise self._fail_protocol(
                    "isolated candidate processor request pipe failed"
                ) from write_error

            response_queue: queue.Queue[tuple[Optional[str], Optional[BaseException]]] = (
                queue.Queue(maxsize=1)
            )

            def read_response() -> None:
                try:
                    line = stdout.readline(self.max_response_bytes + 1)
                    response_queue.put((line, None))
                except BaseException as exc:  # surfaced on the trusted caller thread
                    response_queue.put((None, exc))

            reader_thread = threading.Thread(
                target=read_response,
                name=f"aigp-replay-response-{request_id}",
                daemon=True,
            )
            reader_thread.start()
            try:
                response_line, read_error = response_queue.get(
                    timeout=max(0.0, deadline - time.monotonic())
                )
            except queue.Empty as exc:
                raise self._fail_protocol(
                    "isolated candidate processor response deadline exceeded"
                ) from exc
            if read_error is not None:
                raise self._fail_protocol(
                    "isolated candidate processor response pipe failed"
                ) from read_error
            assert response_line is not None
            if not response_line:
                raise self._fail_protocol(
                    "isolated candidate processor returned no response"
                )
            if (
                not response_line.endswith("\n")
                or len(response_line.encode("utf-8")) > self.max_response_bytes
            ):
                raise self._fail_protocol(
                    "isolated candidate processor response exceeded protocol limit"
                )
            try:
                response = strict_json_loads(response_line)
            except (TypeError, ValueError) as exc:
                raise self._fail_protocol(
                    "isolated candidate processor response is not strict JSON"
                ) from exc
            if (
                type(response) is not dict
                or set(response) != {"schema", "request_id", "result"}
                or response.get("request_id") != request_id
            ):
                raise self._fail_protocol(
                    "isolated candidate processor response is malformed or stale"
                )
            if response["schema"] != "aigp-replay-worker-response/1":
                raise self._fail_protocol(
                    "isolated candidate processor response schema is invalid"
                )
            return response["result"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        if self._protocol_failed:
            if self._process.poll() is None:
                self._terminate_worker_tree()
            self._stderr_thread.join(timeout=1.0)
            return
        try:
            self._process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._protocol_failed = True
            self._terminate_worker_tree()
            raise RuntimeError("isolated candidate worker ignored protocol shutdown")
        self._stderr_thread.join(timeout=1.0)
        if self._process.returncode != 0:
            raise RuntimeError(
                f"isolated candidate worker failed: {self._stderr_snapshot()[-1000:]}"
            )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _validate_processor_detections(value: Any) -> None:
    if type(value) is not list:
        raise TypeError("processor detections must be a list")
    required = {"center_px", "selector_eligible"}
    allowed = required | {"corners_px", "confidence"}
    for detection in value:
        if (
            type(detection) is not dict
            or not required <= set(detection)
            or set(detection) - allowed
            or not _exact_finite_vector(detection["center_px"], (2,))
            or type(detection["selector_eligible"]) is not bool
        ):
            raise ValueError("processor detection has an invalid exact schema")
        if "corners_px" in detection and not _exact_finite_vector(
            detection["corners_px"], (4, 2)
        ):
            raise ValueError("processor detection corners are invalid")
        if "confidence" in detection and not _exact_finite_number(
            detection["confidence"]
        ):
            raise ValueError("processor detection confidence is invalid")


def _validate_processor_tracker(value: Any) -> None:
    if value is None:
        return
    if type(value) is not dict or set(value) != {"target"}:
        raise ValueError("processor tracker must contain exactly target")
    target = value["target"]
    if target is None:
        return
    if (
        type(target) is not dict
        or set(target) != {"center_px"}
        or not _exact_finite_vector(target["center_px"], (2,))
    ):
        raise ValueError("processor tracker target has an invalid exact schema")


def _validate_processor_estimator(value: Any) -> None:
    if not _valid_estimator(value):
        raise ValueError(
            "processor estimator must contain exact healthy/rpy_rad/body_rates fields"
        )


def _validate_processor_command(value: Any) -> None:
    if _command_values(value) is None:
        raise ValueError(
            "processor generated_command must contain exact finite rate/thrust fields"
        )


def _sanitize_candidate_imu(value: Any) -> Optional[Dict[str, Any]]:
    """Copy only raw HIGHRES_IMU fields into candidate-visible context."""

    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("candidate IMU context must be an exact object")
    result: Dict[str, Any] = {}
    if "timestamp_us" in value:
        timestamp = value["timestamp_us"]
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError("candidate IMU timestamp must be a non-negative integer")
        result["timestamp_us"] = timestamp
    for name in ("accel", "gyro"):
        if name in value:
            if not _exact_finite_vector(value[name], (3,)):
                raise ValueError(f"candidate IMU {name} must be a finite 3-vector")
            result[name] = list(value[name])
    if "mag" in value:
        magnetic = value["mag"]
        if magnetic is not None and not _exact_finite_vector(magnetic, (3,)):
            raise ValueError("candidate IMU mag must be null or a finite 3-vector")
        result["mag"] = None if magnetic is None else list(magnetic)
    return result


def _sanitize_candidate_race_status(value: Any) -> Optional[Dict[str, int]]:
    """Copy only fields decoded from the authoritative race-status packet."""

    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("candidate race status must be an exact object")
    result: Dict[str, int] = {}
    for name in (
        "sim_boot_time_ms",
        "race_start_boot_time_ms",
        "race_finish_time_ns",
        "active_gate_index",
        "last_gate_race_time",
    ):
        if name in value:
            field = value[name]
            if type(field) is not int:
                raise ValueError(f"candidate race status {name} must be an exact integer")
            result[name] = field
    return result


def _sanitize_candidate_sensor_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Allowlist one raw sensor record; arbitrary event aliases never pass."""

    record_type = record.get("type")
    sequence = record.get("sequence")
    if record_type not in {"imu", "race_status"} or type(sequence) is not int:
        raise ValueError("candidate sensor record type/sequence is invalid")
    result: Dict[str, Any] = {"type": record_type, "sequence": sequence}
    received = record.get("received_monotonic_s")
    if received is not None:
        if not _exact_finite_number(received) or received < 0.0:
            raise ValueError("candidate sensor receive time must be finite and non-negative")
        result["received_monotonic_s"] = received
    if record_type == "imu":
        result["imu"] = _sanitize_candidate_imu(record.get("imu"))
    else:
        result["race_status"] = _sanitize_candidate_race_status(
            record.get("race_status")
        )
    return result


def process_frames(
    reader: ReplayBundleReader,
    records: Sequence[Mapping[str, Any]],
    processor_spec: str,
    processor_callable: Optional[
        Callable[[np.ndarray, Mapping[str, Any]], Mapping[str, Any]]
    ] = None,
    seed: int = 0,
) -> list[Dict[str, Any]]:
    """Rerun the ordered full stack on every published decoded frame.

    The callable receives a deep-immutable context containing only allowlisted
    raw IMU and authoritative race-status fields. Recorded events and
    detector/tracker/estimator/command outputs are never candidate-visible.
    Enforced latency is always measured by this evaluator; candidate-reported
    timing is forbidden.
    """

    if type(seed) is not int:
        raise ValueError("replay processor seed must be an exact integer")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    processor = processor_callable or _load_processor(processor_spec)
    # Materialize and order the source once.  The causal event cursor below is
    # monotonic, so replay work is O(records + frames), not O(records * frames).
    ordered_records = sorted(records, key=lambda row: int(row.get("sequence", -1)))
    processed_by_token = {
        (
            row.get("generation"),
            row.get("frame_id"),
            row.get("sim_time_ns"),
        ): row
        for row in ordered_records
        if row.get("type") == "frame"
    }
    decoded = [row for row in ordered_records if row.get("type") == "decoded_frame"]
    raw_feed = decoded or [
        row for row in ordered_records if row.get("type") == "frame"
    ]
    feed: list[Mapping[str, Any]] = []
    for source in raw_feed:
        token = (
            source.get("generation"),
            source.get("frame_id"),
            source.get("sim_time_ns"),
        )
        matching_control = processed_by_token.get(token)
        if matching_control is not None:
            if int(matching_control.get("sequence", -1)) < int(
                source.get("sequence", -1)
            ):
                raise ValueError("processed frame precedes its decoded image")
        feed.append(source)
    output: list[Dict[str, Any]] = [
        dict(row)
        for row in ordered_records
        if row.get("type") not in {"frame", "decoded_frame"}
    ]
    causal_sensor_records = [
        row
        for row in ordered_records
        if row.get("type") in {"imu", "race_status"}
    ]
    event_cursor = 0
    prior_sequence = -1
    latest_imu: Any = None
    latest_race_status: Any = None
    latest_phase: Any = None
    for source in feed:
        token = (
            source.get("generation"),
            source.get("frame_id"),
            source.get("sim_time_ns"),
        )
        source_sequence = int(source.get("sequence", -1))
        sensor_events = []
        while event_cursor < len(causal_sensor_records):
            event = causal_sensor_records[event_cursor]
            sequence = int(event.get("sequence", -1))
            # Candidate state is causal to decoded-frame publication, not to
            # a possibly later processed-frame record.  A delayed A control
            # record must never reorder A behind a subsequently decoded B or
            # leak B-era sensors into A.
            if sequence > source_sequence:
                break
            event_cursor += 1
            if prior_sequence < sequence:
                sanitized = _sanitize_candidate_sensor_record(event)
                sensor_events.append(sanitized)
                if event.get("type") == "imu":
                    latest_imu = sanitized["imu"]
                else:
                    latest_race_status = sanitized["race_status"]
        if source.get("type") == "frame":
            latest_imu = _sanitize_candidate_imu(source.get("imu"))
            latest_race_status = _sanitize_candidate_race_status(
                source.get("race_status")
            )
            latest_phase = source.get("phase")
        prior_sequence = max(prior_sequence, source_sequence)
        context = _deep_freeze(
            _json_safe(
                {
                    "schema": "aigp-vq2-full-stack-context/1",
                    "session_id": source.get("session_id"),
                    "decoded_sequence": source_sequence,
                    "generation": source.get("generation"),
                    "frame_id": source.get("frame_id"),
                    "sim_time_ns": source.get("sim_time_ns"),
                    "received_monotonic_s": source.get("received_monotonic_s"),
                    "seed": seed,
                    "imu": latest_imu,
                    "race_status": latest_race_status,
                    "sensor_events": sensor_events,
                }
            )
        )
        image = reader.load_frame(source)
        started = time.perf_counter_ns()
        derived = processor(image, context)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        expected_processor_fields = {
            "detections",
            "tracker",
            "estimator",
            "generated_command",
        }
        if type(derived) is not dict or set(derived) != expected_processor_fields:
            raise TypeError(
                "full-stack processor must return exactly detections, tracker, "
                "estimator, and generated_command"
            )
        _validate_processor_detections(derived["detections"])
        _validate_processor_tracker(derived["tracker"])
        _validate_processor_estimator(derived["estimator"])
        _validate_processor_command(derived["generated_command"])
        row = {
            "schema": RECORD_SCHEMA,
            "session_id": source.get("session_id"),
            "sequence": source_sequence,
            "type": "frame",
            "generation": source.get("generation"),
            "frame_id": source.get("frame_id"),
            "sim_time_ns": source.get("sim_time_ns"),
            "received_monotonic_s": source.get("received_monotonic_s"),
            "detector_latency_ms": None,
            "full_stack_latency_ms": elapsed_ms,
            "detections": _json_safe(derived["detections"]),
            "tracker": _json_safe(derived["tracker"]),
            # Recorded state is retained only for explicitly labeled
            # informational scoring, never supplied to or claimed from the
            # detector processor.
            "imu": latest_imu,
            "estimator": _json_safe(derived["estimator"]),
            "race_status": latest_race_status,
            "generated_command": _json_safe(derived["generated_command"]),
            "sent_command": None,
            "phase": latest_phase,
        }
        output.append(row)
    output.sort(key=lambda row: int(row.get("sequence", -1)))
    return output


def score_bundle(
    bundle: Path | str,
    *,
    annotations_path: Optional[Path | str] = None,
    processor_spec: Optional[str] = None,
    max_center_error_px: float = 80.0,
    isolation_wrapper: Optional[Path | str] = None,
    isolation_wrapper_sha256: Optional[str] = None,
    candidate_worktree: Optional[Path | str] = None,
) -> Dict[str, Any]:
    reader = ReplayBundleReader(bundle)
    _verification, records = reader.verify_and_read(verify_frames=True)
    if annotations_path is None:
        annotations = {}
        annotations_sha256 = None
    else:
        annotation_payload = _read_bounded_secure_file(
            annotations_path,
            maximum_bytes=MAX_REPLAY_ANNOTATIONS_BYTES,
            label="replay annotations",
        )
        annotations = load_annotations_bytes(annotation_payload)
        annotations_sha256 = sha256_bytes(annotation_payload)
    metadata = reader.manifest.get("metadata")
    recorded_seed = metadata.get("seed") if isinstance(metadata, Mapping) else None
    evaluator_seed = recorded_seed if type(recorded_seed) is int else 0
    isolated: Optional[IsolatedReplayProcessor] = None
    if processor_spec:
        processor_code_hash: Optional[str] = _processor_code_hash(
            processor_spec,
            candidate_worktree,
            require_pristine=isolation_wrapper is not None,
        )
        try:
            if isolation_wrapper is not None:
                if isolation_wrapper_sha256 is None:
                    raise ValueError("isolation wrapper SHA-256 is required")
                isolated = IsolatedReplayProcessor(
                    processor_spec,
                    isolation_wrapper,
                    isolation_wrapper_sha256,
                    seed=evaluator_seed,
                    candidate_worktree=candidate_worktree,
                )
            records = process_frames(
                reader,
                records,
                processor_spec,
                processor_callable=isolated,
                seed=evaluator_seed,
            )
        finally:
            if isolated is not None:
                isolated.close()
    else:
        recorded_hash = metadata.get("code_hash") if isinstance(metadata, Mapping) else None
        processor_code_hash = (
            recorded_hash
            if type(recorded_hash) is str and recorded_hash
            else None
        )
    score = score_records(
        records,
        session_id=reader.session_id,
        annotations=annotations,
        max_center_error_px=max_center_error_px,
    )
    score["dataset_hash"] = reader.dataset_hash
    score["annotations_sha256"] = annotations_sha256
    score["processor"] = processor_spec or "recorded"
    score["processor_code_sha256"] = processor_code_hash
    score["candidate_isolation"] = (
        dict(isolated.attestation)
        if processor_spec and isolated is not None
        else {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "unproved",
            "filesystem": "unproved",
            "non_interactive": False,
            "process_tree_containment": "unproved",
            "host_process_access": "unproved",
            "test_only_in_process": True,
        }
    )
    score["isolation_wrapper_sha256"] = (
        isolation_wrapper_sha256 if isolated is not None else "0" * 64
    )
    score["domain_provenance"] = {
        "perception": (
            "candidate_detector_on_all_decoded_frames"
            if processor_spec
            else "recorded_processed_frames"
        ),
        "estimator": "recorded_bundle_context",
        "open_loop_commands": "recorded_bundle_command_stream",
    }
    if processor_spec:
        score["domain_provenance"].update(
            {
                "estimator": "candidate_estimator_on_ordered_sanitized_stream",
                "open_loop_commands": "candidate_generator_on_ordered_sanitized_stream",
            }
        )
        if isolated is not None:
            score["domain_provenance"][
                "worker_transport"
            ] = "candidate_worktree_code_hash"
    score["seed"] = evaluator_seed
    score["evaluator_config_sha256"] = json_hash(
        {
            "schema": "aigp-vq2-replay-evaluator-config/1",
            "max_center_error_px": float(max_center_error_px),
            "processor_contract": "ordered-full-stack-on-all-decoded-frames/1",
            "seed": evaluator_seed,
            "isolation_wrapper_sha256": score["isolation_wrapper_sha256"],
        }
    )
    evaluator_root = Path(__file__).resolve().parents[1]
    evaluator_sources = {}
    for relative in (
        "aigp_loop/__init__.py",
        "aigp_loop/_util.py",
        "aigp_loop/evidence.py",
        "aigp_loop/ledger.py",
        "aigp_loop/promotion.py",
        "aigp_loop/replay.py",
        "scripts/aigp_replay.py",
    ):
        target = secure_relative_regular_file(evaluator_root, relative)
        evaluator_sources[relative] = sha256_bytes(
            read_secure_regular_file(target)
        )
    evaluator_identity = {
        "schema": "aigp-vq2-replay-evaluator-identity/2",
        "sources_sha256": evaluator_sources,
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "byteorder": sys.byteorder,
        },
    }
    score["evaluator_identity"] = evaluator_identity
    score["evaluator_source_sha256"] = json_hash(evaluator_identity)
    score["evaluation_config_sha256"] = score["evaluator_config_sha256"]
    score["evaluator_version"] = (
        "aigp-vq2-replay/1:" + score["evaluator_source_sha256"]
    )
    score["repetitions"] = 1
    score["score_payload_hash"] = json_hash(score)
    return score


def evaluation_input_hash(
    score: Mapping[str, Any], policy_result: Mapping[str, Any]
) -> str:
    """Hash deterministic replay inputs, never timing-bearing score output."""

    names = (
        "dataset_hash",
        "annotations_sha256",
        "policy_hash",
        "processor",
        "processor_code_sha256",
        "evaluator_config_sha256",
        "evaluator_source_sha256",
        "isolation_wrapper_sha256",
    )
    values = (
        score.get("dataset_hash"),
        score.get("annotations_sha256"),
        policy_result.get("policy_hash"),
        score.get("processor"),
        score.get("processor_code_sha256"),
        score.get("evaluator_config_sha256"),
        score.get("evaluator_source_sha256"),
        score.get("isolation_wrapper_sha256"),
    )
    if any(type(value) is not str or not value for value in values):
        raise ValueError(
            "evaluation input requires bundle, labels, policy, processor, and evaluator provenance"
        )
    for name, value in zip(names, values):
        if name != "processor" and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"evaluation input {name} is not a SHA-256 digest")
    return json_hash(dict(zip(names, values)))


def evaluation_result_hash(
    score: Mapping[str, Any], policy_result: Mapping[str, Any]
) -> str:
    """Hash the derived result separately from deterministic input identity."""

    sanitized_score = {
        key: value
        for key, value in score.items()
        if key
        not in {
            "evaluation_evidence_hash",
            "evaluation_input_hash",
            "evaluation_result_hash",
            "policy",
        }
    }
    return json_hash({"score": sanitized_score, "policy": policy_result})


def evaluation_evidence_hash(
    score: Mapping[str, Any], policy_result: Mapping[str, Any]
) -> str:
    """Backward-compatible evidence hash.

    New scores return the deterministic input hash.  Older callers lacking
    versioned evaluator provenance retain the original result-bound hash so
    historical evidence is never silently reinterpreted.
    """

    if (
        score.get("evaluator_config_sha256") is not None
        or score.get("evaluator_source_sha256") is not None
    ):
        return evaluation_input_hash(score, policy_result)

    required = (
        score.get("dataset_hash"),
        score.get("annotations_sha256"),
        policy_result.get("policy_hash"),
        score.get("processor"),
        score.get("processor_code_sha256"),
        score.get("score_payload_hash"),
    )
    if any(type(value) is not str or not value for value in required):
        raise ValueError(
            "evaluation evidence requires dataset, annotations, policy, processor code, and score provenance"
        )
    for name, value in zip(
        (
            "dataset_hash",
            "annotations_sha256",
            "policy_hash",
            "processor",
            "processor_code_sha256",
            "score_payload_hash",
        ),
        required,
    ):
        if name != "processor" and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"evaluation evidence {name} is not a SHA-256 digest")
    return json_hash(
        {
            "bundle_dataset_hash": required[0],
            "annotations_sha256": required[1],
            "policy_hash": required[2],
            "processor": required[3],
            "processor_code_sha256": required[4],
            "score_payload_hash": required[5],
        }
    )


def _metric_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def evaluate_score_policy(
    score: Mapping[str, Any], policy: Mapping[str, Any]
) -> Dict[str, Any]:
    """Apply a versioned fail-closed golden-replay metric policy.

    Policy ``metrics`` entries are dotted score paths with optional numeric
    ``min``/``max`` constraints.  Missing, null, boolean, non-numeric, or
    non-finite evidence is always a violation.
    """

    if type(policy) is not dict:
        raise ValueError("replay policy must be an object")
    if set(policy) != {"schema", "metrics"}:
        raise ValueError("replay policy requires exactly schema and metrics")
    if policy["schema"] != "aigp-vq2-replay-policy/1":
        raise ValueError("unsupported replay policy schema")
    constraints = policy["metrics"]
    if type(constraints) is not dict or not constraints:
        raise ValueError("replay policy requires a non-empty metrics object")
    violations: list[Dict[str, Any]] = []
    observed: Dict[str, Any] = {}
    for path, bounds in sorted(constraints.items()):
        if (
            type(path) is not str
            or not path
            or path.strip() != path
            or any(not component for component in path.split("."))
            or type(bounds) is not dict
        ):
            raise ValueError("policy metric paths and bounds must be exact objects")
        unknown = set(bounds) - {"min", "max"}
        if unknown or not bounds:
            raise ValueError(f"invalid bounds for {path}: {sorted(unknown)}")
        raw = _metric_path(score, path)
        observed[path] = raw
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            violations.append({"metric": path, "reason": "missing_or_nonfinite", "observed": raw})
            continue
        value = float(raw)
        if "min" in bounds:
            minimum = bounds["min"]
            if not _exact_finite_number(minimum):
                raise ValueError(f"non-numeric min policy bound for {path}")
            if value < float(minimum):
                violations.append(
                    {"metric": path, "reason": "below_min", "observed": value, "limit": float(minimum)}
                )
        if "max" in bounds:
            maximum = bounds["max"]
            if not _exact_finite_number(maximum):
                raise ValueError(f"non-numeric max policy bound for {path}")
            if value > float(maximum):
                violations.append(
                    {"metric": path, "reason": "above_max", "observed": value, "limit": float(maximum)}
                )
    return {
        "schema": "aigp-vq2-replay-policy-result/1",
        "policy_hash": json_hash(policy),
        "passed": not violations,
        "constraints": {path: dict(bounds) for path, bounds in sorted(constraints.items())},
        "observed": observed,
        "violations": violations,
    }


def _corpus_member_path(manifest_path: Path, value: Any, field: str) -> Path:
    if type(value) is not str or not value.strip():
        raise ValueError(f"corpus {field} must be a non-empty path string")
    path = Path(value)
    lexical = path if path.is_absolute() else manifest_path.parent / path
    try:
        return (
            secure_directory(lexical)
            if field == "bundle"
            else secure_regular_file(lexical)
        )
    except ValueError as exc:
        raise ValueError(f"corpus {field} is not a secure existing path") from exc


def score_corpus(
    manifest: Path | str,
    *,
    processor_spec: str,
    max_center_error_px: float = 80.0,
    isolation_wrapper: Optional[Path | str] = None,
    isolation_wrapper_sha256: Optional[str] = None,
    candidate_worktree: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Score a frozen, multi-session golden corpus with per-session policy."""

    if (
        type(max_center_error_px) not in {int, float}
        or not math.isfinite(max_center_error_px)
        or max_center_error_px <= 0.0
    ):
        raise ValueError("max_center_error_px must be finite and positive")

    manifest_path = secure_regular_file(manifest)
    manifest_payload = _read_bounded_secure_file(
        manifest_path,
        maximum_bytes=MAX_REPLAY_CORPUS_MANIFEST_BYTES,
        label="replay corpus manifest",
    )
    manifest_sha256 = sha256_bytes(manifest_payload)
    try:
        payload = strict_json_loads(manifest_payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("replay corpus manifest must be UTF-8") from exc
    if (
        type(payload) is not dict
        or set(payload) != {"schema", "sessions"}
        or payload.get("schema") != CORPUS_SCHEMA
        or type(payload.get("sessions")) is not list
        or not payload["sessions"]
        or len(payload["sessions"]) > MAX_REPLAY_CORPUS_SESSION_COUNT
    ):
        raise ValueError("replay corpus manifest has an invalid exact schema")
    expected_member_fields = {"session_id", "bundle", "annotations", "policy"}
    session_scores: list[Dict[str, Any]] = []
    deterministic_inputs: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for member in payload["sessions"]:
        if type(member) is not dict or set(member) != expected_member_fields:
            raise ValueError("corpus session has missing or unknown fields")
        session_id = member["session_id"]
        if type(session_id) is not str or not session_id.strip() or session_id in seen:
            raise ValueError("corpus session IDs must be unique non-empty strings")
        seen.add(session_id)
        bundle = _corpus_member_path(manifest_path, member["bundle"], "bundle")
        annotations = _corpus_member_path(
            manifest_path, member["annotations"], "annotations"
        )
        policy_path = _corpus_member_path(manifest_path, member["policy"], "policy")
        policy_payload = _read_bounded_secure_file(
            policy_path,
            maximum_bytes=MAX_REPLAY_POLICY_BYTES,
            label="replay corpus policy",
        )
        policy_file_sha256 = sha256_bytes(policy_payload)
        try:
            policy = strict_json_loads(policy_payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("replay corpus policy must be UTF-8") from exc
        score = score_bundle(
            bundle,
            annotations_path=annotations,
            processor_spec=processor_spec,
            max_center_error_px=max_center_error_px,
            isolation_wrapper=isolation_wrapper,
            isolation_wrapper_sha256=isolation_wrapper_sha256,
            candidate_worktree=candidate_worktree,
        )
        if score.get("session_id") != session_id:
            raise ValueError(
                f"corpus session identity mismatch: expected {session_id}, "
                f"bundle contains {score.get('session_id')}"
            )
        policy_result = evaluate_score_policy(score, policy)
        policy_result["policy_file_sha256"] = policy_file_sha256
        input_hash = evaluation_input_hash(score, policy_result)
        result_hash = evaluation_result_hash(score, policy_result)
        score["policy"] = policy_result
        score["evaluation_input_hash"] = input_hash
        score["evaluation_evidence_hash"] = input_hash
        score["evaluation_result_hash"] = result_hash
        session_scores.append(score)
        deterministic_inputs.append(
            {"session_id": session_id, "evaluation_input_hash": input_hash}
        )

    def exact_metric(score: Mapping[str, Any], path: str) -> float:
        value = _metric_path(score, path)
        if not _exact_finite_number(value):
            raise ValueError(f"corpus aggregate metric is missing/nonfinite: {path}")
        return float(value)

    labeled_frames = sum(int(score["labeled_frames"]) for score in session_scores)
    truth_count = sum(
        int(score["perception"]["gate_truth_count"]) for score in session_scores
    )
    matches = sum(
        int(score["perception"]["gate_matches"]) for score in session_scores
    )
    false_positives = sum(
        int(score["perception"]["false_positives"]) for score in session_scores
    )
    aggregate = {
        "session_count": len(session_scores),
        "labeled_frames": labeled_frames,
        "gate_truth_count": truth_count,
        "gate_matches": matches,
        "gate_recall": matches / truth_count if truth_count else None,
        "false_positives_per_frame": (
            false_positives / labeled_frames if labeled_frames else None
        ),
        "worst_center_error_px_p95": max(
            exact_metric(score, "perception.center_error_px_p95")
            for score in session_scores
        ),
        "worst_temporal_center_step_px_p95": max(
            exact_metric(score, "perception.temporal_center_step_px_p95")
            for score in session_scores
        ),
        "worst_full_stack_latency_ms_p95": max(
            exact_metric(score, "perception.full_stack_latency_ms_p95")
            for score in session_scores
        ),
    }
    passed = all(score["policy"]["passed"] is True for score in session_scores)
    session_seeds = {score["seed"] for score in session_scores}
    if len(session_seeds) != 1:
        raise ValueError("golden corpus sessions must share one frozen evaluator seed")
    processor_hashes = {score["processor_code_sha256"] for score in session_scores}
    isolation_attestations = {
        canonical_json(score["candidate_isolation"]) for score in session_scores
    }
    if len(processor_hashes) != 1 or len(isolation_attestations) != 1:
        raise ValueError(
            "golden corpus sessions must share one processor and isolation identity"
        )
    combined_input_hash = json_hash(
        {
            "schema": CORPUS_SCHEMA,
            "manifest_sha256": manifest_sha256,
            "processor": processor_spec,
            "sessions": deterministic_inputs,
        }
    )
    result: Dict[str, Any] = {
        "schema": "aigp-vq2-replay-corpus-score/1",
        "corpus_manifest_sha256": manifest_sha256,
        "processor": processor_spec,
        "processor_code_sha256": session_scores[0]["processor_code_sha256"],
        "domain_provenance": {
            "perception": "candidate_detector_on_all_decoded_frames",
            "estimator": "candidate_estimator_on_ordered_sanitized_stream",
            "open_loop_commands": "candidate_generator_on_ordered_sanitized_stream",
        },
        "candidate_isolation": dict(session_scores[0]["candidate_isolation"]),
        "sessions": session_scores,
        "aggregate": aggregate,
        "policy": {
            "schema": "aigp-vq2-replay-corpus-policy-result/1",
            "passed": passed,
            "session_results": [
                {
                    "session_id": score["session_id"],
                    "passed": score["policy"]["passed"],
                    "policy_hash": score["policy"]["policy_hash"],
                    "violations": score["policy"]["violations"],
                }
                for score in session_scores
            ],
        },
        "evaluation_input_hash": combined_input_hash,
        "evaluation_evidence_hash": combined_input_hash,
        "evaluation_config_sha256": json_hash(
            {
                "schema": CORPUS_SCHEMA,
                "manifest_sha256": manifest_sha256,
                "processor": processor_spec,
                "max_center_error_px": float(max_center_error_px),
            }
        ),
        "evaluator_version": session_scores[0]["evaluator_version"],
        "repetitions": len(session_scores),
        "seed": session_scores[0]["seed"],
    }
    result["evaluation_result_hash"] = json_hash(result)
    result["artifact_hashes"] = {
        "corpus_manifest": result["corpus_manifest_sha256"],
        "evaluation_input": combined_input_hash,
        "evaluation_result": result["evaluation_result_hash"],
    }
    return result
