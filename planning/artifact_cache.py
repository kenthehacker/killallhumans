"""Content-addressed, corruption-safe storage for deterministic artifacts.

The benchmark and planner use this module instead of process-global, mutable
"last result" files.  Every artifact is addressed by a digest of its resolved
inputs, algorithm/schema version, relevant source, and numerical environment.

The implementation intentionally has no third-party locking dependency.  A
lock is one small file per artifact key, so unrelated tracks never serialize
behind one global mutex.  Payloads are written beside their destination and
published with :func:`os.replace`, which is atomic on the supported local
filesystems (including Windows/NTFS).
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

import numpy as np


ARTIFACT_ENVELOPE_VERSION = 1
CACHE_ROOT_ENV = "AIGP_CACHE_ROOT"
_REPO = Path(__file__).resolve().parent.parent
NUMERIC_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)


def _valid_npz_array_name(value: Any) -> bool:
    return bool(
        type(value) is str
        and value
        and value.isascii()
        and (value[0].isalpha())
        and all(character.isalnum() or character == "_" for character in value)
    )


def _try_lock_file(handle) -> bool:
    """Take the published lock inode exclusively without blocking."""

    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock_file(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _open_lock_rendezvous(path: Path):
    """Open/create one persistent, regular, non-symlink lock inode."""

    flags = os.O_CREAT | os.O_RDWR
    for optional_flag in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, optional_flag, 0))
    fd = os.open(path, flags, 0o600)
    handle = None
    try:
        opened = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(opened, named):
            raise ValueError(f"artifact lock path is not a regular file: {path}")
        handle = os.fdopen(fd, "r+b")
        fd = -1
        # msvcrt.locking requires the locked byte to exist. Concurrent first
        # openers may both write this identical sentinel; ownership is the OS
        # byte/flock lease, never file contents or existence.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        return handle
    except Exception:
        if handle is not None:
            handle.close()
        elif fd >= 0:
            os.close(fd)
        raise


def default_cache_root() -> Path:
    """Return the cache root, honoring the test/worker override variable."""

    configured = os.environ.get(CACHE_ROOT_ENV)
    return Path(configured).expanduser().resolve() if configured else _REPO / ".cache"


def _canonicalize(value: Any) -> Any:
    """Convert supported values to a stable, strict JSON representation."""

    if dataclasses.is_dataclass(value):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist())
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        invalid_keys = [key for key in value if type(key) is not str]
        if invalid_keys:
            rendered = ", ".join(
                f"{key!r} ({type(key).__name__})" for key in invalid_keys[:3]
            )
            raise TypeError(
                "cache-key mappings require exact string keys; got " + rendered
            )
        return {
            key: _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonicalize(item) for item in value), key=repr)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not cache-key safe: {value!r}")
        # Do not round here. A one-bit numeric config change must invalidate.
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported cache-key value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json_loads(payload: str) -> Any:
    """Decode cache metadata without duplicate keys or JSON extensions."""

    def unique_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in artifact: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON numeric constant: {value}")

    return json.loads(
        payload,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def source_digest(paths: Iterable[Path]) -> str:
    """Hash source contents and repository-relative identities."""

    digest = hashlib.sha256()
    resolved = sorted((Path(path).resolve() for path in paths), key=lambda p: str(p))
    for path in resolved:
        try:
            label = path.relative_to(_REPO).as_posix()
        except ValueError:
            label = path.name
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            # Missing relevant source is itself stable key material. The build
            # may subsequently fail, but it cannot alias an artifact produced
            # while that source existed.
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def dependency_fingerprint() -> Dict[str, Any]:
    """Versions and platform fields known to affect numerical artifacts."""

    versions: Dict[str, Optional[str]] = {}
    for distribution in ("numpy", "scipy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    numpy_build: Dict[str, Any] = {}
    try:
        build = np.__config__.show(mode="dicts")
    except (TypeError, AttributeError):
        build = None
    if isinstance(build, dict):
        dependencies = build.get("Build Dependencies", {})
        numpy_build["blas_lapack"] = {
            name: {
                field: details.get(field)
                for field in ("name", "found", "version", "openblas configuration")
            }
            for name, details in sorted(dependencies.items())
            if name in {"blas", "lapack"} and isinstance(details, dict)
        }
        numpy_build["simd"] = build.get("SIMD Extensions", {})
        compilers = build.get("Compilers", {})
        numpy_build["compilers"] = {
            name: {
                "name": details.get("name"),
                "version": details.get("version"),
            }
            for name, details in sorted(compilers.items())
            if isinstance(details, dict)
        }
    else:
        # NumPy 1.x exposes get_info rather than the dict-form show(). Keep the
        # selected identity fields JSON-safe and deterministic.
        get_info = getattr(np.__config__, "get_info", None)
        if callable(get_info):
            for name in ("blas_opt_info", "lapack_opt_info"):
                info = get_info(name) or {}
                numpy_build[name] = {
                    key: _canonicalize(info[key])
                    for key in sorted(info)
                    if key in {"libraries", "library_dirs", "define_macros", "language"}
                }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "dependencies": versions,
        "numpy_build": numpy_build,
        "numeric_thread_environment": {
            name: os.environ.get(name) for name in NUMERIC_THREAD_ENV_VARS
        },
    }


def artifact_key(
    namespace: str,
    inputs: Any,
    *,
    schema_version: str,
    source_files: Iterable[Path] = (),
    environment: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build a full SHA-256 content address for one artifact layer."""

    material = {
        "namespace": namespace,
        "schema_version": schema_version,
        "inputs": inputs,
        "source_digest": source_digest((*tuple(source_files), Path(__file__))),
        "environment": dict(environment or dependency_fingerprint()),
    }
    return sha256_json(material)


def _arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.hasobject:
            raise TypeError(f"object arrays are forbidden in cached npz payloads: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(array.shape)))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


class ArtifactStore:
    """Versioned content-addressed artifact store rooted at one directory."""

    def __init__(self, root: Optional[os.PathLike[str] | str] = None):
        self.root = Path(root).expanduser().resolve() if root else default_cache_root()

    def path(self, namespace: str, key: str, suffix: str) -> Path:
        if not namespace or any(part in namespace for part in ("..", "\\", "/")):
            raise ValueError(f"invalid cache namespace: {namespace!r}")
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError(f"invalid artifact key: {key!r}")
        return self.root / namespace / f"{key}{suffix}"

    @staticmethod
    def _atomic_write(path: Path, writer) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".partial", dir=str(path.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                writer(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    @contextlib.contextmanager
    def lock(
        self,
        namespace: str,
        key: str,
        *,
        timeout_s: float = 900.0,
        stale_after_s: float = 3600.0,
    ) -> Iterator[None]:
        """Lease a persistent OS-locked rendezvous for ``namespace/key``.

        File existence and contents never represent ownership. The regular
        one-byte rendezvous remains in the cache permanently; ``msvcrt`` byte
        locking on Windows or ``flock`` on POSIX is the only lease. The OS
        releases that lease on process death, so there is no stale-owner/PID
        protocol, partial publication window, or unlink/replacement race.

        ``stale_after_s`` remains only for source compatibility with older
        callers and has no effect under this crash-safe lease protocol.
        """

        lock_path = self.path(namespace, key, ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(timeout_s, bool) or not isinstance(
            timeout_s, (int, float, np.integer, np.floating)
        ):
            raise TypeError("timeout_s must be a real number")
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s < 0.0:
            raise ValueError("timeout_s must be finite and non-negative")
        _ = stale_after_s
        started = time.monotonic()
        handle = _open_lock_rendezvous(lock_path)
        locked = False
        try:
            while True:
                if _try_lock_file(handle):
                    locked = True
                    break
                if time.monotonic() - started >= timeout_s:
                    raise TimeoutError(
                        f"timed out waiting for artifact lock {lock_path}"
                    )
                time.sleep(0.025)

            yield
        finally:
            if locked:
                with contextlib.suppress(OSError):
                    _unlock_file(handle)
            handle.close()

    def load_json(self, namespace: str, key: str) -> Optional[Any]:
        path = self.path(namespace, key, ".json")
        try:
            envelope = _strict_json_loads(path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict):
                return None
            if envelope.get("envelope_version") != ARTIFACT_ENVELOPE_VERSION:
                return None
            if envelope.get("namespace") != namespace or envelope.get("key") != key:
                return None
            payload = envelope["payload"]
            if envelope.get("payload_sha256") != sha256_json(payload):
                return None
            return payload
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # Fail closed: callers rebuild and atomically replace the bad file.
            return None

    def save_json(self, namespace: str, key: str, payload: Any) -> Path:
        canonical_payload = _canonicalize(payload)
        envelope = {
            "envelope_version": ARTIFACT_ENVELOPE_VERSION,
            "namespace": namespace,
            "key": key,
            "payload_sha256": sha256_json(canonical_payload),
            "payload": canonical_payload,
        }
        encoded = json.dumps(
            envelope, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        path = self.path(namespace, key, ".json")
        self._atomic_write(path, lambda handle: handle.write(encoded))
        return path

    def load_npz(self, namespace: str, key: str) -> Optional[Dict[str, np.ndarray]]:
        path = self.path(namespace, key, ".npz")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "__metadata__" not in archive.files:
                    return None
                payload_names = [
                    name for name in archive.files if name != "__metadata__"
                ]
                if len(payload_names) != len(set(payload_names)) or any(
                    not _valid_npz_array_name(name) for name in payload_names
                ):
                    return None
                metadata_bytes = np.asarray(archive["__metadata__"], dtype=np.uint8).tobytes()
                metadata = _strict_json_loads(metadata_bytes.decode("utf-8"))
                if not isinstance(metadata, dict):
                    return None
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "__metadata__"
                }
            if metadata.get("envelope_version") != ARTIFACT_ENVELOPE_VERSION:
                return None
            if metadata.get("namespace") != namespace or metadata.get("key") != key:
                return None
            if metadata.get("payload_sha256") != _arrays_digest(arrays):
                return None
            return arrays
        except (
            FileNotFoundError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            zipfile.BadZipFile,
        ):
            return None

    def save_npz(
        self, namespace: str, key: str, arrays: Mapping[str, np.ndarray]
    ) -> Path:
        if not isinstance(arrays, Mapping):
            raise TypeError("NPZ arrays must be a mapping")
        if any(type(name) is not str for name in arrays):
            raise TypeError("NPZ array names must be exact strings")
        if "__metadata__" in arrays:
            raise ValueError("NPZ array name '__metadata__' is reserved")
        invalid_names = [name for name in arrays if not _valid_npz_array_name(name)]
        if invalid_names:
            raise ValueError(
                "NPZ array names must use unambiguous ASCII identifiers"
            )
        normalized = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
        payload_sha256 = _arrays_digest(normalized)
        metadata = canonical_json_bytes({
            "envelope_version": ARTIFACT_ENVELOPE_VERSION,
            "namespace": namespace,
            "key": key,
            "payload_sha256": payload_sha256,
        })
        archive_arrays = dict(normalized)
        archive_arrays["__metadata__"] = np.frombuffer(metadata, dtype=np.uint8)
        path = self.path(namespace, key, ".npz")

        def _write(handle) -> None:
            np.savez_compressed(handle, **archive_arrays)

        self._atomic_write(path, _write)
        return path


__all__ = [
    "ARTIFACT_ENVELOPE_VERSION",
    "ArtifactStore",
    "CACHE_ROOT_ENV",
    "NUMERIC_THREAD_ENV_VARS",
    "artifact_key",
    "canonical_json_bytes",
    "default_cache_root",
    "dependency_fingerprint",
    "sha256_json",
    "source_digest",
]
