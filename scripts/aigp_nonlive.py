"""Run and adapt trusted synthetic T2-T4 promotion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional, Sequence

_REPO = Path(__file__).resolve().parent.parent
_FULL_TRACK_SET = (
    "aigp_default",
    "figure8",
    "grand_tour",
    "race_01",
    "slalom",
    "straight_hairpin",
    "vertical_cliff",
)
_DOMAIN_TRACK_SET = ("grand_tour", "slalom", "vertical_cliff")
_TRACK_CONFIG_FILES = frozenset(
    f"sim_pybullet/configs/{name}.json" for name in _FULL_TRACK_SET
)
_T4_TEST_ROOTS = (
    "tests",
    "competition/tests",
    "control/tests",
    "estimation/tests",
    "flight_control/tests",
    "gate_detection/tests",
    "gate_sequencing/tests",
    "planning/tests",
    "sim_pybullet/tests",
    "simulation/tests",
)
_FULL_SUITE_PYTEST_ARGS = (
    "-q",
    "-p",
    "pytest_timeout",
    "-p",
    "no:cacheprovider",
    "-o",
    "required_plugins=",
    "-c",
    "pyproject.toml",
    "-m",
    "not live",
    "--timeout=300",
    *_T4_TEST_ROOTS,
)
_STARTUP_CORE_FILES = frozenset(
    {
        "aigp_loop/__init__.py",
        "aigp_loop/_util.py",
        "aigp_loop/ledger.py",
        "aigp_loop/nonlive.py",
        "aigp_loop/promotion.py",
        "aigp_loop/scheduler.py",
        "planning/__init__.py",
        "planning/artifact_cache.py",
        "scripts/aigp_nonlive.py",
        "scripts/benchmark.py",
        "scripts/benchmark_matrix.py",
    }
) | _TRACK_CONFIG_FILES
_MAX_TRUSTED_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_TRUSTED_SOURCE_BYTES = 64 * 1024 * 1024
_RUNTIME_PYCACHE_CONTEXTS: list[tempfile.TemporaryDirectory[str]] = []
_NUMERIC_STARTUP_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENCV_FOR_THREADS_NUM": "1",
}


def _reject_executable_bytecode_boundary() -> None:
    """Reject adjacent bytecode before any delayed repository import."""

    ignored = {
        ".git",
        ".venv",
        ".aigp-loop",
        ".cache",
        ".loop",
        ".research_loop",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    for _directory, names, files in os.walk(
        _REPO, topdown=True, followlinks=False
    ):
        if any(name.casefold() == "__pycache__" for name in names) or any(
            name.casefold().endswith((".pyc", ".pyo")) for name in files
        ):
            raise ValueError("trusted repository contains executable bytecode/cache")
        ignored_folded = {name.casefold() for name in ignored}
        names[:] = [name for name in names if name.casefold() not in ignored_folded]


def _reject_startup_import_collisions() -> None:
    """Reject package/native-extension alternatives to every pinned module."""

    native_suffixes = (".pyd", ".so", ".dll", ".dylib")
    top_level_packages = {
        Path(relative).parts[0]
        for relative in _STARTUP_CORE_FILES
        if relative.endswith(".py")
    }
    for top_level in top_level_packages:
        package = _REPO / top_level
        pinned_initializer = f"{top_level}/__init__.py" in _STARTUP_CORE_FILES
        if not pinned_initializer:
            for initializer in package.iterdir():
                folded = initializer.name.casefold()
                if folded == "__init__.py" or (
                    folded.startswith("__init__.")
                    and folded.endswith(native_suffixes)
                ):
                    raise ValueError(
                        "untrusted import-boundary collision: "
                        + initializer.relative_to(_REPO).as_posix()
                    )
        for sibling in _REPO.iterdir():
            folded = sibling.name.casefold()
            top_folded = top_level.casefold()
            if folded == top_folded + ".py" or (
                folded.startswith(top_folded + ".")
                and folded.endswith(native_suffixes)
            ):
                raise ValueError(
                    "untrusted import-boundary collision: "
                    + sibling.relative_to(_REPO).as_posix()
                )
    for relative_text in (
        relative
        for relative in _STARTUP_CORE_FILES
        if relative.endswith(".py")
    ):
        source = _REPO / relative_text
        relative = Path(relative_text)
        if relative.name == "__init__.py":
            for initializer in source.parent.iterdir():
                folded = initializer.name.casefold()
                if initializer != source and folded.startswith(
                    "__init__."
                ) and folded.endswith(native_suffixes):
                    raise ValueError(
                        "untrusted import-boundary collision: "
                        + initializer.relative_to(_REPO).as_posix()
                    )
            import_parent = source.parent.parent
            stem = source.parent.name
        else:
            for package_alternative in source.parent.iterdir():
                if (
                    package_alternative.name.casefold() == source.stem.casefold()
                    and package_alternative.is_dir()
                ):
                    raise ValueError(
                        "untrusted import-boundary collision: "
                        + package_alternative.relative_to(_REPO).as_posix()
                    )
            import_parent = source.parent
            stem = source.stem
        try:
            siblings = tuple(import_parent.iterdir())
        except OSError as exc:
            raise ValueError("trusted import boundary cannot be inspected") from exc
        for sibling in siblings:
            if sibling == source:
                continue
            folded = sibling.name.casefold()
            if folded.startswith(stem.casefold() + ".") and folded.endswith(
                native_suffixes
            ):
                raise ValueError(
                    f"untrusted import-boundary collision: {sibling.relative_to(_REPO).as_posix()}"
                )


def _verify_exact_track_config_inventory() -> None:
    config_root = _REPO / "sim_pybullet" / "configs"
    try:
        entries = tuple(config_root.iterdir())
    except OSError as exc:
        raise ValueError("reviewed track config directory is missing") from exc
    observed = {entry.relative_to(_REPO).as_posix() for entry in entries}
    if observed != _TRACK_CONFIG_FILES:
        raise ValueError("track config inventory differs from reviewed exact set")


def _strict_json(text: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    return json.loads(
        text, object_pairs_hook=unique, parse_constant=reject_constant
    )


def _secure_repository_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if (
        path.is_absolute()
        or path.drive
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"trusted file path is unsafe: {relative}")
    probe = root
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for component in path.parts:
        probe = probe / component
        info = probe.lstat()
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & reparse_flag
        ):
            raise ValueError(f"trusted file path is indirect: {relative}")
    if not stat.S_ISREG(probe.lstat().st_mode):
        raise ValueError(f"trusted path is not a regular file: {relative}")
    resolved = probe.resolve(strict=True)
    resolved.relative_to(root)
    return resolved


def _read_stable_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    """Snapshot trusted bytes without following or accepting replacement."""

    flags = os.O_RDONLY
    for optional_flag in ("O_BINARY", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, optional_flag, 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"trusted file could not be opened safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError(f"trusted file exceeds its bounded regular-file contract: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            first = handle.read(maximum_bytes + 1)
            handle.seek(0)
            second = handle.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)
    finally:
        os.close(descriptor)
    signature = lambda info: (
        info.st_mode,
        info.st_dev,
        info.st_ino,
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000)),
    )
    if (
        len(first) > maximum_bytes
        or first != second
        or signature(before) != signature(after)
        or not os.path.samestat(after, named)
        or named.st_size != after.st_size
    ):
        raise ValueError(f"trusted file mutated while being read: {path}")
    return first


def _verify_startup_boundary(manifest_argument: str) -> dict[str, str]:
    """Verify pinned repository bytes before any repository package import."""

    _reject_executable_bytecode_boundary()
    _reject_startup_import_collisions()
    _verify_exact_track_config_inventory()
    raw_manifest = Path(manifest_argument)
    if raw_manifest.is_absolute():
        manifest_relative = raw_manifest.relative_to(_REPO)
    else:
        manifest_relative = raw_manifest
    manifest_path = _secure_repository_file(
        _REPO, manifest_relative.as_posix()
    )
    try:
        manifest_bytes = _read_stable_regular_file(
            manifest_path, maximum_bytes=_MAX_TRUSTED_MANIFEST_BYTES
        )
        manifest = _strict_json(manifest_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("trusted evaluator manifest must be UTF-8") from exc
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema", "files"}
        or manifest.get("schema") != "aigp-trusted-evaluator-files/1"
        or type(manifest.get("files")) is not dict
        or not manifest["files"]
        or not _STARTUP_CORE_FILES <= set(manifest["files"])
    ):
        raise ValueError("trusted evaluator manifest has an invalid exact schema")
    for relative, expected in manifest["files"].items():
        if (
            type(relative) is not str
            or type(expected) is not str
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("trusted evaluator manifest contains an invalid digest")
        target = _secure_repository_file(_REPO, relative)
        target_bytes = _read_stable_regular_file(
            target, maximum_bytes=_MAX_TRUSTED_SOURCE_BYTES
        )
        if hashlib.sha256(target_bytes).hexdigest() != expected:
            raise ValueError(f"trusted evaluator hash mismatch: {relative}")
    # ``scripts`` is intentionally a namespace package.  Adding an
    # unmanifested initializer would execute it before a pinned submodule.
    for collision in ("scripts/__init__.py", "scripts.py", "aigp_loop.py"):
        if (_REPO / collision).exists():
            raise ValueError(f"untrusted import-boundary collision: {collision}")
    return dict(manifest["files"])


def _audit_t4_test_boundary(trusted_files: dict[str, str]) -> None:
    """Require the collected T4 suite/discovery inputs to be exactly pinned."""

    required_discovery = {"conftest.py", "pyproject.toml"}
    if not required_discovery <= set(trusted_files):
        raise ValueError("T4 trusted manifest omits root pytest discovery inputs")
    for relative in required_discovery:
        _secure_repository_file(_REPO, relative)
    observed: set[str] = set()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    non_executable_cache_directories = {
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    for relative_root in _T4_TEST_ROOTS:
        root = _REPO / relative_root
        if not root.is_dir():
            raise ValueError(f"reviewed T4 test root is missing: {relative_root}")
        for directory, names, files in os.walk(root, topdown=True, followlinks=False):
            base = Path(directory)
            for name in names:
                if name.casefold() == "__pycache__":
                    raise ValueError("T4 boundary contains executable bytecode cache")
                info = (base / name).lstat()
                if stat.S_ISLNK(info.st_mode) or (
                    getattr(info, "st_file_attributes", 0) & reparse_flag
                ):
                    raise ValueError("T4 test inventory contains path indirection")
            names[:] = [
                name
                for name in names
                if name.casefold()
                not in {
                    cache_name.casefold()
                    for cache_name in non_executable_cache_directories
                }
            ]
            for name in files:
                if name.casefold().endswith((".pyc", ".pyo")):
                    raise ValueError("T4 boundary contains executable bytecode")
                target = base / name
                relative = target.relative_to(_REPO).as_posix()
                _secure_repository_file(_REPO, relative)
                observed.add(relative)
    expected = {
        relative
        for relative in trusted_files
        if any(
            relative == root or relative.startswith(root + "/")
            for root in _T4_TEST_ROOTS
        )
    }
    if observed != expected:
        raise ValueError(
            "T4 test inventory differs from trusted manifest: "
            f"missing={sorted(expected - observed)[:8]}, "
            f"untrusted={sorted(observed - expected)[:8]}"
        )

    discovery_names = {
        "conftest.py",
        "pytest.ini",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "sitecustomize.py",
        "usercustomize.py",
        "pytest.py",
        "pytest_timeout.py",
    }
    discovered: set[str] = set()
    for directory, names, files in os.walk(_REPO, topdown=True, followlinks=False):
        if any(name.casefold() == "__pycache__" for name in names) or any(
            name.casefold().endswith((".pyc", ".pyo")) for name in files
        ):
            raise ValueError("T4 boundary contains executable bytecode cache")
        names[:] = [
            name
            for name in names
            if name.casefold()
            not in {
                ".git",
                ".venv",
                ".aigp-loop",
                ".cache",
                ".loop",
                ".research_loop",
                *non_executable_cache_directories,
                "__pycache__",
            }
        ]
        if any(name.casefold() in {"pytest", "pytest_timeout"} for name in names):
            raise ValueError("T4 repository contains a pytest import-shadow directory")
        base = Path(directory)
        for name in files:
            folded = name.casefold()
            if folded in discovery_names or folded.endswith(".pth"):
                relative = (base / name).relative_to(_REPO).as_posix()
                _secure_repository_file(_REPO, relative)
                discovered.add(relative)
    if not discovered <= set(trusted_files):
        raise ValueError(
            "T4 has untrusted pytest/startup discovery inputs: "
            + ", ".join(sorted(discovered - set(trusted_files))[:8])
        )


def _apply_numeric_startup_environment() -> None:
    """Set native-library caps before importing NumPy/OpenCV/benchmarks."""

    os.environ.update(_NUMERIC_STARTUP_ENVIRONMENT)


def _full_suite(timeout_s: float) -> dict:
    with (
        tempfile.TemporaryDirectory(prefix="aigp-t4-pycache-") as raw_prefix,
        tempfile.TemporaryDirectory(prefix="aigp-t4-cache-") as raw_cache,
    ):
        return _run_full_suite(
            timeout_s,
            Path(raw_prefix).resolve(strict=True),
            Path(raw_cache).resolve(strict=True),
        )


def _run_full_suite(
    timeout_s: float, pycache_prefix: Path, cache_root: Path
) -> dict:
    # Override the fast default marker expression: T4 is the complete
    # non-live suite, including slow/benchmark tests but never live tests.
    command = [
        sys.executable,
        "-I",
        "-X",
        f"pycache_prefix={pycache_prefix}",
        "-m",
        "pytest",
        *_FULL_SUITE_PYTEST_ARGS,
    ]
    class StreamDigest:
        def __init__(self) -> None:
            self.digest = hashlib.sha256()
            self.tail = bytearray()
            self.error = None

        def drain(self, stream) -> None:
            try:
                while True:
                    chunk = stream.read(65_536)
                    if not chunk:
                        return
                    self.digest.update(chunk)
                    self.tail.extend(chunk)
                    if len(self.tail) > 16_000:
                        del self.tail[:-16_000]
            except (OSError, ValueError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"

    from aigp_loop.scheduler import TrialScheduler, _WindowsJobContainment

    containment = None
    delegated_posix_containment = False
    if os.name == "nt":
        containment = _WindowsJobContainment()
        launch = {
            "creationflags": (
                subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004
            )
        }
    else:
        scheduler_token = os.environ.get("AIGP_TRIAL_ID")
        if scheduler_token:
            if os.getpgrp() != os.getpid():
                raise RuntimeError(
                    "scheduler containment token is present outside an owned process group"
                )
            # Remain in the outer scheduler's exact process group. If this
            # host dies abruptly, the scheduler's every-exit killpg still
            # owns pytest and every ordinary descendant.
            delegated_posix_containment = True
            launch = {}
        else:
            launch = {"start_new_session": True}
    # Direct/operator-reviewed invocations must be as deterministic as the
    # scheduler path.  In particular, inherited PYTEST_ADDOPTS/PYTEST_PLUGINS
    # must not be able to add collection roots, plugins, or configuration.
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
    pytest_environment = {
        key: value
        for key, value in os.environ.items()
        if key.casefold() in allowed_folded
    }
    pytest_environment.update(_NUMERIC_STARTUP_ENVIRONMENT)
    pytest_environment.update(
        {
            "AIGP_CACHE_ROOT": str(cache_root),
            "AIGP_PROMOTION_TIER": "4",
            "AIGP_TRIAL_OFFLINE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPYCACHEPREFIX": str(pycache_prefix),
            "PYTHONUNBUFFERED": "1",
        }
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=str(_REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=pytest_environment,
            **launch,
        )
    except Exception:
        if containment is not None:
            containment.close()
        raise
    if containment is not None:
        try:
            containment.attach_and_resume(process)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=3.0)
            finally:
                containment.close()
            raise
    assert process.stdout is not None and process.stderr is not None
    stdout = StreamDigest()
    stderr = StreamDigest()
    threads = [
        threading.Thread(target=stdout.drain, args=(process.stdout,), daemon=True),
        threading.Thread(target=stderr.drain, args=(process.stderr,), daemon=True),
    ]
    for thread in threads:
        thread.start()
    containment_error = None
    delegated_timeout = False
    returncode = 125
    try:
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            returncode = 124
    finally:
        try:
            if containment is not None:
                containment.terminate_and_prove(process)
            elif delegated_posix_containment:
                if returncode == 124:
                    delegated_timeout = True
                    process.kill()
                    process.wait(timeout=3.0)
            else:
                TrialScheduler._terminate_process_tree(process)
        except Exception as exc:
            containment_error = f"{type(exc).__name__}: {exc}"
    for thread in threads:
        thread.join(timeout=5.0)
    if any(thread.is_alive() for thread in threads):
        process.stdout.close()
        process.stderr.close()
        for thread in threads:
            thread.join(timeout=1.0)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("full-suite output drain threads did not terminate")
    if containment_error is not None:
        raise RuntimeError(
            "full-suite descendant cleanup was not proved: " + containment_error
        )
    if delegated_timeout:
        raise RuntimeError(
            "nested full-suite timeout requires outer scheduler process-group cleanup"
        )
    if stdout.error is not None or stderr.error is not None:
        raise RuntimeError(
            "full-suite output drain failed: "
            + "; ".join(item for item in (stdout.error, stderr.error) if item)
        )
    framed_digest = hashlib.sha256(
        b"stdout\0"
        + stdout.digest.digest()
        + b"\0stderr\0"
        + stderr.digest.digest()
    ).hexdigest()
    tail = (
        bytes(stdout.tail).decode("utf-8", errors="replace")
        + "\n"
        + bytes(stderr.tail).decode("utf-8", errors="replace")
    )[-32_000:]
    return {
        "schema": "aigp-nonlive-pytest/1",
        "passed": returncode == 0,
        "returncode": returncode,
        "pytest_args": list(_FULL_SUITE_PYTEST_ARGS),
        "output_sha256": framed_digest,
        "output_tail": tail,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--full-suite", action="store_true")
    parser.add_argument("--full-suite-timeout-s", type=float, default=900.0)
    parser.add_argument("--trusted-manifest", required=True)
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.duration)
        or args.duration <= 0.0
        or not math.isfinite(args.dt)
        or args.dt <= 0.0
        or type(args.workers) is not int
        or args.workers < 1
        or not math.isfinite(args.full_suite_timeout_s)
        or args.full_suite_timeout_s <= 0.0
    ):
        parser.error("duration, dt, workers, and suite timeout must be finite and positive")
    trusted_files = _verify_startup_boundary(args.trusted_manifest)
    if args.tier == 4:
        _audit_t4_test_boundary(trusted_files)
    pycache_context = tempfile.TemporaryDirectory(prefix="aigp-nonlive-pycache-")
    _RUNTIME_PYCACHE_CONTEXTS.append(pycache_context)
    pycache_prefix = Path(pycache_context.name).resolve(strict=True)
    if _REPO == pycache_prefix or _REPO in pycache_prefix.parents:
        raise RuntimeError("trusted bytecode prefix must be outside repository")
    sys.pycache_prefix = str(pycache_prefix)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_prefix)
    _apply_numeric_startup_environment()
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    # Repository imports are deliberately delayed until after the standalone
    # manifest audit above.  With ``python -I scripts/aigp_nonlive.py``, no
    # candidate startup hook or package initializer can run first.
    from aigp_loop._util import json_hash
    from aigp_loop.nonlive import adapt_matrix_evidence, evaluator_source_hashes
    from aigp_loop.promotion import Tier
    from scripts.benchmark import prepare_course
    from scripts.benchmark_matrix import (
        _list_configs,
        _load_config,
        run_matrix,
        worker_numeric_environment,
    )

    tier = Tier(args.tier)
    names = tuple(item for item in args.configs.split(",") if item)
    if not names or len(names) != len(set(names)):
        parser.error("--configs must contain unique non-empty track names")
    if tier is Tier.T2_WARM_SIM and names != ("race_01",):
        parser.error("T2 requires exactly race_01")
    if tier is Tier.T3_DOMAIN_TRACKS and tuple(sorted(names)) != _DOMAIN_TRACK_SET:
        parser.error("T3 requires the exact reviewed three-track subset")
    if tier is Tier.T4_FULL_NON_LIVE and tuple(sorted(names)) != _FULL_TRACK_SET:
        parser.error("T4 requires the exact seven-track config set")
    if (tier is Tier.T4_FULL_NON_LIVE) is not args.full_suite:
        parser.error("--full-suite is required exactly for T4")
    available = {path.stem: path for path in _list_configs()}
    missing = set(names) - set(available)
    if missing:
        parser.error(f"unknown configs: {sorted(missing)}")
    # Validate the reviewed evaluator boundary before spending time on either
    # the full suite or the matrix.
    source_hashes = evaluator_source_hashes(_REPO, args.trusted_manifest)
    suite = None
    config_paths = [available[name] for name in names]
    cache_preparation = None
    if tier is Tier.T2_WARM_SIM:
        # A cold workspace is a valid starting state, but T2 must time and
        # score a real rollout over prepared artifacts.  Build/verify only the
        # deterministic planning layers here; do not spend a bootstrap rollout
        # and do not populate/read the final-result cache.
        # The preparation fingerprint is part of every artifact key.  Apply
        # the same numeric cap as the measured matrix worker, then restore the
        # trusted host before orchestration fingerprints its own environment.
        with worker_numeric_environment() as preparation_fingerprint:
            prepared = prepare_course(_load_config(config_paths[0]), dt=args.dt)
        if prepared.dependency_fingerprint != preparation_fingerprint:
            raise RuntimeError("prepared course escaped the capped worker environment")
        preparation_result = {
            "track": names[0],
            "prepared_course": prepared.artifact_key,
            "artifact_keys": dict(prepared.artifact_keys),
            "cache_states": dict(prepared.cache_states),
            "planning_config_sha256": prepared.config_hash,
            "dependency_fingerprint": prepared.dependency_fingerprint,
        }
        cache_preparation = {
            "schema": "aigp-cache-preparation/3",
            "preparation_result_sha256": json_hash(preparation_result),
            "dependency_fingerprint_sha256": json_hash(
                preparation_fingerprint
            ),
            "cache_hit_or_miss": (
                "hit"
                if set(prepared.cache_states.values()) == {"hit"}
                else "miss"
            ),
        }
    matrix = run_matrix(
        config_paths,
        duration=args.duration,
        dt=args.dt,
        max_workers=args.workers,
        include_results=True,
        # T2 scores a real rollout over warm prepared layers.  A cached final
        # benchmark result is evidence reuse, not a simulation trial.
        use_result_cache=tier is not Tier.T2_WARM_SIM,
    )
    # T4 runs pytest last. Under scheduler containment the nested process
    # inherits the outer process group, and the scheduler kills/proves that
    # group immediately after this host emits its evidence and exits.
    if args.full_suite:
        suite = _full_suite(args.full_suite_timeout_s)
    result = adapt_matrix_evidence(
        matrix,
        tier=tier,
        expected_tracks=names,
        source_hashes=source_hashes,
        full_nonlive_suite=suite,
        cache_preparation=cache_preparation,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if all(result["promotion"]["hard_gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
