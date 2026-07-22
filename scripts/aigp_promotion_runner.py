"""Run the exact non-live promotion suite once and publish durable results.

State is content-addressed by the clean candidate commit, development lock,
Python runtime, installed environment, and semantic pytest command.  A second
caller attaches to an active owner instead of launching a duplicate suite.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from aigp_loop._util import environment_fingerprint, json_hash, run_checked
from planning.artifact_cache import ArtifactStore


_DEVELOPMENT_LOCK = Path("requirements/development-test.lock.txt")
_SEMANTIC_PYTEST_ARGV = (
    "-B",
    "-s",
    "-u",
    "-m",
    "pytest",
    "-vv",
    "--color=no",
    "-p",
    "pytest_timeout",
    "-p",
    "scripts.aigp_pytest_progress",
    "-p",
    "no:cacheprovider",
    "-o",
    "required_plugins=",
    "-c",
    "pyproject.toml",
    "-m",
    "not live",
    "--timeout=300",
    "--durations=25",
    "--junitxml={junitxml}",
    "--basetemp={basetemp}",
)
_NUMERIC_ENVIRONMENT = {
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
_ALLOWED_ENVIRONMENT = frozenset(
    {
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
)
_STATE_ROOT_ENV = "AIGP_PROMOTION_STATE_ROOT"
_IDENTITY_NAMESPACE = "promotion-test-identities-v1"
_STATUS_NAMESPACE = "promotion-test-status-v1"
_RESULT_NAMESPACE = "promotion-test-results-v1"
_RUN_LOCK_NAMESPACE = "promotion-test-run-locks-v1"
_HEARTBEAT_SECONDS = 30.0
_PROMOTION_TIMEOUT_SECONDS = 900.0
_MAX_LOG_BYTES = 16 * 1024 * 1024
_MAX_OUTPUT_TAIL_BYTES = 32_000


class PromotionRunError(RuntimeError):
    """Fail-closed promotion orchestration error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _strict_json_loads(text: str) -> Any:
    def unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
        object_pairs_hook=unique,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _git_common_directory(repository: Path) -> Path:
    raw = Path(run_checked(["git", "rev-parse", "--git-common-dir"], cwd=repository))
    return (repository / raw).resolve() if not raw.is_absolute() else raw.resolve()


def _git_worktree_directories(repository: Path) -> tuple[Path, ...]:
    raw = run_checked(
        ["git", "worktree", "list", "--porcelain", "-z"], cwd=repository
    )
    worktrees = tuple(
        Path(field.removeprefix("worktree ")).resolve()
        for field in raw.split("\0")
        if field.startswith("worktree ")
    )
    if not worktrees:
        raise PromotionRunError("Git did not report any candidate worktrees")
    return worktrees


def _repository_name(repository: Path) -> str:
    common = _git_common_directory(repository)
    return common.parent.name if common.name.casefold() == ".git" else common.name


def _inherited_pytest_environment() -> dict[str, str]:
    allowed = {name.casefold() for name in _ALLOWED_ENVIRONMENT}
    return {
        name: value
        for name, value in os.environ.items()
        if name.casefold() in allowed
    }


def _pytest_environment_key_material() -> dict[str, str]:
    environment = _inherited_pytest_environment()
    environment.update(_NUMERIC_ENVIRONMENT)
    environment.update(
        {
            "AIGP_CACHE_ROOT": "{attempt}/artifact-cache",
            "AIGP_PROMOTION_PROGRESS_PATH": "{attempt}/progress.json",
            "AIGP_PROMOTION_RUN_KEY": "{promotion-key}",
            "AIGP_PROMOTION_HEARTBEAT_SECONDS": str(_HEARTBEAT_SECONDS),
            "AIGP_PROMOTION_TIER": "4",
            "AIGP_TRIAL_OFFLINE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": "{attempt}/pycache",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _state_root(repository: Path, configured: str | None = None) -> Path:
    raw_configured = configured or os.environ.get(_STATE_ROOT_ENV)
    if raw_configured:
        base = Path(raw_configured).expanduser().resolve()
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"]).resolve() / "AIGP" / "promotion-tests" / "v1"
    else:
        state_home = os.environ.get("XDG_STATE_HOME")
        base = (
            Path(state_home).expanduser().resolve()
            if state_home
            else Path.home().resolve() / ".local" / "state"
        ) / "aigp" / "promotion-tests" / "v1"
    if os.name == "nt" and str(base).startswith("\\\\"):
        raise PromotionRunError("promotion state root must use a local filesystem")
    common_directory = _git_common_directory(repository)
    common = os.path.normcase(str(common_directory))
    scope = hashlib.sha256(common.encode("utf-8")).hexdigest()[:24]
    root = base / scope
    forbidden_roots = {
        repository.resolve(),
        common_directory,
        *(_git_worktree_directories(repository)),
    }
    if any(_is_within(root, forbidden) for forbidden in forbidden_roots):
        raise PromotionRunError(
            "promotion state root must be outside every candidate worktree and "
            "the shared Git directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise PromotionRunError("promotion state root must be a regular local directory")
    return root


def _clean_commit(repository: Path) -> str:
    previous: tuple[str, str, tuple[str, ...]] | None = None
    for _ in range(4):
        commit = run_checked(["git", "rev-parse", "HEAD"], cwd=repository)
        status = run_checked(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
        )
        ignored_raw = run_checked(
            [
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--ignored=matching",
                "--untracked-files=all",
            ],
            cwd=repository,
        )
        ignored = tuple(
            entry[3:]
            for entry in ignored_raw.split("\0")
            if entry.startswith("!! ")
        )
        current = (commit, status, ignored)
        if previous == current:
            break
        previous = current
    else:
        raise PromotionRunError("candidate changed while Git provenance was captured")
    if status:
        preview = ", ".join(status.splitlines()[:8])
        raise PromotionRunError(
            "test-promotion requires a clean exact-commit worktree; observed " + preview
        )
    if ignored:
        preview = ", ".join(ignored[:8])
        raise PromotionRunError(
            "test-promotion requires an exact pristine checkout without ignored "
            "files or directories; observed "
            + preview
        )
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise PromotionRunError("Git HEAD is not an exact hexadecimal object identity")
    return commit


def _promotion_identity(repository: Path) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    commit_before = _clean_commit(repository)
    lock_path = repository / _DEVELOPMENT_LOCK
    if lock_path.is_symlink() or not lock_path.is_file():
        raise PromotionRunError("development test lock is missing or indirect")
    lock_digest = _sha256_file(lock_path)
    runtime_fingerprint = environment_fingerprint()
    commit_after = _clean_commit(repository)
    if commit_after != commit_before:
        raise PromotionRunError("candidate commit changed while identity was captured")
    return {
        "schema": "aigp-promotion-test-identity/1",
        "repository": _repository_name(repository),
        "commit": commit_before,
        "development_lock": {
            "path": _DEVELOPMENT_LOCK.as_posix(),
            "sha256": lock_digest,
        },
        "python": {
            "implementation": platform.python_implementation().casefold(),
            "version": sys.version,
            "cache_tag": sys.implementation.cache_tag,
            "platform": sys.platform,
            "machine": platform.machine(),
            "environment_sha256": runtime_fingerprint,
        },
        "pytest": {
            "argv": list(_SEMANTIC_PYTEST_ARGV),
            "cwd": "$REPOSITORY_ROOT",
            "environment_policy": "aigp-promotion-pytest-environment/1",
            "environment_sha256": json_hash(_pytest_environment_key_material()),
        },
        "supervisor": {
            "policy": "aigp-promotion-supervisor/1",
            "wall_timeout_s": _PROMOTION_TIMEOUT_SECONDS,
        },
    }


def _promotion_key(identity: Mapping[str, Any]) -> str:
    return json_hash(identity)


def _load_store_payload(
    store: ArtifactStore, namespace: str, key: str
) -> Any | None:
    path = store.path(namespace, key, ".json")
    if not path.exists():
        return None
    payload = store.load_json(namespace, key)
    if payload is None:
        raise PromotionRunError(f"corrupt durable promotion state: {path}")
    return payload


def _ensure_identity(
    store: ArtifactStore, key: str, identity: Mapping[str, Any]
) -> None:
    existing = _load_store_payload(store, _IDENTITY_NAMESPACE, key)
    if existing is None:
        store.save_json(_IDENTITY_NAMESPACE, key, dict(identity))
        existing = _load_store_payload(store, _IDENTITY_NAMESPACE, key)
    if existing != identity:
        raise PromotionRunError("promotion key collided with different identity material")


def _validated_status(
    value: Any, key: str, state_root: Path
) -> dict[str, Any] | None:
    if value is None:
        return None
    if (
        type(value) is not dict
        or value.get("schema") != "aigp-promotion-test-status/1"
        or value.get("key") != key
        or type(value.get("attempt_id")) is not str
        or not value["attempt_id"]
        or Path(value["attempt_id"]).name != value["attempt_id"]
        or value["attempt_id"] in {".", ".."}
        or value.get("phase") not in {"running", "finished"}
        or type(value.get("attempt_directory")) is not str
        or type(value.get("heartbeat_sequence")) is not int
        or isinstance(value.get("heartbeat_sequence"), bool)
        or value["heartbeat_sequence"] < 0
    ):
        raise PromotionRunError("durable promotion status has an invalid schema")
    expected = state_root / "attempts" / key / value["attempt_id"]
    supplied = Path(value["attempt_directory"])
    try:
        expected_resolved = expected.resolve()
        supplied_resolved = supplied.resolve()
    except OSError as exc:
        raise PromotionRunError(
            "durable promotion status has an invalid attempt directory"
        ) from exc
    if (
        not supplied.is_absolute()
        or supplied_resolved != expected_resolved
        or not _is_within(expected_resolved, state_root)
    ):
        raise PromotionRunError(
            "durable promotion status attempt directory is outside its bound state path"
        )
    return value


def _validated_result(
    value: Any, key: str, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    if value is None:
        return None
    if (
        type(value) is not dict
        or value.get("schema") != "aigp-promotion-test-result/1"
        or value.get("key") != key
        or value.get("identity") != identity
        or type(value.get("attempt_id")) is not str
        or value.get("outcome")
        not in {
            "passed",
            "failed",
            "timeout",
            "interrupted",
            "infrastructure_error",
        }
        or type(value.get("runner_exit_code")) is not int
        or isinstance(value.get("runner_exit_code"), bool)
        or not 0 <= value["runner_exit_code"] <= 255
    ):
        raise PromotionRunError("durable promotion result has an invalid schema")
    return value


def _load_progress(path: Path, key: str) -> dict[str, Any] | None:
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None
    if (
        type(value) is not dict
        or value.get("schema") != "aigp-promotion-pytest-progress/1"
        or value.get("run_key") != key
    ):
        return None
    return value


def _format_progress(progress: Mapping[str, Any] | None) -> str:
    if progress is None:
        return "pytest progress is not available yet"
    collected = progress.get("collected")
    completed = progress.get("completed")
    current = progress.get("current_test") or "between tests"
    elapsed = progress.get("current_test_elapsed_s")
    elapsed_text = f" ({elapsed:.1f}s)" if type(elapsed) is float else ""
    return f"{completed}/{collected or '?'} complete; {current}{elapsed_text}"


class _ConsolePublisher:
    """Best-effort console output that can never backpressure pytest."""

    def __init__(self) -> None:
        self.items: queue.Queue[bytes | None] = queue.Queue(maxsize=512)
        self.thread = threading.Thread(
            target=self._run,
            name="aigp-promotion-console",
            daemon=True,
        )
        self.thread.start()

    def publish(self, value: str | bytes) -> None:
        payload = value.encode("utf-8", errors="replace") if isinstance(value, str) else value
        for offset in range(0, len(payload), 8192):
            try:
                self.items.put_nowait(payload[offset : offset + 8192])
            except queue.Full:
                return

    def _run(self) -> None:
        try:
            descriptor = sys.stdout.fileno()
        except (AttributeError, OSError, ValueError):
            descriptor = None
        while True:
            payload = self.items.get()
            if payload is None:
                return
            try:
                if descriptor is None:
                    sys.stdout.write(payload.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
                else:
                    os.write(descriptor, payload)
            except (BrokenPipeError, OSError, ValueError):
                return

    def close(self) -> None:
        with contextlib.suppress(queue.Full):
            self.items.put_nowait(None)
        self.thread.join(timeout=1.0)


class _OutputCapture:
    def __init__(
        self, stream: Any, log: Any, console: _ConsolePublisher
    ) -> None:
        self.stream = stream
        self.log = log
        self.console = console
        self.digest = hashlib.sha256()
        self.tail = bytearray()
        self.bytes_seen = 0
        self.bytes_stored = 0
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._drain,
            name="aigp-promotion-output-drain",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _drain(self) -> None:
        log_enabled = True
        read = getattr(self.stream, "read1", self.stream.read)
        try:
            with self.log:
                while True:
                    try:
                        # BufferedReader.read() may wait for the requested byte
                        # count.  read1() returns after one raw-pipe read so
                        # pytest's verbose node IDs are visible immediately.
                        chunk = read(65_536)
                    except (OSError, ValueError) as exc:
                        self.error = f"{type(exc).__name__}: {exc}"
                        return
                    if not chunk:
                        break
                    self.bytes_seen += len(chunk)
                    self.digest.update(chunk)
                    self.tail.extend(chunk)
                    if len(self.tail) > _MAX_OUTPUT_TAIL_BYTES:
                        del self.tail[:-_MAX_OUTPUT_TAIL_BYTES]
                    remaining = max(0, _MAX_LOG_BYTES - self.bytes_stored)
                    if remaining and log_enabled:
                        stored = chunk[:remaining]
                        try:
                            self.log.write(stored)
                            self.log.flush()
                            self.bytes_stored += len(stored)
                        except (OSError, ValueError) as exc:
                            # Never abandon the stdout pipe: a full child pipe
                            # could otherwise deadlock pytest.  The supervisor
                            # observes this error and terminates the contained
                            # process while this thread keeps draining.
                            self.error = f"{type(exc).__name__}: {exc}"
                            log_enabled = False
                    self.console.publish(chunk)
                if log_enabled:
                    try:
                        os.fsync(self.log.fileno())
                    except (OSError, ValueError) as exc:
                        self.error = f"{type(exc).__name__}: {exc}"
        except (OSError, ValueError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def finish(self) -> dict[str, Any]:
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            with contextlib.suppress(OSError, ValueError):
                self.stream.close()
            self.thread.join(timeout=1.0)
        if self.thread.is_alive():
            self.error = "output drain thread did not terminate"
        return {
            "sha256": self.digest.hexdigest(),
            "bytes_seen": self.bytes_seen,
            "bytes_stored": self.bytes_stored,
            "truncated": self.bytes_stored < self.bytes_seen,
            "tail": bytes(self.tail).decode("utf-8", errors="replace"),
            "error": self.error,
        }


def _pytest_command(attempt_directory: Path) -> list[str]:
    replacements = {
        "junitxml": str(attempt_directory / "junit.xml"),
        "basetemp": str(attempt_directory / "pytest-tmp"),
    }
    return [
        sys.executable,
        *(argument.format(**replacements) for argument in _SEMANTIC_PYTEST_ARGV),
    ]


def _pytest_environment(
    attempt_directory: Path, progress_path: Path, key: str
) -> dict[str, str]:
    environment = _inherited_pytest_environment()
    environment.update(_NUMERIC_ENVIRONMENT)
    environment.update(
        {
            "AIGP_CACHE_ROOT": str(attempt_directory / "artifact-cache"),
            "AIGP_PROMOTION_PROGRESS_PATH": str(progress_path),
            "AIGP_PROMOTION_RUN_KEY": key,
            "AIGP_PROMOTION_HEARTBEAT_SECONDS": str(_HEARTBEAT_SECONDS),
            "AIGP_PROMOTION_TIER": "4",
            "AIGP_TRIAL_OFFLINE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": str(attempt_directory / "pycache"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _status_payload(
    key: str,
    attempt_id: str,
    attempt_directory: Path,
    *,
    started_at: str,
    heartbeat_sequence: int,
    phase: str = "running",
    progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "aigp-promotion-test-status/1",
        "key": key,
        "attempt_id": attempt_id,
        "phase": phase,
        "owner_pid": os.getpid(),
        "started_at_utc": started_at,
        "heartbeat_at_utc": _utc_now(),
        "heartbeat_sequence": heartbeat_sequence,
        "attempt_directory": str(attempt_directory),
        "progress": dict(progress) if progress is not None else None,
    }


def _run_pytest(
    repository: Path,
    store: ArtifactStore,
    key: str,
    attempt_id: str,
    attempt_directory: Path,
    started_at: str,
) -> tuple[
    int | None,
    bool,
    bool,
    dict[str, Any],
    str | None,
    str | None,
]:
    from aigp_loop.scheduler import TrialScheduler, _WindowsJobContainment

    progress_path = attempt_directory / "progress.json"
    log_path = attempt_directory / "pytest.log"
    command = _pytest_command(attempt_directory)
    environment = _pytest_environment(attempt_directory, progress_path, key)
    console = _ConsolePublisher()
    console.publish(
        f"[promotion] key={key} attempt={attempt_id}\n"
        f"[promotion] durable state: {attempt_directory}\n"
    )
    containment = None
    launch_options: dict[str, Any] = {}
    if os.name == "nt":
        containment = _WindowsJobContainment()
        launch_options["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004
        )
    else:
        launch_options["start_new_session"] = True
    process: subprocess.Popen[bytes] | None = None
    capture: _OutputCapture | None = None
    log = None
    interrupted = False
    timed_out = False
    run_error: str | None = None
    cleanup_error: str | None = None
    pytest_exit_code: int | None = None
    heartbeat_sequence = 0
    try:
        log = log_path.open("xb")
        process = subprocess.Popen(
            command,
            cwd=str(repository),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            env=environment,
            **launch_options,
        )
        if containment is not None:
            try:
                containment.attach_and_resume(process)
            except Exception:
                process.kill()
                process.wait(timeout=3.0)
                containment.close()
                containment = None
                raise
        assert process.stdout is not None
        capture = _OutputCapture(process.stdout, log, console)
        log = None
        capture.start()
        deadline = time.monotonic() + _PROMOTION_TIMEOUT_SECONDS
        next_heartbeat = time.monotonic()
        while process.poll() is None:
            if capture.error is not None:
                raise PromotionRunError(
                    "pytest output capture failed while the process was active: "
                    + capture.error
                )
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                console.publish(
                    f"[promotion] hard {_PROMOTION_TIMEOUT_SECONDS:g}-second "
                    "suite timeout reached; "
                    "proving descendant cleanup\n"
                )
                break
            if now >= next_heartbeat:
                progress = _load_progress(progress_path, key)
                heartbeat_sequence += 1
                store.save_json(
                    _STATUS_NAMESPACE,
                    key,
                    _status_payload(
                        key,
                        attempt_id,
                        attempt_directory,
                        started_at=started_at,
                        heartbeat_sequence=heartbeat_sequence,
                        progress=progress,
                    ),
                )
                console.publish(
                    "[promotion] heartbeat: " + _format_progress(progress) + "\n"
                )
                next_heartbeat = now + _HEARTBEAT_SECONDS
            time.sleep(0.1)
        if not timed_out:
            pytest_exit_code = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        console.publish("[promotion] owner interrupted; proving descendant cleanup\n")
    except BaseException as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        console.publish(
            "[promotion] runner failure; proving descendant cleanup: "
            + run_error
            + "\n"
        )
    finally:
        if process is not None:
            try:
                if containment is not None:
                    containment.terminate_and_prove(process)
                else:
                    TrialScheduler._terminate_process_tree(process)
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
        elif containment is not None:
            containment.close()
        output = (
            capture.finish()
            if capture is not None
            else {
                "sha256": hashlib.sha256(b"").hexdigest(),
                "bytes_seen": 0,
                "bytes_stored": 0,
                "truncated": False,
                "tail": "",
                "error": None,
            }
        )
        if log is not None:
            log.close()
        console.close()
    return (
        pytest_exit_code,
        interrupted,
        timed_out,
        output,
        cleanup_error,
        run_error,
    )


def _junit_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        return {
            "tests": sum(int(suite.get("tests", "0")) for suite in suites),
            "failures": sum(int(suite.get("failures", "0")) for suite in suites),
            "errors": sum(int(suite.get("errors", "0")) for suite in suites),
            "skipped": sum(int(suite.get("skipped", "0")) for suite in suites),
            "time_s": round(sum(float(suite.get("time", "0")) for suite in suites), 6),
        }
    except (ET.ParseError, OSError, TypeError, ValueError):
        return None


def _execute_under_lock(
    repository: Path,
    store: ArtifactStore,
    state_root: Path,
    key: str,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_id = f"{int(time.time())}-{uuid.uuid4().hex}"
    attempt_directory = state_root / "attempts" / key / attempt_id
    started_at = _utc_now()
    # Status publication is the durable start of an attempt.  Publish it
    # before creating attempt-local files so a killed --fresh owner cannot
    # leave an unrecorded newer attempt behind an older reusable result.
    store.save_json(
        _STATUS_NAMESPACE,
        key,
        _status_payload(
            key,
            attempt_id,
            attempt_directory,
            started_at=started_at,
            heartbeat_sequence=0,
        ),
    )
    try:
        attempt_directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PromotionRunError(
            f"could not create durable attempt directory {attempt_directory}: {exc}"
        ) from exc
    monotonic_started = time.monotonic()
    runner_error: str | None = None
    try:
        (
            pytest_exit_code,
            interrupted,
            timed_out,
            output,
            cleanup_error,
            runner_error,
        ) = _run_pytest(
            repository, store, key, attempt_id, attempt_directory, started_at
        )
    except BaseException as exc:
        pytest_exit_code = None
        interrupted = isinstance(exc, KeyboardInterrupt)
        timed_out = False
        cleanup_error = None
        output = {
            "sha256": hashlib.sha256(b"").hexdigest(),
            "bytes_seen": 0,
            "bytes_stored": 0,
            "truncated": False,
            "tail": "",
            "error": None,
        }
        runner_error = f"{type(exc).__name__}: {exc}"
    infrastructure_errors = [runner_error] if runner_error is not None else []
    identity_after: Mapping[str, Any] | None = None
    try:
        identity_after = _promotion_identity(repository)
    except Exception as exc:
        infrastructure_errors.append(
            f"post-run provenance failed: {type(exc).__name__}: {exc}"
        )
    if identity_after is not None and identity_after != identity:
        infrastructure_errors.append(
            "candidate or runtime identity changed during promotion tests"
        )
    if cleanup_error is not None:
        infrastructure_errors.append(
            "descendant cleanup was not proved: " + cleanup_error
        )
    if output.get("error") is not None:
        infrastructure_errors.append(
            "pytest output capture failed: " + str(output["error"])
        )
    runner_error = "; ".join(infrastructure_errors) or None
    if interrupted:
        outcome = "interrupted"
        runner_exit_code = 130
    elif runner_error is not None:
        outcome = "infrastructure_error"
        runner_exit_code = 125
    elif timed_out:
        outcome = "timeout"
        runner_exit_code = 124
    elif pytest_exit_code == 0:
        outcome = "passed"
        runner_exit_code = 0
    else:
        outcome = "failed"
        runner_exit_code = (
            int(pytest_exit_code)
            if type(pytest_exit_code) is int and 1 <= pytest_exit_code <= 255
            else 1
        )
    junit_path = attempt_directory / "junit.xml"
    progress_path = attempt_directory / "progress.json"
    result = {
        "schema": "aigp-promotion-test-result/1",
        "key": key,
        "identity": dict(identity),
        "attempt_id": attempt_id,
        "outcome": outcome,
        "pytest_exit_code": pytest_exit_code,
        "runner_exit_code": runner_exit_code,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_s": round(time.monotonic() - monotonic_started, 6),
        "wall_timeout_s": _PROMOTION_TIMEOUT_SECONDS,
        "runner_error": runner_error,
        "pytest_command": _pytest_command(attempt_directory),
        "progress": _load_progress(progress_path, key),
        "junit": {
            "path": str(junit_path),
            "sha256": _sha256_file(junit_path) if junit_path.is_file() else None,
            "summary": _junit_summary(junit_path),
        },
        "output": {
            **output,
            "path": str(attempt_directory / "pytest.log"),
        },
        "result_path": str(attempt_directory / "result.json"),
    }
    _atomic_json(attempt_directory / "result.json", result)
    store.save_json(_RESULT_NAMESPACE, key, result)
    latest_status = _validated_status(
        _load_store_payload(store, _STATUS_NAMESPACE, key), key, state_root
    )
    final_heartbeat_sequence = (
        int(latest_status["heartbeat_sequence"]) + 1
        if latest_status is not None
        else 1
    )
    store.save_json(
        _STATUS_NAMESPACE,
        key,
        _status_payload(
            key,
            attempt_id,
            attempt_directory,
            started_at=started_at,
            heartbeat_sequence=final_heartbeat_sequence,
            phase="finished",
            progress=result["progress"],
        ),
    )
    return result


def _tail_log(path: Path, offset: int, console: _ConsolePublisher) -> int:
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            while True:
                chunk = handle.read(65_536)
                if not chunk:
                    break
                console.publish(chunk)
                offset += len(chunk)
    except FileNotFoundError:
        pass
    return offset


def _attach(
    store: ArtifactStore,
    state_root: Path,
    key: str,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    console = _ConsolePublisher()
    console.publish(f"[promotion] attaching to active key={key}\n")
    attempt_id: str | None = None
    attempt_directory: Path | None = None
    progress_path: Path | None = None
    log_path: Path | None = None
    offset = 0
    try:
        next_heartbeat = time.monotonic()
        while True:
            status = _validated_status(
                _load_store_payload(store, _STATUS_NAMESPACE, key),
                key,
                state_root,
            )
            if status is not None and status["attempt_id"] != attempt_id:
                attempt_id = str(status["attempt_id"])
                attempt_directory = Path(str(status["attempt_directory"]))
                progress_path = attempt_directory / "progress.json"
                log_path = attempt_directory / "pytest.log"
                offset = log_path.stat().st_size if log_path.exists() else 0
                console.publish(
                    f"[promotion] active attempt={attempt_id}\n"
                    f"[promotion] durable state: {attempt_directory}\n"
                )
            if log_path is not None:
                offset = _tail_log(log_path, offset, console)
            now = time.monotonic()
            if now >= next_heartbeat:
                console.publish(
                    "[promotion] attached heartbeat: "
                    + _format_progress(
                        _load_progress(progress_path, key)
                        if progress_path is not None
                        else None
                    )
                    + "\n"
                )
                next_heartbeat = now + _HEARTBEAT_SECONDS
            try:
                with store.lock(_RUN_LOCK_NAMESPACE, key, timeout_s=0.0):
                    final_status = _validated_status(
                        _load_store_payload(store, _STATUS_NAMESPACE, key),
                        key,
                        state_root,
                    )
                    result = _validated_result(
                        _load_store_payload(store, _RESULT_NAMESPACE, key),
                        key,
                        identity,
                    )
                    if result is None or (
                        final_status is not None
                        and result["attempt_id"] != final_status["attempt_id"]
                    ):
                        evidence = (
                            final_status["attempt_directory"]
                            if final_status is not None
                            else state_root / "attempts" / key
                        )
                        raise PromotionRunError(
                            "promotion owner exited without a matching durable result; "
                            f"evidence is at {evidence}; refusing an automatic rerun "
                            "(use --fresh after review)"
                        )
                    return result
            except TimeoutError:
                time.sleep(0.25)
    except KeyboardInterrupt as exc:
        raise PromotionRunError(
            "detached from the active promotion run; the owner was not interrupted"
        ) from exc
    finally:
        console.close()


def _print_result(result: Mapping[str, Any], *, reused: bool) -> None:
    prefix = "reused" if reused else "completed"
    summary = (
        result.get("junit", {}).get("summary")
        if isinstance(result.get("junit"), dict)
        else None
    )
    detail = ""
    if isinstance(summary, dict):
        detail = (
            f" tests={summary.get('tests')} failures={summary.get('failures')}"
            f" errors={summary.get('errors')} skipped={summary.get('skipped')}"
        )
    console = _ConsolePublisher()
    console.publish(
        f"[promotion] {prefix} outcome={result['outcome']}"
        f" elapsed={result.get('elapsed_s')}s{detail}\n"
        f"[promotion] result: {result.get('result_path', 'unknown')}\n"
    )
    console.close()


def run_promotion(
    repository: Path = _REPO,
    *,
    configured_state_root: str | None = None,
    fresh: bool = False,
) -> int:
    repository = repository.resolve(strict=True)
    if Path.cwd().resolve() != repository:
        raise PromotionRunError("test-promotion must run from the repository root")
    identity = _promotion_identity(repository)
    key = _promotion_key(identity)
    state_root = _state_root(repository, configured_state_root)
    store = ArtifactStore(state_root)
    _ensure_identity(store, key, identity)
    result: dict[str, Any]
    reused = False
    try:
        with store.lock(_RUN_LOCK_NAMESPACE, key, timeout_s=0.0):
            result = _validated_result(
                _load_store_payload(store, _RESULT_NAMESPACE, key), key, identity
            )
            status = _validated_status(
                _load_store_payload(store, _STATUS_NAMESPACE, key), key, state_root
            )
            if not fresh and result is not None:
                if status is None or status["attempt_id"] == result["attempt_id"]:
                    reused = True
                else:
                    result = None
            if not fresh and result is None and status is not None:
                raise PromotionRunError(
                    "a prior promotion attempt has no reusable terminal result; "
                    f"evidence is at {status['attempt_directory']}; refusing an "
                    "automatic rerun (use --fresh after review)"
                )
            if not reused:
                result = _execute_under_lock(
                    repository, store, state_root, key, identity
                )
    except TimeoutError:
        if fresh:
            raise PromotionRunError(
                "--fresh cannot replace an active promotion run; attach without it"
            )
        result = _attach(store, state_root, key, identity)
        reused = True
    # Terminal console writes are deliberately outside the owner lock.  A
    # stalled output consumer must never retain promotion ownership.
    _print_result(result, reused=reused)
    return int(result["runner_exit_code"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "explicitly start a new attempt after reviewing an earlier terminal or "
            "incomplete result; never replaces an active run"
        ),
    )
    args = parser.parse_args(argv)
    try:
        return run_promotion(fresh=args.fresh)
    except PromotionRunError as exc:
        print(f"test-promotion refused: {exc}", file=sys.stderr, flush=True)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
