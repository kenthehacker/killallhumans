"""Promotion-only pytest progress and heartbeat publication.

The plugin is loaded explicitly by ``scripts.aigp_promotion_runner``.  It is
not registered globally because ordinary and nested pytest runs should keep
their existing behavior.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PROGRESS_PATH_ENV = "AIGP_PROMOTION_PROGRESS_PATH"
_RUN_KEY_ENV = "AIGP_PROMOTION_RUN_KEY"
_HEARTBEAT_SECONDS_ENV = "AIGP_PROMOTION_HEARTBEAT_SECONDS"
_SCHEMA = "aigp-promotion-pytest-progress/1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class _ProgressState:
    def __init__(self, path: Path, run_key: str, heartbeat_seconds: float) -> None:
        self.path = path
        self.run_key = run_key
        self.heartbeat_seconds = heartbeat_seconds
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.current_started: float | None = None
        self.final = False
        self.publication_error: str | None = None
        self.value: dict[str, Any] = {
            "schema": _SCHEMA,
            "run_key": run_key,
            "pid": os.getpid(),
            "sequence": 0,
            "updated_at_utc": _utc_now(),
            "phase": "starting",
            "collected": None,
            "completed": 0,
            "current_test": None,
            "current_test_elapsed_s": None,
            "last_outcome": None,
            "exitstatus": None,
        }

    def publish(self, **updates: Any) -> None:
        with self.lock:
            self.value.update(updates)
            if self.current_started is not None:
                self.value["current_test_elapsed_s"] = round(
                    max(0.0, time.monotonic() - self.current_started), 6
                )
            else:
                self.value["current_test_elapsed_s"] = None
            self.value["sequence"] += 1
            self.value["updated_at_utc"] = _utc_now()
            snapshot = dict(self.value)
            try:
                _atomic_json(self.path, snapshot)
                self.publication_error = None
            except (OSError, TypeError, ValueError) as exc:
                # Progress is diagnostic evidence.  The supervisor still owns
                # pytest's exit result and cleanup decision if publication is
                # temporarily unavailable.
                self.publication_error = f"{type(exc).__name__}: {exc}"

    def start_test(self, nodeid: str) -> None:
        with self.lock:
            self.current_started = time.monotonic()
            self.value.update(phase="running", current_test=nodeid)

    def record_outcome(self, outcome: str) -> None:
        with self.lock:
            self.value["last_outcome"] = outcome

    def finish_test(self) -> None:
        with self.lock:
            self.current_started = None
            self.value["completed"] = int(self.value["completed"]) + 1
            self.value["current_test"] = None

    def start(self) -> None:
        self.publish(phase="collecting")

        def heartbeat() -> None:
            while not self.stop.wait(self.heartbeat_seconds):
                self.publish()

        self.thread = threading.Thread(
            target=heartbeat,
            name="aigp-promotion-pytest-heartbeat",
            daemon=True,
        )
        self.thread.start()

    def publish_terminal(self, **updates: Any) -> None:
        deadline = time.monotonic() + 1.0
        while True:
            self.publish(**updates)
            if self.publication_error is None or time.monotonic() >= deadline:
                return
            time.sleep(0.02)

    def finish(self, exitstatus: int) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.heartbeat_seconds * 2.0))
        self.current_started = None
        self.final = True
        self.publish_terminal(
            phase="finished",
            current_test=None,
            exitstatus=int(exitstatus),
        )

    def close_interrupted(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.heartbeat_seconds * 2.0))
        if not self.final:
            self.current_started = None
            self.publish_terminal(phase="interrupted", current_test=None)


_STATE: _ProgressState | None = None


def pytest_configure(config: Any) -> None:
    del config
    global _STATE
    if _STATE is not None:
        raise RuntimeError("promotion progress plugin was configured twice")
    raw_path = os.environ.get(_PROGRESS_PATH_ENV)
    run_key = os.environ.get(_RUN_KEY_ENV, "")
    if not raw_path:
        raise RuntimeError(f"{_PROGRESS_PATH_ENV} is required")
    if len(run_key) != 64 or any(char not in "0123456789abcdef" for char in run_key):
        raise RuntimeError(f"{_RUN_KEY_ENV} must be a SHA-256 digest")
    try:
        heartbeat_seconds = float(
            os.environ.get(_HEARTBEAT_SECONDS_ENV, "30.0")
        )
    except ValueError as exc:
        raise RuntimeError(
            f"{_HEARTBEAT_SECONDS_ENV} must be numeric"
        ) from exc
    if not math.isfinite(heartbeat_seconds) or not 0.01 <= heartbeat_seconds <= 300.0:
        raise RuntimeError(
            f"{_HEARTBEAT_SECONDS_ENV} must be finite and between 0.01 and 300"
        )
    path = Path(raw_path).expanduser().resolve()
    _STATE = _ProgressState(path, run_key, heartbeat_seconds)


def pytest_sessionstart(session: Any) -> None:
    del session
    assert _STATE is not None
    _STATE.start()


def pytest_collection_finish(session: Any) -> None:
    assert _STATE is not None
    _STATE.publish(phase="running", collected=len(session.items))


def pytest_runtest_logstart(nodeid: str, location: Any) -> None:
    del location
    assert _STATE is not None
    _STATE.start_test(nodeid)


def pytest_runtest_logreport(report: Any) -> None:
    assert _STATE is not None
    if report.when == "call" or report.failed or report.skipped:
        _STATE.record_outcome(f"{report.when}:{report.outcome}")


def pytest_runtest_logfinish(nodeid: str, location: Any) -> None:
    del location
    assert _STATE is not None
    _STATE.finish_test()


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    del session
    assert _STATE is not None
    _STATE.finish(int(exitstatus))


def pytest_unconfigure(config: Any) -> None:
    del config
    global _STATE
    if _STATE is not None:
        _STATE.close_interrupted()
        _STATE = None
