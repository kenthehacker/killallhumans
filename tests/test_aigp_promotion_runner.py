from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from planning.artifact_cache import ArtifactStore
from scripts import aigp_promotion_runner as runner
from scripts import aigp_pytest_progress as progress_plugin


_ROOT = Path(__file__).resolve().parents[1]
_KEY = "a" * 64


def _identity() -> dict:
    return {
        "schema": "aigp-promotion-test-identity/1",
        "repository": "killallhumans",
        "commit": "b" * 40,
        "development_lock": {
            "path": "requirements/development-test.lock.txt",
            "sha256": "c" * 64,
        },
        "python": {
            "implementation": "cpython",
            "version": "3.12.0",
            "cache_tag": "cpython-312",
            "platform": "win32",
            "machine": "AMD64",
            "environment_sha256": "d" * 64,
        },
        "pytest": {
            "argv": list(runner._SEMANTIC_PYTEST_ARGV),
            "cwd": "$REPOSITORY_ROOT",
            "environment_policy": "aigp-promotion-pytest-environment/1",
            "environment_sha256": "2" * 64,
        },
        "supervisor": {
            "policy": "aigp-promotion-supervisor/1",
            "wall_timeout_s": 900.0,
        },
    }


def _result(identity: dict, *, attempt_id: str = "attempt-1", code: int = 0) -> dict:
    return {
        "schema": "aigp-promotion-test-result/1",
        "key": runner._promotion_key(identity),
        "identity": identity,
        "attempt_id": attempt_id,
        "outcome": "passed" if code == 0 else "failed",
        "runner_exit_code": code,
        "elapsed_s": 1.25,
        "junit": {
            "summary": {
                "tests": 2,
                "failures": int(code != 0),
                "errors": 0,
                "skipped": 0,
            }
        },
        "output": {"path": "external/pytest.log"},
    }


def _status(key: str, attempt_directory: Path, attempt_id: str = "attempt-1") -> dict:
    return {
        "schema": "aigp-promotion-test-status/1",
        "key": key,
        "attempt_id": attempt_id,
        "phase": "running",
        "owner_pid": 123,
        "started_at_utc": "2026-07-21T00:00:00+00:00",
        "heartbeat_at_utc": "2026-07-21T00:00:01+00:00",
        "heartbeat_sequence": 1,
        "attempt_directory": str(attempt_directory),
        "progress": None,
    }


def _patch_identity(monkeypatch: pytest.MonkeyPatch, identity: dict) -> None:
    monkeypatch.setattr(runner, "_promotion_identity", lambda repository: identity)


def _external_state_root(tmp_path: Path) -> Path:
    return runner._state_root(_ROOT, str(tmp_path))


def _attempt_directory(
    state_root: Path, key: str, attempt_id: str = "attempt-1"
) -> Path:
    path = state_root / "attempts" / key / attempt_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_key_material_changes_for_each_bound_input():
    identity = _identity()
    baseline = runner._promotion_key(identity)
    mutations = (
        ("commit", "e" * 40),
        ("development_lock.sha256", "f" * 64),
        ("python.version", "3.12.1"),
        ("python.environment_sha256", "1" * 64),
        ("pytest.argv", [*identity["pytest"]["argv"], "--new-policy"]),
        ("pytest.environment_sha256", "3" * 64),
        ("supervisor.wall_timeout_s", 901.0),
    )
    for field, value in mutations:
        changed = copy.deepcopy(identity)
        target = changed
        parts = field.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        assert runner._promotion_key(changed) != baseline


def test_key_material_excludes_worktree_and_state_paths():
    encoded = json.dumps(_identity(), sort_keys=True)
    assert "worktree" not in encoded.casefold()
    assert runner._STATE_ROOT_ENV not in encoded
    assert "{junitxml}" in encoded
    assert "{basetemp}" in encoded


def test_promotion_command_is_visible_bounded_and_never_live(tmp_path):
    command = runner._pytest_command(tmp_path)
    assert command[1:6] == ["-B", "-s", "-u", "-m", "pytest"]
    assert "-vv" in command
    assert "--color=no" in command
    assert "--durations=25" in command
    assert "scripts.aigp_pytest_progress" in command
    assert "no:cacheprovider" in command
    marker_index = command.index("-m", command.index("pytest") + 1)
    assert command[marker_index + 1] == "not live"
    assert "--timeout=300" in command
    assert all(forbidden not in command for forbidden in ("preflight", "gate0", "hover"))
    assert all(root not in command for root in ("tests", "competition/tests"))


def test_output_capture_keeps_draining_after_log_write_failure():
    payload = b"x" * 131_073

    class Read1Only(io.BytesIO):
        def read(self, size=-1):
            raise AssertionError("buffered read() would delay live output")

    class FailingLog:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def write(self, chunk):
            raise OSError("simulated full disk")

        def flush(self):
            return None

        def fileno(self):
            return -1

    class RecordingConsole:
        def __init__(self):
            self.payload = bytearray()

        def publish(self, chunk):
            self.payload.extend(chunk)

    console = RecordingConsole()
    capture = runner._OutputCapture(Read1Only(payload), FailingLog(), console)
    capture.start()
    output = capture.finish()

    assert output["bytes_seen"] == len(payload)
    assert output["bytes_stored"] == 0
    assert output["error"] == "OSError: simulated full disk"
    assert bytes(console.payload) == payload


def test_promotion_environment_scrubs_inherited_pytest_controls(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-m live")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted_plugin")
    progress = tmp_path / "progress.json"
    environment = runner._pytest_environment(tmp_path, progress, _KEY)
    assert "PYTEST_ADDOPTS" not in environment
    assert "PYTEST_PLUGINS" not in environment
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["AIGP_PROMOTION_RUN_KEY"] == _KEY
    assert Path(environment["AIGP_CACHE_ROOT"]).is_relative_to(tmp_path)
    assert Path(environment["PYTHONPYCACHEPREFIX"]).is_relative_to(tmp_path)


def test_promotion_environment_key_binds_inherited_semantic_values(monkeypatch):
    monkeypatch.setenv("LANG", "promotion-language-one")
    first = runner.json_hash(runner._pytest_environment_key_material())
    monkeypatch.setenv("LANG", "promotion-language-two")
    second = runner.json_hash(runner._pytest_environment_key_material())
    assert first != second


def test_clean_commit_rejects_ignored_behavioral_inputs(tmp_path, monkeypatch):
    commit = "b" * 40

    def checked(argv, *, cwd):
        if argv[1:3] == ["rev-parse", "HEAD"]:
            return commit
        if "--ignored=matching" in argv:
            return "!! control/residual_weights.npz\0"
        return ""

    monkeypatch.setattr(runner, "run_checked", checked)
    with pytest.raises(runner.PromotionRunError, match="exact pristine checkout"):
        runner._clean_commit(tmp_path)


def test_state_root_is_external_and_shared_by_common_git_directory(
    tmp_path, monkeypatch
):
    common = tmp_path / "shared" / ".git"
    common.mkdir(parents=True)
    external = tmp_path / "state"
    monkeypatch.setattr(runner, "_git_common_directory", lambda repository: common)
    first = runner._state_root(_ROOT, str(external))
    second = runner._state_root(_ROOT, str(external))
    assert first == second
    assert not runner._is_within(first, _ROOT)


def test_state_root_rejects_candidate_local_state(tmp_path):
    with pytest.raises(runner.PromotionRunError, match="outside"):
        runner._state_root(_ROOT, str(_ROOT / ".artifacts" / tmp_path.name))


def test_state_root_rejects_shared_git_directory(tmp_path, monkeypatch):
    common = tmp_path / "shared" / ".git"
    common.mkdir(parents=True)
    monkeypatch.setattr(runner, "_git_common_directory", lambda repository: common)
    monkeypatch.setattr(
        runner, "_git_worktree_directories", lambda repository: (_ROOT,)
    )
    with pytest.raises(runner.PromotionRunError, match="shared Git"):
        runner._state_root(_ROOT, str(common / "promotion-state"))


def test_state_root_rejects_another_linked_worktree(tmp_path, monkeypatch):
    common = tmp_path / "shared" / ".git"
    common.mkdir(parents=True)
    other_worktree = tmp_path / "other-worktree"
    other_worktree.mkdir()
    monkeypatch.setattr(runner, "_git_common_directory", lambda repository: common)
    monkeypatch.setattr(
        runner,
        "_git_worktree_directories",
        lambda repository: (_ROOT, other_worktree),
    )
    with pytest.raises(runner.PromotionRunError, match="every candidate worktree"):
        runner._state_root(_ROOT, str(other_worktree / "promotion-state"))


def test_status_attempt_directory_is_bound_to_external_state(tmp_path):
    state_root = _external_state_root(tmp_path)
    expected = _attempt_directory(state_root, _KEY)
    status = _status(_KEY, expected)
    status["attempt_directory"] = str(tmp_path / "untrusted-indirection")
    with pytest.raises(runner.PromotionRunError, match="bound state path"):
        runner._validated_status(status, _KEY, state_root)


@pytest.mark.parametrize("code", [0, 1])
def test_terminal_result_is_reused_without_executing_pytest(
    tmp_path, monkeypatch, capsys, code
):
    identity = _identity()
    key = runner._promotion_key(identity)
    store = ArtifactStore(_external_state_root(tmp_path))
    store.save_json(runner._IDENTITY_NAMESPACE, key, identity)
    store.save_json(runner._RESULT_NAMESPACE, key, _result(identity, code=code))
    _patch_identity(monkeypatch, identity)
    monkeypatch.setattr(
        runner,
        "_execute_under_lock",
        lambda *args, **kwargs: pytest.fail("cached result must prevent execution"),
    )
    assert (
        runner.run_promotion(
            _ROOT, configured_state_root=str(tmp_path), fresh=False
        )
        == code
    )
    assert "reused" in capsys.readouterr().out


def test_incomplete_attempt_refuses_automatic_rerun(tmp_path, monkeypatch):
    identity = _identity()
    key = runner._promotion_key(identity)
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    store.save_json(runner._IDENTITY_NAMESPACE, key, identity)
    attempt_directory = _attempt_directory(state_root, key)
    store.save_json(
        runner._STATUS_NAMESPACE, key, _status(key, attempt_directory)
    )
    _patch_identity(monkeypatch, identity)
    with pytest.raises(runner.PromotionRunError) as exc_info:
        runner.run_promotion(_ROOT, configured_state_root=str(tmp_path))
    assert "refusing an automatic rerun" in str(exc_info.value)
    assert str(attempt_directory) in str(exc_info.value)


def test_incomplete_new_attempt_does_not_reuse_an_older_result(
    tmp_path, monkeypatch
):
    identity = _identity()
    key = runner._promotion_key(identity)
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    store.save_json(runner._IDENTITY_NAMESPACE, key, identity)
    store.save_json(
        runner._RESULT_NAMESPACE,
        key,
        _result(identity, attempt_id="old-attempt"),
    )
    store.save_json(
        runner._STATUS_NAMESPACE,
        key,
        _status(
            key,
            _attempt_directory(state_root, key, "new-attempt"),
            attempt_id="new-attempt",
        ),
    )
    _patch_identity(monkeypatch, identity)
    with pytest.raises(runner.PromotionRunError, match="refusing an automatic rerun"):
        runner.run_promotion(_ROOT, configured_state_root=str(tmp_path))


def test_corrupt_result_is_not_treated_as_a_cache_miss(tmp_path, monkeypatch):
    identity = _identity()
    key = runner._promotion_key(identity)
    store = ArtifactStore(_external_state_root(tmp_path))
    store.save_json(runner._IDENTITY_NAMESPACE, key, identity)
    result_path = store.path(runner._RESULT_NAMESPACE, key, ".json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("{not-json", encoding="utf-8")
    _patch_identity(monkeypatch, identity)
    with pytest.raises(runner.PromotionRunError, match="corrupt durable"):
        runner.run_promotion(_ROOT, configured_state_root=str(tmp_path))


def test_second_caller_attaches_and_reuses_active_result(tmp_path, monkeypatch):
    identity = _identity()
    key = runner._promotion_key(identity)
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    runner._ensure_identity(store, key, identity)
    attempt_directory = _attempt_directory(state_root, key)
    ready = threading.Event()

    def owner() -> None:
        with store.lock(runner._RUN_LOCK_NAMESPACE, key, timeout_s=1.0):
            store.save_json(
                runner._STATUS_NAMESPACE,
                key,
                _status(key, attempt_directory),
            )
            ready.set()
            time.sleep(0.2)
            store.save_json(runner._RESULT_NAMESPACE, key, _result(identity))

    thread = threading.Thread(target=owner)
    thread.start()
    assert ready.wait(timeout=2.0)
    _patch_identity(monkeypatch, identity)
    assert runner.run_promotion(_ROOT, configured_state_root=str(tmp_path)) == 0
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_second_caller_waits_through_owner_status_startup_window(
    tmp_path, monkeypatch
):
    identity = _identity()
    key = runner._promotion_key(identity)
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    runner._ensure_identity(store, key, identity)
    attempt_directory = _attempt_directory(state_root, key)
    ready = threading.Event()

    def owner() -> None:
        with store.lock(runner._RUN_LOCK_NAMESPACE, key, timeout_s=1.0):
            ready.set()
            time.sleep(0.1)
            store.save_json(
                runner._STATUS_NAMESPACE,
                key,
                _status(key, attempt_directory),
            )
            time.sleep(0.1)
            store.save_json(runner._RESULT_NAMESPACE, key, _result(identity))

    thread = threading.Thread(target=owner)
    thread.start()
    assert ready.wait(timeout=2.0)
    _patch_identity(monkeypatch, identity)
    assert runner.run_promotion(_ROOT, configured_state_root=str(tmp_path)) == 0
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_attached_caller_switches_from_stale_to_new_attempt(tmp_path, monkeypatch):
    identity = _identity()
    key = runner._promotion_key(identity)
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    runner._ensure_identity(store, key, identity)
    old_directory = _attempt_directory(state_root, key, "old-attempt")
    new_directory = _attempt_directory(state_root, key, "new-attempt")
    store.save_json(
        runner._STATUS_NAMESPACE,
        key,
        _status(key, old_directory, attempt_id="old-attempt"),
    )
    ready = threading.Event()

    def owner() -> None:
        with store.lock(runner._RUN_LOCK_NAMESPACE, key, timeout_s=1.0):
            ready.set()
            time.sleep(0.1)
            store.save_json(
                runner._STATUS_NAMESPACE,
                key,
                _status(key, new_directory, attempt_id="new-attempt"),
            )
            time.sleep(0.1)
            store.save_json(
                runner._RESULT_NAMESPACE,
                key,
                _result(identity, attempt_id="new-attempt"),
            )

    thread = threading.Thread(target=owner)
    thread.start()
    assert ready.wait(timeout=2.0)
    _patch_identity(monkeypatch, identity)
    assert runner.run_promotion(_ROOT, configured_state_root=str(tmp_path)) == 0
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_attached_owner_exit_without_result_refuses_duplicate(
    tmp_path, monkeypatch
):
    identity = _identity()
    key = runner._promotion_key(identity)
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    runner._ensure_identity(store, key, identity)
    attempt_directory = _attempt_directory(state_root, key)
    ready = threading.Event()

    def owner() -> None:
        with store.lock(runner._RUN_LOCK_NAMESPACE, key, timeout_s=1.0):
            store.save_json(
                runner._STATUS_NAMESPACE,
                key,
                _status(key, attempt_directory),
            )
            ready.set()
            time.sleep(0.2)

    thread = threading.Thread(target=owner)
    thread.start()
    assert ready.wait(timeout=2.0)
    _patch_identity(monkeypatch, identity)
    with pytest.raises(runner.PromotionRunError, match="without a matching durable result"):
        runner.run_promotion(_ROOT, configured_state_root=str(tmp_path))
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_fresh_is_explicit_recovery_for_incomplete_attempt(tmp_path, monkeypatch):
    identity = _identity()
    key = runner._promotion_key(identity)
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    store.save_json(runner._IDENTITY_NAMESPACE, key, identity)
    store.save_json(
        runner._STATUS_NAMESPACE,
        key,
        _status(key, _attempt_directory(state_root, key)),
    )
    expected = _result(identity, attempt_id="attempt-2")
    calls = []

    def execute(*args):
        calls.append(args)
        return expected

    _patch_identity(monkeypatch, identity)
    monkeypatch.setattr(runner, "_execute_under_lock", execute)
    assert (
        runner.run_promotion(
            _ROOT, configured_state_root=str(tmp_path), fresh=True
        )
        == 0
    )
    assert len(calls) == 1


def test_terminal_reporting_happens_after_run_lock_release(tmp_path, monkeypatch):
    identity = _identity()
    key = runner._promotion_key(identity)
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    store.save_json(runner._IDENTITY_NAMESPACE, key, identity)
    store.save_json(runner._RESULT_NAMESPACE, key, _result(identity))
    reported = []

    def report(result, *, reused):
        with store.lock(runner._RUN_LOCK_NAMESPACE, key, timeout_s=0.0):
            reported.append((result["attempt_id"], reused))

    _patch_identity(monkeypatch, identity)
    monkeypatch.setattr(runner, "_print_result", report)
    assert runner.run_promotion(_ROOT, configured_state_root=str(tmp_path)) == 0
    assert reported == [("attempt-1", True)]


def test_execution_publishes_integrity_wrapped_terminal_result(
    tmp_path, monkeypatch
):
    identity = _identity()
    key = runner._promotion_key(identity)
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    _patch_identity(monkeypatch, identity)
    directory_existed_at_attempt_start = []
    original_save_json = store.save_json

    def save_json(namespace, observed_key, value):
        if (
            namespace == runner._STATUS_NAMESPACE
            and value.get("heartbeat_sequence") == 0
        ):
            directory_existed_at_attempt_start.append(
                Path(value["attempt_directory"]).exists()
            )
        return original_save_json(namespace, observed_key, value)

    monkeypatch.setattr(store, "save_json", save_json)

    def run_pytest(
        repository,
        observed_store,
        observed_key,
        attempt_id,
        attempt_directory,
        started_at,
    ):
        observed_store.save_json(
            runner._STATUS_NAMESPACE,
            observed_key,
            runner._status_payload(
                observed_key,
                attempt_id,
                attempt_directory,
                started_at=started_at,
                heartbeat_sequence=7,
            ),
        )
        return (
            0,
            False,
            False,
            {
                "sha256": "e" * 64,
                "bytes_seen": 10,
                "bytes_stored": 10,
                "truncated": False,
                "tail": "passed",
                "error": None,
            },
            None,
            None,
        )

    monkeypatch.setattr(runner, "_run_pytest", run_pytest)
    result = runner._execute_under_lock(_ROOT, store, state_root, key, identity)
    assert result["outcome"] == "passed"
    assert result["runner_exit_code"] == 0
    assert Path(result["result_path"]).is_file()
    published = runner._validated_result(
        runner._load_store_payload(store, runner._RESULT_NAMESPACE, key),
        key,
        identity,
    )
    assert published == result
    status = runner._validated_status(
        runner._load_store_payload(store, runner._STATUS_NAMESPACE, key),
        key,
        state_root,
    )
    assert status is not None
    assert status["phase"] == "finished"
    assert status["attempt_id"] == result["attempt_id"]
    assert status["heartbeat_sequence"] == 8
    assert directory_existed_at_attempt_start == [False]


def test_execution_preserves_runner_output_and_cleanup_failures(
    tmp_path, monkeypatch
):
    identity = _identity()
    key = runner._promotion_key(identity)
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    _patch_identity(monkeypatch, identity)
    observed_output = {
        "sha256": "f" * 64,
        "bytes_seen": 123,
        "bytes_stored": 100,
        "truncated": True,
        "tail": "diagnostic tail",
        "error": None,
    }
    monkeypatch.setattr(
        runner,
        "_run_pytest",
        lambda *args: (
            None,
            False,
            False,
            observed_output,
            "job cleanup failed",
            "status publication failed",
        ),
    )

    result = runner._execute_under_lock(_ROOT, store, state_root, key, identity)

    assert result["outcome"] == "infrastructure_error"
    assert result["output"]["bytes_seen"] == 123
    assert "status publication failed" in result["runner_error"]
    assert "descendant cleanup was not proved: job cleanup failed" in result["runner_error"]


def test_pytest_supervisor_enforces_aggregate_wall_timeout(tmp_path, monkeypatch):
    state_root = _external_state_root(tmp_path)
    store = ArtifactStore(state_root)
    attempt_directory = _attempt_directory(state_root, _KEY)
    monkeypatch.setattr(runner, "_PROMOTION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(
        runner,
        "_pytest_command",
        lambda attempt: [
            sys.executable,
            "-B",
            "-u",
            "-c",
            "import time; print('started', flush=True); time.sleep(5)",
        ],
    )
    started = time.monotonic()

    (
        pytest_exit_code,
        interrupted,
        timed_out,
        output,
        cleanup_error,
        run_error,
    ) = runner._run_pytest(
        _ROOT,
        store,
        _KEY,
        "attempt-1",
        attempt_directory,
        runner._utc_now(),
    )

    assert time.monotonic() - started < 4.0
    assert pytest_exit_code is None
    assert interrupted is False
    assert timed_out is True
    assert cleanup_error is None
    assert run_error is None
    assert output["bytes_seen"] > 0


def test_progress_plugin_does_not_publish_once_per_test(tmp_path, monkeypatch):
    snapshots = []
    monkeypatch.setattr(
        progress_plugin,
        "_atomic_json",
        lambda path, value: snapshots.append(copy.deepcopy(value)),
    )
    state = progress_plugin._ProgressState(tmp_path / "progress.json", _KEY, 30.0)
    state.publish(phase="running", collected=100)
    publications_before_tests = len(snapshots)

    for index in range(100):
        state.start_test(f"tests/test_probe.py::test_{index}")
        state.record_outcome("call:passed")
        state.finish_test()

    assert len(snapshots) == publications_before_tests
    state.publish()
    assert len(snapshots) == publications_before_tests + 1
    assert snapshots[-1]["completed"] == 100


def test_progress_plugin_publishes_current_test_heartbeats_and_final_state(
    tmp_path,
):
    probe = tmp_path / "test_progress_probe.py"
    probe.write_text(
        "import time\n\ndef test_deliberately_visible():\n    time.sleep(0.45)\n",
        encoding="utf-8",
    )
    progress = tmp_path / "progress.json"
    base_temp = tmp_path / "pytest-tmp"
    attempt_directory = tmp_path / "attempt"
    attempt_directory.mkdir()
    environment = runner._pytest_environment(
        attempt_directory, progress, _KEY
    )
    environment["AIGP_PROMOTION_HEARTBEAT_SECONDS"] = "0.05"
    process = subprocess.Popen(
        [
            sys.executable,
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
            str(_ROOT / "pyproject.toml"),
            "-m",
            "not live",
            "--timeout=2",
            "--durations=25",
            f"--basetemp={base_temp}",
            str(probe),
        ],
        cwd=_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    observed_sequences = set()
    observed_current = False
    deadline = time.monotonic() + 5.0
    while process.poll() is None and time.monotonic() < deadline:
        try:
            value = json.loads(progress.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Windows can transiently deny a reader while os.replace swaps
            # the diagnostic snapshot.  The production progress reader
            # treats the same condition as one unavailable heartbeat.
            time.sleep(0.01)
            continue
        if value.get("current_test"):
            observed_current = True
            observed_sequences.add(value["sequence"])
        time.sleep(0.02)
    output = process.communicate(timeout=5.0)[0]
    assert process.returncode == 0, output
    final = json.loads(progress.read_text(encoding="utf-8"))
    assert observed_current
    assert len(observed_sequences) >= 2
    assert final["phase"] == "finished"
    assert final["collected"] == 1
    assert final["completed"] == 1
    assert final["current_test"] is None
    assert final["exitstatus"] == 0
    assert "test_deliberately_visible" in output
    assert "slowest 25 durations" in output.lower()
