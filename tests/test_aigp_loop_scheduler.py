from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import aigp_loop.scheduler as scheduler_module

from aigp_loop._util import environment_fingerprint, git_provenance, json_hash
from aigp_loop.ledger import TrialKey, TrialLedger
from aigp_loop.promotion import PromotionLadder, Tier
from aigp_loop.scheduler import (
    CommandStep,
    GitWorktreePool,
    SingleMerger,
    TierCommand,
    TrialScheduler,
    load_tier_commands,
)
from scripts.aigp_trials import main as trials_main
from scripts.aigp_pytest import audit_candidate


def _repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "candidate.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "candidate.py"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "candidate"], cwd=path, check=True)
    return path


def _trial(ledger: TrialLedger, repo: Path, name: str = "one") -> str:
    commit, dirty, code = git_provenance(repo)
    config = {"name": name}
    identifier, _ = ledger.create_or_get_trial(
        key=TrialKey(code, json_hash(config), "dataset", 1, "eval-v1"),
        commit_hash=commit,
        dirty_diff_hash=dirty,
        resolved_config=config,
        environment_fingerprint=environment_fingerprint(),
    )
    return identifier


@pytest.mark.parametrize("link_kind", ["file", "parent"])
def test_scheduler_trusted_hash_verification_rejects_link_components(
    tmp_path, link_kind
):
    root = tmp_path / "trusted-root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    target = nested / "evaluator.py"
    target.write_text("trusted", encoding="utf-8")
    digest = scheduler_module.sha256_file(target)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evaluator.py").write_text("trusted", encoding="utf-8")
    try:
        if link_kind == "file":
            target.unlink()
            target.symlink_to(outside / "evaluator.py")
        else:
            target.unlink()
            nested.rmdir()
            nested.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable on this host: {exc}")
    step = CommandStep(
        argv=(sys.executable, "-c", "pass"),
        timeout_s=1.0,
        trusted_files_sha256=(("nested/evaluator.py", digest),),
    )
    with pytest.raises(RuntimeError, match="unsafe"):
        TrialScheduler._verify_trusted_files(root, step)


def test_scheduler_rejects_canonical_trust_manifest_digest_mismatch(tmp_path):
    root = tmp_path / "trusted-root"
    manifest = root / scheduler_module._TRUSTED_MANIFEST_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"schema":"changed"}\n', encoding="utf-8")
    step = CommandStep(
        argv=(sys.executable, "-c", "pass"),
        timeout_s=1.0,
        trusted_files_sha256=(
            (scheduler_module._TRUSTED_MANIFEST_PATH, "0" * 64),
        ),
    )
    with pytest.raises(RuntimeError, match="trusted evaluator hash mismatch"):
        TrialScheduler._verify_trusted_files(root, step)


def test_scheduler_uses_bounded_pipe_drains_for_hostile_output(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                (
                    sys.executable,
                    "-c",
                    "import sys;sys.stdout.write('x'*8000000);sys.stderr.write('y'*8000000)",
                ),
                timeout_s=10.0,
            )
        },
        owner="scheduler-test",
        lease_ttl_s=3.0,
    )
    monkeypatch.setattr(
        scheduler_module.tempfile,
        "TemporaryFile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("scheduler output must not use an unbounded temporary file")
        ),
    )
    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifier
    row = ledger.get_trial(identifier)
    assert row["status"] == "failed"
    assert ledger.completed_tiers(identifier) == ()
    checkpoint = ledger.get_checkpoint(identifier, 0)
    assert checkpoint["status"] == "failed"
    assert "exceeded 1000000 bytes" in checkpoint["stdout_stderr_tail"]
    assert len(checkpoint["stdout_stderr_tail"]) <= 32_000


def test_truncated_metrics_tail_cannot_be_parsed_or_promoted(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    program = (
        "import json,sys; "
        "sys.stdout.write(' ' * 1100000); "
        "sys.stdout.write(json.dumps({'apparently_valid': True}) + '\\n')"
    )
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                ("{python}", "-c", program),
                timeout_s=10.0,
                metrics_from_stdout=True,
                require_hard_gates=False,
            )
        },
        owner="truncated-metrics-test",
        lease_ttl_s=3.0,
    )

    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifier
    row = ledger.get_trial(identifier)
    checkpoint = ledger.get_checkpoint(identifier, 0)
    assert row["status"] == "failed"
    assert checkpoint["status"] == "failed"
    assert checkpoint["metrics"] == {}
    assert "exceeded 1000000 bytes" in checkpoint["stdout_stderr_tail"]


def test_scheduler_and_merger_share_one_orchestration_lease(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED, (sys.executable, "-c", "pass"), 1.0
            )
        },
        owner="scheduler-owner",
        lease_ttl_s=3.0,
    )
    assert ledger.acquire_global_lease(
        scheduler_module._ORCHESTRATION_LEASE, "merger-owner", ttl_s=30.0
    )
    with pytest.raises(RuntimeError, match="scheduler owns"):
        scheduler.run_once(through=Tier.T0_AFFECTED)
    ledger.release_global_lease(
        scheduler_module._ORCHESTRATION_LEASE, "merger-owner"
    )

    monkeypatch.setattr(scheduler_module, "validate_promotion_chain", lambda *args: {})
    assert ledger.acquire_global_lease(
        scheduler_module._ORCHESTRATION_LEASE, "scheduler-owner", ttl_s=30.0
    )
    with pytest.raises(RuntimeError, match="merger owns"):
        SingleMerger(ledger, repo, owner="merger-owner").merge_completed(identifier)
    ledger.release_global_lease(
        scheduler_module._ORCHESTRATION_LEASE, "scheduler-owner"
    )


@pytest.mark.parametrize(
    "relative",
    [
        "competition/conftest.py",
        "pytest.ini",
        "gate_detection/pytest.ini",
        "estimation/sitecustomize.py",
        "gate_detection/injected_plugin.pth",
        "pytest.py",
        "nested/pytest/__init__.py",
    ],
)
def test_t1_pytest_audit_rejects_candidate_discovery_and_startup_hooks(
    tmp_path, relative
):
    root = tmp_path / "candidate"
    policy = {
        "schema": "aigp-t1-pytest-policy/1",
        "expected_passed": 1,
        "pytest_version": "9.1.1",
        "pytest_timeout_version": "2.4.0",
        "test_files": ["tests/test_one.py"],
        "trusted_discovery_files": ["conftest.py", "pyproject.toml"],
    }
    required = {
        "conftest.py",
        "pyproject.toml",
        "tests/test_one.py",
        "config/t1_pytest.ini",
        "config/t1_pytest_policy.json",
        "scripts/aigp_pytest.py",
    }
    hashes = {}
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        hashes[name] = scheduler_module.sha256_file(path)
    manifest = {"schema": "aigp-trusted-evaluator-files/1", "files": hashes}
    assert len(audit_candidate(root, manifest, policy)) == 1
    injected = root / relative
    injected.parent.mkdir(parents=True, exist_ok=True)
    injected.write_text("raise RuntimeError('must never load')", encoding="utf-8")
    with pytest.raises(ValueError, match="unreviewed"):
        audit_candidate(root, manifest, policy)


def test_t1_pytest_audit_rejects_executable_bytecode_cache(tmp_path):
    root = tmp_path / "candidate"
    policy = {
        "schema": "aigp-t1-pytest-policy/1",
        "expected_passed": 1,
        "pytest_version": "9.1.1",
        "pytest_timeout_version": "2.4.0",
        "test_files": ["tests/test_one.py"],
        "trusted_discovery_files": ["conftest.py", "pyproject.toml"],
    }
    required = {
        "conftest.py",
        "pyproject.toml",
        "tests/test_one.py",
        "config/t1_pytest.ini",
        "config/t1_pytest_policy.json",
        "scripts/aigp_pytest.py",
    }
    hashes = {}
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        hashes[name] = scheduler_module.sha256_file(path)
    manifest = {"schema": "aigp-trusted-evaluator-files/1", "files": hashes}
    runtime_cache = root / ".venv" / "Lib" / "site-packages" / "__pycache__"
    runtime_cache.mkdir(parents=True)
    (runtime_cache / "runtime.cpython-312.pyc").write_bytes(b"venv runtime")
    assert len(audit_candidate(root, manifest, policy)) == 1
    cache = root / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "test_one.cpython-312.pyc").write_bytes(b"executable")
    with pytest.raises(ValueError, match="executable bytecode"):
        audit_candidate(root, manifest, policy)


def test_successful_command_cannot_leave_background_descendant(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    marker = tmp_path / "escaped-child.txt"
    child_code = (
        "import pathlib,time; time.sleep(0.8); "
        f"pathlib.Path({str(marker)!r}).write_text('escaped', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}])"
    )
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED, ("{python}", "-c", parent_code), 5.0
            )
        },
        owner="containment-test",
        lease_ttl_s=3.0,
    )
    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifier
    time.sleep(1.0)
    assert not marker.exists()


def test_taskkill_failure_is_not_claimed_as_tree_cleanup(monkeypatch):
    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

        def wait(self, timeout):
            return 1

    process = FakeProcess()
    monkeypatch.setattr(scheduler_module.os, "name", "nt")
    monkeypatch.setattr(
        scheduler_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(RuntimeError, match="did not prove"):
        TrialScheduler._terminate_process_tree(process)
    assert process.killed


def test_enqueue_rejects_dirty_candidate_instead_of_misrepresenting_worktree(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "candidate.py").write_text("VALUE = 2\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="commit the exact code first"):
        trials_main(
            [
                "--ledger",
                str(tmp_path / "ledger.sqlite3"),
                "enqueue",
                "--repo",
                str(repo),
                "--config",
                str(config),
                "--dataset-hash",
                "data",
                "--evaluator-version",
                "eval",
                "--seed",
                "1",
            ]
        )


def test_metric_tier_fails_closed_when_hard_gate_evidence_is_missing(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                (sys.executable, "-c", "print('{}')"),
                5.0,
                metrics_from_stdout=True,
                require_hard_gates=True,
            ),
        },
        owner="scheduler-test",
        lease_ttl_s=3.0,
    )
    scheduler.run_once(through=Tier.T0_AFFECTED)
    row = ledger.get_trial(identifier)
    assert row["status"] == "failed"
    assert "missing hard-gate evidence" in row["failure_reason"]
    assert ledger.completed_tiers(identifier) == ()


def test_worktree_provenance_mismatch_fails_before_command(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)

    class DirtyPool:
        def ensure(self, _trial_id, _commit):
            (repo / "candidate.py").write_text("VALUE = 99\n", encoding="utf-8")
            return repo

    scheduler = TrialScheduler(
        ledger,
        DirtyPool(),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                (sys.executable, "-c", "raise SystemExit('must not run')"),
                5.0,
            )
        },
        owner="scheduler-test",
        lease_ttl_s=3.0,
    )
    scheduler.run_once(through=Tier.T0_AFFECTED)
    row = ledger.get_trial(identifier)
    assert row["status"] == "failed"
    assert "provenance" in row["failure_reason"]
    assert ledger.completed_tiers(identifier) == ()


def test_command_that_mutates_candidate_or_materialized_config_fails_postcheck(tmp_path):
    for target in ("candidate", "config"):
        repo = _repo(tmp_path / f"repo-{target}")
        ledger = TrialLedger(tmp_path / f"ledger-{target}.sqlite3")
        identifier = _trial(ledger, repo, target)
        code = (
            "from pathlib import Path; Path('candidate.py').write_text('VALUE = 9\\n')"
            if target == "candidate"
            else (
                "import os,stat; p=os.environ['AIGP_RESOLVED_CONFIG']; "
                "os.chmod(p, stat.S_IWRITE); open(p,'w',encoding='utf-8').write('{}')"
            )
        )
        scheduler = TrialScheduler(
            ledger,
            GitWorktreePool(repo, tmp_path / f"worktrees-{target}"),
            {
                Tier.T0_AFFECTED: TierCommand(
                    Tier.T0_AFFECTED, ("{python}", "-c", code), 5.0
                )
            },
            owner=f"integrity-{target}",
            lease_ttl_s=3.0,
        )
        scheduler.run_once(through=Tier.T0_AFFECTED)
        row = ledger.get_trial(identifier)
        assert row["status"] == "failed"
        assert row["failure_reason"] == "trial input integrity failed"


def test_resume_rejects_ignored_payload_left_by_crashed_candidate_before_execution(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path / "repo-crash-poison")
    (repo / ".gitignore").write_text("control/\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "ignore runtime models"], cwd=repo, check=True)
    ledger = TrialLedger(tmp_path / "crash-poison.sqlite3")
    identifier = _trial(ledger, repo, "crash-poison")
    pool = GitWorktreePool(repo, tmp_path / "worktrees-crash-poison")
    row = ledger.get_trial(identifier)
    worktree = pool.ensure(identifier, row["commit_hash"])
    poison = worktree / "control" / "residual_weights.npz"
    poison.parent.mkdir()
    poison.write_bytes(b"ignored payload left before scheduler crash")
    assert subprocess.run(
        ["git", "check-ignore", "-q", str(poison)], cwd=worktree
    ).returncode == 0

    scheduler = TrialScheduler(
        ledger,
        pool,
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                ("{python}", "-c", "raise AssertionError('must never execute')"),
                5.0,
            )
        },
        owner="crash-poison-resume",
        lease_ttl_s=3.0,
    )
    monkeypatch.setattr(
        scheduler,
        "_run_tier_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("poisoned worktree reached candidate execution")
        ),
    )
    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifier
    failed = ledger.get_trial(identifier)
    assert failed["status"] == "failed"
    assert "not an exact pristine checkout" in failed["failure_reason"]


def test_worktree_pool_rejects_trial_id_directory_alias(tmp_path):
    repo = _repo(tmp_path / "repo-alias")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    pool = GitWorktreePool(repo, tmp_path / "worktrees-alias")
    first = pool.ensure("trial-a", commit)
    alias = pool.root / "trial-b"
    try:
        alias.symlink_to(first, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks unavailable on this host: {exc}")
    with pytest.raises(ValueError, match="symlink|reparse"):
        pool.ensure("trial-b", commit)


def test_worktree_pool_rejects_dangling_trial_id_alias(tmp_path):
    repo = _repo(tmp_path / "repo-dangling-alias")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    pool = GitWorktreePool(repo, tmp_path / "worktrees-dangling-alias")
    alias = pool.root / "trial-dangling"
    try:
        alias.symlink_to(
            tmp_path / "missing-worktree-target", target_is_directory=True
        )
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks unavailable on this host: {exc}")
    assert not alias.exists()
    with pytest.raises(ValueError, match="symlink|reparse"):
        pool.ensure("trial-dangling", commit)


def test_worktree_pool_checks_lexical_leaf_without_following_exists(
    tmp_path, monkeypatch
):
    """The pre-Git guard must not depend on link-following ``exists()``."""

    repo = _repo(tmp_path / "repo-lexical-leaf")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    pool = GitWorktreePool(repo, tmp_path / "worktrees-lexical-leaf")
    leaf = pool.root / "trial-lexical"
    leaf.write_text("reserved by another owner", encoding="utf-8")
    real_exists = Path.exists

    def link_following_exists(path):
        return False if path == leaf else real_exists(path)

    monkeypatch.setattr(Path, "exists", link_following_exists)
    with pytest.raises(ValueError, match="not a directory"):
        pool.ensure("trial-lexical", commit)


def test_posix_process_group_cleanup_reaps_leader(monkeypatch):
    """A signaled leader must be waited before the group-absence proof."""

    import aigp_loop.scheduler as scheduler_module

    state = {"signaled": False, "polls": 0, "waits": 0}

    class FakeProcess:
        pid = 43210
        returncode = None

        def poll(self):
            state["polls"] += 1
            if state["signaled"]:
                self.returncode = -15
            return self.returncode

        def wait(self, timeout=None):
            state["waits"] += 1
            self.returncode = -15
            return self.returncode

    def fake_killpg(_group_id, requested_signal):
        if requested_signal == scheduler_module.signal.SIGTERM:
            state["signaled"] = True
            return
        if requested_signal == 0:
            if state["polls"] == 0:
                return
            raise ProcessLookupError
        raise AssertionError(f"unexpected signal: {requested_signal}")

    monkeypatch.setattr(scheduler_module.os, "name", "posix")
    monkeypatch.setattr(scheduler_module.os, "killpg", fake_killpg, raising=False)
    process = FakeProcess()
    TrialScheduler._terminate_process_tree(process)
    assert process.returncode == -15
    assert state["polls"] >= 1


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX process groups")
def test_posix_process_group_cleanup_proves_real_leader_gone():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        TrialScheduler._terminate_process_tree(process)
        assert process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)


def test_run_once_reconciles_failed_checkpoint_without_rerunning(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED, ("{python}", "-c", "raise SystemExit(2)"), 5.0
            )
        },
        owner="resume-failed",
        lease_ttl_s=3.0,
    )
    calls = {"commands": 0, "finishes": 0}
    real_run = scheduler._run_tier_command
    real_finish = ledger.finish_trial

    def counted(*args, **kwargs):
        calls["commands"] += 1
        return real_run(*args, **kwargs)

    def crash_once(*args, **kwargs):
        calls["finishes"] += 1
        if calls["finishes"] == 1:
            raise RuntimeError("crash after checkpoint")
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_run_tier_command", counted)
    monkeypatch.setattr(ledger, "finish_trial", crash_once)
    with pytest.raises(RuntimeError, match="crash after checkpoint"):
        scheduler.run_once(through=Tier.T0_AFFECTED)
    assert ledger.get_checkpoint(identifier, 0)["status"] == "failed"
    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifier
    assert calls["commands"] == 1
    assert ledger.get_trial(identifier)["status"] == "failed"


def test_run_round_reconciles_completed_checkpoint_before_yield(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifiers = [_trial(ledger, repo, f"candidate-{index}") for index in range(2)]
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED, ("{python}", "-c", "pass"), 5.0
            )
        },
        owner="resume-round",
        lease_ttl_s=3.0,
    )
    calls = {"commands": 0, "yields": 0}
    real_run = scheduler._run_tier_command
    real_yield = ledger.yield_trial

    def counted(*args, **kwargs):
        calls["commands"] += 1
        return real_run(*args, **kwargs)

    def crash_once(*args, **kwargs):
        calls["yields"] += 1
        if calls["yields"] == 1:
            raise RuntimeError("crash before yield")
        return real_yield(*args, **kwargs)

    monkeypatch.setattr(scheduler, "_run_tier_command", counted)
    monkeypatch.setattr(ledger, "yield_trial", crash_once)
    with pytest.raises(RuntimeError, match="crash before yield"):
        scheduler.run_round(Tier.T0_AFFECTED)
    decision = scheduler.run_round(Tier.T0_AFFECTED)
    assert set(decision["promoted"]) == set(identifiers)
    assert calls["commands"] == 2
    assert all(ledger.get_trial(item)["status"] == "pending" for item in identifiers)


def test_invalid_t1_replay_is_ineligible_before_any_later_tier():
    invalid_replay = {
        "schema": "aigp-vq2-replay-score/1",
        "processor": "candidate:run",
        "processor_code_sha256": "c" * 64,
        "candidate_isolation": {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "denied",
            "filesystem": "readonly-worktree-only",
            "non_interactive": True,
            "process_tree_containment": "kill-on-wrapper-exit",
            "host_process_access": "denied",
            "wrapper_sha256": "f" * 64,
        },
        "domain_provenance": {
            "perception": "candidate_detector_on_all_decoded_frames",
            "estimator": "candidate_estimator_on_ordered_sanitized_stream",
            "open_loop_commands": "candidate_generator_on_ordered_sanitized_stream",
        },
        "evaluation_input_hash": "e" * 64,
        "policy": {"passed": False},
        "perception": {},
    }
    evaluation = TrialScheduler._promotion_evaluation(
        "candidate", Tier.T1_VQ2_REPLAY, invalid_replay
    )
    assert evaluation.eligibility is not None
    assert evaluation.eligibility.passed is False
    assert PromotionLadder().decide([evaluation]).promoted == ()


def test_scheduler_refuses_untrusted_t2_to_t4_commands(tmp_path):
    repo = _repo(tmp_path / "repo-untrusted")
    ledger = TrialLedger(tmp_path / "untrusted.sqlite3")
    with pytest.raises(ValueError, match="trusted evaluator file binding"):
        TrialScheduler(
            ledger,
            GitWorktreePool(repo, tmp_path / "worktrees-untrusted"),
            {
                Tier.T2_WARM_SIM: TierCommand(
                    Tier.T2_WARM_SIM, ("{python}", "-c", "pass"), 5.0
                )
            },
        )


def test_scheduler_api_rejects_unwrapped_or_fabricated_metrics_t1(tmp_path):
    repo = _repo(tmp_path / "repo-t1-api")
    ledger = TrialLedger(tmp_path / "t1-api.sqlite3")
    pool = GitWorktreePool(repo, tmp_path / "worktrees-t1-api")
    with pytest.raises(ValueError, match="trusted evaluator file binding"):
        TrialScheduler(
            ledger,
            pool,
            {
                Tier.T1_VQ2_REPLAY: TierCommand(
                    Tier.T1_VQ2_REPLAY,
                    ("{python}", "-c", "print('{}')"),
                    1.0,
                    metrics_from_stdout=True,
                )
            },
        )

    fabricated = CommandStep(
        ("{python}", "-c", "print('{}')"),
        1.0,
        metrics_from_stdout=True,
        trusted_files_sha256=(("candidate.py", "0" * 64),),
        isolation_wrapper=str(tmp_path / "wrapper.exe"),
        isolation_wrapper_sha256="1" * 64,
    )
    with pytest.raises(ValueError, match="trusted-host replay metrics"):
        TrialScheduler(
            ledger,
            pool,
            {
                Tier.T1_VQ2_REPLAY: TierCommand(
                    tier=Tier.T1_VQ2_REPLAY,
                    steps=(fabricated,),
                )
            },
        )


def test_trusted_t1_replay_shape_requires_complete_host_import_closure(tmp_path):
    repo = _repo(tmp_path / "repo-t1-closure")
    ledger = TrialLedger(tmp_path / "t1-closure.sqlite3")
    incomplete = CommandStep(
        (
            "{python}",
            "-I",
            "{trusted_replay}",
            "corpus",
            "private.json",
            "--processor",
            "candidate:run",
            "--isolation-wrapper",
            "C:/wrapper.exe",
            "--isolation-wrapper-sha256",
            "0" * 64,
        ),
        1.0,
        metrics_from_stdout=True,
        trusted_files_sha256=(("scripts/aigp_replay.py", "0" * 64),),
        trusted_host=True,
    )
    with pytest.raises(ValueError, match="trusted-host replay metrics"):
        TrialScheduler(
            ledger,
            GitWorktreePool(repo, tmp_path / "worktrees-t1-closure"),
            {
                Tier.T1_VQ2_REPLAY: TierCommand(
                    tier=Tier.T1_VQ2_REPLAY,
                    steps=(incomplete,),
                )
            },
        )


def test_non_live_scheduler_rejects_t5_command():
    with pytest.raises(ValueError, match="cannot contain a T5"):
        TierCommand(Tier.T5_AUTHORIZED_LIVE, ("python", "x.py"), 1.0)


@pytest.mark.parametrize("value", [1, "true"])
def test_scheduler_rejects_non_boolean_hard_gate_even_when_optional(value):
    assert "non-boolean" in TrialScheduler._hard_gate_failure(
        {"valid": value}, required=False
    )


def test_ledger_integrated_t0_round_promotes_every_candidate(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifiers = [_trial(ledger, repo, name=f"candidate-{index}") for index in range(4)]
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED, ("{python}", "-c", "pass"), 5.0
            ),
        },
        owner="round-test",
        lease_ttl_s=3.0,
    )
    first = scheduler.run_round(Tier.T0_AFFECTED, keep_fraction=0.5)
    assert len(first["promoted"]) == 4
    assert len(first["eliminated_by_halving"]) == 0
    assert all(0 in ledger.completed_tiers(identifier) for identifier in identifiers)
    assert sum(ledger.get_trial(identifier)["status"] == "pending" for identifier in identifiers) == 4


def test_command_plan_rejects_coerced_flags_and_unknown_keys(tmp_path):
    path = tmp_path / "commands.json"
    path.write_text(
        json.dumps(
            {
                "schema": "aigp-promotion-commands/1",
                "tiers": [
                    {
                        "tier": 0,
                        "argv": ["{python}", "-c", "pass"],
                        "timeout_s": 1,
                        "metrics_from_stdout": "false",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="exact bool"):
        load_tier_commands(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_tier_commands(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("argv", {"{python}": 1, "-c": 2, "pass": 3}, "exact argv arrays"),
        (
            "trusted_files_sha256",
            [["scripts/aigp_pytest.py", "0" * 64]],
            "exact object",
        ),
    ],
)
def test_command_plan_rejects_container_type_coercion(
    tmp_path, field, value, message
):
    path = tmp_path / "commands.json"
    tier = {
        "tier": 0,
        "argv": ["{python}", "-I", "scripts/aigp_pytest.py", "affected", "x.py"],
        "timeout_s": 1,
        "metrics_from_stdout": False,
        "require_hard_gates": False,
        "trusted_files_sha256": {
            "scripts/aigp_pytest.py": "0" * 64,
            "config/t1_pytest.ini": "1" * 64,
            "config/t1_pytest_policy.json": "2" * 64,
        },
    }
    tier[field] = value
    path.write_text(
        json.dumps({"schema": "aigp-promotion-commands/1", "tiers": [tier]}),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match=message):
        load_tier_commands(path)


def test_composite_tier_retains_each_structured_step_metric(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                steps=(
                    CommandStep(
                        ("{python}", "-c", 'print("{\\\"tests\\\":3}")'),
                        5.0,
                        metrics_from_stdout=True,
                        require_hard_gates=False,
                    ),
                    CommandStep(
                        ("{python}", "-c", 'print("{\\\"policy_passed\\\":true}")'),
                        5.0,
                        metrics_from_stdout=True,
                        require_hard_gates=False,
                    ),
                ),
            )
        },
        owner="composite-test",
        lease_ttl_s=3.0,
    )
    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifier
    assert ledger.get_checkpoint(identifier, 0)["metrics"] == {
        "step_0": {"tests": 3},
        "step_1": {"policy_passed": True},
    }


def test_nonfinite_metric_stdout_fails_terminally_instead_of_stranding_lease(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                ("{python}", "-c", 'print("{\\\"metric\\\":NaN}")'),
                5.0,
                metrics_from_stdout=True,
                require_hard_gates=False,
            )
        },
        owner="nan-test",
        lease_ttl_s=3.0,
    )
    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifier
    row = ledger.get_trial(identifier)
    assert row["status"] == "failed"
    assert row["lease_owner"] is None
    assert "exited 3" in row["failure_reason"]
    assert "non-standard JSON constant" in row["stdout_stderr_tail"]


def test_metrics_step_with_empty_stdout_fails_terminally(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                ("{python}", "-c", "pass"),
                5.0,
                metrics_from_stdout=True,
                require_hard_gates=False,
            )
        },
        owner="empty-metrics-test",
        lease_ttl_s=3.0,
    )
    scheduler.run_once(through=Tier.T0_AFFECTED)
    row = ledger.get_trial(identifier)
    assert row["status"] == "failed"
    assert "metrics stdout is empty" in row["stdout_stderr_tail"]


def test_single_merger_requires_complete_checkpoint_chain(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    assert ledger.lease_trial(identifier, "worker")
    ledger.checkpoint(identifier, 4, owner="worker", status="completed")
    ledger.finish_trial(identifier, "worker", success=True)
    with pytest.raises(ValueError, match="T0-T4 checkpoint chain"):
        SingleMerger(ledger, repo).merge_completed(identifier)


def test_t1_replay_scores_rank_by_reliability_then_centering_and_stability():
    def metrics(recall, center, stability):
        from aigp_loop.promotion import _REPLAY_PROMOTION_REQUIRED_BOUNDS

        constraints = {
            path: dict(bounds)
            for path, bounds in _REPLAY_PROMOTION_REQUIRED_BOUNDS.items()
        }
        observed = {
            path: bounds.get("min", bounds.get("max"))
            for path, bounds in constraints.items()
        }
        observed.update(
            {
                "perception.gate_recall": recall,
                "perception.center_error_px_p95": center,
                "perception.temporal_center_step_px_p95": stability,
            }
        )
        return {
            "step_1": {
                "schema": "aigp-vq2-replay-score/1",
                "processor": "candidate:run",
                "processor_code_sha256": "c" * 64,
                "candidate_isolation": {
                    "schema": "aigp-replay-isolation-attestation/1",
                    "network": "denied",
                    "filesystem": "readonly-worktree-only",
                    "non_interactive": True,
                    "process_tree_containment": "kill-on-wrapper-exit",
                    "host_process_access": "denied",
                    "wrapper_sha256": "f" * 64,
                },
                "domain_provenance": {
                    "perception": "candidate_detector_on_all_decoded_frames",
                    "estimator": "candidate_estimator_on_ordered_sanitized_stream",
                    "open_loop_commands": "candidate_generator_on_ordered_sanitized_stream",
                },
                "evaluation_evidence_hash": "e" * 64,
                "policy": {
                    "schema": "aigp-vq2-replay-policy-result/1",
                    "policy_hash": json_hash(
                        {
                            "schema": "aigp-vq2-replay-policy/1",
                            "metrics": constraints,
                        }
                    ),
                    "passed": True,
                    "constraints": constraints,
                    "observed": observed,
                    "violations": [],
                },
                "perception": {
                    "gate_recall": recall,
                    "center_error_px_p95": center,
                    "temporal_center_step_px_p95": stability,
                },
            }
        }

    reliable = TrialScheduler._promotion_evaluation(
        "reliable", Tier.T1_VQ2_REPLAY, metrics(1.0, 8.0, 10.0)
    )
    centered = TrialScheduler._promotion_evaluation(
        "centered", Tier.T1_VQ2_REPLAY, metrics(0.99, 1.0, 1.0)
    )
    from aigp_loop.promotion import PromotionLadder

    decision = PromotionLadder(keep_fraction=0.5).decide([centered, reliable])
    assert decision.promoted == ("reliable",)


def test_distinct_resolved_configs_are_hash_verified_and_delivered_to_commands(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifiers = [_trial(ledger, repo, name=name) for name in ("alpha", "beta")]
    program = (
        "import json,os; "
        "d=json.load(open(os.environ['AIGP_RESOLVED_CONFIG'], encoding='utf-8')); "
        "print(json.dumps({'name':d['name'],'config_hash':os.environ['AIGP_CONFIG_HASH']}))"
    )
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                ("{python}", "-c", program, "{config}"),
                5.0,
                metrics_from_stdout=True,
                require_hard_gates=False,
            )
        },
        owner="config-test",
        lease_ttl_s=3.0,
    )
    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifiers[0]
    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifiers[1]
    observed = {
        ledger.get_checkpoint(identifier, 0)["metrics"]["name"]:
        ledger.get_checkpoint(identifier, 0)["metrics"]["config_hash"]
        for identifier in identifiers
    }
    assert observed == {
        "alpha": ledger.get_trial(identifiers[0])["config_hash"],
        "beta": ledger.get_trial(identifiers[1])["config_hash"],
    }
    assert scheduler.run_once(through=Tier.T0_AFFECTED) is None


def test_scheduler_fails_closed_on_runtime_environment_drift(tmp_path):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    config = {"name": "drift"}
    commit, dirty, code = git_provenance(repo)
    identifier, _ = ledger.create_or_get_trial(
        key=TrialKey(code, json_hash(config), "dataset", 1, "eval-v1"),
        commit_hash=commit,
        dirty_diff_hash=dirty,
        resolved_config=config,
        environment_fingerprint="stale-environment",
    )
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                ("{python}", "-c", "raise SystemExit('must not execute')"),
                5.0,
            )
        },
        owner="environment-test",
        lease_ttl_s=3.0,
    )
    scheduler.run_once(through=Tier.T0_AFFECTED)
    row = ledger.get_trial(identifier)
    assert row["status"] == "failed"
    assert "environment fingerprint drifted" in row["failure_reason"]


def test_scheduler_rejects_command_plan_not_frozen_into_trial_key(tmp_path):
    repo = _repo(tmp_path / "repo-command-plan")
    ledger = TrialLedger(tmp_path / "command-plan.sqlite3")
    frozen = TierCommand(
        Tier.T0_AFFECTED, ("{python}", "-c", "pass"), 5.0
    )
    configured = TierCommand(
        Tier.T0_AFFECTED,
        ("{python}", "-c", "raise AssertionError('must never execute')"),
        5.0,
    )
    identities = []
    for tier in range(5):
        command_hash = (
            json_hash(scheduler_module.dataclasses.asdict(frozen))
            if tier == 0
            else json_hash({"unused-tier": tier})
        )
        identities.append(
            {
                "tier": tier,
                "dataset_hash": str(tier) * 64,
                "config_hash": chr(ord("a") + tier) * 64,
                "seed": 1,
                "repetitions": 1,
                "evaluator_version": f"tier-{tier}",
                "command_plan_sha256": command_hash,
            }
        )
    manifest = {
        "schema": "aigp-promotion-ladder-manifest/2",
        "tiers": identities,
    }
    config = {"promotion_ladder_manifest": manifest}
    manifest_hash = json_hash(manifest)
    commit, dirty, code_hash = git_provenance(repo)
    identifier, _created = ledger.create_or_get_trial(
        key=TrialKey(
            code_hash,
            json_hash(config),
            manifest_hash,
            1,
            f"aigp-ladder/2:{manifest_hash}",
        ),
        commit_hash=commit,
        dirty_diff_hash=dirty,
        resolved_config=config,
        environment_fingerprint=environment_fingerprint(),
    )
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees-command-plan"),
        {Tier.T0_AFFECTED: configured},
        owner="command-plan-test",
        lease_ttl_s=3.0,
    )
    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifier
    row = ledger.get_trial(identifier)
    assert row["status"] == "failed"
    assert "command plan differs from the frozen TrialKey" in row["failure_reason"]


def test_offline_child_environment_scrubs_inherited_credentials(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    identifier = _trial(ledger, repo)
    monkeypatch.setenv("AIGP_TEST_SECRET_TOKEN", "must-not-reach-child")
    program = (
        "import json,os; "
        "print(json.dumps({'secret_present':'AIGP_TEST_SECRET_TOKEN' in os.environ,"
        "'path_present':any(k.casefold()=='path' for k in os.environ)}))"
    )
    scheduler = TrialScheduler(
        ledger,
        GitWorktreePool(repo, tmp_path / "worktrees"),
        {
            Tier.T0_AFFECTED: TierCommand(
                Tier.T0_AFFECTED,
                ("{python}", "-c", program),
                5.0,
                metrics_from_stdout=True,
                require_hard_gates=False,
            )
        },
        owner="scrub-test",
        lease_ttl_s=3.0,
    )
    assert scheduler.run_once(through=Tier.T0_AFFECTED) == identifier
    metrics = ledger.get_checkpoint(identifier, 0)["metrics"]
    assert metrics == {"secret_present": False, "path_present": True}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda score: score.update(policy={"passed": False}),
        lambda score: score.pop("evaluation_evidence_hash"),
        lambda score: score.update(processor="recorded"),
        lambda score: score["candidate_isolation"].update(network="unproved"),
        lambda score: score["domain_provenance"].update(
            estimator="recorded_bundle_context"
        ),
    ],
)
def test_t1_replay_eligibility_fails_closed_without_policy_evidence_and_candidate_code(mutation):
    score = {
        "schema": "aigp-vq2-replay-score/1",
        "processor": "candidate:run",
        "processor_code_sha256": "c" * 64,
        "candidate_isolation": {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "denied",
            "filesystem": "readonly-worktree-only",
            "non_interactive": True,
            "process_tree_containment": "kill-on-wrapper-exit",
            "host_process_access": "denied",
            "wrapper_sha256": "f" * 64,
        },
        "domain_provenance": {
            "perception": "candidate_detector_on_all_decoded_frames",
            "estimator": "candidate_estimator_on_ordered_sanitized_stream",
            "open_loop_commands": "candidate_generator_on_ordered_sanitized_stream",
        },
        "evaluation_evidence_hash": "e" * 64,
        "policy": {"passed": True},
        "perception": {
            "gate_recall": 1.0,
            "center_error_px_p95": 1.0,
            "temporal_center_step_px_p95": 1.0,
        },
    }
    mutation(score)
    evaluation = TrialScheduler._promotion_evaluation(
        "candidate", Tier.T1_VQ2_REPLAY, score
    )
    assert evaluation.hard_gates is None
    assert evaluation.eligibility.passed is False
    assert PromotionLadder().decide([evaluation]).promoted == ()


def test_trusted_manifest_publish_is_atomic_and_never_overwrites_implicitly(
    tmp_path
):
    repo = _repo(tmp_path / "manifest-repo")
    output = tmp_path / "trusted.json"
    argv = [
        "--ledger",
        str(tmp_path / "manifest-ledger.sqlite3"),
        "build-trusted-manifest",
        "--repo",
        str(repo),
        "--out",
        str(output),
        "candidate.py",
    ]
    assert trials_main(argv) == 0
    first = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        trials_main(argv)
    assert output.read_bytes() == first
    (repo / "candidate.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert trials_main([*argv, "--overwrite"]) == 0
    assert output.read_bytes() != first
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_trusted_manifest_rejects_symlinked_evaluator_file(tmp_path):
    repo = _repo(tmp_path / "symlink-repo")
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 9\n", encoding="utf-8")
    linked = repo / "linked.py"
    try:
        linked.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this host: {exc}")
    with pytest.raises(ValueError, match="symlink"):
        trials_main(
            [
                "--ledger",
                str(tmp_path / "symlink-ledger.sqlite3"),
                "build-trusted-manifest",
                "--repo",
                str(repo),
                "--out",
                str(tmp_path / "must-not-exist.json"),
                "linked.py",
            ]
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_trusted_manifest_rejects_git_tracked_bytecode_anywhere(tmp_path):
    repo = _repo(tmp_path / "bytecode-repo")
    cache = repo / "unrelated" / "__pycache__"
    cache.mkdir(parents=True)
    payload = cache / "shadow.cpython-312.pyc"
    payload.write_bytes(b"tracked executable bytecode")
    subprocess.run(["git", "add", "-f", str(payload)], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "tracked bytecode"], cwd=repo, check=True
    )
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="tracked executable bytecode"):
        trials_main(
            [
                "--ledger",
                str(tmp_path / "bytecode-ledger.sqlite3"),
                "build-trusted-manifest",
                "--repo",
                str(repo),
                "--out",
                str(output),
                "candidate.py",
            ]
        )
    assert not output.exists()


def test_checked_in_example_ladder_is_structurally_runnable_and_preparable(
    tmp_path
):
    root = Path(__file__).resolve().parents[1]
    commands = load_tier_commands(root / "config" / "promotion_commands.example.json")
    assert set(commands) == set(Tier) - {Tier.T5_AUTHORIZED_LIVE}
    t1_steps = commands[Tier.T1_VQ2_REPLAY].steps
    assert t1_steps[0].isolation_wrapper is not None
    assert t1_steps[0].isolation_wrapper_sha256 == "0" * 64
    assert t1_steps[1].trusted_host is True
    assert t1_steps[1].argv[:4] == (
        "{python}",
        "-I",
        "{trusted_replay}",
        "corpus",
    )
    assert "--isolation-wrapper" in t1_steps[1].argv
    assert "--isolation-wrapper-sha256" in t1_steps[1].argv
    for tier in (Tier.T2_WARM_SIM, Tier.T3_DOMAIN_TRACKS, Tier.T4_FULL_NON_LIVE):
        assert all(step.trusted_files_sha256 for step in commands[tier].steps)

    output = tmp_path / "candidate.json"
    argv = [
        "--ledger",
        str(tmp_path / "prepare-ledger.sqlite3"),
        "prepare-ladder-config",
        "--base-config",
        str(root / "config" / "promotion_candidate_base.example.json"),
        "--tier-identities",
        str(root / "config" / "promotion_ladder_identities.example.json"),
        "--commands",
        str(root / "config" / "promotion_commands.example.json"),
        "--out",
        str(output),
    ]
    assert trials_main(argv) == 0
    prepared = json.loads(output.read_text(encoding="utf-8"))
    assert prepared["promotion_ladder_manifest"]["schema"] == (
        "aigp-promotion-ladder-manifest/2"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        trials_main(argv)
    assert trials_main([*argv, "--overwrite"]) == 0


def test_command_loader_accepts_only_the_canonical_trust_manifest():
    root = Path(__file__).resolve().parents[1]
    command_document = root / "config" / "promotion_commands.example.json"
    with pytest.raises(ValueError, match="canonical repository trust manifest"):
        scheduler_module._trusted_manifest_files(
            command_document, "config/alternate_trusted_files.json"
        )


def test_t1_and_nonlive_commands_bind_the_canonical_manifest_argv():
    hashes = lambda names: tuple(sorted((name, "0" * 64) for name in names))
    pytest_step = CommandStep(
        argv=("{python}", "-I", "scripts/aigp_pytest.py", "vq2"),
        timeout_s=30,
        require_hard_gates=False,
        trusted_files_sha256=hashes(scheduler_module._TRUSTED_T1_PYTEST_FILES),
        isolation_wrapper="reviewed-wrapper",
        isolation_wrapper_sha256="0" * 64,
    )
    replay_step = CommandStep(
        argv=(
            "{python}",
            "-I",
            "{trusted_replay}",
            "corpus",
            "private-corpus.json",
            "--processor",
            "candidate:run",
            "--isolation-wrapper",
            "reviewed-wrapper",
            "--isolation-wrapper-sha256",
            "0" * 64,
            "--trusted-manifest",
            scheduler_module._TRUSTED_MANIFEST_PATH,
        ),
        timeout_s=30,
        metrics_from_stdout=True,
        require_hard_gates=False,
        trusted_files_sha256=hashes(
            scheduler_module._TRUSTED_REPLAY_HOST_FILES
            | {scheduler_module._TRUSTED_MANIFEST_PATH}
        ),
        trusted_host=True,
    )
    t1 = TierCommand(
        tier=Tier.T1_VQ2_REPLAY, steps=(pytest_step, replay_step)
    )
    scheduler_module._validate_t1_command(t1)
    replay_argv = list(replay_step.argv)
    replay_argv[replay_argv.index("--trusted-manifest") + 1] = (
        "config/alternate_trusted_files.json"
    )
    bad_replay = dataclasses.replace(replay_step, argv=tuple(replay_argv))
    with pytest.raises(ValueError, match="trusted-host replay metrics"):
        scheduler_module._validate_t1_command(
            dataclasses.replace(t1, steps=(pytest_step, bad_replay))
        )
    unbound_replay = dataclasses.replace(
        replay_step,
        trusted_files_sha256=hashes(
            scheduler_module._TRUSTED_REPLAY_HOST_FILES
        ),
    )
    with pytest.raises(ValueError, match="trusted-host replay metrics"):
        scheduler_module._validate_t1_command(
            dataclasses.replace(t1, steps=(pytest_step, unbound_replay))
        )

    nonlive_step = CommandStep(
        argv=(
            "{python}",
            "-I",
            "scripts/aigp_nonlive.py",
            "--tier",
            "2",
            "--configs",
            "race_01",
            "--trusted-manifest",
            scheduler_module._TRUSTED_MANIFEST_PATH,
        ),
        timeout_s=30,
        metrics_from_stdout=True,
        require_hard_gates=True,
        trusted_files_sha256=hashes(
            scheduler_module._TRUSTED_NONLIVE_FILES
            | {scheduler_module._TRUSTED_MANIFEST_PATH}
        ),
    )
    nonlive = TierCommand(tier=Tier.T2_WARM_SIM, steps=(nonlive_step,))
    scheduler_module._validate_nonlive_command(nonlive)
    nonlive_argv = list(nonlive_step.argv)
    nonlive_argv[nonlive_argv.index("--trusted-manifest") + 1] = (
        "config/alternate_trusted_files.json"
    )
    bad_nonlive = dataclasses.replace(nonlive_step, argv=tuple(nonlive_argv))
    with pytest.raises(ValueError, match="isolated non-live script bootstrap"):
        scheduler_module._validate_nonlive_command(
            dataclasses.replace(nonlive, steps=(bad_nonlive,))
        )
    unbound_nonlive = dataclasses.replace(
        nonlive_step,
        trusted_files_sha256=hashes(scheduler_module._TRUSTED_NONLIVE_FILES),
    )
    with pytest.raises(ValueError, match="isolated non-live script bootstrap"):
        scheduler_module._validate_nonlive_command(
            dataclasses.replace(nonlive, steps=(unbound_nonlive,))
        )


def test_trusted_replay_host_ignores_candidate_startup_and_import_hooks(tmp_path):
    root = Path(__file__).resolve().parents[1]
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    marker = tmp_path / "candidate-hook-ran"
    hook = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
    )
    (candidate / "sitecustomize.py").write_text(hook, encoding="utf-8")
    (candidate / "aigp_loop").mkdir()
    (candidate / "aigp_loop" / "__init__.py").write_text(
        hook, encoding="utf-8"
    )
    (candidate / "json.py").write_text(hook, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(candidate)

    run = subprocess.run(
        [
            sys.executable,
            "-I",
            str(root / "scripts" / "aigp_replay.py"),
            "--help",
        ],
        cwd=candidate,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15.0,
        shell=False,
    )

    assert run.returncode == 0, run.stderr
    assert "Verify and score private VQ2 replay bundles" in run.stdout
    assert not marker.exists()


def test_isolation_wrapper_configuration_must_pair_path_and_hash():
    with pytest.raises(ValueError, match="must be paired"):
        CommandStep(
            ("{python}", "-c", "pass"),
            1.0,
            isolation_wrapper="C:/reviewed/wrapper.exe",
        )
