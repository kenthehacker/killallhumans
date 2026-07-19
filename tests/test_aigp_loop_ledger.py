from __future__ import annotations

import json
import sqlite3
import time

import pytest

from aigp_loop._util import json_hash
from aigp_loop.ledger import TrialKey, TrialLedger


def _create_trial(ledger: TrialLedger, *, suffix: str = "a") -> tuple[str, TrialKey]:
    config = {"candidate": suffix, "gain": 1.0}
    key = TrialKey(
        code_hash=f"code-{suffix}",
        config_hash=json_hash(config),
        dataset_hash="dataset-1",
        seed=7,
        evaluator_version="vq2-replay-v1",
    )
    identifier, created = ledger.create_or_get_trial(
        key=key,
        commit_hash=f"commit-{suffix}",
        dirty_diff_hash="clean-diff-hash",
        resolved_config=config,
        environment_fingerprint="env-1",
        artifact_hashes={"replay": "abc"},
        simulator_build="3385",
    )
    assert created
    return identifier, key


def test_ledger_schema_has_required_provenance_fields(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    with sqlite3.connect(ledger.path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(trials)")}
    assert {
        "trial_id",
        "parent_trial_id",
        "status",
        "lease_owner",
        "heartbeat",
        "started_at",
        "finished_at",
        "commit_hash",
        "dirty_diff_hash",
        "resolved_config",
        "config_hash",
        "dataset_hash",
        "artifact_hashes",
        "simulator_build",
        "evaluator_version",
        "environment_fingerprint",
        "seed",
        "phase_timings",
        "safety_and_completion_metrics",
        "failure_reason",
        "stdout_stderr_tail",
    } <= columns


def test_trial_deduplication_uses_exact_five_part_key(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    identifier, key = _create_trial(ledger)
    config = {"candidate": "a", "gain": 1.0}
    duplicate, created = ledger.create_or_get_trial(
        key=key,
        commit_hash="commit-a",
        dirty_diff_hash="clean-diff-hash",
        resolved_config=config,
        environment_fingerprint="env-1",
        simulator_build="3385",
    )
    assert not created
    assert duplicate == identifier


@pytest.mark.parametrize(
    "override,conflict",
    [
        ({"commit_hash": "other-commit"}, "commit_hash"),
        ({"dirty_diff_hash": "other-diff"}, "dirty_diff_hash"),
        ({"environment_fingerprint": "other-env"}, "environment_fingerprint"),
        ({"parent_trial_id": "other-parent"}, "parent_trial_id"),
        ({"simulator_build": "other-build"}, "simulator_build"),
        ({"candidate_name": "other-name"}, "candidate_name"),
        ({"trial_id": "other-id"}, "trial_id"),
    ],
)
def test_trial_deduplication_rejects_contradictory_metadata(
    tmp_path, override, conflict
):
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    identifier, key = _create_trial(ledger)
    arguments = {
        "key": key,
        "commit_hash": "commit-a",
        "dirty_diff_hash": "clean-diff-hash",
        "resolved_config": {"candidate": "a", "gain": 1.0},
        "environment_fingerprint": "env-1",
        "simulator_build": "3385",
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=conflict):
        ledger.create_or_get_trial(**arguments)

    assert ledger.list_trials() == [ledger.get_trial(identifier)]


def test_trial_deduplication_rechecks_exact_resolved_config(tmp_path, monkeypatch):
    import aigp_loop.ledger as ledger_module

    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    _identifier, key = _create_trial(ledger)
    real_hash = ledger_module.json_hash
    monkeypatch.setattr(
        ledger_module,
        "json_hash",
        lambda value: (
            key.config_hash
            if value == {"candidate": "forged", "gain": 2.0}
            else real_hash(value)
        ),
    )

    with pytest.raises(ValueError, match="resolved_config"):
        ledger.create_or_get_trial(
            key=key,
            commit_hash="commit-a",
            dirty_diff_hash="clean-diff-hash",
            resolved_config={"candidate": "forged", "gain": 2.0},
            environment_fingerprint="env-1",
            simulator_build="3385",
        )


@pytest.mark.parametrize("seed", [True, "7", 7.0, 7.5])
def test_trial_key_rejects_non_exact_integer_seed(seed):
    with pytest.raises(ValueError, match="exact integer"):
        TrialKey("code", "config", "data", seed, "eval").validate()


@pytest.mark.parametrize("field", ["code_hash", "config_hash", "dataset_hash", "evaluator_version"])
def test_trial_key_rejects_coerced_string_provenance(field):
    values = {
        "code_hash": "code",
        "config_hash": "config",
        "dataset_hash": "data",
        "seed": 1,
        "evaluator_version": "eval",
    }
    values[field] = 1
    with pytest.raises(ValueError, match="exact non-empty string"):
        TrialKey(**values).validate()


def test_lease_expiry_heartbeat_and_checkpoint_ownership(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    identifier, _key = _create_trial(ledger)
    assert ledger.lease_trial(identifier, "worker-a", ttl_s=30.0)
    assert not ledger.lease_trial(identifier, "worker-b", ttl_s=30.0)
    with pytest.raises(RuntimeError, match="lease"):
        ledger.checkpoint(identifier, 0, owner="worker-b", status="completed")
    ledger.heartbeat(identifier, "worker-a", ttl_s=30.0)
    ledger.checkpoint(identifier, 0, owner="worker-a", status="completed")
    assert ledger.completed_tiers(identifier) == (0,)
    assert ledger.next_tier(identifier, through=2) == 1

    # Explicit virtual epochs exercise safe lease reclamation deterministically.
    other, _ = _create_trial(ledger, suffix="b")
    assert ledger.lease_trial(other, "old", ttl_s=5.0, now_s=100.0)
    assert ledger.lease_trial(other, "new", ttl_s=5.0, now_s=106.0)


@pytest.mark.parametrize("terminal_status", ["completed", "failed"])
def test_terminal_checkpoint_is_idempotent_but_immutable(tmp_path, terminal_status):
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    identifier, _ = _create_trial(ledger)
    assert ledger.lease_trial(identifier, "worker", ttl_s=30.0)
    kwargs = {
        "owner": "worker",
        "status": terminal_status,
        "metrics": {"score": 1},
        "artifact_hashes": {"proof": "a" * 64},
        "stdout_stderr_tail": "done",
        "elapsed_s": 1.25,
    }
    ledger.checkpoint(identifier, 0, **kwargs)
    ledger.checkpoint(identifier, 0, **kwargs)
    with pytest.raises(RuntimeError, match="immutable"):
        ledger.checkpoint(identifier, 0, **{**kwargs, "metrics": {"score": 2}})
    with pytest.raises(RuntimeError, match="immutable"):
        ledger.checkpoint(
            identifier,
            0,
            **{**kwargs, "status": "failed" if terminal_status == "completed" else "completed"},
        )


def test_promotion_round_rejects_duplicate_members_and_immutable_redecision(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    with pytest.raises(ValueError, match="unique"):
        ledger.create_or_get_promotion_round(0, ["trial-a", "trial-a"])
    promotion_round = ledger.create_or_get_promotion_round(0, ["trial-b", "trial-a"])
    decision = {"survivors": ["trial-a"], "eliminated": ["trial-b"]}
    ledger.decide_promotion_round(promotion_round["round_id"], decision)
    ledger.decide_promotion_round(promotion_round["round_id"], decision)
    with pytest.raises(RuntimeError, match="immutable"):
        ledger.decide_promotion_round(
            promotion_round["round_id"],
            {"survivors": ["trial-b"], "eliminated": ["trial-a"]},
        )


def test_global_singleton_lease_prevents_scheduler_or_merger_overlap(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    assert ledger.acquire_global_lease("scheduler", "one", ttl_s=10, now_s=1)
    assert not ledger.acquire_global_lease("scheduler", "two", ttl_s=10, now_s=2)
    assert ledger.acquire_global_lease("scheduler", "two", ttl_s=10, now_s=12)
    assert not ledger.release_global_lease("scheduler", "one")
    assert ledger.release_global_lease("scheduler", "two")


def test_expired_worker_cannot_heartbeat_or_publish_terminal_result(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    identifier, _ = _create_trial(ledger)
    past = time.time() - 10.0
    assert ledger.lease_trial(identifier, "stalled", ttl_s=1.0, now_s=past)
    with pytest.raises(RuntimeError, match="lease"):
        ledger.heartbeat(identifier, "stalled", ttl_s=30.0)
    with pytest.raises(RuntimeError, match="lease"):
        ledger.finish_trial(identifier, "stalled", success=True)
    assert ledger.lease_trial(identifier, "reclaimer", ttl_s=30.0)
    ledger.finish_trial(identifier, "reclaimer", success=True)
    assert ledger.get_trial(identifier)["status"] == "completed"


def test_terminal_success_and_lease_durations_reject_bool_coercion(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    identifier, _ = _create_trial(ledger)
    with pytest.raises(ValueError, match="ttl_s"):
        ledger.lease_trial(identifier, "worker", ttl_s=True)
    assert ledger.lease_trial(identifier, "worker")
    with pytest.raises(TypeError, match="exact bool"):
        ledger.finish_trial(identifier, "worker", success=1)
    ledger.finish_trial(identifier, "worker", success=False)


def test_unsupported_existing_schema_fails_clearly(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        db.execute("INSERT INTO metadata VALUES('schema_version','999')")
    with pytest.raises(RuntimeError, match="unsupported trial ledger schema 999"):
        TrialLedger(path)


def test_legacy_pretty_json_stream_import_is_one_time_and_untrusted(tmp_path):
    source = tmp_path / "benchmark_history.jsonl"
    objects = [
        {"overall_passed": True, "simulation": {"skipped": True}},
        {"overall_passed": False, "simulation": {"skipped": True}},
    ]
    source.write_text("\n".join(json.dumps(item, indent=2) for item in objects), encoding="utf-8")
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    assert ledger.import_legacy_benchmark_history(source) == 2
    assert ledger.import_legacy_benchmark_history(source) == 2
    rows = ledger.list_trials()
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "completed"
        assert row["resolved_config"]["promotion_eligible"] is False
        evidence = row["safety_and_completion_metrics"]
        assert evidence["trusted"] is False
        assert evidence["comparable_to_current_evaluator"] is False
        assert evidence["promotion_eligible"] is False


def test_legacy_import_resumes_existing_nonterminal_row_before_marking(tmp_path, monkeypatch):
    source = tmp_path / "history.jsonl"
    source.write_text(json.dumps({"overall_passed": True}, indent=2), encoding="utf-8")
    ledger = TrialLedger(tmp_path / "trials.sqlite3")
    real_finish = ledger.finish_trial
    calls = {"count": 0}

    def interrupt_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated interruption")
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(ledger, "finish_trial", interrupt_once)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        ledger.import_legacy_benchmark_history(source)
    assert ledger.list_trials()[0]["status"] == "running"
    assert ledger.import_legacy_benchmark_history(source) == 1
    assert ledger.list_trials()[0]["status"] == "completed"
