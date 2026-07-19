from __future__ import annotations

import inspect
import json
import sqlite3
import time

import pytest

from aigp_loop._util import canonical_json, json_hash
from aigp_loop.campaign import (
    CampaignCandidate,
    PoweredTrialResult,
    PreflightHealth,
    WarmCampaign,
)
from aigp_loop.ledger import TrialKey, TrialLedger
from aigp_loop.nonlive import CORE_EVALUATOR_FILES
from aigp_loop.promotion import (
    _REPLAY_PROMOTION_REQUIRED_BOUNDS,
    validate_promotion_chain,
)
from scripts.aigp_campaign import main as campaign_plan_main


def _source(ledger: TrialLedger, name: str) -> str:
    code_hash = json_hash({"candidate": name})
    command_plans = [
        json_hash({"frozen_test_command": tier}) for tier in range(5)
    ]
    identities = [
        {
            "tier": tier,
            "dataset_hash": str(tier) * 64,
            "config_hash": chr(ord("a") + tier) * 64,
            "seed": 1,
            "repetitions": 1,
            "evaluator_version": f"tier-{tier}-evaluator-v1",
            "command_plan_sha256": command_plans[tier],
        }
        for tier in range(5)
    ]
    manifest = {
        "schema": "aigp-promotion-ladder-manifest/2",
        "tiers": identities,
    }
    manifest_hash = json_hash(manifest)
    config = {"candidate": name, "promotion_ladder_manifest": manifest}
    identifier, _ = ledger.create_or_get_trial(
        key=TrialKey(
            code_hash,
            json_hash(config),
            manifest_hash,
            1,
            f"aigp-ladder/2:{manifest_hash}",
        ),
        commit_hash=f"commit-{name}",
        dirty_diff_hash="clean",
        resolved_config=config,
        environment_fingerprint="env",
    )
    assert ledger.lease_trial(identifier, "offline")

    def artifacts(tier, metrics, **extra):
        manifest_identity = json_hash(identities[tier])
        command_plan = command_plans[tier]
        return {
            "metrics_sha256": json_hash(metrics),
            "manifest_tier_identity_sha256": manifest_identity,
            "command_plan_sha256": command_plan,
            "tier_identity_sha256": json_hash(
                {
                    "manifest_tier_identity_sha256": manifest_identity,
                    "command_plan_sha256": command_plan,
                }
            ),
            **extra,
        }

    t0_metrics = {"tests": "passed"}
    ledger.checkpoint(
        identifier,
        0,
        owner="offline",
        status="completed",
        metrics=t0_metrics,
        artifact_hashes=artifacts(0, t0_metrics),
    )
    constraints = {
        path: dict(bounds)
        for path, bounds in _REPLAY_PROMOTION_REQUIRED_BOUNDS.items()
    }
    observed = {
        path: (
            bounds["min"]
            if "min" in bounds
            else bounds["max"]
        )
        for path, bounds in constraints.items()
    }
    policy = {
        "schema": "aigp-vq2-replay-policy-result/1",
        "policy_hash": json_hash(
            {"schema": "aigp-vq2-replay-policy/1", "metrics": constraints}
        ),
        "passed": True,
        "constraints": constraints,
        "observed": observed,
        "violations": [],
    }
    t1_metrics = {
        "schema": "aigp-vq2-replay-score/1",
        "processor": "candidate:detect",
        "processor_code_sha256": code_hash,
        "candidate_isolation": {
            "schema": "aigp-replay-isolation-attestation/1",
            "network": "denied",
            "filesystem": "readonly-worktree-only",
            "non_interactive": True,
            "process_tree_containment": "kill-on-wrapper-exit",
            "host_process_access": "denied",
            "wrapper_sha256": "f" * 64,
        },
        "evaluation_input_hash": identities[1]["dataset_hash"],
        "evaluation_config_sha256": identities[1]["config_hash"],
        "seed": identities[1]["seed"],
        "repetitions": identities[1]["repetitions"],
        "evaluator_version": identities[1]["evaluator_version"],
        "policy": policy,
        "domain_provenance": {
            "perception": "candidate_detector_on_all_decoded_frames",
            "estimator": "candidate_estimator_on_ordered_sanitized_stream",
            "open_loop_commands": "candidate_generator_on_ordered_sanitized_stream",
        },
    }
    ledger.checkpoint(
        identifier,
        1,
        owner="offline",
        status="completed",
        metrics=t1_metrics,
        artifact_hashes=artifacts(1, t1_metrics),
    )
    hard_gates = {
        "valid": True,
        "completed": True,
        "correct_gate_sequence": True,
        "cleanup_confirmed": True,
        "no_collision": True,
        "no_disqualification": True,
        "no_stale_stream_flight": True,
    }
    for tier in (2, 3, 4):
        trusted_sources = {
            name: "f" * 64 for name in CORE_EVALUATOR_FILES
        }
        metrics = {
            "schema": "aigp-nonlive-promotion-evidence/1",
            "tier": tier,
            "track_identity": {
                2: ["race_01"],
                3: ["grand_tour", "slalom", "vertical_cliff"],
                4: [
                    "aigp_default",
                    "figure8",
                    "grand_tour",
                    "race_01",
                    "slalom",
                    "straight_hairpin",
                    "vertical_cliff",
                ],
            }[tier],
            "domain_provenance": {"powered_resources_used": False},
            "evaluator_identity": {"source_sha256": trusted_sources},
            "evaluation_input_hash": identities[tier]["dataset_hash"],
            "evaluation_config_sha256": identities[tier]["config_hash"],
            "seed": identities[tier]["seed"],
            "repetitions": identities[tier]["repetitions"],
            "evaluator_version": identities[tier]["evaluator_version"],
            "promotion": {"hard_gates": hard_gates},
        }
        ledger.checkpoint(
            identifier,
            tier,
            owner="offline",
            status="completed",
            metrics=metrics,
            artifact_hashes=artifacts(
                tier,
                metrics,
                trusted_evaluator_files_sha256=json_hash(trusted_sources),
            ),
        )
    ledger.finish_trial(identifier, "offline", success=True)
    return identifier


def _success_result():
    return PoweredTrialResult(
        success=True,
        reset_epoch_proved=True,
        countdown_go_observed=True,
        watchdogs_armed=True,
        cleanup_confirmed=True,
        fresh_authoritative_state=True,
        no_stale_stream_flight=True,
        valid=True,
        correct_gate_sequence=True,
        completed=True,
        race_time_s=5.0,
        centering_margin=1.0,
        stability_margin=1.0,
    )


class _Backend:
    simulator_build = "3385"
    offline_during_timed_run = True
    non_interactive_during_timed_run = True
    maximum_powered_trial_s = 1.0
    candidate_code_mode = "external-process-per-candidate"
    powered_watchdog_declaration = {
        "schema": "aigp-powered-watchdog-declaration/1",
        "mechanism": "test-owned hard-stop supervisor",
        "maximum_powered_trial_s": 1.0,
        "hard_stop_before_return": True,
        "implementation_sha256": "d" * 64,
    }

    def __init__(self, outcomes=None, delay=0.0):
        self.outcomes = list(outcomes or [])
        self.delay = delay
        self.calls = []

    def passive_preflight(self):
        return PreflightHealth(True, "3385", 123.0, {"streams": "healthy"})

    def prepare_candidate(self, candidate, provenance):
        return dict(provenance)

    def run_powered_trial(self, candidate):
        self.calls.append(candidate)
        if self.delay:
            time.sleep(self.delay)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return _success_result()

    def compare_baseline(self, reference, current):
        return {
            "passed": abs(reference.race_time_s - current.race_time_s) <= 0.1,
            "race_time_delta_s": current.race_time_s - reference.race_time_s,
        }


def _campaign(tmp_path, *, backend=None):
    ledger = TrialLedger(tmp_path / "ledger.sqlite3")
    baseline = _source(ledger, "baseline")
    candidate = _source(ledger, "candidate")
    baseline_row = ledger.get_trial(baseline)
    candidate_row = ledger.get_trial(candidate)
    campaign = WarmCampaign(
        ledger,
        backend or _Backend(),
        [
            CampaignCandidate(
                baseline,
                "baseline",
                "gate0",
                baseline_row["code_hash"],
                baseline_row["config_hash"],
                baseline_row["dataset_hash"],
                baseline_row["evaluator_version"],
                True,
            ),
            CampaignCandidate(
                candidate,
                "candidate",
                "gate0",
                candidate_row["code_hash"],
                candidate_row["config_hash"],
                candidate_row["dataset_hash"],
                candidate_row["evaluator_version"],
                False,
            ),
        ],
        baseline_every=1,
    )
    return ledger, campaign, baseline, candidate


def test_campaign_requires_exact_plan_bound_authorization(tmp_path):
    _ledger, campaign, _baseline, _candidate = _campaign(tmp_path)
    with pytest.raises(PermissionError, match="exact build and plan"):
        campaign.run(authorization="AUTHORIZE_SOMETHING_ELSE")


def test_campaign_rejects_mixed_powered_stages_for_baseline_drift(tmp_path):
    ledger = TrialLedger(tmp_path / "mixed-stage.sqlite3")
    baseline = _source(ledger, "mixed-baseline")
    candidate = _source(ledger, "mixed-candidate")
    rows = [ledger.get_trial(baseline), ledger.get_trial(candidate)]
    candidates = [
        CampaignCandidate(
            rows[0]["trial_id"], "baseline", "hover",
            rows[0]["code_hash"], rows[0]["config_hash"],
            rows[0]["dataset_hash"], rows[0]["evaluator_version"], True,
        ),
        CampaignCandidate(
            rows[1]["trial_id"], "candidate", "gate0",
            rows[1]["code_hash"], rows[1]["config_hash"],
            rows[1]["dataset_hash"], rows[1]["evaluator_version"], False,
        ),
    ]
    with pytest.raises(ValueError, match="one shared powered campaign stage"):
        WarmCampaign(ledger, _Backend(), candidates, baseline_every=1)


def test_campaign_candidate_requires_exact_sha256_provenance():
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        CampaignCandidate(
            "trial", "candidate", "gate0", "not-a-hash",
            "a" * 64, "b" * 64, "evaluator", False,
        )


def test_promotion_chain_recomputes_manifest_identity_and_exact_processor_hash(tmp_path):
    first = TrialLedger(tmp_path / "identity.sqlite3")
    source = _source(first, "identity")
    checkpoint = first.get_checkpoint(source, 0)
    artifacts = dict(checkpoint["artifact_hashes"])
    artifacts["manifest_tier_identity_sha256"] = "0" * 64
    with sqlite3.connect(first.path) as db:
        db.execute(
            "UPDATE checkpoints SET artifact_hashes=? WHERE trial_id=? AND tier=0",
            (canonical_json(artifacts), source),
        )
    with pytest.raises(ValueError, match="manifest tier identity is stale"):
        validate_promotion_chain(first, source)

    second = TrialLedger(tmp_path / "processor.sqlite3")
    source = _source(second, "processor")
    checkpoint = second.get_checkpoint(source, 1)
    metrics = dict(checkpoint["metrics"])
    metrics["processor_code_sha256"] = "short"
    artifacts = dict(checkpoint["artifact_hashes"])
    artifacts["metrics_sha256"] = json_hash(metrics)
    with sqlite3.connect(second.path) as db:
        db.execute(
            """UPDATE checkpoints SET metrics=?,artifact_hashes=?
               WHERE trial_id=? AND tier=1""",
            (canonical_json(metrics), canonical_json(artifacts), source),
        )
    with pytest.raises(ValueError, match="passing T1 replay"):
        validate_promotion_chain(second, source)

    third = TrialLedger(tmp_path / "command-plan.sqlite3")
    source = _source(third, "command-plan")
    checkpoint = third.get_checkpoint(source, 0)
    artifacts = dict(checkpoint["artifact_hashes"])
    artifacts["command_plan_sha256"] = "f" * 64
    artifacts["tier_identity_sha256"] = json_hash(
        {
            "manifest_tier_identity_sha256": artifacts[
                "manifest_tier_identity_sha256"
            ],
            "command_plan_sha256": artifacts["command_plan_sha256"],
        }
    )
    with sqlite3.connect(third.path) as db:
        db.execute(
            "UPDATE checkpoints SET artifact_hashes=? WHERE trial_id=? AND tier=0",
            (canonical_json(artifacts), source),
        )
    with pytest.raises(ValueError, match="differs from frozen TrialKey"):
        validate_promotion_chain(third, source)


def test_campaign_never_materializes_t5_without_shipped_watchdog_supervisor(tmp_path):
    ledger, campaign, baseline, candidate = _campaign(tmp_path)
    with pytest.raises(RuntimeError, match="execution is unavailable"):
        campaign.run(authorization=campaign.authorization_phrase)
    children = [row for row in ledger.list_trials() if row["parent_trial_id"]]
    assert children == []


def test_plan_only_campaign_contains_no_dormant_powered_executor_path():
    assert not hasattr(WarmCampaign, "_materialize_execution")
    assert not hasattr(WarmCampaign, "_execution_order")
    source = inspect.getsource(WarmCampaign.run)
    for forbidden in (
        "run_powered_trial",
        "prepare_candidate",
        "passive_preflight",
        "acquire_global_lease",
        "lease_trial",
        "checkpoint",
    ):
        assert forbidden not in source


def test_unavailable_executor_never_calls_even_a_failing_backend(tmp_path):
    backend = _Backend([_success_result(), RuntimeError("backend exploded")])
    ledger, campaign, _baseline, candidate = _campaign(tmp_path, backend=backend)
    with pytest.raises(RuntimeError, match="execution is unavailable"):
        campaign.run(authorization=campaign.authorization_phrase)
    assert backend.calls == []
    assert [row for row in ledger.list_trials() if row["parent_trial_id"]] == []


def test_watchdog_declaration_does_not_enable_powered_execution(tmp_path):
    backend = _Backend(delay=0.04)
    backend.maximum_powered_trial_s = 0.01
    backend.powered_watchdog_declaration = {
        **backend.powered_watchdog_declaration,
        "maximum_powered_trial_s": 0.01,
    }
    ledger, campaign, baseline, _candidate = _campaign(tmp_path, backend=backend)
    with pytest.raises(RuntimeError, match="execution is unavailable"):
        campaign.run(authorization=campaign.authorization_phrase)
    assert backend.calls == []
    assert [row for row in ledger.list_trials() if row["parent_trial_id"]] == []


def test_unavailable_executor_acquires_no_live_lease(tmp_path, monkeypatch):
    backend = _Backend(delay=0.05)
    ledger, campaign, baseline, _candidate = _campaign(tmp_path, backend=backend)
    original = ledger.acquire_global_lease
    calls = {"live": 0}

    def lose_on_renew(name, owner, **kwargs):
        if name == "live-campaign":
            calls["live"] += 1
            if calls["live"] >= 2:
                return False
        return original(name, owner, **kwargs)

    monkeypatch.setattr(ledger, "acquire_global_lease", lose_on_renew)
    with pytest.raises(RuntimeError, match="execution is unavailable"):
        campaign.run(authorization=campaign.authorization_phrase)
    assert len(backend.calls) == 0
    assert calls["live"] == 0
    assert [row for row in ledger.list_trials() if row["parent_trial_id"]] == []


def test_stale_stream_or_missing_authoritative_state_is_a_hard_gate():
    result = PoweredTrialResult(
        success=True,
        reset_epoch_proved=True,
        countdown_go_observed=True,
        watchdogs_armed=True,
        cleanup_confirmed=True,
        fresh_authoritative_state=False,
        no_stale_stream_flight=False,
        valid=False,
    )
    assert not result.safety_passed


def test_authorization_hash_changes_with_frozen_candidate_provenance(tmp_path):
    ledger, campaign, baseline, candidate = _campaign(tmp_path)
    original = campaign.authorization_phrase
    rows = [ledger.get_trial(baseline), ledger.get_trial(candidate)]
    changed = CampaignCandidate(
        rows[0]["trial_id"],
        "baseline",
        "gate0",
        "0" * 64,
        rows[0]["config_hash"],
        rows[0]["dataset_hash"],
        rows[0]["evaluator_version"],
        True,
    )
    from aigp_loop.campaign import (
        campaign_plan_hash,
        freeze_backend_contract,
        required_authorization_phrase,
    )

    altered_hash = campaign_plan_hash(
        "3385",
        [changed],
        baseline_every=1,
        backend_contract=freeze_backend_contract(_Backend()),
    )
    assert required_authorization_phrase("3385", altered_hash) != original


def test_unavailable_executor_never_reaches_preflight(tmp_path):
    class Backend(_Backend):
        def passive_preflight(self):
            raise RuntimeError("preflight exploded")

    backend = Backend()
    ledger, campaign, baseline, _candidate = _campaign(tmp_path, backend=backend)
    with pytest.raises(RuntimeError, match="execution is unavailable"):
        campaign.run(authorization=campaign.authorization_phrase)
    assert backend.calls == []
    assert [row for row in ledger.list_trials() if row["parent_trial_id"]] == []


def test_unavailable_executor_never_applies_candidate(tmp_path):
    class Backend(_Backend):
        def prepare_candidate(self, candidate, provenance):
            return {**provenance, "code_hash": "wrong"}

    backend = Backend()
    ledger, campaign, baseline, _candidate = _campaign(tmp_path, backend=backend)
    with pytest.raises(RuntimeError, match="execution is unavailable"):
        campaign.run(authorization=campaign.authorization_phrase)
    assert backend.calls == []
    assert [row for row in ledger.list_trials() if row["parent_trial_id"]] == []


def test_unavailable_executor_never_runs_baseline_drift_sequence(tmp_path):
    drifted = PoweredTrialResult(
        success=True,
        reset_epoch_proved=True,
        countdown_go_observed=True,
        watchdogs_armed=True,
        cleanup_confirmed=True,
        fresh_authoritative_state=True,
        no_stale_stream_flight=True,
        valid=True,
        correct_gate_sequence=True,
        completed=True,
        race_time_s=7.0,
        centering_margin=1.0,
        stability_margin=1.0,
    )
    backend = _Backend([_success_result(), _success_result(), drifted])
    ledger, campaign, baseline, _candidate = _campaign(tmp_path, backend=backend)
    with pytest.raises(RuntimeError, match="execution is unavailable"):
        campaign.run(authorization=campaign.authorization_phrase)
    assert backend.calls == []
    assert [row for row in ledger.list_trials() if row["parent_trial_id"]] == []


def test_campaign_safety_types_reject_bool_numeric_and_coerced_identifiers(tmp_path):
    with pytest.raises(ValueError, match="exact non-empty string"):
        CampaignCandidate(1, "x", "gate0", "c", "f", "d", "e")
    with pytest.raises(ValueError, match="process_uptime"):
        PreflightHealth(True, "3385", True)
    backend = _Backend()
    backend.maximum_powered_trial_s = True
    with pytest.raises(ValueError, match="finite powered bound"):
        _campaign(tmp_path, backend=backend)


def test_powered_result_omitted_completion_evidence_defaults_fail_closed():
    result = PoweredTrialResult(
        success=True,
        reset_epoch_proved=True,
        countdown_go_observed=True,
        watchdogs_armed=True,
        cleanup_confirmed=True,
        fresh_authoritative_state=True,
        no_stale_stream_flight=True,
        valid=True,
    )
    assert result.correct_gate_sequence is False
    assert result.completed is False
    assert result.safety_passed is False


def test_campaign_requires_reviewed_watchdog_and_rejects_contract_drift(tmp_path):
    missing = _Backend()
    missing.powered_watchdog_declaration = {}
    with pytest.raises(ValueError, match="watchdog planning declaration"):
        _campaign(tmp_path / "missing", backend=missing)

    backend = _Backend()
    _ledger, campaign, _baseline, _candidate = _campaign(
        tmp_path / "drift", backend=backend
    )
    authorization = campaign.authorization_phrase
    backend.maximum_powered_trial_s = 0.5
    backend.powered_watchdog_declaration = {
        **backend.powered_watchdog_declaration,
        "maximum_powered_trial_s": 0.5,
    }
    with pytest.raises(RuntimeError, match="changed after authorization"):
        campaign.run(authorization=authorization)
    assert backend.calls == []


def test_campaign_plan_cli_emits_only_for_exact_valid_frozen_plan(tmp_path, capsys):
    from aigp_loop.campaign import freeze_backend_contract

    ledger = TrialLedger(tmp_path / "campaign.sqlite3")
    source_id = _source(ledger, "cli-baseline")
    source = ledger.get_trial(source_id)
    candidate = {
        "trial_id": source_id,
        "label": "baseline",
        "stage": "gate0",
        "code_hash": source["code_hash"],
        "config_hash": source["config_hash"],
        "dataset_hash": source["dataset_hash"],
        "evaluator_version": source["evaluator_version"],
        "is_baseline": True,
    }
    payload = {
        "schema": "aigp-live-campaign-plan-input/1",
        "backend_contract": freeze_backend_contract(_Backend()),
        "candidates": [candidate],
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    assert campaign_plan_main(
        [
            str(plan),
            "--ledger",
            str(ledger.path),
            "--simulator-build",
            "3385",
            "--baseline-every",
            "1",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["authorization_phrase"].startswith("AUTHORIZE_POWERED_VQ2:3385:")
    assert len(output["backend_contract_sha256"]) == 64
    assert len(output["execution_schedule"]) == 1


@pytest.mark.parametrize(
    "mutation,args",
    [
        (lambda value: value.update(schema="wrong"), []),
        (lambda value: value.update(candidates=[]), []),
        (
            lambda value: value["candidates"].append(
                {**value["candidates"][0], "is_baseline": False}
            ),
            [],
        ),
        (
            lambda value: value["candidates"].append(
                {**value["candidates"][0], "trial_id": "second-baseline"}
            ),
            [],
        ),
        (lambda value: None, ["--baseline-every", "0"]),
    ],
)
def test_campaign_plan_cli_refuses_malformed_authorization_plan(tmp_path, mutation, args):
    from aigp_loop.campaign import freeze_backend_contract

    payload = {
        "schema": "aigp-live-campaign-plan-input/1",
        "backend_contract": freeze_backend_contract(_Backend()),
        "candidates": [
            {
                "trial_id": "baseline-trial",
                "label": "baseline",
                "stage": "gate0",
                "code_hash": "c" * 64,
                "config_hash": "f" * 64,
                "dataset_hash": "d" * 64,
                "evaluator_version": "eval-v1",
                "is_baseline": True,
            }
        ],
    }
    mutation(payload)
    plan = tmp_path / "bad-plan.json"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    argv = [
        str(plan),
        "--ledger",
        str(tmp_path / "empty.sqlite3"),
        "--simulator-build",
        "3385",
    ] + args
    with pytest.raises(SystemExit):
        campaign_plan_main(argv)


def test_repeated_execution_attempts_remain_unavailable_and_never_power(tmp_path):
    backend = _Backend()
    _ledger, campaign, _baseline, _candidate = _campaign(
        tmp_path, backend=backend
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="execution is unavailable"):
            campaign.run(authorization=campaign.authorization_phrase)
    assert backend.calls == []


def test_unavailable_executor_does_not_touch_preexisting_ambiguous_occurrence(tmp_path):
    backend = _Backend()
    ledger, campaign, baseline, _candidate = _campaign(tmp_path, backend=backend)
    source = ledger.get_trial(baseline)
    child_config = {
        **source["resolved_config"],
        "preexisting_powered_occurrence": "ambiguous",
    }
    child_id, _created = ledger.create_or_get_trial(
        key=TrialKey(
            code_hash=source["code_hash"],
            config_hash=json_hash(child_config),
            dataset_hash=source["dataset_hash"],
            seed=source["seed"],
            evaluator_version=source["evaluator_version"],
        ),
        commit_hash=source["commit_hash"],
        dirty_diff_hash=source["dirty_diff_hash"],
        resolved_config=child_config,
        environment_fingerprint=source["environment_fingerprint"],
        parent_trial_id=baseline,
        candidate_name="preexisting-ambiguous-live-occurrence",
        simulator_build=campaign.simulator_build,
        artifact_hashes=source["artifact_hashes"],
    )
    assert ledger.lease_trial(child_id, "crashed", ttl_s=1.0)
    ledger.checkpoint(
        child_id,
        5,
        owner="crashed",
        status="running",
        metrics={
            "powered_occurrence_state": "committed_to_power",
            "powered_repeated": False,
        },
    )
    with sqlite3.connect(ledger.path) as db:
        db.execute(
            "UPDATE trials SET lease_expires_at=? WHERE trial_id=?",
            (time.time() - 1.0, child_id),
        )
    with pytest.raises(RuntimeError, match="execution is unavailable"):
        campaign.run(authorization=campaign.authorization_phrase)
    assert backend.calls == []
    row = ledger.get_trial(child_id)
    assert row["status"] == "running"
    assert row["parent_trial_id"] == baseline
    assert ledger.get_checkpoint(child_id, 5)["metrics"]["powered_repeated"] is False


def test_unavailable_executor_starts_no_setup_or_heartbeat(
    tmp_path, monkeypatch
):
    class SlowSetupBackend(_Backend):
        def prepare_candidate(self, candidate, provenance):
            time.sleep(0.04)
            return super().prepare_candidate(candidate, provenance)

        def passive_preflight(self):
            time.sleep(0.04)
            return super().passive_preflight()

    backend = SlowSetupBackend()
    ledger, campaign, baseline, _candidate = _campaign(tmp_path, backend=backend)
    heartbeats = []
    original = ledger.heartbeat

    def observed(trial_id, owner, **kwargs):
        heartbeats.append(trial_id)
        return original(trial_id, owner, **kwargs)

    monkeypatch.setattr(ledger, "heartbeat", observed)
    with pytest.raises(RuntimeError, match="execution is unavailable"):
        campaign.run(authorization=campaign.authorization_phrase)
    assert heartbeats == []
    assert backend.calls == []
    assert [row for row in ledger.list_trials() if row["parent_trial_id"]] == []
