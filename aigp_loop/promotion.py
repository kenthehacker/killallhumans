"""Safety-first promotion ladder and deterministic successive halving."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Mapping, Optional, Sequence

from ._util import json_hash


class Tier(IntEnum):
    T0_AFFECTED = 0
    T1_VQ2_REPLAY = 1
    T2_WARM_SIM = 2
    T3_DOMAIN_TRACKS = 3
    T4_FULL_NON_LIVE = 4
    T5_AUTHORIZED_LIVE = 5


TIER_TARGET_SECONDS: Mapping[Tier, tuple[float, float]] = {
    Tier.T0_AFFECTED: (0.0, 2.0),
    Tier.T1_VQ2_REPLAY: (2.0, 8.0),
    Tier.T2_WARM_SIM: (2.0, 5.0),
    Tier.T3_DOMAIN_TRACKS: (5.0, 90.0),
    Tier.T4_FULL_NON_LIVE: (30.0, 900.0),
    # Live duration is deliberately not optimized or used as a timeout here.
    Tier.T5_AUTHORIZED_LIVE: (0.0, math.inf),
}


@dataclass(frozen=True)
class HardGates:
    """Non-negotiable outcome gates; none can be offset by a faster time."""

    no_collision: bool
    no_disqualification: bool
    no_stale_stream_flight: bool
    cleanup_confirmed: bool
    correct_gate_sequence: bool
    completed: bool
    valid: bool

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.no_collision,
                self.no_disqualification,
                self.no_stale_stream_flight,
                self.cleanup_confirmed,
                self.correct_gate_sequence,
                self.completed,
                self.valid,
            )
        ):
            raise TypeError("every hard-gate field must be an exact bool")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HardGates":
        expected = {
            "no_collision",
            "no_disqualification",
            "no_stale_stream_flight",
            "cleanup_confirmed",
            "correct_gate_sequence",
            "completed",
            "valid",
        }
        if type(value) is not dict:
            raise TypeError("hard_gates must be an exact object")
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                f"hard-gate evidence must be exact; missing={missing}, unknown={unknown}"
            )
        return cls(**{name: value[name] for name in expected})

    @property
    def passed(self) -> bool:
        return all(
            (
                self.no_collision,
                self.no_disqualification,
                self.no_stale_stream_flight,
                self.cleanup_confirmed,
                self.correct_gate_sequence,
                self.completed,
                self.valid,
            )
        )

    def failures(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in (
                ("collision", self.no_collision),
                ("disqualification", self.no_disqualification),
                ("stale_stream_flight", self.no_stale_stream_flight),
                ("cleanup_failure", self.cleanup_confirmed),
                ("gate_sequence", self.correct_gate_sequence),
                ("incomplete", self.completed),
                ("invalid", self.valid),
            )
            if not value
        )


@dataclass(frozen=True)
class QualityVector:
    """Lexicographic quality dimensions used only after hard gates pass."""

    completion_reliability: float = 0.0
    centering_margin: float = 0.0
    stability_margin: float = 0.0
    race_time_s: Optional[float] = None

    def __post_init__(self) -> None:
        values = (
            self.completion_reliability,
            self.centering_margin,
            self.stability_margin,
        )
        if any(type(value) not in {int, float} for value in values):
            raise TypeError("quality values must be numeric and not bool")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("quality values must be finite")
        if self.race_time_s is not None and (
            type(self.race_time_s) not in {int, float}
            or not math.isfinite(self.race_time_s)
            or self.race_time_s < 0.0
        ):
            raise ValueError("race_time_s must be finite and non-negative")

    def ordering_key(self) -> tuple[float, float, float, float]:
        # The negated race time makes smaller times rank later in a normal
        # descending tuple without blending it into a weighted score.
        time_component = -self.race_time_s if self.race_time_s is not None else -math.inf
        return (
            self.completion_reliability,
            self.centering_margin,
            self.stability_margin,
            time_component,
        )


@dataclass(frozen=True)
class TierEligibility:
    """Non-flight eligibility for T0/T1; never claims closed-loop safety."""

    scope: str
    passed: bool
    evidence_hash: Optional[str] = None
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in {"affected-tests", "golden-replay"}:
            raise ValueError("unknown non-flight eligibility scope")
        if type(self.passed) is not bool:
            raise TypeError("eligibility passed must be an exact bool")
        if self.evidence_hash is not None and (
            type(self.evidence_hash) is not str
            or len(self.evidence_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.evidence_hash)
        ):
            raise ValueError("eligibility evidence_hash must be a SHA-256 digest")
        if self.scope == "golden-replay" and self.passed and self.evidence_hash is None:
            raise ValueError("passing golden replay requires an evidence hash")
        if (
            type(self.failures) is not tuple
            or any(type(item) is not str or not item for item in self.failures)
        ):
            raise TypeError("eligibility failures must be non-empty strings")
        if self.passed and self.failures:
            raise ValueError("passing eligibility cannot contain failures")
        if not self.passed and not self.failures:
            raise ValueError("failed eligibility requires an explicit reason")


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    tier: Tier
    hard_gates: Optional[HardGates] = None
    quality: QualityVector = field(default_factory=QualityVector)
    repetitions: int = 1
    metrics: Mapping[str, Any] = field(default_factory=dict)
    eligibility: Optional[TierEligibility] = None

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string")
        if type(self.tier) is not Tier:
            raise TypeError("tier must be a Tier value")
        if self.tier <= Tier.T1_VQ2_REPLAY:
            if self.hard_gates is not None or type(self.eligibility) is not TierEligibility:
                raise TypeError("T0/T1 require scoped non-flight eligibility, not HardGates")
            expected_scope = (
                "affected-tests"
                if self.tier is Tier.T0_AFFECTED
                else "golden-replay"
            )
            if self.eligibility.scope != expected_scope:
                raise ValueError(
                    f"{self.tier.name} requires {expected_scope} eligibility"
                )
        elif type(self.hard_gates) is not HardGates or self.eligibility is not None:
            raise TypeError("T2+ require closed-loop HardGates only")
        if type(self.quality) is not QualityVector:
            raise TypeError("quality must be QualityVector")
        if type(self.repetitions) is not int or self.repetitions < 1:
            raise ValueError("candidate_id and at least one repetition are required")
        if type(self.metrics) is not dict:
            raise TypeError("metrics must be an exact object")


@dataclass(frozen=True)
class PromotionDecision:
    tier: Tier
    promoted: tuple[str, ...]
    rejected_hard_gate: Mapping[str, tuple[str, ...]]
    eliminated_by_halving: tuple[str, ...]
    next_tier: Optional[Tier]


class PromotionLadder:
    """Rank candidates without permitting safety-for-speed tradeoffs."""

    def __init__(self, *, keep_fraction: float = 0.5, minimum_survivors: int = 1) -> None:
        if (
            type(keep_fraction) not in {int, float}
            or not math.isfinite(keep_fraction)
            or not 0.0 < keep_fraction <= 1.0
        ):
            raise ValueError("keep_fraction must be in (0, 1]")
        if type(minimum_survivors) is not int or minimum_survivors < 1:
            raise ValueError("minimum_survivors must be >= 1")
        self.keep_fraction = float(keep_fraction)
        self.minimum_survivors = int(minimum_survivors)

    def decide(self, evaluations: Sequence[CandidateEvaluation]) -> PromotionDecision:
        if not evaluations:
            raise ValueError("at least one evaluation is required")
        tier = evaluations[0].tier
        if any(item.tier != tier for item in evaluations):
            raise ValueError("a halving round cannot mix promotion tiers")
        identifiers = [item.candidate_id for item in evaluations]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate ids must be unique within a round")

        if tier <= Tier.T1_VQ2_REPLAY:
            rejected = {
                item.candidate_id: item.eligibility.failures
                for item in evaluations
                if item.eligibility is not None and not item.eligibility.passed
            }
            eligible = [
                item for item in evaluations
                if item.eligibility is not None and item.eligibility.passed
            ]
        else:
            rejected = {
                item.candidate_id: item.hard_gates.failures()
                for item in evaluations
                if item.hard_gates is not None and not item.hard_gates.passed
            }
            eligible = [
                item for item in evaluations
                if item.hard_gates is not None and item.hard_gates.passed
            ]
        eligible.sort(
            key=lambda item: (item.quality.ordering_key(), item.candidate_id),
            reverse=True,
        )
        if tier is Tier.T0_AFFECTED:
            # T0 is a correctness screen, not a ranking round: every passing
            # candidate advances to replay exactly as the ladder contract
            # states, regardless of a caller's keep_fraction.
            keep = len(eligible)
        elif eligible:
            keep = max(
                self.minimum_survivors,
                int(math.ceil(len(eligible) * self.keep_fraction)),
            )
            keep = min(keep, len(eligible))
        else:
            keep = 0
        promoted = tuple(item.candidate_id for item in eligible[:keep])
        eliminated = tuple(item.candidate_id for item in eligible[keep:])
        next_tier = Tier(int(tier) + 1) if tier < Tier.T5_AUTHORIZED_LIVE else None
        return PromotionDecision(
            tier=tier,
            promoted=promoted,
            rejected_hard_gate=rejected,
            eliminated_by_halving=eliminated,
            next_tier=next_tier,
        )

    def rounds(
        self,
        candidate_ids: Iterable[str],
        *,
        initial_repetitions: int = 1,
        repetition_multiplier: int = 2,
    ) -> tuple[tuple[str, int], ...]:
        """Return a deterministic work budget for a complete halving series.

        This helper plans budgets only; hard-gate outcomes still decide actual
        survival after each round.
        """

        raw_ids = tuple(candidate_ids)
        if any(type(candidate_id) is not str or not candidate_id.strip() for candidate_id in raw_ids):
            raise ValueError("candidate ids must be non-empty strings")
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError("candidate ids must be unique")
        ids = tuple(sorted(raw_ids))
        if (
            type(initial_repetitions) is not int
            or type(repetition_multiplier) is not int
            or initial_repetitions < 1
            or repetition_multiplier < 2
        ):
            raise ValueError("positive initial repetitions and multiplier >=2 required")
        rows: list[tuple[str, int]] = []
        repetitions = initial_repetitions
        remaining = len(ids)
        while remaining:
            for candidate in ids[:remaining]:
                rows.append((candidate, repetitions))
            if remaining == 1:
                break
            remaining = max(1, math.ceil(remaining * self.keep_fraction))
            repetitions *= repetition_multiplier
        return tuple(rows)


def _find_evidence(metrics: Mapping[str, Any], schemas: set[str]) -> Optional[Mapping[str, Any]]:
    # Import locally because evidence scope depends on the Tier enum defined in
    # this module.  Ambiguous duplicate payloads must fail closed rather than
    # whichever nested mapping happens to be visited first winning.
    from .evidence import find_unique_schema_evidence

    return find_unique_schema_evidence(metrics, schemas)


_REPLAY_PROMOTION_REQUIRED_BOUNDS: Mapping[str, Mapping[str, float]] = {
    "annotation_frame_coverage": {"min": 1.0},
    "active_gate_label_coverage": {"min": 1.0},
    "active_gate_label_mismatch_count": {"max": 0.0},
    "perception.gate_truth_count": {"min": 1.0},
    "perception.gate_recall": {"min": 0.99},
    "perception.false_positives_per_frame": {"max": 0.01},
    "perception.center_error_px_p95": {"max": 8.0},
    "perception.corner_error_px_p95": {"max": 12.0},
    "perception.longest_consecutive_missed_frames": {"max": 2.0},
    "perception.temporal_center_step_px_p95": {"max": 30.0},
    "perception.transition_count": {"min": 1.0},
    "perception.unreacquired_count": {"max": 0.0},
    "perception.post_gate_reacquisition_latency_ms_p95": {"max": 200.0},
    "perception.full_stack_latency_ms_p95": {"max": 100.0},
    "estimator.missing_frame_estimates": {"max": 0.0},
    "estimator.invalid_frame_estimates": {"max": 0.0},
    "estimator.health_label_coverage": {"min": 1.0},
    "estimator.health_comparison_coverage": {"min": 1.0},
    "estimator.health_mismatch_count": {"max": 0.0},
    "estimator.rpy_label_coverage": {"min": 1.0},
    "estimator.rpy_comparison_coverage": {"min": 1.0},
    "estimator.rpy_reference_rms_rad": {"min": 0.05},
    "estimator.rpy_rmse_rad": {"max": 0.1},
    "open_loop_commands.replay_frames.generated_count": {"min": 1.0},
    "open_loop_commands.replay_frames.invalid_count": {"max": 0.0},
    "open_loop_commands.replay_frames.envelope_violation_count": {"max": 0.0},
    "open_loop_commands.replay_frames.expected_command_label_coverage": {"min": 1.0},
    "open_loop_commands.replay_frames.expected_command_comparison_coverage": {"min": 1.0},
    "open_loop_commands.replay_frames.expected_command_reference_rms": {"min": 0.05},
    "open_loop_commands.replay_frames.expected_command_rmse": {"max": 0.05},
}


def replay_promotion_policy_failures(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    """Recompute mandatory promotion gates; ``passed=true`` alone is never enough."""

    if evidence.get("schema") == "aigp-vq2-replay-corpus-score/1":
        sessions = evidence.get("sessions")
        if type(sessions) is not list or not sessions:
            return ("corpus has no session policy evidence",)
        failures: list[str] = []
        for index, session in enumerate(sessions):
            if type(session) is not dict:
                failures.append(f"session {index} is malformed")
            else:
                failures.extend(
                    f"session {index}: {failure}"
                    for failure in replay_promotion_policy_failures(session)
                )
        return tuple(failures)
    if evidence.get("schema") != "aigp-vq2-replay-score/1":
        return ("replay score schema is not promotion-capable",)
    policy = evidence.get("policy")
    expected_policy_keys = {
        "schema",
        "policy_hash",
        "passed",
        "constraints",
        "observed",
        "violations",
    }
    actual_policy_keys = set(policy) if type(policy) is dict else set()
    if (
        type(policy) is not dict
        or frozenset(actual_policy_keys)
        not in {frozenset(expected_policy_keys), frozenset(expected_policy_keys | {"policy_file_sha256"})}
        or policy.get("schema") != "aigp-vq2-replay-policy-result/1"
        or policy.get("passed") is not True
        or policy.get("violations") != []
        or type(policy.get("constraints")) is not dict
        or type(policy.get("observed")) is not dict
    ):
        return ("replay policy result is missing, failed, or malformed",)
    if "policy_file_sha256" in policy:
        digest = policy["policy_file_sha256"]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return ("replay policy file hash is malformed",)
    constraints = policy["constraints"]
    observed = policy["observed"]
    if policy.get("policy_hash") != json_hash(
        {"schema": "aigp-vq2-replay-policy/1", "metrics": constraints}
    ):
        return ("replay policy hash does not bind its constraints",)
    failures = []
    for path, required in _REPLAY_PROMOTION_REQUIRED_BOUNDS.items():
        bounds = constraints.get(path)
        value = observed.get(path)
        if type(bounds) is not dict or type(value) not in {int, float} or not math.isfinite(value):
            failures.append(f"mandatory replay metric missing/nonfinite: {path}")
            continue
        if "min" in required:
            minimum = bounds.get("min")
            if (
                type(minimum) not in {int, float}
                or not math.isfinite(minimum)
                or minimum < required["min"]
                or value < required["min"]
            ):
                failures.append(f"mandatory replay minimum is too weak/failed: {path}")
        if "max" in required:
            maximum = bounds.get("max")
            if (
                type(maximum) not in {int, float}
                or not math.isfinite(maximum)
                or maximum > required["max"]
                or value > required["max"]
            ):
                failures.append(f"mandatory replay maximum is too weak/failed: {path}")
    return tuple(failures)


def validate_promotion_chain(ledger: Any, trial_id: str) -> Mapping[str, Any]:
    """Validate the identity-bound T0-T4 attestation used by merge and T5.

    This is intentionally stricter than merely checking five ``completed``
    flags.  Every expensive evaluator result must match the frozen full-ladder
    manifest embedded in the TrialKey and each checkpoint must retain the
    scheduler's metrics/tier-identity artifact hashes.
    """

    trial = ledger.get_trial(trial_id)
    if trial["status"] != "completed" or set(ledger.completed_tiers(trial_id)) != set(range(5)):
        raise ValueError("promotion source requires a completed T0-T4 checkpoint chain")
    config = trial.get("resolved_config")
    manifest = config.get("promotion_ladder_manifest") if isinstance(config, Mapping) else None
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema", "tiers"}
        or manifest.get("schema") != "aigp-promotion-ladder-manifest/2"
        or type(manifest.get("tiers")) is not list
        or len(manifest["tiers"]) != 5
    ):
        raise ValueError("promotion source lacks an exact full-ladder manifest")
    fields = {
        "tier",
        "dataset_hash",
        "config_hash",
        "seed",
        "repetitions",
        "evaluator_version",
        "command_plan_sha256",
    }
    identities: dict[int, Mapping[str, Any]] = {}
    for identity in manifest["tiers"]:
        if type(identity) is not dict or set(identity) != fields:
            raise ValueError("promotion source tier identity is malformed")
        tier_number = identity["tier"]
        if type(tier_number) is not int or tier_number not in range(5) or tier_number in identities:
            raise ValueError("promotion source tier numbers must be unique 0..4")
        for name in ("dataset_hash", "config_hash"):
            value = identity[name]
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"promotion source {name} is not SHA-256")
        if type(identity["seed"]) is not int:
            raise ValueError("promotion source seed must be an exact integer")
        if type(identity["repetitions"]) is not int or identity["repetitions"] < 1:
            raise ValueError("promotion source repetitions must be positive")
        if type(identity["evaluator_version"]) is not str or not identity["evaluator_version"].strip():
            raise ValueError("promotion source evaluator version is missing")
        command_plan = identity["command_plan_sha256"]
        if (
            type(command_plan) is not str
            or len(command_plan) != 64
            or any(
                character not in "0123456789abcdef"
                for character in command_plan
            )
        ):
            raise ValueError("promotion source command plan is not SHA-256")
        identities[tier_number] = identity
    if set(identities) != set(range(5)):
        raise ValueError("promotion source manifest must bind every tier")
    manifest_hash = json_hash(manifest)
    if trial.get("dataset_hash") != manifest_hash:
        raise ValueError("promotion source TrialKey does not bind its ladder manifest")
    if trial.get("evaluator_version") != f"aigp-ladder/2:{manifest_hash}":
        raise ValueError("promotion source evaluator version does not bind its ladder manifest")

    from .evidence import validate_tier_evidence

    tier_evidence: dict[int, Mapping[str, Any]] = {}
    for tier_number in range(5):
        checkpoint = ledger.get_checkpoint(trial_id, tier_number)
        assert checkpoint is not None
        artifacts = checkpoint.get("artifact_hashes")
        if type(artifacts) is not dict:
            raise ValueError("promotion checkpoint artifact hashes are missing")
        for name in (
            "metrics_sha256",
            "manifest_tier_identity_sha256",
            "command_plan_sha256",
            "tier_identity_sha256",
        ):
            digest = artifacts.get(name)
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"promotion checkpoint lacks {name}")
        if artifacts["metrics_sha256"] != json_hash(checkpoint["metrics"]):
            raise ValueError("promotion checkpoint metrics hash is stale")
        tier = Tier(tier_number)
        try:
            tier_evidence[tier_number] = validate_tier_evidence(
                tier, checkpoint["metrics"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"promotion checkpoint {tier.name} evidence scope is invalid: {exc}"
            ) from exc
        if artifacts["manifest_tier_identity_sha256"] != json_hash(
            identities[tier_number]
        ):
            raise ValueError("promotion checkpoint manifest tier identity is stale")
        if (
            artifacts["command_plan_sha256"]
            != identities[tier_number]["command_plan_sha256"]
        ):
            raise ValueError(
                "promotion checkpoint command plan differs from frozen TrialKey"
            )
        if artifacts["tier_identity_sha256"] != json_hash(
            {
                "manifest_tier_identity_sha256": artifacts[
                    "manifest_tier_identity_sha256"
                ],
                "command_plan_sha256": artifacts["command_plan_sha256"],
            }
        ):
            raise ValueError("promotion checkpoint command/tier identity is stale")

    t1_checkpoint = ledger.get_checkpoint(trial_id, int(Tier.T1_VQ2_REPLAY))
    assert t1_checkpoint is not None
    replay = tier_evidence[int(Tier.T1_VQ2_REPLAY)]
    t1_identity = identities[int(Tier.T1_VQ2_REPLAY)]
    t1_observed = (
        {
            "dataset_hash": replay.get(
                "evaluation_input_hash", replay.get("evaluation_evidence_hash")
            ),
            "config_hash": replay.get("evaluation_config_sha256"),
            "seed": replay.get("seed"),
            "repetitions": replay.get("repetitions"),
            "evaluator_version": replay.get("evaluator_version"),
        }
        if replay is not None
        else None
    )
    t1_expected = {name: t1_identity[name] for name in t1_observed} if t1_observed else None
    if (
        replay is None
        or t1_observed != t1_expected
        or type(replay.get("policy")) is not dict
        or replay["policy"].get("passed") is not True
        or replay.get("processor") == "recorded"
        or type(replay.get("processor_code_sha256")) is not str
        or len(replay["processor_code_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in replay["processor_code_sha256"]
        )
        or replay.get("processor_code_sha256") != trial.get("code_hash")
        or type(replay.get("domain_provenance")) is not dict
        or replay["domain_provenance"].get("perception")
        != "candidate_detector_on_all_decoded_frames"
        or replay["domain_provenance"].get("estimator")
        != "candidate_estimator_on_ordered_sanitized_stream"
        or replay["domain_provenance"].get("open_loop_commands")
        != "candidate_generator_on_ordered_sanitized_stream"
        or type(replay.get("candidate_isolation")) is not dict
        or set(replay["candidate_isolation"])
        != {
            "schema",
            "network",
            "filesystem",
            "non_interactive",
            "process_tree_containment",
            "host_process_access",
            "wrapper_sha256",
        }
        or replay["candidate_isolation"].get("schema")
        != "aigp-replay-isolation-attestation/1"
        or replay["candidate_isolation"].get("network") != "denied"
        or replay["candidate_isolation"].get("filesystem")
        != "readonly-worktree-only"
        or replay["candidate_isolation"].get("non_interactive") is not True
        or replay["candidate_isolation"].get("process_tree_containment")
        != "kill-on-wrapper-exit"
        or replay["candidate_isolation"].get("host_process_access") != "denied"
        or type(replay["candidate_isolation"].get("wrapper_sha256")) is not str
        or len(replay["candidate_isolation"]["wrapper_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in replay["candidate_isolation"]["wrapper_sha256"]
        )
    ):
        raise ValueError("promotion source lacks identity-bound passing T1 replay evidence")
    policy_failures = replay_promotion_policy_failures(replay)
    if policy_failures:
        raise ValueError(
            "promotion source has weak/invalid T1 replay policy: "
            + "; ".join(policy_failures)
        )

    from .nonlive import CORE_EVALUATOR_FILES, DOMAIN_TRACK_SET, FULL_TRACK_SET

    for tier in (Tier.T2_WARM_SIM, Tier.T3_DOMAIN_TRACKS, Tier.T4_FULL_NON_LIVE):
        checkpoint = ledger.get_checkpoint(trial_id, int(tier))
        assert checkpoint is not None
        evidence = tier_evidence[int(tier)]
        identity = identities[int(tier)]
        observed = (
            {
                "dataset_hash": evidence.get("evaluation_input_hash"),
                "config_hash": evidence.get("evaluation_config_sha256"),
                "seed": evidence.get("seed"),
                "repetitions": evidence.get("repetitions"),
                "evaluator_version": evidence.get("evaluator_version"),
            }
            if evidence is not None
            else None
        )
        expected = {name: identity[name] for name in observed} if observed else None
        required_tracks = {
            Tier.T2_WARM_SIM: ("race_01",),
            Tier.T3_DOMAIN_TRACKS: DOMAIN_TRACK_SET,
            Tier.T4_FULL_NON_LIVE: FULL_TRACK_SET,
        }[tier]
        try:
            source_hashes = (
                evidence["evaluator_identity"]["source_sha256"]
                if evidence is not None
                and type(evidence.get("evaluator_identity")) is dict
                else None
            )
            passed = bool(
                evidence is not None
                and observed == expected
                and evidence.get("tier") == int(tier)
                and tuple(sorted(evidence.get("track_identity", ())))
                == tuple(sorted(required_tracks))
                and HardGates.from_mapping(evidence["promotion"]["hard_gates"]).passed
                and type(evidence.get("domain_provenance")) is dict
                and evidence["domain_provenance"].get("powered_resources_used") is False
                and type(source_hashes) is dict
                and CORE_EVALUATOR_FILES
                <= set(source_hashes)
                and all(
                    type(digest) is str
                    and len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest)
                    for digest in source_hashes.values()
                )
                and checkpoint["artifact_hashes"].get(
                    "trusted_evaluator_files_sha256"
                )
                == json_hash(source_hashes)
            )
        except (KeyError, TypeError, ValueError):
            passed = False
        if not passed:
            raise ValueError(
                f"promotion source lacks identity-bound passing {tier.name} evidence"
            )
    return manifest
