from __future__ import annotations

import copy

import pytest

from aigp_loop.evidence import (
    EvidenceDomain,
    GateAuthorityClaim,
    TierEvidenceScopeV1,
    find_unique_schema_evidence,
    scope_for_tier,
    validate_tier_evidence,
)
from aigp_loop.nonlive import DOMAIN_TRACK_SET, FULL_TRACK_SET
from aigp_loop.promotion import Tier, _find_evidence
from aigp_loop.scheduler import TrialScheduler


def _replay_evidence(schema: str = "aigp-vq2-replay-score/1"):
    return {
        "schema": schema,
        "domain_provenance": {
            "perception": "candidate_detector_on_all_decoded_frames",
            "estimator": "candidate_estimator_on_ordered_sanitized_stream",
            "open_loop_commands": "candidate_generator_on_ordered_sanitized_stream",
        },
    }


def _nonlive_evidence(tier: Tier):
    tracks = {
        Tier.T2_WARM_SIM: ("race_01",),
        Tier.T3_DOMAIN_TRACKS: DOMAIN_TRACK_SET,
        Tier.T4_FULL_NON_LIVE: FULL_TRACK_SET,
    }[tier]
    return {
        "schema": "aigp-nonlive-promotion-evidence/1",
        "tier": int(tier),
        "track_identity": list(tracks),
        "domain_provenance": {
            "execution": "deterministic_synthetic_kinematic_nonpowered",
            "powered_resources_used": False,
            "cleanup_gate_semantics": "vacuously_true_only_after_synthetic_domain_proof",
            "stale_stream_gate_semantics": "vacuously_true_only_after_synthetic_domain_proof",
            "centering_proxy": "negative_worst_p95_tracking_error_m",
            "stability_proxy": "negative_worst_max_tracking_error_m",
        },
    }


def test_tier_scopes_freeze_open_loop_and_synthetic_claim_boundaries():
    t0 = scope_for_tier(Tier.T0_AFFECTED)
    t1 = scope_for_tier(Tier.T1_VQ2_REPLAY)
    t2 = scope_for_tier(Tier.T2_WARM_SIM)
    assert t0.domain is EvidenceDomain.AFFECTED_TESTS
    assert not t0.closed_loop and not t0.powered
    assert t1.domain is EvidenceDomain.REPLAY_OPEN_LOOP
    assert not t1.closed_loop and not t1.powered
    assert t1.gate_authority is GateAuthorityClaim.NONE
    assert t2.domain is EvidenceDomain.SYNTHETIC_CLOSED_LOOP
    assert t2.closed_loop and not t2.powered
    assert t2.gate_authority is GateAuthorityClaim.SYNTHETIC_SEQUENCE


def test_tier_scope_codec_rejects_domain_relabeling():
    scope = scope_for_tier(Tier.T1_VQ2_REPLAY)
    assert TierEvidenceScopeV1.from_primitive(scope.to_primitive()) == scope
    relabeled = scope.to_primitive()
    relabeled["closed_loop"] = True
    with pytest.raises(ValueError, match="relabel"):
        TierEvidenceScopeV1.from_primitive(relabeled)


def test_t5_has_no_ordinary_evidence_contract():
    with pytest.raises(ValueError, match="unavailable"):
        scope_for_tier(Tier.T5_AUTHORIZED_LIVE)


def test_t0_accepts_only_nonflight_metrics_without_inventing_a_payload_schema():
    metrics = {"passed": True, "tests": 25}
    assert validate_tier_evidence(Tier.T0_AFFECTED, metrics) is metrics
    assert scope_for_tier(Tier.T0_AFFECTED).payload_schemas == ()


def test_t0_allows_flight_named_pytest_identifiers_but_still_scans_their_results():
    metrics = {
        "tests": {
            "test_flight_control_rate_bounds": {"outcome": "passed"},
            "flight_control/tests/test_rate.py": {"outcome": "passed"},
            "flight_control/tests/test_rate.py::test_live_marker_is_excluded": {
                "outcome": "passed"
            },
        }
    }
    assert validate_tier_evidence(Tier.T0_AFFECTED, metrics) is metrics

    metrics["tests"]["test_flight_control_rate_bounds"]["flight"] = False
    with pytest.raises(ValueError, match="cannot contain flight-domain claim"):
        validate_tier_evidence(Tier.T0_AFFECTED, metrics)


@pytest.mark.parametrize(
    "claim",
    [
        {"closed_loop": False},
        {"powered_resources_used": False},
        {"gate-authority": "none"},
        {"official_simulator": False},
        {"execution": "powered"},
        {"simulator": "official-simulator"},
        {"closedLoop": True},
        {"powered_execution": False},
        {"executionMode": "powered_flightsim"},
        {"schema": "aigp-live-trial-evidence/1"},
    ],
)
def test_t0_recursively_rejects_flight_domain_claims_even_when_false(claim):
    metrics = {"results": [{"details": claim}]}
    with pytest.raises(ValueError, match="cannot contain flight-domain claim"):
        validate_tier_evidence(Tier.T0_AFFECTED, metrics)


@pytest.mark.parametrize(
    "schema",
    ["aigp-vq2-replay-score/1", "aigp-vq2-replay-corpus-score/1"],
)
def test_t1_accepts_only_replay_open_loop_schemas(schema):
    evidence = _replay_evidence(schema)
    assert validate_tier_evidence(Tier.T1_VQ2_REPLAY, {"result": evidence}) is evidence
    evidence["domain_provenance"] = {"powered_resources_used": True}
    with pytest.raises(ValueError, match="cannot contain flight-domain claim"):
        validate_tier_evidence(Tier.T1_VQ2_REPLAY, {"result": evidence})


@pytest.mark.parametrize(
    "claim",
    [
        {"closed-loop": False},
        {"powered": False},
        {"gate_authority_verified": False},
        {"execution": "official simulator"},
        {"simulator": "official FlightSim build 3385"},
        {"isPowered": False},
    ],
)
def test_t1_rejects_nested_relabeling_claims_anywhere_in_replay(claim):
    evidence = _replay_evidence()
    evidence["nested"] = {"records": [claim]}
    with pytest.raises(ValueError, match="cannot contain flight-domain claim"):
        validate_tier_evidence(Tier.T1_VQ2_REPLAY, {"result": evidence})


def test_t1_requires_exact_domain_provenance():
    evidence = _replay_evidence()
    evidence.pop("domain_provenance")
    with pytest.raises(ValueError, match="exact domain provenance"):
        validate_tier_evidence(Tier.T1_VQ2_REPLAY, evidence)
    evidence = _replay_evidence()
    evidence["domain_provenance"]["estimator"] = "opaque_estimator_claim"
    with pytest.raises(ValueError, match="exact causal open-loop provenance"):
        validate_tier_evidence(Tier.T1_VQ2_REPLAY, evidence)


@pytest.mark.parametrize(
    "tier",
    [Tier.T2_WARM_SIM, Tier.T3_DOMAIN_TRACKS, Tier.T4_FULL_NON_LIVE],
)
def test_t2_to_t4_require_exact_synthetic_nonpowered_scope(tier):
    evidence = _nonlive_evidence(tier)
    assert validate_tier_evidence(tier, {"wrapped": evidence}) is evidence
    powered = copy.deepcopy(evidence)
    powered["domain_provenance"]["powered_resources_used"] = True
    with pytest.raises(ValueError, match="wrong synthetic domain"):
        validate_tier_evidence(tier, {"wrapped": powered})


def test_cross_tier_nonlive_evidence_is_rejected():
    with pytest.raises(ValueError, match="wrong synthetic domain"):
        validate_tier_evidence(
            Tier.T3_DOMAIN_TRACKS, {"result": _nonlive_evidence(Tier.T2_WARM_SIM)}
        )


@pytest.mark.parametrize(
    "claim",
    [
        {"powered": True},
        {"official_simulator_used": True},
        {"executionMode": "powered_flightsim"},
        {"schema": "aigp-live-trial-evidence/1"},
    ],
)
def test_t2_to_t4_reject_contradictory_nested_domain_claims(claim):
    evidence = _nonlive_evidence(Tier.T2_WARM_SIM)
    evidence["results"] = {"race_01": {"nested": claim}}
    with pytest.raises(ValueError, match="cannot contain flight-domain claim"):
        validate_tier_evidence(Tier.T2_WARM_SIM, evidence)


@pytest.mark.parametrize(
    "tier",
    [Tier.T0_AFFECTED, Tier.T1_VQ2_REPLAY, Tier.T2_WARM_SIM],
)
@pytest.mark.parametrize(
    "claim",
    [
        {"claimedDomain": "powered"},
        {"flightSimulator": "official FlightSim build 3385"},
        {"liveFlight": True},
        {"domainClaim": "official_simulator"},
        {"executionClaim": "powered"},
        {"claimedMode": "live"},
        {"schemaClaim": "aigp-live-trial-evidence/1"},
        {"domainClaim": "flight_sim"},
        {"gateSequenceAuthority": "synthetic_sequence"},
        {"claim": ["powered"]},
        {"claim": {"value": "powered"}},
        {"attestation": ["official simulator", "armed"]},
        {"result": "powered"},
        {"status": "live"},
        {"live": True},
        {"scope": "powered"},
        {"verdict": "powered"},
        {"environment": "official FlightSim"},
        {"kind": "powered"},
        {"phase": "live"},
    ],
)
def test_composite_and_camel_claim_aliases_cannot_relabel_any_nonlive_tier(
    tier, claim
):
    if tier is Tier.T0_AFFECTED:
        metrics = {"results": [{"nested": claim}]}
    elif tier is Tier.T1_VQ2_REPLAY:
        metrics = _replay_evidence()
        metrics["nested"] = claim
    else:
        metrics = _nonlive_evidence(tier)
        metrics["results"] = {"race_01": {"nested": claim}}
    with pytest.raises(ValueError, match="cannot contain flight-domain claim"):
        validate_tier_evidence(tier, metrics)


def test_generic_claim_namespace_allows_benign_nonflight_test_evidence():
    metrics = {"claims": ["affected tests passed", "imports succeeded"]}
    assert validate_tier_evidence(Tier.T0_AFFECTED, metrics) is metrics


@pytest.mark.parametrize("tier", [Tier.T0_AFFECTED, Tier.T1_VQ2_REPLAY])
@pytest.mark.parametrize(
    "claim",
    [
        {"flight": True},
        {"live": True},
        {"armed": True},
        {"gate_passed": True},
        {"gatePassed": False},
        {"domain": "gate_authority"},
        {"simulator": "official"},
    ],
)
def test_nonflight_tiers_reject_reserved_flight_key_and_value_vocabulary(
    tier, claim
):
    if tier is Tier.T0_AFFECTED:
        metrics = {"results": [{"nested": claim}]}
    else:
        metrics = _replay_evidence()
        metrics["nested"] = claim
    with pytest.raises(ValueError, match="cannot contain flight-domain claim"):
        validate_tier_evidence(tier, metrics)


def test_nonlive_provenance_has_an_exact_claim_namespace():
    evidence = _nonlive_evidence(Tier.T2_WARM_SIM)
    evidence["domain_provenance"]["simulator"] = "synthetic"
    with pytest.raises(ValueError, match="wrong synthetic domain"):
        validate_tier_evidence(Tier.T2_WARM_SIM, evidence)


def test_duplicate_matching_evidence_fails_closed_instead_of_first_match_wins():
    metrics = {
        "first": _replay_evidence(),
        "second": _replay_evidence("aigp-vq2-replay-corpus-score/1"),
    }
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        find_unique_schema_evidence(
            metrics,
            {"aigp-vq2-replay-score/1", "aigp-vq2-replay-corpus-score/1"},
        )


def test_promotion_and_scheduler_use_the_unique_evidence_contract():
    metrics = {"first": _replay_evidence(), "second": _replay_evidence()}
    schemas = {"aigp-vq2-replay-score/1"}
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        _find_evidence(metrics, schemas)
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        TrialScheduler._find_schema_evidence(metrics, schemas)


def test_scheduler_validates_t0_scope_before_manifest_early_return():
    scheduler = object.__new__(TrialScheduler)
    trial = {"resolved_config": {}}
    failure = scheduler._tier_evidence_binding_failure(
        trial,
        Tier.T0_AFFECTED,
        {"results": [{"powered": False}]},
    )
    assert failure is not None
    assert "T0_AFFECTED tier evidence scope is invalid" in failure
    assert scheduler._tier_evidence_binding_failure(
        trial, Tier.T0_AFFECTED, {"passed": True}
    ) is None


def test_scheduler_validates_t1_scope_without_a_ladder_manifest():
    scheduler = object.__new__(TrialScheduler)
    trial = {"resolved_config": {}}
    evidence = _replay_evidence()
    evidence["nested"] = {"gate_authority": "none"}
    failure = scheduler._tier_evidence_binding_failure(
        trial, Tier.T1_VQ2_REPLAY, evidence
    )
    assert failure is not None
    assert "T1_VQ2_REPLAY tier evidence scope is invalid" in failure


def test_unique_evidence_search_rejects_shared_sibling_evidence_but_handles_cycles():
    evidence = _replay_evidence()
    metrics = {"first": evidence, "same": evidence}
    metrics["cycle"] = metrics
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        find_unique_schema_evidence(metrics, {"aigp-vq2-replay-score/1"})
    cyclic_wrapper = {"result": evidence}
    cyclic_wrapper["cycle"] = cyclic_wrapper
    assert find_unique_schema_evidence(
        cyclic_wrapper, {"aigp-vq2-replay-score/1"}
    ) is evidence


def test_canonical_corpus_envelope_is_one_evidence_unit_not_duplicate_sessions():
    corpus = _replay_evidence("aigp-vq2-replay-corpus-score/1")
    corpus["sessions"] = [_replay_evidence(), _replay_evidence()]
    schemas = {"aigp-vq2-replay-score/1", "aigp-vq2-replay-corpus-score/1"}
    assert find_unique_schema_evidence({"result": corpus}, schemas) is corpus
    assert _find_evidence({"result": corpus}, schemas) is corpus
    assert TrialScheduler._find_schema_evidence({"result": corpus}, schemas) is corpus


def test_corpus_only_suppresses_canonical_session_children_not_hidden_evidence():
    corpus = _replay_evidence("aigp-vq2-replay-corpus-score/1")
    corpus["sessions"] = [_replay_evidence()]
    corpus["hidden"] = _replay_evidence()
    schemas = {"aigp-vq2-replay-score/1", "aigp-vq2-replay-corpus-score/1"}
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        find_unique_schema_evidence({"result": corpus}, schemas)

    session = _replay_evidence()
    session["hidden"] = _replay_evidence()
    corpus = _replay_evidence("aigp-vq2-replay-corpus-score/1")
    corpus["sessions"] = [session]
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        find_unique_schema_evidence({"result": corpus}, schemas)

    replay = _replay_evidence()
    replay["nested"] = _replay_evidence("aigp-vq2-replay-corpus-score/1")
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        find_unique_schema_evidence({"result": replay}, schemas)


def test_tier_scope_rejects_unknown_fields_and_coerced_tier():
    encoded = scope_for_tier(Tier.T2_WARM_SIM).to_primitive()
    encoded["extra"] = None
    with pytest.raises(ValueError, match="fields must be exact"):
        TierEvidenceScopeV1.from_primitive(encoded)
    encoded = scope_for_tier(Tier.T2_WARM_SIM).to_primitive()
    encoded["tier"] = True
    with pytest.raises(TypeError, match="exact integer"):
        TierEvidenceScopeV1.from_primitive(encoded)
