from __future__ import annotations

from aigp_loop.promotion import (
    CandidateEvaluation,
    HardGates,
    PromotionLadder,
    QualityVector,
    Tier,
    TierEligibility,
)


SAFE = HardGates(True, True, True, True, True, True, True)


def _evaluation(name, *, gates=SAFE, quality=QualityVector()):
    return CandidateEvaluation(name, Tier.T2_WARM_SIM, gates, quality)


def test_safety_failure_cannot_be_bought_back_by_faster_time():
    unsafe_fast = _evaluation(
        "unsafe-fast",
        gates=HardGates(False, True, True, True, True, True, True),
        quality=QualityVector(1.0, 1.0, 1.0, 1.0),
    )
    safe_slow = _evaluation(
        "safe-slow",
        quality=QualityVector(1.0, 0.5, 0.5, 100.0),
    )
    decision = PromotionLadder(keep_fraction=1.0).decide([unsafe_fast, safe_slow])
    assert decision.promoted == ("safe-slow",)
    assert decision.rejected_hard_gate["unsafe-fast"] == ("collision",)


def test_quality_is_lexicographic_not_a_weighted_scalar():
    reliable = _evaluation(
        "reliable",
        quality=QualityVector(0.99, 0.1, 0.1, 50.0),
    )
    centered_fast = _evaluation(
        "centered-fast",
        quality=QualityVector(0.98, 100.0, 100.0, 1.0),
    )
    decision = PromotionLadder(keep_fraction=0.5).decide([centered_fast, reliable])
    assert decision.promoted == ("reliable",)


def test_successive_halving_allocates_more_repetitions_to_fewer_candidates():
    rows = PromotionLadder(keep_fraction=0.5).rounds(["d", "c", "b", "a"])
    assert rows[:4] == (("a", 1), ("b", 1), ("c", 1), ("d", 1))
    assert rows[4:6] == (("a", 2), ("b", 2))
    assert rows[-1] == ("a", 4)


def test_successive_halving_rejects_duplicate_candidate_ids():
    import pytest

    with pytest.raises(ValueError, match="unique"):
        PromotionLadder().rounds(["candidate-a", "candidate-a"])


def test_hard_gates_require_complete_exact_boolean_evidence():
    import pytest

    with pytest.raises(ValueError, match="missing"):
        HardGates.from_mapping({})
    complete = {
        "no_collision": True,
        "no_disqualification": True,
        "no_stale_stream_flight": True,
        "cleanup_confirmed": True,
        "correct_gate_sequence": True,
        "completed": True,
        "valid": True,
    }
    for invalid in (1, "false"):
        evidence = dict(complete)
        evidence["valid"] = invalid
        with pytest.raises(TypeError, match="exact bool"):
            HardGates.from_mapping(evidence)


def test_quality_and_repetition_reject_bool_or_coerced_values():
    import pytest

    with pytest.raises(TypeError, match="not bool"):
        QualityVector(True, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="repetition"):
        CandidateEvaluation("x", Tier.T2_WARM_SIM, SAFE, repetitions=True)
    with pytest.raises(TypeError, match="Tier"):
        CandidateEvaluation("x", 0, SAFE)
    with pytest.raises(ValueError, match="keep_fraction"):
        PromotionLadder(keep_fraction=True)
    with pytest.raises(ValueError, match="minimum_survivors"):
        PromotionLadder(minimum_survivors=True)


def test_replay_tier_uses_scoped_eligibility_not_flight_hard_gates():
    import pytest

    with pytest.raises(TypeError, match="non-flight eligibility"):
        CandidateEvaluation("x", Tier.T1_VQ2_REPLAY, SAFE)
    candidate = CandidateEvaluation(
        "x",
        Tier.T1_VQ2_REPLAY,
        eligibility=TierEligibility("golden-replay", True, "a" * 64),
    )
    assert PromotionLadder().decide([candidate]).promoted == ("x",)
    with pytest.raises(ValueError, match="golden-replay"):
        CandidateEvaluation(
            "wrong-scope",
            Tier.T1_VQ2_REPLAY,
            eligibility=TierEligibility("affected-tests", True),
        )


def test_t0_promotes_every_passing_candidate_before_halving_begins():
    candidates = [
        CandidateEvaluation(
            name,
            Tier.T0_AFFECTED,
            eligibility=TierEligibility("affected-tests", True, str(index) * 64),
        )
        for index, name in enumerate(("alpha", "bravo", "charlie"), start=1)
    ]
    decision = PromotionLadder(keep_fraction=0.01).decide(candidates)
    assert set(decision.promoted) == {"alpha", "bravo", "charlie"}
    assert decision.eliminated_by_halving == ()
