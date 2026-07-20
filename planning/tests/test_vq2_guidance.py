from __future__ import annotations

from dataclasses import replace

import pytest

from competition.vq2_contracts import (
    FeatureCovarianceV1,
    FrameEdge,
    FrameIdentityV1,
    GateAuthorityEpochV1,
    MeasurementTimeBasis,
    PredictionBasis,
    PredictionTimeV1,
    RelativeGateStateV1,
    RelativeStateHealth,
    TrackRole,
    validate_relative_gate_state_sequence,
)
from planning.vq2_guidance import (
    VQ2GuidanceConfig,
    VQ2GuidanceObjectiveKind,
    VQ2GuidancePhase,
    VQ2GuidanceRaceState,
    VQ2GuidanceWithholdingReason,
    VQ2SafetyGuidanceInput,
    _VQ2_GUIDANCE_FIRST_DROPOUT_CAPABILITY,
    _step_vq2_guidance_first_observation_dropout,
    _step_vq2_guidance_impl,
    step_vq2_guidance,
)
from planning.vq2_guidance_scenarios import (
    SYNTHETIC_GUIDANCE_SCOPE,
    evaluate_synthetic_vq2_guidance_scenario,
)


_STATE_FEATURE_ORDER = (
    "bearing_x_norm",
    "bearing_y_norm",
    "log_scale",
    "bearing_rate_x_norm_s",
    "bearing_rate_y_norm_s",
    "expansion_rate_s",
)
_ALL_EDGES = FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT | FrameEdge.BOTTOM
_PHASE_PATH = (
    VQ2GuidancePhase.ACQUIRE,
    VQ2GuidancePhase.ALIGN,
    VQ2GuidancePhase.APPROACH,
    VQ2GuidancePhase.COMMIT,
    VQ2GuidancePhase.CONFIRMATION,
)
_EXPECTED_SCENARIO_DIGEST = (
    "13b5a7b3120826780e529b0d183a4298ba562937d47edf15773092471d4516df"
)


def _authority(
    sequence: int,
    *,
    session_id: str = "session-a",
    reset_epoch: int = 0,
    gate_epoch: int = 0,
    gate_index: int = 0,
    generation: int = 3,
    race_boot_base_ms: int = 4_000,
) -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id=session_id,
        reset_epoch=reset_epoch,
        gate_epoch=gate_epoch,
        expected_gate_index=gate_index,
        race_status_sequence=100 + sequence,
        race_status_boot_ms=race_boot_base_ms + 20 * sequence,
        camera_host_clock_id="host-a",
        camera_stream_id="camera-a",
        camera_generation=generation,
        frame_publication_sequence_not_before=1_000 + 10 * sequence,
        frame_publish_monotonic_ns_not_before=(
            2_000_000_000 + 10_000_000 * sequence
        ),
    )


def _safety(
    authority: GateAuthorityEpochV1,
    phase: VQ2GuidancePhase,
    race_state: VQ2GuidanceRaceState,
    *,
    evaluation_monotonic_ns: int | None = None,
    phase_started_monotonic_ns: int | None = None,
    evaluation_host_clock_id: str | None = None,
) -> VQ2SafetyGuidanceInput:
    evaluation_ns = (
        authority.frame_publish_monotonic_ns_not_before + 200_000_000
        if evaluation_monotonic_ns is None
        else evaluation_monotonic_ns
    )
    return VQ2SafetyGuidanceInput(
        authority=authority,
        phase=phase,
        race_state=race_state,
        evaluation_host_clock_id=(
            authority.camera_host_clock_id
            if evaluation_host_clock_id is None
            else evaluation_host_clock_id
        ),
        evaluation_monotonic_ns=evaluation_ns,
        phase_started_monotonic_ns=(
            evaluation_ns
            if phase_started_monotonic_ns is None
            else phase_started_monotonic_ns
        ),
    )


def _initial(
    *,
    sequence: int = 0,
):
    authority = _authority(sequence)
    safety = _safety(
        authority,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    return step_vq2_guidance(None, safety, active_state=None)


def _ready_phase(
    phase: VQ2GuidancePhase,
    *,
    start_sequence: int = 0,
):
    transition = _initial(sequence=start_sequence)
    target_index = _PHASE_PATH.index(phase)
    safety = transition.memory.safety
    for offset, selected_phase in enumerate(
        _PHASE_PATH[: target_index + 1],
        start=1,
    ):
        authority = _authority(
            start_sequence + offset,
        )
        safety = _safety(
            authority,
            selected_phase,
            VQ2GuidanceRaceState.UNDERWAY,
            phase_started_monotonic_ns=(
                safety.phase_started_monotonic_ns
                if selected_phase is safety.phase
                else None
            ),
        )
        transition = step_vq2_guidance(
            transition.memory,
            safety,
            active_state=None,
        )
    return transition, safety, start_sequence + target_index + 2


def _state(
    safety: VQ2SafetyGuidanceInput,
    *,
    state_sequence: int = 1,
    measurement_update_sequence: int | None = None,
    frame_id: int | None = None,
    candidate_id: str | None = None,
    tracker_id: str = "active-gate-0",
    track_role: TrackRole = TrackRole.ACTIVE,
    bearing_norm: tuple[float, float] = (0.02, -0.01),
    bearing_rate_norm_s: tuple[float, float] = (0.01, -0.01),
    log_scale: float = -0.3,
    expansion_rate_s: float = 0.2,
    variance: tuple[float, float, float, float, float, float] = (
        0.0001,
        0.0001,
        0.0004,
        0.0004,
        0.0004,
        0.0004,
    ),
    clipping: FrameEdge = FrameEdge.NONE,
    health: RelativeStateHealth = RelativeStateHealth.HEALTHY,
    dropout_count: int = 0,
    innovation_accepted: bool | None = True,
    publication_offset: int = 1,
    publish_monotonic_ns: int | None = None,
    measurement_monotonic_ns: int | None = None,
    measurement_uncertainty_ns: int = 1_000_000,
    decision_monotonic_ns: int | None = None,
    prediction_monotonic_ns: int | None = None,
    delay_uncertainty_ns: int = 0,
) -> RelativeGateStateV1:
    authority = safety.authority
    evaluation_ns = safety.evaluation_monotonic_ns
    update_sequence = (
        state_sequence
        if measurement_update_sequence is None
        else measurement_update_sequence
    )
    actual_frame_id = state_sequence if frame_id is None else frame_id
    actual_candidate = (
        f"candidate-{actual_frame_id}" if candidate_id is None else candidate_id
    )
    publish_ns = (
        evaluation_ns - 30_000_000 + publication_offset * 1_000_000
        if publish_monotonic_ns is None
        else publish_monotonic_ns
    )
    measurement_ns = (
        publish_ns - 5_000_000
        if measurement_monotonic_ns is None
        else measurement_monotonic_ns
    )
    decision_ns = (
        publish_ns + 5_000_000
        if decision_monotonic_ns is None
        else decision_monotonic_ns
    )
    prediction_ns = (
        decision_ns if prediction_monotonic_ns is None else prediction_monotonic_ns
    )
    prediction_basis = (
        PredictionBasis.DECISION_TIME
        if prediction_ns == decision_ns
        else PredictionBasis.COMMAND_EFFECT_ESTIMATE
    )
    frame = FrameIdentityV1(
        stream_id=authority.camera_stream_id,
        generation=authority.camera_generation,
        frame_id=actual_frame_id,
    )
    timing = PredictionTimeV1(
        host_clock_id=authority.camera_host_clock_id,
        source_frame=frame,
        source_frame_publication_sequence=(
            authority.frame_publication_sequence_not_before + publication_offset
        ),
        source_frame_publish_monotonic_ns=publish_ns,
        measurement_time_monotonic_ns=measurement_ns,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        measurement_time_model_id=None,
        measurement_uncertainty_ns=measurement_uncertainty_ns,
        decision_time_monotonic_ns=decision_ns,
        prediction_time_monotonic_ns=prediction_ns,
        prediction_basis=prediction_basis,
        delay_model_id=(
            None
            if prediction_basis is PredictionBasis.DECISION_TIME
            else "synthetic-delay-model"
        ),
        delay_uncertainty_ns=delay_uncertainty_ns,
    )
    covariance = FeatureCovarianceV1(
        model_id="synthetic-guidance-test-diagonal",
        feature_order=_STATE_FEATURE_ORDER,
        matrix=tuple(
            tuple(value if row == column else 0.0 for column in range(6))
            for row, value in enumerate(variance)
        ),
    )
    visibility = _ALL_EDGES & ~clipping
    innovation = (
        (None, None, None)
        if dropout_count
        else (
            10.0 if innovation_accepted is False else 1.0,
            9.0,
            innovation_accepted,
        )
    )
    return RelativeGateStateV1(
        timing=timing,
        authority=authority,
        tracker_id=tracker_id,
        state_sequence=state_sequence,
        measurement_update_sequence=update_sequence,
        source_candidate_id=actual_candidate,
        track_role=track_role,
        bearing_norm=bearing_norm,
        bearing_rate_norm_s=bearing_rate_norm_s,
        log_scale=log_scale,
        expansion_rate_s=expansion_rate_s,
        covariance=covariance,
        metric_position_body_frd_m=None,
        metric_velocity_body_frd_m_s=None,
        metric_gate_orientation_body_frd_xyzw=None,
        metric_covariance=None,
        last_clipping=clipping,
        outer_visibility=visibility,
        inner_visibility=visibility,
        normalized_innovation_squared=innovation[0],
        innovation_gate_threshold=innovation[1],
        innovation_accepted=innovation[2],
        dropout_count=dropout_count,
        health=health,
        health_reason=(
            None if health is RelativeStateHealth.HEALTHY else "synthetic"
        ),
    )


def _with_active(phase: VQ2GuidancePhase, **state_kwargs):
    transition, safety, _next_sequence = _ready_phase(phase)
    return step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(safety, **state_kwargs),
    )


def _first_dropout_from(state: RelativeGateStateV1) -> RelativeGateStateV1:
    timing = replace(
        state.timing,
        decision_time_monotonic_ns=(
            state.timing.decision_time_monotonic_ns + 1_000_000
        ),
        prediction_time_monotonic_ns=(
            state.timing.prediction_time_monotonic_ns + 1_000_000
        ),
    )
    return replace(
        state,
        timing=timing,
        state_sequence=state.state_sequence + 1,
        normalized_innovation_squared=None,
        innovation_gate_threshold=None,
        innovation_accepted=None,
        dropout_count=1,
        health=RelativeStateHealth.COASTING,
        health_reason="observation_dropout",
    )


def _guidance_memory_with_first_dropout(phase: VQ2GuidancePhase):
    if phase is VQ2GuidancePhase.ALIGN:
        transition, safety, _next_sequence = _ready_gate1_align()
        tracker_id = "active-gate-1"
    else:
        transition, safety, _next_sequence = _ready_phase(phase)
        tracker_id = "active-gate-0"
    prior = _state(safety, tracker_id=tracker_id)
    accepted = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=prior,
    )
    assert accepted.decision.objective_permitted
    return accepted.memory, safety, _first_dropout_from(prior)


@pytest.mark.parametrize(
    ("phase", "race_state"),
    [
        (VQ2GuidancePhase.ALIGN, VQ2GuidanceRaceState.NOT_UNDERWAY),
        (VQ2GuidancePhase.ACQUIRE, VQ2GuidanceRaceState.UNDERWAY),
        (VQ2GuidancePhase.COMMIT, VQ2GuidanceRaceState.UNDERWAY),
    ],
)
def test_initialization_requires_acquire_and_not_underway(phase, race_state) -> None:
    authority = _authority(0)
    with pytest.raises(ValueError, match="initial guidance input"):
        step_vq2_guidance(
            None,
            _safety(authority, phase, race_state),
            active_state=None,
        )


def test_initialization_is_closed_and_command_free() -> None:
    result = _initial()

    assert not result.decision.objective_permitted
    assert result.decision.objective_kind is VQ2GuidanceObjectiveKind.HOLD
    assert (
        result.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.RACE_NOT_UNDERWAY
    )
    assert (
        result.decision.phase_started_monotonic_ns
        == result.decision.evaluation_monotonic_ns
    )
    assert not hasattr(result.decision, "requested_body_rates_rad_s")
    assert not hasattr(result.decision, "requested_thrust")


def test_initialization_requires_phase_start_at_evaluation_boundary() -> None:
    authority = _authority(0)
    evaluation_ns = authority.frame_publish_monotonic_ns_not_before + 200_000_000
    safety = _safety(
        authority,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
        evaluation_monotonic_ns=evaluation_ns,
        phase_started_monotonic_ns=evaluation_ns - 1,
    )

    with pytest.raises(ValueError, match="phase start must equal evaluation time"):
        step_vq2_guidance(None, safety, active_state=None)


@pytest.mark.parametrize(
    ("gate_epoch", "gate_index"),
    [(1, 1), (0, 1), (1, 0), (41, 41)],
)
def test_fresh_memory_requires_gate_epoch_and_index_zero(
    gate_epoch: int,
    gate_index: int,
) -> None:
    authority = _authority(
        0,
        gate_epoch=gate_epoch,
        gate_index=gate_index,
    )
    safety = _safety(
        authority,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )

    with pytest.raises(ValueError, match="gate epoch/index zero"):
        step_vq2_guidance(None, safety, active_state=None)


def test_fresh_memory_allows_nonzero_reset_epoch_at_gate_zero() -> None:
    authority = _authority(0, reset_epoch=7)
    safety = _safety(
        authority,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )

    result = step_vq2_guidance(None, safety, active_state=None)

    assert result.memory.safety.authority.reset_epoch == 7
    assert result.memory.safety.authority.gate_epoch == 0
    assert result.memory.safety.authority.expected_gate_index == 0


def test_safety_input_rejects_phase_start_one_ns_after_evaluation() -> None:
    authority = _authority(0)
    evaluation_ns = authority.frame_publish_monotonic_ns_not_before + 200_000_000

    with pytest.raises(ValueError, match="phase start cannot postdate evaluation"):
        _safety(
            authority,
            VQ2GuidancePhase.ACQUIRE,
            VQ2GuidanceRaceState.NOT_UNDERWAY,
            evaluation_monotonic_ns=evaluation_ns,
            phase_started_monotonic_ns=evaluation_ns + 1,
        )


def test_evaluation_clock_must_match_authority_clock() -> None:
    authority = _authority(0)
    with pytest.raises(ValueError, match="evaluation host clock"):
        _safety(
            authority,
            VQ2GuidancePhase.ACQUIRE,
            VQ2GuidanceRaceState.NOT_UNDERWAY,
            evaluation_host_clock_id="different-clock",
        )


def test_evaluation_time_cannot_predate_authority_cutover() -> None:
    authority = _authority(0)
    with pytest.raises(ValueError, match="predate the authority cutover"):
        _safety(
            authority,
            VQ2GuidancePhase.ACQUIRE,
            VQ2GuidanceRaceState.NOT_UNDERWAY,
            evaluation_monotonic_ns=(
                authority.frame_publish_monotonic_ns_not_before - 1
            ),
        )


def test_tightening_only_config_accepts_stricter_values() -> None:
    config = VQ2GuidanceConfig(
        uncertainty_sigma_multiplier=4.0,
        align_corridor_half_width_norm=(0.8, 0.8),
        approach_corridor_half_width_norm=(0.15, 0.15),
        commit_corridor_half_width_norm=(0.05, 0.05),
        align_rate_limit_norm_s=(1.5, 1.5),
        approach_rate_limit_norm_s=(0.3, 0.3),
        commit_rate_limit_norm_s=(0.1, 0.1),
        commit_min_log_scale=-0.5,
        commit_min_expansion_rate_s=0.1,
        max_state_decision_age_ns=90_000_000,
        max_measurement_age_ns=140_000_000,
        max_prediction_lead_ns=90_000_000,
        max_measurement_uncertainty_ns=40_000_000,
    )

    assert config.uncertainty_sigma_multiplier == 4.0


def test_sigma_multiplier_has_a_finite_numerical_ceiling() -> None:
    assert VQ2GuidanceConfig(
        uncertainty_sigma_multiplier=1_000_000.0
    ).uncertainty_sigma_multiplier == 1_000_000.0
    with pytest.raises(ValueError, match="numerical ceiling"):
        VQ2GuidanceConfig(uncertainty_sigma_multiplier=1_000_000.1)


@pytest.mark.parametrize(
    "changes",
    [
        {"target_bearing_norm": (0.01, 0.0)},
        {"uncertainty_sigma_multiplier": 2.999},
        {"align_corridor_half_width_norm": (0.951, 0.85)},
        {"approach_corridor_half_width_norm": (0.181, 0.18)},
        {"commit_corridor_half_width_norm": (0.081, 0.08)},
        {"align_rate_limit_norm_s": (2.001, 2.0)},
        {"approach_rate_limit_norm_s": (0.351, 0.35)},
        {"commit_rate_limit_norm_s": (0.181, 0.18)},
        {"commit_min_log_scale": -0.701},
        {"commit_min_expansion_rate_s": 0.049},
        {"max_state_decision_age_ns": 100_000_001},
        {"max_measurement_age_ns": 150_000_001},
        {"max_prediction_lead_ns": 100_000_001},
        {"max_measurement_uncertainty_ns": 50_000_001},
    ],
)
def test_config_rejects_every_looser_limit(changes) -> None:
    with pytest.raises(ValueError, match="tightening-only|fixed"):
        VQ2GuidanceConfig(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "align_corridor_half_width_norm": (0.1, 0.1),
            "approach_corridor_half_width_norm": (0.15, 0.15),
        },
        {
            "approach_rate_limit_norm_s": (0.1, 0.1),
            "commit_rate_limit_norm_s": (0.15, 0.15),
        },
    ],
)
def test_config_preserves_phase_tightening_order(changes) -> None:
    with pytest.raises(ValueError, match="preserve"):
        VQ2GuidanceConfig(**changes)


def test_visual_state_never_advances_acquire_phase() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ACQUIRE)
    first = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(safety, state_sequence=1),
    )
    later_safety = replace(
        safety,
        evaluation_monotonic_ns=safety.evaluation_monotonic_ns + 1_000_000,
    )
    second = step_vq2_guidance(
        first.memory,
        later_safety,
        active_state=_state(
            later_safety,
            state_sequence=2,
            publication_offset=2,
        ),
    )

    assert first.memory.safety.phase is VQ2GuidancePhase.ACQUIRE
    assert second.memory.safety.phase is VQ2GuidancePhase.ACQUIRE
    assert not first.decision.objective_permitted
    assert not second.decision.objective_permitted


def test_phase_change_requires_strict_forward_authority_snapshot() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ACQUIRE)
    attempted_evaluation_ns = safety.evaluation_monotonic_ns + 1
    attempted = replace(
        safety,
        phase=VQ2GuidancePhase.ALIGN,
        evaluation_monotonic_ns=attempted_evaluation_ns,
        phase_started_monotonic_ns=attempted_evaluation_ns,
    )
    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert rejected.decision.phase is VQ2GuidancePhase.ACQUIRE
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_PHASE_TRANSITION_REJECTED
    )


def test_adjacent_phase_change_with_forward_authority_is_accepted() -> None:
    transition, safety, next_sequence = _ready_phase(VQ2GuidancePhase.ACQUIRE)
    align_authority = _authority(next_sequence)
    align_safety = _safety(
        align_authority,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    result = step_vq2_guidance(
        transition.memory,
        align_safety,
        active_state=_state(align_safety),
    )

    assert result.memory.safety.phase is VQ2GuidancePhase.ALIGN
    assert (
        result.memory.safety.phase_started_monotonic_ns
        == align_safety.evaluation_monotonic_ns
    )
    assert (
        result.decision.phase_started_monotonic_ns
        == align_safety.evaluation_monotonic_ns
    )
    assert result.decision.objective_permitted
    assert result.decision.objective_kind is VQ2GuidanceObjectiveKind.RECENTER_ACTIVE_GATE


def test_phase_transition_rejects_start_one_ns_before_evaluation() -> None:
    transition, _safety_value, next_sequence = _ready_phase(
        VQ2GuidancePhase.ACQUIRE
    )
    authority = _authority(next_sequence)
    evaluation_ns = authority.frame_publish_monotonic_ns_not_before + 200_000_000
    attempted = _safety(
        authority,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        evaluation_monotonic_ns=evaluation_ns,
        phase_started_monotonic_ns=evaluation_ns - 1,
    )

    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_PHASE_START_REJECTED
    )
    assert (
        rejected.decision.phase_started_monotonic_ns
        == transition.memory.safety.phase_started_monotonic_ns
    )


def test_race_only_transition_preserves_phase_start_exactly() -> None:
    transition = _initial()
    previous_start_ns = transition.memory.safety.phase_started_monotonic_ns
    underway_safety = _safety(
        _authority(1),
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_started_monotonic_ns=previous_start_ns,
    )

    result = step_vq2_guidance(
        transition.memory,
        underway_safety,
        active_state=None,
    )

    assert result.memory.safety == underway_safety
    assert result.decision.phase_started_monotonic_ns == previous_start_ns


@pytest.mark.parametrize(
    "phase",
    [
        VQ2GuidancePhase.ALIGN,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidancePhase.CONFIRMATION,
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
    ],
)
def test_countdown_cannot_advance_from_acquire(phase) -> None:
    transition = _initial()
    attempted = _safety(
        _authority(1),
        phase,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )

    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_PHASE_TRANSITION_REJECTED
    )


def test_go_and_align_require_two_distinct_forward_transitions() -> None:
    transition = _initial()
    initial_start_ns = transition.memory.safety.phase_started_monotonic_ns
    simultaneous = _safety(
        _authority(1),
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    rejected = step_vq2_guidance(
        transition.memory,
        simultaneous,
        active_state=None,
    )
    assert rejected.memory == transition.memory

    go_safety = _safety(
        _authority(1),
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_started_monotonic_ns=initial_start_ns,
    )
    go = step_vq2_guidance(
        transition.memory,
        go_safety,
        active_state=None,
    )
    assert go.memory.safety == go_safety
    assert go.memory.safety.phase_started_monotonic_ns == initial_start_ns

    align_safety = _safety(
        _authority(2),
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    aligned = step_vq2_guidance(
        go.memory,
        align_safety,
        active_state=None,
    )
    assert aligned.memory.safety == align_safety
    assert aligned.memory.safety.phase_started_monotonic_ns == (
        align_safety.evaluation_monotonic_ns
    )


@pytest.mark.parametrize(
    "terminal_state",
    [VQ2GuidanceRaceState.FINISHED, VQ2GuidanceRaceState.ABORTED],
)
def test_terminal_transition_cannot_simultaneously_advance_phase(
    terminal_state,
) -> None:
    transition, _safety_value, next_sequence = _ready_phase(
        VQ2GuidancePhase.ALIGN
    )
    attempted = _safety(
        _authority(next_sequence),
        VQ2GuidancePhase.APPROACH,
        terminal_state,
    )

    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_PHASE_TRANSITION_REJECTED
    )


@pytest.mark.parametrize(
    "terminal_state",
    [VQ2GuidanceRaceState.FINISHED, VQ2GuidanceRaceState.ABORTED],
)
def test_terminal_state_freezes_phase_on_later_snapshots(terminal_state) -> None:
    transition, safety, next_sequence = _ready_phase(VQ2GuidancePhase.ALIGN)
    phase_start_ns = safety.phase_started_monotonic_ns
    terminal_safety = _safety(
        _authority(next_sequence),
        VQ2GuidancePhase.ALIGN,
        terminal_state,
        phase_started_monotonic_ns=phase_start_ns,
    )
    terminal = step_vq2_guidance(
        transition.memory,
        terminal_safety,
        active_state=None,
    )
    attempted = _safety(
        _authority(next_sequence + 1),
        VQ2GuidancePhase.APPROACH,
        terminal_state,
    )

    rejected = step_vq2_guidance(
        terminal.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == terminal.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_PHASE_TRANSITION_REJECTED
    )


@pytest.mark.parametrize("offset_ns", [-1, 1])
@pytest.mark.parametrize(
    "race_state",
    [VQ2GuidanceRaceState.UNDERWAY, VQ2GuidanceRaceState.FINISHED],
)
def test_same_phase_rewind_or_renewal_fails_closed_and_preserves_memory(
    offset_ns: int,
    race_state: VQ2GuidanceRaceState,
) -> None:
    transition, _safety_value, next_sequence = _ready_phase(
        VQ2GuidancePhase.ALIGN
    )
    previous_start_ns = transition.memory.safety.phase_started_monotonic_ns
    attempted = _safety(
        _authority(next_sequence),
        VQ2GuidancePhase.ALIGN,
        race_state,
        phase_started_monotonic_ns=previous_start_ns + offset_ns,
    )

    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_PHASE_START_REJECTED
    )
    assert rejected.decision.phase_started_monotonic_ns == previous_start_ns


@pytest.mark.parametrize(
    "phase",
    [
        VQ2GuidancePhase.APPROACH,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidancePhase.CONFIRMATION,
    ],
)
def test_phase_skips_fail_closed_and_preserve_memory(phase) -> None:
    transition, _safety_value, next_sequence = _ready_phase(VQ2GuidancePhase.ACQUIRE)
    attempted = _safety(
        _authority(next_sequence),
        phase,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_PHASE_TRANSITION_REJECTED
    )


def test_phase_regression_fails_closed_and_preserves_memory() -> None:
    transition, _safety_value, next_sequence = _ready_phase(VQ2GuidancePhase.ALIGN)
    attempted = _safety(
        _authority(next_sequence),
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert rejected.memory.safety.phase is VQ2GuidancePhase.ALIGN


def test_cross_session_transition_fails_closed_and_preserves_memory() -> None:
    transition, _safety_value, next_sequence = _ready_phase(VQ2GuidancePhase.ALIGN)
    attempted = _safety(
        _authority(next_sequence, session_id="session-b"),
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_SESSION_CHANGED
    )


def test_evaluation_time_regression_fails_closed_and_preserves_memory() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    advanced_safety = replace(
        safety,
        evaluation_monotonic_ns=safety.evaluation_monotonic_ns + 2,
    )
    transition = step_vq2_guidance(
        transition.memory,
        advanced_safety,
        active_state=None,
    )
    attempted = replace(
        advanced_safety,
        evaluation_monotonic_ns=advanced_safety.evaluation_monotonic_ns - 1,
    )
    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_EVALUATION_TIME_REGRESSED
    )


def test_gate_index_cannot_change_without_gate_epoch() -> None:
    transition, _safety_value, next_sequence = _ready_phase(VQ2GuidancePhase.ALIGN)
    attempted = _safety(
        _authority(next_sequence, gate_epoch=0, gate_index=1),
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_AUTHORITY_DISCONTINUITY
    )


def test_gate_credit_requires_confirmation_and_exact_next_gate() -> None:
    transition, _safety_value, next_sequence = _ready_phase(
        VQ2GuidancePhase.CONFIRMATION
    )
    skipped = _safety(
        _authority(next_sequence, gate_epoch=2, gate_index=2),
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    rejected = step_vq2_guidance(
        transition.memory,
        skipped,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_AUTHORITY_DISCONTINUITY
    )


def _confirmation_with_active_track():
    transition, safety, next_sequence = _ready_phase(VQ2GuidancePhase.ACQUIRE)
    transition = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(safety, state_sequence=1),
    )
    for phase in (
        VQ2GuidancePhase.ALIGN,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidancePhase.CONFIRMATION,
    ):
        authority = _authority(next_sequence)
        safety = _safety(
            authority,
            phase,
            VQ2GuidanceRaceState.UNDERWAY,
        )
        transition = step_vq2_guidance(
            transition.memory,
            safety,
            active_state=_state(
                safety,
                state_sequence=next_sequence + 1,
                tracker_id="active-gate-0",
            ),
        )
        next_sequence += 1
    return transition, next_sequence


def _ready_gate1_align():
    """Reach Gate 1 ALIGN only through GO, phases, and exact gate credit."""

    transition, next_sequence = _confirmation_with_active_track()
    post_safety = _safety(
        _authority(next_sequence, gate_epoch=1, gate_index=1),
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = step_vq2_guidance(
        transition.memory,
        post_safety,
        active_state=None,
    )
    next_sequence += 1
    acquire_safety = _safety(
        _authority(next_sequence, gate_epoch=1, gate_index=1),
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = step_vq2_guidance(
        transition.memory,
        acquire_safety,
        active_state=None,
    )
    next_sequence += 1
    align_safety = _safety(
        _authority(next_sequence, gate_epoch=1, gate_index=1),
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = step_vq2_guidance(
        transition.memory,
        align_safety,
        active_state=None,
    )
    return transition, align_safety, next_sequence + 1


def test_forward_gate_credit_retires_active_track_and_holds() -> None:
    transition, next_sequence = _confirmation_with_active_track()
    confirmation_safety = transition.memory.safety
    transition = step_vq2_guidance(
        transition.memory,
        confirmation_safety,
        active_state=None,
        shadow_states=(
            _state(
                confirmation_safety,
                state_sequence=101,
                frame_id=101,
                candidate_id="gate0-shadow",
                tracker_id="gate0-shadow",
                track_role=TrackRole.SHADOW,
                publication_offset=2,
            ),
        ),
    )
    assert len(transition.memory.track_histories) == 2
    gate1_authority = _authority(next_sequence, gate_epoch=1, gate_index=1)
    gate1_safety = _safety(
        gate1_authority,
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    credited = step_vq2_guidance(
        transition.memory,
        gate1_safety,
        active_state=None,
    )

    assert credited.memory.active_source is None
    assert credited.memory.track_histories == ()
    assert credited.memory.retired_active_tracker_ids == ("active-gate-0",)
    assert (
        credited.memory.safety.phase_started_monotonic_ns
        == gate1_safety.evaluation_monotonic_ns
    )
    assert not credited.decision.objective_permitted
    assert (
        credited.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.POST_CREDIT_REACQUIRE_HOLD
    )


def test_gate_transition_rejects_start_one_ns_before_evaluation() -> None:
    transition, next_sequence = _confirmation_with_active_track()
    authority = _authority(next_sequence, gate_epoch=1, gate_index=1)
    evaluation_ns = authority.frame_publish_monotonic_ns_not_before + 200_000_000
    attempted = _safety(
        authority,
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        evaluation_monotonic_ns=evaluation_ns,
        phase_started_monotonic_ns=evaluation_ns - 1,
    )

    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_PHASE_START_REJECTED
    )


def test_retired_tracker_cannot_seed_next_active_gate() -> None:
    transition, next_sequence = _confirmation_with_active_track()
    post_safety = _safety(
        _authority(next_sequence, gate_epoch=1, gate_index=1),
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = step_vq2_guidance(
        transition.memory,
        post_safety,
        active_state=None,
    )
    next_sequence += 1
    acquire_safety = _safety(
        _authority(next_sequence, gate_epoch=1, gate_index=1),
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = step_vq2_guidance(
        transition.memory,
        acquire_safety,
        active_state=None,
    )
    next_sequence += 1
    align_safety = _safety(
        _authority(next_sequence, gate_epoch=1, gate_index=1),
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    rejected = step_vq2_guidance(
        transition.memory,
        align_safety,
        active_state=_state(
            align_safety,
            tracker_id="active-gate-0",
        ),
    )

    assert rejected.memory.active_source is None
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.RETIRED_ACTIVE_TRACK_REUSED
    )


def test_distinct_gate_scoped_tracker_can_seed_gate1_after_exact_credit() -> None:
    transition, safety, _ = _ready_gate1_align()

    accepted = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            tracker_id="active-gate-1",
        ),
    )

    assert accepted.decision.objective_permitted
    assert accepted.memory.active_source is not None
    assert accepted.memory.active_source.tracker_id == "active-gate-1"
    assert "active-gate-0" in accepted.memory.retired_active_tracker_ids


def test_reset_restarts_gate_zero_and_clears_track_history() -> None:
    transition, safety, next_sequence = _ready_phase(VQ2GuidancePhase.ALIGN)
    transition = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(safety),
        shadow_states=(
            _state(
                safety,
                frame_id=101,
                candidate_id="reset-shadow",
                tracker_id="reset-shadow",
                track_role=TrackRole.SHADOW,
                publication_offset=2,
            ),
        ),
    )
    assert len(transition.memory.track_histories) == 2
    reset_authority = _authority(
        next_sequence,
        reset_epoch=1,
        gate_epoch=0,
        gate_index=0,
        generation=4,
        race_boot_base_ms=0,
    )
    reset_safety = _safety(
        reset_authority,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    reset = step_vq2_guidance(
        transition.memory,
        reset_safety,
        active_state=None,
    )

    assert reset.memory.active_source is None
    assert reset.memory.track_histories == ()
    assert reset.memory.retired_active_tracker_ids == ()
    assert reset.memory.safety.authority.reset_epoch == 1
    assert (
        reset.memory.safety.phase_started_monotonic_ns
        == reset_safety.evaluation_monotonic_ns
    )


def test_reset_transition_rejects_start_one_ns_before_evaluation() -> None:
    transition, _safety_value, next_sequence = _ready_phase(
        VQ2GuidancePhase.ALIGN
    )
    authority = _authority(
        next_sequence,
        reset_epoch=1,
        gate_epoch=0,
        gate_index=0,
        generation=4,
        race_boot_base_ms=0,
    )
    evaluation_ns = authority.frame_publish_monotonic_ns_not_before + 200_000_000
    attempted = _safety(
        authority,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
        evaluation_monotonic_ns=evaluation_ns,
        phase_started_monotonic_ns=evaluation_ns - 1,
    )

    rejected = step_vq2_guidance(
        transition.memory,
        attempted,
        active_state=None,
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_PHASE_START_REJECTED
    )


def test_active_authority_mismatch_withholds_without_poisoning_memory() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    foreign_safety = _safety(
        _authority(2, session_id="foreign"),
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        evaluation_monotonic_ns=safety.evaluation_monotonic_ns,
    )
    rejected = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(foreign_safety),
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_AUTHORITY_MISMATCH
    )


def test_shadow_in_active_slot_is_withheld_not_promoted() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    result = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            track_role=TrackRole.SHADOW,
            tracker_id="shadow-0",
        ),
    )

    assert result.memory.active_source is None
    assert (
        result.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_ROLE_REQUIRED
    )


def test_shadow_only_input_never_drives_objective() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    shadow = _state(
        safety,
        track_role=TrackRole.SHADOW,
        tracker_id="shadow-0",
    )
    result = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=None,
        shadow_states=(shadow,),
    )

    assert not result.decision.objective_permitted
    assert result.decision.source is None
    assert (
        result.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_STATE_REQUIRED
    )


def test_stale_shadow_fails_closed_without_affecting_active_memory() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    active = _state(safety)
    stale_shadow = _state(
        safety,
        frame_id=101,
        candidate_id="stale-shadow",
        tracker_id="stale-shadow",
        track_role=TrackRole.SHADOW,
        publish_monotonic_ns=safety.evaluation_monotonic_ns - 120_000_001,
        measurement_monotonic_ns=safety.evaluation_monotonic_ns - 120_000_002,
        decision_monotonic_ns=safety.evaluation_monotonic_ns - 100_000_001,
        prediction_monotonic_ns=safety.evaluation_monotonic_ns - 100_000_001,
    )
    result = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=active,
        shadow_states=(stale_shadow,),
    )

    assert result.memory == transition.memory
    assert (
        result.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SHADOW_INPUT_INVALID
    )


def test_shadow_values_do_not_change_active_objective_math() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    active = _state(safety)
    left_shadow = _state(
        safety,
        frame_id=101,
        candidate_id="shadow-left",
        tracker_id="shadow-left",
        track_role=TrackRole.SHADOW,
        bearing_norm=(-0.9, 0.8),
        publication_offset=2,
    )
    right_shadow = _state(
        safety,
        frame_id=102,
        candidate_id="shadow-right",
        tracker_id="shadow-right",
        track_role=TrackRole.SHADOW,
        bearing_norm=(0.9, -0.8),
        publication_offset=2,
    )
    left = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=active,
        shadow_states=(left_shadow,),
    )
    right = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=active,
        shadow_states=(right_shadow,),
    )

    assert left.decision.objective_permitted
    assert right.decision.objective_permitted
    assert (
        left.decision.conservative_bearing_error_norm
        == right.decision.conservative_bearing_error_norm
    )
    assert left.decision.corridor_margin_norm == right.decision.corridor_margin_norm
    assert left.decision.source == right.decision.source


def test_active_shadow_collision_fails_closed() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    active = _state(safety)
    colliding = replace(active, track_role=TrackRole.SHADOW)
    result = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=active,
        shadow_states=(colliding,),
    )

    assert result.memory == transition.memory
    assert (
        result.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SHADOW_INPUT_INVALID
    )


@pytest.mark.parametrize("collision", ["tracker", "source"])
def test_same_call_shadow_owner_collisions_are_atomic(collision) -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    first = _state(
        safety,
        frame_id=101,
        candidate_id="shadow-a",
        tracker_id="shadow-a",
        track_role=TrackRole.SHADOW,
    )
    second_kwargs = {
        "state_sequence": 2,
        "frame_id": 102,
        "candidate_id": "shadow-b",
        "tracker_id": "shadow-b",
        "track_role": TrackRole.SHADOW,
        "publication_offset": 2,
    }
    if collision == "tracker":
        second_kwargs["tracker_id"] = first.tracker_id
    else:
        second_kwargs["frame_id"] = first.timing.source_frame.frame_id
        second_kwargs["candidate_id"] = first.source_candidate_id
    second = _state(safety, **second_kwargs)

    rejected = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=None,
        shadow_states=(first, second),
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SHADOW_INPUT_INVALID
    )


def test_late_invalid_shadow_rolls_back_earlier_staged_shadow() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    valid_shadow = _state(
        safety,
        frame_id=101,
        candidate_id="valid-shadow",
        tracker_id="valid-shadow",
        track_role=TrackRole.SHADOW,
    )
    stale_decision_ns = safety.evaluation_monotonic_ns - 100_000_001
    stale_shadow = _state(
        safety,
        frame_id=102,
        candidate_id="stale-shadow",
        tracker_id="stale-shadow",
        track_role=TrackRole.SHADOW,
        publication_offset=2,
        publish_monotonic_ns=stale_decision_ns - 2,
        measurement_monotonic_ns=stale_decision_ns - 3,
        decision_monotonic_ns=stale_decision_ns,
        prediction_monotonic_ns=stale_decision_ns,
    )

    rejected = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=None,
        shadow_states=(valid_shadow, stale_shadow),
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SHADOW_INPUT_INVALID
    )


def test_prior_shadow_source_cannot_be_promoted_to_active() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    shadow = _state(
        safety,
        tracker_id="shadow-owner",
        track_role=TrackRole.SHADOW,
    )
    recorded = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=None,
        shadow_states=(shadow,),
    )
    promoted = replace(
        shadow,
        state_sequence=2,
        track_role=TrackRole.ACTIVE,
    )

    rejected = step_vq2_guidance(
        recorded.memory,
        safety,
        active_state=promoted,
    )

    with pytest.raises(ValueError, match="multiple active/shadow tracks"):
        validate_relative_gate_state_sequence((shadow, promoted))
    assert rejected.memory == recorded.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_ASSOCIATION_INVALID
    )


def test_prior_active_source_cannot_be_relabelled_shadow() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    active = _state(safety, tracker_id="active-owner")
    recorded = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=active,
    )
    demoted = replace(
        active,
        state_sequence=2,
        track_role=TrackRole.SHADOW,
    )

    rejected = step_vq2_guidance(
        recorded.memory,
        safety,
        active_state=None,
        shadow_states=(demoted,),
    )

    assert rejected.memory == recorded.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SHADOW_INPUT_INVALID
    )


def test_prior_shadow_source_cannot_change_tracker_owner() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    shadow = _state(
        safety,
        tracker_id="shadow-owner-a",
        track_role=TrackRole.SHADOW,
    )
    recorded = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=None,
        shadow_states=(shadow,),
    )
    transferred = replace(
        shadow,
        state_sequence=2,
        tracker_id="shadow-owner-b",
    )

    rejected = step_vq2_guidance(
        recorded.memory,
        safety,
        active_state=None,
        shadow_states=(transferred,),
    )

    assert rejected.memory == recorded.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SHADOW_INPUT_INVALID
    )


def test_shadow_tracker_role_cannot_change_on_a_new_source() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    shadow = _state(
        safety,
        frame_id=101,
        candidate_id="shadow-source-a",
        tracker_id="shadow-owner",
        track_role=TrackRole.SHADOW,
    )
    recorded = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=None,
        shadow_states=(shadow,),
    )
    promoted = _state(
        safety,
        state_sequence=2,
        measurement_update_sequence=2,
        frame_id=102,
        candidate_id="shadow-source-b",
        tracker_id="shadow-owner",
        track_role=TrackRole.ACTIVE,
        publication_offset=2,
    )

    rejected = step_vq2_guidance(
        recorded.memory,
        safety,
        active_state=promoted,
    )

    assert rejected.memory == recorded.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_ASSOCIATION_INVALID
    )


def test_shadow_prediction_chronology_advances_but_duplicate_and_revisit_fail() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    source_a = _state(
        safety,
        state_sequence=1,
        measurement_update_sequence=1,
        frame_id=101,
        candidate_id="shadow-a",
        tracker_id="shadow-owner",
        track_role=TrackRole.SHADOW,
    )
    first = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=None,
        shadow_states=(source_a,),
    )
    duplicate = step_vq2_guidance(
        first.memory,
        safety,
        active_state=None,
        shadow_states=(source_a,),
    )
    same_source_prediction = replace(source_a, state_sequence=2)
    predicted = step_vq2_guidance(
        first.memory,
        safety,
        active_state=None,
        shadow_states=(same_source_prediction,),
    )
    source_b = _state(
        safety,
        state_sequence=3,
        measurement_update_sequence=2,
        frame_id=102,
        candidate_id="shadow-b",
        tracker_id="shadow-owner",
        track_role=TrackRole.SHADOW,
        publication_offset=2,
    )
    advanced = step_vq2_guidance(
        predicted.memory,
        safety,
        active_state=None,
        shadow_states=(source_b,),
    )
    revisit = replace(source_a, state_sequence=4)
    revisited = step_vq2_guidance(
        advanced.memory,
        safety,
        active_state=None,
        shadow_states=(revisit,),
    )

    validate_relative_gate_state_sequence(
        (source_a, same_source_prediction, source_b)
    )
    assert duplicate.memory == first.memory
    assert predicted.memory != first.memory
    assert advanced.memory != predicted.memory
    assert revisited.memory == advanced.memory
    for rejected in (duplicate, revisited):
        assert (
            rejected.decision.withholding_reason
            is VQ2GuidanceWithholdingReason.SHADOW_INPUT_INVALID
        )


def test_invalid_visual_batch_keeps_valid_safety_progress_only() -> None:
    transition, safety, next_sequence = _ready_phase(VQ2GuidancePhase.ALIGN)
    first_active = _state(safety, state_sequence=1)
    recorded = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=first_active,
    )
    approach_safety = _safety(
        _authority(next_sequence),
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    stale_decision_ns = approach_safety.evaluation_monotonic_ns - 100_000_001
    stale_shadow = _state(
        approach_safety,
        frame_id=101,
        candidate_id="stale-shadow",
        tracker_id="stale-shadow",
        track_role=TrackRole.SHADOW,
        publish_monotonic_ns=stale_decision_ns - 2,
        measurement_monotonic_ns=stale_decision_ns - 3,
        decision_monotonic_ns=stale_decision_ns,
        prediction_monotonic_ns=stale_decision_ns,
    )
    concurrent_active = _state(
        approach_safety,
        state_sequence=2,
        measurement_update_sequence=2,
        frame_id=2,
        publication_offset=2,
    )

    rejected = step_vq2_guidance(
        recorded.memory,
        approach_safety,
        active_state=concurrent_active,
        shadow_states=(stale_shadow,),
    )

    assert rejected.memory.safety == approach_safety
    assert rejected.memory.active_source == recorded.memory.active_source
    assert rejected.memory.track_histories == recorded.memory.track_histories
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SHADOW_INPUT_INVALID
    )


def test_active_tracker_cannot_change_inside_gate_epoch() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    first = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(safety, state_sequence=1),
    )
    later_safety = replace(
        safety,
        evaluation_monotonic_ns=safety.evaluation_monotonic_ns + 1_000_000,
    )
    changed = step_vq2_guidance(
        first.memory,
        later_safety,
        active_state=_state(
            later_safety,
            state_sequence=2,
            tracker_id="replacement",
            publication_offset=2,
        ),
    )

    assert changed.memory.active_source == first.memory.active_source
    assert (
        changed.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_TRACK_CHANGED
    )


def test_repeated_and_revisited_sources_are_stale_and_preserve_memory() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    source_a = _state(
        safety,
        state_sequence=1,
        frame_id=1,
        candidate_id="a",
    )
    first = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=source_a,
    )
    repeated = step_vq2_guidance(first.memory, safety, active_state=source_a)
    later_safety = replace(
        safety,
        evaluation_monotonic_ns=safety.evaluation_monotonic_ns + 2_000_000,
    )
    source_b = _state(
        later_safety,
        state_sequence=2,
        frame_id=2,
        candidate_id="b",
        publication_offset=2,
    )
    second = step_vq2_guidance(
        first.memory,
        later_safety,
        active_state=source_b,
    )
    newest_safety = replace(
        safety,
        evaluation_monotonic_ns=safety.evaluation_monotonic_ns + 4_000_000,
    )
    revisit = _state(
        newest_safety,
        state_sequence=3,
        measurement_update_sequence=1,
        frame_id=1,
        candidate_id="a",
        publication_offset=3,
    )
    revisited = step_vq2_guidance(
        second.memory,
        newest_safety,
        active_state=revisit,
    )

    assert repeated.memory == first.memory
    assert revisited.memory.active_source == second.memory.active_source
    assert (
        repeated.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_STATE_STALE
    )
    assert (
        revisited.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_STATE_STALE
    )


@pytest.mark.parametrize("future_field", ["decision", "publication"])
def test_future_timing_is_withheld_and_preserves_memory(future_field) -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    kwargs = {}
    if future_field == "decision":
        kwargs["decision_monotonic_ns"] = safety.evaluation_monotonic_ns + 1
        kwargs["prediction_monotonic_ns"] = safety.evaluation_monotonic_ns + 1
    else:
        kwargs["publish_monotonic_ns"] = safety.evaluation_monotonic_ns + 1
        kwargs["measurement_monotonic_ns"] = safety.evaluation_monotonic_ns
        kwargs["decision_monotonic_ns"] = safety.evaluation_monotonic_ns + 2
        kwargs["prediction_monotonic_ns"] = safety.evaluation_monotonic_ns + 2
    rejected = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(safety, **kwargs),
    )

    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_TIMING_FUTURE
    )


def test_decision_age_hard_boundary_and_plus_one_ns() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    boundary_decision = safety.evaluation_monotonic_ns - 100_000_000
    accepted = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            publish_monotonic_ns=boundary_decision - 1,
            measurement_monotonic_ns=boundary_decision - 2,
            decision_monotonic_ns=boundary_decision,
            prediction_monotonic_ns=boundary_decision,
        ),
    )
    rejected = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            publish_monotonic_ns=boundary_decision - 2,
            measurement_monotonic_ns=boundary_decision - 3,
            decision_monotonic_ns=boundary_decision - 1,
            prediction_monotonic_ns=boundary_decision - 1,
        ),
    )

    assert accepted.decision.objective_permitted
    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_TIMING_STALE
    )


def test_measurement_age_including_uncertainty_boundary_and_plus_one_ns() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    uncertainty = 1_000_000
    boundary_measurement = (
        safety.evaluation_monotonic_ns - 150_000_000 + uncertainty
    )
    accepted = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            publish_monotonic_ns=safety.evaluation_monotonic_ns - 20_000_000,
            measurement_monotonic_ns=boundary_measurement,
            measurement_uncertainty_ns=uncertainty,
        ),
    )
    rejected = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            publish_monotonic_ns=safety.evaluation_monotonic_ns - 20_000_000,
            measurement_monotonic_ns=boundary_measurement - 1,
            measurement_uncertainty_ns=uncertainty,
        ),
    )

    assert accepted.decision.objective_permitted
    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_TIMING_STALE
    )


def test_measurement_uncertainty_boundary_and_plus_one_ns() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    measurement_ns = safety.evaluation_monotonic_ns - 50_000_000
    accepted = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            measurement_monotonic_ns=measurement_ns,
            measurement_uncertainty_ns=50_000_000,
        ),
    )
    rejected = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            measurement_monotonic_ns=measurement_ns,
            measurement_uncertainty_ns=50_000_001,
        ),
    )

    assert accepted.decision.objective_permitted
    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_TIMING_UNCERTAIN
    )


def test_prediction_lead_including_uncertainty_boundary_and_plus_one_ns() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    evaluation_ns = safety.evaluation_monotonic_ns
    accepted = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            prediction_monotonic_ns=evaluation_ns + 99_000_000,
            delay_uncertainty_ns=1_000_000,
        ),
    )
    rejected = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            prediction_monotonic_ns=evaluation_ns + 99_000_001,
            delay_uncertainty_ns=1_000_000,
        ),
    )

    assert accepted.decision.objective_permitted
    assert rejected.memory == transition.memory
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_PREDICTION_HORIZON
    )


def test_high_uncertainty_top_clipped_gate1_align_is_withheld() -> None:
    transition, safety, _ = _ready_gate1_align()
    result = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            tracker_id="active-gate-1",
            bearing_norm=(0.45, -0.35),
            variance=(0.09, 0.09, 0.01, 0.01, 0.01, 0.01),
            clipping=FrameEdge.TOP,
            health=RelativeStateHealth.DEGRADED,
        ),
    )

    assert not result.decision.objective_permitted
    assert not result.decision.corridor_eligible
    assert (
        result.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.OUTSIDE_UNCERTAINTY_CORRIDOR
    )


def test_lower_uncertainty_top_clipped_gate1_can_only_recenter() -> None:
    transition, safety, _ = _ready_gate1_align()
    result = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=_state(
            safety,
            tracker_id="active-gate-1",
            bearing_norm=(0.45, -0.35),
            variance=(0.0004, 0.0004, 0.01, 0.01, 0.01, 0.01),
            clipping=FrameEdge.TOP,
            health=RelativeStateHealth.DEGRADED,
        ),
    )

    assert result.decision.objective_permitted
    assert result.decision.objective_kind is VQ2GuidanceObjectiveKind.RECENTER_ACTIVE_GATE
    assert result.decision.source is not None
    assert result.decision.source.tracker_id == "active-gate-1"


def test_degraded_unclipped_align_is_recorded_but_motion_withheld() -> None:
    result = _with_active(
        VQ2GuidancePhase.ALIGN,
        health=RelativeStateHealth.DEGRADED,
        clipping=FrameEdge.NONE,
    )

    assert not result.decision.objective_permitted
    assert (
        result.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_STATE_HEALTH
    )
    assert result.decision.source is not None
    assert result.memory.active_source == result.decision.source
    assert len(result.memory.track_histories) == 1


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {"bearing_norm": (0.3, 0.0)},
            VQ2GuidanceWithholdingReason.OUTSIDE_UNCERTAINTY_CORRIDOR,
        ),
        (
            {"bearing_rate_norm_s": (0.5, 0.0)},
            VQ2GuidanceWithholdingReason.BEARING_RATE_UNCERTAIN,
        ),
        (
            {"health": RelativeStateHealth.DEGRADED},
            VQ2GuidanceWithholdingReason.ACTIVE_STATE_HEALTH,
        ),
    ],
)
def test_approach_requires_healthy_tight_corridor_and_rate(changes, reason) -> None:
    result = _with_active(VQ2GuidancePhase.APPROACH, **changes)

    assert not result.decision.objective_permitted
    assert result.decision.withholding_reason is reason


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {"clipping": FrameEdge.TOP},
            VQ2GuidanceWithholdingReason.COMMIT_REQUIRES_UNCLIPPED,
        ),
        (
            {"log_scale": -1.2},
            VQ2GuidanceWithholdingReason.COMMIT_SCALE_UNCERTAIN,
        ),
        (
            {"expansion_rate_s": 0.01},
            VQ2GuidanceWithholdingReason.COMMIT_EXPANSION_UNCERTAIN,
        ),
    ],
)
def test_commit_requires_unclipped_scale_and_expansion(changes, reason) -> None:
    result = _with_active(VQ2GuidancePhase.COMMIT, **changes)

    assert not result.decision.objective_permitted
    assert result.decision.withholding_reason is reason


def test_commit_can_be_guidance_eligible_but_has_no_command_or_credit() -> None:
    result = _with_active(VQ2GuidancePhase.COMMIT)

    assert result.decision.objective_permitted
    assert result.decision.objective_kind is VQ2GuidanceObjectiveKind.COMMIT_ACTIVE_GATE
    assert result.decision.authority.expected_gate_index == 0
    assert not hasattr(result.decision, "passage_confirmed")
    assert not hasattr(result.decision, "requested_thrust")


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {
                "health": RelativeStateHealth.COASTING,
                "dropout_count": 1,
                "innovation_accepted": None,
            },
            VQ2GuidanceWithholdingReason.ACTIVE_STATE_DROPOUT,
        ),
        (
            {
                "health": RelativeStateHealth.DEGRADED,
                "innovation_accepted": False,
            },
            VQ2GuidanceWithholdingReason.ACTIVE_INNOVATION_REJECTED,
        ),
    ],
)
def test_dropout_and_rejected_innovation_withhold(changes, reason) -> None:
    result = _with_active(VQ2GuidancePhase.ALIGN, **changes)

    assert not result.decision.objective_permitted
    assert result.decision.withholding_reason is reason


def test_public_guidance_rejects_even_an_exact_first_dropout_profile() -> None:
    memory, safety, coast = _guidance_memory_with_first_dropout(
        VQ2GuidancePhase.APPROACH
    )

    rejected = step_vq2_guidance(
        memory,
        safety,
        active_state=coast,
    )

    assert not rejected.decision.objective_permitted
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_STATE_DROPOUT
    )


def test_private_first_dropout_guidance_rejects_an_unowned_capability() -> None:
    memory, safety, coast = _guidance_memory_with_first_dropout(
        VQ2GuidancePhase.APPROACH
    )

    with pytest.raises(TypeError, match="invalid first-dropout guidance capability"):
        _step_vq2_guidance_first_observation_dropout(
            memory,
            safety,
            active_state=coast,
            capability=object(),
        )


def test_guidance_impl_rejects_a_raw_first_dropout_boolean() -> None:
    memory, safety, coast = _guidance_memory_with_first_dropout(
        VQ2GuidancePhase.APPROACH
    )

    with pytest.raises(TypeError, match="invalid internal first-dropout capability"):
        _step_vq2_guidance_impl(
            memory,
            safety,
            active_state=coast,
            shadow_states=(),
            config=None,
            first_dropout_capability=True,
        )


@pytest.mark.parametrize(
    "phase",
    (VQ2GuidancePhase.APPROACH, VQ2GuidancePhase.ALIGN),
    ids=("gate0", "gate1"),
)
def test_private_first_dropout_guidance_permits_only_the_owned_motion_phases(
    phase: VQ2GuidancePhase,
) -> None:
    memory, safety, coast = _guidance_memory_with_first_dropout(phase)

    accepted = _step_vq2_guidance_first_observation_dropout(
        memory,
        safety,
        active_state=coast,
        capability=_VQ2_GUIDANCE_FIRST_DROPOUT_CAPABILITY,
    )

    assert accepted.decision.objective_permitted
    assert accepted.decision.withholding_reason is None
    assert accepted.decision.source is not None
    assert accepted.decision.source.state_sequence == coast.state_sequence
    assert (
        accepted.decision.source.measurement_update_sequence
        == coast.measurement_update_sequence
    )


def test_private_first_dropout_guidance_rejects_gate0_align() -> None:
    transition, safety, _next_sequence = _ready_phase(VQ2GuidancePhase.ALIGN)
    prior = _state(safety, tracker_id="active-gate-0")
    accepted = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=prior,
    )
    assert accepted.decision.objective_permitted

    rejected = _step_vq2_guidance_first_observation_dropout(
        accepted.memory,
        safety,
        active_state=_first_dropout_from(prior),
        capability=_VQ2_GUIDANCE_FIRST_DROPOUT_CAPABILITY,
    )

    assert not rejected.decision.objective_permitted
    assert (
        rejected.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.ACTIVE_STATE_DROPOUT
    )


def test_race_finish_and_abort_are_forward_terminal_holds() -> None:
    transition, safety, next_sequence = _ready_phase(VQ2GuidancePhase.ALIGN)
    phase_start_ns = transition.memory.safety.phase_started_monotonic_ns
    for race_state in (
        VQ2GuidanceRaceState.FINISHED,
        VQ2GuidanceRaceState.ABORTED,
    ):
        terminal_safety = _safety(
            _authority(next_sequence),
            VQ2GuidancePhase.ALIGN,
            race_state,
            phase_started_monotonic_ns=phase_start_ns,
        )
        terminal = step_vq2_guidance(
            transition.memory,
            terminal_safety,
            active_state=None,
        )
        assert not terminal.decision.objective_permitted
        assert terminal.decision.objective_kind is VQ2GuidanceObjectiveKind.HOLD
        assert (
            terminal.decision.withholding_reason
            is VQ2GuidanceWithholdingReason.RACE_TERMINAL
        )
        assert terminal.decision.phase_started_monotonic_ns == phase_start_ns


def test_source_correlation_and_evaluation_time_are_exactly_echoed() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    state = _state(safety, state_sequence=17)
    result = step_vq2_guidance(
        transition.memory,
        safety,
        active_state=state,
    )
    source = result.decision.source

    assert source is not None
    assert result.decision.authority == safety.authority
    assert result.decision.evaluation_host_clock_id == safety.evaluation_host_clock_id
    assert result.decision.evaluation_monotonic_ns == safety.evaluation_monotonic_ns
    assert (
        result.decision.phase_started_monotonic_ns
        == safety.phase_started_monotonic_ns
    )
    assert source.decision_time_monotonic_ns == state.timing.decision_time_monotonic_ns
    assert source.prediction_time_monotonic_ns == state.timing.prediction_time_monotonic_ns
    assert source.source_frame == state.timing.source_frame
    assert (
        source.source_frame_publication_sequence
        == state.timing.source_frame_publication_sequence
    )
    assert source.tracker_id == state.tracker_id
    assert source.state_sequence == state.state_sequence
    assert source.measurement_update_sequence == state.measurement_update_sequence
    assert source.source_candidate_id == state.source_candidate_id


def test_transition_is_deterministic_for_identical_inputs() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    state = _state(safety)

    assert step_vq2_guidance(
        transition.memory,
        safety,
        active_state=state,
    ) == step_vq2_guidance(
        transition.memory,
        safety,
        active_state=state,
    )


def test_exact_container_and_config_types_are_required() -> None:
    transition, safety, _ = _ready_phase(VQ2GuidancePhase.ALIGN)
    shadow = _state(
        safety,
        track_role=TrackRole.SHADOW,
        tracker_id="shadow",
    )
    with pytest.raises(TypeError, match="shadow_states must be an exact tuple"):
        step_vq2_guidance(  # type: ignore[arg-type]
            transition.memory,
            safety,
            active_state=None,
            shadow_states=[shadow],
        )
    with pytest.raises(TypeError, match="target_bearing_norm must be an exact 2-tuple"):
        VQ2GuidanceConfig(target_bearing_norm=[0.0, 0.0])  # type: ignore[arg-type]


def test_generated_guidance_scenario_passes_narrow_checks() -> None:
    report = evaluate_synthetic_vq2_guidance_scenario()

    assert report.all_checks_passed
    assert report.fresh_nonzero_gate_initialization_rejected
    assert report.gate0_countdown_phase_change_rejected
    assert report.gate0_visual_phase_non_regression
    assert report.gate0_same_snapshot_phase_change_rejected
    assert report.gate0_phase_start_stable
    assert report.gate0_phase_start_renewal_rejected
    assert report.gate0_forward_phase_accepted
    assert report.gate1_shadow_isolated
    assert report.gate1_shadow_promotion_rejected
    assert report.gate1_high_uncertainty_withheld
    assert report.gate1_low_uncertainty_recenter_permitted


def test_generated_guidance_scenario_has_explicit_nonflight_scope() -> None:
    report = evaluate_synthetic_vq2_guidance_scenario()

    assert report.evidence_scope == SYNTHETIC_GUIDANCE_SCOPE
    assert "nonpowered" in report.evidence_scope
    assert "not_replay" in report.evidence_scope
    assert "not_simulator" in report.evidence_scope
    assert "not_passage_evidence" in report.evidence_scope
    assert all(not hasattr(step, "requested_thrust") for step in report.steps)


def test_generated_guidance_scenario_is_deterministic_and_digest_bound() -> None:
    first = evaluate_synthetic_vq2_guidance_scenario()
    second = evaluate_synthetic_vq2_guidance_scenario()

    assert first == second
    assert first.digest_sha256 == _EXPECTED_SCENARIO_DIGEST
    assert len(first.steps) == 16
    assert first.steps[2].phase == "acquire"
    assert not first.steps[2].objective_permitted
    assert first.steps[-2].withholding_reason == "outside_uncertainty_corridor"
    assert first.steps[-1].objective_kind == "recenter_active_gate"
    assert first.steps[-1].objective_permitted
