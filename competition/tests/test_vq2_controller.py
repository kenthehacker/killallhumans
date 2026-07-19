from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from competition.vq2_contracts import (
    CommandProposalV1,
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
    validate_command_proposal_source,
)
from competition.vq2_controller import (
    ControllerAttitudeInput,
    ControllerInputError,
    ControllerPhaseInput,
    ControllerTickInput,
    PredictiveControllerConfig,
    VQ2ControlPhase,
    propose_vq2_command,
)


_HOST = "host-monotonic-1"
_DECISION_NS = 2_000_000_000
_PROPOSAL_NS = 2_010_000_000
_DEADLINE_NS = 2_030_000_000
_STATE_FEATURE_ORDER = (
    "bearing_x_norm",
    "bearing_y_norm",
    "log_scale",
    "bearing_rate_x_norm_s",
    "bearing_rate_y_norm_s",
    "expansion_rate_s",
)
_METRIC_FEATURE_ORDER = (
    "position_x_body_frd_m",
    "position_y_body_frd_m",
    "position_z_body_frd_m",
    "velocity_x_body_frd_m_s",
    "velocity_y_body_frd_m_s",
    "velocity_z_body_frd_m_s",
    "orientation_error_x_rad",
    "orientation_error_y_rad",
    "orientation_error_z_rad",
)


def _authority(*, gate_index: int = 0, host_clock_id: str = _HOST):
    return GateAuthorityEpochV1(
        session_id="training-session-1",
        reset_epoch=7,
        gate_epoch=gate_index,
        expected_gate_index=gate_index,
        race_status_sequence=20 + gate_index,
        race_status_boot_ms=1_250 + 250 * gate_index,
        camera_host_clock_id=host_clock_id,
        camera_stream_id="camera0",
        camera_generation=3,
        frame_publication_sequence_not_before=8 + gate_index,
        frame_publish_monotonic_ns_not_before=1_900_000_000 + gate_index,
    )


def _covariance(
    diagonal: tuple[float, float, float, float, float, float] = (
        0.01,
        0.01,
        0.04,
        0.25,
        0.25,
        0.25,
    ),
) -> FeatureCovarianceV1:
    return FeatureCovarianceV1(
        model_id="controller-test-relative-state-v1",
        feature_order=_STATE_FEATURE_ORDER,
        matrix=tuple(
            tuple(diagonal[row] if row == column else 0.0 for column in range(6))
            for row in range(6)
        ),
    )


def _timing(
    *,
    decision_ns: int = _DECISION_NS,
    measurement_ns: int | None = None,
    prediction_ns: int | None = None,
    measurement_uncertainty_ns: int = 2_000_000,
) -> PredictionTimeV1:
    measurement = decision_ns - 70_000_000 if measurement_ns is None else measurement_ns
    prediction = decision_ns if prediction_ns is None else prediction_ns
    publication = max(1_990_000_000, measurement)
    estimated = prediction != decision_ns
    return PredictionTimeV1(
        host_clock_id=_HOST,
        source_frame=FrameIdentityV1("camera0", 3, 41),
        source_frame_publication_sequence=9,
        source_frame_publish_monotonic_ns=publication,
        measurement_time_monotonic_ns=measurement,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        measurement_time_model_id=None,
        measurement_uncertainty_ns=measurement_uncertainty_ns,
        decision_time_monotonic_ns=decision_ns,
        prediction_time_monotonic_ns=prediction,
        prediction_basis=(
            PredictionBasis.COMMAND_SEND_ESTIMATE
            if estimated
            else PredictionBasis.DECISION_TIME
        ),
        delay_model_id="controller-test-delay-v1" if estimated else None,
        delay_uncertainty_ns=1_000_000 if estimated else 0,
    )


def _state(
    *,
    gate_index: int = 0,
    timing: PredictionTimeV1 | None = None,
    bearing: tuple[float, float] = (0.0, 0.0),
    bearing_rate: tuple[float, float] = (0.0, 0.0),
    covariance: FeatureCovarianceV1 | None = None,
    state_sequence: int = 12,
    role: TrackRole = TrackRole.ACTIVE,
    health: RelativeStateHealth = RelativeStateHealth.HEALTHY,
) -> RelativeGateStateV1:
    dropout = 1 if health in {RelativeStateHealth.COASTING, RelativeStateHealth.LOST} else 0
    reason = (
        None
        if health in {RelativeStateHealth.HEALTHY, RelativeStateHealth.INITIALIZING}
        else f"test_{health.value}"
    )
    return RelativeGateStateV1(
        timing=timing or _timing(),
        authority=_authority(gate_index=gate_index),
        tracker_id=f"active-gate-{gate_index}",
        state_sequence=state_sequence,
        measurement_update_sequence=5,
        source_candidate_id=f"gate-{gate_index}-candidate-0",
        track_role=role,
        bearing_norm=bearing,
        bearing_rate_norm_s=bearing_rate,
        log_scale=-1.2,
        expansion_rate_s=0.3,
        covariance=covariance or _covariance(),
        metric_position_body_frd_m=None,
        metric_velocity_body_frd_m_s=None,
        metric_gate_orientation_body_frd_xyzw=None,
        metric_covariance=None,
        last_clipping=FrameEdge.NONE,
        outer_visibility=(
            FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT | FrameEdge.BOTTOM
        ),
        inner_visibility=FrameEdge.NONE,
        normalized_innovation_squared=None,
        innovation_gate_threshold=None,
        innovation_accepted=None,
        dropout_count=dropout,
        health=health,
        health_reason=reason,
    )


def _attitude(
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    body_rates: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ControllerAttitudeInput:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return ControllerAttitudeInput(
        orientation_body_to_world_wxyz=(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        body_rates_rad_s=body_rates,
    )


def _tick(
    state: RelativeGateStateV1,
    *,
    proposal_ns: int = _PROPOSAL_NS,
    deadline_ns: int = _DEADLINE_NS,
    host_clock_id: str = _HOST,
    authority: GateAuthorityEpochV1 | None = None,
    minimum_decision_ns: int | None = None,
    minimum_state_sequence: int | None = None,
) -> ControllerTickInput:
    return ControllerTickInput(
        proposal_id=72,
        control_tick_id=71,
        host_clock_id=host_clock_id,
        proposal_monotonic_ns=proposal_ns,
        control_tick_deadline_monotonic_ns=deadline_ns,
        minimum_state_decision_monotonic_ns=(
            state.timing.decision_time_monotonic_ns
            if minimum_decision_ns is None
            else minimum_decision_ns
        ),
        minimum_state_sequence=(
            state.state_sequence
            if minimum_state_sequence is None
            else minimum_state_sequence
        ),
        expected_authority=authority or state.authority,
    )


def _phase(
    *,
    mode: VQ2ControlPhase = VQ2ControlPhase.GATE0_APPROACH,
    elapsed_s: float = 0.50,
    initial_pitch_rad: float = 0.0,
    target_bearing: tuple[float, float] = (0.0, 0.0),
    permitted: bool = True,
) -> ControllerPhaseInput:
    return ControllerPhaseInput(
        mode=mode,
        elapsed_s=elapsed_s,
        initial_pitch_rad=initial_pitch_rad,
        target_bearing_norm=target_bearing,
        objective_permitted=permitted,
        withholding_reason=None if permitted else "guidance_withheld",
    )


def _propose(
    state: RelativeGateStateV1,
    *,
    tick: ControllerTickInput | None = None,
    phase: ControllerPhaseInput | None = None,
    attitude: ControllerAttitudeInput | None = None,
    config: PredictiveControllerConfig | None = None,
) -> CommandProposalV1:
    return propose_vq2_command(
        state,
        tick=tick or _tick(state),
        phase=phase or _phase(),
        attitude=attitude or _attitude(),
        config=config or PredictiveControllerConfig(),
    )


def _legacy_vertical_thrust(control_y: float, control_y_rate: float) -> float:
    proportional = 0.040 * max(-1.0, min(1.0, (180.0 - control_y) / 90.0))
    damping = -0.00070 * max(-300.0, min(300.0, control_y_rate))
    return max(0.21, min(0.32, 0.275 + proportional + damping))


@pytest.mark.parametrize(
    ("elapsed_s", "expected_thrust"),
    (
        (0.0, 0.26),
        (math.nextafter(0.15, 0.0), 0.26),
        (0.15, 0.32),
        (math.nextafter(0.45, 0.0), 0.32),
        (0.45, 0.275),
    ),
)
def test_gate0_elapsed_boundaries_match_legacy_thrust_schedule(
    elapsed_s: float, expected_thrust: float
):
    proposal = _propose(_state(), phase=_phase(elapsed_s=elapsed_s))
    assert proposal.requested_thrust == pytest.approx(expected_thrust, abs=1e-15)
    assert proposal.reason == "legacy_gate0_pixel_pd"


@pytest.mark.parametrize(
    ("bearing_y", "bearing_rate_y"),
    ((0.0, 0.0), (-1.0 / 6.0, 1.0 / 3.0), (-1.0, -4.0), (1.0, 4.0)),
)
def test_gate0_normalized_vertical_pd_matches_legacy_pixel_fixtures(
    bearing_y: float, bearing_rate_y: float
):
    state = _state(bearing=(0.0, bearing_y), bearing_rate=(0.0, bearing_rate_y))
    proposal = _propose(state)
    expected = _legacy_vertical_thrust(
        180.0 * (1.0 + bearing_y),
        180.0 * bearing_rate_y,
    )
    assert proposal.requested_thrust == pytest.approx(expected, abs=1e-15)
    assert proposal.saturation.thrust is (expected in {0.21, 0.32})


def test_gate0_attitude_rate_fixture_matches_legacy_quaternion_pd():
    state = _state(bearing=(0.25, 0.0))
    proposal = _propose(state, phase=_phase(elapsed_s=0.8))
    expected_roll_rate = 2.0 * math.sin((0.15 * 0.25) / 2.0)
    assert proposal.requested_body_rates_rad_s == pytest.approx(
        (expected_roll_rate, 0.0, 0.0), abs=1e-15
    )
    assert proposal.requested_thrust == 0.275


@pytest.mark.parametrize(
    ("elapsed_s", "expected_pitch"),
    ((0.0, -0.1), (0.4, -0.05), (0.8, 0.0), (1.2, 0.0)),
)
def test_gate0_pitch_blend_matches_legacy_boundary_fixtures(
    elapsed_s: float, expected_pitch: float
):
    proposal = _propose(
        _state(),
        phase=_phase(elapsed_s=elapsed_s, initial_pitch_rad=-0.1),
    )
    expected_pitch_rate = math.sin(expected_pitch / 2.0)
    assert proposal.requested_body_rates_rad_s[1] == pytest.approx(
        expected_pitch_rate, abs=1e-15
    )


def test_proposal_is_deterministic_exact_and_source_bound():
    state = _state(bearing=(0.2, -0.1), bearing_rate=(0.05, -0.2))
    first = _propose(state)
    second = _propose(state)
    assert type(first) is CommandProposalV1
    assert first == second
    assert first.to_primitive() == second.to_primitive()
    validate_command_proposal_source(first, state)
    assert first.source_frame == state.timing.source_frame
    assert first.source_state_sequence == state.state_sequence
    assert first.source_measurement_update_sequence == state.measurement_update_sequence


def test_yaw_is_exact_zero_even_with_nonzero_yaw_and_body_yaw_rate():
    proposal = _propose(
        _state(bearing=(0.3, -0.2)),
        attitude=_attitude(yaw=1.2, body_rates=(0.0, 0.0, 9.0)),
    )
    assert proposal.requested_body_rates_rad_s[2] == 0.0
    assert math.copysign(1.0, proposal.requested_body_rates_rad_s[2]) == 1.0
    assert proposal.saturation.body_rate_axes[2] is False


def test_rate_and_thrust_saturation_diagnostics_are_explicit():
    state = _state(bearing=(0.0, -1.0), bearing_rate=(0.0, -4.0))
    proposal = _propose(
        state,
        attitude=_attitude(body_rates=(10.0, -10.0, 0.0)),
    )
    assert proposal.requested_body_rates_rad_s == (-0.25, 0.25, 0.0)
    assert proposal.requested_thrust == 0.32
    assert proposal.saturation.body_rate_axes == (True, True, False)
    assert proposal.saturation.thrust is True


@pytest.mark.parametrize(
    "state",
    (
        _state(role=TrackRole.SHADOW),
        _state(health=RelativeStateHealth.INITIALIZING),
        _state(health=RelativeStateHealth.DEGRADED),
        _state(health=RelativeStateHealth.COASTING),
        _state(health=RelativeStateHealth.UNHEALTHY),
        _state(health=RelativeStateHealth.LOST),
    ),
)
def test_shadow_and_nonhealthy_states_are_withheld_source_less(state):
    proposal = _propose(state)
    assert proposal.is_exact_zero
    assert proposal.source_frame is None
    assert proposal.source_tracker_id is None
    assert proposal.reason.startswith("withheld:")


def test_authority_host_phase_and_objective_mismatches_fail_closed():
    state = _state()
    host_mismatch = _propose(state, tick=_tick(state, host_clock_id="other-clock"))
    authority_mismatch = _propose(
        state,
        tick=_tick(state, authority=_authority(gate_index=1)),
    )
    phase_mismatch = _propose(
        state,
        phase=_phase(mode=VQ2ControlPhase.GATE1_RECENTER),
    )
    objective_withheld = _propose(state, phase=_phase(permitted=False))
    for proposal in (
        host_mismatch,
        authority_mismatch,
        phase_mismatch,
        objective_withheld,
    ):
        assert proposal.is_exact_zero
        assert proposal.source_frame is None
    assert "authority" in authority_mismatch.reason
    assert "objective_withheld" in objective_withheld.reason


def test_decision_age_future_and_both_regression_watermarks_fail_closed():
    config = PredictiveControllerConfig()
    boundary_timing = _timing(measurement_ns=_DECISION_NS)
    state = _state(timing=boundary_timing)
    boundary_tick = _tick(
        state,
        proposal_ns=_DECISION_NS + config.max_state_age_ns,
        deadline_ns=_DECISION_NS + config.max_state_age_ns + 20_000_000,
    )
    assert not _propose(state, tick=boundary_tick).is_exact_zero

    stale_tick = dataclasses.replace(
        boundary_tick,
        proposal_monotonic_ns=boundary_tick.proposal_monotonic_ns + 1,
    )
    assert "decision_stale" in _propose(state, tick=stale_tick).reason

    future_tick = _tick(
        state,
        proposal_ns=_DECISION_NS - 1,
        deadline_ns=_DECISION_NS + 1,
        minimum_decision_ns=0,
    )
    assert "decision_from_future" in _propose(state, tick=future_tick).reason

    decision_regression_tick = _tick(
        state,
        minimum_decision_ns=_DECISION_NS + 1,
    )
    assert "decision_regressed" in _propose(
        state, tick=decision_regression_tick
    ).reason

    sequence_regression_tick = _tick(
        state,
        minimum_state_sequence=state.state_sequence + 1,
    )
    assert "sequence_regressed" in _propose(
        state, tick=sequence_regression_tick
    ).reason


def test_measurement_age_and_prediction_lead_boundaries_are_inclusive():
    config = PredictiveControllerConfig()
    measurement_boundary = _PROPOSAL_NS - config.max_measurement_age_ns
    state = _state(timing=_timing(measurement_ns=measurement_boundary))
    assert not _propose(state).is_exact_zero
    stale = _state(timing=_timing(measurement_ns=measurement_boundary - 1))
    assert "measurement_stale" in _propose(stale).reason

    prediction_boundary = _PROPOSAL_NS + config.max_prediction_lead_ns
    predicted = _state(timing=_timing(prediction_ns=prediction_boundary))
    assert not _propose(predicted).is_exact_zero
    too_far = _state(timing=_timing(prediction_ns=prediction_boundary + 1))
    assert "prediction_too_far_ahead" in _propose(too_far).reason


def test_measurement_and_covariance_uncertainty_gates_are_inclusive():
    config = PredictiveControllerConfig()
    exact = _state(
        timing=_timing(
            measurement_uncertainty_ns=config.max_measurement_uncertainty_ns
        )
    )
    assert not _propose(exact).is_exact_zero
    too_uncertain = _state(
        timing=_timing(
            measurement_uncertainty_ns=config.max_measurement_uncertainty_ns + 1
        )
    )
    rejected = _propose(too_uncertain)
    assert rejected.is_exact_zero
    assert rejected.uncertainty.limited is True

    limits = (0.25, 0.25, 1.0, 16.0, 16.0, 16.0)
    assert not _propose(_state(covariance=_covariance(limits))).is_exact_zero
    for index, limit in enumerate(limits):
        diagonal = list(_covariance().matrix[row][row] for row in range(6))
        diagonal[index] = limit + 1e-6
        proposal = _propose(_state(covariance=_covariance(tuple(diagonal))))
        assert proposal.is_exact_zero
        assert proposal.uncertainty.reason == "relative_state_covariance"


def test_bearing_and_rate_envelopes_fail_closed():
    config = PredictiveControllerConfig()
    bearing = _state(bearing=(config.max_abs_bearing_error_norm + 1e-6, 0.0))
    rate = _state(bearing_rate=(config.max_abs_bearing_rate_norm_s + 1e-6, 0.0))
    assert "bearing_error" in _propose(bearing).reason
    assert "bearing_rate" in _propose(rate).reason


def test_metric_pose_and_uncontrolled_scale_values_do_not_affect_output():
    state = _state(bearing=(0.2, -0.3), bearing_rate=(0.1, -0.2))
    metric_covariance = FeatureCovarianceV1(
        model_id="poisonous-metric-test-v1",
        feature_order=_METRIC_FEATURE_ORDER,
        matrix=tuple(
            tuple(999.0 if row == column else 0.0 for column in range(9))
            for row in range(9)
        ),
    )
    with_metric = dataclasses.replace(
        state,
        log_scale=3.0,
        expansion_rate_s=-7.0,
        metric_position_body_frd_m=(999.0, -999.0, 0.001),
        metric_velocity_body_frd_m_s=(-50.0, 70.0, 90.0),
        metric_gate_orientation_body_frd_xyzw=(0.0, 0.0, 0.0, 1.0),
        metric_covariance=metric_covariance,
    )
    assert _propose(with_metric) == _propose(state)


def test_guidance_target_bearing_is_an_explicit_local_objective():
    state = _state(bearing=(0.3, -0.2))
    proposal = _propose(
        state,
        phase=_phase(target_bearing=(0.3, -0.2)),
    )
    assert proposal.requested_body_rates_rad_s == (0.0, 0.0, 0.0)
    assert proposal.requested_thrust == 0.275


def test_gate1_recenter_uses_tighter_no_forward_progress_envelope():
    state = _state(
        gate_index=1,
        bearing=(0.8, -0.8),
        bearing_rate=(0.5, -0.5),
    )
    proposal = _propose(
        state,
        phase=_phase(
            mode=VQ2ControlPhase.GATE1_RECENTER,
            elapsed_s=0.2,
        ),
    )
    assert not proposal.is_exact_zero
    assert abs(proposal.requested_body_rates_rad_s[0]) <= 0.12
    assert proposal.requested_body_rates_rad_s[1] == 0.0
    assert proposal.requested_body_rates_rad_s[2] == 0.0
    assert 0.21 <= proposal.requested_thrust <= 0.30
    assert proposal.phase == "gate1_recenter"
    assert "pass" not in proposal.reason


def test_gate1_low_uncertainty_degraded_clipped_state_is_explicitly_limited():
    degraded_unclipped = _state(
        gate_index=1,
        bearing=(0.6, -0.5),
        health=RelativeStateHealth.DEGRADED,
    )
    phase = _phase(mode=VQ2ControlPhase.GATE1_RECENTER, elapsed_s=0.2)
    rejected_unclipped = _propose(degraded_unclipped, phase=phase)
    assert rejected_unclipped.is_exact_zero
    assert "degraded_without_clipping" in rejected_unclipped.reason

    degraded = dataclasses.replace(
        degraded_unclipped,
        last_clipping=FrameEdge.TOP,
        outer_visibility=FrameEdge.LEFT | FrameEdge.RIGHT | FrameEdge.BOTTOM,
    )
    proposal = _propose(degraded, phase=phase)
    assert not proposal.is_exact_zero
    assert proposal.uncertainty.limited is True
    assert proposal.uncertainty.reason == (
        "bounded_gate1_recenter_degraded_or_clipped"
    )

    withheld = _propose(degraded, phase=dataclasses.replace(
        phase,
        objective_permitted=False,
        withholding_reason="guidance_uncertainty_gate",
    ))
    assert withheld.is_exact_zero
    assert withheld.source_frame is None


def test_gate1_corridor_and_timeout_are_source_less_zero_not_passage():
    state = _state(gate_index=1, bearing=(0.05, -0.05), bearing_rate=(0.1, -0.1))
    corridor = _propose(
        state,
        phase=_phase(mode=VQ2ControlPhase.GATE1_RECENTER, elapsed_s=0.2),
    )
    timeout = _propose(
        dataclasses.replace(state, bearing_norm=(0.5, -0.5)),
        phase=_phase(mode=VQ2ControlPhase.GATE1_RECENTER, elapsed_s=0.60),
    )
    for proposal in (corridor, timeout):
        assert proposal.is_exact_zero
        assert proposal.source_frame is None
        assert proposal.phase == "gate1_recenter"
        assert "pass" not in proposal.reason
    assert "corridor_reached" in corridor.reason
    assert "time_limit" in timeout.reason


def test_gate1_time_limit_boundary_is_exact():
    state = _state(gate_index=1, bearing=(0.5, 0.5))
    below = _propose(
        state,
        phase=_phase(
            mode=VQ2ControlPhase.GATE1_RECENTER,
            elapsed_s=math.nextafter(0.60, 0.0),
        ),
    )
    at_limit = _propose(
        state,
        phase=_phase(mode=VQ2ControlPhase.GATE1_RECENTER, elapsed_s=0.60),
    )
    assert not below.is_exact_zero
    assert at_limit.is_exact_zero


def test_source_less_withholding_preserves_tick_authority_and_ids():
    state = _state(role=TrackRole.SHADOW)
    tick = _tick(state)
    proposal = _propose(state, tick=tick)
    assert proposal.authority == tick.expected_authority
    assert proposal.proposal_id == tick.proposal_id
    assert proposal.control_tick_id == tick.control_tick_id
    assert proposal.source_frame is None
    with pytest.raises(ValueError, match="has no relative state"):
        validate_command_proposal_source(proposal, state)


def test_local_inputs_and_configuration_reject_unsafe_ambiguity():
    authority = _authority()
    with pytest.raises(ControllerInputError, match="deadline predates"):
        ControllerTickInput(1, 1, _HOST, 10, 9, 0, 0, authority)
    with pytest.raises(ControllerInputError, match="watermark postdates"):
        ControllerTickInput(1, 1, _HOST, 10, 10, 11, 0, authority)
    with pytest.raises(ValueError, match="exact-zero target pitch"):
        _phase(
            mode=VQ2ControlPhase.GATE1_RECENTER,
            initial_pitch_rad=0.01,
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"gate0_max_roll_rad": math.nextafter(0.08, math.inf)},
        {"gate0_pitch_blend_s": math.nextafter(0.8, math.inf)},
        {"gate0_launch_end_s": math.nextafter(0.15, math.inf)},
        {"gate0_boost_end_s": math.nextafter(0.45, math.inf)},
        {"gate0_max_body_rate_rad_s": math.nextafter(0.25, math.inf)},
        {"gate0_max_thrust": math.nextafter(0.32, math.inf)},
        {"gate1_max_roll_rad": math.nextafter(0.05, math.inf)},
        {"gate1_max_body_rate_rad_s": math.nextafter(0.12, math.inf)},
        {"gate1_max_thrust": math.nextafter(0.30, math.inf)},
        {"gate1_max_duration_s": math.nextafter(0.60, math.inf)},
        {"max_state_age_ns": 100_000_001},
        {"max_measurement_age_ns": 150_000_001},
        {"max_prediction_lead_ns": 100_000_001},
        {"max_measurement_uncertainty_ns": 50_000_001},
        {"max_abs_initial_pitch_rad": math.nextafter(0.6108652381980153, math.inf)},
        {"max_abs_bearing_error_norm": math.nextafter(1.50, math.inf)},
        {"max_abs_bearing_rate_norm_s": math.nextafter(4.0, math.inf)},
        {"max_bearing_variance": math.nextafter(0.25, math.inf)},
        {"max_log_scale_variance": math.nextafter(1.0, math.inf)},
        {"max_bearing_rate_variance": math.nextafter(16.0, math.inf)},
        {"max_expansion_rate_variance": math.nextafter(16.0, math.inf)},
        {"gate0_min_thrust": math.nextafter(0.21, 0.0)},
        {"gate1_min_thrust": math.nextafter(0.21, 0.0)},
        {"gate1_corridor_x_norm": math.nextafter(0.10, 0.0)},
        {"gate1_corridor_y_norm": math.nextafter(0.12, 0.0)},
        {"gate1_corridor_rate_norm_s": math.nextafter(0.25, 0.0)},
    ),
)
def test_configuration_cannot_loosen_reviewed_safeguards(changes):
    with pytest.raises(ValueError, match="reviewed hard"):
        PredictiveControllerConfig(**changes)


def test_gain_tuning_remains_bounded_by_hard_output_envelopes():
    config = PredictiveControllerConfig(
        gate0_roll_gain_rad_per_norm=100.0,
        attitude_kp_roll=100.0,
        attitude_kp_pitch=100.0,
        vertical_position_gain=100.0,
        vertical_rate_damping_per_px_s=100.0,
    )
    proposal = _propose(
        _state(bearing=(1.0, -1.0), bearing_rate=(1.0, -1.0)),
        attitude=_attitude(roll=-0.5, pitch=0.5),
        config=config,
    )
    assert all(abs(rate) <= 0.25 for rate in proposal.requested_body_rates_rad_s)
    assert proposal.requested_thrust <= 0.32
    assert proposal.saturation.body_rate_axes[:2] == (True, True)
    assert proposal.saturation.thrust is True


def test_controller_module_has_no_transport_or_powered_imports():
    module_path = Path(__file__).parents[1] / "vq2_controller.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        forbidden in imported
        for imported in imports
        for forbidden in (
            "adapter",
            "aigp_mavlink",
            "scripts",
            "socket",
            "transport",
            "supervisor",
        )
    )
