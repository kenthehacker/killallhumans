"""Pure offline composition of VQ2 guidance and predictive control.

The adapter owns the accepted :class:`VQ2GuidanceMemory` passed from one call
to the next and is the only caller of ``step_vq2_guidance`` in this offline
composition.  It never accepts a caller-assembled guidance decision.  That
state ownership is what lets a coordinated same-phase start renewal fail
against the previously accepted phase start.

Only two mappings exist: Gate 0 ``APPROACH`` and Gate 1 ``ALIGN`` recentering.
Every other phase, including ``COMMIT``, yields a source-less exact-zero
``CommandProposalV1``.  A proposal remains controller intent only; this module
has no scheduler, supervisor approval, runtime, transport, reset, arm, cleanup,
simulator, or system-identification imports.

``ControllerAttitudeInput`` and the Gate 0 pitch basis still have no timestamp,
clock identity, or source correlation.  The frozen proposal cannot bind either
one.  This adapter is therefore offline evidence only and is ineligible for
shadow, runtime, or powered wiring until a reviewed IMU provenance seam exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from competition.vq2_contracts import (
    CommandProposalV1,
    GateAuthorityEpochV1,
    RelativeGateStateV1,
    SaturationDiagnosticsV1,
    TrackRole,
    UncertaintyDiagnosticsV1,
    validate_command_proposal_source,
)
from competition.vq2_controller import (
    ControllerAttitudeInput,
    ControllerPhaseInput,
    ControllerTickInput,
    VQ2ControlPhase,
    propose_vq2_command,
)
from planning.vq2_guidance import (
    VQ2GuidanceDecision,
    VQ2GuidanceMemory,
    VQ2GuidanceObjectiveKind,
    VQ2GuidancePhase,
    VQ2GuidanceRaceState,
    VQ2GuidanceSource,
    VQ2SafetyGuidanceInput,
    step_vq2_guidance,
)


@dataclass(frozen=True, slots=True)
class VQ2Gate0PitchLatch:
    """Offline-only Gate 0 pitch basis bound to one accepted phase entry."""

    session_id: str
    reset_epoch: int
    gate_epoch: int
    expected_gate_index: int
    host_clock_id: str
    phase_started_monotonic_ns: int
    initial_pitch_rad: Optional[float]

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or not self.session_id:
            raise TypeError("session_id must be a non-empty exact string")
        if type(self.host_clock_id) is not str or not self.host_clock_id:
            raise TypeError("host_clock_id must be a non-empty exact string")
        for name in (
            "reset_epoch",
            "gate_epoch",
            "expected_gate_index",
            "phase_started_monotonic_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be an exact nonnegative integer")
        if self.expected_gate_index != 0:
            raise ValueError("the Gate 0 pitch latch is valid only for gate index zero")
        if self.initial_pitch_rad is not None:
            if type(self.initial_pitch_rad) is not float:
                raise TypeError("initial_pitch_rad must be an exact float or None")
            if not math.isfinite(self.initial_pitch_rad):
                raise ValueError("initial_pitch_rad must be finite")


@dataclass(frozen=True, slots=True)
class VQ2Wave2AdapterMemory:
    """Immutable state that a caller must thread through every adapter call."""

    guidance_memory: VQ2GuidanceMemory
    gate0_pitch_latch: Optional[VQ2Gate0PitchLatch]

    def __post_init__(self) -> None:
        if type(self.guidance_memory) is not VQ2GuidanceMemory:
            raise TypeError("guidance_memory must be exact VQ2GuidanceMemory")
        if self.gate0_pitch_latch is not None:
            if type(self.gate0_pitch_latch) is not VQ2Gate0PitchLatch:
                raise TypeError("gate0_pitch_latch has the wrong type")
            expected = _pitch_latch_identity(self.guidance_memory.safety)
            if expected is None or expected != _pitch_latch_identity_from_latch(
                self.gate0_pitch_latch
            ):
                raise ValueError(
                    "Gate 0 pitch latch does not match accepted guidance phase"
                )


@dataclass(frozen=True, slots=True)
class VQ2Wave2AdapterTransition:
    """One pure guidance/controller transition with no send authority.

    A sourced proposal must share the accepted decision authority.  A
    composition failure instead returns a tick-scoped, source-less exact zero;
    its host/authority may intentionally differ from the accepted decision when
    that mismatch is the reason for withholding.
    """

    memory: VQ2Wave2AdapterMemory
    decision: VQ2GuidanceDecision
    proposal: CommandProposalV1

    def __post_init__(self) -> None:
        if type(self.memory) is not VQ2Wave2AdapterMemory:
            raise TypeError("memory must be exact VQ2Wave2AdapterMemory")
        if type(self.decision) is not VQ2GuidanceDecision:
            raise TypeError("decision must be exact VQ2GuidanceDecision")
        if type(self.proposal) is not CommandProposalV1:
            raise TypeError("proposal must be exact CommandProposalV1")
        if not _decision_matches_safety(
            self.decision,
            self.memory.guidance_memory.safety,
        ):
            raise ValueError("adapter decision does not match accepted guidance memory")
        if self.proposal.source_frame is None:
            if not self.proposal.is_exact_zero:
                raise ValueError("a source-less adapter proposal must be exact zero")
        else:
            if self.proposal.authority != self.decision.authority:
                raise ValueError("a sourced proposal must match decision authority")
            if self.decision.source is None or not _proposal_matches_source(
                self.proposal,
                self.decision.source,
            ):
                raise ValueError("a sourced proposal must match guidance source")
            if self.memory.guidance_memory.active_source != self.decision.source:
                raise ValueError(
                    "a sourced proposal requires the retained active source"
                )
            control_phase = _controller_phase_for(self.decision)
            if (
                control_phase is None
                or not self.decision.objective_permitted
                or self.decision.withholding_reason is not None
                or self.decision.race_state is not VQ2GuidanceRaceState.UNDERWAY
                or self.decision.target_bearing_norm != (0.0, 0.0)
                or self.proposal.phase != control_phase.value
            ):
                raise ValueError("a sourced proposal requires a supported decision")
            if control_phase is VQ2ControlPhase.GATE0_APPROACH:
                latch = self.memory.gate0_pitch_latch
                if latch is None or latch.initial_pitch_rad is None:
                    raise ValueError(
                        "a sourced Gate 0 proposal requires its pitch latch"
                    )
            elif self.memory.gate0_pitch_latch is not None:
                raise ValueError("a Gate 1 proposal cannot retain a Gate 0 pitch latch")


_SUPPORTED_MAPPINGS = (
    (
        (
            0,
            VQ2GuidancePhase.APPROACH,
            VQ2GuidanceObjectiveKind.APPROACH_ACTIVE_GATE,
        ),
        VQ2ControlPhase.GATE0_APPROACH,
    ),
    (
        (
            1,
            VQ2GuidancePhase.ALIGN,
            VQ2GuidanceObjectiveKind.RECENTER_ACTIVE_GATE,
        ),
        VQ2ControlPhase.GATE1_RECENTER,
    ),
)


def step_vq2_wave2_adapter(
    memory: Optional[VQ2Wave2AdapterMemory],
    safety: VQ2SafetyGuidanceInput,
    *,
    active_state: Optional[RelativeGateStateV1],
    shadow_states: tuple[RelativeGateStateV1, ...] = (),
    attitude: Optional[ControllerAttitudeInput],
    tick: ControllerTickInput,
    gate0_initial_pitch_rad: Optional[float] = None,
) -> VQ2Wave2AdapterTransition:
    """Advance guidance and return one bounded proposal or exact-zero hold.

    The returned memory is part of the safety argument and must be used for the
    next call.  Malformed exact-type inputs raise.  Valid but incoherent,
    unsupported, stale, uncertain, or withheld inputs produce a source-less
    exact-zero proposal.
    """

    if memory is not None and type(memory) is not VQ2Wave2AdapterMemory:
        raise TypeError("memory must be VQ2Wave2AdapterMemory or None")
    if type(safety) is not VQ2SafetyGuidanceInput:
        raise TypeError("safety must be exact VQ2SafetyGuidanceInput")
    if active_state is not None and type(active_state) is not RelativeGateStateV1:
        raise TypeError("active_state must be RelativeGateStateV1 or None")
    if type(shadow_states) is not tuple or any(
        type(state) is not RelativeGateStateV1 for state in shadow_states
    ):
        raise TypeError("shadow_states must be an exact tuple of relative states")
    if attitude is not None and type(attitude) is not ControllerAttitudeInput:
        raise TypeError("attitude must be ControllerAttitudeInput or None")
    if type(tick) is not ControllerTickInput:
        raise TypeError("tick must be exact ControllerTickInput")
    if gate0_initial_pitch_rad is not None:
        if type(gate0_initial_pitch_rad) is not float:
            raise TypeError("gate0_initial_pitch_rad must be an exact float or None")
        if not math.isfinite(gate0_initial_pitch_rad):
            raise ValueError("gate0_initial_pitch_rad must be finite")

    guidance_transition = step_vq2_guidance(
        None if memory is None else memory.guidance_memory,
        safety,
        active_state=active_state,
        shadow_states=shadow_states,
    )
    next_memory, pitch_failure = _advance_adapter_memory(
        memory,
        guidance_transition.memory,
        gate0_initial_pitch_rad=gate0_initial_pitch_rad,
    )
    decision = guidance_transition.decision

    failure = _adapter_eligibility_failure(
        next_memory,
        decision,
        active_state=active_state,
        attitude=attitude,
        tick=tick,
        pitch_failure=pitch_failure,
    )
    if failure is not None:
        proposal = _zero_proposal(tick, decision, failure)
        return VQ2Wave2AdapterTransition(next_memory, decision, proposal)

    assert active_state is not None
    assert attitude is not None
    control_phase = _controller_phase_for(decision)
    assert control_phase is not None
    initial_pitch_rad = 0.0
    if control_phase is VQ2ControlPhase.GATE0_APPROACH:
        latch = next_memory.gate0_pitch_latch
        assert latch is not None and latch.initial_pitch_rad is not None
        initial_pitch_rad = latch.initial_pitch_rad
    phase = ControllerPhaseInput(
        mode=control_phase,
        phase_host_clock_id=decision.evaluation_host_clock_id,
        phase_started_monotonic_ns=decision.phase_started_monotonic_ns,
        evaluation_monotonic_ns=decision.evaluation_monotonic_ns,
        initial_pitch_rad=initial_pitch_rad,
        target_bearing_norm=decision.target_bearing_norm,
        objective_permitted=True,
        withholding_reason=None,
    )
    proposal = propose_vq2_command(
        active_state,
        attitude=attitude,
        tick=tick,
        phase=phase,
    )
    if proposal.source_frame is None:
        if not proposal.is_exact_zero:
            raise AssertionError("a source-less controller proposal must be exact zero")
    else:
        validate_command_proposal_source(proposal, active_state)
    return VQ2Wave2AdapterTransition(next_memory, decision, proposal)


def _advance_adapter_memory(
    previous: Optional[VQ2Wave2AdapterMemory],
    guidance_memory: VQ2GuidanceMemory,
    *,
    gate0_initial_pitch_rad: Optional[float],
) -> tuple[VQ2Wave2AdapterMemory, Optional[str]]:
    identity = _pitch_latch_identity(guidance_memory.safety)
    if identity is None:
        return VQ2Wave2AdapterMemory(guidance_memory, None), (
            "unexpected_gate0_pitch_basis"
            if gate0_initial_pitch_rad is not None
            else None
        )

    previous_latch = None if previous is None else previous.gate0_pitch_latch
    if (
        previous_latch is not None
        and _pitch_latch_identity_from_latch(previous_latch) == identity
    ):
        failure = None
        if gate0_initial_pitch_rad is not None and (
            previous_latch.initial_pitch_rad is None
            or gate0_initial_pitch_rad != previous_latch.initial_pitch_rad
        ):
            failure = "gate0_pitch_basis_changed"
        return VQ2Wave2AdapterMemory(guidance_memory, previous_latch), failure

    latch = VQ2Gate0PitchLatch(
        session_id=identity[0],
        reset_epoch=identity[1],
        gate_epoch=identity[2],
        expected_gate_index=identity[3],
        host_clock_id=identity[4],
        phase_started_monotonic_ns=identity[5],
        initial_pitch_rad=gate0_initial_pitch_rad,
    )
    return VQ2Wave2AdapterMemory(guidance_memory, latch), (
        "gate0_pitch_basis_missing"
        if gate0_initial_pitch_rad is None
        else None
    )


def _adapter_eligibility_failure(
    memory: VQ2Wave2AdapterMemory,
    decision: VQ2GuidanceDecision,
    *,
    active_state: Optional[RelativeGateStateV1],
    attitude: Optional[ControllerAttitudeInput],
    tick: ControllerTickInput,
    pitch_failure: Optional[str],
) -> Optional[str]:
    safety = memory.guidance_memory.safety
    if not _decision_matches_safety(decision, safety):
        return "decision_memory_mismatch"
    if pitch_failure is not None:
        return pitch_failure
    if (
        decision.race_state is not VQ2GuidanceRaceState.UNDERWAY
        or not decision.objective_permitted
        or decision.withholding_reason is not None
    ):
        reason = (
            "guidance_withheld"
            if decision.withholding_reason is None
            else f"guidance_{decision.withholding_reason.value}"
        )
        return reason
    control_phase = _controller_phase_for(decision)
    if control_phase is None:
        return "unsupported_guidance_mapping"
    if decision.target_bearing_norm != (0.0, 0.0):
        return "noncentered_guidance_target"
    if active_state is None:
        return "active_state_missing"
    if attitude is None:
        return "attitude_missing"
    if active_state.track_role is not TrackRole.ACTIVE:
        return "active_role_mismatch"

    expected_source = _source_from_state(active_state)
    if (
        decision.source != expected_source
        or memory.guidance_memory.active_source != expected_source
    ):
        return "guidance_source_mismatch"
    if (
        decision.authority != safety.authority
        or active_state.authority != decision.authority
        or tick.expected_authority != decision.authority
    ):
        return "authority_mismatch"
    expected_clock = decision.authority.camera_host_clock_id
    if (
        decision.evaluation_host_clock_id != expected_clock
        or active_state.timing.host_clock_id != expected_clock
        or expected_source.host_clock_id != expected_clock
        or tick.host_clock_id != expected_clock
    ):
        return "host_clock_mismatch"
    if (
        tick.expected_phase_started_monotonic_ns
        != decision.phase_started_monotonic_ns
    ):
        return "phase_start_watermark_mismatch"
    if (
        tick.minimum_phase_evaluation_monotonic_ns
        != decision.evaluation_monotonic_ns
    ):
        return "phase_evaluation_watermark_mismatch"
    if (
        tick.minimum_state_decision_monotonic_ns
        != active_state.timing.decision_time_monotonic_ns
    ):
        return "state_decision_watermark_mismatch"
    if tick.minimum_state_sequence != active_state.state_sequence:
        return "state_sequence_watermark_mismatch"
    if decision.evaluation_monotonic_ns > tick.proposal_monotonic_ns:
        return "proposal_predates_guidance_evaluation"

    if control_phase is VQ2ControlPhase.GATE0_APPROACH:
        latch = memory.gate0_pitch_latch
        if latch is None or latch.initial_pitch_rad is None:
            return "gate0_pitch_basis_unavailable"
    elif memory.gate0_pitch_latch is not None:
        return "gate0_pitch_latch_outside_gate0_approach"
    return None


def _decision_matches_safety(
    decision: VQ2GuidanceDecision,
    safety: VQ2SafetyGuidanceInput,
) -> bool:
    return bool(
        decision.authority == safety.authority
        and decision.phase is safety.phase
        and decision.race_state is safety.race_state
        and decision.evaluation_host_clock_id == safety.evaluation_host_clock_id
        and decision.evaluation_monotonic_ns == safety.evaluation_monotonic_ns
        and decision.phase_started_monotonic_ns
        == safety.phase_started_monotonic_ns
    )


def _source_from_state(state: RelativeGateStateV1) -> VQ2GuidanceSource:
    timing = state.timing
    assert timing.source_frame is not None
    assert timing.source_frame_publication_sequence is not None
    assert timing.source_frame_publish_monotonic_ns is not None
    return VQ2GuidanceSource(
        host_clock_id=timing.host_clock_id,
        decision_time_monotonic_ns=timing.decision_time_monotonic_ns,
        prediction_time_monotonic_ns=timing.prediction_time_monotonic_ns,
        source_frame=timing.source_frame,
        source_frame_publication_sequence=timing.source_frame_publication_sequence,
        source_frame_publish_monotonic_ns=timing.source_frame_publish_monotonic_ns,
        tracker_id=state.tracker_id,
        track_role=state.track_role,
        state_sequence=state.state_sequence,
        measurement_update_sequence=state.measurement_update_sequence,
        source_candidate_id=state.source_candidate_id,
    )


def _controller_phase_for(
    decision: VQ2GuidanceDecision,
) -> Optional[VQ2ControlPhase]:
    key = (
        decision.authority.expected_gate_index,
        decision.phase,
        decision.objective_kind,
    )
    for candidate, control_phase in _SUPPORTED_MAPPINGS:
        if key == candidate:
            return control_phase
    return None


def _proposal_matches_source(
    proposal: CommandProposalV1,
    source: VQ2GuidanceSource,
) -> bool:
    return bool(
        proposal.host_clock_id == source.host_clock_id
        and proposal.source_state_decision_monotonic_ns
        == source.decision_time_monotonic_ns
        and proposal.source_state_prediction_monotonic_ns
        == source.prediction_time_monotonic_ns
        and proposal.source_frame == source.source_frame
        and proposal.source_frame_publication_sequence
        == source.source_frame_publication_sequence
        and proposal.source_frame_publish_monotonic_ns
        == source.source_frame_publish_monotonic_ns
        and proposal.source_tracker_id == source.tracker_id
        and proposal.source_track_role is source.track_role
        and proposal.source_state_sequence == source.state_sequence
        and proposal.source_measurement_update_sequence
        == source.measurement_update_sequence
        and proposal.source_candidate_id == source.source_candidate_id
    )


def _pitch_latch_identity(
    safety: VQ2SafetyGuidanceInput,
) -> Optional[tuple[str, int, int, int, str, int]]:
    if (
        safety.authority.expected_gate_index != 0
        or safety.phase is not VQ2GuidancePhase.APPROACH
    ):
        return None
    authority = safety.authority
    return (
        authority.session_id,
        authority.reset_epoch,
        authority.gate_epoch,
        authority.expected_gate_index,
        safety.evaluation_host_clock_id,
        safety.phase_started_monotonic_ns,
    )


def _pitch_latch_identity_from_latch(
    latch: VQ2Gate0PitchLatch,
) -> tuple[str, int, int, int, str, int]:
    return (
        latch.session_id,
        latch.reset_epoch,
        latch.gate_epoch,
        latch.expected_gate_index,
        latch.host_clock_id,
        latch.phase_started_monotonic_ns,
    )


def _zero_proposal(
    tick: ControllerTickInput,
    decision: VQ2GuidanceDecision,
    reason: str,
) -> CommandProposalV1:
    """Build a tick-scoped zero; mismatched tick authority is never relabeled."""

    return CommandProposalV1(
        proposal_id=tick.proposal_id,
        control_tick_id=tick.control_tick_id,
        host_clock_id=tick.host_clock_id,
        proposal_monotonic_ns=tick.proposal_monotonic_ns,
        control_tick_deadline_monotonic_ns=(
            tick.control_tick_deadline_monotonic_ns
        ),
        source_state_decision_monotonic_ns=None,
        source_state_prediction_monotonic_ns=None,
        source_frame=None,
        source_frame_publication_sequence=None,
        source_frame_publish_monotonic_ns=None,
        source_tracker_id=None,
        source_track_role=None,
        source_state_sequence=None,
        source_measurement_update_sequence=None,
        source_candidate_id=None,
        authority=tick.expected_authority,
        requested_body_rates_rad_s=(0.0, 0.0, 0.0),
        requested_thrust=0.0,
        phase=decision.phase.value,
        reason=f"withheld:adapter_{reason}",
        saturation=SaturationDiagnosticsV1(
            body_rate_axes=(False, False, False),
            thrust=False,
        ),
        uncertainty=UncertaintyDiagnosticsV1(limited=False, reason=None),
    )


__all__ = [
    "VQ2Gate0PitchLatch",
    "VQ2Wave2AdapterMemory",
    "VQ2Wave2AdapterTransition",
    "step_vq2_wave2_adapter",
]
