"""Offline IMU-provenance envelope around the reviewed Wave 2 adapter.

The input state is produced by the unchanged raw-camera relative estimator and
is paired locally with exact IMU/derotation correlation evidence for the same
observation and prediction target.  The standalone corrected bearing is never
injected into the capture-time ``/1`` estimator, guidance, or proposal.  This
module uses only the correlated attitude: it propagates the source sample to
the proposal time for control and, on Gate 0 entry, separately to the exact
phase-start time for the immutable pitch basis.

The returned ``CommandProposalV1`` still cannot carry attitude, pitch,
calibration, or derotation identity.  This is deterministic offline evidence
only, with no scheduler, supervisor, approval, runner, MAVLink, transport,
reset, arm, cleanup, network, simulator, or powered dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from competition.vq2_contracts import (
    CommandProposalV1,
    GateObservationV1,
    RelativeGateStateV1,
    TrackRole,
    validate_command_proposal_source,
)
from competition.vq2_controller import ControllerAttitudeInput, ControllerTickInput
from competition.vq2_wave2_adapter import (
    VQ2Gate0PitchLatch,
    VQ2Wave2AdapterMemory,
    VQ2Wave2AdapterTransition,
    step_vq2_wave2_adapter,
)
from estimation.vq2_imu_derotation import (
    VQ2AttitudeDerotationInput,
    VQ2CameraToBodyCalibration,
    VQ2DerotationEvidence,
    VQ2DerotationModel,
)
from estimation.vq2_imu_provenance import VQ2TimestampedAttitude
from estimation.vq2_relative_estimator import (
    RelativeEstimatorUpdate,
    RelativePredictionTarget,
    VQ2ImuCorrelatedEstimatorUpdate,
)
from planning.vq2_guidance import (
    VQ2GuidanceDecision,
    VQ2GuidancePhase,
    VQ2GuidanceRaceState,
    VQ2GuidanceSource,
    VQ2SafetyGuidanceInput,
)


CONTROLLER_ATTITUDE_PROPAGATION_MODEL_ID = (
    "vq2-host-receive-constant-body-rate-v1"
)
HARD_MAX_CONTROLLER_ATTITUDE_EXTRAPOLATION_NS = 20_000_000
HARD_MAX_CONTROLLER_ATTITUDE_EFFECTIVE_AGE_NS = 50_000_000
HARD_MAX_CONTROLLER_ATTITUDE_UNCERTAINTY_RAD = math.radians(5.0)


@dataclass(frozen=True, slots=True)
class VQ2PropagatedAttitudeProvenance:
    """One exact sample propagated to one controller-domain host time."""

    evidence: VQ2DerotationEvidence
    target_host_clock_id: str
    target_monotonic_ns: int
    propagation_model_id: str
    extrapolation_ns: int
    effective_age_ns: int
    angular_uncertainty_rad: float
    orientation_body_to_ned_wxyz: tuple[float, float, float, float]
    body_rates_rad_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        if type(self.evidence) is not VQ2DerotationEvidence:
            raise TypeError("evidence must be exact VQ2DerotationEvidence")
        if type(self.target_host_clock_id) is not str or not self.target_host_clock_id:
            raise TypeError("target_host_clock_id must be a non-empty exact string")
        if (
            type(self.propagation_model_id) is not str
            or self.propagation_model_id
            != CONTROLLER_ATTITUDE_PROPAGATION_MODEL_ID
        ):
            raise ValueError("unsupported controller attitude propagation model")
        for name in (
            "target_monotonic_ns",
            "extrapolation_ns",
            "effective_age_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be an exact nonnegative integer")

        attitude_input = self.evidence.target_attitude
        attitude = attitude_input.attitude
        if self.target_host_clock_id != attitude.host_clock_id:
            raise ValueError("propagated attitude host clock differs from its sample")
        expected_extrapolation = self.target_monotonic_ns - attitude.receive_monotonic_ns
        if expected_extrapolation < 0:
            raise ValueError("propagated attitude target predates sample receipt")
        if self.extrapolation_ns != expected_extrapolation:
            raise ValueError("attitude extrapolation does not match its source time")
        if (
            self.extrapolation_ns
            > HARD_MAX_CONTROLLER_ATTITUDE_EXTRAPOLATION_NS
        ):
            raise ValueError("controller attitude extrapolation exceeds hard bound")
        expected_effective_age = (
            expected_extrapolation + attitude_input.host_time_uncertainty_ns
        )
        if self.effective_age_ns != expected_effective_age:
            raise ValueError("effective attitude age does not match timing uncertainty")
        if (
            self.effective_age_ns
            > HARD_MAX_CONTROLLER_ATTITUDE_EFFECTIVE_AGE_NS
        ):
            raise ValueError("effective controller attitude age exceeds hard bound")

        expected_uncertainty = _propagated_angular_uncertainty_rad(
            self.evidence,
            expected_extrapolation,
        )
        if type(self.angular_uncertainty_rad) is not float or not math.isfinite(
            self.angular_uncertainty_rad
        ):
            raise TypeError("angular_uncertainty_rad must be an exact finite float")
        if not math.isclose(
            self.angular_uncertainty_rad,
            expected_uncertainty,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("propagated attitude uncertainty does not match evidence")
        if (
            self.angular_uncertainty_rad
            > HARD_MAX_CONTROLLER_ATTITUDE_UNCERTAINTY_RAD
        ):
            raise ValueError("controller attitude uncertainty exceeds hard bound")

        expected_orientation = attitude_input.orientation_at_host_time(
            self.target_host_clock_id,
            self.target_monotonic_ns,
        )
        if type(self.orientation_body_to_ned_wxyz) is not tuple or (
            self.orientation_body_to_ned_wxyz != expected_orientation
        ):
            raise ValueError("propagated orientation does not match its exact source")
        if type(self.body_rates_rad_s) is not tuple or (
            self.body_rates_rad_s != attitude.body_rates_rad_s
        ):
            raise ValueError("propagated body rates do not match their exact source")

    @property
    def attitude(self) -> VQ2TimestampedAttitude:
        return self.evidence.target_attitude.attitude


@dataclass(frozen=True, slots=True)
class VQ2Gate0PitchProvenance:
    """Exact bounded phase-entry attitude retained for one Gate 0 latch."""

    session_id: str
    reset_epoch: int
    gate_epoch: int
    expected_gate_index: int
    host_clock_id: str
    phase_started_monotonic_ns: int
    attitude_provenance: VQ2PropagatedAttitudeProvenance
    initial_pitch_rad: float

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
            raise ValueError("Gate 0 pitch provenance is valid only for gate zero")
        if type(self.attitude_provenance) is not VQ2PropagatedAttitudeProvenance:
            raise TypeError(
                "attitude_provenance must be exact propagated attitude evidence"
            )
        if (
            self.attitude_provenance.target_host_clock_id != self.host_clock_id
            or self.attitude_provenance.target_monotonic_ns
            != self.phase_started_monotonic_ns
        ):
            raise ValueError("Gate 0 pitch attitude is not targeted to phase entry")
        attitude = self.attitude_provenance.attitude
        if (
            attitude.session_id != self.session_id
            or attitude.reset_epoch != self.reset_epoch
            or attitude.host_clock_id != self.host_clock_id
        ):
            raise ValueError("Gate 0 pitch attitude does not match phase provenance")
        expected_pitch = _pitch_from_orientation(
            self.attitude_provenance.orientation_body_to_ned_wxyz
        )
        if type(self.initial_pitch_rad) is not float:
            raise TypeError("initial_pitch_rad must be an exact float")
        if self.initial_pitch_rad != expected_pitch:
            raise ValueError("Gate 0 pitch was not derived from phase-entry attitude")

    @property
    def attitude(self) -> VQ2TimestampedAttitude:
        return self.attitude_provenance.attitude


@dataclass(frozen=True, slots=True)
class VQ2Wave3ImuAdapterMemory:
    """Caller-threaded outer memory; still a local trust boundary."""

    inner_memory: VQ2Wave2AdapterMemory
    last_attitude: Optional[VQ2TimestampedAttitude]
    gate0_pitch_provenance: Optional[VQ2Gate0PitchProvenance]

    def __post_init__(self) -> None:
        if type(self.inner_memory) is not VQ2Wave2AdapterMemory:
            raise TypeError("inner_memory must be exact VQ2Wave2AdapterMemory")
        if self.last_attitude is not None and type(
            self.last_attitude
        ) is not VQ2TimestampedAttitude:
            raise TypeError("last_attitude must be VQ2TimestampedAttitude or None")
        if self.gate0_pitch_provenance is not None and type(
            self.gate0_pitch_provenance
        ) is not VQ2Gate0PitchProvenance:
            raise TypeError(
                "gate0_pitch_provenance must be VQ2Gate0PitchProvenance or None"
            )

        safety = self.inner_memory.guidance_memory.safety
        gate0_identity = _gate0_safety_identity(safety)
        if gate0_identity is not None and self.inner_memory.gate0_pitch_latch is None:
            raise ValueError("accepted Gate 0 approach memory must retain its latch")
        _validate_pitch_provenance(
            self.inner_memory.gate0_pitch_latch,
            self.gate0_pitch_provenance,
        )
        if self.gate0_pitch_provenance is not None and self.last_attitude is None:
            raise ValueError("Gate 0 pitch provenance requires retained attitude lineage")
        if self.last_attitude is not None:
            authority = safety.authority
            if (
                self.last_attitude.session_id != authority.session_id
                or self.last_attitude.reset_epoch != authority.reset_epoch
                or self.last_attitude.host_clock_id
                != safety.evaluation_host_clock_id
            ):
                raise ValueError(
                    "retained attitude does not match accepted guidance safety"
                )


@dataclass(frozen=True, slots=True)
class VQ2Wave3ImuAdapterTransition:
    """One outer transition retaining local evidence alongside frozen intent."""

    memory: VQ2Wave3ImuAdapterMemory
    inner_transition: VQ2Wave2AdapterTransition
    active_update: Optional[VQ2ImuCorrelatedEstimatorUpdate]
    accepted_attitude: Optional[VQ2TimestampedAttitude]
    controller_attitude_provenance: Optional[VQ2PropagatedAttitudeProvenance]
    outer_withholding_reason: Optional[str]

    def __post_init__(self) -> None:
        if type(self.memory) is not VQ2Wave3ImuAdapterMemory:
            raise TypeError("memory must be exact VQ2Wave3ImuAdapterMemory")
        if type(self.inner_transition) is not VQ2Wave2AdapterTransition:
            raise TypeError("inner_transition must be exact VQ2Wave2AdapterTransition")
        if self.memory.inner_memory != self.inner_transition.memory:
            raise ValueError("outer memory must retain the exact inner transition memory")
        if self.active_update is not None and type(
            self.active_update
        ) is not VQ2ImuCorrelatedEstimatorUpdate:
            raise TypeError(
                "active_update must be VQ2ImuCorrelatedEstimatorUpdate or None"
            )
        if self.accepted_attitude is not None and type(
            self.accepted_attitude
        ) is not VQ2TimestampedAttitude:
            raise TypeError("accepted_attitude must be VQ2TimestampedAttitude or None")
        if self.controller_attitude_provenance is not None and type(
            self.controller_attitude_provenance
        ) is not VQ2PropagatedAttitudeProvenance:
            raise TypeError(
                "controller_attitude_provenance has the wrong exact type"
            )
        if self.outer_withholding_reason is not None and (
            type(self.outer_withholding_reason) is not str
            or not self.outer_withholding_reason
        ):
            raise TypeError("outer_withholding_reason must be non-empty or None")

        proposal = self.inner_transition.proposal
        if self.outer_withholding_reason is not None:
            if proposal.source_frame is not None or not proposal.is_exact_zero:
                raise ValueError("outer withholding requires source-less exact zero")
            if self.accepted_attitude is not None:
                raise ValueError("outer withholding cannot advance attitude lineage")
            if self.controller_attitude_provenance is not None:
                raise ValueError("outer withholding cannot claim controller attitude")

        if (self.accepted_attitude is None) != (
            self.controller_attitude_provenance is None
        ):
            raise ValueError(
                "accepted attitude and proposal-time provenance are all-or-none"
            )

        if self.accepted_attitude is not None:
            if (
                self.active_update is None
                or not self.active_update.current_observation_accepted
            ):
                raise ValueError("accepted attitude requires an accepted observation")
            accepted_safety = self.memory.inner_memory.guidance_memory.safety
            if (
                self.active_update.state.authority != accepted_safety.authority
                or self.active_update.evidence.observation.authority
                != accepted_safety.authority
                or self.active_update.state.timing.host_clock_id
                != accepted_safety.evaluation_host_clock_id
                or self.active_update.evidence.observation.host_clock_id
                != accepted_safety.evaluation_host_clock_id
            ):
                raise ValueError(
                    "accepted correlation does not match retained safety authority"
                )
            target_attitude = self.active_update.evidence.target_attitude.attitude
            if self.accepted_attitude != target_attitude:
                raise ValueError("accepted attitude does not match IMU correlation")
            if self.memory.last_attitude != self.accepted_attitude:
                raise ValueError("accepted attitude was not retained in outer memory")
            decision_source = self.inner_transition.decision.source
            if (
                decision_source is None
                or not _guidance_source_matches_state(
                    decision_source,
                    self.active_update.state,
                )
                or self.memory.inner_memory.guidance_memory.active_source
                != decision_source
            ):
                raise ValueError(
                    "accepted correlation differs from retained guidance source"
                )

        if self.controller_attitude_provenance is not None:
            if not _is_supported_safety_mapping(
                self.memory.inner_memory.guidance_memory.safety
            ):
                raise ValueError(
                    "controller attitude provenance requires a supported mapping"
                )
            if self.accepted_attitude is None or self.active_update is None:
                raise ValueError("controller attitude requires accepted correlation")
            if self.controller_attitude_provenance.evidence != (
                self.active_update.evidence
            ):
                raise ValueError("controller attitude evidence differs from update")
            if (
                self.controller_attitude_provenance.target_host_clock_id
                != proposal.host_clock_id
                or self.controller_attitude_provenance.target_monotonic_ns
                != proposal.proposal_monotonic_ns
            ):
                raise ValueError("controller attitude is not targeted to proposal time")

        if proposal.source_frame is not None:
            if (
                self.active_update is None
                or not self.active_update.current_observation_accepted
                or self.accepted_attitude is None
                or self.controller_attitude_provenance is None
            ):
                raise ValueError(
                    "sourced proposal requires accepted camera/IMU correlation"
                )
            validate_command_proposal_source(proposal, self.active_update.state)

    @property
    def proposal(self) -> CommandProposalV1:
        return self.inner_transition.proposal

    @property
    def decision(self) -> VQ2GuidanceDecision:
        return self.inner_transition.decision


def step_vq2_wave3_imu_adapter(
    memory: Optional[VQ2Wave3ImuAdapterMemory],
    safety: VQ2SafetyGuidanceInput,
    *,
    active_update: Optional[VQ2ImuCorrelatedEstimatorUpdate],
    shadow_states: tuple[RelativeGateStateV1, ...] = (),
    tick: ControllerTickInput,
) -> VQ2Wave3ImuAdapterTransition:
    """Advance the offline composition or return source-less exact zero."""

    if memory is not None and type(memory) is not VQ2Wave3ImuAdapterMemory:
        raise TypeError("memory must be VQ2Wave3ImuAdapterMemory or None")
    if type(safety) is not VQ2SafetyGuidanceInput:
        raise TypeError("safety must be exact VQ2SafetyGuidanceInput")
    if active_update is not None and type(
        active_update
    ) is not VQ2ImuCorrelatedEstimatorUpdate:
        raise TypeError("active_update must be VQ2ImuCorrelatedEstimatorUpdate or None")
    if type(shadow_states) is not tuple or any(
        type(state) is not RelativeGateStateV1 for state in shadow_states
    ):
        raise TypeError("shadow_states must be an exact tuple of relative states")
    if type(tick) is not ControllerTickInput:
        raise TypeError("tick must be exact ControllerTickInput")

    supported = _is_supported_safety_mapping(safety)
    outer_failure: Optional[str] = None
    update_valid = False
    if active_update is None:
        if supported:
            outer_failure = "imu_correlated_update_missing"
    else:
        outer_failure = _outer_update_failure(
            memory,
            safety,
            active_update=active_update,
            tick=tick,
        )
        update_valid = outer_failure is None

    controller_provenance: Optional[VQ2PropagatedAttitudeProvenance] = None
    entry_pitch_provenance: Optional[VQ2Gate0PitchProvenance] = None
    if update_valid and supported:
        assert active_update is not None
        controller_provenance = _make_propagated_attitude_provenance(
            active_update.evidence,
            target_host_clock_id=tick.host_clock_id,
            target_monotonic_ns=tick.proposal_monotonic_ns,
        )
        if _needs_gate0_entry_provenance(memory, safety):
            phase_failure = _propagated_attitude_failure(
                active_update.evidence,
                target_host_clock_id=safety.evaluation_host_clock_id,
                target_monotonic_ns=safety.phase_started_monotonic_ns,
            )
            if phase_failure is not None:
                outer_failure = f"gate0_phase_{phase_failure}"
                update_valid = False
                controller_provenance = None
            else:
                phase_attitude = _make_propagated_attitude_provenance(
                    active_update.evidence,
                    target_host_clock_id=safety.evaluation_host_clock_id,
                    target_monotonic_ns=safety.phase_started_monotonic_ns,
                )
                entry_pitch_provenance = _make_gate0_pitch_provenance(
                    safety,
                    phase_attitude,
                )

    active_state = (
        active_update.state
        if active_update is not None and update_valid and outer_failure is None
        else None
    )
    controller_attitude = (
        None
        if controller_provenance is None
        else ControllerAttitudeInput(
            orientation_body_to_world_wxyz=(
                controller_provenance.orientation_body_to_ned_wxyz
            ),
            body_rates_rad_s=controller_provenance.body_rates_rad_s,
        )
    )
    pitch_argument = _gate0_pitch_argument(
        memory,
        safety,
        entry_pitch_provenance,
    )
    inner = step_vq2_wave2_adapter(
        None if memory is None else memory.inner_memory,
        safety,
        active_state=active_state,
        shadow_states=(
            shadow_states if active_state is not None else ()
        ),
        attitude=controller_attitude,
        tick=tick,
        gate0_initial_pitch_rad=pitch_argument,
    )

    accepted_safety = inner.memory.guidance_memory.safety == safety
    if not accepted_safety and update_valid:
        if inner.proposal.source_frame is not None or not inner.proposal.is_exact_zero:
            raise AssertionError("rejected safety transition produced sourced intent")
        outer_failure = "guidance_safety_not_accepted"
        update_valid = False
        controller_provenance = None
        entry_pitch_provenance = None
    if outer_failure is not None and (
        inner.proposal.source_frame is not None or not inner.proposal.is_exact_zero
    ):
        raise AssertionError("invalid outer evidence reached a sourced proposal")

    source_accepted = bool(
        active_update is not None
        and _inner_guidance_binds_state(inner, active_update.state)
    )
    if controller_provenance is not None and not source_accepted:
        controller_provenance = None

    accepted_attitude = (
        active_update.evidence.target_attitude.attitude
        if (
            active_update is not None
            and update_valid
            and accepted_safety
            and controller_provenance is not None
        )
        else None
    )
    new_attitude_lineage = accepted_attitude
    if (
        new_attitude_lineage is None
        and entry_pitch_provenance is not None
        and accepted_safety
    ):
        new_attitude_lineage = entry_pitch_provenance.attitude
    next_memory = _next_outer_memory(
        memory,
        inner.memory,
        new_attitude_lineage=new_attitude_lineage,
        entry_pitch_provenance=entry_pitch_provenance,
    )
    return VQ2Wave3ImuAdapterTransition(
        memory=next_memory,
        inner_transition=inner,
        active_update=active_update,
        accepted_attitude=accepted_attitude,
        controller_attitude_provenance=controller_provenance,
        outer_withholding_reason=outer_failure,
    )


def _outer_update_failure(
    memory: Optional[VQ2Wave3ImuAdapterMemory],
    safety: VQ2SafetyGuidanceInput,
    *,
    active_update: VQ2ImuCorrelatedEstimatorUpdate,
    tick: ControllerTickInput,
) -> Optional[str]:
    if type(active_update.estimator_update) is not RelativeEstimatorUpdate:
        return "imu_correlated_update_malformed"
    if type(active_update.estimator_update.measurement_accepted) is not bool:
        return "imu_correlated_update_malformed"
    if type(active_update.state) is not RelativeGateStateV1:
        return "imu_correlated_update_malformed"
    if type(active_update.evidence) is not VQ2DerotationEvidence:
        return "imu_correlation_evidence_malformed"
    evidence = active_update.evidence
    if (
        type(evidence.observation) is not GateObservationV1
        or type(evidence.prediction_target) is not RelativePredictionTarget
        or type(evidence.capture_attitude) is not VQ2AttitudeDerotationInput
        or type(evidence.target_attitude) is not VQ2AttitudeDerotationInput
        or type(evidence.target_attitude.attitude) is not VQ2TimestampedAttitude
        or type(evidence.calibration) is not VQ2CameraToBodyCalibration
        or type(evidence.model) is not VQ2DerotationModel
    ):
        return "imu_correlation_evidence_malformed"
    try:
        active_update.validate_integrity()
    except (AttributeError, TypeError, ValueError):
        return "imu_correlation_evidence_malformed"
    if not active_update.current_observation_accepted:
        return "current_observation_not_accepted"
    state = active_update.state
    attitude = evidence.target_attitude.attitude
    authority = safety.authority
    if state.track_role is not TrackRole.ACTIVE:
        return "active_role_mismatch"
    if state.authority != authority or evidence.observation.authority != authority:
        return "imu_correlation_authority_mismatch"
    if (
        attitude.session_id != authority.session_id
        or attitude.reset_epoch != authority.reset_epoch
    ):
        return "attitude_epoch_mismatch"
    if (
        attitude.host_clock_id != safety.evaluation_host_clock_id
        or attitude.host_clock_id != tick.host_clock_id
        or attitude.host_clock_id != state.timing.host_clock_id
    ):
        return "attitude_host_clock_mismatch"
    if evidence.prediction_target.prediction_time_monotonic_ns != (
        state.timing.prediction_time_monotonic_ns
    ):
        return "imu_correlation_target_mismatch"
    chronology_failure = _attitude_chronology_failure(
        None if memory is None else memory.last_attitude,
        attitude,
    )
    if chronology_failure is not None:
        return chronology_failure
    return _propagated_attitude_failure(
        evidence,
        target_host_clock_id=tick.host_clock_id,
        target_monotonic_ns=tick.proposal_monotonic_ns,
    )


def _propagated_attitude_failure(
    evidence: VQ2DerotationEvidence,
    *,
    target_host_clock_id: str,
    target_monotonic_ns: int,
) -> Optional[str]:
    attitude_input = evidence.target_attitude
    attitude = attitude_input.attitude
    if target_host_clock_id != attitude.host_clock_id:
        return "attitude_host_clock_mismatch"
    if target_monotonic_ns < attitude.receive_monotonic_ns:
        return "attitude_from_future"
    extrapolation_ns = target_monotonic_ns - attitude.receive_monotonic_ns
    if extrapolation_ns > HARD_MAX_CONTROLLER_ATTITUDE_EXTRAPOLATION_NS:
        return "attitude_extrapolation_exceeded"
    effective_age_ns = extrapolation_ns + attitude_input.host_time_uncertainty_ns
    if effective_age_ns > HARD_MAX_CONTROLLER_ATTITUDE_EFFECTIVE_AGE_NS:
        return "attitude_effective_age_exceeded"
    uncertainty = _propagated_angular_uncertainty_rad(
        evidence,
        extrapolation_ns,
    )
    if uncertainty > HARD_MAX_CONTROLLER_ATTITUDE_UNCERTAINTY_RAD:
        return "attitude_uncertainty_exceeded"
    return None


def _attitude_chronology_failure(
    previous: Optional[VQ2TimestampedAttitude],
    current: VQ2TimestampedAttitude,
) -> Optional[str]:
    if previous is None or current == previous:
        return None
    if (
        current.session_id != previous.session_id
        or current.reset_epoch != previous.reset_epoch
    ):
        return None
    if current.host_clock_id != previous.host_clock_id:
        return "attitude_clock_relabelled"
    if current.stream_id != previous.stream_id:
        return "attitude_stream_relabelled"
    if current.receive_monotonic_ns <= previous.receive_monotonic_ns:
        return "attitude_receive_time_regressed_or_relabelled"
    if current.generation < previous.generation:
        return "attitude_generation_regressed"
    if current.generation > previous.generation:
        return None
    if current.sample_sequence <= previous.sample_sequence:
        return "attitude_sequence_regressed_or_relabelled"
    if current.source_time_us <= previous.source_time_us:
        return "attitude_source_time_regressed_or_relabelled"
    return None


def _is_supported_safety_mapping(safety: VQ2SafetyGuidanceInput) -> bool:
    if safety.race_state is not VQ2GuidanceRaceState.UNDERWAY:
        return False
    return bool(
        (
            safety.authority.expected_gate_index == 0
            and safety.phase is VQ2GuidancePhase.APPROACH
        )
        or (
            safety.authority.expected_gate_index == 1
            and safety.phase is VQ2GuidancePhase.ALIGN
        )
    )


def _needs_gate0_entry_provenance(
    memory: Optional[VQ2Wave3ImuAdapterMemory],
    safety: VQ2SafetyGuidanceInput,
) -> bool:
    identity = _gate0_safety_identity(safety)
    if identity is None:
        return False
    previous_latch = None if memory is None else memory.inner_memory.gate0_pitch_latch
    return bool(
        previous_latch is None
        or _inner_latch_identity(previous_latch) != identity
    )


def _gate0_pitch_argument(
    memory: Optional[VQ2Wave3ImuAdapterMemory],
    safety: VQ2SafetyGuidanceInput,
    entry_provenance: Optional[VQ2Gate0PitchProvenance],
) -> Optional[float]:
    identity = _gate0_safety_identity(safety)
    if identity is None or entry_provenance is None:
        return None
    previous_latch = None if memory is None else memory.inner_memory.gate0_pitch_latch
    if previous_latch is not None and _inner_latch_identity(previous_latch) == identity:
        return None
    if _pitch_provenance_identity(entry_provenance) != identity:
        raise AssertionError("new Gate 0 pitch provenance has the wrong identity")
    return entry_provenance.initial_pitch_rad


def _next_outer_memory(
    previous: Optional[VQ2Wave3ImuAdapterMemory],
    inner_memory: VQ2Wave2AdapterMemory,
    *,
    new_attitude_lineage: Optional[VQ2TimestampedAttitude],
    entry_pitch_provenance: Optional[VQ2Gate0PitchProvenance],
) -> VQ2Wave3ImuAdapterMemory:
    safety = inner_memory.guidance_memory.safety
    retained_attitude = new_attitude_lineage
    if retained_attitude is None and previous is not None:
        candidate = previous.last_attitude
        if candidate is not None and (
            candidate.session_id == safety.authority.session_id
            and candidate.reset_epoch == safety.authority.reset_epoch
            and candidate.host_clock_id == safety.evaluation_host_clock_id
        ):
            retained_attitude = candidate

    latch = inner_memory.gate0_pitch_latch
    pitch_provenance: Optional[VQ2Gate0PitchProvenance] = None
    if latch is not None and latch.initial_pitch_rad is not None:
        if (
            previous is not None
            and previous.gate0_pitch_provenance is not None
            and _pitch_provenance_identity(previous.gate0_pitch_provenance)
            == _inner_latch_identity(latch)
            and previous.gate0_pitch_provenance.initial_pitch_rad
            == latch.initial_pitch_rad
        ):
            pitch_provenance = previous.gate0_pitch_provenance
        elif (
            entry_pitch_provenance is not None
            and _pitch_provenance_identity(entry_pitch_provenance)
            == _inner_latch_identity(latch)
            and entry_pitch_provenance.initial_pitch_rad == latch.initial_pitch_rad
        ):
            pitch_provenance = entry_pitch_provenance
        else:
            raise ValueError("populated inner pitch latch lacks outer provenance")
    return VQ2Wave3ImuAdapterMemory(
        inner_memory=inner_memory,
        last_attitude=retained_attitude,
        gate0_pitch_provenance=pitch_provenance,
    )


def _make_propagated_attitude_provenance(
    evidence: VQ2DerotationEvidence,
    *,
    target_host_clock_id: str,
    target_monotonic_ns: int,
) -> VQ2PropagatedAttitudeProvenance:
    attitude_input = evidence.target_attitude
    attitude = attitude_input.attitude
    extrapolation_ns = target_monotonic_ns - attitude.receive_monotonic_ns
    return VQ2PropagatedAttitudeProvenance(
        evidence=evidence,
        target_host_clock_id=target_host_clock_id,
        target_monotonic_ns=target_monotonic_ns,
        propagation_model_id=CONTROLLER_ATTITUDE_PROPAGATION_MODEL_ID,
        extrapolation_ns=extrapolation_ns,
        effective_age_ns=(
            extrapolation_ns + attitude_input.host_time_uncertainty_ns
        ),
        angular_uncertainty_rad=_propagated_angular_uncertainty_rad(
            evidence,
            extrapolation_ns,
        ),
        orientation_body_to_ned_wxyz=attitude_input.orientation_at_host_time(
            target_host_clock_id,
            target_monotonic_ns,
        ),
        body_rates_rad_s=attitude.body_rates_rad_s,
    )


def _make_gate0_pitch_provenance(
    safety: VQ2SafetyGuidanceInput,
    attitude: VQ2PropagatedAttitudeProvenance,
) -> VQ2Gate0PitchProvenance:
    authority = safety.authority
    return VQ2Gate0PitchProvenance(
        session_id=authority.session_id,
        reset_epoch=authority.reset_epoch,
        gate_epoch=authority.gate_epoch,
        expected_gate_index=authority.expected_gate_index,
        host_clock_id=safety.evaluation_host_clock_id,
        phase_started_monotonic_ns=safety.phase_started_monotonic_ns,
        attitude_provenance=attitude,
        initial_pitch_rad=_pitch_from_orientation(
            attitude.orientation_body_to_ned_wxyz
        ),
    )


def _propagated_angular_uncertainty_rad(
    evidence: VQ2DerotationEvidence,
    extrapolation_ns: int,
) -> float:
    attitude_input = evidence.target_attitude
    rates = attitude_input.attitude.body_rates_rad_s
    rate_norm = math.sqrt(sum(component * component for component in rates))
    host_uncertainty_s = attitude_input.host_time_uncertainty_ns * 1e-9
    extrapolation_s = extrapolation_ns * 1e-9
    model_rate_uncertainty = evidence.model.angular_rate_uncertainty_rad_s
    return float(
        attitude_input.orientation_uncertainty_rad
        + rate_norm * host_uncertainty_s
        + model_rate_uncertainty * (host_uncertainty_s + extrapolation_s)
    )


def _pitch_from_orientation(
    orientation_body_to_ned_wxyz: tuple[float, float, float, float],
) -> float:
    w, x, y, z = orientation_body_to_ned_wxyz
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return float(math.asin(sin_pitch))


def _guidance_source_matches_state(
    source: VQ2GuidanceSource,
    state: RelativeGateStateV1,
) -> bool:
    timing = state.timing
    return bool(
        source.host_clock_id == timing.host_clock_id
        and source.decision_time_monotonic_ns
        == timing.decision_time_monotonic_ns
        and source.prediction_time_monotonic_ns
        == timing.prediction_time_monotonic_ns
        and source.source_frame == timing.source_frame
        and source.source_frame_publication_sequence
        == timing.source_frame_publication_sequence
        and source.source_frame_publish_monotonic_ns
        == timing.source_frame_publish_monotonic_ns
        and source.tracker_id == state.tracker_id
        and source.track_role is state.track_role
        and source.state_sequence == state.state_sequence
        and source.measurement_update_sequence
        == state.measurement_update_sequence
        and source.source_candidate_id == state.source_candidate_id
    )


def _inner_guidance_binds_state(
    inner: VQ2Wave2AdapterTransition,
    state: RelativeGateStateV1,
) -> bool:
    source = inner.decision.source
    return bool(
        source is not None
        and _guidance_source_matches_state(source, state)
        and inner.memory.guidance_memory.active_source == source
    )


def _validate_pitch_provenance(
    latch: Optional[VQ2Gate0PitchLatch],
    provenance: Optional[VQ2Gate0PitchProvenance],
) -> None:
    if latch is None or latch.initial_pitch_rad is None:
        if provenance is not None:
            raise ValueError("outer pitch provenance requires a populated inner latch")
        return
    if provenance is None:
        raise ValueError("populated inner pitch latch requires outer provenance")
    if _inner_latch_identity(latch) != _pitch_provenance_identity(provenance):
        raise ValueError("inner and outer Gate 0 pitch identities differ")
    if latch.initial_pitch_rad != provenance.initial_pitch_rad:
        raise ValueError("inner and outer Gate 0 pitch values differ")


def _gate0_safety_identity(
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


def _inner_latch_identity(
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


def _pitch_provenance_identity(
    provenance: VQ2Gate0PitchProvenance,
) -> tuple[str, int, int, int, str, int]:
    return (
        provenance.session_id,
        provenance.reset_epoch,
        provenance.gate_epoch,
        provenance.expected_gate_index,
        provenance.host_clock_id,
        provenance.phase_started_monotonic_ns,
    )


__all__ = [
    "CONTROLLER_ATTITUDE_PROPAGATION_MODEL_ID",
    "HARD_MAX_CONTROLLER_ATTITUDE_EFFECTIVE_AGE_NS",
    "HARD_MAX_CONTROLLER_ATTITUDE_EXTRAPOLATION_NS",
    "HARD_MAX_CONTROLLER_ATTITUDE_UNCERTAINTY_RAD",
    "VQ2Gate0PitchProvenance",
    "VQ2PropagatedAttitudeProvenance",
    "VQ2Wave3ImuAdapterMemory",
    "VQ2Wave3ImuAdapterTransition",
    "step_vq2_wave3_imu_adapter",
]
