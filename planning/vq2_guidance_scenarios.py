"""Deterministic generated scenarios for offline VQ2 guidance evaluation.

The evaluator in this module creates contract-valid synthetic values in memory.
It does not read a capture, replay a flight, connect to FlightSim, model vehicle
dynamics, or provide powered/race evidence.  Its narrow purpose is to make the
mapless guidance authority and uncertainty semantics easy to regression-test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Optional

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
)
from planning.vq2_guidance import (
    VQ2GuidanceObjectiveKind,
    VQ2GuidancePhase,
    VQ2GuidanceRaceState,
    VQ2GuidanceTransition,
    VQ2GuidanceWithholdingReason,
    VQ2SafetyGuidanceInput,
    step_vq2_guidance,
)


SYNTHETIC_GUIDANCE_SCOPE = (
    "deterministic_generated_image_space_unit_scenario_nonpowered;"
    "not_replay_not_simulator_not_passage_evidence"
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


@dataclass(frozen=True, slots=True)
class VQ2SyntheticGuidanceStep:
    label: str
    expected_gate_index: int
    phase: str
    phase_started_monotonic_ns: int
    objective_kind: str
    objective_permitted: bool
    withholding_reason: Optional[str]
    source_tracker_id: Optional[str]

    def to_primitive(self) -> dict[str, object]:
        return {
            "label": self.label,
            "expected_gate_index": self.expected_gate_index,
            "phase": self.phase,
            "phase_started_monotonic_ns": self.phase_started_monotonic_ns,
            "objective_kind": self.objective_kind,
            "objective_permitted": self.objective_permitted,
            "withholding_reason": self.withholding_reason,
            "source_tracker_id": self.source_tracker_id,
        }


@dataclass(frozen=True, slots=True)
class VQ2SyntheticGuidanceReport:
    evidence_scope: str
    gate0_visual_phase_non_regression: bool
    gate0_same_snapshot_phase_change_rejected: bool
    gate0_phase_start_stable: bool
    gate0_phase_start_renewal_rejected: bool
    gate0_forward_phase_accepted: bool
    gate1_shadow_isolated: bool
    gate1_high_uncertainty_withheld: bool
    gate1_low_uncertainty_recenter_permitted: bool
    steps: tuple[VQ2SyntheticGuidanceStep, ...]
    digest_sha256: str

    @property
    def all_checks_passed(self) -> bool:
        return all(
            (
                self.gate0_visual_phase_non_regression,
                self.gate0_same_snapshot_phase_change_rejected,
                self.gate0_phase_start_stable,
                self.gate0_phase_start_renewal_rejected,
                self.gate0_forward_phase_accepted,
                self.gate1_shadow_isolated,
                self.gate1_high_uncertainty_withheld,
                self.gate1_low_uncertainty_recenter_permitted,
            )
        )


def evaluate_synthetic_vq2_guidance_scenario() -> VQ2SyntheticGuidanceReport:
    """Run the small deterministic authority/uncertainty scenario."""

    records: list[VQ2SyntheticGuidanceStep] = []

    initial_authority = _authority(0)
    initial_safety = _safety(
        initial_authority,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    transition = step_vq2_guidance(
        None,
        initial_safety,
        active_state=None,
    )
    records.append(_record("gate0_not_underway_initialization", transition))
    gate0_phase_start_ns = transition.memory.safety.phase_started_monotonic_ns

    gate0_acquire_authority = _authority(1)
    acquire_safety = _safety(
        gate0_acquire_authority,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_started_monotonic_ns=gate0_phase_start_ns,
    )
    transition = step_vq2_guidance(
        transition.memory,
        acquire_safety,
        active_state=_state(
            gate0_acquire_authority,
            state_sequence=1,
            frame_id=1,
            tracker_id="synthetic-active-gate-0",
        ),
    )
    records.append(_record("gate0_acquire", transition))

    visual_safety = replace(
        acquire_safety,
        evaluation_monotonic_ns=acquire_safety.evaluation_monotonic_ns + 100_000,
    )
    visual_update = step_vq2_guidance(
        transition.memory,
        visual_safety,
        active_state=_state(
            gate0_acquire_authority,
            state_sequence=2,
            frame_id=2,
            tracker_id="synthetic-active-gate-0",
            publication_offset=2,
            bearing_norm=(0.0, 0.0),
        ),
    )
    records.append(_record("gate0_visual_update", visual_update))
    gate0_visual_phase_non_regression = bool(
        visual_update.memory.safety.phase is VQ2GuidancePhase.ACQUIRE
        and not visual_update.decision.objective_permitted
    )
    gate0_phase_start_stable = bool(
        transition.memory.safety.phase_started_monotonic_ns
        == gate0_phase_start_ns
        and visual_update.memory.safety.phase_started_monotonic_ns
        == gate0_phase_start_ns
        and visual_update.decision.phase_started_monotonic_ns
        == gate0_phase_start_ns
    )

    same_snapshot_evaluation_ns = visual_safety.evaluation_monotonic_ns + 1
    same_snapshot_phase_change = step_vq2_guidance(
        visual_update.memory,
        replace(
            visual_safety,
            phase=VQ2GuidancePhase.ALIGN,
            evaluation_monotonic_ns=same_snapshot_evaluation_ns,
            phase_started_monotonic_ns=same_snapshot_evaluation_ns,
        ),
        active_state=None,
    )
    records.append(_record("gate0_same_snapshot_align_rejected", same_snapshot_phase_change))
    gate0_same_snapshot_phase_change_rejected = bool(
        same_snapshot_phase_change.memory == visual_update.memory
        and same_snapshot_phase_change.decision.objective_kind
        is VQ2GuidanceObjectiveKind.HOLD
    )

    align_authority = _authority(2)
    phase_start_renewal = step_vq2_guidance(
        visual_update.memory,
        _safety(align_authority, VQ2GuidancePhase.ACQUIRE),
        active_state=None,
    )
    records.append(_record("gate0_phase_start_renewal_rejected", phase_start_renewal))
    gate0_phase_start_renewal_rejected = bool(
        phase_start_renewal.memory == visual_update.memory
        and phase_start_renewal.decision.withholding_reason
        is VQ2GuidanceWithholdingReason.SAFETY_PHASE_START_REJECTED
        and phase_start_renewal.decision.phase_started_monotonic_ns
        == gate0_phase_start_ns
    )

    transition = step_vq2_guidance(
        visual_update.memory,
        _safety(align_authority, VQ2GuidancePhase.ALIGN),
        active_state=_state(
            align_authority,
            state_sequence=3,
            frame_id=3,
            tracker_id="synthetic-active-gate-0",
        ),
    )
    records.append(_record("gate0_forward_align", transition))
    gate0_forward_phase_accepted = bool(
        transition.memory.safety.phase is VQ2GuidancePhase.ALIGN
        and transition.decision.objective_permitted
    )

    for step, phase in enumerate(
        (
            VQ2GuidancePhase.APPROACH,
            VQ2GuidancePhase.COMMIT,
            VQ2GuidancePhase.CONFIRMATION,
        ),
        start=3,
    ):
        authority = _authority(step)
        transition = step_vq2_guidance(
            transition.memory,
            _safety(authority, phase),
            active_state=_state(
                authority,
                state_sequence=step + 2,
                frame_id=step + 2,
                tracker_id="synthetic-active-gate-0",
            ),
        )
        records.append(_record(f"gate0_{phase.value}", transition))

    gate1_post_authority = _authority(6, gate_epoch=1, gate_index=1)
    transition = step_vq2_guidance(
        transition.memory,
        _safety(
            gate1_post_authority,
            VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
        ),
        active_state=None,
    )
    records.append(_record("gate1_post_credit_reacquire", transition))

    gate1_acquire_authority = _authority(7, gate_epoch=1, gate_index=1)
    transition = step_vq2_guidance(
        transition.memory,
        _safety(gate1_acquire_authority, VQ2GuidancePhase.ACQUIRE),
        active_state=None,
    )
    records.append(_record("gate1_acquire", transition))

    gate1_align_authority = _authority(8, gate_epoch=1, gate_index=1)
    gate1_align_safety = _safety(
        gate1_align_authority,
        VQ2GuidancePhase.ALIGN,
    )
    high_uncertainty = _state(
        gate1_align_authority,
        state_sequence=1,
        frame_id=101,
        tracker_id="synthetic-active-gate-1",
        candidate_id="gate1-high-uncertainty",
        bearing_norm=(0.45, -0.35),
        bearing_variance=0.09,
        clipping=FrameEdge.TOP,
        health=RelativeStateHealth.DEGRADED,
    )
    tempting_shadow = _state(
        gate1_align_authority,
        state_sequence=1,
        frame_id=201,
        tracker_id="synthetic-shadow-gate-1",
        candidate_id="gate1-shadow",
        bearing_norm=(0.0, 0.0),
        bearing_variance=0.0001,
        clipping=FrameEdge.NONE,
        health=RelativeStateHealth.HEALTHY,
        track_role=TrackRole.SHADOW,
        publication_offset=2,
    )
    high_result = step_vq2_guidance(
        transition.memory,
        gate1_align_safety,
        active_state=high_uncertainty,
        shadow_states=(tempting_shadow,),
    )
    records.append(_record("gate1_top_clipped_high_uncertainty", high_result))
    gate1_high_uncertainty_withheld = bool(
        not high_result.decision.objective_permitted
        and not high_result.decision.corridor_eligible
    )
    gate1_shadow_isolated = bool(
        gate1_high_uncertainty_withheld
        and high_result.decision.shadow_track_count == 1
        and high_result.decision.source is not None
        and high_result.decision.source.tracker_id == "synthetic-active-gate-1"
    )

    lower_uncertainty = _state(
        gate1_align_authority,
        state_sequence=2,
        frame_id=102,
        tracker_id="synthetic-active-gate-1",
        candidate_id="gate1-lower-uncertainty",
        bearing_norm=(0.45, -0.35),
        bearing_variance=0.0004,
        clipping=FrameEdge.TOP,
        health=RelativeStateHealth.DEGRADED,
        publication_offset=3,
    )
    low_result = step_vq2_guidance(
        high_result.memory,
        gate1_align_safety,
        active_state=lower_uncertainty,
    )
    records.append(_record("gate1_top_clipped_lower_uncertainty", low_result))
    gate1_low_uncertainty_recenter_permitted = bool(
        low_result.decision.objective_permitted
        and low_result.decision.objective_kind
        is VQ2GuidanceObjectiveKind.RECENTER_ACTIVE_GATE
        and low_result.decision.source is not None
        and low_result.decision.source.tracker_id == "synthetic-active-gate-1"
    )

    primitive = {
        "evidence_scope": SYNTHETIC_GUIDANCE_SCOPE,
        "checks": {
            "gate0_visual_phase_non_regression": gate0_visual_phase_non_regression,
            "gate0_same_snapshot_phase_change_rejected": (
                gate0_same_snapshot_phase_change_rejected
            ),
            "gate0_phase_start_stable": gate0_phase_start_stable,
            "gate0_phase_start_renewal_rejected": (
                gate0_phase_start_renewal_rejected
            ),
            "gate0_forward_phase_accepted": gate0_forward_phase_accepted,
            "gate1_shadow_isolated": gate1_shadow_isolated,
            "gate1_high_uncertainty_withheld": gate1_high_uncertainty_withheld,
            "gate1_low_uncertainty_recenter_permitted": (
                gate1_low_uncertainty_recenter_permitted
            ),
        },
        "steps": [record.to_primitive() for record in records],
    }
    encoded = json.dumps(
        primitive,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return VQ2SyntheticGuidanceReport(
        evidence_scope=SYNTHETIC_GUIDANCE_SCOPE,
        gate0_visual_phase_non_regression=gate0_visual_phase_non_regression,
        gate0_same_snapshot_phase_change_rejected=(
            gate0_same_snapshot_phase_change_rejected
        ),
        gate0_phase_start_stable=gate0_phase_start_stable,
        gate0_phase_start_renewal_rejected=(
            gate0_phase_start_renewal_rejected
        ),
        gate0_forward_phase_accepted=gate0_forward_phase_accepted,
        gate1_shadow_isolated=gate1_shadow_isolated,
        gate1_high_uncertainty_withheld=gate1_high_uncertainty_withheld,
        gate1_low_uncertainty_recenter_permitted=(
            gate1_low_uncertainty_recenter_permitted
        ),
        steps=tuple(records),
        digest_sha256=digest,
    )


def _record(label: str, transition: VQ2GuidanceTransition) -> VQ2SyntheticGuidanceStep:
    decision = transition.decision
    return VQ2SyntheticGuidanceStep(
        label=label,
        expected_gate_index=decision.authority.expected_gate_index,
        phase=decision.phase.value,
        phase_started_monotonic_ns=decision.phase_started_monotonic_ns,
        objective_kind=decision.objective_kind.value,
        objective_permitted=decision.objective_permitted,
        withholding_reason=(
            None
            if decision.withholding_reason is None
            else decision.withholding_reason.value
        ),
        source_tracker_id=(
            None if decision.source is None else decision.source.tracker_id
        ),
    )


def _authority(
    step: int,
    *,
    gate_epoch: int = 0,
    gate_index: int = 0,
) -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id="synthetic-guidance-session",
        reset_epoch=0,
        gate_epoch=gate_epoch,
        expected_gate_index=gate_index,
        race_status_sequence=500 + step,
        race_status_boot_ms=5_000 + step * 20,
        camera_host_clock_id="synthetic-host-clock",
        camera_stream_id="synthetic-camera",
        camera_generation=1,
        frame_publication_sequence_not_before=10_000 + step * 10,
        frame_publish_monotonic_ns_not_before=3_000_000_000 + step * 1_000_000,
    )


def _safety(
    authority: GateAuthorityEpochV1,
    phase: VQ2GuidancePhase,
    race_state: VQ2GuidanceRaceState = VQ2GuidanceRaceState.UNDERWAY,
    *,
    phase_started_monotonic_ns: Optional[int] = None,
) -> VQ2SafetyGuidanceInput:
    evaluation_ns = authority.frame_publish_monotonic_ns_not_before + 50_000_000
    return VQ2SafetyGuidanceInput(
        authority=authority,
        phase=phase,
        race_state=race_state,
        evaluation_host_clock_id=authority.camera_host_clock_id,
        evaluation_monotonic_ns=evaluation_ns,
        phase_started_monotonic_ns=(
            evaluation_ns
            if phase_started_monotonic_ns is None
            else phase_started_monotonic_ns
        ),
    )


def _state(
    authority: GateAuthorityEpochV1,
    *,
    state_sequence: int,
    frame_id: int,
    tracker_id: str,
    candidate_id: Optional[str] = None,
    bearing_norm: tuple[float, float] = (0.02, -0.01),
    bearing_variance: float = 0.0001,
    clipping: FrameEdge = FrameEdge.NONE,
    health: RelativeStateHealth = RelativeStateHealth.HEALTHY,
    track_role: TrackRole = TrackRole.ACTIVE,
    publication_offset: int = 1,
) -> RelativeGateStateV1:
    publication_sequence = (
        authority.frame_publication_sequence_not_before + publication_offset
    )
    publish_ns = (
        authority.frame_publish_monotonic_ns_not_before
        + publication_offset * 100_000
    )
    frame = FrameIdentityV1(
        stream_id=authority.camera_stream_id,
        generation=authority.camera_generation,
        frame_id=frame_id,
    )
    timing = PredictionTimeV1(
        host_clock_id=authority.camera_host_clock_id,
        source_frame=frame,
        source_frame_publication_sequence=publication_sequence,
        source_frame_publish_monotonic_ns=publish_ns,
        measurement_time_monotonic_ns=publish_ns - 1_000,
        measurement_time_basis=MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY,
        measurement_time_model_id=None,
        measurement_uncertainty_ns=1_000,
        decision_time_monotonic_ns=publish_ns + state_sequence * 1_000,
        prediction_time_monotonic_ns=publish_ns + state_sequence * 1_000,
        prediction_basis=PredictionBasis.DECISION_TIME,
        delay_model_id=None,
        delay_uncertainty_ns=0,
    )
    diagonal = (
        bearing_variance,
        bearing_variance,
        0.0004,
        0.0004,
        0.0004,
        0.0004,
    )
    covariance = FeatureCovarianceV1(
        model_id="synthetic-guidance-scenario-diagonal",
        feature_order=_STATE_FEATURE_ORDER,
        matrix=tuple(
            tuple(value if row == column else 0.0 for column in range(6))
            for row, value in enumerate(diagonal)
        ),
    )
    visibility = _ALL_EDGES & ~clipping
    return RelativeGateStateV1(
        timing=timing,
        authority=authority,
        tracker_id=tracker_id,
        state_sequence=state_sequence,
        measurement_update_sequence=state_sequence,
        source_candidate_id=candidate_id or f"synthetic-candidate-{frame_id}",
        track_role=track_role,
        bearing_norm=bearing_norm,
        bearing_rate_norm_s=(0.01, -0.01),
        log_scale=-0.3,
        expansion_rate_s=0.2,
        covariance=covariance,
        metric_position_body_frd_m=None,
        metric_velocity_body_frd_m_s=None,
        metric_gate_orientation_body_frd_xyzw=None,
        metric_covariance=None,
        last_clipping=clipping,
        outer_visibility=visibility,
        inner_visibility=visibility,
        normalized_innovation_squared=1.0,
        innovation_gate_threshold=9.0,
        innovation_accepted=True,
        dropout_count=0,
        health=health,
        health_reason=(
            None if health is RelativeStateHealth.HEALTHY else "synthetic_clipping"
        ),
    )


__all__ = [
    "SYNTHETIC_GUIDANCE_SCOPE",
    "VQ2SyntheticGuidanceReport",
    "VQ2SyntheticGuidanceStep",
    "evaluate_synthetic_vq2_guidance_scenario",
]
