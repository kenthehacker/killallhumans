"""Cross-layer checks for the pure, offline Wave 2 adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from competition.adapter import CameraFrame
from competition.vq2_contracts import (
    FeatureCovarianceV1,
    FrameEdge,
    FrameIdentityV1,
    FrameTimingV1,
    GateAuthorityEpochV1,
    MeasurementTimeBasis,
    PredictionBasis,
    PredictionTimeV1,
    RelativeGateStateV1,
    RelativeStateHealth,
    TrackRole,
    validate_command_proposal_source,
    validate_relative_gate_state_source,
)
from competition.vq2_controller import ControllerAttitudeInput, ControllerTickInput
from competition.vq2_runtime import LatestFrameCursorV1
from competition.vq2_vision import VQ2VisionSnapshot
from competition.vq2_wave2_adapter import (
    VQ2Wave2AdapterMemory,
    VQ2Wave2AdapterTransition,
    step_vq2_wave2_adapter,
)
from estimation.vq2_relative_estimator import (
    MissingApertureScaleError,
    RelativePredictionTarget,
    VQ2RelativeGateEstimator,
)
from gate_detection.src.vq2_detector import VQ2GateDetector
from gate_detection.src.vq2_observation_adapter import (
    gate_detection_to_observation_v1,
    gate_detection_with_aperture_to_observation_v1,
)
from planning.vq2_guidance import (
    VQ2GuidanceObjectiveKind,
    VQ2GuidancePhase,
    VQ2GuidanceRaceState,
    VQ2SafetyGuidanceInput,
)


_HOST = "vq2-host-monotonic"
_STREAM = "camera0"
_SESSION = "wave2-offline-integration"
_ALL_EDGES = FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT | FrameEdge.BOTTOM
_STATE_FEATURE_ORDER = (
    "bearing_x_norm",
    "bearing_y_norm",
    "log_scale",
    "bearing_rate_x_norm_s",
    "bearing_rate_y_norm_s",
    "expansion_rate_s",
)


def _authority(
    sequence: int,
    *,
    gate_epoch: int = 0,
    gate_index: int = 0,
    reset_epoch: int = 2,
) -> GateAuthorityEpochV1:
    return GateAuthorityEpochV1(
        session_id=_SESSION,
        reset_epoch=reset_epoch,
        gate_epoch=gate_epoch,
        expected_gate_index=gate_index,
        race_status_sequence=100 + sequence,
        race_status_boot_ms=1_000 + sequence * 100,
        camera_host_clock_id=_HOST,
        camera_stream_id=_STREAM,
        camera_generation=4,
        frame_publication_sequence_not_before=1_000 + sequence * 10,
        frame_publish_monotonic_ns_not_before=(
            1_000_000_000 + sequence * 100_000_000
        ),
    )


def _safety(
    sequence: int,
    phase: VQ2GuidancePhase,
    race_state: VQ2GuidanceRaceState,
    *,
    phase_start_ns: int | None = None,
    gate_epoch: int = 0,
    gate_index: int = 0,
) -> VQ2SafetyGuidanceInput:
    authority = _authority(
        sequence,
        gate_epoch=gate_epoch,
        gate_index=gate_index,
    )
    evaluation_ns = authority.frame_publish_monotonic_ns_not_before + 50_000_000
    return VQ2SafetyGuidanceInput(
        authority=authority,
        phase=phase,
        race_state=race_state,
        evaluation_host_clock_id=_HOST,
        evaluation_monotonic_ns=evaluation_ns,
        phase_started_monotonic_ns=(
            evaluation_ns if phase_start_ns is None else phase_start_ns
        ),
    )


def _state(
    safety: VQ2SafetyGuidanceInput,
    *,
    tracker_id: str,
    state_sequence: int,
    frame_id: int,
    bearing_norm: tuple[float, float] = (0.05, -0.04),
    bearing_rate_norm_s: tuple[float, float] = (0.02, -0.02),
    variance: tuple[float, float, float, float, float, float] = (
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
    ),
    clipping: FrameEdge = FrameEdge.NONE,
    health: RelativeStateHealth = RelativeStateHealth.HEALTHY,
    track_role: TrackRole = TrackRole.ACTIVE,
) -> RelativeGateStateV1:
    authority = safety.authority
    publish_ns = safety.evaluation_monotonic_ns - 30_000_000
    decision_ns = publish_ns + 5_000_000
    covariance = FeatureCovarianceV1(
        model_id="wave2-adapter-diagonal",
        feature_order=_STATE_FEATURE_ORDER,
        matrix=tuple(
            tuple(value if row == column else 0.0 for column in range(6))
            for row, value in enumerate(variance)
        ),
    )
    return RelativeGateStateV1(
        timing=PredictionTimeV1(
            host_clock_id=_HOST,
            source_frame=FrameIdentityV1(_STREAM, 4, frame_id),
            source_frame_publication_sequence=(
                authority.frame_publication_sequence_not_before + 1
            ),
            source_frame_publish_monotonic_ns=publish_ns,
            measurement_time_monotonic_ns=publish_ns - 5_000_000,
            measurement_time_basis=(
                MeasurementTimeBasis.CAMERA_FINAL_PACKET_PROXY
            ),
            measurement_time_model_id=None,
            measurement_uncertainty_ns=1_000_000,
            decision_time_monotonic_ns=decision_ns,
            prediction_time_monotonic_ns=decision_ns,
            prediction_basis=PredictionBasis.DECISION_TIME,
            delay_model_id=None,
            delay_uncertainty_ns=0,
        ),
        authority=authority,
        tracker_id=tracker_id,
        state_sequence=state_sequence,
        measurement_update_sequence=state_sequence,
        source_candidate_id=f"candidate-{frame_id}",
        track_role=track_role,
        bearing_norm=bearing_norm,
        bearing_rate_norm_s=bearing_rate_norm_s,
        log_scale=-0.3,
        expansion_rate_s=0.2,
        covariance=covariance,
        metric_position_body_frd_m=None,
        metric_velocity_body_frd_m_s=None,
        metric_gate_orientation_body_frd_xyzw=None,
        metric_covariance=None,
        last_clipping=clipping,
        outer_visibility=_ALL_EDGES & ~clipping,
        inner_visibility=_ALL_EDGES & ~clipping,
        normalized_innovation_squared=1.0,
        innovation_gate_threshold=9.0,
        innovation_accepted=True,
        dropout_count=0,
        health=health,
        health_reason=(
            None if health is RelativeStateHealth.HEALTHY else "clipped support"
        ),
    )


def _attitude() -> ControllerAttitudeInput:
    return ControllerAttitudeInput(
        orientation_body_to_world_wxyz=(1.0, 0.0, 0.0, 0.0),
        body_rates_rad_s=(0.0, 0.0, 0.0),
    )


def _tick(
    safety: VQ2SafetyGuidanceInput,
    state: RelativeGateStateV1 | None = None,
    *,
    proposal_offset_ns: int = 10_000_000,
    proposal_id: int = 1,
) -> ControllerTickInput:
    proposal_ns = safety.evaluation_monotonic_ns + proposal_offset_ns
    return ControllerTickInput(
        proposal_id=proposal_id,
        control_tick_id=proposal_id,
        host_clock_id=_HOST,
        proposal_monotonic_ns=proposal_ns,
        control_tick_deadline_monotonic_ns=proposal_ns + 10_000_000,
        minimum_state_decision_monotonic_ns=(
            0 if state is None else state.timing.decision_time_monotonic_ns
        ),
        minimum_state_sequence=0 if state is None else state.state_sequence,
        expected_phase_started_monotonic_ns=safety.phase_started_monotonic_ns,
        minimum_phase_evaluation_monotonic_ns=safety.evaluation_monotonic_ns,
        expected_authority=safety.authority,
    )


def _step(
    memory: VQ2Wave2AdapterMemory | None,
    safety: VQ2SafetyGuidanceInput,
    *,
    state: RelativeGateStateV1 | None = None,
    tick: ControllerTickInput | None = None,
    pitch: float | None = None,
    attitude: ControllerAttitudeInput | None = None,
):
    return step_vq2_wave2_adapter(
        memory,
        safety,
        active_state=state,
        attitude=attitude,
        tick=_tick(safety, state) if tick is None else tick,
        gate0_initial_pitch_rad=pitch,
    )


def _enter_gate0_approach(*, with_state: bool, pitch: float | None = -0.1):
    initial = _safety(
        0,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    transition = _step(None, initial)
    go = _safety(
        1,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=initial.phase_started_monotonic_ns,
    )
    transition = _step(transition.memory, go)
    align = _safety(
        2,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = _step(transition.memory, align)
    approach = _safety(
        3,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    state = (
        _state(approach, tracker_id="active-gate-0", state_sequence=1, frame_id=31)
        if with_state
        else None
    )
    transition = _step(
        transition.memory,
        approach,
        state=state,
        pitch=pitch,
        attitude=_attitude() if with_state else None,
    )
    return transition, approach, state


def _enter_gate1_align(*, with_state: bool = True):
    transition, approach, _ = _enter_gate0_approach(with_state=True)
    commit = _safety(
        4,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = _step(transition.memory, commit)
    confirmation = _safety(
        5,
        VQ2GuidancePhase.CONFIRMATION,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    transition = _step(transition.memory, confirmation)
    credited = _safety(
        6,
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    transition = _step(transition.memory, credited)
    acquire = _safety(
        7,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    transition = _step(transition.memory, acquire)
    align = _safety(
        8,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    state = (
        _state(
            align,
            tracker_id="active-gate-1",
            state_sequence=1,
            frame_id=81,
            bearing_norm=(0.40, -0.30),
            bearing_rate_norm_s=(0.05, -0.04),
            variance=(0.0004,) * 6,
            clipping=FrameEdge.TOP,
            health=RelativeStateHealth.DEGRADED,
        )
        if with_state
        else None
    )
    transition = _step(
        transition.memory,
        align,
        state=state,
        attitude=_attitude() if with_state else None,
    )
    return transition, align, state


def _all_source_fields_are_none(proposal) -> bool:
    return all(
        getattr(proposal, name) is None
        for name in (
            "source_state_decision_monotonic_ns",
            "source_state_prediction_monotonic_ns",
            "source_frame",
            "source_frame_publication_sequence",
            "source_frame_publish_monotonic_ns",
            "source_tracker_id",
            "source_track_role",
            "source_state_sequence",
            "source_measurement_update_sequence",
            "source_candidate_id",
        )
    )


def _gate_image() -> np.ndarray:
    image = np.full((160, 200, 3), 18, dtype=np.uint8)
    hsv = np.uint8([[[165, 100, 250]]])
    color = tuple(
        int(channel) for channel in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    )
    cv2.rectangle(image, (50, 30), (150, 130), color, -1)
    cv2.rectangle(image, (60, 40), (140, 120), (18, 18, 18), -1)
    image.flags.writeable = False
    return image


def _timed_gate_detection(
    safety: VQ2SafetyGuidanceInput,
    *,
    frame_id: int = 301,
    publication_offset: int = 1,
):
    publish_ns = safety.evaluation_monotonic_ns - 25_000_000
    timing = FrameTimingV1(
        identity=FrameIdentityV1(_STREAM, 4, frame_id),
        camera_source_time_ns=123_456_000 + frame_id,
        host_clock_id=_HOST,
        publication_sequence=(
            safety.authority.frame_publication_sequence_not_before
            + publication_offset
        ),
        first_unique_packet_monotonic_ns=publish_ns - 15_000_000,
        final_unique_packet_monotonic_ns=publish_ns - 10_000_000,
        reassembly_complete_monotonic_ns=publish_ns - 10_000_000,
        decode_start_monotonic_ns=publish_ns - 9_000_000,
        decode_end_monotonic_ns=publish_ns - 5_000_000,
        publish_monotonic_ns=publish_ns,
    )
    image = _gate_image()
    snapshot = VQ2VisionSnapshot(
        frame_id=frame_id,
        camera_frame=CameraFrame(
            timestamp_us=123_456,
            image=image,
            width=200,
            height=160,
        ),
        sim_time_ns=123_456_000 + frame_id,
        received_monotonic_s=timing.final_unique_packet_monotonic_ns / 1e9,
        generation=4,
        timing=timing,
    )
    selection = LatestFrameCursorV1(
        expected_host_clock_id=_HOST,
        expected_stream_id=_STREAM,
    ).select(snapshot)
    assert selection is not None
    detections = VQ2GateDetector(
        image_width=200,
        image_height=160,
        min_area=100,
    ).detect(image)
    assert len(detections) == 1
    return selection, detections[0]


def test_gate0_approach_returns_an_exact_source_bound_proposal() -> None:
    transition, safety, state = _enter_gate0_approach(with_state=True)

    assert transition.decision.objective_permitted
    assert transition.decision.objective_kind is VQ2GuidanceObjectiveKind.APPROACH_ACTIVE_GATE
    assert not transition.proposal.is_exact_zero
    assert transition.proposal.authority == safety.authority
    assert transition.memory.gate0_pitch_latch is not None
    assert transition.memory.gate0_pitch_latch.initial_pitch_rad == -0.1
    assert state is not None
    validate_command_proposal_source(transition.proposal, state)


def test_gate1_recenter_requires_legal_credit_and_a_distinct_tracker() -> None:
    transition, safety, state = _enter_gate1_align()

    assert transition.decision.objective_permitted
    assert transition.decision.objective_kind is VQ2GuidanceObjectiveKind.RECENTER_ACTIVE_GATE
    assert not transition.proposal.is_exact_zero
    assert transition.proposal.authority == safety.authority
    assert transition.proposal.uncertainty.limited
    assert transition.proposal.source_tracker_id == "active-gate-1"
    assert transition.memory.gate0_pitch_latch is None
    assert state is not None
    validate_command_proposal_source(transition.proposal, state)


@pytest.mark.parametrize("phase", ["gate0", "gate1"])
def test_coordinated_same_phase_start_renewal_is_rejected_by_owned_memory(phase) -> None:
    if phase == "gate0":
        previous, prior_safety, prior_state = _enter_gate0_approach(with_state=True)
        sequence, gate_epoch, gate_index, tracker = 4, 0, 0, "active-gate-0"
        state_sequence, frame_id, pitch = 2, 41, -0.1
    else:
        previous, prior_safety, prior_state = _enter_gate1_align()
        sequence, gate_epoch, gate_index, tracker = 9, 1, 1, "active-gate-1"
        state_sequence, frame_id, pitch = 2, 91, None
    renewed = _safety(
        sequence,
        prior_safety.phase,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=gate_epoch,
        gate_index=gate_index,
    )
    state = _state(
        renewed,
        tracker_id=tracker,
        state_sequence=state_sequence,
        frame_id=frame_id,
        clipping=(FrameEdge.NONE if phase == "gate0" else FrameEdge.TOP),
        health=(
            RelativeStateHealth.HEALTHY
            if phase == "gate0"
            else RelativeStateHealth.DEGRADED
        ),
    )

    rejected = _step(
        previous.memory,
        renewed,
        state=state,
        pitch=pitch,
        attitude=_attitude(),
    )

    assert rejected.memory == previous.memory
    assert rejected.decision.phase_started_monotonic_ns == prior_safety.phase_started_monotonic_ns
    assert rejected.proposal.is_exact_zero
    assert _all_source_fields_are_none(rejected.proposal)
    assert "phase_start" in rejected.proposal.reason
    assert rejected.proposal.authority == renewed.authority
    assert rejected.proposal.authority != rejected.decision.authority


@pytest.mark.parametrize(
    "field",
    [
        "expected_phase_started_monotonic_ns",
        "minimum_phase_evaluation_monotonic_ns",
        "minimum_state_decision_monotonic_ns",
        "minimum_state_sequence",
    ],
)
@pytest.mark.parametrize("delta", [-1, 1])
def test_supported_mapping_requires_exact_tick_watermarks(field, delta) -> None:
    entered, safety, _ = _enter_gate0_approach(with_state=False)
    state = _state(
        safety,
        tracker_id="active-gate-0",
        state_sequence=1,
        frame_id=31,
    )
    tick = _tick(safety, state)
    tick = replace(tick, **{field: getattr(tick, field) + delta})

    rejected = _step(
        entered.memory,
        safety,
        state=state,
        tick=tick,
        attitude=_attitude(),
    )

    assert rejected.decision.objective_permitted
    assert rejected.proposal.is_exact_zero
    assert _all_source_fields_are_none(rejected.proposal)
    assert "watermark_mismatch" in rejected.proposal.reason


def test_supported_mapping_without_attitude_is_exact_zero() -> None:
    entered, safety, _ = _enter_gate0_approach(with_state=False)
    state = _state(
        safety,
        tracker_id="active-gate-0",
        state_sequence=1,
        frame_id=31,
    )
    result = _step(entered.memory, safety, state=state, attitude=None)

    assert result.decision.objective_permitted
    assert result.proposal.is_exact_zero
    assert _all_source_fields_are_none(result.proposal)
    assert "attitude_missing" in result.proposal.reason


@pytest.mark.parametrize("mismatch", ["host", "authority"])
def test_supported_mapping_requires_exact_tick_clock_and_authority(mismatch) -> None:
    entered, safety, _ = _enter_gate0_approach(with_state=False)
    state = _state(
        safety,
        tracker_id="active-gate-0",
        state_sequence=1,
        frame_id=31,
    )
    tick = _tick(safety, state)
    if mismatch == "host":
        tick = replace(tick, host_clock_id="different-host-clock")
    else:
        tick = replace(
            tick,
            expected_authority=replace(
                safety.authority,
                race_status_sequence=safety.authority.race_status_sequence + 1,
            ),
        )

    rejected = _step(
        entered.memory,
        safety,
        state=state,
        tick=tick,
        attitude=_attitude(),
    )
    assert rejected.decision.objective_permitted
    assert rejected.proposal.is_exact_zero
    assert _all_source_fields_are_none(rejected.proposal)
    expected_reason = (
        "host_clock_mismatch" if mismatch == "host" else "authority_mismatch"
    )
    assert expected_reason in rejected.proposal.reason
    assert rejected.proposal.host_clock_id == tick.host_clock_id
    assert rejected.proposal.authority == tick.expected_authority


def test_permitted_gate0_align_is_not_a_controller_mapping() -> None:
    initial = _safety(
        0,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
    )
    transition = _step(None, initial)
    go = _safety(
        1,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=initial.phase_started_monotonic_ns,
    )
    transition = _step(transition.memory, go)
    align = _safety(
        2,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    state = _state(
        align,
        tracker_id="active-gate-0",
        state_sequence=1,
        frame_id=21,
    )
    result = _step(transition.memory, align, state=state, attitude=_attitude())

    assert result.decision.objective_permitted
    assert result.decision.objective_kind is VQ2GuidanceObjectiveKind.RECENTER_ACTIVE_GATE
    assert result.proposal.is_exact_zero
    assert _all_source_fields_are_none(result.proposal)


def test_gate0_pitch_basis_cannot_change_after_phase_entry() -> None:
    previous, prior_safety, _ = _enter_gate0_approach(with_state=True)
    next_safety = _safety(
        4,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=prior_safety.phase_started_monotonic_ns,
    )
    next_state = _state(
        next_safety,
        tracker_id="active-gate-0",
        state_sequence=2,
        frame_id=41,
    )
    changed = _step(
        previous.memory,
        next_safety,
        state=next_state,
        pitch=0.1,
        attitude=_attitude(),
    )

    assert changed.proposal.is_exact_zero
    assert "pitch_basis_changed" in changed.proposal.reason
    assert changed.memory.gate0_pitch_latch is not None
    assert changed.memory.gate0_pitch_latch.initial_pitch_rad == -0.1

    later_safety = _safety(
        5,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=prior_safety.phase_started_monotonic_ns,
    )
    later_state = _state(
        later_safety,
        tracker_id="active-gate-0",
        state_sequence=3,
        frame_id=51,
    )
    recovered = _step(
        changed.memory,
        later_safety,
        state=later_state,
        attitude=_attitude(),
    )
    assert not recovered.proposal.is_exact_zero


def test_missing_gate0_entry_pitch_cannot_be_supplied_late() -> None:
    entered, safety, _ = _enter_gate0_approach(with_state=False, pitch=None)
    assert entered.memory.gate0_pitch_latch is not None
    assert entered.memory.gate0_pitch_latch.initial_pitch_rad is None

    later_safety = _safety(
        4,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=safety.phase_started_monotonic_ns,
    )
    state = _state(
        later_safety,
        tracker_id="active-gate-0",
        state_sequence=1,
        frame_id=41,
    )
    rejected = _step(
        entered.memory,
        later_safety,
        state=state,
        pitch=-0.1,
        attitude=_attitude(),
    )
    assert rejected.proposal.is_exact_zero
    assert rejected.memory.gate0_pitch_latch.initial_pitch_rad is None


def test_reset_clears_gate0_pitch_and_new_approach_requires_a_new_basis() -> None:
    previous, _, _ = _enter_gate0_approach(with_state=True)
    assert previous.memory.gate0_pitch_latch is not None
    old = previous.memory.guidance_memory.safety
    reset_authority = replace(
        old.authority,
        reset_epoch=old.authority.reset_epoch + 1,
        gate_epoch=0,
        expected_gate_index=0,
        race_status_sequence=old.authority.race_status_sequence + 1,
        race_status_boot_ms=old.authority.race_status_boot_ms + 100,
        camera_generation=old.authority.camera_generation + 1,
        frame_publication_sequence_not_before=(
            old.authority.frame_publication_sequence_not_before + 1
        ),
        frame_publish_monotonic_ns_not_before=(
            old.authority.frame_publish_monotonic_ns_not_before + 100_000_000
        ),
    )
    reset_eval = reset_authority.frame_publish_monotonic_ns_not_before + 50_000_000
    reset_safety = VQ2SafetyGuidanceInput(
        authority=reset_authority,
        phase=VQ2GuidancePhase.ACQUIRE,
        race_state=VQ2GuidanceRaceState.NOT_UNDERWAY,
        evaluation_host_clock_id=_HOST,
        evaluation_monotonic_ns=reset_eval,
        phase_started_monotonic_ns=reset_eval,
    )
    transition = _step(previous.memory, reset_safety)
    assert transition.memory.gate0_pitch_latch is None

    def advance(
        prior: VQ2SafetyGuidanceInput,
        phase: VQ2GuidancePhase,
        race_state: VQ2GuidanceRaceState,
        *,
        preserve_start: bool,
    ) -> VQ2SafetyGuidanceInput:
        authority = replace(
            prior.authority,
            race_status_sequence=prior.authority.race_status_sequence + 1,
            race_status_boot_ms=prior.authority.race_status_boot_ms + 100,
            frame_publication_sequence_not_before=(
                prior.authority.frame_publication_sequence_not_before + 1
            ),
            frame_publish_monotonic_ns_not_before=(
                prior.authority.frame_publish_monotonic_ns_not_before + 100_000_000
            ),
        )
        evaluation_ns = authority.frame_publish_monotonic_ns_not_before + 50_000_000
        return VQ2SafetyGuidanceInput(
            authority=authority,
            phase=phase,
            race_state=race_state,
            evaluation_host_clock_id=_HOST,
            evaluation_monotonic_ns=evaluation_ns,
            phase_started_monotonic_ns=(
                prior.phase_started_monotonic_ns
                if preserve_start
                else evaluation_ns
            ),
        )

    go = advance(
        reset_safety,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        preserve_start=True,
    )
    transition = _step(transition.memory, go)
    align = advance(
        go,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        preserve_start=False,
    )
    transition = _step(transition.memory, align)
    approach = advance(
        align,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
        preserve_start=False,
    )
    transition = _step(transition.memory, approach)

    assert transition.memory.gate0_pitch_latch is not None
    assert transition.memory.gate0_pitch_latch.initial_pitch_rad is None
    assert transition.proposal.is_exact_zero
    assert "pitch_basis_missing" in transition.proposal.reason


def test_gate0_permitted_commit_is_always_source_less_zero() -> None:
    previous, _, _ = _enter_gate0_approach(with_state=True)
    commit = _safety(
        4,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidanceRaceState.UNDERWAY,
    )
    state = _state(
        commit,
        tracker_id="active-gate-0",
        state_sequence=2,
        frame_id=41,
        bearing_norm=(0.01, -0.01),
        bearing_rate_norm_s=(0.01, -0.01),
    )
    result = _step(previous.memory, commit, state=state, attitude=_attitude())

    assert result.decision.objective_permitted
    assert result.decision.objective_kind is VQ2GuidanceObjectiveKind.COMMIT_ACTIVE_GATE
    assert result.proposal.is_exact_zero
    assert _all_source_fields_are_none(result.proposal)
    assert "unsupported_guidance_mapping" in result.proposal.reason


def test_gate1_permitted_commit_is_always_source_less_zero() -> None:
    previous, align, _ = _enter_gate1_align()
    approach = _safety(
        9,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    approach_state = _state(
        approach,
        tracker_id="active-gate-1",
        state_sequence=2,
        frame_id=91,
        bearing_norm=(0.03, -0.02),
    )
    previous = _step(previous.memory, approach, state=approach_state)
    assert previous.decision.objective_permitted
    assert previous.proposal.is_exact_zero
    assert _all_source_fields_are_none(previous.proposal)
    commit = _safety(
        10,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    commit_state = _state(
        commit,
        tracker_id="active-gate-1",
        state_sequence=3,
        frame_id=101,
        bearing_norm=(0.01, -0.01),
        bearing_rate_norm_s=(0.01, -0.01),
    )
    result = _step(previous.memory, commit, state=commit_state)

    assert result.decision.objective_permitted
    assert result.proposal.is_exact_zero
    assert _all_source_fields_are_none(result.proposal)


def test_permitted_gate2_align_is_always_source_less_zero() -> None:
    transition, _, _ = _enter_gate1_align()
    approach = _safety(
        9,
        VQ2GuidancePhase.APPROACH,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    transition = _step(
        transition.memory,
        approach,
        state=_state(
            approach,
            tracker_id="active-gate-1",
            state_sequence=2,
            frame_id=91,
            bearing_norm=(0.03, -0.02),
        ),
    )
    commit = _safety(
        10,
        VQ2GuidancePhase.COMMIT,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    transition = _step(
        transition.memory,
        commit,
        state=_state(
            commit,
            tracker_id="active-gate-1",
            state_sequence=3,
            frame_id=101,
            bearing_norm=(0.01, -0.01),
            bearing_rate_norm_s=(0.01, -0.01),
        ),
    )
    confirmation = _safety(
        11,
        VQ2GuidancePhase.CONFIRMATION,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    transition = _step(transition.memory, confirmation)
    credited = _safety(
        12,
        VQ2GuidancePhase.POST_CREDIT_REACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=2,
        gate_index=2,
    )
    transition = _step(transition.memory, credited)
    acquire = _safety(
        13,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=2,
        gate_index=2,
    )
    transition = _step(transition.memory, acquire)
    align = _safety(
        14,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        gate_epoch=2,
        gate_index=2,
    )
    state = _state(
        align,
        tracker_id="active-gate-2",
        state_sequence=1,
        frame_id=141,
    )
    result = _step(transition.memory, align, state=state, attitude=_attitude())

    assert result.decision.objective_permitted
    assert result.decision.objective_kind is VQ2GuidanceObjectiveKind.RECENTER_ACTIVE_GATE
    assert result.proposal.is_exact_zero
    assert _all_source_fields_are_none(result.proposal)
    assert "unsupported_guidance_mapping" in result.proposal.reason


def test_gate1_exact_time_limit_remains_source_less_zero() -> None:
    previous, align, _ = _enter_gate1_align()
    same_phase = _safety(
        9,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=align.phase_started_monotonic_ns,
        gate_epoch=1,
        gate_index=1,
    )
    same_phase = replace(
        same_phase,
        evaluation_monotonic_ns=align.phase_started_monotonic_ns + 590_000_000,
    )
    state = _state(
        same_phase,
        tracker_id="active-gate-1",
        state_sequence=2,
        frame_id=91,
        bearing_norm=(0.40, -0.30),
        clipping=FrameEdge.TOP,
        health=RelativeStateHealth.DEGRADED,
    )
    result = _step(
        previous.memory,
        same_phase,
        state=state,
        tick=_tick(same_phase, state, proposal_offset_ns=10_000_000),
        attitude=_attitude(),
    )

    assert result.decision.objective_permitted
    assert result.proposal.is_exact_zero
    assert _all_source_fields_are_none(result.proposal)
    assert "time_limit" in result.proposal.reason


def test_shadow_cannot_be_promoted_through_adapter_owned_memory() -> None:
    entered, safety, _ = _enter_gate0_approach(with_state=False)
    shadow = _state(
        safety,
        tracker_id="shadow-owner",
        state_sequence=1,
        frame_id=31,
        track_role=TrackRole.SHADOW,
    )
    recorded = step_vq2_wave2_adapter(
        entered.memory,
        safety,
        active_state=None,
        shadow_states=(shadow,),
        attitude=None,
        tick=_tick(safety),
    )
    promoted = replace(shadow, state_sequence=2, track_role=TrackRole.ACTIVE)
    rejected = _step(
        recorded.memory,
        safety,
        state=promoted,
        attitude=_attitude(),
    )

    assert rejected.memory == recorded.memory
    assert rejected.proposal.is_exact_zero
    assert _all_source_fields_are_none(rejected.proposal)
    assert "association_invalid" in rejected.proposal.reason


def test_gate1_controller_corridor_diagnostics_survive_composition() -> None:
    previous, align, _ = _enter_gate1_align()
    uncertain_safety = _safety(
        9,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=align.phase_started_monotonic_ns,
        gate_epoch=1,
        gate_index=1,
    )
    uncertain = _state(
        uncertain_safety,
        tracker_id="active-gate-1",
        state_sequence=2,
        frame_id=91,
        bearing_norm=(0.02, -0.02),
        bearing_rate_norm_s=(0.01, -0.01),
        variance=(0.0016, 0.0016, 0.0001, 0.0016, 0.0016, 0.0001),
    )
    unconfirmed = _step(
        previous.memory,
        uncertain_safety,
        state=uncertain,
        attitude=_attitude(),
    )
    assert unconfirmed.decision.objective_permitted
    assert unconfirmed.proposal.is_exact_zero
    assert "corridor_unconfirmed_limited" in unconfirmed.proposal.reason
    assert unconfirmed.proposal.uncertainty.limited

    confirmed_safety = _safety(
        10,
        VQ2GuidancePhase.ALIGN,
        VQ2GuidanceRaceState.UNDERWAY,
        phase_start_ns=align.phase_started_monotonic_ns,
        gate_epoch=1,
        gate_index=1,
    )
    confirmed = _state(
        confirmed_safety,
        tracker_id="active-gate-1",
        state_sequence=3,
        frame_id=101,
        bearing_norm=(0.02, -0.02),
        bearing_rate_norm_s=(0.01, -0.01),
        variance=(0.000001,) * 6,
    )
    reached = _step(
        unconfirmed.memory,
        confirmed_safety,
        state=confirmed,
        attitude=_attitude(),
    )
    assert reached.decision.objective_permitted
    assert reached.proposal.is_exact_zero
    assert "corridor_reached" in reached.proposal.reason
    assert not reached.proposal.uncertainty.limited


def test_high_uncertainty_gate1_state_is_withheld_before_controller() -> None:
    entered, align, _ = _enter_gate1_align(with_state=False)
    state = _state(
        align,
        tracker_id="active-gate-1",
        state_sequence=1,
        frame_id=81,
        variance=(0.16, 0.16, 0.0001, 0.0001, 0.0001, 0.0001),
        clipping=FrameEdge.TOP,
        health=RelativeStateHealth.DEGRADED,
    )
    result = _step(entered.memory, align, state=state, attitude=_attitude())

    assert not result.decision.objective_permitted
    assert result.proposal.is_exact_zero
    assert _all_source_fields_are_none(result.proposal)
    assert "outside_uncertainty_corridor" in result.proposal.reason


def test_identical_inputs_and_prior_memory_are_deterministic() -> None:
    entered, safety, _ = _enter_gate0_approach(with_state=False)
    state = _state(
        safety,
        tracker_id="active-gate-0",
        state_sequence=1,
        frame_id=31,
    )
    tick = _tick(safety, state)

    first = _step(
        entered.memory,
        safety,
        state=state,
        tick=tick,
        attitude=_attitude(),
    )
    second = _step(
        entered.memory,
        safety,
        state=state,
        tick=tick,
        attitude=_attitude(),
    )
    assert first == second


def test_exported_transition_rejects_a_same_authority_unrelated_source() -> None:
    valid, _, _ = _enter_gate0_approach(with_state=True)
    unrelated = replace(
        valid.proposal,
        source_candidate_id="unrelated-candidate",
    )

    with pytest.raises(ValueError, match="match guidance source"):
        VQ2Wave2AdapterTransition(
            memory=valid.memory,
            decision=valid.decision,
            proposal=unrelated,
        )


@pytest.mark.parametrize(
    "incoherence",
    ["terminal", "off_center", "no_active", "missing_pitch"],
)
def test_exported_transition_rejects_other_sourced_incoherence(incoherence) -> None:
    valid, _, _ = _enter_gate0_approach(with_state=True)
    memory = valid.memory
    decision = valid.decision
    if incoherence == "terminal":
        terminal_safety = replace(
            memory.guidance_memory.safety,
            race_state=VQ2GuidanceRaceState.FINISHED,
        )
        memory = replace(
            memory,
            guidance_memory=replace(
                memory.guidance_memory,
                safety=terminal_safety,
            ),
        )
        decision = replace(
            decision,
            race_state=VQ2GuidanceRaceState.FINISHED,
        )
    elif incoherence == "off_center":
        decision = replace(decision, target_bearing_norm=(0.1, 0.0))
    elif incoherence == "no_active":
        memory = replace(
            memory,
            guidance_memory=replace(
                memory.guidance_memory,
                active_source=None,
                track_histories=(),
            ),
        )
    else:
        memory = replace(memory, gate0_pitch_latch=None)

    with pytest.raises(ValueError):
        VQ2Wave2AdapterTransition(
            memory=memory,
            decision=decision,
            proposal=valid.proposal,
        )


def test_fresh_midcourse_initialization_is_rejected() -> None:
    safety = _safety(
        0,
        VQ2GuidancePhase.ACQUIRE,
        VQ2GuidanceRaceState.NOT_UNDERWAY,
        gate_epoch=1,
        gate_index=1,
    )
    with pytest.raises(ValueError, match="gate epoch/index zero"):
        _step(None, safety)


def test_generated_timed_frame_aperture_to_controller_source_chain() -> None:
    entered, safety, _ = _enter_gate0_approach(with_state=False)
    estimator = VQ2RelativeGateEstimator("active-gate-0")
    result = entered
    selection = observation = estimator_update = state = None
    for index in range(1, 13):
        evaluation_safety = replace(
            safety,
            evaluation_monotonic_ns=(
                safety.evaluation_monotonic_ns + (index - 1) * 40_000_000
            ),
        )
        selection, detection = _timed_gate_detection(
            evaluation_safety,
            frame_id=300 + index,
            publication_offset=index,
        )
        frame = selection.snapshot.camera_frame
        observation = gate_detection_with_aperture_to_observation_v1(
            detection,
            frame.image,
            frame_timing=selection.timing,
            authority=evaluation_safety.authority,
            candidate_id=f"gate-0-aperture-{index}",
            measurement_uncertainty_ns=1_000_000,
            fallback_center_covariance=FeatureCovarianceV1(
                model_id="wave2-adapter-center",
                feature_order=("center_x_norm", "center_y_norm"),
                matrix=((0.000001, 0.0), (0.0, 0.000001)),
            ),
            image_width=frame.width,
            image_height=frame.height,
        )
        estimator_update = estimator.update(
            observation,
            RelativePredictionTarget.at_decision(
                _HOST,
                evaluation_safety.evaluation_monotonic_ns - 5_000_000,
            ),
        )
        state = estimator_update.state
        result = _step(
            result.memory,
            evaluation_safety,
            state=state,
            attitude=_attitude(),
        )
        if result.decision.objective_permitted:
            break

    assert selection is not None
    assert observation is not None
    assert estimator_update is not None and estimator_update.measurement_accepted
    assert state is not None
    validate_relative_gate_state_source(state, observation)
    assert result.decision.objective_permitted
    assert not result.proposal.is_exact_zero
    validate_command_proposal_source(result.proposal, state)
    assert result.proposal.source_frame == selection.timing.identity
    assert result.proposal.source_candidate_id == observation.candidate_id


def test_bbox_only_timed_detection_cannot_reach_the_adapter() -> None:
    entered, safety, _ = _enter_gate0_approach(with_state=False)
    selection, detection = _timed_gate_detection(safety)
    observation = gate_detection_to_observation_v1(
        detection,
        frame_timing=selection.timing,
        authority=safety.authority,
        candidate_id="gate-0-bbox-only",
        measurement_uncertainty_ns=1_000_000,
        center_covariance=FeatureCovarianceV1(
            model_id="wave2-adapter-center",
            feature_order=("center_x_norm", "center_y_norm"),
            matrix=((0.000001, 0.0), (0.0, 0.000001)),
        ),
        image_width=selection.snapshot.camera_frame.width,
        image_height=selection.snapshot.camera_frame.height,
    )

    with pytest.raises(MissingApertureScaleError, match="fitted inner-aperture"):
        VQ2RelativeGateEstimator("active-gate-0").update(
            observation,
            RelativePredictionTarget.at_decision(
                _HOST,
                safety.evaluation_monotonic_ns - 5_000_000,
            ),
        )
    assert entered.proposal.is_exact_zero
    assert _all_source_fields_are_none(entered.proposal)


def test_adapter_module_has_no_runtime_authority_or_system_id_imports() -> None:
    source = Path(__file__).parents[1].joinpath("vq2_wave2_adapter.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "aigp_mavlink",
        "vq2_runtime",
        "SupervisorApprovedCommand",
        "vq2_system_id",
        "socket",
        "pymavlink",
        "send_attitude",
        "SIM_RESET",
    )
    for token in forbidden:
        assert token not in source
    assert "ControllerAttitudeInput" in source
    assert "ineligible for\nshadow, runtime, or powered wiring" in source
