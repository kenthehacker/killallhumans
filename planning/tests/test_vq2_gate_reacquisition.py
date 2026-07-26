"""Tracker-state regressions for credited VQ2 gate reacquisition.

The cross-ID scenario preserves the measured 464,904,700 ns powered-run gap,
but constructs compact detections around it.  These are not JPEG, detector,
receiver, dynamics, or full recorded-sequence replay tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetTrackerConfig,
    MultiTargetVisualTracker,
    VisualDetection,
    VisualDetectionFrame,
    VisualTrackRole,
    visual_track_history_sha256,
)
from competition.vq2_contracts import FrameEdge
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    ConfirmedGateReacquisition,
    ConfirmedGateTransition,
    CreditedUnboundGateAdvance,
    GateGraphError,
    GateGraphPhase,
    GateReacquisitionPending,
    RaceStatusProvenanceBasis,
    RollingVisualGateGraph,
)


_FRAME_PERIOD_NS = 33_000_000
_PUBLISH_DELAY_NS = 1_000_000
_RECORDED_SEAM_EXTRA_NS = 2_904_700
_RECORDED_SEAM_GAP_NS = 464_904_700
_HOST_CLOCK_ID = "vq2-reacquisition-tracker-state-replay"


def _detection(
    source_index: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    *,
    confidence: float = 0.9,
) -> VisualDetection:
    center_unit_x = 0.5 * (center_x + 1.0)
    center_unit_y = 0.5 * (center_y + 1.0)
    return VisualDetection(
        source_index=source_index,
        center_norm=(center_x, center_y),
        bbox_norm=(
            center_unit_x - width / 2.0,
            center_unit_y - height / 2.0,
            center_unit_x + width / 2.0,
            center_unit_y + height / 2.0,
        ),
        confidence=confidence,
    )


def _frame(
    sequence: int,
    detections: tuple[VisualDetection, ...],
) -> VisualDetectionFrame:
    seam_offset_ns = _RECORDED_SEAM_EXTRA_NS if sequence >= 19 else 0
    observation_ns = (
        10_000_000_000
        + sequence * _FRAME_PERIOD_NS
        + seam_offset_ns
    )
    return VisualDetectionFrame(
        token=CameraFrameToken(
            generation=8,
            frame_id=120_000 + sequence,
            publication_sequence=sequence,
            stream_id="vq2-reacquisition-camera",
        ),
        provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
        time_basis_id=_HOST_CLOCK_ID,
        image_size_px=(640, 360),
        detections=detections,
        camera_source_time_ns=30_000_000_000 + sequence * _FRAME_PERIOD_NS,
        final_unique_packet_monotonic_ns=observation_ns,
        publish_monotonic_ns=observation_ns + _PUBLISH_DELAY_NS,
    )


def _race(
    *,
    gate_index: int,
    sequence: int,
    boot_ms: int,
    received_ns: int,
) -> AuthoritativeRaceStatusRef:
    return AuthoritativeRaceStatusRef(
        provenance_basis=RaceStatusProvenanceBasis.LIVE_INGRESS,
        session_id="vq2-reacquisition-session",
        reset_epoch=6,
        race_status_boot_ms=boot_ms,
        active_gate_index=gate_index,
        race_finished=False,
        race_generation=3,
        race_status_sequence=sequence,
        received_monotonic_ns=received_ns,
        host_clock_id=_HOST_CLOCK_ID,
    )


def _current_detection(sequence: int) -> VisualDetection:
    return _detection(
        0,
        -0.03 + 0.001 * sequence,
        0.02,
        0.42,
        0.44,
        confidence=0.95,
    )


def _reviewed_detection(sequence: int) -> VisualDetection:
    return _detection(
        1,
        0.435 + 0.006 * (sequence - 1),
        -0.443 - 0.006 * (sequence - 1),
        0.105,
        0.135,
    )


def _successor_detection(
    sequence: int,
    *,
    source_index: int = 1,
) -> VisualDetection:
    return _detection(
        source_index,
        0.553 + 0.005 * (sequence - 19),
        -0.572 - 0.005 * (sequence - 19),
        0.140,
        0.183,
    )


def _second_compatible_successor(sequence: int) -> VisualDetection:
    return _detection(
        2,
        0.68 + 0.003 * (sequence - 19),
        -0.40 - 0.003 * (sequence - 19),
        0.130,
        0.180,
        confidence=0.86,
    )


@dataclass(frozen=True)
class _CrossIdReplayState:
    tracker: MultiTargetVisualTracker
    graph: RollingVisualGateGraph
    advance: CreditedUnboundGateAdvance
    current_track_id: str
    reviewed_track_id: str
    successor_track_ids: tuple[str, ...]
    credit_frame: VisualDetectionFrame


def _bind_initial_tracks(
    *,
    from_gate_index: int,
) -> tuple[
    MultiTargetVisualTracker,
    RollingVisualGateGraph,
    str,
    str,
]:
    tracker = MultiTargetVisualTracker(
        MultiTargetTrackerConfig(max_association_gap_ns=300_000_000)
    )
    graph = RollingVisualGateGraph()
    first_ids: tuple[str, ...] = ()
    latest_frame: VisualDetectionFrame | None = None
    for sequence in range(1, 6):
        latest_frame = _frame(
            sequence,
            (
                _current_detection(sequence),
                _reviewed_detection(sequence),
            ),
        )
        update = tracker.update(latest_frame)
        if sequence == 1:
            first_ids = update.created_track_ids
    assert latest_frame is not None
    assert len(first_ids) == 2
    current_track_id, reviewed_track_id = first_ids
    baseline_race = _race(
        gate_index=from_gate_index,
        sequence=100,
        boot_ms=1_000,
        received_ns=latest_frame.publish_monotonic_ns + 1,
    )
    graph.bind_initial_current(
        tracker,
        track_id=current_track_id,
        race_status=baseline_race,
    )
    return tracker, graph, current_track_id, reviewed_track_id


def _prepare_cross_id_credited_unbound(
    *,
    from_gate_index: int = 0,
    successor_count: int = 1,
    precredit_successor_frames: int = 3,
    credit_receive_delay_ns: int = 2_000_000,
) -> _CrossIdReplayState:
    tracker, graph, current_track_id, reviewed_track_id = (
        _bind_initial_tracks(from_gate_index=from_gate_index)
    )

    # The reviewed preview disappears for thirteen processed publications and
    # is retired under the unchanged 12-miss identity lease.
    for sequence in range(6, 19):
        tracker.update(
            _frame(
                sequence,
                (_current_detection(sequence),),
            )
        )
        graph.observe(tracker)
    assert tracker.track(reviewed_track_id).role is VisualTrackRole.RETIRED

    assert 1 <= precredit_successor_frames <= 3
    successor_ids_list: list[str] = []
    credit_frame: VisualDetectionFrame | None = None
    for sequence in range(19, 22):
        successor_detections = []
        if sequence >= 22 - precredit_successor_frames:
            successor_detections.append(_successor_detection(sequence))
            if successor_count == 2:
                successor_detections.append(
                    _second_compatible_successor(sequence)
                )
        credit_frame = _frame(
            sequence,
            (
                _current_detection(sequence),
                *successor_detections,
            ),
        )
        update = tracker.update(credit_frame)
        graph.observe(tracker)
        successor_ids_list.extend(
            track_id
            for track_id in update.created_track_ids
            if track_id != current_track_id
        )
    assert credit_frame is not None
    successor_ids = tuple(successor_ids_list)
    assert len(successor_ids) == successor_count

    credit_race = _race(
        gate_index=from_gate_index + 1,
        sequence=101,
        boot_ms=1_250,
        received_ns=(
            credit_frame.publish_monotonic_ns
            + credit_receive_delay_ns
        ),
    )
    advance = graph.confirm_reviewed_advance(
        tracker,
        race_status=credit_race,
        camera_token_at_credit=credit_frame.token,
        reviewed_track_id=reviewed_track_id,
    )
    assert type(advance) is CreditedUnboundGateAdvance
    return _CrossIdReplayState(
        tracker=tracker,
        graph=graph,
        advance=advance,
        current_track_id=current_track_id,
        reviewed_track_id=reviewed_track_id,
        successor_track_ids=successor_ids,
        credit_frame=credit_frame,
    )


def test_live_credit_records_explicit_unbound_gate_without_successor_authority() -> None:
    state = _prepare_cross_id_credited_unbound()
    snapshot = state.graph.latest_snapshot
    assert snapshot is not None

    assert state.advance.from_gate_index == 0
    assert state.advance.to_gate_index == 1
    assert state.advance.retired_track_id == state.current_track_id
    assert state.advance.reviewed_track_id == state.reviewed_track_id
    assert (
        state.advance.alternative_reacquisition_track_ids_at_credit
        == state.successor_track_ids
    )
    assert (
        state.advance.reviewed_history_sha256
        == visual_track_history_sha256(
            state.tracker.track(state.reviewed_track_id).history
        )
    )
    assert snapshot.phase is GateGraphPhase.CREDITED_UNBOUND
    assert snapshot.current_track_id is None
    assert snapshot.current_gate_index == 1
    assert snapshot.pending_unbound_advance == state.advance
    assert not snapshot.authority_usable
    assert snapshot.withholding_reason == "credited_gate_unbound"
    assert (
        state.tracker.track(state.current_track_id).role
        is VisualTrackRole.RETIRED
    )
    assert all(
        state.tracker.track(track_id).authoritative_gate_index is None
        for track_id in state.successor_track_ids
    )


def test_reviewed_advance_returns_exact_retained_promotion_when_available() -> None:
    tracker, graph, current_track_id, reviewed_track_id = (
        _bind_initial_tracks(from_gate_index=3)
    )
    update = tracker.latest_update
    assert update is not None
    tracker.assign_role(reviewed_track_id, VisualTrackRole.NEXT)
    graph.observe(tracker)
    credit_frame: VisualDetectionFrame | None = None
    for sequence in range(6, 9):
        credit_frame = _frame(
            sequence,
            (
                _current_detection(sequence),
                _reviewed_detection(sequence),
            ),
        )
        tracker.update(credit_frame)
        graph.observe(tracker)
    assert credit_frame is not None
    credit_race = _race(
        gate_index=4,
        sequence=101,
        boot_ms=1_250,
        received_ns=credit_frame.publish_monotonic_ns + 2_000_000,
    )

    outcome = graph.confirm_reviewed_advance(
        tracker,
        race_status=credit_race,
        camera_token_at_credit=credit_frame.token,
        reviewed_track_id=reviewed_track_id,
    )

    assert type(outcome) is ConfirmedGateTransition
    assert outcome.retired_track_id == current_track_id
    assert outcome.promoted_track_id == reviewed_track_id
    assert graph.latest_snapshot is not None
    assert graph.latest_snapshot.phase is GateGraphPhase.CURRENT_BOUND
    assert graph.latest_snapshot.current_track_id == reviewed_track_id


def test_delayed_race_finish_closes_credited_unbound_transition_chain() -> None:
    state = _prepare_cross_id_credited_unbound()
    finish_race = replace(
        state.advance.race_status,
        race_status_sequence=102,
        race_status_boot_ms=1_500,
        received_monotonic_ns=(
            state.advance.race_status.received_monotonic_ns + 1
        ),
        race_finished=True,
    )

    snapshot = state.graph.confirm_race_finished(
        state.tracker,
        race_status=finish_race,
        camera_token_at_finish=state.credit_frame.token,
    )

    assert snapshot.phase is GateGraphPhase.RACE_FINISHED
    assert snapshot.race_finished
    assert snapshot.pending_unbound_advance is None
    assert snapshot.confirmed_transitions == (state.advance,)
    assert snapshot.confirmed_transitions[0].promoted_track_id is None


@pytest.mark.parametrize("from_gate_index", (0, 1, 4))
def test_cross_id_postcredit_binding_is_gate_generic(
    from_gate_index: int,
) -> None:
    state = _prepare_cross_id_credited_unbound(
        from_gate_index=from_gate_index,
    )
    successor_id = state.successor_track_ids[0]
    history_before = state.tracker.track(successor_id).history
    postcredit_frame = _frame(
        22,
        (_successor_detection(22),),
    )
    state.tracker.update(postcredit_frame)
    state.graph.observe(state.tracker)

    reacquisition = state.graph.try_confirm_reacquired_current(
        state.tracker,
        credited_advance=state.advance,
        camera_token_at_binding=postcredit_frame.token,
    )
    snapshot = state.graph.latest_snapshot
    assert snapshot is not None

    assert type(reacquisition) is ConfirmedGateReacquisition
    assert reacquisition.gate_index == from_gate_index + 1
    assert reacquisition.reacquired_track_id == successor_id
    assert reacquisition.identity_basis == "fresh-unique-local-track"
    assert not reacquisition.cross_gap_identity_claimed
    reviewed = state.tracker.track(state.reviewed_track_id)
    successor = state.tracker.track(successor_id)
    assert (
        successor.history[0].observation_monotonic_ns
        - reviewed.history[-1].observation_monotonic_ns
        == _RECORDED_SEAM_GAP_NS
    )
    assert (
        _RECORDED_SEAM_GAP_NS
        > state.tracker.config.max_association_gap_ns
    )
    assert reacquisition.history_length_at_binding == len(
        history_before
    ) + 1
    assert (
        state.tracker.track(successor_id).history[:-1]
        == history_before
    )
    assert (
        state.tracker.track(successor_id).authoritative_gate_index
        == from_gate_index + 1
    )
    assert snapshot.phase is GateGraphPhase.CURRENT_BOUND
    assert snapshot.current_track_id == successor_id
    assert snapshot.current_gate_index == from_gate_index + 1
    assert snapshot.pending_unbound_advance is None
    assert snapshot.authority_usable


def test_reacquisition_rejects_newer_token_not_observed_after_race_receipt() -> None:
    state = _prepare_cross_id_credited_unbound(
        credit_receive_delay_ns=2 * _FRAME_PERIOD_NS,
    )
    postcredit_token_but_precredit_timing = _frame(
        22,
        (_successor_detection(22),),
    )
    state.tracker.update(postcredit_token_but_precredit_timing)
    state.graph.observe(state.tracker)
    tracks_before = state.tracker.tracks()
    snapshot_before = state.graph.latest_snapshot

    outcome = state.graph.try_confirm_reacquired_current(
        state.tracker,
        credited_advance=state.advance,
        camera_token_at_binding=(
            postcredit_token_but_precredit_timing.token
        ),
    )

    assert type(outcome) is GateReacquisitionPending
    assert not outcome.ambiguous
    assert "not exact fresh post-credit" in outcome.reason
    assert state.tracker.tracks() == tracks_before
    assert state.graph.latest_snapshot == snapshot_before
    assert state.graph.latest_snapshot is not None
    assert (
        state.graph.latest_snapshot.phase
        is GateGraphPhase.CREDITED_UNBOUND
    )
    assert (
        state.graph.latest_snapshot.pending_unbound_advance
        == state.advance
    )


def test_reacquisition_accepts_clean_to_one_edge_observable_tail() -> None:
    state = _prepare_cross_id_credited_unbound()
    for sequence in range(22, 25):
        detection = _successor_detection(sequence)
        if sequence == 24:
            detection = replace(
                detection,
                clipping=FrameEdge.TOP,
                center_censored=True,
            )
        binding_frame = _frame(sequence, (detection,))
        state.tracker.update(binding_frame)
        state.graph.observe(state.tracker)

    outcome = state.graph.confirm_reacquired_current(
        state.tracker,
        credited_advance=state.advance,
        camera_token_at_binding=binding_frame.token,
    )

    assert type(outcome) is ConfirmedGateReacquisition
    assert outcome.reacquired_track_id == state.successor_track_ids[0]
    assert outcome.stable_frame_tokens[-1] == binding_frame.token
    assert not outcome.cross_gap_identity_claimed


def test_reacquisition_rejects_multi_edge_censored_tail() -> None:
    state = _prepare_cross_id_credited_unbound()
    for sequence in range(22, 25):
        detection = _successor_detection(sequence)
        if sequence == 24:
            detection = replace(
                detection,
                clipping=FrameEdge.TOP | FrameEdge.RIGHT,
                center_censored=True,
            )
        binding_frame = _frame(sequence, (detection,))
        state.tracker.update(binding_frame)
        state.graph.observe(state.tracker)
    tracks_before = state.tracker.tracks()
    snapshot_before = state.graph.latest_snapshot

    outcome = state.graph.try_confirm_reacquired_current(
        state.tracker,
        credited_advance=state.advance,
        camera_token_at_binding=binding_frame.token,
    )

    assert type(outcome) is GateReacquisitionPending
    assert not outcome.ambiguous
    assert "no unique observable local successor" in outcome.reason
    assert state.tracker.tracks() == tracks_before
    assert state.graph.latest_snapshot == snapshot_before


def test_successor_maturing_on_first_postcredit_frame_reacquires() -> None:
    state = _prepare_cross_id_credited_unbound(
        precredit_successor_frames=2,
    )
    successor_id = state.successor_track_ids[0]
    assert (
        state.advance.alternative_reacquisition_track_ids_at_credit
        == ()
    )
    assert state.tracker.track(successor_id).consecutive_frame_count == 2
    postcredit_frame = _frame(
        22,
        (_successor_detection(22),),
    )
    state.tracker.update(postcredit_frame)
    state.graph.observe(state.tracker)

    outcome = state.graph.confirm_reacquired_current(
        state.tracker,
        credited_advance=state.advance,
        camera_token_at_binding=postcredit_frame.token,
    )

    assert type(outcome) is ConfirmedGateReacquisition
    assert outcome.reacquired_track_id == successor_id
    assert outcome.identity_basis == "fresh-unique-local-track"
    assert not outcome.cross_gap_identity_claimed
    assert outcome.stable_frame_tokens[-1] == postcredit_frame.token
    assert state.graph.latest_snapshot is not None
    assert state.graph.latest_snapshot.current_track_id == successor_id
    assert state.graph.latest_snapshot.authority_usable


def test_multiple_compatible_successors_refuse_ambiguously_without_mutation() -> None:
    state = _prepare_cross_id_credited_unbound(successor_count=2)
    postcredit_frame = _frame(
        22,
        (
            _successor_detection(22),
            _second_compatible_successor(22),
        ),
    )
    state.tracker.update(postcredit_frame)
    state.graph.observe(state.tracker)
    tracks_before = state.tracker.tracks()
    snapshot_before = state.graph.latest_snapshot

    outcome = state.graph.try_confirm_reacquired_current(
        state.tracker,
        credited_advance=state.advance,
        camera_token_at_binding=postcredit_frame.token,
    )

    assert type(outcome) is GateReacquisitionPending
    assert outcome.ambiguous
    assert "selection is ambiguous" in outcome.reason
    assert state.tracker.tracks() == tracks_before
    assert state.graph.latest_snapshot == snapshot_before
    assert (
        state.graph.latest_snapshot.pending_unbound_advance
        == state.advance
    )


def test_same_reviewed_id_can_rebind_after_strictly_postcredit_recovery() -> None:
    tracker, graph, current_track_id, reviewed_track_id = (
        _bind_initial_tracks(from_gate_index=2)
    )
    credit_frame: VisualDetectionFrame | None = None
    for sequence in range(6, 9):
        credit_frame = _frame(
            sequence,
            (_current_detection(sequence),),
        )
        tracker.update(credit_frame)
        graph.observe(tracker)
    assert credit_frame is not None
    assert tracker.track(reviewed_track_id).missed_frame_count == 3
    assert tracker.track(reviewed_track_id).role is not VisualTrackRole.RETIRED

    credit_race = _race(
        gate_index=3,
        sequence=101,
        boot_ms=1_250,
        received_ns=credit_frame.publish_monotonic_ns + 2_000_000,
    )
    advance = graph.confirm_unbound_advance(
        tracker,
        race_status=credit_race,
        camera_token_at_credit=credit_frame.token,
        reviewed_track_id=reviewed_track_id,
    )
    for sequence in range(9, 12):
        recovery_frame = _frame(
            sequence,
            (_reviewed_detection(sequence),),
        )
        update = tracker.update(recovery_frame)
        graph.observe(tracker)
        assert reviewed_track_id in update.associated_track_ids

    reacquisition = graph.try_confirm_reacquired_current(
        tracker,
        credited_advance=advance,
        camera_token_at_binding=recovery_frame.token,
    )

    assert reacquisition.reacquired_track_id == reviewed_track_id
    assert reacquisition.identity_basis == "retained-reviewed-local-track"
    assert not reacquisition.cross_gap_identity_claimed
    assert tracker.track(reviewed_track_id).role is VisualTrackRole.CURRENT
    assert tracker.track(reviewed_track_id).authoritative_gate_index == 3
    assert tracker.track(current_track_id).role is VisualTrackRole.RETIRED
    assert graph.latest_snapshot is not None
    assert graph.latest_snapshot.phase is GateGraphPhase.CURRENT_BOUND
    assert graph.latest_snapshot.current_track_id == reviewed_track_id


def test_try_reacquisition_preserves_hard_binding_token_errors() -> None:
    state = _prepare_cross_id_credited_unbound()
    postcredit_frame = _frame(
        22,
        (_successor_detection(22),),
    )
    state.tracker.update(postcredit_frame)
    state.graph.observe(state.tracker)

    with pytest.raises(
        GateGraphError,
        match="not the latest processed frame",
    ):
        state.graph.try_confirm_reacquired_current(
            state.tracker,
            credited_advance=state.advance,
            camera_token_at_binding=state.credit_frame.token,
        )
