from __future__ import annotations

from dataclasses import replace

import pytest

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetVisualTracker,
    VisualDetection,
    VisualDetectionFrame,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    GateRelationshipBasis,
    GateGraphSnapshot,
    RollingVisualGateGraph,
)
from planning.vq2_visual_approach import (
    RollingVisualApproachServo,
    VisualApproachCurrentGeometryUnavailable,
    VisualApproachRefusal,
)
from planning.vq2_visual_servo import MAX_NEXT_GATE_BLEND


_FRAME_PERIOD_NS = 33_000_000
_BASE_OBSERVATION_NS = 10_000_000_000
_HOST_CLOCK_ID = "host-perf-counter"


def _detection(
    source_index: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    *,
    confidence: float = 0.9,
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
) -> VisualDetection:
    center_unit_x = 0.5 * (center_x + 1.0)
    center_unit_y = 0.5 * (center_y + 1.0)
    left = max(0.0, center_unit_x - width / 2.0)
    right = min(1.0, center_unit_x + width / 2.0)
    top = max(0.0, center_unit_y - height / 2.0)
    bottom = min(1.0, center_unit_y + height / 2.0)
    return VisualDetection(
        source_index=source_index,
        center_norm=(center_x, center_y),
        bbox_norm=(left, top, right, bottom),
        confidence=confidence,
        clipping=clipping,
        center_censored=center_censored,
    )


def _frame(
    sequence: int,
    detections: tuple[VisualDetection, ...],
) -> VisualDetectionFrame:
    observation_ns = _BASE_OBSERVATION_NS + sequence * _FRAME_PERIOD_NS
    return VisualDetectionFrame(
        token=CameraFrameToken(
            generation=1,
            frame_id=1_000 + sequence,
            publication_sequence=sequence,
            stream_id="vq2-camera",
        ),
        provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
        time_basis_id=_HOST_CLOCK_ID,
        image_size_px=(640, 360),
        detections=detections,
        camera_source_time_ns=20_000_000_000 + sequence * _FRAME_PERIOD_NS,
        final_unique_packet_monotonic_ns=observation_ns,
        publish_monotonic_ns=observation_ns + 1_000_000,
    )


def _race(received_ns: int) -> AuthoritativeRaceStatusRef:
    return AuthoritativeRaceStatusRef.live(
        session_id="visual-approach-test",
        reset_epoch=1,
        race_generation=1,
        race_status_sequence=1,
        race_status_boot_ms=5_000,
        active_gate_index=0,
        received_monotonic_ns=received_ns,
        host_clock_id=_HOST_CLOCK_ID,
    )


def _current_detection(
    *,
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
) -> VisualDetection:
    return _detection(
        0,
        0.0,
        0.0,
        0.32,
        0.34,
        clipping=clipping,
        center_censored=center_censored,
    )


def _next_detection(
    *,
    center_x: float = 0.30,
    source_index: int = 1,
) -> VisualDetection:
    return _detection(source_index, center_x, 0.0, 0.14, 0.16)


def _build_bound_graph(
    *,
    include_next: bool = True,
) -> tuple[
    MultiTargetVisualTracker,
    RollingVisualGateGraph,
    GateGraphSnapshot,
    str,
    str | None,
    int,
]:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = ""
    next_id: str | None = None
    snapshot: GateGraphSnapshot | None = None
    final_sequence = 5 if include_next else 3
    for sequence in range(1, final_sequence + 1):
        detections = (_current_detection(),)
        if include_next:
            detections += (_next_detection(),)
        update = tracker.update(_frame(sequence, detections))
        if sequence == 1:
            current_id = update.visible_track_ids[0]
            if include_next:
                next_id = update.visible_track_ids[1]
        if sequence == 3:
            assert update.publish_monotonic_ns is not None
            snapshot = graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=_race(update.publish_monotonic_ns + 1_000_000),
            )
        elif sequence > 3:
            snapshot = graph.observe(tracker)
    assert snapshot is not None
    return tracker, graph, snapshot, current_id, next_id, final_sequence


def _advance(
    tracker: MultiTargetVisualTracker,
    graph: RollingVisualGateGraph,
    sequence: int,
    *,
    include_next: bool = True,
    include_provisional: bool = False,
    next_center_x: float = 0.30,
    current_clipping: FrameEdge = FrameEdge.NONE,
    current_center_censored: bool = False,
) -> GateGraphSnapshot:
    detections = (
        _current_detection(
            clipping=current_clipping,
            center_censored=current_center_censored,
        ),
    )
    if include_next:
        detections += (_next_detection(center_x=next_center_x),)
    if include_provisional:
        detections += (
            _detection(2, -0.55, 0.20, 0.12, 0.12),
        )
    tracker.update(_frame(sequence, detections))
    return graph.observe(tracker)


def _now_s(tracker: MultiTargetVisualTracker) -> float:
    update = tracker.latest_update
    assert update is not None
    return update.observation_monotonic_ns / 1_000_000_000.0 + 0.005


def _observe(
    approach: RollingVisualApproachServo,
    snapshot: GateGraphSnapshot,
    tracker: MultiTargetVisualTracker,
):
    return approach.observe(
        snapshot,
        tracker,
        now_monotonic_s=_now_s(tracker),
        segment_elapsed_s=0.5,
        segment_yaw_excursion_rad=0.0,
    )


def test_stable_exact_next_track_blends_only_after_current_corridor() -> None:
    tracker, graph, snapshot, current_id, next_id, sequence = (
        _build_bound_graph()
    )
    assert next_id is not None
    approach = RollingVisualApproachServo(current_id, 0)

    proposal = _observe(approach, snapshot, tracker)
    assert proposal.servo_output.next_gate_blend == 0.0
    assert proposal.withholding_reason == "current_corridor_not_ready"

    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)

    assert proposal.current_target.track_id == current_id
    assert proposal.next_target is not None
    assert proposal.next_target.track_id == next_id
    assert (
        proposal.next_target.frame_token
        == proposal.current_target.frame_token
    )
    assert proposal.candidate_track_ids == (next_id,)
    assert proposal.provisional_track_ids == ()
    assert proposal.relationship_basis is GateRelationshipBasis.SIMULTANEOUS_IMAGE
    assert proposal.servo_output.next_gate_blend == MAX_NEXT_GATE_BLEND
    assert not proposal.servo_output.advance_enabled
    assert proposal.withholding_reason is None
    assert proposal.latched_next_track_id == next_id
    assert approach.latched_next_track_id == next_id


def test_graph_next_identity_ambiguity_refuses_authority() -> None:
    tracker, _, snapshot, current_id, _, _ = _build_bound_graph()
    approach = RollingVisualApproachServo(current_id, 0)

    with pytest.raises(VisualApproachRefusal, match="ambiguous"):
        _observe(
            approach,
            replace(snapshot, next_selection_ambiguous=True),
            tracker,
        )


def test_provisional_contour_withholds_blend_without_changing_latch() -> None:
    tracker, graph, snapshot, current_id, next_id, sequence = (
        _build_bound_graph()
    )
    assert next_id is not None
    approach = RollingVisualApproachServo(current_id, 0)
    _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    assert proposal.servo_output.next_gate_blend == MAX_NEXT_GATE_BLEND
    assert proposal.latched_next_track_id == next_id

    snapshot = _advance(
        tracker,
        graph,
        sequence + 1,
        include_provisional=True,
    )
    proposal = _observe(approach, snapshot, tracker)

    assert proposal.next_target is None
    assert len(proposal.provisional_track_ids) == 1
    assert proposal.servo_output.next_gate_blend == 0.0
    assert not proposal.servo_output.advance_enabled
    assert (
        proposal.withholding_reason
        == "provisional_next_identity_unresolved"
    )
    assert proposal.latched_next_track_id == next_id


def test_stale_qpc_observation_refuses_authority() -> None:
    tracker, _, snapshot, current_id, _, _ = _build_bound_graph()
    update = tracker.latest_update
    assert update is not None
    approach = RollingVisualApproachServo(current_id, 0)

    with pytest.raises(VisualApproachRefusal, match="stale"):
        approach.observe(
            snapshot,
            tracker,
            now_monotonic_s=(
                update.observation_monotonic_ns / 1_000_000_000.0
                + 0.101
            ),
            segment_elapsed_s=0.5,
            segment_yaw_excursion_rad=0.0,
        )


def test_clipped_or_censored_current_aperture_refuses_authority() -> None:
    tracker, graph, _, current_id, _, sequence = _build_bound_graph()
    snapshot = _advance(
        tracker,
        graph,
        sequence + 1,
        current_clipping=FrameEdge.TOP,
        current_center_censored=True,
    )
    approach = RollingVisualApproachServo(current_id, 0)

    with pytest.raises(
        VisualApproachCurrentGeometryUnavailable,
        match="clipped or censored",
    ):
        _observe(approach, snapshot, tracker)


def test_latched_next_identity_cannot_silently_switch() -> None:
    tracker, graph, snapshot, current_id, next_id, sequence = (
        _build_bound_graph()
    )
    assert next_id is not None
    approach = RollingVisualApproachServo(current_id, 0)
    _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    assert proposal.latched_next_track_id == next_id

    replacement_id: str | None = None
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(
            tracker,
            graph,
            sequence,
            next_center_x=-0.50,
        )
        visible_replacements = tuple(
            track_id
            for track_id in tracker.latest_update.visible_track_ids
            if track_id != current_id
        )
        assert len(visible_replacements) == 1
        replacement_id = visible_replacements[0]
    assert replacement_id is not None
    assert replacement_id != next_id
    assert tuple(
        candidate.track_id for candidate in snapshot.next_candidates
        if candidate.promotable and tracker.track(candidate.track_id).visible
    ) == (replacement_id,)

    with pytest.raises(VisualApproachRefusal, match="identity changed"):
        _observe(approach, snapshot, tracker)


def test_latched_next_loss_withdraws_blend_without_changing_identity() -> None:
    tracker, graph, snapshot, current_id, next_id, sequence = (
        _build_bound_graph()
    )
    assert next_id is not None
    approach = RollingVisualApproachServo(current_id, 0)
    _observe(approach, snapshot, tracker)
    for sequence in range(sequence + 1, sequence + 4):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    assert proposal.latched_next_track_id == next_id

    snapshot = _advance(
        tracker,
        graph,
        sequence + 1,
        include_next=False,
    )
    proposal = _observe(approach, snapshot, tracker)

    assert proposal.next_target is None
    assert proposal.servo_output.next_gate_blend == 0.0
    assert not proposal.servo_output.advance_enabled
    assert proposal.withholding_reason == "latched_next_track_unavailable"
    assert proposal.latched_next_track_id == next_id


def test_no_next_candidate_remains_current_only_and_never_advances() -> None:
    tracker, _, snapshot, current_id, _, _ = _build_bound_graph(
        include_next=False
    )
    approach = RollingVisualApproachServo(current_id, 0)

    proposal = _observe(approach, snapshot, tracker)

    assert proposal.current_target.track_id == current_id
    assert proposal.next_target is None
    assert proposal.candidate_track_ids == ()
    assert proposal.provisional_track_ids == ()
    assert proposal.servo_output.next_gate_blend == 0.0
    assert not proposal.servo_output.advance_enabled
    assert proposal.withholding_reason == "no_next_candidate"
    assert proposal.latched_next_track_id is None
