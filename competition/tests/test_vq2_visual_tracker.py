from __future__ import annotations

import math

import pytest

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetTrackerConfig,
    MultiTargetVisualTracker,
    StaleVisualFrameError,
    VisualDetection,
    VisualDetectionFrame,
    VisualTrackRole,
    visual_track_history_sha256,
)
from planning.vq2_gate_graph import (
    AmbiguousGatePromotionError,
    AuthoritativeRaceStatusRef,
    GateRelationshipBasis,
    GateGraphError,
    RollingVisualGateGraph,
)


_FRAME_PERIOD_NS = 33_000_000


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
    appearance: tuple[float, ...] | None = None,
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
        appearance=appearance,
    )


def _frame(
    sequence: int,
    detections: tuple[VisualDetection, ...],
    *,
    generation: int = 7,
    frame_id: int | None = None,
    publication_sequence: int | None = None,
    timing_offset_ns: int = 0,
) -> VisualDetectionFrame:
    final_packet = (
        1_000_000_000
        + sequence * _FRAME_PERIOD_NS
        + timing_offset_ns
    )
    return VisualDetectionFrame(
        token=CameraFrameToken(
            generation=generation,
            frame_id=29_000 + sequence if frame_id is None else frame_id,
            publication_sequence=(
                sequence
                if publication_sequence is None
                else publication_sequence
            ),
            stream_id="vq2-camera",
        ),
        provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
        time_basis_id="vq2-host-monotonic",
        image_size_px=(640, 360),
        detections=detections,
        camera_source_time_ns=5_000_000_000 + sequence * _FRAME_PERIOD_NS,
        final_unique_packet_monotonic_ns=final_packet,
        publish_monotonic_ns=final_packet + 1_000_000,
    )


def _race(
    *,
    sequence: int,
    boot_ms: int,
    gate_index: int,
    received_ns: int,
    finished: bool = False,
    host_clock_id: str = "vq2-host-monotonic",
) -> AuthoritativeRaceStatusRef:
    return AuthoritativeRaceStatusRef.live(
        session_id="shadow-session",
        reset_epoch=3,
        race_generation=12,
        race_status_sequence=sequence,
        race_status_boot_ms=boot_ms,
        active_gate_index=gate_index,
        received_monotonic_ns=received_ns,
        host_clock_id=host_clock_id,
        race_finished=finished,
    )


def test_tracks_every_detection_and_does_not_follow_largest_or_input_order() -> None:
    tracker = MultiTargetVisualTracker()
    first = tracker.update(
        _frame(
            1,
            (
                _detection(0, -0.30, 0.05, 0.42, 0.42, confidence=0.72),
                _detection(1, 0.48, -0.35, 0.14, 0.16, confidence=0.95),
                _detection(2, -0.76, 0.42, 0.08, 0.10, confidence=0.68),
            ),
        )
    )
    assert len(first.visible_tracks) == 3
    left_id, next_id, fragment_id = first.visible_track_ids

    # Detector order, confidence order, and apparent-area order all change.
    second = tracker.update(
        _frame(
            2,
            (
                _detection(0, 0.50, -0.37, 0.15, 0.17, confidence=0.71),
                _detection(1, -0.74, 0.41, 0.10, 0.11, confidence=0.97),
                _detection(2, -0.27, 0.04, 0.39, 0.40, confidence=0.91),
            ),
        )
    )
    assert second.track(left_id).center_norm == pytest.approx((-0.27, 0.04))
    assert second.track(next_id).center_norm == pytest.approx((0.50, -0.37))
    assert second.track(fragment_id).center_norm == pytest.approx((-0.74, 0.41))
    assert {item.detection_source_index for item in second.associations} == {0, 1, 2}


def test_bounded_aperture_occlusion_preserves_motion_consistent_identity() -> None:
    """Mirror the 461.7 ms build-3385 Gate-0 occlusion without semantic labels."""

    tracker = MultiTargetVisualTracker(
        MultiTargetTrackerConfig(max_association_gap_ns=500_000_000)
    )
    track_id = ""
    for sequence in range(1, 6):
        update = tracker.update(
            _frame(
                sequence,
                (
                    _detection(
                        0,
                        0.435 + 0.006 * (sequence - 1),
                        -0.443 - 0.006 * (sequence - 1),
                        0.105,
                        0.135,
                    ),
                ),
            )
        )
        if sequence == 1:
            track_id = update.visible_track_ids[0]
    first_token = tracker.track(track_id).first_token

    for sequence in range(6, 18):
        tracker.update(_frame(sequence, ()))
    assert tracker.track(track_id).missed_frame_count == 12
    assert tracker.track(track_id).role is not VisualTrackRole.RETIRED

    update = tracker.update(
        _frame(
            19,
            (_detection(0, 0.553, -0.572, 0.140, 0.183),),
        )
    )

    assert update.created_track_ids == ()
    assert update.associated_track_ids == (track_id,)
    assert update.visible_track_ids == (track_id,)
    assert update.ambiguous_track_ids == ()
    assert len(update.associations) == 1
    assert update.associations[0].predicted_center_residual_norm < 0.08
    assert update.associations[0].bbox_iou > 0.35
    recovered = tracker.track(track_id)
    bridge = recovered.history[-1].accepted_association
    assert bridge is update.associations[0]
    assert bridge.previous_token == recovered.history[-2].token
    assert bridge.current_token == recovered.history[-1].token
    assert bridge.missed_frame_count_before_association == 12
    assert bridge.observation_gap_ns == 14 * _FRAME_PERIOD_NS
    assert bridge.publication_gap_ns == 14 * _FRAME_PERIOD_NS
    assert bridge.temporal_consistency == pytest.approx(1.0 / 13.0)
    assert not bridge.ambiguous
    assert not bridge.track_ambiguous_before_association
    assert bridge.predicted_center_residual_norm < 0.08
    assert bridge.bbox_iou > 0.35
    assert recovered.first_token == first_token
    assert recovered.total_observation_count == 6
    assert recovered.consecutive_frame_count == 1
    assert recovered.missed_frame_count == 0


def test_new_track_sample_has_no_invented_association_bridge() -> None:
    tracker = MultiTargetVisualTracker()
    update = tracker.update(
        _frame(
            1,
            (_detection(4, 0.22, -0.31, 0.16, 0.20, confidence=0.83),),
        )
    )

    created = update.track(update.created_track_ids[0])
    assert update.associations == ()
    assert created.history[-1].accepted_association is None


def test_nested_detections_retain_distinct_ids_through_contour_changes() -> None:
    tracker = MultiTargetVisualTracker()
    first = tracker.update(
        _frame(
            1,
            (
                _detection(0, 0.02, -0.02, 0.52, 0.62),
                _detection(1, 0.03, -0.01, 0.16, 0.18),
            ),
        )
    )
    outer_id, inner_id = first.visible_track_ids
    for sequence, outer_width, inner_width in (
        (2, 0.48, 0.17),
        (3, 0.55, 0.15),
        (4, 0.50, 0.19),
    ):
        update = tracker.update(
            _frame(
                sequence,
                (
                    _detection(0, 0.02, -0.02, inner_width, inner_width + 0.02),
                    _detection(1, 0.01, -0.01, outer_width, outer_width + 0.10),
                ),
            )
        )
        assert update.track(outer_id).apparent_scale > update.track(inner_id).apparent_scale
        assert not update.track(outer_id).ambiguous
        assert not update.track(inner_id).ambiguous


def test_top_clipped_fragment_continuity_is_degraded_but_keeps_identity() -> None:
    tracker = MultiTargetVisualTracker()
    first = tracker.update(
        _frame(
            1,
            (
                _detection(
                    0,
                    0.64,
                    -0.86,
                    0.18,
                    0.14,
                    clipping=FrameEdge.TOP,
                    center_censored=True,
                ),
            ),
        )
    )
    track_id = first.visible_track_ids[0]
    second = tracker.update(
        _frame(
            2,
            (
                _detection(
                    0,
                    0.67,
                    -0.83,
                    0.22,
                    0.20,
                    clipping=FrameEdge.TOP,
                    center_censored=True,
                ),
            ),
        )
    )
    assert second.visible_track_ids == (track_id,)
    assert second.track(track_id).center_censored
    assert second.track(track_id).clipping == FrameEdge.TOP
    assert second.associations[0].clipping_continuity == 1.0


def test_ambiguous_near_tie_is_explicit_and_cannot_receive_gate_authority() -> None:
    config = MultiTargetTrackerConfig(ambiguity_margin=0.20)
    tracker = MultiTargetVisualTracker(config)
    first = tracker.update(
        _frame(
            1,
            (
                _detection(
                    0,
                    -0.04,
                    0.0,
                    0.18,
                    0.20,
                    appearance=(0.0,),
                ),
                _detection(
                    1,
                    0.04,
                    0.0,
                    0.18,
                    0.20,
                    appearance=(1.0,),
                ),
            ),
        )
    )
    assert len(first.visible_track_ids) == 2
    second = tracker.update(
        _frame(
            2,
            (
                _detection(0, 0.00, 0.0, 0.18, 0.20),
                _detection(1, 0.00, 0.0, 0.18, 0.20),
            ),
        )
    )
    assert set(second.ambiguous_track_ids) == set(first.visible_track_ids)
    assert all(item.ambiguous for item in second.associations)
    evidence_by_track = {
        item.track_id: item for item in second.associations
    }
    for track_id in second.associated_track_ids:
        accepted = second.track(track_id).history[-1].accepted_association
        assert accepted is evidence_by_track[track_id]
        assert accepted.ambiguous
        assert accepted.missed_frame_count_before_association == 0
        assert accepted.temporal_consistency == 1.0
        assert accepted.track_ambiguous_before_association
    tracker.assign_role(first.visible_track_ids[0], VisualTrackRole.CURRENT)
    with pytest.raises(ValueError, match="ambiguous"):
        tracker.confirm_authoritative_gate(
            first.visible_track_ids[0],
            gate_index=0,
            race_status_sequence=1,
            race_status_boot_ms=100,
        )


def test_association_retains_ambiguity_state_before_later_clean_frames() -> None:
    tracker = MultiTargetVisualTracker(
        MultiTargetTrackerConfig(ambiguity_margin=0.20)
    )
    first = tracker.update(
        _frame(
            1,
            (
                _detection(0, -0.04, 0.0, 0.18, 0.20),
                _detection(1, 0.04, 0.0, 0.18, 0.20),
            ),
        )
    )
    second = tracker.update(
        _frame(
            2,
            (
                _detection(0, 0.0, 0.0, 0.18, 0.20),
                _detection(1, 0.0, 0.0, 0.18, 0.20),
            ),
        )
    )
    assert all(item.ambiguous for item in second.associations)
    retained_id, retired_id = first.visible_track_ids
    tracker.retire_track(retired_id)
    third = tracker.update(
        _frame(
            3,
            (
                _detection(0, 0.01, 0.0, 0.18, 0.20),
            ),
        )
    )

    assert third.associated_track_ids == (retained_id,)
    assert all(not item.ambiguous for item in third.associations)
    assert all(
        item.track_ambiguous_before_association
        for item in third.associations
    )


def test_duplicate_stale_and_cross_generation_frames_fail_closed() -> None:
    tracker = MultiTargetVisualTracker()
    first_frame = _frame(1, (_detection(0, 0.0, 0.0, 0.2, 0.2),))
    tracker.update(first_frame)
    duplicate_identity = VisualDetectionFrame(
        token=CameraFrameToken(
            generation=7,
            frame_id=first_frame.token.frame_id,
            publication_sequence=2,
            stream_id="vq2-camera",
        ),
        provenance_basis=first_frame.provenance_basis,
        time_basis_id=first_frame.time_basis_id,
        image_size_px=first_frame.image_size_px,
        detections=first_frame.detections,
        camera_source_time_ns=first_frame.camera_source_time_ns + 1,
        final_unique_packet_monotonic_ns=(
            first_frame.final_unique_packet_monotonic_ns + 1
        ),
        publish_monotonic_ns=first_frame.publish_monotonic_ns + 1,
    )
    with pytest.raises(StaleVisualFrameError, match="already consumed"):
        tracker.update(duplicate_identity)
    with pytest.raises(StaleVisualFrameError, match="generation changed"):
        tracker.update(
            _frame(2, (_detection(0, 0.0, 0.0, 0.2, 0.2),), generation=8)
        )
    tracker.reset_generation(8)
    reset_update = tracker.update(
        _frame(2, (_detection(0, 0.0, 0.0, 0.2, 0.2),), generation=8)
    )
    assert reset_update.created_track_ids[0] != "vq2-track-000001"


def test_normalized_bearing_elevation_and_rates_use_declared_image_basis() -> None:
    tracker = MultiTargetVisualTracker()
    tracker.update(_frame(1, (_detection(0, 0.10, 0.20, 0.2, 0.2),)))
    update = tracker.update(
        _frame(2, (_detection(0, 0.16, 0.14, 0.22, 0.22),))
    )
    track = update.visible_tracks[0]
    assert track.bearing_norm == pytest.approx(0.16)
    assert track.elevation_norm == pytest.approx(-0.14)
    assert track.bearing_rate_norm_s > 0.0
    assert track.elevation_rate_norm_s > 0.0
    assert math.isfinite(track.log_scale_rate_s)
    assert track.log_scale_rate_s > 0.0


def test_pretransition_next_track_promotes_without_reset_or_history_loss() -> None:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = ""
    next_id = ""
    for sequence in range(1, 6):
        update = tracker.update(
            _frame(
                sequence,
                (
                    _detection(
                        0,
                        -0.03 + 0.005 * sequence,
                        0.02,
                        0.30 + 0.01 * sequence,
                        0.34 + 0.01 * sequence,
                    ),
                    _detection(
                        1,
                        0.58 + 0.005 * sequence,
                        -0.62 - 0.004 * sequence,
                        0.13 + 0.004 * sequence,
                        0.15 + 0.004 * sequence,
                    ),
                ),
            )
        )
        if sequence == 1:
            current_id, next_id = update.visible_track_ids
        if sequence == 3:
            baseline = _race(
                sequence=20,
                boot_ms=6_005,
                gate_index=0,
                received_ns=update.publish_monotonic_ns + 2_000_000,
            )
            snapshot = graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=baseline,
            )
        elif sequence > 3:
            snapshot = graph.observe(tracker)

    assert snapshot.current_track_id == current_id
    assert tuple(item.track_id for item in snapshot.next_candidates) == (next_id,)
    relationship = snapshot.next_candidates[0].relationship
    assert relationship is not None
    assert relationship.basis is GateRelationshipBasis.SIMULTANEOUS_IMAGE
    assert relationship.observation_count == 3
    assert relationship.simultaneous_observation_count == 3
    assert relationship.sequential_observation_count == 0
    assert relationship.relative_geometry_usable
    assert relationship.relative_bearing_norm > 0.0
    assert relationship.relative_elevation_norm > 0.0
    before = tracker.track(next_id)
    assert before.authoritative_gate_index is None
    assert before.first_token == CameraFrameToken(7, 29_001, 1, "vq2-camera")

    transition_status = _race(
        sequence=21,
        boot_ms=6_256,
        gate_index=1,
        received_ns=tracker.latest_update.publish_monotonic_ns + 3_000_000,
    )
    transition = graph.confirm_transition(
        tracker,
        race_status=transition_status,
        camera_token_at_credit=tracker.latest_update.token,
    )
    after = tracker.track(next_id)
    retired = tracker.track(current_id)
    assert transition.promoted_track_id == next_id
    assert len(transition.pretransition_frame_tokens) >= 3
    assert transition.history_length_before_promotion == len(before.history)
    assert transition.history_length_after_promotion == len(before.history)
    assert transition.promoted_history_sha256 == (
        visual_track_history_sha256(before.history)
    )
    assert len(transition.promoted_history_sha256) == 64
    assert after.history == before.history
    assert after.first_token == before.first_token
    assert after.role is VisualTrackRole.CURRENT
    assert after.authoritative_gate_index == 1
    assert retired.role is VisualTrackRole.RETIRED
    assert retired.authoritative_gate_index == 0


def test_adjacent_publication_handoff_promotes_without_synthetic_joint_sample() -> None:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    all_edges = (
        FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT | FrameEdge.BOTTOM
    )
    current_id = ""
    for tracker_sequence, publication, frame_id in (
        (1, 167, 40_100),
        (2, 168, 40_350),
        (3, 169, 40_900),
    ):
        update = tracker.update(
            _frame(
                tracker_sequence,
                (
                    _detection(
                        0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        clipping=all_edges,
                        center_censored=True,
                    ),
                ),
                frame_id=frame_id,
                publication_sequence=publication,
            )
        )
        current_id = update.visible_track_ids[0]
    graph.bind_initial_current(
        tracker,
        track_id=current_id,
        race_status=_race(
            sequence=70,
            boot_ms=5_000,
            gate_index=0,
            received_ns=update.publish_monotonic_ns + 1,
        ),
    )

    next_id = ""
    for tracker_sequence, publication, frame_id in (
        (4, 170, 41_700),
        (5, 171, 41_950),
        (6, 172, 42_600),
    ):
        update = tracker.update(
            _frame(
                tracker_sequence,
                (
                    _detection(
                        0,
                        0.5625 + 0.01 * (tracker_sequence - 4),
                        -0.5833 - 0.01 * (tracker_sequence - 4),
                        0.13 + 0.005 * (tracker_sequence - 4),
                        0.16 + 0.005 * (tracker_sequence - 4),
                    ),
                ),
                frame_id=frame_id,
                publication_sequence=publication,
            )
        )
        if tracker_sequence == 4:
            next_id = next(
                track_id
                for track_id in update.visible_track_ids
                if track_id != current_id
            )
        snapshot = graph.observe(tracker)

    assert tuple(item.track_id for item in snapshot.next_candidates) == (next_id,)
    candidate = snapshot.next_candidates[0]
    assert candidate.promotable
    relationship = candidate.relationship
    assert relationship is not None
    assert (
        relationship.basis
        is GateRelationshipBasis.ADJACENT_PUBLICATION_HANDOFF
    )
    assert relationship.current_anchor_token.publication_sequence == 169
    assert relationship.next_anchor_token.publication_sequence == 170
    assert relationship.current_anchor_token.frame_id == 40_900
    assert relationship.next_anchor_token.frame_id == 41_700
    assert relationship.anchor_publication_delta == 1
    assert 0 < relationship.anchor_time_gap_ns <= 100_000_000
    assert relationship.first_token == relationship.next_anchor_token
    assert relationship.observation_count == 3
    assert relationship.simultaneous_observation_count == 0
    assert relationship.sequential_observation_count == 3
    assert not relationship.fresh
    assert relationship.geometry_degraded
    assert not relationship.relative_geometry_usable
    assert not relationship.contended

    before = tracker.track(next_id)
    before_history = before.history
    before_rates = (
        before.bearing_rate_norm_s,
        before.elevation_rate_norm_s,
        before.log_scale_rate_s,
    )
    transition = graph.confirm_transition(
        tracker,
        race_status=_race(
            sequence=71,
            boot_ms=5_250,
            gate_index=1,
            received_ns=update.publish_monotonic_ns + 1,
        ),
        camera_token_at_credit=update.token,
    )
    after = tracker.track(next_id)
    assert transition.promoted_track_id == next_id
    assert transition.retired_track_id == current_id
    assert transition.pretransition_frame_tokens == tuple(
        sample.token for sample in before_history
    )
    assert after.first_token == relationship.next_anchor_token
    assert after.history == before_history
    assert (
        after.bearing_rate_norm_s,
        after.elevation_rate_norm_s,
        after.log_scale_rate_s,
    ) == before_rates
    assert tracker.track(current_id).role is VisualTrackRole.RETIRED


@pytest.mark.parametrize(
    ("first_next_publication", "current_width", "current_clipping", "timing_offset_ns"),
    (
        pytest.param(171, 1.0, FrameEdge(15), 0, id="publication-gap"),
        pytest.param(170, 0.90, FrameEdge.NONE, 0, id="no-crossing-geometry"),
        pytest.param(
            170,
            1.0,
            FrameEdge(15),
            100_000_000,
            id="timing-gap-over-100ms",
        ),
    ),
)
def test_adjacent_handoff_rejects_missing_or_unsupported_crossing_proof(
    first_next_publication: int,
    current_width: float,
    current_clipping: FrameEdge,
    timing_offset_ns: int,
) -> None:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = ""
    for sequence in range(1, 4):
        update = tracker.update(
            _frame(
                sequence,
                (
                    _detection(
                        0,
                        0.0,
                        0.0,
                        current_width,
                        current_width,
                        clipping=current_clipping,
                        center_censored=current_clipping != FrameEdge.NONE,
                    ),
                ),
                publication_sequence=166 + sequence,
            )
        )
        current_id = update.visible_track_ids[0]
    graph.bind_initial_current(
        tracker,
        track_id=current_id,
        race_status=_race(
            sequence=80,
            boot_ms=7_000,
            gate_index=0,
            received_ns=update.publish_monotonic_ns + 1,
        ),
    )

    for offset in range(3):
        update = tracker.update(
            _frame(
                4 + offset,
                (_detection(0, 0.56, -0.58, 0.13, 0.16),),
                publication_sequence=first_next_publication + offset,
                timing_offset_ns=timing_offset_ns,
            )
        )
        snapshot = graph.observe(tracker)

    assert len(snapshot.next_candidates) == 1
    assert snapshot.next_candidates[0].relationship is None
    assert not snapshot.next_candidates[0].promotable
    with pytest.raises(
        GateGraphError,
        match="no stable pretracked next gate is promotable",
    ):
        graph.confirm_transition(
            tracker,
            race_status=_race(
                sequence=81,
                boot_ms=7_250,
                gate_index=1,
                received_ns=update.publish_monotonic_ns + 1,
            ),
            camera_token_at_credit=update.token,
        )


def test_adjacent_handoff_rejects_multiple_newcomers_even_with_explicit_id() -> None:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    all_edges = (
        FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT | FrameEdge.BOTTOM
    )
    for sequence in range(1, 4):
        update = tracker.update(
            _frame(
                sequence,
                (
                    _detection(
                        0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        clipping=all_edges,
                        center_censored=True,
                    ),
                ),
                publication_sequence=166 + sequence,
            )
        )
    current_id = update.visible_track_ids[0]
    graph.bind_initial_current(
        tracker,
        track_id=current_id,
        race_status=_race(
            sequence=90,
            boot_ms=8_000,
            gate_index=0,
            received_ns=update.publish_monotonic_ns + 1,
        ),
    )

    for offset in range(3):
        update = tracker.update(
            _frame(
                4 + offset,
                (
                    _detection(0, 0.56, -0.58, 0.13, 0.16),
                    _detection(1, -0.56, -0.58, 0.13, 0.16),
                ),
                publication_sequence=170 + offset,
            )
        )
        snapshot = graph.observe(tracker)

    assert len(snapshot.next_candidates) == 2
    assert snapshot.next_selection_ambiguous
    assert all(
        candidate.relationship is None and not candidate.promotable
        for candidate in snapshot.next_candidates
    )
    with pytest.raises(
        AmbiguousGatePromotionError,
        match="indistinguishable authority",
    ):
        graph.confirm_transition(
            tracker,
            race_status=_race(
                sequence=91,
                boot_ms=8_250,
                gate_index=1,
                received_ns=update.publish_monotonic_ns + 1,
            ),
            camera_token_at_credit=update.token,
            promoted_track_id=snapshot.next_candidates[0].track_id,
        )


def _bound_aperture_filling_current(
    *,
    include_pretracked_alternative: bool = False,
) -> tuple[MultiTargetVisualTracker, RollingVisualGateGraph, str, str]:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    all_edges = (
        FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT | FrameEdge.BOTTOM
    )
    alternative_id = ""
    for sequence in range(1, 4):
        detections = [
            _detection(
                0,
                0.0,
                0.0,
                1.0,
                1.0,
                clipping=all_edges,
                center_censored=True,
            )
        ]
        if include_pretracked_alternative:
            detections.append(
                _detection(1, -0.48, -0.30, 0.13, 0.16)
            )
        update = tracker.update(
            _frame(
                sequence,
                tuple(detections),
                publication_sequence=166 + sequence,
            )
        )
    current_id = max(
        update.visible_tracks,
        key=lambda track: track.apparent_scale,
    ).track_id
    if include_pretracked_alternative:
        alternative_id = next(
            track_id
            for track_id in update.visible_track_ids
            if track_id != current_id
        )
    graph.bind_initial_current(
        tracker,
        track_id=current_id,
        race_status=_race(
            sequence=100,
            boot_ms=9_000,
            gate_index=0,
            received_ns=update.publish_monotonic_ns + 1,
        ),
    )
    return tracker, graph, current_id, alternative_id


@pytest.mark.parametrize(
    ("publications", "timing_offsets_ns"),
    (
        pytest.param((170, 900, 901), (0, 0, 0), id="publication-gap"),
        pytest.param(
            (170, 171, 172),
            (0, 200_000_000, 200_000_000),
            id="timing-gap",
        ),
    ),
)
def test_adjacent_handoff_invalidates_noncontiguous_successor_observations(
    publications: tuple[int, int, int],
    timing_offsets_ns: tuple[int, int, int],
) -> None:
    tracker, graph, _, _ = _bound_aperture_filling_current()
    for offset, (publication, timing_offset_ns) in enumerate(
        zip(publications, timing_offsets_ns)
    ):
        update = tracker.update(
            _frame(
                4 + offset,
                (_detection(0, 0.56, -0.58, 0.13, 0.16),),
                publication_sequence=publication,
                timing_offset_ns=timing_offset_ns,
            )
        )
        snapshot = graph.observe(tracker)

    assert len(snapshot.next_candidates) == 1
    relationship = snapshot.next_candidates[0].relationship
    assert relationship is not None
    assert relationship.sequential_observation_count == 1
    assert relationship.contended
    assert not snapshot.next_candidates[0].promotable


def test_adjacent_handoff_expires_when_predecessor_track_lease_retires() -> None:
    tracker, graph, _, _ = _bound_aperture_filling_current()
    for offset in range(13):
        update = tracker.update(
            _frame(
                4 + offset,
                (
                    _detection(
                        0,
                        0.56 + 0.002 * offset,
                        -0.58,
                        0.13,
                        0.16,
                    ),
                ),
                publication_sequence=170 + offset,
            )
        )
        snapshot = graph.observe(tracker)
        if offset == 11:
            assert snapshot.next_candidates[0].promotable

    assert snapshot.current_track is not None
    assert snapshot.current_track.role is VisualTrackRole.RETIRED
    assert len(snapshot.next_candidates) == 1
    assert not snapshot.next_candidates[0].promotable
    with pytest.raises(
        GateGraphError,
        match="no stable pretracked next gate is promotable",
    ):
        graph.confirm_transition(
            tracker,
            race_status=_race(
                sequence=101,
                boot_ms=9_250,
                gate_index=1,
                received_ns=update.publish_monotonic_ns + 1,
            ),
            camera_token_at_credit=update.token,
        )


def test_adjacent_handoff_rejects_stale_race_credit_after_camera_freezes() -> None:
    tracker, graph, _, _ = _bound_aperture_filling_current()
    for offset in range(3):
        update = tracker.update(
            _frame(
                4 + offset,
                (_detection(0, 0.56, -0.58, 0.13, 0.16),),
                publication_sequence=170 + offset,
            )
        )
        snapshot = graph.observe(tracker)

    assert snapshot.next_candidates[0].promotable
    with pytest.raises(
        GateGraphError,
        match="stale at race credit",
    ):
        graph.confirm_transition(
            tracker,
            race_status=_race(
                sequence=102,
                boot_ms=9_250,
                gate_index=1,
                received_ns=(
                    update.publish_monotonic_ns + 100_000_001
                ),
            ),
            camera_token_at_credit=update.token,
        )
    assert tracker.track(snapshot.current_track_id).role is VisualTrackRole.CURRENT
    assert tracker.track(
        snapshot.next_candidates[0].track_id
    ).authoritative_gate_index is None


def test_adjacent_handoff_does_not_replace_recent_pretracked_candidate() -> None:
    tracker, graph, _, alternative_id = _bound_aperture_filling_current(
        include_pretracked_alternative=True,
    )
    newcomer_id = ""
    for offset in range(3):
        update = tracker.update(
            _frame(
                4 + offset,
                (_detection(0, 0.56, -0.58, 0.13, 0.16),),
                publication_sequence=170 + offset,
            )
        )
        if offset == 0:
            newcomer_id = next(
                track_id
                for track_id in update.visible_track_ids
                if track_id != alternative_id
            )
        snapshot = graph.observe(tracker)

    candidate = next(
        item for item in snapshot.next_candidates
        if item.track_id == newcomer_id
    )
    assert tracker.track(alternative_id).missed_frame_count == 3
    assert candidate.relationship is None
    assert not candidate.promotable


def test_adjacent_handoff_does_not_replace_recent_ambiguous_identities() -> None:
    tracker = MultiTargetVisualTracker(
        MultiTargetTrackerConfig(ambiguity_margin=0.20)
    )
    graph = RollingVisualGateGraph()
    all_edges = (
        FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT | FrameEdge.BOTTOM
    )
    ambiguous_ids: tuple[str, ...] = ()
    for sequence in range(1, 4):
        candidate_centers = (
            (-0.04, 0.04) if sequence < 3 else (0.0, 0.0)
        )
        update = tracker.update(
            _frame(
                sequence,
                (
                    _detection(
                        0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        clipping=all_edges,
                        center_censored=True,
                    ),
                    _detection(
                        1,
                        candidate_centers[0],
                        -0.30,
                        0.13,
                        0.16,
                    ),
                    _detection(
                        2,
                        candidate_centers[1],
                        -0.30,
                        0.13,
                        0.16,
                    ),
                ),
                publication_sequence=166 + sequence,
            )
        )
        if sequence == 3:
            ambiguous_ids = update.ambiguous_track_ids
    current_id = max(
        update.visible_tracks,
        key=lambda track: track.apparent_scale,
    ).track_id
    assert set(ambiguous_ids) == set(update.visible_track_ids) - {current_id}
    graph.bind_initial_current(
        tracker,
        track_id=current_id,
        race_status=_race(
            sequence=110,
            boot_ms=10_000,
            gate_index=0,
            received_ns=update.publish_monotonic_ns + 1,
        ),
    )

    newcomer_id = ""
    for offset in range(3):
        update = tracker.update(
            _frame(
                4 + offset,
                (_detection(0, 0.56, -0.58, 0.13, 0.16),),
                publication_sequence=170 + offset,
            )
        )
        if offset == 0:
            newcomer_id = update.visible_track_ids[0]
        snapshot = graph.observe(tracker)

    candidate = next(
        item for item in snapshot.next_candidates
        if item.track_id == newcomer_id
    )
    assert all(tracker.track(track_id).missed_frame_count == 3
               for track_id in ambiguous_ids)
    assert candidate.relationship is None
    assert not candidate.promotable


@pytest.mark.parametrize(
    "missed_publications",
    (pytest.param(1, id="one-miss"), pytest.param(2, id="two-misses")),
)
def test_pretracked_next_gate_promotes_across_bounded_camera_misses(
    missed_publications: int,
) -> None:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = ""
    next_id = ""
    for sequence in range(1, 6):
        update = tracker.update(
            _frame(
                sequence,
                (
                    _detection(0, -0.01, 0.01, 0.32, 0.36),
                    _detection(1, 0.55, -0.55, 0.14, 0.16),
                ),
            )
        )
        if sequence == 1:
            current_id, next_id = update.visible_track_ids
        if sequence == 3:
            graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=_race(
                    sequence=50,
                    boot_ms=3_000,
                    gate_index=0,
                    received_ns=update.publish_monotonic_ns + 1,
                ),
            )
        elif sequence > 3:
            graph.observe(tracker)

    proved_history = tracker.track(next_id).history
    for sequence in range(6, 6 + missed_publications):
        update = tracker.update(
            _frame(
                sequence,
                (_detection(0, -0.01, 0.01, 0.32, 0.36),),
            )
        )
        snapshot = graph.observe(tracker)

    assert tracker.track(next_id).missed_frame_count == missed_publications
    assert not tracker.track(next_id).visible
    assert tuple(item.track_id for item in snapshot.next_candidates) == (next_id,)
    assert snapshot.next_candidates[0].promotable
    assert snapshot.next_candidates[0].stable_frame_count == len(proved_history)

    transition = graph.confirm_transition(
        tracker,
        race_status=_race(
            sequence=51,
            boot_ms=3_250,
            gate_index=1,
            received_ns=update.publish_monotonic_ns + 1,
        ),
        camera_token_at_credit=update.token,
    )

    assert transition.promoted_track_id == next_id
    assert transition.promoted_latest_token_before_credit == proved_history[-1].token
    assert transition.pretransition_frame_tokens[-3:] == tuple(
        sample.token for sample in proved_history[-3:]
    )
    assert tracker.track(next_id).history == proved_history


def test_pretracked_next_gate_rejects_more_than_two_camera_misses() -> None:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = ""
    next_id = ""
    for sequence in range(1, 6):
        update = tracker.update(
            _frame(
                sequence,
                (
                    _detection(0, -0.01, 0.01, 0.32, 0.36),
                    _detection(1, 0.55, -0.55, 0.14, 0.16),
                ),
            )
        )
        if sequence == 1:
            current_id, next_id = update.visible_track_ids
        if sequence == 3:
            graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=_race(
                    sequence=60,
                    boot_ms=4_000,
                    gate_index=0,
                    received_ns=update.publish_monotonic_ns + 1,
                ),
            )
        elif sequence > 3:
            graph.observe(tracker)

    proved_history = tracker.track(next_id).history
    for sequence in range(6, 9):
        update = tracker.update(
            _frame(
                sequence,
                (_detection(0, -0.01, 0.01, 0.32, 0.36),),
            )
        )
        snapshot = graph.observe(tracker)

    assert tracker.track(next_id).missed_frame_count == 3
    assert tracker.track(next_id).history == proved_history
    assert all(
        candidate.track_id != next_id for candidate in snapshot.next_candidates
    )
    with pytest.raises(
        GateGraphError,
        match="no stable pretracked next gate is promotable",
    ):
        graph.confirm_transition(
            tracker,
            race_status=_race(
                sequence=61,
                boot_ms=4_250,
                gate_index=1,
                received_ns=update.publish_monotonic_ns + 1,
            ),
            camera_token_at_credit=update.token,
        )


def test_ambiguous_next_candidates_reject_authoritative_promotion() -> None:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    for sequence in range(1, 6):
        update = tracker.update(
            _frame(
                sequence,
                (
                    _detection(0, 0.0, 0.0, 0.30, 0.34),
                    _detection(1, 0.48, -0.30, 0.14, 0.16),
                    _detection(2, -0.48, -0.30, 0.14, 0.16),
                ),
            )
        )
        if sequence == 1:
            current_id = update.visible_track_ids[0]
        if sequence == 3:
            graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=_race(
                    sequence=30,
                    boot_ms=1_000,
                    gate_index=0,
                    received_ns=update.publish_monotonic_ns + 1,
                ),
            )
        elif sequence > 3:
            graph.observe(tracker)
    snapshot = graph.latest_snapshot
    assert snapshot is not None
    assert len(snapshot.next_candidates) == 2
    assert snapshot.next_selection_ambiguous
    with pytest.raises(AmbiguousGatePromotionError):
        graph.confirm_transition(
            tracker,
            race_status=_race(
                sequence=31,
                boot_ms=1_250,
                gate_index=1,
                received_ns=tracker.latest_update.publish_monotonic_ns + 1,
            ),
            camera_token_at_credit=tracker.latest_update.token,
        )


def test_race_finished_is_terminal_and_never_inferred_from_visual_scale() -> None:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    for sequence in range(1, 4):
        update = tracker.update(
            _frame(
                sequence,
                (_detection(0, 0.0, 0.0, 0.20 + 0.10 * sequence, 0.22 + 0.10 * sequence),),
            )
        )
    graph.bind_initial_current(
        tracker,
        track_id=update.visible_track_ids[0],
        race_status=_race(
            sequence=40,
            boot_ms=2_000,
            gate_index=7,
            received_ns=update.publish_monotonic_ns + 1,
        ),
    )
    assert not graph.latest_snapshot.race_finished
    terminal = graph.confirm_race_finished(
        tracker,
        race_status=_race(
            sequence=41,
            boot_ms=2_250,
            gate_index=7,
            received_ns=update.publish_monotonic_ns + 2,
            finished=True,
        ),
        camera_token_at_finish=update.token,
    )
    assert terminal.race_finished
    assert not terminal.authority_usable
    assert terminal.withholding_reason == "race_finished"
    with pytest.raises(GateGraphError, match="after race finish"):
        graph.confirm_transition(
            tracker,
            race_status=_race(
                sequence=42,
                boot_ms=2_500,
                gate_index=8,
                received_ns=update.publish_monotonic_ns + 3,
            ),
            camera_token_at_credit=update.token,
        )


def test_live_race_transition_cannot_cross_host_clock_identity() -> None:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    for sequence in range(1, 4):
        update = tracker.update(
            _frame(sequence, (_detection(0, 0.0, 0.0, 0.24, 0.26),))
        )
    graph.bind_initial_current(
        tracker,
        track_id=update.visible_track_ids[0],
        race_status=_race(
            sequence=50,
            boot_ms=3_000,
            gate_index=4,
            received_ns=update.publish_monotonic_ns + 1,
        ),
    )
    with pytest.raises(GateGraphError, match="host clock changed"):
        graph.confirm_race_finished(
            tracker,
            race_status=_race(
                sequence=51,
                boot_ms=3_250,
                gate_index=4,
                received_ns=update.publish_monotonic_ns + 2,
                finished=True,
                host_clock_id="different-monotonic-clock",
            ),
            camera_token_at_finish=update.token,
        )
