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
)
from planning.vq2_gate_graph import (
    AmbiguousGatePromotionError,
    AuthoritativeRaceStatusRef,
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
) -> VisualDetectionFrame:
    final_packet = 1_000_000_000 + sequence * _FRAME_PERIOD_NS
    return VisualDetectionFrame(
        token=CameraFrameToken(
            generation=generation,
            frame_id=29_000 + sequence,
            publication_sequence=sequence,
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
                _detection(0, -0.04, 0.0, 0.18, 0.20),
                _detection(1, 0.04, 0.0, 0.18, 0.20),
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
    tracker.assign_role(first.visible_track_ids[0], VisualTrackRole.CURRENT)
    with pytest.raises(ValueError, match="ambiguous"):
        tracker.confirm_authoritative_gate(
            first.visible_track_ids[0],
            gate_index=0,
            race_status_sequence=1,
            race_status_boot_ms=100,
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
    assert relationship.observation_count == 3
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
    assert after.history == before.history
    assert after.first_token == before.first_token
    assert after.role is VisualTrackRole.CURRENT
    assert after.authoritative_gate_index == 1
    assert retired.role is VisualTrackRole.RETIRED
    assert retired.authoritative_gate_index == 0


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
