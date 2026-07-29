from __future__ import annotations

import math
from types import SimpleNamespace

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
    VisualInnerApertureGeometry,
    VisualTrackRole,
    visual_track_history_sha256,
)


_FRAME_PERIOD_NS = 33_000_000


def _inner_aperture(
    center_x: float,
    center_y: float,
    *,
    half_width: float = 0.12,
    half_height: float = 0.14,
) -> VisualInnerApertureGeometry:
    return VisualInnerApertureGeometry(
        center_norm=(center_x, center_y),
        half_size_norm=(half_width, half_height),
        log_scale=math.log(math.sqrt(4.0 * half_width * half_height)),
        measurement_std=(0.01, 0.012, 0.03),
        confidence=0.91,
        clipping=FrameEdge.NONE,
        visible_edges=(
            FrameEdge.LEFT
            | FrameEdge.TOP
            | FrameEdge.RIGHT
            | FrameEdge.BOTTOM
        ),
        geometry_model_id="test-inner-quad-v1",
        covariance_model_id="test-inner-diagonal-v1",
    )


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
    inner_aperture: VisualInnerApertureGeometry | None = None,
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
        inner_aperture=inner_aperture,
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


def test_inner_aperture_geometry_is_validated_and_passage_usable() -> None:
    geometry = _inner_aperture(0.07, -0.11)

    assert geometry.fitted
    assert geometry.complete_visibility
    assert geometry.passage_usable
    with pytest.raises(ValueError, match="all present or all absent"):
        VisualInnerApertureGeometry(
            center_norm=(0.0, 0.0),
            half_size_norm=None,
            log_scale=-1.0,
            measurement_std=(0.01, 0.01, 0.03),
            confidence=0.9,
            clipping=FrameEdge.NONE,
            visible_edges=FrameEdge.NONE,
            geometry_model_id="partial-v1",
            covariance_model_id="partial-covariance-v1",
        )


def test_detector_frame_carries_parallel_aperture_geometry_into_history() -> None:
    geometry = _inner_aperture(0.025, -0.08)
    raw_detection = SimpleNamespace(
        bbox=(250, 90, 140, 160),
        center_x=320,
        center_y=170,
        confidence=0.94,
        detection_method="vq2_red_gate",
    )
    frame = VisualDetectionFrame.from_detector_results(
        (raw_detection,),
        generation=7,
        frame_id=29_001,
        publication_sequence=1,
        stream_id="vq2-camera",
        final_unique_packet_monotonic_ns=1_033_000_000,
        publish_monotonic_ns=1_034_000_000,
        time_basis_id="vq2-host-monotonic",
        aperture_geometries=(geometry,),
    )

    assert frame.detections[0].inner_aperture is geometry
    tracker = MultiTargetVisualTracker()
    update = tracker.update(frame)
    track = update.visible_tracks[0]
    assert track.history[-1].inner_aperture is geometry
    with pytest.raises(ValueError, match="aperture_geometries"):
        VisualDetectionFrame.from_detector_results(
            (raw_detection,),
            generation=7,
            frame_id=29_002,
            publication_sequence=2,
            stream_id="vq2-camera",
            final_unique_packet_monotonic_ns=1_066_000_000,
            publish_monotonic_ns=1_067_000_000,
            time_basis_id="vq2-host-monotonic",
            aperture_geometries=(),
        )


def test_inner_aperture_geometry_does_not_change_outer_support_association() -> None:
    baseline = MultiTargetVisualTracker()
    fitted = MultiTargetVisualTracker()
    outer_frames = (
        (
            _detection(0, -0.42, 0.02, 0.22, 0.26),
            _detection(1, 0.43, -0.03, 0.16, 0.18),
        ),
        (
            _detection(0, -0.36, 0.02, 0.23, 0.27),
            _detection(1, 0.38, -0.03, 0.17, 0.19),
        ),
    )
    fitted_frames = (
        (
            _detection(
                0,
                -0.42,
                0.02,
                0.22,
                0.26,
                inner_aperture=_inner_aperture(0.81, -0.70),
            ),
            _detection(
                1,
                0.43,
                -0.03,
                0.16,
                0.18,
                inner_aperture=_inner_aperture(-0.82, 0.69),
            ),
        ),
        (
            _detection(
                0,
                -0.36,
                0.02,
                0.23,
                0.27,
                inner_aperture=_inner_aperture(0.78, -0.66),
            ),
            _detection(
                1,
                0.38,
                -0.03,
                0.17,
                0.19,
                inner_aperture=_inner_aperture(-0.79, 0.65),
            ),
        ),
    )

    baseline_updates = tuple(
        baseline.update(_frame(index, detections))
        for index, detections in enumerate(outer_frames, start=1)
    )
    fitted_updates = tuple(
        fitted.update(_frame(index, detections))
        for index, detections in enumerate(fitted_frames, start=1)
    )

    for baseline_update, fitted_update in zip(
        baseline_updates,
        fitted_updates,
        strict=True,
    ):
        assert fitted_update.created_track_ids == baseline_update.created_track_ids
        assert (
            fitted_update.associated_track_ids
            == baseline_update.associated_track_ids
        )
        assert fitted_update.visible_track_ids == baseline_update.visible_track_ids
        assert fitted_update.associations == baseline_update.associations
    assert tuple(
        tuple(sample.source_index for sample in track.history)
        for track in fitted.tracks()
    ) == tuple(
        tuple(sample.source_index for sample in track.history)
        for track in baseline.tracks()
    )
    assert all(
        sample.inner_aperture is not None
        for track in fitted.tracks()
        for sample in track.history
    )


def test_processed_camera_watermark_includes_detectionless_frames() -> None:
    tracker = MultiTargetVisualTracker()
    first = _frame(
        1,
        (_detection(0, 0.0, 0.0, 0.2, 0.2),),
    )
    detectionless = _frame(2, ())
    tracker.update(first)
    tracker.update(detectionless)

    assert tracker.latest_processed_token_published_by(
        first.publish_monotonic_ns
    ) == first.token
    assert tracker.latest_processed_token_published_by(
        detectionless.publish_monotonic_ns
    ) == detectionless.token
    assert tracker.latest_processed_token_published_by(
        first.publish_monotonic_ns - 1
    ) is None
    with pytest.raises(TypeError, match="exact integer"):
        tracker.latest_processed_token_published_by(True)
    with pytest.raises(ValueError, match="non-negative"):
        tracker.latest_processed_token_published_by(-1)


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


def test_preview_associations_matches_update_without_mutating_tracker() -> None:
    tracker = MultiTargetVisualTracker()
    first = tracker.update(
        _frame(
            1,
            (
                _detection(10, -0.35, 0.02, 0.22, 0.24),
                _detection(20, 0.40, -0.03, 0.18, 0.20),
            ),
        )
    )
    left_id, right_id = first.visible_track_ids
    pending = _frame(
        2,
        (
            _detection(7, 0.38, -0.02, 0.19, 0.21),
            _detection(8, -0.32, 0.01, 0.23, 0.25),
            _detection(9, 0.88, 0.60, 0.08, 0.10),
        ),
    )
    before_update = tracker.latest_update
    before_tracks = tracker.tracks()
    before_by_id = {
        track.track_id: track
        for track in before_tracks
    }

    preview = tracker.preview_associations(pending)

    assert {
        source_index: track.track_id
        for source_index, track in preview.items()
    } == {
        7: right_id,
        8: left_id,
    }
    assert preview[7] == before_by_id[right_id]
    assert preview[8] == before_by_id[left_id]
    assert tracker.preview_associations(pending) == preview
    assert tracker.latest_update is before_update
    assert tracker.tracks() == before_tracks
    assert not tracker.has_processed_token(pending.token)

    accepted = tracker.update(pending)
    accepted_by_source = {
        evidence.detection_source_index: evidence.track_id
        for evidence in accepted.associations
        if not evidence.ambiguous
    }
    assert accepted_by_source == {
        source_index: track.track_id
        for source_index, track in preview.items()
    }
    assert len(accepted.created_track_ids) == 1


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


def test_top_clipped_tangential_fragment_switch_keeps_current_identity() -> None:
    tracker = MultiTargetVisualTracker()
    first = tracker.update(
        _frame(
            1,
            (
                VisualDetection(
                    source_index=0,
                    center_norm=(0.578125, -0.8388888888888889),
                    bbox_norm=(0.6828125, 0.0, 0.896875, 0.1638888888888889),
                    confidence=0.30839832048081417,
                    clipping=FrameEdge.TOP,
                    center_censored=True,
                ),
            ),
        )
    )
    current_id = first.visible_track_ids[0]
    tracker.assign_role(current_id, VisualTrackRole.CURRENT)

    fragmented = tracker.update(
        _frame(
            2,
            (
                VisualDetection(
                    source_index=0,
                    center_norm=(0.421875, -0.8444444444444444),
                    bbox_norm=(0.678125, 0.0, 0.74375, 0.15555555555555556),
                    confidence=0.547340975609756,
                    clipping=FrameEdge.TOP,
                    center_censored=True,
                ),
            ),
        )
    )

    retained = fragmented.track(current_id)
    assert fragmented.created_track_ids == ()
    assert fragmented.associated_track_ids == (current_id,)
    assert retained.visible is True
    assert retained.missed_frame_count == 0
    assert retained.center_norm == pytest.approx((0.421875, -0.8444444444444444))


def test_authoritative_current_fuses_live_gate1_complementary_fragments() -> None:
    tracker = MultiTargetVisualTracker()
    prior = VisualDetection(
        source_index=0,
        center_norm=(0.121875, -0.7277777777777779),
        bbox_norm=(0.4671875, 0.0, 0.65625, 0.275),
        confidence=0.7014894452020992,
        clipping=FrameEdge.TOP,
        center_censored=True,
    )
    first = tracker.update(_frame(1, (prior,)))
    current_id = first.visible_track_ids[0]
    tracker.assign_role(current_id, VisualTrackRole.CURRENT)
    tracker.confirm_authoritative_gate(
        current_id,
        gate_index=1,
        race_status_sequence=1533,
        race_status_boot_ms=6322,
    )

    upper_right = VisualDetection(
        source_index=0,
        center_norm=(0.1875, -0.8222222222222222),
        bbox_norm=(0.528125, 0.0, 0.659375, 0.18055555555555555),
        confidence=0.6405775251457124,
        clipping=FrameEdge.TOP,
        center_censored=True,
    )
    lower_left = VisualDetection(
        source_index=1,
        center_norm=(-0.018750000000000044, -0.5888888888888889),
        bbox_norm=(0.4625, 0.1361111111111111, 0.5203125, 0.2777777777777778),
        confidence=0.5225933086084383,
    )
    unrelated_later_gate = VisualDetection(
        source_index=2,
        center_norm=(-0.571875, -0.46111111111111114),
        bbox_norm=(0.1890625, 0.2111111111111111, 0.2390625, 0.33055555555555555),
        confidence=0.4756841370260929,
    )

    fused = tracker.update(
        _frame(
            2,
            (upper_right, lower_left, unrelated_later_gate),
        )
    )

    current = fused.track(current_id)
    assert current.visible
    assert current.missed_frame_count == 0
    assert current.bbox_norm == pytest.approx(
        (0.4625, 0.0, 0.659375, 0.2777777777777778)
    )
    assert current.center_norm == pytest.approx(
        (0.121875, -0.7222222222222222)
    )
    assert current.clipping is FrameEdge.TOP
    assert current.center_censored
    assert current.history[-1].inner_aperture is None
    assert fused.associated_track_ids == (current_id,)
    assert len(fused.created_track_ids) == 1
    assert fused.associations[0].bbox_iou == pytest.approx(
        0.9507142857142858
    )


def test_non_authoritative_track_fuses_complementary_fragments() -> None:
    # F41 (run 20260729T200803Z-visual-course-a540212f): the diagonal
    # split is a detector artifact, so a next-gate track without any
    # race-authoritative identity must fuse the same way the
    # authoritative CURRENT gate does.
    tracker = MultiTargetVisualTracker()
    prior = VisualDetection(
        source_index=0,
        center_norm=(0.121875, -0.7277777777777779),
        bbox_norm=(0.4671875, 0.0, 0.65625, 0.275),
        confidence=0.70,
        clipping=FrameEdge.TOP,
        center_censored=True,
    )
    first = tracker.update(_frame(1, (prior,)))
    track_id = first.visible_track_ids[0]

    split = tracker.update(
        _frame(
            2,
            (
                VisualDetection(
                    source_index=0,
                    center_norm=(0.1875, -0.8222222222222222),
                    bbox_norm=(
                        0.528125,
                        0.0,
                        0.659375,
                        0.18055555555555555,
                    ),
                    confidence=0.64,
                    clipping=FrameEdge.TOP,
                    center_censored=True,
                ),
                VisualDetection(
                    source_index=1,
                    center_norm=(
                        -0.018750000000000044,
                        -0.5888888888888889,
                    ),
                    bbox_norm=(
                        0.4625,
                        0.1361111111111111,
                        0.5203125,
                        0.2777777777777778,
                    ),
                    confidence=0.52,
                ),
            ),
        )
    )

    retained = split.track(track_id)
    assert retained.visible
    assert retained.missed_frame_count == 0
    assert retained.bbox_norm == pytest.approx(
        (0.4625, 0.0, 0.659375, 0.2777777777777778)
    )
    assert retained.center_norm == pytest.approx(
        (0.121875, -0.7222222222222222)
    )
    assert retained.clipping is FrameEdge.TOP
    assert retained.center_censored
    assert retained.history[-1].inner_aperture is None
    assert split.associated_track_ids == (track_id,)
    assert split.created_track_ids == ()
    assert split.associations[0].bbox_iou == pytest.approx(
        0.9507142857142858
    )


def test_non_authoritative_track_rejects_non_diagonal_fragment_pair() -> None:
    tracker = MultiTargetVisualTracker()
    prior = VisualDetection(
        source_index=0,
        center_norm=(0.121875, -0.7277777777777779),
        bbox_norm=(0.4671875, 0.0, 0.65625, 0.275),
        confidence=0.70,
        clipping=FrameEdge.TOP,
        center_censored=True,
    )
    first = tracker.update(_frame(1, (prior,)))
    track_id = first.visible_track_ids[0]

    upper_right = VisualDetection(
        source_index=0,
        center_norm=(0.1875, -0.8222222222222222),
        bbox_norm=(
            0.528125,
            0.0,
            0.659375,
            0.18055555555555555,
        ),
        confidence=0.64,
        clipping=FrameEdge.TOP,
        center_censored=True,
    )
    # Below the upper contour but to its RIGHT: the complementary
    # diagonal-split direction is lower/LEFT, so this pair must not fuse.
    lower_right = _detection(1, 0.45, -0.58, 0.10, 0.14, confidence=0.52)

    split = tracker.update(_frame(2, (upper_right, lower_right)))

    retained = split.track(track_id)
    assert retained.visible
    assert retained.missed_frame_count == 0
    assert retained.bbox_norm == pytest.approx(
        (0.528125, 0.0, 0.659375, 0.18055555555555555)
    )
    assert split.associated_track_ids == (track_id,)
    assert len(split.created_track_ids) == 1


def test_authoritative_fragment_union_keeps_prior_consistent_live_aspect() -> None:
    tracker = MultiTargetVisualTracker()
    prior = VisualDetection(
        source_index=0,
        center_norm=(0.50625, -0.6388888888888888),
        bbox_norm=(0.60625, 0.0, 0.9015625, 0.3638888888888889),
        confidence=0.70,
        clipping=FrameEdge.TOP,
        center_censored=True,
    )
    first = tracker.update(_frame(1, (prior,)))
    current_id = first.visible_track_ids[0]
    tracker.assign_role(current_id, VisualTrackRole.CURRENT)
    tracker.confirm_authoritative_gate(
        current_id,
        gate_index=1,
        race_status_sequence=1531,
        race_status_boot_ms=6322,
    )

    split = tracker.update(
        _frame(
            2,
            (
                VisualDetection(
                    source_index=0,
                    center_norm=(0.6375, -0.7138888888888889),
                    bbox_norm=(
                        0.725,
                        0.0,
                        0.9125,
                        0.2861111111111111,
                    ),
                    confidence=0.64,
                    clipping=FrameEdge.TOP,
                    center_censored=True,
                ),
                VisualDetection(
                    source_index=1,
                    center_norm=(0.271875, -0.41666666666666663),
                    bbox_norm=(
                        0.60625,
                        0.21388888888888888,
                        0.665625,
                        0.36944444444444446,
                    ),
                    confidence=0.48,
                ),
            ),
        )
    )

    current = split.track(current_id)
    assert current.visible
    assert current.bbox_norm == pytest.approx(
        (0.60625, 0.0, 0.9125, 0.36944444444444446)
    )
    assert current.center_norm == pytest.approx(
        (0.51875, -0.6333333333333333)
    )
    assert split.created_track_ids == ()
    assert split.associations[0].bbox_iou > 0.94


def test_authoritative_current_fuses_live_interior_diagonal_fragments() -> None:
    tracker = MultiTargetVisualTracker()
    prior = VisualDetection(
        source_index=0,
        center_norm=(0.1468750000000001, -0.5888888888888889),
        bbox_norm=(0.484375, 0.05, 0.6625, 0.3638888888888889),
        confidence=0.79308954367804,
    )
    first = tracker.update(_frame(1, (prior,)))
    current_id = first.visible_track_ids[0]
    tracker.assign_role(current_id, VisualTrackRole.CURRENT)
    tracker.confirm_authoritative_gate(
        current_id,
        gate_index=1,
        race_status_sequence=1530,
        race_status_boot_ms=6308,
    )

    split_frame = _frame(
        3,
        (
            VisualDetection(
                source_index=0,
                center_norm=(0.203125, -0.6833333333333333),
                bbox_norm=(
                    0.5375,
                    0.03611111111111111,
                    0.6671875,
                    0.28055555555555556,
                ),
                confidence=0.7800275347353138,
            ),
            VisualDetection(
                source_index=1,
                center_norm=(0.009374999999999911, -0.4),
                bbox_norm=(
                    0.4765625,
                    0.2361111111111111,
                    0.534375,
                    0.36666666666666664,
                ),
                confidence=0.5287835617348927,
            ),
        ),
    )
    preview = tracker.preview_associations(split_frame)
    assert preview[0].track_id == current_id

    fused = tracker.update(split_frame)

    current = fused.track(current_id)
    assert current.visible
    assert current.bbox_norm == pytest.approx(
        (
            0.4765625,
            0.03611111111111111,
            0.6671875,
            0.36666666666666664,
        )
    )
    assert current.center_norm == pytest.approx((0.14375, -0.6))
    assert current.clipping == FrameEdge.NONE
    assert not current.center_censored
    assert current.history[-1].inner_aperture is None
    assert fused.created_track_ids == ()
    assert fused.associations[0].bbox_iou > 0.88


def test_authoritative_current_fuses_near_plane_open_aperture_fragments() -> None:
    tracker = MultiTargetVisualTracker()
    prior = VisualDetection(
        source_index=0,
        center_norm=(0.609375, -0.6277777777777778),
        bbox_norm=(0.609375, 0.0, 1.0, 0.37222222222222223),
        confidence=0.3839373314579045,
        clipping=FrameEdge.TOP | FrameEdge.RIGHT,
        center_censored=True,
    )
    first = tracker.update(_frame(1, (prior,)))
    current_id = first.visible_track_ids[0]
    tracker.assign_role(current_id, VisualTrackRole.CURRENT)
    tracker.confirm_authoritative_gate(
        current_id,
        gate_index=1,
        race_status_sequence=1529,
        race_status_boot_ms=6300,
    )

    split_frame = _frame(
        2,
        (
            VisualDetection(
                source_index=0,
                center_norm=(0.8531249999999999, -0.7166666666666667),
                bbox_norm=(
                    0.8546875,
                    0.0,
                    1.0,
                    0.2833333333333333,
                ),
                confidence=0.612283494808231,
                clipping=FrameEdge.TOP | FrameEdge.RIGHT,
                center_censored=True,
            ),
            VisualDetection(
                source_index=1,
                center_norm=(0.27812499999999996, -0.4222222222222223),
                bbox_norm=(
                    0.6125,
                    0.2,
                    0.665625,
                    0.37777777777777777,
                ),
                confidence=0.3468086112554202,
            ),
        ),
    )
    preview = tracker.preview_associations(split_frame)
    assert preview[0].track_id == current_id

    fused = tracker.update(split_frame)

    current = fused.track(current_id)
    assert current.visible
    assert current.bbox_norm == pytest.approx(
        (0.6125, 0.0, 1.0, 0.37777777777777777)
    )
    assert current.center_norm == pytest.approx(
        (0.6125, -0.6222222222222222)
    )
    assert current.clipping == FrameEdge.TOP | FrameEdge.RIGHT
    assert current.center_censored
    assert current.history[-1].inner_aperture is None
    assert fused.created_track_ids == ()
    assert fused.associations[0].bbox_iou > 0.97


def test_authoritative_current_keeps_thin_near_plane_frame_fragments() -> None:
    tracker = MultiTargetVisualTracker()
    prior = VisualDetection(
        source_index=0,
        center_norm=(0.61875, -0.6111111111111112),
        bbox_norm=(
            0.6203125,
            0.0,
            1.0,
            0.39166666666666666,
        ),
        confidence=0.3123327546631671,
        clipping=FrameEdge.TOP | FrameEdge.RIGHT,
        center_censored=True,
    )
    first = tracker.update(_frame(1, (prior,)))
    current_id = first.visible_track_ids[0]
    tracker.assign_role(current_id, VisualTrackRole.CURRENT)
    tracker.confirm_authoritative_gate(
        current_id,
        gate_index=1,
        race_status_sequence=1532,
        race_status_boot_ms=6400,
    )

    fused = tracker.update(
        _frame(
            2,
            (
                VisualDetection(
                    source_index=2,
                    center_norm=(0.946875, -0.6888888888888889),
                    bbox_norm=(
                        0.9484375,
                        0.0,
                        1.0,
                        0.3138888888888889,
                    ),
                    confidence=0.3354557828562919,
                    clipping=FrameEdge.TOP | FrameEdge.RIGHT,
                    center_censored=True,
                ),
                VisualDetection(
                    source_index=1,
                    center_norm=(0.30000000000000004, -0.38888888888888884),
                    bbox_norm=(0.625, 0.2111111111111111, 0.675, 0.4),
                    confidence=0.2824761012126641,
                ),
            ),
        )
    )

    current = fused.track(current_id)
    assert current.visible
    assert current.bbox_norm == pytest.approx((0.625, 0.0, 1.0, 0.4))
    assert current.center_norm == pytest.approx((0.625, -0.6))
    assert current.clipping == FrameEdge.TOP | FrameEdge.RIGHT
    assert current.center_censored
    assert current.history[-1].inner_aperture is None
    assert fused.created_track_ids == ()
    assert fused.associations[0].bbox_iou == pytest.approx(
        0.9673269873938771
    )


def test_authoritative_current_projects_live_top_right_single_fragment() -> None:
    """Keep the exact run-23 Gate 1 lower-left contour as CURRENT."""

    tracker = MultiTargetVisualTracker()
    prior = VisualDetection(
        source_index=0,
        center_norm=(0.478125, -0.5666666666666667),
        bbox_norm=(0.478125, 0.0, 1.0, 0.4361111111111111),
        confidence=0.30975,
        clipping=FrameEdge.TOP | FrameEdge.RIGHT,
        center_censored=True,
    )
    first = tracker.update(_frame(224, (prior,)))
    current_id = first.visible_track_ids[0]
    tracker.assign_role(current_id, VisualTrackRole.CURRENT)
    tracker.confirm_authoritative_gate(
        current_id,
        gate_index=1,
        race_status_sequence=1529,
        race_status_boot_ms=6300,
    )

    live_fragments = (
        VisualDetection(
            source_index=0,
            center_norm=(0.021875, -0.34444444444444444),
            bbox_norm=(
                0.484375,
                0.21388888888888888,
                0.5390625,
                0.4444444444444444,
            ),
            confidence=0.23960786,
        ),
        VisualDetection(
            source_index=0,
            center_norm=(0.03125, -0.3388888888888889),
            bbox_norm=(
                0.4890625,
                0.21388888888888888,
                0.5421875,
                0.45,
            ),
            confidence=0.22285254,
        ),
        VisualDetection(
            source_index=0,
            center_norm=(0.04375, -0.3222222222222222),
            bbox_norm=(
                0.496875,
                0.21666666666666667,
                0.546875,
                0.46111111111111114,
            ),
            confidence=0.17275413,
        ),
    )
    expected_projected_boxes = (
        (0.484375, 0.0, 1.0, 0.4444444444444444),
        (0.4890625, 0.0, 1.0, 0.45),
        (0.496875, 0.0, 1.0, 0.46111111111111114),
    )

    for sequence, (fragment, expected_bbox) in enumerate(
        zip(live_fragments, expected_projected_boxes, strict=True),
        start=225,
    ):
        continued = tracker.update(_frame(sequence, (fragment,)))
        current = continued.track(current_id)
        assert current.visible
        assert current.missed_frame_count == 0
        assert current.bbox_norm == pytest.approx(expected_bbox)
        assert current.clipping == FrameEdge.TOP | FrameEdge.RIGHT
        assert current.center_censored
        assert current.history[-1].inner_aperture is None
        assert continued.associated_track_ids == (current_id,)
        assert continued.created_track_ids == ()
        assert continued.associations[0].bbox_iou > 0.96


def test_edge_transition_quarantines_only_censored_box_derivatives() -> None:
    tracker = MultiTargetVisualTracker()
    first = tracker.update(
        _frame(
            1,
            (
                _detection(
                    0,
                    0.60,
                    -0.50,
                    0.28,
                    0.30,
                    clipping=FrameEdge.TOP | FrameEdge.RIGHT,
                    center_censored=True,
                ),
            ),
        )
    )
    track_id = first.visible_track_ids[0]

    vertical_uncensor = tracker.update(
        _frame(
            2,
            (
                _detection(
                    0,
                    0.62,
                    -0.20,
                    0.30,
                    0.32,
                    clipping=FrameEdge.RIGHT,
                    center_censored=True,
                ),
            ),
        )
    ).track(track_id)
    assert vertical_uncensor.center_velocity_norm_s == (0.0, 0.0)
    assert vertical_uncensor.log_scale_rate_s == 0.0

    vertically_observable = tracker.update(
        _frame(
            3,
            (
                _detection(
                    0,
                    0.64,
                    -0.18,
                    0.31,
                    0.33,
                    clipping=FrameEdge.RIGHT,
                    center_censored=True,
                ),
            ),
        )
    ).track(track_id)
    assert vertically_observable.center_velocity_norm_s[0] == 0.0
    assert vertically_observable.center_velocity_norm_s[1] > 0.0
    assert vertically_observable.log_scale_rate_s == 0.0

    horizontal_uncensor = tracker.update(
        _frame(
            4,
            (
                _detection(
                    0,
                    0.34,
                    -0.16,
                    0.32,
                    0.34,
                ),
            ),
        )
    ).track(track_id)
    assert horizontal_uncensor.center_velocity_norm_s[0] == 0.0
    assert horizontal_uncensor.center_velocity_norm_s[1] > 0.0
    assert horizontal_uncensor.log_scale_rate_s == 0.0


def test_same_edge_implausible_contraction_becomes_a_visibility_gap() -> None:
    tracker = MultiTargetVisualTracker()
    first = tracker.update(
        _frame(
            1,
            (
                _detection(
                    0,
                    0.178,
                    -0.811,
                    0.236,
                    0.192,
                    clipping=FrameEdge.TOP,
                    center_censored=True,
                ),
            ),
        )
    )
    track_id = first.visible_track_ids[0]
    tracker.assign_role(track_id, VisualTrackRole.CURRENT)

    collapsed = tracker.update(
        _frame(
            2,
            (
                _detection(
                    0,
                    0.281,
                    -0.90,
                    0.172,
                    0.10,
                    clipping=FrameEdge.TOP,
                    center_censored=True,
                ),
            ),
        )
    )

    retained = collapsed.track(track_id)
    assert retained.visible is False
    assert retained.missed_frame_count == 1
    assert retained.latest_token == first.track(track_id).latest_token
    assert track_id not in collapsed.associated_track_ids
    assert len(collapsed.created_track_ids) == 1


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
    ambiguous_frame = _frame(
        2,
        (
            _detection(0, 0.00, 0.0, 0.18, 0.20),
            _detection(1, 0.00, 0.0, 0.18, 0.20),
        ),
    )
    before_tracks = tracker.tracks()
    before_update = tracker.latest_update
    assert tracker.preview_associations(ambiguous_frame) == {}
    assert tracker.tracks() == before_tracks
    assert tracker.latest_update is before_update
    assert not tracker.has_processed_token(ambiguous_frame.token)

    second = tracker.update(ambiguous_frame)
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
        assert not accepted.track_ambiguous_before_association
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


