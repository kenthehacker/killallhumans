from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetVisualTracker,
    VisualDetection,
    VisualDetectionFrame,
    VisualTrackRole,
    visual_track_history_sha256,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    GateGraphError,
    RollingVisualGateGraph,
)


_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "vq2_gate0_observe_20260718T022458_tracker_excerpt.json"
)
_SOURCE_SHA256 = "2fca8cc6d2b5ed0ced6dca8e7683254c60eb774571a7756a315126022032bfd6"
_LIVE_FRAME_PERIOD_NS = 33_000_000


def _load_excerpt() -> dict:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert payload["schema"] == "aigp-vq2-recorded-detector-excerpt/1"
    assert payload["source"] == {
        "path": "captures/vq2_gate0-observe_20260718T022458.jsonl.gz",
        "sha256": _SOURCE_SHA256,
        "event": "vision_detection_frame",
        "record_number_basis": "one-based decompressed JSONL line order",
        "simulator_build": "3385",
        "simulator_mode": "Training",
    }
    assert payload["extraction"]["unobserved_frame_ids"] == [296782]
    return payload


def _recorded_frame(row: dict) -> VisualDetectionFrame:
    width_px, height_px = row["image_size_px"]
    detector_results = []
    for detection in row["detections"]:
        assert detection["selector_eligible"] is True
        x, y, width, height = detection["bbox_xywh_px"]
        center_x, center_y = detection["center_px"]
        assert detection["reported_area_px"] == width * height
        assert 0 <= x <= center_x <= x + width <= width_px
        assert 0 <= y <= center_y <= y + height <= height_px
        detector_results.append(
            SimpleNamespace(
                bbox=(x, y, width, height),
                center_x=center_x,
                center_y=center_y,
                confidence=detection["confidence"],
                detection_method=detection["method"],
            )
        )
    generation, frame_id, source_time_ns = row["frame_token"]
    return VisualDetectionFrame.from_legacy_detector_results(
        detector_results,
        generation=generation,
        frame_id=frame_id,
        camera_source_time_ns=source_time_ns,
        received_monotonic_s=row["received_monotonic_s"],
        image_size_px=(width_px, height_px),
    )


def _legacy_race_status(payload: dict, row: dict) -> AuthoritativeRaceStatusRef:
    race_status_boot_ms = (
        row["race_status_boot_ms"]
        if "race_status_boot_ms" in row
        else row["post_race_boot_ms"]
    )
    active_gate_index = (
        row["active_gate_index"]
        if "active_gate_index" in row
        else row["post_gate_index"]
    )
    return AuthoritativeRaceStatusRef.legacy_capture(
        session_id=f"sha256:{payload['source']['sha256']}",
        reset_epoch=payload["frames"][0]["frame_token"][0],
        legacy_event_order=row["source_record_number"],
        event_wall_time_ns=row["event_wall_time_ns"],
        race_status_boot_ms=race_status_boot_ms,
        active_gate_index=active_gate_index,
    )


def _live_detection(
    source_index: int,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
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
        confidence=0.9,
    )


def _live_frame(
    sequence: int,
    *,
    include_next: bool = True,
) -> VisualDetectionFrame:
    packet_ns = 1_000_000_000 + sequence * _LIVE_FRAME_PERIOD_NS
    detections = [
        _live_detection(
            0,
            -0.03 + 0.004 * sequence,
            0.02,
            0.30 + 0.005 * sequence,
            0.34 + 0.005 * sequence,
        )
    ]
    if include_next:
        detections.append(
            _live_detection(
                1,
                0.52 + 0.004 * sequence,
                -0.56 - 0.003 * sequence,
                0.13 + 0.003 * sequence,
                0.15 + 0.003 * sequence,
            )
        )
    return VisualDetectionFrame(
        token=CameraFrameToken(
            generation=9,
            frame_id=40_000 + sequence,
            publication_sequence=sequence,
            stream_id="vq2-credit-boundary-camera",
        ),
        provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
        time_basis_id="vq2-credit-boundary-clock",
        image_size_px=(640, 360),
        detections=tuple(detections),
        camera_source_time_ns=5_000_000_000 + sequence * _LIVE_FRAME_PERIOD_NS,
        final_unique_packet_monotonic_ns=packet_ns,
        publish_monotonic_ns=packet_ns + 1_000_000,
    )


def _live_race(
    *,
    sequence: int,
    boot_ms: int,
    gate_index: int,
    received_ns: int,
) -> AuthoritativeRaceStatusRef:
    return AuthoritativeRaceStatusRef.live(
        session_id="credit-boundary-session",
        reset_epoch=9,
        race_generation=4,
        race_status_sequence=sequence,
        race_status_boot_ms=boot_ms,
        active_gate_index=gate_index,
        received_monotonic_ns=received_ns,
        host_clock_id="vq2-credit-boundary-clock",
    )


def _prime_live_graph(
    *,
    final_sequence: int,
    missing_next_sequences: tuple[int, ...] = (),
) -> tuple[MultiTargetVisualTracker, RollingVisualGateGraph, str, str]:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = ""
    next_id = ""
    for sequence in range(1, final_sequence + 1):
        update = tracker.update(
            _live_frame(
                sequence,
                include_next=sequence not in missing_next_sequences,
            )
        )
        if sequence == 1:
            current_id, next_id = update.visible_track_ids
        if sequence == 3:
            graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=_live_race(
                    sequence=10,
                    boot_ms=5_000,
                    gate_index=0,
                    received_ns=update.publish_monotonic_ns + 1,
                ),
            )
        elif sequence > 3:
            graph.observe(tracker)
    return tracker, graph, current_id, next_id


def test_recorded_next_gate_is_pretracked_and_promoted_without_reset() -> None:
    payload = _load_excerpt()
    frames = payload["frames"]
    assert [row["frame_token"][1] for row in frames] == [
        296777,
        296778,
        296779,
        296780,
        296781,
        296783,
        296784,
        296785,
        296786,
        296787,
        296788,
        296789,
        296790,
        296791,
    ]

    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_track_id = None
    next_track_id = None
    next_track_ids_before_credit = []
    association_confidences = []
    transition = None
    history_before_promotion = None

    for row in frames:
        frame = _recorded_frame(row)
        assert frame.provenance_basis is FrameProvenanceBasis.LEGACY_CAPTURE
        assert frame.token.publication_sequence is None
        assert frame.publish_monotonic_ns is None
        update = tracker.update(frame)
        frame_id = row["frame_token"][1]

        if frame_id == 296779:
            assert len(update.visible_tracks) == 1
            current_track_id = update.visible_tracks[0].track_id
            baseline = payload["initial_race_status"]
            assert baseline["camera_frame_token"] == row["frame_token"]
            snapshot = graph.bind_initial_current(
                tracker,
                track_id=current_track_id,
                race_status=_legacy_race_status(payload, baseline),
            )
            assert snapshot.current_track_id == current_track_id
            assert snapshot.current_gate_index == 0
            continue

        if current_track_id is None:
            continue
        snapshot = graph.observe(tracker)

        if frame_id == 296780:
            visible_noncurrent = [
                track
                for track in update.visible_tracks
                if track.track_id != current_track_id
            ]
            assert len(visible_noncurrent) == 1
            next_track_id = visible_noncurrent[0].track_id
            assert next_track_id != current_track_id

        if next_track_id is not None and 296780 <= frame_id <= 296787:
            observed = tracker.track(next_track_id)
            assert observed.visible
            assert not observed.ambiguous
            next_track_ids_before_credit.append(observed.track_id)
            evidence = next(
                (
                    item
                    for item in update.associations
                    if item.track_id == next_track_id
                ),
                None,
            )
            if evidence is not None:
                assert not evidence.ambiguous
                association_confidences.append(evidence.confidence)

        if frame_id == 296788:
            assert next_track_id is not None
            promoted_before = tracker.track(next_track_id)
            history_before_promotion = promoted_before.history
            transition_row = payload["authoritative_transition"]
            assert transition_row["decision"] == "passed"
            assert row["race"] == {
                "sim_boot_time_ms": transition_row["post_race_boot_ms"],
                "active_gate_index": transition_row["post_gate_index"],
            }
            transition = graph.confirm_transition(
                tracker,
                race_status=_legacy_race_status(payload, transition_row),
                camera_token_at_credit=CameraFrameToken(
                    generation=row["frame_token"][0],
                    frame_id=row["frame_token"][1],
                ),
                promoted_track_id=next_track_id,
            )
            promoted_after = tracker.track(next_track_id)
            assert promoted_after.history == history_before_promotion
            assert promoted_after.first_token == promoted_before.first_token
            assert promoted_after.track_id == promoted_before.track_id

    assert current_track_id is not None
    assert next_track_id is not None
    assert transition is not None
    assert history_before_promotion is not None

    # Seven fresh pre-credit rows prove a stable secondary identity before the
    # authoritative race packet. The source's absent frame 296782 is not filled.
    assert next_track_ids_before_credit == [next_track_id] * 7
    assert min(association_confidences) > 0.90
    assert transition.from_gate_index == 0
    assert transition.to_gate_index == 1
    assert transition.retired_track_id == current_track_id
    assert transition.promoted_track_id == next_track_id
    assert transition.race_status.race_status_sequence is None
    assert transition.race_status.race_status_boot_ms == 6256
    assert transition.promoted_first_token == CameraFrameToken(1, 296780)
    assert transition.promoted_latest_token_before_credit == CameraFrameToken(
        1, 296788
    )
    assert transition.promoted_history_length_at_credit == 8
    assert transition.promoted_latest_token_at_promotion == CameraFrameToken(
        1, 296788
    )
    assert transition.history_length_before_promotion == 8
    assert transition.history_length_after_promotion == 8
    assert [token.frame_id for token in transition.pretransition_frame_tokens] == [
        296780,
        296781,
        296783,
        296784,
        296785,
        296786,
        296787,
        296788,
    ]

    retired = tracker.track(current_track_id)
    promoted = tracker.track(next_track_id)
    assert retired.role is VisualTrackRole.RETIRED
    assert retired.authoritative_gate_index == 0
    assert promoted.role is VisualTrackRole.CURRENT
    assert promoted.authoritative_gate_index == 1
    assert promoted.authority_race_status_sequence is None
    assert promoted.first_token == CameraFrameToken(1, 296780)
    assert promoted.latest_token == CameraFrameToken(1, 296791)
    assert promoted.total_observation_count == 11
    assert promoted.consecutive_frame_count == 11
    assert not promoted.ambiguous

    recorded_next = [
        row
        for row in frames
        if 296780 <= row["frame_token"][1] <= 296791
    ]
    assert len(recorded_next) == len(promoted.history)
    expected_centers = []
    expected_scales = []
    expected_confidences = []
    expected_source_times = []
    for row in recorded_next:
        detection = next(
            item for item in row["detections"] if not item["crossing_residue"]
        )
        width_px, height_px = row["image_size_px"]
        center_x, center_y = detection["center_px"]
        _x, _y, bbox_width, bbox_height = detection["bbox_xywh_px"]
        expected_centers.append(
            (
                2.0 * center_x / width_px - 1.0,
                2.0 * center_y / height_px - 1.0,
            )
        )
        expected_scales.append(
            math.sqrt(bbox_width * bbox_height / (width_px * height_px))
        )
        expected_confidences.append(detection["confidence"])
        expected_source_times.append(row["frame_token"][2])

    assert [sample.center_norm for sample in promoted.history] == pytest.approx(
        expected_centers
    )
    assert [sample.apparent_scale for sample in promoted.history] == pytest.approx(
        expected_scales
    )
    assert [sample.confidence for sample in promoted.history] == pytest.approx(
        expected_confidences
    )
    assert [
        sample.camera_source_time_ns for sample in promoted.history
    ] == expected_source_times
    assert [
        sample.observation_monotonic_ns for sample in promoted.history
    ] == [
        round(row["received_monotonic_s"] * 1_000_000_000)
        for row in recorded_next
    ]
    assert promoted.bearing_rate_norm_s > 0.0
    assert promoted.elevation_rate_norm_s > 0.0
    assert promoted.log_scale_rate_s > 0.0
    assert promoted.association_confidence > 0.90
    assert promoted.center_censored

    final_snapshot = graph.latest_snapshot
    assert final_snapshot is not None
    assert final_snapshot.current_track_id == next_track_id
    assert final_snapshot.current_gate_index == 1
    assert final_snapshot.current_track == promoted
    assert final_snapshot.confirmed_transitions == (transition,)


def test_live_promotion_freezes_one_post_credit_sample_outside_credit_prefix() -> None:
    tracker, graph, current_id, next_id = _prime_live_graph(final_sequence=6)
    camera_token_at_credit = CameraFrameToken(
        generation=9,
        frame_id=40_005,
        publication_sequence=5,
        stream_id="vq2-credit-boundary-camera",
    )
    credit_received_ns = (
        tracker.frame_publish_time_ns(camera_token_at_credit) + 2_000_000
    )
    before = tracker.track(next_id)
    assert len(before.history) == 6
    assert before.history[-2].token == camera_token_at_credit
    assert before.history[-2].publication_monotonic_ns < credit_received_ns
    assert before.history[-1].publication_monotonic_ns > credit_received_ns

    transition = graph.confirm_transition(
        tracker,
        race_status=_live_race(
            sequence=11,
            boot_ms=5_250,
            gate_index=1,
            received_ns=credit_received_ns,
        ),
        camera_token_at_credit=camera_token_at_credit,
        promoted_track_id=next_id,
    )

    assert transition.retired_track_id == current_id
    assert transition.promoted_track_id == next_id
    assert transition.camera_token_at_credit == camera_token_at_credit
    assert transition.promoted_history_length_at_credit == 5
    assert transition.promoted_latest_token_before_credit == before.history[4].token
    assert transition.promoted_latest_token_at_promotion == before.history[5].token
    assert transition.history_length_before_promotion == 6
    assert transition.history_length_after_promotion == 6
    assert transition.promoted_history_sha256 == visual_track_history_sha256(
        before.history
    )
    assert transition.pretransition_frame_tokens == tuple(
        sample.token for sample in before.history[:5]
    )
    assert tracker.track(next_id).history == before.history


def test_camera_credit_watermark_need_not_be_a_target_observation() -> None:
    tracker, graph, _current_id, next_id = _prime_live_graph(
        final_sequence=6,
        missing_next_sequences=(6,),
    )
    camera_token_at_credit = tracker.latest_update.token
    credit_received_ns = tracker.latest_update.publish_monotonic_ns + 1
    before = tracker.track(next_id)
    assert before.latest_token.publication_sequence == 5
    assert camera_token_at_credit.publication_sequence == 6

    transition = graph.confirm_transition(
        tracker,
        race_status=_live_race(
            sequence=11,
            boot_ms=5_250,
            gate_index=1,
            received_ns=credit_received_ns,
        ),
        camera_token_at_credit=camera_token_at_credit,
        promoted_track_id=next_id,
    )

    assert transition.camera_token_at_credit == camera_token_at_credit
    assert transition.promoted_latest_token_before_credit == before.latest_token
    assert transition.promoted_latest_token_at_promotion == before.latest_token
    assert transition.promoted_history_length_at_credit == len(before.history)
    assert transition.history_length_before_promotion == len(before.history)


def test_live_promotion_preserves_multiple_post_credit_30hz_samples() -> None:
    tracker, graph, current_id, next_id = _prime_live_graph(final_sequence=8)
    camera_token_at_credit = CameraFrameToken(
        generation=9,
        frame_id=40_005,
        publication_sequence=5,
        stream_id="vq2-credit-boundary-camera",
    )
    credit_received_ns = (
        tracker.frame_publish_time_ns(camera_token_at_credit) + 2_000_000
    )
    before = tracker.track(next_id)
    assert len(before.history) == 8
    assert before.history[4].token == camera_token_at_credit
    assert all(
        sample.observation_monotonic_ns > credit_received_ns
        and sample.publication_monotonic_ns > credit_received_ns
        and sample.token.publication_sequence > 5
        for sample in before.history[5:]
    )

    transition = graph.confirm_transition(
        tracker,
        race_status=_live_race(
            sequence=11,
            boot_ms=5_250,
            gate_index=1,
            received_ns=credit_received_ns,
        ),
        camera_token_at_credit=camera_token_at_credit,
        promoted_track_id=next_id,
    )

    assert transition.retired_track_id == current_id
    assert transition.promoted_track_id == next_id
    assert transition.promoted_history_length_at_credit == 5
    assert transition.promoted_latest_token_before_credit == (
        camera_token_at_credit
    )
    assert transition.promoted_latest_token_at_promotion == before.history[-1].token
    assert transition.history_length_before_promotion == len(before.history)
    assert transition.history_length_after_promotion == len(before.history)
    assert transition.promoted_history_sha256 == visual_track_history_sha256(
        before.history
    )
    assert tracker.track(next_id).history == before.history
    assert tracker.track(current_id).role is VisualTrackRole.RETIRED
    assert tracker.track(next_id).role is VisualTrackRole.CURRENT


def test_live_promotion_rejects_insufficient_fresh_precredit_next_history() -> None:
    tracker, graph, current_id, next_id = _prime_live_graph(
        final_sequence=5,
        missing_next_sequences=(3,),
    )
    camera_token_at_credit = tracker.latest_update.token
    credit_received_ns = tracker.latest_update.publish_monotonic_ns + 1
    current_before = tracker.track(current_id)
    next_before = tracker.track(next_id)

    with pytest.raises(
        GateGraphError,
        match="no stable pretracked next gate is promotable",
    ):
        graph.confirm_transition(
            tracker,
            race_status=_live_race(
                sequence=11,
                boot_ms=5_250,
                gate_index=1,
                received_ns=credit_received_ns,
            ),
            camera_token_at_credit=camera_token_at_credit,
            promoted_track_id=next_id,
        )

    assert tracker.track(current_id) == current_before
    assert tracker.track(next_id) == next_before
    assert graph.latest_snapshot.current_track_id == current_id


def test_live_promotion_rejects_ambiguous_stable_next_candidates() -> None:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = ""
    next_ids: tuple[str, ...] = ()
    for sequence in range(1, 6):
        base = _live_frame(sequence)
        mirrored_next = _live_detection(
            2,
            -0.52 - 0.004 * sequence,
            -0.56 - 0.003 * sequence,
            0.13 + 0.003 * sequence,
            0.15 + 0.003 * sequence,
        )
        update = tracker.update(
            replace(
                base,
                detections=base.detections + (mirrored_next,),
            )
        )
        if sequence == 1:
            current_id = update.visible_track_ids[0]
            next_ids = update.visible_track_ids[1:]
            assert len(next_ids) == 2
        if sequence == 3:
            graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=_live_race(
                    sequence=10,
                    boot_ms=5_000,
                    gate_index=0,
                    received_ns=update.publish_monotonic_ns + 1,
                ),
            )
        elif sequence > 3:
            graph.observe(tracker)

    camera_token_at_credit = tracker.latest_update.token
    credit_received_ns = tracker.latest_update.publish_monotonic_ns + 1
    tracks_before = tracker.tracks()

    with pytest.raises(
        GateGraphError,
        match="multiple next-gate tracks have indistinguishable authority",
    ):
        graph.confirm_transition(
            tracker,
            race_status=_live_race(
                sequence=11,
                boot_ms=5_250,
                gate_index=1,
                received_ns=credit_received_ns,
            ),
            camera_token_at_credit=camera_token_at_credit,
            promoted_track_id=next_ids[0],
        )

    assert tracker.tracks() == tracks_before
    assert graph.latest_snapshot.current_track_id == current_id


def test_live_promotion_rejects_race_status_from_a_different_reset_epoch() -> None:
    tracker, graph, current_id, next_id = _prime_live_graph(final_sequence=5)
    camera_token_at_credit = tracker.latest_update.token
    credit_received_ns = tracker.latest_update.publish_monotonic_ns + 1
    tracks_before = tracker.tracks()
    stale_epoch_race = replace(
        _live_race(
            sequence=11,
            boot_ms=5_250,
            gate_index=1,
            received_ns=credit_received_ns,
        ),
        reset_epoch=10,
    )

    with pytest.raises(GateGraphError, match="crossed its proved epoch"):
        graph.confirm_transition(
            tracker,
            race_status=stale_epoch_race,
            camera_token_at_credit=camera_token_at_credit,
            promoted_track_id=next_id,
        )

    assert tracker.tracks() == tracks_before
    assert graph.latest_snapshot.current_track_id == current_id


def test_live_promotion_preserves_neutral_credit_boundary_sample() -> None:
    tracker, graph, current_id, next_id = _prime_live_graph(final_sequence=5)
    camera_token_at_credit = tracker.latest_update.token
    credit_received_ns = tracker.latest_update.publish_monotonic_ns + 2_000_000
    late_frame = replace(
        _live_frame(6),
        final_unique_packet_monotonic_ns=credit_received_ns - 1,
        publish_monotonic_ns=credit_received_ns + 1,
    )
    tracker.update(late_frame)
    graph.observe(tracker)
    before = tracker.track(next_id)

    transition = graph.confirm_transition(
        tracker,
        race_status=_live_race(
            sequence=11,
            boot_ms=5_250,
            gate_index=1,
            received_ns=credit_received_ns,
        ),
        camera_token_at_credit=camera_token_at_credit,
        promoted_track_id=next_id,
    )

    assert transition.retired_track_id == current_id
    assert transition.promoted_track_id == next_id
    assert transition.promoted_history_length_at_credit == len(before.history) - 1
    assert transition.promoted_latest_token_before_credit == camera_token_at_credit
    assert transition.promoted_latest_token_at_promotion == late_frame.token
    assert tracker.track(next_id).history == before.history
