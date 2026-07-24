from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetVisualTracker,
    VisualDetectionFrame,
    VisualTrackRole,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    RollingVisualGateGraph,
)


_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "vq2_gate0_observe_20260718T022458_tracker_excerpt.json"
)
_SOURCE_SHA256 = "2fca8cc6d2b5ed0ced6dca8e7683254c60eb774571a7756a315126022032bfd6"


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
