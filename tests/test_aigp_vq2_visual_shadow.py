"""Focused integration tests for the commandless VQ2 visual-shadow stage."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from competition.adapter import AttitudeRateCommand
from competition.aigp_messages import RaceStatus
from competition.vq2_contracts import FrameIdentityV1, FrameTimingV1
from competition.vq2_visual_tracker import (
    FrameProvenanceBasis,
    VisualDetectionFrame,
    VisualTrackRole,
    visual_track_history_sha256,
)
from gate_detection.src.gate_detector import GateDetection
import planning.vq2_visual_recovery as visual_recovery
import scripts.aigp_vq2_run as vq2_module


_CAMERA_STREAM = "vq2-camera-udp-5600"
_HOST_CLOCK = "host-perf-counter"


class _Vision:
    def __init__(self) -> None:
        self.current_snapshot = None
        self.is_running = False
        self.frames_decoded = 0

    def snapshot(self, **_kwargs):
        return self.current_snapshot

    def stats(self):
        return SimpleNamespace(
            frames_decoded=self.frames_decoded,
            duplicate_datagrams=0,
        )


class _Adapter:
    enable_vision = False
    telemetry_mode = "imu"
    fetch_track_on_connect = False
    latest_telemetry = None
    race_status = None
    latest_received_race_status = None
    is_armed = False

    def __init__(self) -> None:
        self.commands = []
        self.collisions = []

    def drain_imu_samples(self):
        return []

    def drain_collisions(self):
        values = self.collisions
        self.collisions = []
        return values

    def drain_outbound_receipts(self):
        return []

    async def send_attitude_rate(self, command, **_kwargs):
        self.commands.append(command)


def _detection(x, y, width, height, *, confidence=0.90):
    return GateDetection(
        center_x=x + width // 2,
        center_y=y + height // 2,
        bbox=(x, y, width, height),
        corners=np.zeros((4, 2)),
        area=width * height,
        estimated_distance=999.0,
        confidence=confidence,
    )


def _snapshot(
    *,
    frame_id,
    publication_sequence,
    final_packet_ns,
    detections_image=None,
    generation=7,
):
    publish_ns = final_packet_ns + 4_000
    timing = FrameTimingV1(
        identity=FrameIdentityV1(_CAMERA_STREAM, generation, frame_id),
        camera_source_time_ns=frame_id * 1_000,
        host_clock_id=_HOST_CLOCK,
        publication_sequence=publication_sequence,
        first_unique_packet_monotonic_ns=final_packet_ns - 4_000,
        final_unique_packet_monotonic_ns=final_packet_ns,
        reassembly_complete_monotonic_ns=final_packet_ns + 1_000,
        decode_start_monotonic_ns=final_packet_ns + 2_000,
        decode_end_monotonic_ns=final_packet_ns + 3_000,
        publish_monotonic_ns=publish_ns,
    )
    image = (
        np.zeros((360, 640, 3), dtype=np.uint8)
        if detections_image is None
        else detections_image
    )
    return SimpleNamespace(
        frame_id=frame_id,
        sim_time_ns=timing.camera_source_time_ns,
        received_monotonic_s=final_packet_ns / 1_000_000_000.0,
        generation=generation,
        timing=timing,
        camera_frame=SimpleNamespace(image=image),
        age_s=lambda _now=None: 0.0,
    )


def _frame(frame_id, detections, *, final_packet_ns):
    return VisualDetectionFrame.from_detector_results(
        detections,
        generation=7,
        frame_id=frame_id,
        publication_sequence=frame_id,
        stream_id=_CAMERA_STREAM,
        final_unique_packet_monotonic_ns=final_packet_ns,
        publish_monotonic_ns=final_packet_ns + 4_000,
        time_basis_id=_HOST_CLOCK,
        camera_source_time_ns=frame_id * 1_000,
    )


def _set_race(
    adapter,
    *,
    gate_index,
    boot_ms,
    sequence,
    received_ns,
    race_finish_time_ns=-1,
):
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=boot_ms,
        race_start_boot_time_ms=900,
        race_finish_time_ns=race_finish_time_ns,
        active_gate_index=gate_index,
        last_gate_race_time=-1,
    )
    adapter.latest_received_race_status = SimpleNamespace(
        validate_integrity=lambda: None,
        ingress=SimpleNamespace(
            generation=3,
            sequence=sequence,
            received_monotonic_ns=received_ns,
            host_clock_id=_HOST_CLOCK,
        ),
        race_status=SimpleNamespace(
            sim_boot_time_ms=boot_ms,
            active_gate_index=gate_index,
            race_finish_time_ns=race_finish_time_ns,
        ),
    )


def _update_visual(runner, frame):
    update = runner.visual_tracker.update(frame)
    runner._visual_latest_tracker_update = update
    runner._visual_latest_graph_snapshot = runner.visual_gate_graph.observe(
        runner.visual_tracker
    )
    return update


def _prime_bound_gate_graph(
    runner,
    adapter,
    *,
    perf_clock_offset_ns=0,
):
    """Create stable current/next identities and bind only Gate 0."""

    runner._visual_tracking_enabled = True
    runner._visual_reset_epoch = 1
    for offset in range(3):
        update = _update_visual(
            runner,
            _frame(
                100 + offset,
                [
                    _detection(280, 140, 80, 80, confidence=0.95),
                    _detection(
                        490 - 5 * offset,
                        40 + 5 * offset,
                        50,
                        60,
                    ),
                ],
                final_packet_ns=(
                    perf_clock_offset_ns
                    + 100_000_000
                    + 40_000_000 * offset
                ),
            ),
        )
    _set_race(
        adapter,
        gate_index=0,
        boot_ms=1_000,
        sequence=10,
        received_ns=perf_clock_offset_ns + 200_000_000,
    )
    context = vq2_module.StartContext(0.0, -0.31, 320, 180, 6_400, 1_000)
    bound = runner._bind_initial_visual_gate(context)
    current_id = bound.current_track_id
    assert current_id is not None

    for offset in range(3, 5):
        update = _update_visual(
            runner,
            _frame(
                100 + offset,
                [
                    _detection(280, 140, 80, 80, confidence=0.95),
                    _detection(
                        490 - 5 * offset,
                        40 + 5 * offset,
                        50,
                        60,
                    ),
                ],
                final_packet_ns=(
                    perf_clock_offset_ns
                    + 100_000_000
                    + 40_000_000 * offset
                ),
            ),
        )
    next_ids = [
        track.track_id
        for track in update.visible_tracks
        if track.track_id != current_id
    ]
    assert len(next_ids) == 1
    return context, current_id, next_ids[0]


# Integer detector boxes reproducing the exact normalized centers, extents,
# and observation deltas in frames 168--172 of the accepted build-3385
# transition-anchor excerpt.  The absolute QPC epoch is supplied by each test.
_RECOVERY_ANCHOR_BOXES = (
    (168, 441, 32, 79, 75, 100_000_000),
    (169, 443, 31, 80, 76, 127_098_200),
    (170, 445, 28, 83, 78, 163_047_000),
    (171, 447, 26, 86, 80, 196_773_800),
    (172, 449, 23, 89, 82, 231_623_500),
)
_RECOVERY_ESTABLISHED_PREFIX_BOXES = (
    (164, 433, 38, 75, 71, -32_000_000),
    (165, 435, 36, 76, 72, 1_000_000),
    (166, 437, 35, 77, 73, 34_000_000),
    (167, 439, 34, 78, 74, 67_000_000),
)


def _prime_recovery_gate_graph(
    runner,
    adapter,
    *,
    perf_clock_offset_ns,
):
    """Bind Gate 0 with established identity before the exact anchor tail."""

    runner._visual_tracking_enabled = True
    runner._visual_reset_epoch = 1
    context = vq2_module.StartContext(
        0.0,
        -0.31,
        320,
        180,
        6_400,
        1_000,
    )
    current_id = None
    latest_update = None
    for index, (frame_id, x, y, width, height, observed_ns) in enumerate(
        _RECOVERY_ESTABLISHED_PREFIX_BOXES + _RECOVERY_ANCHOR_BOXES
    ):
        runner.vision.current_snapshot = _snapshot(
            frame_id=frame_id,
            publication_sequence=frame_id,
            final_packet_ns=perf_clock_offset_ns + observed_ns,
            generation=7,
        )
        latest_update = _update_visual(
            runner,
            _frame(
                frame_id,
                [
                    _detection(280, 140, 80, 80, confidence=0.95),
                    _detection(x, y, width, height, confidence=0.95),
                ],
                final_packet_ns=perf_clock_offset_ns + observed_ns,
            ),
        )
        if frame_id == 170:
            _set_race(
                adapter,
                gate_index=0,
                boot_ms=1_000,
                sequence=10,
                received_ns=(
                    perf_clock_offset_ns + observed_ns + 5_000_000
                ),
            )
            bound = runner._bind_initial_visual_gate(context)
            current_id = bound.current_track_id
            assert current_id is not None

    assert current_id is not None
    assert latest_update is not None
    next_ids = [
        track.track_id
        for track in latest_update.visible_tracks
        if track.track_id != current_id
    ]
    assert len(next_ids) == 1
    promoted_id = next_ids[0]
    promoted = runner.visual_tracker.track(promoted_id)
    assert tuple(
        sample.token.frame_id for sample in promoted.history
    ) == (164, 165, 166, 167, 168, 169, 170, 171, 172)
    return context, current_id, promoted_id


def _prime_reacquisition_bridge_gate_graph(
    runner,
    adapter,
    *,
    perf_clock_offset_ns,
    skipped_publication=None,
    reacquisition_publication=116,
    final_publication=120,
    race_credit_before_final=False,
):
    """Create one production-tracker identity across a bounded gap."""

    runner._visual_tracking_enabled = True
    runner._visual_reset_epoch = 1
    context = vq2_module.StartContext(
        0.0,
        -0.31,
        320,
        180,
        6_400,
        1_000,
    )
    current_id = None
    target_id = None
    last_observation_ns = 0
    for publication in range(100, final_publication + 1):
        last_observation_ns = (
            perf_clock_offset_ns
            + (publication - 100) * 33_000_000
        )
        if publication == skipped_publication:
            continue
        detections = [
            _detection(280, 140, 80, 80, confidence=0.95),
        ]
        if (
            publication <= 104
            or publication >= reacquisition_publication
        ):
            detections.append(
                _detection(440, 40, 80, 80, confidence=0.95)
            )
        runner.vision.current_snapshot = _snapshot(
            frame_id=publication,
            publication_sequence=publication,
            final_packet_ns=last_observation_ns,
            generation=7,
        )
        update = _update_visual(
            runner,
            _frame(
                publication,
                detections,
                final_packet_ns=last_observation_ns,
            ),
        )
        if publication == 102:
            _set_race(
                adapter,
                gate_index=0,
                boot_ms=1_000,
                sequence=10,
                received_ns=last_observation_ns + 5_000_000,
            )
            bound = runner._bind_initial_visual_gate(context)
            current_id = bound.current_track_id
            assert current_id is not None
        if publication == 104:
            target_ids = [
                track.track_id
                for track in update.visible_tracks
                if track.track_id != current_id
            ]
            assert len(target_ids) == 1
            target_id = target_ids[0]

    assert current_id is not None
    assert target_id is not None
    race_received_ns = last_observation_ns + 18_004_000
    if race_credit_before_final:
        race_received_ns -= 33_000_000
    _set_race(
        adapter,
        gate_index=1,
        boot_ms=1_300,
        sequence=11,
        received_ns=race_received_ns,
    )
    transition = runner._confirm_visual_transition(
        from_gate_index=0,
        to_gate_index=1,
        race_status=runner._visual_race_status_ref(),
    )
    assert transition.retired_track_id == current_id
    assert transition.promoted_track_id == target_id
    return context, transition, runner.visual_tracker.track(target_id)


def test_receiver_watermark_requires_the_exact_latest_camera_token():
    vision = _Vision()
    runner = vq2_module.VQ2Runner(_Adapter(), vision)
    _update_visual(
        runner,
        _frame(
            172,
            [_detection(451, 19, 92, 85, confidence=0.95)],
            final_packet_ns=10_231_623_500,
        ),
    )
    vision.current_snapshot = _snapshot(
        frame_id=172,
        publication_sequence=172,
        final_packet_ns=10_231_623_500,
    )
    expected = vq2_module.VisualCameraFrameToken.from_vision_snapshot(
        vision.current_snapshot
    )

    assert runner._assert_visual_receiver_token_current(expected) == expected

    vision.current_snapshot = _snapshot(
        frame_id=173,
        publication_sequence=173,
        final_packet_ns=10_266_066_600,
    )
    with pytest.raises(
        vq2_module.SafetyAbort,
        match="receiver advanced beyond",
    ):
        runner._assert_visual_receiver_token_current(expected)


def test_race_credit_camera_watermark_includes_detectionless_frame():
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    runner._visual_tracking_enabled = True
    first = _frame(
        171,
        [_detection(451, 19, 92, 85, confidence=0.95)],
        final_packet_ns=10_200_000_000,
    )
    detectionless = _frame(
        172,
        [],
        final_packet_ns=10_233_000_000,
    )
    _update_visual(runner, first)
    _update_visual(runner, detectionless)
    _set_race(
        adapter,
        gate_index=1,
        boot_ms=1_300,
        sequence=11,
        received_ns=detectionless.publish_monotonic_ns + 1,
    )

    token = runner._visual_camera_token_at_race_credit(
        runner._visual_race_status_ref()
    )

    assert token == detectionless.token
    assert all(
        sample.token != detectionless.token
        for track in runner.visual_tracker.tracks()
        for sample in track.history
    )


def test_transition_preserves_multiple_samples_after_captured_credit():
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    _context, current_id, promoted_id = _prime_bound_gate_graph(
        runner,
        adapter,
    )
    _set_race(
        adapter,
        gate_index=1,
        boot_ms=1_300,
        sequence=11,
        received_ns=300_000_000,
    )
    captured_credit = runner._visual_race_status_ref()

    for frame_id, final_packet_ns in (
        (105, 320_000_000),
        (106, 353_000_000),
        (107, 386_000_000),
    ):
        _update_visual(
            runner,
            _frame(
                frame_id,
                [
                    _detection(280, 140, 80, 80, confidence=0.95),
                    _detection(465, 65, 50, 60, confidence=0.90),
                ],
                final_packet_ns=final_packet_ns,
            ),
        )
    _set_race(
        adapter,
        gate_index=1,
        boot_ms=1_350,
        sequence=12,
        received_ns=400_000_000,
    )

    transition = runner._confirm_visual_transition(
        from_gate_index=0,
        to_gate_index=1,
        race_status=captured_credit,
    )

    assert transition.retired_track_id == current_id
    assert transition.promoted_track_id == promoted_id
    assert transition.race_status == captured_credit
    assert transition.race_status.race_status_sequence == 11
    assert transition.camera_token_at_credit.frame_id == 104
    assert transition.promoted_latest_token_at_promotion.frame_id == 107
    assert (
        transition.history_length_before_promotion
        - transition.promoted_history_length_at_credit
    ) == 3
    assert runner._visual_race_status_ref().race_status_sequence == 12


def test_transition_refuses_replacement_when_reviewed_identity_is_unavailable():
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    _context, current_id, replacement_id = _prime_bound_gate_graph(
        runner,
        adapter,
    )
    _set_race(
        adapter,
        gate_index=1,
        boot_ms=1_300,
        sequence=11,
        received_ns=300_000_000,
    )

    with pytest.raises(
        vq2_module.SafetyAbort,
        match="requested promotion track is not promotable",
    ):
        runner._confirm_visual_transition(
            from_gate_index=0,
            to_gate_index=1,
            race_status=runner._visual_race_status_ref(),
            promoted_track_id="reviewed-track-that-is-now-unavailable",
        )

    snapshot = runner.visual_gate_graph.latest_snapshot
    assert snapshot is not None
    assert snapshot.current_gate_index == 0
    assert snapshot.current_track_id == current_id
    assert runner.visual_tracker.track(current_id).role is (
        VisualTrackRole.CURRENT
    )
    assert runner.visual_tracker.track(replacement_id).role is (
        VisualTrackRole.NEXT
    )


def test_sample_consumes_every_detection_with_exact_receiver_provenance(
    monkeypatch,
):
    adapter = _Adapter()
    vision = _Vision()
    runner = vq2_module.VQ2Runner(adapter, vision)
    runner._visual_tracking_enabled = True
    detections = [
        _detection(90, 120, 42, 64, confidence=0.82),
        _detection(390, 100, 96, 110, confidence=0.93),
    ]
    vision.current_snapshot = _snapshot(
        frame_id=41,
        publication_sequence=73,
        final_packet_ns=12_000_000,
    )
    monkeypatch.setattr(runner.detector, "detect", lambda _image: detections)

    runner._sample()

    update = runner._visual_latest_tracker_update
    assert runner._detection_error is None
    assert update is not None
    assert update.provenance_basis is FrameProvenanceBasis.RECEIVER_TIMING_V1
    assert update.token.live_identity_tuple == (
        _CAMERA_STREAM,
        7,
        41,
        73,
    )
    assert update.observation_monotonic_ns == 12_000_000
    assert update.publish_monotonic_ns == 12_004_000
    assert len(update.visible_tracks) == 2
    assert {
        track.history[-1].source_index for track in update.visible_tracks
    } == {0, 1}
    assert len(runner._latest_raw_detections) == 2


def test_visual_trace_retains_exact_association_bridge_provenance(
    monkeypatch,
):
    runner = vq2_module.VQ2Runner(_Adapter(), _Vision())
    runner._visual_tracking_enabled = True
    runner._visual_diagnostic_logging = True
    events = []
    detections = [
        _detection(390, 100, 96, 110, confidence=0.93),
    ]
    monkeypatch.setattr(runner.detector, "detect", lambda _image: detections)
    monkeypatch.setattr(
        runner.recorder,
        "emit",
        lambda event, **fields: events.append((event, fields)),
    )

    for frame_id, publication_sequence, packet_ns in (
        (41, 73, 12_000_000),
        (42, 74, 45_000_000),
    ):
        runner.vision.current_snapshot = _snapshot(
            frame_id=frame_id,
            publication_sequence=publication_sequence,
            final_packet_ns=packet_ns,
        )
        runner._sample()

    visual_events = [
        fields
        for event, fields in events
        if event == "visual_gate_graph_frame"
    ]
    assert len(visual_events) == 2
    associations = visual_events[-1]["associations"]
    assert len(associations) == 1
    association = associations[0]
    evidence = runner._visual_latest_tracker_update.associations[0]
    assert association["previous_frame_token"] == [
        _CAMERA_STREAM,
        7,
        41,
        73,
    ]
    assert association["current_frame_token"] == [
        _CAMERA_STREAM,
        7,
        42,
        74,
    ]
    assert association["missed_frame_count_before_association"] == 0
    assert association["observation_gap_ns"] == 33_000_000
    assert association["publication_gap_ns"] == 33_000_000
    assert association["ambiguous"] is False
    assert association["track_ambiguous_before_association"] is False
    assert association["detection_source_index"] == (
        evidence.detection_source_index
    )
    assert association["cost"] == evidence.cost
    assert association["confidence"] == evidence.confidence
    assert association["bbox_iou"] == evidence.bbox_iou
    assert association["predicted_center_residual_norm"] == (
        evidence.predicted_center_residual_norm
    )
    assert association["log_width_change"] == evidence.log_width_change
    assert association["log_height_change"] == evidence.log_height_change
    assert association["log_area_residual"] == evidence.log_area_residual
    assert association["clipping_continuity"] == (
        evidence.clipping_continuity
    )
    assert association["temporal_consistency"] == (
        evidence.temporal_consistency
    )
    assert association["appearance_distance"] is None


def test_tracker_graph_recovery_emits_nested_reacquisition_bridge(
    monkeypatch,
):
    perf_clock_offset_ns = 10_000_000_000
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    events = []
    monkeypatch.setattr(
        runner.recorder,
        "emit",
        lambda event, **fields: events.append((event, fields)),
    )

    _context, transition, promoted = (
        _prime_reacquisition_bridge_gate_graph(
            runner,
            adapter,
            perf_clock_offset_ns=perf_clock_offset_ns,
        )
    )
    assert tuple(
        sample.token.publication_sequence
        for sample in promoted.history
    ) == (100, 101, 102, 103, 104, 116, 117, 118, 119, 120)
    bridge_sample = promoted.history[5]
    assert bridge_sample.accepted_association is not None
    assert (
        bridge_sample.accepted_association
        .missed_frame_count_before_association
    ) == 11
    assert tuple(
        token.publication_sequence
        for token in transition.pretransition_frame_tokens
    ) == (116, 117, 118, 119, 120)

    authority = visual_recovery.require_promotion_history_authority(
        promoted,
        transition,
        tracker_time_basis_id=_HOST_CLOCK,
    )
    race_received_ns = transition.race_status.received_monotonic_ns
    assert race_received_ns is not None
    admission = visual_recovery.require_transition_recovery_admission(
        promoted,
        transition,
        tracker_time_basis_id=_HOST_CLOCK,
        measured_pitch_rad=-0.04,
        now_monotonic_ns=race_received_ns + 1_000_000,
        promotion_history_authority=authority,
    )
    bridge = admission.reacquisition_bridge
    assert bridge is not None
    assert bridge.predecessor_token.publication_sequence == 104
    assert bridge.reacquisition_token.publication_sequence == 116
    assert bridge.missed_frame_count == 11
    assert bridge.publication_delta == 12
    assert bridge.unobserved_publication_count == 0
    assert bridge.direct_bbox_iou == pytest.approx(1.0)
    assert bridge.average_horizontal_rate_norm_s == pytest.approx(0.0)
    assert bridge.average_vertical_rate_norm_s == pytest.approx(0.0)
    assert bridge.average_log_scale_rate_s == pytest.approx(0.0)
    assert admission.promotion_identity_basis == (
        "bounded_reacquisition_bridge_v1"
    )
    assert admission.cross_gap_identity_claimed is True
    assert admission.visibility_epoch_frame_count == 5
    serialized = asdict(admission)
    assert serialized["promotion_identity_sha256"] == (
        transition.promoted_history_sha256
    )
    assert serialized["reacquisition_bridge"][
        "missed_frame_count"
    ] == 11
    assert serialized["reacquisition_bridge"][
        "unobserved_publication_count"
    ] == 0
    transition_events = [
        fields
        for event, fields in events
        if event == "visual_gate_transition_promoted"
    ]
    assert len(transition_events) == 1
    assert transition_events[0]["promoted_history_sha256"] == (
        transition.promoted_history_sha256
    )


def test_tracker_graph_recovery_uses_complete_epoch_without_cross_gap_claim():
    perf_clock_offset_ns = 10_000_000_000
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())

    _context, transition, promoted = (
        _prime_reacquisition_bridge_gate_graph(
            runner,
            adapter,
            perf_clock_offset_ns=perf_clock_offset_ns,
            reacquisition_publication=117,
            final_publication=121,
        )
    )

    assert tuple(
        sample.token.publication_sequence
        for sample in promoted.history
    ) == (100, 101, 102, 103, 104, 117, 118, 119, 120, 121)
    epoch_root = promoted.history[5]
    assert epoch_root.accepted_association is not None
    assert (
        epoch_root.accepted_association
        .missed_frame_count_before_association
    ) == 12
    assert tuple(
        token.publication_sequence
        for token in transition.pretransition_frame_tokens
    ) == (117, 118, 119, 120, 121)

    authority = visual_recovery.require_promotion_history_authority(
        promoted,
        transition,
        tracker_time_basis_id=_HOST_CLOCK,
    )
    race_received_ns = transition.race_status.received_monotonic_ns
    assert race_received_ns is not None
    admission = visual_recovery.require_transition_recovery_admission(
        promoted,
        transition,
        tracker_time_basis_id=_HOST_CLOCK,
        measured_pitch_rad=-0.04,
        now_monotonic_ns=race_received_ns + 1_000_000,
        promotion_history_authority=authority,
    )

    assert admission.promotion_identity_basis == (
        "complete_current_visibility_epoch_v1"
    )
    assert admission.cross_gap_identity_claimed is False
    assert admission.reacquisition_bridge is None
    assert admission.visibility_epoch_frame_count == 5
    assert tuple(
        token.publication_sequence
        for token in admission.visibility_epoch_tokens
    ) == (117, 118, 119, 120, 121)
    assert tuple(
        later - earlier
        for earlier, later in zip(
            admission.visibility_epoch_tracker_frame_sequences,
            admission.visibility_epoch_tracker_frame_sequences[1:],
        )
    ) == (1, 1, 1, 1)


def test_tracker_graph_recovery_admits_six_frame_epoch_with_one_receiver_skip():
    perf_clock_offset_ns = 10_000_000_000
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())

    _context, transition, promoted = (
        _prime_reacquisition_bridge_gate_graph(
            runner,
            adapter,
            perf_clock_offset_ns=perf_clock_offset_ns,
            skipped_publication=110,
            final_publication=121,
            race_credit_before_final=True,
        )
    )

    assert tuple(
        sample.token.publication_sequence
        for sample in promoted.history
    ) == (100, 101, 102, 103, 104, 116, 117, 118, 119, 120, 121)
    assert tuple(
        token.publication_sequence
        for token in transition.pretransition_frame_tokens
    ) == (116, 117, 118, 119, 120)
    assert transition.promoted_history_length_at_credit == 10
    assert (
        transition.promoted_latest_token_before_credit.publication_sequence
        == 120
    )
    assert transition.history_length_before_promotion == 11
    assert transition.history_length_after_promotion == 11
    assert (
        transition.promoted_latest_token_at_promotion.publication_sequence
        == 121
    )
    bridge_sample = promoted.history[5]
    assert bridge_sample.accepted_association is not None
    assert (
        bridge_sample.accepted_association
        .missed_frame_count_before_association
    ) == 10

    authority = visual_recovery.require_promotion_history_authority(
        promoted,
        transition,
        tracker_time_basis_id=_HOST_CLOCK,
    )
    race_received_ns = transition.race_status.received_monotonic_ns
    assert race_received_ns is not None
    promotion_publication_ns = promoted.history[-1].publication_monotonic_ns
    assert promotion_publication_ns is not None
    admission = visual_recovery.require_transition_recovery_admission(
        promoted,
        transition,
        tracker_time_basis_id=_HOST_CLOCK,
        measured_pitch_rad=-0.04,
        now_monotonic_ns=promotion_publication_ns + 1_000_000,
        promotion_history_authority=authority,
    )
    bridge = admission.reacquisition_bridge
    assert bridge is not None
    assert bridge.missed_frame_count == 10
    assert bridge.tracker_frame_delta == 11
    assert bridge.publication_delta == 12
    assert bridge.unobserved_publication_count == 1
    assert asdict(admission)["reacquisition_bridge"][
        "unobserved_publication_count"
    ] == 1
    assert admission.credit_prefix_token.publication_sequence == 120
    assert admission.promotion_anchor_token.publication_sequence == 121
    assert admission.promotion_anchor_publication_delta_from_credit_s > 0.0

    latest_observation_ns = (
        perf_clock_offset_ns + (122 - 100) * 33_000_000
    )
    runner.vision.current_snapshot = _snapshot(
        frame_id=122,
        publication_sequence=122,
        final_packet_ns=latest_observation_ns,
        generation=7,
    )
    _update_visual(
        runner,
        _frame(
            122,
            [_detection(440, 40, 80, 80, confidence=0.95)],
            final_packet_ns=latest_observation_ns,
        ),
    )
    continued = runner.visual_tracker.track(transition.promoted_track_id)
    latest = continued.history[-1]
    assert latest.publication_monotonic_ns is not None
    continuation = visual_recovery.require_recovery_continuation(
        continued,
        transition,
        previous_token=transition.promoted_latest_token_at_promotion,
        tracker_time_basis_id=_HOST_CLOCK,
        measured_pitch_rad=-0.04,
        recovery_started_monotonic_ns=(
            promotion_publication_ns + 1_000_000
        ),
        now_monotonic_ns=latest.publication_monotonic_ns + 1_000_000,
        promotion_history_authority=authority,
    )
    assert continuation.previous_token.publication_sequence == 121
    assert continuation.frame_token.publication_sequence == 122


def test_initial_gate_binding_rejects_two_plausible_visual_identities():
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    runner._visual_tracking_enabled = True
    for offset in range(3):
        runner._visual_latest_tracker_update = runner.visual_tracker.update(
            _frame(
                20 + offset,
                [
                    _detection(270, 140, 60, 80, confidence=0.95),
                    _detection(370, 140, 60, 80, confidence=0.95),
                ],
                final_packet_ns=10_000_000 + 40_000_000 * offset,
            )
        )

    with pytest.raises(
        vq2_module.SafetyAbort,
        match=r"ambiguous \(candidate_count=2\)",
    ):
        runner._bind_initial_visual_gate(
            vq2_module.StartContext(
                0.0,
                -0.31,
                350,
                180,
                4_800,
                1_000,
            )
        )


@pytest.mark.parametrize(
    "stage",
    [
        vq2_module.VISUAL_SHADOW_STAGE,
        vq2_module.VISUAL_ALIGN_STAGE,
    ],
)
def test_visual_stage_refuses_direct_run_without_fast_cycle_binding(
    monkeypatch,
    stage,
):
    contacted_transport = False

    def load_transport():
        nonlocal contacted_transport
        contacted_transport = True
        raise AssertionError("unbound visual stage must not load live transport")

    monkeypatch.setattr(
        vq2_module,
        "_load_live_transport_dependencies",
        load_transport,
    )

    with pytest.raises(
        PermissionError,
        match="manifest-bound fast-cycle wrapper",
    ):
        asyncio.run(
            vq2_module.run_live(
                stage,
                vq2_module.DEFAULT_MAVLINK_URL,
                None,
            )
        )
    assert contacted_transport is False


def test_shadow_promotes_precredit_track_without_reset_and_sends_only_zero(
    monkeypatch,
):
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    context, initial_current_id, expected_next_id = _prime_bound_gate_graph(
        runner,
        adapter,
    )
    clock = [1.0]
    commands = []
    watchdog_calls = []

    async def run_gate0(_context, *, capture_transition=False):
        assert _context is context
        assert capture_transition is False
        _set_race(
            adapter,
            gate_index=1,
            boot_ms=1_300,
            sequence=11,
            received_ns=300_000_000,
        )
        transition = runner._confirm_visual_transition(
            from_gate_index=0,
            to_gate_index=1,
            race_status=runner._visual_race_status_ref(),
        )
        assert transition.promoted_track_id == expected_next_id
        runner._gate0_transition_proof = vq2_module.GateTransitionProof(
            pre_gate_race_boot_ms=1_000,
            post_gate_race_boot_ms=1_300,
            flight_started_monotonic_s=0.8,
            crossing_started_monotonic_s=0.9,
            pass_confirmed_monotonic_s=1.0,
            next_control_deadline_s=1.0,
            vision_generation=7,
            vision_frame_id=104,
            vision_sim_time_ns=104_000,
            vision_received_monotonic_s=0.26,
            pass_rpy_rad=(0.0, -0.07, 0.0),
        )
        return {"gate0_passed": True}

    sampled = [False]

    def sample_post_credit():
        assert not sampled[0]
        sampled[0] = True
        _update_visual(
            runner,
            _frame(
                105,
                [_detection(465, 65, 50, 60)],
                final_packet_ns=320_000_000,
            ),
        )

    async def capture_command(command, **_kwargs):
        commands.append(command)

    async def next_slot():
        return clock[0]

    async def advance(seconds):
        clock[0] += max(0.0, float(seconds))

    monkeypatch.setattr(runner, "_run_gate0", run_gate0)
    monkeypatch.setattr(runner, "_sample", sample_post_credit)
    monkeypatch.setattr(
        runner,
        "_watchdog",
        lambda **kwargs: watchdog_calls.append(kwargs),
    )
    monkeypatch.setattr(runner, "_send_flight_command", capture_command)
    monkeypatch.setattr(runner, "_record_tick", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "_wait_for_next_flight_command_slot",
        next_slot,
    )
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(monotonic=lambda: clock[0]),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(sleep=advance),
        )
        summary = asyncio.run(runner._run_visual_shadow(context))

    transition = runner._visual_transition
    assert transition is not None
    assert transition.retired_track_id == initial_current_id
    assert transition.promoted_track_id == expected_next_id
    assert transition.history_length_before_promotion == (
        transition.history_length_after_promotion
    )
    assert len(transition.promoted_history_sha256) == 64
    assert len(transition.pretransition_frame_tokens) >= 3
    promoted = runner.visual_tracker.track(expected_next_id)
    assert transition.promoted_history_sha256 == (
        visual_track_history_sha256(
            promoted.history[
                : transition.history_length_after_promotion
            ]
        )
    )
    assert promoted.first_token.frame_id == 100
    assert promoted.latest_token.frame_id == 105
    assert promoted.role is VisualTrackRole.CURRENT
    assert promoted.authoritative_gate_index == 1

    assert commands
    assert all(
        command == AttitudeRateCommand(0.0, 0.0, 0.0, 0.0)
        for command in commands
    )
    assert watchdog_calls == [
        {
            "require_target": False,
            "allow_benign_pad_contact": False,
        }
    ]
    assert summary["shadow_command_authority"] == (
        "legacy_proved_gate0_only"
    )
    assert summary["visual_navigation_command_count"] == 0
    assert summary["post_credit_zero_command_count"] == len(commands)
    assert summary["authoritative_transition"] == [0, 1]
    assert summary["pretransition_frame_count"] >= 3
    assert summary["history_length_before_promotion"] == (
        summary["history_length_after_promotion"]
    )
    assert summary["promoted_history_sha256"] == (
        transition.promoted_history_sha256
    )
    assert all(
        token[:2] == [_CAMERA_STREAM, 7]
        for token in summary["pretransition_frame_tokens"]
    )
    assert summary["post_credit_frame_tokens"] == [
        [_CAMERA_STREAM, 7, 105, 105]
    ]
    assert summary["horizontal_abs_error_trend"]["trend"] == (
        "negative_uninterrupted"
    )
    assert summary["vertical_abs_error_trend"]["trend"] == (
        "negative_uninterrupted"
    )
    assert summary["ambiguous"] is False


@pytest.mark.parametrize(
    (
        "cleanup_gate",
        "cleanup_confirmed",
        "reason_fragment",
    ),
    [
        (1, False, "cleanup unconfirmed"),
        (
            0,
            True,
            "cleanup boundary lacks proved 0->1 authority",
        ),
    ],
)
def test_powered_shadow_requires_authoritative_boundary_and_confirmed_cleanup(
    monkeypatch,
    cleanup_gate,
    cleanup_confirmed,
    reason_fragment,
):
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    context = vq2_module.StartContext(0.0, -0.31, 320, 180, 6_400, 1_000)

    async def no_result(*_args, **_kwargs):
        return None

    async def wait_for_go():
        _set_race(
            adapter,
            gate_index=0,
            boot_ms=1_000,
            sequence=1,
            received_ns=10_000_000,
        )
        return context

    async def run_shadow(_context):
        _set_race(
            adapter,
            gate_index=cleanup_gate,
            boot_ms=1_300,
            sequence=2,
            received_ns=20_000_000,
        )
        runner._visual_transition = SimpleNamespace(
            from_gate_index=0,
            to_gate_index=1,
        )
        return {
            "shadow_command_authority": "legacy_proved_gate0_only",
            "visual_navigation_command_count": 0,
        }

    async def cleanup():
        return cleanup_confirmed

    monkeypatch.setattr(runner, "establish_reset_epoch", no_result)
    monkeypatch.setattr(runner, "normalize_disarmed", no_result)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(
        runner,
        "_bind_initial_visual_gate",
        lambda _context: None,
    )
    monkeypatch.setattr(runner, "arm_confirmed", no_result)
    monkeypatch.setattr(runner, "_run_visual_shadow", run_shadow)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(
        runner.run_powered_stage(
            vq2_module.VISUAL_SHADOW_STAGE,
            write_diagnostic_pngs=False,
        )
    )

    assert result.success is False
    assert result.cleanup_confirmed is cleanup_confirmed
    assert reason_fragment in result.reason
    assert result.details["authoritative_cleanup_entry"] == {
        "gate_index": cleanup_gate,
        "race_finished": False,
        "transitions": [],
        "transition": [0, 1],
    }


class _Orientation:
    def __init__(self, roll=0.0, pitch=-0.05, yaw=0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def to_euler(self):
        return self.roll, self.pitch, self.yaw


def _alignment_estimate():
    # Restricted alignment authority now requires an achieved neutral/braking
    # pitch at entry, not merely a positive requested pitch.
    orientation = _Orientation(pitch=0.0)
    return SimpleNamespace(
        orientation=orientation,
        body_rates=(0.0, 0.0, 0.0),
        yaw=orientation.yaw,
    )


@pytest.mark.parametrize(
    (
        "requested",
        "yaw",
        "reference",
        "yaw_rate",
        "horizontal_error",
        "expected",
    ),
    [
        (
            0.10,
            0.159,
            0.0,
            0.0,
            0.5,
            (
                vq2_module.VISUAL_ALIGN_YAW_SOFT_STOP_RAD
                - 0.159
            )
            / vq2_module.VISUAL_ALIGN_YAW_HOLD_HORIZON_S,
        ),
        (-0.10, 0.161, 0.0, 0.0, 0.5, -0.10),
        (
            -0.10,
            -math.pi + 0.01,
            math.pi - 0.01,
            0.0,
            -0.5,
            -0.10,
        ),
    ],
)
def test_visual_alignment_yaw_envelope_caps_outward_and_retains_inward(
    requested,
    yaw,
    reference,
    yaw_rate,
    horizontal_error,
    expected,
):
    command, excursion = vq2_module.visual_alignment_yaw_rate(
        requested_rate_rad_s=requested,
        measured_yaw_rad=yaw,
        reference_yaw_rad=reference,
        measured_yaw_rate_rad_s=yaw_rate,
        horizontal_error_norm=horizontal_error,
        horizontal_corridor_norm=0.16,
    )

    assert command == pytest.approx(expected)
    assert abs(excursion) <= vq2_module.VISUAL_ALIGN_MAX_YAW_EXCURSION_RAD


def test_visual_alignment_yaw_envelope_aborts_exhausted_outward_authority():
    with pytest.raises(
        vq2_module.SafetyAbort,
        match="authority exhausted",
    ):
        vq2_module.visual_alignment_yaw_rate(
            requested_rate_rad_s=0.10,
            measured_yaw_rad=0.161,
            reference_yaw_rad=0.0,
            measured_yaw_rate_rad_s=0.0,
            horizontal_error_norm=0.5,
            horizontal_corridor_norm=0.16,
        )


def test_visual_alignment_yaw_envelope_aborts_outward_measured_momentum():
    with pytest.raises(
        vq2_module.SafetyAbort,
        match="outward yaw momentum",
    ):
        vq2_module.visual_alignment_yaw_rate(
            requested_rate_rad_s=-0.10,
            measured_yaw_rad=0.159,
            reference_yaw_rad=0.0,
            measured_yaw_rate_rad_s=0.20,
            horizontal_error_norm=0.5,
            horizontal_corridor_norm=0.16,
        )


def test_visual_alignment_current_authority_rejects_a_fresh_track_miss():
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    _context, _initial_id, promoted_id = _prime_bound_gate_graph(
        runner,
        adapter,
    )
    _set_race(
        adapter,
        gate_index=1,
        boot_ms=1_300,
        sequence=11,
        received_ns=300_000_000,
    )
    runner._confirm_visual_transition(
        from_gate_index=0,
        to_gate_index=1,
        race_status=runner._visual_race_status_ref(),
    )
    _update_visual(
        runner,
        _frame(
            105,
            [_detection(465, 15, 50, 75)],
            final_packet_ns=320_000_000,
        ),
    )
    track, target = runner._require_visual_current_target(
        expected_gate_index=1,
        expected_track_id=promoted_id,
        now_s=0.32,
    )
    assert track.track_id == target.track_id == promoted_id

    _update_visual(
        runner,
        _frame(106, [], final_packet_ns=340_000_000),
    )
    with pytest.raises(
        vq2_module.SafetyAbort,
        match="withheld exact current-target authority",
    ):
        runner._require_visual_current_target(
            expected_gate_index=1,
            expected_track_id=promoted_id,
            now_s=0.34,
        )


def test_visual_alignment_current_authority_defaults_to_receiver_clock(
    monkeypatch,
):
    perf_offset_s = 10.0
    perf_offset_ns = round(perf_offset_s * 1_000_000_000)
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    _context, _initial_id, promoted_id = _prime_bound_gate_graph(
        runner,
        adapter,
        perf_clock_offset_ns=perf_offset_ns,
    )
    _set_race(
        adapter,
        gate_index=1,
        boot_ms=1_300,
        sequence=11,
        received_ns=perf_offset_ns + 300_000_000,
    )
    runner._confirm_visual_transition(
        from_gate_index=0,
        to_gate_index=1,
        race_status=runner._visual_race_status_ref(),
    )
    _update_visual(
        runner,
        _frame(
            105,
            [_detection(465, 15, 50, 75)],
            final_packet_ns=perf_offset_ns + 320_000_000,
        ),
    )

    monkeypatch.setattr(
        vq2_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: 0.320,
            perf_counter_ns=lambda: perf_offset_ns + 320_000_000,
        ),
    )
    track, target = runner._require_visual_current_target(
        expected_gate_index=1,
        expected_track_id=promoted_id,
    )
    assert track.track_id == target.track_id == promoted_id
    with pytest.raises(
        vq2_module.SafetyAbort,
        match="promoted visual current target is stale",
    ):
        runner._require_visual_current_target(
            expected_gate_index=1,
            expected_track_id=promoted_id,
            now_s=perf_offset_s + 0.421,
        )


def test_visual_alignment_rejects_graph_or_race_ambiguity():
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    _context, _initial_id, promoted_id = _prime_bound_gate_graph(
        runner,
        adapter,
    )
    _set_race(
        adapter,
        gate_index=1,
        boot_ms=1_300,
        sequence=11,
        received_ns=300_000_000,
    )
    runner._confirm_visual_transition(
        from_gate_index=0,
        to_gate_index=1,
        race_status=runner._visual_race_status_ref(),
    )
    _update_visual(
        runner,
        _frame(
            105,
            [_detection(465, 15, 50, 75)],
            final_packet_ns=320_000_000,
        ),
    )
    graph = runner.visual_gate_graph.latest_snapshot
    assert graph is not None
    runner.visual_gate_graph._latest_snapshot = replace(
        graph,
        next_selection_ambiguous=True,
    )
    with pytest.raises(
        vq2_module.SafetyAbort,
        match="withheld exact current-target authority",
    ):
        runner._require_visual_current_target(
            expected_gate_index=1,
            expected_track_id=promoted_id,
            now_s=0.32,
        )

    _set_race(
        adapter,
        gate_index=2,
        boot_ms=1_400,
        sequence=12,
        received_ns=340_000_000,
    )
    with pytest.raises(
        vq2_module.SafetyAbort,
        match="no-passage Gate-1 boundary",
    ):
        runner._assert_visual_alignment_race_boundary()


@pytest.mark.parametrize(
    ("bbox_norm", "clipping"),
    [
        ((0.20, 0.20, 0.45, 0.40), vq2_module.FrameEdge.NONE),
        (
            (0.20, 0.20, 0.40, 0.20 + 1.0 / 3.0),
            vq2_module.FrameEdge.NONE,
        ),
        (
            (0.20, 0.20, 0.40, 0.40),
            vq2_module.FrameEdge.LEFT | vq2_module.FrameEdge.RIGHT,
        ),
    ],
)
def test_visual_alignment_no_passage_bounds_fail_closed(
    bbox_norm,
    clipping,
):
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    _context, _initial_id, promoted_id = _prime_bound_gate_graph(
        runner,
        adapter,
    )
    _set_race(
        adapter,
        gate_index=1,
        boot_ms=1_300,
        sequence=11,
        received_ns=300_000_000,
    )
    runner._confirm_visual_transition(
        from_gate_index=0,
        to_gate_index=1,
        race_status=runner._visual_race_status_ref(),
    )
    promoted = runner.visual_tracker.track(promoted_id)
    unsafe = replace(
        promoted,
        bbox_norm=bbox_norm,
        clipping=clipping,
    )

    with pytest.raises(
        vq2_module.SafetyAbort,
        match="no-passage geometry",
    ):
        runner._assert_visual_alignment_no_passage(
            unsafe,
            phase="test",
        )


def test_restricted_visual_alignment_preserves_promoted_identity_and_improves(
    monkeypatch,
):
    # Windows build-3385 receiver timestamps use QPC while control pacing uses
    # the coarser GetTickCount64 monotonic clock.  Keep their epochs distinct.
    perf_offset_s = 10.0
    perf_offset_ns = round(perf_offset_s * 1_000_000_000)
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    context, initial_current_id, promoted_id = _prime_bound_gate_graph(
        runner,
        adapter,
        perf_clock_offset_ns=perf_offset_ns,
    )
    runner.estimate = _alignment_estimate()
    clock = [0.320]
    commands = []

    async def run_gate0(
        _context,
        *,
        capture_transition=False,
        visual_next_gate_blend=False,
    ):
        assert _context is context
        assert capture_transition is False
        assert visual_next_gate_blend is True
        _set_race(
            adapter,
            gate_index=1,
            boot_ms=1_300,
            sequence=11,
            received_ns=perf_offset_ns + 300_000_000,
        )
        transition = runner._confirm_visual_transition(
            from_gate_index=0,
            to_gate_index=1,
            race_status=runner._visual_race_status_ref(),
        )
        assert transition.retired_track_id == initial_current_id
        assert transition.promoted_track_id == promoted_id
        runner._gate0_transition_proof = vq2_module.GateTransitionProof(
            pre_gate_race_boot_ms=1_000,
            post_gate_race_boot_ms=1_300,
            flight_started_monotonic_s=0.10,
            crossing_started_monotonic_s=0.29,
            pass_confirmed_monotonic_s=0.30,
            next_control_deadline_s=0.30,
            vision_generation=7,
            vision_frame_id=104,
            vision_sim_time_ns=104_000,
            vision_received_monotonic_s=0.26,
            pass_rpy_rad=(0.0, -0.05, 0.0),
        )
        return {
            "gate0_passed": True,
            "visual_next_gate_blend": {
                "enabled": True,
                "started": True,
                "current_track_id": initial_current_id,
                "blended_next_track_id": promoted_id,
                "fresh_blend_frame_count": 4,
                "command_count": 4,
                "withdrawn_before_confirmation": True,
                "yaw_reference_rad": 0.0,
                "max_abs_yaw_excursion_rad": 0.01,
            },
        }

    publications = [
        (105, 465, 65, 0.320),
        (106, 458, 71, 0.340),
        (107, 451, 77, 0.360),
        (108, 444, 83, 0.380),
        (109, 437, 89, 0.400),
        (110, 430, 95, 0.420),
        (111, 423, 101, 0.440),
        (112, 416, 107, 0.460),
        (113, 409, 113, 0.480),
        (114, 402, 119, 0.500),
        (115, 395, 125, 0.520),
    ]
    publication_index = [0]

    def sample():
        if publication_index[0] >= len(publications):
            return
        frame_id, x, y, observed_s = publications[
            publication_index[0]
        ]
        if clock[0] + 1e-9 < observed_s:
            return
        frame = _frame(
            frame_id,
            [_detection(x, y, 50, 60)],
            final_packet_ns=round(
                (
                    perf_offset_s
                    + (0.299 if frame_id == 105 else observed_s)
                )
                * 1_000_000_000
            ),
        )
        if frame_id == 105:
            # A pre-credit observation decoded and published after credit is
            # not the required fresh post-credit camera observation.
            frame = replace(
                frame,
                publish_monotonic_ns=perf_offset_ns + 305_000_000,
            )
        _update_visual(runner, frame)
        publication_index[0] += 1

    async def send(command, **kwargs):
        start_ns = round(
            (perf_offset_s + clock[0]) * 1_000_000_000
        )
        not_before_ns = kwargs.get("wire_start_not_before_ns")
        deadline_ns = kwargs.get("wire_start_deadline_ns")
        if not_before_ns is not None:
            assert start_ns >= not_before_ns
        if deadline_ns is not None:
            assert start_ns < deadline_ns
        commands.append((clock[0], command, dict(kwargs)))
        runner._last_flight_command_sent_s = clock[0]
        if kwargs.get("require_wire_receipt"):
            runner._last_flight_command_started_ns = start_ns
            return {
                "schema": "aigp-vq2-attitude-target-outbound/1",
                "host_clock_id": "host-perf-counter",
                "api": "send_attitude_rate",
                "outcome": "returned",
                "call_start_monotonic_ns": start_ns,
                "call_end_monotonic_ns": start_ns + 1,
            }
        return None

    async def sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def attitude_command(
        _estimate,
        *,
        target_roll_rad,
        target_pitch_rad,
        thrust,
    ):
        return AttitudeRateCommand(
            roll_rate=target_roll_rad,
            pitch_rate=target_pitch_rad,
            yaw_rate=0.0,
            thrust=thrust,
        )

    monkeypatch.setattr(runner, "_run_gate0", run_gate0)
    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_send_flight_command", send)
    monkeypatch.setattr(runner, "_record_tick", lambda *_args: None)
    monkeypatch.setattr(
        vq2_module,
        "attitude_rate_command",
        attitude_command,
    )
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(
                monotonic=lambda: clock[0],
                perf_counter_ns=lambda: round(
                    (perf_offset_s + clock[0]) * 1_000_000_000
                ),
            ),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(
                sleep=sleep,
                CancelledError=asyncio.CancelledError,
            ),
        )
        summary = asyncio.run(
            runner._run_visual_alignment(context)
        )

    assert summary["success"] is True
    assert summary["authoritative_transition"] == [0, 1]
    assert summary["promoted_current_track_id"] == promoted_id
    assert runner.visual_tracker.track(promoted_id).first_token.frame_id == 100
    assert summary["horizontal_abs_error_trend"]["trend"] == (
        "negative_uninterrupted"
    )
    assert summary["vertical_abs_error_trend"]["trend"] == (
        "negative_uninterrupted"
    )
    assert summary["eligible_joint_frame_count"] >= 3
    assert summary["fresh_control_frame_count"] > (
        summary["eligible_joint_frame_count"]
    )
    assert summary["ambiguity"] is False
    assert summary["post_credit_zero_command_count"] == 1
    late_published_sample = next(
        sample
        for sample in runner.visual_tracker.track(promoted_id).history
        if sample.token.frame_id == 105
    )
    assert (
        late_published_sample.observation_monotonic_ns
        < perf_offset_ns + 300_000_000
        < late_published_sample.publication_monotonic_ns
    )
    assert summary["postpromotion_recovery"] == {
        "required": False,
        "outcome": "not_required",
        "reason": None,
        "hard_duration_s": visual_recovery.RECOVERY_HARD_DURATION_S,
        "max_fresh_frames": (
            visual_recovery.RECOVERY_MAX_FRESH_FRAMES
        ),
        "max_commands": visual_recovery.RECOVERY_MAX_COMMANDS,
        "max_thrust": visual_recovery.RECOVERY_MAX_THRUST,
        "max_initial_postcredit_promotion_frames": 1,
        "stale_credit_anchor_command_allowed": False,
        "promotion_identity_basis": None,
        "cross_gap_identity_claimed": None,
        "anchor_admission": None,
        "fresh_frame_count": 0,
        "command_count": 0,
        "strict_entry_streak": 0,
        "completed_frame_token": None,
        "completion_elapsed_s": None,
        "latest_continuation": None,
        "latest_wire_revalidation": None,
    }
    zero_commands = commands[: summary["post_credit_zero_command_count"]]
    assert len(zero_commands) == 1
    assert zero_commands[0][1] == AttitudeRateCommand(
        0.0,
        0.0,
        0.0,
        0.0,
    )
    navigation_commands = commands[
        summary["post_credit_zero_command_count"]:
    ]
    assert navigation_commands
    assert navigation_commands[0][1].yaw_rate < 0.0
    assert all(
        later[0] - earlier[0]
        >= vq2_module.CONTROL_PERIOD_S - 1e-9
        for earlier, later in zip(commands, commands[1:])
    )
    assert all(
        item[2].get("require_wire_receipt") is True
        and item[2].get("wire_start_deadline_ns") is not None
        for item in navigation_commands
    )
    assert all(
        abs(command.roll_rate)
        <= vq2_module.VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S
        and abs(command.pitch_rate)
        <= vq2_module.VISUAL_ALIGN_MAX_COMMAND_RATE_RAD_S
        and abs(command.yaw_rate)
        <= vq2_module.VISUAL_ALIGN_MAX_YAW_RATE_RAD_S
        and vq2_module.VISUAL_ALIGN_MIN_THRUST
        <= command.thrust
        <= vq2_module.VISUAL_ALIGN_MAX_THRUST
        for _sent_s, command, _kwargs in navigation_commands
    )


def test_visual_alignment_recovers_promoted_anchor_before_restricted_authority(
    monkeypatch,
):
    """Recovery wires the sealed post-credit promotion anchor, never stale credit."""

    perf_offset_s = 10.0
    perf_offset_ns = round(perf_offset_s * 1_000_000_000)
    race_credit_relative_ns = 236_005_700
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    context, initial_current_id, promoted_id = _prime_recovery_gate_graph(
        runner,
        adapter,
        perf_clock_offset_ns=perf_offset_ns,
    )
    orientation = _Orientation(pitch=-0.04)
    runner.estimate = SimpleNamespace(
        orientation=orientation,
        body_rates=(0.0, 0.0, 0.0),
        yaw=orientation.yaw,
    )
    clock = [0.267]
    commands = []
    timeline = []

    async def run_gate0(
        _context,
        *,
        capture_transition=False,
        visual_next_gate_blend=False,
    ):
        assert _context is context
        assert capture_transition is False
        assert visual_next_gate_blend is True
        promotion_observation_s = 0.2660666
        promotion_packet_ns = round(
            (perf_offset_s + promotion_observation_s) * 1_000_000_000
        )
        runner.vision.current_snapshot = _snapshot(
            frame_id=173,
            publication_sequence=173,
            final_packet_ns=promotion_packet_ns,
            generation=7,
        )
        _update_visual(
            runner,
            _frame(
                173,
                [_detection(451, 19, 92, 85, confidence=0.95)],
                final_packet_ns=promotion_packet_ns,
            ),
        )
        _set_race(
            adapter,
            gate_index=1,
            boot_ms=1_300,
            sequence=11,
            received_ns=(
                perf_offset_ns + race_credit_relative_ns
            ),
        )
        transition = runner._confirm_visual_transition(
            from_gate_index=0,
            to_gate_index=1,
            race_status=runner._visual_race_status_ref(),
        )
        assert transition.retired_track_id == initial_current_id
        assert transition.promoted_track_id == promoted_id
        assert transition.camera_token_at_credit.frame_id == 172
        assert transition.promoted_latest_token_before_credit.frame_id == 172
        assert transition.promoted_history_length_at_credit == 9
        assert transition.promoted_latest_token_at_promotion.frame_id == 173
        assert transition.promoted_first_token.frame_id == 164
        assert transition.history_length_before_promotion == 10
        assert transition.history_length_after_promotion == 10
        runner._gate0_transition_proof = vq2_module.GateTransitionProof(
            pre_gate_race_boot_ms=1_000,
            post_gate_race_boot_ms=1_300,
            flight_started_monotonic_s=0.05,
            crossing_started_monotonic_s=clock[0] - 0.020,
            pass_confirmed_monotonic_s=clock[0] - 0.001,
            next_control_deadline_s=clock[0],
            vision_generation=7,
            vision_frame_id=173,
            vision_sim_time_ns=173_000,
            vision_received_monotonic_s=promotion_observation_s,
            pass_rpy_rad=(0.0, -0.04, 0.0),
        )
        return {
            "gate0_passed": True,
            "visual_next_gate_blend": {
                "enabled": True,
                "started": True,
                "current_track_id": initial_current_id,
                "blended_next_track_id": promoted_id,
                "fresh_blend_frame_count": 5,
                "command_count": 5,
                "withdrawn_before_confirmation": True,
                "yaw_reference_rad": 0.0,
                "max_abs_yaw_excursion_rad": 0.01,
            },
        }

    # Frame 173 exactly reproduces the failed live promotion token.  It was
    # published after race ingress but processed before promotion, so it is the
    # sole sealed post-credit promotion sample and is recovery-only.  Frames
    # 174 and 175 move inward and establish two strict ordinary-entry
    # admissions before restricted authority begins.
    publications = [
        (174, 450, 19, 92, 84, 0.296),
        (175, 449, 22, 92, 83, 0.326),
    ]
    publications.extend(
        (
            176 + offset,
            448 - 3 * offset,
            24 + 2 * offset,
            89,
            82,
            0.346 + 0.020 * offset,
        )
        for offset in range(20)
    )
    publication_index = [0]

    def sample():
        if publication_index[0] >= len(publications):
            return
        frame_id, x, y, width, height, observed_s = publications[
            publication_index[0]
        ]
        if clock[0] + 1e-9 < observed_s:
            return
        final_packet_ns = round(
            (perf_offset_s + observed_s) * 1_000_000_000
        )
        runner.vision.current_snapshot = _snapshot(
            frame_id=frame_id,
            publication_sequence=frame_id,
            final_packet_ns=final_packet_ns,
            generation=7,
        )
        _update_visual(
            runner,
            _frame(
                frame_id,
                [_detection(x, y, width, height, confidence=0.95)],
                final_packet_ns=final_packet_ns,
            ),
        )
        publication_index[0] += 1

    async def send(command, **kwargs):
        start_ns = round(
            (perf_offset_s + clock[0]) * 1_000_000_000
        )
        not_before_ns = kwargs.get("wire_start_not_before_ns")
        deadline_ns = kwargs.get("wire_start_deadline_ns")
        if not_before_ns is not None:
            assert start_ns >= not_before_ns
        if deadline_ns is not None:
            assert start_ns < deadline_ns
        current = runner.visual_tracker.track(promoted_id)
        commands.append(
            {
                "sent_s": clock[0],
                "command": command,
                "kwargs": dict(kwargs),
                "track_id": current.track_id,
                "frame_token": current.latest_token,
                "normalized_x": current.center_norm[0],
            }
        )
        runner._last_flight_command_sent_s = clock[0]
        if kwargs.get("require_wire_receipt"):
            runner._last_flight_command_started_ns = start_ns
            wire_token = kwargs.get("wire_visual_token")
            wire_authority = None
            if wire_token is not None:
                wire_authority = {
                    "schema": "aigp-vq2-visual-wire-authority/1",
                    "frame_token": asdict(wire_token),
                    "publication_lock_acquired_monotonic_ns": start_ns,
                    "call_start_monotonic_ns": start_ns,
                    "call_end_monotonic_ns": start_ns + 1,
                    "transport_return_monotonic_ns": start_ns + 1,
                    "publication_pinned_through_transport_return": True,
                }
            return {
                "schema": "aigp-vq2-attitude-target-outbound/1",
                "host_clock_id": _HOST_CLOCK,
                "api": "send_attitude_rate",
                "outcome": "returned",
                "call_start_monotonic_ns": start_ns,
                "call_end_monotonic_ns": start_ns + 1,
                "visual_receiver_authority": wire_authority,
            }
        return None

    async def sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def attitude_command(
        _estimate,
        *,
        target_roll_rad,
        target_pitch_rad,
        thrust,
    ):
        return AttitudeRateCommand(
            roll_rate=target_roll_rad,
            pitch_rate=target_pitch_rad,
            yaw_rate=0.0,
            thrust=thrust,
        )

    def record_tick(stage, elapsed_s, command):
        timeline.append(
            ("tick", stage, clock[0], elapsed_s, command)
        )

    def record_event(event, **fields):
        timeline.append(("event", event, clock[0], fields))
        return True

    monkeypatch.setattr(runner, "_run_gate0", run_gate0)
    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_send_flight_command", send)
    monkeypatch.setattr(runner, "_record_tick", record_tick)
    monkeypatch.setattr(runner.recorder, "emit", record_event)
    monkeypatch.setattr(
        vq2_module,
        "attitude_rate_command",
        attitude_command,
    )
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(
                monotonic=lambda: clock[0],
                perf_counter_ns=lambda: round(
                    (perf_offset_s + clock[0]) * 1_000_000_000
                ),
            ),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(
                sleep=sleep,
                CancelledError=asyncio.CancelledError,
            ),
        )
        summary = asyncio.run(
            runner._run_visual_alignment(context)
        )

    assert summary["success"] is True
    assert summary["authoritative_transition"] == [0, 1]
    assert summary["promoted_current_track_id"] == promoted_id
    recovery = summary["postpromotion_recovery"]
    assert recovery["required"] is True
    assert recovery["outcome"] == "recovered"
    assert recovery["promotion_identity_basis"] == (
        "continuous_transition_visibility_v1"
    )
    assert recovery["cross_gap_identity_claimed"] is False
    assert recovery["fresh_frame_count"] == 2
    assert recovery["command_count"] == 2
    assert recovery["strict_entry_streak"] == 2
    assert recovery["completed_frame_token"]["frame_id"] == 175
    assert recovery["latest_wire_revalidation"]["kind"] == (
        "postcredit_continuation"
    )
    assert recovery["latest_wire_revalidation"]["frame_token"] == (
        recovery["latest_wire_revalidation"]["receiver_token"]
    )
    assert recovery["latest_wire_revalidation"]["frame_token"][
        "frame_id"
    ] == 174
    transition = runner._visual_transition
    assert transition is not None
    assert summary["camera_token_at_credit"]["frame_id"] == 172
    assert summary["promoted_latest_token_before_credit"]["frame_id"] == 172
    assert summary["promoted_history_length_at_credit"] == 9
    assert summary["promoted_latest_token_at_promotion"]["frame_id"] == 173
    assert summary["history_length_before_promotion"] == 10
    assert summary["history_length_after_promotion"] == 10
    assert summary["initial_postcredit_promotion_frame_count"] == 1
    assert recovery["max_initial_postcredit_promotion_frames"] == 1
    assert recovery["stale_credit_anchor_command_allowed"] is False
    promotion_digest = transition.promoted_history_sha256
    assert len(promotion_digest) == 64
    assert recovery["anchor_admission"][
        "promotion_identity_sha256"
    ] == promotion_digest
    assert recovery["latest_continuation"][
        "promotion_identity_sha256"
    ] == promotion_digest
    assert recovery["latest_wire_revalidation"][
        "promotion_identity_sha256"
    ] == promotion_digest
    assert recovery["anchor_admission"]["promotion_identity_basis"] == (
        "continuous_transition_visibility_v1"
    )
    assert recovery["latest_continuation"][
        "promotion_identity_basis"
    ] == "continuous_transition_visibility_v1"
    assert recovery["latest_wire_revalidation"][
        "promotion_identity_basis"
    ] == "continuous_transition_visibility_v1"
    assert recovery["latest_wire_revalidation"][
        "cross_gap_identity_claimed"
    ] is False
    assert recovery["latest_wire_revalidation"][
        "reacquisition_bridge"
    ] is None
    assert recovery["latest_wire_revalidation"]["wire_authority"][
        "publication_pinned_through_transport_return"
    ] is True
    assert (
        recovery["latest_wire_revalidation"][
            "receiver_checked_monotonic_ns"
        ]
        >= recovery["latest_wire_revalidation"][
            "validated_monotonic_ns"
        ]
    )
    assert recovery["anchor_admission"]["promotion_anchor_token"][
        "frame_id"
    ] == 173
    assert recovery["anchor_admission"]["credit_prefix_token"][
        "frame_id"
    ] == 172
    assert recovery["anchor_admission"]["history_tokens"] == tuple(
        {
            "generation": 7,
            "frame_id": frame_id,
            "publication_sequence": frame_id,
            "stream_id": _CAMERA_STREAM,
        }
        for frame_id in range(170, 174)
    )

    recovery_frames = [
        item
        for item in timeline
        if item[:2] == ("event", "visual_alignment_recovery_frame")
    ]
    assert [
        item[3]["strict_entry_streak"] for item in recovery_frames
    ] == [1, 2]
    assert recovery_frames[0][3]["strict_entry"] is not None
    assert [
        (
            item[3]["continuation"]["previous_token"]["frame_id"],
            item[3]["continuation"]["frame_token"]["frame_id"],
        )
        for item in recovery_frames
    ] == [(173, 174), (174, 175)]
    for item in recovery_frames:
        continuation = item[3]["continuation"]
        assert continuation["track_id"] == promoted_id
        assert continuation["frame_token"]["stream_id"] == _CAMERA_STREAM
        assert continuation["frame_token"]["generation"] == 7
        assert (
            continuation["frame_token"]["publication_sequence"]
            == continuation["previous_token"]["publication_sequence"] + 1
        )
        assert continuation["observation_age_s"] <= (
            visual_recovery.RECOVERY_MAX_CONTINUATION_AGE_S
        )
        assert continuation["promotion_identity_sha256"] == (
            promotion_digest
        )
        assert continuation["reacquisition_bridge"] is None
    admission_events = [
        item
        for item in timeline
        if item[:2]
        in {
            ("event", "visual_alignment_recovery_admitted"),
            ("event", "visual_alignment_recovery_anchor"),
        }
    ]
    assert len(admission_events) == 2
    assert all(
        item[3]["admission"]["promotion_identity_sha256"]
        == promotion_digest
        and item[3]["admission"]["reacquisition_bridge"] is None
        for item in admission_events
    )
    transition_event = next(
        item[3]
        for item in timeline
        if item[:2] == ("event", "visual_gate_transition_promoted")
    )
    assert transition_event["camera_token_at_credit"][-2:] == [172, 172]
    assert transition_event["promoted_latest_token_before_credit"][-2:] == [
        172,
        172,
    ]
    assert transition_event["promoted_history_length_at_credit"] == 9
    assert transition_event["promoted_latest_token_at_promotion"][-2:] == [
        173,
        173,
    ]
    completion_index = next(
        index
        for index, item in enumerate(timeline)
        if item[:2]
        == ("event", "visual_alignment_recovery_complete")
    )
    restricted_tick_index = next(
        index
        for index, item in enumerate(timeline)
        if item[:2] == ("tick", "visual-align/restricted")
    )
    assert completion_index < restricted_tick_index

    recovery_commands = commands[: recovery["command_count"]]
    assert [
        item["frame_token"].frame_id for item in recovery_commands
    ] == [173, 174]
    assert all(
        item["frame_token"].frame_id
        != transition.promoted_latest_token_before_credit.frame_id
        for item in recovery_commands
    )
    assert all(
        item["track_id"] == promoted_id
        and item["kwargs"].get("require_wire_receipt") is True
        and item["kwargs"].get("wire_start_deadline_ns") is not None
        and item["kwargs"].get("wire_visual_token")
        == item["frame_token"]
        for item in recovery_commands
    )
    assert all(
        item["kwargs"]["wire_start_deadline_ns"]
        - round((perf_offset_s + item["sent_s"]) * 1_000_000_000)
        <= round(
            visual_recovery.RECOVERY_MAX_VALIDATION_TO_WIRE_DELAY_S
            * 1_000_000_000
        )
        for item in recovery_commands
    )
    assert all(
        command["normalized_x"] * command["command"].yaw_rate < 0.0
        and command["command"].roll_rate == 0.0
        and 0.0
        <= command["command"].pitch_rate
        <= visual_recovery.RECOVERY_MAX_COMMAND_RATE_RAD_S
        and abs(command["command"].yaw_rate)
        <= visual_recovery.RECOVERY_MAX_YAW_RATE_RAD_S
        and vq2_module.VISUAL_ALIGN_MIN_THRUST
        <= command["command"].thrust
        <= visual_recovery.RECOVERY_MAX_THRUST
        for command in recovery_commands
    )
    assert all(
        later["sent_s"] - earlier["sent_s"]
        >= vq2_module.CONTROL_PERIOD_S - 1e-9
        for earlier, later in zip(commands, commands[1:])
    )

    promoted = runner.visual_tracker.track(promoted_id)
    assert promoted.first_token.frame_id == 164
    assert promoted.role is VisualTrackRole.CURRENT
    assert promoted.authoritative_gate_index == 1
    assert tuple(
        sample.token.frame_id for sample in promoted.history[4:12]
    ) == (168, 169, 170, 171, 172, 173, 174, 175)
    assert all(
        sample.provenance_basis
        is FrameProvenanceBasis.RECEIVER_TIMING_V1
        and sample.association_confidence >= (
            visual_recovery.RECOVERY_MIN_ASSOCIATION_CONFIDENCE
        )
        for sample in promoted.history[4:12]
    )


def test_visual_alignment_recovery_dispatch_abort_records_nested_outcome(
    monkeypatch,
):
    perf_offset_s = 10.0
    perf_offset_ns = round(perf_offset_s * 1_000_000_000)
    race_credit_relative_ns = 236_005_700
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    context, initial_current_id, promoted_id = _prime_recovery_gate_graph(
        runner,
        adapter,
        perf_clock_offset_ns=perf_offset_ns,
    )
    orientation = _Orientation(pitch=-0.04)
    runner.estimate = SimpleNamespace(
        orientation=orientation,
        body_rates=(0.0, 0.0, 0.0),
        yaw=orientation.yaw,
    )
    clock = [
        race_credit_relative_ns / 1_000_000_000.0 + 0.001
    ]
    failure_observed_s = []

    async def run_gate0(
        _context,
        *,
        capture_transition=False,
        visual_next_gate_blend=False,
    ):
        assert _context is context
        assert capture_transition is False
        assert visual_next_gate_blend is True
        _set_race(
            adapter,
            gate_index=1,
            boot_ms=1_300,
            sequence=11,
            received_ns=perf_offset_ns + race_credit_relative_ns,
        )
        transition = runner._confirm_visual_transition(
            from_gate_index=0,
            to_gate_index=1,
            race_status=runner._visual_race_status_ref(),
        )
        assert transition.retired_track_id == initial_current_id
        assert transition.promoted_track_id == promoted_id
        runner._gate0_transition_proof = vq2_module.GateTransitionProof(
            pre_gate_race_boot_ms=1_000,
            post_gate_race_boot_ms=1_300,
            flight_started_monotonic_s=0.05,
            crossing_started_monotonic_s=clock[0] - 0.020,
            pass_confirmed_monotonic_s=clock[0] - 0.001,
            next_control_deadline_s=clock[0],
            vision_generation=7,
            vision_frame_id=172,
            vision_sim_time_ns=172_000,
            vision_received_monotonic_s=0.2316235,
            pass_rpy_rad=(0.0, -0.04, 0.0),
        )
        return {
            "gate0_passed": True,
            "visual_next_gate_blend": {
                "enabled": True,
                "started": True,
                "current_track_id": initial_current_id,
                "blended_next_track_id": promoted_id,
                "fresh_blend_frame_count": 5,
                "command_count": 5,
                "withdrawn_before_confirmation": True,
                "yaw_reference_rad": 0.0,
                "max_abs_yaw_excursion_rad": 0.01,
            },
        }

    async def fail_dispatch(_command, **kwargs):
        assert kwargs["require_wire_receipt"] is True
        failure_observed_s.append(clock[0])
        raise RuntimeError("uncertain recovery transport failure")

    async def sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def attitude_command(
        _estimate,
        *,
        target_roll_rad,
        target_pitch_rad,
        thrust,
    ):
        return AttitudeRateCommand(
            roll_rate=target_roll_rad,
            pitch_rate=target_pitch_rad,
            yaw_rate=0.0,
            thrust=thrust,
        )

    monkeypatch.setattr(runner, "_run_gate0", run_gate0)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "_send_flight_command", fail_dispatch)
    monkeypatch.setattr(runner, "_record_tick", lambda *_args: None)
    monkeypatch.setattr(
        vq2_module,
        "attitude_rate_command",
        attitude_command,
    )
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(
                monotonic=lambda: clock[0],
                perf_counter_ns=lambda: round(
                    (perf_offset_s + clock[0]) * 1_000_000_000
                ),
            ),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(
                sleep=sleep,
                CancelledError=asyncio.CancelledError,
            ),
        )
        with pytest.raises(
            vq2_module.SafetyAbort,
            match="recovery dispatch failed closed",
        ):
            asyncio.run(runner._run_visual_alignment(context))

    assert len(failure_observed_s) == 1
    assert clock[0] >= (
        failure_observed_s[0] + vq2_module.CONTROL_PERIOD_S - 1e-9
    )
    summary = runner._visual_alignment_summary
    assert summary is not None
    assert summary["outcome"] == "abort"
    assert summary["success"] is False
    assert summary["postpromotion_recovery"]["required"] is True
    assert summary["postpromotion_recovery"]["outcome"] == "abort"
    assert "recovery dispatch failed closed" in (
        summary["postpromotion_recovery"]["reason"]
    )
    assert summary["postpromotion_recovery"]["command_count"] == 0


def test_visual_alignment_rejects_blended_identity_promotion_mismatch(
    monkeypatch,
):
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    context, initial_current_id, promoted_id = _prime_bound_gate_graph(
        runner,
        adapter,
    )

    async def run_gate0(
        _context,
        *,
        capture_transition=False,
        visual_next_gate_blend=False,
    ):
        assert _context is context
        assert capture_transition is False
        assert visual_next_gate_blend is True
        _set_race(
            adapter,
            gate_index=1,
            boot_ms=1_300,
            sequence=11,
            received_ns=300_000_000,
        )
        transition = runner._confirm_visual_transition(
            from_gate_index=0,
            to_gate_index=1,
            race_status=runner._visual_race_status_ref(),
        )
        assert transition.retired_track_id == initial_current_id
        assert transition.promoted_track_id == promoted_id
        return {
            "gate0_passed": True,
            "visual_next_gate_blend": {
                "enabled": True,
                "started": True,
                "current_track_id": initial_current_id,
                "blended_next_track_id": "different-prepass-track",
                "fresh_blend_frame_count": 4,
                "command_count": 4,
                "withdrawn_before_confirmation": True,
                "yaw_reference_rad": 0.0,
                "max_abs_yaw_excursion_rad": 0.01,
            },
        }

    monkeypatch.setattr(runner, "_run_gate0", run_gate0)

    with pytest.raises(
        vq2_module.SafetyAbort,
        match="blended identity was not authoritatively promoted",
    ):
        asyncio.run(runner._run_visual_alignment(context))


def test_visual_alignment_uncertain_dispatch_reserves_cleanup_slot(
    monkeypatch,
):
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    context, _initial_current_id, promoted_id = _prime_bound_gate_graph(
        runner,
        adapter,
    )
    runner.estimate = _alignment_estimate()
    clock = [0.320]
    failure_observed_s = []

    async def run_gate0(
        _context,
        *,
        capture_transition=False,
        visual_next_gate_blend=False,
    ):
        assert _context is context
        assert capture_transition is False
        assert visual_next_gate_blend is True
        _set_race(
            adapter,
            gate_index=1,
            boot_ms=1_300,
            sequence=11,
            received_ns=300_000_000,
        )
        transition = runner._confirm_visual_transition(
            from_gate_index=0,
            to_gate_index=1,
            race_status=runner._visual_race_status_ref(),
        )
        assert transition.promoted_track_id == promoted_id
        runner._gate0_transition_proof = vq2_module.GateTransitionProof(
            pre_gate_race_boot_ms=1_000,
            post_gate_race_boot_ms=1_300,
            flight_started_monotonic_s=0.10,
            crossing_started_monotonic_s=0.29,
            pass_confirmed_monotonic_s=0.30,
            next_control_deadline_s=0.30,
            vision_generation=7,
            vision_frame_id=104,
            vision_sim_time_ns=104_000,
            vision_received_monotonic_s=0.26,
            pass_rpy_rad=(0.0, -0.05, 0.0),
        )
        return {
            "gate0_passed": True,
            "visual_next_gate_blend": {
                "enabled": True,
                "started": True,
                "current_track_id": _initial_current_id,
                "blended_next_track_id": promoted_id,
                "fresh_blend_frame_count": 4,
                "command_count": 4,
                "withdrawn_before_confirmation": True,
                "yaw_reference_rad": 0.0,
                "max_abs_yaw_excursion_rad": 0.01,
            },
        }

    published = [False]

    def sample():
        if not published[0]:
            _update_visual(
                runner,
                _frame(
                    105,
                    [_detection(465, 65, 50, 60)],
                    final_packet_ns=320_000_000,
                ),
            )
            published[0] = True

    async def uncertain_send(_command, **_kwargs):
        clock[0] += 0.050
        failure_observed_s.append(clock[0])
        raise RuntimeError("uncertain adapter return")

    async def sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    monkeypatch.setattr(runner, "_run_gate0", run_gate0)
    monkeypatch.setattr(runner, "_sample", sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "_send_flight_command",
        uncertain_send,
    )
    monkeypatch.setattr(runner, "_record_tick", lambda *_args: None)
    monkeypatch.setattr(
        vq2_module,
        "attitude_rate_command",
        lambda _estimate, *, target_roll_rad, target_pitch_rad, thrust: (
            AttitudeRateCommand(
                target_roll_rad,
                target_pitch_rad,
                0.0,
                thrust,
            )
        ),
    )
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(
            vq2_module,
            "time",
            SimpleNamespace(
                monotonic=lambda: clock[0],
                perf_counter_ns=lambda: round(
                    clock[0] * 1_000_000_000
                ),
            ),
        )
        clock_patch.setattr(
            vq2_module,
            "asyncio",
            SimpleNamespace(
                sleep=sleep,
                CancelledError=asyncio.CancelledError,
            ),
        )
        with pytest.raises(
            vq2_module.SafetyAbort,
            match="visual alignment command dispatch failed closed",
        ):
            asyncio.run(runner._run_visual_alignment(context))

    assert failure_observed_s == [pytest.approx(0.370)]
    assert clock[0] >= (
        failure_observed_s[0] + vq2_module.CONTROL_PERIOD_S - 1e-9
    )


@pytest.mark.parametrize(
    (
        "cleanup_gate",
        "cleanup_confirmed",
        "criteria_met",
        "reported_track_id",
        "race_finished",
        "expected_success",
    ),
    [
        (1, True, True, "vq2-track-000004", False, True),
        (1, False, True, "vq2-track-000004", False, True),
        (2, True, True, "vq2-track-000004", False, False),
        (1, True, False, "vq2-track-000004", False, False),
        (1, True, True, "vq2-track-999999", False, False),
        (1, True, True, "vq2-track-000004", True, False),
    ],
)
def test_powered_visual_alignment_preserves_navigation_but_requires_cleanup(
    monkeypatch,
    cleanup_gate,
    cleanup_confirmed,
    criteria_met,
    reported_track_id,
    race_finished,
    expected_success,
):
    adapter = _Adapter()
    runner = vq2_module.VQ2Runner(adapter, _Vision())
    context = vq2_module.StartContext(
        0.0,
        -0.31,
        320,
        180,
        6_400,
        1_000,
    )
    promoted_id = "vq2-track-000004"

    async def no_result(*_args, **_kwargs):
        return None

    async def wait_for_go():
        _set_race(
            adapter,
            gate_index=0,
            boot_ms=1_000,
            sequence=1,
            received_ns=10_000_000,
        )
        return context

    async def run_alignment(_context):
        assert _context is context
        _set_race(
            adapter,
            gate_index=cleanup_gate,
            boot_ms=1_300,
            sequence=2,
            received_ns=20_000_000,
            race_finish_time_ns=(25_000_000 if race_finished else -1),
        )
        runner._visual_transition = SimpleNamespace(
            from_gate_index=0,
            to_gate_index=1,
            promoted_track_id=promoted_id,
        )
        summary = {
            "promoted_current_track_id": reported_track_id,
            "alignment_criteria_met": criteria_met,
            "outcome": "success",
            "success": True,
        }
        runner._visual_alignment_summary = dict(summary)
        return summary

    async def cleanup():
        return cleanup_confirmed

    monkeypatch.setattr(runner, "establish_reset_epoch", no_result)
    monkeypatch.setattr(runner, "normalize_disarmed", no_result)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(
        runner,
        "_bind_initial_visual_gate",
        lambda _context: None,
    )
    monkeypatch.setattr(runner, "arm_confirmed", no_result)
    monkeypatch.setattr(
        runner,
        "_run_visual_alignment",
        run_alignment,
    )
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(
        runner.run_powered_stage(
            vq2_module.VISUAL_ALIGN_STAGE,
            write_diagnostic_pngs=False,
        )
    )

    assert result.success is bool(expected_success and cleanup_confirmed)
    assert result.cleanup_confirmed is cleanup_confirmed
    assert result.details["visual_alignment"]["cleanup_confirmed"] is (
        cleanup_confirmed
    )
    assert result.details["visual_alignment"]["outcome"] == (
        "success" if expected_success else "abort"
    )
    if not cleanup_confirmed:
        assert "cleanup unconfirmed" in result.reason
