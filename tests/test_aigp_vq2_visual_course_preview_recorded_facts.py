"""Replay compact tracker-state facts from the failed build-3385 course run.

This is not JPEG, detector, UDP, IMU, or dynamics replay; the source retained
no replay bundle.  Exact logged current-track states are injected at the
visual-target boundary while the production tracker, rolling graph,
``RollingVisualApproachServo``, and its real ``ImageVisualServo`` remain live.
"""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import (
    CameraFrameToken,
    FrameProvenanceBasis,
    MultiTargetVisualTracker,
    VisualDetection,
    VisualDetectionFrame,
    VisualTrackRole,
)
from planning.vq2_gate_graph import (
    AuthoritativeRaceStatusRef,
    GateGraphSnapshot,
    RollingVisualGateGraph,
)
from planning.vq2_visual_approach import (
    RollingVisualApproachServo,
    VisualApproachMode,
    VisualApproachRefusal,
)


_SOURCE = (
    "20260726T022215Z-visual-course-262cd923",
    "c1079d5c2b4ba8bf25e80dbee0a637ea99ffce4a",
    "ca07ccf0b60840e77db43137e9c5e2f33449d23bb6f7933994695e62fbb93e11",
    "4984d281c776fbb9984c9d36a5af4ad3e6d39f3eb6f353306186b47aef3ce8f1",
    None,  # replay_bundle
)
_HOST_CLOCK_ID = "host-perf-counter"
_STREAM_ID = "vq2-camera-udp-5600"
_PUB112_OBSERVATION_NS = 165_383_498_790_200
_FRAME_PERIOD_NS = 33_000_000

# (x, y-image-down, x-rate, y-rate, apparent scale, log-scale rate)
_CURRENT = {
    112: (
        0.0031250000000000444,
        -0.06666666666666665,
        -0.0008636623722228738,
        -0.12011445477715357,
        0.16978847134007657,
        0.014223185954750042,
    ),
    113: (
        0.0031250000000000444,
        -0.011111111111111072,
        -0.0003886480675002932,
        0.8259718687710795,
        0.14734337258405775,
        -2.2395755201947827,
    ),
    117: (
        0.0031250000000000444,
        -0.09444444444444444,
        -0.00001593699981793389,
        -1.2745178073557923,
        0.17187184340535694,
        2.704289851018549,
    ),
    127: (
        0.0031250000000000444,
        -0.21666666666666667,
        -0.002840336567176098,
        -0.277696003126965,
        0.18853878911247948,
        0.24994708472535299,
    ),
    159: (
        0.09375,
        -0.1777777777777778,
        1.3613164101334883,
        0.42071469766107417,
        0.5397482051846028,
        3.345486567112519,
    ),
}
_EXACT_OBSERVATION_NS = {
    112: _PUB112_OBSERVATION_NS,
    113: 165_383_533_511_500,
    117: 165_383_665_626_500,
    127: 165_383_998_826_300,
    159: 165_385_068_124_700,
}


def _observation_ns(sequence: int) -> int:
    return _EXACT_OBSERVATION_NS.get(
        sequence,
        _PUB112_OBSERVATION_NS + (sequence - 112) * _FRAME_PERIOD_NS,
    )


def _detection(source: int, x: float, scale: float) -> VisualDetection:
    unit_x = 0.5 * (x + 1.0)
    half = 0.5 * scale
    return VisualDetection(
        source_index=source,
        center_norm=(x, 0.0),
        bbox_norm=(unit_x - half, 0.5 - half, unit_x + half, 0.5 + half),
        confidence=0.9,
    )


def _frame(sequence: int, *, include_next: bool = True) -> VisualDetectionFrame:
    observation_ns = _observation_ns(sequence)
    detections = (_detection(0, 0.0, 0.30),)
    if include_next:
        detections += (_detection(1, 0.30, 0.14),)
    return VisualDetectionFrame(
        token=CameraFrameToken(
            generation=1,
            frame_id=1_304_218 + sequence,
            publication_sequence=sequence,
            stream_id=_STREAM_ID,
        ),
        provenance_basis=FrameProvenanceBasis.RECEIVER_TIMING_V1,
        time_basis_id=_HOST_CLOCK_ID,
        image_size_px=(640, 360),
        detections=detections,
        camera_source_time_ns=observation_ns - 2_000_000,
        final_unique_packet_monotonic_ns=observation_ns,
        publish_monotonic_ns=observation_ns + 1_000_000,
    )


def _setup() -> tuple[
    MultiTargetVisualTracker,
    RollingVisualGateGraph,
    GateGraphSnapshot,
    str,
    str,
]:
    tracker = MultiTargetVisualTracker()
    graph = RollingVisualGateGraph()
    current_id = next_id = ""
    snapshot = None
    for sequence in range(107, 110):
        update = tracker.update(_frame(sequence))
        if sequence == 107:
            current_id, next_id = update.created_track_ids
        if sequence == 109:
            snapshot = graph.bind_initial_current(
                tracker,
                track_id=current_id,
                race_status=AuthoritativeRaceStatusRef.live(
                    session_id="attempt-2-preview-recorded-facts",
                    reset_epoch=1,
                    race_generation=1,
                    race_status_sequence=1,
                    race_status_boot_ms=5_000,
                    active_gate_index=0,
                    received_monotonic_ns=update.publish_monotonic_ns + 1_000_000,
                    host_clock_id=_HOST_CLOCK_ID,
                ),
            )
    assert snapshot is not None
    return tracker, graph, snapshot, current_id, next_id


def _advance(
    tracker: MultiTargetVisualTracker,
    graph: RollingVisualGateGraph,
    sequence: int,
    *,
    include_next: bool = True,
) -> GateGraphSnapshot:
    update = tracker.latest_update
    assert update is not None
    latest = update.token.publication_sequence
    assert latest is not None and sequence > latest
    snapshot = None
    # Retain every intervening camera publication so the real tracker sees the
    # same continuous cadence; only named logged states reach the servo.
    for item in range(latest + 1, sequence + 1):
        tracker.update(
            _frame(
                item,
                include_next=(include_next or item != sequence),
            )
        )
        snapshot = graph.observe(tracker)
    assert snapshot is not None
    return snapshot


def _observe(
    approach: RollingVisualApproachServo,
    snapshot: GateGraphSnapshot,
    tracker: MultiTargetVisualTracker,
):
    update = tracker.latest_update
    assert update is not None
    return approach.observe(
        snapshot,
        tracker,
        now_monotonic_s=update.observation_monotonic_ns / 1e9 + 0.005,
        segment_elapsed_s=1.0,
        segment_yaw_excursion_rad=0.0,
        mode=VisualApproachMode.APPROACH,
    )


def _inject_recorded_current(
    monkeypatch: pytest.MonkeyPatch,
    approach: RollingVisualApproachServo,
    current_id: str,
) -> None:
    original = approach._target

    def target(track, *, now_monotonic_s, require_current_authority):
        value = original(
            track,
            now_monotonic_s=now_monotonic_s,
            require_current_authority=require_current_authority,
        )
        sequence = value.frame_token.publication_sequence
        if value.track_id != current_id:
            return value
        if sequence in {110, 111}:
            return replace(
                value,
                normalized_y_down=(-0.10 if sequence == 110 else -0.08),
                normalized_x_rate_s=0.0,
                normalized_y_rate_down_s=0.0,
                log_scale=math.log(0.17),
                log_scale_rate_s=0.0,
            )
        row = _CURRENT.get(sequence)
        if row is None:
            return value
        x, y, x_rate, y_rate, scale, scale_rate = row
        return replace(
            value,
            normalized_x=x,
            normalized_y_down=y,
            normalized_x_rate_s=x_rate,
            normalized_y_rate_down_s=y_rate,
            log_scale=math.log(scale),
            log_scale_rate_s=scale_rate,
        )

    monkeypatch.setattr(approach, "_target", target)


def _assert_clean_next(
    snapshot: GateGraphSnapshot,
    tracker: MultiTargetVisualTracker,
    next_id: str,
) -> None:
    track = tracker.track(next_id)
    candidate = next(
        item for item in snapshot.next_candidates if item.track_id == next_id
    )
    assert track.visible and track.missed_frame_count == 0
    assert track.role is VisualTrackRole.NEXT
    assert track.clipping is FrameEdge.NONE
    assert not track.center_censored and not track.ambiguous
    assert candidate.promotable and candidate.relationship is not None
    assert not snapshot.next_selection_ambiguous


def test_logged_preview_spikes_and_one_frame_loss_stay_current_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _SOURCE[-1] is None
    tracker, graph, _snapshot, current_id, next_id = _setup()
    approach = RollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
    )
    _inject_recorded_current(monkeypatch, approach, current_id)

    for sequence in (110, 111, 112):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    _assert_clean_next(snapshot, tracker, next_id)
    assert proposal.passage_admission is not None
    assert proposal.passage_admission.preview_track_id == next_id

    # Exact current-rate discontinuities at 113 and 117 suppress optional
    # preview command authority without refusing the safe current controller.
    for sequence in (113, 117):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
        _assert_clean_next(snapshot, tracker, next_id)
        x, y, x_rate, y_rate, _scale, _scale_rate = _CURRENT[sequence]
        output = proposal.servo_output
        assert output.next_gate_blend == 0.0
        assert output.next_horizontal_error is None
        assert output.next_vertical_error_image_down is None
        assert output.effective_horizontal_error == x
        assert output.effective_vertical_error_image_down == y
        assert output.effective_horizontal_rate_s == x_rate
        assert output.effective_vertical_rate_down_s == y_rate
        assert not output.advance_enabled
        assert proposal.passage_admission is None

    snapshot = _advance(tracker, graph, 127)
    proposal = _observe(approach, snapshot, tracker)
    _assert_clean_next(snapshot, tracker, next_id)
    assert proposal.next_target is not None
    assert not proposal.servo_output.advance_enabled
    assert proposal.passage_admission is None

    # Publication 159 is the logged one-frame next miss; current stays visible.
    snapshot = _advance(tracker, graph, 159, include_next=False)
    proposal = _observe(approach, snapshot, tracker)
    assert tracker.track(current_id).visible
    assert not tracker.track(next_id).visible
    assert tracker.track(next_id).missed_frame_count == 1
    assert proposal.next_target is None
    assert proposal.servo_output.next_gate_blend == 0.0
    assert not proposal.servo_output.advance_enabled
    assert proposal.passage_admission is None

    # Fresh same-ID evidence can re-enter the ordinary generic admission path.
    for sequence in (160, 161, 162):
        snapshot = _advance(tracker, graph, sequence)
        proposal = _observe(approach, snapshot, tracker)
    _assert_clean_next(snapshot, tracker, next_id)
    assert proposal.passage_admission is not None
    assert proposal.passage_admission.preview_track_id == next_id
    assert not proposal.servo_output.advance_enabled


def test_logged_state_replay_keeps_stale_current_fail_closed() -> None:
    tracker, graph, _snapshot, current_id, _next_id = _setup()
    approach = RollingVisualApproachServo(
        current_id,
        0,
        next_gate_blend=0.35,
    )
    snapshot = _advance(tracker, graph, 110)
    update = tracker.latest_update
    assert update is not None

    with pytest.raises(VisualApproachRefusal, match="stale"):
        approach.observe(
            snapshot,
            tracker,
            now_monotonic_s=update.observation_monotonic_ns / 1e9 + 1.0,
            segment_elapsed_s=1.0,
            segment_yaw_excursion_rad=0.0,
            mode=VisualApproachMode.APPROACH,
        )
