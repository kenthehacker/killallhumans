"""Focused integration tests for the commandless VQ2 visual-shadow stage."""

from __future__ import annotations

import asyncio
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
)
from gate_detection.src.gate_detector import GateDetection
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
):
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=boot_ms,
        race_start_boot_time_ms=900,
        race_finish_time_ns=-1,
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
            race_finish_time_ns=-1,
        ),
    )


def _update_visual(runner, frame):
    update = runner.visual_tracker.update(frame)
    runner._visual_latest_tracker_update = update
    runner._visual_latest_graph_snapshot = runner.visual_gate_graph.observe(
        runner.visual_tracker
    )
    return update


def _prime_bound_gate_graph(runner, adapter):
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
                final_packet_ns=100_000_000 + 40_000_000 * offset,
            ),
        )
    _set_race(
        adapter,
        gate_index=0,
        boot_ms=1_000,
        sequence=10,
        received_ns=200_000_000,
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
                final_packet_ns=100_000_000 + 40_000_000 * offset,
            ),
        )
    next_ids = [
        track.track_id
        for track in update.visible_tracks
        if track.track_id != current_id
    ]
    assert len(next_ids) == 1
    return context, current_id, next_ids[0]


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
    assert len(transition.pretransition_frame_tokens) >= 3
    promoted = runner.visual_tracker.track(expected_next_id)
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
    ("cleanup_gate", "cleanup_confirmed", "reason_fragment"),
    [
        (1, False, "cleanup unconfirmed"),
        (0, True, "cleanup boundary lacks proved 0->1 authority"),
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
        "transition": [0, 1],
    }
