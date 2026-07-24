"""Focused integration tests for the commandless VQ2 visual-shadow stage."""

from __future__ import annotations

import asyncio
from dataclasses import replace
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
            0.08,
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
        (-0.08, 0.161, 0.0, 0.0, 0.5, -0.08),
        (
            -0.08,
            -math.pi + 0.01,
            math.pi - 0.01,
            0.0,
            -0.5,
            -0.08,
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
            requested_rate_rad_s=0.08,
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
            requested_rate_rad_s=-0.08,
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
        (106, 458, 71, 0.360),
        (107, 451, 77, 0.380),
        (108, 444, 83, 0.400),
        (109, 437, 89, 0.420),
        (110, 430, 95, 0.440),
        (111, 423, 101, 0.460),
        (112, 416, 107, 0.480),
        (113, 409, 113, 0.500),
        (114, 402, 119, 0.520),
        (115, 395, 125, 0.540),
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
        _update_visual(
            runner,
            _frame(
                frame_id,
                [_detection(x, y, 50, 60)],
                final_packet_ns=round(
                    (perf_offset_s + observed_s) * 1_000_000_000
                ),
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
    assert summary["post_credit_zero_command_count"] == 0
    navigation_commands = commands
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
        (1, False, True, "vq2-track-000004", False, False),
        (2, True, True, "vq2-track-000004", False, False),
        (1, True, False, "vq2-track-000004", False, False),
        (1, True, True, "vq2-track-999999", False, False),
        (1, True, True, "vq2-track-000004", True, False),
    ],
)
def test_powered_visual_alignment_requires_gate1_and_confirmed_cleanup(
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

    assert result.success is expected_success
    assert result.cleanup_confirmed is cleanup_confirmed
    assert result.details["visual_alignment"]["cleanup_confirmed"] is (
        cleanup_confirmed
    )
    assert result.details["visual_alignment"]["outcome"] == (
        "success" if expected_success else "abort"
    )
