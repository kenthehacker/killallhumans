"""Pure safety/geometry tests for the staged VQ2 runner."""

from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.aigp_vq2_run as vq2_module
from competition.adapter import IMUData, Quaternion, TelemetryState
from competition.aigp_messages import RaceStatus
from estimation.imu_attitude import (
    AttitudeEstimate,
    ImuAttitudeConfig,
    ImuAttitudeEstimator,
)
from gate_detection.src.gate_detector import GateDetection
from scripts.aigp_vq2_run import (
    GateTargetTracker,
    ResetProof,
    SafetyAbort,
    VQ2Runner,
    attitude_rate_command,
    clock_rolled_back,
    clock_within_epoch_envelope,
    gate_vertical_reference_px,
    gate_control_center_y_px,
    gate_vertical_thrust,
    crossing_status_decision,
    gate_detection_summary,
    is_close_gate_crossing_candidate,
    is_benign_pad_contact,
    is_crossing_residue,
    next_control_deadline,
    post_gate_observation_deadline,
    select_primary_gate,
)


def _detection(x, y, width, height, confidence=0.8):
    return GateDetection(
        center_x=x + width // 2,
        center_y=y + height // 2,
        bbox=(x, y, width, height),
        corners=np.zeros((4, 2)),
        area=width * height,
        estimated_distance=999.0,
        confidence=confidence,
    )


def _estimate(roll=0.0, pitch=-0.31, yaw=0.0):
    return AttitudeEstimate(
        timestamp_us=1,
        orientation=Quaternion.from_euler(roll, pitch, yaw),
        body_rates=(0.0, 0.0, 0.0),
        gyro_bias=(0.0, 0.0, 0.0),
        accel_trust=1.0,
        healthy=True,
        propagated=True,
    )


def _vision_snapshot(
    *,
    frame_id=100,
    sim_time_ns=1_000,
    received_monotonic_s=0.0,
    generation=1,
):
    return SimpleNamespace(
        frame_id=frame_id,
        sim_time_ns=sim_time_ns,
        received_monotonic_s=received_monotonic_s,
        generation=generation,
        camera_frame=SimpleNamespace(
            image=np.zeros((360, 640, 3), dtype=np.uint8)
        ),
        age_s=lambda now=None: max(
            0.0,
            (received_monotonic_s if now is None else now)
            - received_monotonic_s,
        ),
    )


def test_reset_clock_requires_authoritative_margin():
    assert clock_rolled_back(10_000, 100, 500)
    assert not clock_rolled_back(10_000, 9_600, 500)
    assert not clock_rolled_back(10_000, 9_500, 500)


def test_epoch_envelope_rejects_delayed_old_packet():
    assert clock_within_epoch_envelope(
        500, 1_500, 0.5, units_per_second=1_000.0, slack=700
    )
    assert not clock_within_epoch_envelope(
        500, 1_000_000, 0.5, units_per_second=1_000.0, slack=700
    )


def test_control_deadline_drops_missed_ticks_instead_of_catching_up():
    assert next_control_deadline(10.0, 10.0) == pytest.approx(10.02)

    overrun_now = 10.10
    deadline = next_control_deadline(10.02, overrun_now)
    assert deadline == pytest.approx(overrun_now + 0.02)
    assert deadline - overrun_now >= 0.02 - 1e-12


@pytest.mark.parametrize(
    ("pass_confirmed", "flight_started", "crossing_started", "expected"),
    [
        (10.0, 6.0, 9.9, 10.2),
        (10.0, 6.0, 9.7, 10.1),
        (10.0, 5.05, 9.9, 10.05),
        (10.0, 6.0, None, 10.2),
    ],
)
def test_post_gate_deadline_uses_earliest_fixed_safety_bound(
    pass_confirmed,
    flight_started,
    crossing_started,
    expected,
):
    assert post_gate_observation_deadline(
        pass_confirmed_s=pass_confirmed,
        flight_started_s=flight_started,
        crossing_started_s=crossing_started,
    ) == pytest.approx(expected)


def test_primary_gate_uses_largest_plausible_pixel_box_only():
    far = _detection(410, 138, 27, 45)
    near = _detection(282, 134, 80, 81)
    line = _detection(10, 10, 200, 20)
    low_confidence = _detection(0, 0, 100, 100, confidence=0.01)

    assert select_primary_gate([far, line, near, low_confidence]) is near


def test_gate_tracker_requires_temporal_continuity():
    tracker = GateTargetTracker()
    for frame_id in range(1, 4):
        tracker.update(
            [_detection(280 + frame_id, 134, 80, 80)],
            frame_id=frame_id,
            sim_time_ns=frame_id * 10,
            received_monotonic_s=1.0 + frame_id * 0.01,
        )
    assert tracker.consecutive == 3

    previous = tracker.target
    tracker.update(
        [_detection(500, 300, 40, 40)],
        frame_id=4,
        sim_time_ns=40,
        received_monotonic_s=1.04,
    )
    assert tracker.consecutive == 0
    assert tracker.target is previous


def test_pitch_leveling_moves_expected_gate_reference_down():
    reference = gate_vertical_reference_px(174.0, -0.311, -0.10)
    assert reference == pytest.approx(242.5, abs=1.0)


def test_clipped_square_gate_center_uses_visible_width():
    target = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        center_x=332,
        center_y=210,
        bbox=(79, 60, 506, 300),
        confidence=0.8,
    )
    assert gate_control_center_y_px(target) == pytest.approx(313.0)

    fully_clipped = vq2_module.GateTarget(
        frame_id=2,
        sim_time_ns=2,
        received_monotonic_s=1.1,
        center_x=334,
        center_y=180,
        bbox=(101, 0, 466, 360),
        confidence=0.8,
    )
    assert gate_control_center_y_px(
        fully_clipped,
        previous_center_y=216.5,
    ) == pytest.approx(216.5)


def test_gate_vertical_thrust_has_position_and_motion_damping():
    assert gate_vertical_thrust(150.0, 0.0) > 0.275
    assert gate_vertical_thrust(210.0, 0.0) < 0.275
    assert gate_vertical_thrust(180.0, 60.0) < 0.275
    assert 0.21 <= gate_vertical_thrust(0.0, -999.0) <= 0.32


def test_close_crossing_requires_large_centered_both_edge_clipped_gate():
    target = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=1.0,
        center_x=336,
        center_y=180,
        bbox=(61, 0, 551, 360),
        confidence=0.8,
    )
    assert is_close_gate_crossing_candidate(
        target,
        initial_gate_area=6480,
        control_y=217.0,
    )
    assert not is_close_gate_crossing_candidate(
        target,
        initial_gate_area=6480,
        control_y=270.0,
    )

    insufficient_growth = vq2_module.GateTarget(
        frame_id=2,
        sim_time_ns=2,
        received_monotonic_s=1.1,
        center_x=320,
        center_y=180,
        bbox=(160, 0, 320, 360),
        confidence=0.8,
    )
    assert not is_close_gate_crossing_candidate(
        insufficient_growth,
        initial_gate_area=6480,
        control_y=180.0,
    )

    no_vertical_clip = vq2_module.GateTarget(
        frame_id=3,
        sim_time_ns=3,
        received_monotonic_s=1.2,
        center_x=320,
        center_y=180,
        bbox=(44, 10, 551, 340),
        confidence=0.8,
    )
    assert not is_close_gate_crossing_candidate(
        no_vertical_clip,
        initial_gate_area=6480,
        control_y=180.0,
    )

    off_center = vq2_module.GateTarget(
        frame_id=4,
        sim_time_ns=4,
        received_monotonic_s=1.3,
        # A clipped asymmetric contour can have a centroid substantially away
        # from its image-bounded bbox midpoint.
        center_x=410,
        center_y=180,
        bbox=(120, 0, 520, 360),
        confidence=0.8,
    )
    assert not is_close_gate_crossing_candidate(
        off_center,
        initial_gate_area=6480,
        control_y=180.0,
    )


def test_crossing_wait_requires_authoritative_new_race_status():
    common = {
        "baseline_race_boot_ms": 5998,
        "elapsed_s": 0.10,
    }
    assert crossing_status_decision(
        **common,
        current_race_boot_ms=5998,
        active_gate_index=0,
    ) == "waiting"
    assert crossing_status_decision(
        **common,
        current_race_boot_ms=5998,
        active_gate_index=1,
    ) == "waiting"
    assert crossing_status_decision(
        **common,
        current_race_boot_ms=6249,
        active_gate_index=1,
    ) == "passed"
    assert crossing_status_decision(
        **common,
        current_race_boot_ms=6249,
        active_gate_index=0,
    ) == "not_credited"
    assert crossing_status_decision(
        baseline_race_boot_ms=5998,
        current_race_boot_ms=5998,
        active_gate_index=0,
        elapsed_s=0.40,
    ) == "status_timeout"
    assert crossing_status_decision(
        baseline_race_boot_ms=5998,
        current_race_boot_ms=5998,
        active_gate_index=2,
        elapsed_s=0.01,
    ) == "invalid_gate_index"


@pytest.mark.parametrize(
    "bbox",
    [
        (107, 0, 452, 360),
        (75, 0, 518, 360),
        (0, 0, 640, 360),
        (4, 4, 632, 352),
        (75, 20, 518, 340),
    ],
)
def test_post_pass_crossing_residue_catches_large_clipped_remnants(bbox):
    assert is_crossing_residue(_detection(*bbox))


@pytest.mark.parametrize(
    "bbox",
    [
        (127, 16, 410, 344),
        (409, 138, 28, 45),
        (560, 100, 80, 140),
        (280, 0, 80, 360),
        (160, 40, 320, 280),
    ],
)
def test_post_pass_crossing_residue_preserves_plausible_next_gate(bbox):
    assert not is_crossing_residue(_detection(*bbox))


def test_detection_diagnostics_are_strict_json_and_reject_nonfinite_values():
    detection = _detection(409, 138, 28, 45, confidence=math.nan)
    detection.rotation_deg = math.inf
    summary = gate_detection_summary(detection, detector_index=2)

    assert summary["confidence"] is None
    assert summary["rotation_deg"] is None
    assert "nonfinite_confidence" in summary["selector_rejections"]
    json.dumps(summary, allow_nan=False)


def test_attitude_loop_is_finite_clamped_and_never_commands_yaw():
    command = attitude_rate_command(
        _estimate(),
        target_roll_rad=0.08,
        target_pitch_rad=-0.10,
        thrust=0.27,
    )
    assert abs(command.roll_rate) <= 0.25
    assert abs(command.pitch_rate) <= 0.25
    assert command.yaw_rate == 0.0
    assert math.isfinite(command.thrust)


class _FakeVision:
    def __init__(self):
        self.is_running = False
        self.current_snapshot = None
        self.reset_calls = 0

    def stop(self):
        self.is_running = False

    def start(self):
        self.is_running = True

    def reset(self):
        self.reset_calls += 1

    def snapshot(self, **_kwargs):
        return self.current_snapshot


class _FakeAdapter:
    enable_vision = False
    telemetry_mode = "imu"
    fetch_track_on_connect = False
    is_armed = False
    heartbeat_sequence = 1
    heartbeat_age_s = 0.0
    imu_age_s = 0.0
    race_status_age_s = 0.0
    actuator_age_s = 0.0
    latest_telemetry = None
    race_status = None

    def __init__(self):
        self.reset_calls = 0
        self.arm_calls = 0
        self.imu_samples = []
        self.collisions = []
        self.commands = []

    async def reset(self):
        self.reset_calls += 1

    async def arm(self):
        self.arm_calls += 1

    async def disarm(self):
        pass

    async def send_attitude_rate(self, command):
        self.commands.append(command)

    def drain_imu_samples(self):
        samples = self.imu_samples
        self.imu_samples = []
        return samples

    def drain_collisions(self):
        collisions = self.collisions
        self.collisions = []
        return collisions


def test_post_pass_sampling_filters_residue_before_tracker_selection():
    adapter = _FakeAdapter()
    vision = _FakeVision()
    vision.current_snapshot = _vision_snapshot(
        frame_id=101,
        sim_time_ns=1_010,
        received_monotonic_s=1.0,
    )
    runner = VQ2Runner(adapter, vision)
    residue = _detection(75, 0, 518, 360)
    gate1 = _detection(409, 138, 28, 45)
    assert select_primary_gate([residue, gate1]) is residue
    runner.detector = SimpleNamespace(detect=lambda _image: [residue, gate1])
    runner._post_gate_reacquisition = True

    runner._sample()

    assert runner._latest_accepted_target is not None
    assert runner._latest_accepted_target.bbox == gate1.bbox
    assert runner.tracker.consecutive == 1


def test_post_pass_diagnostic_reference_stays_on_exact_processed_frame():
    adapter = _FakeAdapter()
    vision = _FakeVision()
    processed = _vision_snapshot(
        frame_id=101,
        sim_time_ns=1_010,
        received_monotonic_s=1.0,
        generation=3,
    )
    processed.camera_frame.image.fill(17)
    vision.current_snapshot = processed
    runner = VQ2Runner(adapter, vision)
    runner.detector = SimpleNamespace(
        detect=lambda _image: [_detection(409, 138, 28, 45)]
    )
    runner._post_gate_reacquisition = True

    runner._sample()
    newer = _vision_snapshot(
        frame_id=102,
        sim_time_ns=1_020,
        received_monotonic_s=1.03,
        generation=3,
    )
    newer.camera_frame.image.fill(99)
    vision.current_snapshot = newer

    token, retained_image = runner._post_gate_last_frame
    assert token == (3, 101, 1_010)
    assert retained_image is processed.camera_frame.image
    assert int(retained_image[0, 0, 0]) == 17


def test_emergency_reset_is_sent_even_with_no_fresh_baseline(monkeypatch):
    adapter = _FakeAdapter()
    vision = _FakeVision()
    runner = VQ2Runner(adapter, vision)

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(vq2_module, "RESET_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", no_delay)

    proof = asyncio.run(runner.emergency_reset())

    assert proof is None
    assert adapter.reset_calls == 1


def test_invalid_imu_after_bootstrap_latches_estimator_failure():
    adapter = _FakeAdapter()
    vision = _FakeVision()
    runner = VQ2Runner(adapter, vision)
    runner.estimator = ImuAttitudeEstimator(
        ImuAttitudeConfig(
            calibration_min_samples=1,
            calibration_min_duration_s=0.0,
            gravity_correction_kp=0.0,
            gyro_bias_ki=0.0,
        )
    )
    adapter.imu_samples = [
        IMUData(
            timestamp_us=1_000_000,
            accel=(0.0, 0.0, -9.80665),
            gyro=(0.0, 0.0, 0.0),
        )
    ]
    runner._sample()
    assert runner.estimator.is_ready
    assert runner.estimate is not None

    adapter.imu_samples = [
        IMUData(
            timestamp_us=1_010_000,
            accel=(math.nan, 0.0, -9.80665),
            gyro=(0.0, 0.0, 0.0),
        )
    ]
    runner._sample()

    assert runner._last_imu_us == 1_010_000
    assert runner._estimator_unhealthy_latched
    failures = runner._stream_failures(
        require_estimator=True,
        require_target=False,
        require_armed=False,
    )
    assert any("attitude estimator failure latched" in failure for failure in failures)


def test_delayed_pre_reset_clocks_cannot_unlock_go():
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    proof = ResetProof(
        attempt=1,
        pre_race_boot_ms=10_000,
        post_race_boot_ms=500,
        pre_imu_us=10_000_000,
        post_imu_us=500_000,
        advancing_race_samples=3,
        advancing_imu_samples=5,
        countdown_observed=True,
    )
    runner._accept_reset_proof(proof, restart_vision=False)
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1_000_000,
        race_start_boot_time_ms=100,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    adapter.latest_telemetry = TelemetryState(
        timestamp_us=0,
        position_ned=(0.0, 0.0, 0.0),
        velocity_ned=(0.0, 0.0, 0.0),
        orientation=Quaternion(),
        angular_velocity=(0.0, 0.0, 0.0),
        imu=IMUData(
            timestamp_us=1_000_000_000,
            accel=(0.0, 0.0, -9.81),
            gyro=(0.0, 0.0, 0.0),
        ),
    )

    with pytest.raises(SafetyAbort, match="proved reset epoch"):
        asyncio.run(runner.wait_for_go(timeout_s=0.05))

    assert adapter.arm_calls == 0


def test_unproved_reset_path_never_calls_arm(monkeypatch):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())

    async def reset_fails(**_kwargs):
        raise SafetyAbort("ignored reset")

    async def cleanup_succeeds():
        return True

    monkeypatch.setattr(runner, "establish_reset_epoch", reset_fails)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup_succeeds)

    result = asyncio.run(runner.run_powered_stage("sign-id"))

    assert not result.success
    assert "ignored reset" in result.reason
    assert adapter.arm_calls == 0


def test_only_tiny_spawn_pad_contact_is_classified_benign():
    assert is_benign_pad_contact(
        {"id": 1002, "threat_level": 1, "impulse": 0.0025}
    )
    assert not is_benign_pad_contact(
        {"id": 1001, "threat_level": 1, "impulse": 0.0025}
    )
    assert not is_benign_pad_contact(
        {"id": 1002, "threat_level": 2, "impulse": 0.0025}
    )
    assert not is_benign_pad_contact(
        {"id": 1002, "threat_level": 1, "impulse": 0.02}
    )


def test_repeated_tiny_pad_contacts_exceed_cumulative_launch_budget(monkeypatch):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
    monkeypatch.setattr(runner, "_stream_failures", lambda **_kwargs: [])

    adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": 0.004}
        for _ in range(12)
    ]
    runner._watchdog(
        allow_benign_pad_contact=True,
        enforce_benign_pad_budget=True,
    )

    adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": 0.004}
    ]
    with pytest.raises(SafetyAbort, match="repeated pad contacts"):
        runner._watchdog(
            allow_benign_pad_contact=True,
            enforce_benign_pad_budget=True,
        )

    impulse_adapter = _FakeAdapter()
    impulse_runner = VQ2Runner(impulse_adapter, _FakeVision())
    monkeypatch.setattr(impulse_runner, "_stream_failures", lambda **_kwargs: [])
    impulse_adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": 0.009}
        for _ in range(6)
    ]
    with pytest.raises(SafetyAbort, match="repeated pad contacts"):
        impulse_runner._watchdog(
            allow_benign_pad_contact=True,
            enforce_benign_pad_budget=True,
        )

    late_adapter = _FakeAdapter()
    late_runner = VQ2Runner(late_adapter, _FakeVision())
    monkeypatch.setattr(late_runner, "_stream_failures", lambda **_kwargs: [])
    late_adapter.collisions = [
        {"id": 1002, "threat_level": 1, "impulse": 0.001}
    ]
    with pytest.raises(SafetyAbort, match="collision reported"):
        late_runner._watchdog(
            allow_benign_pad_contact=False,
            enforce_benign_pad_budget=True,
        )


@pytest.mark.parametrize(
    ("post_cross_gate_index", "expected_reason"),
    [(1, None), (0, "not credited")],
)
def test_gate0_confirmation_cuts_thrust_then_uses_new_race_packet(
    monkeypatch,
    post_cross_gate_index,
    expected_reason,
):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1000,
        race_start_boot_time_ms=0,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    vision = _FakeVision()
    runner = VQ2Runner(adapter, vision)
    runner.estimate = _estimate()
    runner.tracker.target = vq2_module.GateTarget(
        frame_id=10,
        sim_time_ns=10,
        received_monotonic_s=0.0,
        center_x=336,
        center_y=180,
        bbox=(61, 0, 551, 360),
        confidence=0.8,
    )
    runner.tracker.consecutive = 3
    vision.current_snapshot = _vision_snapshot(
        frame_id=11,
        sim_time_ns=11,
        received_monotonic_s=0.0,
    )

    clock = [0.0]
    sample_count = [0]
    watchdog_target_requirements = []

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(float(seconds), 0.02)

    def fake_sample():
        sample_count[0] += 1
        if sample_count[0] == 7:
            # A fresh contour and a same-timestamp gate-1 value must not leave
            # the latched zero-thrust confirmation phase or prove passage.
            runner.tracker.target = vq2_module.GateTarget(
                frame_id=11,
                sim_time_ns=11,
                received_monotonic_s=clock[0],
                center_x=336,
                center_y=180,
                bbox=(61, 0, 551, 360),
                confidence=0.8,
            )
            adapter.race_status = RaceStatus(
                sim_boot_time_ms=1000,
                race_start_boot_time_ms=0,
                race_finish_time_ns=-1,
                active_gate_index=post_cross_gate_index,
                last_gate_race_time=(123 if post_cross_gate_index == 1 else -1),
            )
        elif sample_count[0] >= 8:
            adapter.race_status = RaceStatus(
                sim_boot_time_ms=1250,
                race_start_boot_time_ms=0,
                race_finish_time_ns=-1,
                active_gate_index=post_cross_gate_index,
                last_gate_race_time=(123 if post_cross_gate_index == 1 else -1),
            )

    def fake_watchdog(**kwargs):
        watchdog_target_requirements.append(kwargs["require_target"])

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", fake_watchdog)

    context = vq2_module.StartContext(
        spawn_roll_rad=0.0,
        spawn_pitch_rad=-0.31,
        initial_gate_x=322,
        initial_gate_y=174,
        initial_gate_area=6480,
        go_boot_ms=1000,
    )
    if expected_reason is None:
        result = asyncio.run(runner._run_gate0(context))
        assert result["gate0_passed"]
        assert result["crossing_confirmation_used"]
    else:
        with pytest.raises(SafetyAbort, match=expected_reason):
            asyncio.run(runner._run_gate0(context))

    first_zero = next(
        index
        for index, command in enumerate(adapter.commands)
        if command.thrust == 0.0
    )
    assert all(
        command.thrust == 0.0
        and command.roll_rate == 0.0
        and command.pitch_rate == 0.0
        and command.yaw_rate == 0.0
        for command in adapter.commands[first_zero:]
    )
    assert False in watchdog_target_requirements


def _configure_gate1_observer(*, clock, crossing_started_s=9.70):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1250,
        race_start_boot_time_ms=0,
        race_finish_time_ns=-1,
        active_gate_index=1,
        last_gate_race_time=123,
    )
    vision = _FakeVision()
    vision.current_snapshot = _vision_snapshot(
        frame_id=100,
        sim_time_ns=1_000,
        received_monotonic_s=9.99,
        generation=4,
    )
    runner = VQ2Runner(adapter, vision)
    runner.estimate = _estimate(roll=0.01, pitch=-0.05)
    runner._last_flight_command = vq2_module.AttitudeRateCommand(
        0.0, 0.0, 0.0, 0.0
    )
    runner._last_flight_command_sent_s = 9.98
    runner._gate0_transition_proof = vq2_module.GateTransitionProof(
        pre_gate_race_boot_ms=1000,
        post_gate_race_boot_ms=1250,
        flight_started_monotonic_s=5.20,
        crossing_started_monotonic_s=crossing_started_s,
        pass_confirmed_monotonic_s=9.99,
        next_control_deadline_s=clock[0],
        vision_generation=4,
        vision_frame_id=100,
        vision_sim_time_ns=1_000,
        vision_received_monotonic_s=9.99,
        pass_rpy_rad=runner.estimate.orientation.to_euler(),
    )
    details = {
        "gate0_passed": True,
        "gate_transition_proved": True,
        "pre_gate_race_boot_ms": 1000,
        "race_boot_ms": 1250,
    }
    return runner, adapter, vision, details


def _publish_observation_frame(runner, *, frame_id, clock, detection):
    accepted = runner.tracker.update(
        ([] if detection is None else [detection]),
        frame_id=frame_id,
        sim_time_ns=frame_id * 10,
        received_monotonic_s=clock[0],
    )
    runner._latest_detection_generation = 4
    runner._latest_detection_frame_id = frame_id
    runner._latest_detection_frame_sim_ns = frame_id * 10
    runner._latest_detection_received_s = clock[0]
    runner._latest_accepted_target = accepted


def test_gate1_observation_requires_three_new_frames_and_sends_only_paced_zero(
    monkeypatch,
):
    clock = [10.0]
    runner, adapter, vision, details = _configure_gate1_observer(clock=clock)
    sample_count = [0]
    command_times = []
    watchdog_require_target = []

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        sample_count[0] += 1
        frame_id = 100 + sample_count[0]
        _publish_observation_frame(
            runner,
            frame_id=frame_id,
            clock=clock,
            detection=_detection(409 + sample_count[0], 138, 28, 45),
        )

    def fake_watchdog(**kwargs):
        watchdog_require_target.append(kwargs["require_target"])

    async def recorded_send(command):
        adapter.commands.append(command)
        command_times.append(clock[0])

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", fake_watchdog)
    monkeypatch.setattr(adapter, "send_attitude_rate", recorded_send)

    result = asyncio.run(runner._observe_gate1(details))

    assert result["gate1_observed"]
    assert result["frame_count"] == 3
    assert [frame["frame_id"] for frame in result["frames"]] == [101, 102, 103]
    assert len(adapter.commands) == 2
    assert all(
        command.roll_rate == command.pitch_rate == command.yaw_rate == command.thrust == 0.0
        for command in adapter.commands
    )
    assert all(
        later - earlier >= vq2_module.CONTROL_PERIOD_S - 1e-12
        for earlier, later in zip(command_times, command_times[1:])
    )
    assert watchdog_require_target == [False] * 6
    assert vision.reset_calls == 0
    assert vision.is_running is False


def test_gate1_observation_resets_streak_after_missing_frame(monkeypatch):
    clock = [10.0]
    runner, _adapter, _vision, details = _configure_gate1_observer(clock=clock)
    sample_count = [0]

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        sample_count[0] += 1
        detection = (
            None
            if sample_count[0] == 2
            else _detection(410 + sample_count[0], 138, 28, 45)
        )
        _publish_observation_frame(
            runner,
            frame_id=100 + sample_count[0],
            clock=clock,
            detection=detection,
        )

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)

    result = asyncio.run(runner._observe_gate1(details))

    assert [frame["frame_id"] for frame in result["frames"]] == [103, 104, 105]


def test_gate1_observation_uses_fixed_nested_deadline_and_never_catches_up(
    monkeypatch,
):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(
        clock=clock,
        crossing_started_s=9.65,
    )
    command_times = []

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        # Repeated control ticks without a new frame can never build a streak.
        return None

    async def recorded_send(command):
        adapter.commands.append(command)
        command_times.append(clock[0])
        if len(command_times) == 1:
            clock[0] += 0.065  # A send stall must drop, not replay, ticks.

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    monkeypatch.setattr(adapter, "send_attitude_rate", recorded_send)

    with pytest.raises(SafetyAbort, match="timed out"):
        asyncio.run(runner._observe_gate1(details))

    # crossing start + 0.40 is the fixed 10.05 hard limit; the injected send
    # completion overrun is observed once and never followed by catch-up sends.
    assert command_times == [10.0]
    assert clock[0] == pytest.approx(10.065)
    assert all(command.thrust == 0.0 for command in adapter.commands)


def test_gate1_observation_cannot_accept_third_frame_after_hard_deadline(
    monkeypatch,
):
    clock = [10.0]
    runner, _adapter, _vision, details = _configure_gate1_observer(clock=clock)
    sample_count = [0]

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        sample_count[0] += 1
        if sample_count[0] == 3:
            clock[0] += 0.09  # detector stall crosses the fixed 10.10 deadline
        _publish_observation_frame(
            runner,
            frame_id=100 + sample_count[0],
            clock=clock,
            detection=_detection(409 + sample_count[0], 138, 28, 45),
        )

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)

    with pytest.raises(SafetyAbort, match="timed out"):
        asyncio.run(runner._observe_gate1(details))

    assert sample_count[0] == 3
    assert runner.tracker.consecutive == 3
    assert clock[0] > 10.10


def test_gate1_observation_rechecks_deadline_after_diagnostic_logging(monkeypatch):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(clock=clock)
    sample_count = [0]

    class AdvancingRecorder:
        path = None

        def emit(self, event, **fields):
            if event == "post_gate_candidate_frame" and fields["tracker_streak"] == 3:
                clock[0] += 0.09

        def save_png(self, _label, _image):
            raise AssertionError("no image should be encoded while observing")

    runner.recorder = AdvancingRecorder()

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        sample_count[0] += 1
        _publish_observation_frame(
            runner,
            frame_id=100 + sample_count[0],
            clock=clock,
            detection=_detection(409 + sample_count[0], 138, 28, 45),
        )

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)

    with pytest.raises(SafetyAbort, match="timed out"):
        asyncio.run(runner._observe_gate1(details))

    assert sample_count[0] == 3
    assert len(adapter.commands) == 2
    assert clock[0] > 10.10


def test_gate1_observation_rechecks_watchdog_after_logging_stall(monkeypatch):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(clock=clock)
    watchdog_count = [0]

    class AdvancingRecorder:
        path = None

        def emit(self, event, **_fields):
            if event == "post_gate_candidate_frame":
                clock[0] += 0.06

        def save_png(self, _label, _image):
            raise AssertionError("no image should be encoded while observing")

    runner.recorder = AdvancingRecorder()

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        _publish_observation_frame(
            runner,
            frame_id=101,
            clock=clock,
            detection=_detection(410, 138, 28, 45),
        )

    def fake_watchdog(**_kwargs):
        watchdog_count[0] += 1
        if watchdog_count[0] == 2:
            raise SafetyAbort("IMU timestamp not advancing")

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", fake_watchdog)

    with pytest.raises(SafetyAbort, match="IMU timestamp not advancing"):
        asyncio.run(runner._observe_gate1(details))

    assert watchdog_count[0] == 2
    assert clock[0] == pytest.approx(10.06)
    assert adapter.commands == []


def test_gate1_observation_rechecks_deadline_after_final_watchdog(monkeypatch):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(clock=clock)
    sample_count = [0]
    watchdog_count = [0]

    def fake_monotonic():
        return clock[0]

    async def fake_sleep(seconds):
        clock[0] += max(0.0, float(seconds))

    def fake_sample():
        sample_count[0] += 1
        _publish_observation_frame(
            runner,
            frame_id=100 + sample_count[0],
            clock=clock,
            detection=_detection(409 + sample_count[0], 138, 28, 45),
        )

    def fake_watchdog(**_kwargs):
        watchdog_count[0] += 1
        if watchdog_count[0] == 6:
            clock[0] += 0.09

    monkeypatch.setattr(vq2_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", fake_watchdog)

    with pytest.raises(SafetyAbort, match="timed out"):
        asyncio.run(runner._observe_gate1(details))

    assert watchdog_count[0] == 6
    assert len(adapter.commands) == 2


@pytest.mark.parametrize("gate_index", [0, 2])
def test_gate1_observation_aborts_on_any_gate_index_change_before_send(
    monkeypatch,
    gate_index,
):
    clock = [10.0]
    runner, adapter, _vision, details = _configure_gate1_observer(clock=clock)

    def fake_sample():
        adapter.race_status = RaceStatus(
            sim_boot_time_ms=1251,
            race_start_boot_time_ms=0,
            race_finish_time_ns=-1,
            active_gate_index=gate_index,
            last_gate_race_time=123,
        )

    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vq2_module.asyncio, "sleep", lambda _seconds: asyncio.sleep(0))
    monkeypatch.setattr(runner, "_sample", fake_sample)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)

    with pytest.raises(SafetyAbort, match="gate index changed"):
        asyncio.run(runner._observe_gate1(details))

    assert adapter.commands == []


def test_gate0_early_gate1_status_requires_strictly_newer_packet(monkeypatch):
    adapter = _FakeAdapter()
    adapter.is_armed = True
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1000,
        race_start_boot_time_ms=0,
        race_finish_time_ns=-1,
        active_gate_index=1,
        last_gate_race_time=123,
    )
    runner = VQ2Runner(adapter, _FakeVision())
    runner.estimate = _estimate()
    runner.tracker.target = vq2_module.GateTarget(
        frame_id=1,
        sim_time_ns=1,
        received_monotonic_s=0.0,
        center_x=322,
        center_y=174,
        bbox=(282, 134, 80, 80),
        confidence=0.8,
    )
    runner.tracker.consecutive = 3
    monkeypatch.setattr(vq2_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(runner, "_sample", lambda: None)
    monkeypatch.setattr(runner, "_watchdog", lambda **_kwargs: None)
    context = vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    with pytest.raises(SafetyAbort, match="not strictly newer"):
        asyncio.run(runner._run_gate0(context))

    assert adapter.commands == []


def test_gate0_observe_dispatch_preserves_credited_gate0_on_observation_abort(
    monkeypatch,
):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(
        sim_boot_time_ms=1000,
        race_start_boot_time_ms=0,
        race_finish_time_ns=-1,
        active_gate_index=0,
        last_gate_race_time=-1,
    )
    runner = VQ2Runner(adapter, _FakeVision())
    calls = []
    gate0_details = {
        "gate0_passed": True,
        "gate_transition_proved": True,
        "pre_gate_race_boot_ms": 1000,
        "race_boot_ms": 1250,
    }

    async def establish(**_kwargs):
        calls.append("reset")

    async def normalize():
        calls.append("normalize")

    async def wait_for_go():
        calls.append("go")
        return vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    async def arm():
        calls.append("arm")

    async def gate0(_context, *, capture_transition=False):
        calls.append(("gate0", capture_transition))
        return gate0_details

    async def observe(details):
        calls.append(("observe", details is gate0_details))
        raise SafetyAbort("diagnostic observation timeout")

    async def cleanup():
        calls.append("cleanup")
        return True

    monkeypatch.setattr(runner, "establish_reset_epoch", establish)
    monkeypatch.setattr(runner, "normalize_disarmed", normalize)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(runner, "arm_confirmed", arm)
    monkeypatch.setattr(runner, "_run_gate0", gate0)
    monkeypatch.setattr(runner, "_observe_gate1", observe)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(runner.run_powered_stage("gate0-observe"))

    assert not result.success
    assert result.cleanup_confirmed
    assert result.details["gate0"] == gate0_details
    assert result.details["gate1_observation"] == {
        "gate1_observed": False,
        "reason": "diagnostic observation timeout",
    }
    assert calls == [
        "reset",
        "normalize",
        "go",
        "arm",
        ("gate0", True),
        ("observe", True),
        "cleanup",
    ]


def test_gate0_stage_does_not_enter_post_pass_observation(monkeypatch):
    adapter = _FakeAdapter()
    adapter.race_status = RaceStatus(1000, 0, -1, 0, -1)
    runner = VQ2Runner(adapter, _FakeVision())

    async def no_op(*_args, **_kwargs):
        return None

    async def wait_for_go():
        return vq2_module.StartContext(0.0, -0.31, 322, 174, 6400, 1000)

    async def gate0(_context, *, capture_transition=False):
        assert not capture_transition
        return {"gate0_passed": True}

    async def observe(_details):
        raise AssertionError("plain gate0 must not observe gate 1")

    async def cleanup():
        return True

    monkeypatch.setattr(runner, "establish_reset_epoch", no_op)
    monkeypatch.setattr(runner, "normalize_disarmed", no_op)
    monkeypatch.setattr(runner, "wait_for_go", wait_for_go)
    monkeypatch.setattr(runner, "arm_confirmed", no_op)
    monkeypatch.setattr(runner, "_run_gate0", gate0)
    monkeypatch.setattr(runner, "_observe_gate1", observe)
    monkeypatch.setattr(runner, "safe_cleanup", cleanup)

    result = asyncio.run(runner.run_powered_stage("gate0"))

    assert result.success
    assert result.details == {"gate0_passed": True}


def test_diagnostic_png_encoding_is_deferred_until_explicit_flush(tmp_path):
    recorder = vq2_module.JsonlRecorder(str(tmp_path / "trace.jsonl.gz"))
    vision = _FakeVision()
    vision.current_snapshot = _vision_snapshot()
    runner = VQ2Runner(_FakeAdapter(), vision, recorder=recorder)

    metadata = runner._defer_snapshot("gate1 acquired")

    assert metadata is not None
    assert list(tmp_path.glob("*.png")) == []
    paths, errors = runner._flush_deferred_snapshots()
    recorder.close()

    assert errors == []
    assert len(paths) == 1
    assert (tmp_path / "trace_gate1_acquired.png").is_file()
