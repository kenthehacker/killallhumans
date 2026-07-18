"""Pure safety/geometry tests for the staged VQ2 runner."""

from __future__ import annotations

import asyncio
import math

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
    is_benign_pad_contact,
    next_control_deadline,
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
    is_running = False

    def stop(self):
        self.is_running = False

    def start(self):
        self.is_running = True

    def reset(self):
        pass

    def snapshot(self, **_kwargs):
        return None


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

    async def reset(self):
        self.reset_calls += 1

    async def arm(self):
        self.arm_calls += 1

    async def disarm(self):
        pass

    async def send_attitude_rate(self, _command):
        pass

    def drain_imu_samples(self):
        samples = self.imu_samples
        self.imu_samples = []
        return samples

    def drain_collisions(self):
        return []


def test_emergency_reset_is_sent_even_with_no_fresh_baseline(monkeypatch):
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(vq2_module, "RESET_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(vq2_module.asyncio, "sleep", no_delay)

    proof = asyncio.run(runner.emergency_reset())

    assert proof is None
    assert adapter.reset_calls == 1


def test_invalid_imu_after_bootstrap_latches_estimator_failure():
    adapter = _FakeAdapter()
    runner = VQ2Runner(adapter, _FakeVision())
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
