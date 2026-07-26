from __future__ import annotations

import math

import pytest

from competition.vq2_contracts import FrameEdge
from planning.vq2_dynamic_course import (
    AppliedCommandSample,
    CommandGovernor,
    CommandGovernorConfig,
    DynamicCourseCommand,
    DynamicCourseConfig,
    DynamicCourseCore,
    DynamicCourseError,
    GateObservation,
    ImuAttitudeSample,
    MAX_YAW_RATE_RAD_S,
    SUPPORT_THRUST,
)


NS = 1_000_000_000


def _yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    return (
        math.cos(yaw_rad / 2.0),
        0.0,
        0.0,
        math.sin(yaw_rad / 2.0),
    )


def _imu(
    core: DynamicCourseCore,
    time_s: float,
    *,
    yaw_rad: float = 0.0,
    yaw_rate_rad_s: float = 0.0,
) -> None:
    core.record_imu(
        ImuAttitudeSample(
            monotonic_ns=round(time_s * NS),
            body_to_reference_wxyz=_yaw_quaternion(yaw_rad),
            body_rates_rad_s=(0.0, 0.0, yaw_rate_rad_s),
            attitude_uncertainty_rad=0.001,
            source_timestamp_us=round(time_s * 1_000_000),
        )
    )


def _observation(
    track_id: str,
    sequence: int,
    time_s: float,
    *,
    x: float = 0.0,
    y: float = 0.0,
    log_scale: float = 0.0,
    aperture: tuple[float, float] = (0.42, 0.34),
    clipping: FrameEdge = FrameEdge.NONE,
    visible: bool = True,
) -> GateObservation:
    timestamp = round(time_s * NS)
    return GateObservation(
        track_id=track_id,
        frame_sequence=sequence,
        observation_monotonic_ns=timestamp,
        capture_monotonic_ns=timestamp,
        timing_basis="test-exact-host-capture",
        timing_uncertainty_s=0.001,
        center_norm=(x, y) if visible else None,
        log_scale=log_scale if visible else None,
        aperture_half_size_norm=aperture if visible else None,
        clipping=clipping,
        visible=visible,
        confidence=0.95,
        measurement_std=(0.005, 0.005, 0.01),
    )


def _command(
    time_s: float,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    thrust: float = SUPPORT_THRUST,
) -> AppliedCommandSample:
    return AppliedCommandSample(
        monotonic_ns=round(time_s * NS),
        target_roll_rad=roll,
        target_pitch_rad=pitch,
        yaw_rate_rad_s=yaw,
        thrust=thrust,
    )


def _commit_decision(
    core: DynamicCourseCore,
    decision_time_s: float,
    command: DynamicCourseCommand,
) -> None:
    core.record_applied_command(
        _command(
            decision_time_s,
            roll=command.target_roll_rad,
            pitch=command.target_pitch_rad,
            yaw=command.yaw_rate_rad_s,
            thrust=command.thrust,
        )
    )


def test_attitude_rotation_alone_derotates_to_zero_translation() -> None:
    config = DynamicCourseConfig(
        camera_delay_s=0.0,
        horizontal_angle_scale_rad=1.59,
        vertical_angle_scale_rad=1.10,
    )
    core = DynamicCourseCore(config)
    _imu(core, 1.0, yaw_rad=0.0)
    first = core.observe_track(_observation("gate-a", 1, 1.0))

    yaw = 0.18
    _imu(core, 1.1, yaw_rad=yaw, yaw_rate_rad_s=1.8)
    second = core.observe_track(
        _observation(
            "gate-a",
            2,
            1.1,
            x=math.tan(-yaw) / config.horizontal_angle_scale_rad,
        )
    )

    assert first.bearing_rad == pytest.approx((0.0, 0.0), abs=1e-12)
    assert second.bearing_rad == pytest.approx((0.0, 0.0), abs=2e-10)
    assert second.bearing_rate_rad_s == pytest.approx((0.0, 0.0), abs=2e-9)
    assert second.predicted_rotational_rate_rad_s[0] == pytest.approx(-1.8)
    assert second.residual_translational_rate_rad_s == pytest.approx(
        (0.0, 0.0),
        abs=2e-9,
    )


def test_delayed_command_history_is_right_continuous_at_channel_delays() -> None:
    core = DynamicCourseCore(
        DynamicCourseConfig(
            camera_delay_s=0.0,
            roll_command_delay_s=0.080,
            yaw_command_delay_s=0.040,
        )
    )
    core.record_applied_command(_command(1.0))
    core.record_applied_command(_command(1.1, roll=0.12, yaw=-0.15))

    just_before_roll = core.delayed_command_view(1_179_999_999)
    at_roll = core.delayed_command_view(1_180_000_000)
    just_before_yaw = core.delayed_command_view(1_139_999_999)
    at_yaw = core.delayed_command_view(1_140_000_000)

    assert just_before_roll.target_roll_rad == 0.0
    assert just_before_roll.roll_source_monotonic_ns == 1_000_000_000
    assert at_roll.target_roll_rad == pytest.approx(0.12)
    assert at_roll.roll_source_monotonic_ns == 1_100_000_000
    assert just_before_yaw.yaw_rate_rad_s == 0.0
    assert at_yaw.yaw_rate_rad_s == pytest.approx(-0.15)
    assert at_yaw.yaw_source_monotonic_ns == 1_100_000_000


def test_applied_command_accepts_existing_live_spawn_pitch_envelope() -> None:
    sample = AppliedCommandSample(
        monotonic_ns=1,
        target_roll_rad=0.0,
        target_pitch_rad=-0.31,
        yaw_rate_rad_s=0.0,
        thrust=0.26,
    )

    assert sample.target_pitch_rad == -0.31


def test_noisy_alternating_detections_do_not_reverse_roll_each_frame() -> None:
    config = DynamicCourseConfig(
        camera_delay_s=0.0,
        roll_guidance_sign=1.0,
        roll_gain=0.65,
        bearing_alpha=0.95,
        bearing_beta=0.25,
        governor=CommandGovernorConfig(
            max_roll_slew_rad_s=0.35,
            max_roll_accel_rad_s2=1.2,
        ),
    )
    core = DynamicCourseCore(config)
    core.record_applied_command(_command(0.90))
    outputs: list[float] = []
    decision_times: list[float] = []
    for index, image_x in enumerate((0.50, -0.50, 0.50, -0.50, 0.50, -0.50)):
        observation_time = 1.0 + index * 0.040
        decision_time = observation_time + 0.010
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                index + 1,
                observation_time,
                x=image_x,
            )
        )
        if index == 0:
            core.bind(
                current_gate_index=0,
                current_track_id="gate-a",
                successor_track_id=None,
            )
        _imu(core, decision_time)
        decision = core.guide(round(decision_time * NS))
        outputs.append(decision.command.target_roll_rad)
        decision_times.append(decision_time)
        _commit_decision(core, decision_time, decision.command)

    assert any(abs(value) > 1e-5 for value in outputs)
    assert all(
        left * right >= 0.0
        for left, right in zip(outputs, outputs[1:])
        if abs(left) > 1e-9 and abs(right) > 1e-9
    )
    for index in range(1, len(outputs)):
        dt = decision_times[index] - decision_times[index - 1]
        assert abs(outputs[index] - outputs[index - 1]) <= (
            config.governor.max_roll_slew_rad_s * dt + 1e-12
        )


def test_successor_steering_and_state_are_continuous_through_promotion() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    _imu(core, 1.0)
    core.observe_track(_observation("gate-a", 1, 1.0, x=0.00, log_scale=0.20))
    successor_before = core.observe_track(
        _observation("gate-b", 1, 1.0, x=0.42, log_scale=-0.15)
    )
    core.observe_track(_observation("gate-c", 1, 1.0, x=0.68, log_scale=-0.40))
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )

    _imu(core, 1.01)
    precredit = core.guide(1_010_000_000)
    assert precredit.successor_weight > 0.0
    assert precredit.command.yaw_rate_rad_s < 0.0
    _commit_decision(core, 1.01, precredit.command)

    promoted = core.promote_authoritative(
        from_gate_index=0,
        to_gate_index=1,
        promoted_track_id="gate-b",
        next_successor_track_id="gate-c",
        monotonic_ns=1_015_000_000,
    )
    assert promoted.current == successor_before
    assert promoted.last_governed_command == precredit.command

    _imu(core, 1.04)
    postcredit = core.guide(1_040_000_000)
    assert postcredit.current_track_id == "gate-b"
    assert postcredit.successor_track_id == "gate-c"
    assert postcredit.command.yaw_rate_rad_s < 0.0
    assert postcredit.command.yaw_rate_rad_s != 0.0
    assert abs(
        postcredit.command.yaw_rate_rad_s - precredit.command.yaw_rate_rad_s
    ) <= core.config.governor.max_yaw_slew_rad_s2 * 0.030 + 1e-12


def test_off_axis_successor_and_rapid_closure_brake_before_advancing() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90, pitch=-0.02))
    _imu(core, 1.0)
    core.observe_track(_observation("gate-a", 1, 1.0, log_scale=0.00))
    _imu(core, 1.1)
    current = core.observe_track(
        _observation("gate-a", 2, 1.1, x=0.08, log_scale=0.25)
    )
    core.observe_track(
        _observation("gate-b", 1, 1.1, x=0.62, log_scale=-0.30)
    )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    _imu(core, 1.11)
    decision = core.guide(1_110_000_000)

    assert current.expansion_rate_s >= core.config.rapid_expansion_rate_s
    assert current.time_to_contact_s is not None
    assert decision.braking is True
    assert decision.brake_reason == "off_axis_rapid_closure"
    assert decision.proposed_command.target_pitch_rad > 0.0
    assert decision.command.target_pitch_rad > -0.02
    assert decision.command.thrust == pytest.approx(SUPPORT_THRUST)


def test_successor_heading_cannot_reverse_roll_away_from_passage_intercept() -> None:
    core = DynamicCourseCore(
        DynamicCourseConfig(
            camera_delay_s=0.0,
            roll_guidance_sign=1.0,
            roll_gain=0.18,
            lateral_rate_gain=0.045,
        )
    )
    core.record_applied_command(_command(0.90))
    _imu(core, 1.0)
    core.observe_track(_observation("gate-a", 1, 1.0, x=0.40))
    core.observe_track(_observation("gate-b", 1, 1.0, x=-1.40))
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    _imu(core, 1.01)

    decision = core.guide(1_010_000_000)

    assert decision.predicted_successor_bearing_rad is not None
    assert decision.predicted_successor_bearing_rad[0] < 0.0
    assert decision.passage_error_norm[0] > 0.0
    assert decision.proposed_command.target_roll_rad > 0.0


def test_generic_authoritative_lifecycle_continues_past_gate_one() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    _imu(core, 1.0)
    preserved = {}
    for index, track_id in enumerate(("gate-a", "gate-b", "gate-c", "gate-d")):
        preserved[track_id] = core.observe_track(
            _observation(
                track_id,
                1,
                1.0,
                x=index * 0.12,
                log_scale=-index * 0.1,
            )
        )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )

    first = core.promote_authoritative(
        from_gate_index=0,
        to_gate_index=1,
        promoted_track_id="gate-b",
        next_successor_track_id="gate-c",
        monotonic_ns=1_010_000_000,
    )
    second = core.promote_authoritative(
        from_gate_index=1,
        to_gate_index=2,
        promoted_track_id="gate-c",
        next_successor_track_id="gate-d",
        monotonic_ns=1_020_000_000,
    )
    third = core.promote_authoritative(
        from_gate_index=2,
        to_gate_index=3,
        promoted_track_id="gate-d",
        next_successor_track_id=None,
        monotonic_ns=1_030_000_000,
    )

    assert first.current == preserved["gate-b"]
    assert second.current == preserved["gate-c"]
    assert third.current == preserved["gate-d"]
    assert third.current_gate_index == 3
    assert third.current_track_id == "gate-d"
    assert third.successor_track_id is None
    assert third.promotion_count == 3


def test_current_and_successor_references_project_into_one_decision_frame() -> None:
    config = DynamicCourseConfig(camera_delay_s=0.0)
    core = DynamicCourseCore(config)
    core.record_applied_command(_command(0.80))
    _imu(core, 1.0, yaw_rad=0.0)
    core.observe_track(_observation("gate-a", 1, 1.0, x=0.0))

    # B is first observed after a +0.20 rad body yaw.  A world bearing of
    # +0.40 rad therefore appears at +0.20 rad in this later camera.
    _imu(core, 1.1, yaw_rad=0.20)
    core.observe_track(
        _observation(
            "gate-b",
            1,
            1.1,
            x=math.tan(0.20) / config.horizontal_angle_scale_rad,
        )
    )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    _imu(core, 1.2, yaw_rad=0.10)
    decision = core.guide(1_200_000_000)

    assert decision.predicted_successor_bearing_rad is not None
    assert decision.predicted_successor_bearing_rad[0] == pytest.approx(
        0.30,
        abs=2e-8,
    )
    assert decision.passage_point_norm[0] > 0.0


def test_governor_preview_does_not_consume_budget_and_sustained_reversal_crosses_zero() -> None:
    governor = CommandGovernor(
        CommandGovernorConfig(
            max_roll_slew_rad_s=0.40,
            max_roll_accel_rad_s2=4.0,
        )
    )
    neutral = DynamicCourseCommand(0.0, 0.0, 0.0, SUPPORT_THRUST)
    governor.commit(neutral, 1_000_000_000)
    positive = DynamicCourseCommand(0.16, 0.0, 0.0, SUPPORT_THRUST)
    first = governor.preview(positive, 1_031_000_000)
    duplicate = governor.preview(positive, 1_031_000_000)
    assert first == duplicate
    assert governor.last_command == neutral
    governor.commit(first, 1_031_000_000)

    negative = DynamicCourseCommand(-0.16, 0.0, 0.0, SUPPORT_THRUST)
    values = [first.target_roll_rad]
    for step in range(2, 20):
        timestamp = 1_000_000_000 + step * 31_000_000
        command = governor.preview(negative, timestamp)
        values.append(command.target_roll_rad)
        governor.commit(command, timestamp)

    assert any(value == pytest.approx(0.0, abs=1e-12) for value in values)
    assert values[-1] < 0.0
    assert all(
        left * right >= 0.0
        for left, right in zip(values, values[1:])
        if abs(left) > 1e-12 and abs(right) > 1e-12
    )
    assert max(abs(value) for value in values) <= 0.16


def test_governor_does_not_extrapolate_an_overridden_thrust_step() -> None:
    governor = CommandGovernor()
    first_ns = NS
    governor.commit(
        DynamicCourseCommand(0.0, 0.0, 0.0, 0.26),
        first_ns,
    )
    governor.commit(
        DynamicCourseCommand(0.0, 0.0, 0.0, 0.32),
        first_ns + 30_000_000,
        discontinuity_axes=(3,),
    )

    resumed = governor.preview(
        DynamicCourseCommand(0.0, 0.0, 0.0, SUPPORT_THRUST),
        first_ns + 60_000_000,
    )

    assert SUPPORT_THRUST <= resumed.thrust <= 0.32


def test_governor_projects_slew_history_into_the_command_envelope() -> None:
    governor = CommandGovernor()
    first_ns = NS
    governor.commit(
        DynamicCourseCommand(0.0, 0.0, -0.138, SUPPORT_THRUST),
        first_ns,
    )
    governor.commit(
        DynamicCourseCommand(0.0, 0.0, -0.148, SUPPORT_THRUST),
        first_ns + 30_000_000,
    )

    saturated = governor.preview(
        DynamicCourseCommand(0.0, 0.0, -0.140, SUPPORT_THRUST),
        first_ns + 60_000_000,
    )

    assert -MAX_YAW_RATE_RAD_S <= saturated.yaw_rate_rad_s <= 0.0


def test_excess_capture_timing_uncertainty_withholds_derotation() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    _imu(core, 1.0)
    observation = GateObservation(
        track_id="gate-a",
        frame_sequence=1,
        observation_monotonic_ns=1_000_000_000,
        capture_monotonic_ns=1_000_000_000,
        timing_basis="uncertain-proxy",
        timing_uncertainty_s=0.080,
        center_norm=(0.0, 0.0),
        log_scale=0.0,
        aperture_half_size_norm=(0.4, 0.3),
    )

    with pytest.raises(DynamicCourseError, match="timing uncertainty"):
        core.observe_track(observation)


def test_clipped_axis_coasts_without_false_inward_update() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    _imu(core, 1.0)
    first = core.observe_track(_observation("gate-a", 1, 1.0, x=0.30))
    _imu(core, 1.1)
    clipped = core.observe_track(
        _observation(
            "gate-a",
            2,
            1.1,
            x=-0.50,
            log_scale=0.40,
            clipping=FrameEdge.RIGHT,
        )
    )

    assert clipped.bearing_rad[0] == pytest.approx(first.bearing_rad[0])
    assert clipped.log_scale == pytest.approx(first.log_scale)
    assert clipped.bearing_std_rad[0] > first.bearing_std_rad[0]
    assert clipped.censored_axes == (True, False)
    assert clipped.time_to_contact_s is None


def test_measured_yaw_output_never_exceeds_capability_envelope() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.9))
    _imu(core, 1.0)
    core.observe_track(_observation("gate-a", 1, 1.0, x=1.0))
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id=None,
    )
    _imu(core, 1.01)
    decision = core.guide(1_010_000_000)

    assert abs(decision.proposed_command.yaw_rate_rad_s) == MAX_YAW_RATE_RAD_S
    assert abs(decision.command.yaw_rate_rad_s) <= MAX_YAW_RATE_RAD_S
