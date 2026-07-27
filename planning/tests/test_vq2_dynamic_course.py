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
    predict_aperture_relative_crossing,
)


NS = 1_000_000_000


def test_aperture_quotient_rate_removes_pure_axial_expansion() -> None:
    prediction = predict_aperture_relative_crossing(
        center_offset_norm=(0.20, -0.30),
        passage_offset_norm=(0.0, 0.0),
        aperture_half_extent_norm=(0.40, 0.60),
        center_rate_norm_s=(0.40, -0.60),
        aperture_expansion_rate_s=(2.0, 2.0),
        center_std_norm=(0.001, 0.001),
        aperture_log_scale_std=0.0,
        capture_timing_uncertainty_s=0.0,
        horizon_s=0.25,
        allowance_q=(0.50, 0.45),
    )

    assert prediction.current_error_q == pytest.approx((0.50, -0.50))
    assert prediction.rate_q_s == pytest.approx((0.0, 0.0), abs=1e-12)
    assert prediction.predicted_error_q == pytest.approx(
        prediction.current_error_q
    )


def test_aperture_quotient_rate_does_not_expand_fixed_passage_bias() -> None:
    unbiased = predict_aperture_relative_crossing(
        center_offset_norm=(0.10, 0.0),
        passage_offset_norm=(0.0, 0.0),
        aperture_half_extent_norm=(0.50, 0.50),
        center_rate_norm_s=(0.0, 0.0),
        aperture_expansion_rate_s=(1.0, 1.0),
        center_std_norm=(0.001, 0.001),
        aperture_log_scale_std=0.0,
        capture_timing_uncertainty_s=0.0,
        horizon_s=0.20,
        allowance_q=(0.50, 0.45),
    )
    biased = predict_aperture_relative_crossing(
        center_offset_norm=(0.10, 0.0),
        passage_offset_norm=(0.10, 0.0),
        aperture_half_extent_norm=(0.50, 0.50),
        center_rate_norm_s=(0.0, 0.0),
        aperture_expansion_rate_s=(1.0, 1.0),
        center_std_norm=(0.001, 0.001),
        aperture_log_scale_std=0.0,
        capture_timing_uncertainty_s=0.0,
        horizon_s=0.20,
        allowance_q=(0.50, 0.45),
    )

    assert biased.rate_q_s == pytest.approx(unbiased.rate_q_s)
    assert biased.current_error_q[0] - unbiased.current_error_q[0] == (
        pytest.approx(0.20)
    )


def test_ea6335c3_top_hit_is_rejected_by_swept_q_envelope() -> None:
    """The exact last clean state may not reproduce its false clearance."""

    center = (0.06616391217790557, -0.567900045493448)
    aperture = (0.5795765774353157, 0.9644074109630432)
    center_rate = (-0.1698399706228867, 0.5322577589123005)
    horizon = 0.5656508691934667
    prediction = predict_aperture_relative_crossing(
        center_offset_norm=center,
        passage_offset_norm=(0.0, 0.0),
        aperture_half_extent_norm=aperture,
        center_rate_norm_s=center_rate,
        aperture_expansion_rate_s=(
            1.767875141990965,
            1.767875141990965,
        ),
        center_std_norm=(0.01653, 0.02170),
        aperture_log_scale_std=0.046094,
        capture_timing_uncertainty_s=0.020,
        horizon_s=horizon,
        allowance_q=(0.50, 0.45),
    )
    frozen_aperture_vertical = center[1] + center_rate[1] * horizon
    falsified_old_clearance = (
        0.8744073625800745
        - abs(frozen_aperture_vertical)
        - 2.0 * 0.032343626640302645
    )

    assert prediction.current_error_q == pytest.approx(
        (0.114159, -0.588859),
        abs=2e-6,
    )
    assert prediction.rate_q_s == pytest.approx(
        (-0.49486, 1.59293),
        abs=2e-5,
    )
    assert prediction.predicted_error_q[1] == pytest.approx(
        0.31218,
        abs=2e-5,
    )
    assert frozen_aperture_vertical < 0.0
    assert falsified_old_clearance == pytest.approx(
        0.5428922415214223,
        abs=2e-7,
    )
    assert falsified_old_clearance > 0.0
    assert prediction.swept_occupancy_q[1] > 0.63
    assert prediction.clearance_q[1] < 0.0


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
    successor_before = None
    for sequence in range(1, 7):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=0.00,
                log_scale=0.05 + sequence * 0.05,
            )
        )
        successor_before = core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=0.12,
                log_scale=-0.15,
            )
        )
        core.observe_track(
            _observation(
                "gate-c",
                sequence,
                observation_time,
                x=0.24,
                log_scale=-0.40,
            )
        )
    assert successor_before is not None
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )

    _imu(core, 1.21)
    precredit = core.guide(1_210_000_000)
    assert precredit.successor_weight > 0.0
    assert precredit.command.yaw_rate_rad_s < 0.0
    _commit_decision(core, 1.21, precredit.command)

    promoted = core.promote_authoritative(
        from_gate_index=0,
        to_gate_index=1,
        promoted_track_id="gate-b",
        next_successor_track_id="gate-c",
        monotonic_ns=1_215_000_000,
    )
    assert promoted.current == successor_before
    assert promoted.last_governed_command == precredit.command

    _imu(core, 1.24)
    postcredit = core.guide(1_240_000_000)
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


def test_trace_3ff977f_successor_flips_cannot_hunt_current_gate_yaw() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    anchors = (
        (1.00, 0.006, 0.322, 0.000, -1.789),
        (1.32, -0.034, 0.291, -0.050, -1.755),
        (1.57, -0.097, 0.238, -0.109, -1.690),
        (1.82, -0.116, 0.228, -0.118, -1.669),
        (2.07, -0.059, 0.297, -0.054, -1.496),
        (2.13, -0.041, 0.322, -0.033, -1.462),
        (2.38, 0.041, 0.444, 0.056, -1.313),
        (2.63, 0.156, 0.647, 0.166, -0.971),
        (2.85, 0.256, 0.834, 0.239, -0.579),
    )
    def interpolate(
        observation_time: float,
    ) -> tuple[float, float, float, float, float]:
        for left, right in zip(anchors, anchors[1:]):
            if observation_time <= right[0]:
                fraction = (
                    observation_time - left[0]
                ) / (right[0] - left[0])
                return (
                    observation_time,
                    *(
                        left[index]
                        + fraction * (right[index] - left[index])
                        for index in range(1, 5)
                    ),
                )
        return anchors[-1]

    samples = tuple(
        interpolate(
            anchors[0][0]
            + sample_index
            * (anchors[-1][0] - anchors[0][0])
            / 64.0
        )
        for sample_index in range(65)
    )
    decisions = []
    for sequence, (
        observation_time,
        current_x,
        successor_x,
        yaw,
        log_scale,
    ) in enumerate(samples, 1):
        _imu(core, observation_time, yaw_rad=yaw)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=current_x,
                log_scale=log_scale,
            )
        )
        core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=successor_x,
                log_scale=log_scale - 0.35,
            )
        )
        if sequence == 1:
            core.bind(
                current_gate_index=0,
                current_track_id="gate-a",
                successor_track_id="gate-b",
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time, yaw_rad=yaw)
        decision = core.guide(round(decision_time * NS))
        decisions.append(decision)
        _commit_decision(core, decision_time, decision.command)

    assert all(
        decision.predicted_successor_bearing_rad is not None
        and decision.predicted_successor_bearing_rad[0] > 0.0
        for decision in decisions
    )
    assert all(
        decision.successor_prediction_horizon_s
        <= core.config.successor_prediction_max_horizon_s
        for decision in decisions
    )
    legacy_predictions = (0.47, -0.77, 1.08, -1.38)
    assert sum(
        left * right < 0.0
        for left, right in zip(
            legacy_predictions,
            legacy_predictions[1:],
        )
    ) == 3
    active = [
        decision
        for decision in decisions
        if abs(decision.successor_yaw_contribution_rad) > 1e-8
    ]
    assert len(active) >= 8
    assert any(decision.current_yaw_release == 0.0 for decision in active)
    assert all(
        0.0 < decision.successor_yaw_contribution_rad
        <= core.config.successor_max_yaw_contribution_rad
        for decision in active
    )
    assert all(
        left.successor_yaw_contribution_rad
        * right.successor_yaw_contribution_rad
        >= 0.0
        for left, right in zip(active, active[1:])
    )
    for decision in active:
        camera_bearing = math.atan(
            decision.camera_current_center_norm[0]
            * core.config.horizontal_angle_scale_rad
        )
        current_heading = (
            (1.0 - decision.current_yaw_release) * camera_bearing
        )
        combined_heading = (
            current_heading
            + decision.successor_yaw_contribution_rad
        )
        # Before near-plane release, even a coherent successor can only damp
        # current-gate recentering.  It cannot reverse or saturate yaw.
        if decision.current_yaw_release < 1.0:
            assert combined_heading * current_heading >= 0.0
            assert abs(combined_heading) <= abs(current_heading) + 1e-12
        assert (
            abs(decision.proposed_command.yaw_rate_rad_s)
            <= MAX_YAW_RATE_RAD_S
        )
    # Once the near-plane scale is reliable, the stale body-camera centering
    # term is released rather than reinstating the observed low-frequency hunt.
    assert decisions[-1].current_yaw_release > 0.65
    assert abs(decisions[-1].proposed_command.yaw_rate_rad_s) < (
        0.35 * core.config.yaw_gain
        * abs(
            math.atan(
                decisions[-1].camera_current_center_norm[0]
                * core.config.horizontal_angle_scale_rad
            )
        )
    )
    assert all(decision.braking for decision in decisions[-3:])


def test_successor_prediction_cannot_tangent_wrap_across_optical_axis() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    successor = None
    for sequence in range(1, 8):
        observation_time = 1.0 + (sequence - 1) * 0.040
        stable_bearing = 0.15 * sequence
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=0.0,
                log_scale=-1.40 + sequence * 0.18,
            )
        )
        successor = core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=(
                    math.tan(stable_bearing)
                    / core.config.horizontal_angle_scale_rad
                ),
                log_scale=-1.75,
            )
        )
    assert successor is not None
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    _imu(core, 1.25)

    decision = core.guide(1_250_000_000)

    assert decision.successor_prediction_confidence > 0.75
    assert decision.successor_prediction_horizon_s > 0.0
    assert decision.successor_rate_rad_s is not None
    assert (
        successor.bearing_rad[0]
        + decision.successor_rate_rad_s[0]
        * decision.successor_prediction_horizon_s
        > math.pi / 2.0
    )
    assert decision.predicted_successor_bearing_rad is not None
    assert decision.measured_successor_bearing_rad is not None
    assert decision.predicted_successor_bearing_rad[0] > 0.0
    assert (
        decision.predicted_successor_bearing_rad[0]
        - decision.measured_successor_bearing_rad[0]
        <= core.config.successor_prediction_max_extrapolation_rad
        + 1e-12
    )


def test_safe_near_plane_passage_releases_yaw_to_stable_successor() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    for sequence in range(1, 8):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=-0.35,
                log_scale=-0.30,
            )
        )
        core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=0.80,
                log_scale=-1.0,
            )
        )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    _imu(core, 1.25)

    decision = core.guide(1_250_000_000)

    assert decision.current_center_norm[0] < 0.0
    assert abs(decision.passage_error_norm[0]) < 0.04
    assert decision.current_yaw_release > 0.99
    assert decision.passage_yaw_authority > 0.9
    assert decision.successor_prediction_confidence > 0.75
    assert decision.current_time_to_contact_s is None
    assert decision.successor_weight > 0.0
    assert decision.successor_yaw_contribution_rad > 0.0
    assert decision.proposed_command.yaw_rate_rad_s < 0.0
    assert (
        abs(decision.successor_yaw_contribution_rad)
        <= core.config.successor_max_yaw_contribution_rad
    )


def test_body_yaw_cannot_change_stable_passage_or_roll_authority() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    for sequence in range(1, 8):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=-0.20,
                log_scale=-0.30,
            )
        )
        core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=0.30,
                log_scale=-1.00,
            )
        )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    _imu(core, 1.25)
    before = core.guide(1_250_000_000)

    # Change attitude only.  No camera observation or translation occurred.
    _imu(core, 1.30, yaw_rad=0.25)
    after = core.guide(1_300_000_000)

    assert after.current_center_norm == pytest.approx(
        before.current_center_norm
    )
    assert after.current_aperture_half_size_norm == pytest.approx(
        before.current_aperture_half_size_norm
    )
    assert after.passage_point_norm == pytest.approx(
        before.passage_point_norm
    )
    assert after.passage_error_norm == pytest.approx(
        before.passage_error_norm
    )
    assert after.aperture_margin_norm == pytest.approx(
        before.aperture_margin_norm
    )
    assert after.current_yaw_release == pytest.approx(
        before.current_yaw_release
    )
    assert after.passage_yaw_authority == pytest.approx(
        before.passage_yaw_authority
    )
    assert after.successor_weight == pytest.approx(before.successor_weight)
    assert after.proposed_command.target_roll_rad == pytest.approx(
        before.proposed_command.target_roll_rad
    )
    assert after.camera_current_center_norm != pytest.approx(
        before.camera_current_center_norm
    )
    assert after.proposed_command.yaw_rate_rad_s != pytest.approx(
        before.proposed_command.yaw_rate_rad_s
    )


def test_successor_identity_hold_outlives_geometry_dropout_but_is_bounded() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    for sequence in range(1, 5):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation("gate-a", sequence, observation_time)
        )
        core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=0.30,
                log_scale=-0.80,
            )
        )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    _imu(core, 1.16)
    core.observe_track(_observation("gate-a", 5, 1.16))
    successor = core.observe_track(
        _observation("gate-b", 5, 1.16, visible=False)
    )

    assert successor.visible is False
    assert core.retains_successor_lineage("gate-b", 1_450_000_000)
    assert not core.retains_successor_lineage(
        "gate-b",
        1_480_000_000,
    )
    assert not core.retains_successor_lineage(
        "gate-a",
        1_200_000_000,
    )


def test_successor_dropout_requires_fresh_temporal_consistency() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    for sequence in range(1, 6):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=-0.06,
                log_scale=-0.55 + sequence * 0.05,
            )
        )
        core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=0.12,
                log_scale=-0.90,
            )
        )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    _imu(core, 1.17)
    stable = core.guide(1_170_000_000)
    assert stable.successor_prediction_confidence > 0.0
    assert stable.passage_point_norm[0] > 0.0
    _commit_decision(core, 1.17, stable.command)

    _imu(core, 1.20)
    core.observe_track(
        _observation(
            "gate-a",
            6,
            1.20,
            x=-0.06,
            log_scale=-0.20,
        )
    )
    core.observe_track(
        _observation("gate-b", 6, 1.20, visible=False)
    )
    _imu(core, 1.21)
    dropped = core.guide(1_210_000_000)
    assert dropped.successor_prediction_confidence == 0.0
    assert dropped.successor_transition_held
    assert dropped.passage_point_norm == pytest.approx(
        stable.passage_point_norm
    )
    assert dropped.passage_yaw_authority == 0.0
    assert dropped.successor_weight == 0.0
    assert dropped.successor_yaw_contribution_rad == 0.0
    assert (
        dropped.proposed_command.yaw_rate_rad_s
        * stable.proposed_command.yaw_rate_rad_s
        >= 0.0
    )
    assert (
        dropped.command.yaw_rate_rad_s
        * stable.command.yaw_rate_rad_s
        >= 0.0
    )
    _commit_decision(core, 1.21, dropped.command)

    _imu(core, 1.24)
    core.observe_track(
        _observation(
            "gate-a",
            7,
            1.24,
            x=-0.06,
            log_scale=-0.15,
        )
    )
    core.observe_track(
        _observation(
            "gate-b",
            7,
            1.24,
            x=0.12,
            log_scale=-0.85,
        )
    )
    _imu(core, 1.25)
    reacquired = core.guide(1_250_000_000)

    assert reacquired.successor_prediction_confidence == 0.0
    assert not reacquired.successor_transition_held
    assert reacquired.passage_point_norm == pytest.approx(
        stable.passage_point_norm
    )
    assert reacquired.passage_yaw_authority == 0.0
    assert reacquired.successor_weight == 0.0
    assert reacquired.successor_yaw_contribution_rad == 0.0

    _imu(core, 1.40)
    core.observe_track(
        _observation(
            "gate-a",
            8,
            1.40,
            x=-0.06,
            log_scale=-0.10,
        )
    )
    core.observe_track(
        _observation("gate-b", 8, 1.40, visible=False)
    )
    _imu(core, 1.41)
    expired = core.guide(1_410_000_000)

    assert not expired.successor_transition_held
    assert expired.passage_point_norm == (0.0, 0.0)
    assert expired.current_yaw_release == 0.0


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
    for sequence in range(1, 5):
        observation_time = 1.1 + (sequence - 1) * 0.03
        _imu(core, observation_time, yaw_rad=0.20)
        core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=math.tan(0.20) / config.horizontal_angle_scale_rad,
            )
        )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    _imu(core, 1.22, yaw_rad=0.10)
    decision = core.guide(1_220_000_000)

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
