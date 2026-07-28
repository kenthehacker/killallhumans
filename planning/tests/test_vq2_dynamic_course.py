from __future__ import annotations

import math

import pytest

from competition.vq2_contracts import FrameEdge
from planning.vq2_dynamic_course import (
    AppliedCommandSample,
    DynamicCourseCommand,
    DynamicCourseConfig,
    DynamicCourseCore,
    DynamicCourseError,
    GateObservation,
    GuidanceDecision,
    ImuAttitudeSample,
    MAX_TARGET_PITCH_RAD,
    MAX_TARGET_ROLL_RAD,
    MAX_THRUST,
    MAX_YAW_RATE_RAD_S,
    MIN_TARGET_PITCH_RAD,
    MIN_THRUST,
    SUPPORT_THRUST,
    TrackDynamicState,
    _bearing_ray,
    _quat_rotate,
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


def test_1cab9da6_separates_unsafe_approach_sweep_from_safe_terminal_window(
) -> None:
    """The converging final state is unsafe now but 2-sigma safe at contact."""

    current_q = (0.1573332768127754, -1.3900652360978418)
    q_rate = (0.12404522007248783, 1.8692942335278362)
    predicted_std = (0.05534993410482621, 0.14059388915832344)
    timing_uncertainty_s = 0.020
    prediction = predict_aperture_relative_crossing(
        center_offset_norm=current_q,
        passage_offset_norm=(0.0, 0.0),
        aperture_half_extent_norm=(1.0, 1.0),
        center_rate_norm_s=q_rate,
        aperture_expansion_rate_s=(0.0, 0.0),
        center_std_norm=tuple(
            predicted_std[axis]
            - abs(q_rate[axis]) * timing_uncertainty_s
            for axis in range(2)
        ),
        aperture_log_scale_std=0.0,
        capture_timing_uncertainty_s=timing_uncertainty_s,
        horizon_s=0.7710673805867182,
        allowance_q=(0.50, 0.45),
    )

    assert prediction.predicted_error_q == pytest.approx(
        (0.25298049972837156, 0.05128657209432386)
    )
    assert prediction.predicted_std_q == pytest.approx(predicted_std)
    assert prediction.clearance_q == pytest.approx(
        (0.13631963206197606, -1.1464812450733752)
    )
    assert prediction.terminal_clearance_q == pytest.approx(
        (0.13631963206197606, 0.11752564958902927)
    )
    assert current_q[1] * q_rate[1] < 0.0


def test_expanding_aperture_projects_uncertainty_to_crossing_scale() -> None:
    """An inward, expanding gate can be 2-sigma safe at the endpoint."""

    prediction = predict_aperture_relative_crossing(
        center_offset_norm=(
            0.01801119772014466,
            -0.1939082635722633,
        ),
        passage_offset_norm=(0.0, 0.0),
        aperture_half_extent_norm=(
            0.20974697790325306,
            0.3538317436193515,
        ),
        center_rate_norm_s=(
            -0.06663773168763444,
            -0.09887852901171028,
        ),
        aperture_expansion_rate_s=(
            1.6174475604358003,
            1.6174475604358003,
        ),
        center_std_norm=(
            0.013762345989051085,
            0.02827581714755323,
        ),
        aperture_log_scale_std=0.2731951664167071,
        capture_timing_uncertainty_s=0.020,
        horizon_s=0.6182580656466927,
        allowance_q=(0.50, 0.45),
    )

    assert all(
        predicted < current
        for predicted, current in zip(
            prediction.predicted_std_q,
            prediction.current_std_q,
            strict=True,
        )
    )
    assert all(
        clearance > 0.0
        for clearance in prediction.terminal_clearance_q
    )
    assert prediction.clearance_q[1] < 0.0
    assert (
        prediction.current_error_q[1]
        * prediction.rate_q_s[1]
        < 0.0
    )


def test_096f78c4_successor_bias_cannot_hide_unsafe_current_crossing() -> None:
    """The exact terminal q state must be assessed current-center first."""

    center = (0.05151534397, -1.15050750323)
    aperture = (0.39233066014, 0.78292060212)
    rejected_passage = (0.25379023455, -0.21728137506)
    q_rate = (0.174765, 1.975791)
    predicted_std_q = (0.091504, 0.160834)
    horizon = 0.811743
    center_rate = tuple(
        q_rate[axis] * aperture[axis] for axis in range(2)
    )
    center_std = tuple(
        predicted_std_q[axis] * aperture[axis] for axis in range(2)
    )
    common = {
        "center_offset_norm": center,
        "aperture_half_extent_norm": aperture,
        "center_rate_norm_s": center_rate,
        "aperture_expansion_rate_s": (0.0, 0.0),
        "center_std_norm": center_std,
        "aperture_log_scale_std": 0.0,
        "capture_timing_uncertainty_s": 0.0,
        "horizon_s": horizon,
        "allowance_q": (0.50, 0.45),
    }
    centered = predict_aperture_relative_crossing(
        passage_offset_norm=(0.0, 0.0),
        **common,
    )
    rejected = predict_aperture_relative_crossing(
        passage_offset_norm=rejected_passage,
        **common,
    )

    assert rejected.predicted_error_q[0] == pytest.approx(
        0.920048,
        abs=3e-6,
    )
    assert rejected.swept_occupancy_q[0] == pytest.approx(
        1.74993,
        abs=2e-5,
    )
    assert centered.predicted_error_q[0] == pytest.approx(
        0.273170,
        abs=3e-6,
    )
    assert centered.clearance_q[0] > 0.0
    # The current aperture remained vertically unsafe, so positive horizontal
    # reserve cannot authorize either successor passage or yaw.
    assert centered.clearance_q[1] < 0.0


def _euler_quaternion(
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> tuple[float, float, float, float]:
    cr = math.cos(roll_rad / 2.0)
    sr = math.sin(roll_rad / 2.0)
    cp = math.cos(pitch_rad / 2.0)
    sp = math.sin(pitch_rad / 2.0)
    cy = math.cos(yaw_rad / 2.0)
    sy = math.sin(yaw_rad / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _imu(
    core: DynamicCourseCore,
    time_s: float,
    *,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
    yaw_rad: float = 0.0,
    roll_rate_rad_s: float = 0.0,
    pitch_rate_rad_s: float = 0.0,
    yaw_rate_rad_s: float = 0.0,
) -> None:
    core.record_imu(
        ImuAttitudeSample(
            monotonic_ns=round(time_s * NS),
            body_to_reference_wxyz=_euler_quaternion(
                roll_rad,
                pitch_rad,
                yaw_rad,
            ),
            body_rates_rad_s=(
                roll_rate_rad_s,
                pitch_rate_rad_s,
                yaw_rate_rad_s,
            ),
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
    aperture: tuple[float, float] | None = (0.42, 0.34),
    clipping: FrameEdge = FrameEdge.NONE,
    center_censored: bool = False,
    visible: bool = True,
    ambiguous: bool = False,
    confidence: float = 0.95,
    inner_scale_measurement_usable: bool = False,
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
        center_censored=center_censored,
        visible=visible,
        ambiguous=ambiguous,
        confidence=confidence,
        measurement_std=(0.005, 0.005, 0.01),
        inner_scale_measurement_usable=(
            inner_scale_measurement_usable if visible else False
        ),
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


@pytest.mark.parametrize("axis", ("roll", "pitch", "yaw"))
def test_small_body_axis_rotation_preserves_one_stationary_optical_ray(
    axis,
) -> None:
    config = DynamicCourseConfig(
        camera_delay_s=0.0,
        horizontal_angle_scale_rad=1.59,
        vertical_angle_scale_rad=0.55,
        camera_to_body_wxyz=(0.0, 1.0, 0.0, 0.0),
    )
    core = DynamicCourseCore(config)
    initial_center = (0.12, -0.08)
    _imu(core, 1.0)
    first = core.observe_track(
        _observation(
            "gate-a",
            1,
            1.0,
            x=initial_center[0],
            y=initial_center[1],
        )
    )

    angle = 0.08
    cosine = math.cos(angle)
    sine = math.sin(angle)
    initial_ray = (
        1.0,
        initial_center[0] * config.horizontal_angle_scale_rad,
        initial_center[1] * config.vertical_angle_scale_rad,
    )
    forward, right, down = initial_ray
    if axis == "roll":
        # camera_to_body=Rx(pi), so body roll keeps its sign while the
        # stationary ray rotates by the inverse body roll in the camera.
        raw_ray = (
            forward,
            cosine * right + sine * down,
            -sine * right + cosine * down,
        )
    elif axis == "pitch":
        # Rx(pi) changes the body-pitch sign in the decoded camera chart.
        raw_ray = (
            cosine * forward + sine * down,
            right,
            -sine * forward + cosine * down,
        )
    else:
        # Rx(pi) also changes the body-yaw sign in the decoded chart.
        raw_ray = (
            cosine * forward - sine * right,
            sine * forward + cosine * right,
            down,
        )
    raw_center = (
        raw_ray[1]
        / raw_ray[0]
        / config.horizontal_angle_scale_rad,
        raw_ray[2]
        / raw_ray[0]
        / config.vertical_angle_scale_rad,
    )
    attitude = {f"{axis}_rad": angle}
    rate = {f"{axis}_rate_rad_s": angle / 0.1}
    _imu(core, 1.1, **attitude, **rate)
    second = core.observe_track(
        _observation(
            "gate-a",
            2,
            1.1,
            x=raw_center[0],
            y=raw_center[1],
        )
    )

    assert second.bearing_rad == pytest.approx(
        first.bearing_rad,
        abs=2e-10,
    )
    assert second.bearing_rate_rad_s == pytest.approx((0.0, 0.0), abs=2e-9)
    assert second.residual_translational_rate_rad_s == pytest.approx(
        (0.0, 0.0),
        abs=2e-9,
    )


def test_latest_gate0_pitch_and_image_motion_are_optically_invariant() -> None:
    """Capture the build-3385 relationship before changing vertical control."""

    config = DynamicCourseConfig(
        camera_delay_s=0.0,
        vertical_angle_scale_rad=0.55,
        camera_to_body_wxyz=(0.0, 1.0, 0.0, 0.0),
    )
    core = DynamicCourseCore(config)
    _imu(core, 1.0, pitch_rad=-0.310)
    first = core.observe_track(
        _observation("gate-a", 1, 1.0, y=-0.025)
    )

    _imu(core, 2.156, pitch_rad=0.021)
    second = core.observe_track(
        _observation("gate-a", 2, 2.156, y=-0.649)
    )

    assert second.bearing_rad[1] == pytest.approx(
        first.bearing_rad[1],
        abs=0.003,
    )
    assert abs(second.residual_translational_rate_rad_s[1]) < 0.003


def test_stable_geometry_keeps_elevation_invariant_at_large_azimuth() -> None:
    """A horizontal turn cannot magnify one unchanged physical elevation."""

    config = DynamicCourseConfig(camera_delay_s=0.0)

    def decision_at(
        azimuth_rad: float,
        elevation_rad: float,
    ) -> tuple[DynamicCourseCore, GuidanceDecision]:
        core = DynamicCourseCore(config)
        core.record_applied_command(_command(0.90))
        cos_elevation = math.cos(elevation_rad)
        ray = (
            cos_elevation * math.cos(azimuth_rad),
            cos_elevation * math.sin(azimuth_rad),
            math.sin(elevation_rad),
        )
        raw_center = (
            ray[1]
            / ray[0]
            / config.horizontal_angle_scale_rad,
            ray[2]
            / ray[0]
            / config.vertical_angle_scale_rad,
        )
        _imu(core, 1.0)
        core.observe_track(
            _observation(
                "gate-a",
                1,
                1.0,
                x=raw_center[0],
                y=raw_center[1],
                aperture=(0.10, 0.10),
            )
        )
        core.bind(
            current_gate_index=0,
            current_track_id="gate-a",
            successor_track_id=None,
        )
        _imu(core, 1.01)
        return core, core.guide(1_010_000_000)

    _, centered = decision_at(0.20, -0.18)
    _, turned = decision_at(1.30, -0.18)

    assert turned.current_center_norm[0] > centered.current_center_norm[0]
    assert turned.current_center_norm[1] == pytest.approx(
        centered.current_center_norm[1],
        abs=1e-12,
    )
    assert turned.current_center_norm[1] == pytest.approx(
        math.tan(-0.18) / config.vertical_angle_scale_rad,
        abs=1e-12,
    )
    assert abs(turned.camera_current_center_norm[1]) > (
        3.0 * abs(centered.camera_current_center_norm[1])
    )
    assert all(
        math.isfinite(value)
        for value in (
            *turned.current_center_norm,
            *turned.current_aperture_half_size_norm,
            turned.command.target_roll_rad,
            turned.command.target_pitch_rad,
            turned.command.yaw_rate_rad_s,
            turned.command.thrust,
        )
    )
    assert abs(turned.command.target_roll_rad) <= MAX_TARGET_ROLL_RAD
    assert (
        MIN_TARGET_PITCH_RAD
        <= turned.command.target_pitch_rad
        <= MAX_TARGET_PITCH_RAD
    )
    assert abs(turned.command.yaw_rate_rad_s) <= MAX_YAW_RATE_RAD_S
    assert MIN_THRUST <= turned.command.thrust <= MAX_THRUST


def test_4c42bb77_contour_completion_cannot_seed_collective_rate() -> None:
    """Reject the exact one-frame launch contour jump as translation."""

    config = DynamicCourseConfig(camera_delay_s=0.0)
    core = DynamicCourseCore(config)
    core.record_applied_command(
        _command(
            0.90,
            pitch=-0.3101673180451623,
            thrust=0.26,
        )
    )
    samples = (
        (
            1.000,
            0.02777777777777768,
            0.16388888888888886,
            0.1431297003109806,
        ),
        (
            1.031,
            -0.033333333333333326,
            0.22499999999999998,
            0.16770509831248426,
        ),
        (
            1.063,
            -0.02777777777777779,
            0.2222222222222222,
            0.1666666666666667,
        ),
        (
            1.094,
            -0.033333333333333326,
            0.22499999999999998,
            0.16770509831248426,
        ),
    )
    states = []
    for sequence, (time_s, y, aperture_y, scale) in enumerate(
        samples,
        start=1,
    ):
        _imu(core, time_s)
        states.append(
            core.observe_track(
                _observation(
                    "gate-0",
                    sequence,
                    time_s,
                    x=0.006250000000000089,
                    y=y,
                    log_scale=math.log(scale),
                    aperture=(0.125, aperture_y),
                    confidence=0.67,
                )
            )
        )

    core.bind(
        current_gate_index=0,
        current_track_id="gate-0",
        successor_track_id=None,
    )
    decision = core.guide(round(samples[-1][0] * NS))
    first_contour_update = states[1]
    qualified = states[-1]

    assert first_contour_update.residual_translational_rate_rad_s[1] == (
        pytest.approx(0.0, abs=1e-12)
    )
    assert first_contour_update.time_to_contact_s is None
    assert abs(decision.crossing_rate_q_s[1]) < 0.35
    assert abs(
        qualified.residual_translational_rate_rad_s[1]
        / config.vertical_angle_scale_rad
    ) < 0.08


def test_5f132788_single_aperture_collapse_is_not_crossing_geometry() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    states = []
    for sequence in range(1, 6):
        time_s = 1.0 + (sequence - 1) * 0.031
        collapsed = sequence == 5
        _imu(core, time_s)
        states.append(
            core.observe_track(
                _observation(
                    "gate-0",
                    sequence,
                    time_s,
                    y=(
                        0.011111111111111072
                        if collapsed
                        else -0.033333333333333326
                    ),
                    log_scale=math.log(
                        0.149071198499986
                        if collapsed
                        else 0.1666666666666667
                    ),
                    aperture=(
                        0.125,
                        (
                            0.1777777777777777
                            if collapsed
                            else 0.2222222222222222
                        ),
                    ),
                    confidence=0.80,
                )
            )
        )

    before = states[-2]
    after = states[-1]
    assert before.bearing_rate_qualified == (True, True)
    assert before.scale_rate_qualified is True
    assert after.aperture_half_size_norm[1] == pytest.approx(
        0.2222222222222222
    )
    assert after.scale_rate_qualified is True
    assert abs(after.expansion_rate_s) < 0.10


def test_vertical_uncertainty_emits_a_responsive_bounded_brake_demand() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90, pitch=-0.31))
    _imu(core, 1.0)
    core.observe_track(
        _observation(
            "gate-a",
            1,
            1.0,
        )
    )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id=None,
    )
    _imu(core, 1.01)

    decision = core.guide(1_010_000_000)
    values = (
        decision.command.target_roll_rad,
        decision.command.target_pitch_rad,
        decision.command.yaw_rate_rad_s,
        decision.command.thrust,
    )

    assert decision.braking
    assert decision.command == decision.proposed_command
    assert decision.command.target_pitch_rad > 0.0
    assert all(math.isfinite(value) for value in values)
    assert abs(decision.command.target_roll_rad) <= MAX_TARGET_ROLL_RAD
    assert (
        MIN_TARGET_PITCH_RAD
        <= decision.command.target_pitch_rad
        <= MAX_TARGET_PITCH_RAD
    )
    assert abs(decision.command.yaw_rate_rad_s) <= MAX_YAW_RATE_RAD_S
    assert MIN_THRUST <= decision.command.thrust <= MAX_THRUST


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


def test_noisy_alternating_detections_emit_responsive_bounded_demands() -> None:
    config = DynamicCourseConfig(
        camera_delay_s=0.0,
        roll_guidance_sign=1.0,
        roll_gain=0.65,
        bearing_alpha=0.95,
        bearing_beta=0.25,
    )
    core = DynamicCourseCore(config)
    core.record_applied_command(_command(0.90))
    outputs: list[float] = []
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
        assert decision.command == decision.proposed_command
        assert all(
            math.isfinite(value)
            for value in (
                decision.command.target_roll_rad,
                decision.command.target_pitch_rad,
                decision.command.yaw_rate_rad_s,
                decision.command.thrust,
            )
        )
        assert abs(decision.command.target_roll_rad) <= MAX_TARGET_ROLL_RAD
        assert (
            MIN_TARGET_PITCH_RAD
            <= decision.command.target_pitch_rad
            <= MAX_TARGET_PITCH_RAD
        )
        assert abs(decision.command.yaw_rate_rad_s) <= MAX_YAW_RATE_RAD_S
        assert MIN_THRUST <= decision.command.thrust <= MAX_THRUST
        _commit_decision(core, decision_time, decision.command)

    assert any(abs(value) > 1e-5 for value in outputs)
    assert len({round(value, 8) for value in outputs}) > 1


def test_successor_steering_and_state_are_continuous_through_promotion() -> None:
    core = DynamicCourseCore(
        DynamicCourseConfig(
            camera_delay_s=0.0,
            successor_clearance_dwell_s=0.04,
            successor_clearance_ramp_s=0.04,
        )
    )
    core.record_applied_command(_command(0.90))
    successor_before = None
    decisions = []
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
        if sequence == 1:
            core.bind(
                current_gate_index=3,
                current_track_id="gate-a",
                successor_track_id="gate-b",
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        decision = core.guide(round(decision_time * NS))
        decisions.append(decision)
        _commit_decision(core, decision_time, decision.command)
    assert successor_before is not None
    precredit = decisions[-1]
    assert precredit.successor_weight > 0.0
    assert precredit.command.yaw_rate_rad_s < 0.0

    promoted = core.promote_authoritative(
        from_gate_index=3,
        to_gate_index=4,
        promoted_track_id="gate-b",
        next_successor_track_id="gate-c",
        monotonic_ns=1_215_000_000,
    )
    assert promoted.current.track_id == successor_before.track_id
    assert promoted.current.stream_generation == (
        successor_before.stream_generation
    )
    assert promoted.current.frame_sequence == successor_before.frame_sequence
    assert promoted.current.last_measurement_monotonic_ns == (
        successor_before.last_measurement_monotonic_ns
    )
    assert promoted.current.aperture_half_size_norm == pytest.approx(
        successor_before.aperture_half_size_norm
    )
    assert promoted.current.aperture_seed_monotonic_ns == (
        successor_before.aperture_seed_monotonic_ns
    )
    assert promoted.current.aperture_prediction_deadline_monotonic_ns == (
        successor_before.aperture_prediction_deadline_monotonic_ns
    )
    assert promoted.last_applied_command == precredit.command

    _imu(core, 1.24)
    postcredit = core.guide(1_240_000_000)
    assert postcredit.current_track_id == "gate-b"
    assert postcredit.successor_track_id == "gate-c"
    assert postcredit.successor_clearance_authority == 0.0
    assert postcredit.successor_passage_authority == 0.0
    assert postcredit.passage_point_norm == (0.0, 0.0)
    assert postcredit.command == postcredit.proposed_command
    assert postcredit.command.yaw_rate_rad_s != 0.0
    assert math.isfinite(postcredit.command.yaw_rate_rad_s)
    assert (
        abs(postcredit.command.yaw_rate_rad_s)
        <= MAX_YAW_RATE_RAD_S
    )


def test_8319198e_unadmitted_successor_cannot_force_gate0_braking() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90, pitch=-0.02))
    _imu(core, 1.0)
    core.observe_track(_observation("gate-a", 1, 1.0))
    core.observe_track(
        _observation("gate-b", 1, 1.0, x=0.322, log_scale=-0.30)
    )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    _imu(core, 1.01)
    decision = core.guide(1_010_000_000)

    assert decision.predicted_successor_bearing_rad is not None
    assert decision.predicted_successor_bearing_rad[0] == pytest.approx(
        0.473,
        abs=0.002,
    )
    assert decision.successor_weight == 0.0
    assert decision.successor_yaw_contribution_rad == 0.0
    assert decision.braking is True
    assert decision.brake_reason == "vertical_alignment_unsettled"
    assert decision.proposed_command.target_pitch_rad > 0.0
    assert core.course_state().current.bearing_rate_qualified == (
        False,
        False,
    )
    assert decision.command.thrust == pytest.approx(SUPPORT_THRUST)


def test_successor_bias_waits_for_fresh_safe_dwell_then_ramps() -> None:
    core = DynamicCourseCore(
        DynamicCourseConfig(
            camera_delay_s=0.0,
            successor_clearance_dwell_s=0.08,
            successor_clearance_ramp_s=0.08,
        )
    )
    core.record_applied_command(_command(0.90, pitch=-0.02))
    decisions = []
    for sequence in range(1, 12):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=0.0,
                y=0.0,
                log_scale=-0.72 + sequence * 0.035,
                aperture=(0.42, 0.34),
                confidence=0.85,
            )
        )
        core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=0.20,
                y=-0.05,
                log_scale=-1.00,
                confidence=0.90,
            )
        )
        if sequence == 1:
            core.bind(
                current_gate_index=0,
                current_track_id="gate-a",
                successor_track_id="gate-b",
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        decision = core.guide(round(decision_time * NS))
        decisions.append(decision)
        _commit_decision(core, decision_time, decision.command)

    _imu(core, 1.407)
    duplicate = core.guide(1_407_000_000)
    assert duplicate.successor_clearance_dwell_s == pytest.approx(
        decisions[-1].successor_clearance_dwell_s
    )
    assert duplicate.successor_clearance_authority == pytest.approx(
        decisions[-1].successor_clearance_authority
    )
    assert all(
        decision.successor_passage_authority == 0.0
        and decision.passage_point_norm == (0.0, 0.0)
        for decision in decisions
        if decision.successor_clearance_dwell_s
        <= core.config.successor_clearance_dwell_s
    )
    released = [
        decision
        for decision in decisions
        if decision.successor_clearance_authority > 0.0
    ]
    assert released
    assert released[0].successor_clearance_dwell_s > (
        core.config.successor_clearance_dwell_s
    )
    assert 0.0 < released[0].successor_clearance_authority < 1.0
    assert released[0].successor_passage_authority > 0.0
    assert released[0].passage_point_norm[0] > 0.0
    assert all(
        clearance > 0.0
        for clearance in released[0].predicted_crossing_clearance_norm
    )
    assert released[-1].successor_clearance_authority > (
        released[0].successor_clearance_authority
    )


def test_096f78c4_unsafe_current_gate_owns_passage_and_yaw() -> None:
    core = DynamicCourseCore(
        DynamicCourseConfig(
            camera_delay_s=0.0,
            successor_clearance_dwell_s=0.04,
            successor_clearance_ramp_s=0.04,
        )
    )
    core.record_applied_command(_command(0.90))
    decisions = []
    for sequence in range(1, 10):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=0.05,
                y=-0.82,
                log_scale=-0.90 + 0.06 * sequence,
                aperture=(0.39, 0.78),
            )
        )
        core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=0.65,
                y=-0.25,
                log_scale=-1.10,
            )
        )
        if sequence == 1:
            core.bind(
                current_gate_index=4,
                current_track_id="gate-a",
                successor_track_id="gate-b",
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        decision = core.guide(round(decision_time * NS))
        decisions.append(decision)
        _commit_decision(core, decision_time, decision.command)

    qualified = decisions[-1]
    assert qualified.predicted_successor_bearing_rad is not None
    assert qualified.successor_prediction_confidence > 0.0
    assert qualified.centered_crossing_clearance_norm[1] < 0.0
    assert qualified.successor_clearance_authority == 0.0
    assert qualified.successor_passage_authority == 0.0
    assert qualified.passage_point_norm == (0.0, 0.0)
    assert qualified.passage_error_norm == pytest.approx(
        qualified.current_center_norm
    )
    assert qualified.current_yaw_release == 0.0
    assert qualified.passage_yaw_authority == 0.0
    assert qualified.successor_weight == 0.0
    assert qualified.successor_yaw_contribution_rad == 0.0
    assert qualified.braking


def test_admitted_off_axis_successor_still_brakes_before_intercept() -> None:
    core = DynamicCourseCore(
        DynamicCourseConfig(
            camera_delay_s=0.0,
            passage_successor_bias=0.01,
            successor_clearance_dwell_s=0.04,
            successor_clearance_ramp_s=0.04,
        )
    )
    core.record_applied_command(_command(0.90, pitch=-0.02))
    decisions = []
    for sequence in range(1, 8):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                log_scale=-0.50 + 0.06 * sequence,
            )
        )
        core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=0.322,
                log_scale=-0.30,
            )
        )
        if sequence == 1:
            core.bind(
                current_gate_index=0,
                current_track_id="gate-a",
                successor_track_id="gate-b",
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        decision = core.guide(round(decision_time * NS))
        decisions.append(decision)
        _commit_decision(core, decision_time, decision.command)
    decision = decisions[-1]

    assert decision.current_time_to_contact_s is not None
    assert decision.successor_prediction_confidence > 0.75
    assert decision.successor_weight > 0.0
    assert abs(decision.passage_error_norm[0]) < 0.01
    assert decision.braking is True
    assert decision.brake_reason == "off_axis_rapid_closure"
    assert decision.proposed_command.target_pitch_rad > 0.0


def test_unsettled_vertical_crossing_brakes_before_rapid_closure() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90, pitch=-0.02))
    _imu(core, 1.0)
    current = core.observe_track(
        _observation(
            "gate-a",
            1,
            1.0,
            y=0.24,
            aperture=(0.42, 0.34),
        )
    )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id=None,
    )
    _imu(core, 1.01)
    decision = core.guide(1_010_000_000)

    assert current.time_to_contact_s is None
    assert current.expansion_rate_s < core.config.rapid_expansion_rate_s
    assert decision.predicted_crossing_clearance_norm[1] < 0.0
    assert decision.braking is True
    assert decision.brake_reason == "vertical_alignment_unsettled"
    assert decision.proposed_command.target_pitch_rad > 0.0


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


def test_negative_clearance_near_center_cannot_invent_full_bank() -> None:
    config = DynamicCourseConfig(
        camera_delay_s=0.0,
        bearing_alpha=1.0,
        bearing_beta=1.0,
        scale_alpha=1.0,
        scale_beta=1.0,
        roll_guidance_sign=1.0,
        roll_gain=0.18,
        lateral_rate_gain=0.045,
    )
    core = DynamicCourseCore(config)
    core.record_applied_command(_command(0.90))
    decision = None
    current = None
    for sequence in range(1, 8):
        observation_time = 1.0 + (sequence - 1) * 0.040
        log_scale = -1.40 + (sequence - 1) * 0.20
        aperture_scale = math.exp(log_scale + 1.40)
        _imu(core, observation_time)
        current = core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=0.004 + (sequence - 1) * 0.001,
                log_scale=log_scale,
                aperture=(
                    0.010 * aperture_scale,
                    0.008 * aperture_scale,
                ),
            )
        )
        if sequence == 1:
            core.bind(
                current_gate_index=0,
                current_track_id="gate-a",
                successor_track_id=None,
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        decision = core.guide(round(decision_time * NS))
        _commit_decision(core, decision_time, decision.command)

    assert current is not None
    assert decision is not None
    lateral_rate_norm_s = (
        current.residual_translational_rate_rad_s[0]
        / config.horizontal_angle_scale_rad
    )
    stable_bearing_rad = math.atan(
        config.horizontal_angle_scale_rad
        * decision.passage_error_norm[0]
    )
    assert current.bearing_rate_qualified[0]
    assert current.scale_rate_qualified
    assert decision.centered_crossing_clearance_norm[0] < 0.0
    assert decision.passage_error_norm[0] * lateral_rate_norm_s > 0.0
    assert abs(stable_bearing_rad) < config.off_axis_brake_rad
    assert (
        0.0
        < decision.proposed_command.target_roll_rad
        < MAX_TARGET_ROLL_RAD
    )
    assert math.isfinite(decision.proposed_command.target_roll_rad)


def test_unsafe_outward_lateral_motion_uses_full_attitude_then_releases(
) -> None:
    core = DynamicCourseCore(
        DynamicCourseConfig(
            camera_delay_s=0.0,
            bearing_alpha=1.0,
            bearing_beta=1.0,
            scale_alpha=1.0,
            scale_beta=1.0,
            roll_guidance_sign=1.0,
            roll_gain=0.18,
            lateral_rate_gain=0.045,
        )
    )
    core.record_applied_command(_command(0.90))
    decision = None
    current = None
    for sequence in range(1, 8):
        observation_time = 1.0 + (sequence - 1) * 0.040
        log_scale = -1.40 + (sequence - 1) * 0.20
        aperture_scale = math.exp(log_scale + 1.40)
        _imu(core, observation_time)
        current = core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=0.35 + (sequence - 1) * 0.004,
                log_scale=log_scale,
                aperture=(
                    0.14 * aperture_scale,
                    0.11 * aperture_scale,
                ),
            )
        )
        if sequence == 1:
            core.bind(
                current_gate_index=0,
                current_track_id="gate-a",
                successor_track_id=None,
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        decision = core.guide(round(decision_time * NS))
        _commit_decision(core, decision_time, decision.command)

    assert current is not None
    assert decision is not None
    residual_lateral_rate_norm_s = (
        current.residual_translational_rate_rad_s[0]
        / core.config.horizontal_angle_scale_rad
    )
    assert decision.passage_error_norm[0] > 0.0
    assert residual_lateral_rate_norm_s > 0.0
    assert current.expansion_rate_s > 0.0
    assert decision.crossing_rate_q_s[0] < 0.0
    assert decision.centered_crossing_clearance_norm[0] < 0.0
    assert (
        decision.proposed_command.target_roll_rad
        == MAX_TARGET_ROLL_RAD
    )
    assert math.isfinite(decision.proposed_command.target_roll_rad)

    recovered = None
    recovered_decision = None
    for sequence, x in ((8, 0.366), (9, 0.358)):
        observation_time = 1.0 + (sequence - 1) * 0.040
        log_scale = -1.40 + (sequence - 1) * 0.20
        aperture_scale = math.exp(log_scale + 1.40)
        _imu(core, observation_time)
        recovered = core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=x,
                log_scale=log_scale,
                aperture=(
                    0.14 * aperture_scale,
                    0.11 * aperture_scale,
                ),
            )
        )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        recovered_decision = core.guide(round(decision_time * NS))
        _commit_decision(
            core,
            decision_time,
            recovered_decision.command,
        )

    assert recovered is not None
    assert recovered_decision is not None
    recovered_lateral_rate_norm_s = (
        recovered.residual_translational_rate_rad_s[0]
        / core.config.horizontal_angle_scale_rad
    )
    assert recovered_lateral_rate_norm_s < 0.0
    assert recovered_decision.passage_error_norm[0] > 0.0
    assert (
        recovered_decision.centered_crossing_clearance_norm[0]
        < 0.0
    )
    assert (
        0.0
        < recovered_decision.proposed_command.target_roll_rad
        < MAX_TARGET_ROLL_RAD
    )
    assert math.isfinite(
        recovered_decision.proposed_command.target_roll_rad
    )


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
    # This replay never establishes a horizontally safe current-aperture
    # passage.  Successor prediction therefore remains lookahead evidence,
    # but it gets no yaw leverage with which to recreate the observed hunt.
    assert all(
        decision.successor_clearance_authority == 0.0
        and decision.successor_passage_authority == 0.0
        and decision.passage_point_norm == (0.0, 0.0)
        and decision.current_yaw_release == 0.0
        and decision.passage_yaw_authority == 0.0
        and decision.successor_weight == 0.0
        for decision in decisions
    )
    assert all(
        decision.successor_yaw_contribution_rad == 0.0
        for decision in decisions
    )
    for decision in decisions:
        assert (
            abs(decision.proposed_command.yaw_rate_rad_s)
            <= MAX_YAW_RATE_RAD_S
        )
    for decision in decisions:
        current_only = -core.config.yaw_gain * math.atan(
            decision.camera_current_center_norm[0]
            * core.config.horizontal_angle_scale_rad
        )
        assert decision.proposed_command.yaw_rate_rad_s == pytest.approx(
            max(-MAX_YAW_RATE_RAD_S, min(MAX_YAW_RATE_RAD_S, current_only))
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
    core = DynamicCourseCore(
        DynamicCourseConfig(
            camera_delay_s=0.0,
            successor_clearance_dwell_s=0.04,
            successor_clearance_ramp_s=0.04,
        )
    )
    core.record_applied_command(_command(0.90))
    decisions = []
    for sequence in range(1, 9):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=-0.05,
                log_scale=-0.68 + sequence * 0.05,
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
        if sequence == 1:
            core.bind(
                current_gate_index=0,
                current_track_id="gate-a",
                successor_track_id="gate-b",
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        decision = core.guide(round(decision_time * NS))
        decisions.append(decision)
        _commit_decision(core, decision_time, decision.command)
    decision = decisions[-1]

    assert decision.current_center_norm[0] < 0.0
    assert all(
        clearance > 0.0
        for clearance in decision.centered_crossing_clearance_norm
    )
    assert decision.successor_clearance_authority > 0.0
    assert decision.current_yaw_release > 0.0
    assert decision.passage_yaw_authority > 0.0
    assert decision.successor_prediction_confidence > 0.75
    assert decision.current_time_to_contact_s is not None
    assert decision.successor_weight > 0.0
    assert decision.successor_yaw_contribution_rad > 0.0
    current_gate_heading = (
        (1.0 - decision.current_yaw_release)
        * math.atan(
            decision.camera_current_center_norm[0]
            * core.config.horizontal_angle_scale_rad
        )
    )
    assert decision.proposed_command.yaw_rate_rad_s < (
        -core.config.yaw_gain * current_gate_heading
    )
    assert (
        abs(decision.successor_yaw_contribution_rad)
        <= core.config.successor_max_yaw_contribution_rad
    )


def test_clean_aperture_seed_propagates_through_near_plane_censor_and_dropout(
) -> None:
    def propagate(
        *,
        pitch: float,
        thrust: float,
        clipping: FrameEdge,
    ):
        core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
        core.record_applied_command(
            _command(0.80, pitch=pitch, thrust=thrust)
        )
        clean = None
        for sequence in range(1, 5):
            observation_time = 1.0 + (sequence - 1) * 0.030
            _imu(core, observation_time)
            clean = core.observe_track(
                _observation(
                    "gate-a",
                    sequence,
                    observation_time,
                    x=0.04,
                    y=-0.08,
                    log_scale=-0.45,
                )
            )
        assert clean is not None
        core.bind(
            current_gate_index=0,
            current_track_id="gate-a",
            successor_track_id=None,
        )
        _imu(core, 1.095)
        clean_decision = core.guide(1_095_000_000)

        _imu(core, 1.12)
        censored = core.observe_track(
            _observation(
                "gate-a",
                5,
                1.12,
                x=0.90,
                y=-0.95,
                log_scale=0.70,
                aperture=None,
                clipping=clipping,
                center_censored=True,
                ambiguous=False,
            )
        )
        _imu(core, 1.125)
        censored_decision = core.guide(1_125_000_000)

        _imu(core, 1.15)
        dropout = core.observe_track(
            _observation("gate-a", 6, 1.15, visible=False)
        )
        _imu(core, 1.155)
        dropout_decision = core.guide(1_155_000_000)
        return (
            clean,
            clean_decision,
            censored,
            censored_decision,
            dropout,
            dropout_decision,
        )

    nose_down = propagate(
        pitch=-0.12,
        thrust=MAX_THRUST,
        clipping=FrameEdge.LEFT | FrameEdge.TOP | FrameEdge.RIGHT,
    )
    nose_up = propagate(
        pitch=0.12,
        thrust=MIN_THRUST,
        clipping=FrameEdge.BOTTOM,
    )

    for (
        clean,
        clean_decision,
        censored,
        censored_decision,
        dropout,
        dropout_decision,
    ) in (nose_down, nose_up):
        assert clean.aperture_half_size_norm == pytest.approx((0.42, 0.34))
        assert clean.aperture_propagated is False
        assert clean.aperture_dynamics_qualified
        assert clean_decision.current_aperture_propagated is False
        assert clean_decision.current_aperture_dynamics_qualified
        for state, decision in (
            (censored, censored_decision),
            (dropout, dropout_decision),
        ):
            assert state.aperture_propagated
            assert state.aperture_dynamics_qualified
            assert state.aperture_half_size_norm is not None
            assert decision.current_aperture_propagated
            assert decision.current_aperture_dynamics_qualified
            assert decision.current_aperture_half_size_norm is not None
            assert (
                decision.current_aperture_prediction_horizon_remaining_s
                > 0.0
            )
            assert all(
                math.isfinite(value)
                for value in (
                    *state.bearing_rad,
                    *state.aperture_half_size_norm,
                    state.log_scale,
                    *decision.current_crossing_error_q,
                    *decision.crossing_rate_q_s,
                    decision.command.target_roll_rad,
                    decision.command.target_pitch_rad,
                    decision.command.yaw_rate_rad_s,
                    decision.command.thrust,
                )
            )
        assert all(
            censored.bearing_std_rad[axis]
            > clean.bearing_std_rad[axis]
            for axis in range(2)
        )
        assert all(
            dropout.bearing_std_rad[axis]
            > censored.bearing_std_rad[axis]
            for axis in range(2)
        )
        assert censored.log_scale_std > clean.log_scale_std
        assert dropout.log_scale_std > censored.log_scale_std

    nose_down_clean, _, nose_down_censored, *_ = nose_down
    nose_up_clean, _, nose_up_censored, *_ = nose_up
    assert nose_down_clean.aperture_half_size_norm == pytest.approx(
        nose_up_clean.aperture_half_size_norm
    )
    assert nose_down_censored.log_scale > nose_up_censored.log_scale
    assert (
        nose_down_censored.aperture_half_size_norm[0]
        / nose_down_clean.aperture_half_size_norm[0]
        > nose_up_censored.aperture_half_size_norm[0]
        / nose_up_clean.aperture_half_size_norm[0]
    )
    assert nose_down_censored.bearing_rad[1] > (
        nose_up_censored.bearing_rad[1]
    )


def test_clean_ttc_dropout_cannot_shorten_an_earned_aperture_deadline(
) -> None:
    core = DynamicCourseCore(
        DynamicCourseConfig(camera_delay_s=0.0, scale_beta=1.0)
    )
    states = []
    for sequence, log_scale in enumerate(
        (-1.00, -0.94, -0.88, -0.82, -1.10, -1.20),
        start=1,
    ):
        observation_time = 1.0 + (sequence - 1) * 0.030
        _imu(core, observation_time)
        states.append(
            core.observe_track(
                _observation(
                    "gate-a",
                    sequence,
                    observation_time,
                    log_scale=log_scale,
                )
            )
        )

    qualified = states[-2]
    lost_ttc = states[-1]
    assert qualified.time_to_contact_s is not None
    assert qualified.aperture_prediction_deadline_monotonic_ns is not None
    assert lost_ttc.scale_rate_qualified
    assert lost_ttc.time_to_contact_s is None
    assert lost_ttc.aperture_propagated is False
    assert (
        lost_ttc.aperture_prediction_deadline_monotonic_ns
        >= qualified.aperture_prediction_deadline_monotonic_ns
    )
    assert lost_ttc.aperture_prediction_deadline_monotonic_ns > (
        lost_ttc.state_monotonic_ns
        + round(core.config.dropout_hold_s * NS)
    )


def test_clean_unqualified_aperture_separates_steering_horizon_from_passage(
) -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    _imu(core, 1.0)
    seeded = core.observe_track(_observation("gate-a", 1, 1.0))
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id=None,
    )
    _imu(core, 1.01)
    decision = core.guide(1_010_000_000)

    assert not seeded.scale_rate_qualified
    assert not seeded.aperture_dynamics_qualified
    assert seeded.time_to_contact_s is None
    assert seeded.aperture_prediction_deadline_monotonic_ns == (
        seeded.state_monotonic_ns
        + round(core.config.crossing_prediction_max_horizon_s * NS)
    )
    assert decision.current_aperture_half_size_norm is not None
    assert not decision.current_aperture_dynamics_qualified
    assert decision.current_aperture_prediction_horizon_remaining_s > 1.0
    assert decision.current_time_to_contact_s is None
    assert decision.crossing_prediction_horizon_s == 0.0


def test_degraded_inner_scale_qualifies_existing_aperture_without_reseeding(
) -> None:
    config = DynamicCourseConfig(
        camera_delay_s=0.0,
        crossing_prediction_max_horizon_s=0.20,
    )

    def run(*, scale_authority: bool) -> tuple[
        TrackDynamicState,
        TrackDynamicState,
    ]:
        core = DynamicCourseCore(config)
        _imu(core, 1.0)
        clean = core.observe_track(
            _observation(
                "gate-a",
                1,
                1.0,
                log_scale=-1.00,
            )
        )
        state = clean
        for sequence, log_scale in enumerate(
            (-0.94, -0.88, -0.82),
            start=2,
        ):
            time_s = 1.0 + (sequence - 1) * 0.030
            _imu(core, time_s)
            state = core.observe_track(
                _observation(
                    "gate-a",
                    sequence,
                    time_s,
                    log_scale=log_scale,
                    aperture=None,
                    inner_scale_measurement_usable=scale_authority,
                )
            )
        return clean, state

    clean, corrected = run(scale_authority=True)
    outer_clean, outer_only = run(scale_authority=False)

    assert corrected.aperture_half_size_norm is not None
    assert corrected.aperture_propagated
    assert corrected.scale_rate_qualified
    assert corrected.aperture_dynamics_qualified
    assert corrected.expansion_rate_s > 0.0
    assert corrected.time_to_contact_s is not None
    assert corrected.aperture_seed_monotonic_ns == (
        clean.aperture_seed_monotonic_ns
    )
    assert corrected.aperture_prediction_deadline_monotonic_ns > (
        clean.aperture_prediction_deadline_monotonic_ns
    )
    assert corrected.log_scale_std <= outer_only.log_scale_std

    assert outer_only.aperture_half_size_norm is not None
    assert outer_only.aperture_propagated
    assert not outer_only.scale_rate_qualified
    assert not outer_only.aperture_dynamics_qualified
    assert outer_only.aperture_seed_monotonic_ns == (
        outer_clean.aperture_seed_monotonic_ns
    )
    assert outer_only.aperture_prediction_deadline_monotonic_ns == (
        outer_clean.aperture_prediction_deadline_monotonic_ns
    )


def test_fresh_degraded_inner_rehydrates_expired_clean_lineage_for_steering(
) -> None:
    config = DynamicCourseConfig(
        camera_delay_s=0.0,
        crossing_prediction_max_horizon_s=0.05,
    )

    def run(*, scale_authority: bool) -> tuple[
        TrackDynamicState,
        TrackDynamicState,
        TrackDynamicState,
        GuidanceDecision,
    ]:
        core = DynamicCourseCore(config)
        core.record_applied_command(_command(0.90))
        anchor_log_scale = 0.5 * math.log(0.12 * 0.15)
        _imu(core, 1.0)
        clean = core.observe_track(
            _observation(
                "gate-a",
                1,
                1.0,
                log_scale=anchor_log_scale,
                aperture=(0.12, 0.15),
            )
        )
        core.bind(
            current_gate_index=0,
            current_track_id="gate-a",
            successor_track_id=None,
        )
        _imu(core, 1.06)
        expired = core.observe_track(
            _observation(
                "gate-a",
                2,
                1.06,
                log_scale=anchor_log_scale + 0.04,
                aperture=None,
            )
        )
        _imu(core, 1.09)
        corrected = core.observe_track(
            _observation(
                "gate-a",
                3,
                1.09,
                x=0.08,
                y=-0.06,
                log_scale=anchor_log_scale + 0.10,
                aperture=None,
                confidence=0.20,
                inner_scale_measurement_usable=scale_authority,
            )
        )
        _imu(core, 1.095)
        return clean, expired, corrected, core.guide(1_095_000_000)

    clean, expired, rehydrated, decision = run(scale_authority=True)
    _, outer_expired, outer_only, outer_decision = run(
        scale_authority=False
    )

    assert expired.aperture_half_size_norm is None
    assert outer_expired.aperture_half_size_norm is None
    assert rehydrated.aperture_half_size_norm is not None
    assert rehydrated.aperture_half_size_norm == pytest.approx(
        tuple(
            value * math.exp(rehydrated.log_scale - clean.log_scale)
            for value in clean.aperture_half_size_norm or ()
        )
    )
    assert rehydrated.aperture_seed_monotonic_ns == (
        clean.aperture_seed_monotonic_ns
    )
    assert rehydrated.aperture_prediction_deadline_monotonic_ns == (
        rehydrated.state_monotonic_ns
        + round(
            min(
                config.post_credit_current_prediction_max_horizon_s,
                config.crossing_prediction_max_horizon_s,
            )
            * NS
        )
    )
    assert rehydrated.aperture_propagated
    assert not rehydrated.aperture_dynamics_qualified
    assert rehydrated.confidence <= 0.20
    assert rehydrated.log_scale_std >= expired.log_scale_std
    assert all(
        math.isfinite(value) and value > 0.0
        for value in rehydrated.aperture_half_size_norm
    )
    assert decision.current_aperture_half_size_norm is not None
    assert not decision.current_aperture_dynamics_qualified
    assert decision.crossing_prediction_horizon_s == 0.0
    assert all(
        math.isfinite(value)
        for value in decision.terminal_crossing_clearance_norm
    )

    assert outer_only.aperture_half_size_norm is None
    assert outer_only.aperture_seed_monotonic_ns is None
    assert outer_only.aperture_prediction_deadline_monotonic_ns is None
    assert not outer_only.aperture_propagated
    assert outer_decision.current_aperture_half_size_norm is None


def test_graph_vetted_successor_handoff_preserves_roles_and_bounded_state(
) -> None:
    core = DynamicCourseCore(
        DynamicCourseConfig(
            camera_delay_s=0.0,
            crossing_prediction_max_horizon_s=0.20,
        )
    )
    for sequence, log_scale in enumerate(
        (-1.00, -0.94, -0.88, -0.82),
        start=1,
    ):
        time_s = 1.0 + (sequence - 1) * 0.030
        _imu(core, time_s)
        if sequence == 1:
            core.observe_track(
                _observation(
                    "current",
                    sequence,
                    time_s,
                    x=0.0,
                    y=0.0,
                )
            )
        core.observe_track(
            _observation(
                "successor-old",
                sequence,
                time_s,
                x=0.35 + 0.01 * sequence,
                y=-0.18,
                log_scale=log_scale,
                aperture=(0.10, 0.12),
            )
        )
    core.bind(
        current_gate_index=0,
        current_track_id="current",
        successor_track_id="successor-old",
    )
    source = core.course_state().successor
    assert source is not None
    assert source.aperture_dynamics_qualified

    _imu(core, 1.16)
    core.observe_track(
        _observation(
            "successor-old",
            5,
            1.16,
            visible=False,
        )
    )
    replacement = core.observe_track(
        _observation(
            "successor-new",
            1,
            1.16,
            x=0.42,
            y=-0.19,
            log_scale=-0.74,
            aperture=None,
            inner_scale_measurement_usable=True,
        )
    )
    before = core.course_state()
    transferred = core.handoff_graph_vetted_successor_state(
        predecessor_track_id="successor-old",
        replacement_track_id="successor-new",
    )
    after_transfer = core.course_state()

    assert transferred is not None
    assert transferred.track_id == "successor-new"
    assert transferred.aperture_half_size_norm is not None
    assert transferred.aperture_propagated
    assert transferred.aperture_dynamics_qualified
    assert transferred.aperture_seed_monotonic_ns == (
        source.aperture_seed_monotonic_ns
    )
    assert transferred.aperture_prediction_deadline_monotonic_ns > (
        source.aperture_prediction_deadline_monotonic_ns
    )
    assert all(
        transferred.bearing_std_rad[axis]
        >= replacement.bearing_std_rad[axis]
        for axis in range(2)
    )
    assert transferred.log_scale_std >= replacement.log_scale_std
    assert after_transfer.current_gate_index == before.current_gate_index
    assert after_transfer.current_track_id == before.current_track_id
    assert after_transfer.successor_track_id == before.successor_track_id
    assert after_transfer.promotion_count == before.promotion_count == 0

    rebound = core.bind(
        current_gate_index=0,
        current_track_id="current",
        successor_track_id="successor-new",
    )
    assert rebound.current_gate_index == 0
    assert rebound.current_track_id == "current"
    assert rebound.successor_track_id == "successor-new"
    assert rebound.successor is not None
    assert rebound.successor.aperture_propagated
    assert rebound.promotion_count == 0


def test_local_aperture_is_withdrawn_at_decision_time_expiry() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    _imu(core, 1.0)
    seeded = core.observe_track(_observation("gate-a", 1, 1.0))
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id=None,
    )
    assert seeded.aperture_prediction_deadline_monotonic_ns is not None
    expired_ns = (
        seeded.aperture_prediction_deadline_monotonic_ns + 1_000_000
    )
    _imu(core, expired_ns / NS)
    decision = core.guide(expired_ns)

    assert seeded.aperture_half_size_norm is not None
    assert decision.current_aperture_half_size_norm is None
    assert decision.current_aperture_prediction_horizon_remaining_s == 0.0
    assert decision.crossing_allowance_norm == (0.0, 0.0)
    assert decision.centered_crossing_clearance_norm == (0.0, 0.0)
    assert decision.predicted_crossing_clearance_norm == (0.0, 0.0)
    assert decision.terminal_crossing_clearance_norm == (0.0, 0.0)


@pytest.mark.parametrize(
    ("seed_aperture", "clipping", "ambiguous"),
    (
        (False, FrameEdge.TOP, False),
        (True, FrameEdge.TOP, True),
    ),
)
def test_local_aperture_rejects_unseeded_or_ambiguous_censor(
    seed_aperture: bool,
    clipping: FrameEdge,
    ambiguous: bool,
) -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    _imu(core, 1.0)
    core.observe_track(
        _observation(
            "gate-a",
            1,
            1.0,
            aperture=(0.42, 0.34) if seed_aperture else None,
        )
    )
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id=None,
    )
    _imu(core, 1.04)
    rejected = core.observe_track(
        _observation(
            "gate-a",
            2,
            1.04,
            aperture=None,
            clipping=clipping,
            center_censored=True,
            ambiguous=ambiguous,
        )
    )
    _imu(core, 1.045)
    decision = core.guide(1_045_000_000)

    assert rejected.aperture_half_size_norm is None
    assert rejected.aperture_seed_monotonic_ns is None
    assert rejected.aperture_prediction_deadline_monotonic_ns is None
    assert rejected.aperture_propagated is False
    assert rejected.aperture_dynamics_qualified is False
    assert decision.current_aperture_half_size_norm is None
    assert decision.crossing_allowance_norm == (0.0, 0.0)


def test_degraded_scale_cannot_erase_qualified_local_aperture_state() -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.80))
    qualified = None
    for sequence, log_scale in enumerate(
        (-0.80, -0.72, -0.64, -0.56),
        start=1,
    ):
        observation_time = 1.0 + (sequence - 1) * 0.030
        _imu(core, observation_time)
        qualified = core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                log_scale=log_scale,
            )
        )
    assert qualified is not None
    assert qualified.aperture_dynamics_qualified
    assert qualified.expansion_rate_s > 0.0
    assert qualified.time_to_contact_s is not None

    _imu(core, 1.12)
    degraded = core.observe_track(
        _observation(
            "gate-a",
            5,
            1.12,
            x=0.08,
            y=-0.06,
            log_scale=0.80,
            aperture=None,
            confidence=0.20,
        )
    )
    assert degraded.aperture_propagated
    assert degraded.aperture_dynamics_qualified
    assert degraded.raw_log_scale is None
    assert degraded.log_scale != pytest.approx(0.80)
    assert degraded.expansion_rate_s > 0.0
    assert degraded.confidence == pytest.approx(qualified.confidence)
    assert degraded.aperture_seed_monotonic_ns == (
        qualified.aperture_seed_monotonic_ns
    )
    assert degraded.aperture_prediction_deadline_monotonic_ns == (
        qualified.aperture_prediction_deadline_monotonic_ns
    )

    _imu(core, 1.15)
    corrected = core.observe_track(
        _observation(
            "gate-a",
            6,
            1.15,
            x=0.09,
            y=-0.05,
            log_scale=-0.48,
        )
    )
    assert not corrected.aperture_propagated
    assert not corrected.scale_rate_qualified
    assert corrected.aperture_dynamics_qualified
    assert corrected.expansion_rate_s > 0.0
    assert corrected.time_to_contact_s is not None

    _imu(core, 1.18)
    full_frame = core.observe_track(
        _observation(
            "gate-a",
            7,
            1.18,
            x=0.0,
            y=0.0,
            log_scale=0.0,
            aperture=None,
            clipping=(
                FrameEdge.LEFT
                | FrameEdge.TOP
                | FrameEdge.RIGHT
                | FrameEdge.BOTTOM
            ),
            center_censored=True,
            confidence=0.0,
        )
    )
    assert full_frame.aperture_propagated
    assert full_frame.aperture_dynamics_qualified
    assert full_frame.confidence == pytest.approx(corrected.confidence)
    assert full_frame.aperture_prediction_deadline_monotonic_ns == (
        corrected.aperture_prediction_deadline_monotonic_ns
    )
    assert full_frame.bearing_std_rad[0] > corrected.bearing_std_rad[0]
    assert full_frame.bearing_std_rad[1] > corrected.bearing_std_rad[1]


def test_invisible_ambiguity_revokes_aperture_lineage_until_clean_reseed(
) -> None:
    core = DynamicCourseCore(DynamicCourseConfig(camera_delay_s=0.0))
    core.record_applied_command(_command(0.90))
    _imu(core, 1.0)
    seeded = core.observe_track(_observation("gate-a", 1, 1.0))
    assert seeded.aperture_half_size_norm is not None

    _imu(core, 1.04)
    ambiguous = core.observe_track(
        _observation(
            "gate-a",
            2,
            1.04,
            visible=False,
            ambiguous=True,
        )
    )
    assert ambiguous.aperture_half_size_norm is None
    assert ambiguous.aperture_seed_monotonic_ns is None
    assert ambiguous.aperture_prediction_deadline_monotonic_ns is None

    _imu(core, 1.08)
    degraded = core.observe_track(
        _observation(
            "gate-a",
            3,
            1.08,
            aperture=None,
            clipping=FrameEdge.TOP,
            center_censored=True,
        )
    )
    assert degraded.aperture_half_size_norm is None
    assert degraded.aperture_seed_monotonic_ns is None
    assert degraded.aperture_prediction_deadline_monotonic_ns is None
    assert degraded.aperture_propagated is False

    _imu(core, 1.12)
    fresh_degraded_inner = core.observe_track(
        _observation(
            "gate-a",
            4,
            1.12,
            aperture=None,
            inner_scale_measurement_usable=True,
        )
    )
    assert fresh_degraded_inner.aperture_half_size_norm is None
    assert fresh_degraded_inner.aperture_seed_monotonic_ns is None
    assert (
        fresh_degraded_inner.aperture_prediction_deadline_monotonic_ns
        is None
    )
    assert not fresh_degraded_inner.aperture_propagated


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
                x=-0.03,
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


def test_successor_dropout_retains_local_rate_for_reacquisition_continuity(
) -> None:
    core = DynamicCourseCore(
        DynamicCourseConfig(
            camera_delay_s=0.0,
            successor_clearance_dwell_s=0.02,
            successor_clearance_ramp_s=0.02,
        )
    )
    core.record_applied_command(_command(0.90))
    decisions = []
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
                x=0.04 + 0.02 * sequence,
                log_scale=-0.90,
            )
        )
        if sequence == 1:
            core.bind(
                current_gate_index=0,
                current_track_id="gate-a",
                successor_track_id="gate-b",
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        decision = core.guide(round(decision_time * NS))
        decisions.append(decision)
        _commit_decision(core, decision_time, decision.command)
    stable = decisions[-1]
    assert stable.successor_prediction_confidence > 0.0
    assert stable.successor_rate_rad_s is not None
    assert stable.successor_rate_rad_s[0] > 0.0
    assert stable.passage_point_norm[0] > 0.0

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
    assert dropped.successor_clearance_authority == 0.0
    assert dropped.successor_passage_authority == 0.0
    assert dropped.passage_point_norm == (0.0, 0.0)
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
    committed = core.guide(
        1_210_000_000,
        passage_committed=True,
    )
    assert committed.passage_committed
    assert committed.committed_successor_roll_authority == 1.0
    assert committed.committed_successor_target_roll_rad is not None
    assert committed.command.target_roll_rad == pytest.approx(
        committed.committed_successor_target_roll_rad
    )
    assert committed.committed_successor_pitch_authority == 1.0
    assert committed.committed_successor_target_pitch_rad is not None
    assert committed.command.target_pitch_rad == pytest.approx(
        committed.committed_successor_target_pitch_rad
    )
    assert committed.committed_successor_yaw_authority == 1.0
    assert committed.committed_successor_yaw_rate_rad_s is not None
    assert committed.command.yaw_rate_rad_s < 0.0
    assert committed.command.yaw_rate_rad_s == pytest.approx(
        committed.committed_successor_yaw_rate_rad_s
    )
    assert committed.command.thrust == pytest.approx(
        dropped.command.thrust
    )
    assert committed.successor_passage_authority == 0.0
    assert committed.successor_clearance_authority == 0.0
    assert committed.passage_point_norm == (0.0, 0.0)
    assert all(
        math.isfinite(value)
        for value in (
            committed.command.target_roll_rad,
            committed.command.target_pitch_rad,
            committed.command.yaw_rate_rad_s,
            committed.command.thrust,
        )
    )
    assert (
        abs(committed.command.yaw_rate_rad_s)
        <= MAX_YAW_RATE_RAD_S
    )
    assert abs(committed.command.target_roll_rad) <= MAX_TARGET_ROLL_RAD
    assert (
        MIN_TARGET_PITCH_RAD
        <= committed.command.target_pitch_rad
        <= MAX_TARGET_PITCH_RAD
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
            x=0.16,
            log_scale=-0.85,
        )
    )
    _imu(core, 1.25)
    reacquired = core.guide(1_250_000_000)

    assert reacquired.successor_prediction_confidence > 0.0
    assert reacquired.successor_rate_rad_s is not None
    assert reacquired.successor_rate_rad_s[0] > 0.0
    assert reacquired.successor_rate_rad_s[0] == pytest.approx(
        stable.successor_rate_rad_s[0],
        rel=0.25,
    )
    assert reacquired.predicted_successor_bearing_rad is not None
    assert reacquired.measured_successor_bearing_rad is not None
    assert reacquired.predicted_successor_bearing_rad[0] >= (
        reacquired.measured_successor_bearing_rad[0]
    )
    assert not reacquired.successor_transition_held
    # Rate memory returns immediately, while passage/yaw ownership must earn
    # a new clean current-aperture clearance dwell.
    assert reacquired.successor_clearance_authority == 0.0
    assert reacquired.passage_point_norm == (0.0, 0.0)
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
    _imu(core, 1.60)
    assert not core.retains_successor_lineage(
        "gate-b",
        1_600_000_000,
    )
    expired_committed = core.guide(
        1_600_000_000,
        passage_committed=True,
    )
    assert expired_committed.passage_committed
    assert expired_committed.committed_successor_roll_authority == 1.0
    assert expired_committed.committed_successor_target_roll_rad is not None
    assert expired_committed.command.target_roll_rad == pytest.approx(
        expired_committed.committed_successor_target_roll_rad
    )
    assert expired_committed.committed_successor_pitch_authority == 1.0
    assert expired_committed.committed_successor_target_pitch_rad is not None
    assert expired_committed.command.target_pitch_rad == pytest.approx(
        expired_committed.committed_successor_target_pitch_rad
    )
    assert expired_committed.committed_successor_yaw_authority == 1.0
    assert expired_committed.committed_successor_yaw_rate_rad_s is not None
    assert expired_committed.command.yaw_rate_rad_s == pytest.approx(
        expired_committed.committed_successor_yaw_rate_rad_s
    )
    assert expired_committed.command.thrust == pytest.approx(
        expired.command.thrust
    )
    assert all(
        math.isfinite(value)
        for value in (
            expired_committed.command.target_roll_rad,
            expired_committed.command.target_pitch_rad,
            expired_committed.command.yaw_rate_rad_s,
            expired_committed.command.thrust,
        )
    )
    assert (
        abs(expired_committed.command.yaw_rate_rad_s)
        <= MAX_YAW_RATE_RAD_S
    )
    assert (
        abs(expired_committed.command.target_roll_rad)
        <= MAX_TARGET_ROLL_RAD
    )
    assert (
        MIN_TARGET_PITCH_RAD
        <= expired_committed.command.target_pitch_rad
        <= MAX_TARGET_PITCH_RAD
    )

    retained_successor = core.course_state().successor
    assert retained_successor is not None
    committed_deadline_ns = (
        retained_successor.last_measurement_monotonic_ns
        + round(core.config.crossing_prediction_max_horizon_s * NS)
    )
    _imu(core, (committed_deadline_ns + 1) / NS)
    bounded_expiry = core.guide(
        committed_deadline_ns + 1,
        passage_committed=True,
    )
    assert bounded_expiry.passage_committed
    assert bounded_expiry.committed_successor_roll_authority == 0.0
    assert bounded_expiry.committed_successor_target_roll_rad is None
    assert bounded_expiry.committed_successor_pitch_authority == 0.0
    assert bounded_expiry.committed_successor_target_pitch_rad is None
    assert bounded_expiry.committed_successor_yaw_authority == 0.0
    assert bounded_expiry.committed_successor_yaw_rate_rad_s is None
    assert bounded_expiry.committed_successor_camera_center_norm is None
    assert (
        bounded_expiry.committed_successor_camera_center_rate_norm_s
        is None
    )


def test_committed_off_axis_successor_uses_full_bank_then_releases(
) -> None:
    config = DynamicCourseConfig(
        camera_delay_s=0.0,
        bearing_alpha=1.0,
        bearing_beta=1.0,
        scale_alpha=1.0,
        scale_beta=1.0,
        roll_guidance_sign=1.0,
        roll_gain=0.18,
        lateral_rate_gain=0.045,
    )
    core = DynamicCourseCore(config)
    core.record_applied_command(_command(0.90))
    outward = None
    for sequence in range(1, 8):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation("gate-a", sequence, observation_time)
        )
        successor = core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=0.35 + (sequence - 1) * 0.004,
                y=-0.40,
                log_scale=-0.80,
            )
        )
        if sequence == 1:
            core.bind(
                current_gate_index=0,
                current_track_id="gate-a",
                successor_track_id="gate-b",
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        outward = core.guide(
            round(decision_time * NS),
            passage_committed=True,
        )
        _commit_decision(core, decision_time, outward.command)

    assert outward is not None
    assert successor.bearing_std_rad[0] <= (
        config.successor_prediction_max_extrapolation_rad
    )
    assert abs(successor.bearing_rad[0]) >= config.off_axis_brake_rad
    assert (
        successor.bearing_rad[0]
        * successor.bearing_rate_rad_s[0]
        > 0.0
    )
    assert outward.committed_successor_roll_authority == 1.0
    assert outward.committed_successor_target_roll_rad == pytest.approx(
        MAX_TARGET_ROLL_RAD
    )
    assert outward.command.target_roll_rad == pytest.approx(
        MAX_TARGET_ROLL_RAD
    )
    assert outward.committed_successor_target_pitch_rad == pytest.approx(
        config.brake_pitch_rad
    )
    assert outward.command.target_pitch_rad == pytest.approx(
        config.brake_pitch_rad
    )

    recovered = None
    for sequence, x in ((8, 0.370), (9, 0.366)):
        observation_time = 1.0 + (sequence - 1) * 0.040
        _imu(core, observation_time)
        core.observe_track(
            _observation("gate-a", sequence, observation_time)
        )
        successor = core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=x,
                y=-0.40,
                log_scale=-0.80,
            )
        )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        recovered = core.guide(
            round(decision_time * NS),
            passage_committed=True,
        )
        _commit_decision(core, decision_time, recovered.command)

    assert recovered is not None
    assert abs(successor.bearing_rad[0]) >= config.off_axis_brake_rad
    assert successor.bearing_rate_rad_s[0] < 0.0
    assert recovered.committed_successor_roll_authority == 1.0
    assert recovered.committed_successor_target_roll_rad is not None
    assert (
        0.0
        < recovered.committed_successor_target_roll_rad
        < MAX_TARGET_ROLL_RAD
    )
    assert recovered.command.target_roll_rad == pytest.approx(
        recovered.committed_successor_target_roll_rad
    )
    assert recovered.committed_successor_target_pitch_rad is not None
    assert (
        recovered.committed_successor_target_pitch_rad
        < config.brake_pitch_rad
    )


def _precommit_successor_case(
    *,
    successor_x_step: float,
    current_x: float = 0.0,
    current_log_scales: tuple[float, ...] | None = None,
) -> tuple[DynamicCourseCore, GuidanceDecision]:
    config = DynamicCourseConfig(
        camera_delay_s=0.0,
        bearing_alpha=1.0,
        bearing_beta=1.0,
        scale_alpha=1.0,
        scale_beta=1.0,
        roll_guidance_sign=1.0,
        roll_gain=0.18,
        lateral_rate_gain=0.045,
    )
    core = DynamicCourseCore(config)
    core.record_applied_command(_command(0.90))
    decision = None
    for sequence in range(1, 9):
        observation_time = 1.0 + (sequence - 1) * 0.040
        log_scale = (
            -1.20 + (sequence - 1) * 0.03
            if current_log_scales is None
            else current_log_scales[sequence - 1]
        )
        _imu(core, observation_time)
        core.observe_track(
            _observation(
                "gate-a",
                sequence,
                observation_time,
                x=current_x,
                y=-0.30,
                log_scale=log_scale,
            )
        )
        core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=0.40 + (sequence - 1) * successor_x_step,
                y=-0.40,
                log_scale=-1.60,
            )
        )
        if sequence == 1:
            core.bind(
                current_gate_index=0,
                current_track_id="gate-a",
                successor_track_id="gate-b",
            )
        decision_time = observation_time + 0.005
        _imu(core, decision_time)
        decision = core.guide(round(decision_time * NS))
        _commit_decision(core, decision_time, decision.command)
    assert decision is not None
    return core, decision


def test_settled_current_allows_only_bounded_precommit_successor_roll() -> None:
    _, decision = _precommit_successor_case(successor_x_step=0.004)

    assert not decision.passage_committed
    assert decision.centered_crossing_clearance_norm[1] < 0.0
    assert decision.successor_clearance_authority == 0.0
    assert decision.successor_passage_authority == 0.0
    assert decision.passage_yaw_authority == 0.0
    assert decision.successor_yaw_contribution_rad == 0.0
    assert 0.0 < decision.precommit_successor_roll_authority <= 1.0
    assert decision.precommit_successor_target_roll_rad is not None
    assert (
        0.0
        < decision.precommit_successor_target_roll_rad
        <= MAX_TARGET_ROLL_RAD
    )
    assert decision.command.target_roll_rad == pytest.approx(
        decision.precommit_successor_target_roll_rad
    )
    assert decision.command.target_pitch_rad == pytest.approx(
        DynamicCourseConfig().brake_pitch_rad
    )
    assert decision.command.yaw_rate_rad_s == 0.0
    assert decision.command.thrust == pytest.approx(SUPPORT_THRUST)
    assert all(
        math.isfinite(value)
        for value in (
            decision.command.target_roll_rad,
            decision.command.target_pitch_rad,
            decision.command.yaw_rate_rad_s,
            decision.command.thrust,
        )
    )


def test_precommit_successor_roll_releases_or_yields_to_current() -> None:
    _, recovered = _precommit_successor_case(
        successor_x_step=-0.004,
    )
    _, opposing_current = _precommit_successor_case(
        successor_x_step=0.004,
        current_x=-0.06,
    )

    assert recovered.successor_rate_rad_s is not None
    assert recovered.successor_rate_rad_s[0] < 0.0
    assert recovered.precommit_successor_roll_authority == 0.0
    assert recovered.precommit_successor_target_roll_rad is None

    assert opposing_current.current_center_norm[0] < 0.0
    assert opposing_current.command.target_roll_rad < 0.0
    assert opposing_current.precommit_successor_roll_authority == 0.0
    assert opposing_current.precommit_successor_target_roll_rad is None


def test_precommit_roll_retains_closure_seed_but_expires_stale_vision() -> None:
    core, retained = _precommit_successor_case(
        successor_x_step=0.004,
        current_log_scales=(
            -1.20,
            -1.17,
            -1.14,
            -1.11,
            -1.11,
            -1.11,
            -1.11,
            -1.11,
        ),
    )

    assert retained.current_time_to_contact_s is None
    assert retained.precommit_successor_roll_authority > 0.0
    assert retained.precommit_successor_target_roll_rad is not None

    _imu(core, 1.41)
    core.observe_track(
        _observation(
            "gate-a",
            9,
            1.41,
            y=-0.30,
            log_scale=-1.11,
        )
    )
    _imu(core, 1.415)
    expired = core.guide(1_415_000_000)

    assert expired.precommit_successor_roll_authority == 0.0
    assert expired.precommit_successor_target_roll_rad is None


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

    assert first.current.track_id == preserved["gate-b"].track_id
    assert second.current.track_id == preserved["gate-c"].track_id
    assert third.current.track_id == preserved["gate-d"].track_id
    assert first.current.frame_sequence == preserved["gate-b"].frame_sequence
    assert second.current.frame_sequence == preserved["gate-c"].frame_sequence
    assert third.current.frame_sequence == preserved["gate-d"].frame_sequence
    assert first.current.aperture_seed_monotonic_ns == (
        preserved["gate-b"].aperture_seed_monotonic_ns
    )
    assert second.current.aperture_seed_monotonic_ns == (
        preserved["gate-c"].aperture_seed_monotonic_ns
    )
    assert third.current.aperture_seed_monotonic_ns == (
        preserved["gate-d"].aperture_seed_monotonic_ns
    )
    assert third.current_gate_index == 3
    assert third.current_track_id == "gate-d"
    assert third.successor_track_id is None
    assert third.promotion_count == 3


def test_authoritative_promotion_reanchors_large_yaw_without_optical_jump() -> None:
    config = DynamicCourseConfig(camera_delay_s=0.0)
    core = DynamicCourseCore(config)
    core.record_applied_command(_command(0.90))

    def projected_center(
        world_horizontal_rad: float,
        yaw_rad: float,
    ) -> tuple[float, float]:
        world_ray = (
            1.0,
            math.tan(world_horizontal_rad),
            math.tan(-0.18),
        )
        camera_forward = (
            math.cos(yaw_rad) * world_ray[0]
            + math.sin(yaw_rad) * world_ray[1]
        )
        camera_right = (
            -math.sin(yaw_rad) * world_ray[0]
            + math.cos(yaw_rad) * world_ray[1]
        )
        assert camera_forward > 0.0
        return (
            camera_right
            / camera_forward
            / config.horizontal_angle_scale_rad,
            world_ray[2]
            / camera_forward
            / config.vertical_angle_scale_rad,
        )

    successor = None
    last_center = None
    for sequence in range(1, 7):
        observation_time = 1.0 + (sequence - 1) * 0.04
        yaw = (sequence - 1) * 0.20
        world_horizontal = 1.20 + (sequence - 1) * 0.016
        last_center = projected_center(world_horizontal, yaw)
        _imu(core, observation_time, yaw_rad=yaw)
        if sequence == 1:
            core.observe_track(
                _observation(
                    "gate-a",
                    sequence,
                    observation_time,
                    x=0.0,
                    y=0.0,
                )
            )
        successor = core.observe_track(
            _observation(
                "gate-b",
                sequence,
                observation_time,
                x=last_center[0],
                y=last_center[1],
            )
        )
    assert successor is not None
    assert last_center is not None
    core.bind(
        current_gate_index=0,
        current_track_id="gate-a",
        successor_track_id="gate-b",
    )
    before_world_ray = _quat_rotate(
        successor.reference_camera_to_world_wxyz,
        _bearing_ray(successor.bearing_rad),
    )
    before_prediction = core.predict_track_steering(
        "gate-b",
        successor.state_monotonic_ns,
    )

    promoted = core.promote_authoritative(
        from_gate_index=0,
        to_gate_index=1,
        promoted_track_id="gate-b",
        next_successor_track_id=None,
        monotonic_ns=1_205_000_000,
    )
    after_world_ray = _quat_rotate(
        promoted.current.reference_camera_to_world_wxyz,
        _bearing_ray(promoted.current.bearing_rad),
    )
    after_prediction = core.predict_track_steering(
        "gate-b",
        successor.state_monotonic_ns,
    )

    assert after_world_ray == pytest.approx(before_world_ray, abs=1e-12)
    assert after_prediction.camera_center_norm == pytest.approx(
        before_prediction.camera_center_norm,
        abs=1e-12,
    )
    assert after_prediction.camera_center_rate_norm_s == pytest.approx(
        before_prediction.camera_center_rate_norm_s,
        abs=1e-10,
    )
    assert abs(promoted.current.bearing_rad[0]) < 0.40
    assert abs(promoted.current.bearing_rad[1]) < 0.20
    assert promoted.current.raw_center_norm == pytest.approx(last_center)
    assert promoted.current.aperture_half_size_norm == pytest.approx(
        successor.aperture_half_size_norm
    )

    _imu(core, 1.24, yaw_rad=1.10)
    decision = core.guide(1_240_000_000)
    assert all(
        math.isfinite(value)
        for value in (
            *decision.current_center_norm,
            *decision.camera_current_center_norm,
            *decision.current_aperture_half_size_norm,
            decision.command.target_roll_rad,
            decision.command.target_pitch_rad,
            decision.command.yaw_rate_rad_s,
            decision.command.thrust,
        )
    )
    assert abs(decision.camera_current_center_norm[0]) < 0.50
    assert abs(decision.camera_current_center_norm[1]) < 0.50
    assert abs(decision.current_center_norm[1]) < 0.50
    assert abs(decision.command.target_roll_rad) <= MAX_TARGET_ROLL_RAD
    assert (
        MIN_TARGET_PITCH_RAD
        <= decision.command.target_pitch_rad
        <= MAX_TARGET_PITCH_RAD
    )
    assert abs(decision.command.yaw_rate_rad_s) <= MAX_YAW_RATE_RAD_S
    assert MIN_THRUST <= decision.command.thrust <= MAX_THRUST


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
    assert decision.passage_point_norm == (0.0, 0.0)
    assert decision.successor_clearance_authority == 0.0


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
