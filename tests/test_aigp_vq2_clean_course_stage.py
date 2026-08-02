"""Behavior tests for the build-3385 clean visual-course controller.

The suite protects bounded commands, authoritative race ownership, separate
heading and optical-intercept references, and one continuously carried
optical-passage/collective owner with supporting IMU damping.  Vision loss
reduces authority without reviving historical yaw overlays, thrust margins,
floors, or trim ratchets.
"""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest

from competition.vq2_contracts import FrameEdge
from competition.vq2_visual_tracker import CameraFrameToken
from scripts.aigp_vq2_clean_course_stage import (
    FH_UNTRUSTED_TRIGGER_MPS2,
    LAUNCH_BOOST_DURATION_S,
    LAUNCH_BOOST_THRUST,
    MAX_COURSE_YAW_RATE_RAD_S,
    NEVER_MEASURED_S,
    PENDING_CREDIT_HOLD_S,
    ROTATION_COMP_FOCAL_NORM,
    CleanCourseConfig,
    CleanCourseController,
    CleanCourseRuntime,
    CleanCourseState,
    NavigationOutput,
    _directional_brake_response_authority,
    clamp_final_command,
    run_clean_course_stage,
)

SUPPORT = 0.247  # F49: F48-measured hover support (was 0.275)
SPAWN_PITCH = -0.31  # F49: default spawn-relative pitch base
# Tilt-compensated support at the spawn attitude: command() divides the
# support collective by cos(pitch), so the exact level base at the F49
# neutral attitude (pitch=SPAWN_PITCH in the PD-law tests) is this value.
SPAWN_SUPPORT = SUPPORT / math.cos(SPAWN_PITCH)


def _config(**overrides):
    # The climb offset is gate-0-phase feedforward (like launch boost), so the
    # shared helper disables it to keep the closed-loop vertical-sign tests
    # exact; dedicated climb tests enable it explicitly.
    values = {
        "launch_boost_duration_s": 0.0,
        "gate0_climb_vertical_offset_norm": 0.0,
    }
    values.update(overrides)
    return CleanCourseConfig(**values)


def _expected_vertical_trajectory_terms(controller, path, motion):
    """Mirror the outer position/rate/TTC decomposition for law tests."""

    config = controller.config
    cap = max(
        config.vertical_optical_error_max_far_norm,
        config.vertical_optical_error_max_near_norm,
    )
    position_miss = max(-cap, min(cap, path.error))
    rate_observation = path.image_rate_norm_s
    trajectory_ttc_s = config.passage_ttc_max_s
    if motion is not None:
        authority = max(0.0, min(1.0, motion.projection_authority))
        rate_observation += authority * (
            motion.physical_rate_norm_s - rate_observation
        )
        trajectory_ttc_s += authority * (
            motion.ttc_s - trajectory_ttc_s
        )
    trajectory_ttc_s = max(
        config.passage_ttc_min_s,
        min(config.passage_ttc_max_s, trajectory_ttc_s),
    )
    required_rate = -position_miss / trajectory_ttc_s
    rate_miss = max(
        -cap,
        min(
            cap,
            (rate_observation - required_rate)
            * config.commit_blackout_s,
        ),
    )
    if position_miss * rate_miss < 0.0:
        rate_miss = math.copysign(
            min(abs(rate_miss), 0.5 * abs(position_miss)),
            rate_miss,
        )
    position_delta = (
        -config.vertical_optical_collective_gain
        * path.freshness_authority
        * position_miss
    )
    rate_delta = (
        -config.vertical_optical_collective_gain
        * path.freshness_authority
        * rate_miss
    )
    projection_delta = 0.0
    if motion is not None:
        baseline_intercept = path.error + (
            rate_observation * trajectory_ttc_s
        )
        residual = max(
            -cap,
            min(cap, motion.intercept_error - baseline_intercept),
        )
        projection_delta = (
            -config.vertical_optical_collective_gain
            * motion.control_authority
            * residual
        )
        visual = position_delta + rate_delta
        if visual * projection_delta < 0.0:
            projection_delta = math.copysign(
                min(abs(projection_delta), 0.5 * abs(visual)),
                projection_delta,
            )
    return position_delta, rate_delta, projection_delta


def _track(
    track_id,
    x,
    y,
    *,
    scale=0.10,
    confidence=0.90,
    association_confidence=0.90,
    clipping=FrameEdge.NONE,
    center_censored=False,
    visible=True,
    aperture=None,
):
    return SimpleNamespace(
        track_id=track_id,
        center_norm=(float(x), float(y)),
        # Real bbox semantics: apparent_scale ~= sqrt(w*h) of the normalized
        # box, so the half-span is scale/2 (a scale-0.5 box spans half the
        # frame, not the whole frame).
        bbox_norm=(
            x - scale / 2,
            y - scale / 2,
            x + scale / 2,
            y + scale / 2,
        ),
        apparent_scale=float(scale),
        confidence=float(confidence),
        association_confidence=float(association_confidence),
        clipping=clipping,
        center_censored=bool(center_censored),
        ambiguous=False,
        visible=bool(visible),
        inner_aperture=aperture,
    )


def _update(tracks, frame_id=1):
    tracks = tuple(tracks)
    return SimpleNamespace(
        tracks=tracks,
        visible_track_ids=tuple(t.track_id for t in tracks if t.visible),
        token=("test-stream", frame_id),
    )


def _truss(track_id, x, y, *, width=0.50, height=0.23, confidence=0.90):
    """F48 ceiling-truss geometry: top-censored extreme-aspect slab."""

    track = _track(
        track_id, x, y, confidence=confidence, clipping=FrameEdge.TOP
    )
    track.bbox_norm = (
        x - width / 2,
        y - height / 2,
        x + width / 2,
        y + height / 2,
    )
    track.apparent_scale = math.sqrt(width * height)
    return track


def _tracked_controller(track=None, *, config=None, now=100.0):
    """Controller initialized into TRACK with one observed frame."""

    controller = CleanCourseController(config or _config())
    if track is None:
        track = _track("A", 0.0, 0.0)
    controller.initialize(
        _update([track], frame_id=1),
        gate_index=0,
        fallback_center_norm=(0.0, 0.0),
        fallback_apparent_scale=0.10,
        now_s=now,
    )
    controller.observe(_update([track], frame_id=2), now_s=now + 0.033)
    assert controller.state is CleanCourseState.TRACK
    return controller


def _command(
    controller,
    now,
    *,
    roll=0.0,
    pitch=0.0,
    yaw=None,
    a_up=None,
    fh=None,
    accel_trust=None,
):
    return controller.command(
        now_s=now,
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_rad=yaw,
        world_up_accel_m_s2=a_up,
        horizontal_specific_force_mps2=fh,
        accel_trust=accel_trust,
    )


# ---------------------------------------------------------------------------
# Vertical law
# ---------------------------------------------------------------------------


def test_support_tilt_compensation_is_spawn_relative():
    # F95 (20260730T151817Z-visual-course-3d6ceed0): support was tilt-
    # compensated on ABSOLUTE rpy, but the -0.31 spawn attitude is an rpy
    # frame offset — the body is level there (F38: stationary, span flat).
    # At the -0.577 course brake attitude the law inflated support to
    # 0.2906 where true hover is ~0.269 — an open-loop +0.9 m/s^2 climb
    # bias: every gate-1 leg ballooned vz +0.4..+0.5 at the brake attitude
    # (F90/F91/F93/F94) and F94's gate slid out the bottom of the frame
    # into a ground collision (id 1002).  Compensation must be relative
    # to the level attitude; at level both formulas agree (0.2594).
    # A raw y of +0.427 at this brake attitude compensates back to a
    # world-centered gate, isolating the support calculation from the
    # qualified position and rate feedback.
    controller = _tracked_controller(_track("A", 0.0, 0.427))
    brake_attitude = SPAWN_PITCH - 0.267
    now = 100.10
    out = None
    for _ in range(15):
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        controller.current.last_y_measurement_s = now
        out = _command(controller, now, pitch=brake_attitude)
    # Compensated ey = 0.427 - 0.267*1.6 ~= 0: a world-centered gate at the
    # brake attitude, so the emitted collective IS the tilt-compensated
    # support.  True hover at 0.267 rad from level is
    # (0.247/cos(0.31))/cos(0.267) ~= 0.269; the absolute-rpy formula
    # emits >= 0.29.
    assert out.thrust < 0.28
    assert out.thrust == pytest.approx(0.269, abs=5e-3)


def test_identical_global_vertical_sign_at_every_gate():
    config = _config()
    thrusts = []
    controller = _tracked_controller(_track("A", 0.0, 0.20), config=config)
    now = 100.10
    for gate in (0, 1, 2):
        # F103: the descent setpoint tapers near/below spawn altitude —
        # hold an honest altitude at EVERY gate so this sign-contract test
        # stays on the full-authority vertical law it means to exercise.
        controller._alt_est_m = 2.0  # honest altitude (floor quiet)
        if gate > 0:
            successor_id = f"S{gate}"
            controller.observe(
                _update(
                    [_track("current", 0.0, 0.20), _track(successor_id, 0.0, 0.20)],
                    frame_id=10 + gate,
                ),
                now_s=now,
            )
            # F42: promotion requires a persistent successor; seed the age
            # of every candidate (the old track id disappearing makes the
            # "current"-id track a successor candidate too).
            controller._track_first_seen_s["current"] = now - 1.0
            controller._track_first_seen_s[successor_id] = now - 1.0
            promoted = controller.note_race(
                gate_index=gate, race_boot_ms=1000 + gate, now_s=now
            )
            assert promoted
            assert controller.state is CleanCourseState.TRACK
            # Align the promoted hypothesis with the same vertical error.
            controller.current.y_axis.p = 0.20
            controller.current.confidence = 0.9
            controller.current.last_y_measurement_s = now
            controller.current.last_measurement_s = now
            controller._alt_est_m = 2.0  # honest altitude (floor quiet)
        output = _command(controller, now + 0.02)
        assert output.vertical_qualified
        assert output.gate_index == gate
        thrusts.append(output.thrust)
        now += 0.05
    # Image-down positive error keeps the same correction direction across
    # authoritative promotions.  Magnitude is carried continuously rather
    # than reset at each new leg.
    for thrust in thrusts:
        assert thrust < SPAWN_SUPPORT
    assert max(abs(b - a) for a, b in zip(thrusts, thrusts[1:])) < 0.01


def test_vertical_sign_is_the_gate0_minus_form_by_default():
    # pitch=SPAWN_PITCH: the F49 neutral (level-flight) attitude, so the
    # F50 attitude compensation is exactly zero in these law checks;
    # the exact base is the tilt-compensated SPAWN_SUPPORT.
    # Gate 0 shares the optical passage law.  With zero image motion and
    # zero IMU-vz, the gain acts on the plane miss with the global minus
    # sign (gate low -> less collective).
    controller = _tracked_controller(_track("A", 0.0, 0.20))
    controller._alt_est_m = 2.0  # honest altitude: full descent authority (F103)
    output = _command(controller, 100.10, pitch=SPAWN_PITCH)
    low_delta = output.thrust - SPAWN_SUPPORT
    assert low_delta < 0.0
    assert 0.0 < controller._last_vertical_motion.control_authority < 1.0
    controller = _tracked_controller(_track("A", 0.0, -0.20))
    controller._alt_est_m = 2.0  # honest altitude: full descent authority (F103)
    output = _command(controller, 100.10, pitch=SPAWN_PITCH)
    high_delta = output.thrust - SPAWN_SUPPORT
    assert high_delta > 0.0
    assert high_delta == pytest.approx(-low_delta, abs=1e-9)
    # Position plus the required image-rate trajectory remains bounded by the
    # single global optical envelope; it is intentionally a little stronger
    # than static position alone because zero measured motion cannot erase the
    # miss by the outer-led TTC.
    assert abs(high_delta) < (
        controller.config.vertical_optical_collective_gain
        * controller.config.vertical_optical_error_max_far_norm
    )


def test_vertical_error_is_pitch_attitude_compensated():
    # F50 (flight 20260729T222920Z-visual-course-3a8ed087): the vertical
    # servo read image-y with NO attitude compensation, so the F49 nose-up
    # brake (0.15 rad up from spawn) tilted the camera up and the world
    # read ~0.24 norm LOW — the servo "centered" a gate that was really
    # ~1.5-2 m below and held ceiling height into a truss.  (F32/F34/F36
    # saw the same contamination with the opposite sign: nose-down dives
    # read gates HIGH.)  Compensation is zero at the spawn attitude.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    assert controller._compensated_ey(0.0, SPAWN_PITCH) == pytest.approx(0.0)
    # Nose-up 0.15 rad REDUCES the effective ey by 1.6 * 0.15 = 0.24.
    assert controller._compensated_ey(
        0.0, SPAWN_PITCH - 0.15
    ) == pytest.approx(-0.24)
    # Nose-down 0.15 rad raises it (the F32/F34/F36 "gate HIGH" sign).
    assert controller._compensated_ey(
        0.0, SPAWN_PITCH + 0.15
    ) == pytest.approx(0.24)
    # End to end, raw y=+0.24 at the nose-up brake attitude compensates to
    # a level gate and therefore requests tilt-compensated support.
    braked = _tracked_controller(_track("A", 0.0, 0.24))
    braked._alt_est_m = 2.0  # keep the near-ground sink taper out of this check
    out = _command(braked, 100.10, pitch=SPAWN_PITCH - 0.15)
    braked_path = braked._vertical_path_observation(
        braked.current,
        now_s=100.10,
        pitch_rad=SPAWN_PITCH - 0.15,
    )
    assert braked_path.outer_error == pytest.approx(0.0, abs=1e-12)
    assert braked._last_vertical_position_delta == pytest.approx(
        0.0, abs=1e-12
    )
    assert braked._last_vertical_motion_delta == pytest.approx(
        0.0, abs=1e-12
    )
    assert braked._last_vertical_visual_delta == pytest.approx(
        braked._last_vertical_position_delta
        + braked._last_vertical_motion_delta
        + braked._last_vertical_direction_delta,
        abs=1e-12,
    )
    assert braked._last_vertical_collective_target == pytest.approx(
        braked._last_vertical_support
        + braked._last_vertical_visual_delta
        + braked._last_vertical_imu_delta,
        abs=1e-12,
    )
    assert out.thrust == pytest.approx(
        braked._last_vertical_collective_target, abs=1e-9
    )
    assert out.thrust == pytest.approx(
        SPAWN_SUPPORT / math.cos(0.15), abs=1e-9
    )
    # ...while the same reading at the spawn attitude really is a low-gate
    # optical miss.
    level = _tracked_controller(_track("A", 0.0, 0.24))
    level._alt_est_m = 2.0
    out = _command(level, 100.10, pitch=SPAWN_PITCH)
    motion = level._last_vertical_motion
    path = level._vertical_path_observation(
        level.current, now_s=100.10, pitch_rad=SPAWN_PITCH
    )
    expected_position, expected_rate, expected_projection = (
        _expected_vertical_trajectory_terms(level, path, motion)
    )
    assert level._last_vertical_position_delta == pytest.approx(
        expected_position, abs=1e-12
    )
    assert level._last_vertical_motion_delta == pytest.approx(
        expected_rate + expected_projection, abs=1e-12
    )
    assert level._last_vertical_direction_delta == pytest.approx(
        0.0, abs=1e-12
    )
    assert level._last_vertical_visual_delta == pytest.approx(
        expected_position + expected_rate + expected_projection,
        abs=1e-12,
    )
    assert level._last_vertical_collective_target == pytest.approx(
        level._last_vertical_support
        + level._last_vertical_visual_delta
        + level._last_vertical_imu_delta,
        abs=1e-12,
    )
    assert out.thrust == pytest.approx(
        level._last_vertical_collective_target, abs=1e-9
    )
    assert out.thrust < SPAWN_SUPPORT


def test_gate0_takeoff_feedforward_has_noncancellable_minimum_then_transfers():
    # The configured 0.30 value is the bounded minimum launch-energy profile.
    # Near-zero image position or a contradictory first-frame sign may shape
    # feedback above that floor, but cannot retire motor spin-up.  The one
    # collective owner then transfers smoothly to the outer-y trajectory.
    config = CleanCourseConfig(gate0_climb_vertical_offset_norm=0.0)
    high = _tracked_controller(
        _track("A", 0.0, -0.20, scale=0.05), config=config
    )
    boosted = _command(high, 100.10, pitch=SPAWN_PITCH)
    assert 0.29 < boosted.thrust <= config.launch_boost_thrust
    assert high._last_launch_collective_delta > 0.0

    low = _tracked_controller(
        _track("A", 0.0, 0.20, scale=0.05), config=config
    )
    low_output = _command(low, 100.10, pitch=SPAWN_PITCH)
    assert low._last_launch_collective_delta > 0.0
    assert low_output.thrust >= low._last_vertical_support - 0.0005
    assert low_output.thrust == pytest.approx(boosted.thrust, abs=1e-12)
    assert low_output.thrust == pytest.approx(
        config.launch_boost_thrust, abs=1e-12
    )

    first_after = _command(
        high, 100.0 + config.launch_boost_duration_s + 0.05,
        pitch=SPAWN_PITCH,
    )
    # The one collective owner carries the launch request through the seam
    # and then converges to the closed-loop target without a thrust step.
    assert SPAWN_SUPPORT < first_after.thrust < boosted.thrust
    thrusts = [first_after.thrust]
    now = 100.0 + config.launch_boost_duration_s + 0.05
    for _ in range(30):
        now += 0.033
        thrusts.append(_command(high, now, pitch=SPAWN_PITCH).thrust)
    assert all(b <= a + 1e-12 for a, b in zip(thrusts, thrusts[1:]))
    assert thrusts[-1] == pytest.approx(SPAWN_SUPPORT, abs=0.003)


def test_launch_boost_constants_are_the_cut_values():
    # Flight 20260729T094736Z-visual-course-9d430a40: the 0.32 x 0.75 s boost
    # alone built vz ~= +2.3 m/s by t=0.75 (~70% of peak climb velocity) and
    # the trajectory overshot the ~1.8-2 m required climb into the top bar.
    # Cut to 0.30 x 0.40 s; 0.30 stays inside the historically validated
    # 0.30..0.32 launch-thrust range.
    assert LAUNCH_BOOST_THRUST == 0.30
    assert LAUNCH_BOOST_DURATION_S == 0.40
    config = CleanCourseConfig()
    assert config.launch_boost_thrust == 0.30
    assert config.launch_boost_duration_s == 0.40


def test_vertical_loss_decays_toward_support_not_saturation_retention():
    # Start with a saturated sub-support collective (target below center).
    controller = _tracked_controller(_track("A", 0.0, 0.30))
    _settle_commit_passage_covariance(controller.current)
    controller._alt_est_m = 2.0  # honest altitude: full descent authority (F103)
    saturated = _command(controller, 100.10, pitch=SPAWN_PITCH).thrust
    assert saturated < SUPPORT - 0.01
    # The vertical axis becomes unobservable (top clipping censors y); the
    # track stays visible on x.  The derivative term is discarded and the
    # collective must decay smoothly toward tilt-compensated support.
    now = 100.15
    thrusts = []
    for frame in range(40):
        controller.observe(
            _update([_track("A", 0.0, 0.30, clipping=FrameEdge.TOP)], frame_id=10 + frame),
            now_s=now,
        )
        thrusts.append(_command(controller, now + 0.005, pitch=SPAWN_PITCH).thrust)
        now += 0.033
    assert all(math.isfinite(value) for value in thrusts)
    assert thrusts[-1] > saturated + 0.01  # not retained at saturation
    assert thrusts[-1] == pytest.approx(SPAWN_SUPPORT, abs=0.01)  # decayed to support
    assert all(0.21 <= value <= 0.34 for value in thrusts)


# ---------------------------------------------------------------------------
# Directional law
# ---------------------------------------------------------------------------


def test_verified_yaw_and_roll_directions():
    # 2026-07-29 crossing-geometry analysis (Q5): with the target right of
    # center the retired controller pinned yaw at -0.150 and x diverged in
    # 75% of pairs; recentering x>0 requires POSITIVE yaw, with a
    # coordinated positive bank toward the target.
    right = _tracked_controller(_track("A", 0.30, 0.0))
    output = _command(right, 100.10)
    assert output.yaw_rate_rad_s > 0.0
    assert output.target_roll_rad > 0.0

    left = _tracked_controller(_track("A", -0.30, 0.0))
    output = _command(left, 100.10)
    assert output.yaw_rate_rad_s < 0.0
    assert output.target_roll_rad < 0.0


def test_centered_bearing_does_not_unwind_optical_intercept_bank():
    # Camera centering is a heading/FOV condition, not proof that the current
    # lateral velocity will cross the aperture.  With x centered but positive
    # de-dilated optical motion, yaw stays centered while roll keeps bending
    # the flight path toward the predicted plane intercept.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    current = controller.current
    _settle_commit_passage_covariance(current)
    current.x_axis.p = 0.0
    current.raw_x = 0.0
    current.x_axis.v = 0.30
    current.scale_axis.v = 0.50
    current.outer_expansion_rate = 0.50
    current.scale_axis.vv = 0.02**2
    current.last_measurement_s = 100.10
    current.last_x_measurement_s = 100.10

    out = _command(controller, 100.10, pitch=SPAWN_PITCH, yaw=0.0)

    assert controller._last_lateral_motion.bearing_error == pytest.approx(
        0.0, abs=1e-12
    )
    assert controller._last_lateral_motion.physical_rate_norm_s > 0.0
    assert controller._last_lateral_motion.intercept_error > 0.0
    assert controller._last_lateral_motion.control_authority > 0.0
    assert out.yaw_rate_rad_s == pytest.approx(0.0, abs=1e-12)
    assert out.target_roll_rad > 0.0


def test_uncertain_lateral_ttc_cannot_veto_coherent_image_motion():
    # Camera bearing and physical interception have separate owners.  Broad
    # TTC covariance removes the uncertain residual magnitude, but fresh image
    # motion still supplies the short-horizon path: yaw keeps the left-side
    # target in view while roll bends right toward its coherent crossing miss.
    controller = _tracked_controller(_track("A", -0.25, 0.0))
    current = controller.current
    _settle_commit_passage_covariance(current)
    current.x_axis.p = -0.25
    current.raw_x = -0.25
    current.x_axis.v = 1.0
    current.scale_axis.v = 0.50
    current.outer_expansion_rate = 0.50
    current.scale_axis.vv = 10.0**2
    current.last_measurement_s = 100.10
    current.last_x_measurement_s = 100.10

    out = _command(controller, 100.10, pitch=SPAWN_PITCH, yaw=0.0)
    motion = controller._last_lateral_motion

    assert motion.intercept_error > 0.0
    assert motion.control_authority == pytest.approx(0.0, abs=1e-12)
    assert motion.fallback_intercept_error > 0.0
    assert motion.optical_intercept_error > 0.0
    assert controller._last_lateral_baseline_reference_x < 0.0
    assert controller._last_lateral_direction_override_x > 0.0
    assert controller._lateral_intercept_reference_x > 0.0
    assert out.yaw_rate_rad_s < 0.0
    assert out.target_roll_rad > 0.0


def test_gate0_climb_vertical_offset_is_bounded_feedforward():
    # 2026-07-29 analysis (Q1/Q4): cross gate 0 higher so gate 1 is first
    # seen with doubled top-edge margin.  A gate-0 target still ABOVE center
    # (ey < 0, image-down) yields collective above support; the bias stays
    # inside the thrust envelope and retires with the gate-0 phase.
    config = _config(gate0_climb_vertical_offset_norm=0.25)
    controller = _tracked_controller(_track("A", 0.0, -0.10), config=config)
    _settle_commit_passage_covariance(controller.current)
    output = _command(controller, 100.10, pitch=SPAWN_PITCH)
    # Outer y=-0.10 owns the proportional baseline.  The bounded Gate-0
    # offset moves only the passage projection to -0.35; it is feedforward,
    # not a replacement position servo.
    motion = controller._last_vertical_motion
    path = controller._vertical_path_observation(
        controller.current, now_s=100.10, pitch_rad=SPAWN_PITCH
    )
    assert path.outer_error == pytest.approx(-0.10, abs=1e-12)
    assert motion.bearing_error == pytest.approx(-0.35, abs=1e-12)
    expected_position, expected_rate, expected_projection = (
        _expected_vertical_trajectory_terms(controller, path, motion)
    )
    assert controller._last_vertical_position_delta == pytest.approx(
        expected_position, abs=1e-12
    )
    assert controller._last_vertical_motion_delta == pytest.approx(
        expected_rate + expected_projection, abs=1e-12
    )
    assert controller._last_vertical_direction_delta == pytest.approx(
        0.0, abs=1e-12
    )
    assert controller._last_vertical_visual_delta == pytest.approx(
        expected_position + expected_rate + expected_projection,
        abs=1e-12,
    )
    assert controller._last_vertical_collective_target == pytest.approx(
        controller._last_vertical_support
        + controller._last_vertical_visual_delta
        + controller._last_vertical_imu_delta,
        abs=1e-12,
    )
    assert output.thrust == pytest.approx(
        controller._last_vertical_collective_target, abs=1e-9
    )
    assert output.thrust > SPAWN_SUPPORT
    assert 0.21 <= output.thrust <= 0.34

    high = _tracked_controller(_track("A", 0.0, -0.60), config=config)
    assert _command(high, 100.10, pitch=SPAWN_PITCH).thrust <= 0.34

    controller.observe(
        _update(
            [_track("A", 0.0, 0.0), _track("B", 0.30, 0.0, scale=0.05)],
            frame_id=4,
        ),
        now_s=100.12,
    )
    # F42: promotion requires a persistent successor; seed the age.
    controller._track_first_seen_s["B"] = 100.12 - 1.0
    prepromotion_thrust = output.thrust
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=1250, now_s=100.14
    )
    assert promoted
    assert controller.state is CleanCourseState.TRACK
    controller._alt_est_m = 2.0  # honest altitude (floor quiet)
    output = _command(controller, 100.16, pitch=SPAWN_PITCH)
    assert output.gate_index == 1
    # Promotion removes the gate-0 offset from the target, but does not reset
    # collective.  The carried request approaches support monotonically.
    assert SPAWN_SUPPORT < output.thrust < prepromotion_thrust
    previous = output.thrust
    now = 100.16
    for _ in range(30):
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        controller.current.last_y_measurement_s = now
        output = _command(controller, now, pitch=SPAWN_PITCH)
        assert output.thrust <= previous + 1e-12
        previous = output.thrust
    assert output.thrust == pytest.approx(SPAWN_SUPPORT, abs=0.003)


def test_gate0_climb_offset_scales_with_closure():
    # Flight 20260729T085719Z-visual-course-4455fd61: the fixed 0.25 climb
    # offset built a ~2.5-3 m climb into gate 0's top bar.  The bias is now
    # closure-scaled: full at the spawn detection scale (log -1.79), ramping
    # linearly to zero at the crossing-arm scale (log -0.80).  Tracks sit
    # above center (ey < 0) so the never-above-center clamp does not engage.
    config = _config(gate0_climb_vertical_offset_norm=0.25)
    far = _tracked_controller(
        _track("A", 0.0, -0.10, scale=0.1667), config=config
    )
    _settle_commit_passage_covariance(far.current)
    far_thrust = _command(far, 100.10, pitch=SPAWN_PITCH).thrust

    mid = _tracked_controller(
        _track("A", 0.0, -0.10, scale=math.exp(-1.295)), config=config
    )
    _settle_commit_passage_covariance(mid.current)
    mid_thrust = _command(mid, 100.10, pitch=SPAWN_PITCH).thrust

    crossing = _tracked_controller(
        _track("A", 0.0, 0.0, scale=math.exp(-0.80)), config=config
    )
    _settle_commit_passage_covariance(crossing.current)
    crossing_thrust = _command(crossing, 100.10, pitch=SPAWN_PITCH).thrust
    assert far_thrust > mid_thrust > crossing_thrust
    assert crossing_thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)


def test_gate0_climb_offset_never_lifts_aim_above_center():
    # Flight 20260729T094736Z-visual-course-9d430a40: the ramped climb
    # setpoint stayed positive (~+0.1..0.2) past center through t=1.5,
    # holding collective >= support until the climb could not be erased.
    # With the gate at/below image center (ey >= 0) the effective offset is
    # clamped to <= 0 regardless of the closure ramp: the bias may push the
    # aim point UP toward center, never above it.
    config = _config(gate0_climb_vertical_offset_norm=0.25)
    centered = _tracked_controller(
        _track("A", 0.0, 0.0, scale=0.1667), config=config  # spawn scale
    )
    centered_output = _command(centered, 100.10, pitch=SPAWN_PITCH)
    assert centered._last_vertical_position_delta == pytest.approx(
        0.0, abs=1e-12
    )
    assert centered._last_vertical_motion_delta == pytest.approx(
        0.0, abs=1e-12
    )
    assert centered._last_vertical_collective_target == pytest.approx(
        centered._last_vertical_support
        + centered._last_vertical_visual_delta
        + centered._last_vertical_imu_delta,
        abs=1e-12,
    )
    assert centered_output.thrust == pytest.approx(
        SPAWN_SUPPORT, abs=1e-9
    )

    below = _tracked_controller(
        _track("A", 0.0, 0.10, scale=0.1667), config=config
    )
    _settle_commit_passage_covariance(below.current)
    output = _command(below, 100.10, pitch=SPAWN_PITCH)
    # Offset contributes nothing: both the continuous outer baseline and the
    # passage projection remain +0.10.
    motion = below._last_vertical_motion
    path = below._vertical_path_observation(
        below.current, now_s=100.10, pitch_rad=SPAWN_PITCH
    )
    assert path.outer_error == pytest.approx(0.10, abs=1e-12)
    assert motion.bearing_error == pytest.approx(path.error, abs=1e-12)
    expected_position, expected_rate, expected_projection = (
        _expected_vertical_trajectory_terms(below, path, motion)
    )
    assert below._last_vertical_position_delta == pytest.approx(
        expected_position, abs=1e-12
    )
    assert below._last_vertical_motion_delta == pytest.approx(
        expected_rate + expected_projection, abs=1e-12
    )
    assert below._last_vertical_visual_delta == pytest.approx(
        expected_position
        + expected_rate
        + expected_projection
        + below._last_vertical_direction_delta,
        abs=1e-12,
    )
    assert below._last_vertical_collective_target == pytest.approx(
        below._last_vertical_support
        + below._last_vertical_visual_delta
        + below._last_vertical_imu_delta,
        abs=1e-12,
    )
    assert output.thrust == pytest.approx(
        below._last_vertical_collective_target, abs=1e-9
    )
    assert output.thrust < SPAWN_SUPPORT


def test_gate0_closure_dilation_is_not_a_vertical_term():
    # Normalized image rate is not physical m/s.  For a fixed world offset,
    # image rate == log-scale expansion * bearing is pure perspective
    # dilation.  The optical passage model removes it before projecting the
    # miss to the gate plane.
    config = _config()
    controller = _tracked_controller(_track("A", 0.0, -0.10), config=config)
    controller.current.scale_axis.v = 0.20
    controller.current.outer_expansion_rate = 0.20
    controller.current.last_measurement_s = 100.10
    controller.current.last_y_measurement_s = 100.10
    controller.current.y_axis.v = -0.02  # expansion * y: dilation only
    _settle_commit_passage_covariance(controller.current)
    output = _command(controller, 100.10, pitch=SPAWN_PITCH)
    motion = controller._last_vertical_motion

    assert output.vertical_qualified
    assert motion.closure_rate_s == pytest.approx(0.20, abs=1e-9)
    assert motion.ttc_s == pytest.approx(
        controller.config.passage_ttc_max_s, abs=1e-9
    )
    assert motion.physical_rate_norm_s == pytest.approx(0.0, abs=1e-9)
    assert motion.fallback_intercept_error == pytest.approx(-0.11, abs=1e-9)
    assert motion.optical_intercept_error == pytest.approx(-0.10, abs=1e-9)
    assert motion.intercept_error == pytest.approx(
        motion.fallback_intercept_error
        + motion.projection_authority
        * (
            motion.optical_intercept_error
            - motion.fallback_intercept_error
        ),
        abs=1e-12,
    )
    assert motion.intercept_error < 0.0
    path = controller._vertical_path_observation(
        controller.current, now_s=100.10, pitch_rad=SPAWN_PITCH
    )
    expected_position, expected_rate, expected_projection = (
        _expected_vertical_trajectory_terms(controller, path, motion)
    )
    assert expected_rate > 0.0
    assert expected_projection < 0.0
    assert abs(expected_projection) < abs(expected_rate)
    assert controller._last_vertical_position_delta == pytest.approx(
        expected_position, abs=1e-12
    )
    assert controller._last_vertical_motion_delta == pytest.approx(
        expected_rate + expected_projection, abs=1e-12
    )
    assert controller._last_vertical_visual_delta == pytest.approx(
        expected_position
        + expected_rate
        + expected_projection
        + controller._last_vertical_direction_delta,
        abs=1e-12,
    )
    assert controller._last_vertical_collective_target == pytest.approx(
        controller._last_vertical_support
        + controller._last_vertical_visual_delta
        + controller._last_vertical_imu_delta,
        abs=1e-12,
    )
    assert output.thrust == pytest.approx(
        controller._last_vertical_collective_target, abs=1e-9
    )


def test_passage_vertical_projection_keeps_only_physical_image_motion():
    controller = _tracked_controller(_track("A", 0.0, -0.20))
    current = controller.current
    current.scale_axis.v = 0.50
    current.outer_expansion_rate = 0.50
    current.last_measurement_s = 100.10

    # y_dot = expansion*y is a fixed offset growing only through closure.
    current.y_axis.v = -0.10
    dilation_only = controller._passage_motion(
        current,
        current.y_axis,
        -0.20,
        now_s=100.10,
        measurement_age_s=0.0,
    )
    assert dilation_only.ttc_s == pytest.approx(2.0, abs=1e-9)
    assert dilation_only.physical_rate_norm_s == pytest.approx(0.0, abs=1e-9)
    assert dilation_only.fallback_intercept_error == pytest.approx(
        -0.25, abs=1e-9
    )
    assert dilation_only.optical_intercept_error == pytest.approx(
        -0.20, abs=1e-9
    )
    assert dilation_only.intercept_error == pytest.approx(
        dilation_only.fallback_intercept_error
        + dilation_only.projection_authority
        * (
            dilation_only.optical_intercept_error
            - dilation_only.fallback_intercept_error
        ),
        abs=1e-12,
    )
    assert dilation_only.intercept_error < 0.0

    # Additional image-down motion is de-dilated optical motion.  It is
    # projected over an uncertainty-weighted horizon toward TTC, not treated
    # as a metric vertical-velocity request.
    current.y_axis.v = 0.20
    slow_closure = controller._passage_motion(
        current,
        current.y_axis,
        -0.20,
        now_s=100.10,
        measurement_age_s=0.0,
    )
    assert slow_closure.physical_rate_norm_s == pytest.approx(0.30, abs=1e-9)
    assert slow_closure.fallback_intercept_error == pytest.approx(
        -0.10, abs=1e-9
    )
    assert slow_closure.optical_intercept_error == pytest.approx(
        0.40, abs=1e-9
    )
    assert slow_closure.intercept_error == pytest.approx(
        slow_closure.fallback_intercept_error
        + slow_closure.projection_authority
        * (
            slow_closure.optical_intercept_error
            - slow_closure.fallback_intercept_error
        ),
        abs=1e-12,
    )

    # The same de-dilated motion has less time to accumulate at twice the
    # closure rate, demonstrating that expansion/TTC owns the projection.
    current.scale_axis.v = 1.0
    current.outer_expansion_rate = 1.0
    current.y_axis.v = 0.10  # lambda*y + the same +0.30 physical rate
    fast_closure = controller._passage_motion(
        current,
        current.y_axis,
        -0.20,
        now_s=100.10,
        measurement_age_s=0.0,
    )
    assert fast_closure.ttc_s == pytest.approx(1.0, abs=1e-9)
    assert fast_closure.physical_rate_norm_s == pytest.approx(0.30, abs=1e-9)
    assert fast_closure.intercept_error < slow_closure.intercept_error


def test_f161_gate1_weak_closure_blend_preserves_fallback_axis_signs():
    # Recorded F161 Gate-1 adoption: the newborn raw expansion spike strongly
    # contradicted the filtered closure estimate.  F166's robust outer-only
    # control estimate prevents that one raw spike from entering de-dilation
    # at full magnitude, while the trustworthy short image-motion projection
    # remains above and left.
    controller = _tracked_controller(
        _track("A", -0.499448, -0.095734)
    )
    current = controller.current
    _settle_commit_passage_covariance(current)
    current.x_axis.p = -0.499448
    current.x_axis.v = 0.000182
    current.y_axis.p = -0.095734
    current.y_axis.v = -0.010870
    current.scale_axis.v = 0.223664
    current.outer_expansion_rate = 7.25808
    current.scale_axis.vv = 0.02**2
    current.last_measurement_s = 100.10
    current.last_x_measurement_s = 100.10
    current.last_y_measurement_s = 100.10

    x_motion = controller._passage_motion(
        current,
        current.x_axis,
        current.x,
        now_s=100.10,
        measurement_age_s=0.0,
    )
    y_motion = controller._passage_motion(
        current,
        current.y_axis,
        current.y,
        now_s=100.10,
        measurement_age_s=0.0,
    )

    for motion in (x_motion, y_motion):
        assert 0.223664 < motion.closure_rate_s < 0.50
        assert motion.projection_authority < 0.04
        assert motion.fallback_intercept_error < 0.0
        assert motion.intercept_error == pytest.approx(
            motion.fallback_intercept_error
            + motion.projection_authority
            * (
                motion.optical_intercept_error
                - motion.fallback_intercept_error
            ),
            abs=1e-12,
        )
        assert motion.intercept_error < 0.0

    # Even the horizontal complete model only barely crosses zero; the
    # vertical complete model now keeps the correct sign outright.
    assert x_motion.optical_intercept_error > 0.0
    assert y_motion.optical_intercept_error < 0.0

    assert y_motion.fallback_intercept_error == pytest.approx(
        -0.095734 + -0.010870 * controller.config.commit_blackout_s,
        abs=1e-12,
    )
    assert x_motion.fallback_intercept_error == pytest.approx(
        -0.499448 + 0.000182 * controller.config.commit_blackout_s,
        abs=1e-12,
    )


def test_full_credible_closure_reaches_complete_optical_intercept():
    controller = _tracked_controller(_track("A", -0.20, 0.0))
    current = controller.current
    _settle_commit_passage_covariance(current)
    current.x_axis.p = -0.20
    current.x_axis.v = 0.20
    current.x_axis.pp = 0.0
    current.scale_axis.v = 0.50
    current.outer_expansion_rate = 0.50
    current.scale_axis.vv = 0.02**2
    current.last_measurement_s = 100.10
    current.last_x_measurement_s = 100.10

    motion = controller._passage_motion(
        current,
        current.x_axis,
        current.x,
        now_s=100.10,
        measurement_age_s=0.0,
    )

    assert motion.projection_authority == pytest.approx(1.0, abs=1e-12)
    assert motion.fallback_intercept_error < 0.0
    assert motion.optical_intercept_error > 0.0
    assert motion.intercept_error == pytest.approx(
        motion.optical_intercept_error, abs=1e-12
    )


_F162_DIRECTION_ROWS = (
    # t, frame, pitch, filtered y, raw y, ydot, y std, inner log scale,
    # outer log scale, filtered/raw closure, and supporting IMU vz.
    (
        5.359,
        1907094,
        -0.4544197644591279,
        0.15168623663814268,
        0.1611111111111112,
        0.23906230902156228,
        0.05370577868817966,
        -0.8848248999833153,
        -0.9422344336661613,
        1.5921387255277188,
        0.7895452201852513,
        -0.12293242797737247,
    ),
    (
        5.406,
        1907095,
        -0.4554753548939212,
        0.17297428277105434,
        0.1777777777777778,
        0.26082292155121695,
        0.051635861457027844,
        -0.8549817344580977,
        -0.9102777707943349,
        1.4832478163116634,
        0.76868707250974,
        -0.13073258407356997,
    ),
    (
        5.453,
        1907096,
        -0.4562610277027766,
        0.19839781893336988,
        0.2055555555555555,
        0.2918651291337376,
        0.0510593362988328,
        -0.8223013511285067,
        -0.8691093511500126,
        1.3936901087533204,
        0.7890924459173493,
        -0.1412417877699674,
    ),
    (
        5.500,
        1907097,
        -0.45595882332436766,
        0.23519250112015266,
        0.25,
        0.3555700225302015,
        0.051428164410942966,
        -0.7765388577751083,
        -0.8016036813417322,
        1.3459058655489267,
        0.9122435586759983,
        -0.15274400480188632,
    ),
    (
        5.593,
        1907099,
        -0.44425707350146254,
        0.30233038428415315,
        0.31666666666666665,
        0.4516339891684342,
        0.05375702369579412,
        -0.6807685040370061,
        -0.6828375369585774,
        1.281392419069951,
        1.0583266549091752,
        -0.17631775894079366,
    ),
)


def _f162_gate1_controller():
    """Start at the authoritative Gate-1 boundary, without synthetic credit."""

    controller = CleanCourseController(_config())
    controller.initialize(
        _update([_track("A", 0.0, 0.0)], frame_id=1907093),
        gate_index=1,
        fallback_center_norm=(0.0, 0.0),
        fallback_apparent_scale=0.10,
        now_s=105.30,
    )
    controller._alt_est_m = 2.0
    return controller


def _apply_f162_direction_row(controller, row):
    (
        elapsed,
        frame_id,
        pitch,
        filtered_y,
        raw_y,
        image_rate_y,
        y_std,
        log_scale,
        outer_log_scale,
        filtered_closure,
        raw_closure,
        vz_est,
    ) = row
    now = 100.0 + elapsed
    # Consume a distinct recorded camera observation through observe(); the
    # state values below are the controller-boundary values recorded after
    # that observation, not invented controller flags or passage state.
    controller.observe(
        _update(
            [
                _track(
                    "A",
                    -0.10,
                    raw_y,
                    scale=math.exp(outer_log_scale),
                    confidence=0.95,
                )
            ],
            frame_id=frame_id,
        ),
        now_s=now,
    )
    current = controller.current
    current.y_axis.p = filtered_y
    current.raw_y = raw_y
    current.y_axis.v = image_rate_y
    current.y_axis.pp = y_std**2
    current.y_axis.pv = 0.0
    current.y_axis.vv = 0.02**2
    current.scale_axis.p = log_scale
    current.scale_axis.v = filtered_closure
    current.scale_axis.vv = 0.02**2
    current.outer_log_scale = outer_log_scale
    current.outer_log_scale_s = now
    current.outer_expansion_rate = raw_closure
    current.aperture_half_x = None
    current.aperture_half_y = None
    controller._vz_est_m_s = vz_est
    return _command(controller, now, pitch=pitch)


def test_f162_motion_direction_reverses_before_static_bearing_and_imu():
    # F162's first three coherent near-plane samples all project the aperture
    # downward even though static pitch compensation still says it is above
    # center.  Three distinct frames own descent at t=5.453; uncertainty may
    # reduce magnitude, while IMU damping remains only a bounded innovation
    # and cannot veto the visually owned direction.
    controller = _f162_gate1_controller()
    controller._collective = 0.2762406
    outputs = []
    motions = []
    for row in _F162_DIRECTION_ROWS:
        # At the reversal boundary, preserve the actual F162 carried
        # collective from the preceding recorded tick.  The counterfactual
        # therefore has to erase the real climb reserve, not a cold start.
        if row[0] == 5.453:
            controller._collective = 0.27866674133727154
        outputs.append(_apply_f162_direction_row(controller, row))
        motions.append(controller._last_vertical_motion)
        assert controller.state is CleanCourseState.TRACK
        assert controller.gate_index == 1
        assert controller._commit_entry_s is None

    reversal = outputs[2]
    reversal_motion = motions[2]
    assert controller._vertical_direction_sign == 1
    assert controller._vertical_direction_streak >= 3
    assert controller._vertical_direction_source == "coherent_motion"
    assert abs(controller._last_vertical_imu_delta) <= (
        controller.config.vertical_imu_max_opposition_fraction
        * max(
            abs(controller._last_vertical_visual_delta),
            abs(controller._last_vertical_motion_delta),
        )
        + 1e-12
    )
    assert (
        controller._last_vertical_visual_delta
        + controller._last_vertical_imu_delta
        < 0.0
    )
    assert controller._last_vertical_collective_target < (
        controller._last_vertical_support
    )
    # The recorded compensated bearing was still negative at the direction
    # transition; it cannot own or delay this near-plane sign.
    assert reversal_motion.bearing_error < 0.0
    assert reversal_motion.fallback_intercept_error > 0.0
    assert reversal_motion.optical_intercept_error > 0.0
    assert reversal.thrust < 0.27866674133727154
    # By the final exact frame (t=5.593), before first BOTTOM at 5.640, the
    # bounded 0.12/s fast path has driven filtered collective below support.
    assert outputs[-1].thrust < controller._last_vertical_support


def test_republished_frame_cannot_satisfy_direction_streak():
    controller = _f162_gate1_controller()
    row = _F162_DIRECTION_ROWS[0]
    _apply_f162_direction_row(controller, row)
    assert controller._vertical_direction_streak == 1

    frozen_update = _update(
        [
            _track(
                "A",
                -0.10,
                row[4],
                scale=math.exp(row[8]),
                confidence=0.95,
            )
        ],
        frame_id=row[1],
    )
    for offset in (0.020, 0.040, 0.060):
        now = 100.0 + row[0] + offset
        controller.observe(frozen_update, now_s=now)
        # Preserve the same recorded coherent state: only its republication
        # time changed, which is not another camera observation.
        current = controller.current
        current.y_axis.p = row[3]
        current.raw_y = row[4]
        current.y_axis.v = row[5]
        current.y_axis.pp = row[6] ** 2
        current.scale_axis.p = row[7]
        current.scale_axis.v = row[9]
        current.outer_log_scale = row[8]
        current.outer_expansion_rate = row[10]
        _command(controller, now, pitch=row[2])

    assert controller._vertical_direction_streak == 1
    assert controller._vertical_direction_sign == 0


def test_exact_axis_optical_authority_fades_with_intercept_uncertainty():
    controller = _tracked_controller(_track("A", 0.0, 0.20))
    current = controller.current
    current.last_measurement_s = 100.10
    current.last_y_measurement_s = 100.10

    broad = controller._passage_motion(
        current,
        current.y_axis,
        0.20,
        now_s=100.10,
        measurement_age_s=0.0,
    )
    _settle_commit_passage_covariance(current)
    settled = controller._passage_motion(
        current,
        current.y_axis,
        0.20,
        now_s=100.10,
        measurement_age_s=0.0,
    )

    assert settled.intercept_std < broad.intercept_std
    assert 0.0 < broad.control_authority < settled.control_authority < 1.0
    assert settled.control_authority == pytest.approx(
        1.0
        - settled.intercept_std
        / controller.config.passage_motion_full_std_norm,
        abs=1e-12,
    )


def test_expansion_uncertainty_widens_passage_interval_and_blocks_commit():
    # Hold bearing, image motion, and the 0.30/s closure estimate fixed.  Only
    # uncertainty in that expansion rate changes, so TTC and the physical
    # motion sign stay identical while the projected plane interval widens.
    def _approach(scale_rate_variance):
        controller = _commit_controller()
        current = controller.current
        current.scale_axis.v = 0.30
        current.outer_expansion_rate = 0.30
        current.scale_axis.vv = scale_rate_variance
        # Same small positive de-dilated motion on both passage axes.
        current.x_axis.v = 0.30 * current.x + 0.005
        current.y_axis.v = 0.30 * current.y + 0.005
        current.last_measurement_s = 100.10
        current.last_x_measurement_s = 100.10
        current.last_y_measurement_s = 100.10
        motion = controller._passage_motion(
            current,
            current.x_axis,
            current.x,
            now_s=100.10,
            measurement_age_s=0.0,
        )
        return controller, motion

    certain, certain_motion = _approach(0.02**2)
    uncertain, uncertain_motion = _approach(0.50**2)

    assert uncertain_motion.closure_rate_s == pytest.approx(
        certain_motion.closure_rate_s, abs=1e-12
    )
    assert uncertain_motion.ttc_s == pytest.approx(
        certain_motion.ttc_s, abs=1e-12
    )
    assert uncertain_motion.ttc_s == pytest.approx(
        certain.config.passage_ttc_max_s, abs=1e-12
    )
    assert uncertain_motion.physical_rate_norm_s == pytest.approx(
        certain_motion.physical_rate_norm_s, abs=1e-12
    )
    assert certain_motion.physical_rate_norm_s > 0.0
    assert uncertain_motion.intercept_error == pytest.approx(
        certain_motion.intercept_error, abs=1e-12
    )
    assert uncertain_motion.closure_std_s > certain_motion.closure_std_s
    assert uncertain_motion.ttc_std_s > certain_motion.ttc_std_s
    assert uncertain_motion.intercept_std > certain_motion.intercept_std
    assert uncertain_motion.control_authority < certain_motion.control_authority

    # The standard 0.25 half-aperture gives a 0.15 admission margin: the
    # low-uncertainty interval fits, while identical nominal motion with the
    # wider expansion/TTC interval is refused and remains safely in TRACK.
    assert certain._commit_entry_budget_ok(
        100.10, SPAWN_PITCH, certain.config
    )
    assert not uncertain._commit_entry_budget_ok(
        100.10, SPAWN_PITCH, uncertain.config
    )
    _drive_commit_window(certain, 100.10)
    _drive_commit_window(uncertain, 100.10)
    assert certain.state is CleanCourseState.COMMIT
    assert uncertain.state is CleanCourseState.TRACK
    assert uncertain._pre_cross_brake_active


def test_imu_damping_applies_in_predict_and_search_without_optical_motion():
    # Vision loss removes the fresh outer-y owner; it does not switch to a
    # governor, floor, margin, or remembered visual collective.  Supporting
    # IMU damping remains bounded in both blind states.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.observe(_update([], frame_id=3), now_s=100.12)
    controller.observe(_update([], frame_id=4), now_s=100.30)
    assert controller.state is CleanCourseState.PREDICT
    controller._vz_est_m_s = 2.0
    predict = _command(controller, 100.31, pitch=SPAWN_PITCH)
    assert predict.thrust == pytest.approx(controller.config.min_thrust, abs=1e-9)
    assert controller._last_vertical_visual_delta == pytest.approx(0.0, abs=1e-12)
    controller._enter_search(100.32)
    search = _command(controller, 100.34, pitch=SPAWN_PITCH)
    assert search.thrust == pytest.approx(controller.config.min_thrust, abs=1e-9)
    assert controller._last_vertical_visual_delta == pytest.approx(0.0, abs=1e-12)


def test_near_plane_compensated_gate_y_keeps_physical_sink_damping():
    # Compensated geometry still sets the optical correction at the crossing,
    # but it must not erase measured vertical energy.  F146 reached Gate-0
    # credit sinking -0.55 m/s after the range fade removed this damping.
    brake_pitch = -0.46
    controller = _tracked_controller(_track("A", 0.0, 0.10))
    controller.current.outer_log_scale = controller.config.commit_min_log_scale
    controller._vz_est_m_s = -0.35
    out = _command(controller, 100.10, pitch=brake_pitch)
    support = SPAWN_SUPPORT / math.cos(brake_pitch - SPAWN_PITCH)
    compensated = controller._compensated_ey(0.10, brake_pitch)
    assert compensated < 0.0
    # The sink estimate reinforces the clear climb correction, but remains a
    # supporting magnitude rather than becoming a second primary trajectory.
    assert controller._last_vertical_visual_delta > 0.0
    assert 0.0 < controller._last_vertical_imu_delta <= (
        controller._last_vertical_visual_delta
    )
    assert out.thrust > support
    assert out.thrust <= controller.config.max_thrust


def test_qualified_imu_rate_authority_does_not_fade_with_range():
    # The physical-rate estimate is supporting damping, not the visual owner.
    # A fresh centered outer replay leaves its bounded sink response continuous
    # through the range schedule.  Near-plane command safety may cap the final
    # collective, but must not attenuate the IMU term itself or change its sign.
    thrusts = []
    imu_deltas = []
    for outer_log_scale in (-2.0, -1.8, -1.6, -1.4, -1.2):
        scale = math.exp(outer_log_scale)
        controller = _tracked_controller(_track("A", 0.0, 0.0, scale=scale))
        controller.observe(
            _update([_track("A", 0.0, 0.0, scale=scale)], frame_id=3),
            now_s=100.08,
        )
        controller._vz_est_m_s = -0.40
        thrusts.append(_command(controller, 100.10, pitch=SPAWN_PITCH).thrust)
        assert controller._last_vertical_visual_delta == pytest.approx(
            0.0, abs=1e-12
        )
        imu_deltas.append(controller._last_vertical_imu_delta)

    assert imu_deltas == pytest.approx(
        [0.12 * 0.40] * len(imu_deltas), abs=1e-9
    )
    assert all(thrust > SPAWN_SUPPORT for thrust in thrusts)
    assert thrusts[:-1] == pytest.approx(
        [SPAWN_SUPPORT + 0.12 * 0.40] * (len(thrusts) - 1), abs=1e-9
    )


def test_imu_damping_arrests_sink_in_predict_and_search():
    # With optical authority absent, a deep sink saturates the one bounded
    # supporting IMU-damping request in both blind states.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.observe(_update([], frame_id=2), now_s=100.12)  # superseded
    assert controller.state is CleanCourseState.PREDICT
    controller._vz_est_m_s = -0.7
    assert _command(controller, 100.16, pitch=SPAWN_PITCH).thrust == pytest.approx(
        controller.config.max_thrust, abs=1e-9
    )
    controller._enter_search(100.20)
    assert _command(controller, 100.22, pitch=SPAWN_PITCH).thrust == pytest.approx(
        controller.config.max_thrust, abs=1e-9
    )


def test_vz_phantom_sink_cannot_move_coast_exact_zero():
    # 2026-07-30 contract correction: a credible close loss outputs EXACT
    # wire zero (roll/pitch/yaw rates and thrust) for the bounded credit
    # wait — support-thrust coasting through the attitude PD is out of
    # contract.  A phantom sink must not change that: the coast is not a
    # governed control law at all.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    # F102: the gate-0 hot-coast trigger is deleted — COAST arms only from
    # an armed COMMIT, so route the entry through the COMMIT latch.
    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = 100.12
    controller.observe(_update([], frame_id=3), now_s=100.12)  # fresh close loss
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    controller._vz_est_m_s = -1.0
    out = _command(controller, 100.14)
    assert out.thrust == 0.0
    assert (out.target_roll_rad, out.target_pitch_rad, out.yaw_rate_rad_s) == (
        0.0,
        0.0,
        0.0,
    )


def test_vz_leaky_integrator_integrates_and_decays():
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _command(controller, 100.10, a_up=1.0)  # first call: dt = control period
    first = controller._vz_est_m_s
    assert first == pytest.approx(0.02 * (1.0 - 0.02 / 2.5), abs=1e-9)
    _command(controller, 100.12, a_up=1.0)
    assert controller._vz_est_m_s > first  # keeps integrating up
    peak = controller._vz_est_m_s
    for tick in range(5):  # zero accel: the leak pulls the estimate down
        _command(controller, 100.14 + 0.02 * tick, a_up=0.0)
    assert 0.0 < controller._vz_est_m_s < peak


def test_engulfing_full_frame_bbox_is_not_a_measurement():
    # Flights 95644bf5 / 4dbe4b8c ended with a degenerate near-full-frame
    # bbox (640x360 / 640x347, every edge clipped) accepted as a gate
    # measurement 46-48 ms before impact.  That is the gate engulfing the
    # camera at the plane: treat it as no measurement (PREDICT semantics).
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    x_before = controller.current.x_axis.p
    y_before = controller.current.y_axis.p
    meas_before = controller.current.last_y_measurement_s
    engulfing = _track(
        "A",
        0.30,
        0.30,
        scale=1.0,  # bbox spans the whole frame
        clipping=(
            FrameEdge.LEFT | FrameEdge.RIGHT | FrameEdge.TOP | FrameEdge.BOTTOM
        ),
    )
    controller.observe(_update([engulfing], frame_id=5), now_s=100.05)
    # Far-scale first miss stays TRACK, but no axis consumed the bogus box.
    assert controller.state is CleanCourseState.TRACK
    assert controller.current.x_axis.p == x_before
    assert controller.current.y_axis.p == y_before
    assert controller.current.last_y_measurement_s == meas_before


def test_stale_vertical_motion_has_no_optical_authority_and_holds_support():
    # Flight 20260729T104947Z-visual-course-bc8c6003: a phantom vy (+0.38
    # norm/s, seeded as the gate sank through the frame) random-walked
    # unmeasured for 5.4 s and commanded an unrecoverable descent.  Once the
    # last accepted y measurement ages out, the uncertain filter state may be
    # retained for association but has no optical control authority.  With no
    # IMU vertical motion, collective holds tilt-compensated support.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller.current.y_axis.v = 0.38  # phantom rate from the crossing
    output = _command(controller, 100.55, pitch=SPAWN_PITCH)
    assert not output.vertical_qualified
    assert controller.current.y_axis.v == pytest.approx(0.38, abs=1e-12)
    assert controller._last_vertical_motion is None
    assert output.thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)


def test_qualification_regain_restores_optical_authority_from_measurements():
    # A stale filter prediction does not command the aircraft.  Qualification
    # returns only through real measurements, which update the retained state
    # and restore optical-passage authority.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller.current.y_axis.v = 0.38
    stale = _command(controller, 100.55, pitch=SPAWN_PITCH)
    assert not stale.vertical_qualified
    assert controller._last_vertical_motion is None
    now = 100.46
    for frame, y in enumerate((0.02, 0.04, 0.06, 0.08)):
        controller.observe(
            _update([_track("A", 0.0, y)], frame_id=10 + frame), now_s=now
        )
        now += 0.033
    output = _command(controller, now, pitch=SPAWN_PITCH)
    assert output.vertical_qualified
    assert controller._last_vertical_motion is not None
    assert controller._last_vertical_motion.control_authority > 0.0
    assert controller.current.y_axis.v > 0.05
    assert controller.current.y_axis.v != pytest.approx(0.38, abs=1e-9)


def test_censored_adoption_never_claims_fresh_vertical():
    # A censored-axis detection must not seed rate state or measurement
    # freshness, including on adoption/rebind: a TB-clipped fragment adopted
    # out of SEARCH leaves vertical unqualified (collective holds support)
    # instead of servoing on a phantom.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    now = 100.10
    for _ in range(40):
        now += 0.033
        controller.observe(_update([], frame_id=9), now_s=now)
        if controller.state is CleanCourseState.SEARCH:
            break
    assert controller.state is CleanCourseState.SEARCH
    now += 0.033
    controller.observe(
        _update(
            [
                _track(
                    "G",
                    0.05,
                    0.0,
                    clipping=FrameEdge.TOP | FrameEdge.BOTTOM,
                )
            ],
            frame_id=50,
        ),
        now_s=now,
    )
    assert controller.state is CleanCourseState.TRACK  # re-adopted
    output = _command(controller, now + 0.02, pitch=SPAWN_PITCH)
    # The censored creation box is not a y measurement.
    assert not output.vertical_qualified
    assert output.thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)


def test_engulfing_anchor_blocks_search_and_keeps_vertical_unqualified():
    # Flight bc8c6003: 181 ticks of engulfing boxes cycled TRACK<->SEARCH
    # five times on phantom yaw sweeps while flying through the gate plane.
    # Fresh engulfing evidence now anchors the horizontal bearing and blocks
    # SEARCH while vertical stays unqualified (collective holds support).
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    engulfing = _track(
        "A",
        0.30,
        0.10,
        scale=1.0,  # bbox spans the whole frame
        clipping=(
            FrameEdge.LEFT | FrameEdge.RIGHT | FrameEdge.TOP | FrameEdge.BOTTOM
        ),
    )
    now = 100.06
    output = None
    for frame in range(45):  # ~1.5 s of engulfed frames
        controller.observe(_update([engulfing], frame_id=10 + frame), now_s=now)
        # pitch=SPAWN_PITCH: neutral attitude (zero F50 compensation); the
        # exact hold is the tilt-compensated spawn support.
        output = _command(controller, now + 0.005, pitch=SPAWN_PITCH)
        assert controller.state is not CleanCourseState.SEARCH
        assert output.thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)
        now += 0.033
    assert controller.state is CleanCourseState.PREDICT
    # Bearing anchored to the engulfing center; vertical never qualified
    # beyond the accepted-measurement horizon.
    assert controller.last_reliable_bearing[0] == pytest.approx(0.30)
    assert controller.current.y_axis.v == 0.0
    assert not output.vertical_qualified


def test_engulfing_anchor_expires_on_frozen_frames():
    # Flight 20260729T111003Z-visual-course-d52adcd4: a frozen engulfing
    # frame (fid 2762570) republished for 6.1 s kept refreshing the anchor,
    # so it never expired, SEARCH was suppressed, and the loop flew blind
    # into a ground graze.  The anchor refreshes only on a NEW camera frame;
    # a republished frozen frame lets it expire and SEARCH proceeds.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    engulfing = _track(
        "A",
        0.30,
        0.10,
        scale=1.0,  # bbox spans the whole frame
        clipping=(
            FrameEdge.LEFT | FrameEdge.RIGHT | FrameEdge.TOP | FrameEdge.BOTTOM
        ),
    )
    controller.observe(_update([engulfing], frame_id=10), now_s=100.06)
    assert controller._last_engulfing_anchor_s == pytest.approx(100.06)
    now = 100.06
    entered = False
    for _ in range(60):  # ~2 s of the SAME frozen frame republished
        now += 0.033
        controller.observe(_update([engulfing], frame_id=10), now_s=now)
        if controller.state is CleanCourseState.SEARCH:
            entered = True
            break
    assert entered
    # The frozen republication never refreshed the anchor timestamp.
    assert controller._last_engulfing_anchor_s == pytest.approx(100.06)


def test_engulfing_anchor_frozen_production_token_does_not_refresh():
    # The production wrapper replay (d52adcd4, fid 2762570): a real
    # CameraFrameToken whose publication_sequence strictly advances while
    # (generation, frame_id) stays frozen is the SAME camera frame, so it
    # must not refresh the anchor.  _frame_identity keys on
    # (generation, frame_id) only; generation advances solely on a receiver
    # reset (vq2_vision.py), never on republication.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    engulfing = _track(
        "A",
        0.30,
        0.10,
        scale=1.0,
        clipping=(
            FrameEdge.LEFT | FrameEdge.RIGHT | FrameEdge.TOP | FrameEdge.BOTTOM
        ),
    )

    def production_update(publication_sequence):
        return SimpleNamespace(
            tracks=(engulfing,),
            visible_track_ids=(engulfing.track_id,),
            token=CameraFrameToken(
                generation=0,
                frame_id=2762570,
                publication_sequence=publication_sequence,
                stream_id="camera",
            ),
        )

    controller.observe(production_update(41), now_s=100.06)
    assert controller._last_engulfing_anchor_s == pytest.approx(100.06)
    now = 100.06
    entered = False
    for sequence in range(42, 102):  # ~2 s of frozen-frame republications
        now += 0.033
        controller.observe(production_update(sequence), now_s=now)
        if controller.state is CleanCourseState.SEARCH:
            entered = True
            break
    assert entered
    assert controller._last_engulfing_anchor_s == pytest.approx(100.06)


def test_engulfing_anchor_requires_a_visible_track():
    # Flight 20260729T112603Z-visual-course-d5e89c2b: the camera kept
    # streaming (the watchdog never fired) but the crossed gate's
    # authoritative-current track went missed and was never retired
    # (vq2_visual_tracker.py keeps race-authoritative identity past
    # max_missed_frames).  Its frozen engulfing bbox re-published on every
    # FRESH frame (advancing frame identity) and refreshed the anchor for
    # ~4 s, so no frame-identity freshness gate could ever fire.  A missed
    # (invisible) track's bbox is stale content: it must never anchor.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    ghost = _track(
        "A",
        0.30,
        0.10,
        scale=1.0,
        visible=False,  # missed track: tracker re-emits the last bbox
        clipping=(
            FrameEdge.LEFT | FrameEdge.RIGHT | FrameEdge.TOP | FrameEdge.BOTTOM
        ),
    )
    now = 100.06
    entered = False
    for frame in range(60):  # ~2 s of genuinely FRESH frames carrying the ghost
        now += 0.033
        controller.observe(
            SimpleNamespace(
                tracks=(ghost,),
                visible_track_ids=(),  # nothing associated this frame
                token=CameraFrameToken(
                    generation=0,
                    frame_id=2791388 + frame,
                    publication_sequence=100 + frame,
                    stream_id="camera",
                ),
            ),
            now_s=now,
        )
        if controller.state is CleanCourseState.SEARCH:
            entered = True
            break
    assert entered
    assert controller._last_engulfing_anchor_s is None


def test_predict_stall_cap_forces_search_regardless_of_anchor():
    # The d5e89c2b last-resort bound: even with a freshly refreshed anchor,
    # PREDICT older than 1.5 s without an accepted measurement forces SEARCH
    # from command() (observe-side anchor expiry can be suppressed forever).
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.observe(_update([], frame_id=2), now_s=100.12)  # superseded
    assert controller.state is CleanCourseState.PREDICT
    # A fresh anchor plus a sub-cap measurement gap: no transition.
    controller._last_engulfing_anchor_s = 100.50
    _command(controller, 100.55)
    assert controller.state is CleanCourseState.PREDICT
    # Anchor refreshed again but the accepted measurement is now 1.6 s old.
    controller._last_engulfing_anchor_s = 101.63
    output = _command(controller, 101.65)
    assert controller.state is CleanCourseState.SEARCH
    assert output.state is CleanCourseState.SEARCH
    assert abs(output.yaw_rate_rad_s) > 0.0  # real bounded sweep, not a park
    assert output.thrust > 0.0


def _promote_to_gate_one(controller, now_s=100.10):
    """Promote a TRACK controller to gate 1 with successor B accepted."""

    controller.observe(
        _update(
            [_track("A", 0.0, 0.0), _track("B", 0.30, 0.05, scale=0.05)],
            frame_id=3,
        ),
        now_s=now_s - 0.02,
    )
    # F42: promotion credibility requires a persistent successor; seed the
    # association age as if B had been tracked through the approach.
    controller._track_first_seen_s["B"] = now_s - 1.0
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=2500, now_s=now_s
    )
    assert promoted
    assert controller.current.track_id == "B"


def test_gate_one_track_close_loss_predicts_instead_of_coasting():
    # 2026-07-30 unified crossing policy: the exact-zero credit wait
    # (COAST) is reserved for gate 0's proven close-loss path and for a
    # COMMIT-phase close loss at the plane.  A gate-1+ TRACK close loss is
    # a tracking failure, not a crossing: the controller PREDICTs on the
    # hypothesis instead of entering a blind zero-thrust wait mid-course.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.observe(
        _update(
            [
                _track("A", 0.0, 0.0, scale=0.50),
                # Aligned (F45): the crossing coast arms only near center.
                _track("B", 0.10, 0.05, scale=0.50),
            ],
            frame_id=3,
        ),
        now_s=100.08,
    )
    # F42: promotion requires a persistent successor; seed the age.
    controller._track_first_seen_s["B"] = 100.08 - 1.0
    assert controller.note_race(gate_index=1, race_boot_ms=2500, now_s=100.10)
    # Fresh empty frames past the predict gap: the gate-1+ close loss falls
    # to PREDICT (and on to SEARCH as the gap grows) — it must NEVER latch
    # the blind exact-zero credit wait.
    now = 100.10
    seen = set()
    for frame in range(4, 14):  # ~0.33 s without a measurement
        now += 0.033
        controller.observe(_update([], frame_id=frame), now_s=now)
        seen.add(controller.state)
        assert controller.state is not CleanCourseState.COAST_FOR_CREDIT
    assert CleanCourseState.PREDICT in seen


def test_alt_est_clamped_so_biased_integrator_cannot_run_away():
    # F13's alt_est reached -10.7 m (physically impossible).  The estimate
    # is clamped below at -2.0 m — the F92 latch removal keeps the clamp:
    # the estimate still feeds the trace and must stay physically bounded.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 0.0
    controller._vz_est_m_s = -10.0  # F13-scale biased sink
    now = 100.10
    for _ in range(10):  # unclamped integration would reach -3.3 m
        now += 0.033
        _command(controller, now)
    assert controller._alt_est_m == -2.0


def test_fh_trigger_clears_the_brake_regime():
    # F50 (flight 20260729T222920Z-visual-course-3a8ed087): the F49 TRUE
    # brake reads fh 3.0-3.4 — a hard brake IS horizontal specific force —
    # and the 3.0 trigger tripped on the brake itself, pinning the
    # collective at the support + 0.05 floor for the whole gate-1 leg.
    # The F14 biased regime measured 6.5-7.5; 5.0 separates them.
    assert FH_UNTRUSTED_TRIGGER_MPS2 == 5.0
    assert CleanCourseConfig().fh_untrusted_trigger_mps2 == 5.0


def test_fh_gate_engages_only_after_sustained_excess_not_a_transient():
    # F14: the regime gate latches only on fh above the trigger SUSTAINED
    # for 0.3 s; a transient excursion must never freeze the vertical
    # estimate.
    # F50: trigger raised 3.0 -> 5.0 — the F49 hard brake reads fh 3.0-3.4
    # (a hard brake IS horizontal specific force) and tripped its own
    # distrust alarm; the F14 biased regime measured 6.5-7.5.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    now = 100.10
    for _ in range(15):  # 0.5 s at brake-level fh 3.4: must NEVER latch
        now += 0.033
        _command(controller, now, fh=3.4)
    assert not controller._fh_untrusted
    for _ in range(7):  # 0.23 s above the trigger: transient, no latch
        now += 0.033
        _command(controller, now, fh=6.0)
    assert not controller._fh_untrusted
    now += 0.033
    _command(controller, now, fh=1.0)  # below the trigger: timer resets
    for _ in range(7):  # another 0.23 s: still only a transient
        now += 0.033
        _command(controller, now, fh=6.0)
    assert not controller._fh_untrusted
    for _ in range(4):  # 0.36 s continuous: the gate latches
        now += 0.033
        _command(controller, now, fh=6.0)
    assert controller._fh_untrusted


def test_fh_gate_releases_only_below_hysteresis_band():
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    now = 100.10
    for _ in range(11):  # sustained: latch untrusted
        now += 0.033
        _command(controller, now, fh=6.0)
    assert controller._fh_untrusted
    # Inside the 2.0-5.0 hysteresis band the latch holds.
    now += 0.033
    _command(controller, now, fh=2.5)
    assert controller._fh_untrusted
    # Below the release it clears and the sustain timer resets.
    now += 0.033
    _command(controller, now, fh=1.5)
    assert not controller._fh_untrusted
    assert controller._fh_above_since_s is None


def test_fh_freeze_suspends_vz_integration_leaks_to_zero_holds_alt():
    # F14: identical delivered rotor output gave accz -13.0 (slow regime,
    # real climb) vs -8.0 (fast regime) — a smooth fh-proportional DC
    # deficit.  While untrusted the biased a_up is never integrated: only
    # the 2.5 s leak relaxes the frozen vz toward 0, and alt_est is held.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller._vz_est_m_s = -4.36  # F14's phantom sink
    now = 100.10
    for _ in range(11):  # latch untrusted (a_up=None: vz held)
        now += 0.033
        _command(controller, now, fh=6.0)
    assert controller._fh_untrusted
    controller._alt_est_m = -1.0
    for _ in range(10):
        now += 0.033
        _command(controller, now, fh=6.0, a_up=-2.5)  # biased regime sample
    # Leak-only decay: -4.36 * (1 - dt/2.5)^10; integration would have
    # driven it to -5.19 instead.
    assert controller._vz_est_m_s == pytest.approx(
        -4.36 * (1.0 - 0.033 / 2.5) ** 10, abs=1e-9
    )
    assert controller._alt_est_m == -1.0  # frozen exactly


def test_fh_latch_does_not_switch_collective_owner_or_add_margin():
    # Braking can cross the fh trust threshold.  That transition may suspend
    # IMU integration, but it must not select a new thrust law or add the
    # historical +0.05 margin.  With unqualified vision, trusted IMU damping
    # is supporting evidence; once the regime latch invalidates it, the same
    # carried collective converges smoothly to support instead of preserving
    # or replacing the stale inertial request.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller.current.last_y_measurement_s = 99.0
    controller._vz_est_m_s = -0.40
    now = 100.10
    thrusts = []
    trust_states = []
    targets = []
    imu_deltas = []
    supports = []
    for frame in range(15):
        now += 0.033
        out = _command(
            controller,
            now,
            fh=(4.0 if frame < 3 else 6.0),
            pitch=SPAWN_PITCH,
        )
        thrusts.append(out.thrust)
        trust_states.append(not controller._fh_untrusted)
        targets.append(controller._last_vertical_collective_target)
        imu_deltas.append(controller._last_vertical_imu_delta)
        supports.append(controller._last_vertical_support)
    assert controller._fh_untrusted
    assert any(trust_states) and not trust_states[-1]
    expected = (
        SPAWN_SUPPORT + controller.config.vertical_imu_damping_gain * 0.40
    )
    for trusted, target, imu_delta, support in zip(
        trust_states, targets, imu_deltas, supports
    ):
        if trusted:
            assert target == pytest.approx(expected, abs=1e-9)
        else:
            assert imu_delta == pytest.approx(0.0, abs=1e-12)
            assert target == pytest.approx(support, abs=1e-12)
    assert max(thrusts) <= expected + 1e-9
    assert supports[-1] <= thrusts[-1] < expected
    assert max(
        abs(after - before) for before, after in zip(thrusts, thrusts[1:])
    ) < 0.006


def test_partial_accel_trust_qualifies_imu_once_at_control_boundary():
    # Trust is confidence, not a second acceleration scale.  An accepted
    # half-trust sample keeps physical magnitude in the leaky estimate and is
    # weighted exactly once when that estimate contributes supporting damping.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller.current.last_y_measurement_s = 99.0
    _command(
        controller,
        100.10,
        pitch=SPAWN_PITCH,
        a_up=-1.0,
        accel_trust=0.5,
    )

    assert controller._last_vertical_imu_trust == pytest.approx(0.5)
    # Scaling at integration and again at damping would leave <0.01 m/s here.
    assert controller._vz_est_m_s < -0.015
    assert controller._last_vertical_imu_delta == pytest.approx(
        0.5 * controller._last_vertical_imu_raw_delta,
        abs=1e-12,
    )
    assert controller._last_vertical_collective_target == pytest.approx(
        controller._last_vertical_support
        + controller._last_vertical_imu_delta,
        abs=1e-12,
    )


def test_f162_first_bottom_frame_immediately_owns_vertical_direction():
    controller = _f162_gate1_controller()
    controller._collective = 0.2762406
    previous = None
    for row in _F162_DIRECTION_ROWS:
        if row[0] == 5.453:
            controller._collective = 0.27866674133727154
        previous = _apply_f162_direction_row(controller, row)

    now = 105.640
    old_y_stamp = controller.current.last_y_measurement_s
    clipped = _track(
        "A",
        -0.11875,
        0.3222222222222222,
        scale=math.sqrt(0.3953125 * 0.6777777777777778),
        confidence=0.9827799254573163,
        clipping=FrameEdge.BOTTOM,
        center_censored=True,
    )
    clipped.bbox_norm = (
        -0.11875 - 0.3953125 / 2.0,
        0.3222222222222222 - 0.6777777777777778 / 2.0,
        -0.11875 + 0.3953125 / 2.0,
        0.3222222222222222 + 0.6777777777777778 / 2.0,
    )
    controller.observe(
        _update([clipped], frame_id=1907102),
        now_s=now,
    )
    current = controller.current
    # Recorded post-observation controller state for the first clipped frame.
    current.y_axis.p = 0.3222222222222222
    current.y_axis.v = 0.4516339891684342
    current.y_axis.pp = 0.1600567239373518**2
    current.scale_axis.p = -0.6205430603424339
    current.scale_axis.v = 1.281392419069951
    current.scale_axis.vv = 0.02**2
    current.outer_log_scale = -0.6585072468915756
    current.outer_expansion_rate = 0.9554478585024939
    controller._vz_est_m_s = -0.18594823557539994

    out = _command(
        controller,
        now,
        pitch=-0.43438793458248665,
    )

    # The old exact y is only 47 ms old, but BOTTOM is the current frame and
    # therefore wins immediately (F162 incorrectly waited until t=5.843).
    assert current.last_y_measurement_s == pytest.approx(old_y_stamp)
    assert current.last_x_measurement_s == pytest.approx(now)
    assert not out.vertical_qualified
    assert controller._last_vertical_motion.directional_censor == (
        FrameEdge.BOTTOM
    )
    assert controller._vertical_direction_source == "bottom_censor"
    assert controller._vertical_direction_sign == 1
    assert abs(controller._last_vertical_imu_delta) <= (
        controller.config.vertical_imu_max_opposition_fraction
        * max(
            abs(controller._last_vertical_visual_delta),
            abs(controller._last_vertical_motion_delta),
        )
        + 1e-12
    )
    assert (
        controller._last_vertical_visual_delta
        + controller._last_vertical_imu_delta
        < 0.0
    )
    assert controller._last_vertical_collective_target < (
        controller._last_vertical_support
    )
    assert out.thrust < controller._last_vertical_support
    assert out.thrust < previous.thrust
    # Directional y censorship does not invalidate the fresh horizontal axis.
    assert out.yaw_rate_rad_s < 0.0
    assert out.target_roll_rad < 0.0
    assert controller.state is CleanCourseState.TRACK
    assert controller.gate_index == 1
    assert controller._commit_entry_s is None


@pytest.mark.parametrize(
    ("edge", "censored_y", "opposing_vz", "vertical_sign"),
    [
        (FrameEdge.TOP, -0.50, 10.0, 1.0),
        (FrameEdge.BOTTOM, 0.50, -10.0, -1.0),
    ],
)
def test_directional_vertical_censor_preserves_horizontal_and_visual_authority(
    edge, censored_y, opposing_vz, vertical_sign
):
    # Production's center_censored bit is aggregate metadata: TOP/BOTTOM must
    # censor only y.  Repeated clipped frames therefore keep refreshing x and
    # steering laterally while the one-sided vertical inequality commands in
    # the correct direction.  Even an extreme opposing IMU-vz estimate is
    # supporting damping and cannot reverse that clear directional evidence.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    old_y_stamp = controller.current.last_y_measurement_s
    now = 100.05
    out = None
    for frame in range(12):
        now += 0.033
        controller.observe(
            _update(
                [
                    _track(
                        "A",
                        0.20,
                        censored_y,
                        clipping=edge,
                        center_censored=True,
                    )
                ],
                frame_id=20 + frame,
            ),
            now_s=now,
        )
        controller._vz_est_m_s = opposing_vz
        out = _command(controller, now + 0.005, pitch=SPAWN_PITCH)

    assert not out.vertical_qualified
    assert controller.current.raw_x == pytest.approx(0.20, abs=1e-12)
    assert controller.current.last_x_measurement_s == pytest.approx(
        now, abs=1e-12
    )
    assert controller.current.last_y_measurement_s == old_y_stamp
    assert out.yaw_rate_rad_s > 0.0
    assert out.target_roll_rad > 0.0
    assert controller._last_vertical_motion.directional_censor == edge
    assert vertical_sign * (out.thrust - SPAWN_SUPPORT) > 0.0


def test_stale_directional_censor_has_no_optical_authority():
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller.current.y_axis.p = 0.50
    controller.current.vertical_censor_bound = 0.50
    controller.current.vertical_censor_edge = FrameEdge.BOTTOM
    controller.current.last_y_measurement_s = 99.0
    controller.current.last_measurement_s = 99.0
    controller._vz_est_m_s = 0.0

    out = _command(controller, 100.60, pitch=SPAWN_PITCH)

    assert not out.vertical_qualified
    assert controller._last_vertical_motion is None
    assert out.thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)


def _replay_outer_growth(
    controller,
    *,
    track_id,
    x,
    y,
    start_scale,
    closure_rate_s,
    now_s,
    frames,
    frame_id=10,
    clipping=FrameEdge.NONE,
    pitch=SPAWN_PITCH,
):
    """Replay coherent public outer-box observations at one log-scale rate."""

    samples = []
    scale = float(start_scale)
    for offset in range(frames):
        now_s += 0.033
        scale *= math.exp(float(closure_rate_s) * 0.033)
        controller.observe(
            _update(
                [
                    _track(
                        track_id,
                        x,
                        y,
                        scale=scale,
                        clipping=clipping,
                    )
                ],
                frame_id=frame_id + offset,
            ),
            now_s=now_s,
        )
        output = _command(controller, now_s, pitch=pitch)
        samples.append(
            (
                output,
                controller.current.outer_expansion_rate,
                controller.current.outer_scale_axis.v,
                controller._control_closure_estimate(
                    controller.current, now_s
                )[0],
                controller._pitch_energy_brake_demand,
                controller._pre_cross_brake_active,
            )
        )
    return samples, now_s, scale


def test_closure_governor_full_brake_at_high_expansion_rate():
    # F166: approach energy is owned by coherent outer-box motion.  A sustained
    # rate above the full-brake bound must converge continuously to the one
    # course brake reference without requiring an injected derivative state.
    controller = _tracked_controller(_track("A", 0.20, 0.0, scale=0.10))
    samples, _now, _scale = _replay_outer_growth(
        controller,
        track_id="A",
        x=0.20,
        y=0.0,
        start_scale=0.10,
        closure_rate_s=0.70,
        now_s=100.033,
        frames=30,
    )
    targets = [sample[0].target_pitch_rad for sample in samples]
    demands = [sample[4] for sample in samples]
    out = samples[-1][0]
    brake_reference = (
        controller.config.spawn_pitch_rad
        + controller.config.pre_cross_brake_pitch_rad
    )

    assert controller._pre_cross_brake_active
    assert out.state is CleanCourseState.TRACK
    assert demands[-1] > 0.98
    assert out.target_pitch_rad == pytest.approx(brake_reference, abs=0.003)
    assert max(abs(b - a) for a, b in zip(targets, targets[1:])) <= (
        controller.config.pre_cross_brake_slew_rad_s * 0.033 + 1e-12
    )
    assert out.yaw_rate_rad_s > 0.0  # x=+0.20 pursuit stays alive
    assert out.thrust > 0.0  # the vertical owner keeps collective alive


def test_authoritative_promotion_does_not_switch_brake_reference():
    # F124/F166: authoritative promotion must not select a second/deeper
    # brake law.  Build a persistent successor through public observations,
    # promote it through race ownership, and compare its converged response
    # with the same outer evidence on Gate 0.
    gate_one = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    now = 100.033
    for frame in range(20):
        now += 0.033
        gate_one.observe(
            _update(
                [
                    _track("A", 0.0, 0.0, scale=0.10),
                    _track("B", -0.40, 0.05, scale=0.10),
                ],
                frame_id=10 + frame,
            ),
            now_s=now,
        )
        _command(gate_one, now, pitch=SPAWN_PITCH)
    assert gate_one.note_race(
        gate_index=1, race_boot_ms=2500, now_s=now + 0.001
    )

    gate_zero = _tracked_controller(
        _track("A", -0.40, 0.05, scale=0.10)
    )
    zero_samples, _zero_now, _zero_scale = _replay_outer_growth(
        gate_zero,
        track_id="A",
        x=-0.40,
        y=0.05,
        start_scale=0.10,
        closure_rate_s=0.70,
        now_s=100.033,
        frames=30,
    )
    one_samples, _one_now, _one_scale = _replay_outer_growth(
        gate_one,
        track_id="B",
        x=-0.40,
        y=0.05,
        start_scale=0.10,
        closure_rate_s=0.70,
        now_s=now + 0.001,
        frames=30,
        frame_id=100,
    )
    zero_out = zero_samples[-1][0]
    one_out = one_samples[-1][0]
    brake_reference = (
        gate_one.config.spawn_pitch_rad
        + gate_one.config.pre_cross_brake_pitch_rad
    )

    assert gate_zero._pre_cross_brake_active
    assert gate_one._pre_cross_brake_active
    assert one_out.state is CleanCourseState.TRACK
    assert zero_out.target_pitch_rad == pytest.approx(
        brake_reference, abs=0.003
    )
    assert one_out.target_pitch_rad == pytest.approx(
        zero_out.target_pitch_rad, abs=0.003
    )
    assert one_out.yaw_rate_rad_s < 0.0  # current-gate pursuit stays alive


def test_closure_governor_does_not_brake_below_target_rate():
    # Slow closure is free flight: below the 0.35/s target rate the
    # governor contributes nothing and the advance law still closes.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.current.scale_axis.v = 0.1
    out = _command(controller, 100.10, pitch=SPAWN_PITCH)
    assert not controller._pre_cross_brake_active
    # Advance law still closes: above the level (spawn) brake base.
    assert out.target_pitch_rad > SPAWN_PITCH


def test_closure_governor_distrusts_tiny_track_expansion():
    # F33 (64050a81): post-credit, gate 1 (span 0.03-0.04, log_scale ~-2.9)
    # "grew" at 0.9/s — sub-pixel noise on a far tiny track — and pinned a
    # +0.12 brake with aw_fwd -5 m/s^2 for the whole leg, reversing the
    # drone into gate 0's structure.  Below CLOSURE_MIN_LOG_SCALE the
    # governor stays out no matter how large the reported expansion is.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.05))
    controller.current.scale_axis.v = 0.9  # noise-level expansion
    out = _command(controller, 100.10, pitch=SPAWN_PITCH)
    assert not controller._pre_cross_brake_active
    # Advance law still closes: above the level (spawn) brake base.
    assert out.target_pitch_rad > SPAWN_PITCH


def test_gate_zero_launch_pitch_has_no_scale_modality_handoff():
    # F166: launch no longer swaps from raw outer expansion to an aperture-
    # eligible filtered derivative at 0.40 s.  Replay two identical public
    # outer-box streams through that boundary; only the fitted aperture log
    # scale differs.  Aperture noise may shape passage geometry, but cannot
    # change approach energy, pitch, or brake mode on either side of launch.
    def replay(aperture_logs):
        controller = CleanCourseController(
            _config(launch_boost_duration_s=LAUNCH_BOOST_DURATION_S)
        )
        outputs = []
        demands = []
        modes = []
        for index, aperture_log in enumerate(aperture_logs):
            now = 100.0 + 0.033 * index
            track = _f163_trace_track(
                outer_center=(0.0, 0.0),
                outer_span=(0.13, 0.23),
                aperture_center=(0.0, 0.0),
                aperture_half=(0.05, 0.10),
                aperture_log_scale=aperture_log,
            )
            update = _update([track], frame_id=5000 + index)
            if index == 0:
                controller.initialize(
                    update,
                    gate_index=0,
                    fallback_center_norm=(0.0, 0.0),
                    fallback_apparent_scale=math.sqrt(0.13 * 0.23),
                    now_s=now,
                )
            else:
                controller.observe(update, now_s=now)
            outputs.append(_command(controller, now, pitch=SPAWN_PITCH))
            demands.append(controller._pitch_energy_brake_demand)
            modes.append(controller._pitch_energy_brake_active)
        return controller, outputs, demands, modes

    count = 24  # crosses both the 0.40 s warmup and 0.25 s transfer
    nominal_log = 0.5 * math.log(0.05 * 0.10)
    stable = replay([nominal_log] * count)
    noisy = replay(
        [nominal_log + (0.65 if index % 2 else -0.65) for index in range(count)]
    )

    assert [out.target_pitch_rad for out in noisy[1]] == pytest.approx(
        [out.target_pitch_rad for out in stable[1]], abs=1e-12
    )
    assert noisy[2] == pytest.approx(stable[2], abs=1e-12)
    assert noisy[3] == stable[3]
    assert not any(noisy[3])
    assert noisy[0]._control_closure_estimate(
        noisy[0].current, 100.0 + 0.033 * (count - 1)
    )[0] == pytest.approx(0.0, abs=1e-12)


def test_closure_governor_is_a_continuous_blend():
    # Mid-band closure: the pitch target blends partway from the advance
    # law (spawn base) toward the spawn-0.15 Gate-0 brake attitude,
    # without latching the fast-slew brake flag.
    # F101: at log -1.40 the range-ramped target is 0.30, so 0.375/s sits
    # inside the continuous response band below the 0.60 full-brake rate.
    # (The commit
    # regime itself levels a centered gate via the F94 custody floor —
    # no blend is observable there.)
    controller = _tracked_controller(
        _track("A", 0.0, 0.0, scale=math.exp(-1.40))
    )
    controller.current.scale_axis.v = 0.375
    now = 100.10
    out = None
    for _ in range(25):  # generic slew converges to the blended target
        now += 0.033
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert not controller._pre_cross_brake_active
    assert (
        SPAWN_PITCH - 0.15 + 1e-9
        < out.target_pitch_rad
        < SPAWN_PITCH - 1e-9
    )


def test_misalignment_brake_is_invariant_to_its_own_camera_pitch():
    # F116 Gate 0 began with a small physical vertical error.  The raw-y
    # brake pitched the camera up, which moved the gate down in-frame and
    # recursively demanded more nose-up brake until the top-structure hit.
    # The same world-relative gate error must produce the same forward law at
    # level and at a nose-up camera attitude; thrust remains the only vertical
    # translation channel.
    physical_ey = 0.05

    def _at_pitch(pitch_rad):
        raw_ey = physical_ey + (
            (SPAWN_PITCH - pitch_rad) * 1.6
        )
        controller = _tracked_controller(
            _track("A", 0.0, raw_ey, scale=math.exp(-1.50)),
            config=_config(
                target_slew_rad_s=100.0,
                pre_cross_brake_slew_rad_s=100.0,
            ),
        )
        controller.current.scale_axis.v = 0.0
        controller.current.outer_expansion_rate = 0.0
        out = _command(controller, 100.10, pitch=pitch_rad)
        assert controller._compensated_ey(
            controller.current.y, pitch_rad
        ) == pytest.approx(physical_ey, abs=1e-9)
        return controller, out

    level, level_out = _at_pitch(SPAWN_PITCH)
    braked, braked_out = _at_pitch(-0.44)

    assert braked.current.y > level.current.y + 0.20  # camera artifact exists
    assert braked_out.advance_factor == pytest.approx(
        level_out.advance_factor, abs=1e-9
    )
    assert braked_out.target_pitch_rad == pytest.approx(
        level_out.target_pitch_rad, abs=1e-9
    )
    assert braked._pre_cross_brake_active == level._pre_cross_brake_active


def test_misaligned_far_top_censored_gate_keeps_vertical_and_lateral_evidence():
    # A far off-axis target still owns the continuous pitch trajectory and
    # coordinated turn.  Fresh TOP observations are one-sided evidence that
    # the gate remains above the frame; pitch filtering must not erase either
    # the climb direction or the still-valid horizontal pursuit.
    controller = _tracked_controller(
        _track("A", 0.50, -0.80, scale=0.05, clipping=FrameEdge.TOP)
    )
    samples, _now, _scale = _replay_outer_growth(
        controller,
        track_id="A",
        x=0.50,
        y=-0.80,
        start_scale=0.05,
        closure_rate_s=0.0,
        now_s=100.033,
        frames=30,
        clipping=FrameEdge.TOP,
    )
    out = samples[-1][0]
    brake_reference = (
        controller.config.spawn_pitch_rad
        + controller.config.pre_cross_brake_pitch_rad
    )
    assert controller._pre_cross_brake_active
    assert samples[-1][4] > 0.98
    assert out.target_pitch_rad == pytest.approx(brake_reference, abs=0.005)
    # Top-clipped gate: y is not an exact measurement, but the fresh
    # directional inequality retains vertical authority.
    assert not out.vertical_qualified
    assert controller._last_vertical_path_error < 0.0
    assert out.thrust > SPAWN_SUPPORT
    assert out.thrust < controller.config.max_thrust
    # Raised yaw gain: 0.9 * 0.50 pursuit, clamped to the 0.15 production
    # yaw command cap.
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)
    assert out.target_roll_rad > 0.0


def test_closure_governor_brakes_in_predict():
    # The outer-led energy trajectory carries through a bounded PREDICT gap.
    # Establish the rate with real frames first; dropout may age the raw EMA,
    # but the coherent filtered outer state must not switch back to advance.
    controller = _tracked_controller(_track("A", 0.20, 0.0, scale=0.10))
    growth, now, _scale = _replay_outer_growth(
        controller,
        track_id="A",
        x=0.20,
        y=0.0,
        start_scale=0.10,
        closure_rate_s=0.70,
        now_s=100.033,
        frames=20,
        frame_id=20,
    )
    pitch_before_dropout = growth[-1][0].target_pitch_rad
    demand_before_dropout = growth[-1][4]
    assert controller._pre_cross_brake_active
    for frame in range(9):  # ~0.3 s without a measurement -> PREDICT
        now += 0.033
        controller.observe(_update([], frame_id=60 + frame), now_s=now)
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.PREDICT
    assert controller._pre_cross_brake_active
    assert controller._pitch_energy_brake_demand >= demand_before_dropout - 0.05
    assert out.target_pitch_rad <= pitch_before_dropout + 0.005
    assert out.target_pitch_rad < controller.config.spawn_pitch_rad - 0.10


def test_pitch_offsets_follow_the_configured_spawn_attitude():
    # F49/F166: the filtered trajectory is spawn-relative at every point, not
    # only at its asymptotic brake endpoint.  Replay identical outer energy
    # against two measured spawn attitudes and compare the whole response.
    default = _tracked_controller(
        _track("A", 0.20, 0.0, scale=0.10),
        config=_config(spawn_pitch_rad=SPAWN_PITCH),
    )
    shifted_config = _config(spawn_pitch_rad=-0.20)
    shifted = _tracked_controller(
        _track("A", 0.20, 0.0, scale=0.10), config=shifted_config
    )
    default_samples, _now, _scale = _replay_outer_growth(
        default,
        track_id="A",
        x=0.20,
        y=0.0,
        start_scale=0.10,
        closure_rate_s=0.70,
        now_s=100.033,
        frames=30,
        pitch=SPAWN_PITCH,
    )
    shifted_samples, _now, _scale = _replay_outer_growth(
        shifted,
        track_id="A",
        x=0.20,
        y=0.0,
        start_scale=0.10,
        closure_rate_s=0.70,
        now_s=100.033,
        frames=30,
        frame_id=100,
        pitch=shifted_config.spawn_pitch_rad,
    )

    assert default._pre_cross_brake_active
    assert shifted._pre_cross_brake_active
    for default_sample, shifted_sample in zip(
        default_samples, shifted_samples
    ):
        assert (
            shifted_sample[0].target_pitch_rad
            - default_sample[0].target_pitch_rad
        ) == pytest.approx(0.11, abs=1e-9)
    # -0.20 spawn + the common -0.15 pre-cross reference = -0.35.
    assert shifted_samples[-1][0].target_pitch_rad == pytest.approx(
        -0.35, abs=0.003
    )


def test_heading_anchor_clamps_outward_yaw_only():
    # F31: post-loss search/edge-chase wound the heading +2.63 rad off the
    # course bearing, then the drone flew sideways into structure it never
    # saw.  The anchor is captured lazily on the first yaw tick; outward
    # steering past the 1.5 rad cap is blocked, return steering is free.
    controller = _tracked_controller(_track("A", 0.30, 0.0, scale=0.10))
    out = _command(controller, 100.10, yaw=0.0)  # lazy anchor = 0.0
    assert controller._course_anchor_yaw_rad == pytest.approx(0.0)
    assert out.yaw_rate_rad_s > 0.0  # x=+0.30 pursuit steers freely
    # Heading wound past the cap: the same positive pursuit is blocked.
    out = _command(controller, 100.143, yaw=1.6)
    assert out.yaw_rate_rad_s <= 0.0
    # Return steering (target left of center) is always free at the cap.
    controller.current.x_axis.p = -0.30
    out = _command(controller, 100.176, yaw=1.6)
    assert out.yaw_rate_rad_s < 0.0
    # Wrapped excursion: yaw just past -pi relative to a +pi anchor is a
    # small positive excursion, not a full revolution.
    controller2 = _tracked_controller(_track("A", -0.30, 0.0, scale=0.10))
    _command(controller2, 100.10, yaw=math.pi - 0.1)
    out = _command(controller2, 100.143, yaw=-math.pi + 0.05)
    assert out.yaw_rate_rad_s < 0.0  # excursion +0.15: steering unaffected
    # Promotion re-arms the anchor for the new leg.
    _promote_to_gate_one(controller)
    assert controller._course_anchor_yaw_rad is None


def test_search_holds_level_pitch_even_when_blind_and_fast():
    # F49: the F31/F40 blind-at-speed brake was built on ABSOLUTE pitch
    # targets ~0.3 rad nose-down of intent — level flight is the -0.31
    # spawn attitude, not 0, so the "-0.22 blind brake" was in truth a
    # slight dive and never killed the drift.  Under the spawn-relative
    # convention SEARCH always holds the level (spawn) pitch at any fh;
    # the gentle sweep and the vz governor carry the blind leg.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller._enter_search(100.10)
    controller._fh_mps2 = 0.5  # slow: level
    out = _command(controller, 100.143)
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    controller._fh_mps2 = 3.0  # blind and fast: still level, no brake pitch
    now = 100.143
    for _ in range(40):  # the slew never leaves the level attitude
        now += 0.033
        out = _command(controller, now)
    assert out.state is CleanCourseState.SEARCH
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    assert abs(out.yaw_rate_rad_s) > 0.0  # the sweep stays alive


def test_search_level_pitch_held_from_the_first_tick():
    # F49: the slew state is seeded at spawn + brake offset, so the level
    # attitude is held from the first tick of even a typical 0.35-0.8 s
    # search — no dedicated fast slew is needed (the F43 slew problem
    # existed only because the blind brake target was far from level).
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller._enter_search(100.10)
    controller._fh_mps2 = 3.0  # blind and fast
    now = 100.10
    out = None
    for _ in range(10):  # ~0.33 s — a typical short search
        now += 0.033
        out = _command(controller, now)
    assert out.state is CleanCourseState.SEARCH
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)


def test_search_without_bearing_memory_holds_support():
    # F75: no-track SEARCH latches altitude support (the F50 memory
    # descent is removed); with no bearing evidence this is the same
    # plain support hold.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller._enter_search(100.10)
    controller._bearing_memory_valid = False
    out = _command(controller, 100.143, pitch=SPAWN_PITCH)
    assert out.thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)


def test_lone_small_fragment_creeps_instead_of_advancing():
    # F43: a lone small span is ambiguous range evidence — "whole gate far
    # away" or "fragment of a gate that is NEAR" — and the gate-1 leg built
    # fh 3-4 mps2 advancing at +0.08 on a span-(0.04,0.10) fragment.  Below
    # the span bound the leg creeps while centering; a whole gate (or fused
    # union, span above the bound) keeps the full advance law.
    fragment = _tracked_controller(_track("A", 0.0, 0.0, scale=0.06))
    now = 100.10
    out = None
    for _ in range(20):  # generic slew converges to the law target
        now += 0.033
        fragment.current.last_x_measurement_s = now
        out = _command(fragment, now, pitch=SPAWN_PITCH)
    # Never the full advance offset on fragment evidence: capped at the
    # creep offset (spawn + 0.03) while centering.
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH + 0.03, abs=1e-9)
    # Centering authority (yaw) is untouched on a fragment.
    offset = _tracked_controller(_track("A", 0.30, 0.0, scale=0.06))
    offset.current.last_x_measurement_s = 100.10
    assert _command(offset, 100.10, pitch=SPAWN_PITCH).yaw_rate_rad_s > 0.0
    # A confidently whole gate (span above the bound) advances fully.
    whole = _tracked_controller(_track("A", 0.0, 0.0, scale=0.20))
    now = 100.10
    out = None
    for _ in range(20):
        now += 0.033
        whole.current.last_x_measurement_s = now
        out = _command(whole, now, pitch=SPAWN_PITCH)
    assert out.target_pitch_rad > SPAWN_PITCH + 0.04  # full advance, no creep cap


def test_crossing_loss_latches_coast_even_while_fh_untrusted():
    # F32 (8cc53db2): fh went untrusted from BRAKING drag at the engulfed
    # gate-0 plane, the fh guard blocked the coast latch, and the drone
    # flew blind into the frame in PREDICT.  A credible close crossing loss
    # coasts regardless.  2026-07-30: the coast is exact wire zero for the
    # bounded credit wait — no support-thrust PD hold.  F102: the gate-0
    # hot-coast trigger is deleted; the coast arms only from an armed
    # COMMIT, so this test enters through the COMMIT latch.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller._fh_untrusted = True
    now = 100.10
    now += 0.033
    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = now
    controller.observe(_update([], frame_id=20), now_s=now)  # fresh close loss
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    output = _command(controller, now + 0.02)
    assert output.thrust == 0.0


def test_brake_ceiling_band_bounds_collective_while_braking():
    # The remaining Gate-0 crossing envelope bounds the one collective
    # target.  It is independent of fh trust and adds no alternate floor.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller._pre_cross_brake_active = True
    controller._fh_untrusted = False
    controller._vz_est_m_s = 0.0
    # A small correction inside the band passes through.
    assert controller._governed_collective(SUPPORT + 0.03, SUPPORT) == pytest.approx(
        SUPPORT + 0.03, abs=1e-9
    )
    # Deep sub-support demands are lifted to the band bottom (with the F49
    # 0.247 support the band bottom support-0.04 sits below the 0.21
    # min-thrust clamp, so the clamp is the effective floor):
    assert controller._governed_collective(0.20, SUPPORT) == pytest.approx(
        0.21, abs=1e-9
    )
    trusted = controller._governed_collective(0.30, SUPPORT)
    controller._fh_untrusted = True
    untrusted = controller._governed_collective(0.30, SUPPORT)
    assert trusted == pytest.approx(
        SUPPORT + 0.04, abs=1e-9
    )
    assert untrusted == pytest.approx(trusted, abs=1e-9)
    assert controller._governed_collective(0.34, SUPPORT) == pytest.approx(
        SUPPORT + 0.04, abs=1e-9
    )
    # Outside the crossing envelope the sole owner's request passes through;
    # changing fh trust does not alter it.
    controller._pre_cross_brake_active = False
    assert controller._governed_collective(0.28, SUPPORT) == pytest.approx(
        0.28, abs=1e-9
    )

    # Gate HIGH (ey < -0.10): the climb side remains uncapped while the
    # generic command bounds still apply.
    high = _tracked_controller(_track("A", 0.0, -0.50, scale=0.10))
    high._pre_cross_brake_active = True
    high._fh_untrusted = False
    high._vz_est_m_s = 0.0
    assert high._governed_collective(SUPPORT + 0.06, SUPPORT) == pytest.approx(
        SUPPORT + 0.06, abs=1e-9
    )
    assert high._governed_collective(0.20, SUPPORT) == pytest.approx(
        0.21, abs=1e-9  # band bottom below the 0.21 clamp (F49 support)
    )


def test_pre_cross_brake_does_not_suppress_crossing_detection():
    # Replay the shape of a normal crossing under braking: the apparent
    # size keeps growing through crossing_min_log_scale while the brake is
    # active, and the fresh close loss still latches the exact-zero COAST
    # wait (the brake's deceleration must not suppress crossing detection).
    # F102: the near-plane hold is gate-agnostic, so the brake engages at
    # gate 0 too (no aperture on the default track fails the entry budget);
    # the coast itself arms only from an armed COMMIT — enter via the
    # COMMIT latch.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.30))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    now = 100.10
    brake_seen = False
    for frame, scale in enumerate((0.32, 0.36, 0.40, 0.44, 0.48, 0.52)):
        now += 0.033
        controller.observe(
            _update([_track("A", 0.0, 0.0, scale=scale)], frame_id=10 + frame),
            now_s=now,
        )
        _command(controller, now)
        brake_seen = brake_seen or controller._pre_cross_brake_active
    assert brake_seen  # the brake engaged during the final approach
    assert controller.current.outer_log_scale >= -0.80  # crossing armed
    now += 0.033
    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = now
    controller.observe(_update([], frame_id=20), now_s=now)  # fresh close loss
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    output = _command(controller, now + 0.02)
    assert (
        output.target_roll_rad,
        output.target_pitch_rad,
        output.yaw_rate_rad_s,
    ) == (0.0, 0.0, 0.0)
    assert output.thrust == 0.0


def test_pre_cross_brake_custody_floor_scales_with_compensated_ey():
    # F94: the F51 binary relax/hysteresis latch is replaced by a
    # continuous custody floor — the pitch target never goes nose-up past
    # the attitude that places the compensated ey ON the 0.55 far-range
    # bound.  Measured at the level spawn attitude, compensated ey equals
    # raw ey, so the floor is spawn - (0.55 - ey) / 1.6: partial brake as
    # the gate nears the bound, level only when it is genuinely past it,
    # full brake when centered.  No hysteresis — the compensated ey is
    # attitude-invariant, so the floor cannot chatter.
    controller = _tracked_controller(_track("A", 0.0, 0.60, scale=0.08))
    now = 100.033
    scale = 0.08
    frame_id = 20

    def drive(y, ticks):
        nonlocal now, scale, frame_id
        samples, now, scale = _replay_outer_growth(
            controller,
            track_id="A",
            x=0.0,
            y=y,
            start_scale=scale,
            closure_rate_s=0.70,
            now_s=now,
            frames=ticks,
            frame_id=frame_id,
        )
        frame_id += ticks
        return samples[-1][0]

    out = drive(0.60, 20)
    assert controller._pre_cross_brake_active  # ey=0.60: fully misaligned
    # Past the 0.55 bound the gate is genuinely low: custody caps at level.
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    # Approaching the bound the floor admits a partial brake.
    out = drive(0.50, 10)
    assert out.target_pitch_rad == pytest.approx(
        SPAWN_PITCH
        - (0.55 - controller._last_vertical_path_error) / 1.6,
        abs=1e-9,
    )
    out = drive(0.40, 10)
    assert out.target_pitch_rad == pytest.approx(
        SPAWN_PITCH
        - (0.55 - controller._last_vertical_path_error) / 1.6,
        abs=1e-9,
    )
    # Centered: the floor sits below the Gate-0 brake attitude — full demand.
    out = drive(0.0, 10)
    assert controller._pre_cross_brake_active
    assert out.target_pitch_rad == pytest.approx(
        controller.config.spawn_pitch_rad
        + controller.config.pre_cross_brake_pitch_rad,
        abs=0.005,
    )


def test_near_plane_custody_floor_runs_on_the_stale_hypothesis():
    # F65 (20260730T021149Z-visual-course-08f41050): AT the plane the F51
    # guard never fired — the gate sat at ey +0.43 (below the 0.55 fresh
    # bound) and censorship froze measurement freshness, so the -0.46
    # pre-cross brake attitude pitched the gate out of the FOV; the drone
    # wandered blind for ~7 s into the floor/structure.  Inside the commit
    # proximity regime the F94 custody floor runs on the STALE derotated
    # hypothesis (no freshness gate): past the 0.30 bound it surrenders to
    # level, below it the brake is preserved proportionally.
    controller = _tracked_controller(_track("A", 0.30, 0.43, scale=0.50))
    controller.current.scale_axis.v = 0.7  # rapid expansion: full brake demand
    now = 100.10
    out = None
    for _ in range(20):
        now += 0.033
        # Measurements stay STALE (frozen at construction) — the commit
        # regime still reads the hypothesis.
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller._pre_cross_brake_active
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    # Below the bound the floor keeps a partial brake — no binary resume.
    controller.current.y_axis.p = 0.10
    for _ in range(20):
        now += 0.033
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert out.target_pitch_rad == pytest.approx(
        SPAWN_PITCH - (0.30 - 0.10) / 1.6, abs=1e-9
    )


def test_course_leg_custody_floor_relaxes_the_single_brake():
    # F71 (20260730T060005Z-visual-course-f05911e4): on the gate-1 leg the
    # 0.30 bound sat a hair above the achieved hypothesis ey (0.22-0.30
    # through the final second), so the brake attitude walked the gate into
    # engulfing at the plane.  Course legs (gate_index >= 1) use the tighter
    # 0.18 bound, and the F94 floor is continuous: more brake as the gate
    # recovers above the bound, level only when genuinely past it.
    controller = _tracked_controller(_track("A", 0.30, 0.22, scale=0.50))
    controller.gate_index = 1
    # Stopped (expansion 0): the brake demand comes from the ex 0.30
    # misalignment.
    controller.current.scale_axis.v = 0.0
    now = 100.10

    def drive(ticks=20):
        nonlocal now
        out = None
        for _ in range(ticks):
            now += 0.033
            out = _command(controller, now, pitch=SPAWN_PITCH)
        return out

    # ey 0.22 is past the course 0.18 bound: custody caps at level.
    out = drive()
    assert controller._pre_cross_brake_active
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    # ey 0.05: partial brake at the floor — no stickiness at level.
    controller.current.y_axis.p = 0.05
    assert drive().target_pitch_rad == pytest.approx(
        SPAWN_PITCH - (0.18 - 0.05) / 1.6, abs=1e-9
    )
    # ey -0.15 would admit a deeper attitude, but F125's one brake reference
    # remains the maximum request; custody is only allowed to relax it.
    controller.current.y_axis.p = -0.15
    assert drive().target_pitch_rad == pytest.approx(
        SPAWN_PITCH + controller.config.pre_cross_brake_pitch_rad, abs=1e-9
    )
    # Gate 0 keeps the wider 0.30 bound: the same ey 0.22 leaves MORE
    # brake authority, not less.
    gate0 = _tracked_controller(_track("A", 0.30, 0.22, scale=0.50))
    gate0.current.scale_axis.v = 0.7
    now0 = 100.10
    for _ in range(20):
        now0 += 0.033
        out0 = _command(gate0, now0, pitch=SPAWN_PITCH)
    assert out0.target_pitch_rad == pytest.approx(
        SPAWN_PITCH - (0.30 - 0.22) / 1.6, abs=1e-9
    )


def test_clipping_increases_uncertainty_but_does_not_abort():
    clipped = _tracked_controller(
        _track("A", 0.10, 0.0, clipping=FrameEdge.RIGHT)
    )
    clean = _tracked_controller(_track("A", 0.10, 0.0))
    now = 100.10
    for frame in range(5):
        clipped.observe(
            _update(
                [_track("A", 0.10, 0.0, clipping=FrameEdge.RIGHT)],
                frame_id=20 + frame,
            ),
            now_s=now,
        )
        clean.observe(
            _update([_track("A", 0.10, 0.0)], frame_id=20 + frame),
            now_s=now,
        )
        now += 0.033
    assert clipped.current.position_std > clean.current.position_std
    output = _command(clipped, now)
    assert output.state is CleanCourseState.TRACK
    assert all(
        math.isfinite(value)
        for value in (
            output.target_roll_rad,
            output.target_pitch_rad,
            output.yaw_rate_rad_s,
            output.thrust,
        )
    )
    # Clipping saturates corrective steering rather than aborting.
    # (F36 yaw gain 0.9: the clipped ex=0.10 pursuit commands 0.09.)
    assert abs(output.yaw_rate_rad_s) <= 0.10 * 0.9 + 1e-9


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


def test_predict_then_search_on_fresh_empty_frames():
    controller = _tracked_controller(_track("A", 0.05, 0.0, scale=0.10))
    now = 100.06
    controller.observe(_update([], frame_id=5), now_s=now)
    assert controller.state is CleanCourseState.TRACK  # one missed frame
    # Fresh-but-empty frames are not staleness; with the F36 adoption
    # hysteresis (0.25 s gap), ~8 missed frames drop TRACK to PREDICT.
    for frame in range(6, 16):
        now += 0.033
        controller.observe(_update([], frame_id=frame), now_s=now)
        if controller.state is CleanCourseState.PREDICT:
            break
    assert controller.state is CleanCourseState.PREDICT
    # The same controller keeps producing finite bounded commands on the
    # predicted state.
    output = _command(controller, now + 0.02)
    assert output.thrust > 0.0
    for frame in range(16, 48):
        now += 0.033
        controller.observe(_update([], frame_id=frame), now_s=now)
        if controller.state is CleanCourseState.SEARCH:
            break
    assert controller.state is CleanCourseState.SEARCH


def test_frozen_frame_stall_goes_to_predict_and_never_coasts():
    # Flight 20260729T085719Z-visual-course-4455fd61: the camera froze for
    # ~0.27 s (7 ticks, same frame id) at close scale and the tracker
    # republished the frozen frame; the stale close-range loss armed
    # COAST_FOR_CREDIT and latched zero thrust at the gate-0 top bar.  A
    # superseded/frozen frame must go to PREDICT with covariance inflation
    # and must never coast or zero the thrust.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    std_before = controller.current.position_std
    now = 100.12
    controller.observe(_update([], frame_id=2), now_s=now)  # superseded id
    assert controller.state is CleanCourseState.PREDICT
    assert controller.current.position_std > std_before
    for _ in range(7):  # the frozen ~0.27 s republication stall
        now += 0.033
        controller.observe(_update([], frame_id=2), now_s=now)
        assert controller.state is not CleanCourseState.COAST_FOR_CREDIT
        assert _command(controller, now + 0.005).thrust > 0.0
    # Continued fresh empty frames from PREDICT still never coast.
    for frame in range(3, 6):
        now += 0.033
        controller.observe(_update([], frame_id=frame), now_s=now)
        assert controller.state is not CleanCourseState.COAST_FOR_CREDIT
        assert _command(controller, now + 0.005).thrust > 0.0


def test_fresh_close_loss_still_coasts_at_exact_zero():
    # The July-18 bounded credible-crossing wait is preserved: a genuine
    # close-range loss on a FRESH frame (new frame id) still arms the coast
    # latch.  2026-07-30 contract correction: the wait is exact wire zero —
    # no roll/pitch/yaw rate, no thrust — for at most the credit window.
    # F102: the gate-0 hot-coast trigger is deleted — the latch arms only
    # from an armed COMMIT, so this test enters through the COMMIT latch.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = 100.12
    controller.observe(_update([], frame_id=3), now_s=100.12)  # fresh id
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    output = _command(controller, 100.14)
    assert (
        output.target_roll_rad,
        output.target_pitch_rad,
        output.yaw_rate_rad_s,
    ) == (0.0, 0.0, 0.0)
    assert output.thrust == 0.0


def test_search_issues_real_bounded_yaw_sweep():
    config = _config(search_sweep_period_s=0.10)
    controller = _tracked_controller(
        _track("A", 0.40, 0.0, scale=0.10), config=config
    )
    now = 100.10
    for _ in range(40):
        now += 0.033
        controller.observe(_update([], frame_id=9), now_s=now)
        if controller.state is CleanCourseState.SEARCH:
            break
    assert controller.state is CleanCourseState.SEARCH
    yaws = []
    for _ in range(20):
        now += 0.02
        yaws.append(_command(controller, now).yaw_rate_rad_s)
    # Real sweep: nonzero, at the bounded sweep rate, inside the yaw cap.
    assert all(abs(value) == pytest.approx(0.15) for value in yaws)
    assert all(abs(value) <= 0.15 + 1e-9 for value in yaws)
    # Initialized from the last image-right bearing under the measured
    # 2026-07-29 convention: positive yaw recenters a right-side target.
    assert yaws[0] > 0.0
    # Bounded schedule reverses the sweep.
    assert any(value < 0.0 for value in yaws)


def test_search_heading_sweep_starts_at_entry_heading_and_stays_alive():
    # F40 (20260729T193134Z-visual-course-63ed6342): the old incremental
    # sweep integrated the COMMANDED yaw, so at the anchor cap it parked at
    # yaw 1.94 rad — 111 deg off course — for ~7 blind seconds into gate 1.
    # F49: F40's absolute-heading sweep re-centered the scan on the LEG
    # ANCHOR instead of looking where the target was last seen.  The sweep
    # now starts from the heading measured at search entry, first moves
    # toward the last reliable bearing, keeps sweeping (sign changes) from
    # a parked heading, and stays bounded — it never re-centers on the
    # anchor.
    controller = _tracked_controller(_track("A", 0.40, 0.0, scale=0.10))
    controller._enter_search(100.10)
    controller._course_anchor_yaw_rad = 0.4  # deliberately NOT the entry heading
    yaw = 1.9  # parked heading, 1.5 rad off the anchor
    now = 100.10
    yaws = []
    headings = []
    for _ in range(400):  # ~8 s at 50 Hz, first-order heading plant
        now += 0.02
        out = _command(controller, now, yaw=yaw)
        yaws.append(out.yaw_rate_rad_s)
        yaw += out.yaw_rate_rad_s * 0.02
        headings.append(yaw)
    # First motion is toward the last image-right bearing (positive yaw).
    assert yaws[0] > 0.0
    # The sweep stays alive: the command changes sign at least twice.
    flips = sum(1 for a, b in zip(yaws, yaws[1:]) if a * b < 0.0)
    assert flips >= 2
    # The scan stays centered on the entry heading (bounded by the 0.80
    # excursion cap plus plant lag), never returning to the 0.4 anchor.
    worst = max(
        abs(math.remainder(heading - 1.9, 2.0 * math.pi))
        for heading in headings
    )
    assert worst < 1.20
    assert abs(math.remainder(headings[-1] - 0.4, 2.0 * math.pi)) > 0.3


def test_search_reacquisition_allows_same_track_id():
    controller = _tracked_controller(_track("A", 0.20, 0.0, scale=0.10))
    now = 100.10
    for _ in range(40):
        now += 0.033
        controller.observe(_update([], frame_id=9), now_s=now)
        if controller.state is CleanCourseState.SEARCH:
            break
    assert controller.state is CleanCourseState.SEARCH
    now += 0.033
    controller.observe(_update([_track("A", 0.30, 0.0)], frame_id=50), now_s=now)
    assert controller.state is CleanCourseState.TRACK
    assert controller.current.track_id == "A"


def test_search_never_promotes_retained_successor_without_race_credit():
    # F107 (20260801T053629Z-visual-course-cb3892b6): Gate 0 disappeared
    # below frame, then SEARCH adopted retained successor B and drove toward
    # it even though authoritative race ownership remained Gate 0.  A known
    # successor is next-gate evidence and can become current only through an
    # authoritative note_race() increment.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.observe(
        _update(
            [_track("A", 0.0, 0.0, scale=0.50), _track("B", -0.45, 0.10)],
            frame_id=4,
        ),
        now_s=100.08,
    )
    assert controller.successor is not None
    assert controller.successor.track_id == "B"
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller._enter_search(100.12)

    controller.observe(
        _update([_track("B", -0.48, 0.12)], frame_id=5), now_s=100.15
    )

    assert controller.state is CleanCourseState.SEARCH
    assert controller.gate_index == 0
    assert controller.transitions == []
    assert controller.current is not None
    assert controller.current.track_id == "A"
    assert controller.successor is not None
    assert controller.successor.track_id == "B"


def _turn_reference_controller(
    *,
    successor_x=-0.45,
    current_x=0.02,
    now_s=100.10,
):
    """Fresh Gate-0 geometry with optional persistent farther Gate 1."""

    controller = _tracked_controller(
        _track("A", current_x, 0.0, scale=0.45),
    )
    tracks = [_track("A", current_x, 0.0, scale=0.45)]
    if successor_x is not None:
        tracks.append(_track("B", successor_x, 0.05, scale=0.10))
    controller.observe(_update(tracks, frame_id=3), now_s=now_s - 0.02)
    if successor_x is not None:
        controller._track_first_seen_s["B"] = now_s - 1.0
        assert controller.successor is not None
        assert controller.successor.track_id == "B"
    current = controller.current
    current.x_axis.p = current_x
    current.x_axis.v = 0.0
    current.raw_x = current_x
    current.aperture_half_x = 0.25
    current.aperture_half_y = 0.25
    return controller


def test_safe_successor_preview_coordinates_yaw_and_bank():
    turn = _turn_reference_controller()
    current_only = _turn_reference_controller(successor_x=None)
    with_turn = None
    without_turn = None
    for tick in range(12):
        now = 100.10 + 0.04 * tick
        turn.successor.last_measurement_s = now
        turn.successor.last_x_measurement_s = now
        with_turn = _command(turn, now, pitch=SPAWN_PITCH, yaw=0.0)
        without_turn = _command(current_only, now, pitch=SPAWN_PITCH, yaw=0.0)

    assert with_turn.state is CleanCourseState.TRACK
    assert with_turn.gate_index == 0
    assert with_turn.current_track_id == "A"
    assert with_turn.successor_track_id == "B"
    assert with_turn.successor_blend > 0.0
    assert with_turn.yaw_rate_rad_s < 0.0
    # F165: once the current aperture has a safe horizontal reserve, useful
    # preview enters the physical intercept coherently.  F164's yaw-only
    # preview turned the camera left while the aircraft kept banking right.
    assert with_turn.target_roll_rad < 0.0
    assert with_turn.yaw_rate_rad_s * with_turn.target_roll_rad > 0.0
    assert with_turn.target_roll_rad < without_turn.target_roll_rad
    assert without_turn.yaw_rate_rad_s > 0.0
    assert without_turn.target_roll_rad > 0.0
    # The successor changes only the shared lateral reference.
    assert with_turn.target_pitch_rad == pytest.approx(
        without_turn.target_pitch_rad, abs=1e-12
    )
    assert with_turn.thrust == pytest.approx(
        without_turn.thrust, abs=1e-12
    )
    assert turn.gate_index == 0
    assert turn.current.track_id == "A"


def test_continuous_turn_reference_continues_through_safe_commit():
    # A TRACK->COMMIT transition must not undo the coordinated preturn while
    # fresh aperture reserve remains.
    preview = _turn_reference_controller()
    current_only = _turn_reference_controller(successor_x=None)
    for tick in range(10):
        now = 100.10 + 0.04 * tick
        preview.successor.last_measurement_s = now
        preview.successor.last_x_measurement_s = now
        _command(preview, now, pitch=SPAWN_PITCH, yaw=0.0)
        _command(current_only, now, pitch=SPAWN_PITCH, yaw=0.0)
    for controller in (preview, current_only):
        controller.state = CleanCourseState.COMMIT
        controller._commit_entry_s = now

    with_preview = _command(preview, now + 0.04, pitch=SPAWN_PITCH, yaw=0.0)
    without_preview = _command(
        current_only, now + 0.04, pitch=SPAWN_PITCH, yaw=0.0
    )

    assert with_preview.state is CleanCourseState.COMMIT
    assert with_preview.successor_blend > 0.0
    assert with_preview.yaw_rate_rad_s < without_preview.yaw_rate_rad_s
    assert with_preview.target_roll_rad < without_preview.target_roll_rad
    assert with_preview.yaw_rate_rad_s * with_preview.target_roll_rad > 0.0
    assert with_preview.target_pitch_rad == pytest.approx(
        without_preview.target_pitch_rad, abs=1e-12
    )
    assert with_preview.thrust == pytest.approx(
        without_preview.thrust, abs=1e-12
    )


def test_turn_reference_survives_track_predict_commit_credit_and_promotion():
    controller = _turn_reference_controller()
    outputs = []
    now = 100.10
    for tick in range(10):
        now += 0.04
        controller.successor.last_measurement_s = now
        controller.successor.last_x_measurement_s = now
        outputs.append(_command(controller, now, pitch=SPAWN_PITCH, yaw=0.0))

    controller.state = CleanCourseState.PREDICT
    controller.current.aperture_half_x = None
    controller.current.aperture_half_y = None
    controller._last_engulfing_anchor_s = now
    now += 0.04
    outputs.append(_command(controller, now, pitch=SPAWN_PITCH, yaw=-0.01))

    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = now
    now += 0.04
    outputs.append(_command(controller, now, pitch=SPAWN_PITCH, yaw=-0.02))

    # The bounded credit wait is still race-owned, but uses the same turn
    # reference instead of a yaw-only overlay.
    controller.state = CleanCourseState.SEARCH
    controller._pending_credit_until_s = now + 1.0
    now += 0.04
    outputs.append(_command(controller, now, pitch=SPAWN_PITCH, yaw=-0.03))
    assert controller.gate_index == 0

    # Association can assign a fresh id at the gate plane.  Race credit,
    # not a second persistence gate, transfers the same measured bearing.
    controller._track_first_seen_s["B"] = now
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=2400, now_s=now + 0.01
    )
    assert promoted
    assert controller.state is CleanCourseState.TRACK
    assert controller.current.track_id == "B"
    now += 0.04
    outputs.append(_command(controller, now, pitch=SPAWN_PITCH, yaw=-0.04))

    active = [out for out in outputs if abs(out.yaw_rate_rad_s) > 1e-6]
    assert active
    first_left = next(
        index for index, out in enumerate(outputs) if out.yaw_rate_rad_s < 0.0
    )
    handoff = outputs[first_left:]
    assert all(out.yaw_rate_rad_s < 0.0 for out in handoff)
    # Heading continuity is independent of the optical-intercept roll state.
    # The latter may briefly retain the previous path correction while its
    # own filtered plane-miss reference changes sign.
    assert all(
        abs(out.target_roll_rad) <= controller.config.max_target_roll_rad + 1e-9
        for out in handoff
    )
    assert controller._lateral_intercept_reference_x < 0.0
    assert handoff[-1].target_roll_rad < handoff[-2].target_roll_rad
    assert controller.transitions == [(0, 1)]


def test_turn_reference_eligibility_variation_never_reverses_left_handoff():
    """F119/F124: qualification flicker cannot reverse the shared turn."""

    controller = _turn_reference_controller(current_x=0.04)
    now = 100.10
    # Establish a credible left reference before the crossing variations.
    for _ in range(14):
        now += 0.04
        variance = (0.02 / math.sqrt(2.0)) ** 2
        controller.successor.x_axis.pp = variance
        controller.successor.y_axis.pp = variance
        controller.successor.last_measurement_s = now
        controller.successor.last_x_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH, yaw=0.0)
    assert out.yaw_rate_rad_s < 0.0
    assert out.target_roll_rad < 0.0

    samples = []
    variations = (
        # good evidence
        (0.02, 0.0, 0.25, None),
        # weak covariance
        (0.55, 0.0, 0.25, None),
        # old measurement, still inside the bounded prediction horizon
        (0.25, 0.90, 0.25, None),
        # aperture loss at a fresh same-current engulfing observation
        (0.35, 0.30, None, 0.0),
        # bad fragment geometry consumes aperture margin
        (0.20, 0.10, 0.08, None),
        # fresh evidence returns
        (0.02, 0.0, 0.25, None),
    )
    for index, (std, age, aperture, anchor_age) in enumerate(variations):
        now += 0.04
        controller.state = (
            CleanCourseState.PREDICT if index in (2, 3) else CleanCourseState.TRACK
        )
        variance = (std / math.sqrt(2.0)) ** 2
        controller.successor.x_axis.pp = variance
        controller.successor.y_axis.pp = variance
        controller.successor.last_measurement_s = now - age
        controller.successor.last_x_measurement_s = now - age
        controller.current.aperture_half_x = aperture
        controller.current.aperture_half_y = aperture
        controller.current.raw_x = 0.30 if index == 4 else 0.04
        controller._last_engulfing_anchor_s = (
            None if anchor_age is None else now - anchor_age
        )
        samples.append(_command(controller, now, pitch=SPAWN_PITCH, yaw=0.0))

    assert all(sample.yaw_rate_rad_s <= 1e-9 for sample in samples)
    assert all(sample.target_roll_rad <= 1e-9 for sample in samples)
    assert any(sample.yaw_rate_rad_s < -0.02 for sample in samples)
    yaw_steps = [
        abs(right.yaw_rate_rad_s - left.yaw_rate_rad_s)
        for left, right in zip(samples, samples[1:])
    ]
    assert max(yaw_steps) < MAX_COURSE_YAW_RATE_RAD_S


def test_fresh_reassociated_successor_has_no_second_turn_age_gate():
    # F122: the persistent left Gate-1 track changed id at the Gate-0 plane.
    # Successor selection had already admitted the fresh, measured hypothesis,
    # but a second age factor reset its turn authority to exactly zero.
    controller = _turn_reference_controller(current_x=0.0)
    now = 100.10
    controller.successor.track_id = "B-reassociated"
    controller._track_first_seen_s[controller.successor.track_id] = now
    controller.successor.last_measurement_s = now
    controller.successor.last_x_measurement_s = now

    out = _command(controller, now, pitch=SPAWN_PITCH, yaw=0.0)

    assert out.successor_blend > 0.0
    assert out.yaw_rate_rad_s < 0.0
    assert out.target_roll_rad < 0.0
    assert out.yaw_rate_rad_s * out.target_roll_rad > 0.0


def test_bottom_censored_successor_cannot_bypass_current_passage_safety():
    # BOTTOM clipping makes successor y uncertain, but its uncensored x axis
    # remains a current measurement.  The heading preview must use x-axis
    # uncertainty alone rather than letting the growing y covariance erase a
    # valid lateral handoff.
    controller = _turn_reference_controller(
        successor_x=-0.45,
        current_x=0.02,
        now_s=100.10,
    )
    old_y_stamp = controller.successor.last_y_measurement_s
    initial_y_std = controller.successor.y_axis.std
    now = 100.10
    out = None
    for frame in range(15):
        now += 0.033
        controller.observe(
            _update(
                [
                    _track("A", 0.02, 0.0, scale=0.45),
                    _track(
                        "B",
                        -0.45,
                        0.50,
                        scale=0.10,
                        clipping=FrameEdge.BOTTOM,
                        center_censored=True,
                    ),
                ],
                frame_id=60 + frame,
            ),
            now_s=now,
        )
        out = _command(controller, now + 0.005, pitch=SPAWN_PITCH, yaw=0.0)

    assert controller.successor.track_id == "B"
    assert controller.successor.last_x_measurement_s == pytest.approx(
        now, abs=1e-12
    )
    assert controller.successor.last_y_measurement_s == old_y_stamp
    assert controller.successor.y_axis.std > initial_y_std * 3.0
    assert controller.successor.x_axis.std < initial_y_std
    # The successor x axis remains fresh, but F165 no longer treats loss of
    # current aperture geometry as permission to hand it control.  Current A
    # remains the shared yaw/roll owner until its passage is credible.
    assert out.successor_blend == pytest.approx(0.0, abs=1e-12)
    assert out.yaw_rate_rad_s > 0.0
    assert out.target_roll_rad > 0.0


def test_bottom_censored_successor_keeps_x_authority_after_safe_passage_release():
    # Paired with the no-release case above: BOTTOM invalidates successor y,
    # not its measured x axis.  With a fresh, contained current aperture the
    # same recorded clipping class must produce a coherent left yaw AND bank.
    def current_track():
        return _f163_trace_track(
            outer_center=(0.02, 0.0),
            outer_span=(0.45, 0.45),
            track_id="A",
            aperture_center=(0.02, 0.0),
            aperture_half=(0.20, 0.20),
            aperture_log_scale=0.5 * math.log(0.20 * 0.20),
        )

    def successor_track():
        return _f163_trace_track(
            outer_center=(-0.45, 0.50),
            outer_span=(0.10, 0.10),
            track_id="B",
            clipping=FrameEdge.BOTTOM,
        )

    controller = _tracked_controller(current_track())
    controller.observe(
        _update([current_track(), successor_track()], frame_id=3),
        now_s=100.08,
    )
    assert controller.successor is not None
    old_y_stamp = controller.successor.last_outer_y_measurement_s
    now = 100.10
    out = None
    for frame in range(15):
        now += 0.033
        controller.observe(
            _update(
                [current_track(), successor_track()],
                frame_id=60 + frame,
            ),
            now_s=now,
        )
        out = _command(
            controller, now + 0.005, pitch=SPAWN_PITCH, yaw=0.0
        )

    assert controller.successor.last_outer_x_measurement_s == pytest.approx(
        now, abs=1e-12
    )
    assert controller.successor.last_outer_y_measurement_s == old_y_stamp
    assert out.successor_blend > 0.10
    assert out.yaw_rate_rad_s < 0.0
    assert out.target_roll_rad < 0.0
    assert out.yaw_rate_rad_s * out.target_roll_rad > 0.0


def test_successor_reassociation_cannot_recreate_precredit_s_turn():
    """F125/F126: covariance flicker and a new id keep one left handoff."""

    controller = _turn_reference_controller(current_x=0.04)
    now = 100.10
    controller.state = CleanCourseState.PREDICT
    controller.current.outer_log_scale = -0.95
    controller.current.aperture_half_x = None
    controller.current.aperture_half_y = None
    controller._last_engulfing_anchor_s = now

    # Start at the last credible same-sign F122/F125 handoff state.  The old
    # successor then loses covariance/freshness before the detector assigns a
    # fresh id to the same consistently-left Gate 1.  F125 fed the evidence
    # product directly to the reference: yaw went right, then jumped left by
    # 0.126 rad/s on this reassociation.  Track age must not be reintroduced;
    # only continuous evidence authority is carried across the id change.
    controller._turn_aperture_reserve = 0.75
    controller._turn_successor_authority = 0.07
    controller._turn_reference_x = -0.007
    controller._turn_reference_yaw_rad = 0.0
    samples = []
    variations = (
        (0.40, 0.40, None),
        (0.48, 0.50, None),
        (0.565, 0.59, None),
        (0.141, 0.0, "B-reassociated"),
        (0.014, 0.0, None),
        (0.014, 0.0, None),
    )
    for std, age_s, replacement_id in variations:
        now += 0.047
        variance = (std / math.sqrt(2.0)) ** 2
        controller.successor.x_axis.pp = variance
        controller.successor.y_axis.pp = variance
        controller.successor.last_measurement_s = now - age_s
        controller.successor.last_x_measurement_s = now - age_s
        if replacement_id is not None:
            controller.successor.track_id = replacement_id
            controller._track_first_seen_s[replacement_id] = now
        controller._last_engulfing_anchor_s = now
        samples.append(_command(controller, now, pitch=SPAWN_PITCH, yaw=0.0))

    assert all(sample.yaw_rate_rad_s < 0.0 for sample in samples)
    # Safe preview is one coordinated lateral path, not F164's yaw-only
    # overlay.  It stays bounded and left through evidence/reassociation.
    assert all(
        abs(sample.target_roll_rad)
        <= controller.config.max_target_roll_rad + 1e-9
        for sample in samples
    )
    assert samples[-1].target_roll_rad < 0.0
    assert all(
        sample.yaw_rate_rad_s * sample.target_roll_rad > 0.0
        for sample in samples
    )
    yaw_steps = [
        abs(right.yaw_rate_rad_s - left.yaw_rate_rad_s)
        for left, right in zip(samples, samples[1:])
    ]
    assert max(yaw_steps) < 0.08


def test_f168_f167_credit_before_replacement_reconciles_authorized_gate():
    # Exact ordering from duplicate b47c8d21.  Race credit is delivered before
    # tracker frame 2563268.  The top-level target frame froze at 2563264 in
    # the live trace, but these clean tracker tokens continued advancing and
    # are the only valid freshness identity for this replay.
    rows = (
        (2.266, 2563246, (-0.074467, 0.553317, -0.092444), (("000001", (0.068750, -0.088889), (0.335938, 0.586111), 0.995212), ("000002", (-0.362500, -0.127778), (0.076562, 0.166667), 0.625495))),
        (2.313, 2563248, (-0.067452, 0.487299, -0.082652), (("000001", (0.006250, -0.177778), (0.446875, 0.650000), 0.961898),)),
        (2.359, 2563249, (-0.082505, 0.411948, -0.066429), (("000001", (0.021875, -0.222222), (0.462500, 0.691667), 0.952734),)),
        (2.406, 2563251, (-0.064116, 0.363754, -0.055701), ()),
        (2.453, 2563252, (-0.039754, 0.312770, -0.040357), ()),
        (2.484, 2563253, (-0.028495, 0.287961, -0.035850), ()),
        (2.516, 2563254, (-0.002528, 0.240838, -0.029696), ()),
        (2.563, 2563255, (0.042319, 0.212398, -0.000098), ()),
        (2.609, 2563257, (0.071036, 0.181519, 0.050051), ()),
        (2.656, 2563258, (0.105947, 0.148797, 0.115410), ()),
        (2.703, 2563259, (0.105623, 0.122730, 0.141680), ()),
        (2.750, 2563261, (0.029472, -0.010910, 0.096297), ()),
        (2.797, 2563262, (0.050517, -0.030163, 0.121178), ()),
        (2.844, 2563264, (0.073114, 0.004247, 0.226815), (("000003", (-0.465625, -0.450000), (0.106250, 0.219444), 0.702566),)),
        (2.891, 2563265, (0.055314, -0.023534, 0.308919), (("000003", (-0.478125, -0.455556), (0.109375, 0.222222), 0.709395),)),
        (2.922, 2563266, (0.071544, -0.041841, 0.347190), (("000003", (-0.496875, -0.466667), (0.114063, 0.230556), 0.717633),)),
        (2.953, 2563267, (0.088708, -0.039556, 0.340910), (("000003", (-0.518750, -0.472222), (0.117188, 0.236111), 0.720079),)),
        (2.984, 2563268, (0.067378, -0.046544, 0.326438), (("000003", (-0.540625, -0.477778), (0.121875, 0.241667), 0.738271),)),
        (3.016, 2563269, (0.068769, -0.056515, 0.222989), (("000003", (-0.562500, -0.483333), (0.125000, 0.250000), 0.741436),)),
        (3.141, 2563273, (-0.000012, -0.076464, -0.293137), (("000003", (-0.559375, -0.472222), (0.135937, 0.261111), 0.783725),)),
        # One republished tracker frame stays PREDICT at most; the next real
        # token restores TRACK and never becomes a frozen-frame SEARCH.
        (3.172, 2563273, (0.024671, -0.078661, -0.297917), (("000003", (-0.559375, -0.472222), (0.135937, 0.261111), 0.783725),)),
        (3.203, 2563275, (-0.040266, -0.083746, -0.335957), (("000003", (-0.553125, -0.461111), (0.140625, 0.261111), 0.799198),)),
        (3.422, 2563281, (-0.054991, -0.069531, -0.329468), (("000003", (-0.525000, -0.438889), (0.157813, 0.277778), 0.834724), ("000004", (0.003125, -0.155556), (0.029687, 0.086111), 0.420400))),
        (3.703, 2563290, (-0.063707, -0.056485, -0.329596), (("000003", (-0.503125, -0.438889), (0.192187, 0.313889), 0.825139), ("000004", (0.096875, -0.133333), (0.032813, 0.091667), 0.426510))),
        (3.766, 2563291, (-0.066407, -0.054044, -0.329594), (("000003", (-0.503125, -0.438889), (0.196875, 0.319444), 0.807909), ("000004", (0.109375, -0.127778), (0.032813, 0.094444), 0.429344))),
        (3.797, 2563292, (-0.115047, -0.157195, -0.236294), (("000003", (-0.503125, -0.438889), (0.203125, 0.327778), 0.800529), ("000004", (0.121875, -0.127778), (0.032813, 0.094444), 0.429738))),
    )

    def tracks(specs):
        return [
            _f163_trace_track(
                track_id=f"vq2-track-{track_id}",
                outer_center=center,
                outer_span=span,
                confidence=confidence,
            )
            for track_id, center, span, confidence in specs
        ]

    first = rows[0]
    initial_tracks = tracks(first[3])
    gate_zero = initial_tracks[0]
    controller = CleanCourseController(_config())
    controller.initialize(
        _update(initial_tracks, frame_id=first[1]),
        gate_index=0,
        fallback_center_norm=gate_zero.center_norm,
        fallback_apparent_scale=gate_zero.apparent_scale,
        now_s=100.0 + first[0],
    )
    post_credit = []
    previous_frame_id = first[1]
    promoted_lineage = None
    for elapsed, frame_id, body_rates, specs in rows[1:]:
        now = 100.0 + elapsed
        if elapsed == 2.984:
            promoted_lineage = controller.successor
            assert controller.note_race(
                gate_index=1, race_boot_ms=2400, now_s=now
            )
            assert controller.current is promoted_lineage
            # The uniquely compatible tracker alias may inherit the logical
            # successor before credit, but it remains successor-only until
            # this authoritative promotion.
            assert controller.current.track_id == "vq2-track-000003"
            assert controller.state is CleanCourseState.TRACK
        measurement_before = (
            controller.current.last_measurement_s
            if controller.current is not None
            else None
        )
        controller.observe(
            _update(tracks(specs), frame_id=frame_id),
            now_s=now,
            body_rates=body_rates,
        )
        if elapsed >= 2.984:
            assert controller.current is promoted_lineage
            if frame_id == previous_frame_id:
                assert controller.current.last_measurement_s == pytest.approx(
                    measurement_before
                )
            else:
                assert controller.current.last_measurement_s == pytest.approx(now)
            output = _command(
                controller,
                now,
                pitch=SPAWN_PITCH,
                yaw=0.0,
            )
            post_credit.append((elapsed, frame_id, output))
            assert controller.current.track_id == "vq2-track-000003"
            assert output.current_track_id == "vq2-track-000003"
            assert output.successor_track_id != "vq2-track-000003"
            assert controller.state is not CleanCourseState.SEARCH
            assert output.yaw_rate_rad_s < 0.0
            assert output.target_roll_rad < 0.0
            assert output.yaw_rate_rad_s * output.target_roll_rad > 0.0
            assert controller._last_vertical_path_error < 0.0
        previous_frame_id = frame_id

    assert controller._last_reconcile_status == "exact-current"
    assert all(frame_id >= 2563268 for _t, frame_id, _out in post_credit)
    assert any(
        left[1] != right[1]
        for left, right in zip(post_credit, post_credit[1:])
    )
    assert all(
        output.current_track_id != "vq2-track-000004"
        for _elapsed, _frame_id, output in post_credit
    )


def test_f168_promoted_lineage_holds_ambiguous_and_tiny_replacements():
    # Race authority promotes one logical successor, not whichever detector id
    # wins the next confidence ranking.  Two compatible replacements are
    # ambiguous; a tiny fragment and a tracker-ambiguous candidate are not
    # identity evidence.  Even after prediction authority expires they hold a
    # bounded SEARCH until one accepted replacement becomes unique.
    gate_zero = _track("gate-zero", 0.0, 0.0, scale=0.30, confidence=0.95)
    successor = _track(
        "successor-old", -0.30, -0.20, scale=0.17, confidence=0.80
    )
    controller = CleanCourseController(_config())
    controller.initialize(
        _update([gate_zero, successor], frame_id=10),
        gate_index=0,
        fallback_center_norm=gate_zero.center_norm,
        fallback_apparent_scale=gate_zero.apparent_scale,
        now_s=100.0,
    )
    successor_lineage = controller.successor
    assert controller.note_race(
        gate_index=1, race_boot_ms=1000, now_s=100.10
    )
    assert controller.current is successor_lineage
    assert controller.state is CleanCourseState.TRACK

    candidate_a = _track(
        "candidate-a", -0.31, -0.20, scale=0.18, confidence=0.90
    )
    candidate_b = _track(
        "candidate-b", -0.29, -0.19, scale=0.18, confidence=0.89
    )
    controller.observe(
        _update([candidate_a, candidate_b], frame_id=11), now_s=100.13
    )
    assert controller.state is CleanCourseState.SEARCH
    assert controller.current is successor_lineage
    assert controller.current.track_id == "successor-old"
    assert controller.successor is None
    assert controller._last_reconcile_status == "ambiguous-search-hold"

    tiny = _track("tiny-fragment", -0.30, -0.20, scale=0.05, confidence=0.99)
    tracker_ambiguous = _track(
        "tracker-ambiguous", -0.30, -0.20, scale=0.18, confidence=0.99
    )
    tracker_ambiguous.ambiguous = True
    controller.observe(
        _update([tiny, tracker_ambiguous], frame_id=12), now_s=101.70
    )
    assert controller.state is CleanCourseState.SEARCH
    assert controller.current is successor_lineage
    assert controller.current.track_id == "successor-old"
    assert controller.successor is None
    assert controller._last_reconcile_status == "expired-search-hold"
    assert controller._promoted_reconcile_gate_index == 1

    accepted = _track(
        "accepted-replacement", -0.30, -0.20, scale=0.19, confidence=0.85
    )
    controller.observe(_update([accepted], frame_id=13), now_s=101.73)
    assert controller.state is CleanCourseState.TRACK
    assert controller.current is successor_lineage
    assert controller.current.track_id == "accepted-replacement"
    assert controller._last_reconcile_status == "promoted-current-rebound"


def test_f168_tracker_replacement_does_not_inherit_aperture_fit_sigma():
    # Aperture fitting and tracker-alias identity are independent contracts.
    # A deliberately permissive aperture fit must not authorize a spatially
    # incompatible replacement when the identity innovation gate rejects it.
    config = _config(
        aperture_scale_innovation_sigma=100.0,
        track_replacement_innovation_sigma=0.25,
    )
    gate_zero = _track("gate-zero", 0.0, 0.0, scale=0.30, confidence=0.95)
    successor = _track(
        "successor-old", -0.30, -0.20, scale=0.17, confidence=0.80
    )
    controller = CleanCourseController(config)
    controller.initialize(
        _update([gate_zero, successor], frame_id=20),
        gate_index=0,
        fallback_center_norm=gate_zero.center_norm,
        fallback_apparent_scale=gate_zero.apparent_scale,
        now_s=100.0,
    )
    lineage = controller.successor
    assert controller.note_race(
        gate_index=1, race_boot_ms=1000, now_s=100.10
    )

    incompatible = _track(
        "spatial-outlier", -0.20, -0.20, scale=0.17, confidence=0.99
    )
    controller.observe(_update([incompatible], frame_id=21), now_s=100.13)
    assert controller.state is CleanCourseState.SEARCH
    assert controller.current is lineage
    assert controller.current.track_id == "successor-old"
    assert controller._last_reconcile_status == "incompatible-search-hold"

    accepted = _track(
        "spatial-match", -0.30, -0.20, scale=0.17, confidence=0.85
    )
    controller.observe(_update([accepted], frame_id=22), now_s=100.16)
    assert controller.state is CleanCourseState.TRACK
    assert controller.current is lineage
    assert controller.current.track_id == "spatial-match"


def test_f168_release_rebind_keeps_successor_yaw_roll_and_vertical_coherent():
    # Reach COMMIT through a deliberately safe public trace, retain the
    # recorded left Gate-1 successor ordering, and then lose Gate 0 with a
    # stale RIGHT edge.  Exact zero is still mandatory.  During the bounded
    # credit delay a compatible fresh id must inherit successor lineage and
    # own yaw, bank, and outer-y together; the released old-current edge has
    # no remaining authority.
    controller, _outputs, _now = _public_safe_commit_controller()
    current_edge = _f163_trace_track(
        track_id="recorded-current",
        outer_center=(0.20, 0.10),
        outer_span=(0.70, 0.80),
        confidence=0.80,
        clipping=FrameEdge.RIGHT,
    )
    old_successor = _f163_trace_track(
        track_id="old-successor",
        # F168 t=2.344: last accepted alias before the tracker re-keyed the
        # same physical Gate 1 as 000004.
        outer_center=(-0.38125, 0.04444444444444451),
        outer_span=(0.0765625, 0.16944444444444445),
        confidence=0.623,
    )
    controller._track_first_seen_s["old-successor"] = 101.50
    controller.observe(
        _update([current_edge, old_successor], frame_id=2053015),
        now_s=102.50,
    )
    assert controller.current.horizontal_censor_edge == FrameEdge.RIGHT
    controller.observe(
        _update([old_successor], frame_id=2053016), now_s=102.54
    )
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    exact_zero = _command(
        controller, 102.56, pitch=-0.45, yaw=0.0
    )
    assert (
        exact_zero.target_roll_rad,
        exact_zero.target_pitch_rad,
        exact_zero.yaw_rate_rad_s,
        exact_zero.thrust,
    ) == (0.0, 0.0, 0.0, 0.0)
    pending = _command(controller, 102.58, pitch=-0.45, yaw=0.0)
    assert pending.state is CleanCourseState.SEARCH

    replacement = _f163_trace_track(
        track_id="fresh-successor",
        # Exact F168 t=2.937 replacement geometry.  Its perspective aspect
        # (0.469) is suspicious for cold adoption but is uniquely compatible
        # with the maintained, race-scoped successor lineage.
        outer_center=(-0.34375, -0.005555555555555536),
        outer_span=(0.09375, 0.20),
        confidence=0.690,
    )
    controller.observe(
        _update([replacement], frame_id=2053017), now_s=102.62
    )
    assert controller.successor.track_id == "fresh-successor"
    assert controller._last_reconcile_status == "released-successor-rebound"
    successor_lineage = controller.successor
    precredit = _command(controller, 102.63, pitch=-0.40, yaw=0.0)
    assert precredit.yaw_rate_rad_s < 0.0
    assert precredit.target_roll_rad < 0.0
    assert precredit.yaw_rate_rad_s * precredit.target_roll_rad > 0.0
    assert precredit.target_roll_rad <= pending.target_roll_rad
    assert controller._last_vertical_path_error < 0.0
    assert precredit.successor_track_id == "fresh-successor"

    prepromotion_reference = controller._lateral_intercept_reference_x
    assert controller.note_race(
        gate_index=1, race_boot_ms=4000, now_s=102.64
    )
    assert controller.current is successor_lineage
    assert controller.current.track_id == "fresh-successor"
    assert controller._lateral_intercept_reference_x == pytest.approx(
        prepromotion_reference
    )
    # Even a successor that was fresh enough to promote as TRACK can change
    # tracker aliases on the next frame.  The logical hypothesis and its path
    # state survive that immediate handoff.
    second_alias = _f163_trace_track(
        track_id="fresh-successor-2",
        outer_center=(-0.35, -0.01666666666666672),
        outer_span=(0.0953125, 0.20277777777777778),
        confidence=0.86,
    )
    controller.observe(
        _update([second_alias], frame_id=2053018), now_s=102.67
    )
    assert controller.current is successor_lineage
    assert controller.current.track_id == "fresh-successor-2"
    assert controller._last_reconcile_status == "promoted-current-rebound"
    postcredit = _command(controller, 102.68, pitch=-0.40, yaw=0.0)
    assert postcredit.yaw_rate_rad_s < 0.0
    assert postcredit.target_roll_rad < 0.0
    assert postcredit.yaw_rate_rad_s * postcredit.target_roll_rad > 0.0
    assert controller._last_vertical_path_error < 0.0


def test_weak_successor_evidence_decays_turn_reference_smoothly():
    controller = _turn_reference_controller(current_x=0.04)
    now = 100.10
    for _ in range(14):
        now += 0.04
        variance = (0.02 / math.sqrt(2.0)) ** 2
        controller.successor.x_axis.pp = variance
        controller.successor.y_axis.pp = variance
        controller.successor.last_measurement_s = now
        controller.successor.last_x_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH, yaw=0.0)
    authority_before = out.successor_blend
    assert authority_before > 0.0
    assert out.yaw_rate_rad_s < 0.0

    # All evidence factors weaken together.  Raw evidence weight can vanish,
    # but the only command-bearing state is the derotated reference, which
    # must decay continuously instead of reproducing F119's off-full switch.
    controller.successor.x_axis.pp = 2.0
    controller.successor.y_axis.pp = 2.0
    controller.successor.last_measurement_s = now - 2.0
    controller.successor.last_x_measurement_s = now - 2.0
    controller.current.aperture_half_x = None
    controller.current.aperture_half_y = None
    controller._last_engulfing_anchor_s = None
    authorities = []
    references = []
    yaws = []
    for _ in range(6):
        now += 0.04
        out = _command(controller, now, pitch=SPAWN_PITCH, yaw=0.0)
        authorities.append(out.successor_blend)
        references.append(controller._turn_reference_x)
        yaws.append(out.yaw_rate_rad_s)

    assert 0.0 < authorities[0] < authority_before
    assert all(
        right < left for left, right in zip(authorities, authorities[1:])
    )
    assert all(reference < 0.0 for reference in references)
    assert all(
        right > left for left, right in zip(references, references[1:])
    )
    assert all(yaw < 0.0 for yaw in yaws)


def test_fresh_current_passage_keeps_custody_from_opposite_successor():
    # A fresh current lateral claim cannot be erased by a farther opposite
    # successor before aperture geometry releases passage custody.
    controller = _turn_reference_controller(successor_x=0.40, current_x=-0.12)
    controller.current.aperture_half_x = 0.10
    controller.current.aperture_half_y = 0.10
    controller.current.raw_x = -0.12
    controller._turn_aperture_reserve = 0.0
    controller._turn_successor_authority = 0.0
    controller._turn_reference_x = -0.12
    controller._turn_reference_yaw_rad = 0.0
    now = 100.10
    outputs = []
    authorities = []
    for _ in range(50):
        now += 0.04
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        controller.successor.last_measurement_s = now
        controller.successor.last_x_measurement_s = now
        outputs.append(_command(controller, now, pitch=SPAWN_PITCH, yaw=0.0))
        authorities.append(controller._turn_successor_authority)

    assert max(authorities) < 0.20
    assert all(output.yaw_rate_rad_s < -0.005 for output in outputs)
    assert all(output.target_roll_rad < 0.0 for output in outputs)


def test_bottom_y_covariance_cannot_release_fresh_lateral_custody():
    # Bottom censorship may inflate y covariance, but it cannot change the
    # current gate's still-fresh x claim in the shared lateral reference.
    nominal = _turn_reference_controller(successor_x=0.55, current_x=-0.12)
    censored = _turn_reference_controller(successor_x=0.55, current_x=-0.12)
    for controller in (nominal, censored):
        controller.current.aperture_half_x = 0.10
        controller.current.aperture_half_y = 0.10
        controller.current.raw_x = -0.12
        controller._turn_aperture_reserve = 0.0
        controller._turn_successor_authority = 0.0
        controller._turn_reference_x = -0.12
        controller._turn_reference_yaw_rad = 0.0
    censored.current.y_axis.pp = 4.0

    now = 100.10
    for _ in range(12):
        now += 0.04
        lateral = []
        for controller in (nominal, censored):
            controller.current.last_x_measurement_s = now
            controller.successor.last_measurement_s = now
            controller.successor.last_x_measurement_s = now
            reference, authority = controller._turn_reference(
                controller.current,
                controller.successor,
                current_error=-0.12,
                now_s=now,
                yaw_rad=0.0,
                dt=0.04,
            )
            roll, yaw_rate = controller._coordinated_turn_request(
                reference, steer_gain=1.0, yaw_rad=0.0
            )
            lateral.append((reference, authority, roll, yaw_rate))

        assert lateral[1] == pytest.approx(lateral[0], abs=1e-12)


def test_current_claim_age_cannot_create_successor_passage_authority():
    # F164 released successor custody merely because the current aperture/x
    # aged.  Missing current passage evidence must produce no authority at
    # all, independent of how long the stale claim has aged.
    controller = _turn_reference_controller(successor_x=-0.45, current_x=0.0)
    controller.current.aperture_half_x = None
    controller.current.aperture_half_y = None
    controller._last_engulfing_anchor_s = None
    controller._turn_aperture_reserve = 0.0
    controller._turn_successor_authority = 0.0
    controller._turn_reference_x = 0.0
    controller._turn_reference_yaw_rad = 0.0
    weak_std = controller.config.successor_turn_max_std_norm * 0.98
    # Successor heading eligibility is intentionally axis-specific: set the
    # requested weak x standard deviation directly.  Y uncertainty cannot
    # manufacture or erase horizontal authority.
    weak_axis_variance = weak_std**2
    controller.successor.x_axis.pp = weak_axis_variance
    controller.successor.y_axis.pp = weak_axis_variance

    now = 100.10
    authorities = []
    for _ in range(12):
        now += 0.04
        controller.current.last_x_measurement_s = now - 2.0
        controller.successor.last_measurement_s = now
        controller.successor.last_x_measurement_s = now
        _, authority = controller._turn_reference(
            controller.current,
            controller.successor,
            current_error=0.0,
            now_s=now,
            yaw_rad=0.0,
            dt=0.04,
        )
        authorities.append(authority)

    assert authorities == pytest.approx([0.0] * len(authorities), abs=1e-12)


def test_f143_crossing_reassociation_keeps_one_left_turn_time_series():
    # F143's Gate-1 bearing stayed left while stale Gate-0 x drifted right.
    # Weakening successor evidence and a fresh same-bearing id drove the old
    # aperture-product law left -> right -> left, then jumped 0.064 rad/s at
    # credit.  A released current claim must preserve one coordinated sign.
    controller = _turn_reference_controller(successor_x=-0.45, current_x=0.06)
    controller.current.outer_log_scale = -0.90
    controller.current.aperture_half_x = None
    controller.current.aperture_half_y = None
    controller._last_engulfing_anchor_s = None
    controller._turn_aperture_reserve = 0.08
    controller._turn_successor_authority = 0.20
    controller._turn_reference_x = -0.03
    controller._turn_reference_yaw_rad = 0.0

    now = 100.10
    commands = []
    variations = (
        (0.05, 0.00, None),
        (0.08, 0.10, None),
        (0.20, 0.25, None),
        (0.35, 0.40, None),
        (0.48, 0.55, None),
        (0.55, 0.65, None),
        (0.08, 0.00, "B-reassociated"),
        (0.05, 0.00, None),
        (0.05, 0.00, None),
        (0.05, 0.00, None),
        (0.05, 0.00, None),
        (0.05, 0.00, None),
    )
    for std, age_s, replacement_id in variations:
        now += 0.047
        controller.current.last_x_measurement_s = now - 2.0
        variance = (std / math.sqrt(2.0)) ** 2
        controller.successor.x_axis.pp = variance
        controller.successor.y_axis.pp = variance
        controller.successor.last_measurement_s = now - age_s
        controller.successor.last_x_measurement_s = now - age_s
        if replacement_id is not None:
            controller.successor.track_id = replacement_id
            controller._track_first_seen_s[replacement_id] = now
        reference, _ = controller._turn_reference(
            controller.current,
            controller.successor,
            current_error=0.06,
            now_s=now,
            yaw_rad=0.0,
            dt=0.047,
        )
        commands.append(
            controller._coordinated_turn_request(
                reference, steer_gain=1.0, yaw_rad=0.0
            )
        )

    rolls, yaws = zip(*commands)
    assert all(yaw < 0.0 for yaw in yaws)
    assert all(roll < 0.0 for roll in rolls)
    assert max(abs(b - a) for a, b in zip(yaws, yaws[1:])) < 0.06


def test_f144_engulfing_evidence_supersedes_stale_aperture_extent():
    # F144 received fresh same-id engulfing observations from -0.55 s, but a
    # retained non-None aperture extent won the old if/elif precedence.  Its
    # invalid margin target stayed zero, reserve collapsed .012 -> .001, and
    # the consistently-left successor was countermanded by current x=+0.066.
    # Both existing passage signals must feed one continuous reserve target.
    controller = _turn_reference_controller(successor_x=-0.45, current_x=0.10)
    controller.current.outer_log_scale = -0.90
    controller.current.aperture_half_x = 0.05
    controller.current.aperture_half_y = 0.05
    controller.current.raw_x = 0.30  # retained extent has no usable margin
    controller._turn_aperture_reserve = 0.01
    controller._turn_successor_authority = 0.12
    controller._turn_reference_x = -0.02
    controller._turn_reference_yaw_rad = 0.0

    now = 100.10
    reserves = []
    commands = []
    for _ in range(16):
        now += 0.047
        controller._last_engulfing_anchor_s = now
        controller.current.last_x_measurement_s = now
        controller.successor.last_measurement_s = now
        controller.successor.last_x_measurement_s = now
        reference, _ = controller._turn_reference(
            controller.current,
            controller.successor,
            current_error=0.10,
            now_s=now,
            yaw_rad=0.0,
            dt=0.047,
        )
        reserves.append(controller._turn_aperture_reserve)
        commands.append(
            controller._coordinated_turn_request(
                reference, steer_gain=1.0, yaw_rad=0.0
            )
        )

    rolls, yaws = zip(*commands)
    assert all(right > left for left, right in zip(reserves, reserves[1:]))
    assert reserves[-1] > 0.95
    assert all(yaw < 0.0 for yaw in yaws)
    assert all(roll < 0.0 for roll in rolls)


def test_f145_released_current_claim_cannot_reacquire_implicit_weight():
    # F145 fixed the passage reserve, but the final convex blend discarded its
    # evidence-backed current_claim and implicitly restored current error to
    # (1 - successor_authority).  While the old left successor grew uncertain,
    # a released Gate-0 claim therefore kept issuing right yaw until re-id.
    controller = _turn_reference_controller(successor_x=-0.45, current_x=0.10)
    controller.current.outer_log_scale = -0.90
    controller.current.aperture_half_x = 0.05
    controller.current.aperture_half_y = 0.05
    controller.current.raw_x = 0.30
    controller._turn_aperture_reserve = 0.01
    controller._turn_successor_authority = 0.07
    controller._turn_reference_x = 0.02
    controller._turn_reference_yaw_rad = 0.0

    now = 100.10
    yaws = []
    rolls = []
    for tick in range(10):
        now += 0.047
        controller._last_engulfing_anchor_s = now
        controller.current.last_x_measurement_s = now
        std = 0.35 + 0.02 * tick
        variance = (std / math.sqrt(2.0)) ** 2
        controller.successor.x_axis.pp = variance
        controller.successor.y_axis.pp = variance
        age_s = 0.35 + 0.02 * tick
        controller.successor.last_measurement_s = now - age_s
        controller.successor.last_x_measurement_s = now - age_s
        reference, _ = controller._turn_reference(
            controller.current,
            controller.successor,
            current_error=0.10,
            now_s=now,
            yaw_rad=0.0,
            dt=0.047,
        )
        roll, yaw = controller._coordinated_turn_request(
            reference, steer_gain=1.0, yaw_rad=0.0
        )
        yaws.append(yaw)
        rolls.append(roll)

    first_left = next(index for index, yaw in enumerate(yaws) if yaw < -0.005)
    assert first_left <= 6
    assert all(yaw < -0.005 for yaw in yaws[first_left:])
    assert all(roll < 0.0 for roll in rolls[first_left:])
    assert max(abs(b - a) for a, b in zip(yaws, yaws[1:])) < 0.06


def test_weak_opposite_successor_cannot_erase_current_gate_turn():
    # F120 live regression: Gate 1 was still x=-0.309 when a right-side Gate
    # 2 candidate reached only 2.1e-7 authority.  Binary sign arbitration
    # nevertheless snapped the left request to zero for ~1.4 s.
    controller = _turn_reference_controller(successor_x=0.20, current_x=-0.30)
    controller.current.aperture_half_x = 0.50
    controller.current.aperture_half_y = 0.50
    controller.current.raw_x = -0.30
    controller.successor.x_axis.pp = 2.0
    controller.successor.y_axis.pp = 2.0
    controller._turn_aperture_reserve = 0.95
    # The production path always carries the one derotated reference into a
    # successor update; weak opposite evidence must decay that reference
    # continuously rather than erasing or reversing it.
    controller._turn_reference_x = -0.30
    controller._turn_reference_yaw_rad = 0.0
    now = 100.10
    heading, authority = controller._turn_reference(
        controller.current,
        controller.successor,
        current_error=-0.30,
        now_s=now,
        yaw_rad=0.0,
        dt=0.04,
    )
    target_roll, yaw_rate = controller._coordinated_turn_request(
        heading, steer_gain=1.0, yaw_rad=0.0
    )

    assert authority == pytest.approx(0.0)
    assert heading < -0.20
    assert yaw_rate < -0.02
    assert target_roll < 0.0


def test_authoritative_promotion_retains_stale_measured_successor_for_prediction():
    # A measured bearing remains an IMU-derotated hypothesis after race
    # credit.  Staleness selects PREDICT, rather than discarding the bearing
    # into an unrelated SEARCH command; a fresh association resumes TRACK.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.observe(
        _update(
            [_track("A", 0.0, 0.0, scale=0.50), _track("B", -0.45, 0.10)],
            frame_id=4,
        ),
        now_s=100.08,
    )
    assert controller.successor is not None
    assert controller.successor.track_id == "B"
    controller.successor.last_measurement_s = 99.0
    controller.successor.last_x_measurement_s = 99.0

    promoted = controller.note_race(
        gate_index=1, race_boot_ms=2200, now_s=100.10
    )
    assert promoted
    assert controller.state is CleanCourseState.PREDICT
    assert controller.current is not None
    assert controller.current.track_id == "B"
    assert controller.successor is None

    controller.observe(
        _update([_track("B", -0.43, 0.08)], frame_id=5), now_s=100.15
    )

    assert controller.state is CleanCourseState.TRACK
    assert controller.gate_index == 1
    assert controller.current is not None
    assert controller.current.track_id == "B"
    assert controller.successor is None


def test_newborn_suspicious_truss_not_adopted_in_search_reacquisition():
    # F49 (terminal F48 failure): the gate-1 re-acquisition adopted a
    # NEWBORN top-censored extreme-aspect ceiling truss (span 0.50 x 0.23)
    # over the persistent real gate.  A suspicious-geometry track is
    # ineligible until it persists past the re-acquisition age window.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller._enter_search(100.10)
    controller.last_reliable_bearing = (0.10, 0.0)
    gate = _track("G", 0.15, 0.05, scale=0.10, confidence=0.60)
    truss = _truss("T", 0.10, 0.0)  # right at the last bearing
    now = 100.20
    controller._track_first_seen_s["G"] = now - 2.0  # persistent real gate
    controller._track_first_seen_s["T"] = now - 0.05  # newborn truss
    pick = controller._select_search_reacquisition([gate, truss], now)
    assert pick.track_id == "G"
    # Aged past the window the same truss is eligible again: geometry never
    # permanently bans a track, only the newborn adoption.
    controller._track_first_seen_s["T"] = now - 1.0
    pick = controller._select_search_reacquisition([gate, truss], now)
    assert pick.track_id == "T"


def test_newborn_suspicious_truss_not_adopted_as_successor():
    # Same gate on the successor seam: a higher-confidence newborn truss
    # must not outrank the plain gate candidate.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    gate = _track("G", 0.30, 0.0, scale=0.10, confidence=0.50)
    truss = _truss("T", 0.10, -0.60)  # confidence 0.90
    now = 100.20
    controller._track_first_seen_s["G"] = now - 0.10  # newborn too: no aged
    controller._track_first_seen_s["T"] = now - 0.05
    controller._refresh_successor([gate, truss], now)
    assert controller.successor.track_id == "G"
    # Aged past the window the truss is eligible and wins on persistence.
    controller._track_first_seen_s["T"] = now - 1.0
    controller._refresh_successor([gate, truss], now)
    assert controller.successor.track_id == "T"


def test_off_center_close_loss_goes_to_predict_not_coast():
    # F45 (20260729T210351Z-visual-course-b1f5e89f): the close-range track
    # was lost point-blank with the last measured bearing (-0.39,-0.28) and
    # the ballistic coast preserved the offset — the gate slid out of frame
    # uncredited and the leg fell into blind-search churn.  Credit is
    # authoritative; an off-center close loss must NOT arm the coast.  The
    # normal loss path (PREDICT) carries the pursuit for re-acquisition.
    controller = _tracked_controller(_track("A", -0.39, -0.28, scale=0.50))
    controller.observe(_update([], frame_id=5), now_s=100.12)
    assert controller.state is not CleanCourseState.COAST_FOR_CREDIT
    now = 100.12
    for frame in range(6, 16):
        now += 0.033
        controller.observe(_update([], frame_id=frame), now_s=now)
        assert controller.state is not CleanCourseState.COAST_FOR_CREDIT
        if controller.state is CleanCourseState.PREDICT:
            break
    assert controller.state is CleanCourseState.PREDICT


def test_crossing_loss_latches_coast_and_waits_for_newer_race_packet():
    config = _config()
    controller = _tracked_controller(
        _track("A", 0.0, 0.0, scale=0.50), config=config  # close crossing
    )
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    # F102: the gate-0 hot-coast trigger is deleted — COAST arms only from
    # an armed COMMIT; enter through the COMMIT latch.
    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = 100.12
    controller.observe(_update([], frame_id=5), now_s=100.12)
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    output = _command(controller, 100.14)
    assert (
        output.target_roll_rad,
        output.target_pitch_rad,
        output.yaw_rate_rad_s,
    ) == (0.0, 0.0, 0.0)
    assert output.thrust == 0.0
    # A strictly newer race packet without credit ends the wait; vision never
    # declares the pass.
    controller.note_race(gate_index=0, race_boot_ms=2250, now_s=100.20)
    assert controller.state is CleanCourseState.SEARCH
    output = _command(controller, 100.22)
    assert output.thrust > 0.0


def test_crossing_wait_is_bounded_and_authoritative_credit_is_accepted():
    # F102: the gate-0 hot-coast trigger is deleted — COAST arms only from
    # an armed COMMIT; enter through the COMMIT latch.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = 100.12
    controller.observe(_update([], frame_id=5), now_s=100.12)
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    # Authoritative increment during the wait is accepted immediately.
    controller.note_race(gate_index=1, race_boot_ms=2250, now_s=100.20)
    assert controller.gate_index == 1
    assert controller.state is not CleanCourseState.COAST_FOR_CREDIT
    assert controller.transitions == [(0, 1)]

    # The wait is bounded at exactly ONE wire-zero send even with no newer
    # race packet, enforced by the send count rather than a timeout (F68:
    # the ~4 Hz cruise race stream made a 0.25 s zero window a lethal
    # ballistic drop; F69 grazed the bottom bar after 0.10-0.14 s; F71
    # tripped the roll/body-rate limits after 0.06 s).  The second command
    # exits to SEARCH no matter when the scheduler calls it.
    controller2 = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller2.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller2.state = CleanCourseState.COMMIT
    controller2._commit_entry_s = 100.12
    controller2.observe(_update([], frame_id=5), now_s=100.12)
    output = _command(controller2, 100.13)
    assert controller2.state is CleanCourseState.COAST_FOR_CREDIT
    assert output.thrust == 0.0
    output = _command(controller2, 100.13 + 11.0)
    assert controller2.state is CleanCourseState.SEARCH
    assert output.thrust > 0.0


def test_pending_credit_hold_never_sweeps_before_delayed_credit():
    # F76 (20260730T074122Z-visual-course-3a505ef5): after the one-zero
    # send the pending-credit SEARCH ran the generic yaw sweep — +0.15
    # (right) while the retained gate-1 successor sat LEFT (ex -0.43);
    # every detection died 0.4 s later and the leg pinned blind at the
    # yaw cap into the gate-1 structure (collision id 1002).  While
    # authoritative credit is in flight the heading is HELD: a visible
    # left-side successor can never produce positive yaw before delayed
    # credit; the authoritative increment re-acquires the retained
    # (by-construction stale) successor immediately and steers normally.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.observe(
        _update(
            [_track("A", 0.0, 0.0, scale=0.50), _track("B", -0.43, 0.05)],
            frame_id=4,
        ),
        now_s=100.08,
    )
    controller._track_first_seen_s["B"] = 100.08 - 1.0  # persistent (F42)
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller.current.aperture_half_x = 0.25
    controller.current.aperture_half_y = 0.25
    now = 100.10
    # Establish the shared turn reference on approach.  The credit wait
    # continues it; the wait no longer invents a separate yaw posture.
    for _ in range(8):
        now += 0.03
        controller.successor.last_measurement_s = now
        controller.successor.last_x_measurement_s = now
        _command(controller, now, pitch=SPAWN_PITCH, yaw=0.0)
    # F102: the gate-0 hot-coast trigger is deleted — COAST arms only from
    # an armed COMMIT; enter through the COMMIT latch.
    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = now
    now += 0.02
    controller.observe(_update([], frame_id=5), now_s=now)
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    now += 0.02
    out = _command(controller, now, yaw=0.0)  # the single wire-zero send (F72)
    assert out.thrust == 0.0
    # Pending-credit window (~0.33 s << PENDING_CREDIT_HOLD_S): level pitch,
    # governed altitude support, and one bounded successor preturn — no sweep.
    # F78/F167:
    # the still-credible left successor may steer a BOUNDED leftward
    # recentering (see test_pending_credit_recenters_toward_credible_
    # successor), so the invariant here is only the F76 one: never yaw
    # AWAY (positive) from the retained left-side successor.
    for tick in range(10):
        t = now + 0.04 + 0.033 * tick
        out = _command(controller, t, yaw=0.0)
        assert controller.state is CleanCourseState.SEARCH
        assert out.yaw_rate_rad_s <= 0.0
        assert out.target_roll_rad <= 0.0
        assert out.yaw_rate_rad_s * out.target_roll_rad >= 0.0
        assert out.thrust > 0.0
    # Delayed credit inside the window: the retained left-side successor is
    # stale by construction, so it remains the current PREDICT hypothesis —
    # never a blind sweep on the new leg.
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=2400, now_s=t + 0.04
    )
    assert promoted
    assert controller.state is CleanCourseState.PREDICT
    assert controller.current.track_id == "B"
    # A fresh post-credit frame of the adopted gate steers LEFT at once.
    controller.observe(
        _update([_track("B", -0.43, 0.05)], frame_id=6), now_s=t + 0.06
    )
    out = _command(controller, t + 0.09, yaw=0.0)
    assert out.yaw_rate_rad_s < 0.0  # steers LEFT toward the successor
    # Bounded: with no credit the window expires and the sweep resumes.
    expiring = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    expiring.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    expiring.state = CleanCourseState.COMMIT
    expiring._commit_entry_s = 100.12
    expiring.observe(_update([], frame_id=5), now_s=100.12)
    _command(expiring, 100.14)
    out = _command(expiring, 100.18)
    assert out.yaw_rate_rad_s == 0.0
    out = _command(expiring, 100.18 + PENDING_CREDIT_HOLD_S + 1.0)
    assert out.yaw_rate_rad_s != 0.0


def test_pending_credit_recenters_toward_credible_successor():
    # F78 (20260730T082159Z-visual-course-7e18243d): the F76 neutral hold
    # delayed the turn — gate 1 sat visible at x ~-0.51 under gate-0
    # authority for the whole pending window, so the new leg inherited a
    # saturated constant-bearing pursuit it never centered (crossed
    # displaced, no credit).  While credit is in flight, a FRESH,
    # persistent, qualified successor's bearing steers a bounded
    # recentering: yaw and bank left toward a left successor, at/below their
    # command bounds, with no forward advance — and crucially
    # WITHOUT promoting or changing authoritative gate ownership.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.observe(
        _update(
            [_track("A", 0.0, 0.0, scale=0.50), _track("B", -0.51, 0.05)],
            frame_id=4,
        ),
        now_s=100.08,
    )
    controller._track_first_seen_s["B"] = 100.08 - 1.0  # persistent (F42)
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller.current.aperture_half_x = 0.25
    controller.current.aperture_half_y = 0.25
    now = 100.10
    for _ in range(8):
        now += 0.03
        controller.successor.last_measurement_s = now
        controller.successor.last_x_measurement_s = now
        _command(controller, now, pitch=SPAWN_PITCH, yaw=0.0)
    # F102: the gate-0 hot-coast trigger is deleted — COAST arms only from
    # an armed COMMIT; enter through the COMMIT latch.
    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = now
    now += 0.02
    controller.observe(_update([], frame_id=5), now_s=now)
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    now += 0.02
    out = _command(controller, now, yaw=0.0)  # the single wire-zero send (F72)
    assert out.thrust == 0.0
    level_pitch = controller.config.spawn_pitch_rad + controller.config.brake_pitch_rad
    for tick in range(6):
        t = now + 0.04 + 0.05 * tick
        # Fresh successor frames keep B credible through the window.
        controller.observe(
            _update([_track("B", -0.51, 0.05)], frame_id=6 + tick), now_s=t
        )
        out = _command(controller, t + 0.02, yaw=0.0)
        assert controller.state is CleanCourseState.SEARCH
        assert controller.gate_index == 0  # authoritative ownership intact
        assert -MAX_COURSE_YAW_RATE_RAD_S <= out.yaw_rate_rad_s < 0.0
        assert -controller.config.max_target_roll_rad <= out.target_roll_rad < 0.0
        assert out.yaw_rate_rad_s * out.target_roll_rad > 0.0
        # No forward advance before credit: the crossing brake may still be
        # slewing back toward level, but must never cross into nose-down drive.
        assert out.target_pitch_rad <= level_pitch + 1e-6
        assert out.thrust > 0.0


def test_pending_credit_holds_neutral_without_successor():
    # F78: absent successor evidence the pending-credit window retains the
    # F76 neutral heading hold (zero yaw/roll, governed support).  F102:
    # COAST arms only from an armed COMMIT; enter through the COMMIT latch.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = 100.12
    controller.observe(_update([], frame_id=5), now_s=100.12)
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    out = _command(controller, 100.14)  # the single wire-zero send (F72)
    assert out.thrust == 0.0
    for tick in range(6):
        out = _command(controller, 100.18 + 0.033 * tick)
        assert controller.state is CleanCourseState.SEARCH
        assert out.yaw_rate_rad_s == 0.0
        assert out.target_roll_rad == 0.0
        assert out.thrust > 0.0


def test_pending_credit_match_reappearance_never_resumes_track():
    # F78b: the pending-credit no-advance law is an authority overlay,
    # not a SEARCH-command posture — the pre-credit gate's own track id
    # reappearing in a fresh frame must NOT flip SEARCH back to TRACK
    # through the same-track match path and start advancing before the
    # authoritative credit that owns the leg.  F102: COAST arms only from
    # an armed COMMIT; enter through the COMMIT latch.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller.state = CleanCourseState.COMMIT
    controller._commit_entry_s = 100.12
    controller.observe(_update([], frame_id=5), now_s=100.12)
    out = _command(controller, 100.14)  # the single wire-zero send (F72)
    assert out.thrust == 0.0
    out = _command(controller, 100.17)  # coast exit -> pending SEARCH
    assert controller.state is CleanCourseState.SEARCH
    # Gate 0's track id reappears inside the pending window.
    controller.observe(
        _update([_track("A", 0.0, 0.0, scale=0.50)], frame_id=6), now_s=100.20
    )
    assert controller.state is CleanCourseState.SEARCH
    assert controller.gate_index == 0
    # After the window expires without credit the normal match resumes.
    controller.observe(
        _update([_track("A", 0.0, 0.0, scale=0.50)], frame_id=7),
        now_s=100.14 + PENDING_CREDIT_HOLD_S + 0.5,
    )
    assert controller.state is CleanCourseState.TRACK


def test_authoritative_promotion_event_never_vetoed_by_vision():
    controller = _tracked_controller(_track("A", 0.10, 0.0))
    controller.observe(
        _update([_track("A", 0.10, 0.0), _track("B", 0.30, 0.05)], frame_id=3),
        now_s=100.08,
    )
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=2500, now_s=100.10
    )
    assert promoted
    assert controller.gate_index == 1
    assert controller.max_gate_index == 1
    # The selected measured successor keeps its bearing through promotion.
    assert controller.state is CleanCourseState.TRACK
    assert controller.current.track_id == "B"

    # Without any successor bearing the controller enters SEARCH.
    promoted = controller.note_race(
        gate_index=2, race_boot_ms=2750, now_s=100.12
    )
    assert promoted
    assert controller.state is CleanCourseState.SEARCH


def test_track_never_steers_on_unmeasured_x_axis():
    # F40 (20260729T193134Z-visual-course-63ed6342): the adopted aim point
    # was an edge-clipped splinter whose x-axis had never been measured, and
    # command() steered yaw at full authority on it — 0.5 s of max yaw the
    # wrong way threw the real gate off-frame.  An unmeasured/stale x must
    # command zero yaw and wings level (slewing toward 0, not snapping).
    controller = _tracked_controller(_track("A", 0.40, 0.0, scale=0.10))
    controller.current.last_x_measurement_s = NEVER_MEASURED_S
    controller._prev_target_roll = 0.20  # pre-wound bank to unwind
    out = _command(controller, 100.10)
    assert out.state is CleanCourseState.TRACK
    assert out.yaw_rate_rad_s == 0.0
    assert 0.0 < out.target_roll_rad < 0.20  # slewing toward level
    # A fresh x measurement restores normal steering.
    controller.current.last_x_measurement_s = 100.10
    out = _command(controller, 100.12)
    assert out.yaw_rate_rad_s > 0.0


def test_near_plane_stale_x_keeps_derotated_steering():
    # F52 (20260729T232037Z-visual-course-dedf1915): at gate 1's plane the
    # aim track went frame-censored (t=5.78, correctly — censored axes do
    # not update the filter), the derotated filter held the last good
    # bearing (ex=-0.17), and when x_qualified expired (t=6.25) the F40
    # zeroing parked yaw/roll — the drone flew ballistic from a still
    # 0.3-off heading and crossed the plane displaced, no credit.  Near the
    # plane a stale x keeps steering on the derotated hypothesis (the
    # crossing completes in <1 s); the same staleness at FAR range still
    # zeroes steering (F40).
    near = _tracked_controller(_track("A", -0.17, 0.0, scale=0.50))
    near.current.last_x_measurement_s = 100.10 - 1.0  # past X_STEER_MAX_AGE_S
    out = _command(near, 100.10)
    assert out.state is CleanCourseState.TRACK
    # The held ex=-0.17 bearing keeps centering — at this near-plane scale
    # with the F57 boost (0.90 * 2.5 * -0.17 = -0.3825), clamped to the
    # 0.15 production yaw command cap.
    assert out.yaw_rate_rad_s == pytest.approx(-0.15, abs=1e-9)
    assert out.target_roll_rad < 0.0  # slewing toward the boosted bank
    # FAR target, identical staleness: the F40 zeroing still applies.
    far = _tracked_controller(_track("A", -0.17, 0.0, scale=0.10))
    far.current.last_x_measurement_s = 100.10 - 1.0
    out = _command(far, 100.10)
    assert out.yaw_rate_rad_s == 0.0
    assert out.target_roll_rad == 0.0


def test_derotation_focal_and_sign_match_fixed_world_geometry():
    # F140's split x focal was falsified live: it delayed the useful
    # successor turn, doubled pre-credit reversals, and regressed the Gate-1
    # credit setup.  F141 deliberately restores the F127 derotation baseline.
    assert ROTATION_COMP_FOCAL_NORM == pytest.approx(1.6)
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    hypothesis = controller.current
    start_x, start_y = hypothesis.x, hypothesis.y
    controller._predict(hypothesis, 0.033, (0.0, -0.20, 0.40))
    # Both fixed-world flows oppose their corresponding positive body rate.
    assert hypothesis.x - start_x == pytest.approx(-0.40 * 1.6 * 0.033)
    assert hypothesis.y - start_y == pytest.approx(+0.20 * 1.6 * 0.033)

    turn = _turn_reference_controller(successor_x=None, current_x=0.0)
    turn._turn_reference_x = -0.20
    turn._turn_reference_yaw_rad = 0.0
    reference, _ = turn._turn_reference(
        turn.current,
        None,
        current_error=0.0,
        now_s=100.10,
        yaw_rad=-0.10,
        dt=0.0,
    )
    assert reference == pytest.approx(-0.20 + 0.10 * 1.6)


def test_vertical_derotation_keeps_stationary_world_geometry_stationary():
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    hypothesis = controller.current
    pitch = SPAWN_PITCH

    # A stationary gate's raw image y changes only with attitude.  Applying
    # the known pitch flow must leave compensated position and translational
    # visual rate at zero throughout a reversing pitch sequence.
    for pitch_rate in (-0.20, -0.10, 0.15, 0.05):
        dt = 0.033
        controller._predict(hypothesis, dt, (0.0, pitch_rate, 0.0))
        pitch += pitch_rate * dt
        expected_raw_y = (SPAWN_PITCH - pitch) * 1.6
        assert hypothesis.y == pytest.approx(expected_raw_y, abs=1e-12)
        assert controller._compensated_ey(hypothesis.y, pitch) == pytest.approx(
            0.0, abs=1e-12
        )
        assert hypothesis.y_axis.v == pytest.approx(0.0, abs=1e-12)


def test_near_plane_steering_boost_scales_both_lateral_gains():
    # F57: inside the COMMIT proximity regime the TRACK law multiplies both
    # lateral error gains by 2.5 to break the parallax limit cycle (F56's
    # ex stalled at -0.15..-0.18 for the whole approach); the caps are
    # unchanged and far range keeps the proved 0.9/0.5 gains.
    near = _tracked_controller(_track("A", -0.16, 0.0, scale=0.50))
    out = _command(near, 100.10)
    # 0.90 * 2.5 * -0.16 = -0.36, clamped to the 0.15 production yaw
    # command cap (the cap is a COMMAND authority limit, not the measured
    # >=0.42 rad/s airframe response).
    assert out.yaw_rate_rad_s == pytest.approx(-0.15, abs=1e-9)
    far = _tracked_controller(_track("A", -0.16, 0.0, scale=0.10))
    out = _command(far, 100.10)
    assert out.yaw_rate_rad_s == pytest.approx(-0.144, abs=1e-9)  # 0.90*-0.16


def test_commit_fresh_steering_uses_the_outer_range_schedule():
    boosted = _commit_controller()
    boosted.current.scale_axis.p = math.log(0.35)
    out, _ = _drive_commit_window(boosted, 100.10)
    assert boosted.state is CleanCourseState.COMMIT
    # Physical outer range is near-plane, so 0.9*2.5*0.10 reaches the command
    # cap regardless of the slower filtered inner scale estimate.
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)
    plain = _commit_controller()  # filtered scale_axis log remains ~-3.0
    out, _ = _drive_commit_window(plain, 100.10)
    assert plain.state is CleanCourseState.COMMIT
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)


def _settle_commit_passage_covariance(current):
    """Represent a measured, settled approach for downstream COMMIT tests."""

    # The generic hypothesis starts with deliberately broad rate uncertainty.
    # These direct fixtures are one sustain window after a real observation
    # sequence would have tightened it; make that premise explicit so the new
    # uncertainty-bearing passage envelope, rather than initialization noise,
    # controls admission.  Public-boundary reachability is covered separately
    # by the faithful tracker replay in test_aigp_vq2_runner.py.
    for axis in (current.x_axis, current.y_axis):
        axis.pp = 0.01**2
        axis.pv = 0.0
        axis.vv = 0.02**2


def _commit_controller(now_s=100.10):
    """Gate-1 TRACK controller one sustain window short of COMMIT entry:
    near plane (outer log scale -0.50 >= -1.2), fresh uncensored
    measurements on both axes, and the 2026-07-30 entry budget satisfied —
    a usable inner aperture whose 60% margin admits error+blackout drift
    (0.10 + 0 <= 0.6*0.25 on x; 0.05 + 0 <= 0.6*0.25 on y)."""

    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller, now_s=now_s)
    controller._alt_est_m = 2.0  # honest altitude (floor quiet)
    current = controller.current
    current.x_axis.p = 0.10
    current.y_axis.p = 0.05
    current.raw_x = 0.10
    current.raw_y = 0.05
    current.aperture_half_x = 0.25
    current.aperture_half_y = 0.25
    current.outer_log_scale = -0.50
    current.outer_half_span_x = 0.25
    _settle_commit_passage_covariance(current)
    return controller


def _drive_commit_window(controller, now, ticks=12):
    """Tick command() with fresh uncensored stamps (what an uncensored
    same-id track produces) at the level spawn attitude."""

    out = None
    for _ in range(ticks):
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        controller.current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    return out, now


def _public_safe_commit_controller(*, now_s=100.0):
    """Enter Gate-0 COMMIT through the dense, credited F163 camera trace."""

    return _replay_f163_safe_passage(base_s=now_s, stop_at_commit=True)


def test_close_loss_without_armed_commit_never_coasts():
    # F102: the gate-0 scale-triggered hot coast is deleted — ONE crossing
    # policy.  A centered close loss in TRACK (no armed COMMIT) is an
    # ordinary loss: PREDICT carries the pursuit, never the ballistic
    # credit wait.  (Parent behavior: this exact scenario latched
    # COAST_FOR_CREDIT.)
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller.observe(_update([], frame_id=5), now_s=100.12)
    assert controller.state is not CleanCourseState.COAST_FOR_CREDIT
    now = 100.12
    for frame in range(6, 16):
        now += 0.033
        controller.observe(_update([], frame_id=frame), now_s=now)
        assert controller.state is not CleanCourseState.COAST_FOR_CREDIT
        if controller.state is CleanCourseState.PREDICT:
            break
    assert controller.state is CleanCourseState.PREDICT


def _commit_controller_gate_zero(now_s=100.10):
    """F102 gate-0 variant of _commit_controller: same near-plane,
    budget-satisfying TRACK setup but still on the GATE-0 leg (no
    promotion) — the unified crossing policy must arm COMMIT at gate 0
    exactly as it does at gate 1+."""

    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller._alt_est_m = 2.0  # honest altitude (floor quiet)
    current = controller.current
    current.x_axis.p = 0.10
    current.y_axis.p = 0.05
    current.raw_x = 0.10
    current.raw_y = 0.05
    current.aperture_half_x = 0.25
    current.aperture_half_y = 0.25
    current.outer_log_scale = -0.50
    current.outer_half_span_x = 0.25
    _settle_commit_passage_covariance(current)
    return controller


def test_commit_entry_arms_on_gate_zero():
    # F102: COMMIT is gate-agnostic — there is ONE energy-budgeted crossing
    # policy, gate 0 included.  The same sustained, aligned, freshly
    # measured near-plane regime that commits on a gate-1+ leg commits on
    # gate 0.  (Parent behavior: COMMIT required gate_index >= 1, so gate 0
    # could only cross via the deleted hot coast.)
    controller = _commit_controller_gate_zero()
    out, now = _drive_commit_window(controller, 100.10)
    assert controller.state is CleanCourseState.COMMIT

    # Integrated IMU-vz is supporting evidence, not a second crossing owner.
    # A biased/high estimate cannot veto the same clear visual intercept.
    climbing = _commit_controller_gate_zero()
    climbing._vz_est_m_s = 0.64
    _drive_commit_window(climbing, 100.10)
    assert climbing.state is CleanCourseState.COMMIT


def test_gate_zero_far_closure_avoids_early_fast_brake_dive():
    # F108: F107's 0.36/s threshold latched the fast intercept response far
    # from Gate 0, accumulated sink before the near-plane hold, and struck
    # structure without credit.  At outer log -1.9, a 0.36/s observation
    # gets a continuous weak-brake blend but does not latch the fast path.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.15))
    samples, _now, _scale = _replay_outer_growth(
        controller,
        track_id="A",
        x=0.0,
        y=0.0,
        start_scale=0.15,  # outer log ~= -1.90
        closure_rate_s=0.36,
        now_s=100.033,
        frames=30,
    )
    targets = [sample[0].target_pitch_rad for sample in samples]
    demands = [sample[4] for sample in samples]
    brake_states = [sample[5] for sample in samples]
    out = samples[-1][0]
    assert controller.state is CleanCourseState.TRACK
    assert not any(brake_states)
    assert demands[-1] < controller.config.pitch_brake_exit_demand
    # The far 0.36/s trace may ask for a weak continuous energy reduction,
    # but it never takes the historical fast/full-brake dive.
    assert min(targets) > controller.config.spawn_pitch_rad - 0.03
    assert min(targets) > (
        controller.config.spawn_pitch_rad
        + controller.config.pre_cross_brake_pitch_rad
        + 0.10
    )
    assert max(abs(b - a) for a, b in zip(targets, targets[1:])) <= (
        controller.config.pre_cross_brake_slew_rad_s * 0.033 + 1e-12
    )


def test_gate_zero_budget_false_near_plane_hold_keeps_gentle_brake():
    # F109: the F104/F108 full-course escalation pitched Gate 0 sharply up
    # while sinking and twice struck object 1001 without credit.  The
    # near-plane budget-false hold still suppresses advance and demands the
    # brake, but Gate 0 keeps the F102/F103/F109/F110-proven -0.15 attitude.
    controller = _commit_controller_gate_zero()
    current = controller.current
    # Keep the trajectory outside the 60% corridor core.  A centered hot
    # approach at this range is now an explicit point-of-no-return commit.
    current.x_axis.p = 0.16
    current.raw_x = 0.16
    current.y_axis.p = 0.15
    current.raw_y = 0.15
    current.scale_axis.p = -1.0
    current.scale_axis.v = 0.0
    current.outer_log_scale = -1.0
    current.outer_expansion_rate = 1.5
    weak_brake_attitude = (
        controller.config.spawn_pitch_rad
        + controller.config.pre_cross_brake_pitch_rad
    )
    now = 100.10
    out = None
    for _ in range(15):
        now += 0.033
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        current.last_y_measurement_s = now
        out = _command(controller, now, pitch=weak_brake_attitude)

    assert controller.state is CleanCourseState.TRACK
    assert controller._pre_cross_brake_active
    assert out.target_pitch_rad == pytest.approx(weak_brake_attitude, abs=1e-9)


def test_commit_entry_fires_sustained_aligned_near_plane():
    # F53 (20260729T233602Z-visual-course-072c8a7b): the misalignment brake
    # self-locked the F52 drone into a hover 1-2 m short of gate 1's plane.
    # A sustained (~0.1 s, F54), aligned, freshly measured near-plane
    # regime on a gate-1+ leg commits to an inertial crossing instead —
    # 2026-07-30: only once the unified aperture/drift budget passes.
    controller = _commit_controller()
    out, now = _drive_commit_window(controller, 100.10)
    assert controller.state is CleanCourseState.COMMIT
    assert out.state is CleanCourseState.COMMIT
    # COMMIT steers fresh ex=+0.10 with the same near-plane range authority.
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)


def test_commit_entry_refuses_predicted_blackout_drift():
    # F61 (20260730T012351Z-visual-course-5a0fe853): an entry with the
    # bearing inside the corridor but still moving ran its drift
    # uncorrected through the close-range censorship blackout and crossed
    # beside the opening.  2026-07-30: the entry budget includes the
    # PREDICTED blackout displacement — |error| + |rate| * 0.50 s of blind
    # drift must fit inside 60% of the aperture half-extent.  A bearing
    # aligned but drifting at 0.5 norm/s (0.10 + 0.25 > 0.6*0.25) is
    # refused; the same geometry settled commits.
    drifting = _commit_controller()
    drifting.current.x_axis.v = 0.5
    _drive_commit_window(drifting, 100.10)
    assert drifting.state is CleanCourseState.TRACK
    settled = _commit_controller()
    settled.current.x_axis.v = 0.0
    out, _ = _drive_commit_window(settled, 100.10)
    assert settled.state is CleanCourseState.COMMIT


def test_commit_entry_uses_visual_intercept_not_imu_vz_as_a_veto():
    # IMU vertical velocity is an integrated supporting estimate.  It may
    # damp the commanded trajectory, but cannot veto a clear optical passage
    # envelope; uncertainty and aperture containment remain the admission
    # guards (covered by the adjacent drift/aperture tests).
    climbing = _commit_controller()
    climbing._vz_est_m_s = 0.64
    _drive_commit_window(climbing, 100.10)
    assert climbing.state is CleanCourseState.COMMIT
    settled = _commit_controller()
    settled._vz_est_m_s = 0.10
    out, _ = _drive_commit_window(settled, 100.10)
    assert settled.state is CleanCourseState.COMMIT


def test_f162_observations_have_no_live_aperture_commit_overlap():
    # This is deliberately separate from the synthetic public-boundary
    # reachability proof in test_aigp_vq2_runner.py.  F162's real Gate-1
    # observations lost their usable aperture at t=4.656; proximity did not
    # begin until t=5.062, and closure was still above the controlled-approach
    # target.  This remains evidence of no live overlap, not a scalar veto.
    rows = (
        # t, frame, x, y, bbox width/height, pitch,
        # optional (aperture center, half-size, raw log-scale)
        (4.359, 1907073, -0.13125, 0.04444444444444451, 0.1625, 0.2833333333333333, -0.4576878453806965, ((-0.1563614426, -0.0043497345), (0.048412277765566725, 0.08626446521721656), -2.6106008221)),
        (4.640, 1907079, -0.103125, 0.011111111111111072, 0.190625, 0.3111111111111111, -0.444454312243036, ((-0.1303246943, -0.0268291262), (0.05539495859669369, 0.09349205793547188), -2.6589443926)),
        (4.656, 1907080, -0.10, 0.005555555555555536, 0.1953125, 0.3138888888888889, -0.4429891285637212, ((-0.1275914617, -0.0183558378), (0.051281969088538604, 0.09164662350284151), -2.6656886059)),
        (4.812, 1907084, -0.090625, 0.0, 0.2078125, 0.3305555555555555, -0.43636218682192973, None),
        (5.062, 1907088, -0.084375, 0.03333333333333344, 0.246875, 0.3722222222222222, -0.44265084811207817, None),
        (5.125, 1907090, -0.0875, 0.05555555555555558, 0.259375, 0.39166666666666666, -0.44686735045384735, None),
        (5.265, 1907092, -0.096875, 0.11111111111111116, 0.2890625, 0.4388888888888889, -0.4517531147379503, None),
    )

    def observation(row):
        _, _, x, y, width, height, _, aperture_fit = row
        return _f163_trace_track(
            outer_center=(x, y),
            outer_span=(width, height),
            track_id="A",
            aperture_center=(
                None if aperture_fit is None else aperture_fit[0]
            ),
            aperture_half=(
                None if aperture_fit is None else aperture_fit[1]
            ),
            aperture_log_scale=(
                None if aperture_fit is None else aperture_fit[2]
            ),
        )

    first = rows[0]
    controller = CleanCourseController(_config())
    controller.initialize(
        _update([observation(first)], frame_id=first[1]),
        gate_index=1,
        fallback_center_norm=(first[2], first[3]),
        fallback_apparent_scale=math.sqrt(first[4] * first[5]),
        now_s=100.0 + first[0],
    )
    controller._alt_est_m = 2.0
    last_usable_s = None
    first_proximity_s = None
    proximity_closures = []

    for row in rows:
        elapsed, frame_id, *_ = row
        now = 100.0 + elapsed
        if row is not first:
            controller.observe(
                _update([observation(row)], frame_id=frame_id),
                now_s=now,
            )
        current = controller.current
        if current.aperture_half_x is not None:
            last_usable_s = elapsed
        if current.outer_log_scale >= controller.config.commit_min_log_scale:
            if first_proximity_s is None:
                first_proximity_s = elapsed
            proximity_closures.append(
                controller._outer_closure_estimate(current, now)[0]
            )
        _command(controller, now, pitch=row[6])
        assert controller.state is CleanCourseState.TRACK
        assert controller.gate_index == 1
        assert controller._commit_entry_s is None

    assert last_usable_s == pytest.approx(4.656)
    assert first_proximity_s == pytest.approx(5.062)
    assert last_usable_s < first_proximity_s
    assert proximity_closures
    assert min(proximity_closures) > (
        controller.config.closure_target_rate_s
    )
    assert controller.current.aperture_half_x is None
    assert controller.current.aperture_half_y is None


def _f163_trace_track(
    *,
    outer_center,
    outer_span,
    confidence=0.90,
    track_id="recorded-current",
    aperture_center=None,
    aperture_half=None,
    aperture_log_scale=None,
    clipping=FrameEdge.NONE,
):
    """Faithful recorded-state adapter using the real tracker unit contract.

    ``bbox_norm`` is [0, 1], while centers and aperture half-extents are in
    [-1, 1].  Thus a bbox width is already an outer half-extent in center
    coordinates.  This helper deliberately does not reuse the older compact
    ``_track`` fixture, whose square shorthand masks that conversion.
    """

    x, y = outer_center
    width, height = outer_span
    center_unit_x = 0.5 * (x + 1.0)
    center_unit_y = 0.5 * (y + 1.0)
    aperture = None
    if aperture_center is not None:
        aperture = SimpleNamespace(
            center_norm=tuple(float(v) for v in aperture_center),
            log_scale=(
                0.5
                * math.log(
                    max(1e-12, aperture_half[0] * aperture_half[1])
                )
                if aperture_log_scale is None
                else float(aperture_log_scale)
            ),
            confidence=0.90,
            measurement_std=(0.006, 0.009, 0.03),
            passage_usable=True,
            half_size_norm=tuple(float(v) for v in aperture_half),
        )
    return SimpleNamespace(
        track_id=track_id,
        center_norm=(float(x), float(y)),
        bbox_norm=(
            center_unit_x - width / 2.0,
            center_unit_y - height / 2.0,
            center_unit_x + width / 2.0,
            center_unit_y + height / 2.0,
        ),
        apparent_scale=math.sqrt(width * height),
        confidence=float(confidence),
        association_confidence=float(confidence),
        clipping=clipping,
        center_censored=clipping != FrameEdge.NONE,
        ambiguous=False,
        visible=True,
        inner_aperture=aperture,
    )


# Dense credited Gate-0 suffix from F163 run
# 20260801T231436Z-visual-course-571628a6, session SHA-256
# 98D73CC8FB87734DC44745E3283791B15622D12CE86362070F06D8A51C3F1B53.
# Schema: t, frame, outer center/span, confidence, aperture center/half,
# body rates, rpy, accel trust, horizontal specific force.
_F163_SAFE_PASSAGE_ROWS = (
    (1.828, 2052994, (0.021875, 0.1277777778), (0.2125, 0.3694444444), 0.9926981189, (0.02747558174, 0.12069230851), (0.10445159567, 0.19120112953), (-0.05685060276, -0.13435577775, -0.07676949182), (-0.01161387480, -0.41199422433, -0.02421808669), 0.0, 3.31968227541),
    (1.860, 2052995, (0.028125, 0.1388888889), (0.2171875, 0.3777777778), 0.9930474868, (0.03207603041, 0.13486609520), (0.09928892398, 0.18798711633), (-0.06086330666, -0.14927748764, -0.08433219621), (-0.01245555114, -0.41676566920, -0.02726496952), 0.0, 3.34929502497),
    (1.891, 2052996, (0.0375, 0.15), (0.23125, 0.3888888889), 0.9887848804, (0.03464432098, 0.13941187422), (0.11210044264, 0.20044544847), (-0.06483166470, -0.13969718838, -0.09263654629), (-0.01331185702, -0.42233451268, -0.03060859318), 0.0, 3.38353795399),
    (1.922, 2052997, (0.034375, 0.1555555556), (0.228125, 0.3972222222), 0.9914672807, (0.03684542394, 0.14819049608), (0.10867198089, 0.20133092518), (-0.06548158629, -0.10156004304, -0.09443096647), (-0.01395407570, -0.42536905200, -0.03343393328), 0.0, 3.39930493130),
    (1.953, 2052998, (0.053125, 0.1611111111), (0.25, 0.4083333333), 0.9813742815, (0.04032598298, 0.14828635655), (0.11812225482, 0.21245927244), (-0.06907181247, -0.11770118946, -0.09360531548), (-0.01485857626, -0.42928448263, -0.03695478312), 0.0, 3.42640835217),
    (1.985, 2052999, (0.05625, 0.1666666667), (0.25625, 0.4194444444), 0.9789414023, (0.04623212679, 0.15294103447), (0.12019049216, 0.20777105437), (-0.06856188728, -0.11544314945, -0.09417363086), (-0.01560777823, -0.43266183278, -0.03977366948), 0.0, 3.45025952805),
    (2.016, 2053000, (0.065625, 0.1777777778), (0.2640625, 0.4305555556), 0.9758135841, (0.04849709016, 0.16118719293), (0.12130179082, 0.21874685704), (-0.06344126715, -0.08630889083, -0.09266605058), (-0.01638753573, -0.43601220621, -0.04328468894), 0.0, 3.47151104177),
    (2.047, 2053001, (0.06875, 0.1888888889), (0.271875, 0.4444444444), 0.9748436395, (0.05224455717, 0.16765588326), (0.12419605650, 0.22564466289), (-0.06156886174, -0.07590890849, -0.09145576337), (-0.01705827554, -0.43876611817, -0.04676939093), 0.0, 3.49002488837),
    (2.078, 2053002, (0.065625, 0.1944444444), (0.271875, 0.4555555556), 0.9787903487, (0.05574664871, 0.17028945524), (0.12623209884, 0.23045571313), (-0.06112220688, -0.07511904264, -0.08770259300), (-0.01762385229, -0.44092398776, -0.04946362546), 0.0, 3.50350377074),
    (2.110, 2053003, (0.06875, 0.2), (0.278125, 0.4722222222), 0.9829537741, (0.06072936574, 0.17774406340), (0.13229861713, 0.23577746783), (-0.05397962956, -0.06300983751, -0.08044802228), (-0.01826470805, -0.44333571377, -0.05260503123), 0.0, 3.51602412272),
    (2.141, 2053004, (0.06875, 0.2055555556), (0.2828125, 0.4861111111), 0.9871679989, (0.06431546224, 0.18285554660), (0.13634379771, 0.24648341995), (-0.04620899065, -0.05479315097, -0.07526550272), (-0.01870845179, -0.44537403288, -0.05553422208), 0.0, 3.52610154932),
)


def _replay_f163_safe_passage(*, base_s=100.0, stop_at_commit=False):
    """Faithfully replay the dense F163 safe Gate-0 crossing observations."""

    def track(row):
        return _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            confidence=row[4],
            aperture_center=row[5],
            aperture_half=row[6],
        )

    first = _F163_SAFE_PASSAGE_ROWS[0]
    controller = CleanCourseController(_config())
    controller.initialize(
        _update([track(first)], frame_id=first[1]),
        gate_index=0,
        fallback_center_norm=first[2],
        fallback_apparent_scale=math.sqrt(first[3][0] * first[3][1]),
        now_s=base_s + first[0],
    )
    outputs = []
    now = base_s + first[0]
    for index, row in enumerate(_F163_SAFE_PASSAGE_ROWS):
        now = base_s + row[0]
        if index:
            controller.observe(
                _update([track(row)], frame_id=row[1]),
                now_s=now,
                body_rates=row[7],
            )
        outputs.append(
            _command(
                controller,
                now,
                roll=row[8][0],
                pitch=row[8][1],
                yaw=row[8][2],
                fh=row[10],
                accel_trust=row[9],
            )
        )
        if stop_at_commit and controller.state is CleanCourseState.COMMIT:
            break
    return controller, outputs, now


def _replay_f163_rows(rows, *, gate_index):
    controller = CleanCourseController(_config())
    first = rows[0]

    def track(row):
        return _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            aperture_center=row[6],
            aperture_half=row[7],
            clipping=row[9],
        )

    controller.initialize(
        _update([track(first)], frame_id=first[1]),
        gate_index=gate_index,
        fallback_center_norm=first[2],
        fallback_apparent_scale=math.sqrt(first[3][0] * first[3][1]),
        now_s=100.0 + first[0],
    )
    controller._alt_est_m = 2.0
    outputs = []
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        if index:
            controller.observe(
                _update([track(row)], frame_id=row[1]),
                now_s=now,
                body_rates=row[8],
            )
        outputs.append(_command(controller, now, pitch=row[4], yaw=0.0))
    return controller, outputs


@pytest.mark.parametrize(
    ("trace_sha256", "rows"),
    (
        (
            "e3e8429ef13b571e0f3bf454ab9fd393db8686d4aa903caf1440195481b3471c",
            (
                (1.922, 2696544, (0.031250, -0.027778), (0.251563, 0.405556), -0.364656, 0, (0.010424, -0.038464), (0.113205, 0.188197), (-0.047621, -0.182014, -0.047166), FrameEdge.NONE),
                (1.953, 2696545, (0.031250, -0.022222), (0.256250, 0.419444), -0.370211, 0, (0.013264, -0.032444), (0.117761, 0.196182), (-0.046880, -0.243993, -0.047212), FrameEdge.NONE),
                (1.984, 2696546, (0.034375, -0.005556), (0.265625, 0.430556), -0.382806, 0, (0.015843, -0.013825), (0.119704, 0.200810), (-0.038848, -0.379908, -0.051116), FrameEdge.NONE),
                (2.031, 2696547, (0.031250, 0.011111), (0.268750, 0.444444), -0.395643, 0, (0.018291, 0.005432), (0.115649, 0.199979), (-0.046074, -0.174604, -0.061796), FrameEdge.NONE),
                (2.062, 2696548, (0.031250, 0.016667), (0.276563, 0.458333), -0.401968, 0, None, None, (-0.051793, -0.204330, -0.066284), FrameEdge.NONE),
                (2.109, 2696549, (0.031250, 0.027778), (0.279688, 0.472222), -0.411056, 0, (0.031655, 0.039110), (0.100726, 0.201090), (-0.064456, -0.196444, -0.075454), FrameEdge.NONE),
                (2.156, 2696551, (0.034375, 0.038889), (0.300000, 0.508333), -0.418991, 0, (0.035504, 0.022886), (0.145161, 0.257677), (-0.066682, -0.141230, -0.081665), FrameEdge.NONE),
                (2.203, 2696552, (0.037500, 0.044444), (0.309375, 0.530556), -0.425688, 0, (0.038520, 0.022527), (0.148376, 0.267867), (-0.079664, -0.132913, -0.084561), FrameEdge.NONE),
                (2.250, 2696554, (0.046875, 0.050000), (0.334375, 0.575000), -0.431619, 0, (0.045755, 0.028358), (0.158147, 0.287340), (-0.090256, -0.104591, -0.092341), FrameEdge.NONE),
                (2.297, 2696555, (0.053125, 0.055556), (0.351563, 0.597222), -0.435745, 0, (0.051314, 0.026475), (0.165800, 0.300877), (-0.098689, -0.091561, -0.102269), FrameEdge.NONE),
                (2.344, 2696556, (0.056250, 0.055556), (0.368750, 0.630556), -0.439981, 0, (0.058657, 0.024443), (0.174567, 0.311036), (-0.109746, -0.077939, -0.113691), FrameEdge.NONE),
                (2.390, 2696558, (0.012500, 0.066667), (0.468750, 0.694444), -0.443579, 0, (0.071903, 0.016193), (0.189395, 0.340023), (-0.115953, -0.063722, -0.122110), FrameEdge.NONE),
                (2.437, 2696559, (0.025000, 0.066667), (0.481250, 0.725000), -0.446089, 0, (0.076812, 0.013500), (0.195896, 0.355116), (-0.139359, -0.051618, -0.115476), FrameEdge.NONE),
                (2.484, 2696561, (0.068750, 0.072222), (0.520313, 0.825000), -0.448882, 0, (0.092365, 0.005208), (0.217369, 0.399447), (-0.137812, -0.043502, -0.098252), FrameEdge.NONE),
                (2.531, 2696562, (0.090625, 0.077778), (0.542188, 0.880556), -0.450669, 0, (0.099343, 0.000089), (0.229652, 0.427418), (-0.098331, -0.035347, -0.080795), FrameEdge.NONE),
            ),
        ),
        (
            "a41993108aeb7fb6ec55c08fd0fb0b75e53c1a0142a8f063ddd65f5e0d42569f",
            (
                (1.937, 2718484, (0.009375, -0.033333), (0.234375, 0.402778), -0.369783, 0, (0.005982, -0.037666), (0.108583, 0.182825), (-0.070939, -0.219892, -0.052894), FrameEdge.NONE),
                (1.968, 2718485, (0.025000, -0.027778), (0.254688, 0.413889), -0.378288, 0, (0.011092, -0.027007), (0.121359, 0.217607), (-0.072263, -0.282841, -0.055783), FrameEdge.NONE),
                (1.984, 2718486, (0.031250, -0.016667), (0.267188, 0.425000), -0.382728, 0, (0.013806, -0.016821), (0.125058, 0.218849), (-0.070299, -0.328649, -0.058813), FrameEdge.NONE),
                (2.031, 2718487, (0.028125, 0.000000), (0.268750, 0.441667), -0.395596, 0, (0.017030, -0.000921), (0.125847, 0.224592), (-0.065823, -0.199785, -0.069174), FrameEdge.NONE),
                (2.078, 2718489, (0.037500, 0.016667), (0.285938, 0.466667), -0.404113, 0, (0.020198, 0.004040), (0.130215, 0.222431), (-0.076147, -0.213920, -0.075042), FrameEdge.NONE),
                (2.125, 2718490, (0.031250, 0.027778), (0.285938, 0.483333), -0.415256, 0, (0.023831, 0.005754), (0.136876, 0.235036), (-0.083148, -0.166978, -0.092345), FrameEdge.NONE),
                (2.171, 2718491, (0.034375, 0.033333), (0.296875, 0.500000), -0.421710, 0, (0.034485, 0.026493), (0.141826, 0.254488), (-0.099510, -0.141709, -0.104617), FrameEdge.NONE),
                (2.218, 2718493, (0.037500, 0.038889), (0.317188, 0.541667), -0.428159, 0, (0.037169, 0.011692), (0.148427, 0.265362), (-0.111934, -0.121715, -0.112673), FrameEdge.NONE),
                (2.265, 2718494, (0.043750, 0.044444), (0.329688, 0.561111), -0.432907, 0, (0.046420, 0.015437), (0.156607, 0.278136), (-0.127179, -0.100349, -0.124252), FrameEdge.NONE),
                (2.312, 2718496, (0.056250, 0.050000), (0.362500, 0.616667), -0.437554, 0, (0.059553, 0.013493), (0.170348, 0.303392), (-0.138713, -0.086296, -0.136208), FrameEdge.NONE),
                (2.359, 2718497, (-0.003125, 0.050000), (0.445313, 0.647222), -0.441594, 0, (0.066416, 0.011906), (0.176406, 0.316970), (-0.140011, -0.071061, -0.140792), FrameEdge.NONE),
                (2.406, 2718499, (0.025000, 0.055556), (0.471875, 0.711111), -0.444477, 0, (0.079871, 0.005495), (0.191204, 0.346417), (-0.150980, -0.060147, -0.130804), FrameEdge.NONE),
                (2.453, 2718500, (0.043750, 0.055556), (0.487500, 0.752778), -0.447239, 0, (0.087879, -0.001891), (0.199888, 0.361410), (-0.151921, -0.047510, -0.108490), FrameEdge.NONE),
                (2.500, 2718501, (0.062500, 0.061111), (0.506250, 0.802778), -0.449834, 0, (0.092310, -0.006749), (0.206420, 0.387327), (-0.110411, -0.038826, -0.078857), FrameEdge.NONE),
            ),
        ),
    ),
)
def test_f170_f168_three_pixel_tubes_do_not_authorize_commit(
    trace_sha256, rows
):
    # Both the credited and failed F168 flights entered on a 3-4 px margin
    # against the mathematical opening.  The full uncertainty tube now has to
    # fit the existing reduced body/frame-clearance core, not merely its center.
    controller = CleanCourseController(_config())
    first = rows[0]
    confidence_by_frame = {
        2696544: 0.9801375803, 2696545: 0.9767619111,
        2696546: 0.9734714314, 2696547: 0.9756093140,
        2696548: 0.9769510206, 2696549: 0.9817691689,
        2696551: 0.9844122961, 2696552: 0.9859965093,
        2696554: 0.9860908811, 2696555: 0.9860306161,
        2696556: 0.9854176388, 2696558: 0.9603054074,
        2696559: 0.9523105102, 2696561: 0.9596148635,
        2696562: 0.9650251164,
        2718484: 0.9902227913, 2718485: 0.9795292800,
        2718486: 0.9715369811, 2718487: 0.9731805706,
        2718489: 0.9723839222, 2718490: 0.9794483781,
        2718491: 0.9824062335, 2718493: 0.9843695661,
        2718494: 0.9846927543, 2718496: 0.9848094988,
        2718497: 0.9538990087, 2718499: 0.9482702707,
        2718500: 0.9493967827, 2718501: 0.9545299331,
    }

    def track(row):
        return _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            confidence=confidence_by_frame[row[1]],
            aperture_center=row[6],
            aperture_half=row[7],
        )

    controller.initialize(
        _update([track(first)], frame_id=first[1]),
        gate_index=0,
        fallback_center_norm=first[2],
        fallback_apparent_scale=math.sqrt(first[3][0] * first[3][1]),
        now_s=100.0 + first[0],
    )
    chance_margin_frames = []
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        if index:
            controller.observe(
                _update([track(row)], frame_id=row[1]),
                now_s=now,
                body_rates=row[8],
            )
        output = _command(controller, now, pitch=row[4], yaw=0.0)
        admission = controller._last_commit_admission
        corridor = controller._transported_corridor(
            controller.current, now_s=now
        )
        if (
            corridor is not None
            and admission.y_tube is not None
            and admission.y_tube < corridor.half_y
            and admission.y_tube > admission.y_safe_half
        ):
            chance_margin_frames.append((row[0], admission, corridor))
        assert output.state is CleanCourseState.TRACK
        assert controller._commit_entry_s is None

    assert trace_sha256
    assert chance_margin_frames
    for _t, admission, corridor in chance_margin_frames:
        assert admission.y_budget == pytest.approx(
            controller.config.commit_entry_aperture_margin_frac
            * corridor.half_y
        )
        assert admission.y_safe_half == pytest.approx(
            (1.0 - controller.config.commit_body_frame_clearance_frac)
            * corridor.half_y
        )
        assert admission.y_clearance_reserve == pytest.approx(
            corridor.half_y - admission.y_safe_half
        )
        assert admission.y_upper_clearance_reserve == pytest.approx(
            admission.y_clearance_reserve
        )
        assert admission.y_lower_clearance_reserve == pytest.approx(
            admission.y_clearance_reserve
        )
        assert not admission.admissible


def test_f170_commit_sustain_belongs_to_fresh_safe_tube():
    def centered_track():
        return _f163_trace_track(
            outer_center=(0.0, 0.0),
            outer_span=(0.40, 0.50),
            aperture_center=(0.0, 0.0),
            aperture_half=(0.35, 0.35),
        )

    controller = CleanCourseController(_config())
    controller.initialize(
        _update([centered_track()], frame_id=9000),
        gate_index=0,
        fallback_center_norm=(0.0, 0.0),
        fallback_apparent_scale=math.sqrt(0.40 * 0.50),
        now_s=100.0,
    )
    first_safe_s = None
    entry_s = None
    for frame in range(1, 8):
        now = 100.0 + 0.033 * frame
        controller.observe(
            _update([centered_track()], frame_id=9000 + frame), now_s=now
        )
        output = _command(controller, now, pitch=SPAWN_PITCH)
        if controller._last_commit_admission.admissible:
            first_safe_s = first_safe_s or now
        if output.state is CleanCourseState.COMMIT:
            entry_s = now
            break

    assert first_safe_s is not None
    assert entry_s is not None
    assert entry_s - first_safe_s >= controller.config.commit_sustain_s

    # One safe camera frame cannot accrue a 0.10 s lease by command-rate
    # republication: freshness expires at .06 s and resets safe sustain.
    frozen = CleanCourseController(_config())
    frozen_update = _update([centered_track()], frame_id=9100)
    frozen.initialize(
        frozen_update,
        gate_index=0,
        fallback_center_norm=(0.0, 0.0),
        fallback_apparent_scale=math.sqrt(0.40 * 0.50),
        now_s=200.0,
    )
    for offset in (0.02, 0.05, 0.08, 0.12, 0.16):
        frozen.observe(frozen_update, now_s=200.0 + offset)
        output = _command(frozen, 200.0 + offset, pitch=SPAWN_PITCH)
        assert output.state is not CleanCourseState.COMMIT
    assert frozen._commit_safe_since_s is None

    # Proximity time before an unsafe frame cannot leak into the next safe
    # interval.  Fresh safe containment has to restart and sustain in full.
    reset = CleanCourseController(_config())
    reset.initialize(
        _update([centered_track()], frame_id=9200),
        gate_index=0,
        fallback_center_norm=(0.0, 0.0),
        fallback_apparent_scale=math.sqrt(0.40 * 0.50),
        now_s=300.0,
    )
    for frame in (1, 2):
        now = 300.0 + 0.033 * frame
        reset.observe(
            _update([centered_track()], frame_id=9200 + frame), now_s=now
        )
        _command(reset, now, pitch=SPAWN_PITCH)
    assert reset._commit_safe_since_s is not None
    unsafe = _f163_trace_track(
        outer_center=(0.30, 0.0),
        outer_span=(0.40, 0.50),
        aperture_center=(0.30, 0.0),
        aperture_half=(0.35, 0.35),
    )
    reset.observe(_update([unsafe], frame_id=9203), now_s=300.099)
    _command(reset, 300.099, pitch=SPAWN_PITCH)
    assert reset._last_commit_admission.status == (
        "corridor-known/not-contained"
    )
    assert reset._commit_safe_since_s is None

    restarted_safe_s = None
    for frame in range(4, 16):
        now = 300.0 + 0.033 * frame
        reset.observe(
            _update([centered_track()], frame_id=9200 + frame), now_s=now
        )
        output = _command(reset, now, pitch=SPAWN_PITCH)
        if reset._last_commit_admission.admissible:
            restarted_safe_s = restarted_safe_s or now
        if output.state is CleanCourseState.COMMIT:
            break
    assert reset.state is CleanCourseState.COMMIT
    assert restarted_safe_s is not None
    assert reset._commit_entry_s - restarted_safe_s >= (
        reset.config.commit_sustain_s
    )

    # Fresh one-sided vertical censorship is unsafe directional evidence, not
    # a stale blackout.  It immediately revokes the lease while TRACK can
    # still brake/recenter instead of carrying a worsening low/high path.
    censored = _f163_trace_track(
        outer_center=(0.0, 0.35),
        outer_span=(0.40, 0.50),
        clipping=FrameEdge.BOTTOM,
    )
    now += 0.033
    reset.observe(_update([censored], frame_id=9220), now_s=now)
    output = _command(reset, now, pitch=SPAWN_PITCH)
    assert output.state is CleanCourseState.TRACK
    assert reset._last_commit_admission.status == "directionally-censored"
    assert reset._commit_entry_s is None


def test_f166_f163_launch_trace_keeps_aperture_rate_out_of_pitch():
    # F163 trace 20260801T231436Z-visual-course-571628a6, SHA-256
    # 98D73CC8FB87734DC44745E3283791B15622D12CE86362070F06D8A51C3F1B53.
    # F165 restored the fitted aperture derivative as an early pitch trigger;
    # F166 rejects that ownership model.  Replay the same public observations
    # and require pitch energy to follow the co-timed outer expansion even
    # while the inner fit reports the old >0.7/s spike.
    rows = (
        # t, frame, outer center/span/confidence, aperture center/half/log,
        # body rates, measured pitch, measured yaw
        (0.407, 2052952, (0.0, -0.033333333), (0.1296875, 0.227777778), 0.851868156, (-0.000933081, -0.017385977), (0.045903510, 0.090086504), -2.744099305, (-0.008308031, 0.009424807, -0.002845698), -0.305993759, -0.001185045),
        (0.453, 2052953, (-0.003125, -0.033333333), (0.128125, 0.227777778), 0.855911670, (-0.001167147, -0.034773992), (0.046731509, 0.109100640), -2.639410574, (-0.018228127, 0.020392066, -0.004411468), -0.305280245, -0.001392911),
        (0.485, 2052954, (-0.003125, -0.033333333), (0.128125, 0.230555556), 0.857608214, (-0.015355154, -0.036636120), (0.039757721, 0.059959652), -3.019517308, (-0.008626368, 0.017318213, -0.003784995), -0.304711661, -0.001513124),
        (0.516, 2052955, (-0.003125, -0.033333333), (0.128125, 0.230555556), 0.859020659, (-0.014679623, -0.032506987), (0.040688239, 0.068968167), -2.937963221, (-0.001311761, 0.001084940, -0.002400813), -0.304534776, -0.001612994),
        (0.563, 2052957, (0.0, -0.027777778), (0.1328125, 0.233333333), 0.863247345, (0.001467546, -0.020777371), (0.056844649, 0.111670561), -2.529817679, (-0.024912961, 0.009831239, -0.004120315), -0.304370371, -0.001751474),
        (0.594, 2052958, (0.0, -0.027777778), (0.1328125, 0.236111111), 0.868875305, (0.001654859, -0.020068872), (0.061655727, 0.117291094), -2.464642808, (-0.019537756, 0.008246623, -0.005552340), -0.303947059, -0.001946011),
        (0.657, 2052959, (-0.003125, -0.022222222), (0.1375, 0.233333333), 0.865953079, (0.000486117, -0.016697485), (0.057350383, 0.112707698), -2.520766654, (0.003951231, -0.245745698, -0.000897831), -0.310695225, -0.002120039),
        (0.688, 2052960, (-0.003125, -0.011111111), (0.1359375, 0.236111111), 0.871015313, (0.000612717, 0.000513752), (0.061687172, 0.117244633), -2.464585963, (-0.011348313, -0.201032633, 0.000179844), -0.319845210, -0.002079399),
        (0.719, 2052961, (-0.003125, 0.005555556), (0.1375, 0.238888889), 0.875176538, (-0.001012115, 0.006616010), (0.064512614, 0.115768434), -2.448528929, (0.012021052, -0.087661372, -0.000271794), -0.320806494, -0.002077557),
        (0.750, 2052962, (-0.003125, 0.005555556), (0.1375, 0.238888889), 0.878974089, (0.001971353, 0.011304708), (0.065016277, 0.119250847), -2.429821834, (0.013103569, -0.359451378, -0.000589905), -0.328703713, -0.002059033),
    )

    def track(row):
        return _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            confidence=row[4],
            aperture_center=row[5],
            aperture_half=row[6],
            aperture_log_scale=row[7],
        )

    first = rows[0]
    controller = CleanCourseController(_config())
    controller.initialize(
        _update([track(first)], frame_id=first[1]),
        gate_index=0,
        fallback_center_norm=first[2],
        fallback_apparent_scale=math.sqrt(first[3][0] * first[3][1]),
        now_s=100.0 + first[0],
    )
    controller._alt_est_m = 2.0
    samples = []
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        if index:
            controller.observe(
                _update([track(row)], frame_id=row[1]),
                now_s=now,
                body_rates=row[8],
            )
        output = _command(
            controller,
            now,
            pitch=row[9],
            yaw=row[10],
        )
        samples.append(
            (
                row[0],
                controller._pre_cross_brake_active,
                output,
                controller._control_closure_estimate(
                    controller.current, now
                )[0],
                controller._outer_closure_estimate(
                    controller.current, now
                )[0],
                controller.current.scale_axis.v,
                controller._last_vertical_support,
            )
        )

    assert not any(active for _t, active, *_rest in samples)
    assert all(
        sample[2].target_pitch_rad >= SPAWN_PITCH - 1e-12
        for sample in samples
    )
    assert max(sample[5] for sample in samples) > 0.70
    assert max(sample[4] for sample in samples) < 0.20
    # Control and admission consume only outer scale.  The control estimate is
    # deliberately a robust agreement blend of the two outer filters, so it may
    # sit slightly below admission's conservative max without ever following
    # the unrelated aperture-rate spike.
    assert all(0.0 <= sample[3] <= sample[4] for sample in samples)
    assert max(abs(sample[3] - sample[4]) for sample in samples) < 0.02
    assert min(
        sample[2].thrust - sample[6] for sample in samples
    ) > -0.002


def test_f166_republished_outer_frame_cannot_refresh_vertical_authority():
    # A tracker publication can repeat one decoded camera frame.  Replaying
    # an outer-only current track with the identical frame identity must be
    # prediction, not another y observation: timestamps remain at the real
    # source frame, TRACK exits immediately, exact qualification ends at its
    # stricter age, and visual control expires at the bounded PREDICT horizon
    # even though the track object is still present in every publication.
    frozen = _update(
        [_track("A", 0.0, -0.20, scale=0.14)], frame_id=2264999
    )
    controller = CleanCourseController(_config())
    controller.initialize(
        frozen,
        gate_index=1,
        fallback_center_norm=(0.0, -0.20),
        fallback_apparent_scale=0.14,
        now_s=100.0,
    )
    live = _command(controller, 100.0, pitch=SPAWN_PITCH)
    assert live.vertical_qualified
    assert controller._last_vertical_visual_delta > 0.0

    source_measurement_s = controller.current.last_measurement_s
    source_outer_y_s = controller.current.last_outer_y_evidence_s
    source_fresh_y_s = controller._current_fresh_outer_y_observation_s
    source_serial = controller._current_fresh_outer_y_observation_serial

    controller.observe(frozen, now_s=100.05)
    assert controller.state is CleanCourseState.PREDICT
    controller.observe(
        frozen,
        now_s=100.0 + controller.config.vertical_qualify_max_age_s + 0.01,
    )
    unqualified = _command(
        controller,
        100.0 + controller.config.vertical_qualify_max_age_s + 0.02,
        pitch=SPAWN_PITCH,
    )
    assert not unqualified.vertical_qualified
    assert controller._last_vertical_visual_delta > 0.0
    assert controller._current_fresh_outer_y_observation_s == source_fresh_y_s
    assert controller._current_fresh_outer_y_observation_serial == source_serial
    controller.observe(
        frozen,
        now_s=100.0 + controller.config.predict_max_gap_s + 0.01,
    )
    stale = _command(
        controller,
        100.0 + controller.config.predict_max_gap_s + 0.02,
        pitch=SPAWN_PITCH,
    )

    assert controller.state is CleanCourseState.SEARCH
    assert controller.current.last_measurement_s == source_measurement_s
    assert controller.current.last_outer_y_evidence_s == source_outer_y_s
    assert controller._current_fresh_outer_y_observation_s is None
    assert controller._current_fresh_outer_y_observation_serial == 0
    assert not stale.vertical_qualified
    assert controller._last_vertical_visual_delta == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("vertical_sign", "opposing_a_up"),
    [(-1.0, 4.0), (1.0, -4.0)],
)
def test_f166_uncertain_ttc_cannot_veto_fresh_outer_y(
    vertical_sign, opposing_a_up
):
    # F165 Gate 1, run 20260802T011212Z-visual-course-88a0af14.  These
    # co-timed public outer/aperture observations produced broad plane-
    # intercept uncertainty.  Mirror the recorded high gate to cover both
    # signs: the fresh outer position/rate baseline must remain corrective
    # even where TTC projection magnitude has essentially no authority, and
    # deliberately opposing IMU damping may reduce but never reverse it.
    rows = (
        (3.156, 2264641, (-0.3969, -0.2111), (0.1047, 0.2028), (-0.4028, -0.2575), (0.0404, 0.0646)),
        (3.188, 2264643, (-0.3938, -0.2111), (0.1062, 0.2083), (-0.3980, -0.2583), (0.0420, 0.0653)),
        (3.250, 2264644, (-0.3813, -0.2056), (0.1078, 0.2111), (-0.3919, -0.2512), (0.0440, 0.0668)),
        (3.281, 2264645, (-0.3719, -0.1889), (0.1094, 0.2111), (-0.3796, -0.2390), (0.0437, 0.0682)),
        (3.313, 2264646, (-0.3594, -0.1778), (0.1094, 0.2139), (-0.3676, -0.2221), (0.0437, 0.0706)),
        (3.344, 2264647, (-0.3469, -0.1556), (0.1094, 0.2167), (-0.3520, -0.2082), (0.0429, 0.0659)),
    )

    def track(row):
        outer = (row[2][0], vertical_sign * abs(row[2][1]))
        aperture = (row[4][0], vertical_sign * abs(row[4][1]))
        return _f163_trace_track(
            outer_center=outer,
            outer_span=row[3],
            aperture_center=aperture,
            aperture_half=row[5],
        )

    controller = CleanCourseController(_config())
    first = rows[0]
    controller.initialize(
        _update([track(first)], frame_id=first[1]),
        gate_index=1,
        fallback_center_norm=track(first).center_norm,
        fallback_apparent_scale=math.sqrt(first[3][0] * first[3][1]),
        now_s=100.0 + first[0],
    )
    samples = []
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        if index:
            controller.observe(
                _update([track(row)], frame_id=row[1]), now_s=now
            )
        output = _command(
            controller,
            now,
            pitch=SPAWN_PITCH,
            a_up=opposing_a_up,
        )
        samples.append(
            (
                controller._last_vertical_motion,
                controller._last_vertical_visual_delta,
                controller._last_vertical_imu_delta,
                controller._last_vertical_collective_target,
                controller._last_vertical_support,
                output,
            )
        )

    uncertain = [sample for sample in samples if sample[0].control_authority < 0.05]
    assert uncertain
    collective_sign = -vertical_sign
    for motion, visual, imu, target, support, output in uncertain:
        assert motion.intercept_std > 0.0
        assert collective_sign * visual > 0.0
        assert collective_sign * imu <= 0.0
        assert collective_sign * (target - support) > 0.0
        assert controller.config.min_thrust <= output.thrust <= controller.config.max_thrust


def test_f166_first_missing_aperture_frame_returns_vertical_to_outer():
    # F165 Gate 1 lost its last usable aperture at t=4.344.  The very first
    # missing-aperture camera frame must remove the bounded corridor offset;
    # the fresh outer y measurement becomes the complete vertical position
    # owner immediately, without one prediction tick of stale passage y.
    rows = (
        (4.203, 2264673, (-0.1281, -0.1500), (0.1625, 0.3139), -0.4627, (-0.1535, -0.1873), (0.0306, 0.0614)),
        (4.250, 2264675, (-0.1219, -0.1611), (0.1672, 0.3278), -0.4633, (-0.1514, -0.2391), (0.0517, 0.1001)),
        (4.297, 2264676, (-0.1187, -0.1667), (0.1734, 0.3333), -0.4638, (-0.1587, -0.2515), (0.0447, 0.0725)),
        (4.344, 2264677, (-0.1187, -0.1722), (0.1781, 0.3389), -0.4642, None, None),
    )

    def track(row):
        return _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            aperture_center=row[5],
            aperture_half=row[6],
        )

    controller = CleanCourseController(_config())
    first = rows[0]
    controller.initialize(
        _update([track(first)], frame_id=first[1]),
        gate_index=1,
        fallback_center_norm=first[2],
        fallback_apparent_scale=math.sqrt(first[3][0] * first[3][1]),
        now_s=100.0 + first[0],
    )
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        if index:
            controller.observe(
                _update([track(row)], frame_id=row[1]), now_s=now
            )
        _command(controller, now, pitch=row[4])

    outer_error = controller._compensated_ey(
        controller.current.outer_y_axis.p, rows[-1][4]
    )
    stale_passage_error = controller._compensated_ey(
        controller.current.y_axis.p, rows[-1][4]
    )
    assert controller.current.aperture_half_y is None
    assert not controller._last_vertical_aperture_live
    assert controller._last_vertical_aperture_offset == pytest.approx(0.0)
    assert controller._last_vertical_path_error == pytest.approx(
        outer_error, abs=1e-12
    )
    assert abs(outer_error - stale_passage_error) > 0.01
    assert controller._last_vertical_visual_delta > 0.0


def _maximum_advance_to_brake_reversal(pitches):
    """Measure nose-up motion only after launch has advanced past its start."""

    initial_pitch = pitches[0]
    running_nose_down_peak = initial_pitch
    advance_established = False
    maximum_reversal = 0.0
    for pitch in pitches[1:]:
        if pitch > initial_pitch:
            advance_established = True
        if not advance_established:
            # An immediate monotonic brake from the spawn attitude is not the
            # F165 failure.  The failure was advance first, then a large
            # derivative-triggered brake after the launch handoff.
            continue
        running_nose_down_peak = max(running_nose_down_peak, pitch)
        maximum_reversal = max(
            maximum_reversal,
            running_nose_down_peak - pitch,
        )
    return maximum_reversal


def test_f166_launch_transfer_is_bumpless_and_never_below_support():
    # F165 dipped when the fixed boost ended and the accumulated IMU climb
    # estimate was suddenly exposed.  A stationary, clearly high Gate 0
    # exercises both launch handoffs: pitch follows one direction from its
    # bounded initial slew without an advance-to-brake reversal, while fresh
    # outer-y keeps the collective target and filtered wire command on the
    # climb side of tilt support despite opposing IMU damping.
    controller = CleanCourseController(
        _config(launch_boost_duration_s=LAUNCH_BOOST_DURATION_S)
    )
    outputs = []
    supports = []
    targets = []
    modes = []
    nominal_log = 0.5 * math.log(0.05 * 0.10)
    for index in range(30):
        now = 100.0 + 0.033 * index
        track = _f163_trace_track(
            outer_center=(0.0, -0.05),
            outer_span=(0.13, 0.23),
            aperture_center=(0.0, -0.05),
            aperture_half=(0.05, 0.10),
            aperture_log_scale=nominal_log,
        )
        update = _update([track], frame_id=5100 + index)
        if index == 0:
            controller.initialize(
                update,
                gate_index=0,
                fallback_center_norm=(0.0, -0.05),
                fallback_apparent_scale=math.sqrt(0.13 * 0.23),
                now_s=now,
            )
        else:
            controller.observe(update, now_s=now)
        outputs.append(
            _command(
                controller,
                now,
                pitch=SPAWN_PITCH,
                a_up=0.5,
            )
        )
        supports.append(controller._last_vertical_support)
        targets.append(controller._last_vertical_collective_target)
        modes.append(controller._pitch_energy_brake_active)

    pitches = [output.target_pitch_rad for output in outputs]
    assert (
        SPAWN_PITCH
        - controller.config.pre_cross_brake_slew_rad_s
        * controller.config.control_period_s
        <= pitches[0]
        <= SPAWN_PITCH
    )
    maximum_nose_up_reversal = _maximum_advance_to_brake_reversal(pitches)
    # Reject a material advance-to-brake trajectory reversal (the F165
    # launch reversal was about 0.15 rad).  A monotonic brake directly from
    # spawn is safe and is deliberately not classified as a reversal.
    assert maximum_nose_up_reversal < math.radians(0.5)
    assert not any(modes)
    transfer = [index for index in range(30) if 0.40 <= 0.033 * index <= 0.70]
    assert all(targets[index] >= supports[index] for index in transfer)
    assert all(outputs[index].thrust >= supports[index] for index in transfer)


def test_f167_recorded_f166_launch_keeps_visual_primary_collective():
    # F166 run 20260802T025041Z-visual-course-c9059224.  These selected public
    # outer observations and command-boundary vertical accelerations reproduce
    # the launch transient that drove target collective below support from
    # t=.594--1.172 and wire collective below support from .704--1.172.  IMU
    # damping is supporting only: it cannot dwarf the outer path correction,
    # and the carried wire must not reproduce the recorded undershoot.
    rows = (
        # t, frame, outer center/span/conf, rates, pitch, world-up accel
        (0.000, 2441820, (-0.006250, -0.027778), (0.125000, 0.219444), 0.754608, (0.000072, 0.000140, 0.000007), -0.310159, 0.003349),
        (0.063, 2441822, (-0.006250, -0.033333), (0.125000, 0.225000), 0.813381, (0.000754, -0.001062, -0.000044), -0.310459, -0.344913),
        (0.141, 2441825, (-0.003125, -0.033333), (0.126563, 0.225000), 0.818499, (-0.003271, -0.006114, -0.013719), -0.310515, 0.729606),
        (0.204, 2441827, (-0.003125, -0.033333), (0.128125, 0.222222), 0.836780, (-0.019868, -0.000353, -0.005794), -0.310728, 1.381741),
        (0.266, 2441828, (-0.003125, -0.033333), (0.126563, 0.225000), 0.843204, (0.005276, 0.000135, -0.003219), -0.310754, 1.472685),
        (0.329, 2441830, (0.000000, -0.033333), (0.129688, 0.225000), 0.842756, (-0.012464, 0.003328, -0.004710), -0.310619, 1.466791),
        (0.391, 2441832, (0.000000, -0.027778), (0.129688, 0.225000), 0.845212, (0.000779, 0.006823, -0.004866), -0.310300, 1.442861),
        (0.454, 2441834, (-0.003125, -0.027778), (0.128125, 0.227778), 0.853139, (-0.001806, 0.010564, -0.002752), -0.309744, 1.403646),
        (0.532, 2441836, (0.000000, -0.022222), (0.129688, 0.233333), 0.856925, (-0.001484, 0.015140, -0.000950), -0.308796, 1.179828),
        (0.594, 2441838, (-0.003125, -0.022222), (0.131250, 0.230556), 0.863858, (0.004699, 0.019275, -0.000567), -0.307551, 0.747757),
        (0.657, 2441840, (-0.003125, -0.016667), (0.135938, 0.236111), 0.868841, (0.001406, 0.023894, -0.000733), -0.306197, -0.375087),
        (0.704, 2441841, (-0.003125, -0.016667), (0.135938, 0.238889), 0.875468, (0.008070, 0.026009, 0.000331), -0.305320, -1.272684),
        (0.735, 2441842, (-0.003125, -0.016667), (0.135938, 0.241667), 0.881745, (0.004194, 0.029584, 0.001532), -0.304344, -2.022859),
        (0.797, 2441844, (-0.003125, -0.011111), (0.137500, 0.247222), 0.887423, (0.006332, 0.033966, 0.002095), -0.302324, -2.345382),
        (0.860, 2441846, (-0.003125, 0.027778), (0.140625, 0.211111), 0.753603, (0.001345, 0.035609, 0.000361), -0.300343, -2.415678),
        (0.922, 2441848, (-0.003125, 0.016667), (0.140625, 0.216667), 0.724308, (0.001373, 0.014665, -0.000856), -0.298652, -2.187906),
        (0.985, 2441850, (0.000000, -0.016667), (0.145313, 0.250000), 0.861734, (-0.001696, 0.020933, -0.001834), -0.297656, -1.682436),
        (1.047, 2441852, (-0.003125, -0.022222), (0.148438, 0.255556), 0.913310, (-0.000750, 0.028621, -0.002061), -0.295959, -1.437391),
        (1.110, 2441853, (-0.003125, -0.027778), (0.150000, 0.261111), 0.927675, (-0.004719, 0.022240, -0.003624), -0.294375, -1.287698),
        (1.172, 2441855, (0.000000, -0.038889), (0.154687, 0.269444), 0.952243, (0.003849, 0.015363, -0.000745), -0.293186, -1.078695),
    )

    def track(row):
        return _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            confidence=row[4],
        )

    controller = CleanCourseController(
        _config(launch_boost_duration_s=LAUNCH_BOOST_DURATION_S)
    )
    samples = []
    pitches = []
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        update = _update([track(row)], frame_id=row[1])
        if index == 0:
            controller.initialize(
                update,
                gate_index=0,
                fallback_center_norm=row[2],
                fallback_apparent_scale=math.sqrt(row[3][0] * row[3][1]),
                now_s=now,
            )
        else:
            controller.observe(update, now_s=now, body_rates=row[5])
        output = _command(
            controller,
            now,
            pitch=row[6],
            a_up=row[7],
        )
        pitches.append(output.target_pitch_rad)
        samples.append(
            (
                row[0],
                controller._last_vertical_path_error,
                controller._last_vertical_visual_delta,
                controller._last_vertical_imu_delta,
                controller._last_vertical_imu_raw_delta,
                controller._last_vertical_motion_delta,
                controller._last_vertical_support,
                controller._last_vertical_collective_target,
                output.thrust,
                controller._last_launch_collective_delta,
            )
        )

    for (
        elapsed,
        path,
        visual,
        imu,
        imu_raw,
        optical_motion,
        support,
        target,
        wire,
        launch,
    ) in samples:
        # IMU damping remains present as a bounded innovation around optical
        # image motion; it is neither deleted on every fresh frame nor added
        # as a second full copy of the same launch motion.
        innovation_bound = (
            controller.config.vertical_imu_max_opposition_fraction
            * max(abs(visual), abs(optical_motion), 1e-6)
        )
        assert abs(imu) <= innovation_bound + 1e-12
        if abs(imu_raw) > innovation_bound + abs(optical_motion):
            assert abs(imu) < abs(imu_raw)
        assert target == pytest.approx(
            support + visual + imu + launch, abs=1e-12
        )
        assert controller.config.min_thrust <= wire <= controller.config.max_thrust
    assert max(
        abs(right[5] - left[5])
        for left, right in zip(samples, samples[1:])
    ) < 0.015
    # The original wire was below support continuously from .704 to 1.172.
    # With one visual-primary owner the same recorded-state counterfactual
    # returns to the previously smooth F163 support envelope instead of the
    # F166 ~-0.018 collective deficit.
    late = [sample for sample in samples if 0.704 <= sample[0] <= 1.172]
    assert min(
        wire - support
        for *_, support, _target, wire, _launch in late
    ) >= -0.0071
    maximum_nose_up_reversal = _maximum_advance_to_brake_reversal(pitches)
    assert maximum_nose_up_reversal < math.radians(0.5)
    assert sum(
        abs(right - left) for left, right in zip(pitches, pitches[1:])
    ) < 0.10


def test_f168_newest_f167_launch_replay_has_one_bumpless_vertical_owner():
    # F167 duplicate 20260802T035806Z-visual-course-b47c8d21, trace SHA-256
    # B4F9C9610D53D44A86749BBC1E6BDFC397DFAA5218A61117FAC2FF107D53BA20.
    # The .609/.641 command ticks intentionally share tracker frame 18: a
    # command-rate replay must not invent another coherent-motion sample.
    rows = (
        # t, frame, outer center/span/conf, pitch, rates, aperture center/half, a_up
        (0.000, 1, (-0.006250, -0.033333), (0.125000, 0.222222), 0.846257, -0.310343, (-0.000049, 0.000126, 0.0), (-0.005176, -0.046216), (0.049398, 0.104840), 0.003350),
        (0.375, 12, (0.0, -0.022222), (0.128125, 0.225000), 0.849718, -0.313632, (0.003633, -0.010614, -0.000344), (0.002675, -0.006987), (0.053654, 0.102581), 1.416966),
        (0.469, 14, (0.0, -0.016667), (0.129688, 0.227778), 0.854410, -0.314665, (0.005709, -0.009120, 0.003094), (0.002152, -0.009474), (0.054492, 0.104931), 1.321874),
        (0.578, 17, (0.0, -0.005556), (0.131250, 0.233333), 0.860686, -0.315472, (-0.008664, -0.006671, -0.009546), (0.002335, 0.003248), (0.060591, 0.116197), 0.819822),
        (0.609, 18, (0.0, 0.0), (0.132812, 0.233333), 0.854119, -0.315693, (-0.002579, -0.005893, -0.011755), (0.000037, -0.001236), (0.065966, 0.118626), 0.606837),
        (0.641, 18, (0.0, 0.0), (0.132812, 0.233333), 0.854119, -0.315829, (0.003167, -0.004412, -0.005886), (0.000037, -0.001236), (0.065966, 0.118626), 0.335494),
        (0.672, 19, (0.0, 0.0), (0.134375, 0.236111), 0.854300, -0.315936, (-0.002832, -0.003565, -0.000695), (-0.000293, 0.003314), (0.067886, 0.121686), 0.033715),
        (0.703, 20, (0.0, 0.005556), (0.134375, 0.236111), 0.858551, -0.316052, (-0.003070, -0.002931, 0.000333), (0.003512, 0.006195), (0.067047, 0.124361), -0.071521),
        (0.734, 21, (0.0, 0.011111), (0.137500, 0.238889), 0.862748, -0.316088, (0.004421, -0.000299, -0.000702), (0.001454, 0.011676), (0.067310, 0.124435), -0.403316),
        (0.766, 22, (0.0, 0.011111), (0.137500, 0.244444), 0.879133, -0.316006, (0.004264, 0.003645, 0.000487), (0.003827, 0.012353), (0.064554, 0.124475), -0.705802),
        (0.797, 23, (0.0, 0.016667), (0.137500, 0.244444), 0.889103, -0.315841, (0.000540, 0.005251, 0.002399), (0.004198, 0.018659), (0.063788, 0.123007), -0.918585),
        (0.875, 25, (0.0, 0.022222), (0.142188, 0.247222), 0.895798, -0.315296, (0.006856, 0.008487, 0.005611), (0.006090, 0.031870), (0.066813, 0.120908), -1.376124),
        (0.969, 27, (0.0, 0.027778), (0.146875, 0.255556), 0.913806, -0.314330, (0.010915, 0.011837, 0.009216), (-0.000502, 0.026725), (0.076007, 0.131608), -1.996676),
        (1.063, 30, (-0.003125, 0.022222), (0.150000, 0.263889), 0.937356, -0.313174, (0.007135, 0.013733, 0.001391), (-0.001563, 0.025000), (0.078125, 0.138889), -1.805453),
        (1.156, 33, (0.0, 0.022222), (0.154687, 0.275000), 0.960827, -0.311814, (0.001546, 0.013752, -0.000936), (0.000028, 0.023683), (0.079236, 0.138670), -1.327557),
        (1.281, 37, (0.003125, 0.016667), (0.164062, 0.286111), 0.989677, -0.310143, (-0.000498, 0.016166, -0.002576), (-0.000410, 0.014893), (0.084366, 0.151004), -1.154699),
    )
    controller = CleanCourseController(
        _config(launch_boost_duration_s=LAUNCH_BOOST_DURATION_S)
    )
    samples = []
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        track = _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            confidence=row[4],
            aperture_center=row[7],
            aperture_half=row[8],
        )
        update = _update([track], frame_id=row[1])
        if index == 0:
            controller.initialize(
                update,
                gate_index=0,
                fallback_center_norm=row[2],
                fallback_apparent_scale=math.sqrt(row[3][0] * row[3][1]),
                now_s=now,
            )
        else:
            controller.observe(update, now_s=now, body_rates=row[6])
        output = _command(
            controller,
            now,
            pitch=row[5],
            a_up=row[9],
        )
        samples.append(
            {
                "t": row[0],
                "serial": controller._current_fresh_outer_y_observation_serial,
                "visual": controller._last_vertical_visual_delta,
                "motion": controller._last_vertical_motion_delta,
                "imu": controller._last_vertical_imu_delta,
                "imu_raw": controller._last_vertical_imu_raw_delta,
                "support": controller._last_vertical_support,
                "target": controller._last_vertical_collective_target,
                "wire": output.thrust,
                "pitch": output.target_pitch_rad,
            }
        )

    assert samples[4]["serial"] == samples[5]["serial"]
    assert any(abs(sample["imu"]) > 1e-6 for sample in samples)
    for sample in samples:
        bound = (
            controller.config.vertical_imu_max_opposition_fraction
            * max(
                abs(sample["visual"]),
                abs(sample["motion"]),
                1e-6,
            )
        )
        assert abs(sample["imu"]) <= bound + 1e-12
        if abs(sample["imu_raw"]) > bound + abs(sample["motion"]):
            assert abs(sample["imu"]) < abs(sample["imu_raw"])
    crossing = [sample for sample in samples if 0.609 <= sample["t"] <= 0.797]
    assert max(
        abs(right["target"] - left["target"])
        for left, right in zip(crossing, crossing[1:])
    ) < 0.005
    assert min(
        sample["wire"] - sample["support"] for sample in samples
    ) >= -0.0073
    recovery = [sample for sample in samples if sample["t"] >= 0.969]
    assert max(
        left["wire"] - right["wire"]
        for left, right in zip(recovery, recovery[1:])
    ) < 0.0002
    assert _maximum_advance_to_brake_reversal(
        [sample["pitch"] for sample in samples]
    ) < math.radians(0.5)


def test_f170_failed_f168_topology_jump_cannot_cancel_launch_floor():
    # Failed fixed-hash F168 rerun
    # 20260802T052420Z-visual-course-201e7d06, trace SHA-256
    # A41993108AEB7FB6EC55C08FD0FB0B75E53C1A0142A8F063DDD65F5E0D42569F.
    # Frame 2718435 changes detector topology in one observation: the accepted
    # outer height collapses .225 -> .161 and y jumps -.028 -> +.039.  The
    # resulting +.10..+.16/s image rate used to multiply the only launch boost
    # to zero.  Replay only public tracker/attitude/trust observations and
    # require the time-owned minimum energy trajectory to survive intact.
    rows = (
        # t, frame, outer center/span/conf, pitch, rates,
        # aperture center/half, accel trust, horizontal force
        (0.000, 2718426, (-0.006250, -0.033333), (0.125000, 0.225000), 0.845915, -0.309763, (0.000072, 0.000330, 0.000006), (-0.004214, -0.025668), (0.052550, 0.101354), 1.000000, 0.009352),
        (0.015, 2718427, (-0.006250, -0.033333), (0.125000, 0.225000), 0.845446, -0.309898, (-0.000760, -0.009954, -0.000026), (-0.010202, -0.033652), (0.049039, 0.087568), 0.000000, 0.212781),
        (0.046, 2718428, (-0.006250, -0.027778), (0.125000, 0.222222), 0.846411, -0.309979, (0.000737, -0.000817, -0.000068), (-0.005620, -0.028103), (0.051948, 0.100912), 0.054031, 2.061369),
        (0.078, 2718429, (-0.003125, -0.033333), (0.126563, 0.225000), 0.848341, -0.309937, (-0.028696, 0.001311, -0.010752), (-0.004242, -0.029758), (0.053192, 0.103113), 0.000000, 2.963303),
        (0.109, 2718430, (-0.003125, -0.033333), (0.126563, 0.225000), 0.848615, -0.310152, (-0.009553, -0.010369, -0.020648), (-0.003522, -0.025944), (0.051323, 0.099632), 0.000000, 3.357081),
        (0.140, 2718431, (-0.003125, -0.033333), (0.126563, 0.225000), 0.847496, -0.310552, (0.003498, -0.010299, -0.013188), (-0.003566, -0.028111), (0.054481, 0.104906), 0.000000, 3.486620),
        (0.171, 2718432, (-0.003125, -0.027778), (0.126563, 0.222222), 0.845762, -0.310725, (-0.012010, -0.007878, -0.008202), (-0.005217, -0.026266), (0.049769, 0.092402), 0.000000, 3.522960),
        (0.234, 2718434, (-0.003125, -0.027778), (0.125000, 0.225000), 0.840845, -0.311437, (0.005426, -0.011372, -0.007436), (-0.001282, -0.023684), (0.054329, 0.106842), 0.000000, 3.512852),
        (0.296, 2718435, (-0.003125, 0.038889), (0.123438, 0.161111), 0.690865, -0.312025, (-0.015163, -0.010201, -0.007069), (0.025321, 0.004375), (0.041995, 0.016275), 0.000000, 3.353774),
        (0.328, 2718436, (-0.003125, 0.050000), (0.121875, 0.152778), 0.609155, -0.312375, (0.005129, -0.009239, -0.001593), None, None, 0.000000, 3.243124),
        (0.359, 2718437, (-0.003125, 0.050000), (0.126563, 0.155556), 0.578422, -0.312654, (0.020744, -0.007913, 0.010309), None, None, 0.000000, 3.150399),
        (0.390, 2718438, (-0.003125, 0.050000), (0.128125, 0.152778), 0.561129, -0.312902, (0.003811, -0.009764, 0.016285), None, None, 0.000000, 3.055710),
        (0.421, 2718439, (-0.003125, -0.016667), (0.128125, 0.227778), 0.661444, -0.313299, (0.000460, -0.012253, 0.011419), (-0.001311, -0.016008), (0.061358, 0.113241), 0.000000, 2.962143),
        (0.453, 2718440, (-0.003125, -0.011111), (0.128125, 0.233333), 0.759437, -0.313700, (0.013161, -0.010138, 0.003611), (-0.002220, -0.014878), (0.060158, 0.108712), 0.000000, 2.858469),
        (0.484, 2718441, (-0.003125, -0.016667), (0.128125, 0.230556), 0.811026, -0.313884, (0.008940, -0.008124, 0.001899), (-0.002938, -0.016687), (0.067090, 0.119967), 0.000000, 2.844363),
        (0.515, 2718442, (-0.003125, -0.016667), (0.129688, 0.233333), 0.838942, -0.314113, (-0.007007, -0.006263, 0.000998), (-0.002334, -0.014656), (0.066923, 0.116953), 0.000000, 2.853920),
        (0.578, 2718444, (-0.006250, -0.011111), (0.131250, 0.241667), 0.857248, -0.314498, (0.007343, -0.006818, -0.002287), (-0.002832, -0.017178), (0.054563, 0.119481), 0.000000, 2.881183),
        (0.609, 2718445, (-0.003125, -0.016667), (0.134375, 0.236111), 0.865003, -0.314857, (-0.012442, -0.014637, -0.008117), (-0.005900, -0.007915), (0.048060, 0.102577), 0.000000, 2.877826),
    )
    controller = CleanCourseController(
        _config(launch_boost_duration_s=LAUNCH_BOOST_DURATION_S)
    )
    samples = []
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        track = _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            confidence=row[4],
            aperture_center=row[7],
            aperture_half=row[8],
        )
        update = _update([track], frame_id=row[1])
        if index == 0:
            controller.initialize(
                update,
                gate_index=0,
                fallback_center_norm=row[2],
                fallback_apparent_scale=math.sqrt(row[3][0] * row[3][1]),
                now_s=now,
            )
        else:
            controller.observe(update, now_s=now, body_rates=row[6])
        output = _command(
            controller,
            now,
            pitch=row[5],
            fh=row[10],
            accel_trust=row[9],
        )
        samples.append(
            {
                "t": row[0],
                "target": controller._last_vertical_collective_target,
                "wire": output.thrust,
                "support": controller._last_vertical_support,
                "imu": controller._last_vertical_imu_delta,
                "observed_rate": (
                    controller._last_vertical_observed_rate_norm_s
                ),
            }
        )

    glitch = [sample for sample in samples if 0.296 <= sample["t"] <= 0.390]
    assert glitch[0]["observed_rate"] > 0.09
    assert max(sample["observed_rate"] for sample in glitch) > 0.15
    assert all(sample["imu"] == pytest.approx(0.0) for sample in glitch)
    assert all(
        sample["target"] >= LAUNCH_BOOST_THRUST - 1e-12
        for sample in glitch
    )
    for sample in samples:
        if sample["t"] <= LAUNCH_BOOST_DURATION_S:
            floor = LAUNCH_BOOST_THRUST
        else:
            phase = min(
                1.0,
                (sample["t"] - LAUNCH_BOOST_DURATION_S)
                / controller.config.launch_collective_transfer_s,
            )
            progress = phase * phase * (3.0 - 2.0 * phase)
            floor = LAUNCH_BOOST_THRUST + (
                sample["support"] - LAUNCH_BOOST_THRUST
            ) * progress
        assert sample["target"] >= floor - 1e-12
        assert sample["wire"] >= floor - 1e-12
        assert controller.config.min_thrust <= sample["wire"] <= (
            controller.config.max_thrust
        )


def test_f170_zero_trust_imu_cannot_cancel_fresh_gate0_climb():
    # Failed fixed-hash F168 Gate-0 approach.  Fresh outer y worsens across
    # these exact frames while pitch compensation makes the high-gate miss
    # increasingly negative.  The recorded estimator trust is zero; preserve
    # the recorded stale sink estimate as an adversarial supporting input and
    # prove that it cannot subtract from the fresh visual climb trajectory.
    rows = (
        (1.937, 2718484, (0.009375, -0.033333), (0.234375, 0.402778), -0.369783, (-0.070939, -0.219892, -0.052894)),
        (1.968, 2718485, (0.025000, -0.027778), (0.254688, 0.413889), -0.378288, (-0.072263, -0.282841, -0.055783)),
        (1.984, 2718486, (0.031250, -0.016667), (0.267188, 0.425000), -0.382728, (-0.070299, -0.328649, -0.058813)),
        (2.031, 2718487, (0.028125, 0.000000), (0.268750, 0.441667), -0.395596, (-0.065823, -0.199785, -0.069174)),
        (2.078, 2718489, (0.037500, 0.016667), (0.285938, 0.466667), -0.404113, (-0.076147, -0.213920, -0.075042)),
        (2.125, 2718490, (0.031250, 0.027778), (0.285938, 0.483333), -0.415256, (-0.083148, -0.166978, -0.092345)),
        (2.171, 2718491, (0.034375, 0.033333), (0.296875, 0.500000), -0.421710, (-0.099510, -0.141709, -0.104617)),
    )

    def track(row):
        return _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            confidence=0.98,
        )

    controller = CleanCourseController(_config())
    first = rows[0]
    controller.initialize(
        _update([track(first)], frame_id=first[1]),
        gate_index=0,
        fallback_center_norm=first[2],
        fallback_apparent_scale=math.sqrt(first[3][0] * first[3][1]),
        now_s=100.0 + first[0],
    )
    controller._vz_est_m_s = 0.40
    path_errors = []
    visual_deltas = []
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        if index:
            controller.observe(
                _update([track(row)], frame_id=row[1]),
                now_s=now,
                body_rates=row[5],
            )
        output = _command(
            controller,
            now,
            pitch=row[4],
            a_up=8.0,
            accel_trust=0.0,
        )
        path_errors.append(controller._last_vertical_path_error)
        visual_deltas.append(controller._last_vertical_visual_delta)
        assert controller._last_vertical_imu_raw_delta < 0.0
        assert controller._last_vertical_imu_trust == 0.0
        assert controller._last_vertical_imu_delta == pytest.approx(0.0)
        assert controller._last_vertical_collective_target == pytest.approx(
            controller._last_vertical_support
            + controller._last_vertical_visual_delta,
            abs=1e-12,
        )

    assert path_errors[-1] < path_errors[0]
    assert path_errors[-1] < -0.10
    assert visual_deltas[-1] > 0.015
    assert controller._last_vertical_collective_target == pytest.approx(
        controller._last_vertical_support
        + controller._last_vertical_visual_delta
    )
    assert output.thrust > controller._last_vertical_support


_F170_GATE1_LATENCY_ROWS = (
    # t, frame, outer x/y/w/h/confidence, optional aperture x/y/hx/hy,
    # body p/q/r, roll/pitch/yaw, clipping.  F170 trace SHA-256:
    # 5C1283A13340D3231738D3DE053E2A41A0529BE275AC1DFC6CBBF92F8E9612F.
    (3.172, 2846866, -.384375, .050000, .107812, .208333, .719242, -.387794, -.002138, .041599, .066950, .030366, -.002059, -.162959, .018815, -.457629, -.135081, 0),
    (3.203, 2846867, -.384375, .050000, .109375, .208333, .706344, None, None, None, None, .005362, -.003919, -.204062, .021943, -.457592, -.141039, 0),
    (3.234, 2846868, -.381250, .050000, .109375, .211111, .695037, None, None, None, None, -.075884, -.010264, -.262745, .024385, -.457711, -.150160, 0),
    (3.265, 2846869, -.375000, .050000, .109375, .213889, .693108, None, None, None, None, -.113681, -.008137, -.349037, .026265, -.457745, -.162665, 0),
    (3.297, 2846870, -.362500, .050000, .110938, .213889, .692679, None, None, None, None, -.142053, -.008209, -.355832, .027596, -.457698, -.173713, 0),
    (3.328, 2846871, -.353125, .050000, .110938, .216667, .689682, None, None, None, None, -.192543, -.010273, -.329879, .027310, -.457708, -.186853, 0),
    (3.359, 2846872, -.343750, .044444, .114063, .216667, .715680, -.353766, -.010008, .047087, .072925, -.199323, -.009542, -.321934, .025861, -.457758, -.199360, 0),
    (3.390, 2846873, -.334375, .038889, .114062, .219444, .735170, -.345133, -.013822, .045965, .073266, -.202698, -.008839, -.325744, .025060, -.457775, -.206894, 0),
    (3.422, 2846874, -.325000, .038889, .117188, .222222, .740139, -.334848, -.018400, .047345, .070863, -.320763, -.007712, -.331373, .021965, -.457790, -.219667, 0),
    (3.453, 2846875, -.315625, .033333, .115625, .225000, .742123, -.323239, -.022839, .045798, .072425, -.422577, -.009352, -.330899, .013912, -.457882, -.232509, 0),
    (3.484, 2846876, -.300000, .027778, .120313, .227778, .746515, -.311122, -.028167, .041763, .073731, -.446989, -.008372, -.329416, .006305, -.458047, -.242717, 0),
    (3.515, 2846877, -.290625, .022222, .120313, .230556, .749541, -.301540, -.036068, .046475, .073053, -.518645, -.004907, -.329071, -.004536, -.458256, -.255454, 0),
    (3.547, 2846878, -.278125, .016667, .120312, .233333, .751266, -.289563, -.043151, .047092, .073075, -.589112, -.003893, -.329581, -.018576, -.458537, -.268208, 0),
    (3.578, 2846879, -.265625, .005556, .121875, .238889, .754126, -.277966, -.053632, .044482, .076130, -.602403, -.002394, -.329743, -.030691, -.458853, -.278418, 0),
    (3.609, 2846880, -.250000, -.005556, .125000, .241667, .756261, -.260497, -.064712, .046854, .067264, -.609553, .003282, -.329631, -.046150, -.459249, -.291181, 0),
    (3.640, 2846880, -.250000, -.005556, .125000, .241667, .756261, -.260497, -.064712, .046854, .067264, -.610714, .006943, -.329551, -.058574, -.459570, -.301389, 0),
    (3.672, 2846882, -.225000, -.016667, .126562, .250000, .768319, -.235144, -.079418, .046390, .069899, -.552508, .010235, -.329560, -.073165, -.460011, -.314155, 0),
    (3.703, 2846882, -.225000, -.016667, .126562, .250000, .768319, -.235144, -.079418, .046390, .069899, -.529524, .013171, -.329594, -.086058, -.460506, -.326931, 0),
    (3.734, 2846883, -.212500, -.027778, .126563, .247222, .773822, -.226673, -.082517, .044207, .074871, -.474609, .014865, -.329596, -.095325, -.460941, -.337150, 0),
    (3.765, 2846884, -.200000, -.038889, .129688, .247222, .764577, -.214289, -.081582, .039794, .070267, -.453363, .016130, -.329588, -.103552, -.461415, -.347364, 0),
    (3.812, 2846885, -.187500, -.050000, .129688, .247222, .757896, -.202850, -.091523, .045485, .074200, -.423321, .017639, -.329583, -.113031, -.462066, -.360145, 0),
    (3.843, 2846886, -.178125, -.100000, .131250, .200000, .705247, -.189556, -.099490, .045046, .081743, -.394719, .018774, -.329585, -.119825, -.462620, -.370379, 0),
    (3.875, 2846887, -.168750, -.111111, .132812, .197222, .677991, -.179781, -.100359, .041622, .080052, -.320195, .019693, -.329587, -.126293, -.463354, -.383162, 0),
    (3.906, 2846888, -.159375, -.105556, .134375, .205556, .638138, None, None, None, None, -.289586, .019850, -.329586, -.130868, -.464134, -.395947, 0),
    (3.937, 2846889, -.146875, -.105556, .135938, .213889, .627317, None, None, None, None, -.260649, .019563, -.329586, -.133889, -.464804, -.406177, 0),
    (3.968, 2846890, -.137500, -.116667, .139062, .208333, .611879, None, None, None, None, -.226627, .020522, -.329586, -.136376, -.465653, -.418972, 0),
    (4.000, 2846891, -.128125, -.111111, .142188, .216667, .604567, None, None, None, None, -.208162, .020562, -.319440, -.138192, -.466488, -.431617, 0),
    (4.031, 2846892, -.121875, -.105556, .143750, .230556, .601288, None, None, None, None, -.199358, .020432, -.297285, -.139172, -.466946, -.438741, 0),
    (4.062, 2846893, -.115625, -.083333, .145312, .255556, .615159, None, None, None, None, -.204869, .020651, -.258432, -.141965, -.467660, -.451288, 0),
    (4.093, 2846894, -.109375, -.100000, .146875, .247222, .611839, None, None, None, None, -.202687, .019458, -.252290, -.144572, -.468238, -.461225, 0),
    (4.125, 2846895, -.103125, -.050000, .150000, .294444, .664349, None, None, None, None, -.185900, .019378, -.243722, -.146054, -.468587, -.467016, 0),
    (4.156, 2846896, -.096875, -.050000, .153125, .300000, .700167, None, None, None, None, -.168217, .021213, -.222334, -.147465, -.469036, -.475859, 0),
    (4.187, 2846897, -.093750, -.050000, .156250, .302778, .717604, None, None, None, None, -.208880, .020611, -.229069, -.150382, -.469476, -.484741, 0),
    (4.218, 2846898, -.087500, -.050000, .159375, .302778, .727847, None, None, None, None, -.191266, .020012, -.220441, -.152897, -.469875, -.491751, 0),
    (4.250, 2846899, -.081250, -.044444, .160937, .311111, .736359, None, None, None, None, -.129406, .022288, -.193239, -.154391, -.470210, -.499700, 0),
    (4.281, 2846899, -.081250, -.044444, .160937, .311111, .736359, None, None, None, None, -.132413, .021089, -.178823, -.155373, -.470398, -.505430, 0),
    (4.312, 2846900, -.078125, -.044444, .164062, .313889, .785385, -.094048, -.100495, .054877, .091081, -.123780, .017127, -.165999, -.157035, -.470696, -.512092, 0),
    (4.343, 2846901, -.075000, -.038889, .167187, .316667, .812826, None, None, None, None, -.108598, .020162, -.164612, -.157974, -.470956, -.518523, 0),
    (4.375, 2846902, -.071875, -.038889, .170312, .319444, .828040, None, None, None, None, -.114470, .022804, -.168789, -.158676, -.471081, -.523776, 0),
    (4.406, 2846903, -.068750, -.033333, .173437, .322222, .838719, -.087138, -.092771, .058380, .096605, -.119469, .023340, -.171947, -.159603, -.471201, -.529170, 0),
    (4.437, 2846904, -.065625, -.027778, .176562, .327778, .848781, -.082682, -.089152, .058273, .104484, -.099038, .026102, -.168223, -.160207, -.471289, -.535840, 0),
    (4.468, 2846905, -.062500, -.027778, .182812, .330556, .856674, -.084961, -.081074, .056870, .102346, -.103460, .027778, -.168281, -.160559, -.471282, -.541157, 0),
    (4.515, 2846907, -.056250, -.016667, .187500, .338889, .874924, None, None, None, None, -.123478, .029277, -.168343, -.162121, -.471224, -.550418, 0),
    (4.547, 2846907, -.056250, -.016667, .187500, .338889, .874924, None, None, None, None, -.118348, .030396, -.171214, -.163044, -.471175, -.555808, 0),
    (4.578, 2846908, -.053125, -.011111, .192187, .344444, .886584, None, None, None, None, -.091150, .034459, -.167001, -.163606, -.471001, -.562578, 0),
    (4.609, 2846909, -.050000, -.005556, .196875, .344444, .898609, None, None, None, None, -.078492, .034712, -.143991, -.163531, -.470672, -.568719, 0),
    (4.656, 2846911, -.043750, .005556, .206250, .355556, .907547, None, None, None, None, -.073561, .034445, -.111507, -.164187, -.470103, -.574681, 0),
    (4.687, 2846912, -.043750, .011111, .210938, .361111, .910553, None, None, None, None, -.060974, .034902, -.101669, -.164592, -.469499, -.578873, 0),
    (4.718, 2846913, -.040625, .016667, .214063, .363889, .913773, None, None, None, None, -.037891, .034869, -.100855, -.164253, -.468875, -.582977, 0),
    (4.750, 2846914, -.040625, .027778, .220313, .366667, .917725, None, None, None, None, -.033115, .034471, -.096231, -.163751, -.468378, -.586166, 0),
    (4.797, 2846915, -.037500, .033333, .226562, .375000, .921545, None, None, None, None, -.030033, .036795, -.085880, -.163154, -.467507, -.590547, 0),
    (4.828, 2846916, -.037500, .044444, .231250, .380556, .926902, None, None, None, None, -.019865, .037443, -.086005, -.162412, -.466706, -.594060, 0),
    (4.859, 2846917, -.037500, .055556, .237500, .386111, .929785, None, None, None, None, -.004386, .037073, -.087165, -.161064, -.465919, -.597634, 0),
    (4.906, 2846918, -.034375, .072222, .242188, .397222, .933297, None, None, None, None, -.014927, .000812, -.078318, -.159444, -.465488, -.602253, 0),
    (4.953, 2846920, -.031250, .116667, .254688, .425000, .941128, None, None, None, None, .019692, .012422, -.072986, -.157694, -.465961, -.606289, 0),
    (5.000, 2846921, -.031250, .144444, .262500, .441667, .947718, None, None, None, None, .019773, .006882, -.070911, -.154785, -.465924, -.610253, 0),
    (5.047, 2846923, -.028125, .194444, .276562, .475000, .956095, None, None, None, None, .031948, .007786, -.068574, -.152326, -.466111, -.613508, 0),
    (5.093, 2846924, -.028125, .227778, .284375, .494444, .962223, None, None, None, None, .044744, .008193, -.066367, -.148732, -.466179, -.617185, 0),
    (5.140, 2846925, -.025000, .261111, .292188, .519444, .969194, None, None, None, None, .047294, .017166, -.065285, -.145361, -.466171, -.620301, 0),
    (5.187, 2846927, -.021875, .333333, .309375, .561111, .980743, None, None, None, None, .057038, .047595, -.059284, -.140713, -.464745, -.624397, 0),
    (5.234, 2846928, -.012500, .377778, .323437, .594444, .990380, None, None, None, None, .057581, .075573, -.051735, -.136319, -.462217, -.627867, 0),
    (5.281, 2846930, .009375, .405556, .360937, .594444, .993656, None, None, None, None, .074714, .095932, -.039716, -.132490, -.458909, -.630475, 8),
    (5.328, 2846931, .018750, .411111, .379688, .588889, .995320, None, None, None, None, .122055, .122143, -.012071, -.125813, -.452946, -.632920, 8),
    (5.359, 2846932, .031250, .416667, .396875, .583333, .994528, None, None, None, None, .125095, .119864, -.001614, -.123017, -.450442, -.633397, 8),
    (5.406, 2846934, .071875, .433333, .442187, .566667, .975642, None, None, None, None, .157327, .106218, .045379, -.115722, -.444273, -.632562, 8),
    (5.437, 2846934, .071875, .433333, .442187, .566667, .975642, None, None, None, None, .169492, .106943, .059094, -.111667, -.441149, -.631261, 8),
    (5.468, 2846935, .100000, .438889, .465625, .558333, .950231, None, None, None, None, .201986, .101905, .113417, -.106396, -.437210, -.628191, 8),
    (5.515, 2846936, .134375, .450000, .495313, .547222, .919356, None, None, None, None, .211726, .098718, .144831, -.102199, -.434064, -.624355, 8),
)


def _replay_f170_gate1_latency_rows():
    controller = CleanCourseController(_config())
    samples = {}
    outputs = []
    base_s = 100.0
    for index, row in enumerate(_F170_GATE1_LATENCY_ROWS):
        aperture_center = (
            None if row[7] is None else (row[7], row[8])
        )
        aperture_half = (
            None if row[9] is None else (row[9], row[10])
        )
        track = _f163_trace_track(
            outer_center=(row[2], row[3]),
            outer_span=(row[4], row[5]),
            confidence=row[6],
            aperture_center=aperture_center,
            aperture_half=aperture_half,
            clipping=FrameEdge(row[17]),
        )
        now = base_s + row[0]
        update = _update([track], frame_id=row[1])
        if index == 0:
            controller.initialize(
                update,
                gate_index=1,
                fallback_center_norm=(row[2], row[3]),
                fallback_apparent_scale=math.sqrt(row[4] * row[5]),
                now_s=now,
            )
        else:
            controller.observe(
                update,
                now_s=now,
                body_rates=(row[11], row[12], row[13]),
            )
        output = _command(
            controller,
            now,
            roll=row[14],
            pitch=row[15],
            yaw=row[16],
            accel_trust=0.0,
        )
        outputs.append(output)
        vertical_motion = controller._last_vertical_motion
        lateral_motion = controller._last_lateral_motion
        censor_bound = controller.current.vertical_censor_bound
        expected_censored_path = None
        if censor_bound is not None and FrameEdge(row[17]) & (
            FrameEdge.TOP | FrameEdge.BOTTOM
        ):
            if FrameEdge(row[17]) & FrameEdge.BOTTOM:
                raw_bound = max(float(censor_bound), 0.0)
                expected_censored_path = max(
                    controller._compensated_ey(raw_bound, row[15]), 0.0
                )
            else:
                raw_bound = min(float(censor_bound), 0.0)
                expected_censored_path = min(
                    controller._compensated_ey(raw_bound, row[15]), 0.0
                )
        samples[row[0]] = {
            "serial": controller._current_fresh_outer_y_observation_serial,
            "direction_sign": controller._vertical_direction_sign,
            "direction_streak": controller._vertical_direction_streak,
            "direction_supported": controller._vertical_direction_supported,
            "direction_source": controller._vertical_direction_source,
            "vertical_motion": vertical_motion,
            "vertical_visual": controller._last_vertical_visual_delta,
            "vertical_target": controller._last_vertical_collective_target,
            "support": controller._last_vertical_support,
            "wire": output.thrust,
            "path_error": controller._last_vertical_path_error,
            "path_rate": controller._last_vertical_path_rate,
            "expected_censored_path": expected_censored_path,
            "lateral_motion": lateral_motion,
            "lateral_direction_sign": controller._last_lateral_direction_sign,
            "lateral_reversal_sign": controller._last_lateral_reversal_sign,
            "lateral_reference": controller._lateral_intercept_reference_x,
            "output": output,
        }
    return controller, samples, outputs


def test_f171_f170_gate1_motion_reverses_collective_before_censorship():
    controller, samples, outputs = _replay_f170_gate1_latency_rows()

    assert samples[4.218]["direction_sign"] == -1
    assert samples[4.218]["direction_supported"]
    assert samples[4.250]["direction_streak"] == 1
    assert not samples[4.250]["direction_supported"]
    # A command tick that republishes frame 2846899 is not another optical
    # vote; the distinct-frame evidence clock stays at one.
    assert samples[4.281]["serial"] == samples[4.250]["serial"]
    assert samples[4.281]["direction_streak"] == 1
    assert samples[4.312]["direction_streak"] == 2
    assert not samples[4.312]["direction_supported"]

    reversal = samples[4.343]
    motion = reversal["vertical_motion"]
    assert reversal["direction_sign"] == 1
    assert reversal["direction_streak"] == 3
    assert reversal["direction_supported"]
    assert reversal["direction_source"] == "coherent_optical_motion"
    assert motion.bearing_error < 0.0
    assert motion.fallback_intercept_error < 0.0
    assert motion.optical_intercept_error > 0.0
    assert motion.control_authority == pytest.approx(0.0)
    assert reversal["vertical_visual"] < 0.0
    assert reversal["vertical_target"] < reversal["support"]
    assert reversal["wire"] > reversal["support"]

    filtered_reversal = samples[4.687]
    assert filtered_reversal["wire"] < filtered_reversal["support"]
    assert filtered_reversal["wire"] > 0.25
    # Scheduling/evidence costs 93 ms and the unchanged bounded collective
    # path costs 344 ms.  Even the measured 220 ms worst plant delay leaves a
    # 374 ms response window before the first BOTTOM frame; changing cadence
    # or command bounds is therefore not the minimal causal candidate.
    assert 4.343 - 4.250 == pytest.approx(0.093, abs=1e-9)
    assert 4.687 - 4.343 == pytest.approx(0.344, abs=1e-9)
    assert 4.687 + 0.220 < 5.281
    assert all(
        controller.config.min_thrust <= output.thrust <= controller.config.max_thrust
        for output in outputs
    )


def test_f171_f170_fresh_bottom_and_projected_lateral_reversal_own_immediately():
    controller, samples, outputs = _replay_f170_gate1_latency_rows()

    first_bottom = samples[5.281]
    assert first_bottom["direction_sign"] == 1
    assert first_bottom["direction_supported"]
    assert first_bottom["direction_source"] == "bottom_censor"
    assert first_bottom["path_rate"] == pytest.approx(0.0)
    assert first_bottom["vertical_motion"].physical_rate_norm_s == pytest.approx(
        0.0
    )
    assert first_bottom["path_error"] == pytest.approx(
        first_bottom["expected_censored_path"], abs=1e-12
    )

    before = samples[5.359]
    assert before["lateral_motion"].optical_intercept_error < 0.0
    assert before["output"].target_roll_rad < 0.0
    reversal = samples[5.406]
    lateral = reversal["lateral_motion"]
    assert lateral.bearing_error > 0.0
    assert lateral.fallback_intercept_error > 0.0
    assert lateral.optical_intercept_error > 0.0
    assert reversal["lateral_direction_sign"] == 1
    assert reversal["lateral_reversal_sign"] == 1
    assert reversal["output"].yaw_rate_rad_s > 0.0
    assert reversal["lateral_reference"] >= 0.0
    assert reversal["output"].target_roll_rad >= 0.0
    assert all(
        abs(output.yaw_rate_rad_s) <= controller.config.max_yaw_rate_rad_s
        and abs(output.target_roll_rad) <= controller.config.max_target_roll_rad
        for output in outputs
    )


def test_f171_motion_led_direction_rejects_nonfinite_inputs():
    controller = CleanCourseController(_config())
    motion = SimpleNamespace(
        closure_rate_s=0.5,
        bearing_error=0.1,
        fallback_intercept_error=0.2,
        optical_intercept_error=math.nan,
        physical_rate_norm_s=0.1,
    )
    assert controller._motion_led_passage_direction(motion) == (0, 0.0)


_F170_GATE0_RELEASE_ROWS = (
    # t, frame, outer center/span/confidence, aperture center/half, visible,
    # body rates, rpy, accel trust, horizontal specific force.
    (1.672, 2846821, (.003125, .05), (.196875, .3472222222), .9941040108, (.00461444194, .04917092577), (.1000803125, .1796844292), True, (-.0339148010, -.0616019681, -.0312297366), (-.0028487149, -.3373136619, -.0080112092), 0.0, 2.2156701905),
    (1.734, 2846823, (.00625, .05), (.2046875, .3611111111), .9969377066, (.00683486223, .04746985362), (.1034922939, .1881252572), True, (-.0428422990, -.1006881162, -.0403809439), (-.0043146423, -.3418052159, -.0101116389), .0179956131, 2.2307977392),
    (1.953, 2846830, (.03125, .0888888889), (.253125, .4305555556), .9857744747, (.02436292111, .07804796786), (.1213049743, .2241954499), True, (-.0631528425, -.2181077540, -.0779761712), (-.0126675894, -.3803371708, -.0241727579), 0.0, 2.6369388311),
    (2.093, 2846834, (.05, .1111111111), (.2890625, .4861111111), .9848236400, (.04122883549, .08992437), (.1367655367, .2438551863), True, (-.0853715169, -.1569942027, -.0980242680), (-.0183370157, -.4070261612, -.0372106147), 0.0, 2.9075359388),
    (2.125, 2846835, (.053125, .1111111111), (.2953125, .5027777778), .9858755045, (.046372783, .0935012439), (.1416282727, .2560332117), True, (-.0902224111, -.1473847568, -.1018197651), (-.0199959178, -.4124014559, -.0409092023), 0.0, 2.9719491493),
    (2.172, 2846836, (.053125, .1166666667), (.3015625, .5222222222), .9888713162, (.05179165217, .0943702428), (.1472341698, .2632214591), True, (-.0849364626, -.1075569033, -.1074987914), (-.0222592831, -.4192705825, -.0471968969), 0.0, 3.0694015832),
    (2.218, 2846838, (.0625, .1166666667), (.325, .5611111111), .9900667192, (.06102559053, .0870109725), (.1541620128, .2832669833), True, (-.0946219790, -.0989382282, -.1033750738), (-.0240261854, -.4236832885, -.0519093765), 0.0, 3.1312770506),
    (2.562, 2846848, (.115625, .0777777778), (.5625, .9222222222), .9677524953, None, None, False, (-.0074444230, -.0401675790, -.0021202893), (-.0381653072, -.4496967106, -.0758109977), 0.0, 3.4990307857),
)


def _f170_gate0_release_track(row, *, track_id="recorded-gate0"):
    return _f163_trace_track(
        outer_center=row[2],
        outer_span=row[3],
        confidence=row[4],
        track_id=track_id,
        aperture_center=row[5],
        aperture_half=row[6],
    ) if row[7] else _track(
        track_id,
        row[2][0],
        row[2][1],
        scale=math.sqrt(row[3][0] * row[3][1]),
        confidence=row[4],
        visible=False,
    )


def _replay_f170_gate0_release():
    controller = CleanCourseController(_config())
    samples = {}
    outputs = []
    base_s = 100.0
    for index, row in enumerate(_F170_GATE0_RELEASE_ROWS):
        now = base_s + row[0]
        update = _update(
            [_f170_gate0_release_track(row)], frame_id=row[1]
        )
        if index == 0:
            controller.initialize(
                update,
                gate_index=0,
                fallback_center_norm=row[2],
                fallback_apparent_scale=math.sqrt(row[3][0] * row[3][1]),
                now_s=now,
            )
        else:
            controller.observe(update, now_s=now, body_rates=row[8])
        output = _command(
            controller,
            now,
            roll=row[9][0],
            pitch=row[9][1],
            yaw=row[9][2],
            accel_trust=row[10],
            fh=row[11],
        )
        outputs.append(output)
        admission = controller._last_commit_admission
        samples[row[0]] = {
            "state": controller.state,
            "output": output,
            "admission": admission,
        }
    return controller, samples, outputs


def test_f172_f170_revoked_commit_loss_stays_powered_until_current_proof():
    # F171 flight 20260802T073212Z-visual-course-d863a812 followed F170 almost
    # exactly through t=2.500, then a delayed release lease converted the first
    # engulfing-anchor rejection into exact zero at2.547.  Its last nominally
    # safe frame had only 0.66 px vertical reserve, fresh geometry had already
    # revoked COMMIT, and Gate0 received no credit before collision.  A revoked
    # unsafe tube cannot certify a later crossing.  Exact zero remains mandatory
    # for close loss while COMMIT itself still owns the current safety proof.
    controller, samples, outputs = _replay_f170_gate0_release()

    assert samples[1.953]["admission"].admissible
    assert samples[2.093]["state"] is CleanCourseState.COMMIT
    revoked = samples[2.218]
    assert revoked["state"] is CleanCourseState.TRACK
    assert revoked["admission"].status == "corridor-known/not-contained"
    assert revoked["admission"].y_tube > revoked["admission"].y_safe_half
    loss = samples[2.562]
    assert loss["state"] is not CleanCourseState.COAST_FOR_CREDIT
    assert loss["output"].thrust > 0.0
    assert sum(output.thrust == 0.0 for output in outputs) == 0
    assert controller.gate_index == 0


def test_f173_f172_successor_lineage_rejects_one_frame_fragment():
    # F172 flight 20260802T074154Z-visual-course-aac36a6b.  Track 000002
    # represented Gate 1 for 65 accepted frames before its tracker alias went
    # stale.  A one-frame sub-fragment 000003 then displaced it immediately;
    # the physical Gate 1 reappeared as compatible 000004 before credit, but
    # stale 000003 was promoted and excluded the real gate forever.  Replay
    # the advancing clean tracker tokens (the selected-target frame id froze)
    # and the exact credit-before-frame ordering through several Gate-1 ticks.
    rows = (
        # t, clean frame, body rates, visible id/center/span/confidence.
        (2.281, 2966030, (-.1089778781, -.0991894014, -.1074062436), (("000001", (.075, .1222222222), (.35625, .6138888889), .9893300582), ("000002", (-.35625, .05), (.0765625, .1666666667), .6296388358))),
        (2.516, 2966036, (-.0173739247, -.0344168551, -.0029082933), (("000001", (.084375, .0944444444), (.5203125, .825), .9586396671),)),
        (2.562, 2966038, (.2085980385, -.0287490420, .0270199069), (("000001", (.121875, .0666666667), (.56875, .9305555556), .9685039333),)),
        (2.844, 2966046, (.0528511203, -.0014319576, -.0977687239), (("000001", (.06875, 0.0), (.9328125, 1.0), .8229938167),)),
        (2.891, 2966048, (.0606091082, -.0018174283, -.1000110506), (("000001", (0.0, 0.0), (1.0, 1.0), .7916564097), ("000003", (-.890625, .5888888889), (.04375, .0861111111), .5937266667))),
        (2.937, 2966049, (.0113349272, -.0023141864, -.1456660479), (("000001", (-.00625, 0.0), (.9921875, 1.0), .7798526267),)),
        (2.984, 2966051, (-.0897522882, -.0017934821, -.3395731746), (("000005", (-.859375, 0.0), (.1390625, 1.0), .4), ("000004", (-.378125, .0555555556), (.0984375, .2), .7078429997))),
        (3.016, 2966052, (-.1064052076, -.0022058271, -.3616677820), (("000004", (-.36875, .0555555556), (.1, .2), .7079896933),)),
        (3.047, 2966053, (-.2217551560, -.0033812605, -.3347360193), (("000004", (-.35625, .05), (.1015625, .2055555556), .7103847606),)),
        (3.078, 2966054, (-.1912899703, -.0035115860, -.3208803236), (("000004", (-.346875, .05), (.1015625, .2055555556), .7118375524),)),
        (3.125, 2966055, (-.1959250808, -.0019390367, -.3254914879), (("000004", (-.3375, .0444444444), (.1015625, .2055555556), .7156401735),)),
        # A repeated clean token is not a second identity vote.
        (3.156, 2966055, (-.2730655760, -.0014776669, -.3309340774), (("000004", (-.3375, .0444444444), (.1015625, .2055555556), .7156401735),)),
        (3.187, 2966056, (-.2827597946, -.0033534847, -.3314221799), (("000004", (-.328125, .0388888889), (.103125, .2083333333), .7171011858),)),
        (3.219, 2966057, (-.2740092069, -.0018854670, -.3297076821), (("000004", (-.315625, .0388888889), (.1046875, .2111111111), .7142892365),)),
        (3.250, 2966058, (-.2577025980, .0033350207, -.2613406479), (("000004", (-.303125, .0333333333), (.1046875, .2111111111), .7130703579),)),
        (3.281, 2966059, (-.1838798315, .0156853266, -.0416888855), (("000004", (-.296875, .0277777778), (.1046875, .2138888889), .7109744946),)),
        (3.312, 2966060, (-.1231062800, .0219652422, -.0227896887), (("000004", (-.296875, .0222222222), (.10625, .2166666667), .7108266000),)),
    )

    def tracks(specs):
        return [
            _f163_trace_track(
                track_id=f"vq2-track-{track_id}",
                outer_center=center,
                outer_span=span,
                confidence=confidence,
            )
            for track_id, center, span, confidence in specs
        ]

    base_s = 100.0
    first = rows[0]
    initial_tracks = tracks(first[3])
    controller = CleanCourseController(_config())
    controller.initialize(
        _update(initial_tracks, frame_id=first[1]),
        gate_index=0,
        fallback_center_norm=initial_tracks[0].center_norm,
        fallback_apparent_scale=initial_tracks[0].apparent_scale,
        now_s=base_s + first[0],
    )
    lineage = controller.successor
    assert lineage is not None
    assert lineage.track_id == "vq2-track-000002"

    post_credit_frames = []
    for elapsed, frame_id, body_rates, specs in rows[1:]:
        now = base_s + elapsed
        if elapsed == 3.219:
            assert controller.gate_index == 0
            assert controller.current.track_id == "vq2-track-000001"
            assert controller.note_race(
                gate_index=1, race_boot_ms=5200, now_s=now
            )
            assert controller.current is lineage
            assert controller.current.track_id == "vq2-track-000004"
        controller.observe(
            _update(tracks(specs), frame_id=frame_id),
            now_s=now,
            body_rates=body_rates,
        )
        if elapsed == 2.891:
            assert controller.successor is lineage
            assert controller.successor.track_id == "vq2-track-000002"
            assert controller._last_reconcile_status == (
                "successor-lineage-newborn-hold"
            )
        if elapsed == 2.984:
            assert controller.successor is lineage
            assert controller.successor.track_id == "vq2-track-000004"
            assert controller._last_reconcile_status == (
                "successor-lineage-rebound"
            )
        if elapsed >= 3.219:
            output = _command(
                controller, now, pitch=SPAWN_PITCH, yaw=0.0
            )
            post_credit_frames.append(frame_id)
            assert controller.gate_index == 1
            assert controller.current is lineage
            assert controller.current.track_id == "vq2-track-000004"
            assert output.current_track_id == "vq2-track-000004"
            assert controller.state is CleanCourseState.TRACK

    assert post_credit_frames == [2966057, 2966058, 2966059, 2966060]
    assert controller._last_reconcile_status == "exact-current"


def test_f168_commit_pitch_transfer_and_response_are_direction_safe():
    # Recorded F167 COMMIT counterfactual.  The old runtime applied the 0.9537
    # brake demand as a generic high gain while its target reversed toward
    # advance.  Extra authority is now nonzero only for a brake-direction
    # error; its marginal pitch-rate contribution can never be nose-down.
    rows = (
        (1.953, -0.4527, -0.3754, 0.9537),
        (1.984, -0.4217, -0.3840, 0.9537),
        (2.031, -0.3747, -0.3914, 0.9537),
        (2.109, -0.2967, -0.3868, 0.9537),
        (2.219, -0.1867, -0.3504, 0.9537),
        (2.406, -0.1603, -0.2624, 0.9537),
    )
    authorities = []
    for _t, target, measured, demand in rows:
        authority = _directional_brake_response_authority(
            demand,
            target_pitch_rad=target,
            measured_pitch_rad=measured,
            brake_pitch_offset_rad=-0.15,
        )
        authorities.append(authority)
        marginal = authority * (target - measured)
        assert marginal <= 1e-12
    assert authorities[:2] == pytest.approx([0.9537, 0.9537])
    assert authorities[2:] == pytest.approx([0.0] * 4)

    # A safely contained public-observation replay reaches COMMIT without
    # injecting state.  Its first COMMIT tick carries the preceding TRACK
    # target exactly instead of selecting a new phase endpoint.  The marginal
    # F163/F168 passages are intentionally no longer positive fixtures.
    controller, outputs, _now = _public_safe_commit_controller()
    first_commit = next(
        index
        for index, output in enumerate(outputs)
        if output.state is CleanCourseState.COMMIT
    )
    assert first_commit > 0
    admitted_pitch = outputs[first_commit - 1].target_pitch_rad
    assert controller._commit_pitch_target_rad == pytest.approx(admitted_pitch)
    assert all(
        output.target_pitch_rad == pytest.approx(admitted_pitch)
        for output in outputs[first_commit:]
    )


def test_f168_f167_commit_carries_positive_lower_clearance_to_exact_zero():
    # F167's recorded corridor translation is held fixed while only the
    # camera-pitch component is counterfactually reprojected.  Carrying the
    # entry pitch (rather than injecting the +0.15 COMMIT advance) keeps the
    # lower opening margin positive through the exact-zero tick.  The raw
    # recorded image path went negative, so this specifically closes the
    # pitch-owner regression without claiming a full vehicle-dynamics replay.
    rows = (
        (1.953, -0.375353, 0.035272, 0.223011),
        (1.984, -0.384021, 0.035272, 0.223011),
        (2.031, -0.391418, 0.040752, 0.237602),
        (2.109, -0.386751, 0.031477, 0.245366),
        (2.219, -0.350420, -0.063544, 0.289317),
        (2.359, -0.278570, -0.223269, 0.347108),
        (2.453, -0.245939, -0.283425, 0.347108),
        (2.563, -0.218012, -0.341387, 0.347108),
        (2.609, -0.208207, -0.361177, 0.347108),
        (2.656, -0.200133, -0.378936, 0.347108),
    )
    carried_pitch = rows[1][1]
    counterfactual_margins_px = []
    recorded_margins_px = []
    for _t, recorded_pitch, center_y, half_y in rows:
        attitude_invariant_center = center_y - (
            (SPAWN_PITCH - recorded_pitch)
            * ROTATION_COMP_FOCAL_NORM
        )
        carried_center = attitude_invariant_center + (
            (SPAWN_PITCH - carried_pitch)
            * ROTATION_COMP_FOCAL_NORM
        )
        counterfactual_margins_px.append(
            (carried_center + half_y) * 180.0
        )
        recorded_margins_px.append((center_y + half_y) * 180.0)
    assert rows[-1][0] == pytest.approx(2.656)
    assert min(counterfactual_margins_px) > 40.0
    assert recorded_margins_px[-1] < 0.0


def test_f167_predict_keeps_recent_outer_vertical_direction_over_imu():
    # TRACK->PREDICT does not invalidate the most recent accepted outer path.
    # A high Gate 1 with coherent climb direction retains decaying visual
    # correction on the first frozen-frame tick; opposing IMU climb damping
    # cannot instantly become the owner merely because association missed.
    first = _update(
        [_track("A", -0.10, -0.30, scale=0.18)], frame_id=6000
    )
    controller = CleanCourseController(_config())
    controller.initialize(
        first,
        gate_index=1,
        fallback_center_norm=(-0.10, -0.30),
        fallback_apparent_scale=0.18,
        now_s=100.0,
    )
    for frame in range(1, 5):
        now = 100.0 + 0.033 * frame
        controller.observe(
            _update(
                [_track("A", -0.10, -0.30, scale=0.18 + 0.002 * frame)],
                frame_id=6000 + frame,
            ),
            now_s=now,
        )
        _command(controller, now, pitch=SPAWN_PITCH, a_up=2.0)
    assert controller._vertical_direction_supported
    controller.observe(
        _update([], frame_id=6005),
        now_s=now + 0.033,
    )
    controller.observe(
        _update([], frame_id=6006),
        now_s=now + 0.280,
    )
    assert controller.state is CleanCourseState.PREDICT
    predicted = _command(
        controller,
        now + 0.300,
        pitch=SPAWN_PITCH,
        a_up=2.0,
    )

    assert controller._last_vertical_path_error < 0.0
    assert controller._last_vertical_visual_delta > 0.0
    assert controller._last_vertical_imu_delta <= 0.0
    assert controller._last_vertical_collective_target > (
        controller._last_vertical_support
    )
    assert predicted.thrust >= controller._last_vertical_support


def test_f166_recorded_f165_launch_has_no_material_pitch_reversal_or_chatter():
    # Recorded camera observations from F165 run
    # 20260802T011212Z-visual-course-88a0af14, session SHA-256
    # 73A803E7B2D6ABA6333A3966DD75DBB7CBE6AA7DCEBB54199F263CD15C6C2F82.
    # This concise 14/38-frame subset spans launch through t=1.313 s and
    # retains the observed outer box, inner aperture, confidence, pitch, and
    # body rates.  F165 accumulated 0.463 rad of target-pitch variation and
    # toggled its brake 11 times.  F166 must follow one outer-energy path:
    # no material nose-down-to-nose-up reversal and no framewise mode chatter.
    rows = (
        (0.000, 2264547, (-0.006250, -0.033333), (0.125000, 0.222222), 0.846929, -0.309831, (0.000032, 0.000355, 0.000003), (-0.007658, -0.040929), (0.048189, 0.100009)),
        (0.094, 2264550, (-0.006250, -0.033333), (0.125000, 0.222222), 0.846048, -0.309667, (-0.046755, 0.022490, -0.013722), (-0.010824, -0.038074), (0.045898, 0.087275)),
        (0.188, 2264553, (-0.003125, -0.033333), (0.125000, 0.222222), 0.847061, -0.308731, (-0.020252, 0.009649, -0.012066), (-0.008732, -0.034860), (0.045311, 0.073187)),
        (0.313, 2264556, (0.000000, -0.033333), (0.126563, 0.222222), 0.846673, -0.307404, (-0.009277, 0.012007, -0.007024), (-0.001163, -0.034710), (0.051088, 0.087126)),
        (0.406, 2264559, (0.000000, -0.033333), (0.128125, 0.222222), 0.846749, -0.306191, (-0.000183, 0.017218, -0.001122), (0.000803, -0.023426), (0.052560, 0.102275)),
        (0.516, 2264563, (0.000000, -0.027778), (0.129688, 0.227778), 0.856651, -0.304809, (0.000297, 0.022541, -0.000422), (0.003199, -0.020367), (0.059041, 0.112033)),
        (0.609, 2264565, (0.000000, -0.022222), (0.131250, 0.233333), 0.865676, -0.306495, (0.004618, 0.046163, 0.002113), (0.004029, -0.011271), (0.059973, 0.109194)),
        (0.703, 2264568, (-0.003125, -0.016667), (0.134375, 0.238889), 0.864407, -0.312175, (0.007515, -0.099359, 0.003102), (0.001204, -0.013323), (0.065758, 0.121657)),
        (0.797, 2264571, (-0.003125, 0.011111), (0.137500, 0.241667), 0.882903, -0.331401, (0.005888, -0.405923, 0.001936), (0.003392, 0.018726), (0.065651, 0.122941)),
        (0.891, 2264574, (-0.003125, 0.061111), (0.143750, 0.247222), 0.896043, -0.355506, (0.004473, -0.022743, 0.001806), (-0.003229, 0.073874), (0.054701, 0.117793)),
        (1.000, 2264577, (0.000000, 0.061111), (0.145313, 0.250000), 0.910058, -0.357421, (0.003790, -0.006404, -0.000606), (-0.000661, 0.062893), (0.069067, 0.134330)),
        (1.094, 2264580, (-0.003125, 0.072222), (0.150000, 0.261111), 0.931823, -0.367546, (-0.002019, -0.107685, -0.001751), (-0.000072, 0.071860), (0.076499, 0.136473)),
        (1.188, 2264583, (0.000000, 0.083333), (0.154688, 0.272222), 0.960036, -0.375067, (-0.001186, -0.039239, -0.000821), (0.002053, 0.083153), (0.075593, 0.141847)),
        (1.313, 2264586, (0.000000, 0.072222), (0.160938, 0.280556), 0.984940, -0.373416, (0.001604, 0.026416, 0.000465), (0.001851, 0.075100), (0.078072, 0.144345)),
    )

    def track(row):
        return _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            confidence=row[4],
            aperture_center=row[7],
            aperture_half=row[8],
        )

    controller = CleanCourseController(
        _config(launch_boost_duration_s=LAUNCH_BOOST_DURATION_S)
    )
    outputs = []
    brake_modes = []
    first = rows[0]
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        update = _update([track(row)], frame_id=row[1])
        if index == 0:
            controller.initialize(
                update,
                gate_index=0,
                fallback_center_norm=first[2],
                fallback_apparent_scale=math.sqrt(
                    first[3][0] * first[3][1]
                ),
                now_s=now,
            )
        else:
            controller.observe(update, now_s=now, body_rates=row[6])
        outputs.append(
            _command(controller, now, pitch=row[5], yaw=0.0)
        )
        brake_modes.append(controller._pitch_energy_brake_active)

    pitches = [output.target_pitch_rad for output in outputs]
    maximum_nose_up_reversal = _maximum_advance_to_brake_reversal(pitches)
    accumulated_variation = sum(
        abs(right - left) for left, right in zip(pitches, pitches[1:])
    )
    brake_transitions = sum(
        left != right for left, right in zip(brake_modes, brake_modes[1:])
    )

    assert rows[-1][0] >= 1.30
    assert maximum_nose_up_reversal < math.radians(0.5)
    assert brake_transitions <= 1
    assert accumulated_variation < 0.10
    assert accumulated_variation < 0.463 / 4.0


def test_f166_exact_zero_is_one_send_then_immediate_bounded_recovery():
    # Establish COMMIT from the dense credited F163 public observations, then
    # exercise the 31 ms and 47 ms post-zero gaps that separated the credited
    # and failed F168 runs.  The wire zero remains exact and singular; its
    # known bounded impulse is recovered on the next powered tick even when
    # acceleration trust is zero.
    def run(recovery_gap_s, *, pre_zero_delay_s=0.020):
        controller, outputs, now = _replay_f163_safe_passage(
            base_s=100.0,
            stop_at_commit=True,
        )
        assert controller.state is CleanCourseState.COMMIT

        coast_s = now + 0.033
        controller.observe(_update([], frame_id=2053010), now_s=coast_s)
        assert controller.state is CleanCourseState.COAST_FOR_CREDIT
        zero_s = coast_s + pre_zero_delay_s
        zero = _command(
            controller,
            zero_s,
            pitch=SPAWN_PITCH,
            accel_trust=0.0,
        )
        outputs.append(zero)
        assert (
            zero.target_roll_rad,
            zero.target_pitch_rad,
            zero.yaw_rate_rad_s,
            zero.thrust,
        ) == (0.0, 0.0, 0.0, 0.0)
        assert controller._zero_recovery_pending

        assert controller.note_race(
            gate_index=1,
            race_boot_ms=4000,
            now_s=zero_s + 0.010,
        )
        gate_one = _f163_trace_track(
            outer_center=(0.0, -0.30),
            outer_span=(0.12, 0.22),
            track_id="gate-one",
        )
        controller.observe(
            _update([gate_one], frame_id=2053011),
            now_s=zero_s + recovery_gap_s - 0.010,
        )
        recovery = _command(
            controller,
            zero_s + recovery_gap_s,
            pitch=SPAWN_PITCH,
            a_up=-8.0,
            accel_trust=0.0,
        )
        outputs.append(recovery)
        recovery_delta = controller._last_zero_recovery_delta

        assert controller._zero_recovery_applied
        assert not controller._zero_recovery_pending
        assert controller._zero_sink_debt_m_s == 0.0
        assert recovery.thrust == pytest.approx(
            controller._last_vertical_collective_target, abs=1e-12
        )
        assert recovery.thrust > controller._last_vertical_support
        assert (
            controller.config.min_thrust
            <= recovery.thrust
            <= controller.config.max_thrust
        )
        assert sum(output.thrust == 0.0 for output in outputs) == 1

        # The debt is one-shot.  Fresh visual trajectory feedback can remain,
        # but no persistent open-loop zero-recovery owner survives this tick.
        followup = _command(
            controller,
            zero_s + recovery_gap_s + 0.020,
            pitch=SPAWN_PITCH,
            a_up=-8.0,
            accel_trust=0.0,
        )
        assert followup.thrust >= controller.config.min_thrust
        assert controller._last_zero_recovery_delta == 0.0
        return recovery_delta, recovery.thrust

    short_delta, short_thrust = run(0.031)
    long_delta, long_thrust = run(0.047)
    delayed_zero_delta, _ = run(0.031, pre_zero_delay_s=0.090)
    assert long_delta >= short_delta > 0.0
    assert long_thrust >= short_thrust
    # Powered time before the zero send is not part of the zero impulse.
    assert delayed_zero_delta == pytest.approx(short_delta, abs=1e-12)


def test_f163_dense_gate0_trace_reaches_sustained_safe_commit():
    # The dense credited F163 observations retain every distinct camera frame
    # across admission.  Unlike the sparse/marginal F168 samples, their full
    # uncertainty tube remains inside the reduced body/frame corridor for the
    # complete sustain window, proving live passage reachability without
    # direct controller-state injection.
    controller, outputs, now = _replay_f163_safe_passage()

    assert controller.gate_index == 0
    assert controller.state is CleanCourseState.COMMIT
    first_commit = next(
        index
        for index, output in enumerate(outputs)
        if output.state is CleanCourseState.COMMIT
    )
    assert _F163_SAFE_PASSAGE_ROWS[first_commit][0] == pytest.approx(2.078)
    assert controller._commit_safe_since_s == pytest.approx(101.953)
    assert controller._commit_entry_s == pytest.approx(102.078)
    assert controller._last_commit_admission.admissible
    certificate = controller.current.corridor_certificate
    assert certificate is not None
    assert certificate.gate_index == 0
    assert certificate.source_s == pytest.approx(now, abs=1e-6)

    old_current = controller.current
    assert controller.note_race(
        gate_index=1, race_boot_ms=3000, now_s=103.438
    )
    assert controller.gate_index == 1
    assert old_current.corridor_certificate.gate_index == 0
    assert controller.current is None


def test_f163_gate1_trace_transports_geometry_without_modality_false_closure():
    # Negative-labeled F163 Gate 1: the last usable aperture is genuinely left
    # of its corridor.  The certificate remains inspectable for the 0.438 s fit
    # gap, while outer-only closure stays near the recorded 0.473/s instead of
    # jumping to the mixed-modality ~2/s value.  Admission must remain false.
    rows = (
        (3.438, 2053043, (-0.38125, 0.127778), (0.1140625, 0.202778), -0.450153, 1, (-0.395507, 0.087350), (0.045135, 0.077757), (-0.223011, 0.006873, -0.331034), FrameEdge.NONE),
        (4.328, 2053069, (-0.15, 0.044444), (0.15625, 0.280556), -0.456161, 1, (-0.176428, -0.010396), (0.044799, 0.073925), (-0.278520, -0.010782, -0.175512), FrameEdge.NONE),
        (4.516, 2053075, (-0.128125, 0.027778), (0.171875, 0.302778), -0.453767, 1, (-0.156181, -0.028572), (0.054077, 0.090570), (-0.223875, 0.047371, -0.219492), FrameEdge.NONE),
        (4.672, 2053080, (-0.109375, 0.011111), (0.1875, 0.319444), -0.449502, 1, (-0.140326, -0.044394), (0.055520, 0.099789), (-0.157176, 0.066117, -0.200226), FrameEdge.NONE),
        (5.110, 2053093, (-0.1, 0.027778), (0.240625, 0.380556), -0.457940, 1, None, None, (0.041316, -0.008344, -0.064853), FrameEdge.NONE),
    )
    controller, _outputs = _replay_f163_rows(rows, gate_index=1)
    corridor = controller._transported_corridor(
        controller.current, now_s=105.110
    )

    assert controller.gate_index == 1
    assert controller.state is CleanCourseState.TRACK
    assert corridor is not None
    assert corridor.source_age_s == pytest.approx(0.438, abs=1e-6)
    assert not corridor.live
    assert corridor.center_x == pytest.approx(-0.138, abs=0.015)
    assert corridor.half_x == pytest.approx(0.07125, abs=0.004)
    assert corridor.half_y == pytest.approx(0.11888, abs=0.006)
    assert controller.current.outer_expansion_rate == pytest.approx(
        0.473, abs=0.16
    )
    assert controller.current.expansion_rate < 1.0
    assert controller._last_commit_admission.status == (
        "corridor-known/not-contained"
    )
    assert not controller._last_commit_admission.admissible


def test_f167_f166_gate1_trace_keeps_coherent_lateral_intercept_authority():
    # F166 run 20260802T025041Z-visual-course-c9059224, trace SHA-256
    # E2345D27ADB0E22FA9858EF4534C2A1AB980B8BCE0269F5862C21587B1234A9B.
    # The camera bearing improved left-to-center, but both projected endpoints
    # began escaping left again while covariance held control_authority at 0.
    # Replay only normal tracker observations and IMU attitude/rates: fresh
    # outer image motion remains a physical path owner, so bank cannot unwind
    # merely because yaw made the bearing look better.
    rows = (
        # t, frame, outer center/span/conf, aperture center/half, rates, pitch, yaw
        (3.000, 2441910, (-0.428125, -0.394444), (0.114062, 0.219444), 0.724430, (-0.438601, -0.445152), (0.044199, 0.068649), (-0.017411, -0.051506, -0.301625), -0.214778, -0.110072),
        (3.125, 2441914, (-0.393750, -0.327778), (0.117188, 0.225000), 0.732855, (-0.409434, -0.376945), (0.048496, 0.075380), (-0.299430, -0.448958, -0.335780), -0.261323, -0.150727),
        (3.250, 2441918, (-0.343750, -0.233333), (0.121875, 0.236111), 0.748821, (-0.358973, -0.286569), (0.047336, 0.076430), (-0.591809, -0.441315, -0.329715), -0.330260, -0.191020),
        (3.360, 2441921, (-0.306250, -0.194444), (0.128125, 0.250000), 0.774885, (-0.320434, -0.260926), (0.049062, 0.070979), (-0.512699, -0.288887, -0.329506), -0.376210, -0.228173),
        (3.485, 2441925, (-0.262500, -0.172222), (0.137500, 0.266667), 0.773870, (-0.278047, -0.239662), (0.047805, 0.078688), (-0.367664, -0.160233, -0.329579), -0.411054, -0.270455),
        (3.532, 2441926, (-0.253125, -0.166667), (0.139062, 0.272222), 0.773615, (-0.271938, -0.240569), (0.049248, 0.076669), (-0.351168, -0.142659, -0.329585), -0.416587, -0.279586),
        (3.625, 2441929, (-0.231250, -0.166667), (0.146875, 0.286111), 0.777498, (-0.248638, -0.242922), (0.051179, 0.082375), (-0.282222, -0.081637, -0.329588), -0.432852, -0.312150),
        (3.719, 2441932, (-0.209375, -0.172222), (0.154688, 0.302778), 0.773330, (-0.241086, -0.235070), (0.046910, 0.079751), (-0.225886, -0.046854, -0.329588), -0.444025, -0.343255),
        (3.750, 2441933, (-0.203125, -0.172222), (0.157813, 0.305556), 0.747828, None, None, (-0.247260, -0.031372, -0.329588), -0.447494, -0.355424),
        (3.813, 2441934, (-0.196875, -0.177778), (0.164062, 0.308333), 0.737245, None, None, (-0.169514, -0.008925, -0.329588), -0.452792, -0.377580),
        (3.875, 2441936, (-0.190625, -0.183333), (0.168750, 0.319444), 0.734999, None, None, (-0.180970, -0.002653, -0.329588), -0.457194, -0.399991),
        (4.000, 2441940, (-0.171875, -0.200000), (0.185937, 0.341667), 0.752614, None, None, (-0.135423, 0.021483, -0.261356), -0.463190, -0.443165),
        (4.032, 2441941, (-0.171875, -0.200000), (0.190625, 0.350000), 0.757875, None, None, (-0.082437, 0.026789, -0.223829), -0.463929, -0.452303),
        (4.141, 2441944, (-0.175000, -0.216667), (0.207813, 0.369444), 0.589668, None, None, (-0.091214, 0.030485, -0.226391), -0.465921, -0.478633),
        (4.188, 2441946, (-0.178125, -0.222222), (0.217188, 0.383333), 0.560892, None, None, (-0.109086, 0.031502, -0.197793), -0.466450, -0.491835),
    )

    def track(row):
        return _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            confidence=row[4],
            aperture_center=row[5],
            aperture_half=row[6],
        )

    controller = CleanCourseController(_config())
    first = rows[0]
    samples = {}
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        update = _update([track(row)], frame_id=row[1])
        if index == 0:
            controller.initialize(
                update,
                gate_index=1,
                fallback_center_norm=first[2],
                fallback_apparent_scale=math.sqrt(first[3][0] * first[3][1]),
                now_s=now,
            )
        else:
            controller.observe(update, now_s=now, body_rates=row[7])
        output = _command(controller, now, pitch=row[8], yaw=row[9])
        motion = controller._last_lateral_motion
        samples[row[0]] = (
            motion,
            controller._last_lateral_baseline_reference_x,
            controller._lateral_intercept_reference_x,
            output,
        )

    for elapsed in (3.813, 4.032, 4.188):
        motion, baseline, reference, output = samples[elapsed]
        assert motion.control_authority < 0.05
        assert motion.fallback_intercept_error < 0.0
        assert motion.optical_intercept_error < 0.0
        assert baseline < motion.bearing_error
        assert reference < motion.bearing_error
        assert output.target_roll_rad < 0.0
    # From 3.813 to 4.032 the camera bearing improves, while the physical
    # fallback miss worsens.  The carried bank reference remains materially
    # beyond the camera bearing instead of collapsing to bearing-only control.
    early = samples[3.813]
    late = samples[4.032]
    assert abs(late[0].bearing_error) < abs(early[0].bearing_error)
    assert abs(late[0].fallback_intercept_error) > abs(
        early[0].fallback_intercept_error
    )
    assert late[2] < late[0].bearing_error - 0.07
    assert abs(late[2]) > 0.25


def test_f167_f166_zero_authority_successor_cannot_attenuate_current_heading():
    # At F166 t=3.719 Gate 2 first appeared at x=+0.325 while the race-owned
    # Gate 1 remained x=-0.209.  Its safe-passage authority was exactly zero,
    # yet the old successor-present branch reduced current yaw custody.  The
    # public observations below retain the actual geometry: existence alone
    # cannot change either current heading direction or magnitude.
    current_only = CleanCourseController(_config())
    with_successor = CleanCourseController(_config())
    current = _f163_trace_track(
        outer_center=(-0.209375, -0.172222),
        outer_span=(0.154688, 0.302778),
        confidence=0.773330,
        aperture_center=(-0.241086, -0.235070),
        aperture_half=(0.046910, 0.079751),
        track_id="A",
    )
    successor = _f163_trace_track(
        outer_center=(0.325000, 0.277778),
        outer_span=(0.039062, 0.100000),
        confidence=0.450548,
        track_id="B",
    )
    for controller, tracks in (
        (current_only, [current]),
        (with_successor, [current, successor]),
    ):
        controller.initialize(
            _update(tracks, frame_id=2441932),
            gate_index=1,
            fallback_center_norm=current.center_norm,
            fallback_apparent_scale=current.apparent_scale,
            now_s=103.719,
        )
    alone = _command(current_only, 103.719, pitch=-0.444025, yaw=-0.343255)
    preview = _command(with_successor, 103.719, pitch=-0.444025, yaw=-0.343255)

    assert preview.successor_blend == pytest.approx(0.0, abs=1e-12)
    assert with_successor._turn_reference_x == pytest.approx(
        current_only._turn_reference_x, abs=1e-12
    )
    assert preview.yaw_rate_rad_s == pytest.approx(
        alone.yaw_rate_rad_s, abs=1e-12
    )
    assert with_successor._lateral_intercept_reference_x == pytest.approx(
        current_only._lateral_intercept_reference_x, abs=1e-12
    )
    assert preview.target_roll_rad == pytest.approx(
        alone.target_roll_rad, abs=1e-12
    )
    assert preview.yaw_rate_rad_s < 0.0


def test_f164_gate1_trace_never_steers_from_stale_aperture_wrong_side():
    # F164 trace 20260801T235635Z-visual-course-aa5a3f98.  At 4.625 the
    # opposite successor reversed yaw after the current aperture expired; at
    # 6.781 the still-live current outer box was RIGHT/BOTTOM at x=+0.834 while
    # the extrapolated aperture state remained near x=-0.922.  Replay normal
    # observations only: fresh outer current-gate evidence owns both yaw and
    # bank, unsafe successor preview stays revoked, and the right inequality
    # immediately prevents the stale left sign from reaching steering.
    rows = (
        # t, frame, current center/span/conf/edge, optional aperture tuple,
        # optional successor center/span/conf/edge, body rates, pitch, yaw
        (3.906, 2128622, (-0.221875, 0.161111111), (0.146875, 0.238888889), 0.671792014, 0, ((-0.256985049, 0.109378578), (0.035378985, 0.048116964), -3.187878880), ((0.334375, 0.422222222), (0.0328125, 0.10), 0.407808831, 0), (-0.387679480, -0.049033876, -0.329588156), -0.448136049, -0.506405050),
        (4.063, 2128627, (-0.196875, 0.133333333), (0.1609375, 0.258333333), 0.553402497, 0, None, ((0.396875, 0.466666667), (0.0359375, 0.105555556), 0.425165561, 0), (-0.407456390, 0.020964433, -0.329588394), -0.439268571, -0.561557841),
        (4.156, 2128630, (-0.181250, 0.127777778), (0.1703125, 0.272222222), 0.509809626, 0, None, ((0.437500, 0.511111111), (0.0390625, 0.111111111), 0.432195569, 0), (-0.396001480, -0.025096651, -0.232772689), -0.439660522, -0.593350688),
        (4.313, 2128634, (-0.178125, 0.105555556), (0.1796875, 0.288888889), 0.502317802, 0, None, ((0.465625, 0.555555556), (0.0421875, 0.116666667), 0.447672725, 0), (-0.378993652, 0.039850224, -0.145854618), -0.434441250, -0.624350061),
        (4.391, 2128637, (-0.178125, 0.083333333), (0.190625, 0.308333333), 0.489367983, 0, None, ((0.484375, 0.588888889), (0.046875, 0.122222222), 0.449992986, 0), (-0.367389313, 0.025291358, -0.143996131), -0.432447887, -0.636661829),
        (4.516, 2128640, (-0.181250, 0.072222222), (0.1984375, 0.322222222), 0.480734524, 0, None, ((0.506250, 0.627777778), (0.0484375, 0.130555556), 0.447759425, 0), (-0.359443657, 0.002858986, -0.050915416), -0.432616647, -0.653449736),
        (4.625, 2128644, (-0.196875, 0.055555556), (0.2125000, 0.344444444), 0.467926489, 0, None, ((0.512500, 0.694444444), (0.0515625, 0.138888889), 0.454060875, 0), (-0.336344860, -0.037201266, 0.032074861), -0.435148248, -0.653937687),
        (4.766, 2128648, (-0.231250, 0.044444444), (0.2265625, 0.372222222), 0.453144644, 0, None, ((0.500000, 0.766666667), (0.0562500, 0.144444444), 0.456563414, 0), (-0.278529040, -0.063551034, 0.148095403), -0.441651975, -0.637453818),
        (4.922, 2128653, (-0.290625, 0.027777778), (0.2484375, 0.411111111), 0.437458389, 0, None, ((0.468750, 0.850000000), (0.0593750, 0.150000000), 0.456568261, 8), (-0.244444809, -0.053099297, 0.213641826), -0.444682564, -0.603149475),
        (5.094, 2128658, (-0.350000, 0.005555556), (0.2750000, 0.455555556), 0.432574486, 0, None, ((0.440625, 0.883333333), (0.0640625, 0.113888889), 0.515802283, 8), (-0.223431728, -0.050227444, 0.177147333), -0.445419837, -0.566434482),
        (5.281, 2128663, (-0.403125, -0.016666667), (0.3062500, 0.502777778), 0.428927906, 0, None, ((0.434375, 0.916666667), (0.0609375, 0.083333333), 0.583500241, 8), (-0.099544040, -0.046295695, 0.116031874), -0.446941557, -0.535128812),
        (5.453, 2128669, (-0.456250, -0.022222222), (0.3484375, 0.558333333), 0.442261050, 0, None, None, (-0.040477395, -0.035739138, 0.054380503), -0.449527308, -0.518335916),
        (5.625, 2128674, (-0.481250, -0.011111111), (0.3875000, 0.611111111), 0.462169348, 0, None, None, (-0.032426182, -0.024507014, 0.013508462), -0.453045161, -0.511015466),
        (5.797, 2128679, (-0.500000, 0.005555556), (0.4328125, 0.663888889), 0.495050669, 0, None, None, (-0.016110751, -0.015147697, 0.004981172), -0.455345094, -0.508523970),
        (5.953, 2128683, (-0.578125, -0.044444444), (0.3937500, 0.627777778), 0.768541761, 0, None, None, (-0.010042917, -0.009672853, 0.000605964), -0.457081881, -0.507469724),
        (6.141, 2128689, (-0.525000, 0.127777778), (0.4734375, 0.813888889), 0.970908773, 1, None, None, (-0.006195319, -0.005151205, -0.000399903), -0.458325691, -0.507002857),
        (6.313, 2128694, (-0.425000, 0.161111111), (0.5750000, 0.836111111), 0.954952201, 9, None, None, (-0.003302115, -0.002759674, -0.000423318), -0.458963729, -0.506836079),
        (6.484, 2128699, (-0.221875, 0.200000000), (0.7765625, 0.800000000), 0.865331973, 9, None, None, (-0.035145353, 0.018945087, -0.423337739), -0.465090087, -0.525609346),
        (6.531, 2128701, (0.000000, 0.222222222), (0.8765625, 0.777777778), 0.773216592, 8, None, ((0.146875, 0.788888889), (0.1843750, 0.211111111), 0.710723591, 8), (-0.148244567, 0.086754609, -0.346993547), -0.468616403, -0.543340900),
        (6.578, 2128702, (0.103125, 0.233333333), (0.8984375, 0.766666667), 0.718547467, 12, None, ((0.184375, 0.794444444), (0.1828125, 0.202777778), 0.688995965, 8), (-0.101078286, 0.123076354, -0.311965119), -0.468416136, -0.563950916),
        (6.625, 2128704, (0.228125, 0.261111111), (0.7718750, 0.736111111), 0.730221360, 12, None, ((0.262500, 0.816666667), (0.1875000, 0.180555556), 0.570642312, 8), (-0.096793629, 0.136764411, -0.335199904), -0.467682308, -0.580436874),
        (6.781, 2128709, (0.834375, 0.461111111), (0.1671875, 0.538888889), 0.826410803, 12, None, ((0.440625, 0.866666667), (0.1640625, 0.133333333), 0.439284658, 8), (-0.157070331, 0.447318334, -0.106731873), -0.445495336, -0.645270117),
    )

    def tracks(row):
        aperture = row[6]
        current = _f163_trace_track(
            outer_center=row[2],
            outer_span=row[3],
            confidence=row[4],
            track_id="A",
            aperture_center=None if aperture is None else aperture[0],
            aperture_half=None if aperture is None else aperture[1],
            aperture_log_scale=None if aperture is None else aperture[2],
            clipping=FrameEdge(row[5]),
        )
        result = [current]
        successor = row[7]
        if successor is not None:
            result.append(
                _f163_trace_track(
                    outer_center=successor[0],
                    outer_span=successor[1],
                    confidence=successor[2],
                    track_id="B",
                    clipping=FrameEdge(successor[3]),
                )
            )
        return result

    first = rows[0]
    controller = CleanCourseController(_config())
    controller.initialize(
        _update(tracks(first), frame_id=first[1]),
        gate_index=1,
        fallback_center_norm=first[2],
        fallback_apparent_scale=math.sqrt(first[3][0] * first[3][1]),
        now_s=100.0 + first[0],
    )
    controller._alt_est_m = 2.0
    samples = {}
    control_time = 100.0 + first[0]
    for index, row in enumerate(rows):
        now = 100.0 + row[0]
        if index:
            previous = rows[index - 1]
            previous_time = 100.0 + previous[0]
            # The compact rows retain selected camera updates, while the live
            # controller still ran at ~30 Hz between them.  Exercise that same
            # bounded command cadence so slew latency is represented rather
            # than accidentally reduced to one command per selected frame.
            while control_time + 0.033 < now - 1e-9:
                control_time += 0.033
                fraction = (control_time - previous_time) / max(
                    1e-6, now - previous_time
                )
                _command(
                    controller,
                    control_time,
                    pitch=previous[9]
                    + fraction * (row[9] - previous[9]),
                    yaw=previous[10]
                    + fraction * (row[10] - previous[10]),
                )
            controller.observe(
                _update(tracks(row), frame_id=row[1]),
                now_s=now,
                body_rates=row[8],
            )
        output = _command(
            controller,
            now,
            pitch=row[9],
            yaw=row[10],
        )
        control_time = now
        outer_heading = controller._horizontal_control_observable(
            controller.current, now
        )[0]
        samples[row[0]] = (
            output,
            outer_heading,
            controller._turn_reference_x,
            controller._lateral_intercept_reference_x,
            controller.current.x,
            controller.current.outer_x_axis.p,
        )

    for elapsed in (4.516, 4.625, 4.766):
        output, outer_heading, reference, intercept, *_ = samples[elapsed]
        assert outer_heading < 0.0
        assert output.successor_blend == pytest.approx(0.0, abs=1e-12)
        assert reference < 0.0
        assert intercept < 0.0
        assert output.yaw_rate_rad_s < 0.0
        assert output.target_roll_rad < 0.0

    first_right = samples[6.578]
    assert first_right[1] > 0.0
    assert first_right[2] >= 0.0
    assert first_right[3] >= 0.0
    assert first_right[0].successor_blend == pytest.approx(0.0, abs=1e-12)
    assert first_right[0].yaw_rate_rad_s >= 0.0

    final = samples[6.781]
    assert final[1] > 0.50
    assert final[4] < 0.0  # stale aperture projection remains inspectable
    assert final[5] > 0.0  # but fresh outer x owns steering
    assert final[0].yaw_rate_rad_s > 0.0
    assert final[0].target_roll_rad > 0.0
    assert final[0].yaw_rate_rad_s * final[0].target_roll_rad > 0.0
    assert controller.state is CleanCourseState.TRACK
    assert controller.gate_index == 1


def test_commit_entry_requires_inner_aperture_budget():
    # F56 (trace efb189d4 + f55/ frames): an outer-bbox corridor bound
    # LOOSER than the gate's own half-width committed F55 at ex ~-0.17
    # with the gate half-width ~0.125 and crossed beside the left post.
    # 2026-07-30: outer-bbox alignment is NOT the crossing authority — the
    # entry budget is measured against the CURRENT usable inner aperture.
    # An approach aligned inside the outer bbox but with an aperture too
    # tight for the error budget (0.10 > 0.6*0.12) stays in TRACK; and
    # with NO passage-usable aperture at all even a perfectly centered
    # approach stays in TRACK.
    tight = _commit_controller()
    tight.current.aperture_half_x = 0.12
    _drive_commit_window(tight, 100.10)
    assert tight.state is CleanCourseState.TRACK
    blind = _commit_controller()
    blind.current.x_axis.p = 0.0
    blind.current.raw_x = 0.0
    blind.current.aperture_half_x = None  # fit not passage-usable this frame
    _drive_commit_window(blind, 100.10)
    assert blind.state is CleanCourseState.TRACK


def test_near_plane_track_holds_closure_while_commit_budget_false():
    # F73 (20260730T063739Z-visual-course-34c53413): the gate-1 leg kept
    # CLOSING with the entry budget false (ey +0.31 -> +0.49 into bottom
    # censorship); the aim died at the plane, a bottom-right splinter was
    # adopted, and the drone wandered 9 s blind into structure (collision
    # id 1002).  At the plane the angular error rate outruns re-centering,
    # so crossing energy is controlled BEFORE censorship: a gate-1+ TRACK
    # in the near-plane regime with a false budget cuts advance and demands
    # the full brake.  The same budget passing still arms COMMIT.
    # Misaligned beyond the 60% aperture margin: budget false, and the
    # F71 course relax levels the camera (ey 0.35 >= 0.18).
    holding = _commit_controller()
    current = holding.current
    current.x_axis.p = 0.30
    current.y_axis.p = 0.35
    current.raw_x = 0.30
    current.raw_y = 0.35
    current.scale_axis.p = -0.50  # relax's commit regime reads the inner scale
    out, _ = _drive_commit_window(holding, 100.10)
    assert holding.state is CleanCourseState.TRACK  # COMMIT never arms
    assert holding._pre_cross_brake_active
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    # Centered but with a stale x-axis the budget is still false: the hold
    # demands the FULL brake attitude instead of the advance law's
    # nose-down pitch (relax quiet at ey 0.05 < 0.18).
    stalled = _commit_controller()
    now = 100.10
    for _ in range(12):
        now += 0.033
        stalled.current.last_measurement_s = now
        stalled.current.last_y_measurement_s = now
        # x stamp frozen at construction: 0.4 s stale inside the window.
        out = _command(stalled, now, pitch=SPAWN_PITCH)
    assert stalled.state is CleanCourseState.TRACK
    assert stalled._pre_cross_brake_active
    assert out.target_pitch_rad <= SPAWN_PITCH + 1e-9
    # A budget-satisfying first Gate-0 tick still follows the ordinary
    # continuous law; F105 changes brake strength, not its demand threshold.
    gate0 = _tracked_controller(_track("A", 0.10, 0.05, scale=0.50))
    gate0.current.outer_log_scale = -0.50
    gate0.current.last_x_measurement_s = 100.0 - 0.40
    out0 = _command(gate0, 100.10)
    assert out0.target_pitch_rad > SPAWN_PITCH - 0.15 + 1e-9


def test_near_plane_hold_brakes_without_self_blinding():
    # F73/F75 history: closure energy must be arrested in the pre-censorship
    # window, but F73b's blind full brake pitched the gate out of view and
    # F93's binary relax surrendered the brake to level while still hot.
    # F94: the near-plane hold still cuts advance and demands the full
    # brake with the budget false (COMMIT never arms), while the custody
    # floor caps the emitted attitude so the gate can never be pitched out
    # of the FOV.  With the gate genuinely 0.31 below center the floor is
    # LEVEL (the vertical channel owns the re-centering); once the gate
    # recovers to center with closure still hot, the floor admits brake.
    early = _commit_controller()
    current = early.current
    current.x_axis.p = -0.13
    current.y_axis.p = 0.31
    current.raw_x = -0.13
    current.raw_y = 0.31
    current.scale_axis.p = -1.00
    current.scale_axis.v = 0.85
    current.outer_log_scale = -1.00
    current.aperture_half_x = 0.20
    current.aperture_half_y = 0.20
    out, now = _drive_commit_window(early, 100.10)
    assert early.state is CleanCourseState.TRACK  # COMMIT never arms
    assert early._pre_cross_brake_active
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    # Gate recovered to center, still hot (0.85/s): the custody floor now
    # admits a real brake — spawn - (0.18 - 0)/1.6 — instead of F73b's
    # self-blinding full brake or F71's level surrender.
    early.current.y_axis.p = 0.0
    early.current.raw_y = 0.0
    out, now = _drive_commit_window(early, now)
    assert early.state is CleanCourseState.TRACK
    assert early._pre_cross_brake_active
    assert out.target_pitch_rad == pytest.approx(
        SPAWN_PITCH - 0.18 / 1.6, abs=1e-9
    )


def test_hot_closure_keeps_the_single_brake_through_attitude_artifact():
    # F93 (20260730T143851Z-visual-course-7e67b464): the held course brake
    # (-0.577 attitude, 0.267 rad nose-up) had HALVED the closure rate when
    # the attitude artifact walked RAW ey to +0.24; the F71 relax read
    # raw ey >= 0.18 and surrendered the brake to FULL LEVEL with closure
    # still ~1.0/s — the drone re-advanced into the gate-1 plane, the entry
    # budget correctly refused COMMIT (expansion, |vz|), and it hit the
    # structure (id 1001).  The compensated ey (~-0.19: the gate is actually
    # ABOVE center) must keep the custody floor at a hard brake.  On the
    # parent the emitted pitch here is LEVEL.
    controller = _commit_controller()
    current = controller.current
    current.x_axis.p = -0.10
    current.y_axis.p = 0.24
    current.raw_x = -0.10
    current.raw_y = 0.24
    current.scale_axis.p = -0.50  # commit regime reads the inner scale
    current.scale_axis.v = 1.0  # ~1.0 log/s closure: budget false
    current.aperture_half_x = 0.20
    current.aperture_half_y = 0.20
    brake_attitude = SPAWN_PITCH - 0.267  # the held F93 brake attitude
    now = 100.10
    out = None
    for _ in range(15):
        now += 0.033
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        current.last_y_measurement_s = now
        out = _command(controller, now, pitch=brake_attitude)
    assert controller.state is CleanCourseState.TRACK  # COMMIT never arms
    assert controller._pre_cross_brake_active
    # Compensated ey = 0.24 - 0.267*1.6 ~= -0.19, so custody does not relax
    # the request.  F125 removes the obsolete deeper course mode; the one
    # brake remains fully applied instead of surrendering to level.
    assert out.target_pitch_rad == pytest.approx(
        SPAWN_PITCH + controller.config.pre_cross_brake_pitch_rad, abs=1e-9
    )


def test_no_track_search_latches_altitude_support():
    # F75: a no-track SEARCH holds altitude support — the F50 memory
    # descent turned F74's gate-1 miss into a blind 4.8 s sink to the
    # alt-est floor before the structure strike.  With a reliable bearing
    # memory BELOW the camera (remembered ey +0.50, the descent case the
    # F50 servo was built for), the emitted collective stays at support.
    controller = _tracked_controller(_track("A", 0.10, 0.45))
    controller._enter_search(100.10)
    controller._set_reliable_bearing(0.10, 0.50)
    out = _command(controller, 100.143, pitch=SPAWN_PITCH)
    assert out.state is CleanCourseState.SEARCH
    assert out.thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)


def _converged_gate_one_vertical(vz_m_s):
    """Gate-1 TRACK controller with a fresh high-gate outer replay."""

    controller = _tracked_controller(_track("A", 0.0, -0.10, scale=0.20))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 2.0  # honest altitude (floor quiet)
    scale = math.exp(-1.6)
    now = 100.10
    out = None
    for frame_id in range(4, 19):  # converge through real fresh observations
        now += 0.033
        controller.observe(
            _update(
                [_track("B", 0.0, -0.10, scale=scale)],
                frame_id=frame_id,
            ),
            now_s=now,
        )
        controller._vz_est_m_s = vz_m_s  # hold the IMU climb rate
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.TRACK
    return out


def test_far_vertical_arrival_damping_never_vetoes_clear_visual_direction():
    # Far from the crossing, fresh outer-y supplies the continuous climb
    # baseline.  Opposing IMU damping may reduce that request, but cannot take
    # collective below tilt support; supporting sink damping may strengthen it.
    climbing = _converged_gate_one_vertical(0.60)
    settled = _converged_gate_one_vertical(0.0)
    assert SPAWN_SUPPORT < climbing.thrust <= settled.thrust
    sinking = _converged_gate_one_vertical(-0.30)
    assert settled.thrust <= sinking.thrust <= 0.34

    def _gate0_thrust(vz_m_s):
        scale = math.exp(-2.20)
        gate0 = _tracked_controller(
            _track("A", 0.0, -0.10, scale=scale)
        )
        gate0._alt_est_m = 2.0
        now = 100.10
        out = None
        for frame_id in range(3, 18):
            now += 0.033
            gate0.observe(
                _update(
                    [_track("A", 0.0, -0.10, scale=scale)],
                    frame_id=frame_id,
                ),
                now_s=now,
            )
            gate0._vz_est_m_s = vz_m_s
            out = _command(gate0, now, pitch=SPAWN_PITCH)
        return out

    gate0_climbing = _gate0_thrust(0.60)
    gate0_settled = _gate0_thrust(0.0)
    assert SPAWN_SUPPORT < gate0_climbing.thrust <= gate0_settled.thrust
    assert gate0_climbing.thrust > SPAWN_SUPPORT


def test_course_leg_projects_corrected_image_motion_into_optical_intercept():
    # The passage model removes closure dilation and projects the remaining
    # physical bearing motion to the gate plane.  It does not reinterpret the
    # normalized rate as metric vertical velocity.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.20))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 2.0  # honest altitude (floor quiet)
    current = controller.current
    _settle_commit_passage_covariance(current)
    current.y_axis.p = 0.0
    current.raw_y = 0.0
    current.y_axis.v = 0.30  # centered gate moving physically image-down
    current.scale_axis.p = -1.6
    current.scale_axis.v = 0.10
    now = 100.10
    out = None
    for _ in range(15):
        now += 0.033
        controller._vz_est_m_s = 0.0
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.TRACK
    assert out.vertical_qualified
    assert controller._last_vertical_motion_delta == pytest.approx(-0.009)
    assert controller._last_vertical_imu_delta > 0.0
    assert out.thrust == pytest.approx(
        controller._last_vertical_support
        + controller._last_vertical_visual_delta
        + controller._last_vertical_imu_delta,
        abs=1e-12,
    )
    assert out.thrust < SPAWN_SUPPORT - 0.0075


def test_course_leg_closure_excess_leaves_collective_alone():
    # F96: the F77 closure-excess collective cut is deleted.  Under the
    # vz-tracking law a sub-support cut is immediately restored by the
    # tracker (a masked no-op) and was a fourth incoherent vz term in
    # the F95 limit cycle.  Forward braking stays with the closure
    # governor's PITCH law; a hot closure with the gate centered and
    # vz zero leaves the collective at support.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.20))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 2.0
    current = controller.current
    current.y_axis.p = 0.0
    current.raw_y = 0.0
    current.y_axis.v = 0.0
    current.scale_axis.p = -1.6  # span ~0.20, outside the near-plane zone
    current.scale_axis.v = 0.58  # hot closure (target 0.35)
    now = 100.10
    out = None
    for _ in range(15):
        now += 0.033
        controller._vz_est_m_s = 0.0
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.TRACK
    assert out.thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-3)


def test_course_leg_sink_response_stays_off_max_clamp():
    # F96: a centered-gate sink is answered by the SINGLE tracker term
    # plus the bounded trim — the old stack (arrest + trim wound to its
    # cap + PD) summed past the max clamp, the other half of F95's
    # clamp-to-clamp bang.  The F95 trace's own sink excursion (vz
    # -0.40 with the gate near center) must NOT reach 0.34.
    out = _converged_gate_one_vertical(-0.40)
    assert out.thrust < 0.34
    assert out.thrust > SPAWN_SUPPORT  # the sink IS answered, just coherently


def test_clear_visual_correction_is_independent_of_integrated_altitude():
    # Integrated altitude is a supporting estimate and may drift.  It cannot
    # attenuate or veto a clear optical correction toward a low gate.
    def _low_gate_thrust(alt_est_m):
        controller = _tracked_controller(_track("A", 0.0, 0.30, scale=0.20))
        _promote_to_gate_one(controller)
        controller._alt_est_m = alt_est_m
        current = controller.current
        _settle_commit_passage_covariance(current)
        current.y_axis.p = 0.30
        current.raw_y = 0.30
        current.y_axis.v = 0.0
        current.scale_axis.p = -2.5
        current.scale_axis.v = 0.10
        current.outer_log_scale = -2.5
        now = 100.10
        out = None
        for _ in range(15):
            now += 0.033
            controller._vz_est_m_s = 0.0
            current.last_measurement_s = now
            current.last_x_measurement_s = now
            current.last_y_measurement_s = now
            out = _command(controller, now, pitch=SPAWN_PITCH)
        assert controller.state is CleanCourseState.TRACK
        assert controller._last_vertical_motion.control_authority > 0.0
        return out

    biased_low = _low_gate_thrust(-0.10)
    nominal = _low_gate_thrust(2.0)
    assert biased_low.thrust < SPAWN_SUPPORT
    assert biased_low.thrust == pytest.approx(nominal.thrust, abs=1e-9)


def test_optical_vertical_baseline_does_not_weaken_near_the_plane():
    # F167's near-plane cap reduced correction while Gate 1's high miss was
    # worsening.  Fresh outer-y keeps one bounded position baseline at both
    # ranges; outer closure/TTC shapes rate urgency rather than attenuating it.
    # F98: the ramp reads outer_log_scale (COMMIT's own proximity
    # signal) — F97 keyed on the filtered hypothesis scale, which lagged
    # (-1.67) while the gate already engulfed the frame.  This test
    # deliberately leaves the filtered scale behind.
    controller = _tracked_controller(_track("A", 0.0, 0.30, scale=0.20))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 2.0
    current = controller.current
    _settle_commit_passage_covariance(current)
    current.y_axis.p = 0.30  # F96's low-sitting gate at the plane
    current.raw_y = 0.30
    current.y_axis.v = 0.03  # expansion * y: dilation only
    current.scale_axis.p = -1.67  # filtered hypothesis LAGS...
    current.scale_axis.v = 0.10
    current.outer_log_scale = -0.9  # ...while raw proximity is at the plane
    now = 100.10
    out = None
    for _ in range(15):
        now += 0.033
        controller._vz_est_m_s = 0.10
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.TRACK
    # Fresh visual motion owns the trajectory; IMU remains only a bounded
    # innovation around that motion, never a second full damping copy.
    assert out.thrust < SPAWN_SUPPORT
    assert controller._last_vertical_imu_delta < 0.0
    assert abs(controller._last_vertical_imu_delta) <= (
        controller.config.vertical_imu_max_opposition_fraction
        * max(
            abs(controller._last_vertical_visual_delta),
            abs(controller._last_vertical_motion_delta),
        )
        + 1e-12
    )
    # Far away the same full 0.30 position miss applies.
    far = _tracked_controller(_track("A", 0.0, 0.30, scale=0.20))
    _promote_to_gate_one(far)
    far._alt_est_m = 2.0
    far_current = far.current
    _settle_commit_passage_covariance(far_current)
    far_current.y_axis.p = 0.30
    far_current.raw_y = 0.30
    far_current.y_axis.v = 0.03  # expansion * y: dilation only
    far_current.scale_axis.p = -2.5
    far_current.scale_axis.v = 0.10
    far_current.outer_log_scale = -2.5  # beyond the ramp start: full 0.5 cap
    far_now = 100.10
    far_out = None
    for _ in range(15):
        far_now += 0.033
        far._vz_est_m_s = 0.10
        far_current.last_measurement_s = far_now
        far_current.last_x_measurement_s = far_now
        far_current.last_y_measurement_s = far_now
        far_out = _command(far, far_now, pitch=SPAWN_PITCH)
    assert far._last_vertical_imu_delta < 0.0
    assert abs(far._last_vertical_imu_delta) <= (
        far.config.vertical_imu_max_opposition_fraction
        * max(
            abs(far._last_vertical_visual_delta),
            abs(far._last_vertical_motion_delta),
        )
        + 1e-12
    )
    assert far_out.thrust == pytest.approx(out.thrust, abs=1e-12)
    assert out.thrust <= SPAWN_SUPPORT - 0.02


def test_raw_closure_brakes_when_the_filtered_rate_lags():
    # F99 (20260730T164633Z-visual-course-cb4e1b9e): the closure governor
    # read only the Kalman scale_axis.v, which lags ~1 s on a growing
    # track — F98 braked only incidentally (misalignment) and arrived at
    # +1.7..+3.5 log/s.  The raw outer-bbox EMA sees a hot approach within
    # a few frames.  F166 keeps the control estimate between the two outer
    # filters while their agreement grows; it no longer max-fuses them.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.08))
    samples, _now, _scale = _replay_outer_growth(
        controller,
        track_id="A",
        x=0.0,
        y=0.0,
        start_scale=0.08,
        closure_rate_s=math.log(1.05) / 0.033,
        now_s=100.033,
        frames=8,
    )
    out = samples[-1][0]
    assert controller.state is CleanCourseState.TRACK
    # The raw signal converged to the true hot closure...
    assert controller.current.outer_expansion_rate > 0.5
    for _out, raw, filtered, control, _demand, _active in samples:
        assert min(raw, filtered) <= control <= max(raw, filtered)
    assert samples[-1][1] > samples[-1][2]  # filtered outer rate still lags
    # ...and coherent motion crosses brake entry before the filter catches up.
    assert controller._pre_cross_brake_active
    assert out.target_pitch_rad < SPAWN_PITCH - 0.05
    assert out.target_pitch_rad > (
        SPAWN_PITCH + controller.config.pre_cross_brake_pitch_rad
    )


def test_steady_track_keeps_raw_closure_calm():
    # F99 guard: a stationary-size gate must not trip the raw-signal
    # brake — the EMA reads ~0 and the advance/brake blend is unchanged
    # from the filtered-only behavior.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.08))
    now = 100.10
    for frame in range(12):
        now += 0.033
        controller.observe(
            _update([_track("A", 0.0, 0.0, scale=0.08)], frame_id=10 + frame),
            now_s=now,
        )
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert abs(controller.current.outer_expansion_rate) < 0.05
    assert not controller._pre_cross_brake_active


def test_closure_governor_demands_energy_reduction_early():
    # F101 (20260730T173407Z-visual-course-7a862549): the flat 0.35 log/s
    # target permits 3+ m/s at leg start (0.35 log/s at 8-10 m) — more
    # than custody-compatible attitude braking can kill inside the leg.
    # F100's gate-1 leg held pb=1 mid-leg and still ran away to ~1.2
    # log/s at the plane (COMMIT expansion veto, blind structure strike).
    # The target now ramps down with range: at the F100 mid-leg state
    # (outer log -1.90, closure 0.43) the ramped target is 0.17, the
    # governor continuously demands a meaningful fraction of the one F125
    # brake reference rather than leaving the attitude at spawn.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.15))
    controller._alt_est_m = 2.0
    now = 100.033
    # Record a genuinely persistent successor before authoritative promotion.
    for frame in range(20):
        now += 0.033
        controller.observe(
            _update(
                [
                    _track("A", 0.0, 0.0, scale=0.15),
                    _track("B", 0.0, 0.0, scale=0.15),
                ],
                frame_id=100 + frame,
            ),
            now_s=now,
        )
        _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.note_race(
        gate_index=1, race_boot_ms=2500, now_s=now + 0.001
    )
    samples, _now, _scale = _replay_outer_growth(
        controller,
        track_id="B",
        x=0.0,
        y=0.0,
        start_scale=0.15,
        closure_rate_s=0.43,
        now_s=now + 0.001,
        frames=25,
        frame_id=200,
    )
    out = samples[-1][0]
    assert samples[-1][3] > controller.config.closure_far_target_rate_s
    assert 0.40 < samples[-1][4] < controller.config.pitch_brake_enter_demand
    assert not controller._pre_cross_brake_active
    assert out.target_pitch_rad < SPAWN_PITCH - 0.03


def test_blind_hold_tracks_zero_vz_when_fh_trusted():
    # With no fresh optical motion, the supporting IMU damping opposes an
    # inherited climb without inventing a visual request, trim, or margin.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.20))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 2.0
    current = controller.current
    current.y_axis.p = 0.0
    current.raw_y = 0.0
    current.scale_axis.p = -1.6
    current.scale_axis.v = 0.10
    now = 100.10
    out = None
    for _ in range(30):  # ~1 s of the F97 leg-start plateau geometry
        now += 0.033
        controller._vz_est_m_s = 0.36  # the held climb
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        # y stays STALE: vertical unqualified, the F98 blind-hold path
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.TRACK
    assert not out.vertical_qualified
    assert out.thrust == pytest.approx(
        SPAWN_SUPPORT - controller.config.vertical_imu_damping_gain * 0.36,
        # The single collective filter approaches the unchanged blind target
        # continuously rather than stepping to it.
        abs=3e-3,
    )


def test_far_qualified_low_gate_visual_path_is_not_double_damped_by_imu():
    # F90 (20260730T134602Z-visual-course-6e302725): the gate-1 leg
    # inherited vz +0.18 and a 0.024 support trim from the credited
    # gate-0 wait, then climbed vz +0.3..+0.4 for 1.3 s with the gate
    # vertically CENTERED (ey ~+0.03) — the one-sided arrest required a
    # climb COMMAND (ey < 0), the 0.5 m/s governor only caps rate, and
    # the ey PD is ~zero at center, so nothing bled the climb.  The F14
    # latch then pinned it 1.5 s more, the gate fell out the frame
    # bottom, and the recovery dove into the ground (id 1002).  With the
    # gate at/below center ANY positive vz is energy away from the aim:
    # the |ey|-scaled allowance binds in both directions.
    def _gate_low_thrust(vz_m_s):
        controller = _tracked_controller(_track("A", 0.0, 0.03, scale=0.20))
        _promote_to_gate_one(controller)
        controller._alt_est_m = 2.0  # honest altitude (floor quiet)
        current = controller.current
        current.x_axis.p = 0.0
        current.raw_x = 0.0
        current.y_axis.p = 0.03  # gate AT/slightly BELOW center: no climb intent
        current.raw_y = 0.03
        current.y_axis.v = 0.003  # expansion * y: dilation only
        current.scale_axis.p = -1.6
        current.scale_axis.v = 0.10  # settled closure
        now = 100.10
        out = None
        for _ in range(15):  # converge the slews/governors
            now += 0.033
            controller._vz_est_m_s = vz_m_s
            current.last_measurement_s = now
            current.last_x_measurement_s = now
            current.last_y_measurement_s = now
            out = _command(controller, now, pitch=SPAWN_PITCH)
        assert controller.state is CleanCourseState.TRACK
        return out

    climbing = _gate_low_thrust(0.30)
    settled = _gate_low_thrust(0.0)
    # Fresh outer position+image motion owns the descent trajectory.  The
    # integrated vz estimate may strengthen that same sign only inside the
    # bounded innovation envelope; it is never summed as a second full copy.
    assert climbing.thrust <= settled.thrust
    assert settled.thrust - climbing.thrust < 0.004
    assert climbing.thrust < SPAWN_SUPPORT


def test_clear_far_visual_miss_cannot_be_reversed_by_imu_damping():
    def _approach_thrust(ey, vz_m_s, ticks=5):
        controller = _tracked_controller(_track("A", 0.0, ey, scale=0.20))
        _promote_to_gate_one(controller)
        controller._alt_est_m = 2.0  # honest altitude
        current = controller.current
        _settle_commit_passage_covariance(current)
        current.x_axis.p = 0.0
        current.raw_x = 0.0
        current.y_axis.p = ey
        current.raw_y = ey
        current.y_axis.v = 0.10 * ey  # expansion * y: dilation only
        current.scale_axis.p = -1.6
        current.scale_axis.v = 0.10  # settled closure (brake quiet)
        current.scale_axis.vv = 0.02**2
        now = 100.10
        out = None
        for _ in range(ticks):
            now += 0.033
            controller._vz_est_m_s = vz_m_s
            current.last_measurement_s = now
            current.last_x_measurement_s = now
            current.last_y_measurement_s = now
            out = _command(controller, now, pitch=SPAWN_PITCH)
        assert controller.state is CleanCourseState.TRACK
        return out

    # With no visual miss, the supporting IMU sink damping asks for climb.
    centered = _approach_thrust(0.0, -0.40)
    assert centered.thrust == pytest.approx(
        SPAWN_SUPPORT + 0.12 * 0.40, abs=1e-9
    )
    # Once the low-gate intercept is visually clear, that correction owns the
    # direction.  Opposing IMU damping is bounded and cannot turn descent into
    # climb; a larger miss retains at least as much downward authority.
    low = _approach_thrust(0.15, -0.40)
    very_low = _approach_thrust(0.30, -0.40)
    assert very_low.thrust < low.thrust < SPAWN_SUPPORT


def test_gate0_qualified_imu_damping_continues_through_near_plane():
    # F85 (20260730T123020Z-visual-course-34c8dd71): gate 0 arrived at
    # censorship climbing +0.45 m/s; the aperture fit had died to clipping,
    # so COMMIT could not arm (the F83 entry cap never ran), and the
    # credible-loss exact-zero coast converted the climb into a ballistic
    # apex inside the frame — the drone fell into gate 0's LOWER panel
    # (id 1001, no credit).  F82 died the same way at +0.64 (top bar).
    # Fresh outer-y owns the optical path on gate 0 too.  An established climb
    # is damped near and far, but that supporting term cannot reverse the clear
    # climb correction or take collective below tilt support.
    def _gate0_thrust(vz_m_s, log_scale, *, y=-0.13):
        scale = math.exp(log_scale)
        controller = _tracked_controller(
            _track("A", 0.0, y, scale=scale)
        )
        controller._alt_est_m = 2.0  # honest altitude (floor quiet)
        now = 100.10
        out = None
        for frame_id in range(3, 18):
            now += 0.033
            controller.observe(
                _update(
                    [_track("A", 0.0, y, scale=scale)],
                    frame_id=frame_id,
                ),
                now_s=now,
            )
            controller._vz_est_m_s = vz_m_s
            out = _command(controller, now, pitch=SPAWN_PITCH)
        assert controller.state is CleanCourseState.TRACK
        return out

    # Near plane and far away, opposing climb damping may reduce but never
    # veto the fresh high-gate correction.
    climbing = _gate0_thrust(0.45, -0.70)
    settled = _gate0_thrust(0.0, -0.70)
    assert SPAWN_SUPPORT < climbing.thrust <= settled.thrust
    far_climbing = _gate0_thrust(0.45, -2.20)
    far_settled = _gate0_thrust(0.0, -2.20)
    assert SPAWN_SUPPORT < far_climbing.thrust <= far_settled.thrust

    # With no visual miss there is no opposing path request, so the same
    # bounded IMU term remains effective on both sides of the range schedule.
    assert _gate0_thrust(0.45, -0.70, y=0.0).thrust < SPAWN_SUPPORT
    assert _gate0_thrust(0.45, -2.20, y=0.0).thrust < SPAWN_SUPPORT


def test_no_alt_floor_latch_overrides_blind_search():
    # Low estimated altitude cannot arm a second SEARCH owner.  The same
    # IMU-damping target remains constant instead of winding or ratcheting.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.20))
    _promote_to_gate_one(controller)
    now = 100.10
    controller._enter_search(now)
    controller._alt_est_m = 0.2  # F91's blind low-altitude geometry
    # F91's SEARCH was BLIND: age the retained hypothesis's measurements
    # past the freshness horizon so the deleted latch would arm on the
    # very first tick of this scenario on the parent.
    current = controller.current
    if current is not None:
        current.last_measurement_s = now - 1.0
        current.last_x_measurement_s = now - 1.0
    first = None
    out = None
    for tick in range(30):  # ~1 s of the blind SEARCH sink
        now += 0.033
        controller._vz_est_m_s = -0.40
        out = _command(controller, now, pitch=SPAWN_PITCH)
        if tick == 0:
            first = out.thrust
    assert out.state is CleanCourseState.SEARCH
    expected = (
        SPAWN_SUPPORT + controller.config.vertical_imu_damping_gain * 0.40
    )
    assert first == pytest.approx(expected, abs=1e-9)
    assert out.thrust == pytest.approx(first, abs=1e-9)


def test_commit_entry_beats_censorship_onset_at_minus_1p2():
    # F54 (20260729T235858Z-visual-course-c92d42ce): censorship of the aim
    # track began ~0.2 s after outer_log_scale crossed -0.9, so the F53
    # (-0.9 proximity, 0.30 s sustain) calibration never entered and the
    # hover trap repeated into gate 1's structure.  At the F54 calibration
    # (-1.2 proximity, 0.10 s sustain) the commit latches well inside the
    # fresh window: measurements that go censored 0.6 s after the -1.2
    # crossing (the trace had ~0.7 s) still commit.
    controller = _commit_controller()
    controller.current.outer_log_scale = -1.15  # just past the -1.2 bound
    now = 100.10
    entered_elapsed = None
    for tick in range(30):  # ~1.0 s
        now += 0.033
        if tick < 18:  # fresh uncensored measurements for the first 0.6 s
            controller.current.last_measurement_s = now
            controller.current.last_x_measurement_s = now
            controller.current.last_y_measurement_s = now
        # else: censored — the stamps freeze (see _update_hypothesis)
        _command(controller, now, pitch=SPAWN_PITCH)
        if entered_elapsed is None and controller.state is CleanCourseState.COMMIT:
            entered_elapsed = now - 100.10
    assert controller.state is CleanCourseState.COMMIT
    assert entered_elapsed is not None
    # Latched within ~0.2 s of the -1.2 crossing — under the old 0.30 s
    # sustain (and -0.9 threshold) the 0.6 s censorship onset would have
    # killed the fresh-uncensored window before entry.
    assert entered_elapsed < 0.6 - 0.3


def test_commit_entry_requires_fresh_uncensored_both_axes():
    # 2026-07-30: entry requires CURRENT-frame (<=0.06 s) uncensored
    # measurements on BOTH axes — never a 0.30 s-old axis, only smoothed
    # state, or the outer-bbox corridor.  A stale x-axis (0.40 s old, but
    # inside the 0.5 s x-steer horizon so the main law keeps steering
    # normally) blocks entry.
    stale = _commit_controller()
    now = 100.10
    for _ in range(15):
        now += 0.033
        stale.current.last_measurement_s = now
        stale.current.last_y_measurement_s = now
        stale.current.last_x_measurement_s = now - 0.40
        _command(stale, now, pitch=SPAWN_PITCH)
    assert stale.state is CleanCourseState.TRACK
    # Even a 0.10 s-old axis — fresh enough for the old 0.30 s entry
    # window — cannot authorize a crossing.
    aged = _commit_controller()
    now = 100.10
    for _ in range(15):
        now += 0.033
        aged.current.last_measurement_s = now
        aged.current.last_y_measurement_s = now
        aged.current.last_x_measurement_s = now - 0.10
        _command(aged, now, pitch=SPAWN_PITCH)
    assert aged.state is CleanCourseState.TRACK
    # Censored axes never refresh the measurement stamp; a hypothesis with
    # no uncensored x measurement at all can never commit.
    censored = _commit_controller()
    now = 100.10
    for _ in range(15):
        now += 0.033
        censored.current.last_measurement_s = now
        censored.current.last_y_measurement_s = now
        censored.current.last_x_measurement_s = NEVER_MEASURED_S
        _command(censored, now, pitch=SPAWN_PITCH)
    assert censored.state is CleanCourseState.TRACK


def test_commit_entry_requires_alignment_proximity_and_aperture():
    # |ex| beyond the crossing bound: no commit.
    off_axis = _commit_controller()
    off_axis.current.x_axis.p = 0.30
    _drive_commit_window(off_axis, 100.10)
    assert off_axis.state is CleanCourseState.TRACK
    # Far range (outer log scale below the F54 -1.2 entry bound): the
    # sustain timer never starts.
    far = _commit_controller()
    far.current.outer_log_scale = -1.30
    _drive_commit_window(far, 100.10)
    assert far.state is CleanCourseState.TRACK
    # Gate 0 uses the same crossing state, but a track without a usable inner
    # aperture cannot authorize it.
    gate0 = _tracked_controller(_track("A", 0.10, 0.05))
    gate0._alt_est_m = 2.0
    gate0.current.outer_log_scale = -0.50
    _drive_commit_window(gate0, 100.10)
    assert gate0.state is CleanCourseState.TRACK


def test_commit_law_steers_fresh_holds_stale_and_bounds_vertical():
    controller = _commit_controller()
    out, now = _drive_commit_window(controller, 100.10)
    assert controller.state is CleanCourseState.COMMIT
    controller._prev_target_roll = 0.20  # pre-wound bank to unwind
    controller._prev_target_pitch = SPAWN_PITCH - 0.15  # braking attitude
    # F55 fast slew: the advance attitude must actually be reached — the
    # first tick alone moves ~0.033 rad (the generic 0.30 rad/s slew moved
    # 0.0099/tick and never arrived across F54's whole 2 s commit).
    now += 0.033
    controller.current.last_measurement_s = now
    controller.current.last_x_measurement_s = now
    controller.current.last_y_measurement_s = now
    out = _command(controller, now, pitch=SPAWN_PITCH)
    assert out.target_pitch_rad > SPAWN_PITCH - 0.15 + 0.025
    for _ in range(24):
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        controller.current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    # F150: while x is fresh, COMMIT consumes the same continuous outer-range
    # gain as TRACK and pending credit.  At this near-plane fixture the
    # ex=+0.10 request is yaw-capped and asks for 0.125 rad bank.
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)
    assert out.target_roll_rad == pytest.approx(
        0.50
        * controller._course_steer_gain(controller.current)
        * controller._lateral_intercept_reference_x,
        abs=1e-9,
    )
    # COMMIT carries the exact TRACK pitch admitted at the transition.  It
    # must not replace that path with a phase-owned +0.15 advance endpoint.
    assert out.target_pitch_rad == pytest.approx(
        controller._commit_pitch_target_rad, abs=1e-9
    )
    assert out.target_pitch_rad < SPAWN_PITCH + 0.15
    # COMMIT uses the same outer-owned position baseline and bounded
    # motion/projection decomposition as TRACK.
    motion = controller._last_vertical_motion
    path = controller._vertical_path_observation(
        controller.current, now_s=now, pitch_rad=SPAWN_PITCH
    )
    expected_position, expected_rate, expected_projection = (
        _expected_vertical_trajectory_terms(controller, path, motion)
    )
    assert controller._last_vertical_position_delta == pytest.approx(
        expected_position, abs=1e-12
    )
    assert controller._last_vertical_motion_delta == pytest.approx(
        expected_rate + expected_projection, abs=1e-12
    )
    assert controller._last_vertical_visual_delta == pytest.approx(
        expected_position
        + expected_rate
        + expected_projection
        + controller._last_vertical_direction_delta,
        abs=1e-12,
    )
    assert controller._last_vertical_collective_target == pytest.approx(
        controller._last_vertical_support
        + controller._last_vertical_visual_delta
        + controller._last_vertical_imu_delta,
        abs=1e-12,
    )
    assert out.thrust == pytest.approx(
        controller._last_vertical_collective_target, abs=1e-9
    )
    baseline_thrust = out.thrust
    # A larger downward error lowers the one target, but the carried
    # collective cannot jump to it in one control tick.
    controller.current.y_axis.p = 0.50
    now += 0.033
    controller.current.last_measurement_s = now
    controller.current.last_y_measurement_s = now
    out = _command(controller, now, pitch=SPAWN_PITCH)
    descend_target = controller._last_vertical_collective_target
    path = controller._vertical_path_observation(
        controller.current, now_s=now, pitch_rad=SPAWN_PITCH
    )
    expected_capped_position, expected_rate, expected_projection = (
        _expected_vertical_trajectory_terms(
            controller, path, controller._last_vertical_motion
        )
    )
    assert controller._last_vertical_position_delta == pytest.approx(
        expected_capped_position, abs=1e-12
    )
    assert controller._last_vertical_motion_delta == pytest.approx(
        expected_rate + expected_projection, abs=1e-12
    )
    assert descend_target == pytest.approx(
        max(
            controller.config.min_thrust,
            controller._last_vertical_support
            + expected_capped_position
            + expected_rate
            + expected_projection
            + controller._last_vertical_direction_delta
            + controller._last_vertical_imu_delta,
        ),
        abs=1e-12,
    )
    assert descend_target < out.thrust < baseline_thrust
    assert baseline_thrust - out.thrust < 0.01
    controller.current.y_axis.p = 1.0
    now += 0.033
    controller.current.last_measurement_s = now
    controller.current.last_y_measurement_s = now
    deeper = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller._last_vertical_collective_target == pytest.approx(
        descend_target, abs=1e-12
    )
    assert descend_target < deeper.thrust < out.thrust
    # Reversing the visual setpoint changes direction continuously rather
    # than activating a separate climb margin.
    controller.current.y_axis.p = -1.0
    now += 0.033
    controller.current.last_measurement_s = now
    controller.current.last_y_measurement_s = now
    rebound = _command(controller, now, pitch=SPAWN_PITCH)
    climb_target = controller._last_vertical_collective_target
    rebound_path = controller._vertical_path_observation(
        controller.current, now_s=now, pitch_rad=SPAWN_PITCH
    )
    rebound_position, rebound_rate, rebound_projection = (
        _expected_vertical_trajectory_terms(
            controller, rebound_path, controller._last_vertical_motion
        )
    )
    assert controller._last_vertical_position_delta == pytest.approx(
        rebound_position, abs=1e-12
    )
    assert climb_target == pytest.approx(
        controller._last_vertical_support
        + rebound_position
        + rebound_rate
        + rebound_projection
        + controller._last_vertical_direction_delta
        + controller._last_vertical_imu_delta,
        abs=1e-12,
    )
    assert deeper.thrust < rebound.thrust < climb_target
    assert rebound.thrust - deeper.thrust < 0.012
    # A fresh gross horizontal miss invalidates the certified safe tube.  It
    # must return custody to TRACK on this same tick instead of keeping a
    # latched COMMIT trajectory that can no longer fit through the corridor.
    controller.current.y_axis.p = 0.05
    controller.current.x_axis.p = 0.80
    now += 0.033
    controller.current.last_measurement_s = now
    controller.current.last_x_measurement_s = now
    controller.current.last_y_measurement_s = now
    out = _command(controller, now, pitch=SPAWN_PITCH)
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)
    assert controller.state is CleanCourseState.TRACK
    assert controller._commit_pitch_target_rad is None
    assert controller._last_commit_admission.status == (
        "corridor-known/not-contained"
    )
    assert out.target_pitch_rad < SPAWN_PITCH + 0.15

    # F62/F63: once x goes STALE/censored the commit steers the PREDICTED
    # hypothesis with the same range gain — heading-hold committed the residual drift
    # (F61 clipped the left post) and F62's half-gain derate
    # under-corrected (crossed -0.22 left); the prediction tracked the
    # real bearing through the blackout.  F150 keeps the physical outer-range
    # authority continuous across fresh and stale steering.
    controller, _outputs, now = _public_safe_commit_controller(now_s=200.0)
    recorded_entry_pitch = _F163_SAFE_PASSAGE_ROWS[8][8][1]
    for _ in range(3):
        now += 0.033
        out = _command(controller, now, pitch=recorded_entry_pitch)
    controller.current.x_axis.p = 0.10
    controller.current.raw_x = 0.10
    for _ in range(15):
        now += 0.033
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.COMMIT
    # Heading and lateral intercept each retain a continuous filtered
    # reference.  Yaw follows bearing while bank follows predicted plane miss.
    heading_reference = controller._turn_reference_x
    intercept_reference = controller._lateral_intercept_reference_x
    assert 0.05 < heading_reference <= 0.105
    gain = controller._course_steer_gain(controller.current)
    assert out.yaw_rate_rad_s == pytest.approx(
        min(0.15, 0.90 * gain * heading_reference), abs=1e-9
    )
    assert out.target_roll_rad == pytest.approx(
        0.50 * gain * intercept_reference, abs=1e-9
    )


def test_commit_unsafe_optical_motion_revokes_without_imu_direction_reversal():
    controller = _commit_controller()
    _, now = _drive_commit_window(controller, 100.10)
    assert controller.state is CleanCourseState.COMMIT

    # Here the centered image is moving down toward a low plane intercept,
    # while the supporting IMU estimate asks for the opposite correction.
    # Fresh proof that the tube escaped the safe corridor must revoke COMMIT;
    # after custody returns to TRACK, IMU damping may reduce authority but
    # cannot reverse the clear visual descent request.
    controller._vz_est_m_s = -0.80
    controller.current.y_axis.p = 0.0
    controller.current.raw_y = 0.0
    controller.current.y_axis.v = 0.30
    out = None
    for _ in range(30):
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        controller.current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)

    assert controller.state is CleanCourseState.TRACK
    assert controller._last_commit_admission.status == (
        "corridor-known/not-contained"
    )
    motion = controller._last_vertical_motion
    assert motion.physical_rate_norm_s == pytest.approx(0.30, abs=1e-9)
    assert motion.intercept_error > motion.intercept_std
    assert out.thrust < SPAWN_SUPPORT
    assert controller.config.min_thrust <= out.thrust <= controller.config.max_thrust


def test_commit_stale_y_relaxes_continuously_to_zero_vz_reference():
    # Frozen climb-side image evidence ages out and loses optical authority.
    # With a zero IMU-rate estimate, the carried collective relaxes toward
    # support without a mode step.
    controller, _outputs, now = _public_safe_commit_controller(now_s=300.0)
    assert controller.state is CleanCourseState.COMMIT
    # Let the last real camera frame become stale before perturbing the held
    # filter.  Stale prediction is not fresh unsafe proof and may not revoke a
    # legitimately sustained crossing lease.
    recorded_entry_pitch = _F163_SAFE_PASSAGE_ROWS[8][8][1]
    for _ in range(3):
        now += 0.033
        _command(controller, now, pitch=recorded_entry_pitch)
    controller.current.y_axis.p = -0.50
    y_stamp = controller.current.last_y_measurement_s
    thrusts = []
    stale_thrusts = []
    for _ in range(25):
        now += 0.033
        out = _command(controller, now, pitch=SPAWN_PITCH)
        thrusts.append(out.thrust)
        if now - y_stamp > controller.config.vertical_qualify_max_age_s:
            stale_thrusts.append(out.thrust)
    assert controller.state is CleanCourseState.COMMIT
    assert len(stale_thrusts) > 5
    trough = min(range(len(stale_thrusts)), key=stale_thrusts.__getitem__)
    assert trough <= 3
    assert all(
        after >= before - 1e-12
        for before, after in zip(
            stale_thrusts[trough:], stale_thrusts[trough + 1 :]
        )
    )
    assert abs(stale_thrusts[-1] - SPAWN_SUPPORT) < abs(
        stale_thrusts[0] - SPAWN_SUPPORT
    )
    assert max(
        abs(after - before) for before, after in zip(thrusts, thrusts[1:])
    ) < 0.01

    # Fresh low-gate evidence reuses the same owner and eventually asks below
    # support; there is no instantaneous band or overlay command.
    fresh = []
    for frame in range(20):
        now += 0.033
        low_gate = _f163_trace_track(
            track_id=controller.current.track_id,
            outer_center=(0.0, 0.50),
            outer_span=(0.30, 0.50),
        )
        controller.observe(
            _update([low_gate], frame_id=9300 + frame), now_s=now
        )
        fresh.append(_command(controller, now, pitch=SPAWN_PITCH).thrust)
    assert fresh[-1] < SPAWN_SUPPORT
    assert all(after <= before + 1e-12 for before, after in zip(fresh, fresh[1:]))


def test_commit_timeout_drops_hypothesis_and_searches():
    # ~3.0 s without authoritative credit: arrest and SEARCH, with the
    # hypothesis DROPPED — the innovation gate permanently rejects the true
    # gate while the frozen hypothesis lives (association lock-out).
    controller = _commit_controller()
    out, now = _drive_commit_window(controller, 100.10)
    assert controller.state is CleanCourseState.COMMIT
    for _ in range(100):  # ~3.3 s > the 3.0 s commit window
        now += 0.033
        if controller.current is not None:  # dropped at the timeout
            controller.current.last_measurement_s = now
            controller.current.last_x_measurement_s = now
            controller.current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.SEARCH
    assert controller.current is None
    assert out.state is CleanCourseState.SEARCH


def test_commit_credit_promotion_exits_to_next_leg():
    controller = _commit_controller()
    out, now = _drive_commit_window(controller, 100.10)
    assert controller.state is CleanCourseState.COMMIT
    # The COMMIT observe branch keeps the successor fresh for credit.
    now += 0.033
    controller.observe(
        _update(
            [
                _track("B", 0.10, 0.05, scale=0.60),
                _track("C", -0.20, 0.0, scale=0.08),
            ],
            frame_id=30,
        ),
        now_s=now,
    )
    assert controller.state is CleanCourseState.COMMIT
    controller._track_first_seen_s["C"] = now - 1.0
    promoted = controller.note_race(
        gate_index=2, race_boot_ms=3000, now_s=now + 0.02
    )
    assert promoted
    assert controller.gate_index == 2
    assert controller.state is CleanCourseState.TRACK
    assert controller.current.track_id == "C"


def test_promotion_rejects_successor_with_unmeasured_x_axis():
    # F40: the promoted successor was a left-edge-clipped splinter of the
    # just-crossed gate's frame whose x-axis had never been measured
    # (NEVER_MEASURED_S fails every freshness horizon).  Promotion must
    # refuse it and fall to SEARCH instead of adopting a garbage aim point.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller.observe(
        _update(
            [
                _track("A", 0.0, 0.0),
                _track("B", 0.30, 0.05, scale=0.05, clipping=FrameEdge.LEFT),
            ],
            frame_id=3,
        ),
        now_s=100.08,
    )
    assert controller.successor is not None
    assert controller.successor.last_x_measurement_s == NEVER_MEASURED_S
    # Old enough to pass the F42 persistence check: the never-measured
    # x-axis (F41) is what must reject this successor.
    controller._track_first_seen_s["B"] = 100.08 - 1.0
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=2500, now_s=100.10
    )
    assert promoted  # the gate increment is still authoritative
    assert controller.state is CleanCourseState.SEARCH
    assert controller.current is None


def test_successor_prefers_persistent_track_over_newborn_debris():
    # F42 (20260729T201743Z-visual-course-1e24b6d2): a bottom-left debris
    # splinter out-confidenced the real gate halves (0.62-0.71 vs 0.42-0.54)
    # and the pure max-confidence rule adopted it at promotion.  Age beats
    # confidence: a newborn high-confidence track must not steal the
    # successor slot from a persistent lower-confidence one.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    now = 100.10
    frame = 10
    for _ in range(20):  # ~0.66 s: P persists across the approach
        now += 0.033
        controller.observe(
            _update(
                [
                    _track("A", 0.0, 0.0, scale=0.10),
                    _track("P", 0.35, -0.28, scale=0.05, confidence=0.45),
                ],
                frame_id=frame,
            ),
            now_s=now,
        )
        frame += 1
    assert controller.successor.track_id == "P"
    # Newborn debris with HIGHER confidence (uncensored, so only the F42
    # age rule can reject it): the persistent track keeps the slot.
    now += 0.033
    controller.observe(
        _update(
            [
                _track("A", 0.0, 0.0, scale=0.10),
                _track("P", 0.35, -0.28, scale=0.05, confidence=0.45),
                _track("D", -0.96, -0.92, scale=0.05, confidence=0.70),
            ],
            frame_id=frame,
        ),
        now_s=now,
    )
    assert controller.successor.track_id == "P"


def test_promotion_has_no_second_persistence_gate_for_measured_successor():
    # The successor selector still prefers persistent tracks when alternatives
    # exist, but authoritative race credit preserves the already-selected,
    # measured bearing even if association assigned it a fresh id at crossing.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller.observe(
        _update(
            [_track("A", 0.0, 0.0), _track("B", 0.30, 0.05, scale=0.05)],
            frame_id=3,
        ),
        now_s=100.08,
    )
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=2500, now_s=100.10
    )
    assert promoted
    assert controller.state is CleanCourseState.TRACK
    assert controller.current.track_id == "B"


def test_unmeasurable_x_hypothesis_cannot_hold_track():
    # F42 anti-deadlock: the adopted debris splinter's x-axis could never be
    # measured, and the F41 x-steer gate froze yaw/roll at 0 for 0.8 s until
    # the splinter died on its own.  A never-measured-x current older than
    # UNMEASURED_X_FORCE_SEARCH_S must force SEARCH at the next observe.
    controller = _tracked_controller(
        _track("A", 0.40, 0.0, scale=0.10, clipping=FrameEdge.RIGHT)
    )
    assert controller.state is CleanCourseState.TRACK
    assert controller.current.last_x_measurement_s == NEVER_MEASURED_S
    controller.current.created_s = 100.10  # adoption time for this scenario
    now = 100.10
    for frame in range(22):  # ~0.73 s of censored matches: still TRACK
        now += 0.033
        controller.observe(
            _update(
                [_track("A", 0.40, 0.0, scale=0.10, clipping=FrameEdge.RIGHT)],
                frame_id=20 + frame,
            ),
            now_s=now,
        )
    assert controller.state is CleanCourseState.TRACK
    now += 0.033  # ~0.76 s: past the bound, the next observe forces SEARCH
    controller.observe(
        _update(
            [_track("A", 0.40, 0.0, scale=0.10, clipping=FrameEdge.RIGHT)],
            frame_id=50,
        ),
        now_s=now,
    )
    assert controller.state is CleanCourseState.SEARCH


def test_search_reacquisition_falls_back_to_newborn_tracks():
    # F42: persistence is a PREFERENCE at re-acquisition, not a requirement —
    # with only newborn candidates the nearest-to-bearing pick still adopts
    # one, or SEARCH could never recover from a cold start.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    now = 100.10
    for _ in range(40):
        now += 0.033
        controller.observe(_update([], frame_id=9), now_s=now)
        if controller.state is CleanCourseState.SEARCH:
            break
    assert controller.state is CleanCourseState.SEARCH
    now += 0.033
    controller.observe(
        _update([_track("N", 0.10, 0.0)], frame_id=50),  # newborn id only
        now_s=now,
    )
    assert controller.state is CleanCourseState.TRACK
    assert controller.current.track_id == "N"


def test_near_plane_reacquisition_refuses_newborn_splinter():
    # F74 (20260730T071207Z-visual-course-d38f869e): three flights in a row
    # died adopting a one-tick newborn SPLINTER as the aim the instant the
    # real track went engulfing at the gate plane (F70: 0008 at
    # (+0.44,+0.89); F72: 0011 at (+0.49,+0.55); F73: 0010 at (+0.39,+0.67))
    # — each followed by a blind wander into structure.  A FRESH engulfing
    # anchor proves "AT the plane", where a brand-new track is debris:
    # refuse adoption for the persistence window (SEARCH/PREDICT holds the
    # derotated hypothesis).  Same-id re-adoption and the far-range newborn
    # fallback (the F42 test above, no fresh anchor) are untouched.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    now = 100.10
    splinter = _track("S", 0.40, 0.60, scale=0.07)  # corner debris, age 0
    controller._last_engulfing_anchor_s = now
    assert controller._select_search_reacquisition([splinter], now) is None
    # The same track adopts once it has persisted through the window.
    controller._track_first_seen_s["S"] = now - 1.0
    assert controller._select_search_reacquisition([splinter], now) is splinter
    # No fresh anchor (far range): the newborn fallback still fires.
    controller._last_engulfing_anchor_s = now - 10.0
    newborn = _track("N", 0.10, 0.0)
    assert controller._select_search_reacquisition([newborn], now) is newborn


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


def test_finite_bounded_output_across_states():
    controller = _tracked_controller(_track("A", 0.05, 0.0, scale=0.20))
    now = 100.10
    outputs = []
    scenarios = [
        [_track("A", 0.05, 0.0, scale=0.20)],
        [_track("A", 0.45, 0.30, scale=0.35)],
        [_track("A", -0.60, -0.40, scale=0.05, clipping=FrameEdge.LEFT)],
        [],
        [_track("A", 0.0, 0.0, scale=0.60)],
        [],
    ]
    for frame, tracks in enumerate(scenarios * 3):
        now += 0.033
        controller.observe(_update(tracks, frame_id=30 + frame), now_s=now)
        controller.note_race(
            gate_index=controller.gate_index,
            race_boot_ms=3000 + frame,
            now_s=now,
        )
        outputs.append(_command(controller, now + 0.005))
    for output in outputs:
        values = (
            output.target_roll_rad,
            output.target_pitch_rad,
            output.yaw_rate_rad_s,
            output.thrust,
        )
        assert all(math.isfinite(value) for value in values)
        assert abs(output.yaw_rate_rad_s) <= 0.15 + 1e-9
        assert output.thrust == 0.0 or 0.21 <= output.thrust <= 0.34
        assert abs(output.target_roll_rad) <= 0.35 + 1e-9
        # F49: pitch targets are offsets from the spawn attitude, so the
        # envelope is spawn-relative (pre-cross reaches spawn - 0.15).
        assert (
            SPAWN_PITCH - 0.35
            <= output.target_pitch_rad
            <= SPAWN_PITCH + 0.35
        )
        if output.thrust == 0.0:
            assert output.state is CleanCourseState.COAST_FOR_CREDIT


def test_final_clamp_is_the_single_transparent_envelope():
    runtime = _test_runtime()
    command = clamp_final_command(
        _Command(0.40, -0.40, 0.30, 0.40), runtime=runtime
    )
    assert command.roll_rate == pytest.approx(0.25)
    assert command.pitch_rate == pytest.approx(-0.25)
    assert command.yaw_rate == pytest.approx(0.15)
    assert command.thrust == pytest.approx(0.34)
    command = clamp_final_command(_Command(0.0, 0.0, 0.0, 0.05), runtime=runtime)
    assert command.thrust == pytest.approx(0.21)
    # Exact zero is reserved semantics and passes through unchanged.
    command = clamp_final_command(_Command(0.0, 0.0, 0.0, 0.0), runtime=runtime)
    assert command.thrust == 0.0


# ---------------------------------------------------------------------------
# Loop integration (fake host): skipped send, promotion, success summary
# ---------------------------------------------------------------------------


class _Command:
    def __init__(self, roll_rate, pitch_rate, yaw_rate, thrust):
        self.roll_rate = float(roll_rate)
        self.pitch_rate = float(pitch_rate)
        self.yaw_rate = float(yaw_rate)
        self.thrust = float(thrust)


class _Race:
    def __init__(self, gate=0, boot=1000, finished=False):
        self.active_gate_index = gate
        self.sim_boot_time_ms = boot
        self.race_finished = finished


class _Host:
    def __init__(self, update):
        self.race = _Race()
        self.update = update
        self.sent = []
        self.events = []
        self.estimate = SimpleNamespace(
            orientation=SimpleNamespace(to_euler=lambda: (0.0, 0.0, 0.0)),
            body_rates=(0.0, 0.0, 0.0),
        )
        self.recorder = SimpleNamespace(
            emit=lambda event, **fields: self.events.append(event)
        )
        self._visual_course_summary = None
        self.yaw_calibration_profile_evidence = {"sha256": "test"}
        self.adapter = SimpleNamespace(race_status=self.race)
        self.ticks = 0
        self.skipped = 0
        self.script = None

    def _sample(self):
        self.ticks += 1
        if self.script is not None:
            self.script(self)

    @property
    def _visual_latest_tracker_update(self):
        return self.update

    def _watchdog(self, *, require_target=True, **_kwargs):
        assert require_target is False

    async def _wait_for_next_flight_command_slot(self):
        import time

        return time.monotonic()

    async def _send_flight_command(self, command, *, wire_race_gate_index):
        import time

        time.sleep(0)  # keep the loop honest about async boundaries
        self.sent.append((command, wire_race_gate_index))
        if getattr(self, "skip_next", False):
            self.skip_next = False
            self.skipped += 1
            return _SKIPPED
        return None

    def _record_tick(self, *_args, **_kwargs):
        return None


_SKIPPED = object()


def _test_runtime():
    import time

    return CleanCourseRuntime(
        safety_abort_type=RuntimeError,
        monotonic=time.monotonic,
        sleep=asyncio.sleep,
        next_control_deadline=lambda previous, now: max(previous, now),
        attitude_rate_command=_fake_pd,
        attitude_rate_command_type=_Command,
        validate_command=_validate,
        skipped_result=_SKIPPED,
        control_period_s=0.0,
        hard_duration_s=20.0,
        max_yaw_rate_rad_s=0.15,
        max_command_rate_rad_s=0.25,
        min_thrust=0.21,
        max_thrust=0.34,
    )


def _fake_pd(estimate, *, target_roll_rad, target_pitch_rad, thrust,
             intercept_response_authority=0.0):
    # Mirror the live loop: pitch kp scales 0.5 -> 2.0 with intercept
    # authority; roll is always 2.0; the 0.25 wire cap bounds rates.
    pitch_kp = 0.5 + intercept_response_authority * (2.0 - 0.5)
    roll, pitch, _yaw = estimate.orientation.to_euler()
    return _Command(
        max(-0.25, min(0.25, 2.0 * (target_roll_rad - roll))),
        max(-0.25, min(0.25, pitch_kp * (target_pitch_rad - pitch))),
        0.0,
        thrust,
    )


def _validate(command):
    values = (
        command.roll_rate,
        command.pitch_rate,
        command.yaw_rate,
        command.thrust,
    )
    assert all(math.isfinite(value) for value in values)
    assert max(abs(values[0]), abs(values[1]), abs(values[2])) <= 0.25 + 1e-9
    assert 0.0 <= command.thrust <= 0.35


def test_loop_skipped_send_promotes_and_finishes():
    update = _update(
        [_track("A", 0.05, 0.0), _track("B", 0.40, -0.10, scale=0.05)]
    )
    host = _Host(update)

    def script(host):
        if host.ticks == 3:
            # Race boundary advances after command generation: the atomic
            # send skips the obsolete setpoint instead of aborting.
            host.race.active_gate_index = 1
            host.race.sim_boot_time_ms = 1250
            host.skip_next = True
        if host.ticks == 8:
            host.race.race_finished = True

    host.script = script
    context = SimpleNamespace(
        initial_gate_x=322, initial_gate_y=174, initial_gate_area=6400
    )
    summary = asyncio.run(
        run_clean_course_stage(host, context, runtime=_test_runtime())
    )
    assert host.skipped == 1
    assert summary["success"] is True
    assert summary["race_finished"] is True
    assert summary["initial_gate_index"] == 0
    assert summary["final_gate_index"] == 1
    assert summary["maximum_authoritative_gate_index"] >= 1
    assert summary["authoritative_transitions"] == [
        {"from_gate_index": 0, "to_gate_index": 1}
    ]
    assert summary["yaw_calibration_profile"] == {"sha256": "test"}
    assert host._visual_course_summary is summary
    for command, _wire_index in host.sent:
        _validate(command)
        assert abs(command.yaw_rate) <= 0.15 + 1e-9


def test_loop_passes_estimator_accel_trust_to_vertical_owner():
    host = _Host(_update([_track("A", 0.0, -0.10, scale=0.20)]))
    host.estimate.accel_trust = 0.0
    host.estimate.horizontal_specific_force_mps2 = 0.0

    def script(loop_host):
        if loop_host.ticks == 5:
            loop_host.race.race_finished = True

    host.script = script
    controller = CleanCourseController(CleanCourseConfig())
    observed_trust = []
    original_command = controller.command

    def recording_command(**kwargs):
        observed_trust.append(kwargs.get("accel_trust"))
        return original_command(**kwargs)

    controller.command = recording_command
    context = SimpleNamespace(
        initial_gate_x=320, initial_gate_y=162, initial_gate_area=6400
    )
    summary = asyncio.run(
        run_clean_course_stage(
            host,
            context,
            runtime=_test_runtime(),
            controller=controller,
        )
    )

    assert summary["race_finished"] is True
    assert observed_trust
    assert observed_trust == [0.0] * len(observed_trust)
    assert controller._last_vertical_imu_trust == 0.0


def test_loop_coast_holds_exact_zero_then_accepts_credit():
    host = _Host(_update([_track("A", 0.0, 0.0, scale=0.50)]))
    # F102: the gate-0 hot-coast trigger is deleted — COAST arms only from
    # an armed COMMIT.  Inject the controller (same config the stage would
    # build) so the script can enter COMMIT just before the tick-3 close
    # loss and let the surviving COMMIT latch arm the coast.
    rt = _test_runtime()
    controller = CleanCourseController(
        CleanCourseConfig(
            min_thrust=rt.min_thrust,
            max_thrust=rt.max_thrust,
            max_yaw_rate_rad_s=rt.max_yaw_rate_rad_s,
            control_period_s=rt.control_period_s,
            spawn_pitch_rad=rt.spawn_pitch_rad,
        )
    )

    def script(host):
        if host.ticks == 2:
            import time

            controller.state = CleanCourseState.COMMIT
            controller._commit_entry_s = time.monotonic()
        if host.ticks == 3:
            host.update = _update([], frame_id=99)  # close crossing loses target
        if host.ticks == 6:
            # Authoritative credit during the bounded wait.
            host.race.active_gate_index = 1
            host.race.sim_boot_time_ms = 1250
        if host.ticks == 10:
            host.race.race_finished = True

    host.script = script
    context = SimpleNamespace(
        initial_gate_x=322, initial_gate_y=174, initial_gate_area=6400
    )
    summary = asyncio.run(
        run_clean_course_stage(
            host, context, runtime=_test_runtime(), controller=controller
        )
    )
    # 2026-07-30: the coast wait is EXACT WIRE ZERO — the PD is bypassed,
    # no support thrust, no leveling rates.  F72: exactly ONE zero send,
    # bounded by the send count, not a timeout.  Credit is still accepted
    # after the state exits.
    saw_powered = False
    zero_sends = []
    for command, _index in host.sent:
        if command.thrust > 0.05:
            saw_powered = True
        elif saw_powered and command.thrust == 0.0:
            zero_sends.append(command)
    assert len(zero_sends) == 1  # the single exact-zero credit-wait send
    for command in zero_sends:
        assert (
            command.roll_rate,
            command.pitch_rate,
            command.yaw_rate,
            command.thrust,
        ) == (0.0, 0.0, 0.0, 0.0)
    assert summary["final_gate_index"] == 1


def test_loop_coast_bypasses_the_pd_at_exact_zero():
    # F11 historically required exact-zero coast sends because the attitude
    # PD leaked nonzero rates at zero thrust; F26 moved the wait through
    # the PD at support thrust.  2026-07-30 contract correction: the wait
    # is exact wire zero again and the PD is BYPASSED — a tilted attitude
    # estimate cannot leak leveling rates or thrust onto the wire.  F102:
    # the gate-0 hot-coast trigger is deleted — COAST arms only from an
    # armed COMMIT, so inject the controller and enter COMMIT just before
    # the tick-3 close loss (same config the stage would build).
    host = _Host(_update([_track("A", 0.0, 0.0, scale=0.50)]))
    host.estimate = SimpleNamespace(
        orientation=SimpleNamespace(to_euler=lambda: (0.10, -0.08, 0.0)),
        body_rates=(0.0, 0.0, 0.0),
    )
    rt = _test_runtime()
    controller = CleanCourseController(
        CleanCourseConfig(
            min_thrust=rt.min_thrust,
            max_thrust=rt.max_thrust,
            max_yaw_rate_rad_s=rt.max_yaw_rate_rad_s,
            control_period_s=rt.control_period_s,
            spawn_pitch_rad=rt.spawn_pitch_rad,
        )
    )

    def script(host):
        if host.ticks == 2:
            import time

            controller.state = CleanCourseState.COMMIT
            controller._commit_entry_s = time.monotonic()
        if host.ticks == 3:
            host.update = _update([], frame_id=99)  # close crossing loses target
        if host.ticks == 6:
            # Authoritative credit during the bounded wait.
            host.race.active_gate_index = 1
            host.race.sim_boot_time_ms = 1250
        if host.ticks == 10:
            host.race.race_finished = True

    host.script = script
    context = SimpleNamespace(
        initial_gate_x=322, initial_gate_y=174, initial_gate_area=6400
    )
    summary = asyncio.run(
        run_clean_course_stage(
            host, context, runtime=_test_runtime(), controller=controller
        )
    )
    saw_powered = False
    zero_sends = []
    for command, _index in host.sent:
        if command.thrust > 0.05:
            saw_powered = True
        elif saw_powered and command.thrust == 0.0:
            zero_sends.append(command)
    assert len(zero_sends) == 1  # the single exact-zero credit-wait send
    for command in zero_sends:
        assert (
            command.roll_rate,
            command.pitch_rate,
            command.yaw_rate,
            command.thrust,
        ) == (0.0, 0.0, 0.0, 0.0)
    assert summary["final_gate_index"] == 1
    assert summary["success"] is True


@pytest.mark.parametrize("credit_tick", [5, 9, 14])
def test_loop_coast_emits_exactly_one_zero_send(credit_tick):
    # F72: the credible-crossing credit wait is exactly ONE wire-zero send,
    # bounded by the state/send count rather than any timeout — no matter
    # when the scheduler delivers the authoritative credit, the wire sees a
    # single 0/0/0/0 command.  Credit is still accepted after the state
    # exits (it is accepted in EVERY state).  F102: the gate-0 hot-coast
    # trigger is deleted — COAST arms only from an armed COMMIT, so inject
    # the controller (same config the stage would build) and enter COMMIT
    # just before the tick-3 close loss.
    host = _Host(_update([_track("A", 0.0, 0.0, scale=0.50)]))
    rt = _test_runtime()
    controller = CleanCourseController(
        CleanCourseConfig(
            min_thrust=rt.min_thrust,
            max_thrust=rt.max_thrust,
            max_yaw_rate_rad_s=rt.max_yaw_rate_rad_s,
            control_period_s=rt.control_period_s,
            spawn_pitch_rad=rt.spawn_pitch_rad,
        )
    )

    def script(host):
        if host.ticks == 2:
            import time

            controller.state = CleanCourseState.COMMIT
            controller._commit_entry_s = time.monotonic()
        if host.ticks == 3:
            host.update = _update([], frame_id=99)  # close crossing loses target
        if host.ticks == credit_tick:
            # Authoritative credit after the wait.
            host.race.active_gate_index = 1
            host.race.sim_boot_time_ms = 1250
        if host.ticks == credit_tick + 5:
            host.race.race_finished = True

    host.script = script
    context = SimpleNamespace(
        initial_gate_x=322, initial_gate_y=174, initial_gate_area=6400
    )
    summary = asyncio.run(
        run_clean_course_stage(
            host, context, runtime=_test_runtime(), controller=controller
        )
    )
    saw_powered = False
    zero_sends = []
    for command, _index in host.sent:
        if command.thrust > 0.05:
            saw_powered = True
        elif saw_powered and command.thrust == 0.0:
            zero_sends.append(command)
    assert len(zero_sends) == 1
    assert (
        zero_sends[0].roll_rate,
        zero_sends[0].pitch_rate,
        zero_sends[0].yaw_rate,
        zero_sends[0].thrust,
    ) == (0.0, 0.0, 0.0, 0.0)
    assert summary["final_gate_index"] == 1
    assert summary["success"] is True


# ---------------------------------------------------------------------------
# 2026-07-30 contract corrections: yaw command cap, unified crossing entry,
# continuous descent floor, TRACK lateral trim
# ---------------------------------------------------------------------------


def test_yaw_command_output_never_exceeds_the_0p15_production_cap():
    # The v3 profile's 0.50 rad/s was measured PLANT RESPONSE, not command
    # authority.  Controller output clamps to +/-0.15 rad/s on both signs;
    # the wire-side layers are covered by the final-clamp test above and
    # the runner's validate_command regression.
    assert CleanCourseConfig().max_yaw_rate_rad_s == pytest.approx(0.15)
    right = _tracked_controller(_track("A", 0.90, 0.0, scale=0.10))
    out = _command(right, 100.10)
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)
    left = _tracked_controller(_track("A", -0.90, 0.0, scale=0.10))
    out = _command(left, 100.10)
    assert out.yaw_rate_rad_s == pytest.approx(-0.15, abs=1e-9)


def test_commit_entry_holds_hot_closure_before_blackout_point_of_no_return():
    # Fast closure is not a scalar veto, but a contained approach still holds
    # while enough visible braking window remains.  At the first proximity
    # boundary, 0.50/s leaves 0.60 s before the -0.9 blackout regime, longer
    # than the modeled 0.50 s blackout horizon.
    closing = _commit_controller()
    closing.current.scale_axis.v = 0.50
    closing.current.scale_axis.p = -1.20
    closing.current.outer_log_scale = -1.20
    _drive_commit_window(closing, 100.10)
    assert closing.state is CleanCourseState.TRACK


def test_sustained_race_owned_error_is_not_integrated_out_of_turn_request():
    # F149: F148 stalled at x=-0.134 while the old trim integrated -0.050
    # and reduced the current-gate request to -0.084.  A persistent measured
    # bearing is a physical pursuit error, not a bias to ratchet toward zero.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 2.0
    controller.successor = None
    controller._turn_reference_x = None
    current = controller.current
    current.x_axis.p = 0.10
    current.raw_x = 0.10
    current.outer_log_scale = -2.5  # far-range gain remains 1.0
    now = 100.10
    for _ in range(60):  # ~2 s of sustained ex=+0.10
        now += 0.033
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
        assert out.yaw_rate_rad_s == pytest.approx(0.09, abs=1e-9)
        assert out.target_roll_rad > 0.0
    assert controller._turn_reference_x == pytest.approx(0.10, abs=1e-9)


def test_steering_gain_is_one_continuous_outer_range_schedule():
    # F150: F148 exposed a 0.097 rad/s one-tick yaw jump at the old
    # off/full near-plane switch, while F149 still stalled at x=-0.119 before
    # bottom censorship.  Every state now consumes this same continuous
    # existing range ramp; crossing the old threshold cannot switch 1->2.5.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    current = controller.current
    logs = [-2.10, -2.00, -1.80, -1.60, -1.40, -1.20, -1.10]
    gains = []
    yaws = []
    for log_scale in logs:
        current.outer_log_scale = log_scale
        gain = controller._course_steer_gain(current)
        gains.append(gain)
        yaws.append(
            controller._coordinated_turn_request(
                -0.04, steer_gain=gain, yaw_rad=0.0
            )[1]
        )

    assert gains[0] == pytest.approx(1.0)
    assert gains[-1] == pytest.approx(2.5)
    assert all(right >= left for left, right in zip(gains, gains[1:]))
    assert all(right <= left < 0.0 for left, right in zip(yaws, yaws[1:]))
    assert max(abs(right - left) for left, right in zip(yaws, yaws[1:])) < 0.02


def test_commit_close_loss_latches_exact_zero_credit_wait():
    # 2026-07-30: once a COMMIT crossing goes credibly blind at the plane
    # (fresh loss at commit range), the controller stops active blind
    # driving and enters the exact-zero authoritative credit wait — ONE
    # crossing policy, no parallel blind path.
    controller = _commit_controller()
    out, now = _drive_commit_window(controller, 100.10)
    assert controller.state is CleanCourseState.COMMIT
    now += 0.033
    controller.observe(_update([], frame_id=42), now_s=now)  # fresh close loss
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    output = _command(controller, now + 0.02)
    assert (
        output.target_roll_rad,
        output.target_pitch_rad,
        output.yaw_rate_rad_s,
        output.thrust,
    ) == (0.0, 0.0, 0.0, 0.0)
    # Authoritative credit during the wait still promotes.
    promoted = controller.note_race(
        gate_index=2, race_boot_ms=3000, now_s=now + 0.10
    )
    assert promoted
    assert controller.gate_index == 2
