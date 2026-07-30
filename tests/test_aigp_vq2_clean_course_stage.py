"""Behavior tests for the clean visual-course stage (architecture reset M2).

These tests assert envelope and directional behavior only: one global
vertical sign, decay-toward-support on vertical loss, a cut 0.30 x 0.40 s
launch boost, a disabled-then-tested gate-0 climb bias that never lifts the
aim point above image center, full-authority image-rate D bounded by a
symmetric IMU world-vertical-rate climb/descent governor with a
descent-regime hover feedforward (alive in TRACK, PREDICT, and SEARCH,
bypassed only by the coast support-hold latch), phantom
vertical rates zeroed when the accepted-y measurement
ages out (reseeded only by real measurements, never seeded by censored or
engulfing detections), degenerate engulfing detections rejected as
measurements but kept as bearing/existence anchors that block SEARCH,
expire when the camera frame freezes, and never come from a missed
(invisible) track, with a hard 1.5 s PREDICT stall cap that forces SEARCH
regardless of anchor state, a genuine nose-up post-credit brake armed by
every authoritative promotion until the successor is accepted and
vertically qualified (bounded by a 2.75 s timeout and a 2.0 s minimum
hold, slewed in at a dedicated 1.0 rad/s) with a qualification-
gated 0.5 m/s climb cap in the same window, a genuine nose-up pre-crossing
expansion brake (near window or sub-2.5 s expansion TTC, near-field gated
at -1.8, fast slew, lateral pursuit and vz governor alive, crossing
detection unsuppressed), a pre-gate-1 altitude floor
(vz_est-integrated alt_est clamped at >= -2.0 m, 0.7 -> 1.2 m hysteresis)
as a pure collective floor inside the vz governor (F55: the early-return
attitude/lateral override was deleted after three plane-region preemption
deaths), bounded to a 2.5 s latch with a 1.0 s above-release re-arm (F13), an fh inflow-regime gate
(sustained fh > 3.0 for 0.3 s freezes vz/alt integration, blocks floor
arming, suppresses the vz governor, holds support + 0.05 unqualified;
hysteresis release below 2.0), an edge-parked advance-stall cap forcing
SEARCH after 1.5 s without re-centering or approach progress, a coast
latch that holds level attitude at the tilt-compensated support
collective through the normal attitude PD (F26: the retired exact-zero
coast made every crossing ballistic), a raised
0.34 thrust envelope
under the runner's 0.35 hard abort,
verified yaw/roll directions, clipping uncertainty (not abort),
PREDICT->SEARCH on fresh empty frames, frozen-frame stalls that predict and
never coast, a real bounded yaw sweep, finite bounded output, absolute race
authority, the bounded crossing-coast wait on a fresh close loss, and one
final clamp.  They never assert exact internal event dictionaries, mode
sequences, or lineage identities.
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
    NEVER_MEASURED_S,
    PENDING_CREDIT_HOLD_S,
    ROTATION_COMP_FOCAL_NORM,
    CleanCourseConfig,
    CleanCourseController,
    CleanCourseRuntime,
    CleanCourseState,
    NavigationOutput,
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


def _command(controller, now, *, roll=0.0, pitch=0.0, yaw=None, a_up=None, fh=None):
    return controller.command(
        now_s=now,
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_rad=yaw,
        world_up_accel_m_s2=a_up,
        horizontal_specific_force_mps2=fh,
    )


def _blind_current(controller, now, age=1.0):
    """Age the current hypothesis's measurements past the F51 floor-arming
    freshness (and the x-steer horizon) so floor-mechanics tests drive the
    floor while blind — the F51 floor arms only without a live gate."""
    controller.current.last_measurement_s = now - age
    controller.current.last_x_measurement_s = now - age


# ---------------------------------------------------------------------------
# Vertical law
# ---------------------------------------------------------------------------


def test_identical_global_vertical_sign_at_every_gate():
    config = _config()
    thrusts = []
    controller = _tracked_controller(_track("A", 0.0, 0.20), config=config)
    now = 100.10
    for gate in (0, 1, 2):
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
    # Image-down positive error with the same geometry at every gate yields
    # the same signed correction relative to support (ONE GLOBAL SIGN).
    for thrust in thrusts:
        assert thrust < SUPPORT
    assert thrusts[0] == pytest.approx(thrusts[1], abs=1e-9)
    assert thrusts[1] == pytest.approx(thrusts[2], abs=1e-9)


def test_vertical_sign_is_the_gate0_minus_form_by_default():
    # pitch=SPAWN_PITCH: the F49 neutral (level-flight) attitude, so the
    # F50 attitude compensation is exactly zero in these PD-law checks;
    # the exact base is the tilt-compensated SPAWN_SUPPORT.
    controller = _tracked_controller(_track("A", 0.0, 0.20))
    output = _command(controller, 100.10, pitch=SPAWN_PITCH)
    assert output.thrust == pytest.approx(
        SPAWN_SUPPORT - 0.080 * 0.20, abs=1e-9
    )
    controller = _tracked_controller(_track("A", 0.0, -0.20))
    output = _command(controller, 100.10, pitch=SPAWN_PITCH)
    assert output.thrust == pytest.approx(
        SPAWN_SUPPORT + 0.080 * 0.20, abs=1e-9
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
    # End to end: a gate reading 0.24 low while braked 0.15 rad nose-up is
    # a LEVEL gate — the PD asks for the bare tilt-compensated support...
    braked = _tracked_controller(_track("A", 0.0, 0.24))
    out = _command(braked, 100.10, pitch=SPAWN_PITCH - 0.15)
    assert out.thrust == pytest.approx(
        SUPPORT / math.cos(SPAWN_PITCH - 0.15), abs=1e-9
    )
    # ...while the same reading at the spawn attitude really is low.
    level = _tracked_controller(_track("A", 0.0, 0.24))
    out = _command(level, 100.10, pitch=SPAWN_PITCH)
    assert out.thrust == pytest.approx(
        SPAWN_SUPPORT - 0.080 * 0.24, abs=1e-9
    )


def test_gate0_takeoff_boost_is_feedforward_only():
    # Boost-window behavior isolated from the gate-0 climb offset.  A far
    # track keeps the brake ceiling band out of the window (with the F49
    # 0.247 support the band top 0.287 would cap the 0.30 boost).
    config = CleanCourseConfig(gate0_climb_vertical_offset_norm=0.0)
    controller = _tracked_controller(
        _track("A", 0.0, 0.20, scale=0.05), config=config
    )
    boosted = _command(controller, 100.10)
    assert boosted.thrust == pytest.approx(config.launch_boost_thrust)
    after = _command(controller, 100.0 + config.launch_boost_duration_s + 0.05)
    assert after.thrust < SUPPORT  # closed-loop sign resumes unchanged


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
    saturated = _command(controller, 100.10).thrust
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
        thrusts.append(_command(controller, now + 0.005).thrust)
        now += 0.033
    assert all(math.isfinite(value) for value in thrusts)
    assert thrusts[-1] > saturated + 0.01  # not retained at saturation
    assert thrusts[-1] == pytest.approx(SUPPORT, abs=0.01)  # decayed to support
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


def test_gate0_climb_vertical_offset_is_bounded_feedforward():
    # 2026-07-29 analysis (Q1/Q4): cross gate 0 higher so gate 1 is first
    # seen with doubled top-edge margin.  A gate-0 target still ABOVE center
    # (ey < 0, image-down) yields collective above support; the bias stays
    # inside the thrust envelope and retires with the gate-0 phase.
    config = _config(gate0_climb_vertical_offset_norm=0.25)
    controller = _tracked_controller(_track("A", 0.0, -0.10), config=config)
    output = _command(controller, 100.10, pitch=SPAWN_PITCH)
    # e = -0.10 - 0.25 = -0.35 -> full climb correction above support.
    assert output.thrust == pytest.approx(
        SPAWN_SUPPORT + 0.080 * 0.35, abs=1e-9
    )
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
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=1250, now_s=100.14
    )
    assert promoted
    assert controller.state is CleanCourseState.TRACK
    controller._alt_est_m = 2.0  # honest altitude (floor quiet)
    output = _command(controller, 100.16, pitch=SPAWN_PITCH)
    assert output.gate_index == 1
    # Same centered target after promotion: the unbiased law holds support.
    assert output.thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)


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
    assert _command(far, 100.10, pitch=SPAWN_PITCH).thrust == pytest.approx(
        SPAWN_SUPPORT + 0.080 * (0.10 + 0.25), abs=1e-9
    )

    mid = _tracked_controller(
        _track("A", 0.0, -0.10, scale=math.exp(-1.295)), config=config
    )
    mid_offset = 0.25 * (-1.295 - (-0.80)) / (-1.79 - (-0.80))
    assert _command(mid, 100.10, pitch=SPAWN_PITCH).thrust == pytest.approx(
        SPAWN_SUPPORT + 0.080 * (0.10 + mid_offset), abs=1e-9
    )

    crossing = _tracked_controller(
        _track("A", 0.0, 0.0, scale=math.exp(-0.80)), config=config
    )
    assert _command(
        crossing, 100.10, pitch=SPAWN_PITCH
    ).thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)


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
    assert _command(
        centered, 100.10, pitch=SPAWN_PITCH
    ).thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)

    below = _tracked_controller(
        _track("A", 0.0, 0.10, scale=0.1667), config=config
    )
    output = _command(below, 100.10, pitch=SPAWN_PITCH)
    # Offset contributes nothing: only the plain e = +0.10 descent feedback.
    assert output.thrust == pytest.approx(
        SPAWN_SUPPORT - 0.080 * 0.10, abs=1e-9
    )
    assert output.thrust < SPAWN_SUPPORT


def test_vertical_rate_term_keeps_full_authority():
    # The flight-2 D-direction limiter was removed after flight
    # 20260729T094736Z-...-4dbe4b8c: it pinned collective at exactly support
    # through the decisive window (vz 2.7 m/s, t=1.31-1.72) because ey
    # hovered near zero.  Opposing P/D now yields full D authority; honest
    # climb-rate limiting is the IMU vz governor's job (tests below).
    config = _config(gate0_climb_vertical_offset_norm=0.25)
    controller = _tracked_controller(_track("A", 0.0, -0.10), config=config)
    controller.current.y_axis.v = 1.0  # strong rate opposing the P correction
    output = _command(controller, 100.10)
    assert output.vertical_qualified
    # e = -0.35 (P demands climb) but vy = +1.0: D fully reverses the
    # correction and the collective sags to the 0.21 floor.  Under the
    # removed limiter this was pinned at exactly SUPPORT.
    assert output.thrust == pytest.approx(0.21, abs=1e-9)
    assert output.thrust < SUPPORT - 0.03


def test_vz_governor_caps_collective_above_climb_cap():
    # Four gate-0 top-bar flights: bearing pursuit built unbounded vz
    # (2.8-3.35 m/s peaks vs a ~0.9 m/s requirement).  The IMU governor
    # removes K_VZ per m/s over the 1.0 m/s cap from the collective.
    controller = _tracked_controller(_track("A", 0.0, 0.0))  # e = 0, vy ~ 0
    controller._vz_est_m_s = 0.5  # below the cap: no effect
    assert _command(controller, 100.10, pitch=SPAWN_PITCH).thrust == pytest.approx(
        SPAWN_SUPPORT, abs=1e-9
    )
    controller._vz_est_m_s = 2.0  # 1.0 m/s over cap -> -0.03 collective
    assert _command(controller, 100.14, pitch=SPAWN_PITCH).thrust == pytest.approx(
        SPAWN_SUPPORT - 0.03, abs=1e-9
    )


def test_vz_governor_course_leg_damps_climb_continuously():
    # F68 (20260730T052644Z-visual-course-0d49884b): the gate-1 approach
    # climbed at vz +1.0 (the gate-0 1.0 m/s cap) through the final 1.5 s,
    # overshot ~0.4 m above the gate, and the aperture died at the frame
    # bottom.  Gate-1+ legs get a 0.5 m/s bound, SUBTRACTIVE from the PD
    # demand so the arrest is continuous (no min()-against-support
    # chatter); the gate-0 envelope is unchanged.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    helper = controller._governed_collective
    demand = SUPPORT + 0.07
    controller._vz_est_m_s = 0.5  # at the course bound: demand untouched
    assert helper(demand, SUPPORT) == pytest.approx(demand, abs=1e-9)
    controller._vz_est_m_s = 1.0  # 0.5 over -> demand - 0.10*0.5
    assert helper(demand, SUPPORT) == pytest.approx(demand - 0.05, abs=1e-9)
    # Continuous across the bound: sweeping vz moves collective smoothly
    # (the F64 descent-step bang-bang must not reappear on the climb side).
    previous = None
    steps = []
    for k in range(21):  # vz +0.40 .. +0.60 in 0.01 m/s steps
        controller._vz_est_m_s = 0.40 + 0.01 * k
        value = helper(demand, SUPPORT)
        if previous is not None:
            assert value <= previous + 1e-12  # monotone nonincreasing
            steps.append(previous - value)
        previous = value
    assert steps
    assert max(steps) < 0.005  # 0.10 * 0.01 = 0.001 per step, no jump


def test_vz_governor_applies_in_predict_and_search():
    # The governor is IMU-based precisely so vision loss cannot disable it.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.observe(_update([], frame_id=2), now_s=100.12)  # superseded
    assert controller.state is CleanCourseState.PREDICT
    controller._vz_est_m_s = 2.0
    assert _command(controller, 100.16, pitch=SPAWN_PITCH).thrust == pytest.approx(
        SPAWN_SUPPORT - 0.03, abs=1e-9
    )
    controller._enter_search(100.20)
    assert _command(controller, 100.22, pitch=SPAWN_PITCH).thrust == pytest.approx(
        SPAWN_SUPPORT - 0.03, abs=1e-9
    )


def test_vz_governor_floors_collective_below_descent_floor():
    # Flight 20260729T111003Z-visual-course-d52adcd4: a 6.1 s frozen-camera
    # stall blinded the loop while a_up ~= -1.9 m/s^2 sank it into a ground
    # graze; the climb-only governor did nothing.  F63 (5e550551) proved the
    # original -0.5/0.06/+0.025 floor too weak: an established -1.0..-1.5
    # sink ran ~5 s unarrested at thrust 0.30-0.33 (fast-regime hover ~=0.32)
    # and the drone passed under gate 1 into the floor.  F67: the F64
    # 0.10/m/s + 0.04 step bang-banged at the plane (F65's thrust alternated
    # 0.31 <-> 0.22 as vz_est straddled the floor), so the step is deleted
    # and the floor is purely proportional at 0.21/m/s below -0.35 m/s.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    helper = controller._governed_collective
    max_thrust = controller.config.max_thrust
    controller._vz_est_m_s = -0.35  # at the floor boundary: no effect
    assert helper(SUPPORT, SUPPORT) == pytest.approx(SUPPORT, abs=1e-9)
    controller._vz_est_m_s = -0.7  # 0.35 m/s below -> +0.0735 (~F64's +0.075)
    # support + 0.0735: inside the raised 0.34 envelope (F9/F10 headroom
    # restoration; the old 0.32 clamp clipped exactly this case).
    assert helper(SUPPORT, SUPPORT) == pytest.approx(SUPPORT + 0.0735, abs=1e-9)
    # By vz -1.0 the arrest saturates at max_thrust (raw support + 0.1365)
    # — the F63 sink ran unarrested at 0.30-0.33 while the fast-regime
    # hover is ~=0.32.
    controller._vz_est_m_s = -1.0
    assert helper(SUPPORT, SUPPORT) == pytest.approx(max_thrust, abs=1e-9)
    # Deep sinks saturate at max_thrust (flight 039186c8: the unclamped
    # floor boost exceeded the runner's 0.35 envelope abort in SEARCH).
    controller._vz_est_m_s = -1.7  # 1.35 m/s below -> +0.2835 raw
    assert helper(SUPPORT, SUPPORT) == pytest.approx(max_thrust, abs=1e-9)
    # The floor only raises collective, but the governed output is still
    # clamped: a higher command saturates at max_thrust as well.
    controller._vz_est_m_s = -1.0
    assert helper(0.35, SUPPORT) == pytest.approx(max_thrust, abs=1e-9)


def test_vz_descent_floor_raises_command_thrust():
    # Command level (F67 law): vz = -0.7 m/s lifts the emitted collective to
    # support + 0.0735 (proportional 0.21/m/s, no step); a deeper sink
    # (vz -1.0, raw 0.3835) clips at the 0.34 envelope top.
    controller = _tracked_controller(_track("A", 0.0, 0.0))  # e = 0, vy ~ 0
    controller._vz_est_m_s = -0.7
    assert _command(controller, 100.10).thrust == pytest.approx(
        SUPPORT + 0.0735, abs=1e-9
    )
    controller._vz_est_m_s = -1.0
    assert _command(controller, 100.14).thrust == pytest.approx(
        controller.config.max_thrust, abs=1e-9
    )


def test_vz_descent_floor_applies_in_predict_and_search():
    # The floor is IMU-based for the same reason as the climb cap: vision
    # loss (the d52adcd4 stall case) must not disable it.  F67 law: vz -0.7
    # -> support + 0.0735 proportional.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.observe(_update([], frame_id=2), now_s=100.12)  # superseded
    assert controller.state is CleanCourseState.PREDICT
    controller._vz_est_m_s = -0.7
    assert _command(controller, 100.16).thrust == pytest.approx(
        SUPPORT + 0.0735, abs=1e-9
    )
    controller._enter_search(100.20)
    assert _command(controller, 100.22).thrust == pytest.approx(
        SUPPORT + 0.0735, abs=1e-9
    )


def test_vz_phantom_sink_cannot_move_coast_exact_zero():
    # 2026-07-30 contract correction: a credible close loss outputs EXACT
    # wire zero (roll/pitch/yaw rates and thrust) for the bounded credit
    # wait — support-thrust coasting through the attitude PD is out of
    # contract.  A phantom sink must not change that: the coast is not a
    # governed control law at all.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
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


def test_vertical_unqualified_zeroes_phantom_rate_and_holds_support():
    # Flight 20260729T104947Z-visual-course-bc8c6003: a phantom vy (+0.38
    # norm/s, seeded as the gate sank through the frame) random-walked
    # unmeasured for 5.4 s and commanded an unrecoverable descent.  Once the
    # last accepted y measurement ages out, the retained rate is zeroed and
    # the collective holds tilt-compensated support.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller.current.y_axis.v = 0.38  # phantom rate from the crossing
    output = _command(controller, 100.45)  # last y measurement 0.42 s stale
    assert not output.vertical_qualified
    assert controller.current.y_axis.v == 0.0
    assert output.thrust == pytest.approx(SUPPORT, abs=1e-9)


def test_qualification_regain_reseeds_rate_from_real_measurement():
    # After the phantom zeroing, qualification returns only through real
    # measurements and the rate reseeds from them (never the stale value).
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller.current.y_axis.v = 0.38
    _command(controller, 100.45)  # ages out + zeroes the phantom rate
    assert controller.current.y_axis.v == 0.0
    now = 100.46
    for frame, y in enumerate((0.02, 0.04, 0.06, 0.08)):
        controller.observe(
            _update([_track("A", 0.0, y)], frame_id=10 + frame), now_s=now
        )
        now += 0.033
    output = _command(controller, now)
    assert output.vertical_qualified
    reseeded = controller.current.y_axis.v
    assert reseeded > 0.05  # reseeded from the descending measurements
    assert reseeded != pytest.approx(0.38, abs=1e-9)  # never the stale rate


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
    output = _command(controller, now + 0.02)
    # The censored creation box is not a y measurement.
    assert not output.vertical_qualified
    assert output.thrust == pytest.approx(SUPPORT, abs=1e-9)


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


def test_altitude_floor_triggers_and_releases_with_hysteresis():
    # F10/F11/F12: the final 6-10 s before gate 1 ran below 0.7 m with
    # thrust pinned into terrain.  The pre-gate-1 floor (alt_est integrated
    # from the governor's vz_est, seeded 0 at course start) bumps the
    # governed collective to support + margin until the release hysteresis
    # clears (F55: pure collective floor — the attitude/lateral law is
    # never overridden; the early-return override was deleted after the
    # F50/F51/F54 plane-region deaths).  F51: it arms only while blind
    # (driven here with aged measurements).
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 2.0
    controller._vz_est_m_s = -1.0  # a_up=None holds the estimate constant
    now = 100.10
    frame = 10
    out = None
    while controller._alt_est_m >= 0.7:  # ~40 ticks of integrated sink
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
        _blind_current(controller, now)  # F51: the floor arms only blind
        out = _command(controller, now)
        frame += 1
    assert controller._alt_floor_active
    assert out.yaw_rate_rad_s == 0.0
    # F64: sinking at -1.0 m/s saturates the strengthened descent floor at
    # the 0.34 envelope top (raw support + 0.065 + 0.04 = 0.352).
    assert out.thrust == pytest.approx(
        controller.config.max_thrust, abs=1e-9
    )
    # Hysteresis: climbing between 0.7 and 1.2 m keeps the floor active.
    controller._vz_est_m_s = 1.0
    while controller._alt_est_m < 1.0:
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
        _blind_current(controller, now)
        out = _command(controller, now)
        frame += 1
    assert controller._alt_floor_active
    assert out.yaw_rate_rad_s == 0.0
    # Above the 1.2 m release the normal pursuit law resumes.
    while controller._alt_est_m <= 1.2:
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
        out = _command(controller, now)
        frame += 1
    assert not controller._alt_floor_active
    assert out.yaw_rate_rad_s > 0.0  # x=+0.30 pursuit steers again


def test_altitude_floor_is_gated_to_gate_one():
    # The floor protects only the measured pre-gate-1 window: gate 0 keeps
    # the normal law at the same low altitude, and post-gate-1 flight is
    # unaffected (re-anchoring alt_est after gate 1 is a follow-up).
    controller = _tracked_controller(_track("A", 0.30, 0.0))
    controller._alt_est_m = 0.5
    out = _command(controller, 100.10)
    assert not controller._alt_floor_active
    assert out.yaw_rate_rad_s > 0.0  # normal x=+0.30 pursuit at gate 0
    _promote_to_gate_one(controller, now_s=100.12)
    # F51: the floor must NOT arm over the freshly promoted live gate — a
    # fresh accepted track is better altitude evidence than the sagged
    # integrator.
    out = _command(controller, 100.14)
    assert not controller._alt_floor_active
    assert out.yaw_rate_rad_s > 0.0  # gate-1 pursuit keeps steering
    # Blind at the same low altitude: the gate-1 window arms the floor.
    _blind_current(controller, 100.16)
    out = _command(controller, 100.16)
    assert controller._alt_floor_active
    assert out.yaw_rate_rad_s == 0.0
    assert controller.note_race(gate_index=2, race_boot_ms=3000, now_s=100.18)
    _command(controller, 100.20)
    assert not controller._alt_floor_active


def test_altitude_floor_respects_max_thrust():
    # F12's -4.3 m/s sink would demand support + 0.06*3.8 + 0.025 ~= 0.53;
    # the governor's internal clamp keeps the floored demand inside the
    # course envelope (0.34, below the runner's 0.35 hard abort).
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 0.5
    controller._vz_est_m_s = -4.3
    _blind_current(controller, 100.14)  # F51: the floor arms only blind
    out = _command(controller, 100.14)
    assert controller._alt_floor_active
    assert out.thrust == pytest.approx(controller.config.max_thrust, abs=1e-9)


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


def test_altitude_floor_latch_releases_unconditionally_and_rearms_after_hold():
    # F13 (trace 20260729T134958Z-visual-course-82d72cb5): a biased
    # estimator (alt_est -10.7 m, physically impossible) latched the floor
    # at t=5.16 and pinned the profile at full thrust for 4.2 s into
    # terrain.  An episode now releases unconditionally after 2.5 s and
    # re-arms only after alt_est has held above the release altitude for a
    # full second.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 0.5  # below the trigger; vz = 0 holds it there
    now = 100.10
    frame = 10
    _blind_current(controller, now)  # F51: the floor arms only blind
    _command(controller, now)
    assert controller._alt_floor_active
    # The latch must not pin the profile: it releases after 2.5 s even
    # though alt_est is still below the trigger.
    while now <= 100.10 + 2.6:
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
        _command(controller, now)
        frame += 1
    assert not controller._alt_floor_active
    assert controller._alt_floor_cooldown
    # Cooldown: still below the trigger, but no immediate re-trigger.
    now += 0.033
    _command(controller, now)
    assert not controller._alt_floor_active
    # Recover above the release altitude; the cooldown clears only after a
    # full second of sustained recovery.
    controller._vz_est_m_s = 1.0
    while controller._alt_est_m <= 1.2:
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
        _command(controller, now)
        frame += 1
    assert controller._alt_floor_cooldown  # not yet: needs 1.0 s above
    hold_start = now
    while now - hold_start < 1.0:
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
        _command(controller, now)
        frame += 1
    assert not controller._alt_floor_cooldown
    # Sink back below the trigger: the floor re-arms normally (blind, F51).
    controller._vz_est_m_s = -1.0
    out = None
    while controller._alt_est_m >= 0.7:
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
        _blind_current(controller, now)
        out = _command(controller, now)
        frame += 1
    assert controller._alt_floor_active
    assert out.thrust > 0.0


def test_alt_est_clamped_so_biased_integrator_cannot_deepen_floor():
    # F13's alt_est reached -10.7 m (physically impossible).  The estimate
    # is clamped below at -2.0 m: deeper than the 0.7-1.2 m guard band
    # ever needs, never deeper.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 0.0
    controller._vz_est_m_s = -10.0  # F13-scale biased sink
    now = 100.10
    _blind_current(controller, now)  # F51: the floor arms only blind
    for _ in range(10):  # unclamped integration would reach -3.3 m
        now += 0.033
        _command(controller, now)
    assert controller._alt_est_m == -2.0
    assert controller._alt_floor_active  # still guards inside the band


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


def test_alt_floor_never_arms_while_fh_untrusted_but_active_latch_times_out():
    # F14's self-locking loop: the governor pinned 0.34 on the phantom sink
    # and the floor flew biased-"level".  A biased estimate must never
    # START a floor episode; an already-active latch still times out.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 0.5
    now = 100.10
    frame = 10
    _blind_current(controller, now)  # F51: the floor arms only blind
    _command(controller, now)  # trusted: the floor arms normally
    assert controller._alt_floor_active
    # Going fh-untrusted must not clear the active latch...
    for _ in range(11):
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
        _command(controller, now, fh=6.0)
        frame += 1
    assert controller._fh_untrusted
    assert controller._alt_floor_active
    # ...and the latch still times out normally (alt frozen at 0.5).
    while now <= 100.10 + 2.6:
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
        _command(controller, now, fh=6.0)
        frame += 1
    assert not controller._alt_floor_active
    # With the regime still untrusted the floor cannot re-arm even though
    # alt_est is below the trigger.
    now += 0.033
    _command(controller, now, fh=6.0)
    assert not controller._alt_floor_active


def test_alt_floor_bumps_collective_only_and_keeps_the_law():
    # F55 (20260730T000535Z-visual-course-36fb03a4): the floor's
    # early-return override is DELETED after three plane-region deaths from
    # state preemption (F50 promotion freeze, F51 search park, F54 commit
    # climb-over).  The active floor is now a pure collective floor: TRACK
    # keeps its own attitude/lateral law and the governor only prevents
    # descent.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)  # current = B at x=+0.30
    controller._alt_est_m = 0.5
    _blind_current(controller, 100.10)  # F51: the floor arms only blind
    out = _command(controller, 100.10)
    assert controller._alt_floor_active
    # x aged with the rest: stale x -> heading hold, wings level (F40).
    assert out.yaw_rate_rad_s == 0.0
    assert out.target_roll_rad == 0.0
    # A fresh x measurement under the active floor restores the standard
    # x-qualified pursuit gains (yaw 0.90*ex=0.27, clamped to the 0.15
    # production command cap; roll 0.50 on ex=+0.30) — no override parks
    # the lateral law.
    now = 100.10
    for _ in range(20):  # slew the roll target out to its 0.15 command
        now += 0.033
        controller.current.last_x_measurement_s = now
        out = _command(controller, now)
    assert controller._alt_floor_active
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)
    assert out.target_roll_rad == pytest.approx(0.15, abs=1e-9)
    # The pitch is the TRACK law's own target (the ex=0.30 misalignment
    # brake pitches nose-up of spawn), NOT the deleted override's pinned
    # spawn attitude.
    assert out.target_pitch_rad < SPAWN_PITCH - 0.05
    # ...and the floor's only effect is the collective bump, exactly.
    assert controller._governed_collective(SUPPORT, SUPPORT) == pytest.approx(
        SUPPORT + 0.05, abs=1e-9
    )
    assert out.thrust > SUPPORT


def test_alt_floor_does_not_bump_commit_thrust():
    # F55: COMMIT opts out of the floor bump — a forced climb at the plane
    # flies the drone OVER the gate (F54's commit climb-over, thrust pinned
    # at support+margin while the gate slid out of the frame bottom).  The
    # latched floor stays active but the commit thrust is the band servo +
    # vz governor only.
    controller = _commit_controller()  # gate 1, near-plane aligned
    controller._alt_est_m = 0.5
    _blind_current(controller, 100.10)  # arm the floor latch blind
    _command(controller, 100.10)
    assert controller._alt_floor_active
    out, now = _drive_commit_window(controller, 100.10)
    assert controller.state is CleanCourseState.COMMIT
    assert controller._alt_floor_active  # latch held through entry
    # Band servo on ey=0.05 (-0.004); a floor bump would read support+0.05.
    assert out.thrust == pytest.approx(SPAWN_SUPPORT - 0.004, abs=1e-9)


def test_commit_timeout_fires_while_alt_floor_latched():
    # The F54 regression: the floor's early return preempted COMMIT for
    # 3.84 s, blocking the commit timeout until the floor released.  With
    # the override deleted the timeout fires on schedule at 3.0 s even
    # with the floor latched through the first 2.5 s of the commit.
    controller = _commit_controller()
    controller._alt_est_m = 0.5
    _blind_current(controller, 100.10)
    _command(controller, 100.10)
    assert controller._alt_floor_active
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


def test_unqualified_vertical_holds_support_plus_margin_while_fh_untrusted():
    # While fh-untrusted the camera is the only honest vertical channel;
    # when it is unqualified the hold is support + 0.05 — bare support
    # historically sinks for real at -0.8...-1.9 m/s, and F14 measured the
    # biased-regime deficit at ~0.05 collective (flight 99e093fa sank ~1 m
    # in 1.5 s on the old +0.02 hold).
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller.current.last_y_measurement_s = 99.0  # vertical unqualified
    now = 100.10
    for _ in range(11):
        now += 0.033
        _command(controller, now, fh=6.0)
    assert controller._fh_untrusted
    out = None
    for _ in range(30):  # the decay converges onto the hold
        now += 0.033
        out = _command(controller, now, fh=6.0)
    assert not out.vertical_qualified
    assert out.thrust == pytest.approx(SUPPORT + 0.05, abs=1e-3)


def test_descent_floor_cannot_fire_from_frozen_vz_est():
    # A frozen phantom sink must not fire the descent floor/feedforward:
    # with a qualified camera vertical the thrust is the plain PD output.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    controller._vz_est_m_s = -4.0  # frozen F14-scale phantom sink
    now = 100.10
    for frame in range(11):
        now += 0.033
        controller.observe(
            _update([_track("A", 0.0, 0.0)], frame_id=10 + frame), now_s=now
        )
        _command(controller, now, fh=6.0)
    assert controller._fh_untrusted
    now += 0.033
    controller.observe(
        _update([_track("A", 0.0, 0.0)], frame_id=21), now_s=now
    )
    out = _command(controller, now, fh=6.0)
    assert out.vertical_qualified  # fresh camera y
    # Governor suppressed: centered target -> PD asks for bare support, NOT
    # the descent-floor boost that would clamp at 0.34.  But the F21
    # fh-untrusted floor (support + margin) still bounds it from below —
    # while vz/alt are known lies nothing commands less (flight 9828d64c:
    # qualified-PD sagged to 0.254 at fh 7 and sank ~2 m).
    assert out.thrust == pytest.approx(SUPPORT + 0.05, abs=1e-9)


def test_closure_governor_full_brake_at_high_expansion_rate():
    # F31: the vision log-scale expansion rate is the only honest closure
    # signal (fh is a signless drag magnitude).  At/above the full-brake
    # rate the governor commands the gentle brake attitude exactly, at the
    # fast slew, with lateral pursuit and the vz governor alive.
    controller = _tracked_controller(_track("A", 0.20, 0.0, scale=0.10))
    controller.current.scale_axis.v = 0.7  # above CLOSURE_FULL_BRAKE_RATE_S
    now = 100.10
    out = None
    for _ in range(15):  # ~0.5 s: the fast slew attains the attitude
        now += 0.033
        # Fresh x measurements keep arriving (the F40 x-freshness gate
        # otherwise zeroes steering after 0.5 s without one).
        controller.current.last_x_measurement_s = now
        out = _command(controller, now)
    assert controller._pre_cross_brake_active
    assert out.state is CleanCourseState.TRACK
    assert out.target_pitch_rad == pytest.approx(
            # F49: spawn-relative — the -0.15 offset from the -0.31 spawn
            # attitude gives the effective -0.46 TRUE brake.
            controller.config.spawn_pitch_rad
            + controller.config.pre_cross_brake_pitch_rad,
            abs=1e-9,
        )
    assert now - 100.10 <= 0.5  # fast slew, not the generic 0.30 rad/s
    assert out.yaw_rate_rad_s > 0.0  # x=+0.20 pursuit stays alive
    assert out.thrust > 0.0  # the vz governor keeps the collective alive


def test_closure_governor_does_not_brake_below_target_rate():
    # Slow closure is free flight: below the 0.35/s target rate the
    # governor contributes nothing and the advance law still closes.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.current.scale_axis.v = 0.1
    out = _command(controller, 100.10)
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
    out = _command(controller, 100.10)
    assert not controller._pre_cross_brake_active
    # Advance law still closes: above the level (spawn) brake base.
    assert out.target_pitch_rad > SPAWN_PITCH


def test_closure_governor_is_a_continuous_blend():
    # Mid-band rate (0.475/s -> closure 0.5): the pitch target blends
    # halfway from the advance law (spawn base) toward the spawn-0.15
    # pre-cross brake attitude, without latching the fast-slew brake flag.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.current.scale_axis.v = 0.475
    now = 100.10
    out = None
    for _ in range(25):  # generic slew converges to the blended target
        now += 0.033
        out = _command(controller, now)
    assert not controller._pre_cross_brake_active
    assert (
        SPAWN_PITCH - 0.15 + 1e-9
        < out.target_pitch_rad
        < SPAWN_PITCH - 1e-9
    )


def test_misaligned_far_gate_brakes_and_climbs():
    # F35 (d25f23fe): a far off-axis top-clipped gate suppressed only the
    # advance law, leaving pitch ~level while speed built — and the
    # unqualified vertical hold at support + 0.065 merely hovered, so the
    # drone flew LEVEL and FAST into the high gate's lower structure.  The
    # misalignment brake must command the TRUE brake attitude, and the
    # high-gate climb margin must survive (the brake ceiling band only
    # applies in the near/crossing regime).
    controller = _tracked_controller(
        _track("A", 0.50, -0.80, scale=0.05, clipping=FrameEdge.TOP)
    )
    now = 100.10
    out = None
    for _ in range(25):  # fast brake slew converges
        now += 0.033
        # Fresh x measurements keep arriving (the F40 x-freshness gate
        # otherwise zeroes steering after 0.5 s without one).
        controller.current.last_x_measurement_s = now
        out = _command(controller, now)
    assert controller._pre_cross_brake_active
    assert out.target_pitch_rad == pytest.approx(
            # F49: spawn-relative — the -0.15 offset from the -0.31 spawn
            # attitude gives the effective -0.46 TRUE brake.
            controller.config.spawn_pitch_rad
            + controller.config.pre_cross_brake_pitch_rad,
            abs=1e-9,
        )
    # Top-clipped gate: y censored -> unqualified -> one-sided climb hold.
    assert not out.vertical_qualified
    # support + 0.12 exceeds the old +0.065 hover-equivalent margin even
    # after the 0.34 thrust clamp, and the far brake must NOT band it.
    assert out.thrust >= 0.33
    # Raised yaw gain: 0.9 * 0.50 pursuit, clamped to the 0.15 production
    # yaw command cap.
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)


def test_closure_governor_brakes_in_predict():
    # The governor applies in PREDICT too (the scale rate is filter state,
    # live even while the camera is briefly blind).
    controller = _tracked_controller(_track("A", 0.20, 0.0, scale=0.10))
    now = 100.10
    now += 0.033
    controller.observe(
        _update([_track("A", 0.20, 0.0, scale=0.10)], frame_id=39), now_s=now
    )  # fresh measurement so the dropout starts from TRACK
    controller.current.scale_axis.v = 0.7
    for frame in range(9):  # ~0.3 s without a measurement -> PREDICT
        now += 0.033
        controller.observe(_update([], frame_id=40 + frame), now_s=now)
        out = _command(controller, now)
    assert controller.state is CleanCourseState.PREDICT
    for _ in range(4):  # converge the fast brake slew inside the 0.5 s bound
        now += 0.033
        out = _command(controller, now)
    assert controller.state is CleanCourseState.PREDICT
    assert controller._pre_cross_brake_active
    assert out.target_pitch_rad == pytest.approx(
            # F49: spawn-relative — the -0.15 offset from the -0.31 spawn
            # attitude gives the effective -0.46 TRUE brake.
            controller.config.spawn_pitch_rad
            + controller.config.pre_cross_brake_pitch_rad,
            abs=1e-9,
        )


def test_pitch_offsets_follow_the_configured_spawn_attitude():
    # F49: every pitch target is spawn + offset, so a measured spawn
    # attitude other than the -0.31 default shifts all targets together.
    config = _config(spawn_pitch_rad=-0.20)
    controller = _tracked_controller(
        _track("A", 0.20, 0.0, scale=0.10), config=config
    )
    controller.current.scale_axis.v = 0.7  # full closure brake
    now = 100.10
    out = None
    for _ in range(15):  # fast slew attains the brake attitude
        now += 0.033
        controller.current.last_x_measurement_s = now
        out = _command(controller, now)
    assert controller._pre_cross_brake_active
    # -0.20 spawn + (-0.15) pre-cross offset = -0.35 effective brake.
    assert out.target_pitch_rad == pytest.approx(-0.35, abs=1e-9)


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
        out = _command(fragment, now)
    # Never the full advance offset on fragment evidence: capped at the
    # creep offset (spawn + 0.03) while centering.
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH + 0.03, abs=1e-9)
    # Centering authority (yaw) is untouched on a fragment.
    offset = _tracked_controller(_track("A", 0.30, 0.0, scale=0.06))
    offset.current.last_x_measurement_s = 100.10
    assert _command(offset, 100.10).yaw_rate_rad_s > 0.0
    # A confidently whole gate (span above the bound) advances fully.
    whole = _tracked_controller(_track("A", 0.0, 0.0, scale=0.20))
    now = 100.10
    out = None
    for _ in range(20):
        now += 0.033
        whole.current.last_x_measurement_s = now
        out = _command(whole, now)
    assert out.target_pitch_rad > SPAWN_PITCH + 0.04  # full advance, no creep cap


def test_crossing_loss_latches_coast_even_while_fh_untrusted():
    # F32 (8cc53db2): fh went untrusted from BRAKING drag at the engulfed
    # gate-0 plane, the fh guard blocked the coast latch, and the drone
    # flew blind into the frame in PREDICT.  A credible close crossing loss
    # coasts regardless.  2026-07-30: the coast is exact wire zero for the
    # bounded credit wait — no support-thrust PD hold.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller._fh_untrusted = True
    now = 100.10
    now += 0.033
    controller.observe(_update([], frame_id=20), now_s=now)  # fresh close loss
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    output = _command(controller, now + 0.02)
    assert output.thrust == 0.0


def test_brake_ceiling_band_bounds_collective_while_braking():
    # F32: the +0.15 brake stopped the drone horizontally short of gate 0
    # while the vertical channel (PD climb, high-gate bias, fh-untrusted
    # floor) carried it UP at +1.0 m/s into the gate's lower structure.
    # F34: the hard pin AT support removed all centering authority and the
    # drone crossed at 1.07 m into the bottom bar.  While braking, the
    # collective is confined to support +/- 0.04: the qualified PD keeps a
    # small climb budget, but the fh floor and high-gate bias cannot push
    # past the band top.
    # F37: ONE-SIDED — F32/F34/F36 all died with the gate HIGH; a gate
    # above image center gets an uncapped climb (the ceiling would be the
    # F36 starvation all over again), the sink floor still applies.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller._pre_cross_brake_active = True
    controller._fh_untrusted = False
    controller._vz_est_m_s = 0.0
    # A small qualified-PD correction inside the band passes through
    # (trusted, vz neutral): the vertical centering budget survives.
    assert controller._governed_collective(SUPPORT + 0.03, SUPPORT) == pytest.approx(
        SUPPORT + 0.03, abs=1e-9
    )
    # Deep sub-support demands are lifted to the band bottom (with the F49
    # 0.247 support the band bottom support-0.04 sits below the 0.21
    # min-thrust clamp, so the clamp is the effective floor):
    assert controller._governed_collective(0.20, SUPPORT) == pytest.approx(
        0.21, abs=1e-9
    )
    controller._fh_untrusted = True
    # fh floor (support + 0.05) is capped at the band top:
    assert controller._governed_collective(0.30, SUPPORT) == pytest.approx(
        SUPPORT + 0.04, abs=1e-9
    )
    assert controller._governed_collective(0.34, SUPPORT) == pytest.approx(
        SUPPORT + 0.04, abs=1e-9
    )
    # Released: the fh-untrusted floor applies again (a demand below the
    # support+0.05 floor is lifted to it).
    controller._pre_cross_brake_active = False
    assert controller._governed_collective(0.28, SUPPORT) == pytest.approx(
        SUPPORT + 0.05, abs=1e-9
    )

    # Gate HIGH (ey < -0.10): the climb side is uncapped even while
    # braking at the near gate; the sink floor still applies.  (Demand
    # support + 0.06 sits above the old band top but below the 0.34
    # envelope clamp.)
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
    controller.observe(_update([], frame_id=20), now_s=now)  # fresh close loss
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    output = _command(controller, now + 0.02)
    assert (
        output.target_roll_rad,
        output.target_pitch_rad,
        output.yaw_rate_rad_s,
    ) == (0.0, 0.0, 0.0)
    assert output.thrust == 0.0


def test_pre_cross_brake_relaxes_near_bottom_censor_with_hysteresis():
    # F51: the pre-cross brake attitude pitches the camera up and walks the
    # gate DOWN the physical FOV — the brake self-blinds at the near plane.
    # A fresh measurement at/past the 0.55 relax bound drops the pitch
    # target to level (vision custody outranks deceleration); the brake
    # resumes only below the 0.45 resume bound, and hysteresis holds the
    # last state between the bounds.
    controller = _tracked_controller(_track("A", 0.0, 0.60, scale=0.10))
    controller.current.scale_axis.v = 0.7  # rapid expansion too: full brake
    now = 100.10
    out = None
    for _ in range(15):
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        out = _command(controller, now)
    assert controller._pre_cross_brake_active  # ey=0.60: fully misaligned
    assert controller._brake_vision_relax
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    # Between the bounds (0.45 < ey < 0.55) the relaxed state holds.
    controller.current.y_axis.p = 0.50
    for _ in range(5):
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        out = _command(controller, now)
    assert controller._brake_vision_relax
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    # Below the resume bound the brake target (spawn - 0.15) resumes.
    controller.current.y_axis.p = 0.40
    for _ in range(15):
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        out = _command(controller, now)
    assert not controller._brake_vision_relax
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH - 0.15, abs=1e-9)
    # From the braking state, re-entering the band does NOT relax — the
    # brake holds through the hysteresis gap in both directions.
    controller.current.y_axis.p = 0.50
    for _ in range(15):
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        out = _command(controller, now)
    assert not controller._brake_vision_relax
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH - 0.15, abs=1e-9)


def test_near_plane_brake_relaxes_on_the_stale_hypothesis():
    # F65 (20260730T021149Z-visual-course-08f41050): AT the plane the F51
    # guard never fired — the gate sat at ey +0.43 (below the 0.55 fresh
    # bound) and censorship froze measurement freshness, so the -0.46
    # pre-cross brake attitude pitched the gate out of the FOV; the drone
    # wandered blind for ~7 s into the floor/structure.  Inside the commit
    # proximity regime the relax runs on the STALE derotated hypothesis at
    # the lower 0.30 bound and relaxes the pitch target to LEVEL.
    controller = _tracked_controller(_track("A", 0.30, 0.43, scale=0.50))
    controller.current.scale_axis.v = 0.7  # rapid expansion: full brake demand
    now = 100.10
    out = None
    for _ in range(20):
        now += 0.033
        # Measurements stay STALE (frozen at construction) — the far-range
        # freshness gate would never let the relax latch.
        out = _command(controller, now)
    assert controller._pre_cross_brake_active
    assert controller._brake_vision_relax
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    # Below the 0.20 near-plane resume bound the brake target resumes.
    controller.current.y_axis.p = 0.10
    for _ in range(20):
        now += 0.033
        out = _command(controller, now)
    assert not controller._brake_vision_relax
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH - 0.15, abs=1e-9)


def test_course_leg_near_plane_relax_engages_earlier_and_sticks():
    # F71 (20260730T060005Z-visual-course-f05911e4): on the gate-1 leg the
    # F65 0.30 engage bound sat just above the achieved hypothesis ey
    # (0.22-0.30 through the final second), so the relax never fired and the
    # brake attitude walked the gate into engulfing at the plane.  Course
    # legs (gate_index >= 1) engage at 0.18 and resume only below -0.10:
    # leveling the camera swings the image ~0.2 norm up, so a positive
    # resume band would chatter brake/level every slew cycle.  F73b: the
    # relax acts only once closure is arrested (expansion at/below the
    # governor target); while live closure is being arrested the brake
    # attitude outranks custody.
    controller = _tracked_controller(_track("A", 0.30, 0.22, scale=0.50))
    controller.gate_index = 1
    # Stopped and re-centering (expansion 0): custody relax, not arrest.
    # The brake demand comes from the ex 0.30 misalignment.
    controller.current.scale_axis.v = 0.0
    now = 100.10
    out = None
    for _ in range(20):
        now += 0.033
        out = _command(controller, now)
    # ey 0.22 is below the gate-0 0.30 bound but engages the course relax.
    assert controller._pre_cross_brake_active
    assert controller._brake_vision_relax
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    # Sticky through center: leveling swings the image up ~0.2 norm, so ey
    # settling near zero must NOT resume the brake.
    controller.current.y_axis.p = 0.05
    for _ in range(20):
        now += 0.033
        out = _command(controller, now)
    assert controller._brake_vision_relax
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    # The brake resumes only once the gate sits comfortably above center.
    controller.current.y_axis.p = -0.15
    for _ in range(20):
        now += 0.033
        out = _command(controller, now)
    assert not controller._brake_vision_relax
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH - 0.15, abs=1e-9)
    # Gate 0 keeps the F65 bounds: ey 0.22 does NOT relax there.
    gate0 = _tracked_controller(_track("A", 0.30, 0.22, scale=0.50))
    gate0.current.scale_axis.v = 0.7
    now0 = 100.10
    for _ in range(20):
        now0 += 0.033
        out0 = _command(gate0, now0)
    assert not gate0._brake_vision_relax
    assert out0.target_pitch_rad == pytest.approx(SPAWN_PITCH - 0.15, abs=1e-9)


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
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
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


def test_search_keeps_sweep_and_carries_floor_margin_under_alt_floor():
    # F52 (20260729T232037Z-visual-course-dedf1915): SEARCH entry at t=7.44
    # latched the alt-floor on the sagged gate-0 integrator and the
    # override parked the sweep (yaw=0/roll=0, thrust=support+margin) until
    # t=10.25 while the gates slid behind; the drone hit terrain at 13.7 s.
    # The floor must not preempt SEARCH: the sweep keeps full yaw authority
    # and the collective still carries the floor climb margin.
    controller = _tracked_controller(_track("A", 0.40, 0.0, scale=0.10))
    _promote_to_gate_one(controller)
    controller._enter_search(100.10)
    controller._alt_est_m = 0.5  # sagged integrator, below the trigger
    controller.last_reliable_bearing = (0.30, 0.0)  # zero vertical memory
    _blind_current(controller, 100.10)  # F51: the floor arms only blind
    out = _command(controller, 100.12)
    assert out.state is CleanCourseState.SEARCH
    assert controller._alt_floor_active
    # The sweep runs under the active floor (image-right bearing first).
    assert out.yaw_rate_rad_s > 0.0
    # ...and the collective carries the floor climb margin, not bare hold.
    assert out.thrust == pytest.approx(SUPPORT + 0.05, abs=1e-9)


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


def test_centered_close_loss_still_arms_coast():
    # The aligned case keeps the July-18 bounded credible-crossing wait: a
    # near-center fresh close loss arms COAST_FOR_CREDIT (the wait itself
    # is covered by the bounded/coast tests below).
    controller = _tracked_controller(_track("A", 0.10, -0.15, scale=0.50))
    controller.observe(_update([], frame_id=5), now_s=100.12)
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT


def test_crossing_loss_latches_coast_and_waits_for_newer_race_packet():
    config = _config()
    controller = _tracked_controller(
        _track("A", 0.0, 0.0, scale=0.50), config=config  # close crossing
    )
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
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
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
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
    controller.observe(_update([], frame_id=5), now_s=100.12)
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    out = _command(controller, 100.14)  # the single wire-zero send (F72)
    assert out.thrust == 0.0
    # Pending-credit window (~0.33 s << PENDING_CREDIT_HOLD_S): heading
    # held level, zero yaw/roll, governed altitude support — no sweep.
    for tick in range(10):
        out = _command(controller, 100.18 + 0.033 * tick)
        assert controller.state is CleanCourseState.SEARCH
        assert out.yaw_rate_rad_s == 0.0
        assert out.target_roll_rad == 0.0
        assert out.thrust > 0.0
    # Delayed credit inside the window: the retained left-side successor
    # is stale by construction (the crossing swallowed measurements) but
    # is adopted immediately — never a blind sweep on the new leg.
    promoted = controller.note_race(gate_index=1, race_boot_ms=2400, now_s=100.55)
    assert promoted
    assert controller.state is CleanCourseState.TRACK
    assert controller.current.track_id == "B"
    # A fresh post-credit frame of the adopted gate steers LEFT at once.
    controller.observe(
        _update([_track("B", -0.43, 0.05)], frame_id=6), now_s=100.57
    )
    out = _command(controller, 100.60)
    assert out.yaw_rate_rad_s < 0.0  # steers LEFT toward the successor
    # Bounded: with no credit the window expires and the sweep resumes.
    expiring = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    expiring.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    expiring.observe(_update([], frame_id=5), now_s=100.12)
    _command(expiring, 100.14)
    out = _command(expiring, 100.18)
    assert out.yaw_rate_rad_s == 0.0
    out = _command(expiring, 100.18 + PENDING_CREDIT_HOLD_S + 1.0)
    assert out.yaw_rate_rad_s != 0.0


def test_authoritative_promotion_event_never_vetoed_by_vision():
    controller = _tracked_controller(_track("A", 0.10, 0.0))
    controller.observe(
        _update([_track("A", 0.10, 0.0), _track("B", 0.30, 0.05)], frame_id=3),
        now_s=100.08,
    )
    # F42: promotion requires a persistent successor; seed the age.
    controller._track_first_seen_s["B"] = 100.08 - 1.0
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=2500, now_s=100.10
    )
    assert promoted
    assert controller.gate_index == 1
    assert controller.max_gate_index == 1
    # Credible cached successor promoted; no visual proof required.
    assert controller.state is CleanCourseState.TRACK
    assert controller.current.track_id == "B"

    # Without a credible successor the controller enters SEARCH.
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


def test_derotation_focal_matches_measured_camera_geometry():
    # F57 (20260730T003044Z-visual-course-74abd688): the de-rotation focal
    # equals the same camera's measured 1.6 norm/rad already in the file
    # (VERTICAL_PITCH_COMP_NORM_PER_RAD) — at 1.0 every predicted bearing
    # under-rotated by 37.5% (F52 frozen ex -0.156 vs true -0.48).
    assert ROTATION_COMP_FOCAL_NORM == pytest.approx(1.6)
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    hypothesis = controller.current
    start_x, start_y = hypothesis.x, hypothesis.y
    controller._predict(hypothesis, 0.033, (0.0, -0.20, 0.40))
    # drift_x = -yaw_rate * focal * dt; drift_y = pitch_rate * focal * dt.
    assert hypothesis.x - start_x == pytest.approx(-0.40 * 1.6 * 0.033)
    assert hypothesis.y - start_y == pytest.approx(-0.20 * 1.6 * 0.033)


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


def test_commit_fresh_steering_carries_the_near_plane_boost():
    boosted = _commit_controller()
    boosted.current.scale_axis.p = math.log(0.35)  # log_scale -1.05 >= -1.2
    out, _ = _drive_commit_window(boosted, 100.10)
    assert boosted.state is CleanCourseState.COMMIT
    # Trim frozen by the helper: the gains act on ex=0.10 directly —
    # 0.9*2.5*0.10 = 0.225, clamped to the 0.15 production yaw command cap.
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)
    plain = _commit_controller()  # scale_axis log ~-3.0: far-range gains
    out, _ = _drive_commit_window(plain, 100.10)
    assert plain.state is CleanCourseState.COMMIT
    assert out.yaw_rate_rad_s == pytest.approx(0.09, abs=1e-9)  # 0.9*0.10


def _commit_controller(now_s=100.10):
    """Gate-1 TRACK controller one sustain window short of COMMIT entry:
    near plane (outer log scale -0.50 >= -1.2), fresh uncensored
    measurements on both axes, and the 2026-07-30 entry budget satisfied —
    a usable inner aperture whose 60% margin admits error+blackout drift
    (0.10 + 0 <= 0.6*0.25 on x; 0.05 + 0 <= 0.6*0.25 on y).  The lateral
    trim integrator is frozen (gain 0) so steering-law assertions stay
    exact; the dedicated trim test exercises the integrator itself."""

    controller = _tracked_controller(
        _track("A", 0.0, 0.0), config=_config(ex_trim_gain=0.0)
    )
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
    # COMMIT steers while x is fresh (yaw 0.90 on ex=+0.10, trim frozen by
    # the helper).
    assert out.yaw_rate_rad_s == pytest.approx(0.09, abs=1e-9)


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
    # Gate 0 is untouched: the same geometry keeps its proved advance law.
    gate0 = _tracked_controller(_track("A", 0.10, 0.05, scale=0.50))
    gate0.current.outer_log_scale = -0.50
    gate0.current.last_x_measurement_s = 100.0 - 0.40
    out0 = _command(gate0, 100.10)
    assert out0.target_pitch_rad > SPAWN_PITCH - 0.15 + 1e-9


def test_early_arrest_brakes_before_censorship_without_self_blinding():
    # F75 on the F72/F74 approach geometry: closure energy is arrested in
    # the pre-censorship band [-1.2, -0.9) where the whole gate is still
    # visible, and the near-plane zone keeps vision custody.  F73b's hold
    # only engaged at -0.9 — by then the F74 approach (fh 2.5 -> 3.6 while
    # braking) could no longer be stopped, and its arrest suppression
    # pitched the gate out of view at the plane.
    # Phase A (F74 t~4-5 s state): outer log scale -1.00, span expanding
    # at 0.85/s, ey +0.31 (the relax WOULD level), budget false.  On
    # fb41bc2 the emitted pitch here is LEVEL (no hold until -0.9, relax
    # levels) — the drone carried ~3 m/s into the censorship zone.  The
    # widened hold must emit the TRUE brake this early.
    early = _commit_controller()
    current = early.current
    current.x_axis.p = -0.12
    current.y_axis.p = 0.31
    current.raw_x = -0.12
    current.raw_y = 0.31
    current.scale_axis.p = -1.00
    current.scale_axis.v = 0.85
    current.outer_log_scale = -1.00
    current.aperture_half_x = 0.20
    current.aperture_half_y = 0.20
    out, _ = _drive_commit_window(early, 100.10)
    assert early.state is CleanCourseState.TRACK  # COMMIT never arms
    assert early._pre_cross_brake_active
    assert not early._brake_vision_relax  # arrest outranks custody here
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH - 0.15, abs=1e-9)
    # Phase B (same leg at the -0.9 censorship onset, still hot): the
    # arrest no longer suppresses the F51/F65 relax — near-plane vision
    # custody outranks the brake, so the emitted pitch is LEVEL on the
    # fresh ey +0.31 instead of the self-blinding brake attitude F73b
    # emitted at the plane.
    plane = _commit_controller()
    current = plane.current
    current.x_axis.p = -0.12
    current.y_axis.p = 0.31
    current.raw_x = -0.12
    current.raw_y = 0.31
    current.scale_axis.p = -0.60
    current.scale_axis.v = 0.85
    current.outer_log_scale = -0.80
    current.aperture_half_x = 0.20
    current.aperture_half_y = 0.20
    out, _ = _drive_commit_window(plane, 100.10)
    assert plane.state is CleanCourseState.TRACK
    assert plane._brake_vision_relax  # custody preserved at the plane
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)


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


def test_course_closure_excess_brakes_collective_below_support():
    # F77 (20260730T080147Z-visual-course-87a36ce9): the F76 gate-1 leg
    # adopted the left successor cleanly and STILL blew through the
    # plane — fh 3-4 the whole leg, a +2.1 m balloon, and the gate
    # walked out the bottom-left corner at the plane.  Attitude-only
    # braking never decelerates (F74/F75/F76 all held fh 2.5-5 through
    # the -0.46 brake attitude), and the (gate-0-evidenced) brake
    # ceiling band floored the collective near support: nothing removed
    # the energy.  Closure excess above the governor target subtracts
    # collective BELOW support on gate-1+ legs; gate 0 keeps its proved
    # envelope (and the brake band) untouched.
    controller = _tracked_controller(_track("A", 0.0, 0.05, scale=0.20))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 2.0  # honest altitude (floor quiet)
    current = controller.current
    current.x_axis.p = 0.0
    current.raw_x = 0.0
    current.scale_axis.p = -1.6  # span ~0.20, outside the near-plane zone
    current.scale_axis.v = 0.58  # F76-measured hot closure (target 0.35)
    now = 100.10
    out = None
    for _ in range(15):  # converge the slews/governors
        now += 0.033
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.TRACK
    assert out.thrust < SPAWN_SUPPORT - 0.03  # real deceleration authority
    # Settled closure restores the normal support-level vertical law.
    current.scale_axis.v = 0.10
    for _ in range(15):
        now += 0.033
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert out.thrust > SPAWN_SUPPORT - 0.03
    # Gate 0 is untouched: the same hot closure keeps its envelope.
    gate0 = _tracked_controller(_track("A", 0.0, 0.05, scale=0.20))
    gate0.current.scale_axis.v = 0.58
    now0 = 100.10
    for _ in range(15):
        now0 += 0.033
        gate0.current.last_measurement_s = now0
        gate0.current.last_x_measurement_s = now0
        gate0.current.last_y_measurement_s = now0
        out0 = _command(gate0, now0, pitch=SPAWN_PITCH)
    assert out0.thrust > SPAWN_SUPPORT - 0.03


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


def test_commit_entry_requires_alignment_proximity_and_gate_one():
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
    # Gate 0 is deliberately excluded — its climb-bias path is working.
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
    # F56: while x is FRESH the commit steers with the TRACK P gains on
    # the derotated hypothesis (yaw 0.90, roll 0.50) — the entry offset
    # is finished, not frozen (F55 crossed beside the post).  2026-07-30:
    # the gains act on the trim-corrected error (trim frozen at 0 by the
    # helper, so ex=+0.10): yaw 0.09, roll 0.05.
    assert out.yaw_rate_rad_s == pytest.approx(0.09, abs=1e-9)
    assert out.target_roll_rad == pytest.approx(0.05, abs=1e-9)
    # F58: the real 0.15 rad advance drive, not the coast's 0.05 nudge.
    # F66: the F60 vertical-aim term is deleted — in commit the attitude is
    # the forward drive only; vertical translation is the servo's alone.
    assert out.target_pitch_rad == pytest.approx(
        SPAWN_PITCH + 0.15, abs=1e-9
    )
    # Bounded vertical servo on the compensated ey (0.05 -> -0.004).
    assert out.thrust == pytest.approx(SPAWN_SUPPORT - 0.004, abs=1e-9)
    # The servo tracks inside the band (0.50 -> -0.04)...
    controller.current.y_axis.p = 0.50
    out = _command(controller, now + 0.033, pitch=SPAWN_PITCH)
    assert out.thrust == pytest.approx(SPAWN_SUPPORT - 0.04, abs=1e-9)
    # ...and is BOUNDED to the +/-0.05 band around support: slammed
    # bottom-bar dive evidence (-0.08 raw) cannot descend harder than the
    # band (the 0.21 envelope clamp sits just above the band bottom here).
    controller.current.y_axis.p = 1.0
    out = _command(controller, now + 0.066, pitch=SPAWN_PITCH)
    assert out.thrust >= SPAWN_SUPPORT - 0.05
    assert out.thrust < SPAWN_SUPPORT - 0.03
    # Slammed climb evidence is capped at the band top exactly.
    controller.current.y_axis.p = -1.0
    out = _command(controller, now + 0.099, pitch=SPAWN_PITCH)
    assert out.thrust == pytest.approx(SPAWN_SUPPORT + 0.05, abs=1e-9)
    # The progress-removers stay bypassed: a bearing that would fully
    # engage the misalignment brake in TRACK gets no brake pitch — the
    # advance continues; steering follows the fresh bearing (yaw gain
    # 0.9*0.8 clamped to the 0.15 production yaw command cap).
    controller.current.y_axis.p = 0.05
    controller.current.x_axis.p = 0.80
    now += 0.132
    controller.current.last_measurement_s = now
    controller.current.last_x_measurement_s = now
    controller.current.last_y_measurement_s = now
    out = _command(controller, now, pitch=SPAWN_PITCH)
    assert out.yaw_rate_rad_s == pytest.approx(0.15, abs=1e-9)
    assert out.target_pitch_rad == pytest.approx(
        SPAWN_PITCH + 0.15, abs=1e-9
    )
    # F62/F63: once x goes STALE/censored the commit steers the PREDICTED
    # hypothesis at FULL gain — heading-hold committed the residual drift
    # (F61 clipped the left post) and F62's half-gain derate
    # under-corrected (crossed -0.22 left); the prediction tracked the
    # real bearing through the blackout.  ex=+0.10 (trim frozen at 0) ->
    # yaw 0.90*0.10 = 0.09, roll 0.50*0.10 = 0.05 (boost off at this
    # log scale).
    controller.current.x_axis.p = 0.10
    for _ in range(18):
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.COMMIT
    assert out.yaw_rate_rad_s == pytest.approx(0.09, abs=1e-9)
    assert out.target_roll_rad == pytest.approx(0.05, abs=1e-9)


def test_commit_stale_y_never_climbs_on_a_frozen_bearing():
    # F58 (20260730T004618Z-visual-course-cae7b894): the band servo's y
    # input froze at censorship and output the CLIMB side (thrust pinned
    # support+0.05) for the whole commit — the drone drifted up over the
    # opening.  Stale/censored y clamps the servo correction to <= 0:
    # bare support instead of a frozen climb; the descend side stays live
    # (and the vz governor bounds the sink).  Fresh y keeps the
    # bidirectional band (covered by the law test above).
    controller = _commit_controller()
    out, now = _drive_commit_window(controller, 100.10)
    assert controller.state is CleanCourseState.COMMIT
    controller.current.y_axis.p = -0.50  # frozen climb-side servo (+0.04)
    for _ in range(15):  # the y stamp ages past the 0.30 s window
        now += 0.033
        controller.current.last_measurement_s = now
        controller.current.last_x_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller.state is CleanCourseState.COMMIT
    # Correction clamped to 0 — bare support, never support + band.
    assert out.thrust == pytest.approx(SPAWN_SUPPORT, abs=1e-9)
    # The descend side survives the clamp (0.50 -> -0.04).
    controller.current.y_axis.p = 0.50
    out = _command(controller, now + 0.033, pitch=SPAWN_PITCH)
    assert out.thrust == pytest.approx(SPAWN_SUPPORT - 0.04, abs=1e-9)


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


def test_promotion_requires_persistent_successor():
    # F42: an otherwise credible but NEWBORN successor must not be adopted
    # (debris is newborn every frame); the same successor is adopted once
    # it has been associated past successor_min_age_s.
    young = _tracked_controller(_track("A", 0.0, 0.0))
    young.observe(
        _update(
            [_track("A", 0.0, 0.0), _track("B", 0.30, 0.05, scale=0.05)],
            frame_id=3,
        ),
        now_s=100.08,
    )
    promoted = young.note_race(gate_index=1, race_boot_ms=2500, now_s=100.10)
    assert promoted
    assert young.state is CleanCourseState.SEARCH
    assert young.current is None

    aged = _tracked_controller(_track("A", 0.0, 0.0))
    now = 100.10
    frame = 10
    for _ in range(20):  # ~0.66 s of association before the increment
        now += 0.033
        aged.observe(
            _update(
                [_track("A", 0.0, 0.0), _track("B", 0.30, 0.05, scale=0.05)],
                frame_id=frame,
            ),
            now_s=now,
        )
        frame += 1
    promoted = aged.note_race(gate_index=1, race_boot_ms=2500, now_s=now)
    assert promoted
    assert aged.state is CleanCourseState.TRACK
    assert aged.current.track_id == "B"


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


def test_loop_coast_holds_exact_zero_then_accepts_credit():
    host = _Host(_update([_track("A", 0.0, 0.0, scale=0.50)]))

    def script(host):
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
        run_clean_course_stage(host, context, runtime=_test_runtime())
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
    # estimate cannot leak leveling rates or thrust onto the wire.
    host = _Host(_update([_track("A", 0.0, 0.0, scale=0.50)]))
    host.estimate = SimpleNamespace(
        orientation=SimpleNamespace(to_euler=lambda: (0.10, -0.08, 0.0)),
        body_rates=(0.0, 0.0, 0.0),
    )

    def script(host):
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
        run_clean_course_stage(host, context, runtime=_test_runtime())
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
    # exits (it is accepted in EVERY state).
    host = _Host(_update([_track("A", 0.0, 0.0, scale=0.50)]))

    def script(host):
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
        run_clean_course_stage(host, context, runtime=_test_runtime())
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


def test_commit_entry_refuses_excessive_closure():
    # Arrival energy is the uncontrolled budget: an approach still closing
    # faster than the governor target (expansion rate 0.50/s > 0.35) is
    # refused even with a usable aperture and a centered, settled bearing —
    # TRACK keeps controlling energy outside the censorship blackout
    # instead of committing a hot crossing.
    closing = _commit_controller()
    closing.current.scale_axis.v = 0.50
    _drive_commit_window(closing, 100.10)
    assert closing.state is CleanCourseState.TRACK


def test_descent_floor_is_continuous_across_its_threshold():
    # F65/F66b: the on/off descent-floor step (+0.04 feedforward) bang-banged
    # thrust 0.21<->0.31 at the gate plane as vz_est straddled the floor.
    # The floor is now one proportional law: sweeping vz across the -0.35
    # threshold moves collective smoothly — monotone, no step.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    helper = controller._governed_collective
    previous = None
    steps = []
    for k in range(21):  # vz -0.30 .. -0.50 in 0.01 m/s steps
        controller._vz_est_m_s = -0.30 - 0.01 * k
        value = helper(SUPPORT, SUPPORT)
        if previous is not None:
            assert value >= previous - 1e-12  # monotone nondecreasing
            steps.append(value - previous)
        previous = value
    assert steps
    assert max(steps) < 0.005  # ~0.21 * 0.01 = 0.0021 per step, no jump


def test_track_lateral_trim_integrates_bounded_and_nulls_the_orbit():
    # The off-axis pursuit is a P-loop that equilibrates at ex ~-0.08 every
    # flight (yaw holds the bearing against translation parallax).  The
    # TRACK-phase trim integrates the sustained error on gate-1+ legs and
    # nulls the orbit BEFORE entry, bounded at +/-0.15 — replacing the
    # F63 COMMIT-only fixed bias.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 2.0
    current = controller.current  # B: far scale, no commit arming, no boost
    current.x_axis.p = 0.10
    now = 100.10
    first_yaw = None
    out = None
    for _ in range(60):  # ~2 s of sustained ex=+0.10
        now += 0.033
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
        if first_yaw is None:
            first_yaw = out.yaw_rate_rad_s
    assert first_yaw == pytest.approx(0.09, abs=2e-3)  # 0.9 * 0.10 pre-trim
    assert 0.04 < controller._ex_trim <= 0.15
    assert out.yaw_rate_rad_s < first_yaw  # the trim eats the orbit error
    # Anti-windup: an arbitrarily long sustained error cannot wind the trim
    # past its bound.
    for _ in range(400):
        now += 0.033
        current.last_measurement_s = now
        current.last_x_measurement_s = now
        current.last_y_measurement_s = now
        out = _command(controller, now, pitch=SPAWN_PITCH)
    assert controller._ex_trim <= 0.15 + 1e-12
    # With the trim converged onto the orbit error, the steering law sees a
    # nulled error and commands zero yaw.
    frozen = _tracked_controller(
        _track("A", 0.0, 0.0), config=_config(ex_trim_gain=0.0)
    )
    _promote_to_gate_one(frozen)
    frozen._alt_est_m = 2.0
    frozen.current.x_axis.p = 0.10
    frozen._ex_trim = 0.10
    now = 100.133
    frozen.current.last_measurement_s = now
    frozen.current.last_x_measurement_s = now
    frozen.current.last_y_measurement_s = now
    out = _command(frozen, now, pitch=SPAWN_PITCH)
    assert out.yaw_rate_rad_s == pytest.approx(0.0, abs=1e-9)


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
