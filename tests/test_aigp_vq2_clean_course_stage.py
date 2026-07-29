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
overriding everything but the coast support-hold latch with a level
attitude, zero yaw, and a governed climb collective, bounded to a 2.5 s
latch with a 1.0 s above-release re-arm (F13), an fh inflow-regime gate
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
    # graze; the climb-only governor did nothing.  The symmetric floor adds
    # K_VZ_DESCENT per m/s below the -0.5 m/s sink bound, plus the fixed
    # +0.025 descent-regime hover feedforward (flight d5e89c2b) whenever the
    # estimate is below the floor.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    helper = controller._governed_collective
    max_thrust = controller.config.max_thrust
    controller._vz_est_m_s = -0.5  # at the floor boundary: no effect
    assert helper(SUPPORT, SUPPORT) == pytest.approx(SUPPORT, abs=1e-9)
    controller._vz_est_m_s = -1.0  # 0.5 m/s below -> +0.03 + 0.025 feedforward
    # support + 0.055: inside the raised 0.34 envelope (F9/F10 headroom
    # restoration; the old 0.32 clamp clipped exactly this case).
    assert helper(SUPPORT, SUPPORT) == pytest.approx(SUPPORT + 0.055, abs=1e-9)
    # Deep sinks saturate at max_thrust (flight 039186c8: the unclamped
    # floor boost exceeded the runner's 0.35 envelope abort in SEARCH).
    controller._vz_est_m_s = -1.7  # 1.2 m/s below -> +0.072 + 0.025 raw
    assert helper(SUPPORT, SUPPORT) == pytest.approx(max_thrust, abs=1e-9)
    # The floor only raises collective, but the governed output is still
    # clamped: a higher command saturates at max_thrust as well.
    controller._vz_est_m_s = -1.0
    assert helper(0.35, SUPPORT) == pytest.approx(max_thrust, abs=1e-9)


def test_vz_descent_floor_raises_command_thrust():
    # Command level: vz = -0.7 m/s lifts the emitted collective to
    # support + 0.012 (proportional) + 0.025 (feedforward) = 0.312, below
    # the raised 0.34 clamp so it stays observable; a deeper sink (vz -1.0,
    # raw 0.33) now fits inside the envelope instead of clipping at 0.32.
    controller = _tracked_controller(_track("A", 0.0, 0.0))  # e = 0, vy ~ 0
    controller._vz_est_m_s = -0.7
    assert _command(controller, 100.10).thrust == pytest.approx(
        SUPPORT + 0.037, abs=1e-9
    )
    controller._vz_est_m_s = -1.0
    assert _command(controller, 100.14).thrust == pytest.approx(
        SUPPORT + 0.055, abs=1e-9
    )


def test_vz_descent_floor_applies_in_predict_and_search():
    # The floor is IMU-based for the same reason as the climb cap: vision
    # loss (the d52adcd4 stall case) must not disable it.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.observe(_update([], frame_id=2), now_s=100.12)  # superseded
    assert controller.state is CleanCourseState.PREDICT
    controller._vz_est_m_s = -0.7
    assert _command(controller, 100.16).thrust == pytest.approx(
        SUPPORT + 0.037, abs=1e-9
    )
    controller._enter_search(100.20)
    assert _command(controller, 100.22).thrust == pytest.approx(
        SUPPORT + 0.037, abs=1e-9
    )


def test_vz_phantom_sink_cannot_move_coast_support_hold():
    # The coast hold emits the tilt-compensated support collective,
    # UNGOVERNED: a phantom sink must neither zero it (the old ballistic
    # latch, flight 22ceaa6f) nor boost a climb into the top bar.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller.observe(_update([], frame_id=3), now_s=100.12)  # fresh close loss
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    controller._vz_est_m_s = -1.0
    out = _command(controller, 100.14)
    assert out.thrust == pytest.approx(SUPPORT, abs=1e-9)
    assert (out.target_roll_rad, out.target_pitch_rad, out.yaw_rate_rad_s) == (
        0.0,
        SPAWN_PITCH + 0.05,  # F38/F49 coast advance nudge (spawn-relative):
        # carry through the engulfed plane
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
    # from the governor's vz_est, seeded 0 at course start) overrides
    # everything but the coast latch: level (spawn) attitude and a governed
    # climb collective until the release hysteresis clears.  F51: it arms
    # only while blind (driven here with aged measurements) and keeps
    # x-qualified lateral authority while active.
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
    # Sinking at -1.0 m/s: support + 0.06 * 0.5 + 0.025 feedforward.
    assert out.thrust == pytest.approx(SUPPORT + 0.055, abs=1e-9)
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
    # the governor's internal clamp keeps the override inside the course
    # envelope (0.34, below the runner's 0.35 hard abort).
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 0.5
    controller._vz_est_m_s = -4.3
    _blind_current(controller, 100.14)  # F51: the floor arms only blind
    out = _command(controller, 100.14)
    assert controller._alt_floor_active
    assert out.thrust == pytest.approx(controller.config.max_thrust, abs=1e-9)


def test_altitude_floor_never_overrides_coast_support_hold():
    # The coast support-hold latch still wins over the floor: a fresh close
    # loss inside the low-altitude window keeps the coast's own collective.
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
    controller.observe(_update([], frame_id=4), now_s=100.12)  # fresh close loss
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    controller._alt_est_m = 0.3
    controller._vz_est_m_s = -1.0
    assert _command(controller, 100.14).thrust == pytest.approx(SUPPORT, abs=1e-9)


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


def test_alt_floor_override_keeps_lateral_authority():
    # F51: the floor owns ONLY the collective and the level pitch — while
    # blind-armed, a fresh x measurement still steers yaw/roll toward the
    # gate (F50 parked lateral authority for the whole 2.5 s latch and the
    # gate-1 track walked off the frame).
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)  # current = B at x=+0.30
    controller._alt_est_m = 0.5
    _blind_current(controller, 100.10)  # F51: the floor arms only blind
    out = _command(controller, 100.10)
    assert controller._alt_floor_active
    # x aged with the rest: stale x -> heading hold, wings level.
    assert out.yaw_rate_rad_s == 0.0
    assert out.target_roll_rad == 0.0
    # A fresh x measurement under the active floor restores the standard
    # x-qualified pursuit gains (yaw 0.90, roll 0.50 on ex=+0.30).
    now = 100.10
    for _ in range(20):  # slew the roll target out to its 0.15 command
        now += 0.033
        controller.current.last_x_measurement_s = now
        out = _command(controller, now)
    assert controller._alt_floor_active
    assert out.yaw_rate_rad_s == pytest.approx(0.27, abs=1e-9)
    assert out.target_roll_rad == pytest.approx(0.15, abs=1e-9)
    # The collective stays the governed recovery climb, above bare support.
    assert out.thrust > SUPPORT


def test_alt_floor_override_pitches_to_spawn_not_absolute_zero():
    # F51: "level" under the F49 spawn-relative convention is SPAWN_PITCH;
    # an absolute 0.0 target is +0.31 rad physical nose-down.  The floor
    # pitch target is the spawn attitude, slewed in from any prior target.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 0.5
    _blind_current(controller, 100.10)  # F51: the floor arms only blind
    controller._prev_target_pitch = SPAWN_PITCH - 0.15  # braking attitude
    now = 100.10
    out = None
    for _ in range(20):
        now += 0.033
        out = _command(controller, now)
    assert controller._alt_floor_active
    assert out.target_pitch_rad == pytest.approx(SPAWN_PITCH, abs=1e-9)
    assert out.target_pitch_rad != 0.0


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
    # Raised yaw gain/cap: 0.9 * 0.50 pursuit, inside the 0.50 cap.
    assert out.yaw_rate_rad_s == pytest.approx(0.45, abs=1e-9)


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


def test_search_descends_toward_remembered_low_bearing():
    # F50 (flight 20260729T222920Z-visual-course-3a8ed087): F49's SEARCH
    # held the support floor at ceiling height for 8 s while the gate sat
    # ~1.5 m below — a search that cannot descend never re-acquires a gate
    # that sank below the FOV.  With a reliable bearing memory, SEARCH
    # servos the collective on the remembered attitude-compensated ey,
    # bounded to a small band around support.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller._enter_search(100.10)
    controller._set_reliable_bearing(0.0, 0.50)  # gate below center
    out = _command(controller, 100.143, pitch=SPAWN_PITCH)
    # Proportional descent: -0.080 * 0.50 = -0.04 around the level base.
    assert out.thrust == pytest.approx(SPAWN_SUPPORT - 0.080 * 0.50, abs=1e-9)
    assert out.thrust < SPAWN_SUPPORT
    # A remembered HIGH bearing climbs by the same term.
    controller._set_reliable_bearing(0.0, -0.50)
    out = _command(controller, 100.20, pitch=SPAWN_PITCH)
    assert out.thrust == pytest.approx(SPAWN_SUPPORT + 0.080 * 0.50, abs=1e-9)
    # The correction is bounded to the band, and the min-thrust envelope
    # still floors the result.
    controller._set_reliable_bearing(0.0, 0.95)
    out = _command(controller, 100.26, pitch=SPAWN_PITCH)
    assert out.thrust == pytest.approx(0.21, abs=1e-9)


def test_search_memory_descent_yields_to_fh_untrusted_floor():
    # The memory servo still passes through _governed_collective: while
    # vz/alt are known lies the support + margin floor overrides the
    # descent correction (the F21 blind-sink protections win).
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller._enter_search(100.10)
    controller._set_reliable_bearing(0.0, 0.50)
    controller._fh_untrusted = True
    out = _command(controller, 100.143, pitch=SPAWN_PITCH, fh=6.0)
    assert out.thrust == pytest.approx(SPAWN_SUPPORT + 0.05, abs=1e-9)


def test_search_without_bearing_memory_holds_support():
    # No bearing evidence yet: the vertical memory servo stays out and the
    # search holds the plain support collective (pre-F50 behavior).
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
    # flew blind into the frame in PREDICT.  The coast holds the SUPPORT
    # collective at level attitude (F25), so there is no zero-thrust drop
    # to guard against: a credible close crossing loss coasts regardless.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller._fh_untrusted = True
    now = 100.10
    now += 0.033
    controller.observe(_update([], frame_id=20), now_s=now)  # fresh close loss
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    output = _command(controller, now + 0.02)
    assert output.thrust == pytest.approx(SUPPORT, abs=1e-9)


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
    ) == (0.0, SPAWN_PITCH + 0.05, 0.0)
    assert output.thrust == pytest.approx(SUPPORT, abs=1e-9)


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


def test_fresh_close_loss_still_coasts_and_holds_support():
    # The July-18 bounded credible-crossing wait is preserved: a genuine
    # close-range loss on a FRESH frame (new frame id) still arms the coast
    # latch — now a level-attitude support hold, not a ballistic zero.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller.observe(_update([], frame_id=3), now_s=100.12)  # fresh id
    assert controller.state is CleanCourseState.COAST_FOR_CREDIT
    output = _command(controller, 100.14)
    assert (
        output.target_roll_rad,
        output.target_pitch_rad,
        output.yaw_rate_rad_s,
    ) == (0.0, SPAWN_PITCH + 0.05, 0.0)
    assert output.thrust == pytest.approx(SUPPORT, abs=1e-9)


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
    assert all(abs(value) == pytest.approx(0.20) for value in yaws)
    assert all(abs(value) <= 0.25 for value in yaws)
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
    ) == (0.0, SPAWN_PITCH + 0.05, 0.0)
    assert output.thrust == pytest.approx(SUPPORT, abs=1e-9)
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

    # The wait is bounded at 0.40 s even with no newer race packet.
    controller2 = _tracked_controller(_track("A", 0.0, 0.0, scale=0.50))
    controller2.note_race(gate_index=0, race_boot_ms=2000, now_s=100.10)
    controller2.observe(_update([], frame_id=5), now_s=100.12)
    output = _command(controller2, 100.12 + 0.41)
    assert controller2.state is CleanCourseState.SEARCH
    assert output.thrust > 0.0


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
    # yaw gain 0.90 on the held ex=-0.17 bearing keeps centering.
    assert out.yaw_rate_rad_s == pytest.approx(-0.153, abs=1e-9)
    assert out.target_roll_rad < 0.0  # slewing toward the 0.50*ex bank
    # FAR target, identical staleness: the F40 zeroing still applies.
    far = _tracked_controller(_track("A", -0.17, 0.0, scale=0.10))
    far.current.last_x_measurement_s = 100.10 - 1.0
    out = _command(far, 100.10)
    assert out.yaw_rate_rad_s == 0.0
    assert out.target_roll_rad == 0.0


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
        assert abs(output.yaw_rate_rad_s) <= 0.50 + 1e-9
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
        assert abs(command.yaw_rate) <= 0.25 + 1e-9


def test_loop_coast_holds_support_then_accepts_credit():
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
    # F26: the coast wait holds level attitude at the support collective —
    # no ballistic zero, and the PD wire sends carry support thrust.
    support_sends = [
        command
        for command, _index in host.sent
        if command.thrust == pytest.approx(SUPPORT, abs=1e-6)
        and command.yaw_rate == 0.0
    ]
    assert support_sends  # the support-hold coast happened
    assert summary["final_gate_index"] == 1


def test_loop_coast_levels_attitude_through_pd_at_support_thrust():
    # F11 historically required exact-zero coast sends because the attitude
    # PD leaked nonzero rates at zero thrust.  F26 retired the ballistic
    # coast: the wait now goes through the PD with level targets at support
    # thrust, so a nonzero attitude produces genuine LEVELING rates (that is
    # the point — active attitude hold instead of a falling latch).
    host = _Host(_update([_track("A", 0.0, 0.0, scale=0.50)]))
    host.estimate = SimpleNamespace(
        orientation=SimpleNamespace(to_euler=lambda: (0.10, -0.08, 0.0)),
        body_rates=(0.0, 0.0, 0.0),
    )
    probe = _fake_pd(
        host.estimate, target_roll_rad=0.0, target_pitch_rad=0.0, thrust=SUPPORT
    )
    assert (probe.roll_rate, probe.pitch_rate) != (0.0, 0.0)  # PD levels it

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
    support_sends = [
        command
        for command, _index in host.sent
        if command.thrust == pytest.approx(SUPPORT, abs=5e-3)
    ]
    assert support_sends  # the bounded coast wait emitted support sends
    for command in support_sends:
        assert math.isfinite(command.roll_rate)
        assert math.isfinite(command.pitch_rate)
        assert command.yaw_rate == 0.0
    assert summary["final_gate_index"] == 1
    assert summary["success"] is True
