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
    LAUNCH_BOOST_DURATION_S,
    LAUNCH_BOOST_THRUST,
    CleanCourseConfig,
    CleanCourseController,
    CleanCourseRuntime,
    CleanCourseState,
    NavigationOutput,
    clamp_final_command,
    run_clean_course_stage,
)

SUPPORT = 0.275


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
    controller = _tracked_controller(_track("A", 0.0, 0.20))
    output = _command(controller, 100.10)
    assert output.thrust == pytest.approx(
        SUPPORT - 0.080 * 0.20, abs=1e-9
    )
    controller = _tracked_controller(_track("A", 0.0, -0.20))
    output = _command(controller, 100.10)
    assert output.thrust == pytest.approx(
        SUPPORT + 0.080 * 0.20, abs=1e-9
    )


def test_gate0_takeoff_boost_is_feedforward_only():
    # Boost-window behavior isolated from the gate-0 climb offset.
    config = CleanCourseConfig(gate0_climb_vertical_offset_norm=0.0)
    controller = _tracked_controller(_track("A", 0.0, 0.20), config=config)
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
    output = _command(controller, 100.10)
    # e = -0.10 - 0.25 = -0.35 -> full climb correction above support.
    assert output.thrust == pytest.approx(SUPPORT + 0.080 * 0.35, abs=1e-9)
    assert 0.21 <= output.thrust <= 0.34

    high = _tracked_controller(_track("A", 0.0, -0.60), config=config)
    assert _command(high, 100.10).thrust <= 0.34

    controller.observe(
        _update(
            [_track("A", 0.0, 0.0), _track("B", 0.30, 0.0, scale=0.05)],
            frame_id=4,
        ),
        now_s=100.12,
    )
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=1250, now_s=100.14
    )
    assert promoted
    assert controller.state is CleanCourseState.TRACK
    controller._alt_est_m = 2.0  # honest altitude (floor quiet)
    output = _command(controller, 100.16)
    assert output.gate_index == 1
    # Same centered target after promotion: the unbiased law holds support.
    assert output.thrust == pytest.approx(SUPPORT, abs=1e-9)


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
    assert _command(far, 100.10).thrust == pytest.approx(
        SUPPORT + 0.080 * (0.10 + 0.25), abs=1e-9
    )

    mid = _tracked_controller(
        _track("A", 0.0, -0.10, scale=math.exp(-1.295)), config=config
    )
    mid_offset = 0.25 * (-1.295 - (-0.80)) / (-1.79 - (-0.80))
    assert _command(mid, 100.10).thrust == pytest.approx(
        SUPPORT + 0.080 * (0.10 + mid_offset), abs=1e-9
    )

    crossing = _tracked_controller(
        _track("A", 0.0, 0.0, scale=math.exp(-0.80)), config=config
    )
    assert _command(crossing, 100.10).thrust == pytest.approx(SUPPORT, abs=1e-9)


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
    assert _command(centered, 100.10).thrust == pytest.approx(SUPPORT, abs=1e-9)

    below = _tracked_controller(
        _track("A", 0.0, 0.10, scale=0.1667), config=config
    )
    output = _command(below, 100.10)
    # Offset contributes nothing: only the plain e = +0.10 descent feedback.
    assert output.thrust == pytest.approx(SUPPORT - 0.080 * 0.10, abs=1e-9)
    assert output.thrust < SUPPORT


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
    assert output.thrust < SUPPORT - 0.05


def test_vz_governor_caps_collective_above_climb_cap():
    # Four gate-0 top-bar flights: bearing pursuit built unbounded vz
    # (2.8-3.35 m/s peaks vs a ~0.9 m/s requirement).  The IMU governor
    # removes K_VZ per m/s over the 1.0 m/s cap from the collective.
    controller = _tracked_controller(_track("A", 0.0, 0.0))  # e = 0, vy ~ 0
    controller._vz_est_m_s = 0.5  # below the cap: no effect
    assert _command(controller, 100.10).thrust == pytest.approx(SUPPORT, abs=1e-9)
    controller._vz_est_m_s = 2.0  # 1.0 m/s over cap -> -0.03 collective
    assert _command(controller, 100.14).thrust == pytest.approx(
        SUPPORT - 0.03, abs=1e-9
    )


def test_vz_governor_applies_in_predict_and_search():
    # The governor is IMU-based precisely so vision loss cannot disable it.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.observe(_update([], frame_id=2), now_s=100.12)  # superseded
    assert controller.state is CleanCourseState.PREDICT
    controller._vz_est_m_s = 2.0
    assert _command(controller, 100.16).thrust == pytest.approx(
        SUPPORT - 0.03, abs=1e-9
    )
    controller._enter_search(100.20)
    assert _command(controller, 100.22).thrust == pytest.approx(
        SUPPORT - 0.03, abs=1e-9
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
    # 0.275 + 0.055 = 0.33 raw: inside the raised 0.34 envelope (F9/F10
    # headroom restoration; the old 0.32 clamp clipped exactly this case).
    assert helper(SUPPORT, SUPPORT) == pytest.approx(SUPPORT + 0.055, abs=1e-9)
    # Deep sinks saturate at max_thrust (flight 039186c8: the unclamped
    # floor boost exceeded the runner's 0.35 envelope abort in SEARCH).
    controller._vz_est_m_s = -1.5  # 1.0 m/s below -> +0.06 + 0.025 = 0.36 raw
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
        output = _command(controller, now + 0.005)
        assert controller.state is not CleanCourseState.SEARCH
        assert output.thrust == pytest.approx(SUPPORT, abs=1e-9)
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
    promoted = controller.note_race(
        gate_index=1, race_boot_ms=2500, now_s=now_s
    )
    assert promoted
    assert controller.current.track_id == "B"


def test_altitude_floor_triggers_and_releases_with_hysteresis():
    # F10/F11/F12: the final 6-10 s before gate 1 ran below 0.7 m with
    # thrust pinned into terrain.  The pre-gate-1 floor (alt_est integrated
    # from the governor's vz_est, seeded 0 at course start) overrides
    # everything but the coast latch: level attitude, zero yaw, and a
    # governed climb collective until the release hysteresis clears.
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
        out = _command(controller, now)
        frame += 1
    assert controller._alt_floor_active
    assert out.yaw_rate_rad_s == 0.0
    # Sinking at -1.0 m/s: support + 0.06 * 0.5 + 0.025 feedforward = 0.33.
    assert out.thrust == pytest.approx(0.33, abs=1e-9)
    # Hysteresis: climbing between 0.7 and 1.2 m keeps the floor active.
    controller._vz_est_m_s = 1.0
    while controller._alt_est_m < 1.0:
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
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
    out = _command(controller, 100.14)
    assert controller._alt_floor_active  # the gate-1 window is live at 0.5 m
    assert out.yaw_rate_rad_s == 0.0
    assert controller.note_race(gate_index=2, race_boot_ms=3000, now_s=100.16)
    _command(controller, 100.18)
    assert not controller._alt_floor_active


def test_altitude_floor_respects_max_thrust():
    # F12's -4.3 m/s sink would demand support + 0.06*3.8 + 0.025 ~= 0.53;
    # the governor's internal clamp keeps the override inside the course
    # envelope (0.34, below the runner's 0.35 hard abort).
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    _promote_to_gate_one(controller)
    controller._alt_est_m = 0.5
    controller._vz_est_m_s = -4.3
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
                _track("B", 0.30, 0.05, scale=0.50),
            ],
            frame_id=3,
        ),
        now_s=100.08,
    )
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
    # Sink back below the trigger: the floor re-arms normally.
    controller._vz_est_m_s = -1.0
    out = None
    while controller._alt_est_m >= 0.7:
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
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
    for _ in range(10):  # unclamped integration would reach -3.3 m
        now += 0.033
        _command(controller, now)
    assert controller._alt_est_m == -2.0
    assert controller._alt_floor_active  # still guards inside the band


def test_fh_gate_engages_only_after_sustained_excess_not_a_transient():
    # F14: the regime gate latches only on fh > 3.0 SUSTAINED for 0.3 s; a
    # transient excursion must never freeze the vertical estimate.
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    now = 100.10
    for _ in range(7):  # 0.23 s above the trigger: transient, no latch
        now += 0.033
        _command(controller, now, fh=4.0)
    assert not controller._fh_untrusted
    now += 0.033
    _command(controller, now, fh=1.0)  # below the trigger: timer resets
    for _ in range(7):  # another 0.23 s: still only a transient
        now += 0.033
        _command(controller, now, fh=4.0)
    assert not controller._fh_untrusted
    for _ in range(4):  # 0.36 s continuous: the gate latches
        now += 0.033
        _command(controller, now, fh=4.0)
    assert controller._fh_untrusted


def test_fh_gate_releases_only_below_hysteresis_band():
    controller = _tracked_controller(_track("A", 0.0, 0.0))
    now = 100.10
    for _ in range(11):  # sustained: latch untrusted
        now += 0.033
        _command(controller, now, fh=4.0)
    assert controller._fh_untrusted
    # Inside the 2.0-3.0 hysteresis band the latch holds.
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
        _command(controller, now, fh=4.0)
    assert controller._fh_untrusted
    controller._alt_est_m = -1.0
    for _ in range(10):
        now += 0.033
        _command(controller, now, fh=4.0, a_up=-2.5)  # biased regime sample
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
    _command(controller, now)  # trusted: the floor arms normally
    assert controller._alt_floor_active
    # Going fh-untrusted must not clear the active latch...
    for _ in range(11):
        now += 0.033
        controller.observe(
            _update([_track("B", 0.30, 0.05, scale=0.05)], frame_id=frame),
            now_s=now,
        )
        _command(controller, now, fh=4.0)
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
        _command(controller, now, fh=4.0)
        frame += 1
    assert not controller._alt_floor_active
    # With the regime still untrusted the floor cannot re-arm even though
    # alt_est is below the trigger.
    now += 0.033
    _command(controller, now, fh=4.0)
    assert not controller._alt_floor_active


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
        _command(controller, now, fh=4.0)
    assert controller._fh_untrusted
    out = None
    for _ in range(30):  # the decay converges onto the hold
        now += 0.033
        out = _command(controller, now, fh=4.0)
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
        _command(controller, now, fh=4.0)
    assert controller._fh_untrusted
    now += 0.033
    controller.observe(
        _update([_track("A", 0.0, 0.0)], frame_id=21), now_s=now
    )
    out = _command(controller, now, fh=4.0)
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
        out = _command(controller, now)
    assert controller._pre_cross_brake_active
    assert out.state is CleanCourseState.TRACK
    assert out.target_pitch_rad == pytest.approx(0.15, abs=1e-9)
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
    assert out.target_pitch_rad < 0.0  # advance law still closes


def test_closure_governor_is_a_continuous_blend():
    # Mid-band rate (0.475/s -> closure 0.5): the pitch target blends
    # halfway from the advance law (<= -0.02) toward the +0.15 brake
    # attitude, without latching the fast-slew brake flag.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller.current.scale_axis.v = 0.475
    now = 100.10
    out = None
    for _ in range(25):  # generic slew converges to the blended target
        now += 0.033
        out = _command(controller, now)
    assert not controller._pre_cross_brake_active
    assert -0.02 + 1e-9 < out.target_pitch_rad < 0.15 - 1e-9


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
    for frame in range(3):  # ~0.1 s without a measurement -> PREDICT
        now += 0.033
        controller.observe(_update([], frame_id=40 + frame), now_s=now)
        out = _command(controller, now)
    assert controller.state is CleanCourseState.PREDICT
    for _ in range(10):  # inside the 0.5 s PREDICT bound
        now += 0.033
        out = _command(controller, now)
    assert controller.state is CleanCourseState.PREDICT
    assert controller._pre_cross_brake_active
    assert out.target_pitch_rad == pytest.approx(0.15, abs=1e-9)


def test_heading_anchor_clamps_outward_yaw_only():
    # F31: post-loss search/edge-chase wound the heading +2.63 rad off the
    # course bearing, then the drone flew sideways into structure it never
    # saw.  The anchor is captured lazily on the first yaw tick; outward
    # steering past the 0.9 rad cap is blocked, return steering is free.
    controller = _tracked_controller(_track("A", 0.30, 0.0, scale=0.10))
    out = _command(controller, 100.10, yaw=0.0)  # lazy anchor = 0.0
    assert controller._course_anchor_yaw_rad == pytest.approx(0.0)
    assert out.yaw_rate_rad_s > 0.0  # x=+0.30 pursuit steers freely
    # Heading wound past the cap: the same positive pursuit is blocked.
    out = _command(controller, 100.143, yaw=1.0)
    assert out.yaw_rate_rad_s <= 0.0
    # Return steering (target left of center) is always free at the cap.
    controller.current.x_axis.p = -0.30
    out = _command(controller, 100.176, yaw=1.0)
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


def test_search_commands_brake_attitude_when_blind_and_fast():
    # F31: blind legs were where speed ran away (the closure governor has
    # no expansion signal in SEARCH).  Above fh 1.5 SEARCH commands the
    # brake attitude; slow (takeoff) SEARCH stays near level.
    controller = _tracked_controller(_track("A", 0.0, 0.0, scale=0.10))
    controller._enter_search(100.10)
    controller._fh_mps2 = 0.5  # slow: near-level target
    out = _command(controller, 100.143)
    assert out.target_pitch_rad <= 0.0
    controller._fh_mps2 = 3.0  # blind and fast: brake
    now = 100.143
    for _ in range(20):  # generic slew converges to the brake attitude
        now += 0.033
        out = _command(controller, now)
    assert out.state is CleanCourseState.SEARCH
    assert out.target_pitch_rad == pytest.approx(0.15, abs=1e-9)
    assert abs(out.yaw_rate_rad_s) > 0.0  # the sweep stays alive


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
    assert output.thrust == pytest.approx(SUPPORT, abs=1e-9)


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
    assert abs(output.yaw_rate_rad_s) <= 0.15 * 0.5 + 1e-9


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


def test_predict_then_search_on_fresh_empty_frames():
    controller = _tracked_controller(_track("A", 0.05, 0.0, scale=0.10))
    now = 100.06
    controller.observe(_update([], frame_id=5), now_s=now)
    assert controller.state is CleanCourseState.TRACK  # one missed frame
    now += 0.05
    controller.observe(_update([], frame_id=6), now_s=now)
    assert controller.state is CleanCourseState.PREDICT
    # Fresh-but-empty frames are not staleness; the same controller keeps
    # producing finite bounded commands on the predicted state.
    output = _command(controller, now + 0.02)
    assert output.thrust > 0.0
    for _ in range(30):
        now += 0.033
        controller.observe(_update([], frame_id=7), now_s=now)
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
    ) == (0.0, 0.0, 0.0)
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
        assert abs(output.yaw_rate_rad_s) <= 0.25 + 1e-9
        assert output.thrust == 0.0 or 0.21 <= output.thrust <= 0.34
        assert abs(output.target_roll_rad) <= 0.25 + 1e-9
        assert -0.35 <= output.target_pitch_rad <= 0.15 + 1e-9
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
        if command.thrust == pytest.approx(0.275, abs=1e-6)
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
        host.estimate, target_roll_rad=0.0, target_pitch_rad=0.0, thrust=0.275
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
        if command.thrust == pytest.approx(0.275, abs=5e-3)
    ]
    assert support_sends  # the bounded coast wait emitted support sends
    for command in support_sends:
        assert math.isfinite(command.roll_rate)
        assert math.isfinite(command.pitch_rate)
        assert command.yaw_rate == 0.0
    assert summary["final_gate_index"] == 1
    assert summary["success"] is True
