"""Clean minimal VQ2 visual-course navigation stage (architecture reset M2).

This module replaces the retired ``aigp_vq2_visual_course_stage`` coordinator
as the navigation owner for the powered ``visual-course`` stage.  It carries
exactly five runtime states (``TRACK``, ``PREDICT``, ``COAST_FOR_CREDIT``,
``COMMIT``, ``SEARCH``), one small variable-dt estimator per retained target
hypothesis, one continuous control law, one attitude PD, one explicit yaw
channel, one transparent final clamp, and one atomic race-active send per
tick.

Authority model:

- ``active_gate_index`` increments and ``race_finished`` are authoritative.
  They are accepted immediately as events; vision never vetoes race credit
  and never declares a pass.
- ``track_id`` is a local visual-continuity hypothesis only, never a gate
  number.
- The July-18 credible-crossing wait survives as the single
  ``COAST_FOR_CREDIT`` state: after a credible close crossing loses the
  target on a FRESH camera frame, latch zero-rate/zero-thrust for exactly
  ONE wire-zero send, bounded by the send count rather than a timeout (the
  contract bound is AT MOST 0.40 s of exact zero; every timed window —
  0.25/0.10/0.06 s — still paid a multi-tick ballistic drop at the plane:
  F68/F69/F71).  Credit is accepted in ANY state, so the wait continues as
  a normal ``SEARCH`` after the single zero.  A
  superseded/frozen frame (same camera-frame identity republished during a
  camera stall) must never arm the coast; it goes to ``PREDICT`` with
  covariance inflation instead (flight 20260729T085719Z-visual-course-
  4455fd61).

Control-law constant sources:

- ``SUPPORT_COLLECTIVE`` / ``VERTICAL_ERROR_GAIN`` / ``VERTICAL_RATE_GAIN`` /
  error and rate bounds: the live-proved Gate-0 collective law
  (``_gate0_proved_vertical_collective`` in the retired stage).
- ``VERTICAL_FEEDBACK_SIGN``: empirically confirmed by the 2026-07-29
  crossing-geometry analysis; see the comment at its definition.
- ``YAW_ERROR_SIGN`` / ``ROLL_ERROR_SIGN``: the 2026-07-29 crossing-geometry
  analysis (Q5) falsified the retired controller's lateral direction
  post-credit; see the comments at their definitions.  Magnitudes are the
  proved gate-1-recenter roll gain and the visual-align yaw gain.
- ``GATE0_CLIMB_VERTICAL_OFFSET_NORM``: DISABLED (0.0) after three gate-0
  top-bar strikes showed any positive pre-crossing climb bias is re-climbed
  to and produces unrecoverable overshoot; see the comment at its
  definition.  The closure-scaling machinery remains tested for possible
  post-credit reuse.
- ``VZ_CLIMB_CAP_M_S`` / ``VZ_GOVERNOR_GAIN`` / ``VZ_LEAK_TAU_S``: the
  IMU-based world-vertical-rate climb governor added after the fourth
  top-bar collision showed bearing pursuit builds unbounded vz; see the
  comment at its definition.  It supersedes the removed flight-2
  D-direction limiter as the honest rate-limit mechanism.  A symmetric
  proportional descent floor extends it; see the constant blocks.
- ``CLOSURE_TARGET_RATE_S`` / ``CLOSURE_FULL_BRAKE_RATE_S``: the vision
  closure-rate governor (F31).  The filtered log-scale expansion rate is
  the only honest closure signal — fh is a signless drag magnitude that
  conflates speed with braking — so speed is capped CONTINUOUSLY at every
  range: the pitch target blends from the advance law toward the gentle
  ``PRE_CROSS_BRAKE_PITCH_RAD`` attitude as the expansion rate rises past
  the target.  This replaces the retired fh closure governor (wrong
  signal), the near-field log_scale/TTC triggers (late), and the
  post-credit brake window (all deleted with F31).  The COAST latch is
  exact wire zero on every channel (July-18 contract item 9, restored
  2026-07-30; the F25/F26 support-thrust coast through the attitude PD is
  out of contract).
- ``FH_UNTRUSTED_*``: the F14 inflow-regime gate.  vz_est is invalidated
  by REGIME (a smooth fh-proportional thrust deficit), not attitude or
  vibration, so sustained fh > 5.0 freezes the vz/alt integrators,
  suppresses every vz-based governor floor/cap, and
  falls back to the camera-qualified vertical PD (support + margin when
  unqualified); the latch releases below fh 2.0.  (Trigger raised 3.0 ->
  5.0 in F50: the F49 hard brake reads fh 3.0-3.4 and tripped its own
  distrust alarm; the F14 biased regime measured 6.5-7.5.)
- Thrust envelope ``[MIN_COURSE_THRUST, MAX_COURSE_THRUST]`` and yaw cap: the
  accepted v3 yaw profile and the visual-course thrust envelope from the
  July-18 safety contract (max raised 0.32 -> 0.34 under the 0.35 hard
  abort for the fast-regime hover shift).

This module never imports the runner; the async loop receives the runner as a
duck-typed host plus an explicit :class:`CleanCourseRuntime` primitive bundle,
mirroring the seam style of the retired stage module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from competition.vq2_contracts import FrameEdge
from scripts.aigp_vq2_yaw_profile import (
    DEFAULT_YAW_CALIBRATION_PROFILE_PATH,
    YAW_CALIBRATION_PLAN_ID,
    YAW_CALIBRATION_PLAN_SHA256,
    YAW_CALIBRATION_PROFILE_ID,
    YAW_CALIBRATION_PROFILE_SCHEMA,
    YAW_CALIBRATION_PROFILE_SHA256,
    YAW_CALIBRATION_SOURCE_COMMIT,
    YAW_CONTROLLER_TO_BODY_SIGN,
    YAW_CONTROLLER_TO_IMAGE_SIGN,
    YAW_CONTROL_HOLD_HORIZON_S,
    YAW_MAX_CALIBRATION_ATTITUDE_EXCURSION_RAD,
    YAW_MAX_CALIBRATION_MEASURED_RATE_RAD_S,
    YAW_MAX_COMMAND_RATE_RAD_S,
    YAW_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S,
    YAW_MAX_GYRO_RESPONSE_DELAY_S,
    YAW_OBSERVED_MAX_MEASURED_RATE_RAD_S,
    load_yaw_calibration_profile,
    yaw_calibration_profile_evidence,
)

# ---------------------------------------------------------------------------
# Control-law constants (see module docstring for sources).
# ---------------------------------------------------------------------------

# Global sign of the stable vertical feedback with the image-down vertical
# error.  EMPIRICALLY CONFIRMED by the 2026-07-29 crossing-geometry analysis
# (`docs/aigp/2026-07-29-vq2-crossing-geometry-analysis.md`, Q2): in the clean
# pre-credit identification condition, more collective climbs (corr +0.913
# with world-up acceleration) and the target moves DOWN in frame (residual
# corr +0.130 after pitch-derotation), so `BASE - K*e` is the stabilizing
# sign at every gate.  One global sign; gate-0 takeoff boost is feedforward
# only and does not change it.
VERTICAL_FEEDBACK_SIGN = -1.0

# F49 (20260730 flight-48 measurement): the true hover support in the clean
# pre-credit condition is 0.247, not the 0.275 carried from gate-0 proving —
# at 0.275 every "level" hold climbed ~+0.3 m/s into the top-bar geometry.
SUPPORT_COLLECTIVE = 0.247  # F48-measured hover support collective
VERTICAL_ERROR_GAIN = 0.080  # GATE0_PROVED_COLLECTIVE_ERROR_GAIN
VERTICAL_RATE_GAIN = 0.126  # GATE0_PROVED_COLLECTIVE_RATE_GAIN
VERTICAL_MAX_ABS_ERROR_NORM = 0.50  # GATE0_PROVED_COLLECTIVE_MAX_ABS_ERROR
VERTICAL_MAX_ABS_RATE_NORM_S = 5.0 / 3.0  # GATE0_PROVED_COLLECTIVE_MAX_ABS_RATE

MIN_COURSE_THRUST = 0.21  # MIN_VISUAL_THRUST (active visual-course envelope)
# Raised 0.32 -> 0.34 (flights 20260729T114842Z-visual-course-039186c8 and
# bf13f18's F10): carrying gate-0 attack speed into the post-credit phase
# collapsed thrust effectiveness (VRS-like fast regime: measured effective
# hover 0.335-0.36 while the low-speed fit predicts +2.9 m/s^2 at 0.32).
# The runner's hard envelope abort stays 0.35 (validate_command), and this
# also restores proportional descent-floor headroom: the vz = -1.0 floor
# target 0.33 was clipped by the old 0.32 clamp.
MAX_COURSE_THRUST = 0.34  # MAX_VISUAL_THRUST, below the 0.35 hard abort

# IMU-based world-vertical-rate governor (climb only).  Four consecutive
# gate-0 top-bar collisions (flights 20260729T085719Z-visual-course-4455fd61,
# 20260729T094736Z-visual-course-9d430a40, 95644bf5, and 4dbe4b8c) showed
# bearing-pursuit vertical control necessarily builds UNBOUNDED vz: every
# flight peaked at 2.8-3.35 m/s within 1.3 s against a required ~2 m climb
# over ~2.2 s (~0.9 m/s average) and arrived at the gate plane with vz
# +0.6..+1.0 m/s.  Measured plant (199 airborne samples, all four flights):
# a_up = 66.7*thrust - 18.44 (residual sigma 0.59; hover at 0.277 ~=
# support 0.275), so 0.01 collective ~= 0.67 m/s^2 and 2 m/s over the cap
# costs -0.06 collective ~= -4 m/s^2, inside the 0.21 floor's authority.
# The governor is IMU-based precisely so it stays alive when vision is
# censored or stalled.
VZ_CLIMB_CAP_M_S = 1.0  # generous bound vs the ~0.9 m/s average requirement
VZ_GOVERNOR_GAIN = 0.03  # collective per m/s over the cap (see block above)
# F68 (20260730T052644Z-visual-course-0d49884b): the 1.0 m/s cap is a gate-0
# climb-out bound, but gate-1+ legs inherit it — the gate-1 approach climbed
# at vz +1.0 through the final 1.5 s and overshot ~0.4 m ABOVE the gate, the
# aperture died at the frame bottom before the geometry could recover, and
# the blind SEARCH drifted into the structure.  Inter-gate vertical
# corrections are ~0.2-0.3 m/s, so course legs get a 0.5 m/s cap with real
# teeth (0.10/m/s, SUBTRACTIVE from the PD demand — a min()-against-support
# cap would bang-bang the demand the way the F64 descent step did).  The
# vertical FEEDBACK law stays one global PD at every gate; this is a safety
# envelope, and envelopes already differ by leg.
VZ_CLIMB_CAP_COURSE_M_S = 0.5  # gate-1+ leg climb-rate bound
VZ_GOVERNOR_COURSE_GAIN = 0.10  # continuous subtraction per m/s over it
VZ_LEAK_TAU_S = 2.5  # leaky-integrator time constant (bias/noise guard)
GRAVITY_M_S2 = 9.80665  # ImuAttitudeConfig.gravity_mps2
# Symmetric descent floor (flight 20260729T111003Z-visual-course-d52adcd4):
# a 6.1 s frozen-camera stall blinded the loop while a_up ~= -1.9 m/s^2 sank
# it into a ground graze.  The hover regime shifts ~0.275 -> ~0.30 support
# in fast/descending flight.  F63 (20260730T015429Z-visual-course-5e550551)
# proved the original floor too weak AND too late: an established -0.5..-1.5
# sink ran ~5 s unarrested at thrust 0.30-0.33 (fast-regime hover ~=0.32)
# and the drone passed UNDER gate 1 into the floor.  F67: the F64 law
# (0.10/m/s plus a +0.04 step feedforward) bang-banged at the plane — F65's
# last 0.5 s alternated 0.31 <-> 0.22 tick-to-tick as vz_est straddled the
# floor, each high tick re-climbing toward the top panel — so the step is
# DELETED and the gain raised 0.10 -> 0.21, which reproduces the F64
# authority at vz -0.7 (support + 0.0735 ~= the proved 0.075) while staying
# continuous at the boundary.  Engage at -0.35 m/s (before momentum builds);
# the command saturates at the 0.34 envelope top by vz ~= -0.8.
VZ_DESCENT_FLOOR_M_S = -0.35  # sink-rate bound, mirroring the climb cap
VZ_DESCENT_GOVERNOR_GAIN = 0.21  # collective per m/s below the floor
# F96: this arrest now applies to GATE 0 near the plane ONLY — the
# gate-1+ qualified path it was built for is the unified vz-tracking
# law (see COURSE_VZ_* below), which carries the same "desired vz -> 0
# as ey -> center" semantics as its setpoint.  The F78/F91/F93 history
# below is kept for provenance.
# F78 vertical arrival arrest (20260730T082159Z-visual-course-7e18243d):
# the gate-1 approach climbed at vz +0.65 THROUGH the opening — the
# compensated error sat at ey ~-0.10 (gate just above center), the PD's
# small P-term kept commanding climb, and the 0.5 m/s course governor
# only caps the RATE, so the gate sank ey -0.10 -> +0.16/+0.27 before
# censorship and the crossing went high.  The governor binds only at the
# cap; what is missing is a STOPPING law: as the compensated error
# approaches center the desired vertical velocity must approach zero.
# While the IMU vz is positive, allowed climb is proportional to the
# REMAINING error (vz_allow = ARREST_VZ_PER_NORM * |ey|); any excess
# subtracts collective continuously.  Descents (vz <= 0) are untouched —
# no blanket reduction, no sink; the descent floor below still bounds
# any sag, and the F14 fh-untrusted latch (frozen vz) keeps its
# no-vz-adjustment contract.  F78b: gate-1+ legs only — on gate 0 it
# arrested the proved climb-out and armed the pre-gate-1 altitude floor
# (20260730T085104Z-visual-course-34b59dea; the latch itself is deleted in
# F92 after four ballooned gate-1 legs).  F91: two-sided — F90
# climbed vz +0.3..+0.4 with the gate CENTERED (ey ~0, no climb command,
# so the one-sided arrest never engaged), ballooned ~0.5 m over the gate
# under the F14 latch, lost it out the frame bottom, and dove into the
# ground (20260730T134602Z-visual-course-6e302725, id 1002).  With the
# gate at/below center any positive vz is energy away from the aim.
VERTICAL_ARREST_VZ_PER_NORM = 1.0  # m/s allowed climb per norm of |ey|
VERTICAL_ARREST_COLLECTIVE_GAIN = 0.15  # subtraction per m/s of excess

# F96 unified course-leg vertical law (flight
# 20260730T153947Z-visual-course-a2741311).  F95 removed the support
# inflation and the one-sided balloon with it, but exposed the limit
# cycle underneath: on the gate-1+ qualified TRACK path vz feedback was
# stacked four-deep and incoherent — the camera D term (0.126*vy, a
# lagged image copy of vz), the two-sided arrest (0.15*excess over a
# deadband), the vz-center trim integral, and the course governor
# (climb cap 0.10 + descent floor 0.21).  Each was tuned as if it were
# the only vz term; in phase they summed ~0.13 collective (~2.5 m/s^2)
# for a 0.4 m/s vz error, so the leg oscillated clamp-to-clamp
# (thrust 0.21 <-> 0.34, vz +/-0.4..0.5), |vz| never met COMMIT's
# <= 0.25 entry budget at the plane, the gate engulfed the frame with
# no crossing path, and the blind search flew into the ground (id
# 1002).  The replacement is ONE vz-tracking law on gate-1+ qualified
# fh-trusted TRACK: the desired climb rate is proportional to the
# compensated error (the arrest's per-norm allowance as a SETPOINT, no
# deadband — it goes to zero as ey approaches center), and a single
# coherent gain tracks it.  Gate 0 keeps its proved PD envelope; the
# F14 fh-untrusted path keeps the camera PD (a frozen vz_est cannot be
# tracked); blind paths keep their support floors and the governor.
COURSE_VZ_DES_PER_NORM = 1.0  # m/s desired climb per norm of compensated ey
COURSE_VZ_DES_MAX_M_S = 0.5  # matches the VZ_CLIMB_CAP_COURSE_M_S envelope
COURSE_VZ_TRACK_GAIN = 0.12  # collective per m/s of vz tracking error
# F97 (20260730T160909Z-visual-course-a4bfb6d3): F96 flew smoothly but the
# tracker's vz_des saturated at -0.5 pulling a low-sitting gate to center,
# holding vz -0.46 into the plane — COMMIT's |vz| <= 0.25 entry budget
# vetoed the crossing exactly as F95's oscillation residue had.  The
# setpoint clamp and the entry budget were in direct spec conflict.  Near
# the commit regime the setpoint is capped at 0.20 (inside the budget with
# margin), ramping continuously from the far value so the vertical
# correction happens earlier in the approach, not at the plane.
COURSE_VZ_DES_COMMIT_M_S = 0.20  # vz_des cap at/inside commit_min_log_scale
COURSE_VZ_DES_RAMP_START_LOG_SCALE = -2.0  # far end of the cap ramp
# F97: the same flight's mid-leg overshoot (alt +0.19 with ey ~0) was
# propped by the vz-center trim, wound to its cap during the blind
# post-crossing window (F90 blind anti-sink, working as designed) and
# leaking back at only 0.02/s — it fought the tracker for the whole leg.
# A wound positive trim while the tracker wants LESS climb than the
# measured vz is suspect by construction: bleed it fast.
COURSE_VZ_TRIM_FAST_LEAK_S = 0.10  # collective/s when trim fights the tracker

# Launch boost is pure feedforward (it ignores ey).  Flight
# 20260729T094736Z-visual-course-9d430a40: the 0.32 x 0.75 s boost alone
# built vz ~= +2.3 m/s by t=0.75 (~70% of the +3.2 m/s peak climb velocity)
# and the trajectory overshot the ~1.8-2 m required climb into gate 0's top
# bar.  Cut to 0.30 x 0.40 s: 0.30 stays inside the historically validated
# 0.30..0.32 launch-thrust range (aigp_vq2_visual_config); the shorter
# duration is the main lever and deliberately departs from the retired
# visual config's 0.45..0.60 lifecycle window, which never bound this stage.
LAUNCH_BOOST_THRUST = 0.30
LAUNCH_BOOST_DURATION_S = 0.40

# Gate-0-phase feedforward vertical setpoint offset (image-down norm).
# DISABLED (0.0) after three consecutive gate-0 top-bar strikes
# (20260729T085719Z-4455fd61, 20260729T094736Z-9d430a40,
# 20260729T100733Z-95644bf5).  The 2026-07-29 crossing-geometry analysis
# (Q1/Q4) motivated a +0.25 bias to cross higher and see gate 1 sooner,
# but the spawn geometry already requires ~1.8-2 m of climb, so the loop
# climbs anyway: any positive offset is re-climbed to (at the 0.32
# ceiling right after boost) and peak vz was ~3.2 m/s in ALL THREE
# flights while the [0.21, 0.32] envelope can only erase ~2.5 m/s in the
# remaining descent window.  The ramp-to-crossing and >=0-center clamps
# trimmed the aim (ey 0.44 -> 0.40 -> 0.20 at close scale) but not the
# energy.  With 0.0 the modeled peak vz is ~1.2-1.5 m/s and the aim
# settles inside the aperture.  If gate-1 post-crossing visibility needs
# extra height, add it as a POST-CREDIT climb, not a pre-crossing bias.
GATE0_CLIMB_VERTICAL_OFFSET_NORM = 0.0

# Reference (spawn) detection log scale for the closure-scaled gate-0 climb
# bias.  Flight 20260729T085719Z-visual-course-4455fd61 spawned with gate 0
# detected at bbox (282,134,80,80) in a 640x360 frame: apparent_scale =
# sqrt((80/640)*(80/360)) = 0.1667, ln(0.1667) = -1.79.  Cross-referenced
# with docs/aigp/2026-07-29-vq2-crossing-geometry-analysis.md.
GATE0_CLIMB_REFERENCE_LOG_SCALE = -1.79

# Lateral error signs, per the 2026-07-29 crossing-geometry analysis (Q5):
# post-credit gate 1 sits at median x=+0.57 while the retired controller
# pinned yaw at -0.150 and the target moved AWAY from center in 75% of pairs
# (+0.91 norm/s mean).  A negative yaw rotates the camera left, pushing a
# right-side target further right; recentering x>0 therefore requires a
# POSITIVE yaw.  The report notes a forward-closure expansion confound (pure
# closure also drifts off-axis features radially outward), but the rotational
# contribution of the old yaw was independently away-from-center.
YAW_ERROR_SIGN = +1.0  # flip this one line if the first flight contradicts
# 0.30 -> 0.50 (post-credit pursuit redesign, F26/F27/L13/L18 trace
# analysis): the initial post-credit yaw direction was already correct, but
# 0.3*ex = 0.17 rad/s at the typical x ~= +0.57 gate-1 handoff bearing could
# not center a NEAR off-axis gate before translation parallax swept it to
# the frame edge.  The 0.25 cap is reached at |ex| >= 0.5, well inside the
# measured ~0.5 rad/s plant capability.
# 0.50 -> 0.90 (F35, d25f23fe): the whole 4.5 s gate-1 leg ran with yaw
# SATURATED at the 0.25 cap while the gate bearing grew +0.35 -> +0.95 —
# maximum authority was still too little turn.  With the cap raised to the
# measured 0.5 rad/s plant authority, 0.9 puts the cap on at |ex| >= 0.55.
YAW_ERROR_GAIN = 0.90
# Roll: the old saturated +/-0.25 roll oscillation never recentered either
# (corr(roll_cmd, dx/dt)=+0.18 is too weak/saturated to identify the roll
# channel), so the roll sign follows the yaw verdict: bank INTO the
# correction (positive bank toward a right-side target), coordinated with
# the positive yaw, translating the vehicle toward the gate's lateral
# position so the target bearing moves toward center.
ROLL_ERROR_SIGN = +1.0  # flip this one line if the first flight contradicts
# 0.24 -> 0.50 (F39b, 4a4b7792): the gate-1 pursuit yawed at the 0.5 rad/s
# cap while the close gate's bearing STILL escaped +0.44 -> +0.95 — pure
# yaw cannot hold a close off-axis gate against translation parallax;
# lateral translation (bank) is what bends the path.  The old gain built
# only ~2.4 m/s^2 and the 0.30 rad/s generic slew never attained even
# that inside the 0.7 s window.
ROLL_ERROR_GAIN = 0.50
# 0.12 -> 0.25 rad (post-credit pursuit redesign): yaw alone cannot center a
# near off-axis gate — the trace shows ex GROWING (+0.19 -> +0.95) while the
# drone yawed +0.5 rad toward it, because momentum keeps the path straight
# (nose != path).  Lateral translation is what bends the path; 0.12 rad
# (6.9 deg, ~1.2 m/s^2) was too weak.  0.25 rad (14.3 deg, ~2.5 m/s^2) stays
# far inside the runner's 25 deg roll abort.
# 0.25 -> 0.35 rad (F39b): 20 deg bank ~= 3.4 m/s^2 lateral, still inside
# the runner's 25 deg roll abort with margin.
MAX_TARGET_ROLL_RAD = 0.35  # coordinated-turn lateral translation cap
# The generic 0.30 rad/s target slew made even the OLD roll cap
# unattainable inside a pursuit window; large pursuit errors get the fast
# slew (same value as the brake pitch slew).
ROLL_PURSUIT_SLEW_RAD_S = 1.0  # fast roll slew while |ex| is large
ROLL_PURSUIT_FAST_EX_NORM = 0.30  # |ex| above this engages the fast slew
# Contract correction (2026-07-30): the PRODUCTION yaw command cap is
# restored to +/-0.15 rad/s.  The 0.50 carried here since F35 (itself raised
# from 0.25 after flights 4ba3922b/89a175a9/d058b8a0) is the calibrated
# profile's max_abs_MEASURED_yaw_rate (plant response to a 0.15 command
# with the ~2.3x build-3385 overgain), never command authority — the v3
# production profile commanded 0.15.  The runner's measured-authority
# guard compares limits against the measured envelope, so 0.15 still passes.
MAX_COURSE_YAW_RATE_RAD_S = 0.15  # production yaw COMMAND cap (v3 profile)

# Softened -0.18 -> -0.12 (flight 4ba3922b), then -0.12 -> -0.08 (F29,
# d8169633): fh is a SIGNLESS magnitude (hypot of world horizontal specific
# force), so hard braking reads exactly like the fast glide — the F29 brake
# stack (pre-cross +0.12 for 0.9 s then pcb +0.15 for 1.3 s) drove fh to
# 4-5, tripped the untrusted latch on the braking itself, and put the drone
# in the inflow/VRS regime where even the 0.34 clamp lost ~2 m/s^2 of lift
# (az -7.9 at full collective) — terrain at the gate while braking at full
# throttle.  (Written under the INVERTED pre-F38 pitch convention: the
# "brake stack" was in truth a powered dive; see the F38 block below.)
# --- F38/F49 VERIFIED PITCH CONVENTION (build 3385) ------------------------
# The controller long assumed POSITIVE pitch target = nose-up (brake) and
# NEGATIVE = nose-down (advance).  The F37 trace (62bc5772) disproved it:
# (1) the spawn attitude is -0.31 with the gate span FLAT (0.13 for 0.38 s)
#     — negative pitch holds the drone nearly stationary;
# (2) the "brake" target +0.15 was fully attained and closure ACCELERATED
#     (span 0.32 -> 1.00 in 0.5 s, log-rate ~2.3/s) into the gate-0 impact;
# (3) the "advance" target ~-0.06 nearly STOPPED closure.
# So in this build positive pitch = nose-DOWN (accelerate), negative =
# nose-UP (brake).  Every pre-F38 "brake" episode (F29/F31/F32/F34/F36/F37)
# was a powered dive into the gate — this is why the brake "never worked"
# and why fh (thrust-tilt magnitude) always "grew through the brake".
# --- F49 SPAWN-RELATIVE PITCH TARGETS --------------------------------------
# F48 then showed the F38 targets were still wrong in ABSOLUTE terms: level
# flight is the SPAWN attitude spawn_pitch ~= -0.31 rad, not 0.  Every
# absolute target below (brake 0.02, advance 0.08, pre-cross -0.28) was
# therefore ~0.3 rad nose-down of intent — the "near-level brake" was a
# sustained dive.  The constants are now OFFSETS from the measured spawn
# attitude: effective target = spawn_pitch_rad + offset, so a POSITIVE
# offset is nose-down (advance) relative to level and a NEGATIVE offset is
# nose-up (brake).  The runner passes the real spawn pitch; the default is
# the measured -0.31.
SPAWN_PITCH_RAD_DEFAULT = -0.31  # measured build-3385 level-flight attitude
ADVANCE_PITCH_RAD = 0.08  # nose-down closure OFFSET when aligned/confident
BRAKE_PITCH_RAD = 0.00  # near-level braking OFFSET (spawn attitude = level)
ANGULAR_FULL_BRAKE_NORM = 0.35  # angular error that fully suppresses advance
# F46 (20260729T211439Z-visual-course-4356c153): err ~0.5 gave brake demand
# ~0.85 -> target pitch -0.11, attained, but the approach still closed
# ~2.6x in 1.2 s and the bearing stalled at -0.37 — velocity never bent
# (nose chased the LOS, path orbited past the gate's left side).  Speed is
# what beats yaw/roll authority via parallax, so saturate the brake early:
# at err >= 0.35 the approach is a hover-turn, not a powered flyby.
EXPANSION_BRAKE_FREE_S = 1.5  # expansion rate below which no braking applies
EXPANSION_BRAKE_SPAN_S = 3.0  # span from free advance to full expansion brake
NEAR_FREE_LOG_SCALE = -1.5  # far enough that near-plane risk does not brake
NEAR_BRAKE_LOG_SCALE = -0.9  # close enough that closure is fully braked
# F57 (20260730T003044Z-visual-course-74abd688 + f56/ frames): the F56
# corridor correctly blocked mis-aimed commits, but the pursuit NEVER
# satisfied it — ex stalled at -0.15..-0.18 for the whole approach (a
# P-pursuit limit cycle: yaw gain 0.9 commands only -0.14 rad/s against
# close-range parallax while the yaw cap is 0.5 and the airframe measurably
# responds to >=0.42).  Inside the COMMIT proximity regime both lateral
# error gains are boosted so ex actually converges into the corridor before
# censorship; the caps are unchanged (at ex -0.16 the boosted yaw command
# is -0.36, inside the cap and the measured response).  Far range keeps
# the proved 0.9/0.5 gains.
NEAR_PLANE_STEER_GAIN_MULT = 2.5  # near-regime lateral gain multiplier

CROSSING_MIN_LOG_SCALE = -0.80  # retired stage crossing_arm_min_log_scale
# F45 (20260729T210351Z-visual-course-b1f5e89f): the crossing coast is
# ballistic (yaw 0, support collective), so arming it on an OFF-CENTER
# close loss preserves the offset — the last measured bearing (-0.39,-0.28)
# slid -0.69 -> -0.93 censored and out of frame, no authoritative credit
# came, and the leg fell into blind-search churn (impulse 4.91).  Race
# credit is authoritative and arrives via race packet whenever the drone
# truly passes the plane — the stage does not need the coast to OBTAIN
# credit, so the coast may only own the 0.4 s wait of an ALIGNED crossing:
# the last freshly MEASURED bearing must sit inside these bounds.
CROSSING_MAX_ABS_EX_NORM = 0.20  # |ex| bound to arm the crossing coast
CROSSING_MAX_ABS_EY_NORM = 0.25  # |ey| bound to arm the crossing coast
# The alignment check reads the last MEASURED bearing, never a long-
# predicted one; the horizon matches the engulfing anchor freshness (the
# engulfed plane is blind, so a genuine centered crossing's last accepted
# measurement can be this old at the loss).
CROSSING_MEAS_MAX_AGE_S = 0.50
# F68 (20260730T053935Z-visual-course-1e82777a): the cruise-phase race
# stream publishes at ~4 Hz, so "wait for a newer packet" held the exact
# zero for a full 0.25 s — at 1 m altitude with forward speed the ballistic
# + fast-regime lift-loss collapse measured -12.6 m/s^2 (vz +0.33 -> -2.83,
# alt -0.35 -> -0.81) and the recovery could not arrest before terrain.
# F69 (20260730T055004Z-visual-course-352d481c): even a 0.10-0.14 s window
# cost vz -1.7 at the plane and the 0.34-clamped recovery (~2 m/s^2 net
# over fast-regime hover) could not re-climb 0.06-0.1 m in the 0.3 s before
# the bottom-bar graze.  F71 (20260730T061726Z-visual-course-fa8cb298): the
# 0.06 s window still ended in a post-coast recovery excursion that tripped
# the roll (-40.8 deg) and body-rate (46.5 rad/s) limits (collision id
# 1002, impulse 8.32).  F72: the credit wait is exactly ONE wire-zero SEND,
# bounded by the send count rather than any timeout value — the smallest
# window the July-18 contract (AT MOST 0.40 s of exact zero) admits, and
# the only one that never pays a multi-tick ballistic drop.  Credit does
# not depend on the window: a true pass is accepted in ANY state (F67
# credited gate 0 in SEARCH 0.5 s after its coast exited).
# F53 (20260729T233602Z-visual-course-072c8a7b): past near_brake_log_scale
# the misalignment brake self-locks — the brake attitude pushes the gate
# image down, the raw ey reads as misalignment, advance goes to 0, and the
# resulting hover starves the filter into a blind search (floor collision
# in F52).  An aligned, fresh, SUSTAINED near-plane regime commits to an
# inertial crossing (COMMIT state) instead.  Entry requires the close
# regime to persist (the F52 span stalled at 0.56, well under CROSSING_MIN
# -0.80), a fresh UNCENSORED both-axis measurement tighter than the
# crossing horizon (the bearing moves ~0.33 norm in 0.7 s at the plane, so
# a 0.5 s-old bearing is ~0.2 norm wrong), and crossing-coast alignment on
# the F50 compensated ey — gate-1+ legs only for now.
# F54 (20260729T235858Z-visual-course-c92d42ce): the F53 calibration never
# entered — censorship of the aim track began essentially SIMULTANEOUS with
# outer_log_scale crossing the -0.9 proximity threshold, so the 0.30 s
# sustain outlived the ~0.2 s fresh-uncensored window and the hover trap
# repeated into gate 1's structure.  The clean commit point in that trace
# sat at log_scale ~-1.2 (uncensored, fresh, aligned, ~0.7 s before
# censorship onset, ~1.2-1.5 s before the plane): entry proximity now arms
# at -1.2 with a 0.10 s sustain, and the timeout grows to cover the longer
# inertial leg.  NEAR_BRAKE_LOG_SCALE stays for the brake law and the F52-A
# steering hold.
COMMIT_MIN_LOG_SCALE = -1.2  # entry proximity bound (span ~0.3)
COMMIT_SUSTAIN_S = 0.10  # near-plane regime must persist this long to arm
COMMIT_MEAS_MAX_AGE_S = 0.30  # fresh uncensored both-axis window at entry
# F58 (20260730T004618Z-visual-course-cae7b894): the pre-cross brake bled
# the approach to ~0 speed at entry and the coast-sized 0.05 rad advance
# only rebuilds ~1 m/s in the window — the commit never reached the plane
# while tangential velocity carried it past the gate's face.  0.15 rad
# (~1.5 m/s^2) covers the 3-4 m brake-stall drive inside a 3.0 s window.
COMMIT_ADVANCE_PITCH_RAD = 0.15  # real advance drive from standstill
# F60's COMMIT_AIM vertical-aim pitch term was deleted in F66 (see the
# commit law): it made the attitude a second vertical channel that fought
# the collective servo at the plane (F63 dive-under, F65 top-panel slam).
COMMIT_TIMEOUT_S = 3.0  # no credit this long -> arrest and search
# ---------------------------------------------------------------------------
# 2026-07-30 unified crossing-entry budget (replaces the F55/F56/F61/F62
# patch stack — outer-bbox corridor, quiet-bearing rate gate, fixed ex bias
# — which admitted entries at the post on 0.30 s-old axes: six gate-1
# deaths, all at the plane).  ONE final-approach policy: TRACK holds/brakes
# OUTSIDE the censorship blackout until the budget below passes; COMMIT is
# the only gate-1+ crossing path; credible close loss inside an armed
# COMMIT latches the exact-zero credit wait (no blind driving).
# Entry requires, at the CURRENT frame (never a 0.30 s-old axis):
#   * a currently usable inner aperture (passage_usable) — an unfit aperture
#     at close range is the best available "about to go blind" signal;
#   * uncensored same-id measurements on BOTH axes, this frame;
#   * |error| + |rate| * COMMIT_BLACKOUT_S <= margin * aperture_half_extent
#     on BOTH axes, evaluated on the WORST of the raw measurement and the
#     filtered prediction — predicted blackout drift is part of the budget;
#   * closure/expansion at or below the governor's target rate, so the
#     crossing never starts at unarrested approach energy.
# If any term fails, TRACK keeps braking/re-centering; the entry condition
# is never loosened.
COMMIT_ENTRY_MEAS_MAX_AGE_S = 0.06  # "current frame" at ~30 Hz camera
COMMIT_BLACKOUT_S = 0.50  # measured close-range censorship window 0.3-0.6 s
COMMIT_ENTRY_APERTURE_MARGIN_FRAC = 0.60  # error+drift within 60% of half
COMMIT_ENTRY_MAX_EXPANSION_RATE_S = 0.35  # <= closure governor target
# IMU vertical velocity is part of the blackout-displacement budget too.
# F82 (20260730T112130Z-visual-course-93a8eecf): the truthful panel-gate
# aperture sits high at close range, so the gate-0 approach arrived
# below-center still climbing; the vision-only vy term under-measured the
# real +0.64 m/s, the budget admitted the entry, and the blind coast
# carried the climb into the top bar (id 1001, no credit).  F80's proved
# crossing entered at vz ~0.0 — the blackout must start with dead
# vertical energy on EVERY gate, so entry refuses |vz| above this bound
# and TRACK keeps holding/re-centering outside censorship until it dies.
COMMIT_ENTRY_MAX_VZ_M_S = 0.25
# Every pitch target stays clear of the runner's MIN_PITCH_RAD (-35 deg)
# watchdog.  F84 (20260730T121408Z-visual-course-533d563c): the F80 course
# brake target (spawn -0.31 + course brake -0.30 = -0.61 rad) sat 0.001 rad
# INSIDE the abort, and the sustained gate-1 misalignment brake slewed
# straight into it ("pitch limit exceeded (-35.0deg)") after a credited
# gate-0 crossing.  -33 deg keeps 2 deg of settling margin.
PITCH_TARGET_MIN_RAD = math.radians(-33.0)
# TRACK-phase lateral trim (2026-07-30): the gate-1 off-axis pursuit is a
# P-loop orbiting the gate — it equilibrates at ex ~-0.08 EVERY flight
# (yaw holds the bearing constant against translation parallax; the F57
# gain boost only moved the equilibrium).  A small bounded integral on the
# raw ex, active only near the linear regime and while x is fresh, learns
# the feedforward that nulls the orbit so entries arrive centered.  This
# replaces the COMMIT-only F62 0.08 bias, which compensated the entry
# offset only during the blind finish.
EX_TRIM_GAIN = 0.30  # trim norm/s per norm of sustained ex
EX_TRIM_MAX_NORM = 0.15  # anti-windup bound (~2x the measured equilibrium)
EX_TRIM_ACTIVE_EX_NORM = 0.25  # integrate only inside the near regime
# 2026-07-30 TRACK-phase vertical centering trim (the vertical EX_TRIM):
# the ey PD around center is far too weak to hold altitude (error gain
# 0.080/norm: the measured -0.03 droop yields +0.002 collective), so a
# slightly-underestimated support settles ON the -0.35 m/s descent-floor
# boundary as a STABLE equilibrium — continuous by F65 design, so the
# floor adds exactly zero there — or, under a larger brake-attitude
# deficit, droops BELOW it.  F87 (20260730T125108Z-visual-course-
# 95059527) rode vz ~-0.35 with ey ~0 through the whole gate-0 final
# approach, arrived ~0.3-0.4 m low, and the credible-loss zero coast
# dropped it onto the lower lip (id 1001, no credit); F86 sank the
# gate-1 leg alt -0.4 -> -2.0 the same way (ey ~0 is degenerate:
# approach + descent keeps the gate centered).  While TRACK holds the
# gate at/above center with a qualified vertical channel, integrate a
# bounded upward trim against the residual sink until vz ~0; hold it on
# demanded descents (ey > active bound, where the P law owns the
# descent) and leak it on real climbs.  One-sided by design: the F78
# arrest already owns climb overshoot.  F88 (20260730T131916Z-visual-
# course-9bde29e4): gate 0 CREDITED with dead vertical energy (vz +0.08,
# alt +0.11 at the coast) after the trim arrested the mid-approach sink
# — then the gate-1 leg sank at vz -0.55 with ey +0.03 (a far-left gate
# under misalignment braking) and hit the ground (id 1002) with the
# trim gated out by the original 0.02 ey bound.  The P law's descent
# demand at |ey| 0.10 is 0.008 collective — negligible — so the active
# band widens to 0.10: a genuine low gate (ey > 0.10) still holds the
# trim, anything nearer center learns.  F89 (20260730T132758Z-visual-
# course-ef525c4c): gate 0 credited again, but the gate-1 leg's sink
# (vz -0.47 sustained) developed while the vertical channel was
# UNQUALIFIED (stale y-axis) — the trim lived inside the qualified
# branch and could not engage — and the following blind SEARCH held
# bare support+margin for 5+ s of vz -0.45..-0.48 into the ground
# (id 1002).  F90: the integrator is a shared helper; the blind paths
# (SEARCH, pending-credit hold, unqualified TRACK/PREDICT) add the
# trim to their support holds and wind it with NO ey gate — a blind
# path holds altitude by definition, so any IMU-trusted sink is
# unwanted.  COAST (exact-zero contract) and COMMIT (its own bounded
# crossing servo) never wind or consume it.
VZ_CENTER_TRIM_GAIN = 0.08  # collective per (m/s of sink) per second
VZ_CENTER_TRIM_MAX = 0.06  # anti-windup bound (~2x the measured deficit)
VZ_CENTER_TRIM_ACTIVE_EY_NORM = 0.10  # integrate only near center (TRACK)
VZ_CENTER_TRIM_SINK_M_S = -0.10  # deadband above estimator noise
VZ_CENTER_TRIM_LEAK_S = 0.02  # collective/s release on real climbs

PREDICT_FRAME_GAP_S = 0.25  # ~8 camera frames; 0.06 (~2) flapped TRACK/SEARCH
# 7 times in the 4.2 s F35 gate-1 leg, each flap dumping the pursuit fix.
PREDICT_MAX_GAP_S = 0.50  # short-gap bound before SEARCH
# Hard wall-clock cap on PREDICT (flight 20260729T112603Z-visual-course-
# d5e89c2b): an engulfing anchor refreshed every fresh frame by a MISSED
# (never-retired authoritative-current) tracker track parked PREDICT ~4 s
# while the camera streamed normally — every anchor expiry rule lives in
# observe(), so no freshness gate could end it.  command() runs every tick,
# so it forces SEARCH once the last ACCEPTED measurement is this old,
# regardless of anchor state.  1.5 s is 3x the anchor horizon: long enough
# that a genuine engulfed crossing keeps its SEARCH suppression, short
# enough to end a blind park before the ground does.
PREDICT_STALL_FORCE_SEARCH_S = 1.50
# X-axis steering freshness (F40, 20260729T193134Z-visual-course-63ed6342):
# at the gate-1 promotion the controller adopted an edge-clipped splinter of
# the just-crossed gate's frame whose x-axis had NEVER been measured
# (seeded pos_var 0.01) and steered yaw at full authority on the garbage
# point — 0.5 s of max yaw the wrong way threw the real gate off-frame.  An
# x measurement older than this (or never taken) may not command yaw/roll.
X_STEER_MAX_AGE_S = 0.5
# F42 (20260729T201743Z-visual-course-1e24b6d2) anti-deadlock: an adopted
# debris splinter whose x-axis could NEVER be measured held TRACK for 0.8 s
# with the x-steer gate freezing yaw/roll at 0 until the splinter died on
# its own.  An unmeasurable hypothesis this old forces SEARCH in observe().
UNMEASURED_X_FORCE_SEARCH_S = 0.75
# Closure-rate governor (F31 redesign after F26/F28/F29/F30 all died the
# same way — terrain at the gate-1 threshold in the high-drag regime with
# frozen estimates): vision expansion rate is the ONLY honest closure
# signal (fh is a signless drag magnitude that conflates speed with
# braking).  Speed is now capped CONTINUOUSLY at every range: as the
# filtered log-scale rate rises past the target, the pitch target blends
# from the advance law toward the gentle brake attitude, reaching it at the
# full-brake rate.  This replaces the fh closure governor (wrong signal),
# the near-field log_scale/TTC triggers (late), and the post-credit brake
# (no job left once crossings are slow — and its hard nose-up episodes
# were themselves VRS generators).
CLOSURE_TARGET_RATE_S = 0.35  # log-scale rate the governor holds (TTC ~3 s)
CLOSURE_FULL_BRAKE_RATE_S = 0.60  # rate at which the full brake pitch applies
# F96: the F77 closure-excess collective brake (COURSE_CLOSURE_BRAKE_COLLECTIVE)
# is deleted with its call site — under the unified vz-tracking law a
# sub-support cut is immediately restored by the tracker (a masked no-op)
# and was a fourth incoherent vz term in the F95 limit cycle.
# Governor trust gate (F33): expansion from a tiny far track is sub-pixel
# noise — post-credit, gate 1 (span 0.03-0.04, log_scale ~-2.9) "grew" at
# 0.9/s and pinned a +0.12 brake with aw_fwd -5 m/s^2 for the whole leg,
# reversing the drone into gate 0's structure.  Below this log_scale the
# expansion rate is untrustworthy and the governor stays out of the loop
# (far-field real closure at our speeds never exceeds the target anyway:
# 4 m/s at 10 m is 0.4/s with scale already >= ~0.08).
CLOSURE_MIN_LOG_SCALE = -2.6
# Fragment advance gate (F43, 20260729T202844Z-visual-course-ee8fd1e5): the
# gate-1 leg advanced at full +0.08 on a LONE span-(0.04,0.10) fragment
# (log_scale ~-3.7), built fh 3-4 mps2, and parallax outran yaw authority
# into the gate-1 structure.  A small span is "whole gate far away" OR
# "fragment of a gate that is NEAR" — absolute log_scale cannot tell them
# apart.  VisualTrack exposes no union marker (vq2_tracked_fragment_union
# lives on the internal VisualDetection only), so the bound is span-based:
# F43's lone fragments measured apparent_scale <= 0.125 (log <= -2.08),
# the fused unions 0.255-0.286, and the proved gate-0 spawn ~0.17 (log
# ~-1.76) — log -2.0 separates them with margin on both sides.  Below it
# the leg creeps while centering instead of advancing at full pitch.
FRAGMENT_ADVANCE_MIN_LOG_SCALE = -2.0  # lone-fragment span bound (log apparent)
FRAGMENT_CREEP_PITCH_RAD = 0.03  # creep OFFSET from spawn while centering a lone fragment
# Brake ceiling band (F33/F34): while the governor brakes, the collective
# is confined to support +/- this band.  F33's hard pin AT support removed
# all vertical centering authority and F34 crossed at 1.07 m into the
# bottom bar (impulse 3.74) — approach geometry raises the gate's
# elevation the whole way in, so the qualified PD must retain a small
# climb budget; F32's climb-into-frame came from the fh floor and the
# high-gate bias, which this band still caps.
BRAKE_CEILING_BAND = 0.04
PRE_CROSS_BRAKE_PITCH_RAD = -0.15  # nose-up brake OFFSET from spawn (F49).
# TRUE nose-up brake attitude under the verified F38 convention; as an
# offset from the -0.31 spawn attitude the effective target is ~-0.46.
# (Pre-F38 this was +0.15 absolute — a powered DIVE into the gate; F31's
# "5x too weak brake" was the sign error, not the magnitude.)  F46: -0.15
# absolute (~1.5 m/s^2 + drag) could not kill the approach speed inside the
# last 1.2 s to the gate-1 plane — the drone crossed the threshold still
# fast and 0.37 norm off-center.  F49 restores that deceleration authority
# as an offset: with level flight at -0.31, the effective -0.46 roughly
# doubles it so a misaligned approach actually stops.
PRE_CROSS_BRAKE_SLEW_RAD_S = 1.0  # fast slew while the governor brakes
# F80: course-leg brake authority (gate_index >= 1).  F79
# (20260730T093737Z-visual-course-14d98732) held the -0.46 TRUE brake for
# the whole gate-1 leg (misalignment demand saturated from the first tick)
# and still closed log-scale -1.6 -> -0.57 in 2.4 s: the leg inherits the
# gate-0 crossing's ~5 m/s, and ~1.5 m/s^2 of brake bleeds <3 m/s before
# the plane, so the drone passed ~2 m right of the aperture with yaw at
# the cap (parallax outran yaw authority again).  Course legs start hot by
# construction (COMMIT drives through the previous plane), so they get
# twice the brake offset: -0.30 from spawn (effective -0.61, ~3 m/s^2 +
# drag), enough to stop inside the observed far window and re-center at
# hover as the F54 hold intends.  The F51/F65 vision-custody relax still
# outranks the brake near the plane; gate 0 keeps the proved -0.15.
COURSE_PRE_CROSS_BRAKE_PITCH_RAD = -0.30
# F51 near-plane brake self-blinding guard (F50 t=15 episode): the brake
# attitude (rpy_p ~-0.45, ~0.14 rad nose-up from spawn) pitches the camera
# up, so near the plane the gate slides DOWN the frame — measured ey
# reached +0.93 and bottom-censored out of view while the brake held for
# 1.5 s.  Measurement compensation (F50) cannot extend the physical FOV.
# F94 replaces the binary relax/resume latch with a continuous
# custody-preserving floor (see the clamp at the pitch law): the pitch
# target never goes nose-up past the attitude that places the compensated
# ey exactly ON the bound below, so the approach always holds the maximum
# custody-compatible brake.  Vision custody bounds deceleration; it no
# longer abandons it in one step.
BRAKE_RELAX_EY_NORM = 0.55  # far-range compensated ey custody bound
# F65 near-plane extension (20260730T021149Z-visual-course-08f41050): AT
# the plane the F51 guard never fired — F64's gate sat at ey +0.33..+0.43
# (below the 0.55 fresh bound), censorship then froze measurement
# freshness, and the -0.46 brake attitude pitched the gate out of the FOV
# for the remaining ~7 s (blind wander into the floor/structure).  Inside
# the commit proximity regime the floor runs on the derotated HYPOTHESIS
# (the F52 best-evidence rationale), with a lower bound.
NEAR_BRAKE_RELAX_EY_NORM = 0.30  # commit-regime custody bound
# F71 (20260730T060005Z-visual-course-f05911e4): on the gate-1 leg the 0.30
# engage bound sat a hair above the achieved hypothesis ey (0.22-0.30 for
# the whole final second), so the relax never fired; the -0.41..-0.43 brake
# attitude walked the gate down the FOV into engulfing at the plane and a
# one-tick newborn corner splinter was adopted as the aim (collision id
# 1001, impulse 5.86).  Course legs (gate_index >= 1) get a tighter bound.
NEAR_BRAKE_RELAX_COURSE_EY_NORM = 0.18  # course-leg custody bound
# Course-heading anchor (F31): after losing the gate-1 track the drone
# search-swept and edge-chased its heading +2.63 rad off the course
# bearing, then flew sideways/backwards at ~0.65g drag into structure it
# never squarely saw.  The yaw channel is clamped so cumulative excursion
# from the leg anchor (initial yaw, re-anchored on every authoritative
# promotion) can never exceed this; only return steering is allowed at
# the cap.
# 0.9 -> 1.5 (F39b, 4a4b7792): the gate-1 handoff sits 0.45-0.6 rad off the
# leg start; the pursuit plus the SEARCH recovery legitimately needed
# ~1.0-1.3 rad, and the 0.9 cap PARKED the blind recovery (yaw frozen at
# 1.33 for 2.4 s until a soft gate-1 graze).  1.5 still blocks F31's 2.63
# rad wander with margin.
COURSE_HEADING_ANCHOR_CAP_RAD = 1.5
# F92: the pre-gate-1 altitude-floor latch (ALT_FLOOR_*) is DELETED — see
# the removal note in command().  alt_est stays for the vz-center trim's
# blind anti-sink, still clamped below at ALT_EST_MIN_M (F13).
ALT_EST_MIN_M = -2.0  # biased-integrator clamp on the altitude estimate
# F14 inflow-regime gate (agent-10, verified with the actuator/estimator
# tick fields): identical delivered rotor output 0.34 gives accz -13.0 in
# the slow regime (real climb, t=2.59) and -8.0 in the fast regime (t=8.5)
# — vz_est is invalidated by REGIME, a smooth fh-proportional DC deficit
# (~0.9*fh - 0.5), NOT vibration (accz std 0.19, the cleanest of the
# flight) and NOT attitude (kp=0, gyro drift <= 0.2 rad).  Trusted fh <
# ~2.0, biased fh 6.5-7.5; F49's hard brake reads 3.0-3.4 and must NOT
# trip the gate, so the trigger sits at 5.0 (see below).  While untrusted
# the stage freezes vz_est (the
# leak relaxes it toward 0; the biased a_up is never integrated), holds
# alt_est, falls back to the camera-qualified vertical PD (support +
# margin when unqualified — bare support historically sinks for real at
# -0.8...-1.9), and suppresses every vz-based governor floor/cap so the
# descent feedforward cannot fire from the frozen estimate.  This breaks
# the F14 self-locking loop: governor pinned 0.34 on the phantom sink, the
# floor flew biased-"level".
# F50 (flight 20260729T222920Z-visual-course-3a8ed087): the F49 TRUE brake
# reads fh 3.0-3.4 — a hard nose-up brake IS horizontal specific force — so
# the 3.0 trigger tripped on the brake itself, latched _fh_untrusted, and
# floored the collective at support + 0.05 for the whole gate-1 leg (pinned
# ~0.31, ceiling height, truss graze).  The F14 pathological regime measured
# 6.5-7.5; 5.0 separates braking from the biased regime with margin on both
# sides.
FH_UNTRUSTED_TRIGGER_MPS2 = 5.0  # biased regime above this horizontal force
FH_TRUSTED_RELEASE_MPS2 = 2.0  # hysteresis release below this
FH_UNTRUSTED_SUSTAIN_S = 0.3  # transients shorter than this never latch
# 0.02 -> 0.05 (flight 20260729T151236Z-visual-course-99e093fa): the fh gate
# froze vz/alt at t=2.83 post-credit and the unqualified hold at support+0.02
# sank for real ~1 m in 1.5 s into gentle terrain contacts — F14 measured the
# biased-regime deficit at ~0.05 collective, so the margin must cover all of
# it, not part of it.
FH_UNTRUSTED_VERTICAL_MARGIN = 0.05  # unqualified hold: support + margin
# High-gate climb bias (post-credit pursuit redesign, agent-10 F26/F27/L13/
# L18 trace analysis): gate 1 is handed off HIGH (ey ~ -0.69, 20% already
# top-clipped at credit), and a censored/unqualified y-axis decays the
# collective to support + margin — a real sink in the biased regime — so the
# gate migrates UP, clips harder, and the track dies ~1.5 s post-credit in
# every flight.  When the tracked gate is high, the unqualified hold must
# CLIMB toward it, not hover: support + 0.065 ~= the 0.34 thrust clamp.
# 0.065 -> 0.12 (F35, d25f23fe): the +0.065 "climb" margin only matched
# fast-regime hover (~0.32) — alt pinned at 1.59 m for the whole gate-1 leg
# while the gate sat at ey -0.9, and the drone flew LEVEL into the gate's
# lower structure.  A top-clipped gate is a one-sided measurement: the ONLY
# safe direction is up.  support + 0.12 ~= 0.40 is a real climb that also
# un-clips the y-axis so the qualified PD can take over.
HIGH_GATE_Y_NORM = -0.30  # hypothesis y below this counts as "gate is high"
HIGH_GATE_CLIMB_MARGIN = 0.12  # unqualified hold margin while the gate is high
# F50 pitch-attitude compensation of the vertical error (flight
# 20260729T222920Z-visual-course-3a8ed087): the vertical servo read the
# aim's image-y with NO attitude compensation, so the F49 nose-up brake
# (rpy_p -0.46, ~0.15 rad up from the -0.31 spawn attitude) tilted the
# camera up and the world read LOW in frame — ey drifted +0.06 -> +0.68
# while the servo "centered" a gate that was really ~1.5-2 m below, and
# the leg held ceiling height into a truss.  (F32/F34/F36 saw the same
# contamination with the opposite sign: nose-DOWN dives read gates HIGH at
# ey -0.5..-0.7.)  With image-down-positive ey, a nose-up attitude (rpy_p
# below spawn_pitch_rad) shifts the world DOWN in frame, so the effective
# error is ey_true = ey_measured - (spawn_pitch_rad - rpy_p) * this gain;
# it is zero at the spawn attitude.
VERTICAL_PITCH_COMP_NORM_PER_RAD = 1.6  # image-norm vertical shift per rad
# SEARCH vertical band: bounds the COMMIT/TRACK ey collective servo.  The
# F50 SEARCH memory-descent that shared this band is REMOVED (F75): it
# turned F74's gate-1 miss into a blind sink to the alt-est floor — a
# no-track SEARCH now latches altitude support and lets the sweep do the
# re-acquisition.
SEARCH_VERTICAL_MEMORY_BAND = 0.05  # collective band around support
SEARCH_COVARIANCE_STD_NORM = 0.35  # position std that forces SEARCH
# F76 (20260730T074122Z-visual-course-3a505ef5): after the one-zero send
# the pending-credit SEARCH ran the generic yaw sweep — +0.15 (right)
# while the retained gate-1 successor sat LEFT (ex -0.43); every
# detection died 0.4 s later and the leg pinned blind at the yaw cap
# into the gate-1 structure (collision id 1002).  While authoritative
# credit is still in flight the SEARCH never runs the generic sweep, and
# note_race re-acquires the retained successor evidence the moment the
# increment lands.  F78 (20260730T082159Z-visual-course-7e18243d): the
# neutral heading hold also DELAYED the turn — gate 1 sat visible at
# x ~-0.51 the whole window — so a fresh/persistent/qualified successor
# bearing now steers a bounded recentering inside the window (no roll,
# no advance, no pre-credit adoption; absent/ambiguous evidence keeps
# the neutral hold).  Bounded: expiry without credit resumes the normal
# sweep (the crossing was not authoritative).
PENDING_CREDIT_HOLD_S = 1.00  # post-one-zero heading-hold window
# Real scan, not a wiggle (post-credit pursuit redesign): 0.12 rad/s with a
# 1.2 s reversal made +-8 deg legs that could never reach gate 1's typical
# ~26-35 deg handoff bearing, and the reversal actively undid turn progress
# (L18: 6.3 s of +-0.12 sweep achieving nothing).  The 0.8 rad excursion
# bound gives ~46 deg legs, first leg toward the last bearing.  2026-07-30:
# the rate sits AT the 0.15 production yaw command cap — the fallback
# incremental sweep is controller output and must not exceed it.
SEARCH_YAW_RATE_RAD_S = 0.15  # bounded sweep rate (at the yaw command cap)
SEARCH_SWEEP_PERIOD_S = 6.00  # reversal backstop; the excursion bound fires first
SEARCH_MAX_EXCURSION_RAD = 0.80  # bounded sweep excursion before reversal
# F40 (20260729T193134Z-visual-course-63ed6342): the SEARCH sweep integrated
# the COMMANDED yaw rate, not the actual heading — at the course-anchor cap
# (anchor 0.417 + 1.5) the sweep parked at yaw 1.94 rad, 111 deg off course,
# for ~7 blind seconds into gate 1, never scanning the cone where the gate
# actually was.  The sweep now sweeps an absolute desired heading around
# the leg anchor so a parked airframe still scans the band.
SEARCH_SWEEP_RATE_RAD_S = 0.5  # absolute-heading sweep rate around the anchor
SEARCH_SWEEP_GAIN = 2.0  # heading-error gain to the bounded yaw rate

SUCCESSOR_BLEND_MAX = 0.50  # continuous lookahead ceiling
BLEND_FAR_LOG_SCALE = -1.6  # below this the successor gets no blend
BLEND_NEAR_LOG_SCALE = -0.9  # at this closure the blend ceiling applies
PROMOTE_MAX_STD_NORM = 0.60  # cached-successor credibility at promotion
# 0.30 -> 0.60 (F35, d25f23fe): promotion dropped the fresh gate-1 successor
# (std 0.14-0.34, age 0) into SEARCH, and the reacquisition churn that
# followed flapped TRACK/SEARCH 7 times before impact.  The successor is the
# best evidence of the next gate at the one moment the course geometry is
# known — adopt it unless it is truly stale.
PROMOTE_MAX_AGE_S = 0.50  # cached-successor freshness at promotion
# F42 (20260729T201743Z-visual-course-1e24b6d2): confidence provably cannot
# separate the real gate from detector debris — a bottom-left splinter
# out-confidenced the real gate-1 halves (0.62-0.71 vs 0.42-0.54) and was
# adopted at promotion.  PERSISTENCE can: debris is newborn every frame
# while the real gate halves stayed associated for seconds.  Successor
# ranking, promotion credibility, and re-acquisition all prefer track age.
SUCCESSOR_MIN_AGE_S = 0.5  # persistence required of a promoted successor
REACQUIRE_MIN_AGE_S = 0.3  # persistence preferred at SEARCH re-acquisition
# F49 newborn suspicious-geometry adoption gate (terminal F48 failure): the
# gate-1 promotion adopted a NEWBORN top-censored extreme-aspect ceiling
# truss (span 0.50 x 0.23, aspect ~2.17) over the persistent real gate.
# Persistence alone could not reject it inside the 0.3/0.5 s age windows,
# so a track whose bbox geometry is impossible for a gate (extreme aspect,
# or a wide top-censored slab) may not be adopted until it has persisted
# past the re-acquisition age window.
SUSPICIOUS_ASPECT_MAX = 2.0  # width/height above this is not gate geometry
SUSPICIOUS_ASPECT_MIN = 0.5  # height/width above this is not gate geometry
SUSPICIOUS_TOP_CENSORED_ASPECT = 1.0  # wide slab censored at the frame top

COLLECTIVE_DECAY_TAU_S = 0.25  # smooth decay toward support on vertical loss
# Qualified vertical measurement horizon.  The bound applies to the last
# ACCEPTED (non-engulfing, non-censored) y measurement: flight
# 20260729T104947Z-visual-course-bc8c6003 flew 5.4 s on a phantom vertical
# state after the crossing, so hypothesis creation must never claim a fresh
# y from a censored detection, and the retained rate is zeroed the moment
# qualification is lost (see command()).
VERTICAL_QUALIFY_MAX_AGE_S = 0.30

TARGET_SLEW_RAD_S = 0.30  # single transparent target slew rate
CLIPPED_STEERING_FRACTION = 0.5  # clipping saturates corrective steering

APERTURE_MIN_CONFIDENCE = 0.20  # fitted inner-aperture acceptance floor
OUTER_MEAS_STD_NORM = 0.06  # outer bbox center measurement std

# Degenerate engulfing-detection rejection.  Flights 95644bf5 and 4dbe4b8c
# both ended with a near-full-frame bbox (640x360 / 640x347, every edge
# clipped) accepted as a gate measurement 46-48 ms before impact.  A box
# covering most of the frame is the gate engulfing the camera at the plane,
# not a usable measurement; it is treated as no measurement (PREDICT
# semantics).  The observed all-edges-clipped boxes are caught by the span
# rule (full-frame width).
ENGULFING_BBOX_SPAN_FRACTION = 0.9  # bbox width/height vs the whole frame
ENGULFING_BBOX_AREA_FRACTION = 0.7  # bbox area vs the whole frame
# An engulfing box is useless for scale/vertical servo but its center still
# says "gate centered, very close": flight bc8c6003 cycled TRACK<->SEARCH
# five times on phantom yaw sweeps while flying through the gate plane.
# Fresh engulfing evidence anchors the horizontal bearing and blocks the
# PREDICT->SEARCH transition; it never updates any filter axis.
ENGULFING_ANCHOR_MAX_AGE_S = 0.50  # existence-evidence freshness horizon
SCALE_MEAS_STD = 0.10  # log-scale measurement std
MIN_MEAS_CONFIDENCE = 0.05  # confidence noise floor divisor

PROCESS_VAR_POS = 0.05  # per-second position process variance
PROCESS_VAR_RATE = 0.5  # per-second rate random-walk variance
LATENCY_VAR_NORM = 0.0004  # per-frame unknown-capture-latency inflation
CENSOR_INFLATE_VAR_NORM = 0.01  # censored-axis per-frame inflation
CLIPPED_INFLATE_VAR_NORM = 0.004  # clipping uncertainty inflation
INITIAL_POS_VAR_NORM = 0.01  # fresh measured hypothesis position variance
INITIAL_RATE_VAR = 0.25
SYNTHETIC_POS_VAR_NORM = 0.16  # StartContext-only fallback hypothesis
# F57 (20260730T003044Z-visual-course-74abd688): the de-rotation focal was
# inconsistent with the SAME camera's measured geometry already in this
# file (VERTICAL_PITCH_COMP_NORM_PER_RAD = 1.6) — at 1.0 every predicted
# bearing under-rotated by 37.5%, which is why the frozen hypothesis
# lagged the true gate bearing in F52 (frozen ex -0.156 vs true -0.48) and
# why stale-bearing steering under-corrects.  The measured 1.6 now drives
# every derotation consumer (PREDICT, coast, F52-A hold, COMMIT steering).
ROTATION_COMP_FOCAL_NORM = 1.6  # normalized focal length for de-rotation
ROTATION_COMP_UNCERTAINTY = 0.25  # fraction of comp drift added as variance
# Timestamp sentinel for "this axis has never had an accepted measurement"
# (censored creation detection); any horizon check against it fails.
NEVER_MEASURED_S = -1e9

CONTROL_PERIOD_S = 0.02  # 50 Hz pacing (runner-owned invariant)

# Controller identity reported in result.json / recorder evidence for the
# visual-course stage.  The retired VisualNavigationConfig evidence still in
# the runner reports legacy servo/lifecycle parameters this stage never
# reads; the clean stage binds its real named constants instead.
CLEAN_COURSE_CONTROLLER_FAMILY = "aigp-vq2-clean-course/1"
CLEAN_COURSE_CONFIG_SCHEMA = "aigp-vq2-clean-course-config/1"


class CleanCourseState(str, Enum):
    """The exactly five runtime states of the clean course stage."""

    TRACK = "track"
    PREDICT = "predict"
    COAST_FOR_CREDIT = "coast_for_credit"
    COMMIT = "commit"
    SEARCH = "search"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


class _AxisFilter:
    """One 2-state (position, rate) variable-dt Kalman filter."""

    __slots__ = ("p", "v", "pp", "pv", "vv")

    def __init__(self, p: float, v: float, var_p: float, var_v: float) -> None:
        self.p = float(p)
        self.v = float(v)
        self.pp = float(var_p)
        self.pv = 0.0
        self.vv = float(var_v)

    def predict(
        self,
        dt: float,
        *,
        drift: float = 0.0,
        process_var_pos: float = PROCESS_VAR_POS,
        process_var_rate: float = PROCESS_VAR_RATE,
    ) -> None:
        """Constant-velocity prediction with an optional known drift."""

        self.p += self.v * dt + drift
        self.pp += 2.0 * dt * self.pv + dt * dt * self.vv + process_var_pos * dt
        self.pv += dt * self.vv
        self.vv += process_var_rate * dt

    def update(self, z: float, r: float) -> None:
        """Position measurement update with noise variance ``r``."""

        innovation = z - self.p
        s = self.pp + max(1e-9, r)
        k_p = self.pp / s
        k_v = self.pv / s
        self.p += k_p * innovation
        self.v += k_v * innovation
        self.pp -= k_p * self.pp
        pv_old = self.pv
        self.pv -= k_p * pv_old
        self.vv -= k_v * pv_old

    def inflate(self, var_add: float) -> None:
        self.pp += var_add

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.pp))


class _Hypothesis:
    """Retained current/successor target hypothesis with its small filter."""

    __slots__ = (
        "track_id",
        "x_axis",
        "y_axis",
        "scale_axis",
        "confidence",
        "outer_log_scale",
        "outer_half_span_x",
        "clipped",
        "created_s",
        "last_measurement_s",
        "last_x_measurement_s",
        "last_y_measurement_s",
        "raw_x",
        "raw_y",
        "aperture_half_x",
        "aperture_half_y",
    )

    def __init__(
        self,
        *,
        track_id: Optional[str],
        x: float,
        y: float,
        log_scale: float,
        confidence: float,
        pos_var: float,
        now_s: float,
    ) -> None:
        self.track_id = track_id
        self.x_axis = _AxisFilter(x, 0.0, pos_var, INITIAL_RATE_VAR)
        self.y_axis = _AxisFilter(y, 0.0, pos_var, INITIAL_RATE_VAR)
        self.scale_axis = _AxisFilter(log_scale, 0.0, pos_var, INITIAL_RATE_VAR)
        self.confidence = _clamp01(confidence)
        self.outer_log_scale = float(log_scale)
        # F56: outer bbox x half-width in norm — the COMMIT corridor bound
        # is measured in gate units.  Square-box proxy from the area scale
        # here; the real bbox half-width overwrites it on every uncensored
        # x measurement (see _update_hypothesis).
        self.outer_half_span_x = 0.5 * math.exp(float(log_scale))
        self.clipped = False
        self.created_s = float(now_s)
        self.last_measurement_s = float(now_s)
        self.last_x_measurement_s = float(now_s)
        self.last_y_measurement_s = float(now_s)
        # Last uncensored RAW measurement per axis (the commit-entry budget
        # evaluates the worst of raw vs filtered) and the CURRENT frame's
        # usable inner-aperture half-extents (None when the fit is not
        # passage-usable — itself the "about to go blind" signal).
        self.raw_x = float(x)
        self.raw_y = float(y)
        self.aperture_half_x: Optional[float] = None
        self.aperture_half_y: Optional[float] = None

    @property
    def x(self) -> float:
        return self.x_axis.p

    @property
    def y(self) -> float:
        return self.y_axis.p

    @property
    def vx(self) -> float:
        return self.x_axis.v

    @property
    def vy(self) -> float:
        return self.y_axis.v

    @property
    def log_scale(self) -> float:
        return self.scale_axis.p

    @property
    def expansion_rate(self) -> float:
        return self.scale_axis.v

    @property
    def position_std(self) -> float:
        return math.hypot(self.x_axis.std, self.y_axis.std)


@dataclass(frozen=True)
class NavigationOutput:
    """Exactly what navigation may ask for on one tick."""

    target_roll_rad: float
    target_pitch_rad: float
    yaw_rate_rad_s: float
    thrust: float
    state: CleanCourseState
    gate_index: int
    advance_factor: float = 0.0
    successor_blend: float = 0.0
    vertical_qualified: bool = False
    current_track_id: Optional[str] = None
    successor_track_id: Optional[str] = None


@dataclass(frozen=True)
class CleanCourseConfig:
    """Tunable bounds for :class:`CleanCourseController` (test-friendly)."""

    vertical_feedback_sign: float = VERTICAL_FEEDBACK_SIGN
    support_collective: float = SUPPORT_COLLECTIVE
    vertical_error_gain: float = VERTICAL_ERROR_GAIN
    vertical_rate_gain: float = VERTICAL_RATE_GAIN
    vertical_max_abs_error_norm: float = VERTICAL_MAX_ABS_ERROR_NORM
    vertical_max_abs_rate_norm_s: float = VERTICAL_MAX_ABS_RATE_NORM_S
    vertical_arrest_vz_per_norm: float = VERTICAL_ARREST_VZ_PER_NORM
    vertical_arrest_collective_gain: float = VERTICAL_ARREST_COLLECTIVE_GAIN
    course_vz_des_per_norm: float = COURSE_VZ_DES_PER_NORM
    course_vz_des_max_m_s: float = COURSE_VZ_DES_MAX_M_S
    course_vz_track_gain: float = COURSE_VZ_TRACK_GAIN
    course_vz_des_commit_m_s: float = COURSE_VZ_DES_COMMIT_M_S
    course_vz_des_ramp_start_log_scale: float = COURSE_VZ_DES_RAMP_START_LOG_SCALE
    course_vz_trim_fast_leak_s: float = COURSE_VZ_TRIM_FAST_LEAK_S
    min_thrust: float = MIN_COURSE_THRUST
    max_thrust: float = MAX_COURSE_THRUST
    launch_boost_thrust: float = LAUNCH_BOOST_THRUST
    launch_boost_duration_s: float = LAUNCH_BOOST_DURATION_S
    gate0_climb_vertical_offset_norm: float = GATE0_CLIMB_VERTICAL_OFFSET_NORM
    gate0_climb_reference_log_scale: float = GATE0_CLIMB_REFERENCE_LOG_SCALE
    roll_error_sign: float = ROLL_ERROR_SIGN
    roll_error_gain: float = ROLL_ERROR_GAIN
    max_target_roll_rad: float = MAX_TARGET_ROLL_RAD
    roll_pursuit_slew_rad_s: float = ROLL_PURSUIT_SLEW_RAD_S
    roll_pursuit_fast_ex_norm: float = ROLL_PURSUIT_FAST_EX_NORM
    yaw_error_sign: float = YAW_ERROR_SIGN
    yaw_error_gain: float = YAW_ERROR_GAIN
    max_yaw_rate_rad_s: float = MAX_COURSE_YAW_RATE_RAD_S
    # F49: measured build-3385 level-flight attitude; every *_pitch_rad
    # below is an OFFSET from this base (effective = spawn + offset).
    spawn_pitch_rad: float = SPAWN_PITCH_RAD_DEFAULT
    advance_pitch_rad: float = ADVANCE_PITCH_RAD
    brake_pitch_rad: float = BRAKE_PITCH_RAD
    angular_full_brake_norm: float = ANGULAR_FULL_BRAKE_NORM
    expansion_brake_free_s: float = EXPANSION_BRAKE_FREE_S
    expansion_brake_span_s: float = EXPANSION_BRAKE_SPAN_S
    near_free_log_scale: float = NEAR_FREE_LOG_SCALE
    near_brake_log_scale: float = NEAR_BRAKE_LOG_SCALE
    crossing_min_log_scale: float = CROSSING_MIN_LOG_SCALE
    crossing_max_abs_ex_norm: float = CROSSING_MAX_ABS_EX_NORM
    crossing_max_abs_ey_norm: float = CROSSING_MAX_ABS_EY_NORM
    commit_sustain_s: float = COMMIT_SUSTAIN_S
    commit_meas_max_age_s: float = COMMIT_MEAS_MAX_AGE_S
    commit_timeout_s: float = COMMIT_TIMEOUT_S
    commit_min_log_scale: float = COMMIT_MIN_LOG_SCALE
    commit_advance_pitch_rad: float = COMMIT_ADVANCE_PITCH_RAD
    commit_entry_meas_max_age_s: float = COMMIT_ENTRY_MEAS_MAX_AGE_S
    commit_blackout_s: float = COMMIT_BLACKOUT_S
    commit_entry_aperture_margin_frac: float = COMMIT_ENTRY_APERTURE_MARGIN_FRAC
    commit_entry_max_expansion_rate_s: float = COMMIT_ENTRY_MAX_EXPANSION_RATE_S
    commit_entry_max_vz_m_s: float = COMMIT_ENTRY_MAX_VZ_M_S
    pitch_target_min_rad: float = PITCH_TARGET_MIN_RAD
    ex_trim_gain: float = EX_TRIM_GAIN
    ex_trim_max_norm: float = EX_TRIM_MAX_NORM
    ex_trim_active_ex_norm: float = EX_TRIM_ACTIVE_EX_NORM
    vz_center_trim_gain: float = VZ_CENTER_TRIM_GAIN
    vz_center_trim_max: float = VZ_CENTER_TRIM_MAX
    vz_center_trim_active_ey_norm: float = VZ_CENTER_TRIM_ACTIVE_EY_NORM
    vz_center_trim_sink_m_s: float = VZ_CENTER_TRIM_SINK_M_S
    vz_center_trim_leak_s: float = VZ_CENTER_TRIM_LEAK_S
    near_plane_steer_gain_mult: float = NEAR_PLANE_STEER_GAIN_MULT
    predict_frame_gap_s: float = PREDICT_FRAME_GAP_S
    predict_max_gap_s: float = PREDICT_MAX_GAP_S
    x_steer_max_age_s: float = X_STEER_MAX_AGE_S
    closure_target_rate_s: float = CLOSURE_TARGET_RATE_S
    closure_full_brake_rate_s: float = CLOSURE_FULL_BRAKE_RATE_S
    closure_min_log_scale: float = CLOSURE_MIN_LOG_SCALE
    fragment_advance_min_log_scale: float = FRAGMENT_ADVANCE_MIN_LOG_SCALE
    fragment_creep_pitch_rad: float = FRAGMENT_CREEP_PITCH_RAD
    pre_cross_brake_pitch_rad: float = PRE_CROSS_BRAKE_PITCH_RAD
    course_pre_cross_brake_pitch_rad: float = COURSE_PRE_CROSS_BRAKE_PITCH_RAD
    pre_cross_brake_slew_rad_s: float = PRE_CROSS_BRAKE_SLEW_RAD_S
    brake_relax_ey_norm: float = BRAKE_RELAX_EY_NORM
    near_brake_relax_ey_norm: float = NEAR_BRAKE_RELAX_EY_NORM
    near_brake_relax_course_ey_norm: float = NEAR_BRAKE_RELAX_COURSE_EY_NORM
    brake_ceiling_band: float = BRAKE_CEILING_BAND
    course_heading_anchor_cap_rad: float = COURSE_HEADING_ANCHOR_CAP_RAD
    alt_est_min_m: float = ALT_EST_MIN_M
    fh_untrusted_trigger_mps2: float = FH_UNTRUSTED_TRIGGER_MPS2
    fh_trusted_release_mps2: float = FH_TRUSTED_RELEASE_MPS2
    fh_untrusted_sustain_s: float = FH_UNTRUSTED_SUSTAIN_S
    fh_untrusted_vertical_margin: float = FH_UNTRUSTED_VERTICAL_MARGIN
    high_gate_y_norm: float = HIGH_GATE_Y_NORM
    high_gate_climb_margin: float = HIGH_GATE_CLIMB_MARGIN
    vertical_pitch_comp_norm_per_rad: float = VERTICAL_PITCH_COMP_NORM_PER_RAD
    search_vertical_memory_band: float = SEARCH_VERTICAL_MEMORY_BAND
    search_covariance_std_norm: float = SEARCH_COVARIANCE_STD_NORM
    search_yaw_rate_rad_s: float = SEARCH_YAW_RATE_RAD_S
    search_sweep_period_s: float = SEARCH_SWEEP_PERIOD_S
    search_max_excursion_rad: float = SEARCH_MAX_EXCURSION_RAD
    search_sweep_rate_rad_s: float = SEARCH_SWEEP_RATE_RAD_S
    search_sweep_gain: float = SEARCH_SWEEP_GAIN
    pending_credit_hold_s: float = PENDING_CREDIT_HOLD_S
    successor_blend_max: float = SUCCESSOR_BLEND_MAX
    blend_far_log_scale: float = BLEND_FAR_LOG_SCALE
    blend_near_log_scale: float = BLEND_NEAR_LOG_SCALE
    promote_max_std_norm: float = PROMOTE_MAX_STD_NORM
    promote_max_age_s: float = PROMOTE_MAX_AGE_S
    successor_min_age_s: float = SUCCESSOR_MIN_AGE_S
    collective_decay_tau_s: float = COLLECTIVE_DECAY_TAU_S
    vertical_qualify_max_age_s: float = VERTICAL_QUALIFY_MAX_AGE_S
    target_slew_rad_s: float = TARGET_SLEW_RAD_S
    clipped_steering_fraction: float = CLIPPED_STEERING_FRACTION
    control_period_s: float = CONTROL_PERIOD_S


def clean_course_controller_evidence(
    *, candidate_commit: Optional[str]
) -> Dict[str, Any]:
    """Bind the clean course controller identity to its exact source commit.

    Same envelope shape as the runner's ``controller_config_evidence`` so it
    can be recorded verbatim as the visual-course ``controller`` evidence.
    ``effective_parameters`` are the real named constants of the default
    :class:`CleanCourseConfig`, not the retired visual servo/lifecycle set.
    """

    if candidate_commit is not None and (
        type(candidate_commit) is not str
        or len(candidate_commit) != 40
        or any(character not in "0123456789abcdef" for character in candidate_commit)
    ):
        raise ValueError("candidate_commit must be 40 lowercase hexadecimal characters")
    parameters = {
        field.name: getattr(CleanCourseConfig(), field.name)
        for field in fields(CleanCourseConfig)
    }
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return {
        "git_commit": candidate_commit,
        "config_schema": CLEAN_COURSE_CONFIG_SCHEMA,
        "controller_family": CLEAN_COURSE_CONTROLLER_FAMILY,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "effective_parameters": parameters,
    }


class CleanCourseController:
    """Four-state selector/estimator/control-law owner for one course run."""

    def __init__(self, config: Optional[CleanCourseConfig] = None) -> None:
        self.config = config or CleanCourseConfig()
        self.state = CleanCourseState.SEARCH
        self.gate_index = 0
        self.max_gate_index = 0
        self.transitions: List[Tuple[int, int]] = []
        self.current: Optional[_Hypothesis] = None
        self.successor: Optional[_Hypothesis] = None
        self.last_reliable_bearing: Tuple[float, float] = (0.0, 0.0)
        # F50: False until any real bearing evidence (track, engulfing
        # anchor, successor, cache, or the initialize fallback) is recorded;
        # the SEARCH vertical memory servo holds support without it.
        self._bearing_memory_valid = False
        self.successor_bearing_cache: Dict[int, Tuple[float, float]] = {}
        self._track_first_seen_s: Dict[str, float] = {}
        self._track_last_seen_s: Dict[str, float] = {}
        self._course_start_s: Optional[float] = None
        self._last_observe_s: Optional[float] = None
        self._last_command_s: Optional[float] = None
        self._collective: Optional[float] = None
        self._prev_target_roll = 0.0
        self._prev_target_pitch = (
            self.config.spawn_pitch_rad + self.config.brake_pitch_rad
        )
        self._coast_zero_sent = False
        self._coast_race_boot_ms: Optional[int] = None
        self._last_race_boot_ms: Optional[int] = None
        # F76: bounded post-one-zero heading-hold window (see
        # PENDING_CREDIT_HOLD_S); set at the coast exit, cleared on any
        # authoritative promotion.
        self._pending_credit_until_s: Optional[float] = None
        self._search_direction = 1.0
        self._search_elapsed_s = 0.0
        self._search_excursion_rad = 0.0
        # F49: SEARCH sweep base heading — the yaw measured at search entry,
        # not the leg anchor (see _search_yaw_heading).
        self._search_base_yaw_rad: Optional[float] = None
        # F53: near-plane COMMIT sustain timer and entry stamp (see the
        # COMMIT_* constant block).
        self._near_plane_since_s: Optional[float] = None
        self._commit_entry_s: Optional[float] = None
        # 2026-07-30 TRACK-phase lateral trim (see the EX_TRIM_* block): a
        # small bounded integral on raw ex that nulls the off-axis pursuit
        # orbit equilibrium before entry; reset on target/promotion change.
        self._ex_trim: float = 0.0
        # Vertical centering trim (see the VZ_CENTER_TRIM_* block): a
        # bounded one-sided integral that learns the support correction
        # needed to stop a centered-gate sink.  A vehicle/regime property,
        # so it survives target/promotion changes; reset on start only.
        self._vz_center_trim: float = 0.0
        # Underlying camera-frame identity of the last consumed update; a
        # republished frozen frame (same identity) is never fresh evidence.
        self._last_frame_identity: Optional[Tuple[Any, Any]] = None
        # Leaky world-vertical-rate estimate (m/s, up positive) fed by IMU
        # specific force; the stage starts on the pad, so zero is honest.
        self._vz_est_m_s = 0.0
        # Freshness of the last engulfing-box bearing/existence anchor (see
        # ENGULFING_ANCHOR_MAX_AGE_S); never a filter measurement.
        self._last_engulfing_anchor_s: Optional[float] = None
        # Camera-frame identity that set the anchor; a frozen frame
        # republished by the tracker (flight d52adcd4: 6.1 s stall) must not
        # keep refreshing it, or the anchor never expires and SEARCH is
        # suppressed for the whole blind descent.
        self._last_engulfing_anchor_identity: Optional[Tuple[Any, Any]] = None
        # IMU altitude estimate (m) integrated from the governor's vz_est,
        # seeded 0 at course start (takeoff pad reference) and clamped below
        # at alt_est_min_m (F13).  F92: the pre-gate-1 altitude-floor latch
        # it used to guard is deleted (see command()); the estimate stays
        # for the trace and the vz-center trim's blind anti-sink.
        self._alt_est_m = 0.0
        # Pre-crossing expansion brake latch for the tick trace; recomputed
        # every main-path tick (see the PRE_CROSS_BRAKE_* constant block).
        self._pre_cross_brake_active = False
        # F14 inflow-regime gate state (see the FH_* constant block):
        # sustained-high-fh timer, latched untrusted flag, and the last fh
        # seen (0.0 = trusted until the first host estimate arrives).
        self._fh_mps2 = 0.0
        self._fh_untrusted = False
        self._fh_above_since_s: Optional[float] = None
        # Course-heading anchor (F31): yaw at the leg start (lazily
        # captured on the first command tick with a live yaw measurement,
        # re-armed on every authoritative promotion).  Yaw commands that
        # would wind the heading past the anchor cap are clamped.
        self._course_anchor_yaw_rad: Optional[float] = None

    # -- initialization ----------------------------------------------------

    def initialize(
        self,
        update: Any,
        *,
        gate_index: int,
        fallback_center_norm: Tuple[float, float],
        fallback_apparent_scale: float,
        now_s: float,
    ) -> None:
        """Bind the initial current/successor hypotheses at the course start.

        Selection comes from the tracker update, the authoritative gate 0, and
        the ``StartContext`` initial gate center/area fallback.
        """

        self.gate_index = int(gate_index)
        self.max_gate_index = int(gate_index)
        self._course_start_s = float(now_s)
        self._ex_trim = 0.0
        self._vz_center_trim = 0.0
        identity = _frame_identity(update)
        if identity is not None:
            self._last_frame_identity = identity
        tracks = _visible_tracks(update)
        current_track = None
        if tracks:
            fx, fy = fallback_center_norm
            current_track = min(
                tracks,
                key=lambda track: math.hypot(
                    float(track.center_norm[0]) - fx,
                    float(track.center_norm[1]) - fy,
                ),
            )
        if current_track is not None:
            self.current = self._hypothesis_from_track(current_track, now_s)
            self.state = CleanCourseState.TRACK
            self._set_reliable_bearing(self.current.x, self.current.y)
        else:
            self.current = _Hypothesis(
                track_id=None,
                x=float(fallback_center_norm[0]),
                y=float(fallback_center_norm[1]),
                log_scale=math.log(max(1e-6, fallback_apparent_scale)),
                confidence=0.0,
                pos_var=SYNTHETIC_POS_VAR_NORM,
                now_s=now_s,
            )
            self._set_reliable_bearing(
                float(fallback_center_norm[0]),
                float(fallback_center_norm[1]),
            )
            self._enter_search(now_s)
        others = [
            track
            for track in tracks
            if current_track is None or track.track_id != current_track.track_id
        ]
        if others:
            best = max(others, key=lambda track: float(track.confidence))
            self.successor = self._hypothesis_from_track(best, now_s)

    # -- perception ---------------------------------------------------------

    def observe(
        self,
        update: Any,
        *,
        now_s: float,
        body_rates: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Consume one new tracker update (dropout = prediction only)."""

        cfg = self.config
        if self._last_observe_s is None:
            dt = cfg.control_period_s
        else:
            dt = _clamp(now_s - self._last_observe_s, 1e-3, 0.25)
        self._last_observe_s = float(now_s)
        tracks = _visible_tracks(update)
        # F42 track persistence bookkeeping: confidence provably cannot
        # separate the real gate from detector debris, but persistence can —
        # debris is newborn every frame, the real gate stays associated for
        # seconds.  Record first-seen per track id; prune ids unseen > 2 s.
        for track in tracks:
            track_id = str(track.track_id)
            if track_id not in self._track_first_seen_s:
                self._track_first_seen_s[track_id] = float(now_s)
            self._track_last_seen_s[track_id] = float(now_s)
        for track_id, last_seen in list(self._track_last_seen_s.items()):
            if now_s - last_seen > 2.0:
                del self._track_last_seen_s[track_id]
                self._track_first_seen_s.pop(track_id, None)
        # Frame freshness: the tracker republishes during a camera stall with
        # a new publication token but the SAME underlying camera-frame
        # identity.  Only a new camera frame is new evidence; an update whose
        # identity cannot be determined is conservatively treated as fresh.
        identity = _frame_identity(update)
        fresh = identity is None or identity != self._last_frame_identity
        if identity is not None:
            self._last_frame_identity = identity

        if self.current is not None:
            self._predict(self.current, dt, body_rates)
        if self.successor is not None:
            self._predict(self.successor, dt, body_rates)

        # Engulfing-box anchor (flight bc8c6003): useless for scale/vertical
        # servo, but its center still says "gate centered, very close", so it
        # refreshes horizontal bearing/existence evidence and (below) blocks
        # the PREDICT->SEARCH churn while flying through the gate plane.
        # Freshness-gated on the camera-frame identity (flight d52adcd4): a
        # republished frozen frame must not keep refreshing the anchor, or it
        # never expires during a camera stall.  An update whose identity
        # cannot be determined is conservatively treated as fresh, matching
        # the `fresh` rule above.
        anchor = _engulfing_anchor_track(update, self._current_track_id())
        if anchor is not None and (
            identity is None or identity != self._last_engulfing_anchor_identity
        ):
            self._last_engulfing_anchor_identity = identity
            self._last_engulfing_anchor_s = float(now_s)
            self._set_reliable_bearing(
                float(anchor.center_norm[0]),
                self.last_reliable_bearing[1],
            )

        # COAST_FOR_CREDIT: only the same track_id may resume tracking; the
        # bounded wait itself is governed by note_race/command.
        if self.state is CleanCourseState.COAST_FOR_CREDIT:
            resumed = self._find(tracks, self._current_track_id())
            if resumed is not None:
                self._update_hypothesis(self.current, resumed, now_s)
                self._exit_coast()
                self.state = CleanCourseState.TRACK
            self._refresh_successor(tracks, now_s)
            return

        match = self._find(tracks, self._current_track_id())

        # COMMIT (F53): the near-plane commit is an inertial crossing —
        # keep the hypothesis and successor fresh for the credit exit (the
        # vertical servo reads the live filter).  On a CREDIBLE CLOSE LOSS
        # (fresh frame, close hypothesis, no match) the armed crossing stops
        # all active blind driving and latches the exact-zero authoritative
        # credit wait (July-18 contract, restored 2026-07-30); only
        # note_race (credit) or the single-send command() exit may otherwise
        # leave the state.
        if self.state is CleanCourseState.COMMIT:
            if match is not None:
                self._update_hypothesis(self.current, match, now_s)
                self._set_reliable_bearing(self.current.x, self.current.y)
            elif (
                fresh
                and self.current is not None
                and self.current.outer_log_scale >= cfg.commit_min_log_scale
            ):
                self.state = CleanCourseState.COAST_FOR_CREDIT
                self._coast_zero_sent = False
                self._coast_race_boot_ms = self._last_race_boot_ms
            self._refresh_successor(tracks, now_s)
            return

        # F78b: the pending-credit window is an authority overlay, not just
        # a SEARCH-command posture — the same-track match path must not
        # flip back to TRACK (and start advancing) before the credit that
        # owns the leg, any more than the re-acquisition pick may (F78).
        pending_credit = (
            self._pending_credit_until_s is not None
            and now_s < self._pending_credit_until_s
        )
        if match is not None and not (
            self.state is CleanCourseState.SEARCH and pending_credit
        ):
            self._update_hypothesis(self.current, match, now_s)
            self.state = CleanCourseState.TRACK
        elif self.state is CleanCourseState.SEARCH or self.current is None:
            # F78 (20260730T082159Z-visual-course-7e18243d): while
            # authoritative credit is still in flight the pending-credit
            # SEARCH only RECENTERS on the successor bearing (see
            # command()) — it never ADOPTS it as the aim, which would
            # claim the next gate and start advancing before the credit
            # that owns the leg.  (F77 escaped only circumstantially: the
            # F74 near-plane newborn guard happened to cover the whole
            # window.)  On credit note_race promotes the retained
            # successor immediately; on expiry the normal pick resumes.
            adopted = (
                None
                if pending_credit
                else self._select_search_reacquisition(tracks, now_s)
            )
            if adopted is not None:
                self.current = self._hypothesis_from_track(adopted, now_s)
                self.state = CleanCourseState.TRACK
                # New target, new geometry: relearn the orbit trim.
                self._ex_trim = 0.0
        else:
            gap = now_s - self.current.last_measurement_s
            if (
                self.state is CleanCourseState.TRACK
                and fresh
                # Gate-0 legs only (2026-07-30 unification): gate-1+
                # crossings take ONE path — the aperture-budget COMMIT; a
                # gate-1+ close loss without an armed commit is NOT a
                # credible crossing and falls through to PREDICT/re-center.
                # Gate-0 never commits (its climb-bias path credits every
                # flight through this coast) and stays untouched.
                and self.gate_index == 0
                and self.current.outer_log_scale >= cfg.crossing_min_log_scale
                # F45 (20260729T210351Z-visual-course-b1f5e89f): the coast is
                # ballistic, so an OFF-CENTER close loss must not arm it —
                # the (-0.39,-0.28) last bearing slid out of frame
                # uncredited.  Credit is authoritative (it arrives by race
                # packet on a true pass); the coast only owns the wait of an
                # ALIGNED crossing.  The bearing must come from a fresh
                # MEASUREMENT, not a long prediction; an off-center or
                # stale-bearing loss falls through to PREDICT so the
                # derotated hypothesis carries the pursuit and TRACK can
                # resume on re-acquisition.
                and now_s - self.current.last_x_measurement_s
                <= CROSSING_MEAS_MAX_AGE_S
                and now_s - self.current.last_y_measurement_s
                <= CROSSING_MEAS_MAX_AGE_S
                and abs(self.current.x) <= cfg.crossing_max_abs_ex_norm
                and abs(self.current.y) <= cfg.crossing_max_abs_ey_norm
            ):
                # Credible close crossing lost the target on a FRESH frame:
                # latch the single bounded credit wait from the July-18
                # contract.  Flight 20260729T085719Z-visual-course-4455fd61:
                # a ~0.27 s camera stall republished one frozen frame id and
                # the stale close-range loss latched zero thrust at the
                # gate-0 top bar, so a superseded frame must never arm this.
                # The F22 fh-untrusted guard (flight 6bebd725: zero thrust
                # latched at speed) was retired in F33: the coast has held
                # the SUPPORT collective at level attitude since F25, so
                # there is no zero-thrust drop to guard against — and the
                # guard blocked the F32 coast at the engulfed gate-0 plane
                # (fh was high from BRAKING drag, not speed), sending the
                # drone blind into the frame in PREDICT instead.
                self.state = CleanCourseState.COAST_FOR_CREDIT
                self._coast_zero_sent = False
                self._coast_race_boot_ms = self._last_race_boot_ms
            else:
                if not fresh and self.state is CleanCourseState.TRACK:
                    # Frozen-frame stall: the republication carries no new
                    # information, so predict (covariance inflates in
                    # _predict) and let command() decay the collective toward
                    # support instead of coasting or holding a stale fix.
                    self.state = CleanCourseState.PREDICT
                if gap > cfg.predict_frame_gap_s:
                    self.state = CleanCourseState.PREDICT
                anchored = (
                    self._last_engulfing_anchor_s is not None
                    and now_s - self._last_engulfing_anchor_s
                    <= ENGULFING_ANCHOR_MAX_AGE_S
                )
                if (
                    not anchored
                    and self.state is CleanCourseState.PREDICT
                    and (
                        gap > cfg.predict_max_gap_s
                        or self.current.position_std
                        > cfg.search_covariance_std_norm
                    )
                ):
                    self._enter_search(now_s)

        # F42 anti-deadlock: an adopted debris splinter whose x-axis can
        # NEVER be measured held TRACK for 0.8 s with the F41 x-steer gate
        # freezing yaw/roll at 0 until the splinter died on its own.  An
        # unmeasurable adopted hypothesis this old must not hold the state.
        if (
            self.state is CleanCourseState.TRACK
            and self.current is not None
            and self.current.last_x_measurement_s <= NEVER_MEASURED_S + 1
            and now_s - self.current.created_s > UNMEASURED_X_FORCE_SEARCH_S
        ):
            self._enter_search(now_s)

        self._refresh_successor(tracks, now_s)
        if self.current is not None and match is not None:
            self._set_reliable_bearing(self.current.x, self.current.y)

    # -- authoritative race authority ---------------------------------------

    def note_race(
        self,
        *,
        gate_index: int,
        race_boot_ms: int,
        now_s: float,
    ) -> bool:
        """Accept authoritative race state.  Promotion is an event.

        Returns True when an authoritative gate increment was accepted.
        """

        self._last_race_boot_ms = int(race_boot_ms)
        if (
            self.state is CleanCourseState.COAST_FOR_CREDIT
            and self._coast_race_boot_ms is not None
            and int(race_boot_ms) > self._coast_race_boot_ms
            and int(gate_index) == self.gate_index
        ):
            # A strictly newer race packet arrived without credit: the
            # crossing was not authoritative.  Resume searching.
            self._exit_coast()
            self._enter_search(now_s)
        if int(gate_index) <= self.gate_index:
            return False

        previous = self.gate_index
        self.gate_index = int(gate_index)
        self.max_gate_index = max(self.max_gate_index, self.gate_index)
        self.transitions.append((previous, self.gate_index))
        # Promotion: the next gate's pursuit gets a fresh orbit trim.
        self._ex_trim = 0.0
        if self.state is CleanCourseState.COAST_FOR_CREDIT:
            self._exit_coast()
        # F76: an authoritative increment settles the pending-credit hold.
        pending_credit = self._pending_credit_until_s is not None
        self._pending_credit_until_s = None

        successor = self.successor
        credible = (
            successor is not None
            and successor.position_std <= self.config.promote_max_std_norm
            and now_s - successor.last_measurement_s
            <= self.config.promote_max_age_s
            # F40: the promoted successor must have a real measured x-axis —
            # NEVER_MEASURED_S (-1e9) fails every horizon check, so an
            # edge-clipped splinter with an unmeasured x can never be
            # adopted as the aim point.
            and now_s - successor.last_x_measurement_s
            <= self.config.promote_max_age_s
            # F42: the successor must also be PERSISTENT — debris is newborn
            # every frame while the real gate stays associated; a
            # never-seen id ages 0 and fails.
            and self._track_age_s(successor.track_id, now_s)
            >= self.config.successor_min_age_s
        )
        if credible:
            self.current = successor
            self.successor = None
            self.state = CleanCourseState.TRACK
            self._set_reliable_bearing(self.current.x, self.current.y)
        elif (
            pending_credit
            and successor is not None
            and now_s - successor.last_x_measurement_s
            <= ENGULFING_ANCHOR_MAX_AGE_S + self.config.pending_credit_hold_s
            and self._track_age_s(successor.track_id, now_s)
            >= self.config.successor_min_age_s
        ):
            # F76: delayed credit after a pending-credit hold.  The coast
            # crossing swallowed fresh measurements, so the retained
            # successor is stale by construction — but it is the only real
            # bearing evidence on the new leg, and the F75 sweep away from
            # it blinded the leg within 0.4 s.  Re-acquire it directly
            # (persistence still required: newborn debris never qualifies)
            # and let the normal TRACK law re-center on it.
            self.current = successor
            self.successor = None
            self.state = CleanCourseState.TRACK
            self._set_reliable_bearing(self.current.x, self.current.y)
        else:
            self.current = None
            # Off-by-one fix (codex review): _refresh_successor writes the
            # cache under the gate being ATTACKED (pre-promotion index), so
            # the newly-current gate's bearing lives under gate_index - 1.
            cached = self.successor_bearing_cache.get(self.gate_index - 1)
            if successor is not None:
                self._set_reliable_bearing(successor.x, successor.y)
            elif cached is not None:
                self._set_reliable_bearing(cached[0], cached[1])
            self._enter_search(now_s)
        # Re-seed the collective tracker so a retained saturated sub-support
        # command can never survive into the next gate.
        self._collective = None
        # Re-arm the course-heading anchor for the new leg (F31).
        self._course_anchor_yaw_rad = None
        return True

    # -- the one continuous control law -------------------------------------

    def command(
        self,
        *,
        now_s: float,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: Optional[float] = None,
        world_up_accel_m_s2: Optional[float] = None,
        horizontal_specific_force_mps2: Optional[float] = None,
    ) -> NavigationOutput:
        """Produce the single navigation request for one tick."""

        cfg = self.config
        self._pre_cross_brake_active = False  # main path recomputes below
        if yaw_rad is not None and self._course_anchor_yaw_rad is None:
            self._course_anchor_yaw_rad = float(yaw_rad)
        if self._last_command_s is None:
            dt = cfg.control_period_s
        else:
            dt = _clamp(now_s - self._last_command_s, 1e-3, 0.10)
        self._last_command_s = float(now_s)

        # F14 inflow-regime gate (see the FH_* constant block): vz_est is
        # invalidated by REGIME (fh-proportional DC thrust deficit), not by
        # attitude or vibration.  Sustained fh above the trigger latches the
        # untrusted state; it clears below the release hysteresis.
        if horizontal_specific_force_mps2 is not None:
            self._fh_mps2 = float(horizontal_specific_force_mps2)
        if self._fh_untrusted:
            if self._fh_mps2 < cfg.fh_trusted_release_mps2:
                self._fh_untrusted = False
                self._fh_above_since_s = None
        elif self._fh_mps2 > cfg.fh_untrusted_trigger_mps2:
            if self._fh_above_since_s is None:
                self._fh_above_since_s = now_s
            elif now_s - self._fh_above_since_s >= cfg.fh_untrusted_sustain_s:
                self._fh_untrusted = True
        else:
            self._fh_above_since_s = None

        if world_up_accel_m_s2 is not None:
            if not self._fh_untrusted:
                # Leaky world-vertical-rate integrator for the climb
                # governor; IMU-fed so it stays alive in every state,
                # including COAST.
                self._vz_est_m_s += float(world_up_accel_m_s2) * dt
            # While fh-untrusted the integration is SUSPENDED (a biased
            # regime a_up is never integrated) and only the leak relaxes
            # the frozen estimate toward 0.
            self._vz_est_m_s -= self._vz_est_m_s * dt / VZ_LEAK_TAU_S
        if not self._fh_untrusted:
            # IMU altitude estimate from course start (takeoff pad = 0),
            # clamped below (F13: a biased integrator reached -10.7 m,
            # physically impossible, and drove the floor deeper than its
            # guard band needs).  Held while fh-untrusted (F14).
            self._alt_est_m = max(
                cfg.alt_est_min_m, self._alt_est_m + self._vz_est_m_s * dt
            )

        if self.state is CleanCourseState.COAST_FOR_CREDIT:
            if self._coast_zero_sent:
                # F72: exactly ONE wire-zero send is the whole credit wait,
                # bounded by the send count rather than another timeout
                # value — F68/F69/F71 showed every timed window pays a
                # multi-tick ballistic drop.  Credit remains acceptable in
                # EVERY state, so after the single zero the wait continues
                # as a normal SEARCH.
                # F76: ...but while the authoritative packet is still in
                # flight the search HOLDS the course heading (bounded) —
                # the generic sweep yawed away from the retained successor
                # and blinded the new leg (see PENDING_CREDIT_HOLD_S).
                self._exit_coast()
                self._pending_credit_until_s = (
                    now_s + cfg.pending_credit_hold_s
                )
                self._enter_search(now_s)
            else:
                self._coast_zero_sent = True
                # July-18 contract item 9: the credible-crossing credit wait
                # is EXACT WIRE ZERO on all four channels while awaiting a
                # newer authoritative race packet.  The F25/F26
                # support-thrust coast through the attitude PD is out of
                # contract: an actively driven blind wait is still blind
                # driving.  The send path bypasses the attitude PD for this
                # state so the wire is exactly 0/0/0/0; credit remains the
                # only passage authority.
                return NavigationOutput(
                    target_roll_rad=0.0,
                    target_pitch_rad=0.0,
                    yaw_rate_rad_s=0.0,
                    thrust=0.0,
                    state=self.state,
                    gate_index=self.gate_index,
                    current_track_id=self._current_track_id(),
                    successor_track_id=self._successor_track_id(),
                )

        # Hard PREDICT stall cap (flight d5e89c2b): anchor expiry lives in
        # observe(), which a continuously refreshed anchor can keep
        # suppressing indefinitely.  command() runs every tick, so it owns
        # the last-resort bound: PREDICT this long without an accepted
        # measurement forces SEARCH regardless of anchor state.
        if (
            self.state is CleanCourseState.PREDICT
            and self.current is not None
            and now_s - self.current.last_measurement_s
            > PREDICT_STALL_FORCE_SEARCH_S
        ):
            self._enter_search(now_s)

        # F95: the tilt compensation is relative to the LEVEL (spawn)
        # attitude, not absolute rpy.  The -0.31 spawn attitude is an rpy
        # frame offset — the body is level there (F38: stationary, span
        # flat), so true tilt is pitch_rad - spawn_pitch_rad.  The
        # absolute-rpy formula inflated support at the brake attitude
        # (0.2906 at -0.577 vs the true ~0.269 at 0.267 rad from level),
        # an open-loop +0.9 m/s^2 climb bias — the chronic gate-1 balloon
        # (F90/F91/F93/F94 all ballooned vz +0.4..+0.5 at the brake
        # attitude) that the vertical arrest kept chasing downstream.  At
        # level both formulas agree (0.2594), so every level-tuned path
        # (the F14/F21 margins, the F83 floor, gate 0's envelope) is
        # unchanged; F83 already proved the same inflation wrong in the
        # fh-untrusted floor and bounded it there.
        level_hover = cfg.support_collective / math.cos(cfg.spawn_pitch_rad)
        support = _clamp(
            level_hover
            / max(
                0.85,
                math.cos(roll_rad)
                * math.cos(pitch_rad - cfg.spawn_pitch_rad),
            ),
            cfg.min_thrust,
            cfg.max_thrust,
        )

        # F92 (20260730T135630Z-visual-course-2aa541ba): the pre-gate-1
        # altitude-floor latch is DELETED.  Its max() pinned support+0.05
        # over the governor and arrest for four ballooned gate-1 legs
        # (F78/F78b/F79, then F91 armed blind at alt -0.09 and held the pin
        # for the whole leg into a gate-1 structure hit); both yield
        # patches (F79, F86) failed in flight.  Blind anti-sink is owned by
        # the vz-center trim (F88/F89/F90) and the continuous vz descent
        # floor, which arrested F90's blind -0.5 sink in ~0.5 s.

        # F53 near-plane COMMIT (see the COMMIT_* constant block): the
        # misalignment brake self-locks short of the plane, so a sustained,
        # aligned, freshly measured close regime commits to an inertial
        # crossing.  TRACK only; gate-1+ legs only (gate-0's climb-bias path
        # is working and stays untouched).  F54: proximity arms at
        # commit_min_log_scale (-1.2), NOT near_brake_log_scale — censorship
        # onset coincides with the -0.9 crossing, killing the fresh-
        # uncensored window before the old threshold ever sustained.
        near_plane_close = (
            self.state is CleanCourseState.TRACK
            and self.current is not None
            and self.current.outer_log_scale >= cfg.commit_min_log_scale
        )
        if near_plane_close:
            if self._near_plane_since_s is None:
                self._near_plane_since_s = now_s
        else:
            self._near_plane_since_s = None
        # 2026-07-30 unified entry budget (see the COMMIT_ENTRY_* block):
        # ONE crossing policy.  Entry requires the CURRENT frame to prove a
        # usable inner aperture, uncensored both-axis measurements, low
        # closure, and error+predicted-blackout-drift inside the aperture
        # with margin — worst case of raw vs filtered.  A failed budget
        # never loosens: TRACK keeps braking/re-centering outside the
        # censorship blackout until the crossing is provably aligned.
        if (
            near_plane_close
            and self.gate_index >= 1
            and now_s - self._near_plane_since_s >= cfg.commit_sustain_s
            and self._commit_entry_budget_ok(now_s, pitch_rad, cfg)
        ):
            self.state = CleanCourseState.COMMIT
            self._commit_entry_s = float(now_s)

        if self.state is CleanCourseState.COMMIT:
            commit_timed_out = (
                self._commit_entry_s is None
                or now_s - self._commit_entry_s > cfg.commit_timeout_s
            )
            if self.current is None or commit_timed_out:
                # No authoritative credit inside the bounded commit window:
                # arrest forward motion (the SEARCH branch below slews
                # pitch back to level) and search.  Dropping the hypothesis
                # is REQUIRED — the innovation gate permanently rejects the
                # true gate while the frozen hypothesis lives (association
                # lock-out).
                self.current = None
                self._commit_entry_s = None
                self._enter_search(now_s)
            else:
                # Inertial crossing with a fresh-window finish (F56, trace
                # efb189d4): while the x measurement is fresh the derotated
                # hypothesis is the best aim evidence, so steer with the
                # TRACK-style P gains to finish centering (F55 froze a
                # ~0.22 norm entry offset and crossed beside the left post;
                # ~0.25 s of uncensored measurements remained after entry).
                # Once x is stale/censored, steer on the PREDICTED
                # hypothesis at half gain (F62) — heading-hold commits the
                # residual drift: F61's prediction tracked the real bearing
                # (-0.02 -> -0.15 across the blackout) while heading-hold
                # zeroed corrections and the drone clipped the left post.
                # (F52's over-rotation was a frozen TRACK hold far from the
                # plane; the COMMIT prediction is bounded by the 3.0 s
                # timeout.)  The vertical channel
                # keeps the F50 compensated-ey servo BOUNDED to a small band
                # around support (a flat support hold repeats the F33/F34
                # bottom-bar death vertically); the vz governor stays the
                # climb/sink limiter.  Only
                # the progress-removers are bypassed: misalignment brake,
                # closure governor, expansion factor, x-staleness zeroing
                # (F52-A), brake-relax.  The engulfing anchor is never
                # consulted for steering.
                # F57: the near-plane boost applies here too — at
                # commit range the far-range gains limit-cycle against
                # parallax (see NEAR_PLANE_STEER_GAIN_MULT).  F62/F63: stale
                # x steers the prediction at FULL gain — F61/F62 proved the
                # prediction tracks the real bearing through the blackout,
                # and the F62 half-gain derate under-corrected (crossed
                # -0.22 norm left at the plane).
                commit_steer_gain = 1.0
                if self.current.log_scale >= cfg.commit_min_log_scale:
                    commit_steer_gain = cfg.near_plane_steer_gain_mult
                # 2026-07-30: the aim carries the TRACK-learned orbit trim
                # (EX_TRIM_* block) instead of the F63 fixed 0.08 bias — the
                # trim nulls the repeatable lateral P-loop equilibrium at
                # its source, so the entry arrives centered about the true
                # crossing point rather than compensating it blind.
                commit_ex = self.current.x - self._ex_trim
                commit_yaw = _clamp(
                    cfg.yaw_error_sign
                    * cfg.yaw_error_gain
                    * commit_steer_gain
                    * commit_ex,
                    -cfg.max_yaw_rate_rad_s,
                    cfg.max_yaw_rate_rad_s,
                )
                commit_yaw = self._anchor_clamped_yaw(commit_yaw, yaw_rad)
                commit_roll = _clamp(
                    cfg.roll_error_sign
                    * cfg.roll_error_gain
                    * commit_steer_gain
                    * commit_ex,
                    -cfg.max_target_roll_rad,
                    cfg.max_target_roll_rad,
                )
                commit_correction = _clamp(
                    cfg.vertical_feedback_sign
                    * cfg.vertical_error_gain
                    * self._compensated_ey(self.current.y, pitch_rad),
                    -cfg.search_vertical_memory_band,
                    cfg.search_vertical_memory_band,
                )
                if (
                    now_s - self.current.last_y_measurement_s
                    > cfg.commit_meas_max_age_s
                ):
                    # F58: y is stale/censored — never CLIMB on a frozen
                    # bearing (F57's frozen servo pinned support+0.05 for
                    # the whole commit and drifted up over the opening).
                    # The descend side stays live; the vz governor bounds
                    # the sink.
                    commit_correction = min(commit_correction, 0.0)
                commit_hold = support + commit_correction
                self._collective = commit_hold
                # F66: the F60 vertical-aim pitch term is DELETED.  In
                # commit the attitude is the forward drive, not a second
                # vertical channel — the aim and the collective servo read
                # the same ey and fought at the plane: the dive rotated the
                # camera down (growing ey -> more dive) while the body kept
                # its pre-dive velocity vector.  F63 dove UNDER gate 1;
                # F65 (11b13f53) slammed the top panel 0.47 s after entry
                # with the opening 0.45 norm below the aim.  Vertical
                # translation now lives ONLY in the bounded ey servo above
                # (plus the vz descent floor).
                return NavigationOutput(
                    target_roll_rad=self._slew_roll(commit_roll, dt),
                    # F55: the advance attitude must actually be reached —
                    # the generic 0.30 rad/s slew only moved rpy_p from
                    # -0.42 to -0.32 across F54's whole 2 s commit.  Use
                    # the braking-regime fast slew.  F58: the drive is
                    # COMMIT_ADVANCE_PITCH_RAD (the coast's 0.05 offset was
                    # sized for a 0.4 s wait, not a 3-4 m brake-stall
                    # drive); the coast law's own offset is untouched.
                    target_pitch_rad=self._slew_pitch(
                        cfg.spawn_pitch_rad + cfg.commit_advance_pitch_rad,
                        dt,
                        slew_rad_s=cfg.pre_cross_brake_slew_rad_s,
                    ),
                    yaw_rate_rad_s=commit_yaw,
                    thrust=self._governed_collective(commit_hold, support),
                    state=self.state,
                    gate_index=self.gate_index,
                    current_track_id=self._current_track_id(),
                    successor_track_id=self._successor_track_id(),
                )

        if self.state is CleanCourseState.SEARCH:
            # Flight 25361816: the unqualified hold margin must apply here
            # too — SEARCH at bare support in the fh-untrusted regime sank
            # ~1 m/s for real into terrain (the margin only covered the
            # TRACK path).
            margin = (
                cfg.fh_untrusted_vertical_margin if self._fh_untrusted else 0.0
            )
            if (
                self._pending_credit_until_s is not None
                and now_s < self._pending_credit_until_s
            ):
                # F76 pending-credit heading hold (see the
                # PENDING_CREDIT_HOLD_S block): the authoritative packet is
                # still in flight — level pitch, governed altitude support,
                # NO generic sweep and NO forward advance: F75 swept +0.15
                # away from the retained left-side successor and blinded
                # the new leg within 0.4 s.  F78
                # (20260730T082159Z-visual-course-7e18243d): the neutral
                # hold also DELAYED the turn — gate 1 sat visible at
                # x ~-0.51 for the whole window and steering only began at
                # the delayed credit, handing the new leg a saturated
                # constant-bearing pursuit it never centered.  While
                # credit is in flight, a fresh/persistent/qualified
                # successor's BEARING steers a bounded recentering (same
                # gain and cap as the TRACK law, no roll, no advance, no
                # promotion — authoritative ownership is unchanged);
                # absent or ambiguous evidence keeps the neutral hold.
                # F90: blind paths hold altitude by definition, so any
                # IMU-trusted sink is unwanted — wind the support trim
                # with no ey gate (see _wind_vz_center_trim).
                self._wind_vz_center_trim(dt)
                hold = support + margin + self._vz_center_trim
                self._collective = hold
                recenter_yaw = 0.0
                successor = self.successor
                if (
                    successor is not None
                    and successor.position_std <= cfg.promote_max_std_norm
                    and now_s - successor.last_measurement_s
                    <= cfg.promote_max_age_s
                    and now_s - successor.last_x_measurement_s
                    <= cfg.promote_max_age_s
                    and self._track_age_s(successor.track_id, now_s)
                    >= cfg.successor_min_age_s
                ):
                    recenter_yaw = _clamp(
                        cfg.yaw_error_sign * cfg.yaw_error_gain * successor.x,
                        -cfg.max_yaw_rate_rad_s,
                        cfg.max_yaw_rate_rad_s,
                    )
                return NavigationOutput(
                    target_roll_rad=self._slew_roll(0.0, dt),
                    target_pitch_rad=self._slew_pitch(
                        cfg.spawn_pitch_rad + cfg.brake_pitch_rad, dt
                    ),
                    yaw_rate_rad_s=recenter_yaw,
                    thrust=self._governed_collective(hold, support),
                    state=self.state,
                    gate_index=self.gate_index,
                    current_track_id=self._current_track_id(),
                    successor_track_id=self._successor_track_id(),
                )
            # F49: absolute-heading sweep from the search-entry heading,
            # first toward the last reliable bearing — the F40 anchor-
            # centered sweep re-centered the scan on the course heading
            # instead of where the target was last seen.
            sweep_yaw = self._search_yaw_heading(dt, yaw_rad)
            # F75: altitude support is LATCHED in no-track SEARCH.  The F50
            # memory-descent servo turned F74's gate-1 miss into a blind
            # 4.8 s sink to the alt-est floor (-2.0) while fh grew 1.3 ->
            # 4.3 — a searching drone that descends on a frozen bearing
            # converts a recoverable miss into a floor/structure crash.
            # SEARCH holds altitude at support; re-acquisition is the
            # sweep's job.  (The TRACK/COMMIT ey servo is untouched.)
            # F90: the support trim winds here too — a blind SEARCH at
            # bare support sank F89's gate-1 leg vz -0.47 for 5+ s into
            # the ground while the sweep ran.
            self._wind_vz_center_trim(dt)
            search_hold = support + margin + self._vz_center_trim
            self._collective = search_hold
            target_roll = self._slew_roll(0.0, dt)
            # F49: SEARCH always holds the LEVEL (spawn-attitude) pitch.
            # The F31/F40 blind-at-speed brake was built on absolute pitch
            # targets ~0.3 rad nose-down of intent (level flight is the
            # -0.31 spawn attitude, not 0), so the "brake" never braked;
            # under the spawn-relative convention the honest blind posture
            # is level — the gentle sweep and the vz governor carry the leg.
            target_pitch = self._slew_pitch(
                cfg.spawn_pitch_rad + cfg.brake_pitch_rad, dt
            )
            return NavigationOutput(
                target_roll_rad=target_roll,
                target_pitch_rad=target_pitch,
                yaw_rate_rad_s=sweep_yaw,
                # The IMU climb governor applies here too: vision loss must
                # never disable it.
                thrust=self._governed_collective(search_hold, support),
                state=self.state,
                gate_index=self.gate_index,
                current_track_id=self._current_track_id(),
                successor_track_id=self._successor_track_id(),
            )

        current = self.current
        if current is None:
            # Defensive: no hypothesis outside SEARCH should be impossible,
            # but never emit an unbounded command if it happens.
            self._enter_search(now_s)
            fallback_hold = support + (
                cfg.fh_untrusted_vertical_margin if self._fh_untrusted else 0.0
            )
            self._collective = fallback_hold
            return NavigationOutput(
                target_roll_rad=0.0,
                target_pitch_rad=cfg.spawn_pitch_rad + cfg.brake_pitch_rad,
                yaw_rate_rad_s=0.0,
                thrust=self._governed_collective(fallback_hold, support),
                state=self.state,
                gate_index=self.gate_index,
            )

        # Successor blending REMOVED from the lateral/vertical aim (flight
        # ab6252b2): track 07 (gate 1) was centered and approached to span
        # 0.34/conf 0.87, then slid left to x=-0.95 while the yaw command
        # sat at ~0 — the blend toward a far successor at x=+0.6 cancelled
        # the pursuit error exactly when the close gate escaped (codex:
        # "remove unproved successor blending").  The aim is always the
        # current gate; the successor hypothesis machinery stays for
        # promotion only.
        blend = 0.0
        ex = current.x
        ey = current.y
        # F50: the VERTICAL channel servos on the pitch-attitude-compensated
        # error (nose-up brake attitude reads the world LOW in frame; see
        # the VERTICAL_PITCH_COMP_NORM_PER_RAD block).  The angular-error
        # brake below keeps the RAW ey: it measures camera pointing, and
        # pitch attitude is itself the braking actuator, so compensating it
        # there would release the brake while still pitched up.
        ey_vertical = self._compensated_ey(ey, pitch_rad)
        # F40 (20260729T193134Z-visual-course-63ed6342): never steer on an
        # x-axis without a fresh accepted measurement — an unmeasured or
        # stale x (edge-clipped splinter, censored axis) is a garbage aim
        # point.  The y/vertical path is deliberately untouched (F35 servos
        # on censored-y by design).  F52: the near-plane regime is excepted
        # at the zeroing site below — there the derotated hypothesis is the
        # best aim evidence and the crossing completes in <1 s.
        x_qualified = (
            now_s - current.last_x_measurement_s <= cfg.x_steer_max_age_s
        )
        # 2026-07-30 TRACK-phase lateral trim (EX_TRIM_* block): the off-axis
        # pursuit is a P-loop that equilibrates at ex ~-0.08 every flight
        # (yaw holds the bearing constant against translation parallax).
        # Integrate the raw ex while fresh and near the linear regime; the
        # bounded trim learns the feedforward that nulls the orbit BEFORE
        # entry, replacing the F62 COMMIT-only 0.08 bias.  Steady state of
        # trim += ki*ex is ex -> 0, not ex -> trim.
        if (
            self.state is CleanCourseState.TRACK
            and self.gate_index >= 1
            and x_qualified
            and abs(current.x) <= cfg.ex_trim_active_ex_norm
        ):
            self._ex_trim = _clamp(
                self._ex_trim + cfg.ex_trim_gain * current.x * dt,
                -cfg.ex_trim_max_norm,
                cfg.ex_trim_max_norm,
            )
        ex -= self._ex_trim

        # Vertical: ONE GLOBAL SIGN at every gate (empirically confirmed by
        # the 2026-07-29 crossing-geometry analysis).  The gate-0 phase adds
        # a feedforward vertical setpoint offset so the vehicle crosses
        # higher and gate 1 is first seen with doubled top-edge margin;
        # the offset never changes the feedback sign and disappears on
        # promotion.  The offset is closure-scaled: flight
        # 20260729T085719Z-visual-course-4455fd61 held the fixed 0.25 bias
        # into gate 0's top bar, so it ramps linearly from full at the spawn
        # detection log scale to zero at the crossing-arm log scale.  And it
        # may only push the aim point UP toward image center, never above
        # it: flight 20260729T094736Z-visual-course-9d430a40 kept a positive
        # (~+0.1..0.2) setpoint past center through t=1.5, holding collective
        # >= support and delaying the descent until the climb could no
        # longer be erased, so the offset is clamped to <= 0 whenever the
        # gate is at/below center (ey >= 0).  Loss of qualified vertical
        # state discards the derivative term and decays collective smoothly
        # toward tilt-compensated support; a saturated sub-support collective
        # is never retained.
        vertical_setpoint_offset = 0.0
        if self.gate_index == 0:
            span = (
                cfg.gate0_climb_reference_log_scale - cfg.crossing_min_log_scale
            )
            closure = (
                _clamp01((current.log_scale - cfg.crossing_min_log_scale) / span)
                if abs(span) > 1e-9
                else 0.0
            )
            vertical_setpoint_offset = (
                cfg.gate0_climb_vertical_offset_norm * closure
            )
            if ey_vertical >= 0.0:
                # Gate at/below center (attitude-compensated, F50): the
                # climb bias may not lift the aim above center (flight
                # 9d430a40, see block comment).
                vertical_setpoint_offset = min(vertical_setpoint_offset, 0.0)
        vertical_qualified = (
            self.state is CleanCourseState.TRACK
            and now_s - current.last_y_measurement_s
            <= cfg.vertical_qualify_max_age_s
            and current.y_axis.std <= cfg.search_covariance_std_norm
        )
        # Vision closure-rate governor (F31, see the CLOSURE_* constant
        # block): the filtered log-scale rate is the only honest closure
        # signal — fh is a signless drag magnitude that conflates speed with
        # braking.  Speed is capped CONTINUOUSLY at every range: the pitch
        # target below blends from the advance law toward the gentle brake
        # attitude as the expansion rate rises past the target.  Applies in
        # TRACK and PREDICT alike (the SEARCH path returned above).
        # F33 trust gate: expansion from tiny far tracks is sub-pixel noise,
        # so the governor only runs above CLOSURE_MIN_LOG_SCALE.
        closure_rate = (
            current.expansion_rate
            if current.log_scale >= cfg.closure_min_log_scale
            else 0.0
        )
        closure_brake = _clamp01(
            (closure_rate - cfg.closure_target_rate_s)
            / (cfg.closure_full_brake_rate_s - cfg.closure_target_rate_s)
        )
        # Misalignment brake (F35, d25f23fe): a fully misaligned gate only
        # suppressed ADVANCE, leaving the pitch law at brake_pitch (near
        # level, still creeping forward) — the gate-1 leg held yaw at the
        # cap with pitch ~level while fh grew 4 -> 7.3 into gate-1-area
        # structure.  Speed with no alignment is pure risk: blend toward
        # the TRUE brake attitude with the same signal that suppresses
        # advance.
        angular_error = math.hypot(ex, ey)
        align = _clamp01(1.0 - angular_error / cfg.angular_full_brake_norm)
        brake_demand = max(closure_brake, 1.0 - align)
        pre_cross_brake = brake_demand > 0.5
        self._pre_cross_brake_active = pre_cross_brake
        arrest_error: Optional[float] = None  # qualified branch sets this
        vz_tracked = False  # qualified gate-1+ branch sets this (F96)
        if vertical_qualified:
            bounded_error = _clamp(
                ey_vertical - vertical_setpoint_offset,
                -cfg.vertical_max_abs_error_norm,
                cfg.vertical_max_abs_error_norm,
            )
            bounded_rate = _clamp(
                current.vy,
                -cfg.vertical_max_abs_rate_norm_s,
                cfg.vertical_max_abs_rate_norm_s,
            )
            # Full D authority.  The flight-2 direction limiter (clip D to
            # |P| on disagreement) was REMOVED after flight
            # 20260729T094736Z-...-4dbe4b8c: with ey hovering near zero it
            # zeroed the only vz feedback and pinned collective at exactly
            # tilt-compensated support through the decisive t=1.31-1.72
            # window at vz 2.7 m/s.  Honest climb-rate limiting now lives in
            # the IMU vz governor below, which cannot go blind with vision.
            # One GLOBAL vertical law at every gate (the F67 gate-dependent
            # D-removal was reverted): a high equilibrium is now refused at
            # the COMMIT entry budget instead of tuned out per-gate.
            # VZ_CENTER_TRIM (see the constant block): the PD is too weak
            # around center to hold altitude, so an underestimated support
            # settles ON the descent-floor boundary (or droops below it)
            # and the approach sinks with ey ~0.  Learn the support
            # correction while TRACK holds the gate near center with a
            # qualified, IMU-trusted vertical channel; hold on demanded
            # descents (ey > active bound), leak on real climbs, and never
            # wind into a saturated collective.  The trim is added AFTER
            # the PD so every governor and floor below sees the corrected
            # demand.  F90: the same integrator also covers the blind
            # paths via _wind_vz_center_trim call sites below.
            if (
                self.state is CleanCourseState.TRACK
                and ey_vertical <= cfg.vz_center_trim_active_ey_norm
            ):
                self._wind_vz_center_trim(dt)
            if self.gate_index >= 1 and not self._fh_untrusted:
                # F96 unified course-leg vertical law (see the
                # COURSE_VZ_* constant block): ONE vz-tracking term
                # replaces the stacked camera D + arrest + closure
                # collective cut + course governor that limit-cycled
                # clamp-to-clamp in F95.  The desired climb rate is
                # proportional to the compensated error — it goes to
                # zero as ey approaches center, so vertical energy is
                # arrested BEFORE the center crossing by construction.
                # Gate 0 keeps the proved PD envelope below; the F14
                # fh-untrusted path cannot track a frozen vz_est and
                # keeps the camera PD.
                # F97: reconcile the setpoint with the COMMIT |vz| <= 0.25
                # entry budget (see the COURSE_VZ_DES_COMMIT_M_S block):
                # the cap ramps 0.5 -> 0.20 across the approach so the
                # tracker cannot carry a budget-vetoing vertical rate
                # into the plane.
                ramp = _clamp01(
                    (current.log_scale - cfg.course_vz_des_ramp_start_log_scale)
                    / (
                        cfg.commit_min_log_scale
                        - cfg.course_vz_des_ramp_start_log_scale
                    )
                )
                vz_des_max = cfg.course_vz_des_max_m_s - (
                    cfg.course_vz_des_max_m_s - cfg.course_vz_des_commit_m_s
                ) * ramp
                vz_des = _clamp(
                    -cfg.course_vz_des_per_norm * bounded_error,
                    -vz_des_max,
                    vz_des_max,
                )
                # F97: a wound positive trim while the tracker wants LESS
                # climb than the measured vz is suspect by construction
                # (it fought the tracker for F96's whole leg); bleed it
                # fast, then emit with the bled value.
                if (
                    self._vz_center_trim > 0.0
                    and self._vz_est_m_s > vz_des + 0.05
                ):
                    self._vz_center_trim = max(
                        0.0,
                        self._vz_center_trim
                        - cfg.course_vz_trim_fast_leak_s * dt,
                    )
                collective = (
                    support
                    + self._vz_center_trim
                    + cfg.course_vz_track_gain * (vz_des - self._vz_est_m_s)
                )
                vz_tracked = True
            else:
                collective = (
                    support
                    + self._vz_center_trim
                    + cfg.vertical_feedback_sign
                    * (
                        cfg.vertical_error_gain * bounded_error
                        + cfg.vertical_rate_gain * bounded_rate
                    )
                )
            # The F78/F91/F93 vertical energy arrest needs this branch's
            # bounded error, hoisted here; F96 scopes it to gate 0 (the
            # gate-1+ qualified path is the unified vz-tracking law).
            arrest_error = bounded_error
            self._collective = collective
        else:
            # F14: while fh-untrusted the camera is the only honest vertical
            # channel; when it is unqualified, hold support + margin instead
            # of bare support, which historically sinks for real (-0.8...
            # -1.9 m/s against the biased-regime thrust deficit).
            # High-gate climb bias (see the HIGH_GATE_* block): a censored
            # y-axis must not sink the drone below a gate that sits HIGH —
            # the hypothesis y is still the best evidence of that.
            margin = (
                cfg.fh_untrusted_vertical_margin if self._fh_untrusted else 0.0
            )
            # Gate-high classification uses the compensated error too (F50):
            # a nose-up brake attitude must not read a level gate as LOW,
            # and the historical nose-down dives must not read it as HIGH.
            if ey_vertical < cfg.high_gate_y_norm:
                margin = max(margin, cfg.high_gate_climb_margin)
            # F90: the unqualified hold is blind — wind the support trim
            # against any IMU-trusted sink (no ey gate; a stale y-axis was
            # what made the path blind in the first place).
            self._wind_vz_center_trim(dt)
            hold = support + margin + self._vz_center_trim
            if self._collective is None:
                self._collective = hold
            # Flight bc8c6003: a phantom vy (+0.38 norm/s, seeded as the gate
            # sank through the frame) random-walked unmeasured for 5.4 s and
            # commanded an unrecoverable descent.  A stale rate is never
            # reused: zero it while vertical is unqualified; the next real
            # measurement reseeds it through the filter coupling.
            current.y_axis.v = 0.0
            alpha = min(1.0, dt / cfg.collective_decay_tau_s)
            self._collective += (hold - self._collective) * alpha
            collective = self._collective

        # F96: the F77 closure-excess collective brake is DELETED.  It
        # only fired on the gate-1+ qualified TRACK path — exactly the
        # path the unified vz-tracking law now owns — where its min()
        # cut was a fourth incoherent vz term in the F95 limit cycle.
        # Its original evidence (F74/F75/F76: "attitude-only braking
        # never decelerated") was measured WITH the F95 support
        # inflation biasing the brake attitude +0.9 m/s^2 climb; forward
        # braking stays with the closure-governor pitch law above, and
        # vertical energy is the vz tracker's job.

        # F78/F91/F93 vertical energy arrest (see the VERTICAL_ARREST_*
        # block), F96-scoped to GATE 0 near the plane (F86: every
        # credited gate-0 crossing entered with dead vertical energy).
        # The gate-1+ qualified path that F91/F93 added the two-sided
        # law for is now the unified vz-tracking law, which arrests
        # energy continuously as the vz setpoint approaches zero near
        # center — no deadband, one coherent gain.
        near_plane = current.log_scale >= cfg.commit_min_log_scale
        if (
            arrest_error is not None
            and self.gate_index == 0
            and near_plane
            and not self._fh_untrusted
        ):
            if self._vz_est_m_s > 0.0:
                vz_allow = cfg.vertical_arrest_vz_per_norm * abs(arrest_error)
                excess = self._vz_est_m_s - vz_allow
                if excess > 0.0:
                    collective -= cfg.vertical_arrest_collective_gain * excess
            else:
                vz_allow = cfg.vertical_arrest_vz_per_norm * max(
                    0.0, arrest_error
                )
                excess = -self._vz_est_m_s - vz_allow
                if excess > 0.0:
                    collective += cfg.vertical_arrest_collective_gain * excess

        # Gate-0 takeoff boost is feedforward only; it never changes the
        # closed-loop vertical sign.
        if (
            self.gate_index == 0
            and self._course_start_s is not None
            and now_s - self._course_start_s < cfg.launch_boost_duration_s
        ):
            collective = cfg.launch_boost_thrust
        # IMU world-vertical-rate governor, after the PD law and the
        # feedforward boost, before the final clamp.  It caps what bearing
        # pursuit cannot (see the VZ_CLIMB_CAP_M_S constant block) and stays
        # alive through TRACK, PREDICT, and SEARCH alike.  The brake-ceiling
        # band inside classifies gate-high on the F50 compensated error.
        # F92: the latched pre-gate-1 altitude floor and its F79/F86 yield
        # kwarg are deleted (four ballooned gate-1 legs: F78/F78b/F79/F91);
        # blind anti-sink is owned by the vz-center trim and the continuous
        # vz descent floor.
        collective = self._governed_collective(
            collective, support, gate_y=ey_vertical, vz_tracked=vz_tracked
        )
        thrust = _clamp(collective, cfg.min_thrust, cfg.max_thrust)

        # Lateral: per the 2026-07-29 crossing-geometry analysis, positive
        # image-x error requires POSITIVE yaw (negative yaw rotates the
        # camera left and pushes a right-side target further right) and a
        # coordinated positive bank toward the target.  Both signs are
        # one-line flippable named constants pending first-flight
        # confirmation.  Clipping no longer saturates corrective steering
        # (codex, flights 4480d0a6/ab6252b2): the clip penalty halves yaw
        # exactly when the target is escaping at the frame edge.
        steer_cap = 1.0
        # F57 near-plane steering boost (see the NEAR_PLANE_STEER_GAIN_MULT
        # block): inside the COMMIT proximity regime the proved far-range
        # gains limit-cycle against close-range parallax (ex stalled at
        # -0.15..-0.18 for F56's whole approach).  Caps unchanged.
        steer_gain = 1.0
        if current.log_scale >= cfg.commit_min_log_scale:
            steer_gain = cfg.near_plane_steer_gain_mult
        yaw_rate = _clamp(
            cfg.yaw_error_sign * cfg.yaw_error_gain * steer_gain * ex,
            -cfg.max_yaw_rate_rad_s * steer_cap,
            cfg.max_yaw_rate_rad_s * steer_cap,
        )
        yaw_rate = self._anchor_clamped_yaw(yaw_rate, yaw_rad)
        target_roll = _clamp(
            cfg.roll_error_sign * cfg.roll_error_gain * steer_gain * ex,
            -cfg.max_target_roll_rad * steer_cap,
            cfg.max_target_roll_rad * steer_cap,
        )
        if not x_qualified and current.log_scale < cfg.near_brake_log_scale:
            # F40: no fresh x measurement -> no yaw/roll authority; hold
            # heading and wings level (slewing toward 0) instead of chasing
            # a phantom bearing off the frame.  F52 near-plane exception
            # (20260729T232037Z-visual-course-dedf1915): the F40 rationale
            # covers FAR targets lost off-frame.  At the gate plane the
            # derotated hypothesis bearing is the best aim evidence and the
            # crossing completes in <1 s — zeroing steering there (the aim
            # track went frame-censored at t=5.78, x_qualified expired at
            # t=6.25) flew ballistic from a still 0.3-off heading and
            # crossed gate 1's plane displaced, no credit.  Near the plane,
            # keep steering on the derotated hypothesis.
            yaw_rate = 0.0
            target_roll = 0.0

        # Pitch controls closure continuously: advance when aligned and
        # confident, brake progressively with angular error, uncertainty,
        # rapid expansion, or near-plane risk.  (angular_error/align and the
        # fused brake demand are computed with the closure governor above.)
        confidence = _clamp01(current.confidence)
        uncertainty = _clamp01(
            1.0 - current.position_std / cfg.search_covariance_std_norm
        )
        expansion = _clamp01(
            1.0
            - max(0.0, current.expansion_rate - cfg.expansion_brake_free_s)
            / cfg.expansion_brake_span_s
        )
        near_plane = _clamp01(
            (cfg.near_brake_log_scale - current.log_scale)
            / (cfg.near_brake_log_scale - cfg.near_free_log_scale)
        )
        advance = align * confidence * uncertainty * expansion * near_plane
        # F73 (20260730T063739Z-visual-course-34c53413): TRACK kept closing
        # while the COMMIT entry budget was false — the gate-1 aim walked
        # ey +0.31 -> +0.49 into bottom censorship at the plane, the track
        # died, a bottom-right splinter was adopted, and the drone wandered
        # 9 s blind into structure (collision id 1002).  At the plane the
        # angular error rate (~offset/distance) outruns any re-centering
        # servo, so the crossing energy must be controlled BEFORE
        # censorship: on a gate-1+ TRACK in the near-plane regime with a
        # false entry budget, cut the advance law and demand the full
        # brake — hold OUTSIDE the blackout and re-center.  The same budget
        # passing arms COMMIT on the same tick; nothing else about the
        # crossing changes.
        # F75: widen the hold from the censorship-onset zone (-0.9) to the
        # commit-regime entry (-1.2).  F74 held the true brake -0.46 from
        # -0.9 onward and closure still ROSE (fh 2.5 -> 3.6): by -0.9 the
        # approach energy is already beyond what the brake attitude can
        # arrest before the plane.  The only fresh, uncensored window on
        # the F54 timeline starts at -1.2, so arrest there — stop while
        # still seeing, re-center yaw/vertical at hover, and let the entry
        # budget arm COMMIT from a standstill instead of arriving hot.
        near_plane_hold = (
            self.state is CleanCourseState.TRACK
            and self.gate_index >= 1
            and current.outer_log_scale >= cfg.commit_min_log_scale
            and not self._commit_entry_budget_ok(now_s, pitch_rad, cfg)
        )
        if near_plane_hold:
            advance = 0.0
            brake_demand = 1.0
            pre_cross_brake = True
            self._pre_cross_brake_active = True
        # Closure-rate governor (F31) + misalignment brake (F35): continuous
        # blend toward the TRUE brake attitude as either the vision expansion
        # rate rises past the target or the gate sits off-axis — speed is
        # capped at every range, and never kept while not pointing at the
        # gate.
        law_pitch = cfg.spawn_pitch_rad + (
            cfg.brake_pitch_rad
            + (cfg.advance_pitch_rad - cfg.brake_pitch_rad) * advance
        )
        # F43 (20260729T202844Z-visual-course-ee8fd1e5): a lone small span is
        # ambiguous range evidence — "whole gate far away" or "fragment of a
        # gate that is NEAR" (the gate-1 leg advanced at +0.08 on a
        # span-(0.04,0.10) fragment, built fh 3-4, and parallax outran yaw
        # authority into the structure).  Below the span bound the advance
        # law is capped at the creep pitch while yaw/roll centering runs;
        # the closure governor and the brake blends above are untouched, and
        # a fused union or whole gate (span above the bound) advances fully.
        if current.log_scale < cfg.fragment_advance_min_log_scale:
            law_pitch = min(
                law_pitch, cfg.spawn_pitch_rad + cfg.fragment_creep_pitch_rad
            )
        # F80 (see the COURSE_PRE_CROSS_BRAKE_PITCH_RAD block): course legs
        # inherit the previous crossing's energy, so the TRUE brake doubles
        # to -0.30 from spawn at gate_index >= 1; gate 0 keeps the proved
        # -0.15.  Same single blend, same relax/custody machinery.
        brake_pitch_offset = (
            cfg.course_pre_cross_brake_pitch_rad
            if self.gate_index >= 1
            else cfg.pre_cross_brake_pitch_rad
        )
        target_pitch = law_pitch + brake_demand * (
            (cfg.spawn_pitch_rad + brake_pitch_offset) - law_pitch
        )
        # F94 custody-preserving brake floor — REPLACES the F51/F65/F71
        # binary relax latch, its hysteresis, and the F73b/F75
        # blind-arrest override with one continuous clamp.  The brake
        # attitude pitches the camera up and the gate slides DOWN the
        # frame; the latch relaxed the pitch target to FULL LEVEL the
        # moment raw/hypothesis ey crossed the bound.  F93
        # (20260730T143851Z-visual-course-7e67b464) shows that surrender
        # is the killer: the held brake had HALVED the closure rate
        # (0.45 -> 0.27 log/s) when the attitude artifact walked raw ey to
        # +0.24 >= 0.18, the latch leveled the camera with closure still
        # ~1.0/s, and the drone re-advanced into the gate-1 plane — the
        # entry budget correctly refused COMMIT (expansion, |vz|) and it
        # hit the structure (id 1001).  The compensated ey separates the
        # attitude artifact from true geometry: raw ey at a candidate
        # attitude is ey_vertical + (spawn - pitch) * comp_gain, so the
        # floor attitude that places the gate exactly ON the relax bound
        # is spawn - (bound - ey_vertical) / comp_gain.  The pitch target
        # may never go nose-up past it — the drone always holds the
        # MAXIMUM custody-compatible brake — and the floor never rises
        # above level: a genuinely low gate is re-centered by the vertical
        # channel, never by advancing.  ey_vertical is attitude-invariant
        # by construction, so the floor is stable without any hysteresis.
        commit_regime = current.log_scale >= cfg.commit_min_log_scale
        relax_bound = (
            cfg.near_brake_relax_course_ey_norm
            if commit_regime and self.gate_index >= 1
            else cfg.near_brake_relax_ey_norm
            if commit_regime
            else cfg.brake_relax_ey_norm
        )
        custody_floor = cfg.spawn_pitch_rad - (
            (relax_bound - ey_vertical) / cfg.vertical_pitch_comp_norm_per_rad
        )
        custody_floor = min(
            custody_floor,
            cfg.spawn_pitch_rad
            + (0.0 if commit_regime else cfg.brake_pitch_rad),
        )
        target_pitch = max(target_pitch, custody_floor)

        return NavigationOutput(
            target_roll_rad=self._slew_roll(
                target_roll,
                dt,
                slew_rad_s=(
                    cfg.roll_pursuit_slew_rad_s
                    if abs(ex) > cfg.roll_pursuit_fast_ex_norm
                    else None
                ),
            ),
            # The braking regime gets the dedicated fast slew (F12: the
            # generic 0.30 rad/s slew never attained the brake attitude
            # inside the hold); normal steering keeps the transparent slew.
            target_pitch_rad=self._slew_pitch(
                target_pitch,
                dt,
                slew_rad_s=(
                    cfg.pre_cross_brake_slew_rad_s
                    if pre_cross_brake
                    else None
                ),
            ),
            yaw_rate_rad_s=yaw_rate,
            thrust=thrust,
            state=self.state,
            gate_index=self.gate_index,
            advance_factor=advance,
            successor_blend=blend,
            vertical_qualified=vertical_qualified,
            current_track_id=self._current_track_id(),
            successor_track_id=self._successor_track_id(),
        )

    # -- internals -----------------------------------------------------------

    def _current_track_id(self) -> Optional[str]:
        return self.current.track_id if self.current is not None else None

    def _set_reliable_bearing(self, x: float, y: float) -> None:
        """Record real bearing evidence and validate the F50 memory servo."""

        self.last_reliable_bearing = (float(x), float(y))
        self._bearing_memory_valid = True

    def _commit_entry_budget_ok(
        self, now_s: float, pitch_rad: float, cfg: "CleanCourseConfig"
    ) -> bool:
        """2026-07-30 unified crossing-entry budget (COMMIT_ENTRY_* block).

        True only when the CURRENT frame proves: a passage-usable inner
        aperture, uncensored both-axis same-id measurements (never a
        0.30 s-old axis), closure at/below the governor target, and
        |error| + |rate| * blackout inside the aperture half-extents with
        margin on BOTH axes — the error evaluated as the WORST of the raw
        measurement and the filtered prediction.  Anything less keeps
        TRACK braking/re-centering outside the censorship blackout.
        """

        current = self.current
        if current is None:
            return False
        if (
            now_s - current.last_x_measurement_s
            > cfg.commit_entry_meas_max_age_s
            or now_s - current.last_y_measurement_s
            > cfg.commit_entry_meas_max_age_s
        ):
            return False
        if current.aperture_half_x is None or current.aperture_half_y is None:
            return False
        if current.expansion_rate > cfg.commit_entry_max_expansion_rate_s:
            return False
        # F82: the blackout-displacement budget includes IMU vz, not just
        # vision vy — a +0.64 m/s climb carried a "passing" entry into the
        # top bar.  Dead vertical energy only (see COMMIT_ENTRY_MAX_VZ_M_S).
        if abs(self._vz_est_m_s) > cfg.commit_entry_max_vz_m_s:
            return False
        ex = max(abs(current.x), abs(current.raw_x))
        if ex + abs(current.vx) * cfg.commit_blackout_s > (
            cfg.commit_entry_aperture_margin_frac * current.aperture_half_x
        ):
            return False
        ey = max(
            abs(self._compensated_ey(current.y, pitch_rad)),
            abs(self._compensated_ey(current.raw_y, pitch_rad)),
        )
        if ey + abs(current.vy) * cfg.commit_blackout_s > (
            cfg.commit_entry_aperture_margin_frac * current.aperture_half_y
        ):
            return False
        return True

    def _compensated_ey(self, ey: float, pitch_rad: float) -> float:
        """F50 pitch-attitude-compensated vertical error (image-down +).

        A nose-up attitude (rpy_p below spawn_pitch_rad) tilts the camera
        up and shifts the world DOWN in frame, so the measured ey reads
        high; zero at the spawn attitude.
        """

        cfg = self.config
        return float(ey) - (
            (cfg.spawn_pitch_rad - float(pitch_rad))
            * cfg.vertical_pitch_comp_norm_per_rad
        )

    def _successor_track_id(self) -> Optional[str]:
        return self.successor.track_id if self.successor is not None else None

    @staticmethod
    def _find(tracks: List[Any], track_id: Optional[str]) -> Optional[Any]:
        if track_id is None:
            return None
        for track in tracks:
            if track.track_id == track_id:
                return track
        return None

    def _track_age_s(self, track_id: Optional[str], now_s: float) -> float:
        """Seconds since this track id was first seen (0.0 if unknown)."""

        if track_id is None:
            return 0.0
        first_seen = self._track_first_seen_s.get(str(track_id))
        if first_seen is None:
            return 0.0
        return float(now_s) - first_seen

    def _hypothesis_from_track(self, track: Any, now_s: float) -> _Hypothesis:
        center, log_scale, _stds = _track_measurement(track)
        hypothesis = _Hypothesis(
            track_id=str(track.track_id),
            x=center[0],
            y=center[1],
            log_scale=log_scale,
            confidence=float(track.confidence),
            pos_var=INITIAL_POS_VAR_NORM,
            now_s=now_s,
        )
        # Flight bc8c6003: a censored axis on the creating detection is not a
        # measurement.  Never claim measurement freshness from it on adoption
        # or promotion rebind (the rate itself already seeds at zero); the
        # axis requalifies only when a real uncensored measurement arrives.
        clipping = getattr(track, "clipping", FrameEdge.NONE)
        if type(clipping) is not FrameEdge:
            clipping = FrameEdge.NONE
        center_censored = bool(getattr(track, "center_censored", False))
        x_censored = (
            center_censored or bool(clipping & (FrameEdge.LEFT | FrameEdge.RIGHT))
        )
        if x_censored:
            hypothesis.last_x_measurement_s = NEVER_MEASURED_S
        else:
            # F56: adoption with an uncensored x carries the creating
            # detection's true bbox half-width for the COMMIT corridor.
            bbox = getattr(track, "bbox_norm", None)
            if bbox is not None and len(bbox) >= 4:
                hypothesis.outer_half_span_x = 0.5 * (
                    float(bbox[2]) - float(bbox[0])
                )
        if center_censored or bool(clipping & (FrameEdge.TOP | FrameEdge.BOTTOM)):
            hypothesis.last_y_measurement_s = NEVER_MEASURED_S
        return hypothesis

    def _predict(
        self,
        hypothesis: _Hypothesis,
        dt: float,
        body_rates: Tuple[float, float, float],
    ) -> None:
        """Predict with short-term rotation compensation and latency growth.

        Frames are paired with the latest host-received IMU body rates.  The
        compensation uses a normalized-focal linear flow model; because the
        capture latency and exact camera response carry uncertainty, the
        covariance absorbs a fraction of the applied drift plus a fixed
        per-frame latency inflation.
        """

        # FRD body rates: (roll, pitch, yaw).  A positive yaw rate sweeps
        # fixed image features toward image-left; a positive pitch rate
        # sweeps them downward in the effective Rx(pi) image.
        pitch_rate = float(body_rates[1])
        yaw_rate = float(body_rates[2])
        drift_x = -yaw_rate * ROTATION_COMP_FOCAL_NORM * dt
        drift_y = pitch_rate * ROTATION_COMP_FOCAL_NORM * dt
        hypothesis.x_axis.predict(dt, drift=drift_x)
        hypothesis.y_axis.predict(dt, drift=drift_y)
        hypothesis.scale_axis.predict(dt)
        compensation_var = ROTATION_COMP_UNCERTAINTY * (
            abs(drift_x) + abs(drift_y)
        )
        hypothesis.x_axis.inflate(LATENCY_VAR_NORM + compensation_var)
        hypothesis.y_axis.inflate(LATENCY_VAR_NORM + compensation_var)

    def _update_hypothesis(
        self,
        hypothesis: _Hypothesis,
        track: Any,
        now_s: float,
    ) -> None:
        (zx, zy), z_log_scale, stds = _track_measurement(track)
        clipping = getattr(track, "clipping", FrameEdge.NONE)
        if type(clipping) is not FrameEdge:
            clipping = FrameEdge.NONE
        center_censored = bool(getattr(track, "center_censored", False))
        x_censored = (
            center_censored or bool(clipping & (FrameEdge.LEFT | FrameEdge.RIGHT))
        )
        y_censored = (
            center_censored or bool(clipping & (FrameEdge.TOP | FrameEdge.BOTTOM))
        )
        confidence = max(
            MIN_MEAS_CONFIDENCE,
            float(track.confidence)
            * float(getattr(track, "association_confidence", 1.0)),
        )
        # Confidence-weighted measurement noise, not binary authority classes.
        r_x = (stds[0] ** 2) / confidence
        r_y = (stds[1] ** 2) / confidence
        r_scale = (stds[2] ** 2) / confidence
        # A censored axis is unobserved (never a forced-zero "stationary"
        # rate): update observable axes, predict/inflate censored ones.
        if x_censored:
            hypothesis.x_axis.inflate(CENSOR_INFLATE_VAR_NORM)
        else:
            hypothesis.x_axis.update(zx, r_x)
            hypothesis.last_x_measurement_s = float(now_s)
            hypothesis.raw_x = float(zx)
        if y_censored:
            hypothesis.y_axis.inflate(CENSOR_INFLATE_VAR_NORM)
        else:
            hypothesis.y_axis.update(zy, r_y)
            hypothesis.last_y_measurement_s = float(now_s)
            hypothesis.raw_y = float(zy)
        if x_censored or y_censored:
            hypothesis.scale_axis.inflate(CENSOR_INFLATE_VAR_NORM)
        else:
            hypothesis.scale_axis.update(z_log_scale, r_scale)
        hypothesis.confidence = _clamp01(float(track.confidence))
        hypothesis.clipped = clipping is not FrameEdge.NONE
        if hypothesis.clipped:
            # Clipping increases uncertainty; it is not an abort condition.
            hypothesis.x_axis.inflate(CLIPPED_INFLATE_VAR_NORM)
            hypothesis.y_axis.inflate(CLIPPED_INFLATE_VAR_NORM)
        hypothesis.last_measurement_s = float(now_s)
        hypothesis.outer_log_scale = math.log(
            max(1e-6, float(track.apparent_scale))
        )
        if not x_censored:
            # F56: the corridor bound needs the gate's true half-width; a
            # censored-x bbox underreports it, so only uncensored frames
            # refresh the span.
            bbox = getattr(track, "bbox_norm", None)
            if bbox is not None and len(bbox) >= 4:
                hypothesis.outer_half_span_x = 0.5 * (
                    float(bbox[2]) - float(bbox[0])
                )
        # 2026-07-30 entry budget: the CURRENT frame's usable inner aperture
        # (passage_usable) is recorded as half-extents; a missing/unusable
        # fit clears them, so a close-range aperture loss is itself visible
        # to the commit-entry check as "about to go blind".
        aperture = _track_aperture(track)
        if (
            aperture is not None
            and bool(getattr(aperture, "passage_usable", False))
            and getattr(aperture, "half_size_norm", None) is not None
        ):
            hypothesis.aperture_half_x = 0.5 * float(aperture.half_size_norm[0])
            hypothesis.aperture_half_y = 0.5 * float(aperture.half_size_norm[1])
        else:
            hypothesis.aperture_half_x = None
            hypothesis.aperture_half_y = None

    def _refresh_successor(self, tracks: List[Any], now_s: float) -> None:
        current_id = self._current_track_id()
        others = [track for track in tracks if track.track_id != current_id]
        # F49: a newborn suspicious-geometry track (extreme aspect or a
        # wide top-censored slab — ceiling truss, never gate geometry) may
        # not be adopted until it has persisted; the terminal F48 promotion
        # took exactly such a newborn truss over the persistent real gate.
        others = [
            track
            for track in others
            if not (
                _suspicious_adoption_geometry(track)
                and self._track_age_s(track.track_id, now_s)
                < REACQUIRE_MIN_AGE_S
            )
        ]
        if not others:
            if (
                self.successor is not None
                and now_s - self.successor.last_measurement_s > 1.0
            ):
                self.successor = None
            return
        # F40: prefer a successor whose x-axis is observable — a
        # center-censored or LEFT/RIGHT-clipped track is the splinter-
        # fragment geometry that promoted onto a never-measured x-axis.
        eligible = []
        for track in others:
            clipping = getattr(track, "clipping", FrameEdge.NONE)
            if type(clipping) is not FrameEdge:
                clipping = FrameEdge.NONE
            center_censored = bool(getattr(track, "center_censored", False))
            if not center_censored and not bool(
                clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)
            ):
                eligible.append(track)
        # F42: among the x-observable candidates PERSISTENCE outranks
        # confidence — the real gate halves stayed associated for seconds
        # while the higher-confidence debris splinter was newborn every
        # frame.  Rank the age-qualified candidates by age (confidence
        # tie-break); fall back to uncensored-preferred max confidence when
        # nothing has persisted long enough yet.
        aged = [
            track
            for track in eligible
            if self._track_age_s(track.track_id, now_s)
            >= self.config.successor_min_age_s
        ]
        if aged:
            best = max(
                aged,
                key=lambda track: (
                    self._track_age_s(track.track_id, now_s),
                    float(track.confidence),
                ),
            )
        else:
            best = max(
                eligible or others, key=lambda track: float(track.confidence)
            )
        if (
            self.successor is not None
            and self.successor.track_id == best.track_id
        ):
            self._update_hypothesis(self.successor, best, now_s)
        else:
            self.successor = self._hypothesis_from_track(best, now_s)
        self.successor_bearing_cache[self.gate_index] = (
            self.successor.x,
            self.successor.y,
        )

    def _select_search_reacquisition(
        self, tracks: List[Any], now_s: float
    ) -> Optional[Any]:
        """Re-acquisition in SEARCH; the SAME track_id may be re-adopted."""

        if not tracks:
            return None
        if self.current is not None and self.current.track_id is not None:
            same = self._find(tracks, self.current.track_id)
            if same is not None:
                return same
        # F49: newborn suspicious-geometry tracks (ceiling-truss slabs,
        # extreme aspects) are ineligible until they persist — the terminal
        # F48 re-acquisition adopted one over the persistent real gate.
        eligible = [
            track
            for track in tracks
            if not (
                _suspicious_adoption_geometry(track)
                and self._track_age_s(track.track_id, now_s)
                < REACQUIRE_MIN_AGE_S
            )
        ]
        if not eligible:
            return None
        bx, by = self.last_reliable_bearing
        # F42: prefer persistent tracks for the nearest-to-bearing pick —
        # debris is newborn every frame while the real gate stays
        # associated.  Fall back to all tracks when nothing has persisted.
        persistent = [
            track
            for track in eligible
            if self._track_age_s(track.track_id, now_s) >= REACQUIRE_MIN_AGE_S
        ]
        # F74 (20260730T071207Z-visual-course-d38f869e): at the gate plane a
        # one-tick newborn SPLINTER was adopted as the aim the instant the
        # real track went engulfing — the THIRD flight in a row killed this
        # way (F70: 0008 at (+0.44,+0.89); F72: 0011 at (+0.49,+0.55); F73:
        # 0010 at (+0.39,+0.67)), each followed by a blind wander into
        # structure.  A fresh engulfing anchor proves "we are AT the plane",
        # and AT the plane a brand-new track is debris, not the gate: hold
        # SEARCH/PREDICT on the derotated hypothesis for the persistence
        # window instead of adopting.  Same-id re-adoption (above) is
        # untouched, and far-range cold start keeps the newborn fallback.
        near_plane = (
            self._last_engulfing_anchor_s is not None
            and now_s - self._last_engulfing_anchor_s
            <= ENGULFING_ANCHOR_MAX_AGE_S
        )
        candidates = persistent or ([] if near_plane else eligible)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda track: (
                math.hypot(
                    float(track.center_norm[0]) - bx,
                    float(track.center_norm[1]) - by,
                ),
                -float(track.confidence),
            ),
        )

    def _successor_blend(
        self,
        current: _Hypothesis,
        successor: Optional[_Hypothesis],
    ) -> float:
        if successor is None:
            return 0.0
        cfg = self.config
        closure = _clamp01(
            (current.log_scale - cfg.blend_far_log_scale)
            / (cfg.blend_near_log_scale - cfg.blend_far_log_scale)
        )
        trust = _clamp01(successor.confidence) * _clamp01(
            1.0 - successor.position_std / cfg.search_covariance_std_norm
        )
        return cfg.successor_blend_max * closure * trust

    def _wind_vz_center_trim(self, dt: float) -> None:
        """One-sided support-trim integrator (see the VZ_CENTER_TRIM block).

        The caller decides WHEN to run it: qualified TRACK gates on the ey
        active bound (a demanded descent is not a sink to fight); blind
        paths (SEARCH, unqualified TRACK/PREDICT) hold altitude by
        definition, so any sink is unwanted and there is no ey gate (F90).
        A sink winds the trim up (bounded, anti-windup at saturation); a
        real climb leaks it back down.  Never runs while fh-untrusted — a
        biased-regime vz is not evidence — and COAST/COMMIT never call it,
        so the exact-zero wait and the crossing law keep their semantics.
        """
        if self._fh_untrusted or dt <= 0.0:
            return
        cfg = self.config
        saturated = (
            self._collective is not None
            and self._collective >= cfg.max_thrust - 0.005
        )
        if self._vz_est_m_s < cfg.vz_center_trim_sink_m_s and not saturated:
            self._vz_center_trim = _clamp(
                self._vz_center_trim
                + cfg.vz_center_trim_gain * (-self._vz_est_m_s) * dt,
                0.0,
                cfg.vz_center_trim_max,
            )
        elif self._vz_est_m_s > -cfg.vz_center_trim_sink_m_s:
            self._vz_center_trim = max(
                0.0, self._vz_center_trim - cfg.vz_center_trim_leak_s * dt
            )

    def _governed_collective(
        self,
        collective: float,
        support: float,
        gate_y: Optional[float] = None,
        vz_tracked: bool = False,
    ) -> float:
        """IMU climb/descent-rate governor: bound collective by estimated vz.

        Applied wherever a nonzero collective is emitted (TRACK, PREDICT,
        SEARCH, and the defensive fallback) so vision loss never disables
        it; the COAST exact-zero wait and the abort/cleanup zeros bypass it
        by construction.
        Symmetric: caps collective above the climb cap and floors it below
        the descent floor (flight d52adcd4 sank ~-1.9 m/s^2 into a ground
        graze while the frozen frame suppressed SEARCH).  F67: the floor is
        purely proportional (0.21/m/s) — the F64 +0.04 step feedforward
        bang-banged against the ey servo at the plane (F65: thrust
        0.31 <-> 0.22 tick-to-tick as vz_est straddled the floor).
        """

        if self._fh_untrusted:
            # F14: vz_est is frozen and regime-biased, so NO vz-based
            # adjustment may fire — not the climb cap, and above all not
            # the descent floor/feedforward, which pinned F14's thrust at
            # the clamp on a phantom -4.36 m/s sink.  Camera-qualified PD
            # output only, hard-clamped to the envelope.
            # F21 (9828d64c) floor: an invisible "missed" track kept
            # refreshing vertical_qualified from its frozen ghost position,
            # so the qualified-PD path sagged collective 0.318 -> 0.254 over
            # 1.5 s at fh 6.5-7.5 — below real hover in the fast regime — and
            # the drone sank ~2 m into terrain.  While vz/alt are known lies,
            # NOTHING may command below support + margin (the F14-measured
            # biased-regime deficit); honest qualified PD may still command
            # above the floor.  This closes the blind-sink family in one
            # place (F19 SEARCH hold, F20 coast gate, F21 qualified-PD sag)
            # instead of per-path patches.
            # F83 (20260730T113315Z-visual-course-57671d35): the floor's
            # support term is tilt-INFLATED by attitude — at the -0.55
            # pre-cross brake attitude support rose 0.2594 -> 0.2906 and
            # the +0.05 floor pinned MAX thrust (0.34) for the whole 1.3 s
            # latch while the same latch disabled the vz climb governor:
            # an open-loop +2.4 m/s^2 balloon carried the drone OVER gate
            # 1 (no credit; ground collision after the fall).  The F14/F21
            # deficit was measured at ~level attitude, so the floor's
            # support term is bounded at the spawn-level tilt compensation;
            # qualified PD may still command above the floor.
            level_support = self.config.support_collective / max(
                0.85, math.cos(self.config.spawn_pitch_rad)
            )
            floor = (
                min(support, level_support)
                + self.config.fh_untrusted_vertical_margin
            )
            governed = _clamp(
                max(collective, floor),
                self.config.min_thrust,
                self.config.max_thrust,
            )
        else:
            # F68: gate-0 keeps the proved 1.0/0.03 climb-out envelope;
            # gate-1+ legs get the 0.5/0.10 course envelope, SUBTRACTIVE
            # from the PD demand so the arrest is continuous (no cap
            # chatter against the demand).
            # F96: when the unified vz-tracking law computed the collective
            # (gate-1+ qualified fh-trusted TRACK) the climb cap is skipped —
            # it would be a second incoherent vz term on top of the tracker
            # (the F95 limit cycle).  The descent floor stays: it is a
            # one-sided hard lower bound, not feedback.
            if self.gate_index == 0:
                excess = self._vz_est_m_s - VZ_CLIMB_CAP_M_S
                if excess > 0.0:
                    collective = min(
                        collective, support - VZ_GOVERNOR_GAIN * excess
                    )
            elif not vz_tracked:
                course_excess = self._vz_est_m_s - VZ_CLIMB_CAP_COURSE_M_S
                if course_excess > 0.0:
                    collective -= VZ_GOVERNOR_COURSE_GAIN * course_excess
            descent_excess = VZ_DESCENT_FLOOR_M_S - self._vz_est_m_s
            if descent_excess > 0.0:
                collective = max(
                    collective,
                    support + VZ_DESCENT_GOVERNOR_GAIN * descent_excess,
                )
            # Hard clamp here, not only at the main-path call site: the SEARCH
            # and defensive-fallback returns emit the governed value directly,
            # and an unclamped deep-sink floor boost (support + gain*excess +
            # feedforward) exceeded the runner's 0.35 envelope abort in flight
            # 20260729T115619Z-visual-course-039186c8.
            governed = _clamp(
                collective, self.config.min_thrust, self.config.max_thrust
            )
        # F33/F34 brake ceiling band: while the closure governor brakes,
        # the collective is confined to support +/- brake_ceiling_band.
        # F34's hard pin AT support removed all centering authority and the
        # drone crossed at 1.07 m into the bottom bar.
        # F36: the band only makes sense at the crossing — with the F36
        # misalignment brake, _pre_cross_brake_active also fires FAR from
        # the gate, where pinning the collective would kill the high-gate
        # climb that must run while the gate is top-clipped (F35 flew level
        # into the gate-1 lower structure at exactly that geometry).
        # F37 (82306488 + F32 8cc53db2 re-analysis): the band is ONE-SIDED.
        # F32, F34, and F36 ALL died at gate 0 the same way — the gate sat
        # HIGH (ey -0.5..-0.7, top-clipped) through the final approach and
        # the drone never centered vertically.  In F36 the band was the
        # direct cause: it capped the qualified PD's climb at support+0.04
        # from t=0.19 on.  The F32 "climb into structure" the band was
        # built against was misdiagnosed — F32's gate was high too
        # (ey -0.71 at span 0.24).  A gate above image center needs the
        # climb uncapped; the ceiling only applies when the gate is NOT
        # above center (the only geometry where climbing is overshoot).
        # The sink floor applies either way.
        # F77: the band is GATE-0 ONLY — its entire evidence base (F32,
        # F34, F36) is gate-0 crossings, and on gate-1+ legs its lower
        # side capped the closure-excess collective braking at support-0.04
        # (a third of the authority) through F74/F75/F76's hot approaches.
        near_gate = (
            self.current is not None
            and self.current.log_scale >= self.config.closure_min_log_scale
        )
        if (
            self.gate_index == 0
            and self._pre_cross_brake_active
            and near_gate
        ):
            band_lo = max(
                self.config.min_thrust, support - self.config.brake_ceiling_band
            )
            # F50: the main path passes the attitude-compensated gate y;
            # direct callers (tests) fall back to the raw hypothesis y.
            band_gate_y = gate_y
            if band_gate_y is None and self.current is not None:
                band_gate_y = self.current.y
            gate_high = band_gate_y is not None and band_gate_y < -0.10
            if gate_high:
                governed = max(governed, band_lo)
            else:
                governed = _clamp(
                    governed,
                    band_lo,
                    min(
                        self.config.max_thrust,
                        support + self.config.brake_ceiling_band,
                    ),
                )
        # F92: the pre-gate-1 altitude-floor max() that used to apply here
        # is deleted (F78/F78b/F79/F91 balloons); blind anti-sink is the
        # vz-center trim plus the vz descent floor above.
        return governed

    def _anchor_clamped_yaw(
        self, yaw_rate: float, yaw_rad: Optional[float]
    ) -> float:
        """Block steering that winds the heading past the leg anchor (F31).

        Only the outward direction is blocked; return steering is always
        free.  No-op without a live yaw measurement or an anchor.
        """
        if yaw_rad is None or self._course_anchor_yaw_rad is None:
            return yaw_rate
        excursion = math.remainder(
            float(yaw_rad) - self._course_anchor_yaw_rad, 2.0 * math.pi
        )
        cap = self.config.course_heading_anchor_cap_rad
        if excursion > cap:
            return min(yaw_rate, 0.0)
        if excursion < -cap:
            return max(yaw_rate, 0.0)
        return yaw_rate

    def _search_yaw(self, dt: float) -> float:
        cfg = self.config
        self._search_elapsed_s += dt
        self._search_excursion_rad += (
            self._search_direction * cfg.search_yaw_rate_rad_s * dt
        )
        if (
            self._search_elapsed_s >= cfg.search_sweep_period_s
            or abs(self._search_excursion_rad) >= cfg.search_max_excursion_rad
        ):
            self._search_direction *= -1.0
            self._search_elapsed_s = 0.0
            self._search_excursion_rad = 0.0
        return self._search_direction * cfg.search_yaw_rate_rad_s

    def _search_yaw_heading(self, dt: float, yaw_rad: Optional[float]) -> float:
        """Absolute-heading sweep from the search-entry heading (F49).

        ``_search_yaw`` integrates the COMMANDED sweep, so a clamped or
        parked airframe stopped scanning — F40 parked at yaw 1.94 rad, 111
        deg off course, for ~7 blind seconds into gate 1.  F40 swept the
        excursion around the LEG ANCHOR, which re-centered the scan on the
        course heading instead of looking where the target was last seen;
        F49 sweeps from the heading measured at search entry, with the
        first sweep direction still seeded from the last reliable bearing
        (``_enter_search``), so the scan starts toward the lost target.
        The heading error commands the rate, so the scan survives a park.
        Falls back to the legacy incremental sweep without a live yaw
        measurement.
        """

        cfg = self.config
        if yaw_rad is None:
            return self._search_yaw(dt)
        if self._search_base_yaw_rad is None:
            self._search_base_yaw_rad = float(yaw_rad)
        self._search_excursion_rad += (
            self._search_direction * cfg.search_sweep_rate_rad_s * dt
        )
        if abs(self._search_excursion_rad) >= cfg.search_max_excursion_rad:
            self._search_direction *= -1.0
            self._search_excursion_rad = _clamp(
                self._search_excursion_rad,
                -cfg.search_max_excursion_rad,
                cfg.search_max_excursion_rad,
            )
        desired = self._search_base_yaw_rad + self._search_excursion_rad
        return _clamp(
            cfg.search_sweep_gain
            * math.remainder(desired - float(yaw_rad), 2.0 * math.pi),
            -cfg.max_yaw_rate_rad_s,
            cfg.max_yaw_rate_rad_s,
        )

    def _enter_search(self, now_s: float) -> None:
        self.state = CleanCourseState.SEARCH
        # Initialize the real bounded yaw sweep from the last observed
        # target/successor bearing: under the measured 2026-07-29 yaw
        # convention a last image-right bearing is recentered by a POSITIVE
        # yaw, so the sweep starts in that direction first.
        bearing_x = self.last_reliable_bearing[0]
        self._search_direction = 1.0 if bearing_x >= 0.0 else -1.0
        self._search_elapsed_s = 0.0
        self._search_excursion_rad = 0.0
        # F49: the sweep base is re-seeded from the live yaw on the first
        # _search_yaw_heading call of this search, so each search starts
        # scanning from the CURRENT heading, not the leg anchor.
        self._search_base_yaw_rad = None

    def _exit_coast(self) -> None:
        self._coast_zero_sent = False
        self._coast_race_boot_ms = None

    def _slew_roll(
        self,
        target: float,
        dt: float,
        slew_rad_s: Optional[float] = None,
    ) -> float:
        limit = (slew_rad_s or self.config.target_slew_rad_s) * dt
        self._prev_target_roll = _clamp(
            target, self._prev_target_roll - limit, self._prev_target_roll + limit
        )
        return self._prev_target_roll

    def _slew_pitch(
        self,
        target: float,
        dt: float,
        slew_rad_s: Optional[float] = None,
    ) -> float:
        # Optional per-call slew rate: the closure-governor braking regime
        # uses a dedicated faster slew so the brake attitude is actually
        # attained (F12); everything else keeps the transparent target slew.
        # F84: every target stays clear of the runner's -35 deg pitch
        # watchdog (see PITCH_TARGET_MIN_RAD) — one clamp here covers all
        # four call sites (advance, brake, commit-advance, fallback).
        target = max(target, self.config.pitch_target_min_rad)
        limit = (
            slew_rad_s
            if slew_rad_s is not None
            else self.config.target_slew_rad_s
        ) * dt
        self._prev_target_pitch = _clamp(
            target,
            self._prev_target_pitch - limit,
            self._prev_target_pitch + limit,
        )
        return self._prev_target_pitch


def _frame_identity(update: Any) -> Optional[Tuple[Any, Any]]:
    """Underlying camera-frame identity of one tracker update.

    The production token is ``CameraFrameToken``: ``(generation, frame_id)``
    is the camera-frame identity while ``publication_sequence`` strictly
    advances on every republication — including republications of a FROZEN
    frame during a camera stall — so it must never count as freshness.
    Tests use plain ``(stream_id, frame_id)`` tuple tokens.  Returns None
    when no usable identity exists (caller treats the update as fresh).
    """

    token = getattr(update, "token", None) if update is not None else None
    if token is None:
        return None
    generation = getattr(token, "generation", None)
    frame_id = getattr(token, "frame_id", None)
    if generation is not None and frame_id is not None:
        return (generation, frame_id)
    if isinstance(token, (tuple, list)) and len(token) >= 2:
        return (token[0], token[1])
    return None


def _is_engulfing_detection(track: Any) -> bool:
    """True when a track bbox engulfs the camera (gate at the plane).

    See the ENGULFING_BBOX_* constant block: a near-full-frame box is not a
    usable gate measurement.  Conservatively False when the bbox shape is
    missing or malformed.
    """

    bbox = getattr(track, "bbox_norm", None)
    if bbox is None or len(bbox) < 4:
        return False
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    if width <= 0.0 or height <= 0.0:
        return False
    return (
        width >= ENGULFING_BBOX_SPAN_FRACTION
        or height >= ENGULFING_BBOX_SPAN_FRACTION
        or width * height >= ENGULFING_BBOX_AREA_FRACTION
    )


def _suspicious_adoption_geometry(track: Any) -> bool:
    """True when a track bbox geometry is impossible for a gate (F49).

    The terminal F48 promotion adopted a newborn top-censored extreme-
    aspect ceiling truss (span 0.50 x 0.23, aspect ~2.17) over the
    persistent real gate; neither confidence nor the sub-second persistence
    windows could reject it.  Suspicious = extreme aspect ratio, or a wide
    slab censored at the frame TOP (ceiling structure, never a gate).
    Conservatively False when the bbox shape is missing or malformed.
    """

    bbox = getattr(track, "bbox_norm", None)
    if bbox is None or len(bbox) < 4:
        return False
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    if width <= 0.0 or height <= 0.0:
        return False
    aspect = width / height
    if aspect > SUSPICIOUS_ASPECT_MAX or aspect < SUSPICIOUS_ASPECT_MIN:
        return True
    clipping = getattr(track, "clipping", FrameEdge.NONE)
    if type(clipping) is not FrameEdge:
        clipping = FrameEdge.NONE
    return bool(clipping & FrameEdge.TOP) and (
        aspect > SUSPICIOUS_TOP_CENSORED_ASPECT
    )


def _track_visible(track: Any, visible_ids: Any) -> bool:
    """Duck-typed per-track visibility, shared by both acceptance seams.

    A MISSED track (``visible=False``) is a propagated ghost: the tracker
    re-emits its last-associated bbox on every fresh frame, and a
    never-retired authoritative-current track keeps that bbox alive
    indefinitely (flight d5e89c2b: an invisible engulfing ghost refreshed
    the anchor for ~4 s of genuinely fresh frames, so no frame-identity
    freshness gate could ever fire).  Stale content must never anchor.
    """

    visible = getattr(track, "visible", None)
    if visible is None:
        visible_ids = set(visible_ids or ())
        return track.track_id in visible_ids if visible_ids else True
    return bool(visible)


def _engulfing_anchor_track(update: Any, track_id: Optional[str]) -> Optional[Any]:
    """Best engulfing box in one update for bearing/existence anchoring.

    Bearing/existence evidence only: the box never updates any filter axis.
    Prefers the current track_id, then the most confident engulfing box.
    Only tracks the tracker associated on THIS frame (visible) qualify.
    """

    if update is None:
        return None
    visible_ids = getattr(update, "visible_track_ids", ()) or ()
    engulfing = [
        track
        for track in (getattr(update, "tracks", ()) or ())
        if _track_visible(track, visible_ids) and _is_engulfing_detection(track)
    ]
    if not engulfing:
        return None
    if track_id is not None:
        for track in engulfing:
            if track.track_id == track_id:
                return track
    return max(engulfing, key=lambda track: float(getattr(track, "confidence", 0.0)))


def _visible_tracks(update: Any) -> List[Any]:
    """Duck-typed visible-track extraction from one tracker update.

    Degenerate engulfing boxes are dropped here, the single measurement
    acceptance seam, so they can neither update nor be adopted anywhere.
    """

    if update is None:
        return []
    tracks = list(getattr(update, "tracks", ()) or ())
    visible_ids = set(getattr(update, "visible_track_ids", ()) or ())
    result = []
    for track in tracks:
        if _track_visible(track, visible_ids) and not _is_engulfing_detection(track):
            result.append(track)
    return result


def _world_up_accel_m_s2(
    orientation: Any,
    accel: Tuple[float, float, float],
) -> float:
    """World-up acceleration from FRD specific force rotated by attitude.

    ``orientation`` is the estimator quaternion rotating FRD body vectors
    into NED.  Accelerometers measure specific force, so with NED
    down-positive ``a_up = -(R f_b).z - g``; hover therefore yields ~0 at
    any attitude and a +2 m/s^2 climb yields +2.
    """

    w = float(orientation.w)
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    ax, ay, az = (float(value) for value in accel)
    ned_down = (
        2.0 * (x * z - w * y) * ax
        + 2.0 * (y * z + w * x) * ay
        + (1.0 - 2.0 * (x * x + y * y)) * az
    )
    return -ned_down - GRAVITY_M_S2


def _track_aperture(track: Any) -> Any:
    """The track's co-timed inner-aperture fit, or None.

    Live VisualTrack carries the fit on its latest history sample, not
    as a top-level attribute (codex wiring finding): without this the
    aperture branch was dead and every measurement fell back to the
    outer bbox.
    """

    aperture = getattr(track, "inner_aperture", None)
    if aperture is None:
        history = getattr(track, "history", None)
        if history:
            aperture = getattr(history[-1], "inner_aperture", None)
    return aperture


def _track_measurement(
    track: Any,
) -> Tuple[Tuple[float, float], float, Tuple[float, float, float]]:
    """Prefer a valid fitted inner aperture; fall back to the outer bbox.

    Returns ``((x, y), log_scale, (std_x, std_y, std_scale))``.  The outer
    fallback carries larger covariance.  Detector ``estimated_distance`` is a
    placeholder and is never consulted.
    """

    aperture = _track_aperture(track)
    if (
        aperture is not None
        and getattr(aperture, "center_norm", None) is not None
        and getattr(aperture, "log_scale", None) is not None
        and float(getattr(aperture, "confidence", 0.0)) >= APERTURE_MIN_CONFIDENCE
    ):
        stds = getattr(aperture, "measurement_std", None)
        if stds is not None:
            meas_stds = (
                max(1e-3, float(stds[0])),
                max(1e-3, float(stds[1])),
                max(1e-3, float(stds[2])),
            )
        else:
            meas_stds = (OUTER_MEAS_STD_NORM, OUTER_MEAS_STD_NORM, SCALE_MEAS_STD)
        return (
            (float(aperture.center_norm[0]), float(aperture.center_norm[1])),
            float(aperture.log_scale),
            meas_stds,
        )
    center = track.center_norm
    log_scale = math.log(max(1e-6, float(track.apparent_scale)))
    return (
        (float(center[0]), float(center[1])),
        log_scale,
        (OUTER_MEAS_STD_NORM, OUTER_MEAS_STD_NORM, SCALE_MEAS_STD),
    )


# ---------------------------------------------------------------------------
# Runtime seam: one attitude PD, one explicit yaw channel, one final clamp,
# validation, and one atomic race-active send per tick.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanCourseRuntime:
    """Injected runner primitives for the async stage loop."""

    safety_abort_type: type
    monotonic: Callable[[], float]
    sleep: Callable[[float], Awaitable[None]]
    next_control_deadline: Callable[[float, float], float]
    attitude_rate_command: Callable[..., Any]
    attitude_rate_command_type: type
    validate_command: Callable[[Any], None]
    skipped_result: Any
    control_period_s: float
    hard_duration_s: float
    max_yaw_rate_rad_s: float
    max_command_rate_rad_s: float
    min_thrust: float
    max_thrust: float
    # F49: measured level-flight attitude from the GO/start context; every
    # controller pitch offset is relative to this base.
    spawn_pitch_rad: float = SPAWN_PITCH_RAD_DEFAULT


def clamp_final_command(
    command: Any,
    *,
    runtime: CleanCourseRuntime,
) -> Any:
    """The single transparent final clamp applied to every navigation send.

    Roll/pitch body rates are capped at the runner's conservative envelope,
    yaw at the accepted v3 profile production cap, and thrust at the active
    visual-course envelope.  Exact-zero thrust passes through unchanged; it
    is reserved for crossing-coast, abort, and cleanup semantics.
    """

    max_rate = runtime.max_command_rate_rad_s
    max_yaw = runtime.max_yaw_rate_rad_s
    thrust = float(command.thrust)
    if thrust != 0.0:
        thrust = _clamp(thrust, runtime.min_thrust, runtime.max_thrust)
    return runtime.attitude_rate_command_type(
        _clamp(float(command.roll_rate), -max_rate, max_rate),
        _clamp(float(command.pitch_rate), -max_rate, max_rate),
        _clamp(float(command.yaw_rate), -max_yaw, max_yaw),
        thrust,
    )


def _clean_course_tick_trace(
    controller: CleanCourseController,
    update: Any,
    *,
    now_s: float,
    state_entry_s: float,
) -> Dict[str, Any]:
    """One compact per-tick perception/controller snapshot for the recorder.

    Flight 20260729T112603Z-visual-course-d5e89c2b parked ~4 s on an
    invisible engulfing ghost and the trace held only one box, leaving
    gate-1 acquisition unflyable-blind.  Every tick now carries the full
    candidate list with acceptance dispositions, the successor internals,
    the governor estimate and clamp, and the state dwell, merged into the
    runner's existing ``tick`` record (one dict, no new file).
    """

    tracks = []
    visible_ids = set(getattr(update, "visible_track_ids", ()) or ())
    for track in (getattr(update, "tracks", ()) or ()):
        bbox = getattr(track, "bbox_norm", None)
        width = height = None
        if bbox is not None and len(bbox) >= 4:
            width = float(bbox[2]) - float(bbox[0])
            height = float(bbox[3]) - float(bbox[1])
        visible = _track_visible(track, visible_ids)
        engulfing = _is_engulfing_detection(track)
        if engulfing:
            # The d5e89c2b failure mode gets its own category: an invisible
            # (missed, never-retired) engulfing ghost, never an anchor.
            why = "engulfing_anchor" if visible else "engulfing_ghost"
        elif not visible:
            why = "missed"
        elif bool(getattr(track, "center_censored", False)):
            why = "censored"
        else:
            why = None
        tracks.append(
            {
                "id": getattr(track, "track_id", None),
                "center": [float(v) for v in getattr(track, "center_norm", ())],
                "span": [width, height],
                "confidence": float(getattr(track, "confidence", 0.0)),
                "accepted": bool(visible and not engulfing),
                "why_rejected": why,
            }
        )
    successor = controller.successor
    successor_trace = None
    if successor is not None:
        successor_trace = {
            "track_id": successor.track_id,
            "bearing": [successor.x, successor.y],
            "log_scale": successor.log_scale,
            "confidence": successor.confidence,
            "position_std": successor.position_std,
            "age_s": now_s - successor.last_measurement_s,
            "matched": any(
                row["id"] == successor.track_id and row["accepted"] for row in tracks
            ),
        }
    token = getattr(update, "token", None)
    token_trace = None
    if token is not None and hasattr(token, "frame_id"):
        token_trace = [
            getattr(token, "generation", None),
            getattr(token, "frame_id", None),
            getattr(token, "publication_sequence", None),
        ]
    current = controller.current
    return {
        "state": controller.state.value,
        "state_dwell_s": now_s - state_entry_s,
        "token": token_trace,
        "tracks": tracks,
        "successor": successor_trace,
        "vz_est_m_s": controller._vz_est_m_s,
        "thrust_clamp": [controller.config.min_thrust, controller.config.max_thrust],
        "anchor_age_s": (
            None
            if controller._last_engulfing_anchor_s is None
            else now_s - controller._last_engulfing_anchor_s
        ),
        "measurement_gap_s": (
            None if current is None else now_s - current.last_measurement_s
        ),
        "post_credit_brake": False,
        "pre_cross_brake": controller._pre_cross_brake_active,
        "course_anchor_yaw_rad": controller._course_anchor_yaw_rad,
        "alt_est_m": controller._alt_est_m,
        "vz_center_trim": controller._vz_center_trim,
        "fh_mps2": controller._fh_mps2,
        "fh_trusted": not controller._fh_untrusted,
    }


async def run_clean_course_stage(
    host: Any,
    context: Any,
    *,
    runtime: CleanCourseRuntime,
    config: Optional[CleanCourseConfig] = None,
) -> Dict[str, Any]:
    """Run the clean course loop against the duck-typed runner host.

    Retained runner-owned hard boundaries: ``_watchdog(require_target=False)``
    each tick, 50 Hz pacing with missed-tick drop, finite/bounded command
    validation, the atomic race-only send, and one hard attempt timeout.
    ``safe_cleanup`` remains in the runner's ``finally`` path.
    """

    rt = runtime
    if config is None:
        config = CleanCourseConfig(
            min_thrust=rt.min_thrust,
            max_thrust=rt.max_thrust,
            max_yaw_rate_rad_s=rt.max_yaw_rate_rad_s,
            control_period_s=rt.control_period_s,
            spawn_pitch_rad=rt.spawn_pitch_rad,
        )
    controller = CleanCourseController(config)

    host._sample()
    race = host.adapter.race_status
    initial_gate = int(race.active_gate_index) if race is not None else 0
    update = host._visual_latest_tracker_update
    image_half_width = 320.0
    image_half_height = 180.0
    fallback_center = (
        (float(context.initial_gate_x) - image_half_width) / image_half_width,
        (float(context.initial_gate_y) - image_half_height) / image_half_height,
    )
    fallback_scale = math.sqrt(
        max(0, context.initial_gate_area) / (2.0 * image_half_width * image_half_height)
    )
    controller.initialize(
        update,
        gate_index=initial_gate,
        fallback_center_norm=fallback_center,
        fallback_apparent_scale=fallback_scale,
        now_s=rt.monotonic(),
    )

    flight_start = await host._wait_for_next_flight_command_slot()
    hard_deadline = flight_start + rt.hard_duration_s
    next_tick = flight_start
    command_count = 0
    zero_command_count = 0
    last_consumed_token: Any = None
    last_reported_state = controller.state
    trace_state = controller.state
    state_entry_s = controller._course_start_s

    try:
        while True:
            now = rt.monotonic()
            elapsed = now - flight_start
            if now >= hard_deadline:
                raise rt.safety_abort_type(
                    "visual-course hard attempt timeout reached"
                )
            host._sample()
            host._watchdog(require_target=False)
            race = host.adapter.race_status
            if race is not None and bool(race.race_finished):
                break
            if race is not None:
                promoted = controller.note_race(
                    gate_index=int(race.active_gate_index),
                    race_boot_ms=int(race.sim_boot_time_ms),
                    now_s=now,
                )
                if promoted:
                    host.recorder.emit(
                        "clean_course_authoritative_promotion",
                        from_gate_index=controller.transitions[-1][0],
                        to_gate_index=controller.transitions[-1][1],
                        state=controller.state.value,
                        current_track_id=controller._current_track_id(),
                    )
            update = host._visual_latest_tracker_update
            token = getattr(update, "token", None) if update is not None else None
            if update is not None and token is not None and token != last_consumed_token:
                last_consumed_token = token
                estimate = host.estimate
                controller.observe(
                    update,
                    now_s=now,
                    body_rates=(
                        tuple(float(value) for value in estimate.body_rates)
                        if estimate is not None
                        else (0.0, 0.0, 0.0)
                    ),
                )
            if controller.state is not last_reported_state:
                host.recorder.emit(
                    "clean_course_state",
                    previous_state=last_reported_state.value,
                    state=controller.state.value,
                    gate_index=controller.gate_index,
                    elapsed_s=elapsed,
                )
                last_reported_state = controller.state
                trace_state = controller.state
                state_entry_s = now

            estimate = host.estimate
            if estimate is None:
                raise rt.safety_abort_type(
                    "visual-course lost the IMU attitude estimate"
                )
            roll_rad, pitch_rad, yaw_rad = estimate.orientation.to_euler()
            # IMU world-up acceleration for the climb governor: HIGHRES_IMU
            # specific force (FRD) rotated by the estimator quaternion,
            # minus gravity.  Same telemetry seam _record_tick already logs.
            telemetry = getattr(host.adapter, "latest_telemetry", None)
            imu = getattr(telemetry, "imu", None) if telemetry is not None else None
            accel = getattr(imu, "accel", None) if imu is not None else None
            world_up_accel = (
                _world_up_accel_m_s2(estimate.orientation, accel)
                if accel is not None
                else None
            )
            nav = controller.command(
                now_s=now,
                roll_rad=roll_rad,
                pitch_rad=pitch_rad,
                yaw_rad=yaw_rad,
                world_up_accel_m_s2=world_up_accel,
                horizontal_specific_force_mps2=getattr(
                    estimate, "horizontal_specific_force_mps2", None
                ),
            )
            # One attitude PD for roll/pitch; yaw stays an explicit
            # channel.  COAST_FOR_CREDIT bypasses the PD entirely (July-18
            # contract item 9, restored 2026-07-30): the bounded credible-
            # crossing credit wait is EXACT WIRE ZERO on every channel —
            # support-thrust coasting through the PD is out of contract.
            # F26 (e89a3aa2): the brake targets were never attained because
            # the default pitch kp (0.5) uses only ~0.03-0.08 of the 0.25
            # rad/s wire authority — fh grew 2.7 -> 8.7 through the
            # post-credit brake while measured pitch sat at +0.05-0.10.
            # Brake ticks (closure governor) now request the full
            # intercept pitch response (kp 2.0, same as roll; the wire
            # governor stays authoritative); fine tracking keeps the gentle
            # default.
            if controller.state is CleanCourseState.COAST_FOR_CREDIT:
                command = rt.attitude_rate_command_type(0.0, 0.0, 0.0, 0.0)
            else:
                braking = controller._pre_cross_brake_active
                pd_command = rt.attitude_rate_command(
                    estimate,
                    target_roll_rad=nav.target_roll_rad,
                    target_pitch_rad=nav.target_pitch_rad,
                    thrust=nav.thrust,
                    intercept_response_authority=1.0 if braking else 0.0,
                )
                command = rt.attitude_rate_command_type(
                    float(pd_command.roll_rate),
                    float(pd_command.pitch_rate),
                    float(nav.yaw_rate_rad_s),
                    float(pd_command.thrust),
                )
            command = clamp_final_command(command, runtime=rt)
            rt.validate_command(command)
            result = await host._send_flight_command(
                command,
                wire_race_gate_index=controller.gate_index,
            )
            if result is rt.skipped_result:
                # The authoritative race boundary advanced before the wire:
                # skip the obsolete command, sample new state next tick,
                # accept the promotion, and continue.  Never abort on it.
                host.recorder.emit(
                    "clean_course_command_skipped_race_boundary",
                    gate_index=controller.gate_index,
                )
            if command.thrust == 0.0 and command.roll_rate == 0.0:
                zero_command_count += 1
            else:
                command_count += 1
            if controller.state is not trace_state:
                # command() can transition (coast timeout, PREDICT stall cap)
                # after the observe-side report above; keep dwell honest.
                # last_reported_state stays untouched so the next tick still
                # emits the clean_course_state event.
                trace_state = controller.state
                state_entry_s = now
            host._record_tick(
                "visual-course",
                elapsed,
                command,
                extra={
                    "clean_course": _clean_course_tick_trace(
                        controller,
                        update,
                        now_s=now,
                        state_entry_s=state_entry_s,
                    )
                },
            )
            next_tick = rt.next_control_deadline(next_tick, rt.monotonic())
            await rt.sleep(max(0.0, next_tick - rt.monotonic()))
    except BaseException as exc:
        if host._visual_course_summary is None:
            host._visual_course_summary = _course_summary(
                controller,
                host,
                success=False,
                outcome="abort",
                reason=str(exc) or type(exc).__name__,
                race_finished=bool(
                    race is not None and race.race_finished
                ),
                command_count=command_count,
                zero_command_count=zero_command_count,
            )
        raise

    summary = _course_summary(
        controller,
        host,
        success=True,
        outcome="race_finished",
        reason="authoritative race_finished",
        race_finished=True,
        command_count=command_count,
        zero_command_count=zero_command_count,
    )
    host._visual_course_summary = summary
    host.recorder.emit(
        "clean_course_finished",
        final_gate_index=summary["final_gate_index"],
        transitions=len(controller.transitions),
        commands=command_count,
        exact_zero_commands=zero_command_count,
    )
    return summary


def _course_summary(
    controller: CleanCourseController,
    host: Any,
    *,
    success: bool,
    outcome: str,
    reason: str,
    race_finished: bool,
    command_count: int,
    zero_command_count: int,
) -> Dict[str, Any]:
    return {
        "stage": "visual-course",
        "success": bool(success),
        "outcome": outcome,
        "reason": reason,
        "race_finished": bool(race_finished),
        "initial_gate_index": 0,
        "maximum_authoritative_gate_index": int(controller.max_gate_index),
        "final_gate_index": int(controller.gate_index),
        "authoritative_transitions": [
            {"from_gate_index": before, "to_gate_index": after}
            for before, after in controller.transitions
        ],
        "segments": [],
        "visual_navigation_command_count": int(command_count),
        "exact_zero_command_count": int(zero_command_count),
        "yaw_calibration_profile": host.yaw_calibration_profile_evidence,
    }


# --- Runner-facing stage authority (moved from the retired coordinator) ---


@dataclass(frozen=True, slots=True)
class VisualCourseStageLimits:
    """Code-owned bounds for the clean course stage envelope."""

    control_period_s: float = CONTROL_PERIOD_S
    course_hard_duration_s: float = 120.0
    max_command_rate_rad_s: float = 0.25
    max_yaw_rate_rad_s: float = MAX_COURSE_YAW_RATE_RAD_S
    max_measured_yaw_rate_rad_s: float = 0.50
    min_thrust: float = MIN_COURSE_THRUST
    max_thrust: float = MAX_COURSE_THRUST

    def __post_init__(self) -> None:
        numeric = (
            self.control_period_s,
            self.course_hard_duration_s,
            self.max_command_rate_rad_s,
            self.max_yaw_rate_rad_s,
            self.max_measured_yaw_rate_rad_s,
            self.min_thrust,
            self.max_thrust,
        )
        if not all(
            type(value) in {int, float} and math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("visual-course limits must be finite")


DEFAULT_VISUAL_COURSE_LIMITS = VisualCourseStageLimits()


_YAW_PROFILE_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class VisualCourseYawProfile:
    """Module-issued identity of the exact tracked three-run yaw profile."""

    schema: str
    profile_id: str
    profile_sha256: str
    source_commit: str
    plan_id: str
    plan_sha256: str
    controller_to_body_sign: int
    controller_to_image_sign: int
    max_abs_yaw_rate_command_rad_s: float
    max_gyro_response_delay_s: float
    max_first_image_observation_delay_s: float
    max_attitude_excursion_rad: float
    max_abs_measured_yaw_rate_rad_s: float
    observed_max_abs_measured_yaw_rate_rad_s: float
    control_hold_horizon_s: float

    def __init__(
        self,
        *,
        issuer: object,
        schema: str,
        profile_id: str,
        profile_sha256: str,
        source_commit: str,
        plan_id: str,
        plan_sha256: str,
        controller_to_body_sign: int,
        controller_to_image_sign: int,
        max_abs_yaw_rate_command_rad_s: float,
        max_gyro_response_delay_s: float,
        max_first_image_observation_delay_s: float,
        max_attitude_excursion_rad: float,
        max_abs_measured_yaw_rate_rad_s: float,
        observed_max_abs_measured_yaw_rate_rad_s: float,
        control_hold_horizon_s: float,
    ) -> None:
        if issuer is not _YAW_PROFILE_ISSUER:
            raise TypeError(
                "visual-course yaw profiles must come from the tracked loader"
            )
        if (
            schema != YAW_CALIBRATION_PROFILE_SCHEMA
            or profile_id != YAW_CALIBRATION_PROFILE_ID
            or profile_sha256 != YAW_CALIBRATION_PROFILE_SHA256
            or source_commit != YAW_CALIBRATION_SOURCE_COMMIT
            or plan_id != YAW_CALIBRATION_PLAN_ID
            or plan_sha256 != YAW_CALIBRATION_PLAN_SHA256
        ):
            raise ValueError("visual-course yaw profile identity is not frozen")
        if (
            controller_to_body_sign != YAW_CONTROLLER_TO_BODY_SIGN
            or controller_to_image_sign != YAW_CONTROLLER_TO_IMAGE_SIGN
            or max_abs_yaw_rate_command_rad_s
            != YAW_MAX_COMMAND_RATE_RAD_S
            or max_gyro_response_delay_s
            != YAW_MAX_GYRO_RESPONSE_DELAY_S
            or max_first_image_observation_delay_s
            != YAW_MAX_FIRST_IMAGE_OBSERVATION_DELAY_S
            or max_attitude_excursion_rad
            != YAW_MAX_CALIBRATION_ATTITUDE_EXCURSION_RAD
            or max_abs_measured_yaw_rate_rad_s
            != YAW_MAX_CALIBRATION_MEASURED_RATE_RAD_S
            or observed_max_abs_measured_yaw_rate_rad_s
            != YAW_OBSERVED_MAX_MEASURED_RATE_RAD_S
            or control_hold_horizon_s != YAW_CONTROL_HOLD_HORIZON_S
        ):
            raise ValueError("visual-course yaw authority is not frozen")
        for name, value in (
            ("schema", schema),
            ("profile_id", profile_id),
            ("profile_sha256", profile_sha256),
            ("source_commit", source_commit),
            ("plan_id", plan_id),
            ("plan_sha256", plan_sha256),
            ("controller_to_body_sign", controller_to_body_sign),
            ("controller_to_image_sign", controller_to_image_sign),
            (
                "max_abs_yaw_rate_command_rad_s",
                max_abs_yaw_rate_command_rad_s,
            ),
            ("max_gyro_response_delay_s", max_gyro_response_delay_s),
            (
                "max_first_image_observation_delay_s",
                max_first_image_observation_delay_s,
            ),
            ("max_attitude_excursion_rad", max_attitude_excursion_rad),
            (
                "max_abs_measured_yaw_rate_rad_s",
                max_abs_measured_yaw_rate_rad_s,
            ),
            (
                "observed_max_abs_measured_yaw_rate_rad_s",
                observed_max_abs_measured_yaw_rate_rad_s,
            ),
            ("control_hold_horizon_s", control_hold_horizon_s),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def load_tracked(
        cls,
        path: Any = DEFAULT_YAW_CALIBRATION_PROFILE_PATH,
    ) -> "VisualCourseYawProfile":
        """Load and validate the tracked sign-plus-capability authority."""

        profile = load_yaw_calibration_profile(path)
        evidence = yaw_calibration_profile_evidence(profile)
        authority = evidence["authority"]
        return cls(
            issuer=_YAW_PROFILE_ISSUER,
            schema=YAW_CALIBRATION_PROFILE_SCHEMA,
            profile_id=evidence["profile_id"],
            profile_sha256=evidence["sha256"],
            source_commit=evidence["source_commit"],
            plan_id=evidence["plan_id"],
            plan_sha256=evidence["plan_sha256"],
            controller_to_body_sign=authority["controller_to_body_sign"],
            controller_to_image_sign=authority[
                "controller_to_image_sign"
            ],
            max_abs_yaw_rate_command_rad_s=authority[
                "max_abs_yaw_rate_command_rad_s"
            ],
            max_gyro_response_delay_s=authority[
                "max_gyro_response_delay_s"
            ],
            max_first_image_observation_delay_s=authority[
                "max_first_image_observation_delay_s"
            ],
            max_attitude_excursion_rad=authority[
                "max_attitude_excursion_rad"
            ],
            max_abs_measured_yaw_rate_rad_s=authority[
                "max_abs_measured_yaw_rate_rad_s"
            ],
            observed_max_abs_measured_yaw_rate_rad_s=(
                profile["capability"]["max_abs_body_rate_rad_s"]
            ),
            control_hold_horizon_s=authority["control_hold_horizon_s"],
        )

    def to_evidence(self) -> Dict[str, Any]:
        """Match the strict manifest identity emitted by the profile module."""

        return {
            "profile_id": self.profile_id,
            "sha256": self.profile_sha256,
            "source_commit": self.source_commit,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "authority": {
                "controller_to_body_sign": self.controller_to_body_sign,
                "controller_to_image_sign": self.controller_to_image_sign,
                "max_abs_yaw_rate_command_rad_s": (
                    self.max_abs_yaw_rate_command_rad_s
                ),
                "max_gyro_response_delay_s": (
                    self.max_gyro_response_delay_s
                ),
                "max_first_image_observation_delay_s": (
                    self.max_first_image_observation_delay_s
                ),
                "max_attitude_excursion_rad": (
                    self.max_attitude_excursion_rad
                ),
                "max_abs_measured_yaw_rate_rad_s": (
                    self.max_abs_measured_yaw_rate_rad_s
                ),
                "control_hold_horizon_s": self.control_hold_horizon_s,
            },
        }
