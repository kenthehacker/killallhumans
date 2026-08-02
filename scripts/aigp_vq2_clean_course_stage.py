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

- ``PASSAGE_*`` / ``VERTICAL_OPTICAL_*``: expansion-derived time-to-contact,
  de-dilated image motion, and propagated uncertainty define the passage
  trajectory.  IMU vertical velocity is bounded damping, not a visual veto.
- ``YAW_ERROR_SIGN`` / ``ROLL_ERROR_SIGN``: the 2026-07-29 crossing-geometry
  analysis (Q5) falsified the retired controller's lateral direction
  post-credit; see the comments at their definitions.  Magnitudes are the
  proved gate-1-recenter roll gain and the visual-align yaw gain.
- ``GATE0_CLIMB_VERTICAL_OFFSET_NORM``: DISABLED (0.0) after three gate-0
  top-bar strikes showed any positive pre-crossing climb bias is re-climbed
  to and produces unrecoverable overshoot; see the comment at its
  definition.  The closure-scaling machinery remains tested for possible
  post-credit reuse.
- ``VZ_LEAK_TAU_S``: bias guard for the IMU world-vertical-rate estimate.
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
  vibration, so sustained fh > 5.0 freezes the vz/alt integrators without
  selecting a different collective law; the latch releases below fh 2.0.
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

# F49 (20260730 flight-48 measurement): the true hover support in the clean
# pre-credit condition is 0.247, not the 0.275 carried from gate-0 proving —
# at 0.275 every "level" hold climbed ~+0.3 m/s into the top-bar geometry.
SUPPORT_COLLECTIVE = 0.247  # F48-measured hover support collective
VERTICAL_MAX_ABS_ERROR_NORM = 0.50  # GATE0_PROVED_COLLECTIVE_MAX_ABS_ERROR

MIN_COURSE_THRUST = 0.21  # MIN_VISUAL_THRUST (active visual-course envelope)
# Raised 0.32 -> 0.34 (flights 20260729T114842Z-visual-course-039186c8 and
# bf13f18's F10): carrying gate-0 attack speed into the post-credit phase
# collapsed thrust effectiveness (VRS-like fast regime: measured effective
# hover 0.335-0.36 while the low-speed fit predicts +2.9 m/s^2 at 0.32).
# The runner's hard envelope abort stays 0.35 (validate_command), and this
# also restores proportional descent-floor headroom: the vz = -1.0 floor
# target 0.33 was clipped by the old 0.32 clamp.
MAX_COURSE_THRUST = 0.34  # MAX_VISUAL_THRUST, below the 0.35 hard abort

VZ_LEAK_TAU_S = 2.5  # leaky-integrator time constant (bias/noise guard)
GRAVITY_M_S2 = 9.80665  # ImuAttitudeConfig.gravity_mps2
# F100: the F78/F91/F93 vertical arrival arrest is DELETED.  Its last
# scope (gate 0 near the plane, F96) was a second vertical term stacked on
# the same path as the passage controller — the F95
# limit-cycle sin.  The optical passage miss plus bounded IMU damping owns
# collective on every gate.  Provenance:
# F78 (20260730T082159Z-...-7e18243d) climb-through-the-opening, F91
# (20260730T134602Z-...-6e302725) centered-gate balloon, F99
# (20260730T170522Z-...-50b0c982) gate-0 sink into the lower structure.

# Passage motion is controlled in optical space.  A normalized image angle is
# never relabelled as a metric vertical velocity: log-scale expansion supplies
# a bounded time-to-contact, de-dilated image motion predicts plane miss, and
# the collective acts directly on that optical miss.  IMU vz is only a damping
# term and may not reverse a clear visual correction.
PASSAGE_TTC_MIN_S = 0.35
PASSAGE_TTC_MAX_S = 3.00
PASSAGE_MIN_CLOSURE_RATE_S = 0.08
PASSAGE_MOTION_MODEL_STD_NORM = 0.025
PASSAGE_MOTION_FULL_STD_NORM = 0.50
# Aperture scale is a control-quality/approach-energy observable, not the
# outer-box range used by passage admission.  A usable fit can still be a
# geometric outlier (F164 produced three 9-23 sigma collapses around t=1 s),
# so the independent aperture series rejects only statistically impossible
# updates instead of injecting them into either control or admission.
APERTURE_SCALE_INNOVATION_SIGMA = 6.0
VERTICAL_OPTICAL_ERROR_MAX_FAR_NORM = 0.50
VERTICAL_OPTICAL_ERROR_MAX_NEAR_NORM = 0.20
VERTICAL_OPTICAL_COLLECTIVE_GAIN = 0.12
VERTICAL_IMU_DAMPING_GAIN = 0.12
VERTICAL_IMU_MAX_OPPOSITION_FRACTION = 0.50
VERTICAL_CENSORED_AUTHORITY = 0.65
# F163: near the gate plane, a static attitude-compensated bearing is not a
# trajectory direction.  Three distinct image measurements must agree in the
# two complete passage projections and in de-dilated image motion before that
# direction can override projection magnitude uncertainty.  A directional
# clip is already one-sided evidence and therefore owns direction at once.
VERTICAL_DIRECTION_STREAK_FRAMES = 3
# The ordinary 0.25 s collective carry filter consumed F162's remaining
# correction window.  A credible direction change gets a short, bounded slew
# toward its target; all other collective changes retain the ordinary filter.
VERTICAL_DIRECTION_FAST_SLEW_PER_S = 0.12
VERTICAL_DIRECTION_FAST_WINDOW_S = 0.50
COMMIT_ENTRY_SIGMA_MULT = 0.50

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
# F102: the gate-0 scale-triggered "crossing coast" (CROSSING_MAX_ABS_*,
# CROSSING_MEAS_MAX_AGE_S) is deleted — every gate now crosses via the
# energy-budgeted COMMIT, so those bounds have no reader left.
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
# F164 replaces that pointwise conjunction with two explicit observables.
# Outer bbox center/scale has its own derivative state and exclusively owns
# closure/TTC.  A race-gate-owned aperture certificate stores the fitted inner
# opening relative to the co-timed outer support and transports it across a
# bounded fit blackout.  The current center must lie in the conservative core;
# the full fallback/TTC projection hull plus bounded uncertainty must lie in
# the full aperture.  Longitudinal admission is then either a controlled
# approach or a contained point-of-no-return whose predicted visual blackout
# is inside COMMIT_BLACKOUT_S.  Hot closure is no longer a dimensionless scalar
# veto, and missing geometry never becomes crossing authority.
COMMIT_ENTRY_MEAS_MAX_AGE_S = 0.06  # "current frame" at ~30 Hz camera
COMMIT_BLACKOUT_S = 0.50  # measured close-range censorship window 0.3-0.6 s
COMMIT_ENTRY_APERTURE_MARGIN_FRAC = 0.60  # error+drift within 60% of half
# IMU vertical velocity is supporting evidence, never passage authority.  The
# former hard |vz| gate could veto a visually clear correction using a leaky,
# regime-gated integral; the optical miss interval now owns admission while vz
# supplies bounded damping in the collective law.
# Every pitch target stays clear of the runner's MIN_PITCH_RAD (-35 deg)
# watchdog.  F84 (20260730T121408Z-visual-course-533d563c): the F80 course
# brake target (spawn -0.31 + course brake -0.30 = -0.61 rad) sat 0.001 rad
# INSIDE the abort, and the sustained gate-1 misalignment brake slewed
# straight into it ("pitch limit exceeded (-35.0deg)") after a credited
# gate-0 crossing.  -33 deg keeps 2 deg of settling margin.
PITCH_TARGET_MIN_RAD = math.radians(-33.0)
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
# F108 (20260801T054550Z-visual-course-8b530eed): F106's 0.36/s threshold
# put Gate 0 on the fast brake response far from the plane.  F107 then
# accumulated a vertical sink before the near-plane hold (vz -0.11 m/s,
# altitude -0.09 m at span 0.55), pitched to the custody limit, and struck
# the structure without credit.  Restore F104's live-proven 0.60/s response
# ceiling: the range-ramped governor still blends brake continuously, while
# the explicit near-plane budget-false hold still demands full braking.
CLOSURE_FULL_BRAKE_RATE_S = 0.60
# F101 (20260730T173407Z-visual-course-7a862549): a range-flat 0.35
# target permits 3+ m/s at leg start (0.35 log/s at 8-10 m), more than
# custody-compatible attitude braking (~0.4-0.7 m/s^2, capped by the F94
# floor near the plane) can dissipate inside the leg — F100's gate-1 leg
# held pb=1 mid-leg and still ran away to ~1.2 log/s at the plane
# (COMMIT veto, blind structure strike).  The target ramps from a low
# far value (an implicit constant-speed cap while the full gate is
# visible and custody is free) up to the unchanged 0.35 entry budget at
# the commit regime.
CLOSURE_FAR_TARGET_RATE_S = 0.15  # far-range closure target (~1.2 m/s at 8 m)
CLOSURE_FAR_LOG_SCALE = -2.0  # ramp start; 0.35 target at commit_min_log_scale
# F96: the F77 closure-excess collective brake (COURSE_CLOSURE_BRAKE_COLLECTIVE)
# is deleted with its call site — the optical passage law owns collective,
# and the old cut was a fourth incoherent vertical term in the F95 limit cycle.
# Governor trust gate (F33): expansion from a tiny far track is sub-pixel
# noise — post-credit, gate 1 (span 0.03-0.04, log_scale ~-2.9) "grew" at
# 0.9/s and pinned a +0.12 brake with aw_fwd -5 m/s^2 for the whole leg,
# reversing the drone into gate 0's structure.  Below this log_scale the
# expansion rate is untrustworthy and the governor stays out of the loop
# (far-field real closure at our speeds never exceeds the target anyway:
# 4 m/s at 10 m is 0.4/s with scale already >= ~0.08).
CLOSURE_MIN_LOG_SCALE = -2.6
# F99 (20260730T164633Z-visual-course-cb4e1b9e): the closure governor's
# signal is the Kalman scale_axis.v, which lags ~1 s on a fresh/adopted
# track — F98 braked only incidentally (misalignment), the governor never
# saw the true closure, and the leg arrived at expansion +1.7..+3.5 log/s
# through every centered fresh frame (COMMIT's 0.35 entry limit vetoed
# first; gate 0's hot 2.5-3.5 m/s crossings are the same lag).  The raw
# outer-bbox log-scale rate (EMA over uncensored frames) is fast enough;
# the governor takes the FASTER of the two, so braking never goes below
# today's.  Physics: holding the 0.35 log/s cap over a 15 m leg from
# 3 m/s needs ~0.3 m/s^2 — inside the measured 0.4-0.7 brake authority,
# so the signal, not the airframe, was the bottleneck.
OUTER_EXPANSION_TAU_S = 0.20  # EMA time constant for the raw closure rate
OUTER_EXPANSION_MAX_AGE_S = 0.15  # freshness bound on the raw signal
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
PRE_CROSS_BRAKE_PITCH_RAD = -0.15  # one nose-up brake OFFSET from spawn
# TRUE nose-up brake attitude under the verified F38 convention.  F111 moved
# this from -0.15 to -0.20 to reduce inherited Gate-1 closure, but F113 and
# F114 then struck Gate 0's object-1001 structure without credit.  Across the
# live sequence, the last visible outer-top passage margin degraded from
# -0.349/-0.318 at F109/F110's -0.15 to -0.300/-0.290 at F111/F112 and then
# -0.107 in F113.  F115 restores the passage-proven -0.15; inter-gate energy
# remains the course leg's job, never a reason to make Gate 0 marginal.
# (Pre-F38 this was +0.15 absolute — a powered DIVE into the gate; F31's
# "5x too weak brake" was the sign error, not the magnitude.)  F46: -0.15
# absolute (~1.5 m/s^2 + drag) could not kill the approach speed inside the
# last 1.2 s to the gate-1 plane — the drone crossed the threshold still
# fast and 0.37 norm off-center.  F49 restores that deceleration authority
# as an offset: with level flight at -0.31, the effective -0.46 roughly
# doubles it so a misaligned approach actually stops.
PRE_CROSS_BRAKE_SLEW_RAD_S = 1.0  # fast slew while the governor brakes
# F125 removes the Gate-1-only doubled brake.  F124's otherwise improved
# handoff switched from the Gate-0 brake to -0.30 at promotion, pitched the
# camera to -0.578, and drove Gate 1 from raw y +0.128 to +0.600 before the
# custody floor could react.  One continuous -0.15 brake reference now owns
# every leg; authoritative promotion cannot change pitch mode under the same
# closure/alignment evidence.
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
# The IMU altitude estimate bounds the low-altitude desired-sink taper.
ALT_EST_MIN_M = -2.0  # biased-integrator clamp on the altitude estimate
# The fh regime gate protects only IMU integration.  It never selects a
# different collective law or adds a thrust margin.
FH_UNTRUSTED_TRIGGER_MPS2 = 5.0  # biased regime above this horizontal force
FH_TRUSTED_RELEASE_MPS2 = 2.0  # hysteresis release below this
FH_UNTRUSTED_SUSTAIN_S = 0.3  # transients shorter than this never latch
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

# F120 continuous turn reference.  F116-F119 split lateral guidance into a
# current-gate correction, a binary/off-full successor preview, a separate
# prebank, and a pending-credit yaw overlay.  F119 alternated +/-0.15 rad/s
# through the crossing and reached Gate-0 credit with essentially zero net
# yaw.  One filtered reference now blends passage alignment and the retained
# successor hypothesis.  The hypothesis is IMU-derotated in _predict(); its
# confidence, covariance, measurement age, range ordering, closure, and the
# filtered current-aperture reserve all reduce authority continuously.  F127
# carries their product continuously across tracker-id reassociation without
# restoring the deleted persistence/age qualification.  The derotated reference
# remains the sole command-bearing state.
TURN_REFERENCE_TAU_S = 0.15
BLEND_FAR_LOG_SCALE = -1.6  # below this the successor gets no blend
BLEND_NEAR_LOG_SCALE = -0.9  # at this apparent scale closure reaches one
SUCCESSOR_MIN_LOG_SCALE_GAP = 0.25  # successor must remain visibly farther
SUCCESSOR_PREVIEW_PROJECTION_S = 0.10
SUCCESSOR_TURN_MAX_STD_NORM = 0.60  # zero turn authority at this uncertainty
# F42 (20260729T201743Z-visual-course-1e24b6d2): confidence provably cannot
# separate the real gate from detector debris — a bottom-left splinter
# out-confidenced the real gate-1 halves (0.62-0.71 vs 0.42-0.54) and was
# adopted at promotion.  PERSISTENCE can: debris is newborn every frame
# while the real gate halves stayed associated for seconds.  Successor
# ranking and re-acquisition prefer track age; F124 removes the duplicate age
# gate from turn authority after selection has already admitted a hypothesis.
SUCCESSOR_MIN_AGE_S = 0.5
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
# file (VERTICAL_PITCH_COMP_NORM_PER_RAD = 1.6) -- at 1.0 every predicted
# bearing under-rotated by 37.5%, which is why the frozen hypothesis
# lagged the true gate bearing in F52 (frozen ex -0.156 vs true -0.48) and
# why stale-bearing steering under-corrects.  F140's x=0.9 discriminator
# regressed the live handoff, so F141 returns to the F127 baseline here.
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


@dataclass(frozen=True)
class _PassageMotion:
    """Uncertain one-axis optical trajectory to the current gate plane."""

    bearing_error: float
    physical_rate_norm_s: float
    closure_rate_s: float
    closure_std_s: float
    ttc_s: float
    ttc_std_s: float
    projection_authority: float
    fallback_intercept_error: float
    optical_intercept_error: float
    intercept_error: float
    bearing_std: float
    intercept_std: float
    freshness_authority: float
    measurement_authority: float
    control_authority: float
    directional_censor: FrameEdge = FrameEdge.NONE


@dataclass
class _ApertureCorridorCertificate:
    """Inner opening geometry expressed against one outer-track observation.

    The certificate is deliberately not another target tracker.  It records
    the aperture-to-outer-box geometry from one passage-usable camera frame so
    that the opening can be transported by the separately filtered outer
    track during a short fit blackout.  ``gate_index`` is assigned only while
    the hypothesis is the race-owned current target; successor vision may
    prepare geometry, but cannot authorize a crossing.
    """

    track_id: Optional[str]
    gate_index: Optional[int]
    frame_identity: Optional[Tuple[Any, Any]]
    source_s: float
    aperture_center_x: float
    aperture_center_y: float
    aperture_half_x: float
    aperture_half_y: float
    offset_ratio_x: float
    offset_ratio_y: float
    half_ratio_x: float
    half_ratio_y: float
    center_std_x_norm: float
    center_std_y_norm: float


@dataclass(frozen=True)
class _TransportedCorridor:
    track_id: Optional[str]
    gate_index: int
    frame_identity: Optional[Tuple[Any, Any]]
    source_age_s: float
    center_x: float
    center_y: float
    half_x: float
    half_y: float
    center_std_x: float
    center_std_y: float
    live: bool


@dataclass(frozen=True)
class _CommitAdmission:
    """Inspectable result of the single passage-admission decision."""

    admissible: bool
    status: str
    corridor_known: bool = False
    corridor_live: bool = False
    corridor_age_s: Optional[float] = None
    x_tube: Optional[float] = None
    y_tube: Optional[float] = None
    x_budget: Optional[float] = None
    y_budget: Optional[float] = None
    closure_rate_s: Optional[float] = None
    closure_agreement: Optional[float] = None
    ttc_s: Optional[float] = None
    longitudinal_reachable: bool = False


class _Hypothesis:
    """Retained current/successor target hypothesis with its small filter."""

    __slots__ = (
        "track_id",
        "x_axis",
        "y_axis",
        "outer_x_axis",
        "outer_y_axis",
        "outer_scale_axis",
        "scale_axis",
        "passage_source",
        "confidence",
        "outer_log_scale",
        "outer_log_scale_s",
        "outer_expansion_rate",
        "outer_half_span_x",
        "outer_half_span_y",
        "outer_raw_x",
        "outer_raw_y",
        "clipped",
        "clipping_edges",
        "vertical_censor_edge",
        "vertical_censor_bound",
        "horizontal_censor_edge",
        "horizontal_censor_bound",
        "created_s",
        "last_measurement_s",
        "last_outer_x_measurement_s",
        "last_outer_y_measurement_s",
        "last_outer_x_evidence_s",
        "last_outer_y_evidence_s",
        "last_x_measurement_s",
        "last_y_measurement_s",
        "last_aperture_scale_measurement_s",
        "raw_x",
        "raw_y",
        "aperture_half_x",
        "aperture_half_y",
        "corridor_certificate",
    )

    def __init__(
        self,
        *,
        track_id: Optional[str],
        x: float,
        y: float,
        log_scale: float,
        outer_x: Optional[float] = None,
        outer_y: Optional[float] = None,
        outer_log_scale: Optional[float] = None,
        passage_source: str = "outer",
        confidence: float,
        pos_var: float,
        now_s: float,
    ) -> None:
        self.track_id = track_id
        self.x_axis = _AxisFilter(x, 0.0, pos_var, INITIAL_RATE_VAR)
        self.y_axis = _AxisFilter(y, 0.0, pos_var, INITIAL_RATE_VAR)
        outer_x_value = float(x if outer_x is None else outer_x)
        outer_y_value = float(y if outer_y is None else outer_y)
        outer_scale_value = float(
            log_scale if outer_log_scale is None else outer_log_scale
        )
        self.outer_x_axis = _AxisFilter(
            outer_x_value, 0.0, pos_var, INITIAL_RATE_VAR
        )
        self.outer_y_axis = _AxisFilter(
            outer_y_value, 0.0, pos_var, INITIAL_RATE_VAR
        )
        # Admission range/closure is an outer-box observable.  It must never
        # ingest inner-aperture scale (F162/F163's modality jump manufactured
        # a ~2/s closure spike precisely when the aperture disappeared).
        self.outer_scale_axis = _AxisFilter(
            outer_scale_value, 0.0, pos_var, INITIAL_RATE_VAR
        )
        # Approach energy and passage motion retain their own aperture-scale
        # series.  It is updated only by statistically credible usable
        # aperture fits and is never fed an outer-box fallback.  ``log_scale``
        # remains this control-only value for the focused helper/test seam.
        self.scale_axis = _AxisFilter(
            float(log_scale), 0.0, pos_var, INITIAL_RATE_VAR
        )
        self.passage_source = str(passage_source)
        self.confidence = _clamp01(confidence)
        self.outer_log_scale = outer_scale_value
        self.outer_log_scale_s = float(now_s)
        self.outer_expansion_rate = 0.0
        # ``bbox_norm`` spans [0, 1] while center/aperture coordinates span
        # [-1, 1].  A bbox width is therefore already a half-extent in the
        # center coordinate system (0.5*w in unit space, multiplied by two).
        self.outer_half_span_x = math.exp(outer_scale_value)
        self.outer_half_span_y = self.outer_half_span_x
        self.outer_raw_x = outer_x_value
        self.outer_raw_y = outer_y_value
        self.clipped = False
        self.clipping_edges = FrameEdge.NONE
        # Directional censorship is still one-sided geometry.  In
        # particular, a same-id gate leaving through the frame BOTTOM says
        # the gate is low; losing the numeric y measurement must not turn
        # that evidence into a renewed climb command (F115).
        self.vertical_censor_edge = FrameEdge.NONE
        self.vertical_censor_bound: Optional[float] = None
        self.horizontal_censor_edge = FrameEdge.NONE
        self.horizontal_censor_bound: Optional[float] = None
        self.created_s = float(now_s)
        self.last_measurement_s = float(now_s)
        self.last_outer_x_measurement_s = float(now_s)
        self.last_outer_y_measurement_s = float(now_s)
        self.last_outer_x_evidence_s = float(now_s)
        self.last_outer_y_evidence_s = float(now_s)
        self.last_x_measurement_s = float(now_s)
        self.last_y_measurement_s = float(now_s)
        self.last_aperture_scale_measurement_s = (
            float(now_s) if passage_source == "aperture" else NEVER_MEASURED_S
        )
        # Last passage-coordinate sample per axis and the CURRENT frame's live
        # inner-aperture half-extents.  The persistent, gate-owned certificate
        # is separate and therefore survives a bounded missing-fit interval.
        self.raw_x = float(x)
        self.raw_y = float(y)
        self.aperture_half_x: Optional[float] = None
        self.aperture_half_y: Optional[float] = None
        self.corridor_certificate: Optional[
            _ApertureCorridorCertificate
        ] = None

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

    @property
    def outer_position_std(self) -> float:
        return math.hypot(self.outer_x_axis.std, self.outer_y_axis.std)

    @property
    def outer_filtered_log_scale(self) -> float:
        return self.outer_scale_axis.p

    @property
    def outer_filtered_expansion_rate(self) -> float:
        return self.outer_scale_axis.v


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

    support_collective: float = SUPPORT_COLLECTIVE
    vertical_max_abs_error_norm: float = VERTICAL_MAX_ABS_ERROR_NORM
    passage_ttc_min_s: float = PASSAGE_TTC_MIN_S
    passage_ttc_max_s: float = PASSAGE_TTC_MAX_S
    passage_min_closure_rate_s: float = PASSAGE_MIN_CLOSURE_RATE_S
    passage_motion_model_std_norm: float = PASSAGE_MOTION_MODEL_STD_NORM
    passage_motion_full_std_norm: float = PASSAGE_MOTION_FULL_STD_NORM
    aperture_scale_innovation_sigma: float = APERTURE_SCALE_INNOVATION_SIGMA
    vertical_optical_error_max_far_norm: float = VERTICAL_OPTICAL_ERROR_MAX_FAR_NORM
    vertical_optical_error_max_near_norm: float = VERTICAL_OPTICAL_ERROR_MAX_NEAR_NORM
    vertical_optical_collective_gain: float = VERTICAL_OPTICAL_COLLECTIVE_GAIN
    vertical_imu_damping_gain: float = VERTICAL_IMU_DAMPING_GAIN
    vertical_imu_max_opposition_fraction: float = (
        VERTICAL_IMU_MAX_OPPOSITION_FRACTION
    )
    vertical_censored_authority: float = VERTICAL_CENSORED_AUTHORITY
    vertical_direction_streak_frames: int = VERTICAL_DIRECTION_STREAK_FRAMES
    vertical_direction_fast_slew_per_s: float = (
        VERTICAL_DIRECTION_FAST_SLEW_PER_S
    )
    vertical_direction_fast_window_s: float = VERTICAL_DIRECTION_FAST_WINDOW_S
    commit_entry_sigma_mult: float = COMMIT_ENTRY_SIGMA_MULT
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
    commit_sustain_s: float = COMMIT_SUSTAIN_S
    commit_timeout_s: float = COMMIT_TIMEOUT_S
    commit_min_log_scale: float = COMMIT_MIN_LOG_SCALE
    commit_advance_pitch_rad: float = COMMIT_ADVANCE_PITCH_RAD
    commit_entry_meas_max_age_s: float = COMMIT_ENTRY_MEAS_MAX_AGE_S
    commit_blackout_s: float = COMMIT_BLACKOUT_S
    commit_entry_aperture_margin_frac: float = COMMIT_ENTRY_APERTURE_MARGIN_FRAC
    pitch_target_min_rad: float = PITCH_TARGET_MIN_RAD
    near_plane_steer_gain_mult: float = NEAR_PLANE_STEER_GAIN_MULT
    predict_frame_gap_s: float = PREDICT_FRAME_GAP_S
    predict_max_gap_s: float = PREDICT_MAX_GAP_S
    x_steer_max_age_s: float = X_STEER_MAX_AGE_S
    closure_target_rate_s: float = CLOSURE_TARGET_RATE_S
    closure_full_brake_rate_s: float = CLOSURE_FULL_BRAKE_RATE_S
    closure_far_target_rate_s: float = CLOSURE_FAR_TARGET_RATE_S
    closure_far_log_scale: float = CLOSURE_FAR_LOG_SCALE
    closure_min_log_scale: float = CLOSURE_MIN_LOG_SCALE
    outer_expansion_tau_s: float = OUTER_EXPANSION_TAU_S
    outer_expansion_max_age_s: float = OUTER_EXPANSION_MAX_AGE_S
    fragment_advance_min_log_scale: float = FRAGMENT_ADVANCE_MIN_LOG_SCALE
    fragment_creep_pitch_rad: float = FRAGMENT_CREEP_PITCH_RAD
    pre_cross_brake_pitch_rad: float = PRE_CROSS_BRAKE_PITCH_RAD
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
    vertical_pitch_comp_norm_per_rad: float = VERTICAL_PITCH_COMP_NORM_PER_RAD
    search_covariance_std_norm: float = SEARCH_COVARIANCE_STD_NORM
    search_yaw_rate_rad_s: float = SEARCH_YAW_RATE_RAD_S
    search_sweep_period_s: float = SEARCH_SWEEP_PERIOD_S
    search_max_excursion_rad: float = SEARCH_MAX_EXCURSION_RAD
    search_sweep_rate_rad_s: float = SEARCH_SWEEP_RATE_RAD_S
    search_sweep_gain: float = SEARCH_SWEEP_GAIN
    pending_credit_hold_s: float = PENDING_CREDIT_HOLD_S
    turn_reference_tau_s: float = TURN_REFERENCE_TAU_S
    blend_far_log_scale: float = BLEND_FAR_LOG_SCALE
    blend_near_log_scale: float = BLEND_NEAR_LOG_SCALE
    successor_min_log_scale_gap: float = SUCCESSOR_MIN_LOG_SCALE_GAP
    successor_preview_projection_s: float = SUCCESSOR_PREVIEW_PROJECTION_S
    successor_turn_max_std_norm: float = SUCCESSOR_TURN_MAX_STD_NORM
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
    """Five-state selector/estimator/control-law owner for one course run."""

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
        self._last_commit_admission = _CommitAdmission(
            False, "not-evaluated"
        )
        # Underlying camera-frame identity of the last consumed update; a
        # republished frozen frame (same identity) is never fresh evidence.
        self._last_frame_identity: Optional[Tuple[Any, Any]] = None
        # Explicit fresh-camera observation ownership for the current track.
        # Tracker republishes may advance hypothesis timestamps, so the
        # near-plane direction streak must not infer freshness from those
        # timestamps alone.
        self._current_fresh_observation_track_id: Optional[str] = None
        self._current_fresh_observation_s: Optional[float] = None
        self._current_fresh_y_observation_s: Optional[float] = None
        self._current_fresh_y_observation_serial = 0
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
        # IMU altitude estimate (m), seeded at the takeoff pad.  It is trace
        # evidence only: a leaky integral may support diagnosis but cannot
        # veto a clear optical passage correction.
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
        # F120 continuous lateral reference.  The reference itself carries
        # through crossing and authoritative promotion; it is derotated by
        # measured yaw between command ticks, then filtered toward the latest
        # current/successor evidence.  No raw image bearing is latched.
        self._turn_reference_x: Optional[float] = None
        self._turn_reference_yaw_rad: Optional[float] = None
        self._turn_aperture_reserve = 0.0
        self._turn_successor_authority = 0.0
        self._successor_heading_blend = 0.0
        self._successor_heading_error_norm: Optional[float] = None
        # Camera heading and physical interception are separate states.  Yaw
        # keeps the target in the FOV; roll follows the de-dilated optical
        # plane miss, so yaw-centering cannot unwind a still-needed bank.
        self._lateral_intercept_reference_x: Optional[float] = None
        self._last_lateral_motion: Optional[_PassageMotion] = None
        self._last_vertical_motion: Optional[_PassageMotion] = None
        # Near-plane vertical direction is a trajectory state, deliberately
        # separate from the attitude-compensated image bearing.  The streak is
        # advanced only by distinct y measurements, never by 50 Hz command
        # ticks replaying one camera frame.  TOP/BOTTOM censorship bypasses
        # the streak because it is direct one-sided evidence.
        self._vertical_direction_track_id: Optional[str] = None
        self._vertical_direction_last_y_observation_serial = 0
        self._vertical_direction_streak_sign = 0
        self._vertical_direction_streak = 0
        self._vertical_direction_sign = 0
        self._vertical_direction_supported = False
        self._vertical_direction_source: Optional[str] = None
        self._vertical_direction_fast_until_s: Optional[float] = None
        self._vertical_direction_edge_active = False
        self._vertical_direction_magnitude = 0.0
        # Trace the terms at their real controller boundary.  These are
        # diagnostics only and never become a second collective owner.
        self._last_vertical_support: Optional[float] = None
        self._last_vertical_visual_delta = 0.0
        self._last_vertical_imu_delta = 0.0
        self._last_vertical_collective_target: Optional[float] = None
        self._far_vertical_direction_sign = 0
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
        self._turn_reference_x = None
        self._turn_reference_yaw_rad = None
        self._turn_aperture_reserve = 0.0
        self._turn_successor_authority = 0.0
        self._lateral_intercept_reference_x = None
        self._last_lateral_motion = None
        self._last_vertical_motion = None
        self._reset_vertical_direction()
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
            self.current = self._hypothesis_from_track(
                current_track, now_s, gate_index=self.gate_index
            )
            self.state = CleanCourseState.TRACK
            self._set_reliable_bearing(self.current.x, self.current.y)
            self._seed_current_fresh_observation(self.current, now_s=now_s)
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
                before_y_s = self.current.last_y_measurement_s
                self._update_hypothesis(self.current, resumed, now_s)
                self._record_current_fresh_observation(
                    self.current,
                    now_s=now_s,
                    fresh=fresh,
                    previous_y_measurement_s=before_y_s,
                )
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
                before_y_s = self.current.last_y_measurement_s
                self._update_hypothesis(self.current, match, now_s)
                self._record_current_fresh_observation(
                    self.current,
                    now_s=now_s,
                    fresh=fresh,
                    previous_y_measurement_s=before_y_s,
                )
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
            before_y_s = self.current.last_y_measurement_s
            self._update_hypothesis(self.current, match, now_s)
            self._record_current_fresh_observation(
                self.current,
                now_s=now_s,
                fresh=fresh,
                previous_y_measurement_s=before_y_s,
            )
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
                self.current = self._hypothesis_from_track(
                    adopted, now_s, gate_index=self.gate_index
                )
                if fresh:
                    self._seed_current_fresh_observation(
                        self.current, now_s=now_s
                    )
                else:
                    self._clear_current_fresh_observation()
                # After an authoritative increment can no longer promote a
                # marginal cached successor directly, SEARCH is allowed to
                # qualify and adopt that now-current gate.  Remove its stale
                # successor label so one track id cannot own both roles.
                if (
                    self.successor is not None
                    and self.successor.track_id == adopted.track_id
                ):
                    self.successor = None
                self.state = CleanCourseState.TRACK
        else:
            gap = now_s - self.current.last_measurement_s
            # F102: the gate-0 credible-close-loss coast is DELETED — every
            # gate now crosses through the aperture/energy-budgeted COMMIT
            # (ONE final-approach/crossing policy, the standing contract).
            # The hot scale-triggered coast crossed gate 0 at ~1.0 log/s
            # (~2.4 m/s) every flight, and the next leg inherited that
            # energy: F100 ran away to ~1.2 log/s at the gate-1 plane
            # (structure strike); F101 lost the gate at the bottom edge
            # under the brake and VRS-sank into the ground (id 1002).  A
            # gate-0 close loss without an armed COMMIT is NOT a credible
            # crossing and falls through to PREDICT/re-center, exactly
            # like gate-1+ legs.
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
        # Directional passage evidence belongs to the race-owned gate that
        # produced it.  Never carry a prior gate's near-plane reversal into
        # the newly authoritative leg.
        self._reset_vertical_direction()
        self._clear_current_fresh_observation()
        self._near_plane_since_s = None
        self._commit_entry_s = None
        self._last_commit_admission = _CommitAdmission(
            False, "authoritative-gate-change"
        )
        # Preserve the filtered bearing through promotion, but start the new
        # gate's future-successor aperture calculation from zero.  The newly
        # promoted current hypothesis is the same derotated bearing that fed
        # the pre-credit reference, so no control overlay changes sign here.
        self._turn_aperture_reserve = 0.0
        self._turn_successor_authority = 0.0
        # Engulfing release is evidence about the gate just credited.  Carrying
        # F164's Gate-0 anchor into Gate 1 granted the new successor a false
        # preturn lease before Gate 1 had any safe corridor of its own.
        self._last_engulfing_anchor_s = None
        self._last_engulfing_anchor_identity = None
        if self.state is CleanCourseState.COAST_FOR_CREDIT:
            self._exit_coast()
        # An authoritative increment settles the pending-credit hold.
        self._pending_credit_until_s = None

        successor = self.successor
        if (
            successor is not None
            # No image-x measurement means there is no bearing to derotate.
            # Every measured successor, however uncertain or short-lived,
            # remains the same probabilistic hypothesis after authoritative
            # race promotion; normal prediction and association own its
            # growing uncertainty instead of a second binary admission gate.
            and successor.last_x_measurement_s > NEVER_MEASURED_S + 1.0
        ):
            self.current = successor
            self.successor = None
            # Successor vision may prepare a keyed geometric certificate, but
            # this authoritative race event is the only operation that turns
            # it into current-gate passage authority.
            if self.current.corridor_certificate is not None:
                self.current.corridor_certificate.gate_index = self.gate_index
            self.state = (
                CleanCourseState.TRACK
                if now_s - self.current.last_measurement_s
                <= self.config.predict_frame_gap_s
                else CleanCourseState.PREDICT
            )
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
        # The physical intercept is gate-owned path state, not camera heading
        # memory.  Seed it from the newly race-owned current hypothesis instead
        # of carrying the previous gate's bank request across promotion.
        self._lateral_intercept_reference_x = (
            self._horizontal_control_observable(self.current, now_s)[0]
            if self.current is not None
            else None
        )
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
        # Directional support is re-earned from this tick's live passage
        # evidence.  The established sign/streak persists across command
        # ticks, but cannot silently authorize SEARCH or stale vision.
        self._vertical_direction_supported = False
        self._vertical_direction_source = None
        self._vertical_direction_magnitude = 0.0
        self._last_vertical_support = None
        self._last_vertical_visual_delta = 0.0
        self._last_vertical_imu_delta = 0.0
        self._last_vertical_collective_target = None
        self._far_vertical_direction_sign = 0
        self._pre_cross_brake_active = False  # main path recomputes below
        self._successor_heading_blend = 0.0
        self._successor_heading_error_norm = None
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
        # level both formulas agree (0.2594).
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

        # F53 near-plane COMMIT (see the COMMIT_* constant block): the
        # misalignment brake self-locks short of the plane, so a sustained,
        # aligned, freshly measured close regime commits to an inertial
        # crossing.  F102: gate-agnostic.  Gate 0's old scale-triggered hot
        # coast (credible-close-loss -> COAST at ~2.4 m/s, no energy budget)
        # inherited runaway closure into every subsequent leg — F99/F100/
        # F101 all died on energy the gate-0 crossing created.  There is ONE
        # crossing policy: every gate crosses via this energy-budgeted
        # COMMIT, gate 0 included.  F54: proximity arms at
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
        commit_admission = (
            self._commit_admission(now_s, pitch_rad, cfg)
            if near_plane_close
            else _CommitAdmission(False, "outside-proximity")
        )
        if not near_plane_close and self.state is not CleanCourseState.COMMIT:
            self._last_commit_admission = commit_admission
        # F164 unified entry model (see the COMMIT_ENTRY_* block): one outer
        # closure/TTC state plus one gate-owned aperture certificate.  A
        # contained trajectory commits from controlled closure or at the
        # bounded visual point-of-no-return; otherwise TRACK continues
        # braking/re-centering outside the censorship blackout.
        if (
            near_plane_close
            and now_s - self._near_plane_since_s >= cfg.commit_sustain_s
            and commit_admission.admissible
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
                commit_steer_gain = self._course_steer_gain(self.current)
                commit_current_heading, _axis, _age, commit_heading_edge = (
                    self._horizontal_control_observable(self.current, now_s)
                )
                commit_heading_ex, commit_blend = self._turn_reference(
                    self.current,
                    self.successor,
                    current_error=commit_current_heading,
                    now_s=now_s,
                    yaw_rad=yaw_rad,
                    dt=dt,
                )
                commit_intercept_ex = self._lateral_intercept_reference(
                    self.current,
                    self.successor,
                    successor_authority=commit_blend,
                    now_s=now_s,
                    dt=dt,
                )
                commit_roll, commit_yaw = self._coordinated_turn_request(
                    commit_heading_ex,
                    steer_gain=commit_steer_gain,
                    yaw_rad=yaw_rad,
                    intercept_x=commit_intercept_ex,
                )
                commit_vertical_qualified = (
                    now_s - self.current.last_y_measurement_s
                    <= cfg.vertical_qualify_max_age_s
                    and self.current.y_axis.std
                    <= cfg.search_covariance_std_norm
                    and not (
                        self.current.vertical_censor_edge
                        & (FrameEdge.TOP | FrameEdge.BOTTOM)
                    )
                )
                commit_ey = self._compensated_ey(
                    self.current.y, pitch_rad
                )
                commit_vertical_motion = self._vertical_passage_motion(
                    self.current,
                    commit_ey,
                    now_s=now_s,
                    vertical_qualified=commit_vertical_qualified,
                    pitch_rad=pitch_rad,
                )
                commit_hold = self._vertical_collective_target(
                    self.current,
                    support,
                    commit_vertical_motion,
                )
                commit_target = self._governed_collective(
                    commit_hold,
                    support,
                    gate_y=(commit_ey if commit_vertical_qualified else None),
                )
                # F66: the F60 vertical-aim pitch term is DELETED.  In
                # commit the attitude is the forward drive, not a second
                # vertical channel — the aim and the collective servo read
                # the same ey and fought at the plane: the dive rotated the
                # camera down (growing ey -> more dive) while the body kept
                # its pre-dive velocity vector.  F63 dove UNDER gate 1;
                # F65 (11b13f53) slammed the top panel 0.47 s after entry
                # with the opening 0.45 norm below the aim.  Vertical
                # translation now lives only in the shared vertical-rate
                # owner above.
                return NavigationOutput(
                    target_roll_rad=self._slew_roll(
                        commit_roll,
                        dt,
                        directional_censor=commit_heading_edge,
                    ),
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
                    thrust=self._continuous_collective(
                        commit_target, dt, now_s=now_s
                    ),
                    state=self.state,
                    gate_index=self.gate_index,
                    successor_blend=commit_blend,
                    current_track_id=self._current_track_id(),
                    successor_track_id=self._successor_track_id(),
                )

        if self.state is CleanCourseState.SEARCH:
            search_hold = self._vertical_collective_target(
                self.current,
                support,
                None,
            )
            search_target = self._governed_collective(search_hold, support)
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
                pending_roll = 0.0
                pending_yaw = 0.0
                pending_blend = 0.0
                if self.current is not None:
                    pending_current_heading, _axis, _age, _edge = (
                        self._horizontal_control_observable(
                            self.current, now_s
                        )
                    )
                    pending_reference, pending_blend = self._turn_reference(
                        self.current,
                        self.successor,
                        current_error=pending_current_heading,
                        now_s=now_s,
                        yaw_rad=yaw_rad,
                        dt=dt,
                    )
                    pending_steer_gain = self._course_steer_gain(self.current)
                    pending_roll, pending_yaw = self._coordinated_turn_request(
                        pending_reference,
                        steer_gain=pending_steer_gain,
                        yaw_rad=yaw_rad,
                        intercept_x=self._lateral_intercept_reference(
                            self.current,
                            self.successor,
                            successor_authority=pending_blend,
                            now_s=now_s,
                            dt=dt,
                        ),
                    )
                    # Authoritative credit is still pending: successor
                    # bearing may keep the camera from losing the next gate,
                    # but the exact-zero-forward hold does not begin a new
                    # lateral interception before ownership advances.
                    pending_roll = 0.0
                    self._prev_target_roll = 0.0
                return NavigationOutput(
                    target_roll_rad=pending_roll,
                    target_pitch_rad=self._slew_pitch(
                        cfg.spawn_pitch_rad + cfg.brake_pitch_rad, dt
                    ),
                    yaw_rate_rad_s=pending_yaw,
                    thrust=self._continuous_collective(
                        search_target, dt, now_s=now_s
                    ),
                    state=self.state,
                    gate_index=self.gate_index,
                    successor_blend=pending_blend,
                    current_track_id=self._current_track_id(),
                    successor_track_id=self._successor_track_id(),
                )
            # F49: absolute-heading sweep from the search-entry heading,
            # first toward the last reliable bearing — the F40 anchor-
            # centered sweep re-centered the scan on the course heading
            # instead of where the target was last seen.
            sweep_yaw = self._search_yaw_heading(dt, yaw_rad)
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
                thrust=self._continuous_collective(
                    search_target, dt, now_s=now_s
                ),
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
            fallback_hold = self._vertical_collective_target(
                None,
                support,
                None,
            )
            fallback_target = self._governed_collective(fallback_hold, support)
            return NavigationOutput(
                target_roll_rad=0.0,
                target_pitch_rad=cfg.spawn_pitch_rad + cfg.brake_pitch_rad,
                yaw_rate_rad_s=0.0,
                thrust=self._continuous_collective(
                    fallback_target, dt, now_s=now_s
                ),
                state=self.state,
                gate_index=self.gate_index,
            )

        # F116 separates HEADING from PHYSICAL INTERCEPT.  Successor preview
        # never changes pitch, thrust, passage, or race ownership; current-gate
        # ``ex``/``ey`` remain authoritative, with F118 adding only a small
        # aperture-leased prebank to the current roll correction.
        # Historical context (flight
        # ab6252b2): track 07 (gate 1) was centered and approached to span
        # 0.34/conf 0.87, then slid left to x=-0.95 while the yaw command
        # sat at ~0 — the blend toward a far successor at x=+0.6 cancelled
        # the pursuit error exactly when the close gate escaped.  F116 makes
        # that displaced geometry ineligible and keeps successor influence
        # out of every translational channel.
        blend = 0.0
        ex, heading_axis, heading_age_s, heading_censor = (
            self._horizontal_control_observable(current, now_s)
        )
        heading_ex = ex
        ey = current.y
        # F50: the VERTICAL channel servos on the pitch-attitude-compensated
        # error (nose-up brake attitude reads the world LOW in frame; see
        # the VERTICAL_PITCH_COMP_NORM_PER_RAD block).  F117 uses that same
        # physical error for the forward misalignment brake.  F116 showed the
        # old RAW-ey coupling was positive feedback: braking pitched the
        # camera up, moved Gate 0 down in-frame, and the camera artifact asked
        # for still more brake until the vehicle met the top structure.
        ey_vertical = self._compensated_ey(ey, pitch_rad)
        # F40 (20260729T193134Z-visual-course-63ed6342): never steer on an
        # x-axis without a fresh accepted measurement — an unmeasured or
        # stale x (edge-clipped splinter, censored axis) is a garbage aim
        # point.  The y/vertical path is deliberately untouched (F35 servos
        # on censored-y by design).  F52: the near-plane regime is excepted
        # at the zeroing site below — there the derotated hypothesis is the
        # best aim evidence and the crossing completes in <1 s.
        x_qualified = bool(
            heading_age_s <= cfg.x_steer_max_age_s
            and (
                heading_censor != FrameEdge.NONE
                or heading_axis.std <= cfg.search_covariance_std_norm
            )
        )
        # F149: F148 reached x=-0.134, then the retired TRACK trim had
        # integrated -0.050 and reduced the race-owned error to -0.084.  Its
        # own contract deliberately drove a sustained nonzero image error to
        # zero yaw, opposing the only coordinated turn reference.  Current x
        # now enters that reference directly; no lateral integrator, bias, or
        # second command owner remains.

        # F120: one lateral reference owns yaw and bank before, through, and
        # after the crossing.  Passage alignment and the IMU-derotated
        # successor bearing are blended continuously; no preturn overlay can
        # countermand the current controller or be countermanded by it.
        heading_ex, blend = self._turn_reference(
            current,
            self.successor,
            current_error=ex,
            now_s=now_s,
            yaw_rad=yaw_rad,
            dt=dt,
        )
        intercept_ex = self._lateral_intercept_reference(
            current,
            self.successor,
            successor_authority=blend,
            now_s=now_s,
            dt=dt,
        )

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
        # gate is at/below center (ey >= 0).  Exact measurements use
        # de-dilated optical motion to predict the plane intercept.  TOP/BOTTOM
        # censorship retains a one-sided correction; fully stale evidence
        # drops visual authority and leaves bounded IMU damping around support.
        vertical_setpoint_offset = 0.0
        if self.gate_index == 0:
            span = (
                cfg.gate0_climb_reference_log_scale - cfg.crossing_min_log_scale
            )
            closure = (
                _clamp01(
                    (current.outer_log_scale - cfg.crossing_min_log_scale)
                    / span
                )
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
            and not (
                current.vertical_censor_edge
                & (FrameEdge.TOP | FrameEdge.BOTTOM)
            )
        )
        floor_gate_y = ey_vertical if vertical_qualified else None
        # Vision closure-rate governor (F31, see the CLOSURE_* constant
        # block): the filtered log-scale rate is the only honest closure
        # signal — fh is a signless drag magnitude that conflates speed with
        # braking.  Speed is capped CONTINUOUSLY at every range: the pitch
        # target below blends from the advance law toward the gentle brake
        # attitude as the expansion rate rises past the target.  Applies in
        # TRACK and PREDICT alike (the SEARCH path returned above).
        # F33 trust gate: expansion from tiny far tracks is sub-pixel noise,
        # so the governor only runs above CLOSURE_MIN_LOG_SCALE.
        # F99: the Kalman rate lags ~1 s on a fresh/adopted track (see the
        # OUTER_EXPANSION_* block) — the governor only braked incidentally
        # and every leg arrived hot.  Take the FASTER of the filtered rate
        # and the raw outer-bbox rate (fresh only): braking never goes
        # below what the filtered signal alone would command.
        raw_closure = (
            current.outer_expansion_rate
            if now_s - current.last_measurement_s
            <= cfg.outer_expansion_max_age_s
            else 0.0
        )
        closure_rate = 0.0
        if current.outer_log_scale >= cfg.closure_min_log_scale:
            # F114 (20260801T063210Z-visual-course-9260728c): only the
            # failed Gate-0 run produced a full-brake pulse at t=0.047...
            # 0.110 s while its raw outer box was exactly stationary
            # (outer EMA 0.0).  The fresh hypothesis's lagging scale rate
            # invoked the high-authority brake loop during launch, moved
            # pitch -0.31 -> -0.33, and the approach stayed above the
            # aperture into the top structure.  During the bounded Gate-0
            # launch boost, use the fast raw outer-box signal alone; real
            # raw closure and the independent misalignment brake remain
            # fully authoritative.  After the boost, resume the normal
            # faster-of-two governor.
            launch_scale_warmup = (
                self.gate_index == 0
                and self._course_start_s is not None
                and now_s - self._course_start_s
                < cfg.launch_boost_duration_s
            )
            if launch_scale_warmup:
                closure_rate = raw_closure
            else:
                closure_rate, _agreement, _std = (
                    self._control_closure_estimate(current, now_s)
                )
        # F101 approach-energy profile (20260730T173407Z-...-7a862549):
        # F100's gate-1 leg braked mid-leg (pb=1, pitch to -0.57) yet the
        # closure held 0.43 log/s and then RAN AWAY to ~1.2 log/s at the
        # plane once the F94 custody floor capped the brake attitude —
        # attitude braking authority (~0.4-0.7 m/s^2, custody-limited to
        # less near the plane) cannot kill 2.5-3.5 m/s inside the leg
        # remainder, so the COMMIT expansion budget vetoed and the blind
        # drone hit the gate-1 structure.  A constant 0.35 log/s target
        # only bounds speed NEAR the plane; far out it permits 3+ m/s.
        # The target now RAMPS with range: a low far target
        # (constant-speed cap in log terms) bleeding energy while the
        # full gate is visible and custody is free, rising to the
        # unchanged 0.35 entry budget at the commit regime.  Same law on
        # every leg — gate-0 entry energy is what the next leg inherits.
        target_frac = _clamp01(
            (current.outer_log_scale - cfg.closure_far_log_scale)
            / (cfg.commit_min_log_scale - cfg.closure_far_log_scale)
        )
        closure_target = cfg.closure_far_target_rate_s + (
            cfg.closure_target_rate_s - cfg.closure_far_target_rate_s
        ) * target_frac
        closure_brake = _clamp01(
            (closure_rate - closure_target)
            / (cfg.closure_full_brake_rate_s - closure_target)
        )
        # Misalignment brake (F35, d25f23fe): a fully misaligned gate only
        # suppressed ADVANCE, leaving the pitch law at brake_pitch (near
        # level, still creeping forward) — the gate-1 leg held yaw at the
        # cap with pitch ~level while fh grew 4 -> 7.3 into gate-1-area
        # structure.  Speed with no alignment is pure risk: blend toward
        # the TRUE brake attitude with the same signal that suppresses
        # advance.
        angular_error = math.hypot(ex, ey_vertical)
        align = _clamp01(1.0 - angular_error / cfg.angular_full_brake_norm)
        brake_demand = max(closure_brake, 1.0 - align)
        pre_cross_brake = brake_demand > 0.5
        self._pre_cross_brake_active = pre_cross_brake
        vertical_motion = self._vertical_passage_motion(
            current,
            ey_vertical - vertical_setpoint_offset,
            now_s=now_s,
            vertical_qualified=vertical_qualified,
            pitch_rad=pitch_rad,
        )
        self._far_vertical_direction_sign = self._far_outer_vertical_direction(
            current,
            now_s=now_s,
            pitch_rad=pitch_rad,
        )
        collective = self._vertical_collective_target(
            current,
            support,
            vertical_motion,
            supporting_direction_sign=self._far_vertical_direction_sign,
        )
        if vertical_motion is not None:
            floor_gate_y = vertical_motion.bearing_error

        # F96: the F77 closure-excess collective brake is DELETED.  It
        # only fired on the gate-1+ qualified TRACK path — exactly the
        # path the optical passage law now owns — where its min()
        # cut was a fourth incoherent vertical term in the F95 limit cycle.
        # Its original evidence (F74/F75/F76: "attitude-only braking
        # never decelerated") was measured WITH the F95 support
        # inflation biasing the brake attitude +0.9 m/s^2 climb; forward
        # braking stays with the closure-governor pitch law above, and
        # vertical passage is the optical controller's job.

        # F100: the gate-0 near-plane vertical energy arrest is DELETED —
        # the optical passage law now owns every visually qualified TRACK
        # path, and a second vertical term here was the F95 limit-cycle sin.
        # Image motion and expansion predict the aperture intercept instead
        # of relabelling an image angle as a metric velocity setpoint.

        # Gate-0 takeoff boost is feedforward only; it never changes the
        # closed-loop vertical sign.
        if (
            self.gate_index == 0
            and self._course_start_s is not None
            and now_s - self._course_start_s < cfg.launch_boost_duration_s
        ):
            collective = cfg.launch_boost_thrust
        collective_target = self._governed_collective(
            collective,
            support,
            gate_y=floor_gate_y,
        )
        thrust = self._continuous_collective(
            collective_target, dt, now_s=now_s
        )

        # Lateral: per the 2026-07-29 crossing-geometry analysis, positive
        # image-x error requires POSITIVE yaw (negative yaw rotates the
        # camera left and pushes a right-side target further right) and a
        # coordinated positive bank toward the target.  Both signs are
        # one-line flippable named constants pending first-flight
        # confirmation.  Clipping no longer saturates corrective steering
        # (codex, flights 4480d0a6/ab6252b2): the clip penalty halves yaw
        # exactly when the target is escaping at the frame edge.
        # F150: the F57 off/full gain switch arrived too late to center Gate 1
        # and defeated the continuous turn-reference filter with a command
        # step.  TRACK, COMMIT, and the credit wait now read the same existing
        # outer-range ramp.  No new state or authority owner is introduced.
        steer_gain = self._course_steer_gain(current)
        target_roll, yaw_rate = self._coordinated_turn_request(
            heading_ex,
            steer_gain=steer_gain,
            yaw_rad=yaw_rad,
            intercept_x=intercept_ex,
        )
        if (
            not x_qualified
            and blend <= 0.0
            and current.outer_log_scale < cfg.near_brake_log_scale
        ):
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
            - max(0.0, closure_rate - cfg.expansion_brake_free_s)
            / cfg.expansion_brake_span_s
        )
        near_plane = _clamp01(
            (cfg.near_brake_log_scale - current.outer_log_scale)
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
        # censorship: on ANY near-plane TRACK with a false entry budget (F102:
        # gate-agnostic — gate 0 holds outside the blackout too), cut the
        # advance law and demand the full brake — hold OUTSIDE the blackout
        # and re-center.  The same budget passing arms COMMIT on the same
        # tick; nothing else about the crossing changes.
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
            and current.outer_log_scale >= cfg.commit_min_log_scale
            and not commit_admission.admissible
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
        aperture_scale_fresh = bool(
            current.passage_source == "aperture"
            and now_s - current.last_aperture_scale_measurement_s
            <= cfg.outer_expansion_max_age_s
        )
        # Before an aperture has ever been certified, a whole fresh outer box
        # remains legitimate fragment-size evidence.  Once aperture control
        # has taken ownership, its expiry must not silently hand this control
        # decision back to the admission/range observable.
        fragment_control_log_scale = (
            current.log_scale
            if aperture_scale_fresh
            else (
                current.outer_log_scale
                if current.passage_source == "outer"
                else None
            )
        )
        if (
            fragment_control_log_scale is None
            or fragment_control_log_scale
            < cfg.fragment_advance_min_log_scale
        ):
            law_pitch = min(
                law_pitch, cfg.spawn_pitch_rad + cfg.fragment_creep_pitch_rad
            )
        # F125: promotion does not select a deeper course-specific brake.
        # Closure and alignment continuously set brake demand, while this one
        # reference and the one custody floor remain unchanged across legs.
        target_pitch = law_pitch + brake_demand * (
            (cfg.spawn_pitch_rad + cfg.pre_cross_brake_pitch_rad) - law_pitch
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
        commit_regime = current.outer_log_scale >= cfg.commit_min_log_scale
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
                    if abs(intercept_ex) > cfg.roll_pursuit_fast_ex_norm
                    else None
                ),
                directional_censor=heading_censor,
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
        """Return the structured corridor/TTC admission result as a bool."""

        return self._commit_admission(now_s, pitch_rad, cfg).admissible

    def _commit_admission(
        self, now_s: float, pitch_rad: float, cfg: "CleanCourseConfig"
    ) -> _CommitAdmission:
        """Evaluate one gate-owned trajectory tube and longitudinal model.

        Outer-box observations alone own closure/TTC; a separately transported
        aperture certificate owns geometry.  The current center must sit in the
        conservative core, while both complete optical projection endpoints
        plus bounded model/transport uncertainty must fit in the full opening.
        Fast closure is not categorically rejected: a contained approach may
        commit at the modeled censorship point-of-no-return.
        """

        current = self.current
        if current is None:
            result = _CommitAdmission(False, "no-current")
            self._last_commit_admission = result
            return result

        corridor = self._transported_corridor(current, now_s=now_s)
        certificate = current.corridor_certificate
        if corridor is None and certificate is None:
            # Compatibility for focused direct-state unit fixtures.  The live
            # reachability proof below enters through public observations and
            # always uses a real certificate.
            if (
                current.aperture_half_x is not None
                and current.aperture_half_y is not None
                and now_s - current.last_x_measurement_s
                <= cfg.commit_entry_meas_max_age_s
                and now_s - current.last_y_measurement_s
                <= cfg.commit_entry_meas_max_age_s
            ):
                corridor = _TransportedCorridor(
                    track_id=current.track_id,
                    gate_index=self.gate_index,
                    frame_identity=None,
                    source_age_s=0.0,
                    center_x=current.raw_x,
                    center_y=current.raw_y,
                    half_x=current.aperture_half_x,
                    half_y=current.aperture_half_y,
                    center_std_x=current.x_axis.std,
                    center_std_y=current.y_axis.std,
                    live=True,
                )
        if corridor is None:
            status = (
                "corridor-owner-mismatch"
                if certificate is not None
                and certificate.gate_index != self.gate_index
                else "corridor-unknown-or-expired"
            )
            result = _CommitAdmission(False, status)
            self._last_commit_admission = result
            return result

        common = {
            "corridor_known": True,
            "corridor_live": corridor.live,
            "corridor_age_s": corridor.source_age_s,
        }
        fresh_outer_frame = bool(
            self._current_fresh_observation_track_id == current.track_id
            and self._current_fresh_observation_s is not None
            and now_s - self._current_fresh_observation_s
            <= cfg.commit_entry_meas_max_age_s
        )
        direct_fixture = corridor.frame_identity is None and certificate is None
        if not (fresh_outer_frame or direct_fixture):
            result = _CommitAdmission(False, "stale-outer-frame", **common)
            self._last_commit_admission = result
            return result
        if current.clipping_edges != FrameEdge.NONE:
            result = _CommitAdmission(False, "directionally-censored", **common)
            self._last_commit_admission = result
            return result

        x_motion = self._passage_motion(
            current,
            current.x_axis,
            current.x,
            now_s=now_s,
            measurement_age_s=now_s - current.last_x_measurement_s,
            admission_closure=True,
        )
        compensated_center_y = self._compensated_ey(
            corridor.center_y, pitch_rad
        )
        y_motion = self._passage_motion(
            current,
            current.y_axis,
            self._compensated_ey(current.y, pitch_rad),
            now_s=now_s,
            measurement_age_s=now_s - current.last_y_measurement_s,
            admission_closure=True,
        )
        x_budget = cfg.commit_entry_aperture_margin_frac * corridor.half_x
        y_budget = cfg.commit_entry_aperture_margin_frac * corridor.half_y
        x_projection = max(
            abs(x_motion.fallback_intercept_error),
            abs(x_motion.optical_intercept_error),
        )
        y_projection = max(
            abs(y_motion.fallback_intercept_error),
            abs(y_motion.optical_intercept_error),
        )
        x_tube = x_projection + cfg.commit_entry_sigma_mult * (
            x_motion.bearing_std
            + cfg.passage_motion_model_std_norm
            + corridor.center_std_x
        )
        y_tube = y_projection + cfg.commit_entry_sigma_mult * (
            y_motion.bearing_std
            + cfg.passage_motion_model_std_norm
            + corridor.center_std_y
        )
        if (
            abs(corridor.center_x) > x_budget
            or abs(compensated_center_y) > y_budget
            or x_tube > corridor.half_x
            or y_tube > corridor.half_y
        ):
            result = _CommitAdmission(
                False,
                "corridor-known/not-contained",
                x_tube=x_tube,
                y_tube=y_tube,
                x_budget=x_budget,
                y_budget=y_budget,
                **common,
            )
            self._last_commit_admission = result
            return result

        closure, agreement, closure_std_s = self._outer_closure_estimate(
            current, now_s
        )
        if closure >= cfg.passage_min_closure_rate_s:
            raw_ttc_s = 1.0 / closure
            time_to_blackout_s = max(
                0.0, cfg.near_brake_log_scale - current.outer_log_scale
            ) / closure
            imminent_blackout = bool(
                cfg.passage_ttc_min_s
                <= raw_ttc_s
                <= cfg.commit_timeout_s
                and time_to_blackout_s <= cfg.commit_blackout_s
            )
            ttc_s: Optional[float] = raw_ttc_s
        else:
            imminent_blackout = False
            ttc_s = None
        controlled_approach = (
            closure + cfg.commit_entry_sigma_mult * closure_std_s
            <= cfg.closure_target_rate_s
        )
        longitudinal_reachable = controlled_approach or imminent_blackout
        result = _CommitAdmission(
            longitudinal_reachable,
            "admissible" if longitudinal_reachable else "longitudinal-hold",
            x_tube=x_tube,
            y_tube=y_tube,
            x_budget=x_budget,
            y_budget=y_budget,
            closure_rate_s=closure,
            closure_agreement=agreement,
            ttc_s=ttc_s,
            longitudinal_reachable=longitudinal_reachable,
            **common,
        )
        self._last_commit_admission = result
        return result

    def _robust_closure_rate(
        self, current: _Hypothesis, now_s: float
    ) -> Tuple[float, float]:
        """Compatibility seam for the control-only passage closure."""

        closure, agreement, _std = self._control_closure_estimate(
            current, now_s
        )
        return closure, agreement

    def _outer_closure_estimate(
        self, current: _Hypothesis, now_s: float
    ) -> Tuple[float, float, float]:
        """Truthful outer-only closure for admission and range safety."""

        cfg = self.config
        # ``scale_axis`` is initialized from and continuously mirrors the
        # outer measurement until a usable inner aperture is first observed.
        # Keep that pre-aperture compatibility seam explicit: a number of
        # controller-boundary replays construct an outer-only hypothesis and
        # then advance its legacy scale state directly.  In live operation the
        # two axes are identical in this state; after an aperture certificate
        # exists, admission unconditionally reads the independent outer axis.
        direct_outer_only = bool(
            current.passage_source == "outer"
            and current.corridor_certificate is None
        )
        outer_axis = (
            current.scale_axis
            if direct_outer_only
            else current.outer_scale_axis
        )
        filtered = max(0.0, float(outer_axis.v))
        raw_fresh = (
            now_s - current.last_measurement_s <= cfg.outer_expansion_max_age_s
        )
        raw = max(0.0, float(current.outer_expansion_rate)) if raw_fresh else 0.0
        closure = max(filtered, raw)
        closure_std = math.sqrt(max(0.0, outer_axis.vv))
        if closure < cfg.passage_min_closure_rate_s:
            return closure, 0.0, closure_std
        disagreement = abs(filtered - raw) if raw_fresh else closure
        agreement = _clamp01(
            1.0 - disagreement / max(cfg.closure_full_brake_rate_s, closure)
        )
        return closure, agreement, max(closure_std, disagreement)

    def _control_closure_estimate(
        self, current: _Hypothesis, now_s: float
    ) -> Tuple[float, float, float]:
        """Independent approach-energy closure, never a modality-mixed state.

        Fresh aperture expansion is a leading cue for pitch/trajectory control;
        outer expansion is the continuous fallback.  Their rates may be fused
        conservatively, but their absolute log scales never enter one filter.
        Missing or rejected aperture fits therefore cannot create F163's
        outer/aperture jump, while F164's honest outer admission state cannot
        silently delete the early Gate-0 brake.
        """

        outer, outer_agreement, outer_std = self._outer_closure_estimate(
            current, now_s
        )
        control_scale_fresh = bool(
            (
                current.passage_source == "aperture"
                and now_s - current.last_aperture_scale_measurement_s
                <= self.config.outer_expansion_max_age_s
            )
            or (
                current.passage_source == "outer"
                and now_s - current.last_measurement_s
                <= self.config.outer_expansion_max_age_s
            )
        )
        aperture = (
            max(0.0, float(current.expansion_rate))
            if control_scale_fresh
            else 0.0
        )
        if not control_scale_fresh:
            return outer, outer_agreement, outer_std
        aperture_std = math.sqrt(max(0.0, current.scale_axis.vv))
        closure = max(outer, aperture)
        disagreement = abs(aperture - outer)
        agreement = _clamp01(
            1.0
            - disagreement
            / max(self.config.closure_full_brake_rate_s, closure)
        )
        return closure, min(outer_agreement, agreement), max(
            outer_std, aperture_std, disagreement
        )

    def _passage_motion(
        self,
        current: _Hypothesis,
        axis: _AxisFilter,
        bearing_error: float,
        *,
        now_s: float,
        measurement_age_s: float,
        directional_censor: FrameEdge = FrameEdge.NONE,
        admission_closure: bool = False,
    ) -> _PassageMotion:
        """Predict one uncertain optical-axis miss at the gate plane.

        With log-scale expansion ``lambda`` and a de-rotated image rate,
        ``q = image_rate - lambda * bearing`` removes pure perspective
        dilation.  ``bearing + q / lambda`` is the constant-velocity plane
        miss in current-depth units.  Authority transfers from a short image-
        motion projection to that complete bounded-TTC model only as fresh
        position and consistent closure evidence reduce uncertainty; weak
        closure cannot enter de-dilation at full strength.  Position, image-
        rate, expansion-rate, and model-disagreement uncertainty all propagate
        into the intercept interval used by control and COMMIT admission.
        """

        cfg = self.config
        error = float(bearing_error)
        closure, closure_agreement, closure_std_s = (
            self._outer_closure_estimate(current, now_s)
            if admission_closure
            else self._control_closure_estimate(current, now_s)
        )
        closure_credible = closure >= cfg.passage_min_closure_rate_s
        model_closure = closure if closure_credible else 0.0
        if closure_credible:
            ttc_s = _clamp(
                1.0 / closure,
                cfg.passage_ttc_min_s,
                cfg.passage_ttc_max_s,
            )
            ttc_std_s = closure_std_s / max(1e-6, closure * closure)
        else:
            ttc_s = cfg.commit_blackout_s
            ttc_std_s = 0.0
        image_rate = float(axis.v)
        physical_rate = image_rate - model_closure * error
        freshness = _clamp01(
            1.0 - max(0.0, float(measurement_age_s)) / cfg.predict_max_gap_s
        )
        position_certainty = _clamp01(
            1.0 - axis.std / cfg.search_covariance_std_norm
        )
        projection_authority = (
            freshness * position_certainty * closure_agreement
        )
        # Blend complete models, never just their horizons.  With weak or
        # contradictory closure evidence, the short-horizon fallback is
        # ``p + p_dot*T0`` and does not consume lambda at all.  Only credible
        # closure transfers authority to the TTC/de-dilated model.  F161's
        # newborn Gate-1 raw expansion spike (7.26/s versus filtered 0.22/s)
        # otherwise entered ``q`` at full strength despite 3% agreement and
        # flipped both vertical and lateral intercept signs.
        fallback_horizon_s = cfg.commit_blackout_s
        fallback_intercept = error + image_rate * fallback_horizon_s
        optical_intercept = error + physical_rate * ttc_s
        intercept_error = fallback_intercept + projection_authority * (
            optical_intercept - fallback_intercept
        )
        # The blended intercept is linear in p and p_dot for fixed lambda:
        #   h = (1-A)*(p + T0*p_dot) + A*((1-lambda*T1)*p + T1*p_dot)
        # Propagate that actual model, including the position/rate covariance.
        position_gain = 1.0 - (
            projection_authority * model_closure * ttc_s
        )
        rate_gain = fallback_horizon_s + projection_authority * (
            ttc_s - fallback_horizon_s
        )
        projected_variance = (
            position_gain * position_gain * axis.pp
            + 2.0 * position_gain * rate_gain * axis.pv
            + rate_gain * rate_gain * axis.vv
            + cfg.passage_motion_model_std_norm**2
        )
        model_disagreement = optical_intercept - fallback_intercept
        projected_variance += (
            projection_authority
            * (1.0 - projection_authority)
            * model_disagreement
        ) ** 2
        if closure_credible:
            unclamped_ttc_s = 1.0 / closure
            ttc_gradient = (
                -1.0 / (closure * closure)
                if cfg.passage_ttc_min_s
                < unclamped_ttc_s
                < cfg.passage_ttc_max_s
                else 0.0
            )
            closure_sensitivity = projection_authority * (
                -error * ttc_s + physical_rate * ttc_gradient
            )
            projected_variance += (
                closure_sensitivity * closure_std_s
            ) ** 2
        intercept_std = math.sqrt(max(0.0, projected_variance))
        if directional_censor != FrameEdge.NONE:
            # A directional clip is inequality evidence whose sign remains
            # valid while the frame is fresh; covariance cannot turn BOTTOM
            # into climb or TOP into descent.  Confidence and age still fade
            # its bounded authority.
            measurement_authority = (
                cfg.vertical_censored_authority
                * _clamp01(current.confidence)
                * freshness
            )
            control_authority = measurement_authority
        else:
            # Exact observations use the full projected intercept uncertainty,
            # not merely a binary position-covariance qualification.  The
            # correction fades continuously as the predicted passage interval
            # becomes uninformative.
            uncertainty_authority = _clamp01(
                1.0
                - intercept_std
                / max(1e-6, cfg.passage_motion_full_std_norm)
            )
            measurement_authority = freshness * position_certainty
            control_authority = freshness * uncertainty_authority
        return _PassageMotion(
            bearing_error=error,
            physical_rate_norm_s=physical_rate,
            closure_rate_s=closure,
            closure_std_s=closure_std_s,
            ttc_s=ttc_s,
            ttc_std_s=ttc_std_s,
            projection_authority=projection_authority,
            fallback_intercept_error=fallback_intercept,
            optical_intercept_error=optical_intercept,
            intercept_error=intercept_error,
            bearing_std=axis.std,
            intercept_std=intercept_std,
            freshness_authority=freshness,
            measurement_authority=measurement_authority,
            control_authority=control_authority,
            directional_censor=directional_censor,
        )

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

    def _far_outer_vertical_direction(
        self,
        current: _Hypothesis,
        *,
        now_s: float,
        pitch_rad: float,
    ) -> int:
        """Return clear far-field outer-y direction without claiming magnitude.

        This is vertical energy shaping, not a second passage controller.  It
        prevents supporting IMU damping from commanding the opposite side of
        tilt support while fresh outer geometry clearly says the gate is high
        or low.  Static pitch compensation is deliberately excluded from the
        near-plane regime, where optical motion/censorship owns direction.
        """

        cfg = self.config
        if current.outer_log_scale >= cfg.commit_min_log_scale:
            return 0
        evidence_age = now_s - current.last_outer_y_evidence_s
        if evidence_age > cfg.vertical_qualify_max_age_s:
            return 0
        edge = current.vertical_censor_edge & (
            FrameEdge.TOP | FrameEdge.BOTTOM
        )
        if edge & FrameEdge.BOTTOM:
            return 1
        if edge & FrameEdge.TOP:
            return -1
        error = float(current.outer_y_axis.p)
        corridor = self._transported_corridor(current, now_s=now_s)
        if corridor is not None and corridor.live:
            offset = _clamp(
                corridor.center_y - current.outer_y_axis.p,
                -current.outer_half_span_y,
                current.outer_half_span_y,
            )
            error = current.outer_y_axis.p + offset
        error = self._compensated_ey(error, pitch_rad)
        if abs(error) <= current.outer_y_axis.std:
            return 0
        return 1 if error > 0.0 else -1

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

    def _refresh_corridor_certificate(
        self,
        hypothesis: _Hypothesis,
        track: Any,
        *,
        now_s: float,
        gate_index: Optional[int],
        scale_credible: bool = True,
    ) -> None:
        """Capture a passage-usable aperture against the co-timed outer box."""

        aperture = _track_aperture(track)
        aperture_meas = _aperture_track_measurement(track)
        clipping = getattr(track, "clipping", FrameEdge.NONE)
        if type(clipping) is not FrameEdge:
            clipping = FrameEdge.NONE
        if (
            aperture is None
            or aperture_meas is None
            or not scale_credible
            or not bool(getattr(aperture, "passage_usable", False))
            or getattr(aperture, "half_size_norm", None) is None
            or bool(
                clipping
                & (FrameEdge.LEFT | FrameEdge.RIGHT | FrameEdge.TOP | FrameEdge.BOTTOM)
            )
        ):
            return
        bbox = getattr(track, "bbox_norm", None)
        if bbox is None or len(bbox) < 4:
            return
        outer_half_x = float(bbox[2]) - float(bbox[0])
        outer_half_y = float(bbox[3]) - float(bbox[1])
        half_x = float(aperture.half_size_norm[0])
        half_y = float(aperture.half_size_norm[1])
        if min(outer_half_x, outer_half_y, half_x, half_y) <= 1e-6:
            return
        center, _log_scale, stds = aperture_meas
        outer_center = track.center_norm
        identity = self._last_frame_identity
        existing = hypothesis.corridor_certificate
        # A tracker republication of the same camera frame is not fresh
        # aperture evidence and may not extend the certificate lifetime.
        if (
            existing is not None
            and identity is not None
            and existing.frame_identity == identity
        ):
            return
        hypothesis.corridor_certificate = _ApertureCorridorCertificate(
            track_id=hypothesis.track_id,
            gate_index=gate_index,
            frame_identity=identity,
            source_s=float(now_s),
            aperture_center_x=float(center[0]),
            aperture_center_y=float(center[1]),
            aperture_half_x=half_x,
            aperture_half_y=half_y,
            offset_ratio_x=(float(center[0]) - float(outer_center[0]))
            / outer_half_x,
            offset_ratio_y=(float(center[1]) - float(outer_center[1]))
            / outer_half_y,
            half_ratio_x=half_x / outer_half_x,
            half_ratio_y=half_y / outer_half_y,
            center_std_x_norm=max(1e-3, float(stds[0])),
            center_std_y_norm=max(1e-3, float(stds[1])),
        )

    def _transported_corridor(
        self,
        hypothesis: _Hypothesis,
        *,
        now_s: float,
        gate_index: Optional[int] = None,
    ) -> Optional[_TransportedCorridor]:
        """Transport one gate-owned aperture certificate on the outer track."""

        certificate = hypothesis.corridor_certificate
        owner = self.gate_index if gate_index is None else int(gate_index)
        if (
            certificate is None
            or certificate.track_id != hypothesis.track_id
            or certificate.gate_index != owner
        ):
            return None
        age_s = max(0.0, float(now_s) - certificate.source_s)
        if age_s > self.config.predict_max_gap_s:
            return None
        live = age_s <= self.config.commit_entry_meas_max_age_s
        if live:
            center_x = certificate.aperture_center_x
            center_y = certificate.aperture_center_y
            half_x = certificate.aperture_half_x
            half_y = certificate.aperture_half_y
        else:
            center_x = (
                hypothesis.outer_x_axis.p
                + certificate.offset_ratio_x * hypothesis.outer_half_span_x
            )
            center_y = (
                hypothesis.outer_y_axis.p
                + certificate.offset_ratio_y * hypothesis.outer_half_span_y
            )
            half_x = certificate.half_ratio_x * hypothesis.outer_half_span_x
            half_y = certificate.half_ratio_y * hypothesis.outer_half_span_y
        growth = age_s * self.config.passage_motion_model_std_norm
        center_std_x = (
            certificate.center_std_x_norm
            if live
            else math.hypot(
                certificate.center_std_x_norm,
                hypothesis.outer_x_axis.std,
            )
            + growth
        )
        center_std_y = (
            certificate.center_std_y_norm
            if live
            else math.hypot(
                certificate.center_std_y_norm,
                hypothesis.outer_y_axis.std,
            )
            + growth
        )
        return _TransportedCorridor(
            track_id=hypothesis.track_id,
            gate_index=owner,
            frame_identity=certificate.frame_identity,
            source_age_s=age_s,
            center_x=float(center_x),
            center_y=float(center_y),
            half_x=float(half_x),
            half_y=float(half_y),
            center_std_x=float(center_std_x),
            center_std_y=float(center_std_y),
            live=live,
        )

    def _hypothesis_from_track(
        self,
        track: Any,
        now_s: float,
        *,
        gate_index: Optional[int] = None,
    ) -> _Hypothesis:
        outer_center, outer_log_scale, _outer_stds = _outer_track_measurement(track)
        aperture_meas = _aperture_track_measurement(track)
        if aperture_meas is None:
            center = outer_center
            passage_log_scale = outer_log_scale
            passage_source = "outer"
        else:
            center = aperture_meas[0]
            passage_log_scale = aperture_meas[1]
            passage_source = "aperture"
        hypothesis = _Hypothesis(
            track_id=str(track.track_id),
            x=center[0],
            y=center[1],
            log_scale=passage_log_scale,
            outer_x=outer_center[0],
            outer_y=outer_center[1],
            outer_log_scale=outer_log_scale,
            passage_source=passage_source,
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
        # ``center_censored`` is aggregate tracker metadata: production sets
        # it for any clipped edge.  Edge bits remain the axis contract.  Only
        # an aggregate censor with no directional bits invalidates both axes.
        nondirectional_censor = center_censored and clipping == FrameEdge.NONE
        hypothesis.clipped = clipping != FrameEdge.NONE
        hypothesis.clipping_edges = clipping
        x_censored = (
            nondirectional_censor
            or bool(clipping & (FrameEdge.LEFT | FrameEdge.RIGHT))
        )
        if x_censored:
            hypothesis.last_x_measurement_s = NEVER_MEASURED_S
            hypothesis.last_outer_x_measurement_s = NEVER_MEASURED_S
            horizontal_edge = clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)
            hypothesis.horizontal_censor_edge = horizontal_edge
            hypothesis.horizontal_censor_bound = (
                float(track.center_norm[0])
                if horizontal_edge != FrameEdge.NONE
                else None
            )
            hypothesis.last_outer_x_evidence_s = (
                float(now_s)
                if horizontal_edge != FrameEdge.NONE
                else NEVER_MEASURED_S
            )
        else:
            # F56: adoption with an uncensored x carries the creating
            # detection's true bbox half-width for the COMMIT corridor.
            bbox = getattr(track, "bbox_norm", None)
            if bbox is not None and len(bbox) >= 4:
                hypothesis.outer_half_span_x = (
                    float(bbox[2]) - float(bbox[0])
                )
                hypothesis.outer_half_span_y = (
                    float(bbox[3]) - float(bbox[1])
                )
        if nondirectional_censor or bool(
            clipping & (FrameEdge.TOP | FrameEdge.BOTTOM)
        ):
            hypothesis.last_y_measurement_s = NEVER_MEASURED_S
            hypothesis.last_outer_y_measurement_s = NEVER_MEASURED_S
            hypothesis.vertical_censor_edge = clipping & (
                FrameEdge.TOP | FrameEdge.BOTTOM
            )
            hypothesis.vertical_censor_bound = float(track.center_norm[1])
            hypothesis.last_outer_y_evidence_s = (
                float(now_s)
                if hypothesis.vertical_censor_edge != FrameEdge.NONE
                else NEVER_MEASURED_S
            )
        aperture = _track_aperture(track)
        if (
            aperture is not None
            and bool(getattr(aperture, "passage_usable", False))
            and getattr(aperture, "half_size_norm", None) is not None
        ):
            hypothesis.aperture_half_x = float(aperture.half_size_norm[0])
            hypothesis.aperture_half_y = float(aperture.half_size_norm[1])
        self._refresh_corridor_certificate(
            hypothesis,
            track,
            now_s=now_s,
            gate_index=gate_index,
        )
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
        # fixed image features toward image-left.  F143 corrects the vertical
        # sign to match the controller's own compensated coordinate:
        # raw_y = world_y + (spawn_pitch - pitch) * focal, hence fixed-world
        # flow is -pitch_rate * focal.  F142 used the opposite prediction,
        # manufactured a positive y-rate during the nose-up brake, and kept
        # thrust sub-support into a -1.07 m/s Gate-0 sink.
        pitch_rate = float(body_rates[1])
        yaw_rate = float(body_rates[2])
        drift_x = -yaw_rate * ROTATION_COMP_FOCAL_NORM * dt
        drift_y = -pitch_rate * ROTATION_COMP_FOCAL_NORM * dt
        hypothesis.x_axis.predict(dt, drift=drift_x)
        hypothesis.y_axis.predict(dt, drift=drift_y)
        hypothesis.outer_x_axis.predict(dt, drift=drift_x)
        hypothesis.outer_y_axis.predict(dt, drift=drift_y)
        hypothesis.scale_axis.predict(dt)
        hypothesis.outer_scale_axis.predict(dt)
        compensation_var = ROTATION_COMP_UNCERTAINTY * (
            abs(drift_x) + abs(drift_y)
        )
        hypothesis.x_axis.inflate(LATENCY_VAR_NORM + compensation_var)
        hypothesis.y_axis.inflate(LATENCY_VAR_NORM + compensation_var)
        hypothesis.outer_x_axis.inflate(LATENCY_VAR_NORM + compensation_var)
        hypothesis.outer_y_axis.inflate(LATENCY_VAR_NORM + compensation_var)

    def _update_hypothesis(
        self,
        hypothesis: _Hypothesis,
        track: Any,
        now_s: float,
    ) -> None:
        (outer_zx, outer_zy), outer_z_log_scale, outer_stds = (
            _outer_track_measurement(track)
        )
        aperture_meas = _aperture_track_measurement(track)
        clipping = getattr(track, "clipping", FrameEdge.NONE)
        if type(clipping) is not FrameEdge:
            clipping = FrameEdge.NONE
        center_censored = bool(getattr(track, "center_censored", False))
        nondirectional_censor = center_censored and clipping == FrameEdge.NONE
        x_censored = (
            nondirectional_censor
            or bool(clipping & (FrameEdge.LEFT | FrameEdge.RIGHT))
        )
        y_censored = (
            nondirectional_censor
            or bool(clipping & (FrameEdge.TOP | FrameEdge.BOTTOM))
        )
        hypothesis.vertical_censor_edge = (
            clipping & (FrameEdge.TOP | FrameEdge.BOTTOM)
            if y_censored
            else FrameEdge.NONE
        )
        confidence = max(
            MIN_MEAS_CONFIDENCE,
            float(track.confidence)
            * float(getattr(track, "association_confidence", 1.0)),
        )
        outer_r_x = (outer_stds[0] ** 2) / confidence
        outer_r_y = (outer_stds[1] ** 2) / confidence
        outer_r_scale = (outer_stds[2] ** 2) / confidence

        # Outer association/FOV/range state has one and only one measurement
        # modality.  A censored coordinate remains an inequality rather than a
        # forced-zero observation.  Its timestamp is still fresh directional
        # evidence: RIGHT means x is at least the clipped bound, LEFT means at
        # most the bound.  Never inject that bound as an exact Kalman sample.
        horizontal_edge = clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)
        if x_censored:
            hypothesis.outer_x_axis.inflate(CENSOR_INFLATE_VAR_NORM)
            hypothesis.horizontal_censor_edge = horizontal_edge
            hypothesis.horizontal_censor_bound = (
                float(track.center_norm[0])
                if horizontal_edge != FrameEdge.NONE
                else None
            )
            if horizontal_edge != FrameEdge.NONE:
                hypothesis.last_outer_x_evidence_s = float(now_s)
        else:
            hypothesis.outer_x_axis.update(outer_zx, outer_r_x)
            hypothesis.outer_raw_x = float(outer_zx)
            hypothesis.last_outer_x_measurement_s = float(now_s)
            hypothesis.last_outer_x_evidence_s = float(now_s)
            hypothesis.horizontal_censor_edge = FrameEdge.NONE
            hypothesis.horizontal_censor_bound = None
        if y_censored:
            hypothesis.outer_y_axis.inflate(CENSOR_INFLATE_VAR_NORM)
            # A clipped coordinate is an inequality, not an exact center and
            # not "stationary".  The clipped outer-box center is conservative:
            # BOTTOM is a lower bound on image-down error; TOP is an upper
            # bound.  Preserve that directional fact while covariance grows.
            censor_bound = float(track.center_norm[1])
            hypothesis.vertical_censor_bound = censor_bound
            if hypothesis.vertical_censor_edge != FrameEdge.NONE:
                hypothesis.last_outer_y_evidence_s = float(now_s)
        else:
            hypothesis.outer_y_axis.update(outer_zy, outer_r_y)
            hypothesis.outer_raw_y = float(outer_zy)
            hypothesis.last_outer_y_measurement_s = float(now_s)
            hypothesis.last_outer_y_evidence_s = float(now_s)
            hypothesis.vertical_censor_bound = None
        if x_censored or y_censored:
            hypothesis.outer_scale_axis.inflate(CENSOR_INFLATE_VAR_NORM)
            if hypothesis.passage_source == "outer":
                hypothesis.scale_axis.inflate(CENSOR_INFLATE_VAR_NORM)
        else:
            hypothesis.outer_scale_axis.update(
                outer_z_log_scale, outer_r_scale
            )
            if hypothesis.passage_source == "outer":
                hypothesis.scale_axis.update(
                    outer_z_log_scale, outer_r_scale
                )
        hypothesis.confidence = _clamp01(float(track.confidence))
        hypothesis.clipped = clipping != FrameEdge.NONE
        hypothesis.clipping_edges = clipping
        if hypothesis.clipped:
            # Clipping increases uncertainty; it is not an abort condition.
            hypothesis.outer_x_axis.inflate(CLIPPED_INFLATE_VAR_NORM)
            hypothesis.outer_y_axis.inflate(CLIPPED_INFLATE_VAR_NORM)
        hypothesis.last_measurement_s = float(now_s)
        new_outer_log_scale = outer_z_log_scale
        # F99: fast raw closure signal (see the OUTER_EXPANSION_* block) —
        # EMA of the per-frame outer log-scale rate, so the closure
        # governor does not wait ~1 s for the Kalman scale_axis.v to
        # tighten on a fresh/adopted track.
        dt_outer = float(now_s) - hypothesis.outer_log_scale_s
        if dt_outer > 1e-6:
            raw_rate = (
                new_outer_log_scale - hypothesis.outer_log_scale
            ) / dt_outer
            alpha = dt_outer / (self.config.outer_expansion_tau_s + dt_outer)
            hypothesis.outer_expansion_rate += alpha * (
                raw_rate - hypothesis.outer_expansion_rate
            )
        hypothesis.outer_log_scale = new_outer_log_scale
        hypothesis.outer_log_scale_s = float(now_s)
        bbox = getattr(track, "bbox_norm", None)
        if bbox is not None and len(bbox) >= 4:
            if not x_censored:
                hypothesis.outer_half_span_x = (
                    float(bbox[2]) - float(bbox[0])
                )
            if not y_censored:
                hypothesis.outer_half_span_y = (
                    float(bbox[3]) - float(bbox[1])
                )

        # Record CURRENT-frame geometry separately from the persistent
        # certificate.  Missing aperture fits clear only these live fields;
        # they cannot inject outer coordinates into the passage derivative.
        # The aperture scale state is independent from the outer admission
        # state.  Reject a fit whose scale is statistically impossible before
        # it can refresh aim geometry or approach energy.
        aperture_scale_credible = False
        if aperture_meas is not None and not (x_censored or y_censored):
            _aperture_center, aperture_log_scale, aperture_stds = aperture_meas
            aperture_scale_r = (aperture_stds[2] ** 2) / confidence
            if hypothesis.passage_source == "outer":
                aperture_scale_credible = True
            else:
                innovation_std = math.sqrt(
                    max(1e-12, hypothesis.scale_axis.pp + aperture_scale_r)
                )
                aperture_scale_credible = bool(
                    abs(aperture_log_scale - hypothesis.scale_axis.p)
                    <= self.config.aperture_scale_innovation_sigma
                    * innovation_std
                )
        aperture = _track_aperture(track)
        if (
            aperture is not None
            and aperture_scale_credible
            and bool(getattr(aperture, "passage_usable", False))
            and getattr(aperture, "half_size_norm", None) is not None
        ):
            hypothesis.aperture_half_x = float(aperture.half_size_norm[0])
            hypothesis.aperture_half_y = float(aperture.half_size_norm[1])
        else:
            hypothesis.aperture_half_x = None
            hypothesis.aperture_half_y = None

        active_gate_index = self.gate_index if hypothesis is self.current else None
        self._refresh_corridor_certificate(
            hypothesis,
            track,
            now_s=now_s,
            gate_index=active_gate_index,
            scale_credible=aperture_scale_credible,
        )

        # Passage center state is either a direct aperture series, a transported
        # certificate driven by the outer series, or (until the first aperture
        # is ever seen) an outer-center fallback.  It never alternates between
        # aperture and outer measurements in one derivative filter.
        corridor = (
            self._transported_corridor(hypothesis, now_s=now_s)
            if active_gate_index is not None
            else None
        )
        if aperture_scale_credible:
            (passage_x, passage_y), aperture_log_scale, passage_stds = (
                aperture_meas
            )
            if hypothesis.passage_source == "outer":
                hypothesis.x_axis = _AxisFilter(
                    passage_x, 0.0, passage_stds[0] ** 2, INITIAL_RATE_VAR
                )
                hypothesis.y_axis = _AxisFilter(
                    passage_y, 0.0, passage_stds[1] ** 2, INITIAL_RATE_VAR
                )
                hypothesis.scale_axis = _AxisFilter(
                    aperture_log_scale,
                    0.0,
                    passage_stds[2] ** 2,
                    INITIAL_RATE_VAR,
                )
                hypothesis.passage_source = "aperture"
            else:
                hypothesis.x_axis.update(
                    passage_x, (passage_stds[0] ** 2) / confidence
                )
                hypothesis.y_axis.update(
                    passage_y, (passage_stds[1] ** 2) / confidence
                )
                hypothesis.scale_axis.update(
                    aperture_log_scale,
                    (passage_stds[2] ** 2) / confidence,
                )
            hypothesis.raw_x = float(passage_x)
            hypothesis.raw_y = float(passage_y)
            hypothesis.last_x_measurement_s = float(now_s)
            hypothesis.last_y_measurement_s = float(now_s)
            hypothesis.last_aperture_scale_measurement_s = float(now_s)
        elif hypothesis.passage_source == "outer":
            if x_censored:
                hypothesis.x_axis.inflate(CENSOR_INFLATE_VAR_NORM)
            else:
                hypothesis.x_axis.update(outer_zx, outer_r_x)
                hypothesis.raw_x = float(outer_zx)
                hypothesis.last_x_measurement_s = float(now_s)
            if y_censored:
                hypothesis.y_axis.inflate(CENSOR_INFLATE_VAR_NORM)
            else:
                hypothesis.y_axis.update(outer_zy, outer_r_y)
                hypothesis.raw_y = float(outer_zy)
                hypothesis.last_y_measurement_s = float(now_s)
        elif corridor is not None:
            if x_censored:
                hypothesis.x_axis.inflate(CENSOR_INFLATE_VAR_NORM)
            else:
                hypothesis.x_axis.update(
                    corridor.center_x,
                    max(1e-6, corridor.center_std_x**2) / confidence,
                )
                hypothesis.raw_x = float(corridor.center_x)
                hypothesis.last_x_measurement_s = float(now_s)
            if y_censored:
                hypothesis.y_axis.inflate(CENSOR_INFLATE_VAR_NORM)
            else:
                hypothesis.y_axis.update(
                    corridor.center_y,
                    max(1e-6, corridor.center_std_y**2) / confidence,
                )
                hypothesis.raw_y = float(corridor.center_y)
                hypothesis.last_y_measurement_s = float(now_s)
        else:
            hypothesis.x_axis.inflate(CENSOR_INFLATE_VAR_NORM)
            hypothesis.y_axis.inflate(CENSOR_INFLATE_VAR_NORM)

        if y_censored and hypothesis.vertical_censor_bound is not None:
            if clipping & FrameEdge.BOTTOM:
                hypothesis.y_axis.p = max(
                    hypothesis.y_axis.p, hypothesis.vertical_censor_bound
                )
            elif clipping & FrameEdge.TOP:
                hypothesis.y_axis.p = min(
                    hypothesis.y_axis.p, hypothesis.vertical_censor_bound
                )
        if hypothesis.clipped:
            hypothesis.x_axis.inflate(CLIPPED_INFLATE_VAR_NORM)
            hypothesis.y_axis.inflate(CLIPPED_INFLATE_VAR_NORM)

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
            nondirectional_censor = (
                center_censored and clipping == FrameEdge.NONE
            )
            if not nondirectional_censor and not bool(
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
        # F107 (20260801T053629Z-visual-course-cb3892b6): after the stronger
        # Gate-0 brake pushed the current gate below frame, generic SEARCH
        # adopted the already-retained Gate-1 successor and drove toward it
        # while authoritative race ownership remained Gate 0.  A retained
        # successor is explicitly next-gate evidence while a current gate
        # still owns the leg; only note_race() may promote it then.  F110:
        # once an authoritative increment has cleared current ownership, the
        # retained hypothesis describes the gate that is NOW authorized and
        # SEARCH must be allowed to qualify/adopt it (F109 otherwise adopted
        # tiny track 06 instead of retained Gate-1 track 04).
        successor_id = (
            self._successor_track_id() if self.current is not None else None
        )
        # F49: newborn suspicious-geometry tracks (ceiling-truss slabs,
        # extreme aspects) are ineligible until they persist — the terminal
        # F48 re-acquisition adopted one over the persistent real gate.
        eligible = [
            track
            for track in tracks
            if track.track_id != successor_id
            and not (
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

    def _horizontal_control_observable(
        self,
        hypothesis: _Hypothesis,
        now_s: float,
    ) -> Tuple[float, _AxisFilter, float, FrameEdge]:
        """Return the continuous outer-owned horizontal steering evidence.

        A fresh aperture contributes only its bounded co-timed aim offset.
        Once that fit ages, the live outer bearing remains the owner.  A
        LEFT/RIGHT clipped observation is an inequality and immediately owns
        direction without being injected as an exact filter measurement.
        """

        # Focused legacy fixtures without an aperture genuinely have one outer
        # series; preserve their direct-state seam.  Production hypotheses that
        # have ever acquired an aperture always consume the explicit outer axis.
        direct_outer = bool(
            hypothesis.passage_source == "outer"
            and hypothesis.corridor_certificate is None
        )
        axis = hypothesis.x_axis if direct_outer else hypothesis.outer_x_axis
        error = float(axis.p)
        age_s = (
            now_s - hypothesis.last_x_measurement_s
            if direct_outer
            else now_s - hypothesis.last_outer_x_evidence_s
        )
        edge = FrameEdge.NONE
        if (
            now_s - hypothesis.last_outer_x_evidence_s
            <= self.config.predict_max_gap_s
        ):
            edge = hypothesis.horizontal_censor_edge & (
                FrameEdge.LEFT | FrameEdge.RIGHT
            )
            if edge != FrameEdge.NONE:
                age_s = now_s - hypothesis.last_outer_x_evidence_s
        if edge != FrameEdge.NONE and hypothesis.horizontal_censor_bound is not None:
            bound = float(hypothesis.horizontal_censor_bound)
            if edge & FrameEdge.RIGHT:
                error = max(error, bound, 0.0)
            elif edge & FrameEdge.LEFT:
                error = min(error, bound, 0.0)
        elif hypothesis is self.current:
            corridor = self._transported_corridor(hypothesis, now_s=now_s)
            if corridor is not None and corridor.live:
                # The aperture is an aim OFFSET carried by the continuous
                # outer state, never a replacement absolute steering track.
                offset = _clamp(
                    corridor.center_x - hypothesis.outer_x_axis.p,
                    -hypothesis.outer_half_span_x,
                    hypothesis.outer_half_span_x,
                )
                error = float(hypothesis.outer_x_axis.p + offset)
        return _clamp(error, -1.0, 1.0), axis, max(0.0, age_s), edge

    def _horizontal_passage_release(
        self,
        current: _Hypothesis,
        now_s: float,
    ) -> float:
        """Continuous safe-current-path lease for coherent successor preturn."""

        cfg = self.config
        corridor = self._transported_corridor(current, now_s=now_s)
        error, axis, age_s, edge = self._horizontal_control_observable(
            current, now_s
        )
        if corridor is not None and corridor.half_x > 0.0 and edge == FrameEdge.NONE:
            motion = self._passage_motion(
                current,
                axis,
                error,
                now_s=now_s,
                measurement_age_s=age_s,
            )
            budget = cfg.commit_entry_aperture_margin_frac * corridor.half_x
            tube = max(
                abs(motion.fallback_intercept_error),
                abs(motion.optical_intercept_error),
            ) + cfg.commit_entry_sigma_mult * (
                motion.bearing_std
                + cfg.passage_motion_model_std_norm
                + corridor.center_std_x
            )
            if budget > 1e-9:
                center_reserve = _clamp01(
                    1.0 - abs(corridor.center_x) / budget
                )
                tube_reserve = _clamp01(
                    1.0 - tube / corridor.half_x
                )
                return min(center_reserve, tube_reserve)
            return 0.0
        if (
            current.corridor_certificate is None
            and current.aperture_half_x is not None
            and current.aperture_half_x > 0.0
        ):
            # Direct-state unit fixtures predate the tracker certificate seam.
            budget = (
                cfg.commit_entry_aperture_margin_frac
                * current.aperture_half_x
            )
            projected = (
                max(abs(current.x), abs(current.raw_x))
                + abs(current.vx) * cfg.successor_preview_projection_s
            )
            return (
                _clamp01(1.0 - projected / budget)
                if budget > 1e-9
                else 0.0
            )
        return 0.0

    def _turn_reference(
        self,
        current: _Hypothesis,
        successor: Optional[_Hypothesis],
        *,
        current_error: float,
        now_s: float,
        yaw_rad: Optional[float],
        dt: float,
    ) -> Tuple[float, float]:
        """One continuous current-passage/successor lateral reference.

        Both hypotheses are already IMU-derotated by ``_predict``.  Between
        command ticks the carried reference is derotated by measured yaw, then
        filtered toward a confidence/covariance/freshness-weighted blend.  No
        raw successor image coordinate is cached, and no state transition can
        inject an independent yaw or bank request.
        """

        cfg = self.config
        if yaw_rad is not None:
            yaw = float(yaw_rad)
            if (
                self._turn_reference_x is not None
                and self._turn_reference_yaw_rad is not None
            ):
                delta_yaw = math.atan2(
                    math.sin(yaw - self._turn_reference_yaw_rad),
                    math.cos(yaw - self._turn_reference_yaw_rad),
                )
                self._turn_reference_x -= (
                    delta_yaw * ROTATION_COMP_FOCAL_NORM
                )
            self._turn_reference_yaw_rad = yaw

        alpha = _clamp01(dt / max(1e-6, cfg.turn_reference_tau_s + dt))
        # Successor preturn is a release from a demonstrably safe CURRENT
        # horizontal path, never a reward for stale aperture covariance.  The
        # tube includes fresh outer-owned optical interception and corridor
        # uncertainty; missing/expired geometry yields zero release.
        aperture_target = self._horizontal_passage_release(current, now_s)
        if self._last_engulfing_anchor_s is not None:
            # F145: both existing passage observations feed the same filtered
            # reserve.  F144 retained a stale non-None aperture extent, so the
            # old if/elif precedence shadowed fresh same-id engulfing evidence
            # and collapsed reserve .012 -> .001.  Age the anchor evidence
            # continuously and take the stronger physical passage margin;
            # there is no new mode, latch, threshold, or command owner.
            anchor_age_s = max(
                0.0, now_s - self._last_engulfing_anchor_s
            )
            anchor_passage = _clamp01(
                1.0 - anchor_age_s / ENGULFING_ANCHOR_MAX_AGE_S
            )
            aperture_target = max(aperture_target, anchor_passage)
        self._turn_aperture_reserve += alpha * (
            aperture_target - self._turn_aperture_reserve
        )
        self._turn_aperture_reserve = _clamp01(self._turn_aperture_reserve)

        desired_authority = 0.0
        desired = float(current_error)
        if successor is not None:
            successor_heading, successor_axis, successor_age, _successor_edge = (
                self._horizontal_control_observable(successor, now_s)
            )
            closure_span = cfg.blend_near_log_scale - cfg.blend_far_log_scale
            closure = (
                _clamp01(
                    (current.outer_log_scale - cfg.blend_far_log_scale)
                    / closure_span
                )
                if abs(closure_span) > 1e-9
                else 0.0
            )
            confidence = _clamp01(successor.confidence)
            uncertainty = _clamp01(
                1.0
                - successor_axis.std / cfg.successor_turn_max_std_norm
            )
            freshness = _clamp01(
                1.0 - successor_age / PREDICT_STALL_FORCE_SEARCH_S
            )
            range_order = _clamp01(
                (current.outer_log_scale - successor.outer_log_scale)
                / max(1e-6, cfg.successor_min_log_scale_gap)
            )
            successor_weight = (
                closure
                * confidence
                * uncertainty
                * freshness
                * range_order
            )
            current_confidence = _clamp01(current.confidence)
            _current_heading, current_axis, current_age, _edge = (
                self._horizontal_control_observable(current, now_s)
            )
            current_uncertainty = _clamp01(
                1.0 - current_axis.std / cfg.search_covariance_std_norm
            )
            current_freshness = _clamp01(
                1.0 - max(0.0, current_age) / cfg.x_steer_max_age_s
            )
            current_claim = (
                (1.0 - self._turn_aperture_reserve)
                * current_confidence
                * current_uncertainty
                * current_freshness
            )
            # F165: only gate-owned safe-passage reserve releases preturn.
            # Aging aperture state cannot grant the next gate yaw authority
            # while a fresh outer current gate remains visibly misaligned.
            desired_authority = (
                self._turn_aperture_reserve * successor_weight
            )
        self._turn_successor_authority += alpha * (
            desired_authority - self._turn_successor_authority
        )
        self._turn_successor_authority = _clamp01(
            self._turn_successor_authority
        )
        authority = (
            self._turn_successor_authority if successor is not None else 0.0
        )
        _owned_heading, _owned_axis, _owned_age, owned_edge = (
            self._horizontal_control_observable(current, now_s)
        )
        if owned_edge != FrameEdge.NONE:
            # A one-sided current-gate observation revokes any carried preturn
            # lease immediately; passage is no longer demonstrably secured.
            self._turn_successor_authority = 0.0
            authority = 0.0
        if successor is not None:
            # F146: F145 computed an evidence-backed current claim, then the
            # final blend discarded it and implicitly restored current error
            # to (1 - successor authority).  That recreated the rightward
            # counterturn whenever the old left bearing grew uncertain.  Use
            # the two claims actually supported by evidence; unclaimed weight
            # is neutral, and the existing derotated reference filter supplies
            # continuity.  Weak evidence now decays toward zero rather than
            # granting the opposing current error invented authority.
            current_weight = min(current_claim, 1.0 - authority)
            desired = (
                current_weight * float(current_error)
                + authority * successor_heading
            )

        if self._turn_reference_x is None:
            self._turn_reference_x = float(desired)
        else:
            self._turn_reference_x += alpha * (
                float(desired) - self._turn_reference_x
            )
        self._turn_reference_x = _clamp(self._turn_reference_x, -1.0, 1.0)
        if owned_edge & FrameEdge.RIGHT:
            self._turn_reference_x = max(self._turn_reference_x, 0.0)
        elif owned_edge & FrameEdge.LEFT:
            self._turn_reference_x = min(self._turn_reference_x, 0.0)

        self._successor_heading_blend = authority
        self._successor_heading_error_norm = self._turn_reference_x
        return self._turn_reference_x, authority

    def _coordinated_turn_request(
        self,
        reference_x: float,
        *,
        steer_gain: float,
        yaw_rad: Optional[float],
        intercept_x: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Map separate FOV-heading and physical-intercept references.

        The optional argument preserves the old pure helper seam for focused
        tests, but production always supplies the optical plane-miss reference:
        yaw may center the camera while roll continues bending the velocity
        vector toward the aperture.
        """

        cfg = self.config
        yaw_rate = _clamp(
            cfg.yaw_error_sign * cfg.yaw_error_gain * steer_gain * reference_x,
            -cfg.max_yaw_rate_rad_s,
            cfg.max_yaw_rate_rad_s,
        )
        yaw_rate = self._anchor_clamped_yaw(yaw_rate, yaw_rad)
        roll_reference = reference_x if intercept_x is None else intercept_x
        target_roll = _clamp(
            cfg.roll_error_sign
            * cfg.roll_error_gain
            * steer_gain
            * roll_reference,
            -cfg.max_target_roll_rad,
            cfg.max_target_roll_rad,
        )
        return target_roll, yaw_rate

    def _passage_vertical_error(
        self,
        current: _Hypothesis,
        vertical_error: float,
    ) -> float:
        """Compatibility seam returning the TTC-based optical plane miss."""

        motion = self._passage_motion(
            current,
            current.y_axis,
            vertical_error,
            now_s=current.last_measurement_s,
            measurement_age_s=0.0,
        )
        return motion.intercept_error

    def _vertical_passage_motion(
        self,
        current: _Hypothesis,
        vertical_error: float,
        *,
        now_s: float,
        vertical_qualified: bool,
        pitch_rad: float,
    ) -> Optional[_PassageMotion]:
        """Return exact or one-sided vertical optical passage evidence."""

        censor = current.vertical_censor_edge & (
            FrameEdge.TOP | FrameEdge.BOTTOM
        )
        fresh_current_owned = bool(
            self._current_fresh_observation_track_id == current.track_id
            and self._current_fresh_observation_s is not None
            and now_s - self._current_fresh_observation_s
            <= self.config.predict_max_gap_s
        )
        # A fresh directional clip is newer evidence than the last exact y
        # sample.  F162 kept treating BOTTOM frames as uncensored for about
        # 0.20 s because last_y_measurement_s still satisfied the 0.30 s
        # exact-axis qualification.  Consume the one-sided observation on its
        # first frame instead of waiting for that old exact stamp to expire.
        if (
            censor != FrameEdge.NONE
            and fresh_current_owned
            and now_s - current.last_measurement_s
            <= self.config.predict_max_gap_s
        ):
            age_s = now_s - current.last_measurement_s
            error = float(vertical_error)
            if current.vertical_censor_bound is not None:
                bound = self._compensated_ey(
                    current.vertical_censor_bound, pitch_rad
                )
                if censor & FrameEdge.BOTTOM:
                    error = max(error, bound, 0.0)
                elif censor & FrameEdge.TOP:
                    error = min(error, bound, 0.0)
        elif vertical_qualified:
            censor = FrameEdge.NONE
            age_s = now_s - current.last_y_measurement_s
            error = float(vertical_error)
        else:
            self._last_vertical_motion = None
            self._vertical_direction_edge_active = False
            return None

        motion = self._passage_motion(
            current,
            current.y_axis,
            error,
            now_s=now_s,
            measurement_age_s=age_s,
            directional_censor=censor,
        )
        intercept = motion.intercept_error
        if censor & FrameEdge.BOTTOM:
            intercept = max(intercept, error, 0.0)
        elif censor & FrameEdge.TOP:
            intercept = min(intercept, error, 0.0)
        if intercept != motion.intercept_error:
            motion = _PassageMotion(
                bearing_error=motion.bearing_error,
                physical_rate_norm_s=motion.physical_rate_norm_s,
                closure_rate_s=motion.closure_rate_s,
                closure_std_s=motion.closure_std_s,
                ttc_s=motion.ttc_s,
                ttc_std_s=motion.ttc_std_s,
                projection_authority=motion.projection_authority,
                fallback_intercept_error=motion.fallback_intercept_error,
                optical_intercept_error=motion.optical_intercept_error,
                intercept_error=intercept,
                bearing_std=motion.bearing_std,
                intercept_std=motion.intercept_std,
                freshness_authority=motion.freshness_authority,
                measurement_authority=motion.measurement_authority,
                control_authority=motion.control_authority,
                directional_censor=motion.directional_censor,
            )
        self._update_vertical_direction(current, motion, now_s=now_s)
        self._last_vertical_motion = motion
        return motion

    def _clear_current_fresh_observation(self) -> None:
        self._current_fresh_observation_track_id = None
        self._current_fresh_observation_s = None
        self._current_fresh_y_observation_s = None
        self._current_fresh_y_observation_serial = 0

    def _seed_current_fresh_observation(
        self, current: _Hypothesis, *, now_s: float
    ) -> None:
        """Seed explicit freshness from a newly consumed camera frame."""

        self._current_fresh_observation_track_id = current.track_id
        self._current_fresh_observation_s = float(now_s)
        if current.last_y_measurement_s > NEVER_MEASURED_S + 1.0:
            self._current_fresh_y_observation_s = float(now_s)
            self._current_fresh_y_observation_serial = 1
        else:
            self._current_fresh_y_observation_s = None
            self._current_fresh_y_observation_serial = 0

    def _record_current_fresh_observation(
        self,
        current: _Hypothesis,
        *,
        now_s: float,
        fresh: bool,
        previous_y_measurement_s: float,
    ) -> None:
        """Record one current-track update only if its camera frame is new."""

        if not fresh:
            return
        if self._current_fresh_observation_track_id != current.track_id:
            self._seed_current_fresh_observation(current, now_s=now_s)
            return
        self._current_fresh_observation_s = float(now_s)
        if (
            current.last_y_measurement_s
            > float(previous_y_measurement_s) + 1e-9
        ):
            self._current_fresh_y_observation_s = float(now_s)
            self._current_fresh_y_observation_serial += 1

    def _reset_vertical_direction(
        self, track_id: Optional[str] = None
    ) -> None:
        """Clear trajectory-direction evidence at a gate/track boundary."""

        self._vertical_direction_track_id = track_id
        self._vertical_direction_last_y_observation_serial = 0
        self._vertical_direction_streak_sign = 0
        self._vertical_direction_streak = 0
        self._vertical_direction_sign = 0
        self._vertical_direction_supported = False
        self._vertical_direction_source = None
        self._vertical_direction_fast_until_s = None
        self._vertical_direction_edge_active = False
        self._vertical_direction_magnitude = 0.0

    @staticmethod
    def _coherent_vertical_motion_sign(motion: _PassageMotion) -> int:
        """Return a shared optical trajectory sign, independent of bearing."""

        values = (
            motion.fallback_intercept_error,
            motion.optical_intercept_error,
            motion.physical_rate_norm_s,
        )
        if all(value > 0.0 for value in values):
            return 1
        if all(value < 0.0 for value in values):
            return -1
        return 0

    def _update_vertical_direction(
        self,
        current: _Hypothesis,
        motion: _PassageMotion,
        *,
        now_s: float,
    ) -> None:
        """Update direction from distinct optical observations or clipping.

        Static pitch compensation remains useful for level-frame position and
        conservative correction magnitude, but it does not vote on the
        near-plane trajectory sign.  Exact direction requires three distinct
        y measurements whose short-horizon projection, TTC projection, and
        de-dilated image motion agree.  Projection covariance controls normal
        magnitude; it cannot erase this separately established direction.
        """

        cfg = self.config
        if self._vertical_direction_track_id != current.track_id:
            self._reset_vertical_direction(current.track_id)

        censor = motion.directional_censor & (
            FrameEdge.TOP | FrameEdge.BOTTOM
        )
        if censor != FrameEdge.NONE:
            sign = 1 if censor & FrameEdge.BOTTOM else -1
            first_edge_frame = not self._vertical_direction_edge_active
            self._vertical_direction_edge_active = True
            if sign != self._vertical_direction_sign or first_edge_frame:
                self._vertical_direction_fast_until_s = (
                    now_s + cfg.vertical_direction_fast_window_s
                )
            self._vertical_direction_sign = sign
            self._vertical_direction_streak_sign = sign
            self._vertical_direction_streak = max(
                self._vertical_direction_streak,
                cfg.vertical_direction_streak_frames,
            )
            self._vertical_direction_supported = True
            self._vertical_direction_source = (
                "bottom_censor" if sign > 0 else "top_censor"
            )
            return

        self._vertical_direction_edge_active = False
        # Coherent exact motion only takes directional ownership in the
        # near-plane regime.  Far away, the ordinary uncertainty-weighted
        # passage controller retains continuous authority.
        if current.outer_log_scale < cfg.commit_min_log_scale:
            self._vertical_direction_streak_sign = 0
            self._vertical_direction_streak = 0
            return

        sign = self._coherent_vertical_motion_sign(motion)
        fresh_y_owned = bool(
            self._current_fresh_observation_track_id == current.track_id
            and self._current_fresh_y_observation_s is not None
            and now_s - self._current_fresh_y_observation_s
            <= cfg.vertical_qualify_max_age_s
        )
        direction_resolved = bool(
            fresh_y_owned
            and sign != 0
            and abs(motion.physical_rate_norm_s) * cfg.commit_blackout_s
            > motion.bearing_std
        )
        observation_serial = self._current_fresh_y_observation_serial
        is_new_measurement = (
            fresh_y_owned
            and observation_serial
            > self._vertical_direction_last_y_observation_serial
        )
        if is_new_measurement:
            self._vertical_direction_last_y_observation_serial = (
                observation_serial
            )
            if direction_resolved:
                if sign == self._vertical_direction_streak_sign:
                    self._vertical_direction_streak += 1
                else:
                    self._vertical_direction_streak_sign = sign
                    self._vertical_direction_streak = 1
            else:
                self._vertical_direction_streak_sign = 0
                self._vertical_direction_streak = 0

            if (
                direction_resolved
                and self._vertical_direction_streak
                >= cfg.vertical_direction_streak_frames
                and sign != self._vertical_direction_sign
            ):
                self._vertical_direction_sign = sign
                self._vertical_direction_fast_until_s = (
                    now_s + cfg.vertical_direction_fast_window_s
                )

        if (
            direction_resolved
            and sign == self._vertical_direction_sign
            and self._vertical_direction_streak
            >= cfg.vertical_direction_streak_frames
        ):
            self._vertical_direction_supported = True
            self._vertical_direction_source = "coherent_motion"

    def _vertical_collective_target(
        self,
        current: Optional[_Hypothesis],
        support: float,
        motion: Optional[_PassageMotion],
        *,
        supporting_direction_sign: int = 0,
    ) -> float:
        """One optical correction plus bounded, non-vetoing IMU damping."""

        cfg = self.config
        visual_delta = 0.0
        visually_clear = False
        if current is not None and motion is not None:
            ramp = self._course_range_ramp(current)
            error_cap = cfg.vertical_optical_error_max_far_norm - (
                cfg.vertical_optical_error_max_far_norm
                - cfg.vertical_optical_error_max_near_norm
            ) * ramp
            if self._vertical_direction_supported:
                if motion.directional_censor != FrameEdge.NONE:
                    # A one-sided bound owns direction immediately.  The cap
                    # keeps its response inside the same optical envelope.
                    magnitude = min(abs(motion.bearing_error), error_cap)
                else:
                    # Magnitude uncertainty is handled conservatively without
                    # discarding a coherent sign: use the smaller complete
                    # model endpoint, not their uncertain blend/covariance.
                    magnitude = min(
                        abs(motion.fallback_intercept_error),
                        abs(motion.optical_intercept_error),
                        error_cap,
                    )
                # Direction remains owned while valid, but correction
                # magnitude fades to zero across the bounded prediction gap.
                # This prevents an established sign from turning a frozen
                # frame into full stale collective authority.
                magnitude *= motion.freshness_authority
                self._vertical_direction_magnitude = magnitude
                visual_delta = (
                    -cfg.vertical_optical_collective_gain
                    * self._vertical_direction_sign
                    * magnitude
                )
                visually_clear = magnitude > 0.0
            else:
                # F162 baseline away from a resolved near-plane direction:
                # covariance continuously fades the blended optical miss.
                bounded_miss = _clamp(
                    motion.intercept_error, -error_cap, error_cap
                )
                visual_delta = (
                    -cfg.vertical_optical_collective_gain
                    * motion.control_authority
                    * bounded_miss
                )
                visually_clear = bool(
                    motion.directional_censor != FrameEdge.NONE
                    or abs(motion.intercept_error) > motion.intercept_std
                )

        if (
            supporting_direction_sign != 0
            and not self._vertical_direction_supported
            and supporting_direction_sign * visual_delta > 0.0
        ):
            visual_delta = 0.0
        imu_delta = -cfg.vertical_imu_damping_gain * self._vz_est_m_s
        if (
            self._vertical_direction_supported
            and visually_clear
            and self._vertical_direction_sign * imu_delta > 0.0
        ):
            # IMU vz is supporting damping, never a visual veto.  In F162 the
            # leaky estimate still said "sinking" and added climb after the
            # coherent optical trajectory already required descent.
            imu_delta = 0.0
        elif (
            supporting_direction_sign != 0
            and supporting_direction_sign * imu_delta > 0.0
        ):
            # Far from the plane, fresh outer geometry supplies direction only.
            # It may shape energy around tilt support, but it never supplies a
            # metric vertical velocity or overrides near-plane optical motion.
            imu_delta = 0.0
        elif visually_clear and visual_delta * imu_delta < 0.0:
            imu_delta = math.copysign(
                min(
                    abs(imu_delta),
                    cfg.vertical_imu_max_opposition_fraction
                    * abs(visual_delta),
                ),
                imu_delta,
            )
        target = support + visual_delta + imu_delta
        self._last_vertical_support = float(support)
        self._last_vertical_visual_delta = float(visual_delta)
        self._last_vertical_imu_delta = float(imu_delta)
        self._last_vertical_collective_target = float(target)
        return target

    def _lateral_intercept_reference(
        self,
        current: _Hypothesis,
        successor: Optional[_Hypothesis],
        *,
        successor_authority: float,
        now_s: float,
        dt: float,
    ) -> float:
        """Continuous outer-owned roll reference plus safe coherent preturn."""

        current_error, current_axis, current_age, current_edge = (
            self._horizontal_control_observable(current, now_s)
        )
        current_motion = self._passage_motion(
            current,
            current_axis,
            current_error,
            now_s=now_s,
            measurement_age_s=current_age,
            directional_censor=current_edge,
        )
        self._last_lateral_motion = current_motion
        desired = current_error + current_motion.control_authority * (
            current_motion.intercept_error - current_error
        )
        if current_edge & FrameEdge.RIGHT:
            desired = max(desired, current_error, 0.0)
        elif current_edge & FrameEdge.LEFT:
            desired = min(desired, current_error, 0.0)

        # The same safe-passage authority that moves yaw may preturn the
        # physical path.  Before that lease exists, successor influence is
        # exactly zero in both channels; afterward the blend is coherent.
        authority = _clamp01(successor_authority)
        if successor is not None and authority > 0.0:
            successor_error, successor_axis, successor_age, successor_edge = (
                self._horizontal_control_observable(successor, now_s)
            )
            successor_motion = self._passage_motion(
                successor,
                successor_axis,
                successor_error,
                now_s=now_s,
                measurement_age_s=successor_age,
                directional_censor=successor_edge,
            )
            successor_desired = successor_error + (
                successor_motion.control_authority
                * (successor_motion.intercept_error - successor_error)
            )
            desired = (
                (1.0 - authority) * desired
                + authority * successor_desired
            )
        alpha = _clamp01(
            dt / max(1e-6, self.config.turn_reference_tau_s + dt)
        )
        if self._lateral_intercept_reference_x is None:
            self._lateral_intercept_reference_x = float(desired)
        else:
            self._lateral_intercept_reference_x += alpha * (
                float(desired) - self._lateral_intercept_reference_x
            )
        self._lateral_intercept_reference_x = _clamp(
            self._lateral_intercept_reference_x, -1.0, 1.0
        )
        # One-sided current evidence takes directional ownership immediately.
        # The underlying magnitude remains filtered and the final roll target
        # keeps its existing slew/bounds, but stale opposite aperture state can
        # never command the wrong image side for another tick.
        if current_edge & FrameEdge.RIGHT:
            self._lateral_intercept_reference_x = max(
                self._lateral_intercept_reference_x, 0.0
            )
        elif current_edge & FrameEdge.LEFT:
            self._lateral_intercept_reference_x = min(
                self._lateral_intercept_reference_x, 0.0
            )
        return self._lateral_intercept_reference_x

    def _course_range_ramp(self, current: _Hypothesis) -> float:
        """Continuous far-to-crossing optical authority transfer."""

        cfg = self.config
        return _clamp01(
            (current.outer_log_scale - cfg.closure_far_log_scale)
            / (
                cfg.commit_min_log_scale
                - cfg.closure_far_log_scale
            )
        )

    def _course_steer_gain(self, current: _Hypothesis) -> float:
        """One continuous far-to-crossing steering gain for every state."""

        cfg = self.config
        return 1.0 + (
            cfg.near_plane_steer_gain_mult - 1.0
        ) * self._course_range_ramp(current)

    def _continuous_collective(
        self, target: float, dt: float, *, now_s: float
    ) -> float:
        """Carry one bounded collective, with a bounded direction reversal."""

        target = _clamp(target, self.config.min_thrust, self.config.max_thrust)
        if self._collective is None:
            self._collective = target
        else:
            direction_delta = target - self._collective
            direction_matches = bool(
                self._vertical_direction_supported
                and (
                    (
                        self._vertical_direction_sign > 0
                        and direction_delta < 0.0
                    )
                    or (
                        self._vertical_direction_sign < 0
                        and direction_delta > 0.0
                    )
                )
            )
            fast_active = bool(
                direction_matches
                and self._vertical_direction_fast_until_s is not None
                and now_s <= self._vertical_direction_fast_until_s
            )
            alpha = _clamp01(
                dt
                / max(
                    1e-6,
                    self.config.collective_decay_tau_s + dt,
                )
            )
            filtered_step = alpha * direction_delta
            if fast_active:
                limit = (
                    self.config.vertical_direction_fast_slew_per_s * dt
                )
                fast_step = _clamp(
                    direction_delta, -limit, limit
                )
                # This path removes response latency; it must never make an
                # already-faster ordinary filter slower for a large reversal.
                self._collective += (
                    fast_step
                    if abs(fast_step) > abs(filtered_step)
                    else filtered_step
                )
            else:
                self._collective += filtered_step
        self._collective = _clamp(
            self._collective, self.config.min_thrust, self.config.max_thrust
        )
        return self._collective

    def _governed_collective(
        self,
        collective: float,
        support: float,
        gate_y: Optional[float] = None,
    ) -> float:
        """Apply the remaining crossing envelope to one collective target."""

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
            and self.current.outer_log_scale
            >= self.config.closure_min_log_scale
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
        self._reset_vertical_direction()
        self._clear_current_fresh_observation()
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
        directional_censor: FrameEdge = FrameEdge.NONE,
    ) -> float:
        # A fresh LEFT/RIGHT clip is a one-sided current-gate constraint, not
        # an uncertain scalar measurement.  It must take sign ownership on its
        # first frame: carrying an opposite roll target through the ordinary
        # slew window recreated F164's wrong-side interception even after yaw
        # had correctly changed direction.  Neutralize only the contradicted
        # sign, then apply the existing bounded slew toward the requested bank.
        horizontal_edge = directional_censor & (
            FrameEdge.LEFT | FrameEdge.RIGHT
        )
        if (
            horizontal_edge & FrameEdge.RIGHT
            and target >= 0.0
            and self._prev_target_roll < 0.0
        ) or (
            horizontal_edge & FrameEdge.LEFT
            and target <= 0.0
            and self._prev_target_roll > 0.0
        ):
            self._prev_target_roll = 0.0
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


def _aperture_track_measurement(
    track: Any,
) -> Optional[Tuple[Tuple[float, float], float, Tuple[float, float, float]]]:
    """Return one valid inner-aperture measurement without an outer fallback."""

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
    return None


def _outer_track_measurement(
    track: Any,
) -> Tuple[Tuple[float, float], float, Tuple[float, float, float]]:
    """Return only the outer-box observable used by range and closure.

    F162/F163 alternated inner-aperture and outer-box center/scale in one
    derivative state.  This function is intentionally incapable of seeing an
    aperture, making that modality switch impossible at the real adapter.
    """

    center = track.center_norm
    log_scale = math.log(max(1e-6, float(track.apparent_scale)))
    return (
        (float(center[0]), float(center[1])),
        log_scale,
        (OUTER_MEAS_STD_NORM, OUTER_MEAS_STD_NORM, SCALE_MEAS_STD),
    )


def _track_measurement(
    track: Any,
) -> Tuple[Tuple[float, float], float, Tuple[float, float, float]]:
    """Compatibility seam for callers that need the passage aim sample.

    Production filtering calls the explicit aperture and outer adapters
    separately.  This helper preserves the public test seam without allowing
    its result to enter the outer derivative state.
    """

    aperture = _aperture_track_measurement(track)
    return aperture if aperture is not None else _outer_track_measurement(track)


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
            "outer_bearing": [
                successor.outer_x_axis.p,
                successor.outer_y_axis.p,
            ],
            "log_scale": successor.log_scale,
            "outer_log_scale": successor.outer_log_scale,
            "confidence": successor.confidence,
            "position_std": successor.position_std,
            "age_s": now_s - successor.last_measurement_s,
            "outer_x_evidence_age_s": (
                now_s - successor.last_outer_x_evidence_s
            ),
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

    def passage_motion_trace(
        motion: Optional[_PassageMotion],
    ) -> Optional[Dict[str, Any]]:
        if motion is None:
            return None
        return {
            "bearing_error": motion.bearing_error,
            "physical_rate_norm_s": motion.physical_rate_norm_s,
            "closure_rate_s": motion.closure_rate_s,
            "closure_std_s": motion.closure_std_s,
            "ttc_s": motion.ttc_s,
            "ttc_std_s": motion.ttc_std_s,
            "projection_authority": motion.projection_authority,
            "fallback_intercept_error": motion.fallback_intercept_error,
            "optical_intercept_error": motion.optical_intercept_error,
            "intercept_error": motion.intercept_error,
            "bearing_std": motion.bearing_std,
            "intercept_std": motion.intercept_std,
            "freshness_authority": motion.freshness_authority,
            "measurement_authority": motion.measurement_authority,
            "control_authority": motion.control_authority,
            "uncertainty_weighted_miss": (
                motion.control_authority * motion.intercept_error
            ),
            "directional_censor": int(motion.directional_censor),
        }

    current_trace = None
    if current is not None:
        control_heading_x, _heading_axis, _heading_age, _heading_edge = (
            controller._horizontal_control_observable(current, now_s)
        )
        control_closure, control_closure_agreement, control_closure_std = (
            controller._control_closure_estimate(current, now_s)
        )
        corridor = controller._transported_corridor(current, now_s=now_s)
        corridor_trace = None
        if corridor is not None:
            corridor_trace = {
                "track_id": corridor.track_id,
                "gate_index": corridor.gate_index,
                "frame_identity": corridor.frame_identity,
                "source_age_s": corridor.source_age_s,
                "center_norm": [corridor.center_x, corridor.center_y],
                "half_size_norm": [corridor.half_x, corridor.half_y],
                "center_std_norm": [
                    corridor.center_std_x,
                    corridor.center_std_y,
                ],
                "live": corridor.live,
            }
        current_trace = {
            "track_id": current.track_id,
            "bearing": [current.x, current.y],
            "raw_bearing": [current.raw_x, current.raw_y],
            "image_rate_norm_s": [current.vx, current.vy],
            "axis_std": [current.x_axis.std, current.y_axis.std],
            "passage_source": current.passage_source,
            "outer_bearing": [
                current.outer_x_axis.p,
                current.outer_y_axis.p,
            ],
            "outer_raw_bearing": [current.outer_raw_x, current.outer_raw_y],
            "outer_image_rate_norm_s": [
                current.outer_x_axis.v,
                current.outer_y_axis.v,
            ],
            "outer_axis_std": [
                current.outer_x_axis.std,
                current.outer_y_axis.std,
            ],
            "log_scale": current.log_scale,
            "control_expansion_rate_s": current.expansion_rate,
            "last_aperture_scale_age_s": (
                now_s - current.last_aperture_scale_measurement_s
            ),
            "outer_log_scale": current.outer_log_scale,
            "outer_filtered_log_scale": current.outer_filtered_log_scale,
            "outer_filtered_expansion_rate_s": (
                current.outer_filtered_expansion_rate
            ),
            "expansion_rate_s": current.expansion_rate,
            "outer_expansion_rate_s": current.outer_expansion_rate,
            "control_closure_rate_s": control_closure,
            "control_closure_agreement": control_closure_agreement,
            "control_closure_std_s": control_closure_std,
            "control_heading_x": control_heading_x,
            "aperture_half_size_norm": [
                current.aperture_half_x,
                current.aperture_half_y,
            ],
            "measurement_age_s": now_s - current.last_measurement_s,
            "x_measurement_age_s": now_s - current.last_x_measurement_s,
            "y_measurement_age_s": now_s - current.last_y_measurement_s,
            "outer_x_measurement_age_s": (
                now_s - current.last_outer_x_measurement_s
            ),
            "outer_x_evidence_age_s": (
                now_s - current.last_outer_x_evidence_s
            ),
            "horizontal_censor_edge": int(current.horizontal_censor_edge),
            "horizontal_censor_bound": current.horizontal_censor_bound,
            "vertical_censor_bound": current.vertical_censor_bound,
            "corridor": corridor_trace,
        }
    admission = controller._last_commit_admission
    return {
        "state": controller.state.value,
        "state_dwell_s": now_s - state_entry_s,
        "token": token_trace,
        "tracks": tracks,
        "current": current_trace,
        "commit_admission": {
            "admissible": admission.admissible,
            "status": admission.status,
            "corridor_known": admission.corridor_known,
            "corridor_live": admission.corridor_live,
            "corridor_age_s": admission.corridor_age_s,
            "x_tube": admission.x_tube,
            "y_tube": admission.y_tube,
            "x_budget": admission.x_budget,
            "y_budget": admission.y_budget,
            "closure_rate_s": admission.closure_rate_s,
            "closure_agreement": admission.closure_agreement,
            "ttc_s": admission.ttc_s,
            "longitudinal_reachable": admission.longitudinal_reachable,
        },
        "successor": successor_trace,
        "turn_successor_authority": controller._successor_heading_blend,
        "turn_reference_x": controller._successor_heading_error_norm,
        "target_roll_rad": controller._prev_target_roll,
        "target_pitch_rad": controller._prev_target_pitch,
        "lateral_intercept_reference_x": (
            controller._lateral_intercept_reference_x
        ),
        "lateral_passage_motion": passage_motion_trace(
            controller._last_lateral_motion
        ),
        "vertical_passage_motion": passage_motion_trace(
            controller._last_vertical_motion
        ),
        "vertical_control": {
            "direction_sign": controller._vertical_direction_sign,
            "direction_streak": controller._vertical_direction_streak,
            "direction_supported": controller._vertical_direction_supported,
            "direction_source": controller._vertical_direction_source,
            "direction_magnitude": controller._vertical_direction_magnitude,
            "fresh_y_observation_serial": (
                controller._current_fresh_y_observation_serial
            ),
            "fresh_y_observation_age_s": (
                None
                if controller._current_fresh_y_observation_s is None
                else now_s - controller._current_fresh_y_observation_s
            ),
            "fast_response_remaining_s": (
                None
                if controller._vertical_direction_fast_until_s is None
                else max(
                    0.0,
                    controller._vertical_direction_fast_until_s - now_s,
                )
            ),
            "tilt_support": controller._last_vertical_support,
            "visual_delta": controller._last_vertical_visual_delta,
            "imu_delta": controller._last_vertical_imu_delta,
            "target_collective": (
                controller._last_vertical_collective_target
            ),
            "filtered_collective": controller._collective,
            "far_outer_direction_sign": (
                controller._far_vertical_direction_sign
            ),
        },
        "turn_aperture_reserve": controller._turn_aperture_reserve,
        "vertical_censor_edge": (
            None if current is None else int(current.vertical_censor_edge)
        ),
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
        "fh_mps2": controller._fh_mps2,
        "fh_trusted": not controller._fh_untrusted,
    }


async def run_clean_course_stage(
    host: Any,
    context: Any,
    *,
    runtime: CleanCourseRuntime,
    config: Optional[CleanCourseConfig] = None,
    controller: Optional["CleanCourseController"] = None,
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
    if controller is None:
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
