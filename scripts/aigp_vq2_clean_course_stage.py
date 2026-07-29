"""Clean minimal VQ2 visual-course navigation stage (architecture reset M2).

This module replaces the retired ``aigp_vq2_visual_course_stage`` coordinator
as the navigation owner for the powered ``visual-course`` stage.  It carries
exactly four runtime states (``TRACK``, ``PREDICT``, ``COAST_FOR_CREDIT``,
``SEARCH``), one small variable-dt estimator per retained target hypothesis,
one continuous control law, one attitude PD, one explicit yaw channel, one
transparent final clamp, and one atomic race-active send per tick.

Authority model:

- ``active_gate_index`` increments and ``race_finished`` are authoritative.
  They are accepted immediately as events; vision never vetoes race credit
  and never declares a pass.
- ``track_id`` is a local visual-continuity hypothesis only, never a gate
  number.
- The July-18 bounded credible-crossing wait survives as the single
  ``COAST_FOR_CREDIT`` state: after a credible close crossing loses the
  target on a FRESH camera frame, latch zero-rate/zero-thrust and wait at
  most 0.40 s for a strictly newer race packet.  A superseded/frozen frame
  (same camera-frame identity republished during a camera stall) must never
  arm the coast; it goes to ``PREDICT`` with covariance inflation instead
  (flight 20260729T085719Z-visual-course-4455fd61).

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
  descent floor with hover feedforward and a post-credit 0.5 m/s climb
  cap (qualification-gated) extend it; see the constant blocks.
- ``POST_CREDIT_BRAKE_PITCH_RAD`` / ``POST_CREDIT_BRAKE_TIMEOUT_S``: a
  genuine nose-up brake after every authoritative promotion until the
  successor is accepted and vertically qualified (bounded by the timeout),
  added after flights 039186c8/F10 carried gate-0 attack closure into the
  post-credit phase and collapsed thrust effectiveness.
- ``PRE_CROSS_BRAKE_PITCH_RAD`` / ``PRE_CROSS_BRAKE_TTC_S``: the brake must
  START before the plane (codex F9-F11 analysis: post-credit braking alone
  can never kill ~3 m/s inside the ~0.5 s post-credit visibility window),
  so TRACK inside the near/expansion window commands a genuine nose-up
  attitude at the fast brake slew while lateral pursuit and the vz
  governor stay active.  The COAST latch also bypasses the runner's
  attitude PD on the wire so coast sends are exact zeros (F11 sent nonzero
  roll/pitch rates at zero thrust).
- ``FH_UNTRUSTED_*``: the F14 inflow-regime gate.  vz_est is invalidated
  by REGIME (a smooth fh-proportional thrust deficit), not attitude or
  vibration, so sustained fh > 3.0 freezes the vz/alt integrators, blocks
  alt-floor arming, suppresses every vz-based governor floor/cap, and
  falls back to the camera-qualified vertical PD (support + margin when
  unqualified); the latch releases below fh 2.0.  ``EDGE_PARK_*`` caps the
  edge-parked zero-advance dwell (F14's last 4 s) with a forced SEARCH.
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

SUPPORT_COLLECTIVE = 0.275  # GATE0_PROVED_COLLECTIVE_BASE (proved hover support)
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
VZ_LEAK_TAU_S = 2.5  # leaky-integrator time constant (bias/noise guard)
GRAVITY_M_S2 = 9.80665  # ImuAttitudeConfig.gravity_mps2
# Symmetric descent floor (flight 20260729T111003Z-visual-course-d52adcd4):
# a 6.1 s frozen-camera stall blinded the loop while a_up ~= -1.9 m/s^2 sank
# it into a ground graze.  The hover regime shifts ~0.275 -> ~0.30 support
# in fast/descending flight, so from the measured plant (a_up =
# 66.7*thrust - 18.44) the gain adds +0.06 collective at vz = -1.5 m/s
# (~+4 m/s^2 of arrest authority) and tapers to zero at the floor boundary.
VZ_DESCENT_FLOOR_M_S = -0.5  # sink-rate bound, mirroring the climb cap
VZ_DESCENT_GOVERNOR_GAIN = 0.06  # collective per m/s below the floor
# Descent-regime hover feedforward (flight 20260729T112603Z-visual-course-
# d5e89c2b): a ~-0.5 m/s^2 sink persisted ~4 s while the leaky vz estimate
# (tau 2.5 s) wound up and the proportional floor alone reached only ~0.31
# by ground contact; the effective fast-regime hover is ~=0.32.  A fixed
# +0.025 (mid of the diagnosed +0.02..0.03) applies whenever vz is below
# the floor, so full arrest authority arrives with the FIRST confirmed
# sub-floor estimate instead of seconds later.  A shorter downward leak
# tau was rejected: steady-state vz_est = a_up*tau would sit above the
# -0.5 floor and the proportional floor would never engage at all.
VZ_DESCENT_HOVER_FEEDFORWARD = 0.025  # step feedforward while below the floor

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
YAW_ERROR_GAIN = 0.30
# Roll: the old saturated +/-0.25 roll oscillation never recentered either
# (corr(roll_cmd, dx/dt)=+0.18 is too weak/saturated to identify the roll
# channel), so the roll sign follows the yaw verdict: bank INTO the
# correction (positive bank toward a right-side target), coordinated with
# the positive yaw, translating the vehicle toward the gate's lateral
# position so the target bearing moves toward center.
ROLL_ERROR_SIGN = +1.0  # flip this one line if the first flight contradicts
ROLL_ERROR_GAIN = 0.24
MAX_TARGET_ROLL_RAD = 0.12  # GATE1_RECENTER_ROLL cap
# Raised 0.15 -> 0.25 (flights 4ba3922b/89a175a9/d058b8a0): accepted gate-1
# tracks repeatedly slid to the x ~= 0.95 frame edge with yaw pinned at the
# cap while the v3 authority profile measured ~0.5 rad/s of plant capability.
# Bearing rates of near off-axis gates at surviving closure exceed 0.15 rad/s.
# 0.25 sits at the runner's hard MAX_COMMAND_RATE_RAD_S wire clamp and inside
# the measured-authority envelope the runner now checks against.
MAX_COURSE_YAW_RATE_RAD_S = 0.25  # runner wire clamp is 0.25

# Softened -0.18 -> -0.12 (flight 4ba3922b): the whole gate-0 transit is
# ~2 s, so the advance attitude builds most of the closure the pre-cross
# brake must then kill, and it arrives too late to kill it all (crossing
# span grew 0.72 -> 0.89 in 0.14 s).  A gentler advance trades ~1 s of
# approach time for a crossing speed the brake can actually manage.
ADVANCE_PITCH_RAD = -0.12  # nose-down closure target when aligned/confident
BRAKE_PITCH_RAD = -0.02  # near-level braking target
ANGULAR_FULL_BRAKE_NORM = 0.60  # angular error that fully suppresses advance
EXPANSION_BRAKE_FREE_S = 1.5  # expansion rate below which no braking applies
EXPANSION_BRAKE_SPAN_S = 3.0  # span from free advance to full expansion brake
NEAR_FREE_LOG_SCALE = -1.5  # far enough that near-plane risk does not brake
NEAR_BRAKE_LOG_SCALE = -0.9  # close enough that closure is fully braked

CROSSING_MIN_LOG_SCALE = -0.80  # retired stage crossing_arm_min_log_scale
CROSSING_CREDIT_WAIT_S = 0.40  # July-18 safety contract item 9

PREDICT_FRAME_GAP_S = 0.06  # ~2 camera frames without a measurement
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
# Post-credit brake (flights 20260729T114842Z-visual-course-039186c8 and
# F10): gate 0 is crossed at ~3+ m/s closure, and carrying that attack
# speed into the post-credit phase (a) collapsed thrust effectiveness in
# the fast/VRS-like regime (measured vz_est derivative ~-0.5 m/s^2 with
# thrust pinned at the clamp; effective hover 0.335-0.36) and (b) pushed
# near off-axis gate-1 bearing rates past the 0.15 rad/s yaw cap, sliding
# two accepted gate-1 tracks to x ~= 0.96 and losing them.  After every
# authoritative promotion the stage therefore actively pitches back until
# the successor is accepted AND vertically qualified, with a bounded
# timeout so a lost gate cannot brake forever.  Pitch sign: NEGATIVE is
# nose-down forward advance (ADVANCE_PITCH_RAD = -0.18), so a genuine
# brake is POSITIVE nose-up; +0.12 sits well inside the +/-0.25 pitch cap.
# Side benefit: pitch-back tilts the camera up toward gate 1's known
# high-first-sight position.  The climb cap also tightens to 0.5 m/s for
# the same unqualified window (F10 climbed at vz +1.0 for ~1.4 s chasing
# an unqualified low-conf bearing, spending ~0.7 m of altitude); the full
# 1.0 m/s cap returns the moment vertical is qualified.
POST_CREDIT_BRAKE_PITCH_RAD = 0.18  # nose-up brake attitude (see block above)
POST_CREDIT_BRAKE_TIMEOUT_S = 2.75  # bounded brake even with no reacquisition
POST_CREDIT_CLIMB_CAP_M_S = 0.5  # climb cap while post-credit unqualified
# Minimum brake hold (flight 20260729T125400Z-visual-course-4480d0a6): gate 1
# is often already accepted AND vertically qualified at the credit tick, so
# the qualification release fired within one 20 ms tick and the brake never
# engaged — the flag stayed False for the entire F11 trace and the attack
# closure was never killed.  The brake now holds for at least this long
# regardless of qualification; qualification only releases it afterwards.
# 1.0 -> 2.0 s (flight 20260729T125958Z-visual-course-d058b8a0): the F12
# brake held its 1.0 s but released before the slew-limited attitude change
# had killed any closure (track 0006 span grew x3.8 in the next 1.6 s).
POST_CREDIT_BRAKE_MIN_HOLD_S = 2.0
# Dedicated brake slew (same flight): the generic 0.30 rad/s target slew
# moved pitch only from -0.085 to ~=0 inside the 1.0 s F12 hold, so the
# brake attitude was never attained and closure was never killed.  At
# 1.0 rad/s the worst-case swing (advance -0.18 to brake +0.18) lands in
# ~0.36 s of window start.  Applies ONLY while the brake window is active;
# normal steering keeps the transparent 0.30 rad/s slew.
POST_CREDIT_BRAKE_SLEW_RAD_S = 1.0
# Pre-crossing expansion brake (independent codex analysis of the F9-F11
# traces): post-credit braking alone can never be robust — even +0.12 rad
# pitch-back yields only g*tan(0.12) ~= 1.18 m/s^2, so killing a ~3 m/s
# attack closure needs ~2.5 s while the gate disappears ~0.5 s after
# credit.  The brake therefore STARTS before the plane: while TRACKing the
# current gate inside the near window (log_scale at/past
# NEAR_BRAKE_LOG_SCALE), or with the filtered expansion rate saying
# time-to-contact below ~2.5 s, the stage commands a genuine nose-up
# attitude at the fast brake slew so the crossing happens at ~1-1.5 m/s
# instead of 3+.  Timing (agent-10, F13): at 3 m/s closure TTC 1.2 s IS
# log_scale -1.1, inside the old near field — the old trigger could never
# create braking distance.  TTC 2.5 s fires at log_scale ~= -1.6...-1.8,
# buying ~0.9-1.0 s of genuine brake, so the expansion trigger's near-field
# gate moved out to -1.8 to let the earlier TTC actually bind.  Lateral
# pursuit and the vz governor stay fully active;
# the altitude floor and the exact-zero COAST latch still preempt.  The
# near threshold stays below CROSSING_MIN_LOG_SCALE so apparent size keeps
# growing through the crossing-arm scale and the engulfing/COAST detection
# fires unchanged under braking.  The continuous near/expansion derate
# (advance -> brake_pitch_rad) remains as the far-field shaping; the
# post-credit brake window is unchanged and becomes the cleanup for
# residual closure.
PRE_CROSS_BRAKE_PITCH_RAD = 0.12  # genuine nose-up pre-plane brake attitude
PRE_CROSS_BRAKE_TTC_S = 2.5  # expansion-rate time-to-contact trigger (F13)
PRE_CROSS_BRAKE_NEAR_LOG_SCALE = -1.8  # near-field gate for the TTC trigger
PRE_CROSS_BRAKE_SLEW_RAD_S = 1.0  # fast slew, shared with the brake window
# Pre-gate-1 altitude floor (terrain insurance; flights F10/F11/F12 all
# flew their final 6-10 s below 0.7 m with thrust pinned at the clamp into
# terrain hits).  alt_est integrates the governor's IMU vz_est from course
# start (takeoff pad = 0); drift is bounded inside this <15 s
# gate-0-credit -> gate-1-credit window.  Hysteresis: trigger below 0.7 m,
# release above 1.2 m.  Inactive at gate_index 0 (takeoff/climb-out) and
# once gate_index >= 2 — post-gate-1 reference re-anchoring is a follow-up.
# F13 bounds (trace 20260729T134958Z-visual-course-82d72cb5): a biased
# estimator (vz_est -3.8, alt_est -10.7 m, physically impossible) latched
# the floor at t=5.16 and pinned the profile at full thrust into terrain
# for 4.2 s.  The floor stays, but it can no longer pin the profile: an
# episode releases unconditionally after ALT_FLOOR_MAX_LATCH_S and re-arms
# only after alt_est has held above the release altitude for
# ALT_FLOOR_REARM_S, and alt_est is clamped below at ALT_EST_MIN_M so a
# biased integrator cannot push the floor deeper than its 0.7-1.2 m guard
# band ever needs.
ALT_FLOOR_TRIGGER_M = 0.7
ALT_FLOOR_RELEASE_M = 1.2
ALT_FLOOR_CLIMB_MARGIN = 0.05  # support + margin -> governed recovery climb
ALT_FLOOR_MAX_LATCH_S = 2.5  # unconditional per-episode release (F13)
ALT_FLOOR_REARM_S = 1.0  # alt_est must hold above release this long to re-arm
ALT_EST_MIN_M = -2.0  # biased-integrator clamp on the altitude estimate
# F14 inflow-regime gate (agent-10, verified with the actuator/estimator
# tick fields): identical delivered rotor output 0.34 gives accz -13.0 in
# the slow regime (real climb, t=2.59) and -8.0 in the fast regime (t=8.5)
# — vz_est is invalidated by REGIME, a smooth fh-proportional DC deficit
# (~0.9*fh - 0.5), NOT vibration (accz std 0.19, the cleanest of the
# flight) and NOT attitude (kp=0, gyro drift <= 0.2 rad).  Trusted fh <
# ~2.0, biased fh > ~3.0.  While untrusted the stage freezes vz_est (the
# leak relaxes it toward 0; the biased a_up is never integrated), holds
# alt_est, blocks alt-floor arming (an active latch still times out
# normally), falls back to the camera-qualified vertical PD (support +
# margin when unqualified — bare support historically sinks for real at
# -0.8...-1.9), and suppresses every vz-based governor floor/cap so the
# descent feedforward cannot fire from the frozen estimate.  This breaks
# the F14 self-locking loop: governor pinned 0.34 on the phantom sink, the
# floor flew biased-"level".
FH_UNTRUSTED_TRIGGER_MPS2 = 3.0  # biased regime above this horizontal force
FH_TRUSTED_RELEASE_MPS2 = 2.0  # hysteresis release below this
FH_UNTRUSTED_SUSTAIN_S = 0.3  # transients shorter than this never latch
FH_UNTRUSTED_VERTICAL_MARGIN = 0.02  # unqualified hold: support + margin
# Edge-parked advance stall (F14, agent-10 Q5): a track parked at the frame
# edge (angular error at/past angular_full_brake_norm) forces align = 0 ->
# advance = 0 -> perpetual near-level pitch.  F14 chased edge-parked tracks
# for its last 4 s, never resumed advance, and held fh ~= 6 with the regime
# gate latched.  Cap the dwell: parked this long without the track
# re-centering (angular error back below 0.5*norm) or log_scale growing
# (approach progress) forces SEARCH so the sweep reacquires a centered
# track.  The align/advance law itself is unchanged.
EDGE_PARK_MAX_DWELL_S = 1.5
EDGE_PARK_PROGRESS_LOG_SCALE = 0.05  # log_scale growth that re-anchors dwell
SEARCH_COVARIANCE_STD_NORM = 0.35  # position std that forces SEARCH
SEARCH_YAW_RATE_RAD_S = 0.12  # bounded sweep inside the 0.15 yaw cap
SEARCH_SWEEP_PERIOD_S = 1.20  # bounded reversal schedule
SEARCH_MAX_EXCURSION_RAD = 0.80  # bounded sweep excursion before reversal

SUCCESSOR_BLEND_MAX = 0.50  # continuous lookahead ceiling
BLEND_FAR_LOG_SCALE = -1.6  # below this the successor gets no blend
BLEND_NEAR_LOG_SCALE = -0.9  # at this closure the blend ceiling applies
PROMOTE_MAX_STD_NORM = 0.30  # cached-successor credibility at promotion
PROMOTE_MAX_AGE_S = 0.50  # cached-successor freshness at promotion

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
ROTATION_COMP_FOCAL_NORM = 1.0  # normalized focal length for de-rotation
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
    """The exactly four runtime states of the clean course stage."""

    TRACK = "track"
    PREDICT = "predict"
    COAST_FOR_CREDIT = "coast_for_credit"
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
        "clipped",
        "created_s",
        "last_measurement_s",
        "last_x_measurement_s",
        "last_y_measurement_s",
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
        self.clipped = False
        self.created_s = float(now_s)
        self.last_measurement_s = float(now_s)
        self.last_x_measurement_s = float(now_s)
        self.last_y_measurement_s = float(now_s)

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
    min_thrust: float = MIN_COURSE_THRUST
    max_thrust: float = MAX_COURSE_THRUST
    launch_boost_thrust: float = LAUNCH_BOOST_THRUST
    launch_boost_duration_s: float = LAUNCH_BOOST_DURATION_S
    gate0_climb_vertical_offset_norm: float = GATE0_CLIMB_VERTICAL_OFFSET_NORM
    gate0_climb_reference_log_scale: float = GATE0_CLIMB_REFERENCE_LOG_SCALE
    roll_error_sign: float = ROLL_ERROR_SIGN
    roll_error_gain: float = ROLL_ERROR_GAIN
    max_target_roll_rad: float = MAX_TARGET_ROLL_RAD
    yaw_error_sign: float = YAW_ERROR_SIGN
    yaw_error_gain: float = YAW_ERROR_GAIN
    max_yaw_rate_rad_s: float = MAX_COURSE_YAW_RATE_RAD_S
    advance_pitch_rad: float = ADVANCE_PITCH_RAD
    brake_pitch_rad: float = BRAKE_PITCH_RAD
    angular_full_brake_norm: float = ANGULAR_FULL_BRAKE_NORM
    expansion_brake_free_s: float = EXPANSION_BRAKE_FREE_S
    expansion_brake_span_s: float = EXPANSION_BRAKE_SPAN_S
    near_free_log_scale: float = NEAR_FREE_LOG_SCALE
    near_brake_log_scale: float = NEAR_BRAKE_LOG_SCALE
    crossing_min_log_scale: float = CROSSING_MIN_LOG_SCALE
    crossing_credit_wait_s: float = CROSSING_CREDIT_WAIT_S
    predict_frame_gap_s: float = PREDICT_FRAME_GAP_S
    predict_max_gap_s: float = PREDICT_MAX_GAP_S
    post_credit_brake_pitch_rad: float = POST_CREDIT_BRAKE_PITCH_RAD
    post_credit_brake_timeout_s: float = POST_CREDIT_BRAKE_TIMEOUT_S
    post_credit_climb_cap_m_s: float = POST_CREDIT_CLIMB_CAP_M_S
    post_credit_brake_min_hold_s: float = POST_CREDIT_BRAKE_MIN_HOLD_S
    post_credit_brake_slew_rad_s: float = POST_CREDIT_BRAKE_SLEW_RAD_S
    pre_cross_brake_pitch_rad: float = PRE_CROSS_BRAKE_PITCH_RAD
    pre_cross_brake_ttc_s: float = PRE_CROSS_BRAKE_TTC_S
    pre_cross_brake_near_log_scale: float = PRE_CROSS_BRAKE_NEAR_LOG_SCALE
    pre_cross_brake_slew_rad_s: float = PRE_CROSS_BRAKE_SLEW_RAD_S
    alt_floor_trigger_m: float = ALT_FLOOR_TRIGGER_M
    alt_floor_release_m: float = ALT_FLOOR_RELEASE_M
    alt_floor_climb_margin: float = ALT_FLOOR_CLIMB_MARGIN
    alt_floor_max_latch_s: float = ALT_FLOOR_MAX_LATCH_S
    alt_floor_rearm_s: float = ALT_FLOOR_REARM_S
    alt_est_min_m: float = ALT_EST_MIN_M
    fh_untrusted_trigger_mps2: float = FH_UNTRUSTED_TRIGGER_MPS2
    fh_trusted_release_mps2: float = FH_TRUSTED_RELEASE_MPS2
    fh_untrusted_sustain_s: float = FH_UNTRUSTED_SUSTAIN_S
    fh_untrusted_vertical_margin: float = FH_UNTRUSTED_VERTICAL_MARGIN
    edge_park_max_dwell_s: float = EDGE_PARK_MAX_DWELL_S
    edge_park_progress_log_scale: float = EDGE_PARK_PROGRESS_LOG_SCALE
    search_covariance_std_norm: float = SEARCH_COVARIANCE_STD_NORM
    search_yaw_rate_rad_s: float = SEARCH_YAW_RATE_RAD_S
    search_sweep_period_s: float = SEARCH_SWEEP_PERIOD_S
    search_max_excursion_rad: float = SEARCH_MAX_EXCURSION_RAD
    successor_blend_max: float = SUCCESSOR_BLEND_MAX
    blend_far_log_scale: float = BLEND_FAR_LOG_SCALE
    blend_near_log_scale: float = BLEND_NEAR_LOG_SCALE
    promote_max_std_norm: float = PROMOTE_MAX_STD_NORM
    promote_max_age_s: float = PROMOTE_MAX_AGE_S
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
        self.successor_bearing_cache: Dict[int, Tuple[float, float]] = {}
        self._course_start_s: Optional[float] = None
        self._last_observe_s: Optional[float] = None
        self._last_command_s: Optional[float] = None
        self._collective: Optional[float] = None
        self._prev_target_roll = 0.0
        self._prev_target_pitch = BRAKE_PITCH_RAD
        self._coast_entry_s: Optional[float] = None
        self._coast_race_boot_ms: Optional[int] = None
        self._last_race_boot_ms: Optional[int] = None
        self._search_direction = 1.0
        self._search_elapsed_s = 0.0
        self._search_excursion_rad = 0.0
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
        # Post-credit brake window (flights 039186c8/F10): deadline set on
        # every authoritative promotion; released early when the successor
        # is accepted and vertically qualified.  While active, command()
        # pitches back genuinely and tightens the climb cap.
        self._post_credit_deadline_s: Optional[float] = None
        self._post_credit_armed_s: Optional[float] = None
        # IMU altitude estimate (m) integrated from the governor's vz_est,
        # seeded 0 at course start (takeoff pad reference) and clamped below
        # at alt_est_min_m (F13).  Guards only the
        # bounded pre-gate-1 window; see the ALT_FLOOR_* constant block.
        self._alt_est_m = 0.0
        self._alt_floor_active = False
        # F13 latch bounds: the episode start for the unconditional release,
        # the post-timeout cooldown, and the continuous-above-release timer
        # that clears the cooldown (re-arm).
        self._alt_floor_latch_s: Optional[float] = None
        self._alt_floor_cooldown = False
        self._alt_floor_above_release_since_s: Optional[float] = None
        self._active_climb_cap_m_s = VZ_CLIMB_CAP_M_S
        # Pre-crossing expansion brake latch for the tick trace; recomputed
        # every main-path tick (see the PRE_CROSS_BRAKE_* constant block).
        self._pre_cross_brake_active = False
        # F14 inflow-regime gate state (see the FH_* constant block):
        # sustained-high-fh timer, latched untrusted flag, and the last fh
        # seen (0.0 = trusted until the first host estimate arrives).
        self._fh_mps2 = 0.0
        self._fh_untrusted = False
        self._fh_above_since_s: Optional[float] = None
        # Edge-parked advance-stall dwell (see the EDGE_PARK_* block).
        self._edge_park_since_s: Optional[float] = None
        self._edge_park_log_scale: Optional[float] = None

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
            self.last_reliable_bearing = (self.current.x, self.current.y)
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
            self.last_reliable_bearing = (
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
            self.last_reliable_bearing = (
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
        if match is not None:
            self._update_hypothesis(self.current, match, now_s)
            self.state = CleanCourseState.TRACK
        elif self.state is CleanCourseState.SEARCH or self.current is None:
            adopted = self._select_search_reacquisition(tracks)
            if adopted is not None:
                self.current = self._hypothesis_from_track(adopted, now_s)
                self.state = CleanCourseState.TRACK
        else:
            gap = now_s - self.current.last_measurement_s
            if (
                self.state is CleanCourseState.TRACK
                and fresh
                and self.current.outer_log_scale >= cfg.crossing_min_log_scale
            ):
                # Credible close crossing lost the target on a FRESH frame:
                # latch the single bounded credit wait from the July-18
                # contract.  Flight 20260729T085719Z-visual-course-4455fd61:
                # a ~0.27 s camera stall republished one frozen frame id and
                # the stale close-range loss latched zero thrust at the
                # gate-0 top bar, so a superseded frame must never arm this.
                self.state = CleanCourseState.COAST_FOR_CREDIT
                self._coast_entry_s = float(now_s)
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

        self._refresh_successor(tracks, now_s)
        if self.current is not None and match is not None:
            self.last_reliable_bearing = (self.current.x, self.current.y)

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
        if self.state is CleanCourseState.COAST_FOR_CREDIT:
            self._exit_coast()

        successor = self.successor
        credible = (
            successor is not None
            and successor.position_std <= self.config.promote_max_std_norm
            and now_s - successor.last_measurement_s
            <= self.config.promote_max_age_s
        )
        if credible:
            self.current = successor
            self.successor = None
            self.state = CleanCourseState.TRACK
            self.last_reliable_bearing = (self.current.x, self.current.y)
        else:
            self.current = None
            cached = self.successor_bearing_cache.get(self.gate_index)
            if successor is not None:
                self.last_reliable_bearing = (successor.x, successor.y)
            elif cached is not None:
                self.last_reliable_bearing = cached
            self._enter_search(now_s)
        # Re-seed the collective tracker so a retained saturated sub-support
        # command can never survive into the next gate.
        self._collective = None
        # Arm the post-credit brake window (flights 039186c8/F10): kill the
        # gate-0 attack closure before it collapses thrust effectiveness and
        # outruns the yaw cap on the next gate's bearing.
        self._post_credit_deadline_s = (
            float(now_s) + self.config.post_credit_brake_timeout_s
        )
        self._post_credit_armed_s = float(now_s)
        self._active_climb_cap_m_s = self.config.post_credit_climb_cap_m_s
        return True

    # -- the one continuous control law -------------------------------------

    def command(
        self,
        *,
        now_s: float,
        roll_rad: float,
        pitch_rad: float,
        world_up_accel_m_s2: Optional[float] = None,
        horizontal_specific_force_mps2: Optional[float] = None,
    ) -> NavigationOutput:
        """Produce the single navigation request for one tick."""

        cfg = self.config
        self._pre_cross_brake_active = False  # main path recomputes below
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
            assert self._coast_entry_s is not None
            if now_s - self._coast_entry_s > cfg.crossing_credit_wait_s:
                self._exit_coast()
                self._enter_search(now_s)
            else:
                # July-18 bounded credible-crossing wait: exact zero latch.
                # Exact-zero thrust is reserved for this state, abort, and
                # cleanup.
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

        # Edge-parked advance stall (F14, agent-10 Q5; see the EDGE_PARK_*
        # constant block): a track parked at the frame edge forces
        # align = 0 -> advance = 0 -> perpetual near-level pitch.  A dwell
        # this long without re-centering (angular error back below
        # 0.5*norm) or log_scale growth (approach progress) forces SEARCH
        # so the sweep reacquires a centered track.  The align/advance law
        # itself is unchanged.
        if self.state is CleanCourseState.TRACK and self.current is not None:
            edge_angular_error = math.hypot(self.current.x, self.current.y)
            if edge_angular_error >= cfg.angular_full_brake_norm:
                if self._edge_park_since_s is None:
                    self._edge_park_since_s = now_s
                    self._edge_park_log_scale = self.current.log_scale
                elif (
                    self.current.log_scale
                    > self._edge_park_log_scale + cfg.edge_park_progress_log_scale
                ):
                    # Approach progress: re-anchor the dwell window.
                    self._edge_park_since_s = now_s
                    self._edge_park_log_scale = self.current.log_scale
                elif now_s - self._edge_park_since_s > cfg.edge_park_max_dwell_s:
                    self._edge_park_since_s = None
                    self._edge_park_log_scale = None
                    self._enter_search(now_s)
            elif edge_angular_error < 0.5 * cfg.angular_full_brake_norm:
                self._edge_park_since_s = None
                self._edge_park_log_scale = None
        else:
            self._edge_park_since_s = None
            self._edge_park_log_scale = None

        # Post-credit brake window (flights 039186c8/F10): timeout release
        # here (a lost gate cannot brake forever); the qualification release
        # happens in the main path where vertical_qualified is computed.
        # The tighter climb cap applies for the whole unqualified window.
        if (
            self._post_credit_deadline_s is not None
            and now_s >= self._post_credit_deadline_s
        ):
            self._post_credit_deadline_s = None
        self._active_climb_cap_m_s = (
            cfg.post_credit_climb_cap_m_s
            if self._post_credit_deadline_s is not None
            else VZ_CLIMB_CAP_M_S
        )

        support = _clamp(
            cfg.support_collective
            / max(0.85, math.cos(roll_rad) * math.cos(pitch_rad)),
            cfg.min_thrust,
            cfg.max_thrust,
        )

        # Pre-gate-1 altitude floor (F10/F11/F12: the final 6-10 s ran below
        # 0.7 m with thrust pinned into terrain).  Hysteresis 0.7 -> 1.2 m,
        # gate-1 window only; the exact-zero COAST latch above still wins.
        # F13 bounds: an episode releases unconditionally after
        # alt_floor_max_latch_s (a biased estimator pinned the F13 floor for
        # 4.2 s into terrain) and re-arms only after alt_est has held above
        # the release altitude for alt_floor_rearm_s.
        if self.gate_index == 1:
            if self._alt_est_m > cfg.alt_floor_release_m:
                if self._alt_floor_above_release_since_s is None:
                    self._alt_floor_above_release_since_s = now_s
            else:
                self._alt_floor_above_release_since_s = None
            if self._alt_floor_active:
                if self._alt_floor_latch_s is None:
                    self._alt_floor_latch_s = now_s
                timed_out = (
                    now_s - self._alt_floor_latch_s > cfg.alt_floor_max_latch_s
                )
                self._alt_floor_active = (
                    self._alt_est_m <= cfg.alt_floor_release_m and not timed_out
                )
                if not self._alt_floor_active:
                    self._alt_floor_latch_s = None
                    if timed_out:
                        # Only a timeout release needs the re-arm cooldown;
                        # the plain hysteresis release re-arms immediately.
                        self._alt_floor_cooldown = True
            else:
                self._alt_floor_latch_s = None
                if self._alt_floor_cooldown:
                    since = self._alt_floor_above_release_since_s
                    if (
                        since is not None
                        and now_s - since >= cfg.alt_floor_rearm_s
                    ):
                        self._alt_floor_cooldown = False
                if not self._alt_floor_cooldown and not self._fh_untrusted:
                    # Arming is blocked while fh-untrusted (F14: a biased
                    # vz/alt estimate must never START a floor episode); an
                    # already-active latch still times out normally above.
                    self._alt_floor_active = (
                        self._alt_est_m < cfg.alt_floor_trigger_m
                    )
                    if self._alt_floor_active:
                        self._alt_floor_latch_s = now_s
        else:
            self._alt_floor_active = False
            self._alt_floor_latch_s = None
            self._alt_floor_cooldown = False
            self._alt_floor_above_release_since_s = None
        if self._alt_floor_active:
            # Terrain recovery override: level attitude, zero yaw, governed
            # climb collective.  Everything else yields; the governor keeps
            # it inside the thrust envelope.
            return NavigationOutput(
                target_roll_rad=self._slew_roll(0.0, dt),
                target_pitch_rad=self._slew_pitch(0.0, dt),
                yaw_rate_rad_s=0.0,
                thrust=self._governed_collective(
                    support + cfg.alt_floor_climb_margin, support
                ),
                state=self.state,
                gate_index=self.gate_index,
                current_track_id=self._current_track_id(),
                successor_track_id=self._successor_track_id(),
            )

        if self.state is CleanCourseState.SEARCH:
            sweep_yaw = self._search_yaw(dt)
            self._collective = support
            target_roll = self._slew_roll(0.0, dt)
            target_pitch = self._slew_pitch(
                cfg.post_credit_brake_pitch_rad
                if self._post_credit_deadline_s is not None
                else cfg.brake_pitch_rad,
                dt,
                slew_rad_s=(
                    cfg.post_credit_brake_slew_rad_s
                    if self._post_credit_deadline_s is not None
                    else None
                ),
            )
            return NavigationOutput(
                target_roll_rad=target_roll,
                target_pitch_rad=target_pitch,
                yaw_rate_rad_s=sweep_yaw,
                # The IMU climb governor applies here too: vision loss must
                # never disable it.
                thrust=self._governed_collective(support, support),
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
            self._collective = support
            return NavigationOutput(
                target_roll_rad=0.0,
                target_pitch_rad=cfg.brake_pitch_rad,
                yaw_rate_rad_s=0.0,
                thrust=self._governed_collective(support, support),
                state=self.state,
                gate_index=self.gate_index,
            )

        # Continuous successor lookahead: a weak successor reduces the blend,
        # it never zeroes it through a binary authority product.
        blend = self._successor_blend(current, self.successor)
        ex = current.x
        ey = current.y
        if blend > 0.0 and self.successor is not None:
            ex = (1.0 - blend) * current.x + blend * self.successor.x
            ey = (1.0 - blend) * current.y + blend * self.successor.y

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
            if ey >= 0.0:
                # Gate at/below image center: the climb bias may not lift the
                # aim point above center (flight 9d430a40, see block comment).
                vertical_setpoint_offset = min(vertical_setpoint_offset, 0.0)
        vertical_qualified = (
            self.state is CleanCourseState.TRACK
            and now_s - current.last_y_measurement_s
            <= cfg.vertical_qualify_max_age_s
            and current.y_axis.std <= cfg.search_covariance_std_norm
        )
        # Qualification release for the post-credit brake, but only after the
        # minimum hold (flight 4480d0a6): gate 1 is often already qualified at
        # the credit tick, and an instant release made the brake a no-op while
        # the gate-0 attack closure was still carried.
        if (
            vertical_qualified
            and self._post_credit_deadline_s is not None
            and self._post_credit_armed_s is not None
            and now_s - self._post_credit_armed_s
            >= cfg.post_credit_brake_min_hold_s
        ):
            self._post_credit_deadline_s = None
            self._active_climb_cap_m_s = VZ_CLIMB_CAP_M_S
        post_credit_brake = self._post_credit_deadline_s is not None
        # Pre-crossing expansion brake (codex F9-F11 analysis, see the
        # PRE_CROSS_BRAKE_* constant block): the brake must START before the
        # plane.  TRACK-only: PREDICT keeps the continuous derate, and the
        # post-credit window owns the post-plane phase.  The expansion
        # (time-to-contact) trigger is gated to the near field so far-range
        # scale noise cannot stall the approach.
        pre_cross_brake = (
            self.state is CleanCourseState.TRACK
            and not post_credit_brake
            and (
                current.log_scale >= cfg.near_brake_log_scale
                or (
                    current.log_scale >= cfg.pre_cross_brake_near_log_scale
                    and current.expansion_rate * cfg.pre_cross_brake_ttc_s > 1.0
                )
            )
        )
        self._pre_cross_brake_active = pre_cross_brake
        if vertical_qualified:
            bounded_error = _clamp(
                ey - vertical_setpoint_offset,
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
            collective = support + cfg.vertical_feedback_sign * (
                cfg.vertical_error_gain * bounded_error
                + cfg.vertical_rate_gain * bounded_rate
            )
            self._collective = collective
        else:
            # F14: while fh-untrusted the camera is the only honest vertical
            # channel; when it is unqualified, hold support + margin instead
            # of bare support, which historically sinks for real (-0.8...
            # -1.9 m/s against the biased-regime thrust deficit).
            hold = support + (
                cfg.fh_untrusted_vertical_margin if self._fh_untrusted else 0.0
            )
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
        # alive through TRACK, PREDICT, and SEARCH alike.
        collective = self._governed_collective(collective, support)
        thrust = _clamp(collective, cfg.min_thrust, cfg.max_thrust)

        # Lateral: per the 2026-07-29 crossing-geometry analysis, positive
        # image-x error requires POSITIVE yaw (negative yaw rotates the
        # camera left and pushes a right-side target further right) and a
        # coordinated positive bank toward the target.  Both signs are
        # one-line flippable named constants pending first-flight
        # confirmation.  Clipping saturates corrective steering.
        steer_cap = (
            cfg.clipped_steering_fraction if current.clipped else 1.0
        )
        yaw_rate = _clamp(
            cfg.yaw_error_sign * cfg.yaw_error_gain * ex,
            -cfg.max_yaw_rate_rad_s * steer_cap,
            cfg.max_yaw_rate_rad_s * steer_cap,
        )
        target_roll = _clamp(
            cfg.roll_error_sign * cfg.roll_error_gain * ex,
            -cfg.max_target_roll_rad * steer_cap,
            cfg.max_target_roll_rad * steer_cap,
        )

        # Pitch controls closure continuously: advance when aligned and
        # confident, brake progressively with angular error, uncertainty,
        # rapid expansion, or near-plane risk.
        angular_error = math.hypot(ex, ey)
        align = _clamp01(1.0 - angular_error / cfg.angular_full_brake_norm)
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
        if post_credit_brake:
            # Post-credit brake (flights 039186c8/F10): a genuine nose-up
            # pitch-back (positive; ADVANCE_PITCH_RAD = -0.18 is nose-down)
            # to kill the gate-0 attack closure before it collapses thrust
            # effectiveness and outruns the yaw cap.  Lateral yaw/roll
            # pursuit above and the vz governor stay fully active.  Now only
            # the cleanup for residual closure: the pre-crossing brake below
            # owns the approach.
            target_pitch = cfg.post_credit_brake_pitch_rad
        elif pre_cross_brake:
            # Pre-crossing expansion brake (codex F9-F11 analysis): a
            # genuine nose-up attitude in the last ~1-1.5 s before the plane
            # so the drone crosses at ~1-1.5 m/s instead of 3+.  Lateral
            # yaw/roll pursuit above and the vz governor stay fully active;
            # the altitude floor and the COAST latch still preempt.
            target_pitch = cfg.pre_cross_brake_pitch_rad
        else:
            target_pitch = (
                cfg.brake_pitch_rad
                + (cfg.advance_pitch_rad - cfg.brake_pitch_rad) * advance
            )

        return NavigationOutput(
            target_roll_rad=self._slew_roll(target_roll, dt),
            # Both brake regimes get the dedicated fast slew (F12: the
            # generic 0.30 rad/s slew never attained the brake attitude
            # inside the hold); normal steering keeps the transparent slew.
            target_pitch_rad=self._slew_pitch(
                target_pitch,
                dt,
                slew_rad_s=(
                    cfg.post_credit_brake_slew_rad_s
                    if post_credit_brake
                    else (
                        cfg.pre_cross_brake_slew_rad_s
                        if pre_cross_brake
                        else None
                    )
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
        if center_censored or bool(clipping & (FrameEdge.LEFT | FrameEdge.RIGHT)):
            hypothesis.last_x_measurement_s = NEVER_MEASURED_S
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
        if y_censored:
            hypothesis.y_axis.inflate(CENSOR_INFLATE_VAR_NORM)
        else:
            hypothesis.y_axis.update(zy, r_y)
            hypothesis.last_y_measurement_s = float(now_s)
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

    def _refresh_successor(self, tracks: List[Any], now_s: float) -> None:
        current_id = self._current_track_id()
        others = [track for track in tracks if track.track_id != current_id]
        if not others:
            if (
                self.successor is not None
                and now_s - self.successor.last_measurement_s > 1.0
            ):
                self.successor = None
            return
        best = max(others, key=lambda track: float(track.confidence))
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

    def _select_search_reacquisition(self, tracks: List[Any]) -> Optional[Any]:
        """Re-acquisition in SEARCH; the SAME track_id may be re-adopted."""

        if not tracks:
            return None
        if self.current is not None and self.current.track_id is not None:
            same = self._find(tracks, self.current.track_id)
            if same is not None:
                return same
        bx, by = self.last_reliable_bearing
        return min(
            tracks,
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

    def _governed_collective(self, collective: float, support: float) -> float:
        """IMU climb/descent-rate governor: bound collective by estimated vz.

        Applied wherever a nonzero collective is emitted (TRACK, PREDICT,
        SEARCH, and the defensive fallback) so vision loss never disables
        it; the exact-zero COAST/abort latch bypasses it by construction.
        Symmetric: caps collective above the climb cap and floors it below
        the descent floor (flight d52adcd4 sank ~-1.9 m/s^2 into a ground
        graze while the frozen frame suppressed SEARCH).  The climb cap is
        dynamic: the full VZ_CLIMB_CAP_M_S, tightened to the post-credit
        cap while the post-credit brake window is active (F10's unqualified
        post-credit climb).  Below the floor a
        fixed descent-regime hover feedforward (flight d5e89c2b: effective
        fast-regime hover ~=0.32, proportional floor alone reached only
        ~0.31 by contact) steps in with the first confirmed sub-floor
        estimate instead of waiting on the leaky integrator.
        """

        if self._fh_untrusted:
            # F14: vz_est is frozen and regime-biased, so NO vz-based
            # adjustment may fire — not the climb cap, and above all not
            # the descent floor/feedforward, which pinned F14's thrust at
            # the clamp on a phantom -4.36 m/s sink.  Camera-qualified PD
            # output only, hard-clamped to the envelope.
            return _clamp(
                collective, self.config.min_thrust, self.config.max_thrust
            )
        excess = self._vz_est_m_s - self._active_climb_cap_m_s
        if excess > 0.0:
            collective = min(collective, support - VZ_GOVERNOR_GAIN * excess)
        descent_excess = VZ_DESCENT_FLOOR_M_S - self._vz_est_m_s
        if descent_excess > 0.0:
            collective = max(
                collective,
                support
                + VZ_DESCENT_GOVERNOR_GAIN * descent_excess
                + VZ_DESCENT_HOVER_FEEDFORWARD,
            )
        # Hard clamp here, not only at the main-path call site: the SEARCH
        # and defensive-fallback returns emit the governed value directly,
        # and an unclamped deep-sink floor boost (support + gain*excess +
        # feedforward) exceeded the runner's 0.35 envelope abort in flight
        # 20260729T115619Z-visual-course-039186c8.
        return _clamp(collective, self.config.min_thrust, self.config.max_thrust)

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

    def _exit_coast(self) -> None:
        self._coast_entry_s = None
        self._coast_race_boot_ms = None

    def _slew_roll(self, target: float, dt: float) -> float:
        limit = self.config.target_slew_rad_s * dt
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
        # Optional per-call slew rate: the post-credit brake window uses a
        # dedicated faster slew so the brake attitude is actually attained
        # (F12); everything else keeps the transparent target slew.
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


def _track_measurement(
    track: Any,
) -> Tuple[Tuple[float, float], float, Tuple[float, float, float]]:
    """Prefer a valid fitted inner aperture; fall back to the outer bbox.

    Returns ``((x, y), log_scale, (std_x, std_y, std_scale))``.  The outer
    fallback carries larger covariance.  Detector ``estimated_distance`` is a
    placeholder and is never consulted.
    """

    aperture = getattr(track, "inner_aperture", None)
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
        "post_credit_brake": controller._post_credit_deadline_s is not None,
        "pre_cross_brake": controller._pre_cross_brake_active,
        "alt_est_m": controller._alt_est_m,
        "alt_floor_active": controller._alt_floor_active,
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
            roll_rad, pitch_rad, _yaw = estimate.orientation.to_euler()
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
                world_up_accel_m_s2=world_up_accel,
                horizontal_specific_force_mps2=getattr(
                    estimate, "horizontal_specific_force_mps2", None
                ),
            )
            if (
                nav.state is CleanCourseState.COAST_FOR_CREDIT
                and nav.thrust == 0.0
            ):
                # July-18 safety contract: the coast latch is exact zeros on
                # the WIRE.  The attitude PD would trade the zero target
                # attitude against the current attitude and emit NONZERO
                # roll/pitch rates at zero thrust (flight F11: t=2.156 rates
                # (-0.0663,+0.0388,0), t=2.203 (-0.0455,+0.0318,0)), so the
                # genuine coast latch bypasses the PD entirely.  Only this
                # latch qualifies; every other command keeps the PD path.
                command = rt.attitude_rate_command_type(0.0, 0.0, 0.0, 0.0)
            else:
                # One attitude PD for roll/pitch; yaw stays an explicit
                # channel.
                pd_command = rt.attitude_rate_command(
                    estimate,
                    target_roll_rad=nav.target_roll_rad,
                    target_pitch_rad=nav.target_pitch_rad,
                    thrust=nav.thrust,
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
