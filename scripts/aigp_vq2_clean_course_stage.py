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
- ``GATE0_CLIMB_VERTICAL_OFFSET_NORM``: the 2026-07-29 analysis (Q1/Q4)
  pre-crossing climb recommendation, closure-scaled between
  ``GATE0_CLIMB_REFERENCE_LOG_SCALE`` and ``CROSSING_MIN_LOG_SCALE`` after
  flight 20260729T085719Z-visual-course-4455fd61 climbed into the gate-0 top
  bar under the fixed offset.
- Thrust envelope ``[MIN_COURSE_THRUST, MAX_COURSE_THRUST]`` and yaw cap: the
  accepted v3 yaw profile and the visual-course thrust envelope from the
  July-18 safety contract.

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
MAX_COURSE_THRUST = 0.32  # MAX_VISUAL_THRUST

LAUNCH_BOOST_THRUST = 0.32  # proved 0.32-thrust launch (feedforward only)
LAUNCH_BOOST_DURATION_S = 0.75  # inside the validated 0.45..1.0 boost window

# Gate-0-phase feedforward vertical setpoint offset (image-down norm).  The
# 2026-07-29 crossing-geometry analysis (Q1/Q4) shows gate 1 first appears at
# median y=-0.69 (~9 deg from the top edge, 20% of runs already TOP-clipped
# at first sight) because the vehicle crosses gate 0 too low.  Biasing the
# gate-0 vertical setpoint +0.25 norm (holding the gate-0 target lower in
# frame) crosses ~1-1.2 m higher, putting gate-1 first-seen near y=-0.3 and
# roughly doubling the post-credit observability window.  Feedforward only,
# gate-0 phase only, inside the thrust envelope (max +0.02 above support).
# Flight 20260729T085719Z-visual-course-4455fd61 showed the FIXED offset held
# the climb into gate 0's top bar (ey ran to +0.44 against the +0.25
# setpoint), so the bias is closure-scaled: full at the spawn detection scale
# and ramping linearly to zero at CROSSING_MIN_LOG_SCALE (see command()).
GATE0_CLIMB_VERTICAL_OFFSET_NORM = 0.25

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
MAX_COURSE_YAW_RATE_RAD_S = 0.15  # accepted v3 profile production cap

ADVANCE_PITCH_RAD = -0.18  # nose-down closure target when aligned/confident
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
VERTICAL_QUALIFY_MAX_AGE_S = 0.30  # qualified vertical measurement horizon

TARGET_SLEW_RAD_S = 0.30  # single transparent target slew rate
CLIPPED_STEERING_FRACTION = 0.5  # clipping saturates corrective steering

APERTURE_MIN_CONFIDENCE = 0.20  # fitted inner-aperture acceptance floor
OUTER_MEAS_STD_NORM = 0.06  # outer bbox center measurement std
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
                if self.state is CleanCourseState.PREDICT and (
                    gap > cfg.predict_max_gap_s
                    or self.current.position_std > cfg.search_covariance_std_norm
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
        return True

    # -- the one continuous control law -------------------------------------

    def command(
        self,
        *,
        now_s: float,
        roll_rad: float,
        pitch_rad: float,
    ) -> NavigationOutput:
        """Produce the single navigation request for one tick."""

        cfg = self.config
        if self._last_command_s is None:
            dt = cfg.control_period_s
        else:
            dt = _clamp(now_s - self._last_command_s, 1e-3, 0.10)
        self._last_command_s = float(now_s)

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

        support = _clamp(
            cfg.support_collective
            / max(0.85, math.cos(roll_rad) * math.cos(pitch_rad)),
            cfg.min_thrust,
            cfg.max_thrust,
        )

        if self.state is CleanCourseState.SEARCH:
            sweep_yaw = self._search_yaw(dt)
            self._collective = support
            target_roll = self._slew_roll(0.0, dt)
            target_pitch = self._slew_pitch(cfg.brake_pitch_rad, dt)
            return NavigationOutput(
                target_roll_rad=target_roll,
                target_pitch_rad=target_pitch,
                yaw_rate_rad_s=sweep_yaw,
                thrust=support,
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
                thrust=support,
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
        # detection log scale to zero at the crossing-arm log scale.  Loss of
        # qualified vertical state discards the derivative term and decays
        # collective smoothly toward tilt-compensated support; a saturated
        # sub-support collective is never retained.
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
        vertical_qualified = (
            self.state is CleanCourseState.TRACK
            and now_s - current.last_y_measurement_s
            <= cfg.vertical_qualify_max_age_s
            and current.y_axis.std <= cfg.search_covariance_std_norm
        )
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
            p_term = cfg.vertical_error_gain * bounded_error
            d_term = cfg.vertical_rate_gain * bounded_rate
            # Flight 20260729T085719Z-visual-course-4455fd61: during the
            # gate-0 climb the vy derivative term overwhelmed the P restoring
            # term and cut collective 0.32 -> 0.22 while the target was still
            # below the climb setpoint.  D may damp the P-commanded
            # correction but never reverse its direction; when P and D agree
            # (true overshoot) D keeps full authority.
            if p_term * d_term < 0.0 and abs(d_term) > abs(p_term):
                d_term = math.copysign(abs(p_term), d_term)
            collective = support + cfg.vertical_feedback_sign * (
                p_term + d_term
            )
            self._collective = collective
        else:
            if self._collective is None:
                self._collective = support
            alpha = min(1.0, dt / cfg.collective_decay_tau_s)
            self._collective += (support - self._collective) * alpha
            collective = self._collective

        # Gate-0 takeoff boost is feedforward only; it never changes the
        # closed-loop vertical sign.
        if (
            self.gate_index == 0
            and self._course_start_s is not None
            and now_s - self._course_start_s < cfg.launch_boost_duration_s
        ):
            collective = cfg.launch_boost_thrust
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
        target_pitch = (
            cfg.brake_pitch_rad
            + (cfg.advance_pitch_rad - cfg.brake_pitch_rad) * advance
        )

        return NavigationOutput(
            target_roll_rad=self._slew_roll(target_roll, dt),
            target_pitch_rad=self._slew_pitch(target_pitch, dt),
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
        return _Hypothesis(
            track_id=str(track.track_id),
            x=center[0],
            y=center[1],
            log_scale=log_scale,
            confidence=float(track.confidence),
            pos_var=INITIAL_POS_VAR_NORM,
            now_s=now_s,
        )

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

    def _slew_pitch(self, target: float, dt: float) -> float:
        limit = self.config.target_slew_rad_s * dt
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


def _visible_tracks(update: Any) -> List[Any]:
    """Duck-typed visible-track extraction from one tracker update."""

    if update is None:
        return []
    tracks = list(getattr(update, "tracks", ()) or ())
    visible_ids = set(getattr(update, "visible_track_ids", ()) or ())
    result = []
    for track in tracks:
        visible = getattr(track, "visible", None)
        if visible is None:
            visible = track.track_id in visible_ids if visible_ids else True
        if visible:
            result.append(track)
    return result


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

            estimate = host.estimate
            if estimate is None:
                raise rt.safety_abort_type(
                    "visual-course lost the IMU attitude estimate"
                )
            roll_rad, pitch_rad, _yaw = estimate.orientation.to_euler()
            nav = controller.command(
                now_s=now,
                roll_rad=roll_rad,
                pitch_rad=pitch_rad,
            )
            # One attitude PD for roll/pitch; yaw stays an explicit channel.
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
            host._record_tick("visual-course", elapsed, command)
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
