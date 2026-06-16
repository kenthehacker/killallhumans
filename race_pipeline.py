"""
Main autonomous racing pipeline — integrates all modules.

This is the top-level orchestrator that connects:
  - Competition interface (MAVLink or PyBullet)
  - Gate perception (Phase1 detector + PnP)
  - State estimation (EKF with PnP drift correction)
  - Trajectory planning (time-optimal + racing line)
  - Control (geometric tracker)
  - Gate sequencing (platform-agnostic)

Architecture follows the research-backed design from MASTERPLAN.md:

  Camera → Gate Detector → PnP Pose → State Estimator (EKF)
                                            ↓
  Telemetry → State Predictor → Trajectory Tracker → Attitude Command
                                            ↑
                        Pre-computed Racing Trajectory
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from competition.adapter import (
    AttitudeCommand,
    CameraFrame,
    CompetitionInterface,
    TelemetryState,
)
from competition.aigp_geometry import AIGP_VQ1_MAX_RUN_DURATION_S
from competition.session import RaceSession
from control.mpc_tracker import GeometricTracker, SimplePositionTracker, TrackerConfig
from estimation.ekf import DroneEKF, EKFConfig
from estimation.gate_pnp import (
    CameraIntrinsics,
    GateGeometry,
    GatePnPEstimator,
    detect_gate_corners,
)
from estimation.state_predictor import LatencyConfig, StatePredictor
from gate_sequencing.sequencer import GateSequencer, GateSpec, RaceState
from planning.dynamic_replanner import DynamicReplanner, ReplanConfig, ReplanTrigger
from planning.racing_line import RacingLineOptimizer, SpeedProfiler
from planning.trajectory_optimizer import (
    DroneConstraints,
    GateWaypoint,
    RaceTrajectory,
    TrajectoryOptimizer,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the racing pipeline.

    Defaults match the AIGP VQ1 spec (VADR-TS-002): 640×360 forward
    camera tilted 20° up, 1.5 m gate openings. Legacy tracks override
    via explicit args at construction time.
    """
    # Control rate
    target_hz: float = 100.0

    # Camera (AIGP VQ1 — VADR-TS-002 §3.8 + §4.6)
    camera_fov_h: float = 90.0
    image_width: int = 640
    image_height: int = 360
    # Body→camera tilt in radians, positive = nose-up. AIGP camera is
    # tilted 20° upward; legacy / debug paths can override to 0.
    camera_pitch_offset_rad: float = math.radians(20.0)

    # Gate geometry (AIGP VQ1 — VADR-TS-002 §3.7)
    gate_width: float = 1.5
    gate_height: float = 1.5

    # Detection
    use_detection: bool = True
    detection_confidence_threshold: float = 0.3

    # Planning
    # iter-005b (Opus F4 BLOCKER): default lowered from 12.0 to 8.0 to
    # match the synthetic bench's new default. The previous 12.0 was a
    # competition-path overfit: only race_01-like gate spacing could
    # follow it. Tracks with tight turns (slalom-like) needed lower
    # velocities. Callers can still override via `PipelineConfig(max_speed=...)`.
    max_speed: float = 8.0
    trajectory_dt: float = 0.01

    # Control mode
    use_geometric_tracker: bool = True

    # Iter-021 (P0 strip-down): minimal pure-pursuit gate-to-gate control.
    # When True, the pipeline BYPASSES the entire min-snap trajectory stack
    # (RacingLineOptimizer, TrajectoryOptimizer, TOPP, accel projection,
    # dynamic replan, should_slow_down) and instead flies a feasible
    # velocity-tracking reference straight at the current sequencer gate at
    # ``minimal_cruise_speed``, with horizontal accel hard-clamped to the
    # real tilt envelope. This is the handoff's "get a repeatable 6/6 at low
    # speed before any optimization" path — the min-snap reference demands
    # ~18 m/s²/61° tilt the drone cannot make, which is what flips it.
    minimal_control: bool = False
    minimal_cruise_speed: float = 3.0
    minimal_max_tilt_rad: float = 0.62
    # iter-36: velocity-tracking gain (1/s). accel = kv*(v_des - v). The
    # cross-track UNDERSHOOT at the Y-staggered gates is kv-limited, NOT
    # accel-clamp-limited (the g*tan(max_tilt) clamp is ~0% engaged near the
    # worst gates while the drone's Y-velocity lags v_des) — so raising kv
    # tightens lateral tracking while preserving pure-pursuit's natural
    # cross-track damping (v_des decelerates to 0 at the gate regardless of kv),
    # unlike iter-34's cross_gain term which removed that damping and overshot.
    minimal_kv: float = 3.0
    # iter-37: vertical-channel knobs. The decoupled vertical law is
    # vz = clip(vert_gain*dz, -max_vert_speed, +max_vert_speed). At higher
    # cruise the steep CLIMB legs (e.g. gate1->gate2, +8.6m) become the binding
    # constraint: the drone arrives LOW and nearly clips the bottom bar (cruise
    # 8.0: gate2 vertical frame clearance only 0.11m, vs 0.51m lateral). Raise
    # these to keep the climb pace with cruise. -1.0 sentinel => use the
    # MinimalControllerConfig default.
    minimal_vert_gain: float = -1.0
    minimal_max_vert_speed: float = -1.0
    # iter-38: vertical glide-slope FEEDFORWARD. 0 = original proportional law
    # (vz=vert_gain*dz, which lags a descending course and crosses every gate
    # ABOVE centre — iter-38 baseline measured -0.12..-0.33m at all six gates);
    # >0 descends at speed*dz/horiz_dist so the drone arrives at gate altitude
    # on time (1.0 = exact time-to-arrival). A velocity feedforward, not a gain,
    # so it does not trigger the vert_gain>1 thrust->roll instability.
    minimal_vert_ff: float = 0.0
    # iter-34: cross-track (Y) convergence gain for the decoupled horizontal
    # law. 0 = original pure pursuit. >0 fixes the cross-track undershoot at
    # speed (verified roadmap #4).
    minimal_cross_gain: float = 0.0
    # Aim the pure-pursuit target this many metres BEYOND the current gate
    # (toward the next gate). Pursuing the gate centre directly makes the drone
    # arrive and hover AT the gate (parking in the frame -> collisions, and the
    # sequencer never sees a plane crossing -> no pass). Aiming through the
    # gate makes it fly across the plane and on to the next gate.
    # iter-31 speed-up: 4.0->2.0. The cross-track crossing offset is a
    # geometric pure-pursuit lag ~ leg/(leg+through_dist) and is SPEED-
    # INVARIANT; halving through_dist halves the offset (gate3/4 ~0.54->~0.30 m,
    # restoring centring margin in the 1.5 m opening) while still flying THROUGH
    # the gate (parking only happened at through_dist≈0).
    minimal_through_dist: float = 2.0
    # Vertical offset (NED z, metres) added to the aim point. The drone parks
    # dead-centre on gate.position but gets blocked ~0.2 m short of the plane,
    # which suggests gate.position is not the OPENING centre vertically (e.g.
    # it's the gate base). Negative = aim higher (NED z is down). Swept live.
    minimal_aim_z_offset: float = 0.0
    # iter-40 VERTICAL ANTICIPATORY DESCENT (bounded, in metres). The drone
    # arrives ABOVE every gate opening (P-only vertical lag on a descending
    # course; ~0.3 m at cruise 8, ~0.5-0.65 m at cruise 9 where it becomes the
    # binding frame clearance). When >0, aim the ALTITUDE down toward the NEXT
    # (lower) gate by up to this many METRES, ramped in over the last
    # ``minimal_lookahead_band_m`` metres of approach, so the drone descends early
    # and crosses centred. Z-ONLY: a full 3D lookahead (iter-40) halved the
    # vertical lag but its lateral component corner-cut the slalom (gate2 lat
    # 0.326->0.399 m). BOUNDED in metres, NOT a fraction of the leg drop: iter-40
    # crash — fraction 0.4 of the 5 m gate0->gate1 drop = 2 m down-aim dove into
    # gate0's bottom frame (128 collisions). ~0.4 m stays inside the 0.75 m
    # half-opening even at the near-level gate0. 0 = off. Swept live.
    minimal_lookahead_m: float = 0.0
    minimal_lookahead_band_m: float = 12.0
    # iter-41 LATERAL LEAD (anti-undershoot, for high cruise). At the Y-staggered
    # slalom gates the drone arrives SHORT in Y (cross-track tracking lag + turn
    # authority): undershoot grows 0.34 m @ cruise 9 -> 0.55 m @ cruise 10.5,
    # where it eats the frame clearance (0.04 m, clip+tumble) and caps speed.
    # When >0, aim the Y PAST the gate by up to this many metres IN THE SLALOM
    # TRAVEL DIRECTION (sign of prev-gate -> this-gate Y change), ramped in over
    # the last ``minimal_lookahead_band_m`` metres, so the undershooting drone
    # arrives centred. Bounded (stays in the 0.75 m half-opening). This is the
    # OPPOSITE of a next-gate lookahead (which worsens undershoot by pulling Y
    # toward the next, opposite-side gate). Swept live; 0 = off.
    minimal_lat_lead_m: float = 0.0
    # iter-42 VARIABLE SPEED (turn/descent-aware profile). The binding constraint
    # at high cruise is the SPEED-DEPENDENT VERTICAL LAG (drone arrives above the
    # opening; at cruise 10 gates 1/2/4 are vertical-bound, clearance 0.14-0.29 m)
    # plus the lateral undershoot — both shrink if the drone is SLOWER through the
    # hard gates. So fly a fast BASE (minimal_cruise_speed = peak/straight speed)
    # and BRAKE approaching the geometrically-tight gates: leg cruise =
    # base * clip(1 - speed_brake * turn_angle, speed_min_frac, 1), where
    # turn_angle is the 3D direction change (incoming vs outgoing leg) at the
    # TARGET gate — large for steep-descent + Y-reversal gates (g2->g3->g4), ~0
    # for the near-collinear legs (which keep full speed for the peak-velocity
    # target). 0 = off (constant cruise). Decouples peak speed from the slalom
    # turn limit — the user-endorsed unlock. Swept live.
    minimal_speed_brake: float = 0.0      # 1/rad; >0 enables variable speed
    minimal_speed_min_frac: float = 0.5   # floor on the braked cruise fraction
    # The turn-angle brake misses the steep-but-straight DESCENT gates (g1: legs
    # nearly collinear in 3D -> tiny turn angle, but it must lose ~5-9 m and so
    # arrives above the opening -> vertical-bound, 0.34 m @ base 13.5). Add the
    # incoming leg's vertical SLOPE (|dz|/|dxy|) to the difficulty so descent-
    # heavy gates also brake. difficulty = max(turn_angle, descent_gain*slope).
    minimal_speed_descent_gain: float = 0.0  # >0 also brakes steep descents
    # iter-44: final-gate arrival ramp radius (m). The drone OVERSHOOTS the
    # finish gate5's Y at high speed (the only fast leg ends there) and tumbles
    # on the roll-back. Braking gate5's approach from FARTHER out bleeds the
    # cross-track velocity before the plane while the mid-leg still carries the
    # peak. Default 8.0 (the controller's own default).
    minimal_arrival_radius: float = 8.0

    # Iter-035: clean trajectory-race harness (the principled alternative to
    # minimal pure-pursuit). minimal_control stays False so configure() still
    # builds the trajectory, but the control loop flies that precomputed
    # trajectory with the GeometricTracker on RAW telemetry and SKIPS the
    # replan / state-predictor / should_slow_down machinery — an apples-to-
    # apples A/B vs minimal where only the controller (racing-line + velocity
    # feedforward tracking vs gate-by-gate pure pursuit) differs. The optimizer
    # is constrained to the REAL measured envelope (NOT the bench placeholders
    # that made the old min-snap reference infeasible), and the tracker is
    # given the calibrated thrust + the live-sim roll sign.
    trajectory_race: bool = False
    # PLANNING accel budget for the optimizer. 7.5 (the real usable lateral
    # accel) made the accel-peak projection over-stretch every through-gate
    # segment chasing an unreachable target -> 36 s (slower than minimal's
    # 26 s). 15.0 lets the optimizer plan smoothly; the ~17 m/s² interior kink
    # peaks that remain are CLAMPED to the real ~7 m/s² envelope at the tracker
    # (max_lateral_accel), so it stays feasible. Recovers ~8 s at no centering
    # cost (measured offline). "Plan smooth, clamp safe."
    traj_max_accel_mps2: float = 15.0
    traj_max_tilt_rad: float = 0.62       # real usable tilt (also caps tracker lateral accel ~7)
    traj_max_thrust_n: float = 37.0       # calibrated hover thrust (not 42)

    # Estimation
    use_ekf: bool = True
    use_pnp: bool = True
    pnp_mode: str = "backup"
    use_state_predictor: bool = True

    # Replan execution. When True, a mid-race trajectory rebuild runs on a
    # background thread while the control loop keeps tracking the (still
    # valid) current trajectory, swapping atomically when the rebuild
    # completes. The legacy synchronous path ran the ~1.8 s optimisation
    # inline in the 100 Hz callback, blinding the controller and causing a
    # recovery death-spiral (audit Blocker 9). Set False for the old
    # blocking behaviour.
    async_replan: bool = True

    # Iter-015: ML training infra (continues iter-014's feature-trace
    # hook). When True, the underlying GeometricTracker captures
    # per-step features + nominal commands to `pipeline.tracker.feature_trace`.
    # Default off — production callers see zero overhead. A future
    # `scripts/collect_residual_dataset.py` flips this on, runs a race,
    # and dumps the trace to .npz for training.
    trace_tracker_features: bool = False


class RacePipeline:
    """
    Complete autonomous racing pipeline.

    Lifecycle:
      1. configure() — set up gates and compute trajectory
      2. run() — execute the race via the competition interface

    The pipeline pre-computes a time-optimal trajectory offline,
    then tracks it online using the geometric controller, with
    real-time adjustments from gate detection and state estimation.
    """

    def __init__(
        self,
        interface: CompetitionInterface,
        config: PipelineConfig = None,
    ):
        self.interface = interface
        self.config = config or PipelineConfig()

        # Modules
        # Iter-002 (B5, composer-25-4 F4): thread PipelineConfig's
        # camera_pitch_offset_rad into the CameraIntrinsics built by
        # from_fov. Previously the offset was stored on PipelineConfig
        # but the constructed intrinsics dropped it, leaving
        # CameraIntrinsics.pitch_offset_rad at its module default. The
        # offset is irrelevant for position-only PnP with coincident
        # camera/body origins (spec VADR-TS-002 §3.8), but it matters
        # for any future code that derives orientation from a gate
        # detection — wire it through now so the bug class can't
        # silently regress.
        self.camera = CameraIntrinsics.from_fov(
            self.config.camera_fov_h,
            self.config.image_width,
            self.config.image_height,
        )
        self.camera.pitch_offset_rad = self.config.camera_pitch_offset_rad
        self.gate_geometry = GateGeometry(
            self.config.gate_width, self.config.gate_height
        )
        self.pnp_estimator = GatePnPEstimator(self.camera, self.gate_geometry)
        self.ekf = DroneEKF(EKFConfig())
        self.state_predictor = StatePredictor(LatencyConfig())
        # Iter-015: wire trace_features through PipelineConfig so the
        # collection script can opt in without monkey-patching.
        _tracker_kwargs = dict(trace_features=self.config.trace_tracker_features)
        if self.config.trajectory_race:
            # Iter-035: fly the live AIGP sim — calibrated thrust (37 N, not the
            # 42 N drone_spec placeholder, which under-commands hover), the real
            # usable tilt clamp (0.62 rad, which also bounds lateral accel to
            # ~7 m/s²), and the live-sim roll-convention sign that the minimal
            # controller proved is needed (+roll -> +Y at yaw=pi).
            _tracker_kwargs.update(
                max_thrust_n=self.config.traj_max_thrust_n,
                max_tilt_rad=self.config.traj_max_tilt_rad,
                sim_roll_sign=-1.0,
                # Cap lateral accel at the real envelope (g·tan(0.62)≈7) BEFORE
                # attitude extraction, so the ~17 m/s² min-snap kink peaks are
                # clamped cleanly (no realized-accel overshoot, no climb
                # transient) — matches the proven minimal-controller envelope.
                max_lateral_accel=9.81 * math.tan(self.config.traj_max_tilt_rad),
            )
        _tracker_cfg = TrackerConfig(**_tracker_kwargs)
        self.tracker = GeometricTracker(_tracker_cfg)
        self.simple_tracker = SimplePositionTracker(TrackerConfig())

        # Iter-021 (P0): minimal pure-pursuit controller. Cheap to build; only
        # used when config.minimal_control is True (bypasses the whole
        # trajectory stack — see _control_callback).
        from control.minimal_controller import (
            MinimalController,
            MinimalControllerConfig,
        )
        self.minimal_controller = MinimalController(
            MinimalControllerConfig(
                cruise_speed=self.config.minimal_cruise_speed,
                arrival_radius=self.config.minimal_arrival_radius,
                max_tilt_rad=self.config.minimal_max_tilt_rad,
                cross_gain=self.config.minimal_cross_gain,
                kv=self.config.minimal_kv,
                # Keep the lateral-accel clamp CONSISTENT with the tilt clamp
                # (g*tan(max_tilt)). Otherwise raising --max-tilt is a no-op:
                # accel[:2] is clamped to the default max_lateral_accel (7.0)
                # FIRST, and 7.0 maps back to atan2(7.0,g)=0.62 rad, so the
                # higher tilt clamp never binds. iter-36: the binding gates are
                # tilt-SATURATED at the flip (cmd_roll 94-100%), so lateral
                # authority is the real limiter — make --max-tilt actually
                # change it. Backward-compatible: 0.62 -> 7.0 (unchanged).
                max_lateral_accel=9.81 * math.tan(self.config.minimal_max_tilt_rad),
                **({"vert_gain": self.config.minimal_vert_gain}
                   if self.config.minimal_vert_gain > 0 else {}),
                **({"max_vert_speed": self.config.minimal_max_vert_speed}
                   if self.config.minimal_max_vert_speed > 0 else {}),
                vert_ff=self.config.minimal_vert_ff,
            )
        )

        # Phase 1 detector — instantiated once, reused every frame (Phase 3 fix)
        self._detector = None
        try:
            from gate_detection.src.phase1_detector import Phase1GateDetector
            self._detector = Phase1GateDetector(
                camera_fov_horizontal=self.config.camera_fov_h,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
            )
        except ImportError:
            logger.warning("Phase1GateDetector not available — detection disabled")

        # State
        self.sequencer: Optional[GateSequencer] = None
        self.trajectory: Optional[RaceTrajectory] = None
        # Dynamic replanner — wires the same crash/iff-highlighted/
        # off-track surface used by the sim runner into the real-flight
        # path so a hardware run can rebuild the trajectory mid-race
        # instead of dead-reckoning along the original (P1-14).
        self.replanner: Optional[DynamicReplanner] = None
        self._replan_count: int = 0
        self._last_replan_reasons: List[str] = []
        self._last_lateral_err: float = 0.0
        self._race_start_time: float = 0.0
        self._gate_specs: List[GateSpec] = []
        self._initialized = False

        # Monotonic reference-anchor clock (see _control_callback). Mirrors
        # the pattern visual_demo.py settled on after iter 8 / iter 11:
        # wall-clock sampling ignores gate progress, so when perception
        # stalls or the drone lags a climb, the reference marches off and
        # the tracker has nothing to chase. Anchor advances only when the
        # drone actually makes progress along the path.
        self._ref_progress_time: float = 0.0
        # Lookahead — same 0.3 s used by visual_demo's PD feedforward.
        self._ref_lookahead_s: float = 0.3
        # Proximity sanity radius used when associating detections to the
        # current target gate. Detections whose PnP-recovered world pose
        # is further than this from ``sequencer.current_gate.position``
        # are rejected as misassociated (another gate in frame, or a
        # spurious detection) rather than fed into the EKF.
        self._pnp_gate_match_radius_m: float = 3.0
        self._ekf_live_initialized: bool = False
        self._last_lpn_stamp_ms: Optional[int] = None
        self._last_odom_reset_counter: Optional[int] = None

        # Deferred replan trigger. ``DynamicReplanner.evaluate()`` reports
        # each crash/miss/off-track event exactly once (on its rising edge),
        # then consumes it. If that single edge happens to land inside the
        # replan cooldown window, ``should_replan()`` rejects it and — with
        # no memory of the event — it is lost forever (a strut graze during
        # cooldown ⇒ no recovery; an off-track entry during cooldown ⇒ stuck
        # in RECOVERY). We remember an unserved-but-triggered evaluation here
        # and merge it into later ticks so the replan fires the moment the
        # cooldown expires. Cleared only once a rebuild actually succeeds.
        self._pending_trigger = None

        # Async replan state (audit Blocker 9). The rebuild runs on a single
        # background worker; the control loop keeps flying the current
        # trajectory until the worker publishes a result, which is swapped in
        # atomically on the control thread. ``_rebuild_lock`` guards only the
        # small result handoff.
        self._rebuild_lock = threading.Lock()
        self._rebuild_in_flight = False
        self._rebuild_result = None  # (trajectory_or_None, trigger)
        self._rebuild_thread = None

        # Frozen-telemetry watchdog. A silently-dead telemetry feed (the
        # mavlink RX subscription dies, or the sim stops publishing) leaves
        # the controller flying on a stale state estimate: it keeps emitting
        # the same correction every tick, which on the live path shows up as
        # the drone "spinning in circles" or drifting while the recorded
        # state never changes. Nothing previously detected this. We count
        # consecutive control ticks whose telemetry timestamp has not
        # advanced and raise a loud, once-per-episode error when the feed has
        # clearly stalled. Detection only — flight commands are unchanged.
        self._last_telem_stamp_us: Optional[int] = None
        self._telem_stale_ticks: int = 0
        self._telem_frozen_ticks: int = 0  # cumulative, for the run summary
        self._telem_stale_warned: bool = False
        # ~0.5 s of a 100 Hz control loop with no new telemetry.
        self._telem_stale_tick_limit: int = 50

    def configure(
        self,
        gates: List[GateSpec],
        start_position: Tuple[float, float, float] = (0, 0, 0),
        start_velocity: Tuple[float, float, float] = (0, 0, 0),
    ) -> None:
        """
        Set up the race: configure gates and pre-compute trajectory.

        Call this before run().
        """
        self._gate_specs = gates
        self.sequencer = GateSequencer(gates)
        self.replanner = DynamicReplanner(ReplanConfig())
        self._replan_count = 0
        self._last_replan_reasons = []
        self._last_lateral_err = 0.0
        self._pending_trigger = None
        self._rebuild_in_flight = False
        self._rebuild_result = None
        self._ekf_live_initialized = False
        self._last_lpn_stamp_ms = None
        self._last_odom_reset_counter = None

        # Iter-021 (P0): minimal control flies straight at the sequencer's
        # current gate — no precomputed trajectory needed. Skip the (slow)
        # min-snap optimization entirely; that stack is exactly what we are
        # bypassing.
        if not self.config.minimal_control:
            self._build_trajectory_from(start_position, start_velocity, gates)
        self.ekf.initialize(start_position, start_velocity, timestamp_s=0.0)
        self._initialized = True

    def _build_trajectory_from(
        self,
        start_position: Tuple[float, float, float],
        start_velocity: Tuple[float, float, float],
        remaining_gates: List[GateSpec],
    ) -> None:
        """Build and assign ``self.trajectory`` (synchronous). Used by
        configure() and the legacy synchronous replan path."""
        self.trajectory = self._compute_trajectory(
            start_position, start_velocity, remaining_gates
        )

    def _compute_trajectory(
        self,
        start_position: Tuple[float, float, float],
        start_velocity: Tuple[float, float, float],
        remaining_gates: List[GateSpec],
    ):
        """Pure trajectory build through ``remaining_gates`` from the current
        state — returns a RaceTrajectory and touches no shared pipeline state,
        so it is safe to call from a background replan thread."""
        gate_waypoints = [
            GateWaypoint(
                position=g.position,
                normal=_gate_normal(g.yaw, g.pitch),
                width=g.interior_width,
                height=g.interior_height,
                yaw=g.yaw,
            )
            for g in remaining_gates
        ]

        logger.info(
            "Optimizing racing line through %d gates...", len(remaining_gates),
        )
        if self.config.trajectory_race:
            # The default max_lateral_offset=0.6 (fraction of half-width) puts a
            # fixed ~0.45 m offset on every VQ1 gate — most of the 0.75 m
            # half-opening, gone before the tracker even runs. Tighten to 0.15
            # (~0.11 m) for the narrow 1.5 m gates; cuts the worst-gate
            # reference offset 45 cm -> 13 cm at ~no time cost (measured).
            from planning.racing_line import RacingLineConfig
            line_optimizer = RacingLineOptimizer(
                RacingLineConfig(max_lateral_offset=0.15)
            )
        else:
            line_optimizer = RacingLineOptimizer()
        optimized_waypoints = line_optimizer.optimize(
            gate_waypoints, start_position,
        )

        profiler = SpeedProfiler(max_speed=self.config.max_speed)
        waypoint_positions = [start_position] + [
            g.position for g in optimized_waypoints
        ]
        speeds = profiler.profile(waypoint_positions)
        logger.info(
            "Speed profile: min=%.1f max=%.1f m/s", min(speeds), max(speeds),
        )

        logger.info("Computing time-optimal trajectory...")
        if self.config.trajectory_race:
            # Iter-035: constrain to the REAL measured AIGP envelope, not the
            # bench placeholders (15 m/s² / 0.85 rad). The old min-snap demanded
            # ~18 m/s² / 61° the drone cannot make and flipped it; planning at
            # the true ~7 m/s² / 35° budget yields a trajectory the fixed inner
            # loop can actually track.
            constraints = DroneConstraints(
                max_velocity=self.config.max_speed,
                max_acceleration=self.config.traj_max_accel_mps2,
                max_tilt_angle=self.config.traj_max_tilt_rad,
                max_thrust=self.config.traj_max_thrust_n,
            )
        else:
            constraints = DroneConstraints(max_velocity=self.config.max_speed)
        traj_optimizer = TrajectoryOptimizer(
            constraints=constraints,
            dt_sample=self.config.trajectory_dt,
        )
        trajectory = traj_optimizer.optimize(
            optimized_waypoints, start_position, start_velocity,
        )
        logger.info(
            "Trajectory: %.1fs total, %d points",
            trajectory.total_time,
            len(trajectory.points),
        )
        return trajectory

    async def run(self, address: str = "udp://:14540") -> None:
        """
        Execute the race.

        Connects to the competition interface and runs the full
        perception → estimation → planning → control loop.
        """
        if not self._initialized:
            raise RuntimeError("Call configure() before run()")

        session = RaceSession(
            self.interface,
            target_hz=self.config.target_hz,
            address=address,
        )
        session.on_telemetry = self._control_callback
        # Iter-001 review (B2, consensus BLOCKER): stop on ANY terminal
        # sequencer state — completion, out-of-order DQ, or a recorded
        # crash. The prior version only checked is_complete, so a DQ run
        # silently kept flying.
        # Stop on the SIM's authoritative race_finished (it credits passes,
        # and its detection can lead our geometric sequencer — at higher
        # speed our sequencer lagged gate 5 by seconds, so the run flew well
        # past the sim's actual finish before stopping). Also stop on any
        # terminal sequencer state (completion / DQ / crash).
        def _should_stop() -> bool:
            rs = getattr(self.interface, "race_status", None)
            if rs is not None and rs.race_finished:
                return True
            # iter-38: divergence guard tripped in the control callback.
            if getattr(self, "_diverged", False):
                return True
            # Do NOT stop on our geometric sequencer's is_complete — it can fire
            # a beat BEFORE the drone is fully through the last gate, cutting the
            # run off before the SIM credits the final crossing (cruise 3.5:
            # our 6/6 but sim only 5/6). The SIM's race_finished above is the
            # sole completion authority. Still stop on terminal failures.
            return self.sequencer is not None and (
                self.sequencer.is_disqualified
                or self.sequencer.is_timed_out
                or self.sequencer.last_crash is not None
            )

        session.should_stop = _should_stop

        # Iter-002 review B1 (5/7 BLOCKER) + iter-003 M5 (4/7 MAJOR): the
        # 8-minute timeout check uses BOTH a monotonic wall-clock fallback
        # AND a sim-time path that takes precedence when telemetry carries
        # `timestamp_us`. Sim-time is preferred because a non-realtime
        # simulator (slowed down, paused, or fast-forwarded) would
        # false-trip or under-trip a wall-clock-only check.
        self._race_start_time = time.monotonic()
        self._race_start_sim_time_s: Optional[float] = None  # set on first telem tick
        self._ref_progress_time = 0.0
        # iter-38: divergence-guard state (see _control_callback + _in_course_box).
        self._diverged = False
        self._diverged_ticks = 0
        self.sequencer.start()

        logger.info("Starting race...")
        metrics = await session.run()
        logger.info("Race finished: %s", metrics)

    def _gate_difficulty(self, idx: int) -> float:
        """Static per-gate difficulty for the variable-speed brake (cached):
        the 3D direction-change angle (incoming vs outgoing leg) combined with
        the incoming leg's vertical SLOPE (|dz|/|dxy|) so steep-but-straight
        descent gates also brake. Returns 0 for gate0 and the final gate (no
        brake — they keep full speed for the peak)."""
        cache = getattr(self, "_gate_diff_cache", None)
        if cache is None:
            cache = self._gate_diff_cache = {}
        if idx in cache:
            return cache[idx]
        specs = self._gate_specs
        diff = 0.0
        if 0 < idx < len(specs):
            p_prev = np.array(specs[idx - 1].position, dtype=float)
            p_cur = np.array(specs[idx].position, dtype=float)
            inc = p_cur - p_prev
            nxt = (np.array(specs[idx + 1].position, dtype=float) - p_cur
                   if idx + 1 < len(specs) else inc)
            ni, nn = float(np.linalg.norm(inc)), float(np.linalg.norm(nxt))
            theta = 0.0
            if ni > 1e-6 and nn > 1e-6:
                theta = math.acos(float(np.clip(np.dot(inc, nxt) / (ni * nn), -1.0, 1.0)))
            slope = abs(float(inc[2])) / max(1e-6, float(np.linalg.norm(inc[:2])))
            diff = max(theta, self.config.minimal_speed_descent_gain * slope)
        cache[idx] = diff
        return diff

    def _control_callback(
        self,
        telem: TelemetryState,
        frame: Optional[CameraFrame],
    ) -> Optional[AttitudeCommand]:
        """
        Main control callback — called at target_hz.

        Returns an AttitudeCommand to send to the simulator.
        """
        if self.sequencer is None:
            return None
        # Minimal control needs no precomputed trajectory; every other mode
        # tracks one, so require it there.
        if not self.config.minimal_control and self.trajectory is None:
            return None

        # 0. Frozen-telemetry watchdog (detection only).
        self._check_telemetry_freshness(telem)

        # 1. Update state estimation
        position, velocity, yaw = self._update_state_estimate(telem)

        # 1b. DIVERGENCE / INSTABILITY GUARD (iter-38). Abort a run that has gone
        # unstable (inner-loop limit cycle — sustained high gyro) or flown clean
        # off the course box, before it logs more garbage telemetry or risks
        # wedging the sim. _should_stop ends the run; we also hold a safe hover so
        # the divergent command isn't sent. (See _GYRO_INSTABILITY_RADS etc.)
        unstable = _gyro_unstable(telem.angular_velocity)
        out_of_box = not _in_course_box(position)
        if unstable or out_of_box:
            self._diverged_ticks = getattr(self, "_diverged_ticks", 0) + 1
            if self._diverged_ticks >= _GUARD_TRIP_TICKS and not self._diverged:
                self._diverged = True
                logger.error(
                    "DIVERGENCE GUARD tripped after %d ticks (gyro_unstable=%s, "
                    "out_of_box=%s, pos=(%.0f,%.0f,%.0f)) — aborting run to avoid "
                    "garbage telemetry and a wedged sim.",
                    self._diverged_ticks, unstable, out_of_box,
                    *(float(c) if math.isfinite(float(c)) else float("nan")
                      for c in position),
                )
        else:
            self._diverged_ticks = 0
        if getattr(self, "_diverged", False):
            yaw_safe = yaw if math.isfinite(yaw) else math.pi
            return AttitudeCommand(0.0, 0.0, yaw_safe, 0.27)  # safe hover; run ends

        # 2. Gate detection and PnP (if camera frame available).
        # ``detection_active`` distinguishes "detector looked but saw
        # nothing" (dropout, counts against should_slow_down) from
        # "no camera this tick" (don't count). A competition bridge
        # that returns None for camera frames used to latch slow-down
        # permanently after 0.3 s even during nominal flight.
        gate_detected = False
        detection_active = frame is not None and self.config.use_detection
        if detection_active:
            gate_detected = self._process_detection(frame, position, yaw)

        # 3. Update gate sequencer
        passed = self.sequencer.update(
            position, gate_detected, detection_active=detection_active
        )
        if passed:
            logger.info(
                "Gate %s passed! (%d/%d)",
                passed.gate_id, self.sequencer.gates_passed, self.sequencer.total_gates,
            )

        if self.sequencer.is_complete and not self.config.trajectory_race:
            logger.info("All gates passed! Race complete.")
            return AttitudeCommand(0, 0, yaw, 0.4)  # hover
        # trajectory_race: our geometric sequencer can flag complete a beat
        # before the SIM credits the final crossing, so do NOT hover here —
        # keep flying the trajectory's tail (the virtual finish sits 2 m past
        # the last gate) until the SIM's race_finished stops the run.

        # Iter-001 review (B2): terminal failure must abort the control
        # loop, not silently keep tracking the next reference. Hover the
        # vehicle in place; the session-level should_stop will pick up
        # the same signal and exit the run.
        if self.sequencer.is_disqualified:
            logger.error("Race terminated: DQ — %s", self.sequencer.dq_reason)
            return AttitudeCommand(0, 0, yaw, 0.4)  # hover (run is over)
        if self.sequencer.last_crash is not None:
            gate_id, _xyz = self.sequencer.last_crash
            logger.error("Race terminated: gate-frame crash on %s", gate_id)
            return AttitudeCommand(0, 0, yaw, 0.4)  # hover
        # Iter-002 (5/7 reviews): enforce VQ1 8-minute cap. Sequencer's
        # TIMED_OUT state is set by RaceSession or the bench based on
        # elapsed sim time; here we just abort the loop cleanly.
        if self.sequencer.is_timed_out:
            logger.error(
                "Race terminated: timeout — %s",
                self.sequencer.timeout_reason or "max_run_duration_exceeded",
            )
            return AttitudeCommand(0, 0, yaw, 0.4)  # hover

        # Iter-003 M5: prefer sim-time elapsed (from telem.timestamp_us)
        # over wall-clock. A simulator running below realtime would
        # under-time-out under wall-clock; one running above would over-
        # time-out. With sim time we measure what the COMPETITION sees.
        # Fall back to monotonic wall-clock if telem has no timestamp.
        if telem.timestamp_us is not None and telem.timestamp_us > 0:
            sim_time_s = telem.timestamp_us / 1e6
            if self._race_start_sim_time_s is None:
                self._race_start_sim_time_s = sim_time_s
            elapsed = sim_time_s - self._race_start_sim_time_s
        else:
            elapsed = time.monotonic() - self._race_start_time
        if elapsed > AIGP_VQ1_MAX_RUN_DURATION_S and not self.sequencer.is_timed_out:
            self.sequencer.mark_timed_out(
                f"vq1_max_run_duration_exceeded:{elapsed:.1f}s"
            )

        # 3z. Iter-021 (P0): minimal pure-pursuit path. Bypasses the entire
        # trajectory/replan/predictor/should_slow_down stack below — fly a
        # feasible velocity reference straight at the current gate. State
        # estimate (EKF), sequencer pass-detection and terminal handling
        # above are all reused.
        if self.config.minimal_control:
            gate = self.sequencer.current_gate
            if gate is None:
                # Our geometric sequencer thinks the course is done, but the SIM
                # is the authority. If the sim hasn't flagged race_finished yet,
                # KEEP flying forward THROUGH the last gate so the sim fully
                # registers the final crossing (our is_complete can lead the sim
                # by a beat). Otherwise hover.
                rs = getattr(self.interface, "race_status", None)
                if rs is not None and not rs.race_finished and self._gate_specs:
                    gate = self._gate_specs[-1]
                else:
                    return AttitudeCommand(0, 0, yaw, 0.4)  # done: hover
            # Aim BEYOND the gate, ALONG ITS NORMAL, so the drone approaches
            # perpendicular and flies straight through the CENTRE of the
            # opening (not toward the next gate, which cuts the corner and
            # crosses the plane off-centre -> frame collisions + the sequencer
            # crediting an off-centre/early pass, as in min_v9). The sequencer
            # advances on the clean plane crossing.
            # iter-43: VARIABLE SPEED — PER-LEG brake + SLEW-limited transitions.
            # The binding constraint at high base is the speed-dependent VERTICAL
            # LAG: on a steep descent leg the drone must lose altitude over the
            # WHOLE leg, so it must be slow for the WHOLE leg (a proximity/near-
            # gate brake let it barrel down g1->g2 at full speed and arrive 3.9 m
            # high -> crash). So the brake is per-leg (set by the current TARGET
            # gate's difficulty = max(turn angle, descent slope)). To avoid the
            # "rocks too hard / discrete flight plans" jolt at a gate pass (user),
            # SLEW-limit the cruise so it RAMPS between legs instead of stepping;
            # the kv loop smooths further. min_frac keeps the braked gates at a
            # speed they handle cleanly (~9 m/s), not crawling.
            if self.config.minimal_speed_brake > 0.0:
                base = self.config.minimal_cruise_speed
                diff = self._gate_difficulty(getattr(gate, "sequence_index", 0))
                factor = max(self.config.minimal_speed_min_frac,
                             min(1.0, 1.0 - self.config.minimal_speed_brake * diff))
                target_cruise = base * factor
                now = time.monotonic()
                prev_t = getattr(self, "_cruise_t", now)
                dt = max(1e-3, min(0.1, now - prev_t))
                self._cruise_t = now
                prev_c = getattr(self, "_cruise_cmd", target_cruise)
                step = 12.0 * dt  # m/s per s slew (ramps a 5 m/s change in ~0.4 s)
                self._cruise_cmd = prev_c + max(-step, min(step, target_cruise - prev_c))
                self.minimal_controller.cfg.cruise_speed = self._cruise_cmd
            cur = np.array(gate.position, dtype=float)
            normal = np.array(_gate_normal(gate.yaw, gate.pitch), dtype=float)
            # Orient the normal in the direction of travel (away from the drone).
            if float(np.dot(normal, cur - np.array(position, dtype=float))) < 0:
                normal = -normal
            aim = cur + self.config.minimal_through_dist * normal
            aim[2] += self.config.minimal_aim_z_offset
            # iter-40: VERTICAL-ONLY anticipatory descent, BOUNDED in metres.
            # The drone arrives ABOVE every opening (P-only vertical lag; ~0.3 m
            # at cruise 8, ~0.5-0.65 m at cruise 9 where it binds clearance). Aim
            # the altitude DOWN toward the next (lower) gate to descend early and
            # cross centred. Z-only (no lateral shift) so it does NOT corner-cut
            # the slalom (iter-40: a 3D lookahead cut gate2 lat 0.326->0.399 m).
            # BOUNDED to ``minimal_lookahead_m`` METRES (NOT a fraction of the
            # leg drop — iter-40 crash: fraction 0.4 of the 5 m gate0->gate1 drop
            # = 2 m down-aim → dove into gate0's bottom frame, 128 collisions).
            # A fixed ~0.4 m bound stays well inside the 0.75 m half-opening even
            # where anticipation isn't needed (e.g. the near-level gate0).
            look_m = self.config.minimal_lookahead_m
            if look_m > 0.0:
                nxt = self.sequencer.next_gate
                if nxt is not None:
                    nxt_z = float(nxt.position[2]) + self.config.minimal_aim_z_offset
                    d_cur = float(np.linalg.norm(
                        cur - np.array(position, dtype=float)))
                    band = max(1e-3, self.config.minimal_lookahead_band_m)
                    ramp = max(0.0, min(1.0, 1.0 - d_cur / band))
                    # descend only (next gate lower => nxt_z > aim[2]), bounded.
                    aim[2] += min(look_m, max(0.0, nxt_z - aim[2])) * ramp
            # iter-41: lateral lead — aim PAST the gate's Y in the slalom travel
            # direction so the undershooting drone arrives centred (anti-
            # undershoot for high cruise). Bounded; ramped near the gate.
            lead_m = self.config.minimal_lat_lead_m
            if lead_m > 0.0 and getattr(gate, "sequence_index", 0) > 0:
                prev = self._gate_specs[gate.sequence_index - 1]
                d_y = cur[1] - float(prev.position[1])
                if abs(d_y) > 1e-3:
                    d_cur = float(np.linalg.norm(
                        cur - np.array(position, dtype=float)))
                    band = max(1e-3, self.config.minimal_lookahead_band_m)
                    ramp = max(0.0, min(1.0, 1.0 - d_cur / band))
                    aim[1] += lead_m * (1.0 if d_y > 0 else -1.0) * ramp
            return self.minimal_controller.compute(
                position, velocity, yaw, tuple(aim), is_final_gate=False,
            )

        # 3z'. Iter-035 (P-traj): clean trajectory-race path. Fly the
        # precomputed (real-envelope) trajectory with the GeometricTracker on
        # RAW telemetry. Deliberately BYPASSES the replan / state-predictor /
        # should_slow_down stack below so the only difference vs the minimal
        # A/B baseline is the controller (racing-line + velocity feedforward
        # tracking vs gate-by-gate pure pursuit). Same reset/countdown, same
        # body-rate inner loop, same SIM-authoritative stop.
        if self.config.trajectory_race:
            # Iter-035 (reviewer fix): the GeometricTracker — unlike the minimal
            # controller — has NO non-finite guard, and on raw telemetry a
            # single NaN/inf sample (odom reset, dropped field) becomes a NaN
            # command the adapter rejects ("thrust must be finite"), killing the
            # whole run. Mirror the minimal controller: hover on bad telemetry.
            tc = self.tracker.config
            hover_thrust = float(np.clip(
                tc.mass * tc.gravity / tc.max_thrust_n,
                tc.min_thrust_normalized, tc.max_thrust_normalized,
            ))
            if not (np.all(np.isfinite(position)) and np.all(np.isfinite(velocity))):
                return AttitudeCommand(0.0, 0.0, math.pi, hover_thrust)
            # Forward-anchored reference (never snap backward on the descending
            # slalom) + the same 0.3 s lookahead the trajectory path uses.
            closest = self.trajectory.find_closest_forward(
                tuple(position),
                self._ref_progress_time,
                search_window_s=2.0,
            )
            self._ref_progress_time = closest.time
            self._last_lateral_err = float(
                np.linalg.norm(np.array(position) - np.array(closest.position))
            )
            lookahead_t = min(
                closest.time + self._ref_lookahead_s,
                self.trajectory.total_time,
            )
            ref = self.trajectory.sample(lookahead_t)
            # Hold the proven -X heading (yaw=pi). All VQ1 gates pass with a
            # fixed heading (the minimal controller proves this), and pinning
            # yaw keeps the roll/pitch extraction frame identical to the
            # minimal controller's so the sim_roll_sign=-1 fix applies cleanly.
            ref = _ref_override_yaw(ref, math.pi)
            cmd = self.tracker.track(position, velocity, yaw, ref)
            # Output guard: never hand the adapter a non-finite command.
            if not all(math.isfinite(x) for x in (
                cmd.roll_rad, cmd.pitch_rad, cmd.yaw_rad, cmd.thrust,
            )):
                return AttitudeCommand(0.0, 0.0, math.pi, hover_thrust)
            return cmd

        # 3a. Dynamic replan: mirrors sim_pybullet/runner._maybe_replan.
        #     A crash/miss/off-track surface event rebuilds the
        #     trajectory from the drone's current state. The cooldown
        #     guard inside DynamicReplanner prevents replan storms.
        sim_time = time.monotonic() - self._race_start_time
        self._maybe_replan(sim_time, position, velocity)

        # 4. State prediction (latency compensation)
        if self.config.use_state_predictor:
            accel = telem.imu.accel if telem.imu else None
            position, velocity, (_, _, yaw) = self.state_predictor.predict(
                position, velocity,
                (telem.roll, telem.pitch, yaw),
                telem.angular_velocity,
                accel,
            )

        # 5. Pick a reference that respects gate progress.
        # Wall-clock sampling (``trajectory.sample(elapsed)``) marches
        # forward even when the drone stalls, so the tracker ends up
        # chasing a ghost reference that the sequencer has already
        # abandoned. Instead, anchor by spatial closeness on the
        # trajectory — but only search forward of the last anchor
        # point so we don't snap back on self-overlapping geometries
        # like a helix (same fix as visual_demo iter 8).
        closest = self.trajectory.find_closest_forward(
            tuple(position),
            self._ref_progress_time,
            search_window_s=2.0,
        )
        self._ref_progress_time = closest.time
        # Distance from drone to closest point on the trajectory feeds the
        # replanner's sustained_lateral_error trigger on the next tick.
        self._last_lateral_err = float(
            np.linalg.norm(np.array(position) - np.array(closest.position))
        )
        lookahead_t = min(
            closest.time + self._ref_lookahead_s,
            self.trajectory.total_time,
        )
        ref = self.trajectory.sample(lookahead_t)

        # If the sequencer has flagged RECOVERY, override the reference
        # position with the sequencer's recovery target so the tracker
        # actually flies back toward the missed gate instead of
        # continuing along the precomputed path (which is what produced
        # the dead-reckoning drift Codex flagged).
        recovery_target = self.sequencer.get_recovery_target()
        if recovery_target is not None:
            ref = _ref_override_position(ref, recovery_target)

        # 6. Track reference
        if self.config.use_geometric_tracker:
            cmd = self.tracker.track(position, velocity, yaw, ref)
        else:
            cmd = self.simple_tracker.track(
                position, velocity, yaw,
                ref.position, ref.velocity, ref.yaw,
            )

        # 7. Speed reduction if needed.
        # NOTE: the thrust*0.7 cut here looks like it should sink the drone
        # (audit MAJOR — and it does in *level* detection-dropout flight), but
        # it is load-bearing for off-track RECOVERY on the *descending* VQ1
        # course: during the lateral-recovery phase the tracker saturates
        # thrust high, and this cut tempers the climb. Both alternatives tried
        # in iter 12 (floor-at-hover, leave-thrust-unchanged) regressed the
        # working 15 m/s-gust recovery from 6/6 to 2/6 (altitude divergence).
        # Left as-is; a proper fix needs the tracker's saturated thrust
        # allocation reworked (Blocker 11 family), not this scale factor.
        if self.sequencer.should_slow_down():
            cmd = AttitudeCommand(
                roll_rad=cmd.roll_rad * 0.5,
                pitch_rad=cmd.pitch_rad * 0.5,
                yaw_rad=cmd.yaw_rad,
                thrust=cmd.thrust * 0.7,
            )

        return cmd

    def _maybe_replan(
        self,
        sim_time: float,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
    ) -> None:
        """Evaluate the replanner; rebuild the trajectory on a positive
        trigger. Mirrors sim_pybullet/runner._maybe_replan."""
        if self.replanner is None or self.sequencer is None:
            return

        # 0. Land any background rebuild that finished since the last tick.
        self._apply_pending_rebuild(sim_time)

        trig = self.replanner.evaluate(
            sim_time=sim_time,
            sequencer=self.sequencer,
            lateral_error=self._last_lateral_err,
        )
        # Merge with any trigger that fired earlier but could not be served
        # because of the cooldown, so a cooldown-blocked event is retried
        # rather than dropped (see _pending_trigger in __init__).
        trig = _merge_triggers(self._pending_trigger, trig)
        if not self.replanner.should_replan(trig, sim_time):
            # Cooldown (or no trigger). Remember a live trigger so the next
            # eligible tick still serves it; otherwise clear.
            self._pending_trigger = trig if trig.triggered else None
            return
        # Remaining gates from the sequencer's vantage point. The current
        # gate (highlighted target) is the first remaining; passed gates
        # are skipped.
        remaining = self._gate_specs[self.sequencer.gates_passed:]
        if not remaining:
            self._pending_trigger = None
            return

        if self.config.async_replan:
            # Don't block the control loop: kick off a background rebuild and
            # keep flying the current trajectory until it completes. Only one
            # rebuild runs at a time; the trigger stays pending until landed.
            if not self._rebuild_in_flight:
                self._start_async_rebuild(position, velocity, remaining, trig)
            self._pending_trigger = trig
            return

        # Legacy synchronous path (blocks ~1.8 s in the callback — Blocker 9).
        try:
            self._build_trajectory_from(position, velocity, remaining)
        except Exception:
            logger.exception("Replanned trajectory rebuild failed; keeping prior.")
            self._pending_trigger = trig
            return
        self._ref_progress_time = 0.0
        self.replanner.mark_replanned(sim_time, trig)
        self._pending_trigger = None
        self._replan_count += 1
        self._last_replan_reasons = trig.reasons
        logger.info(
            "REPLAN #%d at t=%.2fs reasons=%s",
            self._replan_count, sim_time, trig.reasons,
        )

    def _start_async_rebuild(
        self,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
        remaining_gates: List[GateSpec],
        trigger: ReplanTrigger,
    ) -> None:
        """Spawn the single background rebuild worker. The worker computes a
        new trajectory from a snapshot of the current state and publishes the
        result; it touches no shared pipeline state besides the guarded
        handoff."""
        self._rebuild_in_flight = True
        pos = tuple(position)
        vel = tuple(velocity)
        gates = list(remaining_gates)

        def _worker():
            try:
                traj = self._compute_trajectory(pos, vel, gates)
            except Exception:
                logger.exception("Async replan rebuild failed; keeping prior.")
                traj = None
            with self._rebuild_lock:
                self._rebuild_result = (traj, trigger)

        self._rebuild_thread = threading.Thread(
            target=_worker, name="aigp-replan", daemon=True
        )
        self._rebuild_thread.start()

    def _apply_pending_rebuild(self, sim_time: float) -> None:
        """On the control thread: swap in a completed background rebuild, if
        any. Atomic from the loop's perspective (single trajectory
        assignment)."""
        with self._rebuild_lock:
            result = self._rebuild_result
            self._rebuild_result = None
        if result is None:
            return
        new_traj, trig = result
        self._rebuild_in_flight = False
        if new_traj is None:
            # Build failed — keep the old trajectory and let the still-pending
            # trigger retry on a later tick.
            return
        self.trajectory = new_traj
        self._ref_progress_time = 0.0
        self.replanner.mark_replanned(sim_time, trig)
        self._pending_trigger = None
        self._replan_count += 1
        self._last_replan_reasons = trig.reasons
        logger.info(
            "REPLAN #%d landed at t=%.2fs reasons=%s",
            self._replan_count, sim_time, trig.reasons,
        )

    def _update_state_estimate(
        self,
        telem: TelemetryState,
    ) -> Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        float,
    ]:
        """Fuse the AIGP-provided pose into the EKF and return control state.

        LOCAL_POSITION_NED is the sole position/velocity source. ODOMETRY's
        quaternion is treated as ground-truth attitude passthrough, because
        the current EKF does not observe orientation. Updates are gated by
        sim-clock stamps so repeated telemetry snapshots do not apply the
        same measurement multiple times.
        """
        position = telem.position_ned
        velocity = telem.velocity_ned
        roll, pitch, yaw = telem.orientation.to_euler()

        if not self.config.use_ekf:
            return position, velocity, yaw

        reset_counter = telem.odom_reset_counter
        reset_changed = (
            reset_counter is not None
            and self._last_odom_reset_counter is not None
            and reset_counter != self._last_odom_reset_counter
        )
        init_time_s = _telemetry_timestamp_s(telem)
        if not self._ekf_live_initialized or reset_changed:
            self.ekf.initialize(
                position,
                velocity,
                orientation=(roll, pitch, yaw),
                timestamp_s=init_time_s,
            )
            self._ekf_live_initialized = True
            self._last_lpn_stamp_ms = telem.lpn_time_boot_ms
            self._last_odom_reset_counter = reset_counter
            return self.ekf.position, self.ekf.velocity, yaw

        self.ekf.set_orientation(roll, pitch, yaw)
        if telem.imu is not None:
            self.ekf.predict(
                telem.imu.accel,
                telem.imu.gyro,
                telem.imu.timestamp_us / 1e6,
            )

        lpn_stamp = telem.lpn_time_boot_ms
        source_healthy = telem.odom_quality is None or telem.odom_quality >= 50
        if lpn_stamp is not None and lpn_stamp != self._last_lpn_stamp_ms:
            if source_healthy:
                self.ekf.update_odometry(position, velocity)
            self._last_lpn_stamp_ms = lpn_stamp
        self._last_odom_reset_counter = reset_counter

        return self.ekf.position, self.ekf.velocity, yaw

    def _check_telemetry_freshness(self, telem: TelemetryState) -> None:
        """Watchdog for a frozen/dead telemetry feed.

        Increments a counter every control tick on which the telemetry
        timestamp has not advanced (or is missing). When the feed has been
        frozen for ``_telem_stale_tick_limit`` consecutive ticks the
        controller is provably flying on a stale state estimate, so we log a
        single loud error per stall. ``_telem_frozen_ticks`` accumulates the
        total for the end-of-run summary. This only observes telemetry; it
        does not alter the commands sent to the vehicle.
        """
        stamp = telem.timestamp_us
        if stamp is not None and stamp != self._last_telem_stamp_us:
            # Fresh sample — reset the consecutive-stall counter.
            self._last_telem_stamp_us = stamp
            self._telem_stale_ticks = 0
            self._telem_stale_warned = False
            return

        # Same (or missing) timestamp as last tick: the feed has not updated.
        self._telem_stale_ticks += 1
        self._telem_frozen_ticks += 1
        if (
            self._telem_stale_ticks >= self._telem_stale_tick_limit
            and not self._telem_stale_warned
        ):
            self._telem_stale_warned = True
            logger.error(
                "Telemetry feed FROZEN: timestamp %s has not advanced for %d "
                "control ticks (~%.1f s). The controller is flying on a stale "
                "state estimate — commands will not respond to the vehicle's "
                "actual motion (this is the 'spinning in circles' failure mode). "
                "Check the MAVLink RX subscription / sim publish rate.",
                stamp, self._telem_stale_ticks,
                self._telem_stale_ticks / max(1, self.config.target_hz),
            )

    def _process_detection(
        self,
        frame: CameraFrame,
        position: Tuple[float, float, float],
        yaw: float,
    ) -> bool:
        """
        Run gate detection and PnP pose estimation.

        Returns True only if a detection was successfully associated with
        the current target gate — not merely "some gate was detected".
        The prior behaviour returned True for any nonempty detection
        list, and unconditionally anchored PnP to ``sequencer.current_gate``
        regardless of which gate the detector actually saw. That is a
        dangerous EKF input if another gate (or a false positive) ranks
        first by confidence: the estimator gets a drone position anchored
        to the wrong gate's world pose.

        Association strategy: for each detection we run PnP once, convert
        the pose to an implied drone position anchored to the *current*
        target gate, and accept only when that position is within
        ``_pnp_gate_match_radius_m`` of the drone's current EKF-estimated
        position. That collapses to "the detection is geometrically
        consistent with seeing the current gate from here", which is
        what we want. Misassociated detections fall outside the radius
        and are discarded silently.

        Phase 3 fixes retained: detector is instantiated once in
        __init__(), and detect_with_corners() feeds fitted corners
        directly to PnP without re-fitting.
        """
        if self._detector is None:
            return False

        detections = self._detector.detect_with_corners(frame.image)
        if not detections:
            return False

        gate = self.sequencer.current_gate
        if gate is None:
            return False

        gate_pos = np.array(gate.position, dtype=float)
        drone_pos = np.array(position, dtype=float)
        best_match: Optional[Tuple[float, np.ndarray]] = None

        for det in detections:
            corners = getattr(det, "corners", None)
            if not isinstance(corners, np.ndarray) or corners.shape != (4, 2):
                continue

            pose = self.pnp_estimator.estimate_gate_pose(corners)
            if pose is None or pose.reprojection_error >= 5.0:
                continue

            implied_drone_pos = self.pnp_estimator.gate_pose_to_drone_position(
                pose, gate.position, gate.yaw,
                self.ekf.orientation,
                gate_world_pitch=getattr(gate, 'pitch', 0.0),
                gate_world_roll=getattr(gate, 'roll', 0.0),
            )
            match_err = float(np.linalg.norm(
                np.array(implied_drone_pos, dtype=float) - drone_pos
            ))
            if match_err > self._pnp_gate_match_radius_m:
                # This detection is geometrically inconsistent with being
                # the current target gate — most likely a different gate
                # in frame. Skip it rather than poisoning the EKF.
                continue

            if best_match is None or match_err < best_match[0]:
                best_match = (match_err, np.array(implied_drone_pos, dtype=float))

        if best_match is None:
            return False

        if self.config.use_pnp:
            self.ekf.update_pnp_position(tuple(best_match[1]))

        # Also sanity-check proximity to the gate itself so we don't
        # count a matched detection as "seeing the current gate" when
        # the drone is still far upstream — this keeps the dropout
        # counter semantics tight against the gate being actively tracked.
        gate_range = float(np.linalg.norm(drone_pos - gate_pos))
        _ = gate_range  # currently advisory; reserved for future FOV logic
        return True


def _gate_normal(yaw: float, pitch: float = 0.0) -> Tuple[float, float, float]:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (cy * cp, sy * cp, -sp)


# iter-38: DIVERGENCE / INSTABILITY GUARD thresholds. Two independent signals,
# both validated against live captures, feed one consecutive-tick counter; once
# it reaches _GUARD_TRIP_TICKS the run aborts EARLY (writes its partial capture
# and SIM_RESETs) so it cannot keep logging garbage telemetry or risk wedging the
# sim into the stuck post-race state (DCGame mem balloons, SIM_RESET stops
# returning the track map) that needs a GUI restart — the failure that blocked
# iter-37.
#
# 1) INNER-LOOP INSTABILITY (the PRIMARY, validated signal): the iter-37
#    vert_gain=2.0 divergence sustained gyro |w| >4 rad/s for ~14 ticks (max
#    8.4), whereas EVERY clean run — and even a scrapey 84-collision run —
#    stayed under 3 rad/s (max 2.2). So |w| > 4.0 for 10 consecutive ticks is a
#    clean separator (no false positives observed) that catches a limit-cycle
#    tumble. Non-finite gyro also counts (itself a divergence signal).
# 2) OUT-OF-COURSE-BOX (a cheap secondary net for a true fly-away): real course
#    is x[-162,0] y[-10,10] z[-1,27] (NED); these generous bounds only trip on a
#    genuine fly-away (normal end-of-race stops on race_finished long before).
_GYRO_INSTABILITY_RADS = 4.0
_GUARD_TRIP_TICKS = 10
_COURSE_BOX_NED = ((-175.0, 15.0), (-15.0, 15.0), (-10.0, 35.0))


def _in_course_box(pos) -> bool:
    """True if pos (NED x,y,z) is inside the generous course bounding box."""
    try:
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    except (TypeError, IndexError, ValueError):
        return False
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = _COURSE_BOX_NED
    return (
        math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)
        and xlo <= px <= xhi and ylo <= py <= yhi and zlo <= pz <= zhi
    )


def _gyro_unstable(gyro) -> bool:
    """True if the body angular rate signals inner-loop instability — magnitude
    above _GYRO_INSTABILITY_RADS, or non-finite (itself a divergence signal)."""
    if gyro is None:
        return False  # no gyro this tick: don't accuse (other signals still run)
    try:
        vals = [float(c) for c in gyro]
    except (TypeError, ValueError):
        return True
    if not all(math.isfinite(v) for v in vals):
        return True
    return math.sqrt(sum(v * v for v in vals)) > _GYRO_INSTABILITY_RADS


def _merge_triggers(
    pending: Optional[ReplanTrigger], current: ReplanTrigger
) -> ReplanTrigger:
    """OR a deferred (cooldown-blocked) trigger into the current evaluation.

    Lets a replan reason that fired on an earlier tick survive until it is
    actually served, instead of being consumed by ``evaluate()``'s
    fire-once edge detection while the cooldown is still active.
    """
    if pending is None:
        return current
    return ReplanTrigger(
        gate_collision=pending.gate_collision or current.gate_collision,
        missed_gate=pending.missed_gate or current.missed_gate,
        off_track=pending.off_track or current.off_track,
        sustained_lateral_error=(
            pending.sustained_lateral_error or current.sustained_lateral_error
        ),
        crashed_gate_id=current.crashed_gate_id or pending.crashed_gate_id,
    )


def _telemetry_timestamp_s(telem: TelemetryState) -> float:
    if telem.imu is not None:
        return telem.imu.timestamp_us / 1e6
    if telem.odom_time_usec is not None:
        return telem.odom_time_usec / 1e6
    if telem.timestamp_us is not None:
        return telem.timestamp_us / 1e6
    return 0.0


def _ref_override_position(
    ref, target_position: Tuple[float, float, float]
):
    """Return a TrajectoryPoint with its position replaced by ``target_position``.

    Used by ``RacePipeline._control_callback`` when the sequencer signals
    RECOVERY: we want the tracker to fly *toward the recovery target*
    without discarding the precomputed reference's velocity/yaw hints.
    ``TrajectoryPoint`` is a frozen dataclass, so this rebuilds a shallow
    copy with the overridden position tuple.
    """
    from dataclasses import replace

    return replace(ref, position=tuple(target_position))


def _ref_override_yaw(ref, yaw: float):
    """Return a TrajectoryPoint with its yaw replaced by ``yaw``.

    Iter-035: the trajectory-race path pins the reference heading to the
    proven -X course heading (yaw=pi) regardless of the yaw the optimizer
    assigned, so the GeometricTracker's attitude-extraction frame matches the
    minimal controller's (the one the live sim's roll convention was measured
    against)."""
    from dataclasses import replace

    return replace(ref, yaw=float(yaw))
