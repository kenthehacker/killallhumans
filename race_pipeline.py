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
        _tracker_cfg = TrackerConfig(trace_features=self.config.trace_tracker_features)
        self.tracker = GeometricTracker(_tracker_cfg)
        self.simple_tracker = SimplePositionTracker(TrackerConfig())

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
        traj_optimizer = TrajectoryOptimizer(
            constraints=DroneConstraints(max_velocity=self.config.max_speed),
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
        session.should_stop = lambda: (
            self.sequencer is not None and (
                self.sequencer.is_complete
                or self.sequencer.is_disqualified
                or self.sequencer.is_timed_out
                or self.sequencer.last_crash is not None
            )
        )

        # Iter-002 review B1 (5/7 BLOCKER) + iter-003 M5 (4/7 MAJOR): the
        # 8-minute timeout check uses BOTH a monotonic wall-clock fallback
        # AND a sim-time path that takes precedence when telemetry carries
        # `timestamp_us`. Sim-time is preferred because a non-realtime
        # simulator (slowed down, paused, or fast-forwarded) would
        # false-trip or under-trip a wall-clock-only check.
        self._race_start_time = time.monotonic()
        self._race_start_sim_time_s: Optional[float] = None  # set on first telem tick
        self._ref_progress_time = 0.0
        self.sequencer.start()

        logger.info("Starting race...")
        metrics = await session.run()
        logger.info("Race finished: %s", metrics)

    def _control_callback(
        self,
        telem: TelemetryState,
        frame: Optional[CameraFrame],
    ) -> Optional[AttitudeCommand]:
        """
        Main control callback — called at target_hz.

        Returns an AttitudeCommand to send to the simulator.
        """
        if self.sequencer is None or self.trajectory is None:
            return None

        # 0. Frozen-telemetry watchdog (detection only).
        self._check_telemetry_freshness(telem)

        # 1. Update state estimation
        position, velocity, yaw = self._update_state_estimate(telem)

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

        if self.sequencer.is_complete:
            logger.info("All gates passed! Race complete.")
            return AttitudeCommand(0, 0, yaw, 0.4)  # hover

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
