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
    """Configuration for the racing pipeline."""
    # Control rate
    target_hz: float = 100.0

    # Camera
    camera_fov_h: float = 90.0
    image_width: int = 640
    image_height: int = 480

    # Gate geometry
    gate_width: float = 1.2
    gate_height: float = 1.2

    # Detection
    use_detection: bool = True
    detection_confidence_threshold: float = 0.3

    # Planning
    max_speed: float = 12.0
    trajectory_dt: float = 0.01

    # Control mode
    use_geometric_tracker: bool = True

    # Estimation
    use_ekf: bool = True
    use_pnp: bool = True
    use_state_predictor: bool = True


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
        self.camera = CameraIntrinsics.from_fov(
            self.config.camera_fov_h,
            self.config.image_width,
            self.config.image_height,
        )
        self.gate_geometry = GateGeometry(
            self.config.gate_width, self.config.gate_height
        )
        self.pnp_estimator = GatePnPEstimator(self.camera, self.gate_geometry)
        self.ekf = DroneEKF(EKFConfig())
        self.state_predictor = StatePredictor(LatencyConfig())
        self.tracker = GeometricTracker(TrackerConfig())
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

        # Convert gate specs to waypoints for trajectory planning
        gate_waypoints = [
            GateWaypoint(
                position=g.position,
                normal=_gate_normal(g.yaw, g.pitch),
                width=g.interior_width,
                height=g.interior_height,
                yaw=g.yaw,
            )
            for g in gates
        ]

        # Optimize racing line
        logger.info("Optimizing racing line through %d gates...", len(gates))
        line_optimizer = RacingLineOptimizer()
        optimized_waypoints = line_optimizer.optimize(gate_waypoints, start_position)

        # Compute speed profile
        profiler = SpeedProfiler(max_speed=self.config.max_speed)
        waypoint_positions = [start_position] + [g.position for g in optimized_waypoints]
        speeds = profiler.profile(waypoint_positions)
        logger.info("Speed profile: min=%.1f max=%.1f m/s", min(speeds), max(speeds))

        # Generate time-optimal trajectory
        logger.info("Computing time-optimal trajectory...")
        traj_optimizer = TrajectoryOptimizer(
            constraints=DroneConstraints(max_velocity=self.config.max_speed),
            dt_sample=self.config.trajectory_dt,
        )
        self.trajectory = traj_optimizer.optimize(
            optimized_waypoints, start_position, start_velocity
        )
        logger.info(
            "Trajectory: %.1fs total, %d points",
            self.trajectory.total_time,
            len(self.trajectory.points),
        )

        # Initialize EKF
        self.ekf.initialize(start_position, start_velocity, timestamp_s=0.0)
        self._initialized = True

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
        session.should_stop = lambda: (
            self.sequencer is not None and self.sequencer.is_complete
        )

        self._race_start_time = time.time()
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

        # 1. Update state estimation
        position = telem.position_ned
        velocity = telem.velocity_ned
        yaw = telem.yaw

        if self.config.use_ekf:
            # Feed telemetry to EKF
            if telem.imu is not None:
                self.ekf.predict(
                    telem.imu.accel, telem.imu.gyro,
                    telem.imu.timestamp_us / 1e6,
                )
            self.ekf.update_odometry(position, velocity)
            position = self.ekf.position
            velocity = self.ekf.velocity
            _, _, yaw = self.ekf.orientation

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

        # 7. Speed reduction if needed
        if self.sequencer.should_slow_down():
            cmd = AttitudeCommand(
                roll_rad=cmd.roll_rad * 0.5,
                pitch_rad=cmd.pitch_rad * 0.5,
                yaw_rad=cmd.yaw_rad,
                thrust=cmd.thrust * 0.7,
            )

        return cmd

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
    return (cy * cp, sy * cp, sp)


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
