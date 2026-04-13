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

        elapsed = time.time() - self._race_start_time

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

        # 2. Gate detection and PnP (if camera frame available)
        gate_detected = False
        if frame is not None and self.config.use_detection:
            gate_detected = self._process_detection(frame, position, yaw)

        # 3. Update gate sequencer
        passed = self.sequencer.update(position, gate_detected)
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

        # 5. Get reference from trajectory
        ref = self.trajectory.sample(elapsed)

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

        Returns True if the current target gate was detected.

        Phase 3 fixes:
        - Detector instantiated once in __init__(), not every frame
        - Uses detect_with_corners() for fitted quadrilateral corners
        - Passes fitted corners directly to PnP (no re-fitting via detect_gate_corners)
        """
        if self._detector is None:
            return False

        # Use detect_with_corners() for proper PnP-ready corners
        detections = self._detector.detect_with_corners(frame.image)

        if not detections:
            return False

        # Use the highest-confidence detection
        best = detections[0]

        # Feed fitted corners directly to PnP (no re-fitting needed)
        if self.config.use_pnp and best.corners is not None:
            corners = best.corners
            # Ensure corners are (4, 2) float64 array
            if isinstance(corners, np.ndarray) and corners.shape == (4, 2):
                pose = self.pnp_estimator.estimate_gate_pose(corners)
                if pose is not None and pose.reprojection_error < 5.0:
                    gate = self.sequencer.current_gate
                    if gate is not None:
                        drone_pos = self.pnp_estimator.gate_pose_to_drone_position(
                            pose, gate.position, gate.yaw,
                            self.ekf.orientation,
                            gate_world_pitch=getattr(gate, 'pitch', 0.0),
                        )
                        self.ekf.update_pnp_position(drone_pos)

        return True


def _gate_normal(yaw: float, pitch: float = 0.0) -> Tuple[float, float, float]:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (cy * cp, sy * cp, sp)
