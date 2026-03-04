"""
Closed-loop drone racing simulation runner.

Ties together: Crazyflie physics (gym-pybullet-drones) -> camera rendering ->
gate detection -> gate sequencing -> DSLPIDControl -> physics.

Supports two detection modes:
  --use-detection: runs the real gate_detection pipeline on the rendered frame
  (default): uses sim metadata (known gate positions projected into camera frame)

Usage:
    python3 -m sim_pybullet.runner --config sim_pybullet/configs/race_01.json
    python3 -m sim_pybullet.runner --config sim_pybullet/configs/race_01.json --use-detection
    python3 -m sim_pybullet.runner --config sim_pybullet/configs/race_01.json --detector phase1
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

try:
    import pybullet as p
except ImportError:
    print("ERROR: pybullet is required. Install with: pip install pybullet")
    sys.exit(1)

from flight_control.types import DroneState, TargetState
from flight_control.adapter import gate_detection_to_target, CameraModel

from .env import DroneRaceEnv, RaceConfig
from .sequencer import GateSequencer


class RacingLine:
    """
    Catmull-Rom spline through race waypoints: smooth curved racing path.

    Passes through every waypoint with C1 continuity so the drone can
    anticipate upcoming gate directions rather than making rigid-angle turns.
    The caller gets a closest-point + lookahead query each control step, giving
    a continuously updated target that blends smoothly from gate to gate.
    """

    def __init__(self, waypoints: List[np.ndarray], samples_per_seg: int = 60):
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        self._wps = [np.array(w, dtype=float) for w in waypoints]
        self._n = samples_per_seg
        self._build()

    @staticmethod
    def _cr(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
            p3: np.ndarray, t: float) -> np.ndarray:
        """Catmull-Rom interpolation at parameter t in [0, 1)."""
        t2, t3 = t * t, t * t * t
        return 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

    def _build(self):
        wps, n = self._wps, self._n
        padded = [wps[0]] + wps + [wps[-1]]   # ghost endpoints for open spline

        pts, seg_idxs = [], []
        for seg in range(len(wps) - 1):
            p0, p1, p2, p3 = padded[seg], padded[seg + 1], padded[seg + 2], padded[seg + 3]
            for i in range(n):
                pts.append(self._cr(p0, p1, p2, p3, i / n))
                seg_idxs.append(seg)
        pts.append(wps[-1])
        seg_idxs.append(len(wps) - 2)

        self._pts = np.array(pts, dtype=float)
        self._seg_idxs = np.array(seg_idxs, dtype=int)

        # Cumulative arc lengths
        diffs = np.diff(self._pts, axis=0)
        self._arcs = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))]
        )
        self.total_length = float(self._arcs[-1])

        # Normalized tangents (central differences)
        T = np.empty_like(self._pts)
        T[0] = self._pts[1] - self._pts[0]
        T[-1] = self._pts[-1] - self._pts[-2]
        T[1:-1] = self._pts[2:] - self._pts[:-2]
        norms = np.linalg.norm(T, axis=1, keepdims=True)
        self._tangents = T / np.where(norms < 1e-9, 1.0, norms)

    @property
    def points(self) -> np.ndarray:
        return self._pts

    @property
    def seg_indices(self) -> np.ndarray:
        return self._seg_idxs

    def waypoint_arc(self, wp_idx: int) -> float:
        """Arc length at input waypoint wp_idx (0=start, 1=first gate, …)."""
        idx = min(wp_idx * self._n, len(self._arcs) - 1)
        return float(self._arcs[idx])

    def query(
        self,
        drone_pos: np.ndarray,
        lookahead_m: float,
        min_arc: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Find closest spline point at/after min_arc, advance by lookahead_m.

        Returns:
            target_position  — 3D point on the spline (np.ndarray)
            target_tangent   — unit direction at that point (np.ndarray)
            lateral_error    — distance from drone to the closest spline point (float)
        """
        search_start = int(np.searchsorted(self._arcs, max(0.0, min_arc - 1.0)))
        s_pts = self._pts[search_start:]
        s_arcs = self._arcs[search_start:]
        s_tan = self._tangents[search_start:]

        if len(s_pts) == 0:
            return self._pts[-1], self._tangents[-1], 0.0

        dists = np.linalg.norm(s_pts - drone_pos, axis=1)
        local = int(np.argmin(dists))
        lat_err = float(dists[local])
        base_arc = float(s_arcs[local])

        target_arc = min(base_arc + lookahead_m, self.total_length)
        la_idx = int(np.searchsorted(s_arcs, target_arc))
        la_idx = min(la_idx, len(s_pts) - 1)

        return s_pts[la_idx], s_tan[la_idx], lat_err


class RaceRunner:
    """
    Main simulation loop: physics -> detect -> plan -> control -> render.
    """

    def __init__(
        self,
        config_path: str,
        use_detection: bool = False,
        detector_type: str = "classical",
        gui: bool = False,
        max_time: float = 120.0,
        render_fps: int = 30,
        sim_speed: float = 1.0,
    ):
        self.use_detection = use_detection
        self.detector_type = detector_type
        self.max_time = max_time
        self.render_fps = render_fps
        self.sim_speed = sim_speed

        race_config = DroneRaceEnv.load_config(config_path)
        self.env = DroneRaceEnv(race_config=race_config, gui=gui)
        self.sequencer = GateSequencer(race_config.gates)

        # Smooth Catmull-Rom racing line: start position + all gate centers.
        # Pre-built once; used every control step for spline-following.
        _rl_wps = [np.array(race_config.start_position)] + [
            np.array([g.pose.x, g.pose.y, g.pose.z]) for g in race_config.gates
        ]
        self._racing_line = RacingLine(_rl_wps)

        # Dim all gates, then highlight only the current target
        for gate in race_config.gates:
            self.env.dim_gate(gate.gate_id)
        first = self.sequencer.current_gate
        if first:
            self.env.highlight_gate(first.gate_id)

        self._detector = None
        self._fused_detector = None
        if use_detection:
            self._init_detector(detector_type)

        self._camera_model = CameraModel(
            fov_horizontal_deg=self.env.drone.config.camera_fov,
        )

        # Spectator camera orbit state
        self._spec_dist = 5.0
        self._spec_yaw = 0.0    # offset from drone heading
        self._spec_pitch = 20.0  # degrees above horizontal

        # Render every N control steps so display runs at ~render_fps.
        # ctrl_freq=240, render_fps=30 → render every 8 steps.
        ctrl_freq = self.env.drone.config.ctrl_freq
        self._render_interval = max(1, round(ctrl_freq / render_fps))

        # Number of drone.step() calls per display loop iteration.
        # sim_speed > 1 makes the simulation run faster than real time.
        self._steps_per_loop = max(1, int(sim_speed))

        # PyBullet debug line IDs for path visualization
        self._target_line_id: int = -1
        self._racing_line_ids: list = []
        self._draw_racing_lines()

        # Telemetry logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = log_dir / f"race_{ts}.csv"
        self._log_file = open(self._log_path, "w", newline="")
        self._csv_writer = csv.writer(self._log_file)
        self._csv_writer.writerow([
            "sim_time", "step",
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
            "roll", "pitch", "yaw",
            "target_gate_id", "target_x", "target_y", "target_z",
            "dist_to_gate",
            "gates_passed", "total_gates",
        ])

    def _init_detector(self, detector_type: str):
        """Lazy-load the gate detection pipeline."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gate_detection" / "src"))

        if detector_type == "phase1":
            from phase1_detector import Phase1GateDetector
            self._detector = Phase1GateDetector()
        elif detector_type == "fused":
            from fused_gate_detector import FusedGateDetector
            model_path = str(
                Path(__file__).resolve().parent.parent
                / "gate_detection" / "models" / "best.pt"
            )
            self._fused_detector = FusedGateDetector(model_path=model_path)
        else:
            from gate_detector import GateDetector
            self._detector = GateDetector()

    def run(self) -> dict:
        """
        Run the full race loop. Returns a results dict on completion.
        """
        start_time = time.time()
        frame_count = 0
        sim_time = 0.0

        print(f"Race started: {self.sequencer.total_gates} gates")
        print(f"Detection mode: {'real pipeline' if self.use_detection else 'sim metadata'}")
        print(f"Sim speed: {self.sim_speed}x")

        while True:
            sim_time = self.env.get_sim_time()

            # 1. Get current drone state
            state_dict = self.env.drone.get_state()
            drone_state = DroneState(
                position=state_dict["position"],
                velocity=state_dict["velocity"],
                yaw=state_dict["yaw"],
            )

            # 2. Check gate pass-through
            passed = self.sequencer.update(drone_state.position)
            if passed:
                print(f"  PASSED gate {passed.gate_id} at t={sim_time:.2f}s")
                self.env.dim_gate(passed.gate_id)
                next_gate = self.sequencer.current_gate
                if next_gate:
                    self.env.highlight_gate(next_gate.gate_id)
                self._draw_racing_lines()  # refresh: passed segment turns grey

            if self.sequencer.is_complete:
                elapsed = time.time() - start_time
                print(f"\nRace complete! All {self.sequencer.total_gates} gates passed.")
                print(f"  Sim time: {sim_time:.2f}s")
                print(f"  Wall time: {elapsed:.2f}s")
                break

            if sim_time > self.max_time:
                print(f"\nTime limit reached ({self.max_time}s)")
                print(f"  Gates passed: {self.sequencer.gates_passed}/{self.sequencer.total_gates}")
                break

            # Crash detection
            if drone_state.position[2] < 0.1:
                print(f"\nCrashed! Altitude: {drone_state.position[2]:.2f}m")
                break
            if drone_state.position[2] > 50.0:
                print(f"\nFlew too high: {drone_state.position[2]:.2f}m")
                break

            # 3. Determine target
            target = self._get_target(drone_state)
            self._update_target_line(drone_state.position, target.position)

            # 4. Log telemetry (before stepping so we log state that triggered the command)
            self._log_frame(sim_time, self.env.step_count, state_dict, drone_state, target)

            # 5. Step physics + DSLPIDControl (one or more times for sim_speed multiplier)
            for _ in range(self._steps_per_loop):
                self.env.drone.step(target.position, target.velocity, target.yaw)

            # 6. Render at display rate
            if self.env.step_count % self._render_interval == 0:
                self._render_frame(drone_state, sim_time, frame_count)
                frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nQuit by user")
                    break
                if key == ord("r"):
                    self._reset()
                    continue
                self._handle_camera_keys(key)

        cv2.destroyAllWindows()
        self._log_file.close()
        print(f"Telemetry saved to {self._log_path}")
        self.env.close()

        return {
            "gates_passed": self.sequencer.gates_passed,
            "total_gates": self.sequencer.total_gates,
            "sim_time": sim_time,
            "complete": self.sequencer.is_complete,
        }

    def _get_target(self, drone_state: DroneState) -> TargetState:
        """Get the next target position from sequencer or detection."""
        gate = self.sequencer.current_gate
        if gate is None:
            return TargetState(position=drone_state.position)

        if self.use_detection and (self._detector or self._fused_detector):
            return self._target_from_detection(drone_state)

        return self._target_from_sim_metadata(drone_state)

    def _target_from_sim_metadata(self, drone_state: DroneState) -> TargetState:
        """
        Spline-following controller with lookahead and lateral-error speed boost.

        Instead of targeting a rigid approach waypoint per gate, we follow a
        smooth Catmull-Rom spline through all gates.  The 3m lookahead means the
        drone always steers toward a point ahead of its spline position, creating
        natural curves that anticipate the next gate's direction.

        Speed boost: if the drone drifts off the racing line, cruise speed is
        scaled up (up to +50%) so position correction is rapid.
        """
        gate = self.sequencer.current_gate
        if gate is None:
            return TargetState(position=drone_state.position)

        drone_pos = np.array(drone_state.position)
        gate_pos = np.array([gate.pose.x, gate.pose.y, gate.pose.z])
        dist_to_gate = float(np.linalg.norm(gate_pos - drone_pos))

        # Don't search spline points before the previously passed gate —
        # prevents the closest-point search from latching onto a segment behind us.
        min_arc = self._racing_line.waypoint_arc(self.sequencer.gates_passed)

        # Lookahead: target a point 3m ahead on the spline from closest position.
        # The spline tangent at that point is used as the velocity feedforward
        # and yaw target — the drone naturally carves smooth corners.
        LOOKAHEAD = 3.0
        target_pt, tangent, lateral_err = self._racing_line.query(
            drone_pos, LOOKAHEAD, min_arc=min_arc
        )
        target_yaw = float(math.atan2(float(tangent[1]), float(tangent[0])))

        # Base cruise speed, reduced when the spline tangent is far from current
        # heading (sharp remaining curve).  Spline already smooths most turns
        # so the reduction is mild compared to the old rigid-waypoint approach.
        yaw_diff = abs(math.atan2(
            math.sin(target_yaw - drone_state.yaw),
            math.cos(target_yaw - drone_state.yaw),
        ))
        base_speed = 2.0 * max(0.6, 1.0 - 0.4 * min(yaw_diff / math.radians(90), 1.0))

        # Throttle boost: lateral deviation from racing line → increase speed to
        # snap back.  1m off = +25%; 2m off = +50% (capped).
        speed = base_speed * (1.0 + min(lateral_err / 2.0, 0.5))

        # Within 1.5m of the gate: lock position target to the exact gate center
        # so the drone punches through cleanly rather than curving around it.
        if dist_to_gate < 1.5:
            target_pos = tuple(gate_pos)
        else:
            target_pos = tuple(target_pt)

        return TargetState(
            position=target_pos,
            velocity=tuple(tangent * speed),
            yaw=target_yaw,
        )

    def _target_from_detection(self, drone_state: DroneState) -> TargetState:
        """Run gate detection on the camera image and convert to target."""
        image = self.env.drone.get_camera_image()

        if self._fused_detector:
            fused_dets = self._fused_detector.detect(image)
            if fused_dets:
                det = fused_dets[0].final
                return gate_detection_to_target(det, drone_state, self._camera_model)
        elif self._detector:
            detections = self._detector.detect(image)
            if detections:
                det = detections[0]
                return gate_detection_to_target(det, drone_state, self._camera_model)

        # Fallback to sim metadata if detection found nothing
        return self._target_from_sim_metadata(drone_state)

    def _render_frame(
        self,
        drone_state: DroneState,
        sim_time: float,
        frame_count: int,
    ):
        """Render and display dual-camera view with HUD."""
        # 1st person (FPV)
        fpv = self.env.drone.get_camera_image()

        if self.use_detection and self._detector:
            detections = self._detector.detect(fpv)
            fpv = self._detector.get_debug_visualization(fpv, detections)
        elif self.use_detection and self._fused_detector:
            fused_dets = self._fused_detector.detect(fpv)
            fpv = self._fused_detector.get_debug_visualization(fpv, fused_dets)
        else:
            self._draw_sim_bboxes(fpv)

        # 3rd person (spectator) — must be rendered before calling project_points_to_spectator
        spectator = self.env.drone.get_spectator_image(
            distance=self._spec_dist,
            yaw_offset=self._spec_yaw,
            pitch_offset=self._spec_pitch,
        )

        # Racing-line overlay: project gate waypoints into camera views as 2D lines.
        # PyBullet addUserDebugLine only shows in the GUI window, not getCameraImage.
        self._draw_racing_lines_2d(spectator, use_spectator=True)
        self._draw_racing_lines_2d(fpv, use_spectator=False)

        # Recovery indicator
        if self.env.drone.in_recovery:
            cv2.putText(spectator, "RECOVERY", (10, spectator.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(spectator, "RECOVERY", (10, spectator.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 60, 255), 2, cv2.LINE_AA)

        # HUD overlay
        self._draw_hud(fpv, drone_state, sim_time, "FPV")
        self._draw_hud(spectator, drone_state, sim_time, "Spectator")

        # Combine side by side
        h1, w1 = fpv.shape[:2]
        h2, w2 = spectator.shape[:2]
        target_h = max(h1, h2)
        if h1 != target_h:
            fpv = cv2.resize(fpv, (int(w1 * target_h / h1), target_h))
        if h2 != target_h:
            spectator = cv2.resize(spectator, (int(w2 * target_h / h2), target_h))

        combined = np.hstack([fpv, spectator])
        cv2.imshow("Drone Race Simulation", combined)

    def _draw_sim_bboxes(self, fpv: np.ndarray):
        """Draw bounding boxes on FPV using known gate positions (sim metadata)."""
        for gate in self.sequencer.all_gates:
            corners_3d = self._gate_opening_corners(gate)
            projected = self.env.drone.project_points_to_fpv(corners_3d)

            # Skip if any point is behind camera
            if np.any(projected[:, 2] <= 0):
                continue

            pts_2d = projected[:, :2]
            x_min, y_min = pts_2d.min(axis=0)
            x_max, y_max = pts_2d.max(axis=0)

            h, w = fpv.shape[:2]
            if x_max < 0 or x_min > w or y_max < 0 or y_min > h:
                continue

            x1 = int(max(0, x_min))
            y1 = int(max(0, y_min))
            x2 = int(min(w - 1, x_max))
            y2 = int(min(h - 1, y_max))

            is_target = (
                self.sequencer.current_gate is not None
                and gate.gate_id == self.sequencer.current_gate.gate_id
            )
            is_passed = gate.gate_id in self.sequencer.passed_gate_ids

            if is_target:
                color = (0, 255, 0)
                thickness = 2
            elif is_passed:
                color = (100, 100, 100)
                thickness = 1
            else:
                color = (180, 180, 180)
                thickness = 1

            cv2.rectangle(fpv, (x1, y1), (x2, y2), color, thickness)
            label = gate.gate_id
            if is_target:
                label += " [TARGET]"
            cv2.putText(
                fpv, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    @staticmethod
    def _gate_opening_corners(gate) -> np.ndarray:
        """Compute the 4 inner corners of a gate in world coordinates.

        Gate lies in the local YZ plane (bars wide in Y, facing +X direction).
        Local corners are at (0, ±hw, ±hh) before world transform.
        """
        hw = gate.config.interior_width_m / 2.0
        hh = gate.config.interior_height_m / 2.0
        # Gate face is in local YZ plane, so X=0 for all corners.
        local_corners = [
            (0, -hw, -hh),
            (0,  hw, -hh),
            (0,  hw,  hh),
            (0, -hw,  hh),
        ]

        pose = gate.pose
        cr, sr = math.cos(pose.roll), math.sin(pose.roll)
        cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
        cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)

        world_corners = []
        for lx, ly, lz in local_corners:
            x1, y1, z1 = lx, cr * ly - sr * lz, sr * ly + cr * lz
            x2, y2, z2 = cp * x1 + sp * z1, y1, -sp * x1 + cp * z1
            x3, y3, z3 = cy * x2 - sy * y2, sy * x2 + cy * y2, z2
            world_corners.append([
                x3 + pose.x, y3 + pose.y, z3 + pose.z
            ])
        return np.array(world_corners)

    @staticmethod
    def _put_text(image: np.ndarray, text: str, pos, scale: float,
                  color, thickness: int = 1):
        """Render text with a black outline so it's readable on any background."""
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def _draw_hud(
        self,
        image: np.ndarray,
        state: DroneState,
        sim_time: float,
        label: str,
    ):
        """Draw heads-up display on a camera view."""
        h, w = image.shape[:2]
        speed = math.sqrt(sum(v ** 2 for v in state.velocity))
        alt = state.position[2]
        gate = self.sequencer.current_gate
        gate_name = gate.gate_id if gate else "DONE"

        lines = [
            (f"{label}", (0, 255, 255)),
            (f"Speed: {speed:.1f} m/s  Alt: {alt:.1f}m", (0, 255, 255)),
            (f"Target: {gate_name}  [{self.sequencer.gates_passed}/{self.sequencer.total_gates}]",
             (0, 255, 0) if gate else (180, 180, 180)),
            (f"Time: {sim_time:.1f}s", (0, 255, 255)),
        ]

        y = 20
        for text, color in lines:
            self._put_text(image, text, (10, y), 0.5, color)
            y += 22

        if gate:
            dx = gate.pose.x - state.position[0]
            dy = gate.pose.y - state.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            self._put_text(image, f"Dist to gate: {dist:.1f}m",
                           (10, h - 15), 0.45, (0, 220, 220))

        if label == "Spectator":
            self._put_text(image, "WASD: orbit  +/-: zoom  Q: quit  R: reset",
                           (10, h - 35), 0.35, (200, 200, 200))

    def _draw_racing_lines_2d(self, image: np.ndarray, use_spectator: bool):
        """
        Draw the smooth Catmull-Rom racing spline as 2D projected lines on the image.

        PyBullet addUserDebugLine only shows in the GUI window, not getCameraImage().
        This method projects the dense spline points through the camera matrices
        and draws them with cv2.line() — producing a smooth curved racing line.
        """
        all_gates = self.sequencer.all_gates
        if not all_gates:
            return

        # Subsample the spline (every 4th point) to keep projection cost low
        # while still producing a visually smooth curve.
        STEP = 4
        spline_pts = self._racing_line.points[::STEP]
        seg_idxs = self._racing_line.seg_indices[::STEP]

        if use_spectator:
            projected = self.env.drone.project_points_to_spectator(spline_pts)
        else:
            projected = self.env.drone.project_points_to_fpv(spline_pts)

        passed_ids = set(self.sequencer.passed_gate_ids)
        current_id = self.sequencer.current_gate.gate_id if self.sequencer.current_gate else None
        h, w = image.shape[:2]
        margin = 500

        for i in range(len(spline_pts) - 1):
            pt1 = projected[i]
            pt2 = projected[i + 1]
            if pt1[2] <= 0 or pt2[2] <= 0:
                continue

            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])

            if (max(x1, x2) < -margin or min(x1, x2) > w + margin
                    or max(y1, y2) < -margin or min(y1, y2) > h + margin):
                continue

            # Color by which gate this segment leads toward
            seg = int(seg_idxs[i])
            if seg < len(all_gates):
                gate = all_gates[seg]
                if gate.gate_id in passed_ids:
                    color = (80, 80, 80)       # grey — passed
                elif gate.gate_id == current_id:
                    color = (0, 220, 255)      # yellow — current target
                else:
                    color = (255, 120, 30)     # blue-orange — upcoming
            else:
                color = (80, 80, 80)

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    def _log_frame(self, sim_time, step, state_dict, drone_state, target):
        gate = self.sequencer.current_gate
        gate_id = gate.gate_id if gate else "DONE"
        tx, ty, tz = target.position
        dist = math.sqrt(
            (tx - drone_state.position[0]) ** 2
            + (ty - drone_state.position[1]) ** 2
            + (tz - drone_state.position[2]) ** 2
        )
        self._csv_writer.writerow([
            f"{sim_time:.4f}", step,
            f"{drone_state.position[0]:.4f}",
            f"{drone_state.position[1]:.4f}",
            f"{drone_state.position[2]:.4f}",
            f"{drone_state.velocity[0]:.4f}",
            f"{drone_state.velocity[1]:.4f}",
            f"{drone_state.velocity[2]:.4f}",
            f"{state_dict['roll']:.4f}",
            f"{state_dict['pitch']:.4f}",
            f"{state_dict['yaw']:.4f}",
            gate_id,
            f"{tx:.4f}", f"{ty:.4f}", f"{tz:.4f}",
            f"{dist:.4f}",
            self.sequencer.gates_passed,
            self.sequencer.total_gates,
        ])

    def _handle_camera_keys(self, key: int):
        """Adjust spectator camera orbit based on keyboard input."""
        if key in (81, 2, 63234) or key == ord("a"):
            self._spec_yaw -= 5.0
        elif key in (83, 3, 63235) or key == ord("d"):
            self._spec_yaw += 5.0
        elif key in (82, 0, 63232) or key == ord("w"):
            self._spec_pitch = min(80.0, self._spec_pitch + 5.0)
        elif key in (84, 1, 63233) or key == ord("s"):
            self._spec_pitch = max(-10.0, self._spec_pitch - 5.0)
        elif key == ord("=") or key == ord("+"):
            self._spec_dist = max(2.0, self._spec_dist - 1.0)
        elif key == ord("-"):
            self._spec_dist = min(25.0, self._spec_dist + 1.0)

    def _draw_racing_lines(self):
        """Draw the smooth racing spline in the PyBullet GUI window."""
        for lid in self._racing_line_ids:
            p.removeUserDebugItem(lid, physicsClientId=self.env.client)
        self._racing_line_ids.clear()

        all_gates = self.sequencer.all_gates
        if not all_gates:
            return

        passed = set(self.sequencer.passed_gate_ids)
        current_id = self.sequencer.current_gate.gate_id if self.sequencer.current_gate else None
        pts = self._racing_line.points
        seg_idxs = self._racing_line.seg_indices

        # Subsample to ~150 debug line segments for GPU/CPU budget
        step = max(1, len(pts) // 150)
        for i in range(0, len(pts) - step, step):
            seg = int(seg_idxs[min(i, len(seg_idxs) - 1)])
            if seg < len(all_gates):
                gate = all_gates[seg]
                if gate.gate_id in passed:
                    color = [0.4, 0.4, 0.4]
                elif gate.gate_id == current_id:
                    color = [1.0, 0.9, 0.0]
                else:
                    color = [0.2, 0.6, 1.0]
            else:
                color = [0.4, 0.4, 0.4]

            j = min(i + step, len(pts) - 1)
            lid = p.addUserDebugLine(
                pts[i].tolist(), pts[j].tolist(),
                color, lineWidth=2.0,
                physicsClientId=self.env.client,
            )
            self._racing_line_ids.append(lid)

    def _update_target_line(
        self,
        drone_pos: tuple,
        target_pos: tuple,
    ):
        """Draw/update the yellow line from drone to its current target each tick."""
        from_pt = list(drone_pos)
        to_pt = list(target_pos)
        if self._target_line_id == -1:
            self._target_line_id = p.addUserDebugLine(
                from_pt, to_pt,
                [1.0, 1.0, 0.0], lineWidth=2.0,
                physicsClientId=self.env.client,
            )
        else:
            p.addUserDebugLine(
                from_pt, to_pt,
                [1.0, 1.0, 0.0], lineWidth=2.0,
                replaceItemUniqueId=self._target_line_id,
                physicsClientId=self.env.client,
            )

    def _reset(self):
        """Reset the simulation."""
        self.env.reset()
        self.sequencer.reset()
        self._target_line_id = -1
        for gate in self.env.race_config.gates:
            self.env.dim_gate(gate.gate_id)
        first = self.sequencer.current_gate
        if first:
            self.env.highlight_gate(first.gate_id)
        self._draw_racing_lines()
        print("Simulation reset")


def main():
    parser = argparse.ArgumentParser(description="Drone race simulation")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "race_01.json"),
        help="Path to race config JSON",
    )
    parser.add_argument(
        "--use-detection",
        action="store_true",
        help="Use real gate detection pipeline on rendered frames",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="classical",
        choices=["classical", "fused", "phase1"],
        help="Which detector to use (only applies with --use-detection)",
    )
    parser.add_argument(
        "--pybullet-gui",
        action="store_true",
        help="Open PyBullet's own 3D viewer (not needed, we render our own views)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=120.0,
        help="Max simulation time in seconds",
    )
    parser.add_argument(
        "--sim-speed",
        type=float,
        default=1.0,
        help="Simulation speed multiplier (default 1x, higher=faster)",
    )
    args = parser.parse_args()

    runner = RaceRunner(
        config_path=args.config,
        use_detection=args.use_detection,
        detector_type=args.detector,
        gui=args.pybullet_gui,
        max_time=args.max_time,
        sim_speed=args.sim_speed,
    )
    results = runner.run()
    print(f"\nResults: {results}")
    return 0 if results["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
