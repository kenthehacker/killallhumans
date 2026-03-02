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
        """Compute target from known gate position with approach waypoint and pass-through velocity."""
        gate = self.sequencer.current_gate
        if gate is None:
            return TargetState(position=drone_state.position)

        gate_pos = np.array([gate.pose.x, gate.pose.y, gate.pose.z])
        drone_pos = np.array(drone_state.position)
        gate_normal = self.sequencer._gate_normal(gate.pose)

        CRUISE_SPEED = 2.0
        APPROACH_OFFSET = 2.0

        dist_to_gate = np.linalg.norm(gate_pos - drone_pos)

        if dist_to_gate > APPROACH_OFFSET * 1.5:
            target_pos = gate_pos - gate_normal * APPROACH_OFFSET
            # Approach phase: velocity and yaw point toward the waypoint
            to_target = target_pos - drone_pos
            to_target_xy = np.array([to_target[0], to_target[1], 0.0])
            dist_xy = np.linalg.norm(to_target_xy)
            if dist_xy > 0.1:
                approach_dir = to_target_xy / dist_xy
                target_vel = tuple(approach_dir * CRUISE_SPEED)
                target_yaw = math.atan2(float(approach_dir[1]), float(approach_dir[0]))
            else:
                target_vel = tuple(gate_normal * CRUISE_SPEED)
                target_yaw = math.atan2(float(gate_normal[1]), float(gate_normal[0]))
        else:
            target_pos = gate_pos
            # Close to gate: align with gate normal for pass-through
            target_vel = tuple(gate_normal * CRUISE_SPEED)
            target_yaw = math.atan2(float(gate_normal[1]), float(gate_normal[0]))

        return TargetState(
            position=tuple(target_pos),
            velocity=target_vel,
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

        # 3rd person (spectator)
        spectator = self.env.drone.get_spectator_image(
            distance=self._spec_dist,
            yaw_offset=self._spec_yaw,
            pitch_offset=self._spec_pitch,
        )

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
            f"{label}",
            f"Speed: {speed:.1f} m/s  Alt: {alt:.1f}m",
            f"Target: {gate_name}  [{self.sequencer.gates_passed}/{self.sequencer.total_gates}]",
            f"Time: {sim_time:.1f}s",
        ]

        y = 20
        for line in lines:
            cv2.putText(
                image, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA,
            )
            y += 22

        if gate:
            dx = gate.pose.x - state.position[0]
            dy = gate.pose.y - state.position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            cv2.putText(
                image, f"Dist to gate: {dist:.1f}m",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 200, 200), 1, cv2.LINE_AA,
            )

        if label == "Spectator":
            cv2.putText(
                image, "WASD: orbit  +/-: zoom  Q: quit",
                (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (150, 150, 150), 1, cv2.LINE_AA,
            )

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
        """Draw blue lines through all gate centers as a static racing-line preview."""
        for lid in self._racing_line_ids:
            p.removeUserDebugItem(lid, physicsClientId=self.env.client)
        self._racing_line_ids.clear()

        gates = self.sequencer.all_gates
        if not gates:
            return

        passed = set(self.sequencer.passed_gate_ids)
        waypoints = [list(self.env.race_config.start_position)] + [
            [g.pose.x, g.pose.y, g.pose.z] for g in gates
        ]

        for i in range(len(waypoints) - 1):
            gate_id = gates[i].gate_id if i < len(gates) else None
            color = [0.4, 0.4, 0.4] if gate_id and gate_id in passed else [0.2, 0.6, 1.0]
            lid = p.addUserDebugLine(
                waypoints[i], waypoints[i + 1],
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
