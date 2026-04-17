#!/usr/bin/env python3
"""
Visual demo — runs the FULL new pipeline against the PyBullet simulation.

Pipeline:  competition adapter → estimation → planning → control → sequencing

Dual-view visualization:
  Left:   FPV camera with gate detection overlay (bboxes, PnP corners, distance)
  Right:  Top-down 2D map (drone pos, EKF estimate, racing line, gates, speed)

Real-time telemetry HUD:
  Speed, altitude, gate progress, EKF uncertainty, tracking error, loop frequency

Usage:
    python3 scripts/visual_demo.py --config sim_pybullet/configs/race_01.json
    python3 scripts/visual_demo.py --max-time 60 --sim-speed 2
    python3 scripts/visual_demo.py --no-ekf --geometric
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure repo root is on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# --- PyBullet sim ---
from sim_pybullet.env import DroneRaceEnv, RaceConfig
from sim_pybullet.sequencer import GateSequencer as SimSequencer
from simulation.model_types import Gate

# --- New pipeline modules ---
from estimation.ekf import DroneEKF, EKFConfig
from estimation.gate_pnp import CameraIntrinsics, GateGeometry, GatePnPEstimator
from estimation.state_predictor import StatePredictor, LatencyConfig
from planning.trajectory_optimizer import (
    DroneConstraints, GateWaypoint, RaceTrajectory, TrajectoryOptimizer, TrajectoryPoint,
)
from planning.racing_line import RacingLineOptimizer, SpeedProfiler
from control.mpc_tracker import SimplePositionTracker, GeometricTracker, TrackerConfig
from gate_sequencing.sequencer import (
    GateSequencer as NewGateSequencer, GateSpec, RaceState,
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def gates_to_specs(gates: List[Gate]) -> List[GateSpec]:
    return [
        GateSpec(
            gate_id=g.gate_id,
            position=(g.pose.x, g.pose.y, g.pose.z),
            yaw=g.pose.yaw, pitch=g.pose.pitch, roll=g.pose.roll,
            interior_width=g.config.interior_width_m,
            interior_height=g.config.interior_height_m,
            sequence_index=g.sequence_index or 0,
        )
        for g in gates
    ]


def gates_to_waypoints(gates: List[Gate]) -> List[GateWaypoint]:
    wps = []
    for g in gates:
        cy, sy = math.cos(g.pose.yaw), math.sin(g.pose.yaw)
        cp, sp = math.cos(g.pose.pitch), math.sin(g.pose.pitch)
        wps.append(GateWaypoint(
            position=(g.pose.x, g.pose.y, g.pose.z),
            normal=(cy * cp, sy * cp, sp),
            width=g.config.interior_width_m,
            height=g.config.interior_height_m,
            yaw=g.pose.yaw,
        ))
    return wps


def gate_opening_corners(gate: Gate) -> np.ndarray:
    """4 inner corners in world coordinates."""
    hw = gate.config.interior_width_m / 2.0
    hh = gate.config.interior_height_m / 2.0
    local = [(0, -hw, -hh), (0, hw, -hh), (0, hw, hh), (0, -hw, hh)]
    p = gate.pose
    cr, sr = math.cos(p.roll), math.sin(p.roll)
    cp, sp = math.cos(p.pitch), math.sin(p.pitch)
    cy, sy = math.cos(p.yaw), math.sin(p.yaw)
    out = []
    for lx, ly, lz in local:
        x1, y1, z1 = lx, cr * ly - sr * lz, sr * ly + cr * lz
        x2, y2, z2 = cp * x1 + sp * z1, y1, -sp * x1 + cp * z1
        x3, y3, z3 = cy * x2 - sy * y2, sy * x2 + cy * y2, z2
        out.append([x3 + p.x, y3 + p.y, z3 + p.z])
    return np.array(out)


def _put_text(img, text, pos, scale, color, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────
# Top-down 2D map renderer
# ──────────────────────────────────────────────────────────────────

class TopDownMap:
    """Renders a 2D bird's-eye view of the race course."""

    def __init__(self, width=640, height=480,
                 bounds_min=(-5.0, -15.0), bounds_max=(60.0, 15.0)):
        self.w, self.h = width, height
        self.bmin, self.bmax = bounds_min, bounds_max
        self.drone_trail: deque = deque(maxlen=600)
        self.ekf_trail: deque = deque(maxlen=600)

    def _w2px(self, x: float, y: float) -> Tuple[int, int]:
        fx = (x - self.bmin[0]) / (self.bmax[0] - self.bmin[0])
        fy = (y - self.bmin[1]) / (self.bmax[1] - self.bmin[1])
        return int(fx * (self.w - 20) + 10), int((1 - fy) * (self.h - 40) + 10)

    def render(self, drone_pos, drone_yaw, ekf_pos, gates, passed_ids,
               current_gate_id, trajectory, speed_profile_speeds,
               wp_positions) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 30, dtype=np.uint8)

        # --- trajectory with speed-based colouring ---
        if trajectory is not None:
            step = max(1, len(trajectory.points) // 400)
            pts = trajectory.points[::step]
            for i in range(len(pts) - 1):
                p0 = self._w2px(pts[i].position[0], pts[i].position[1])
                p1 = self._w2px(pts[i + 1].position[0], pts[i + 1].position[1])
                sp = math.sqrt(sum(v ** 2 for v in pts[i].velocity))
                t = min(sp / 12.0, 1.0)
                cv2.line(img, p0, p1, (int(255 * (1 - t)), 80, int(255 * t)), 1, cv2.LINE_AA)

        # --- gates ---
        for gate in gates:
            px, py = self._w2px(gate.pose.x, gate.pose.y)
            is_passed = gate.gate_id in passed_ids
            is_cur = gate.gate_id == current_gate_id

            color = (0, 255, 0) if is_cur else (80, 80, 80) if is_passed else (200, 200, 200)
            rad = 8 if is_cur else 6
            thick = 2 if is_cur else 1
            cv2.circle(img, (px, py), rad, color, thick)

            # normal indicator
            yaw = gate.pose.yaw
            cv2.line(img, (px, py),
                     (int(px + math.cos(yaw) * 10), int(py - math.sin(yaw) * 10)),
                     color, 1)
            cv2.putText(img, gate.gate_id.replace("gate-", ""),
                        (px + 10, py - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, color, 1, cv2.LINE_AA)

        # --- drone trail ---
        self.drone_trail.append((drone_pos[0], drone_pos[1]))
        for i in range(1, len(self.drone_trail)):
            a = i / len(self.drone_trail)
            cv2.line(img,
                     self._w2px(*self.drone_trail[i - 1]),
                     self._w2px(*self.drone_trail[i]),
                     (0, int(255 * a), int(255 * a)), 1, cv2.LINE_AA)

        # --- EKF trail ---
        if ekf_pos is not None:
            self.ekf_trail.append((ekf_pos[0], ekf_pos[1]))
            for i in range(1, len(self.ekf_trail)):
                a = i / len(self.ekf_trail)
                cv2.line(img,
                         self._w2px(*self.ekf_trail[i - 1]),
                         self._w2px(*self.ekf_trail[i]),
                         (int(200 * a), 0, int(200 * a)), 1, cv2.LINE_AA)

        # --- drone marker + heading ---
        dx, dy = self._w2px(drone_pos[0], drone_pos[1])
        cv2.circle(img, (dx, dy), 5, (0, 255, 255), -1)
        cv2.arrowedLine(img, (dx, dy),
                        (int(dx + math.cos(drone_yaw) * 14),
                         int(dy - math.sin(drone_yaw) * 14)),
                        (0, 255, 255), 2, tipLength=0.4)

        # --- EKF marker ---
        if ekf_pos:
            ex, ey = self._w2px(ekf_pos[0], ekf_pos[1])
            cv2.circle(img, (ex, ey), 4, (255, 0, 255), -1)

        # --- legend ---
        _put_text(img, "Cyan=Drone  Magenta=EKF  Green=Target",
                  (10, self.h - 10), 0.32, (160, 160, 160))
        _put_text(img, "Trajectory: Blue=slow  Red=fast",
                  (10, self.h - 25), 0.32, (160, 160, 160))

        return img


# ──────────────────────────────────────────────────────────────────
# Main visual demo
# ──────────────────────────────────────────────────────────────────

class VisualDemo:
    """Runs the full new pipeline against PyBullet with live dual-view viz."""

    def __init__(self, config_path: str, max_time=120.0, sim_speed=1.0,
                 gui=False, use_ekf=True, use_geometric=False,
                 no_render=False):
        self.max_time = max_time
        self.sim_speed = sim_speed
        self.use_ekf = use_ekf
        self.no_render = no_render

        # ── Sim environment ──
        race_config = DroneRaceEnv.load_config(config_path)
        self.env = DroneRaceEnv(race_config=race_config, gui=gui)
        self.race_config = race_config

        # Sim sequencer (for gate highlight management)
        self.sim_seq = SimSequencer(race_config.gates)
        for g in race_config.gates:
            self.env.dim_gate(g.gate_id)
        f = self.sim_seq.current_gate
        if f:
            self.env.highlight_gate(f.gate_id)

        # ── New pipeline modules ──
        start_pos = race_config.start_position
        gate_specs = gates_to_specs(race_config.gates)
        gate_waypoints = gates_to_waypoints(race_config.gates)

        # New gate sequencer
        self.sequencer = NewGateSequencer(gate_specs)
        self.sequencer.start()

        # EKF
        self.ekf = DroneEKF(EKFConfig())
        self.ekf.initialize(start_pos, (0, 0, 0), timestamp_s=0.0)

        # PnP estimator
        cam = CameraIntrinsics.from_fov(self.env.drone.config.camera_fov, 640, 480)
        g0 = race_config.gates[0].config
        self.pnp = GatePnPEstimator(cam, GateGeometry(g0.interior_width_m, g0.interior_height_m))

        # State predictor
        self.state_predictor = StatePredictor(LatencyConfig())

        # ── Trajectory planning ──
        print("Optimizing racing line...")
        rl_opt = RacingLineOptimizer()
        opt_wps = rl_opt.optimize(gate_waypoints, start_pos)

        profiler = SpeedProfiler(max_speed=10.0)
        self.wp_positions = [start_pos] + [g.position for g in opt_wps]
        self.speeds = profiler.profile(self.wp_positions)
        print(f"  Speed profile: min={min(self.speeds):.1f} max={max(self.speeds):.1f} m/s")

        print("Computing time-optimal trajectory...")
        traj_opt = TrajectoryOptimizer(
            constraints=DroneConstraints(max_velocity=10.0),
            dt_sample=0.02,
        )
        self.trajectory = traj_opt.optimize(opt_wps, start_pos, (0, 0, 0))
        print(f"  Trajectory: {self.trajectory.total_time:.1f}s, "
              f"{len(self.trajectory.points)} points")

        # ── Controller ──
        # Use a tracker for HUD readout; actual sim stepping uses GPDDrone.step()
        # which internally runs DSLPIDControl.
        tc = TrackerConfig(
            kp_xy=6.0, kd_xy=4.0, kp_z=8.0, kd_z=5.0,
            mass=0.027, gravity=9.81, max_thrust_n=0.6,
        )
        self.tracker = GeometricTracker(tc) if use_geometric else SimplePositionTracker(tc)

        # ── Visualization ──
        bmin = race_config.field_bounds_min
        bmax = race_config.field_bounds_max
        self.topdown = TopDownMap(
            bounds_min=(bmin[0] - 2, bmin[1] - 2),
            bounds_max=(bmax[0] + 2, bmax[1] + 2),
        )

        # Timing / metrics
        self._loop_times: deque = deque(maxlen=120)
        self._tracking_errors: deque = deque(maxlen=200)

        ctrl_freq = self.env.drone.config.ctrl_freq
        self._render_interval = max(1, round(ctrl_freq / 30))
        self._steps_per_loop = max(1, int(sim_speed))

        # ── CSV telemetry logging ──
        logs_dir = _REPO / "logs"
        logs_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = logs_dir / f"visual_demo_{ts}.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_columns = [
            "sim_time", "step_count",
            "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
            "roll", "pitch", "yaw",
            "ref_pos_x", "ref_pos_y", "ref_pos_z",
            "ref_vel_x", "ref_vel_y", "ref_vel_z",
            "target_pos_x", "target_pos_y", "target_pos_z",
            "target_vel_x", "target_vel_y", "target_vel_z",
            "tracking_error_m", "current_gate_id", "gates_passed",
            "target_source", "loop_dt_ms",
        ]
        self._csv_writer.writerow(self._csv_columns)
        self._csv_flush_counter = 0

    # ──────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        print(f"\n{'='*55}")
        print(f"  AI Grand Prix — Visual Pipeline Demo")
        print(f"  Gates: {len(self.race_config.gates)}  "
              f"Max time: {self.max_time}s  Sim speed: {self.sim_speed}x")
        print(f"  EKF: {'ON' if self.use_ekf else 'OFF'}  "
              f"Controller: {type(self.tracker).__name__}")
        print(f"  Press Q to quit, R to reset")
        print(f"{'='*55}\n")

        while True:
            loop_t0 = time.perf_counter()
            sim_time = self.env.get_sim_time()

            # 1. Drone state
            sd = self.env.drone.get_state()
            pos, vel, yaw = sd["position"], sd["velocity"], sd["yaw"]
            rpy = sd["orientation_euler"]

            # 2. EKF
            ekf_pos = ekf_vel = None
            ekf_unc = 0.0
            if self.use_ekf:
                gyro = sd.get("angular_velocity", (0, 0, 0))
                self.ekf.predict((0, 0, -9.81), gyro, sim_time)
                self.ekf.update_odometry(pos, vel)
                ekf_pos, ekf_vel = self.ekf.position, self.ekf.velocity
                ekf_unc = self.ekf.position_uncertainty

            # 3. Gate sequencers
            sim_passed = self.sim_seq.update(pos)
            if sim_passed:
                self.env.dim_gate(sim_passed.gate_id)
                nxt = self.sim_seq.current_gate
                if nxt:
                    self.env.highlight_gate(nxt.gate_id)

            new_passed = self.sequencer.update(pos)
            if new_passed:
                print(f"  PASSED {new_passed.gate_id} "
                      f"[{self.sequencer.gates_passed}/{self.sequencer.total_gates}] "
                      f"t={sim_time:.2f}s")

            # 4. Termination checks
            if self.sequencer.is_complete:
                print(f"\nRace complete! All {self.sequencer.total_gates} gates "
                      f"in {sim_time:.2f}s sim time")
                break
            if sim_time > self.max_time:
                print(f"\nTime limit ({self.max_time}s). "
                      f"Gates: {self.sequencer.gates_passed}/{self.sequencer.total_gates}")
                break
            if pos[2] < 0.05:
                print(f"\nCrash! Alt={pos[2]:.2f}m")
                break

            # 5. Trajectory reference
            ref = self.trajectory.sample(sim_time)
            closest = self.trajectory.find_closest(pos)
            trk_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, closest.position)))
            self._tracking_errors.append(trk_err)

            # 6. Compute target for GPDDrone.step()
            target_pos = ref.position
            target_vel = ref.velocity
            target_yaw = ref.yaw
            target_source = "trajectory"

            # Past the trajectory end? Navigate directly to next gate.
            if sim_time > self.trajectory.total_time and not self.sequencer.is_complete:
                gate = self.sequencer.current_gate
                if gate:
                    gpos = np.array(gate.position)
                    dpos = np.array(pos)
                    direction = gpos - dpos
                    dist = float(np.linalg.norm(direction))
                    if dist > 0.1:
                        target_pos = tuple(gpos)
                        target_vel = tuple(direction / dist * min(dist * 2, 5.0))
                        target_yaw = float(math.atan2(direction[1], direction[0]))
                        target_source = "gate_fallback"

            if self.sequencer.should_slow_down():
                target_vel = tuple(v * 0.3 for v in target_vel)

            # 7. Step physics
            for _ in range(self._steps_per_loop):
                self.env.drone.step(target_pos, target_vel, target_yaw)

            # 8. Timing
            loop_dt = time.perf_counter() - loop_t0
            self._loop_times.append(loop_dt)

            # 8b. CSV telemetry row
            cur_gate = self.sequencer.current_gate
            cur_gate_id = cur_gate.gate_id if cur_gate else "none"
            self._csv_writer.writerow([
                f"{sim_time:.4f}", self.env.step_count,
                f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
                f"{vel[0]:.4f}", f"{vel[1]:.4f}", f"{vel[2]:.4f}",
                f"{rpy[0]:.4f}", f"{rpy[1]:.4f}", f"{rpy[2]:.4f}",
                f"{ref.position[0]:.4f}", f"{ref.position[1]:.4f}", f"{ref.position[2]:.4f}",
                f"{ref.velocity[0]:.4f}", f"{ref.velocity[1]:.4f}", f"{ref.velocity[2]:.4f}",
                f"{target_pos[0]:.4f}", f"{target_pos[1]:.4f}", f"{target_pos[2]:.4f}",
                f"{target_vel[0]:.4f}", f"{target_vel[1]:.4f}", f"{target_vel[2]:.4f}",
                f"{trk_err:.4f}", cur_gate_id, self.sequencer.gates_passed,
                target_source, f"{loop_dt * 1000:.2f}",
            ])
            self._csv_flush_counter += 1
            if self._csv_flush_counter % 240 == 0:
                self._csv_file.flush()

            # 9. Render
            if not self.no_render and self.env.step_count % self._render_interval == 0:
                self._render(sd, sim_time, ref, ekf_pos, ekf_unc, trk_err)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nQuit by user")
                    break
                if key == ord("r"):
                    self._reset()

        # Close CSV telemetry
        self._csv_file.flush()
        self._csv_file.close()
        print(f"Telemetry CSV: {self._csv_path}")

        if not self.no_render:
            cv2.destroyAllWindows()
        self.env.close()

        avg_trk = (sum(self._tracking_errors) / len(self._tracking_errors)
                    if self._tracking_errors else 0)
        avg_hz = (1.0 / (sum(self._loop_times) / len(self._loop_times))
                  if self._loop_times else 0)
        return {
            "gates_passed": self.sequencer.gates_passed,
            "total_gates": self.sequencer.total_gates,
            "sim_time": self.env.get_sim_time(),
            "complete": self.sequencer.is_complete,
            "avg_tracking_error": avg_trk,
            "avg_loop_hz": avg_hz,
            "csv_path": str(self._csv_path),
        }

    # ──────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────

    def _render(self, sd, sim_time, ref, ekf_pos, ekf_unc, trk_err):
        pos, vel, yaw = sd["position"], sd["velocity"], sd["yaw"]
        speed = math.sqrt(sum(v ** 2 for v in vel))

        # ── FPV + overlay ──
        fpv = self.env.drone.get_camera_image()
        self._draw_fpv_overlay(fpv, sd)
        self._draw_hud_fpv(fpv, sim_time, speed, pos[2], yaw, ekf_unc, trk_err)

        # ── Top-down map ──
        cur_id = self.sequencer.current_gate.gate_id if self.sequencer.current_gate else None
        passed_ids = list(self.sequencer._passed)
        topdown = self.topdown.render(
            pos, yaw, ekf_pos, self.race_config.gates,
            passed_ids, cur_id, self.trajectory, self.speeds, self.wp_positions,
        )
        self._draw_hud_map(topdown, sim_time, speed, ref)

        # ── Combine side-by-side ──
        h1, w1 = fpv.shape[:2]
        h2, w2 = topdown.shape[:2]
        th = max(h1, h2)
        if h1 != th:
            fpv = cv2.resize(fpv, (int(w1 * th / h1), th))
        if h2 != th:
            topdown = cv2.resize(topdown, (int(w2 * th / h2), th))
        cv2.imshow("AI Grand Prix - Visual Demo", np.hstack([fpv, topdown]))

    def _draw_fpv_overlay(self, fpv, sd):
        """Gate bounding boxes, PnP corner dots, and distance labels."""
        cur = self.sim_seq.current_gate
        passed = set(self.sim_seq.passed_gate_ids)
        dp = np.array(sd["position"])

        for gate in self.race_config.gates:
            corners_3d = gate_opening_corners(gate)
            proj = self.env.drone.project_points_to_fpv(corners_3d)
            if np.any(proj[:, 2] <= 0):
                continue

            pts = proj[:, :2]
            xmn, ymn = pts.min(axis=0)
            xmx, ymx = pts.max(axis=0)
            h, w = fpv.shape[:2]
            if xmx < 0 or xmn > w or ymx < 0 or ymn > h:
                continue

            x1 = int(max(0, xmn)); y1 = int(max(0, ymn))
            x2 = int(min(w - 1, xmx)); y2 = int(min(h - 1, ymx))

            is_tgt = cur and gate.gate_id == cur.gate_id
            is_done = gate.gate_id in passed
            color = (0, 255, 0) if is_tgt else (80, 80, 80) if is_done else (180, 180, 180)
            thick = 2 if is_tgt else 1
            cv2.rectangle(fpv, (x1, y1), (x2, y2), color, thick)

            # PnP corner markers
            for px, py, _ in proj:
                ix, iy = int(px), int(py)
                if 0 <= ix < w and 0 <= iy < h:
                    cv2.circle(fpv, (ix, iy), 3, (0, 255, 255), -1)

            gp = np.array([gate.pose.x, gate.pose.y, gate.pose.z])
            dist = float(np.linalg.norm(gp - dp))
            lbl = gate.gate_id + (f" [{dist:.1f}m]" if is_tgt else "")
            _put_text(fpv, lbl, (x1, y1 - 5), 0.38, color)

    def _draw_hud_fpv(self, img, sim_time, speed, alt, yaw, ekf_unc, trk_err):
        gate = self.sequencer.current_gate
        gname = gate.gate_id if gate else "DONE"
        gp, gt = self.sequencer.gates_passed, self.sequencer.total_gates
        hz = (1.0 / (sum(self._loop_times) / len(self._loop_times))
              if self._loop_times else 0)

        lines = [
            ("FPV", (0, 255, 255)),
            (f"Speed: {speed:.1f} m/s  Alt: {alt:.1f}m", (0, 255, 255)),
            (f"Target: {gname}  [{gp}/{gt}]",
             (0, 255, 0) if gate else (160, 160, 160)),
            (f"Time: {sim_time:.1f}s", (0, 255, 255)),
            (f"EKF unc: {ekf_unc:.3f}m", (255, 180, 0)),
            (f"Track err: {trk_err:.2f}m", (0, 180, 255)),
            (f"Loop: {hz:.0f} Hz", (200, 200, 200)),
        ]
        y = 20
        for txt, c in lines:
            _put_text(img, txt, (10, y), 0.42, c)
            y += 19

    def _draw_hud_map(self, img, sim_time, speed, ref):
        rs = math.sqrt(sum(v ** 2 for v in ref.velocity))
        lines = [
            (f"Time: {sim_time:.1f}s", (0, 255, 255)),
            (f"Speed: {speed:.1f} m/s  (ref: {rs:.1f})", (0, 255, 255)),
            (f"Ref: ({ref.position[0]:.1f}, {ref.position[1]:.1f}, "
             f"{ref.position[2]:.1f})", (160, 160, 160)),
            (f"Progress: {self.sequencer.gates_passed}/{self.sequencer.total_gates}",
             (0, 255, 0)),
        ]
        y = 15
        for txt, c in lines:
            _put_text(img, txt, (10, y), 0.33, c)
            y += 16

    # ──────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────

    def _reset(self):
        self.env.reset()
        self.sim_seq.reset()
        self.sequencer.reset()
        self.sequencer.start()
        self.ekf.initialize(self.race_config.start_position, (0, 0, 0), timestamp_s=0.0)
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
        self._loop_times.clear()
        self._tracking_errors.clear()
        self.topdown.drone_trail.clear()
        self.topdown.ekf_trail.clear()
        for g in self.race_config.gates:
            self.env.dim_gate(g.gate_id)
        f = self.sim_seq.current_gate
        if f:
            self.env.highlight_gate(f.gate_id)
        print("Demo reset")


# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Grand Prix — Visual Pipeline Demo",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(_REPO / "sim_pybullet" / "configs" / "race_01.json"),
        help="Race config JSON path",
    )
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument("--sim-speed", type=float, default=1.0)
    parser.add_argument("--pybullet-gui", action="store_true",
                        help="Open PyBullet native viewer")
    parser.add_argument("--no-ekf", action="store_true",
                        help="Disable EKF (raw sim state)")
    parser.add_argument("--geometric", action="store_true",
                        help="Use GeometricTracker instead of SimplePositionTracker")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable visualization (headless mode)")
    args = parser.parse_args()

    demo = VisualDemo(
        config_path=args.config,
        max_time=args.max_time,
        sim_speed=args.sim_speed,
        gui=args.pybullet_gui,
        use_ekf=not args.no_ekf,
        use_geometric=args.geometric,
        no_render=args.no_render,
    )
    results = demo.run()

    print(f"\n{'='*55}")
    print("Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print(f"{'='*55}")
    return 0 if results["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
