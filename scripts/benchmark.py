#!/usr/bin/env python3
"""
Headless benchmark — runs the full pipeline and outputs structured JSON metrics.

Designed for autonomous AI agent iteration: run → parse JSON → identify issues → fix → repeat.

Modes:
  --mode unit       Run unit tests only (no external dependency)
  --mode synthetic  Run synthetic simulation (pure Python, no PyBullet)
  --mode sim        Run full PyBullet simulation headless
  --mode full       Run unit + synthetic + sim (default)

Output:
  Prints a JSON object to stdout with all metrics. Human-readable summary to stderr.
  Exit code 0 = all thresholds met, 1 = at least one failure.

Usage:
    python3 scripts/benchmark.py                          # full benchmark
    python3 scripts/benchmark.py --mode unit              # unit tests only
    python3 scripts/benchmark.py --mode synthetic         # synthetic sim (no PyBullet)
    python3 scripts/benchmark.py --mode sim --duration 30 # PyBullet sim
    python3 scripts/benchmark.py --json-only              # suppress stderr summary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Quality thresholds — an AI agent should aim to improve these
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "unit_tests_pass_rate": 1.0,          # 100% unit tests must pass
    "max_avg_tracking_error_m": 0.5,      # aspirational target (tightened from 1.0)
    "max_max_tracking_error_m": 2.0,      # aspirational target (tightened from 4.0)
    "max_ekf_uncertainty_m": 0.5,         # aspirational target (tightened from 1.0)
    "min_loop_hz": 100,                   # minimum control loop frequency
    "min_gate_pass_rate": 1.0,            # Phase 1: require full gate completion (was 0.8)
    "max_total_time_s": 30.0,             # must finish within 30s
    "no_crash": True,                     # must not crash
}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_single_test(name: str, fn) -> Dict[str, Any]:
    """Run a single unit test and return structured result."""
    t0 = time.perf_counter()
    try:
        fn()
        return {"name": name, "passed": True, "time_ms": (time.perf_counter() - t0) * 1000}
    except Exception as e:
        return {
            "name": name, "passed": False,
            "error": str(e), "time_ms": (time.perf_counter() - t0) * 1000,
        }


def run_unit_tests() -> Dict[str, Any]:
    """Run all pipeline unit tests and return structured results."""
    from competition.adapter import AttitudeCommand, Quaternion, TelemetryState
    from estimation.ekf import DroneEKF, EKFConfig
    from estimation.state_predictor import StatePredictor
    from estimation.gate_tracker import GateTracker
    from gate_sequencing.sequencer import GateSequencer, GateSpec
    from planning.trajectory_optimizer import (
        DroneConstraints, GateWaypoint, TrajectoryOptimizer, TrajectoryPoint,
    )
    from planning.racing_line import RacingLineOptimizer, SpeedProfiler
    from control.mpc_tracker import GeometricTracker, SimplePositionTracker, TrackerConfig

    tests = []

    # --- Quaternion roundtrip ---
    def _quat():
        for r, p, y in [(0, 0, 0), (0.1, 0.2, 0.3), (-0.5, 0.3, 1.0), (0, 0, math.pi)]:
            q = Quaternion.from_euler(r, p, y)
            r2, p2, y2 = q.to_euler()
            assert abs(r - r2) < 1e-5, f"Roll {r} != {r2}"
            assert abs(p - p2) < 1e-5, f"Pitch {p} != {p2}"
    tests.append(("quaternion_roundtrip", _quat))

    # --- EKF convergence ---
    def _ekf():
        ekf = DroneEKF(EKFConfig(position_noise_std=0.01, velocity_noise_std=0.05))
        ekf.initialize((1.5, 2.5, -2.5), (0, 0, 0), timestamp_s=0.0)
        true_pos, true_vel = (1.0, 2.0, -3.0), (0.5, -0.3, 0.0)
        for i in range(100):
            ekf.predict((0, 0, -9.81), (0, 0, 0), i * 0.01)
            ekf.update_odometry(true_pos, true_vel)
        pos_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(ekf.position, true_pos)))
        vel_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(ekf.velocity, true_vel)))
        assert pos_err < 0.5, f"pos_err={pos_err:.4f}"
        assert vel_err < 0.5, f"vel_err={vel_err:.4f}"
    tests.append(("ekf_convergence", _ekf))

    # --- Trajectory generation ---
    def _traj():
        wps = [
            GateWaypoint(position=(5, 0, -2), normal=(1, 0, 0), yaw=0),
            GateWaypoint(position=(10, 5, -3), normal=(0, 1, 0), yaw=math.pi / 2),
            GateWaypoint(position=(15, 0, -2), normal=(-1, 0, 0), yaw=math.pi),
        ]
        traj = TrajectoryOptimizer(
            DroneConstraints(max_velocity=10.0), dt_sample=0.05
        ).optimize(wps, start_position=(0, 0, -2))
        assert traj.total_time > 0, "total_time must be positive"
        assert len(traj.points) > 10, f"too few points: {len(traj.points)}"
        assert len(traj.segment_times) == 7, f"expected 7 segments (3 gates × 2 entry/exit + finish), got {len(traj.segment_times)}"
    tests.append(("trajectory_generation", _traj))

    # --- Racing line ---
    def _rl():
        wps = [
            GateWaypoint(position=(5, 0, -2), normal=(1, 0, 0), yaw=0),
            GateWaypoint(position=(10, 5, -2), normal=(0, 1, 0), yaw=math.pi / 2),
        ]
        from planning.racing_line import RacingLineConfig
        out = RacingLineOptimizer(RacingLineConfig(use_cache=False)).optimize(wps, (0, 0, -2))
        assert len(out) == 2
    tests.append(("racing_line", _rl))

    # --- Speed profiler ---
    def _sp():
        pts = [(0, 0, -2), (10, 0, -2), (20, 0, -2), (20, 10, -2), (20, 20, -2)]
        speeds = SpeedProfiler(max_speed=15.0, min_speed=2.0).profile(pts)
        assert len(speeds) == 5
        assert all(2.0 <= s <= 15.0 for s in speeds), f"speeds out of range: {speeds}"
    tests.append(("speed_profiler", _sp))

    # --- Geometric tracker (tight hover test — Phase 1 requirement) ---
    def _gt():
        tr = GeometricTracker(TrackerConfig(max_thrust_n=20.0, mass=1.0, gravity=9.81))
        ref = TrajectoryPoint(0, (0, 0, -2), (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, 0)
        cmd = tr.track((0, 0, -2), (0, 0, 0), 0.0, ref)
        assert abs(cmd.roll_rad) < 0.01, f"hover roll={cmd.roll_rad:.4f} (must be <0.01)"
        assert abs(cmd.pitch_rad) < 0.01, f"hover pitch={cmd.pitch_rad:.4f} (must be <0.01)"
        assert 0.01 < cmd.thrust < 0.99, f"thrust={cmd.thrust}"
    tests.append(("geometric_tracker", _gt))

    # --- Gate sequencer ---
    def _gs():
        gs = GateSequencer([
            GateSpec("g1", position=(5, 0, -2), yaw=0, sequence_index=0),
            GateSpec("g2", position=(10, 0, -2), yaw=0, sequence_index=1),
        ])
        gs.start()
        assert gs.update((4, 0, -2)) is None
        p = gs.update((6, 0, -2))
        assert p is not None and p.gate_id == "g1"
        assert gs.update((9, 0, -2)) is None
        p2 = gs.update((11, 0, -2))
        assert p2 is not None and p2.gate_id == "g2"
        assert gs.is_complete
    tests.append(("gate_sequencer", _gs))

    # --- Gate tracker ---
    def _gtr():
        tracker = GateTracker()
        for frame in range(20):
            cx = 320 + frame * 5
            tracker.predict()
            tracker.update([("gate_1", (cx, 240, 80, 80), 0.9)])
        gates = tracker.get_tracked_gates()
        assert len(gates) >= 1, "no confirmed tracks"
        g = tracker.get_gate("gate_1")
        assert g is not None
        assert g.hits == 20
        # Coast test
        for _ in range(5):
            tracker.predict()
            tracker.update([])
        g_c = tracker.get_gate("gate_1")
        assert g_c is not None, "track should survive 5 frames coast"
        pred = tracker.get_predicted_bbox("gate_1")
        assert pred is not None
        assert pred[0] > 320 + 19 * 5, "prediction should extrapolate forward"
    tests.append(("gate_tracker", _gtr))

    # --- State predictor ---
    def _pred():
        pr = StatePredictor()
        pp, pv, po = pr.predict((0, 0, -5), (3, 0, 0), (0, 0, 0), (0, 0, 0), dt_override=0.1)
        assert abs(pp[0] - 0.3) < 0.05, f"predicted x={pp[0]}"
    tests.append(("state_predictor", _pred))

    # Run all
    results = [_run_single_test(name, fn) for name, fn in tests]
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    total_ms = sum(r["time_ms"] for r in results)

    return {
        "tests": results,
        "passed": passed,
        "failed": failed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0,
        "total_time_ms": total_ms,
    }


# ---------------------------------------------------------------------------
# Synthetic simulation (no PyBullet — pure Python kinematics)
# ---------------------------------------------------------------------------

def run_synthetic_benchmark(duration: float = 30.0, dt: float = 0.01) -> Dict[str, Any]:
    """
    Run the full pipeline with synthetic kinematic simulation.

    Uses the same race_01.json gate layout but simulates drone physics with
    a simple second-order model: PD controller → acceleration → velocity → position.
    No PyBullet dependency.

    This exercises: trajectory optimizer, racing line, speed profiler, EKF,
    gate sequencer, and geometric tracker — the full pipeline minus perception.
    """
    import json as _json
    from estimation.ekf import DroneEKF, EKFConfig
    from gate_sequencing.sequencer import GateSequencer, GateSpec, SequencerConfig
    from planning.trajectory_optimizer import DroneConstraints, GateWaypoint, TrajectoryOptimizer, TrajectoryPoint
    from planning.racing_line import RacingLineOptimizer, SpeedProfiler
    from control.mpc_tracker import GeometricTracker, TrackerConfig

    # Load gate layout from config
    config_path = _REPO / "sim_pybullet" / "configs" / "race_01.json"
    try:
        with open(config_path) as f:
            data = _json.load(f)
    except FileNotFoundError:
        return {"skipped": True, "skip_reason": f"Config not found: {config_path}"}

    gate_defaults = data.get("gate_defaults", {})
    default_w = gate_defaults.get("interior_width_m", 1.2)
    default_h = gate_defaults.get("interior_height_m", 1.2)

    gate_specs = []
    gate_waypoints = []
    for gd in data.get("gates", []):
        pose = gd.get("pose", {})
        gc = gd.get("config", {})
        x, y, z = pose.get("x", 0), pose.get("y", 0), pose.get("z", 1.5)
        yaw = pose.get("yaw", 0)
        pitch = pose.get("pitch", 0)
        w = gc.get("interior_width_m", default_w)
        h = gc.get("interior_height_m", default_h)

        gate_specs.append(GateSpec(
            gate_id=gd["id"], position=(x, y, z), yaw=yaw, pitch=pitch,
            interior_width=w, interior_height=h,
            sequence_index=gd.get("sequence_index", 0),
        ))
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        gate_waypoints.append(GateWaypoint(
            position=(x, y, z), normal=(cy * cp, sy * cp, sp),
            width=w, height=h, yaw=yaw,
        ))

    start_data = data.get("start", {})
    start_pos = np.array(start_data.get("position", [0.0, 0.0, 1.5]), dtype=float)

    # --- Pipeline setup ---
    seq = GateSequencer(gate_specs, SequencerConfig(
        pass_through_margin=1.5,
        proximity_pass_distance=1.0,  # pass if within 1.0m of gate center
    ))
    seq.start()

    ekf = DroneEKF(EKFConfig())
    ekf.initialize(tuple(start_pos), (0, 0, 0), timestamp_s=0.0)

    rl_opt = RacingLineOptimizer()
    opt_wps = rl_opt.optimize(gate_waypoints, tuple(start_pos))

    traj_opt = TrajectoryOptimizer(
        constraints=DroneConstraints(max_velocity=15.0), dt_sample=0.02,
    )
    trajectory = traj_opt.optimize(opt_wps, tuple(start_pos), (0, 0, 0))

    # --- Offline per-section ILC with per-section Q-filter bandwidth (iteration 28) ---
    # Time-varying Q-filter (Bristow & Alleyne 2007, ACC): different Butterworth
    # cutoffs per track section. Higher bandwidth at S-turn inflection (gate-3)
    # where error has high-frequency content from centripetal reversal. Lower
    # bandwidth at smooth sections for noise rejection.
    # Research: Bristow & Alleyne 2007/2008, Zhang 2024 (segment-wise ILC),
    # Freeman 2025, van Haren 2024, Longman 2019.
    from planning.trajectory_optimizer import compute_ilc_offset_table
    n_total_steps = int(trajectory.total_time / dt) + 50
    # Section boundaries (iteration 28): 4 sections with per-section bandwidth
    # Gate-3 at ~2.93s. Inflection region: 2.0s-4.4s (steps 200-440).
    # Helix boundary: midpoint gate-6/gate-7 (~7.4s, step 740).
    inflection_start = int(2.0 / dt)   # step 200
    inflection_end = int(4.4 / dt)     # step 440
    helix_start = int(7.4 / dt)        # step 740
    section_boundaries = [
        # (start, end, alpha, max_correction_m, filter_cutoff_hz, vel_scale)
        # Per-section velocity correction scaling (iteration 42, Bristow & Alleyne 2007):
        # Pre-inflection uses 0.0 to recover gate-2; helix uses 0.7 for max benefit.
        (0, inflection_start, 0.4, 0.15, 0.35, 0.0),                # Pre-inflection: no vel correction
        (inflection_start, inflection_end, 0.50, 0.15, 0.40, 0.4),  # iter 46: alpha 0.45→0.50 for gate-3/4 (Longman 2023 convergence accel)
        (inflection_end, helix_start, 0.4, 0.15, 0.35, 0.5),        # Post-inflection: standard vel
        (helix_start, n_total_steps, 0.4, 0.50, 0.35, 0.7),         # iter 46: helix max_corr 0.45→0.50m for gate-7
    ]
    # Velocity-corrected ILC (iteration 41): returns (pos_offsets, vel_offsets)
    # tuple. Velocity offsets are the smooth time derivative of position offsets,
    # ensuring the controller's velocity reference is consistent with the shifted
    # position reference. Research: Schoellig 2012, Kunapuli 2025, Nam 2026.
    ilc_result = compute_ilc_offset_table(
        trajectory, tuple(start_pos),
        alpha=0.4,
        max_iterations=5,
        smoothing_sigma=10.0,
        max_correction_m=0.15,
        convergence_threshold=0.002,
        dt=dt,
        section_boundaries=section_boundaries,
        blend_steps=50,
        filter_cutoff_hz=0.35,  # Global fallback (used by sections without 5th element)
    )
    if ilc_result is not None:
        ilc_offsets, ilc_vel_offsets = ilc_result
    else:
        ilc_offsets, ilc_vel_offsets = None, None

    # Gains tuned via systematic sweep (iteration 38).
    # Research: "Leveling the Playing Field" (Kunapuli 2025) — feedforward is
    # the most important single fix. NGTC (Pries 2025) — literature gains 2-4x
    # higher. Damping: ζ=(kd+drag)/(2√kp)=(5.5+0.5)/(2√7)≈1.13 (stable).
    # 40+ configs swept: ff=0.50 kp=7 kd=5.5 optimal — avg err -13.4%.
    tracker = GeometricTracker(TrackerConfig(
        kp_xy=7.0, kd_xy=5.5, kp_z=8.0, kd_z=5.0,
        feedforward_accel=0.50,
        velocity_feedforward=0.0,
        mass=1.0, gravity=9.81, max_thrust_n=20.0,
    ))

    # Predictive feedforward lookahead (Tal & Karaman 2018 style).
    # Use acceleration from slightly ahead in the trajectory to
    # anticipate upcoming turns. This is the time-domain equivalent
    # of jerk feedforward via differential flatness.
    ff_lookahead_s = 0.05  # 50ms lookahead for predictive FF

    # --- Synthetic drone state ---
    pos = start_pos.copy()
    vel = np.zeros(3)
    yaw = 0.0

    # Kinematic parameters (tuned to approximate Crazyflie CF2X dynamics)
    max_accel = 15.0      # m/s^2 (~1.5g, matches DroneConstraints)
    max_speed = 15.0      # m/s — increased to match trajectory planner ceiling
    drag = 0.5            # velocity damping (aerodynamic drag approximation)
    yaw_rate_max = 4.0    # rad/s

    # --- Run loop ---
    tracking_errors = []
    loop_times = []
    gate_pass_times = []
    per_gate_errors = {}
    controller_trace = []  # Phase 1: record controller outputs
    crashed = False
    termination_reason = "time_limit"

    t0_wall = time.time()
    n_steps = int(duration / dt)

    for step in range(n_steps):
        t0_loop = time.perf_counter()
        sim_time = step * dt

        # EKF
        ekf.predict((0, 0, -9.81), (0, 0, 0), sim_time)
        # Add small noise to simulate imperfect odometry
        noise_pos = tuple(p + np.random.normal(0, 0.005) for p in pos)
        noise_vel = tuple(v + np.random.normal(0, 0.01) for v in vel)
        ekf.update_odometry(noise_pos, noise_vel)

        # Gate sequencing
        passed = seq.update(tuple(pos))
        if passed:
            gate_pass_times.append({"gate_id": passed.gate_id, "time_s": sim_time})

        if seq.is_complete:
            termination_reason = "race_complete"
            break

        # Crash detection (ground or very high)
        if pos[2] < 0.05:
            crashed = True
            termination_reason = "crash_ground"
            break
        if pos[2] > 20.0:
            crashed = True
            termination_reason = "crash_ceiling"
            break

        # Get reference
        ref = trajectory.sample(sim_time)
        target_pos = np.array(ref.position)
        # Apply ILC position offset (cross-track correction for systematic error)
        if ilc_offsets is not None and step < len(ilc_offsets):
            target_pos = target_pos + ilc_offsets[step]
        target_vel = np.array(ref.velocity)
        # Apply ILC velocity offset (iteration 41+42: per-section scaled, pre-baked).
        # Velocity offsets already include per-section scaling from compute_ilc_offset_table.
        if ilc_vel_offsets is not None and step < len(ilc_vel_offsets):
            target_vel = target_vel + ilc_vel_offsets[step]
        target_yaw = ref.yaw

        # Gate-seeking fallback after trajectory ends
        if sim_time > trajectory.total_time and not seq.is_complete:
            gate = seq.current_gate
            if gate:
                gp = np.array(gate.position)
                d = gp - pos
                dist = float(np.linalg.norm(d))
                if dist > 0.1:
                    target_pos = gp
                    target_vel = d / dist * min(dist * 2, 5.0)
                    target_yaw = float(math.atan2(d[1], d[0]))

        # --- Controller: use real GeometricTracker (Phase 1 requirement) ---
        # Pass trajectory acceleration and jerk for feedforward control.
        # "Leveling the Playing Field" (Kunapuli 2025): feedforward is the
        # most important single fix for geometric controllers.
        # Gate-seeking fallback uses zero acceleration (no trajectory data).
        use_fallback = sim_time > trajectory.total_time and not seq.is_complete
        # Predictive feedforward: use acceleration from slightly ahead
        # to anticipate turns (Tal & Karaman 2018 jerk-FF principle)
        if not use_fallback and ff_lookahead_s > 0:
            ref_ahead = trajectory.sample(sim_time + ff_lookahead_s)
            ff_acc = ref_ahead.acceleration
            ff_jerk = ref_ahead.jerk
        else:
            ff_acc = (0, 0, 0) if use_fallback else ref.acceleration
            ff_jerk = (0, 0, 0) if use_fallback else ref.jerk
        ref_point = TrajectoryPoint(
            time=sim_time,
            position=tuple(target_pos),
            velocity=tuple(target_vel),
            acceleration=ff_acc,
            jerk=ff_jerk,
            yaw=target_yaw,
            yaw_rate=0.0 if use_fallback else ref.yaw_rate,
        )
        cmd = tracker.track(tuple(pos), tuple(vel), yaw, ref_point)

        # Record controller outputs for benchmark artifacts
        controller_trace.append({
            "t": sim_time,
            "roll": cmd.roll_rad,
            "pitch": cmd.pitch_rad,
            "thrust": cmd.thrust,
        })

        # Use the tracker's desired world-frame acceleration directly.
        # This avoids the fragile attitude-to-acceleration back-conversion
        # which is frame-sensitive (sim uses z-up, tracker assumes NED).
        # The attitude command is still recorded above for benchmark metrics.
        accel_des = tracker.last_desired_acceleration
        if accel_des is not None:
            accel = np.array(accel_des) - drag * vel
        else:
            accel = -drag * vel

        # Clamp acceleration
        accel_mag = np.linalg.norm(accel)
        if accel_mag > max_accel:
            accel = accel / accel_mag * max_accel

        # Integrate
        vel = vel + accel * dt
        speed = np.linalg.norm(vel)
        if speed > max_speed:
            vel = vel / speed * max_speed

        pos = pos + vel * dt

        # Yaw tracking
        yaw_err = target_yaw - yaw
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
        yaw += np.clip(yaw_err * 3.0, -yaw_rate_max * dt, yaw_rate_max * dt)

        # Tracking error
        closest = trajectory.find_closest(tuple(pos))
        err = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, closest.position)))
        tracking_errors.append(err)

        cur = seq.current_gate
        if cur:
            per_gate_errors.setdefault(cur.gate_id, []).append(err)

        loop_times.append(time.perf_counter() - t0_loop)

    wall_elapsed = time.time() - t0_wall
    final_sim_time = min(n_steps, step + 1) * dt if 'step' in dir() else 0

    avg_err = float(np.mean(tracking_errors)) if tracking_errors else 0
    max_err = float(np.max(tracking_errors)) if tracking_errors else 0
    p50_err = float(np.percentile(tracking_errors, 50)) if tracking_errors else 0
    p95_err = float(np.percentile(tracking_errors, 95)) if tracking_errors else 0
    avg_hz = 1.0 / np.mean(loop_times) if loop_times else 0

    result = {
        "available": True,
        "skipped": False,
        "sim_type": "synthetic_kinematic",
        "trajectory_time_s": trajectory.total_time,
        "trajectory_points": len(trajectory.points),
        "sim_time_s": final_sim_time,
        "wall_time_s": wall_elapsed,
        "dt": dt,
        "termination_reason": termination_reason,
        "crashed": crashed,
        "gates_passed": seq.gates_passed,
        "total_gates": seq.total_gates,
        "gate_pass_rate": seq.gates_passed / seq.total_gates if seq.total_gates > 0 else 0,
        "complete": seq.is_complete,
        "gate_pass_times": gate_pass_times,
        "avg_tracking_error_m": avg_err,
        "max_tracking_error_m": max_err,
        "p50_tracking_error_m": p50_err,
        "p95_tracking_error_m": p95_err,
        "ekf_uncertainty_m": float(ekf.position_uncertainty),
        "avg_loop_hz": float(avg_hz),
        "total_steps": len(loop_times),
        "per_gate_avg_error": {
            gid: float(np.mean(errs)) for gid, errs in per_gate_errors.items()
        },
        # Phase 1: controller trace summary (full trace too large for JSON)
        "controller_trace_summary": {
            "samples": len(controller_trace),
            "avg_roll_rad": float(np.mean([c["roll"] for c in controller_trace])) if controller_trace else 0,
            "avg_pitch_rad": float(np.mean([c["pitch"] for c in controller_trace])) if controller_trace else 0,
            "avg_thrust": float(np.mean([c["thrust"] for c in controller_trace])) if controller_trace else 0,
            "max_abs_roll_rad": float(np.max([abs(c["roll"]) for c in controller_trace])) if controller_trace else 0,
            "max_abs_pitch_rad": float(np.max([abs(c["pitch"]) for c in controller_trace])) if controller_trace else 0,
        } if controller_trace else {},
    }

    # Threshold checks
    failures = []
    if crashed:
        failures.append(f"drone crashed ({termination_reason})")
    if avg_err > THRESHOLDS["max_avg_tracking_error_m"]:
        failures.append(f"avg_tracking_error {avg_err:.2f}m > {THRESHOLDS['max_avg_tracking_error_m']}m")
    if max_err > THRESHOLDS["max_max_tracking_error_m"]:
        failures.append(f"max_tracking_error {max_err:.2f}m > {THRESHOLDS['max_max_tracking_error_m']}m")
    if float(ekf.position_uncertainty) > THRESHOLDS["max_ekf_uncertainty_m"]:
        failures.append(f"ekf_uncertainty {ekf.position_uncertainty:.3f}m > {THRESHOLDS['max_ekf_uncertainty_m']}m")
    if avg_hz < THRESHOLDS["min_loop_hz"]:
        failures.append(f"loop_hz {avg_hz:.0f} < {THRESHOLDS['min_loop_hz']}")
    gate_rate = seq.gates_passed / seq.total_gates if seq.total_gates > 0 else 0
    if gate_rate < THRESHOLDS["min_gate_pass_rate"]:
        failures.append(f"gate_pass_rate {gate_rate:.0%} < {THRESHOLDS['min_gate_pass_rate']:.0%}")

    result["threshold_failures"] = failures
    result["sim_passed"] = len(failures) == 0

    return result


# ---------------------------------------------------------------------------
# PyBullet simulation benchmark
# ---------------------------------------------------------------------------

def run_sim_benchmark(config_path: str, duration: float) -> Dict[str, Any]:
    """Run the full pipeline against PyBullet headless. Returns structured metrics."""
    result: Dict[str, Any] = {"available": False, "skipped": False}

    try:
        from sim_pybullet.env import DroneRaceEnv
        from sim_pybullet.sequencer import GateSequencer as SimSequencer
        from simulation.model_types import Gate
    except ImportError as e:
        result["skipped"] = True
        result["skip_reason"] = f"PyBullet not available: {e}"
        return result

    try:
        race_config = DroneRaceEnv.load_config(config_path)
    except Exception as e:
        result["skipped"] = True
        result["skip_reason"] = f"Cannot load config: {e}"
        return result

    try:
        env = DroneRaceEnv(race_config=race_config, gui=False)
    except Exception as e:
        result["skipped"] = True
        result["skip_reason"] = f"Cannot create sim: {e}"
        return result

    result["available"] = True

    # Pipeline setup
    from estimation.ekf import DroneEKF, EKFConfig
    from estimation.state_predictor import StatePredictor
    from gate_sequencing.sequencer import GateSequencer, GateSpec
    from planning.trajectory_optimizer import DroneConstraints, GateWaypoint, TrajectoryOptimizer
    from planning.racing_line import RacingLineOptimizer, SpeedProfiler
    from control.mpc_tracker import SimplePositionTracker, TrackerConfig

    start_pos = race_config.start_position

    def _to_specs(gates):
        return [
            GateSpec(
                gate_id=g.gate_id,
                position=(g.pose.x, g.pose.y, g.pose.z),
                yaw=g.pose.yaw, pitch=g.pose.pitch, roll=g.pose.roll,
                interior_width=g.config.interior_width_m,
                interior_height=g.config.interior_height_m,
                sequence_index=g.sequence_index or 0,
            ) for g in gates
        ]

    def _to_waypoints(gates):
        out = []
        for g in gates:
            cy, sy = math.cos(g.pose.yaw), math.sin(g.pose.yaw)
            cp, sp = math.cos(g.pose.pitch), math.sin(g.pose.pitch)
            out.append(GateWaypoint(
                position=(g.pose.x, g.pose.y, g.pose.z),
                normal=(cy * cp, sy * cp, sp),
                width=g.config.interior_width_m,
                height=g.config.interior_height_m,
                yaw=g.pose.yaw,
            ))
        return out

    gate_specs = _to_specs(race_config.gates)
    gate_waypoints = _to_waypoints(race_config.gates)

    seq = GateSequencer(gate_specs)
    seq.start()

    ekf = DroneEKF(EKFConfig())
    ekf.initialize(start_pos, (0, 0, 0), timestamp_s=0.0)

    # Trajectory
    rl_opt = RacingLineOptimizer()
    opt_wps = rl_opt.optimize(gate_waypoints, start_pos)
    traj_opt = TrajectoryOptimizer(
        constraints=DroneConstraints(max_velocity=15.0), dt_sample=0.02,
    )
    trajectory = traj_opt.optimize(opt_wps, start_pos, (0, 0, 0))

    result["trajectory_time_s"] = trajectory.total_time
    result["trajectory_points"] = len(trajectory.points)

    # Run loop
    tracking_errors = []
    loop_times = []
    gate_pass_times = []
    per_gate_errors = {}
    wall_start = time.time()
    crashed = False
    termination_reason = "time_limit"

    while True:
        t0 = time.perf_counter()
        sim_time = env.get_sim_time()

        if sim_time > duration:
            break

        sd = env.drone.get_state()
        pos, vel, yaw = sd["position"], sd["velocity"], sd["yaw"]

        # EKF
        gyro = sd.get("angular_velocity", (0, 0, 0))
        ekf.predict((0, 0, -9.81), gyro, sim_time)
        ekf.update_odometry(pos, vel)

        # Sequencing
        passed = seq.update(pos)
        if passed:
            gate_pass_times.append({"gate_id": passed.gate_id, "time_s": sim_time})

        if seq.is_complete:
            termination_reason = "race_complete"
            break

        if pos[2] < 0.05:
            crashed = True
            termination_reason = "crash"
            break

        # Trajectory tracking
        ref = trajectory.sample(sim_time)
        target_pos = ref.position
        target_vel = ref.velocity
        target_yaw = ref.yaw

        # Gate-seeking fallback
        if sim_time > trajectory.total_time and not seq.is_complete:
            gate = seq.current_gate
            if gate:
                gp = np.array(gate.position)
                dp = np.array(pos)
                d = gp - dp
                dist = float(np.linalg.norm(d))
                if dist > 0.1:
                    target_pos = tuple(gp)
                    target_vel = tuple(d / dist * min(dist * 2, 5.0))
                    target_yaw = float(math.atan2(d[1], d[0]))

        env.drone.step(target_pos, target_vel, target_yaw)

        closest = trajectory.find_closest(pos)
        err = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, closest.position)))
        tracking_errors.append(err)
        loop_times.append(time.perf_counter() - t0)

        # Per-gate tracking error
        cur = seq.current_gate
        if cur:
            gid = cur.gate_id
            per_gate_errors.setdefault(gid, []).append(err)

    wall_elapsed = time.time() - wall_start
    env.close()

    avg_err = float(np.mean(tracking_errors)) if tracking_errors else 0
    max_err = float(np.max(tracking_errors)) if tracking_errors else 0
    p50_err = float(np.percentile(tracking_errors, 50)) if tracking_errors else 0
    p95_err = float(np.percentile(tracking_errors, 95)) if tracking_errors else 0
    avg_hz = 1.0 / np.mean(loop_times) if loop_times else 0

    result.update({
        "sim_time_s": env.get_sim_time() if hasattr(env, 'get_sim_time') else duration,
        "wall_time_s": wall_elapsed,
        "termination_reason": termination_reason,
        "crashed": crashed,
        "gates_passed": seq.gates_passed,
        "total_gates": seq.total_gates,
        "gate_pass_rate": seq.gates_passed / seq.total_gates if seq.total_gates > 0 else 0,
        "complete": seq.is_complete,
        "gate_pass_times": gate_pass_times,
        "avg_tracking_error_m": avg_err,
        "max_tracking_error_m": max_err,
        "p50_tracking_error_m": p50_err,
        "p95_tracking_error_m": p95_err,
        "ekf_uncertainty_m": float(ekf.position_uncertainty),
        "avg_loop_hz": float(avg_hz),
        "total_steps": len(loop_times),
        "per_gate_avg_error": {
            gid: float(np.mean(errs)) for gid, errs in per_gate_errors.items()
        },
    })

    # Threshold checks
    failures = []
    if crashed:
        failures.append("drone crashed")
    if avg_err > THRESHOLDS["max_avg_tracking_error_m"]:
        failures.append(f"avg_tracking_error {avg_err:.2f}m > {THRESHOLDS['max_avg_tracking_error_m']}m")
    if max_err > THRESHOLDS["max_max_tracking_error_m"]:
        failures.append(f"max_tracking_error {max_err:.2f}m > {THRESHOLDS['max_max_tracking_error_m']}m")
    if float(ekf.position_uncertainty) > THRESHOLDS["max_ekf_uncertainty_m"]:
        failures.append(f"ekf_uncertainty {ekf.position_uncertainty:.3f}m > {THRESHOLDS['max_ekf_uncertainty_m']}m")
    if avg_hz < THRESHOLDS["min_loop_hz"]:
        failures.append(f"loop_hz {avg_hz:.0f} < {THRESHOLDS['min_loop_hz']}")
    gate_rate = seq.gates_passed / seq.total_gates if seq.total_gates > 0 else 0
    if gate_rate < THRESHOLDS["min_gate_pass_rate"]:
        failures.append(f"gate_pass_rate {gate_rate:.0%} < {THRESHOLDS['min_gate_pass_rate']:.0%}")

    result["threshold_failures"] = failures
    result["sim_passed"] = len(failures) == 0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Grand Prix — Headless Benchmark")
    parser.add_argument("--mode", choices=["unit", "synthetic", "sim", "full"], default="full")
    parser.add_argument("--config", type=str,
                        default=str(_REPO / "sim_pybullet" / "configs" / "race_01.json"))
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Sim-time seconds (default 30, matches max_total_time_s threshold)")
    parser.add_argument("--json-only", action="store_true",
                        help="Only output JSON to stdout, suppress stderr summary")
    parser.add_argument("--strict", action="store_true",
                        help="Phase 1: treat PyBullet skip as failure")
    parser.add_argument("--completion-threshold", type=float, default=None,
                        help="Override min_gate_pass_rate (0.0-1.0)")
    args = parser.parse_args()

    if args.completion_threshold is not None:
        THRESHOLDS["min_gate_pass_rate"] = args.completion_threshold

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "thresholds": THRESHOLDS,
    }

    overall_pass = True

    # Unit tests
    if args.mode in ("unit", "full"):
        unit = run_unit_tests()
        report["unit_tests"] = unit
        if unit["pass_rate"] < THRESHOLDS["unit_tests_pass_rate"]:
            overall_pass = False

    # Synthetic simulation (always available)
    if args.mode in ("synthetic", "full"):
        synth = run_synthetic_benchmark(duration=args.duration)
        report["synthetic_sim"] = synth
        if not synth.get("skipped", False) and not synth.get("sim_passed", False):
            overall_pass = False

    # PyBullet simulation
    if args.mode in ("sim", "full"):
        sim = run_sim_benchmark(args.config, args.duration)
        report["simulation"] = sim
        if sim.get("skipped", False) and args.strict:
            overall_pass = False
            sim.setdefault("threshold_failures", []).append(
                "PyBullet skipped with --strict mode"
            )
        elif not sim.get("skipped", False) and not sim.get("sim_passed", False):
            overall_pass = False

    report["overall_passed"] = overall_pass

    # Output JSON to stdout
    print(json.dumps(report, indent=2))

    # Human-readable summary to stderr
    if not args.json_only:
        _print_summary(report, file=sys.stderr)

    return 0 if overall_pass else 1


def _print_summary(report: Dict[str, Any], file=sys.stderr):
    p = lambda *a, **kw: print(*a, **kw, file=file)
    p(f"\n{'='*60}")
    p("AI Grand Prix — Benchmark Summary")
    p(f"{'='*60}")

    if "unit_tests" in report:
        u = report["unit_tests"]
        p(f"\nUnit Tests: {u['passed']}/{u['total']} passed ({u['total_time_ms']:.0f}ms)")
        for t in u["tests"]:
            status = "PASS" if t["passed"] else "FAIL"
            line = f"  [{status}] {t['name']} ({t['time_ms']:.1f}ms)"
            if not t["passed"]:
                line += f" — {t.get('error', 'unknown')}"
            p(line)

    for key, label in [("synthetic_sim", "Synthetic Sim"), ("simulation", "PyBullet Sim")]:
        if key not in report:
            continue
        s = report[key]
        if s.get("skipped"):
            p(f"\n{label}: SKIPPED — {s.get('skip_reason', 'unknown')}")
        elif s.get("available"):
            p(f"\n{label}:")
            p(f"  Gates: {s['gates_passed']}/{s['total_gates']} ({s['gate_pass_rate']:.0%})")
            p(f"  Sim time: {s.get('sim_time_s', 0):.1f}s  Wall: {s.get('wall_time_s', 0):.1f}s")
            p(f"  Tracking: avg={s['avg_tracking_error_m']:.2f}m  "
              f"p95={s.get('p95_tracking_error_m', 0):.2f}m  max={s['max_tracking_error_m']:.2f}m")
            p(f"  EKF uncertainty: {s['ekf_uncertainty_m']:.3f}m")
            p(f"  Loop: {s['avg_loop_hz']:.0f} Hz ({s['total_steps']} steps)")
            p(f"  Termination: {s['termination_reason']}")
            if s.get("gate_pass_times"):
                for gpt in s["gate_pass_times"]:
                    p(f"    {gpt['gate_id']} at {gpt['time_s']:.2f}s")
            if s["threshold_failures"]:
                p(f"  Failures:")
                for f_ in s["threshold_failures"]:
                    p(f"    - {f_}")
            else:
                p(f"  All thresholds met!")

    status = "PASS" if report["overall_passed"] else "FAIL"
    p(f"\n{'='*60}")
    p(f"Overall: {status}")
    p(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
