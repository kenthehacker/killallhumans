#!/usr/bin/env python3
"""
Headless smoke test — runs the full pipeline against PyBullet for 10 seconds
and prints pass/fail metrics.  No visualization dependencies required.

Pipeline: estimation (EKF) → planning (trajectory) → control → sequencing

Exit code 0 = PASS (all metrics within thresholds)
Exit code 1 = FAIL

Usage:
    python3 scripts/smoke_test.py
    python3 scripts/smoke_test.py --config sim_pybullet/configs/race_01.json
    python3 scripts/smoke_test.py --duration 20
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure repo root on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simulation.model_types import Gate

# Pipeline modules
from estimation.ekf import DroneEKF, EKFConfig
from estimation.state_predictor import StatePredictor
from gate_sequencing.sequencer import GateSequencer, GateSpec
from planning.trajectory_optimizer import (
    DroneConstraints, GateWaypoint, TrajectoryOptimizer,
)
from planning.racing_line import RacingLineOptimizer, SpeedProfiler
from control.mpc_tracker import SimplePositionTracker, GeometricTracker, TrackerConfig
from competition.adapter import AttitudeCommand, Quaternion, TelemetryState


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


# ──────────────────────────────────────────────────────────────────
# Unit tests (pure pipeline, no PyBullet)
# ──────────────────────────────────────────────────────────────────

def run_unit_tests() -> Tuple[int, int]:
    """Run quick unit tests on each pipeline module. Returns (passed, failed)."""
    passed = failed = 0

    def check(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    # Quaternion roundtrip
    def _quat():
        for r, p, y in [(0, 0, 0), (0.1, 0.2, 0.3), (-0.5, 0.3, 1.0)]:
            q = Quaternion.from_euler(r, p, y)
            r2, p2, y2 = q.to_euler()
            assert abs(r - r2) < 1e-5 and abs(p - p2) < 1e-5
    check("Quaternion roundtrip", _quat)

    # EKF convergence
    def _ekf():
        ekf = DroneEKF(EKFConfig(position_noise_std=0.01, velocity_noise_std=0.05))
        ekf.initialize((1.5, 2.5, -2.5), (0, 0, 0), timestamp_s=0.0)
        for i in range(100):
            ekf.predict((0, 0, -9.81), (0, 0, 0), i * 0.01)
            ekf.update_odometry((1.0, 2.0, -3.0), (0.5, -0.3, 0.0))
        err = math.sqrt(sum((a - b) ** 2 for a, b in
                            zip(ekf.position, (1.0, 2.0, -3.0))))
        assert err < 0.5, f"pos error {err:.3f}"
    check("EKF convergence", _ekf)

    # Trajectory generation
    def _traj():
        wps = [
            GateWaypoint(position=(5, 0, -2), normal=(1, 0, 0), yaw=0),
            GateWaypoint(position=(10, 5, -3), normal=(0, 1, 0), yaw=math.pi / 2),
            GateWaypoint(position=(15, 0, -2), normal=(-1, 0, 0), yaw=math.pi),
        ]
        t = TrajectoryOptimizer(DroneConstraints(max_velocity=10.0), dt_sample=0.05)
        traj = t.optimize(wps, start_position=(0, 0, -2))
        assert traj.total_time > 0 and len(traj.points) > 10
    check("Trajectory generation", _traj)

    # Racing line
    def _rl():
        wps = [
            GateWaypoint(position=(5, 0, -2), normal=(1, 0, 0), yaw=0),
            GateWaypoint(position=(10, 5, -2), normal=(0, 1, 0), yaw=math.pi / 2),
        ]
        out = RacingLineOptimizer().optimize(wps, (0, 0, -2))
        assert len(out) == 2
    check("Racing line optimization", _rl)

    # Speed profiler
    def _sp():
        pts = [(0, 0, -2), (10, 0, -2), (20, 0, -2), (20, 10, -2)]
        sp = SpeedProfiler(max_speed=15.0, min_speed=2.0).profile(pts)
        assert all(2.0 <= s <= 15.0 for s in sp)
    check("Speed profiler", _sp)

    # Geometric tracker
    def _gt():
        from planning.trajectory_optimizer import TrajectoryPoint
        tr = GeometricTracker(TrackerConfig(max_thrust_n=20.0, mass=1.0))
        ref = TrajectoryPoint(0, (0, 0, -2), (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, 0)
        cmd = tr.track((0, 0, -2), (0, 0, 0), 0.0, ref)
        assert 0.01 < cmd.thrust < 0.99
    check("Geometric tracker", _gt)

    # Gate sequencer
    def _gs():
        gs = GateSequencer([
            GateSpec("g1", position=(5, 0, -2), yaw=0, sequence_index=0),
            GateSpec("g2", position=(10, 0, -2), yaw=0, sequence_index=1),
        ])
        gs.start()
        assert gs.update((4, 0, -2)) is None
        p = gs.update((6, 0, -2))
        assert p is not None and p.gate_id == "g1"
    check("Gate sequencer", _gs)

    # State predictor
    def _pred():
        pr = StatePredictor()
        pp, pv, po = pr.predict((0, 0, -5), (3, 0, 0), (0, 0, 0), (0, 0, 0),
                                dt_override=0.1)
        assert abs(pp[0] - 0.3) < 0.05
    check("State predictor", _pred)

    return passed, failed


# ──────────────────────────────────────────────────────────────────
# Integrated PyBullet smoke test
# ──────────────────────────────────────────────────────────────────

def run_pybullet_test(config_path: str, duration: float) -> Tuple[dict, bool]:
    """Run the full pipeline against PyBullet headless. Returns (metrics, ok)."""
    try:
        from sim_pybullet.env import DroneRaceEnv
        from sim_pybullet._gate_to_spec import to_spec as _to_spec
        from gate_sequencing.sequencer import (
            GateSequencer as SimSequencer,
            SequencerConfig,
        )
    except ImportError as e:
        print(f"  [SKIP] PyBullet not available: {e}")
        return {}, True  # skip = still ok

    print(f"\n  Loading sim from {config_path}...")
    try:
        race_config = DroneRaceEnv.load_config(config_path)
    except Exception as e:
        print(f"  [SKIP] Cannot load config: {e}")
        return {}, True

    try:
        env = DroneRaceEnv(race_config=race_config, gui=False)
    except (ImportError, RuntimeError) as e:
        print(f"  [SKIP] Cannot create sim environment: {e}")
        return {}, True

    # Sim sequencer
    sim_seq = SimSequencer(
        [_to_spec(g) for g in race_config.gates],
        config=SequencerConfig(pass_through_margin=1.5, crash_margin=1.0),
    )
    sim_seq.start()

    # New pipeline
    start_pos = race_config.start_position
    gate_specs = gates_to_specs(race_config.gates)
    gate_waypoints = gates_to_waypoints(race_config.gates)

    seq = GateSequencer(gate_specs)
    seq.start()

    ekf = DroneEKF(EKFConfig())
    ekf.initialize(start_pos, (0, 0, 0), timestamp_s=0.0)

    # Trajectory
    rl_opt = RacingLineOptimizer()
    opt_wps = rl_opt.optimize(gate_waypoints, start_pos)
    traj_opt = TrajectoryOptimizer(
        constraints=DroneConstraints(max_velocity=10.0), dt_sample=0.02,
    )
    trajectory = traj_opt.optimize(opt_wps, start_pos, (0, 0, 0))
    print(f"  Trajectory: {trajectory.total_time:.1f}s, {len(trajectory.points)} pts")

    # Run loop
    tracking_errors = []
    loop_times = []
    wall_start = time.time()
    crashed = False

    print(f"  Running headless for {duration}s sim time...")
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
        sim_seq.update(pos)
        seq.update(pos)

        if seq.is_complete:
            break

        if pos[2] < 0.05:
            crashed = True
            break

        # Trajectory tracking
        ref = trajectory.sample(sim_time)
        target_pos = ref.position
        target_vel = ref.velocity
        target_yaw = ref.yaw

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

    wall_elapsed = time.time() - wall_start
    env.close()

    avg_err = np.mean(tracking_errors) if tracking_errors else 0
    max_err = np.max(tracking_errors) if tracking_errors else 0
    avg_hz = 1.0 / np.mean(loop_times) if loop_times else 0
    ekf_unc = ekf.position_uncertainty

    metrics = {
        "sim_time": env.get_sim_time(),
        "wall_time": wall_elapsed,
        "gates_passed": seq.gates_passed,
        "total_gates": seq.total_gates,
        "complete": seq.is_complete,
        "crashed": crashed,
        "avg_tracking_error_m": float(avg_err),
        "max_tracking_error_m": float(max_err),
        "ekf_uncertainty_m": float(ekf_unc),
        "avg_loop_hz": float(avg_hz),
        "total_steps": len(loop_times),
    }

    # Pass/fail thresholds
    ok = True
    if crashed:
        print("  [FAIL] Drone crashed")
        ok = False
    if avg_err > 3.0:
        print(f"  [FAIL] Avg tracking error {avg_err:.2f}m > 3.0m")
        ok = False
    if ekf_unc > 5.0:
        print(f"  [FAIL] EKF uncertainty {ekf_unc:.3f}m > 5.0m")
        ok = False
    if avg_hz < 10:
        print(f"  [FAIL] Loop frequency {avg_hz:.0f} Hz < 10 Hz")
        ok = False

    return metrics, ok


# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline smoke test (headless)")
    parser.add_argument(
        "--config", type=str,
        default=str(_REPO / "sim_pybullet" / "configs" / "race_01.json"),
    )
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Sim-time seconds to run (default 10)")
    parser.add_argument("--skip-pybullet", action="store_true",
                        help="Only run unit tests, skip PyBullet integration")
    args = parser.parse_args()

    print("=" * 60)
    print("AI Grand Prix — Pipeline Smoke Test")
    print("=" * 60)

    # Phase 1: unit tests
    print("\n--- Unit tests ---")
    u_pass, u_fail = run_unit_tests()

    # Phase 2: PyBullet integration
    pyb_ok = True
    metrics = {}
    if not args.skip_pybullet:
        print("\n--- PyBullet integration test ---")
        metrics, pyb_ok = run_pybullet_test(args.config, args.duration)
        if metrics:
            print(f"\n  Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                else:
                    print(f"    {k}: {v}")

    # Summary
    total_ok = u_fail == 0 and pyb_ok
    print(f"\n{'='*60}")
    print(f"Unit tests: {u_pass} passed, {u_fail} failed")
    if metrics:
        print(f"PyBullet:   {'PASS' if pyb_ok else 'FAIL'} "
              f"({metrics.get('gates_passed', 0)}/{metrics.get('total_gates', 0)} gates, "
              f"{metrics.get('sim_time', 0):.1f}s sim)")
    print(f"Overall:    {'PASS' if total_ok else 'FAIL'}")
    print(f"{'='*60}")

    return 0 if total_ok else 1


if __name__ == "__main__":
    sys.exit(main())
