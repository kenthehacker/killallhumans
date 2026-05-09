#!/usr/bin/env python3
"""
Fast coordinate-descent search for optimal helix gate offsets.

Research basis:
- CdBO (Cully 2018): Coordinate descent BO for per-section racing optimization
- BO Racing Line (Heilmeier 2020): Sim oracle for trajectory evaluation
- TACO (Sanghvi 2025): Trajectory-aware controller optimization

Varies gate-7 and gate-8 offsets, evaluates with fast kinematic sim
(no ILC — ILC will be recomputed for the final winner).
"""

import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from planning.trajectory_optimizer import DroneConstraints, GateWaypoint, TrajectoryOptimizer
from planning.racing_line import RacingLineConfig


def load_track():
    config_path = os.path.join(REPO, "sim_pybullet", "configs", "race_01.json")
    with open(config_path) as f:
        data = json.load(f)
    gate_defaults = data.get("gate_defaults", {})
    default_w = gate_defaults.get("interior_width_m", 1.2)
    default_h = gate_defaults.get("interior_height_m", 1.2)
    gate_waypoints = []
    for gd in data.get("gates", []):
        pose = gd.get("pose", {})
        gc = gd.get("config", {})
        x, y, z = pose.get("x", 0), pose.get("y", 0), pose.get("z", 1.5)
        yaw = pose.get("yaw", 0)
        pitch = pose.get("pitch", 0)
        w = gc.get("interior_width_m", default_w)
        h = gc.get("interior_height_m", default_h)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        gate_waypoints.append(GateWaypoint(
            position=(x, y, z), normal=(cy * cp, sy * cp, sp),
            width=w, height=h, yaw=yaw,
        ))
    start_pos = tuple(data.get("start", {}).get("position", [0.0, 0.0, 1.5]))
    return gate_waypoints, start_pos


def apply_offsets(gate_waypoints, offsets):
    """Apply lateral/vertical offsets to get optimized gate positions."""
    n = len(gate_waypoints)
    result = []
    for i, gate in enumerate(gate_waypoints):
        lat_off = offsets[i]
        vert_off = offsets[n + i]
        cy = math.cos(gate.yaw)
        sy = math.sin(gate.yaw)
        right = np.array([-sy, cy, 0])
        up = np.array([0, 0, -1])  # NED
        pos = np.array(gate.position)
        pos = pos + right * lat_off * gate.width * 0.5
        pos = pos + up * vert_off * gate.height * 0.5
        result.append(GateWaypoint(
            position=tuple(pos), normal=gate.normal,
            width=gate.width, height=gate.height, yaw=gate.yaw,
        ))
    return result


def fast_eval(gate_waypoints, start_pos, offsets, dt=0.01):
    """
    Fast trajectory + kinematic sim evaluation (no ILC).
    Returns per-gate errors and race time.
    """
    opt_gates = apply_offsets(gate_waypoints, offsets)
    traj_opt = TrajectoryOptimizer(
        constraints=DroneConstraints(max_velocity=15.0), dt_sample=0.02,
    )
    trajectory = traj_opt.optimize(opt_gates, start_pos, (0, 0, 0))

    # Kinematic sim (matches benchmark physics)
    pos = np.array(start_pos, dtype=float)
    vel = np.zeros(3)
    kp_xy, kd_xy = 6.0, 4.0
    kp_z, kd_z = 8.0, 5.0
    ff_accel = 0.4
    ff_lookahead_s = 0.05

    gate_centers = [np.array(g.position) for g in gate_waypoints]
    n_gates = len(gate_centers)
    tracking_errors = []
    per_gate_errors = {}
    n_steps = int(trajectory.total_time / dt) + 50

    for step in range(n_steps):
        sim_time = step * dt
        if sim_time > trajectory.total_time + 0.5:
            break

        ref = trajectory.sample(sim_time)
        target_pos = np.array(ref.position)
        target_vel = np.array(ref.velocity)

        if ff_lookahead_s > 0 and sim_time + ff_lookahead_s <= trajectory.total_time:
            ff_acc_vec = np.array(trajectory.sample(sim_time + ff_lookahead_s).acceleration)
        else:
            ff_acc_vec = np.array(ref.acceleration)

        pos_err = target_pos - pos
        vel_err = target_vel - vel
        accel_des = np.zeros(3)
        accel_des[0] = kp_xy * pos_err[0] + kd_xy * vel_err[0]
        accel_des[1] = kp_xy * pos_err[1] + kd_xy * vel_err[1]
        accel_des[2] = kp_z * pos_err[2] + kd_z * vel_err[2]
        accel_des += ff_accel * ff_acc_vec

        accel = accel_des - 0.5 * vel  # drag
        accel_mag = np.linalg.norm(accel)
        if accel_mag > 15.0:
            accel = accel / accel_mag * 15.0
        vel = vel + accel * dt
        speed = np.linalg.norm(vel)
        if speed > 15.0:
            vel = vel / speed * 15.0
        pos = pos + vel * dt

        closest = trajectory.find_closest(tuple(pos))
        err = math.sqrt(sum((a - b)**2 for a, b in zip(pos, closest.position)))
        tracking_errors.append(err)

        # Assign to nearest gate
        dists = [float(np.linalg.norm(pos - gc)) for gc in gate_centers]
        nearest = int(np.argmin(dists))
        gate_id = f"gate-{nearest + 1}"
        per_gate_errors.setdefault(gate_id, []).append(err)

    per_gate_avg = {gid: float(np.mean(errs)) for gid, errs in per_gate_errors.items()}
    avg_err = float(np.mean(tracking_errors)) if tracking_errors else 999.0

    return {
        "avg_error": avg_err,
        "race_time": trajectory.total_time,
        "per_gate_avg": per_gate_avg,
        "g7": per_gate_avg.get("gate-7", 999.0),
        "g8": per_gate_avg.get("gate-8", 999.0),
    }


def main():
    gate_waypoints, start_pos = load_track()
    n = len(gate_waypoints)

    cache_path = os.path.join(REPO, "planning", "racing_line_cache.json")
    with open(cache_path) as f:
        cache = json.load(f)
    offsets = np.array(cache["offsets"], dtype=float)

    print(f"Gate count: {n}, Offset count: {len(offsets)}")
    print(f"Gate-7 (idx 6): lat={offsets[6]:.4f}, vert={offsets[n+6]:.4f}")
    print(f"Gate-8 (idx 7): lat={offsets[7]:.4f}, vert={offsets[n+7]:.4f}")
    print(f"Gate-6 (idx 5): lat={offsets[5]:.4f}, vert={offsets[n+5]:.4f}")
    print(f"Gate-9 (idx 8): lat={offsets[8]:.4f}, vert={offsets[n+8]:.4f}")

    # Evaluate baseline
    print("\n--- Baseline (no ILC) ---")
    t0 = time.time()
    base = fast_eval(gate_waypoints, start_pos, offsets)
    t1 = time.time()
    print(f"  avg={base['avg_error']:.4f}m, g7={base['g7']:.4f}m, g8={base['g8']:.4f}m, "
          f"time={base['race_time']:.2f}s  ({t1-t0:.1f}s)")

    # 1D coordinate descent over gate-7 and gate-8 offsets
    # Parameters to search: gate-7 lat, gate-7 vert, gate-8 lat, gate-8 vert
    # Also include gate-6 lat/vert and gate-9 lat/vert as secondary
    param_specs = [
        (6, "gate-7 lat", True),     # primary
        (n+6, "gate-7 vert", True),  # primary
        (7, "gate-8 lat", True),     # primary
        (n+7, "gate-8 vert", True),  # primary
        (5, "gate-6 lat", False),    # secondary
        (n+5, "gate-6 vert", False), # secondary
        (8, "gate-9 lat", False),    # secondary
        (n+8, "gate-9 vert", False), # secondary
    ]

    best = offsets.copy()
    best_score = base['avg_error']
    best_g7 = base['g7']
    best_g8 = base['g8']

    for round_num, step_size in enumerate([0.2, 0.08, 0.03], 1):
        print(f"\n=== ROUND {round_num} (step={step_size}) ===")
        for pi, name, is_primary in param_specs:
            current = best[pi]
            # Try a range of values around current
            deltas = [-2*step_size, -step_size, step_size, 2*step_size]
            for delta in deltas:
                new_val = np.clip(current + delta, -0.6, 0.6)
                if abs(new_val - current) < 0.005:
                    continue
                test = best.copy()
                test[pi] = new_val
                try:
                    result = fast_eval(gate_waypoints, start_pos, test)
                    # Score: weighted combination favoring helix improvement
                    score = result['avg_error']
                    g7 = result['g7']
                    g8 = result['g8']
                    helix = g7 + g8

                    improved = False
                    if score < best_score - 0.0005:
                        improved = True
                    elif score <= best_score + 0.001 and helix < (best_g7 + best_g8) - 0.005:
                        improved = True

                    tag = "BETTER" if improved else "skip"
                    if improved or is_primary:
                        print(f"  [{tag}] {name}: {current:.3f}→{new_val:.3f} | "
                              f"avg={score:.4f} g7={g7:.4f} g8={g8:.4f} t={result['race_time']:.2f}")

                    if improved:
                        best[pi] = new_val
                        best_score = score
                        best_g7 = g7
                        best_g8 = g8
                        current = new_val  # update for next delta
                except Exception as e:
                    print(f"  [ERROR] {name}: {current:.3f}→{new_val:.3f}: {e}")

    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"avg_error: {best_score:.4f}m (was {base['avg_error']:.4f}m)")
    print(f"gate-7: {best_g7:.4f}m (was {base['g7']:.4f}m)")
    print(f"gate-8: {best_g8:.4f}m (was {base['g8']:.4f}m)")
    print(f"\nOptimal offsets:")
    print(f"  gate-6: lat={best[5]:.4f}, vert={best[n+5]:.4f}")
    print(f"  gate-7: lat={best[6]:.4f}, vert={best[n+6]:.4f}")
    print(f"  gate-8: lat={best[7]:.4f}, vert={best[n+7]:.4f}")
    print(f"  gate-9: lat={best[8]:.4f}, vert={best[n+8]:.4f}")

    # Show all per-gate errors for final best
    final = fast_eval(gate_waypoints, start_pos, best)
    print(f"\nPer-gate errors (no ILC):")
    for gid in sorted(final['per_gate_avg'].keys(), key=lambda x: int(x.split('-')[1])):
        baseline_val = base['per_gate_avg'].get(gid, 0)
        final_val = final['per_gate_avg'][gid]
        delta = final_val - baseline_val
        arrow = "↓" if delta < -0.001 else "↑" if delta > 0.001 else "="
        print(f"  {gid}: {final_val:.4f}m ({delta:+.4f} {arrow})")

    # Output the new offsets for cache
    print(f"\nNew offsets JSON:")
    print(json.dumps(best.tolist()))

    # Save to a temp file for easy use
    out_path = os.path.join(REPO, "scripts", "best_helix_offsets.json")
    with open(out_path, "w") as f:
        json.dump({"offsets": best.tolist(), "score": best_score, "g7": best_g7, "g8": best_g8}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
