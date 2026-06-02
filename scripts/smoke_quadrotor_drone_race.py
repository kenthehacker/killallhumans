"""
Iter-029 smoke test: can `QuadrotorDrone + step_reference` fly a
race_01 trajectory headless?

Builds the minimum-viable PyBullet harness: connect, load ground
plane, instantiate QuadrotorDrone at race_01's start position,
generate the race trajectory via the standard planner stack, then
call `step_reference` in a loop and measure gate passes / tracking
error.

This validates the iter-026b/c plumbing BEFORE doing the wholesale
DroneRaceEnv backend swap. If even this minimal harness can't fly
race_01, the iter-026 plan needs revisiting.

Usage:
    /opt/homebrew/Caskroom/miniconda/base/envs/drone/bin/python \
        scripts/smoke_quadrotor_drone_race.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pybullet as p  # noqa: E402

from sim_pybullet.drone import QuadrotorDrone, DroneConfig  # noqa: E402
from gate_sequencing.sequencer import (  # noqa: E402
    GateSequencer, GateSpec, SequencerConfig,
)
from planning.racing_line import RacingLineOptimizer, SpeedProfiler  # noqa: E402
from planning.trajectory_optimizer import (  # noqa: E402
    DroneConstraints, GateWaypoint, TrajectoryOptimizer,
)
from planning.auto_velocity import derive_safe_max_velocity  # noqa: E402


def _build_gates(cfg: dict):
    """Convert race_01.json gates → planner GateWaypoints + sequencer GateSpecs."""
    waypoints, specs = [], []
    for i, g in enumerate(cfg["gates"]):
        pose = g["pose"]
        x, y, z = pose["x"], pose["y"], pose["z"]
        yaw = pose["yaw"]
        normal = (math.cos(yaw), math.sin(yaw), 0.0)
        waypoints.append(GateWaypoint(
            position=(x, y, z),
            normal=normal,
            width=cfg["gate_defaults"]["interior_width_m"],
            height=cfg["gate_defaults"]["interior_height_m"],
            yaw=yaw,
        ))
        specs.append(GateSpec(
            gate_id=g["id"],
            position=(x, y, z),
            yaw=yaw,
            sequence_index=g.get("sequence_index", i),
        ))
    return waypoints, specs


def main():
    with open(_REPO / "sim_pybullet" / "configs" / "race_01.json") as f:
        cfg = json.load(f)

    # Build planner artifacts
    waypoints, specs = _build_gates(cfg)
    start_pos = tuple(cfg["start"]["position"])
    # Iter-030 (composer's #1 finding): use the SAME max_velocity
    # resolution the PyBullet matrix path uses
    # (scripts/benchmark.py:_run_pybullet_bench), not just
    # derive_safe_max_velocity. Race_01.json sets
    # plan_max_speed_mps=4.0; without this honoring the JSON, the
    # smoke planned at 15 m/s (3.75× the matrix path) and the
    # resulting trajectory had peaks the QuadrotorDrone couldn't
    # follow — iter-029's "physics blocker" was actually a
    # mis-configured smoke test.
    if "max_velocity_mps" in cfg:
        max_v = float(cfg["max_velocity_mps"])
    elif "planner" in cfg and "plan_max_speed_mps" in cfg["planner"]:
        max_v = float(cfg["planner"]["plan_max_speed_mps"])
    else:
        max_v = derive_safe_max_velocity(specs)

    rl_opt = RacingLineOptimizer()
    opt_wps = rl_opt.optimize(waypoints, start_pos)

    traj_opt = TrajectoryOptimizer(
        constraints=DroneConstraints(max_velocity=max_v),
        dt_sample=0.01,
    )
    trajectory = traj_opt.optimize(opt_wps, start_pos, (0, 0, 0))

    # Build sequencer
    seq = GateSequencer(specs, SequencerConfig(enforce_in_order=True))
    seq.start()

    # PyBullet setup — headless
    # Iter-030 fix: timestep = control rate so applyExternalForce is
    # called every physics step. PyBullet's applyExternalForce queues
    # a force for the NEXT stepSimulation only; if we ran physics at
    # 240Hz with control at 120Hz, the drone would only get force
    # half the time → effective half-gravity hover failure.
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 120.0, physicsClientId=client)
    # Simple ground plane (no urdf needed for a smoke test)
    p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)

    drone = QuadrotorDrone(
        physics_client=client,
        config=DroneConfig(),
        start_position=start_pos,
        start_yaw=cfg["start"]["yaw"],
    )

    dt = 1.0 / 120.0  # control rate
    progress_t = 0.0
    progress_max_lag = 1.5
    duration = 30.0
    sim_time = 0.0
    crashes = 0
    tracking_errors = []

    while sim_time < duration:
        state = drone.get_state()
        pos = state["position"]

        # Sequencer
        passed = seq.update(pos)
        if seq.is_complete:
            break
        if seq.is_disqualified:
            print(f"DQ: {seq.dq_reason}")
            break
        if seq.last_crash is not None:
            crashes += 1
            print(f"crash: {seq.last_crash}")
            break

        # Progress clock
        ref_now = trajectory.sample(progress_t)
        lag = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, ref_now.position)))
        if lag < progress_max_lag and progress_t < trajectory.total_time:
            progress_t = min(progress_t + dt, trajectory.total_time)

        ref = trajectory.sample(progress_t)
        drone.step_reference(ref)

        # Single physics step at the control rate so the queued force
        # actually integrates (see iter-030 fix at p.setTimeStep above).
        p.stepSimulation(physicsClientId=client)
        sim_time += dt

        # Compute tracking error
        closest = trajectory.find_closest(pos)
        err = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, closest.position)))
        tracking_errors.append(err)

        # Ground/ceiling guard
        if pos[2] < 0.05:
            print(f"crash_ground at t={sim_time:.2f}s, pos={pos}")
            break
        if pos[2] > 20.0:
            print(f"crash_ceiling at t={sim_time:.2f}s, pos={pos}")
            break

    p.disconnect(physicsClientId=client)

    avg_err = float(np.mean(tracking_errors)) if tracking_errors else 0.0
    print(json.dumps({
        "gates_passed": seq.gates_passed,
        "total_gates": seq.total_gates,
        "is_complete": seq.is_complete,
        "is_disqualified": seq.is_disqualified,
        "sim_time_s": sim_time,
        "avg_tracking_error_m": avg_err,
        "max_tracking_error_m": float(max(tracking_errors)) if tracking_errors else 0.0,
        "crashes": crashes,
    }, indent=2))


if __name__ == "__main__":
    main()
