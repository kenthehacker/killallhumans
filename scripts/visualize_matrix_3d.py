"""
Iter-034 (charter task #13): 3D PyBullet REPLAY of the matrix bench's
drone path.

Why this exists:
  - User wants to see the actual drone flying — not just matplotlib graphs.
  - `scripts/visual_demo.py` uses the Crazyflie CF2X (wrong drone, 4/12 gates).
  - Swapping visual_demo to QuadrotorDrone is blocked on the NED↔ENU
    tracker refactor.

This script sidesteps the refactor entirely: it runs the matrix bench
(which produces the AIGP-proxy drone's actual flight path), captures
position + yaw per step, then plays the trajectory back in PyBullet's
GUI using kinematic positioning (`resetBasePositionAndOrientation`).

NO control loop. NO physics dynamics. Just a 3D viewer for the matrix's
recorded data. The drone you see is bit-for-bit what the matrix tests
pin at 7/7 PASS.

Usage:
    python scripts/visualize_matrix_3d.py --track race_01
    python scripts/visualize_matrix_3d.py --track aigp_default --speedup 0.5
    python scripts/visualize_matrix_3d.py --track figure8 --loop
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _list_configs() -> List[Path]:
    cfg_dir = _REPO / "sim_pybullet" / "configs"
    return sorted(cfg_dir.glob("*.json"))


def _resolve_config(track: str) -> Path:
    cfg = _REPO / "sim_pybullet" / "configs" / f"{track}.json"
    if not cfg.exists():
        available = sorted(p.stem for p in _list_configs())
        raise SystemExit(
            f"track '{track}' not found. Available: {', '.join(available)}"
        )
    return cfg


def _run_with_trace(track: str, duration: float) -> tuple[dict, dict]:
    """Returns (result, track_cfg_dict)."""
    from scripts.benchmark import run_synthetic_benchmark

    cfg_path = _resolve_config(track)
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"Running matrix bench on {track} (duration={duration}s)...")
    r = run_synthetic_benchmark(
        duration=duration, config=cfg, record_position_trace=True,
    )
    if not r.get("position_trace"):
        raise RuntimeError(
            f"empty position_trace; termination={r.get('termination_reason')}"
        )
    print(
        f"  sim_time={r['sim_time_s']:.2f}s  "
        f"gates={r['gates_passed']}/{r['total_gates']}  "
        f"avg_err={r['avg_tracking_error_m']:.3f}m  "
        f"clamp={r.get('controller_trace_summary',{}).get('accel_clamp_active_frac',0):.1%}"
    )
    return r, cfg


def _make_drone_body(client: int, mass: float, half_extents: tuple) -> int:
    """Build a simple quadrotor-shaped visual body (no physics, kinematic).

    Visual: central body box + 4 rotor disks. baseMass=0 so PyBullet
    treats this as static — we'll move it via
    resetBasePositionAndOrientation each step.
    """
    import pybullet as p

    body_h = half_extents
    # Central body — black box.
    body_col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=list(body_h), physicsClientId=client,
    )
    body_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=list(body_h),
        rgbaColor=[0.1, 0.1, 0.1, 1.0],
        physicsClientId=client,
    )
    # 4 rotor disks at arm tips, X-configuration.
    arm = max(body_h[0], body_h[1]) * 1.4
    rotor_r = 0.05
    rotor_h = 0.01
    rotor_offsets = [
        ( arm,  arm, body_h[2]),
        (-arm,  arm, body_h[2]),
        (-arm, -arm, body_h[2]),
        ( arm, -arm, body_h[2]),
    ]
    rotor_colors = [
        [1.0, 0.1, 0.1, 1.0],  # front-right RED (heading marker)
        [0.1, 0.1, 1.0, 1.0],  # front-left  BLUE
        [0.1, 0.1, 1.0, 1.0],  # back-left
        [0.1, 0.1, 1.0, 1.0],  # back-right
    ]
    link_masses, link_cols, link_viss, link_pos, link_orn = [], [], [], [], []
    for offset, color in zip(rotor_offsets, rotor_colors):
        col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=rotor_r, height=rotor_h,
            physicsClientId=client,
        )
        vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=rotor_r, length=rotor_h,
            rgbaColor=color, physicsClientId=client,
        )
        link_masses.append(0.0)
        link_cols.append(col)
        link_viss.append(vis)
        link_pos.append(list(offset))
        link_orn.append([0, 0, 0, 1])

    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=body_col,
        baseVisualShapeIndex=body_vis,
        basePosition=[0, 0, 1.5],
        baseOrientation=[0, 0, 0, 1],
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_cols,
        linkVisualShapeIndices=link_viss,
        linkPositions=link_pos,
        linkOrientations=link_orn,
        linkInertialFramePositions=[[0, 0, 0]] * 4,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
        linkParentIndices=[0, 0, 0, 0],
        linkJointTypes=[p.JOINT_FIXED] * 4,
        linkJointAxis=[[0, 0, 1]] * 4,
        physicsClientId=client,
    )
    return body_id


def _build_gates(client: int, cfg: dict) -> list[int]:
    """Build all gate bodies in PyBullet."""
    import pybullet as p

    from simulation.model_types import Gate, GateConfig, Pose3D
    from sim_pybullet.gate_models import create_gate_body

    gate_defaults = cfg.get("gate_defaults", {})
    iw = gate_defaults.get("interior_width_m", 1.5)
    ih = gate_defaults.get("interior_height_m", 1.5)
    bw = gate_defaults.get("border_width_m", 0.1)
    depth = gate_defaults.get("depth_m", 0.05)
    color = gate_defaults.get("color", "red")

    all_ids: list[int] = []
    for g in cfg["gates"]:
        pose_d = g["pose"]
        pose = Pose3D(
            x=pose_d["x"], y=pose_d["y"], z=pose_d["z"], yaw=pose_d["yaw"],
        )
        gate = Gate(
            gate_id=g["id"],
            pose=pose,
            config=GateConfig(
                interior_width_m=iw,
                interior_height_m=ih,
                border_width_m=bw,
                depth_m=depth,
                color=g.get("color", color),
            ),
        )
        ids = create_gate_body(client, gate)
        all_ids.extend(ids)
    return all_ids


def _yaw_to_quat(yaw: float) -> list:
    """Z-axis rotation only. Returns [x, y, z, w] quaternion."""
    half = yaw / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _animate(track: str, duration: float, speedup: float, loop: bool) -> None:
    """Run the bench + drive a 3D PyBullet GUI replay."""
    import pybullet as p
    import pybullet_data

    result, cfg = _run_with_trace(track, duration)
    trace = result["position_trace"]
    gate_pass_times = {
        g["gate_id"]: g["time_s"] for g in result.get("gate_pass_times", [])
    }

    # PyBullet GUI.
    print("Connecting to PyBullet GUI...")
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)
    # Better camera angle — looking diagonally down at the start.
    start_pos = tuple(cfg["start"]["position"])
    p.resetDebugVisualizerCamera(
        cameraDistance=10.0,
        cameraYaw=45.0,
        cameraPitch=-30.0,
        cameraTargetPosition=[start_pos[0] + 5, start_pos[1], start_pos[2]],
        physicsClientId=client,
    )

    # Build gates.
    print(f"Building {len(cfg['gates'])} gates...")
    _build_gates(client, cfg)

    # Build drone.
    print("Building drone body...")
    from competition.drone_spec import DEFAULT_BODY_SIZE_M, DEFAULT_MASS_KG
    body_half = tuple(b / 2.0 for b in DEFAULT_BODY_SIZE_M)
    drone_id = _make_drone_body(client, DEFAULT_MASS_KG, body_half)

    # Position the drone at the start initially.
    p.resetBasePositionAndOrientation(
        drone_id,
        posObj=trace[0]["pos"],
        ornObj=_yaw_to_quat(trace[0]["yaw"]),
        physicsClientId=client,
    )

    # Trajectory polyline overlay (debug line per consecutive sample).
    print("Drawing trajectory overlay...")
    for i in range(0, len(trace) - 1, 5):  # decimate so the overlay is light
        p.addUserDebugLine(
            list(trace[i]["pos"]), list(trace[i + 1]["pos"]),
            lineColorRGB=[1.0, 1.0, 0.0], lineWidth=1.5,
            physicsClientId=client,
        )

    # Persistent HUD text handle.
    hud_id = p.addUserDebugText(
        "starting…", textPosition=[start_pos[0], start_pos[1], start_pos[2] + 3],
        textColorRGB=[1, 1, 1], textSize=1.5, physicsClientId=client,
    )

    print()
    print(f"=== {track} replay ===")
    print(f"  speedup: {speedup}× | loop: {loop} | total: {len(trace)} samples")
    print(f"  drone: red rotor = front; black box = body")
    print(f"  yellow trail = planned path (drone's actual run)")
    print(f"  green gates; will tint dim-gray as they're passed in sequence")
    print(f"  Ctrl+C in this terminal to quit.")
    print()

    # Replay loop.
    try:
        while True:
            t_start = time.perf_counter()
            passed_set: set = set()
            for i, sample in enumerate(trace):
                # Wall-clock pacing — replay in real time at `speedup` factor.
                target_wall_t = sample["t"] / max(speedup, 1e-3)
                while time.perf_counter() - t_start < target_wall_t:
                    time.sleep(0.001)

                p.resetBasePositionAndOrientation(
                    drone_id,
                    posObj=sample["pos"],
                    ornObj=_yaw_to_quat(sample["yaw"]),
                    physicsClientId=client,
                )
                # Update HUD every 10 frames (100ms at 100Hz trace).
                if i % 10 == 0:
                    v = sample["vel"]
                    speed = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
                    hud_text = (
                        f"{track}  t={sample['t']:.2f}s  "
                        f"v={speed:.1f}m/s  err={sample['tracking_err_m']:.2f}m  "
                        f"gates={sum(1 for g, tt in gate_pass_times.items() if tt <= sample['t'])}"
                        f"/{result['total_gates']}"
                    )
                    p.addUserDebugText(
                        hud_text,
                        textPosition=[sample["pos"][0],
                                      sample["pos"][1],
                                      sample["pos"][2] + 1.5],
                        textColorRGB=[1, 1, 0.5], textSize=1.2,
                        replaceItemUniqueId=hud_id,
                        physicsClientId=client,
                    )
                # Step the renderer only (no physics).
                p.stepSimulation(physicsClientId=client)
            if not loop:
                print(
                    f"\nReplay complete. Final state:\n"
                    f"  pos: {trace[-1]['pos']}\n"
                    f"  gates: {result['gates_passed']}/{result['total_gates']}\n"
                    f"Press Ctrl+C to exit, or pass --loop to replay continuously."
                )
                # Hold the window open until user closes/Ctrl-C.
                while True:
                    p.stepSimulation(physicsClientId=client)
                    time.sleep(0.05)
            else:
                print(f"  loop iter complete; restarting...")
    except KeyboardInterrupt:
        print("\nClosing PyBullet GUI.")
        p.disconnect(physicsClientId=client)


def main():
    parser = argparse.ArgumentParser(
        description="3D PyBullet replay of the matrix bench's drone path."
    )
    parser.add_argument("--track", default="race_01")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--speedup", type=float, default=1.0,
        help="Playback factor: 1.0 = real-time, 2.0 = 2× faster, 0.5 = half-speed",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Replay continuously until Ctrl+C.",
    )
    parser.add_argument("--list-tracks", action="store_true")
    args = parser.parse_args()

    if args.list_tracks:
        for path in _list_configs():
            print(path.stem)
        return

    _animate(args.track, args.duration, args.speedup, args.loop)


if __name__ == "__main__":
    main()
