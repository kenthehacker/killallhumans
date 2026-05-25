"""
Iter-033 (charter task #13): visualize the matrix bench's drone path.

Why this script exists:
  - The matrix bench (`scripts/benchmark.py`, `tests/test_benchmark_matrix.py`)
    is the truth tier of the project — it uses the 1 kg / 20 N AIGP-proxy
    drone envelope from `competition/drone_spec.py`, runs at 100 Hz, and
    drives the regression suite (7/7 PASS as of iter-032).
  - `scripts/visual_demo.py` uses a different drone (Crazyflie CF2X,
    27 g / 0.6 N) with a different physics stack (gym-pybullet-drones
    + DSLPIDControl), so it CANNOT visualize the AIGP-proxy behavior
    the matrix tests on.
  - Swapping visual_demo's backend to the AIGP-proxy was blocked by a
    NED↔ENU coordinate frame mismatch (see
    `.loop/synthesis/iter_030_step_reference_frame_blocker.md`).

This script bypasses the frame-conversion rabbit hole entirely by
replaying the matrix bench's recorded drone-position trace via
matplotlib animation. You see the EXACT trajectory the matrix tests
produced — same drone, same physics, same controller.

Usage:
    python scripts/visualize_matrix.py --track race_01 [--duration 30]
                                       [--save FILE.mp4] [--interactive]

Renders a 2×2 grid:
  top-left:   top-down (x,y) — drone + planned trajectory + gates
  top-right:  side (x,z) — drone + trajectory + gates as height markers
  bottom-left: tracking error over time
  bottom-right: speed over time + gate-pass events

Pass `--interactive` to drive the animation interactively (slow). Without
it the animation runs through and exits — useful for `--save` to MP4.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _list_configs() -> List[Path]:
    """Return the matrix's track JSON paths."""
    cfg_dir = _REPO / "sim_pybullet" / "configs"
    return sorted(cfg_dir.glob("*.json"))


def _resolve_config(track: str) -> Path:
    """Resolve a track name (e.g. 'race_01') to its config JSON."""
    cfg = _REPO / "sim_pybullet" / "configs" / f"{track}.json"
    if not cfg.exists():
        available = sorted(p.stem for p in _list_configs())
        raise SystemExit(
            f"track '{track}' not found. Available: {', '.join(available)}"
        )
    return cfg


def _run_with_trace(track: str, duration: float) -> dict:
    """Run the matrix bench with position recording on. Returns the result dict.

    This is exactly what `scripts/benchmark_matrix.py` runs in
    `pytest tests/test_benchmark_matrix.py`, plus the per-step position
    trace turned on so we can visualize.
    """
    from scripts.benchmark import run_synthetic_benchmark

    cfg_path = _resolve_config(track)
    with open(cfg_path) as f:
        cfg = json.load(f)
    result = run_synthetic_benchmark(
        duration=duration, config=cfg, record_position_trace=True,
    )
    if not result.get("position_trace"):
        raise RuntimeError(
            "position_trace empty — the bench didn't run? "
            f"termination_reason={result.get('termination_reason')}"
        )
    return result


def _animate(
    track: str,
    duration: float,
    save_path: Optional[Path],
    interactive: bool,
    speedup: float,
) -> None:
    """Run + render. Uses matplotlib's FuncAnimation."""
    # Lazy import — visualizer doesn't need matplotlib in the test path.
    import matplotlib
    if save_path and not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle

    print(f"Running matrix bench on {track} (duration={duration}s)...")
    result = _run_with_trace(track, duration)
    pos_trace = result["position_trace"]
    gate_pass_times = {g["gate_id"]: g["time_s"] for g in result.get("gate_pass_times", [])}
    print(
        f"  sim_time={result['sim_time_s']:.2f}s  "
        f"gates={result['gates_passed']}/{result['total_gates']}  "
        f"avg_err={result['avg_tracking_error_m']:.3f}m  "
        f"sim_passed={result['sim_passed']}"
    )

    # Pull the planned trajectory by re-building it (cheap; same code as bench).
    # The bench doesn't expose `trajectory.points` directly in result, so we
    # reconstruct via the same planning pipeline.
    cfg_path = _resolve_config(track)
    with open(cfg_path) as f:
        cfg = json.load(f)
    from planning.trajectory_optimizer import (
        DroneConstraints, GateWaypoint, TrajectoryOptimizer,
    )
    from planning.racing_line import RacingLineOptimizer
    waypoints = []
    for g in cfg["gates"]:
        pose = g["pose"]
        normal = (math.cos(pose["yaw"]), math.sin(pose["yaw"]), 0.0)
        waypoints.append(GateWaypoint(
            position=(pose["x"], pose["y"], pose["z"]),
            normal=normal,
            width=cfg["gate_defaults"]["interior_width_m"],
            height=cfg["gate_defaults"]["interior_height_m"],
            yaw=pose["yaw"],
        ))
    start_pos = tuple(cfg["start"]["position"])
    racing_line = RacingLineOptimizer().optimize(waypoints, start_pos)
    traj = TrajectoryOptimizer().optimize(racing_line, start_pos, (0, 0, 0))
    plan_xyz = np.array([p.position for p in traj.points])

    # Drone path from the actual bench run.
    drone_xyz = np.array([p["pos"] for p in pos_trace])
    drone_t = np.array([p["t"] for p in pos_trace])
    drone_err = np.array([p["tracking_err_m"] for p in pos_trace])
    drone_speed = np.array([
        math.sqrt(p["vel"][0]**2 + p["vel"][1]**2 + p["vel"][2]**2)
        for p in pos_trace
    ])

    gates_xyz = np.array([(g["pose"]["x"], g["pose"]["y"], g["pose"]["z"])
                          for g in cfg["gates"]])
    gate_ids = [g["id"] for g in cfg["gates"]]
    gate_yaws = np.array([g["pose"]["yaw"] for g in cfg["gates"]])
    gate_width = cfg["gate_defaults"]["interior_width_m"]
    gate_height = cfg["gate_defaults"]["interior_height_m"]

    # Bounds.
    all_xy = np.concatenate([plan_xyz[:, :2], drone_xyz[:, :2], gates_xyz[:, :2]])
    pad = 2.0
    x_lo, y_lo = all_xy.min(axis=0) - pad
    x_hi, y_hi = all_xy.max(axis=0) + pad
    z_lo = min(plan_xyz[:, 2].min(), drone_xyz[:, 2].min(), gates_xyz[:, 2].min()) - 1.0
    z_hi = max(plan_xyz[:, 2].max(), drone_xyz[:, 2].max(), gates_xyz[:, 2].max()) + 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Matrix bench: {track}  ({result['gates_passed']}/{result['total_gates']} gates · "
        f"{result['sim_time_s']:.2f}s · avg_err {result['avg_tracking_error_m']:.3f}m · "
        f"clamp {result.get('controller_trace_summary',{}).get('accel_clamp_active_frac',0):.1%})",
        fontsize=12,
    )
    ax_topdown, ax_side = axes[0]
    ax_err, ax_speed = axes[1]

    # Top-down.
    ax_topdown.set_title("Top-down (x, y)")
    ax_topdown.set_xlabel("x (m)"); ax_topdown.set_ylabel("y (m)")
    ax_topdown.set_xlim(x_lo, x_hi); ax_topdown.set_ylim(y_lo, y_hi)
    ax_topdown.set_aspect("equal", "box")
    ax_topdown.plot(plan_xyz[:, 0], plan_xyz[:, 1], "k--", alpha=0.4, label="planned")
    # Gates as line segments (oriented by yaw).
    for i, (gx, gy, _) in enumerate(gates_xyz):
        yaw = gate_yaws[i]
        # Gate is perpendicular to yaw; half-width along (cos(yaw+π/2), sin(yaw+π/2)).
        nx = -math.sin(yaw); ny = math.cos(yaw)
        half = gate_width / 2.0
        x0, x1 = gx - half * nx, gx + half * nx
        y0, y1 = gy - half * ny, gy + half * ny
        ax_topdown.plot([x0, x1], [y0, y1], "g-", linewidth=3, alpha=0.7)
        ax_topdown.annotate(gate_ids[i], (gx, gy), fontsize=7, alpha=0.6)

    # Side (x, z).
    ax_side.set_title("Side (x, z)")
    ax_side.set_xlabel("x (m)"); ax_side.set_ylabel("z (m)")
    ax_side.set_xlim(x_lo, x_hi); ax_side.set_ylim(z_lo, z_hi)
    ax_side.plot(plan_xyz[:, 0], plan_xyz[:, 2], "k--", alpha=0.4)
    for i, (gx, gy, gz) in enumerate(gates_xyz):
        ax_side.plot(
            [gx, gx], [gz - gate_height/2, gz + gate_height/2],
            "g-", linewidth=3, alpha=0.7,
        )

    # Tracking error.
    ax_err.set_title("Tracking error (m)")
    ax_err.set_xlabel("t (s)"); ax_err.set_ylabel("‖pos - closest_ref‖")
    ax_err.set_xlim(0, drone_t.max())
    err_ymax = max(0.5, drone_err.max() * 1.1)
    ax_err.set_ylim(0, err_ymax)

    # Speed.
    ax_speed.set_title("Speed (m/s) + gate passes")
    ax_speed.set_xlabel("t (s)"); ax_speed.set_ylabel("‖vel‖")
    ax_speed.set_xlim(0, drone_t.max())
    speed_ymax = max(2.0, drone_speed.max() * 1.1)
    ax_speed.set_ylim(0, speed_ymax)
    for gid, gt in gate_pass_times.items():
        ax_speed.axvline(gt, color="g", linestyle=":", alpha=0.5)

    # Animated artists.
    drone_dot_td, = ax_topdown.plot([], [], "ro", markersize=10)
    drone_trail_td, = ax_topdown.plot([], [], "r-", alpha=0.6, linewidth=1)
    drone_dot_side, = ax_side.plot([], [], "ro", markersize=10)
    drone_trail_side, = ax_side.plot([], [], "r-", alpha=0.6, linewidth=1)
    err_line, = ax_err.plot([], [], "b-")
    speed_line, = ax_speed.plot([], [], "b-")
    time_text = ax_topdown.text(0.02, 0.98, "", transform=ax_topdown.transAxes,
                                verticalalignment="top", fontsize=10)

    # Downsample for fluid playback.
    target_fps = 30
    n_total = len(drone_xyz)
    if duration <= 0:
        duration = drone_t.max()
    n_frames = max(60, min(n_total, int(target_fps * (drone_t.max() / speedup))))
    sample_idx = np.linspace(0, n_total - 1, n_frames).astype(int)

    def init():
        for line in (drone_dot_td, drone_trail_td, drone_dot_side, drone_trail_side,
                     err_line, speed_line):
            line.set_data([], [])
        return (drone_dot_td, drone_trail_td, drone_dot_side, drone_trail_side,
                err_line, speed_line, time_text)

    def update(frame):
        i = sample_idx[frame]
        drone_dot_td.set_data([drone_xyz[i, 0]], [drone_xyz[i, 1]])
        drone_trail_td.set_data(drone_xyz[:i + 1, 0], drone_xyz[:i + 1, 1])
        drone_dot_side.set_data([drone_xyz[i, 0]], [drone_xyz[i, 2]])
        drone_trail_side.set_data(drone_xyz[:i + 1, 0], drone_xyz[:i + 1, 2])
        err_line.set_data(drone_t[:i + 1], drone_err[:i + 1])
        speed_line.set_data(drone_t[:i + 1], drone_speed[:i + 1])
        time_text.set_text(
            f"t={drone_t[i]:.2f}s  pos=({drone_xyz[i,0]:.1f},{drone_xyz[i,1]:.1f},{drone_xyz[i,2]:.1f})  "
            f"err={drone_err[i]:.3f}m  v={drone_speed[i]:.1f}m/s"
        )
        return (drone_dot_td, drone_trail_td, drone_dot_side, drone_trail_side,
                err_line, speed_line, time_text)

    interval_ms = int(1000.0 / target_fps)
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=interval_ms, blit=True, repeat=False,
    )

    if save_path:
        print(f"Saving animation to {save_path}…")
        try:
            anim.save(str(save_path), fps=target_fps, dpi=120)
            print(f"  saved {save_path}")
        except Exception as e:
            print(f"  save failed: {e}")
            print("  (try installing ffmpeg or use --interactive)")
    if interactive:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Visualize the matrix bench's drone path.")
    p.add_argument("--track", default="race_01", help="Track name (race_01, slalom, ...)")
    p.add_argument("--duration", type=float, default=30.0, help="Sim duration (s)")
    p.add_argument("--save", default=None, help="Save animation as MP4 to this path")
    p.add_argument("--interactive", action="store_true",
                   help="Show interactively (default: only save)")
    p.add_argument("--speedup", type=float, default=1.0,
                   help="Playback speedup factor (1.0 = real-time, 2.0 = 2× faster)")
    p.add_argument("--list-tracks", action="store_true",
                   help="List available track names and exit")
    args = p.parse_args()

    if args.list_tracks:
        for path in _list_configs():
            print(path.stem)
        return

    save = Path(args.save) if args.save else None
    # If neither --save nor --interactive, default to interactive (matches
    # what visual_demo.py does when run from a terminal).
    interactive = args.interactive or save is None
    _animate(args.track, args.duration, save, interactive, args.speedup)


if __name__ == "__main__":
    main()
