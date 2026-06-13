"""Fast closed-loop flight harness — real pipeline vs point-mass NED physics.

The live AI-GP simulator only emits telemetry in Virtual Qualifier mode (a
GUI action), and the committed captures are frozen ``--dry-run`` artifacts.
This harness gives the iteration loop *real* flight-behaviour telemetry to
read every iteration without the GUI sim: it drives the actual
``RacePipeline`` control callback (geometric tracker + gate sequencer +
dynamic replanner) against a textbook quadrotor point-mass model integrated
in sim time, so it runs in well under a second and would surface spinning /
circling / divergence / gate-stall exactly as the analyzer flags them.

It writes the same JSONL(.gz) schema as ``scripts/aigp_vq1_run.py`` so
``scripts/analyze_telemetry.py`` reads its output directly.

Scope: isolates trajectory tracking + sequencing (``use_ekf=False``,
``use_detection=False``) so the flight dynamics — not perception/estimation
plumbing — are what's under test. A later iteration can layer EKF/detection.

Usage::

    python scripts/sim_closed_loop.py [--max-speed 8.0] [--start-yaw 0.0]
                                      [--record captures/sim_closed_loop.jsonl.gz]
                                      [--analyze]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from typing import List, Optional, Tuple

import numpy as np

from competition.adapter import Quaternion, TelemetryState
from gate_sequencing.sequencer import GateSpec
from race_pipeline import PipelineConfig, RacePipeline

# VQ1 first-contact gate map (NED, metres) — same as aigp_vq1_run FakeAdapter.
VQ1_GATES: List[Tuple] = [
    (0, (-23.3, -0.4, -0.03), 2.72),
    (1, (-46.9, -2.5, 5.07), 2.72),
    (2, (-74.6, 1.2, 13.67), 2.72),
    (3, (-111.5, -5.1, 24.57), 2.72),
    (4, (-135.5, -0.8, 25.36), 2.72),
    (5, (-159.2, -4.4, 25.97), 2.72),
]


def _build_gates() -> List[GateSpec]:
    gates = []
    for gid, pos, size in VQ1_GATES:
        # Gates face along the course (travel toward -X) → yaw ≈ π.
        gates.append(GateSpec(
            gate_id=f"G{gid}",
            position=pos,
            sequence_index=gid,
            yaw=math.pi,
            interior_width=size,
            interior_height=size,
        ))
    return gates


def _world_accel(cmd, mass: float, gravity: float, max_thrust_n: float) -> np.ndarray:
    """NED world acceleration from an attitude command (thrust along body -z)."""
    phi, theta, psi = cmd.roll_rad, cmd.pitch_rad, cmd.yaw_rad
    T = cmd.thrust * max_thrust_n
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    bz = (
        cpsi * sth * cphi + spsi * sphi,
        spsi * sth * cphi - cpsi * sphi,
        cth * cphi,
    )
    return np.array([
        -T / mass * bz[0],
        -T / mass * bz[1],
        -T / mass * bz[2] + gravity,
    ])


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def run(max_speed: float = 8.0, start_yaw: float = 0.0,
        max_sim_s: float = 90.0,
        perturb: Optional[Tuple[float, Tuple[float, float, float]]] = None,
        replan_blind_s: float = 0.0,
        ) -> Tuple[list, dict]:
    """Run the closed loop.

    ``perturb`` optionally injects a one-shot velocity impulse
    ``(t_seconds, (dvx, dvy, dvz))`` to simulate a gust/disturbance and
    exercise the off-track + replanner recovery stack.

    ``replan_blind_s`` models audit Blocker 9: the live pipeline rebuilds the
    trajectory synchronously inside the 100 Hz control callback (~1.8 s), so
    no fresh attitude command is sent while it optimizes. When > 0, every tick
    on which a replan fires is followed by that many seconds of "blind" flight
    holding the last command (the drone coasts), advancing sim time. This lets
    us measure whether recovery survives the real-time blind gap rather than
    assuming an instantaneous replan.
    """
    config = PipelineConfig(
        max_speed=max_speed,
        use_ekf=False,
        use_detection=False,
        use_state_predictor=False,
    )
    gates = _build_gates()
    pipeline = RacePipeline.__new__(RacePipeline)
    # Use the real __init__ so every collaborator is wired up.
    RacePipeline.__init__(pipeline, interface=None, config=config)

    start = (0.0, 0.0, 0.0)
    pipeline.configure(gates, start_position=start)
    # Normally set at the top of the async run(); we drive the callback
    # directly, so seed the race-clock anchors the callback expects.
    # The callback falls back to wall-clock when a telemetry stamp is <= 0,
    # so use a real monotonic baseline (a 0.0 baseline would read as ~boot
    # uptime elapsed and trip the 8-min timeout on tick 0).
    pipeline._race_start_time = time.monotonic()
    pipeline._race_start_sim_time_s = None
    # run() also transitions the sequencer out of WAITING; do it here since
    # we drive the control callback directly (else update() no-ops forever).
    pipeline.sequencer.start()

    # Drone dynamics constants (match TrackerConfig defaults / drone_spec).
    mass, gravity, max_thrust_n = 1.0, 9.81, 20.0
    k_yaw = 4.0           # first-order yaw tracking gain (rad/s per rad)
    max_yaw_rate = 4.0    # rad/s saturation (realistic racing-drone slew)

    dt = 1.0 / config.target_hz
    pos = np.array(start, dtype=float)
    vel = np.zeros(3)
    yaw = float(start_yaw)

    telem_log: list = []
    n_steps = int(max_sim_s / dt)
    reason = "max_sim_time"
    perturb_step = int(perturb[0] / dt) if perturb is not None else -1
    blind_steps = int(replan_blind_s / dt)
    entered_recovery = False
    prev_cmd = None
    replan_seen = 0
    blind_ticks = 0
    tick = 0

    def _integrate(cmd):
        nonlocal pos, vel, yaw
        acc = _world_accel(cmd, mass, gravity, max_thrust_n)
        vel = vel + acc * dt
        pos = pos + vel * dt
        yaw = _wrap(yaw + np.clip(k_yaw * _wrap(cmd.yaw_rad - yaw),
                                  -max_yaw_rate, max_yaw_rate) * dt)

    def _record(cmd, t_us, blind):
        ref_pos = ref_vel = ref_yaw = None
        if pipeline.trajectory is not None:
            try:
                pt = pipeline.trajectory.sample(pipeline._ref_progress_time)
                ref_pos = list(pt.position)
                ref_vel = list(pt.velocity)
                ref_yaw = float(pt.yaw)
            except Exception:
                pass
        entry = {
            "t_us": t_us,
            "pos": [float(p) for p in pos],
            "vel": [float(v) for v in vel],
            "yaw": yaw,
            "gates_passed": pipeline.sequencer.gates_passed if pipeline.sequencer else 0,
            "ref_pos": ref_pos, "ref_vel": ref_vel, "ref_yaw": ref_yaw,
            "blind": blind,
        }
        if cmd is not None:
            entry["cmd_roll"] = round(cmd.roll_rad, 4)
            entry["cmd_pitch"] = round(cmd.pitch_rad, 4)
            entry["cmd_yaw"] = round(cmd.yaw_rad, 4)
            entry["cmd_thrust"] = round(cmd.thrust, 4)
        telem_log.append(entry)

    while tick < n_steps:
        if tick == perturb_step:
            vel = vel + np.array(perturb[1], dtype=float)
        # Strictly positive sim timestamp -> callback uses sim-time elapsed.
        t_us = int((tick + 1) * dt * 1e6)
        telem = TelemetryState(
            timestamp_us=t_us,
            position_ned=tuple(pos),
            velocity_ned=tuple(vel),
            orientation=Quaternion.from_euler(0.0, 0.0, yaw),
            angular_velocity=(0.0, 0.0, 0.0),
        )
        cmd = pipeline._control_callback(telem, None)
        _record(cmd, t_us, blind=False)
        tick += 1
        if pipeline.sequencer is not None and \
                pipeline.sequencer.state.name == "RECOVERY":
            entered_recovery = True

        if cmd is None:
            reason = "callback_returned_none"
            break
        if pipeline.sequencer is not None and pipeline.sequencer.is_complete:
            reason = "race_complete"
            break
        if pipeline.sequencer is not None and pipeline.sequencer.last_crash is not None:
            reason = "crash"
            break

        # Blind window (audit Blocker 9): a replan just fired and the live
        # pipeline would now block ~replan_blind_s optimizing, sending no new
        # command. Model that as coasting on the previous command while the
        # sequencer still observes physics (so a crash mid-blind is caught).
        rc = getattr(pipeline, "_replan_count", 0)
        crashed_blind = False
        if blind_steps and rc > replan_seen and prev_cmd is not None:
            for _ in range(blind_steps):
                if tick >= n_steps:
                    break
                _integrate(prev_cmd)
                if pipeline.sequencer is not None:
                    pipeline.sequencer.update(
                        tuple(pos), gate_detected=False, detection_active=False
                    )
                _record(prev_cmd, int((tick + 1) * dt * 1e6), blind=True)
                blind_ticks += 1
                tick += 1
                if pipeline.sequencer is not None and \
                        pipeline.sequencer.last_crash is not None:
                    crashed_blind = True
                    break
        replan_seen = rc
        if crashed_blind:
            reason = "crash"
            break

        _integrate(cmd)
        prev_cmd = cmd

    summary = {
        "termination_reason": reason,
        "sim_time_s": round(len(telem_log) * dt, 2),
        "gates_passed": pipeline.sequencer.gates_passed if pipeline.sequencer else 0,
        "total_gates": pipeline.sequencer.total_gates if pipeline.sequencer else len(gates),
        "final_pos": [round(float(p), 2) for p in pos],
        "replans": getattr(pipeline, "_replan_count", 0),
        "entered_recovery": entered_recovery,
        "blind_ticks": blind_ticks,
    }
    return telem_log, summary


def _write(telem_log: list, path: str) -> None:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as f:
        for row in telem_log:
            f.write(json.dumps(row) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-speed", type=float, default=8.0)
    ap.add_argument("--start-yaw", type=float, default=0.0)
    ap.add_argument("--max-sim-s", type=float, default=90.0)
    ap.add_argument("--record", default="captures/sim_closed_loop.jsonl.gz")
    ap.add_argument("--analyze", action="store_true",
                    help="run scripts/analyze_telemetry.py on the result")
    args = ap.parse_args(argv)

    telem_log, summary = run(args.max_speed, args.start_yaw, args.max_sim_s)
    print("SUMMARY:", json.dumps(summary))
    if args.record:
        import os
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        _write(telem_log, args.record)
        print(f"Wrote {len(telem_log)} samples -> {args.record}")
        if args.analyze:
            from scripts.analyze_telemetry import analyze, print_report
            print_report(analyze(args.record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
