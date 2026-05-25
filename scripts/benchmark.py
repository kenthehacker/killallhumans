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
import dataclasses
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


def _dataclass_from_overrides(cls, overrides: dict):
    """Construct dataclass config from known override keys only."""
    if not overrides:
        return cls()
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in overrides.items() if k in valid_fields}
    return cls(**filtered)


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

def run_synthetic_benchmark(
    duration: float = 30.0,
    dt: float = 0.01,
    config: Optional[Dict[str, Any]] = None,
    tracker_config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full pipeline with synthetic kinematic simulation.

    Uses the same race_01.json gate layout (by default) but simulates drone
    physics with a simple second-order model: PD controller → acceleration →
    velocity → position. No PyBullet dependency.

    Args:
        duration: max sim time in seconds.
        dt: control loop step.
        config: if provided, used in place of `race_01.json` (lets unit tests
            inject a deliberately-crash-inducing course; see iter-001 A7).

    This exercises: trajectory optimizer, racing line, speed profiler, EKF,
    gate sequencer, and geometric tracker — the full pipeline minus perception.
    """
    import json as _json
    # Deterministic seed for reproducible benchmark results (SimpleFlight 2024,
    # Testing Pipeline 2025: fixed seeds are a competition deployment best practice).
    np.random.seed(42)
    from competition.aigp_geometry import AIGP_VQ1_MAX_RUN_DURATION_S
    from estimation.ekf import DroneEKF, EKFConfig
    from gate_sequencing.sequencer import GateSequencer, GateSpec, SequencerConfig
    from planning.trajectory_optimizer import DroneConstraints, GateWaypoint, TrajectoryOptimizer, TrajectoryPoint
    from planning.racing_line import RacingLineOptimizer, SpeedProfiler
    from control.mpc_tracker import GeometricTracker, TrackerConfig

    # Load gate layout from config (or use the caller-supplied dict).
    if config is not None:
        data = config
    else:
        config_path = _REPO / "sim_pybullet" / "configs" / "race_01.json"
        try:
            with open(config_path) as f:
                data = _json.load(f)
        except FileNotFoundError:
            return {"skipped": True, "skip_reason": f"Config not found: {config_path}"}

    gate_defaults = data.get("gate_defaults", {})
    default_w = gate_defaults.get("interior_width_m", 1.2)
    default_h = gate_defaults.get("interior_height_m", 1.2)

    # Per-track border_width may differ from the AIGP default (0.6 m); race_01
    # uses 0.18 m, etc. The bench used to drop this and let GateSpec's default
    # win, which silently changed the geometric crash zone whenever the
    # default shifted. Propagate it explicitly.
    default_border = gate_defaults.get("border_width_m")

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
        # Per-gate border override > course-level default > GateSpec default.
        per_gate_border = gc.get("border_width_m", default_border)

        gs_kwargs = dict(
            gate_id=gd["id"], position=(x, y, z), yaw=yaw, pitch=pitch,
            interior_width=w, interior_height=h,
            sequence_index=gd.get("sequence_index", 0),
        )
        if per_gate_border is not None:
            gs_kwargs["border_width"] = per_gate_border
        gate_specs.append(GateSpec(**gs_kwargs))
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        gate_waypoints.append(GateWaypoint(
            position=(x, y, z), normal=(cy * cp, sy * cp, sp),
            width=w, height=h, yaw=yaw,
        ))

    start_data = data.get("start", {})
    start_pos = np.array(start_data.get("position", [0.0, 0.0, 1.5]), dtype=float)

    # --- Pipeline setup ---
    # Iter-002 review M7 (4/7 reviews MAJOR): align pass_through_margin
    # with the platform default (1.0). The previous synthetic bench used
    # 1.5, which produced different DQ behaviour from the PyBullet bench
    # for the same trajectory — platform honesty drift. The crash-margin
    # opening still uses the strict bare-opening test, so this only
    # tightens what counts as a credited pass.
    seq = GateSequencer(gate_specs, SequencerConfig(
        pass_through_margin=1.0,
        proximity_pass_distance=1.0,  # pass if within 1.0m of gate center
    ))
    seq.start()

    ekf = DroneEKF(EKFConfig())
    ekf.initialize(tuple(start_pos), (0, 0, 0), timestamp_s=0.0)

    # iter-006 F3 (consensus MAJOR): the 8.0 / per-track 6.0 magic
    # numbers from iter-005 are now replaced with a geometry-derived
    # centripetal-acceleration limit. Per-track explicit overrides via
    # `max_velocity_mps` still take precedence for hand-tuned tracks
    # (race_01 stays at its sweep-tuned value if it sets one); else we
    # auto-derive from gate spacing + bend angle.
    # Iter-009: compute max_velocity BEFORE building the racing-line
    # optimizer so its BO scorer uses the right velocity (F9 fix).
    from planning.auto_velocity import derive_safe_max_velocity
    if "max_velocity_mps" in data:
        max_velocity = float(data["max_velocity_mps"])
    else:
        max_velocity = derive_safe_max_velocity(gate_specs)

    # Iter-009i (F9 fix, 4-agent research swarm consensus 2026-05-24):
    # path-velocity decoupling (Heilmeier 2019, Kapania 2016). The
    # racing-line geometry is now selected at a fixed `select_velocity_mps=15.0`
    # (legacy basin, see RacingLineConfig docstring), while the
    # downstream trajectory generator below executes at the auto-derived
    # `max_velocity`. This is the conceptually-correct decoupling that
    # the iter-009 attempt got wrong: it had been coupling SELECTION and
    # EXECUTION through the same velocity, causing the BO oracle to
    # pick a different optimal basin at lower velocity (aigp_default
    # crash at gate-1).
    # Iter-009l (Opus M5 fix): synthetic and PyBullet bench paths must
    # resolve to the SAME effective `select_velocity_mps` for the same
    # track config — Opus's adversarial review flagged that synthetic
    # explicitly pinning 15.0 (iter-009i) while PyBullet inherited the
    # dataclass default was a drift risk. Now both paths use the
    # dataclass default. Single source of truth in
    # `RacingLineConfig.select_velocity_mps`.
    rl_opt = RacingLineOptimizer()
    opt_wps = rl_opt.optimize(gate_waypoints, tuple(start_pos))

    traj_opt = TrajectoryOptimizer(
        constraints=DroneConstraints(max_velocity=max_velocity), dt_sample=0.02,
    )
    trajectory = traj_opt.optimize(opt_wps, tuple(start_pos), (0, 0, 0))

    # iter-004 Phase 1 (research swarm consensus): validate the planned
    # trajectory by replaying a fresh sequencer against samples BEFORE
    # the kinematic sim runs. If the validator says the plan would DQ /
    # crash, the bench surfaces it via `plan_validation` so we get an
    # early warning (and a metric for iter-005's corridor work).
    from planning.plan_validator import validate_trajectory
    plan_validation = validate_trajectory(trajectory, gate_specs, dt=dt)

    # --- Offline per-section ILC (iter-001 A9 — course-agnostic) ----------
    # Section partition: prefer the course config's explicit
    # `ilc_section_overrides` block (e.g. race_01's hand-tuned 4-section
    # helix schedule); else derive from trajectory curvature via
    # `planning.ilc_sections.derive_section_boundaries`. Global hyper-
    # parameters live in `config/ilc_defaults.json`; the course may patch
    # any of them via `ilc_global_overrides`.
    # Research: Bristow & Alleyne 2007 (ACC) — segment-wise ILC with
    # per-section Q-filter bandwidth. Zhang 2024 — segment-wise ILC
    # prevents cross-contamination. van Haren 2024 — class-specific cutoffs.
    from planning.trajectory_optimizer import compute_ilc_offset_table
    from planning.ilc_sections import derive_section_boundaries, load_ilc_config

    ilc_defaults = load_ilc_config()
    ilc_global = {**ilc_defaults["global"], **data.get("ilc_global_overrides", {})}

    n_total_steps = int(trajectory.total_time / dt) + 50

    section_overrides = data.get("ilc_section_overrides")
    if section_overrides:
        # Iter-009 (Opus F10 MAJOR): support fractional section boundaries
        # so race_01's sweep-tuned schedule survives velocity changes that
        # alter trajectory step count. If `ilc_section_overrides_format` is
        # "fractions" (or any value in the first column is <2.0), the
        # start/end columns are scaled by n_total_steps. Otherwise the
        # legacy absolute-step interpretation applies.
        override_format = data.get("ilc_section_overrides_format", "auto")
        is_fractions = (
            override_format == "fractions"
            or (
                override_format == "auto"
                and section_overrides
                and max(s[0] for s in section_overrides) < 2.0
                and max(s[1] for s in section_overrides) <= 1.0 + 1e-6
            )
        )
        if is_fractions:
            section_boundaries = [
                (int(s[0] * n_total_steps), int(s[1] * n_total_steps)) + tuple(s[2:])
                for s in section_overrides
            ]
        else:
            section_boundaries = [tuple(s) for s in section_overrides]
    else:
        section_boundaries = derive_section_boundaries(
            trajectory, dt, config=ilc_defaults,
        )
    # Velocity-corrected ILC (iteration 41): returns (pos_offsets, vel_offsets)
    # tuple. Velocity offsets are the smooth time derivative of position
    # offsets, ensuring the controller's velocity reference is consistent
    # with the shifted position reference.
    ilc_result = compute_ilc_offset_table(
        trajectory, tuple(start_pos),
        alpha=ilc_global["alpha"],
        max_iterations=ilc_global["max_iterations"],
        smoothing_sigma=ilc_global["smoothing_sigma"],
        max_correction_m=ilc_global["max_correction_m"],
        convergence_threshold=ilc_global["convergence_threshold"],
        dt=dt,
        section_boundaries=section_boundaries,
        blend_steps=ilc_global["blend_steps"],
        filter_cutoff_hz=ilc_global["filter_cutoff_hz"],
        momentum_gamma=ilc_global["momentum_gamma"],
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
    # iter-005 (research-swarm consensus + plan-validator diagnostic):
    # tracker overshoot is the dominant failure mode on tight geometries.
    # Letting callers override gains via `tracker_config_overrides` opens
    # an experimentation seam without touching race_01's tuning.
    tracker_kwargs = dict(
        kp_xy=7.0, kd_xy=5.5, kp_z=8.0, kd_z=5.0,
        feedforward_accel=0.50,
        velocity_feedforward=0.0,
        mass=1.0, gravity=9.81, max_thrust_n=20.0,
    )
    if tracker_config_overrides:
        tracker_kwargs.update(tracker_config_overrides)
    # Course-level overrides via track config (no per-call arg needed for
    # plumbing into matrix experiments / per-track tuning).
    course_tracker_overrides = data.get("tracker_overrides", {})
    if course_tracker_overrides:
        tracker_kwargs.update(course_tracker_overrides)
    tracker = GeometricTracker(TrackerConfig(**tracker_kwargs))

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

        # iter-002 review M6 (5/7 reviews MAJOR): enforce the VQ1 8-minute
        # cap on bench paths too. Without this, the bench could run a
        # trajectory that exceeds 480 s without surfacing a timeout — the
        # competition rule would silently be violated.
        if sim_time > AIGP_VQ1_MAX_RUN_DURATION_S:
            seq.mark_timed_out(
                f"vq1_max_run_duration_exceeded:{sim_time:.1f}s"
            )
            termination_reason = f"timed_out:{seq.timeout_reason}"
            break

        # iter-001 A7 + iter-002 (composer-25 F6/F7): terminal failures
        # come from the sequencer first, then the kinematic-sim envelope.
        # `crashed` and `disqualified` are SEPARATE signals — a DQ is a
        # rule violation, not a physical impact. Bench reports both
        # truthfully so the result-dict consumer can distinguish them.
        if seq.last_crash is not None:
            crashed = True
            termination_reason = f"crash_gate:{seq.last_crash[0]}"
            break
        if seq.is_disqualified:
            # Note: NOT setting crashed=True — DQ is its own terminal
            # signal, surfaced in the result dict's `disqualified` field.
            termination_reason = f"disqualified:{seq.dq_reason}"
            break

        # Crash detection (ground or very high) — kept as the catch-all for
        # cases the sequencer can't see (e.g. drone exits the airspace bounds).
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
        # iter-001 A7: terminal-failure surface for the synthesised honesty
        # contract. `crashed` covers physical impacts; `disqualified` covers
        # rule violations (out-of-order pass, etc). Either makes `sim_passed`
        # False.
        "disqualified": bool(seq.is_disqualified),
        "dq_reason": seq.dq_reason,
        "last_crash_gate": seq.last_crash[0] if seq.last_crash else None,
        # iter-004 Phase 1: pre-flight plan validation — does the planned
        # trajectory itself (under perfect tracking) legally complete?
        # Distinct from `sim_passed` which folds tracking error etc.
        "plan_validation": {
            "ok": plan_validation.ok,
            "reason": plan_validation.reason,
            "gates_passed": plan_validation.gates_passed,
            "crashed": plan_validation.crashed,
            "disqualified": plan_validation.disqualified,
            "dq_reason": plan_validation.dq_reason,
            "last_crash_gate": plan_validation.last_crash_gate,
            "first_failure_time_s": plan_validation.first_failure_time_s,
        },
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
    if seq.is_disqualified:
        failures.append(f"drone disqualified ({seq.dq_reason})")
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
    if final_sim_time > THRESHOLDS["max_total_time_s"]:
        failures.append(f"race_time {final_sim_time:.1f}s > {THRESHOLDS['max_total_time_s']}s")

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
    from competition.aigp_geometry import AIGP_VQ1_MAX_RUN_DURATION_S
    from estimation.ekf import DroneEKF, EKFConfig
    from gate_sequencing.sequencer import GateSequencer, GateSpec, SequencerConfig
    from planning.trajectory_optimizer import (
        DroneConstraints,
        GateWaypoint,
        PlannerConfig,
        TrajectoryOptimizer,
    )
    from planning.racing_line import RacingLineConfig, RacingLineOptimizer

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

    seq_cfg = _dataclass_from_overrides(
        SequencerConfig,
        {"proximity_pass_distance": 1.2, **race_config.sequencer_overrides},
    )
    seq = GateSequencer(gate_specs, config=seq_cfg)
    seq.start()

    ekf = DroneEKF(EKFConfig())
    ekf.initialize(start_pos, (0, 0, 0), timestamp_s=0.0)

    # Trajectory
    racing_line_cfg = _dataclass_from_overrides(
        RacingLineConfig, race_config.racing_line_overrides
    )
    planner_cfg = _dataclass_from_overrides(
        PlannerConfig, race_config.planner_overrides
    )

    # iter-007 (3-way BLOCKER fix to iter-005b's dead code): RaceConfig
    # now actually has the max_velocity_mps field, so this getattr does
    # what iter-005b claimed. Fallback chain matches the synthetic bench:
    #   1. explicit `max_velocity_mps` in track JSON
    #   2. legacy `planner_overrides.plan_max_speed_mps`
    #   3. auto-derive from gate geometry
    from planning.auto_velocity import derive_safe_max_velocity
    explicit_max_v = race_config.max_velocity_mps
    if explicit_max_v is not None:
        pybullet_max_v = float(explicit_max_v)
    elif race_config.planner_overrides.get("plan_max_speed_mps") is not None:
        pybullet_max_v = float(planner_cfg.plan_max_speed_mps)
    else:
        pybullet_max_v = derive_safe_max_velocity(gate_specs)

    rl_opt = RacingLineOptimizer(config=racing_line_cfg)
    opt_wps = rl_opt.optimize(gate_waypoints, start_pos)
    traj_opt = TrajectoryOptimizer(
        constraints=DroneConstraints(max_velocity=pybullet_max_v),
        dt_sample=0.02,
        planner_config=planner_cfg,
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

    # Progress clock: advances only when drone is close to its current
    # reference. Replaces wall-clock sampling so a stalled / bumped drone
    # doesn't have its plan fly away from it.
    progress_t = 0.0
    progress_max_lag_m = 1.5  # hold reference if drone is more than this far away

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

        # Iter-003 M6 mirror: enforce VQ1 8-minute cap on the PyBullet
        # bench too. The synthetic bench already has this check; mirror
        # it here so both platforms honour the competition rule.
        if sim_time > AIGP_VQ1_MAX_RUN_DURATION_S:
            seq.mark_timed_out(
                f"vq1_max_run_duration_exceeded:{sim_time:.1f}s"
            )
            termination_reason = f"timed_out:{seq.timeout_reason}"
            break

        # iter-001 A7 + iter-002 (composer-25 F6/F7): DQ is terminal on
        # the PyBullet path too — but it's NOT a crash. Frame-strut
        # crashes still flow primarily through `env.gate_contact()`
        # (the contact manifold is authoritative); the sequencer's
        # geometric crash classification is a secondary signal.
        if seq.is_disqualified:
            # Not setting crashed=True — surfaced via the result dict's
            # disqualified field.
            termination_reason = f"disqualified:{seq.dq_reason}"
            break

        if pos[2] < 0.05:
            crashed = True
            termination_reason = "crash_ground"
            break

        # Iter-008 F12 (Opus, platform-drift MINOR): synthetic bench has
        # a ceiling check; PyBullet now matches. Without this the same
        # trajectory would terminate at z>20 in the synthetic bench but
        # silently fly out of the airspace in the PyBullet bench.
        if pos[2] > 20.0:
            crashed = True
            termination_reason = "crash_ceiling"
            break

        # Gate-contact crash detection: any contact point against an
        # un-passed gate counts as a crash. Passing through the gate
        # opening triggers the sequencer first (handled above), so a
        # contact remaining here means we've hit a frame strut.
        hit_gate = env.gate_contact()
        if hit_gate is not None:
            seq.mark_collision(hit_gate)
            crashed = True
            termination_reason = f"crash_gate:{hit_gate}"
            break

        # If env didn't report a contact but the geometric sequencer's
        # P1-6 branch flagged a frame strike (e.g. sub-frame proximity not
        # quite touching), trust it.
        if seq.last_crash is not None:
            crashed = True
            termination_reason = f"crash_gate:{seq.last_crash[0]}"
            break

        # Progress-clock advance: only if drone is keeping up with the
        # reference. If we're lagging, hold the reference and let the
        # tracker pull us back to it.
        ref_now = trajectory.sample(progress_t)
        lag = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, ref_now.position)))
        if lag < progress_max_lag_m and progress_t < trajectory.total_time:
            dt_sim = env.race_config.timestep
            progress_t = min(progress_t + dt_sim, trajectory.total_time)

        # Trajectory tracking (sampled by progress clock, not wall clock)
        ref = trajectory.sample(progress_t)
        target_pos = ref.position
        target_vel = ref.velocity
        target_yaw = ref.yaw

        # Gate-seeking fallback (always-armed): if we've drifted off the
        # plan AND there's still an un-passed gate, seek straight at it.
        gate = seq.current_gate
        if gate is not None and lag >= progress_max_lag_m:
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
        # iter-001 A7: same honesty surface as the synthetic bench.
        "disqualified": bool(seq.is_disqualified),
        "dq_reason": seq.dq_reason,
        "last_crash_gate": seq.last_crash[0] if seq.last_crash else None,
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
        failures.append(f"drone crashed ({termination_reason})")
    if seq.is_disqualified:
        failures.append(f"drone disqualified ({seq.dq_reason})")
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
        if sim.get("skipped", False) and (args.strict or args.mode == "sim"):
            overall_pass = False
            sim.setdefault("threshold_failures", []).append(
                "PyBullet skipped"
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
