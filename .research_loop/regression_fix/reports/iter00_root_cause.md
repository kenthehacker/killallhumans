# Iteration 0 — Root Cause Investigation (Claude + Codex cross-validated)

## TL;DR
The drone's failure on `visual_demo.py --config race_01.json` (4/12 gates, alt=0.04m crash) is **NOT** a regression from iter 50. It is a **latent architectural mismatch** that has existed since `visual_demo.py` was first created. All 50 prior iterations validated only on the kinematic sim, because `benchmark.py` skips PyBullet ("gym-pybullet-drones is required").

## Primary Root Cause
- Min-snap polynomial trajectory emits peak velocities of **17.04 m/s** despite `max_velocity=10.0` (velocity limits only constrain boundary conditions, not polynomial mid-segment).
- Crazyflie CF2X max horizontal accel ≈ **6.87 m/s²** (35° tilt, 0.027 kg, 0.265 N gravity).
- `visual_demo.py:365` uses **time-based** `trajectory.sample(sim_time)` → reference races ahead of the drone the moment physics starts; DSLPIDControl D-term saturates at max tilt from t=0.

## Why the drone "flew off the map"
- Trajectory ends at t=14.25s with the drone at x≈56m (never touched gate-1).
- Fallback logic (`visual_demo.py:377-386`) then targets gate-1 at (8,0,1.5) — drone must decelerate from ~15 m/s, reverse, and fly back ~48m.
- Hence gate-1 pass at **t=20.79s** instead of expected ~1.3s, oscillation into gate 2/3/4, and eventual tumble → alt=0.04m crash.

## Fixes (in priority order)
1. **CRITICAL** — Rewrite `visual_demo.py:365-375` to use `trajectory.find_closest(pos)` + short (0.3s) lookahead, with command-speed clamp (max 5 m/s).
2. **IMPORTANT** — Clamp polynomial velocities inside `planning/trajectory_optimizer.py::_generate_trajectory` after evaluation.
3. **RECOMMENDED** — Port ILC offset computation from `benchmark.py:290-330` into `visual_demo.py::__init__`.
4. **REQUIRED** — Add high-frequency CSV telemetry logging to `visual_demo.py` (28 columns at 48 Hz, `logs/visual_demo_{ts}.csv`).
5. **BENCHMARK FIX** — Enable `pybullet` sim path in `benchmark.py` so future iterations validate against real physics.

## Secondary finding
`scripts/benchmark.py` has `simulation.skipped=true` in every entry of `benchmark_history.jsonl`. `overall_passed=true` there reflected unit-test + kinematic sim only — **not** Crazyflie physics. This means iterations 1-50 optimized the wrong target; any regression loop must validate on the real PyBullet sim.

## Telemetry columns (48 Hz)
```
sim_time, step_count,
pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw,
ref_pos_x, ref_pos_y, ref_pos_z, ref_vel_x, ref_vel_y, ref_vel_z,
target_pos_x, target_pos_y, target_pos_z, target_vel_x, target_vel_y, target_vel_z,
tracking_error_m, current_gate_id, gates_passed, target_source, loop_dt_ms
```
