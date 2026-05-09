# AI Grand Prix — Claude Code Instructions

## Project Overview
Autonomous drone racing system for the AI Grand Prix competition (Anduril/DCL, $500K prize, VQ1 deadline May 2026).

## Autonomous Iteration Protocol

You are expected to **autonomously iterate** on this codebase to improve race performance. Follow this loop:

### 1. Run Benchmark
```bash
cd ~/Personal/killallhumans
python3 scripts/benchmark.py --mode full 2>/dev/null
```
This outputs JSON to stdout. Parse it to understand current performance.

For unit tests only (fast, no PyBullet):
```bash
python3 scripts/benchmark.py --mode unit 2>/dev/null
```

For simulation only:
```bash
python3 scripts/benchmark.py --mode sim --duration 20 2>/dev/null
```

### 2. Parse Results
The JSON output contains:
- `unit_tests.tests[]` — each test with `passed`, `error`, `time_ms`
- `simulation.gates_passed` / `simulation.total_gates` — gate completion
- `simulation.avg_tracking_error_m` — cross-track error (lower = better)
- `simulation.p95_tracking_error_m` — 95th percentile error
- `simulation.ekf_uncertainty_m` — state estimation quality
- `simulation.avg_loop_hz` — control loop speed
- `simulation.crashed` — whether the drone crashed
- `simulation.per_gate_avg_error` — per-gate tracking quality
- `simulation.threshold_failures[]` — what thresholds were violated
- `overall_passed` — true if all thresholds met

### 3. Identify Issues
From the benchmark results, prioritize:
1. **Crashes** — fix immediately (usually control or trajectory issue)
2. **Failed unit tests** — fix the module in question
3. **Low gate pass rate** — trajectory doesn't reach gates, or sequencer misses them
4. **High tracking error** — controller gains, trajectory quality, or EKF drift
5. **Low loop Hz** — computational bottleneck (usually in trajectory sampling)

### 4. Make Improvements
Edit the relevant module, then re-run the benchmark to verify improvement.

### 5. Repeat
Continue the loop until `overall_passed: true` and metrics improve.

## Quality Targets (current thresholds — tightened to aspirational in iteration 2)
| Metric | Threshold | Next Target |
|--------|-----------|-------------|
| Unit test pass rate | 100% | 100% |
| Avg tracking error | < 0.5m | < 0.25m |
| Max tracking error | < 2.0m | < 1.0m |
| EKF uncertainty | < 0.5m | < 0.1m |
| Gate pass rate | 100% | 100% |
| Loop frequency | > 100 Hz | > 100 Hz |
| No crash | required | required |
| Race time | < 30s | < 14s |

When all thresholds are met, tighten them toward aspirational targets.

## Key Modules (edit these)

### Estimation (`estimation/`)
- `ekf.py` — 15-state Extended Kalman Filter. Tune `EKFConfig` noise params.
- `gate_pnp.py` — PnP gate pose estimation for drift correction.
- `gate_tracker.py` — Kalman filter for temporal gate tracking.
- `state_predictor.py` — Latency compensation (forward-predicts state).

### Planning (`planning/`)
- `trajectory_optimizer.py` — Min-snap polynomial trajectory. Tune `DroneConstraints`, segment time allocation.
- `racing_line.py` — Lateral offset optimization + curvature-aware speed profiling.

### Control (`control/`)
- `mpc_tracker.py` — Geometric tracker (SE(3) Lee et al.) and simple PD tracker. Tune `TrackerConfig` gains.

### Gate Sequencing (`gate_sequencing/`)
- `sequencer.py` — Gate pass-through detection. Tune `pass_through_margin`.

### Pipeline (`race_pipeline.py`)
- Top-level orchestrator. Integrates all modules.

### Competition Interface (`competition/`)
- `adapter.py` — Abstract interface
- `mavlink_bridge.py` — MAVSDK-Python adapter for competition
- `pybullet_adapter.py` — PyBullet adapter for local testing

## Visual Demo (for human review)
```bash
python3 scripts/visual_demo.py --config sim_pybullet/configs/race_01.json
```
Opens a dual-view window (FPV + top-down map). Press Q to quit, R to reset.
**Use `--pybullet-gui` to also open the 3D PyBullet viewer.**

## Architecture
```
Camera → Gate Detector → PnP Pose → State Estimator (EKF)
                                      ↓
Telemetry → State Predictor → Trajectory Tracker → Attitude Command
                               ↑
                Pre-computed Racing Trajectory (min-snap polynomials)
```

## Coordinate Frames
- **Competition/MAVLink**: NED (North-East-Down)
- **PyBullet sim**: ENU (East-North-Up)
- The adapter layer handles conversion. All pipeline modules use NED internally.

## Dependencies
```bash
pip install -r requirements.txt
```
Core: numpy, scipy, mavsdk, opencv-python
Sim: pybullet, gym-pybullet-drones, pyvista, PyQt6

## Research Papers (see MASTERPLAN.md)
- TOGT Planner (Qin 2024) — time-optimal gate-traversing
- "On Your Own" (Romero 2025) — dual-stage EKF + MPC
- Swift (Kaufmann, Nature 2023) — RL champion-level racing
- Perception-Aware Planning (ETH 2025) — FOV constraints
- Drift-Corrected VIO (arXiv 2512.20475) — gate-based drift correction

## Do NOT
- Modify `sim_pybullet/` physics (treat as ground truth)
- Add unnecessary dependencies
- Over-engineer — keep changes minimal and measurable
- Skip running benchmarks after changes
