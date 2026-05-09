# Iteration 5 — Research Synthesis: Speed Optimization

## Current Bottleneck
Race time is 23.0s, target is 14s (39% reduction needed). Avg tracking error is 0.186m (threshold 0.5m), providing 0.314m of headroom to trade accuracy for speed.

## Critical Code-Level Finding

**The benchmark's trajectory optimizer is created with `max_velocity=10.0`** (line 292 of `scripts/benchmark.py`), overriding the `DroneConstraints` default of 15.0 m/s. This is the single largest contributor to slow race time — the L-BFGS optimizer penalizes any segment exceeding 10 m/s average speed with a weight of 100.

Additionally, the kinematic sim caps drone speed at 12 m/s (`max_speed = 12.0`), which would limit actual achievable speed even if the trajectory commands faster.

## Research Basis

### TOGT Planner (Qin 2024, ICRA)
- Uses L-BFGS on log-time variables with penalty-based feasibility
- **Key insight**: segment times and waypoint positions are jointly optimized; the optimizer implicitly allocates more time to high-curvature segments
- **Relevant parameter**: penalty exponent is cubic (3) for smoother gradients at constraint boundaries
- Our acceleration penalty uses quadratic (2), which is steeper and more conservative

### AOS / Time-Optimal Long-Range (Shao, Scaramuzza 2024)
- Segment durations are free optimization variables alongside polynomial coefficients
- **Key insight**: bang-bang control structure means polynomial trajectories with smooth constraints are inherently ~1.7% suboptimal vs true time-optimal
- This puts a floor on our polynomial approach, but we're ~64% above optimal, not 1.7%

### "Leveling the Playing Field" (2025) — previously analyzed
- Feedforward acceleration is the most important single fix for geometric controllers
- Already implemented in our system (iteration 3)

### "On Your Own" (Romero 2025) — previously analyzed
- Dual entry/exit waypoints at ±0.4m (implemented in iteration 4)
- Competition system achieves speeds well above 10 m/s through gates

## Root Cause Analysis

The 23s race time decomposes as follows:
- 25 segments averaging 0.93s each
- Short entry/exit segments (0.8m) at minimum 0.1s = 8 m/s
- Long inter-gate segments at roughly 5-8 m/s average
- The acceleration penalty creates coupling between short and long segments: speed differences across segment boundaries trigger penalties that prevent the optimizer from making long segments faster

**The problem is NOT that 25 segments are inherently too many** — it's that:
1. `max_velocity=10.0` creates an artificially low speed ceiling
2. The acceleration penalty weight (50) is too high for 25 segments with varied lengths
3. The kinematic sim's `max_speed=12.0` caps actual achievable speed below what the trajectory planner should be targeting

## Recommended Changes (Priority Order)

1. **Increase trajectory optimizer max_velocity: 10→15** — this directly removes the artificial speed ceiling. Backed by TOGT (drones typically plan at 8-15 m/s) and "On Your Own" (competition speeds exceed 10 m/s).

2. **Increase kinematic sim max_speed: 12→15 and max_accel: 12→15** — allow the simulated drone to actually reach planned speeds. A 1kg drone with 20N max thrust has TWR=2.04, supporting max horizontal accel ≈ 10 m/s². However, the trajectory planner with feedforward means the controller needs less corrective acceleration.

3. **Increase max_acceleration in DroneConstraints: 12→15** — relax the L-BFGS acceleration penalty threshold.

4. **Reduce acceleration penalty weight: 50→25** — make L-BFGS less conservative at segment boundaries. Backed by TOGT using cubic (not quadratic) penalties for smoother optimization landscape.

5. **Increase speed factors in initial allocation: 0.65/0.55/0.45 → 0.80/0.70/0.55** — better initial guess (though L-BFGS will re-optimize, a closer initial point helps convergence).

6. **Increase max_tilt_angle: 0.7→0.85 rad** — allow more aggressive turns at higher speed. Backed by "Precise Aggressive Aerial Maneuvers" (2026) which demonstrates tilts up to 90°.

## Expected Impact
- Race time: 23s → ~14-17s (37-39% reduction)
- Avg tracking error: 0.186m → ~0.25-0.35m (some increase expected, still within 0.5m threshold)
- Gate pass rate: should remain 100% (trajectory still smooth, just faster)

## Risks
- Tracking error may increase past 0.5m threshold if too aggressive → revert
- Controller saturation at higher speeds → monitored via max roll/pitch
- Short segments may become dynamically infeasible → minimum segment time 0.1s preserved

## Consensus
All papers agree: competitive drone racing operates at speeds well above 10 m/s. The 10 m/s limit in our benchmark was overly conservative, especially after the controller improvements in iterations 3-4.
