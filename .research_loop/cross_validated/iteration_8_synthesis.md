# Iteration 8 — Research Synthesis: Feedforward Acceleration Bug Fix

## Current Bottleneck
**trajectory_planning** — speed-accuracy Pareto tradeoff. Race time 16.74s (was 14.73s at iter 5) while avg tracking error 0.285m. The system trades speed for accuracy because the controller can only REACT to trajectory deviations, never ANTICIPATE them.

## Critical Discovery: Feedforward Acceleration Is Zeroed Out

In `scripts/benchmark.py` lines 376-384, the benchmark creates a reference TrajectoryPoint for the controller with **acceleration hard-coded to (0,0,0)**:

```python
ref_point = TrajectoryPoint(
    time=sim_time,
    position=tuple(target_pos),
    velocity=tuple(target_vel),
    acceleration=(0, 0, 0),   # BUG: always zero!
    jerk=(0, 0, 0),           # BUG: always zero!
    yaw=target_yaw,
    yaw_rate=0.0,
)
```

Meanwhile, `trajectory.sample(sim_time)` returns a full TrajectoryPoint WITH acceleration computed from min-snap polynomials. The acceleration, jerk, and yaw_rate fields are simply DISCARDED.

The controller (`control/mpc_tracker.py` line 113) computes:
```python
accel_des = kp * ep + kd * ev + feedforward_accel * ref_acc
```

With `feedforward_accel=1.0` and `ref_acc=(0,0,0)`, the feedforward term is ALWAYS ZERO. The controller operates as pure PD despite being designed for feedforward.

## Research Consensus

### Paper 1: "Leveling the Playing Field" (Kunapuli et al., 2025)
- **Key finding**: "Feedforward information is the most important single fix for geometric controllers"
- Feedforward requires up to 4th-order position derivatives from the reference trajectory
- When GC is given proper feedforward, the gap with RL narrows dramatically
- GC achieves ZERO steady-state error with proper feedforward (RL does not)
- For pre-planned trajectory tracking (our use case), GC with feedforward is competitive with RL

### Paper 2: "Accurate Tracking of Aggressive Quadrotor Trajectories" (Tal & Karaman, 2021)
- Tracks trajectories at 12.9 m/s with 2.1g acceleration and 6.6 cm RMS error
- Key: tracking reference jerk and snap through feedforward angular velocity and angular acceleration
- Demonstrates that higher-order feedforward (beyond just acceleration) significantly improves aggressive flight tracking

### Paper 3: "Differential Flatness with Rotor Drag" (Faessler et al., 2018)
- Feedforward terms computed directly from trajectory via differential flatness
- Compared to treating drag as unknown disturbance, feedforward reduces tracking error significantly at high speed
- The simplest form: F_ff = m * (p_ddot_d + g * z_W) for thrust feedforward

### Paper 4: "DATT" (Huang et al., CoRL 2023)
- Feedforward-feedback-adaptive control structure
- 34-36% smaller tracking errors than adaptive MPC baselines
- The feedforward component is the primary contributor to steady-state tracking accuracy

### Paper 5: "TACO" (Sanghvi et al., 2025)
- Trajectory-aware controller optimization: gains should be tuned per-section
- Confirms that feedforward thrust and moment from reference trajectory are the baseline requirement

## Consensus
All 5 papers agree: **feedforward acceleration from the reference trajectory is the foundational requirement for accurate trajectory tracking.** Without it, any PD/PID/GC controller must build position error before generating corrective acceleration — this is exactly what creates the speed-accuracy tradeoff we observe.

## Contradictions
None. The research is unanimous on this point. The only debate is about whether to go beyond acceleration feedforward to include jerk/snap feedforward (Tal & Karaman) or to use MPC instead of GC (various). But acceleration feedforward is the minimum requirement everyone agrees on.

## Actionable Proposal

### Primary Fix: Pass trajectory acceleration to the controller
Change benchmark.py to use `ref.acceleration`, `ref.jerk`, and `ref.yaw_rate` from `trajectory.sample()` instead of zeroing them out. This activates the existing feedforward capability.

### Expected Impact
- **Tracking accuracy**: Should improve significantly (est. 20-40% reduction in avg error) because the controller will anticipate turns and acceleration changes
- **Race time**: Can potentially REDUCE time inflation (or increase time_weight) since the controller can now handle faster trajectories
- **Pareto frontier shift**: Better accuracy AND better speed simultaneously — breaking the tradeoff that has dominated iterations 5-7

### Risk Assessment
- Low risk: this is fixing a bug, not introducing new logic
- The trajectory's acceleration values are already computed by well-tested min-snap polynomials
- If acceleration feedforward causes issues (unlikely), we can scale it down via `feedforward_accel` parameter
- Gate-seeking fallback case (after trajectory ends) still uses zero acceleration — this is correct since we don't have trajectory data there

### Why This Wasn't Caught Before
- Iterations 1-7 focused on trajectory planning (time allocation, inflation, waypoints) and controller gains
- The controller config `feedforward_accel=1.0` gave the impression feedforward was active
- The benchmark's ref_point construction is a few lines away from the trajectory sampling, making the disconnect non-obvious
- The "Leveling the Playing Field" paper was analyzed in iter 7 but the implementation gap wasn't checked against the benchmark code
