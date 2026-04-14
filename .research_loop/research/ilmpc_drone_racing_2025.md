# Improving Drone Racing Performance Through Iterative Learning MPC
- **URL**: https://arxiv.org/abs/2508.01103
- **Year**: 2025
- **Venue**: arXiv preprint
- **Authors**: Haocheng Zhao, Niklas Schluter, Lukas Brunke, Angela P. Schoellig (Technical University of Munich)

## Key Contribution

This paper enhances Iterative Learning Model Predictive Control (LMPC) for autonomous drone racing through three key innovations: (1) an adaptive cost function that spatially modulates the balance between time-optimality and centerline tracking, (2) a shifted local safe set construction that prevents the drone from taking unsafe shortcuts through gate corners, and (3) a Cartesian-based arc-length formulation that avoids the singularity issues inherent in Frenet-frame parameterizations used by prior LMPC work.

The central insight is that standard LMPC, when applied naively to drone racing, tends to converge to corner-cutting trajectories that violate gate-passage constraints. By introducing a spatially-varying penalty that increases near gates (via sigmoid weighting) and constructing artificial safe set points on the opposite side of the centerline from shortcut trajectories, the method achieves reliable iterative improvement while maintaining gate passage. The approach demonstrates up to 60.85% lap time improvement starting from slow PID-initialized trajectories, and a meaningful 6.05% improvement even when initialized from an aggressively-tuned MPCC++ controller (the current state-of-the-art for model-based drone racing).

## Technical Approach

### LMPC Framework

The method builds on the standard LMPC formulation where data from previous successful laps is stored and used to construct terminal constraints (safe sets) and terminal cost approximations (cost-to-go). At each iteration j, the controller solves a finite-horizon optimal control problem where the terminal state must lie within a convex safe set SS^j constructed from states visited in previous successful iterations. The terminal cost Q^j(x) approximates the cost-to-go from the terminal state, computed from stored trajectory data.

The safe set is constructed using K-nearest neighbors from the stored trajectory data, and the terminal constraint requires the predicted terminal state to be a convex combination of these K neighbors. This guarantees recursive feasibility: if a feasible solution existed at the previous iteration, one exists at the current iteration.

### Adaptive Cost Function

The stage cost combines two objectives:

- **Time-optimal cost**: l_t(u) = c + ||u||^2_R, where c is a constant penalizing each timestep (encouraging faster completion) and R penalizes control effort.
- **Lateral deviation cost**: l_a(x) penalizing deviation from the centerline, weighted by a spatially-varying function gamma(s) that increases near gates.

The spatial weighting gamma(s) uses sigmoid functions centered on gate arc-length positions, creating higher penalties in the vicinity of gates and lower penalties between gates. This allows the optimizer to find aggressive shortcuts between gates while maintaining accuracy through gates. The full stage cost is h(x,u) = l_t(u) + gamma(s) * l_a(x).

### Arc-Length Parameterization (Cartesian Formulation)

Rather than transforming the drone state into a Frenet frame (which has singularities when the drone deviates significantly from the reference path), the method works in Cartesian coordinates and augments the state vector with an arc-length variable s. The centerline is represented as a cubic Hermite spline interpolating gate center positions, parameterized by arc length.

At each control step, the current arc-length is estimated by solving a local optimization (minimizing distance from the drone position to the nearest point on the spline) using L-BFGS-B. To make this computationally tractable, a k-d tree is pre-built over discretized centerline points, reducing lookup time from ~4.21 ms to ~0.68 ms.

### Modified Local Safe Set

Standard LMPC safe sets can inadvertently encourage corner-cutting because stored states from previous iterations that cut corners become part of the safe set. The paper addresses this by:

1. For each safe set point that is on the "shortcut side" of the centerline (relative to a gate), constructing an artificial mirrored point on the opposite side.
2. Adding a quadratic penalty to these shifted points proportional to their Euclidean distance from the original point.
3. This effectively makes corner-cutting terminal states more expensive, steering the optimizer toward trajectories that pass cleanly through gates.

### Drone Model

The system uses a 9-dimensional state vector (position, velocity, Euler angles) with 4-dimensional control inputs (collective thrust + commanded roll, pitch, yaw angles). The model includes first-order attitude dynamics with identified parameters (alpha_phi=-6.00, alpha_theta=-3.96, beta_phi=6.21, beta_theta=4.08). The model is discretized at the control frequency using a 4th-order Runge-Kutta integrator.

## Results

### Simulation (Split-S Track, 4 Gates)

| Initial Controller | Initial Lap Time | Converged Lap Time | Improvement |
|---|---|---|---|
| PID (0.5 m/s) | 23.55s | 8.42s | 64.25% |
| MPCC++ (mu=0.02) | 11.84s | 6.04s | 48.99% |
| MPCC++ (mu=0.10) | 7.71s | 5.92s | 23.22% |

### Real-World (Figure-Eight Track, 4 Gates, Crazyflie 2.1)

| Initial Controller | Initial Lap Time | Converged Lap Time | Improvement |
|---|---|---|---|
| PID (0.5 m/s) | 17.09s | 6.69s | 60.85% |
| MPCC++ (mu=0.10) | 6.45s | 6.06s | 6.05% |

### Convergence Behavior

- From PID initialization: rapid improvement in first 3-5 iterations, convergence within ~15 iterations.
- From MPCC++ initialization: smaller but consistent improvements over ~10 iterations.
- Different initializations converge to different local optima (the final lap times are not identical), confirming nonconvexity of the problem.

### Computational Performance

- Prediction horizon: N=8 steps at 30 Hz control frequency.
- Safe set size: K=20 nearest neighbors.
- Average solver time: 16.66 +/- 2.28 ms (N=5, K=20), scaling roughly linearly with both N and K.
- Solver: Acados v0.4.1 with SQP (5 SQP iterations, 20 QP iterations max).

### Ablation Study

The ablation on the Split-S track with PID initialization (Table II) is particularly informative:
- Time-optimal cost alone: fastest initial improvement but fails gate passage by iteration 2.
- Lateral deviation only: safe but minimal speed gain.
- Combined adaptive cost + shifted safe set: achieves 8.47s with reliable gate passage across all iterations.

## Relevance to Our System

This paper is directly relevant to our autonomous racing stack in several ways:

**Iterative trajectory improvement**: Our current system uses a pre-computed min-snap trajectory that is fixed during the race. The LMPC framework demonstrates that iteratively refining trajectories based on actual flight data can yield dramatic improvements. Even starting from a reasonable trajectory (analogous to our min-snap output), the 6-23% improvement from MPCC++-initialized runs suggests meaningful gains are available.

**Trajectory caching and reuse**: The LMPC safe set mechanism provides a principled way to store and reuse trajectory data across iterations. This is relevant to our benchmark loop: we could store successful trajectory segments from previous runs and use them to warm-start or constrain future trajectory optimization. The K-nearest-neighbor safe set construction is computationally lightweight and could be integrated without significant overhead.

**Convergence to local optima**: The paper confirms that different initializations converge to different local optima. This validates our approach of investing effort in good initial trajectory generation (min-snap + racing line optimization) since the LMPC refinement will converge near the initial trajectory's basin of attraction.

**Adaptive gate-proximity weighting**: The spatially-varying cost that increases near gates is directly applicable. Our controller could benefit from tighter tracking near gates (where precision matters for scoring) and more aggressive maneuvers between gates (where speed matters).

**Arc-length parameterization**: Our trajectory is parameterized by time, but an arc-length parameterization could improve our trajectory tracker's robustness when the drone deviates from the planned path (the arc-length formulation naturally handles "catching up" without the singularity issues of progress-based tracking).

## Actionable Takeaways

1. **Implement spatially-varying tracking precision**: Increase controller tracking gains (or reduce position error tolerance) near gates and relax them between gates. This can be done in our existing geometric tracker by modulating `TrackerConfig` gains as a function of distance to the next gate.

2. **Add trajectory caching across benchmark iterations**: Store successful trajectory + state data from each benchmark run. Use this to warm-start the trajectory optimizer in subsequent runs, biasing it toward proven-feasible regions of state space.

3. **Explore arc-length parameterization for the tracker**: Convert our time-based trajectory following to an arc-length or progress-based formulation. This would make the tracker more robust to timing deviations -- if the drone falls behind schedule, it naturally re-synchronizes rather than chasing a time-indexed reference that has moved ahead.

4. **Implement iterative trajectory refinement loop**: After each successful benchmark run, use the actual flown trajectory as a new initial guess for the trajectory optimizer. The paper shows this converges within 10-15 iterations even from poor initializations.

5. **Add lateral deviation monitoring near gates**: Implement a gate-proximity-aware error metric that weights tracking error more heavily near gates. This aligns with competition scoring where gate passage is binary (pass/fail).

6. **Consider LMPC terminal constraints for safety**: The safe set concept could be adapted for our MPC tracker to guarantee that predicted trajectories remain within a convex hull of previously-successful states, preventing the controller from entering unrecoverable configurations.

7. **Investigate lower control frequencies with longer horizons**: The paper achieves good results at 30 Hz (vs our 100+ Hz target), suggesting that computational budget could be reallocated from control frequency to prediction horizon length or trajectory optimization.

## Limitations & Caveats

1. **Scale mismatch**: The experiments use Crazyflie 2.1 micro-quadrotors on small indoor tracks (figure-eight, split-S) with motion capture. Our competition involves larger drones on longer outdoor courses. The dynamics, speed regime, and aerodynamic effects differ substantially.

2. **Requires multiple laps to converge**: LMPC needs 10-15 successful laps to converge. In competition, we race a single attempt on a potentially unseen track layout. The iterative refinement is only useful during practice/simulation, not during the actual race.

3. **Motion capture dependency**: The real-world experiments rely on 200 Hz motion capture for state estimation. Our system must use onboard VIO + gate detection, introducing estimation noise and latency that the paper does not address.

4. **Simplified dynamics model**: The 9-state model with first-order attitude dynamics is significantly simpler than full quadrotor dynamics. At higher speeds with aerodynamic drag and rotor effects, model mismatch would likely be larger.

5. **Local optima sensitivity**: The paper itself acknowledges convergence to local optima. Without a good initialization, the method may converge to a suboptimal racing line.

6. **No perception-in-the-loop**: The method assumes perfect state knowledge. Integration with vision-based gate detection and the associated noise/dropout is not addressed.

7. **Computational cost**: At 16.66 ms per solve with N=5, K=20, scaling to longer horizons or larger safe sets could exceed real-time constraints on embedded hardware.

## Key Parameters / Constants

| Parameter | Value | Description |
|---|---|---|
| Control frequency | 30 Hz | LMPC control rate |
| Prediction horizon N | 5-8 | Number of MPC steps |
| Safe set size K | 20 | K-nearest neighbors |
| Solver time (N=5,K=20) | 16.66 +/- 2.28 ms | Average LMPC solve |
| Arc-length lookup (k-d tree) | 0.68 ms | Position-to-arc-length |
| SQP iterations | 5 max | Acados solver setting |
| QP iterations | 20 max | Inner QP solver |
| Attitude params alpha_phi | -6.00 | Roll dynamics |
| Attitude params alpha_theta | -3.96 | Pitch dynamics |
| Input gain beta_phi | 6.21 | Roll input gain |
| Input gain beta_theta | 4.08 | Pitch input gain |
| Discretization | RK4 | Integration method |
| Centerline spline | Cubic Hermite | Interpolation type |
| State dimension | 9 (pos, vel, euler) | Drone state |
| Control dimension | 4 (thrust + angles) | Control inputs |
| Motion capture rate | 200 Hz | Ground truth state |
| Convergence iterations | 10-15 | Laps to converge |
