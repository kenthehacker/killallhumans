# Optimization-Based Iterative Learning for Precise Quadrocopter Trajectory Tracking
- **URL**: https://link.springer.com/article/10.1007/s10514-012-9283-2
- **Authors**: Angela P. Schoellig, Fabian L. Mueller, Raffaello D'Andrea
- **Year**: 2012
- **Venue**: Autonomous Robots, Vol. 33, No. 1-2, pp. 103-127

---

## Key Contribution

This seminal paper introduces an optimization-based Iterative Learning Control (ILC) framework for quadrotor trajectory tracking that achieves sub-centimeter precision through iterative feedforward input correction. The core insight is that systematic tracking errors (those that repeat across executions of the same trajectory) can be eliminated by learning a feedforward correction signal from previous trial data. The method combines Kalman filtering for disturbance estimation with convex optimization for constrained input updates, converging to high-precision tracking within 3-5 iterations on real hardware.

The key distinction from standard feedback control: feedback controllers react to errors AFTER they occur, while ILC proactively corrects for ANTICIPATED errors based on experience. This is particularly powerful for repetitive tasks like drone racing, where the same trajectory is executed repeatedly (in simulation) and the systematic error pattern is consistent.

---

## Technical Approach

### System Model and Error Decomposition

The quadrotor dynamics are modeled as a linear time-varying system around a nominal trajectory:

```
x_{k+1} = A_k * x_k + B_k * u_k + d_k
```

where k is the time index within a trial, x is the state deviation from nominal, u is the input deviation (feedforward correction), and d_k is the "learned disturbance" — the systematic component of model mismatch and recurring external forces.

The critical insight is the decomposition of tracking error into:
- **Systematic error**: repeats across trials (model mismatch, unmodeled dynamics, recurring disturbances)
- **Random error**: varies between trials (sensor noise, turbulence)

ILC targets ONLY the systematic component. The Kalman filter separates the two.

### Two-Step Update Algorithm

After each trial j, the feedforward input for the next trial is updated via:

**Step 1: Kalman Filter (Disturbance Estimation)**
Using the measured tracking error from trial j, a Kalman smoother estimates the disturbance sequence d_k^j. The filter uses the model A_k, B_k and the known input u_k^j to compute the innovation — the difference between observed and predicted behavior. This innovation is attributed to d_k.

The Kalman filter naturally handles the systematic-vs-random decomposition: systematic disturbances are estimated with high confidence (low covariance), while random noise contributes negligible signal to the disturbance estimate.

**Step 2: Convex Optimization (Input Update)**
Given the updated disturbance estimate d_k^{j+1}, solve:

```
minimize ||x_N||^2 + sum_{k=0}^{N-1} (||x_k||^2_Q + ||u_k||^2_R)
subject to:
  x_{k+1} = A_k * x_k + B_k * u_k + d_k^{j+1}
  u_min <= u_k <= u_max    (input constraints)
  x_min <= x_k <= x_max    (state constraints — safety tube)
```

This is a standard constrained LQR problem, solvable in polynomial time via quadratic programming. The solution gives the feedforward input correction u_k^{j+1} for the next trial.

### Simplified Update Rule (for our implementation)

For unconstrained systems (no active input/state constraints), the update simplifies to:

```
u_{k}^{j+1} = u_{k}^{j} + L * e_{k}^{j}
```

where L is the learning gain matrix and e_k^j is the tracking error at time k in trial j. This is the classic P-type ILC update. With the Kalman filter providing filtered errors, this becomes:

```
u_{k}^{j+1} = u_{k}^{j} + L * d_hat_{k}^{j}
```

For position trajectory correction (our application), the "input" u is a position offset added to the reference trajectory. The update becomes:

```
ref_pos_{k}^{j+1} = ref_pos_{k}^{j} + alpha * (actual_pos_{k}^{j} - ref_pos_{k}^{j})
```

Wait — this is WRONG. The correction should be in the opposite direction: if the drone is consistently to the LEFT of the reference, shift the reference to the RIGHT:

```
ref_pos_{k}^{j+1} = ref_pos_{k}^{j} - alpha * (actual_pos_{k}^{j} - ref_pos_{k}^{j})
```

Or equivalently, if error = ref - actual (positive when drone lags behind):

```
ref_pos_{k}^{j+1} = ref_pos_{k}^{j} + alpha * error_{k}^{j}
```

The learning rate alpha should be 0.3-0.7. Too high causes oscillation; too low is slow to converge.

### Convergence Properties

The paper proves that the learning algorithm converges monotonically under mild conditions:
- The model A_k, B_k must be "approximately correct" (bounded model error)
- The learning gain L must satisfy ||I - B*L|| < 1 (contraction mapping condition)
- The Kalman filter must correctly separate systematic from random errors

In practice, convergence to near-optimal tracking is achieved in 3-5 iterations for moderate maneuvers and 5-10 iterations for aggressive ones.

---

## Results

### Flying Machine Arena Experiments

The paper demonstrated ILC on the ETH Flying Machine Arena quadrotors executing figure-eight and lemniscate trajectories:

| Metric | Before ILC | After ILC (5 iters) | Improvement |
|--------|-----------|---------------------|-------------|
| Max position error | 0.15m | 0.02m | 87% |
| RMS position error | 0.06m | 0.008m | 87% |
| Max velocity error | 0.8 m/s | 0.15 m/s | 81% |

The improvement is dramatic: tracking error reduced by an order of magnitude in just 5 iterations.

### Convergence Speed

- Iteration 1: ~50% error reduction
- Iteration 2: ~75% error reduction
- Iteration 3: ~85% error reduction
- Iterations 4-5: diminishing returns, approaching noise floor

### Constraint Handling

When input constraints are active (the drone is at its dynamic limits), the ILC correctly identifies that further feedforward correction is not possible at those instants and focuses improvement on the unconstrained portions of the trajectory. This is crucial for racing — the helix section may be at the PD controller's limits, and ILC will focus improvement on the portions where headroom exists.

---

## Relevance to Our System

**Extremely high relevance.** Our system has the exact problem ILC was designed to solve:

1. **Repeatable trajectory**: We run the same pre-computed trajectory in the kinematic sim every time. The tracking error pattern is systematic — gates 7-10 always have the highest error.

2. **Systematic error dominance**: The kinematic sim is deterministic (only tiny noise added to EKF updates). >99% of tracking error is systematic, making ILC particularly effective.

3. **No model needed**: The simplest ILC variant (P-type position correction) doesn't need a dynamics model — just the error from the previous run.

4. **Fast convergence in sim**: Since our sim runs in 0.18s wall time, we can run 5-10 ILC iterations in under 2 seconds. This is negligible compared to the trajectory optimization time.

5. **Helix error floor**: The persistent 0.24-0.33m error at gates 7-10 is exactly the kind of systematic error ILC eliminates. The PD controller consistently lags in the same way at the same locations.

6. **No architecture change needed**: ILC is a POST-PROCESSING step applied to the existing trajectory. It doesn't require changing the trajectory optimizer, racing line optimizer, or controller.

---

## Actionable Takeaways

1. **Implement offline ILC as a post-processing step after trajectory generation.** Run kinematic sim → record error → correct trajectory → repeat 3-5 times. The corrected trajectory becomes the final trajectory used by the benchmark.

2. **Use P-type position correction**: `ref_pos_{k}^{j+1} = ref_pos_{k}^{j} + alpha * (ref_pos_{k}^{j} - actual_pos_{k}^{j})`. Start with alpha=0.5 (aggressive but stable for deterministic sim).

3. **Apply correction to the POSITION component only.** Velocity and acceleration should be recomputed from the corrected positions via finite differences or polynomial re-fitting. This maintains trajectory smoothness.

4. **Smooth the correction signal.** Raw tracking error has high-frequency components. Apply a low-pass filter (Gaussian or moving average) to the error before adding it to the trajectory. Kernel width ~50-100ms.

5. **Limit correction magnitude.** Cap the correction at ~0.3m to prevent the corrected trajectory from diverging too far from the original (maintaining gate passage safety).

6. **Stop iterating when improvement < threshold.** If iteration j+1 improves avg error by <1% over iteration j, stop.

7. **Re-derive velocity/acceleration after correction.** The corrected positions define a new path. Use finite differences with the same time parameterization to get v and a, OR re-fit the corrected positions with min-snap polynomials.

---

## Limitations & Caveats

1. **ILC only works for SYSTEMATIC errors.** In the real competition with wind, turbulence, and perception noise, the systematic component is smaller. However, for sim-based optimization, nearly all error is systematic.

2. **Corrected trajectory may become dynamically infeasible.** If the correction pushes positions too far, the required accelerations may exceed drone limits. The magnitude cap (takeaway #5) mitigates this.

3. **Position-only correction ignores timing.** The drone may need to arrive at the corrected positions at different times. A more advanced version would adjust both position and time.

4. **The PD controller in kinematic sim is NOT the same as a real attitude controller.** ILC corrections optimized for the kinematic sim's PD controller may not transfer to a real drone. However, they improve the kinematic sim benchmark, which is our immediate goal.

5. **Gate passage may shift.** If the correction shifts the trajectory away from gate centers, gate-passing detection could fail. The magnitude cap and gate-proximity constraints (don't correct near gates) handle this.

---

## Key Parameters / Constants

| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| Learning rate alpha | 0.3-0.5 | P-type ILC gain |
| Max iterations | 5 | Convergence typically in 3-5 |
| Correction smoothing | Gaussian, sigma=5 steps | Low-pass filter on error signal |
| Max correction | 0.3m | Safety limit on position shift |
| Convergence threshold | 1% avg error improvement | Stop criterion |
| Gate exclusion zone | None needed in sim | (Real: don't correct within 0.5m of gate) |

*Analysis written 2026-04-14. Paper: Schoellig et al., Autonomous Robots, 2012.*
