# Quasi-Periodic Gaussian Process Predictive Iterative Learning Control

- **URL**: https://arxiv.org/abs/2602.18014
- **Year**: 2026

---

## Key Contribution

Classical ILC applies the P-type update `u_{i+1} = u_i + L_i * e_i`, which converges when the disturbance repeats identically across iterations. In reality, disturbances drift (wind shifts, actuator wear, thermal effects), causing standard ILC to stall at a nonzero residual or diverge. This paper makes three contributions:

1. **Theoretical insight**: ILC generates a quasi-periodic dynamical system. The error sequence across iterations is not i.i.d. but exhibits structured inter-iteration correlation that can be modeled and exploited.

2. **Predictive ILC via QPGP**: A Gaussian Process with a quasi-periodic kernel predicts the *next* iteration's error before it occurs. The controller applies a feedforward correction for the predicted error, not just the observed one. This yields a two-term update: `u_{i+1} = u_i + L_i * e_i + K_i * e_hat_{i+1}`, where `e_hat_{i+1}` is the GP-predicted next-iteration error.

3. **Computational tractability**: Standard GP-based ILC scales O(i^2 * p^3) where i is the iteration count and p is the number of timesteps per iteration, because the GP conditions on all past data. By exploiting the quasi-periodic (Markovian in iteration space) structure, the QPGP reduces complexity to O(p^3) for element-wise prediction and O(p) for block prediction, independent of how many iterations have been run. This is critical for real-time or long-running iterative tasks.

## Technical Approach

### Quasi-Periodic Error Model

The paper models the per-dimension error as a first-order autoregressive process across iterations:

```
e_{i+1,j} = omega_j * e_{i,j} + epsilon_{i+1,j}
```

where `omega_j in (-1, 1)` is the inter-iteration correlation parameter for output dimension j, and `epsilon_{i+1,j} ~ N(0, K_j)` is a zero-mean Gaussian with within-iteration covariance `K_j`. The key structural assumption is that `(I_np - G_i * L_i)` is approximately diagonal, which holds for weakly-coupled systems or systems with feedback diagonalization. This decoupling enables independent per-dimension modeling.

### Two Prediction Strategies

**Block prediction** (O(p) per iteration): Simply predicts `e_hat_{i+1,j} = omega_j * e_{i,j}`. This is a scaled copy of the previous error -- fast but ignores the within-iteration temporal structure. It cannot incorporate partial observations from the current iteration.

**Element-wise prediction** (O(p^3) per iteration): Sequentially predicts each timestep t within iteration i+1 using the GP conditional:

```
e_hat_{i+1,j}(t) = omega_j * e_{i,j}(t) + K_{j;1,t-1} * K_{j;t-1}^{-1} * (e_{i+1,j}^{t-1} - omega_j * e_{i,j}^{t-1})
```

This is a Bayesian update: it starts from the block prediction but refines it using within-iteration observations already collected at timesteps 1..t-1. The Cholesky factorization of K_j is computed once and reused, keeping the cost at O(p^3) total rather than per-step.

### Parameter Estimation

The parameters `(omega_j, K_j)` are estimated from accumulated error data via a two-stage procedure:

- **Stage I**: Alternating minimization on a reduced marginal likelihood to estimate `omega_j` and an unconstrained covariance `K_tilde_j`.
- **Stage II**: Project `K_tilde_j` onto valid periodic positive-definite kernels via diagonal averaging and spectral truncation.

Parameters are re-estimated after each iteration using all accumulated data, enabling online adaptation to slowly drifting dynamics.

### Convergence Theory (Theorem 1)

For element-wise prediction, the error evolves as:

```
e_{i+1}^(t) = B_i * e_i^(t) + (z_i^(t) - z_{i+1}^(t))
B_i = I - G_i^(t) * L_i^(t) - G_i^(t) * K_i^(t) * M_i^(t)
```

If `||B_i||_2 <= gamma_e < 1` (contraction) and `||K_j||_2 < C/2` (bounded covariance), then:
- Mean error converges to zero: `lim_{i->inf} E(e_i) = 0`
- Covariance remains bounded: `lim_{i->inf} ||Cov(e_i)||_2 < C`

The contraction condition constrains the joint design of L_i and K_i. The predictive gain K_i provides an additional degree of freedom beyond L_i alone, making it easier to satisfy the contraction condition while still making large corrections.

## Results

### Vehicle Trajectory Tracking

- Standard ILC: 0.23s for 100 iterations (but slowest convergence)
- QPGP-PILC block: 18.74s (81x standard, but far faster convergence)
- QPGP-PILC element: 19.31s (best convergence, comparable cost to block)
- Full GP-PILC: 649.23s (2800x standard, similar accuracy to element-wise QPGP)
- Sparse GP-PILC (M=100): 19.67s (similar cost to QPGP, slightly worse accuracy)

At iteration 50, QPGP-PILC trajectories visually align with the reference while standard ILC retains visible deviations. The vehicle model includes heading drift (d_theta = 0.04) and a slowly growing bias (b_1 = 0.01 * k), making standard ILC's stationarity assumption fail.

### 3-Link Manipulator

With iteration-dependent sinusoidal biases and Gaussian noise on each joint, QPGP-PILC achieves the fastest error reduction. When a mid-iteration disturbance is injected, standard ILC requires multiple iterations to recover; QPGP-PILC recovers significantly faster because the predictive component anticipates the nominal error pattern, limiting the disturbance's propagation into future iterations.

Constant gains used: L = 0.25, K = 0.3.

### Hello Robot Stretch 3 (Hardware)

On a cable-driven arm with inherent backlash tracking a Lissajous curve at ~10 Hz, QPGP-PILC outperforms standard ILC in convergence rate. Gains: L = 0.05, K = 1.5. The high K/L ratio (30:1) is notable -- it indicates the predictive correction dominates over the reactive correction, which makes sense on hardware where the disturbance pattern (backlash hysteresis) is highly repeatable.

## Relevance to Our System

Our drone racing system already uses an offline ILC-style iteration loop (benchmark -> identify errors -> adjust -> repeat). Several aspects of QPGP-PILC are directly relevant:

1. **Per-gate error prediction**: Our `per_gate_avg_error` metric tracks gate-by-gate tracking quality. The quasi-periodic model could predict which gates will have high error on the next iteration based on patterns from previous runs. The inter-iteration correlation `omega` would capture how persistent gate-specific errors are (e.g., a consistently problematic S-turn gate).

2. **Gain scheduling for ILC**: The paper's 1/i annealing schedule for L and K is directly applicable to our tuning loop. Early iterations should make aggressive corrections (large gains); later iterations should stabilize. Our current approach of manual tuning iteration-by-iteration could be replaced with a principled annealing schedule.

3. **Drift handling**: Our system operates in simulation where dynamics are deterministic, but when we move to hardware (VQ1 competition), actuator drift and wind variation will cause exactly the kind of iteration-to-iteration disturbance drift this paper addresses. The QPGP framework provides a principled way to maintain convergence under such drift.

4. **Feedforward trajectory correction**: The predictive term `K_i * e_hat_{i+1}` is essentially a feedforward correction to the trajectory. We could implement a similar mechanism in our `trajectory_optimizer.py` or `racing_line.py`: after each benchmark run, predict where tracking error will occur on the next run and pre-distort the reference trajectory to compensate.

5. **Computational feasibility**: At O(p^3) for element-wise prediction, QPGP-PILC is viable even in real-time. For our offline iteration loop (where we have seconds to minutes between runs), the computational cost is negligible.

## Actionable Takeaways

1. **Implement iteration-aware trajectory adjustment**: After each benchmark run, record per-timestep tracking error. Fit a simple AR(1) model (`omega * e_prev + noise`) to predict next-run errors. Apply the predicted error as a feedforward offset to the reference trajectory.

2. **Use the K/L ratio insight**: The hardware results show K/L = 30:1, meaning the predictive correction should be much larger than the reactive correction when disturbance patterns are repeatable. In our sim (highly repeatable dynamics), the predictive component should dominate.

3. **Adopt 1/i gain annealing**: Replace ad-hoc gain tuning iteration counts with a 1/i schedule. This provides theoretical convergence guarantees while still making large early corrections.

4. **Per-dimension decoupling**: Model lateral error, altitude error, and speed error independently (as the paper does per output dimension). Each dimension may have different `omega` values -- lateral error from wind gusts may be less correlated across iterations than altitude error from motor degradation.

5. **Element-wise prediction for mid-lap correction**: The element-wise predictor can update its prediction of upcoming gates using errors already observed at earlier gates within the same lap. This is directly applicable to online trajectory replanning during a race.

## Limitations & Caveats

- **Diagonal approximation**: The method assumes `(I - G * L)` is approximately diagonal, meaning outputs are weakly coupled. Quadrotor dynamics have significant cross-coupling (roll-yaw, pitch-speed), which could degrade the per-dimension decomposition. Feedback linearization or INDI could help diagonalize the system first.

- **Constant or slowly varying Jacobian**: The convergence proof requires the iteration operator `B_i` to remain contractive. In aggressive drone racing with highly nonlinear dynamics, the Jacobian `G_i` varies significantly with operating point, potentially violating the contraction condition at high speeds or aggressive maneuvers.

- **Gaussian noise assumption**: Real drone racing errors are often non-Gaussian (e.g., gate detection failures, discrete mode switches, wind gusts with heavy tails). The GP posterior may underestimate uncertainty in these cases.

- **Assumes repeated trajectory**: ILC fundamentally requires running the same trajectory repeatedly. In race conditions, we run the same track but may want to change the racing line between iterations, which breaks the periodicity assumption. The quasi-periodic model handles slow drift but not deliberate trajectory changes.

- **Finite-sample estimation**: With few iterations (our typical workflow is 5-40 iterations), estimating `omega` and `K` accurately is challenging. The paper acknowledges this and recommends maintaining a "comfortable" contraction margin -- but this limits how aggressive the gains can be.

- **10 Hz hardware experiments**: The hardware validation was at ~10 Hz, far below our target 100+ Hz control loop. The computational overhead of element-wise prediction at high rates needs verification, though for our offline iteration loop this is not a concern.

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| omega_j | (-1, 1) | Inter-iteration correlation; closer to 1 means errors persist across iterations |
| gamma (contraction) | [0, 1) | Must be strictly below 1 for convergence; lower = faster convergence |
| L (vehicle) | 1/i schedule | Annealed learning gain |
| K (vehicle) | 1/i schedule | Annealed predictive gain |
| L (manipulator) | 0.25 | Constant reactive gain |
| K (manipulator) | 0.3 | Constant predictive gain (K/L = 1.2) |
| L (Stretch hardware) | 0.05 | Conservative reactive gain for hardware |
| K (Stretch hardware) | 1.5 | Aggressive predictive gain (K/L = 30) |
| Vehicle speed | 8 m/s | Constant longitudinal velocity |
| Heading drift | 0.04 rad/iteration | Simulated iteration-dependent disturbance |
| Bias growth | 0.01/iteration | Slowly growing systematic error |
| Process noise sigma^2 | 0.015 | Gaussian noise variance |
| Stretch control freq | ~10 Hz | Hardware real-time constraint |
| Stretch latency | 0.1-0.15s | End-to-end loop delay |
| Complexity (block) | O(p) per iteration | Fast but less accurate prediction |
| Complexity (element) | O(p^3) per iteration | Accurate, still tractable |
| Complexity (full GP) | O(i^2 * p^3) per iteration | Intractable for long runs |
| 100-iter runtime (QPGP element) | 19.31s | vs 649.23s for full GP (33x speedup) |
