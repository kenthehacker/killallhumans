# Quasi-Periodic Gaussian Process Predictive Iterative Learning Control

- **URL**: https://arxiv.org/abs/2602.18014
- **Authors**: Unnati Nigam, Radhendushka Srivastava, Faezeh Marzbanrad, Michael Burke
- **Year**: 2026
- **Venue**: arXiv preprint (cs.RO, eess.SY, stat.ML) — submitted February 20, 2026

---

## Key Contribution

Standard iterative learning control (ILC) assumes that the disturbance affecting a repetitive task is identical across iterations. When this holds, the classical P-type update `u_{j+1} = u_j + L * e_j` is provably convergent. But in real robotic systems — drones experiencing wind variation, manipulators undergoing thermal expansion or joint wear, vehicles on degrading surfaces — the disturbance drifts slowly across trials. Standard ILC then either diverges (if the drift is fast) or converges to a nonzero residual error.

The paper's core contribution is extending ILC with a Quasi-Periodic Gaussian Process (QPGP) that explicitly models how the per-iteration error profile evolves across trials. Rather than applying the previous iteration's error directly, the method GP-predicts what the next iteration's error will be and corrects for it proactively:

```
u_{i+1} = u_i + L_i * e_i + K_i * e_hat_{i+1}
```

where `e_hat_{i+1}` is the QPGP's prediction of the next-iteration error. The "quasi-periodic" kernel reflects that each iteration is approximately (but not exactly) periodic: the drone re-runs the same trajectory, but actuator wear or wind conditions may shift the disturbance pattern slowly over time.

A secondary contribution is computational: naive GP-based predictive ILC costs O(i² p³) at iteration i (because the full history grows). The structural equation formulation of QPGPs reduces this to O(p³) per iteration — constant regardless of how many iterations have elapsed — enabling real-time parameter estimation within the control loop.

---

## Technical Approach

### System Model

The paper models the closed-loop dynamics as:

```
y_i = g(u_i) + z_i
```

where `y_i` is the output at iteration i, `g(·)` is the closed-loop plant map, and `z_i` is the disturbance. The error evolves as:

```
e_{i+1} = (I - G_i L_i) e_i + (z_i - z_{i+1})
```

The key distinction from standard ILC is that the disturbance increment `z_i - z_{i+1}` is nonzero and GP-predicted rather than assumed zero.

### QPGP Structural Equation

The quasi-periodic error evolution across iterations is modeled as:

```
x_{i+1} = omega * x_i + epsilon_{i+1}
```

where `omega in (-1, 1)` is a scalar capturing inter-iteration correlation (how strongly one trial's error predicts the next), and `epsilon_i ~ N(0, K_j)` is i.i.d. Gaussian noise with within-period covariance kernel `K_j`. For multi-output systems (one per control dimension), a diagonal approximation decouples dimensions:

```
e_{i+1} ≈ (Omega ⊗ I_p) * e_i + epsilon_{i+1},  Omega = diag{omega_1, ..., omega_n}
```

This is the structural equation that enables the O(p³) inference: because `omega` is constant across iterations, the covariance structure factors cleanly and the GP posterior can be computed without assembling the full i×i×p×p covariance matrix.

### Two Prediction Strategies

**Block prediction (O(p) per iteration):**
```
e_hat_{i+1,j} = omega_j * e_{i,j}
```
Scales the previous error block by the inter-iteration correlation. Cheapest option; ignores within-period covariance structure.

**Element-wise prediction (O(p³) per iteration):**
```
e_hat_{i+1,j}(t) = omega_j * e_{i,j}(t) + K_{j;1,t-1} * K_{j;t-1}^{-1} * (e_{i+1,j}^{(t-1)} - omega_j * e_{i,j}^{(t-1)})
```
Sequentially conditions on all observations within the current iteration up to timestep t-1, exploiting the within-period covariance kernel `K_j`. This is the "element-wise" variant that achieves the fastest convergence.

### Parameter Estimation (Two-Stage Algorithm)

After each iteration, the hyperparameters `{omega_j, K_j}` are re-estimated:

**Stage I:** Alternating minimization on a reduced likelihood that excludes the initial error block `e_{1,j}`. This iteratively estimates `omega_j` and an unconstrained covariance `K_tilde_j`.

**Stage II:** Project `K_tilde_j` onto the space of valid periodic covariance kernels by averaging along diagonals (enforcing stationarity) and applying spectral truncation (enforcing positive definiteness). This yields the final `K_hat_j`.

Parameter estimates converge in probability to the true parameters as i → ∞ (consistent estimators). Initialization from the previous iteration's estimates makes re-estimation cheap at each iteration.

### Convergence (Theorem 1)

For oracle predictors (perfect `e_hat_{i+1}`), mean and covariance convergence requires spectral bounds:
- Element-wise: `sup_i ||B_i||_2 ≤ gamma_E < 1`
- Block: `sup_i ||A_i||_2 ≤ gamma_B < 1`

with `max_j ||K_j||_2 < C/2`. These conditions constrain the learning gain L and prediction gain K relative to the plant gain G. They are analogous to the standard ILC convergence condition but extended to the predictive setting.

Note: the paper does not provide an explicit Q-filter in the traditional ILC sense. The "robustification" is achieved through the GP structure itself — the quasi-periodic kernel implicitly smooths the learned correction and the `omega` parameter controls how aggressively previous errors are applied.

---

## Results

### Vehicle Trajectory Tracking (100 iterations)

- Racetrack: `R(s) = 10 + 2 sin(2s) + sin(3s)`, uniformly spaced points
- Speed: 8 m/s; Wheelbase: 0.5 m
- Steering bias: b_0 = 0.15, drift: b_1 = 0.01 (per-iteration drift)
- Measurement noise: sigma^2 = 0.015
- Gain schedule: 1/i annealing

Computation times (100 iterations):
| Method | Time (sec) |
|--------|-----------|
| Standard ILC | 0.23 |
| QPGP-block | 18.74 |
| QPGP-element | 19.31 |
| GP-PILC (naive) | 649.23 |

At iteration 50, QPGP-PILC trajectory closely aligns with reference; standard ILC shows visible deviations. Element-wise variant converges fastest; block variant achieves comparable accuracy at lower cost.

### Manipulator (3-Link Planar)

- Link lengths: l_1 = 1.0, l_2 = 1.0, l_3 = 0.5 m
- Reference: Circle at (1.5, 1.0), radius 0.5 m
- Joint bias: sinusoidal + Gaussian components, sigma in {0.1, 0.2}
- Learning gain: L = 0.25; Prediction gain: K = 0.3

QPGP-PILC shows fastest convergence. When an external disturbance is injected mid-trial, standard ILC requires many iterations to recover; QPGP-PILC restores pre-disturbance accuracy within 2-3 iterations.

### Stretch Robot (Real Hardware)

- Trajectory: Lissajous curve `y(t) = 0.25 + 0.04 sin(3t + pi/2)`, `z(t) = 0.45 + 0.02 sin(2t)`
- Control loop: ~10 Hz (0.1 s sleep); Latency: 0.1-0.15 s
- Learning gain: L = 0.05; Prediction gain: K = 1.5

Achieves faster convergence than standard ILC. Better tracking at iteration 25 with reduced deviations. Handles actuator noise and latency effectively.

---

## Relevance to Our System

Our current ILC implementation uses a per-section correction scheme (iteration 26) with Gaussian smoothing (sigma=10 timesteps) applied to the learned correction signal. The QPGP paper is relevant in three ways:

**1. The "quasi-periodic" assumption maps to our situation.** Each benchmark run on the same track is approximately periodic: same trajectory shape, but small variations in initial conditions, EKF state, and control response. The `omega` parameter would capture iteration-to-iteration correlation in our position error profile.

**2. The block prediction variant is directly implementable.** The update `e_hat_{i+1,j} = omega * e_{i,j}` is a one-liner that replaces our current `delta = -alpha * error_array` update. If omega ≈ 0.7-0.9 (typical for slowly-drifting systems), this gives a softer correction than alpha=1, reducing overshoot.

**3. The element-wise prediction exploits within-trial spatial correlation.** Our 1400-step trajectory has strong within-period correlation: error at step t is correlated with error at steps t±50 (half a gate segment). The GP kernel `K_j` captures this and refines predictions as the trial progresses — conceptually similar to our current Gaussian smoothing but learned from data rather than hand-tuned.

**Current gap:** Our sigma=10 Gaussian smoothing is a fixed, non-data-driven choice. QPGP would learn the effective correlation length from the error covariance data across iterations. If the true correlation length is longer (e.g., sigma=30-50 at some gates), our current fixed smoothing under-smooths and leaves noise in the correction.

---

## Actionable Takeaways

**1. Replace fixed Gaussian smoothing with learned omega scaling.**
Instead of always applying `delta = -alpha * error_array`, use `delta = -omega * alpha * error_array` where `omega` is estimated from iteration-to-iteration error correlation:
```python
# After iteration i, estimate omega per gate section:
omega_j = dot(e_i, e_{i-1}) / dot(e_{i-1}, e_{i-1})
omega_j = clip(omega_j, 0.0, 0.95)  # prevent negative or explosive gains
```

**2. Use block prediction as a drop-in replacement.**
The block prediction `e_hat_{i+1} = omega * e_i` is essentially a "momentum" term. It would make corrections more conservative in early iterations (when error is large and noisy) and more aggressive in later iterations (when residual error is small and consistent).

**3. Implement K * e_hat feedforward for faster convergence.**
Add the prediction gain K term:
```python
u_{i+1} = u_i + L * e_i + K * omega * e_i  # = u_i + (L + K*omega) * e_i
```
This is just a gain rescaling, but derived from the convergence theory rather than hand-tuning.

**4. Adaptive smoothing via kernel length-scale estimation.**
Stage II of the parameter estimation (diagonal averaging of K_tilde) gives an empirical estimate of the within-period covariance structure. This could replace our hand-tuned sigma=10 Gaussian with a data-driven kernel length-scale. Practical implementation: compute the empirical autocorrelation of the error signal and fit a Gaussian to it to extract sigma.

**5. Control loop at 100Hz (dt=0.01s), p=1400 steps.**
Block prediction costs O(p) = O(1400) per iteration — negligible. Element-wise prediction costs O(p³) = O(2.7 billion) — impractical at full resolution. Solution: subsample to p_eff=100-200 points for GP inference, then interpolate back to 1400 steps. At p=200, O(p³) = O(8 million) — fast at 100Hz.

---

## Limitations & Caveats

1. **No quadrotor experiments.** All validation is on ground vehicles and manipulators. Transfer to a drone with 6DOF dynamics and aerodynamic coupling is nontrivial.

2. **Convergence theorem requires oracle predictor.** The formal convergence guarantee assumes `e_hat_{i+1}` is accurate. In practice, early iterations have poor estimates (few data points), and the convergence rate may degrade significantly.

3. **O(p³) per iteration is still expensive at full resolution.** For p=1400 (our trajectory length), even a single GP solve per iteration is ~2.7B operations. The paper's Stretch robot experiment uses p≈50-100 points at 10Hz. At 100Hz with 1400 steps, subsampling is mandatory.

4. **Hyperparameter tuning required.** The quasi-periodic kernel has multiple hyperparameters (omega, sigma^2, K_j entries). The two-stage estimation algorithm may not converge in the first 5-10 iterations, meaning early corrections are driven by poor estimates.

5. **Deterministic sim reduces benefit.** In our PyBullet-based benchmark, the disturbance z_i is perfectly reproducible across iterations. The QPGP inter-iteration prediction (omega term) adds no value when z_i = z_{i+1} always — standard ILC with L=1 suffices. The benefit emerges at competition time with real-world variability.

6. **The paper is a preprint (February 2026).** No peer review yet; the convergence theorem proof details are not fully verifiable.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| omega | in (-1, 1) | Inter-iteration correlation coefficient |
| omega (typical) | 0.7-0.9 | Reasonable range for slowly-drifting systems |
| L (learning gain) | 0.05-0.25 | Conservative to prevent overshoot |
| K (prediction gain) | 0.3-1.5 | Scales the predictive correction |
| Complexity (block) | O(p) | Per iteration, negligible |
| Complexity (element-wise) | O(p³) | Per iteration; requires subsampling for p>200 |
| Complexity (naive GP-PILC) | O(i² p³) | Prohibitive for large i |
| Noise level (vehicle) | sigma^2 = 0.015 | Measurement noise variance |
| Convergence | ~50 iterations | For vehicle tracking at 8 m/s |
| Real robot loop rate | ~10 Hz | Stretch robot experiment |
| Gain schedule | 1/i annealing | Diminishing step size for noise robustness |
| Effective p for GP | 100-200 | Practical subsampling to make O(p³) tractable |

For our drone system at 100Hz with 1400 steps (~14s trajectory):
- Recommended omega initial estimate: 0.8 (assume 80% correlation across iterations)
- Recommended learning gain: L = 0.3 (conservative; our current per-section alpha)
- Recommended K: 0.5 (adds 50% of the omega-scaled previous error as prediction)
- Effective p for GP: 140 (1 point per 10 timesteps = 1 per 0.1s = ~10Hz resolution)

*Analysis written 2026-04-14. Paper: arXiv:2602.18014. Code: https://github.com/unnati-nigam/QPGP-PILC*
