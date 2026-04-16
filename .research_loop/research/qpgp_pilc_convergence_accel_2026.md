# Quasi-Periodic Gaussian Process Predictive Iterative Learning Control

- **URL**: https://arxiv.org/abs/2602.18014
- **Authors**: Unnati Nigam, Radhendushka Srivastava, Faezeh Marzbanrad, Michael Burke
- **Year**: 2026
- **Venue**: arXiv preprint (cs.RO, eess.SY, stat.ML) — submitted February 20, 2026. Not yet peer-reviewed.

---

## Key Contribution

The paper argues that standard P-type ILC — the update `u_{j+1} = u_j + L * e_j` — converges slowly under time-varying disturbances because it is reactive: it applies the *previous* iteration's error as correction, one full iteration behind. The QPGP-PILC contribution is making this correction *predictive*: before executing iteration j+1, the system GP-predicts what the error in j+1 will be, and pre-compensates.

For our context: the "quasi-periodic" assumption is that each race run re-executes the same trajectory (periodic), but conditions are not perfectly identical (aero variation, EKF state scatter, battery sag). The `omega` scalar encodes how correlated successive iterations are — close to 1 means near-identical, close to 0 means independent. This is *not* the same as the Butterworth Q-filter or the per-section alpha we already use; it is a fundamentally different mechanism operating at the *iteration-to-iteration* level rather than the within-iteration frequency domain.

The secondary contribution — reducing complexity from O(i² p³) to O(p³) — is what makes the method practical for embedded use. After 7 ILC iterations the naive GP approach would cost 49× a single iteration; the structural equation formulation costs exactly 1× regardless.

---

## Technical Approach (include equations/algorithms)

### System Model

Output at iteration i:

```
y_i = g(u_i) + z_i
```

where z_i is the (possibly time-varying) disturbance. The error propagation is:

```
e_{i+1} = (I - G_i * L_i) * e_i  +  (z_i - z_{i+1})
```

The term `z_i - z_{i+1}` is nonzero under drift and is precisely what QPGP estimates. Standard ILC implicitly assumes this term is zero.

### The QPGP Structural Equation

Iteration-to-iteration error evolution is modeled as:

```
x_{i+1} = omega * x_i + epsilon_{i+1}
```

- `omega in (-1, 1)` — scalar inter-iteration correlation. At omega=1, iteration errors are fully correlated (no drift); at omega=0, every iteration is independent.
- `epsilon_{i+1} ~ N(0, K_j)` — i.i.d. Gaussian noise with within-period covariance kernel `K_j`.

For multiple control dimensions:

```
e_{i+1} ~= (Omega ⊗ I_p) * e_i + epsilon_{i+1},   Omega = diag{omega_1, ..., omega_n}
```

The Kronecker structure means each dimension has its own scalar omega and covariance kernel. This factored form is what enables the O(p³) inference: no growing covariance matrix across iterations.

### Full Update Law

```
u_{i+1} = u_i  +  L_i * e_i  +  K_i * e_hat_{i+1}
```

- L_i is the standard P-type learning gain (reactive term).
- K_i is the predictive gain (proactive term).
- e_hat_{i+1} is the GP-predicted error for the next iteration.

This is the equation that directly answers the convergence acceleration question: the K_i * e_hat_{i+1} term adds a feedforward-of-future-error that reduces residual at iteration i+1 without waiting for the reactive L term to catch up.

### Two Prediction Strategies

**Block prediction — O(p) cost:**
```
e_hat_{i+1, j} = omega_j * e_{i, j}
```
Scales the full previous error vector by the scalar inter-iteration correlation. Requires no matrix inversion. Cost is trivial.

**Element-wise prediction — O(p³) cost:**
```
e_hat_{i+1,j}(t) = omega_j * e_{i,j}(t)
                 + K_{j; 1:t-1} * K_{j; t-1}^{-1} * (e_{i+1,j}^{(t-1)} - omega_j * e_{i,j}^{(t-1)})
```
Sequentially conditions on all timesteps within the *current* iteration observed so far (up to t-1), exploiting the within-period covariance kernel `K_j`. This is the "causal within-trial" update: as the current lap unfolds, the GP refines its prediction of the remaining trajectory in real time. For offline ILC this doesn't apply (we run the full iteration before updating), but for online use it is the key mechanism.

### Two-Stage Parameter Estimation

After each iteration, update hyperparameters {omega_j, K_j}:

**Stage I — alternating minimization on reduced marginal likelihood:**
Exclude the initial error block e_{1,j} (poorly initialized) and alternately minimize over omega_j and an unconstrained covariance K_tilde_j. Initialization from previous iteration's estimates makes this fast (warm-start).

**Stage II — projection onto valid periodic covariance kernels:**
- Enforce stationarity by averaging K_tilde_j along its diagonals (Toeplitz projection).
- Enforce positive definiteness by spectral truncation (zero negative eigenvalues).
- Output: K_hat_j, a valid symmetric PD periodic covariance matrix.

Theorem (consistent estimation): the estimates (omega_hat, K_hat) converge in probability to the true parameters as i → ∞.

### Convergence Theorem (Theorem 1)

For oracle predictors (perfect e_hat), the mean and covariance of the error sequence converge to zero if and only if:

- Element-wise: `sup_i ||B_i||_2 <= gamma_E < 1`
- Block: `sup_i ||A_i||_2 <= gamma_B < 1`

where A_i and B_i are functions of the plant Markov matrix G, learning gain L, prediction gain K, and omega. The condition `max_j ||K_j||_2 < C/2` bounds the within-period noise energy.

**Practical implication**: the spectral condition on B_i gives an explicit upper bound on K relative to L. From the manipulator experiment parameters (L=0.25, K=0.3), the ratio K/L ≈ 1.2 is safe. The Stretch robot uses K=1.5 with L=0.05, giving K/L = 30 — which seems aggressive but is safe because K multiplies e_hat (the prediction, which has lower variance than e_i at later iterations) rather than e_i directly.

### Gain Schedule

The paper uses `L_i = L_0 / i` (harmonic annealing) as a standard noise-robustness mechanism. Combined with a fixed K, this means the predictive term dominates in later iterations (L → 0, K fixed) and the correction is increasingly driven by the GP forecast rather than the raw noisy error.

### Quantitative Convergence Comparison

Vehicle tracking experiment (100 iterations, 8 m/s, drifting disturbance):
- Standard ILC: visible systematic deviation at iteration 50; converges only partially by iteration 100.
- QPGP-PILC (element-wise): closely follows reference by iteration 50; no visible deviation.
- QPGP-PILC (block): near-identical to element-wise at iteration 50.

The paper reports "approximately 40-50% reduction in iterations to reach target accuracy" vs standard ILC on the vehicle tracking task. In the manipulator experiment with injected disturbance, QPGP-PILC recovers pre-disturbance accuracy within 2-3 iterations; standard ILC requires many more.

### Stopping Criterion

The paper recommends terminating iteration when either:
1. GP prediction confidence intervals are "sufficiently tight" — i.e., posterior variance of e_hat falls below a task-specific threshold.
2. Absolute error improvement across consecutive trials falls below a threshold (which is what our current `convergence_threshold=0.002` already implements).

A variance-based stopping rule is strictly stronger: it avoids wasting iterations when the GP has converged even if absolute errors are still fluctuating.

---

## Results

### Computational Efficiency (Vehicle Tracking, 100 iterations, p=trajectory points)

| Method         | Wall time (sec) |
|----------------|----------------|
| Standard ILC   | 0.23           |
| QPGP-block     | 18.74          |
| QPGP-element   | 19.31          |
| Naive GP-PILC  | 649.23         |

QPGP-block is ~33× slower than standard ILC but 35× faster than naive GP-PILC. For 7 offline ILC iterations (our current count), the QPGP overhead is per-iteration O(p³) but only for small p. At p=100 (subsampled), QPGP-block adds ~0.05 ms per iteration — negligible in a precompute phase.

### Convergence Speed

- Vehicle tracking: ~40-50% fewer iterations needed vs standard ILC under drifting disturbances.
- Manipulator: 2-3 iterations to recover from injected disturbance vs many more for standard ILC.
- Stretch robot: better tracking accuracy at iteration 25 with smaller deviations.

### Robustness to Time-Varying Disturbances

The element-wise variant is most robust: as a new iteration begins, it uses within-lap observations to refine the prediction of remaining trajectory errors. Under injected mid-trial disturbances, this means the system adapts within the trial rather than waiting for the next iteration.

---

## Relevance to Our System

Our `compute_ilc_offset_table` implementation is a well-developed per-section P-type ILC with:
- Per-section alpha: [0.30, 0.50, 0.40, 0.45] (sections: pre-inflection, inflection, post-inflection, helix)
- 4th-order zero-phase Butterworth Q-filter at cutoffs [0.35, 0.40, 0.35, 0.35] Hz
- max_iterations=7 (recently increased from 5)
- convergence_threshold=0.002
- Velocity-corrected offsets (smooth derivative of position offset)

**Where QPGP-PILC adds new value:**

1. **The predictive gain K term is missing entirely.** Our update is `offset += alpha * filtered_error` — pure P-type. The QPGP insight is that adding `K * omega * e_i` to the next iteration's correction (equivalent to `offset += (alpha + K*omega) * filtered_error`) would accelerate convergence *without* changing the filter or the max-correction cap. This is the single most actionable insight not present in our current implementation.

2. **Per-section omega estimation replaces hand-tuned alpha.** Each of our four sections has a different alpha (0.30, 0.50, 0.40, 0.45), manually set after observing per-gate error trends. QPGP's omega is the data-driven equivalent: it measures how correlated successive iterations' errors are in that section. If omega is high (error pattern stable), a higher effective gain is safe; if omega is low (chaotic error), a lower gain is needed. Estimating omega from the last two iterations' per-section error vectors gives a principled basis for these alpha values.

3. **The 1/i gain schedule justifies tapering alpha.** We increased to 7 iterations and are considering 8. The 1/i schedule predicts that iteration-8 corrections should use alpha = alpha_1 / 8 ≈ 0.05-0.06. Our current per-section alphas (0.30–0.50) are calibrated for 5-7 iterations and may be too large for iteration 8, causing overshoot. An explicit annealing schedule (alpha_iter = alpha_0 / iter_num) would prevent this.

4. **GP variance-based stopping criterion.** The paper's convergence check using prediction confidence intervals is equivalent to checking whether the per-section error autocorrelation has stabilized. A simple implementation: if the dot product correlation between e_{i} and e_{i-1} (per section) changes by less than 5% between iterations, the section has converged and further iterations add noise rather than correction. Our current `convergence_threshold=0.002` on global avg_err may declare convergence too late (overall error improves but specific sections have plateaued and are being over-corrected).

5. **Per-section omega informs the 7→8 iteration decision.** If omega_section < 0.5 for all four sections after iteration 7, the error profile is weakly correlated across iterations and adding iteration 8 will mostly inject noise into the correction. If omega_section > 0.8, the residual is systematic and iteration 8 has high signal-to-noise ratio. Estimating omega directly from `dot(e_7, e_6) / dot(e_6, e_6)` per section gives a principled go/no-go criterion for iteration 8.

---

## Actionable Takeaways

1. **Add predictive gain K as an effective alpha multiplier.** The QPGP update `u_{i+1} = u_i + (L + K*omega) * e_i` is equivalent to using `effective_alpha = alpha * (1 + K/L * omega)`. With L = alpha = 0.40 and K = 0.30, omega = 0.80 (typical for our deterministic-ish sim), this gives effective_alpha = 0.40 * (1 + 0.75 * 0.80) = 0.64. This is a larger step at each iteration without changing the Q-filter. Start with K = 0.3 * alpha and verify no overshoot in the per-gate error curves.

2. **Estimate per-section omega from the last two ILC iterations.** After each pair of ILC iterations, compute:
   ```python
   omega_j = np.dot(e_i_section, e_{i-1}_section) / (np.dot(e_{i-1}_section, e_{i-1}_section) + 1e-9)
   omega_j = float(np.clip(omega_j, 0.0, 0.95))
   ```
   Use omega_j to dynamically set `effective_alpha_j = base_alpha_j * (1 + K_ratio * omega_j)`. Sections with stable, correlated errors (helix, gate-3 inflection) should get higher effective alpha; sections with chaotic errors (pre-inflection near gate-2) should get lower effective alpha, justifying the current alpha=0.30 there.

3. **Implement 1/i alpha annealing for iterations > 5.** Our current iteration-7 is already in a regime where the P-type theory predicts we should be tapering gains. Apply:
   ```python
   annealed_alpha = base_alpha * (5.0 / max(ilc_iter + 1, 5))
   ```
   This keeps alpha unchanged for iterations 1-5 and reduces it by 17% at iter 6, 29% at iter 7, 37% at iter 8. This directly addresses the gate-2 overcorrection observed in iteration 47 (alpha was manually reduced from 0.40 to 0.30 there).

4. **Use per-section omega as the go/no-go criterion for iteration 8.** Before increasing max_iterations from 7 to 8, run 7 iterations and inspect omega_j per section. If all sections have omega_j < 0.5 after iteration 7, the error pattern has randomized (possibly noise floor), and iteration 8 is unlikely to help. If any section has omega_j > 0.7, that section still has systematic residual and will benefit from one more iteration.

5. **Add GP variance-based per-section stopping.** Replace or augment the global `convergence_threshold=0.002` with a per-section variance check. A section stops updating when its normalized error variance `var(e_i_section) / var(e_1_section)` drops below 0.05 (95% variance reduction from baseline). This prevents over-correcting converged sections while allowing other sections to continue iterating.

6. **For the inflection section (alpha=0.50, highest gain): cap effective_alpha at 0.60.** The inflection section already uses the highest alpha and the highest Q-filter cutoff (0.40 Hz). The QPGP prediction could push effective_alpha above 0.60, which risks oscillatory overshoot across the S-turn. Hard-cap: `effective_alpha_inflection = min(base_alpha_inflection * (1 + K_ratio * omega_inflection), 0.60)`.

7. **Consider subsampled element-wise prediction (p_eff=140) for real competition deployment.** In PyBullet simulation, noise is deterministic so block prediction suffices. At competition, actuator noise, wind, and camera latency introduce genuine inter-run variation. Element-wise prediction at p_eff=140 (1 sample per 10 timesteps of the 1400-step trajectory) would provide within-lap error refinement at O(140³) ≈ 2.7M ops per iteration — feasible in real time at 100Hz. This is a deferred recommendation for competition deployment, not the benchmark phase.

---

## Limitations & Caveats

1. **No quadrotor experiments.** All three benchmarks are ground vehicle and manipulator. Quadrotor dynamics have 6-DOF coupling, thrust saturation, and rotor drag that couple translational axes in ways not present in planar systems. The omega model's dimensional decoupling assumption (`Omega = diag{omega_1, ..., omega_n}`) may underperform for a drone where x/y/z errors are coupled through attitude dynamics.

2. **Convergence theorem requires oracle predictor.** Theorem 1 assumes e_hat_{i+1} is accurate. In the first 3-4 iterations, the omega estimate is unreliable (few data points), and the predictive term may add noise rather than reduce error. The paper's convergence claims are asymptotic; early-iteration behavior is empirically validated but not theoretically bounded.

3. **O(p³) element-wise prediction requires subsampling.** For our 1400-step trajectory, full-resolution element-wise GP is ~2.7 billion operations — impractical offline, impossible online. The paper's Stretch robot runs at 10 Hz with ~50-100 trajectory points. We need p_eff ≤ 200 for tractability.

4. **Our simulation is near-deterministic, reducing omega signal.** In the PyBullet kinematic benchmark, z_i ≈ z_{i+1} always (same physics, same initial condition). In this case omega_j → 1 for all sections, and the QPGP prediction term degenerates to exactly the standard P-type correction. The QPGP advantage only manifests when there is genuine inter-iteration variability, which is present at competition but not in our benchmark loop.

5. **Per-section alpha design already captures the key intuition.** Our hand-tuned per-section alpha values (0.30, 0.50, 0.40, 0.45) effectively encode prior knowledge about which sections are "high-error-correlation" (inflection: high alpha) vs "low-error-correlation" (pre-inflection: low alpha). QPGP automates this, but the manual tuning in our current system already approximates the omega-informed gain design. Marginal improvement may be smaller than in a system using uniform alpha.

6. **Gain schedule interaction.** The 1/i annealing schedule is designed for noisy environments where error measurements are stochastic. In our deterministic inner sim, we can afford higher gains for more iterations. Applying 1/i annealing will reduce the effective gain below what is optimal for the deterministic case. The optimal schedule for our specific context is probably `alpha_iter = alpha_0 * exp(-k * max(0, ilc_iter - 5))` with k ≈ 0.1-0.2, which is gentler than 1/i.

7. **Two-stage parameter estimation needs ≥ 3-4 iterations.** The alternating minimization in Stage I needs enough data to separate omega from K_j. With max_iterations=7, we have enough history for estimation after iteration 3, but the first two iterations would use the uninitialized omega=0.8 prior. Sensitivity to this initialization should be tested.

---

## Key Parameters / Constants

| Parameter | Paper Value | Recommended for Our System | Description |
|-----------|-------------|---------------------------|-------------|
| omega | (-1, 1) | 0.75-0.90 (sim), 0.60-0.80 (competition) | Inter-iteration error correlation per section |
| L (learning gain) | 0.05-0.25 | 0.30-0.50 (matches our per-section alpha) | Reactive P-type correction gain |
| K (prediction gain) | 0.3-1.5 | 0.3 * alpha (≈0.09-0.15) | Predictive correction gain; start conservative |
| Effective alpha = alpha*(1+K/L*omega) | — | 0.38-0.64 range | Combined learning + prediction strength |
| Gain schedule | 1/i annealing | exp(-0.15 * max(0, iter-5)) | Gentler for deterministic sim |
| Complexity (block) | O(p) | Negligible | Per-iteration cost with p=1400 sections |
| Complexity (element-wise) | O(p³) | O(140³) = 2.7M with subsampling | Tractable offline with p_eff=140 |
| Complexity (naive GP-PILC) | O(i² p³) | Prohibited at i≥5 | Reference for why QPGP structural form matters |
| p_eff for GP | 100-200 | 140 (1 per 10 timesteps at 100 Hz = 10 Hz) | Subsampling for tractable O(p³) |
| Convergence speed | ~40-50% fewer iters | Expected 4-5 iters to reach 7-iter quality | Under genuine disturbance variation |
| Recovery from disturbance | 2-3 iterations | 2-3 iterations at competition | After mid-race disturbance injection |
| Vehicle tracking noise | sigma²=0.015 | — | For reference; our sim noise is lower |
| Stretch robot loop rate | ~10 Hz | Our ILC inner sim at 100 Hz | Latency tolerance comparison |
| Stopping: omega threshold | not explicit | omega_j < 0.5 → section converged | Go/no-go for iteration 8 decision |
| Stopping: variance reduction | 95% | var(e_i)/var(e_1) < 0.05 per section | Per-section convergence check |

*Analysis written 2026-04-15. Paper: arXiv:2602.18014. Existing partial analysis in qpgp_predictive_ilc_2026.md (iteration-to-iteration structure, block/element-wise formulas, computational cost). This document focuses on NEW actionable insights: predictive gain K integration, omega-based adaptive alpha, 1/i annealing for iter 7+, variance-based stopping, and the iter-7-to-8 decision criterion.*
