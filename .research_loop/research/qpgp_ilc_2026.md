# Quasi-Periodic Gaussian Process Predictive Iterative Learning Control
- **URL**: https://arxiv.org/abs/2602.18014
- **Authors**: Not fully specified in abstract
- **Year**: 2026
- **Venue**: arXiv preprint

---

## Key Contribution

This paper enhances conventional Iterative Learning Control (ILC) by incorporating Quasi-Periodic Gaussian Processes (QPGPs) into a predictive framework. The core innovation is the ability to PREDICT the next iteration's error profile before executing it, enabling proactive rather than reactive feedforward correction. Standard ILC uses error from trial j to correct trial j+1; this method uses GP regression to predict what the error in trial j+1 WOULD be, and pre-compensates accordingly.

The method is particularly valuable for systems with time-varying (drifting) disturbances — something standard ILC handles poorly. By modeling the disturbance evolution across iterations with a quasi-periodic GP kernel, the method maintains accuracy even as conditions change.

---

## Technical Approach

### Extension of Standard ILC

Standard P-type ILC update: `u_{j+1} = u_j + L * e_j`

This assumes the disturbance in trial j+1 will be identical to trial j. When disturbances drift (e.g., wind changes, battery degradation, temperature effects), this assumption fails and ILC may diverge.

The QPGP extension replaces the simple error with a GP-predicted error:

```
u_{j+1} = u_j + L * GP_predict(e_1, e_2, ..., e_j)
```

The GP model learns the pattern of how disturbances evolve across iterations and extrapolates to the next iteration.

### GP Model Purpose

The QPGP captures:
- **Quasi-periodic structure**: disturbances that repeat approximately (e.g., thermal cycles, repetitive mechanical wear)
- **Slow drift**: gradual changes in system behavior across iterations
- **Fast transients**: sudden disturbance changes that should be learned quickly

The quasi-periodic kernel: `k(i, i') = sigma^2 * exp(-2*sin^2(pi*|i-i'|/p) / l^2) * exp(-|i-i'|^2 / (2*l'^2))`

The periodic component captures repeating patterns; the squared-exponential envelope captures drift.

### Computational Efficiency

Key advantage: O(p^3) per iteration instead of O(i^2 * p^3) for standard GP-ILC, where p = trajectory discretization points and i = iteration count. This is achieved by a structural equation formulation that avoids growing the GP dataset with each iteration.

---

## Results

### Experimental Validation

Three benchmarks tested:
1. Autonomous vehicle trajectory tracking
2. Three-link manipulator control
3. Real-world Stretch robot experiments

**No quadrotor experiments** — the paper is general-purpose ILC, not drone-specific.

Claims: "converges faster and remains robust under injected and natural disturbances while reducing computational cost" compared to standard ILC and conventional GP-based methods.

No specific iteration counts or quantitative convergence metrics provided in the abstract.

---

## Relevance to Our System

**Moderate relevance.** The QPGP extension is most valuable for systems with drifting disturbances — but our kinematic sim is deterministic, so disturbances DON'T drift. Standard P-type ILC is sufficient and simpler.

However, the paper validates the ILC approach more broadly:
1. Confirms ILC is still an active research area (2026 publication)
2. Shows that even simple P-type ILC converges well for stationary systems
3. The predictive aspect could be useful if we move to PyBullet sim with stochastic elements

For our current iteration, standard P-type ILC (Schoellig 2012) is the right choice. The QPGP extension would be relevant for competition deployment where conditions vary between runs.

---

## Actionable Takeaways

1. **Use standard P-type ILC for our deterministic sim.** The QPGP extension is overkill for a deterministic kinematic sim with no drifting disturbances.

2. **Consider QPGP-ILC for PyBullet integration later.** When we move to PyBullet with more realistic physics (and potentially stochastic elements), the GP-based approach could help.

3. **The O(p^3) complexity is acceptable.** Even for 1400 trajectory points, p^3 ≈ 2.7 billion — but the GP only needs to model a low-dimensional correction, not the full state. In practice, a coarser discretization (p=100) suffices for the correction signal.

---

## Limitations & Caveats

1. No quadrotor experiments — all validation is on manipulators and ground vehicles
2. No convergence iteration counts provided
3. The quasi-periodic kernel requires hyperparameter tuning (period p, lengthscales l, l')
4. Overkill for deterministic systems

---

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Complexity | O(p^3) per iteration | vs O(i^2*p^3) for standard GP-ILC |
| Kernel | Quasi-periodic (periodic * SE) | Captures drift + repetition |
| Convergence | "faster than standard ILC" | No specific numbers |

*Analysis written 2026-04-14. Paper: arXiv:2602.18014.*
