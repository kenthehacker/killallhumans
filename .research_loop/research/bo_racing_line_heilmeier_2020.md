# Computing the Racing Line using Bayesian Optimization

- **URL**: https://arxiv.org/abs/2002.04794
- **Authors**: Achin Jain, Manfred Morari (University of Pennsylvania / ETH Zurich)
- **Year**: 2020
- **Venue**: 59th IEEE Conference on Decision and Control (CDC), pp. 6192–6197

> Note on title: The paper is by Jain & Morari, not Heilmeier. Alexander Heilmeier is the author of a separate but related 2020 paper on minimum-curvature trajectory planning. The title "Computing the Racing Line using Bayesian Optimization" belongs to Jain & Morari (arXiv:2002.04794 / IEEE CDC 2020). The filename follows the user's requested convention.

---

## Key Contribution

Jain and Morari introduce a fully data-driven method for computing optimal racing lines by framing lap-time minimization as a black-box Bayesian optimization (BO) problem. The core insight is that the racing line can be parameterized by lateral waypoint deviations from the track center line, and that a Gaussian Process (GP) surrogate trained on (waypoint vector → minimum lap time) pairs can guide intelligent sampling far more efficiently than either exhaustive dynamic programming or naive random search. Unlike classical optimal control formulations that demand a closed-form vehicle model and differentiable cost landscape, their approach requires only the ability to evaluate minimum traversal time for any fixed path — a black-box oracle that can be any simulator or physical measurement.

The paper is also foundational infrastructure for the broader BayesRace framework (Jain, O'Kelly, Chaudhari, Morari — CoRL 2020), which extends the racing line computation into adaptive GP-corrected MPC controllers. The racing line BO is therefore positioned not as an isolated planner but as the front-end of a complete autonomous racing stack. The key practical claim is that a racing line competitive with dynamic-programming solutions can be obtained in under three minutes of wall-clock time, with a projected 10x speedup from a C++ reimplementation of the inner lap-time simulator.

---

## Technical Approach

### Problem Formulation

The racing line is formally the solution to a minimum-time optimal control problem over a fixed track with friction limits. This problem is NP-hard in general due to nonlinear dynamics and path constraints. Jain & Morari sidestep the continuous control formulation by decomposing it into two decoupled steps:

1. **Path selection**: choose a smooth geometric path through the track.
2. **Speed profile**: given the path, solve for the minimum-time speed schedule subject to curvature-dependent friction limits (a 1D convex problem solvable in closed form with a point-mass or kinematic bicycle model).

This separation is standard in autonomous racing literature (the "path + velocity decoupling" assumption) and is exact when the vehicle dynamics are dominated by lateral acceleration limits.

### Trajectory Parameterization

The track is represented by its center-line waypoints and a (possibly variable) track half-width at each waypoint. The algorithm places **n nodes** along the center line, with higher density near corners. Each node i is assigned a scalar lateral deviation w_i perpendicular to the center-line tangent, subject to |w_i| ≤ half-width. The full trajectory parameter vector is:

    w = [w_1, w_2, ..., w_n] ∈ R^n,    |w_i| ≤ half_width_i

The resulting waypoints are joined by **2D cubic spline interpolation** to produce a smooth, feasible path. For the UC Berkeley benchmark track with 10 nodes, each node has an (x, y) deviation, producing a **20-dimensional search space** — right at the boundary of where standard GP regression is typically considered reliable (the community rule of thumb is ≤15–20 dimensions).

### Minimum-Time Evaluation (Inner Oracle)

Given a fixed spline path, minimum traversal time τ(w) is computed via a point-mass or kinematic bicycle model with a friction circle constraint. The vehicle accelerates/decelerates along the path subject to:

    a_lat² + a_lon² ≤ μ²g²    (friction circle)
    v²/R ≤ μg                  (lateral acceleration limit at radius R)

This reduces to a 1D optimal control problem along the arc-length coordinate, solvable with forward-backward integration (the "minimum-time" algorithm of Verschueren et al. or the classic apex-finding method). This oracle evaluation dominates runtime — the paper reports it consumes more than 80% of total compute time per iteration.

### Bayesian Optimization Loop

The outer loop is standard BO:

1. Initialize with a small number of randomly sampled parameter vectors w, evaluate τ(w) for each.
2. Fit a GP surrogate f̂(w) ~ GP(μ, k) to the observed (w, τ) pairs, where k is a kernel capturing smoothness of the lap-time landscape.
3. Select the next query point by maximizing an **Expected Improvement (EI)** acquisition function:

        w_{t+1} = argmax_{w} EI(w | D_t)
        EI(w | D_t) = E[max(τ_best - f(w), 0)]

   EI balances exploitation (query near the current best) with exploration (query where GP uncertainty is high), avoiding the random-search problem of wasting evaluations on known-poor regions.
4. Evaluate τ(w_{t+1}) via the oracle, update the dataset D_t, refit the GP, and repeat until convergence or budget exhaustion.

The result of the BO loop is the w* achieving minimum observed τ, which defines the optimal racing line. The algorithm requires no gradient of τ with respect to w — the oracle is treated as a pure black box.

### Key Design Choices

- **Surrogate model**: Gaussian Process with a stationary kernel (exact kernel not specified in available excerpts, but Matern-5/2 is the standard choice in racing BO literature for smooth but non-infinitely-differentiable objectives).
- **Acquisition function**: Expected Improvement (EI), a default in BO packages such as SMAC and Spearmint.
- **Kernel hyperparameters**: inferred via marginal likelihood maximization (slice sampling mentioned as one approach in related BO literature).
- **Computational bottleneck**: the inner lap-time simulator, not the GP fitting or EI optimization. The authors note that C++ code generation for the simulator would yield a 10x speedup.

---

## Results

### Computational Speed

The published headline result is that BO computes a racing line **in under three minutes** (Python implementation) for the tested tracks. This is faster than dynamic programming approaches on the same problem and vastly faster than exhaustive random search, which requires enumerating an effectively infinite feasible set.

The 10x projected speedup via C++ code generation would reduce this to roughly 18 seconds, which would enable inter-lap reoptimization during a race.

### Quality vs. Baselines

BO consistently finds shorter lap times than naive random search given the same oracle evaluation budget. The key efficiency gain is that BO uses the GP posterior to avoid evaluating obviously poor trajectories — random search has no such mechanism and wastes the majority of its budget in unpromising regions of the search space.

Specific numerical lap-time deltas (in seconds) over random search for each track are contained in the paper's result tables, which were not accessible in full-text form during this analysis. The qualitative finding is that BO requires significantly fewer oracle calls to reach the same trajectory quality as random search.

### Platforms Tested

- Two ETH Zurich tracks for 1/43-scale miniature autonomous cars.
- One UC Berkeley track for the F1TENTH 1/10-scale platform.

The UC Berkeley track defines the official 20-dimensional benchmark (10 nodes × 2 lateral DOF per node).

---

## Relevance to Our System

Our system (`planning/racing_line.py`) currently uses multi-start L-BFGS-B with a **proxy objective** of `path_length + smoothness_weight × curvature²` evaluated over 10 candidate starts (N_STARTS=10). The fundamental limitation is that this proxy objective does not measure what we actually care about — kinematic-sim tracking error — so L-BFGS can converge to solutions that look geometrically smooth but are suboptimal from the drone's perspective.

The Jain & Morari framework is a direct architectural template for upgrading our racing line selection:

**Module affected**: `planning/racing_line.py`, specifically the `optimize()` method and its objective function.

**The key idea to borrow**: replace the proxy objective with a black-box oracle call — our existing PyBullet simulation or a fast kinematic forward-pass — and use BO to select among (or interpolate between) candidate lateral offset vectors. We already have 10 candidate racing lines from multi-start L-BFGS; Jain & Morari give us a principled framework for: (a) selecting the best among them, (b) generating new candidates in promising regions, and (c) quantifying uncertainty to know when to stop.

**Why this is hard to apply directly**: our track has only ~10 gates, so n is small. Our search space is effectively 10 × 2 = 20 dimensional (lateral and vertical offsets per gate), identical to the UC Berkeley benchmark in the paper. This is good news — we are operating in exactly the dimensionality regime where standard GPs work well.

**Current proxy vs. proposed oracle**: our objective is `sum(||p_{i+1} - p_i||) + 0.4 × sum(angle²)`, which approximates time by path length and smoothness. The Jain & Morari oracle runs a physics-based minimum-time simulation. For us, the oracle would be a short PyBullet rollout (or our kinematic predictor) measuring actual tracking error along the candidate racing line — directly aligning objective with metric.

---

## Actionable Takeaways

1. **Replace proxy selection with oracle-based BO.** Use our existing PyBullet sim (or a fast 2D kinematic rollout) as the oracle τ(w). Fit a GP to the (lateral-offset-vector → avg_tracking_error) observations from the 10 L-BFGS candidates we already have. Use EI to pick the next candidate to evaluate, rather than relying on the proxy ranking.

2. **Reuse our 10 multi-start L-BFGS results as BO warm-start data.** Instead of discarding 9 of the 10 candidates after picking the best by proxy, feed all 10 (offset vectors, oracle-evaluated tracking errors) as initial observations D_0 into the GP. This bootstraps the BO with a rich prior at zero extra cost.

3. **Implement a lightweight kinematic oracle.** A full PyBullet rollout may be too slow for BO's inner loop (BO needs 20–50 oracle evaluations minimum). Build a fast 2D kinematic simulation (bicycle or point-mass model) that computes predicted path-following error along a fixed trajectory using our MPC/geometric tracker's known behavior. Target <1 second per oracle call to make BO tractable.

4. **Keep the search space the same (n×2 lateral offsets).** Our current parameterization — 2D lateral offsets per gate, bounded by gate half-width — maps exactly to the paper's framework. No reparameterization needed.

5. **Use Expected Improvement (EI) as the acquisition function.** EI is the default in scikit-optimize, GPyOpt, and BoTorch, all available in our environment. Start with a Matern-5/2 kernel (smooth but robust to non-smooth objective landscapes).

6. **Set an oracle budget of 30–50 evaluations.** The paper achieves competitive results with modest budgets on similarly-sized problems. With 10 warm-start observations, 20–40 BO iterations (each taking <1s with a kinematic oracle) should converge.

7. **Consider multi-fidelity BO.** Use the cheap proxy objective (our current L-BFGS cost) as the low-fidelity approximation and the kinematic oracle as high-fidelity. This is a natural extension of the paper's framework and could reduce oracle calls by another 2–3x.

8. **If the kinematic oracle is still too slow, use the GP to select among the 10 existing candidates only.** Even fitting a GP to 10 points and ranking them by posterior mean (or lower confidence bound) is strictly better than ranking by proxy, because it uses observed tracking-error data rather than a geometric surrogate.

9. **Apply the C++ speedup insight.** The paper notes that the oracle dominates compute at >80% of time, and C++ gives 10x speedup. For us, this suggests that if the PyBullet oracle is the bottleneck, porting the kinematic integration to a compiled extension (or using numba) will dominate any other BO implementation optimization.

10. **Increase node density near corners.** The paper explicitly places more nodes near track corners to prevent the optimizer from cutting around them. Our current uniform gate spacing may be suboptimal; the gate at the S-turn (gate-3) and helix (gates 5–8) would benefit from denser parameterization within each gate's interior.

---

## Limitations & Caveats

**Ground vehicle vs. drone dynamics.** The paper's vehicle model is a planar kinematic bicycle with a friction circle. Our drone is a 6-DoF rigid body with quadrotor dynamics, drag, and attitude loop bandwidth. The point-mass minimum-time oracle does not model these effects. Any oracle we build must account for the drone's acceleration bandwidth and yaw rate limits, not just lateral g-limits.

**3D vs. 2D search space.** The paper operates entirely in the horizontal plane. Our gates are at varying altitudes (the helix section), and the optimal offset is 3D (lateral within the gate plane, plus vertical). Our parameterization already handles this via 2D offsets in the gate's local frame, so the extension is natural — but it doubles the effective problem dimensionality compared to the paper's planar benchmark.

**GP scaling with dimensionality.** The paper's 20D problem is at the edge of standard GP reliability. Our full 3D problem (if we add vertical offsets) would be 30D (10 gates × 3D), which is beyond the comfortable regime. Dimensionality reduction (e.g., REMBO or additive GPs) may be needed if vertical offsets are included.

**Oracle cost.** The paper's oracle (kinematic minimum-time integration) runs fast. If our oracle requires a full PyBullet rollout (physically simulated at 240 Hz for 20 seconds), each evaluation takes ~10s of wall time — making 50 BO iterations cost ~8 minutes. This is acceptable for offline optimization but motivates a fast kinematic surrogate.

**Static track assumption.** The paper optimizes for a known, fixed track. At competition, track geometry may differ from our prior, requiring rapid re-optimization. The BO framework supports this (transfer learning from a prior GP), but it requires careful implementation.

**No formal convergence guarantee for EI at high dimension.** EI's convergence guarantees are largely established for low-dimensional problems. At 20D, convergence is empirical rather than theoretically guaranteed, and the paper's results are on specific small-scale tracks.

---

## Key Parameters / Constants

| Parameter | Value | Source |
|-----------|-------|--------|
| Number of track nodes (n) | 10 (UC Berkeley benchmark) | Paper §III |
| Search space dimension | 2n = 20 (x,y deviation per node) | Paper §III |
| Acquisition function | Expected Improvement (EI) | Paper §II-C |
| BO compute time (Python) | < 3 minutes | Paper §IV |
| Projected speedup (C++) | 10x → ~18 seconds | Paper §IV |
| Oracle fraction of total runtime | > 80% | Paper §IV |
| Node density | Higher near corners | Paper §III |
| Spline type | 2D cubic spline interpolation | Paper §III |
| Track inputs required | Center-line xy, track width, 3 vehicle params | Paper §III |
| Platforms tested | 1/43-scale (ETH), 1/10-scale F1TENTH (UC Berkeley) | Paper §I |

The "3 vehicle parameters that can be physically measured" mentioned for the kinematic model are not specified in available excerpts but are consistent with standard bicycle model identification: maximum lateral acceleration (μg), maximum longitudinal acceleration, and vehicle wheelbase.

---

Sources:
- [arXiv:2002.04794 — Computing the racing line using Bayesian optimization](https://arxiv.org/abs/2002.04794)
- [IEEE CDC 2020 — DOI:10.1109/CDC42340.2020.9304147](https://ieeexplore.ieee.org/document/9304147/)
- [GitHub: jainachin/bayesrace](https://github.com/jainachin/bayesrace)
- [Semantic Scholar entry](https://www.semanticscholar.org/paper/Computing-the-racing-line-using-Bayesian-Jain-Morari/c587335f8cac9b8b4185b833547022344c04d94f)
