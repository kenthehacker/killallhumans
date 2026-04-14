# COP: Control & Observability-aware Planning
- **URL**: https://arxiv.org/abs/2203.06982
- **Authors**: Christoph Bohm, Pascal Brault, Quentin Delamare, Paolo Robuffo Giordano, Stephan Weiss
- **Year**: 2022
- **Venue**: IEEE ICRA 2022

## Key Contribution

This paper presents the first framework that jointly optimizes trajectory tracking robustness (via closed-loop state/input sensitivity) and state estimation quality (via observability) for quadrotor UAVs. The core insight is that these two objectives are fundamentally antagonistic: robust tracking favors trajectories insensitive to parameter perturbations, while good observability requires trajectories that excite parameter-dependent dynamics. Rather than picking one or the other, the authors formulate a Multi-Objective Optimization Problem (MOOP) and reduce it to a Single-Objective Optimization Problem (SOOP) using the Augmented Weighted Tchebycheff (AWT) method. This lets them find Pareto-optimal compromise trajectories that simultaneously improve both tracking error and estimation uncertainty compared to unoptimized baselines.

## Technical Approach

### System Model
The quadrotor has a 13-dimensional state vector (position r, velocity v, quaternion q, angular velocity omega). The uncertain parameters are the thrust coefficient k_f and drag coefficient k_m. A Lee geometric controller (SE(3)) tracks reference trajectories with gains k_r, k_v, k_i (position/velocity/integral) and k_q, k_omega (attitude).

### Three Objectives
1. **State Sensitivity F_Pi**: Integral of the norm of dx/dp over the trajectory duration T, measuring how much states vary with parameter perturbations. Lower is more robust.
2. **Input Sensitivity F_Theta**: Integral of the norm of du/dp, measuring control effort sensitivity. Lower means less actuator stress under uncertainty.
3. **Observability F_E2LOG**: Negative of the minimum singular value of the empirical local observability Gramian. Maximizing the smallest eigenvalue ensures the estimation problem remains well-conditioned.

### Augmented Weighted Tchebycheff Method
The key equation is:

```
U(a) = max_i { lambda_i * |F_i(a) - F_i^O| } + rho * sum_j |F_j(a) - F_j^O|
```

Where:
- `lambda_i = w_i / |F_i^N - F_i^O|` normalizes each objective by its range on the Pareto front
- `F_i^O` is the utopia point (best achievable value for objective i individually)
- `F_i^N` is the nadir point (worst value of objective i across all individual optima)
- `w_i` are user-defined weights summing to 1
- `rho` is a small augmentation parameter in [0.0001, 0.01], set to 0.0001 in experiments

The normalization by (F_i^N - F_i^O) is critical: it makes the weights scale-invariant and prevents one objective from dominating simply because it has larger numerical values. The max-based (Tchebycheff) term ensures the solution stays near the Pareto front, while the augmentation (rho * sum) breaks ties and ensures strict Pareto optimality.

**Why not linear scalarization?** The paper explicitly warns that linear scalarization (weighted sum) only works when the Pareto front is convex. For concave Pareto fronts, linear scalarization converges to extrema (pure single-objective solutions), missing the interior compromise solutions entirely. The AWT method handles both convex and concave fronts.

### Multi-Step Optimization Pipeline
1. **Precondition**: Generate random Bezier trajectory, optimize for dynamic feasibility (rotor speed limits) and basic tracking accuracy.
2. **Individual minimization**: Solve for each objective independently to compute utopia and nadir points. This is mandatory for the normalization.
3. **Combined AWT optimization**: Apply the AWT utility with equal weights [1/3, 1/3, 1/3].
4. **Post-filtering**: Accept the COP solution only if it improves over the initial trajectory on both the S/I-S metric and E2LOG. This prevents pathological cases where the optimizer degrades both objectives.

Trajectories are parameterized as piecewise Bezier curves (5 pieces, degree d = 2*n_jc - 1) with C^(d-1) continuity. The solver is COBYLA (derivative-free) from NLOPT.

### Pareto Front Exploration
The utopia/nadir computation (Step 2) effectively characterizes the Pareto front extremes. By varying w_i across the simplex, one could trace out the full Pareto front, though the paper only evaluates the equal-weight balanced point. The normalization ensures that any weight vector maps to a meaningful trade-off point regardless of objective scaling.

## Results

Experiments use the Hummingbird quadrotor model over 20-second trajectories with 20 random target waypoints:

**Tracking performance** (30 flights per trajectory, parameter perturbation +/-1%):
- S/I-S optimized trajectories achieve the best tracking (median ~0.8x baseline error)
- COP optimized trajectories achieve intermediate tracking (~1.0x baseline)
- E2LOG optimized trajectories are worst for tracking (~1.2x baseline)
- At +/-5% perturbation, differences diminish — sensitivity metrics lose effectiveness at large deviations

**Estimation uncertainty** (10 runs, +/-30% initial parameter error):
- E2LOG optimized: lowest final uncertainty on k_f (~0.08 std)
- COP optimized: comparable at ~0.10 std
- S/I-S optimized: ~0.12 std
- Baseline: ~0.15 std

**Trade-off summary**: COP achieves approximately 70% of pure S/I-S tracking performance while capturing ~80% of pure E2LOG estimation performance. Neither objective is maximally satisfied, but the combined solution dominates the unoptimized baseline on both axes.

**Computation**: Full COP pipeline takes ~2.4 hours on a single-threaded AMD Ryzen 5 3600. Individual objective optimizations take 15-38 minutes each.

## Relevance to Our System

Our racing line selector (`planning/racing_line.py`, `_sim_based_selection`) currently evaluates 10 candidates using a composite score:

```python
score = 0.7 * avg_err + 0.3 * worst_gate_err
```

This is pure linear scalarization of two error metrics, and race_time is computed but discarded (only used as a tiebreaker in sort). We want to add race_time as a real objective while maintaining tracking accuracy. The COP paper directly informs this:

1. **Linear scalarization is fragile**: Our current 0.7/0.3 weighting assumes a convex trade-off surface between avg_error and worst_gate_error. If the Pareto front between tracking error and race time is concave (plausible — aggressive trajectories that are fast may have discontinuously worse tracking), linear weighting will snap to one extreme or the other.

2. **The AWT normalization is the key insight**: Before combining objectives, we must normalize each by its range across our 10 candidates. Compute utopia (best avg_err across candidates, best race_time across candidates) and nadir (worst of each). Then the Tchebycheff formulation ensures we find the candidate closest to the ideal point in normalized space, regardless of whether errors are in meters and times are in seconds.

3. **Post-filtering prevents regression**: The paper's accept-only-if-improved-on-both-axes rule maps directly to our needs. We should reject any candidate that is Pareto-dominated (worse on both tracking and time than another candidate).

## Actionable Takeaways

1. **Replace linear scalarization with AWT in `_sim_based_selection`**: Compute utopia/nadir across all 10 candidates for (avg_err, worst_gate_err, race_time). Apply AWT with rho=0.0001. Start with equal weights [1/3, 1/3, 1/3], then tune.

2. **Pareto-filter first**: Before scoring, eliminate Pareto-dominated candidates. If candidate A is worse than candidate B on all objectives, discard A. This is cheap and prevents selecting clearly inferior solutions.

3. **Normalize before combining**: The critical step is dividing each objective by (nadir - utopia) across the candidate set. Without this, a 0.1m error difference and a 1.0s time difference are incommensurable.

4. **Use Tchebycheff (max) not sum**: `score = max(lambda_i * |F_i - F_i_utopia|) + rho * sum(...)` ensures the selected candidate is as close to the ideal point as possible in the worst-case dimension, preventing one objective from being sacrificed entirely.

5. **Weight tuning maps to priorities**: w_tracking > w_time if we want to preserve current tracking quality while getting modest time improvements. Start conservative (e.g., [0.5, 0.2, 0.3] for avg_err, worst_gate_err, race_time) and adjust based on benchmark regressions.

## Limitations & Caveats

- **Computational cost**: The full COP pipeline is expensive (2.4 hours). However, we only need the selection step (apply AWT scoring to pre-computed candidates), not the trajectory re-optimization, so cost is negligible in our case.
- **Sensitivity metrics degrade at large perturbations**: At +/-5% parameter error, the S/I-S metric loses predictive power. Our sim-based evaluation directly measures tracking error rather than sensitivity, so this limitation does not affect us.
- **Only one Pareto point explored**: The paper computes a single balanced solution rather than the full Pareto front. For our 10-candidate discrete set, we can trivially compute the full Pareto front and select from it.
- **Assumes full state knowledge**: The sensitivity analysis assumes perfect state estimation during tracking. Our kinematic sim also assumes perfect tracking, so this is a matched assumption.
- **No real-world validation**: Results are simulation-only. The authors acknowledge the need for closed-loop flight experiments.
- **COBYLA local optimizer**: May not find global optimum. Not relevant to our selection problem (we evaluate all candidates exhaustively).

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| rho (augmentation) | 0.0001 | AWT augmentation term; range [0.0001, 0.01] |
| w_i (COP weights) | [1/3, 1/3, 1/3] | Equal weighting across three objectives |
| w_i (S/I-S weights) | [1/2, 1/2, 0] | Pure tracking robustness |
| Trajectory duration T | 20 s | Fixed for all experiments |
| Bezier pieces | 5 | Piecewise trajectory segments |
| Parameter perturbation (tracking) | +/-1%, +/-5% | k_f, k_m perturbation range |
| Parameter perturbation (estimation) | +/-30% | Initial guess error for IEKF |
| Solver | COBYLA (NLOPT) | Derivative-free constrained optimization |
| Integrator | dopri5 (SciPy) | Numerical trajectory integration |
| k_f (thrust coeff) | ~3.375e-4 N/s^2 | Hummingbird quadrotor nominal |
| k_m (drag coeff) | ~0.016 m | Hummingbird quadrotor nominal |
