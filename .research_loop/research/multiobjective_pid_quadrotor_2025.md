# Methods for Multi-objective Optimization PID Controller for Quadrotor UAVs

- **URL**: https://arxiv.org/abs/2509.17423
- **Authors**: Andrea Vaiuso, Gabriele Immordino, Ludovica Onofri, Giuliano Coppotelli, Marcello Righi
- **Year**: 2025
- **Venue**: arXiv:2509.17423 [eess.SY, cs.SY] — submitted September 22, 2025

---

## Key Contribution

This paper addresses the long-standing practical challenge of PID gain tuning for quadrotor UAVs, which remains sensitive to parameter selection and typically yields suboptimal trade-offs when done manually. The authors construct a unified, high-fidelity simulation platform — integrating blade element momentum (BEM) aerodynamics, a neural network surrogate for rotor efficiency, 6-DOF rigid body dynamics, Dryden turbulence, and a data-driven acoustic model — and then use it to benchmark three families of black-box optimizers (metaheuristics, Bayesian optimization, and deep reinforcement learning) against a multi-objective cost function that jointly penalizes tracking error, power consumption, and noise emissions.

The core contribution is twofold: (1) a structured multi-objective composite cost function with eight weighted sub-terms that captures competing real-world objectives, and (2) an empirical comparison showing that Grey Wolf Optimization (GWO) consistently dominates the other methods, achieving up to a 42.7% reduction in composite cost relative to a Ziegler-Nichols manual baseline while simultaneously cutting attitude oscillations by 77.8%, noise by 35.9%, and power by 25.7%. Crucially, the paper demonstrates that optimized gains generalize to unseen missions, validating the approach beyond the training trajectory.

---

## Technical Approach

### Controller Architecture

The quadrotor runs five nested PID control loops with 15 tunable parameters (Kp, Ki, Kd per loop group):

1. Horizontal position (x, y) — shared gains
2. Altitude (z)
3. Attitude (roll, pitch) — shared; yaw separate
4. Horizontal speed
5. Vertical speed

Standard robustness mechanisms are applied: anti-windup integral clamping, individual command saturation, and feed-forward hover compensation.

### Multi-objective Cost Function (Eq. 17)

```
J(P) = w_t·C_t + w_d·C_d + w_o·C_o + w_c·C_c + w_os·C_os + w_p·C_p + w_n·C_n + C_nm
```

Eight weighted sub-terms:

| Symbol | Meaning |
|--------|---------|
| C_t    | Total mission time |
| C_d    | Euclidean distance error at final position |
| C_o    | Attitude oscillation (sum of absolute roll/pitch changes) |
| C_to   | Thrust oscillation |
| C_c    | Completion penalty (binary, for failed missions) |
| C_os   | Overshoot past waypoints |
| C_p    | Total electrical energy consumed |
| C_n    | Noise: p-norm of broadband sound power level (SWL) vector + its maximum |

This is a scalar composite; weights encode designer priorities. No Pareto front is computed — the approach is a weighted-sum scalarization, which is simpler but faster to optimize with black-box methods.

### Simulation Infrastructure

**Aerodynamics:** Blade Element Momentum Theory (BEMT), with a 3-layer neural network surrogate (16-32-16 units) trained on 8,000 samples (RPM 0–4,000, freestream 1–20 m/s) achieving a 1,900× speedup at ~5.75% SMAPE validation error. Each simulation runs at 125 Hz for 150 seconds wall-clock (~15 s compute time).

**Turbulence:** Dryden stochastic model. Turbulent velocity components (u, v, w) are projected onto each rotor disk. Including turbulence during optimization improved controller generalization.

**Acoustic model:** A second DNN (3 fully connected layers, LeakyReLU) predicts 63 third-octave-band SWL values from flight state, trained on 147 samples (SMAPE 3.29%, RMSE 2.5). SPL at ground receivers is then computed via:

```
L_p(f,t) = L_w^total(f,t) − A_sp(d(t)) − Atm_abs(f)·d(t) + DI(f,θ(t))
```

where A_sp(d) = 10·log₁₀(4πd²) is free-field spherical spreading.

### Optimization Algorithms

**Metaheuristics (population-based):**
- **GWO (Grey Wolf Optimizer):** Encircling/hunting strategy with α, β, δ hierarchy guiding population updates.
- **PSO (Particle Swarm):** Velocity updates from personal best + global best with inertia χ and coefficients c₁, c₂.
- **GA (Genetic Algorithm):** Fitness-proportionate selection, arithmetic crossover, Gaussian mutation.

**Bayesian Optimization (BO):** Probabilistic GP surrogate + acquisition function. Sample-efficient but carries high per-iteration overhead due to surrogate fitting.

**Deep Reinforcement Learning:** SAC (stochastic, with entropy bonus) and TD3 (deterministic, with noise injection). The MDP is single-step (H=1), so the action is the PID gain vector and reward is −J(P). This collapses to reward regression rather than genuine temporal credit assignment.

All methods were given 6,000 function evaluations. Warm-starting (initializing from the Ziegler-Nichols baseline) and bounded search (tight parameter ranges) were explored as additional variants.

---

## Results

### Composite Cost (Training Mission, Table 4 equivalent)

| Method | Wide bounds, no warm start | Bounded + warm start, no turbulence | Bounded + warm start, turbulence |
|--------|---------------------------|--------------------------------------|----------------------------------|
| Manual (ZN baseline) | 218.2 | 218.2 | 218.2 |
| GWO    | 271.9 (worse at cold-start wide) | **167.6 (−23%)** | **159.1 (−27%)** |
| PSO    | 531.8 | 170.4 | 162.9 |
| GA     | 420.8 | 177.5 | 175.9 |
| BO     | 389.6 | 210.2 | 183.1 |

Note: GWO performs worse than baseline at wide cold-start, suggesting it is sensitive to initialization and search space extent — warm-starting is essential.

### Generalization to Unseen Missions (Table 6, GWO vs. Manual)

| Metric | Improvement |
|--------|-------------|
| Overall composite cost | −42.7% |
| Attitude oscillation | −77.8% |
| Noise emissions | −35.9% |
| Power consumption | −25.7% |
| Mission time | −13.6% |
| Overshoot | −2.5% |

### Acoustic Results (Table 7)

| Metric | Value |
|--------|-------|
| Average SPL at receivers | −4.16 dB |
| Average source SWL | −4.05 dB |
| Average electrical power | −14.8% |

A 3–5 dB perceptual reduction is considered significant for human listeners.

### Computational Overhead (Table 5)

| Method | Time per iteration | Overhead vs. baseline sim |
|--------|--------------------|--------------------------|
| Baseline sim | 15.0 s | — |
| GWO | 17.04 s | +13.6% |
| PSO | 17.37 s | +15.8% |
| GA | 17.66 s | +17.7% |
| SAC | 17.95 s | +19.6% |
| TD3 | 22.56 s | +50.4% |
| BO | 34.39 s | +102.6% |

GWO has the lowest overhead, making it the most budget-efficient choice at scale.

---

## Relevance to Our System

This paper is only tangentially relevant to the autonomous drone racing stack. Our system uses a geometric SE(3) tracker (Lee et al.) with fixed gains in `control/mpc_tracker.py`, not PID loops. Nevertheless, several aspects are worth extracting:

**Direct relevance — gain tuning methodology:** The paper's framework of defining a scalar composite cost function and running black-box optimization over controller parameters is directly applicable to tuning our `TrackerConfig` gains (position/attitude/rate Kp, Kd). We could formulate a similar J(P) using our simulation's avg_tracking_error_m, p95_tracking_error_m, and loop_hz as sub-terms, then apply GWO or BO to search over gain space.

**Racing line selection context (primary motivation):** The paper's core insight — that a well-designed multi-objective cost function can be optimized efficiently with GWO across 6,000 function evaluations in reasonable wall-clock time — is directly applicable to our racing line selection problem. We currently have 10 candidate racing lines from multi-start L-BFGS and want to select the best by running each through our kinematic simulator. The paper validates: (a) running multiple simulator evaluations per optimization step is feasible at low overhead, and (b) a scalar composite cost outperforms single-objective proxies. For our racing line selection, we should define a composite J = w1·avg_tracking_error + w2·race_time + w3·max_tracking_error and evaluate each of the 10 candidates against it — this is exactly the "few-evaluation" regime BO or GWO excels at.

**Module impact:**
- `control/mpc_tracker.py` — TrackerConfig gain optimization via composite cost
- `planning/racing_line.py` — racing line selection using simulator-evaluated composite J rather than proxy (path_length + curvature²)
- `scripts/benchmark.py` — already provides the evaluation oracle; just needs wrapping in an optimizer loop

**What does NOT apply:** The acoustic and power objectives (C_p, C_n) are irrelevant to racing — we optimize for speed and tracking precision, not noise or energy. The DJI Matrice 300 platform (5.2 kg, max 13.89 m/s horizontal) is far slower than racing drones (~30+ m/s), so gain values and absolute costs are not transferable.

---

## Actionable Takeaways

1. **Use GWO or PSO for gain optimization over TrackerConfig.** Both achieve ~13–16% overhead vs. bare simulation and consistently outperform GA and BO. Implement a simple GWO loop (population ~10–20 wolves, 50–100 iterations) that evaluates `benchmark.py --mode sim` and minimizes a composite cost of avg_tracking_error + max_tracking_error.

2. **Warm-start from current working gains.** GWO performs worse than the manual baseline at wide-bounds cold-start (271.9 vs. 218.2). Always initialize the population around the current `TrackerConfig` values with bounded perturbations (±30–50%).

3. **Include turbulence/noise during optimization to improve generalization.** Controllers tuned with turbulence present generalized better to unseen missions. In our case, this means varying the racing line slightly during optimization rather than tuning on a single fixed trajectory.

4. **Replace proxy racing line selection criterion with simulator-evaluated composite J.** The paper validates that optimizing a well-defined multi-objective cost directly outperforms surrogate proxies by large margins. For our 10 candidate racing lines, evaluate each with `benchmark.py --mode sim` and score by J = w1·avg_tracking_error + w2·race_time + w3·completion_penalty. Select the candidate with minimum J. This requires 10 sim runs (~10–20 seconds each = 100–200 s total), which is tractable.

5. **Use attitude oscillation as a tuning signal.** The paper's C_o term (sum of absolute roll/pitch changes) is easy to compute from our simulation telemetry and correlates with control aggression and motor wear. Add it as a secondary metric in our benchmark JSON output.

6. **Normalize sub-costs before weighting.** The paper's cost function mixes terms with very different magnitudes (position error in meters vs. acoustic SWL in dB). Normalize each sub-term by its baseline value before applying weights to avoid one term dominating.

7. **Do not use RL for this problem.** The paper's RL agents (SAC, TD3) underperform the manual baseline. The single-step MDP formulation collapses RL to reward regression, offering no advantage over direct black-box search. GWO is simpler, faster, and better.

8. **Set evaluation budget to ~100–200 function evaluations for racing line selection.** With 10 candidates, a single pass is trivially achievable. For gain tuning with 15 parameters, 200–500 evaluations (GWO with population 20, 10–25 iterations) is sufficient based on the paper's convergence behavior.

---

## Limitations & Caveats

**Platform mismatch:** The DJI Matrice 300 (5.2 kg, max 13.89 m/s) is a slow, heavy commercial drone. Racing drones are sub-1 kg and fly at 30+ m/s, with fundamentally different dynamics, aerodynamic regimes, and controller bandwidth requirements. The specific gain values, cost weights, and absolute improvement percentages from this paper cannot be transferred directly.

**Weighted-sum scalarization:** The paper does not compute a Pareto front. The weights w_t, w_d, etc. are design choices, not derived optimally. In a racing context, mission time and tracking error are in strong conflict — weighted-sum scalarization may hide Pareto-optimal solutions. For our setting, since we have a single primary objective (minimize race time subject to tracking error < 0.5m), constrained optimization or a hard-threshold formulation would be more principled.

**GWO cold-start failure:** GWO performed worse than the manual baseline at wide bounds without warm-starting. This suggests it is not a robust global optimizer — it relies on the initial population being near the optimum. For our gain tuning, if the current TrackerConfig is already well-tuned, GWO will refine it; if gains are far from optimal (e.g., after a major code change), GWO may not find the global optimum.

**No real-flight validation:** All results are simulation-only. The acoustic model is a DNN trained on 147 samples. Sim-to-real transfer is not addressed, which matters for a competition environment.

**RL framing is degenerate:** The single-step MDP with H=1 means SAC and TD3 have no temporal credit assignment advantage over bandit methods. The underperformance of RL here is an artifact of problem formulation, not a fundamental limitation of RL for drone control.

**6,000 evaluations at 15 s/sim = 25 hours of compute.** For our use case with 10 racing line candidates, this budget is unnecessary — but for gain optimization over 15 parameters, a realistic budget matters. Our `benchmark.py --mode sim --duration 20` likely runs faster than 15 s, so the overhead is lower.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| PID loops tuned | 5 (15 parameters) | Horizontal pos, altitude, attitude, h-speed, v-speed |
| Simulation frequency | 125 Hz | Control loop rate |
| Simulation duration | 150 s | Per training mission |
| Wall-clock per sim | ~15 s | With BEMT surrogate |
| Total budget | 6,000 evaluations | All optimizer comparisons |
| GWO overhead | +13.6% per iter | Lowest among all methods |
| BO overhead | +102.6% per iter | Highest, due to GP fitting |
| BEMT surrogate speedup | 1,900× | vs. full aerodynamic simulation |
| BEMT architecture | 3 layers (16-32-16) | Trained on 8,000 samples |
| BEMT validation SMAPE | ~5.75% | Acceptable for surrogate-in-the-loop |
| Acoustic DNN architecture | 3 FC layers, LeakyReLU | 63 third-octave bands output |
| Acoustic DNN SMAPE | 3.29% | Trained on 147 samples |
| Acoustic speedup | 27× | vs. polynomial model |
| ZN baseline cost | J = 218.2 | Reference point for all comparisons |
| GWO best cost (turbulence) | J = 159.1 | −27% vs. ZN |
| Unseen mission improvement | −42.7% | Overall composite cost |
| Oscillation reduction | −77.8% | Most dramatic sub-term improvement |
| Noise reduction | −35.9% / −4.16 dB SPL | Perceptually significant threshold: 3–5 dB |
| Power reduction | −25.7% (mission) / −14.8% (electrical) | |
| ZN tuning formula | Kp=0.6·Ku, Ki=Kp/(Tu/2), Kd=Kp·(Tu/8) | Classical starting point for warm init |
| Spherical spreading | A_sp(d) = 10·log₁₀(4πd²) | Acoustic propagation |
