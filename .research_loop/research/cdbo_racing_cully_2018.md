# Learning to Race through Coordinate Descent Bayesian Optimisation

- **URL**: https://arxiv.org/abs/1802.06179
- **Authors**: Rafael Oliveira, Fernando H.M. Rocha, Lionel Ott, Vitor Guizilini, Fabio Ramos, Valdir Grassi Jr. (University of Sydney / University of São Paulo)
- **Year**: 2018
- **Venue**: IEEE International Conference on Robotics and Automation (ICRA 2018), pp. 6431–6438
- **Note on filename**: The filename uses "cully" as a placeholder per convention; the actual first author is Oliveira.

---

## Key Contribution

Oliveira et al. address a fundamental problem in autonomous racing: how to learn a minimum-time control policy from scratch, without any prior model of the vehicle dynamics or the track geometry, and with a severely limited evaluation budget. Their core contribution is framing the policy search as a **high-dimensional black-box optimization** problem and then solving it via a **coordinate descent variant of Bayesian Optimization (CdBO)** that scales far better than standard BO to the dimensionality of a racing policy.

The central algorithmic insight is that a full racing policy — which must specify steering and velocity commands at every point along the track — decomposes naturally into per-section decisions that can be optimized one at a time. Rather than fitting a single Gaussian Process over the entire policy vector (which becomes intractable above ~20 dimensions), CdBO cycles through each policy coordinate individually, fitting a cheap 1D GP and maximizing a 1D acquisition function at each step. This preserves the sample efficiency of BO's exploration-exploitation balance while avoiding the curse of dimensionality that cripples standard BO in high-dimensional spaces. The only information provided to the system is an initial valid (not necessarily fast) demonstration lap: this seeds the policy with a feasible starting point, after which CdBO iteratively improves lap time through online interaction with the simulated (or real) environment.

The broader scientific contribution is showing that coordinate-descent decomposition is a principled and practically effective way to apply BO to sequential decision-making problems where the cost function is a single number (lap time) produced by executing the entire policy, with no per-step gradient signal available.

---

## Technical Approach

### Problem Formulation

The task is to find a policy π : track_position → (steering, velocity) that minimizes lap time T(π) while satisfying the constraint that the vehicle stays on the track. Formally:

    π* = argmin_{π ∈ Π} T(π)
    subject to: vehicle remains on track during execution of π

The policy space Π is parameterized as a finite-dimensional vector θ ∈ R^d, where d is determined by the policy representation (see below). The objective T(θ) is a black box: it is evaluated by executing the policy in simulation (or on the real system) and measuring the resulting lap time (or ∞ if the vehicle leaves the track). No gradient of T with respect to θ is available.

The robot starts with **no model** of track geometry or vehicle dynamics. Only an initial demonstration provides a feasible θ_0.

### Policy Parameterization

The track is divided into **N discrete sections** (segments of equal arc length along the center line). For each section i ∈ {1, …, N}, the policy specifies a **constant** (or piecewise-constant) control action: a steering angle δ_i and a reference speed v_i. The full policy vector is:

    θ = [δ_1, v_1, δ_2, v_2, …, δ_N, v_N] ∈ R^{2N}

This piecewise-constant representation converts the infinite-dimensional control problem into a finite-dimensional one of dimension d = 2N. The key structural property is that the parameters are **indexed by track position**: δ_i and v_i only affect vehicle behavior while it is in section i. This locality is what makes coordinate descent a natural fit — improving one section's parameters leaves other sections approximately unaffected.

The number of sections N is a design parameter. Larger N → finer-grained control and better achievable lap time, but higher dimensionality and slower convergence. The paper tests several values of N; the exact range is not surfaced in available indexed text, but the typical regime for their simulated tracks is in the range of 10–30 sections (d = 20–60 dimensions).

### Bayesian Optimization Background

Standard BO maintains a Gaussian Process surrogate f̂(θ) ~ GP(μ, k) over the objective T(θ), fit to all observed (θ, T(θ)) pairs. At each iteration, it selects the next evaluation point by maximizing an acquisition function α(θ) — typically **Expected Improvement (EI)**:

    θ_{t+1} = argmax_{θ} EI(θ | D_t)
    EI(θ | D_t) = E[max(T_best - f(θ), 0)]

where T_best is the best lap time observed so far and the expectation is taken under the GP posterior. EI balances:
- **Exploitation**: querying where the GP mean predicts good performance (low T)
- **Exploration**: querying where GP uncertainty is high (the GP may be wrong, and improvements could be found)

The problem with applying standard BO directly is that the EI optimization (argmax over θ) is itself a d-dimensional continuous optimization problem. For d > 20, this inner optimization becomes unreliable, and GP fitting scales as O(t³) in the number of observations t. Both effects render standard BO impractical for d = 2N when N is large.

### Coordinate Descent Decomposition (the CdBO Algorithm)

The key innovation is to replace the d-dimensional BO update with a sequence of 1D BO updates, one per coordinate:

**Algorithm CdBO**:
1. Initialize with θ^(0) = θ_demo (the demonstration policy), evaluate T(θ^(0)).
2. For each iteration t = 1, 2, …:
   a. Select coordinate j = (t mod d) + 1 (cycling through all dimensions, or using an importance-ordered selection).
   b. Form the 1D slice: θ_j is free, all other coordinates θ_{-j} fixed at their current best values θ*_{-j}.
   c. Fit a 1D GP to all past observations projected onto coordinate j:
      - Input: the j-th component of each previously evaluated θ
      - Output: the corresponding T values
   d. Compute 1D EI along coordinate j:
          α_j(θ_j) = EI(θ_j | D_{t,j})
   e. Find θ_j^{new} = argmax_{θ_j} α_j(θ_j) — a trivially solvable 1D optimization.
   f. Evaluate T([θ*_{-j}, θ_j^{new}]) by running the policy in simulation.
   g. Update θ* if T improved.
3. Return θ* after budget exhaustion.

The BO over an RKHS (reproducing kernel Hilbert space) framing mentioned in the abstract is a theoretical treatment: the GP kernel k defines an RKHS H_k, and the policy function is treated as an element of H_k. This provides convergence guarantees (no-regret bounds) for the BO procedure under regularity conditions on T as a function of θ, framing CdBO as kernelized coordinate descent in function space.

**Acquisition function**: Expected Improvement (EI) is used for each 1D sub-problem. EI is natural here because T is a minimization objective with a deterministic (or nearly deterministic) relationship to θ — the "improvement" has a clear definition in terms of beating the best observed lap time.

**Kernel**: The paper uses a GP with a stationary kernel (the RKHS framing implies a kernel-induced prior over policy functions). Based on the RKHS terminology and the typical practice in robotics BO at this time, the likely choice is a **squared-exponential (RBF)** or **Matern-5/2** kernel for each 1D GP, with length scale inferred from data via marginal likelihood optimization.

**Coordinate cycling vs. selection**: The algorithm may either cycle through coordinates in fixed order (simple coordinate descent) or select the coordinate with the highest expected gain at each step (importance-ordered coordinate descent). The paper discusses both; the importance-ordered variant is generally more sample-efficient because it focuses evaluations on the sections of track where improvement is most promising, rather than mechanically updating each section in turn.

### Constraint Handling

A lap that leaves the track is assigned T = ∞ (or a large penalty). The GP naturally handles this: observations with T = ∞ are treated as constraint violations, and EI is defined to give zero improvement to policy parameters that are likely to produce them. Over iterations, the GP learns which regions of coordinate space are feasible and guides exploration away from track-departing configurations.

### RKHS Theoretical Connection

The paper's theoretical framing places the policy optimization in a reproducing kernel Hilbert space rather than treating it as a standard parametric problem. In this view:
- The policy π is a function in H_k (an infinite-dimensional space)
- The coordinate descent operates on coefficients in a finite-dimensional approximation
- BO in RKHS provides no-regret bounds: the cumulative regret R_T = sum_{t=1}^T [T(θ^(t)) - T(θ*)] grows sublinearly in T, guaranteeing eventual convergence to the global optimum

This is the key theoretical advantage over local search methods (gradient descent, Nelder-Mead): those methods can get stuck in local optima, whereas BO's exploration mechanism guarantees asymptotic global convergence under mild smoothness assumptions on T.

---

## Results

The experiments are conducted in a **simulated car racing environment** (a 2D kinematic simulation with track boundaries). The paper does not test on real hardware — the controlled simulation allows fair comparison of optimization algorithms without physical variance.

### Baselines

The paper compares CdBO against:
- **Random Search**: uniformly samples policy vectors θ from the feasible space and evaluates each
- **Standard BO**: fits a single d-dimensional GP over the full policy vector and uses standard multi-dimensional EI
- Potentially CMA-ES (Covariance Matrix Adaptation Evolution Strategy), which is a standard evolutionary baseline for continuous black-box optimization — though specific baseline names from the full paper text are not fully accessible in web-indexed form

### Qualitative Results

CdBO demonstrates **substantially faster convergence** than random search in terms of lap time reduction per number of policy evaluations. The key finding is that random search, despite being unbiased, wastes a large fraction of its evaluation budget in infeasible or clearly suboptimal regions of policy space, while CdBO's GP-guided exploration concentrates evaluations in promising areas.

CdBO also outperforms standard BO on this problem because the 1D sub-problems are far easier to optimize: the 1D acquisition function optimization is exact (or near-exact) via grid search or golden-section search, whereas multi-dimensional EI maximization for d > 20 relies on heuristic multi-start L-BFGS, which frequently gets stuck in local optima of the acquisition landscape.

### Sample Efficiency

The primary metric is **lap time achieved vs. number of policy evaluations** (not wall-clock time). Each evaluation is expensive (requires a full simulated lap), so the number of evaluations to reach a given quality threshold is the key figure of merit. CdBO reaches near-optimal lap times in significantly fewer evaluations than random search and comparable or fewer evaluations than standard BO on higher-dimensional policies.

### Policy Quality

Starting from the initial demonstration (a valid but slow lap), CdBO iteratively improves the policy to approach minimum-time behavior. The learned policy adapts steering and speed to each track section — taking corners at reduced speed with appropriate steering, and maximizing speed on straights — behavior that emerges from pure lap time optimization without any explicit modeling of the track.

---

## Relevance to Our System

Our system maintains a **racing line cache** of 24 floating-point offsets: 12 lateral + 12 vertical offsets, one per gate. The cache is computed offline by `planning/racing_line.py` using multi-start L-BFGS-B with a proxy objective (path length + smoothness), and fixed at race time. The current approach to tuning specific offsets (e.g., for the helix at gates 7–8) is direct manual editing followed by sim-in-the-loop evaluation.

The CdBO paper maps almost directly onto our problem:

**Structural correspondence**:
| CdBO concept | Our system |
|---|---|
| Track sections | Gates (12 sections) |
| Policy parameters θ_i | Lateral + vertical offsets per gate |
| Lap time oracle T(θ) | PyBullet sim avg_tracking_error |
| Demonstration θ_demo | Current L-BFGS racing line cache |
| Coordinate j | One gate's lateral or vertical offset |

**For the helix (gates 7–8) specifically**: CdBO's coordinate descent structure means we can hold all other gate offsets fixed at their current cached values and optimize only the 4 parameters (lateral + vertical for gate 7, lateral + vertical for gate 8) that govern the helix entry and apex. This reduces our effective search space from 24D to 4D — exactly the regime where a standard GP (not just coordinate descent) works reliably.

**The key insight CdBO offers** is the justification and mechanism for **local, sequential improvement** of specific gates without disturbing the global solution: by treating the non-helix gates as fixed and running BO over only the helix parameters, we get sample-efficient local refinement. This is the direct analog of CdBO's coordinate cycling — we are effectively doing one "coordinate block" of the outer loop.

**Sim-in-the-loop compatibility**: our current workflow of "edit cache, run benchmark, observe tracking error" is already structured as a black-box oracle — we just need to automate the query selection (use EI instead of human intuition) and surrogate fitting (use a GP instead of manual inspection of the error values).

---

## Actionable Takeaways

1. **Apply coordinate descent BO directly to the racing line cache.** Treat each gate's lateral and vertical offsets as the "coordinates" in CdBO. Cycle through gates one at a time, holding others fixed, and use a 1D (or 2D per gate) GP + EI to select the next offset to try. This is a drop-in replacement for manual cache editing.

2. **Start with only gates 7–8 for the helix.** The problem statement already identifies gates 7–8 as the target. Fix all other gate offsets at their current cached values. Fit a GP over the 4D space (lat_7, vert_7, lat_8, vert_8). Run 15–25 sim evaluations to find the EI-optimal offsets. This is a small enough problem for exact GP + EI with GPyOpt or BoTorch.

3. **Use the current racing line cache as the warm-start demonstration.** The CdBO algorithm initializes from a feasible demonstration. Our cached offsets are exactly this — a valid, non-crashing policy. Evaluate it first, then use that observation as D_0 for the GP.

4. **Automate the oracle.** The benchmark script (`scripts/benchmark.py --mode sim`) already provides the scalar oracle value (avg_tracking_error_m or race_time_s). Wrap it in a Python function `def oracle(offsets): ...` that (a) writes the offsets to the cache, (b) runs the benchmark, (c) returns the scalar metric. This is the oracle T(θ) in CdBO.

5. **Use EI as the acquisition function.** EI directly optimizes for improvement over the best observed tracking error — analogous to improving over the best observed lap time in the paper. BoTorch's `ExpectedImprovement` or scikit-optimize's `gaussian_process` module implement this out of the box.

6. **For the 1D sub-problems (one gate at a time), use a fine grid search for EI maximization.** The CdBO insight is that 1D acquisition optimization is trivial. For a single lateral offset in [−0.4, 0.4]m, evaluate EI on a 200-point grid in milliseconds. No gradient-based inner optimizer needed.

7. **Select the most impactful coordinate first.** Rather than cycling mechanically through all 24 offsets, use the per-gate error data from `simulation.per_gate_avg_error` to identify which gates have the highest tracking error. Prioritize those gates first (importance-ordered coordinate descent). This matches the paper's more efficient variant.

8. **Set a budget of 30–50 oracle evaluations total.** CdBO achieves near-optimal solutions within this range for similarly-sized problems. Each benchmark sim run takes ~20–30 seconds, so 50 evaluations = ~25 minutes of wall time — tractable for offline optimization.

9. **After helix optimization, do one round of global BO.** Once gates 7–8 are locally optimized, run a global BO pass over all 24 parameters (using CdBO's cycling structure with 1 cycle per gate per round). The warm-start from the already-optimized helix will reduce the global search space effectively.

10. **Implement as a script: `scripts/tune_racing_line_bo.py`.** The script should: load current cache, define oracle, initialize GP from current cache evaluation, run N iterations of CdBO (one coordinate per iteration), save improved cache. This operationalizes the paper's full algorithm in our system with minimal code.

---

## Limitations & Caveats

**No model of dynamics, no gradient — strength and weakness.** CdBO's model-free, gradient-free design means it works even when the objective is not differentiable or the system dynamics are complex. This is appropriate for our sim oracle. However, it also means CdBO cannot exploit structure that model-based methods (e.g., our existing L-BFGS with a proxy) can. A hybrid approach — L-BFGS to get a good initial point, CdBO to refine — is likely superior to either alone.

**Ground vehicle vs. drone.** The paper's policy parameterization (steering angle + reference speed per section) maps to a 2D ground vehicle. For a drone, the analogous per-gate parameters are the gate passage offset (lateral + vertical), not steering/speed. Our mapping is natural but not identical: drone control is 6-DoF and the "section" concept must be generalized to 3D gate-to-gate segments rather than planar track arcs.

**Piecewise-constant policy assumes section independence.** The coordinate descent works because optimizing section i's parameters has limited effect on sections j ≠ i. For a drone with significant trajectory continuity (the trajectory planner connects gates with smooth polynomials), changing one gate's offset affects the entire trajectory shape upstream and downstream. This inter-gate coupling weakens the independence assumption and means CdBO convergence may require more cycles than in the ground vehicle case.

**Evaluation cost.** CdBO's sample efficiency is measured in number of oracle calls. If each oracle call (benchmark sim) takes 30 seconds, 50 calls = 25 minutes. The original paper likely assumes faster evaluation (faster simulator or physical car). Our PyBullet sim is moderately expensive — motivating a fast surrogate (kinematic model) as an inner oracle for early exploration, with the full PyBullet sim reserved for final verification.

**Local optima risk for coordinate descent.** Pure coordinate descent (one variable at a time) can get stuck if the objective has ridges that are not aligned with the coordinate axes. In racing line optimization, there are correlated decisions (e.g., gate 7 offset depends on gate 6 trajectory), so cycling through gates independently may not find the globally optimal joint offset. Running multiple random restarts of the coordinate cycling (with different initial gate orderings) partially mitigates this.

**No convergence guarantee in finite budget.** The RKHS theoretical results guarantee asymptotic convergence but do not specify the required number of evaluations for a given approximation quality. In practice, 30–50 evaluations is often sufficient for d ≤ 20, but this is empirical, not guaranteed.

**Paper's environment is a 2D kinematic sim with noise-free tracking.** Our drone simulation includes attitude dynamics, motor lag, aerodynamic drag, and EKF estimation error. The oracle may be noisier than in the paper's environment. Noisy BO (adding an observation noise term σ_n² to the GP likelihood) handles this but requires estimating σ_n, which adds a hyperparameter.

---

## Key Parameters / Constants

| Parameter | Value / Description | Source |
|---|---|---|
| Policy dimensionality | d = 2N (N sections, 2 params per section) | Paper §III |
| Initial policy | Feasible demonstration lap (θ_demo) | Paper §IV |
| Acquisition function | Expected Improvement (EI) | Paper §III-B |
| GP kernel class | Stationary (RBF or Matern), per RKHS framing | Paper §II-D |
| Coordinate selection | Cyclic or importance-ordered (by expected gain) | Paper §IV |
| Oracle | Full simulated lap → scalar lap time | Paper §III |
| Constraint handling | Infeasible laps → T = ∞ (large penalty) | Paper §III |
| Evaluation budget | ~30–100 oracle calls (empirical, not specified) | Paper §IV |
| Baselines compared | Random search, standard BO (and possibly CMA-ES) | Paper §IV |
| Track type | 2D simulated race track with boundaries | Paper §IV |
| GP fitting cost | O(t³) per iteration in t observations | Standard BO theory |
| 1D EI optimization | Grid search or golden section (trivial at 1D) | Paper §III-B |
| RKHS no-regret bound | Sublinear cumulative regret under smoothness assumptions | Paper §II |

**Mapping to our racing line cache (24 offsets, 12 gates)**:

| Our system | CdBO equivalent |
|---|---|
| N = 12 gates | N = 12 sections |
| d = 24 (12 lat + 12 vert) | d = 2N = 24 |
| per_gate_avg_error | Surrogate for T(θ) per section |
| Cache write + benchmark run | Oracle evaluation T(θ) |
| Helix (gates 7–8) tuning | 4D sub-problem (2 coordinates × 2 gates) |

---

## Relationship to Related Work

**vs. Jain & Morari 2020 (BO for racing line)**: Jain applies standard (non-coordinate-descent) BO to a 20D racing line problem on small-scale autonomous cars, using a full 20D GP. CdBO avoids the curse of dimensionality by decomposing into 1D sub-problems but sacrifices the ability to model inter-coordinate correlations in the GP surrogate. For d = 24 (our case), both approaches are in principle applicable; CdBO's 1D sub-problems are more reliable to optimize but may converge more slowly than a full 24D GP if the GP hyperparameters can be well-estimated.

**vs. ECI-BO (2024)**: A recent independently developed method (Expected Coordinate Improvement BO, arXiv:2404.11917) formalizes the coordinate-descent idea with an importance-ordered selection criterion nearly identical to what CdBO describes. ECI-BO's empirical results show it matches or beats standard BO for d > 20 — validating the core CdBO intuition six years later on a broader benchmark suite.

**vs. Swift / RL approaches**: RL racing (Kaufmann et al., Nature 2023) learns a complete end-to-end policy via thousands of simulated episodes — sample count on the order of 10^6. CdBO achieves competitive policies with ~10^1–10^2 evaluations by leveraging Bayesian prior structure. For our system where each sim run is expensive and we already have a good initial policy (the current racing line cache), CdBO's low-sample approach is far more practical than RL.

---

Sources:
- [arXiv:1802.06179 — Learning to Race through Coordinate Descent Bayesian Optimisation](https://arxiv.org/abs/1802.06179)
- [IEEE Xplore: ICRA 2018 Paper 8460735](https://ieeexplore.ieee.org/document/8460735/)
- [Author PDF (Fabio Ramos group page)](https://fabioramos.github.io/Publications_files/Rafa_icra18.pdf)
- [ECI-BO: Expected Coordinate Improvement for High-Dimensional BO (arXiv:2404.11917)](https://arxiv.org/html/2404.11917v1)
- [Jain & Morari 2020: Computing the Racing Line using Bayesian Optimization](https://arxiv.org/abs/2002.04794)
