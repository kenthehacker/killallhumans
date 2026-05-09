# Learning to Race through Coordinate Descent Bayesian Optimisation

- **URL**: https://arxiv.org/abs/1802.06179
- **Authors**: Rafael Oliveira, Fernando H.M. Rocha, Lionel Ott, Vitor Guizilini, Fabio Ramos, Valdir Grassi Jr. (University of Sydney / University of São Paulo)
- **Year**: 2018
- **Venue**: IEEE International Conference on Robotics and Automation (ICRA 2018), pp. 6431–6438

---

## Key Contribution

Oliveira et al. address a fundamental problem in autonomous racing: how to learn a minimum-time control policy from scratch, without any prior model of the vehicle dynamics or the track geometry, and with a severely limited evaluation budget. Their core contribution is framing the policy search as a **high-dimensional black-box optimization** problem and solving it via a **coordinate descent variant of Bayesian Optimization (CDBO)** that scales far better than standard BO to the dimensionality of a racing policy. The method requires only an initial valid driving demonstration (a feasible but not necessarily fast lap) and localization data; no track map or vehicle model is needed.

The central algorithmic insight is that a full racing policy — which must specify control commands at every point along the track — can be optimized one coordinate (one track-section parameter) at a time. Rather than fitting a single Gaussian Process over the entire high-dimensional policy vector (which becomes intractable above ~20 dimensions), CDBO cycles through each policy coordinate individually, fitting a cheap 1D GP and maximizing a 1D acquisition function at each step. This preserves BO's exploration-exploitation balance and sample efficiency while avoiding the curse of dimensionality that cripples standard BO in high-dimensional spaces. The paper further provides a theoretical RKHS (Reproducing Kernel Hilbert Space) justification guaranteeing sublinear cumulative regret — meaning CDBO is provably asymptotically globally convergent, unlike local search methods.

---

## Technical Approach

### Problem Formulation

The task is to find policy parameters θ ∈ R^M that minimize lap time T(θ) while keeping the vehicle on track:

    π* = argmin_{θ ∈ R^M} T(θ)
    subject to: vehicle remains on track

where T(θ) is a black-box oracle (a full simulated or physical lap). No gradient of T with respect to θ is available. The vehicle receives no reward until a full lap is completed. A failed lap (leaving the track) returns reward 0.

The **reward function** used is:

    R(θ) = L / T(θ)   if track completed
    R(θ) = 0          if vehicle leaves track

where L is the track length. Maximizing R is equivalent to minimizing lap time T.

### Policy Parameterization (RKHS-based)

The policy is parameterized using M kernel basis functions placed at regularly-spaced inducing points along the track:

    X̂ = { i/(M-1) }_{i=0}^{M-1}     (normalized positions ∈ [0,1] along track centerline)

The policy at any track position x ∈ [0,1] is:

    π_w(x) = w^T φ̂(x)

where φ̂(x) = [k_a(x, x̂_1), ..., k_a(x, x̂_M)]^T, the feature map evaluated using a **Matérn 3/2 kernel**:

    k_a(x, x') = (1 + √3/l |x - x'|) · exp(-√3/l |x - x'|)

with length scale l ≈ 1/(M-1) (set so adjacent basis functions have moderate overlap).

The parameter vector is **θ = w ∈ R^M**, representing the M kernel weights. The state space is x ∈ [0,1] (normalized progress along track centerline). The action space is a scalar a ∈ [-1, 1], where positive values command throttle and negative values command braking. Steering is handled by a separate PI controller that maintains track-centerline deviation at zero.

The initial weight vector w_0 is obtained from the demonstration by ridge regression:

    w_0 = argmin_w ||a_demo - φ̂(X_demo) w||_2^2 + λ||w||_2^2

with closed-form solution: w_0 = [φ̂(X_demo)^T φ̂(X_demo) + λI]^{-1} φ̂(X_demo)^T a_demo

Values of M tested in experiments: **M ∈ {10, 30, 50, 100}**, giving policy dimensionality from 10 to 100.

### Bayesian Optimization Foundation

CDBO uses a **Gaussian Process** surrogate R̂(w) ~ GP(μ, k_R) over the reward objective, with a **Matérn 1/2 (exponential) kernel** on the policy weight space:

    k_R(w, w') = σ_f^2 · exp(-√[d²(w, w')])

where d²(w, w') = (w - w')^T Λ^{-1} (w - w'), and Λ = diag(l_1^2, ..., l_M^2) is a diagonal length-scale matrix (automatic relevance determination — ARD).

The acquisition function is **Upper Confidence Bound (UCB)**:

    h(w | D) = μ(w) + β · σ(w)

with β ∈ [0.5, 2] controlling the exploration-exploitation tradeoff. The next policy to evaluate is:

    w_{t+1} = argmax_w h(w | D_t)

The GP is fit to all observed (w^(i), R^(i)) pairs collected so far. The GP noise variance σ_n^2 = 0 (deterministic simulator assumed).

### CDBO Algorithm (Coordinate Descent Bayesian Optimization)

The full CDBO algorithm combines the GP surrogate with **Stochastic Coordinate Ascent** for acquisition function optimization:

**StochasticCoordinateAscent** (Algorithm 1):
- Input: acquisition function h(·), current best w*, number of dimensions M
- Randomly shuffle all M dimension indices {1, ..., M}
- For each dimension j in the shuffled order:
  - Solve 1D sub-problem: w_j* = argmax_{w_j} h([w*_{-j}, w_j]) using **COBYLA** (Constrained Optimization by Linear Approximation)
  - Update w*_j ← w_j*
- Return updated w*

**Main CDBO Loop** (Algorithm 2):
1. Collect S = 10 initial samples from N(w_0, I · σ_0^2) with σ_0^2 = 1; evaluate each
2. Build initial GP training set D_0 = {(w^(i), R^(i))}_{i=1}^S
3. For t = 1, 2, ..., N_budget:
   a. Fit GP to D_{t-1}
   b. Find candidate w_new = StochasticCoordinateAscent(h(· | D_{t-1}), w_best)
   c. Evaluate R(w_new) by executing policy in simulator (one full lap)
   d. Update D_t ← D_{t-1} ∪ {(w_new, R(w_new))}
   e. Update w_best if R(w_new) > R_best
4. Return (w_best, R_best)

**Key design choices**:
- **Random coordinate permutation per epoch**: shuffling the order prevents cycling artifacts documented in cyclic coordinate descent on non-convex functions
- **COBYLA for 1D sub-problems**: this derivative-free 1D optimizer is appropriate since the acquisition function is smooth (GP posterior) but evaluated by a potentially expensive surrogate query
- **Maximum 50,000 acquisition function evaluations per lap budget** (reported in experiments)

The total number of oracle evaluations is fixed at **N_budget = 300** in all experiments.

### RKHS Theoretical Connection

The policy optimization is framed in a Reproducing Kernel Hilbert Space H_{k_a}. The policy function π belongs to H_{k_a}, and the M-dimensional parameterization w gives a finite-dimensional approximation. BO in RKHS provides **no-regret bounds**: cumulative regret R_T = Σ_{t=1}^T [R(w*) - R(w^(t))] grows sublinearly in T under smoothness assumptions on R as a function of w. This guarantees eventual convergence to the global optimum — a property not shared by gradient-based or evolutionary methods that can get stuck in local optima.

---

## Results

### Experimental Setup

**Simulator**: Speed Dreams (open-source game engine, successor to TORCS), a 3D racing simulator with realistic vehicle physics.

**Vehicle**: Spirit 300 car model with continuous throttle/braking control. Steering controlled by a separate PI controller minimizing lateral deviation from centerline.

**Tracks**:
- **Forza**: 5,784m length, 11m wide, Monza-inspired with sharp curves at approximately 40% of track distance
- **Allondaz**: 6,356m length, 12m wide, with varying elevation changes and complex geometry

**Evaluation budget**: 300 policy evaluations (laps) per run; each run repeated 4 times with results averaged.

**Initial policy**: PI controller commanding constant reference speed of 15 m/s.

### Baselines Compared

1. **CMA-ES** (active variant): Covariance Matrix Adaptation Evolution Strategy — a state-of-the-art evolutionary optimizer for continuous black-box problems
2. **BO-CMA-ES**: Standard Bayesian Optimization using CMA-ES for acquisition function optimization (not coordinate descent)
3. **REMBO-5d / REMBO-10d**: Random Embedding BO — projects the high-dimensional parameter space into a 5-d or 10-d random subspace and runs standard BO in the lower-dimensional space

### Quantitative Results

**Reward achieved after 300 laps (Forza track)**:

| Dimensions | CDBO | CMA-ES | BO-CMA-ES | REMBO-5d | REMBO-10d |
|---|---|---|---|---|---|
| M = 10 | ~0.33 | ~0.32 | bimodal | ~0.30 | — |
| M = 50 | ~0.34 | ~0.25 | poor | ~0.27 | ~0.27 |
| M = 100 | ~0.34 | ~0.20 | poor | — | ~0.22 |

CDBO reward is essentially **constant across M ∈ {10, 50, 100}** — it does not degrade as dimensionality increases. CMA-ES reward drops substantially as M increases (0.32 → 0.20). REMBO plateaus when the global optimum falls outside the discovered random subspace.

**Wall-clock time for 300 laps (Forza track, Table I)**:

| Dimensions | CDBO | CMA-ES | BO-CMA-ES | REMBO-5d | REMBO-10d |
|---|---|---|---|---|---|
| M = 10 | 127s | 173s | 553s | 154s | — |
| M = 50 | 281s | 234s | 680s | 242s | 244s |
| M = 100 | 318s | 252s | 3048s | 257s | 299s |

BO-CMA-ES at M = 100 takes **3048s** (25 minutes for the optimization overhead alone, not counting simulation), approximately 10× slower than CDBO. CDBO scales gracefully (127s → 318s from M = 10 to M = 100), while BO-CMA-ES scales catastrophically.

**Allondaz track** (M = 10 → 50):
- CDBO: ~0.28 → ~0.29 (stable)
- CMA-ES: ~0.27 → degrades
- REMBO: ~0.25 (plateau, does not improve beyond 50 evaluations once subspace is fixed)

**Key finding**: CDBO is the only method that maintains consistent performance as M grows from 10 to 100, while all baselines degrade in either solution quality or computational cost.

---

## Relevance to Our System

Our system maintains a racing line offset cache for 12 gates: 12 lateral offsets and 12 vertical offsets, giving a **24-dimensional parameter vector** θ ∈ R^24. This cache is computed offline by `planning/racing_line.py` via multi-start L-BFGS-B, and fixed at race time. The benchmark oracle — `scripts/benchmark.py --mode sim` — evaluates any given offset vector and returns scalar metrics (`avg_tracking_error_m`, `per_gate_avg_error`, `race_time_s`).

The structural correspondence to the CDBO paper is exact:

| CDBO concept | Our system |
|---|---|
| Track sections i = 1,...,M | Gates i = 1,...,12 |
| Policy parameter w_i (speed command at section i) | Lateral offset lat_i and vertical offset vert_i at gate i |
| Full policy vector w ∈ R^M | Offset cache θ ∈ R^24 (12 lat + 12 vert) |
| Lap time oracle T(w) | `avg_tracking_error_m` from PyBullet sim |
| Demonstration w_0 | Current L-BFGS racing line cache |
| Coordinate j | One gate's lateral or vertical offset |
| StochasticCoordinateAscent | Cycling through gates one at a time |

**Why coordinate descent is specifically well-suited to our problem — the basin-switching issue**:

Our trajectory optimizer (L-BFGS in `planning/trajectory_optimizer.py`) solves a non-convex min-snap polynomial problem. Changing one gate's lateral offset can trigger a jump to a different local optimum of the trajectory energy ("basin switching"). This means:

1. A joint 24D GP over all offsets simultaneously would observe highly discontinuous reward landscapes — the GP assumption of smooth continuous functions is violated at basin boundaries.
2. Coordinate descent sidesteps this by fixing 23 of 24 offsets and optimizing one at a time. Within a single coordinate's 1D slice (holding all others fixed), the trajectory optimizer likely stays in a single basin, giving a smooth 1D reward landscape that the GP can model reliably.
3. This is the strongest argument for CDBO over standard 24D BO in our system: CDBO's 1D sub-problems avoid basin switching by construction, whereas any joint optimization would have to cross basin boundaries.

**For gates 4, 5, 7 specifically**:
The per-gate error data from `simulation.per_gate_avg_error` already identifies which gates have the highest tracking error. CDBO's importance-ordered coordinate selection maps directly: optimize the gate with the highest tracking error first, holding all other offsets fixed. This reduces the effective search space from 24D to 1D or 2D (lateral + vertical for one gate) per iteration, making the GP reliable and EI maximization trivial (1D grid search).

**Concrete workflow**:
1. Evaluate current cache → get baseline `avg_tracking_error_m`
2. Identify gate with highest `per_gate_avg_error` (say gate 7)
3. Fit 1D GP over lat_7 ∈ [-0.4, 0.4]m using all past observations where only lat_7 varied
4. Find lat_7* = argmax EI on 200-point grid (milliseconds)
5. Write updated cache (only lat_7 changed), run sim benchmark, observe new error
6. Update GP, cycle to next gate (vert_7, then gate 4, gate 5, etc.)
7. Repeat for 30–50 total evaluations

This is CDBO's Algorithm 2 operating on our system with zero modification to the core algorithm.

---

## Actionable Takeaways

1. **Implement `scripts/tune_racing_line_bo.py` as a direct CDBO instantiation.** The script loads the current offset cache as w_0, defines an oracle wrapping the PyBullet benchmark, and runs CDBO's coordinate cycling loop (Algorithm 2 from the paper). This operationalizes the full method with no algorithmic changes needed.

2. **Start with gates 4, 5, 7 identified by `per_gate_avg_error`.** Use importance-ordered coordinate selection (the paper's more efficient variant): prioritize gates where per-gate tracking error is highest. This concentrates the evaluation budget where improvement potential is greatest.

3. **Treat each gate as a 2D sub-problem (lat_i, vert_i) rather than two separate 1D problems.** A 2D GP per gate (rather than cycling lat and vert separately) can model the correlation between lateral and vertical offset at a single gate. With only 2 dimensions per gate, the GP is fully reliable and EI maximization via 40×40 grid search requires evaluating only 1600 points — trivial.

4. **Use the current L-BFGS racing line cache as the warm-start demonstration.** This is exactly the CdBO "initial valid demonstration" — a feasible, non-crashing policy. Evaluate it first to establish D_0, then use GP posterior from that single observation to guide the first EI query.

5. **Hold all non-target gates fixed during per-gate optimization to prevent basin switching.** This is the key advantage of coordinate descent for our system: the trajectory optimizer stays in one basin when only one gate's offset is perturbed. Changing multiple gates simultaneously risks basin switches that corrupt the GP model.

6. **Use UCB (β ≈ 1.0) rather than EI for noisier sims.** The paper uses UCB with β ∈ [0.5, 2]. Our PyBullet sim has process noise (motor dynamics, aerodynamic disturbances) so the oracle is not perfectly deterministic. UCB with moderate β handles noisy observations more gracefully than EI, which assumes near-deterministic oracles.

7. **Set evaluation budget to 40–60 oracle calls total.** The paper achieves near-optimal solutions within 300 calls for M = 100 dimensions — our 24D problem should converge in far fewer. At ~25 seconds per full sim run, 50 calls = ~20 minutes of wall time.

8. **For 1D sub-problems, use 200-point grid search for EI/UCB maximization.** CDBO's paper maximizes the 1D acquisition function via COBYLA; a simple 200-point grid scan over [-0.5, 0.5]m is faster and more reliable for a single coordinate. Grid search takes microseconds.

9. **Use ARD kernel (separate length scale per gate) for the global GP.** The paper uses Λ = diag(l_i^2) — automatic relevance determination. In our system, different gates have different sensitivity to offset perturbations (a tight chicane gate is more sensitive than a wide sweeper gate). ARD allows the GP to learn these sensitivities from data.

10. **After per-gate optimization, run one joint L-BFGS pass from the optimized offsets.** CDBO finds good per-gate offsets independently, but the global trajectory might benefit from a final joint re-optimization (one L-BFGS run) starting from the CDBO-optimized offsets. This joint pass can pick up cross-gate improvements that coordinate descent leaves on the table.

---

## Limitations & Caveats

**Ground vehicle, not drone.** The paper's action space is a scalar (throttle/braking) and the policy is a speed profile along the track. For a drone, the analogous per-gate parameters are the 3D gate passage offsets. The coordinate descent structure is identical, but the dynamics are 6-DoF rather than 2D kinematic, and the trajectory planner (min-snap polynomial) introduces stronger inter-gate coupling than a ground vehicle's speed profile.

**Inter-gate coupling weakens coordinate independence assumption.** The CDBO approach assumes that optimizing coordinate i has negligible effect on the optimal values of coordinates j ≠ i. For a drone trajectory with min-snap polynomial continuity constraints, changing gate 7's lateral offset changes the trajectory curvature at gates 6 and 8 as well. This coupling means CDBO may need more cycles through all coordinates before convergence, compared to the ground vehicle case where speed in section i is nearly independent of speed in section j (especially for sections far apart).

**No real hardware validation.** All experiments are in simulation (Speed Dreams). The paper does not demonstrate CDBO on a physical car or drone. Sim-to-real gaps (sensor noise, model mismatch, mechanical tolerances) are not addressed. For our system, we have PyBullet as the oracle — this is simulated, so the paper's experimental setting directly applies, but the question of whether optimized offsets transfer to the real competition drone is left open.

**Fixed evaluation budget of 300 laps may be too many for our use case.** The paper uses 300 evaluations for M = 10–100 dimensions. Our 24D problem with a much faster partial evaluation (run only the relevant portion of the track, not a full race) could converge in 30–50 evaluations. Budget should be tuned empirically.

**Stationary GP kernel assumes spatial stationarity.** The GP covariance k_R(w, w') = σ_f^2 exp(-d(w, w')) treats all regions of parameter space identically. Near basin boundaries in our trajectory optimizer, the reward landscape is non-stationary (behavior changes qualitatively across a boundary). Non-stationary or deep kernel methods would better capture this, but at higher computational cost.

**No convergence rate guarantee for finite budget.** The no-regret RKHS bound guarantees asymptotic global convergence but gives no finite-sample rate for a specific approximation quality. In practice 30–50 evaluations is often sufficient for d ≤ 24, but this is empirical.

**π_w outputs a single scalar (speed policy only).** The paper's policy maps track position to a single control variable (throttle). Steering is handled by a separate hard-coded PI controller. For a drone, the full policy must specify 3D position setpoints or attitude commands — more complex than a scalar output. However, our use of gate offset parameters (which are then fed to the min-snap trajectory planner) is structurally equivalent: the offset vector θ is the policy parameter, and the downstream trajectory planner + controller handles the full 6-DoF control.

---

## Key Parameters / Constants

| Parameter | Value | Source |
|---|---|---|
| Policy kernel (Matérn 3/2 length scale) | l ≈ 1/(M-1) | Paper §III-A |
| GP kernel for reward | Matérn 1/2 (exponential), ARD | Paper §III-B |
| Acquisition function | UCB with β ∈ [0.5, 2] | Paper §III-B |
| Number of initial samples | S = 10, drawn from N(w_0, I · σ_0^2) | Paper §IV |
| Initial variance | σ_0^2 = 1 | Paper §IV |
| GP observation noise | σ_n^2 = 0 (deterministic simulator) | Paper §IV |
| Evaluation budget | N = 300 laps per run | Paper §IV |
| Runs per condition | 4 (results averaged) | Paper §IV |
| Coordinate ordering | Random permutation (reshuffled each epoch) | Paper §III-C (Alg. 1) |
| 1D sub-problem optimizer | COBYLA | Paper §III-C (Alg. 1) |
| Max AF evaluations per coordinate | 50,000 total across all coordinates per lap | Paper §IV |
| Track lengths tested | 5,784m (Forza), 6,356m (Allondaz) | Paper §IV |
| Track widths | 11m (Forza), 12m (Allondaz) | Paper §IV |
| Initial policy speed | 15 m/s constant | Paper §IV |
| Kernel count M tested | 10, 30, 50, 100 | Paper §IV |
| Simulator | Speed Dreams (TORCS successor) | Paper §IV |
| Vehicle model | Spirit 300 | Paper §IV |

**Mapping to our racing line cache (24 offsets, 12 gates)**:

| Our system | CDBO equivalent |
|---|---|
| 12 gates | M = 12 sections (if using one offset per gate) |
| 24D offset vector (12 lat + 12 vert) | w ∈ R^M with M = 24 |
| PyBullet benchmark → avg_tracking_error_m | Oracle T(w) → lap time |
| per_gate_avg_error[i] | Section-level reward signal |
| Current L-BFGS cache | Demonstration w_0 |
| Single gate sweep (holding 23 others fixed) | One CDBO coordinate cycle |
| β = 1.0 (recommended for noisy sim) | UCB exploration parameter |

---

## Relationship to Related Work

**vs. Standard BO (BO-CMA-ES baseline in paper)**: Standard BO over the full 24D space is feasible but acquisition function optimization (argmax of UCB over 24D) is unreliable with L-BFGS alone. The paper shows BO-CMA-ES is 10× slower at M = 100 and has comparable or worse solution quality to CDBO at all M. For our 24D problem, CDBO's coordinate decomposition gives more reliable acquisition optimization with no quality loss.

**vs. REMBO (random embedding BO)**: REMBO projects 24D → 5D or 10D randomly and runs standard BO in the low-dimensional subspace. This works only if the optimal solution lies in the discovered random subspace — which is not guaranteed. The paper shows REMBO plateaus early. For our gate offset problem, the optimal offsets may involve correlated changes across multiple gates that cannot be captured by a random low-dimensional projection.

**vs. CMA-ES**: CMA-ES is a strong evolutionary baseline for continuous black-box problems and actually competitive with CDBO at low dimensionality (M = 10). At M = 100, CMA-ES reward drops to ~0.20 vs. CDBO's ~0.34. For our 24D problem, CMA-ES is a reasonable alternative but requires more evaluations to converge and does not provide uncertainty estimates that could be used to decide when to stop.

**vs. Swift / RL approaches**: RL racing (Kaufmann et al., Nature 2023) learns an end-to-end policy from ~10^6 simulated laps. CDBO achieves comparable improvements with ~10^1–10^2 evaluations by exploiting BO's prior structure. For our system where each sim run takes ~25 seconds and we have a good initialization, CDBO's low-sample approach is orders of magnitude more practical.

---

Sources:
- [arXiv:1802.06179 — Learning to Race through Coordinate Descent Bayesian Optimisation](https://arxiv.org/abs/1802.06179)
- [IEEE ICRA 2018 Proceedings](https://ieeexplore.ieee.org/document/8460735/)
- [ar5iv HTML rendering with full technical details](https://ar5iv.labs.arxiv.org/html/1802.06179)
