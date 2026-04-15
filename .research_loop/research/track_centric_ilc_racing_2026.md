# Track-centric Iterative Learning for Global Trajectory Optimization in Autonomous Racing

- **URL**: https://arxiv.org/abs/2601.21027
- **Authors**: Youngim Nam, Jungbin Kim, Kyungtae Kang, Cheolhyeon Kwon
- **Year**: 2026
- **Venue**: arXiv preprint (cs.RO — Robotics), submitted January 28, 2026

---

## Key Contribution

This paper addresses a fundamental gap in autonomous racing: the disconnect between trajectory *planning* (what path to follow) and trajectory *tracking* (how well the vehicle actually follows it). Most ILC work in racing focuses on tracking-level corrections — reducing the deviation from a fixed reference. This paper argues that fixing the reference trajectory and only correcting tracking misses the bigger opportunity: the *reference itself* may be suboptimal given real vehicle dynamics. The authors propose a **track-centric iterative learning** framework that uses accumulated lap data to iteratively re-optimize the full-horizon reference trajectory, not just the tracking controller feedforward.

The three core contributions are:

1. A **wavelet-based trajectory parameterization** that compresses the high-dimensional trajectory space into a low-dimensional (Nθ = 10) optimization variable, enabling tractable Bayesian optimization over full lap trajectories.

2. A **Gaussian Process residual dynamics model** that learns the gap between the nominal vehicle model and actual behavior, enabling increasingly accurate closed-loop simulation for evaluating candidate trajectories without repeated hardware trials.

3. An **asymptotic optimality guarantee** (Theorem 1) showing that as the dynamics model converges, the iteratively refined trajectory converges to within δ(NBO) of the true optimum — a suboptimality bound determined only by the Bayesian optimization budget, not by dynamics uncertainty.

The headline result: up to **20.7% lap time reduction** over a nominal (no-learning) baseline, consistently outperforming methods that apply learning only at the tracking level.

---

## Technical Approach

### Problem Formulation

The vehicle state is expressed in the **Frenet frame** (arc-length coordinates along the track centerline):

```
x := [s, ey, eψ, vx, vy, w]
```

where s is arc-length progress, ey is lateral deviation, eψ is heading error, vx/vy are longitudinal/lateral velocities, and w is yaw rate. This track-centric representation decouples the path-following geometry from global position, enabling the optimization to focus on what actually determines lap time: lateral placement and speed profile around the track.

The dynamics follow a nonlinear bicycle model with Pacejka tire forces. A residual term g(z) accounts for model mismatch:

```
ẋ = f(x, u) + Bg · g(z)
```

where z := [vx, vy, w, a, δ] are the tire-relevant states and g is learned via Gaussian Process regression.

### Wavelet-Based Trajectory Parameterization

Rather than optimizing waypoints, spline knots, or polynomial coefficients directly, the method represents the lateral deviation profile ey(s) and velocity profile vx(s) — both functions of arc-length — using **Discrete Wavelet Transform (DWT)** decomposition with Daubechies-4 (db4) wavelets at decomposition level L = 6.

The key insight is that lap time is dominated by the *global trend* of the lateral and speed profiles (which corners to apex early, which to sacrifice entry for exit speed), not fine-grained local oscillations. Accordingly, only the **coarsest-level approximation coefficients** are used as optimization variables — 5 coefficients for ey(s) and 5 for vx(s), giving Nθ = 10 total. Detail coefficients (high-frequency content) remain fixed at initialization.

The advantages over spline parameterization:
- Wavelets provide multiresolution decomposition — the same Nθ = 10 variables describe global structure adaptively rather than being tied to fixed knot locations
- Near apex regions (where curvature is highest and lateral freedom is most constrained), wavelets outperform cubic splines in capturing the optimal lateral displacement
- Inverse DWT reconstruction at Ns = 256 arc-length segments is O(Ns log Ns), making repeated evaluations fast

### Gaussian Process Residual Dynamics

A sparse GP with 200 inducing points and RBF kernel learns the residual dynamics g(z) from accumulated data D_g collected across real-hardware (or high-fidelity simulation) laps. The sparse approximation maintains computational tractability as the dataset grows across iterations.

The GP serves as a surrogate for the true dynamics mismatch: as more laps are collected, the GP posterior becomes more accurate, reducing the gap between simulated lap time J_g^j(θ) and true lap time J_g*(θ).

### Bayesian Optimization Framework

Given the low-dimensional parameter vector θ ∈ ℝ^10, global trajectory optimization is formulated as:

```
θ* = argmin J(θ)
```

where J(θ) is the total lap time evaluated by running closed-loop MPC tracking in simulation using the current GP dynamics model, then reconstructing the trajectory from θ via inverse DWT.

A GP surrogate model Ĵ(θ) is maintained over the BO search space, with the **Lower Confidence Bound (LCB)** acquisition function:

```
α(θ) = μn(θ) - β^(1/2) · σn(θ)
```

This balances exploitation (low mean lap time) with exploration (high uncertainty regions). The BO runs for NBO = 70 evaluations per iteration, each evaluation consisting of a full closed-loop simulation. Because the surrogate is cheap to query and the GP dynamics model makes simulation fast, 70 evaluations are tractable per iteration.

### The Three-Phase Iterative Learning Cycle

At each iteration j:

1. **Dynamics Update**: Train GP on the accumulated dataset D_g^(j-1) from all prior real-hardware laps.

2. **Trajectory Optimization**: Run BO with NBO = 70 evaluations using the updated GP dynamics model as the simulation oracle. Output: optimized θ^j → reconstructed trajectory τ^j.

3. **Data Collection**: Deploy τ^j on real hardware. Collect new dynamics data D_g^new. Update dataset: D_g^j ← D_g^(j-1) ∪ D_g^new.

This outer loop is iterated until convergence (observed at ~10 iterations in experiments).

### Theoretical Guarantees

**Proposition 1** (Lap Time Evaluation Error): The error between simulated and true lap time is bounded:

```
|J_g^j(θ) - J_g*(θ)| ≤ C^j · ε^j
```

where ε^j = ‖g^j - g*‖∞ is the GP approximation error and C^j is a propagation constant derived from linearizing state deviation dynamics across Ns arc-length steps. As j → ∞ and more data is collected, ε^j → 0.

**Theorem 1** (Asymptotic Optimality):

```
lim sup(j→∞) J_g*(θ^j) ≤ J* + δ(NBO)
```

The true lap time achieved by the learned trajectory approaches the true optimum J* up to δ(NBO), the suboptimality bound from BO with NBO evaluations. This bound improves with more BO evaluations but is independent of iteration count — it represents the fundamental limit of the BO search over the Nθ = 10 dimensional space.

---

## Results

### Simulation (15 Scenarios)

The 15 scenarios perturb the nominal tire model parameters:
- Tire stiffness B* ∈ [1.1, 1.3] (nominal B = 1.3)
- Shape factor C* ∈ [1.3, 1.5] (nominal C = 1.5)
- Friction coefficient μ* ∈ [0.8, 1.2] (nominal μ = 1.2)
- Yaw inertia Iz* ∈ [0.014, 0.024] kg·m² (nominal Iz = 0.024)

Vehicle parameters: mass m = 3.0 kg, wheelbase Lf = Lr = 0.14 m (1/10-scale platform).

Comparison at iteration 10 (example scenario):
| Method | Lap Time (s) |
|---|---|
| Nominal (no learning) | 11.2 |
| GP-Track (GP in MPC only, fixed ref) | 9.8 |
| GP-Opt+Track (GP in opt and tracking) | 9.55 |
| **Proposed (track-centric ILC)** | **8.75** |

Across all 15 scenarios, average improvement over nominal baseline: **20.7%**. The proposed method consistently beats all ablations including GP-Opt+Track, which uses the GP in the optimizer but with a spline parameterization and no iterative outer loop.

Ablation findings:
- **Wavelet vs. spline**: Wavelet parameterization provides ~1–3% additional lap time reduction vs. cubic splines, concentrated near apex regions where curvature constrains lateral freedom most
- **Iterative vs. non-iterative**: Running the full 3-phase cycle vs. a single-shot optimization with equivalent total data yields meaningful improvement, confirming that the iterative dynamics refinement (not just data quantity) drives convergence

### Hardware Experiments (1/10-Scale Platform)

Setup: Intel NUC onboard, ROS framework, indoor localization, FORCESPRO MPC solver (50 ms sampling). Two controller configurations tested — well-tuned MPC and deliberately poorly-tuned MPC weights.

Results:
- Consistent lap time reduction across all iterations for both controller configurations
- Convergence to a stable minimum by iteration ~10
- Gap between BO-predicted lap time (TBO) and actual tracked lap time (Treal) narrows monotonically as GP dynamics improves — confirming the theoretical bound in Proposition 1
- Performance robust to controller tuning quality: the framework adapts to whatever tracking controller is used, since trajectory quality is evaluated via real hardware trials

---

## Relevance to Our System

Our system uses min-snap polynomial trajectories, TOPP-RA speed retiming, an ILC layer for tracking error reduction, and post-optimization inflation factors on gate margins. We are progressively reducing inflation factors as ILC improves tracking accuracy. The key question: how quickly and safely can we reduce margins?

This paper directly addresses the complementary problem. Our current ILC operates at the **tracking level** — it learns feedforward corrections to reduce deviation from a fixed reference trajectory. The track-centric framework operates at the **planning level** — it uses accumulated tracking performance data to iteratively re-optimize the reference itself.

**Critical insight for our system**: reducing inflation factors faster requires confidence that the drone will stay within tighter corridors. That confidence comes from two sources: (1) better tracking (ILC's job), and (2) a better reference that accounts for how the drone actually flies (this paper's job). Our current approach handles (1) but not (2). The reference trajectories are generated once by the min-snap optimizer with fixed constraints, without feedback from actual flight performance. If the drone consistently cuts a corner differently than planned, the ILC corrects for the deviation but the reference itself never adapts.

Concretely applicable elements:
- **Frenet-frame (arc-length) parameterization** maps cleanly onto our gate-sequenced racing. Our gates define discrete waypoints; the track-centric representation continuously parameterizes the corridor between gates. This could replace or augment our per-gate waypoint representation.
- **Wavelet compression of trajectory space** (Nθ = 10) is striking. Our min-snap trajectories have many more degrees of freedom (polynomial coefficients per segment × number of segments). If the global trend dominates lap time, our optimizer may be over-parameterized for the available data, making Bayesian optimization over the full space intractable. Compressing to wavelet coefficients makes BO tractable.
- **GP residual dynamics** addresses the model mismatch we see between our simulated drone and real flight behavior. Our EKF handles state estimation uncertainty, but the trajectory optimizer uses a fixed aerodynamic model. GP-learned residuals would close this loop.
- **Outer iterative cycle** provides a principled framework for our inflation factor reduction: after each simulated lap, update the dynamics model and re-optimize the trajectory. As the optimized trajectory converges toward what the drone actually does, the gap between planned and flown paths shrinks — which is precisely the condition that justifies reducing inflation factors. Rather than manually deciding when to reduce margins, the convergence of TBO → Treal gives an objective criterion.
- **Asymptotic optimality theorem** is operationally useful: it tells us that after sufficient iterations, the only remaining suboptimality is from the BO search budget (δ(NBO)), not from dynamics uncertainty. This sets a principled stopping criterion — once |TBO - Treal| < ε, further iteration is bounded by BO resolution, not model error.

For the VQ1 deadline, the most immediately actionable element is the **outer loop architecture**: run the drone on the track (in sim or hardware), collect tracking residuals, update a learned dynamics model, re-run trajectory optimization with that model, repeat. This is distinct from our current ILC which only updates feedforward inputs — this updates the reference trajectory itself.

---

## Actionable Takeaways

1. **Separate reference optimization from tracking correction**. Our ILC currently reduces tracking error around a fixed reference. Add an outer loop that re-optimizes the reference itself using accumulated performance data. These two loops operate at different timescales: the ILC updates within a few laps (trial-to-trial), the reference optimizer updates every 5–10 laps when enough data has accumulated for the GP to be meaningful.

2. **Compress trajectory representation for BO**. Instead of running Bayesian optimization over the full min-snap polynomial coefficient space (high-dimensional, poorly conditioned), project into wavelet approximation coefficients. 10 variables (Nθ = 10) with Daubechies-4 db4 at level L = 6 spans the global structure of a lap. This makes BO with NBO = 70 evaluations tractable and prevents overfitting to sparse data.

3. **Use arc-length (Frenet) parameterization for the reference**. Our current min-snap trajectories are parameterized in time; converting to arc-length parameterization decouples the path geometry from the speed profile, matching how this framework represents ey(s) and vx(s) separately. Speed and path can then be re-optimized independently given different data volumes.

4. **Use GP-predicted vs. actual lap time gap as inflation factor reduction criterion**. When TBO ≈ Treal (GP error small, dynamics well-learned), the trajectory optimizer's predictions are reliable. This is the principled moment to reduce safety margins: the optimizer now "knows" how the drone actually flies, so its corridor placement recommendations can be trusted with tighter inflation factors.

5. **Sparse GP with ~200 inducing points** is the recommended implementation for the residual dynamics model. This is computationally tractable at 50 ms MPC sampling and scales gracefully as data accumulates across iterations.

6. **NBO = 70 BO evaluations per reference optimization cycle** is the experimentally validated budget. Given that each evaluation is a closed-loop simulation (not hardware), 70 evaluations should be tractable in our PyBullet sim environment within a few minutes.

7. **Iterate to convergence (~10 outer loops)**. Experiments show diminishing returns beyond 10 iterations. Use |TBO - Treal| < 0.05s (5% of a ~1s gate segment time) as the stopping criterion for individual gate segments.

---

## Limitations & Caveats

**1/10-scale ground vehicle, not a drone.** All hardware experiments use a small RC car platform with a bicycle dynamics model. The aerodynamic forces, rotor dynamics, and three-dimensional trajectory freedom of a racing drone are fundamentally different. The Frenet-frame formulation assumes a 2D track (lateral deviation ey, heading error eψ); a 3D gate-to-gate drone trajectory requires extension to at minimum a 3D Frenet frame or a different arc-length parameterization. This is non-trivial.

**Fixed MPC tracking controller.** The framework is described as "controller-agnostic" but relies on MPC as the baseline tracking controller. Our system uses a geometric SE(3) tracker, which has different linearization properties for the GP surrogate. The GP residual dynamics model z := [vx, vy, w, a, δ] is specific to the bicycle model features; we would need to define equivalent features for quadrotor dynamics (e.g., body rates, thrust, attitude angles).

**Nθ = 10 may be too coarse for gate-constrained racing.** The coarsest wavelet approximation captures global trends but may miss gate-level constraints — the drone must pass precisely through each gate's physical boundary. Constraining ey(s) to remain within gate apertures at discrete arc-length positions would require incorporating hard constraints into the BO, converting it from unconstrained to constrained BO, which complicates the LCB acquisition function and convergence guarantees.

**NBO = 70 assumes cheap simulation.** Each BO evaluation runs a full closed-loop simulation. For a ~1 km track with 50 ms sampling, that is ~400 simulation steps per evaluation × 70 evaluations = ~28,000 simulation steps per outer iteration. In PyBullet this is fast, but if the GP prediction per step is expensive (200 inducing points × RBF kernel), wall-clock time could become a bottleneck.

**Gaussian Process scalability.** The sparse GP with 200 inducing points is manageable for the bicycle model's 5-dimensional input z. Quadrotor dynamics in 3D may require a higher-dimensional feature vector, increasing GP computation. Careful feature selection will be needed to keep inference fast.

**Convergence proof assumes GP error ε^j → 0.** In practice, for strongly nonlinear aerodynamic regimes (high-speed flight near obstacles, ground effect near gates), the GP may not converge to true dynamics with finite data. The asymptotic guarantee holds asymptotically; in early iterations (j < 5), the dynamics model is unreliable and BO-recommended trajectories may be dangerously aggressive.

**Hardware results use three repeated trials per trajectory.** This averaging reduces noise in the TBO ↔ Treal comparison but requires 3× the laps per outer iteration in real hardware. In simulation, single evaluations are noise-free, so this concern is reduced for our PyBullet-based iteration loop.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|---|---|---|
| Wavelet family | Daubechies-4 (db4) | Chosen for compact support and regularity |
| Decomposition level | L = 6 | Number of DWT levels |
| Optimization dimension | Nθ = 10 | 5 ey coefficients + 5 vx coefficients |
| Arc-length segments | Ns = 256 | Trajectory discretization for reconstruction |
| BO evaluations per iteration | NBO = 70 | Black-box search budget per outer loop |
| BO acquisition function | LCB with β^(1/2) scaling | Lower Confidence Bound exploration-exploitation |
| GP inducing points | 200 | Sparse GP approximation for residual dynamics |
| GP kernel | RBF | Radial Basis Function for smooth dynamics |
| GP feature vector | z = [vx, vy, w, a, δ] | Tire-relevant states for residual learning |
| MPC sampling time | 0.05 s (20 Hz) | Tracking controller step size |
| MPC solver | FORCESPRO | Real-time NLP solver |
| Outer iterations to convergence | ~10 | Observed in simulation and hardware |
| Hardware trials per trajectory | 3 | Repeated evaluations to reduce noise |
| Vehicle mass | m = 3.0 kg | 1/10-scale RC car |
| Wheelbase | Lf = Lr = 0.14 m | Symmetric front/rear |
| Nominal tire stiffness B | 1.3 | Pacejka model |
| Nominal shape factor C | 1.5 | Pacejka model |
| Nominal friction coefficient μ | 1.2 | Pacejka model |
| Nominal yaw inertia Iz | 0.024 kg·m² | 1/10-scale vehicle |
| Average lap time improvement | 20.7% | Over nominal baseline, 15 scenarios |
| Convergence metric | TBO ≈ Treal | BO prediction vs. actual tracked lap time |
