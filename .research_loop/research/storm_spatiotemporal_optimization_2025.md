# STORM: Spatial-Temporal Iterative Optimization for Reliable Multicopter Trajectory Generation
- **URL**: https://arxiv.org/abs/2503.03252
- **Authors**: Jinhao Zhang, Zhexuan Zhou, Wenlong Xia, Youmin Gong, Jie Mei (corresponding)
- **Institution**: Department of Automation, School of Intelligence Science and Engineering, Harbin Institute of Technology, Shenzhen, China
- **Year**: 2025
- **Venue**: arXiv:2503.03252 [cs.RO], submitted March 5, 2025. Code: hitsz-mas.github.io/STORM

---

## Key Contribution

STORM addresses a fundamental tension in UAV trajectory optimization: existing methods either achieve strong constraint satisfaction at the cost of trajectory aggressiveness (conservative planners) or achieve fast trajectories while frequently violating kinodynamic or safety constraints (aggressive planners like MINCO). The core contribution is a **spatial-temporal decoupling framework** that alternates between two lightweight convex subproblems — a Quadratic Program (QP) for spatial (path shape) optimization and a Linear Program (LP) for temporal (segment-time) optimization — rather than solving a single expensive nonlinear program. This decomposition makes each subproblem individually tractable and enables strong, certifiable constraint satisfaction throughout iteration.

The second major contribution is a **guidance gradient mechanism**: a momentum-based term that prevents the iterative process from losing aggressiveness as the spatial and temporal subproblems converge. Without this mechanism, the decoupled iterations tend toward overly conservative timings because temporal compression competes with spatial feasibility. The guidance gradient carries gradient history across iterations (similar to heavy-ball momentum), compensating for this and consistently producing shorter-duration trajectories than MINCO while simultaneously achieving lower constraint violation rates.

---

## Technical Approach

### Trajectory Representation

STORM uses non-uniform B-splines of degree p=3. The key property exploited is the **convex hull property**: a B-spline is guaranteed to lie within the convex hull of its control points. This allows safety constraints (e.g., stay within a safe flight corridor polygon) to be enforced directly as linear constraints on control points, making the spatial subproblem a pure QP.

Trajectory derivatives (velocity, acceleration, jerk) are expressed as linear functions of control points via matrix recurrence (Proposition 1 in the paper). Specifically, the k-th derivative of the B-spline is itself a B-spline of degree p-k, whose control points are linear combinations of the original control points scaled by inverse knot-interval durations. This makes kinodynamic constraint satisfaction tractable within the LP.

### Problem Formulation

The full optimization minimizes:

```
J = (1/2) ∫ ‖jerk‖² dt  +  ρ · 1ᵀT
```

where T is the vector of knot intervals (segment durations), ρ is a scalar weight trading off path energy against total flight time, and the integral penalizes jerk squared (minimum-jerk criterion).

Subject to:
- Safety corridor constraints: C(t) ⊆ F (free space defined by safe flight corridors)
- Boundary conditions: prescribed position and velocity at start and end
- Kinodynamic constraints: bounds on velocity, acceleration, jerk norms
- Non-negativity: T ≽ 0

### Spatial-Temporal Decoupling

The key insight is that if segment times T are fixed, the spatial problem (optimizing control points Q) is a QP:

```
min_{Q}  (1/2) Q^T H Q + f^T Q
s.t.     A_safe Q ≤ b_safe       (SFC linear constraints via convex hull)
         A_bc Q = b_bc            (boundary conditions)
```

Conversely, if control points Q are fixed, the temporal problem (optimizing T) reduces to an LP after linearizing the kinodynamic constraints. The kinodynamic bounds like ‖v_i(T)‖ ≤ v_max are nonlinear in T (because velocity magnitudes scale as 1/T_i), but within a neighborhood of the current iterate T_k, they are approximated as:

```
V_i(T[i,V]) ≈ V_i(T_k[i,V])  within N(T_k[i,V])
```

with decay factors γ_ik ∈ (0,1) enforcing that the approximation tightens as the solver converges. The LP minimizes ρ·1ᵀT subject to these linearized kinodynamic constraints, ensuring total time monotonically decreases while kinodynamic feasibility is maintained.

### Algorithm Flow (Algorithm 1)

```
Input: waypoints, SFC corridors, kinodynamic limits, ρ, ρ_m, ε, γ
Initialize T₀ uniformly, Q₀ from minimum-jerk with T₀

repeat:
  1. Q_{k+1} ← OptimizeCPs(T_k)          // QP: fix T, optimize Q
  2. T_{k+1} ← OptimizeKnots(Q_{k+1})    // LP: fix Q, optimize T with linearized constraints
  3. if ρ ≤ ρ_m:
       UpdateGuidance(∇J)                  // momentum gradient accumulation
       Inject guidance into QP/LP          // steer toward faster solutions
  4. Decay γ monotonically                 // tighten linearization neighborhood
until |T_{k+1} - T_k| < ε or LP infeasible for m consecutive iters
```

### Guidance Gradient Mechanism (Algorithm 2)

The guidance gradient is a momentum vector accumulated across iterations:

```
g ← γ · g  +  ∇_{Q,T} J
```

where γ is a momentum decay coefficient. Normalized components are computed:

```
grad_Q ← c · normalize(g_Q)
grad_T ← (c/ρ) · normalize(g_T)
```

and the confidence level c scales the guidance strength. This term is added to both subproblem objectives, effectively "remembering" the overall gradient direction across the decoupled iterations. Without it, the QP tends to widen paths (increasing jerk) to gain time-allocation flexibility, which the LP then exploits but at the cost of path quality. The guidance keeps the optimization on a productive manifold.

The guidance is activated only when ρ ≤ ρ_m (= 100), i.e., when the temporal weight is low enough that the optimizer would otherwise lose track of the speed objective. At higher ρ, the temporal term is strong enough to drive aggressiveness without additional guidance.

### Constraint Linearization Strategy

Kinodynamic constraints on velocity and acceleration become:

```
A_i(T[i,A]) ≤ a_max
```

which is nonlinear in T. The linearization approximates this in a shrinking neighborhood around the current iterate using decay factors γ_{ik}. The key design choice is that γ decreases monotonically (square-root decay strategy in experiments), so early iterations allow larger linearization errors (faster progress) while later iterations enforce tighter neighborhoods (better constraint satisfaction at convergence).

---

## Results

### Simulation Benchmark (4,000+ trajectories)

Environment: 30×30×4 m³ maps with random obstacles, minimum trajectory length 15m.

**Trajectory quality vs. MINCO (2,000 large-scale trajectories):**

| Method | Avg Length (m) | Avg Duration (s) | Avg Jerk Energy (m²/s⁶) |
|---|---|---|---|
| MINCO | 25.49 | 9.43 | 3.68 |
| STORM (γ=0.3, aggressive) | 25.52 | **8.63** | 5.7 |
| STORM (γ=0.1, smooth) | **25.04** | 9.86 | **2.1** |

At γ=0.3: 8.5% shorter flight time than MINCO with comparable path length. At γ=0.1: 5.7% longer flight time but 43% lower jerk energy — more tracking-friendly trajectory shapes.

**Constraint violation rates:**

| Method | SFC violation | Velocity violation | Acceleration violation |
|---|---|---|---|
| MINCO | 2.6% | 0.94% | not reported |
| STORM (γ=0.3) | 0.45% | 0.6% | **0.0%** |
| STORM (γ=0.1) | **0.3%** | **0.0%** | **0.0%** |

STORM achieves 6-8x lower safety corridor violation rate and zero acceleration violations across all 2,000 test cases. This is a decisive advantage: MINCO's acceleration violations in simulation directly correspond to infeasible thrust demands in hardware.

### Ablation Studies

- **Decay factor γ effect**: Higher γ reduces convergence iterations but produces more aggressive (higher jerk) trajectories. Lower γ takes more iterations but yields smoother solutions. Duration decreases 0.45-2.6% as γ increases (within tested range).
- **Guidance gradient threshold ρ_m**: Setting ρ_m=100 is optimal. Below this value, guidance is unnecessary (temporal weight already drives aggressiveness). Above this value (activating guidance earlier), marginal benefit.
- **ρ sensitivity**: Performance stable for ρ in range [100, 512]. At ρ < 100 without guidance, trajectory durations increase by ~15%.

### Real-World Experiments

- Hardware: Intel i5-1240P CPU, RealSense D455 depth camera
- Localization: VINS-Fusion (visual-inertial odometry)
- Tracking: NMPC (nonlinear MPC)
- Planning frequency: **50 Hz** online re-planning
- Speed constraints: v_max = 0.6 m/s, a_max = 2 m/s² (conservative for indoor experiment)
- Result: **13% average flight time reduction** vs. MINCO under identical constraints
- Zero constraint violations during all real flights

---

## Relevance to Our System

Our system uses min-snap polynomial trajectories (not B-splines) with L-BFGS time allocation optimization in `planning/trajectory_optimizer.py`, and a lateral racing-line optimizer with multi-start L-BFGS-B in `planning/racing_line.py`. The current bottleneck is uneven tracking error across gates: the S-turn (gate-3) and helix (gate-7) regions accumulate error due to aggressive timing, while straight segments are under-utilized.

STORM's spatial-temporal decoupling is highly relevant to this problem for three specific reasons:

**1. Decoupled temporal re-timing.** Our current L-BFGS minimizes total time as a single scalar objective with no per-segment constraint enforcement. This produces segment times that are often too aggressive in tight turns and too conservative in straights. STORM's LP subproblem directly maps to what we want: given fixed control points (our polynomial coefficients), find segment times T that minimize total duration while satisfying per-segment velocity/acceleration bounds. We could implement this as a post-processing LP pass after our existing min-snap solve.

**2. Selective segment compression.** The γ decay factor in STORM controls how aggressively the LP compresses each segment. In our context, we want to compress straight-segment times (where tracking error is low) while preserving or even expanding S-turn/helix times (where tracking error is high). STORM's framework makes this straightforward: set tighter kinodynamic margins on high-error segments before running the LP. The LP will then naturally leave those segments longer. This is more principled than our current manual S-turn junction inflation in `trajectory_optimizer.py`.

**3. Constraint satisfaction under kinodynamic limits.** Our `DroneConstraints` class defines max_velocity=15 m/s, max_acceleration=20 m/s², max_jerk=50 m/s³. Currently L-BFGS enforces these as soft penalties, leading to occasional violations at segment boundaries (the comment in the code notes "rough estimate overestimates actual accel"). STORM's LP enforces them as hard constraints via the linearization, eliminating this class of error.

The B-spline representation itself is not directly applicable since we use piecewise polynomials, but the decoupling idea — separate QP for shape, LP for timing — translates cleanly. Our spatial problem is already a QP (min-snap with linear constraints at gates), and we could add an LP temporal pass on top of our existing solver.

**Affected modules:**
- `planning/trajectory_optimizer.py`: Add LP-based temporal re-timing as a post-processing step after min-snap polynomial solve. The LP inputs are the per-segment kinodynamic constraint evaluations; the LP output replaces or corrects the L-BFGS-optimized segment times.
- `planning/racing_line.py`: Replace or supplement the time-allocation component of the racing-line optimizer. The guidance gradient mechanism could replace the smoothness_weight heuristic currently used to prevent over-aggressive corner-cutting.

---

## Actionable Takeaways

1. **Implement LP temporal re-timing pass.** After solving the min-snap QP for control points, add a secondary LP (using scipy.optimize.linprog or cvxpy) that takes the fixed polynomial coefficients and finds minimum-duration segment times satisfying velocity, acceleration, and jerk bounds. This is equivalent to STORM's OptimizeKnots step applied once (not iteratively). Expected impact: tighter kinodynamic compliance and potentially 8-13% time reduction on straight segments.

2. **Per-segment kinodynamic margin scaling.** Before the LP, identify high-tracking-error segments (S-turns, helix) from `simulation.per_gate_avg_error` and inflate their kinodynamic constraints (reduce effective v_max/a_max for those segments by a factor of 0.7-0.85). The LP will naturally allocate more time to these segments. This replaces the manual junction inflation heuristic.

3. **Iterative spatial-temporal alternation.** Run the QP→LP loop for 3-5 iterations (not just once). Each QP pass re-shapes the path given the LP-updated timing; each LP pass re-times given the QP-updated shape. In practice, convergence is fast (the paper shows most problems converge in under 10 iterations). This tightens constraint satisfaction significantly.

4. **Momentum guidance for the L-BFGS racing line.** In `racing_line.py`, the multi-start L-BFGS-B currently uses smoothness_weight as a proxy for guidance. Replace with a proper gradient momentum term: accumulate ∇J across restarts and inject a weighted version into the next restart's initial gradient. This replaces the heuristic smoothness_weight=0.40 tuning.

5. **Set ρ=512 for initial temporal optimization, reduce to ρ_m=100 before activating guidance.** Our current approach uses a single scalar speed_weight; restructure to match STORM's schedule: start with high temporal penalty to get fast solution, then add guidance gradient once temporal weight drops below threshold.

6. **Use γ=0.1 configuration (smooth variant) for high-error gates.** For gates 3 and 4 (S-turn), the γ=0.1 setting produces 43% lower jerk energy at the cost of 5% longer time. Given our current gate-3 tracking error of 0.345m, a smoother trajectory there would reduce tracking error further and likely improve overall lap time by improving controller tracking on the whole course.

7. **Validate constraint violation rate.** Add a post-optimization check that evaluates velocity and acceleration at dense sample points along each polynomial segment and reports the violation fraction. This matches STORM's evaluation methodology and gives us a direct proxy for tracking error: high violation fraction → high tracking error.

---

## Limitations & Caveats

**B-spline vs. polynomial basis.** STORM's convex hull property and control-point linearization are specific to B-splines. Our min-snap uses monomial/power basis polynomials, where control points do not have the same geometric interpretation. The spatial QP decoupling still applies (min-snap is already a QP), but the safety corridor constraint linearization does not translate directly. We would need to either switch to B-splines or use a different constraint formulation for the spatial subproblem.

**Low-speed regime.** Real-world experiments use v_max=0.6 m/s, a_max=2 m/s². Our competition targets v_max=15 m/s, a_max=20 m/s². The linearization neighborhood assumptions in the LP may be less valid at high speeds where the kinodynamic constraint functions are more curved. The convergence guarantees may not hold without re-tuning γ and ρ for this regime.

**No gate-passing constraints.** STORM optimizes for general obstacle avoidance via safe flight corridors. It does not include the gate pass-through geometric constraints that we need (drone must pass through a 1.2×1.2m opening at the correct attitude). Integrating gate constraints into the QP spatial problem requires extending the SFC formulation to include half-space constraints on gate normal planes, which adds non-trivial complexity.

**NMPC tracking assumed.** The paper assumes a model predictive controller for tracking, which is more capable of following kinodynamically tight trajectories than our geometric SE(3) tracker (Lee et al.) in `control/mpc_tracker.py`. The claimed 13% time reduction assumes the tracker can exploit the faster timing. If our tracker has higher latency or bandwidth limits, the actual benefit will be smaller.

**Offline planning only.** STORM runs at 50 Hz with online re-planning in the real-world experiments, but for a pre-defined race course, offline optimization is sufficient and we have no re-planning requirement. This means we can afford more iterations (10-20 vs. their online budget) without the latency concern.

**Small-scale course vs. large-scale benchmark.** The 30×30×4 m³ benchmark is large relative to our race course. The 8-13% time improvement figure is for long trajectories (avg 25m). Our race segments are shorter (3-8m between gates), so the linearization may converge in fewer iterations and the absolute time savings may be smaller in seconds.

---

## Key Parameters / Constants

These are directly usable values from the paper:

| Parameter | Value | Description |
|---|---|---|
| B-spline degree | p = 3 | Cubic B-splines |
| Temporal weight | ρ = 512 | Initial value for time penalty weight |
| Guidance threshold | ρ_m = 100 | Activate guidance gradient when ρ ≤ ρ_m |
| Convergence tolerance | ε = 0.05 s | Stop when ‖T_k - T_{k-1}‖ < 0.05s |
| Momentum decay | γ ∈ [0.1, 0.3] | 0.1 = smooth/low-jerk; 0.3 = fast/aggressive |
| Gradient normalization confidence | c | Scales guidance injection strength |
| Decay strategy | square-root | γ decreases as 1/√k across iterations |
| Real experiment v_max | 0.6 m/s | Conservative; our limit is 15 m/s |
| Real experiment a_max | 2 m/s² | Conservative; our limit is 20 m/s² |
| Online planning frequency | 50 Hz | For real-time re-planning |
| Time improvement over MINCO | 8.5-13% | Trajectory duration reduction |
| SFC violation reduction | 6-8x | vs. MINCO (2.6% → 0.3-0.45%) |
| Acceleration violation reduction | ∞ | MINCO has nonzero accel violations; STORM has zero |
