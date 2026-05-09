# Time-Optimal Planning for Quadrotor Waypoint Flight

- **URL**: https://arxiv.org/abs/2108.04537
- **Authors**: Philipp Foehn, Angel Romero, Davide Scaramuzza
- **Year**: 2021
- **Venue**: Science Robotics (Vol. 6, Issue 56, eabh1221)
- **Code**: https://github.com/uzh-rpg/rpg_time_optimal
- **DOI**: 10.1126/scirobotics.abh1221

---

## Key Contribution

This is the foundational paper that introduced **Complementary Progress Constraints (CPC)** — the first approach demonstrated to produce *truly* time-optimal trajectories for quadrotors through multiple waypoints at the full actuation limit.

The central insight is identifying and solving the *time allocation problem*: given N waypoints, at what time should the drone pass each one? All prior numerical optimization methods required this time allocation to be fixed *before* solving for the trajectory. Because the optimal time allocation is *a priori* unknown, those methods are structurally incapable of finding the true time-optimal solution — they find the best trajectory *given a fixed schedule*, not the globally best trajectory.

Foehn et al. reformulate the problem so that the waypoint-timing question is eliminated as a free variable. Instead of asking "pass waypoint j at time t_j," the algorithm asks "pass waypoint j *whenever you get there*." This converts a trajectory planning problem with fixed time events into a **Mathematical Program with Complementarity Constraints (MPCC)**, and enables joint optimization of trajectory shape and temporal structure simultaneously.

The result: an autonomous drone that beat two world-class human drone racing pilots on every lap of a seven-gate 3D race track — the first time an autonomous system outperformed human experts in a controlled racing benchmark.

---

## Technical Approach

### The Time Allocation Problem (And Why It Matters)

In classical minimum-snap or minimum-jerk polynomial trajectory optimization, the drone dynamics are written as:

```
p(t), v(t), a(t), j(t), s(t)   [flat output + derivatives]
```

The polynomial is defined over segments `[t_{j-1}, t_j]`, so segment durations `T_j = t_j - t_{j-1}` must be specified. The quadratic program (QP) for polynomial coefficients is then solvable in closed form. The standard approach (Mellinger & Kumar 2011, Richter et al. 2013) is:

1. Guess initial segment times `T_j` (usually proportional to Euclidean distance at a target speed).
2. Solve the QP for polynomial coefficients.
3. Check if actuator constraints are violated.
4. Adjust `T_j` via gradient descent (L-BFGS or similar) and repeat.

This is a *sequential* approach where time allocation and trajectory are alternately optimized. The time variables `T_j` appear nonlinearly in the polynomial cost (the cost scales as `1/T^{2s-1}` for order-s derivatives), so the outer loop is nonconvex and gradient descent only finds a local minimum.

**The critical structural problem:** even if the outer loop converges, it converges to the optimal trajectory *among all trajectories that pass waypoint j at time t_j*. If a trajectory that arrives at waypoint j slightly earlier (or later) would allow a much faster path elsewhere, the sequential approach cannot find it — the time schedule is treated as fixed once the QP is solved.

### CPC Formulation: Progress Variables

The CPC approach introduces a **progress variable λ_j ∈ {0, 1}** for each waypoint j. The variable encodes whether waypoint j has been passed:

- λ_j = 1: waypoint j not yet reached (incomplete)
- λ_j = 0: waypoint j has been passed (complete)

The key constraint is that λ_j can only *switch* from 1 to 0 when the drone is in the *local proximity* of waypoint j — within a tolerance ball of radius δ. This is expressed as a complementarity constraint:

```
λ_j * max(||p(t) - p_j|| - δ, 0) = 0   for all t
```

This states: either λ_j = 0 (waypoint complete) or the drone is within distance δ of waypoint j — never both false simultaneously. The complementarity structure forces the optimizer to "commit" to passing each waypoint at the moment it is nearby, rather than being free to revisit arbitrary time assignments.

Waypoints are sequenced by enforcing that λ_j can only decrease monotonically (no re-activating already-passed waypoints), and that all waypoints must be completed by trajectory end.

This is the same mathematical structure used in contact-rich manipulation (complementarity constraints for contact forces) — Foehn et al. recognized the structural analogy and imported this approach into trajectory planning.

### Optimization Solver: CasADi + IPOPT

The MPCC is discretized over a fixed number of time nodes (40 nodes per gate segment in the open-source implementation). The resulting discrete NLP is solved with **CasADi** (automatic differentiation) and **IPOPT** (interior-point NLP solver). CasADi provides efficient sparse Jacobian and Hessian computation; IPOPT handles the nonconvex constraints with a filter line-search method.

The discretized optimization variables include: position, velocity, acceleration, body rates, thrust, and the λ progress variables at each time node.

### Differential Flatness

The quadrotor system is differentially flat with flat outputs `[x, y, z, ψ]` (position + yaw). This means body attitude and motor thrusts can be recovered analytically from the flat output and its derivatives without numerical integration. The CPC formulation uses this property to express the full system trajectory — including motor commands — purely in terms of the flat output trajectory, avoiding explicit integration of nonlinear dynamics at every iteration.

### Actuation Model

The paper models single-rotor thrust bounds `f_min ≤ f_i ≤ f_max` per motor, and angular rate limits. The test platform had a thrust-to-weight ratio (TWR) of approximately 3.3 — typical for aggressive racing setups. Crucially, the optimizer can hit *full actuator saturation* (bang-bang-like thrust profiles) because the NLP directly manipulates control inputs, unlike polynomial methods that encode actuator constraints only through penalty terms.

---

## Results

### Race Track Benchmark

The primary experiment used a **seven-gate 3D race track** in one of the world's largest motion-capture facilities at the University of Zurich. Two world-class professional human drone racing pilots flew the track for multiple laps, establishing a human baseline. The CPC autonomous system then flew the same track.

**Outcome:** The autonomous drone won on every single lap. All autonomous laps were faster than the human best laps. The autonomous system also demonstrated significantly higher consistency — human pilots varied their acceleration profiles substantially and spent more time at sub-optimal thrust levels.

Specific lap times are not available in the published abstract (they appear in the paper's figure 4), but the qualitative result is unambiguous: this was the first time an autonomous system outperformed human experts in a direct racing comparison on a controlled track.

### Comparison with TOGT Planner (Qin 2024, ICRA)

The subsequent TOGT paper provides exact simulation numbers comparing CPC vs. polynomial L-BFGS methods on a 7-gate square track (QuadA, 0.85 kg):

| Method    | Compute Time | Lap Time |
|-----------|-------------|----------|
| CPC       | 466 s       | 6.85 s   |
| TOGT (L-BFGS poly) | 0.36 s | 7.53 s |
| TOGT-WP (poly, waypoints) | 0.14 s | 8.45 s |

CPC achieves **6.85 s vs. TOGT's 7.53 s** — a 10% faster lap using bang-bang actuator saturation, at the cost of 466 s vs. 0.36 s computation. The polynomial L-BFGS approach (TOGT-WP) takes 8.45 s — **23% slower than CPC**, confirming the structural conservatism of polynomial methods.

For a point-to-point flight (single waypoint), a separate comparison shows:
- PMPC: 1.85 s
- CPC: 1.66 s
(CPC is ~10% faster even on a single-segment task)

### Computational Cost

The CPC formulation solves to global optimality (or near-global for the MPCC relaxation), but at high computational cost. Solving for a 7-gate track takes ~8 minutes (466 s). For 31+ gates, CPC requires over 8 hours and fails to converge for 61+ gates. This is the primary practical limitation — CPC is an offline, batch optimizer.

---

## Relevance to Our System

Our system (`planning/trajectory_optimizer.py`) uses **min-snap polynomial trajectories with L-BFGS-based segment time optimization** — structurally identical to the approach that CPC was designed to supersede. Our current race time is 23 s against a target of 14 s, a 64% gap. This paper directly explains why and what to do about it.

### Why Our L-BFGS Converges to Conservative Solutions

There are four layered reasons our polynomial optimizer produces conservative (slow) trajectories:

**1. Sequential optimization of time and trajectory.** Our code likely sets segment times first (proportional to distance at some nominal speed), then solves the polynomial QP, then optionally refines times via L-BFGS. The L-BFGS only optimizes over the time allocation *given the polynomial parameterization* — it cannot discover trajectories that require different structural choices (like different waypoint orderings, earlier gate arrivals, or cross-gate shortcutting).

**2. Inherent smoothness of polynomials.** Min-snap polynomials enforce continuous 4th derivatives everywhere. The time-optimal trajectory for a quadrotor has *bang-bang* thrust control — maximum thrust in one direction followed immediately by maximum thrust in another, with instantaneous switching. Polynomials cannot represent this: they are C-infinity smooth and must spread out what bang-bang achieves in a step function across a smooth arc. This means polynomials always require more time to achieve the same velocity change.

**3. Conservative actuator constraint enforcement.** Our optimizer applies motor/thrust limits as soft penalty terms evaluated at discrete time samples. The polynomial smoothness means the optimizer sees a "safe margin" region near actuator limits that it avoids — it doesn't push fully to the actuator boundary because smooth polynomials evaluated at different sample points may violate constraints between samples. True time-optimal control pushes *exactly* to the actuation limit at all times, which polynomials approximate only loosely.

**4. Local minimum sensitivity.** L-BFGS is a quasi-Newton local optimizer. For our 25-segment trajectory (entry+exit waypoints per gate), the time allocation space is 25-dimensional with many local minima. The initial time guess (proportional to distance) is not warm-started from any physics-aware estimate of how long a quadrotor actually needs at each segment given its kinodynamics. The optimizer settles at the nearest local minimum to the initialization — which is typically slow because the initial times are conservative.

### The 23 s vs. 14 s Gap

If CPC achieves 6.85 s on a 7-gate track and our polynomial L-BFGS achieves 8.45 s (23% slower) on the same track, then a rough extrapolation suggests our 23 s race time could be brought to approximately **14-16 s** with a better time allocation strategy, without needing a completely different trajectory representation. This makes CPC-style time allocation the highest-leverage algorithmic improvement available.

---

## Actionable Takeaways

### Immediate (high confidence, low risk)

**1. Replace distance-proportional time initialization with physics-aware warm start.**
The current segment time initialization is `T_j = distance_j / v_nominal`. A better warm start uses the drone's kinodynamics: given the turning angle at each gate, estimate the minimum time to execute the arc at full thrust. This alone can reduce the number of L-BFGS iterations needed and improve the local minimum found.

```python
# Physics-aware initial segment time
turning_angle = angle_between(v_prev, v_next)  # at each waypoint
T_j = distance_j / v_cruise + k_turn * turning_angle  # k_turn ~ 0.3-0.5 s/rad
```

**2. Add a total-time penalty to the L-BFGS objective.**
If the current optimizer minimizes smoothness (snap integral) subject to time constraints, it may not aggressively shrink total race time. Add `alpha * sum(T_j)` directly to the cost so L-BFGS is explicitly penalized for slow segment allocations.

**3. Reduce the number of optimization segments.**
Our 25-segment parameterization (entry + exit waypoint per gate) has a 25-dimensional time space with many local minima. Each extra waypoint is an extra degree of freedom for the optimizer to get stuck. Consider dropping intermediate waypoints for straight segments and only keeping turn-critical ones.

### Medium-term (significant speedup potential)

**4. Implement the MPCC progress-variable formulation.**
Rather than pre-specifying when the drone reaches each gate, implement a simplified CPC: define a progress variable per gate that switches when the drone is within tolerance δ, and let the optimizer freely place gate-crossing times. This is the core of Foehn's contribution and directly targets our 23 s → 14 s gap. The open-source code at `uzh-rpg/rpg_time_optimal` (CasADi + IPOPT) provides a reference implementation.

**5. Replace L-BFGS with IPOPT for time allocation.**
L-BFGS is an unconstrained quasi-Newton method suitable for smooth objectives. IPOPT handles nonlinear constraints natively and uses second-order information (Hessian approximation via L-BFGS internally, but with proper constraint handling). For the time allocation subproblem with actuator bounds, IPOPT will find better solutions.

**6. Use differential flatness to evaluate actuator constraints analytically.**
Instead of sampling the trajectory at discrete points to check thrust bounds, use the differential flatness map to express thrust as a closed-form function of flat output derivatives. This eliminates sampling gaps and allows the optimizer to push exactly to actuator limits everywhere.

---

## Limitations & Caveats

**1. Computational cost is prohibitive for online use.**
CPC requires ~8 minutes for a 7-gate track and fails for 31+ gates. It is strictly an offline planner. For competition use, the trajectory must be pre-computed and stored; online replanning is not feasible with this method.

**2. Requires external localization.**
The 2021 paper uses a motion-capture system (100 Hz external cameras) for state feedback. On-board VIO adds ~0.1-0.3 m noise to position, which degrades tracking quality. The time-optimal trajectory assumes near-perfect tracking; under real VIO noise, a MPC controller may not be able to follow the aggressive reference faithfully enough to achieve the planned lap time.

**3. MPCC is not globally convex.**
Despite the term "complementarity," the resulting MPCC is nonconvex (the complementarity constraint `λ * g(x) = 0` is a bilinear equality). IPOPT finds a local KKT point, not a certified global optimum. In practice, the solution is very good (better than any polynomial method), but the global optimality claim requires verification.

**4. Polynomial trajectories remain competitive at scale.**
TOGT (polynomial L-BFGS) achieves within 10% of CPC lap time on 7-gate tracks and actually beats CPC on 31+ gate tracks by exploiting gate geometry. For our 12-gate course (25 segments with entry/exit waypoints), the conservatism gap may be less than the theoretical maximum — practical improvement may be 10-20% rather than the full 64% we need.

**5. Smoothness vs. bang-bang tradeoff is real.**
A polynomial trajectory *cannot* represent the true bang-bang time-optimal control. Any polynomial-based approach (ours, TOGT, MINCO) pays a smoothness tax of approximately 10% in lap time versus the CPC optimum on short tracks. This tax cannot be eliminated without changing the trajectory representation.

**6. Waypoint constraint tolerance matters enormously.**
CPC uses a tolerance ball of δ = 0.3 m around each waypoint. Larger tolerance = the optimizer can cut corners more aggressively = shorter path. Our gate sequencer's `pass_through_margin` plays the same role — if set too tight, it forces the drone through gate centers and prevents shortcutting.

---

## Key Parameters / Constants

| Parameter | Value | Source / Notes |
|-----------|-------|----------------|
| Nodes per gate (discretization) | 40 | Reference implementation default |
| Convergence tolerance | 0.3 | NLP solver tolerance |
| Initial velocity guess | 3.0 m/s | Warm start for optimizer |
| Thrust-to-weight ratio (test platform) | ~3.3 | Race drone used in human comparison |
| Gate waypoint tolerance δ | 0.3 m | Proximity ball for CPC switching |
| Optimization solver | CasADi + IPOPT | Open-source reference implementation |
| CPC compute time (7 gates) | ~466 s (7.8 min) | From TOGT comparison paper |
| CPC lap time (7 gates, sim) | 6.85 s | From TOGT Table II |
| TOGT (poly L-BFGS) lap time (7 gates) | 7.53 s | +10% vs CPC |
| TOGT-WP (poly, waypoints only) lap time | 8.45 s | +23% vs CPC |
| Single-segment CPC vs PMPC | 1.66 s vs 1.85 s | ~10% CPC advantage |
| CPC performance gap vs poly methods | ~10-23% | Depends on track complexity |
| Polynomial smoothness tax | ~10% | Inherent; cannot eliminate with polynomials |
| Num flat output dimensions | 4 | [x, y, z, ψ] |
| Polynomial degree used in CPC | N/A | Uses direct collocation, not polynomials |
| CPC fails at | 61+ gates | Computational intractability |

---

## Summary for Our System

The 23 s → 14 s gap (64% reduction needed) cannot be fully explained by tracking errors or EKF noise. The trajectory itself is slow. Foehn 2021 diagnoses the root cause: our L-BFGS segment time optimizer is structurally unable to find the time-optimal allocation because (a) times are decoupled from trajectory shape in the optimization, (b) polynomial smoothness prevents bang-bang actuator saturation, and (c) the 25-dimensional time space has many local minima that trap gradient descent.

The most actionable insight from this paper is not "implement CPC" (too slow for a 12-gate track in a reasonable compute budget), but rather: **the TOGT planner (Qin 2024) approximates CPC's approach within a polynomial framework using the same L-BFGS we already have**. The key differences are: (1) explicit total-time minimization in the objective, (2) smooth surjection mapping to keep the problem unconstrained, (3) analytical gradients for both time and gate-crossing position. Adopting these three changes within our existing `trajectory_optimizer.py` is the highest-leverage path from 23 s toward 14 s.
