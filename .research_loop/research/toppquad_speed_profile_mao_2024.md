# TOPPQuad: Dynamically-Feasible Time-Optimal Path Parametrization for Quadrotors

- **URL**: https://arxiv.org/abs/2309.11637
- **Authors**: Katherine Mao, Igor Spasojevic, M. Ani Hsieh, Vijay Kumar
- **Year**: 2024
- **Venue**: IROS 2024

---

## Key Contribution

TOPPQuad contributes a time-optimal path parametrization (TOPP) algorithm that respects the **full rigid-body dynamics and individual motor thrust constraints** of a quadrotor, rather than relying on convex relaxations or overly conservative simplifications. The method takes a pre-planned geometric path (from any collision-avoidance or racing-line planner) and computes the fastest dynamically-feasible speed profile along that path. The core insight is that existing min-snap and min-jerk trajectory methods either (a) ignore motor-level constraints, producing trajectories that saturate actuators at aggressive segments, or (b) apply conservative time scaling that leaves significant time on the table. TOPPQuad closes this gap by directly embedding thrust constraints into the optimization.

The phrase "path parametrization" is key: the **geometric path is not changed**. Only the time schedule — how fast to traverse each portion — is optimized. This makes TOPPQuad a natural post-processing layer for any trajectory planner that produces smooth geometric paths.

A secondary contribution is the treatment of **rotational dynamics inside the TOPP formulation**. Most prior TOPP methods for quadrotors handle only thrust magnitude bounds and ignore the coupling between translational and rotational motion. TOPPQuad explicitly integrates quaternion kinematics and angular rate constraints, producing speed profiles that are feasible not just translationally but also rotationally. This is critical for sharp turns and aggressive maneuvers where the required attitude change rate can saturate body-rate limits even when thrust bounds alone appear satisfied.

---

## Technical Approach

### Problem Setup and Decoupling

The fundamental separation of concerns that makes TOPP powerful is the following:

- **Geometric path**: γ: [0, S_end] → ℝ³, parameterized by arc-length s. This encodes the shape of the trajectory — the racing line — but carries no information about timing.
- **Time parametrization**: a function χ(t) such that the drone position is p(t) = γ(χ(t)). This encodes when the drone reaches each point on the path.

By fixing γ (from the racing line optimizer) and optimizing only χ, the problem decouples into two sequential steps:
1. Find the best geometric path through the gates (handled by `racing_line.py`).
2. Find the fastest feasible time schedule for traversing that path (TOPPQuad's job).

This decoupling is exactly what a two-phase approach to our race time regression needs: the smooth geometric path from iteration 13 (which gives 0.179m tracking) is fixed and good, and we only need a better time schedule to traverse it faster.

### Square Speed Reparametrization: h(s)

The central mathematical device is the **squared speed profile**:

```
h(s) := (ds/dt)²   [square of the arc-length traversal speed as a function of arc-length s]
```

This substitution converts time derivatives into spatial derivatives:

```
d/dt = √h(s) · d/ds
d²/dt² = (1/2) h'(s) · d/ds + h(s) · d²/ds²
```

The traversal time is then:

```
T = ∫₀^S_end 1/√h(s) ds
```

which is the objective to minimize. The key benefit of the h(s) reparametrization is that **velocity bounds become linear in h**, and (more importantly) **acceleration bounds become linear in both h and h'**, which is critical for making the constraint structure tractable to a nonlinear solver.

Specifically, a bound on translational acceleration magnitude |a| ≤ a_max is equivalent to a bound on |½ γ'(s) h'(s) + γ''(s) h(s)| ≤ a_max. Given known geometry γ, this is a linear constraint in {h, h'} at each s. If only translational constraints were present, the problem would reduce to a convex program. The addition of rotational dynamics (quaternion kinematics) and per-motor thrust bounds makes it non-convex, but the h(s) representation still dramatically reduces the problem size and makes it numerically tractable.

### Quadrotor Dynamics Model

Standard rigid-body dynamics in the spatial domain:

```
ṗ = v
v̇ = R e₃ (c/m) + g
q̇ = (1/2) Ω(ω) q
ω̇ = J⁻¹(τ - ω × Jω)
```

where p is position, v velocity, R ∈ SO(3) rotation matrix, q ∈ S³ quaternion, ω body angular rate, c total collective thrust, τ body torque, m mass, J inertia. Motor thrusts u ∈ ℝ⁴ map to collective thrust and body torque via the allocation matrix F:

```
[c, τ]ᵀ = F u
```

### Constraint Derivation: The Core Equation

After substituting the h(s) reparametrization, the translational and rotational Newton-Euler equations become (Eq. 18 in the paper):

```
[m(½ γ'(s) h'(s) + γ''(s) h(s) - g)]       [R(s)e₃   0 ]
[J(½ ω(s) h'(s) + α(s) h(s)) + ω×Jω · h(s)] = [0        I₃] F u
```

where α(s) := ω'(s) is angular acceleration with respect to arc-length. Given a candidate speed profile {h(s), h'(s)} and known geometry {γ(s), ω(s), α(s)}, this equation uniquely determines the required motor thrust vector u(s). The feasibility constraint becomes:

```
u_min ≤ u(s) ≤ u_max   ∀ s ∈ [0, S_end]
```

This formulation is crucial: it means TOPPQuad does not solve for motor thrusts separately from the trajectory — they are derived quantities from the speed profile, and feasibility is checked directly against the hardware thrust bounds.

### Orientation Propagation Along the Path

Orientation q(s) is propagated spatially using:

```
q'(s) = (1/2) Ω(ω(s)) q(s)
```

with normalized forward Euler updates to maintain unit-quaternion constraint:

```
q_{i+1} = (q_i + (1/2) Ω(ω_i) q_i Δs) / ‖q_i + (1/2) Ω(ω_i) q_i Δs‖
```

The attitude at each path point s_i is determined by the desired heading (pointing toward the next gate), and then angular rate ω(s_i) is computed from the attitude sequence. This is the same differential flatness relationship that min-snap planners use, now expressed spatially rather than temporally.

### Decision Variables and Solver

The full NLP discretizes h(s) over N=300 grid points and solves for:
- h(s_i) — squared speed at each grid point
- h'(s_i) — derivative of squared speed
- q(s_i) — quaternion at each grid point
- ω(s_i) — body angular rate
- α(s_i) — angular acceleration in spatial domain
- u(s_i) — motor thrust vector

The objective is minimization of T = Σ 2Δs / (√h_i + √h_{i+1}) (trapezoidal integration).

Solver: IPOPT (interior-point NLP) via CasADi automatic-differentiation interface, allowing efficient Jacobian/Hessian computation. Warm-started from the min-snap speed profile (v=1 m/s baseline trajectory achieves 98.1% convergence success vs. 60.9% for v=5 m/s initialization).

### Re-timing Existing Trajectories

TOPPQuad is explicitly described as a post-processing technique: it accepts any sufficiently smooth existing trajectory (min-snap, min-jerk, spline) and re-times it. The geometric path γ(s) is extracted by arc-length reparametrization of the original trajectory's position sequence, and then the optimization finds the fastest feasible timing. This is the intended use case for recovering speed from our current smooth 17.70s trajectory.

---

## Results

### Simulation Benchmarks (200 randomized 4-waypoint trajectories, 10×10×10 m³)

- **40–50% faster** than α-scaled minimum-snap trajectories (uniform time scaling)
- **~10% faster** than prior TOPP methods using relaxed (aggregate not per-motor) thrust constraints
- Consistently maintained individual motor thrusts within hardware limits while baselines frequently exceeded them in aggressive segments
- Average speed increase approximately 1.8 m/s over unconstrained baseline trajectories

### Hardware Validation (CrazyFlie 2.0, m=32 g, u_max=0.14375 N per motor)

| Trajectory | TOPPQuad | Min-Snap baseline | Improvement |
|---|---|---|---|
| Straight line | 2.6 s | 3.4 s | 24% faster |
| L-curve | 3.6 s | 4.0 s | 10% faster |
| Lissajous | 9.1 s | crashed (2 m/s) | succeeded |
| 3D X-curve | — | crashed (2 m/s) | succeeded |

The Lissajous and 3D X-curve results are qualitatively significant: TOPPQuad produced shorter-duration trajectories than the baseline conservative min-snap, yet the baseline crashed while TOPPQuad succeeded. This demonstrates that dynamic feasibility is not just about matching speed — it is about ensuring the trajectory does not demand attitude maneuvers that exceed actuator bandwidth, even at the same nominal speed.

### Real-World Forest Flight

Applied to a 336 m GPS-logged flight path: TOPPQuad trajectory time 136 s vs. original 236 s, a **43% reduction** with verified thrust feasibility throughout.

### SE(3) Controller Gains Used in Hardware Validation (Table III)

```
K_p (position):  [8, 8, 19]
K_d (velocity):  [5.5, 5.5, 8.7]
K_R (rotation):  [2812, 2812, 163]
K_ω (body rate): [128, 128, 73]
```

The ratio K_R_xy / K_R_z ≈ 17:1 and K_ω_xy / K_ω_z ≈ 1.75:1 reflect the physical asymmetry of quadrotor attitude authority. These ratios serve as a useful sanity check for any SE(3) controller tuning.

---

## Relevance to Our System

### Current Situation (Iteration 13)

Our system has reached a favorable tracking quality (0.179m avg error, max 0.697m) but at the cost of race time: 17.70s vs. the 13.31s we had before iteration 13 and our target of ~14–15s. The smooth local minimum found by the `racing_line.py` L-BFGS optimizer produces a geometrically excellent path but traverses it too conservatively. The trajectory optimizer then computes segment times via L-BFGS that minimize a smoothness/time trade-off, producing a trajectory the PD controller can track but that is significantly slower than necessary.

The core mismatch: our speed profiling in `racing_line.py` uses a curvature-based heuristic (v_max = sqrt(a_max / curvature)) that is independent of actual motor thrust feasibility. On straight segments and shallow curves, this likely imposes an incorrect speed cap. More importantly, the L-BFGS time allocation in `trajectory_optimizer.py` does not directly enforce dynamic feasibility — it encodes feasibility only implicitly through an acceleration penalty term.

### How TOPPQuad-Style Speed Profiling Addresses This

TOPPQuad's key offering for our situation is **separating geometry from timing and then finding the globally fastest feasible timing for a fixed geometry**.

The smooth racing line from iteration 13 is the geometry we want to keep — it produces excellent tracking because it avoids aggressive curvature changes. The question is: what is the fastest speed at which our drone can traverse this specific path while remaining dynamically feasible?

The answer is not what our `SpeedProfiler` computes (which uses a centripetal acceleration formula at waypoints) — it is determined by the interplay of centripetal demand, attitude change rate, and motor thrust allocation simultaneously. This is exactly what TOPPQuad solves.

**Concrete application to our system:**

1. **Extract the geometric path** from the iteration 13 racing line as a sequence of densely-sampled (N=300) arc-length-parameterized 3D positions. This is straightforward: evaluate the min-snap trajectory at fine time steps, then reparametrize by cumulative arc-length.

2. **Compute the TOPPQuad h(s) optimization** for this fixed path. Given our drone's mass (~1 kg), max thrust per motor (~5 N), and inertia, the optimization finds h(s) that is as large as possible everywhere subject to the motor constraint u(s) ≤ u_max. On straight segments where the path has low curvature, h(s) will be large (fast traversal). At helix turns and S-turns where curvature and attitude rate demand is high, h(s) will be smaller but still larger than our current conservative heuristic.

3. **The expected result**: the recovered trajectory keeps exactly the same geometric shape as the iteration 13 racing line (so tracking quality is preserved) but traverses straights and shallow curves faster. A 40% speed improvement on straight-dominated segments could reduce race time from 17.70s to around 14–15s, which is our target.

### Why This Is Better Than Our Current Approach

Our current `SpeedProfiler` uses `v_max = sqrt(a_max / curvature)` at discrete waypoints. This has three problems:

**Problem 1 — waypoint-level curvature is too coarse.** Curvature at 7-10 gate waypoints does not capture the actual path curvature at every point between gates. The polynomial interpolation can have high curvature mid-segment that the waypoint estimate misses.

**Problem 2 — centripetal formula uses aggregate acceleration, not per-motor thrust.** The formula v² = a_max * r assumes a single acceleration budget. TOPPQuad reveals that the actual constraint is per-motor thrust, which couples both centripetal force (translational acceleration) and the required torque for attitude change (rotational dynamics). The centripetal formula is a lower bound on feasibility that ignores the rotational coupling — it can be either too conservative (slowing too much on straight approaches) or too optimistic (not accounting for attitude rate limits in rapid direction changes).

**Problem 3 — no forward/backward propagation across the full path.** Our `SpeedProfiler` does a two-pass forward-backward sweep at waypoint resolution. TOPPQuad operates at N=300 points and couples the speed profile as a continuous function h(s) across the entire path length, enforcing constraints everywhere.

### Simplified Approximation: Thrust-Cap Speed Profile

If implementing full TOPPQuad (CasADi + IPOPT, N=300, 30s solve) is too heavy for the current iteration, a useful approximation exists:

For a segment with local curvature κ(s), the required centripetal acceleration is v²·κ(s). The total acceleration demand including gravity compensation is approximately:

```
a_total(s) = sqrt((v²·κ(s))² + g²)
```

The maximum feasible speed at this point is:

```
v_max(s) = sqrt((a_max_net - g_component) / κ(s))
```

where a_max_net is derived from u_max·4/m (total max thrust divided by mass). This gives a continuous speed cap as a function of arc-length rather than discrete waypoints. Combined with the existing forward-backward sweep, this would be a significant improvement over the current waypoint-level curvature formula without requiring a full NLP solve.

### Integration Points in Our Codebase

- **`planning/racing_line.py` → `SpeedProfiler`**: Replace the waypoint-level curvature heuristic with arc-length-parameterized continuous curvature from the polynomial trajectory. Compute v_max(s) at N≥100 points, then run the forward-backward sweep at that resolution. This is a 1–2 day implementation without external solvers.

- **`planning/trajectory_optimizer.py`**: Add a post-hoc feasibility check: after computing the min-snap polynomial, evaluate Eq. 18 at 300 arc-length points to compute required u(s). Flag any s where u(s) > u_max — these are the exact locations where the trajectory is asking for more than the hardware can provide.

- **Full TOPPQuad integration (offline)**: Add a `planning/topp_retimer.py` module that takes a `RaceTrajectory` as input, extracts the geometric path, runs the CasADi/IPOPT optimization, and outputs a new `RaceTrajectory` with the same geometry but faster timing. This is a complete implementation of TOPPQuad and could yield the full 40% improvement.

---

## Actionable Takeaways

1. **Re-time the iteration 13 trajectory using TOPPQuad.** The smooth geometric path from iteration 13 is exactly the right input: it has low curvature, is trackable by the PD controller, and represents the correct racing line shape. The only deficiency is the time schedule. TOPPQuad finds the fastest feasible schedule for this fixed path. Expected race time improvement: 17.70s → ~14–15s.

2. **Implement a continuous arc-length speed profile as a lightweight approximation.** Reparametrize the racing line as arc-length s, sample curvature κ(s) at N=100–300 points, compute v_max(s) = sqrt(a_max / κ(s)) at each point, and run the forward-backward sweep. This replaces the current waypoint-level computation and should capture speed opportunities on straights that the current approach misses. Estimated implementation: 30–60 lines in `racing_line.py`.

3. **Add the TOPP feasibility diagnostic.** After computing the min-snap trajectory, evaluate required motor thrusts at 300 arc-length points using Eq. 18. Print the fraction of arc-length where u(s) > u_max and which gates are worst. This immediately reveals whether our current trajectory is asking for physically infeasible maneuvers and where.

4. **Use CasADi + IPOPT for the full offline retimer.** Both are pip-installable. The integration fits as a post-processing stage in `planning/trajectory_optimizer.py` or as a standalone `planning/topp_retimer.py`. Warm-start from the current min-snap speed profile as recommended by the paper.

5. **Set N=300 discretization for the TOPP grid.** The paper shows this is sufficient for adequate constraint enforcement at Δs ≈ 0.025 m resolution. For a 17.70s trajectory at ~8 m/s average speed, the path length is approximately 140 m, giving Δs ≈ 0.47 m at N=300 — coarser than the paper's test cases. Use N=1000 for our longer path.

6. **Warm-start the TOPP NLP with a slow initial guess.** Per the paper, initializing at v=1 m/s achieves 98.1% convergence success vs. 60.9% at v=5 m/s. Always use a slow initial guess and let IPOPT find the fastest feasible profile.

7. **Preserve the geometric path exactly when re-timing.** The tracking quality improvement from iteration 13 came from the smooth geometric path, not from the timing. Any re-timing that modifies the geometric path risks losing this quality. TOPPQuad guarantees geometry is fixed, which is exactly what we need.

---

## Limitations & Caveats

**Computational cost (~30s per trajectory).** The full NLP with N=300 grid points and IPOPT takes approximately 30 seconds on a desktop CPU. This is fine for offline pre-race planning but not online re-planning. Our architecture pre-computes trajectories before the race, so this limitation is not binding. For our longer path (N=1000 recommended), solve time may increase to 2–5 minutes — still acceptable offline.

**No aerodynamic drag model.** TOPPQuad ignores air drag and motor dynamics. At high speeds (>5 m/s), drag significantly increases required thrust and reduces achievable speed. The computed speed profiles will be slightly optimistic on high-speed straight segments. For our use case this introduces a ~5–10% over-estimate of achievable speed on fast straights, which can be addressed by using a conservative u_max (e.g., 90% of physical limit).

**Non-convex NLP without global optimality guarantees.** IPOPT converges to a local minimum. In practice, solutions are significantly faster than baselines, suggesting good-quality local optima — especially when warm-started with a reasonable initial guess. However, there is no certificate of time-optimality.

**Requires a pre-planned geometric path.** TOPPQuad is a path parametrizer, not a path planner. It takes a fixed geometric path and finds the fastest timing. If the geometric path itself is suboptimal (wrong racing line shape), TOPPQuad cannot correct it. The iteration 13 racing line geometry is our input — if it has any geometric inefficiencies (e.g., the S-turn at gates 3-4), TOPPQuad will faithfully produce the fastest feasible timing through that geometry but cannot fix the geometry itself.

**CrazyFlie-specific hardware validation.** All hardware experiments use a 32 g CrazyFlie 2.0. Competition racing quads are typically 200–400 g with different thrust-to-weight ratios, inertia tensors, and arm lengths. The method transfers completely but the specific numerical parameters (u_max = 0.14375 N, SE(3) gains) require re-calibration for our platform. Our `DroneConstraints` class already specifies mass=1.0 kg and max_thrust=20.0 N — these feed directly into the TOPPQuad formulation.

**Quaternion integration accuracy.** The paper uses a simple normalized forward Euler quaternion update rather than an exponential map or Lie-group integrator. At N=300 over a 7.5 m path (Δs=0.025 m), this is sufficient. For our ~140 m path with N=300 (Δs=0.47 m), integration error accumulates significantly and we should use N=1000 or a higher-order integrator (e.g., quaternion slerp or Runge-Kutta on SO(3)).

**No gate-crossing geometry constraints.** TOPPQuad treats the path as a free-space curve and enforces only dynamic feasibility. It does not explicitly constrain the drone to pass within the gate opening at gate-crossing arc-length values. For racing, we need to couple TOPPQuad with gate-aware path generation. In our system, the racing line geometry from `racing_line.py` already encodes correct gate pass-through points, and re-timing does not change those positions, so this limitation is structurally avoided by our architecture.

**Local minimum sensitivity in our racing line optimizer.** The iteration 13 observation that `racing_line.py` L-BFGS has two distinct local minima (fast/inaccurate basin at 12.78s, smooth/accurate basin at 17.70s) is a separate concern from TOPPQuad. Even with perfect speed profiling via TOPPQuad, if the geometric path is in the wrong basin, we get either bad tracking or suboptimal geometry. TOPPQuad operates downstream of this choice and cannot fix it.

---

## Key Parameters / Constants

| Parameter | Value | Context |
|---|---|---|
| Discretization grid N | 300 | Paper experiments; Δs ≈ 0.025 m for 7.5 m paths |
| Recommended N for our system | 1000 | ~140 m path at Δs ≈ 0.14 m resolution |
| Motor thrust u_max (CrazyFlie) | 0.14375 N per motor | CrazyFlie 2.0 hardware validation |
| Vehicle mass (CrazyFlie) | 0.032 kg | Hardware experiments |
| Our drone mass | 1.0 kg | `DroneConstraints.mass` |
| Our max thrust | 20.0 N total | `DroneConstraints.max_thrust` = 5.0 N/motor assumed |
| SE(3) position gain K_p | [8, 8, 19] | Hardware validation, xy vs z asymmetry |
| SE(3) velocity gain K_d | [5.5, 5.5, 8.7] | Hardware validation |
| SE(3) rotation gain K_R | [2812, 2812, 163] | K_R_xy/K_R_z ≈ 17:1 |
| SE(3) body rate gain K_ω | [128, 128, 73] | K_ω_xy/K_ω_z ≈ 1.75:1 |
| Solve time (N=300) | ~30 s | Desktop CPU, IPOPT via CasADi |
| Warm-start velocity | 1 m/s | 98.1% convergence vs. 60.9% at 5 m/s |
| Simulation time reduction | 40–50% | vs. α-scaled min-snap baselines |
| Forest flight time reduction | 43% | 236 s → 136 s on 336 m path |
| Hardware line trajectory | 24% faster | 3.4 s → 2.6 s |
| Hardware L-curve | 10% faster | 4.0 s → 3.6 s |
| Path space for simulation | 10×10×10 m³ | 200 randomized 4-waypoint scenarios |
| Integration method | Forward Euler + quaternion normalization | Eq. 23, normalized to maintain unit norm |

The asymmetric SE(3) gain ratios (K_R_xy/K_R_z ≈ 17:1, K_ω_xy/K_ω_z ≈ 1.75:1) are worth comparing against our `TrackerConfig` in `control/mpc_tracker.py` as a sanity check on rotational authority balance.
