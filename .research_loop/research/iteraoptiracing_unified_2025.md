# IteraOptiRacing: Unified Planning-Control Framework

- **URL**: https://arxiv.org/abs/2507.09714
- **Authors**: Yifan Zeng, Yihan Li, Suiyi He, Koushil Sreenath, Jun Zeng
- **Year**: 2025
- **Venue**: arXiv preprint (cs.RO, eess.SY) — submitted July 13, 2025

---

## Key Contribution

IteraOptiRacing presents a **unified planning-control framework** for real-time autonomous racing that eliminates the traditional hard separation between trajectory planning and tracking control. The central claim is that prior iterative racing methods (e.g., LMPC variants) suffer from *nonsmooth mode-switching* between a time-optimal mode and an obstacle-avoidance mode, which introduces discontinuities in control output and degrades performance in dynamic multi-agent scenarios.

The three core contributions are:

1. **Unified i2LQR formulation**: A single optimization problem derived from the Iterative Linear Quadratic Regulator for Iterative Tasks (i2LQR) that simultaneously handles time minimization and multi-obstacle avoidance without switching logic.

2. **Parallelizable real-time execution**: The algorithm solves K independent iLQR subproblems (one per candidate terminal state) and selects the best feasible solution, enabling >33 Hz update rates on a consumer-grade CPU (Intel i7-8700, 3.20 GHz) while baselines operate at 15–30 Hz.

3. **Demonstrated superiority in dynamic multi-agent racing**: Across 300 randomized test scenarios with up to 9 obstacle vehicles on three track geometries, IteraOptiRacing achieves 100% full-vehicle-overtaking success where all three LMPC baselines fail entirely in multiple test categories.

---

## Technical Approach

### Vehicle Model

The framework uses a **6-state dynamic bicycle model** parameterized in Frenet (curvilinear) coordinates. The state vector is:

```
x = [vx, vy, ωz, eψ, sc, ey]
```

where `vx`, `vy` are longitudinal and lateral velocities, `ωz` is yaw rate, `eψ` is heading error relative to track centerline, `sc` is arc length progress along track, and `ey` is lateral deviation from centerline. Control inputs are acceleration `a` and steering angle `δ`.

This Frenet-frame parameterization is important: it makes track-boundary constraints trivially expressible as bounds on `ey` and enables direct progress-based cost formulation without reference to absolute position.

### Core Algorithm: i2LQR

At each timestep `t`, the algorithm performs:

**Step 1 — Terminal set construction.** The K=32 nearest historical states to the current state are retrieved from a dataset accumulated over prior laps. These form candidate target terminal states `{zg}`. The distance metric used for nearest-neighbor selection is a weighted norm over the state space.

**Step 2 — Parallel iLQR optimization.** For each of the K candidate terminal states, an independent iLQR problem is solved over a prediction horizon N=12. The objective function is:

```
J = p(x_{t+N|t}, zg) + Σ_{k=0}^{N-1} l(x_{t+k|t}, u_{t+k|t})
```

where `p(·)` is a terminal cost enforcing proximity to `zg` (weighted by adaptive matrix `QN`), and `l(·)` penalizes control magnitude and control rate changes. The dynamics are linearized as affine time-varying models:

```
x_{t+k+1|t} = A_{t+k|t} · x_{t+k|t} + B_{t+k|t} · u_{t+k|t} + C_{t+k|t}
```

The linearization is updated at each backward-forward iLQR pass, allowing the algorithm to track nonlinear dynamics accurately over the horizon.

**Step 3 — Constraint handling via exponential barrier functions.** Non-convex constraints (track boundaries, collision avoidance) are converted into soft penalty terms:

```
c = q1 · exp(q2 · f)
```

where `f ≤ 0` encodes the constraint. For obstacle avoidance, the constraint is elliptical:

```
1 - || x_{t+k|t} - x_{p,t+k|t} ||²_P < 0
```

The matrix P incorporates vehicle dimensions (length `l`, width `d`), a static safety margin `ssafe`, and a dynamic safe headway `tsafe = 2s`. This 2-second time headway is a fixed design choice tuned for the test scenarios.

**Step 4 — Adaptive weight adjustment.** If a candidate trajectory violates collision constraints, the algorithm reduces the terminal tracking weight `QN` by factor `mQN = 20`, reduces stage control weights `R` by factor `mR = 5`, and increases the obstacle barrier steepness `q2` by `mq2 = 0.1`. This makes the optimizer prioritize collision avoidance over time-optimality for that candidate.

**Step 5 — Trajectory selection.** All K trajectories are evaluated for safety (Equation 10 in the paper — an explicit collision check). Among collision-free trajectories, the one minimizing a reachability-weighted cost-to-go is selected (Equations 12–13). If no trajectory is collision-free, the least-violating one is chosen as a fallback.

The key parameters governing tracking-vs-convergence behavior are:

| Parameter | No obstacles | With obstacles |
|-----------|-------------|----------------|
| Tracking ratio ε | 0.4 | 1.0 |
| Convergence ratio ψ | 0.0 | 0.03 |
| Safety margin ratio ϵ | 5 | 5 |
| Prediction ratio γ | 2 | 2 |

The shift from ε₁=0.4 to ε₂=1.0 when obstacles are present is notable: it essentially doubles down on tracking historical states (which implicitly encode a time-optimal line) when obstacle pressure increases, rather than relaxing toward a more conservative trajectory.

### Why "Unified"

The prior LMPC-based approaches required an explicit mode switch: when no obstacle was nearby, optimize lap time; when an obstacle was detected, switch to a separate overtaking controller. This switch creates a discontinuity in the cost landscape and can cause oscillatory behavior near the switching boundary. IteraOptiRacing subsumes both objectives into a single cost with adaptive weights, so the transition is smooth and the optimizer naturally discovers the best tradeoff at each step.

---

## Results

### Computational Performance

- Update rate: **>33 Hz** on Intel i7-8700 (3.20 GHz, 6-core)
- All baselines: 15–30 Hz on identical hardware
- The K=32 parallel iLQR problems are solved concurrently, exploiting multi-core parallelism. This is the primary source of the throughput advantage.

### Racing Performance (300 randomized tests, 3 track geometries, 9 obstacle vehicles)

**Speed range V1 (0.2–0.4 m/s obstacles):**
- IteraOptiRacing: success concentrates at maximum overtakes, zero outright failures
- All baselines: nonzero failure rates, lower mean overtake counts

**Speed range V2 (0.4–0.6 m/s obstacles):**
- IteraOptiRacing: peak at 6–7 vehicle overtakes per lap
- Baselines: peak at lower counts

**Speed range V3 (0.6–0.8 m/s obstacles — highest difficulty):**
- IteraOptiRacing: degraded but still superior across all metrics
- Multiple baselines: 0% success rate on at least one track geometry

**Full-vehicle-overtaking success rate:**
- IteraOptiRacing: **100%** on L-shaped, M-shaped, and elliptical tracks (~51m length each)
- LMPC with local re-planning: 0% on at least one track
- LMPC with slacked terminal state: 0% on at least one track
- LMPC with convex hull slack: 0% on at least one track

The tracks used are small-scale (51m circumference), suggesting a ground vehicle testbed rather than full-scale motorsport. Obstacle speeds of 0.2–0.8 m/s are correspondingly slow.

---

## Relevance to Our System

Our system uses min-snap polynomial trajectories with TOPP-RA speed retiming, Iterative Learning Control (ILC) to reduce tracking error across laps, and post-optimization inflation factors that are progressively reduced as ILC improves accuracy. The accuracy-speed tradeoff is central: as ILC drives tracking error down, we want to tighten safety margins and allow faster trajectories.

IteraOptiRacing is relevant along several dimensions:

**1. The terminal-set / historical-data mechanism as a drone racing primitive.**
The i2LQR approach of maintaining a dataset of past states and selecting K-nearest candidates as terminal targets is structurally analogous to what our ILC does: past trajectory data is used to shape future control. The key difference is that i2LQR operates online within a receding-horizon MPC, while our ILC operates offline between laps. A hybrid — running receding-horizon optimization that seeds its terminal cost from ILC-corrected waypoints — could capture both the within-lap reactivity of MPC and the cross-lap refinement of ILC.

**2. Exponential barrier functions for constraint softening.**
Our current TOPP-RA retimer enforces hard velocity/acceleration constraints. Adding exponential barriers as soft constraints in the trajectory optimization step could allow the solver to gracefully trade off against gate margins when the trajectory is near infeasible, rather than hard-failing or requiring manual slack parameters.

**3. Parallel trajectory candidates for gate sequencing.**
When our gate sequencer faces ambiguity (e.g., two candidate gate orderings), running parallel trajectory evaluations (analogous to the K-candidate iLQR) and selecting the one with lowest predicted tracking cost would be a natural extension. This could replace the current heuristic pass-through detection.

**4. Adaptive weight adjustment as a safety-margin reduction strategy.**
The paper's scheme of reducing `QN` and increasing `q2` when constraints are violated is effectively a runtime margin adaptation. Our ILC-driven inflation factor reduction works the same way but happens between laps rather than within a lap. Integrating an online version — where the inflation factor decreases as ILC confidence grows during a race — could accelerate margin reduction beyond what the current per-lap update achieves.

**5. Update rate considerations.**
The paper achieves >33 Hz on a 6-core i7. Our pipeline targets >100 Hz. Running K=32 independent optimization problems at our required loop rate would require either significant parallelism or a much shorter horizon than N=12. This constrains the direct applicability of the full i2LQR stack for our system.

**Key limitation for direct transfer**: The paper addresses ground vehicle racing with a dynamic bicycle model. Our drone operates in 3D with full SE(3) attitude dynamics, making the 6-state Frenet model inapplicable without significant extension. The elliptical obstacle model (2D cross-section) also does not generalize to 3D gate passage constraints.

---

## Actionable Takeaways

1. **Adopt the K-nearest-neighbor terminal set construction for MPC initialization.** Rather than using the nominal min-snap trajectory waypoints to initialize the MPC horizon, query the ILC-corrected state dataset for the K nearest historical states and use them as candidate terminal targets. This directly leverages ILC's accumulated accuracy improvements within each lap's receding-horizon optimization. Implementation: after each ILC iteration, export the corrected state trajectory to a lookup table indexed by arc-length progress `sc`. At runtime, retrieve K=8–16 neighbors per timestep and run parallel short-horizon optimizations.

2. **Replace hard constraint violations in TOPP-RA with exponential barrier softening.** When TOPP-RA retiming fails (trajectory exceeds thrust/velocity bounds), currently the system falls back to a conservative retime. Instead, add an exponential barrier term to the TOPP-RA QP that allows soft violation with increasing cost, producing a feasible but slightly constraint-violating trajectory rather than a hard failure. This is particularly useful for gate-passage constraints where a small lateral violation is preferable to a large speed reduction.

3. **Implement parallel trajectory selection for gate ordering ambiguity.** When the gate sequencer detects competing candidate gate orderings (e.g., from noisy PnP estimates), evaluate both orderings through a short horizon MPC or trajectory evaluation, and select based on predicted cost-to-go. The i2LQR selection logic (Equations 12–13) provides a ready-made recipe.

4. **Design an online inflation-factor scheduler analogous to the adaptive weight mechanism.** Track the running mean tracking error per gate across the current lap. When error is below the ILC-target threshold at a gate, reduce the downstream inflation factor for that gate segment in real time. This is equivalent to the paper's `mQN` reduction when constraints are satisfied.

5. **Benchmark the K-parallel MPC approach at 100 Hz.** Use Python multiprocessing (not threading — GIL-bound) with K=8 horizon-N=6 problems. Profile whether this is feasible before committing to the architecture. If not feasible at 100 Hz, consider K=4 or falling back to the terminal-set insight without full parallelism (i.e., select the single best historical terminal state rather than optimizing over K candidates).

---

## Limitations & Caveats

**1. Ground vehicle, small-scale testbed.** All experiments use a dynamic bicycle model on 51m tracks with obstacle speeds of 0.2–0.8 m/s. The translation to 3D drone dynamics with attitudes, thrust limits, and gate passage constraints requires non-trivial adaptation. The Frenet state parameterization assumes 2D motion on a known track, which does not extend directly to 3D flight paths.

**2. No ablation on K or N.** The paper fixes K=32 and N=12 throughout. It is unclear how sensitive performance is to these choices, or whether smaller values (K=8, N=6) sufficient for a drone's tighter computational budget would preserve the benefits.

**3. tsafe = 2s is a fixed heuristic.** The safe headway of 2 seconds is appropriate for ground vehicles at the tested speeds but would be poorly calibrated for drone racing where relative speeds and collision geometries are very different.

**4. No comparison to model-predictive path integral (MPPI) or sampling-based methods.** The baselines are all LMPC variants. MPPI and other sampling-based MPC approaches have shown competitive performance in racing and would be stronger baselines.

**5. Dataset cold-start problem.** The algorithm requires prior lap data to build the terminal set. In a competition setting with an unfamiliar track, the first lap must use a bootstrap policy (e.g., the nominal min-snap trajectory). The paper does not discuss how degraded performance is during the initial cold-start laps.

**6. Soft constraint reliance.** Converting all non-convex constraints to exponential barriers means constraint satisfaction is never guaranteed — only penalized. In safety-critical applications (gate passage in drone racing), a hard violation during a momentary weight imbalance could cause a crash. A hybrid approach (hard constraints for critical gate passages, soft for track boundaries) would be more appropriate.

**7. Obstacle prediction horizon.** The obstacle trajectory prediction assumes a constant-velocity model (implied by the formulation). In a competitive drone race, opponents may maneuver aggressively, making the 2-second prediction horizon unreliable.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| K | 32 | Number of nearest-neighbor terminal state candidates |
| N | 12 | MPC prediction horizon (timesteps) |
| ε₁ | 0.4 | Tracking ratio (no obstacles) — controls terminal cost weight |
| ε₂ | 1.0 | Tracking ratio (with obstacles) |
| ψ₁ | 0.0 | Convergence ratio (no obstacles) |
| ψ₂ | 0.03 | Convergence ratio (with obstacles) |
| ϵ | 5 | Safety margin ratio for collision ellipse |
| γ | 2 | Prediction ratio for dynamic obstacle headway |
| tsafe | 2.0 s | Safe headway for dynamic obstacle avoidance |
| mQN | 20 | Terminal weight reduction factor when collision detected |
| mR | 5 | Stage control weight reduction factor when collision detected |
| mq₂ | 0.1 | Barrier steepness increase per adaptation step |
| Hardware | Intel i7-8700, 3.20 GHz, 6-core | Reference computation platform |
| Update rate | >33 Hz | Achieved online optimization frequency |
| Track length | ~51 m | Test track circumference (L, M, elliptical geometries) |
| Obstacle count | Up to 9 | Number of dynamic vehicles in stress tests |
| Obstacle speeds | 0.2–0.8 m/s | Three speed ranges tested (V1, V2, V3) |
| iLQR iterations | 2 | Number of backward-forward passes per candidate |

---

## Summary Assessment

IteraOptiRacing is a well-engineered extension of LMPC-based racing to multi-agent dynamic scenarios, with the core novelty being the unified cost formulation that eliminates mode-switching. The parallel K-candidate architecture is practically important — it converts a combinatorial selection problem into embarrassingly parallel optimization, which is a clean engineering insight.

For our drone racing stack, the most directly useful ideas are: (a) seeding MPC terminal costs from ILC-corrected historical data, (b) exponential barrier softening for near-infeasible trajectory segments, and (c) the online adaptive weight adjustment as a complement to our between-lap inflation-factor reduction. The full i2LQR framework cannot be transplanted directly due to the 2D ground vehicle model assumption and the computational budget mismatch at 100 Hz, but the constituent ideas are modular and applicable in isolation.
