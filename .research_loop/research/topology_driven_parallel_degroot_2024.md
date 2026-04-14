# Topology-Driven Parallel Trajectory Optimization in Dynamic Environments

**URL:** https://arxiv.org/abs/2401.06021
**Authors:** Oscar de Groot, Laura Ferranti, Dariu M. Gavrila, Javier Alonso-Mora
**Year:** 2024 (submitted January 11, 2024; revised October 4, 2024)
**Venue:** IEEE Transactions on Robotics

---

## 1. Key Contribution

The paper introduces T-MPC (Topology-driven Model Predictive Control), a framework that explicitly exploits homotopy class diversity to escape local minima in trajectory optimization. The central insight is that gradient-based trajectory optimizers make high-level routing decisions implicitly through initialization — whichever starting guess the optimizer receives, it tends to stay in the same topological neighborhood (same side of an obstacle). When the initialized guess lands in a poor homotopy class, the solver converges to a locally optimal but globally suboptimal trajectory, with no mechanism to escape.

T-MPC addresses this by running P independent local optimizers in parallel, each seeded with a guidance trajectory from a distinct homotopy class (e.g., pass left of obstacle vs. pass right). A receding-horizon selector then executes the globally cheapest feasible result at each control step. This transforms what was a single-shot gradient descent into a structured parallel search over the topological space of trajectories, providing a probabilistic guarantee of global optimality across homotopy classes (called "Homotopy Globally Optimal" or HGO), subject to mild conditions.

---

## 2. Technical Approach

### 2.1 Global Guidance Planner

The guidance planner constructs a sparse roadmap (Visibility-PRM) in the augmented state space R^2 x [0,T], encoding both spatial position and time to capture dynamic obstacle motion. Obstacle-time cylinders represent where pedestrians will be over the planning horizon. A depth-first search over this roadmap finds P shortest paths, each confirmed to be in a distinct homotopy class via H-signature checks (comparing winding numbers of trajectories around obstacle centroids).

Multiple goal positions arranged on a 5x5 grid are used to ensure coverage when single goal positions become blocked. This is critical in cluttered dynamic environments where direct paths are unavailable.

### 2.2 Homotopy Class Enforcement (Local Planner Constraint)

Each of the P local planners receives one guidance trajectory and solves the standard MPC problem augmented with a linear half-space constraint:

    g_H(x_k, o_k^j, tau_{i,k}) <= 0

This constraint is constructed from the signed separation between the guidance trajectory and each obstacle at each time step. The robot's optimized path must remain on the same side of each obstacle as its assigned guidance trajectory. The constraint relaxation factor beta in [0,1] controls how tightly the trajectory must follow the guidance; beta near 0 leaves the constraint inactive at obstacle boundaries, allowing the optimizer more freedom while still enforcing the topological class.

### 2.3 Parallel Execution and Selection

All P local optimizers run in parallel threads, each with a hard time budget (50ms deadline in experiments, 20Hz control frequency). A non-guided planner (vanilla MPC, identical to what you would run without T-MPC) is included as planner P+1, ensuring T-MPC never regresses below baseline MPC performance. The trajectory with lowest cost J* is selected for execution. An optional consistency weight c_i in [0,1] biases the selector toward the previously chosen homotopy class to reduce chattering:

    J_selected = J*_i + c_i * (switching_penalty)

### 2.4 Theoretical Guarantees

- **Theorem 1 (HGO):** When homotopy constraints are inactive at convergence, the guidance planner covers all feasible classes, and minimal-cost selection is used, T-MPC returns the global optimum over homotopy classes.
- **Theorem 2:** Including the non-guided planner as a fallback guarantees T-MPC cost is always <= standard MPC cost. It cannot be worse than running a single optimizer.

### 2.5 Handling Dynamic Obstacles

Unlike static-obstacle homotopy methods, T-MPC uses winding numbers in the space-time cylinder, tracking the relative angle trajectory between robot and moving obstacle over the horizon. Two trajectories are in the same dynamic homotopy class if they go around the pedestrian's space-time tube from the same side. This extends naturally to multiple obstacles (2^M possible classes with M obstacles), though the guidance planner only explores P << 2^M of them.

---

## 3. Results (Quantitative)

All experiments used 200 simulation trials, compared against LMPCC (baseline single-optimizer MPC), TEB (Time Elastic Band), motion primitives, and visibility-based planners.

**Table II — Interactive Navigation (deterministic pedestrian motion):**
| Scenario | Method | Duration (s) | Safety | Runtime |
|---|---|---|---|---|
| 4 pedestrians | T-MPC++ | 13.0 (±0.1) | 100% | 19.4ms |
| 8 pedestrians | T-MPC++ | 13.2 (±0.6) | 96% | 21.4ms |
| 12 pedestrians | T-MPC++ | 13.6 (±1.0) | 93% | 20.1ms |
| 12 pedestrians | LMPCC (baseline) | higher variance | 90% | — |

**Table III — Uncertain Obstacle Motion (chance-constrained variant TCC-MPC++):**
| Risk (epsilon) | Duration (s) | Safety | Runtime |
|---|---|---|---|
| High (0.1) | 14.1 (±0.8) | 96% | 34.4ms |
| Medium (0.01) | 15.1 (±1.4) | 93% | 35.8ms |
| Low (0.001) | 16.1 (±1.3) | 97% | 38.1ms |

**Guidance planner alone:** ~5ms average across all environments.
**Total framework (4 parallel MPC instances):** 18–21ms.
**Uncertain variant (5 parallel instances):** 34–38ms.
**Control frequency:** 20Hz (50ms per cycle).

Key takeaway: T-MPC++ achieves faster task completion (lower duration) with higher safety and lower variance than all baselines, especially in crowded scenarios where local minima proliferate.

---

## 4. Relevance to Our System

Our system uses two separate L-BFGS-B optimizations:

1. **`racing_line.py`** — optimizes lateral gate offsets (2D positions within gate openings) using `scipy.optimize.minimize` with L-BFGS-B. The comment in the code explicitly notes that `smooth_weight=0.40` "steers the racing line L-BFGS into a qualitatively smoother local minimum."

2. **`trajectory_optimizer.py`** — optimizes segment time allocation with L-BFGS-B on log-time variables, and separately applies a FOV penalty.

Both optimizations are single-shot gradient descent from a single initialization — exactly the failure mode T-MPC addresses. When a race track has a tight chicane or a gate approached at a steep angle, the optimizer will converge to whichever local minimum is closest to the initial guess, with no ability to consider qualitatively different routing choices.

The "homotopy classes" analogy in drone racing is: for each gate, the drone can approach from slightly left-of-center vs. right-of-center, or enter the gate at a shallower vs. steeper angle. These produce qualitatively different polynomial spline shapes (different curvature, velocity profiles). The single-initialization L-BFGS-B always collapses to one choice.

Unlike T-MPC's dynamic obstacle avoidance context, our obstacles are fixed (gates are static, track is predetermined). This actually simplifies the topology computation — the homotopy classes for a racing line are primarily defined by the lateral position within each gate, and the space is lower-dimensional. The parallel optimizer idea maps cleanly: run N_parallel instances of L-BFGS-B with diverse initializations across the gate-offset space, then select the trajectory with lowest cost.

---

## 5. Actionable Takeaways (Numbered)

1. **Replace single-start L-BFGS with multi-start L-BFGS in `racing_line.py`.** Initialize P=4 to 8 instances from diverse lateral offset vectors (e.g., uniformly spaced from left-max to right-max within each gate). Run in parallel using `concurrent.futures.ThreadPoolExecutor`. Select the result with minimum cost. This directly mirrors T-MPC's core idea at negligible cost — scipy L-BFGS-B for our track completes in ~1ms, so 8 parallel runs still finish in under 10ms.

2. **Diversify segment-time initialization in `trajectory_optimizer.py`.** The time allocation L-BFGS-B in `_optimize_time_allocation` starts from a single time guess. Run 3–5 starts from different initial total-time budgets (e.g., 0.7x, 1.0x, 1.3x of current default) and take the minimum-cost result.

3. **Add a "topological fingerprint" to detect and prevent local minimum collapse.** After optimization, compute the lateral offsets relative to each gate center. If two consecutive planning calls produce identical offsets within epsilon, perturb the initialization and re-optimize once. This is analogous to T-MPC's consistency propagation step.

4. **Use a coarse global search to seed the fine L-BFGS.** T-MPC's guidance planner uses a lightweight graph search (PRM with n<100 samples, ~5ms) to identify distinct homotopy-class seeds before passing them to the expensive local optimizer. In our context, a grid search over lateral offsets (e.g., 5-point grid per gate) costs ~0.1ms and provides much better initialization diversity than a single fixed starting point.

5. **Accept regression prevention through fallback.** Following Theorem 2, always include the current best trajectory (from prior planning) as one of the P candidates. This guarantees the multi-start approach never degrades below the current single-start performance — even if some new candidates fail to converge or diverge, the fallback candidate wins selection.

6. **Apply T-MPC's consistency weighting to stabilize replanning.** If the racing line is re-optimized mid-race (e.g., online replanning), use a consistency bias c_i ~ 0.7 to avoid abrupt routing switches caused by noise in cost evaluation. This is especially important if planning is triggered at high frequency.

7. **Monitor per-homotopy-class cost distribution.** Log the costs from all P parallel runs, not just the winner. If the winning class consistently changes across trials, the cost landscape has multiple shallow minima — this signals that tighter regularization or a different objective formulation is needed rather than just more starts.

---

## 6. Limitations and Caveats

**Domain mismatch — dynamic vs. static environment.** T-MPC was designed and validated for dynamic pedestrian avoidance, not for fixed racing tracks. The homotopy machinery (winding numbers, space-time PRM) is more complex than needed for our problem. The core idea (parallel optimization over diverse seeds) is directly applicable, but the formal homotopy framework is not worth porting.

**Pedestrian-scale dynamics are different from drone racing.** T-MPC's robot moves at 2 m/s reference velocity with a 0.725m radius. Our drone reaches 10–20+ m/s with a body radius under 0.3m. The topological complexity for a fast drone in a tight gate sequence is qualitatively different — the primary challenge is curvature and velocity feasibility, not obstacle side-passing.

**P=4 may be insufficient for complex race tracks.** With M gates, the number of combinatorial routing choices grows exponentially. T-MPC acknowledges that 2^M homotopy classes exist with M obstacles; P=4 covers only a tiny fraction in crowded environments. For our fixed track, the gate sequence is known ahead of time, enabling smarter initialization (e.g., exhaustive grid search at each gate independently).

**No formal convergence time bounds.** The 50ms hard deadline for local planners means some MPC instances may be cut off before convergence in the dynamic setting. In our offline planning context this is less of an issue — we can afford longer optimization until convergence.

**Homotopy constraints may over-restrict the optimizer.** The linear half-space constraints that enforce homotopy class can be overly conservative if the guidance trajectory is poorly positioned. For our use case, this translates to: if the seed for a given parallel L-BFGS run is at an extreme offset, the optimizer may be confined to a poor region. This is mitigated by using a fine enough initial grid.

**Theoretical guarantees require strong conditions.** The HGO guarantee requires that homotopy constraints are inactive at the optimum (meaning the optimizer didn't hit the constraint boundary) and that all feasible homotopy classes are covered. Neither condition is guaranteed in practice; the results are empirical rather than provably optimal.

---

## 7. Key Parameters and Constants

From the paper's experimental setup:

| Parameter | Value | Description |
|---|---|---|
| P | 4 | Number of parallel trajectory candidates |
| N | 30 | MPC/guidance planner horizon (time steps) |
| h | 0.05s | Time step (= 20Hz) |
| n (PRM samples) | < 100 (≈30 used) | Roadmap nodes for guidance planner |
| Goal grid | 5 x 5 = 25 goals | Multiple terminal goals for robustness |
| beta | ~0 (range 0–1) | Homotopy constraint relaxation factor |
| c_i | 0.75 | Consistency weight for class selection |
| Reference velocity | 2 m/s | Target speed for cost function |
| Guidance planner time | ~5ms | Wall-clock time for graph search |
| Total framework time | 18–21ms | Including 4 parallel MPC solves |
| Hard deadline | 50ms | Thread kill time (= 1 control period) |

**Cost function weights (local planner):**
| Weight | Value | Penalizes |
|---|---|---|
| w_c | 0.05 | Contour (cross-track) error |
| w_l | 0.75 | Lag error |
| w_v | 0.55 | Velocity deviation |
| w_omega | 0.85 | Angular velocity |
| w_a | 0.34 | Acceleration |

**Mapping to our system for multi-start L-BFGS:**
- P = 4–8 parallel starts (practical starting point)
- Grid over lateral offsets: 5 points per gate from -0.4m to +0.4m
- Fallback: always include prior-best as one candidate (c_0 = 1)
- Consistency bias for replanning: c_i = 0.7 on current winner
- Time budget per L-BFGS call: unconstrained offline; cap at 100ms for online use
