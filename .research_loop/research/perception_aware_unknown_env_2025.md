# Perception-aware Planning for Quadrotor Flight in Unknown and Feature-limited Environments

- **URL**: https://arxiv.org/abs/2503.15273
- **Authors**: Chenxin Yu, Zihong Lu, Jie Mei, Boyu Zhou
- **Year**: 2025
- **Venue**: IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025); arXiv preprint March 2025 (v2: July 2025)
- **Code**: https://github.com/Robotics-STAR-Lab/LA-Planner

---

## Key Contribution

This paper addresses autonomous quadrotor navigation in unmapped, visually degraded environments — spaces with sparse or unevenly distributed visual features where standard VIO-based state estimation drifts catastrophically. The core problem is a fundamental tension: the camera's limited FOV must simultaneously serve two competing masters — tracking existing features for localization and scanning unknown regions for obstacle detection and exploration. Existing methods either (a) use fixed head-forward yaw (fast but blind to localization needs) or (b) compute FOV-constrained trajectories offline with full environment knowledge (correct but inapplicable when the map is unknown).

The primary technical contribution is a unified perception-aware online replanner consisting of three tightly coupled components: (1) a **Viewpoint Transition Graph** that adaptively selects intermediate waypoints balancing localizability and exploration progress; (2) a **localizable corridor** construction method that encodes per-node yaw feasibility bounds analytically, enabling cheap gradient-based optimization without repeated visibility evaluations; and (3) a **yaw trajectory optimizer** using MINCO³ parameterization with a logistic transformation to handle angular bounds as unconstrained optimization. Together, these reduce replanning runtime by 1.9–8.5× over prior APACE-family methods while achieving 100% navigation success in scenarios where all baselines fail.

---

## Technical Approach

### Problem Formulation

The system navigates a quadrotor from start to goal using only onboard RGB-D camera + IMU. Three simultaneous constraints must be satisfied at all times:
1. **Collision avoidance** with unknown obstacles
2. **Localizability**: VIO tracking error must remain bounded (achievable by keeping enough co-visible features in FOV between consecutive trajectory nodes)
3. **Mission progress**: navigation must complete within acceptable time limits

Quadrotor dynamics are assumed differentially flat — position and yaw `(p, ψ)` are the planning outputs; attitude, thrust, and angular rates follow analytically.

### Module 1: Adaptive Local Target Selection via Viewpoint Transition Graph

The planner maintains two types of candidate viewpoints:

**Frontier-based viewpoints** `v^fro = (p, ψ)` are sampled around frontiers (boundaries of known/unknown space):

```
p_i,j^fro = FC̄_i + r_k[cos θ_j, sin θ_j, 0]^T + [0, 0, z_m]^T
```

Each candidate must satisfy:
- At least `V_thr` known features visible in its FOV
- At least `FC_thr` frontier cells of cluster i visible in its FOV

**Feature-based viewpoints** `v^fea` bridge feature clusters, sampling positions from the intersection region of two cluster visibility zones `R_i ∩ R_j` such that both cluster centroids are simultaneously visible. These viewpoints allow the drone to transition between feature-rich zones without losing localizability.

The two viewpoint types are combined into a directed graph with edge costs:

```
C(v_a, v_b) = max(‖p_a - p_b‖/v_max, ‖ψ_a - ψ_b‖/ψ̇_max)
```

This is the minimum time considering translation and rotation as parallel motions. An edge is only created if the pair satisfies a co-visibility constraint:

```
|Ω_a ∩ Ω_b| > C_thr
```

where `Ω_a, Ω_b` are the feature sets visible at each viewpoint.

**Graph edge topology**: current state → feature viewpoints (unidirectional), feature ↔ feature (bidirectional), feature → frontier viewpoints (unidirectional). This topology enforces localizability at every transition.

**Viewpoint selection via Dijkstra**: the selected frontier viewpoint `v^fro*` maximizes:

```
R = ω_p · (-t_sg) + ω_n · G_nav
```

where `t_sg = t_sl + t_lg` is the total estimated travel time (current → frontier → goal), and:

```
G_nav = w · ℓ = (v_goal · v_yaw)/(‖v_goal‖ ‖v_yaw‖) · ℓ
```

`w` is the cosine alignment between goal direction and yaw direction; `ℓ` is the count of frontier cells visible. This reward preferentially selects viewpoints that explore toward the goal rather than in arbitrary directions.

### Module 2: Position Trajectory Generation

A degree-3 uniform B-spline is optimized over control points `C = {C_0, ..., C_M}`:

```
arg min_{C, Δt} f_s + w_t · t_p + λ_c · f_c + λ_d(f_v + f_a)
```

- `f_s`: smoothness (integral of squared jerk/snap)
- `t_p`: trajectory duration (minimize time)
- `f_c`: collision cost (penalize proximity to obstacles)
- `f_v + f_a`: dynamic feasibility (velocity and acceleration limits)

Initial path provided by Kinodynamic A* search.

### Module 3: Yaw Trajectory Generation (Three-Phase)

This is the most novel and directly applicable component. The problem of generating a perception-aware yaw trajectory is decomposed into:

**Phase 1: Localizable Corridor Construction**

For each intermediate trajectory node `N_i = (p_i, a_i, ψ_i)`, compute a **valid yaw interval** `(l_i, u_i)` such that any yaw within the interval keeps all critical features in FOV.

Critical features `L_i^cri` are selected using co-visibility analysis:
- Between nodes `N_{i-1}` and `N_i`: select top `C_thr` features by co-visibility score → set `K_1`
- Between nodes `N_i` and `N_{i+1}`: select top `C_thr` features → set `K_2`
- `L_i^cri = K_1 ∪ K_2`

Co-visibility score for feature `f` between nodes:
```
μ(f, N_{i-1}, N_i) = v(f, T_{i-1}) · v(f, T_i)
```
where `v(·,·)` is a differentiable visibility metric (smooth sigmoid-style function of camera-to-feature angles vs. FOV half-angles).

The corridor bounds are then found by bidirectional incremental search from the initial yaw `ψ_i`: step clockwise and counterclockwise until any critical feature exits FOV, recording the boundary angles. This converts a per-optimization-step visibility check (expensive) into a one-time precomputation.

**Phase 2: Initial Path on Yaw Graph**

Time between consecutive position samples:
```
t_ψ = t_p · (⌊t_p/t_f⌋ + 1)^(-1)
```
where `t_f = 0.35s` is the maximum VIO drift time without a visual update. This ensures sampling frequency is sufficient to prevent localization from going stale between yaw graph nodes.

A graph is built over uniformly sampled yaw angles at each position, with edges constrained by:
- Co-visibility: `|L_{i,j} ∩ L_{i+1,k}| > C_thr`
- Smoothness: `‖ψ_{i,j} - ψ_{i+1,k}‖ < ψ̇_max · t_ψ`

Edge cost:
```
c_s(S) = Σ (ψ_{i+1} - ψ_i)² · [1 + exp(-μ_e · c_e(N_i, N_{i+1}))]
```

The exploration gain `c_e` modulates how much the optimizer tolerates yaw changes: if the next node reveals many new frontiers or points toward the goal, larger yaw changes are penalized less.

Dijkstra yields the initial yaw sequence `Ω_ψ`.

**Phase 3: MINCO³ Yaw Trajectory Optimization**

The corridor bounds `(l_i, u_i)` from Phase 1 convert the bounded yaw optimization into an unconstrained problem via logistic transformation:
```
r_i = (ψ_i - l_i)/(u_i - l_i)
q_i = -log((1 - r_i)/r_i)
```

MINCO³ is then optimized over the unconstrained `q_i` variables with cost:
```
J_ψ = Σ [J_s(ψ_i) + λ_d·J_d(ψ_i) - λ_e·J_e(ψ_i)]
```

- `J_s`: yaw smoothness
- `J_d`: dynamic feasibility (yaw rate/acceleration limits)
- `J_e`: exploration capability (visible frontier cells from yaw `ψ_i`, using differentiable visibility metric filtered for unblocked, not-yet-observed cells)

---

## Results

### Simulation Benchmarks (100 runs per scenario)

| Scenario | Method | Success Rate | Nav Time (s) | Replanning Runtime (ms) |
|----------|--------|--------------|--------------|------------------------|
| Scene (a) | Zhang [3] | 23/100 | 38.26 | 102.60 |
| Scene (a) | APACE-R | 0/100 | — | 408.33 |
| Scene (a) | APACE-E | 42/100 | 62.87 | 391.04 |
| **Scene (a)** | **Ours** | **100/100** | **42.41** | **45.21** |
| Scene (b) | Zhang | 0/100 | — | 54.17 |
| Scene (b) | APACE-R | 49/100 | 18.69 | 301.78 |
| Scene (b) | APACE-E | 100/100 | 30.80 | 201.44 |
| **Scene (b)** | **Ours** | **100/100** | **15.76** | **23.12** |
| Scene (c) | Zhang | 0/100 | — | 81.41 |
| Scene (c) | APACE-R | 0/100 | — | 772.66 |
| Scene (c) | APACE-E | 0/100 | — | 472.90 |
| **Scene (c)** | **Ours** | **92/100** | **110.95** | **62.33** |

Runtime advantages: 1.9–8.5× faster replanning than APACE methods; 1.1–4.7× faster than Zhang. Scene (c) involves a long corridor with dead-ends, sharp turns, and staircases — all baselines achieve 0% success.

### Ablation: Viewpoint Transition Graph

Removing the graph and assuming direct frontier access causes complete failure in dead-end scenarios (co-visibility constraint cannot be satisfied along the path, trajectory generation fails, drone trapped). The graph provides critical path-finding through feature-constrained space.

### Real-World Experiments (Intel NUC i7-1260P, ORB-SLAM3)

| Scenario | Path Length | Goal Error | VIO RMSE |
|----------|-------------|------------|----------|
| Dead-end (12m goal) | 27.66m | 0.79m | 0.53m |
| Obstacle-rich partially-lit (12m goal) | 19.21m | 0.60m | 0.37m |

Both experiments conducted in dark environments. System runs in real-time on consumer-grade hardware.

---

## Relevance to Our System

Our current bottleneck is the `_relax_for_fov()` method in `/Users/kenichi.matsuo/Personal/killallhumans/planning/trajectory_optimizer.py`. This method runs up to 5 iterations of trajectory generation, each time evaluating `add_fov_constraints()` and inflating times on high-curvature segments by 10% per iteration. This post-hoc relaxation adds approximately 3.5 seconds to a baseline 12-second trajectory — a ~29% overhead — because it operates after the fact on a trajectory that was not planned with FOV feasibility in mind from the start.

This paper's approach is architecturally superior: it **encodes the localizable corridor before optimization**, meaning the yaw optimizer never generates a trajectory that violates FOV constraints. The iterative outer loop is eliminated entirely. The key insight for our use case:

**The localizable corridor converts a constrained yaw problem into an unconstrained one.** Rather than iteratively slowing down segments until the gate stays visible, we precompute the valid yaw range `(l_i, u_i)` at each trajectory node and optimize freely within it via the logistic reparameterization. No penalty weight tuning, no iterative inflation, no post-hoc penalty evaluation.

Our system's pipeline maps onto this as follows:

- `trajectory_optimizer.py` → `_relax_for_fov()` is the direct target for replacement. The localizable corridor approach would replace the entire `_relax_for_fov()` + `add_fov_constraints()` call chain.
- `FOVConfig` in `trajectory_optimizer.py` (horizontal_fov_rad=90°, vertical_fov_rad=60°, margin_fraction=0.8) feeds directly into the corridor construction step — these are exactly the half-angle bounds `(l_i, u_i)` needed.
- The position trajectory generation (B-spline in the paper, polynomial min-snap in our system) is orthogonal to the yaw corridor approach; they compose cleanly.
- Our system does not currently do exploration (gates are known), so the exploration terms `J_e`, `G_nav`, and `c_e` are irrelevant. The corridor construction and MINCO³ yaw optimization are purely the perception-aware pieces.

For drone racing, the "localizability" constraint maps onto "gate visibility": the next gate must remain in FOV during approach. Instead of tracking visual features for VIO, we want the upcoming gate centroid to stay within `(l_i, u_i)`. The corridor construction algorithm applies directly with this substitution.

---

## Actionable Takeaways

1. **Replace `_relax_for_fov()` with proactive corridor-based yaw planning.** Compute per-node yaw bounds `(l_i, u_i)` guaranteeing gate visibility before running any trajectory optimization. Eliminate the 5-iteration outer loop entirely. Expected savings: the full 3.5s overhead, possibly recovering to near-optimal segment times.

2. **Implement the bidirectional incremental yaw corridor search.** At each sampled trajectory point, start from the initial yaw (e.g., pointing toward next gate centroid) and sweep ±Δψ in small steps until the gate exits the camera FOV half-angles (90°×0.8/2 = 36° horizontal, 60°×0.8/2 = 24° vertical). Record the boundary angles. This is O(N × FOV/Δψ) — negligibly cheap relative to trajectory generation.

3. **Use the logistic reparameterization `q = -log((1-r)/r)` to convert bounded yaw optimization to unconstrained.** This allows gradient-based optimizers (scipy BFGS/L-BFGS-B) to work without inequality constraints on yaw, eliminating penalty weight sensitivity and numerical conditioning issues. Can be integrated into `_optimize_time_allocation()`.

4. **Adopt the co-visibility-driven yaw graph search (Phase 2 of paper) as an initialization for yaw optimization.** For our system: instead of co-visible VIO features, use gate centroid visibility. The graph nodes are sampled yaw angles at each trajectory point; edges require that the approaching gate stays visible. Dijkstra gives a feasible starting yaw sequence for the MINCO³/polynomial optimizer.

5. **Apply the `t_ψ` sampling formula** to set trajectory evaluation density. With `t_f = 0.35s` as the maximum inter-update interval, the number of yaw evaluation points is `⌊t_p/0.35⌋ + 1`. For a 12–16s trajectory, this gives 34–46 evaluation points — adequate for smooth gate tracking without oversampling.

6. **Decouple yaw trajectory optimization from segment time inflation.** The paper shows that yaw and position can be optimized independently (position first, then yaw within the corridor). Our current system entangles them: FOV violations cause segment time inflation, which changes the position trajectory, which changes FOV angles. Breaking this coupling allows faster, more predictable convergence.

7. **Consider MINCO³ for yaw parameterization as a replacement for the current polynomial representation.** MINCO³ is computationally efficient for minimum-control trajectory optimization and handles time allocation natively. See the MINCO paper (Wang et al., 2022) for implementation reference.

---

## Limitations & Caveats

**Unknown environment assumption does not match racing.** The paper's fundamental premise is that the environment map is unknown and must be built online via RGB-D frontier detection. In drone racing, all gate positions and the track layout are known in advance. The viewpoint transition graph, frontier-based viewpoints, and exploration gain terms (`G_nav`, `J_e`, `μ_f`) are entirely inapplicable and would add unnecessary complexity.

**No velocity constraint on gate traversal.** The paper's V_max = 1.5 m/s is extremely conservative compared to racing speeds (10–20+ m/s). At racing speeds, the localizable corridor width (valid yaw range) narrows because the drone tilts more aggressively, reducing the angular window in which the gate stays visible. The corridor construction must account for attitude tilt from aerodynamic forces and thrust, not just position-based camera geometry.

**Assumption of feature-based VIO.** The paper builds around ORB-SLAM3 and feature co-visibility for state estimation. Our system uses an EKF fused with gate PnP corrections — the "localizability" constraint maps more naturally to gate visibility than to arbitrary feature co-visibility. The corridor construction (per-node yaw bounds for gate FOV) is directly usable, but the feature-cluster graph structure is not.

**Real-world experiments use low-speed, coarse platforms.** Navigation time ~27–110s over 12m distances indicates this is a slow exploration system. At racing speeds, replanning latency must be sub-10ms; the paper's 23–62ms per trajectory generation is acceptable for moderate replanning rates but would need profiling in our stack.

**Corridor width may collapse at high curvature.** At tight turns, the body tilts significantly. If the gate is nearly perpendicular to the trajectory at a sharp turn, the valid yaw range `(l_i, u_i)` may be very narrow or empty, making corridor-based planning degenerate to time inflation anyway. This is the exact case `_relax_for_fov()` currently handles; a hybrid approach may be needed for extreme curvature.

**No treatment of partial gate occlusion.** The paper assumes features are either in FOV or not (binary visibility with soft sigmoid approximation). For gate traversal, partial occlusion by the gate frame itself, or by the drone body at high tilt angles, is not addressed.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| `t_f` | **0.35 s** | Maximum VIO drift interval without visual update; drives yaw evaluation point density via `t_ψ = t_p / (⌊t_p/t_f⌋ + 1)` |
| `v_max` | 1.5 m/s | Max translational velocity in experiments (not applicable to racing) |
| Edge cost | `max(‖Δp‖/v_max, ‖Δψ‖/ψ̇_max)` | Parallel translation+rotation time cost for viewpoint transitions |
| Logistic bound mapping | `r_i = (ψ_i - l_i)/(u_i - l_i)`, `q_i = -log((1-r_i)/r_i)` | Converts bounded yaw interval to unconstrained variable for gradient optimization |
| Co-visibility threshold | `C_thr` (tunable) | Minimum number of co-visible features required on each graph edge; analog for us is "gate centroid must be within camera FOV" |
| Corridor search step | Not specified; implied small (likely 1–5°) | Bidirectional angular sweep step for bound computation |
| Reward weights | `ω_p`, `ω_n` (tunable) | Time efficiency vs. navigation gain in viewpoint selection (exploration-specific; not applicable) |
| Yaw cost weights | `λ_d`, `λ_e`, `μ_e`, `μ_f`, `μ_g` (tunable) | Yaw trajectory optimizer weights; `λ_d` (dynamic feasibility) and `λ_e` (exploration) are relevant analogs for racing: replace `λ_e`·J_e with gate-visibility reward |
| Smoothness modulation | `exp(-μ_e · c_e)` in yaw path cost | Exponential gate on yaw change penalty — high exploration gain permits larger yaw swings |

The most immediately usable constant is `t_f = 0.35s`. This directly sizes the yaw trajectory evaluation density and provides a principled bound on how long a drone can coast without a visual update before VIO accuracy degrades unacceptably. For our system, the analog is: how long between gate-visibility confirmations. If the trajectory produces 0.35s windows of gate-in-FOV, that is the minimum acceptable replanning cadence.
