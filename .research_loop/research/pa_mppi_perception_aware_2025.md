# PA-MPPI: Perception-Aware Model Predictive Path Integral Control

- **URL**: https://arxiv.org/abs/2509.14978
- **Authors**: Yifan Zhai, Rudolf Reiter, Davide Scaramuzza
- **Year**: 2025 (submitted September 2025; accepted January 2026)
- **Venue**: IEEE Robotics and Automation Letters (RA-L), 2026; arXiv preprint
- **Affiliation**: Robotics and Perception Group, Department of Informatics, University of Zurich
- **Funding**: EU Horizon Europe AUTOASSESS (grant 101120732); European Research Council AGILEFLIGHT (ERC grant 864042)

---

## Key Contribution

PA-MPPI addresses a fundamental gap in standard MPPI (Model Predictive Path Integral) control: while MPPI handles non-convex free-space navigation well and naturally respects quadrotor dynamics through sampling, it cannot explore unknown terrain to find paths around large obstacles. It gets stuck — the sampled trajectories all go through the obstacle rather than around it — because the standard MPPI cost function only penalizes the current known occupancy and has no mechanism to direct exploration toward useful unknowns.

The paper's contribution is a novel **perception-aware cost function** embedded directly inside the MPPI sampling loop that drives trajectory optimization toward frontiers likely to reveal traversable space toward the goal. This extends MPPI from a local obstacle-avoiding tracker into a global navigation system capable of navigating unseen environments without any external reference trajectory or prior map.

Secondary contributions:
1. Tight integration of a depth-sensor-based 3D occupancy mapping module (OctoMap-based voxel grid, updated at 10 Hz) with the MPPI sampling loop.
2. Demonstration that PA-MPPI can serve as a safe action policy for navigation foundation models (NoMaD) that propose infeasible goal poses due to monocular scale ambiguity.
3. 50 Hz real-time operation on commodity laptop-class hardware (i7-13800H CPU + NVIDIA A1000 GPU, 6 GB VRAM).

The paper's problem domain is **unknown environment navigation** (search-and-rescue, autonomous exploration) — not drone racing. This distinction is critical for understanding applicability to our system.

---

## Technical Approach

### MPPI Background

MPPI is a sampling-based stochastic optimal control method. At each control step:
1. Sample N perturbations around the nominal control sequence.
2. Roll out N trajectories in parallel (GPU-parallelized) using the quadrotor dynamics model.
3. Evaluate a scalar cost L for each rollout.
4. Compute exponentially weighted average over samples to update the nominal control:

```
w^j = exp(-(L^j - L_min) / λ) / Σ exp(-(L^i - L_min) / λ)
```

Temperature λ = 0.02 (very low — strongly concentrates weight on low-cost samples).

The first action in the updated sequence executes at Δt_ctrl = 0.02 s, yielding 50 Hz control.

### Quadrotor Dynamics Model

Full 13-state rigid-body model: position p_WB (3), quaternion q_WB (4), linear velocity v_WB (3), angular velocity ω_B (3). Control input u = [c, ω_B]^T where c is collective thrust. Individual motor thrusts are clipped to enforce actuator saturation. H = 15 prediction steps at Δt_pred = 0.1 s gives 1.5 s horizon.

### Cost Function Architecture: Two-Phase Design

The cost function switches between two regimes based on whether the goal has line-of-sight:

**Phase 1 — Goal Occluded (Exploration Mode):**

Goal attraction (progress reward):
```
ℓ_goal = -c_goal · max(0.0, d_0 - d_k),   c_goal = 0.125
```
where d_0 is initial distance and d_k is distance at step k. This is a small progress incentive only — conservative to avoid committing to paths that may dead-end.

Terminal position weighting: 10× ℓ_goal,H-1 (large terminal bonus for getting close).

Plus perception cost (see below), collision cost, and control effort penalty.

**Phase 2 — Goal Visible (Navigation Mode):**

Once line-of-sight to goal is confirmed, perception cost deactivates and standard navigation costs dominate:
- Strong goal pull: c_goal = 5.0 (40× stronger)
- Progress incentive: ℓ_progress = -c_progress · ||p_i - p_{i+1}|| (reward for moving)
- Velocity damping near the target

Collision cost in both phases: `c_collision · 𝟙{𝒢(p) ≠ 0}` with c_collision = 15.0 — applied as a hard-like binary penalty to occupied voxels.

Action cost: R = diag(0.01, 0.025, 0.025, 0.2) on control input deviation.

### Perception Cost Formulation (The Core Innovation)

The perception cost has two additive components, applied only in Phase 1:

**Component 1 — Point of Interest (PoI) Alignment:**

```
ℓ_PoI = c_PoI · (1 - ⟨x̂_WB, d̂_goal⟩)²,   c_PoI = 5.0
```

where x̂_WB is the camera principal axis (body x-axis) and d̂_goal is the unit vector toward the goal. This penalizes misalignment between camera heading and goal direction, encouraging the camera to face where it needs to look. The cost deactivates within c_thresh = 0.5 m of the goal to avoid degenerate behavior at close range.

**Component 2 — Ray-Tracing Frontier Evaluation:**

A ray is cast from the drone's current position toward the goal using the **3D Digital Differential Analyzer (DDA)** algorithm. The ray intersects the occupancy voxel grid and finds the first non-free voxel r(t*):

- If occupied (wall/obstacle): cost += c_occupied = 2.0
- If unknown (frontier): cost += c_unknown = -4.0 (negative cost = reward)

The sign convention is critical: unknown voxels are rewarded because reaching a position where the ray to the goal enters unknown space means new terrain will be revealed that might open a path forward. Occupied voxels are penalized to avoid committing to trajectories that point directly into walls.

This mechanism "biases trajectories toward informative frontiers to explore unknown regions and advance towards the goal" — it effectively encodes "go to where you can see new things in the goal direction."

### Computational Architecture of the Perception Cost

The ray-tracing evaluation is the most computationally expensive component. The authors make a deliberate approximation: **ray-tracing is only performed at the terminal horizon step k = H-1**, not at every step k = 0...H-1. This reduces the cost of perception evaluation by ~15× (factor of H) at the price of evaluating only where the drone will be at the end of its prediction window, not along the full trajectory.

At N = 17,500 parallel samples with H = 15 steps, full per-step ray-tracing would require 17,500 × 15 = 262,500 DDA queries per control cycle. Terminal-only tracing reduces this to 17,500 DDA queries — manageable on a GPU at 50 Hz.

The map update runs at 10 Hz (decoupled from the 50 Hz MPPI loop), so the occupancy grid queried by MPPI may be 0–100 ms stale. This is an acknowledged approximation.

### Full MPPI Configuration

| Parameter | Value |
|---|---|
| Samples N | 17,500 |
| Horizon steps H | 15 |
| Prediction timestep Δt_pred | 0.1 s |
| Control frequency | 50 Hz |
| Control timestep Δt_ctrl | 0.02 s |
| Temperature λ | 0.02 |
| GPU | NVIDIA A1000 (6 GB VRAM) |
| CPU | Intel i7-13800H |
| Map update rate | 10 Hz |
| Voxel states | Occupied / Free / Unknown |

---

## Results

### Synthetic Obstacle Scenarios

Three benchmark environments of varying difficulty:
- **C-wall**: A curved wall blocking direct path, varying gap width w ∈ {1.0, 2.0, 3.0} m
- **Hole-in-wall**: Quadrotor must find and traverse a narrow aperture
- **4-walls**: Sequential walls requiring multi-step navigation

Baseline comparison is SUPER (state-of-the-art hierarchical planner for unknown environments, replanning at 10 Hz with geometric trajectory tracking).

Selected results:

| Scenario | Metric | PA-MPPI | SUPER |
|---|---|---|---|
| C-wall w=3.0m | Time (s) | 5.14 ± 0.16 | 6.00 ± 0.05 |
| C-wall w=3.0m | Energy (J) | 249.5 ± 9.5 | 286.8 ± 42.0 |
| C-wall w=3.0m | Success rate | 100% | 100% |
| C-wall w=1.0m | Time (s) | 2.91 ± 0.26 | 4.54 ± 0.24 |
| 4-walls | Time (s) | 5.78 ± 0.81 | 7.57 ± 0.07 |

PA-MPPI achieves comparable or better success rates with significantly faster times and lower energy consumption. The energy advantage stems from MPPI sampling dynamically feasible trajectories directly rather than running a geometric path planner first and then tracking it, which often produces conservative or dynamically infeasible references.

Note: EGO-Planner (gradient-based trajectory optimizer, another baseline) fails entirely on harder configurations because its A* front-end deviates too far from the physically feasible region.

### Wind Disturbance Robustness

Random xy-plane wind of 1–3 m/s applied during evaluation. Success rate degrades as wind speed increases above 2.0 m/s. The controller has limited external disturbance compensation capability — no adaptive or L1-style robustness term is included.

### Real-World Hardware Experiments

PA-MPPI successfully navigates a room with a door at 50 Hz. Also demonstrated as action policy for NoMaD (vision-language navigation model), correcting infeasible goal poses in two scenarios including navigation around a ping-pong table obstacle.

Minor collision events observed (mean voxel penetration ~4 cm) due to model simplification in dynamics — the quadrotor's aerodynamic body volume is not fully captured in the cost function.

---

## Relevance to Our System

### Perception Cost Integration: Soft Penalty, Not Hard Constraint

The perception awareness in PA-MPPI is implemented as a **soft penalty within a sampling-based optimizer**, not as a hard constraint and not as post-processing. This is the architectural question asked in the task brief.

Specifically:
- The PoI alignment cost (c_PoI = 5.0) and ray-tracing cost (c_unknown = -4.0, c_occupied = 2.0) are additive terms in a scalar cost function L evaluated per sample.
- Samples that happen to align camera with goal or expose unknown frontiers accumulate lower (more negative) cost and receive higher weights in the exponential averaging.
- There is no projection step, no constraint satisfaction check, and no outer loop enforcing feasibility. Trajectories that violate perception objectives simply receive higher cost and are weighted down, but not excluded.

**Comparison to post-processing FOV relaxation** (our current `_relax_for_fov()` approach):

| Approach | Mechanism | Overhead | Coupling |
|---|---|---|---|
| Our `_relax_for_fov()` | Post-hoc segment time inflation (5 outer iterations, 10%/step) | ~3.5 s added to trajectory | Strong: changes position trajectory |
| PA-MPPI perception cost | Soft penalty inside MPPI sampling, no separate step | Zero marginal overhead for softness | None: no outer loop |
| ETH PA-TOG (perception_aware_planning_eth_2026.md) | NLP soft constraint with slack variable, offline | 68 s offline planning | Decoupled from tracker |
| Localizable corridor (perception_aware_unknown_env_2025.md) | Pre-computed valid yaw bounds, unconstrained optimization within | ~45 ms per replan | Decoupled from position |

PA-MPPI's approach is the only one that adds zero overhead for perception awareness relative to its base planning loop — because MPPI evaluates a cost function over samples regardless, and adding a term to that cost function is nearly free (just additional arithmetic per sample, parallelized on GPU).

### Computational Overhead Analysis

The PA-MPPI perception cost adds:
1. **PoI alignment**: One dot product per sample per step. Negligible.
2. **Ray-tracing**: One 3D DDA query per sample at the terminal step. With N = 17,500 samples, this is 17,500 voxel grid traversals per control cycle, parallelized on GPU. On the A1000, this runs within the 20 ms budget for 50 Hz.

By comparison, our current `_relax_for_fov()` runs up to 5 full trajectory generation iterations after the fact, each involving polynomial coefficient solving and FOV angle evaluation — O(N_segments^3) algebraic operations. The marginal overhead of PA-MPPI's perception cost is orders of magnitude lower.

However, there is a critical caveat: PA-MPPI runs in an environment where it has no pre-planned trajectory. The 17,500 sample evaluations replace trajectory generation entirely. For our system, which uses a pre-planned min-snap polynomial trajectory, the architectural question is whether to replace trajectory tracking with MPPI altogether or to use PA-MPPI-style cost terms inside a different planner.

### Applicability to Drone Racing

PA-MPPI is designed for **unknown environment navigation**, not racing through a known gate sequence. The direct applicability to our competition setting is limited but specific:

**Applicable ideas:**
1. **Sampling-based online replanning with perception cost.** If we implemented MPPI as a trajectory tracker (following the ref trajectory as a reference cost), the PoI alignment term could be included to keep gates in camera FOV during execution. This is the intersection of PA-MPPI's approach and our racing use case.
2. **Soft perception cost as a term in trajectory optimization.** The formulation (weighted dot product between camera axis and gate bearing) is directly usable inside any sampling-based optimizer. If we replace our min-snap solver with MPPI-style trajectory generation (as in `mppi_reference_free_racing_2025.md`), adding the gate-alignment term is straightforward.
3. **Two-phase cost switching.** Our system could adopt the same paradigm: when approaching a gate (goal occluded by upcoming gate frame), emphasize perception alignment; once the gate is visible and confirmed, switch to aggressive racing cost (progress and speed).

**Not applicable:**
- The exploration/frontier mechanism (Phase 1 goal occlusion handling via unknown voxel reward) has no analog in racing — all gates are known in advance.
- The occupancy mapping module (10 Hz OctoMap) is irrelevant — we operate in a known environment.
- The 1.5 m/s operating speed is 10–20× slower than racing; the dynamics regime is entirely different.

### Contrast with Reference-Free MPPI Racing

The `mppi_reference_free_racing_2025.md` paper in our library is the more directly racing-relevant MPPI work. PA-MPPI complements it by showing how to add perception costs into MPPI sampling with minimal overhead. If we were to implement reference-free MPPI racing (N=17,500+ parallel samples, 50 Hz), adding the PoI alignment term from PA-MPPI would be a one-line cost function addition.

---

## Actionable Takeaways

1. **The soft-penalty perception cost in MPPI is zero-overhead relative to baseline MPPI.** If we ever move toward MPPI-based trajectory tracking, adding a gate-alignment term (cosine similarity between camera axis and next gate bearing) costs essentially nothing computationally and would improve gate visibility during high-speed approach.

2. **Two-phase cost switching (occluded → visible) is a clean pattern for managing competing objectives.** Our system currently runs a single trajectory throughout the race. Adapting a two-phase cost (perception-alignment when approaching gate, speed-maximization when gate is confirmed visible) would mirror the PA-MPPI architecture and could reduce tracking error near gates.

3. **Terminal-horizon-only perception evaluation is the right approximation for high-rate MPPI.** Evaluating camera alignment only at k = H-1 (the end of the prediction window) rather than at every step cuts perception cost by factor H = 15. For racing, this means checking "will my camera see the next gate at the end of this 1.5 s prediction window?" — a sensible heuristic for gate visibility planning.

4. **The PoI alignment cost formula is directly transplantable.** `(1 - ⟨x̂_WB, d̂_gate⟩)²` with weight ~5.0 provides a differentiable, bounded (0 to 1) penalty for camera-gate misalignment. In our system: replace `d̂_goal` with `d̂_next_gate` and apply during the approach segment to each gate.

5. **Do not import the architecture wholesale.** PA-MPPI's core innovation — frontier-directed exploration in unknown environments — is orthogonal to racing. The architectural dependency on OctoMap, 10 Hz map updates, and NoMaD integration adds complexity with no benefit in our known-track setting. Extract only the cost function patterns (PoI alignment, phase switching).

6. **The A1000 GPU running 17,500 samples at 50 Hz is a useful datapoint.** Our benchmark target is >100 Hz control loop. If MPPI were adopted, we would need either a more powerful GPU or reduced N to hit this frequency. The paper's hardware budget is approximately 50% of our target frequency on similar-class hardware — suggesting N ~8,000–10,000 might be needed for 100 Hz.

---

## Limitations & Caveats

**Problem domain mismatch.** PA-MPPI is a navigation planner for unknown environments. Our competition uses a fully-mapped track with pre-surveyed gate positions. The central contribution — perception-driven frontier exploration — is inapplicable. The paper's real contribution for our purposes is a recipe for embedding soft perception costs inside MPPI with low overhead.

**Very low operating speed.** Experiments conducted at max ~1.5 m/s. Racing speeds are 10–25 m/s. The quadrotor dynamics regime is fundamentally different: at racing speeds, the drone operates at 30–60° pitch angles, meaning the camera (assumed to point in the body x-direction) is already substantially downward-tilted. The PoI alignment cost, which encourages facing the goal, would need retuning to account for the camera's attitude at racing tilt angles. The cost may actually fight against optimal racing attitude (nose-down = fast) when the goal is ahead-and-horizontal.

**Ray-tracing at terminal step only is an approximation that breaks at high speed.** At 15 m/s with H = 15 steps and Δt_pred = 0.1 s, the prediction window spans 1.5 s and ~22.5 m of forward distance. The occupancy state at the terminal position depends on what the drone will have observed along the trajectory — which the terminal-only DDA query does not capture. At navigation speeds this is acceptable; at racing speeds the approximation error is larger.

**50 Hz control vs. our >100 Hz requirement.** The paper achieves 50 Hz on an A1000. Our CLAUDE.md requires >100 Hz. If adopting PA-MPPI's MPPI formulation, significant optimization (fewer samples, faster GPU, reduced horizon) would be needed to meet our frequency requirement.

**Minor collisions accepted.** The system incurs ~4 cm voxel penetration in some trials. In drone racing, any contact with a gate structure disqualifies the run. Collision costs would need to be substantially larger or formulated differently.

**Wind robustness degrades above 2 m/s.** No adaptive or robust control is integrated. For outdoor competition in variable wind conditions, this is a concern.

**Fixed 3D boundary.** The current implementation requires a bounded workspace. Racing in large outdoor venues may not satisfy this, depending on the map size. The authors identify this as future work.

---

## Key Parameters / Constants

| Parameter | Value | Description |
|---|---|---|
| N | 17,500 | Parallel MPPI samples per control step |
| H | 15 | Prediction horizon steps |
| Δt_pred | 0.1 s | Per-step prediction timestep (1.5 s total horizon) |
| Δt_ctrl | 0.02 s | Control execution interval (50 Hz) |
| λ | 0.02 | MPPI temperature (low = greedy exploitation) |
| c_goal (occluded) | 0.125 | Weak goal attraction in exploration phase |
| c_goal (visible) | 5.0 | Strong goal attraction in navigation phase |
| c_PoI | 5.0 | Weight on camera-goal alignment cost |
| c_unknown | -4.0 | Reward for ray entering unknown voxel |
| c_occupied | 2.0 | Penalty for ray entering occupied voxel |
| c_threshold | 0.5 m | Distance at which PoI alignment deactivates |
| c_collision | 15.0 | Per-step collision penalty (occupied voxel) |
| R | diag(0.01, 0.025, 0.025, 0.2) | Action cost weighting matrix |
| Map update rate | 10 Hz | OctoMap occupancy grid update frequency |
| Voxel size | ~0.1 m (inferred) | OctoMap resolution (standard 10 cm) |
| Ray-tracing evaluation | Terminal step k=H-1 only | DDA applied once per sample, not per step |
| Hardware (GPU) | NVIDIA A1000 6 GB VRAM | Parallelization platform |
| Hardware (CPU) | Intel i7-13800H | Mapping and orchestration |
| Operating speed | ~1.5 m/s | Experimental flight speed (navigation domain) |

The most practically useful numbers for our system are c_PoI = 5.0 (cost scale for camera-gate alignment) and the terminal-horizon evaluation pattern, which together provide a ready-to-use recipe for adding lightweight gate visibility incentives to any MPPI-based controller without additional computation passes.
