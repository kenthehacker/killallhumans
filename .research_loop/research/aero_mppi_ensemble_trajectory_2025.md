# AERO-MPPI: Anchor-Guided Ensemble Trajectory Optimization for Agile Mapless Drone Navigation

**Source:** https://arxiv.org/abs/2509.17340
**Venue:** ICRA 2026
**Authors:** Not listed in fetched content
**Date fetched:** 2026-04-15

---

## Summary

AERO-MPPI is a GPU-accelerated, mapless navigation framework for agile drone flight in cluttered 3D environments. It replaces the traditional mapping-planning-control pipeline with a unified perception-to-action loop that runs entirely on-device (NVIDIA Jetson Orin NX). The core innovation is an **anchor-guided ensemble of Model Predictive Path Integral (MPPI) optimizers**, where spatial anchors extracted from LiDAR point clouds seed multiple parallel trajectory optimizers, enabling robust exploration across topologically distinct path classes. The system achieves sustained flight above 7 m/s with greater than 80% success rate in dense obstacle fields.

---

## 1. Key Contribution: Avoiding Local Minima via Ensemble Optimization

Single-instance MPPI is a stochastic sampling-based optimizer that draws K random perturbations around a nominal trajectory, evaluates their cost under a dynamics model, and computes a weighted average update. In practice, when the trajectory cost landscape is **non-convex** — as it always is in cluttered 3D environments — a single MPPI instance converges to whichever local minimum its current nominal trajectory is nearest to. If that minimum corresponds to a collision or a slow detour, there is no mechanism to escape it.

AERO-MPPI solves this by running **M = 15 parallel MPPI instances simultaneously**, each initialized to a different guiding trajectory constructed from a distinct spatial anchor. Because each instance explores a different region of the solution space, the ensemble collectively covers multiple homotopy classes (path topologies). The best trajectory across all instances is selected each planning cycle, so the system naturally gravitates toward whichever local minimum is globally superior among the explored set.

This is a **divide-and-explore** strategy: instead of hoping that a single optimizer's random samples happen to escape a bad basin, the problem is decomposed so that each optimizer owns a different basin by construction.

---

## 2. Technical Approach: What Are Anchors and How Do They Diversify Exploration

Anchors are spatially distributed intermediate waypoints derived from a **two-stage multi-resolution LiDAR partition** of the drone's forward field of view (90° horizontal × 54° vertical).

**Stage 1 — High-resolution partition (3° cells):**
The spherical FOV is discretized into a 120×60 grid. For each cell, the nearest obstacle distance is computed from the point cloud in the drone body frame.

**Stage 2 — Coarse partition (18° cells):**
A 20×10 coarse grid is formed by 6×6 pooling of the fine grid. Within each coarse cell, the direction with maximum obstacle clearance is identified and projected forward at a fixed look-ahead distance of **ℓ = 5 m** to produce the coarse safe point (anchor endpoint):

```
p_ref[I,J] = p0 + ℓ · d[I,J]
```

From the 20×10 coarse grid, **Mh = 5 horizontal and Mv = 3 vertical anchors** are selected, giving M = 15 anchor endpoints in total. Each anchor endpoint is then used to construct a **guiding trajectory** via three fifth-order polynomials (one per spatial axis):

```
f_μ(t) = Σ_{i=0}^{5} a_{i,μ} t^i
```

Boundary conditions at t=0 (current position, velocity, acceleration) and t=T (anchor endpoint, zero velocity, zero acceleration) uniquely determine the six coefficients per axis in closed form, yielding a **dynamically feasible reference trajectory** tailored to each anchor.

Each MPPI instance then samples K = 128 random perturbations around its own guiding trajectory, evaluates them under a two-stage multi-objective cost (collision avoidance + goal-directed velocity), and produces an optimized trajectory. The cost function weights are: tracking Qtrack = 15.0, velocity norm Qvnorm = 0.15, position goal Qp = 3.0, and control smoothness Qc = Qc∆ = 0.5. The MPPI temperature is λ = 0.1.

The key diversity mechanism is that anchors are chosen at geometrically separated points in free space, which forces the guiding trajectories to pass through fundamentally different regions. Because MPPI sampling is local (Gaussian perturbations around the nominal), each instance explores its own neighborhood without drifting toward the others.

---

## 3. Parallel Instances

**M = 15 MPPI instances** run in parallel:
- Mh = 5 horizontal anchor columns
- Mv = 3 vertical anchor rows
- Total: 5 × 3 = 15 instances

Each instance uses:
- K = 128 rollouts per planning cycle
- N = 25 horizon steps
- Δt = 50 ms time step (T = 1.25 s lookahead)

All 15 × 128 = 1,920 rollout evaluations are executed concurrently via NVIDIA Warp GPU kernels, achieving real-time performance onboard the Jetson Orin NX.

---

## 4. Results: Ensemble vs. Single-MPPI and Non-Convex Environments

**Success rates:** AERO-MPPI maintains above 80% success across all test scenarios at 7 m/s. Baseline methods (Ego-Planner, Fast-Planner) exhibit substantially lower success rates at equivalent speeds.

**Average velocity comparison (50 successful runs):**

| Method | Forest | Verticals | Inclines |
|--------|--------|-----------|---------|
| AERO-MPPI | 5.60 m/s | 3.52 m/s | 5.08 m/s |
| Ego-Planner | 1.82 m/s | — | — |
| Fast-Planner | 2.53 m/s | — | — |

AERO-MPPI achieves 2.2–3.1× higher average velocity than mapping-based baselines in the forest scenario. The maximum recorded velocity is 8.20 m/s in a 1000-obstacle environment.

The benefit is most pronounced in **non-convex scenarios** (dense forests, vertical pillar arrays, inclined obstacle arrangements) where path topology choices have large impact on success. The ensemble's ability to simultaneously evaluate multiple homotopy classes is the direct mechanism for the improvement: a single optimizer would commit to one path class and either succeed or fail without awareness of alternatives.

---

## 5. Relevance to Our L-BFGS Trajectory Time Optimization

Our current trajectory optimizer uses **L-BFGS** with `time_weight = 2.3` to jointly optimize polynomial segment coefficients and time allocations. The time_weight term penalizes total race time, creating a non-convex landscape because the optimal time allocation for each segment depends on the polynomial coefficients and vice versa. Specific known failure modes:

- **Gate 4 error -5.0%** — overshooting after the helix, suggesting the optimizer converges to a timing allocation that is locally optimal but globally suboptimal.
- ILC corrections up to 0.50 m are needed to compensate for trajectory errors that the optimizer should have avoided.
- Race time has been trending down toward 13.3 s but each marginal gain requires increasingly aggressive hyperparameter tuning rather than algorithmic improvement.

The AERO-MPPI result directly identifies the root cause: **non-convex optimization with a single initial point converges to a local minimum that is sensitive to initialization**. Our L-BFGS starts from a single nominal trajectory (likely a uniform time allocation or a previous iteration's solution), which biases it toward the nearest basin.

AERO-MPPI's lesson is that **structured multi-start initialization** — not more iterations or tighter tolerances — is the correct remedy. Each "anchor" in their system corresponds to what we would call a "seed trajectory" in our context.

---

## 6. Actionable Takeaways: Multi-Start L-BFGS with Structured Seeds

The direct translation of AERO-MPPI to our optimizer is a **multi-start L-BFGS** scheme:

**A. Enumerate diverse initial time allocations**
Instead of one time vector T = [t1, t2, ..., tN], generate S = 5–10 candidate initializations that differ structurally:
- Uniform allocation (current baseline)
- Velocity-profile-derived allocation (proportional to segment arc length)
- Gate-curvature-weighted allocation (more time to tight turns)
- Aggressive short allocation (push total time toward 12 s)
- Conservative long allocation (preserve feasibility, total ~16 s)

**B. Run L-BFGS from each seed in parallel**
Each L-BFGS run converges to its local minimum. The total compute is S × (single L-BFGS cost), which is acceptable if parallelized.

**C. Select the globally best result**
The seed that yields the lowest cost (race time + tracking error surrogate) wins. This is the trajectory sent to the ILC layer.

**D. Warm-start across iterations**
After selecting the winner, use it as one of the seeds in the next planning call. This preserves the "refinement" behavior of single-start while also exploring alternatives.

**E. Anchor the seeds to gate geometry**
Following the AERO-MPPI principle, seeds should be geometrically meaningful rather than random:
- For each gate, compute the entry angle that minimizes curvature at exit; use this to derive a segment time.
- This is analogous to their LiDAR-derived anchor points: geometry-informed, not random.

---

## 7. Key Parameters and Selection Strategy

From AERO-MPPI, the design choices that transfer to our context:

| AERO-MPPI Parameter | Value | Our Analog |
|---------------------|-------|------------|
| Number of anchors M | 15 (5×3) | Number of L-BFGS seeds S = 5–8 |
| Look-ahead distance ℓ | 5 m | Planning horizon (all gates) |
| Anchor grid structure | 5 horizontal × 3 vertical | Time allocation families × scaling factors |
| MPPI rollouts per instance K | 128 | L-BFGS iterations per seed (e.g., 100) |
| Horizon steps N | 25 | Polynomial segments (our N varies per track) |
| Selection criterion | Lowest ensemble cost | Lowest (race_time + λ·tracking_error) |
| Warm-start | Best trajectory carries to next cycle | Best seed reused next L-BFGS call |

**Recommended number of seeds for our use case:** Start with S = 6:
1. Uniform time allocation
2. Arc-length-proportional allocation
3. Curvature-weighted allocation (more time in helix and tight turns)
4. Scaled-down version of best previous iteration (aggressive)
5. Scaled-up version of best previous iteration (conservative feasibility)
6. Random perturbation of the best known solution (exploration)

The structured seeds (1–5) ensure coverage of distinct basins; seed 6 provides local stochastic exploration similar to MPPI's random sampling within each anchor.

**Selection strategy from the paper:** The best trajectory among all instances is chosen purely by objective value — no heuristic filtering. In our case this means evaluating each L-BFGS result's predicted race time + a penalty for constraint violations (e.g., maximum acceleration exceeded), then taking the argmin.

---

## Connection to Related Work

AERO-MPPI shares the multi-start philosophy with:
- **Topology-driven parallel planning** (de Groot 2024, already in research/) — explicit homotopy class enumeration
- **Spatially aware CMA-ES racing** (already in research/) — population-based search over trajectory space
- Our current **ILC layer** partially compensates for suboptimal L-BFGS convergence; multi-start would reduce the ILC correction burden

The key AERO-MPPI insight that extends beyond their specific setting: **the anchor extraction mechanism is a structured prior that encodes domain knowledge into the initialization**. For our problem, the analogous domain knowledge is gate geometry and curvature — we should use these to construct seeds rather than relying on uniform initialization.

---

## Conclusion

AERO-MPPI provides strong empirical and theoretical motivation for moving from single-start to multi-start L-BFGS in our trajectory optimizer. The mechanism is identical in principle: non-convex optimization converges to a local minimum that depends on initialization, and structured initialization from diverse geometrically meaningful seeds covers distinct basins. With 5–8 seeds and parallel L-BFGS execution, we can expect to find substantially lower-cost trajectories for the helix and gate-4 region specifically, potentially eliminating the need for large ILC correction caps (currently 0.50 m) and reducing average tracking error below the current 0.37% regime.

**Implementation priority:** Medium-high. The change is isolated to `planning/trajectory_optimizer.py` (the L-BFGS call site) and does not require changes to the control, EKF, or gate sequencer layers.
