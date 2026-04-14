# Spatially-Aware Adaptive Trajectory Optimization with Controller-Guided Feedback
- **URL**: https://arxiv.org/abs/2602.15642
- **Authors**: Alexander Wachter, Alexander Willert, Marc-Philip Ecker, Christian Hartl-Nesic
- **Year**: 2026
- **Venue**: Accepted at ICRA 2026 (submitted February 17, 2026)

---

## Key Contribution

This paper presents a closed-loop trajectory optimization framework that treats controller tracking errors not as noise to be filtered out, but as informative signals about the track's local physical characteristics. The core insight is that when a vehicle fails to track a planned trajectory in a particular spatial region, this reveals something true and exploitable about that region — the planned acceleration demands exceed what is locally feasible, whether due to friction variation, unmodeled dynamics, or sensor-actuator latency. Rather than requiring explicit identification of these parameters, the method learns them implicitly by spatially attributing tracking errors to trajectory segments and iteratively tightening local acceleration limits.

The second major contribution is the integration of CMA-ES (Covariance Matrix Adaptation Evolution Strategy) as the trajectory search engine, operating over a NURBS-parameterized path space. This combination is significant: NURBS provides a compact, smooth, globally coherent parameterization of the entire lap trajectory, while CMA-ES provides derivative-free population-based global search that is robust to the non-convex, multi-modal structure of minimum-lap-time optimization. The two components together, guided by the adaptive spatial constraint map, produce a system that converges to near-optimal trajectories purely from online tracking experience — without explicit friction identification, tire modeling, or model calibration.

---

## Technical Approach

### NURBS Trajectory Parameterization

The trajectory is represented as a cubic NURBS (Non-Uniform Rational B-Spline) of degree p=3, ensuring C² continuity everywhere. The spatial curve is defined by:

    c(u) = [sum_i N_{i,p}(u) * w_i * p_i] / [sum_i N_{i,p}(u) * w_i]

where u ∈ [0,1] is the normalized arc parameter, p_i are control points, w_i are rational weights, and N_{i,p}(u) are the B-spline basis functions. The temporal mapping is simply u(t) = t/T, where T is the lap time to be optimized.

Physical derivatives (velocity, acceleration) scale with lap time as:

    q^(k)(t) = (1/T^k) * c^(k)(t/T)

This means the same NURBS curve can be re-timed by varying T alone, without changing the spatial shape. The optimization parameter vector θ contains: free control point positions, rational weights, and knot positions. Closure constraints (the lap must close) reduce the free degrees of freedom by 6 for a 2D trajectory.

This parameterization is central to the method's computational tractability: a smooth, closed, feasible trajectory can be described by a compact θ vector, making CMA-ES population-based search over trajectory space practical.

### Time-Optimal Speed Profile

Given a spatial path c(u), the minimum achievable lap time T is determined by the tightest constraint across all track locations:

    T = max_{u ∈ [0,1]} {
        v(u) / v_max,
        sqrt(|a_par(u)| / a_par_max(q(u))),
        sqrt(|a_perp(u)| / a_perp_max(q(u)))
    }

where:
- a_par(t) = [c'(u) · c''(u)] / [T² * ||c'(u)||²] is the longitudinal acceleration
- a_perp(t) = [c'(u) × c''(u)] / [T² * ||c'(u)||²] is the lateral (centripetal) acceleration

Crucially, the acceleration limits a_par_max and a_perp_max are not constants — they are spatially varying functions provided by the adaptive constraint map M described below. This means the speed profile is recomputed whenever the map is updated, automatically slowing down in regions where previous laps showed tracking failures.

### CMA-ES Global Search

CMA-ES searches the parameter space θ = (control points, weights, knot positions) to minimize:

    J(θ) = T(θ) + λ_dist * Φ_distance(θ) + λ_curv * Φ_curvature(θ)

where:
- T(θ) is the minimum lap time for the given path shape
- Φ_distance penalizes boundary violations (track limits)
- Φ_curvature penalizes turning radii that exceed the vehicle's mechanical limits

CMA-ES is used because the lap time objective is non-convex in θ: different control point configurations can yield qualitatively different trajectory topologies (e.g., early-apex vs. late-apex through a hairpin), and gradient-based optimizers get trapped in whichever local basin the initialization falls into. CMA-ES maintains a full covariance matrix over the population of candidate trajectories, allowing it to learn correlations between control point positions and exploit promising search directions while still exploring.

The paper does not disclose CMA-ES hyperparameters (population size, initial sigma, maximum iterations) but reports that the unguided baseline (no spatial feedback) converges at 4000 CMA-ES iterations to a 20.02s lap time on the F1Aut track.

### Adaptive Spatial Constraint Map

A grid M ∈ R^{X×Y} assigns a scaling factor in [0,1] to each spatial cell, where 1.0 means the nominal friction/grip assumption holds and values below 1.0 reduce the local acceleration limit:

    a_max(x,y) = M_{x,y} * a_nominal

Initially M is uniform at 1.0. After each lap, tracking errors from the MPC controller are processed through a "blame region" mechanism and used to update M via a Kalman-filter-inspired update rule:

    K_{x,y} = V_{x,y} / (V_{x,y} + R)         # Kalman gain
    M_{x,y}^+ = M_{x,y}^- + K_{x,y} * e         # map update
    V_{x,y}^+ = (1 - K_{x,y}) * V_{x,y}^- + Q  # uncertainty update

where R is the measurement noise covariance (confidence in the tracking error signal), Q is the process noise (allows the map to drift over time, enabling adaptation to changing conditions), and e is the modulated tracking error signal assigned to this cell.

The uncertainty V_{x,y} starts high (trusting new observations heavily) and decreases as more laps accumulate in a given region, naturally weighting early laps less and later laps more — consistent with Bayesian online learning.

### Blame Region Computation

The key algorithmic novelty is attributing tracking errors causally to the right trajectory segment. A naive approach (blame the segment closest to the error location) is incorrect because vehicles overshoot: the physical error occurs downstream of the segment that actually demanded the infeasible acceleration.

The paper's solution:
1. Compute the longitudinal acceleration profile a_par(u) along the planned trajectory.
2. Find all sign-change indices Z = {i | sign(a_par[i+1]) ≠ sign(a_par[i])} — these are the transitions between acceleration and deceleration phases (braking points and apex exits).
3. When a tracking error occurs at arc position u_err, locate the minimum-distance point on the planned path.
4. Attribute the error to the zero-crossing Z[j] immediately preceding u_err in arc-length.
5. Update constraint map cells within a radius of the blamed zero-crossing.

The error signal e is asymmetrically modulated:
- If the error ê is below a threshold e_th (small deviation, possibly just noise):  e = w+ * ê  (downweight)
- If the error ê is above e_th (genuine tracking failure):  e = w- * ê  (upweight)

This asymmetry prevents noise-driven over-tightening of the constraint map while still aggressively updating on genuine failures. Specific values for e_th, w+, w- are not disclosed.

### MPC Tracking Controller

The tracking controller uses a kinematic single-track model with state x = [x, y, θ, δ, v]^T and control u = [a, δ̇]^T. It produces tracking errors (minimum Euclidean distance to the reference path at each time step) that are fed back to the constraint map. The controller is a standard receding-horizon MPC and is not the paper's main contribution — the novelty is in using whatever errors this controller produces as the learning signal.

---

## Results

### Simulation (Table I in paper)

| Track       | Baseline (s) | Adaptive (s) | Improvement |
|-------------|-------------|--------------|-------------|
| F1Aut       | 20.02       | 16.54        | 17.38%      |
| Wall1       | 16.82       | 15.71        | 6.60%       |
| Levine      | 11.08       | 10.42        | 5.96%       |
| Operngasse  | 7.49        | 6.24         | 16.69%      |

The baseline is a CMA-ES-optimized trajectory without spatial feedback — already well-optimized. The adaptive method adds 7-8 additional laps of online learning on top.

### Convergence

- Baseline (no feedback): 4000 CMA-ES iterations → 20.02s
- With feedback: 8 additional laps (~500 iterations/lap) → 16.54s (17.38% improvement)
- Low-friction regions (80% grip reduction): convergence within 3 laps

### Real-World F1Tenth Platform

Tested across three tire configurations without any explicit friction parametrization:

| Tire Config           | Adapted Lap Time (s) | vs. Non-Adapted Baseline (7.53s) |
|-----------------------|---------------------|----------------------------------|
| High friction (slick) | 5.29                | -29.7%                           |
| Mixed (half slick)    | 5.56                | -26.2%                           |
| Low friction (off-road) | 5.73              | -23.9%                           |
| Average improvement   | —                   | **7.60%** (within-config)        |

The 7.60% cited "overall improvement" is measured within each friction condition (adaptive vs. non-adaptive with the same tires), not the absolute improvement from baseline. The method's key result is that it automatically adjusts to each tire compound without being told the friction coefficient — the spatial map learns what is feasible purely from tracking error.

---

## Relevance to Our System

Our current `planning/racing_line.py` uses multi-start L-BFGS-B with N_STARTS=10, combining zero-initialization and late-apex seedings plus 8 random seeds. The critical limitation this paper addresses directly is the **bipartite local minima problem**: L-BFGS-B is a gradient-based optimizer that, starting from any particular initialization, converges to the nearest local minimum in the racing line objective. With 10 starts, we are sampling 10 points from the optimization landscape — but these 10 points may cluster into only 2 distinct basins (the "bipartite" structure mentioned in our system context), leaving large regions of potentially better trajectory space unexplored.

CMA-ES addresses this directly. Unlike L-BFGS-B, CMA-ES is a derivative-free, population-based method that maintains a multivariate Gaussian over the parameter space and adapts its covariance matrix based on which directions yield improvement. Practically, this means:

1. **Cross-basin exploration**: CMA-ES can maintain population members in multiple basins simultaneously, preventing premature convergence to a single local minimum.
2. **Correlated parameter exploration**: The covariance matrix allows CMA-ES to learn that, e.g., the offset at gate-3 and gate-4 are correlated (both belong to the same S-turn), and propose moves that adjust both coherently. L-BFGS-B uses gradient directions that may not capture these geometric correlations across gates.
3. **No gradient required**: Our racing line objective involves kinematic simulation rollouts (as of iteration 22) that produce noisy, non-smooth objective values. CMA-ES handles this naturally; L-BFGS-B requires numerical gradients that are noisy and potentially misleading.

The **spatial feedback mechanism** is also highly relevant. Our current sim-based selection (iteration 22) runs a forward simulation to evaluate candidates but does not feed this information back to modify the constraint assumptions for the next optimization round. This paper shows that using simulation errors to tighten local acceleration limits — and then re-optimizing with CMA-ES under the updated constraints — produces significantly better trajectories than a single-pass optimization. For us, this would mean: run our PyBullet benchmark, identify gates with high tracking error, reduce the speed profile limits near those gates, re-run racing line optimization with updated constraints, and iterate.

The **NURBS parameterization** is also a better choice than our current approach for generating intermediate candidates. Our L-BFGS-B optimizes over a vector of per-gate lateral offsets — a discrete parameterization that cannot express trajectory shapes between gate centers. NURBS control points, by contrast, can be placed anywhere, allowing the optimizer to discover racing lines that cut across gate regions rather than threading each gate individually.

---

## Actionable Takeaways

1. **Replace L-BFGS-B with CMA-ES for racing line search.** Use the `cma` Python package (lightweight, no extra dependencies beyond numpy) to optimize the per-gate lateral offset vector θ. Initialize with our current best racing line as the mean, sigma ~ 0.2 (roughly 20% of max lateral offset range). CMA-ES with population size 4+floor(3*ln(N_gates)) ≈ 10-15 will explore meaningfully more of the basin structure than our current 10 L-BFGS-B starts.

2. **Implement a spatial acceleration constraint map driven by benchmark errors.** After each benchmark run, parse `simulation.per_gate_avg_error` to identify high-error gates. For each gate with error > threshold (e.g., 0.3m), reduce the speed profile limit for the trajectory segment approaching that gate by a factor (e.g., multiply `DroneConstraints.max_velocity` by 0.9 for that segment). Re-run racing line optimization under the updated constraints. This implements the paper's core adaptive loop at the level of our existing benchmark infrastructure.

3. **Adopt asymmetric error modulation for constraint updates.** Only tighten local constraints when tracking error exceeds a noise floor (e.g., 0.15m, roughly 2× our EKF uncertainty). Small errors below this threshold should not reduce the constraint map, to prevent over-conservative convergence. This is the paper's w+/w- asymmetry, directly applicable.

4. **Use the blame-region causal attribution when identifying which segment to constrain.** High tracking error at gate N is not always caused by the trajectory at gate N — it may be caused by an excessive speed or curvature demand at gate N-1 (braking too late into the corner). For gate i with high error, also check the trajectory curvature/velocity at gate i-1 and tighten the constraint there if it shows high centripetal acceleration demand.

5. **Generate intermediate NURBS-style candidates by perturbing control point positions, not just endpoint offsets.** Our L-BFGS-B optimization works on per-gate offsets (one degree of freedom per gate). NURBS-style intermediate candidates can be generated by adding control points between gates and optimizing their lateral positions too. This expands the search space to include trajectories that smoothly arc between gate openings rather than targeting each gate center independently.

6. **Run CMA-ES warm-started from the iteration-23 best candidate.** Do not restart from scratch — use the current best racing line as the initial mean for CMA-ES, with sigma small enough to preserve the current solution while still exploring neighbors. This guarantees no regression (equivalent to T-MPC's Theorem 2 guarantee, de Groot 2024).

7. **Apply the adaptive constraint update iteratively between benchmark runs.** The paper converges in 8 laps. Our benchmark runs 20s of simulation per call. Three to five benchmark calls with constraint updates between them could yield similar convergence, at a cost of ~2-3 minutes of compute.

---

## Limitations & Caveats

**Ground vehicle focus, 2D only.** The paper is evaluated exclusively on planar racing circuits with the F1Tenth platform. The drone racing problem introduces a third spatial dimension: altitude changes, 3D gate orientations, and roll/pitch coupling with lateral acceleration. The NURBS representation extends naturally to 3D (just add z to the control point coordinates), but the time-optimal speed profile computation must account for vertical acceleration budget consumption (thrust used fighting gravity is unavailable for lateral maneuvers).

**CMA-ES computational cost.** CMA-ES with population size ~15 evaluates 15 complete trajectory rollouts per generation. If each rollout requires full kinematic simulation, this is expensive. However, our PyBullet-based simulation runs at ~20Hz wall time, and a 20s race takes ~1s of simulation. With 15 candidates × multiple generations × multiple benchmark calls, total compute could be 5-30 minutes. This is acceptable for pre-race offline optimization but not for real-time replanning.

**No explicit hyperparameter disclosure.** The paper does not reveal its CMA-ES population size, initial sigma, R and Q values for the Kalman update, or e_th for the error modulation. The key numerical constants are absent, requiring empirical tuning for our specific system. The qualitative prescription (small R for stable maps, larger Q for adapting to changing conditions) is the most actionable guidance available.

**Blame region complexity for drone racing.** The paper's blame region uses longitudinal acceleration sign transitions (braking/acceleration phase boundaries) to causally attribute errors. For a drone flying a gate-to-gate trajectory, the analogous partitioning would be based on the thrust profile and attitude transitions. In a helix segment (like gate-7 in our course), the "acceleration phase boundaries" are less well-defined than in ground-vehicle racing. The attribution mechanism may require adaptation.

**NURBS parameterization replaces, not supplements, per-gate offsets.** Adopting NURBS fully would require refactoring `planning/trajectory_optimizer.py` significantly, since our polynomial trajectory optimizer currently takes gate waypoints as inputs, not NURBS control points. A hybrid approach (use CMA-ES to optimize over gate offsets as today, keeping the polynomial trajectory layer intact) is much less disruptive and captures the main benefit (better search) without architectural changes.

**Friction learning is not our bottleneck.** The paper's most dramatic results come from adapting to varying friction conditions without explicit identification — a critical capability for ground vehicles that wear tires and race on variable surfaces. Our drone operates in air: aerodynamic drag parameters are much more stable than tire friction, and our main source of tracking error is geometric (trajectory curvature demands vs. controller bandwidth), not grip variation. The iterative constraint tightening is still useful, but the magnitude of improvement may be smaller than the 17% the paper reports.

---

## Key Parameters / Constants

From the paper and related work, the following numerical values are either disclosed or inferable:

- **CMA-ES iterations to baseline convergence**: ~4000 iterations without feedback (F1Aut track, 20.02s lap time)
- **Feedback convergence**: ~8 additional laps, ~500 CMA-ES iterations per lap re-optimization
- **Low-friction convergence**: ~3 laps (M grid updates rapid when errors are large)
- **Lap time improvements**: 5.96% to 17.38% across tracks (median ~12%)
- **Real-world improvement**: 7.60% within each tire configuration
- **NURBS degree**: p = 3 (cubic), ensuring C² continuity
- **Constraint scaling**: M_{x,y} ∈ [0, 1], where 0 means fully blocked and 1 means nominal limits
- **Kalman noise ratio R/Q**: Not disclosed. High R/Q → slow adaptation (stable map); low R/Q → fast adaptation (responsive to new data). For slowly varying conditions, R >> Q is standard.
- **Error asymmetry weights w+, w-**: Not disclosed. The asymmetry (w- > w+ for errors above threshold) is the key design choice. Typical values in ILC literature: w+ = 0.5, w- = 1.5.
- **Error threshold e_th**: Described as "balance between position measurement noise and maximum admissible deviation" — in practice, this should be set to ~2-3× the RMS noise floor of the tracking error. Our system's EKF uncertainty is ~0.05m (iteration 23), suggesting e_th ≈ 0.10-0.15m.
- **Blame radius**: Not quantified. For a drone at ~10m/s with ~0.2s controller lag, a blame radius corresponding to ~2m arc-length (one gate spacing) is reasonable.
- **Reduction factor per blame event**: Not disclosed. Conservative starting point: reduce a_max by 5-10% per blame event, with the Kalman gain controlling effective step size.

---

*Analysis written 2026-04-14. Sources: [arXiv:2602.15642](https://arxiv.org/abs/2602.15642), paper abstract and HTML version fetched 2026-04-14.*
