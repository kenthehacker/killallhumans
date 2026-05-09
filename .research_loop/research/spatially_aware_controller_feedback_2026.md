# Spatially-Aware Adaptive Trajectory Optimization with Controller-Guided Feedback for Autonomous Racing

- **URL**: https://arxiv.org/abs/2602.15642
- **Authors**: Alexander Wachter, Alexander Willert, Marc-Philip Ecker, Christian Hartl-Nesic
- **Year**: 2026 (ICRA 2026, submitted February 17, 2026)

## Key Contribution

This paper presents a closed-loop raceline optimization framework that treats controller tracking errors not as noise to be rejected, but as spatially-meaningful signals about local track-vehicle interaction limits. The core insight is that when a controller fails to follow the planned trajectory at a particular track location, it reveals something about the actual feasible acceleration envelope at that point — either the limits are too aggressive, or the trajectory shape is locally poor. By mapping these errors back to the track geometry using a "blame region" computation and updating a spatially-indexed constraint map via a Kalman-inspired filter, the system progressively refines the acceleration constraints used by the global NURBS/CMA-ES optimizer.

The result is a fully closed-loop system that improves lap times by 5.96% to 17.38% in simulation, and achieves 7.60% improvement on real F1Tenth hardware across varying tire friction conditions — without ever explicitly parameterizing friction. This is particularly significant for our use case: the system learns from actual closed-loop behavior rather than relying on accurate a-priori dynamic models.

## Technical Approach

### Trajectory Representation: Cubic NURBS

The racing line is represented as a cubic NURBS curve (degree p = 3) mapping normalized parameter u ∈ [0,1] to 2D spatial positions:

```
c(u) = Σᵢ (wᵢ · Nᵢ,₃(u) · pᵢ) / Σᵢ (wᵢ · Nᵢ,₃(u))
```

where pᵢ ∈ ℝ² are control points, wᵢ > 0 are weights, and Nᵢ,p(u) are B-spline basis functions. Cubic NURBS guarantees C² continuity, which ensures smooth curvature transitions — essential for feasibility in vehicle dynamics and continuity of the acceleration constraints. The temporal parametrization is:

```
u(t) = t / T
```

where T is lap time. The free optimization variables are control point positions p_free, weights w_free, and interior knot locations u_free, with closure constraints applied to ensure a valid loop.

### Time-Optimal Parametrization and Acceleration Decomposition

Given the spatial curve c(u) and temporal parameter u(t), velocity and acceleration decompose as:

```
v(t) = (1/T) · ‖c⁽¹⁾(u)‖₂

a_∥(t) = [c⁽¹⁾(u) · c⁽²⁾(u)] / [T² · ‖c⁽¹⁾(u)‖₂²]   (longitudinal)

a_⊥(t) = [c⁽¹⁾(u) × c⁽²⁾(u)] / [T² · ‖c⁽¹⁾(u)‖₂²]   (lateral)
```

Spatially-varying acceleration limits are defined via a 2D grid map M_{x,y} over the track domain:

```
a_∥,max(x,y) = M_{x,y} · a_∥,nominal
a_⊥,max(x,y) = M_{x,y} · a_⊥,nominal
```

Initially M_{x,y} = 1 everywhere (nominal limits). The key innovation is that this map is updated online based on controller feedback.

### CMA-ES Global Optimization

The raceline optimizer solves:

```
θ* = argmin_θ  T(θ) + λ_dist · Φ_distance + λ_curv · Φ_curvature
```

where:
- `T(θ)` is the lap time under the current constraint map
- `Φ_distance` penalizes control points that deviate from the track centerline (for boundary feasibility)
- `Φ_curvature = ∫₀¹ max(0, κ(u) - κ_max)² du` penalizes curvature violations

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is used because the objective is non-differentiable (due to the piecewise structure of the constraint map) and CMA-ES is gradient-free, globally exploring, and handles the moderate dimensionality (tens of control points) efficiently. No differentiable trajectory optimization library is needed.

### Blame Region Computation

When the controller fails to track the reference trajectory, the error must be attributed to a *spatial region* of the track (not just the time instant of error), because the trajectory was poorly shaped in the run-up to the failure point. The blame region is computed using longitudinal acceleration sign transitions:

```
S = sign(a_∥)
Z = {i | S[i+1] ≠ S[i]}    (sign transition indices)
i_transition = max({z ∈ Z | z < i_min} ∪ {Z[-1]})
```

where i_min is the index of the point on the reference trajectory closest to the actual vehicle position at the time of peak error. The blame region spans from i_transition to i_min — essentially, the stretch of track where the vehicle was decelerating into the problematic turn.

This is more principled than simply attributing error to the current position: a vehicle that enters a corner too fast carries the blame back to where it started braking.

### Kalman-Inspired Spatial Update

The constraint map M_{x,y} is updated using a Kalman filter-like recursion for each cell in the blame region:

```
K_{x,y}   = V_{x,y} / (V_{x,y} + R)           (Kalman gain)
M⁺_{x,y}  = M⁻_{x,y} + K_{x,y} · e            (map update)
V⁺_{x,y}  = (1 - K_{x,y}) · V⁻_{x,y} + Q      (uncertainty update)
```

where:
- `V_{x,y}` is the per-cell uncertainty (variance)
- `R` is measurement noise (how much to trust a single error reading)
- `Q` is process noise (how much the limit changes lap-to-lap)
- `e` is the modulated error signal: `e = clip(tracking_error - e_th, 0, ∞) · sign(a_∥)`

The threshold e_th prevents small numerical errors from polluting the map. Only errors exceeding e_th trigger a map update, and only in the appropriate direction (the sign of longitudinal acceleration determines whether to tighten longitudinal or lateral limits).

### Overall Closed-Loop Cycle

1. **Optimize**: Run CMA-ES with current constraint map M to produce new NURBS raceline and T
2. **Track**: MPC controller attempts to follow the new trajectory
3. **Evaluate**: Measure per-timestep tracking errors, compute blame regions
4. **Update**: Apply Kalman update to M_{x,y} cells in blame regions
5. **Goto 1**: Re-optimize with updated constraints

Convergence in simulation is achieved in approximately 10 laps.

## Results

### Simulation (4 tracks, kinematic single-track model + MPC controller)

| Track | Static (no feedback) | Adaptive (with feedback) | Min Curvature baseline | Homeomorphism baseline |
|-------|---------------------|--------------------------|------------------------|------------------------|
| F1Aut | 20.02 s | **16.54 s** | 22.12 s | 21.83 s |
| Wall1 | 16.82 s | **15.71 s** | 17.32 s | 16.92 s |
| Levine | 11.08 s | **10.42 s** | 11.39 s | 12.18 s |
| Operngasse | 7.49 s | **6.24 s** | 8.37 s | 7.58 s |

Improvement over static: 5.96% (Wall1) to 17.38% (Operngasse).
Both baselines (minimum curvature racing line, homeomorphism-based) are beaten on all four tracks.

### Real Hardware (F1Tenth car, circular track, 3 tire configurations)

| Tire | Initial lap | Converged lap | Improvement |
|------|-------------|---------------|-------------|
| High friction | 7.53 s | 5.29 s | 29.75% |
| Medium friction | 7.53 s | 5.56 s | 26.16% |
| Low friction | 7.53 s | 5.73 s | 23.90% |

Average hardware improvement: 7.60% across compounds. No friction parameter was ever specified — the system learned the limits from closed-loop behavior. Convergence in ~10 laps.

## Relevance to Our System

Our system uses min-snap polynomial trajectories optimized with L-BFGS and per-section ILC with Butterworth Q-filter. This paper's framework is conceptually very close to what our ILC does, but the spatial constraint map mechanism is significantly more principled than our current approach.

**Direct mapping to our ILC**: Our ILC currently collects per-lap tracking errors and uses a Q-filtered correction to update feedforward. This is equivalent to the paper's "update M based on error" step — but we apply it in the *time domain* (correcting the feedforward signal at each timestep) rather than the *spatial domain* (updating acceleration limits at each track location). The spatial approach is more robust to timing jitter because the constraint is anchored to physical track position.

**Blame region = our per-section ILC boundaries**: Our ILC divides the track into sections (roughly corresponding to each gate approach and exit). The blame region computation is a data-driven way to determine section boundaries automatically, rather than our fixed manual segmentation. The sign-of-longitudinal-acceleration heuristic for finding the start of the blame region closely matches where our ILC sections should begin.

**CMA-ES as an alternative to L-BFGS for raceline optimization**: Our `planning/racing_line.py` uses L-BFGS for the lateral offset optimization. CMA-ES is gradient-free and can escape local minima that gradient methods like L-BFGS cannot. For gate-7's helix (high curvature, non-convex feasible region), CMA-ES on a NURBS representation could find qualitatively better lines.

**The Kalman update is directly portable**: The M/V/K recursion is 5 lines of numpy code. We could implement a 2D grid map over our sim arena and accumulate per-lap error signals, then use the map to modulate the constraint envelope in `planning/trajectory_optimizer.py`.

**Specific applicability to gate-7**: Gate-7 (helix, 0.284 m error) is exactly the kind of high-curvature, locally infeasible section where the spatial constraint map would reduce the planned acceleration in the helix approach corridor, forcing the optimizer to find a more achievable line.

## Actionable Takeaways

1. **Implement a 2D spatial constraint map** over the competition arena (grid resolution ~0.5 m). Initialize M_{x,y} = 1.0 everywhere. After each benchmark run, run the blame region computation and apply the Kalman update.

2. **Replace fixed ILC section boundaries with blame regions**: Compute `S = sign(a_∥)` along the current trajectory, find sign transitions, and use those as natural section split points. This will auto-localize the helix section correctly.

3. **Switch raceline optimizer from L-BFGS to CMA-ES** on NURBS control points, particularly for re-optimizing the gate-7 approach. Use `scipy.optimize.differential_evolution` or the `cma` package. Penalty weights: λ_dist and λ_curv both around 0.1–1.0 (tune empirically).

4. **Apply the Kalman-update M to modulate velocity profile limits** in `planning/racing_line.py`: before computing the curvature-limited speed profile, multiply the lateral acceleration limit by `M_{x,y}` at each arc-length sample. This makes the speed profile conservative in regions where tracking has historically failed.

5. **Set threshold e_th = 0.05 m** (approximately 1/4 of our current avg error) to filter out negligible errors before map updates. Only spatial cells with consistent >0.05 m error should see their limit reduced.

6. **Use process noise Q = 0.01, measurement noise R = 0.05** as initial Kalman parameters. This implies the system is slow to update (conservative) but eventually converges. Tune Q upward if the map needs to respond faster to track condition changes.

7. **Run 10+ benchmark laps with the adaptive system** to allow convergence before declaring final performance. Our current single-run benchmark may underestimate the benefit of iterative methods.

## Limitations & Caveats

- **Ground vehicle (F1Tenth) vs. quadrotor**: The kinematic single-track model has 2D planar dynamics. Our drone operates in 3D with 6-DOF dynamics, thrust vectoring, and aerodynamic drag. The longitudinal/lateral acceleration decomposition needs extension to 3D (longitudinal, lateral, and vertical components), which is more complex.

- **Constant altitude sections only**: The blame region computation relies on 2D spatial grids. For our helix gate (varying altitude), a 3D grid or a curvilinear coordinate along the helix axis would be needed.

- **No gate clearance constraints**: The paper optimizes for lap time on an open track. Our gates impose hard clearance constraints that the NURBS optimizer must also respect. CMA-ES would need hard penalty terms for gate constraint violations.

- **Convergence requires multiple laps**: The system needs ~10 laps to converge. In competition, we may only get 1-2 practice runs. Pre-converging on the sim and transferring the map to real deployment is a necessary adaptation.

- **MPC controller assumed**: The controller used in the paper is MPC with the constraint map integrated. Our controller is a PD + feedforward system. The blame region logic still applies, but the feedback path is simpler and the map update may need re-tuning.

- **No disturbance/wind testing**: Real-world experiments tested friction variation but not external forces (wind). Our competition environment may introduce drift forces not present in the F1Tenth corridor tests.

## Key Parameters / Constants

| Parameter | Value | Usage |
|-----------|-------|-------|
| NURBS degree p | 3 | Cubic, C² continuity |
| Error threshold e_th | Not specified (tune to ~0.05 m for us) | Blame region filter |
| Process noise Q | Not specified (suggest 0.01) | Kalman map update |
| Measurement noise R | Not specified (suggest 0.05) | Kalman map update |
| Initial map value M₀ | 1.0 | Nominal acceleration limits |
| CMA-ES curvature penalty weight λ_curv | Not specified | Tune: 0.1–1.0 |
| CMA-ES distance penalty weight λ_dist | Not specified | Tune: 0.1–1.0 |
| Grid resolution | Not specified (suggest 0.5 m for us) | Spatial constraint map |
| Convergence laps | ~10 laps | Before deploying result |
| Simulation improvement range | 5.96%–17.38% | Track-dependent |
| Hardware improvement | 7.60% avg | Across friction compounds |
| High friction converged lap | 5.29 s from 7.53 s | 29.75% improvement |
