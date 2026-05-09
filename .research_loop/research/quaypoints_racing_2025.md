# QuayPoints: A Reasoning Framework for Autonomous Racing

- **URL**: https://arxiv.org/abs/2510.10886
- **Year**: 2025
- **Authors**: (submitted to IEEE)
- **Platform**: F1TENTH 1:10 scale autonomous racing (RoboRacer tracks)

---

## Key Contribution

QuayPoints addresses a fundamental information asymmetry in hierarchical autonomous racing planners: the global planner knows which trajectory is time-optimal but communicates this knowledge to the local planner only as a sparse sequence of waypoints. The local planner is then left to make reactive, context-free decisions when it encounters opponents or obstacles. The result is a local planner that deviates from the optimal line without understanding which deviations are catastrophically expensive and which are nearly free.

The paper's core contribution is a method to identify and communicate **QuayPoints** — track regions where lateral deviation from the optimal racing line produces disproportionately large lap-time penalties. These are the geometrically constrained "pinch points" (apices, chicane exits) that any time-optimal trajectory must pass through regardless of how much the optimizer is perturbed. By marking these regions and encoding them as exponentially penalized cost nodes in the local planner's graph, the system successfully increases overtaking capability from opponents traveling at 65% of ego speed (baseline) to 75% of ego speed, across four distinct RoboRacer tracks.

---

## Technical Approach

### Lambda Parameterization of Track Width

The foundational mechanism is a scalar parameter λ ∈ [0, 1] that continuously spans the full lateral width of the track at any arc-length position s:

```
p(s) = λ · p_inner(s) + (1 − λ) · p_outer(s)
```

Here p_inner and p_outer are the inner and outer track boundary positions at that arc-length station. Setting λ = 0 pins a waypoint to the outer wall; λ = 1 pins it to the inner wall; λ = 0.5 places it at the track centerline. The optimal racing line is then a function λ*(s) that the time-optimal trajectory optimizer resolves.

This parameterization is elegant because it decouples the lateral optimization variable from the absolute track geometry. The optimizer works entirely in normalized "fraction of track width" space and the physical positions follow by interpolation. This also means the same framework applies across tracks of different widths without re-tuning.

### Generating 55 Alternate Racelines

The QuayPoints extraction procedure requires running the trajectory optimizer multiple times under different lateral constraints to observe which positions remain stable. The method discretizes λ into 11 values {0.0, 0.1, 0.2, ..., 1.0} and enumerates all valid (λ_min, λ_max) pairs where λ_min < λ_max. This yields C(11, 2) = 55 distinct constrained optimization problems, each of which restricts the optimizer to a different sub-band of track width:

- (0.0, 0.1): optimizer forced to hug the outer wall
- (0.4, 0.6): optimizer restricted to the middle third
- (0.9, 1.0): optimizer forced to hug the inner wall
- ... and 51 other combinations

For each constrained problem, a full time-optimal lap trajectory is computed via Hermite collocation with IPOPT. The objective is to minimize total lap time:

```
Minimize  T_f = ∫₀¹ (1 / √b(s)) ds
```

where b(s) := (ds/dt)² and a(s) := d²s/dt² are substitutions that convert the time-optimal problem into a convex-in-b form amenable to numerical collocation. Vehicle dynamics follow a kinematic bicycle model with friction circle constraint: a_long² + a_lat² ≤ μg. This is solved offline, requiring 2–6 hours per track.

### Identifying QuayPoints via Cross-Trajectory Variance

After collecting 55 optimal racelines (each with its own λ*(s) profile), the method computes the standard deviation of λ* at each arc-length station across all 55 trajectories:

```
σ_i = sqrt( (1/N) · Σ_j (λ*_{i,j} − μ_i)² )
```

A low σ_i (threshold empirically set at 0.10) means that across all 55 constrained racelines — even those that were forced to very different parts of the track — the optimal solution still converged to the same normalized lateral position at station i. This consistency reveals that the geometry enforces a unique optimal line at that point regardless of entry or exit conditions. These are the QuayPoints.

Conversely, high σ_i indicates "free zones" where the optimizer happily takes many different lateral positions without significant time penalty — these are safe locations for the local planner to deviate when overtaking.

### Quantifying Time Sensitivity

The paper validates the intuition by measuring the lap-time cost of forcing lateral deviations at QuayPoints vs. non-QuayPoints:

- **At QuayPoints**: 1.26% lap-time penalty per unit deviation (on average)
- **At non-QuayPoints**: 0.5% lap-time penalty per unit deviation

This ~2.5× difference in sensitivity justifies treating QuayPoints as hard constraints and non-QuayPoints as soft constraints in the local planner.

### Local Planner Integration

The local planner is a multilayer graph-based planner (similar to a lattice planner). QuayPoints integration requires only two offline modifications:

1. **Adaptive layer spacing**: graph nodes are sampled more densely near QuayPoint stations, so the planner has finer lateral resolution where it matters most.

2. **Exponential cost penalization at QuayPoints**: instead of a linear lateral deviation cost `w_rl · |d_lat,j|`, the cost at QuayPoint nodes becomes `w_rl · exp(10 · |d_lat,j|)`. The exponential growth rapidly makes large deviations prohibitively expensive, effectively enforcing a near-hard constraint at QuayPoints without modifying the graph search algorithm.

The online graph search is unchanged — only the precomputed cost tables are modified. This preserves real-time performance.

### Continuous Interpolation Between Racing Lines

The λ parameterization inherently supports continuous interpolation. Any convex combination of two racelines λ_A*(s) and λ_B*(s) produces a valid physical trajectory:

```
λ_interp*(s) = α · λ_A*(s) + (1 − α) · λ_B*(s),  α ∈ [0,1]
```

The physical position follows by substituting λ_interp*(s) back into the p = λ·p_inner + (1−λ)·p_outer formula. This enables a planner to smoothly blend between, say, an inner-bias aggressive line and an outer-bias defensive line as a function of opponent position, track conditions, or remaining race distance.

---

## Results

All experiments used the F1TENTH 1:10 scale platform with the RoboRacer track set (Nürburgring, Monza, IMS, Oschersleben, Silverstone for analysis; four tracks for overtaking experiments).

| Metric | Baseline | QuayPoints |
|--------|----------|------------|
| Max opponent speed for successful overtake | ~65% of ego | ~75% of ego |
| Time penalty at QuayPoints per deviation | — | 1.26% |
| Time penalty at non-QuayPoints per deviation | — | 0.5% |
| Tracks evaluated | 4 | 4 |
| Offline extraction time per track | — | 2–6 hours |

The 10 percentage-point improvement in overtaking envelope (65% → 75% opponent speed) is described as consistent across all four tested tracks, suggesting the approach is robust to track geometry variation.

---

## Relevance to Our System

Our current `racing_line.py` runs a multi-start L-BFGS-B optimizer (10 starts: 1 zero, 1 late-apex, 8 random) over gate-passage offsets. As documented in the module header, the optimizer effectively has a **bipartite candidate pool** — the 10 seeds tend to collapse into two distinct basins of attraction (a "cut corners" solution and a "fly through centers" solution). The sim-based selection then picks between these two.

QuayPoints directly addresses this limitation in several ways:

**1. Diagnosing which gates are true QuayPoints vs. free zones.** In our gate layout, some gates are at course apices (e.g., the S-turn at gate-3, the helix entry) where offset choice strongly determines tracking error. Others are on long straights where centering or edge-hugging makes little difference. The λ-variance analysis would tell us algorithmically which is which, rather than relying on per-gate error post-hoc.

**2. Generating a richer candidate pool.** Instead of 2 effective basins, we could run 11–55 constrained L-BFGS optimizations, each with `max_lateral_offset` re-centered at different λ values (e.g., force offset ∈ [−0.6, −0.2] to explore inner-bias solutions for a given gate). The sim-based selection in iteration 22–23 already has the infrastructure to evaluate and rank any number of candidates; the bottleneck is candidate diversity, not evaluation cost.

**3. Implementing exponential penalization at QuayPoint gates.** Currently `smoothness_weight` and `speed_weight` are uniform across all gates. If gate-3 (S-turn) is identified as a QuayPoint, we could apply an exponential penalty on its offset deviation in the objective function, effectively hardening the constraint there while relaxing it at free-zone gates.

**4. λ-based interpolation for adaptive racing lines.** If we identify that gate-7 (helix) is a QuayPoint at λ* ≈ 0.3 (inner bias) but gate-5 is a free zone, we can construct intermediate racelines by fixing gate-7's offset near λ* = 0.3 and varying gate-5's offset continuously. This is more principled than our current random-seed exploration, and would produce a family of trajectories that are all time-optimal at the constrained gates while exploring the free-zone degrees of freedom.

**5. The σ < 0.10 threshold as a gating criterion.** We already track `per_gate_avg_error` — gates with high cross-run variance in our current candidates likely correspond to free zones (the optimizer doesn't care), while gates with low cross-run variance in the offset chosen by L-BFGS likely correspond to QuayPoints. We could compute this cheaply from our existing 10-start outputs without any new offline computation.

---

## Actionable Takeaways

1. **Add λ-parameterized offset bounds to L-BFGS starts.** Instead of initializing with random offsets sampled from the full `[−max_off, +max_off]` range, generate N_STARTS seeds that systematically cover sub-bands: first third, middle third, outer third. This guarantees homotopic diversity across all gates, not just at the global optimum level.

2. **Compute cross-candidate offset variance per gate.** After the existing 10-start L-BFGS pool is evaluated, compute σ(offset_i) across the top-k candidates (k = 5 or all 10). Gates with σ < 0.10 * max_off are QuayPoint candidates — flag them and apply tighter offset constraints in the final trajectory.

3. **Apply exponential cost at QuayPoint gates.** In the L-BFGS objective function, replace the uniform `path_length + smoothness_weight * curvature` with gate-specific weights that exponentially penalize offset deviation at identified QuayPoint gates. This prevents the optimizer from treating a high-cost apex the same as a low-cost straight-gate.

4. **Use λ-interp to fill the gap between the two current basins.** Generate 3–5 intermediate candidates by interpolating the offset vectors of the two dominant basin solutions: `offsets_interp = α * offsets_A + (1−α) * offsets_B` for α ∈ {0.25, 0.5, 0.75}. Add these to the candidate pool before sim-selection. This directly combats the bipartite collapse documented in the codebase.

5. **Do NOT apply QuayPoint logic uniformly.** The paper's core finding is that sensitivity is highly heterogeneous across track positions. Applying exponential cost everywhere would re-introduce the smoothness/speed tradeoff uniformly. The value is in selective hardening at QuayPoints and selective relaxation at free zones.

6. **Offline QuayPoint map is track-specific.** If the race course layout changes (different gate arrangement), the QuayPoint map must be recomputed. Since our offline computation is fast (L-BFGS, not IPOPT on full kinematic bicycle), we could run a 55-candidate sweep in minutes rather than hours, making this practical before each competition.

7. **The exponential cost factor of 10 in `exp(10 · |d_lat|)` is empirically tuned on F1TENTH.** For our system with gate half-widths of ~0.6m and offset bounds of ±0.36m, the equivalent exponential scale should be calibrated: at the boundary offset (0.36m), the cost should be ~10× the linear cost, implying a scale factor of ~ln(10)/0.36 ≈ 6.4.

---

## Limitations & Caveats

**Computational cost.** The paper reports 2–6 hours per track for extracting QuayPoints using IPOPT with Hermite collocation. This is because it requires solving 55 full trajectory optimization problems at race-lap fidelity. For our drone racing system where the "track" is a fixed gate layout in 3D, the equivalent computation using L-BFGS would be much faster (seconds per candidate), but the 55-candidate overhead may still be noticeable.

**F1TENTH vs. drone dynamics.** The paper uses a kinematic bicycle model with a 2D friction circle. Our system operates in 3D with quadrotor dynamics (SE(3) attitude, rotor thrust limits, inertial coupling). The λ parameterization concept transfers directly, but the specific threshold σ < 0.10 and cost scale factor exp(10·|d|) are calibrated for ground vehicle dynamics and may need re-tuning for aerial vehicles.

**Binary classification.** QuayPoints are identified via a hard threshold σ < 0.10. This is a discontinuous decision boundary over what is actually a continuous sensitivity spectrum. A smoother approach would weight the exponential cost by (1 − σ/σ_max), giving naturally graded penalties across the track.

**No dynamic adaptation.** The QuayPoint map is computed offline and is static during racing. An online variant that updates the sensitivity estimates based on observed tracking errors during the race would be more powerful, but the paper does not explore this.

**Local planner architecture dependency.** The integration was designed for a multilayer graph-based local planner. Applying the same concepts to our min-snap polynomial trajectory system requires translating "exponential node costs" into "gate-specific objective weights," which is conceptually equivalent but architecturally different.

**Track geometry assumption.** The framework assumes a well-defined track boundary (inner and outer walls) for the λ interpolation. Drone racing gates define only the pass-through plane, not continuous track walls. The analog is to parameterize each gate's pass-through offset as a fraction of gate half-width, which is what our current system already does — so this is not a blocker, but the gate-to-gate interpolation of λ*(s) needs to be defined between discrete gates rather than along a continuous wall.

---

## Key Parameters / Constants

| Parameter | Value | Meaning |
|-----------|-------|---------|
| λ ∈ [0, 1] | continuous | Normalized lateral position (0 = outer wall, 1 = inner wall) |
| λ discretization step | 0.10 | Step size for generating constrained raceline variants |
| Number of constrained racelines | 55 | C(11, 2) combinations of (λ_min, λ_max) |
| QuayPoint σ threshold | 0.10 | Max cross-raceline std-dev of λ to qualify as a QuayPoint |
| Exponential cost scale | 10 | Factor in exp(10 · \|d_lat\|) at QuayPoint nodes |
| Time penalty at QuayPoints | 1.26% per deviation | Sensitivity of lap time to lateral deviation at QuayPoints |
| Time penalty at non-QuayPoints | 0.5% per deviation | Sensitivity at free zones (~2.5× less than QuayPoints) |
| Overtaking improvement | 65% → 75% opponent speed | Main performance result across 4 tracks |
| Offline extraction time | 2–6 hours per track | Using IPOPT + Hermite collocation on F1TENTH platform |
| Friction coefficient | μ (unspecified, standard F1TENTH) | Friction circle constraint: a_long² + a_lat² ≤ μg |
| Track set | RoboRacer (Nürburgring, Monza, IMS, Oschersleben, Silverstone) | 5 tracks for analysis, 4 for overtaking experiments |
