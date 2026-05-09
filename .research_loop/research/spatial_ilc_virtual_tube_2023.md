# Autonomous Drone Racing: Time-Optimal Spatial Iterative Learning Control within a Virtual Tube
- **URL**: https://arxiv.org/abs/2306.15992
- **Authors**: Shuli Lv, Yan Gao, Jiaxing Che, Quan Quan (Beihang University, Beijing, P.R. China)
- **Year**: 2023
- **Venue**: arXiv

---

## Key Contribution

This paper proposes a model-free, online learning method for autonomous drone racing that iteratively discovers time-optimal speed profiles through experience, without requiring system identification or offline optimization. The central insight is borrowed from elite human racers: start conservatively and progressively push speed until the limits of the track are found. The method reformulates the racing problem in the **spatial domain** (parameterized by arc length along the racing line rather than time), which decouples speed optimization from path following and enables convergence analysis of the iterative process.

The second major contribution is an extension of classical Iterative Learning Control (ILC) theory to handle time-optimal control problems. Standard ILC assumes fixed trial duration; the authors' spatial reformulation sidesteps this constraint because every trial traverses the same spatial domain [0, L] regardless of how fast or slow the drone flies. The resulting spatial ILC controller provably converges to a uniformly ultimately bounded tracking error while simultaneously pushing the speed profile toward the feasible maximum — achieving race times within ~1% of SQP-computed optima at 0.5–1.1% of the computational cost.

---

## Technical Approach

### Spatial Domain Reformulation

Classical ILC fails for time-optimal problems because trial lengths vary with speed. The paper resolves this by switching independent variable from time t to arc length l along the generator curve:

```
l = ∫₀ᵗ v(s) ds
```

Since v > 0 throughout (the drone always moves forward), l is monotonically increasing in t and the mapping is globally invertible. Under the spatial differentiator ∇ = d/dl, the drone dynamics become:

```
∇p(l) = (1/v(l)) · v(l)          [position]
∇v(l) = -(τ/v(l)) · (v(l) - v_c(l))   [velocity, first-order lag with maneuverability τ]
```

This is a first-order spatial model parameterized by the scalar maneuverability constant τ (the drone's response bandwidth), which is **not required to be known** — the method is model-free.

### Virtual Tube Construction

A virtual tube T_V ⊂ R² is constructed around the planned racing line:

- The **generator curve** γ connects gate crossing waypoints p̃⁰, p̃¹, ..., p̃^(N+1) through all N gates.
- At each arc length position l, a **cross-section** perpendicular to γ defines the tube boundary.
- **Tube radius** r_t(l) is the distance from the generator curve center to the tube wall at position l. The tube is tighter in corners (small r_t) and wider on straights.
- The constraint T_V ∩ G = ∅ ensures the tube boundary does not clip any gate frame — the drone must pass through gates, and the tube guides it to do so.

The projection operator m(p) maps the drone's current 2D position p to the nearest point on the generator curve. The **lateral path error** at each arc-length step is:

```
e_p,k(l) = m(p(l)) - p(l)
```

The paper notes that explicit tube construction algorithms are not detailed; Assumption 2 states the drone can obtain virtual tube information via measurement or prior knowledge.

### Controller Architecture

The velocity command at each spatial step has two additive components:

```
v_c'(l) = v_h(l) + v_p(l)
```

**Pace controller** (drives speed along the path):
```
v_h(l) = v(l) · v*(l) · t_c(l)
```
where v*(l) is the learned scalar speed profile and t_c(l) is the unit tangent of the generator curve. This steers the drone at speed v*(l) along the nominal racing line.

**Path convergence controller** (corrects lateral deviation):
```
v_p(l) = k₀(l) · v(l) · e_p,k(l)
```
with gain:
```
k₀(l) = k₂ + k₃·K(l) + k₄·(1/r_t(l))
```
The gain increases with curvature K(l) (tighter corners get stronger correction) and with the inverse tube radius 1/r_t(l) (near tube walls, lateral correction strengthens). k₂, k₃, k₄ > 0 are tunable constants.

### Iterative Learning Law

After each practice lap k, the speed profile v*(l) is updated everywhere along the track. The update uses a **PD-type learning law** in the spatial domain:

```
v*_{k+1}(l) = v*_k(l) - χ(k_p · ‖e_{p,k}(l)‖ + k_d · ∇‖e_{p,k}(l)‖)
```

- **k_p > 0**: proportional gain — large steady-state error causes larger speed reduction.
- **k_d > 0**: derivative gain — growing error (drone drifting outward) causes speed reduction; shrinking error allows less reduction.
- **χ(x)**: nonlinear activation (deadzone + linear gain):
  ```
  χ(x) = k_χ(x) · (x - x_th),   χ(x) = 0 for x ≤ x_th
  ```
  with 0 < α ≤ k_χ(x) ≤ β (bounded gain).

**Interpretation of χ**: The threshold x_th separates a "safety zone" (tracking error acceptable, no speed change) from a "danger zone" (tracking error too large, reduce speed). When errors are small the drone is allowed to maintain or increase speed; when near the tube wall it slows down. This mimics a racer easing off the throttle on corner entry when they feel the car pushing toward the wall.

The learning law always reduces v*(l) when tracking error is large. The initial v*₀(l) = v_max everywhere (start at maximum speed) is not used; instead the paper initializes conservatively (2 m/s in real flights) and lets the law increase speed where error budget allows — in practice, sections of track that are easy (straights) saturate at v_max while corners stabilize at a corner-limited speed.

### Convergence Theorem

**Theorem 1** (informal): Under Assumptions 1–3 (bounded model perturbation, tube information available, initial error within tube), if the condition:

```
|1 - k_χ · k_d · γ₄| < 1
```

holds (where γ₄ is a constant derived from the closed-loop spatial dynamics), then the speed update sequence v*_k(l) is uniformly ultimately bounded (UUB) as k → ∞, and the tracking error e_{p,k}(l) converges to a neighborhood of zero.

The proof uses: (1) a differential inequality on ‖e_{p,k}(l)‖ bounding it by an integral over v*_k; (2) discrete contraction mapping on the update rule; (3) UUB via [25]'s framework. The condition on k_χ · k_d · γ₄ is a gain-tuning requirement — the derivative learning gain must not overshoot.

### Optimality Analysis

For circular arcs of radius r with tube half-width x_th/k_p, the lap time is shown to be:

```
T = L · (r + x_th/k_p) · √(k'² + (r + x_th/k_p)²) / (r · v_max)
```

As r → ∞ (straight), T → L/v_max (theoretical minimum). The ratio T/T_optimal ≈ 1 for typical parameters, confirming near-optimality. The key insight: tracking error equilibrium x_th/k_p acts as an effective radius increase, which for large track radii becomes negligible.

---

## Results

### Simulation vs. SQP Benchmark (Table 1)

The paper compares against Sequential Quadratic Programming (SQP), which solves the exact time-optimal control problem offline. Four maneuverability parameter values τ = {1, 5, 10, 30} were tested:

| τ | Spatial ILC Lap Time / SQP Lap Time | Computation (ILC / SQP) |
|---|---|---|
| 1  | 101.8% | 0.8%  |
| 5  | 100.7% | 1.1%  |
| 10 | 100.2% | 0.9%  |
| 30 | 100.3% | 0.5%  |

The spatial ILC achieves within 2% of globally optimal lap time while using less than 1.1% of the computational time. Note the ILC is model-free — it does not know τ during operation — yet matches the SQP result which has full model access.

### Soccer Field Benchmark (Table 2)

Comparison on a standardized gate-racing course against:
- HJB (Hamilton-Jacobi-Bellman, offline value function)
- HJB-RL (HJB with reinforcement learning fine-tuning)
- Supervised Learning
- Move-On-Spline

| Method | Shortest Lap Time | Average Lap Time |
|---|---|---|
| HJB-RL (best prior) | 28.99 s | 30.36 s |
| **Spatial ILC** | **24.02 s** | **24.32 s** |

Improvement: **17% faster shortest lap, 20% faster average lap** over the best prior method. Online training converges within approximately 20 iterations (12 practice laps equivalent).

### Real Quadrotor Experiments

- Environment: 30 m × 30 m outdoor area
- No system identification performed — fully model-free
- Initial speed: 2 m/s (conservative); v_max = 8 m/s
- Training time: 7 flights, total 178.71 seconds

| Iteration | Lap Time |
|---|---|
| 1 | 50.35 s |
| 4 | 20.19 s |
| Converged | ~20 s |

Lap time halved within 4 practice flights. The method is robust to real-world disturbances and model mismatch — τ was unknown throughout.

---

## Relevance to Our System

Our stack uses **min-snap polynomial trajectories** with a geometric SE(3) controller, pre-computing the entire trajectory before the race. The spatial ILC framework offers a complementary or alternative approach at two levels:

**1. Speed Profile Refinement (high relevance).** Our TOPP-style speed retiming produces a speed profile along a fixed geometric path. The spatial ILC learning law is directly analogous — it adjusts v*(l) at each arc-length position based on observed tracking error from a previous run. We could implement a post-race speed profile update step that reads the per-gate tracking error from simulation runs and reduces v*(l) in sections where error exceeds a threshold. This is essentially a data-driven TOPP that adapts to controller limitations rather than assuming perfect tracking.

**2. Helix/Corner Sections (high relevance).** Our per-gate error data shows gate-3 (a tight turn) dominates tracking error. The spatial ILC gain k₀(l) = k₂ + k₃·K(l) + k₄/r_t(l) automatically increases lateral correction effort in high-curvature regions. We could adopt a curvature-aware gain schedule in our geometric tracker's feedforward term — boost correction gains where curvature is highest to reduce cross-track error on the helix section.

**3. Virtual Tube as Safety Constraint (moderate relevance).** Our trajectory optimizer does not explicitly enforce corridor constraints. Encoding a soft tube constraint around the racing line — with radius derived from gate aperture geometry — could prevent the optimizer from generating paths that clip gate frames even under disturbances. This is especially relevant for narrow gates.

**4. Iterative Benchmarking Protocol (direct applicability).** The paper's "train like a racer" loop matches our autonomous iteration protocol exactly. The spatial ILC formalizes what we are already doing manually: run → observe error → tighten speed → re-run. We can systematize this with a per-arc-length speed update after each simulation run.

**5. Limitation on path optimization.** The paper explicitly excludes path optimization — it only optimizes speed along a fixed geometric path. Our system already has strong path optimization (min-snap + racing line). The spatial ILC contribution is therefore additive: use our existing path, add iterative speed learning on top.

---

## Actionable Takeaways

1. **Implement a spatial-domain speed profile updater.** After each benchmark run, extract per-gate (or per-arc-length-bin) tracking error. Apply the update rule: `v*_{k+1}(l) = v*_k(l) - k_p * error(l)` with a deadzone threshold x_th. This automates the speed retiming loop we currently do by hand.

2. **Add curvature-weighted lateral gain scheduling to the geometric tracker.** Compute curvature K(l) along the pre-planned trajectory. In `mpc_tracker.py`, scale the cross-track correction gain by (1 + k₃·K(l) + k₄/r_t(l)) to automatically strengthen lateral authority in corners. Start with k₃ = 0.5, k₄ = 0.2 and tune empirically.

3. **Introduce a deadzone in speed reduction.** When implementing the iterative speed update, use χ with threshold x_th ≈ 0.1–0.15 m (current avg error ~0.25 m, so this is the "safe" band). Below x_th, do not reduce speed; above it, reduce proportionally. This prevents over-conservative speed on easy sections.

4. **Initialize at v_max and use the learning law to discover feasible speed.** Instead of manually setting conservative speeds for corners, set all segment speeds to v_max in the TOPP and let one or two simulation runs of the ILC law discover which sections need slowdown. The spatial parameterization ensures this is iteration-count-efficient (converges in ~7–20 laps per the paper).

5. **Construct a virtual tube representation from gate geometry.** For each gate, compute the aperture half-width as r_t at that arc-length position. Interpolate smoothly between gates. Use this r_t(l) in the gain schedule (item 2) and as a soft constraint in trajectory optimization to prevent near-wall trajectories.

6. **Apply the PD-type update (with derivative term ∇‖e_p‖).** The derivative term is critical for convergence. Numerically, ∇‖e_p(l)‖ ≈ (‖e_p(l+Δl)‖ - ‖e_p(l)‖) / Δl — a finite difference over arc-length bins. This penalizes sections where error is growing (drone heading toward tube wall) more than sections with steady error.

7. **Impose a minimum speed floor.** In the real experiment, initial speed 2 m/s prevented crashes while the algorithm learned. In simulation, set a minimum v_min per segment (e.g., 2 m/s on straights, 1 m/s on tight turns) to prevent numerical issues if the learning law over-reduces speed.

---

## Limitations & Caveats

**2D analysis only.** The convergence theorem and optimality analysis are developed for 2D (planar) motion. The authors explicitly flag "drone racing in 3D space" as future work. Our system flies a 3D helix and banked turns — the tube radius calculation and curvature definition would need 3D generalization (using torsion and the Frenet-Serret frame).

**Path optimization excluded.** The method only optimizes speed; the geometric path is fixed. If the fixed path is suboptimal (e.g., takes a wide line through a corner when a tighter apex would be faster), the ILC cannot correct this. We must ensure our min-snap + racing_line optimizer produces a near-optimal geometric path before applying ILC-style speed learning.

**No attitude dynamics.** The drone model is a second-order mass-point with first-order velocity response (maneuverability τ). Attitude dynamics, rotor drag, and motor saturation are abstracted away. Our SE(3) controller operates at the attitude level, so the correspondence between v*(l) updates and actual attitude commands is indirect — the inner loop controller must be fast enough to track the speed commands accurately.

**Gate information assumed known.** The virtual tube requires knowledge of gate poses (to construct the generator curve). This is consistent with our known-map assumption but would fail under significant localization error. If EKF uncertainty is high (> 0.5 m), tube containment guarantees degrade.

**Convergence requires bounded perturbations (Assumption 1).** The UUB result holds only if ‖Δv(l)‖ ≤ ε·v(l), i.e., model errors are proportionally bounded. In turbulent or windy outdoor environments this may not hold. The paper's real-world tests were conducted in calm outdoor conditions.

**No collision avoidance beyond tube.** The virtual tube only prevents exiting the tube boundary. Obstacles within the tube (e.g., gate posts at grazing angles) are not handled. In narrow-gate scenarios, the tube radius r_t equals roughly half the gate aperture, and the drone must be held tightly to the centerline.

**Convergence speed depends on k_d · γ₄.** The γ₄ constant depends on the (unknown) closed-loop dynamics and must be implicitly tuned through k_d. If k_d is set too large, the update overshoots and may cause oscillation across laps. The paper does not give a systematic tuning procedure for γ₄.

---

## Key Parameters / Constants

| Parameter | Value / Range | Meaning |
|---|---|---|
| v_max | 8 m/s (real experiment) | Speed saturation limit |
| v_init | 2 m/s (real experiment) | Initial conservative speed |
| x_th | > 0 (demarcation threshold) | Deadzone for activation function χ |
| k_p | > 0 | Proportional gain in ILC update |
| k_d | > 0 | Derivative gain in ILC update; convergence requires \|1 - k_χ·k_d·γ₄\| < 1 |
| k₂, k₃, k₄ | > 0 | Components of position correction gain k₀(l) = k₂ + k₃·K(l) + k₄/r_t(l) |
| α, β | 0 < α ≤ k_χ(x) ≤ β | Bounds on nonlinear gain within χ |
| γ₄ | derived from dynamics | Coupling constant in convergence condition |
| Path discretization | 1500 points | Simulation spatial resolution |
| SQP optimality tolerance | 1e-5 | Reference optimizer precision |
| Convergence (simulation) | ~20 iterations | Laps to reach stable speed profile |
| Convergence (real) | 7 flights, 178.71 s | Total real-world training time |
| Lap time improvement | 50.35 s → 20.19 s | Iteration 1 → iteration 4 in real flight |
| Speedup vs HJB-RL | 17% (shortest), 20% (average) | Benchmark comparison on soccer field course |
| Speedup vs SQP (compute) | 89–99.5% reduction | ILC uses 0.5–1.1% of SQP computation |
| Lap time optimality gap | < 2% of SQP | ILC achieves 100.2–101.8% of optimal |

---

*Analysis written 2026-04-14. Paper: arXiv:2306.15992, Lv et al., Beihang University.*
