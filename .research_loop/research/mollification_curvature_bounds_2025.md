# Efficient Generation of Smooth Paths with Curvature Guarantees by Mollification
- **URL**: https://arxiv.org/abs/2512.13183
- **Authors**: Alfredo González-Calvin, Juan F. Jiménez, Héctor García de Marina
- **Year**: 2025 (submitted December 15, 2025; revised March 5, 2026)
- **Venue**: arXiv (cs.RO, eess.SY)

---

## Key Contribution

The paper introduces a **mollification-based path smoothing** framework that converts piecewise-linear (non-differentiable) waypoint sequences into smooth, infinitely differentiable curves with **provably bounded curvature**. The central contributions are:

1. **Formal curvature bound derivation** (Equation 8): given a waypoint sequence and a desired maximum curvature κ_max, the required mollification parameter ε can be computed analytically — no iterative solving or numerical search is needed.
2. **Closed-form curvature expression** for two-segment paths (Equation 7), from which the bound is derived.
3. **Proof that mollification is computationally cheaper** than standard interpolation (B-splines, Bézier curves) because the mollifier has compact support and no global solve is required.
4. **Preservation theorems**: mollified paths are contained in the convex hull of the original, have equal or shorter arc length, and preserve monotonicity, convexity, and quasiconvexity.
5. **Real-time suitability** on embedded/microcontroller hardware.

The key insight distinguishing this from standard spline smoothing: curvature bounds are *derived from geometry* (segment vectors and turn angle), not tuned empirically.

---

## Technical Approach

### Problem Formulation (Problem 1)

Given a piecewise-linear path f composed of p segments, find a path operator T_ε(f) satisfying:
1. T_ε(f) → f as ε → 0 (recovers original path in the limit)
2. T_ε(f) ∈ C^p(ℝ, ℝ^n) for p ≥ 2 (twice continuously differentiable, as required by trajectory trackers)
3. T_ε(f) is computationally simple to evaluate

### Mollification Operator

The mollifier φ is a smooth, compactly supported bump function. The canonical example (Equation 1):

```
φ(x) = { c₁ · exp(-1 / (1 - x²))   for |x| < 1
        { 0                           for |x| ≥ 1
```

where c₁ > 0 is the normalization constant ensuring ∫φ dλ = 1.

The scaled mollifier is:

```
φ_ε(x) = (1/ε) · φ(x/ε)
```

with support [-ε, ε].

The mollified path is:

```
F_ε(t) = (f * φ_ε)(t) = ∫_{-ε, ε} f(t - s) φ_ε(s) ds
```

Because φ_ε has compact support, only a local window of f contributes to each evaluation — this is why it is O(1) per query rather than O(n).

### Derivatives Without Differentiating f

A critical property: even if f is not differentiable (piecewise linear), the derivatives of F_ε exist and can be computed by differentiating the mollifier instead:

```
F_ε'(t)  = (f * φ_ε')(t)
F_ε''(t) = (f * φ_ε'')(t)
```

For a two-segment path with segments P̃₁ = P₁ - P₀ and P̃₂ = P₂ - P₁ meeting at junction t = 1:

```
F_ε''(t) = φ_ε(t - 1) · (P̃₂ - P̃₁)
```

This is proportional to the **direction change** at the junction, scaled by the mollifier value at that point. Far from junctions, F_ε'' = 0.

### Curvature Formula (Equation 7)

Define:
- A₁(t) = ∫_{(-∞, 1)} φ_ε(t - s) ds  (portion of mollifier weight on segment 1 side)
- A₂(t) = ∫_{(1, ∞)} φ_ε(t - s) ds   (portion on segment 2 side)
- A₁(t) + A₂(t) = 1 always

Then the curvature at t is:

```
κ(t) = φ_ε(t - 1) · ‖P̃₁ ∧ P̃₂‖₂ / ‖P̃₁·A₁(t) + P̃₂·A₂(t)‖₂³
```

where ∧ denotes the cross product (in 2D: scalar wedge product). The numerator is proportional to the sine of the turn angle; the denominator is the cube of the velocity magnitude. Peak curvature occurs at t = 1 (the junction itself), where φ_ε is maximized.

### Curvature Upper Bound (Equation 8) — the Key Result

```
κ(t) ≤ (1/ε) · ‖φ‖_∞ · ‖P̃₁ ∧ P̃₂‖₂ · M(P̃₁, P̃₂)
```

where:
```
M(P̃₁, P̃₂) = 1 / ‖P̃₁·s̄ + P̃₂·(1 - s̄)‖₂³
```

and:
```
s̄ = ⟨P̃₂ - P̃₁, P̃₂⟩ / ‖P̃₂ - P̃₁‖₂²
```

This bound is **exact** (not just an upper bound) when s̄ ∈ [0, 1], which holds for non-degenerate configurations.

**Physical interpretation of each factor:**
- `1/ε`: larger ε smooths more aggressively, reducing curvature
- `‖φ‖_∞`: peak value of the mollifier (fixed once the kernel is chosen)
- `‖P̃₁ ∧ P̃₂‖₂`: encodes the sine of the turn angle — larger turn angle → larger curvature
- `M(P̃₁, P̃₂)`: encodes the effect of segment lengths — longer segments → smaller M → smaller curvature

**To design for κ ≤ κ_max:** invert Equation 8 to solve for ε:

```
ε ≥ (‖φ‖_∞ / κ_max) · ‖P̃₁ ∧ P̃₂‖₂ · M(P̃₁, P̃₂)
```

### Multi-Segment Extension

For a path with p ≥ 2 junctions:
1. Apply Equation 8 to each consecutive pair of segments to compute ε_i
2. Use ε = max_i(ε_i)
3. **Validity condition**: ε_i < 1/2 for all i (so that smoothing zones around adjacent junctions do not overlap)

When ε_i < 1/2, the mollification at junction i depends only on the two segments at that junction — other segments do not contribute. This locality property is what makes the bound exact.

### Geometric / Preservation Properties

- **Convex hull containment**: F_ε(t) ∈ co{f(t)} — the mollified path never leaves the convex hull of the original waypoints.
- **Arc length**: L(F_ε) ≤ L(f) — smoothing does not lengthen the path.
- **Affine invariance**: mollification commutes with affine transformations.
- **Monotonicity preservation**: if f is monotone in any coordinate, F_ε is too.
- **Convergence rate**: F_ε → f uniformly as ε → 0; for Lipschitz f, F_ε is also Lipschitz for any ε > 0.
- **Uniform agreement on sub-intervals**: for ε ∈ (0, 1/2), F_ε(t) = f(t) exactly on intervals at least 1 - 2ε away from any junction.

### Reparametrization Warning (Section 4.4)

Mollification does not behave well under nonlinear reparametrization. Even for affine reparametrizations, ε does not transform cleanly. This means the curvature bound must be computed in the **parametric domain used for mollification**, not after time-rescaling. For arc-length parametrization, ε has direct geometric meaning (meters). For time parametrization, the curvature bound reflects curvature with respect to time, not space.

---

## Results

The paper's experimental section (Section 6) includes:

- **2D planar navigation** with polygonal obstacles — mollified paths satisfy curvature bounds at all junctions
- **3D path mollification** (Section 6.3) — extension to three-dimensional waypoint sequences
- **Comparison with interpolation methods** (Section 6.1): B-splines, Bézier curves, Gaussian-process-based smoothing
- **Path-following algorithm integration** (Section 6.2): drop-in compatibility with standard trajectory trackers (the paper's primary stated target is systems requiring C² paths)
- **Embedded/microcontroller experiment** (Section 6.4): real-time demonstration, though specific timing numbers were not extracted from the HTML render

Claimed advantages over interpolation:
- Computationally cheaper per evaluation (compact support = O(1) vs O(n) for global splines)
- No global linear system solve
- Curvature is bounded, not just smooth

The paper does not report UAV experiments, race-track scenarios, or lap-time benchmarks.

---

## Relevance to Our System

### Our Problem

We use minimum-snap polynomial trajectory generation. Waypoints are placed at fixed `ENTRY_EXIT_OFFSET = 0.4m` along gate normals on both sides of each gate. At sharp turns (e.g., the helix or S-turns with 49–94° direction changes), consecutive entry/exit waypoints are often only `0.8m` apart (entry of gate N and exit of gate N−1 are close). The min-snap polynomial must fit through these closely spaced waypoints, producing very high curvature that:

1. Exceeds the controller's tracking capability (leading to overshoot and cross-track error)
2. Forces very short segment times that compound into feasibility failures
3. Creates polynomial curvature spikes at waypoint junctions that can cause near-crashes

### What Mollification Theory Tells Us

The curvature bound (Equation 8) directly quantifies the relationship we've been treating empirically:

```
κ_max ≤ (1/ε) · ‖φ‖_∞ · sin(θ) · |P̃| / ‖...‖³
```

For two segments of equal length L meeting at turn angle θ:

```
κ_max ∝ sin(θ) / (ε · L²)
```

This says: **curvature scales as sin(θ) / L²** (inversely as the square of segment length, linearly with turn angle). For our 0.4m offsets and 90° turns, curvature is ~6.25 times higher than it would be with 1.0m offsets. This is why increasing ENTRY_EXIT_OFFSET dramatically helps.

More precisely, doubling the offset from 0.4m to 0.8m reduces polynomial curvature by a factor of ~4x (L² scaling), not 2x. This quadratic relationship is **not obvious from the min-snap formulation alone** but falls directly out of the mollification curvature bound.

### Applicability

The paper treats paths as piecewise-linear inputs, mollifying them into smooth outputs. Our min-snap polynomials are already smooth everywhere except at waypoint junctions where boundary conditions are enforced. The analogy is:

- Our "junction" is the matched boundary condition between adjacent polynomial segments
- Our "turn angle" is the angle between consecutive approach directions
- Our "segment lengths" are the entry/exit offset distances (0.4m)

The mollification framework provides a **mathematical model** for why short offset distances cause disproportionately large curvature peaks — the L² inverse scaling — which our current compound-curvature heuristics (20% S-turn boost, 15% helix boost) are implicitly approximating but not deriving.

### The ε-to-Offset Mapping

In our setting, ε corresponds to the entry/exit offset distance. Larger ε (longer offsets) → smoother turns → lower curvature. The bound says:

```
ε ≥ (‖φ‖_∞ / κ_max) · ‖P̃₁ ∧ P̃₂‖₂ · M(P̃₁, P̃₂)
```

We could compute the **required minimum offset** for each gate given its turn angle and approach speed, rather than using a global 0.4m constant. This would give principled, per-gate offset distances instead of heuristic compound-curvature boosts.

---

## Actionable Takeaways

1. **Replace the global `ENTRY_EXIT_OFFSET = 0.4m` with a per-gate, curvature-derived offset.** Use Equation 8 inverted: for gate i with turn angle θ_i and approach segment length L, compute the minimum offset d_i such that κ(d_i, θ_i) ≤ κ_max. κ_max = max_body_rate / max_velocity ≈ 6.0/15.0 = 0.4 rad/m or determined from controller bandwidth. This eliminates the need for compound-curvature empirical boosts.

2. **Adopt the L² scaling law for offset design.** Since κ ∝ 1/L², to halve the curvature at a sharp gate you must increase the offset by √2, not 2. Apply this to: sharp helix gates (current inflation of 1.10–1.15 may need to move the physical offset, not just inflate time). Specifically, the 90° helix gates at ~4m inter-gate distances may need offsets of 0.8–1.0m rather than 0.4m.

3. **Use the Menger curvature formula in the TOPP retimer.** We already compute `k = 2 * cross_mag / (n1 * n2 * chord)` (Menger curvature). The mollification paper confirms this is precisely the right quantity — it equals κ(t=junction) from Equation 7. Our implementation is correct. No change needed here.

4. **Gate-adaptive mollification parameter ε as a pre-processing step.** For each junction (entry → gate → exit waypoints), compute ε_i from the turn angle and desired κ_max, then set the physical offset distance to ε_i. For sharp turns, this will automatically produce larger offsets. For straight or shallow gates, 0.4m is already sufficient (ε is small).

5. **Replace the 20% S-turn compound curvature boost and 15% helix boost with geometry-derived boosts.** At S-turn junctions, the effective turn angle seen by the polynomial is the sum of consecutive turn angles (the drone's lateral velocity must reverse). Use the compound angle formula: θ_effective = θ_1 + θ_2. Plug into Equation 8. This yields principled boosts rather than empirically tuned constants.

6. **Verify the ε < 1/2 non-overlap condition.** In our parametrization, the junction is at the gate center. Entry and exit are both 0.4m away. If inter-gate distance is < 1.0m, the mollification zones from adjacent junctions can overlap, invalidating the curvature bound. For the helix with 3.6–5.7m inter-gate distances and 0.4m offsets, we are safely within the non-overlap condition (0.4 << 1.8m half-inter-gate).

7. **3D mollification is exact.** The paper explicitly covers 3D paths (Section 6.3). The curvature bound extends identically since the cross product ‖P̃₁ ∧ P̃₂‖₂ is naturally 3D. No 2D approximation is needed.

8. **Use mollification for online re-planning.** If a gate is detected at a different position than expected, a mollified correction to the waypoint sequence avoids rerunning full min-snap optimization. The compact support means only the two segments adjacent to the corrected gate change.

---

## Limitations & Caveats

1. **Piecewise-linear input assumption.** The paper assumes the input path f is piecewise linear. Our min-snap polynomials are smooth everywhere except at junction constraints. The mollification theory applies directly to the waypoint-to-waypoint geometry, not to the polynomial itself. We must be careful not to conflate the mollification parameter ε with time parametrization in min-snap.

2. **No time parametrization.** Mollification gives curvature of a spatial path parametrized by a scalar t (arc length or junction index). Our trajectories are time-parametrized. The curvature bound gives spatial curvature κ_spatial; actual centripetal acceleration is a_c = v² · κ_spatial where v is speed. The paper does not address how to incorporate speed variation — we still need the TOPP retimer for that.

3. **No dynamics.** The paper is purely geometric — it says nothing about the drone's thrust limits, drag, or body-rate constraints. The curvature bound must be combined with our existing kinodynamic feasibility checks, not substituted for them.

4. **Kernel normalization constant ‖φ‖_∞ must be computed.** For the canonical mollifier φ(x) = c₁ exp(-1/(1-x²)), the constant c₁ ≈ 2.252 (numerical). This must be known to apply Equation 8 quantitatively. Different kernel choices give different ‖φ‖_∞ values.

5. **Reparametrization issue.** As noted in Section 4.4, mollification does not commute with nonlinear time reparametrization. We cannot mollify in the physical time domain and then retiming the result — we must mollify in the geometric (arc-length or junction-indexed) domain and apply TOPP separately.

6. **No experimental UAV or race results.** The paper targets ground robots and mobile systems with Dubins-style turning constraints. There are no drone-racing benchmarks, no gate-passing experiments, and no comparison against min-snap or time-optimal polynomial planners. The applicability claim is theoretical.

7. **Overlap condition (ε < 1/2) limits sharp-turn handling.** If required ε > 0.5 (the curvature at the junction is so high that the smoothing zone would need to extend beyond the halfway point to the next junction), the multi-segment bound breaks down and a global solve is required. This could occur for very sharp turns with very short segments — exactly the difficult case we face.

8. **2D vs. 3D curvature definition.** In 3D, the curvature of a space curve is the magnitude of the curvature vector (Frenet-Serret). The paper extends to 3D via the vector cross product, but the curvature bound still applies to the spatial curvature κ, not the component curvatures in individual axes. Our min-snap solves each axis independently; the joint 3D curvature must be checked post-hoc.

---

## Key Parameters / Constants

| Parameter | Value / Expression | Meaning |
|-----------|-------------------|---------|
| Mollifier peak ‖φ‖_∞ | c₁ · e^{-1} (numerical: ~0.826 · c₁) | Peak value of canonical mollifier φ |
| Normalization constant c₁ | ≈ 2.252 (numerical) | Ensures ∫φ dλ = 1 for canonical kernel |
| ‖φ‖_∞ (combined) | ≈ 2.252 × 0.826 ≈ 1.86 | Effective peak of unit-normalized mollifier |
| Non-overlap condition | ε < 0.5 (in junction-indexed units) | Required for per-junction curvature bound to be exact |
| Curvature bound | κ ≤ (1/ε) · ‖φ‖_∞ · ‖P̃₁ ∧ P̃₂‖₂ · M(P̃₁, P̃₂) | Main result (Equation 8) |
| Arc length preservation | L(F_ε) ≤ L(f) | Mollification does not lengthen paths |
| Convergence rate | Uniform on compact sets | Standard mollification theory |
| ε design formula | ε ≥ ‖φ‖_∞ · ‖P̃₁ ∧ P̃₂‖₂ · M / κ_max | Invert Eq. 8 to design for κ_max |
| s̄ (weight parameter) | ⟨P̃₂ - P̃₁, P̃₂⟩ / ‖P̃₂ - P̃₁‖₂² | Used in computing M(P̃₁, P̃₂) |

**Scaling intuition for equal-length segments of length L at turn angle θ:**
```
κ_max ≈ (‖φ‖_∞ / ε) · sin(θ) / L²
```

For our system (θ = 90°, L = 0.4m):
```
κ_max ≈ 1.86 / ε · 1.0 / 0.16 ≈ 11.6 / ε
```

For κ_max = 0.4 rad/m (controller limit), required ε ≈ 29m — far exceeding the entry/exit offset. This confirms that 0.4m offsets are **fundamentally insufficient** for 90° turns at our controller bandwidth. The offset must be ~1.5–2.0m for sharp gates, consistent with the "On Your Own" paper's recommendation of 1.25m for Split-S maneuvers.
