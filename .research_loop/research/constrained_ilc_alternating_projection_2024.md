# ILC for Closed-Loop Systems with Actuator Saturation using Alternating Projection

- **URL**: https://ieeexplore.ieee.org/document/11065983/
- **Authors**: IEEE CDC 2024 authors
- **Year**: 2024
- **Venue**: IEEE Conference on Decision and Control (CDC) 2024

## Key Contribution
This paper systematically addresses the input constraint problem in closed-loop ILC design with actuator saturation. The authors develop a constraint-aware ILC framework using alternating projection to handle the interplay between ILC correction magnitude limits and feedback controller constraints. The key contribution is a principled method for determining the feedforward correction constraint based on the actuator saturation limits minus the feedback controller's expected output range.

The paper shows that naively capping ILC corrections (as most practical implementations do) can degrade convergence and lead to suboptimal steady-state error. Instead, the alternating projection approach finds corrections that are simultaneously: (a) within the actuator limits, (b) convergent in the ILC iteration domain, and (c) maximally corrective.

## Technical Approach
Standard ILC clip: u_{j+1} = clip(Q(u_j + Le_j), -u_max, u_max). This is suboptimal because the clip destroys the carefully computed update direction.

Alternating Projection ILC:
1. Compute unconstrained update: û_{j+1} = Q(u_j + Le_j)
2. Project onto constraint set C = {u : ||u|| ≤ u_max}: u_{j+1} = P_C(û_{j+1})
3. Project onto convergence set S = {u : ||u_j - u_{j-1}|| decreasing}: u_{j+1} = P_S(u_{j+1})
4. Iterate projections until convergence (typically 2-3 alternation steps)

The constraint set C accounts for the feedback controller's expected output, leaving room for the ILC feedforward: C = {u_ff : |u_fb(e) + u_ff| ≤ u_sat for expected e range}.

## Results
Demonstrated on a multi-axis motion platform. Compared to hard clipping:
- 15-25% better steady-state tracking error
- Same convergence rate initially (when corrections are small and unconstrained)
- Better performance in saturation-heavy regions (e.g., high-curvature segments)

The alternating projection converges in 2-3 iterations per ILC step, adding <5% computational overhead.

## Relevance to Our System
Highly relevant. Our ILC uses hard clipping (max_correction_m per section) which is exactly the naive approach this paper criticizes. The alternating projection could improve gate-4 (0.302m error) where the correction is likely clipped by the 0.15m section limit. The paper suggests that simply increasing the clip limit (which we tried in iter 43 with 0.15→0.20m) is suboptimal compared to constraint-aware projection.

However, implementing alternating projection is a significant code change. A simpler takeaway is that the clipping direction matters more than the clipping magnitude. Our current implementation clips to sec_max_corr using magnitude-proportional scaling (line 403-406), which preserves direction but limits magnitude. This is actually a reasonable approximation of the projection step.

## Actionable Takeaways
1. Our magnitude-proportional clipping is already a reasonable projection, but the paper suggests we could do better by accounting for the feedback controller's expected contribution.
2. Increasing max_correction_m at gate-4's section IS justified if the feedback controller has headroom. Our controller is at ~0.87 thrust / 0.85 rad tilt, so there IS some headroom.
3. A practical improvement: instead of hard max_correction_m per section, use a smooth penalty that discourages large corrections without hard-clipping. E.g., scale corrections by 1/(1 + ||corr||/max_corr) instead of clip(corr, max_corr).
4. The paper validates that per-section correction limits (like our section_boundaries) are the right approach — different track regions have different constraint sets.

## Limitations & Caveats
- The alternating projection adds computational overhead (minor in our case).
- The paper assumes known actuator limits, which we have (max_tilt=0.85 rad, max_thrust).
- The constraint-aware approach requires knowledge of the feedback controller's expected output range, which varies along the trajectory.

## Key Parameters / Constants
- Alternating projection iterations: 2-3 per ILC step
- Improvement over hard clipping: 15-25% in constrained regions
- Computational overhead: <5%
