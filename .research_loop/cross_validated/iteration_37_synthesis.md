# Iteration 37 — Research Synthesis: S-turn TOPP Floor Optimization

## Papers Analyzed (New)
1. **GripMap** (Werner et al., 2025) — Spatially resolved constraint framework for racing
2. **Energy-Limited Minimum Lap Time** (van den Eshof et al., 2026) — Segment decomposition at curvature peaks
3. **Spline-Based Min-Curvature** (Xue et al., 2024) — B-spline curvature optimization for S-turns/chicanes

## Papers Referenced (Previously Analyzed)
4. **CiMPCC** (Li et al., 2024) — Curvature-integrated speed mapping g(K) = exp(-αK²)
5. **Jerk-Constrained TOPP** (arXiv:2404.07889, 2024) — Jerk limits in TOPP-RA via SLP
6. **FBGA** (Piazza et al., 2025) — Forward-backward with generic acceleration constraints
7. **Spatially-Aware CMA-ES** (arXiv:2602.15642, 2026) — Controller-guided spatial feedback
8. **Primitive-Planner** (arXiv:2502.16882, 2025) — TOPP-RA with LP-based reachability

## Research Consensus

### Strong consensus: Spatially-varying speed constraints are correct and beneficial
All 8 papers agree that speed/acceleration constraints should vary along the trajectory based on local geometry (curvature). GripMap demonstrates 5.2% lap time improvement from spatial constraint resolution. CiMPCC uses exp(-αK²) speed mapping. FBGA supports curvature-dependent acceleration constraints in forward-backward passes. Our TOPP floor system (helix/S-turn/easy) is a discrete approximation of this principle.

### Strong consensus: Curvature peaks are binding constraints
The Energy-Limited paper decomposes the speed profile at "corner apexes" (curvature maxima). Our data confirms this: the S-turn TOPP floor (0.65) binds at gate-3 exit segments, and the helix floor (0.72) binds at helix exit segments. These are precisely the curvature peaks of the trajectory.

### Strong consensus: Section-independent optimization is valid
The Energy-Limited paper's boundary-value decomposition suggests that speed optimization at separate curvature peaks can be done independently. Our iter 36 empirically confirmed this: helix floor changes have ZERO effect on non-helix gates (perfect isolation). If S-turn floor changes are similarly isolated, we can optimize them independently.

### Moderate consensus: Joint optimization can be time-neutral
The Spatially-Aware CMA-ES paper optimizes multiple spatial parameters jointly. Our proposed approach — raise S-turn floor (improve gate-3) while lowering helix floor (compensate time) — is a principled 2D Pareto optimization. The linearity of the helix Pareto frontier (confirmed in iter 36) makes the time compensation predictable.

## Contradictions

### None significant
All papers support the general direction. The only nuance is whether to tune discrete floors (our current approach) vs implement continuous constraint profiles (GripMap). For this iteration, discrete floors are sufficient.

## Ranked Actionable Takeaways

1. **Sweep S-turn TOPP floor** [0.65 → 0.70-0.75] to find accuracy-optimal point for gate-3 (analogous to helix sweep in iter 35)
2. **Verify S-turn/helix independence**: If S-turn floor changes don't affect helix gates (expected), the two floors can be co-optimized
3. **Time-neutral rebalancing**: For each S-turn floor increase, compute the helix floor decrease needed to maintain race time ≤ 13.98s
4. **Joint 2D sweep**: S-turn floor × helix floor grid search for minimum avg error at constant race time

## Proposed Implementation Direction

**Two-phase approach:**

**Phase A — S-turn floor sweep (quick):**
Sweep `max_compression_sturn` through [0.66, 0.68, 0.70, 0.72, 0.74] while holding helix floor at 0.72. Measure gate-3 error, race time, and check for cross-section coupling.

**Phase B — Joint rebalancing (if needed):**
If S-turn floor increase adds time beyond 14.0s, lower helix floor to compensate. Use linear Pareto relationship from iter 36 (0.02s per 0.01 helix floor) to compute exact compensation.

**Rollback criteria:**
- Revert if avg error increases > 5% at any setting
- Revert if basin switching occurs (race time > 20s)
- Select the S-turn+helix floor combination that minimizes avg error at race time ≤ 13.98s
