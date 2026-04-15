# Iteration 45 — Research Synthesis: TOPP Acceleration Budget Increase + Floor Reduction + Gate-3 ILC

## Current Problem
Race time at 13.51s with avg tracking error 0.159m. Gate-3 at 0.226m is the binding constraint (0.024m headroom). The system has 0.091m avg error headroom (0.159m vs 0.25m threshold). After halving inflation in iter 44, TOPP compression floors and acceleration budgets are the next speed-limiting layers.

## Research Consensus (from 121+ analyzed papers)

### 1. TOPP acceleration budgets are conservative
- **TOPPQuad (Mao 2024)**: Uses full quadrotor dynamics for TOPP, showing that simplified centripetal/longitudinal budgets leave 15-20% speed on the table vs full-dynamic TOPP.
- **FBGA (Piazza 2025)**: Forward-backward algorithm matches OC baselines within 0.36%. The key: acceleration budgets should match physical limits, not conservative proxies.
- **Physical analysis**: max_tilt=0.85 rad → g*tan(0.85) ≈ 11.4 m/s² centripetal capacity. Current budget a_centripetal=10.0 uses only 88% of physical capacity. Increasing to 11.0 recovers 5% turn speed (sqrt(11/10) = 1.049).
- **Drag-aware planning (Zhang 2024)**: Aerodynamic drag at high speed provides restoring force that acts like additional damping, supporting higher centripetal budgets.

### 2. Compression floors bind less after inflation halving
- **Iter 17 lesson**: "max_compression floor is the binding constraint, not curvature-based speed limits." But this was pre-inflation-halving. With iter 44's 50% inflation reduction, input times to TOPP are 4% shorter, meaning floors (which are ratios) produce lower absolute minimums.
- **FBGA (Piazza 2025)**: Forward-backward should converge to the true speed limit, not an artificial floor. Floors exist as safety margins for controller tracking, but with ILC compensation, the margin can be reduced.
- **Iter 21 lesson**: "easy floor only saved 0.01s" — but system was fundamentally different (no ILC, higher inflation).

### 3. ILC alpha increase for gate-3 is well-supported
- **Schoellig 2012**: ILC convergence rate proportional to alpha. At alpha=0.4, convergence is gradual; 0.45 would be ~12.5% faster convergence.
- **Bristow & Alleyne 2007**: Q-filter bandwidth and learning gain jointly determine convergence. Current inflection cutoff=0.40 Hz with alpha=0.40 is conservative for a section with high-frequency content (centripetal reversal).
- **Track-centric ILC 2026**: Section-specific ILC naturally adapts to section difficulty. Higher alpha in difficult sections (like the S-turn inflection) is the intended use.
- **Iter 26 lesson**: alpha=0.5/0.6 for ALL sections caused saturation. But alpha=0.45 for ONLY the inflection section (steps 200-440) is much more conservative.

### 4. Per-gate headroom analysis (post iter-44)

| Gate | Error | Headroom | Can Absorb Speed Increase? |
|------|-------|----------|---------------------------|
| gate-1 | 0.113m | 0.137m | Yes — lots of room |
| gate-2 | 0.216m | 0.034m | Minimal — needs protection |
| gate-3 | 0.226m | 0.024m | **NO — binding constraint** |
| gate-4 | 0.190m | 0.060m | Small |
| gate-5 | 0.161m | 0.089m | Moderate |
| gate-6 | 0.080m | 0.170m | Yes — lots of room |
| gate-7 | 0.186m | 0.064m | Small |
| gate-8 | 0.192m | 0.058m | Small |
| gate-9 | 0.117m | 0.133m | Yes |
| gate-10 | 0.151m | 0.099m | Moderate |
| gate-11 | 0.126m | 0.124m | Yes |
| gate-12 | 0.170m | 0.080m | Moderate |

Gate-3 ILC improvement is essential before further speed increases.

## Proposed Direction
**Three-pronged speed recovery:**
1. Increase TOPP acceleration budgets (a_centripetal 10.0→11.0, a_longitudinal 8.0→9.5) — directly allows faster speeds
2. Reduce TOPP compression floors (helix 0.72→0.68, easy 0.59→0.55, protected 0.65→0.61) — removes artificial minimums
3. Increase gate-3 ILC alpha (0.4→0.45) — compensates for tracking error increase at the binding constraint

## Contradictions & Risks
- **Risk**: Gate-3 may still exceed 0.25m even with ILC alpha increase if speed increase is too aggressive. Mitigation: The ILC alpha increase should reduce gate-3 error first, creating headroom.
- **Risk**: Iter 17 showed floors were binding, not curvature limits. The acceleration budget increase may have no effect if floors still bind. But with halved inflation + floor reduction, geometry-optimal times may now dominate.
- **Risk**: Multiple simultaneous changes make it hard to attribute improvements. Accept this tradeoff for speed of iteration (6 remaining).
- **Contradiction**: Iter 39 showed S-turn floor decrease caused basin switching. We keep S-turn floor at 0.70.
