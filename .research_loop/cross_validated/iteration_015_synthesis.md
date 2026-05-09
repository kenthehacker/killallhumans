# Iteration 15 — Research Synthesis: TOPP-Style Speed Retiming

## Current Bottleneck
- Race time: 14.62s (target: <14s aspirational)
- Avg tracking error: 0.254m (target: <0.25m aspirational)
- Worst gate: gate-3 at 0.439m (S-turn)
- The trajectory pipeline uses heuristic `_compress_times` to selectively speed up easy segments
- The heuristic uses waypoint-level distance/speed estimates, not actual polynomial curvature

## Papers Analyzed

### Previously Analyzed (directly relevant)
1. **TOPPQuad** (Mao et al., IROS 2024, arXiv:2309.11637)
   - Fix geometric path, optimize speed profile separately → 40-50% faster than min-snap
   - Squared speed h(s) decouples geometry from timing
   - Forward-backward propagation with per-motor thrust constraints

2. **Sequence Modeling for TOPPQuad** (Mao et al., 2025, arXiv:2506.13915)
   - LSTM predicts TOPP speed profiles 136x faster
   - Confirms geometry-timing decoupling principle

3. **Multi-Fidelity RL Replanning** (Ryou et al., IJRR 2024, arXiv:2403.08152)
   - Binary search over scalar time scale preserving allocation ratios
   - 4.7% time reduction over baseline min-snap

### New Papers (this iteration)
4. **FBGA: Forward-Backward Generic Acceleration** (Piazza, RA-L 2025, arXiv:2509.26428)
   - Forward-backward algorithm for time-optimal velocity profiles
   - Matches optimal control within 0.11-0.36%
   - Up to 1000x faster than NLP-based TOPP
   - Key: handles generic (non-convex) acceleration constraints

5. **CPC: Complementary Progress Constraints** (Foehn & Scaramuzza, 2021, arXiv:2007.06255)
   - Joint trajectory + time-allocation optimization via complementarity constraints
   - Published in Science Robotics as truly time-optimal quadrotor planning
   - More complex than TOPP post-processing but achieves global time-optimality

## Research Consensus

All five papers agree on the fundamental principle: **geometry-timing decoupling enables significant speed improvements**. The key insights:

1. **The geometric path should be fixed first, then re-timed** (TOPPQuad, FBGA, MFRL)
2. **Forward-backward propagation is near-optimal** (FBGA: within 0.36% of OC)
3. **Dense curvature sampling is critical** — waypoint-level is too coarse (TOPPQuad: N=300+)
4. **Speed limits come from curvature × acceleration budget** — v_max(s) = sqrt(a_max/κ(s))
5. **Longitudinal acceleration limits handle speed transitions** between segments

## Proposed Approach

Replace `_compress_times` with a TOPP-RA-style retimer that:

1. Generates trajectory once to evaluate polynomials
2. Samples each segment densely (20 pts) to compute actual curvature from derivatives
3. For each segment, finds max curvature → determines speed limit
4. Runs forward-backward propagation across all segments for acceleration feasibility
5. Computes new segment times from the optimal speed profile
6. Regenerates trajectory with new times

### Why This Over Full TOPPQuad NLP

- Our simulation is kinematic (no motor model), so per-motor thrust constraints aren't needed
- Simple acceleration bounds are sufficient for our PD controller
- FBGA shows forward-backward matches NLP within 0.36%
- Implementation is ~60 lines, no external dependencies (CasADi/IPOPT not needed)

### Expected Impact
- Current _compress_times uses distance/speed heuristics → misses curvature-based opportunities
- TOPP retimer uses actual polynomial curvature → finds optimal speed for every segment
- Expected: race time 14.62→13.5-14.0s (5-8% improvement)
- Tracking error should remain stable since geometric path is preserved by min-snap regeneration

## Contradictions / Risks

- CPC (Foehn 2021) argues that decoupled approaches are suboptimal vs joint optimization. However, the gap is small for smooth paths (our racing line is already well-optimized).
- Changing segment times changes polynomial shape slightly. Need to verify tracking doesn't regress.
- The acceleration budget split (centripetal vs longitudinal) is approximate — real dynamics couple both.
