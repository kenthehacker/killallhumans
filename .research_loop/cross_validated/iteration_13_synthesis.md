# Iteration 13 — Research Synthesis

## Bottleneck Re-Assessment

The iteration 12 diagnostic identified "trajectory optimizer non-determinism" as the #1 priority. **Empirical testing in this iteration disproved this**: 5 separate Python processes all produce identical trajectory times (13.5914142702s). The non-determinism observed in iteration 12 was likely due to incomplete code revert during that session's extensive experimentation (16+ configurations tested across 6 approaches).

**Actual bottleneck**: Trajectory planning for helix section (gate-7: 0.659m, gate-8: 0.528m tracking error). Controller tuning is exhausted in kinematic sim (proven in iteration 12). The remaining lever is trajectory shaping.

## Papers Analyzed

### 1. AROLA: Modular Architecture for Racing (arXiv:2602.02730, 2026)
- Standardized 8-layer architecture with Race Monitor benchmarking framework
- Key insight for us: consistency score metric (run N races, compute std dev) — validates our optimizer determinism
- APE mean of 0.19m achievable with MPC at 5 m/s; we're at 0.358m at ~10 m/s, suggesting room for improvement
- Not directly actionable for this iteration's bottleneck

### 2. ILMPC for Drone Racing (arXiv:2508.01103, 2025)
- Iterative Learning MPC: 6.05% improvement even on top of MPCC++ (state-of-the-art)
- **Most actionable insight**: spatially-varying cost that increases tracking precision near gates
- Convergence to local optima confirms importance of good initial trajectory (our min-snap approach)
- Suggests we should invest in trajectory quality over controller complexity

### 3. Reference-Free Racing via MPPI (arXiv:2509.14726, 2025)
- Gate progress objective eliminates need for reference trajectory entirely
- Achieves competitive or better race times than trajectory tracking on simple tracks
- Requires GPU (0.4ms desktop, 6.7ms embedded) — not feasible for current implementation
- **Key insight**: aggressive thrust utilization (sustain max thrust longer) — suggests our trajectory may be too conservative

## Cross-Paper Consensus

1. **Gates are regions, not points** (TOGT Planner, ILMPC, MPPI): All papers agree that optimal racing lines don't pass through gate centers. More aggressive corner-cutting within gate openings produces faster AND smoother paths.

2. **Trajectory quality > controller complexity** (ILMPC, MPPI): Both papers show that improving the reference trajectory (or eliminating it for direct gate-seeking) has more impact than tuning the tracking controller. This aligns with our iteration 12 finding.

3. **Speed-accuracy tradeoff is spatially varying** (ILMPC): Tracking precision should be highest near gates and relaxed between them. Our current uniform tracking is suboptimal.

## Contradictions

- MPPI paper claims reference-free is better; ILMPC paper iteratively improves references. Both are valid approaches for different settings. For us (no GPU, kinematic sim), trajectory-based is correct.

## Recommended Direction

Since controller tuning is exhausted (iter 12) and trajectory non-determinism is a non-issue (proved this iteration), the highest-impact change is **improving the racing line through the helix via more aggressive corner-cutting**:

1. The RacingLineOptimizer currently has conservative parameters (max_lateral_offset=0.4, smoothness_weight=0.3)
2. Increasing corner-cutting aggressiveness would allow the optimizer to find paths with lower curvature through the helix
3. This directly reduces the tracking demand on the PD controller without requiring controller changes
4. Research backing: TOGT Planner (gates as regions), Swift (aggressive corner cutting), ILMPC (trajectory quality over controller tuning)

Additionally, the `_inflate_sharp_turns` function's centripetal threshold (4.5 m/s²) may need tuning for the helix specifically.
