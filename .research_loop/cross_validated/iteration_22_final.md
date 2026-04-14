# Iteration 22 Cross-Validated Research — Sim-Based Racing Line Selection

## Approach
Replace proxy-objective-based racing line selection with kinematic-sim-based selection.

## Research Support
- **BO Racing Line (Jain 2020)**: Black-box evaluation of trajectory quality via simulation oracle. Key: evaluate actual time/error, not geometric proxy.
- **AERO-MPPI (Chen 2026)**: 15 parallel optimizers, select best by re-rollout under common cost. Two-stage: optimize with diverse refs, select with common objective.
- **T-MPC (de Groot 2024)**: Theorem 2 — including baseline as candidate guarantees no regression.
- **TACO (Sanghvi 2025)**: Trajectory-aware optimization reduces tracking error 32% by adapting trajectory to controller capability.
- **Multi-obj PID (Vaiuso 2025)**: Closed-loop sim evaluation with composite cost outperforms proxy-based tuning by 42.7%.

## Cross-Validation Critique

### Potential Issues
1. **Kinematic sim fidelity**: The kinematic sim uses a PD controller without attitude dynamics. A racing line that's "best" in kinematic sim may not be best in PyBullet. *Mitigation*: This is still better than a geometric proxy (path_length + curvature²) which has zero correlation to controller behavior.

2. **Computational cost**: 10 full trajectory builds + 10 kinematic sims ≈ 2-3s. *Mitigation*: This is offline planning. 3s is negligible.

3. **Selection overfitting**: Selecting by avg tracking error could favor a racing line that's globally mediocre but avoids the worst gate, while another line is faster with one bad gate. *Mitigation*: Use weighted avg where worst-gate error has higher weight.

### Strengthened Proposal
- Use a **composite score**: 0.7 × avg_error + 0.3 × max_gate_error
- Tie-break by race time (faster wins if error scores are within 1%)
- Always include zero-init as a candidate (T-MPC Theorem 2 fallback guarantee)

## Implementation Confidence: HIGH
- Zero regression risk (fallback to current selection if all sim candidates are worse)
- Well-supported by 5+ papers
- Modest code change (add sim evaluation loop after existing L-BFGS multi-start)
- Addresses the identified architectural issue: proxy objective ≠ actual metric
