# Iteration 2 Research Synthesis — Threshold Tightening + Trajectory Optimization

## Bottleneck: threshold_tightening
Current performance already meets aspirational targets. The opportunity is to:
1. Tighten benchmark thresholds to lock in gains
2. Improve trajectory quality for even better metrics, setting up for faster race times

## Papers Analyzed

### 1. TOGT Planner (Qin et al., ICRA 2024) — arxiv:2309.06837
**Key insights for our system:**
- Gates should be treated as regions, not points. Optimizing traversal position within gate openings saves several % on lap time
- Curvature-aware time allocation: the optimizer naturally allocates more time to sharp turns. Their initialization uses distance/nominal_speed, similar to ours
- Actionable: replace uniform 0.6 speed factor with curvature-aware per-segment allocation
- "Short, high-curvature segments naturally require more time to stay within thrust limits"
- Initial time allocation: `T_i = distance_i / nominal_speed + k_curv * turn_angle_i`

### 2. Leveling the Playing Field (Kunapuli et al., 2025) — arxiv:2506.17832
**Key insights for our system:**
- **Feedforward is the single most important fix** for geometric controllers. Missing reference acceleration terms are the largest contributor to tracking error
- Our GeometricTracker already has feedforward_accel=0.8, but could benefit from full feedforward (1.0)
- GC achieves lower steady-state error than RL — ideal for gate-passing accuracy
- Bayesian gain tuning closes remaining gap with RL
- "Full feedforward requires up to 4th-order position derivatives" — our min-snap polys provide these

## Consensus Across Papers
1. **Time allocation matters**: Both TOGT and the survey literature agree that per-segment time optimization is critical for racing performance
2. **Feedforward is essential**: Multiple papers confirm that GC without feedforward underperforms; with full feedforward it's competitive with RL
3. **Curvature drives time allocation**: Tighter turns need more time, straight segments can be faster

## Contradictions
- None significant for this iteration's scope

## Proposed Implementation Direction
Three targeted changes, ordered by risk (lowest first):

1. **Tighten thresholds** (zero risk, just config)
2. **Curvature-aware time allocation** in trajectory_optimizer.py (medium confidence, backed by TOGT)
3. **Increase feedforward weight** from 0.8 to 1.0 in benchmark tracker config (backed by "Leveling the Playing Field")

## Expected Impact
- Avg tracking error: 0.333m → likely 0.25-0.32m (from feedforward improvement)
- Gate-12 error: 0.693m → potentially 0.4-0.5m (from better last-segment time)
- Overall: solidify all aspirational thresholds, prepare for race time reduction
