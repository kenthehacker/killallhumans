# Iteration 7 Research Synthesis — Speed-Aware Dynamic Feasibility for S-Turn Gates

## Papers Analyzed (New in This Iteration)
1. **TOPPQuad** (Mao et al., IROS 2024) — Time-optimal path parametrization with full rigid-body dynamics and per-motor thrust constraints
2. **Alternating Peak Optimization** (de Vries et al., ECC 2024) — Already analyzed iter 4; key insight: peak kappa (max constraint violation ratio) per segment is the correct metric
3. **Aggressiveness-Aware Control** (Colombo et al., 2026) — GP-based gain scheduling to minimize controller aggressiveness while maintaining tracking

## Previously Analyzed (Used)
4. **Realtime Min-Time Trajectories** (Teissing et al., RA-L 2024) — Boundary velocity optimization with norm-constrained thrust
5. **Leveling the Playing Field** (Kunapuli et al., 2025) — Feedforward is the single most important fix for geometric controllers
6. **TACO** (Sanghvi et al., 2025) — Controller parameters should adapt to local trajectory characteristics
7. **LMPC** (Zhao et al., IROS 2025) — Adaptive cost function for per-section aggressiveness tuning

## Current Bottleneck Analysis

**Problem**: Gates 3-4 (S-turn) have tracking errors of 0.661m/0.598m — the worst gates. Turn angles (48°/38°) are below the current 60° inflation threshold, but long approach distances (11.7m/10.5m) allow the drone to build high speed before the turn.

**Critical code finding**: The kinematic sim uses `tracker.last_desired_acceleration` directly (raw PD+feedforward, NOT tilt-clamped), with only a total acceleration cap at 15 m/s². The 0.85 rad "tilt saturation" reported in controller traces is a cosmetic artifact — it affects the recorded roll/pitch but NOT the actual simulated drone motion. The real constraint is the PD controller's finite bandwidth combined with the acceleration cap.

**Root cause**: The L-BFGS optimizer allocates time based on distance and velocity constraints, but doesn't account for the lateral acceleration demands at turns. For gates 3-4, the optimizer produces fast segments (high average speed) because the turn angles are moderate. But the PD controller (kp_xy=6, kd_xy=4) cannot redirect the drone fast enough at high speed, resulting in overshoot.

## Research Consensus

1. **Dynamic feasibility must account for both speed AND curvature** (TOPPQuad, Teissing, Alternating Peak). The centripetal acceleration required is a_c = v²/r ≈ v² × κ, where κ is path curvature. A moderate turn at high speed requires the same centripetal force as a sharp turn at low speed.

2. **Post-optimization feasibility checking is the correct approach** (Alternating Peak, TOPPQuad). Rather than distorting the L-BFGS objective (which we tried and failed in iteration 6 with curvature-speed penalty), evaluate the actual trajectory's peak accelerations AFTER optimization and selectively inflate only infeasible segments.

3. **The inflation should be proportional to the feasibility violation** (Alternating Peak: kappa ratio, TOPPQuad: thrust margin). Over-inflating wastes time; under-inflating leaves tracking errors.

## Contradictions / Tensions

- **Turn angle alone vs speed×curvature**: The iteration 6 approach (threshold at 60°) catches sharp turns. But gate-3 has only 48° turn at 11.7m approach — a speed×curvature metric would catch this.
- **Race time recovery**: Inflation at gates 3-4 will slow the race further (currently 15.28s, target <14s). The backlog suggests increasing time_weight. These goals conflict.
- **Controller vs trajectory**: Aggressiveness-Aware paper suggests reducing controller gains (less aggressive). Our problem is the opposite — we need the controller to be MORE capable at turns. But this is less relevant since the kinematic sim doesn't use tilt limits.

## Proposed Implementation Direction

**Centripetal acceleration feasibility check** (highest confidence, strongest evidence):

After L-BFGS optimization, for each gate transition:
1. Estimate average speed: v = distance / time
2. Compute curvature from gate-center turn angle: κ ≈ turn_angle / approach_distance
3. Estimate centripetal acceleration: a_c = v² × κ
4. If a_c exceeds a threshold (fraction of max_accel), inflate the segments around the gate

This extends the existing `_inflate_sharp_turns()` to use a physics-based metric (centripetal acceleration) instead of a geometry-only metric (turn angle threshold). It captures both the gate-7 case (sharp turn, moderate speed → high a_c) and the gate-3/4 case (moderate turn, high speed → high a_c).

**Backed by**: TOPPQuad (per-motor feasibility), Alternating Peak (peak kappa), Teissing (boundary velocity constraints), TACO (section-adaptive aggressiveness).

## Risk Assessment
- **Race time will increase** by ~0.5-1.0s at gates 3-4 due to inflation
- **Gate-7 should remain unaffected** since it's already handled by angle-based inflation
- **Other gates unlikely to be caught** since they have smaller turns AND shorter approaches
- **Revert criteria**: if avg error doesn't improve by 0.03m+ or race time increases by >1.5s
