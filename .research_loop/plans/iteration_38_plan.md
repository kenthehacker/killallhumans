# Iteration 38 Plan — Full Feedforward + PD Gain Optimization

## Objective
Reduce avg tracking error from 0.174m by increasing feedforward acceleration weight from 0.4 to 1.0 and optimizing PD gains. Target: 10-30% avg error improvement based on steady-state error analysis.

## Research Basis
- **Leveling the Playing Field (2025)**: feedforward is the most important single fix for geometric controllers
- **Tal & Karaman (2018)**: full differential flatness feedforward achieves 6.6cm RMS at 12.9 m/s
- **NGTC (2025)**: DFBC baseline uses kp=25, kd=11 (mass-normalized) with full feedforward
- **TACO (2025)**: gain ranges kp=[2,15], kd=[1,10]; trajectory-aware tuning improves 52%

## Steady-State Error Formula
```
error ≈ [(1-ff)*ref_acc + drag*ref_vel] / kp
```
At helix (ref_acc≈5 m/s², ref_vel≈8 m/s, drag=0.5):
- Current (ff=0.4, kp=6): error ≈ 1.17m
- Target (ff=1.0, kp=12): error ≈ 0.33m → 72% reduction

## Files to Modify
- **`scripts/benchmark.py`** lines 334-339: TrackerConfig parameters
  - `feedforward_accel`: 0.4 → sweep [0.6, 0.8, 1.0]
  - `kp_xy`: 6.0 → sweep [8, 10, 12, 15]
  - `kd_xy`: 4.0 → sweep [4, 5, 6, 7, 8]

## Algorithm
### Phase A: Feedforward sweep (at current PD gains)
1. Test ff=0.6, 0.8, 1.0 with kp=6, kd=4
2. Identify if ff=1.0 improves or if there's an optimal intermediate value
3. Fix ff at best value

### Phase B: PD gain sweep (at best ff)
4. Sweep kp_xy: [8, 10, 12, 15] at kd=4
5. Fix kp at best value
6. Sweep kd_xy: [4, 5, 6, 7, 8] at best kp
7. Select (ff, kp, kd) combination with best avg error

### Phase C: Validation
8. Run final config 3x to verify determinism
9. Compare all metrics against baseline

## Risk Assessment
- **Basin switching**: TOPP floors and racing line are not modified. Only controller gains change. The trajectory is pre-computed and deterministic. Risk: LOW.
- **Overshoot on corners**: Higher kp + ff=1.0 might overshoot on sharp turns. Mitigation: the damping ratio analysis shows ζ≥0.79 for all planned configs.
- **Max error regression**: Higher ff might increase peak error if feedforward overshoots. Monitor max_tracking_error and worst gate.
- **Race time impact**: Purely a controller change — trajectory is unchanged. Race time should remain ~14.02s.

## Rollback Criteria
- Revert if avg_tracking_error increases by >2%
- Revert if max_tracking_error increases by >5%
- Revert if gate_pass_rate drops below 100%
- Revert if crash occurs

## Test Plan
1. Run benchmark after each parameter change (~10s per run)
2. Monitor: avg_error, max_error, per_gate_errors, race_time, determinism
3. All sweeps total: ~15 configurations × 10s = ~150s
