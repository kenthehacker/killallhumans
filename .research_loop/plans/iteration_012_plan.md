# Iteration 12 — Implementation Plan: Trajectory-Aware Gain Scheduling

## Objective
Reduce helix tracking error (gate-7: 0.659m → <0.50m) by scheduling controller gains based on trajectory curvature, while maintaining straight-segment performance and race time.

## Research Basis
- TACO (Sanghvi 2025): trajectory-aware gain adaptation at 2Hz, 62% error reduction on hard trajectories
- Task-Parameter Nexus (Shen 2024): P-gain should increase for aggressive turns; D-gain should increase less to avoid oscillation
- RL Gain Scheduling (2024): 43-49% ISE reduction from adaptive gains
- Aggressiveness-Aware Control (Colombo 2026): formal framework for minimum-gain tracking bounds

## Files to Modify
1. **`scripts/benchmark.py`** — Add gain scheduling logic in the synthetic sim loop (lines ~296-406)

## Algorithm Changes

### In `run_synthetic_benchmark()`:

1. **Before the sim loop**, initialize EMA state:
```python
curvature_ema = 0.0  # smoothed curvature signal
base_kp_xy = 6.0     # base gains (from TrackerConfig)
base_kd_xy = 4.0
```

2. **Inside the sim loop** (after computing ref_ahead), compute gain schedule:
```python
# Curvature signal: acceleration magnitude at lookahead point
acc_mag = np.linalg.norm(ref_ahead.acceleration)

# Smooth with EMA to avoid segment-boundary spikes
ema_alpha = 0.05  # ~20 step window at dt=0.01
curvature_ema = ema_alpha * acc_mag + (1 - ema_alpha) * curvature_ema

# Normalize to [0, 1]
acc_scale = 30.0  # reference: moderate turn
curvature_score = min(1.0, curvature_ema / acc_scale)

# Schedule gains (TACO-inspired)
kp_boost = 1.0 + 0.5 * curvature_score  # kp: 6.0 → 9.0
kd_boost = 1.0 + 0.25 * curvature_score  # kd: 4.0 → 5.0

# Apply to tracker config
tracker.config.kp_xy = base_kp_xy * kp_boost
tracker.config.kd_xy = base_kd_xy * kd_boost
```

3. Gains revert to base values on straight segments automatically via the EMA decay.

## Sweep Parameters
- kp_boost_max: [0.3, 0.5, 0.8, 1.0] (i.e., kp max = 7.8, 9.0, 10.8, 12.0)
- kd_boost_max: [0, 0.25, 0.5] (i.e., kd max = 4.0, 5.0, 6.0)
- acc_scale: [20, 30, 50] (sensitivity to curvature)
- ema_alpha: [0.05, 0.1, 0.2] (smoothing aggressiveness)

## Risk Assessment
- **Regression on straights**: Low risk — gains unchanged when curvature_score ≈ 0
- **Oscillation during turn recovery**: Medium risk — mitigated by conservative kd boost
- **Gate pass regression**: Low risk — gain boost helps tracking, shouldn't break gate detection

## Rollback Criteria
- If avg_tracking_error increases by >5% vs baseline (0.358m → >0.376m), revert
- If any gate is no longer passed, revert immediately
- If race time increases by >0.5s, revert

## Test Plan
1. Implement base version with conservative parameters
2. Run benchmark
3. If improvement, sweep parameters for optimal configuration
4. If no improvement, try alternative: reduce feedforward during saturation instead
