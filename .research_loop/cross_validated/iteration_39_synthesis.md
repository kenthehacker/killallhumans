# Iteration 39 Research Synthesis — ILC-Controller Gain Mismatch

## Current Bottleneck
Control: gate-2 regression (+22%) after iteration 38's PD gain + feedforward upgrade.

## Critical Code Finding: ILC Gain Mismatch

The `compute_ilc_offset_table()` function in `planning/trajectory_optimizer.py` (lines 218-221)
uses **stale controller gains** from before iteration 38:
```python
kp_xy, kd_xy = 6.0, 4.0  # OLD — iter 38 changed to 7.0, 5.5
kp_z, kd_z = 8.0, 5.0
ff_accel = 0.4             # OLD — iter 38 changed to 0.50
```

The benchmark runner uses updated gains: kp_xy=7.0, kd_xy=5.5, ff_accel=0.50.

### Impact Analysis
The ILC computes correction offsets by running a kinematic sim with the OLD weaker controller,
measuring systematic tracking error, then generating position offsets to compensate. These
offsets are then applied in the benchmark which uses the NEW stronger controller.

The mismatch means:
1. **ILC error estimates are wrong**: The old controller (kp=6, ff=0.4) produces different
   error patterns than the new one (kp=7, ff=0.50). The systematic error being corrected
   may not match what the new controller actually produces.
2. **Over-correction at gate-2**: The old controller had lower feedforward, meaning less
   overshoot at straight→turn transitions. The ILC corrections computed for the old controller
   may ADD to the new controller's already-higher feedforward overshoot instead of correcting it.
3. **Under-correction at gates 3-8**: The new controller tracks better at these gates, so
   ILC corrections sized for the old controller may be too large.

### Why Gate-2 Specifically?
Gate-2 is in the pre-inflection ILC section (t=0-2.0s, alpha=0.4, max_correction=0.15m,
cutoff=0.35Hz). The straight→turn transition at gate-2 (t≈1.92s) is where the higher
feedforward (0.50 vs 0.40) causes the most additional overshoot. The ILC was calibrated
to correct a different error pattern here.

## Research Papers (2 analyzed in parallel)

### Newton-Raphson Flow for Aggressive Quadrotor Tracking (2408.11197)
- Proposes closed-form predictor for lookahead in nonlinear tracking
- Addresses integrator wind-up causing transient overshoots
- Key insight: lookahead duration should match system response time

### Tal & Karaman 2018 (1809.04048) — already partially analyzed
- Jerk/snap feedforward reduces overshoot at turns
- Our 50ms lookahead approximates jerk feedforward via time-shifting
- The lookahead duration trades off anticipation vs. overshoot

## Consensus
1. **Fix the ILC mismatch first** — this is a code bug, not a tuning problem
2. The ILC inner sim MUST match the benchmark controller to produce valid corrections
3. After fixing, gate-2 may improve because ILC will compute corrections for the
   actual controller dynamics
4. Lookahead reduction (0.05→0.03-0.04s) is a secondary lever if ILC fix is insufficient

## Recommended Action
Update `compute_ilc_offset_table()` gains to match benchmark: kp_xy=7.0, kd_xy=5.5, ff_accel=0.50.
This is the simplest, most impactful change: fix a genuine code inconsistency.
