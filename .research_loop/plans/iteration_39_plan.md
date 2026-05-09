# Iteration 39 Plan — Fix ILC-Controller Gain Mismatch

## Objective
Fix gate-2 regression by synchronizing ILC inner sim controller gains with the benchmark
runner's gains (updated in iteration 38). Target: reduce gate-2 error from 0.218m and
improve avg error from 0.151m.

## Research Basis
- Code inspection: `compute_ilc_offset_table()` uses stale gains (kp=6, kd=4, ff=0.4)
  while the benchmark uses updated gains (kp=7, kd=5.5, ff=0.50) from iter 38.
- TACO (Sanghvi 2025): Controller-trajectory coupling means ILC corrections MUST be
  computed with the same controller that will execute them.
- Tal & Karaman 2018: Feedforward timing affects tracking at transitions.

## Files to Modify
1. **`planning/trajectory_optimizer.py`** (lines 218-221):
   - Change `kp_xy, kd_xy = 6.0, 4.0` → `kp_xy, kd_xy = 7.0, 5.5`
   - Change `ff_accel = 0.4` → `ff_accel = 0.50`

That's it. One file, 2 lines changed.

## Algorithm Changes
None — the ILC algorithm is unchanged. We're only fixing the controller parameters
inside its inner kinematic simulation to match the benchmark's actual controller.

## Risk Assessment
- **Low risk**: This fixes a genuine code inconsistency, not a tuning experiment
- **Possible regression**: If the old ILC corrections happened to help by accident,
  fixing the mismatch could shift error patterns. We verify with full benchmark.
- **Gate-2**: Should improve because ILC now models the actual controller dynamics
- **Other gates**: May shift slightly as ILC rebalances corrections

## Rollback Criteria
- If avg_tracking_error increases by >5% from 0.151m baseline, revert
- If any gate regresses by >30% from current levels, revert
- If race time changes significantly (>0.5s), revert

## Test Plan
1. Run unit tests (fast) — verify no breakage
2. Run full benchmark — compare against baseline:
   - avg_error: 0.1507m
   - gate-2: 0.2183m
   - race_time: 14.01s
