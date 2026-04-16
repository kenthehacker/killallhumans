# Iteration 48 Plan — ILC 8 Iterations + Per-Section Alpha Rebalancing

## Objective
Increase ILC convergence depth from 7→8 iterations to further reduce systematic tracking error, particularly at gate-5 (0.264m, worst gate) and gate-4 (0.241m). Target: 3-5% avg error reduction while maintaining race time ≤13.35s and no gate regression >20%.

## Research Basis
- **Longman 2023** (arXiv:2307.15912): Model-based warmstarting accelerates ILC convergence; more iterations approach steady-state
- **QPGP-PILC 2026** (arXiv:2602.18014): Deeper iteration convergence consistently improves tracking
- **Liu, Zheng & Chen 2023**: Section-specific gains ensure monotone convergence per-section
- **Iteration 47 lesson**: Going 5→7 iterations required pre-inflection alpha 0.40→0.30 to prevent gate-2 over-correction. Same principle applies for 7→8.

## Files to Modify
- `scripts/benchmark.py` (lines 316-328): ILC config parameters only

## Algorithm Changes

### Step 1: Test max_iterations=8 with current alphas (Config C — minimal change)
Just change `max_iterations=7` → `max_iterations=8`. If convergence_threshold triggers early, the 8th iteration won't run, confirming we've plateaued. If it does run, check metrics.

### Step 2: If Step 1 shows improvement, test alpha rebalancing variants
- **Config A** (conservative): pre=0.28, infl=0.47, post=0.37, helix=0.42
- **Config B** (gate-5 targeted): pre=0.28, infl=0.47, post=0.42, helix=0.42

### Step 3: If Step 1 shows regression, reduce convergence_threshold to 0.001 and re-test
This ensures the ILC runs to full 8 iterations.

## Risk Assessment
- **Gate-2 regression**: Pre-inflection alpha may need further reduction. Monitor.
- **Gate-9 regression**: Helix alpha increase in iter 47 already showed +16.3% gate-9 regression. More iterations could amplify this.
- **Race time increase**: More ILC corrections = larger offsets = potential controller saturation. Monitor max_tracking_error.

## Rollback Criteria
- Revert if avg error increases >2%
- Revert if any per-gate error regresses >20%
- Revert if race time increases >0.5s
- Revert if crash or gate miss

## Test Plan
1. Run full benchmark with max_iterations=8 (Config C)
2. Compare per-gate breakdown vs baseline
3. If regression, try Config A or B
4. If all configs regress, revert and document as failed approach
