# Iteration 50 Plan — Competition Robustness Polish

## Objective
Add deterministic seeding for guaranteed benchmark reproducibility, and verify
the system's competition readiness through multi-trial validation. Target: zero
metric changes (maintenance iteration), with improved code quality.

## Research Basis
- Testing Pipeline (2025): recommends deterministic seeding for regression testing
- SimpleFlight (2024): uses fixed seeds for reproducible evaluation
- UZH RPG Reality Gap (2025): staged validation with reproducible SIL as foundation

## Files to Modify

### 1. `scripts/benchmark.py`
- Add `np.random.seed(42)` at the start of `run_synthetic_benchmark()` for deterministic
  noise generation. This ensures identical results across runs regardless of system state.
- Add race_time_s check to threshold_failures for completeness.

## Algorithm Changes
None. This is a code quality / robustness polish iteration.

## Risk Assessment
- **Regression risk**: MINIMAL. Adding a fixed seed can only make results more deterministic.
  The seed value 42 will produce specific noise values that differ from the unseeded
  random stream, but with σ=5mm noise the impact is negligible.
- **If the seed changes results**: The EKF is filtering 5mm noise against 162mm tracking
  error. Any change would be <0.1% and well within noise floor.

## Rollback Criteria
Revert if any metric regresses by >1% or any threshold fails.

## Test Plan
1. Run benchmark 3 times, verify identical results
2. Verify all unit tests still pass
3. Verify no threshold failures
