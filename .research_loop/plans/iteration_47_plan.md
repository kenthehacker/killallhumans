# Iteration 47 Plan: Multi-Start L-BFGS Time Allocation Optimization

## Objective
Replace single-start L-BFGS with multi-start L-BFGS to escape local minima in the trajectory time allocation. Target: reduce gate-4 error (0.287m → <0.265m) and/or reduce race time (13.31s → <13.2s) by finding a better L-BFGS local minimum.

## Research Basis
- cuRobo (arXiv:2310.17274): 12-seed parallel L-BFGS achieves 99.8% success vs 98.5% single-start
- MISO (arXiv:2411.02158): 5-14× improvement from multi-start across DDP/MPPI/iLQR
- AERO-MPPI (arXiv:2509.17340): 15 structured anchor seeds cover distinct basins in non-convex landscapes

## Files to Modify
1. `planning/trajectory_optimizer.py` — `_optimize_time_allocation()` method (lines 1234-1320)

## Algorithm Changes

### Current (single-start):
```python
result = minimize(objective, log_times, method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-6})
```

### New (multi-start with 4 seeds):
```python
seeds = [
    log_times,                          # Seed 0: original (guarantees no regression)
    log_times * 0.85,                   # Seed 1: aggressive (15% faster initial times)
    log_times * 1.15,                   # Seed 2: conservative (15% slower)
    graduated_warmstart(log_times),     # Seed 3: graduated optimization (tw=1.5→2.3)
]
best = min(minimize(obj, s, ...) for s in seeds, key=lambda r: r.fun)
```

### Graduated warm-start (Seed 3):
1. First optimize with time_weight=1.5 (smoother landscape, wider basin of attraction)
2. Use that solution as initial point for time_weight=2.3 optimization
3. This is a continuation/homotopy approach — the smooth problem leads to a basin that the aggressive problem can then refine

### Selection criterion:
Use the L-BFGS objective value (time_weight * total_time + penalties). This is the same objective used during optimization.

## Risk Assessment
- **No regression guaranteed**: Seed 0 is the current initialization, so the best result will always be at least as good as the current trajectory
- **Compute cost**: 4× L-BFGS runs. Current planning takes ~150ms, will become ~600ms. Acceptable for offline pre-computation.
- **Determinism**: No random elements — all seeds are deterministic functions of the waypoints

## Rollback Criteria
- If avg tracking error increases >5% OR race time increases >0.2s → revert
- If any per-gate error increases >20% → revert

## Test Plan
1. Run unit tests to verify no syntax errors
2. Run full benchmark to compare metrics
3. Compare per-gate error breakdown
