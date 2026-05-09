# Iteration 19 Plan — Multi-Start Racing Line Optimization

## Objective
Escape L-BFGS local minimum in the S-turn region (gates 3-4) to reduce gate-4 tracking error from 0.413m toward 0.35m. Secondary: potential race time improvement if a faster basin exists.

## Research Basis
- **T-MPC** (de Groot et al., T-RO 2024): Parallel optimization from diverse homotopy seeds, with fallback guarantee (Theorem 2: multi-start never regresses below single-start)
- **AERO-MPPI** (Chen et al., ICRA 2026): Ensemble of M=15 parallel optimizers from structurally different initializations escapes local minima that single-instance optimization cannot
- **F1-Init** (Shehadeh et al., 2026): Initialization sensitivity causes convergence to suboptimal local minima; smart initialization reduces iterations by 17% and tracking error by 34% on hardware

## Files to Modify
1. **`planning/racing_line.py`** — `RacingLineOptimizer.optimize()` method only

## Algorithm Changes

### Current behavior (single-start):
```python
x0 = np.zeros(n * 2)  # always starts from gate centers
result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100})
```

### New behavior (multi-start):
```python
N_STARTS = 10
candidates = []

# Start 0: current baseline (zero initialization — fallback guarantee)
x0_zero = np.zeros(n * 2)
candidates.append(x0_zero)

# Start 1: geometric S-turn prior (late-apex pattern)
# For each gate, compute turn direction and set offset to cut inside
x0_apex = _compute_late_apex_init(gates, start_position)
candidates.append(x0_apex)

# Starts 2-9: random initializations uniformly sampled in bounds
rng = np.random.default_rng(42)  # deterministic for reproducibility
for _ in range(N_STARTS - 2):
    x0_rand = rng.uniform(-max_off, max_off, n * 2)
    candidates.append(x0_rand)

# Run L-BFGS-B from each candidate
best_result = None
for x0 in candidates:
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
    if best_result is None or result.fun < best_result.fun:
        best_result = result
```

### Late-apex initialization heuristic:
For each gate i with index 1..n-1:
1. Compute turn direction: cross(v_in, v_out).z
2. If turning left (positive cross): offset = +max_off * 0.5 (cut inside right)
3. If turning right (negative cross): offset = -max_off * 0.5 (cut inside left)
4. First and last gate: offset = 0

## Risk Assessment
- **Regression risk: ZERO.** The zero-initialization candidate is always included. If no random start finds a better basin, the result is identical to current. (T-MPC Theorem 2)
- **Computation cost: Negligible.** 10× current L-BFGS time = ~50ms. Benchmark takes 190ms total.
- **Deterministic behavior: Preserved.** Using `rng = np.random.default_rng(42)` for reproducible random starts.

## Rollback Criteria
- If benchmark avg error increases by >5% OR race time increases by >5% → revert
- These criteria are very conservative; zero-initialization fallback makes regression nearly impossible

## Test Plan
1. Run unit tests after edit: `python3 scripts/benchmark.py --mode unit 2>/dev/null`
2. Run full benchmark: `python3 scripts/benchmark.py --mode full 2>/dev/null`
3. Compare gate-4 error specifically (target: <0.40m, currently 0.413m)
4. Check that no other gate regressed by >0.01m
