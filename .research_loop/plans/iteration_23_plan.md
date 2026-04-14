# Iteration 23 — Implementation Plan: Speed-Aware Composite Score

## Objective
Recover race time from 14.15s to <14.0s (target: ~0.15s improvement) by adding a normalized race_time term to the sim-based racing line selection composite score, without regressing avg_tracking_error beyond 0.21m.

## Research Basis
- **COP** (Bohm et al., ICRA 2022): Normalize objectives by (nadir - utopia) range before combining → prevents scale mismatch between meters and seconds
- **CiMPCC** (Li et al., 2025): Curvature-modulated speed targets → speed optimization must be geometry-aware, not blind
- **ILMPC** (Zhao et al., 2025): Adaptive cost weighting → pure time optimization causes gate misses; tracking weight must remain dominant near gates
- **BO Racing Line** (Jain & Morari, 2020): Sim oracle for multi-objective trajectory selection

## Files to Modify
1. **`planning/racing_line.py`** — `_select_by_sim()` method only (lines 186-259)

## Algorithm Changes

### Current (iteration 22):
```python
score = 0.7 * avg_err + 0.3 * worst_gate_err
scores.sort(key=lambda x: (x[0], x[1]))  # x[1] = race_time as tiebreaker
```

### Proposed (iteration 23):
```python
# Two-pass scoring with normalization (COP: normalize before combining)

# Pass 1: Collect raw metrics from all candidates
raw_metrics = []  # list of (avg_err, worst_gate_err, race_time, idx)
for idx, result in enumerate(all_results):
    # ... existing evaluation code ...
    raw_metrics.append((avg_err, worst_gate_err, race_time, idx))

# Pass 2: Normalize and score
# Compute min/max for each metric across valid candidates
valid = [(a, w, t, i) for a, w, t, i in raw_metrics if a < 999.0]
if not valid:
    return fallback_selection()

avg_errs  = [m[0] for m in valid]
worst_errs = [m[1] for m in valid]
times     = [m[2] for m in valid]

# Min-max normalization (COP: range normalization)
def normalize(vals):
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng < 1e-9:
        return [0.0] * len(vals)
    return [(v - lo) / rng for v in vals]

norm_avg   = normalize(avg_errs)
norm_worst = normalize(worst_errs)
norm_time  = normalize(times)

# Composite: tracking-dominant with meaningful time incentive
# Weights: [0.5, 0.2, 0.3] — research-backed (ILMPC ablation: pure time → gate miss)
W_AVG, W_WORST, W_TIME = 0.5, 0.2, 0.3
scores = []
for j, (a, w, t, i) in enumerate(valid):
    score = W_AVG * norm_avg[j] + W_WORST * norm_worst[j] + W_TIME * norm_time[j]
    scores.append((score, i))

scores.sort()
return scores[0][1]
```

## Risk Assessment
- **Tracking regression**: If the time weight is too high, a fast-but-inaccurate racing line could be selected. Mitigation: 0.3 time weight is conservative; normalization means time can only contribute 0.3 to the total score.
- **No improvement**: If all candidates have similar race_time, normalization makes the time term noise. In this case, selection degenerates to the current behavior → safe.
- **Gate-specific regression**: A faster racing line might improve overall time but worsen one gate. Mitigation: worst_gate_err term (0.2 weight) penalizes this.

## Rollback Criteria
- If avg_tracking_error > 0.215m (5% regression from 0.206m): revert
- If any gate regresses >25% compared to baseline: revert
- If race time doesn't improve at all: consider weight adjustment before reverting

## Test Plan
1. Run unit tests first (`--mode unit`) to verify no syntax/import errors
2. Run full benchmark to measure new metrics
3. Compare per-gate error breakdown vs baseline
4. Verify race_time < 14.1s (target: <14.0s)
