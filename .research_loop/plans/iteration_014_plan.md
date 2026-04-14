# Iteration 14 Plan — Speed Recovery via Reduced Post-Optimization Inflation

## Objective
Reduce race time from 17.70s to <15.5s while maintaining avg tracking error <0.3m.

## Research Basis
- **TOPPQuad (Mao, IROS 2024)**: Geometry-timing decoupling; post-hoc speed optimization preserves path quality
- **MFRL (Ryou, IJRR 2024)**: Binary search over uniform time scale to find minimum feasible speed
- **ETH 2026 (arXiv:2603.04305)**: Proper FOV constraints add only +8.1% time; ours adds +14.1%
- **KAIST 2025 (arXiv:2512.20475)**: Heading-based FOV control adds +0% race time

## Files to Modify

### 1. `planning/trajectory_optimizer.py`

**Change A: Reduce FOV relaxation aggressiveness** (method: `_relax_for_fov`)
- Reduce per-segment multiplier: 1.07 → 1.03
- Reduce max iterations: 3 → 2
- Reduce total cap: 25% → 8%
- Rationale: L-BFGS already includes FOV penalty; kinematic sim has no camera; ETH paper shows +8.1% is sufficient

**Change B: Reduce proximity inflation** (method: `_inflate_sharp_turns`)
- Reduce max proximity factor: 0.25 → 0.12
- Rationale: Smooth racing line (smooth=0.40) already produces helix tracking of 0.09-0.17m; less inflation needed

**Change C: Add post-optimization uniform time scaling** (new method: `_compress_times`)
- After all inflation, try uniform 5% time reduction
- Verify by checking max acceleration/velocity bounds (lightweight analytical check)
- If feasible, apply; iterate up to 15% total compression
- Inspired by MFRL binary search and TOPPQuad approach
- This is a simple version of the TOPP concept: find the fastest feasible timing for the fixed geometry

### 2. `planning/racing_line.py` — NO CHANGES
The racing line geometry stays fixed (smooth basin, smoothness_weight=0.40).

## Algorithm Changes (Pseudocode)

### _relax_for_fov (modified)
```python
def _relax_for_fov(...):
    max_total = pre_relax_total * 1.08  # cap: 8% (was 25%)
    for _iteration in range(2):  # was 3
        ...
        times[i] *= 1.03  # was 1.07
```

### _inflate_sharp_turns (modified)
```python
# Proximity inflation
proximity_factor = 1.0 + 0.12 * (1.0 - dist_between / 6.0)  # was 0.25
```

### _compress_times (new)
```python
def _compress_times(self, waypoints, segment_times, start_velocity):
    """Post-optimization: try uniform time compression (MFRL-inspired)."""
    times = list(segment_times)
    best_times = list(times)
    # Try 5%, 8%, 10%, 12% compression
    for pct in [0.05, 0.08, 0.10, 0.12]:
        candidate = [t * (1.0 - pct) for t in times]
        if self._check_feasibility(waypoints, candidate):
            best_times = candidate
    return best_times

def _check_feasibility(self, waypoints, times):
    """Check if trajectory with given times is dynamically feasible."""
    for i in range(len(times)):
        dist = float(np.linalg.norm(waypoints[i+1] - waypoints[i]))
        avg_v = dist / max(times[i], 0.01)
        if avg_v > self.constraints.max_velocity:
            return False
        if i < len(times) - 1:
            dist2 = float(np.linalg.norm(waypoints[i+2] - waypoints[i+1]))
            v2 = dist2 / max(times[i+1], 0.01)
            accel = abs(v2 - avg_v) / max(times[i], 0.01)
            if accel > self.constraints.max_acceleration * 0.95:
                return False
    return True
```

## Risk Assessment
- Reducing FOV relaxation: LOW risk (kinematic sim has no camera, L-BFGS FOV penalty remains)
- Reducing proximity inflation: MEDIUM risk (helix tracking may increase from 0.10m to 0.15m)
- Uniform time compression: LOW risk (analytical feasibility check prevents violation)

## Rollback Criteria
- Revert ALL changes if:
  - Avg tracking error > 0.35m
  - Gate pass rate < 100%
  - Any crash
  - Race time doesn't improve by at least 0.5s

## Test Plan
1. Make changes
2. Run unit tests: `python3 scripts/benchmark.py --mode unit`
3. Run full benchmark: `python3 scripts/benchmark.py --mode full`
4. Compare all metrics against baseline
5. If regression, try backing off changes one at a time
