# Iteration 10 Plan — Reduce FOV Relaxation Overhead

## Objective
Reduce race time from 15.56s to ~12.5s by cutting the FOV relaxation overhead from 3.53s to <0.5s, while maintaining 100% gate pass rate and avg error < 0.5m.

## Research Basis
- ETH 2026 (arXiv:2603.04305): FOV soft constraints add only +8.1% time (1.08s on 13.3s)
- KAIST 2025 (arXiv:2512.20475): Heading-based FOV control adds +0% race time
- Consensus: Post-hoc inflation is 3-4× more expensive than necessary

## Files to Modify
1. `planning/trajectory_optimizer.py` — `_relax_for_fov()` method (lines 406-442)

## Algorithm Changes

### Current `_relax_for_fov()`:
```python
for _iteration in range(5):          # 5 iterations
    if penalty < 0.5:                 # very low break threshold
        break
    for i in range(len(times) - 1):
        if turn > 0.5:               # >30 degrees — catches most segments
            times[i] *= 1.1          # 10% increase per segment per iteration
```

### New `_relax_for_fov()`:
```python
for _iteration in range(2):          # 2 iterations max (down from 5)
    if penalty < 100.0:              # realistic threshold (ETH shows penalty in hundreds is normal)
        break

    # Compute per-segment FOV penalty to identify worst offenders
    # Only inflate segments with actual FOV violations, not all turning segments
    per_seg_penalty = compute_per_segment_fov(points, gates)

    for i in range(len(times) - 1):
        if per_seg_penalty[i] > threshold:  # only high-penalty segments
            times[i] *= 1.03         # 3% increase (down from 10%)

    # Cap total inflation at 10% of pre-relaxation time
    total_before = sum(original_times)
    total_after = sum(times)
    if total_after > total_before * 1.10:
        scale = (total_before * 1.10) / total_after
        times = [t * scale for t in times]
```

## Risk Assessment
- **Gate pass rate regression**: Possible if the trajectory becomes too fast for the PD controller. Mitigated by the 10% inflation cap and the fact that L-BFGS already includes FOV penalty.
- **Tracking error increase**: Possible on high-curvature segments. Mitigated by keeping the sharp-turn inflation from _inflate_sharp_turns() unchanged.
- **The L-BFGS FOV penalty (weight=10) remains active** in the optimizer, providing a baseline level of FOV awareness.

## Rollback Criteria
- If gate pass rate drops below 100%: revert
- If avg error increases >50%: revert
- If race time doesn't improve by at least 1s: revert (change was ineffective)

## Test Plan
1. Run unit tests after code change
2. Run full benchmark
3. Verify 100% gate pass rate
4. Compare race time, avg error, max error, per-gate errors
