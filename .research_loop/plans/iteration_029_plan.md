# Iteration 29 Plan — Speed Recovery via Inflation Reduction

## Objective
Reduce race time from 14.03s to <13.5s by reducing post-optimization inflation factors that are now overly conservative due to ILC compensation. Maintain avg tracking error <0.25m.

## Research basis
- Spatial ILC Virtual Tube (Lv 2023): progressive speedup after ILC convergence
- ILMPC (arXiv:2508.01103): iterative lap time improvement via adaptive cost
- TACO (Sanghvi 2025): trajectory adapts to improved controller capability
- FBGA (Piazza 2025): compression floors are binding constraint for race time
- TOPPQuad (Mao 2024): post-hoc inflation breaks geometry-timing decoupling
- COP (Tzoumanikas 2022): Pareto-aware accuracy-speed tradeoff

## Files to modify

### 1. `planning/trajectory_optimizer.py` — `_inflate_sharp_turns()`
Lines ~732-765: Reduce S-turn inflation factors.

Changes:
```python
# Line 738: s_turn_inflate = 1.12 → 1.08 (junction)
# Line 740: s_turn_inflate = 1.10 → 1.06 (standard second-gate)
# Line 747: times[approach_seg] *= 1.03 → 1.01 (approach decel)
# Line 760: times[depart_seg] *= 1.04 → 1.02 (first departure)
# Line 765: times[depart_seg] *= 1.02 → 1.01 (junction departure)
```

### 2. `planning/trajectory_optimizer.py` — `_topp_retime()`
Lines ~834-835: Lower compression floors.

Changes:
```python
# Line 834: max_compression_protected = 0.68 → 0.64
# Line 835: max_compression_easy = 0.63 → 0.58
```

## Algorithm changes
No new algorithms. Pure parameter reduction in two existing functions:
1. `_inflate_sharp_turns`: Reduce 5 inflation factors by 1-4% each
2. `_topp_retime`: Lower 2 compression floors by 4-5% each

The ILC (computed in benchmark.py) runs AFTER trajectory generation, so it will automatically re-converge on the new faster trajectory.

## Risk assessment
- **Gate-2 regression**: Most vulnerable to compression floor changes. Monitor carefully. Abort if gate-2 error > 0.25m.
- **Gate-3/4 S-turn regression**: Reduced inflation may increase S-turn errors. ILC should compensate, but if avg error > 0.25m, revert S-turn changes.
- **Race time insufficient**: If race time doesn't improve by >0.3s, the changes aren't worth the accuracy cost.

## Rollback criteria
- Revert ALL changes if: avg error > 0.25m OR gate pass rate < 100%
- Revert TOPP changes only if: gate-2 error regresses >20% but S-turn changes are fine
- Revert S-turn changes only if: S-turn region (gates 2-5) avg error regresses >15%

## Test plan
1. Apply S-turn inflation changes only → benchmark
2. If S-turn changes pass, apply TOPP floor changes → benchmark
3. If both pass, commit. If only S-turn passes, commit S-turn only.
