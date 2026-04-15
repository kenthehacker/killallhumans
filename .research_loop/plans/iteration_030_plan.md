# Iteration 30 Plan — Inflation Reduction Round 2 + End Speed

## Objective
Reduce race time from 13.80s to ~13.55s by applying round 2 of post-optimization inflation reduction plus end speed increase. Maintain avg tracking error <0.25m.

## Research basis
- Spatial ILC Virtual Tube (Lv 2023): progressive speedup after ILC convergence
- ILMPC (arXiv:2508.01103): iterative lap time improvement
- SPIRAL (2025): incremental speed increase is safer than aggressive one-shot
- Track-centric ILC (arXiv:2601.21027): iterative trajectory optimization
- FBGA (Piazza 2025): compression floors are binding constraint
- Iter 29 empirical: 1-3% per parameter is safe, >3% causes basin switching

## Files to modify

### 1. `planning/trajectory_optimizer.py` — `_inflate_sharp_turns()`
Lines ~738-765: Reduce S-turn inflation factors by 1-2% each.

Changes:
```python
# Line 738: s_turn_inflate = 1.10 → 1.08 (junction, -2%)
# Line 740: s_turn_inflate = 1.08 → 1.06 (standard, -2%)
# Line 747: times[approach_seg] *= 1.02 → 1.01 (approach decel, -1%)
# Line 760: times[depart_seg] *= 1.03 → 1.02 (first departure, -1%)
# Line 765: times[depart_seg] *= 1.01 → 1.005 (junction departure, -0.5%)
```

### 2. `planning/trajectory_optimizer.py` — `_topp_retime()`
Lines ~834-835: Lower compression floors by 2%.

Changes:
```python
# Line 834: max_compression_protected = 0.66 → 0.64 (-2%)
# Line 835: max_compression_easy = 0.60 → 0.58 (-2%)
```

### 3. `planning/trajectory_optimizer.py` — `_topp_retime()` backward pass
Line ~953: Increase end speed factor.

Changes:
```python
# Line 953: max_v * 0.65 → max_v * 0.70 (+5%)
```

## Algorithm changes
No new algorithms. Pure parameter reduction + end speed increase:
1. `_inflate_sharp_turns`: Reduce 5 inflation factors by 0.5-2% each
2. `_topp_retime`: Lower 2 compression floors by 2% each
3. `_topp_retime` backward pass: Increase terminal speed from 65% to 70%

## Risk assessment
- **Gate-8 regression**: Most sensitive gate from iter 29 (+19%). Another round could push it to ~0.245m, near threshold. Monitor.
- **Gate-5 regression**: Second most sensitive (+16%). Expected to reach ~0.185m, safe.
- **Racing line basin switching**: All individual changes ≤2%. Combined effective change on any segment < 4% (cumulative S-turn + approach = ~3%). Should be safe per iter 29 lesson.
- **End speed**: Orthogonal to inflation. Only affects last segments. Cannot cause basin switching.

## Rollback criteria
- Revert ALL changes if: avg error > 0.25m OR gate pass rate < 100%
- Revert S-turn changes only if: gate-8 error > 0.25m but TOPP+end speed changes are fine
- Revert end speed if: gate-12 error regresses >20%
- If race time doesn't improve by >0.15s, revert all (not worth accuracy cost)

## Test plan
1. Apply ALL changes at once (each is ≤2%, total risk is low)
2. Run full benchmark
3. If gate-8 > 0.25m: revert S-turn changes, keep TOPP + end speed, re-benchmark
4. If all pass, commit
