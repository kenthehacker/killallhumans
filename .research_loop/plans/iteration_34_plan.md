# Iteration 34 Plan — Racing Line Offset Optimization for Helix Gates

## Objective
Reduce tracking error at gate-7 (0.284m) and gate-8 (0.235m) by optimizing their
pass-through offsets in the racing line cache. Target: gate-7 < 0.260m, gate-8 < 0.215m.

## Research Basis
- CdBO (Cully 2018): Coordinate descent optimization for racing parameters
- Gradient-free Multi-domain (Zheng 2022): CMA-ES/sim-in-the-loop evaluation
- VPMPCC (Li 2024): Gate offset affects compound helix tracking quality
- TOGT (Qin 2024): Gates are regions — optimal pass-through point matters

## Files to Modify
1. `planning/racing_line_cache.json` — Update 3 offset values
2. `scripts/helix_offset_search.py` — New search script (supporting tool)

## Algorithm Changes
Update offsets in racing_line_cache.json:
- offsets[6] (gate-7 lat): 0.555 → 0.600
- offsets[7] (gate-8 lat): -0.600 → -0.200
- offsets[19] (gate-8 vert): -0.067 → 0.333

These were found by coordinate descent search with fast kinematic sim evaluation.

## Risk Assessment
- Gate-2 showed +0.054m regression in no-ILC evaluation — ILC may compensate
- Gate-12 showed +0.027m regression in no-ILC evaluation — monitor
- If full benchmark shows overall regression, revert to original cache

## Rollback Criteria
- If avg tracking error increases by > 3%: revert
- If any per-gate error increases by > 20%: revert
- If race time increases by > 0.5s: revert

## Test Plan
1. Update cache with optimized offsets
2. Run full benchmark (includes ILC recomputation)
3. Compare all metrics against baseline
4. Verify per-gate errors — especially gate-2 and gate-12 for regression
