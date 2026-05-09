# Iteration 32 Plan — Adaptive Entry/Exit Offset for Helix Entry

**STATUS: FAILED — ALL approaches reverted. See iteration_032_report.md.**

## Objective
Reduce gate-7 tracking error from 0.284m toward 0.255-0.265m by increasing the entry/exit offset at high-curvature gates, specifically the helix entry transition. This addresses the root cause identified in iteration 31: gate-7's error is trajectory-shape-driven, not speed-driven.

## Research Basis
- **On Your Own (Romero 2025)**: Uses 0.4m for normal gates, 1.25m for Split-S — precedent for variable offsets
- **TOGT (Qin 2024)**: Gates are regions, not points — optimal pass-through position varies
- **Richter et al. (MIT)**: Min-snap polynomial curvature ∝ turn_angle / segment_length² — larger segments reduce curvature
- **CiMPCC (Li 2024)**: Compound curvature in sequential same-direction turns
- **PMPC (2024)**: Gate traversal curves naturally adapt to angular change between gates

## Files to Modify
1. `planning/trajectory_optimizer.py` — `generate_trajectory()` method, waypoint placement section (lines ~490-511)

## Algorithm Changes

### Change 1: Compute per-gate turn angles BEFORE waypoint placement

Before the waypoint loop (after line 490), compute turn angles from gate centers:

```python
# Compute turn angle at each gate to determine adaptive offset
gate_centers = [np.array(g.position, dtype=float) for g in gates]
gate_turn_angles = [0.0] * len(gates)  # radians
for gi in range(1, len(gates) - 1):
    v_in = gate_centers[gi] - gate_centers[gi - 1]
    v_out = gate_centers[gi + 1] - gate_centers[gi]
    n1, n2 = np.linalg.norm(v_in), np.linalg.norm(v_out)
    if n1 > 0.1 and n2 > 0.1:
        cos_a = np.clip(np.dot(v_in, v_out) / (n1 * n2), -1, 1)
        gate_turn_angles[gi] = math.acos(cos_a)
```

### Change 2: Adaptive offset per gate

Replace the fixed `ENTRY_EXIT_OFFSET = 0.4` with a per-gate computation:

```python
BASE_OFFSET = 0.4       # meters, baseline per "On Your Own" paper
MAX_OFFSET = 0.8         # meters, maximum for sharpest turns
ANGLE_THRESHOLD = 0.8    # radians (~46°), above this offset increases
MAX_ANGLE = 1.5          # radians (~86°), at this angle offset is MAX_OFFSET
```

For each gate gi:
```python
angle = gate_turn_angles[gi]
if angle > ANGLE_THRESHOLD:
    # Linear interpolation from BASE to MAX based on turn angle
    t = min((angle - ANGLE_THRESHOLD) / (MAX_ANGLE - ANGLE_THRESHOLD), 1.0)
    offset = BASE_OFFSET + t * (MAX_OFFSET - BASE_OFFSET)
else:
    offset = BASE_OFFSET
```

### Change 3: Safety check — prevent waypoint overlap

After computing offset for gate gi, ensure the exit waypoint of gate[gi-1] and entry waypoint of gate[gi] maintain at least 2.0m separation:

```python
if gi > 0:
    inter_gate_dist = np.linalg.norm(gate_centers[gi] - gate_centers[gi - 1])
    prev_offset = offsets[gi - 1]
    max_safe_offset = (inter_gate_dist - prev_offset - 2.0)  # 2.0m min gap
    offset = min(offset, max(BASE_OFFSET, max_safe_offset))
```

### Expected gate-7 behavior
- Gate-7 turn angle: 68.5° = 1.196 rad
- t = min((1.196 - 0.8) / (1.5 - 0.8), 1.0) = min(0.566, 1.0) = 0.566
- offset = 0.4 + 0.566 * (0.8 - 0.4) = 0.4 + 0.226 = 0.626m
- Through-gate segment: 2 × 0.626 = 1.252m (vs current 0.8m, +56% longer)
- Expected curvature reduction: ~(0.4/0.626)² ≈ 59% of original

### Which gates are affected
- Gate-1 (25.2°): 0.44 rad → no change (below threshold)
- Gate-2 (30.7°): 0.54 rad → no change
- Gate-3 (48.2°): 0.84 rad → slight increase to ~0.42m
- Gate-4 (36.4°): 0.64 rad → no change
- Gate-5 (28.1°): 0.49 rad → no change
- Gate-6 (93.9°): 1.64 rad → MAX_OFFSET 0.8m (but capped by safety)
- **Gate-7 (68.5°): 1.20 rad → ~0.63m** (primary target)
- Gate-8 through Gate-11: various helix interior angles
- Gate-12: exit gate

## Risk Assessment
- This changes waypoint geometry for the ENTIRE trajectory, which means the L-BFGS optimizer will find a different local minimum
- However, the racing line optimizer (`_select_by_sim`) evaluates candidates using the full trajectory pipeline, so it should adapt
- The iteration 6 failure was with ADAPTIVE offsets that were optimized by L-BFGS — here we compute offsets BEFORE L-BFGS, so the optimizer sees the offsets as fixed
- S-turn gates (1-5) mostly won't be affected (turn angles below threshold)
- Gate-6 and gate-7 will have the largest changes

## Rollback Criteria
- If avg tracking error > 0.200m (8% worse than current 0.185m), revert
- If race time > 14.5s (5% worse than current 13.79s), revert
- If any gate error exceeds 0.35m, revert
- If unit tests fail, revert

## Test Plan
1. Run unit tests first (should pass — no API changes)
2. Run full benchmark
3. Compare gate-7 error, gate-3 error, avg error, race time
4. Check that S-turn gates (1-5) didn't regress significantly
