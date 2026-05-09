# Iteration 42 Plan: Per-Section Velocity Correction Scaling

## Objective
Replace uniform 0.5x velocity correction scaling with per-section scaling to recover gate-2
while maintaining or improving helix tracking. Target: gate-2 recovery to ~0.215m,
avg error ≤0.140m.

## Research Basis
- Bristow & Alleyne 2007: Time-varying filter design is strictly better than LTI
- Zhang 2026: Segment-independent parameters prevent cross-contamination
- Iteration 41 data: gate-2 regression is caused by velocity correction in pre-inflection section

## Files to Modify

### 1. `planning/trajectory_optimizer.py` — `compute_ilc_offset_table()`
- Build a per-step velocity scaling array from section_boundaries
- Each section boundary tuple gets an optional 6th element: velocity scaling factor
- If not provided, default to 0.5
- Apply per-step scaling in the ILC inner sim: `target_vel = ref.velocity + vel_scale[step] * vel_offset[step]`
- Apply per-step scaling to the returned velocity offsets (pre-bake the scaling)

### 2. `scripts/benchmark.py`
- Build the same per-step velocity scaling array from section_boundaries
- Apply per-step scaling: `target_vel += vel_scale[step] * ilc_vel_offsets[step]`
- Remove the hardcoded 0.5 scaling factor

## Algorithm Changes

### Velocity Scaling Array Construction
```python
vel_scale = np.ones(n_steps) * 0.5  # default
for sec_def in section_boundaries:
    sec_start, sec_end = sec_def[0], sec_def[1]
    sec_vel_scale = sec_def[5] if len(sec_def) > 5 else 0.5
    vel_scale[sec_start:sec_end] = sec_vel_scale
# Smooth transitions at boundaries (blend_steps)
```

### Section Configuration
```python
section_boundaries = [
    # (start, end, alpha, max_correction_m, filter_cutoff_hz, vel_scale)
    (0, inflection_start, 0.4, 0.15, 0.35, 0.0),          # Pre-inflection: NO vel correction
    (inflection_start, inflection_end, 0.4, 0.15, 0.40, 0.3),  # Inflection: conservative
    (inflection_end, helix_start, 0.4, 0.15, 0.35, 0.5),       # Post-inflection: standard
    (helix_start, n_total_steps, 0.4, 0.35, 0.35, 0.7),        # Helix: aggressive
]
```

## Risk Assessment
- **Low risk**: This is a refinement of an already-working feature (velocity correction)
- **No basin switching risk**: Racing line and trajectory optimizer are untouched
- **Gate-2 regression**: Should be fully recovered by setting pre-inflection scale to 0.0
- **Helix regression**: Unlikely — helix benefits from higher scaling (0.5→0.7)

## Rollback Criteria
- If avg error increases by >3% (0.144m), revert
- If any gate error increases by >20%, revert
- If race time changes by >0.3s, revert

## Test Plan
1. Implement per-section scaling with initial config: [0.0, 0.3, 0.5, 0.7]
2. Run benchmark, check gate-2 recovery and overall metrics
3. If gate-2 not recovered, try [0.0, 0.2, 0.5, 0.6]
4. If helix doesn't improve at 0.7, fall back to [0.0, 0.3, 0.5, 0.5]
