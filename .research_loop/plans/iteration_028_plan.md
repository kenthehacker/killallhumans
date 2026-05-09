# Iteration 28 Plan — Per-Section Butterworth Bandwidth for Gate-3 Recovery

## Objective
Reduce gate-3 tracking error from 0.292m to ≤0.230m by using a higher Butterworth Q-filter cutoff at the S-turn inflection, while maintaining avg error ≤0.179m. The section-varying bandwidth is backed by Bristow & Alleyne 2007/2008 (time-varying Q-filter ILC).

## Research Basis
- **Bristow & Alleyne 2007 (ACC)**: Time-varying Q-filter bandwidth based on STFT of error. Matched bandwidth strictly dominates any LTI design. Per-section convergence guaranteed independently.
- **Zhang, Meng & Cai 2024**: Segment-wise ILC with independent per-section updates.
- **Freeman et al. 2025**: Butterworth Q-filter for robust ILC. Zero-phase, 4th order, cutoff below model fidelity frequency.

## Files to Modify

### 1. `planning/trajectory_optimizer.py` — `compute_ilc_offset_table()`
**Change**: Extend `section_boundaries` tuple format to accept a 5th element: per-section `filter_cutoff_hz`.

Current format: `(start_step, end_step, alpha, max_correction_m)`
New format: `(start_step, end_step, alpha, max_correction_m, filter_cutoff_hz)`

If `filter_cutoff_hz` is present in the tuple (5th element), design a section-specific Butterworth filter for that section. Otherwise, fall back to the global `filter_cutoff_hz` parameter.

```python
# In the per-section loop:
if len(sec_def) > 4 and sec_def[4] is not None:
    sec_cutoff = sec_def[4]
    sec_Wn = sec_cutoff / nyquist
    if sec_Wn >= 1.0:
        sec_Wn = 0.99
    sec_b, sec_a = butter(4, sec_Wn, btype='low')
else:
    sec_b, sec_a = butter_b, butter_a  # global filter
```

### 2. `scripts/benchmark.py` — ILC call site
**Change**: Split S-turn section into 3 sub-sections with per-section cutoffs.

Current:
```python
section_boundaries = [
    (0, section_boundary_step, 0.4, 0.15),             # S-turn
    (section_boundary_step, n_total_steps, 0.4, 0.35),  # Helix
]
```

New (3-section with gate-3 high-bandwidth):
```python
# Gate-3 is at ~2.94s → step 294. Define inflection region: steps 200-440.
inflection_start = int(2.0 / dt)   # step 200
inflection_end = int(4.4 / dt)     # step 440
section_boundaries = [
    (0, inflection_start, 0.4, 0.15, 0.35),             # Pre-inflection: 0.35 Hz
    (inflection_start, inflection_end, 0.4, 0.15, SWEEP),  # Gate-3 inflection: SWEEP Hz
    (inflection_end, n_total_steps, 0.4, 0.35, 0.35),    # Post-inflection + helix: 0.35 Hz
]
```

## Algorithm Changes

The only algorithmic change is: instead of one global Butterworth filter applied to all sections, each section designs its own Butterworth filter from its own `filter_cutoff_hz`. The learning law remains P-type ILC with cross-track projection. Everything else is unchanged.

## Sweep Plan
Test gate-3 inflection cutoff at: {0.50, 0.60, 0.75, 1.00} Hz.
Select the value that minimizes gate-3 error without regressing avg error beyond 0.179m.

## Risk Assessment
- **Gate-3 boundary artifacts**: Butterworth filtfilt at higher cutoff has wider transition region. Current pad_len=min(60, len-1) should be sufficient for 240-step section.
- **Other sections unchanged**: They use the same 0.35 Hz as current — no regression risk.
- **Noise at gate-3**: Higher cutoff passes more noise. If gate-3 oscillates instead of improving, reduce cutoff.

## Rollback Criteria
- If avg error increases >2% (>0.182m): revert all changes
- If any gate error increases >25%: revert
- If no improvement at gate-3: revert

## Test Plan
1. Implement code changes
2. Run unit tests (should pass — no interface changes)
3. Run sweep of gate-3 cutoffs
4. Select best cutoff
5. Run full benchmark with best config
6. Compare against baseline
