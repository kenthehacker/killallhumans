# Iteration 43 Plan: ILC Acceleration Feedforward Correction

## Objective
Add acceleration correction from ILC position offsets (d²Δp/dt²) to complete the full reference consistency chain (position + velocity + acceleration). Target: 2-5% avg error reduction.

## Research Basis
- **Schoellig et al. 2012**: Full feedforward correction includes all derivatives of the reference offset
- **Tal & Karaman 2018**: Higher-order derivative feedforward reduces tracking error
- **Bristow & Alleyne 2007**: Per-section parameter design strictly dominates uniform design
- **Robust Agile Flight 2025**: Jerk feedforward important; Butterworth filtering of derivative signals at 50 Hz cutoff

## Files to Modify

### 1. `planning/trajectory_optimizer.py` — `compute_ilc_offset_table()`
**Changes**:
- After computing `cumulative_vel_offset = np.gradient(cumulative_offset, dt, axis=0)`, compute `cumulative_accel_offset = np.gradient(cumulative_vel_offset, dt, axis=0)` (BEFORE per-section vel scaling is applied)
- Apply Butterworth low-pass filter to `cumulative_accel_offset` to suppress second-derivative noise. Use ~1.0 Hz cutoff (lower than position offset's 0.35 Hz effective bandwidth, since differentiation amplifies high-frequency content)
- Build per-section acceleration scaling array (`accel_scale`) from 7th element of section_boundaries
- Apply `accel_scale` to the acceleration offset
- In the ILC inner sim loop: add `accel_scale[step] * cumulative_accel_offset[step]` to the feedforward acceleration
- Return tuple of (pos_offsets, vel_offsets, accel_offsets)

### 2. `scripts/benchmark.py`
**Changes**:
- Update section_boundaries to include 7th element (acceleration scaling):
  - Pre-inflection: 0.0 (no accel correction, protect gate-2)
  - Inflection: 0.2 (conservative, protect S-turn)
  - Post-inflection: 0.3 (moderate)
  - Helix: 0.5 (aggressive)
- Unpack ilc_result as 3-tuple: (ilc_offsets, ilc_vel_offsets, ilc_accel_offsets)
- Apply `ilc_accel_offsets` to the feedforward acceleration in the sim loop

## Algorithm Changes (Pseudocode)

### In `compute_ilc_offset_table()` — after ILC convergence:
```python
# Compute raw acceleration offset (second derivative of position offset)
raw_vel_offset = np.gradient(cumulative_offset, dt, axis=0)  # before vel scaling
raw_accel_offset = np.gradient(raw_vel_offset, dt, axis=0)

# Apply Butterworth low-pass filter to suppress differentiation noise
# Use 1.0 Hz cutoff — conservative to avoid noise amplification
from scipy.signal import butter, filtfilt
nyq = 0.5 / dt  # 50 Hz
Wn_accel = 1.0 / nyq  # normalized cutoff
b_acc, a_acc = butter(4, Wn_accel, btype='low')
filtered_accel_offset = np.zeros_like(raw_accel_offset)
pad = min(60, len(raw_accel_offset) - 1)
for axis in range(3):
    padded = np.pad(raw_accel_offset[:, axis], pad, mode='reflect')
    filtered_accel_offset[:, axis] = filtfilt(b_acc, a_acc, padded)[pad:-pad]

# Apply per-section acceleration scaling
accel_scale = np.full(n_steps, 0.3)  # default
for sec_def in section_boundaries:
    s, e = min(sec_def[0], n_steps), min(sec_def[1], n_steps)
    sec_accel_scale = sec_def[6] if len(sec_def) > 6 else 0.3
    accel_scale[s:e] = sec_accel_scale
filtered_accel_offset *= accel_scale[:, np.newaxis]

# Apply velocity scaling (unchanged from iter 42)
cumulative_vel_offset = raw_vel_offset.copy()
cumulative_vel_offset *= vel_scale[:, np.newaxis]

return (cumulative_offset, cumulative_vel_offset, filtered_accel_offset)
```

### In ILC inner sim loop (new line after `accel_des += ff_accel * ff_acc_vec`):
```python
# Add acceleration correction from ILC offsets
if ilc_iter > 0:
    accel_des += ff_accel * accel_scale[step] * cumulative_accel_offset[step]
```

Wait — this should use the ILC's ff_accel (0.4), and the accel_scale is already baked into the offset. So:
```python
if ilc_iter > 0:
    accel_des += ff_accel * cumulative_accel_offset[step]  # scaling already baked in
```

### In benchmark sim loop:
```python
if ilc_accel_offsets is not None and step < len(ilc_accel_offsets):
    # Add to feedforward acceleration (ref_point.acceleration)
    ff_acc = tuple(a + da for a, da in zip(ff_acc, ilc_accel_offsets[step]))
```

## Risk Assessment
- **Noise amplification**: Second derivative amplifies noise quadratically. MITIGATED by Butterworth filtering at 1.0 Hz and conservative per-section scaling.
- **Gate-2 regression**: Pre-inflection scaling at 0.0 prevents any gate-2 impact (same strategy as velocity).
- **ILC inner sim divergence**: The acceleration correction changes the ILC dynamics. MITIGATED by applying only after position/velocity convergence (the acceleration correction is based on converged offsets).

## Rollback Criteria
- If avg error regresses > 1%, revert all changes
- If any gate regresses > 20%, revert
- If race time changes > 0.1s, revert (acceleration should not affect timing)

## Test Plan
1. Run unit tests first (`--mode unit`)
2. Run full benchmark and compare against baseline
3. If initial scaling doesn't help, sweep acceleration filter cutoff (0.5-2.0 Hz) and per-section scaling
