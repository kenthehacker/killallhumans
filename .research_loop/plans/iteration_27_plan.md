# Iteration 27 Plan — Butterworth Q-Filter for ILC

## Objective
Replace Gaussian smoothing in the ILC with a zero-phase Butterworth low-pass Q-filter. This provides:
1. Higher effective bandwidth (1.37 Hz → 2.5 Hz) → corrections preserve more spatial detail
2. Sharper rolloff → better high-frequency noise rejection above cutoff
3. Principled frequency-domain convergence guarantees (vs empirical sigma tuning)

**Target**: Avg tracking error 0.185m → 0.175m (-5%), with no regression in max error or gate pass rate.

## Research Basis
- **van Haren et al. 2024 (ECC)**: Q-filter design in frequency domain; Butterworth for ILC robustification
- **Freeman et al. 2025 (Int. J. Control)**: 4th-order Butterworth, cutoff below 4 Hz, filtfilt implementation with reflect-padding
- **Longman 2019**: Butterworth preferred over other filter types for ILC (no passband overshoot)
- **Mashhadireza 2025 (arXiv:2511.11850)**: 4th-order Butterworth in practical ILC system

## Key Discovery
Gaussian σ=10 has -3dB at **1.37 Hz** (verified numerically). This is very conservative — the PD controller has bandwidth of ~3-5 Hz. A Butterworth at 2.5 Hz doubles the useful correction bandwidth while maintaining a sharper high-frequency rolloff.

## Files to Modify

### 1. `planning/trajectory_optimizer.py`
- Add `filter_cutoff_hz` parameter to `compute_ilc_offset_table()` (default: None for backward compat)
- When `filter_cutoff_hz` is set, use `scipy.signal.butter` + `scipy.signal.filtfilt` instead of `gaussian_filter1d`
- Add reflect-padding of 60 samples before/after each section before filtering
- Add per-section convergence tracking (independent convergence per section)
- Keep `smoothing_sigma` as fallback when `filter_cutoff_hz` is None

### 2. `scripts/benchmark.py`
- Pass `filter_cutoff_hz=2.5` to `compute_ilc_offset_table()`
- Update comments to reference new research

## Algorithm Changes

### Butterworth Q-Filter (replaces Gaussian smoothing)
```python
from scipy.signal import butter, filtfilt

# Design 4th-order Butterworth low-pass
nyquist = 0.5 / dt  # 50 Hz at dt=0.01
Wn = filter_cutoff_hz / nyquist
b, a = butter(4, Wn, btype='low')

# Apply per-section with reflect-padding
for axis in range(3):
    signal = sec_ct[:, axis]
    # Reflect-pad to handle boundary effects
    pad_len = min(60, len(signal) - 1)
    padded = np.pad(signal, pad_len, mode='reflect')
    filtered = filtfilt(b, a, padded)
    sec_smoothed[:, axis] = filtered[pad_len:-pad_len]
```

### Per-Section Convergence
```python
# Track per-section average error independently
section_converged = [False] * len(section_boundaries)
section_prev_err = [None] * len(section_boundaries)

# Only update sections that haven't converged
if not section_converged[sec_idx]:
    # ... apply correction ...
    if sec_improvement < convergence_threshold:
        section_converged[sec_idx] = True

# Stop when ALL sections converge (not global average)
if all(section_converged):
    break
```

## Risk Assessment
- **Low risk**: Filter swap is localized to the smoothing step. All other ILC logic unchanged.
- **Boundary effects**: Reflect-padding of 60 samples handles filtfilt startup artifacts.
- **Regression risk**: If Butterworth cutoff is too high, corrections will contain noise → increased error. Mitigated by testing multiple cutoffs.
- **Performance risk**: filtfilt is slightly more expensive than gaussian_filter1d, but negligible for 1400-step arrays.

## Rollback Criteria
- If avg tracking error increases by >3% vs baseline
- If max tracking error increases by >10%
- If any gate regresses by >20%
- If gate pass rate drops below 100%

## Test Plan
1. Implement Butterworth filter option in trajectory_optimizer.py
2. Run unit tests to verify no breakage
3. Test cutoff frequency sweep: {1.5, 2.0, 2.5, 3.0, 4.0} Hz
4. Select best cutoff based on avg error (primary) and max error (secondary)
5. Run full benchmark with selected parameters
6. Compare against baseline metrics
