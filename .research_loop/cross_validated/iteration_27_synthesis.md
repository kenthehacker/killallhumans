# Iteration 27 Research Synthesis — Butterworth Q-Filter for ILC

## Research Base
Papers analyzed this iteration (5 new):
1. **Longman 2019** — "On the Choice of Filtfilt, Circulant, and Cliff Filters for ILC Robustification"
2. **Mashhadireza & Sadighi 2025** — "Neural Network-Augmented ILC for Friction Compensation" (arXiv:2511.11850)
3. **Zhao et al. 2025** — "Improving Drone Racing Performance Through Iterative Learning MPC" (arXiv:2508.01103)
4. **Nigam et al. 2026** — "Quasi-Periodic GP Predictive ILC" (arXiv:2602.18014)
5. **Freeman et al. 2025** — "Robust ILC for Unstable MIMO Systems" (Int. J. Control)

Previously analyzed (relevant to synthesis):
6. **van Haren et al. 2024** — "Frequency-Domain Approach for Enhanced Performance in Finite-Time ILC" (ECC 2024)
7. **Schoellig et al. 2012** — "Optimization-based ILC for Precise Quadrocopter Trajectory Tracking"

## Consensus (5/5 papers agree)

### 1. Gaussian smoothing is suboptimal for ILC robustification
Every paper that discusses Q-filter design uses either:
- Zero-phase Butterworth (Freeman 2025, Mashhadireza 2025, van Haren 2024)
- DFT spectral cutoff / "cliff filter" (Longman 2019)
- GP-based predictive filtering (Nigam 2026)

None use Gaussian smoothing. The Gaussian kernel has:
- No sharp frequency cutoff → leaks high-frequency noise into corrections
- Amplitude distortion in passband (Gaussian rolls off too early)
- No convergence guarantees from frequency-domain ILC theory

### 2. Filter cutoff must be below controller bandwidth
- Freeman 2025: Q-filter cutoff **below 4 Hz** for robust convergence; above 4 Hz → divergence
- van Haren 2024: cutoff at 40 Hz for 10 Hz feedback bandwidth (4:1 ratio)
- Our system: PD controller bandwidth ~3-5 Hz → cutoff should be **1.5-3 Hz**
- Our current Gaussian σ=10 ≈ **5 Hz equivalent** — at the upper edge of safe convergence

### 3. Zero-phase filtering is mandatory for ILC
- Freeman 2025: "Causal filtering is explicitly prohibited because phase delay time-shifts corrections"
- All papers implement zero-phase via `filtfilt` or DFT methods
- Our Gaussian with `gaussian_filter1d` is already zero-phase (symmetric kernel) ✓

## Contradictions

### Cliff filter vs Butterworth filtfilt
- Longman 2019 claims cliff filter (DFT zeroing) is strictly superior to filtfilt for finite-length trials
- Freeman 2025 and Mashhadireza 2025 successfully use Butterworth filtfilt with proper boundary handling
- Resolution: cliff filter is theoretically optimal but requires careful DFT handling; Butterworth filtfilt is more robust and simpler for our case with reflect-padding

### Alpha selection
- van Haren 2024 advocates α=1 with proper Q-filter
- Our iteration 26 showed α>0.4 causes cumulative offset saturation
- Resolution: our saturation is due to the max_correction clip, not filter quality. With a better filter, α=0.5 may work but must be tested carefully

## Actionable Ranking (by relevance to our bottleneck)

### 1. Replace Gaussian with zero-phase Butterworth Q-filter [HIGH IMPACT]
- **What**: Replace `gaussian_filter1d(cross_track_err, sigma=10)` with `filtfilt(b, a, cross_track_err)` using 4th-order Butterworth at fc=2.5 Hz
- **Why**: Sharper cutoff → cleaner high-frequency rejection. Current σ=10 ≈ 5 Hz is borderline unsafe per Freeman 2025
- **Expected impact**: Avg error -3-5%, max error reduction from cleaner corrections
- **Risk**: Low — same ILC architecture, just a filter swap
- **Research backing**: Freeman 2025, Mashhadireza 2025, van Haren 2024

### 2. Add reflect-padding for boundary handling [MEDIUM IMPACT]
- **What**: Pad cross-track error with ~60 reflected samples before filtfilt, trim after
- **Why**: filtfilt boundary effects corrupt first/last ~50 samples at 2 Hz cutoff
- **Expected impact**: Better corrections near race start and end
- **Research backing**: Freeman 2025, Longman 2019

### 3. Per-section convergence with independent iteration counts [MEDIUM IMPACT]
- **What**: Track avg error per section; only update sections that haven't converged
- **Why**: Current global convergence can terminate S-turn early when helix converges fast
- **Expected impact**: S-turn gets more iterations → gate-4 improvement
- **Research backing**: Liu 2023 (monotonically convergent ILC)

### 4. Sweep cutoff frequencies [LOW RISK, HIGH VALUE]
- **What**: Test fc = 1.5, 2.0, 2.5, 3.0, 4.0 Hz systematically
- **Why**: Optimal cutoff depends on PD controller bandwidth and model accuracy
- **Expected impact**: Find sweet spot between noise rejection and correction fidelity
- **Research backing**: All 5 papers

## Implementation Direction

**Primary change**: Replace Gaussian smoothing with zero-phase 4th-order Butterworth low-pass filter at ~2.5 Hz cutoff, with reflect-padding. Implement per-section as before.

**Secondary change**: Add per-section convergence tracking so each section iterates independently.

**Parameters**:
- Butterworth order: 4 (Freeman 2025 recommends 4-5 for 100 Hz sampling)
- Cutoff frequency: 2.5 Hz initially (Wn = 0.05 at Nyquist=50 Hz)
- Reflect-padding: 60 samples at each boundary
- Alpha: keep 0.4 (don't change two things at once)
- Max_correction: keep current per-section values (0.15m/0.35m)

**Sweep plan**: After implementing, test fc = {1.5, 2.0, 2.5, 3.0, 4.0} Hz to find optimal.
