# Iteration 28 — Research Synthesis: Per-Section Q-Filter Bandwidth for ILC

## Papers Analyzed (New in This Iteration)
1. **Bristow & Alleyne 2007** — "A Time-Varying Q-Filter Design for ILC" (ACC 2007, TAC 2008, ASME 2008)
2. **Zhang, Meng & Tan 2026** — "Segment-Based Two-Loop Adaptive ILC" (arXiv:2602.14660)
3. **Ewering et al. 2025** — "Dual ILC for MIMO Dynamics" (arXiv:2509.18723)

## Existing Relevant Papers (from previous iterations)
4. **Zhang, Meng & Cai 2024** — Segment-wise ILC with virtual memory slots
5. **Freeman et al. 2025** — Robust ILC with Butterworth Q-filter
6. **van Haren et al. 2024** — Frequency-domain ILC design
7. **Longman 2019** — Filtfilt, circulant, and cliff filters for ILC

## Research Consensus

### Strong Agreement (all papers)
- **Time-varying (section-varying) Q-filter bandwidth strictly dominates any single global bandwidth** when the trajectory has heterogeneous error frequency content. This is proven theoretically (Bristow 2007/2008) and validated experimentally.
- **Per-section convergence is guaranteed** when each section's individual iteration operator contracts (spectral radius < 1). No complex joint-spectral-radius analysis needed for deterministic traversal.
- **Butterworth zero-phase is the correct filter family** — maximally flat passband ensures ||Q||∞ ≤ 1, monotone rolloff, and zero phase delay.

### Moderate Agreement
- **Bandwidth should be set from time-frequency analysis of error** (STFT/wavelet). Bristow 2007 provides the formal procedure; our empirical sweep is the practical equivalent.
- **3-4 sections are sufficient** for most trajectories. Finer segmentation increases tuning complexity without proportional benefit.

### Contradictions
- None. The three paper families (Bristow-Alleyne, Zhang et al., Freeman et al.) are complementary, not contradictory.

## Key Insight: Matched-Bandwidth Rule

The core finding from Bristow & Alleyne: set `f_c(m) = max{ω : STFT_m(ω) > σ_n²}` — use enough bandwidth to capture error power above the noise floor, no more. This is the principled version of what we need:

- **Gate-3 S-turn inflection**: Error has frequency content at 0.4-0.8 Hz (from centripetal sign reversal). The 0.35 Hz global cutoff removes this. Local cutoff should be ~0.5-0.8 Hz.
- **All other sections**: Error is below 0.35 Hz. Global cutoff optimal.

## Proposed Implementation

**Approach: 3-section ILC with per-section Butterworth cutoff**

1. **Section 1** (steps 0 to ~200, gates 1-2): `cutoff = 0.35 Hz`
2. **Section 2** (steps ~200 to ~440, gate-3 inflection): `cutoff = 0.50-0.75 Hz` (sweep)
3. **Section 3** (steps ~440 to end, gates 4-12): `cutoff = 0.35 Hz`

Expected: Gate-3 error 0.292→0.220-0.240m, avg error maintained or slightly improved.

## Risk Assessment
- **Main risk**: Gate-3 boundary artifacts from Butterworth filtfilt at higher cutoff. Mitigate with increased reflect-padding (pad_len=min(60, len-1) already handles this).
- **Secondary risk**: Higher cutoff passes more noise at gate-3, possibly worsening convergence. Monitor per-section error monotonicity.
- **Low risk**: Breaking other sections — they use the same 0.35 Hz cutoff as current.
