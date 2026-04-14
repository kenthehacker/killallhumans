# Iteration 26 — Research Synthesis: Per-Section ILC with Adaptive Gains

## Papers Analyzed (This Iteration)
1. **van Haren et al. 2024** — "A Frequency-Domain Approach for Enhanced Performance and Task Flexibility in Finite-Time ILC" (ECC 2024)
2. **Zhang, Meng & Cai 2024** — "Segment-wise Learning Control for Trajectory Tracking" (Science China Information Sciences)
3. **Liu, Zheng & Chen 2023** — "Monotonically Convergent ILC by Time Varying Learning Gain Revisited" (Automatica)

## Previously Analyzed (Relevant)
4. **Schoellig et al. 2012** — Optimization-Based ILC for Quadrocopter (foundational)
5. **Lv et al. 2023** — Spatial ILC within Virtual Tube
6. **Zhao et al. 2025** — ILMPC for Drone Racing (IROS 2025)

---

## Consensus Across Papers

### Strong Agreement (3+ papers)
1. **Segment/section decomposition is the right approach** for heterogeneous trajectories. Zhang 2024 formalizes this with virtual memory slots per segment. Liu 2023 shows time-varying gains enable per-section tuning. Lv 2023 uses spatial ILC in a virtual tube for drone racing.

2. **Fixed global learning rate is suboptimal** when trajectory sections have different dynamic characteristics. All three new papers support adapting the learning process to local trajectory properties.

3. **Q-filter / robustness filter is essential** for stable convergence. van Haren 2024 shows Q^f controls convergence at high frequencies. Our current Gaussian smoothing (sigma=10) is an ad-hoc approximation of a proper robustness filter.

4. **Multi-pass ILC converges rapidly** — typically 3-5 iterations for P-type ILC on deterministic systems (Schoellig 2012, Liu 2023). Our current 5-iteration budget is appropriate.

### Contradictions / Tensions
1. **Gain scheduling direction**: Liu 2023 proves exponentially *increasing* gains guarantee ∞-norm monotone convergence, while intuition suggests decreasing gains for stability. For our case (helix is at the END of the trajectory), increasing gain actually aligns with putting more correction effort on the helix.

2. **Alpha = 1 vs conservative alpha**: van Haren 2024 advocates alpha=1 with proper Q-filter, while our experience (alpha=0.4) and Schoellig 2012 suggest conservative gains. The resolution: alpha can be increased if the robustness filter is properly designed.

---

## Actionable Takeaways (Ranked)

### Priority 1: Per-Section ILC with Independent Offset Tables
**Source**: Zhang 2024 + Liu 2023 + state.json backlog priority #1

Split the trajectory into 2-3 sections at gate boundaries:
- **Section A (S-turn)**: gates 1-6, timesteps 0 to ~680
- **Section B (Helix)**: gates 7-12, timesteps ~680 to end

Each section gets:
- Independent cumulative_offset array
- Independent convergence check
- Independent learning rate alpha

This directly addresses the gate-4 regression: the S-turn section's ILC won't be corrupted by helix corrections, and vice versa.

### Priority 2: Replace Gaussian Smoothing with Zero-Phase Butterworth Q-Filter
**Source**: van Haren 2024

Replace `scipy.ndimage.gaussian_filter1d(cross_track, sigma=10)` with `scipy.signal.filtfilt(b, a, cross_track)` using a Butterworth low-pass at fc ≈ 2-3 Hz (corresponding to controller bandwidth / 2).

Benefits: principled frequency-domain convergence guarantee, cleaner correction signal.

### Priority 3: Increase Alpha with Better Q-Filter
**Source**: van Haren 2024

With a proper Q-filter, alpha can be increased from 0.4 to 0.6-0.8, enabling convergence in 2-3 iterations instead of 5. Faster convergence = more iterations available for fine-tuning.

### Priority 4: Section-Specific Alpha Values
**Source**: Liu 2023 + Zhang 2024

Use different learning rates per section:
- S-turn (gates 1-6): alpha_s = 0.5 (moderate, these gates have lower error)
- Helix (gates 7-12): alpha_h = 0.6 (more aggressive, higher error to correct)

Liu 2023 shows increasing gains along the trajectory are theoretically motivated.

---

## Proposed Implementation Direction

**Per-section ILC with improved Q-filter**: Segment the trajectory at the S-turn/helix boundary (approximately at gate-6 pass-through). Apply ILC independently per section with section-specific alpha values. Replace Gaussian smoothing with zero-phase Butterworth low-pass filter.

Expected impact:
- Gate-4 regression eliminated (section independence)
- Helix gates further improved (dedicated helix learning)
- Avg error 0.195m → ~0.185m
- No race time regression (ILC is position-only)

Risk: Section boundary may introduce a discontinuity in the offset table. Mitigation: overlap the sections by 50 timesteps and blend with a linear ramp at the boundary.
