# Iteration 42 Research Synthesis: Per-Section Velocity Correction Scaling

## Current State
- Race time: 14.08s, avg error: 0.140m, max error: 0.701m, 100% gate pass
- Gate-2 worst gate at 0.242m (+13.4% regression from iteration 41's velocity correction)
- Velocity correction scaling 0.5x uniform applied in both ILC inner sim and benchmark

## Core Insight: Uniform Velocity Scaling is Suboptimal

Iteration 41 proved that velocity corrections improve tracking by 6.7% overall, but the
uniform 0.5x scaling creates a gate-2 regression because:
1. Gate-2 is in the pre-inflection section (step 0-200, time 0-2.0s)
2. The ILC gain mismatch (kp=6 in ILC vs kp=7 in benchmark) amplifies velocity
   corrections differently in the early flight phase
3. The pre-inflection section has minimal ILC offsets → velocity corrections are small
   but the RELATIVE perturbation to the velocity field is large

## Research Basis

### Bristow & Alleyne 2007 (Time-Varying Q-filter)
- **Key insight**: The Q-filter bandwidth should vary as a function of time within each trial
- **Proof**: Time-varying filter design is strictly better than any LTI (uniform) filter
- **Application**: The same principle applies to velocity correction scaling — the scaling
  factor should vary per section to match local dynamics

### Zhang, Meng & Cai 2026 (Segment-Based AILC)
- **Key insight**: Segment-independent learning prevents cross-contamination
- **Per-section parameters**: Different sections should have independent tuning parameters
- **Application**: Velocity scaling is another per-section parameter that should be tuned independently

### Iteration 41 Empirical Data
- 0.0x (no vel correction): baseline (0.150m avg, gate-2 at 0.214m)
- 0.3x uniform: 0.142m avg (-5.4%), gate-2 +7.9%
- 0.5x uniform: 0.140m avg (-6.7%), gate-2 +13.4%
- 1.0x uniform: 0.143m avg (-5.0%), gate-2 +26.8%

The pattern: helix gates improve monotonically with higher scaling, but gate-2 degrades.
Per-section scaling can break this tradeoff.

## Proposed Approach

### Per-Section Velocity Correction Scaling
Extend section_boundaries to include a velocity scaling factor per section:

| Section | Steps | Current Scale | Proposed Scale | Rationale |
|---------|-------|---------------|----------------|-----------|
| Pre-inflection | 0-200 | 0.5x | 0.0x | Minimal ILC offsets; velocity correction hurts gate-2 |
| Gate-3 inflection | 200-440 | 0.5x | 0.3x | Moderate offsets; conservative scaling for S-turn |
| Post-inflection | 440-740 | 0.5x | 0.5x | Good response to velocity correction |
| Helix | 740+ | 0.5x | 0.7x | Largest offsets; maximum benefit from velocity correction |

### Implementation Strategy
1. Add per-section velocity scaling to both `trajectory_optimizer.py` (ILC inner sim) and `benchmark.py`
2. Use a step-indexed scaling array computed from section boundaries
3. Sweep 2-3 configurations to find optimal per-section scales
4. The scaling array blends smoothly at section boundaries (same blend_steps=50 as position offsets)

### Expected Impact
- Gate-2: recover from 0.242m to ~0.214m (back to baseline)
- Helix gates (4-10): further improvement from 0.7x scaling (vs current 0.5x)
- Avg error: maintain ~0.140m or slight improvement
