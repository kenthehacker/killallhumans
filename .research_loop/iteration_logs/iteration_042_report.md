# Iteration 42 — Per-Section Velocity Correction Scaling

**Date**: 2026-04-15
**Bottleneck**: system_integration (gate-2 regression recovery)
**Status**: COMMITTED — avg error 0.140→0.138m (-1.2%), gate-2 0.242→0.214m (-11.8%)
**Commit**: 2d72cd6

---

## Section 1: Summary
- Iteration 42, timestamp 2026-04-15T14:47Z
- Bottleneck: system_integration (gate-2 regression from uniform velocity correction)
- One-line outcome: **Per-section velocity scaling recovers gate-2 (-11.8%) while improving avg error (-1.2%); cumulative -7.8% vs iter 40**

---

## Section 2: Research

### Papers re-examined (no new papers — direct follow-up from iter 41)
1. **Bristow & Alleyne 2007** (ACC): Time-varying Q-filter design
   - Proves time-varying (per-section) filter design strictly dominates LTI (uniform) design
   - Directly applicable: velocity scaling is another per-section parameter
2. **Zhang, Meng & Cai 2026**: Segment-based AILC
   - Segment-independent parameters prevent cross-contamination
   - Per-section velocity scaling follows the same principle as per-section bandwidth

### Key insight from research
**Bristow & Alleyne's strict dominance result** applies to any per-section parameter, not just filter bandwidth. The velocity correction scaling is analogous to a per-section learning gain — sections with high sensitivity (gate-2) need conservative scaling, while sections with large offsets (helix) benefit from aggressive scaling.

### Research consensus vs contradictions
- **Consensus**: Per-section parameters are strictly better than uniform (Bristow 2007, Zhang 2026, iteration 41 data)
- **No contradictions found**

---

## Section 3: Implementation

### Changes made
1. **`planning/trajectory_optimizer.py`** — `compute_ilc_offset_table()`:
   - Added per-step velocity scaling array (`vel_scale`) built from section_boundaries 6th element
   - Applied per-step scaling in ILC inner sim: `target_vel = ref.velocity + vel_scale[step] * vel_offset[step]`
   - Pre-baked per-section scaling into returned velocity offsets: `cumulative_vel_offset *= vel_scale[:, np.newaxis]`
   - This means the caller (benchmark) applies offsets directly without knowing about sections

2. **`scripts/benchmark.py`**:
   - Updated section_boundaries to include 6th element (velocity scaling):
     - Pre-inflection: 0.0 (no velocity correction — recovers gate-2)
     - Inflection: 0.4 (moderate — protects S-turn)
     - Post-inflection: 0.5 (standard)
     - Helix: 0.7 (aggressive — maximizes helix benefit)
   - Simplified velocity offset application: direct addition (no hardcoded 0.5 scale)

### Plan adherence
- Followed plan exactly for the per-section approach
- Deviated from initial plan's inflection scaling (0.3→0.4) based on sweep results
- Deviated from initial plan's helix scaling (0.7 confirmed optimal via sweep)

### Configuration sweep results
| Config | Pre-infl | Inflection | Post-infl | Helix | Avg Error | Gate-2 | Gate-4 |
|--------|----------|------------|-----------|-------|-----------|--------|--------|
| Iter 41 | 0.5 | 0.5 | 0.5 | 0.5 | 0.1401 | 0.2422 | 0.1342 |
| A | 0.0 | 0.3 | 0.5 | 0.7 | 0.1388 | 0.2137 | 0.1480 |
| B | 0.1 | 0.4 | 0.5 | 0.6 | 0.1387 | 0.2193 | 0.1413 |
| C | 0.0 | 0.4 | 0.5 | 0.6 | 0.1386 | 0.2137 | 0.1422 |
| **D** | **0.0** | **0.4** | **0.5** | **0.7** | **0.1384** | **0.2137** | **0.1422** |

Config D selected: best avg error while maintaining full gate-2 recovery.

---

## Section 4: Benchmark Comparison

### Full metrics table

| Metric | Before (iter 41) | After | Delta | Direction |
|--------|-------------------|-------|-------|-----------|
| Avg tracking error | 0.1401m | 0.1384m | **-1.2%** | improved |
| Max tracking error | 0.7011m | 0.7269m | +3.7% | regressed |
| P50 tracking error | 0.1167m | 0.1134m | **-2.9%** | improved |
| P95 tracking error | 0.3422m | 0.3323m | **-2.9%** | improved |
| EKF uncertainty | 0.0119m | 0.0119m | 0% | same |
| Race time | 14.08s | 14.08s | 0% | same |
| Gate pass rate | 100% | 100% | 0% | same |
| Loop Hz | 7752 | 7674 | -1.0% | same |
| Crashed | false | false | — | same |
| Unit tests | 9/9 | 9/9 | — | same |

### Cumulative improvement (vs iteration 40, pre-velocity-correction)

| Metric | Iter 40 | Iter 42 | Cumulative Delta |
|--------|---------|---------|-----------------|
| Avg tracking error | 0.1501m | 0.1384m | **-7.8%** |
| Max tracking error | 0.7269m | 0.7269m | 0% |
| Gate-2 | 0.2137m | 0.2137m | 0% |
| Gate-9 | 0.1437m | 0.1075m | **-25.2%** |

### Per-gate error breakdown

| Gate | Before | After | Delta | Direction |
|------|--------|-------|-------|-----------|
| gate-1 | 0.1147 | 0.1129 | -1.6% | ~ |
| gate-2 | 0.2422 | 0.2137 | **-11.8%** | **recovered** |
| gate-3 | 0.1885 | 0.1907 | +1.2% | ~ |
| gate-4 | 0.1342 | 0.1422 | +6.0% | mild regr |
| gate-5 | 0.1395 | 0.1392 | -0.2% | ~ |
| gate-6 | 0.0734 | 0.0736 | +0.3% | ~ |
| gate-7 | 0.1695 | 0.1687 | -0.5% | ~ |
| gate-8 | 0.1318 | 0.1289 | -2.2% | improved |
| gate-9 | 0.1072 | 0.1075 | +0.3% | ~ |
| gate-10 | 0.1491 | 0.1442 | -3.3% | improved |
| gate-11 | 0.1153 | 0.1160 | +0.6% | ~ |
| gate-12 | 0.1408 | 0.1410 | +0.1% | ~ |

### Threshold status
ALL thresholds passing. No threshold failures.

---

## Section 5: Deep Diagnostic

### Root cause diagnosis
Gate-2's regression in iteration 41 was caused by velocity corrections in the pre-inflection section amplifying the ILC gain mismatch (kp=6 vs kp=7). By setting pre-inflection velocity scaling to 0.0, gate-2 returns to its pre-velocity-correction baseline (0.214m). The inflection section at 0.4 scaling provides a moderate velocity correction benefit without destabilizing the S-turn. The helix benefits most from higher scaling (0.7) because it has the largest ILC offsets and the velocity mismatch there was the dominant error source.

### Telemetry signals
| Signal | Before | After |
|--------|--------|-------|
| max_abs_roll_rad | 0.85 | 0.85 |
| max_abs_pitch_rad | 0.85 | 0.85 |
| avg_pitch_rad | -0.096 | -0.096 |
| avg_thrust | 0.837 | 0.837 |

Controller outputs unchanged — the per-section velocity scaling redistributes correction effort without changing control authority.

### Gate-4 regression analysis
Gate-4 regressed 6.0% (0.134→0.142). Gate-4 is at time 4.32s, just inside the inflection section boundary (4.4s = step 440). The inflection section's velocity scaling changed from 0.5 to 0.4, which changes the ILC's learned position offsets slightly. This is a minor tradeoff for the 11.8% gate-2 recovery.

### Max error analysis
Max error increased from 0.701 to 0.727m. This occurs in the pre-inflection section where velocity scaling is now 0.0 (same as iter 40). The max error is identical to iteration 40's value, confirming that the max error location is in the pre-inflection section where the velocity correction provided benefit in iter 41 but is now removed.

### Trend analysis
- **Improving**: Two consecutive improving iterations (41, 42) after 5-iteration plateau
- **Velocity correction architecture**: Opens a new optimization dimension (per-section scaling) that has more room to explore
- **Cumulative improvement**: Avg error 0.150→0.138m over 2 iterations (-7.8%)
- **Diminishing returns approaching**: Per-section scaling tuning has diminishing marginal returns (~1% per sweep)

---

## Section 6: Forward-Looking

### Improvements backlog (prioritized)

1. **Acceleration correction from ILC offsets** (priority 1)
   - Area: system_integration
   - Problem: Feedforward acceleration is still from original trajectory, not corrected path
   - Approach: Compute second time derivative of ILC position offset, add to feedforward acceleration with per-section scaling
   - Expected impact: 3-5% avg error reduction; requires careful noise management
   - Research: Schoellig 2012 (full feedforward correction), Nam 2026 (velocity+accel profiles)

2. **ILC gain re-synchronization with velocity corrections** (priority 2)
   - Area: system_integration
   - Problem: ILC inner sim uses kp=6, kd=4, ff=0.4 while benchmark uses kp=7, kd=5.5, ff=0.50
   - Approach: Re-attempt sync now that velocity corrections are per-section. The sync failed in iter 39 without velocity corrections — the per-section velocity scaling may change the dynamics enough
   - Expected impact: HIGH RISK — could improve convergence or cause regression
   - Research: Wu 2024 (mismatch compensation)

3. **Pre-inflection velocity correction with reduced scaling** (priority 3)
   - Area: system_integration
   - Problem: Pre-inflection at 0.0 leaves max error at 0.727m (same as iter 40)
   - Approach: Try 0.1-0.2 pre-inflection scaling with higher inflection scaling to see if max error can be reduced without gate-2 regression
   - Expected impact: Max error reduction 0.727→~0.710m
   - Research: Bristow 2008 (optimal time-varying bandwidth)

4. **PyBullet full-physics validation** (priority 4)
   - Area: system_integration
   - Problem: 42 iterations of kinematic sim tuning unvalidated in full physics
   - Approach: Run benchmark with PyBullet to discover real-world viability
   - Expected impact: Discover transfer gap and prioritize accordingly
   - Research: NGTC (Pries 2025), LoL-NMPC (Gupta 2025)

### Architectural recommendations
- The per-section velocity scaling framework is now mature. Future iterations should explore:
  - Acceleration corrections (second derivative of offset) with per-section scaling
  - ILC gain re-synchronization (now feasible with per-section velocity corrections)
  - Fine-tuning the section boundary positions (currently at fixed time values)
- The section_boundaries tuple is getting long (6 elements). Consider refactoring to a named config object if more parameters are added.

### Next bottleneck selected
**system_integration** — acceleration correction from ILC offsets (the next level of feedforward consistency)

### What NOT to try
- Uniform velocity scaling tuning — already swept exhaustively (0.0-1.0x in iter 41, per-section in iter 42)
- Pre-inflection velocity scaling > 0.1 — causes gate-2 regression
- Further parameter tuning of racing line, TOPP floor, or controller gains — exhausted in iterations 36-40
- ILC gain synchronization without per-section velocity corrections — failed in iter 39

---

## Section 7: Lessons Learned

### What worked
- **Per-section velocity scaling completely recovers gate-2**: Pre-inflection at 0.0 restores gate-2 to iter 40 baseline
- **Helix at 0.7 is better than 0.5**: Higher velocity scaling in the helix section provides incremental improvement
- **Systematic sweep**: Testing 4 configurations identified the optimal balance

### What didn't work
- **Config A [0.0, 0.3, 0.5, 0.7]**: Gate-4 regressed 10.3% due to inflection scaling at 0.3
- **Config B [0.1, 0.4, 0.5, 0.6]**: Pre-inflection at 0.1 still causes gate-2 partial regression

### Surprises
- **Gate-2 recovery is exact**: 0.2137m in iter 42 = 0.2137m in iter 40. The 0.0 pre-inflection scaling perfectly restores the baseline behavior for that section
- **Inflection scaling 0.4 > 0.3**: The inflection section benefits from moderate velocity correction despite the S-turn. The gate-3 region has enough ILC offset to benefit from some velocity consistency
- **Max error reverts to iter 40**: Confirms the max error location is in the pre-inflection section

### Suggestions for improving the iteration process
- Per-section parameters should always be swept when adding new correction dimensions
- The section_boundaries tuple format is becoming unwieldy — consider switching to a dict/dataclass
- When testing per-section configs, only 3-4 configs are needed: the extremes and 1-2 interpolations
