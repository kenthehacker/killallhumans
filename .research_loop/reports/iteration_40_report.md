# Iteration 40 Report

## Bottleneck: system_integration (ILC-controller coupling)

## Research Phase
Analyzed 3 papers:
1. **NGTC** (Pries & Ryll 2025) — Max tilt β=56° (0.977 rad), DFBC degrades 10x at tilt saturation
2. **ILC Mismatch Compensation** (Wu et al. 2024) — Plant-model mismatch in ILC causes convergence to suboptimal corrections
3. **LoL-NMPC** (Gupta et al. 2025) — 22-29% tracking improvement from modeling actuator saturation

All papers agreed: **unmodeled controller saturation is the primary source of tracking degradation in aggressive flight**.

## Hypothesis
The ILC inner sim models an unconstrained plant (no tilt limit), while the benchmark controller clips at max_tilt_rad=0.85 (49°). This saturation mismatch causes ILC corrections to be too aggressive for the benchmark's actual capability. Increasing max_tilt_rad should reduce the mismatch.

## Approaches Tested

### 1. max_tilt_rad sweep (0.85 → 1.05)
**Result**: ZERO effect on all metrics. All values produced identical results.
**Root cause**: `last_desired_acceleration` stores PRE-CLIP acceleration (line 200 of mpc_tracker.py). The kinematic sim's physics loop uses `accel_des` before tilt clipping occurs. Tilt clipping is cosmetic in the kinematic sim — it only affects the roll/pitch angles used for attitude commands, which are not used by the kinematic sim dynamics.

**This is a critical architectural finding**: any future iteration targeting tilt limits will have zero effect in kinematic sim. This only matters in PyBullet/real hardware where attitude dynamics affect translational motion.

### 2. Racing line eval gains sync (kp=6→7, kd=4→5.5, ff=0.4→0.50)
**Result**: ZERO metric change. Same racing line basin selected (candidate_idx=11).
**Explanation**: The racing line cache key doesn't include eval gains, but even after cache deletion and full re-optimization, the same basin wins. The kinematic eval sim has sufficiently large margin that gain changes don't flip the basin selection. Committed as code correctness fix.

### 3. Compound curvature boost sweep
Tested 5 configurations of (s_turn_boost, helix_boost):

| Config | s_turn | helix | Race Time | Avg Error | Max Error |
|--------|--------|-------|-----------|-----------|-----------|
| baseline | 1.20 | 1.15 | 14.07s | 0.1501m | 0.727m |
| lower_both | 1.10 | 1.10 | 14.01s | 0.1687m | 0.555m |
| higher_both | 1.30 | 1.20 | 14.11s | 0.1477m | 0.727m |
| no_boosts | 1.00 | 1.00 | 13.88s | 0.1748m | 0.555m |
| aggressive | 1.35 | 1.25 | 14.13s | 0.1470m | 0.727m |

**Analysis**: There's a clean trade-off. Higher boosts → slower (safer at turns) → lower avg error but higher race time. The best avg error (0.1470 at 1.35/1.25) is only -2.1% better than baseline but adds +0.06s race time. Not a clear improvement — it shifts the Pareto frontier rather than advancing it.

Per-gate analysis shows improvement is concentrated at gates 6-8 (helix region): gate-6 improves 17%, gate-7 improves 7.6%, gate-8 improves 4.7%. But gates 9-10 regress slightly.

### 4. Additional fine sweep (1.25/1.15, 1.30/1.15, 1.35/1.25)
Confirmed 1.35/1.25 is the best for avg error but the race time trade-off makes it ambiguous.

## Outcome: No clear improvement found

After 4 approaches, no change produced an unambiguous improvement (better in ALL key metrics). The system is at a local optimum where:
- **Trajectory**: Locked by basin switching boundaries (±1% TOPP floor change triggers catastrophic failure)
- **Controller**: Locked by ILC-controller coupling (any controller change is masked by ILC)
- **ILC**: Locked in fragile equilibrium (parameter changes cause 27s race time)
- **Speed profiling**: At Pareto frontier between race time and tracking error

## Committed Changes
1. Racing line eval gains sync (kp=7, kd=5.5, ff=0.50) — code correctness, zero metric impact

## Key Findings for Future Iterations
1. **max_tilt_rad is cosmetic in kinematic sim** — only affects attitude (roll/pitch angles), not dynamics. `last_desired_acceleration` returns pre-clip values.
2. **Compound curvature boosts are at Pareto frontier** — can trade race time for tracking error, but can't improve both.
3. **Racing line basin selection is robust to eval gain changes** — same basin wins regardless of controller gains.
4. **The ILC-controller coupling remains the dominant architectural constraint** — it prevents independent optimization of any single component.

## Metrics
- Before: race_time=14.07s, avg_error=0.1501m, max_error=0.727m, gates=12/12
- After: race_time=14.07s, avg_error=0.1501m, max_error=0.727m, gates=12/12 (unchanged)

## Recommendation for Iteration 41
The system needs an **architectural change** to break out of the local optimum:
1. **Re-derive ILC from scratch with current gains** — full ILC parameter search (alpha, cutoff, sections, iterations) starting from zero offsets with the current controller
2. **PyBullet validation** — test whether kinematic sim findings transfer to full physics
3. **MPCC/path-following controller** — replace PD+feedforward with a controller that directly optimizes path-following, bypassing the ILC coupling entirely
