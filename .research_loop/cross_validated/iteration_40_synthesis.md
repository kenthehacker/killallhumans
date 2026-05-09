# Iteration 40 Research Synthesis: Tilt Constraint as ILC-Controller Coupling Root Cause

## Papers Analyzed
1. **NGTC** (Pries & Ryll 2025) — Max tilt β=56° (0.977 rad), DFBC degrades 10x at saturation
2. **ILC Mismatch Compensation** (Wu et al. 2024) — Plant-model mismatch in ILC causes convergence to suboptimal corrections
3. **LoL-NMPC** (Gupta et al. 2025) — 22-29% tracking improvement from modeling actuator saturation

## Research Consensus

All three papers agree on a core principle: **controller saturation effects that aren't modeled in the control/planning pipeline cause significant tracking degradation**.

- NGTC shows DFBC error goes from 0.23m → 2.39m (10x) when tilt-saturating
- LoL-NMPC shows 3x prediction error improvement (0.6m → 0.2m) from modeling saturation
- ILC Mismatch paper shows ILC convergence degrades when the internal plant model doesn't match actuation limits

## Contradictions

None significant. All papers agree that unmodeled saturation is the primary source of degradation in aggressive flight tracking.

## Key Insight: The Tilt Constraint IS the ILC-Controller Coupling Mechanism

The ILC inner sim uses a simple PD controller with **no tilt constraint** — it limits only total acceleration magnitude (15 m/s²). The benchmark controller clips roll and pitch to **max_tilt_rad=0.85 (49°)**, which limits lateral acceleration to approximately g·tan(0.85) ≈ 11.2 m/s².

This means:
1. **ILC computes corrections assuming the controller can produce 15 m/s² lateral acceleration**
2. **The benchmark controller can only produce ~11.2 m/s² lateral acceleration due to tilt clipping**
3. **The ILC corrections are "too aggressive" for the benchmark's actual capability at turns**
4. **This mismatch partially explains why ILC gain sync failed**: with stronger gains (kp=7), the controller would demand even more tilt → hit the limit more → deviate more from ILC predictions

The ILC-controller coupling is not (only) about gain mismatch — it's about **saturation mismatch**.

## Proposed Implementation: Increase max_tilt_rad

Increasing max_tilt_rad from 0.85 (49°) to ~0.98 (56°) would:
- Increase max lateral acceleration from 11.2 → 14.9 m/s² (33% increase)
- Reduce saturation frequency at tight gates (2, 3, 7)
- Better align benchmark behavior with ILC's unconstrained model
- NOT affect ILC computation (ILC has no tilt model)

This is supported by:
- NGTC uses β=56° as standard (0.977 rad)
- Our damping ratio ζ=1.13 provides stability margin
- The ILC corrections were computed for an unconstrained plant

## Risk Assessment

- **Low risk**: Tilt increase only affects the benchmark controller, not ILC or trajectory
- **Moderate risk**: Higher tilt enables faster velocity buildup → more drag interaction
- **Low risk**: Damping ratio ζ=1.13 (overdamped) should prevent oscillation
- **Mitigation**: Sweep values incrementally (0.90, 0.95, 1.00) rather than jumping to 0.98

## Evidence Strength: STRONG

Three independent papers confirm that tilt saturation degrades DFBC tracking. Our system shows both roll and pitch hitting the 0.85 limit. The ILC inner sim has no tilt constraint. The logic chain is robust.
