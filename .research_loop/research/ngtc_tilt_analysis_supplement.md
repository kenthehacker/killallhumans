# NGTC: Tilt Constraint Analysis Supplement (Extracted for Iteration 40)

- **URL**: https://arxiv.org/abs/2510.12611
- **Authors**: Lukas Pries, Markus Ryll
- **Year**: 2025
- **Original analysis**: ngtc_robust_agile_2025.md

## Tilt-Specific Findings (New Extraction for Iteration 40)

### Maximum Tilt Angle
The NGTC paper specifies **β = 56° (0.977 rad)** as the maximum tilt angle parameter in Table I of the paper. This is the tilt limit used in both simulation and hardware experiments.

### Comparison with Our System
Our current max_tilt_rad = 0.85 rad (49°) is **7 degrees more conservative** than the NGTC baseline. The NGTC paper uses β=56° for their DFBC baseline as well as their neural-augmented controller, suggesting this is a standard value for aggressive quadrotor flight.

### Saturation Effects on DFBC
The paper provides critical data on what happens when DFBC (which is structurally identical to our GeometricTracker — a PD controller with differential flatness feedforward) hits actuator saturation:

| Trajectory Type | DFBC Error | NGTC Error | Improvement |
|----------------|------------|------------|-------------|
| Feasible (Horizontal Loop) | 0.23m | 0.20m | 13% |
| **Infeasible (Horizontal Loop)** | **2.39m** | **1.42m** | **40%** |
| Feasible (Helix) | 0.11m | 0.12m | -9% |
| **Infeasible (Helix)** | **1.69m** | **1.09m** | **35%** |

**Key finding**: DFBC tracking error increases by **10-15x** when trajectories become actuator-infeasible. This is because tilt clipping creates a cascading error: the controller demands more tilt than allowed → acceleration is reduced → position error grows → controller demands even more tilt → sustained saturation.

### Geometric Controller Baseline Gains
| Parameter | Value | Our Value | Ratio |
|-----------|-------|-----------|-------|
| Kx (position) | 18.0 | 7.0 | 2.6x |
| Kv (velocity) | 8.0 | 5.5 | 1.5x |
| Max tilt β | 56° (0.977 rad) | 49° (0.85 rad) | 1.15x |

The NGTC paper uses gains 2-3x higher than ours, with a higher tilt limit. This suggests our gains are conservative and the tilt limit is a binding constraint.

### Why Higher Tilt Helps
1. **More lateral acceleration authority**: At 0.85 rad, max lateral accel ≈ g·tan(0.85) = 11.2 m/s². At 0.98 rad, max lateral accel ≈ g·tan(0.98) = 14.9 m/s² (33% increase).
2. **Reduces saturation frequency**: Gates 2, 3, 7 require aggressive turns where the controller saturates. More headroom means the controller stays in the linear regime longer.
3. **Better matches ILC assumptions**: The ILC inner sim has no tilt limit, computing corrections assuming the controller CAN produce high lateral accelerations. Increasing max_tilt makes the benchmark controller behave more like the ILC expects.

### Actionable Takeaways for Iteration 40

1. **Increase max_tilt_rad from 0.85 to 0.98 rad** (matching NGTC's 56°). This is supported by NGTC as a standard value for aggressive racing flight.
2. **Conservative option**: 0.90 rad (51.6°) as a midpoint if 0.98 causes instability.
3. **Aggressive option**: 1.05 rad (60°) which would make the mag limit (15 m/s²) binding before the tilt limit, effectively removing the tilt constraint.
4. **This change does NOT affect ILC**: The ILC inner sim has no tilt model, so its corrections remain unchanged. Only the benchmark controller behavior changes.
5. **Risk**: Higher tilt allows more aggressive accelerations, which combined with ff=0.50 could cause overshoot. The damping ratio ζ=1.13 should prevent this, but needs verification.

### Limitation
NGTC achieves its best results with neural augmentation, not just by increasing tilt limits. Our approach of simply increasing the tilt limit captures only part of the benefit. However, for our kinematic sim, the tilt limit is likely the dominant constraint since we don't have real actuator dynamics.
