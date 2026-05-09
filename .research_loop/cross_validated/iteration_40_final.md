# Iteration 40 Cross-Validated Research — FINAL

## Synthesis
The ILC-controller coupling identified in iteration 39 has a specific mechanism: the ILC inner sim models an **unconstrained** plant (no tilt limit), while the benchmark controller clips at max_tilt_rad=0.85 (49°). Three papers (NGTC, LoL-NMPC, ILC Mismatch Compensation) confirm that unmodeled saturation is the primary source of tracking degradation in aggressive quadrotor flight.

## Validated Approach: Increase max_tilt_rad
- From: 0.85 rad (49°)
- To: sweep 0.90, 0.95, 1.00, 1.05 rad
- Rationale: Reduce plant-model mismatch between ILC (unconstrained) and benchmark (constrained)
- Literature support: NGTC uses 0.977 rad (56°) as standard for aggressive flight

## Cross-Validation Challenges and Responses

### Challenge 1: Velocity Instability
Higher tilt → more lateral accel → higher velocities → more drag. At max_speed=15 m/s, velocity clamping could create artifacts.
**Mitigation**: Monitor max velocity in sweep results. If velocity approaches 15 m/s, the change is too aggressive.

### Challenge 2: ILC Corrections Compatibility
The ILC corrections were computed for the OLD controller (kp=6, kd=4, ff=0.4) with NO tilt limit. Increasing tilt in the benchmark makes the benchmark behave MORE like the ILC expects. The corrections should become MORE accurate, not less.
**Validation**: Compare per-gate errors before/after to confirm improvement at high-error gates.

### Challenge 3: Damping Ratio at Higher Tilt
With larger tilt angles, the linearization assumption weakens. ζ=1.13 was computed for small angles. At 0.98 rad (56°), the nonlinear effects are significant.
**Mitigation**: Start with conservative 0.90 and sweep up. Monitor for oscillation (check per-gate error patterns).

### Challenge 4: DroneConstraints.max_tilt_angle vs TrackerConfig.max_tilt_rad
Two separate parameters. DroneConstraints affects trajectory feasibility checking. TrackerConfig affects the actual controller. Must change BOTH consistently.
**Verification**: Check if DroneConstraints.max_tilt_angle is used in trajectory optimization.

## Final Recommendation
Sweep max_tilt_rad from 0.90 to 1.05 in steps of 0.05. Keep the best value. This is a low-risk, high-potential change because:
1. It doesn't touch ILC parameters (avoiding the coupling trap)
2. It doesn't touch the trajectory (avoiding basin switching)
3. It only changes how aggressively the controller can respond to errors
4. The ILC corrections become MORE aligned with controller capability
