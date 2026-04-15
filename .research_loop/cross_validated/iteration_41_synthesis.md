# Iteration 41 Research Synthesis: Velocity-Corrected ILC

## Current State
- Race time: 14.07s, avg error: 0.150m, max error: 0.727m, 100% gate pass
- System confirmed at local optimum after 40 iterations
- Parameter tuning exhausted across controller gains, ILC parameters, racing line, and TOPP retiming

## Core Insight: ILC Position-Velocity Mismatch

The current ILC computes position offsets and adds them to the reference position at runtime. However, the reference velocity remains from the ORIGINAL trajectory (trajectory_optimizer.py:250, benchmark.py:409). This creates a fundamental inconsistency:

- **Position reference**: ref.position + ilc_offset → the drone should be SOMEWHERE ELSE
- **Velocity reference**: ref.velocity → the drone should be moving AS IF IT WERE ON THE ORIGINAL PATH

This mismatch means:
1. The PD controller's velocity error term (`kd * (ref.velocity - actual_vel)`) is pulling the drone toward the original path's velocity profile, partially undoing the ILC correction
2. The feedforward acceleration (from the original trajectory) is oriented along the original path, not the corrected path
3. As ILC offsets grow (e.g., helix section with 0.35m max), the velocity mismatch grows proportionally

## Research Basis

### Schoellig et al. 2012 (Optimization-based ILC for quadrotors)
- Key finding: ILC should correct FEEDFORWARD INPUTS, not just reference positions
- Their approach modifies both the position reference AND the velocity/acceleration feedforward
- Achieved 87% error reduction in 3-5 iterations
- **Our gap**: We only correct position, leaving feedforward uncorrected

### "Leveling the Playing Field" (Kunapuli 2025)
- Key finding: feedforward is the single most important component for geometric tracking
- "Feedforward information is crucial to encode future reference information"
- **Our gap**: Our ILC modifies where the drone should be, but not how it should get there

### Track-centric ILC (Nam et al. 2026)
- Co-optimizes both position (lateral deviation) AND velocity profiles
- Achieved 20.7% lap time reduction by jointly optimizing reference trajectory
- **Our gap**: We optimize position corrections but velocity profile remains locked

### Segment-based AILC (Zhang et al. 2026)
- Proves segment-independent learning prevents cross-contamination
- Recommends per-section bandwidth selection: S-turn needs ~1.8 Hz, helix needs ~0.35 Hz
- Current inflection bandwidth (0.40 Hz) was improved in iter 28 but still suboptimal
- **Our gap**: S-turn section still has low bandwidth; velocity corrections could be more impactful here

### ILC Mismatch Compensation (Wu 2024, Wang 2024)
- Addresses gap between ILC learning model and real system dynamics
- Shows that model mismatch causes residual tracking error that can't be eliminated by more iterations
- **Our gap**: The position-velocity mismatch IS a form of model mismatch — the ILC learns in a system where velocity matches position (inner sim), but executes in one where they don't

## Consensus
Multiple papers agree: ILC that corrects ONLY position and leaves velocity/acceleration uncorrected is suboptimal. The velocity correction should be the time derivative of the position offset, ensuring consistency between the corrected position and the corrected velocity.

## Proposed Approach: Velocity-Corrected ILC

### Algorithm
1. Run ILC as before — compute position offsets via cross-track correction with Butterworth Q-filter
2. In the ILC inner sim, ALSO apply velocity corrections derived from the current cumulative_offset
3. Velocity correction: `vel_offset[k] = (pos_offset[k+1] - pos_offset[k-1]) / (2*dt)` (central difference)
4. Since position offsets are Butterworth-filtered (smooth), their derivatives are also smooth — NO FD noise problem
5. Return both position and velocity offset arrays
6. In benchmark: apply both `target_pos + ilc_pos_offset` AND `target_vel + ilc_vel_offset`

### Why This Is Different From Iteration 25's Failure
- Iteration 25 tried "Cross-track ILC with FD derivatives" — this RECOMPUTED THE ENTIRE TRAJECTORY'S velocity/acceleration from shifted positions, introducing noise in all derivative orders
- This approach only computes the derivative of the ILC OFFSET ITSELF, which is a Butterworth-filtered smooth signal. The original trajectory derivatives remain untouched
- The derivative is computed via numpy.gradient (central differences with proper boundary handling), not raw FD on position data

### Expected Impact
- Reduced position-velocity mismatch → better tracking, especially in high-offset regions (helix)
- The velocity correction helps the PD controller "agree" on both position AND velocity targets
- Should allow the ILC to converge to lower residual error since the inner sim more accurately models the benchmark
- Conservative estimate: 5-15% reduction in avg tracking error
