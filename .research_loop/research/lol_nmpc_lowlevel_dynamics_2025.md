# LoL-NMPC: Low-Level Dynamics Integration in NMPC for UAVs

- **URL**: https://arxiv.org/abs/2506.02169
- **Authors**: Parakh M. Gupta, Ondřej Procházka, Jan Hřebec, Matej Novosad, Robert Pěnička, Martin Saska
- **Year**: 2025
- **Venue**: arXiv preprint (Czech Technical University, Multi-robot Systems Group)

## Key Contribution

LoL-NMPC addresses a fundamental problem in high-performance quadrotor control: the mismatch between the high-level controller's model of actuator behavior and the actual low-level PID dynamics. Standard approaches (including DFBC/geometric control and standard NMPC) assume that the low-level rate controller responds instantaneously to body rate commands — an assumption that breaks down during aggressive maneuvers where actuator saturation occurs. LoL-NMPC explicitly models the PID controller dynamics within the NMPC state vector, augmenting it with integral error terms and normalized motor angular velocities. This allows the optimizer to predict and avoid actuator saturation before it occurs, rather than reacting after the fact.

The fundamental insight is profound for our setting: when a high-level controller commands accelerations that require specific roll/pitch angles, and those angles are clipped by a max_tilt constraint, the actual executed trajectory deviates from the controller's prediction. This prediction error accumulates over the maneuver, degrading tracking performance. By modeling the saturation within the planning/control loop, the system can plan around it.

## Technical Approach

The standard quadrotor control hierarchy is:
1. High-level controller → desired body rates
2. Low-level PID → motor throttle commands
3. Motor dynamics → forces/torques

LoL-NMPC augments the NMPC state vector to include:
- Standard states: position, quaternion, velocity, body rates
- PID integral error (3 axes)
- Normalized angular velocity per motor

The mixer matrix M maps desired body rates to motor commands: `r_c = Cx^pid + Du^pid`. Crucially, motor commands are bounded: `r_min ≤ r_c ≤ r_max`. By expressing these bounds as linear constraints in the optimization, LoL-NMPC prevents the optimizer from commanding infeasible body rates.

## Results

**Simulation**: 22.31% average tracking error reduction, up to 29.16% on CPC trajectories at 3.5g.
**Real-world**: 21.97% average RMSE reduction, up to 38.6% improvement on polynomial trajectories at 2.5g. Speeds up to 98.57 km/h.

Standard NMPC shows 0.6m prediction error at the 10th prediction step vs LoL-NMPC's 0.2m — a 3x improvement in prediction accuracy when actuator saturation is modeled.

## Relevance to Our System

This paper directly illuminates our ILC-controller mismatch problem. Our ILC inner sim uses a simple PD controller without tilt constraints, while the benchmark controller clips at max_tilt_rad=0.85 (49°). This is exactly the kind of "high-level vs low-level model mismatch" that LoL-NMPC identifies:

1. **The ILC thinks the plant can produce more lateral acceleration than it actually can** (no tilt limit in ILC vs 0.85 rad limit in benchmark)
2. **This mismatch causes the ILC to compute suboptimal corrections** — corrections calibrated for an unconstrained plant applied to a constrained one
3. **The solution is to either model the constraint in the ILC or relax the constraint in the benchmark** to reduce the mismatch

## Actionable Takeaways

1. **Increase max_tilt_rad in the benchmark controller** from 0.85 rad (49°) toward 0.98 rad (56°, matching NGTC) or higher, reducing the gap between ILC's unconstrained model and the benchmark's constrained reality
2. **Add tilt limiting to the ILC inner sim** to make its plant model match the benchmark (but this requires recomputing all ILC corrections)
3. **The simpler approach (increasing tilt limit) is preferable** because it requires only changing one parameter and doesn't disrupt the converged ILC corrections

## Limitations & Caveats

- LoL-NMPC uses NMPC, not PD+ILC like our system
- The specific state augmentation approach doesn't directly apply to our ILC formulation
- Our kinematic sim doesn't model motor dynamics, so the full benefit of low-level modeling isn't available
- The 56% computation increase may matter for real-time deployment

## Key Parameters / Constants

- Body rate limits: 6 rad/s per axis
- Thrust-to-weight ratio: ~7
- Standard NMPC prediction error at 10 steps: 0.6m vs LoL-NMPC: 0.2m
- 22-29% tracking improvement from modeling low-level dynamics
- Real-world validation at up to 3.5g and 98.57 km/h
