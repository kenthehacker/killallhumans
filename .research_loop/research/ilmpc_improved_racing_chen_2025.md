# Improving Drone Racing Performance Through Iterative Learning MPC
- **URL**: https://arxiv.org/abs/2508.01103
- **Authors**: Haocheng Zhao, Niklas Schluter, Lukas Brunke, Angela P. Schoellig (TU Munich)
- **Year**: 2025

## Key Contribution

This paper presents a Learning Model Predictive Control (LMPC) framework for autonomous drone racing that iteratively improves lap performance by leveraging data from previous laps. The three main novelties are: (1) an adaptive cost function that dynamically weights time-optimal objectives against centerline adherence using a mirrored sigmoid schedule tied to arc-length position, (2) a shifted local safe set that prevents the optimizer from converging to excessively corner-cutting trajectories that would be unsafe under model mismatch, and (3) a Cartesian-coordinate formulation that avoids Frenet-frame singularities while still accommodating spatial safety constraints. Unlike classical ILC which applies additive feedforward corrections, LMPC constructs a control-invariant safe set from the union of all previously successful trajectories and uses it as a terminal constraint in the MPC optimization, allowing the controller to explore progressively faster solutions while guaranteeing recursive feasibility.

## Technical Approach

The drone state is 9-dimensional (3D position, 3D velocity, 3 Euler angles) with 4-dimensional control input (collective thrust plus three commanded Euler angles). The dynamics are modeled as a control-affine nonlinear system with translational dynamics from Newton's law and rotational dynamics approximated as first-order integrators with identified time constants.

The LMPC stage cost has two components: a time-optimal term l_t(u) = c + ||u||^2_R that penalizes additional time steps, and a lateral deviation term l_d(x) that penalizes distance from the centerline normalized by the local track radius. The adaptive weighting gamma(s) blends these via mirrored sigmoid functions at each gate -- near gates the deviation penalty is high (keeping the drone on target), while on straightaways the time-optimal cost dominates (allowing aggressive shortcuts).

The safe set is constructed as the union of states visited in all previous successful laps. At each MPC step, the k=20 nearest neighbors (by Euclidean distance) form a local convex safe set used as the terminal constraint. The key innovation is that some of these neighbors are artificially shifted across the centerline to create a symmetric feasible region, preventing the optimizer from always converging to the inside of corners. The cost-to-go Q^j(x) for each historical state is computed as the actual cost incurred from that state to lap completion, providing a terminal cost approximation.

Arc-length parameterization uses piecewise cubic Hermite interpolation through gate centers. Arc-length lookup is accelerated from 4.21 ms (brute force) to 0.68 ms using a k-d tree, which is critical given the 30 Hz control frequency. The MPC uses a prediction horizon of N=8, solved via SQP with at most 5 iterations and 20 QP iterations per SQP step (convergence tolerance 1e-4).

Critically, this is NOT classical ILC with explicit velocity feedforward corrections. The velocity information from previous iterations enters implicitly through the safe set and cost-to-go function. The MPC predicts velocity via the dynamics model, and the safe set constrains the terminal state to be near previously achieved (position, velocity) pairs. This means velocity "learning" happens indirectly -- as the safe set expands with faster trajectories, higher velocities become feasible terminal states.

## Results

Starting from a conservative PID initialization at 0.5 m/s, the method achieves 60.85% lap time reduction in real-world experiments (17.09s to 6.69s) and 64.25% in simulation. Starting from the stronger MPCC++ baseline (which is already near time-optimal), improvements of 6-30% are still achieved depending on the MPCC++ tuning aggressiveness.

Convergence typically occurs within 3-10 iterations depending on initialization quality. The convergence is monotonic (non-decreasing performance per iteration) but reaches local optima due to nonlinearity and nonconvexity. The final converged lap time depends on initialization -- starting from better initial trajectories leads to better final performance, suggesting the landscape has multiple local optima.

Real-world experiments use the Crazyflie 2.1 quadrotor with motion capture at 200 Hz and identified first-order attitude dynamics (alpha_phi=-6.00, alpha_theta=-3.96, beta_phi=6.21, beta_theta=4.08). The LMPC runs at 30 Hz versus 90 Hz for the baseline controllers, making the computational overhead a practical consideration.

## Relevance to Our System

Our system currently uses a geometric SE(3) tracker with pre-computed min-snap trajectories. The LMPC approach is relevant in several ways:

1. **Iterative trajectory refinement**: Rather than computing one trajectory offline, we could iteratively improve our racing line using data from previous simulation runs. The safe set concept is particularly interesting -- it provides a principled way to expand the envelope of feasible trajectories without risking crashes.

2. **Adaptive cost weighting near gates**: The sigmoid-based blending between time-optimal and centerline-tracking objectives near gates is directly applicable. Our current racing line optimizer could benefit from gate-proximity-aware cost shaping -- being precise near gates but aggressive on straightaways.

3. **Arc-length parameterization with k-d tree**: The k-d tree acceleration for arc-length lookup (6x speedup) is a useful practical technique if we move toward spatial parameterization of our trajectory.

4. **Velocity learning is implicit, not explicit**: This is an important finding. Unlike dual-channel ILC approaches that separately correct position and velocity feedforward signals, LMPC learns velocity improvements indirectly through the expanding safe set. For our system, this suggests that explicit velocity ILC corrections (as in the dual ILC papers) may offer faster convergence since they directly address velocity tracking errors.

## Actionable Takeaways

1. **Gate-proximity cost shaping**: Implement sigmoid-based weighting that increases tracking precision near gates and relaxes it on straightaways. This can be added to our existing trajectory optimizer or racing line module without changing the control architecture.

2. **Safe set from simulation rollouts**: Store successful simulation trajectories and use them to define feasible regions for trajectory optimization. Even without full LMPC, using historical trajectory data to warm-start or constrain our optimizer would be valuable.

3. **Shifted safe set for anti-corner-cutting**: When optimizing racing lines, add virtual constraint points on the opposite side of the centerline to prevent excessive shortcuts that are only feasible under perfect conditions.

4. **k-d tree for arc-length queries**: If we implement spatial trajectory parameterization, use k-d tree acceleration for the arc-length projection step.

5. **Iteration budget**: Plan for 5-10 improvement iterations in simulation before deploying. The diminishing returns after ~5 iterations suggest this is sufficient for most of the performance gain.

## Limitations & Caveats

1. **Local optima**: The algorithm converges to local optima that depend on initialization quality. Better initial trajectories lead to better final performance, so the initialization strategy matters significantly.

2. **Low control frequency**: LMPC runs at 30 Hz versus 90 Hz for baseline controllers. The computational cost of the safe set lookup and SQP solve is non-trivial. For our 100+ Hz requirement, a full LMPC implementation may be too expensive without significant optimization.

3. **No explicit velocity learning**: Unlike classical ILC which directly corrects velocity feedforward signals, LMPC learns velocity improvements only indirectly. This may lead to slower convergence on velocity-dominated errors compared to dedicated velocity ILC approaches.

4. **Motion capture dependency**: All real-world results use motion capture at 200 Hz. Transfer to vision-based state estimation (our setting) would introduce additional uncertainty that could degrade the safe set guarantees.

5. **Small platform**: Results on Crazyflie 2.1 with 0.4m gates. Scaling behavior to larger drones and tracks is not demonstrated.

6. **No comparison to pure ILC or RL**: The baselines are PID and MPCC++. A direct comparison to classical ILC or reinforcement learning approaches would better contextualize the contribution.

## Key Parameters / Constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| Control frequency | 30 Hz | LMPC; baselines run at 90 Hz |
| Prediction horizon N | 8 steps | Selected via runtime analysis |
| Safe set neighbors k | 20 | For local convex approximation |
| Max SQP iterations | 5 | Per MPC solve |
| Max QP iterations | 20 | Per SQP iteration |
| Convergence tolerance | 1e-4 | SQP termination |
| Gate size | 0.4m x 0.4m | Crazyflie-scale |
| Attitude time constants | alpha_phi=-6.00, alpha_theta=-3.96 | Identified first-order model |
| Attitude gains | beta_phi=6.21, beta_theta=4.08 | Identified first-order model |
| Arc-length k-d tree lookup | 0.68 ms | Down from 4.21 ms brute force |
| Convergence iterations | 3-10 | Depends on initialization |
| Best real-world lap time | 6.06s | From MPCC++ mu=0.1 init |
| Lap time improvement range | 6-61% | Depends on initialization |
