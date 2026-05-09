# Improving Drone Racing Performance Through Iterative Learning MPC

- **URL**: https://arxiv.org/abs/2508.01103
- **Authors**: Haocheng Zhao, Niklas Schluter, Lukas Brunke, Angela P. Schoellig
- **Year**: 2025
- **Venue**: arXiv preprint (v3)
- **Institution**: Learning Systems and Robotics Lab, Technical University of Munich

## Key Contribution

This paper adapts Iterative Learning Model Predictive Control (LMPC) for autonomous drone racing through three interlocking innovations: (1) an adaptive cost function that dynamically balances time-optimal performance against centerline adherence using a gate-proximity-aware weighting schedule, (2) a modified local safe set that discourages aggressive corner-cutting by artificially injecting states on the opposite side of the centerline, and (3) a Cartesian-based arc-length parameterization that avoids the singularities and integration wind-up errors inherent in Frenet-frame formulations. The system iteratively improves lap times starting from a conservative initial trajectory, converging to significantly faster racing lines over 10-20 iterations without requiring predefined reference trajectories beyond gate positions and ordering.

## Technical Approach

The core of the method is the LMPC formulation where each iteration solves a finite-horizon optimal control problem using data from all prior iterations as a "safe set" -- a convex hull of previously visited states that the optimizer can target as terminal conditions. This is what allows the system to progressively improve without a fixed reference trajectory.

### Adaptive Cost Function

The stage cost combines two terms:

- **Time-optimal cost**: `l_t(u) = c + ||u||^2_R` where c > 0 penalizes each additional time step, directly incentivizing faster completion.
- **Lateral deviation penalty**: `l_d(x) = ||[p - p_c(s)] / R_c(s)||^2_Qd` which penalizes deviation from the centerline, normalized by the corridor radius R_c(s) at arc-length position s.

The combined cost is `h(x,u) = l_t(u) + gamma(s) * l_d(x)` where gamma(s) is the adaptive weighting function. The critical insight is that gamma(s) is not constant -- it uses mirrored sigmoid functions centered at each gate location. Near gates, gamma(s) is high, enforcing tight centerline adherence to ensure gate passage. Between gates, gamma(s) is low, giving the optimizer freedom to cut corners and find faster paths. This spatial modulation is the key mechanism that prevents the common failure mode where pure time-optimal costs lead to gate misses after just a few aggressive iterations.

The ablation study in Table II confirms this design is essential: using only the time-optimal cost causes gate misses after iteration 2, while using only lateral deviation achieves negligible lap time improvement (about 0.3%). The combined adaptive cost with the modified safe set converges reliably to an 8.47s lap time.

### Modified Local Safe Set

Standard LMPC builds a safe set from previously observed states, but this biases the optimizer toward one side of the track. The paper addresses this by computing the average displacement of safe set states from the centerline, then generating artificial mirror states on the opposite side with conservatively over-approximated costs. This provides the optimizer with escape routes from corner-cutting tendencies while maintaining theoretical feasibility guarantees. The opposite corridor is defined as the region where `(p - p_c(s_bar))^T * p_bar <= 0`, ensuring the generated states are geometrically on the other side of the track.

### Arc-Length Parameterization

Gate centers are connected via piecewise cubic Hermite interpolation (CHI) and parameterized by arc length s. A k-d tree enables efficient nearest-point queries (0.68 +/- 0.17 ms per query). The arc-length estimate is corrected at each timestep via feedback, avoiding the drift that plagues pure integration-based Frenet approaches.

## Results

**Simulation (Split-S track):**
- From PID initialization (0.5 m/s): 64.25% lap time reduction (23.55s to 8.42s)
- From conservative MPCC++ (mu=0.02): 48.99% reduction (11.84s to 6.04s)
- From aggressive MPCC++ (mu=0.10): 23.22% reduction (7.71s to 5.92s)

**Real-world (Figure-8 track, Crazyflie 2.1):**
- From PID: 60.85% improvement (17.09s to 6.69s)
- From aggressive MPCC++: 6.05% improvement (6.45s to 6.06s)

The real-world experiments used a 200 Hz motion capture system with 0.4m gates on a Crazyflie 2.1. The LMPC ran at 30 Hz (compared to 90 Hz for baseline controllers) with an average solver time of 16.66 +/- 2.28 ms using acados with SQP/HPIPM.

## Relevance to Our System

Our sim-based racing line selection evaluates 10 candidate racing lines with a composite score of `0.7 * avg_tracking_error + 0.3 * worst_gate_error`. We want to add a race time term. This paper's adaptive weighting strategy offers several directly applicable insights:

1. **Spatially varying weights**: Rather than using a single global weight for race time vs. tracking accuracy, we should modulate the weight based on proximity to gates. Near gates, tracking accuracy should dominate (high weight on error terms); between gates, race time should dominate. This maps to our system as: for each candidate racing line, compute a weighted score where the gate-proximity segments use our current error-heavy weighting while inter-gate segments allow more weight on speed.

2. **Sigmoid-based scheduling**: The mirrored sigmoid approach gives a smooth, tunable transition. We could implement this as `gamma(s) = sum_over_gates(sigmoid(k * (d_gate - |s - s_gate|)))` where k controls transition sharpness and d_gate controls the gate influence radius. A reasonable starting point based on the paper would be to have high tracking weight within 1-2 gate radii and low tracking weight elsewhere.

3. **Three-term composite score**: Extend our scoring to `alpha(s) * avg_tracking_error + beta(s) * worst_gate_error + (1 - alpha(s) - beta(s)) * race_time` where alpha and beta are high near gates and low between gates. The paper's ablation shows that purely optimizing for time leads to gate misses, confirming that our gate-error term must remain significant near gates.

4. **Iterative refinement**: The LMPC framework improves over iterations by using prior solutions as warm starts. Our 10-candidate evaluation could similarly be made iterative -- pick the best candidate, perturb around it to generate new candidates biased toward the current best, and re-evaluate. This is essentially what the LMPC safe set provides: a memory of good solutions that constrains future exploration.

## Actionable Takeaways

1. **Implement gate-proximity-aware scoring**: Modify the racing line composite score to use spatially varying weights. Near gates, keep high weight on tracking error (0.7+). Between gates, shift weight toward race time (0.4-0.5 weight on time).

2. **Add corridor radius normalization**: The paper normalizes lateral deviation by corridor radius R_c(s). We should similarly normalize tracking error by gate size or approach corridor width so that the same absolute error is penalized more in tight sections.

3. **Consider a three-phase weight schedule**: (a) gate approach -- high tracking weight, (b) gate transit -- maximum tracking weight, (c) inter-gate cruise -- high time weight. Use sigmoid blending between phases.

4. **Safe set concept for candidate generation**: Instead of generating 10 independent candidates each iteration, bias new candidates toward the convex hull of previous top performers. This reduces wasted evaluations on clearly suboptimal regions of the racing line space.

5. **Solver parameters for reference**: N=8 horizon, K=20 safe set neighbors, 30 Hz control rate, SQP with 5 max iterations. These are useful baselines if we ever move to an online MPC approach.

## Limitations & Caveats

1. **Local optima**: The LMPC converges to local, not global, optima due to the nonconvex nature of the problem. The quality of the final solution depends on the initialization trajectory.
2. **Low control frequency**: 30 Hz LMPC vs. 90 Hz baselines. At high speeds in tight tracks, this may be insufficient for disturbance rejection. Our system runs at 100+ Hz which is more appropriate for aggressive racing.
3. **Yaw not optimized**: The paper holds yaw constant, which limits applicability in tracks requiring significant heading changes or where perception constraints (camera FOV) matter. Our system needs yaw planning for gate detection.
4. **Motion capture dependency**: Real experiments relied on 200 Hz mocap. The sim-to-real gap with onboard-only sensing (our competition scenario) is not addressed.
5. **Small gate size**: 0.4m gates on Crazyflie (a micro quadrotor). Scaling behavior to larger drones and gates at higher speeds is not validated.
6. **No aerodynamic drag modeling**: The 9-state model omits rotor drag effects, which become significant at the speeds our system targets (5+ m/s).

## Key Parameters / Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| Control frequency | 30 Hz | LMPC solve rate |
| Prediction horizon N | 8 | MPC lookahead steps |
| Safe set neighbors K | 20 | Number of nearest states used |
| Time penalty c | > 0 (not specified exactly) | Per-step time cost |
| Solver | acados v0.4.1 SQP + HPIPM | QP backend |
| Max SQP iterations | 5 | Per MPC solve |
| Max QP iterations | 20 | Per SQP step |
| Convergence tolerance | 1e-4 | SQP termination |
| Avg solver time | 16.66 +/- 2.28 ms | For N=8, K=20 |
| k-d tree query time | 0.68 +/- 0.17 ms | Arc-length lookup |
| State dimension | 9 | [position, velocity, Euler angles] |
| Control dimension | 4 | [thrust, roll_cmd, pitch_cmd, yaw_cmd] |
| System ID params | alpha_phi=-6.00, alpha_theta=-3.96 | First-order attitude dynamics |
| System ID params | beta_phi=6.21, beta_theta=4.08 | First-order attitude dynamics |
| Gate size (real) | 0.4m | Crazyflie experiments |
| Mocap rate | 200 Hz | State estimation input |
| Convergence iterations | 10-20 | Typical for lap time convergence |
