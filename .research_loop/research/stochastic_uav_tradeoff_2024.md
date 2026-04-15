# Stochastic Control of UAVs: An Optimal Tradeoff between Performance, Flight Smoothness and Control Effort

**Authors:** George Rapakoulias, Panagiotis Tsiotras (Georgia Tech)
**Published:** September 2024, arXiv:2409.10369
**Venue:** Systems and Control

## Key Contribution

This paper introduces Optimal Covariance Steering (OCS) as a principled framework for balancing three competing objectives in UAV trajectory tracking: tracking accuracy, flight smoothness (measured via angular acceleration), and control effort. Traditional high-gain approaches like Incremental Nonlinear Dynamic Inversion (INDI) achieve tight tracking but at the cost of aggressive actuator usage and jerky flight. The authors show that by formulating the control problem as a semidefinite program (SDP) over the distribution of tracking errors — not just the mean trajectory — they can systematically tune the tradeoff between position accuracy and actuator smoothness through covariance constraints. The second contribution is a hybrid disturbance estimator combining a linear drag model with a small ReLU neural network, adapted online via an Extended Kalman Filter, which produces cleaner wind estimates than the low-pass filtering approach used by INDI.

## Technical Approach

The system has two major components:

**1. Aerodynamic Disturbance Estimation.** The drag force is modeled as `f_d = C_d(w - v) + Phi(q, eta_bar)(w - v)`, where `C_d` is a diagonal linear drag matrix and `Phi` is a neural network with architecture {8, 20, 20, 3} that takes attitude quaternion and motor RPM as inputs. The wind velocity `w` is estimated online using an EKF with process noise Q and measurement noise R, using the Jacobian `H_k = C_d + Phi(q_k, eta_bar_k)`. This is significantly more principled than INDI's approach of differentiating accelerometer readings through a low-pass filter, which introduces phase lag and noise amplification.

**2. Optimal Covariance Steering (OCS).** The quadrotor translational dynamics are reduced to a 3D double integrator under stochastic disturbances: `x_{k+1} = A_k x_k + B_k u_k + D_k w_k`. The control law is affine: `u_k = K_k(x_k - mu_k) + v_k`, where `K_k` is a feedback gain and `v_k` is a feedforward term. The optimization minimizes a cost combining state penalty (Q = I_6), control penalty (R = I_3), and covariance terms, subject to: (a) terminal covariance constraint `Sigma_N <= Sigma_f`, (b) chance constraints on position `P(x_k in X) >= 1 - epsilon_1`, and (c) chance constraints on control `P(u_k in U) >= 1 - epsilon_2`. The chance constraints are converted to deterministic SDP constraints via the 3-sigma rule: `L^T Sigma_k L <= (delta_x/3)^2 Sigma_c` for position bounds of delta_x = 0.025m, and `Y_k <= (delta_u/3)^2 I_3` for control bounds of delta_u = 10N.

The key insight is that the covariance constraints directly control how much the controller is allowed to "spread" the tracking error distribution. Tightening `Sigma_f` forces more aggressive control; loosening it permits smoother flight at the cost of slightly larger tracking errors.

## Results

Experiments were conducted on a 680g 5-inch racing quadrotor with PX4 autopilot and RockPi-4B offboard computer, using external vision at 100 Hz, tested in Georgia Tech's Indoor Flight Laboratory with industrial fans (3-4 m/s) and leaf blowers (up to 10 m/s).

**Figure-8 Tracking (12 m/s max speed, 22 m/s^2 max accel):**

| Method | RMS Position Error (cm) | RMS Angular Accel (rad/s^2) |
|--------|------------------------|-----------------------------|
| OCS + EKF (proposed) | 5.9 | 17.1 |
| OCS + INDI | 5.3 | 18.6 |
| LQR + EKF | 4.9 | 19.0 |
| LQR + INDI | 7.3 | 23.2 |

**Landing with Cone Constraints:**

| Method | RMS Position Error (cm) | RMS Angular Accel (rad/s^2) |
|--------|------------------------|-----------------------------|
| OCS + EKF | 6.0 | 10.1 |
| OCS + INDI | 5.6 | 12.0 |
| LQR + EKF | 4.6 | 16.0 |
| LQR + INDI | 3.6 | 17.1 |

The OCS methods consistently achieve 10-26% lower angular acceleration (smoother flight) with only 1-2 cm increase in tracking error compared to LQR baselines. The EKF-based estimator produces cleaner drag estimates than INDI's low-pass filter approach.

## Relevance to Our System

This paper is directly relevant to our gate-3 binding constraint problem. Our current situation — 0.226m error at gate-3 with only 0.024m headroom — suggests we may be operating at the edge of what our controller can achieve without sacrificing smoothness or inducing oscillations. The OCS framework offers a principled way to reason about this tradeoff:

1. **Per-section covariance budgets.** Rather than uniform controller gains, we could assign tighter covariance constraints near gates (where tracking accuracy matters) and looser constraints on straight segments (where smoothness matters for speed). This directly maps to our per-section ILC corrections — instead of heuristic ILC caps, we could compute section-specific gain matrices from an OCS solve.

2. **Disturbance estimation architecture.** Their hybrid linear+NN drag estimator with EKF adaptation is a cleaner approach than our current EKF's process noise model. At 12 m/s speeds on a figure-8, they achieve 5.9 cm RMS — substantially better than our 15.9 cm average. While our track is more complex, the estimation architecture deserves study.

3. **TOPP interaction.** The smoothness-performance tradeoff is exactly what limits our TOPP compression. If we could formally relax smoothness constraints on sections where tracking error is low, we could compress more aggressively without hitting the tracking error ceiling.

## Actionable Takeaways

- **Covariance-aware gain scheduling:** Implement section-dependent controller gains derived from the OCS principle — tight near gates, relaxed between them. This could replace or augment our ILC per-section scaling.
- **Drag estimation upgrade:** Consider augmenting our EKF with a learned residual drag model (even a small NN with {8,20,20,3} architecture), adapted online. Their EKF formulation for wind estimation is straightforward to integrate.
- **Chance constraint tuning:** The 3-sigma position bound of delta_x = 0.025m is extremely tight. For our system with 0.25m threshold at gates, a corresponding delta_x of ~0.08m (0.25/3) would give 99.7% confidence of staying within threshold — a useful design parameter.
- **Smoothness metric:** Track angular acceleration as a diagnostic. If our gate-3 error is driven by actuator saturation or oscillation, the smoothness metric will reveal it before tracking error does.

## Limitations & Caveats

- The OCS formulation requires solving an SDP at each planning step, which may be too slow for real-time replanning (their system pre-computes for a known trajectory). For our fixed racing line this is acceptable — we pre-compute gains offline.
- Results are demonstrated at 12 m/s on a figure-8, not on a multi-gate racing track. The dynamics at racing speeds (potentially 15-20 m/s) with sharp turns may stress the linear double-integrator approximation.
- The 1-2 cm accuracy sacrifice vs LQR may be significant at our error margins — we have only 2.4 cm headroom at gate-3. However, the smoothness improvement could enable faster speeds that net out positively on race time.
- External vision at 100 Hz is a luxury; our system uses onboard estimation which has larger uncertainty. The covariance bounds would need to be adjusted accordingly.

## Key Parameters/Constants

| Parameter | Value | Context |
|-----------|-------|---------|
| Sampling time (DeltaT) | 0.01 s (100 Hz) | Control loop rate |
| State penalty Q | I_6 | Identity weighting |
| Control penalty R | I_3 | Identity weighting |
| Position bound delta_x | 0.025 m | 3-sigma chance constraint |
| Control bound delta_u | 10 N | 3-sigma chance constraint |
| Disturbance matrix D | blkdiag(0.01 I_3, 0.1 I_3) | Process noise scaling |
| NN architecture | {8, 20, 20, 3} | ReLU, drag correction |
| Drone mass | 680 g | 5-inch racing quad |
| Max thrust | 39 N | Platform limit |
| Max speed tested | 12 m/s | Figure-8 trajectory |
| Max acceleration | 22 m/s^2 | Figure-8 trajectory |
| Trajectory steps N | 540 | 5.4 second horizon |
