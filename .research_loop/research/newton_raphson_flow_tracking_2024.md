# Newton-Raphson Flow for Aggressive Quadrotor Tracking Control
- **URL**: https://arxiv.org/abs/2408.11197
- **Year**: 2024
- **Authors**: Evanns Morales-Cuadrado, Christian Llanes, Yorai Wardi, Samuel Coogan (Georgia Institute of Technology)

## Key Contribution

This paper presents the first hardware validation of the Newton-Raphson flow tracking controller on a real, open-loop unstable, underactuated nonlinear plant — specifically a 6-DOF quadrotor. The Newton-Raphson flow controller is an integrator-type scheme that continuously drives to zero the error between a predicted future output and the reference trajectory at that future time. The key practical novelty is pairing this controller with a simple closed-form predictor derived from linearized hover dynamics, requiring knowledge of only vehicle mass and gravitational constant — comparable to what PID tuning needs. The controller is shown to outperform the well-tuned cascaded PID architecture native to PX4 Autopilot across all tested trajectories, sometimes dramatically.

## Technical Approach

**Core idea.** The standard Newton-Raphson iteration for root-finding, reformulated as a continuous-time ODE, yields the update law:

    du/dt = alpha * (d_rho/du)^{-1} * [r(t+T) - rho(x(t), u(t))]        (Eq. 7)

where `rho(x, u)` is a predictor of the system output T seconds into the future, `r(t+T)` is the reference at that future time, and `alpha` is a scalar "speedup" gain. The controller integrates this ODE forward, so the input `u(t)` evolves continuously rather than being set directly.

**Predictor design.** The predictor uses linearized hover dynamics with a zero-order hold on the input over the lookahead window `[t, t+T]`:

    x_hat(t+T) = A_tilde * x(t) + B_tilde * u(t)
    y_hat(t+T) = C * x_hat(t+T)

where `A_tilde = e^{AT}` and `B_tilde = integral_0^T e^{A(T-tau)} B d_tau` are precomputed offline. The state is 9-dimensional (position, velocity, Euler angles); the output is 4-dimensional (x, y, z, yaw). This closed-form predictor requires no online ODE integration and has negligible compute cost.

**Lookahead time T.** The paper uses **T = 0.8 seconds** in all hardware experiments. This is a notably long lookahead — an order of magnitude larger than the 50 ms used in our system. The authors note that the predictor does not need to be accurate; what matters is that it is continuously differentiable and computable. The error bound (Eq. 8) shows that prediction error `nu_1` enters additively and independently of `alpha`, while trajectory rate `nu_2` and disturbances are attenuated by `1/alpha`. This means an imprecise predictor is tolerable as long as `alpha` is large.

**Speedup gain alpha.** Set to **alpha = 30** in all experiments. This gain plays two roles: (1) it may stabilize the closed-loop system, and (2) it reduces the asymptotic tracking error upper bound proportionally to `1/alpha` for terms related to trajectory speed and disturbances.

**Overshoot prevention via Integral Control Barrier Functions (I-CBFs).** Because the controller is integrator-type, it is susceptible to integrator wind-up causing large transient overshoots and actuator saturation. The paper addresses this by augmenting the update law with a minimal intervention term `eta(t)` derived from an I-CBF. The barrier function `b(x, u)` encodes angular rate limits (RateMin = -0.8 rad/s, RateMax = +0.8 rad/s for roll/pitch/yaw rates). The I-CBF computes the smallest correction `eta` needed to keep the commanded rates within these bounds, intervening only when the nominal update would violate them. This produces smooth rate limiting (unlike hard clamping) while preserving the asymptotic tracking behavior of the nominal controller. The CBF parameter is `gamma = 1.0`.

**Control loop rate.** Commands published at **100 Hz**. Average computation time per step: **7.1e-5 ± 3.0e-5 seconds** (well under 10 ms, running on a Raspberry Pi 4).

## Results

Hardware experiments on a Holybro x500v2 (1.69 kg, 0.5 m diagonal) with OptiTrack motion capture for position feedback. Tested on five trajectories (vertical circle, horizontal circle, horizontal lemniscate, vertical short/tall lemniscate). Period: 3.14 s for circles, 6.28 s for lemniscates.

**RMSE comparison (meters):**

| Trajectory | Newton-Raphson | PX4 Baseline |
|---|---|---|
| Vertical Circle | 0.051 | 0.266 |
| Horizontal Circle | 0.168 | 0.611 |
| Horizontal Lemniscate | 0.155 | 0.188 |
| Vertical Short Lemniscate | 0.045 | 0.148 |
| Vertical Tall Lemniscate | 0.097 | 0.225 |

At 2x speed (period 3.14 s for lemniscates), Newton-Raphson degrades gracefully (0.105–0.127 m RMSE) while PX4 degrades severely (0.272–0.287 m RMSE). The improvement factor ranges from 2x to 4x depending on trajectory.

## Relevance to Our System

Our current system uses a PD+feedforward (geometric) controller with a **50 ms lookahead** that is causing overshoot at gate-2 (straight-to-turn transition). This paper is directly relevant in several ways:

1. **The lookahead direction problem.** Our 50 ms lookahead projects the reference forward along the trajectory. At a straight-to-turn transition, this lookahead point is still in the straight segment when the drone begins the turn, causing a late correction response and overshoot. The Newton-Raphson paper uses T = 0.8 s but with a linearized predictor — the key insight is that the predictor must point at `r(t+T)` on the *actual reference*, not just locally. If we could smoothly ramp down the effective lookahead before turn entry, we could reduce the lag.

2. **Integrator-type update vs. direct output.** Our PD controller directly outputs a force/torque command. The NR-flow controller instead integrates `du/dt` — this naturally limits the rate of change of the command, which is analogous to rate limiting. The I-CBF then provides a principled, smooth clamp. For our gate-2 overshoot, the abrupt feedforward change at the transition is the likely culprit; an integrator-type update would inherently smooth this.

3. **Overshoot prevention via I-CBF.** The I-CBF framework is directly applicable to our problem. We could add an I-CBF on angular rate or lateral acceleration to prevent the drone from responding too aggressively to a sudden reference direction change. This is more principled than reducing gains or shortening lookahead.

4. **Predictor design.** The linearized hover predictor (mass + gravity only) achieves excellent results despite being crude. This suggests we do not need a complex nonlinear predictor for our feedforward term. Our current feedforward could be restructured as a NR-flow update step with the same cheap predictor.

## Actionable Takeaways

1. **Shorten lookahead dynamically near turns.** The NR-flow result shows that a long T (0.8 s) works well on smooth trajectories but implicitly relies on the predictor being directionally correct. For our 50 ms fixed lookahead, reduce it adaptively as a function of upcoming curvature (e.g., scale T by `min(1, kappa_threshold / kappa)` where kappa is trajectory curvature ahead). This directly targets the gate-2 overshoot.

2. **Add I-CBF-style smooth rate limiting to the feedforward.** Instead of hard-clamping the feedforward thrust/torque at transitions, apply the minimal-intervention formula (Eq. 20) to keep angular rate commands within a safe bound. This requires only the current rate, the nominal update, and the barrier gradient — all available in `mpc_tracker.py`.

3. **Try the full NR-flow update law as a drop-in replacement for the PD tracker.** The closed-form predictor requires only mass (known) and g (known). The update law (Eq. 7) with alpha=30, T=0.1–0.2 s (shorter than the paper's 0.8 s to match our race speed) and the I-CBF with gamma=1.0 could replace the geometric controller in `control/mpc_tracker.py`. Expected improvement: 2-4x RMSE reduction based on paper results.

4. **Use the asymptotic error bound (Eq. 8) to choose alpha.** The bound is `nu_1 + nu_2/alpha`. With our fast trajectories, `nu_2 = ||dr/dt||` is large. Setting alpha > 50 should substantially reduce the error contribution from trajectory velocity.

## Limitations & Caveats

- **Trajectories are slow by racing standards.** The paper's fastest test uses a 3.14 s period lemniscate at roughly 1–2 m/s peak velocity. AI Grand Prix gates are traversed at 10–15 m/s. The linearized hover predictor will be increasingly inaccurate at high speed and high angle-of-attack, degrading `nu_1`. Whether the controller remains competitive at race speeds is unknown.
- **T = 0.8 s is far too long at race speeds.** At 12 m/s, 0.8 s corresponds to nearly 10 m of lookahead — the drone would be reacting to a gate it has not yet reached. A much shorter T (0.05–0.1 s) would be needed, which likely degrades predictor accuracy less but may require retuning alpha.
- **No gate-passing or trajectory switching.** The paper tests smooth periodic trajectories, not the discrete waypoint-to-waypoint structure of drone racing. Behavior at segment transitions (where `r(t+T)` jumps to a new polynomial) is untested.
- **Single hardware platform.** Experiments use one airframe at moderate speeds. Generalization to our specific sim dynamics needs separate validation.
- **I-CBF requires the full state vector and control input.** In our system, the latency-compensated state from `state_predictor.py` would need to be threaded into the barrier computation.

## Key Parameters / Constants

| Parameter | Value | Role |
|---|---|---|
| Lookahead time T | 0.8 s | Prediction horizon |
| Speedup alpha | 30 | Asymptotic error reduction |
| CBF parameter gamma | 1.0 | I-CBF convergence rate |
| RateMin / RateMax | ±0.8 rad/s | Angular rate safety limits |
| Control frequency | 100 Hz | Command update rate |
| Average compute time | 7.1e-5 s | Per-step latency on RPi 4 |
| State dimension | 9 (pos, vel, Euler) | Predictor state space |
| Output dimension | 4 (x, y, z, yaw) | Controlled outputs |
| Quad mass (x500v2) | 1.69 kg | Only dynamic parameter needed |
