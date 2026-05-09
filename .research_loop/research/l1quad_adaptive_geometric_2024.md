# ℒ₁Quad: ℒ₁ Adaptive Augmentation of Geometric Control for Agile Quadrotors

- **URL**: https://arxiv.org/abs/2302.07208
- **Authors**: Zhuohuan Wu, Sheng Cheng, Pan Zhao, Aditya Gahlawat, Kasey A. Ackerman, Arun Lakshmanan, Chengyu Yang, Jiahao Yu, Naira Hovakimyan
- **Year**: 2025 (arXiv preprint 2023, published in IEEE TCST Vol. 33, No. 2, pp. 597–612, March 2025)
- **Venue**: IEEE Transactions on Control Systems Technology (DOI: 10.1109/TCST.2024.3521182)

---

## Key Contribution

ℒ₁Quad presents a control architecture that augments a standard geometric controller (operating on the SE(3) manifold) with an ℒ₁ adaptive compensation layer, providing formal guarantees of uniformly bounded transient response in the presence of nonlinear, time- and state-dependent uncertainties. The framework handles three classes of disturbances simultaneously — external forces (wind, downwash, ground effect), actuation uncertainties (damaged propellers, voltage drop), and model mismatch (mass/inertia errors, unmodeled drag) — without requiring parametric structure or prior knowledge of those disturbances.

The key theoretical novelty is being the first work to extend ℒ₁ adaptive control to nonlinear reference systems on the full SE(3) manifold. Prior ℒ₁ formulations for quadrotors were restricted to SO(3) (rotation only) or relied on linearized/Euler-angle dynamics. By operating on SE(3) with a nonlinear reference model, ℒ₁Quad handles the coupled translational and rotational uncertainties that arise in aggressive flight, and provides a tube-based performance certificate: the trajectory tracks within a computable radius ρ for all time, with ρ tunable through filter bandwidth and sampling rate. Experiments on a physical quadrotor across 11 uncertainty scenarios show roughly five-times smaller RMSE compared to the baseline geometric controller alone.

---

## Technical Approach

### Baseline Geometric Controller

The foundation is the geometric tracking controller of Lee et al. (2010), extended to trajectory tracking on SE(3). The desired force vector is:

```
F_d = -K_p * e_p - K_v * e_v - m*g*e_3 + m*p̈_d
```

where `e_p = p - p_d` and `e_v = v - v_d` are position and velocity errors, `K_p`, `K_v ∈ ℝ³ˣ³` are positive-definite gain matrices, and `m*p̈_d` is the feedforward acceleration term.

The desired moment in the body frame is:

```
M_b = -K_R * e_R - K_Ω * e_Ω + Ω × J·Ω - J(Ω̂ R^T R_d Ω_d - R^T R_d Ω̇_d)
```

where `e_R` is the rotation error on SO(3), `e_Ω = Ω - R^T R_d Ω_d` is the angular velocity error, `J` is the inertia tensor, and `K_R`, `K_Ω` are attitude gains.

### Uncertainty Model

The quadrotor dynamics are augmented with two classes of uncertainty:

**Matched uncertainties** `σ_m(t, x)` enter through the same channels as control inputs — specifically as perturbations along the body z-axis (thrust channel) and in the body-frame moment channels. Sources include: unmodeled aerodynamic drag (both linear and nonlinear), mass uncertainty, propeller damage causing asymmetric thrust, and battery sag affecting motor thrust curves.

**Unmatched uncertainties** `σ_um(t, x)` enter perpendicular to control directions — lateral forces in the body x-y plane. Sources include: side wind, aerodynamic coupling effects. These cannot be directly cancelled by thrust/moment commands.

Both classes are assumed to be continuous, bounded, and Lipschitz continuous with bounded partial derivatives (Assumption 1 in the paper). This is satisfied by aerodynamic drag forces (which are smooth functions of velocity).

### ℒ₁ Adaptive Law

The ℒ₁ architecture decouples estimation from control, which is the key property enabling arbitrarily fast adaptation without sacrificing robustness.

**State predictor** — propagates a shadow model of the dynamics using current uncertainty estimates:

```
ẑ̇ = f(ẑ) + B(R)(u_b + σ̂_m)
```

The predictor state `ẑ` tracks the true state `z` with an error driven by the uncertainty estimation error `σ̃_m = σ_m - σ̂_m`.

**Adaptation law** — piecewise-constant update at sampling period T:

```
σ̂_m(t) = σ̂_m(kT),   for t ∈ [kT, (k+1)T)
```

At each sample boundary, the update is computed to minimize the state prediction error over the prior interval. This piecewise-constant structure ensures numerical stability and allows the adaptation rate to be increased (smaller T) without bound while preserving the closed-loop stability margin.

**Control law** — the ℒ₁ augmentation directly cancels estimated matched uncertainty:

```
u_ad = -C(s) * σ̂_m
```

where `C(s)` is a strictly proper low-pass filter with bandwidth `ω_c`. The total force/moment command is `u = u_geo + u_ad`, where `u_geo` is the baseline geometric controller output. The low-pass filter is critical: it attenuates high-frequency noise in `σ̂_m` while still passing meaningful disturbance estimates. The bandwidth `ω_c` (typically 5–20 rad/s in practice) directly controls the tradeoff: higher bandwidth → faster adaptation → smaller tube radius ρ, but more sensitivity to measurement noise.

### Performance Guarantee (Theorem 1)

The key theoretical result establishes that if:
1. The nominal (undisturbed) system is exponentially stable with Lyapunov decay rate β (Proposition 1)
2. Matched uncertainty satisfies the Lipschitz bounds of Assumption 1
3. Unmatched uncertainty magnitude satisfies: `Δσ_um < (γ̲ρ² - V₀) / (c₄ρ)`

Then the full state trajectory satisfies `d(x(t), x_d(t)) ≤ ρ` for all `t ≥ 0`, where ρ is computable from the filter bandwidth `ω_c`, sampling period T, and Lyapunov constants.

The Lyapunov function used to establish Proposition 1 is:

```
V = ½K_p‖e_p‖² + ½e_Ω^T J e_Ω + ½m‖e_v‖² + c₁ e_p^T e_v + c₂ e_R^T e_Ω + K_R Ψ(R, R_d)
```

where `Ψ(R, R_d)` is the attitude error function on SO(3), and `c₁`, `c₂ > 0` are chosen to make the cross-terms bounded by a positive-definite quadratic.

### Implementation Details

The system runs at **400 Hz** on a Pixhawk 4 mini with modified ArduPilot firmware. Position feedback comes from a Vicon motion-capture system at 50 Hz, fused with IMU at 400 Hz using ArduPilot's built-in EKF. The quadrotor weighs **0.63 kg** with **0.22 m** motor-to-motor diagonal, using T-Motor F60 2550KV motors with T5150 tri-blade propellers.

Key tunable parameters exposed in the ArduPilot parameter tree:
- `ASV`, `ASOMEGA` — state predictor gains
- `CTOFFQ1THRUST`, `CTOFFQ1MOMENT`, `CTOFFQ2MOMENT` — filter cutoff frequencies (ω_c per channel)
- `L1ENABLE` — on/off toggle for the adaptive augmentation
- Geometric gains: `GEOCTRL_KPX/Y/Z`, `GEOCTRL_KVX/Y/Z`, `GEOCTRL_KRX/Y/Z`, `GEOCTRL_KOX/Y/Z`

Open-source implementations exist for both ArduPilot (Pixhawk) and Crazyflie platforms.

---

## Results

The framework was validated through experiments on a physical quadrotor across **11 uncertainty categories**:

1. Injected artificial disturbance forces
2. Sloshing payload (liquid-filled container adding dynamic, unmodeled forces)
3. Chipped propeller (asymmetric thrust/torque imbalance)
4. Mixed propeller types (different lift coefficients per motor)
5. Battery voltage drop during flight (decreasing motor efficiency)
6. Ground effect (altered induced velocity near surfaces)
7. Downwash from overhead obstacles
8. Tunnel navigation (confined aerodynamic environment)
9. Hanging off-center weights (shifted center of mass)
10–11. Benchmark scenarios with added and slung masses

**Key quantitative result**: The ℒ₁ augmentation achieves, on average, **five times smaller trajectory tracking error** (RMSE) compared to the baseline geometric controller alone without the adaptive layer. This figure comes from the predecessor paper (arXiv:2109.06998) which established the same architecture on a simpler system; the full ℒ₁Quad paper extends this to the SE(3) formulation and validates across all 11 scenarios.

The controller uses a **single parameter set across all 11 uncertainty types and all tested trajectories** without retuning. Baselines compared against include: the standalone geometric controller (Lee et al.), and at least two other state-of-the-art adaptive/robust controllers from the literature (references [15], [47], [18] in the paper). ℒ₁Quad "significantly outperforms" all baselines with "consistently small tracking errors."

Trajectories tested include circular paths at speeds up to 2 m/s and more aggressive lemniscate/figure-eight trajectories. The hardware platform and motion-capture feedback system set an upper bound on trajectory aggressiveness, but the architecture is not speed-limited in principle.

---

## Relevance to Our System

Our system uses a geometric controller (Lee et al. SE(3)) with partial feedforward at weight=0.4. Full feedforward (weight=1.0) causes overshoot because our kinematic simulator applies linear aerodynamic drag (coefficient 0.5) that the controller's dynamics model does not account for. The controller commands thrust to accelerate toward a waypoint, but the drag simultaneously decelerates the vehicle — the controller sees this as a position error and adds more thrust, overshooting when the drag then decelerates it on the other side.

ℒ₁Quad addresses exactly this failure mode through the matched uncertainty channel:

**Drag as a matched uncertainty**: Linear aerodynamic drag produces a force `F_drag = -c_d * v` along the velocity vector. In body frame, the dominant component is along the body z-axis (when flying forward) — exactly the matched uncertainty channel that `σ̂_m` estimates. The state predictor will observe that the actual deceleration exceeds what the nominal model predicts, estimate the residual as `σ̂_m`, and apply `u_ad = -C(s) * σ̂_m` to pre-cancel it. This allows the baseline feedforward to run at or near 1.0 without overshoot.

**Specific benefit for our pipeline**:
- With drag estimated and cancelled by ℒ₁, the geometric controller can use **full feedforward** (weight → 1.0) because the model mismatch is compensated online.
- Currently our feedforward is throttled to 0.4 to avoid overshoot — meaning 60% of the ideal trajectory-tracking acceleration is discarded. ℒ₁ compensation could recover most of that, substantially reducing tracking error at high speeds.
- Per-gate tracking errors in our system are largest at gates 2–4 (high-curvature turns). These are precisely where drag forces deviate most from the nominal model (velocity changes rapidly), and where ℒ₁ adaptation would provide the largest benefit.
- The ℒ₁ estimate also captures state-dependent drag variations (e.g., higher drag at peak speed), which our static drag coefficient cannot model.

**Unmatched uncertainty (lateral forces)**: In our sim, drag is purely linear, so it acts along the velocity vector. In banked turns, the drag vector has a lateral component in the body x-y plane — this falls in the unmatched channel and cannot be directly cancelled. However, the stability guarantee still holds as long as this lateral drag is bounded (it is, since it scales with velocity which is bounded by our trajectory).

**Implementation in our stack**: Our `control/mpc_tracker.py` computes the geometric controller output. An ℒ₁ augmentation layer could sit between the tracker output and the actuator command: (1) maintain a shadow state predictor using the tracker's force/moment commands plus the current `σ̂_m` estimate, (2) update `σ̂_m` at each control loop step using the predictor error, (3) apply `u_ad = -σ̂_m` (with simple low-pass filter at ~10–15 rad/s) to the force command before converting to thrust + attitude. This requires no changes to the trajectory planner or EKF.

---

## Actionable Takeaways

1. **Implement a minimal ℒ₁ augmentation layer in `control/mpc_tracker.py`**: Add a state predictor that tracks velocity using the current force command plus `σ̂_m`, then update `σ̂_m = -(1/m)(p̈_measured - p̈_predicted)` at each timestep. Apply `F_ad = -LPF(σ̂_m)` to the thrust vector. This is ~30 lines of code.

2. **Increase feedforward weight from 0.4 toward 1.0 once ℒ₁ is active**: Start at 0.6, validate no overshoot, then step to 0.8 and 1.0. The ℒ₁ layer will compensate for drag mismatch that currently makes full feedforward unstable.

3. **Tune the low-pass filter bandwidth ω_c carefully**: Start at 5 rad/s (conservative). If the sim noise is low enough, push to 10–15 rad/s for faster drag compensation. Higher bandwidth gives smaller tracking error tube but amplifies measurement noise.

4. **Use piecewise-constant adaptation (update once per control tick, hold until next tick)**: This is numerically stable and exactly what the theory requires. Do not use continuous integration which can drift.

5. **Initialize `σ̂_m = 0` at trajectory start**: The ℒ₁ layer will converge to the correct drag estimate within a few timesteps (roughly 1/ω_c seconds), so error at the first gate may be slightly elevated but subsequent gates benefit fully.

6. **Keep the geometric controller gains unchanged**: ℒ₁ augmentation is designed to sit on top of an existing, tuned controller without requiring gain retuning. The baseline stability (Proposition 1) is preserved exactly because ℒ₁ only adds a filtered signal.

7. **Log `σ̂_m` over time**: In simulation, the estimated drag should converge to approximately `c_d * v ≈ 0.5 * v`. If it does, this validates the ℒ₁ estimator is correctly capturing the drag coefficient, and you can consider feeding this forward into the trajectory planner as a correction to the drag model.

8. **Consider using ℒ₁ to enable velocity-profile aggression**: Once drag is compensated, the speed profile in `planning/racing_line.py` can push to higher peak velocities through gates, since the controller can now track aggressive trajectories more faithfully.

---

## Limitations & Caveats

- **Matched uncertainty only**: The ℒ₁ layer directly cancels forces/moments along the thrust axis. Lateral aerodynamic effects (body-x, body-y drag) are unmatched uncertainties and cannot be cancelled — only bounded. For our sim's linear drag model, this is a secondary effect, but for real hardware with significant lateral drag it may matter.

- **Requires state derivative estimates**: The adaptation law needs `p̈_measured` (linear acceleration) or a proxy. In our sim, this can be computed from velocity differences across timesteps, but will be noisy. An IMU would provide this directly on hardware.

- **Lipschitz assumption on uncertainties**: The theory requires drag forces to be Lipschitz in state. Linear drag trivially satisfies this. Quadratic drag also satisfies it for bounded velocity. Very abrupt disturbances (e.g., a gust impulse) may violate the bounds temporarily, causing transient error spikes before the estimator converges.

- **No motor dynamics modeled**: The paper explicitly notes that propeller aerodynamic effects and motor dynamics are not in the baseline model. Our sim also ignores these, so this limitation is consistent — but it means ℒ₁ estimates all unmodeled effects lumped together, which may reduce interpretability.

- **Bandwidth-noise tradeoff at high loop rates**: At 400 Hz with a 5 rad/s LPF, the filter introduces ~200 ms lag in disturbance rejection. At our 100+ Hz loop rate, tuning ω_c to 15 rad/s (~65 ms lag) is probably the right target, but should be validated against sim noise levels.

- **Gate-to-gate variation**: The ℒ₁ estimate `σ̂_m` adapts to the current operating point. At gate transitions where speed changes rapidly, the estimate lags the true drag by roughly 1/ω_c seconds. This may cause transient errors at gates where the drone decelerates sharply.

- **Experimental speeds were modest (≤2 m/s in cited platform)**: Our racing system targets much higher speeds. At 10+ m/s, drag forces are substantially larger (proportional to velocity for linear drag). The fundamental approach scales, but larger `σ̂_m` magnitudes mean more aggressive filter bandwidth is needed to track the faster-changing drag, which in turn requires better state estimation noise floors.

---

## Key Parameters / Constants

| Parameter | Symbol | Typical Value | Notes |
|-----------|--------|---------------|-------|
| LPF bandwidth | ω_c | 5–20 rad/s | Core tuning knob; higher = faster adaptation but more noise sensitivity |
| Adaptation sampling period | T | 2.5–10 ms | Inverse of adaptation rate; set to control loop period |
| Tracking tube radius | ρ | function of ω_c, T | Decreases with larger ω_c and smaller T |
| State predictor gain | ASV | hardware-dependent | Scales predictor error correction; must keep predictor stable |
| Position gains | K_p | per-axis | Same as baseline geometric controller |
| Velocity gains | K_v | per-axis | Same as baseline geometric controller |
| Rotation gains | K_R, K_Ω | per-axis | Same as baseline geometric controller |
| Lyapunov cross-coupling | c₁, c₂ | > 0, small | Must satisfy W > 0 for Proposition 1 |
| Hardware mass | m | 0.63 kg (reference platform) | Scales force uncertainty magnitudes |
| Control loop rate | — | 400 Hz | Both geometric + ℒ₁ run at same rate |
| Unmatched uncertainty bound | Δσ_um | < (γ̲ρ² - V₀)/(c₄ρ) | Must be verified for stability guarantee |

The five-times-smaller tracking error result from the predecessor paper (arXiv:2109.06998) provides a useful expected improvement bound. In our system, where drag mismatch is the dominant error source, the improvement could be similar or larger since ℒ₁ directly targets the drag mismatch mechanism.
