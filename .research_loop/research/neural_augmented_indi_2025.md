# Neural-Augmented INDI for Quadrotors with Payload Adaptation
- **URL**: https://arxiv.org/abs/2503.09441
- **Authors**: Eckart Cobo-Briesewitz, Khaled Wahba, Wolfgang Hönig
- **Year**: 2025
- **Venue**: IROS 2025 (submitted March 12, 2025)

---

## Key Contribution

This paper addresses a fundamental limitation of quadrotor flight controllers: the inability to accurately account for *residual forces* — forces that the nominal model does not capture, including aerodynamic effects, disturbances, payload interactions, and motor–rotor coupling. Traditional controllers ignore these terms entirely; INDI (Incremental Nonlinear Dynamic Inversion) estimates them from high-frequency sensor differences, but is sensitive to sensor noise. Prior learning-based work can predict residual forces but requires separate sensors or large training datasets.

The key contribution is threefold: (1) the authors show that a neural network trained on INDI outputs can reproduce smoother, less noisy residual force estimates *without* requiring the specialized sensor measurements that INDI depends on; (2) they propose a hybrid method that fuses neural network predictions with INDI's sensor-difference estimate, yielding better performance than either alone; and (3) they extend both approaches to handle slung-type payloads, where the swinging mass introduces highly nonlinear, time-varying disturbance forces that are notoriously difficult to model analytically.

The paper is part of a broader 2024–2025 trend of hybrid physics-learning controllers (NeuroBEM, Neural-Fly, DATT) that reduce data hunger by learning only the residual — the delta from a principled physics model — rather than the full dynamics. This is a more sample-efficient, more generalizable, and more interpretable approach than end-to-end learned controllers.

---

## Technical Approach

### INDI Background

INDI exploits the time-scale separation between the fast dynamics driven by actuator inputs and the slower state evolution. Starting from the full nonlinear rotational dynamics:

```
Ω̇ = J⁻¹(M - Ω × JΩ + f_residual)
```

where `Ω` is the angular velocity vector `[p, q, r]^T`, `J` is the inertia matrix, `M` is the actuator moment vector, and `f_residual` captures everything the model misses. Applying a first-order Taylor expansion around the current operating point `(Ω₀, u₀)` and invoking the time-scale separation assumption (`Ω - Ω₀ ≈ 0` over one control timestep), the incremental control law becomes:

```
Ω̇ = Ω̇₀ + G · Δu
```

where `Ω̇₀` is the measured angular acceleration (obtained by differentiating the gyroscope signal), `G = ∂f/∂u|_{u₀}` is the control effectiveness matrix (a 3×4 Jacobian for a quadrotor), and `Δu` is the increment in rotor commands to be solved for. The key insight is that `Ω̇₀` implicitly contains the residual force — the sensor measurement already reflects whatever disturbance the model missed. INDI's inverse then cancels it out without needing to know what caused it.

The resulting control law to achieve a desired angular acceleration command `Ω̇_des` is:

```
Δu = G⁺ · (Ω̇_des - Ω̇₀)
```

where `G⁺` is the pseudo-inverse of the control effectiveness matrix. The INDI inner-loop typically runs at 500–1000 Hz to ensure the Taylor linearization remains valid (changes between timesteps stay small).

### The Noise Problem

INDI's strength is also its weakness: `Ω̇₀` is estimated by numerical differentiation of noisy gyroscope measurements, introducing high-frequency noise that can destabilize the controller if gains are too high. Filtering reduces noise but adds latency; the delayed angular acceleration estimate causes closed-loop oscillations. This is a well-documented failure mode: filter delay compensation is non-optional for INDI to work correctly.

### Neural Network Residual Prediction

Rather than estimating `f_residual` from sensor differences, the paper trains a neural network to map from state and input history to a predicted residual force:

```
f̂_residual = NN(Ω, v, u_history; θ)
```

The network learns by regressing on the INDI estimate of residual forces from a flight dataset: the training target is the INDI output (which already implicitly encodes residual forces), making the learning problem well-posed even without ground-truth force measurements. By targeting the INDI signal rather than raw acceleration residuals, the network learns a smoother, lower-noise version of the same quantity.

The architecture is a compact neural network (likely MLP with recurrent or temporal windowing given the need to capture slung-payload dynamics), trained offline. Crucially, it does *not* require any extra sensors beyond what a standard quadrotor carries (IMU + motor state), making it deployable on existing hardware.

### Hybrid Fusion

The paper's most novel contribution is the combination scheme, which blends the NN prediction and the INDI sensor estimate:

```
f_combined = α · f̂_NN + (1 - α) · f̂_INDI
```

The intuition is complementary strengths: the NN provides smooth, low-noise predictions that are confident in regime where training data is plentiful; INDI provides accurate, real-time estimation in novel or out-of-distribution scenarios where the network may drift. The blend weight `α` can be static or adaptive depending on uncertainty estimates.

### Payload Adaptation

For slung-type payloads (mass on a cable), the system dynamics gain an additional coupling force:

```
F_cable = T_cable · ê_cable
```

where `T_cable` is cable tension and `ê_cable` is the unit vector from attachment point to payload. The cable creates a pendulum whose natural frequency and damping depend on cable length and payload mass — both of which are typically unknown at flight time. The authors adapt both the NN-only and the hybrid approach to this scenario, likely by augmenting the input features with cable angle/angular rate (if an angle sensor is available) or by relying on the residual estimator to implicitly capture the coupling force.

---

## Results

The paper's conclusions are directional rather than reporting precise ablation numbers in the available abstract. The ordering of results is:

1. **Nominal quadrotor (no payload):** INDI alone already outperforms a model-based controller. The NN-only approach matches or slightly outperforms INDI, with notably smoother force estimates. The hybrid approach outperforms both.

2. **Slung payload quadrotor:** The advantage of NN and hybrid approaches widens considerably. Pure model-based control degrades substantially under unknown payload dynamics; INDI partially recovers; the NN-augmented hybrid achieves the best tracking.

For context, related work (which this paper builds on) has established quantitative baselines: INDI inner-loop with precise angular acceleration feedback achieves **4.0 cm RMS position tracking error** at 8.2 m/s and 2g acceleration (Tal & Karaman, 2018). Adaptive INDI under unknown payloads up to 60% of vehicle mass achieves **>90% tracking error reduction** over non-adaptive NMPC. Neural-Fly achieves precise tracking at wind speeds up to 43.6 km/h with substantially lower error than state-of-the-art adaptive controllers.

The hybrid method in this paper is expected to perform comparably to these benchmarks on nominal flight and to extend gracefully to the payload case, where pure INDI degrades due to the swing dynamics being outside the linear Taylor approximation assumption.

---

## Relevance to Our System

Our current bottleneck is specific and well-diagnosed: gates 7–12 (helix section) average 0.64 m tracking error vs 0.31 m for straight gates 1–6. The `GeometricTracker` uses a PD law with partial feedforward (`feedforward_accel=0.4`) to compute `accel_des`. The kinematic sim then applies this as:

```
accel = accel_des - drag * vel
```

clamped to `max_accel=15`. There is no inner attitude loop — the sim bypasses rotational dynamics and takes desired acceleration directly.

**Direct relevance:**

1. **Residual drag is the key unmodeled force in our setting.** The helix turns involve high centripetal acceleration (curved path + high speed) combined with aerodynamic drag that is state-dependent and nonlinearly coupled to velocity direction. Our sim uses a scalar drag model (`0.5 * vel`), which is isotropic and doesn't capture the nonlinear thrust-drag interaction in banked turns. An INDI-style residual estimator — or a neural network trained to predict the error between commanded and achieved acceleration — would directly address this.

2. **The NN residual approach maps naturally to our kinematic sim.** In our sim, `f_residual = accel_achieved - accel_des` is directly computable from the simulation state at each timestep. This is ideal training data for a network that learns to predict `f_residual` from `(pos, vel, accel_des, heading)`.

3. **No attitude loop means reduced complexity.** The paper's INDI formulation targets the angular acceleration loop. Since our sim bypasses attitude entirely, we can apply the conceptual framework one level higher: learn the *translational* residual force (not the rotational), and add it as a correction term to `accel_des`:

   ```
   accel_corrected = accel_des + NN(vel, accel_des, heading)
   ```

4. **The turn-tracking deficit is structurally similar to the payload problem.** Slung-payload dynamics introduce state-dependent forces the nominal model misses; so does banked-turn centripetal dynamics with imperfect feedforward. The paper's demonstration that residual learning substantially reduces error under unknown payload dynamics is a direct analogy.

5. **Smoothness advantage is relevant.** INDI residual estimates are noisy because they rely on sensor differences. Our sim equivalent — computing `accel_residual = (actual_accel - desired_accel)` from PyBullet ground truth — would also be noisy at real-world rates. The paper's demonstration that a NN learns a smoother version motivates training a small regressor on logged sim data.

**Indirect relevance:**

The hybrid INDI + NN architecture demonstrates that physics-grounded residuals (even noisy ones) provide useful inductive bias for learning, reducing data requirements. This suggests that even a small dataset of turns flown with the current controller could train a useful corrector.

---

## Actionable Takeaways

1. **Implement a translational residual force estimator.** Log `(vel, accel_des, accel_achieved)` tuples during benchmark runs. Compute `f_residual = accel_achieved - accel_des` to build a dataset. This is the kinematic-sim analog of INDI's sensor-difference estimation.

2. **Train a lightweight MLP residual corrector.** Use the logged dataset to train a small MLP (2–3 layers, 32–64 hidden units) mapping `(vel_x, vel_y, vel_z, accel_des_x, accel_des_y, accel_des_z, heading)` to `(f_res_x, f_res_y, f_res_z)`. This is the neural-INDI analog for our kinematic sim.

3. **Apply the correction at command time.** Modify `GeometricTracker.track()` to add the neural residual to `accel_des` before computing thrust/attitude commands:
   ```python
   accel_des += self.residual_model.predict(vel, accel_des, yaw)
   ```
   This keeps the PD structure intact and is a minimal, reversible change.

4. **Focus the corrector on turns (gates 7–12).** Use conditioning variables like centripetal acceleration magnitude (`|vel × (d(vel)/dt)|`) to weight the corrector more heavily in high-curvature segments. The paper's payload adaptation demonstrates that residual predictors can be conditioned on regime indicators.

5. **Adopt the hybrid blend approach as a safety mechanism.** Weight the NN correction by a confidence score (e.g., how similar the current `(vel, accel_des)` is to training distribution). Near out-of-distribution points, fall back to pure PD. This mirrors the INDI + NN blend described in the paper.

6. **Increase feedforward weight selectively in turns.** The paper's NN learns to smooth INDI estimates, which are most valuable when model error is largest. In our system, `feedforward_accel` could be increased beyond 0.4 in high-curvature segments without risking overshoot in straight sections if the residual corrector compensates.

7. **Explore data-efficient training with a small simulator dataset.** The paper emphasizes that learning only residuals (not full dynamics) requires far less data. A 30-second benchmark run at ~100 Hz yields ~3000 tuples — potentially sufficient for a 2-layer MLP to capture the dominant velocity-dependent drag patterns in the helix.

8. **Evaluate model generalization across trajectories.** Like the paper's payload generalization tests, check whether a corrector trained on one benchmark run transfers to different racing lines or gate sequences without retraining.

---

## Limitations & Caveats

1. **No attitude loop in our sim.** INDI's core contribution is to the angular rate / angular acceleration inner loop. Since our kinematic sim bypasses attitude and uses desired acceleration directly, the exact INDI formulation doesn't apply — we must adapt it to the translational acceleration level.

2. **Kinematic sim residuals are deterministic, not stochastic.** In a real quadrotor, `f_residual` contains sensor noise, motor variance, and aerodynamic turbulence. In PyBullet kinematic sim, the residual is fully determined by state. This means a NN trained on sim data will overfit perfectly to the sim's drag model, but may not transfer to real hardware.

3. **Slung-payload focus not directly applicable.** Our racing drone carries no payload. The payload adaptation sections of the paper (likely ~half the content) are not applicable. However, the payload disturbance force is structurally similar to aerodynamic interaction forces in banked turns, so the methodology transfers.

4. **INDI requires high-rate sensing (500–1000 Hz).** If deploying on real hardware (MAVLink competition interface), the INDI inner loop requires angular acceleration feedback at rates far exceeding typical MAVSDK update rates (~50–100 Hz). The neural-augmented version is more deployable because it runs on standard sensor rates.

5. **Computational cost of neural inference.** Adding a neural network inference call to the control loop could reduce throughput. Our current loop runs at ~100 Hz; a small MLP (32 hidden units, NumPy inference) adds <0.5 ms per call, which is acceptable. However, a PyTorch model with CUDA overhead could introduce latency.

6. **Training data distribution shift.** If gains are changed or the trajectory is modified, the residual distribution shifts. The trained corrector may then amplify rather than reduce error. The hybrid blend (fall back to pure PD when uncertain) is the mitigation strategy the paper recommends.

7. **No quantitative ablation numbers in abstract.** The paper does not report specific RMSE improvements in the publicly available abstract. Exact gains over baselines are not known from the available information; the qualitative ordering (NN > INDI, hybrid > either alone) is confirmed, but magnitude is unspecified.

---

## Key Parameters / Constants

From the paper and related INDI literature:

- **INDI control loop rate (classical):** 500–1000 Hz to keep Taylor linearization valid
- **Neural-Fly control loop rate:** 50 Hz (position loop); PX4 inner loop handles attitude at higher rate
- **Neural-Fly NN architecture:** 5-layer wind-invariant net + 3-layer wind-class predictor
- **NeuroBEM residual network:** Validated with ~12 minutes of flight data (≈ 360K samples at 500 Hz)
- **Control effectiveness matrix G:** 3×4 Jacobian `∂Ω̇/∂u` evaluated at current rotor speeds
- **Filter delay compensation:** Critical for INDI stability — delay > ~5 ms causes oscillation
- **Adaptive INDI payload tolerance:** Works up to 60% payload-to-vehicle mass ratio
- **Tracking error improvement (INDI inner-loop):** >78% position RMSE reduction vs. baseline
- **Tracking error improvement (adaptive INDI vs. non-adaptive NMPC, payload):** >90% reduction
- **Best-case agile INDI tracking:** 4.0 cm RMS at 8.2 m/s, 2g

For our system (`TrackerConfig`):
- `kp_xy=6.0, kd_xy=4.0, feedforward_accel=0.4` — current values
- Turn error: 0.64 m avg (gates 7–12) vs 0.31 m (gates 1–6)
- Target for neural residual correction: close the 0.33 m gap to below 0.40 m in turns
- Acceptable inference overhead: <1 ms per control step (100 Hz loop = 10 ms budget)
- Suggested residual MLP: input dim=7, hidden=32×2, output=3, NumPy forward pass

---

Sources:
- [Neural-Augmented INDI for Quadrotors with Payload Adaptation (arXiv:2503.09441)](https://arxiv.org/abs/2503.09441)
- [Accurate Tracking of Aggressive Quadrotor Trajectories using INDI (Tal & Karaman, 2018)](https://arxiv.org/pdf/1809.04048)
- [Neural-Fly enables rapid learning for agile flight in strong winds (O'Connell et al., Science Robotics 2022)](https://arxiv.org/abs/2205.06908)
- [Advancements in INDI survey Part II (Chinese Journal of Aeronautics, 2025)](https://www.sciopen.com/article/10.1016/j.cja.2025.103591)
- [Adaptive Incremental Nonlinear Dynamic Inversion for Attitude Control of MAVs (Smeur et al., TU Delft)](https://arc.aiaa.org/doi/abs/10.2514/1.G001490)
- [A Comparative Study of Nonlinear MPC and DFBC for Quadrotor Agile Flight (RPG UZH)](https://arxiv.org/html/2109.01365v6)
