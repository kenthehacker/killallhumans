# Aggressiveness-Aware Learning-based Control of Quadrotor UAVs with Safety Guarantees

- **URL**: https://arxiv.org/abs/2602.21936
- **Authors**: Leonardo Colombo (Centre for Automation and Robotics, CSIC-UPM, Spain), Thomas Beckers (Vanderbilt University, USA), Juan I. Giribet (Universidad de San Andrés and CONICET, Argentina)
- **Year**: 2026
- **Venue**: arXiv preprint (2602.21936v1)

---

## Key Contribution

This paper formalizes and resolves a fundamental tension in quadrotor control: higher feedback gains improve disturbance rejection but increase control aggressiveness (actuator stress, input sensitivity, safety risk). The authors introduce an **aggressiveness metric** — a local input-sensitivity measure inspired by stiffness concepts in soft robotics — that quantifies how hard the controller is "working" relative to the tracking error. They then show that by learning unknown disturbances with Gaussian Processes (GPs) and coupling gain scheduling to the GP posterior uncertainty, one can maintain tracking guarantees while keeping aggressiveness at the minimum necessary level.

The central theoretical insight is that better disturbance model accuracy directly enables smaller feedback gains while still satisfying tracking bounds. As the GP oracle improves (more data, lower posterior uncertainty), the minimum viable feedback gain decreases, asymptotically approaching a baseline that is independent of the data. This is formalized in Corollary 1 and Theorem 1, which provide high-probability practical exponential tracking guarantees with an explicit dependence on oracle quality. The framework is highly practical: it provides a six-step implementation procedure and supports offline, online, or hybrid GP training configurations.

---

## Technical Approach

### Problem Setup

The quadrotor is modeled with translational and rotational dynamics that include unknown generalized disturbance forces f_trans(x) and moments f_rot(x). These represent aerodynamic effects, model mismatch, prop wash, and similar phenomena that degrade trajectory tracking. The state is (position, velocity, rotation matrix R ∈ SO(3), angular velocity ω in body frame). Control inputs are total thrust T and body torques τ_b.

### Aggressiveness Metric (Eq. 14)

The feedback-induced aggressiveness is defined as:

    s(H, x) = ‖∂/∂e (h_fb(x) H e)‖ = ‖h_fb(x) H‖

where `e` is the tracking error vector, `H` is the feedback gain matrix, and `h_fb(x)` is a state-dependent feedback shaping function. This metric measures how sensitive the commanded forces/moments are to small perturbations in tracking error — a high value means the controller reacts violently to small errors, increasing actuator wear and crash risk.

The key observation: aggressiveness is linear in the gain matrix norm ‖H‖, so reducing gains reduces aggressiveness proportionally.

### GP-Based Disturbance Oracle

Six independent GP regressors (one per disturbance component — 3 translational, 3 rotational) learn the unknown disturbance f(x) from labeled flight data. The GP posterior mean provides the disturbance estimate f̂_N(x), and the posterior variance provides a high-probability error bound:

    P{ ‖f(x) - f̂_N(x)‖ ≤ ρ̄_N(x, δ) ∀x ∈ X_c } ≥ δ

where δ is the desired confidence level (e.g., 0.95) and ρ̄_N depends on GP posterior variance and information-gain quantities. Critically, ρ̄_N decreases as more data is collected.

### Augmented Control Law (Eq. 24)

    u = h_ff(x, x_d) - K_dyn(x) f̂_N(x) + h_fb(x) H_N e

- `h_ff(x, x_d)`: nominal geometric feedforward (same as standard Lee et al. SE(3) controller)
- `K_dyn(x) f̂_N(x)`: GP disturbance compensation term (subtract the learned disturbance)
- `h_fb(x) H_N e`: feedback term with gain matrix H_N scheduled to oracle quality

The geometric structure (SO(3)/SE(3)) is preserved; the GP augments rather than replaces the geometric tracking law.

### Gain Scheduling via Optimization (Eq. 23)

The gain matrix H_N is chosen by solving:

    minimize s(H, x) = ‖h_fb(x) H‖
    subject to: practical exponential tracking bound is satisfied

The constraint requires that the GP error bound ρ̄_N is small enough:

    sup_x ρ̄_N(x, δ) ≤ (c₁ / 2c₂) ε

where ε is the desired steady-state tracking tolerance. When this condition is met, the minimum gain that still satisfies tracking can be computed analytically. The aggressiveness bound then scales as:

    s(H_N, x) ≤ α₁ ρ̄_N(x, δ) + α₂

where α₂ is a baseline aggressiveness independent of the oracle — the irreducible minimum. As ρ̄_N → 0, s → α₂ (Corollary 1).

### Decoupled Scheduling (Proposition 1)

The framework separates translational and rotational error bounds, allowing independent gain tuning for each subsystem based on their respective GP accuracy. This is practically important because translational disturbances (aerodynamics) and rotational disturbances (gyroscopic effects, propeller asymmetries) are often of different magnitudes and frequencies.

### Gating Mechanism

A gating function controls when the GP compensation is active. In the experiments, the mean gate activation in steady-state was 0.930 (93% of time steps), indicating the learned model was reliable enough to use throughout most of the trajectory.

### Implementation Procedure (Six Steps)

1. **State Estimation**: Measure/estimate current state via IMU fusion or EKF.
2. **Disturbance Labeling**: Construct data-driven disturbance estimates by subtracting known model terms from measured accelerations.
3. **Oracle Training**: Update GP on dataset (offline batch, online streaming, or hybrid).
4. **Error Bound Computation**: Calculate high-probability bound ρ̄_N(·, δ) over the expected operating region.
5. **Gain Scheduling**: Enforce the feasibility condition; optionally decouple translational/rotational blocks.
6. **Control Application**: Execute the augmented law (Eq. 24).

---

## Results

### Simulation Parameters

- Quadrotor: m = 1 kg, diagonal inertia matrix
- Reference trajectory: nontrivial x-y excitation, altitude modulation, constant yaw
- Tracking tolerance: ε = 0.1 m over T = 20 s horizon
- Disturbance scaling: DIST_SCALE ∈ {1 (moderate), 3 (severe)}
- Aggressiveness metrics: RMS thrust rate |Ṫ|_RMS and torque rate ‖τ̇‖_RMS (measured separately for transient and steady-state phases)

### Moderate Disturbance (DIST_SCALE = 1)

| Controller          | Position Error ‖e_p(T)‖ | Thrust Rate |Ṫ|_RMS,tr |
|--------------------|------------------------|--------------------------|
| Fixed-low gains     | 0.116 m                | 7.492 N/s                |
| Fixed-high gains    | 0.060 m                | 9.925 N/s (+32%)         |
| Aggressiveness-aware| 0.094 m                | 8.121 N/s                |

The aggressiveness-aware approach achieves intermediate tracking error with only slightly higher aggressiveness than fixed-low gains.

### Severe Disturbance (DIST_SCALE = 3)

| Controller               | Position Error | Notes                          |
|--------------------------|----------------|--------------------------------|
| Fixed-low gains           | 0.351 m        | Violates ε = 0.1 m tolerance  |
| Fixed-high gains          | 0.190 m        | Still violates tolerance       |
| Aggressiveness-aware (best)| 0.134 m       | trans_scale = 2.5              |

Fixed-high gains still fail to meet the 0.1 m tolerance under severe disturbance — no amount of gain increase can compensate without a model.

### GP-Augmented Under Severe Disturbance (Table I)

With learned disturbance compensation (GP trained on labeled data):

| Metric                          | Value                        |
|---------------------------------|------------------------------|
| Position error ‖e_p(T)‖         | **0.028 m** (vs. 0.351 m baseline) |
| Thrust rate |Ṫ|_RMS,tr          | 8.066 N/s (same as fixed-low) |
| Torque rate ‖τ̇‖_RMS,ss         | 0.175 Nm/s                   |
| Gate activation (steady-state)  | 0.930 (93%)                  |
| Feedback gain norm ‖H‖_F        | 17.866 (same as fixed-low)   |

Key result: **84% reduction in tracking error** (0.351 → 0.028 m) while maintaining the aggressiveness of the conservative fixed-low controller. The GP compensation, not gain increase, does the work.

---

## Relevance to Our System

Our current system uses a geometric SE(3) controller (Lee et al.) with a tilt limit of 0.85 rad tracking min-snap polynomial trajectories. The primary problem is controller saturation at moderate-angle gates (48°/38°) with high approach speed, causing 0.6 m+ tracking errors. The system is already using the geometric framework that this paper builds on — the augmentation is fully compatible.

**Direct relevance:**

1. **The core problem is identical.** We are experiencing high tracking error at sharp turns not because our controller structure is wrong, but because we are fighting aerodynamic and inertial disturbances that the geometric controller cannot compensate without either violating tilt limits (crashing) or reducing speed (losing race time). This paper's framework is precisely designed for this tradeoff.

2. **GP disturbance compensation addresses our gate errors.** The 0.6 m+ errors at gates 3 and 7 (the sharp turns) correspond to unmodeled aerodynamic forces — prop wash, induced drag, ground effect interaction. A GP trained on the PyBullet simulation data could learn these disturbances and pre-compensate before each gate.

3. **Gain scheduling gives us the aggressiveness knob.** We currently have a single set of controller gains tuned conservatively to avoid crashes. With the aggressiveness-aware framework, we could use lower gains in straight sections (reducing jitter) and schedule up to higher gains at gate approaches, with the GP handling the disturbance burst.

4. **Decoupled translational/rotational scheduling** (Proposition 1) maps directly to our separate position and attitude control loops. We can tune each independently based on which subsystem is more accurately modeled.

5. **The six-step implementation** is concrete enough to implement in our `control/mpc_tracker.py` and `estimation/ekf.py` stack within a few hundred lines of code. The GP layer sits on top of the geometric controller without requiring architectural changes.

---

## Actionable Takeaways

1. **Collect disturbance data from PyBullet runs.** During simulation, log the difference between commanded and actual accelerations (translational + rotational) along with the state vector. This is the labeled dataset for GP training. The disturbance labeling step (step 2 in the paper) is straightforward given we have ground-truth physics from PyBullet.

2. **Train 6 independent GP regressors** on this data. Use `sklearn.gaussian_process.GaussianProcessRegressor` or `GPyTorch` for efficiency. The input features should be: position, velocity, attitude (as rotation matrix or quaternion), angular velocity. Start with an RBF kernel.

3. **Integrate GP compensation into `control/mpc_tracker.py`.** The augmented control law (Eq. 24) adds a GP feedforward term: subtract `K_dyn(x) * f̂_N(x)` from the commanded thrust/torques before sending to the motor allocator. This requires no changes to the geometric tracking math.

4. **Implement gain scheduling tied to GP uncertainty.** When the GP posterior variance is high (low confidence), use conservative gains. When variance is low (high confidence, typically in the straight sections we have trained on), reduce gains to lower aggressiveness. The feasibility condition `sup_x ρ̄_N(x,δ) ≤ (c₁/2c₂)ε` gives the criterion.

5. **Gate-specific disturbance models.** Since our worst errors occur at specific gates (3 and 7), train separate GP models for each gate's approach corridor. The input state space is smaller (only the local approach region), so the GP will be more accurate with fewer data points.

6. **Use the gating mechanism to protect against bad GP predictions.** Only apply GP compensation when the posterior variance is below a threshold. The paper uses 93% gate activation in steady-state — we should aim for similar but can start with a conservative threshold and relax it as confidence builds.

7. **Decouple translational and rotational compensation.** Our translational errors at gates are primarily caused by aerodynamic drag during high-speed turns, while rotational errors come from gyroscopic effects and prop asymmetries. Train separate GP models for each and tune gains independently per Proposition 1.

8. **Use the aggressiveness metric `s(H,x) = ‖h_fb(x) H‖` as a monitoring signal.** Log it during benchmark runs. If aggressiveness spikes at gate approaches, this indicates the controller is fighting disturbances rather than tracking — a signal that the GP is not compensating enough in that region.

---

## Limitations & Caveats

1. **Simulation-to-real gap.** The paper validates in simulation only (no hardware experiments). For us, this is less critical since we are also in simulation (PyBullet), but our final competition will run on real hardware. The GP trained on PyBullet data may not generalize to real aerodynamics. This is a known sim-to-real transfer problem.

2. **GP computational cost.** Standard GPs scale as O(N³) for training and O(N) for inference. In our real-time control loop (100+ Hz), online GP inference must be fast enough. The paper uses offline-trained GPs primarily, with optional online refinement via streaming updates. For us, offline training on PyBullet data and online inference is the practical path. With sparse GPs or inducing point methods, inference can be made fast enough.

3. **GP input dimensionality.** The full state vector (position, velocity, attitude, angular velocity) is 12+ dimensional. Standard GP kernels suffer in high dimensions. The paper does not explicitly address this — in practice, careful feature selection or dimensionality reduction (e.g., using only local velocity and attitude) will be necessary.

4. **Assumes the GP can cover the operating region.** The feasibility condition requires `sup_x ρ̄_N(x,δ)` to be small over the expected flight region. If the competition track contains trajectory segments far from the training distribution (e.g., different gate spacings, wind conditions), the GP uncertainty will be high and the gain scheduling will revert to conservative gains — losing the benefit.

5. **Tracking tolerance ε = 0.1 m is for a general trajectory, not gate-passing.** For us, the relevant metric is gate-passing margin (~gate_width/2). Our gates have a nominal half-width; we need to translate the error bound into a gate clearance requirement.

6. **The disturbance model assumes time-invariant disturbances conditioned on state.** Wind gusts, battery voltage sag, and propeller wear are time-varying effects that the GP (conditioned only on state) cannot capture. In practice, online refinement mitigates this but does not fully solve it.

7. **The paper addresses trajectory tracking, not time-optimal racing.** The min-aggressiveness objective may conflict with min-time racing: going faster requires more aggressive control. The framework provides a principled tradeoff but does not automatically optimize for lap time.

---

## Key Parameters / Constants

From the simulation experiments:

- **Tracking tolerance**: ε = 0.1 m (the target steady-state bound)
- **Simulation horizon**: T = 20 s
- **Quadrotor mass**: m = 1 kg (simulation, not a parameter to copy but context for scaling)
- **Confidence level**: δ = 0.95 (implied by "high-probability" bounds — typical GP safety margin)
- **Gate activation threshold**: mean = 0.930 (93% steady-state application rate — use as a starting target)
- **Feedback gain norm (fixed-low baseline)**: ‖H‖_F = 17.866 (the minimum that worked without GP compensation at DIST_SCALE = 1)
- **Aggressiveness under GP compensation**: |Ṫ|_RMS,tr = 8.066 N/s (matched fixed-low, despite 84% better tracking)
- **Aggressiveness scaling constants**: α₁, α₂ in s(H_N,x) ≤ α₁ ρ̄_N + α₂ — must be computed from our system's h_fb(x) and dynamics, but the linear structure is usable directly

For gain scheduling, the key condition is:

    sup_x ρ̄_N(x, δ) ≤ (c₁ / 2c₂) ε

where c₁, c₂ are positive constants from the Lyapunov analysis of the nominal geometric tracker. These are controller-specific and must be derived from our Lee et al. implementation — typically c₁ and c₂ are related to the eigenvalues of the gain matrices K_p, K_v, K_R, K_ω.
