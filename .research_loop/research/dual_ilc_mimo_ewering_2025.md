# Dual Iterative Learning Control for Multiple-Input Multiple-Output Dynamics with Validation in Robotic Systems

- **URL**: https://arxiv.org/abs/2509.18723
- **Authors**: Jan-Hendrik Ewering, Alessandro Papa, Simon F.G. Ehlers, Thomas Seel, Michael Meindl (Institute of Mechatronic Systems, Leibniz Universität Hannover)
- **Year**: 2025
- **Venue**: arXiv preprint (eess.SY — Electrical Engineering and Systems Science)

---

## Key Contribution

This paper introduces **Dual Iterative Learning Control (DILC)**, a framework that simultaneously learns both (a) a plant model and (b) a tracking control input for MIMO dynamical systems, without requiring any prior system knowledge or manual parameter tuning. The core insight is that model identification and trajectory-tracking feedforward synthesis are *structurally dual* problems — both can be formulated as ILC update laws over lifted trajectory representations. Leveraging this duality, a single algorithmic framework handles both learning tasks simultaneously, with provable convergence guarantees.

The three stated contributions are:
1. A novel DILC scheme enabling simultaneous model and control learning for MIMO systems with unknown plant dynamics.
2. Theoretical convergence analysis (three theorems) establishing conditions for monotonic and exponential error reduction.
3. Extensive empirical validation across simulated industrial robots and real-world nonlinear robotic platforms.

---

## Technical Approach (Detailed)

### Problem Setup

The system is modeled in **lifted (batch) form** over a trial of N timesteps:

```
y_j = P * u_j
```

where:
- `u_j ∈ ℝ^(NI)` — full input trajectory for trial j
- `y_j ∈ ℝ^(NO)` — full output trajectory
- `P` — unknown block-lower-triangular Toeplitz matrix (causal LTI dynamics), size `(NO × NI)`

The tracking error is `e_j = r - y_j` where `r` is the reference trajectory. The goal is to learn `u_j` such that `e_j → 0`.

### Standard ILC Update (P-type)

Given a plant model `M_j ≈ P`, the standard ILC input update (Eq. 8) is:

```
u_{j+1} = u_j + L_j * e_j
```

where `L_j = D(M_j, W_j)` is a learning gain computed from a **design function** `D`. Two design functions are supported:

**Gradient ILC (G-ILC)** (Eq. 12):
```
L_j = D_G(M_j, W_j) = M_j^T * W_j
```

**Norm-Optimal ILC (NO-ILC)** (Eq. 13):
```
L_j = D_NO(M_j, W_j, S_j) = (M_j^T * W_j * M_j + S_j)^{-1} * M_j^T * W_j
```

where `W_j` is a symmetric positive definite output-weighting matrix and `S_j` is an input-regularization matrix. The NO-ILC form is the familiar pseudo-inverse / Tikhonov-regularized update.

### Iterative Model Learning (IML): The Dual Problem

The key innovation is recognizing that learning `M_j` from input-output data `(u_j, y_j)` is itself an ILC problem in a lifted space. Two **lifting operators** are defined:

- **Input lifting operator** `L_u`: transforms input trajectory vector `u` into a structured matrix `U` (block-lower-triangular with Toeplitz structure)
- **Model lifting operator** `L_m`: vectorizes the plant matrix into `m_j = L_m(M_j)`

**Lemma 1** establishes that `A * x = L_u(x) * L_m(A)` for block-lower-triangular matrices, which means the prediction `ŷ_j(u) = P * u` can be rewritten as:

```
ŷ_j(u) ≈ U_j * m_j     where U_j = L_u(u_j)
```

This transforms model identification into a linear regression problem that has exactly the same structure as the ILC tracking problem. The model update law (Eq. 23) then mirrors the ILC update:

```
m_{j+1} = m_j + L̂_j * ê_j(y_j, u_j)
```

where `L̂_j = D̂(U_j)` is computed using the same design function machinery applied to the lifted input matrix.

### Full DILC Algorithm (Algorithm 1)

Each trial j proceeds in three steps:

1. **Execute**: Apply `u_j` to the system, measure output `y_j`, compute tracking error `e_j = r - y_j`.
2. **Model update (IML)**: Compute lifted input matrix `U_j = L_u(u_j)`, compute model learning gain `L̂_j = D̂(U_j)`, update `m_{j+1} = m_j + L̂_j * ê_j(y_j, u_j)`, reshape to `M_{j+1}`.
3. **Control update (ILC)**: Compute control learning gain `L_j = D(M_{j+1})`, update `u_{j+1} = u_j + L_j * e_j`.

The naming convention reflects the design function choice for each step independently. For example, GNOG-DILC uses G-ILC for model learning and NO-ILC for control learning.

### MIMO-to-SISO Decomposition (Lemma 2)

A MIMO system with O inputs and O outputs decomposes into O² SISO subsystems, each described by a lower-triangular Toeplitz matrix of size N×N. This is used for:
- Theoretical analysis (reduce MIMO convergence to SISO conditions)
- Self-parametrization (compute per-channel weighting automatically)

### Convergence Theorems

**Theorem 1 (Monotonic Model Convergence)**: Under Assumption 1 (persistency of excitation), IML achieves monotonic model convergence when `‖I − L̂_j * U_j‖ ≤ 1`. The convergence may be non-exponential (characterized by a KL-function bound) due to model overparameterization — there are O²N parameters to infer but each trial provides only ON constraints, requiring O consecutive trials for the problem to become well-posed.

**Theorem 2 (Exponential Prediction Convergence)**: The prediction error satisfies:
```
ê_{j+1} = (I − U_j * L̂_j) * ê_j
```
Exponential convergence holds when `‖I − U_j * L̂_j‖ ≤ γ` for some `γ ∈ [0, 1)`.

**Theorem 3 (DILC Tracking Convergence)**: After a threshold trial J (when model error enters a sufficient neighborhood of P), tracking error achieves exponential monotonic convergence:
```
‖e_{j+1}‖ ≤ α * ‖e_j‖     for α ∈ [0, 1),  ∀j ≥ J
```
This requires both design functions to be continuous and individually satisfy their convergence requirements. Before trial J, monotone tracking improvement is not guaranteed.

### Persistency of Excitation (Assumption 1)

The excitation condition (Eq. 6) requires:
```
rank([ū_j(1), ū_{j+O-1}(1)]) = O
```
That is, the initial input samples across O consecutive trials must span full rank. This is a mild condition — it only constrains the first time sample across O trials, not the richness of the entire trajectory.

### Self-Parametrization (No Manual Tuning)

The weighting matrices `W_j` and `S_j` are computed from the current model estimate, enabling autonomous operation. For Gradient methods (Eq. 40):
```
w̃_i = 1 / ‖[M̃_{i,1}, …, M̃_{i,O}]‖²
```
This normalizes each output dimension by the row-wise sensitivity of the model, balancing learning rates across channels with very different gains. For Norm-Optimal methods, `Q_j` and `S_j` are similarly derived from model structure.

---

## Results

### Simulated 6-Axis Robot (UR10e, MuJoCo)
- Nonlinear 6-DOF arm with measurement noise (σ = 10⁻⁵ rad)
- Highly dynamic reference with discontinuities
- NONO-DILC and GNO-DILC both converge within 100–150 trials to near-perfect tracking
- NOG-DILC **fails** (stagnating error norm) — demonstrating that not all design function combinations work

### Two-Link Planar Robot (Real Hardware)
- Significant gearbox backlash and Coulomb friction (nonlinear effects)
- GG-DILC achieves convergence in ≤50 trials across three different reference trajectories
- Monotonic learning despite hysteresis — the linear Toeplitz model assumption is violated but DILC still works

### Three-Wheeled Inverted Pendulum (TWIPR, Real Hardware)
- Underactuated system with coupled dynamics; varying initial conditions across trials
- Varying initial conditions violate the standard ILC repeatability assumption
- NOG-DILC achieves >80% error reduction in the first 10 trials, full convergence in ≤20 trials
- Demonstrates robustness to mild ILC assumption violations

### Key Quantitative Claims
- Convergence within **10–20 trials** for standard tasks
- Under **100 trials** for complex 6-DOF motions with discontinuous references
- Zero prior system knowledge required
- Scales to 6+ DOF systems (O²N model parameters for 6-axis = 36N parameters)

---

## Relevance to Our System

Our system uses **P-type ILC with Butterworth Q-filter** applied per section of the drone racing trajectory. DILC is directly relevant in the following ways:

### 1. Convergence Framework Applies to Our P-Type ILC

Our per-section P-type update `u_{j+1}^s = u_j^s + L * e_j^s` corresponds exactly to Eq. 8 in the paper with a fixed (not learned) plant model. The convergence conditions from Theorem 3 reduce to: `‖I − P * L‖ < 1` in the lifted domain. Our Q-filter modifies L to ensure this condition by attenuating high-frequency updates.

### 2. Per-Section ILC Is Not Directly Analyzed

The paper treats ILC over full trial trajectories using global lifting. It does **not** analyze segment-wise or per-section ILC. This is an important gap — our implementation applies independent ILC updates per section of the trajectory, and convergence of the combined system is not guaranteed by Theorem 3 alone (which assumes a unified lifted representation).

### 3. Q-Filter as an Implicit Convergence Mechanism

The paper avoids Q-filters entirely, instead relying on weighting matrices `W_j` and `S_j` to control convergence. In our system, the Butterworth Q-filter plays the analogous role: it attenuates the learning gain at high frequencies where model uncertainty is large, enforcing `‖(I − Q(z) * P * L)‖ < 1` in the frequency domain. Different filter bandwidths per section corresponds to different effective learning gains — the paper's analysis supports this as valid as long as each section's effective gain satisfies the contraction condition.

### 4. Model Uncertainty Before Convergence (Threshold Trial J)

Theorem 3 warns that before threshold trial J, tracking may not improve monotonically. In our system, the "model" is fixed (we use our known drone dynamics), so we are already past threshold J conceptually — this makes our system better conditioned than the DILC setting and convergence from trial 1 is more likely.

### 5. MIMO Decomposition Matches Our Multi-Axis Control

The MIMO-to-SISO decomposition (Lemma 2) supports treating each spatial axis (x, y, z) as independent SISO ILC channels — which is what our per-axis Butterworth Q-filter implicitly does. Cross-axis coupling is the main source of approximation error.

---

## Actionable Takeaways

1. **Verify per-section convergence condition independently per section**: For each section s with Butterworth cutoff `ω_c^s`, check that `‖(I − Q_s * P_s * L_s)‖ < 1` in the lifted domain. Different bandwidths are valid as long as each section independently satisfies this contraction bound.

2. **Tighter bandwidth → slower convergence, better robustness**: The paper's Theorem 2 shows convergence rate is `γ = ‖I − U * L̂‖`. Lower-bandwidth Q-filters reduce the effective `L` magnitude, increasing `γ` toward 1 (slower convergence) but reducing risk of divergence. Choose per-section bandwidths based on local trajectory curvature and expected model mismatch.

3. **Section boundary effects are unanalyzed — add stitching validation**: Since the paper only handles full-trajectory lifting, our per-section boundary conditions (how corrections at section s affect section s+1) are not covered by their theory. Empirically monitor whether corrections in one section destabilize adjacent sections.

4. **Consider NO-ILC (Norm-Optimal) update for critical high-curvature sections**: The NO-ILC form `L = (M^T W M + S)^{-1} M^T W` automatically regularizes the update magnitude via `S`. For sections near sharp turns where the drone model is less accurate, adding input regularization (larger `S`) prevents overcorrection.

5. **Watch for the NOG-DILC failure mode**: The paper shows that the combination GG (Gradient model learning + Gradient control) works, but NOG (Norm-Optimal model + Gradient control) can fail. For us: if using different ILC strategies per section (e.g., tighter filter for one section, looser for another), verify empirically that switching logic does not cause divergence at section boundaries.

6. **Monotonic convergence is not guaranteed early**: Before the drone dynamics model is well-identified, DILC warns that early trials may not improve monotonically. For our system with a known physics model, this is less of a concern, but if we adapt the model online, expect potential non-monotone learning during the first few iterations.

7. **Persistency of excitation is automatically satisfied in racing**: Because each lap has a different initial perturbation (sensor noise, wind), the excitation condition `rank([ū_j(1), …]) = O` is satisfied trivially. No special dithering signal is needed.

8. **Self-parametrization concept is useful for auto-tuning Butterworth cutoffs**: The paper's weighting `w̃_i = 1 / ‖[M̃_{i,1}, …]‖²` normalizes by model sensitivity. We could adapt this: sections where the drone model has lower confidence (e.g., high-curvature gates) get lower-bandwidth Q-filters automatically. This is a principled alternative to manually tuning per-section bandwidths.

---

## Limitations & Caveats

1. **No Q-filter analysis**: The paper avoids frequency-domain filtering entirely. Convergence results are in the operator-norm (lifted matrix) domain. Direct mapping to Butterworth filter cutoff frequency requires a separate frequency-domain analysis (e.g., via the standard ILC convergence condition in the z-domain).

2. **No per-section or segment-wise ILC theory**: All results assume the ILC operates over the full trial trajectory with a single global lifted representation. Our per-section approach violates this assumption. The paper does not bound how section-independence degrades convergence.

3. **Linear time-invariant assumption**: The plant is modeled as a block-lower-triangular Toeplitz matrix (LTI causal system). Drone dynamics are nonlinear, especially at high speeds and during gate traversal. The experiments show DILC is practically robust to mild nonlinearity, but theoretical guarantees strictly require LTI.

4. **Threshold trial J is uncharacterized**: Theorem 3 guarantees convergence after J, but provides no bound on J in terms of system parameters. In the worst case, J could be large, meaning tracking does not improve for many trials.

5. **NOG-DILC failure case unexplained**: The 6-axis robot simulation shows stagnating error with one design function combination (NOG), but the paper does not derive conditions to predict which combinations will succeed.

6. **No noise or disturbance robustness theory**: Robustness claims are empirical only (noise level 10⁻⁵ rad in simulation). For real drone racing with significant sensor noise and wind gusts, formal robustness bounds would require separate analysis (e.g., via robust ILC frameworks).

7. **Computational cost scales as O²N²**: The lifted matrices grow quadratically with trajectory length N and number of I/O channels O. For long trajectories, inverting these matrices (required for NO-ILC) is expensive. Our per-section approach mitigates this by working on shorter segments.

---

## Key Parameters / Constants

| Symbol | Meaning | Value / Condition |
|--------|---------|-------------------|
| `α` | Tracking error convergence rate | `α ∈ [0, 1)` for exponential convergence |
| `γ` | Prediction error contraction bound | `γ ∈ [0, 1)`, equals `‖I − U_j * L̂_j‖` |
| `J` | Threshold trial from which tracking converges | Uncharacterized; system-dependent |
| `O` | Number of inputs / outputs (MIMO dimension) | 6 for UR10e arm; 3 for TWIPR |
| `N` | Timesteps per trial | Unspecified, varies by experiment |
| `W_j` | Output weighting matrix (symmetric PD) | Set via self-parametrization, Eq. 40 |
| `S_j` | Input regularization matrix | Set via self-parametrization, Eq. 42 |
| `σ_noise` | Measurement noise (simulation) | `10⁻⁵ rad` (negligible) |
| Trials to convergence | Standard tasks | 10–20 trials |
| Trials to convergence | Complex 6-DOF | < 100 trials |
| Model overparameterization | Parameters vs. constraints | O²N parameters, ON constraints per trial |
| Excitation window | Trials needed for well-posedness | O consecutive trials (Assumption 1) |

---

*Analysis written 2026-04-14 for the AI Grand Prix drone racing ILC project.*
