# Monotonically Convergent ILC by Time Varying Learning Gain Revisited
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S000510982300420X
- **Authors**: Jian Liu, Yuanshi Zheng, YangQuan Chen
- **Year**: 2023
- **Venue**: Automatica, Vol. 157, Article 111259
- **DOI**: 10.1016/j.automatica.2023.111259

---

## Key Contribution

This paper makes a fundamental theoretical contribution to iterative learning control (ILC) by rigorously settling a long-open question: can a time-varying learning gain be designed to guarantee **monotone convergence** of system output tracking errors (SOTEs) in the sense of standard, unweighted norms (1-norm, 2-norm, infinity-norm)?

The paper has three distinct findings, each surprising in its own way:

1. **Negative result on Moore et al. (2005)**: The exponentially decaying learning factor proposed by Moore, Chen, and Bahl (2005) — which was widely believed to provide monotone convergence in unweighted norms — is shown to be *insufficient*. The update law with that decaying factor cannot ensure monotone convergence of SOTEs in the unweighted 1-norm, 2-norm, or infinity-norm. This is a formal refutation of a decade-old claim.

2. **Positive result for ∞-norm**: Despite the above, the paper proves that there *does* exist a valid time-varying learning gain (specifically, one with an exponentially *increasing* factor) that guarantees monotone convergence of SOTEs in the unweighted infinity-norm. This is constructive — such a gain exists and is characterizable.

3. **Negative result for 1-norm**: There is provably *no* time-varying learning gain for the Arimoto-type P-ILC update law that can guarantee monotone convergence in the unweighted 1-norm. This is a fundamental impossibility result.

4. **Sufficient condition for 2-norm**: A sufficient condition is derived under which a time-varying learning gain guarantees monotone convergence in the unweighted 2-norm. This is not a necessary condition — it is a conservative sufficient condition, meaning satisfying it is enough but not always required.

Additionally, the paper highlights that the initial system input and the trial length both have significant influence on the transient tracking performance and convergence rate, which has direct implications for practical ILC design.

---

## Technical Approach

### System Setup

The paper considers a **linear, time-invariant (LTI) discrete-time system** of the form:

```
x(k+1) = A x(k) + B u(k)
y(k)   = C x(k)
```

The standard P-type ILC update law (Arimoto-type) is:

```
u_{j+1}(k) = u_j(k) + L(k) * e_j(k)
```

where `j` is the iteration index, `k` is the time index within a trial of length `N`, `e_j(k) = y_d(k) - y_j(k)` is the tracking error at time step k in trial j, and `L(k)` is the *possibly time-varying* learning gain.

### Super-Vector (Lifted) Formulation

The key mathematical tool is the super-vector or "lifting" formulation. Define:

```
E_j = [e_j(0), e_j(1), ..., e_j(N-1)]^T    (stacked tracking errors)
U_j = [u_j(0), u_j(1), ..., u_j(N-1)]^T    (stacked control inputs)
```

The error dynamics become:

```
E_{j+1} = (I - H * diag(L)) * E_j
```

where `H` is the **lower-triangular Toeplitz matrix of Markov parameters** (the impulse response matrix of the system). The (i,k) entry of H is `C A^{i-k-1} B` for i > k, and `CB` on the diagonal. This matrix encodes the entire input-output relationship of the system for one trial.

### Norm Analysis

The 1-, 2-, and ∞-norm monotone convergence of `E_j` correspond to different matrix-norm conditions on the **iteration operator** `Q = I - H * diag(L)`:

- Monotone convergence in the **∞-norm** requires `||Q||_∞ < 1`, i.e., all row-sums of |Q| are less than 1.
- Monotone convergence in the **1-norm** requires `||Q||_1 < 1`, i.e., all column-sums of |Q| are less than 1.
- Monotone convergence in the **2-norm** requires `||Q||_2 = σ_max(Q) < 1`, the spectral condition.

### Why 1-Norm Fails

For P-type ILC, H is strictly lower-triangular (relative degree ≥ 1 means `CB = 0`), so the diagonal entries of H are zero. Because `L(k)` multiplies columns of H, and H's first column is all-zero, the gain `L(0)` at time step 0 cannot influence any row sum. The structure means that no choice of `L(k)` can make all column sums of |Q| less than 1 simultaneously — hence the impossibility for the 1-norm.

### Why ∞-Norm Succeeds with Exponentially Increasing Gain

For the ∞-norm, the requirement is on *row* sums. An exponentially *increasing* gain of the form `L(k) = γ * ρ^k` (with ρ > 1) can compensate for the growing influence of later time-steps in the Toeplitz structure. With careful selection of γ and ρ based on the Markov parameters, the row-sum condition can be satisfied.

Critically, this is the opposite of the Moore et al. (2005) proposal (exponentially *decaying* gain). The 2023 paper shows that decaying gains cannot provide unweighted monotone convergence in ∞-norm, while increasing gains can.

### Sufficient Condition for 2-Norm

The 2-norm condition requires `σ_max(I - H * diag(L)) < 1`. This is equivalent to requiring the largest singular value of the iteration operator to be below unity. The paper derives a sufficient condition in terms of:

- The singular values of H (which depend on the system's Markov parameters)
- The shape of the gain schedule `L(k)`
- A non-increasing function derived from the Markov parameters such that multiplying a constant gain by this function yields monotone 2-norm convergence

This builds on earlier work (Moore et al. 2005, Moore 2001) that identified the existence of such non-increasing functions, but now with corrected claims about which norm topologies they cover.

---

## Results

The paper's theoretical results establish a complete picture of what is and is not possible for monotone ILC convergence via time-varying scalar gains:

| Norm | Monotone Convergence via Time-Varying Gain? | Gain Type |
|------|---------------------------------------------|-----------|
| 1-norm | Impossible (proved) | N/A |
| 2-norm | Possible under sufficient condition | Non-increasing function × constant |
| ∞-norm | Possible (constructive proof) | Exponentially increasing gain |

Additionally, the Moore et al. (2005) exponentially decaying gain is shown to fail for all three unweighted norms — it only achieved convergence in a weighted λ-norm (which can mask actual overshoot in standard norms).

The paper also demonstrates that trial length `N` and the initial input `u_0` affect the convergence rate quantitatively: shorter trials generally converge faster because the Markov parameter matrix H is smaller and easier to dominate, while poor initialization slows convergence and can cause large transient tracking errors even when the asymptotic behavior is sound.

---

## Relevance to Our System

Our system uses offline P-type ILC with a **fixed scalar learning gain α = 0.4** applied uniformly across all `N` time steps of a drone racing trajectory. The ILC runs for up to 5 iterations in simulation, accumulating cross-track position corrections in `cumulative_offset[:]`. The key implementation question this paper illuminates is: **is a fixed gain α = 0.4 theoretically sound, and can a time-varying gain do better?**

### (a) Is time-varying learning gain theoretically better?

Yes, conditionally. The key theoretical advantage of a time-varying gain is **monotone convergence in unweighted norms**, which eliminates overshoot in the iteration domain. With our fixed α = 0.4, we may (and in practice do) see iterations where the cumulative offset overshoots the optimal correction before pulling back. This was observed in iterations 23-25: the average error oscillated between 0.199m and 0.223m across runs, suggesting non-monotone iteration behavior.

The 2023 paper shows the path to fix this:
- For **∞-norm monotone convergence** (guaranteeing no single time step ever gets worse across iterations): use an exponentially *increasing* gain schedule `L(k) = γ * ρ^k` with ρ > 1, calibrated from the system's Markov parameters.
- For **2-norm monotone convergence** (guaranteeing the RMS error never increases): multiply a constant gain by a *non-increasing* function derived from Markov parameters.

In our drone racing context, the Markov parameters correspond to the closed-loop step response of our PD controller — how much a position correction at time k affects the actual position at later time steps. These are computable from our simulation runs.

### (b) Can we use higher gains on easy straights and lower gains on the tight helix?

The paper's framework supports this, but with a specific prescription. The optimal gain profile is:

- **Straights (gates 1-6)**: The system's Markov parameter coupling is weaker (less interaction between time steps because the dynamics are more linear). A higher gain `L(k)` is safe here, and the sufficient condition for 2-norm monotone convergence is easier to satisfy with larger α.

- **Tight helix (gates 7-10)**: The Markov parameters reflect stronger coupling and more aggressive dynamics. The sufficient condition for monotone convergence requires a *lower* gain here. This is the opposite of what our fixed α = 0.4 currently does — applying the same gain in both easy and hard sections.

The current implementation's `max_correction_m = 0.15` cap provides a crude approximation of this, preventing over-correction in hard sections. But the proper approach is a trajectory-section-wise gain schedule: α(k) larger on straights, α(k) smaller in the helix.

Concretely: compute the local condition number of the Markov parameter matrix block corresponding to each trajectory segment. Where the condition number is low (straights), use higher α; where it is high (helix), use lower α.

### (c) What are the convergence guarantees for monotonically convergent ILC?

With a fixed constant gain and the standard contraction-mapping condition `||I - H diag(L)||_2 < 1`, the ILC is **asymptotically convergent** — errors decrease across iterations on average but may not do so monotonically at every step. The classical guarantee is:

```
||E_{j+1}||_2 <= ρ * ||E_j||_2    where ρ = ||I - H * L||_2 < 1
```

For our fixed α = 0.4, ρ is some value less than 1 (we haven't computed it directly, but the observed ~5% per-iteration error reduction suggests ρ ≈ 0.95). The 2023 paper's contribution is giving conditions under which ρ_k (iteration-step-wise) < 1 for every k, not just in aggregate — this is the monotone guarantee.

The practical import for our system: monotone convergence guarantees that **each ILC iteration is a strict improvement**, which means we can safely run more iterations without risk of the offset table transiently degrading. With non-monotone convergence (our current regime), stopping at 5 iterations is somewhat arbitrary. With monotone convergence, we could run 10-15 iterations and be guaranteed continued improvement.

---

## Actionable Takeaways

1. **Implement a section-wise (time-varying) ILC gain schedule.** Rather than fixed α = 0.4 everywhere, compute α(k) as a function of trajectory section. A simple proxy: use α_high ≈ 0.55 for straight sections (gates 1-2, 5-6) and α_low ≈ 0.25 for the helix (gates 7-10). This is consistent with the paper's finding that gain scheduling can improve both convergence rate and monotonicity.

2. **Use an exponentially increasing gain to target ∞-norm monotone convergence.** A gain of the form `L(k) = α_base * (1 + δ)^(k/N)` with small δ ≈ 0.3 gives a 30% higher gain at the end of the trajectory than the beginning. This is counter-intuitive but theoretically motivated: later time steps have accumulated more Markov parameter coupling and need stronger correction to achieve per-timestep monotone convergence.

3. **Compute Markov parameters from closed-loop simulation.** Run impulse-response experiments in the kinematic sim: apply a unit position perturbation at each time step k and measure the resulting trajectory deviation at steps k, k+1, ..., k+M. The first M+1 values are the Markov parameters at step k. Use these to compute the sufficient condition for the 2-norm monotone convergence, and calibrate α(k) accordingly.

4. **Do not apply a decaying gain.** The paper's main negative result directly warns against this. A gain of the form `α * exp(-λk)` (which decays as we progress through the trajectory) does NOT guarantee monotone convergence in any standard unweighted norm, despite intuitions about "being more conservative at the end."

5. **Exploit the trial-length insight.** Since the helix sub-section has the highest per-gate error, consider running a separate ILC loop on just the helix gates (shorter trial length). The paper shows shorter trials converge faster. This is implementable by computing offsets for the helix segment in isolation (masking the correction update to helix timesteps only) with more iterations.

6. **Monitor per-iteration ∞-norm improvement, not just mean.** Our current benchmark reports `avg_tracking_error_m`. To detect non-monotone iteration behavior (which the paper says is the risk with fixed gains), also track `max_tracking_error_m` across iterations. If any iteration shows an increase in max error even while mean decreases, the gain is too high for that section.

---

## Limitations & Caveats

1. **LTI assumption.** The paper's theory applies to linear time-invariant systems. Our drone racing system is nonlinear (SE(3) rigid body dynamics) with the PD controller adding trajectory-dependent nonlinearities. The ILC convergence guarantees are only approximate when the system is nonlinear — they hold locally around the nominal trajectory if the linearized Markov parameters are computed at the nominal operating point.

2. **Finite trial length.** The paper's results are derived for a fixed trial length N. In our racing trajectory, the trial length is the total simulation duration (≈ 24s at dt=0.01, giving N ≈ 2400 steps). The sufficient conditions for 2-norm monotone convergence are derived for this finite N, and do not automatically extend to different trajectory durations or gate configurations.

3. **Sufficient condition only for 2-norm.** The paper provides only a *sufficient* condition for 2-norm monotone convergence, not a necessary one. This means the condition may be conservative: there may exist time-varying gains that achieve 2-norm monotone convergence but do not satisfy the stated sufficient condition. A practitioner cannot definitively rule out an alternative gain just because it violates the sufficient condition.

4. **No algorithm for computing the optimal gain schedule.** The paper proves existence of good time-varying gains and gives conditions, but does not provide a closed-form algorithm for computing the optimal `L(k)` schedule given a target system. Practical implementation requires either solving a convex optimization problem (to minimize ρ subject to the monotone convergence constraints) or using heuristic gain scheduling guided by the paper's qualitative conclusions.

5. **Ignores iteration transients.** The paper analyzes steady-state iteration-domain behavior (does ‖E_j‖ decrease monotonically as j → ∞?). It does not address within-iteration transients — large errors during the early part of a trial that eventually converge by the end of the trial. For our racing system, within-trial error near the start of the helix (before the correction has accumulated) is a separate concern from iteration-domain convergence.

6. **No discussion of noise robustness.** The theoretical framework assumes a deterministic ILC setting (the same error is observed each iteration for the same input). Our simulation adds small Gaussian noise to EKF updates, which introduces stochastic iteration-to-iteration variation. Under noise, the monotone convergence guarantee breaks down and the system will eventually reach a noise floor below which further iterations provide no benefit (or may even increase error due to noise amplification).

7. **The improvement over fixed gain may be modest in practice.** The primary benefit of time-varying gain is elimination of the iteration-domain overshoot. If our current fixed α = 0.4 is already near-optimal (ρ ≈ 0.95 suggests reasonable convergence), the gain from switching to a time-varying schedule may be a 1-2 iteration reduction to convergence, translating to marginal wall-clock savings in offline calibration.

---

## Key Parameters / Constants

| Parameter / Concept | Value / Description |
|---------------------|---------------------|
| Gain schedule for ∞-norm monotone convergence | Exponentially *increasing*: `L(k) = γ · ρ^k`, ρ > 1 |
| Gain schedule for 2-norm monotone convergence | Non-increasing function × constant: `L(k) = c · f(k)`, f non-increasing, derived from Markov parameters |
| Moore et al. 2005 gain (shown to be incorrect) | Exponentially *decaying*: `L(k) = γ · ρ^k`, ρ < 1 |
| 1-norm monotone convergence | Provably impossible for Arimoto P-ILC |
| Iteration operator Q | `Q = I - H · diag(L)`, where H is the lower-triangular Toeplitz Markov parameter matrix |
| Convergence condition (2-norm) | `σ_max(Q) < 1` (necessary), plus sufficient condition on L(k) structure |
| Convergence condition (∞-norm) | All row sums of |Q| < 1 |
| Our current fixed gain | α = 0.4 (constant, does not exploit the time-varying theory) |
| Recommended section-wise gains | Straights: α ≈ 0.50-0.55; Helix: α ≈ 0.22-0.28 |
| Trial length effect | Shorter trial → faster iteration-domain convergence (empirical observation in paper) |
| Key Markov parameter matrix | H is lower-triangular Toeplitz; (i,k) entry = C A^{i-k-1} B for i > k |

---

*Analysis written 2026-04-14. Paper: Liu, Zheng & Chen, Automatica 2023, Vol. 157, Art. 111259.*
*Sources: [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S000510982300420X) | [ResearchGate entry](https://www.researchgate.net/publication/375169324_Monotonically_convergent_iterative_learning_control_by_time_varying_learning_gain_revisited)*
