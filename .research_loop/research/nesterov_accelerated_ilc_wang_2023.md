# Fast Data-Driven ILC with Nesterov Acceleration
- **URL**: https://arxiv.org/abs/2312.14326
- **Authors**: Jia Wang, Leander Hemelhof, Ivan Markovsky, Panagiotis Patrinos
- **Year**: 2023
- **Venue**: arXiv (submitted December 21, 2023)

---

## Key Contribution

Wang et al. propose a data-driven iterative learning control (ILC) framework for linear time-invariant systems where the system dynamics are completely unknown and only input-output data is available. The core algorithmic contribution is the integration of Nesterov momentum acceleration into the ILC update rule, replacing the standard gradient-projection step with a two-step momentum extrapolation that achieves O(1/j²) convergence instead of the classical O(1/j) rate — a quadratic improvement in iteration efficiency.

The second major contribution is a non-parametric system representation based on Willems' fundamental lemma and Hankel matrix factorizations. Rather than identifying a parametric model (e.g., ARX, state-space), the method constructs an implicit model of the lifted input-output map G directly from offline trajectory data via SVD-based low-rank approximation. This simultaneously handles measurement noise, unknown system order, and structured output disturbances without requiring a separate identification phase. Together, the non-parametric representation and Nesterov-accelerated update form a complete framework with formal convergence guarantees under bounded output disturbances and box-constrained inputs.

---

## Technical Approach

### Problem Setup

The system is a SISO LTI plant with unknown matrices (A, B, C):

    x_j(k+1) = A x_j(k) + B u_j(k)
    y_j(k)   = C x_j(k) + d_j(k)

where j indexes the trial (ILC iteration), k indexes time within a trial, and d_j(k) is an unknown bounded output disturbance with |d_j(k)| ≤ d̄. Inputs are constrained: u_j(k) ∈ Q := [u̲, ū].

In lifted (batch) form over N timesteps: **y_j = G u_j + c**, where G is a lower-triangular N×N Toeplitz matrix of Markov parameters and c captures initial condition and disturbance contributions. The control objective is:

    min_{u ∈ Q} J(u) = (1/2) ||r - (G u + c)||²

The gradient is ∇J(u) = -Gᵀ(r - Gu - c) = -Gᵀ e_j, where e_j = r - y_j is the tracking error signal for trial j.

### Non-Parametric System Identification via Hankel Matrices

Because G is unknown, it is estimated from offline persistently exciting data of length T+1. A Hankel matrix is constructed from the data and partitioned into past/future/boundary components. The fundamental lemma guarantees that any length-N trajectory of the system lies in the column space of the Hankel matrix. To handle output noise:

1. QR decomposition isolates the input component.
2. SVD of the residual block D₂ yields a rank-n approximation D̄₂ = Σᵢ₌₁ⁿ σᵢ uᵢ vᵢᵀ (retaining only the n largest singular values, where n is the system order).
3. The estimated Toeplitz matrix G̃ is recovered and projected onto lower-triangular (causal) structure via a causality operator P₀.

This produces an inexact oracle: the computed gradient ∇̃J(u_j) = -G̃ᵀ ẽ_j differs from the true gradient by a bounded amount. The authors formalize this as an (ε, δ)-oracle in the sense of Devolder et al. (2014), enabling the convergence analysis to account for both model mismatch and disturbance simultaneously.

### Algorithm 1: Classical Data-Driven ILC (P-type gradient projection)

    Initialize u₀ ∈ Q, Lipschitz constant L = ||G̃||²
    For j = 0, 1, ..., M:
        1. Apply u_j to system, measure perturbed output ỹ_j
        2. Compute error: ẽ_j = r - ỹ_j
        3. Compute inexact gradient: ∇̃J(u_j) = -G̃ᵀ ẽ_j
        4. Projected gradient step:
           u_{j+1} = Π_Q( u_j + (1/L) G̃ᵀ ẽ_j )

This is exactly a P-type ILC update with learning gain L⁻¹ and a saturation projection. The Lipschitz constant L = ||G̃||² plays the role of the inverse learning rate.

**Convergence (Theorem 4.2):** J(u_j) - J(u*) ≤ L||u* - u₀||²/(4j) + δ, where the residual δ = 2Δ₁ + 2Δ₂D captures the irreducible floor due to model error and disturbances (Δ₁, Δ₂ are functions of d̄ and ||G̃ - G||, D is the diameter of Q).

### Algorithm 2: Fast Data-Driven ILC (Nesterov Momentum)

The Nesterov-accelerated variant adds an extrapolation step between projected gradient iterates:

    Initialize μ₀ = y₀ = u₀ ∈ Q, ρ₀ = 1
    For j = 0, 1, ..., M:
        1-3. Same as Algorithm 1: apply y_j, measure ỹ_j, compute ∇̃J(y_j)
        4. Projected gradient step on the extrapolated point:
           μ_j = Π_Q( y_j + (1/L) G̃ᵀ ẽ_j )
        5. Update momentum coefficient:
           ρ_{j+1} = (1 + √(1 + 4ρ_j²)) / 2
        6. Nesterov extrapolation:
           y_{j+1} = μ_j + ((ρ_j - 1)/ρ_{j+1}) (μ_j - μ_{j-1})
        7. Apply y_{j+1} to system next trial

The extrapolation coefficient (ρ_j - 1)/ρ_{j+1} grows toward 1 as j increases, so earlier iterates get little momentum while later iterates get nearly full momentum — exactly the FISTA/Nesterov schedule.

**Convergence (Theorem 4.3):** J(y_j) - J(u*) ≤ 2L||u* - y₀||²/j² + δ

The j² denominator versus j in Algorithm 1 is the key gain: to achieve the same J - J* level, fast ILC needs √(classical_iters) iterations.

### Hybrid Switching Strategy

The paper identifies a practical failure mode of pure Nesterov acceleration under persistent disturbances: the momentum term can accumulate error from noisy gradient estimates and cause oscillation or even divergence in later iterations. To address this, the authors propose a hybrid strategy that monitors the per-iteration objective improvement and switches from the fast ILC (Algorithm 2) back to classical ILC (Algorithm 1) when the improvement stagnates or reverses. The specific switching condition is described as "empirical but effective": if J(y_{j+1}) > J(y_j) + tolerance, reset momentum (set ρ = 1, revert to gradient projection only). This is analogous to the restart heuristics used in FISTA for non-smooth problems (O'Donoghue & Candès 2015). The hybrid scheme preserves the fast convergence rate in early iterations when noise is not yet accumulating but reverts to the monotone O(1/j) classical rate once the disturbance floor is approached.

### Convergence Under Inexact Oracle

The key technical insight is that both algorithms operate under an inexact oracle: the gradient G̃ᵀ ẽ_j is a biased estimate of the true gradient Gᵀ e_j. The bias has two components:

- **Disturbance bias**: ∇̃J(u_j) - ∇J(u_j) includes G̃ᵀ d_j, bounded by ||G̃|| √N d̄
- **Model bias**: (G̃ - G)ᵀ(r - ỹ_j), bounded by ||G̃ - G|| √(2J̄)

The total oracle accuracy Δ₂ = ||G̃ - G||₂ √(2J̄) + ||G̃||₂ √N d̄. As long as this is finite, both algorithms converge to a neighborhood of the optimum of radius proportional to δ = 2Δ₁ + 2Δ₂D.

---

## Results

The paper validates on three cases:

**Toy SISO example**: Classical ILC converges at O(1/j); fast ILC reaches the same error floor in approximately half the number of iterations (visual convergence plots in Figure 2). For a 10-iteration budget, fast ILC achieves the disturbance floor by iteration ~4 while classical ILC requires ~8-9 iterations.

**Batch of random LTI systems**: Over a set of randomly generated stable SISO plants with sinusoidal disturbances of varying frequency and amplitude, the fast ILC consistently outperforms classical ILC when the iteration count is small (j ≤ 10), with the gap being largest in the 3-6 iteration range. Beyond ~15 iterations both methods converge to the same floor.

**High-precision robotic motion system (Case Study)**: Applied to a precision stage with unknown dynamics and persistent periodic disturbance. The data-driven G̃ (estimated from T=200 samples of offline data) reproduces the system well enough for ILC to converge. The fast variant reaches target tracking accuracy in 5 iterations versus ~9 for classical, a ~44% reduction in required trials.

The critical quantitative threshold is the disturbance floor δ, which in the robotic case study is on the order of the measurement noise level. Both algorithms converge to this floor; fast ILC just reaches it faster.

---

## Relevance to Our System

Our ILC (`compute_ilc_offset_table` in `planning/trajectory_optimizer.py`, called from `scripts/benchmark.py`) is structurally a P-type ILC matching Algorithm 1 of this paper with the following correspondence:

| This Paper | Our System |
|---|---|
| u_j (lifted input) | cumulative_offset (N×3 position correction table) |
| e_j = r - y_j (error signal) | cross_track error at each timestep |
| G̃ᵀ e_j (inexact gradient) | sec_alpha * Butterworth-filtered cross_track |
| Π_Q(·) (box projection) | np.clip to sec_max_corr |
| L⁻¹ (step size) | sec_alpha (per-section learning rate) |
| j (ILC iteration) | ilc_iter in range(max_iterations=8) |

The convergence guarantee from Theorem 4.2 directly explains the behavior we observe: after ~5-6 iterations, improvements become marginal (we hit the disturbance floor set by the kinematic sim mismatch with PyBullet). We increased max_iterations from 5→7→8 across iterations 46-48 to push closer to this floor, which is exactly the O(1/j) convergence prediction.

**Nesterov acceleration is directly applicable to our ILC**: replacing our current P-type update with the Algorithm 2 momentum schedule could reach the same convergence floor in ~4-5 iterations instead of 8. This would either (a) allow us to run fewer iterations for same quality (faster offline precompute), or (b) at fixed 8 iterations, achieve lower final error by converging faster and spending more iterations at the floor.

The hybrid switching strategy is also directly relevant: our system already exhibits the symptom of over-correction at high iteration counts (gate-2 over-correction forced alpha reductions in iterations 47-48). The Nesterov restart heuristic would naturally handle this by reverting to classical ILC when the per-section error stops improving, avoiding the manual alpha rebalancing we've been doing.

The inexact oracle framework validates our Q-filter design choice. Our Butterworth Q-filter is essentially a regularizer on the gradient estimate G̃ᵀ e_j: it band-limits the correction to below 0.35-0.46 Hz, which suppresses high-frequency noise that would inflate the oracle error Δ₂. The paper's analysis predicts that reducing Δ₂ (smaller d̄ or better G̃) directly reduces the convergence floor δ, consistent with our observation that tighter filter cutoffs reduce steady-state ILC error.

The non-parametric Hankel/SVD representation is less directly relevant: our system identifies G̃ implicitly through the kinematic simulation (we do not need to learn it from offline data). However, the SVD low-rank projection idea maps to our Butterworth smoothing: both are frequency-domain regularizers that suppress noise amplification in the learning update.

**Affected modules**: `planning/trajectory_optimizer.py` (compute_ilc_offset_table, the ILC inner loop), `scripts/benchmark.py` (ILC call site with section_boundaries and max_iterations).

---

## Actionable Takeaways

1. **Add Nesterov momentum to the per-section ILC update.** In `compute_ilc_offset_table`, maintain a momentum state `mu_prev` per section. After each projected gradient step to `mu_curr = sec_alpha * sec_smoothed`, apply extrapolation: `y_next = mu_curr + ((rho - 1)/rho_next) * (mu_curr - mu_prev)` where rho follows the schedule `rho_{j+1} = (1 + sqrt(1 + 4*rho_j^2)) / 2`. Use `y_next` as the update instead of `mu_curr`. This requires ~10 lines of code per section.

2. **Implement the hybrid restart.** After each ILC iteration, compare the per-section average error to the previous iteration. If a section's error increases by more than a small tolerance (e.g., 0.5%), reset that section's momentum (rho → 1, mu_prev → mu_curr). This prevents gate-2 over-correction without requiring manual alpha reductions.

3. **Reduce max_iterations to 5-6 with momentum.** The O(1/j²) convergence means that 5 Nesterov iterations should achieve roughly the same result as 8-9 classical iterations. This cuts ILC precompute time by ~38%.

4. **Tune step size via Lipschitz constant.** The paper sets L = ||G̃||² (spectral norm of the estimated system matrix). For our kinematic sim, G is implicitly defined by the PD controller dynamics. We can estimate an effective L by computing the maximum response amplitude of our controller to a unit impulse offset and use L⁻¹ = 1/L as the base learning rate before per-section alpha scaling.

5. **Apply the oracle accuracy formula to set Q-filter cutoff.** The convergence floor scales with Δ₂ = ||G̃ - G|| √(2J̄) + ||G̃|| √N d̄. Our kinematic sim mismatch ||G̃ - G|| is fixed, but we control N (trajectory length) and d̄ (effective disturbance via filter cutoff). Tightening filter cutoff reduces d̄ at the cost of slower learning. The current 0.35-0.46 Hz values appear near-optimal given our 8-iteration budget; with Nesterov, we can afford slightly tighter cutoffs (less d̄, lower floor) since faster convergence compensates.

6. **Per-section momentum state.** Since we already have independent per-section offset arrays (`section_offsets[sec_idx]`), momentum state can be maintained independently per section as `rho[sec_idx]` and `mu_prev[sec_idx]`. This preserves the cross-contamination isolation property of our per-section design.

7. **Evaluate whether 8 iterations are still needed.** After implementing Nesterov, run the benchmark with max_iterations=5 and compare. If the error floor is the same, use 5 to save compute. If 5 is insufficient, keep 8 for extra margin.

---

## Limitations & Caveats

**SISO assumption.** The paper is formalized for single-input single-output systems with scalar input u(k) ∈ [u̲, ū]. Our ILC operates on a 3D vector offset (x, y, z) — effectively a MIMO system. The convergence theorems require re-derivation for the vector case, though the algorithm structure generalizes directly (replace box projection with element-wise clipping, which our code already does).

**Linear time-invariant assumption.** The paper's convergence analysis assumes G is constant (LTI). Our drone dynamics are nonlinear and time-varying (speed-dependent drag, throttle saturation). The kinematic sim used in our ILC is a linearization. The paper's framework applies approximately, but the disturbance floor δ will be larger than theory predicts due to the nonlinear residuals not captured by G̃.

**Trial-to-trial reproducibility.** ILC requires that each trial starts from the same initial condition and follows the same reference. The paper assumes d_j(k) is bounded but not necessarily the same across trials. In our case the ILC is offline (kinematic sim is deterministic), so this is trivially satisfied. However, if we ever move to online ILC (learning across real flight attempts), trial-to-trial repeatability would need to be assessed.

**Nesterov momentum instability under non-monotone disturbances.** The paper notes that Nesterov momentum can diverge if the gradient noise is non-stationary. Our Q-filter suppresses this, but the restart heuristic is still critical. Without restarts, momentum accumulation at later iterations can produce corrections larger than the per-section max_correction_m limit — exactly the gate-2 overshoot we observed with high alpha values.

**Finite horizon / non-repetitive task.** Classical ILC convergence theory assumes the same reference trajectory is repeated indefinitely. We use it to pre-correct a single trajectory offline (8 iterations in simulation), not for true repetitive task learning. This means we can use only 8 iterations of "experience," making the quadratic convergence improvement of Nesterov disproportionately valuable in our setting.

**Box constraint assumption.** The paper's projection is onto a simple box Q = [u̲, ū]. Our constraint is a magnitude clip (||offset|| ≤ max_correction_m), which is an L2 ball, not a box. The projection operator is different (normalization vs. element-wise clipping). The convergence theorems apply to any closed convex set, so the qualitative results hold, but the Lipschitz constant L may differ.

**Disturbance floor is irreducible.** The term δ = 2Δ₁ + 2Δ₂D cannot be driven to zero regardless of how many iterations are used, as long as the kinematic sim differs from PyBullet. More Nesterov iterations beyond the floor do not help (and can hurt via momentum overshoot). The restart heuristic and convergence_threshold stopping criterion already handle this correctly.

---

## Key Parameters / Constants

| Parameter | Paper Symbol | Value / Formula | Our Analogue |
|---|---|---|---|
| Lipschitz constant | L | ||G̃||²₂ (spectral norm squared) | ~1/sec_alpha |
| Step size (learning rate) | 1/L | scalar | sec_alpha (0.30–0.50 currently) |
| Momentum sequence | ρ_{j+1} | (1 + √(1 + 4ρ_j²)) / 2, ρ_0 = 1 | Not currently used |
| Extrapolation coefficient | (ρ_j - 1)/ρ_{j+1} | → 1 as j → ∞ | Not currently used |
| Oracle accuracy (function) | Δ₁ | √(2J̄N) d̄ + (N/2)d̄² | Implicit in Q-filter cutoff |
| Oracle accuracy (gradient) | Δ₂ | ||G̃-G|| √(2J̄) + ||G̃|| √N d̄ | Kinematic sim mismatch + filter |
| Disturbance bound | d̄ | Max disturbance magnitude | ~PyBullet mismatch amplitude |
| Offline data length | T | ≥ n + N - 1 (persistency of excitation) | Not applicable (sim-based) |
| System order | n | Used in SVD truncation rank | Not applicable |
| Convergence floor | δ | 2Δ₁ + 2Δ₂D | ~0.002 m (our convergence_threshold) |
| Restart tolerance | — | Not specified numerically; described as "empirical" | Suggest: 0.5% per-section error increase |
| Classical convergence rate | — | O(1/j) | Matches our observed 8-iteration diminishing returns |
| Nesterov convergence rate | — | O(1/j²) | Expected speedup: reach floor in ~4-5 iters |

**Practical Nesterov schedule** (first 10 iterations):

| j | ρ_j | extrapolation coeff (ρ_{j-1}-1)/ρ_j |
|---|---|---|
| 0 | 1.000 | 0.000 |
| 1 | 1.618 | 0.000 |
| 2 | 2.058 | 0.376 |
| 3 | 2.414 | 0.508 |
| 4 | 2.732 | 0.564 |
| 5 | 3.027 | 0.599 |
| 6 | 3.303 | 0.622 |
| 7 | 3.565 | 0.638 |
| 8 | 3.817 | 0.651 |

This shows that for j=1 (first Nesterov step), there is no momentum contribution — this is safe. Momentum builds gradually to ~0.65 by iteration 8, which is the regime our current 8-iteration budget operates in.
