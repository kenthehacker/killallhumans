# A Frequency-Domain Approach for Enhanced Performance and Task Flexibility in Finite-Time ILC
- **URL**: https://arxiv.org/abs/2403.02039
- **Authors**: Max van Haren, Kentaro Tsurumoto, Masahiro Mae, Lennart Blanken, Wataru Ohnishi, Tom Oomen
- **Year**: 2024
- **Venue**: ECC 2024 (European Control Conference)

---

## Key Contribution

This paper resolves a long-standing tension in Iterative Learning Control (ILC) between two dominant design paradigms:

1. **Frequency-domain ILC** — intuitive filter design using loop-shaping techniques (learning filter, robustness filter), high performance in terms of convergence speed and achievable steady-state error, but inflexible when the reference task changes between trials.

2. **Basis function ILC** (norm-optimal / task-flexible) — parameterizes the feedforward signal as a weighted sum of basis functions (e.g., reference derivatives), which generalizes across tasks and can leverage prior trials, but gives up the frequency-design transparency.

The key insight is that these two paradigms are **not mutually exclusive**. The authors derive Theorem 1, which identifies the exact norm-optimal cost function weighting matrices that **recover** the frequency-domain ILC update law. Having made this equivalence explicit, they then construct an overparameterized feedforward signal that blends both:

```
f_j = ψ θ_j + f_j^f
```

where `ψ` is the basis function matrix, `θ_j` the task-flexible parameters, and `f_j^f` the frequency-domain component. The combined update is solved as a single norm-optimal optimization, and the weighting matrices are set to deliberately steer basis functions toward reference-correlated (generalizable) content while leaving residual high-frequency, system-dynamics-specific errors to the frequency-domain filter.

---

## Technical Approach

### Frequency-Domain ILC and its Finite-Time Realization

The infinite-time frequency-domain update law is:

```
F_{j+1}^f(z) = Q^f(z) (F_j^f(z) + α L^f(z) E_j(z))
```

where:
- `Q^f(z)` is the **robustness filter** — zero-phase low-pass, suppresses noise and ensures convergence at high frequencies where the model is inaccurate.
- `L^f(z)` is the **learning filter** — designed as the ZPETC (Zero Phase Error Tracking Controller) approximation of the inverse plant sensitivity, converting error to a feedforward correction.
- `α` is the scalar learning gain.
- `E_j(z)` is the error spectrum on trial `j`.

The finite-time implementation replaces these filters with **convolution matrices** `Q^f` and `L^f` acting on the finite-length error vector `e_j`. This makes the update law a standard matrix equation amenable to norm-optimal analysis.

### Theorem 1: Norm-Optimal Equivalence

The central theoretical result shows that minimizing the norm-optimal cost:

```
V_j = e_j^T W_e^f e_j + f_j^{fT} W_f^f f_j^f + Δf_j^{fT} W_{Δf}^f Δf_j^f
```

with the specific weighting matrices:

```
W_e^f   = α (L^f)^T L^f
W_f^f   = (Q^f)^{-1} - I
W_{Δf}^f = (1 - α) I
```

is **equivalent** to the frequency-domain update law. This is not just an approximation — it is an exact algebraic identity. The implication is that all the well-understood frequency-domain design tools (Bode plots, loop-shaping, bandwidth selection) now have a direct mapping into the norm-optimal cost function.

### Basis Function Integration

The overparameterized signal `f_j = ψ θ_j + f_j^f` is optimized jointly. The weighting matrices are designed so that `W_θ` (the cost on basis function weights `θ_j`) is much smaller than `W_f^f`, so the optimizer preferentially uses basis functions for reference-dependent content and reserves the frequency-domain component for residual dynamics. This separation is enforced structurally, not heuristically.

Basis functions in the experiment are chosen as `ψ = [r̈, r‴, r⁴]` — second, third, and fourth derivatives of the reference trajectory — which correspond to inertia, jerk, and snap feedforward terms. This is physically motivated: a well-modeled linear system has feedforward that is a linear combination of reference derivatives.

### Convergence Condition

The combined system converges if and only if for all frequencies ω:

```
|Q^f(e^{jω}) (1 - α J(e^{jω}) L^f(e^{jω}))| < 1
```

where `J(e^{jω})` is the true plant. This is the standard frequency-domain ILC convergence condition: the robustness filter `Q^f` must attenuate frequencies where the learning `α L^f J` does not perfectly cancel the system. Zero-phase design of `Q^f` (a Butterworth filter applied forward and backward) ensures the weighting matrices remain symmetric and positive definite, which is a technical requirement for the norm-optimal equivalence.

### Experimental Validation

The benchmark system is a two-mass-spring-damper with deliberate model mismatch (model mass 25% heavier, model stiffness 80% too high). Trials 1-10 use a fourth-order polynomial reference (N=229 samples at 10 ms). Trials 11-20 switch to a different reference. Parameters:

- Feedback controller bandwidth: 10 Hz
- Robustness filter: zero-phase 2nd-order Butterworth at 40 Hz cutoff
- Learning filter: ZPETC-based
- Learning gain: `α = 1`
- Basis functions: `[r̈, r‴, r⁴]`

The combined method achieves lower error than either pure approach over trials 1-10, and when the reference changes at trial 11, the basis function component retains task-transferable content (inverse inertia estimate), while the frequency-domain component is re-learned from scratch — giving better initial performance on the new task than pure frequency-domain ILC.

---

## Results

- Convergence is faster and to a lower error floor than either pure frequency-domain ILC or pure basis-function ILC alone.
- After a reference change, the combined method benefits from the retained basis function estimates (the `θ` vector does not need to be reset), while the frequency-domain component adapts.
- The estimated basis function weights (Figure 9 in the paper) converge to values consistent with the true inverse dynamics of the plant, confirming that the method is learning physically meaningful feedforward structure.
- The approach is practical: it requires only standard Bode/loop-shaping design for `Q^f` and `L^f`, plus a choice of basis functions, with no exotic optimization machinery.

---

## Relevance to Our System

Our system uses offline P-type ILC to pre-compute a position-offset correction table before each race. The implementation in `planning/trajectory_optimizer.py` (`compute_ilc_offset_table`) runs 3-5 iterations of:

1. Simulate the PD controller tracking the polynomial trajectory (with current offset applied).
2. Compute cross-track error at each timestep.
3. Smooth the error with a Gaussian kernel (`sigma=10` steps = 100 ms at dt=0.01).
4. Clip to `max_correction=0.15 m`.
5. Accumulate: `offset += alpha * smoothed_cross_track_error`, with `alpha=0.4`.

This is exactly the P-type (proportional-type) frequency-domain ILC in time-domain clothing. The Gaussian smoothing plays the role of the robustness filter `Q^f` — it attenuates high-frequency content in the error signal before incorporating it as a correction, preventing over-fitting to noise. The scalar `alpha` maps directly to the learning gain in `F_{j+1} = Q(F_j + α L E_j)`, with our implicit learning filter `L` being the identity (we trust the error directly, without phase correction).

**Key gaps this paper illuminates:**

1. **No ZPETC / phase-correcting learning filter**: Our current `L = I` means we apply corrections without accounting for the PD controller's phase lag. At higher frequencies (tight corners, gate approach), the controller's response is delayed, so the error we observe at time `t` was caused by a reference mismatch at time `t - τ`, where `τ` is the lag. Applying the correction at `t` rather than `t - τ` can cause the ILC to be self-defeating near bandwidth limits.

2. **Fixed Gaussian as robustness filter**: The 100 ms Gaussian is frequency-agnostic. Loop-shaping would let us explicitly pass corrections up to, say, 5 Hz (our controller bandwidth) and hard-suppress above 10 Hz. This would reduce iterations needed for convergence and prevent high-frequency oscillation in the offset table.

3. **No basis function generalization**: Our offset table is trajectory-specific and discarded after each run. If we parameterized the feedforward correction as a linear combination of `[r̈, r‴, r⁴]` (reference acceleration, jerk, snap), the learned weights would encode model mismatch (effective mass, drag) and transfer across gates and speed profiles. This is especially valuable for our multi-gate course where similar dynamics appear at different gates.

4. **Along-track correction is zero**: We deliberately zero out along-track corrections to avoid changing timing. This is correct but means we cannot correct for systematic lag (the drone always lags the trajectory temporally). A frequency-domain framework would separate spatial and temporal errors naturally.

---

## Actionable Takeaways

1. **Add a phase-lead pre-filter (ZPETC approximation)** as the learning filter `L^f`. Before accumulating a correction, shift the error signal backward by approximately `τ = 1 / (2 * controller_bandwidth)` steps using a simple lead filter or fractional delay. For our `kp=6, kd=4` PD gains, the closed-loop bandwidth is roughly 3-5 Hz; a shift of 1-2 timesteps (10-20 ms) should partially compensate the plant phase.

   Implementation: replace `cumulative_offset += alpha * smoothed` with `cumulative_offset += alpha * gaussian_filter1d(shifted_cross_track, sigma)` where the shift is applied before Gaussian smoothing.

2. **Replace Gaussian with a proper band-limited robustness filter**: Design a zero-phase Butterworth low-pass at `fc = controller_bandwidth / 2` (e.g., 2-3 Hz = period 20-30 timesteps at dt=0.01). Apply with `scipy.signal.filtfilt` for zero-phase effect. This is more principled than tuning `sigma` empirically and maps directly to frequency-domain convergence guarantees.

3. **Parameterize corrections as basis functions**: Fit the learned offset table to `θ = argmin ||ψ θ - offset_table||^2` where `ψ` contains columns `[r̈, r‴, r⁴]` evaluated at each timestep. Store `θ` and reconstruct the offset as `ψ θ`. This compresses the table and makes the physics interpretable (e.g., if `θ[0]` is large, we are under-modeling effective mass).

4. **Set `alpha = 1` and rely on the robustness filter for stability**: The paper uses `α = 1` — full-step learning — because `Q^f` is doing the stability work. Our current `α = 0.4` is compensating for the lack of a proper robustness filter. With a well-designed `Q^f`, we should be able to converge in 2-3 iterations instead of 5.

5. **Monitor convergence in the frequency domain**: Compute the FFT of the cross-track error vector before and after each ILC iteration. If the error power is not decreasing at the frequencies where `Q^f` passes, the learning filter `L^f` needs adjustment. This gives a much more diagnostic view than scalar average error.

---

## Limitations & Caveats

- The paper validates on a linear, second-order mechanical system. Our drone PD controller is nonlinear (clipping, drag), and the ILC sim is kinematic. The frequency-domain convergence guarantees assume LTI (linear time-invariant) dynamics, so they are approximate for our use case.
- The basis function approach works best when the reference is a polynomial or smooth function — which is exactly our case (min-snap trajectories are polynomial). This is a favorable match.
- ZPETC requires a model of the plant. Our kinematic sim is a rough model; the ZPETC-derived learning filter will only be as good as this model. However, even an approximate phase lead should help compared to `L = I`.
- The paper does not address the case where the plant is simulated (open-loop ILC), only feedback-controlled trials. Our offline ILC with a deterministic kinematic sim is closer to the feedback case (the PD controller is in the loop), so the analysis applies.
- `N = 229` samples in the benchmark is much shorter than our trajectory (typically 3000-6000 steps at dt=0.01 for a 30-60 s race). The computational cost of the finite-time matrix formulation (`Q^f` as a dense N×N matrix) would be prohibitive. The filtfilt implementation of the robustness filter scales as O(N) and is preferred.

---

## Key Parameters / Constants

| Symbol | Paper Value | Description |
|--------|------------|-------------|
| `α` | 1.0 | Learning gain (scalar) |
| `fc` (robustness filter) | 40 Hz | Cutoff for zero-phase Butterworth |
| `N` | 229 samples | Trial length |
| `dt` (feedback) | 10 ms | Feedback sample period |
| Basis functions | `[r̈, r‴, r⁴]` | 2nd, 3rd, 4th reference derivatives |
| `W_e^f` | `α (L^f)^T L^f` | Error weight in norm-optimal equiv. |
| `W_f^f` | `(Q^f)^{-1} - I` | Signal regularization weight |
| `W_{Δf}^f` | `(1 - α) I` | Update increment weight |
| Benchmark system | 2-mass-spring-damper | m₁_true=0.072kg, k_true=1000 N/m |
| Model mismatch | m₁_model=0.09kg, k_model=1800 N/m | ~25% mass, ~80% stiffness error |

**Our current parameters for comparison:**

| Parameter | Our Value | Recommended Change |
|-----------|-----------|-------------------|
| `alpha` | 0.4 | Increase to 0.8-1.0 with proper Q^f |
| Robustness filter | Gaussian sigma=10 steps (100 ms) | Zero-phase Butterworth, fc=2-3 Hz |
| Learning filter | Identity (L=I) | ZPETC with ~1-2 step lead |
| Max iterations | 5 | 2-3 (should converge faster with alpha=1) |
| max_correction | 0.15 m | Unchanged (physically motivated) |
| Parameterization | Dense offset table (N×3) | Basis function weights (3×3 matrix) |
