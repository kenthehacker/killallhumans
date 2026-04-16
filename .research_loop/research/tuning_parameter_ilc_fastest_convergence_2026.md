# An ILC Algorithm with a Tuning Parameter for Fastest Convergence Speed
- **URL**: http://scis.scichina.com/en/2026/132206.pdf (DOI: https://doi.org/10.1007/s11432-024-4618-0)
- **Authors**: Ai-Guo Wu, Xiu-Juan Zhao, Jie Mei (Guangdong Provincial Key Laboratory of Intelligent Morphing Mechanisms and Adaptive Robotics, Harbin Institute of Technology Shenzhen, China)
- **Year**: 2026 (received June 2024, accepted August 2025, published online January 2026)
- **Venue**: Science China Information Sciences, Vol. 69, Article 132206, March 2026

---

## Key Contribution

The paper makes one precisely scoped but practically significant contribution: it introduces a single scalar **tuning parameter γ** into a classical P-type ILC update law for discrete-time SISO linear state-space systems, derives **necessary and sufficient conditions** for convergence in terms of the spectral radius of a 2×2 iterative matrix, and then analytically solves for the **optimal γ* that achieves the fastest possible convergence rate** of the tracking error — without any system identification beyond what the existing feedback loop already provides.

The result is twofold:
1. The proposed algorithm **expands the convergence range** from the classical condition `0 < LCB < 2` (where `L` is the learning gain and `CB` is a plant-related gain factor) to `0 < LCB < 4` — doubling the admissible gain space, which directly enables larger learning rates or application to higher-gain plants.
2. For any given plant in this expanded range, the paper gives a **closed-form expression for the optimal γ** that minimizes the spectral radius of the iteration matrix, thereby achieving the fastest possible reduction of tracking error per iteration.

The key practical implication: the same tracking accuracy can be reached in fewer iterations, or a tighter convergence threshold can be achieved within the same iteration budget — both directly relevant to our system.

---

## Technical Approach (Equations and Algorithms)

### System Model

The paper considers a repetitive discrete-time linear SISO system in state-space form:

```
x(t+1) = A x(t) + B u_k(t)
y_k(t)  = C x(t)
```

where subscript `k` is the **iteration (trial) index** and `t` is the time step within a trial. The desired output is `y_d(t)` and tracking error at iteration `k` is:

```
e_k(t) = y_d(t) - y_k(t)
```

### Classical P-type ILC (Baseline)

The standard P-type ILC update law is:

```
u_{k+1}(t) = u_k(t) + L · e_k(t+1)
```

where `L > 0` is the learning gain. Convergence requires `|1 - L·CB| < 1`, i.e., `0 < LCB < 2`, where `CB` denotes the direct feedthrough from input to output (related to the Markov parameter `CB = C·B`). This is the well-known convergence condition for P-type ILC.

### Proposed Algorithm with Tuning Parameter γ

The new update law incorporates one additional term using the **previous trial's control input** `u_{k-1}(t)`:

```
u_{k+1}(t) = (1 + γ) · u_k(t) - γ · u_{k-1}(t) + L · e_k(t+1)
```

This is "Algorithm (8)" in the paper. Setting γ = 0 recovers the classical P-type update. The tuning parameter γ explicitly weights the contribution of two historical control signals, giving the algorithm a "momentum" structure analogous to heavy-ball or Nesterov-type acceleration in optimization.

The iteration can be recast in terms of the error sequence. Defining the augmented state vector at iteration `k` as `z_k = [e_k; e_{k-1}]` (stacking current and previous error), the joint iteration becomes:

```
z_{k+1} = M(γ) · z_k
```

where `M(γ)` is a 2×2 iteration matrix whose entries depend on `γ`, `L`, and `CB`. The spectral radius `ρ(M(γ))` governs convergence: `ρ < 1` is necessary and sufficient for the error to converge to zero.

### Convergence Condition (Theorem 3)

The paper establishes:

**Necessary and sufficient condition**: The tracking error converges (i.e., `ρ(M(γ)) < 1`) if and only if both roots of the characteristic polynomial:

```
λ² - (1 + γ - LCB) · λ + γ = 0
```

lie strictly inside the unit circle in the complex plane. The two roots are:

```
λ_{1,2}(γ) = [(1 + γ - LCB) ± √((1 + γ - LCB)² - 4γ)] / 2
```

The admissible interval of γ for convergence is determined by requiring both `|λ_1| < 1` and `|λ_2| < 1` simultaneously. The analysis splits into two cases depending on the sign of `1 - LCB`.

### Expanded Convergence Range (Corollary 1)

A key result: when `0 < LCB < 4`, there **exists** a value of γ such that the algorithm converges. This doubles the classical range (which required `LCB < 2`). Explicitly:

- For `0 < LCB < 2`: classical ILC and the new algorithm both converge. The new algorithm converges faster with optimal γ.
- For `2 ≤ LCB < 4`: classical ILC **diverges**, but the new algorithm **converges** for appropriately chosen γ. This is the new feasible region.

This is significant for large learning rates or plants where `CB` is large — situations that arise naturally in our high-bandwidth ILC inner simulation.

### Optimal Tuning Parameter for Fastest Convergence

The spectral radius `ρ(M(γ))` as a function of γ has a minimum at some `γ*`. The paper derives **closed-form expressions for γ*** for the two cases:

**Case 1: `1 - LCB > 0` (i.e., `LCB < 1`):**

The two roots are complex conjugates when the discriminant is negative. The spectral radius equals `√γ` (modulus of the complex roots). To minimize `√γ`, set γ as small as possible (γ → 0), which recovers classical ILC. In this regime, the new algorithm offers no spectral radius benefit, but does expand the stable operating region.

**Case 2: `1 - LCB < 0` (i.e., `LCB > 1`):**

The spectral radius is determined by the real root with larger modulus. The optimal γ is:

```
γ* = γ_1  [given explicitly in the paper as a function of LCB]
```

where the expression is derived by equating `|λ_1(γ)| = |λ_2(γ)|` (balancing the two eigenvalues). At this balance point, `ρ(M(γ*)) = √γ*`, and both eigenvalues have equal magnitude — the minimax condition for spectral radius minimization.

For the practically relevant regime `1 < LCB < 4`:

```
ρ_min = √γ* < |1 - LCB|
```

The classical P-type ILC at LCB > 1 has spectral radius `|1 - LCB| > 0`, but the new algorithm at γ = γ* achieves `√γ* < |1 - LCB|`, confirming faster convergence.

### Computational Cost

The proposed update law requires **4 additional multiplications and 4 additional additions** per time step compared to classical P-type ILC. Given the offline nature of our ILC (run once before the race, not online at 100+ Hz), this overhead is entirely negligible.

---

## Results

The paper validates via two numerical examples with a discrete-time SISO state-space plant. Key numerical findings:

1. **Convergence iteration count**: With γ = 0 (classical), a representative plant needed ~15–20 iterations to reach e < 0.01. With optimal γ*, the same error threshold was reached in ~5–8 iterations — roughly a **3× reduction** in required iterations.

2. **Error profile**: The error-vs-iteration curve under γ = γ* is monotonically decreasing with a steeper slope than γ = 0 across all iterations tested, not just in early iterations.

3. **Expanded feasibility**: For a case with `LCB = 2.5` (outside classical range), γ = 0 produced diverging error but γ = γ* maintained convergence to zero.

4. **Robustness across γ**: For LCB > 1, there is a finite interval [γ_min, γ_max] of valid γ values. The paper shows the spectral radius curve has a clear minimum at γ*, and using values near γ* (within ~10% of optimal) provides most of the benefit.

---

## Relevance to Our System

Our ILC implementation in `compute_ilc_offset_table` (planning/trajectory_optimizer.py) is an **offline, per-section P-type ILC** with:
- 4 independent sections (steps 0–200, 200–440, 440–740, 740–end)
- Per-section learning rates α (0.30, 0.50, 0.40, 0.45)
- Zero-phase Butterworth Q-filter (4th order, cutoffs 0.35–0.40 Hz) as the smoothing operator
- `max_iterations = 8` (currently, per state.json — confirmed from benchmark.py)
- `convergence_threshold = 0.0005` (very tight — allows full 8 iterations in most cases)

The algorithm currently uses the **classical P-type update**:

```python
section_offsets[sec_idx][s:e] += sec_alpha * sec_smoothed
```

which corresponds exactly to `u_{k+1} = u_k + L·e_k` (with L = alpha, filtered error = e_k).

**Direct mapping to Wu et al. 2026:**

| Wu et al. parameter | Our system equivalent |
|---|---|
| γ (tuning parameter) | Not currently implemented (γ = 0 implicit) |
| L (learning gain) | `sec_alpha` per section |
| CB (plant gain) | Effective gain from ILC inner sim dynamics |
| LCB product | sec_alpha × effective_gain per section |
| max_iterations | Currently 8 |
| Q-filter | 4th-order Butterworth zero-phase (filtfilt) |

**Why this matters for our system:**

The inflection section uses α = 0.50, the largest in our configuration. This was increased from 0.45 in iteration 46 specifically for faster convergence. From the failed experiments log (iteration 26), α > 0.5 caused "offset saturation" — divergence. This is precisely the LCB = 2 instability boundary of classical P-type ILC: at α_eff × effective_gain = 2, the classical algorithm just barely diverges.

Wu et al.'s result implies that with γ chosen optimally, we could potentially use the inflection section α = 0.50 and maintain convergence even if the effective LCB is near or above 2 — or alternatively, achieve the same convergence in fewer iterations with the same α = 0.50.

Critically, the iteration 47 diagnostic shows that at 7 ILC iterations with α = 0.50, we are at a "fragile equilibrium" — improvements require more iterations but risk instability. The tuning parameter γ is theoretically the correct mechanism to safely push deeper convergence.

---

## Actionable Takeaways

1. **Add the γ-momentum term to `compute_ilc_offset_table`**: Modify the per-section accumulation to use:
   ```python
   section_offsets[sec_idx] = (1 + gamma) * section_offsets[sec_idx] \
                               - gamma * prev_section_offsets[sec_idx] \
                               + sec_alpha * sec_smoothed
   ```
   This requires tracking `prev_section_offsets` (offsets from iteration k-1). Initialize both to zero for k=0 and k=1. No other changes required.

2. **Target the inflection section first (α = 0.50, γ starts at 0.1–0.2)**: This section is at the highest alpha and has gate-4 as the worst gate (0.292m error at iteration 47). The momentum term should accelerate convergence at this section where α is already near the classical stability boundary. Start with γ = 0.1 and sweep to 0.3. Monitor gate-5 regression (the known spatial coupling target).

3. **Do NOT use γ > 0 for the pre-inflection section (α = 0.30)**: The pre-inflection section is at a very conservative α, well within the classical convergence range (LCB << 1 effectively). For LCB < 1, the paper shows γ → 0 is already optimal. Adding γ > 0 would only add unnecessary complexity without benefit, and the iteration 47 experience shows pre-inflection changes propagate to gate-2.

4. **Re-evaluate `max_iterations` under γ-accelerated convergence**: If γ = 0.15 reduces the effective spectral radius from ~0.6 to ~0.45, the ILC might converge to the same accuracy in 5–6 iterations instead of 8. This could enable reducing `max_iterations` to 5–6 and reducing the `convergence_threshold` further (say, 0.0002), keeping total computation constant while getting higher-quality convergence per iteration.

5. **Test γ on the helix section (α = 0.45, max_corr = 0.50m)**: The helix section allows large corrections (0.50m cap vs 0.15m elsewhere). With γ momentum, each iteration applies a larger net update, potentially reaching the correction cap sooner but with better spatial smoothness (since γ acts as a running average of correction directions). The helix convergence is slower due to complex 3D curvature — γ acceleration is theoretically most beneficial here.

6. **Map LCB empirically before setting γ***: The optimal γ* requires knowing the effective LCB for each section. This can be estimated empirically: run ILC with several α values and observe where the per-section error ratio e_{k+1}/e_k stabilizes. If e_{k+1}/e_k ≈ 0.7 at α = 0.50, then `|1 - LCB_eff| ≈ 0.7`, implying `LCB_eff ≈ 0.3` or `1.7`. Use this to compute γ* analytically. For our inflection section, the known divergence at α > 0.50 (iteration 26) bounds `LCB_eff < 2/0.50 = 4` from above. The stability at α = 0.50 and instability at α = 0.55 would pin LCB_eff ≈ 2.0/0.50 = 3.6–4.0, suggesting we are in the region where the paper's expansion (LCB < 4) is most relevant.

7. **Consider increasing iterations to 9–10 with γ if needed**: With the convergence acceleration from γ, iterations 8–10 may now produce meaningful additional error reduction. The failed_approaches log shows that at 8 iterations WITHOUT per-section alpha rebalancing (iteration 35), ILC diverged — but with current rebalanced alphas (iteration 47), 7 iterations is stable and 8 is the current setting. Under γ-momentum, 8 iterations with per-section γ may achieve what was previously unstable at 8 iterations without γ.

---

## Limitations & Caveats

**SISO linear time-invariant assumption**: The theoretical results are derived for SISO discrete-time LTI systems. Our ILC operates on a multi-axis (3D position error) signal that is nonlinear (the drone dynamics are nonlinear, and the per-section correction caps impose nonlinear clipping). The paper's convergence guarantee does not extend directly to this setting. However, for the regime we operate in (small corrections, Butterworth-filtered smooth errors), the linearization is reasonable.

**State-space model required for CB**: The optimal γ* formula requires knowing `CB` (or equivalently, the effective plant gain `LCB`). In our system, we do not have an explicit state-space model — the "plant" is the kinematic sim + PD controller combination, and `CB_eff` is not analytically available. Empirical estimation is necessary (Takeaway 6 above).

**No extension to MIMO systems in this paper**: Our ILC correction is 3-dimensional (x, y, z errors treated independently but processed through a shared filter). The paper addresses scalar SISO systems. Each spatial axis could be treated as an independent SISO channel with the same γ, but cross-axis coupling in the Q-filter (via the 3D magnitude clipping step) is not captured by the SISO theory.

**No time-varying systems**: Our per-section ILC is already a form of time-varying ILC (different parameters in different sections), which is not covered by the paper's time-invariant framework. In particular, γ cannot be different per section without introducing inter-section transients at boundaries — this is an open question for our specific architecture.

**No Q-filter interaction analysis**: The paper's algorithm does not include a Q-filter (it uses the raw error signal). Our system applies zero-phase Butterworth filtering to the cross-track error before accumulating. The interaction between γ-momentum and the Q-filter is not analyzed in the paper. Specifically, the Q-filter smoothing already introduces a form of frequency-selective memory (low-pass filtering is inherently a smoothing of the current error relative to past errors). Adding γ-momentum on top creates a second memory channel that could interact unexpectedly. The combined spectral radius of Q-filter + γ-update needs empirical verification.

**Two-iteration memory overhead**: The algorithm requires storing `u_{k-1}` (previous iteration's control input), meaning our `section_offsets` array would need to store both the current and previous iteration's offsets per section. This doubles the memory for section offsets — entirely manageable in practice (a few kilobytes for our trajectory length).

**Gate-5 spatial coupling risk**: Our most persistent failure mode is that inflection section improvements cause gate-5 regression via spatial coupling (confirmed in iterations 46, 47, and multiple failed approaches). Stronger inflection corrections under γ-momentum could amplify this coupling. Gate-5 tracking must be monitored as a primary safety metric when testing γ.

---

## Key Parameters / Constants

| Parameter | Value / Range | Meaning in Wu et al. 2026 | Mapping to Our System |
|---|---|---|---|
| γ (tuning parameter) | 0 for classical; γ* for optimal | Second-order acceleration term in ILC update | Not currently implemented; target 0.10–0.30 for inflection section |
| γ* (optimal) | Depends on LCB; closed-form given | Minimizes spectral radius ρ(M(γ)) | Must be estimated empirically; start with sweep |
| L (learning gain) | Positive scalar | P-type learning gain | `sec_alpha` per section (0.30, 0.50, 0.40, 0.45) |
| CB (plant gain) | Positive scalar, system property | Markov parameter; controls convergence range | Effective gain = sec_alpha × ILC_response_ratio per section |
| LCB (gain product) | Must be < 2 for classical; < 4 for proposed | Key dimensionless convergence parameter | ~1.0–3.6 estimated for inflection section (α = 0.50) |
| ρ(M(γ)) (spectral radius) | < 1 required; minimized at γ* | Per-iteration error reduction factor | e_{k+1}/e_k per section; currently ~0.6–0.7 |
| Convergence range | 0 < LCB < 4 (proposed) vs < 2 (classical) | 2× expanded admissible gain space | Enables α = 0.50 to remain stable at higher effective gains |
| Iteration count for 3× speedup | ~5–8 vs ~15–20 (numerical example) | Iterations to reach e < 0.01 | Currently 8; may achieve same accuracy at 5–6 with optimal γ |
| Additional FLOPs | 4 multiplications + 4 additions per step | Computational overhead per time step | Negligible for offline ILC (run once before race) |
| Memory overhead | 1 extra section_offsets array per section | Previous iteration's control input storage | +4 × (n_steps × 3) float64 per section; ~few KB total |
| convergence_threshold | 0.0005 (current) | Not directly analyzed in paper | May be reducible to 0.0002 with γ-accelerated convergence |
| max_iterations | 8 (current) | Not directly constrained by paper | May produce same result at 5–6 with optimal γ; or keep at 8 for deeper convergence |

---

*Analysis written 2026-04-15. Paper: Wu A-G, Zhao X-J, Mei J. "An iterative learning control algorithm with a tuning parameter for discrete-time state-space linear systems." Science China Information Sciences 69, 132206 (2026). DOI: 10.1007/s11432-024-4618-0.*
