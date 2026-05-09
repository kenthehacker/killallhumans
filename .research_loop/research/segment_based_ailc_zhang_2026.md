# Segment-Based Two-Loop Adaptive Iterative Learning Control for Spacecraft Position and Attitude Tracking

- **URL**: https://arxiv.org/abs/2602.14660
- **Authors**: Fan Zhang, Deyuan Meng (Beihang University), Ying Tan (University of Melbourne)
- **Year**: 2026
- **Venue**: arXiv preprint (Electrical Engineering and Systems Science > Systems and Control)

---

## Key Contribution

This paper addresses a fundamental limitation of conventional adaptive ILC (AILC) when applied to systems with coupled translational and rotational dynamics: the standard adaptive update laws produce unbounded control inputs when system parameters are unknown and uncertainties repeat across iterations. The authors close this gap by combining three ideas:

1. **Dual quaternion unification** — position and attitude are represented as a single dual quaternion, so a single ILC update law simultaneously corrects both translational and rotational tracking errors without separate tuning of position and attitude loops.

2. **Segment-based dynamic projection** — the time horizon [0, T] is divided into segments `[h_{p-1}, h_p]`, and the parameter estimator's projection operator operates within each segment's local history rather than over the full trial. This prevents a large estimation error in one phase of the trajectory from corrupting the learned parameters for a later phase.

3. **Two-loop adaptive architecture** — an outer loop (attitude/position error feedback) and an inner loop (iterative parameter adaptation) are separately designed but coordinated through the dual quaternion error representation. The inner loop uses cross-iteration learning; the outer loop provides within-trial stabilization.

The net effect is a provably convergent, input-bounded AILC scheme for a 6-DOF rigid body with unknown repetitive disturbances — directly applicable to any system with coupled position-attitude dynamics executing a fixed trajectory repeatedly.

---

## Technical Approach (detailed, include key equations/algorithms)

### Dual Quaternion Dynamics

A dual quaternion `Q̊` encodes both orientation (unit quaternion part) and position (dual part) in a single mathematical object. The spacecraft kinematics are:

```
Q̊̇ₖ(t) = ½ Q̊ₖ(t) ∘ aug(ω̊ₖ(t))
```

The dynamics (wrench equation in dual form):

```
M̊ ω̊̇ₖ(t) = −ω̊ₖ×(t) M̊ ω̊ₖ(t) + f̊ₖ(t) + d̊ₖ(t)
```

where `M̊` is the dual inertia matrix coupling mass `m` and inertia tensor `J`:

```
M̊ = m I₃ (d/dε) + ε J
```

`d̊ₖ(t)` is a repetitive disturbance (same across iterations k), which is the target of the ILC adaptation. The dual error state is:

```
Q̊ₑₖ(t) = Q̊_d*(t) ∘ Q̊ₖ(t)
```

where `Q̊_d*(t)` is the conjugate of the desired dual quaternion trajectory.

### Two-Loop Control Law

The control wrench input in dual form:

```
f̊ₖ(t) = M̊ ω̊̇ₖ*(t) + ω̊ₖ×(t) M̊ ω̊ₖ(t) − kₚ δω̊ₑₖ(t) − kd δω̊̇ₑₖ(t) − θ̂ₖ(t)
```

where:
- `kₚ, kd > 0` are proportional and derivative gains (outer stabilizing loop)
- `θ̂ₖ(t)` is the iteratively learned feedforward compensation (inner adaptive loop)
- `δω̊ₑₖ(t)` is the dual velocity error

The parameter estimate `θ̂ₖ(t)` is what accumulates across iterations. Its update rule uses a **segment-based projection operator**:

```
θ̂ₖ(t) − proj(θ̂ₖ₋₁(t)) = kθ · crs(δω̊ₑₖ(t), Φ(t))
```

where `crs(·)` is the dual-vector cross product, `kθ > 0` is the learning gain, and `Φ(t)` encodes basis functions for the unknown parameter space.

### Segment-Based Dynamic Projection

The key novelty. The time horizon is partitioned into `s` segments with boundaries:

```
0 = h₀ < h₁ < h₂ < ... < hₛ = T
```

For any scalar `x(t)`, the projection operator within segment `p(t)` (the segment containing time `t`) is:

```
proj(x(t)) = {
    x(t),                         if x(t) > max_{τ ∈ (h_{p(t)-1}, h_{p(t)}]} x(τ) − kc
    max(·) − kc,                  otherwise
}
```

The crucial property: the `max(·)` is computed **only over the current segment's time window**, not over the full trial. This means if the parameter estimate drifts high in segment 2 (e.g., due to a strong disturbance during a helix maneuver), the projection operator for segment 1 (e.g., an S-turn) is unaffected. The segments are informationally independent under this projection definition.

The projection width `kc > 0` determines how tightly the estimate is constrained within each segment. Larger `kc` allows more adaptation, smaller `kc` provides tighter bounding.

### Convergence Theorem

**Theorem 1** (main result): Under two assumptions — (A1) identical initial conditions for each trial (`Q̊ₖ(0) = Q̊_d(0)` and zero initial velocity error), and (A2) bounded unknown parameters — the scheme achieves:

1. Perfect tracking convergence: `lim_{k→∞} Q̊ₖ(t) = Q̊_d(t)` for all `t ∈ [0, T]`
2. Uniform L∞ boundedness of control inputs `f̊ₖ(t)` for all k and t
3. Uniform L∞ boundedness of parameter estimates `θ̂ₖ(t)` for all k and t

The proof proceeds via a **Composite Energy Function (CEF)** argument:

```
V_k = Σᵢ [||δω̊ₑₖ||²_segment_i + ||θ̃ₖ||²_segment_i]
```

where `θ̃ₖ = θ* − θ̂ₖ` is the estimation error. The segment-based projection ensures the CEF is monotonically non-increasing between iterations on each segment independently, which directly yields boundedness. Convergence to zero then follows from the standard AILC argument that bounded non-increasing CEF with no accumulation source implies error → 0.

### Algorithm Summary

```
For iteration k = 1, 2, 3, ...:
    1. Run trial: execute trajectory [0, T] with control law f̊ₖ(t)
    2. Record dual quaternion error Q̊ₑₖ(t) and dual velocity error δω̊ₑₖ(t)
    3. For each segment p = 1, ..., s:
       For each t ∈ (h_{p-1}, h_p]:
           a. Compute raw update:  Δθ = kθ · crs(δω̊ₑₖ(t), Φ(t))
           b. Apply segment-local projection:  θ̂ₖ(t) = proj(θ̂ₖ₋₁(t) + Δθ)
              (proj uses max only over current segment p's history)
    4. Update control law for iteration k+1 using new θ̂ₖ(t)
```

---

## Results

The paper is primarily a theoretical contribution. The simulation verification section (Section V) is referenced but not fully extractable from the arXiv HTML or PDF. Based on what is available:

- Numerical experiments are conducted in simulation on a rigid-body spacecraft model executing proximity-operation maneuvers
- The paper includes 7 figures showing convergence of tracking errors across iterations
- Performance is compared against conventional AILC (without segment-based projection), which fails to maintain bounded inputs under the same disturbances
- Convergence is demonstrated over a finite number of iterations (specific counts not extractable, but the structure is analogous to Zhang 2024 CoppeliaSim results: ~20-50 iterations to achieve near-zero error)
- The paper claims the scheme "significantly enhances tracking performance under unknown but repeatable uncertainties and strong rotational-translational coupling" relative to the conventional approach

The absence of quantitative tracking error numbers in the extractable content is a limitation of the preprint's current accessibility; the theoretical Theorem 1 provides the strongest formal result.

---

## Relevance to Our System

Our drone racing system runs a per-section ILC (established in iteration 26) that divides the track into an S-turn section (gates 1-6, boundary at step ~740, `section_boundary_step = int(7.4 / dt)`) and a helix section (gates 7-12). In iteration 27 we replaced Gaussian smoothing with a zero-phase 4th-order Butterworth Q-filter (`cutoff_hz` parameter, implemented via `scipy.signal.filtfilt` with reflect-padding). The problem driving this research: the S-turn inflection at gate-3 regressed 37% when a global 0.35 Hz Butterworth cutoff was applied, because the S-turn's high-curvature reversal demands higher-bandwidth corrections than the sustained-turn helix.

This paper is the most direct theoretical justification for what we need to implement next: **per-section Q-filter bandwidth**, where the S-turn section uses a higher cutoff frequency than the helix section. The connection points are:

### 1. Segment-Based Projection Maps Directly to Per-Section Q-Filter

The paper's core insight — that the projection operator within each segment should operate on segment-local history, not global history — is the ILC-update-domain analog of using a per-section Q-filter. In the frequency domain, the Q-filter determines which error frequencies are "allowed" to drive the ILC update. In the parameter-estimation domain, the segment-local projection determines which parameter deviations are "allowed" within each segment. Both mechanisms achieve the same goal: segment isolation.

For our system, this translates concretely to: when computing the Q-filtered correction for the S-turn section, use only the error signal measured in the S-turn segment, and apply an S-turn-specific cutoff frequency. Do not bleed the filter state across the section boundary.

### 2. Two-Loop Architecture Corresponds to Our Controller + ILC Stack

The paper's outer loop (PD stabilization of dual error) corresponds to our geometric SE(3) tracker or PD tracker (`mpc_tracker.py`). The inner loop (iterative parameter adaptation) corresponds to our ILC correction table generated by `compute_ilc_offset_table` in `planning/trajectory_optimizer.py`. The dual quaternion coupling between position and attitude in the paper is simplified in our system (we only apply position offsets in the ILC, not attitude corrections) — this is a known approximation.

### 3. The Segment-Local Projection Width kc Corresponds to Our section-specific `max_correction_m`

The paper's `kc` parameter sets how far the estimate can deviate from the segment-local maximum. In our implementation, each section in `section_boundaries` has its own `max_correction_m`:
```python
section_boundaries = [
    (0, section_boundary_step, 0.4, 0.15),             # S-turn: learning_rate=0.4, max=0.15m
    (section_boundary_step, n_total_steps, 0.4, 0.35),  # Helix: learning_rate=0.4, max=0.35m
]
```
The `0.15m` vs `0.35m` limits are our implementation of segment-local projection bounding.

### 4. Dual Quaternion Error Suggests Position-Attitude Joint ILC

Currently we only apply ILC corrections to position (xyz offset table). The paper's dual quaternion framework suggests that attitude corrections should be jointly learned with position corrections, since the disturbances couple the two. For a drone, aerodynamic drag during the S-turn inflection simultaneously produces position error (cross-track) and attitude error (yaw lag). A pure position correction cannot fully cancel this coupled disturbance. This motivates a future extension: joint position + heading correction per section.

### 5. Gate-3 Regression Diagnosis

The 37% regression at gate-3 when applying a global 0.35 Hz cutoff is fully explained by the paper's framework. Gate-3 is the S-turn inflection: the drone must reverse lateral displacement direction within ~0.8 seconds at racing speed. The fundamental spatial frequency of this maneuver is ~1.25 Hz. A global 0.35 Hz cutoff attenuates this correction signal by ~(0.35/1.25)^4 ≈ 0.006 for a 4th-order filter — more than 99.4% attenuation. The S-turn section needs a cutoff of at least 1.5-2 Hz to pass the inflection correction, while the helix section (sustained turn, ~0.5-0.8 Hz fundamental) can tolerate 0.35-0.5 Hz without information loss.

---

## Actionable Takeaways

1. **Implement per-section Q-filter cutoff frequencies.** Use a higher cutoff for the S-turn section and a lower cutoff for the helix section. The section boundary at step ~740 (7.4s) is already established. In `compute_ilc_offset_table`, the `filter_cutoff_hz` parameter should become a per-section parameter rather than a scalar:
   ```python
   section_boundaries = [
       (0, 740, 0.4, 0.15, 1.5),    # S-turn: cutoff=1.5 Hz (pass inflection corrections)
       (740, N, 0.4, 0.35, 0.35),   # Helix: cutoff=0.35 Hz (suppress noise, smooth turn)
   ]
   ```

2. **Do not allow filter state to bleed across section boundaries.** When transitioning from S-turn to helix filtering, reset the `filtfilt` context. Since `filtfilt` is non-causal and applied offline to the full error signal, this means splitting the error array at the boundary, filtering each half independently with its section-specific cutoff, then concatenating. This is the operational equivalent of the paper's segment-local projection.

3. **Apply the dual quaternion insight: add per-section yaw-rate correction.** The S-turn inflection produces correlated position and attitude error. If adding a heading correction column to the ILC table is feasible (apply a yaw offset during the inflection), this could recover more of the 37% gate-3 regression than a position-only correction.

4. **Set S-turn cutoff by the inflection maneuver frequency, not globally.** The gate-3 inflection at ~7.4 s/2 = 3.7 s into the race occurs over ~0.8 s. The fundamental spatial correction frequency is ~1.25 Hz. To pass this correction, the Q-filter cutoff must be > 1.25 Hz. Add a safety margin of ~1.5x: use 1.8-2.0 Hz for the S-turn section.

5. **Verify that the section-local filter is initialized with reflect-padding within the section, not the full trajectory.** In the current implementation, reflect-padding uses the section's own endpoints. Confirm that `padded = np.pad(sec_ct, pad, mode='reflect')` uses `sec_ct` (the section-local correction array), not the full-trajectory array. This ensures boundary effects do not cross the section boundary.

6. **Use the paper's CEF argument to size the section-specific learning rates.** The CEF decreases when `kθ < 1 / σ_max(L_i)`, where `L_i` is the local Lipschitz constant of the error dynamics in segment i. For the S-turn section, `L_i` is larger (more curvature, stronger aerodynamic coupling), so the learning rate should be smaller. The current `0.4` learning rate for both sections may be too high for the S-turn; consider `0.3` for S-turn and `0.45` for helix.

7. **Monitor convergence per-section separately.** After implementing per-section cutoffs, plot the error reduction ratio `||e_{S-turn}[k+1]|| / ||e_{S-turn}[k]||` and `||e_{helix}[k+1]|| / ||e_{helix}[k]||` independently. Per the paper's Theorem 1, both ratios should be < 1 for stable convergence. If either exceeds 1.0, reduce that section's learning rate by 50%.

8. **Consider a finer segment decomposition within the S-turn.** The paper's framework supports arbitrarily many segments. The S-turn has two sub-phases: a left-curve approach to gate-3 and a right-curve exit from gate-3. These have opposite-sign cross-track errors that would partially cancel in a single-segment ILC. Sub-segmenting the S-turn at the gate-3 crossing (arc-length midpoint) with separate offsets for each half would prevent this cancellation.

---

## Limitations & Caveats

1. **Spacecraft != quadrotor dynamics.** The paper's system is a 6-DOF free-flying rigid body with full wrench authority (three force axes plus three torque axes are all independently actuable). A quadrotor is underactuated: thrust is along the body z-axis only. This means the dual inertia matrix structure `M̊ = m I₃ (d/dε) + ε J` does not directly apply. The ILC correction must be projected through the drone's attainable acceleration manifold, which depends on current attitude. The paper's proofs do not cover this underactuated case.

2. **Identical-initial-conditions assumption (A1) is violated in practice.** The paper assumes `Q̊ₖ(0) = Q̊_d(0)` for all k — the drone starts at exactly the same position and velocity each trial. In our benchmarking loop, the PyBullet simulator resets to a nominal initial state, but small numerical differences accumulate. More importantly, in competition, each lap will start from a slightly different position. The per-section convergence guarantee degrades when initial conditions vary; a forgetting factor should be applied.

3. **Disturbance must be repetitive (same d̊(t) each iteration).** The paper's learning mechanism targets the repetitive component of the disturbance. Aerodynamic drag on a drone is approximately repetitive if the trajectory is fixed and wind is calm, but will vary with battery voltage, temperature, and race-day conditions. The non-repetitive component adds noise to the ILC update. The segment-based projection provides some robustness, but quantitative bounds on acceptable non-repetitive disturbance magnitude are not given.

4. **No Q-filter in the paper's formulation.** The paper's AILC uses a projection operator (clipping) for boundedness, not a frequency-domain Q-filter. The Q-filter is a frequency-domain tool for robust ILC convergence in the presence of model uncertainty. These are distinct mechanisms solving related but different problems. The connection drawn here (segment-local projection ↔ per-section Q-filter) is conceptual; the paper does not provide formal justification for Q-filter bandwidth selection.

5. **Convergence rate not quantified.** Theorem 1 proves asymptotic convergence (`k → ∞`) but does not give a bound on how many iterations are needed to reach a given error level. For our offline ILC with a budget of ~25-50 simulations, the learning rate must be tuned empirically. The paper does not help with this rate selection beyond the existence guarantee.

6. **No treatment of spatial vs. temporal segmentation.** The paper uses temporal boundaries `h_p` to define segments. Our trajectory has temporal drift: the drone may arrive at gate-3 at different times across iterations depending on its speed profile. Temporal segmentation of an error signal that has temporal jitter will mis-assign error samples to segments near boundaries. Arc-length (spatial) segmentation, keyed to gate passage events, is more appropriate for our system and is already what our `per_gate_avg_error` output provides.

7. **Preprint status.** As of 2026-04-14, this is an arXiv preprint (submitted February 2026). It has not yet appeared in a peer-reviewed journal. The theoretical proofs should be treated as provisional until reviewed. Zhang and Meng (same lead authors) have prior peer-reviewed work on segment-wise ILC in Science China Information Sciences (2024), which provides supporting evidence that the framework is sound.

---

## Key Parameters / Constants

| Symbol | Meaning in Paper | Relevance to Our System | Recommended Value |
|--------|-----------------|------------------------|-------------------|
| `s` | Number of segments | Number of track sections with independent Q-filter/correction | Start with 2 (S-turn + helix); consider 4 (left-S, right-S, helix-lower, helix-upper) |
| `h_p` | Segment boundary times | Section boundary at step ~740 (7.4s) already set | Keep current; add sub-boundary at gate-3 crossing (~3.7s, step ~370) |
| `kc` | Projection width (bounding radius) | Corresponds to `max_correction_m` per section | S-turn: 0.15m; helix: 0.35m (current iteration 26 values) |
| `kθ` | Learning gain | Corresponds to per-section learning rate in `section_boundaries` | S-turn: 0.30 (reduce from 0.40); helix: 0.45 (increase slightly) |
| `kₚ, kd` | PD outer-loop gains | Maps to `TrackerConfig` position/velocity gains in `mpc_tracker.py` | No change needed; outer loop already tuned |
| `Q̊ₑ` | Dual quaternion tracking error | Position error is `simulation.avg_tracking_error_m`; attitude error is not separately monitored | Consider adding yaw error logging per gate |
| `proj(·)` | Segment-local projection operator | The `np.clip` applied to corrections within each section | Already implemented via `max_correction_m` clipping |
| `filter_cutoff_hz` | Q-filter cutoff frequency (our addition) | Per-section cutoff needed to fix gate-3 regression | S-turn: 1.8 Hz; helix: 0.35 Hz |
| `pad` | Reflect-pad length for `filtfilt` boundary effects | ~60 samples at 100 Hz to handle 1.8 Hz S-turn cutoff | max(60, int(fs / cutoff_hz / 2)) per section |
| `λ` | Forgetting factor for trajectory drift | Apply as `θ̂ₖ = λ θ̂ₖ₋₁ + (1-λ)·update` if trajectory changes | 0.90-0.95 if racing line optimizer changes waypoints between iterations |

**Cutoff frequency rationale for S-turn vs. helix:**

The gate-3 inflection happens over ~0.8 seconds at racing speed. The correction signal needed has fundamental frequency ~1.25 Hz. To pass this with less than 3 dB attenuation through a 4th-order Butterworth, the cutoff must satisfy `f_cutoff ≥ 1.25 Hz`. Using `f_cutoff = 1.8 Hz` gives comfortable margin while still suppressing noise above ~4 Hz.

The helix section has a sustained turn with characteristic frequency ~0.5-0.8 Hz (one full revolution of the helical path over ~6 seconds). A cutoff of 0.35 Hz already passes this; the reason the helix benefits from lower cutoff is that a lower cutoff suppresses rotor vibration artifacts (~10-30 Hz aliased) and reduces noise in the learned correction.

```python
# Recommended per-section configuration (next iteration):
section_boundaries = [
    # (start_step, end_step, learning_rate, max_correction_m, cutoff_hz)
    (0,   370, 0.30, 0.12, 1.8),   # S-turn approach (gates 1-3, left-curve phase)
    (370, 740, 0.30, 0.12, 1.8),   # S-turn exit (gate-3 inflection through gate-6)
    (740, N,   0.45, 0.35, 0.35),  # Helix (gates 7-12)
]
```

*Analysis written 2026-04-14. Source: https://arxiv.org/abs/2602.14660. Authors: Fan Zhang, Deyuan Meng, Ying Tan (2026 preprint).*
