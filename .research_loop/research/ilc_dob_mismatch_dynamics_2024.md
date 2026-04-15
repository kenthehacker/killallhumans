# ILC with DOB for Mismatched Dynamics
- **URL**: https://arxiv.org/abs/2404.10231
- **Authors**: Harsh Modi, Zhu Chen, Xiao Liang, Minghui Zheng (University at Buffalo, SUNY)
- **Year**: 2024 (published IEEE Robotics and Automation Letters, April 2024; DOI: 10.1109/LRA.2024.3391026)

## Key Contribution

The paper presents what the authors claim is "the first attempt to implement learning with DOB for systems with mismatched dynamics." The core insight is that Iterative Learning Control (ILC) and Disturbance Observer (DOB) are individually insufficient for robust disturbance rejection in challenging scenarios — ILC converges slowly or fails when disturbances change, while DOB provides only reactive single-iteration compensation. The unified framework allows a sequence of dynamically different systems (e.g., different UAVs with different masses and controller gains) to pass learned disturbance estimates forward to the next system, improving rejection speed and accuracy across iterations.

The practical novelty is the explicit design of cross-system learning filters L1 and L2 that account for dynamic differences between the "donor" system (which generated the last error and disturbance estimate) and the "recipient" system (which will execute the next run). Without these filters, naively transferring a learning signal computed for one system's dynamics to another system with different gain or mass would amplify error rather than reduce it. The framework provides the correct mathematical transformation — essentially a model-mismatch compensation layer on top of the ILC update — along with a formal convergence guarantee expressed as a bound on acceptable modeling uncertainty.

## Technical Approach

The framework models each system j using three transfer functions relating the reference input, external disturbance, and learning signal to the output:

- **Gr,j**: reference → output, depends on plant Pj, controller Cj, filter Qj
- **Gd,j**: disturbance → output
- **Gf,j**: learning signal → output

A DOB is embedded in the closed loop. The DOB estimate of the disturbance feeding into the next ILC update is the central object of the learning procedure. The combined learning signal for iteration j is:

```
df,j = L1,j{ej-1} + L2,j{df,j-1}
```

where ej-1 is the tracking error from the previous system and df,j-1 is the previous learning signal. The two filters are designed as:

```
L1,j = (Ĝf,j Ĝd,j-1)^-1 Ĝd,j
L2,j = (Ĝf,j Ĝd,j-1)^-1 Ĝd,j Ĝf,j-1
```

where hat-symbols denote identified (estimated) models. The key design insight is that L1 converts the previous system's tracking error into an equivalent disturbance-domain correction for the current system by dividing out the previous system's disturbance-to-output dynamics and multiplying in the current system's. L2 does the same for the propagated learning signal. This is a frequency-domain filter design problem: the filters are non-causal in general (due to model inverse Ĝf,j^-1), so in practice stable approximate inverses are used.

The tracking error under this scheme satisfies:

```
ej = Te1,j{ej'} + Te2,j{df,j-1}
```

where ej' is the error without learning (baseline DOB-only), and Te1,j, Te2,j are transfer operators that depend on the learning filter choices. With the optimal filter choice above, the convergence condition is:

```
‖Δj-1‖ < ‖(1 + Pj-1 Cj-1) / 2‖
```

where Δj represents the modeling uncertainty of system j. Intuitively, better-designed baseline controllers (with higher loop gain ‖1 + PC‖) tolerate larger model mismatch before the learning scheme destabilizes. For the case of zero modeling uncertainty (perfect identified models), both Te1,j and Te2,j collapse to zero, giving ej → 0 in one iteration — the ideal ILC convergence rate. In practice, experiments converge by iteration 2-3 rather than 1.

Regarding the mismatch specifically: the cross-system gain variation across the three tested UAVs spans proportional gains 1.2 to 3.0 (a 2.5× ratio). The filters L1 and L2 explicitly compensate for this by inverting the previous system's effective dynamics and applying the current system's dynamics. The method works as long as the uncertainty bound is satisfied — it does not require the systems to be identical, only that their models are "well identified."

The DOB design follows standard practice: a low-pass filter Q(s) shapes the disturbance bandwidth, with the requirement that Q(jω) ≈ 1 for ω below the control bandwidth (so disturbances within bandwidth are fully rejected) and Q(jω) → 0 at high frequency (to avoid amplifying sensor noise). The paper uses first- or second-order Butterworth filters for Q.

## Results

Three physical UAVs were tested: an F450 (0.921 kg, proportional gain 3.0), an S500 (1.001 kg, proportional gain 1.2), and a Tarot 650 (1.234 kg, proportional gain 1.9). Four scenarios were validated:

1. Stationary hover under sinusoidal disturbance (0.9425 rad/s, 1 m/s²)
2. Circular trajectory (1m radius) under the same disturbance
3. Circular trajectory under impulse disturbance (2 m/s² half-sine)
4. Diamond-shaped trajectory under rectified sinusoidal disturbance (1.4138 rad/s, 1 m/s²)

Quantitative RMSE results (from Figure 8 and supplementary data):

- **Without DOB baseline**: highest error across all scenarios and UAVs
- **DOB-only**: marginal improvement, estimates are delayed and scaled incorrectly for impulse and rectified disturbances
- **Learning iteration 1**: significant error reduction across all scenarios; for scenarios 1-2, disturbance estimates become "almost perfect"
- **Learning iteration 2-3**: errors become "almost negligible" — approximately 80-95% RMSE reduction relative to DOB-only

For scenarios 3-4 (more challenging disturbance shapes), the DOB-only estimates were scaled down by roughly 50% (scenario 3) or highly inaccurate (scenario 4 rectified case). After learning iteration 1-2, the DOB estimates recovered to near-perfect reconstruction of the actual disturbance.

Simulations (where models are exact) show single-iteration convergence (Te1,j = Te2,j = 0 identically). Experiments converge in 2-3 iterations due to residual modeling uncertainty. The cyclic ordering (UAV1 → UAV2 → UAV3 → UAV1...) allows continuous refinement.

## Relevance to Our System

Our ILC implementation in `planning/trajectory_optimizer.py` uses an inner kinematic simulation with kp_xy=6.0, kd_xy=4.0, ff_accel=0.4 to pre-compute correction offsets, while the benchmark runs the geometric tracker with kp_xy=7.0, kd_xy=5.5, feedforward_accel=0.50 (`TrackerConfig` in `control/mpc_tracker.py`). This is exactly the "mismatched dynamics" scenario addressed by Modi et al. The inner sim is "system j-1" and the benchmark is "system j" — they share the same structural form (PD + feedforward) but differ in all three gain parameters.

Iteration 39 attempted to sync these gains but failed, suggesting the mismatch is load-bearing in some way (possibly the inner sim's lower gains produce smoother, more conservative ILC corrections that degrade when executed by the more aggressive benchmark controller). This is consistent with the paper's finding that naive transfer of learning signals between mismatched systems can amplify errors — the L1/L2 filters exist precisely to prevent this.

The paper's convergence bound `‖Δ‖ < ‖(1 + PC)/2‖` quantifies how much mismatch is tolerable. With our kp mismatch of 6 vs 7 (a ~17% difference in one gain), and kd mismatch of 4 vs 5.5 (a ~38% difference), the modeling uncertainty Δ is non-trivial. Whether it falls within the convergence bound depends on the effective loop gain at the trajectory frequencies (roughly 1-3 Hz for our racing trajectories). The paper's UAVs tolerated 2.5× gain ratios, suggesting our ~1.4× mismatch should in principle be handleable with proper filter design.

Critically, the paper does not simply suggest "make the gains match." Instead, it provides the mathematical transformation to correctly map corrections computed under one system's dynamics to another. For our use case, this means computing the ILC correction in the inner sim (kp=6) and then applying L1-scaled corrections to the benchmark (kp=7) rather than applying the raw offset directly. The scaling is approximately Ĝf,benchmark/Ĝf,inner_sim evaluated at the relevant frequencies — in the PD case this reduces to a gain ratio that is knowable analytically.

The 80-95% RMSE improvement reported by the paper (in 2-3 iterations) corresponds well to the kind of improvement we are seeking: our current avg error of ~0.14m and we are targeting <0.1m. If the DOB component can reject per-lap disturbances that the ILC hasn't fully converged on, this could close the remaining gap.

## Actionable Takeaways

1. **Implement a mismatch-correcting scale factor on ILC updates.** Instead of applying the raw position offset from the inner sim (kp=6) to the benchmark (kp=7), scale the offset by the gain ratio. A first approximation: the effective position response scales with kp, so the benchmark tracker over-corrects by a factor of 7/6 ≈ 1.167 relative to what the inner sim predicted. Scaling the ILC offset down by this factor before applying to the reference trajectory may reduce the over-correction seen in prior iterations.

2. **Separately scale velocity offsets by the kd ratio.** The velocity offset currently uses the same scale as position. But the velocity response scales with kd, so the benchmark's kd=5.5 relative to inner kd=4.0 gives a ratio of 1.375. The `vel_scale` parameter in the ILC code should account for this separately from the position scale.

3. **Add a feedforward mismatch correction.** The ff_accel mismatch (0.4 inner vs. 0.50 benchmark) means the benchmark applies 25% more feedforward acceleration than the inner sim predicted. This systematic over-actuation is independent of position error and appears as a consistent phase advance in the trajectory tracking. A simple fix: compute what acceleration the inner sim "intended" and what the benchmark "actually applies," and pre-correct the reference trajectory acceleration profile.

4. **Replace the current single-gain Q-filter with a frequency-domain mismatch-aware filter.** The paper's L1 filter design `(Ĝf,j Ĝd,j-1)^-1 Ĝd,j` in the frequency domain for our PD systems reduces to a ratio of transfer functions that is computable analytically. For the x/y axes with drag=0.5: Ĝf(s) ≈ (kp + s*kd) / (s^2 + s*(kd+drag) + kp). Evaluate the ratio at 1-3 Hz to determine the correct ILC learning gain.

5. **Run ILC for 3 iterations rather than the current default.** The paper shows that even with significant mismatch, 2-3 iterations are sufficient to converge. If ILC is currently running fewer iterations, extending it (with proper convergence checking) should help.

6. **Consider a DOB residual layer in the tracker.** The paper's DOB component provides per-timestep disturbance rejection orthogonal to the ILC feedforward. Adding a simple DOB to `mpc_tracker.py` that estimates and rejects persistent force disturbances could complement the ILC offsets, especially for sections with high systematic error (gate-2 is already identified as problematic).

7. **When the inner sim gains cannot match the benchmark, compute the analytical correction factor.** From the paper's framework, the correction scale for the position axis is `(kp_benchmark + j*omega*kd_benchmark) / (kp_inner + j*omega*kd_inner)` evaluated at the dominant trajectory frequency. For our system at ~2 Hz (omega ≈ 12.6 rad/s): numerator = 7 + 12.6j*5.5 = 7+69.3j, denominator = 6 + 12.6j*4.0 = 6+50.4j. The magnitude ratio is |7+69.3j|/|6+50.4j| ≈ 69.65/50.75 ≈ 1.37. This implies the ILC position offsets should be attenuated by a factor of ~0.73 before being applied to the benchmark reference.

## Limitations & Caveats

**LTI assumption.** The framework is built on linear time-invariant system approximations. Our geometric tracker is nonlinear (SE(3) Lee controller), and the kinematic inner sim is also nonlinear (clamped acceleration). The LTI analysis is valid only near the operating point. At high tilt angles or aggressive maneuvers, the linearization error grows and the convergence bounds may not hold.

**Slowly time-varying reference.** The paper assumes slowly time-varying or repetitive trajectories. Our racing trajectory is time-optimal and includes aggressive acceleration phases, tight gate transitions, and a helix. The ILC framework assumes the disturbance is "iteration-repetitive," which holds if the trajectory is the same every lap. This is satisfied in our benchmark (same trajectory every run), but aggressive maneuver sections may violate the slowly-varying assumption used to derive the filter designs.

**Identical trajectories across systems.** The paper requires that UAV j executes exactly the same desired trajectory as UAV j-1. In our setting, the "trajectory" is fixed, but the closed-loop response differs between the inner sim and the benchmark due to gain differences — so the actual executed trajectory differs, not just the dynamics. This is a subtler form of mismatch than the paper addresses, and the learning filters may need additional correction terms.

**Plant inverse stability.** The L1/L2 filter design requires inverting Ĝf,j, the learning-signal-to-output transfer function. For our PD controller, this involves a term (s^2 + s*kd + kp)^-1 which is stable (both poles in LHP for positive kp, kd). So stable inversion is feasible, unlike non-minimum-phase systems.

**Gain variation range.** The paper validated 2.5× proportional gain ratios between UAVs. Our mismatch is ~1.17× in kp and ~1.38× in kd. Both are within the validated range, suggesting the method should work. However, the combined mismatch across kp, kd, and ff simultaneously was not explicitly tested — each isolated UAV differed mainly in mass and one gain.

**No hardware validation at racing speeds.** The paper's UAVs performed circular and diamond trajectories at modest speeds (1m radius circle). Our racing track involves speeds of 5-10 m/s with tight gate tolerances (~0.3m). The disturbance rejection bandwidth required at these speeds is higher, and the Q-filter cutoff must be tuned accordingly.

## Key Parameters / Constants

- **Proportional gains tested**: 1.2, 1.9, 3.0 (ratio up to 2.5×) — our mismatch of 6 vs 7 is within this range
- **UAV masses**: 0.921 kg, 1.001 kg, 1.234 kg (ratio ~1.34×)
- **Disturbance frequencies tested**: 0.9425 rad/s (~0.15 Hz), 1.4138 rad/s (~0.225 Hz) — much lower than our trajectory bandwidth; ILC may need higher-bandwidth filters for racing
- **Convergence iterations**: 1 iteration (simulations, perfect models), 2-3 iterations (experiments, real mismatch)
- **Error reduction**: ~80-95% RMSE reduction by iteration 2-3 relative to DOB-only baseline
- **Disturbance estimation accuracy**: ~50% underestimation in DOB-only for impulse disturbances; corrected to near-perfect after 1-2 learning iterations
- **Q-filter design**: Butterworth low-pass, order 1-2, cutoff below controller bandwidth. For our 3-5 Hz controller bandwidth, a 2-3 Hz cutoff is appropriate (consistent with the Bristow & Alleyne 2007 recommendation already implemented in our ILC Q-filter)
- **Uncertainty bound for convergence**: `‖Δ‖ < ‖(1 + PC)/2‖`. For our kp=7 benchmark, the DC loop gain is approximately 1 + kp/s^2 → ∞ at DC, but at 2 Hz it is approximately |1 + 7/(j2π*2)^2| ≈ |1 - 0.044| ≈ 0.956 + contributions from kd. The effective loop gain ‖1+PC‖ at trajectory frequencies needs to be evaluated numerically to verify the bound holds for our specific gain mismatch.
- **Mismatch-correcting scale for position ILC at 2 Hz**: approximately 0.73 (reciprocal of 1.37 magnitude ratio computed above) — this is the factor to apply to our ILC position offsets before injecting them into the benchmark reference trajectory.
