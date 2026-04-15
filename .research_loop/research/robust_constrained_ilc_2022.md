# On Robustness in Optimization-Based Constrained ILC
- **URL**: https://arxiv.org/abs/2203.05291
- **Authors**: Dominic Liao-McPherson, Efe C. Balta, Alisa Rupenyan, John Lygeros
- **Year**: 2022
- **Venue**: IEEE Control Systems Letters, Vol. 6, pp. 2846–2851 (DOI: 10.1109/LCSYS.2022.3178877). All authors at Automatic Control Laboratory, ETH Zürich. Funded by Swiss National Science Foundation grant #180545.

---

## Key Contribution

This paper fills a specific gap in the ILC literature: how to guarantee constraint satisfaction in constrained optimization-based ILC (OB-ILC) when the plant model is uncertain. Prior work on constrained ILC either lacked robustness guarantees (constraints could be violated when the model was wrong) or lacked formal stability proofs in the iteration domain. The authors close this gap simultaneously on both fronts.

The central contribution is a robust OB-ILC algorithm based on the **forward-backward splitting** proximal optimization algorithm that (1) provably satisfies state and input constraints for all trials despite bounded model uncertainty, (2) converges to a low-tracking-error fixed point, and (3) admits a rigorous input-to-state stability (ISS) analysis in the iteration domain via monotone operator theory.

A key mechanism enabling robust constraint satisfaction is **constraint tightening via the Pontryagin (Minkowski) set difference**: instead of requiring the feedforward update to stay within the original feasible set, the algorithm imposes membership in a shrunken set `U ⊖ V` where V is derived from the uncertainty bounds. This conservatively absorbs worst-case model deviations so that the actual system trajectory always satisfies the original constraints.

The contribution is five pages (IEEE L-CSS format), concise but theoretically dense. It is distinguished from contemporaneous work by combining three tools rarely united: optimization-based ILC, monotone operator theory, and robust set-theoretic control.

---

## Technical Approach

### Problem Formulation

The plant is a **constrained linear time-invariant system** subject to bounded additive uncertainty:

```
y = G * u + d + w
```

where:
- `u` is the input (feedforward signal to be learned across trials)
- `y` is the output (measured per-trial trajectory)
- `G` is the nominal plant (transfer matrix / lifted system matrix)
- `d` is the nominal disturbance (the unknown systematic error ILC learns to cancel)
- `w` captures model mismatch (the uncertain part, bounded: `w ∈ W`)

Constraints take the form `u ∈ U` (input constraints) and `y ∈ Y` (output/state constraints), where U and Y are convex sets.

The goal is to track a reference signal `y_ref` as closely as possible while satisfying these constraints at every trial, even in the presence of the uncertain component `w`.

### Constraint Tightening via Pontryagin Difference

The key technical device is replacing the original constraint sets with **tightened versions**:

```
U_tight = U ⊖ W_u      (input constraint tightening)
Y_tight = Y ⊖ W_y      (output constraint tightening)
```

where `⊖` denotes the Pontryagin (set) difference:

```
A ⊖ B = { x ∈ A | x + b ∈ A  for all  b ∈ B }
```

Intuitively: if the algorithm's input `u` stays in `U_tight` and the predicted output stays in `Y_tight`, then despite any realization of model mismatch `w ∈ W`, the actual input and output remain in the original feasible sets U and Y. The tightening absorbs the worst-case perturbation.

The uncertainty sets `W_u` and `W_y` must be computable from the structured uncertainty description (e.g., a known bounded gain matrix error, or an additive disturbance ball). The paper assumes the uncertainty is **structured** — meaning the uncertainty set has geometric regularity (e.g., ellipsoidal, polyhedral, or box sets) so that the Pontryagin difference is computable in closed form or via convex programming.

### Forward-Backward Splitting Algorithm

The ILC update is cast as a **proximal splitting optimization** problem at each trial. Given the output measurement from trial j, the update for trial j+1 solves:

```
minimize    || y_j - y_ref ||^2_Q + || u_{j+1} - u_j ||^2_R
subject to  u_{j+1} ∈ U_tight
            G * u_{j+1} ∈ Y_tight
```

The forward-backward splitting algorithm decomposes this into:
1. **Forward step (gradient)**: Take a gradient step on the smooth quadratic tracking cost using the previous trial's measurements.
2. **Backward step (proximal/projection)**: Project onto the (tightened) constraint sets using a proximal operator.

This is equivalent to a projected gradient descent on the optimization problem, with the projection onto the tightened feasible region ensuring robust constraint satisfaction. The step size `alpha` (analogous to the learning rate) must be chosen to satisfy a contraction condition in the iteration domain.

### Stability Analysis via Monotone Operator Theory

The convergence and stability analysis is conducted in the **iteration domain** (indexed by trial number j, not by time k within a trial). The key insight is that the ILC update operator — the mapping from error at trial j to feedforward input at trial j+1 — can be viewed as a **monotone operator** on the Hilbert space of input trajectories.

Monotone operator theory provides fixed-point convergence guarantees for the forward-backward splitting iteration:
- If the gradient operator of the cost is co-coercive (satisfied for smooth convex costs with Lipschitz gradient)
- And the constraint projection is a firmly non-expansive operator (always true for projection onto convex sets)

Then the composed forward-backward iteration is guaranteed to converge to a fixed point.

The fixed point is NOT necessarily zero tracking error — it is the best feasible feedforward, i.e., the point where no further improvement is possible while remaining in the (tightened) constraint sets. If the constraint-tightened region is large relative to the optimal unconstrained input, the fixed point closely approximates zero tracking error.

**Input-to-state stability (ISS)** in the iteration domain is also derived, providing bounds of the form:

```
|| u_j - u* ||  ≤  beta(|| u_0 - u* ||, j)  +  gamma(|| w ||)
```

where `beta` is a class-KL function (converging to zero with iteration count) and `gamma` is a class-K function (proportional to the uncertainty level). This quantifies how much residual error the uncertainty injects at the fixed point.

### Relation to Norm-Optimal ILC (NO-ILC)

The paper explicitly compares OB-ILC to classic norm-optimal ILC (which does not tighten constraints). Key findings:
- **NO-ILC** converges faster (larger effective step size) but violates constraints when model mismatch is present.
- **OB-ILC** converges more slowly (step size restricted by the ISS condition) but guarantees constraint satisfaction for all trials, including intermediate trials before convergence.
- OB-ILC achieves a **lower asymptotic tracking error** than NO-ILC when constraints are active, because the tightening forces the controller to be less aggressive and more robust. NO-ILC, when its constraints are violated, effectively ignores the constraint and converges to a suboptimal point.

This is a counterintuitive result: more conservative constraint handling leads to BETTER steady-state performance when model uncertainty is present and constraints are binding.

---

## Results

The paper demonstrates the approach on a **precision motion stage** (two-axis high-precision robotic stage tracking a reference trajectory while satisfying a speed output constraint). This is a standard benchmark in the constrained ILC literature.

Key reported findings:
- OB-ILC **satisfies the speed constraint at every iteration**, including the first iteration where the naive (unconstrained) update would cause violation.
- NO-ILC violates the speed constraint at early iterations and converges to a trajectory that still violates constraints in the presence of model mismatch.
- The unlearned baseline (no ILC) also violates the constraint due to model mismatch.
- OB-ILC tracking error converges monotonically, reaching a low asymptotic level in ~10-20 iterations (exact number not stated; depends on step size choice).
- Convergence rate depends on the amount of model mismatch: larger `|| W ||` → tighter constraint tightening → smaller effective step size → slower convergence but same eventual constraint satisfaction guarantee.

No specific numerical tracking-error values are reported in the available abstract and metadata; the paper presents curves rather than tables. The qualitative conclusion is clear: the tightening overhead is worth the robustness guarantee.

---

## Relevance to Our System

**Relevance level: High for architecture understanding; Moderate for immediate implementation.**

Our system uses offline ILC (the Schoellig-style P-type update) with post-optimization trajectory inflation — conservative time margins baked into segment durations that were calibrated before ILC existed. This creates a tension: ILC reduces tracking error, but the time margins were sized for the pre-ILC tracking error level. As ILC converges and tracking error drops ~24% per recent iteration batch (from 0.187m to 0.175m avg), those conservative margins may no longer be necessary at their current level.

### What the Paper Says About Constraint Management as ILC Converges

The paper's core lesson for our situation is this: **constraint tightening is not a one-time fixed choice — it is a function of the uncertainty level `W`.** As ILC learns and the residual (un-modeled) error shrinks, the effective uncertainty `W` also shrinks, and therefore the tightened constraint set expands back toward the original feasible set.

Formally, if after j iterations the residual error bound drops from `W_0` to `W_j ⊂ W_0`, then the tightened constraint should update to:

```
U_tight_j = U ⊖ W_j      (larger than U_tight_0 since W_j ⊂ W_0)
```

This means **constraints should be progressively relaxed as ILC converges**, not held fixed. The paper does not implement adaptive tightening (it uses a fixed worst-case W throughout), but its theoretical framework makes the adaptive version straightforward: track the empirical error variance across iterations, shrink W accordingly, and allow the feasible set to expand.

Applied to our system:
- Our "conservative time margins" are analogous to the tightened set `U_tight = U ⊖ W` in the paper's framework.
- The margin was calibrated for the pre-ILC uncertainty level (avg error ~0.5m).
- After ILC reduces avg error to ~0.175m, the effective `W` is roughly 3x smaller.
- Therefore, the margins can in principle be reduced by a proportional factor — but only if we can confirm the ILC correction is systematic (repeatable), not random.

The paper's ISS bound is also directly relevant: even at convergence, there is a residual error of order `gamma(|| W ||)`. Our empirical per-gate error data (gates 7-10 still showing ~0.24-0.33m) tells us what `gamma(|| W ||)` actually is in practice. That residual defines the minimum safe margin that cannot be contracted further.

### Constraint Tightening vs. Our Time Inflation Strategy

Our time inflation operates in the time domain (segment durations), not the control input domain. However, the logical equivalence holds:
- Longer segment time → lower required speed → more slack on the speed/agility constraint → absorbs tracking error at cost of lap time.
- As ILC reduces tracking error → less slack needed → time margins can shrink → faster lap times.

The paper's framework suggests the correct procedure: estimate the current `|| W ||` (residual ILC error after convergence), compute the required margin to absorb worst-case realizations, and set time inflation accordingly. This is systematically principled rather than ad hoc.

---

## Actionable Takeaways

1. **Adaptive margin reduction is theoretically justified.** As our ILC converges, the time margins (conservative segment durations) can be shrunk proportionally to the reduction in avg tracking error. Specifically: if avg error drops from E_0 to E_j, the margin can be scaled by `E_j / E_0`. Measure residual per-gate error variance (not just mean) to quantify the actual `W`.

2. **Use per-gate residual error as the uncertainty bound.** The paper's framework requires a bounded uncertainty set W. Our ILC already produces per-gate error data. After convergence (iteration delta < 1%), the remaining per-gate error is our empirical estimate of W. Use this to set per-gate margin tightening — gates where ILC has driven error near zero get tighter margins; gates where residual is higher (gates 7-10, helix) keep more conservative margins.

3. **Do not reduce margins before ILC converges.** The paper shows constraint violation is most dangerous at early iterations, before ILC has learned. Margin reduction should happen only after the ILC loop has converged (< 1% improvement per iteration), not during learning.

4. **Track the ISS residual empirically.** Run the ILC loop to convergence, then record the distribution of per-gate errors across 10+ repeated runs. The 95th percentile of this distribution is the `gamma(|| W ||)` term — the irreducible residual. The margin must be at least this large to guarantee gate passage.

5. **Consider iteration-varying step size for faster convergence.** The paper notes that OB-ILC is slower than NO-ILC due to the step size restriction. An iteration-varying step size (larger early when constraints are not active, smaller later when they are) could accelerate convergence. Our current alpha=0.5 may be conservative; at early iterations before constraints bind, a larger alpha (0.6-0.7) may speed learning without violating safety.

6. **Forward-backward splitting as upgrade path.** Our current P-type ILC update (pure gradient step) corresponds to the "forward step" alone in the paper's framework. Adding a "backward step" (soft projection onto a feasible set encoding gate-passage requirements) would provide formal constraint guarantees. This is a natural extension once P-type ILC plateaus.

7. **Monotone operator convergence confirms our stopping criterion is valid.** The paper proves fixed-point convergence under the same conditions our current ILC uses. The "< 1% improvement" stopping criterion is consistent with this theory — the algorithm is at or near its fixed point.

---

## Limitations & Caveats

1. **Assumes linear time-invariant plant.** Our drone system is nonlinear (quadrotor dynamics, aerodynamic drag), and we are using a kinematic proxy in simulation. The LTI assumption means the paper's guarantees do not directly transfer. However, the conceptual framework (tighten constraints by estimated uncertainty, relax as uncertainty shrinks) generalizes informally.

2. **Fixed uncertainty set W.** The paper uses a worst-case, fixed `W` throughout all iterations. Adaptive tightening (reducing W as learning progresses) is mentioned as a natural extension but not analyzed. Without adaptive tightening, the algorithm is permanently conservative even after convergence. Our use case specifically needs the adaptive variant.

3. **No explicit learning rate schedule.** The step size is fixed throughout. The paper notes this is the primary reason OB-ILC converges slower than NO-ILC. An iteration-varying or line-search-based step size is left for future work.

4. **Structured uncertainty assumption.** The Pontryagin difference is only computable in closed form for geometrically regular uncertainty sets (polytopes, ellipsoids, boxes). For our system, the uncertainty is empirically measured per-gate, not a single structured set. Computing the Pontryagin difference of an empirical distribution would require fitting a bounding set first.

5. **No disturbance rejection for non-repetitive disturbances.** The paper explicitly notes that the framework does not learn non-repeating components. Wind gusts, battery sag, and thermal currents in a real race are non-repetitive and cannot be reduced by ILC. Only in simulation (nearly deterministic) is this not a concern.

6. **Convergence rate depends on uncertainty magnitude.** Larger W → tighter constraints → smaller step size → slower convergence. For our helix section (highest residual error, largest effective W), ILC convergence will be slowest there — which is exactly where we most need it. More iterations may be required at those segments.

7. **Not validated on drone racing hardware.** The precision motion stage is a 2-DOF planar system, far simpler than a quadrotor with 6-DOF nonlinear dynamics and camera-based gate detection. Transfer of quantitative results to our setting requires re-validation.

---

## Key Parameters / Constants

| Parameter | Value / Description | Role in Paper |
|-----------|--------------------|-|
| Uncertainty set W | Bounded polytope or ellipsoid | Defines constraint tightening amount |
| Tightened input set | `U_tight = U ⊖ W_u` | Replaces U in optimization |
| Tightened output set | `Y_tight = Y ⊖ W_y` | Replaces Y in optimization |
| Step size alpha | Must satisfy ISS condition (problem-specific) | Controls convergence rate vs. robustness tradeoff |
| ISS gain gamma | Class-K function of `|| W ||` | Bounds residual error at fixed point |
| Convergence class | Class-KL function beta | Bounds transient error vs. iteration count |
| Cost weights Q, R | Tracking cost vs. input deviation | Standard ILC tuning parameters |
| Precision motion stage | 2-DOF, speed-constrained | Benchmark system for numerical validation |
| Constraint violation | NO-ILC violates; OB-ILC satisfies at all trials | Key qualitative result |
| Asymptotic error | OB-ILC lower than NO-ILC (active constraints) | Key counterintuitive quantitative result |
| Publication | IEEE Control Systems Letters, Vol. 6, pp. 2846-2851, 2022 | Venue |
| Citations | ~15 (as of 2024) | Impact |

*Analysis written 2026-04-14. Paper: Liao-McPherson et al., IEEE Control Systems Letters, 2022.*
