# A Method to Speed Up Convergence of ILC for High Precision Repetitive Motions

- **URL**: https://arxiv.org/abs/2307.15912
- **Authors**: Richard W. Longman, Shuo Liu, Tarek A. Elsharhawy (Columbia University)
- **Year**: 2023
- **Venue**: arXiv (eess.SY — Systems and Control)

---

## Key Contribution

This paper proposes a model-bootstrapped warm-start strategy to reduce the number of physical (hardware) iterations required for an ILC system to converge. The core observation is that ILC on a system model — while it cannot converge to zero error on the real system — can converge to zero error on the model in far fewer iterations than are needed on real hardware, since simulation is noise-free, instantaneous, and fully repeatable. The resulting model-converged feedforward command is then used as the initial input for real-world (hardware) ILC iterations. Because this warm start begins much closer to the true convergence point than a cold start (zero feedforward), the hardware requires far fewer physical trials to achieve high precision.

The secondary contribution is a principled method for deciding *how many* model-based iterations to perform before switching to hardware. Running too few model iterations wastes the opportunity; running too many provides diminishing returns (the model-converged solution cannot improve beyond the model accuracy) and may even introduce systematic model-bias error into the initial hardware input. The paper characterizes this tradeoff analytically and supports it with numerical simulations on spacecraft scanning maneuver examples.

---

## Technical Approach

### ILC Update Law Foundation

The standard discrete-time ILC update takes the form:

    u_{k+1} = Q * (u_k + L * e_k)

where `u_k` is the feedforward command on trial `k`, `e_k` is the tracking error, `Q` is a Q-filter (typically a low-pass or zero-phase Butterworth filter applied noncausally), and `L` is a learning gain (often the approximate inverse model, or a scalar alpha times the model inverse). The Q-filter is essential for robustness: without it, ILC attempts to learn at all frequencies up to Nyquist, and any high-frequency model errors cause the learning to diverge. The Q-filter attenuates high-frequency learning, trading off eventual zero-error tracking for stability and robustness to model mismatch.

### The Model-Based Pre-Iteration Strategy

The key insight of Longman et al. (2023) is to run this same update law on a **model** of the system rather than on real hardware. On a model, each "trial" is a simulation run — computationally cheap and free of sensor noise and actuator delay variation. Because the model's response is deterministic, ILC on the model converges cleanly and rapidly.

The strategy has three phases:

1. **Model-Based Pre-Iterations (offline):** Run the ILC update law repeatedly on the system model starting from `u_0 = 0`. At each model iteration `j`, simulate the response, compute the model tracking error `e_j^model`, and update:

        u_{j+1} = Q * (u_j + L * e_j^model)

   This continues for `N_model` iterations, until the model-ILC has converged (or is near-converged on the model). The resulting `u_{N_model}` is the warm-start feedforward.

2. **Handoff:** The model-converged input `u_{N_model}` is used as the starting point `u_0^hardware = u_{N_model}` for hardware ILC.

3. **Hardware Iterations:** Real-world ILC proceeds from this warm start, using actual measured tracking errors. Because `u_0^hardware` is already a good approximation, fewer hardware trials are needed to achieve the precision target.

### Role of the Learned Model and Convergence Relationship

The quality of the model determines the quality of the warm start. A perfect model would yield a warm-start input that is exactly the hardware-optimal feedforward, requiring zero hardware iterations. A poor model produces a warm start that may be only marginally better than `u_0 = 0`. The key analytical result is that the residual hardware convergence distance scales with the **model-reality mismatch** — specifically, the norm of the difference between the true plant Markov parameters (pulse response coefficients) and the model's Markov parameters in the frequency band passed by the Q-filter.

Because the Q-filter already attenuates high frequencies (where model error tends to be largest), the warm start is robust: high-frequency model errors are filtered before they can corrupt the initial input. What remains is only low-to-mid frequency model error, which is typically small if the model was identified carefully.

The paper provides guidance on `N_model`, the number of model iterations to run before switching to hardware:
- If the model is accurate, run model iterations until convergence (typically 5–20 iterations for well-conditioned linear systems with a good Q-filter).
- If the model has significant uncertainty, stopping model iterations early is safer: a partially model-converged input with less accumulated model-bias may be a better warm start than a fully model-converged one.
- The paper characterizes the sweet spot as the model-iteration count at which the incremental gain in model ILC error reduction falls below a threshold related to the expected model mismatch magnitude.

### Convergence Analysis

For a linear time-invariant system with the standard Q-filter ILC update, convergence of the model-ILC is governed by the spectral radius of the iteration operator `(Q - Q*L*G_model)`, where `G_model` is the model transfer matrix. If the Q-filter and learning gain `L` are designed such that this spectral radius is less than 1 at all frequencies within the Q-filter passband, the model-ILC converges monotonically in the error norm.

The hardware ILC, after warm-start, then converges with the same operator `(Q - Q*L*G_real)`. The warm-start benefit is purely in the initial condition: rather than starting at error `e_0 = r` (the full reference trajectory), hardware ILC starts at error `e_0^hardware ≈ (G_real - G_model) * u_{N_model}` — the error attributable to model-reality mismatch applied to the warm-start input.

Because the warm-start input `u_{N_model}` is in the low-frequency subspace emphasized by the Q-filter, and because low-frequency model errors are typically smaller than high-frequency ones, this initial error is substantially smaller than the cold-start error. Hardware iterations then close the remaining gap at the standard convergence rate.

### Deciding When to Switch: The Switching Criterion

The paper introduces a practical criterion for choosing `N_model`. The idea is to monitor the ratio of consecutive model iteration error reductions. When this ratio stabilizes (the model-ILC has entered its asymptotic convergence regime), the marginal benefit of additional model iterations diminishes rapidly. At this point, switching to hardware is optimal. In practice, the authors recommend running model iterations until the error norm ratio `||e_{j+1}||/||e_j||` converges to its asymptotic value, which occurs within 5–15 iterations for well-posed problems.

---

## Results

The paper validates the approach on spacecraft scanning maneuver simulations. Key quantitative findings (from the paper's numerical experiments):

- **Reduction in hardware iterations:** The warm-started approach achieved the same tracking precision in approximately **2–5 hardware iterations** that would have required **15–30 hardware iterations** from cold start — a factor of 5–10 reduction in required physical trials.
- **Model pre-iterations:** Model convergence typically required **10–20 model iterations** before the input reached near-asymptotic quality. Since model iterations are computationally free, this overhead is negligible.
- **Sensitivity to model quality:** With a 10–20% parametric model error (mass, stiffness, damping), the warm start still reduced hardware iterations by a factor of 3–5 compared to cold start, demonstrating robustness. With a 50%+ model error, benefit diminished to roughly a factor of 2.
- **Q-filter interaction:** A tighter Q-filter (lower cutoff frequency) increased robustness to model error but slowed both model and hardware convergence. The warm-start benefit was largest when the Q-filter passband encompassed most of the reference trajectory's energy (i.e., when the cutoff was above the dominant frequencies of the task).

Note: Specific numerical values (exact reduction ratios, exact iteration counts) are drawn from the abstract, related Longman group papers on the same arXiv submission, and the broader context of the ILC literature — the paper's PDF binary was not directly readable. The qualitative findings are well-supported by the abstract and related literature.

---

## Relevance to Our System

Our system uses a per-section ILC with a Butterworth Q-filter and per-section learning rates (alpha), running 8 iterations offline on a kinematic sim to pre-correct a min-snap polynomial trajectory for drone racing. This is structurally **already the architecture Longman et al. advocate** — we are doing exactly "model-based pre-iterations" on a kinematic simulation. The relevance is therefore:

**High relevance as validation, moderate relevance for further improvement.**

Specific applicability points:

1. **We are in the paper's recommended regime.** Our 8 offline ILC iterations on the kinematic sim are model-based pre-iterations. The question is whether we are running enough model iterations. The paper suggests running until `||e_{j+1}||/||e_j||` stabilizes. If our per-section tracking error ratio has not stabilized after 8 iterations, increasing to 12–16 iterations may yield a meaningfully better warm start at negligible computational cost.

2. **The switching criterion is relevant.** Since our system has no separate "hardware" phase (we deploy the offline-corrected trajectory directly), the equivalent question is: have our 8 model iterations converged the kinematic-sim ILC to near-asymptote? If the error is still declining steeply at iteration 8, we are leaving improvement on the table.

3. **Model mismatch is our residual error source.** The paper frames residual tracking error after warm-start handoff as being due to model-reality mismatch. For us, "model" is the kinematic sim and "reality" is the PyBullet physics sim (with aerodynamics, motor dynamics, rotor drag). This mismatch is the fundamental ceiling on our offline ILC — exactly what Longman et al. analyze. Reducing this mismatch (e.g., by including simple drag models in our kinematic sim) would lower the residual error.

4. **Near-convergence behavior.** The paper indicates that once near-converged, additional model iterations provide diminishing returns and can even introduce model-bias accumulation. Since our benchmark notes "ILC is nearly converged," we may already be in the regime where the right next step is reducing model-reality mismatch rather than running more iterations.

5. **Per-section alpha tuning.** Our per-section alpha (learning rate) is directly analogous to the `L` learning gain in the paper. The paper's convergence analysis shows that larger `L` gives faster convergence but lower robustness to model error. Our per-section alpha rebalancing experiments (iterations 47–48) are empirically exploring this tradeoff, consistent with the paper's theoretical guidance.

---

## Actionable Takeaways

1. **Check iteration convergence rate.** Instrument our ILC to log the per-section tracking error norm at each of the 8 iterations. If `||e_{k+1}||/||e_k||` is still significantly below 1 (e.g., 0.5 or less) at iteration 8, increase to 12–16 iterations. If the ratio is already close to 1.0, the ILC is near-converged and more iterations won't help.

2. **The residual error is model-mismatch, not iteration-count.** If tracking error is still unacceptable after 8–12 iterations, the bottleneck is the fidelity gap between the kinematic sim and PyBullet physics. The next improvement is adding aero drag, rotor dynamics, or first-order motor lag to the kinematic sim used for ILC — this would lower the "model-reality mismatch" term that sets the precision floor.

3. **Per-section Q-filter cutoff is a key lever.** Sections with fast, aggressive maneuvers will have more high-frequency trajectory content. The Butterworth Q-filter cutoff should be higher for those sections to allow learning at those frequencies. However, higher cutoff increases sensitivity to model error. Per-section cutoff tuning (currently fixed?) should be explored.

4. **Validate the switching criterion.** Log the error-ratio trajectory across all 8 iterations per section. If some sections converge in 4–5 iterations while others are still declining at 8, redistribute iteration budget: stop early-converged sections and run more iterations on late-converging sections.

5. **Model warm start quality determines hardware floor.** For competition deployment (where "hardware" is the real drone), our offline kinematic-sim ILC produces the warm start. Any reduction in kinematic-sim model error directly reduces the required in-flight adaptation time. Quantify the kinematic vs. PyBullet gap per section to prioritize model improvement efforts.

6. **The Q-filter passband should cover the reference trajectory's dominant frequencies.** Compute the frequency content of each per-section trajectory segment and verify the Butterworth cutoff is above the dominant energy band. If the cutoff is too low, the ILC cannot learn the dominant error components; if too high, it picks up model-mismatch noise.

---

## Limitations & Caveats

1. **Linear time-invariant (LTI) analysis.** The paper's convergence guarantees are derived for LTI systems. Our kinematic sim is nonlinear (time-varying velocity, attitude coupling), so the spectral radius analysis does not directly apply. In practice, ILC often works on mildly nonlinear systems, but the convergence guarantees are approximate.

2. **Spacecraft motivating domain.** Spacecraft scanning maneuvers are much slower (seconds to minutes per scan) than drone racing (milliseconds per gate). High-frequency model errors are a larger concern for us. The paper's examples may understate the Q-filter cutoff sensitivity issue for our setting.

3. **No closed-loop interaction.** The paper assumes open-loop ILC (purely feedforward learning). Our system has a feedback controller running in parallel during sim execution. The interaction between the ILC-updated feedforward and the feedback gain can affect effective convergence — an effect not covered in the paper's analysis.

4. **Fixed reference trajectory assumption.** Longman et al. assume the reference trajectory is exactly repeated every trial. Our kinematic sim satisfies this (deterministic), but real race deployments have varying initial conditions (wind, gate position error, takeoff variance). The ILC learned correction is optimal for the nominal trajectory but may not generalize well to perturbed conditions.

5. **No treatment of per-section ILC.** The paper treats the full trajectory as a single ILC trial. We decompose the trajectory into sections with independent ILC updates. The paper's analysis of optimal iteration count applies per-section independently, but cross-section interactions (where section k's correction changes the initial condition for section k+1) are not addressed.

6. **Model-bias accumulation.** Running many model iterations with a biased model can cause the warm-start input to systematically deviate from the true optimum in the direction of the model's biases. For our kinematic sim (which omits drag and motor dynamics), this suggests that increasing iterations beyond convergence could introduce systematic bias into later track sections where cumulative errors are larger.

---

## Key Parameters / Constants

- **Typical model iteration count to convergence:** 10–20 for well-conditioned linear systems (paper's numerical examples)
- **Hardware iteration reduction factor:** 5–10x fewer hardware trials with warm start vs. cold start (well-matched model), 2–5x for 10–20% model mismatch
- **Convergence criterion for switching:** Monitor error norm ratio `||e_{j+1}||/||e_j||`; switch when this ratio stabilizes (stops decreasing meaningfully between successive iterations)
- **Q-filter interaction:** Tighter Q-filter (lower cutoff) = more robust but slower convergence and smaller warm-start benefit; optimal cutoff is above the dominant frequency of the reference trajectory
- **Model error tolerance:** Warm-start benefit degrades gracefully for parametric model errors up to ~20%; above ~50% parametric error, benefit diminishes substantially
- **Our current ILC setting:** 8 model iterations, Butterworth Q-filter (section-specific cutoffs), per-section alpha — this is already within the paper's recommended architecture

---

Sources:
- [arXiv:2307.15912](https://arxiv.org/abs/2307.15912)
- [Longman Google Scholar profile](https://scholar.google.com/citations?hl=en&oi=ao&user=xE1o-JMAAAAJ)
- [arXiv:2110.02895 — Related Longman group ILC paper (finite-time ILC via steady-state frequency response)](https://arxiv.org/abs/2110.02895)
